#pragma once

// The PACKED layout's device kernels: one block per pack, staging the pack's nodes in
// shared memory, with the ghost rows reduced afterwards by a second kernel.
//
// The device kernels, in this layout's own cuda/ subfolder: CUDA sources belong beside the host
// sources they mirror, and DESIGN.md asks each mesh format for one. They were all in
// cuda/cvfem_hex8_ns_cuda.cu, a single 2646-line translation unit holding every layout's kernels
// and every layout's launchers together.
//
// What stayed behind is the launchers -- the `launch_*` host functions that pick a block size, an
// instantiation and a stream. That is the same split as the host side: the kernel computes, the
// launcher decides.
//
// Included from inside cvfem_hex8_ns_cuda.cu's anonymous namespace, so these keep the internal
// linkage they had. One translation unit includes them; a __global__ in an anonymous namespace
// reached from two would be two different kernels with one name.


// ---------------------------------------------------------------- residual kernel

// GEOM selects the geometry model: CVFEM_CUDA_GEOM_AFFINE reads one precomputed
// adjugate and determinant per element, CVFEM_CUDA_GEOM_ISOPARAM evaluates the
// trilinear Jacobian at each of the 12 sub-control-surface points from the element's
// node coordinates. Isoparametric therefore needs the coordinates staged per pack --
// the same three arrays Rhie-Chow already stages, so when both are on they share one
// buffer instead of holding two copies.
template <int FLUSH, bool WITH_RC, int GEOM>
__global__ void cvfem_hex8_residual_pack_kernel(
        const ptrdiff_t nelements, const ptrdiff_t n_elements_per_pack,
        const double rho, const double mu,
        const uint16_t  *const __restrict__ elems,
        const ptrdiff_t *const __restrict__ owned_nodes_ptr,
        const ptrdiff_t *const __restrict__ n_shared,
        const ptrdiff_t *const __restrict__ ghost_ptr,
        const int32_t   *const __restrict__ ghost_idx,
        const double    *const __restrict__ adj,
        const double    *const __restrict__ det,
        const double    *const __restrict__ u,
        double *const __restrict__ r,
        double *const __restrict__ ghost_buf,
        const double    *const __restrict__ px,
        const double    *const __restrict__ py,
        const double    *const __restrict__ pz,
        const double    *const __restrict__ pgx,
        const double    *const __restrict__ pgy,
        const double    *const __restrict__ pgz,
        const double rc_scale) {
    extern __shared__ double smem[];

    const ptrdiff_t p            = blockIdx.x;
    const ptrdiff_t owned        = owned_nodes_ptr[p];
    const ptrdiff_t n_contiguous = owned_nodes_ptr[p + 1] - owned;
    const ptrdiff_t gbegin       = ghost_ptr[p];
    const ptrdiff_t n_ghost      = ghost_ptr[p + 1] - gbegin;
    const ptrdiff_t total_nodes  = n_contiguous + n_ghost;

    constexpr bool NEED_XYZ = (GEOM == CVFEM_CUDA_GEOM_ISOPARAM) || WITH_RC;

    double *const s_u   = smem;
    double *const s_out = smem + (ptrdiff_t)CVFEM_CUDA_NF * total_nodes;
    // Coordinates, staged when the geometry is isoparametric or Rhie-Chow needs them.
    double *const s_xyz = smem + 2 * (ptrdiff_t)CVFEM_CUDA_NF * total_nodes;
    // The nodal pressure gradient sits after them, so the coordinate block has the same
    // address in both cases and one gather serves both consumers.
    double *const s_pg  = s_xyz + (NEED_XYZ ? 3 * total_nodes : 0);

    // Stage the pack's fields, and zero the accumulator. Owned ids map to a contiguous
    // global window; ghosts resolve through ghost_idx (PACKED_FORMAT.md section 2).
    for (ptrdiff_t i = threadIdx.x; i < total_nodes; i += blockDim.x) {
        const ptrdiff_t g = (i < n_contiguous) ? (owned + i)
                                               : (ptrdiff_t)ghost_idx[gbegin + i - n_contiguous];
        const double *const src = &u[g * CVFEM_CUDA_NF];
        double *const       dst = &s_u[i * CVFEM_CUDA_NF];
        double *const       acc = &s_out[i * CVFEM_CUDA_NF];
#pragma unroll
        for (int f = 0; f < CVFEM_CUDA_NF; ++f) { dst[f] = src[f]; acc[f] = 0.0; }
        if (NEED_XYZ) {
            double *const q = &s_xyz[i * 3];
            q[0] = px[g]; q[1] = py[g]; q[2] = pz[g];
        }
        if (WITH_RC) {
            double *const q = &s_pg[i * 3];
            q[0] = pgx[g]; q[1] = pgy[g]; q[2] = pgz[g];
        }
    }
    __syncthreads();

    const ptrdiff_t e_start = p * n_elements_per_pack;
    const ptrdiff_t e_end   = min(nelements, (p + 1) * n_elements_per_pack);

    for (ptrdiff_t e = e_start + threadIdx.x; e < e_end; e += blockDim.x) {
        uint16_t ev[CVFEM_HEX8_N_NODES];
        double   ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES];
        double   uz[CVFEM_HEX8_N_NODES], pe[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const uint16_t l = elems[(ptrdiff_t)a * nelements + e];
            ev[a]                    = l;
            const double *const node = &s_u[(ptrdiff_t)l * CVFEM_CUDA_NF];
            ux[a] = node[0]; uy[a] = node[1]; uz[a] = node[2]; pe[a] = node[3];
        }

        // The kernels want element-local arrays of 8, so gather from shared.
        double ex[8], ey[8], ez[8];
        if (NEED_XYZ) {
#pragma unroll
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const double *const q = &s_xyz[(ptrdiff_t)ev[a] * 3];
                ex[a] = q[0]; ey[a] = q[1]; ez[a] = q[2];
            }
        }

        double adj_e[9];
        if (GEOM == CVFEM_CUDA_GEOM_AFFINE) {
#pragma unroll
            for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];
        }

        double re[CVFEM_HEX8_N_DOF];
        if (WITH_RC) {
            double rgx[8], rgy[8], rgz[8];
#pragma unroll
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const double *const q = &s_pg[(ptrdiff_t)ev[a] * 3];
                rgx[a] = q[0]; rgy[a] = q[1]; rgz[a] = q[2];
            }
            Hex8RhieChowT<double> rc;
            rc.x = ex; rc.y = ey; rc.z = ez;
            rc.pgx = rgx; rc.pgy = rgy; rc.pgz = rgz; rc.scale = rc_scale;
            if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM)
                cvfem_hex8_ns_upwind_residual_isoparam(rho, mu, ex, ey, ez, ux, uy, uz, pe, re, rc);
            else
                cvfem_hex8_ns_upwind_residual_sumfact(rho, mu, adj_e, det[e], ux, uy, uz, pe, re, rc);
        } else if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM) {
            cvfem_hex8_ns_upwind_residual_isoparam(rho, mu, ex, ey, ez, ux, uy, uz, pe, re);
        } else {
            cvfem_hex8_ns_upwind_residual(rho, mu, adj_e, det[e], ux, uy, uz, pe, re);
        }

        // Elements within a pack share nodes, so this accumulation needs atomics even
        // though the pack's buffer is block-private.
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            double *const acc = &s_out[(ptrdiff_t)ev[a] * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) atomicAdd(&acc[f], re[a * 4 + f]);
        }
    }
    __syncthreads();

    if (FLUSH == CVFEM_CUDA_FLUSH_TWO_PASS) {
        // Owned nodes have exactly one writing pack in this mode, so a plain store is
        // race-free; ghosts go to their own slot and are gathered afterwards.
        for (ptrdiff_t i = threadIdx.x; i < n_contiguous; i += blockDim.x) {
            const double *const acc = &s_out[i * CVFEM_CUDA_NF];
            double *const       dst = &r[(owned + i) * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) dst[f] = acc[f];
        }
        for (ptrdiff_t i = threadIdx.x; i < n_ghost; i += blockDim.x) {
            const double *const acc = &s_out[(n_contiguous + i) * CVFEM_CUDA_NF];
            double *const       dst = &ghost_buf[(gbegin + i) * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) dst[f] = acc[f];
        }
    } else {
        // One pass. The owned prefix below n_not_shared is touched by no other pack
        // (PACKED_FORMAT.md section 3), so it needs no atomics; everything above it can
        // race with another pack's ghost flush.
        const ptrdiff_t n_not_shared = n_contiguous - n_shared[p];
        for (ptrdiff_t i = threadIdx.x; i < n_not_shared; i += blockDim.x) {
            const double *const acc = &s_out[i * CVFEM_CUDA_NF];
            double *const       dst = &r[(owned + i) * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) dst[f] += acc[f];
        }
        for (ptrdiff_t i = n_not_shared + threadIdx.x; i < n_contiguous; i += blockDim.x) {
            const double *const acc = &s_out[i * CVFEM_CUDA_NF];
            double *const       dst = &r[(owned + i) * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) atomicAdd(&dst[f], acc[f]);
        }
        for (ptrdiff_t i = threadIdx.x; i < n_ghost; i += blockDim.x) {
            const double *const acc = &s_out[(n_contiguous + i) * CVFEM_CUDA_NF];
            double *const       dst = &r[(ptrdiff_t)ghost_idx[gbegin + i] * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) atomicAdd(&dst[f], acc[f]);
        }
    }
}

// y = J(u) v. Structurally the residual kernel with a third staged array; the flush is
// identical, so the two share the same ghost-reduce pass.
template <int FLUSH, int GEOM>
__global__ void cvfem_hex8_jacobian_action_pack_kernel(
        const ptrdiff_t nelements, const ptrdiff_t n_elements_per_pack,
        const double rho, const double mu,
        const uint16_t  *const __restrict__ elems,
        const ptrdiff_t *const __restrict__ owned_nodes_ptr,
        const ptrdiff_t *const __restrict__ n_shared,
        const ptrdiff_t *const __restrict__ ghost_ptr,
        const int32_t   *const __restrict__ ghost_idx,
        const double    *const __restrict__ adj,
        const double    *const __restrict__ det,
        const double    *const __restrict__ u,
        const double    *const __restrict__ vin,
        double *const __restrict__ r,
        double *const __restrict__ ghost_buf,
        const double    *const __restrict__ px,
        const double    *const __restrict__ py,
        const double    *const __restrict__ pz) {
    extern __shared__ double smem[];

    const ptrdiff_t p            = blockIdx.x;
    const ptrdiff_t owned        = owned_nodes_ptr[p];
    const ptrdiff_t n_contiguous = owned_nodes_ptr[p + 1] - owned;
    const ptrdiff_t gbegin       = ghost_ptr[p];
    const ptrdiff_t n_ghost      = ghost_ptr[p + 1] - gbegin;
    const ptrdiff_t total_nodes  = n_contiguous + n_ghost;
    const ptrdiff_t stride       = (ptrdiff_t)CVFEM_CUDA_NF * total_nodes;

    constexpr bool NEED_XYZ = (GEOM == CVFEM_CUDA_GEOM_ISOPARAM);

    double *const s_u   = smem;
    double *const s_v   = smem + stride;
    double *const s_out = smem + 2 * stride;
    double *const s_xyz = smem + 3 * stride;

    for (ptrdiff_t i = threadIdx.x; i < total_nodes; i += blockDim.x) {
        const ptrdiff_t g = (i < n_contiguous) ? (owned + i)
                                               : (ptrdiff_t)ghost_idx[gbegin + i - n_contiguous];
        const double *const su = &u[g * CVFEM_CUDA_NF];
        const double *const sv = &vin[g * CVFEM_CUDA_NF];
#pragma unroll
        for (int f = 0; f < CVFEM_CUDA_NF; ++f) {
            s_u[i * CVFEM_CUDA_NF + f]   = su[f];
            s_v[i * CVFEM_CUDA_NF + f]   = sv[f];
            s_out[i * CVFEM_CUDA_NF + f] = 0.0;
        }
        if (NEED_XYZ) {
            double *const c = &s_xyz[i * 3];
            c[0] = px[g]; c[1] = py[g]; c[2] = pz[g];
        }
    }
    __syncthreads();

    const ptrdiff_t e_start = p * n_elements_per_pack;
    const ptrdiff_t e_end   = min(nelements, (p + 1) * n_elements_per_pack);

    for (ptrdiff_t e = e_start + threadIdx.x; e < e_end; e += blockDim.x) {
        uint16_t ev[CVFEM_HEX8_N_NODES];
        double ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES], uz[CVFEM_HEX8_N_NODES];
        double vx[CVFEM_HEX8_N_NODES], vy[CVFEM_HEX8_N_NODES], vz[CVFEM_HEX8_N_NODES];
        double pe[CVFEM_HEX8_N_NODES], q[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const uint16_t l = elems[(ptrdiff_t)a * nelements + e];
            ev[a] = l;
            const double *const nu = &s_u[(ptrdiff_t)l * CVFEM_CUDA_NF];
            const double *const nv = &s_v[(ptrdiff_t)l * CVFEM_CUDA_NF];
            ux[a] = nu[0]; uy[a] = nu[1]; uz[a] = nu[2]; pe[a] = nu[3];
            vx[a] = nv[0]; vy[a] = nv[1]; vz[a] = nv[2]; q[a]  = nv[3];
        }
        double re[CVFEM_HEX8_N_DOF];
        if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM) {
            double ex[8], ey[8], ez[8];
#pragma unroll
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const double *const c = &s_xyz[(ptrdiff_t)ev[a] * 3];
                ex[a] = c[0]; ey[a] = c[1]; ez[a] = c[2];
            }
            cvfem_hex8_ns_upwind_jacobian_action_isoparam(rho, mu, ex, ey, ez, ux, uy, uz,
                                                          vx, vy, vz, q, re);
        } else {
            double adj_e[9];
#pragma unroll
            for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];
            // LIM is a compile-time template parameter with no default, and the device path carries no
            // limiter: 0 is the unlimited branch, which is what this call got when the limiter was a
            // runtime argument defaulting to zero. Without the explicit argument the call does not
            // compile at all, and nothing noticed because CUDA is not built on the development machine.
            cvfem_hex8_ns_upwind_jacobian_action</*LIM=*/0>(rho, mu, adj_e, det[e], ux, uy, uz,
                                                 vx, vy, vz, q, re);
        }
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            double *const acc = &s_out[(ptrdiff_t)ev[a] * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) atomicAdd(&acc[f], re[a * 4 + f]);
        }
        (void)pe;
    }
    __syncthreads();

    if (FLUSH == CVFEM_CUDA_FLUSH_TWO_PASS) {
        for (ptrdiff_t i = threadIdx.x; i < n_contiguous; i += blockDim.x)
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f)
                r[(owned + i) * CVFEM_CUDA_NF + f] = s_out[i * CVFEM_CUDA_NF + f];
        for (ptrdiff_t i = threadIdx.x; i < n_ghost; i += blockDim.x)
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f)
                ghost_buf[(gbegin + i) * CVFEM_CUDA_NF + f] =
                        s_out[(n_contiguous + i) * CVFEM_CUDA_NF + f];
    } else {
        const ptrdiff_t n_not_shared = n_contiguous - n_shared[p];
        for (ptrdiff_t i = threadIdx.x; i < n_not_shared; i += blockDim.x)
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f)
                r[(owned + i) * CVFEM_CUDA_NF + f] += s_out[i * CVFEM_CUDA_NF + f];
        for (ptrdiff_t i = n_not_shared + threadIdx.x; i < n_contiguous; i += blockDim.x)
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f)
                atomicAdd(&r[(owned + i) * CVFEM_CUDA_NF + f], s_out[i * CVFEM_CUDA_NF + f]);
        for (ptrdiff_t i = threadIdx.x; i < n_ghost; i += blockDim.x)
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f)
                atomicAdd(&r[(ptrdiff_t)ghost_idx[gbegin + i] * CVFEM_CUDA_NF + f],
                          s_out[(n_contiguous + i) * CVFEM_CUDA_NF + f]);
    }
}

// Gather the staged ghost contributions. Each destination appears in exactly one row,
// so this is race-free and bit-deterministic.
__global__ void cvfem_hex8_ghost_reduce_kernel(
        const ptrdiff_t n_rows,
        const ptrdiff_t *const __restrict__ reduce_ptr,
        const ptrdiff_t *const __restrict__ reduce_idx,
        const int32_t   *const __restrict__ reduce_dest,
        const double    *const __restrict__ ghost_buf,
        double *const __restrict__ r) {
    for (ptrdiff_t row = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; row < n_rows;
         row += (ptrdiff_t)blockDim.x * gridDim.x) {
        double acc[CVFEM_CUDA_NF] = {0.0, 0.0, 0.0, 0.0};
        const ptrdiff_t b = reduce_ptr[row], e = reduce_ptr[row + 1];
        for (ptrdiff_t j = b; j < e; ++j) {
            const double *const g = &ghost_buf[reduce_idx[j] * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) acc[f] += g[f];
        }
        double *const dst = &r[(ptrdiff_t)reduce_dest[row] * CVFEM_CUDA_NF];
#pragma unroll
        for (int f = 0; f < CVFEM_CUDA_NF; ++f) dst[f] += acc[f];
    }
}

// --------------------------------------------------- packed-mesh assembly
//
// Assembly on the packed mesh, for comparison against the element-parallel form that
// uses global ids. A pack-local BSR cannot be staged -- one element alone produces 64
// blocks x 16 doubles = 8 KiB and a pack holds thousands of blocks -- so what is staged
// is the *read* side: the pack's fields go into shared memory once and the element loop
// gathers them through the packed mesh's uint16 local ids. The write side is unchanged,
// straight into the global BSR through element_slots with atomicAdd.
//
// That makes the comparison a clean one. Both forms write identically, so the difference
// measures exactly what the packed mesh addresses: how the fields are read.
template <int VARIANT, int GEOM>
__global__ void cvfem_hex8_assemble_packed_kernel(
        const ptrdiff_t nelements, const ptrdiff_t n_elements_per_pack,
        const double rho, const double mu,
        const uint16_t  *const __restrict__ elems,
        const ptrdiff_t *const __restrict__ owned_nodes_ptr,
        const ptrdiff_t *const __restrict__ ghost_ptr,
        const int32_t   *const __restrict__ ghost_idx,
        const int32_t   *const __restrict__ slots,
        const double    *const __restrict__ adj,
        const double    *const __restrict__ det,
        const double    *const __restrict__ u,
        double *const __restrict__ values,
        const double    *const __restrict__ px,
        const double    *const __restrict__ py,
        const double    *const __restrict__ pz) {
    extern __shared__ double smem[];
    constexpr bool ISO = (GEOM == CVFEM_CUDA_GEOM_ISOPARAM);

    const ptrdiff_t p            = blockIdx.x;
    const ptrdiff_t owned        = owned_nodes_ptr[p];
    const ptrdiff_t n_contiguous = owned_nodes_ptr[p + 1] - owned;
    const ptrdiff_t gbegin       = ghost_ptr[p];
    const ptrdiff_t n_ghost      = ghost_ptr[p + 1] - gbegin;
    const ptrdiff_t total_nodes  = n_contiguous + n_ghost;

    double *const s_u   = smem;
    double *const s_xyz = smem + (ptrdiff_t)CVFEM_CUDA_NF * total_nodes;

    for (ptrdiff_t i = threadIdx.x; i < total_nodes; i += blockDim.x) {
        const ptrdiff_t g = (i < n_contiguous) ? (owned + i)
                                               : (ptrdiff_t)ghost_idx[gbegin + i - n_contiguous];
        const double *const src = &u[g * CVFEM_CUDA_NF];
        double *const       dst = &s_u[i * CVFEM_CUDA_NF];
#pragma unroll
        for (int f = 0; f < CVFEM_CUDA_NF; ++f) dst[f] = src[f];
        if (ISO) {
            double *const c = &s_xyz[i * 3];
            c[0] = px[g]; c[1] = py[g]; c[2] = pz[g];
        }
    }
    __syncthreads();

    const ptrdiff_t e_start = p * n_elements_per_pack;
    const ptrdiff_t e_end   = min(nelements, (p + 1) * n_elements_per_pack);

    for (ptrdiff_t e = e_start + threadIdx.x; e < e_end; e += blockDim.x) {
        double ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES];
        double uz[CVFEM_HEX8_N_NODES], pe[CVFEM_HEX8_N_NODES];
        double ex[CVFEM_HEX8_N_NODES], ey[CVFEM_HEX8_N_NODES], ez[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const uint16_t l  = elems[(ptrdiff_t)a * nelements + e];
            const double *const nd = &s_u[(ptrdiff_t)l * CVFEM_CUDA_NF];
            ux[a] = nd[0]; uy[a] = nd[1]; uz[a] = nd[2]; pe[a] = nd[3];
            if (ISO) {
                const double *const c = &s_xyz[(ptrdiff_t)l * 3];
                ex[a] = c[0]; ey[a] = c[1]; ez[a] = c[2];
            }
        }
        const int32_t *const es = &slots[e * 64];
        if constexpr (ISO) {
            cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<true>(
                    rho, mu, ex, ey, ez, ux, uy, uz, es, values);
        } else {
            double adj_e[9];
#pragma unroll
            for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];
            if constexpr (VARIANT == CVFEM_CUDA_JAC_HANDWRITTEN)
                cvfem_hex8_ns_upwind_jacobian_add_slots<true>(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
            else
                cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        }
        (void)pe;
    }
}

// ------------------------------------------------ bit-reproducible residual
//
// Neither existing flush mode is reproducible run to run. The two-pass mode removes the
// atomics from the global reduction, but both modes still accumulate a pack's elements
// with atomicAdd into shared memory, and an atomic fixes no summation order. Getting a
// reproducible answer needs the accumulation itself ordered, which means turning the
// per-element scatter into a per-node gather.
//
// Phase 1 computes each element's 32 residual values and stores them. No accumulation,
// so nothing to race. Phase 2 gives one thread per node and walks that node's elements
// in increasing element order through the CSR, so every sum happens in the same order on
// every run and at every block size.
//
// The cost is the scratch: 32 doubles per element written and read back, which is why
// this is an additional mode and not a replacement for the fast ones.
template <int GEOM>
__global__ void cvfem_hex8_residual_store_kernel(
        const ptrdiff_t nelements, const double rho, const double mu,
        const int32_t *const __restrict__ elements,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        double *const __restrict__ elem_r,
        const double  *const __restrict__ px,
        const double  *const __restrict__ py,
        const double  *const __restrict__ pz) {
    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < nelements;
         e += (ptrdiff_t)blockDim.x * gridDim.x) {
        double ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES];
        double uz[CVFEM_HEX8_N_NODES], pe[CVFEM_HEX8_N_NODES];
        double ex[CVFEM_HEX8_N_NODES], ey[CVFEM_HEX8_N_NODES], ez[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const int32_t g = elements[(ptrdiff_t)a * nelements + e];
            const double *const nd = &u[(ptrdiff_t)g * CVFEM_CUDA_NF];
            ux[a] = nd[0]; uy[a] = nd[1]; uz[a] = nd[2]; pe[a] = nd[3];
            if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM) { ex[a] = px[g]; ey[a] = py[g]; ez[a] = pz[g]; }
        }
        double re[CVFEM_HEX8_N_DOF];
        if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM) {
            cvfem_hex8_ns_upwind_residual_isoparam(rho, mu, ex, ey, ez, ux, uy, uz, pe, re);
        } else {
            double adj_e[9];
#pragma unroll
            for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];
            cvfem_hex8_ns_upwind_residual(rho, mu, adj_e, det[e], ux, uy, uz, pe, re);
        }
#pragma unroll
        for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) elem_r[e * CVFEM_HEX8_N_DOF + i] = re[i];
    }
}
