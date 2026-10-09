#pragma once

// The STANDARD (atomic) layout's device kernels: one thread per element writing the global
// arrays with atomicAdd, plus the boundary closure and the block diagonal.
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

enum BoundaryOp { BOUNDARY_RESIDUAL = 0, BOUNDARY_JV = 1, BOUNDARY_ASSEMBLE = 2 };


__global__ void cvfem_hex8_residual_gather_kernel(
        const ptrdiff_t nnodes,
        const ptrdiff_t *const __restrict__ n2e_ptr,
        const int32_t   *const __restrict__ n2e_enc,
        const double    *const __restrict__ elem_r,
        double *const __restrict__ r) {
    for (ptrdiff_t n = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; n < nnodes;
         n += (ptrdiff_t)blockDim.x * gridDim.x) {
        double acc[CVFEM_CUDA_NF] = {0.0, 0.0, 0.0, 0.0};
        // n2e_enc is built in increasing element order, so this sum is order-fixed.
        for (ptrdiff_t k = n2e_ptr[n]; k < n2e_ptr[n + 1]; ++k) {
            const int32_t enc = n2e_enc[k];
            const double *const src = &elem_r[(ptrdiff_t)(enc >> 3) * CVFEM_HEX8_N_DOF
                                              + (enc & 7) * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) acc[f] += src[f];
        }
        double *const dst = &r[n * CVFEM_CUDA_NF];
#pragma unroll
        for (int f = 0; f < CVFEM_CUDA_NF; ++f) dst[f] = acc[f];
    }
}

// ------------------------------------------- standard-mesh matrix-free baseline
//
// The same residual and J*v computed WITHOUT the packed mesh: one thread per element,
// grid-stride, global node ids, accumulating straight into the global vector with
// atomicAdd. No packs, no shared memory, no ghost machinery -- this is what the
// operators look like on an ordinary element->node connectivity, and it is the
// baseline the block-per-pack kernels have to beat.
//
// The CPU has had this comparison all along, because its `atomic` layout is exactly
// this and its `packed` layout is the pack-based one. The device had only the packed
// form, so what the format was worth here had never been measured.
template <int GEOM>
__global__ void cvfem_hex8_residual_global_kernel(
        const ptrdiff_t nelements, const double rho, const double mu,
        const int32_t *const __restrict__ elements,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        double *const __restrict__ r,
        const double  *const __restrict__ px,
        const double  *const __restrict__ py,
        const double  *const __restrict__ pz) {
    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < nelements;
         e += (ptrdiff_t)blockDim.x * gridDim.x) {
        int32_t gid[CVFEM_HEX8_N_NODES];
        double  ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES];
        double  uz[CVFEM_HEX8_N_NODES], pe[CVFEM_HEX8_N_NODES];
        double  ex[CVFEM_HEX8_N_NODES], ey[CVFEM_HEX8_N_NODES], ez[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const int32_t g = elements[(ptrdiff_t)a * nelements + e];
            gid[a]          = g;
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
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            double *const dst = &r[(ptrdiff_t)gid[a] * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) atomicAdd(&dst[f], re[a * 4 + f]);
        }
    }
}

template <int GEOM>
__global__ void cvfem_hex8_jacobian_action_global_kernel(
        const ptrdiff_t nelements, const double rho, const double mu,
        const int32_t *const __restrict__ elements,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        const double  *const __restrict__ vin,
        double *const __restrict__ r,
        const double  *const __restrict__ px,
        const double  *const __restrict__ py,
        const double  *const __restrict__ pz) {
    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < nelements;
         e += (ptrdiff_t)blockDim.x * gridDim.x) {
        int32_t gid[CVFEM_HEX8_N_NODES];
        double  ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES], uz[CVFEM_HEX8_N_NODES];
        double  vx[CVFEM_HEX8_N_NODES], vy[CVFEM_HEX8_N_NODES], vz[CVFEM_HEX8_N_NODES];
        double  q[CVFEM_HEX8_N_NODES];
        double  ex[CVFEM_HEX8_N_NODES], ey[CVFEM_HEX8_N_NODES], ez[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const int32_t g = elements[(ptrdiff_t)a * nelements + e];
            gid[a]          = g;
            const double *const nu = &u[(ptrdiff_t)g * CVFEM_CUDA_NF];
            const double *const nv = &vin[(ptrdiff_t)g * CVFEM_CUDA_NF];
            ux[a] = nu[0]; uy[a] = nu[1]; uz[a] = nu[2];
            vx[a] = nv[0]; vy[a] = nv[1]; vz[a] = nv[2]; q[a] = nv[3];
            if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM) { ex[a] = px[g]; ey[a] = py[g]; ez[a] = pz[g]; }
        }

        double re[CVFEM_HEX8_N_DOF];
        if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM) {
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
            double *const dst = &r[(ptrdiff_t)gid[a] * CVFEM_CUDA_NF];
#pragma unroll
            for (int f = 0; f < CVFEM_CUDA_NF; ++f) atomicAdd(&dst[f], re[a * 4 + f]);
        }
    }
}

// ---------------------------------------------------------------- assembly

// Element-parallel, grid-stride, writing straight into the global BSR with atomicAdd.
// Both kernel families already accumulate through CVFEM_ATOMIC_ADD, which expands to
// atomicAdd under __CUDA_ARCH__, so the same source serves host and device.
template <int VARIANT, int GEOM = CVFEM_CUDA_GEOM_AFFINE, int PART = CVFEM_HEX8_PART_ALL>
__global__ void cvfem_hex8_assemble_bsr_kernel(
        const ptrdiff_t nelements, const double rho, const double mu,
        const int32_t *const __restrict__ elements,
        const int32_t *const __restrict__ slots,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        double *const __restrict__ values,
        const double  *const __restrict__ px = nullptr,
        const double  *const __restrict__ py = nullptr,
        const double  *const __restrict__ pz = nullptr) {
    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < nelements;
         e += (ptrdiff_t)blockDim.x * gridDim.x) {
        double ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES];
        double uz[CVFEM_HEX8_N_NODES], pe[CVFEM_HEX8_N_NODES];
        double ex[CVFEM_HEX8_N_NODES], ey[CVFEM_HEX8_N_NODES], ez[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const ptrdiff_t g    = elements[(ptrdiff_t)a * nelements + e];
            const double *const n = &u[g * CVFEM_CUDA_NF];
            ux[a] = n[0]; uy[a] = n[1]; uz[a] = n[2]; pe[a] = n[3];
            if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM) { ex[a] = px[g]; ey[a] = py[g]; ez[a] = pz[g]; }
        }
        if (GEOM == CVFEM_CUDA_GEOM_ISOPARAM) {
            const int32_t *const es = &slots[(ptrdiff_t)e * 64];
            if constexpr (VARIANT == CVFEM_CUDA_JAC_ISOPARAM_SYMPY)
                cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_isoparam(
                        rho, mu, ex, ey, ez, ux, uy, uz, es, values);
            else
                cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<true, PART>(
                        rho, mu, ex, ey, ez, ux, uy, uz, es, values);
            continue;
        }
        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];

        const int32_t *const es = &slots[e * 64];
        if      constexpr (VARIANT == CVFEM_CUDA_JAC_HANDWRITTEN)
            cvfem_hex8_ns_upwind_jacobian_add_slots<true>(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        else if constexpr (VARIANT == CVFEM_CUDA_JAC_SYMPY)
            cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        else if constexpr (VARIANT == CVFEM_CUDA_JAC_SYMPY_BLOCK)
            cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_blockwise(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
#ifdef CVFEM_ENABLE_SUBPAR
        else if constexpr (VARIANT == CVFEM_CUDA_JAC_SYMPY_ROW)
            cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_rowwise(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
#endif
#ifdef CVFEM_ENABLE_SUBPAR
        else
            cvfem_hex8_ns_upwind_sympy_jacobian_add_bsr_slots_facewise(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
#endif
        (void)pe;
    }
}

// Geometry-only assembly. Reads no velocity; run once per mesh.
__global__ void cvfem_hex8_assemble_linear_kernel(
        const ptrdiff_t nelements, const double mu,
        const int32_t *const __restrict__ slots,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        double *const __restrict__ values) {
    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < nelements;
         e += (ptrdiff_t)blockDim.x * gridDim.x) {
        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];
        cvfem_hex8_ns_upwind_jacobian_add_slots_linear<true>(mu, adj_e, det[e],
                                                             &slots[e * 64], values);
    }
}

// Velocity-dependent assembly, added on top of the restored linear part.
__global__ void cvfem_hex8_assemble_nonlinear_kernel(
        const ptrdiff_t nelements, const double rho, const double mu,
        const int32_t *const __restrict__ elements,
        const int32_t *const __restrict__ slots,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        double *const __restrict__ values) {
    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < nelements;
         e += (ptrdiff_t)blockDim.x * gridDim.x) {
        double ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES], uz[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const ptrdiff_t g     = elements[(ptrdiff_t)a * nelements + e];
            const double *const n = &u[g * CVFEM_CUDA_NF];
            ux[a] = n[0]; uy[a] = n[1]; uz[a] = n[2];
        }
        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];
        cvfem_hex8_ns_upwind_jacobian_add_slots_nonlinear<true>(rho, mu, adj_e, det[e],
                                                                ux, uy, uz,
                                                                &slots[e * 64], values);
    }
}

// Gather the touched blocks out of the full linear matrix into the compact buffer, once.
__global__ void cvfem_hex8_compact_linear_kernel(
        const ptrdiff_t n_blocks,
        const int32_t *const __restrict__ block_ids,
        const double  *const __restrict__ src,
        double *const __restrict__ compact) {
    for (ptrdiff_t t = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; t < n_blocks * 16;
         t += (ptrdiff_t)blockDim.x * gridDim.x) {
        const ptrdiff_t b = t >> 4, k = t & 15;
        compact[t] = src[(ptrdiff_t)block_ids[b] * 16 + k];
    }
}

// The viscous half for the pairs that are written once and never revisited.
__global__ void cvfem_hex8_assemble_static_kernel(
        const ptrdiff_t nelements, const double mu,
        const int32_t  *const __restrict__ slots,
        const uint16_t *const __restrict__ masks,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        double *const __restrict__ values) {
    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < nelements;
         e += (ptrdiff_t)blockDim.x * gridDim.x) {
        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];
        cvfem_hex8_ns_upwind_jacobian_add_slots_static<true>(
                mu, adj_e, det[e], &slots[e * 64], masks, values);
    }
}

// Everything that is rebuilt each iteration: the viscous half for the recomputed pairs,
// plus convection. Runs into blocks that were just zeroed.
__global__ void cvfem_hex8_assemble_dynamic_kernel(
        const ptrdiff_t nelements, const double rho, const double mu,
        const int32_t  *const __restrict__ elements,
        const int32_t  *const __restrict__ slots,
        const uint16_t *const __restrict__ masks,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        double *const __restrict__ values) {
    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < nelements;
         e += (ptrdiff_t)blockDim.x * gridDim.x) {
        double ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES], uz[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const ptrdiff_t g     = elements[(ptrdiff_t)a * nelements + e];
            const double *const n = &u[g * CVFEM_CUDA_NF];
            ux[a] = n[0]; uy[a] = n[1]; uz[a] = n[2];
        }
        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];
        cvfem_hex8_ns_upwind_jacobian_add_slots_dynamic<true>(rho, mu, adj_e, det[e],
                                                              ux, uy, uz, &slots[e * 64],
                                                              masks, values);
    }
}

// DIAG_MODE: 0 = everything, 1 = viscous only (constant), 2 = velocity-dependent only.
template <int DIAG_MODE>
__global__ void cvfem_hex8_assemble_diag_kernel(
        const ptrdiff_t nelements, const double rho, const double mu,
        const int32_t *const __restrict__ elements,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        double *const __restrict__ diag,
        const double  *const __restrict__ px = nullptr,
        const double  *const __restrict__ py = nullptr,
        const double  *const __restrict__ pz = nullptr) {
    for (ptrdiff_t e = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; e < nelements;
         e += (ptrdiff_t)blockDim.x * gridDim.x) {
        // -1 everywhere off the diagonal: those writes are dropped, so only the 8
        // diagonal blocks of this element are touched.
        int32_t sl[64];
        double  ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES], uz[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const int32_t g = elements[(ptrdiff_t)a * nelements + e];
#pragma unroll
            for (int b = 0; b < CVFEM_HEX8_N_NODES; ++b) sl[a * 8 + b] = -1;
            sl[a * 8 + a] = g;
            const double *const n = &u[(ptrdiff_t)g * CVFEM_CUDA_NF];
            ux[a] = n[0]; uy[a] = n[1]; uz[a] = n[2];
        }
        if constexpr (DIAG_MODE == 3) {
            // Isoparametric: same negative-slot trick, geometry rebuilt per element.
            double ex[CVFEM_HEX8_N_NODES], ey[CVFEM_HEX8_N_NODES], ez[CVFEM_HEX8_N_NODES];
#pragma unroll
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const int32_t g = elements[(ptrdiff_t)a * nelements + e];
                ex[a] = px[g]; ey[a] = py[g]; ez[a] = pz[g];
            }
            cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<true>(
                    rho, mu, ex, ey, ez, ux, uy, uz, sl, diag);
            continue;
        }
        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];

        if      constexpr (DIAG_MODE == 0)
            cvfem_hex8_ns_upwind_jacobian_add_slots<true>(rho, mu, adj_e, det[e], ux, uy, uz, sl, diag);
        else if constexpr (DIAG_MODE == 1)
            cvfem_hex8_ns_upwind_jacobian_add_slots_linear<true>(mu, adj_e, det[e], sl, diag);
        else
            cvfem_hex8_ns_upwind_jacobian_add_slots_nonlinear<true>(rho, mu, adj_e, det[e],
                                                                    ux, uy, uz, sl, diag);
    }
}

template <int OP>
__global__ void cvfem_hex8_boundary_kernel(
        const ptrdiff_t n_boundary, const ptrdiff_t nelements,
        const double rho, const double mu,
        const double Lx, const double Ly, const double Lz,
        const int32_t *const __restrict__ blist,
        const int32_t *const __restrict__ elements,
        const int32_t *const __restrict__ slots,
        const double  *const __restrict__ px,
        const double  *const __restrict__ py,
        const double  *const __restrict__ pz,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        const double  *const __restrict__ vin,
        double *const __restrict__ r,
        double *const __restrict__ values) {
    for (ptrdiff_t t = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; t < n_boundary;
         t += (ptrdiff_t)blockDim.x * gridDim.x) {
        const ptrdiff_t e = blist[t];
        int32_t ev[CVFEM_HEX8_N_NODES];
        double  x[CVFEM_HEX8_N_NODES], y[CVFEM_HEX8_N_NODES], z[CVFEM_HEX8_N_NODES];
        double  ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES], uz[CVFEM_HEX8_N_NODES];
        double  pe[CVFEM_HEX8_N_NODES];
        double  vx[CVFEM_HEX8_N_NODES], vy[CVFEM_HEX8_N_NODES], vz[CVFEM_HEX8_N_NODES];
        double  q[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const int32_t g = elements[(ptrdiff_t)a * nelements + e];
            ev[a] = g;
            x[a] = px[g]; y[a] = py[g]; z[a] = pz[g];
            const double *const nd = &u[(ptrdiff_t)g * CVFEM_CUDA_NF];
            ux[a] = nd[0]; uy[a] = nd[1]; uz[a] = nd[2]; pe[a] = nd[3];
            if (OP == BOUNDARY_JV) {
                const double *const nv = &vin[(ptrdiff_t)g * CVFEM_CUDA_NF];
                vx[a] = nv[0]; vy[a] = nv[1]; vz[a] = nv[2]; q[a] = nv[3];
            }
        }
        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];

        if (OP == BOUNDARY_ASSEMBLE) {
            // Accumulates straight into the global BSR through CVFEM_ATOMIC_ADD.
            boundary_scs_add_jacobian<true, false>(rho, mu, adj_e, det[e], Lx, Ly, Lz,
                                            x, y, z, ux, uy, uz, &slots[e * 64], values);
        } else {
            double re[CVFEM_HEX8_N_DOF];
#pragma unroll
            for (int i = 0; i < CVFEM_HEX8_N_DOF; ++i) re[i] = 0.0;
            if (OP == BOUNDARY_RESIDUAL)
                boundary_scs_add_residual<false>(rho, mu, adj_e, det[e], Lx, Ly, Lz,
                                          x, y, z, ux, uy, uz, pe, re);
            else
                boundary_scs_add_jacobian_action<false>(rho, mu, adj_e, det[e], Lx, Ly, Lz,
                                                 x, y, z, ux, uy, uz, vx, vy, vz, q, re);
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                double *const dst = &r[(ptrdiff_t)ev[a] * CVFEM_CUDA_NF];
#pragma unroll
                for (int f = 0; f < CVFEM_CUDA_NF; ++f) {
                    const double v = re[a * 4 + f];
                    if (v != 0.0) atomicAdd(&dst[f], v);
                }
            }
        }
    }
}
