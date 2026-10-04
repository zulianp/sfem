#pragma once

// The ELEMENT-COLOURED layout's device kernel: a colour is a contiguous element range, so
// no two threads of a launch touch the same node and the atomics are gone.
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


// Same element kernel as the atomic version, but writing with a plain += because the
// colouring guarantees no two threads in flight touch the same matrix block. VARIANT 0
// is the hand-written kernel with Atomic=false; the SymPy *_local_slots family already
// accumulates without atomics.
template <int VARIANT>
__global__ void cvfem_hex8_assemble_ecolored_kernel(
        const ptrdiff_t n_in_color, const ptrdiff_t color_begin, const ptrdiff_t nelements,
        const double rho, const double mu,
        const int32_t *const __restrict__ order,
        const int32_t *const __restrict__ elements,
        const int32_t *const __restrict__ slots,
        const double  *const __restrict__ adj,
        const double  *const __restrict__ det,
        const double  *const __restrict__ u,
        double *const __restrict__ values,
        const double  *const __restrict__ px = nullptr,
        const double  *const __restrict__ py = nullptr,
        const double  *const __restrict__ pz = nullptr) {
    for (ptrdiff_t t = blockIdx.x * (ptrdiff_t)blockDim.x + threadIdx.x; t < n_in_color;
         t += (ptrdiff_t)blockDim.x * gridDim.x) {
        const ptrdiff_t e = order[color_begin + t];
        double ux[CVFEM_HEX8_N_NODES], uy[CVFEM_HEX8_N_NODES];
        double uz[CVFEM_HEX8_N_NODES], pe[CVFEM_HEX8_N_NODES];
#pragma unroll
        for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
            const ptrdiff_t g     = elements[(ptrdiff_t)a * nelements + e];
            const double *const n = &u[g * CVFEM_CUDA_NF];
            ux[a] = n[0]; uy[a] = n[1]; uz[a] = n[2]; pe[a] = n[3];
        }
        double adj_e[9];
#pragma unroll
        for (int c = 0; c < 9; ++c) adj_e[c] = adj[(ptrdiff_t)c * nelements + e];

        const int32_t *const es = &slots[e * 64];
        if constexpr (VARIANT == CVFEM_CUDA_JAC_ISOPARAM) {
            // Colouring makes the writes race-free, so this accumulates without atomics
            // exactly as the affine coloured variants do.
            double ex[CVFEM_HEX8_N_NODES], ey[CVFEM_HEX8_N_NODES], ez[CVFEM_HEX8_N_NODES];
#pragma unroll
            for (int a = 0; a < CVFEM_HEX8_N_NODES; ++a) {
                const ptrdiff_t g = elements[(ptrdiff_t)a * nelements + e];
                ex[a] = px[g]; ey[a] = py[g]; ez[a] = pz[g];
            }
            cvfem_hex8_ns_upwind_jacobian_add_slots_isoparam<false>(
                    rho, mu, ex, ey, ez, ux, uy, uz, es, values);
            (void)pe;
            continue;
        }
        if      constexpr (VARIANT == CVFEM_CUDA_JAC_HANDWRITTEN)
            cvfem_hex8_ns_upwind_jacobian_add_slots<false>(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        else if constexpr (VARIANT == CVFEM_CUDA_JAC_SYMPY)
            cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
        else if constexpr (VARIANT == CVFEM_CUDA_JAC_SYMPY_BLOCK)
            cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_blockwise(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
#ifdef CVFEM_ENABLE_SUBPAR
        else if constexpr (VARIANT == CVFEM_CUDA_JAC_SYMPY_ROW)
            cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_rowwise(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
#endif
#ifdef CVFEM_ENABLE_SUBPAR
        else
            cvfem_hex8_ns_upwind_sympy_jacobian_add_local_slots_facewise(rho, mu, adj_e, det[e], ux, uy, uz, es, values);
#endif
        (void)pe;
    }
}
