#include "sfem_GeneratedNeoHookeanOgden_cuda_c_abi.hpp"

extern "C" int cu_neohookean_ogden_proteus_hex8_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz,
        void *const stream
);
extern "C" int cu_neohookean_ogden_proteus_hex8_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz,
        void *const stream
);
extern "C" int cu_neohookean_ogden_proteus_hex8_hessian_bsr_i_msoa(
        const int scalar_bytes,
        const int values_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values,
        void *const stream
);
extern "C" int cu_neohookean_ogden_proteus_hex8_objective_steps_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value,
        void *const stream
);
extern "C" int cu_neohookean_ogden_proteus_hex8_apply_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz,
        void *const stream
);
extern "C" int cu_neohookean_ogden_proteus_hex8_gradient_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz,
        void *const stream
);
extern "C" int cu_neohookean_ogden_proteus_hex8_objective_steps_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value,
        void *const stream
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_neohookean_ogden_proteus_hex8_apply_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_neohookean_ogden_proteus_hex8_gradient_soa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_neohookean_ogden_proteus_hex8_objective_soa_diagnostics(
        void
);

namespace {
  //! Reorder a device table of connectivity rows on the device.
  template <int N>
  struct ConnectivityOrder {
    int source[N];
  };

  template <typename T, int N>
  __global__ void permute_connectivity(T *const *const elements,
                                       const ConnectivityOrder<N> order,
                                       T **const permuted) {
    const int k = threadIdx.x;
    if (k < N) permuted[k] = elements[order.source[k]];
  }
}  // namespace

extern "C" int cu_neohookean_ogden_hex8_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 8 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<8> proteus_elements_order{{0, 1, 3, 2, 4, 5, 7, 6}};
    permute_connectivity<idx_t, 8><<<1, 8, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_neohookean_ogden_proteus_hex8_apply_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" int cu_neohookean_ogden_hex8_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 8 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<8> proteus_elements_order{{0, 1, 3, 2, 4, 5, 7, 6}};
    permute_connectivity<idx_t, 8><<<1, 8, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_neohookean_ogden_proteus_hex8_gradient_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" int cu_neohookean_ogden_hex8_hessian_bsr_i_msoa(
        const int scalar_bytes,
        const int values_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 8 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<8> proteus_elements_order{{0, 1, 3, 2, 4, 5, 7, 6}};
    permute_connectivity<idx_t, 8><<<1, 8, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_neohookean_ogden_proteus_hex8_hessian_bsr_i_msoa(scalar_bytes, values_bytes, nelements, nnodes, proteus_elements, points, lmbda, mu, u_stride, ux, uy, uz, rowptr, colidx, values, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" int cu_neohookean_ogden_hex8_objective_steps_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 8 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<8> proteus_elements_order{{0, 1, 3, 2, 4, 5, 7, 6}};
    permute_connectivity<idx_t, 8><<<1, 8, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_neohookean_ogden_proteus_hex8_objective_steps_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" int cu_neohookean_ogden_hex8_apply_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 8 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<8> proteus_elements_order{{0, 1, 3, 2, 4, 5, 7, 6}};
    permute_connectivity<idx_t, 8><<<1, 8, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_neohookean_ogden_proteus_hex8_apply_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, out_stride, outx, outy, outz, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" int cu_neohookean_ogden_hex8_gradient_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t out_stride,
        void *const RSTR outx,
        void *const RSTR outy,
        void *const RSTR outz,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 8 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<8> proteus_elements_order{{0, 1, 3, 2, 4, 5, 7, 6}};
    permute_connectivity<idx_t, 8><<<1, 8, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_neohookean_ogden_proteus_hex8_gradient_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, out_stride, outx, outy, outz, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" int cu_neohookean_ogden_hex8_objective_steps_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_adj4,
        const geom_t *const RSTR g_adj5,
        const geom_t *const RSTR g_adj6,
        const geom_t *const RSTR g_adj7,
        const geom_t *const RSTR g_adj8,
        const geom_t *const RSTR g_det0,
        const real_t lmbda,
        const real_t mu,
        const ptrdiff_t u_stride,
        const void *const RSTR ux,
        const void *const RSTR uy,
        const void *const RSTR uz,
        const ptrdiff_t h_stride,
        const void *const RSTR hx,
        const void *const RSTR hy,
        const void *const RSTR hz,
        const int nsteps,
        const void *const RSTR steps,
        void *const RSTR value,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 8 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<8> proteus_elements_order{{0, 1, 3, 2, 4, 5, 7, 6}};
    permute_connectivity<idx_t, 8><<<1, 8, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_neohookean_ogden_proteus_hex8_objective_steps_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, lmbda, mu, u_stride, ux, uy, uz, h_stride, hx, hy, hz, nsteps, steps, value, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_neohookean_ogden_hex8_apply_soa_diagnostics(
        void
) {
    return cu_neohookean_ogden_proteus_hex8_apply_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_neohookean_ogden_hex8_gradient_soa_diagnostics(
        void
) {
    return cu_neohookean_ogden_proteus_hex8_gradient_soa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_neohookean_ogden_hex8_objective_soa_diagnostics(
        void
) {
    return cu_neohookean_ogden_proteus_hex8_objective_soa_diagnostics();
}
