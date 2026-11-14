#include "sfem_GeneratedLinearKelvinVoigt_cuda_c_abi.hpp"

extern "C" int cu_linear_kelvin_voigt_total_proteus_quad4_hessian_bsr_i_msoa(
        const int scalar_bytes,
        const int values_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t lmbda,
        const real_t mu,
        const real_t u_dt_shift,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values,
        void *const stream
);
extern "C" int cu_linear_kelvin_voigt_total_proteus_quad4_jacobian_action_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR direction,
        void *const RSTR output,
        void *const stream
);
extern "C" int cu_linear_kelvin_voigt_total_proteus_quad4_jacobian_action_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t lmbda,
        const real_t mu,
        const real_t u_dt_shift,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const stream
);
extern "C" int cu_linear_kelvin_voigt_total_proteus_quad4_residual_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        void *const RSTR output,
        void *const stream
);
extern "C" int cu_linear_kelvin_voigt_total_proteus_quad4_residual_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t lmbda,
        const real_t mu,
        const real_t u_dt_shift,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const stream
);
extern "C" int cu_linear_kelvin_voigt_total_proteus_quad4_jacobian_action_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t eta_b,
        const real_t eta_s,
        const real_t lmbda,
        const real_t mu,
        const real_t u_dt_shift,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const stream
);
extern "C" int cu_linear_kelvin_voigt_total_proteus_quad4_residual_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t eta_b,
        const real_t eta_s,
        const real_t lmbda,
        const real_t mu,
        const real_t u_dt_shift,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const stream
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_linear_kelvin_voigt_total_proteus_quad4_jacobian_action_esoa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_linear_kelvin_voigt_total_proteus_quad4_residual_esoa_diagnostics(
        void
);

#if defined(SFEM_ENABLE_HIP)
#include <unordered_map>
#endif

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

#if defined(SFEM_ENABLE_HIP)
  //! The permuted table's storage under HIP: one per host thread and stream, kept for the
  //! process and rewritten in stream order before each kernel that reads it. HIP 6.4 does
  //! not capture a stream-ordered allocation into a graph (a crash in the runtime), and the
  //! products of a captured solver run (the coarse CG's) allocate nothing this way.
  template <typename T, int N>
  T **permuted_connectivity_table(const cudaStream_t stream) {
    static thread_local std::unordered_map<cudaStream_t, T **> tables;
    T **&table = tables[stream];
    if (!table && cudaMalloc((void **)&table, N * sizeof(T *)) != cudaSuccess) table = nullptr;
    return table;
  }
#endif
}  // namespace

extern "C" int cu_linear_kelvin_voigt_total_quad4_hessian_bsr_i_msoa(
        const int scalar_bytes,
        const int values_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t lmbda,
        const real_t mu,
        const real_t u_dt_shift,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values,
        void *const stream
) {
#if defined(SFEM_ENABLE_HIP)
    idx_t **proteus_elements = permuted_connectivity_table<idx_t, 4>((cudaStream_t)stream);
    if (!proteus_elements) return SFEM_FAILURE;
#else
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 4 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
#endif
    const ConnectivityOrder<4> proteus_elements_order{{0, 1, 3, 2}};
    permute_connectivity<idx_t, 4><<<1, 4, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_linear_kelvin_voigt_total_proteus_quad4_hessian_bsr_i_msoa(scalar_bytes, values_bytes, nelements, nnodes, proteus_elements, points, eta_b, eta_s, lmbda, mu, u_dt_shift, rowptr, colidx, values, stream);
#if !defined(SFEM_ENABLE_HIP)
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
#endif
    return status;
}

extern "C" int cu_linear_kelvin_voigt_total_quad4_jacobian_action_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR direction,
        void *const RSTR output,
        void *const stream
) {
#if defined(SFEM_ENABLE_HIP)
    idx_t **proteus_elements = permuted_connectivity_table<idx_t, 4>((cudaStream_t)stream);
    if (!proteus_elements) return SFEM_FAILURE;
#else
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 4 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
#endif
    const ConnectivityOrder<4> proteus_elements_order{{0, 1, 3, 2}};
    permute_connectivity<idx_t, 4><<<1, 4, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_linear_kelvin_voigt_total_proteus_quad4_jacobian_action_i_maos(scalar_bytes, nelements, nnodes, proteus_elements, points, parameters, direction, output, stream);
#if !defined(SFEM_ENABLE_HIP)
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
#endif
    return status;
}

extern "C" int cu_linear_kelvin_voigt_total_quad4_jacobian_action_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t lmbda,
        const real_t mu,
        const real_t u_dt_shift,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const stream
) {
#if defined(SFEM_ENABLE_HIP)
    idx_t **proteus_elements = permuted_connectivity_table<idx_t, 4>((cudaStream_t)stream);
    if (!proteus_elements) return SFEM_FAILURE;
#else
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 4 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
#endif
    const ConnectivityOrder<4> proteus_elements_order{{0, 1, 3, 2}};
    permute_connectivity<idx_t, 4><<<1, 4, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_linear_kelvin_voigt_total_proteus_quad4_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, eta_b, eta_s, lmbda, mu, u_dt_shift, direction_stride, u0_direction, u1_direction, out_stride, u0_out, u1_out, stream);
#if !defined(SFEM_ENABLE_HIP)
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
#endif
    return status;
}

extern "C" int cu_linear_kelvin_voigt_total_quad4_residual_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        const void *const RSTR previous,
        void *const RSTR output,
        void *const stream
) {
#if defined(SFEM_ENABLE_HIP)
    idx_t **proteus_elements = permuted_connectivity_table<idx_t, 4>((cudaStream_t)stream);
    if (!proteus_elements) return SFEM_FAILURE;
#else
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 4 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
#endif
    const ConnectivityOrder<4> proteus_elements_order{{0, 1, 3, 2}};
    permute_connectivity<idx_t, 4><<<1, 4, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_linear_kelvin_voigt_total_proteus_quad4_residual_i_maos(scalar_bytes, nelements, nnodes, proteus_elements, points, parameters, current, previous, output, stream);
#if !defined(SFEM_ENABLE_HIP)
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
#endif
    return status;
}

extern "C" int cu_linear_kelvin_voigt_total_quad4_residual_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t eta_b,
        const real_t eta_s,
        const real_t lmbda,
        const real_t mu,
        const real_t u_dt_shift,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const stream
) {
#if defined(SFEM_ENABLE_HIP)
    idx_t **proteus_elements = permuted_connectivity_table<idx_t, 4>((cudaStream_t)stream);
    if (!proteus_elements) return SFEM_FAILURE;
#else
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 4 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
#endif
    const ConnectivityOrder<4> proteus_elements_order{{0, 1, 3, 2}};
    permute_connectivity<idx_t, 4><<<1, 4, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_linear_kelvin_voigt_total_proteus_quad4_residual_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, u0, u1, previous_stride, u0_old, u1_old, out_stride, u0_out, u1_out, stream);
#if !defined(SFEM_ENABLE_HIP)
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
#endif
    return status;
}

extern "C" int cu_linear_kelvin_voigt_total_quad4_jacobian_action_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t eta_b,
        const real_t eta_s,
        const real_t lmbda,
        const real_t mu,
        const real_t u_dt_shift,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const stream
) {
#if defined(SFEM_ENABLE_HIP)
    idx_t **proteus_elements = permuted_connectivity_table<idx_t, 4>((cudaStream_t)stream);
    if (!proteus_elements) return SFEM_FAILURE;
#else
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 4 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
#endif
    const ConnectivityOrder<4> proteus_elements_order{{0, 1, 3, 2}};
    permute_connectivity<idx_t, 4><<<1, 4, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_linear_kelvin_voigt_total_proteus_quad4_jacobian_action_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, direction_stride, u0_direction, u1_direction, out_stride, u0_out, u1_out, stream);
#if !defined(SFEM_ENABLE_HIP)
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
#endif
    return status;
}

extern "C" int cu_linear_kelvin_voigt_total_quad4_residual_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_adj0,
        const geom_t *const RSTR g_adj1,
        const geom_t *const RSTR g_adj2,
        const geom_t *const RSTR g_adj3,
        const geom_t *const RSTR g_det0,
        const real_t eta_b,
        const real_t eta_s,
        const real_t lmbda,
        const real_t mu,
        const real_t u_dt_shift,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const ptrdiff_t previous_stride,
        const void *const RSTR u0_old,
        const void *const RSTR u1_old,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const stream
) {
#if defined(SFEM_ENABLE_HIP)
    idx_t **proteus_elements = permuted_connectivity_table<idx_t, 4>((cudaStream_t)stream);
    if (!proteus_elements) return SFEM_FAILURE;
#else
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 4 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
#endif
    const ConnectivityOrder<4> proteus_elements_order{{0, 1, 3, 2}};
    permute_connectivity<idx_t, 4><<<1, 4, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_linear_kelvin_voigt_total_proteus_quad4_residual_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_det0, eta_b, eta_s, lmbda, mu, u_dt_shift, current_stride, u0, u1, previous_stride, u0_old, u1_old, out_stride, u0_out, u1_out, stream);
#if !defined(SFEM_ENABLE_HIP)
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
#endif
    return status;
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_linear_kelvin_voigt_total_quad4_jacobian_action_esoa_diagnostics(
        void
) {
    return cu_linear_kelvin_voigt_total_proteus_quad4_jacobian_action_esoa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_linear_kelvin_voigt_total_quad4_residual_esoa_diagnostics(
        void
) {
    return cu_linear_kelvin_voigt_total_proteus_quad4_residual_esoa_diagnostics();
}
