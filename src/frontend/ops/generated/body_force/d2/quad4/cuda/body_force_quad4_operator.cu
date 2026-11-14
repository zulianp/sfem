#include "sfem_GeneratedBodyForce_cuda_c_abi.hpp"

extern "C" int cu_body_force_proteus_quad4_residual_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output,
        void *const stream
);
extern "C" int cu_body_force_proteus_quad4_residual_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const real_t g0,
        const real_t g1,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const stream
);
extern "C" int cu_body_force_proteus_quad4_residual_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const real_t g0,
        const real_t g1,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const stream
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_body_force_proteus_quad4_jacobian_action_esoa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_body_force_proteus_quad4_residual_esoa_diagnostics(
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

extern "C" int cu_body_force_quad4_residual_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
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
    const int status = cu_body_force_proteus_quad4_residual_i_maos(scalar_bytes, nelements, nnodes, proteus_elements, points, parameters, output, stream);
#if !defined(SFEM_ENABLE_HIP)
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
#endif
    return status;
}

extern "C" int cu_body_force_quad4_residual_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const real_t g0,
        const real_t g1,
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
    const int status = cu_body_force_proteus_quad4_residual_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, density, g0, g1, out_stride, u0_out, u1_out, stream);
#if !defined(SFEM_ENABLE_HIP)
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
#endif
    return status;
}

extern "C" int cu_body_force_quad4_residual_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const real_t g0,
        const real_t g1,
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
    const int status = cu_body_force_proteus_quad4_residual_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_det0, density, g0, g1, out_stride, u0_out, u1_out, stream);
#if !defined(SFEM_ENABLE_HIP)
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
#endif
    return status;
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_body_force_quad4_jacobian_action_esoa_diagnostics(
        void
) {
    return cu_body_force_proteus_quad4_jacobian_action_esoa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_body_force_quad4_residual_esoa_diagnostics(
        void
) {
    return cu_body_force_proteus_quad4_residual_esoa_diagnostics();
}
