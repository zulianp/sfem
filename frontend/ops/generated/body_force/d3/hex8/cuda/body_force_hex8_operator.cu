#include "sfem_GeneratedBodyForce_cuda_c_abi.hpp"

extern "C" int cu_body_force_proteus_hex8_residual_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output,
        void *const stream
);
extern "C" int cu_body_force_proteus_hex8_residual_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const real_t g0,
        const real_t g1,
        const real_t g2,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out,
        void *const stream
);
extern "C" int cu_body_force_proteus_hex8_residual_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const real_t g0,
        const real_t g1,
        const real_t g2,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out,
        void *const stream
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_body_force_proteus_hex8_jacobian_action_esoa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_body_force_proteus_hex8_residual_esoa_diagnostics(
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

extern "C" int cu_body_force_hex8_residual_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        void *const RSTR output,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 8 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<8> proteus_elements_order{{0, 1, 3, 2, 4, 5, 7, 6}};
    permute_connectivity<idx_t, 8><<<1, 8, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_body_force_proteus_hex8_residual_i_maos(scalar_bytes, nelements, nnodes, proteus_elements, points, parameters, output, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" int cu_body_force_hex8_residual_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const real_t g0,
        const real_t g1,
        const real_t g2,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 8 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<8> proteus_elements_order{{0, 1, 3, 2, 4, 5, 7, 6}};
    permute_connectivity<idx_t, 8><<<1, 8, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_body_force_proteus_hex8_residual_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, density, g0, g1, g2, out_stride, u0_out, u1_out, u2_out, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" int cu_body_force_hex8_residual_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const real_t g0,
        const real_t g1,
        const real_t g2,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 8 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<8> proteus_elements_order{{0, 1, 3, 2, 4, 5, 7, 6}};
    permute_connectivity<idx_t, 8><<<1, 8, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_body_force_proteus_hex8_residual_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_det0, density, g0, g1, g2, out_stride, u0_out, u1_out, u2_out, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_body_force_hex8_jacobian_action_esoa_diagnostics(
        void
) {
    return cu_body_force_proteus_hex8_jacobian_action_esoa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_body_force_hex8_residual_esoa_diagnostics(
        void
) {
    return cu_body_force_proteus_hex8_residual_esoa_diagnostics();
}
