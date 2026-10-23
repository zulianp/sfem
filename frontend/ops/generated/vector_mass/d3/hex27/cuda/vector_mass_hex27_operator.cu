#include "sfem_GeneratedVectorMass_cuda_c_abi.hpp"

extern "C" int cu_vector_mass_proteus_hex27_hessian_bsr_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values,
        void *const stream
);
extern "C" int cu_vector_mass_proteus_hex27_jacobian_action_i_maos(
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
extern "C" int cu_vector_mass_proteus_hex27_jacobian_action_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const void *const RSTR u2_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out,
        void *const stream
);
extern "C" int cu_vector_mass_proteus_hex27_residual_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        void *const RSTR output,
        void *const stream
);
extern "C" int cu_vector_mass_proteus_hex27_residual_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const void *const RSTR u2,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out,
        void *const stream
);
extern "C" int cu_vector_mass_proteus_hex27_jacobian_action_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const void *const RSTR u2_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out,
        void *const stream
);
extern "C" int cu_vector_mass_proteus_hex27_residual_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const void *const RSTR u2,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out,
        void *const stream
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_vector_mass_proteus_hex27_jacobian_action_esoa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_vector_mass_proteus_hex27_residual_esoa_diagnostics(
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

extern "C" int cu_vector_mass_hex27_hessian_bsr_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const count_t *const RSTR rowptr,
        const idx_t *const RSTR colidx,
        void *const RSTR values,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 27 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<27> proteus_elements_order{{0, 8, 1, 11, 24, 9, 3, 10, 2, 16, 20, 17, 23, 26, 21, 19, 22, 18, 4, 12, 5, 15, 25, 13, 7, 14, 6}};
    permute_connectivity<idx_t, 27><<<1, 27, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_vector_mass_proteus_hex27_hessian_bsr_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, density, rowptr, colidx, values, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" int cu_vector_mass_hex27_jacobian_action_i_maos(
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
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 27 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<27> proteus_elements_order{{0, 8, 1, 11, 24, 9, 3, 10, 2, 16, 20, 17, 23, 26, 21, 19, 22, 18, 4, 12, 5, 15, 25, 13, 7, 14, 6}};
    permute_connectivity<idx_t, 27><<<1, 27, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_vector_mass_proteus_hex27_jacobian_action_i_maos(scalar_bytes, nelements, nnodes, proteus_elements, points, parameters, direction, output, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" int cu_vector_mass_hex27_jacobian_action_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const void *const RSTR u2_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 27 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<27> proteus_elements_order{{0, 8, 1, 11, 24, 9, 3, 10, 2, 16, 20, 17, 23, 26, 21, 19, 22, 18, 4, 12, 5, 15, 25, 13, 7, 14, 6}};
    permute_connectivity<idx_t, 27><<<1, 27, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_vector_mass_proteus_hex27_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, density, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" int cu_vector_mass_hex27_residual_i_maos(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const void *const RSTR parameters,
        const void *const RSTR current,
        void *const RSTR output,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 27 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<27> proteus_elements_order{{0, 8, 1, 11, 24, 9, 3, 10, 2, 16, 20, 17, 23, 26, 21, 19, 22, 18, 4, 12, 5, 15, 25, 13, 7, 14, 6}};
    permute_connectivity<idx_t, 27><<<1, 27, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_vector_mass_proteus_hex27_residual_i_maos(scalar_bytes, nelements, nnodes, proteus_elements, points, parameters, current, output, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" int cu_vector_mass_hex27_residual_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t density,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const void *const RSTR u2,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 27 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<27> proteus_elements_order{{0, 8, 1, 11, 24, 9, 3, 10, 2, 16, 20, 17, 23, 26, 21, 19, 22, 18, 4, 12, 5, 15, 25, 13, 7, 14, 6}};
    permute_connectivity<idx_t, 27><<<1, 27, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_vector_mass_proteus_hex27_residual_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, density, current_stride, u0, u1, u2, out_stride, u0_out, u1_out, u2_out, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" int cu_vector_mass_hex27_jacobian_action_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const ptrdiff_t direction_stride,
        const void *const RSTR u0_direction,
        const void *const RSTR u1_direction,
        const void *const RSTR u2_direction,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 27 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<27> proteus_elements_order{{0, 8, 1, 11, 24, 9, 3, 10, 2, 16, 20, 17, 23, 26, 21, 19, 22, 18, 4, 12, 5, 15, 25, 13, 7, 14, 6}};
    permute_connectivity<idx_t, 27><<<1, 27, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_vector_mass_proteus_hex27_jacobian_action_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_det0, density, direction_stride, u0_direction, u1_direction, u2_direction, out_stride, u0_out, u1_out, u2_out, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" int cu_vector_mass_hex27_residual_a_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const RSTR g_det0,
        const real_t density,
        const ptrdiff_t current_stride,
        const void *const RSTR u0,
        const void *const RSTR u1,
        const void *const RSTR u2,
        const ptrdiff_t out_stride,
        void *const RSTR u0_out,
        void *const RSTR u1_out,
        void *const RSTR u2_out,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 27 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<27> proteus_elements_order{{0, 8, 1, 11, 24, 9, 3, 10, 2, 16, 20, 17, 23, 26, 21, 19, 22, 18, 4, 12, 5, 15, 25, 13, 7, 14, 6}};
    permute_connectivity<idx_t, 27><<<1, 27, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_vector_mass_proteus_hex27_residual_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_det0, density, current_stride, u0, u1, u2, out_stride, u0_out, u1_out, u2_out, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_vector_mass_hex27_jacobian_action_esoa_diagnostics(
        void
) {
    return cu_vector_mass_proteus_hex27_jacobian_action_esoa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_vector_mass_hex27_residual_esoa_diagnostics(
        void
) {
    return cu_vector_mass_proteus_hex27_residual_esoa_diagnostics();
}
