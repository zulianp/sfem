#include "sfem_GeneratedNavierStokes_cuda_c_abi.hpp"

extern "C" int cu_navier_stokes_form_2_p_u_proteus_hex27_proteus_hex8_jacobian_action_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[3],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
);
extern "C" int cu_navier_stokes_form_2_p_u_proteus_hex27_proteus_hex8_jacobian_action_a_msoa(
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
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[3],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_navier_stokes_form_2_p_u_proteus_hex27_proteus_hex8_jacobian_action_esoa_diagnostics(
        void
);
extern "C" const sfem::codegen::KernelDiagnostics * cu_navier_stokes_form_2_p_u_proteus_hex27_proteus_hex8_residual_esoa_diagnostics(
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

extern "C" int cu_navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[3],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 27 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<27> proteus_elements_order{{0, 8, 1, 11, 24, 9, 3, 10, 2, 16, 20, 17, 23, 26, 21, 19, 22, 18, 4, 12, 5, 15, 25, 13, 7, 14, 6}};
    permute_connectivity<idx_t, 27><<<1, 27, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_navier_stokes_form_2_p_u_proteus_hex27_proteus_hex8_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, proteus_elements, points, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" int cu_navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_a_msoa(
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
        const ptrdiff_t direction_stride,
        const void *const RSTR u_direction_data[3],
        const void *const RSTR p_direction_data,
        const ptrdiff_t out_stride,
        void *const RSTR u_out[3],
        void *const RSTR p_out,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 27 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<27> proteus_elements_order{{0, 8, 1, 11, 24, 9, 3, 10, 2, 16, 20, 17, 23, 26, 21, 19, 22, 18, 4, 12, 5, 15, 25, 13, 7, 14, 6}};
    permute_connectivity<idx_t, 27><<<1, 27, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_navier_stokes_form_2_p_u_proteus_hex27_proteus_hex8_jacobian_action_a_msoa(scalar_bytes, nelements, nnodes, proteus_elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, direction_stride, u_direction_data, p_direction_data, out_stride, u_out, p_out, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_navier_stokes_form_2_p_u_hex27_hex8_jacobian_action_esoa_diagnostics(
        void
) {
    return cu_navier_stokes_form_2_p_u_proteus_hex27_proteus_hex8_jacobian_action_esoa_diagnostics();
}

extern "C" const sfem::codegen::KernelDiagnostics * cu_navier_stokes_form_2_p_u_hex27_hex8_residual_esoa_diagnostics(
        void
) {
    return cu_navier_stokes_form_2_p_u_proteus_hex27_proteus_hex8_residual_esoa_diagnostics();
}
