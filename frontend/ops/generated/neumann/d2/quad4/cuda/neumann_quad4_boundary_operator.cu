#include "sfem_GeneratedNeumann_cuda_c_abi.hpp"

extern "C" int cu_neumann_proteus_quad4_edgeshell2_boundary_residual_ss_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
        const real_t t0,
        const real_t t1,
        const int out_stride,
        real_t *const RSTR out0,
        real_t *const RSTR out1,
        void *const stream
);
extern "C" int cu_neumann_proteus_quad4_edgeshell2_boundary_residual_ss_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
        const float t0,
        const float t1,
        const int out_stride,
        float *const RSTR out0,
        float *const RSTR out1,
        void *const stream
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

extern "C" int cu_neumann_quad4_edgeshell2_boundary_residual_ss_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
        const real_t t0,
        const real_t t1,
        const int out_stride,
        real_t *const RSTR out0,
        real_t *const RSTR out1,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 4 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<4> proteus_elements_order{{0, 1, 3, 2}};
    permute_connectivity<idx_t, 4><<<1, 4, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_neumann_proteus_quad4_edgeshell2_boundary_residual_ss_soa(nsides, nnodes, proteus_elements, parent, side_idx, points, t0, t1, out_stride, out0, out1, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}

extern "C" int cu_neumann_quad4_edgeshell2_boundary_residual_ss_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
        const float t0,
        const float t1,
        const int out_stride,
        float *const RSTR out0,
        float *const RSTR out1,
        void *const stream
) {
    idx_t **proteus_elements = nullptr;
    if (cudaMallocAsync((void **)&proteus_elements, 4 * sizeof(idx_t *), (cudaStream_t)stream) != cudaSuccess) return SFEM_FAILURE;
    const ConnectivityOrder<4> proteus_elements_order{{0, 1, 3, 2}};
    permute_connectivity<idx_t, 4><<<1, 4, 0, (cudaStream_t)stream>>>(elements, proteus_elements_order, proteus_elements);
    const int status = cu_neumann_proteus_quad4_edgeshell2_boundary_residual_ss_soa_float(nsides, nnodes, proteus_elements, parent, side_idx, points, t0, t1, out_stride, out0, out1, stream);
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
    return status;
}
