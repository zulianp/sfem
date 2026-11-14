#include "sfem_GeneratedNeumannGeneral_cuda_c_abi.hpp"

extern "C" int cu_neumann_general_proteus_quad4_edgeshell2_boundary_residual_ss_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
        const real_t t0,
        const real_t t0_010,
        const real_t t0_100,
        const real_t t1,
        const real_t t1_010,
        const real_t t1_100,
        const int out_stride,
        real_t *const RSTR out0,
        real_t *const RSTR out1,
        void *const stream
);
extern "C" int cu_neumann_general_proteus_quad4_edgeshell2_boundary_residual_ss_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
        const float t0,
        const float t0_010,
        const float t0_100,
        const float t1,
        const float t1_010,
        const float t1_100,
        const int out_stride,
        float *const RSTR out0,
        float *const RSTR out1,
        void *const stream
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

extern "C" int cu_neumann_general_quad4_edgeshell2_boundary_residual_ss_soa(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
        const real_t t0,
        const real_t t0_010,
        const real_t t0_100,
        const real_t t1,
        const real_t t1_010,
        const real_t t1_100,
        const int out_stride,
        real_t *const RSTR out0,
        real_t *const RSTR out1,
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
    const int status = cu_neumann_general_proteus_quad4_edgeshell2_boundary_residual_ss_soa(nsides, nnodes, proteus_elements, parent, side_idx, points, t0, t0_010, t0_100, t1, t1_010, t1_100, out_stride, out0, out1, stream);
#if !defined(SFEM_ENABLE_HIP)
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
#endif
    return status;
}

extern "C" int cu_neumann_general_quad4_edgeshell2_boundary_residual_ss_soa_float(
        const ptrdiff_t nsides,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const element_idx_t *const RSTR parent,
        const int16_t *const RSTR side_idx,
        const geom_t *const *const RSTR points,
        const float t0,
        const float t0_010,
        const float t0_100,
        const float t1,
        const float t1_010,
        const float t1_100,
        const int out_stride,
        float *const RSTR out0,
        float *const RSTR out1,
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
    const int status = cu_neumann_general_proteus_quad4_edgeshell2_boundary_residual_ss_soa_float(nsides, nnodes, proteus_elements, parent, side_idx, points, t0, t0_010, t0_100, t1, t1_010, t1_100, out_stride, out0, out1, stream);
#if !defined(SFEM_ENABLE_HIP)
    cudaFreeAsync(proteus_elements, (cudaStream_t)stream);
#endif
    return status;
}
