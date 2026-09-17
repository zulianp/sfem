#include <stddef.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_GENERATED_SCALAR_T
#endif
#endif
#if defined(__has_include)
#if __has_include("sfem_macros.hpp")
#include "sfem_macros.hpp"
#endif
#endif
#ifndef SFEM_GENERATED_SCALAR_T
#define SFEM_GENERATED_SCALAR_T
typedef double real_t;
typedef ptrdiff_t idx_t;
typedef ptrdiff_t element_idx_t;
typedef ptrdiff_t count_t;
typedef double geom_t;
#endif
#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
#endif
#ifndef RSTR
#define RSTR SFEM_RESTRICT
#endif

#include <math.h>
#include "../../../../cuda/kernel_math.cuh"

namespace sfem {
namespace codegen {

template <typename s_t>
struct neumann_quad4_edgeshell2_boundary_residual_soa_reference_data {
  static constexpr int NS = 2;
  static constexpr int NQ = 2;

  static __host__ __device__ __forceinline__ const s_t *shape() {
    static const s_t data[4] = {
      s_t(0.78867513459481287),
      s_t(0.21132486540518708),
      s_t(0.21132486540518713),
      s_t(0.78867513459481287)
    };
    return data;
  }

  static __host__ __device__ __forceinline__ const s_t *grad() {
    static const s_t data[4] = {
      s_t(-1),
      s_t(1),
      s_t(-1),
      s_t(1)
    };
    return data;
  }

  static __host__ __device__ __forceinline__ const s_t *weight() {
    static const s_t data[2] = {
      s_t(0.5),
      s_t(0.5)
    };
    return data;
  }
};

template <typename s_t>
__host__ __device__ __forceinline__ s_t neumann_quad4_edgeshell2_boundary_residual_soa_measure(
    const int q,
    const idx_t *const RSTR ev,
    const geom_t *const *const RSTR points) {
  const s_t *const grad = neumann_quad4_edgeshell2_boundary_residual_soa_reference_data<s_t>::grad();
  const int n_shape = neumann_quad4_edgeshell2_boundary_residual_soa_reference_data<s_t>::NS;
    s_t dx0 = s_t(0);
  s_t dx1 = s_t(0);
  for (int i = 0; i < n_shape; ++i) {
    const s_t gi = grad[q * n_shape + i];
    const idx_t node = ev[i];
    dx0 += s_t(points[0][node]) * gi;
    dx1 += s_t(points[1][node]) * gi;
  }
  return sqrt(dx0 * dx0 + dx1 * dx1);
}

__host__ __device__ __forceinline__ const int *neumann_quad4_edgeshell2_boundary_residual_soa_side_nodes() {
  static const int data[8] = {
    0,
    1,
    1,
    2,
    2,
    3,
    3,
    0
  };
  return data;
}

__host__ __device__ __forceinline__ void neumann_quad4_edgeshell2_boundary_residual_soa_gather_sideset_element(
    const element_idx_t parent_element,
    const int side,
    idx_t **const RSTR elements,
    idx_t *const RSTR ev) {
  const int *const RSTR side_nodes = neumann_quad4_edgeshell2_boundary_residual_soa_side_nodes();
  constexpr int n_shape = 2;
  for (int i = 0; i < n_shape; ++i) {
    ev[i] = elements[side_nodes[side * n_shape + i]][parent_element];
  }
}

template <typename s_t>
__host__ __device__ __forceinline__ void neumann_quad4_edgeshell2_boundary_residual_soa_element(
    const idx_t *const RSTR ev,
    const geom_t *const *const RSTR points, const s_t t0, const s_t t1,
    s_t element_vector[2][2]) {
  const s_t *const shape = neumann_quad4_edgeshell2_boundary_residual_soa_reference_data<s_t>::shape();
  const s_t *const weight = neumann_quad4_edgeshell2_boundary_residual_soa_reference_data<s_t>::weight();
  const int n_shape = neumann_quad4_edgeshell2_boundary_residual_soa_reference_data<s_t>::NS;
  const int n_qp = neumann_quad4_edgeshell2_boundary_residual_soa_reference_data<s_t>::NQ;

    const s_t coeff0 = -t0;
    const s_t coeff1 = -t1;

  for (int q = 0; q < n_qp; ++q) {
    const s_t dS = neumann_quad4_edgeshell2_boundary_residual_soa_measure<s_t>(q, ev, points);
    const s_t qw = weight[q] * dS;


    for (int i = 0; i < n_shape; ++i) {
      const s_t test = shape[q * n_shape + i] * qw;
        element_vector[0][i] += coeff0 * test;
        element_vector[1][i] += coeff1 * test;
    }
  }
}

template <typename s_t>
__host__ __device__ __forceinline__ void neumann_quad4_edgeshell2_boundary_residual_soa_scatter_element(
    const idx_t *const RSTR ev,
    const s_t element_vector[2][2],
    const int out_stride,
    s_t *const RSTR out0,
    s_t *const RSTR out1) {
  constexpr int n_shape = 2;
  for (int i = 0; i < n_shape; ++i) {
    const idx_t node = ev[i];
            atomicAdd(&(out0[node * out_stride]), element_vector[0][i]);
            atomicAdd(&(out1[node * out_stride]), element_vector[1][i]);
  }
}

template <typename s_t>
__global__ void neumann_quad4_edgeshell2_boundary_residual_soa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points, const s_t t0, const s_t t1,
    const int out_stride,
    s_t *const RSTR out0,
    s_t *const RSTR out1) {
  for (ptrdiff_t e = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += (ptrdiff_t)blockDim.x * gridDim.x) {
    idx_t ev[2];
    s_t element_vector[2][2];
    for (int i = 0; i < 2; ++i) {
      ev[i] = elements[i][e];
    }
    for (int c = 0; c < 2; ++c) {
      for (int i = 0; i < 2; ++i) {
        element_vector[c][i] = s_t(0);
      }
    }
    neumann_quad4_edgeshell2_boundary_residual_soa_element<s_t>(ev, points, t0, t1, element_vector);
    neumann_quad4_edgeshell2_boundary_residual_soa_scatter_element<s_t>(ev, element_vector, out_stride, out0, out1);
  }


}

template <typename s_t>
__global__ void neumann_quad4_edgeshell2_boundary_residual_ss_soa_impl(
    const ptrdiff_t nsides,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const s_t t0, const s_t t1,
    const int out_stride,
    s_t *const RSTR out0,
    s_t *const RSTR out1) {
  for (ptrdiff_t s = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; s < nsides; s += (ptrdiff_t)blockDim.x * gridDim.x) {
    idx_t ev[2];
    s_t element_vector[2][2];
    neumann_quad4_edgeshell2_boundary_residual_soa_gather_sideset_element(parent[s], side_idx[s], elements, ev);
    for (int c = 0; c < 2; ++c) {
      for (int i = 0; i < 2; ++i) {
        element_vector[c][i] = s_t(0);
      }
    }
    neumann_quad4_edgeshell2_boundary_residual_soa_element<s_t>(ev, points, t0, t1, element_vector);
    neumann_quad4_edgeshell2_boundary_residual_soa_scatter_element<s_t>(ev, element_vector, out_stride, out0, out1);
  }


}

}  // namespace codegen
}  // namespace sfem

extern "C" int cu_neumann_quad4_edgeshell2_boundary_residual_soa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points, const real_t t0, const real_t t1,
    const int out_stride,
    real_t *const RSTR out0,
    real_t *const RSTR out1,
    void *const stream) {
  const int block_size = 256;
  const int grid_size = (int)((nelements + block_size - 1) / block_size);
  sfem::codegen::neumann_quad4_edgeshell2_boundary_residual_soa_impl<real_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
      nelements, nnodes, elements, points, t0, t1, out_stride, out0, out1);
  return SFEM_SUCCESS;
}

extern "C" int cu_neumann_quad4_edgeshell2_boundary_residual_soa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points, const float t0, const float t1,
    const int out_stride,
    float *const RSTR out0,
    float *const RSTR out1,
    void *const stream) {
  const int block_size = 256;
  const int grid_size = (int)((nelements + block_size - 1) / block_size);
  sfem::codegen::neumann_quad4_edgeshell2_boundary_residual_soa_impl<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
      nelements, nnodes, elements, points, t0, t1, out_stride, out0, out1);
  return SFEM_SUCCESS;
}

extern "C" int cu_neumann_quad4_edgeshell2_boundary_residual_ss_soa(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const real_t t0, const real_t t1,
    const int out_stride,
    real_t *const RSTR out0,
    real_t *const RSTR out1,
    void *const stream) {
  const int block_size = 256;
  const int grid_size = (int)((nsides + block_size - 1) / block_size);
  sfem::codegen::neumann_quad4_edgeshell2_boundary_residual_ss_soa_impl<real_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
      nsides, nnodes, elements, parent, side_idx, points, t0, t1, out_stride, out0, out1);
  return SFEM_SUCCESS;
}

extern "C" int cu_neumann_quad4_edgeshell2_boundary_residual_ss_soa_float(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const float t0, const float t1,
    const int out_stride,
    float *const RSTR out0,
    float *const RSTR out1,
    void *const stream) {
  const int block_size = 256;
  const int grid_size = (int)((nsides + block_size - 1) / block_size);
  sfem::codegen::neumann_quad4_edgeshell2_boundary_residual_ss_soa_impl<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
      nsides, nnodes, elements, parent, side_idx, points, t0, t1, out_stride, out0, out1);
  return SFEM_SUCCESS;
}
