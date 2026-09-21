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
#include "../../../../cuda/kernel_diagnostics.cuh"

namespace sfem {
namespace codegen {

template <typename s_t>
struct neumann_general_tet10_trishell6_boundary_residual_soa_reference_data {
  static constexpr int NS = 6;
  static constexpr int NQ = 6;

  static __host__ __device__ __forceinline__ const s_t *shape() {
    static const s_t data[36] = {
      s_t(-0.084730493093977982),
      s_t(-0.04820837781551205),
      s_t(-0.04820837781551205),
      s_t(0.1928335112620482),
      s_t(0.79548022620090564),
      s_t(0.1928335112620482),
      s_t(-0.04820837781551205),
      s_t(-0.084730493093977968),
      s_t(-0.04820837781551205),
      s_t(0.19283351126204817),
      s_t(0.19283351126204817),
      s_t(0.79548022620090564),
      s_t(-0.04820837781551205),
      s_t(-0.04820837781551205),
      s_t(-0.084730493093977968),
      s_t(0.79548022620090564),
      s_t(0.19283351126204817),
      s_t(0.19283351126204817),
      s_t(0.5176323419876725),
      s_t(-0.074803807748196505),
      s_t(-0.074803807748196505),
      s_t(0.29921523099278602),
      s_t(0.03354481152314847),
      s_t(0.29921523099278602),
      s_t(-0.074803807748196505),
      s_t(0.5176323419876725),
      s_t(-0.074803807748196505),
      s_t(0.29921523099278602),
      s_t(0.29921523099278602),
      s_t(0.03354481152314847),
      s_t(-0.074803807748196505),
      s_t(-0.074803807748196505),
      s_t(0.5176323419876725),
      s_t(0.03354481152314847),
      s_t(0.29921523099278602),
      s_t(0.29921523099278602)
    };
    return data;
  }

  static __host__ __device__ __forceinline__ const s_t *grad() {
    static const s_t data[72] = {
      s_t(0.56758792732771912),
      s_t(0.56758792732771912),
      s_t(0.78379396366385956),
      s_t(0),
      s_t(0),
      s_t(0.78379396366385956),
      s_t(-1.3513818909915787),
      s_t(-1.7837939636638596),
      s_t(1.7837939636638596),
      s_t(1.7837939636638596),
      s_t(-1.7837939636638596),
      s_t(-1.3513818909915787),
      s_t(-0.78379396366385956),
      s_t(-0.78379396366385956),
      s_t(-0.56758792732771912),
      s_t(0),
      s_t(0),
      s_t(0.78379396366385956),
      s_t(1.3513818909915787),
      s_t(-0.43241207267228082),
      s_t(1.7837939636638596),
      s_t(0.43241207267228082),
      s_t(-1.7837939636638596),
      s_t(0),
      s_t(-0.78379396366385956),
      s_t(-0.78379396366385956),
      s_t(0.78379396366385956),
      s_t(0),
      s_t(0),
      s_t(-0.56758792732771912),
      s_t(5.5511151231257827e-17),
      s_t(-1.7837939636638596),
      s_t(0.43241207267228082),
      s_t(1.7837939636638596),
      s_t(-0.43241207267228082),
      s_t(1.3513818909915787),
      s_t(-2.2673902919218341),
      s_t(-2.2673902919218341),
      s_t(-0.63369514596091703),
      s_t(0),
      s_t(0),
      s_t(-0.63369514596091703),
      s_t(2.9010854378827511),
      s_t(-0.36630485403908297),
      s_t(0.36630485403908297),
      s_t(0.36630485403908297),
      s_t(-0.36630485403908297),
      s_t(2.9010854378827511),
      s_t(0.63369514596091703),
      s_t(0.63369514596091703),
      s_t(2.2673902919218341),
      s_t(0),
      s_t(0),
      s_t(-0.63369514596091703),
      s_t(-2.9010854378827511),
      s_t(-3.2673902919218341),
      s_t(0.36630485403908297),
      s_t(3.2673902919218341),
      s_t(-0.36630485403908297),
      s_t(0),
      s_t(0.63369514596091703),
      s_t(0.63369514596091703),
      s_t(-0.63369514596091703),
      s_t(0),
      s_t(0),
      s_t(2.2673902919218341),
      s_t(0),
      s_t(-0.36630485403908297),
      s_t(3.2673902919218341),
      s_t(0.36630485403908297),
      s_t(-3.2673902919218341),
      s_t(-2.9010854378827511)
    };
    return data;
  }

  static __host__ __device__ __forceinline__ const s_t *weight() {
    static const s_t data[6] = {
      s_t(0.11169079483900569),
      s_t(0.11169079483900569),
      s_t(0.11169079483900569),
      s_t(0.054975871827660998),
      s_t(0.054975871827660998),
      s_t(0.054975871827660998)
    };
    return data;
  }
};

template <typename s_t>
__host__ __device__ __forceinline__ s_t neumann_general_tet10_trishell6_boundary_residual_soa_measure(
    const int q,
    const idx_t *const RSTR ev,
    const geom_t *const *const RSTR points) {
  const s_t *const grad = neumann_general_tet10_trishell6_boundary_residual_soa_reference_data<s_t>::grad();
  const int n_shape = neumann_general_tet10_trishell6_boundary_residual_soa_reference_data<s_t>::NS;
    s_t dxdr0 = s_t(0);
  s_t dxdr1 = s_t(0);
  s_t dxdr2 = s_t(0);
  s_t dxds0 = s_t(0);
  s_t dxds1 = s_t(0);
  s_t dxds2 = s_t(0);
  for (int i = 0; i < n_shape; ++i) {
    const s_t gr = grad[(q * n_shape + i) * 2 + 0];
    const s_t gs = grad[(q * n_shape + i) * 2 + 1];
    const idx_t node = ev[i];
    const s_t x = s_t(points[0][node]);
    const s_t y = s_t(points[1][node]);
    const s_t z = s_t(points[2][node]);
    dxdr0 += x * gr;
    dxdr1 += y * gr;
    dxdr2 += z * gr;
    dxds0 += x * gs;
    dxds1 += y * gs;
    dxds2 += z * gs;
  }
  const s_t c0 = dxdr1 * dxds2 - dxdr2 * dxds1;
  const s_t c1 = dxdr2 * dxds0 - dxdr0 * dxds2;
  const s_t c2 = dxdr0 * dxds1 - dxdr1 * dxds0;
  return sqrt(c0 * c0 + c1 * c1 + c2 * c2);
}

__host__ __device__ __forceinline__ const int *neumann_general_tet10_trishell6_boundary_residual_soa_side_nodes() {
  static const int data[24] = {
    0,
    1,
    3,
    4,
    8,
    7,
    1,
    2,
    3,
    5,
    9,
    8,
    0,
    3,
    2,
    7,
    9,
    6,
    0,
    2,
    1,
    6,
    5,
    4
  };
  return data;
}

__host__ __device__ __forceinline__ void neumann_general_tet10_trishell6_boundary_residual_soa_gather_sideset_element(
    const element_idx_t parent_element,
    const int side,
    idx_t **const RSTR elements,
    idx_t *const RSTR ev) {
  const int *const RSTR side_nodes = neumann_general_tet10_trishell6_boundary_residual_soa_side_nodes();
  constexpr int n_shape = 6;
  for (int i = 0; i < n_shape; ++i) {
    ev[i] = elements[side_nodes[side * n_shape + i]][parent_element];
  }
}

template <typename s_t>
__host__ __device__ __forceinline__ void neumann_general_tet10_trishell6_boundary_residual_soa_element(
    const idx_t *const RSTR ev,
    const geom_t *const *const RSTR points, const s_t t0, const s_t t0_001, const s_t t0_010, const s_t t0_100, const s_t t1, const s_t t1_001, const s_t t1_010, const s_t t1_100, const s_t t2, const s_t t2_001, const s_t t2_010, const s_t t2_100,
    s_t element_vector[3][6]) {
  const s_t *const shape = neumann_general_tet10_trishell6_boundary_residual_soa_reference_data<s_t>::shape();
  const s_t *const weight = neumann_general_tet10_trishell6_boundary_residual_soa_reference_data<s_t>::weight();
  const int n_shape = neumann_general_tet10_trishell6_boundary_residual_soa_reference_data<s_t>::NS;
  const int n_qp = neumann_general_tet10_trishell6_boundary_residual_soa_reference_data<s_t>::NQ;



  for (int q = 0; q < n_qp; ++q) {
    const s_t dS = neumann_general_tet10_trishell6_boundary_residual_soa_measure<s_t>(q, ev, points);
    const s_t qw = weight[q] * dS;
    s_t x0 = s_t(0);
    s_t x1 = s_t(0);
    s_t x2 = s_t(0);
    for (int j = 0; j < n_shape; ++j) {
      const s_t phi = shape[q * n_shape + j];
      const idx_t node = ev[j];
      x0 += s_t(points[0][node]) * phi;
      x1 += s_t(points[1][node]) * phi;
      x2 += s_t(points[2][node]) * phi;
    }
    const s_t coeff0 = t0 + t0_001*x2 + t0_010*x1 + t0_100*x0;
    const s_t coeff1 = t1 + t1_001*x2 + t1_010*x1 + t1_100*x0;
    const s_t coeff2 = t2 + t2_001*x2 + t2_010*x1 + t2_100*x0;

    for (int i = 0; i < n_shape; ++i) {
      const s_t test = shape[q * n_shape + i] * qw;
        element_vector[0][i] += coeff0 * test;
        element_vector[1][i] += coeff1 * test;
        element_vector[2][i] += coeff2 * test;
    }
  }
}

template <typename s_t>
__host__ __device__ __forceinline__ void neumann_general_tet10_trishell6_boundary_residual_soa_scatter_element(
    const idx_t *const RSTR ev,
    const s_t element_vector[3][6],
    const int out_stride,
    s_t *const RSTR out0,
    s_t *const RSTR out1,
    s_t *const RSTR out2) {
  constexpr int n_shape = 6;
  for (int i = 0; i < n_shape; ++i) {
    const idx_t node = ev[i];
            atomicAdd(&(out0[node * out_stride]), element_vector[0][i]);
            atomicAdd(&(out1[node * out_stride]), element_vector[1][i]);
            atomicAdd(&(out2[node * out_stride]), element_vector[2][i]);
  }
}

template <typename s_t>
__global__ void neumann_general_tet10_trishell6_boundary_residual_soa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points, const s_t t0, const s_t t0_001, const s_t t0_010, const s_t t0_100, const s_t t1, const s_t t1_001, const s_t t1_010, const s_t t1_100, const s_t t2, const s_t t2_001, const s_t t2_010, const s_t t2_100,
    const int out_stride,
    s_t *const RSTR out0,
    s_t *const RSTR out1,
    s_t *const RSTR out2) {
  for (ptrdiff_t e = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; e < nelements; e += (ptrdiff_t)blockDim.x * gridDim.x) {
    idx_t ev[6];
    s_t element_vector[3][6];
    for (int i = 0; i < 6; ++i) {
      ev[i] = elements[i][e];
    }
    for (int c = 0; c < 3; ++c) {
      for (int i = 0; i < 6; ++i) {
        element_vector[c][i] = s_t(0);
      }
    }
    neumann_general_tet10_trishell6_boundary_residual_soa_element<s_t>(ev, points, t0, t0_001, t0_010, t0_100, t1, t1_001, t1_010, t1_100, t2, t2_001, t2_010, t2_100, element_vector);
    neumann_general_tet10_trishell6_boundary_residual_soa_scatter_element<s_t>(ev, element_vector, out_stride, out0, out1, out2);
  }


}

template <typename s_t>
__global__ void neumann_general_tet10_trishell6_boundary_residual_ss_soa_impl(
    const ptrdiff_t nsides,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const s_t t0, const s_t t0_001, const s_t t0_010, const s_t t0_100, const s_t t1, const s_t t1_001, const s_t t1_010, const s_t t1_100, const s_t t2, const s_t t2_001, const s_t t2_010, const s_t t2_100,
    const int out_stride,
    s_t *const RSTR out0,
    s_t *const RSTR out1,
    s_t *const RSTR out2) {
  for (ptrdiff_t s = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; s < nsides; s += (ptrdiff_t)blockDim.x * gridDim.x) {
    idx_t ev[6];
    s_t element_vector[3][6];
    neumann_general_tet10_trishell6_boundary_residual_soa_gather_sideset_element(parent[s], side_idx[s], elements, ev);
    for (int c = 0; c < 3; ++c) {
      for (int i = 0; i < 6; ++i) {
        element_vector[c][i] = s_t(0);
      }
    }
    neumann_general_tet10_trishell6_boundary_residual_soa_element<s_t>(ev, points, t0, t0_001, t0_010, t0_100, t1, t1_001, t1_010, t1_100, t2, t2_001, t2_010, t2_100, element_vector);
    neumann_general_tet10_trishell6_boundary_residual_soa_scatter_element<s_t>(ev, element_vector, out_stride, out0, out1, out2);
  }


}

}  // namespace codegen
}  // namespace sfem

extern "C" int cu_neumann_general_tet10_trishell6_boundary_residual_soa(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points, const real_t t0, const real_t t0_001, const real_t t0_010, const real_t t0_100, const real_t t1, const real_t t1_001, const real_t t1_010, const real_t t1_100, const real_t t2, const real_t t2_001, const real_t t2_010, const real_t t2_100,
    const int out_stride,
    real_t *const RSTR out0,
    real_t *const RSTR out1,
    real_t *const RSTR out2,
    void *const stream) {
  const int block_size = 256;
  const int grid_size = (int)((nelements + block_size - 1) / block_size);
  sfem::codegen::neumann_general_tet10_trishell6_boundary_residual_soa_impl<real_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
      nelements, nnodes, elements, points, t0, t0_001, t0_010, t0_100, t1, t1_001, t1_010, t1_100, t2, t2_001, t2_010, t2_100, out_stride, out0, out1, out2);
  return sfem::codegen::launch_status("neumann_general_tet10_trishell6_boundary_residual_soa_impl");
}

extern "C" int cu_neumann_general_tet10_trishell6_boundary_residual_soa_float(
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const *const RSTR points, const float t0, const float t0_001, const float t0_010, const float t0_100, const float t1, const float t1_001, const float t1_010, const float t1_100, const float t2, const float t2_001, const float t2_010, const float t2_100,
    const int out_stride,
    float *const RSTR out0,
    float *const RSTR out1,
    float *const RSTR out2,
    void *const stream) {
  const int block_size = 256;
  const int grid_size = (int)((nelements + block_size - 1) / block_size);
  sfem::codegen::neumann_general_tet10_trishell6_boundary_residual_soa_impl<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
      nelements, nnodes, elements, points, t0, t0_001, t0_010, t0_100, t1, t1_001, t1_010, t1_100, t2, t2_001, t2_010, t2_100, out_stride, out0, out1, out2);
  return sfem::codegen::launch_status("neumann_general_tet10_trishell6_boundary_residual_soa_impl");
}

extern "C" int cu_neumann_general_tet10_trishell6_boundary_residual_ss_soa(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const real_t t0, const real_t t0_001, const real_t t0_010, const real_t t0_100, const real_t t1, const real_t t1_001, const real_t t1_010, const real_t t1_100, const real_t t2, const real_t t2_001, const real_t t2_010, const real_t t2_100,
    const int out_stride,
    real_t *const RSTR out0,
    real_t *const RSTR out1,
    real_t *const RSTR out2,
    void *const stream) {
  const int block_size = 256;
  const int grid_size = (int)((nsides + block_size - 1) / block_size);
  sfem::codegen::neumann_general_tet10_trishell6_boundary_residual_ss_soa_impl<real_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
      nsides, nnodes, elements, parent, side_idx, points, t0, t0_001, t0_010, t0_100, t1, t1_001, t1_010, t1_100, t2, t2_001, t2_010, t2_100, out_stride, out0, out1, out2);
  return sfem::codegen::launch_status("neumann_general_tet10_trishell6_boundary_residual_ss_soa_impl");
}

extern "C" int cu_neumann_general_tet10_trishell6_boundary_residual_ss_soa_float(
    const ptrdiff_t nsides,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const element_idx_t *const RSTR parent,
    const int16_t *const RSTR side_idx,
    const geom_t *const *const RSTR points, const float t0, const float t0_001, const float t0_010, const float t0_100, const float t1, const float t1_001, const float t1_010, const float t1_100, const float t2, const float t2_001, const float t2_010, const float t2_100,
    const int out_stride,
    float *const RSTR out0,
    float *const RSTR out1,
    float *const RSTR out2,
    void *const stream) {
  const int block_size = 256;
  const int grid_size = (int)((nsides + block_size - 1) / block_size);
  sfem::codegen::neumann_general_tet10_trishell6_boundary_residual_ss_soa_impl<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
      nsides, nnodes, elements, parent, side_idx, points, t0, t0_001, t0_010, t0_100, t1, t1_001, t1_010, t1_100, t2, t2_001, t2_010, t2_100, out_stride, out0, out1, out2);
  return sfem::codegen::launch_status("neumann_general_tet10_trishell6_boundary_residual_ss_soa_impl");
}
