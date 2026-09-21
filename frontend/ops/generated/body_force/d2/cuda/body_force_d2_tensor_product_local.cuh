#ifndef BODY_FORCE_D2_TENSOR_PRODUCT_LOCAL_HPP
#define BODY_FORCE_D2_TENSOR_PRODUCT_LOCAL_HPP

#include <math.h>
#include <stddef.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_GENERATED_SCALAR_T
#endif
#endif
#include "../../../cuda/kernel_math.cuh"
#include "../../../cuda/tensor_product_kernels.cuh"

#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT
#endif
#ifndef RSTR
#define RSTR SFEM_RESTRICT
#endif
#ifndef SFEM_GENERATED_SCALAR_T
#define SFEM_GENERATED_SCALAR_T
typedef double real_t;
typedef ptrdiff_t idx_t;
typedef ptrdiff_t element_idx_t;
typedef ptrdiff_t count_t;
typedef double geom_t;
#endif

namespace sfem {
namespace codegen {

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void body_force_d2_tensor_product_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR q_weight_1d,
    const s_t density,
    const s_t g0,
    const s_t g1,
    s_t *const RSTR output[2 * NS]
) {
  static constexpr int ND = 2;
  static constexpr int NC = 2;
  s_t value_coeff[NC * NQ * VS];
  static constexpr int NQ1 = integer_root(NQ, ND);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = q / NQ1;
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
    const s_t *const RSTR det_q = determinant + q * geometry_stride;
    s_t *const RSTR value_coeff_q0 = &value_coeff[q * VS];
    s_t *const RSTR value_coeff_q1 = &value_coeff[(NQ + q) * VS];
    {
      const s_t det = det_q[0];
      const s_t value_coeff0 = -density*g0;
      const s_t value_coeff1 = -density*g1;
      value_coeff_q0[0] = qw * det * value_coeff0;
      value_coeff_q1[0] = qw * det * value_coeff1;
    }
  }
  tensor_integrate_value<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, value_coeff, output);
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void body_force_d2_tensor_product_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR q_weight_1d,
    const s_t density,
    const s_t g0,
    const s_t g1,
    s_t output[2 * NS][VS]
) {
  static constexpr int ND = 2;
  static constexpr int NC = 2;
  s_t value_coeff[NC * NQ * VS];
  static constexpr int NQ1 = integer_root(NQ, ND);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = q / NQ1;
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
    const s_t *const RSTR det_q = determinant + q * geometry_stride;
    s_t *const RSTR value_coeff_q0 = &value_coeff[q * VS];
    s_t *const RSTR value_coeff_q1 = &value_coeff[(NQ + q) * VS];
    {
      const s_t det = det_q[0];
      const s_t value_coeff0 = -density*g0;
      const s_t value_coeff1 = -density*g1;
      value_coeff_q0[0] = qw * det * value_coeff0;
      value_coeff_q1[0] = qw * det * value_coeff1;
    }
  }
  tensor_integrate_value_contiguous<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, value_coeff, output);
}


} // namespace codegen
} // namespace sfem

#endif
