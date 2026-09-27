#ifndef BODY_FORCE_D2_SIMPLEX_LOCAL_HPP
#define BODY_FORCE_D2_SIMPLEX_LOCAL_HPP

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

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void body_force_d2_simplex_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape,
    const s_t *const RSTR q_weight,
    const s_t density,
    const s_t g0,
    const s_t g1,
    s_t *const RSTR output[2 * NS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t value_coeff0_values;
    s_t value_coeff1_values;
    {
      const s_t value_coeff0 = -density*g0;
      const s_t value_coeff1 = -density*g1;
      value_coeff0_values = value_coeff0;
      value_coeff1_values = value_coeff1;
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_value = shape[q * NS + test];
      s_t *const RSTR output_row0 = output[test * NC];
      s_t *const RSTR output_row1 = output[test * NC + 1];
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        output_row0[0] += q_weight[q] * det * (value_coeff0_values * test_value);
        output_row1[0] += q_weight[q] * det * (value_coeff1_values * test_value);
      }
    }
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void body_force_d2_simplex_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape,
    const s_t *const RSTR q_weight,
    const s_t density,
    const s_t g0,
    const s_t g1,
    s_t output[2 * NS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t value_coeff0_values;
    s_t value_coeff1_values;
    {
      const s_t value_coeff0 = -density*g0;
      const s_t value_coeff1 = -density*g1;
      value_coeff0_values = value_coeff0;
      value_coeff1_values = value_coeff1;
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_value = shape[q * NS + test];
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        output[test * NC] += q_weight[q] * det * (value_coeff0_values * test_value);
        output[test * NC + 1] += q_weight[q] * det * (value_coeff1_values * test_value);
      }
    }
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void body_force_d2_simplex_tri3_residual_block(
    const int ne,
    const s_t *const RSTR determinant,
    const s_t density,
    const s_t g0,
    const s_t g1,
    s_t *const RSTR output[2 * NS]
) {
  {
    const ptrdiff_t goff = 0;
    const s_t det = determinant[goff];
    const s_t value_coeff0 = -density*g0;
    const s_t value_coeff1 = -density*g1;
    output[0][0] += ((s_t(1) / s_t(2))) * (det) * (((s_t(1) / s_t(3))) * (value_coeff0));
    output[1][0] += ((s_t(1) / s_t(2))) * (det) * (((s_t(1) / s_t(3))) * (value_coeff1));
    output[2][0] += ((s_t(1) / s_t(2))) * (det) * (((s_t(1) / s_t(3))) * (value_coeff0));
    output[3][0] += ((s_t(1) / s_t(2))) * (det) * (((s_t(1) / s_t(3))) * (value_coeff1));
    output[4][0] += ((s_t(1) / s_t(2))) * (det) * (((s_t(1) / s_t(3))) * (value_coeff0));
    output[5][0] += ((s_t(1) / s_t(2))) * (det) * (((s_t(1) / s_t(3))) * (value_coeff1));
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void body_force_d2_simplex_tri3_residual_block_contiguous(
    const int ne,
    const s_t *const RSTR determinant,
    const s_t density,
    const s_t g0,
    const s_t g1,
    s_t output[2 * NS]
) {
  {
    const ptrdiff_t goff = 0;
    const s_t det = determinant[goff];
    const s_t value_coeff0 = -density*g0;
    const s_t value_coeff1 = -density*g1;
    output[0] += ((s_t(1) / s_t(2))) * (det) * (((s_t(1) / s_t(3))) * (value_coeff0));
    output[1] += ((s_t(1) / s_t(2))) * (det) * (((s_t(1) / s_t(3))) * (value_coeff1));
    output[2] += ((s_t(1) / s_t(2))) * (det) * (((s_t(1) / s_t(3))) * (value_coeff0));
    output[3] += ((s_t(1) / s_t(2))) * (det) * (((s_t(1) / s_t(3))) * (value_coeff1));
    output[4] += ((s_t(1) / s_t(2))) * (det) * (((s_t(1) / s_t(3))) * (value_coeff0));
    output[5] += ((s_t(1) / s_t(2))) * (det) * (((s_t(1) / s_t(3))) * (value_coeff1));
  }
}


} // namespace codegen
} // namespace sfem

#endif
