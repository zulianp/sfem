#ifndef BODY_FORCE_D3_SIMPLEX_LOCAL_HPP
#define BODY_FORCE_D3_SIMPLEX_LOCAL_HPP

#include <math.h>
#include <stddef.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_GENERATED_SCALAR_T
#endif
#endif
#include "../../kernel_math.hpp"
#include "../../tensor_product_kernels.hpp"

#ifndef SFEM_INLINE
#define SFEM_INLINE inline
#endif
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
static SFEM_INLINE void body_force_d3_simplex_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape,
    const s_t *const RSTR q_weight,
    const s_t density,
    const s_t g0,
    const s_t g1,
    const s_t g2,
    s_t *const RSTR output[3 * NS]
) {
  static constexpr int NC = 3;
  for (int q = 0; q < NQ; ++q) {
    s_t value_coeff0_values[VS];
    s_t value_coeff1_values[VS];
    s_t value_coeff2_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t value_coeff0 = -density*g0;
      const s_t value_coeff1 = -density*g1;
      const s_t value_coeff2 = -density*g2;
      value_coeff0_values[lane] = value_coeff0;
      value_coeff1_values[lane] = value_coeff1;
      value_coeff2_values[lane] = value_coeff2;
    }
    for (int test = 0; test < NS; ++test) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        const s_t test_value = shape[q * NS + test];
        output[test * NC][lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value);
        output[test * NC + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value);
        output[test * NC + 2][lane] += q_weight[q] * det * (value_coeff2_values[lane] * test_value);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void body_force_d3_simplex_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape,
    const s_t *const RSTR q_weight,
    const s_t density,
    const s_t g0,
    const s_t g1,
    const s_t g2,
    s_t output[3 * NS][VS]
) {
  static constexpr int NC = 3;
  for (int q = 0; q < NQ; ++q) {
    s_t value_coeff0_values[VS];
    s_t value_coeff1_values[VS];
    s_t value_coeff2_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t value_coeff0 = -density*g0;
      const s_t value_coeff1 = -density*g1;
      const s_t value_coeff2 = -density*g2;
      value_coeff0_values[lane] = value_coeff0;
      value_coeff1_values[lane] = value_coeff1;
      value_coeff2_values[lane] = value_coeff2;
    }
    for (int test = 0; test < NS; ++test) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        const s_t test_value = shape[q * NS + test];
        output[test * NC][lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value);
        output[test * NC + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value);
        output[test * NC + 2][lane] += q_weight[q] * det * (value_coeff2_values[lane] * test_value);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void body_force_d3_simplex_tet4_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape,
    const s_t *const RSTR q_weight,
    const s_t density,
    const s_t g0,
    const s_t g1,
    const s_t g2,
    s_t *const RSTR output[3 * NS]
) {
  static constexpr int NC = 3;
  for (int q = 0; q < NQ; ++q) {
    s_t value_coeff0_values[VS];
    s_t value_coeff1_values[VS];
    s_t value_coeff2_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t value_coeff0 = -density*g0;
      const s_t value_coeff1 = -density*g1;
      const s_t value_coeff2 = -density*g2;
      value_coeff0_values[lane] = value_coeff0;
      value_coeff1_values[lane] = value_coeff1;
      value_coeff2_values[lane] = value_coeff2;
    }
    for (int test = 0; test < NS; ++test) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        const s_t test_value = shape[q * NS + test];
        output[test * NC][lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value);
        output[test * NC + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value);
        output[test * NC + 2][lane] += q_weight[q] * det * (value_coeff2_values[lane] * test_value);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void body_force_d3_simplex_tet4_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape,
    const s_t *const RSTR q_weight,
    const s_t density,
    const s_t g0,
    const s_t g1,
    const s_t g2,
    s_t output[3 * NS][VS]
) {
  static constexpr int NC = 3;
  for (int q = 0; q < NQ; ++q) {
    s_t value_coeff0_values[VS];
    s_t value_coeff1_values[VS];
    s_t value_coeff2_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t value_coeff0 = -density*g0;
      const s_t value_coeff1 = -density*g1;
      const s_t value_coeff2 = -density*g2;
      value_coeff0_values[lane] = value_coeff0;
      value_coeff1_values[lane] = value_coeff1;
      value_coeff2_values[lane] = value_coeff2;
    }
    for (int test = 0; test < NS; ++test) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        const s_t test_value = shape[q * NS + test];
        output[test * NC][lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value);
        output[test * NC + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value);
        output[test * NC + 2][lane] += q_weight[q] * det * (value_coeff2_values[lane] * test_value);
      }
    }
  }
}


} // namespace codegen
} // namespace sfem

#endif
