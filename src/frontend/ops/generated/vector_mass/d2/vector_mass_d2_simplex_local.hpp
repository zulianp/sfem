#ifndef VECTOR_MASS_D2_SIMPLEX_LOCAL_HPP
#define VECTOR_MASS_D2_SIMPLEX_LOCAL_HPP

#include <math.h>
#include <stddef.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_GENERATED_SCALAR_T
#endif
#endif
#include "kernel_math.hpp"
#include "tensor_product_kernels.hpp"

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
static SFEM_INLINE void vector_mass_d2_simplex_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[2 * NS],
    const s_t density,
    s_t *const RSTR output[2 * NS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_values[VS];
    s_t u1_values[VS];
    s_t value_coeff0_values[VS];
    s_t value_coeff1_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u0_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = current[trial * NC];
      const s_t trial_shape = shape[q * NS + trial];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        u0_values[lane] += coeff_stream[lane] * trial_shape;
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u1_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = current[trial * NC + 1];
      const s_t trial_shape = shape[q * NS + trial];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        u1_values[lane] += coeff_stream[lane] * trial_shape;
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t u0 = u0_values[lane];
      const s_t u1 = u1_values[lane];
      value_coeff0_values[lane] = density*u0;
      value_coeff1_values[lane] = density*u1;
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_value = shape[q * NS + test];
      s_t *const RSTR output_row0 = output[test * NC];
      s_t *const RSTR output_row1 = output[test * NC + 1];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        output_row0[lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value);
        output_row1[lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void vector_mass_d2_simplex_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape,
    const s_t *const RSTR q_weight,
    const s_t current[2 * NS][VS],
    const s_t density,
    s_t output[2 * NS][VS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_values[VS];
    s_t u1_values[VS];
    s_t value_coeff0_values[VS];
    s_t value_coeff1_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u0_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = current[trial * NC];
      const s_t trial_shape = shape[q * NS + trial];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        u0_values[lane] += coeff_stream[lane] * trial_shape;
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u1_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = current[trial * NC + 1];
      const s_t trial_shape = shape[q * NS + trial];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        u1_values[lane] += coeff_stream[lane] * trial_shape;
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t u0 = u0_values[lane];
      const s_t u1 = u1_values[lane];
      value_coeff0_values[lane] = density*u0;
      value_coeff1_values[lane] = density*u1;
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_value = shape[q * NS + test];
      s_t *const RSTR output_row0 = output[test * NC];
      s_t *const RSTR output_row1 = output[test * NC + 1];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        output_row0[lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value);
        output_row1[lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void vector_mass_d2_simplex_tri3_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[2 * NS],
    const s_t density,
    s_t *const RSTR output[2 * NS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_values[VS];
    s_t u1_values[VS];
    s_t value_coeff0_values[VS];
    s_t value_coeff1_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u0_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = current[trial * NC];
      const s_t trial_shape = shape[q * NS + trial];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        u0_values[lane] += coeff_stream[lane] * trial_shape;
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u1_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = current[trial * NC + 1];
      const s_t trial_shape = shape[q * NS + trial];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        u1_values[lane] += coeff_stream[lane] * trial_shape;
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t u0 = u0_values[lane];
      const s_t u1 = u1_values[lane];
      value_coeff0_values[lane] = density*u0;
      value_coeff1_values[lane] = density*u1;
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_value = shape[q * NS + test];
      s_t *const RSTR output_row0 = output[test * NC];
      s_t *const RSTR output_row1 = output[test * NC + 1];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        output_row0[lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value);
        output_row1[lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void vector_mass_d2_simplex_tri3_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape,
    const s_t *const RSTR q_weight,
    const s_t current[2 * NS][VS],
    const s_t density,
    s_t output[2 * NS][VS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_values[VS];
    s_t u1_values[VS];
    s_t value_coeff0_values[VS];
    s_t value_coeff1_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u0_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = current[trial * NC];
      const s_t trial_shape = shape[q * NS + trial];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        u0_values[lane] += coeff_stream[lane] * trial_shape;
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u1_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = current[trial * NC + 1];
      const s_t trial_shape = shape[q * NS + trial];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        u1_values[lane] += coeff_stream[lane] * trial_shape;
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t u0 = u0_values[lane];
      const s_t u1 = u1_values[lane];
      value_coeff0_values[lane] = density*u0;
      value_coeff1_values[lane] = density*u1;
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_value = shape[q * NS + test];
      s_t *const RSTR output_row0 = output[test * NC];
      s_t *const RSTR output_row1 = output[test * NC + 1];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        output_row0[lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value);
        output_row1[lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void vector_mass_d2_simplex_jacobian_action_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR direction[2 * NS],
    const s_t density,
    s_t *const RSTR output[2 * NS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_direction_values[VS];
    s_t u1_direction_values[VS];
    s_t value_coeff0_values[VS];
    s_t value_coeff1_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u0_direction_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = direction[trial * NC];
      const s_t trial_shape = shape[q * NS + trial];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        u0_direction_values[lane] += coeff_stream[lane] * trial_shape;
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u1_direction_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = direction[trial * NC + 1];
      const s_t trial_shape = shape[q * NS + trial];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        u1_direction_values[lane] += coeff_stream[lane] * trial_shape;
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t u0_direction = u0_direction_values[lane];
      const s_t u1_direction = u1_direction_values[lane];
      value_coeff0_values[lane] = density*u0_direction;
      value_coeff1_values[lane] = density*u1_direction;
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_value = shape[q * NS + test];
      s_t *const RSTR output_row0 = output[test * NC];
      s_t *const RSTR output_row1 = output[test * NC + 1];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        output_row0[lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value);
        output_row1[lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void vector_mass_d2_simplex_jacobian_action_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape,
    const s_t *const RSTR q_weight,
    const s_t direction[2 * NS][VS],
    const s_t density,
    s_t output[2 * NS][VS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_direction_values[VS];
    s_t u1_direction_values[VS];
    s_t value_coeff0_values[VS];
    s_t value_coeff1_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u0_direction_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = direction[trial * NC];
      const s_t trial_shape = shape[q * NS + trial];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        u0_direction_values[lane] += coeff_stream[lane] * trial_shape;
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u1_direction_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = direction[trial * NC + 1];
      const s_t trial_shape = shape[q * NS + trial];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        u1_direction_values[lane] += coeff_stream[lane] * trial_shape;
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t u0_direction = u0_direction_values[lane];
      const s_t u1_direction = u1_direction_values[lane];
      value_coeff0_values[lane] = density*u0_direction;
      value_coeff1_values[lane] = density*u1_direction;
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_value = shape[q * NS + test];
      s_t *const RSTR output_row0 = output[test * NC];
      s_t *const RSTR output_row1 = output[test * NC + 1];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        output_row0[lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value);
        output_row1[lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void vector_mass_d2_simplex_tri3_jacobian_action_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR direction[2 * NS],
    const s_t density,
    s_t *const RSTR output[2 * NS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_direction_values[VS];
    s_t u1_direction_values[VS];
    s_t value_coeff0_values[VS];
    s_t value_coeff1_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u0_direction_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = direction[trial * NC];
      const s_t trial_shape = shape[q * NS + trial];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        u0_direction_values[lane] += coeff_stream[lane] * trial_shape;
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u1_direction_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = direction[trial * NC + 1];
      const s_t trial_shape = shape[q * NS + trial];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        u1_direction_values[lane] += coeff_stream[lane] * trial_shape;
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t u0_direction = u0_direction_values[lane];
      const s_t u1_direction = u1_direction_values[lane];
      value_coeff0_values[lane] = density*u0_direction;
      value_coeff1_values[lane] = density*u1_direction;
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_value = shape[q * NS + test];
      s_t *const RSTR output_row0 = output[test * NC];
      s_t *const RSTR output_row1 = output[test * NC + 1];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        output_row0[lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value);
        output_row1[lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void vector_mass_d2_simplex_tri3_jacobian_action_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape,
    const s_t *const RSTR q_weight,
    const s_t direction[2 * NS][VS],
    const s_t density,
    s_t output[2 * NS][VS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_direction_values[VS];
    s_t u1_direction_values[VS];
    s_t value_coeff0_values[VS];
    s_t value_coeff1_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u0_direction_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = direction[trial * NC];
      const s_t trial_shape = shape[q * NS + trial];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        u0_direction_values[lane] += coeff_stream[lane] * trial_shape;
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u1_direction_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = direction[trial * NC + 1];
      const s_t trial_shape = shape[q * NS + trial];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        u1_direction_values[lane] += coeff_stream[lane] * trial_shape;
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t u0_direction = u0_direction_values[lane];
      const s_t u1_direction = u1_direction_values[lane];
      value_coeff0_values[lane] = density*u0_direction;
      value_coeff1_values[lane] = density*u1_direction;
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_value = shape[q * NS + test];
      s_t *const RSTR output_row0 = output[test * NC];
      s_t *const RSTR output_row1 = output[test * NC + 1];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        output_row0[lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value);
        output_row1[lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value);
      }
    }
  }
}

template <typename s_t, int NQ, int NS>
static SFEM_INLINE void vector_mass_d2_simplex_hessian_block(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape,
    const s_t *const RSTR q_weight,
    const s_t density,
    s_t *const RSTR element_matrix
) {
  for (int entry = 0; entry < 2 * NS * 2 * NS; ++entry) {
    element_matrix[entry] = s_t(0);
  }
  s_t value_coeff0_values;
  s_t value_coeff1_values;
  s_t tangent[4];
  for (int q = 0; q < NQ; ++q) {
    {
      tangent[0] = density;
      tangent[1] = 0;
      tangent[2] = 0;
      tangent[3] = density;
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t trial_value = shape[q * NS + trial];
        value_coeff0_values = trial_value * tangent[0];
        value_coeff1_values = trial_value * tangent[1];
      }
      for (int test = 0; test < NS; ++test) {
        {
          const ptrdiff_t goff = q * geometry_stride;
          const s_t det = determinant[goff];
          const s_t test_value = shape[q * NS + test];
          element_matrix[(0 * NS + test) * 2 * NS + 0 * NS + trial] += q_weight[q] * det * (value_coeff0_values * test_value);
          element_matrix[(1 * NS + test) * 2 * NS + 0 * NS + trial] += q_weight[q] * det * (value_coeff1_values * test_value);
        }
      }
      {
        const s_t trial_value = shape[q * NS + trial];
        value_coeff0_values = trial_value * tangent[2];
        value_coeff1_values = trial_value * tangent[3];
      }
      for (int test = 0; test < NS; ++test) {
        {
          const ptrdiff_t goff = q * geometry_stride;
          const s_t det = determinant[goff];
          const s_t test_value = shape[q * NS + test];
          element_matrix[(0 * NS + test) * 2 * NS + 1 * NS + trial] += q_weight[q] * det * (value_coeff0_values * test_value);
          element_matrix[(1 * NS + test) * 2 * NS + 1 * NS + trial] += q_weight[q] * det * (value_coeff1_values * test_value);
        }
      }
    }
  }
}

} // namespace codegen
} // namespace sfem

#endif
