#ifndef VECTOR_MASS_D3_TENSOR_PRODUCT_LOCAL_HPP
#define VECTOR_MASS_D3_TENSOR_PRODUCT_LOCAL_HPP

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
static SFEM_INLINE void vector_mass_d3_tensor_product_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR q_weight_1d,
    const s_t *const RSTR current[3 * NS],
    const s_t density,
    s_t *const RSTR output[3 * NS]
) {
  static constexpr int ND = 3;
  static constexpr int NC = 3;
  s_t current_value[NC * NQ * VS];
  tensor_evaluate_value<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, current, current_value);
  s_t value_coeff[NC * NQ * VS];
  static constexpr int NQ1 = integer_root(NQ, ND);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
    const s_t *const RSTR det_q = determinant + q * geometry_stride;
    const s_t *const RSTR current_value_q0 = &current_value[q * VS];
    const s_t *const RSTR current_value_q1 = &current_value[(NQ + q) * VS];
    const s_t *const RSTR current_value_q2 = &current_value[(2 * NQ + q) * VS];
    s_t *const RSTR value_coeff_q0 = &value_coeff[q * VS];
    s_t *const RSTR value_coeff_q1 = &value_coeff[(NQ + q) * VS];
    s_t *const RSTR value_coeff_q2 = &value_coeff[(2 * NQ + q) * VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t det = det_q[lane];
      const s_t u0 = current_value_q0[lane];
      const s_t u1 = current_value_q1[lane];
      const s_t u2 = current_value_q2[lane];
      const s_t value_coeff0 = density*u0;
      const s_t value_coeff1 = density*u1;
      const s_t value_coeff2 = density*u2;
      value_coeff_q0[lane] = qw * det * value_coeff0;
      value_coeff_q1[lane] = qw * det * value_coeff1;
      value_coeff_q2[lane] = qw * det * value_coeff2;
    }
  }
  tensor_integrate_value<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, value_coeff, output);
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void vector_mass_d3_tensor_product_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR q_weight_1d,
    const s_t current[3 * NS][VS],
    const s_t density,
    s_t output[3 * NS][VS]
) {
  static constexpr int ND = 3;
  static constexpr int NC = 3;
  s_t current_value[NC * NQ * VS];
  tensor_evaluate_value_contiguous<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, current, current_value);
  s_t value_coeff[NC * NQ * VS];
  static constexpr int NQ1 = integer_root(NQ, ND);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
    const s_t *const RSTR det_q = determinant + q * geometry_stride;
    const s_t *const RSTR current_value_q0 = &current_value[q * VS];
    const s_t *const RSTR current_value_q1 = &current_value[(NQ + q) * VS];
    const s_t *const RSTR current_value_q2 = &current_value[(2 * NQ + q) * VS];
    s_t *const RSTR value_coeff_q0 = &value_coeff[q * VS];
    s_t *const RSTR value_coeff_q1 = &value_coeff[(NQ + q) * VS];
    s_t *const RSTR value_coeff_q2 = &value_coeff[(2 * NQ + q) * VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t det = det_q[lane];
      const s_t u0 = current_value_q0[lane];
      const s_t u1 = current_value_q1[lane];
      const s_t u2 = current_value_q2[lane];
      const s_t value_coeff0 = density*u0;
      const s_t value_coeff1 = density*u1;
      const s_t value_coeff2 = density*u2;
      value_coeff_q0[lane] = qw * det * value_coeff0;
      value_coeff_q1[lane] = qw * det * value_coeff1;
      value_coeff_q2[lane] = qw * det * value_coeff2;
    }
  }
  tensor_integrate_value_contiguous<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, value_coeff, output);
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void vector_mass_d3_tensor_product_jacobian_action_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR q_weight_1d,
    const s_t *const RSTR direction[3 * NS],
    const s_t density,
    s_t *const RSTR output[3 * NS]
) {
  static constexpr int ND = 3;
  static constexpr int NC = 3;
  s_t direction_value[NC * NQ * VS];
  tensor_evaluate_value<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, direction, direction_value);
  s_t value_coeff[NC * NQ * VS];
  static constexpr int NQ1 = integer_root(NQ, ND);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
    const s_t *const RSTR det_q = determinant + q * geometry_stride;
    const s_t *const RSTR direction_value_q0 = &direction_value[q * VS];
    const s_t *const RSTR direction_value_q1 = &direction_value[(NQ + q) * VS];
    const s_t *const RSTR direction_value_q2 = &direction_value[(2 * NQ + q) * VS];
    s_t *const RSTR value_coeff_q0 = &value_coeff[q * VS];
    s_t *const RSTR value_coeff_q1 = &value_coeff[(NQ + q) * VS];
    s_t *const RSTR value_coeff_q2 = &value_coeff[(2 * NQ + q) * VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t det = det_q[lane];
      const s_t u0_direction = direction_value_q0[lane];
      const s_t u1_direction = direction_value_q1[lane];
      const s_t u2_direction = direction_value_q2[lane];
      const s_t value_coeff0 = density*u0_direction;
      const s_t value_coeff1 = density*u1_direction;
      const s_t value_coeff2 = density*u2_direction;
      value_coeff_q0[lane] = qw * det * value_coeff0;
      value_coeff_q1[lane] = qw * det * value_coeff1;
      value_coeff_q2[lane] = qw * det * value_coeff2;
    }
  }
  tensor_integrate_value<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, value_coeff, output);
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void vector_mass_d3_tensor_product_jacobian_action_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR q_weight_1d,
    const s_t direction[3 * NS][VS],
    const s_t density,
    s_t output[3 * NS][VS]
) {
  static constexpr int ND = 3;
  static constexpr int NC = 3;
  s_t direction_value[NC * NQ * VS];
  tensor_evaluate_value_contiguous<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, direction, direction_value);
  s_t value_coeff[NC * NQ * VS];
  static constexpr int NQ1 = integer_root(NQ, ND);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
    const s_t *const RSTR det_q = determinant + q * geometry_stride;
    const s_t *const RSTR direction_value_q0 = &direction_value[q * VS];
    const s_t *const RSTR direction_value_q1 = &direction_value[(NQ + q) * VS];
    const s_t *const RSTR direction_value_q2 = &direction_value[(2 * NQ + q) * VS];
    s_t *const RSTR value_coeff_q0 = &value_coeff[q * VS];
    s_t *const RSTR value_coeff_q1 = &value_coeff[(NQ + q) * VS];
    s_t *const RSTR value_coeff_q2 = &value_coeff[(2 * NQ + q) * VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t det = det_q[lane];
      const s_t u0_direction = direction_value_q0[lane];
      const s_t u1_direction = direction_value_q1[lane];
      const s_t u2_direction = direction_value_q2[lane];
      const s_t value_coeff0 = density*u0_direction;
      const s_t value_coeff1 = density*u1_direction;
      const s_t value_coeff2 = density*u2_direction;
      value_coeff_q0[lane] = qw * det * value_coeff0;
      value_coeff_q1[lane] = qw * det * value_coeff1;
      value_coeff_q2[lane] = qw * det * value_coeff2;
    }
  }
  tensor_integrate_value_contiguous<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, value_coeff, output);
}

template <typename s_t, int NQ, int NS>
static SFEM_INLINE void vector_mass_d3_tensor_product_hessian_block(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR q_weight_1d,
    const s_t density,
    s_t *const RSTR element_matrix
) {
  static constexpr int ND = 3;
  static constexpr int NC = 3;
  for (int entry = 0; entry < 3 * NS * 3 * NS; ++entry) {
    element_matrix[entry] = s_t(0);
  }
  s_t value_coeff_c0[NC * NQ];
  s_t value_coeff_c1[NC * NQ];
  s_t value_coeff_c2[NC * NQ];
  static constexpr int NQ1 = integer_root(NQ, ND);
  static constexpr int NS1 = integer_root(NS, ND);
  s_t * column[NC * NS];
  static constexpr int N_TANGENT = 9;
  s_t tangent[N_TANGENT * NQ];
  for (int q = 0; q < NQ; ++q) {
    s_t *const RSTR tangent_value_d0_value0_q = &tangent[q];
    s_t *const RSTR tangent_value_d0_value1_q = &tangent[(NQ + q)];
    s_t *const RSTR tangent_value_d0_value2_q = &tangent[(2 * NQ + q)];
    s_t *const RSTR tangent_value_d1_value0_q = &tangent[(3 * NQ + q)];
    s_t *const RSTR tangent_value_d1_value1_q = &tangent[(4 * NQ + q)];
    s_t *const RSTR tangent_value_d1_value2_q = &tangent[(5 * NQ + q)];
    s_t *const RSTR tangent_value_d2_value0_q = &tangent[(6 * NQ + q)];
    s_t *const RSTR tangent_value_d2_value1_q = &tangent[(7 * NQ + q)];
    s_t *const RSTR tangent_value_d2_value2_q = &tangent[(8 * NQ + q)];
    {
      tangent_value_d0_value0_q[0] = density;
      tangent_value_d0_value1_q[0] = 0;
      tangent_value_d0_value2_q[0] = 0;
      tangent_value_d1_value0_q[0] = 0;
      tangent_value_d1_value1_q[0] = density;
      tangent_value_d1_value2_q[0] = 0;
      tangent_value_d2_value0_q[0] = 0;
      tangent_value_d2_value1_q[0] = 0;
      tangent_value_d2_value2_q[0] = density;
    }
  }
  for (int trial = 0; trial < NS; ++trial) {
    const int trial_x = trial % NS1;
    const int trial_y = (trial / NS1) % NS1;
    const int trial_z = trial / (NS1 * NS1);
    for (int q = 0; q < NQ; ++q) {
      const int q_x = q % NQ1;
      const int q_y = (q / NQ1) % NQ1;
      const int q_z = q / (NQ1 * NQ1);
      const s_t qw = q_weight_1d[q_x] * q_weight_1d[q_y] * q_weight_1d[q_z];
      const s_t *const RSTR det_q = determinant + q * geometry_stride;
      const s_t *const RSTR tangent_value_d0_value0_q = &tangent[q];
      const s_t *const RSTR tangent_value_d0_value1_q = &tangent[(NQ + q)];
      const s_t *const RSTR tangent_value_d0_value2_q = &tangent[(2 * NQ + q)];
      s_t *const RSTR value_coeff_q0_c0 = &value_coeff_c0[q];
      s_t *const RSTR value_coeff_q1_c0 = &value_coeff_c0[(NQ + q)];
      s_t *const RSTR value_coeff_q2_c0 = &value_coeff_c0[(2 * NQ + q)];
      const s_t *const RSTR tangent_value_d1_value0_q = &tangent[(3 * NQ + q)];
      const s_t *const RSTR tangent_value_d1_value1_q = &tangent[(4 * NQ + q)];
      const s_t *const RSTR tangent_value_d1_value2_q = &tangent[(5 * NQ + q)];
      s_t *const RSTR value_coeff_q0_c1 = &value_coeff_c1[q];
      s_t *const RSTR value_coeff_q1_c1 = &value_coeff_c1[(NQ + q)];
      s_t *const RSTR value_coeff_q2_c1 = &value_coeff_c1[(2 * NQ + q)];
      const s_t *const RSTR tangent_value_d2_value0_q = &tangent[(6 * NQ + q)];
      const s_t *const RSTR tangent_value_d2_value1_q = &tangent[(7 * NQ + q)];
      const s_t *const RSTR tangent_value_d2_value2_q = &tangent[(8 * NQ + q)];
      s_t *const RSTR value_coeff_q0_c2 = &value_coeff_c2[q];
      s_t *const RSTR value_coeff_q1_c2 = &value_coeff_c2[(NQ + q)];
      s_t *const RSTR value_coeff_q2_c2 = &value_coeff_c2[(2 * NQ + q)];
      {
        const s_t det = det_q[0];
        const s_t trial_value = shape_1d[q_x * NS1 + trial_x] * shape_1d[q_y * NS1 + trial_y] * shape_1d[q_z * NS1 + trial_z];
        const s_t value_coeff0_c0 = trial_value * tangent_value_d0_value0_q[0];
        const s_t value_coeff1_c0 = trial_value * tangent_value_d0_value1_q[0];
        const s_t value_coeff2_c0 = trial_value * tangent_value_d0_value2_q[0];
        value_coeff_q0_c0[0] = qw * det * value_coeff0_c0;
        value_coeff_q1_c0[0] = qw * det * value_coeff1_c0;
        value_coeff_q2_c0[0] = qw * det * value_coeff2_c0;
        const s_t value_coeff0_c1 = trial_value * tangent_value_d1_value0_q[0];
        const s_t value_coeff1_c1 = trial_value * tangent_value_d1_value1_q[0];
        const s_t value_coeff2_c1 = trial_value * tangent_value_d1_value2_q[0];
        value_coeff_q0_c1[0] = qw * det * value_coeff0_c1;
        value_coeff_q1_c1[0] = qw * det * value_coeff1_c1;
        value_coeff_q2_c1[0] = qw * det * value_coeff2_c1;
        const s_t value_coeff0_c2 = trial_value * tangent_value_d2_value0_q[0];
        const s_t value_coeff1_c2 = trial_value * tangent_value_d2_value1_q[0];
        const s_t value_coeff2_c2 = trial_value * tangent_value_d2_value2_q[0];
        value_coeff_q0_c2[0] = qw * det * value_coeff0_c2;
        value_coeff_q1_c2[0] = qw * det * value_coeff1_c2;
        value_coeff_q2_c2[0] = qw * det * value_coeff2_c2;
      }
    }
    for (int out_shape = 0; out_shape < NS; ++out_shape) {
      column[out_shape * NC + 0] = &element_matrix[(0 * NS + out_shape) * 3 * NS + 0 * NS + trial];
      column[out_shape * NC + 1] = &element_matrix[(1 * NS + out_shape) * 3 * NS + 0 * NS + trial];
      column[out_shape * NC + 2] = &element_matrix[(2 * NS + out_shape) * 3 * NS + 0 * NS + trial];
    }
    tensor_integrate_value_scalar<s_t, NQ, NS, ND, NC>(
        shape_1d, value_coeff_c0, column);
    for (int out_shape = 0; out_shape < NS; ++out_shape) {
      column[out_shape * NC + 0] = &element_matrix[(0 * NS + out_shape) * 3 * NS + 1 * NS + trial];
      column[out_shape * NC + 1] = &element_matrix[(1 * NS + out_shape) * 3 * NS + 1 * NS + trial];
      column[out_shape * NC + 2] = &element_matrix[(2 * NS + out_shape) * 3 * NS + 1 * NS + trial];
    }
    tensor_integrate_value_scalar<s_t, NQ, NS, ND, NC>(
        shape_1d, value_coeff_c1, column);
    for (int out_shape = 0; out_shape < NS; ++out_shape) {
      column[out_shape * NC + 0] = &element_matrix[(0 * NS + out_shape) * 3 * NS + 2 * NS + trial];
      column[out_shape * NC + 1] = &element_matrix[(1 * NS + out_shape) * 3 * NS + 2 * NS + trial];
      column[out_shape * NC + 2] = &element_matrix[(2 * NS + out_shape) * 3 * NS + 2 * NS + trial];
    }
    tensor_integrate_value_scalar<s_t, NQ, NS, ND, NC>(
        shape_1d, value_coeff_c2, column);
  }
}

} // namespace codegen
} // namespace sfem

#endif
