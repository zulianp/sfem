#ifndef TWO_PHASE_FLOW_FORM_1_P_C_D3_SIMPLEX_LOCAL_HPP
#define TWO_PHASE_FLOW_FORM_1_P_C_D3_SIMPLEX_LOCAL_HPP

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
__host__ __device__ __forceinline__ void two_phase_flow_form_1_p_c_d3_simplex_residual_block(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[2 * NS],
    const s_t *const RSTR previous[2 * NS],
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t K_4,
    const s_t K_5,
    const s_t K_6,
    const s_t K_7,
    const s_t K_8,
    const s_t M_c,
    const s_t P_r,
    const s_t R,
    const s_t S_res,
    const s_t T,
    const s_t Z,
    const s_t dt,
    const s_t m,
    const s_t mu_c,
    const s_t porosity,
    s_t *const RSTR output[2 * NS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t p_w_values;
    s_t p_w_old_values;
    s_t p_c_values;
    s_t p_c_grad_0_ref_values;
    s_t p_c_grad_1_ref_values;
    s_t p_c_grad_2_ref_values;
    s_t p_c_old_values;
    s_t value_coeff1_values;
    s_t grad_coeff1_0_values;
    s_t grad_coeff1_1_values;
    s_t grad_coeff1_2_values;
    {
      p_w_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = current[trial * NC];
      const s_t trial_shape = shape[q * NS + trial];
      {
        p_w_values += coeff_stream[0] * trial_shape;
      }
    }
    {
      p_w_old_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = previous[trial * NC];
      const s_t trial_shape = shape[q * NS + trial];
      {
        p_w_old_values += coeff_stream[0] * trial_shape;
      }
    }
    {
      p_c_values = s_t(0);
      p_c_grad_0_ref_values = s_t(0);
      p_c_grad_1_ref_values = s_t(0);
      p_c_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = current[trial * NC + 1];
      const s_t trial_shape = shape[q * NS + trial];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        p_c_values += coeff_stream[0] * trial_shape;
        p_c_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        p_c_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
        p_c_grad_2_ref_values += coeff_stream[0] * trial_grad_2_ref;
      }
    }
    {
      p_c_old_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = previous[trial * NC + 1];
      const s_t trial_shape = shape[q * NS + trial];
      {
        p_c_old_values += coeff_stream[0] * trial_shape;
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t adj4 = adjugate[4][goff];
      const s_t adj5 = adjugate[5][goff];
      const s_t adj6 = adjugate[6][goff];
      const s_t adj7 = adjugate[7][goff];
      const s_t adj8 = adjugate[8][goff];
      const s_t p_w = p_w_values;
      const s_t p_w_old = p_w_old_values;
      const s_t p_c = p_c_values;
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values;
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values;
      const s_t p_c_grad_2_ref = p_c_grad_2_ref_values;
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
      const s_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
      const s_t p_c_old = p_c_old_values;
      const s_t residual_tmp0 = S_res + s_t(-1);
      const s_t residual_tmp1 = pow_m1(P_r);
      const s_t residual_tmp2 = (s_t(1) - m)/m;
      const s_t residual_tmp3 = pow(pow(residual_tmp1*(p_c - p_w), m) + s_t(1), residual_tmp2);
      const s_t residual_tmp4 = s_t(1) - S_res;
      const s_t residual_tmp5 = M_c/(R*T*Z);
      const s_t residual_tmp6 = p_c*residual_tmp5*pow(s_t(1) - residual_tmp3, C_ka1)*(pow(residual_tmp3, C_ka2) + s_t(-1))/mu_c;
      value_coeff1_values = -porosity*residual_tmp5*(-p_c*(residual_tmp0*residual_tmp3 + residual_tmp4) + p_c_old*(residual_tmp0*pow(pow(residual_tmp1*(p_c_old - p_w_old), m) + s_t(1), residual_tmp2) + residual_tmp4))/dt;
      grad_coeff1_0_values = residual_tmp6*(-K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2);
      grad_coeff1_1_values = residual_tmp6*(-K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2);
      grad_coeff1_2_values = residual_tmp6*(-K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2);
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
      const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
      const s_t test_grad_ref2 = grad_ref_z[q * NS + test];
      const s_t test_value = shape[q * NS + test];
      s_t *const RSTR output_row1 = output[test * NC + 1];
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        const s_t adj0 = adjugate[0][goff];
        const s_t adj1 = adjugate[1][goff];
        const s_t adj2 = adjugate[2][goff];
        const s_t adj3 = adjugate[3][goff];
        const s_t adj4 = adjugate[4][goff];
        const s_t adj5 = adjugate[5][goff];
        const s_t adj6 = adjugate[6][goff];
        const s_t adj7 = adjugate[7][goff];
        const s_t adj8 = adjugate[8][goff];
        const s_t test_grad0 = (test_grad_ref0 * adj0 + test_grad_ref1 * adj3 + test_grad_ref2 * adj6) / det;
        const s_t test_grad1 = (test_grad_ref0 * adj1 + test_grad_ref1 * adj4 + test_grad_ref2 * adj7) / det;
        const s_t test_grad2 = (test_grad_ref0 * adj2 + test_grad_ref1 * adj5 + test_grad_ref2 * adj8) / det;
        output_row1[0] += q_weight[q] * det * (value_coeff1_values * test_value + grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1 + grad_coeff1_2_values * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void two_phase_flow_form_1_p_c_d3_simplex_residual_block_contiguous(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t current[2 * NS],
    const s_t previous[2 * NS],
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t K_4,
    const s_t K_5,
    const s_t K_6,
    const s_t K_7,
    const s_t K_8,
    const s_t M_c,
    const s_t P_r,
    const s_t R,
    const s_t S_res,
    const s_t T,
    const s_t Z,
    const s_t dt,
    const s_t m,
    const s_t mu_c,
    const s_t porosity,
    s_t output[2 * NS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t p_w_values;
    s_t p_w_old_values;
    s_t p_c_values;
    s_t p_c_grad_0_ref_values;
    s_t p_c_grad_1_ref_values;
    s_t p_c_grad_2_ref_values;
    s_t p_c_old_values;
    s_t value_coeff1_values;
    s_t grad_coeff1_0_values;
    s_t grad_coeff1_1_values;
    s_t grad_coeff1_2_values;
    {
      p_w_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_shape = shape[q * NS + trial];
      {
        p_w_values += current[trial * NC] * trial_shape;
      }
    }
    {
      p_w_old_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_shape = shape[q * NS + trial];
      {
        p_w_old_values += previous[trial * NC] * trial_shape;
      }
    }
    {
      p_c_values = s_t(0);
      p_c_grad_0_ref_values = s_t(0);
      p_c_grad_1_ref_values = s_t(0);
      p_c_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_shape = shape[q * NS + trial];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        p_c_values += current[trial * NC + 1] * trial_shape;
        p_c_grad_0_ref_values += current[trial * NC + 1] * trial_grad_0_ref;
        p_c_grad_1_ref_values += current[trial * NC + 1] * trial_grad_1_ref;
        p_c_grad_2_ref_values += current[trial * NC + 1] * trial_grad_2_ref;
      }
    }
    {
      p_c_old_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_shape = shape[q * NS + trial];
      {
        p_c_old_values += previous[trial * NC + 1] * trial_shape;
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t adj4 = adjugate[4][goff];
      const s_t adj5 = adjugate[5][goff];
      const s_t adj6 = adjugate[6][goff];
      const s_t adj7 = adjugate[7][goff];
      const s_t adj8 = adjugate[8][goff];
      const s_t p_w = p_w_values;
      const s_t p_w_old = p_w_old_values;
      const s_t p_c = p_c_values;
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values;
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values;
      const s_t p_c_grad_2_ref = p_c_grad_2_ref_values;
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
      const s_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
      const s_t p_c_old = p_c_old_values;
      const s_t residual_tmp0 = S_res + s_t(-1);
      const s_t residual_tmp1 = pow_m1(P_r);
      const s_t residual_tmp2 = (s_t(1) - m)/m;
      const s_t residual_tmp3 = pow(pow(residual_tmp1*(p_c - p_w), m) + s_t(1), residual_tmp2);
      const s_t residual_tmp4 = s_t(1) - S_res;
      const s_t residual_tmp5 = M_c/(R*T*Z);
      const s_t residual_tmp6 = p_c*residual_tmp5*pow(s_t(1) - residual_tmp3, C_ka1)*(pow(residual_tmp3, C_ka2) + s_t(-1))/mu_c;
      value_coeff1_values = -porosity*residual_tmp5*(-p_c*(residual_tmp0*residual_tmp3 + residual_tmp4) + p_c_old*(residual_tmp0*pow(pow(residual_tmp1*(p_c_old - p_w_old), m) + s_t(1), residual_tmp2) + residual_tmp4))/dt;
      grad_coeff1_0_values = residual_tmp6*(-K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2);
      grad_coeff1_1_values = residual_tmp6*(-K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2);
      grad_coeff1_2_values = residual_tmp6*(-K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2);
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
      const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
      const s_t test_grad_ref2 = grad_ref_z[q * NS + test];
      const s_t test_value = shape[q * NS + test];
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        const s_t adj0 = adjugate[0][goff];
        const s_t adj1 = adjugate[1][goff];
        const s_t adj2 = adjugate[2][goff];
        const s_t adj3 = adjugate[3][goff];
        const s_t adj4 = adjugate[4][goff];
        const s_t adj5 = adjugate[5][goff];
        const s_t adj6 = adjugate[6][goff];
        const s_t adj7 = adjugate[7][goff];
        const s_t adj8 = adjugate[8][goff];
        const s_t test_grad0 = (test_grad_ref0 * adj0 + test_grad_ref1 * adj3 + test_grad_ref2 * adj6) / det;
        const s_t test_grad1 = (test_grad_ref0 * adj1 + test_grad_ref1 * adj4 + test_grad_ref2 * adj7) / det;
        const s_t test_grad2 = (test_grad_ref0 * adj2 + test_grad_ref1 * adj5 + test_grad_ref2 * adj8) / det;
        output[test * NC + 1] += q_weight[q] * det * (value_coeff1_values * test_value + grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1 + grad_coeff1_2_values * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void two_phase_flow_form_1_p_c_d3_simplex_tet4_residual_block(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[2 * NS],
    const s_t *const RSTR previous[2 * NS],
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t K_4,
    const s_t K_5,
    const s_t K_6,
    const s_t K_7,
    const s_t K_8,
    const s_t M_c,
    const s_t P_r,
    const s_t R,
    const s_t S_res,
    const s_t T,
    const s_t Z,
    const s_t dt,
    const s_t m,
    const s_t mu_c,
    const s_t porosity,
    s_t *const RSTR output[2 * NS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t p_w_values;
    s_t p_w_old_values;
    s_t p_c_values;
    s_t p_c_grad_0_ref_values;
    s_t p_c_grad_1_ref_values;
    s_t p_c_grad_2_ref_values;
    s_t p_c_old_values;
    s_t value_coeff1_values;
    s_t grad_coeff1_0_values;
    s_t grad_coeff1_1_values;
    s_t grad_coeff1_2_values;
    {
      p_w_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = current[trial * NC];
      const s_t trial_shape = shape[q * NS + trial];
      {
        p_w_values += coeff_stream[0] * trial_shape;
      }
    }
    {
      p_w_old_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = previous[trial * NC];
      const s_t trial_shape = shape[q * NS + trial];
      {
        p_w_old_values += coeff_stream[0] * trial_shape;
      }
    }
    {
      p_c_values = s_t(0);
      p_c_grad_0_ref_values = s_t(0);
      p_c_grad_1_ref_values = s_t(0);
      p_c_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = current[trial * NC + 1];
      const s_t trial_shape = shape[q * NS + trial];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        p_c_values += coeff_stream[0] * trial_shape;
        p_c_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        p_c_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
        p_c_grad_2_ref_values += coeff_stream[0] * trial_grad_2_ref;
      }
    }
    {
      p_c_old_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = previous[trial * NC + 1];
      const s_t trial_shape = shape[q * NS + trial];
      {
        p_c_old_values += coeff_stream[0] * trial_shape;
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t adj4 = adjugate[4][goff];
      const s_t adj5 = adjugate[5][goff];
      const s_t adj6 = adjugate[6][goff];
      const s_t adj7 = adjugate[7][goff];
      const s_t adj8 = adjugate[8][goff];
      const s_t p_w = p_w_values;
      const s_t p_w_old = p_w_old_values;
      const s_t p_c = p_c_values;
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values;
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values;
      const s_t p_c_grad_2_ref = p_c_grad_2_ref_values;
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
      const s_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
      const s_t p_c_old = p_c_old_values;
      const s_t residual_tmp0 = S_res + s_t(-1);
      const s_t residual_tmp1 = pow_m1(P_r);
      const s_t residual_tmp2 = (s_t(1) - m)/m;
      const s_t residual_tmp3 = pow(pow(residual_tmp1*(p_c - p_w), m) + s_t(1), residual_tmp2);
      const s_t residual_tmp4 = s_t(1) - S_res;
      const s_t residual_tmp5 = M_c/(R*T*Z);
      const s_t residual_tmp6 = p_c*residual_tmp5*pow(s_t(1) - residual_tmp3, C_ka1)*(pow(residual_tmp3, C_ka2) + s_t(-1))/mu_c;
      value_coeff1_values = -porosity*residual_tmp5*(-p_c*(residual_tmp0*residual_tmp3 + residual_tmp4) + p_c_old*(residual_tmp0*pow(pow(residual_tmp1*(p_c_old - p_w_old), m) + s_t(1), residual_tmp2) + residual_tmp4))/dt;
      grad_coeff1_0_values = residual_tmp6*(-K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2);
      grad_coeff1_1_values = residual_tmp6*(-K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2);
      grad_coeff1_2_values = residual_tmp6*(-K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2);
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
      const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
      const s_t test_grad_ref2 = grad_ref_z[q * NS + test];
      const s_t test_value = shape[q * NS + test];
      s_t *const RSTR output_row1 = output[test * NC + 1];
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        const s_t adj0 = adjugate[0][goff];
        const s_t adj1 = adjugate[1][goff];
        const s_t adj2 = adjugate[2][goff];
        const s_t adj3 = adjugate[3][goff];
        const s_t adj4 = adjugate[4][goff];
        const s_t adj5 = adjugate[5][goff];
        const s_t adj6 = adjugate[6][goff];
        const s_t adj7 = adjugate[7][goff];
        const s_t adj8 = adjugate[8][goff];
        const s_t test_grad0 = (test_grad_ref0 * adj0 + test_grad_ref1 * adj3 + test_grad_ref2 * adj6) / det;
        const s_t test_grad1 = (test_grad_ref0 * adj1 + test_grad_ref1 * adj4 + test_grad_ref2 * adj7) / det;
        const s_t test_grad2 = (test_grad_ref0 * adj2 + test_grad_ref1 * adj5 + test_grad_ref2 * adj8) / det;
        output_row1[0] += q_weight[q] * det * (value_coeff1_values * test_value + grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1 + grad_coeff1_2_values * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void two_phase_flow_form_1_p_c_d3_simplex_tet4_residual_block_contiguous(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t current[2 * NS],
    const s_t previous[2 * NS],
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t K_4,
    const s_t K_5,
    const s_t K_6,
    const s_t K_7,
    const s_t K_8,
    const s_t M_c,
    const s_t P_r,
    const s_t R,
    const s_t S_res,
    const s_t T,
    const s_t Z,
    const s_t dt,
    const s_t m,
    const s_t mu_c,
    const s_t porosity,
    s_t output[2 * NS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t p_w_values;
    s_t p_w_old_values;
    s_t p_c_values;
    s_t p_c_grad_0_ref_values;
    s_t p_c_grad_1_ref_values;
    s_t p_c_grad_2_ref_values;
    s_t p_c_old_values;
    s_t value_coeff1_values;
    s_t grad_coeff1_0_values;
    s_t grad_coeff1_1_values;
    s_t grad_coeff1_2_values;
    {
      p_w_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_shape = shape[q * NS + trial];
      {
        p_w_values += current[trial * NC] * trial_shape;
      }
    }
    {
      p_w_old_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_shape = shape[q * NS + trial];
      {
        p_w_old_values += previous[trial * NC] * trial_shape;
      }
    }
    {
      p_c_values = s_t(0);
      p_c_grad_0_ref_values = s_t(0);
      p_c_grad_1_ref_values = s_t(0);
      p_c_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_shape = shape[q * NS + trial];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        p_c_values += current[trial * NC + 1] * trial_shape;
        p_c_grad_0_ref_values += current[trial * NC + 1] * trial_grad_0_ref;
        p_c_grad_1_ref_values += current[trial * NC + 1] * trial_grad_1_ref;
        p_c_grad_2_ref_values += current[trial * NC + 1] * trial_grad_2_ref;
      }
    }
    {
      p_c_old_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_shape = shape[q * NS + trial];
      {
        p_c_old_values += previous[trial * NC + 1] * trial_shape;
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t adj4 = adjugate[4][goff];
      const s_t adj5 = adjugate[5][goff];
      const s_t adj6 = adjugate[6][goff];
      const s_t adj7 = adjugate[7][goff];
      const s_t adj8 = adjugate[8][goff];
      const s_t p_w = p_w_values;
      const s_t p_w_old = p_w_old_values;
      const s_t p_c = p_c_values;
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values;
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values;
      const s_t p_c_grad_2_ref = p_c_grad_2_ref_values;
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
      const s_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
      const s_t p_c_old = p_c_old_values;
      const s_t residual_tmp0 = S_res + s_t(-1);
      const s_t residual_tmp1 = pow_m1(P_r);
      const s_t residual_tmp2 = (s_t(1) - m)/m;
      const s_t residual_tmp3 = pow(pow(residual_tmp1*(p_c - p_w), m) + s_t(1), residual_tmp2);
      const s_t residual_tmp4 = s_t(1) - S_res;
      const s_t residual_tmp5 = M_c/(R*T*Z);
      const s_t residual_tmp6 = p_c*residual_tmp5*pow(s_t(1) - residual_tmp3, C_ka1)*(pow(residual_tmp3, C_ka2) + s_t(-1))/mu_c;
      value_coeff1_values = -porosity*residual_tmp5*(-p_c*(residual_tmp0*residual_tmp3 + residual_tmp4) + p_c_old*(residual_tmp0*pow(pow(residual_tmp1*(p_c_old - p_w_old), m) + s_t(1), residual_tmp2) + residual_tmp4))/dt;
      grad_coeff1_0_values = residual_tmp6*(-K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2);
      grad_coeff1_1_values = residual_tmp6*(-K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2);
      grad_coeff1_2_values = residual_tmp6*(-K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2);
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
      const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
      const s_t test_grad_ref2 = grad_ref_z[q * NS + test];
      const s_t test_value = shape[q * NS + test];
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        const s_t adj0 = adjugate[0][goff];
        const s_t adj1 = adjugate[1][goff];
        const s_t adj2 = adjugate[2][goff];
        const s_t adj3 = adjugate[3][goff];
        const s_t adj4 = adjugate[4][goff];
        const s_t adj5 = adjugate[5][goff];
        const s_t adj6 = adjugate[6][goff];
        const s_t adj7 = adjugate[7][goff];
        const s_t adj8 = adjugate[8][goff];
        const s_t test_grad0 = (test_grad_ref0 * adj0 + test_grad_ref1 * adj3 + test_grad_ref2 * adj6) / det;
        const s_t test_grad1 = (test_grad_ref0 * adj1 + test_grad_ref1 * adj4 + test_grad_ref2 * adj7) / det;
        const s_t test_grad2 = (test_grad_ref0 * adj2 + test_grad_ref1 * adj5 + test_grad_ref2 * adj8) / det;
        output[test * NC + 1] += q_weight[q] * det * (value_coeff1_values * test_value + grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1 + grad_coeff1_2_values * test_grad2);
      }
    }
  }
}


} // namespace codegen
} // namespace sfem

#endif
