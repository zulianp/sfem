#ifndef TWO_PHASE_FLOW_FORM_2_P_C_P_C_D2_SIMPLEX_LOCAL_HPP
#define TWO_PHASE_FLOW_FORM_2_P_C_P_C_D2_SIMPLEX_LOCAL_HPP

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
__host__ __device__ __forceinline__ void two_phase_flow_form_2_p_c_p_c_d2_simplex_jacobian_action_block(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[2 * NS],
    const s_t *const RSTR direction[2 * NS],
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
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
    s_t p_c_values;
    s_t p_c_grad_0_ref_values;
    s_t p_c_grad_1_ref_values;
    s_t p_c_direction_values;
    s_t p_c_direction_grad_0_ref_values;
    s_t p_c_direction_grad_1_ref_values;
    s_t value_coeff1_values;
    s_t grad_coeff1_0_values;
    s_t grad_coeff1_1_values;
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
      p_c_values = s_t(0);
      p_c_grad_0_ref_values = s_t(0);
      p_c_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = current[trial * NC + 1];
      const s_t trial_shape = shape[q * NS + trial];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        p_c_values += coeff_stream[0] * trial_shape;
        p_c_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        p_c_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
      }
    }
    {
      p_c_direction_values = s_t(0);
      p_c_direction_grad_0_ref_values = s_t(0);
      p_c_direction_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = direction[trial * NC + 1];
      const s_t trial_shape = shape[q * NS + trial];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        p_c_direction_values += coeff_stream[0] * trial_shape;
        p_c_direction_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        p_c_direction_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t p_w = p_w_values;
      const s_t p_c = p_c_values;
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values;
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values;
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj2) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj3) / det;
      const s_t p_c_direction = p_c_direction_values;
      const s_t p_c_direction_grad_0_ref = p_c_direction_grad_0_ref_values;
      const s_t p_c_direction_grad_1_ref = p_c_direction_grad_1_ref_values;
      const s_t p_c_direction_grad_0 = (p_c_direction_grad_0_ref * adj0 + p_c_direction_grad_1_ref * adj2) / det;
      const s_t p_c_direction_grad_1 = (p_c_direction_grad_0_ref * adj1 + p_c_direction_grad_1_ref * adj3) / det;
      const s_t residual_tmp0 = p_c - p_w;
      const s_t residual_tmp1 = pow(residual_tmp0/P_r, m);
      const s_t residual_tmp2 = residual_tmp1 + s_t(1);
      const s_t residual_tmp3 = s_t(1) - m;
      const s_t residual_tmp4 = pow(residual_tmp2, residual_tmp3/m);
      const s_t residual_tmp5 = residual_tmp4*(S_res + s_t(-1));
      const s_t residual_tmp6 = p_c*residual_tmp1*residual_tmp3/(residual_tmp0*residual_tmp2);
      const s_t residual_tmp7 = pow_m1(dt);
      const s_t residual_tmp8 = pow_m1(R);
      const s_t residual_tmp9 = pow_m1(T);
      const s_t residual_tmp10 = pow_m1(Z);
      const s_t residual_tmp11 = M_c*residual_tmp10*residual_tmp8*residual_tmp9;
      const s_t residual_tmp12 = pow_m1(mu_c);
      const s_t residual_tmp13 = s_t(1) - residual_tmp4;
      const s_t residual_tmp14 = pow(residual_tmp13, C_ka1);
      const s_t residual_tmp15 = pow(residual_tmp4, C_ka2);
      const s_t residual_tmp16 = residual_tmp14*(residual_tmp15 + s_t(-1));
      const s_t residual_tmp17 = p_c*residual_tmp11*residual_tmp12*residual_tmp16;
      const s_t residual_tmp18 = p_c_direction_grad_0*residual_tmp17;
      const s_t residual_tmp19 = p_c_direction_grad_1*residual_tmp17;
      const s_t residual_tmp20 = -K_0*p_c_grad_0 - K_1*p_c_grad_1;
      const s_t residual_tmp21 = dt*residual_tmp16;
      const s_t residual_tmp22 = residual_tmp20*residual_tmp21;
      const s_t residual_tmp23 = C_ka2*dt*residual_tmp14*residual_tmp15*residual_tmp6;
      const s_t residual_tmp24 = C_ka1*residual_tmp4*residual_tmp6/residual_tmp13;
      const s_t residual_tmp25 = -K_2*p_c_grad_0 - K_3*p_c_grad_1;
      const s_t residual_tmp26 = residual_tmp21*residual_tmp25;
      value_coeff1_values = -p_c_direction*porosity*residual_tmp11*residual_tmp7*(S_res - residual_tmp5*residual_tmp6 - residual_tmp5 + s_t(-1));
      grad_coeff1_0_values = -K_0*residual_tmp18 - K_1*residual_tmp19 + M_c*p_c_direction*residual_tmp10*residual_tmp12*residual_tmp7*residual_tmp8*residual_tmp9*(residual_tmp20*residual_tmp23 - residual_tmp22*residual_tmp24 + residual_tmp22);
      grad_coeff1_1_values = -K_2*residual_tmp18 - K_3*residual_tmp19 + M_c*p_c_direction*residual_tmp10*residual_tmp12*residual_tmp7*residual_tmp8*residual_tmp9*(residual_tmp23*residual_tmp25 - residual_tmp24*residual_tmp26 + residual_tmp26);
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
      const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
      const s_t test_value = shape[q * NS + test];
      s_t *const RSTR output_row1 = output[test * NC + 1];
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        const s_t adj0 = adjugate[0][goff];
        const s_t adj1 = adjugate[1][goff];
        const s_t adj2 = adjugate[2][goff];
        const s_t adj3 = adjugate[3][goff];
        const s_t test_grad0 = (test_grad_ref0 * adj0 + test_grad_ref1 * adj2) / det;
        const s_t test_grad1 = (test_grad_ref0 * adj1 + test_grad_ref1 * adj3) / det;
        output_row1[0] += q_weight[q] * det * (value_coeff1_values * test_value + grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1);
      }
    }
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void two_phase_flow_form_2_p_c_p_c_d2_simplex_jacobian_action_block_contiguous(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
    const s_t current[2 * NS],
    const s_t direction[2 * NS],
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
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
    s_t p_c_values;
    s_t p_c_grad_0_ref_values;
    s_t p_c_grad_1_ref_values;
    s_t p_c_direction_values;
    s_t p_c_direction_grad_0_ref_values;
    s_t p_c_direction_grad_1_ref_values;
    s_t value_coeff1_values;
    s_t grad_coeff1_0_values;
    s_t grad_coeff1_1_values;
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
      p_c_values = s_t(0);
      p_c_grad_0_ref_values = s_t(0);
      p_c_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_shape = shape[q * NS + trial];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        p_c_values += current[trial * NC + 1] * trial_shape;
        p_c_grad_0_ref_values += current[trial * NC + 1] * trial_grad_0_ref;
        p_c_grad_1_ref_values += current[trial * NC + 1] * trial_grad_1_ref;
      }
    }
    {
      p_c_direction_values = s_t(0);
      p_c_direction_grad_0_ref_values = s_t(0);
      p_c_direction_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_shape = shape[q * NS + trial];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        p_c_direction_values += direction[trial * NC + 1] * trial_shape;
        p_c_direction_grad_0_ref_values += direction[trial * NC + 1] * trial_grad_0_ref;
        p_c_direction_grad_1_ref_values += direction[trial * NC + 1] * trial_grad_1_ref;
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t p_w = p_w_values;
      const s_t p_c = p_c_values;
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values;
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values;
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj2) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj3) / det;
      const s_t p_c_direction = p_c_direction_values;
      const s_t p_c_direction_grad_0_ref = p_c_direction_grad_0_ref_values;
      const s_t p_c_direction_grad_1_ref = p_c_direction_grad_1_ref_values;
      const s_t p_c_direction_grad_0 = (p_c_direction_grad_0_ref * adj0 + p_c_direction_grad_1_ref * adj2) / det;
      const s_t p_c_direction_grad_1 = (p_c_direction_grad_0_ref * adj1 + p_c_direction_grad_1_ref * adj3) / det;
      const s_t residual_tmp0 = p_c - p_w;
      const s_t residual_tmp1 = pow(residual_tmp0/P_r, m);
      const s_t residual_tmp2 = residual_tmp1 + s_t(1);
      const s_t residual_tmp3 = s_t(1) - m;
      const s_t residual_tmp4 = pow(residual_tmp2, residual_tmp3/m);
      const s_t residual_tmp5 = residual_tmp4*(S_res + s_t(-1));
      const s_t residual_tmp6 = p_c*residual_tmp1*residual_tmp3/(residual_tmp0*residual_tmp2);
      const s_t residual_tmp7 = pow_m1(dt);
      const s_t residual_tmp8 = pow_m1(R);
      const s_t residual_tmp9 = pow_m1(T);
      const s_t residual_tmp10 = pow_m1(Z);
      const s_t residual_tmp11 = M_c*residual_tmp10*residual_tmp8*residual_tmp9;
      const s_t residual_tmp12 = pow_m1(mu_c);
      const s_t residual_tmp13 = s_t(1) - residual_tmp4;
      const s_t residual_tmp14 = pow(residual_tmp13, C_ka1);
      const s_t residual_tmp15 = pow(residual_tmp4, C_ka2);
      const s_t residual_tmp16 = residual_tmp14*(residual_tmp15 + s_t(-1));
      const s_t residual_tmp17 = p_c*residual_tmp11*residual_tmp12*residual_tmp16;
      const s_t residual_tmp18 = p_c_direction_grad_0*residual_tmp17;
      const s_t residual_tmp19 = p_c_direction_grad_1*residual_tmp17;
      const s_t residual_tmp20 = -K_0*p_c_grad_0 - K_1*p_c_grad_1;
      const s_t residual_tmp21 = dt*residual_tmp16;
      const s_t residual_tmp22 = residual_tmp20*residual_tmp21;
      const s_t residual_tmp23 = C_ka2*dt*residual_tmp14*residual_tmp15*residual_tmp6;
      const s_t residual_tmp24 = C_ka1*residual_tmp4*residual_tmp6/residual_tmp13;
      const s_t residual_tmp25 = -K_2*p_c_grad_0 - K_3*p_c_grad_1;
      const s_t residual_tmp26 = residual_tmp21*residual_tmp25;
      value_coeff1_values = -p_c_direction*porosity*residual_tmp11*residual_tmp7*(S_res - residual_tmp5*residual_tmp6 - residual_tmp5 + s_t(-1));
      grad_coeff1_0_values = -K_0*residual_tmp18 - K_1*residual_tmp19 + M_c*p_c_direction*residual_tmp10*residual_tmp12*residual_tmp7*residual_tmp8*residual_tmp9*(residual_tmp20*residual_tmp23 - residual_tmp22*residual_tmp24 + residual_tmp22);
      grad_coeff1_1_values = -K_2*residual_tmp18 - K_3*residual_tmp19 + M_c*p_c_direction*residual_tmp10*residual_tmp12*residual_tmp7*residual_tmp8*residual_tmp9*(residual_tmp23*residual_tmp25 - residual_tmp24*residual_tmp26 + residual_tmp26);
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
      const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
      const s_t test_value = shape[q * NS + test];
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        const s_t adj0 = adjugate[0][goff];
        const s_t adj1 = adjugate[1][goff];
        const s_t adj2 = adjugate[2][goff];
        const s_t adj3 = adjugate[3][goff];
        const s_t test_grad0 = (test_grad_ref0 * adj0 + test_grad_ref1 * adj2) / det;
        const s_t test_grad1 = (test_grad_ref0 * adj1 + test_grad_ref1 * adj3) / det;
        output[test * NC + 1] += q_weight[q] * det * (value_coeff1_values * test_value + grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1);
      }
    }
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void two_phase_flow_form_2_p_c_p_c_d2_simplex_tri3_jacobian_action_block(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[2 * NS],
    const s_t *const RSTR direction[2 * NS],
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
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
    s_t p_c_values;
    s_t p_c_grad_0_ref_values;
    s_t p_c_grad_1_ref_values;
    s_t p_c_direction_values;
    s_t p_c_direction_grad_0_ref_values;
    s_t p_c_direction_grad_1_ref_values;
    s_t value_coeff1_values;
    s_t grad_coeff1_0_values;
    s_t grad_coeff1_1_values;
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
      p_c_values = s_t(0);
      p_c_grad_0_ref_values = s_t(0);
      p_c_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = current[trial * NC + 1];
      const s_t trial_shape = shape[q * NS + trial];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        p_c_values += coeff_stream[0] * trial_shape;
        p_c_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        p_c_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
      }
    }
    {
      p_c_direction_values = s_t(0);
      p_c_direction_grad_0_ref_values = s_t(0);
      p_c_direction_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = direction[trial * NC + 1];
      const s_t trial_shape = shape[q * NS + trial];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        p_c_direction_values += coeff_stream[0] * trial_shape;
        p_c_direction_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        p_c_direction_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t p_w = p_w_values;
      const s_t p_c = p_c_values;
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values;
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values;
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj2) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj3) / det;
      const s_t p_c_direction = p_c_direction_values;
      const s_t p_c_direction_grad_0_ref = p_c_direction_grad_0_ref_values;
      const s_t p_c_direction_grad_1_ref = p_c_direction_grad_1_ref_values;
      const s_t p_c_direction_grad_0 = (p_c_direction_grad_0_ref * adj0 + p_c_direction_grad_1_ref * adj2) / det;
      const s_t p_c_direction_grad_1 = (p_c_direction_grad_0_ref * adj1 + p_c_direction_grad_1_ref * adj3) / det;
      const s_t residual_tmp0 = p_c - p_w;
      const s_t residual_tmp1 = pow(residual_tmp0/P_r, m);
      const s_t residual_tmp2 = residual_tmp1 + s_t(1);
      const s_t residual_tmp3 = s_t(1) - m;
      const s_t residual_tmp4 = pow(residual_tmp2, residual_tmp3/m);
      const s_t residual_tmp5 = residual_tmp4*(S_res + s_t(-1));
      const s_t residual_tmp6 = p_c*residual_tmp1*residual_tmp3/(residual_tmp0*residual_tmp2);
      const s_t residual_tmp7 = pow_m1(dt);
      const s_t residual_tmp8 = pow_m1(R);
      const s_t residual_tmp9 = pow_m1(T);
      const s_t residual_tmp10 = pow_m1(Z);
      const s_t residual_tmp11 = M_c*residual_tmp10*residual_tmp8*residual_tmp9;
      const s_t residual_tmp12 = pow_m1(mu_c);
      const s_t residual_tmp13 = s_t(1) - residual_tmp4;
      const s_t residual_tmp14 = pow(residual_tmp13, C_ka1);
      const s_t residual_tmp15 = pow(residual_tmp4, C_ka2);
      const s_t residual_tmp16 = residual_tmp14*(residual_tmp15 + s_t(-1));
      const s_t residual_tmp17 = p_c*residual_tmp11*residual_tmp12*residual_tmp16;
      const s_t residual_tmp18 = p_c_direction_grad_0*residual_tmp17;
      const s_t residual_tmp19 = p_c_direction_grad_1*residual_tmp17;
      const s_t residual_tmp20 = -K_0*p_c_grad_0 - K_1*p_c_grad_1;
      const s_t residual_tmp21 = dt*residual_tmp16;
      const s_t residual_tmp22 = residual_tmp20*residual_tmp21;
      const s_t residual_tmp23 = C_ka2*dt*residual_tmp14*residual_tmp15*residual_tmp6;
      const s_t residual_tmp24 = C_ka1*residual_tmp4*residual_tmp6/residual_tmp13;
      const s_t residual_tmp25 = -K_2*p_c_grad_0 - K_3*p_c_grad_1;
      const s_t residual_tmp26 = residual_tmp21*residual_tmp25;
      value_coeff1_values = -p_c_direction*porosity*residual_tmp11*residual_tmp7*(S_res - residual_tmp5*residual_tmp6 - residual_tmp5 + s_t(-1));
      grad_coeff1_0_values = -K_0*residual_tmp18 - K_1*residual_tmp19 + M_c*p_c_direction*residual_tmp10*residual_tmp12*residual_tmp7*residual_tmp8*residual_tmp9*(residual_tmp20*residual_tmp23 - residual_tmp22*residual_tmp24 + residual_tmp22);
      grad_coeff1_1_values = -K_2*residual_tmp18 - K_3*residual_tmp19 + M_c*p_c_direction*residual_tmp10*residual_tmp12*residual_tmp7*residual_tmp8*residual_tmp9*(residual_tmp23*residual_tmp25 - residual_tmp24*residual_tmp26 + residual_tmp26);
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
      const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
      const s_t test_value = shape[q * NS + test];
      s_t *const RSTR output_row1 = output[test * NC + 1];
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        const s_t adj0 = adjugate[0][goff];
        const s_t adj1 = adjugate[1][goff];
        const s_t adj2 = adjugate[2][goff];
        const s_t adj3 = adjugate[3][goff];
        const s_t test_grad0 = (test_grad_ref0 * adj0 + test_grad_ref1 * adj2) / det;
        const s_t test_grad1 = (test_grad_ref0 * adj1 + test_grad_ref1 * adj3) / det;
        output_row1[0] += q_weight[q] * det * (value_coeff1_values * test_value + grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1);
      }
    }
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void two_phase_flow_form_2_p_c_p_c_d2_simplex_tri3_jacobian_action_block_contiguous(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
    const s_t current[2 * NS],
    const s_t direction[2 * NS],
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
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
    s_t p_c_values;
    s_t p_c_grad_0_ref_values;
    s_t p_c_grad_1_ref_values;
    s_t p_c_direction_values;
    s_t p_c_direction_grad_0_ref_values;
    s_t p_c_direction_grad_1_ref_values;
    s_t value_coeff1_values;
    s_t grad_coeff1_0_values;
    s_t grad_coeff1_1_values;
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
      p_c_values = s_t(0);
      p_c_grad_0_ref_values = s_t(0);
      p_c_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_shape = shape[q * NS + trial];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        p_c_values += current[trial * NC + 1] * trial_shape;
        p_c_grad_0_ref_values += current[trial * NC + 1] * trial_grad_0_ref;
        p_c_grad_1_ref_values += current[trial * NC + 1] * trial_grad_1_ref;
      }
    }
    {
      p_c_direction_values = s_t(0);
      p_c_direction_grad_0_ref_values = s_t(0);
      p_c_direction_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_shape = shape[q * NS + trial];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        p_c_direction_values += direction[trial * NC + 1] * trial_shape;
        p_c_direction_grad_0_ref_values += direction[trial * NC + 1] * trial_grad_0_ref;
        p_c_direction_grad_1_ref_values += direction[trial * NC + 1] * trial_grad_1_ref;
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t p_w = p_w_values;
      const s_t p_c = p_c_values;
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values;
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values;
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj2) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj3) / det;
      const s_t p_c_direction = p_c_direction_values;
      const s_t p_c_direction_grad_0_ref = p_c_direction_grad_0_ref_values;
      const s_t p_c_direction_grad_1_ref = p_c_direction_grad_1_ref_values;
      const s_t p_c_direction_grad_0 = (p_c_direction_grad_0_ref * adj0 + p_c_direction_grad_1_ref * adj2) / det;
      const s_t p_c_direction_grad_1 = (p_c_direction_grad_0_ref * adj1 + p_c_direction_grad_1_ref * adj3) / det;
      const s_t residual_tmp0 = p_c - p_w;
      const s_t residual_tmp1 = pow(residual_tmp0/P_r, m);
      const s_t residual_tmp2 = residual_tmp1 + s_t(1);
      const s_t residual_tmp3 = s_t(1) - m;
      const s_t residual_tmp4 = pow(residual_tmp2, residual_tmp3/m);
      const s_t residual_tmp5 = residual_tmp4*(S_res + s_t(-1));
      const s_t residual_tmp6 = p_c*residual_tmp1*residual_tmp3/(residual_tmp0*residual_tmp2);
      const s_t residual_tmp7 = pow_m1(dt);
      const s_t residual_tmp8 = pow_m1(R);
      const s_t residual_tmp9 = pow_m1(T);
      const s_t residual_tmp10 = pow_m1(Z);
      const s_t residual_tmp11 = M_c*residual_tmp10*residual_tmp8*residual_tmp9;
      const s_t residual_tmp12 = pow_m1(mu_c);
      const s_t residual_tmp13 = s_t(1) - residual_tmp4;
      const s_t residual_tmp14 = pow(residual_tmp13, C_ka1);
      const s_t residual_tmp15 = pow(residual_tmp4, C_ka2);
      const s_t residual_tmp16 = residual_tmp14*(residual_tmp15 + s_t(-1));
      const s_t residual_tmp17 = p_c*residual_tmp11*residual_tmp12*residual_tmp16;
      const s_t residual_tmp18 = p_c_direction_grad_0*residual_tmp17;
      const s_t residual_tmp19 = p_c_direction_grad_1*residual_tmp17;
      const s_t residual_tmp20 = -K_0*p_c_grad_0 - K_1*p_c_grad_1;
      const s_t residual_tmp21 = dt*residual_tmp16;
      const s_t residual_tmp22 = residual_tmp20*residual_tmp21;
      const s_t residual_tmp23 = C_ka2*dt*residual_tmp14*residual_tmp15*residual_tmp6;
      const s_t residual_tmp24 = C_ka1*residual_tmp4*residual_tmp6/residual_tmp13;
      const s_t residual_tmp25 = -K_2*p_c_grad_0 - K_3*p_c_grad_1;
      const s_t residual_tmp26 = residual_tmp21*residual_tmp25;
      value_coeff1_values = -p_c_direction*porosity*residual_tmp11*residual_tmp7*(S_res - residual_tmp5*residual_tmp6 - residual_tmp5 + s_t(-1));
      grad_coeff1_0_values = -K_0*residual_tmp18 - K_1*residual_tmp19 + M_c*p_c_direction*residual_tmp10*residual_tmp12*residual_tmp7*residual_tmp8*residual_tmp9*(residual_tmp20*residual_tmp23 - residual_tmp22*residual_tmp24 + residual_tmp22);
      grad_coeff1_1_values = -K_2*residual_tmp18 - K_3*residual_tmp19 + M_c*p_c_direction*residual_tmp10*residual_tmp12*residual_tmp7*residual_tmp8*residual_tmp9*(residual_tmp23*residual_tmp25 - residual_tmp24*residual_tmp26 + residual_tmp26);
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
      const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
      const s_t test_value = shape[q * NS + test];
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        const s_t adj0 = adjugate[0][goff];
        const s_t adj1 = adjugate[1][goff];
        const s_t adj2 = adjugate[2][goff];
        const s_t adj3 = adjugate[3][goff];
        const s_t test_grad0 = (test_grad_ref0 * adj0 + test_grad_ref1 * adj2) / det;
        const s_t test_grad1 = (test_grad_ref0 * adj1 + test_grad_ref1 * adj3) / det;
        output[test * NC + 1] += q_weight[q] * det * (value_coeff1_values * test_value + grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1);
      }
    }
  }
}

} // namespace codegen
} // namespace sfem

#endif
