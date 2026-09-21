#ifndef TWO_PHASE_FLOW_FORM_1_P_C_D2_SIMPLEX_LOCAL_HPP
#define TWO_PHASE_FLOW_FORM_1_P_C_D2_SIMPLEX_LOCAL_HPP

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
__host__ __device__ __forceinline__ void two_phase_flow_form_1_p_c_d2_simplex_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[2 * NS],
    const s_t *const RSTR previous[2 * NS],
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
    s_t p_w_values[VS];
    s_t p_w_old_values[VS];
    s_t p_c_values[VS];
    s_t p_c_grad_0_ref_values[VS];
    s_t p_c_grad_1_ref_values[VS];
    s_t p_c_old_values[VS];
    s_t value_coeff1_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    {
      p_w_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC][0];
        p_w_values[0] += coeff * shape[q * NS + trial];
      }
    }
    {
      p_w_old_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC][0];
        p_w_old_values[0] += coeff * shape[q * NS + trial];
      }
    }
    {
      p_c_values[0] = s_t(0);
      p_c_grad_0_ref_values[0] = s_t(0);
      p_c_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 1][0];
        p_c_values[0] += coeff * shape[q * NS + trial];
        p_c_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        p_c_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      p_c_old_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC + 1][0];
        p_c_old_values[0] += coeff * shape[q * NS + trial];
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t p_w = p_w_values[0];
      const s_t p_w_old = p_w_old_values[0];
      const s_t p_c = p_c_values[0];
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values[0];
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values[0];
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj2) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj3) / det;
      const s_t p_c_old = p_c_old_values[0];
      const s_t residual_tmp0 = S_res + s_t(-1);
      const s_t residual_tmp1 = pow_m1(P_r);
      const s_t residual_tmp2 = (s_t(1) - m)/m;
      const s_t residual_tmp3 = pow(pow(residual_tmp1*(p_c - p_w), m) + s_t(1), residual_tmp2);
      const s_t residual_tmp4 = s_t(1) - S_res;
      const s_t residual_tmp5 = M_c/(R*T*Z);
      const s_t residual_tmp6 = p_c*residual_tmp5*pow(s_t(1) - residual_tmp3, C_ka1)*(pow(residual_tmp3, C_ka2) + s_t(-1))/mu_c;
      const s_t value_coeff1 = -porosity*residual_tmp5*(-p_c*(residual_tmp0*residual_tmp3 + residual_tmp4) + p_c_old*(residual_tmp0*pow(pow(residual_tmp1*(p_c_old - p_w_old), m) + s_t(1), residual_tmp2) + residual_tmp4))/dt;
      const s_t grad_coeff1_0 = residual_tmp6*(-K_0*p_c_grad_0 - K_1*p_c_grad_1);
      const s_t grad_coeff1_1 = residual_tmp6*(-K_2*p_c_grad_0 - K_3*p_c_grad_1);
      value_coeff1_values[0] = value_coeff1;
      grad_coeff1_0_values[0] = grad_coeff1_0;
      grad_coeff1_1_values[0] = grad_coeff1_1;
    }
    for (int test = 0; test < NS; ++test) {
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        const s_t adj0 = adjugate[0][goff];
        const s_t adj1 = adjugate[1][goff];
        const s_t adj2 = adjugate[2][goff];
        const s_t adj3 = adjugate[3][goff];
        const s_t test_value = shape[q * NS + test];
        const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj2) / det;
        const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj3) / det;
        output[test * NC + 1][0] += q_weight[q] * det * (value_coeff1_values[0] * test_value + grad_coeff1_0_values[0] * test_grad0 + grad_coeff1_1_values[0] * test_grad1);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void two_phase_flow_form_1_p_c_d2_simplex_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
    const s_t current[2 * NS][VS],
    const s_t previous[2 * NS][VS],
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
    s_t output[2 * NS][VS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t p_w_values[VS];
    s_t p_w_old_values[VS];
    s_t p_c_values[VS];
    s_t p_c_grad_0_ref_values[VS];
    s_t p_c_grad_1_ref_values[VS];
    s_t p_c_old_values[VS];
    s_t value_coeff1_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    {
      p_w_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC][0];
        p_w_values[0] += coeff * shape[q * NS + trial];
      }
    }
    {
      p_w_old_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC][0];
        p_w_old_values[0] += coeff * shape[q * NS + trial];
      }
    }
    {
      p_c_values[0] = s_t(0);
      p_c_grad_0_ref_values[0] = s_t(0);
      p_c_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 1][0];
        p_c_values[0] += coeff * shape[q * NS + trial];
        p_c_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        p_c_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      p_c_old_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC + 1][0];
        p_c_old_values[0] += coeff * shape[q * NS + trial];
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t p_w = p_w_values[0];
      const s_t p_w_old = p_w_old_values[0];
      const s_t p_c = p_c_values[0];
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values[0];
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values[0];
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj2) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj3) / det;
      const s_t p_c_old = p_c_old_values[0];
      const s_t residual_tmp0 = S_res + s_t(-1);
      const s_t residual_tmp1 = pow_m1(P_r);
      const s_t residual_tmp2 = (s_t(1) - m)/m;
      const s_t residual_tmp3 = pow(pow(residual_tmp1*(p_c - p_w), m) + s_t(1), residual_tmp2);
      const s_t residual_tmp4 = s_t(1) - S_res;
      const s_t residual_tmp5 = M_c/(R*T*Z);
      const s_t residual_tmp6 = p_c*residual_tmp5*pow(s_t(1) - residual_tmp3, C_ka1)*(pow(residual_tmp3, C_ka2) + s_t(-1))/mu_c;
      const s_t value_coeff1 = -porosity*residual_tmp5*(-p_c*(residual_tmp0*residual_tmp3 + residual_tmp4) + p_c_old*(residual_tmp0*pow(pow(residual_tmp1*(p_c_old - p_w_old), m) + s_t(1), residual_tmp2) + residual_tmp4))/dt;
      const s_t grad_coeff1_0 = residual_tmp6*(-K_0*p_c_grad_0 - K_1*p_c_grad_1);
      const s_t grad_coeff1_1 = residual_tmp6*(-K_2*p_c_grad_0 - K_3*p_c_grad_1);
      value_coeff1_values[0] = value_coeff1;
      grad_coeff1_0_values[0] = grad_coeff1_0;
      grad_coeff1_1_values[0] = grad_coeff1_1;
    }
    for (int test = 0; test < NS; ++test) {
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        const s_t adj0 = adjugate[0][goff];
        const s_t adj1 = adjugate[1][goff];
        const s_t adj2 = adjugate[2][goff];
        const s_t adj3 = adjugate[3][goff];
        const s_t test_value = shape[q * NS + test];
        const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj2) / det;
        const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj3) / det;
        output[test * NC + 1][0] += q_weight[q] * det * (value_coeff1_values[0] * test_value + grad_coeff1_0_values[0] * test_grad0 + grad_coeff1_1_values[0] * test_grad1);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void two_phase_flow_form_1_p_c_d2_simplex_tri3_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[2 * NS],
    const s_t *const RSTR previous[2 * NS],
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
    s_t p_w_values[VS];
    s_t p_w_old_values[VS];
    s_t p_c_values[VS];
    s_t p_c_grad_0_ref_values[VS];
    s_t p_c_grad_1_ref_values[VS];
    s_t p_c_old_values[VS];
    s_t value_coeff1_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    {
      p_w_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC][0];
        p_w_values[0] += coeff * shape[q * NS + trial];
      }
    }
    {
      p_w_old_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC][0];
        p_w_old_values[0] += coeff * shape[q * NS + trial];
      }
    }
    {
      p_c_values[0] = s_t(0);
      p_c_grad_0_ref_values[0] = s_t(0);
      p_c_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 1][0];
        p_c_values[0] += coeff * shape[q * NS + trial];
        p_c_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        p_c_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      p_c_old_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC + 1][0];
        p_c_old_values[0] += coeff * shape[q * NS + trial];
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t p_w = p_w_values[0];
      const s_t p_w_old = p_w_old_values[0];
      const s_t p_c = p_c_values[0];
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values[0];
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values[0];
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj2) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj3) / det;
      const s_t p_c_old = p_c_old_values[0];
      const s_t residual_tmp0 = S_res + s_t(-1);
      const s_t residual_tmp1 = pow_m1(P_r);
      const s_t residual_tmp2 = (s_t(1) - m)/m;
      const s_t residual_tmp3 = pow(pow(residual_tmp1*(p_c - p_w), m) + s_t(1), residual_tmp2);
      const s_t residual_tmp4 = s_t(1) - S_res;
      const s_t residual_tmp5 = M_c/(R*T*Z);
      const s_t residual_tmp6 = p_c*residual_tmp5*pow(s_t(1) - residual_tmp3, C_ka1)*(pow(residual_tmp3, C_ka2) + s_t(-1))/mu_c;
      const s_t value_coeff1 = -porosity*residual_tmp5*(-p_c*(residual_tmp0*residual_tmp3 + residual_tmp4) + p_c_old*(residual_tmp0*pow(pow(residual_tmp1*(p_c_old - p_w_old), m) + s_t(1), residual_tmp2) + residual_tmp4))/dt;
      const s_t grad_coeff1_0 = residual_tmp6*(-K_0*p_c_grad_0 - K_1*p_c_grad_1);
      const s_t grad_coeff1_1 = residual_tmp6*(-K_2*p_c_grad_0 - K_3*p_c_grad_1);
      value_coeff1_values[0] = value_coeff1;
      grad_coeff1_0_values[0] = grad_coeff1_0;
      grad_coeff1_1_values[0] = grad_coeff1_1;
    }
    for (int test = 0; test < NS; ++test) {
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        const s_t adj0 = adjugate[0][goff];
        const s_t adj1 = adjugate[1][goff];
        const s_t adj2 = adjugate[2][goff];
        const s_t adj3 = adjugate[3][goff];
        const s_t test_value = shape[q * NS + test];
        const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj2) / det;
        const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj3) / det;
        output[test * NC + 1][0] += q_weight[q] * det * (value_coeff1_values[0] * test_value + grad_coeff1_0_values[0] * test_grad0 + grad_coeff1_1_values[0] * test_grad1);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void two_phase_flow_form_1_p_c_d2_simplex_tri3_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
    const s_t current[2 * NS][VS],
    const s_t previous[2 * NS][VS],
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
    s_t output[2 * NS][VS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t p_w_values[VS];
    s_t p_w_old_values[VS];
    s_t p_c_values[VS];
    s_t p_c_grad_0_ref_values[VS];
    s_t p_c_grad_1_ref_values[VS];
    s_t p_c_old_values[VS];
    s_t value_coeff1_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    {
      p_w_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC][0];
        p_w_values[0] += coeff * shape[q * NS + trial];
      }
    }
    {
      p_w_old_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC][0];
        p_w_old_values[0] += coeff * shape[q * NS + trial];
      }
    }
    {
      p_c_values[0] = s_t(0);
      p_c_grad_0_ref_values[0] = s_t(0);
      p_c_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 1][0];
        p_c_values[0] += coeff * shape[q * NS + trial];
        p_c_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        p_c_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      p_c_old_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC + 1][0];
        p_c_old_values[0] += coeff * shape[q * NS + trial];
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t p_w = p_w_values[0];
      const s_t p_w_old = p_w_old_values[0];
      const s_t p_c = p_c_values[0];
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values[0];
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values[0];
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj2) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj3) / det;
      const s_t p_c_old = p_c_old_values[0];
      const s_t residual_tmp0 = S_res + s_t(-1);
      const s_t residual_tmp1 = pow_m1(P_r);
      const s_t residual_tmp2 = (s_t(1) - m)/m;
      const s_t residual_tmp3 = pow(pow(residual_tmp1*(p_c - p_w), m) + s_t(1), residual_tmp2);
      const s_t residual_tmp4 = s_t(1) - S_res;
      const s_t residual_tmp5 = M_c/(R*T*Z);
      const s_t residual_tmp6 = p_c*residual_tmp5*pow(s_t(1) - residual_tmp3, C_ka1)*(pow(residual_tmp3, C_ka2) + s_t(-1))/mu_c;
      const s_t value_coeff1 = -porosity*residual_tmp5*(-p_c*(residual_tmp0*residual_tmp3 + residual_tmp4) + p_c_old*(residual_tmp0*pow(pow(residual_tmp1*(p_c_old - p_w_old), m) + s_t(1), residual_tmp2) + residual_tmp4))/dt;
      const s_t grad_coeff1_0 = residual_tmp6*(-K_0*p_c_grad_0 - K_1*p_c_grad_1);
      const s_t grad_coeff1_1 = residual_tmp6*(-K_2*p_c_grad_0 - K_3*p_c_grad_1);
      value_coeff1_values[0] = value_coeff1;
      grad_coeff1_0_values[0] = grad_coeff1_0;
      grad_coeff1_1_values[0] = grad_coeff1_1;
    }
    for (int test = 0; test < NS; ++test) {
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        const s_t adj0 = adjugate[0][goff];
        const s_t adj1 = adjugate[1][goff];
        const s_t adj2 = adjugate[2][goff];
        const s_t adj3 = adjugate[3][goff];
        const s_t test_value = shape[q * NS + test];
        const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj2) / det;
        const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj3) / det;
        output[test * NC + 1][0] += q_weight[q] * det * (value_coeff1_values[0] * test_value + grad_coeff1_0_values[0] * test_grad0 + grad_coeff1_1_values[0] * test_grad1);
      }
    }
  }
}


} // namespace codegen
} // namespace sfem

#endif
