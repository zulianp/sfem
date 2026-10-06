#ifndef TWO_PHASE_FLOW_FORM_2_P_W_P_W_D3_SIMPLEX_LOCAL_HPP
#define TWO_PHASE_FLOW_FORM_2_P_W_P_W_D3_SIMPLEX_LOCAL_HPP

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
__host__ __device__ __forceinline__ void two_phase_flow_form_2_p_w_p_w_d3_simplex_jacobian_action_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[2 * NS],
    const s_t *const RSTR direction[2 * NS],
    const s_t C_kw1,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t K_4,
    const s_t K_5,
    const s_t K_6,
    const s_t K_7,
    const s_t K_8,
    const s_t P_r,
    const s_t S_res,
    const s_t dt,
    const s_t kappa_T,
    const s_t m,
    const s_t mu_w,
    const s_t p_wr,
    const s_t porosity,
    const s_t rho_w0,
    s_t *const RSTR output[2 * NS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t p_w_values[VS];
    s_t p_w_grad_0_ref_values[VS];
    s_t p_w_grad_1_ref_values[VS];
    s_t p_w_grad_2_ref_values[VS];
    s_t p_w_direction_values[VS];
    s_t p_w_direction_grad_0_ref_values[VS];
    s_t p_w_direction_grad_1_ref_values[VS];
    s_t p_w_direction_grad_2_ref_values[VS];
    s_t p_c_values[VS];
    s_t value_coeff0_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff0_2_values[VS];
    {
      p_w_values[0] = s_t(0);
      p_w_grad_0_ref_values[0] = s_t(0);
      p_w_grad_1_ref_values[0] = s_t(0);
      p_w_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC][0];
        p_w_values[0] += coeff * shape[q * NS + trial];
        p_w_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        p_w_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        p_w_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      p_w_direction_values[0] = s_t(0);
      p_w_direction_grad_0_ref_values[0] = s_t(0);
      p_w_direction_grad_1_ref_values[0] = s_t(0);
      p_w_direction_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = direction[trial * NC][0];
        p_w_direction_values[0] += coeff * shape[q * NS + trial];
        p_w_direction_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        p_w_direction_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        p_w_direction_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      p_c_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 1][0];
        p_c_values[0] += coeff * shape[q * NS + trial];
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
      const s_t p_w = p_w_values[0];
      const s_t p_w_grad_0_ref = p_w_grad_0_ref_values[0];
      const s_t p_w_grad_1_ref = p_w_grad_1_ref_values[0];
      const s_t p_w_grad_2_ref = p_w_grad_2_ref_values[0];
      const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
      const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
      const s_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
      const s_t p_w_direction = p_w_direction_values[0];
      const s_t p_w_direction_grad_0_ref = p_w_direction_grad_0_ref_values[0];
      const s_t p_w_direction_grad_1_ref = p_w_direction_grad_1_ref_values[0];
      const s_t p_w_direction_grad_2_ref = p_w_direction_grad_2_ref_values[0];
      const s_t p_w_direction_grad_0 = (p_w_direction_grad_0_ref * adj0 + p_w_direction_grad_1_ref * adj3 + p_w_direction_grad_2_ref * adj6) / det;
      const s_t p_w_direction_grad_1 = (p_w_direction_grad_0_ref * adj1 + p_w_direction_grad_1_ref * adj4 + p_w_direction_grad_2_ref * adj7) / det;
      const s_t p_w_direction_grad_2 = (p_w_direction_grad_0_ref * adj2 + p_w_direction_grad_1_ref * adj5 + p_w_direction_grad_2_ref * adj8) / det;
      const s_t p_c = p_c_values[0];
      const s_t residual_tmp0 = S_res + s_t(-1);
      const s_t residual_tmp1 = p_c - p_w;
      const s_t residual_tmp2 = pow(residual_tmp1/P_r, m);
      const s_t residual_tmp3 = residual_tmp2 + s_t(1);
      const s_t residual_tmp4 = s_t(1) - m;
      const s_t residual_tmp5 = pow(residual_tmp3, residual_tmp4/m);
      const s_t residual_tmp6 = -residual_tmp0*residual_tmp5;
      const s_t residual_tmp7 = exp(kappa_T*(p_w - p_wr));
      const s_t residual_tmp8 = kappa_T*residual_tmp7;
      const s_t residual_tmp9 = residual_tmp2*residual_tmp4*residual_tmp7/(residual_tmp1*residual_tmp3);
      const s_t residual_tmp10 = p_w_direction*rho_w0/dt;
      const s_t residual_tmp11 = pow_m1(mu_w);
      const s_t residual_tmp12 = residual_tmp0*residual_tmp5;
      const s_t residual_tmp13 = S_res - residual_tmp12;
      const s_t residual_tmp14 = pow(residual_tmp13, pow_m1(C_kw1));
      const s_t residual_tmp15 = s_t(1) - residual_tmp14;
      const s_t residual_tmp16 = pow(residual_tmp15, C_kw1);
      const s_t residual_tmp17 = residual_tmp16 + s_t(-1);
      const s_t residual_tmp18 = pow_2(residual_tmp17);
      const s_t residual_tmp19 = sqrt(residual_tmp13);
      const s_t residual_tmp20 = residual_tmp18*residual_tmp19;
      const s_t residual_tmp21 = residual_tmp11*residual_tmp20*residual_tmp7*rho_w0;
      const s_t residual_tmp22 = p_w_direction_grad_0*residual_tmp21;
      const s_t residual_tmp23 = p_w_direction_grad_1*residual_tmp21;
      const s_t residual_tmp24 = p_w_direction_grad_2*residual_tmp21;
      const s_t residual_tmp25 = dt*(K_0*p_w_grad_0 + K_1*p_w_grad_1 + K_2*p_w_grad_2);
      const s_t residual_tmp26 = residual_tmp20*residual_tmp8;
      const s_t residual_tmp27 = residual_tmp12*residual_tmp9/residual_tmp19;
      const s_t residual_tmp28 = residual_tmp25*residual_tmp27;
      const s_t residual_tmp29 = ((s_t(1) / s_t(2)))*residual_tmp18;
      const s_t residual_tmp30 = s_t(2)*residual_tmp14*residual_tmp16*residual_tmp17/residual_tmp15;
      const s_t residual_tmp31 = residual_tmp10*residual_tmp11;
      const s_t residual_tmp32 = K_3*p_w_grad_0 + K_4*p_w_grad_1 + K_5*p_w_grad_2;
      const s_t residual_tmp33 = dt*residual_tmp26;
      const s_t residual_tmp34 = dt*residual_tmp27;
      const s_t residual_tmp35 = residual_tmp29*residual_tmp34;
      const s_t residual_tmp36 = residual_tmp30*residual_tmp34;
      const s_t residual_tmp37 = K_6*p_w_grad_0 + K_7*p_w_grad_1 + K_8*p_w_grad_2;
      const s_t value_coeff0 = porosity*residual_tmp10*(-residual_tmp6*residual_tmp9 + residual_tmp8*(S_res + residual_tmp6));
      const s_t grad_coeff0_0 = K_0*residual_tmp22 + K_1*residual_tmp23 + K_2*residual_tmp24 + residual_tmp31*(residual_tmp25*residual_tmp26 + residual_tmp28*residual_tmp29 - residual_tmp28*residual_tmp30);
      const s_t grad_coeff0_1 = K_3*residual_tmp22 + K_4*residual_tmp23 + K_5*residual_tmp24 + residual_tmp31*(residual_tmp32*residual_tmp33 + residual_tmp32*residual_tmp35 - residual_tmp32*residual_tmp36);
      const s_t grad_coeff0_2 = K_6*residual_tmp22 + K_7*residual_tmp23 + K_8*residual_tmp24 + residual_tmp31*(residual_tmp33*residual_tmp37 + residual_tmp35*residual_tmp37 - residual_tmp36*residual_tmp37);
      value_coeff0_values[0] = value_coeff0;
      grad_coeff0_0_values[0] = grad_coeff0_0;
      grad_coeff0_1_values[0] = grad_coeff0_1;
      grad_coeff0_2_values[0] = grad_coeff0_2;
    }
    for (int test = 0; test < NS; ++test) {
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
        const s_t test_value = shape[q * NS + test];
        const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj3 + grad_ref_z[q * NS + test] * adj6) / det;
        const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj4 + grad_ref_z[q * NS + test] * adj7) / det;
        const s_t test_grad2 = (grad_ref_x[q * NS + test] * adj2 + grad_ref_y[q * NS + test] * adj5 + grad_ref_z[q * NS + test] * adj8) / det;
        output[test * NC][0] += q_weight[q] * det * (value_coeff0_values[0] * test_value + grad_coeff0_0_values[0] * test_grad0 + grad_coeff0_1_values[0] * test_grad1 + grad_coeff0_2_values[0] * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void two_phase_flow_form_2_p_w_p_w_d3_simplex_jacobian_action_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t current[2 * NS][VS],
    const s_t direction[2 * NS][VS],
    const s_t C_kw1,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t K_4,
    const s_t K_5,
    const s_t K_6,
    const s_t K_7,
    const s_t K_8,
    const s_t P_r,
    const s_t S_res,
    const s_t dt,
    const s_t kappa_T,
    const s_t m,
    const s_t mu_w,
    const s_t p_wr,
    const s_t porosity,
    const s_t rho_w0,
    s_t output[2 * NS][VS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t p_w_values[VS];
    s_t p_w_grad_0_ref_values[VS];
    s_t p_w_grad_1_ref_values[VS];
    s_t p_w_grad_2_ref_values[VS];
    s_t p_w_direction_values[VS];
    s_t p_w_direction_grad_0_ref_values[VS];
    s_t p_w_direction_grad_1_ref_values[VS];
    s_t p_w_direction_grad_2_ref_values[VS];
    s_t p_c_values[VS];
    s_t value_coeff0_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff0_2_values[VS];
    {
      p_w_values[0] = s_t(0);
      p_w_grad_0_ref_values[0] = s_t(0);
      p_w_grad_1_ref_values[0] = s_t(0);
      p_w_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC][0];
        p_w_values[0] += coeff * shape[q * NS + trial];
        p_w_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        p_w_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        p_w_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      p_w_direction_values[0] = s_t(0);
      p_w_direction_grad_0_ref_values[0] = s_t(0);
      p_w_direction_grad_1_ref_values[0] = s_t(0);
      p_w_direction_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = direction[trial * NC][0];
        p_w_direction_values[0] += coeff * shape[q * NS + trial];
        p_w_direction_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        p_w_direction_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        p_w_direction_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      p_c_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 1][0];
        p_c_values[0] += coeff * shape[q * NS + trial];
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
      const s_t p_w = p_w_values[0];
      const s_t p_w_grad_0_ref = p_w_grad_0_ref_values[0];
      const s_t p_w_grad_1_ref = p_w_grad_1_ref_values[0];
      const s_t p_w_grad_2_ref = p_w_grad_2_ref_values[0];
      const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
      const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
      const s_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
      const s_t p_w_direction = p_w_direction_values[0];
      const s_t p_w_direction_grad_0_ref = p_w_direction_grad_0_ref_values[0];
      const s_t p_w_direction_grad_1_ref = p_w_direction_grad_1_ref_values[0];
      const s_t p_w_direction_grad_2_ref = p_w_direction_grad_2_ref_values[0];
      const s_t p_w_direction_grad_0 = (p_w_direction_grad_0_ref * adj0 + p_w_direction_grad_1_ref * adj3 + p_w_direction_grad_2_ref * adj6) / det;
      const s_t p_w_direction_grad_1 = (p_w_direction_grad_0_ref * adj1 + p_w_direction_grad_1_ref * adj4 + p_w_direction_grad_2_ref * adj7) / det;
      const s_t p_w_direction_grad_2 = (p_w_direction_grad_0_ref * adj2 + p_w_direction_grad_1_ref * adj5 + p_w_direction_grad_2_ref * adj8) / det;
      const s_t p_c = p_c_values[0];
      const s_t residual_tmp0 = S_res + s_t(-1);
      const s_t residual_tmp1 = p_c - p_w;
      const s_t residual_tmp2 = pow(residual_tmp1/P_r, m);
      const s_t residual_tmp3 = residual_tmp2 + s_t(1);
      const s_t residual_tmp4 = s_t(1) - m;
      const s_t residual_tmp5 = pow(residual_tmp3, residual_tmp4/m);
      const s_t residual_tmp6 = -residual_tmp0*residual_tmp5;
      const s_t residual_tmp7 = exp(kappa_T*(p_w - p_wr));
      const s_t residual_tmp8 = kappa_T*residual_tmp7;
      const s_t residual_tmp9 = residual_tmp2*residual_tmp4*residual_tmp7/(residual_tmp1*residual_tmp3);
      const s_t residual_tmp10 = p_w_direction*rho_w0/dt;
      const s_t residual_tmp11 = pow_m1(mu_w);
      const s_t residual_tmp12 = residual_tmp0*residual_tmp5;
      const s_t residual_tmp13 = S_res - residual_tmp12;
      const s_t residual_tmp14 = pow(residual_tmp13, pow_m1(C_kw1));
      const s_t residual_tmp15 = s_t(1) - residual_tmp14;
      const s_t residual_tmp16 = pow(residual_tmp15, C_kw1);
      const s_t residual_tmp17 = residual_tmp16 + s_t(-1);
      const s_t residual_tmp18 = pow_2(residual_tmp17);
      const s_t residual_tmp19 = sqrt(residual_tmp13);
      const s_t residual_tmp20 = residual_tmp18*residual_tmp19;
      const s_t residual_tmp21 = residual_tmp11*residual_tmp20*residual_tmp7*rho_w0;
      const s_t residual_tmp22 = p_w_direction_grad_0*residual_tmp21;
      const s_t residual_tmp23 = p_w_direction_grad_1*residual_tmp21;
      const s_t residual_tmp24 = p_w_direction_grad_2*residual_tmp21;
      const s_t residual_tmp25 = dt*(K_0*p_w_grad_0 + K_1*p_w_grad_1 + K_2*p_w_grad_2);
      const s_t residual_tmp26 = residual_tmp20*residual_tmp8;
      const s_t residual_tmp27 = residual_tmp12*residual_tmp9/residual_tmp19;
      const s_t residual_tmp28 = residual_tmp25*residual_tmp27;
      const s_t residual_tmp29 = ((s_t(1) / s_t(2)))*residual_tmp18;
      const s_t residual_tmp30 = s_t(2)*residual_tmp14*residual_tmp16*residual_tmp17/residual_tmp15;
      const s_t residual_tmp31 = residual_tmp10*residual_tmp11;
      const s_t residual_tmp32 = K_3*p_w_grad_0 + K_4*p_w_grad_1 + K_5*p_w_grad_2;
      const s_t residual_tmp33 = dt*residual_tmp26;
      const s_t residual_tmp34 = dt*residual_tmp27;
      const s_t residual_tmp35 = residual_tmp29*residual_tmp34;
      const s_t residual_tmp36 = residual_tmp30*residual_tmp34;
      const s_t residual_tmp37 = K_6*p_w_grad_0 + K_7*p_w_grad_1 + K_8*p_w_grad_2;
      const s_t value_coeff0 = porosity*residual_tmp10*(-residual_tmp6*residual_tmp9 + residual_tmp8*(S_res + residual_tmp6));
      const s_t grad_coeff0_0 = K_0*residual_tmp22 + K_1*residual_tmp23 + K_2*residual_tmp24 + residual_tmp31*(residual_tmp25*residual_tmp26 + residual_tmp28*residual_tmp29 - residual_tmp28*residual_tmp30);
      const s_t grad_coeff0_1 = K_3*residual_tmp22 + K_4*residual_tmp23 + K_5*residual_tmp24 + residual_tmp31*(residual_tmp32*residual_tmp33 + residual_tmp32*residual_tmp35 - residual_tmp32*residual_tmp36);
      const s_t grad_coeff0_2 = K_6*residual_tmp22 + K_7*residual_tmp23 + K_8*residual_tmp24 + residual_tmp31*(residual_tmp33*residual_tmp37 + residual_tmp35*residual_tmp37 - residual_tmp36*residual_tmp37);
      value_coeff0_values[0] = value_coeff0;
      grad_coeff0_0_values[0] = grad_coeff0_0;
      grad_coeff0_1_values[0] = grad_coeff0_1;
      grad_coeff0_2_values[0] = grad_coeff0_2;
    }
    for (int test = 0; test < NS; ++test) {
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
        const s_t test_value = shape[q * NS + test];
        const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj3 + grad_ref_z[q * NS + test] * adj6) / det;
        const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj4 + grad_ref_z[q * NS + test] * adj7) / det;
        const s_t test_grad2 = (grad_ref_x[q * NS + test] * adj2 + grad_ref_y[q * NS + test] * adj5 + grad_ref_z[q * NS + test] * adj8) / det;
        output[test * NC][0] += q_weight[q] * det * (value_coeff0_values[0] * test_value + grad_coeff0_0_values[0] * test_grad0 + grad_coeff0_1_values[0] * test_grad1 + grad_coeff0_2_values[0] * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void two_phase_flow_form_2_p_w_p_w_d3_simplex_tet4_jacobian_action_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[2 * NS],
    const s_t *const RSTR direction[2 * NS],
    const s_t C_kw1,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t K_4,
    const s_t K_5,
    const s_t K_6,
    const s_t K_7,
    const s_t K_8,
    const s_t P_r,
    const s_t S_res,
    const s_t dt,
    const s_t kappa_T,
    const s_t m,
    const s_t mu_w,
    const s_t p_wr,
    const s_t porosity,
    const s_t rho_w0,
    s_t *const RSTR output[2 * NS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t p_w_values[VS];
    s_t p_w_grad_0_ref_values[VS];
    s_t p_w_grad_1_ref_values[VS];
    s_t p_w_grad_2_ref_values[VS];
    s_t p_w_direction_values[VS];
    s_t p_w_direction_grad_0_ref_values[VS];
    s_t p_w_direction_grad_1_ref_values[VS];
    s_t p_w_direction_grad_2_ref_values[VS];
    s_t p_c_values[VS];
    s_t value_coeff0_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff0_2_values[VS];
    {
      p_w_values[0] = s_t(0);
      p_w_grad_0_ref_values[0] = s_t(0);
      p_w_grad_1_ref_values[0] = s_t(0);
      p_w_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC][0];
        p_w_values[0] += coeff * shape[q * NS + trial];
        p_w_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        p_w_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        p_w_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      p_w_direction_values[0] = s_t(0);
      p_w_direction_grad_0_ref_values[0] = s_t(0);
      p_w_direction_grad_1_ref_values[0] = s_t(0);
      p_w_direction_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = direction[trial * NC][0];
        p_w_direction_values[0] += coeff * shape[q * NS + trial];
        p_w_direction_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        p_w_direction_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        p_w_direction_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      p_c_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 1][0];
        p_c_values[0] += coeff * shape[q * NS + trial];
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
      const s_t p_w = p_w_values[0];
      const s_t p_w_grad_0_ref = p_w_grad_0_ref_values[0];
      const s_t p_w_grad_1_ref = p_w_grad_1_ref_values[0];
      const s_t p_w_grad_2_ref = p_w_grad_2_ref_values[0];
      const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
      const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
      const s_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
      const s_t p_w_direction = p_w_direction_values[0];
      const s_t p_w_direction_grad_0_ref = p_w_direction_grad_0_ref_values[0];
      const s_t p_w_direction_grad_1_ref = p_w_direction_grad_1_ref_values[0];
      const s_t p_w_direction_grad_2_ref = p_w_direction_grad_2_ref_values[0];
      const s_t p_w_direction_grad_0 = (p_w_direction_grad_0_ref * adj0 + p_w_direction_grad_1_ref * adj3 + p_w_direction_grad_2_ref * adj6) / det;
      const s_t p_w_direction_grad_1 = (p_w_direction_grad_0_ref * adj1 + p_w_direction_grad_1_ref * adj4 + p_w_direction_grad_2_ref * adj7) / det;
      const s_t p_w_direction_grad_2 = (p_w_direction_grad_0_ref * adj2 + p_w_direction_grad_1_ref * adj5 + p_w_direction_grad_2_ref * adj8) / det;
      const s_t p_c = p_c_values[0];
      const s_t residual_tmp0 = S_res + s_t(-1);
      const s_t residual_tmp1 = p_c - p_w;
      const s_t residual_tmp2 = pow(residual_tmp1/P_r, m);
      const s_t residual_tmp3 = residual_tmp2 + s_t(1);
      const s_t residual_tmp4 = s_t(1) - m;
      const s_t residual_tmp5 = pow(residual_tmp3, residual_tmp4/m);
      const s_t residual_tmp6 = -residual_tmp0*residual_tmp5;
      const s_t residual_tmp7 = exp(kappa_T*(p_w - p_wr));
      const s_t residual_tmp8 = kappa_T*residual_tmp7;
      const s_t residual_tmp9 = residual_tmp2*residual_tmp4*residual_tmp7/(residual_tmp1*residual_tmp3);
      const s_t residual_tmp10 = p_w_direction*rho_w0/dt;
      const s_t residual_tmp11 = pow_m1(mu_w);
      const s_t residual_tmp12 = residual_tmp0*residual_tmp5;
      const s_t residual_tmp13 = S_res - residual_tmp12;
      const s_t residual_tmp14 = pow(residual_tmp13, pow_m1(C_kw1));
      const s_t residual_tmp15 = s_t(1) - residual_tmp14;
      const s_t residual_tmp16 = pow(residual_tmp15, C_kw1);
      const s_t residual_tmp17 = residual_tmp16 + s_t(-1);
      const s_t residual_tmp18 = pow_2(residual_tmp17);
      const s_t residual_tmp19 = sqrt(residual_tmp13);
      const s_t residual_tmp20 = residual_tmp18*residual_tmp19;
      const s_t residual_tmp21 = residual_tmp11*residual_tmp20*residual_tmp7*rho_w0;
      const s_t residual_tmp22 = p_w_direction_grad_0*residual_tmp21;
      const s_t residual_tmp23 = p_w_direction_grad_1*residual_tmp21;
      const s_t residual_tmp24 = p_w_direction_grad_2*residual_tmp21;
      const s_t residual_tmp25 = dt*(K_0*p_w_grad_0 + K_1*p_w_grad_1 + K_2*p_w_grad_2);
      const s_t residual_tmp26 = residual_tmp20*residual_tmp8;
      const s_t residual_tmp27 = residual_tmp12*residual_tmp9/residual_tmp19;
      const s_t residual_tmp28 = residual_tmp25*residual_tmp27;
      const s_t residual_tmp29 = ((s_t(1) / s_t(2)))*residual_tmp18;
      const s_t residual_tmp30 = s_t(2)*residual_tmp14*residual_tmp16*residual_tmp17/residual_tmp15;
      const s_t residual_tmp31 = residual_tmp10*residual_tmp11;
      const s_t residual_tmp32 = K_3*p_w_grad_0 + K_4*p_w_grad_1 + K_5*p_w_grad_2;
      const s_t residual_tmp33 = dt*residual_tmp26;
      const s_t residual_tmp34 = dt*residual_tmp27;
      const s_t residual_tmp35 = residual_tmp29*residual_tmp34;
      const s_t residual_tmp36 = residual_tmp30*residual_tmp34;
      const s_t residual_tmp37 = K_6*p_w_grad_0 + K_7*p_w_grad_1 + K_8*p_w_grad_2;
      const s_t value_coeff0 = porosity*residual_tmp10*(-residual_tmp6*residual_tmp9 + residual_tmp8*(S_res + residual_tmp6));
      const s_t grad_coeff0_0 = K_0*residual_tmp22 + K_1*residual_tmp23 + K_2*residual_tmp24 + residual_tmp31*(residual_tmp25*residual_tmp26 + residual_tmp28*residual_tmp29 - residual_tmp28*residual_tmp30);
      const s_t grad_coeff0_1 = K_3*residual_tmp22 + K_4*residual_tmp23 + K_5*residual_tmp24 + residual_tmp31*(residual_tmp32*residual_tmp33 + residual_tmp32*residual_tmp35 - residual_tmp32*residual_tmp36);
      const s_t grad_coeff0_2 = K_6*residual_tmp22 + K_7*residual_tmp23 + K_8*residual_tmp24 + residual_tmp31*(residual_tmp33*residual_tmp37 + residual_tmp35*residual_tmp37 - residual_tmp36*residual_tmp37);
      value_coeff0_values[0] = value_coeff0;
      grad_coeff0_0_values[0] = grad_coeff0_0;
      grad_coeff0_1_values[0] = grad_coeff0_1;
      grad_coeff0_2_values[0] = grad_coeff0_2;
    }
    for (int test = 0; test < NS; ++test) {
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
        const s_t test_value = shape[q * NS + test];
        const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj3 + grad_ref_z[q * NS + test] * adj6) / det;
        const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj4 + grad_ref_z[q * NS + test] * adj7) / det;
        const s_t test_grad2 = (grad_ref_x[q * NS + test] * adj2 + grad_ref_y[q * NS + test] * adj5 + grad_ref_z[q * NS + test] * adj8) / det;
        output[test * NC][0] += q_weight[q] * det * (value_coeff0_values[0] * test_value + grad_coeff0_0_values[0] * test_grad0 + grad_coeff0_1_values[0] * test_grad1 + grad_coeff0_2_values[0] * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void two_phase_flow_form_2_p_w_p_w_d3_simplex_tet4_jacobian_action_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t current[2 * NS][VS],
    const s_t direction[2 * NS][VS],
    const s_t C_kw1,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t K_4,
    const s_t K_5,
    const s_t K_6,
    const s_t K_7,
    const s_t K_8,
    const s_t P_r,
    const s_t S_res,
    const s_t dt,
    const s_t kappa_T,
    const s_t m,
    const s_t mu_w,
    const s_t p_wr,
    const s_t porosity,
    const s_t rho_w0,
    s_t output[2 * NS][VS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t p_w_values[VS];
    s_t p_w_grad_0_ref_values[VS];
    s_t p_w_grad_1_ref_values[VS];
    s_t p_w_grad_2_ref_values[VS];
    s_t p_w_direction_values[VS];
    s_t p_w_direction_grad_0_ref_values[VS];
    s_t p_w_direction_grad_1_ref_values[VS];
    s_t p_w_direction_grad_2_ref_values[VS];
    s_t p_c_values[VS];
    s_t value_coeff0_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff0_2_values[VS];
    {
      p_w_values[0] = s_t(0);
      p_w_grad_0_ref_values[0] = s_t(0);
      p_w_grad_1_ref_values[0] = s_t(0);
      p_w_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC][0];
        p_w_values[0] += coeff * shape[q * NS + trial];
        p_w_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        p_w_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        p_w_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      p_w_direction_values[0] = s_t(0);
      p_w_direction_grad_0_ref_values[0] = s_t(0);
      p_w_direction_grad_1_ref_values[0] = s_t(0);
      p_w_direction_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = direction[trial * NC][0];
        p_w_direction_values[0] += coeff * shape[q * NS + trial];
        p_w_direction_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        p_w_direction_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        p_w_direction_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      p_c_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 1][0];
        p_c_values[0] += coeff * shape[q * NS + trial];
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
      const s_t p_w = p_w_values[0];
      const s_t p_w_grad_0_ref = p_w_grad_0_ref_values[0];
      const s_t p_w_grad_1_ref = p_w_grad_1_ref_values[0];
      const s_t p_w_grad_2_ref = p_w_grad_2_ref_values[0];
      const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
      const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
      const s_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
      const s_t p_w_direction = p_w_direction_values[0];
      const s_t p_w_direction_grad_0_ref = p_w_direction_grad_0_ref_values[0];
      const s_t p_w_direction_grad_1_ref = p_w_direction_grad_1_ref_values[0];
      const s_t p_w_direction_grad_2_ref = p_w_direction_grad_2_ref_values[0];
      const s_t p_w_direction_grad_0 = (p_w_direction_grad_0_ref * adj0 + p_w_direction_grad_1_ref * adj3 + p_w_direction_grad_2_ref * adj6) / det;
      const s_t p_w_direction_grad_1 = (p_w_direction_grad_0_ref * adj1 + p_w_direction_grad_1_ref * adj4 + p_w_direction_grad_2_ref * adj7) / det;
      const s_t p_w_direction_grad_2 = (p_w_direction_grad_0_ref * adj2 + p_w_direction_grad_1_ref * adj5 + p_w_direction_grad_2_ref * adj8) / det;
      const s_t p_c = p_c_values[0];
      const s_t residual_tmp0 = S_res + s_t(-1);
      const s_t residual_tmp1 = p_c - p_w;
      const s_t residual_tmp2 = pow(residual_tmp1/P_r, m);
      const s_t residual_tmp3 = residual_tmp2 + s_t(1);
      const s_t residual_tmp4 = s_t(1) - m;
      const s_t residual_tmp5 = pow(residual_tmp3, residual_tmp4/m);
      const s_t residual_tmp6 = -residual_tmp0*residual_tmp5;
      const s_t residual_tmp7 = exp(kappa_T*(p_w - p_wr));
      const s_t residual_tmp8 = kappa_T*residual_tmp7;
      const s_t residual_tmp9 = residual_tmp2*residual_tmp4*residual_tmp7/(residual_tmp1*residual_tmp3);
      const s_t residual_tmp10 = p_w_direction*rho_w0/dt;
      const s_t residual_tmp11 = pow_m1(mu_w);
      const s_t residual_tmp12 = residual_tmp0*residual_tmp5;
      const s_t residual_tmp13 = S_res - residual_tmp12;
      const s_t residual_tmp14 = pow(residual_tmp13, pow_m1(C_kw1));
      const s_t residual_tmp15 = s_t(1) - residual_tmp14;
      const s_t residual_tmp16 = pow(residual_tmp15, C_kw1);
      const s_t residual_tmp17 = residual_tmp16 + s_t(-1);
      const s_t residual_tmp18 = pow_2(residual_tmp17);
      const s_t residual_tmp19 = sqrt(residual_tmp13);
      const s_t residual_tmp20 = residual_tmp18*residual_tmp19;
      const s_t residual_tmp21 = residual_tmp11*residual_tmp20*residual_tmp7*rho_w0;
      const s_t residual_tmp22 = p_w_direction_grad_0*residual_tmp21;
      const s_t residual_tmp23 = p_w_direction_grad_1*residual_tmp21;
      const s_t residual_tmp24 = p_w_direction_grad_2*residual_tmp21;
      const s_t residual_tmp25 = dt*(K_0*p_w_grad_0 + K_1*p_w_grad_1 + K_2*p_w_grad_2);
      const s_t residual_tmp26 = residual_tmp20*residual_tmp8;
      const s_t residual_tmp27 = residual_tmp12*residual_tmp9/residual_tmp19;
      const s_t residual_tmp28 = residual_tmp25*residual_tmp27;
      const s_t residual_tmp29 = ((s_t(1) / s_t(2)))*residual_tmp18;
      const s_t residual_tmp30 = s_t(2)*residual_tmp14*residual_tmp16*residual_tmp17/residual_tmp15;
      const s_t residual_tmp31 = residual_tmp10*residual_tmp11;
      const s_t residual_tmp32 = K_3*p_w_grad_0 + K_4*p_w_grad_1 + K_5*p_w_grad_2;
      const s_t residual_tmp33 = dt*residual_tmp26;
      const s_t residual_tmp34 = dt*residual_tmp27;
      const s_t residual_tmp35 = residual_tmp29*residual_tmp34;
      const s_t residual_tmp36 = residual_tmp30*residual_tmp34;
      const s_t residual_tmp37 = K_6*p_w_grad_0 + K_7*p_w_grad_1 + K_8*p_w_grad_2;
      const s_t value_coeff0 = porosity*residual_tmp10*(-residual_tmp6*residual_tmp9 + residual_tmp8*(S_res + residual_tmp6));
      const s_t grad_coeff0_0 = K_0*residual_tmp22 + K_1*residual_tmp23 + K_2*residual_tmp24 + residual_tmp31*(residual_tmp25*residual_tmp26 + residual_tmp28*residual_tmp29 - residual_tmp28*residual_tmp30);
      const s_t grad_coeff0_1 = K_3*residual_tmp22 + K_4*residual_tmp23 + K_5*residual_tmp24 + residual_tmp31*(residual_tmp32*residual_tmp33 + residual_tmp32*residual_tmp35 - residual_tmp32*residual_tmp36);
      const s_t grad_coeff0_2 = K_6*residual_tmp22 + K_7*residual_tmp23 + K_8*residual_tmp24 + residual_tmp31*(residual_tmp33*residual_tmp37 + residual_tmp35*residual_tmp37 - residual_tmp36*residual_tmp37);
      value_coeff0_values[0] = value_coeff0;
      grad_coeff0_0_values[0] = grad_coeff0_0;
      grad_coeff0_1_values[0] = grad_coeff0_1;
      grad_coeff0_2_values[0] = grad_coeff0_2;
    }
    for (int test = 0; test < NS; ++test) {
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
        const s_t test_value = shape[q * NS + test];
        const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj3 + grad_ref_z[q * NS + test] * adj6) / det;
        const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj4 + grad_ref_z[q * NS + test] * adj7) / det;
        const s_t test_grad2 = (grad_ref_x[q * NS + test] * adj2 + grad_ref_y[q * NS + test] * adj5 + grad_ref_z[q * NS + test] * adj8) / det;
        output[test * NC][0] += q_weight[q] * det * (value_coeff0_values[0] * test_value + grad_coeff0_0_values[0] * test_grad0 + grad_coeff0_1_values[0] * test_grad1 + grad_coeff0_2_values[0] * test_grad2);
      }
    }
  }
}

} // namespace codegen
} // namespace sfem

#endif
