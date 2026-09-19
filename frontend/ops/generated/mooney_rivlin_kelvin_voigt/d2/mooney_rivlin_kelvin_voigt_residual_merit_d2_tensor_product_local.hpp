#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_RESIDUAL_MERIT_D2_TENSOR_PRODUCT_LOCAL_HPP
#define MOONEY_RIVLIN_KELVIN_VOIGT_RESIDUAL_MERIT_D2_TENSOR_PRODUCT_LOCAL_HPP

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
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_residual_merit_d2_tensor_product_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR q_weight_1d,
    const s_t *const RSTR current[2 * NS],
    const s_t *const RSTR previous[2 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t *const RSTR output[2 * NS]
) {
  static constexpr int ND = 2;
  static constexpr int NC = 2;
  s_t current_value[NC * NQ * VS];
  s_t current_grad_ref[NC * NQ * ND * VS];
  tensor_evaluate<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, grad_1d, current, current_value, current_grad_ref);
  s_t previous_value[NC * NQ * VS];
  s_t previous_grad_ref[NC * NQ * ND * VS];
  tensor_evaluate<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, grad_1d, previous, previous_value, previous_grad_ref);
  s_t value_coeff[NC * NQ * VS];
  s_t grad_coeff_ref[NC * NQ * ND * VS];
  static constexpr int NQ1 = integer_root(NQ, ND);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = q / NQ1;
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
    const s_t *const RSTR det_q = determinant + q * geometry_stride;
    const s_t *const RSTR adj_q0 = adjugate[0] + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adjugate[1] + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adjugate[2] + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adjugate[3] + q * geometry_stride;
    const s_t *const RSTR current_grad_ref_q0_0 = &current_grad_ref[(q * ND) * VS];
    const s_t *const RSTR current_grad_ref_q0_1 = &current_grad_ref[(q * ND + 1) * VS];
    const s_t *const RSTR previous_grad_ref_q0_0 = &previous_grad_ref[(q * ND) * VS];
    const s_t *const RSTR previous_grad_ref_q0_1 = &previous_grad_ref[(q * ND + 1) * VS];
    const s_t *const RSTR current_grad_ref_q1_0 = &current_grad_ref[((NQ + q) * ND) * VS];
    const s_t *const RSTR current_grad_ref_q1_1 = &current_grad_ref[((NQ + q) * ND + 1) * VS];
    const s_t *const RSTR previous_grad_ref_q1_0 = &previous_grad_ref[((NQ + q) * ND) * VS];
    const s_t *const RSTR previous_grad_ref_q1_1 = &previous_grad_ref[((NQ + q) * ND + 1) * VS];
    s_t *const RSTR value_coeff_q0 = &value_coeff[q * VS];
    s_t *const RSTR grad_coeff_ref_q0_0 = &grad_coeff_ref[(q * ND) * VS];
    s_t *const RSTR grad_coeff_ref_q0_1 = &grad_coeff_ref[(q * ND + 1) * VS];
    s_t *const RSTR value_coeff_q1 = &value_coeff[(NQ + q) * VS];
    s_t *const RSTR grad_coeff_ref_q1_0 = &grad_coeff_ref[((NQ + q) * ND) * VS];
    s_t *const RSTR grad_coeff_ref_q1_1 = &grad_coeff_ref[((NQ + q) * ND + 1) * VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t det = det_q[lane];
      const s_t adj0 = adj_q0[lane];
      const s_t adj1 = adj_q1[lane];
      const s_t adj2 = adj_q2[lane];
      const s_t adj3 = adj_q3[lane];
      const s_t u0_grad_0_ref = current_grad_ref_q0_0[lane];
      const s_t u0_grad_1_ref = current_grad_ref_q0_1[lane];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = previous_grad_ref_q0_0[lane];
      const s_t u0_old_grad_1_ref = previous_grad_ref_q0_1[lane];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = current_grad_ref_q1_0[lane];
      const s_t u1_grad_1_ref = current_grad_ref_q1_1[lane];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = previous_grad_ref_q1_0[lane];
      const s_t u1_old_grad_1_ref = previous_grad_ref_q1_1[lane];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
      const s_t residual_tmp0 = u1_grad_1 + s_t(1);
      const s_t residual_tmp1 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp2 = u0_grad_0 + s_t(1);
      const s_t residual_tmp3 = lmbda*(residual_tmp0*residual_tmp2 - residual_tmp1 + s_t(-1));
      const s_t residual_tmp4 = residual_tmp0*u1_grad_0 + residual_tmp2*u0_grad_1;
      const s_t residual_tmp5 = s_t(2)*u0_grad_1;
      const s_t residual_tmp6 = pow_2(residual_tmp2) + pow_2(u1_grad_0);
      const s_t residual_tmp7 = s_t(2)*residual_tmp2;
      const s_t residual_tmp8 = pow_2(residual_tmp0) + pow_2(u0_grad_1);
      const s_t residual_tmp9 = residual_tmp6 + residual_tmp8;
      const s_t residual_tmp10 = pow_m1(-residual_tmp1 + residual_tmp2 + u0_grad_0*u1_grad_1 + u1_grad_1);
      const s_t residual_tmp11 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp12 = u1_grad_1*u_dt_shift + u1_old_grad_1;
      const s_t residual_tmp13 = u0_grad_1*u_dt_shift + u0_old_grad_1;
      const s_t residual_tmp14 = u1_grad_0*u_dt_shift + u1_old_grad_0;
      const s_t residual_tmp15 = eta_s*(-residual_tmp0*residual_tmp14 + residual_tmp11*u0_grad_1 + residual_tmp12*u1_grad_0 - residual_tmp13*residual_tmp2);
      const s_t residual_tmp16 = residual_tmp13*u1_grad_0;
      const s_t residual_tmp17 = residual_tmp0*residual_tmp11;
      const s_t residual_tmp18 = -residual_tmp12*residual_tmp2 + residual_tmp14*u0_grad_1;
      const s_t residual_tmp19 = eta_b*(residual_tmp16 - residual_tmp17 + residual_tmp18);
      const s_t residual_tmp20 = -residual_tmp16 + residual_tmp17 + residual_tmp18;
      const s_t residual_tmp21 = -eta_s*residual_tmp20 + residual_tmp19;
      const s_t residual_tmp22 = s_t(2)*u1_grad_0;
      const s_t residual_tmp23 = s_t(2)*residual_tmp0;
      const s_t residual_tmp24 = eta_s*residual_tmp20 + residual_tmp19;
      const s_t grad_coeff0_0 = mu*(s_t(2)*residual_tmp2*residual_tmp9 - residual_tmp4*residual_tmp5 - residual_tmp6*residual_tmp7 + s_t(4)*u0_grad_0 - s_t(6)*u1_grad_1 + s_t(-2)) + residual_tmp0*residual_tmp3 + residual_tmp10*(-residual_tmp0*residual_tmp21 + residual_tmp15*u0_grad_1);
      const s_t grad_coeff0_1 = mu*(-residual_tmp4*residual_tmp7 - residual_tmp5*residual_tmp8 + residual_tmp5*residual_tmp9 + s_t(4)*u0_grad_1 + s_t(6)*u1_grad_0) + residual_tmp10*(-residual_tmp15*residual_tmp2 + residual_tmp21*u1_grad_0) - residual_tmp3*u1_grad_0;
      const s_t grad_coeff1_0 = mu*(-residual_tmp22*residual_tmp6 + residual_tmp22*residual_tmp9 - residual_tmp23*residual_tmp4 + s_t(6)*u0_grad_1 + s_t(4)*u1_grad_0) + residual_tmp10*(-residual_tmp0*residual_tmp15 + residual_tmp24*u0_grad_1) - residual_tmp3*u0_grad_1;
      const s_t grad_coeff1_1 = mu*(s_t(2)*residual_tmp0*residual_tmp9 - residual_tmp22*residual_tmp4 - residual_tmp23*residual_tmp8 - s_t(6)*u0_grad_0 + s_t(4)*u1_grad_1 + s_t(-2)) + residual_tmp10*(residual_tmp15*u1_grad_0 - residual_tmp2*residual_tmp24) + residual_tmp2*residual_tmp3;
      value_coeff_q0[lane] = s_t(0);
      grad_coeff_ref_q0_0[lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
      grad_coeff_ref_q0_1[lane] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
      value_coeff_q1[lane] = s_t(0);
      grad_coeff_ref_q1_0[lane] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
      grad_coeff_ref_q1_1[lane] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
    }
  }
  tensor_integrate<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_residual_merit_d2_tensor_product_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR q_weight_1d,
    const s_t current[2 * NS][VS],
    const s_t previous[2 * NS][VS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t output[2 * NS][VS]
) {
  static constexpr int ND = 2;
  static constexpr int NC = 2;
  s_t current_value[NC * NQ * VS];
  s_t current_grad_ref[NC * NQ * ND * VS];
  tensor_evaluate_contiguous<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, grad_1d, current, current_value, current_grad_ref);
  s_t previous_value[NC * NQ * VS];
  s_t previous_grad_ref[NC * NQ * ND * VS];
  tensor_evaluate_contiguous<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, grad_1d, previous, previous_value, previous_grad_ref);
  s_t value_coeff[NC * NQ * VS];
  s_t grad_coeff_ref[NC * NQ * ND * VS];
  static constexpr int NQ1 = integer_root(NQ, ND);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = q / NQ1;
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
    const s_t *const RSTR det_q = determinant + q * geometry_stride;
    const s_t *const RSTR adj_q0 = adjugate[0] + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adjugate[1] + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adjugate[2] + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adjugate[3] + q * geometry_stride;
    const s_t *const RSTR current_grad_ref_q0_0 = &current_grad_ref[(q * ND) * VS];
    const s_t *const RSTR current_grad_ref_q0_1 = &current_grad_ref[(q * ND + 1) * VS];
    const s_t *const RSTR previous_grad_ref_q0_0 = &previous_grad_ref[(q * ND) * VS];
    const s_t *const RSTR previous_grad_ref_q0_1 = &previous_grad_ref[(q * ND + 1) * VS];
    const s_t *const RSTR current_grad_ref_q1_0 = &current_grad_ref[((NQ + q) * ND) * VS];
    const s_t *const RSTR current_grad_ref_q1_1 = &current_grad_ref[((NQ + q) * ND + 1) * VS];
    const s_t *const RSTR previous_grad_ref_q1_0 = &previous_grad_ref[((NQ + q) * ND) * VS];
    const s_t *const RSTR previous_grad_ref_q1_1 = &previous_grad_ref[((NQ + q) * ND + 1) * VS];
    s_t *const RSTR value_coeff_q0 = &value_coeff[q * VS];
    s_t *const RSTR grad_coeff_ref_q0_0 = &grad_coeff_ref[(q * ND) * VS];
    s_t *const RSTR grad_coeff_ref_q0_1 = &grad_coeff_ref[(q * ND + 1) * VS];
    s_t *const RSTR value_coeff_q1 = &value_coeff[(NQ + q) * VS];
    s_t *const RSTR grad_coeff_ref_q1_0 = &grad_coeff_ref[((NQ + q) * ND) * VS];
    s_t *const RSTR grad_coeff_ref_q1_1 = &grad_coeff_ref[((NQ + q) * ND + 1) * VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t det = det_q[lane];
      const s_t adj0 = adj_q0[lane];
      const s_t adj1 = adj_q1[lane];
      const s_t adj2 = adj_q2[lane];
      const s_t adj3 = adj_q3[lane];
      const s_t u0_grad_0_ref = current_grad_ref_q0_0[lane];
      const s_t u0_grad_1_ref = current_grad_ref_q0_1[lane];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = previous_grad_ref_q0_0[lane];
      const s_t u0_old_grad_1_ref = previous_grad_ref_q0_1[lane];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = current_grad_ref_q1_0[lane];
      const s_t u1_grad_1_ref = current_grad_ref_q1_1[lane];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = previous_grad_ref_q1_0[lane];
      const s_t u1_old_grad_1_ref = previous_grad_ref_q1_1[lane];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
      const s_t residual_tmp0 = u1_grad_1 + s_t(1);
      const s_t residual_tmp1 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp2 = u0_grad_0 + s_t(1);
      const s_t residual_tmp3 = lmbda*(residual_tmp0*residual_tmp2 - residual_tmp1 + s_t(-1));
      const s_t residual_tmp4 = residual_tmp0*u1_grad_0 + residual_tmp2*u0_grad_1;
      const s_t residual_tmp5 = s_t(2)*u0_grad_1;
      const s_t residual_tmp6 = pow_2(residual_tmp2) + pow_2(u1_grad_0);
      const s_t residual_tmp7 = s_t(2)*residual_tmp2;
      const s_t residual_tmp8 = pow_2(residual_tmp0) + pow_2(u0_grad_1);
      const s_t residual_tmp9 = residual_tmp6 + residual_tmp8;
      const s_t residual_tmp10 = pow_m1(-residual_tmp1 + residual_tmp2 + u0_grad_0*u1_grad_1 + u1_grad_1);
      const s_t residual_tmp11 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp12 = u1_grad_1*u_dt_shift + u1_old_grad_1;
      const s_t residual_tmp13 = u0_grad_1*u_dt_shift + u0_old_grad_1;
      const s_t residual_tmp14 = u1_grad_0*u_dt_shift + u1_old_grad_0;
      const s_t residual_tmp15 = eta_s*(-residual_tmp0*residual_tmp14 + residual_tmp11*u0_grad_1 + residual_tmp12*u1_grad_0 - residual_tmp13*residual_tmp2);
      const s_t residual_tmp16 = residual_tmp13*u1_grad_0;
      const s_t residual_tmp17 = residual_tmp0*residual_tmp11;
      const s_t residual_tmp18 = -residual_tmp12*residual_tmp2 + residual_tmp14*u0_grad_1;
      const s_t residual_tmp19 = eta_b*(residual_tmp16 - residual_tmp17 + residual_tmp18);
      const s_t residual_tmp20 = -residual_tmp16 + residual_tmp17 + residual_tmp18;
      const s_t residual_tmp21 = -eta_s*residual_tmp20 + residual_tmp19;
      const s_t residual_tmp22 = s_t(2)*u1_grad_0;
      const s_t residual_tmp23 = s_t(2)*residual_tmp0;
      const s_t residual_tmp24 = eta_s*residual_tmp20 + residual_tmp19;
      const s_t grad_coeff0_0 = mu*(s_t(2)*residual_tmp2*residual_tmp9 - residual_tmp4*residual_tmp5 - residual_tmp6*residual_tmp7 + s_t(4)*u0_grad_0 - s_t(6)*u1_grad_1 + s_t(-2)) + residual_tmp0*residual_tmp3 + residual_tmp10*(-residual_tmp0*residual_tmp21 + residual_tmp15*u0_grad_1);
      const s_t grad_coeff0_1 = mu*(-residual_tmp4*residual_tmp7 - residual_tmp5*residual_tmp8 + residual_tmp5*residual_tmp9 + s_t(4)*u0_grad_1 + s_t(6)*u1_grad_0) + residual_tmp10*(-residual_tmp15*residual_tmp2 + residual_tmp21*u1_grad_0) - residual_tmp3*u1_grad_0;
      const s_t grad_coeff1_0 = mu*(-residual_tmp22*residual_tmp6 + residual_tmp22*residual_tmp9 - residual_tmp23*residual_tmp4 + s_t(6)*u0_grad_1 + s_t(4)*u1_grad_0) + residual_tmp10*(-residual_tmp0*residual_tmp15 + residual_tmp24*u0_grad_1) - residual_tmp3*u0_grad_1;
      const s_t grad_coeff1_1 = mu*(s_t(2)*residual_tmp0*residual_tmp9 - residual_tmp22*residual_tmp4 - residual_tmp23*residual_tmp8 - s_t(6)*u0_grad_0 + s_t(4)*u1_grad_1 + s_t(-2)) + residual_tmp10*(residual_tmp15*u1_grad_0 - residual_tmp2*residual_tmp24) + residual_tmp2*residual_tmp3;
      value_coeff_q0[lane] = s_t(0);
      grad_coeff_ref_q0_0[lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
      grad_coeff_ref_q0_1[lane] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
      value_coeff_q1[lane] = s_t(0);
      grad_coeff_ref_q1_0[lane] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
      grad_coeff_ref_q1_1[lane] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
    }
  }
  tensor_integrate_contiguous<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_residual_merit_d2_tensor_product_jacobian_action_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR q_weight_1d,
    const s_t *const RSTR current[2 * NS],
    const s_t *const RSTR previous[2 * NS],
    const s_t *const RSTR direction[2 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t *const RSTR output[2 * NS]
) {
  static constexpr int ND = 2;
  static constexpr int NC = 2;
  s_t current_value[NC * NQ * VS];
  s_t current_grad_ref[NC * NQ * ND * VS];
  tensor_evaluate<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, grad_1d, current, current_value, current_grad_ref);
  s_t previous_value[NC * NQ * VS];
  s_t previous_grad_ref[NC * NQ * ND * VS];
  tensor_evaluate<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, grad_1d, previous, previous_value, previous_grad_ref);
  s_t direction_value[NC * NQ * VS];
  s_t direction_grad_ref[NC * NQ * ND * VS];
  tensor_evaluate<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, grad_1d, direction, direction_value, direction_grad_ref);
  s_t value_coeff[NC * NQ * VS];
  s_t grad_coeff_ref[NC * NQ * ND * VS];
  static constexpr int NQ1 = integer_root(NQ, ND);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = q / NQ1;
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
    const s_t *const RSTR det_q = determinant + q * geometry_stride;
    const s_t *const RSTR adj_q0 = adjugate[0] + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adjugate[1] + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adjugate[2] + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adjugate[3] + q * geometry_stride;
    const s_t *const RSTR current_grad_ref_q0_0 = &current_grad_ref[(q * ND) * VS];
    const s_t *const RSTR current_grad_ref_q0_1 = &current_grad_ref[(q * ND + 1) * VS];
    const s_t *const RSTR previous_grad_ref_q0_0 = &previous_grad_ref[(q * ND) * VS];
    const s_t *const RSTR previous_grad_ref_q0_1 = &previous_grad_ref[(q * ND + 1) * VS];
    const s_t *const RSTR direction_grad_ref_q0_0 = &direction_grad_ref[(q * ND) * VS];
    const s_t *const RSTR direction_grad_ref_q0_1 = &direction_grad_ref[(q * ND + 1) * VS];
    const s_t *const RSTR current_grad_ref_q1_0 = &current_grad_ref[((NQ + q) * ND) * VS];
    const s_t *const RSTR current_grad_ref_q1_1 = &current_grad_ref[((NQ + q) * ND + 1) * VS];
    const s_t *const RSTR previous_grad_ref_q1_0 = &previous_grad_ref[((NQ + q) * ND) * VS];
    const s_t *const RSTR previous_grad_ref_q1_1 = &previous_grad_ref[((NQ + q) * ND + 1) * VS];
    const s_t *const RSTR direction_grad_ref_q1_0 = &direction_grad_ref[((NQ + q) * ND) * VS];
    const s_t *const RSTR direction_grad_ref_q1_1 = &direction_grad_ref[((NQ + q) * ND + 1) * VS];
    s_t *const RSTR value_coeff_q0 = &value_coeff[q * VS];
    s_t *const RSTR grad_coeff_ref_q0_0 = &grad_coeff_ref[(q * ND) * VS];
    s_t *const RSTR grad_coeff_ref_q0_1 = &grad_coeff_ref[(q * ND + 1) * VS];
    s_t *const RSTR value_coeff_q1 = &value_coeff[(NQ + q) * VS];
    s_t *const RSTR grad_coeff_ref_q1_0 = &grad_coeff_ref[((NQ + q) * ND) * VS];
    s_t *const RSTR grad_coeff_ref_q1_1 = &grad_coeff_ref[((NQ + q) * ND + 1) * VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t det = det_q[lane];
      const s_t adj0 = adj_q0[lane];
      const s_t adj1 = adj_q1[lane];
      const s_t adj2 = adj_q2[lane];
      const s_t adj3 = adj_q3[lane];
      const s_t u0_grad_0_ref = current_grad_ref_q0_0[lane];
      const s_t u0_grad_1_ref = current_grad_ref_q0_1[lane];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = previous_grad_ref_q0_0[lane];
      const s_t u0_old_grad_1_ref = previous_grad_ref_q0_1[lane];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u0_direction_grad_0_ref = direction_grad_ref_q0_0[lane];
      const s_t u0_direction_grad_1_ref = direction_grad_ref_q0_1[lane];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = current_grad_ref_q1_0[lane];
      const s_t u1_grad_1_ref = current_grad_ref_q1_1[lane];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = previous_grad_ref_q1_0[lane];
      const s_t u1_old_grad_1_ref = previous_grad_ref_q1_1[lane];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
      const s_t u1_direction_grad_0_ref = direction_grad_ref_q1_0[lane];
      const s_t u1_direction_grad_1_ref = direction_grad_ref_q1_1[lane];
      const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj2) / det;
      const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj3) / det;
      const s_t residual_tmp0 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp1 = u1_grad_1 + s_t(1);
      const s_t residual_tmp2 = -residual_tmp0 + residual_tmp1 + u0_grad_0*u1_grad_1 + u0_grad_0;
      const s_t residual_tmp3 = pow_m1(residual_tmp2);
      const s_t residual_tmp4 = residual_tmp1*u_dt_shift;
      const s_t residual_tmp5 = u1_grad_1*u_dt_shift + u1_old_grad_1;
      const s_t residual_tmp6 = -residual_tmp4 + residual_tmp5;
      const s_t residual_tmp7 = eta_s*residual_tmp6;
      const s_t residual_tmp8 = eta_s*u0_old_grad_1;
      const s_t residual_tmp9 = u0_grad_1*u_dt_shift;
      const s_t residual_tmp10 = eta_b*(s_t(2)*residual_tmp9 + u0_old_grad_1);
      const s_t residual_tmp11 = residual_tmp10 + residual_tmp8;
      const s_t residual_tmp12 = pow_m2(residual_tmp2);
      const s_t residual_tmp13 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp14 = u0_grad_0 + s_t(1);
      const s_t residual_tmp15 = residual_tmp9 + u0_old_grad_1;
      const s_t residual_tmp16 = u1_grad_0*u_dt_shift;
      const s_t residual_tmp17 = residual_tmp16 + u1_old_grad_0;
      const s_t residual_tmp18 = eta_s*(-residual_tmp1*residual_tmp17 + residual_tmp13*u0_grad_1 - residual_tmp14*residual_tmp15 + residual_tmp5*u1_grad_0);
      const s_t residual_tmp19 = residual_tmp15*u1_grad_0;
      const s_t residual_tmp20 = residual_tmp1*residual_tmp13;
      const s_t residual_tmp21 = -residual_tmp14*residual_tmp5 + residual_tmp17*u0_grad_1;
      const s_t residual_tmp22 = eta_b*(residual_tmp19 - residual_tmp20 + residual_tmp21);
      const s_t residual_tmp23 = eta_s*(-residual_tmp19 + residual_tmp20 + residual_tmp21);
      const s_t residual_tmp24 = residual_tmp22 - residual_tmp23;
      const s_t residual_tmp25 = -residual_tmp1*residual_tmp24 + residual_tmp18*u0_grad_1;
      const s_t residual_tmp26 = lmbda*residual_tmp1;
      const s_t residual_tmp27 = s_t(2)*mu*residual_tmp1*u0_grad_1 + residual_tmp26*u0_grad_1;
      const s_t residual_tmp28 = pow_2(residual_tmp1);
      const s_t residual_tmp29 = eta_b*(-residual_tmp4 - residual_tmp5);
      const s_t residual_tmp30 = -residual_tmp6;
      const s_t residual_tmp31 = -eta_s*residual_tmp30 + residual_tmp29;
      const s_t residual_tmp32 = -residual_tmp1;
      const s_t residual_tmp33 = residual_tmp12*residual_tmp25;
      const s_t residual_tmp34 = residual_tmp26*u1_grad_0;
      const s_t residual_tmp35 = residual_tmp1*u1_grad_0;
      const s_t residual_tmp36 = s_t(2)*residual_tmp35;
      const s_t residual_tmp37 = residual_tmp14*u_dt_shift;
      const s_t residual_tmp38 = residual_tmp13 - residual_tmp37;
      const s_t residual_tmp39 = eta_s*residual_tmp38;
      const s_t residual_tmp40 = eta_b*(s_t(2)*residual_tmp16 + u1_old_grad_0);
      const s_t residual_tmp41 = -eta_s*u1_old_grad_0 + residual_tmp40;
      const s_t residual_tmp42 = s_t(2)*u1_grad_1 + s_t(2);
      const s_t residual_tmp43 = s_t(2)*residual_tmp0 + s_t(6);
      const s_t residual_tmp44 = eta_b*(-residual_tmp13 - residual_tmp37);
      const s_t residual_tmp45 = -eta_s*residual_tmp38 + residual_tmp44;
      const s_t residual_tmp46 = eta_s*u1_old_grad_0;
      const s_t residual_tmp47 = -residual_tmp14;
      const s_t residual_tmp48 = lmbda*(-residual_tmp0 + residual_tmp1*residual_tmp14 + s_t(-1));
      const s_t residual_tmp49 = residual_tmp1*residual_tmp14;
      const s_t residual_tmp50 = lmbda*residual_tmp49 + residual_tmp48;
      const s_t residual_tmp51 = pow_2(u1_grad_0);
      const s_t residual_tmp52 = -residual_tmp14*residual_tmp18 + residual_tmp24*u1_grad_0;
      const s_t residual_tmp53 = residual_tmp12*residual_tmp52;
      const s_t residual_tmp54 = s_t(2)*residual_tmp14;
      const s_t residual_tmp55 = lmbda*residual_tmp14*u1_grad_0 + mu*residual_tmp54*u1_grad_0;
      const s_t residual_tmp56 = residual_tmp14*u0_grad_1;
      const s_t residual_tmp57 = s_t(2)*u0_grad_0 + s_t(2);
      const s_t residual_tmp58 = -residual_tmp18;
      const s_t residual_tmp59 = lmbda*residual_tmp0 + mu*(s_t(4)*residual_tmp0 - s_t(2)*residual_tmp49 + s_t(6)) - residual_tmp48;
      const s_t residual_tmp60 = pow_2(u0_grad_1);
      const s_t residual_tmp61 = residual_tmp10 - residual_tmp8;
      const s_t residual_tmp62 = residual_tmp22 + residual_tmp23;
      const s_t residual_tmp63 = -residual_tmp1*residual_tmp18 + residual_tmp62*u0_grad_1;
      const s_t residual_tmp64 = residual_tmp12*residual_tmp63;
      const s_t residual_tmp65 = eta_s*residual_tmp30 + residual_tmp29;
      const s_t residual_tmp66 = lmbda*residual_tmp56;
      const s_t residual_tmp67 = residual_tmp54*u0_grad_1;
      const s_t residual_tmp68 = residual_tmp39 + residual_tmp44;
      const s_t residual_tmp69 = residual_tmp40 + residual_tmp46;
      const s_t residual_tmp70 = -residual_tmp14*residual_tmp62 + residual_tmp18*u1_grad_0;
      const s_t residual_tmp71 = pow_2(residual_tmp14);
      const s_t residual_tmp72 = residual_tmp12*residual_tmp70;
      const s_t grad_coeff0_0 = u0_direction_grad_0*(lmbda*residual_tmp28 + mu*(s_t(2)*residual_tmp28 + s_t(4)) + residual_tmp3*(-residual_tmp1*residual_tmp31 - residual_tmp8*u0_grad_1) + residual_tmp32*residual_tmp33) + u0_direction_grad_1*(-mu*residual_tmp36 + residual_tmp12*residual_tmp25*u1_grad_0 + residual_tmp3*(-residual_tmp1*residual_tmp41 + residual_tmp18 + residual_tmp39*u0_grad_1) - residual_tmp34) + u1_direction_grad_0*(residual_tmp12*residual_tmp25*u0_grad_1 - residual_tmp27 + residual_tmp3*(-residual_tmp1*residual_tmp11 + residual_tmp7*u0_grad_1)) + u1_direction_grad_1*(mu*(s_t(2)*residual_tmp14*residual_tmp42 - residual_tmp43) + residual_tmp3*(-residual_tmp1*residual_tmp45 - residual_tmp24 - residual_tmp46*u0_grad_1) + residual_tmp33*residual_tmp47 + residual_tmp50);
      const s_t grad_coeff0_1 = u0_direction_grad_0*(mu*(-residual_tmp36 - s_t(4)*residual_tmp56 + s_t(2)*residual_tmp57*u0_grad_1) + residual_tmp3*(residual_tmp14*residual_tmp8 + residual_tmp31*u1_grad_0 + residual_tmp58) + residual_tmp32*residual_tmp53 - residual_tmp34) + u0_direction_grad_1*(lmbda*residual_tmp51 + mu*(s_t(2)*residual_tmp51 + s_t(4)) + residual_tmp3*(-residual_tmp14*residual_tmp39 + residual_tmp41*u1_grad_0) + residual_tmp53*u1_grad_0) + u1_direction_grad_0*(residual_tmp3*(residual_tmp11*u1_grad_0 - residual_tmp14*residual_tmp7 + residual_tmp24) + residual_tmp53*u0_grad_1 + residual_tmp59) + u1_direction_grad_1*(residual_tmp12*residual_tmp47*residual_tmp52 + residual_tmp3*(residual_tmp14*residual_tmp46 + residual_tmp45*u1_grad_0) - residual_tmp55);
      const s_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp12*residual_tmp32*residual_tmp63 - residual_tmp27 + residual_tmp3*(residual_tmp1*residual_tmp8 + residual_tmp65*u0_grad_1)) + u0_direction_grad_1*(residual_tmp3*(-residual_tmp1*residual_tmp39 + residual_tmp62 + residual_tmp69*u0_grad_1) + residual_tmp59 + residual_tmp64*u1_grad_0) + u1_direction_grad_0*(lmbda*residual_tmp60 + mu*(s_t(2)*residual_tmp60 + s_t(4)) + residual_tmp3*(-residual_tmp1*residual_tmp7 + residual_tmp61*u0_grad_1) + residual_tmp64*u0_grad_1) + u1_direction_grad_1*(mu*(-s_t(4)*residual_tmp35 + s_t(2)*residual_tmp42*u1_grad_0 - residual_tmp67) + residual_tmp3*(residual_tmp1*residual_tmp46 + residual_tmp58 + residual_tmp68*u0_grad_1) + residual_tmp47*residual_tmp64 - residual_tmp66);
      const s_t grad_coeff1_1 = u0_direction_grad_0*(mu*(s_t(2)*residual_tmp1*residual_tmp57 - residual_tmp43) + residual_tmp3*(-residual_tmp14*residual_tmp65 - residual_tmp62 - residual_tmp8*u1_grad_0) + residual_tmp32*residual_tmp72 + residual_tmp50) + u0_direction_grad_1*(residual_tmp12*residual_tmp70*u1_grad_0 + residual_tmp3*(-residual_tmp14*residual_tmp69 + residual_tmp39*u1_grad_0) - residual_tmp55) + u1_direction_grad_0*(-mu*residual_tmp67 + residual_tmp12*residual_tmp70*u0_grad_1 + residual_tmp3*(-residual_tmp14*residual_tmp61 + residual_tmp18 + residual_tmp7*u1_grad_0) - residual_tmp66) + u1_direction_grad_1*(lmbda*residual_tmp71 + mu*(s_t(2)*residual_tmp71 + s_t(4)) + residual_tmp3*(-residual_tmp14*residual_tmp68 - residual_tmp46*u1_grad_0) + residual_tmp47*residual_tmp72);
      value_coeff_q0[lane] = s_t(0);
      grad_coeff_ref_q0_0[lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
      grad_coeff_ref_q0_1[lane] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
      value_coeff_q1[lane] = s_t(0);
      grad_coeff_ref_q1_0[lane] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
      grad_coeff_ref_q1_1[lane] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
    }
  }
  tensor_integrate<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_residual_merit_d2_tensor_product_jacobian_action_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR q_weight_1d,
    const s_t current[2 * NS][VS],
    const s_t previous[2 * NS][VS],
    const s_t direction[2 * NS][VS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t output[2 * NS][VS]
) {
  static constexpr int ND = 2;
  static constexpr int NC = 2;
  s_t current_value[NC * NQ * VS];
  s_t current_grad_ref[NC * NQ * ND * VS];
  tensor_evaluate_contiguous<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, grad_1d, current, current_value, current_grad_ref);
  s_t previous_value[NC * NQ * VS];
  s_t previous_grad_ref[NC * NQ * ND * VS];
  tensor_evaluate_contiguous<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, grad_1d, previous, previous_value, previous_grad_ref);
  s_t direction_value[NC * NQ * VS];
  s_t direction_grad_ref[NC * NQ * ND * VS];
  tensor_evaluate_contiguous<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, grad_1d, direction, direction_value, direction_grad_ref);
  s_t value_coeff[NC * NQ * VS];
  s_t grad_coeff_ref[NC * NQ * ND * VS];
  static constexpr int NQ1 = integer_root(NQ, ND);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = q / NQ1;
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
    const s_t *const RSTR det_q = determinant + q * geometry_stride;
    const s_t *const RSTR adj_q0 = adjugate[0] + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adjugate[1] + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adjugate[2] + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adjugate[3] + q * geometry_stride;
    const s_t *const RSTR current_grad_ref_q0_0 = &current_grad_ref[(q * ND) * VS];
    const s_t *const RSTR current_grad_ref_q0_1 = &current_grad_ref[(q * ND + 1) * VS];
    const s_t *const RSTR previous_grad_ref_q0_0 = &previous_grad_ref[(q * ND) * VS];
    const s_t *const RSTR previous_grad_ref_q0_1 = &previous_grad_ref[(q * ND + 1) * VS];
    const s_t *const RSTR direction_grad_ref_q0_0 = &direction_grad_ref[(q * ND) * VS];
    const s_t *const RSTR direction_grad_ref_q0_1 = &direction_grad_ref[(q * ND + 1) * VS];
    const s_t *const RSTR current_grad_ref_q1_0 = &current_grad_ref[((NQ + q) * ND) * VS];
    const s_t *const RSTR current_grad_ref_q1_1 = &current_grad_ref[((NQ + q) * ND + 1) * VS];
    const s_t *const RSTR previous_grad_ref_q1_0 = &previous_grad_ref[((NQ + q) * ND) * VS];
    const s_t *const RSTR previous_grad_ref_q1_1 = &previous_grad_ref[((NQ + q) * ND + 1) * VS];
    const s_t *const RSTR direction_grad_ref_q1_0 = &direction_grad_ref[((NQ + q) * ND) * VS];
    const s_t *const RSTR direction_grad_ref_q1_1 = &direction_grad_ref[((NQ + q) * ND + 1) * VS];
    s_t *const RSTR value_coeff_q0 = &value_coeff[q * VS];
    s_t *const RSTR grad_coeff_ref_q0_0 = &grad_coeff_ref[(q * ND) * VS];
    s_t *const RSTR grad_coeff_ref_q0_1 = &grad_coeff_ref[(q * ND + 1) * VS];
    s_t *const RSTR value_coeff_q1 = &value_coeff[(NQ + q) * VS];
    s_t *const RSTR grad_coeff_ref_q1_0 = &grad_coeff_ref[((NQ + q) * ND) * VS];
    s_t *const RSTR grad_coeff_ref_q1_1 = &grad_coeff_ref[((NQ + q) * ND + 1) * VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t det = det_q[lane];
      const s_t adj0 = adj_q0[lane];
      const s_t adj1 = adj_q1[lane];
      const s_t adj2 = adj_q2[lane];
      const s_t adj3 = adj_q3[lane];
      const s_t u0_grad_0_ref = current_grad_ref_q0_0[lane];
      const s_t u0_grad_1_ref = current_grad_ref_q0_1[lane];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = previous_grad_ref_q0_0[lane];
      const s_t u0_old_grad_1_ref = previous_grad_ref_q0_1[lane];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u0_direction_grad_0_ref = direction_grad_ref_q0_0[lane];
      const s_t u0_direction_grad_1_ref = direction_grad_ref_q0_1[lane];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = current_grad_ref_q1_0[lane];
      const s_t u1_grad_1_ref = current_grad_ref_q1_1[lane];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = previous_grad_ref_q1_0[lane];
      const s_t u1_old_grad_1_ref = previous_grad_ref_q1_1[lane];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
      const s_t u1_direction_grad_0_ref = direction_grad_ref_q1_0[lane];
      const s_t u1_direction_grad_1_ref = direction_grad_ref_q1_1[lane];
      const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj2) / det;
      const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj3) / det;
      const s_t residual_tmp0 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp1 = u1_grad_1 + s_t(1);
      const s_t residual_tmp2 = -residual_tmp0 + residual_tmp1 + u0_grad_0*u1_grad_1 + u0_grad_0;
      const s_t residual_tmp3 = pow_m1(residual_tmp2);
      const s_t residual_tmp4 = residual_tmp1*u_dt_shift;
      const s_t residual_tmp5 = u1_grad_1*u_dt_shift + u1_old_grad_1;
      const s_t residual_tmp6 = -residual_tmp4 + residual_tmp5;
      const s_t residual_tmp7 = eta_s*residual_tmp6;
      const s_t residual_tmp8 = eta_s*u0_old_grad_1;
      const s_t residual_tmp9 = u0_grad_1*u_dt_shift;
      const s_t residual_tmp10 = eta_b*(s_t(2)*residual_tmp9 + u0_old_grad_1);
      const s_t residual_tmp11 = residual_tmp10 + residual_tmp8;
      const s_t residual_tmp12 = pow_m2(residual_tmp2);
      const s_t residual_tmp13 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp14 = u0_grad_0 + s_t(1);
      const s_t residual_tmp15 = residual_tmp9 + u0_old_grad_1;
      const s_t residual_tmp16 = u1_grad_0*u_dt_shift;
      const s_t residual_tmp17 = residual_tmp16 + u1_old_grad_0;
      const s_t residual_tmp18 = eta_s*(-residual_tmp1*residual_tmp17 + residual_tmp13*u0_grad_1 - residual_tmp14*residual_tmp15 + residual_tmp5*u1_grad_0);
      const s_t residual_tmp19 = residual_tmp15*u1_grad_0;
      const s_t residual_tmp20 = residual_tmp1*residual_tmp13;
      const s_t residual_tmp21 = -residual_tmp14*residual_tmp5 + residual_tmp17*u0_grad_1;
      const s_t residual_tmp22 = eta_b*(residual_tmp19 - residual_tmp20 + residual_tmp21);
      const s_t residual_tmp23 = eta_s*(-residual_tmp19 + residual_tmp20 + residual_tmp21);
      const s_t residual_tmp24 = residual_tmp22 - residual_tmp23;
      const s_t residual_tmp25 = -residual_tmp1*residual_tmp24 + residual_tmp18*u0_grad_1;
      const s_t residual_tmp26 = lmbda*residual_tmp1;
      const s_t residual_tmp27 = s_t(2)*mu*residual_tmp1*u0_grad_1 + residual_tmp26*u0_grad_1;
      const s_t residual_tmp28 = pow_2(residual_tmp1);
      const s_t residual_tmp29 = eta_b*(-residual_tmp4 - residual_tmp5);
      const s_t residual_tmp30 = -residual_tmp6;
      const s_t residual_tmp31 = -eta_s*residual_tmp30 + residual_tmp29;
      const s_t residual_tmp32 = -residual_tmp1;
      const s_t residual_tmp33 = residual_tmp12*residual_tmp25;
      const s_t residual_tmp34 = residual_tmp26*u1_grad_0;
      const s_t residual_tmp35 = residual_tmp1*u1_grad_0;
      const s_t residual_tmp36 = s_t(2)*residual_tmp35;
      const s_t residual_tmp37 = residual_tmp14*u_dt_shift;
      const s_t residual_tmp38 = residual_tmp13 - residual_tmp37;
      const s_t residual_tmp39 = eta_s*residual_tmp38;
      const s_t residual_tmp40 = eta_b*(s_t(2)*residual_tmp16 + u1_old_grad_0);
      const s_t residual_tmp41 = -eta_s*u1_old_grad_0 + residual_tmp40;
      const s_t residual_tmp42 = s_t(2)*u1_grad_1 + s_t(2);
      const s_t residual_tmp43 = s_t(2)*residual_tmp0 + s_t(6);
      const s_t residual_tmp44 = eta_b*(-residual_tmp13 - residual_tmp37);
      const s_t residual_tmp45 = -eta_s*residual_tmp38 + residual_tmp44;
      const s_t residual_tmp46 = eta_s*u1_old_grad_0;
      const s_t residual_tmp47 = -residual_tmp14;
      const s_t residual_tmp48 = lmbda*(-residual_tmp0 + residual_tmp1*residual_tmp14 + s_t(-1));
      const s_t residual_tmp49 = residual_tmp1*residual_tmp14;
      const s_t residual_tmp50 = lmbda*residual_tmp49 + residual_tmp48;
      const s_t residual_tmp51 = pow_2(u1_grad_0);
      const s_t residual_tmp52 = -residual_tmp14*residual_tmp18 + residual_tmp24*u1_grad_0;
      const s_t residual_tmp53 = residual_tmp12*residual_tmp52;
      const s_t residual_tmp54 = s_t(2)*residual_tmp14;
      const s_t residual_tmp55 = lmbda*residual_tmp14*u1_grad_0 + mu*residual_tmp54*u1_grad_0;
      const s_t residual_tmp56 = residual_tmp14*u0_grad_1;
      const s_t residual_tmp57 = s_t(2)*u0_grad_0 + s_t(2);
      const s_t residual_tmp58 = -residual_tmp18;
      const s_t residual_tmp59 = lmbda*residual_tmp0 + mu*(s_t(4)*residual_tmp0 - s_t(2)*residual_tmp49 + s_t(6)) - residual_tmp48;
      const s_t residual_tmp60 = pow_2(u0_grad_1);
      const s_t residual_tmp61 = residual_tmp10 - residual_tmp8;
      const s_t residual_tmp62 = residual_tmp22 + residual_tmp23;
      const s_t residual_tmp63 = -residual_tmp1*residual_tmp18 + residual_tmp62*u0_grad_1;
      const s_t residual_tmp64 = residual_tmp12*residual_tmp63;
      const s_t residual_tmp65 = eta_s*residual_tmp30 + residual_tmp29;
      const s_t residual_tmp66 = lmbda*residual_tmp56;
      const s_t residual_tmp67 = residual_tmp54*u0_grad_1;
      const s_t residual_tmp68 = residual_tmp39 + residual_tmp44;
      const s_t residual_tmp69 = residual_tmp40 + residual_tmp46;
      const s_t residual_tmp70 = -residual_tmp14*residual_tmp62 + residual_tmp18*u1_grad_0;
      const s_t residual_tmp71 = pow_2(residual_tmp14);
      const s_t residual_tmp72 = residual_tmp12*residual_tmp70;
      const s_t grad_coeff0_0 = u0_direction_grad_0*(lmbda*residual_tmp28 + mu*(s_t(2)*residual_tmp28 + s_t(4)) + residual_tmp3*(-residual_tmp1*residual_tmp31 - residual_tmp8*u0_grad_1) + residual_tmp32*residual_tmp33) + u0_direction_grad_1*(-mu*residual_tmp36 + residual_tmp12*residual_tmp25*u1_grad_0 + residual_tmp3*(-residual_tmp1*residual_tmp41 + residual_tmp18 + residual_tmp39*u0_grad_1) - residual_tmp34) + u1_direction_grad_0*(residual_tmp12*residual_tmp25*u0_grad_1 - residual_tmp27 + residual_tmp3*(-residual_tmp1*residual_tmp11 + residual_tmp7*u0_grad_1)) + u1_direction_grad_1*(mu*(s_t(2)*residual_tmp14*residual_tmp42 - residual_tmp43) + residual_tmp3*(-residual_tmp1*residual_tmp45 - residual_tmp24 - residual_tmp46*u0_grad_1) + residual_tmp33*residual_tmp47 + residual_tmp50);
      const s_t grad_coeff0_1 = u0_direction_grad_0*(mu*(-residual_tmp36 - s_t(4)*residual_tmp56 + s_t(2)*residual_tmp57*u0_grad_1) + residual_tmp3*(residual_tmp14*residual_tmp8 + residual_tmp31*u1_grad_0 + residual_tmp58) + residual_tmp32*residual_tmp53 - residual_tmp34) + u0_direction_grad_1*(lmbda*residual_tmp51 + mu*(s_t(2)*residual_tmp51 + s_t(4)) + residual_tmp3*(-residual_tmp14*residual_tmp39 + residual_tmp41*u1_grad_0) + residual_tmp53*u1_grad_0) + u1_direction_grad_0*(residual_tmp3*(residual_tmp11*u1_grad_0 - residual_tmp14*residual_tmp7 + residual_tmp24) + residual_tmp53*u0_grad_1 + residual_tmp59) + u1_direction_grad_1*(residual_tmp12*residual_tmp47*residual_tmp52 + residual_tmp3*(residual_tmp14*residual_tmp46 + residual_tmp45*u1_grad_0) - residual_tmp55);
      const s_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp12*residual_tmp32*residual_tmp63 - residual_tmp27 + residual_tmp3*(residual_tmp1*residual_tmp8 + residual_tmp65*u0_grad_1)) + u0_direction_grad_1*(residual_tmp3*(-residual_tmp1*residual_tmp39 + residual_tmp62 + residual_tmp69*u0_grad_1) + residual_tmp59 + residual_tmp64*u1_grad_0) + u1_direction_grad_0*(lmbda*residual_tmp60 + mu*(s_t(2)*residual_tmp60 + s_t(4)) + residual_tmp3*(-residual_tmp1*residual_tmp7 + residual_tmp61*u0_grad_1) + residual_tmp64*u0_grad_1) + u1_direction_grad_1*(mu*(-s_t(4)*residual_tmp35 + s_t(2)*residual_tmp42*u1_grad_0 - residual_tmp67) + residual_tmp3*(residual_tmp1*residual_tmp46 + residual_tmp58 + residual_tmp68*u0_grad_1) + residual_tmp47*residual_tmp64 - residual_tmp66);
      const s_t grad_coeff1_1 = u0_direction_grad_0*(mu*(s_t(2)*residual_tmp1*residual_tmp57 - residual_tmp43) + residual_tmp3*(-residual_tmp14*residual_tmp65 - residual_tmp62 - residual_tmp8*u1_grad_0) + residual_tmp32*residual_tmp72 + residual_tmp50) + u0_direction_grad_1*(residual_tmp12*residual_tmp70*u1_grad_0 + residual_tmp3*(-residual_tmp14*residual_tmp69 + residual_tmp39*u1_grad_0) - residual_tmp55) + u1_direction_grad_0*(-mu*residual_tmp67 + residual_tmp12*residual_tmp70*u0_grad_1 + residual_tmp3*(-residual_tmp14*residual_tmp61 + residual_tmp18 + residual_tmp7*u1_grad_0) - residual_tmp66) + u1_direction_grad_1*(lmbda*residual_tmp71 + mu*(s_t(2)*residual_tmp71 + s_t(4)) + residual_tmp3*(-residual_tmp14*residual_tmp68 - residual_tmp46*u1_grad_0) + residual_tmp47*residual_tmp72);
      value_coeff_q0[lane] = s_t(0);
      grad_coeff_ref_q0_0[lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
      grad_coeff_ref_q0_1[lane] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
      value_coeff_q1[lane] = s_t(0);
      grad_coeff_ref_q1_0[lane] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
      grad_coeff_ref_q1_1[lane] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
    }
  }
  tensor_integrate_contiguous<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

} // namespace codegen
} // namespace sfem

#endif
