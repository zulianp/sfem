#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_TOTAL_D2_TENSOR_PRODUCT_LOCAL_HPP
#define MOONEY_RIVLIN_KELVIN_VOIGT_TOTAL_D2_TENSOR_PRODUCT_LOCAL_HPP

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
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_total_d2_tensor_product_residual_block(
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
  s_t current_value[NC * NQ];
  s_t current_grad_ref[NC * NQ * ND];
  tensor_evaluate_scalar<s_t, NQ, NS, ND, NC>(
      shape_1d, grad_1d, current, current_value, current_grad_ref);
  s_t previous_value[NC * NQ];
  s_t previous_grad_ref[NC * NQ * ND];
  tensor_evaluate_scalar<s_t, NQ, NS, ND, NC>(
      shape_1d, grad_1d, previous, previous_value, previous_grad_ref);
  s_t value_coeff[NC * NQ];
  s_t grad_coeff_ref[NC * NQ * ND];
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
    const s_t *const RSTR current_grad_ref_q0_0 = &current_grad_ref[(q * ND)];
    const s_t *const RSTR current_grad_ref_q0_1 = &current_grad_ref[(q * ND + 1)];
    const s_t *const RSTR previous_grad_ref_q0_0 = &previous_grad_ref[(q * ND)];
    const s_t *const RSTR previous_grad_ref_q0_1 = &previous_grad_ref[(q * ND + 1)];
    const s_t *const RSTR current_grad_ref_q1_0 = &current_grad_ref[((NQ + q) * ND)];
    const s_t *const RSTR current_grad_ref_q1_1 = &current_grad_ref[((NQ + q) * ND + 1)];
    const s_t *const RSTR previous_grad_ref_q1_0 = &previous_grad_ref[((NQ + q) * ND)];
    const s_t *const RSTR previous_grad_ref_q1_1 = &previous_grad_ref[((NQ + q) * ND + 1)];
    s_t *const RSTR value_coeff_q0 = &value_coeff[q];
    s_t *const RSTR grad_coeff_ref_q0_0 = &grad_coeff_ref[(q * ND)];
    s_t *const RSTR grad_coeff_ref_q0_1 = &grad_coeff_ref[(q * ND + 1)];
    s_t *const RSTR value_coeff_q1 = &value_coeff[(NQ + q)];
    s_t *const RSTR grad_coeff_ref_q1_0 = &grad_coeff_ref[((NQ + q) * ND)];
    s_t *const RSTR grad_coeff_ref_q1_1 = &grad_coeff_ref[((NQ + q) * ND + 1)];
    {
      const s_t det = det_q[0];
      const s_t adj0 = adj_q0[0];
      const s_t adj1 = adj_q1[0];
      const s_t adj2 = adj_q2[0];
      const s_t adj3 = adj_q3[0];
      const s_t u0_grad_0_ref = current_grad_ref_q0_0[0];
      const s_t u0_grad_1_ref = current_grad_ref_q0_1[0];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = previous_grad_ref_q0_0[0];
      const s_t u0_old_grad_1_ref = previous_grad_ref_q0_1[0];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = current_grad_ref_q1_0[0];
      const s_t u1_grad_1_ref = current_grad_ref_q1_1[0];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = previous_grad_ref_q1_0[0];
      const s_t u1_old_grad_1_ref = previous_grad_ref_q1_1[0];
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
      value_coeff_q0[0] = s_t(0);
      grad_coeff_ref_q0_0[0] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
      grad_coeff_ref_q0_1[0] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
      value_coeff_q1[0] = s_t(0);
      grad_coeff_ref_q1_0[0] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
      grad_coeff_ref_q1_1[0] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
    }
  }
  tensor_integrate_scalar<s_t, NQ, NS, ND, NC>(
      shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_total_d2_tensor_product_residual_block_contiguous(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR q_weight_1d,
    const s_t current[2 * NS],
    const s_t previous[2 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t output[2 * NS]
) {
  static constexpr int ND = 2;
  static constexpr int NC = 2;
  s_t current_value[NC * NQ];
  s_t current_grad_ref[NC * NQ * ND];
  tensor_evaluate_contiguous_scalar<s_t, NQ, NS, ND, NC>(
      shape_1d, grad_1d, current, current_value, current_grad_ref);
  s_t previous_value[NC * NQ];
  s_t previous_grad_ref[NC * NQ * ND];
  tensor_evaluate_contiguous_scalar<s_t, NQ, NS, ND, NC>(
      shape_1d, grad_1d, previous, previous_value, previous_grad_ref);
  s_t value_coeff[NC * NQ];
  s_t grad_coeff_ref[NC * NQ * ND];
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
    const s_t *const RSTR current_grad_ref_q0_0 = &current_grad_ref[(q * ND)];
    const s_t *const RSTR current_grad_ref_q0_1 = &current_grad_ref[(q * ND + 1)];
    const s_t *const RSTR previous_grad_ref_q0_0 = &previous_grad_ref[(q * ND)];
    const s_t *const RSTR previous_grad_ref_q0_1 = &previous_grad_ref[(q * ND + 1)];
    const s_t *const RSTR current_grad_ref_q1_0 = &current_grad_ref[((NQ + q) * ND)];
    const s_t *const RSTR current_grad_ref_q1_1 = &current_grad_ref[((NQ + q) * ND + 1)];
    const s_t *const RSTR previous_grad_ref_q1_0 = &previous_grad_ref[((NQ + q) * ND)];
    const s_t *const RSTR previous_grad_ref_q1_1 = &previous_grad_ref[((NQ + q) * ND + 1)];
    s_t *const RSTR value_coeff_q0 = &value_coeff[q];
    s_t *const RSTR grad_coeff_ref_q0_0 = &grad_coeff_ref[(q * ND)];
    s_t *const RSTR grad_coeff_ref_q0_1 = &grad_coeff_ref[(q * ND + 1)];
    s_t *const RSTR value_coeff_q1 = &value_coeff[(NQ + q)];
    s_t *const RSTR grad_coeff_ref_q1_0 = &grad_coeff_ref[((NQ + q) * ND)];
    s_t *const RSTR grad_coeff_ref_q1_1 = &grad_coeff_ref[((NQ + q) * ND + 1)];
    {
      const s_t det = det_q[0];
      const s_t adj0 = adj_q0[0];
      const s_t adj1 = adj_q1[0];
      const s_t adj2 = adj_q2[0];
      const s_t adj3 = adj_q3[0];
      const s_t u0_grad_0_ref = current_grad_ref_q0_0[0];
      const s_t u0_grad_1_ref = current_grad_ref_q0_1[0];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = previous_grad_ref_q0_0[0];
      const s_t u0_old_grad_1_ref = previous_grad_ref_q0_1[0];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = current_grad_ref_q1_0[0];
      const s_t u1_grad_1_ref = current_grad_ref_q1_1[0];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = previous_grad_ref_q1_0[0];
      const s_t u1_old_grad_1_ref = previous_grad_ref_q1_1[0];
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
      value_coeff_q0[0] = s_t(0);
      grad_coeff_ref_q0_0[0] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
      grad_coeff_ref_q0_1[0] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
      value_coeff_q1[0] = s_t(0);
      grad_coeff_ref_q1_0[0] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
      grad_coeff_ref_q1_1[0] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
    }
  }
  tensor_integrate_contiguous_scalar<s_t, NQ, NS, ND, NC>(
      shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_total_d2_tensor_product_jacobian_action_block(
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
  s_t current_value[NC * NQ];
  s_t current_grad_ref[NC * NQ * ND];
  tensor_evaluate_scalar<s_t, NQ, NS, ND, NC>(
      shape_1d, grad_1d, current, current_value, current_grad_ref);
  s_t previous_value[NC * NQ];
  s_t previous_grad_ref[NC * NQ * ND];
  tensor_evaluate_scalar<s_t, NQ, NS, ND, NC>(
      shape_1d, grad_1d, previous, previous_value, previous_grad_ref);
  s_t direction_value[NC * NQ];
  s_t direction_grad_ref[NC * NQ * ND];
  tensor_evaluate_scalar<s_t, NQ, NS, ND, NC>(
      shape_1d, grad_1d, direction, direction_value, direction_grad_ref);
  s_t value_coeff[NC * NQ];
  s_t grad_coeff_ref[NC * NQ * ND];
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
    const s_t *const RSTR current_grad_ref_q0_0 = &current_grad_ref[(q * ND)];
    const s_t *const RSTR current_grad_ref_q0_1 = &current_grad_ref[(q * ND + 1)];
    const s_t *const RSTR previous_grad_ref_q0_0 = &previous_grad_ref[(q * ND)];
    const s_t *const RSTR previous_grad_ref_q0_1 = &previous_grad_ref[(q * ND + 1)];
    const s_t *const RSTR direction_grad_ref_q0_0 = &direction_grad_ref[(q * ND)];
    const s_t *const RSTR direction_grad_ref_q0_1 = &direction_grad_ref[(q * ND + 1)];
    const s_t *const RSTR current_grad_ref_q1_0 = &current_grad_ref[((NQ + q) * ND)];
    const s_t *const RSTR current_grad_ref_q1_1 = &current_grad_ref[((NQ + q) * ND + 1)];
    const s_t *const RSTR previous_grad_ref_q1_0 = &previous_grad_ref[((NQ + q) * ND)];
    const s_t *const RSTR previous_grad_ref_q1_1 = &previous_grad_ref[((NQ + q) * ND + 1)];
    const s_t *const RSTR direction_grad_ref_q1_0 = &direction_grad_ref[((NQ + q) * ND)];
    const s_t *const RSTR direction_grad_ref_q1_1 = &direction_grad_ref[((NQ + q) * ND + 1)];
    s_t *const RSTR value_coeff_q0 = &value_coeff[q];
    s_t *const RSTR grad_coeff_ref_q0_0 = &grad_coeff_ref[(q * ND)];
    s_t *const RSTR grad_coeff_ref_q0_1 = &grad_coeff_ref[(q * ND + 1)];
    s_t *const RSTR value_coeff_q1 = &value_coeff[(NQ + q)];
    s_t *const RSTR grad_coeff_ref_q1_0 = &grad_coeff_ref[((NQ + q) * ND)];
    s_t *const RSTR grad_coeff_ref_q1_1 = &grad_coeff_ref[((NQ + q) * ND + 1)];
    {
      const s_t det = det_q[0];
      const s_t adj0 = adj_q0[0];
      const s_t adj1 = adj_q1[0];
      const s_t adj2 = adj_q2[0];
      const s_t adj3 = adj_q3[0];
      const s_t u0_grad_0_ref = current_grad_ref_q0_0[0];
      const s_t u0_grad_1_ref = current_grad_ref_q0_1[0];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = previous_grad_ref_q0_0[0];
      const s_t u0_old_grad_1_ref = previous_grad_ref_q0_1[0];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u0_direction_grad_0_ref = direction_grad_ref_q0_0[0];
      const s_t u0_direction_grad_1_ref = direction_grad_ref_q0_1[0];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = current_grad_ref_q1_0[0];
      const s_t u1_grad_1_ref = current_grad_ref_q1_1[0];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = previous_grad_ref_q1_0[0];
      const s_t u1_old_grad_1_ref = previous_grad_ref_q1_1[0];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
      const s_t u1_direction_grad_0_ref = direction_grad_ref_q1_0[0];
      const s_t u1_direction_grad_1_ref = direction_grad_ref_q1_1[0];
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
      value_coeff_q0[0] = s_t(0);
      grad_coeff_ref_q0_0[0] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
      grad_coeff_ref_q0_1[0] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
      value_coeff_q1[0] = s_t(0);
      grad_coeff_ref_q1_0[0] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
      grad_coeff_ref_q1_1[0] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
    }
  }
  tensor_integrate_scalar<s_t, NQ, NS, ND, NC>(
      shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_total_d2_tensor_product_jacobian_action_block_contiguous(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR q_weight_1d,
    const s_t current[2 * NS],
    const s_t previous[2 * NS],
    const s_t direction[2 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t output[2 * NS]
) {
  static constexpr int ND = 2;
  static constexpr int NC = 2;
  s_t current_value[NC * NQ];
  s_t current_grad_ref[NC * NQ * ND];
  tensor_evaluate_contiguous_scalar<s_t, NQ, NS, ND, NC>(
      shape_1d, grad_1d, current, current_value, current_grad_ref);
  s_t previous_value[NC * NQ];
  s_t previous_grad_ref[NC * NQ * ND];
  tensor_evaluate_contiguous_scalar<s_t, NQ, NS, ND, NC>(
      shape_1d, grad_1d, previous, previous_value, previous_grad_ref);
  s_t direction_value[NC * NQ];
  s_t direction_grad_ref[NC * NQ * ND];
  tensor_evaluate_contiguous_scalar<s_t, NQ, NS, ND, NC>(
      shape_1d, grad_1d, direction, direction_value, direction_grad_ref);
  s_t value_coeff[NC * NQ];
  s_t grad_coeff_ref[NC * NQ * ND];
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
    const s_t *const RSTR current_grad_ref_q0_0 = &current_grad_ref[(q * ND)];
    const s_t *const RSTR current_grad_ref_q0_1 = &current_grad_ref[(q * ND + 1)];
    const s_t *const RSTR previous_grad_ref_q0_0 = &previous_grad_ref[(q * ND)];
    const s_t *const RSTR previous_grad_ref_q0_1 = &previous_grad_ref[(q * ND + 1)];
    const s_t *const RSTR direction_grad_ref_q0_0 = &direction_grad_ref[(q * ND)];
    const s_t *const RSTR direction_grad_ref_q0_1 = &direction_grad_ref[(q * ND + 1)];
    const s_t *const RSTR current_grad_ref_q1_0 = &current_grad_ref[((NQ + q) * ND)];
    const s_t *const RSTR current_grad_ref_q1_1 = &current_grad_ref[((NQ + q) * ND + 1)];
    const s_t *const RSTR previous_grad_ref_q1_0 = &previous_grad_ref[((NQ + q) * ND)];
    const s_t *const RSTR previous_grad_ref_q1_1 = &previous_grad_ref[((NQ + q) * ND + 1)];
    const s_t *const RSTR direction_grad_ref_q1_0 = &direction_grad_ref[((NQ + q) * ND)];
    const s_t *const RSTR direction_grad_ref_q1_1 = &direction_grad_ref[((NQ + q) * ND + 1)];
    s_t *const RSTR value_coeff_q0 = &value_coeff[q];
    s_t *const RSTR grad_coeff_ref_q0_0 = &grad_coeff_ref[(q * ND)];
    s_t *const RSTR grad_coeff_ref_q0_1 = &grad_coeff_ref[(q * ND + 1)];
    s_t *const RSTR value_coeff_q1 = &value_coeff[(NQ + q)];
    s_t *const RSTR grad_coeff_ref_q1_0 = &grad_coeff_ref[((NQ + q) * ND)];
    s_t *const RSTR grad_coeff_ref_q1_1 = &grad_coeff_ref[((NQ + q) * ND + 1)];
    {
      const s_t det = det_q[0];
      const s_t adj0 = adj_q0[0];
      const s_t adj1 = adj_q1[0];
      const s_t adj2 = adj_q2[0];
      const s_t adj3 = adj_q3[0];
      const s_t u0_grad_0_ref = current_grad_ref_q0_0[0];
      const s_t u0_grad_1_ref = current_grad_ref_q0_1[0];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = previous_grad_ref_q0_0[0];
      const s_t u0_old_grad_1_ref = previous_grad_ref_q0_1[0];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u0_direction_grad_0_ref = direction_grad_ref_q0_0[0];
      const s_t u0_direction_grad_1_ref = direction_grad_ref_q0_1[0];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = current_grad_ref_q1_0[0];
      const s_t u1_grad_1_ref = current_grad_ref_q1_1[0];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = previous_grad_ref_q1_0[0];
      const s_t u1_old_grad_1_ref = previous_grad_ref_q1_1[0];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
      const s_t u1_direction_grad_0_ref = direction_grad_ref_q1_0[0];
      const s_t u1_direction_grad_1_ref = direction_grad_ref_q1_1[0];
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
      value_coeff_q0[0] = s_t(0);
      grad_coeff_ref_q0_0[0] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
      grad_coeff_ref_q0_1[0] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
      value_coeff_q1[0] = s_t(0);
      grad_coeff_ref_q1_0[0] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
      grad_coeff_ref_q1_1[0] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
    }
  }
  tensor_integrate_contiguous_scalar<s_t, NQ, NS, ND, NC>(
      shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_total_d2_tensor_product_hessian_block(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR q_weight_1d,
    const s_t current[2 * NS],
    const s_t previous[2 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t *const RSTR element_matrix
) {
  static constexpr int ND = 2;
  static constexpr int NC = 2;
  for (int entry = 0; entry < 2 * NS * 2 * NS; ++entry) {
    element_matrix[entry] = s_t(0);
  }
  s_t current_value[NC * NQ];
  s_t current_grad_ref[NC * NQ * ND];
  tensor_evaluate_contiguous_scalar<s_t, NQ, NS, ND, NC>(
      shape_1d, grad_1d, current, current_value, current_grad_ref);
  s_t previous_value[NC * NQ];
  s_t previous_grad_ref[NC * NQ * ND];
  tensor_evaluate_contiguous_scalar<s_t, NQ, NS, ND, NC>(
      shape_1d, grad_1d, previous, previous_value, previous_grad_ref);
  s_t value_coeff_c0[NC * NQ];
  s_t value_coeff_c1[NC * NQ];
  s_t grad_coeff_ref_c0[NC * NQ * ND];
  s_t grad_coeff_ref_c1[NC * NQ * ND];
  static constexpr int NQ1 = integer_root(NQ, ND);
  static constexpr int NS1 = integer_root(NS, ND);
  s_t * column[NC * NS];
  static constexpr int N_TANGENT = 16;
  s_t tangent[N_TANGENT * NQ];
  for (int q = 0; q < NQ; ++q) {
    const s_t *const RSTR det_q = determinant + q * geometry_stride;
    const s_t *const RSTR adj_q0 = adjugate[0] + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adjugate[1] + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adjugate[2] + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adjugate[3] + q * geometry_stride;
    const s_t *const RSTR current_grad_ref_q0_0 = &current_grad_ref[(q * ND)];
    const s_t *const RSTR current_grad_ref_q0_1 = &current_grad_ref[(q * ND + 1)];
    const s_t *const RSTR previous_grad_ref_q0_0 = &previous_grad_ref[(q * ND)];
    const s_t *const RSTR previous_grad_ref_q0_1 = &previous_grad_ref[(q * ND + 1)];
    const s_t *const RSTR current_grad_ref_q1_0 = &current_grad_ref[((NQ + q) * ND)];
    const s_t *const RSTR current_grad_ref_q1_1 = &current_grad_ref[((NQ + q) * ND + 1)];
    const s_t *const RSTR previous_grad_ref_q1_0 = &previous_grad_ref[((NQ + q) * ND)];
    const s_t *const RSTR previous_grad_ref_q1_1 = &previous_grad_ref[((NQ + q) * ND + 1)];
    s_t *const RSTR tangent_grad_d0_0_grad0_0_q = &tangent[q];
    s_t *const RSTR tangent_grad_d0_0_grad0_1_q = &tangent[(NQ + q)];
    s_t *const RSTR tangent_grad_d0_0_grad1_0_q = &tangent[(2 * NQ + q)];
    s_t *const RSTR tangent_grad_d0_0_grad1_1_q = &tangent[(3 * NQ + q)];
    s_t *const RSTR tangent_grad_d0_1_grad0_0_q = &tangent[(4 * NQ + q)];
    s_t *const RSTR tangent_grad_d0_1_grad0_1_q = &tangent[(5 * NQ + q)];
    s_t *const RSTR tangent_grad_d0_1_grad1_0_q = &tangent[(6 * NQ + q)];
    s_t *const RSTR tangent_grad_d0_1_grad1_1_q = &tangent[(7 * NQ + q)];
    s_t *const RSTR tangent_grad_d1_0_grad0_0_q = &tangent[(8 * NQ + q)];
    s_t *const RSTR tangent_grad_d1_0_grad0_1_q = &tangent[(9 * NQ + q)];
    s_t *const RSTR tangent_grad_d1_0_grad1_0_q = &tangent[(10 * NQ + q)];
    s_t *const RSTR tangent_grad_d1_0_grad1_1_q = &tangent[(11 * NQ + q)];
    s_t *const RSTR tangent_grad_d1_1_grad0_0_q = &tangent[(12 * NQ + q)];
    s_t *const RSTR tangent_grad_d1_1_grad0_1_q = &tangent[(13 * NQ + q)];
    s_t *const RSTR tangent_grad_d1_1_grad1_0_q = &tangent[(14 * NQ + q)];
    s_t *const RSTR tangent_grad_d1_1_grad1_1_q = &tangent[(15 * NQ + q)];
    {
      const s_t det = det_q[0];
      const s_t adj0 = adj_q0[0];
      const s_t adj1 = adj_q1[0];
      const s_t adj2 = adj_q2[0];
      const s_t adj3 = adj_q3[0];
      const s_t u0_grad_0_ref = current_grad_ref_q0_0[0];
      const s_t u0_grad_1_ref = current_grad_ref_q0_1[0];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = previous_grad_ref_q0_0[0];
      const s_t u0_old_grad_1_ref = previous_grad_ref_q0_1[0];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = current_grad_ref_q1_0[0];
      const s_t u1_grad_1_ref = current_grad_ref_q1_1[0];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = previous_grad_ref_q1_0[0];
      const s_t u1_old_grad_1_ref = previous_grad_ref_q1_1[0];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
      const s_t tangent_tmp0 = u1_grad_1 + s_t(1);
      const s_t tangent_tmp1 = pow_2(tangent_tmp0);
      const s_t tangent_tmp2 = u0_grad_1*u1_grad_0;
      const s_t tangent_tmp3 = tangent_tmp0 - tangent_tmp2 + u0_grad_0*u1_grad_1 + u0_grad_0;
      const s_t tangent_tmp4 = pow_m1(tangent_tmp3);
      const s_t tangent_tmp5 = eta_s*u0_old_grad_1;
      const s_t tangent_tmp6 = tangent_tmp0*u_dt_shift;
      const s_t tangent_tmp7 = u1_grad_1*u_dt_shift + u1_old_grad_1;
      const s_t tangent_tmp8 = eta_b*(-tangent_tmp6 - tangent_tmp7);
      const s_t tangent_tmp9 = -tangent_tmp6 + tangent_tmp7;
      const s_t tangent_tmp10 = -tangent_tmp9;
      const s_t tangent_tmp11 = -eta_s*tangent_tmp10 + tangent_tmp8;
      const s_t tangent_tmp12 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t tangent_tmp13 = u0_grad_0 + s_t(1);
      const s_t tangent_tmp14 = u0_grad_1*u_dt_shift;
      const s_t tangent_tmp15 = tangent_tmp14 + u0_old_grad_1;
      const s_t tangent_tmp16 = u1_grad_0*u_dt_shift;
      const s_t tangent_tmp17 = tangent_tmp16 + u1_old_grad_0;
      const s_t tangent_tmp18 = eta_s*(-tangent_tmp0*tangent_tmp17 + tangent_tmp12*u0_grad_1 - tangent_tmp13*tangent_tmp15 + tangent_tmp7*u1_grad_0);
      const s_t tangent_tmp19 = tangent_tmp15*u1_grad_0;
      const s_t tangent_tmp20 = tangent_tmp0*tangent_tmp12;
      const s_t tangent_tmp21 = -tangent_tmp13*tangent_tmp7 + tangent_tmp17*u0_grad_1;
      const s_t tangent_tmp22 = eta_b*(tangent_tmp19 - tangent_tmp20 + tangent_tmp21);
      const s_t tangent_tmp23 = eta_s*(-tangent_tmp19 + tangent_tmp20 + tangent_tmp21);
      const s_t tangent_tmp24 = tangent_tmp22 - tangent_tmp23;
      const s_t tangent_tmp25 = -tangent_tmp0*tangent_tmp24 + tangent_tmp18*u0_grad_1;
      const s_t tangent_tmp26 = pow_m2(tangent_tmp3);
      const s_t tangent_tmp27 = -tangent_tmp0;
      const s_t tangent_tmp28 = tangent_tmp26*tangent_tmp27;
      const s_t tangent_tmp29 = tangent_tmp0*u1_grad_0;
      const s_t tangent_tmp30 = lmbda*tangent_tmp29;
      const s_t tangent_tmp31 = tangent_tmp13*u0_grad_1;
      const s_t tangent_tmp32 = s_t(2)*tangent_tmp29;
      const s_t tangent_tmp33 = s_t(2)*u0_grad_0 + s_t(2);
      const s_t tangent_tmp34 = -tangent_tmp18;
      const s_t tangent_tmp35 = -tangent_tmp13*tangent_tmp18 + tangent_tmp24*u1_grad_0;
      const s_t tangent_tmp36 = eta_s*tangent_tmp10 + tangent_tmp8;
      const s_t tangent_tmp37 = tangent_tmp22 + tangent_tmp23;
      const s_t tangent_tmp38 = -tangent_tmp0*tangent_tmp18 + tangent_tmp37*u0_grad_1;
      const s_t tangent_tmp39 = tangent_tmp0*u0_grad_1;
      const s_t tangent_tmp40 = s_t(2)*mu;
      const s_t tangent_tmp41 = lmbda*tangent_tmp39 + tangent_tmp39*tangent_tmp40;
      const s_t tangent_tmp42 = s_t(2)*tangent_tmp2 + s_t(6);
      const s_t tangent_tmp43 = -tangent_tmp13*tangent_tmp37 + tangent_tmp18*u1_grad_0;
      const s_t tangent_tmp44 = lmbda*(tangent_tmp0*tangent_tmp13 - tangent_tmp2 + s_t(-1));
      const s_t tangent_tmp45 = tangent_tmp0*tangent_tmp13;
      const s_t tangent_tmp46 = lmbda*tangent_tmp45 + tangent_tmp44;
      const s_t tangent_tmp47 = tangent_tmp13*u_dt_shift;
      const s_t tangent_tmp48 = tangent_tmp12 - tangent_tmp47;
      const s_t tangent_tmp49 = eta_s*tangent_tmp48;
      const s_t tangent_tmp50 = eta_b*(s_t(2)*tangent_tmp16 + u1_old_grad_0);
      const s_t tangent_tmp51 = -eta_s*u1_old_grad_0 + tangent_tmp50;
      const s_t tangent_tmp52 = pow_2(u1_grad_0);
      const s_t tangent_tmp53 = tangent_tmp26*u1_grad_0;
      const s_t tangent_tmp54 = eta_s*u1_old_grad_0;
      const s_t tangent_tmp55 = tangent_tmp50 + tangent_tmp54;
      const s_t tangent_tmp56 = lmbda*tangent_tmp2 + mu*(s_t(4)*tangent_tmp2 - s_t(2)*tangent_tmp45 + s_t(6)) - tangent_tmp44;
      const s_t tangent_tmp57 = tangent_tmp13*u1_grad_0;
      const s_t tangent_tmp58 = lmbda*tangent_tmp57 + tangent_tmp40*tangent_tmp57;
      const s_t tangent_tmp59 = eta_s*tangent_tmp9;
      const s_t tangent_tmp60 = eta_b*(s_t(2)*tangent_tmp14 + u0_old_grad_1);
      const s_t tangent_tmp61 = tangent_tmp5 + tangent_tmp60;
      const s_t tangent_tmp62 = tangent_tmp26*u0_grad_1;
      const s_t tangent_tmp63 = pow_2(u0_grad_1);
      const s_t tangent_tmp64 = -tangent_tmp5 + tangent_tmp60;
      const s_t tangent_tmp65 = lmbda*tangent_tmp31;
      const s_t tangent_tmp66 = s_t(2)*tangent_tmp31;
      const s_t tangent_tmp67 = s_t(2)*u1_grad_1 + s_t(2);
      const s_t tangent_tmp68 = eta_b*(-tangent_tmp12 - tangent_tmp47);
      const s_t tangent_tmp69 = -eta_s*tangent_tmp48 + tangent_tmp68;
      const s_t tangent_tmp70 = -tangent_tmp13;
      const s_t tangent_tmp71 = tangent_tmp26*tangent_tmp70;
      const s_t tangent_tmp72 = tangent_tmp49 + tangent_tmp68;
      const s_t tangent_tmp73 = pow_2(tangent_tmp13);
      tangent_grad_d0_0_grad0_0_q[0] = lmbda*tangent_tmp1 + mu*(s_t(2)*tangent_tmp1 + s_t(4)) + tangent_tmp25*tangent_tmp28 + tangent_tmp4*(-tangent_tmp0*tangent_tmp11 - tangent_tmp5*u0_grad_1);
      tangent_grad_d0_0_grad0_1_q[0] = mu*(-s_t(4)*tangent_tmp31 - tangent_tmp32 + s_t(2)*tangent_tmp33*u0_grad_1) + tangent_tmp28*tangent_tmp35 - tangent_tmp30 + tangent_tmp4*(tangent_tmp11*u1_grad_0 + tangent_tmp13*tangent_tmp5 + tangent_tmp34);
      tangent_grad_d0_0_grad1_0_q[0] = tangent_tmp26*tangent_tmp27*tangent_tmp38 + tangent_tmp4*(tangent_tmp0*tangent_tmp5 + tangent_tmp36*u0_grad_1) - tangent_tmp41;
      tangent_grad_d0_0_grad1_1_q[0] = mu*(s_t(2)*tangent_tmp0*tangent_tmp33 - tangent_tmp42) + tangent_tmp28*tangent_tmp43 + tangent_tmp4*(-tangent_tmp13*tangent_tmp36 - tangent_tmp37 - tangent_tmp5*u1_grad_0) + tangent_tmp46;
      tangent_grad_d0_1_grad0_0_q[0] = -mu*tangent_tmp32 + tangent_tmp25*tangent_tmp26*u1_grad_0 - tangent_tmp30 + tangent_tmp4*(-tangent_tmp0*tangent_tmp51 + tangent_tmp18 + tangent_tmp49*u0_grad_1);
      tangent_grad_d0_1_grad0_1_q[0] = lmbda*tangent_tmp52 + mu*(s_t(2)*tangent_tmp52 + s_t(4)) + tangent_tmp35*tangent_tmp53 + tangent_tmp4*(-tangent_tmp13*tangent_tmp49 + tangent_tmp51*u1_grad_0);
      tangent_grad_d0_1_grad1_0_q[0] = tangent_tmp38*tangent_tmp53 + tangent_tmp4*(-tangent_tmp0*tangent_tmp49 + tangent_tmp37 + tangent_tmp55*u0_grad_1) + tangent_tmp56;
      tangent_grad_d0_1_grad1_1_q[0] = tangent_tmp26*tangent_tmp43*u1_grad_0 + tangent_tmp4*(-tangent_tmp13*tangent_tmp55 + tangent_tmp49*u1_grad_0) - tangent_tmp58;
      tangent_grad_d1_0_grad0_0_q[0] = tangent_tmp25*tangent_tmp26*u0_grad_1 + tangent_tmp4*(-tangent_tmp0*tangent_tmp61 + tangent_tmp59*u0_grad_1) - tangent_tmp41;
      tangent_grad_d1_0_grad0_1_q[0] = tangent_tmp35*tangent_tmp62 + tangent_tmp4*(-tangent_tmp13*tangent_tmp59 + tangent_tmp24 + tangent_tmp61*u1_grad_0) + tangent_tmp56;
      tangent_grad_d1_0_grad1_0_q[0] = lmbda*tangent_tmp63 + mu*(s_t(2)*tangent_tmp63 + s_t(4)) + tangent_tmp38*tangent_tmp62 + tangent_tmp4*(-tangent_tmp0*tangent_tmp59 + tangent_tmp64*u0_grad_1);
      tangent_grad_d1_0_grad1_1_q[0] = -mu*tangent_tmp66 + tangent_tmp26*tangent_tmp43*u0_grad_1 + tangent_tmp4*(-tangent_tmp13*tangent_tmp64 + tangent_tmp18 + tangent_tmp59*u1_grad_0) - tangent_tmp65;
      tangent_grad_d1_1_grad0_0_q[0] = mu*(s_t(2)*tangent_tmp13*tangent_tmp67 - tangent_tmp42) + tangent_tmp25*tangent_tmp71 + tangent_tmp4*(-tangent_tmp0*tangent_tmp69 - tangent_tmp24 - tangent_tmp54*u0_grad_1) + tangent_tmp46;
      tangent_grad_d1_1_grad0_1_q[0] = tangent_tmp26*tangent_tmp35*tangent_tmp70 + tangent_tmp4*(tangent_tmp13*tangent_tmp54 + tangent_tmp69*u1_grad_0) - tangent_tmp58;
      tangent_grad_d1_1_grad1_0_q[0] = mu*(-s_t(4)*tangent_tmp29 - tangent_tmp66 + s_t(2)*tangent_tmp67*u1_grad_0) + tangent_tmp38*tangent_tmp71 + tangent_tmp4*(tangent_tmp0*tangent_tmp54 + tangent_tmp34 + tangent_tmp72*u0_grad_1) - tangent_tmp65;
      tangent_grad_d1_1_grad1_1_q[0] = lmbda*tangent_tmp73 + mu*(s_t(2)*tangent_tmp73 + s_t(4)) + tangent_tmp4*(-tangent_tmp13*tangent_tmp72 - tangent_tmp54*u1_grad_0) + tangent_tmp43*tangent_tmp71;
    }
  }
  for (int trial = 0; trial < NS; ++trial) {
    const int trial_x = trial % NS1;
    const int trial_y = trial / NS1;
    for (int q = 0; q < NQ; ++q) {
      const int q_x = q % NQ1;
      const int q_y = q / NQ1;
      const s_t qw = q_weight_1d[q_x] * q_weight_1d[q_y];
      const s_t *const RSTR det_q = determinant + q * geometry_stride;
      const s_t *const RSTR adj_q0 = adjugate[0] + q * geometry_stride;
      const s_t *const RSTR adj_q1 = adjugate[1] + q * geometry_stride;
      const s_t *const RSTR adj_q2 = adjugate[2] + q * geometry_stride;
      const s_t *const RSTR adj_q3 = adjugate[3] + q * geometry_stride;
      const s_t trial_grad_ref0 = grad_1d[q_x * NS1 + trial_x] * shape_1d[q_y * NS1 + trial_y];
      const s_t trial_grad_ref1 = shape_1d[q_x * NS1 + trial_x] * grad_1d[q_y * NS1 + trial_y];
      const s_t *const RSTR tangent_grad_d0_0_grad0_0_q = &tangent[q];
      const s_t *const RSTR tangent_grad_d0_1_grad0_0_q = &tangent[(4 * NQ + q)];
      const s_t *const RSTR tangent_grad_d0_0_grad0_1_q = &tangent[(NQ + q)];
      const s_t *const RSTR tangent_grad_d0_1_grad0_1_q = &tangent[(5 * NQ + q)];
      const s_t *const RSTR tangent_grad_d0_0_grad1_0_q = &tangent[(2 * NQ + q)];
      const s_t *const RSTR tangent_grad_d0_1_grad1_0_q = &tangent[(6 * NQ + q)];
      const s_t *const RSTR tangent_grad_d0_0_grad1_1_q = &tangent[(3 * NQ + q)];
      const s_t *const RSTR tangent_grad_d0_1_grad1_1_q = &tangent[(7 * NQ + q)];
      s_t *const RSTR value_coeff_q0_c0 = &value_coeff_c0[q];
      s_t *const RSTR grad_coeff_ref_q0_0_c0 = &grad_coeff_ref_c0[(q * ND)];
      s_t *const RSTR grad_coeff_ref_q0_1_c0 = &grad_coeff_ref_c0[(q * ND + 1)];
      s_t *const RSTR value_coeff_q1_c0 = &value_coeff_c0[(NQ + q)];
      s_t *const RSTR grad_coeff_ref_q1_0_c0 = &grad_coeff_ref_c0[((NQ + q) * ND)];
      s_t *const RSTR grad_coeff_ref_q1_1_c0 = &grad_coeff_ref_c0[((NQ + q) * ND + 1)];
      const s_t *const RSTR tangent_grad_d1_0_grad0_0_q = &tangent[(8 * NQ + q)];
      const s_t *const RSTR tangent_grad_d1_1_grad0_0_q = &tangent[(12 * NQ + q)];
      const s_t *const RSTR tangent_grad_d1_0_grad0_1_q = &tangent[(9 * NQ + q)];
      const s_t *const RSTR tangent_grad_d1_1_grad0_1_q = &tangent[(13 * NQ + q)];
      const s_t *const RSTR tangent_grad_d1_0_grad1_0_q = &tangent[(10 * NQ + q)];
      const s_t *const RSTR tangent_grad_d1_1_grad1_0_q = &tangent[(14 * NQ + q)];
      const s_t *const RSTR tangent_grad_d1_0_grad1_1_q = &tangent[(11 * NQ + q)];
      const s_t *const RSTR tangent_grad_d1_1_grad1_1_q = &tangent[(15 * NQ + q)];
      s_t *const RSTR value_coeff_q0_c1 = &value_coeff_c1[q];
      s_t *const RSTR grad_coeff_ref_q0_0_c1 = &grad_coeff_ref_c1[(q * ND)];
      s_t *const RSTR grad_coeff_ref_q0_1_c1 = &grad_coeff_ref_c1[(q * ND + 1)];
      s_t *const RSTR value_coeff_q1_c1 = &value_coeff_c1[(NQ + q)];
      s_t *const RSTR grad_coeff_ref_q1_0_c1 = &grad_coeff_ref_c1[((NQ + q) * ND)];
      s_t *const RSTR grad_coeff_ref_q1_1_c1 = &grad_coeff_ref_c1[((NQ + q) * ND + 1)];
      {
        const s_t det = det_q[0];
        const s_t adj0 = adj_q0[0];
        const s_t adj1 = adj_q1[0];
        const s_t adj2 = adj_q2[0];
        const s_t adj3 = adj_q3[0];
        const s_t trial_grad0 = (trial_grad_ref0 * adj0 + trial_grad_ref1 * adj2) / det;
        const s_t trial_grad1 = (trial_grad_ref0 * adj1 + trial_grad_ref1 * adj3) / det;
        const s_t grad_coeff0_0_c0 = trial_grad0 * tangent_grad_d0_0_grad0_0_q[0] + trial_grad1 * tangent_grad_d0_1_grad0_0_q[0];
        const s_t grad_coeff0_1_c0 = trial_grad0 * tangent_grad_d0_0_grad0_1_q[0] + trial_grad1 * tangent_grad_d0_1_grad0_1_q[0];
        const s_t grad_coeff1_0_c0 = trial_grad0 * tangent_grad_d0_0_grad1_0_q[0] + trial_grad1 * tangent_grad_d0_1_grad1_0_q[0];
        const s_t grad_coeff1_1_c0 = trial_grad0 * tangent_grad_d0_0_grad1_1_q[0] + trial_grad1 * tangent_grad_d0_1_grad1_1_q[0];
        value_coeff_q0_c0[0] = s_t(0);
        grad_coeff_ref_q0_0_c0[0] = qw * (adj0 * grad_coeff0_0_c0 + adj1 * grad_coeff0_1_c0);
        grad_coeff_ref_q0_1_c0[0] = qw * (adj2 * grad_coeff0_0_c0 + adj3 * grad_coeff0_1_c0);
        value_coeff_q1_c0[0] = s_t(0);
        grad_coeff_ref_q1_0_c0[0] = qw * (adj0 * grad_coeff1_0_c0 + adj1 * grad_coeff1_1_c0);
        grad_coeff_ref_q1_1_c0[0] = qw * (adj2 * grad_coeff1_0_c0 + adj3 * grad_coeff1_1_c0);
        const s_t grad_coeff0_0_c1 = trial_grad0 * tangent_grad_d1_0_grad0_0_q[0] + trial_grad1 * tangent_grad_d1_1_grad0_0_q[0];
        const s_t grad_coeff0_1_c1 = trial_grad0 * tangent_grad_d1_0_grad0_1_q[0] + trial_grad1 * tangent_grad_d1_1_grad0_1_q[0];
        const s_t grad_coeff1_0_c1 = trial_grad0 * tangent_grad_d1_0_grad1_0_q[0] + trial_grad1 * tangent_grad_d1_1_grad1_0_q[0];
        const s_t grad_coeff1_1_c1 = trial_grad0 * tangent_grad_d1_0_grad1_1_q[0] + trial_grad1 * tangent_grad_d1_1_grad1_1_q[0];
        value_coeff_q0_c1[0] = s_t(0);
        grad_coeff_ref_q0_0_c1[0] = qw * (adj0 * grad_coeff0_0_c1 + adj1 * grad_coeff0_1_c1);
        grad_coeff_ref_q0_1_c1[0] = qw * (adj2 * grad_coeff0_0_c1 + adj3 * grad_coeff0_1_c1);
        value_coeff_q1_c1[0] = s_t(0);
        grad_coeff_ref_q1_0_c1[0] = qw * (adj0 * grad_coeff1_0_c1 + adj1 * grad_coeff1_1_c1);
        grad_coeff_ref_q1_1_c1[0] = qw * (adj2 * grad_coeff1_0_c1 + adj3 * grad_coeff1_1_c1);
      }
    }
    for (int out_shape = 0; out_shape < NS; ++out_shape) {
      column[out_shape * NC + 0] = &element_matrix[(0 * NS + out_shape) * 2 * NS + 0 * NS + trial];
      column[out_shape * NC + 1] = &element_matrix[(1 * NS + out_shape) * 2 * NS + 0 * NS + trial];
    }
    tensor_integrate_scalar<s_t, NQ, NS, ND, NC>(
        shape_1d, grad_1d, value_coeff_c0, grad_coeff_ref_c0, column);
    for (int out_shape = 0; out_shape < NS; ++out_shape) {
      column[out_shape * NC + 0] = &element_matrix[(0 * NS + out_shape) * 2 * NS + 1 * NS + trial];
      column[out_shape * NC + 1] = &element_matrix[(1 * NS + out_shape) * 2 * NS + 1 * NS + trial];
    }
    tensor_integrate_scalar<s_t, NQ, NS, ND, NC>(
        shape_1d, grad_1d, value_coeff_c1, grad_coeff_ref_c1, column);
  }
}

} // namespace codegen
} // namespace sfem

#endif
