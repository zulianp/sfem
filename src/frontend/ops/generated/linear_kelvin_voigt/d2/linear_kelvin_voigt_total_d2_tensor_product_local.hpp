#ifndef LINEAR_KELVIN_VOIGT_TOTAL_D2_TENSOR_PRODUCT_LOCAL_HPP
#define LINEAR_KELVIN_VOIGT_TOTAL_D2_TENSOR_PRODUCT_LOCAL_HPP

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
static SFEM_INLINE void linear_kelvin_voigt_total_d2_tensor_product_residual_block(
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
      const s_t residual_tmp0 = u1_grad_1*u_dt_shift;
      const s_t residual_tmp1 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp2 = eta_s*(-residual_tmp0 + residual_tmp1 - u1_old_grad_1);
      const s_t residual_tmp3 = s_t(2)*u0_grad_0;
      const s_t residual_tmp4 = s_t(2)*u1_grad_1;
      const s_t residual_tmp5 = eta_b*(residual_tmp0 + residual_tmp1 + u1_old_grad_1) + ((s_t(1) / s_t(2)))*lmbda*(residual_tmp3 + residual_tmp4);
      const s_t residual_tmp6 = eta_s*(u0_grad_1*u_dt_shift + u0_old_grad_1 + u1_grad_0*u_dt_shift + u1_old_grad_0) + mu*(u0_grad_1 + u1_grad_0);
      const s_t grad_coeff0_0 = mu*residual_tmp3 + residual_tmp2 + residual_tmp5;
      const s_t grad_coeff0_1 = residual_tmp6;
      const s_t grad_coeff1_0 = residual_tmp6;
      const s_t grad_coeff1_1 = mu*residual_tmp4 - residual_tmp2 + residual_tmp5;
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
static SFEM_INLINE void linear_kelvin_voigt_total_d2_tensor_product_residual_block_contiguous(
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
      const s_t residual_tmp0 = u1_grad_1*u_dt_shift;
      const s_t residual_tmp1 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp2 = eta_s*(-residual_tmp0 + residual_tmp1 - u1_old_grad_1);
      const s_t residual_tmp3 = s_t(2)*u0_grad_0;
      const s_t residual_tmp4 = s_t(2)*u1_grad_1;
      const s_t residual_tmp5 = eta_b*(residual_tmp0 + residual_tmp1 + u1_old_grad_1) + ((s_t(1) / s_t(2)))*lmbda*(residual_tmp3 + residual_tmp4);
      const s_t residual_tmp6 = eta_s*(u0_grad_1*u_dt_shift + u0_old_grad_1 + u1_grad_0*u_dt_shift + u1_old_grad_0) + mu*(u0_grad_1 + u1_grad_0);
      const s_t grad_coeff0_0 = mu*residual_tmp3 + residual_tmp2 + residual_tmp5;
      const s_t grad_coeff0_1 = residual_tmp6;
      const s_t grad_coeff1_0 = residual_tmp6;
      const s_t grad_coeff1_1 = mu*residual_tmp4 - residual_tmp2 + residual_tmp5;
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
static SFEM_INLINE void linear_kelvin_voigt_total_d2_tensor_product_jacobian_action_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR q_weight_1d,
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
    const s_t *const RSTR direction_grad_ref_q0_0 = &direction_grad_ref[(q * ND) * VS];
    const s_t *const RSTR direction_grad_ref_q0_1 = &direction_grad_ref[(q * ND + 1) * VS];
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
      const s_t u0_direction_grad_0_ref = direction_grad_ref_q0_0[lane];
      const s_t u0_direction_grad_1_ref = direction_grad_ref_q0_1[lane];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
      const s_t u1_direction_grad_0_ref = direction_grad_ref_q1_0[lane];
      const s_t u1_direction_grad_1_ref = direction_grad_ref_q1_1[lane];
      const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj2) / det;
      const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj3) / det;
      const s_t residual_tmp0 = eta_s*u_dt_shift;
      const s_t residual_tmp1 = eta_b*u_dt_shift + lmbda;
      const s_t residual_tmp2 = -residual_tmp0 + residual_tmp1;
      const s_t residual_tmp3 = s_t(2)*mu + residual_tmp0 + residual_tmp1;
      const s_t residual_tmp4 = mu + residual_tmp0;
      const s_t residual_tmp5 = residual_tmp4*u0_direction_grad_1 + residual_tmp4*u1_direction_grad_0;
      const s_t grad_coeff0_0 = residual_tmp2*u1_direction_grad_1 + residual_tmp3*u0_direction_grad_0;
      const s_t grad_coeff0_1 = residual_tmp5;
      const s_t grad_coeff1_0 = residual_tmp5;
      const s_t grad_coeff1_1 = residual_tmp2*u0_direction_grad_0 + residual_tmp3*u1_direction_grad_1;
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
static SFEM_INLINE void linear_kelvin_voigt_total_d2_tensor_product_jacobian_action_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR q_weight_1d,
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
    const s_t *const RSTR direction_grad_ref_q0_0 = &direction_grad_ref[(q * ND) * VS];
    const s_t *const RSTR direction_grad_ref_q0_1 = &direction_grad_ref[(q * ND + 1) * VS];
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
      const s_t u0_direction_grad_0_ref = direction_grad_ref_q0_0[lane];
      const s_t u0_direction_grad_1_ref = direction_grad_ref_q0_1[lane];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
      const s_t u1_direction_grad_0_ref = direction_grad_ref_q1_0[lane];
      const s_t u1_direction_grad_1_ref = direction_grad_ref_q1_1[lane];
      const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj2) / det;
      const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj3) / det;
      const s_t residual_tmp0 = eta_s*u_dt_shift;
      const s_t residual_tmp1 = eta_b*u_dt_shift + lmbda;
      const s_t residual_tmp2 = -residual_tmp0 + residual_tmp1;
      const s_t residual_tmp3 = s_t(2)*mu + residual_tmp0 + residual_tmp1;
      const s_t residual_tmp4 = mu + residual_tmp0;
      const s_t residual_tmp5 = residual_tmp4*u0_direction_grad_1 + residual_tmp4*u1_direction_grad_0;
      const s_t grad_coeff0_0 = residual_tmp2*u1_direction_grad_1 + residual_tmp3*u0_direction_grad_0;
      const s_t grad_coeff0_1 = residual_tmp5;
      const s_t grad_coeff1_0 = residual_tmp5;
      const s_t grad_coeff1_1 = residual_tmp2*u0_direction_grad_0 + residual_tmp3*u1_direction_grad_1;
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

template <typename s_t, int NQ, int NS>
static SFEM_INLINE void linear_kelvin_voigt_total_d2_tensor_product_hessian_block(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR q_weight_1d,
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
      const s_t tangent_tmp0 = eta_s*u_dt_shift;
      const s_t tangent_tmp1 = eta_b*u_dt_shift + lmbda;
      const s_t tangent_tmp2 = s_t(2)*mu + tangent_tmp0 + tangent_tmp1;
      const s_t tangent_tmp3 = -tangent_tmp0 + tangent_tmp1;
      const s_t tangent_tmp4 = mu + tangent_tmp0;
      tangent_grad_d0_0_grad0_0_q[0] = tangent_tmp2;
      tangent_grad_d0_0_grad0_1_q[0] = 0;
      tangent_grad_d0_0_grad1_0_q[0] = 0;
      tangent_grad_d0_0_grad1_1_q[0] = tangent_tmp3;
      tangent_grad_d0_1_grad0_0_q[0] = 0;
      tangent_grad_d0_1_grad0_1_q[0] = tangent_tmp4;
      tangent_grad_d0_1_grad1_0_q[0] = tangent_tmp4;
      tangent_grad_d0_1_grad1_1_q[0] = 0;
      tangent_grad_d1_0_grad0_0_q[0] = 0;
      tangent_grad_d1_0_grad0_1_q[0] = tangent_tmp4;
      tangent_grad_d1_0_grad1_0_q[0] = tangent_tmp4;
      tangent_grad_d1_0_grad1_1_q[0] = 0;
      tangent_grad_d1_1_grad0_0_q[0] = tangent_tmp3;
      tangent_grad_d1_1_grad0_1_q[0] = 0;
      tangent_grad_d1_1_grad1_0_q[0] = 0;
      tangent_grad_d1_1_grad1_1_q[0] = tangent_tmp2;
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
