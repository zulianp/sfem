#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_VISCOUS_D2_TENSOR_PRODUCT_LOCAL_HPP
#define MOONEY_RIVLIN_KELVIN_VOIGT_VISCOUS_D2_TENSOR_PRODUCT_LOCAL_HPP

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
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_viscous_d2_tensor_product_residual_block(
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
      const s_t residual_tmp1 = pow_m1(residual_tmp0 + u0_grad_0*u1_grad_1 + u0_grad_0 - u0_grad_1*u1_grad_0);
      const s_t residual_tmp2 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp3 = u1_grad_1*u_dt_shift + u1_old_grad_1;
      const s_t residual_tmp4 = u0_grad_0 + s_t(1);
      const s_t residual_tmp5 = u0_grad_1*u_dt_shift + u0_old_grad_1;
      const s_t residual_tmp6 = u1_grad_0*u_dt_shift + u1_old_grad_0;
      const s_t residual_tmp7 = eta_s*(-residual_tmp0*residual_tmp6 + residual_tmp2*u0_grad_1 + residual_tmp3*u1_grad_0 - residual_tmp4*residual_tmp5);
      const s_t residual_tmp8 = residual_tmp5*u1_grad_0;
      const s_t residual_tmp9 = residual_tmp0*residual_tmp2;
      const s_t residual_tmp10 = -residual_tmp3*residual_tmp4 + residual_tmp6*u0_grad_1;
      const s_t residual_tmp11 = eta_b*(residual_tmp10 + residual_tmp8 - residual_tmp9);
      const s_t residual_tmp12 = residual_tmp10 - residual_tmp8 + residual_tmp9;
      const s_t residual_tmp13 = -eta_s*residual_tmp12 + residual_tmp11;
      const s_t residual_tmp14 = eta_s*residual_tmp12 + residual_tmp11;
      const s_t grad_coeff0_0 = residual_tmp1*(-residual_tmp0*residual_tmp13 + residual_tmp7*u0_grad_1);
      const s_t grad_coeff0_1 = residual_tmp1*(residual_tmp13*u1_grad_0 - residual_tmp4*residual_tmp7);
      const s_t grad_coeff1_0 = residual_tmp1*(-residual_tmp0*residual_tmp7 + residual_tmp14*u0_grad_1);
      const s_t grad_coeff1_1 = residual_tmp1*(-residual_tmp14*residual_tmp4 + residual_tmp7*u1_grad_0);
      value_coeff_q0[0] = s_t(0);
      grad_coeff_ref_q0_0[0] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
      grad_coeff_ref_q0_1[0] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
      value_coeff_q1[0] = s_t(0);
      grad_coeff_ref_q1_0[0] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
      grad_coeff_ref_q1_1[0] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
    }
  }
  tensor_integrate<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_viscous_d2_tensor_product_residual_block_contiguous(
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
      const s_t residual_tmp1 = pow_m1(residual_tmp0 + u0_grad_0*u1_grad_1 + u0_grad_0 - u0_grad_1*u1_grad_0);
      const s_t residual_tmp2 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp3 = u1_grad_1*u_dt_shift + u1_old_grad_1;
      const s_t residual_tmp4 = u0_grad_0 + s_t(1);
      const s_t residual_tmp5 = u0_grad_1*u_dt_shift + u0_old_grad_1;
      const s_t residual_tmp6 = u1_grad_0*u_dt_shift + u1_old_grad_0;
      const s_t residual_tmp7 = eta_s*(-residual_tmp0*residual_tmp6 + residual_tmp2*u0_grad_1 + residual_tmp3*u1_grad_0 - residual_tmp4*residual_tmp5);
      const s_t residual_tmp8 = residual_tmp5*u1_grad_0;
      const s_t residual_tmp9 = residual_tmp0*residual_tmp2;
      const s_t residual_tmp10 = -residual_tmp3*residual_tmp4 + residual_tmp6*u0_grad_1;
      const s_t residual_tmp11 = eta_b*(residual_tmp10 + residual_tmp8 - residual_tmp9);
      const s_t residual_tmp12 = residual_tmp10 - residual_tmp8 + residual_tmp9;
      const s_t residual_tmp13 = -eta_s*residual_tmp12 + residual_tmp11;
      const s_t residual_tmp14 = eta_s*residual_tmp12 + residual_tmp11;
      const s_t grad_coeff0_0 = residual_tmp1*(-residual_tmp0*residual_tmp13 + residual_tmp7*u0_grad_1);
      const s_t grad_coeff0_1 = residual_tmp1*(residual_tmp13*u1_grad_0 - residual_tmp4*residual_tmp7);
      const s_t grad_coeff1_0 = residual_tmp1*(-residual_tmp0*residual_tmp7 + residual_tmp14*u0_grad_1);
      const s_t grad_coeff1_1 = residual_tmp1*(-residual_tmp14*residual_tmp4 + residual_tmp7*u1_grad_0);
      value_coeff_q0[0] = s_t(0);
      grad_coeff_ref_q0_0[0] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
      grad_coeff_ref_q0_1[0] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
      value_coeff_q1[0] = s_t(0);
      grad_coeff_ref_q1_0[0] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
      grad_coeff_ref_q1_1[0] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
    }
  }
  tensor_integrate_contiguous<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_viscous_d2_tensor_product_jacobian_action_block(
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
      const s_t residual_tmp0 = u1_grad_1 + s_t(1);
      const s_t residual_tmp1 = residual_tmp0 + u0_grad_0*u1_grad_1 + u0_grad_0 - u0_grad_1*u1_grad_0;
      const s_t residual_tmp2 = pow_m1(residual_tmp1);
      const s_t residual_tmp3 = residual_tmp0*u_dt_shift;
      const s_t residual_tmp4 = u1_grad_1*u_dt_shift + u1_old_grad_1;
      const s_t residual_tmp5 = -residual_tmp3 + residual_tmp4;
      const s_t residual_tmp6 = eta_s*residual_tmp5;
      const s_t residual_tmp7 = eta_s*u0_old_grad_1;
      const s_t residual_tmp8 = u0_grad_1*u_dt_shift;
      const s_t residual_tmp9 = eta_b*(s_t(2)*residual_tmp8 + u0_old_grad_1);
      const s_t residual_tmp10 = residual_tmp7 + residual_tmp9;
      const s_t residual_tmp11 = pow_m2(residual_tmp1);
      const s_t residual_tmp12 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp13 = u0_grad_0 + s_t(1);
      const s_t residual_tmp14 = residual_tmp8 + u0_old_grad_1;
      const s_t residual_tmp15 = u1_grad_0*u_dt_shift;
      const s_t residual_tmp16 = residual_tmp15 + u1_old_grad_0;
      const s_t residual_tmp17 = eta_s*(-residual_tmp0*residual_tmp16 + residual_tmp12*u0_grad_1 - residual_tmp13*residual_tmp14 + residual_tmp4*u1_grad_0);
      const s_t residual_tmp18 = residual_tmp14*u1_grad_0;
      const s_t residual_tmp19 = residual_tmp0*residual_tmp12;
      const s_t residual_tmp20 = -residual_tmp13*residual_tmp4 + residual_tmp16*u0_grad_1;
      const s_t residual_tmp21 = eta_b*(residual_tmp18 - residual_tmp19 + residual_tmp20);
      const s_t residual_tmp22 = eta_s*(-residual_tmp18 + residual_tmp19 + residual_tmp20);
      const s_t residual_tmp23 = residual_tmp21 - residual_tmp22;
      const s_t residual_tmp24 = residual_tmp11*(-residual_tmp0*residual_tmp23 + residual_tmp17*u0_grad_1);
      const s_t residual_tmp25 = eta_b*(-residual_tmp3 - residual_tmp4);
      const s_t residual_tmp26 = -residual_tmp5;
      const s_t residual_tmp27 = -eta_s*residual_tmp26 + residual_tmp25;
      const s_t residual_tmp28 = -residual_tmp0;
      const s_t residual_tmp29 = residual_tmp13*u_dt_shift;
      const s_t residual_tmp30 = residual_tmp12 - residual_tmp29;
      const s_t residual_tmp31 = eta_s*residual_tmp30;
      const s_t residual_tmp32 = eta_b*(s_t(2)*residual_tmp15 + u1_old_grad_0);
      const s_t residual_tmp33 = -eta_s*u1_old_grad_0 + residual_tmp32;
      const s_t residual_tmp34 = eta_b*(-residual_tmp12 - residual_tmp29);
      const s_t residual_tmp35 = -eta_s*residual_tmp30 + residual_tmp34;
      const s_t residual_tmp36 = eta_s*u1_old_grad_0;
      const s_t residual_tmp37 = -residual_tmp13;
      const s_t residual_tmp38 = residual_tmp11*(-residual_tmp13*residual_tmp17 + residual_tmp23*u1_grad_0);
      const s_t residual_tmp39 = -residual_tmp17;
      const s_t residual_tmp40 = -residual_tmp7 + residual_tmp9;
      const s_t residual_tmp41 = residual_tmp21 + residual_tmp22;
      const s_t residual_tmp42 = residual_tmp11*(-residual_tmp0*residual_tmp17 + residual_tmp41*u0_grad_1);
      const s_t residual_tmp43 = eta_s*residual_tmp26 + residual_tmp25;
      const s_t residual_tmp44 = residual_tmp31 + residual_tmp34;
      const s_t residual_tmp45 = residual_tmp32 + residual_tmp36;
      const s_t residual_tmp46 = residual_tmp11*(-residual_tmp13*residual_tmp41 + residual_tmp17*u1_grad_0);
      const s_t grad_coeff0_0 = u0_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp27 - residual_tmp7*u0_grad_1) + residual_tmp24*residual_tmp28) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp33 + residual_tmp17 + residual_tmp31*u0_grad_1) + residual_tmp24*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp10 + residual_tmp6*u0_grad_1) + residual_tmp24*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp35 - residual_tmp23 - residual_tmp36*u0_grad_1) + residual_tmp24*residual_tmp37);
      const s_t grad_coeff0_1 = u0_direction_grad_0*(residual_tmp2*(residual_tmp13*residual_tmp7 + residual_tmp27*u1_grad_0 + residual_tmp39) + residual_tmp28*residual_tmp38) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp31 + residual_tmp33*u1_grad_0) + residual_tmp38*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(residual_tmp10*u1_grad_0 - residual_tmp13*residual_tmp6 + residual_tmp23) + residual_tmp38*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(residual_tmp13*residual_tmp36 + residual_tmp35*u1_grad_0) + residual_tmp37*residual_tmp38);
      const s_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp2*(residual_tmp0*residual_tmp7 + residual_tmp43*u0_grad_1) + residual_tmp28*residual_tmp42) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp31 + residual_tmp41 + residual_tmp45*u0_grad_1) + residual_tmp42*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp6 + residual_tmp40*u0_grad_1) + residual_tmp42*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(residual_tmp0*residual_tmp36 + residual_tmp39 + residual_tmp44*u0_grad_1) + residual_tmp37*residual_tmp42);
      const s_t grad_coeff1_1 = u0_direction_grad_0*(residual_tmp2*(-residual_tmp13*residual_tmp43 - residual_tmp41 - residual_tmp7*u1_grad_0) + residual_tmp28*residual_tmp46) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp45 + residual_tmp31*u1_grad_0) + residual_tmp46*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp13*residual_tmp40 + residual_tmp17 + residual_tmp6*u1_grad_0) + residual_tmp46*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp44 - residual_tmp36*u1_grad_0) + residual_tmp37*residual_tmp46);
      value_coeff_q0[0] = s_t(0);
      grad_coeff_ref_q0_0[0] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
      grad_coeff_ref_q0_1[0] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
      value_coeff_q1[0] = s_t(0);
      grad_coeff_ref_q1_0[0] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
      grad_coeff_ref_q1_1[0] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
    }
  }
  tensor_integrate<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_viscous_d2_tensor_product_jacobian_action_block_contiguous(
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
      const s_t residual_tmp0 = u1_grad_1 + s_t(1);
      const s_t residual_tmp1 = residual_tmp0 + u0_grad_0*u1_grad_1 + u0_grad_0 - u0_grad_1*u1_grad_0;
      const s_t residual_tmp2 = pow_m1(residual_tmp1);
      const s_t residual_tmp3 = residual_tmp0*u_dt_shift;
      const s_t residual_tmp4 = u1_grad_1*u_dt_shift + u1_old_grad_1;
      const s_t residual_tmp5 = -residual_tmp3 + residual_tmp4;
      const s_t residual_tmp6 = eta_s*residual_tmp5;
      const s_t residual_tmp7 = eta_s*u0_old_grad_1;
      const s_t residual_tmp8 = u0_grad_1*u_dt_shift;
      const s_t residual_tmp9 = eta_b*(s_t(2)*residual_tmp8 + u0_old_grad_1);
      const s_t residual_tmp10 = residual_tmp7 + residual_tmp9;
      const s_t residual_tmp11 = pow_m2(residual_tmp1);
      const s_t residual_tmp12 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp13 = u0_grad_0 + s_t(1);
      const s_t residual_tmp14 = residual_tmp8 + u0_old_grad_1;
      const s_t residual_tmp15 = u1_grad_0*u_dt_shift;
      const s_t residual_tmp16 = residual_tmp15 + u1_old_grad_0;
      const s_t residual_tmp17 = eta_s*(-residual_tmp0*residual_tmp16 + residual_tmp12*u0_grad_1 - residual_tmp13*residual_tmp14 + residual_tmp4*u1_grad_0);
      const s_t residual_tmp18 = residual_tmp14*u1_grad_0;
      const s_t residual_tmp19 = residual_tmp0*residual_tmp12;
      const s_t residual_tmp20 = -residual_tmp13*residual_tmp4 + residual_tmp16*u0_grad_1;
      const s_t residual_tmp21 = eta_b*(residual_tmp18 - residual_tmp19 + residual_tmp20);
      const s_t residual_tmp22 = eta_s*(-residual_tmp18 + residual_tmp19 + residual_tmp20);
      const s_t residual_tmp23 = residual_tmp21 - residual_tmp22;
      const s_t residual_tmp24 = residual_tmp11*(-residual_tmp0*residual_tmp23 + residual_tmp17*u0_grad_1);
      const s_t residual_tmp25 = eta_b*(-residual_tmp3 - residual_tmp4);
      const s_t residual_tmp26 = -residual_tmp5;
      const s_t residual_tmp27 = -eta_s*residual_tmp26 + residual_tmp25;
      const s_t residual_tmp28 = -residual_tmp0;
      const s_t residual_tmp29 = residual_tmp13*u_dt_shift;
      const s_t residual_tmp30 = residual_tmp12 - residual_tmp29;
      const s_t residual_tmp31 = eta_s*residual_tmp30;
      const s_t residual_tmp32 = eta_b*(s_t(2)*residual_tmp15 + u1_old_grad_0);
      const s_t residual_tmp33 = -eta_s*u1_old_grad_0 + residual_tmp32;
      const s_t residual_tmp34 = eta_b*(-residual_tmp12 - residual_tmp29);
      const s_t residual_tmp35 = -eta_s*residual_tmp30 + residual_tmp34;
      const s_t residual_tmp36 = eta_s*u1_old_grad_0;
      const s_t residual_tmp37 = -residual_tmp13;
      const s_t residual_tmp38 = residual_tmp11*(-residual_tmp13*residual_tmp17 + residual_tmp23*u1_grad_0);
      const s_t residual_tmp39 = -residual_tmp17;
      const s_t residual_tmp40 = -residual_tmp7 + residual_tmp9;
      const s_t residual_tmp41 = residual_tmp21 + residual_tmp22;
      const s_t residual_tmp42 = residual_tmp11*(-residual_tmp0*residual_tmp17 + residual_tmp41*u0_grad_1);
      const s_t residual_tmp43 = eta_s*residual_tmp26 + residual_tmp25;
      const s_t residual_tmp44 = residual_tmp31 + residual_tmp34;
      const s_t residual_tmp45 = residual_tmp32 + residual_tmp36;
      const s_t residual_tmp46 = residual_tmp11*(-residual_tmp13*residual_tmp41 + residual_tmp17*u1_grad_0);
      const s_t grad_coeff0_0 = u0_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp27 - residual_tmp7*u0_grad_1) + residual_tmp24*residual_tmp28) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp33 + residual_tmp17 + residual_tmp31*u0_grad_1) + residual_tmp24*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp10 + residual_tmp6*u0_grad_1) + residual_tmp24*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp35 - residual_tmp23 - residual_tmp36*u0_grad_1) + residual_tmp24*residual_tmp37);
      const s_t grad_coeff0_1 = u0_direction_grad_0*(residual_tmp2*(residual_tmp13*residual_tmp7 + residual_tmp27*u1_grad_0 + residual_tmp39) + residual_tmp28*residual_tmp38) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp31 + residual_tmp33*u1_grad_0) + residual_tmp38*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(residual_tmp10*u1_grad_0 - residual_tmp13*residual_tmp6 + residual_tmp23) + residual_tmp38*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(residual_tmp13*residual_tmp36 + residual_tmp35*u1_grad_0) + residual_tmp37*residual_tmp38);
      const s_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp2*(residual_tmp0*residual_tmp7 + residual_tmp43*u0_grad_1) + residual_tmp28*residual_tmp42) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp31 + residual_tmp41 + residual_tmp45*u0_grad_1) + residual_tmp42*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp6 + residual_tmp40*u0_grad_1) + residual_tmp42*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(residual_tmp0*residual_tmp36 + residual_tmp39 + residual_tmp44*u0_grad_1) + residual_tmp37*residual_tmp42);
      const s_t grad_coeff1_1 = u0_direction_grad_0*(residual_tmp2*(-residual_tmp13*residual_tmp43 - residual_tmp41 - residual_tmp7*u1_grad_0) + residual_tmp28*residual_tmp46) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp45 + residual_tmp31*u1_grad_0) + residual_tmp46*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp13*residual_tmp40 + residual_tmp17 + residual_tmp6*u1_grad_0) + residual_tmp46*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp44 - residual_tmp36*u1_grad_0) + residual_tmp37*residual_tmp46);
      value_coeff_q0[0] = s_t(0);
      grad_coeff_ref_q0_0[0] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
      grad_coeff_ref_q0_1[0] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
      value_coeff_q1[0] = s_t(0);
      grad_coeff_ref_q1_0[0] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
      grad_coeff_ref_q1_1[0] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
    }
  }
  tensor_integrate_contiguous<s_t, NQ, NS, VS, ND, NC>(
      ne, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_viscous_d2_tensor_product_hessian_block(
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
    const s_t u_dt_shift,
    s_t *const RSTR element_matrix
) {
  static constexpr int ND = 2;
  static constexpr int NC = 2;
  for (int entry = 0; entry < 2 * NS * 2 * NS; ++entry) {
    element_matrix[entry] = s_t(0);
  }
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
  static constexpr int NS1 = integer_root(NS, ND);
  s_t * column[NC * NS];
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
        const s_t trial_grad_ref0 = grad_1d[q_x * NS1 + trial_x] * shape_1d[q_y * NS1 + trial_y];
        const s_t trial_grad_ref1 = shape_1d[q_x * NS1 + trial_x] * grad_1d[q_y * NS1 + trial_y];
        const s_t trial_grad0 = (trial_grad_ref0 * adj0 + trial_grad_ref1 * adj2) / det;
        const s_t trial_grad1 = (trial_grad_ref0 * adj1 + trial_grad_ref1 * adj3) / det;
        const s_t residual_tmp0 = u1_grad_1 + s_t(1);
        const s_t residual_tmp1 = residual_tmp0 + u0_grad_0*u1_grad_1 + u0_grad_0 - u0_grad_1*u1_grad_0;
        const s_t residual_tmp2 = pow_m1(residual_tmp1);
        const s_t residual_tmp3 = eta_s*u0_grad_1;
        const s_t residual_tmp4 = residual_tmp0*u_dt_shift;
        const s_t residual_tmp5 = u1_grad_1*u_dt_shift + u1_old_grad_1;
        const s_t residual_tmp6 = eta_b*(-residual_tmp4 - residual_tmp5);
        const s_t residual_tmp7 = residual_tmp4 - residual_tmp5;
        const s_t residual_tmp8 = -eta_s*residual_tmp7 + residual_tmp6;
        const s_t residual_tmp9 = -residual_tmp0;
        const s_t residual_tmp10 = pow_m2(residual_tmp1);
        const s_t residual_tmp11 = u0_grad_0*u_dt_shift + u0_old_grad_0;
        const s_t residual_tmp12 = u0_grad_0 + s_t(1);
        const s_t residual_tmp13 = u0_grad_1*u_dt_shift + u0_old_grad_1;
        const s_t residual_tmp14 = u1_grad_0*u_dt_shift;
        const s_t residual_tmp15 = residual_tmp14 + u1_old_grad_0;
        const s_t residual_tmp16 = eta_s*(-residual_tmp0*residual_tmp15 + residual_tmp11*u0_grad_1 - residual_tmp12*residual_tmp13 + residual_tmp5*u1_grad_0);
        const s_t residual_tmp17 = residual_tmp13*u1_grad_0;
        const s_t residual_tmp18 = residual_tmp0*residual_tmp11;
        const s_t residual_tmp19 = -residual_tmp12*residual_tmp5 + residual_tmp15*u0_grad_1;
        const s_t residual_tmp20 = eta_b*(residual_tmp17 - residual_tmp18 + residual_tmp19);
        const s_t residual_tmp21 = eta_s*(-residual_tmp17 + residual_tmp18 + residual_tmp19);
        const s_t residual_tmp22 = residual_tmp20 - residual_tmp21;
        const s_t residual_tmp23 = residual_tmp10*(-residual_tmp0*residual_tmp22 + residual_tmp16*u0_grad_1);
        const s_t residual_tmp24 = residual_tmp11 - residual_tmp12*u_dt_shift;
        const s_t residual_tmp25 = eta_b*(s_t(2)*residual_tmp14 + u1_old_grad_0);
        const s_t residual_tmp26 = -eta_s*u1_old_grad_0 + residual_tmp25;
        const s_t residual_tmp27 = eta_s*residual_tmp12;
        const s_t residual_tmp28 = residual_tmp10*(-residual_tmp12*residual_tmp16 + residual_tmp22*u1_grad_0);
        const s_t residual_tmp29 = eta_s*residual_tmp0;
        const s_t residual_tmp30 = eta_s*residual_tmp7 + residual_tmp6;
        const s_t residual_tmp31 = residual_tmp20 + residual_tmp21;
        const s_t residual_tmp32 = residual_tmp10*(-residual_tmp0*residual_tmp16 + residual_tmp31*u0_grad_1);
        const s_t residual_tmp33 = eta_s*u1_old_grad_0 + residual_tmp25;
        const s_t residual_tmp34 = eta_s*u1_grad_0;
        const s_t residual_tmp35 = residual_tmp10*(-residual_tmp12*residual_tmp31 + residual_tmp16*u1_grad_0);
        const s_t grad_coeff0_0 = trial_grad0*(residual_tmp2*(-residual_tmp0*residual_tmp8 - residual_tmp3*u0_old_grad_1) + residual_tmp23*residual_tmp9) + trial_grad1*(residual_tmp2*(-residual_tmp0*residual_tmp26 + residual_tmp16 + residual_tmp24*residual_tmp3) + residual_tmp23*u1_grad_0);
        const s_t grad_coeff0_1 = trial_grad0*(residual_tmp2*(-residual_tmp16 + residual_tmp27*u0_old_grad_1 + residual_tmp8*u1_grad_0) + residual_tmp28*residual_tmp9) + trial_grad1*(residual_tmp2*(-residual_tmp24*residual_tmp27 + residual_tmp26*u1_grad_0) + residual_tmp28*u1_grad_0);
        const s_t grad_coeff1_0 = trial_grad0*(residual_tmp2*(residual_tmp29*u0_old_grad_1 + residual_tmp30*u0_grad_1) + residual_tmp32*residual_tmp9) + trial_grad1*(residual_tmp2*(-residual_tmp24*residual_tmp29 + residual_tmp31 + residual_tmp33*u0_grad_1) + residual_tmp32*u1_grad_0);
        const s_t grad_coeff1_1 = trial_grad0*(residual_tmp2*(-residual_tmp12*residual_tmp30 - residual_tmp31 - residual_tmp34*u0_old_grad_1) + residual_tmp35*residual_tmp9) + trial_grad1*(residual_tmp2*(-residual_tmp12*residual_tmp33 + residual_tmp24*residual_tmp34) + residual_tmp35*u1_grad_0);
        value_coeff_q0[0] = s_t(0);
        grad_coeff_ref_q0_0[0] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
        grad_coeff_ref_q0_1[0] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
        value_coeff_q1[0] = s_t(0);
        grad_coeff_ref_q1_0[0] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
        grad_coeff_ref_q1_1[0] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
      }
    }
    for (int out_shape = 0; out_shape < NS; ++out_shape) {
      column[out_shape * NC + 0] = &element_matrix[(0 * NS + out_shape) * 2 * NS + 0 * NS + trial];
      column[out_shape * NC + 1] = &element_matrix[(1 * NS + out_shape) * 2 * NS + 0 * NS + trial];
    }
    tensor_integrate<s_t, NQ, NS, VS, ND, NC>(
        ne, shape_1d, grad_1d, value_coeff, grad_coeff_ref, column);
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
        const s_t trial_grad_ref0 = grad_1d[q_x * NS1 + trial_x] * shape_1d[q_y * NS1 + trial_y];
        const s_t trial_grad_ref1 = shape_1d[q_x * NS1 + trial_x] * grad_1d[q_y * NS1 + trial_y];
        const s_t trial_grad0 = (trial_grad_ref0 * adj0 + trial_grad_ref1 * adj2) / det;
        const s_t trial_grad1 = (trial_grad_ref0 * adj1 + trial_grad_ref1 * adj3) / det;
        const s_t residual_tmp0 = u1_grad_1 + s_t(1);
        const s_t residual_tmp1 = residual_tmp0 + u0_grad_0*u1_grad_1 + u0_grad_0 - u0_grad_1*u1_grad_0;
        const s_t residual_tmp2 = pow_m1(residual_tmp1);
        const s_t residual_tmp3 = u1_grad_1*u_dt_shift + u1_old_grad_1;
        const s_t residual_tmp4 = -residual_tmp0*u_dt_shift + residual_tmp3;
        const s_t residual_tmp5 = eta_s*u0_grad_1;
        const s_t residual_tmp6 = eta_s*u0_old_grad_1;
        const s_t residual_tmp7 = u0_grad_1*u_dt_shift;
        const s_t residual_tmp8 = eta_b*(s_t(2)*residual_tmp7 + u0_old_grad_1);
        const s_t residual_tmp9 = residual_tmp6 + residual_tmp8;
        const s_t residual_tmp10 = pow_m2(residual_tmp1);
        const s_t residual_tmp11 = u0_grad_0*u_dt_shift + u0_old_grad_0;
        const s_t residual_tmp12 = u0_grad_0 + s_t(1);
        const s_t residual_tmp13 = residual_tmp7 + u0_old_grad_1;
        const s_t residual_tmp14 = u1_grad_0*u_dt_shift + u1_old_grad_0;
        const s_t residual_tmp15 = eta_s*(-residual_tmp0*residual_tmp14 + residual_tmp11*u0_grad_1 - residual_tmp12*residual_tmp13 + residual_tmp3*u1_grad_0);
        const s_t residual_tmp16 = residual_tmp13*u1_grad_0;
        const s_t residual_tmp17 = residual_tmp0*residual_tmp11;
        const s_t residual_tmp18 = -residual_tmp12*residual_tmp3 + residual_tmp14*u0_grad_1;
        const s_t residual_tmp19 = eta_b*(residual_tmp16 - residual_tmp17 + residual_tmp18);
        const s_t residual_tmp20 = eta_s*(-residual_tmp16 + residual_tmp17 + residual_tmp18);
        const s_t residual_tmp21 = residual_tmp19 - residual_tmp20;
        const s_t residual_tmp22 = residual_tmp10*(-residual_tmp0*residual_tmp21 + residual_tmp15*u0_grad_1);
        const s_t residual_tmp23 = residual_tmp12*u_dt_shift;
        const s_t residual_tmp24 = residual_tmp11 - residual_tmp23;
        const s_t residual_tmp25 = eta_b*(-residual_tmp11 - residual_tmp23);
        const s_t residual_tmp26 = -eta_s*residual_tmp24 + residual_tmp25;
        const s_t residual_tmp27 = -residual_tmp12;
        const s_t residual_tmp28 = eta_s*residual_tmp12;
        const s_t residual_tmp29 = residual_tmp10*(-residual_tmp12*residual_tmp15 + residual_tmp21*u1_grad_0);
        const s_t residual_tmp30 = -residual_tmp6 + residual_tmp8;
        const s_t residual_tmp31 = eta_s*residual_tmp0;
        const s_t residual_tmp32 = residual_tmp19 + residual_tmp20;
        const s_t residual_tmp33 = residual_tmp10*(-residual_tmp0*residual_tmp15 + residual_tmp32*u0_grad_1);
        const s_t residual_tmp34 = eta_s*residual_tmp24 + residual_tmp25;
        const s_t residual_tmp35 = eta_s*u1_grad_0;
        const s_t residual_tmp36 = residual_tmp10*(-residual_tmp12*residual_tmp32 + residual_tmp15*u1_grad_0);
        const s_t grad_coeff0_0 = trial_grad0*(residual_tmp2*(-residual_tmp0*residual_tmp9 + residual_tmp4*residual_tmp5) + residual_tmp22*u0_grad_1) + trial_grad1*(residual_tmp2*(-residual_tmp0*residual_tmp26 - residual_tmp21 - residual_tmp5*u1_old_grad_0) + residual_tmp22*residual_tmp27);
        const s_t grad_coeff0_1 = trial_grad0*(residual_tmp2*(residual_tmp21 - residual_tmp28*residual_tmp4 + residual_tmp9*u1_grad_0) + residual_tmp29*u0_grad_1) + trial_grad1*(residual_tmp2*(residual_tmp26*u1_grad_0 + residual_tmp28*u1_old_grad_0) + residual_tmp27*residual_tmp29);
        const s_t grad_coeff1_0 = trial_grad0*(residual_tmp2*(residual_tmp30*u0_grad_1 - residual_tmp31*residual_tmp4) + residual_tmp33*u0_grad_1) + trial_grad1*(residual_tmp2*(-residual_tmp15 + residual_tmp31*u1_old_grad_0 + residual_tmp34*u0_grad_1) + residual_tmp27*residual_tmp33);
        const s_t grad_coeff1_1 = trial_grad0*(residual_tmp2*(-residual_tmp12*residual_tmp30 + residual_tmp15 + residual_tmp35*residual_tmp4) + residual_tmp36*u0_grad_1) + trial_grad1*(residual_tmp2*(-residual_tmp12*residual_tmp34 - residual_tmp35*u1_old_grad_0) + residual_tmp27*residual_tmp36);
        value_coeff_q0[0] = s_t(0);
        grad_coeff_ref_q0_0[0] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
        grad_coeff_ref_q0_1[0] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
        value_coeff_q1[0] = s_t(0);
        grad_coeff_ref_q1_0[0] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
        grad_coeff_ref_q1_1[0] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
      }
    }
    for (int out_shape = 0; out_shape < NS; ++out_shape) {
      column[out_shape * NC + 0] = &element_matrix[(0 * NS + out_shape) * 2 * NS + 1 * NS + trial];
      column[out_shape * NC + 1] = &element_matrix[(1 * NS + out_shape) * 2 * NS + 1 * NS + trial];
    }
    tensor_integrate<s_t, NQ, NS, VS, ND, NC>(
        ne, shape_1d, grad_1d, value_coeff, grad_coeff_ref, column);
  }
}

} // namespace codegen
} // namespace sfem

#endif
