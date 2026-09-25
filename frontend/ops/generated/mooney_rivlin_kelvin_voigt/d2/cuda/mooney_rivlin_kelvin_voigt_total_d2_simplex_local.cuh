#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_TOTAL_D2_SIMPLEX_LOCAL_HPP
#define MOONEY_RIVLIN_KELVIN_VOIGT_TOTAL_D2_SIMPLEX_LOCAL_HPP

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
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_total_d2_simplex_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[2 * NS],
    const s_t *const RSTR previous[2 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t *const RSTR output[2 * NS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_grad_0_ref_values[VS];
    s_t u0_grad_1_ref_values[VS];
    s_t u0_old_grad_0_ref_values[VS];
    s_t u0_old_grad_1_ref_values[VS];
    s_t u1_grad_0_ref_values[VS];
    s_t u1_grad_1_ref_values[VS];
    s_t u1_old_grad_0_ref_values[VS];
    s_t u1_old_grad_1_ref_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    {
      u0_grad_0_ref_values[0] = s_t(0);
      u0_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC][0];
        u0_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      u0_old_grad_0_ref_values[0] = s_t(0);
      u0_old_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC][0];
        u0_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      u1_grad_0_ref_values[0] = s_t(0);
      u1_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 1][0];
        u1_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      u1_old_grad_0_ref_values[0] = s_t(0);
      u1_old_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC + 1][0];
        u1_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t u0_grad_0_ref = u0_grad_0_ref_values[0];
      const s_t u0_grad_1_ref = u0_grad_1_ref_values[0];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[0];
      const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[0];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = u1_grad_0_ref_values[0];
      const s_t u1_grad_1_ref = u1_grad_1_ref_values[0];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[0];
      const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[0];
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
      grad_coeff0_0_values[0] = grad_coeff0_0;
      grad_coeff0_1_values[0] = grad_coeff0_1;
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
        const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj2) / det;
        const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj3) / det;
        output[test * NC][0] += q_weight[q] * det * (grad_coeff0_0_values[0] * test_grad0 + grad_coeff0_1_values[0] * test_grad1);
        output[test * NC + 1][0] += q_weight[q] * det * (grad_coeff1_0_values[0] * test_grad0 + grad_coeff1_1_values[0] * test_grad1);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_total_d2_simplex_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
    const s_t current[2 * NS][VS],
    const s_t previous[2 * NS][VS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t output[2 * NS][VS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_grad_0_ref_values[VS];
    s_t u0_grad_1_ref_values[VS];
    s_t u0_old_grad_0_ref_values[VS];
    s_t u0_old_grad_1_ref_values[VS];
    s_t u1_grad_0_ref_values[VS];
    s_t u1_grad_1_ref_values[VS];
    s_t u1_old_grad_0_ref_values[VS];
    s_t u1_old_grad_1_ref_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    {
      u0_grad_0_ref_values[0] = s_t(0);
      u0_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC][0];
        u0_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      u0_old_grad_0_ref_values[0] = s_t(0);
      u0_old_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC][0];
        u0_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      u1_grad_0_ref_values[0] = s_t(0);
      u1_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 1][0];
        u1_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      u1_old_grad_0_ref_values[0] = s_t(0);
      u1_old_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC + 1][0];
        u1_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t u0_grad_0_ref = u0_grad_0_ref_values[0];
      const s_t u0_grad_1_ref = u0_grad_1_ref_values[0];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[0];
      const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[0];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = u1_grad_0_ref_values[0];
      const s_t u1_grad_1_ref = u1_grad_1_ref_values[0];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[0];
      const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[0];
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
      grad_coeff0_0_values[0] = grad_coeff0_0;
      grad_coeff0_1_values[0] = grad_coeff0_1;
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
        const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj2) / det;
        const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj3) / det;
        output[test * NC][0] += q_weight[q] * det * (grad_coeff0_0_values[0] * test_grad0 + grad_coeff0_1_values[0] * test_grad1);
        output[test * NC + 1][0] += q_weight[q] * det * (grad_coeff1_0_values[0] * test_grad0 + grad_coeff1_1_values[0] * test_grad1);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_total_d2_simplex_tri3_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[2 * NS],
    const s_t *const RSTR previous[2 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t *const RSTR output[2 * NS]
) {
  for (int q = 0; q < NQ; ++q) {
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t u0_grad_0_ref = -(current[0][0]) + current[2][0];
      const s_t u0_grad_1_ref = -(current[0][0]) + current[4][0];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = -(previous[0][0]) + previous[2][0];
      const s_t u0_old_grad_1_ref = -(previous[0][0]) + previous[4][0];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = -(current[1][0]) + current[3][0];
      const s_t u1_grad_1_ref = -(current[1][0]) + current[5][0];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = -(previous[1][0]) + previous[3][0];
      const s_t u1_old_grad_1_ref = -(previous[1][0]) + previous[5][0];
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
      const s_t grad_coeff0_0_value = grad_coeff0_0;
      const s_t grad_coeff0_1_value = grad_coeff0_1;
      const s_t grad_coeff1_0_value = grad_coeff1_0;
      const s_t grad_coeff1_1_value = grad_coeff1_1;
      const s_t test0_grad0 = (-(adj0) - adj2) / det;
      const s_t test0_grad1 = (-(adj1) - adj3) / det;
      const s_t test1_grad0 = (adj0) / det;
      const s_t test1_grad1 = (adj1) / det;
      const s_t test2_grad0 = (adj2) / det;
      const s_t test2_grad1 = (adj3) / det;
      output[0][0] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1);
      output[1][0] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1);
      output[2][0] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1);
      output[3][0] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1);
      output[4][0] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1);
      output[5][0] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1);
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_total_d2_simplex_tri3_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR q_weight,
    const s_t current[2 * NS][VS],
    const s_t previous[2 * NS][VS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t output[2 * NS][VS]
) {
  for (int q = 0; q < NQ; ++q) {
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t u0_grad_0_ref = -(current[0][0]) + current[2][0];
      const s_t u0_grad_1_ref = -(current[0][0]) + current[4][0];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = -(previous[0][0]) + previous[2][0];
      const s_t u0_old_grad_1_ref = -(previous[0][0]) + previous[4][0];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = -(current[1][0]) + current[3][0];
      const s_t u1_grad_1_ref = -(current[1][0]) + current[5][0];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = -(previous[1][0]) + previous[3][0];
      const s_t u1_old_grad_1_ref = -(previous[1][0]) + previous[5][0];
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
      const s_t grad_coeff0_0_value = grad_coeff0_0;
      const s_t grad_coeff0_1_value = grad_coeff0_1;
      const s_t grad_coeff1_0_value = grad_coeff1_0;
      const s_t grad_coeff1_1_value = grad_coeff1_1;
      const s_t test0_grad0 = (-(adj0) - adj2) / det;
      const s_t test0_grad1 = (-(adj1) - adj3) / det;
      const s_t test1_grad0 = (adj0) / det;
      const s_t test1_grad1 = (adj1) / det;
      const s_t test2_grad0 = (adj2) / det;
      const s_t test2_grad1 = (adj3) / det;
      output[0][0] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1);
      output[1][0] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1);
      output[2][0] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1);
      output[3][0] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1);
      output[4][0] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1);
      output[5][0] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1);
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_total_d2_simplex_jacobian_action_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
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
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_grad_0_ref_values[VS];
    s_t u0_grad_1_ref_values[VS];
    s_t u0_old_grad_0_ref_values[VS];
    s_t u0_old_grad_1_ref_values[VS];
    s_t u0_direction_grad_0_ref_values[VS];
    s_t u0_direction_grad_1_ref_values[VS];
    s_t u1_grad_0_ref_values[VS];
    s_t u1_grad_1_ref_values[VS];
    s_t u1_old_grad_0_ref_values[VS];
    s_t u1_old_grad_1_ref_values[VS];
    s_t u1_direction_grad_0_ref_values[VS];
    s_t u1_direction_grad_1_ref_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    {
      u0_grad_0_ref_values[0] = s_t(0);
      u0_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC][0];
        u0_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      u0_old_grad_0_ref_values[0] = s_t(0);
      u0_old_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC][0];
        u0_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      u0_direction_grad_0_ref_values[0] = s_t(0);
      u0_direction_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = direction[trial * NC][0];
        u0_direction_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_direction_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      u1_grad_0_ref_values[0] = s_t(0);
      u1_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 1][0];
        u1_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      u1_old_grad_0_ref_values[0] = s_t(0);
      u1_old_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC + 1][0];
        u1_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      u1_direction_grad_0_ref_values[0] = s_t(0);
      u1_direction_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = direction[trial * NC + 1][0];
        u1_direction_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_direction_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t u0_grad_0_ref = u0_grad_0_ref_values[0];
      const s_t u0_grad_1_ref = u0_grad_1_ref_values[0];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[0];
      const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[0];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u0_direction_grad_0_ref = u0_direction_grad_0_ref_values[0];
      const s_t u0_direction_grad_1_ref = u0_direction_grad_1_ref_values[0];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = u1_grad_0_ref_values[0];
      const s_t u1_grad_1_ref = u1_grad_1_ref_values[0];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[0];
      const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[0];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
      const s_t u1_direction_grad_0_ref = u1_direction_grad_0_ref_values[0];
      const s_t u1_direction_grad_1_ref = u1_direction_grad_1_ref_values[0];
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
      grad_coeff0_0_values[0] = grad_coeff0_0;
      grad_coeff0_1_values[0] = grad_coeff0_1;
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
        const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj2) / det;
        const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj3) / det;
        output[test * NC][0] += q_weight[q] * det * (grad_coeff0_0_values[0] * test_grad0 + grad_coeff0_1_values[0] * test_grad1);
        output[test * NC + 1][0] += q_weight[q] * det * (grad_coeff1_0_values[0] * test_grad0 + grad_coeff1_1_values[0] * test_grad1);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_total_d2_simplex_jacobian_action_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
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
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_grad_0_ref_values[VS];
    s_t u0_grad_1_ref_values[VS];
    s_t u0_old_grad_0_ref_values[VS];
    s_t u0_old_grad_1_ref_values[VS];
    s_t u0_direction_grad_0_ref_values[VS];
    s_t u0_direction_grad_1_ref_values[VS];
    s_t u1_grad_0_ref_values[VS];
    s_t u1_grad_1_ref_values[VS];
    s_t u1_old_grad_0_ref_values[VS];
    s_t u1_old_grad_1_ref_values[VS];
    s_t u1_direction_grad_0_ref_values[VS];
    s_t u1_direction_grad_1_ref_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    {
      u0_grad_0_ref_values[0] = s_t(0);
      u0_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC][0];
        u0_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      u0_old_grad_0_ref_values[0] = s_t(0);
      u0_old_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC][0];
        u0_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      u0_direction_grad_0_ref_values[0] = s_t(0);
      u0_direction_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = direction[trial * NC][0];
        u0_direction_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_direction_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      u1_grad_0_ref_values[0] = s_t(0);
      u1_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 1][0];
        u1_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      u1_old_grad_0_ref_values[0] = s_t(0);
      u1_old_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC + 1][0];
        u1_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      u1_direction_grad_0_ref_values[0] = s_t(0);
      u1_direction_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = direction[trial * NC + 1][0];
        u1_direction_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_direction_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t u0_grad_0_ref = u0_grad_0_ref_values[0];
      const s_t u0_grad_1_ref = u0_grad_1_ref_values[0];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[0];
      const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[0];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u0_direction_grad_0_ref = u0_direction_grad_0_ref_values[0];
      const s_t u0_direction_grad_1_ref = u0_direction_grad_1_ref_values[0];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = u1_grad_0_ref_values[0];
      const s_t u1_grad_1_ref = u1_grad_1_ref_values[0];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[0];
      const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[0];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
      const s_t u1_direction_grad_0_ref = u1_direction_grad_0_ref_values[0];
      const s_t u1_direction_grad_1_ref = u1_direction_grad_1_ref_values[0];
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
      grad_coeff0_0_values[0] = grad_coeff0_0;
      grad_coeff0_1_values[0] = grad_coeff0_1;
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
        const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj2) / det;
        const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj3) / det;
        output[test * NC][0] += q_weight[q] * det * (grad_coeff0_0_values[0] * test_grad0 + grad_coeff0_1_values[0] * test_grad1);
        output[test * NC + 1][0] += q_weight[q] * det * (grad_coeff1_0_values[0] * test_grad0 + grad_coeff1_1_values[0] * test_grad1);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_total_d2_simplex_tri3_jacobian_action_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR q_weight,
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
  for (int q = 0; q < NQ; ++q) {
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t u0_grad_0_ref = -(current[0][0]) + current[2][0];
      const s_t u0_grad_1_ref = -(current[0][0]) + current[4][0];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = -(previous[0][0]) + previous[2][0];
      const s_t u0_old_grad_1_ref = -(previous[0][0]) + previous[4][0];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u0_direction_grad_0_ref = -(direction[0][0]) + direction[2][0];
      const s_t u0_direction_grad_1_ref = -(direction[0][0]) + direction[4][0];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = -(current[1][0]) + current[3][0];
      const s_t u1_grad_1_ref = -(current[1][0]) + current[5][0];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = -(previous[1][0]) + previous[3][0];
      const s_t u1_old_grad_1_ref = -(previous[1][0]) + previous[5][0];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
      const s_t u1_direction_grad_0_ref = -(direction[1][0]) + direction[3][0];
      const s_t u1_direction_grad_1_ref = -(direction[1][0]) + direction[5][0];
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
      const s_t grad_coeff0_0_value = grad_coeff0_0;
      const s_t grad_coeff0_1_value = grad_coeff0_1;
      const s_t grad_coeff1_0_value = grad_coeff1_0;
      const s_t grad_coeff1_1_value = grad_coeff1_1;
      const s_t test0_grad0 = (-(adj0) - adj2) / det;
      const s_t test0_grad1 = (-(adj1) - adj3) / det;
      const s_t test1_grad0 = (adj0) / det;
      const s_t test1_grad1 = (adj1) / det;
      const s_t test2_grad0 = (adj2) / det;
      const s_t test2_grad1 = (adj3) / det;
      output[0][0] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1);
      output[1][0] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1);
      output[2][0] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1);
      output[3][0] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1);
      output[4][0] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1);
      output[5][0] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1);
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_total_d2_simplex_tri3_jacobian_action_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR q_weight,
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
  for (int q = 0; q < NQ; ++q) {
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t u0_grad_0_ref = -(current[0][0]) + current[2][0];
      const s_t u0_grad_1_ref = -(current[0][0]) + current[4][0];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = -(previous[0][0]) + previous[2][0];
      const s_t u0_old_grad_1_ref = -(previous[0][0]) + previous[4][0];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u0_direction_grad_0_ref = -(direction[0][0]) + direction[2][0];
      const s_t u0_direction_grad_1_ref = -(direction[0][0]) + direction[4][0];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = -(current[1][0]) + current[3][0];
      const s_t u1_grad_1_ref = -(current[1][0]) + current[5][0];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = -(previous[1][0]) + previous[3][0];
      const s_t u1_old_grad_1_ref = -(previous[1][0]) + previous[5][0];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
      const s_t u1_direction_grad_0_ref = -(direction[1][0]) + direction[3][0];
      const s_t u1_direction_grad_1_ref = -(direction[1][0]) + direction[5][0];
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
      const s_t grad_coeff0_0_value = grad_coeff0_0;
      const s_t grad_coeff0_1_value = grad_coeff0_1;
      const s_t grad_coeff1_0_value = grad_coeff1_0;
      const s_t grad_coeff1_1_value = grad_coeff1_1;
      const s_t test0_grad0 = (-(adj0) - adj2) / det;
      const s_t test0_grad1 = (-(adj1) - adj3) / det;
      const s_t test1_grad0 = (adj0) / det;
      const s_t test1_grad1 = (adj1) / det;
      const s_t test2_grad0 = (adj2) / det;
      const s_t test2_grad1 = (adj3) / det;
      output[0][0] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1);
      output[1][0] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1);
      output[2][0] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1);
      output[3][0] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1);
      output[4][0] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1);
      output[5][0] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1);
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_total_d2_simplex_tri3_hessian_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t current[2 * NS][VS],
    const s_t previous[2 * NS][VS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t *const RSTR element_matrix
) {
  const int q = 0;
  {
    const ptrdiff_t goff = q * geometry_stride;
    const s_t det = determinant[goff];
    const s_t adj0 = adjugate[0][goff];
    const s_t adj1 = adjugate[1][goff];
    const s_t adj2 = adjugate[2][goff];
    const s_t adj3 = adjugate[3][goff];
    const s_t u0_grad_0_ref = -(current[0][0]) + current[2][0];
    const s_t u0_grad_1_ref = -(current[0][0]) + current[4][0];
    const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
    const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
    const s_t u0_old_grad_0_ref = -(previous[0][0]) + previous[2][0];
    const s_t u0_old_grad_1_ref = -(previous[0][0]) + previous[4][0];
    const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
    const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
    const s_t u1_grad_0_ref = -(current[1][0]) + current[3][0];
    const s_t u1_grad_1_ref = -(current[1][0]) + current[5][0];
    const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
    const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
    const s_t u1_old_grad_0_ref = -(previous[1][0]) + previous[3][0];
    const s_t u1_old_grad_1_ref = -(previous[1][0]) + previous[5][0];
    const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
    const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
    const s_t basis0_grad0 = (-(adj0) - adj2) / det;
    const s_t basis0_grad1 = (-(adj1) - adj3) / det;
    const s_t basis1_grad0 = (adj0) / det;
    const s_t basis1_grad1 = (adj1) / det;
    const s_t basis2_grad0 = (adj2) / det;
    const s_t basis2_grad1 = (adj3) / det;
    const s_t element_matrix_tmp0 = u1_grad_1 + s_t(1);
    const s_t element_matrix_tmp1 = pow_2(element_matrix_tmp0);
    const s_t element_matrix_tmp2 = u0_grad_1*u1_grad_0;
    const s_t element_matrix_tmp3 = element_matrix_tmp0 - element_matrix_tmp2 + u0_grad_0*u1_grad_1 + u0_grad_0;
    const s_t element_matrix_tmp4 = pow_m1(element_matrix_tmp3);
    const s_t element_matrix_tmp5 = eta_s*u0_old_grad_1;
    const s_t element_matrix_tmp6 = element_matrix_tmp0*u_dt_shift;
    const s_t element_matrix_tmp7 = u1_grad_1*u_dt_shift + u1_old_grad_1;
    const s_t element_matrix_tmp8 = eta_b*(-element_matrix_tmp6 - element_matrix_tmp7);
    const s_t element_matrix_tmp9 = -element_matrix_tmp6 + element_matrix_tmp7;
    const s_t element_matrix_tmp10 = -element_matrix_tmp9;
    const s_t element_matrix_tmp11 = -element_matrix_tmp10*eta_s + element_matrix_tmp8;
    const s_t element_matrix_tmp12 = -element_matrix_tmp0;
    const s_t element_matrix_tmp13 = pow_m2(element_matrix_tmp3);
    const s_t element_matrix_tmp14 = u0_grad_0*u_dt_shift + u0_old_grad_0;
    const s_t element_matrix_tmp15 = u0_grad_0 + s_t(1);
    const s_t element_matrix_tmp16 = u0_grad_1*u_dt_shift;
    const s_t element_matrix_tmp17 = element_matrix_tmp16 + u0_old_grad_1;
    const s_t element_matrix_tmp18 = u1_grad_0*u_dt_shift;
    const s_t element_matrix_tmp19 = element_matrix_tmp18 + u1_old_grad_0;
    const s_t element_matrix_tmp20 = eta_s*(-element_matrix_tmp0*element_matrix_tmp19 + element_matrix_tmp14*u0_grad_1 - element_matrix_tmp15*element_matrix_tmp17 + element_matrix_tmp7*u1_grad_0);
    const s_t element_matrix_tmp21 = element_matrix_tmp17*u1_grad_0;
    const s_t element_matrix_tmp22 = element_matrix_tmp0*element_matrix_tmp14;
    const s_t element_matrix_tmp23 = -element_matrix_tmp15*element_matrix_tmp7 + element_matrix_tmp19*u0_grad_1;
    const s_t element_matrix_tmp24 = eta_b*(element_matrix_tmp21 - element_matrix_tmp22 + element_matrix_tmp23);
    const s_t element_matrix_tmp25 = eta_s*(-element_matrix_tmp21 + element_matrix_tmp22 + element_matrix_tmp23);
    const s_t element_matrix_tmp26 = element_matrix_tmp24 - element_matrix_tmp25;
    const s_t element_matrix_tmp27 = -element_matrix_tmp0*element_matrix_tmp26 + element_matrix_tmp20*u0_grad_1;
    const s_t element_matrix_tmp28 = element_matrix_tmp13*element_matrix_tmp27;
    const s_t element_matrix_tmp29 = element_matrix_tmp1*lmbda + element_matrix_tmp12*element_matrix_tmp28 + element_matrix_tmp4*(-element_matrix_tmp0*element_matrix_tmp11 - element_matrix_tmp5*u0_grad_1) + mu*(s_t(2)*element_matrix_tmp1 + s_t(4));
    const s_t element_matrix_tmp30 = element_matrix_tmp0*u1_grad_0;
    const s_t element_matrix_tmp31 = element_matrix_tmp30*lmbda;
    const s_t element_matrix_tmp32 = s_t(2)*element_matrix_tmp30;
    const s_t element_matrix_tmp33 = element_matrix_tmp15*u_dt_shift;
    const s_t element_matrix_tmp34 = element_matrix_tmp14 - element_matrix_tmp33;
    const s_t element_matrix_tmp35 = element_matrix_tmp34*eta_s;
    const s_t element_matrix_tmp36 = eta_b*(s_t(2)*element_matrix_tmp18 + u1_old_grad_0);
    const s_t element_matrix_tmp37 = element_matrix_tmp36 - eta_s*u1_old_grad_0;
    const s_t element_matrix_tmp38 = element_matrix_tmp13*element_matrix_tmp27*u1_grad_0 - element_matrix_tmp31 - element_matrix_tmp32*mu + element_matrix_tmp4*(-element_matrix_tmp0*element_matrix_tmp37 + element_matrix_tmp20 + element_matrix_tmp35*u0_grad_1);
    const s_t element_matrix_tmp39 = basis0_grad0*element_matrix_tmp29 + basis0_grad1*element_matrix_tmp38;
    const s_t element_matrix_tmp40 = pow_2(u1_grad_0);
    const s_t element_matrix_tmp41 = -element_matrix_tmp15*element_matrix_tmp20 + element_matrix_tmp26*u1_grad_0;
    const s_t element_matrix_tmp42 = element_matrix_tmp13*element_matrix_tmp41;
    const s_t element_matrix_tmp43 = element_matrix_tmp4*(-element_matrix_tmp15*element_matrix_tmp35 + element_matrix_tmp37*u1_grad_0) + element_matrix_tmp40*lmbda + element_matrix_tmp42*u1_grad_0 + mu*(s_t(2)*element_matrix_tmp40 + s_t(4));
    const s_t element_matrix_tmp44 = element_matrix_tmp15*u0_grad_1;
    const s_t element_matrix_tmp45 = s_t(2)*u0_grad_0 + s_t(2);
    const s_t element_matrix_tmp46 = -element_matrix_tmp20;
    const s_t element_matrix_tmp47 = element_matrix_tmp12*element_matrix_tmp42 - element_matrix_tmp31 + element_matrix_tmp4*(element_matrix_tmp11*u1_grad_0 + element_matrix_tmp15*element_matrix_tmp5 + element_matrix_tmp46) + mu*(-element_matrix_tmp32 - s_t(4)*element_matrix_tmp44 + s_t(2)*element_matrix_tmp45*u0_grad_1);
    const s_t element_matrix_tmp48 = basis0_grad0*element_matrix_tmp47 + basis0_grad1*element_matrix_tmp43;
    const s_t element_matrix_tmp49 = ((s_t(1) / s_t(2)))*det;
    const s_t element_matrix_tmp50 = element_matrix_tmp10*eta_s + element_matrix_tmp8;
    const s_t element_matrix_tmp51 = element_matrix_tmp24 + element_matrix_tmp25;
    const s_t element_matrix_tmp52 = -element_matrix_tmp0*element_matrix_tmp20 + element_matrix_tmp51*u0_grad_1;
    const s_t element_matrix_tmp53 = element_matrix_tmp0*u0_grad_1;
    const s_t element_matrix_tmp54 = s_t(2)*mu;
    const s_t element_matrix_tmp55 = element_matrix_tmp53*element_matrix_tmp54 + element_matrix_tmp53*lmbda;
    const s_t element_matrix_tmp56 = element_matrix_tmp12*element_matrix_tmp13*element_matrix_tmp52 + element_matrix_tmp4*(element_matrix_tmp0*element_matrix_tmp5 + element_matrix_tmp50*u0_grad_1) - element_matrix_tmp55;
    const s_t element_matrix_tmp57 = eta_s*u1_old_grad_0;
    const s_t element_matrix_tmp58 = element_matrix_tmp36 + element_matrix_tmp57;
    const s_t element_matrix_tmp59 = element_matrix_tmp13*element_matrix_tmp52;
    const s_t element_matrix_tmp60 = element_matrix_tmp0*element_matrix_tmp15;
    const s_t element_matrix_tmp61 = lmbda*(element_matrix_tmp0*element_matrix_tmp15 - element_matrix_tmp2 + s_t(-1));
    const s_t element_matrix_tmp62 = element_matrix_tmp2*lmbda - element_matrix_tmp61 + mu*(s_t(4)*element_matrix_tmp2 - s_t(2)*element_matrix_tmp60 + s_t(6));
    const s_t element_matrix_tmp63 = element_matrix_tmp4*(-element_matrix_tmp0*element_matrix_tmp35 + element_matrix_tmp51 + element_matrix_tmp58*u0_grad_1) + element_matrix_tmp59*u1_grad_0 + element_matrix_tmp62;
    const s_t element_matrix_tmp64 = basis0_grad0*element_matrix_tmp56 + basis0_grad1*element_matrix_tmp63;
    const s_t element_matrix_tmp65 = -element_matrix_tmp15*element_matrix_tmp51 + element_matrix_tmp20*u1_grad_0;
    const s_t element_matrix_tmp66 = element_matrix_tmp15*u1_grad_0;
    const s_t element_matrix_tmp67 = element_matrix_tmp54*element_matrix_tmp66 + element_matrix_tmp66*lmbda;
    const s_t element_matrix_tmp68 = element_matrix_tmp13*element_matrix_tmp65*u1_grad_0 + element_matrix_tmp4*(-element_matrix_tmp15*element_matrix_tmp58 + element_matrix_tmp35*u1_grad_0) - element_matrix_tmp67;
    const s_t element_matrix_tmp69 = s_t(2)*element_matrix_tmp2 + s_t(6);
    const s_t element_matrix_tmp70 = element_matrix_tmp13*element_matrix_tmp65;
    const s_t element_matrix_tmp71 = element_matrix_tmp60*lmbda + element_matrix_tmp61;
    const s_t element_matrix_tmp72 = element_matrix_tmp12*element_matrix_tmp70 + element_matrix_tmp4*(-element_matrix_tmp15*element_matrix_tmp50 - element_matrix_tmp5*u1_grad_0 - element_matrix_tmp51) + element_matrix_tmp71 + mu*(s_t(2)*element_matrix_tmp0*element_matrix_tmp45 - element_matrix_tmp69);
    const s_t element_matrix_tmp73 = basis0_grad0*element_matrix_tmp72 + basis0_grad1*element_matrix_tmp68;
    const s_t element_matrix_tmp74 = basis1_grad0*element_matrix_tmp29 + basis1_grad1*element_matrix_tmp38;
    const s_t element_matrix_tmp75 = basis1_grad0*element_matrix_tmp47 + basis1_grad1*element_matrix_tmp43;
    const s_t element_matrix_tmp76 = basis1_grad0*element_matrix_tmp56 + basis1_grad1*element_matrix_tmp63;
    const s_t element_matrix_tmp77 = basis1_grad0*element_matrix_tmp72 + basis1_grad1*element_matrix_tmp68;
    const s_t element_matrix_tmp78 = basis2_grad0*element_matrix_tmp29 + basis2_grad1*element_matrix_tmp38;
    const s_t element_matrix_tmp79 = basis2_grad0*element_matrix_tmp47 + basis2_grad1*element_matrix_tmp43;
    const s_t element_matrix_tmp80 = basis2_grad0*element_matrix_tmp56 + basis2_grad1*element_matrix_tmp63;
    const s_t element_matrix_tmp81 = basis2_grad0*element_matrix_tmp72 + basis2_grad1*element_matrix_tmp68;
    const s_t element_matrix_tmp82 = eta_b*(-element_matrix_tmp14 - element_matrix_tmp33);
    const s_t element_matrix_tmp83 = -element_matrix_tmp34*eta_s + element_matrix_tmp82;
    const s_t element_matrix_tmp84 = -element_matrix_tmp15;
    const s_t element_matrix_tmp85 = element_matrix_tmp13*element_matrix_tmp41*element_matrix_tmp84 + element_matrix_tmp4*(element_matrix_tmp15*element_matrix_tmp57 + element_matrix_tmp83*u1_grad_0) - element_matrix_tmp67;
    const s_t element_matrix_tmp86 = eta_b*(s_t(2)*element_matrix_tmp16 + u0_old_grad_1);
    const s_t element_matrix_tmp87 = element_matrix_tmp5 + element_matrix_tmp86;
    const s_t element_matrix_tmp88 = element_matrix_tmp9*eta_s;
    const s_t element_matrix_tmp89 = element_matrix_tmp4*(-element_matrix_tmp15*element_matrix_tmp88 + element_matrix_tmp26 + element_matrix_tmp87*u1_grad_0) + element_matrix_tmp42*u0_grad_1 + element_matrix_tmp62;
    const s_t element_matrix_tmp90 = basis0_grad0*element_matrix_tmp89 + basis0_grad1*element_matrix_tmp85;
    const s_t element_matrix_tmp91 = element_matrix_tmp13*element_matrix_tmp27*u0_grad_1 + element_matrix_tmp4*(-element_matrix_tmp0*element_matrix_tmp87 + element_matrix_tmp88*u0_grad_1) - element_matrix_tmp55;
    const s_t element_matrix_tmp92 = s_t(2)*u1_grad_1 + s_t(2);
    const s_t element_matrix_tmp93 = element_matrix_tmp28*element_matrix_tmp84 + element_matrix_tmp4*(-element_matrix_tmp0*element_matrix_tmp83 - element_matrix_tmp26 - element_matrix_tmp57*u0_grad_1) + element_matrix_tmp71 + mu*(s_t(2)*element_matrix_tmp15*element_matrix_tmp92 - element_matrix_tmp69);
    const s_t element_matrix_tmp94 = basis0_grad0*element_matrix_tmp91 + basis0_grad1*element_matrix_tmp93;
    const s_t element_matrix_tmp95 = pow_2(element_matrix_tmp15);
    const s_t element_matrix_tmp96 = element_matrix_tmp35 + element_matrix_tmp82;
    const s_t element_matrix_tmp97 = element_matrix_tmp4*(-element_matrix_tmp15*element_matrix_tmp96 - element_matrix_tmp57*u1_grad_0) + element_matrix_tmp70*element_matrix_tmp84 + element_matrix_tmp95*lmbda + mu*(s_t(2)*element_matrix_tmp95 + s_t(4));
    const s_t element_matrix_tmp98 = element_matrix_tmp44*lmbda;
    const s_t element_matrix_tmp99 = s_t(2)*element_matrix_tmp44;
    const s_t element_matrix_tmp100 = -element_matrix_tmp5 + element_matrix_tmp86;
    const s_t element_matrix_tmp101 = element_matrix_tmp13*element_matrix_tmp65*u0_grad_1 + element_matrix_tmp4*(-element_matrix_tmp100*element_matrix_tmp15 + element_matrix_tmp20 + element_matrix_tmp88*u1_grad_0) - element_matrix_tmp98 - element_matrix_tmp99*mu;
    const s_t element_matrix_tmp102 = basis0_grad0*element_matrix_tmp101 + basis0_grad1*element_matrix_tmp97;
    const s_t element_matrix_tmp103 = pow_2(u0_grad_1);
    const s_t element_matrix_tmp104 = element_matrix_tmp103*lmbda + element_matrix_tmp4*(-element_matrix_tmp0*element_matrix_tmp88 + element_matrix_tmp100*u0_grad_1) + element_matrix_tmp59*u0_grad_1 + mu*(s_t(2)*element_matrix_tmp103 + s_t(4));
    const s_t element_matrix_tmp105 = element_matrix_tmp4*(element_matrix_tmp0*element_matrix_tmp57 + element_matrix_tmp46 + element_matrix_tmp96*u0_grad_1) + element_matrix_tmp59*element_matrix_tmp84 - element_matrix_tmp98 + mu*(-s_t(4)*element_matrix_tmp30 + s_t(2)*element_matrix_tmp92*u1_grad_0 - element_matrix_tmp99);
    const s_t element_matrix_tmp106 = basis0_grad0*element_matrix_tmp104 + basis0_grad1*element_matrix_tmp105;
    const s_t element_matrix_tmp107 = basis1_grad0*element_matrix_tmp89 + basis1_grad1*element_matrix_tmp85;
    const s_t element_matrix_tmp108 = basis1_grad0*element_matrix_tmp91 + basis1_grad1*element_matrix_tmp93;
    const s_t element_matrix_tmp109 = basis1_grad0*element_matrix_tmp101 + basis1_grad1*element_matrix_tmp97;
    const s_t element_matrix_tmp110 = basis1_grad0*element_matrix_tmp104 + basis1_grad1*element_matrix_tmp105;
    const s_t element_matrix_tmp111 = basis2_grad0*element_matrix_tmp89 + basis2_grad1*element_matrix_tmp85;
    const s_t element_matrix_tmp112 = basis2_grad0*element_matrix_tmp91 + basis2_grad1*element_matrix_tmp93;
    const s_t element_matrix_tmp113 = basis2_grad0*element_matrix_tmp101 + basis2_grad1*element_matrix_tmp97;
    const s_t element_matrix_tmp114 = basis2_grad0*element_matrix_tmp104 + basis2_grad1*element_matrix_tmp105;
    element_matrix[0] = element_matrix_tmp49*(basis0_grad0*element_matrix_tmp39 + basis0_grad1*element_matrix_tmp48);
    element_matrix[6] = element_matrix_tmp49*(basis1_grad0*element_matrix_tmp39 + basis1_grad1*element_matrix_tmp48);
    element_matrix[12] = element_matrix_tmp49*(basis2_grad0*element_matrix_tmp39 + basis2_grad1*element_matrix_tmp48);
    element_matrix[18] = element_matrix_tmp49*(basis0_grad0*element_matrix_tmp64 + basis0_grad1*element_matrix_tmp73);
    element_matrix[24] = element_matrix_tmp49*(basis1_grad0*element_matrix_tmp64 + basis1_grad1*element_matrix_tmp73);
    element_matrix[30] = element_matrix_tmp49*(basis2_grad0*element_matrix_tmp64 + basis2_grad1*element_matrix_tmp73);
    element_matrix[1] = element_matrix_tmp49*(basis0_grad0*element_matrix_tmp74 + basis0_grad1*element_matrix_tmp75);
    element_matrix[7] = element_matrix_tmp49*(basis1_grad0*element_matrix_tmp74 + basis1_grad1*element_matrix_tmp75);
    element_matrix[13] = element_matrix_tmp49*(basis2_grad0*element_matrix_tmp74 + basis2_grad1*element_matrix_tmp75);
    element_matrix[19] = element_matrix_tmp49*(basis0_grad0*element_matrix_tmp76 + basis0_grad1*element_matrix_tmp77);
    element_matrix[25] = element_matrix_tmp49*(basis1_grad0*element_matrix_tmp76 + basis1_grad1*element_matrix_tmp77);
    element_matrix[31] = element_matrix_tmp49*(basis2_grad0*element_matrix_tmp76 + basis2_grad1*element_matrix_tmp77);
    element_matrix[2] = element_matrix_tmp49*(basis0_grad0*element_matrix_tmp78 + basis0_grad1*element_matrix_tmp79);
    element_matrix[8] = element_matrix_tmp49*(basis1_grad0*element_matrix_tmp78 + basis1_grad1*element_matrix_tmp79);
    element_matrix[14] = element_matrix_tmp49*(basis2_grad0*element_matrix_tmp78 + basis2_grad1*element_matrix_tmp79);
    element_matrix[20] = element_matrix_tmp49*(basis0_grad0*element_matrix_tmp80 + basis0_grad1*element_matrix_tmp81);
    element_matrix[26] = element_matrix_tmp49*(basis1_grad0*element_matrix_tmp80 + basis1_grad1*element_matrix_tmp81);
    element_matrix[32] = element_matrix_tmp49*(basis2_grad0*element_matrix_tmp80 + basis2_grad1*element_matrix_tmp81);
    element_matrix[3] = element_matrix_tmp49*(basis0_grad0*element_matrix_tmp94 + basis0_grad1*element_matrix_tmp90);
    element_matrix[9] = element_matrix_tmp49*(basis1_grad0*element_matrix_tmp94 + basis1_grad1*element_matrix_tmp90);
    element_matrix[15] = element_matrix_tmp49*(basis2_grad0*element_matrix_tmp94 + basis2_grad1*element_matrix_tmp90);
    element_matrix[21] = element_matrix_tmp49*(basis0_grad0*element_matrix_tmp106 + basis0_grad1*element_matrix_tmp102);
    element_matrix[27] = element_matrix_tmp49*(basis1_grad0*element_matrix_tmp106 + basis1_grad1*element_matrix_tmp102);
    element_matrix[33] = element_matrix_tmp49*(basis2_grad0*element_matrix_tmp106 + basis2_grad1*element_matrix_tmp102);
    element_matrix[4] = element_matrix_tmp49*(basis0_grad0*element_matrix_tmp108 + basis0_grad1*element_matrix_tmp107);
    element_matrix[10] = element_matrix_tmp49*(basis1_grad0*element_matrix_tmp108 + basis1_grad1*element_matrix_tmp107);
    element_matrix[16] = element_matrix_tmp49*(basis2_grad0*element_matrix_tmp108 + basis2_grad1*element_matrix_tmp107);
    element_matrix[22] = element_matrix_tmp49*(basis0_grad0*element_matrix_tmp110 + basis0_grad1*element_matrix_tmp109);
    element_matrix[28] = element_matrix_tmp49*(basis1_grad0*element_matrix_tmp110 + basis1_grad1*element_matrix_tmp109);
    element_matrix[34] = element_matrix_tmp49*(basis2_grad0*element_matrix_tmp110 + basis2_grad1*element_matrix_tmp109);
    element_matrix[5] = element_matrix_tmp49*(basis0_grad0*element_matrix_tmp112 + basis0_grad1*element_matrix_tmp111);
    element_matrix[11] = element_matrix_tmp49*(basis1_grad0*element_matrix_tmp112 + basis1_grad1*element_matrix_tmp111);
    element_matrix[17] = element_matrix_tmp49*(basis2_grad0*element_matrix_tmp112 + basis2_grad1*element_matrix_tmp111);
    element_matrix[23] = element_matrix_tmp49*(basis0_grad0*element_matrix_tmp114 + basis0_grad1*element_matrix_tmp113);
    element_matrix[29] = element_matrix_tmp49*(basis1_grad0*element_matrix_tmp114 + basis1_grad1*element_matrix_tmp113);
    element_matrix[35] = element_matrix_tmp49*(basis2_grad0*element_matrix_tmp114 + basis2_grad1*element_matrix_tmp113);
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_total_d2_simplex_hessian_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
    const s_t current[2 * NS][VS],
    const s_t previous[2 * NS][VS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t *const RSTR element_matrix
) {
  static constexpr int NC = 2;
  for (int entry = 0; entry < 2 * NS * 2 * NS; ++entry) {
    element_matrix[entry] = s_t(0);
  }
  s_t u0_grad_0_ref_values[VS];
  s_t u0_grad_1_ref_values[VS];
  s_t u0_old_grad_0_ref_values[VS];
  s_t u0_old_grad_1_ref_values[VS];
  s_t u1_grad_0_ref_values[VS];
  s_t u1_grad_1_ref_values[VS];
  s_t u1_old_grad_0_ref_values[VS];
  s_t u1_old_grad_1_ref_values[VS];
  s_t grad_coeff0_0_values[VS];
  s_t grad_coeff0_1_values[VS];
  s_t grad_coeff1_0_values[VS];
  s_t grad_coeff1_1_values[VS];
  s_t tangent[16][VS];
  for (int q = 0; q < NQ; ++q) {
    {
      u0_grad_0_ref_values[0] = s_t(0);
      u0_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC][0];
        u0_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      u0_old_grad_0_ref_values[0] = s_t(0);
      u0_old_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC][0];
        u0_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      u1_grad_0_ref_values[0] = s_t(0);
      u1_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 1][0];
        u1_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      u1_old_grad_0_ref_values[0] = s_t(0);
      u1_old_grad_1_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC + 1][0];
        u1_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t u0_grad_0_ref = u0_grad_0_ref_values[0];
      const s_t u0_grad_1_ref = u0_grad_1_ref_values[0];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[0];
      const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[0];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = u1_grad_0_ref_values[0];
      const s_t u1_grad_1_ref = u1_grad_1_ref_values[0];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[0];
      const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[0];
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
      const s_t tangent_grad_d0_0_grad0_0 = lmbda*tangent_tmp1 + mu*(s_t(2)*tangent_tmp1 + s_t(4)) + tangent_tmp25*tangent_tmp28 + tangent_tmp4*(-tangent_tmp0*tangent_tmp11 - tangent_tmp5*u0_grad_1);
      const s_t tangent_grad_d0_0_grad0_1 = mu*(-s_t(4)*tangent_tmp31 - tangent_tmp32 + s_t(2)*tangent_tmp33*u0_grad_1) + tangent_tmp28*tangent_tmp35 - tangent_tmp30 + tangent_tmp4*(tangent_tmp11*u1_grad_0 + tangent_tmp13*tangent_tmp5 + tangent_tmp34);
      const s_t tangent_grad_d0_0_grad1_0 = tangent_tmp26*tangent_tmp27*tangent_tmp38 + tangent_tmp4*(tangent_tmp0*tangent_tmp5 + tangent_tmp36*u0_grad_1) - tangent_tmp41;
      const s_t tangent_grad_d0_0_grad1_1 = mu*(s_t(2)*tangent_tmp0*tangent_tmp33 - tangent_tmp42) + tangent_tmp28*tangent_tmp43 + tangent_tmp4*(-tangent_tmp13*tangent_tmp36 - tangent_tmp37 - tangent_tmp5*u1_grad_0) + tangent_tmp46;
      const s_t tangent_grad_d0_1_grad0_0 = -mu*tangent_tmp32 + tangent_tmp25*tangent_tmp26*u1_grad_0 - tangent_tmp30 + tangent_tmp4*(-tangent_tmp0*tangent_tmp51 + tangent_tmp18 + tangent_tmp49*u0_grad_1);
      const s_t tangent_grad_d0_1_grad0_1 = lmbda*tangent_tmp52 + mu*(s_t(2)*tangent_tmp52 + s_t(4)) + tangent_tmp35*tangent_tmp53 + tangent_tmp4*(-tangent_tmp13*tangent_tmp49 + tangent_tmp51*u1_grad_0);
      const s_t tangent_grad_d0_1_grad1_0 = tangent_tmp38*tangent_tmp53 + tangent_tmp4*(-tangent_tmp0*tangent_tmp49 + tangent_tmp37 + tangent_tmp55*u0_grad_1) + tangent_tmp56;
      const s_t tangent_grad_d0_1_grad1_1 = tangent_tmp26*tangent_tmp43*u1_grad_0 + tangent_tmp4*(-tangent_tmp13*tangent_tmp55 + tangent_tmp49*u1_grad_0) - tangent_tmp58;
      const s_t tangent_grad_d1_0_grad0_0 = tangent_tmp25*tangent_tmp26*u0_grad_1 + tangent_tmp4*(-tangent_tmp0*tangent_tmp61 + tangent_tmp59*u0_grad_1) - tangent_tmp41;
      const s_t tangent_grad_d1_0_grad0_1 = tangent_tmp35*tangent_tmp62 + tangent_tmp4*(-tangent_tmp13*tangent_tmp59 + tangent_tmp24 + tangent_tmp61*u1_grad_0) + tangent_tmp56;
      const s_t tangent_grad_d1_0_grad1_0 = lmbda*tangent_tmp63 + mu*(s_t(2)*tangent_tmp63 + s_t(4)) + tangent_tmp38*tangent_tmp62 + tangent_tmp4*(-tangent_tmp0*tangent_tmp59 + tangent_tmp64*u0_grad_1);
      const s_t tangent_grad_d1_0_grad1_1 = -mu*tangent_tmp66 + tangent_tmp26*tangent_tmp43*u0_grad_1 + tangent_tmp4*(-tangent_tmp13*tangent_tmp64 + tangent_tmp18 + tangent_tmp59*u1_grad_0) - tangent_tmp65;
      const s_t tangent_grad_d1_1_grad0_0 = mu*(s_t(2)*tangent_tmp13*tangent_tmp67 - tangent_tmp42) + tangent_tmp25*tangent_tmp71 + tangent_tmp4*(-tangent_tmp0*tangent_tmp69 - tangent_tmp24 - tangent_tmp54*u0_grad_1) + tangent_tmp46;
      const s_t tangent_grad_d1_1_grad0_1 = tangent_tmp26*tangent_tmp35*tangent_tmp70 + tangent_tmp4*(tangent_tmp13*tangent_tmp54 + tangent_tmp69*u1_grad_0) - tangent_tmp58;
      const s_t tangent_grad_d1_1_grad1_0 = mu*(-s_t(4)*tangent_tmp29 - tangent_tmp66 + s_t(2)*tangent_tmp67*u1_grad_0) + tangent_tmp38*tangent_tmp71 + tangent_tmp4*(tangent_tmp0*tangent_tmp54 + tangent_tmp34 + tangent_tmp72*u0_grad_1) - tangent_tmp65;
      const s_t tangent_grad_d1_1_grad1_1 = lmbda*tangent_tmp73 + mu*(s_t(2)*tangent_tmp73 + s_t(4)) + tangent_tmp4*(-tangent_tmp13*tangent_tmp72 - tangent_tmp54*u1_grad_0) + tangent_tmp43*tangent_tmp71;
      tangent[0][0] = tangent_grad_d0_0_grad0_0;
      tangent[1][0] = tangent_grad_d0_0_grad0_1;
      tangent[2][0] = tangent_grad_d0_0_grad1_0;
      tangent[3][0] = tangent_grad_d0_0_grad1_1;
      tangent[4][0] = tangent_grad_d0_1_grad0_0;
      tangent[5][0] = tangent_grad_d0_1_grad0_1;
      tangent[6][0] = tangent_grad_d0_1_grad1_0;
      tangent[7][0] = tangent_grad_d0_1_grad1_1;
      tangent[8][0] = tangent_grad_d1_0_grad0_0;
      tangent[9][0] = tangent_grad_d1_0_grad0_1;
      tangent[10][0] = tangent_grad_d1_0_grad1_0;
      tangent[11][0] = tangent_grad_d1_0_grad1_1;
      tangent[12][0] = tangent_grad_d1_1_grad0_0;
      tangent[13][0] = tangent_grad_d1_1_grad0_1;
      tangent[14][0] = tangent_grad_d1_1_grad1_0;
      tangent[15][0] = tangent_grad_d1_1_grad1_1;
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        const s_t adj0 = adjugate[0][goff];
        const s_t adj1 = adjugate[1][goff];
        const s_t adj2 = adjugate[2][goff];
        const s_t adj3 = adjugate[3][goff];
        const s_t trial_grad0 = (grad_ref_x[q * NS + trial] * adj0 + grad_ref_y[q * NS + trial] * adj2) / det;
        const s_t trial_grad1 = (grad_ref_x[q * NS + trial] * adj1 + grad_ref_y[q * NS + trial] * adj3) / det;
        const s_t grad_coeff0_0 = trial_grad0 * tangent[0][0] + trial_grad1 * tangent[4][0];
        const s_t grad_coeff0_1 = trial_grad0 * tangent[1][0] + trial_grad1 * tangent[5][0];
        const s_t grad_coeff1_0 = trial_grad0 * tangent[2][0] + trial_grad1 * tangent[6][0];
        const s_t grad_coeff1_1 = trial_grad0 * tangent[3][0] + trial_grad1 * tangent[7][0];
        grad_coeff0_0_values[0] = grad_coeff0_0;
        grad_coeff0_1_values[0] = grad_coeff0_1;
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
          const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj2) / det;
          const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj3) / det;
          element_matrix[(0 * NS + test) * 2 * NS + 0 * NS + trial] += q_weight[q] * det * (grad_coeff0_0_values[0] * test_grad0 + grad_coeff0_1_values[0] * test_grad1);
          element_matrix[(1 * NS + test) * 2 * NS + 0 * NS + trial] += q_weight[q] * det * (grad_coeff1_0_values[0] * test_grad0 + grad_coeff1_1_values[0] * test_grad1);
        }
      }
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        const s_t adj0 = adjugate[0][goff];
        const s_t adj1 = adjugate[1][goff];
        const s_t adj2 = adjugate[2][goff];
        const s_t adj3 = adjugate[3][goff];
        const s_t trial_grad0 = (grad_ref_x[q * NS + trial] * adj0 + grad_ref_y[q * NS + trial] * adj2) / det;
        const s_t trial_grad1 = (grad_ref_x[q * NS + trial] * adj1 + grad_ref_y[q * NS + trial] * adj3) / det;
        const s_t grad_coeff0_0 = trial_grad0 * tangent[8][0] + trial_grad1 * tangent[12][0];
        const s_t grad_coeff0_1 = trial_grad0 * tangent[9][0] + trial_grad1 * tangent[13][0];
        const s_t grad_coeff1_0 = trial_grad0 * tangent[10][0] + trial_grad1 * tangent[14][0];
        const s_t grad_coeff1_1 = trial_grad0 * tangent[11][0] + trial_grad1 * tangent[15][0];
        grad_coeff0_0_values[0] = grad_coeff0_0;
        grad_coeff0_1_values[0] = grad_coeff0_1;
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
          const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj2) / det;
          const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj3) / det;
          element_matrix[(0 * NS + test) * 2 * NS + 1 * NS + trial] += q_weight[q] * det * (grad_coeff0_0_values[0] * test_grad0 + grad_coeff0_1_values[0] * test_grad1);
          element_matrix[(1 * NS + test) * 2 * NS + 1 * NS + trial] += q_weight[q] * det * (grad_coeff1_0_values[0] * test_grad0 + grad_coeff1_1_values[0] * test_grad1);
        }
      }
    }
  }
}

} // namespace codegen
} // namespace sfem

#endif
