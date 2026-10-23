#ifndef LINEAR_KELVIN_VOIGT_TOTAL_D2_SIMPLEX_LOCAL_HPP
#define LINEAR_KELVIN_VOIGT_TOTAL_D2_SIMPLEX_LOCAL_HPP

#include <math.h>
#include <stddef.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_GENERATED_SCALAR_T
#endif
#endif
#include "kernel_math.cuh"
#include "tensor_product_kernels.cuh"

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
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d2_simplex_residual_block(
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
    s_t u0_grad_0_ref_values;
    s_t u0_grad_1_ref_values;
    s_t u0_old_grad_0_ref_values;
    s_t u0_old_grad_1_ref_values;
    s_t u1_grad_0_ref_values;
    s_t u1_grad_1_ref_values;
    s_t u1_old_grad_0_ref_values;
    s_t u1_old_grad_1_ref_values;
    s_t grad_coeff0_0_values;
    s_t grad_coeff0_1_values;
    s_t grad_coeff1_0_values;
    s_t grad_coeff1_1_values;
    {
      u0_grad_0_ref_values = s_t(0);
      u0_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = current[trial * NC];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        u0_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        u0_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
      }
    }
    {
      u0_old_grad_0_ref_values = s_t(0);
      u0_old_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = previous[trial * NC];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        u0_old_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        u0_old_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
      }
    }
    {
      u1_grad_0_ref_values = s_t(0);
      u1_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = current[trial * NC + 1];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        u1_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        u1_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
      }
    }
    {
      u1_old_grad_0_ref_values = s_t(0);
      u1_old_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = previous[trial * NC + 1];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        u1_old_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        u1_old_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t u0_grad_0_ref = u0_grad_0_ref_values;
      const s_t u0_grad_1_ref = u0_grad_1_ref_values;
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values;
      const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values;
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = u1_grad_0_ref_values;
      const s_t u1_grad_1_ref = u1_grad_1_ref_values;
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values;
      const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values;
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
      const s_t residual_tmp0 = u1_grad_1*u_dt_shift;
      const s_t residual_tmp1 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp2 = eta_s*(-residual_tmp0 + residual_tmp1 - u1_old_grad_1);
      const s_t residual_tmp3 = s_t(2)*u0_grad_0;
      const s_t residual_tmp4 = s_t(2)*u1_grad_1;
      const s_t residual_tmp5 = eta_b*(residual_tmp0 + residual_tmp1 + u1_old_grad_1) + ((s_t(1) / s_t(2)))*lmbda*(residual_tmp3 + residual_tmp4);
      const s_t residual_tmp6 = eta_s*(u0_grad_1*u_dt_shift + u0_old_grad_1 + u1_grad_0*u_dt_shift + u1_old_grad_0) + mu*(u0_grad_1 + u1_grad_0);
      grad_coeff0_0_values = mu*residual_tmp3 + residual_tmp2 + residual_tmp5;
      grad_coeff0_1_values = residual_tmp6;
      grad_coeff1_0_values = residual_tmp6;
      grad_coeff1_1_values = mu*residual_tmp4 - residual_tmp2 + residual_tmp5;
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
      const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
      s_t *const RSTR output_row0 = output[test * NC];
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
        output_row0[0] += q_weight[q] * det * (grad_coeff0_0_values * test_grad0 + grad_coeff0_1_values * test_grad1);
        output_row1[0] += q_weight[q] * det * (grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1);
      }
    }
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d2_simplex_residual_block_contiguous(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
    const s_t current[2 * NS],
    const s_t previous[2 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t output[2 * NS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_grad_0_ref_values;
    s_t u0_grad_1_ref_values;
    s_t u0_old_grad_0_ref_values;
    s_t u0_old_grad_1_ref_values;
    s_t u1_grad_0_ref_values;
    s_t u1_grad_1_ref_values;
    s_t u1_old_grad_0_ref_values;
    s_t u1_old_grad_1_ref_values;
    s_t grad_coeff0_0_values;
    s_t grad_coeff0_1_values;
    s_t grad_coeff1_0_values;
    s_t grad_coeff1_1_values;
    {
      u0_grad_0_ref_values = s_t(0);
      u0_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        u0_grad_0_ref_values += current[trial * NC] * trial_grad_0_ref;
        u0_grad_1_ref_values += current[trial * NC] * trial_grad_1_ref;
      }
    }
    {
      u0_old_grad_0_ref_values = s_t(0);
      u0_old_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        u0_old_grad_0_ref_values += previous[trial * NC] * trial_grad_0_ref;
        u0_old_grad_1_ref_values += previous[trial * NC] * trial_grad_1_ref;
      }
    }
    {
      u1_grad_0_ref_values = s_t(0);
      u1_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        u1_grad_0_ref_values += current[trial * NC + 1] * trial_grad_0_ref;
        u1_grad_1_ref_values += current[trial * NC + 1] * trial_grad_1_ref;
      }
    }
    {
      u1_old_grad_0_ref_values = s_t(0);
      u1_old_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        u1_old_grad_0_ref_values += previous[trial * NC + 1] * trial_grad_0_ref;
        u1_old_grad_1_ref_values += previous[trial * NC + 1] * trial_grad_1_ref;
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t u0_grad_0_ref = u0_grad_0_ref_values;
      const s_t u0_grad_1_ref = u0_grad_1_ref_values;
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
      const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values;
      const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values;
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
      const s_t u1_grad_0_ref = u1_grad_0_ref_values;
      const s_t u1_grad_1_ref = u1_grad_1_ref_values;
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values;
      const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values;
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
      const s_t residual_tmp0 = u1_grad_1*u_dt_shift;
      const s_t residual_tmp1 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp2 = eta_s*(-residual_tmp0 + residual_tmp1 - u1_old_grad_1);
      const s_t residual_tmp3 = s_t(2)*u0_grad_0;
      const s_t residual_tmp4 = s_t(2)*u1_grad_1;
      const s_t residual_tmp5 = eta_b*(residual_tmp0 + residual_tmp1 + u1_old_grad_1) + ((s_t(1) / s_t(2)))*lmbda*(residual_tmp3 + residual_tmp4);
      const s_t residual_tmp6 = eta_s*(u0_grad_1*u_dt_shift + u0_old_grad_1 + u1_grad_0*u_dt_shift + u1_old_grad_0) + mu*(u0_grad_1 + u1_grad_0);
      grad_coeff0_0_values = mu*residual_tmp3 + residual_tmp2 + residual_tmp5;
      grad_coeff0_1_values = residual_tmp6;
      grad_coeff1_0_values = residual_tmp6;
      grad_coeff1_1_values = mu*residual_tmp4 - residual_tmp2 + residual_tmp5;
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
      const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        const s_t adj0 = adjugate[0][goff];
        const s_t adj1 = adjugate[1][goff];
        const s_t adj2 = adjugate[2][goff];
        const s_t adj3 = adjugate[3][goff];
        const s_t test_grad0 = (test_grad_ref0 * adj0 + test_grad_ref1 * adj2) / det;
        const s_t test_grad1 = (test_grad_ref0 * adj1 + test_grad_ref1 * adj3) / det;
        output[test * NC] += q_weight[q] * det * (grad_coeff0_0_values * test_grad0 + grad_coeff0_1_values * test_grad1);
        output[test * NC + 1] += q_weight[q] * det * (grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1);
      }
    }
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d2_simplex_tri3_residual_block(
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR current[2 * NS],
    const s_t *const RSTR previous[2 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t *const RSTR output[2 * NS]
) {
  {
    const ptrdiff_t goff = 0;
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
    const s_t test0_grad0 = (-(adj0) - adj2) / det;
    const s_t test0_grad1 = (-(adj1) - adj3) / det;
    const s_t test1_grad0 = (adj0) / det;
    const s_t test1_grad1 = (adj1) / det;
    const s_t test2_grad0 = (adj2) / det;
    const s_t test2_grad1 = (adj3) / det;
    output[0][0] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff0_0 * test0_grad0 + grad_coeff0_1 * test0_grad1);
    output[1][0] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff1_0 * test0_grad0 + grad_coeff1_1 * test0_grad1);
    output[2][0] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff0_0 * test1_grad0 + grad_coeff0_1 * test1_grad1);
    output[3][0] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff1_0 * test1_grad0 + grad_coeff1_1 * test1_grad1);
    output[4][0] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff0_0 * test2_grad0 + grad_coeff0_1 * test2_grad1);
    output[5][0] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff1_0 * test2_grad0 + grad_coeff1_1 * test2_grad1);
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d2_simplex_tri3_residual_block_contiguous(
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t current[2 * NS],
    const s_t previous[2 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t output[2 * NS]
) {
  {
    const ptrdiff_t goff = 0;
    const s_t det = determinant[goff];
    const s_t adj0 = adjugate[0][goff];
    const s_t adj1 = adjugate[1][goff];
    const s_t adj2 = adjugate[2][goff];
    const s_t adj3 = adjugate[3][goff];
    const s_t u0_grad_0_ref = -(current[0]) + current[2];
    const s_t u0_grad_1_ref = -(current[0]) + current[4];
    const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
    const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
    const s_t u0_old_grad_0_ref = -(previous[0]) + previous[2];
    const s_t u0_old_grad_1_ref = -(previous[0]) + previous[4];
    const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
    const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
    const s_t u1_grad_0_ref = -(current[1]) + current[3];
    const s_t u1_grad_1_ref = -(current[1]) + current[5];
    const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
    const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
    const s_t u1_old_grad_0_ref = -(previous[1]) + previous[3];
    const s_t u1_old_grad_1_ref = -(previous[1]) + previous[5];
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
    const s_t test0_grad0 = (-(adj0) - adj2) / det;
    const s_t test0_grad1 = (-(adj1) - adj3) / det;
    const s_t test1_grad0 = (adj0) / det;
    const s_t test1_grad1 = (adj1) / det;
    const s_t test2_grad0 = (adj2) / det;
    const s_t test2_grad1 = (adj3) / det;
    output[0] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff0_0 * test0_grad0 + grad_coeff0_1 * test0_grad1);
    output[1] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff1_0 * test0_grad0 + grad_coeff1_1 * test0_grad1);
    output[2] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff0_0 * test1_grad0 + grad_coeff0_1 * test1_grad1);
    output[3] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff1_0 * test1_grad0 + grad_coeff1_1 * test1_grad1);
    output[4] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff0_0 * test2_grad0 + grad_coeff0_1 * test2_grad1);
    output[5] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff1_0 * test2_grad0 + grad_coeff1_1 * test2_grad1);
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d2_simplex_jacobian_action_block(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
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
    s_t u0_direction_grad_0_ref_values;
    s_t u0_direction_grad_1_ref_values;
    s_t u1_direction_grad_0_ref_values;
    s_t u1_direction_grad_1_ref_values;
    s_t grad_coeff0_0_values;
    s_t grad_coeff0_1_values;
    s_t grad_coeff1_0_values;
    s_t grad_coeff1_1_values;
    {
      u0_direction_grad_0_ref_values = s_t(0);
      u0_direction_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = direction[trial * NC];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        u0_direction_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        u0_direction_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
      }
    }
    {
      u1_direction_grad_0_ref_values = s_t(0);
      u1_direction_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = direction[trial * NC + 1];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        u1_direction_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        u1_direction_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t u0_direction_grad_0_ref = u0_direction_grad_0_ref_values;
      const s_t u0_direction_grad_1_ref = u0_direction_grad_1_ref_values;
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
      const s_t u1_direction_grad_0_ref = u1_direction_grad_0_ref_values;
      const s_t u1_direction_grad_1_ref = u1_direction_grad_1_ref_values;
      const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj2) / det;
      const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj3) / det;
      const s_t residual_tmp0 = eta_s*u_dt_shift;
      const s_t residual_tmp1 = eta_b*u_dt_shift + lmbda;
      const s_t residual_tmp2 = -residual_tmp0 + residual_tmp1;
      const s_t residual_tmp3 = s_t(2)*mu + residual_tmp0 + residual_tmp1;
      const s_t residual_tmp4 = mu + residual_tmp0;
      const s_t residual_tmp5 = residual_tmp4*u0_direction_grad_1 + residual_tmp4*u1_direction_grad_0;
      grad_coeff0_0_values = residual_tmp2*u1_direction_grad_1 + residual_tmp3*u0_direction_grad_0;
      grad_coeff0_1_values = residual_tmp5;
      grad_coeff1_0_values = residual_tmp5;
      grad_coeff1_1_values = residual_tmp2*u0_direction_grad_0 + residual_tmp3*u1_direction_grad_1;
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
      const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
      s_t *const RSTR output_row0 = output[test * NC];
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
        output_row0[0] += q_weight[q] * det * (grad_coeff0_0_values * test_grad0 + grad_coeff0_1_values * test_grad1);
        output_row1[0] += q_weight[q] * det * (grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1);
      }
    }
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d2_simplex_jacobian_action_block_contiguous(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
    const s_t direction[2 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t output[2 * NS]
) {
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_direction_grad_0_ref_values;
    s_t u0_direction_grad_1_ref_values;
    s_t u1_direction_grad_0_ref_values;
    s_t u1_direction_grad_1_ref_values;
    s_t grad_coeff0_0_values;
    s_t grad_coeff0_1_values;
    s_t grad_coeff1_0_values;
    s_t grad_coeff1_1_values;
    {
      u0_direction_grad_0_ref_values = s_t(0);
      u0_direction_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        u0_direction_grad_0_ref_values += direction[trial * NC] * trial_grad_0_ref;
        u0_direction_grad_1_ref_values += direction[trial * NC] * trial_grad_1_ref;
      }
    }
    {
      u1_direction_grad_0_ref_values = s_t(0);
      u1_direction_grad_1_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      {
        u1_direction_grad_0_ref_values += direction[trial * NC + 1] * trial_grad_0_ref;
        u1_direction_grad_1_ref_values += direction[trial * NC + 1] * trial_grad_1_ref;
      }
    }
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      const s_t u0_direction_grad_0_ref = u0_direction_grad_0_ref_values;
      const s_t u0_direction_grad_1_ref = u0_direction_grad_1_ref_values;
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
      const s_t u1_direction_grad_0_ref = u1_direction_grad_0_ref_values;
      const s_t u1_direction_grad_1_ref = u1_direction_grad_1_ref_values;
      const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj2) / det;
      const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj3) / det;
      const s_t residual_tmp0 = eta_s*u_dt_shift;
      const s_t residual_tmp1 = eta_b*u_dt_shift + lmbda;
      const s_t residual_tmp2 = -residual_tmp0 + residual_tmp1;
      const s_t residual_tmp3 = s_t(2)*mu + residual_tmp0 + residual_tmp1;
      const s_t residual_tmp4 = mu + residual_tmp0;
      const s_t residual_tmp5 = residual_tmp4*u0_direction_grad_1 + residual_tmp4*u1_direction_grad_0;
      grad_coeff0_0_values = residual_tmp2*u1_direction_grad_1 + residual_tmp3*u0_direction_grad_0;
      grad_coeff0_1_values = residual_tmp5;
      grad_coeff1_0_values = residual_tmp5;
      grad_coeff1_1_values = residual_tmp2*u0_direction_grad_0 + residual_tmp3*u1_direction_grad_1;
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
      const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        const s_t adj0 = adjugate[0][goff];
        const s_t adj1 = adjugate[1][goff];
        const s_t adj2 = adjugate[2][goff];
        const s_t adj3 = adjugate[3][goff];
        const s_t test_grad0 = (test_grad_ref0 * adj0 + test_grad_ref1 * adj2) / det;
        const s_t test_grad1 = (test_grad_ref0 * adj1 + test_grad_ref1 * adj3) / det;
        output[test * NC] += q_weight[q] * det * (grad_coeff0_0_values * test_grad0 + grad_coeff0_1_values * test_grad1);
        output[test * NC + 1] += q_weight[q] * det * (grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1);
      }
    }
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d2_simplex_tri3_jacobian_action_block(
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR direction[2 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t *const RSTR output[2 * NS]
) {
  {
    const ptrdiff_t goff = 0;
    const s_t det = determinant[goff];
    const s_t adj0 = adjugate[0][goff];
    const s_t adj1 = adjugate[1][goff];
    const s_t adj2 = adjugate[2][goff];
    const s_t adj3 = adjugate[3][goff];
    const s_t u0_direction_grad_0_ref = -(direction[0][0]) + direction[2][0];
    const s_t u0_direction_grad_1_ref = -(direction[0][0]) + direction[4][0];
    const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
    const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
    const s_t u1_direction_grad_0_ref = -(direction[1][0]) + direction[3][0];
    const s_t u1_direction_grad_1_ref = -(direction[1][0]) + direction[5][0];
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
    const s_t test0_grad0 = (-(adj0) - adj2) / det;
    const s_t test0_grad1 = (-(adj1) - adj3) / det;
    const s_t test1_grad0 = (adj0) / det;
    const s_t test1_grad1 = (adj1) / det;
    const s_t test2_grad0 = (adj2) / det;
    const s_t test2_grad1 = (adj3) / det;
    output[0][0] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff0_0 * test0_grad0 + grad_coeff0_1 * test0_grad1);
    output[1][0] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff1_0 * test0_grad0 + grad_coeff1_1 * test0_grad1);
    output[2][0] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff0_0 * test1_grad0 + grad_coeff0_1 * test1_grad1);
    output[3][0] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff1_0 * test1_grad0 + grad_coeff1_1 * test1_grad1);
    output[4][0] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff0_0 * test2_grad0 + grad_coeff0_1 * test2_grad1);
    output[5][0] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff1_0 * test2_grad0 + grad_coeff1_1 * test2_grad1);
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d2_simplex_tri3_jacobian_action_block_contiguous(
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t direction[2 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t output[2 * NS]
) {
  {
    const ptrdiff_t goff = 0;
    const s_t det = determinant[goff];
    const s_t adj0 = adjugate[0][goff];
    const s_t adj1 = adjugate[1][goff];
    const s_t adj2 = adjugate[2][goff];
    const s_t adj3 = adjugate[3][goff];
    const s_t u0_direction_grad_0_ref = -(direction[0]) + direction[2];
    const s_t u0_direction_grad_1_ref = -(direction[0]) + direction[4];
    const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
    const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
    const s_t u1_direction_grad_0_ref = -(direction[1]) + direction[3];
    const s_t u1_direction_grad_1_ref = -(direction[1]) + direction[5];
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
    const s_t test0_grad0 = (-(adj0) - adj2) / det;
    const s_t test0_grad1 = (-(adj1) - adj3) / det;
    const s_t test1_grad0 = (adj0) / det;
    const s_t test1_grad1 = (adj1) / det;
    const s_t test2_grad0 = (adj2) / det;
    const s_t test2_grad1 = (adj3) / det;
    output[0] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff0_0 * test0_grad0 + grad_coeff0_1 * test0_grad1);
    output[1] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff1_0 * test0_grad0 + grad_coeff1_1 * test0_grad1);
    output[2] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff0_0 * test1_grad0 + grad_coeff0_1 * test1_grad1);
    output[3] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff1_0 * test1_grad0 + grad_coeff1_1 * test1_grad1);
    output[4] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff0_0 * test2_grad0 + grad_coeff0_1 * test2_grad1);
    output[5] += ((s_t(1) / s_t(2))) * (det) * (grad_coeff1_0 * test2_grad0 + grad_coeff1_1 * test2_grad1);
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d2_simplex_tri3_hessian_block(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
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
    const s_t basis0_grad0 = (-(adj0) - adj2) / det;
    const s_t basis0_grad1 = (-(adj1) - adj3) / det;
    const s_t basis1_grad0 = (adj0) / det;
    const s_t basis1_grad1 = (adj1) / det;
    const s_t basis2_grad0 = (adj2) / det;
    const s_t basis2_grad1 = (adj3) / det;
    const s_t element_matrix_tmp0 = pow_2(basis0_grad1);
    const s_t element_matrix_tmp1 = eta_s*u_dt_shift;
    const s_t element_matrix_tmp2 = element_matrix_tmp1 + mu;
    const s_t element_matrix_tmp3 = pow_2(basis0_grad0);
    const s_t element_matrix_tmp4 = eta_b*u_dt_shift + lmbda;
    const s_t element_matrix_tmp5 = element_matrix_tmp1 + element_matrix_tmp4 + s_t(2)*mu;
    const s_t element_matrix_tmp6 = ((s_t(1) / s_t(2)))*det;
    const s_t element_matrix_tmp7 = basis0_grad1*element_matrix_tmp2;
    const s_t element_matrix_tmp8 = basis0_grad0*element_matrix_tmp5;
    const s_t element_matrix_tmp9 = element_matrix_tmp6*(basis1_grad0*element_matrix_tmp8 + basis1_grad1*element_matrix_tmp7);
    const s_t element_matrix_tmp10 = element_matrix_tmp6*(basis2_grad0*element_matrix_tmp8 + basis2_grad1*element_matrix_tmp7);
    const s_t element_matrix_tmp11 = -element_matrix_tmp1 + element_matrix_tmp4;
    const s_t element_matrix_tmp12 = basis0_grad0*element_matrix_tmp11;
    const s_t element_matrix_tmp13 = element_matrix_tmp6*(basis0_grad0*element_matrix_tmp7 + basis0_grad1*element_matrix_tmp12);
    const s_t element_matrix_tmp14 = element_matrix_tmp6*(basis1_grad0*element_matrix_tmp7 + basis1_grad1*element_matrix_tmp12);
    const s_t element_matrix_tmp15 = element_matrix_tmp6*(basis2_grad0*element_matrix_tmp7 + basis2_grad1*element_matrix_tmp12);
    const s_t element_matrix_tmp16 = pow_2(basis1_grad1);
    const s_t element_matrix_tmp17 = pow_2(basis1_grad0);
    const s_t element_matrix_tmp18 = basis1_grad1*element_matrix_tmp2;
    const s_t element_matrix_tmp19 = basis1_grad0*basis2_grad0;
    const s_t element_matrix_tmp20 = element_matrix_tmp6*(basis2_grad1*element_matrix_tmp18 + element_matrix_tmp19*element_matrix_tmp5);
    const s_t element_matrix_tmp21 = basis1_grad0*element_matrix_tmp11;
    const s_t element_matrix_tmp22 = element_matrix_tmp6*(basis0_grad0*element_matrix_tmp18 + basis0_grad1*element_matrix_tmp21);
    const s_t element_matrix_tmp23 = element_matrix_tmp6*(basis1_grad0*element_matrix_tmp18 + basis1_grad1*element_matrix_tmp21);
    const s_t element_matrix_tmp24 = element_matrix_tmp6*(basis2_grad0*element_matrix_tmp18 + basis2_grad1*element_matrix_tmp21);
    const s_t element_matrix_tmp25 = pow_2(basis2_grad1);
    const s_t element_matrix_tmp26 = pow_2(basis2_grad0);
    const s_t element_matrix_tmp27 = basis2_grad1*element_matrix_tmp2;
    const s_t element_matrix_tmp28 = basis2_grad0*element_matrix_tmp11;
    const s_t element_matrix_tmp29 = element_matrix_tmp6*(basis0_grad0*element_matrix_tmp27 + basis0_grad1*element_matrix_tmp28);
    const s_t element_matrix_tmp30 = element_matrix_tmp6*(basis1_grad0*element_matrix_tmp27 + basis1_grad1*element_matrix_tmp28);
    const s_t element_matrix_tmp31 = element_matrix_tmp6*(basis2_grad0*element_matrix_tmp27 + basis2_grad1*element_matrix_tmp28);
    const s_t element_matrix_tmp32 = basis0_grad0*element_matrix_tmp2;
    const s_t element_matrix_tmp33 = basis0_grad1*element_matrix_tmp5;
    const s_t element_matrix_tmp34 = element_matrix_tmp6*(basis1_grad0*element_matrix_tmp32 + basis1_grad1*element_matrix_tmp33);
    const s_t element_matrix_tmp35 = element_matrix_tmp6*(basis2_grad0*element_matrix_tmp32 + basis2_grad1*element_matrix_tmp33);
    const s_t element_matrix_tmp36 = element_matrix_tmp6*(basis1_grad1*basis2_grad1*element_matrix_tmp5 + element_matrix_tmp19*element_matrix_tmp2);
    element_matrix[0] = element_matrix_tmp6*(element_matrix_tmp0*element_matrix_tmp2 + element_matrix_tmp3*element_matrix_tmp5);
    element_matrix[6] = element_matrix_tmp9;
    element_matrix[12] = element_matrix_tmp10;
    element_matrix[18] = element_matrix_tmp13;
    element_matrix[24] = element_matrix_tmp14;
    element_matrix[30] = element_matrix_tmp15;
    element_matrix[1] = element_matrix_tmp9;
    element_matrix[7] = element_matrix_tmp6*(element_matrix_tmp16*element_matrix_tmp2 + element_matrix_tmp17*element_matrix_tmp5);
    element_matrix[13] = element_matrix_tmp20;
    element_matrix[19] = element_matrix_tmp22;
    element_matrix[25] = element_matrix_tmp23;
    element_matrix[31] = element_matrix_tmp24;
    element_matrix[2] = element_matrix_tmp10;
    element_matrix[8] = element_matrix_tmp20;
    element_matrix[14] = element_matrix_tmp6*(element_matrix_tmp2*element_matrix_tmp25 + element_matrix_tmp26*element_matrix_tmp5);
    element_matrix[20] = element_matrix_tmp29;
    element_matrix[26] = element_matrix_tmp30;
    element_matrix[32] = element_matrix_tmp31;
    element_matrix[3] = element_matrix_tmp13;
    element_matrix[9] = element_matrix_tmp22;
    element_matrix[15] = element_matrix_tmp29;
    element_matrix[21] = element_matrix_tmp6*(element_matrix_tmp0*element_matrix_tmp5 + element_matrix_tmp2*element_matrix_tmp3);
    element_matrix[27] = element_matrix_tmp34;
    element_matrix[33] = element_matrix_tmp35;
    element_matrix[4] = element_matrix_tmp14;
    element_matrix[10] = element_matrix_tmp23;
    element_matrix[16] = element_matrix_tmp30;
    element_matrix[22] = element_matrix_tmp34;
    element_matrix[28] = element_matrix_tmp6*(element_matrix_tmp16*element_matrix_tmp5 + element_matrix_tmp17*element_matrix_tmp2);
    element_matrix[34] = element_matrix_tmp36;
    element_matrix[5] = element_matrix_tmp15;
    element_matrix[11] = element_matrix_tmp24;
    element_matrix[17] = element_matrix_tmp31;
    element_matrix[23] = element_matrix_tmp35;
    element_matrix[29] = element_matrix_tmp36;
    element_matrix[35] = element_matrix_tmp6*(element_matrix_tmp2*element_matrix_tmp26 + element_matrix_tmp25*element_matrix_tmp5);
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d2_simplex_hessian_block(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t *const RSTR element_matrix
) {
  for (int entry = 0; entry < 2 * NS * 2 * NS; ++entry) {
    element_matrix[entry] = s_t(0);
  }
  s_t grad_coeff0_0_values;
  s_t grad_coeff0_1_values;
  s_t grad_coeff1_0_values;
  s_t grad_coeff1_1_values;
  s_t tangent[16];
  for (int q = 0; q < NQ; ++q) {
    {
      const s_t tangent_tmp0 = eta_s*u_dt_shift;
      const s_t tangent_tmp1 = eta_b*u_dt_shift + lmbda;
      const s_t tangent_tmp2 = s_t(2)*mu + tangent_tmp0 + tangent_tmp1;
      const s_t tangent_tmp3 = -tangent_tmp0 + tangent_tmp1;
      const s_t tangent_tmp4 = mu + tangent_tmp0;
      tangent[0] = tangent_tmp2;
      tangent[1] = 0;
      tangent[2] = 0;
      tangent[3] = tangent_tmp3;
      tangent[4] = 0;
      tangent[5] = tangent_tmp4;
      tangent[6] = tangent_tmp4;
      tangent[7] = 0;
      tangent[8] = 0;
      tangent[9] = tangent_tmp4;
      tangent[10] = tangent_tmp4;
      tangent[11] = 0;
      tangent[12] = tangent_tmp3;
      tangent[13] = 0;
      tangent[14] = 0;
      tangent[15] = tangent_tmp2;
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_grad_ref0 = grad_ref_x[q * NS + trial];
      const s_t trial_grad_ref1 = grad_ref_y[q * NS + trial];
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        const s_t adj0 = adjugate[0][goff];
        const s_t adj1 = adjugate[1][goff];
        const s_t adj2 = adjugate[2][goff];
        const s_t adj3 = adjugate[3][goff];
        const s_t trial_grad0 = (trial_grad_ref0 * adj0 + trial_grad_ref1 * adj2) / det;
        const s_t trial_grad1 = (trial_grad_ref0 * adj1 + trial_grad_ref1 * adj3) / det;
        grad_coeff0_0_values = trial_grad0 * tangent[0] + trial_grad1 * tangent[4];
        grad_coeff0_1_values = trial_grad0 * tangent[1] + trial_grad1 * tangent[5];
        grad_coeff1_0_values = trial_grad0 * tangent[2] + trial_grad1 * tangent[6];
        grad_coeff1_1_values = trial_grad0 * tangent[3] + trial_grad1 * tangent[7];
      }
      for (int test = 0; test < NS; ++test) {
        const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
        const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
        {
          const ptrdiff_t goff = q * geometry_stride;
          const s_t det = determinant[goff];
          const s_t adj0 = adjugate[0][goff];
          const s_t adj1 = adjugate[1][goff];
          const s_t adj2 = adjugate[2][goff];
          const s_t adj3 = adjugate[3][goff];
          const s_t test_grad0 = (test_grad_ref0 * adj0 + test_grad_ref1 * adj2) / det;
          const s_t test_grad1 = (test_grad_ref0 * adj1 + test_grad_ref1 * adj3) / det;
          element_matrix[(0 * NS + test) * 2 * NS + 0 * NS + trial] += q_weight[q] * det * (grad_coeff0_0_values * test_grad0 + grad_coeff0_1_values * test_grad1);
          element_matrix[(1 * NS + test) * 2 * NS + 0 * NS + trial] += q_weight[q] * det * (grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1);
        }
      }
      {
        const ptrdiff_t goff = q * geometry_stride;
        const s_t det = determinant[goff];
        const s_t adj0 = adjugate[0][goff];
        const s_t adj1 = adjugate[1][goff];
        const s_t adj2 = adjugate[2][goff];
        const s_t adj3 = adjugate[3][goff];
        const s_t trial_grad0 = (trial_grad_ref0 * adj0 + trial_grad_ref1 * adj2) / det;
        const s_t trial_grad1 = (trial_grad_ref0 * adj1 + trial_grad_ref1 * adj3) / det;
        grad_coeff0_0_values = trial_grad0 * tangent[8] + trial_grad1 * tangent[12];
        grad_coeff0_1_values = trial_grad0 * tangent[9] + trial_grad1 * tangent[13];
        grad_coeff1_0_values = trial_grad0 * tangent[10] + trial_grad1 * tangent[14];
        grad_coeff1_1_values = trial_grad0 * tangent[11] + trial_grad1 * tangent[15];
      }
      for (int test = 0; test < NS; ++test) {
        const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
        const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
        {
          const ptrdiff_t goff = q * geometry_stride;
          const s_t det = determinant[goff];
          const s_t adj0 = adjugate[0][goff];
          const s_t adj1 = adjugate[1][goff];
          const s_t adj2 = adjugate[2][goff];
          const s_t adj3 = adjugate[3][goff];
          const s_t test_grad0 = (test_grad_ref0 * adj0 + test_grad_ref1 * adj2) / det;
          const s_t test_grad1 = (test_grad_ref0 * adj1 + test_grad_ref1 * adj3) / det;
          element_matrix[(0 * NS + test) * 2 * NS + 1 * NS + trial] += q_weight[q] * det * (grad_coeff0_0_values * test_grad0 + grad_coeff0_1_values * test_grad1);
          element_matrix[(1 * NS + test) * 2 * NS + 1 * NS + trial] += q_weight[q] * det * (grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1);
        }
      }
    }
  }
}

} // namespace codegen
} // namespace sfem

#endif
