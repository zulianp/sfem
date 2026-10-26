#ifndef LINEAR_KELVIN_VOIGT_TOTAL_D3_SIMPLEX_LOCAL_HPP
#define LINEAR_KELVIN_VOIGT_TOTAL_D3_SIMPLEX_LOCAL_HPP

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
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d3_simplex_residual_block(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[3 * NS],
    const s_t *const RSTR previous[3 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t *const RSTR output[3 * NS]
) {
  static constexpr int NC = 3;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_grad_0_ref_values;
    s_t u0_grad_1_ref_values;
    s_t u0_grad_2_ref_values;
    s_t u0_old_grad_0_ref_values;
    s_t u0_old_grad_1_ref_values;
    s_t u0_old_grad_2_ref_values;
    s_t u1_grad_0_ref_values;
    s_t u1_grad_1_ref_values;
    s_t u1_grad_2_ref_values;
    s_t u1_old_grad_0_ref_values;
    s_t u1_old_grad_1_ref_values;
    s_t u1_old_grad_2_ref_values;
    s_t u2_grad_0_ref_values;
    s_t u2_grad_1_ref_values;
    s_t u2_grad_2_ref_values;
    s_t u2_old_grad_0_ref_values;
    s_t u2_old_grad_1_ref_values;
    s_t u2_old_grad_2_ref_values;
    s_t grad_coeff0_0_values;
    s_t grad_coeff0_1_values;
    s_t grad_coeff0_2_values;
    s_t grad_coeff1_0_values;
    s_t grad_coeff1_1_values;
    s_t grad_coeff1_2_values;
    s_t grad_coeff2_0_values;
    s_t grad_coeff2_1_values;
    s_t grad_coeff2_2_values;
    {
      u0_grad_0_ref_values = s_t(0);
      u0_grad_1_ref_values = s_t(0);
      u0_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = current[trial * NC];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        u0_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        u0_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
        u0_grad_2_ref_values += coeff_stream[0] * trial_grad_2_ref;
      }
    }
    {
      u0_old_grad_0_ref_values = s_t(0);
      u0_old_grad_1_ref_values = s_t(0);
      u0_old_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = previous[trial * NC];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        u0_old_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        u0_old_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
        u0_old_grad_2_ref_values += coeff_stream[0] * trial_grad_2_ref;
      }
    }
    {
      u1_grad_0_ref_values = s_t(0);
      u1_grad_1_ref_values = s_t(0);
      u1_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = current[trial * NC + 1];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        u1_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        u1_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
        u1_grad_2_ref_values += coeff_stream[0] * trial_grad_2_ref;
      }
    }
    {
      u1_old_grad_0_ref_values = s_t(0);
      u1_old_grad_1_ref_values = s_t(0);
      u1_old_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = previous[trial * NC + 1];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        u1_old_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        u1_old_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
        u1_old_grad_2_ref_values += coeff_stream[0] * trial_grad_2_ref;
      }
    }
    {
      u2_grad_0_ref_values = s_t(0);
      u2_grad_1_ref_values = s_t(0);
      u2_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = current[trial * NC + 2];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        u2_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        u2_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
        u2_grad_2_ref_values += coeff_stream[0] * trial_grad_2_ref;
      }
    }
    {
      u2_old_grad_0_ref_values = s_t(0);
      u2_old_grad_1_ref_values = s_t(0);
      u2_old_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = previous[trial * NC + 2];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        u2_old_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        u2_old_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
        u2_old_grad_2_ref_values += coeff_stream[0] * trial_grad_2_ref;
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
      const s_t u0_grad_0_ref = u0_grad_0_ref_values;
      const s_t u0_grad_1_ref = u0_grad_1_ref_values;
      const s_t u0_grad_2_ref = u0_grad_2_ref_values;
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
      const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
      const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values;
      const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values;
      const s_t u0_old_grad_2_ref = u0_old_grad_2_ref_values;
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
      const s_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
      const s_t u1_grad_0_ref = u1_grad_0_ref_values;
      const s_t u1_grad_1_ref = u1_grad_1_ref_values;
      const s_t u1_grad_2_ref = u1_grad_2_ref_values;
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
      const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
      const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values;
      const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values;
      const s_t u1_old_grad_2_ref = u1_old_grad_2_ref_values;
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
      const s_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
      const s_t u2_grad_0_ref = u2_grad_0_ref_values;
      const s_t u2_grad_1_ref = u2_grad_1_ref_values;
      const s_t u2_grad_2_ref = u2_grad_2_ref_values;
      const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
      const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
      const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
      const s_t u2_old_grad_0_ref = u2_old_grad_0_ref_values;
      const s_t u2_old_grad_1_ref = u2_old_grad_1_ref_values;
      const s_t u2_old_grad_2_ref = u2_old_grad_2_ref_values;
      const s_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
      const s_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
      const s_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = s_t(2)*u0_grad_0;
      const s_t residual_tmp1 = u0_grad_0*u_dt_shift;
      const s_t residual_tmp2 = u1_grad_1*u_dt_shift;
      const s_t residual_tmp3 = u2_grad_2*u_dt_shift;
      const s_t residual_tmp4 = residual_tmp2 + residual_tmp3 + u1_old_grad_1 + u2_old_grad_2;
      const s_t residual_tmp5 = ((s_t(2) / s_t(3)))*eta_s;
      const s_t residual_tmp6 = residual_tmp1 + u0_old_grad_0;
      const s_t residual_tmp7 = s_t(2)*u1_grad_1;
      const s_t residual_tmp8 = s_t(2)*u2_grad_2;
      const s_t residual_tmp9 = eta_b*(residual_tmp4 + residual_tmp6) + ((s_t(1) / s_t(2)))*lmbda*(residual_tmp0 + residual_tmp7 + residual_tmp8);
      const s_t residual_tmp10 = eta_s*(u0_grad_1*u_dt_shift + u0_old_grad_1 + u1_grad_0*u_dt_shift + u1_old_grad_0) + mu*(u0_grad_1 + u1_grad_0);
      const s_t residual_tmp11 = eta_s*(u0_grad_2*u_dt_shift + u0_old_grad_2 + u2_grad_0*u_dt_shift + u2_old_grad_0) + mu*(u0_grad_2 + u2_grad_0);
      const s_t residual_tmp12 = eta_s*(u1_grad_2*u_dt_shift + u1_old_grad_2 + u2_grad_1*u_dt_shift + u2_old_grad_1) + mu*(u1_grad_2 + u2_grad_1);
      grad_coeff0_0_values = mu*residual_tmp0 - residual_tmp5*(-s_t(2)*residual_tmp1 + residual_tmp4 - s_t(2)*u0_old_grad_0) + residual_tmp9;
      grad_coeff0_1_values = residual_tmp10;
      grad_coeff0_2_values = residual_tmp11;
      grad_coeff1_0_values = residual_tmp10;
      grad_coeff1_1_values = mu*residual_tmp7 - residual_tmp5*(-s_t(2)*residual_tmp2 + residual_tmp3 + residual_tmp6 - s_t(2)*u1_old_grad_1 + u2_old_grad_2) + residual_tmp9;
      grad_coeff1_2_values = residual_tmp12;
      grad_coeff2_0_values = residual_tmp11;
      grad_coeff2_1_values = residual_tmp12;
      grad_coeff2_2_values = mu*residual_tmp8 - residual_tmp5*(residual_tmp2 - s_t(2)*residual_tmp3 + residual_tmp6 + u1_old_grad_1 - s_t(2)*u2_old_grad_2) + residual_tmp9;
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
      const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
      const s_t test_grad_ref2 = grad_ref_z[q * NS + test];
      s_t *const RSTR output_row0 = output[test * NC];
      s_t *const RSTR output_row1 = output[test * NC + 1];
      s_t *const RSTR output_row2 = output[test * NC + 2];
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
        output_row0[0] += q_weight[q] * det * (grad_coeff0_0_values * test_grad0 + grad_coeff0_1_values * test_grad1 + grad_coeff0_2_values * test_grad2);
        output_row1[0] += q_weight[q] * det * (grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1 + grad_coeff1_2_values * test_grad2);
        output_row2[0] += q_weight[q] * det * (grad_coeff2_0_values * test_grad0 + grad_coeff2_1_values * test_grad1 + grad_coeff2_2_values * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d3_simplex_residual_block_contiguous(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t current[3 * NS],
    const s_t previous[3 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t output[3 * NS]
) {
  static constexpr int NC = 3;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_grad_0_ref_values;
    s_t u0_grad_1_ref_values;
    s_t u0_grad_2_ref_values;
    s_t u0_old_grad_0_ref_values;
    s_t u0_old_grad_1_ref_values;
    s_t u0_old_grad_2_ref_values;
    s_t u1_grad_0_ref_values;
    s_t u1_grad_1_ref_values;
    s_t u1_grad_2_ref_values;
    s_t u1_old_grad_0_ref_values;
    s_t u1_old_grad_1_ref_values;
    s_t u1_old_grad_2_ref_values;
    s_t u2_grad_0_ref_values;
    s_t u2_grad_1_ref_values;
    s_t u2_grad_2_ref_values;
    s_t u2_old_grad_0_ref_values;
    s_t u2_old_grad_1_ref_values;
    s_t u2_old_grad_2_ref_values;
    s_t grad_coeff0_0_values;
    s_t grad_coeff0_1_values;
    s_t grad_coeff0_2_values;
    s_t grad_coeff1_0_values;
    s_t grad_coeff1_1_values;
    s_t grad_coeff1_2_values;
    s_t grad_coeff2_0_values;
    s_t grad_coeff2_1_values;
    s_t grad_coeff2_2_values;
    {
      u0_grad_0_ref_values = s_t(0);
      u0_grad_1_ref_values = s_t(0);
      u0_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        u0_grad_0_ref_values += current[trial * NC] * trial_grad_0_ref;
        u0_grad_1_ref_values += current[trial * NC] * trial_grad_1_ref;
        u0_grad_2_ref_values += current[trial * NC] * trial_grad_2_ref;
      }
    }
    {
      u0_old_grad_0_ref_values = s_t(0);
      u0_old_grad_1_ref_values = s_t(0);
      u0_old_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        u0_old_grad_0_ref_values += previous[trial * NC] * trial_grad_0_ref;
        u0_old_grad_1_ref_values += previous[trial * NC] * trial_grad_1_ref;
        u0_old_grad_2_ref_values += previous[trial * NC] * trial_grad_2_ref;
      }
    }
    {
      u1_grad_0_ref_values = s_t(0);
      u1_grad_1_ref_values = s_t(0);
      u1_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        u1_grad_0_ref_values += current[trial * NC + 1] * trial_grad_0_ref;
        u1_grad_1_ref_values += current[trial * NC + 1] * trial_grad_1_ref;
        u1_grad_2_ref_values += current[trial * NC + 1] * trial_grad_2_ref;
      }
    }
    {
      u1_old_grad_0_ref_values = s_t(0);
      u1_old_grad_1_ref_values = s_t(0);
      u1_old_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        u1_old_grad_0_ref_values += previous[trial * NC + 1] * trial_grad_0_ref;
        u1_old_grad_1_ref_values += previous[trial * NC + 1] * trial_grad_1_ref;
        u1_old_grad_2_ref_values += previous[trial * NC + 1] * trial_grad_2_ref;
      }
    }
    {
      u2_grad_0_ref_values = s_t(0);
      u2_grad_1_ref_values = s_t(0);
      u2_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        u2_grad_0_ref_values += current[trial * NC + 2] * trial_grad_0_ref;
        u2_grad_1_ref_values += current[trial * NC + 2] * trial_grad_1_ref;
        u2_grad_2_ref_values += current[trial * NC + 2] * trial_grad_2_ref;
      }
    }
    {
      u2_old_grad_0_ref_values = s_t(0);
      u2_old_grad_1_ref_values = s_t(0);
      u2_old_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        u2_old_grad_0_ref_values += previous[trial * NC + 2] * trial_grad_0_ref;
        u2_old_grad_1_ref_values += previous[trial * NC + 2] * trial_grad_1_ref;
        u2_old_grad_2_ref_values += previous[trial * NC + 2] * trial_grad_2_ref;
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
      const s_t u0_grad_0_ref = u0_grad_0_ref_values;
      const s_t u0_grad_1_ref = u0_grad_1_ref_values;
      const s_t u0_grad_2_ref = u0_grad_2_ref_values;
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
      const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
      const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values;
      const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values;
      const s_t u0_old_grad_2_ref = u0_old_grad_2_ref_values;
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
      const s_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
      const s_t u1_grad_0_ref = u1_grad_0_ref_values;
      const s_t u1_grad_1_ref = u1_grad_1_ref_values;
      const s_t u1_grad_2_ref = u1_grad_2_ref_values;
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
      const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
      const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values;
      const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values;
      const s_t u1_old_grad_2_ref = u1_old_grad_2_ref_values;
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
      const s_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
      const s_t u2_grad_0_ref = u2_grad_0_ref_values;
      const s_t u2_grad_1_ref = u2_grad_1_ref_values;
      const s_t u2_grad_2_ref = u2_grad_2_ref_values;
      const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
      const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
      const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
      const s_t u2_old_grad_0_ref = u2_old_grad_0_ref_values;
      const s_t u2_old_grad_1_ref = u2_old_grad_1_ref_values;
      const s_t u2_old_grad_2_ref = u2_old_grad_2_ref_values;
      const s_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
      const s_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
      const s_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = s_t(2)*u0_grad_0;
      const s_t residual_tmp1 = u0_grad_0*u_dt_shift;
      const s_t residual_tmp2 = u1_grad_1*u_dt_shift;
      const s_t residual_tmp3 = u2_grad_2*u_dt_shift;
      const s_t residual_tmp4 = residual_tmp2 + residual_tmp3 + u1_old_grad_1 + u2_old_grad_2;
      const s_t residual_tmp5 = ((s_t(2) / s_t(3)))*eta_s;
      const s_t residual_tmp6 = residual_tmp1 + u0_old_grad_0;
      const s_t residual_tmp7 = s_t(2)*u1_grad_1;
      const s_t residual_tmp8 = s_t(2)*u2_grad_2;
      const s_t residual_tmp9 = eta_b*(residual_tmp4 + residual_tmp6) + ((s_t(1) / s_t(2)))*lmbda*(residual_tmp0 + residual_tmp7 + residual_tmp8);
      const s_t residual_tmp10 = eta_s*(u0_grad_1*u_dt_shift + u0_old_grad_1 + u1_grad_0*u_dt_shift + u1_old_grad_0) + mu*(u0_grad_1 + u1_grad_0);
      const s_t residual_tmp11 = eta_s*(u0_grad_2*u_dt_shift + u0_old_grad_2 + u2_grad_0*u_dt_shift + u2_old_grad_0) + mu*(u0_grad_2 + u2_grad_0);
      const s_t residual_tmp12 = eta_s*(u1_grad_2*u_dt_shift + u1_old_grad_2 + u2_grad_1*u_dt_shift + u2_old_grad_1) + mu*(u1_grad_2 + u2_grad_1);
      grad_coeff0_0_values = mu*residual_tmp0 - residual_tmp5*(-s_t(2)*residual_tmp1 + residual_tmp4 - s_t(2)*u0_old_grad_0) + residual_tmp9;
      grad_coeff0_1_values = residual_tmp10;
      grad_coeff0_2_values = residual_tmp11;
      grad_coeff1_0_values = residual_tmp10;
      grad_coeff1_1_values = mu*residual_tmp7 - residual_tmp5*(-s_t(2)*residual_tmp2 + residual_tmp3 + residual_tmp6 - s_t(2)*u1_old_grad_1 + u2_old_grad_2) + residual_tmp9;
      grad_coeff1_2_values = residual_tmp12;
      grad_coeff2_0_values = residual_tmp11;
      grad_coeff2_1_values = residual_tmp12;
      grad_coeff2_2_values = mu*residual_tmp8 - residual_tmp5*(residual_tmp2 - s_t(2)*residual_tmp3 + residual_tmp6 + u1_old_grad_1 - s_t(2)*u2_old_grad_2) + residual_tmp9;
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
      const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
      const s_t test_grad_ref2 = grad_ref_z[q * NS + test];
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
        output[test * NC] += q_weight[q] * det * (grad_coeff0_0_values * test_grad0 + grad_coeff0_1_values * test_grad1 + grad_coeff0_2_values * test_grad2);
        output[test * NC + 1] += q_weight[q] * det * (grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1 + grad_coeff1_2_values * test_grad2);
        output[test * NC + 2] += q_weight[q] * det * (grad_coeff2_0_values * test_grad0 + grad_coeff2_1_values * test_grad1 + grad_coeff2_2_values * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d3_simplex_tet4_residual_block(
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR current[3 * NS],
    const s_t *const RSTR previous[3 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t *const RSTR output[3 * NS]
) {
  {
    const ptrdiff_t goff = 0;
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
    const s_t u0_grad_0_ref = -(current[0][0]) + current[3][0];
    const s_t u0_grad_1_ref = -(current[0][0]) + current[6][0];
    const s_t u0_grad_2_ref = -(current[0][0]) + current[9][0];
    const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
    const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
    const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
    const s_t u0_old_grad_0_ref = -(previous[0][0]) + previous[3][0];
    const s_t u0_old_grad_1_ref = -(previous[0][0]) + previous[6][0];
    const s_t u0_old_grad_2_ref = -(previous[0][0]) + previous[9][0];
    const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
    const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
    const s_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
    const s_t u1_grad_0_ref = -(current[1][0]) + current[4][0];
    const s_t u1_grad_1_ref = -(current[1][0]) + current[7][0];
    const s_t u1_grad_2_ref = -(current[1][0]) + current[10][0];
    const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
    const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
    const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
    const s_t u1_old_grad_0_ref = -(previous[1][0]) + previous[4][0];
    const s_t u1_old_grad_1_ref = -(previous[1][0]) + previous[7][0];
    const s_t u1_old_grad_2_ref = -(previous[1][0]) + previous[10][0];
    const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
    const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
    const s_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
    const s_t u2_grad_0_ref = -(current[2][0]) + current[5][0];
    const s_t u2_grad_1_ref = -(current[2][0]) + current[8][0];
    const s_t u2_grad_2_ref = -(current[2][0]) + current[11][0];
    const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
    const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
    const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
    const s_t u2_old_grad_0_ref = -(previous[2][0]) + previous[5][0];
    const s_t u2_old_grad_1_ref = -(previous[2][0]) + previous[8][0];
    const s_t u2_old_grad_2_ref = -(previous[2][0]) + previous[11][0];
    const s_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
    const s_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
    const s_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
    const s_t residual_tmp0 = s_t(2)*u0_grad_0;
    const s_t residual_tmp1 = u0_grad_0*u_dt_shift;
    const s_t residual_tmp2 = u1_grad_1*u_dt_shift;
    const s_t residual_tmp3 = u2_grad_2*u_dt_shift;
    const s_t residual_tmp4 = residual_tmp2 + residual_tmp3 + u1_old_grad_1 + u2_old_grad_2;
    const s_t residual_tmp5 = ((s_t(2) / s_t(3)))*eta_s;
    const s_t residual_tmp6 = residual_tmp1 + u0_old_grad_0;
    const s_t residual_tmp7 = s_t(2)*u1_grad_1;
    const s_t residual_tmp8 = s_t(2)*u2_grad_2;
    const s_t residual_tmp9 = eta_b*(residual_tmp4 + residual_tmp6) + ((s_t(1) / s_t(2)))*lmbda*(residual_tmp0 + residual_tmp7 + residual_tmp8);
    const s_t residual_tmp10 = eta_s*(u0_grad_1*u_dt_shift + u0_old_grad_1 + u1_grad_0*u_dt_shift + u1_old_grad_0) + mu*(u0_grad_1 + u1_grad_0);
    const s_t residual_tmp11 = eta_s*(u0_grad_2*u_dt_shift + u0_old_grad_2 + u2_grad_0*u_dt_shift + u2_old_grad_0) + mu*(u0_grad_2 + u2_grad_0);
    const s_t residual_tmp12 = eta_s*(u1_grad_2*u_dt_shift + u1_old_grad_2 + u2_grad_1*u_dt_shift + u2_old_grad_1) + mu*(u1_grad_2 + u2_grad_1);
    const s_t grad_coeff0_0 = mu*residual_tmp0 - residual_tmp5*(-s_t(2)*residual_tmp1 + residual_tmp4 - s_t(2)*u0_old_grad_0) + residual_tmp9;
    const s_t grad_coeff0_1 = residual_tmp10;
    const s_t grad_coeff0_2 = residual_tmp11;
    const s_t grad_coeff1_0 = residual_tmp10;
    const s_t grad_coeff1_1 = mu*residual_tmp7 - residual_tmp5*(-s_t(2)*residual_tmp2 + residual_tmp3 + residual_tmp6 - s_t(2)*u1_old_grad_1 + u2_old_grad_2) + residual_tmp9;
    const s_t grad_coeff1_2 = residual_tmp12;
    const s_t grad_coeff2_0 = residual_tmp11;
    const s_t grad_coeff2_1 = residual_tmp12;
    const s_t grad_coeff2_2 = mu*residual_tmp8 - residual_tmp5*(residual_tmp2 - s_t(2)*residual_tmp3 + residual_tmp6 + u1_old_grad_1 - s_t(2)*u2_old_grad_2) + residual_tmp9;
    const s_t test0_grad0 = (-(adj0) - adj3 - adj6) / det;
    const s_t test0_grad1 = (-(adj1) - adj4 - adj7) / det;
    const s_t test0_grad2 = (-(adj2) - adj5 - adj8) / det;
    const s_t test1_grad0 = (adj0) / det;
    const s_t test1_grad1 = (adj1) / det;
    const s_t test1_grad2 = (adj2) / det;
    const s_t test2_grad0 = (adj3) / det;
    const s_t test2_grad1 = (adj4) / det;
    const s_t test2_grad2 = (adj5) / det;
    const s_t test3_grad0 = (adj6) / det;
    const s_t test3_grad1 = (adj7) / det;
    const s_t test3_grad2 = (adj8) / det;
    output[0][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test0_grad0 + grad_coeff0_1 * test0_grad1 + grad_coeff0_2 * test0_grad2);
    output[1][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test0_grad0 + grad_coeff1_1 * test0_grad1 + grad_coeff1_2 * test0_grad2);
    output[2][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test0_grad0 + grad_coeff2_1 * test0_grad1 + grad_coeff2_2 * test0_grad2);
    output[3][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test1_grad0 + grad_coeff0_1 * test1_grad1 + grad_coeff0_2 * test1_grad2);
    output[4][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test1_grad0 + grad_coeff1_1 * test1_grad1 + grad_coeff1_2 * test1_grad2);
    output[5][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test1_grad0 + grad_coeff2_1 * test1_grad1 + grad_coeff2_2 * test1_grad2);
    output[6][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test2_grad0 + grad_coeff0_1 * test2_grad1 + grad_coeff0_2 * test2_grad2);
    output[7][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test2_grad0 + grad_coeff1_1 * test2_grad1 + grad_coeff1_2 * test2_grad2);
    output[8][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test2_grad0 + grad_coeff2_1 * test2_grad1 + grad_coeff2_2 * test2_grad2);
    output[9][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test3_grad0 + grad_coeff0_1 * test3_grad1 + grad_coeff0_2 * test3_grad2);
    output[10][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test3_grad0 + grad_coeff1_1 * test3_grad1 + grad_coeff1_2 * test3_grad2);
    output[11][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test3_grad0 + grad_coeff2_1 * test3_grad1 + grad_coeff2_2 * test3_grad2);
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d3_simplex_tet4_residual_block_contiguous(
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t current[3 * NS],
    const s_t previous[3 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t output[3 * NS]
) {
  {
    const ptrdiff_t goff = 0;
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
    const s_t u0_grad_0_ref = -(current[0]) + current[3];
    const s_t u0_grad_1_ref = -(current[0]) + current[6];
    const s_t u0_grad_2_ref = -(current[0]) + current[9];
    const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
    const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
    const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
    const s_t u0_old_grad_0_ref = -(previous[0]) + previous[3];
    const s_t u0_old_grad_1_ref = -(previous[0]) + previous[6];
    const s_t u0_old_grad_2_ref = -(previous[0]) + previous[9];
    const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
    const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
    const s_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
    const s_t u1_grad_0_ref = -(current[1]) + current[4];
    const s_t u1_grad_1_ref = -(current[1]) + current[7];
    const s_t u1_grad_2_ref = -(current[1]) + current[10];
    const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
    const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
    const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
    const s_t u1_old_grad_0_ref = -(previous[1]) + previous[4];
    const s_t u1_old_grad_1_ref = -(previous[1]) + previous[7];
    const s_t u1_old_grad_2_ref = -(previous[1]) + previous[10];
    const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
    const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
    const s_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
    const s_t u2_grad_0_ref = -(current[2]) + current[5];
    const s_t u2_grad_1_ref = -(current[2]) + current[8];
    const s_t u2_grad_2_ref = -(current[2]) + current[11];
    const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
    const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
    const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
    const s_t u2_old_grad_0_ref = -(previous[2]) + previous[5];
    const s_t u2_old_grad_1_ref = -(previous[2]) + previous[8];
    const s_t u2_old_grad_2_ref = -(previous[2]) + previous[11];
    const s_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
    const s_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
    const s_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
    const s_t residual_tmp0 = s_t(2)*u0_grad_0;
    const s_t residual_tmp1 = u0_grad_0*u_dt_shift;
    const s_t residual_tmp2 = u1_grad_1*u_dt_shift;
    const s_t residual_tmp3 = u2_grad_2*u_dt_shift;
    const s_t residual_tmp4 = residual_tmp2 + residual_tmp3 + u1_old_grad_1 + u2_old_grad_2;
    const s_t residual_tmp5 = ((s_t(2) / s_t(3)))*eta_s;
    const s_t residual_tmp6 = residual_tmp1 + u0_old_grad_0;
    const s_t residual_tmp7 = s_t(2)*u1_grad_1;
    const s_t residual_tmp8 = s_t(2)*u2_grad_2;
    const s_t residual_tmp9 = eta_b*(residual_tmp4 + residual_tmp6) + ((s_t(1) / s_t(2)))*lmbda*(residual_tmp0 + residual_tmp7 + residual_tmp8);
    const s_t residual_tmp10 = eta_s*(u0_grad_1*u_dt_shift + u0_old_grad_1 + u1_grad_0*u_dt_shift + u1_old_grad_0) + mu*(u0_grad_1 + u1_grad_0);
    const s_t residual_tmp11 = eta_s*(u0_grad_2*u_dt_shift + u0_old_grad_2 + u2_grad_0*u_dt_shift + u2_old_grad_0) + mu*(u0_grad_2 + u2_grad_0);
    const s_t residual_tmp12 = eta_s*(u1_grad_2*u_dt_shift + u1_old_grad_2 + u2_grad_1*u_dt_shift + u2_old_grad_1) + mu*(u1_grad_2 + u2_grad_1);
    const s_t grad_coeff0_0 = mu*residual_tmp0 - residual_tmp5*(-s_t(2)*residual_tmp1 + residual_tmp4 - s_t(2)*u0_old_grad_0) + residual_tmp9;
    const s_t grad_coeff0_1 = residual_tmp10;
    const s_t grad_coeff0_2 = residual_tmp11;
    const s_t grad_coeff1_0 = residual_tmp10;
    const s_t grad_coeff1_1 = mu*residual_tmp7 - residual_tmp5*(-s_t(2)*residual_tmp2 + residual_tmp3 + residual_tmp6 - s_t(2)*u1_old_grad_1 + u2_old_grad_2) + residual_tmp9;
    const s_t grad_coeff1_2 = residual_tmp12;
    const s_t grad_coeff2_0 = residual_tmp11;
    const s_t grad_coeff2_1 = residual_tmp12;
    const s_t grad_coeff2_2 = mu*residual_tmp8 - residual_tmp5*(residual_tmp2 - s_t(2)*residual_tmp3 + residual_tmp6 + u1_old_grad_1 - s_t(2)*u2_old_grad_2) + residual_tmp9;
    const s_t test0_grad0 = (-(adj0) - adj3 - adj6) / det;
    const s_t test0_grad1 = (-(adj1) - adj4 - adj7) / det;
    const s_t test0_grad2 = (-(adj2) - adj5 - adj8) / det;
    const s_t test1_grad0 = (adj0) / det;
    const s_t test1_grad1 = (adj1) / det;
    const s_t test1_grad2 = (adj2) / det;
    const s_t test2_grad0 = (adj3) / det;
    const s_t test2_grad1 = (adj4) / det;
    const s_t test2_grad2 = (adj5) / det;
    const s_t test3_grad0 = (adj6) / det;
    const s_t test3_grad1 = (adj7) / det;
    const s_t test3_grad2 = (adj8) / det;
    output[0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test0_grad0 + grad_coeff0_1 * test0_grad1 + grad_coeff0_2 * test0_grad2);
    output[1] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test0_grad0 + grad_coeff1_1 * test0_grad1 + grad_coeff1_2 * test0_grad2);
    output[2] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test0_grad0 + grad_coeff2_1 * test0_grad1 + grad_coeff2_2 * test0_grad2);
    output[3] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test1_grad0 + grad_coeff0_1 * test1_grad1 + grad_coeff0_2 * test1_grad2);
    output[4] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test1_grad0 + grad_coeff1_1 * test1_grad1 + grad_coeff1_2 * test1_grad2);
    output[5] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test1_grad0 + grad_coeff2_1 * test1_grad1 + grad_coeff2_2 * test1_grad2);
    output[6] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test2_grad0 + grad_coeff0_1 * test2_grad1 + grad_coeff0_2 * test2_grad2);
    output[7] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test2_grad0 + grad_coeff1_1 * test2_grad1 + grad_coeff1_2 * test2_grad2);
    output[8] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test2_grad0 + grad_coeff2_1 * test2_grad1 + grad_coeff2_2 * test2_grad2);
    output[9] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test3_grad0 + grad_coeff0_1 * test3_grad1 + grad_coeff0_2 * test3_grad2);
    output[10] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test3_grad0 + grad_coeff1_1 * test3_grad1 + grad_coeff1_2 * test3_grad2);
    output[11] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test3_grad0 + grad_coeff2_1 * test3_grad1 + grad_coeff2_2 * test3_grad2);
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d3_simplex_jacobian_action_block(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR direction[3 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t *const RSTR output[3 * NS]
) {
  static constexpr int NC = 3;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_direction_grad_0_ref_values;
    s_t u0_direction_grad_1_ref_values;
    s_t u0_direction_grad_2_ref_values;
    s_t u1_direction_grad_0_ref_values;
    s_t u1_direction_grad_1_ref_values;
    s_t u1_direction_grad_2_ref_values;
    s_t u2_direction_grad_0_ref_values;
    s_t u2_direction_grad_1_ref_values;
    s_t u2_direction_grad_2_ref_values;
    s_t grad_coeff0_0_values;
    s_t grad_coeff0_1_values;
    s_t grad_coeff0_2_values;
    s_t grad_coeff1_0_values;
    s_t grad_coeff1_1_values;
    s_t grad_coeff1_2_values;
    s_t grad_coeff2_0_values;
    s_t grad_coeff2_1_values;
    s_t grad_coeff2_2_values;
    {
      u0_direction_grad_0_ref_values = s_t(0);
      u0_direction_grad_1_ref_values = s_t(0);
      u0_direction_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = direction[trial * NC];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        u0_direction_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        u0_direction_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
        u0_direction_grad_2_ref_values += coeff_stream[0] * trial_grad_2_ref;
      }
    }
    {
      u1_direction_grad_0_ref_values = s_t(0);
      u1_direction_grad_1_ref_values = s_t(0);
      u1_direction_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = direction[trial * NC + 1];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        u1_direction_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        u1_direction_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
        u1_direction_grad_2_ref_values += coeff_stream[0] * trial_grad_2_ref;
      }
    }
    {
      u2_direction_grad_0_ref_values = s_t(0);
      u2_direction_grad_1_ref_values = s_t(0);
      u2_direction_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t *const RSTR coeff_stream = direction[trial * NC + 2];
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        u2_direction_grad_0_ref_values += coeff_stream[0] * trial_grad_0_ref;
        u2_direction_grad_1_ref_values += coeff_stream[0] * trial_grad_1_ref;
        u2_direction_grad_2_ref_values += coeff_stream[0] * trial_grad_2_ref;
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
      const s_t u0_direction_grad_0_ref = u0_direction_grad_0_ref_values;
      const s_t u0_direction_grad_1_ref = u0_direction_grad_1_ref_values;
      const s_t u0_direction_grad_2_ref = u0_direction_grad_2_ref_values;
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
      const s_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
      const s_t u1_direction_grad_0_ref = u1_direction_grad_0_ref_values;
      const s_t u1_direction_grad_1_ref = u1_direction_grad_1_ref_values;
      const s_t u1_direction_grad_2_ref = u1_direction_grad_2_ref_values;
      const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
      const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
      const s_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
      const s_t u2_direction_grad_0_ref = u2_direction_grad_0_ref_values;
      const s_t u2_direction_grad_1_ref = u2_direction_grad_1_ref_values;
      const s_t u2_direction_grad_2_ref = u2_direction_grad_2_ref_values;
      const s_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
      const s_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
      const s_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = eta_s*u_dt_shift;
      const s_t residual_tmp1 = eta_b*u_dt_shift + lmbda;
      const s_t residual_tmp2 = -(s_t(2) / s_t(3))*residual_tmp0 + residual_tmp1;
      const s_t residual_tmp3 = residual_tmp2*u1_direction_grad_1;
      const s_t residual_tmp4 = residual_tmp2*u2_direction_grad_2;
      const s_t residual_tmp5 = s_t(2)*mu + ((s_t(4) / s_t(3)))*residual_tmp0 + residual_tmp1;
      const s_t residual_tmp6 = mu + residual_tmp0;
      const s_t residual_tmp7 = residual_tmp6*u0_direction_grad_1 + residual_tmp6*u1_direction_grad_0;
      const s_t residual_tmp8 = residual_tmp6*u0_direction_grad_2 + residual_tmp6*u2_direction_grad_0;
      const s_t residual_tmp9 = residual_tmp2*u0_direction_grad_0;
      const s_t residual_tmp10 = residual_tmp6*u1_direction_grad_2 + residual_tmp6*u2_direction_grad_1;
      grad_coeff0_0_values = residual_tmp3 + residual_tmp4 + residual_tmp5*u0_direction_grad_0;
      grad_coeff0_1_values = residual_tmp7;
      grad_coeff0_2_values = residual_tmp8;
      grad_coeff1_0_values = residual_tmp7;
      grad_coeff1_1_values = residual_tmp4 + residual_tmp5*u1_direction_grad_1 + residual_tmp9;
      grad_coeff1_2_values = residual_tmp10;
      grad_coeff2_0_values = residual_tmp8;
      grad_coeff2_1_values = residual_tmp10;
      grad_coeff2_2_values = residual_tmp3 + residual_tmp5*u2_direction_grad_2 + residual_tmp9;
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
      const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
      const s_t test_grad_ref2 = grad_ref_z[q * NS + test];
      s_t *const RSTR output_row0 = output[test * NC];
      s_t *const RSTR output_row1 = output[test * NC + 1];
      s_t *const RSTR output_row2 = output[test * NC + 2];
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
        output_row0[0] += q_weight[q] * det * (grad_coeff0_0_values * test_grad0 + grad_coeff0_1_values * test_grad1 + grad_coeff0_2_values * test_grad2);
        output_row1[0] += q_weight[q] * det * (grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1 + grad_coeff1_2_values * test_grad2);
        output_row2[0] += q_weight[q] * det * (grad_coeff2_0_values * test_grad0 + grad_coeff2_1_values * test_grad1 + grad_coeff2_2_values * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d3_simplex_jacobian_action_block_contiguous(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t direction[3 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t output[3 * NS]
) {
  static constexpr int NC = 3;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_direction_grad_0_ref_values;
    s_t u0_direction_grad_1_ref_values;
    s_t u0_direction_grad_2_ref_values;
    s_t u1_direction_grad_0_ref_values;
    s_t u1_direction_grad_1_ref_values;
    s_t u1_direction_grad_2_ref_values;
    s_t u2_direction_grad_0_ref_values;
    s_t u2_direction_grad_1_ref_values;
    s_t u2_direction_grad_2_ref_values;
    s_t grad_coeff0_0_values;
    s_t grad_coeff0_1_values;
    s_t grad_coeff0_2_values;
    s_t grad_coeff1_0_values;
    s_t grad_coeff1_1_values;
    s_t grad_coeff1_2_values;
    s_t grad_coeff2_0_values;
    s_t grad_coeff2_1_values;
    s_t grad_coeff2_2_values;
    {
      u0_direction_grad_0_ref_values = s_t(0);
      u0_direction_grad_1_ref_values = s_t(0);
      u0_direction_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        u0_direction_grad_0_ref_values += direction[trial * NC] * trial_grad_0_ref;
        u0_direction_grad_1_ref_values += direction[trial * NC] * trial_grad_1_ref;
        u0_direction_grad_2_ref_values += direction[trial * NC] * trial_grad_2_ref;
      }
    }
    {
      u1_direction_grad_0_ref_values = s_t(0);
      u1_direction_grad_1_ref_values = s_t(0);
      u1_direction_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        u1_direction_grad_0_ref_values += direction[trial * NC + 1] * trial_grad_0_ref;
        u1_direction_grad_1_ref_values += direction[trial * NC + 1] * trial_grad_1_ref;
        u1_direction_grad_2_ref_values += direction[trial * NC + 1] * trial_grad_2_ref;
      }
    }
    {
      u2_direction_grad_0_ref_values = s_t(0);
      u2_direction_grad_1_ref_values = s_t(0);
      u2_direction_grad_2_ref_values = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_grad_0_ref = grad_ref_x[q * NS + trial];
      const s_t trial_grad_1_ref = grad_ref_y[q * NS + trial];
      const s_t trial_grad_2_ref = grad_ref_z[q * NS + trial];
      {
        u2_direction_grad_0_ref_values += direction[trial * NC + 2] * trial_grad_0_ref;
        u2_direction_grad_1_ref_values += direction[trial * NC + 2] * trial_grad_1_ref;
        u2_direction_grad_2_ref_values += direction[trial * NC + 2] * trial_grad_2_ref;
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
      const s_t u0_direction_grad_0_ref = u0_direction_grad_0_ref_values;
      const s_t u0_direction_grad_1_ref = u0_direction_grad_1_ref_values;
      const s_t u0_direction_grad_2_ref = u0_direction_grad_2_ref_values;
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
      const s_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
      const s_t u1_direction_grad_0_ref = u1_direction_grad_0_ref_values;
      const s_t u1_direction_grad_1_ref = u1_direction_grad_1_ref_values;
      const s_t u1_direction_grad_2_ref = u1_direction_grad_2_ref_values;
      const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
      const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
      const s_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
      const s_t u2_direction_grad_0_ref = u2_direction_grad_0_ref_values;
      const s_t u2_direction_grad_1_ref = u2_direction_grad_1_ref_values;
      const s_t u2_direction_grad_2_ref = u2_direction_grad_2_ref_values;
      const s_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
      const s_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
      const s_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = eta_s*u_dt_shift;
      const s_t residual_tmp1 = eta_b*u_dt_shift + lmbda;
      const s_t residual_tmp2 = -(s_t(2) / s_t(3))*residual_tmp0 + residual_tmp1;
      const s_t residual_tmp3 = residual_tmp2*u1_direction_grad_1;
      const s_t residual_tmp4 = residual_tmp2*u2_direction_grad_2;
      const s_t residual_tmp5 = s_t(2)*mu + ((s_t(4) / s_t(3)))*residual_tmp0 + residual_tmp1;
      const s_t residual_tmp6 = mu + residual_tmp0;
      const s_t residual_tmp7 = residual_tmp6*u0_direction_grad_1 + residual_tmp6*u1_direction_grad_0;
      const s_t residual_tmp8 = residual_tmp6*u0_direction_grad_2 + residual_tmp6*u2_direction_grad_0;
      const s_t residual_tmp9 = residual_tmp2*u0_direction_grad_0;
      const s_t residual_tmp10 = residual_tmp6*u1_direction_grad_2 + residual_tmp6*u2_direction_grad_1;
      grad_coeff0_0_values = residual_tmp3 + residual_tmp4 + residual_tmp5*u0_direction_grad_0;
      grad_coeff0_1_values = residual_tmp7;
      grad_coeff0_2_values = residual_tmp8;
      grad_coeff1_0_values = residual_tmp7;
      grad_coeff1_1_values = residual_tmp4 + residual_tmp5*u1_direction_grad_1 + residual_tmp9;
      grad_coeff1_2_values = residual_tmp10;
      grad_coeff2_0_values = residual_tmp8;
      grad_coeff2_1_values = residual_tmp10;
      grad_coeff2_2_values = residual_tmp3 + residual_tmp5*u2_direction_grad_2 + residual_tmp9;
    }
    for (int test = 0; test < NS; ++test) {
      const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
      const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
      const s_t test_grad_ref2 = grad_ref_z[q * NS + test];
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
        output[test * NC] += q_weight[q] * det * (grad_coeff0_0_values * test_grad0 + grad_coeff0_1_values * test_grad1 + grad_coeff0_2_values * test_grad2);
        output[test * NC + 1] += q_weight[q] * det * (grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1 + grad_coeff1_2_values * test_grad2);
        output[test * NC + 2] += q_weight[q] * det * (grad_coeff2_0_values * test_grad0 + grad_coeff2_1_values * test_grad1 + grad_coeff2_2_values * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d3_simplex_tet4_jacobian_action_block(
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR direction[3 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t *const RSTR output[3 * NS]
) {
  {
    const ptrdiff_t goff = 0;
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
    const s_t u0_direction_grad_0_ref = -(direction[0][0]) + direction[3][0];
    const s_t u0_direction_grad_1_ref = -(direction[0][0]) + direction[6][0];
    const s_t u0_direction_grad_2_ref = -(direction[0][0]) + direction[9][0];
    const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
    const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
    const s_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
    const s_t u1_direction_grad_0_ref = -(direction[1][0]) + direction[4][0];
    const s_t u1_direction_grad_1_ref = -(direction[1][0]) + direction[7][0];
    const s_t u1_direction_grad_2_ref = -(direction[1][0]) + direction[10][0];
    const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
    const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
    const s_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
    const s_t u2_direction_grad_0_ref = -(direction[2][0]) + direction[5][0];
    const s_t u2_direction_grad_1_ref = -(direction[2][0]) + direction[8][0];
    const s_t u2_direction_grad_2_ref = -(direction[2][0]) + direction[11][0];
    const s_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
    const s_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
    const s_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
    const s_t residual_tmp0 = eta_s*u_dt_shift;
    const s_t residual_tmp1 = eta_b*u_dt_shift + lmbda;
    const s_t residual_tmp2 = -(s_t(2) / s_t(3))*residual_tmp0 + residual_tmp1;
    const s_t residual_tmp3 = residual_tmp2*u1_direction_grad_1;
    const s_t residual_tmp4 = residual_tmp2*u2_direction_grad_2;
    const s_t residual_tmp5 = s_t(2)*mu + ((s_t(4) / s_t(3)))*residual_tmp0 + residual_tmp1;
    const s_t residual_tmp6 = mu + residual_tmp0;
    const s_t residual_tmp7 = residual_tmp6*u0_direction_grad_1 + residual_tmp6*u1_direction_grad_0;
    const s_t residual_tmp8 = residual_tmp6*u0_direction_grad_2 + residual_tmp6*u2_direction_grad_0;
    const s_t residual_tmp9 = residual_tmp2*u0_direction_grad_0;
    const s_t residual_tmp10 = residual_tmp6*u1_direction_grad_2 + residual_tmp6*u2_direction_grad_1;
    const s_t grad_coeff0_0 = residual_tmp3 + residual_tmp4 + residual_tmp5*u0_direction_grad_0;
    const s_t grad_coeff0_1 = residual_tmp7;
    const s_t grad_coeff0_2 = residual_tmp8;
    const s_t grad_coeff1_0 = residual_tmp7;
    const s_t grad_coeff1_1 = residual_tmp4 + residual_tmp5*u1_direction_grad_1 + residual_tmp9;
    const s_t grad_coeff1_2 = residual_tmp10;
    const s_t grad_coeff2_0 = residual_tmp8;
    const s_t grad_coeff2_1 = residual_tmp10;
    const s_t grad_coeff2_2 = residual_tmp3 + residual_tmp5*u2_direction_grad_2 + residual_tmp9;
    const s_t test0_grad0 = (-(adj0) - adj3 - adj6) / det;
    const s_t test0_grad1 = (-(adj1) - adj4 - adj7) / det;
    const s_t test0_grad2 = (-(adj2) - adj5 - adj8) / det;
    const s_t test1_grad0 = (adj0) / det;
    const s_t test1_grad1 = (adj1) / det;
    const s_t test1_grad2 = (adj2) / det;
    const s_t test2_grad0 = (adj3) / det;
    const s_t test2_grad1 = (adj4) / det;
    const s_t test2_grad2 = (adj5) / det;
    const s_t test3_grad0 = (adj6) / det;
    const s_t test3_grad1 = (adj7) / det;
    const s_t test3_grad2 = (adj8) / det;
    output[0][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test0_grad0 + grad_coeff0_1 * test0_grad1 + grad_coeff0_2 * test0_grad2);
    output[1][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test0_grad0 + grad_coeff1_1 * test0_grad1 + grad_coeff1_2 * test0_grad2);
    output[2][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test0_grad0 + grad_coeff2_1 * test0_grad1 + grad_coeff2_2 * test0_grad2);
    output[3][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test1_grad0 + grad_coeff0_1 * test1_grad1 + grad_coeff0_2 * test1_grad2);
    output[4][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test1_grad0 + grad_coeff1_1 * test1_grad1 + grad_coeff1_2 * test1_grad2);
    output[5][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test1_grad0 + grad_coeff2_1 * test1_grad1 + grad_coeff2_2 * test1_grad2);
    output[6][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test2_grad0 + grad_coeff0_1 * test2_grad1 + grad_coeff0_2 * test2_grad2);
    output[7][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test2_grad0 + grad_coeff1_1 * test2_grad1 + grad_coeff1_2 * test2_grad2);
    output[8][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test2_grad0 + grad_coeff2_1 * test2_grad1 + grad_coeff2_2 * test2_grad2);
    output[9][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test3_grad0 + grad_coeff0_1 * test3_grad1 + grad_coeff0_2 * test3_grad2);
    output[10][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test3_grad0 + grad_coeff1_1 * test3_grad1 + grad_coeff1_2 * test3_grad2);
    output[11][0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test3_grad0 + grad_coeff2_1 * test3_grad1 + grad_coeff2_2 * test3_grad2);
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d3_simplex_tet4_jacobian_action_block_contiguous(
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t direction[3 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t output[3 * NS]
) {
  {
    const ptrdiff_t goff = 0;
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
    const s_t u0_direction_grad_0_ref = -(direction[0]) + direction[3];
    const s_t u0_direction_grad_1_ref = -(direction[0]) + direction[6];
    const s_t u0_direction_grad_2_ref = -(direction[0]) + direction[9];
    const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
    const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
    const s_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
    const s_t u1_direction_grad_0_ref = -(direction[1]) + direction[4];
    const s_t u1_direction_grad_1_ref = -(direction[1]) + direction[7];
    const s_t u1_direction_grad_2_ref = -(direction[1]) + direction[10];
    const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
    const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
    const s_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
    const s_t u2_direction_grad_0_ref = -(direction[2]) + direction[5];
    const s_t u2_direction_grad_1_ref = -(direction[2]) + direction[8];
    const s_t u2_direction_grad_2_ref = -(direction[2]) + direction[11];
    const s_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
    const s_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
    const s_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
    const s_t residual_tmp0 = eta_s*u_dt_shift;
    const s_t residual_tmp1 = eta_b*u_dt_shift + lmbda;
    const s_t residual_tmp2 = -(s_t(2) / s_t(3))*residual_tmp0 + residual_tmp1;
    const s_t residual_tmp3 = residual_tmp2*u1_direction_grad_1;
    const s_t residual_tmp4 = residual_tmp2*u2_direction_grad_2;
    const s_t residual_tmp5 = s_t(2)*mu + ((s_t(4) / s_t(3)))*residual_tmp0 + residual_tmp1;
    const s_t residual_tmp6 = mu + residual_tmp0;
    const s_t residual_tmp7 = residual_tmp6*u0_direction_grad_1 + residual_tmp6*u1_direction_grad_0;
    const s_t residual_tmp8 = residual_tmp6*u0_direction_grad_2 + residual_tmp6*u2_direction_grad_0;
    const s_t residual_tmp9 = residual_tmp2*u0_direction_grad_0;
    const s_t residual_tmp10 = residual_tmp6*u1_direction_grad_2 + residual_tmp6*u2_direction_grad_1;
    const s_t grad_coeff0_0 = residual_tmp3 + residual_tmp4 + residual_tmp5*u0_direction_grad_0;
    const s_t grad_coeff0_1 = residual_tmp7;
    const s_t grad_coeff0_2 = residual_tmp8;
    const s_t grad_coeff1_0 = residual_tmp7;
    const s_t grad_coeff1_1 = residual_tmp4 + residual_tmp5*u1_direction_grad_1 + residual_tmp9;
    const s_t grad_coeff1_2 = residual_tmp10;
    const s_t grad_coeff2_0 = residual_tmp8;
    const s_t grad_coeff2_1 = residual_tmp10;
    const s_t grad_coeff2_2 = residual_tmp3 + residual_tmp5*u2_direction_grad_2 + residual_tmp9;
    const s_t test0_grad0 = (-(adj0) - adj3 - adj6) / det;
    const s_t test0_grad1 = (-(adj1) - adj4 - adj7) / det;
    const s_t test0_grad2 = (-(adj2) - adj5 - adj8) / det;
    const s_t test1_grad0 = (adj0) / det;
    const s_t test1_grad1 = (adj1) / det;
    const s_t test1_grad2 = (adj2) / det;
    const s_t test2_grad0 = (adj3) / det;
    const s_t test2_grad1 = (adj4) / det;
    const s_t test2_grad2 = (adj5) / det;
    const s_t test3_grad0 = (adj6) / det;
    const s_t test3_grad1 = (adj7) / det;
    const s_t test3_grad2 = (adj8) / det;
    output[0] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test0_grad0 + grad_coeff0_1 * test0_grad1 + grad_coeff0_2 * test0_grad2);
    output[1] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test0_grad0 + grad_coeff1_1 * test0_grad1 + grad_coeff1_2 * test0_grad2);
    output[2] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test0_grad0 + grad_coeff2_1 * test0_grad1 + grad_coeff2_2 * test0_grad2);
    output[3] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test1_grad0 + grad_coeff0_1 * test1_grad1 + grad_coeff0_2 * test1_grad2);
    output[4] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test1_grad0 + grad_coeff1_1 * test1_grad1 + grad_coeff1_2 * test1_grad2);
    output[5] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test1_grad0 + grad_coeff2_1 * test1_grad1 + grad_coeff2_2 * test1_grad2);
    output[6] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test2_grad0 + grad_coeff0_1 * test2_grad1 + grad_coeff0_2 * test2_grad2);
    output[7] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test2_grad0 + grad_coeff1_1 * test2_grad1 + grad_coeff1_2 * test2_grad2);
    output[8] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test2_grad0 + grad_coeff2_1 * test2_grad1 + grad_coeff2_2 * test2_grad2);
    output[9] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test3_grad0 + grad_coeff0_1 * test3_grad1 + grad_coeff0_2 * test3_grad2);
    output[10] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test3_grad0 + grad_coeff1_1 * test3_grad1 + grad_coeff1_2 * test3_grad2);
    output[11] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test3_grad0 + grad_coeff2_1 * test3_grad1 + grad_coeff2_2 * test3_grad2);
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d3_simplex_tet4_hessian_block(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
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
    const s_t adj4 = adjugate[4][goff];
    const s_t adj5 = adjugate[5][goff];
    const s_t adj6 = adjugate[6][goff];
    const s_t adj7 = adjugate[7][goff];
    const s_t adj8 = adjugate[8][goff];
    const s_t basis0_grad0 = (-(adj0) - adj3 - adj6) / det;
    const s_t basis0_grad1 = (-(adj1) - adj4 - adj7) / det;
    const s_t basis0_grad2 = (-(adj2) - adj5 - adj8) / det;
    const s_t basis1_grad0 = (adj0) / det;
    const s_t basis1_grad1 = (adj1) / det;
    const s_t basis1_grad2 = (adj2) / det;
    const s_t basis2_grad0 = (adj3) / det;
    const s_t basis2_grad1 = (adj4) / det;
    const s_t basis2_grad2 = (adj5) / det;
    const s_t basis3_grad0 = (adj6) / det;
    const s_t basis3_grad1 = (adj7) / det;
    const s_t basis3_grad2 = (adj8) / det;
    const s_t element_matrix_tmp0 = pow_2(basis0_grad1);
    const s_t element_matrix_tmp1 = eta_s*u_dt_shift;
    const s_t element_matrix_tmp2 = element_matrix_tmp1 + mu;
    const s_t element_matrix_tmp3 = element_matrix_tmp0*element_matrix_tmp2;
    const s_t element_matrix_tmp4 = pow_2(basis0_grad2);
    const s_t element_matrix_tmp5 = element_matrix_tmp2*element_matrix_tmp4;
    const s_t element_matrix_tmp6 = pow_2(basis0_grad0);
    const s_t element_matrix_tmp7 = eta_b*u_dt_shift + lmbda;
    const s_t element_matrix_tmp8 = ((s_t(4) / s_t(3)))*element_matrix_tmp1 + element_matrix_tmp7 + s_t(2)*mu;
    const s_t element_matrix_tmp9 = ((s_t(1) / s_t(6)))*det;
    const s_t element_matrix_tmp10 = basis0_grad1*element_matrix_tmp2;
    const s_t element_matrix_tmp11 = basis1_grad1*element_matrix_tmp10;
    const s_t element_matrix_tmp12 = basis0_grad2*element_matrix_tmp2;
    const s_t element_matrix_tmp13 = basis1_grad2*element_matrix_tmp12;
    const s_t element_matrix_tmp14 = basis0_grad0*element_matrix_tmp8;
    const s_t element_matrix_tmp15 = element_matrix_tmp9*(basis1_grad0*element_matrix_tmp14 + element_matrix_tmp11 + element_matrix_tmp13);
    const s_t element_matrix_tmp16 = basis2_grad1*element_matrix_tmp10;
    const s_t element_matrix_tmp17 = basis2_grad2*element_matrix_tmp12;
    const s_t element_matrix_tmp18 = element_matrix_tmp9*(basis2_grad0*element_matrix_tmp14 + element_matrix_tmp16 + element_matrix_tmp17);
    const s_t element_matrix_tmp19 = basis3_grad1*element_matrix_tmp10;
    const s_t element_matrix_tmp20 = basis3_grad2*element_matrix_tmp12;
    const s_t element_matrix_tmp21 = element_matrix_tmp9*(basis3_grad0*element_matrix_tmp14 + element_matrix_tmp19 + element_matrix_tmp20);
    const s_t element_matrix_tmp22 = -(s_t(2) / s_t(3))*element_matrix_tmp1 + element_matrix_tmp7;
    const s_t element_matrix_tmp23 = basis0_grad0*element_matrix_tmp22;
    const s_t element_matrix_tmp24 = element_matrix_tmp9*(basis0_grad0*element_matrix_tmp10 + basis0_grad1*element_matrix_tmp23);
    const s_t element_matrix_tmp25 = element_matrix_tmp9*(basis1_grad0*element_matrix_tmp10 + basis1_grad1*element_matrix_tmp23);
    const s_t element_matrix_tmp26 = element_matrix_tmp9*(basis2_grad0*element_matrix_tmp10 + basis2_grad1*element_matrix_tmp23);
    const s_t element_matrix_tmp27 = element_matrix_tmp9*(basis3_grad0*element_matrix_tmp10 + basis3_grad1*element_matrix_tmp23);
    const s_t element_matrix_tmp28 = element_matrix_tmp9*(basis0_grad0*element_matrix_tmp12 + basis0_grad2*element_matrix_tmp23);
    const s_t element_matrix_tmp29 = element_matrix_tmp9*(basis1_grad0*element_matrix_tmp12 + basis1_grad2*element_matrix_tmp23);
    const s_t element_matrix_tmp30 = element_matrix_tmp9*(basis2_grad0*element_matrix_tmp12 + basis2_grad2*element_matrix_tmp23);
    const s_t element_matrix_tmp31 = element_matrix_tmp9*(basis3_grad0*element_matrix_tmp12 + basis3_grad2*element_matrix_tmp23);
    const s_t element_matrix_tmp32 = pow_2(basis1_grad1);
    const s_t element_matrix_tmp33 = element_matrix_tmp2*element_matrix_tmp32;
    const s_t element_matrix_tmp34 = pow_2(basis1_grad2);
    const s_t element_matrix_tmp35 = element_matrix_tmp2*element_matrix_tmp34;
    const s_t element_matrix_tmp36 = pow_2(basis1_grad0);
    const s_t element_matrix_tmp37 = basis1_grad1*element_matrix_tmp2;
    const s_t element_matrix_tmp38 = basis2_grad1*element_matrix_tmp37;
    const s_t element_matrix_tmp39 = basis1_grad2*element_matrix_tmp2;
    const s_t element_matrix_tmp40 = basis2_grad2*element_matrix_tmp39;
    const s_t element_matrix_tmp41 = basis1_grad0*element_matrix_tmp8;
    const s_t element_matrix_tmp42 = element_matrix_tmp9*(basis2_grad0*element_matrix_tmp41 + element_matrix_tmp38 + element_matrix_tmp40);
    const s_t element_matrix_tmp43 = basis3_grad1*element_matrix_tmp37;
    const s_t element_matrix_tmp44 = basis3_grad2*element_matrix_tmp39;
    const s_t element_matrix_tmp45 = element_matrix_tmp9*(basis3_grad0*element_matrix_tmp41 + element_matrix_tmp43 + element_matrix_tmp44);
    const s_t element_matrix_tmp46 = basis1_grad0*element_matrix_tmp22;
    const s_t element_matrix_tmp47 = element_matrix_tmp9*(basis0_grad0*element_matrix_tmp37 + basis0_grad1*element_matrix_tmp46);
    const s_t element_matrix_tmp48 = element_matrix_tmp9*(basis1_grad0*element_matrix_tmp37 + basis1_grad1*element_matrix_tmp46);
    const s_t element_matrix_tmp49 = element_matrix_tmp9*(basis2_grad0*element_matrix_tmp37 + basis2_grad1*element_matrix_tmp46);
    const s_t element_matrix_tmp50 = element_matrix_tmp9*(basis3_grad0*element_matrix_tmp37 + basis3_grad1*element_matrix_tmp46);
    const s_t element_matrix_tmp51 = element_matrix_tmp9*(basis0_grad0*element_matrix_tmp39 + basis0_grad2*element_matrix_tmp46);
    const s_t element_matrix_tmp52 = element_matrix_tmp9*(basis1_grad0*element_matrix_tmp39 + basis1_grad2*element_matrix_tmp46);
    const s_t element_matrix_tmp53 = element_matrix_tmp9*(basis2_grad0*element_matrix_tmp39 + basis2_grad2*element_matrix_tmp46);
    const s_t element_matrix_tmp54 = element_matrix_tmp9*(basis3_grad0*element_matrix_tmp39 + basis3_grad2*element_matrix_tmp46);
    const s_t element_matrix_tmp55 = pow_2(basis2_grad1);
    const s_t element_matrix_tmp56 = element_matrix_tmp2*element_matrix_tmp55;
    const s_t element_matrix_tmp57 = pow_2(basis2_grad2);
    const s_t element_matrix_tmp58 = element_matrix_tmp2*element_matrix_tmp57;
    const s_t element_matrix_tmp59 = pow_2(basis2_grad0);
    const s_t element_matrix_tmp60 = basis2_grad1*element_matrix_tmp2;
    const s_t element_matrix_tmp61 = basis3_grad1*element_matrix_tmp60;
    const s_t element_matrix_tmp62 = basis2_grad2*element_matrix_tmp2;
    const s_t element_matrix_tmp63 = basis3_grad2*element_matrix_tmp62;
    const s_t element_matrix_tmp64 = basis2_grad0*basis3_grad0;
    const s_t element_matrix_tmp65 = element_matrix_tmp9*(element_matrix_tmp61 + element_matrix_tmp63 + element_matrix_tmp64*element_matrix_tmp8);
    const s_t element_matrix_tmp66 = basis2_grad0*element_matrix_tmp22;
    const s_t element_matrix_tmp67 = element_matrix_tmp9*(basis0_grad0*element_matrix_tmp60 + basis0_grad1*element_matrix_tmp66);
    const s_t element_matrix_tmp68 = element_matrix_tmp9*(basis1_grad0*element_matrix_tmp60 + basis1_grad1*element_matrix_tmp66);
    const s_t element_matrix_tmp69 = element_matrix_tmp9*(basis2_grad0*element_matrix_tmp60 + basis2_grad1*element_matrix_tmp66);
    const s_t element_matrix_tmp70 = element_matrix_tmp9*(basis3_grad0*element_matrix_tmp60 + basis3_grad1*element_matrix_tmp66);
    const s_t element_matrix_tmp71 = element_matrix_tmp9*(basis0_grad0*element_matrix_tmp62 + basis0_grad2*element_matrix_tmp66);
    const s_t element_matrix_tmp72 = element_matrix_tmp9*(basis1_grad0*element_matrix_tmp62 + basis1_grad2*element_matrix_tmp66);
    const s_t element_matrix_tmp73 = element_matrix_tmp9*(basis2_grad0*element_matrix_tmp62 + basis2_grad2*element_matrix_tmp66);
    const s_t element_matrix_tmp74 = element_matrix_tmp9*(basis3_grad0*element_matrix_tmp62 + basis3_grad2*element_matrix_tmp66);
    const s_t element_matrix_tmp75 = pow_2(basis3_grad1);
    const s_t element_matrix_tmp76 = element_matrix_tmp2*element_matrix_tmp75;
    const s_t element_matrix_tmp77 = pow_2(basis3_grad2);
    const s_t element_matrix_tmp78 = element_matrix_tmp2*element_matrix_tmp77;
    const s_t element_matrix_tmp79 = pow_2(basis3_grad0);
    const s_t element_matrix_tmp80 = basis3_grad1*element_matrix_tmp2;
    const s_t element_matrix_tmp81 = basis3_grad0*element_matrix_tmp22;
    const s_t element_matrix_tmp82 = element_matrix_tmp9*(basis0_grad0*element_matrix_tmp80 + basis0_grad1*element_matrix_tmp81);
    const s_t element_matrix_tmp83 = element_matrix_tmp9*(basis1_grad0*element_matrix_tmp80 + basis1_grad1*element_matrix_tmp81);
    const s_t element_matrix_tmp84 = element_matrix_tmp9*(basis2_grad0*element_matrix_tmp80 + basis2_grad1*element_matrix_tmp81);
    const s_t element_matrix_tmp85 = element_matrix_tmp9*(basis3_grad0*element_matrix_tmp80 + basis3_grad1*element_matrix_tmp81);
    const s_t element_matrix_tmp86 = basis3_grad2*element_matrix_tmp2;
    const s_t element_matrix_tmp87 = element_matrix_tmp9*(basis0_grad0*element_matrix_tmp86 + basis0_grad2*element_matrix_tmp81);
    const s_t element_matrix_tmp88 = element_matrix_tmp9*(basis1_grad0*element_matrix_tmp86 + basis1_grad2*element_matrix_tmp81);
    const s_t element_matrix_tmp89 = element_matrix_tmp9*(basis2_grad0*element_matrix_tmp86 + basis2_grad2*element_matrix_tmp81);
    const s_t element_matrix_tmp90 = element_matrix_tmp9*(basis3_grad0*element_matrix_tmp86 + basis3_grad2*element_matrix_tmp81);
    const s_t element_matrix_tmp91 = element_matrix_tmp2*element_matrix_tmp6;
    const s_t element_matrix_tmp92 = basis0_grad0*element_matrix_tmp2;
    const s_t element_matrix_tmp93 = basis1_grad0*element_matrix_tmp92;
    const s_t element_matrix_tmp94 = basis0_grad1*element_matrix_tmp8;
    const s_t element_matrix_tmp95 = element_matrix_tmp9*(basis1_grad1*element_matrix_tmp94 + element_matrix_tmp13 + element_matrix_tmp93);
    const s_t element_matrix_tmp96 = basis2_grad0*element_matrix_tmp92;
    const s_t element_matrix_tmp97 = element_matrix_tmp9*(basis2_grad1*element_matrix_tmp94 + element_matrix_tmp17 + element_matrix_tmp96);
    const s_t element_matrix_tmp98 = basis3_grad0*element_matrix_tmp92;
    const s_t element_matrix_tmp99 = element_matrix_tmp9*(basis3_grad1*element_matrix_tmp94 + element_matrix_tmp20 + element_matrix_tmp98);
    const s_t element_matrix_tmp100 = basis0_grad1*element_matrix_tmp22;
    const s_t element_matrix_tmp101 = element_matrix_tmp9*(basis0_grad2*element_matrix_tmp10 + basis0_grad2*element_matrix_tmp100);
    const s_t element_matrix_tmp102 = element_matrix_tmp9*(basis1_grad1*element_matrix_tmp12 + basis1_grad2*element_matrix_tmp100);
    const s_t element_matrix_tmp103 = element_matrix_tmp9*(basis2_grad1*element_matrix_tmp12 + basis2_grad2*element_matrix_tmp100);
    const s_t element_matrix_tmp104 = element_matrix_tmp9*(basis3_grad1*element_matrix_tmp12 + basis3_grad2*element_matrix_tmp100);
    const s_t element_matrix_tmp105 = element_matrix_tmp2*element_matrix_tmp36;
    const s_t element_matrix_tmp106 = basis1_grad0*element_matrix_tmp2;
    const s_t element_matrix_tmp107 = basis2_grad0*element_matrix_tmp106;
    const s_t element_matrix_tmp108 = basis1_grad1*element_matrix_tmp8;
    const s_t element_matrix_tmp109 = element_matrix_tmp9*(basis2_grad1*element_matrix_tmp108 + element_matrix_tmp107 + element_matrix_tmp40);
    const s_t element_matrix_tmp110 = basis3_grad0*element_matrix_tmp106;
    const s_t element_matrix_tmp111 = element_matrix_tmp9*(basis3_grad1*element_matrix_tmp108 + element_matrix_tmp110 + element_matrix_tmp44);
    const s_t element_matrix_tmp112 = basis1_grad1*element_matrix_tmp22;
    const s_t element_matrix_tmp113 = element_matrix_tmp9*(basis0_grad2*element_matrix_tmp112 + basis1_grad2*element_matrix_tmp10);
    const s_t element_matrix_tmp114 = element_matrix_tmp9*(basis1_grad2*element_matrix_tmp112 + basis1_grad2*element_matrix_tmp37);
    const s_t element_matrix_tmp115 = element_matrix_tmp9*(basis2_grad1*element_matrix_tmp39 + basis2_grad2*element_matrix_tmp112);
    const s_t element_matrix_tmp116 = element_matrix_tmp9*(basis3_grad1*element_matrix_tmp39 + basis3_grad2*element_matrix_tmp112);
    const s_t element_matrix_tmp117 = element_matrix_tmp2*element_matrix_tmp59;
    const s_t element_matrix_tmp118 = element_matrix_tmp2*element_matrix_tmp64;
    const s_t element_matrix_tmp119 = element_matrix_tmp9*(basis2_grad1*basis3_grad1*element_matrix_tmp8 + element_matrix_tmp118 + element_matrix_tmp63);
    const s_t element_matrix_tmp120 = basis2_grad1*element_matrix_tmp22;
    const s_t element_matrix_tmp121 = element_matrix_tmp9*(basis0_grad2*element_matrix_tmp120 + basis2_grad2*element_matrix_tmp10);
    const s_t element_matrix_tmp122 = element_matrix_tmp9*(basis1_grad2*element_matrix_tmp120 + basis2_grad2*element_matrix_tmp37);
    const s_t element_matrix_tmp123 = element_matrix_tmp9*(basis2_grad2*element_matrix_tmp120 + basis2_grad2*element_matrix_tmp60);
    const s_t element_matrix_tmp124 = element_matrix_tmp9*(basis3_grad1*element_matrix_tmp62 + basis3_grad2*element_matrix_tmp120);
    const s_t element_matrix_tmp125 = element_matrix_tmp2*element_matrix_tmp79;
    const s_t element_matrix_tmp126 = basis3_grad1*element_matrix_tmp22;
    const s_t element_matrix_tmp127 = element_matrix_tmp9*(basis0_grad2*element_matrix_tmp126 + basis3_grad2*element_matrix_tmp10);
    const s_t element_matrix_tmp128 = element_matrix_tmp9*(basis1_grad2*element_matrix_tmp126 + basis3_grad2*element_matrix_tmp37);
    const s_t element_matrix_tmp129 = element_matrix_tmp9*(basis2_grad2*element_matrix_tmp126 + basis3_grad2*element_matrix_tmp60);
    const s_t element_matrix_tmp130 = element_matrix_tmp9*(basis3_grad2*element_matrix_tmp126 + basis3_grad2*element_matrix_tmp80);
    const s_t element_matrix_tmp131 = basis0_grad2*element_matrix_tmp8;
    const s_t element_matrix_tmp132 = element_matrix_tmp9*(basis1_grad2*element_matrix_tmp131 + element_matrix_tmp11 + element_matrix_tmp93);
    const s_t element_matrix_tmp133 = element_matrix_tmp9*(basis2_grad2*element_matrix_tmp131 + element_matrix_tmp16 + element_matrix_tmp96);
    const s_t element_matrix_tmp134 = element_matrix_tmp9*(basis3_grad2*element_matrix_tmp131 + element_matrix_tmp19 + element_matrix_tmp98);
    const s_t element_matrix_tmp135 = basis1_grad2*element_matrix_tmp8;
    const s_t element_matrix_tmp136 = element_matrix_tmp9*(basis2_grad2*element_matrix_tmp135 + element_matrix_tmp107 + element_matrix_tmp38);
    const s_t element_matrix_tmp137 = element_matrix_tmp9*(basis3_grad2*element_matrix_tmp135 + element_matrix_tmp110 + element_matrix_tmp43);
    const s_t element_matrix_tmp138 = element_matrix_tmp9*(basis2_grad2*basis3_grad2*element_matrix_tmp8 + element_matrix_tmp118 + element_matrix_tmp61);
    element_matrix[0] = element_matrix_tmp9*(element_matrix_tmp3 + element_matrix_tmp5 + element_matrix_tmp6*element_matrix_tmp8);
    element_matrix[12] = element_matrix_tmp15;
    element_matrix[24] = element_matrix_tmp18;
    element_matrix[36] = element_matrix_tmp21;
    element_matrix[48] = element_matrix_tmp24;
    element_matrix[60] = element_matrix_tmp25;
    element_matrix[72] = element_matrix_tmp26;
    element_matrix[84] = element_matrix_tmp27;
    element_matrix[96] = element_matrix_tmp28;
    element_matrix[108] = element_matrix_tmp29;
    element_matrix[120] = element_matrix_tmp30;
    element_matrix[132] = element_matrix_tmp31;
    element_matrix[1] = element_matrix_tmp15;
    element_matrix[13] = element_matrix_tmp9*(element_matrix_tmp33 + element_matrix_tmp35 + element_matrix_tmp36*element_matrix_tmp8);
    element_matrix[25] = element_matrix_tmp42;
    element_matrix[37] = element_matrix_tmp45;
    element_matrix[49] = element_matrix_tmp47;
    element_matrix[61] = element_matrix_tmp48;
    element_matrix[73] = element_matrix_tmp49;
    element_matrix[85] = element_matrix_tmp50;
    element_matrix[97] = element_matrix_tmp51;
    element_matrix[109] = element_matrix_tmp52;
    element_matrix[121] = element_matrix_tmp53;
    element_matrix[133] = element_matrix_tmp54;
    element_matrix[2] = element_matrix_tmp18;
    element_matrix[14] = element_matrix_tmp42;
    element_matrix[26] = element_matrix_tmp9*(element_matrix_tmp56 + element_matrix_tmp58 + element_matrix_tmp59*element_matrix_tmp8);
    element_matrix[38] = element_matrix_tmp65;
    element_matrix[50] = element_matrix_tmp67;
    element_matrix[62] = element_matrix_tmp68;
    element_matrix[74] = element_matrix_tmp69;
    element_matrix[86] = element_matrix_tmp70;
    element_matrix[98] = element_matrix_tmp71;
    element_matrix[110] = element_matrix_tmp72;
    element_matrix[122] = element_matrix_tmp73;
    element_matrix[134] = element_matrix_tmp74;
    element_matrix[3] = element_matrix_tmp21;
    element_matrix[15] = element_matrix_tmp45;
    element_matrix[27] = element_matrix_tmp65;
    element_matrix[39] = element_matrix_tmp9*(element_matrix_tmp76 + element_matrix_tmp78 + element_matrix_tmp79*element_matrix_tmp8);
    element_matrix[51] = element_matrix_tmp82;
    element_matrix[63] = element_matrix_tmp83;
    element_matrix[75] = element_matrix_tmp84;
    element_matrix[87] = element_matrix_tmp85;
    element_matrix[99] = element_matrix_tmp87;
    element_matrix[111] = element_matrix_tmp88;
    element_matrix[123] = element_matrix_tmp89;
    element_matrix[135] = element_matrix_tmp90;
    element_matrix[4] = element_matrix_tmp24;
    element_matrix[16] = element_matrix_tmp47;
    element_matrix[28] = element_matrix_tmp67;
    element_matrix[40] = element_matrix_tmp82;
    element_matrix[52] = element_matrix_tmp9*(element_matrix_tmp0*element_matrix_tmp8 + element_matrix_tmp5 + element_matrix_tmp91);
    element_matrix[64] = element_matrix_tmp95;
    element_matrix[76] = element_matrix_tmp97;
    element_matrix[88] = element_matrix_tmp99;
    element_matrix[100] = element_matrix_tmp101;
    element_matrix[112] = element_matrix_tmp102;
    element_matrix[124] = element_matrix_tmp103;
    element_matrix[136] = element_matrix_tmp104;
    element_matrix[5] = element_matrix_tmp25;
    element_matrix[17] = element_matrix_tmp48;
    element_matrix[29] = element_matrix_tmp68;
    element_matrix[41] = element_matrix_tmp83;
    element_matrix[53] = element_matrix_tmp95;
    element_matrix[65] = element_matrix_tmp9*(element_matrix_tmp105 + element_matrix_tmp32*element_matrix_tmp8 + element_matrix_tmp35);
    element_matrix[77] = element_matrix_tmp109;
    element_matrix[89] = element_matrix_tmp111;
    element_matrix[101] = element_matrix_tmp113;
    element_matrix[113] = element_matrix_tmp114;
    element_matrix[125] = element_matrix_tmp115;
    element_matrix[137] = element_matrix_tmp116;
    element_matrix[6] = element_matrix_tmp26;
    element_matrix[18] = element_matrix_tmp49;
    element_matrix[30] = element_matrix_tmp69;
    element_matrix[42] = element_matrix_tmp84;
    element_matrix[54] = element_matrix_tmp97;
    element_matrix[66] = element_matrix_tmp109;
    element_matrix[78] = element_matrix_tmp9*(element_matrix_tmp117 + element_matrix_tmp55*element_matrix_tmp8 + element_matrix_tmp58);
    element_matrix[90] = element_matrix_tmp119;
    element_matrix[102] = element_matrix_tmp121;
    element_matrix[114] = element_matrix_tmp122;
    element_matrix[126] = element_matrix_tmp123;
    element_matrix[138] = element_matrix_tmp124;
    element_matrix[7] = element_matrix_tmp27;
    element_matrix[19] = element_matrix_tmp50;
    element_matrix[31] = element_matrix_tmp70;
    element_matrix[43] = element_matrix_tmp85;
    element_matrix[55] = element_matrix_tmp99;
    element_matrix[67] = element_matrix_tmp111;
    element_matrix[79] = element_matrix_tmp119;
    element_matrix[91] = element_matrix_tmp9*(element_matrix_tmp125 + element_matrix_tmp75*element_matrix_tmp8 + element_matrix_tmp78);
    element_matrix[103] = element_matrix_tmp127;
    element_matrix[115] = element_matrix_tmp128;
    element_matrix[127] = element_matrix_tmp129;
    element_matrix[139] = element_matrix_tmp130;
    element_matrix[8] = element_matrix_tmp28;
    element_matrix[20] = element_matrix_tmp51;
    element_matrix[32] = element_matrix_tmp71;
    element_matrix[44] = element_matrix_tmp87;
    element_matrix[56] = element_matrix_tmp101;
    element_matrix[68] = element_matrix_tmp113;
    element_matrix[80] = element_matrix_tmp121;
    element_matrix[92] = element_matrix_tmp127;
    element_matrix[104] = element_matrix_tmp9*(element_matrix_tmp3 + element_matrix_tmp4*element_matrix_tmp8 + element_matrix_tmp91);
    element_matrix[116] = element_matrix_tmp132;
    element_matrix[128] = element_matrix_tmp133;
    element_matrix[140] = element_matrix_tmp134;
    element_matrix[9] = element_matrix_tmp29;
    element_matrix[21] = element_matrix_tmp52;
    element_matrix[33] = element_matrix_tmp72;
    element_matrix[45] = element_matrix_tmp88;
    element_matrix[57] = element_matrix_tmp102;
    element_matrix[69] = element_matrix_tmp114;
    element_matrix[81] = element_matrix_tmp122;
    element_matrix[93] = element_matrix_tmp128;
    element_matrix[105] = element_matrix_tmp132;
    element_matrix[117] = element_matrix_tmp9*(element_matrix_tmp105 + element_matrix_tmp33 + element_matrix_tmp34*element_matrix_tmp8);
    element_matrix[129] = element_matrix_tmp136;
    element_matrix[141] = element_matrix_tmp137;
    element_matrix[10] = element_matrix_tmp30;
    element_matrix[22] = element_matrix_tmp53;
    element_matrix[34] = element_matrix_tmp73;
    element_matrix[46] = element_matrix_tmp89;
    element_matrix[58] = element_matrix_tmp103;
    element_matrix[70] = element_matrix_tmp115;
    element_matrix[82] = element_matrix_tmp123;
    element_matrix[94] = element_matrix_tmp129;
    element_matrix[106] = element_matrix_tmp133;
    element_matrix[118] = element_matrix_tmp136;
    element_matrix[130] = element_matrix_tmp9*(element_matrix_tmp117 + element_matrix_tmp56 + element_matrix_tmp57*element_matrix_tmp8);
    element_matrix[142] = element_matrix_tmp138;
    element_matrix[11] = element_matrix_tmp31;
    element_matrix[23] = element_matrix_tmp54;
    element_matrix[35] = element_matrix_tmp74;
    element_matrix[47] = element_matrix_tmp90;
    element_matrix[59] = element_matrix_tmp104;
    element_matrix[71] = element_matrix_tmp116;
    element_matrix[83] = element_matrix_tmp124;
    element_matrix[95] = element_matrix_tmp130;
    element_matrix[107] = element_matrix_tmp134;
    element_matrix[119] = element_matrix_tmp137;
    element_matrix[131] = element_matrix_tmp138;
    element_matrix[143] = element_matrix_tmp9*(element_matrix_tmp125 + element_matrix_tmp76 + element_matrix_tmp77*element_matrix_tmp8);
  }
}

template <typename s_t, int NQ, int NS>
__host__ __device__ __forceinline__ void linear_kelvin_voigt_total_d3_simplex_hessian_block(
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t *const RSTR element_matrix
) {
  for (int entry = 0; entry < 3 * NS * 3 * NS; ++entry) {
    element_matrix[entry] = s_t(0);
  }
  s_t grad_coeff0_0_values;
  s_t grad_coeff0_1_values;
  s_t grad_coeff0_2_values;
  s_t grad_coeff1_0_values;
  s_t grad_coeff1_1_values;
  s_t grad_coeff1_2_values;
  s_t grad_coeff2_0_values;
  s_t grad_coeff2_1_values;
  s_t grad_coeff2_2_values;
  s_t tangent[81];
  for (int q = 0; q < NQ; ++q) {
    {
      const s_t tangent_tmp0 = eta_s*u_dt_shift;
      const s_t tangent_tmp1 = eta_b*u_dt_shift + lmbda;
      const s_t tangent_tmp2 = s_t(2)*mu + ((s_t(4) / s_t(3)))*tangent_tmp0 + tangent_tmp1;
      const s_t tangent_tmp3 = -(s_t(2) / s_t(3))*tangent_tmp0 + tangent_tmp1;
      const s_t tangent_tmp4 = mu + tangent_tmp0;
      tangent[0] = tangent_tmp2;
      tangent[1] = 0;
      tangent[2] = 0;
      tangent[3] = 0;
      tangent[4] = tangent_tmp3;
      tangent[5] = 0;
      tangent[6] = 0;
      tangent[7] = 0;
      tangent[8] = tangent_tmp3;
      tangent[9] = 0;
      tangent[10] = tangent_tmp4;
      tangent[11] = 0;
      tangent[12] = tangent_tmp4;
      tangent[13] = 0;
      tangent[14] = 0;
      tangent[15] = 0;
      tangent[16] = 0;
      tangent[17] = 0;
      tangent[18] = 0;
      tangent[19] = 0;
      tangent[20] = tangent_tmp4;
      tangent[21] = 0;
      tangent[22] = 0;
      tangent[23] = 0;
      tangent[24] = tangent_tmp4;
      tangent[25] = 0;
      tangent[26] = 0;
      tangent[27] = 0;
      tangent[28] = tangent_tmp4;
      tangent[29] = 0;
      tangent[30] = tangent_tmp4;
      tangent[31] = 0;
      tangent[32] = 0;
      tangent[33] = 0;
      tangent[34] = 0;
      tangent[35] = 0;
      tangent[36] = tangent_tmp3;
      tangent[37] = 0;
      tangent[38] = 0;
      tangent[39] = 0;
      tangent[40] = tangent_tmp2;
      tangent[41] = 0;
      tangent[42] = 0;
      tangent[43] = 0;
      tangent[44] = tangent_tmp3;
      tangent[45] = 0;
      tangent[46] = 0;
      tangent[47] = 0;
      tangent[48] = 0;
      tangent[49] = 0;
      tangent[50] = tangent_tmp4;
      tangent[51] = 0;
      tangent[52] = tangent_tmp4;
      tangent[53] = 0;
      tangent[54] = 0;
      tangent[55] = 0;
      tangent[56] = tangent_tmp4;
      tangent[57] = 0;
      tangent[58] = 0;
      tangent[59] = 0;
      tangent[60] = tangent_tmp4;
      tangent[61] = 0;
      tangent[62] = 0;
      tangent[63] = 0;
      tangent[64] = 0;
      tangent[65] = 0;
      tangent[66] = 0;
      tangent[67] = 0;
      tangent[68] = tangent_tmp4;
      tangent[69] = 0;
      tangent[70] = tangent_tmp4;
      tangent[71] = 0;
      tangent[72] = tangent_tmp3;
      tangent[73] = 0;
      tangent[74] = 0;
      tangent[75] = 0;
      tangent[76] = tangent_tmp3;
      tangent[77] = 0;
      tangent[78] = 0;
      tangent[79] = 0;
      tangent[80] = tangent_tmp2;
    }
    for (int trial = 0; trial < NS; ++trial) {
      const s_t trial_grad_ref0 = grad_ref_x[q * NS + trial];
      const s_t trial_grad_ref1 = grad_ref_y[q * NS + trial];
      const s_t trial_grad_ref2 = grad_ref_z[q * NS + trial];
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
        const s_t trial_grad0 = (trial_grad_ref0 * adj0 + trial_grad_ref1 * adj3 + trial_grad_ref2 * adj6) / det;
        const s_t trial_grad1 = (trial_grad_ref0 * adj1 + trial_grad_ref1 * adj4 + trial_grad_ref2 * adj7) / det;
        const s_t trial_grad2 = (trial_grad_ref0 * adj2 + trial_grad_ref1 * adj5 + trial_grad_ref2 * adj8) / det;
        grad_coeff0_0_values = trial_grad0 * tangent[0] + trial_grad1 * tangent[9] + trial_grad2 * tangent[18];
        grad_coeff0_1_values = trial_grad0 * tangent[1] + trial_grad1 * tangent[10] + trial_grad2 * tangent[19];
        grad_coeff0_2_values = trial_grad0 * tangent[2] + trial_grad1 * tangent[11] + trial_grad2 * tangent[20];
        grad_coeff1_0_values = trial_grad0 * tangent[3] + trial_grad1 * tangent[12] + trial_grad2 * tangent[21];
        grad_coeff1_1_values = trial_grad0 * tangent[4] + trial_grad1 * tangent[13] + trial_grad2 * tangent[22];
        grad_coeff1_2_values = trial_grad0 * tangent[5] + trial_grad1 * tangent[14] + trial_grad2 * tangent[23];
        grad_coeff2_0_values = trial_grad0 * tangent[6] + trial_grad1 * tangent[15] + trial_grad2 * tangent[24];
        grad_coeff2_1_values = trial_grad0 * tangent[7] + trial_grad1 * tangent[16] + trial_grad2 * tangent[25];
        grad_coeff2_2_values = trial_grad0 * tangent[8] + trial_grad1 * tangent[17] + trial_grad2 * tangent[26];
      }
      for (int test = 0; test < NS; ++test) {
        const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
        const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
        const s_t test_grad_ref2 = grad_ref_z[q * NS + test];
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
          element_matrix[(0 * NS + test) * 3 * NS + 0 * NS + trial] += q_weight[q] * det * (grad_coeff0_0_values * test_grad0 + grad_coeff0_1_values * test_grad1 + grad_coeff0_2_values * test_grad2);
          element_matrix[(1 * NS + test) * 3 * NS + 0 * NS + trial] += q_weight[q] * det * (grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1 + grad_coeff1_2_values * test_grad2);
          element_matrix[(2 * NS + test) * 3 * NS + 0 * NS + trial] += q_weight[q] * det * (grad_coeff2_0_values * test_grad0 + grad_coeff2_1_values * test_grad1 + grad_coeff2_2_values * test_grad2);
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
        const s_t trial_grad0 = (trial_grad_ref0 * adj0 + trial_grad_ref1 * adj3 + trial_grad_ref2 * adj6) / det;
        const s_t trial_grad1 = (trial_grad_ref0 * adj1 + trial_grad_ref1 * adj4 + trial_grad_ref2 * adj7) / det;
        const s_t trial_grad2 = (trial_grad_ref0 * adj2 + trial_grad_ref1 * adj5 + trial_grad_ref2 * adj8) / det;
        grad_coeff0_0_values = trial_grad0 * tangent[27] + trial_grad1 * tangent[36] + trial_grad2 * tangent[45];
        grad_coeff0_1_values = trial_grad0 * tangent[28] + trial_grad1 * tangent[37] + trial_grad2 * tangent[46];
        grad_coeff0_2_values = trial_grad0 * tangent[29] + trial_grad1 * tangent[38] + trial_grad2 * tangent[47];
        grad_coeff1_0_values = trial_grad0 * tangent[30] + trial_grad1 * tangent[39] + trial_grad2 * tangent[48];
        grad_coeff1_1_values = trial_grad0 * tangent[31] + trial_grad1 * tangent[40] + trial_grad2 * tangent[49];
        grad_coeff1_2_values = trial_grad0 * tangent[32] + trial_grad1 * tangent[41] + trial_grad2 * tangent[50];
        grad_coeff2_0_values = trial_grad0 * tangent[33] + trial_grad1 * tangent[42] + trial_grad2 * tangent[51];
        grad_coeff2_1_values = trial_grad0 * tangent[34] + trial_grad1 * tangent[43] + trial_grad2 * tangent[52];
        grad_coeff2_2_values = trial_grad0 * tangent[35] + trial_grad1 * tangent[44] + trial_grad2 * tangent[53];
      }
      for (int test = 0; test < NS; ++test) {
        const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
        const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
        const s_t test_grad_ref2 = grad_ref_z[q * NS + test];
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
          element_matrix[(0 * NS + test) * 3 * NS + 1 * NS + trial] += q_weight[q] * det * (grad_coeff0_0_values * test_grad0 + grad_coeff0_1_values * test_grad1 + grad_coeff0_2_values * test_grad2);
          element_matrix[(1 * NS + test) * 3 * NS + 1 * NS + trial] += q_weight[q] * det * (grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1 + grad_coeff1_2_values * test_grad2);
          element_matrix[(2 * NS + test) * 3 * NS + 1 * NS + trial] += q_weight[q] * det * (grad_coeff2_0_values * test_grad0 + grad_coeff2_1_values * test_grad1 + grad_coeff2_2_values * test_grad2);
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
        const s_t trial_grad0 = (trial_grad_ref0 * adj0 + trial_grad_ref1 * adj3 + trial_grad_ref2 * adj6) / det;
        const s_t trial_grad1 = (trial_grad_ref0 * adj1 + trial_grad_ref1 * adj4 + trial_grad_ref2 * adj7) / det;
        const s_t trial_grad2 = (trial_grad_ref0 * adj2 + trial_grad_ref1 * adj5 + trial_grad_ref2 * adj8) / det;
        grad_coeff0_0_values = trial_grad0 * tangent[54] + trial_grad1 * tangent[63] + trial_grad2 * tangent[72];
        grad_coeff0_1_values = trial_grad0 * tangent[55] + trial_grad1 * tangent[64] + trial_grad2 * tangent[73];
        grad_coeff0_2_values = trial_grad0 * tangent[56] + trial_grad1 * tangent[65] + trial_grad2 * tangent[74];
        grad_coeff1_0_values = trial_grad0 * tangent[57] + trial_grad1 * tangent[66] + trial_grad2 * tangent[75];
        grad_coeff1_1_values = trial_grad0 * tangent[58] + trial_grad1 * tangent[67] + trial_grad2 * tangent[76];
        grad_coeff1_2_values = trial_grad0 * tangent[59] + trial_grad1 * tangent[68] + trial_grad2 * tangent[77];
        grad_coeff2_0_values = trial_grad0 * tangent[60] + trial_grad1 * tangent[69] + trial_grad2 * tangent[78];
        grad_coeff2_1_values = trial_grad0 * tangent[61] + trial_grad1 * tangent[70] + trial_grad2 * tangent[79];
        grad_coeff2_2_values = trial_grad0 * tangent[62] + trial_grad1 * tangent[71] + trial_grad2 * tangent[80];
      }
      for (int test = 0; test < NS; ++test) {
        const s_t test_grad_ref0 = grad_ref_x[q * NS + test];
        const s_t test_grad_ref1 = grad_ref_y[q * NS + test];
        const s_t test_grad_ref2 = grad_ref_z[q * NS + test];
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
          element_matrix[(0 * NS + test) * 3 * NS + 2 * NS + trial] += q_weight[q] * det * (grad_coeff0_0_values * test_grad0 + grad_coeff0_1_values * test_grad1 + grad_coeff0_2_values * test_grad2);
          element_matrix[(1 * NS + test) * 3 * NS + 2 * NS + trial] += q_weight[q] * det * (grad_coeff1_0_values * test_grad0 + grad_coeff1_1_values * test_grad1 + grad_coeff1_2_values * test_grad2);
          element_matrix[(2 * NS + test) * 3 * NS + 2 * NS + trial] += q_weight[q] * det * (grad_coeff2_0_values * test_grad0 + grad_coeff2_1_values * test_grad1 + grad_coeff2_2_values * test_grad2);
        }
      }
    }
  }
}

} // namespace codegen
} // namespace sfem

#endif
