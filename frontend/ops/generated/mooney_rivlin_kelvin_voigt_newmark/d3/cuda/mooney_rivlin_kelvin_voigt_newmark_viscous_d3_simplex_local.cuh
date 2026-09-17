#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_VISCOUS_D3_SIMPLEX_LOCAL_HPP
#define MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_VISCOUS_D3_SIMPLEX_LOCAL_HPP

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
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[3 * NS],
    const s_t *const RSTR previous[3 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t newmark_velocity_alpha,
    s_t *const RSTR output[3 * NS]
) {
  static constexpr int NC = 3;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_grad_0_ref_values[VS];
    s_t u0_grad_1_ref_values[VS];
    s_t u0_grad_2_ref_values[VS];
    s_t u0_old_grad_0_ref_values[VS];
    s_t u0_old_grad_1_ref_values[VS];
    s_t u0_old_grad_2_ref_values[VS];
    s_t u1_grad_0_ref_values[VS];
    s_t u1_grad_1_ref_values[VS];
    s_t u1_grad_2_ref_values[VS];
    s_t u1_old_grad_0_ref_values[VS];
    s_t u1_old_grad_1_ref_values[VS];
    s_t u1_old_grad_2_ref_values[VS];
    s_t u2_grad_0_ref_values[VS];
    s_t u2_grad_1_ref_values[VS];
    s_t u2_grad_2_ref_values[VS];
    s_t u2_old_grad_0_ref_values[VS];
    s_t u2_old_grad_1_ref_values[VS];
    s_t u2_old_grad_2_ref_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff0_2_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    s_t grad_coeff1_2_values[VS];
    s_t grad_coeff2_0_values[VS];
    s_t grad_coeff2_1_values[VS];
    s_t grad_coeff2_2_values[VS];
    {
      u0_grad_0_ref_values[0] = s_t(0);
      u0_grad_1_ref_values[0] = s_t(0);
      u0_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC][0];
        u0_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u0_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u0_old_grad_0_ref_values[0] = s_t(0);
      u0_old_grad_1_ref_values[0] = s_t(0);
      u0_old_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC][0];
        u0_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u0_old_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u1_grad_0_ref_values[0] = s_t(0);
      u1_grad_1_ref_values[0] = s_t(0);
      u1_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 1][0];
        u1_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u1_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u1_old_grad_0_ref_values[0] = s_t(0);
      u1_old_grad_1_ref_values[0] = s_t(0);
      u1_old_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC + 1][0];
        u1_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u1_old_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u2_grad_0_ref_values[0] = s_t(0);
      u2_grad_1_ref_values[0] = s_t(0);
      u2_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 2][0];
        u2_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u2_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u2_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u2_old_grad_0_ref_values[0] = s_t(0);
      u2_old_grad_1_ref_values[0] = s_t(0);
      u2_old_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC + 2][0];
        u2_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u2_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u2_old_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
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
      const s_t u0_grad_0_ref = u0_grad_0_ref_values[0];
      const s_t u0_grad_1_ref = u0_grad_1_ref_values[0];
      const s_t u0_grad_2_ref = u0_grad_2_ref_values[0];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
      const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
      const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[0];
      const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[0];
      const s_t u0_old_grad_2_ref = u0_old_grad_2_ref_values[0];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
      const s_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
      const s_t u1_grad_0_ref = u1_grad_0_ref_values[0];
      const s_t u1_grad_1_ref = u1_grad_1_ref_values[0];
      const s_t u1_grad_2_ref = u1_grad_2_ref_values[0];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
      const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
      const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[0];
      const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[0];
      const s_t u1_old_grad_2_ref = u1_old_grad_2_ref_values[0];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
      const s_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
      const s_t u2_grad_0_ref = u2_grad_0_ref_values[0];
      const s_t u2_grad_1_ref = u2_grad_1_ref_values[0];
      const s_t u2_grad_2_ref = u2_grad_2_ref_values[0];
      const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
      const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
      const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
      const s_t u2_old_grad_0_ref = u2_old_grad_0_ref_values[0];
      const s_t u2_old_grad_1_ref = u2_old_grad_1_ref_values[0];
      const s_t u2_old_grad_2_ref = u2_old_grad_2_ref_values[0];
      const s_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
      const s_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
      const s_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = u0_grad_0*u1_grad_1;
      const s_t residual_tmp1 = u0_grad_1*u2_grad_0;
      const s_t residual_tmp2 = u0_grad_2*u2_grad_1;
      const s_t residual_tmp3 = u1_grad_2*u2_grad_1;
      const s_t residual_tmp4 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp5 = u0_grad_2*u2_grad_0;
      const s_t residual_tmp6 = u2_grad_2 + s_t(1);
      const s_t residual_tmp7 = -residual_tmp5 + residual_tmp6 + u0_grad_0*u2_grad_2 + u0_grad_0;
      const s_t residual_tmp8 = -residual_tmp3 + u1_grad_1*u2_grad_2 + u1_grad_1;
      const s_t residual_tmp9 = residual_tmp0 - residual_tmp4;
      const s_t residual_tmp10 = pow_m1(residual_tmp0*u2_grad_2 + residual_tmp1*u1_grad_2 + residual_tmp2*u1_grad_0 - residual_tmp3*u0_grad_0 - residual_tmp4*u2_grad_2 - residual_tmp5*u1_grad_1 + residual_tmp7 + residual_tmp8 + residual_tmp9);
      const s_t residual_tmp11 = -u0_grad_1*u1_grad_2 + u0_grad_2*u1_grad_1 + u0_grad_2;
      const s_t residual_tmp12 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
      const s_t residual_tmp13 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
      const s_t residual_tmp14 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
      const s_t residual_tmp15 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
      const s_t residual_tmp16 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
      const s_t residual_tmp17 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
      const s_t residual_tmp18 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
      const s_t residual_tmp19 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
      const s_t residual_tmp20 = residual_tmp9 + u0_grad_0 + u1_grad_1 + s_t(1);
      const s_t residual_tmp21 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
      const s_t residual_tmp22 = residual_tmp6 + residual_tmp8;
      const s_t residual_tmp23 = residual_tmp11*residual_tmp12 + residual_tmp13*residual_tmp14 + residual_tmp15*residual_tmp16 + residual_tmp17*residual_tmp18 - residual_tmp19*residual_tmp20 - residual_tmp21*residual_tmp22;
      const s_t residual_tmp24 = -residual_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
      const s_t residual_tmp25 = -residual_tmp1 + u0_grad_0*u2_grad_1 + u2_grad_1;
      const s_t residual_tmp26 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
      const s_t residual_tmp27 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
      const s_t residual_tmp28 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
      const s_t residual_tmp29 = residual_tmp12*residual_tmp24 - residual_tmp13*residual_tmp7 + residual_tmp16*residual_tmp26 + residual_tmp18*residual_tmp27 + residual_tmp19*residual_tmp25 - residual_tmp22*residual_tmp28;
      const s_t residual_tmp30 = residual_tmp24*residual_tmp28;
      const s_t residual_tmp31 = residual_tmp25*residual_tmp27;
      const s_t residual_tmp32 = residual_tmp11*residual_tmp21;
      const s_t residual_tmp33 = residual_tmp14*residual_tmp15;
      const s_t residual_tmp34 = residual_tmp26*residual_tmp7;
      const s_t residual_tmp35 = -residual_tmp34;
      const s_t residual_tmp36 = residual_tmp17*residual_tmp20;
      const s_t residual_tmp37 = -residual_tmp36;
      const s_t residual_tmp38 = residual_tmp30 + residual_tmp31 + residual_tmp32 + residual_tmp33 + residual_tmp35 + residual_tmp37;
      const s_t residual_tmp39 = residual_tmp12*residual_tmp22;
      const s_t residual_tmp40 = residual_tmp13*residual_tmp16 + residual_tmp18*residual_tmp19 - residual_tmp39;
      const s_t residual_tmp41 = s_t(3)*eta_b*(residual_tmp38 + residual_tmp40);
      const s_t residual_tmp42 = s_t(2)*eta_s;
      const s_t residual_tmp43 = residual_tmp41 + residual_tmp42*(s_t(2)*residual_tmp13*residual_tmp16 + s_t(2)*residual_tmp18*residual_tmp19 - residual_tmp38 - s_t(2)*residual_tmp39);
      const s_t residual_tmp44 = residual_tmp11*residual_tmp28 + residual_tmp14*residual_tmp26 - residual_tmp15*residual_tmp7 + residual_tmp17*residual_tmp25 - residual_tmp20*residual_tmp27 + residual_tmp21*residual_tmp24;
      const s_t residual_tmp45 = residual_tmp41 + residual_tmp42*(s_t(2)*residual_tmp24*residual_tmp28 + s_t(2)*residual_tmp25*residual_tmp27 - residual_tmp32 - residual_tmp33 - s_t(2)*residual_tmp34 - residual_tmp37 - residual_tmp40);
      const s_t residual_tmp46 = residual_tmp41 + residual_tmp42*(s_t(2)*residual_tmp11*residual_tmp21 + s_t(2)*residual_tmp14*residual_tmp15 - residual_tmp30 - residual_tmp31 - residual_tmp35 - s_t(2)*residual_tmp36 - residual_tmp40);
      const s_t grad_coeff0_0 = residual_tmp10*(eta_s*(residual_tmp11*residual_tmp23 + residual_tmp24*residual_tmp29) - (s_t(1) / s_t(3))*residual_tmp22*residual_tmp43);
      const s_t grad_coeff0_1 = residual_tmp10*(-eta_s*(-residual_tmp14*residual_tmp23 + residual_tmp29*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp16*residual_tmp43);
      const s_t grad_coeff0_2 = residual_tmp10*(-eta_s*(residual_tmp20*residual_tmp23 - residual_tmp25*residual_tmp29) + ((s_t(1) / s_t(3)))*residual_tmp18*residual_tmp43);
      const s_t grad_coeff1_0 = residual_tmp10*(-eta_s*(-residual_tmp11*residual_tmp44 + residual_tmp22*residual_tmp29) + ((s_t(1) / s_t(3)))*residual_tmp24*residual_tmp45);
      const s_t grad_coeff1_1 = residual_tmp10*(eta_s*(residual_tmp14*residual_tmp44 + residual_tmp16*residual_tmp29) - (s_t(1) / s_t(3))*residual_tmp45*residual_tmp7);
      const s_t grad_coeff1_2 = residual_tmp10*(-eta_s*(-residual_tmp18*residual_tmp29 + residual_tmp20*residual_tmp44) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp45);
      const s_t grad_coeff2_0 = residual_tmp10*(-eta_s*(residual_tmp22*residual_tmp23 - residual_tmp24*residual_tmp44) + ((s_t(1) / s_t(3)))*residual_tmp11*residual_tmp46);
      const s_t grad_coeff2_1 = residual_tmp10*(-eta_s*(-residual_tmp16*residual_tmp23 + residual_tmp44*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp14*residual_tmp46);
      const s_t grad_coeff2_2 = residual_tmp10*(eta_s*(residual_tmp18*residual_tmp23 + residual_tmp25*residual_tmp44) - (s_t(1) / s_t(3))*residual_tmp20*residual_tmp46);
      grad_coeff0_0_values[0] = grad_coeff0_0;
      grad_coeff0_1_values[0] = grad_coeff0_1;
      grad_coeff0_2_values[0] = grad_coeff0_2;
      grad_coeff1_0_values[0] = grad_coeff1_0;
      grad_coeff1_1_values[0] = grad_coeff1_1;
      grad_coeff1_2_values[0] = grad_coeff1_2;
      grad_coeff2_0_values[0] = grad_coeff2_0;
      grad_coeff2_1_values[0] = grad_coeff2_1;
      grad_coeff2_2_values[0] = grad_coeff2_2;
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
        const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj3 + grad_ref_z[q * NS + test] * adj6) / det;
        const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj4 + grad_ref_z[q * NS + test] * adj7) / det;
        const s_t test_grad2 = (grad_ref_x[q * NS + test] * adj2 + grad_ref_y[q * NS + test] * adj5 + grad_ref_z[q * NS + test] * adj8) / det;
        output[test * NC][0] += q_weight[q] * det * (grad_coeff0_0_values[0] * test_grad0 + grad_coeff0_1_values[0] * test_grad1 + grad_coeff0_2_values[0] * test_grad2);
        output[test * NC + 1][0] += q_weight[q] * det * (grad_coeff1_0_values[0] * test_grad0 + grad_coeff1_1_values[0] * test_grad1 + grad_coeff1_2_values[0] * test_grad2);
        output[test * NC + 2][0] += q_weight[q] * det * (grad_coeff2_0_values[0] * test_grad0 + grad_coeff2_1_values[0] * test_grad1 + grad_coeff2_2_values[0] * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t current[3 * NS][VS],
    const s_t previous[3 * NS][VS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t newmark_velocity_alpha,
    s_t output[3 * NS][VS]
) {
  static constexpr int NC = 3;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_grad_0_ref_values[VS];
    s_t u0_grad_1_ref_values[VS];
    s_t u0_grad_2_ref_values[VS];
    s_t u0_old_grad_0_ref_values[VS];
    s_t u0_old_grad_1_ref_values[VS];
    s_t u0_old_grad_2_ref_values[VS];
    s_t u1_grad_0_ref_values[VS];
    s_t u1_grad_1_ref_values[VS];
    s_t u1_grad_2_ref_values[VS];
    s_t u1_old_grad_0_ref_values[VS];
    s_t u1_old_grad_1_ref_values[VS];
    s_t u1_old_grad_2_ref_values[VS];
    s_t u2_grad_0_ref_values[VS];
    s_t u2_grad_1_ref_values[VS];
    s_t u2_grad_2_ref_values[VS];
    s_t u2_old_grad_0_ref_values[VS];
    s_t u2_old_grad_1_ref_values[VS];
    s_t u2_old_grad_2_ref_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff0_2_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    s_t grad_coeff1_2_values[VS];
    s_t grad_coeff2_0_values[VS];
    s_t grad_coeff2_1_values[VS];
    s_t grad_coeff2_2_values[VS];
    {
      u0_grad_0_ref_values[0] = s_t(0);
      u0_grad_1_ref_values[0] = s_t(0);
      u0_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC][0];
        u0_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u0_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u0_old_grad_0_ref_values[0] = s_t(0);
      u0_old_grad_1_ref_values[0] = s_t(0);
      u0_old_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC][0];
        u0_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u0_old_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u1_grad_0_ref_values[0] = s_t(0);
      u1_grad_1_ref_values[0] = s_t(0);
      u1_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 1][0];
        u1_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u1_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u1_old_grad_0_ref_values[0] = s_t(0);
      u1_old_grad_1_ref_values[0] = s_t(0);
      u1_old_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC + 1][0];
        u1_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u1_old_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u2_grad_0_ref_values[0] = s_t(0);
      u2_grad_1_ref_values[0] = s_t(0);
      u2_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 2][0];
        u2_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u2_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u2_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u2_old_grad_0_ref_values[0] = s_t(0);
      u2_old_grad_1_ref_values[0] = s_t(0);
      u2_old_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC + 2][0];
        u2_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u2_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u2_old_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
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
      const s_t u0_grad_0_ref = u0_grad_0_ref_values[0];
      const s_t u0_grad_1_ref = u0_grad_1_ref_values[0];
      const s_t u0_grad_2_ref = u0_grad_2_ref_values[0];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
      const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
      const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[0];
      const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[0];
      const s_t u0_old_grad_2_ref = u0_old_grad_2_ref_values[0];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
      const s_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
      const s_t u1_grad_0_ref = u1_grad_0_ref_values[0];
      const s_t u1_grad_1_ref = u1_grad_1_ref_values[0];
      const s_t u1_grad_2_ref = u1_grad_2_ref_values[0];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
      const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
      const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[0];
      const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[0];
      const s_t u1_old_grad_2_ref = u1_old_grad_2_ref_values[0];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
      const s_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
      const s_t u2_grad_0_ref = u2_grad_0_ref_values[0];
      const s_t u2_grad_1_ref = u2_grad_1_ref_values[0];
      const s_t u2_grad_2_ref = u2_grad_2_ref_values[0];
      const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
      const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
      const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
      const s_t u2_old_grad_0_ref = u2_old_grad_0_ref_values[0];
      const s_t u2_old_grad_1_ref = u2_old_grad_1_ref_values[0];
      const s_t u2_old_grad_2_ref = u2_old_grad_2_ref_values[0];
      const s_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
      const s_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
      const s_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = u0_grad_0*u1_grad_1;
      const s_t residual_tmp1 = u0_grad_1*u2_grad_0;
      const s_t residual_tmp2 = u0_grad_2*u2_grad_1;
      const s_t residual_tmp3 = u1_grad_2*u2_grad_1;
      const s_t residual_tmp4 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp5 = u0_grad_2*u2_grad_0;
      const s_t residual_tmp6 = u2_grad_2 + s_t(1);
      const s_t residual_tmp7 = -residual_tmp5 + residual_tmp6 + u0_grad_0*u2_grad_2 + u0_grad_0;
      const s_t residual_tmp8 = -residual_tmp3 + u1_grad_1*u2_grad_2 + u1_grad_1;
      const s_t residual_tmp9 = residual_tmp0 - residual_tmp4;
      const s_t residual_tmp10 = pow_m1(residual_tmp0*u2_grad_2 + residual_tmp1*u1_grad_2 + residual_tmp2*u1_grad_0 - residual_tmp3*u0_grad_0 - residual_tmp4*u2_grad_2 - residual_tmp5*u1_grad_1 + residual_tmp7 + residual_tmp8 + residual_tmp9);
      const s_t residual_tmp11 = -u0_grad_1*u1_grad_2 + u0_grad_2*u1_grad_1 + u0_grad_2;
      const s_t residual_tmp12 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
      const s_t residual_tmp13 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
      const s_t residual_tmp14 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
      const s_t residual_tmp15 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
      const s_t residual_tmp16 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
      const s_t residual_tmp17 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
      const s_t residual_tmp18 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
      const s_t residual_tmp19 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
      const s_t residual_tmp20 = residual_tmp9 + u0_grad_0 + u1_grad_1 + s_t(1);
      const s_t residual_tmp21 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
      const s_t residual_tmp22 = residual_tmp6 + residual_tmp8;
      const s_t residual_tmp23 = residual_tmp11*residual_tmp12 + residual_tmp13*residual_tmp14 + residual_tmp15*residual_tmp16 + residual_tmp17*residual_tmp18 - residual_tmp19*residual_tmp20 - residual_tmp21*residual_tmp22;
      const s_t residual_tmp24 = -residual_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
      const s_t residual_tmp25 = -residual_tmp1 + u0_grad_0*u2_grad_1 + u2_grad_1;
      const s_t residual_tmp26 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
      const s_t residual_tmp27 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
      const s_t residual_tmp28 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
      const s_t residual_tmp29 = residual_tmp12*residual_tmp24 - residual_tmp13*residual_tmp7 + residual_tmp16*residual_tmp26 + residual_tmp18*residual_tmp27 + residual_tmp19*residual_tmp25 - residual_tmp22*residual_tmp28;
      const s_t residual_tmp30 = residual_tmp24*residual_tmp28;
      const s_t residual_tmp31 = residual_tmp25*residual_tmp27;
      const s_t residual_tmp32 = residual_tmp11*residual_tmp21;
      const s_t residual_tmp33 = residual_tmp14*residual_tmp15;
      const s_t residual_tmp34 = residual_tmp26*residual_tmp7;
      const s_t residual_tmp35 = -residual_tmp34;
      const s_t residual_tmp36 = residual_tmp17*residual_tmp20;
      const s_t residual_tmp37 = -residual_tmp36;
      const s_t residual_tmp38 = residual_tmp30 + residual_tmp31 + residual_tmp32 + residual_tmp33 + residual_tmp35 + residual_tmp37;
      const s_t residual_tmp39 = residual_tmp12*residual_tmp22;
      const s_t residual_tmp40 = residual_tmp13*residual_tmp16 + residual_tmp18*residual_tmp19 - residual_tmp39;
      const s_t residual_tmp41 = s_t(3)*eta_b*(residual_tmp38 + residual_tmp40);
      const s_t residual_tmp42 = s_t(2)*eta_s;
      const s_t residual_tmp43 = residual_tmp41 + residual_tmp42*(s_t(2)*residual_tmp13*residual_tmp16 + s_t(2)*residual_tmp18*residual_tmp19 - residual_tmp38 - s_t(2)*residual_tmp39);
      const s_t residual_tmp44 = residual_tmp11*residual_tmp28 + residual_tmp14*residual_tmp26 - residual_tmp15*residual_tmp7 + residual_tmp17*residual_tmp25 - residual_tmp20*residual_tmp27 + residual_tmp21*residual_tmp24;
      const s_t residual_tmp45 = residual_tmp41 + residual_tmp42*(s_t(2)*residual_tmp24*residual_tmp28 + s_t(2)*residual_tmp25*residual_tmp27 - residual_tmp32 - residual_tmp33 - s_t(2)*residual_tmp34 - residual_tmp37 - residual_tmp40);
      const s_t residual_tmp46 = residual_tmp41 + residual_tmp42*(s_t(2)*residual_tmp11*residual_tmp21 + s_t(2)*residual_tmp14*residual_tmp15 - residual_tmp30 - residual_tmp31 - residual_tmp35 - s_t(2)*residual_tmp36 - residual_tmp40);
      const s_t grad_coeff0_0 = residual_tmp10*(eta_s*(residual_tmp11*residual_tmp23 + residual_tmp24*residual_tmp29) - (s_t(1) / s_t(3))*residual_tmp22*residual_tmp43);
      const s_t grad_coeff0_1 = residual_tmp10*(-eta_s*(-residual_tmp14*residual_tmp23 + residual_tmp29*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp16*residual_tmp43);
      const s_t grad_coeff0_2 = residual_tmp10*(-eta_s*(residual_tmp20*residual_tmp23 - residual_tmp25*residual_tmp29) + ((s_t(1) / s_t(3)))*residual_tmp18*residual_tmp43);
      const s_t grad_coeff1_0 = residual_tmp10*(-eta_s*(-residual_tmp11*residual_tmp44 + residual_tmp22*residual_tmp29) + ((s_t(1) / s_t(3)))*residual_tmp24*residual_tmp45);
      const s_t grad_coeff1_1 = residual_tmp10*(eta_s*(residual_tmp14*residual_tmp44 + residual_tmp16*residual_tmp29) - (s_t(1) / s_t(3))*residual_tmp45*residual_tmp7);
      const s_t grad_coeff1_2 = residual_tmp10*(-eta_s*(-residual_tmp18*residual_tmp29 + residual_tmp20*residual_tmp44) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp45);
      const s_t grad_coeff2_0 = residual_tmp10*(-eta_s*(residual_tmp22*residual_tmp23 - residual_tmp24*residual_tmp44) + ((s_t(1) / s_t(3)))*residual_tmp11*residual_tmp46);
      const s_t grad_coeff2_1 = residual_tmp10*(-eta_s*(-residual_tmp16*residual_tmp23 + residual_tmp44*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp14*residual_tmp46);
      const s_t grad_coeff2_2 = residual_tmp10*(eta_s*(residual_tmp18*residual_tmp23 + residual_tmp25*residual_tmp44) - (s_t(1) / s_t(3))*residual_tmp20*residual_tmp46);
      grad_coeff0_0_values[0] = grad_coeff0_0;
      grad_coeff0_1_values[0] = grad_coeff0_1;
      grad_coeff0_2_values[0] = grad_coeff0_2;
      grad_coeff1_0_values[0] = grad_coeff1_0;
      grad_coeff1_1_values[0] = grad_coeff1_1;
      grad_coeff1_2_values[0] = grad_coeff1_2;
      grad_coeff2_0_values[0] = grad_coeff2_0;
      grad_coeff2_1_values[0] = grad_coeff2_1;
      grad_coeff2_2_values[0] = grad_coeff2_2;
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
        const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj3 + grad_ref_z[q * NS + test] * adj6) / det;
        const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj4 + grad_ref_z[q * NS + test] * adj7) / det;
        const s_t test_grad2 = (grad_ref_x[q * NS + test] * adj2 + grad_ref_y[q * NS + test] * adj5 + grad_ref_z[q * NS + test] * adj8) / det;
        output[test * NC][0] += q_weight[q] * det * (grad_coeff0_0_values[0] * test_grad0 + grad_coeff0_1_values[0] * test_grad1 + grad_coeff0_2_values[0] * test_grad2);
        output[test * NC + 1][0] += q_weight[q] * det * (grad_coeff1_0_values[0] * test_grad0 + grad_coeff1_1_values[0] * test_grad1 + grad_coeff1_2_values[0] * test_grad2);
        output[test * NC + 2][0] += q_weight[q] * det * (grad_coeff2_0_values[0] * test_grad0 + grad_coeff2_1_values[0] * test_grad1 + grad_coeff2_2_values[0] * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_tet4_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[3 * NS],
    const s_t *const RSTR previous[3 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t newmark_velocity_alpha,
    s_t *const RSTR output[3 * NS]
) {
  for (int q = 0; q < NQ; ++q) {
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
      const s_t residual_tmp0 = u0_grad_0*u1_grad_1;
      const s_t residual_tmp1 = u0_grad_1*u2_grad_0;
      const s_t residual_tmp2 = u0_grad_2*u2_grad_1;
      const s_t residual_tmp3 = u1_grad_2*u2_grad_1;
      const s_t residual_tmp4 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp5 = u0_grad_2*u2_grad_0;
      const s_t residual_tmp6 = u2_grad_2 + s_t(1);
      const s_t residual_tmp7 = -residual_tmp5 + residual_tmp6 + u0_grad_0*u2_grad_2 + u0_grad_0;
      const s_t residual_tmp8 = -residual_tmp3 + u1_grad_1*u2_grad_2 + u1_grad_1;
      const s_t residual_tmp9 = residual_tmp0 - residual_tmp4;
      const s_t residual_tmp10 = pow_m1(residual_tmp0*u2_grad_2 + residual_tmp1*u1_grad_2 + residual_tmp2*u1_grad_0 - residual_tmp3*u0_grad_0 - residual_tmp4*u2_grad_2 - residual_tmp5*u1_grad_1 + residual_tmp7 + residual_tmp8 + residual_tmp9);
      const s_t residual_tmp11 = -u0_grad_1*u1_grad_2 + u0_grad_2*u1_grad_1 + u0_grad_2;
      const s_t residual_tmp12 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
      const s_t residual_tmp13 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
      const s_t residual_tmp14 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
      const s_t residual_tmp15 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
      const s_t residual_tmp16 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
      const s_t residual_tmp17 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
      const s_t residual_tmp18 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
      const s_t residual_tmp19 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
      const s_t residual_tmp20 = residual_tmp9 + u0_grad_0 + u1_grad_1 + s_t(1);
      const s_t residual_tmp21 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
      const s_t residual_tmp22 = residual_tmp6 + residual_tmp8;
      const s_t residual_tmp23 = residual_tmp11*residual_tmp12 + residual_tmp13*residual_tmp14 + residual_tmp15*residual_tmp16 + residual_tmp17*residual_tmp18 - residual_tmp19*residual_tmp20 - residual_tmp21*residual_tmp22;
      const s_t residual_tmp24 = -residual_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
      const s_t residual_tmp25 = -residual_tmp1 + u0_grad_0*u2_grad_1 + u2_grad_1;
      const s_t residual_tmp26 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
      const s_t residual_tmp27 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
      const s_t residual_tmp28 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
      const s_t residual_tmp29 = residual_tmp12*residual_tmp24 - residual_tmp13*residual_tmp7 + residual_tmp16*residual_tmp26 + residual_tmp18*residual_tmp27 + residual_tmp19*residual_tmp25 - residual_tmp22*residual_tmp28;
      const s_t residual_tmp30 = residual_tmp24*residual_tmp28;
      const s_t residual_tmp31 = residual_tmp25*residual_tmp27;
      const s_t residual_tmp32 = residual_tmp11*residual_tmp21;
      const s_t residual_tmp33 = residual_tmp14*residual_tmp15;
      const s_t residual_tmp34 = residual_tmp26*residual_tmp7;
      const s_t residual_tmp35 = -residual_tmp34;
      const s_t residual_tmp36 = residual_tmp17*residual_tmp20;
      const s_t residual_tmp37 = -residual_tmp36;
      const s_t residual_tmp38 = residual_tmp30 + residual_tmp31 + residual_tmp32 + residual_tmp33 + residual_tmp35 + residual_tmp37;
      const s_t residual_tmp39 = residual_tmp12*residual_tmp22;
      const s_t residual_tmp40 = residual_tmp13*residual_tmp16 + residual_tmp18*residual_tmp19 - residual_tmp39;
      const s_t residual_tmp41 = s_t(3)*eta_b*(residual_tmp38 + residual_tmp40);
      const s_t residual_tmp42 = s_t(2)*eta_s;
      const s_t residual_tmp43 = residual_tmp41 + residual_tmp42*(s_t(2)*residual_tmp13*residual_tmp16 + s_t(2)*residual_tmp18*residual_tmp19 - residual_tmp38 - s_t(2)*residual_tmp39);
      const s_t residual_tmp44 = residual_tmp11*residual_tmp28 + residual_tmp14*residual_tmp26 - residual_tmp15*residual_tmp7 + residual_tmp17*residual_tmp25 - residual_tmp20*residual_tmp27 + residual_tmp21*residual_tmp24;
      const s_t residual_tmp45 = residual_tmp41 + residual_tmp42*(s_t(2)*residual_tmp24*residual_tmp28 + s_t(2)*residual_tmp25*residual_tmp27 - residual_tmp32 - residual_tmp33 - s_t(2)*residual_tmp34 - residual_tmp37 - residual_tmp40);
      const s_t residual_tmp46 = residual_tmp41 + residual_tmp42*(s_t(2)*residual_tmp11*residual_tmp21 + s_t(2)*residual_tmp14*residual_tmp15 - residual_tmp30 - residual_tmp31 - residual_tmp35 - s_t(2)*residual_tmp36 - residual_tmp40);
      const s_t grad_coeff0_0 = residual_tmp10*(eta_s*(residual_tmp11*residual_tmp23 + residual_tmp24*residual_tmp29) - (s_t(1) / s_t(3))*residual_tmp22*residual_tmp43);
      const s_t grad_coeff0_1 = residual_tmp10*(-eta_s*(-residual_tmp14*residual_tmp23 + residual_tmp29*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp16*residual_tmp43);
      const s_t grad_coeff0_2 = residual_tmp10*(-eta_s*(residual_tmp20*residual_tmp23 - residual_tmp25*residual_tmp29) + ((s_t(1) / s_t(3)))*residual_tmp18*residual_tmp43);
      const s_t grad_coeff1_0 = residual_tmp10*(-eta_s*(-residual_tmp11*residual_tmp44 + residual_tmp22*residual_tmp29) + ((s_t(1) / s_t(3)))*residual_tmp24*residual_tmp45);
      const s_t grad_coeff1_1 = residual_tmp10*(eta_s*(residual_tmp14*residual_tmp44 + residual_tmp16*residual_tmp29) - (s_t(1) / s_t(3))*residual_tmp45*residual_tmp7);
      const s_t grad_coeff1_2 = residual_tmp10*(-eta_s*(-residual_tmp18*residual_tmp29 + residual_tmp20*residual_tmp44) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp45);
      const s_t grad_coeff2_0 = residual_tmp10*(-eta_s*(residual_tmp22*residual_tmp23 - residual_tmp24*residual_tmp44) + ((s_t(1) / s_t(3)))*residual_tmp11*residual_tmp46);
      const s_t grad_coeff2_1 = residual_tmp10*(-eta_s*(-residual_tmp16*residual_tmp23 + residual_tmp44*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp14*residual_tmp46);
      const s_t grad_coeff2_2 = residual_tmp10*(eta_s*(residual_tmp18*residual_tmp23 + residual_tmp25*residual_tmp44) - (s_t(1) / s_t(3))*residual_tmp20*residual_tmp46);
      const s_t grad_coeff0_0_value = grad_coeff0_0;
      const s_t grad_coeff0_1_value = grad_coeff0_1;
      const s_t grad_coeff0_2_value = grad_coeff0_2;
      const s_t grad_coeff1_0_value = grad_coeff1_0;
      const s_t grad_coeff1_1_value = grad_coeff1_1;
      const s_t grad_coeff1_2_value = grad_coeff1_2;
      const s_t grad_coeff2_0_value = grad_coeff2_0;
      const s_t grad_coeff2_1_value = grad_coeff2_1;
      const s_t grad_coeff2_2_value = grad_coeff2_2;
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
      output[0][0] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1 + grad_coeff0_2_value * test0_grad2);
      output[1][0] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1 + grad_coeff1_2_value * test0_grad2);
      output[2][0] += q_weight[q] * det * (grad_coeff2_0_value * test0_grad0 + grad_coeff2_1_value * test0_grad1 + grad_coeff2_2_value * test0_grad2);
      output[3][0] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1 + grad_coeff0_2_value * test1_grad2);
      output[4][0] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1 + grad_coeff1_2_value * test1_grad2);
      output[5][0] += q_weight[q] * det * (grad_coeff2_0_value * test1_grad0 + grad_coeff2_1_value * test1_grad1 + grad_coeff2_2_value * test1_grad2);
      output[6][0] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1 + grad_coeff0_2_value * test2_grad2);
      output[7][0] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1 + grad_coeff1_2_value * test2_grad2);
      output[8][0] += q_weight[q] * det * (grad_coeff2_0_value * test2_grad0 + grad_coeff2_1_value * test2_grad1 + grad_coeff2_2_value * test2_grad2);
      output[9][0] += q_weight[q] * det * (grad_coeff0_0_value * test3_grad0 + grad_coeff0_1_value * test3_grad1 + grad_coeff0_2_value * test3_grad2);
      output[10][0] += q_weight[q] * det * (grad_coeff1_0_value * test3_grad0 + grad_coeff1_1_value * test3_grad1 + grad_coeff1_2_value * test3_grad2);
      output[11][0] += q_weight[q] * det * (grad_coeff2_0_value * test3_grad0 + grad_coeff2_1_value * test3_grad1 + grad_coeff2_2_value * test3_grad2);
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_tet4_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t current[3 * NS][VS],
    const s_t previous[3 * NS][VS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t newmark_velocity_alpha,
    s_t output[3 * NS][VS]
) {
  for (int q = 0; q < NQ; ++q) {
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
      const s_t residual_tmp0 = u0_grad_0*u1_grad_1;
      const s_t residual_tmp1 = u0_grad_1*u2_grad_0;
      const s_t residual_tmp2 = u0_grad_2*u2_grad_1;
      const s_t residual_tmp3 = u1_grad_2*u2_grad_1;
      const s_t residual_tmp4 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp5 = u0_grad_2*u2_grad_0;
      const s_t residual_tmp6 = u2_grad_2 + s_t(1);
      const s_t residual_tmp7 = -residual_tmp5 + residual_tmp6 + u0_grad_0*u2_grad_2 + u0_grad_0;
      const s_t residual_tmp8 = -residual_tmp3 + u1_grad_1*u2_grad_2 + u1_grad_1;
      const s_t residual_tmp9 = residual_tmp0 - residual_tmp4;
      const s_t residual_tmp10 = pow_m1(residual_tmp0*u2_grad_2 + residual_tmp1*u1_grad_2 + residual_tmp2*u1_grad_0 - residual_tmp3*u0_grad_0 - residual_tmp4*u2_grad_2 - residual_tmp5*u1_grad_1 + residual_tmp7 + residual_tmp8 + residual_tmp9);
      const s_t residual_tmp11 = -u0_grad_1*u1_grad_2 + u0_grad_2*u1_grad_1 + u0_grad_2;
      const s_t residual_tmp12 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
      const s_t residual_tmp13 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
      const s_t residual_tmp14 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
      const s_t residual_tmp15 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
      const s_t residual_tmp16 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
      const s_t residual_tmp17 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
      const s_t residual_tmp18 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
      const s_t residual_tmp19 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
      const s_t residual_tmp20 = residual_tmp9 + u0_grad_0 + u1_grad_1 + s_t(1);
      const s_t residual_tmp21 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
      const s_t residual_tmp22 = residual_tmp6 + residual_tmp8;
      const s_t residual_tmp23 = residual_tmp11*residual_tmp12 + residual_tmp13*residual_tmp14 + residual_tmp15*residual_tmp16 + residual_tmp17*residual_tmp18 - residual_tmp19*residual_tmp20 - residual_tmp21*residual_tmp22;
      const s_t residual_tmp24 = -residual_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
      const s_t residual_tmp25 = -residual_tmp1 + u0_grad_0*u2_grad_1 + u2_grad_1;
      const s_t residual_tmp26 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
      const s_t residual_tmp27 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
      const s_t residual_tmp28 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
      const s_t residual_tmp29 = residual_tmp12*residual_tmp24 - residual_tmp13*residual_tmp7 + residual_tmp16*residual_tmp26 + residual_tmp18*residual_tmp27 + residual_tmp19*residual_tmp25 - residual_tmp22*residual_tmp28;
      const s_t residual_tmp30 = residual_tmp24*residual_tmp28;
      const s_t residual_tmp31 = residual_tmp25*residual_tmp27;
      const s_t residual_tmp32 = residual_tmp11*residual_tmp21;
      const s_t residual_tmp33 = residual_tmp14*residual_tmp15;
      const s_t residual_tmp34 = residual_tmp26*residual_tmp7;
      const s_t residual_tmp35 = -residual_tmp34;
      const s_t residual_tmp36 = residual_tmp17*residual_tmp20;
      const s_t residual_tmp37 = -residual_tmp36;
      const s_t residual_tmp38 = residual_tmp30 + residual_tmp31 + residual_tmp32 + residual_tmp33 + residual_tmp35 + residual_tmp37;
      const s_t residual_tmp39 = residual_tmp12*residual_tmp22;
      const s_t residual_tmp40 = residual_tmp13*residual_tmp16 + residual_tmp18*residual_tmp19 - residual_tmp39;
      const s_t residual_tmp41 = s_t(3)*eta_b*(residual_tmp38 + residual_tmp40);
      const s_t residual_tmp42 = s_t(2)*eta_s;
      const s_t residual_tmp43 = residual_tmp41 + residual_tmp42*(s_t(2)*residual_tmp13*residual_tmp16 + s_t(2)*residual_tmp18*residual_tmp19 - residual_tmp38 - s_t(2)*residual_tmp39);
      const s_t residual_tmp44 = residual_tmp11*residual_tmp28 + residual_tmp14*residual_tmp26 - residual_tmp15*residual_tmp7 + residual_tmp17*residual_tmp25 - residual_tmp20*residual_tmp27 + residual_tmp21*residual_tmp24;
      const s_t residual_tmp45 = residual_tmp41 + residual_tmp42*(s_t(2)*residual_tmp24*residual_tmp28 + s_t(2)*residual_tmp25*residual_tmp27 - residual_tmp32 - residual_tmp33 - s_t(2)*residual_tmp34 - residual_tmp37 - residual_tmp40);
      const s_t residual_tmp46 = residual_tmp41 + residual_tmp42*(s_t(2)*residual_tmp11*residual_tmp21 + s_t(2)*residual_tmp14*residual_tmp15 - residual_tmp30 - residual_tmp31 - residual_tmp35 - s_t(2)*residual_tmp36 - residual_tmp40);
      const s_t grad_coeff0_0 = residual_tmp10*(eta_s*(residual_tmp11*residual_tmp23 + residual_tmp24*residual_tmp29) - (s_t(1) / s_t(3))*residual_tmp22*residual_tmp43);
      const s_t grad_coeff0_1 = residual_tmp10*(-eta_s*(-residual_tmp14*residual_tmp23 + residual_tmp29*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp16*residual_tmp43);
      const s_t grad_coeff0_2 = residual_tmp10*(-eta_s*(residual_tmp20*residual_tmp23 - residual_tmp25*residual_tmp29) + ((s_t(1) / s_t(3)))*residual_tmp18*residual_tmp43);
      const s_t grad_coeff1_0 = residual_tmp10*(-eta_s*(-residual_tmp11*residual_tmp44 + residual_tmp22*residual_tmp29) + ((s_t(1) / s_t(3)))*residual_tmp24*residual_tmp45);
      const s_t grad_coeff1_1 = residual_tmp10*(eta_s*(residual_tmp14*residual_tmp44 + residual_tmp16*residual_tmp29) - (s_t(1) / s_t(3))*residual_tmp45*residual_tmp7);
      const s_t grad_coeff1_2 = residual_tmp10*(-eta_s*(-residual_tmp18*residual_tmp29 + residual_tmp20*residual_tmp44) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp45);
      const s_t grad_coeff2_0 = residual_tmp10*(-eta_s*(residual_tmp22*residual_tmp23 - residual_tmp24*residual_tmp44) + ((s_t(1) / s_t(3)))*residual_tmp11*residual_tmp46);
      const s_t grad_coeff2_1 = residual_tmp10*(-eta_s*(-residual_tmp16*residual_tmp23 + residual_tmp44*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp14*residual_tmp46);
      const s_t grad_coeff2_2 = residual_tmp10*(eta_s*(residual_tmp18*residual_tmp23 + residual_tmp25*residual_tmp44) - (s_t(1) / s_t(3))*residual_tmp20*residual_tmp46);
      const s_t grad_coeff0_0_value = grad_coeff0_0;
      const s_t grad_coeff0_1_value = grad_coeff0_1;
      const s_t grad_coeff0_2_value = grad_coeff0_2;
      const s_t grad_coeff1_0_value = grad_coeff1_0;
      const s_t grad_coeff1_1_value = grad_coeff1_1;
      const s_t grad_coeff1_2_value = grad_coeff1_2;
      const s_t grad_coeff2_0_value = grad_coeff2_0;
      const s_t grad_coeff2_1_value = grad_coeff2_1;
      const s_t grad_coeff2_2_value = grad_coeff2_2;
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
      output[0][0] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1 + grad_coeff0_2_value * test0_grad2);
      output[1][0] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1 + grad_coeff1_2_value * test0_grad2);
      output[2][0] += q_weight[q] * det * (grad_coeff2_0_value * test0_grad0 + grad_coeff2_1_value * test0_grad1 + grad_coeff2_2_value * test0_grad2);
      output[3][0] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1 + grad_coeff0_2_value * test1_grad2);
      output[4][0] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1 + grad_coeff1_2_value * test1_grad2);
      output[5][0] += q_weight[q] * det * (grad_coeff2_0_value * test1_grad0 + grad_coeff2_1_value * test1_grad1 + grad_coeff2_2_value * test1_grad2);
      output[6][0] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1 + grad_coeff0_2_value * test2_grad2);
      output[7][0] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1 + grad_coeff1_2_value * test2_grad2);
      output[8][0] += q_weight[q] * det * (grad_coeff2_0_value * test2_grad0 + grad_coeff2_1_value * test2_grad1 + grad_coeff2_2_value * test2_grad2);
      output[9][0] += q_weight[q] * det * (grad_coeff0_0_value * test3_grad0 + grad_coeff0_1_value * test3_grad1 + grad_coeff0_2_value * test3_grad2);
      output[10][0] += q_weight[q] * det * (grad_coeff1_0_value * test3_grad0 + grad_coeff1_1_value * test3_grad1 + grad_coeff1_2_value * test3_grad2);
      output[11][0] += q_weight[q] * det * (grad_coeff2_0_value * test3_grad0 + grad_coeff2_1_value * test3_grad1 + grad_coeff2_2_value * test3_grad2);
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_jacobian_action_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[3 * NS],
    const s_t *const RSTR previous[3 * NS],
    const s_t *const RSTR direction[3 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t newmark_velocity_alpha,
    s_t *const RSTR output[3 * NS]
) {
  static constexpr int NC = 3;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_grad_0_ref_values[VS];
    s_t u0_grad_1_ref_values[VS];
    s_t u0_grad_2_ref_values[VS];
    s_t u0_old_grad_0_ref_values[VS];
    s_t u0_old_grad_1_ref_values[VS];
    s_t u0_old_grad_2_ref_values[VS];
    s_t u0_direction_grad_0_ref_values[VS];
    s_t u0_direction_grad_1_ref_values[VS];
    s_t u0_direction_grad_2_ref_values[VS];
    s_t u1_grad_0_ref_values[VS];
    s_t u1_grad_1_ref_values[VS];
    s_t u1_grad_2_ref_values[VS];
    s_t u1_old_grad_0_ref_values[VS];
    s_t u1_old_grad_1_ref_values[VS];
    s_t u1_old_grad_2_ref_values[VS];
    s_t u1_direction_grad_0_ref_values[VS];
    s_t u1_direction_grad_1_ref_values[VS];
    s_t u1_direction_grad_2_ref_values[VS];
    s_t u2_grad_0_ref_values[VS];
    s_t u2_grad_1_ref_values[VS];
    s_t u2_grad_2_ref_values[VS];
    s_t u2_old_grad_0_ref_values[VS];
    s_t u2_old_grad_1_ref_values[VS];
    s_t u2_old_grad_2_ref_values[VS];
    s_t u2_direction_grad_0_ref_values[VS];
    s_t u2_direction_grad_1_ref_values[VS];
    s_t u2_direction_grad_2_ref_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff0_2_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    s_t grad_coeff1_2_values[VS];
    s_t grad_coeff2_0_values[VS];
    s_t grad_coeff2_1_values[VS];
    s_t grad_coeff2_2_values[VS];
    {
      u0_grad_0_ref_values[0] = s_t(0);
      u0_grad_1_ref_values[0] = s_t(0);
      u0_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC][0];
        u0_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u0_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u0_old_grad_0_ref_values[0] = s_t(0);
      u0_old_grad_1_ref_values[0] = s_t(0);
      u0_old_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC][0];
        u0_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u0_old_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u0_direction_grad_0_ref_values[0] = s_t(0);
      u0_direction_grad_1_ref_values[0] = s_t(0);
      u0_direction_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = direction[trial * NC][0];
        u0_direction_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_direction_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u0_direction_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u1_grad_0_ref_values[0] = s_t(0);
      u1_grad_1_ref_values[0] = s_t(0);
      u1_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 1][0];
        u1_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u1_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u1_old_grad_0_ref_values[0] = s_t(0);
      u1_old_grad_1_ref_values[0] = s_t(0);
      u1_old_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC + 1][0];
        u1_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u1_old_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u1_direction_grad_0_ref_values[0] = s_t(0);
      u1_direction_grad_1_ref_values[0] = s_t(0);
      u1_direction_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = direction[trial * NC + 1][0];
        u1_direction_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_direction_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u1_direction_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u2_grad_0_ref_values[0] = s_t(0);
      u2_grad_1_ref_values[0] = s_t(0);
      u2_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 2][0];
        u2_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u2_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u2_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u2_old_grad_0_ref_values[0] = s_t(0);
      u2_old_grad_1_ref_values[0] = s_t(0);
      u2_old_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC + 2][0];
        u2_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u2_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u2_old_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u2_direction_grad_0_ref_values[0] = s_t(0);
      u2_direction_grad_1_ref_values[0] = s_t(0);
      u2_direction_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = direction[trial * NC + 2][0];
        u2_direction_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u2_direction_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u2_direction_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
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
      const s_t u0_grad_0_ref = u0_grad_0_ref_values[0];
      const s_t u0_grad_1_ref = u0_grad_1_ref_values[0];
      const s_t u0_grad_2_ref = u0_grad_2_ref_values[0];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
      const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
      const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[0];
      const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[0];
      const s_t u0_old_grad_2_ref = u0_old_grad_2_ref_values[0];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
      const s_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
      const s_t u0_direction_grad_0_ref = u0_direction_grad_0_ref_values[0];
      const s_t u0_direction_grad_1_ref = u0_direction_grad_1_ref_values[0];
      const s_t u0_direction_grad_2_ref = u0_direction_grad_2_ref_values[0];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
      const s_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
      const s_t u1_grad_0_ref = u1_grad_0_ref_values[0];
      const s_t u1_grad_1_ref = u1_grad_1_ref_values[0];
      const s_t u1_grad_2_ref = u1_grad_2_ref_values[0];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
      const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
      const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[0];
      const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[0];
      const s_t u1_old_grad_2_ref = u1_old_grad_2_ref_values[0];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
      const s_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
      const s_t u1_direction_grad_0_ref = u1_direction_grad_0_ref_values[0];
      const s_t u1_direction_grad_1_ref = u1_direction_grad_1_ref_values[0];
      const s_t u1_direction_grad_2_ref = u1_direction_grad_2_ref_values[0];
      const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
      const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
      const s_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
      const s_t u2_grad_0_ref = u2_grad_0_ref_values[0];
      const s_t u2_grad_1_ref = u2_grad_1_ref_values[0];
      const s_t u2_grad_2_ref = u2_grad_2_ref_values[0];
      const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
      const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
      const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
      const s_t u2_old_grad_0_ref = u2_old_grad_0_ref_values[0];
      const s_t u2_old_grad_1_ref = u2_old_grad_1_ref_values[0];
      const s_t u2_old_grad_2_ref = u2_old_grad_2_ref_values[0];
      const s_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
      const s_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
      const s_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
      const s_t u2_direction_grad_0_ref = u2_direction_grad_0_ref_values[0];
      const s_t u2_direction_grad_1_ref = u2_direction_grad_1_ref_values[0];
      const s_t u2_direction_grad_2_ref = u2_direction_grad_2_ref_values[0];
      const s_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
      const s_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
      const s_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = u0_grad_0*u1_grad_1;
      const s_t residual_tmp1 = u0_grad_1*u1_grad_2;
      const s_t residual_tmp2 = u0_grad_2*u2_grad_1;
      const s_t residual_tmp3 = u1_grad_2*u2_grad_1;
      const s_t residual_tmp4 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp5 = u0_grad_2*u2_grad_0;
      const s_t residual_tmp6 = u2_grad_2 + s_t(1);
      const s_t residual_tmp7 = -residual_tmp3 + residual_tmp6 + u1_grad_1*u2_grad_2 + u1_grad_1;
      const s_t residual_tmp8 = -residual_tmp5 + u0_grad_0*u2_grad_2 + u0_grad_0;
      const s_t residual_tmp9 = residual_tmp0 - residual_tmp4;
      const s_t residual_tmp10 = residual_tmp0*u2_grad_2 + residual_tmp1*u2_grad_0 + residual_tmp2*u1_grad_0 - residual_tmp3*u0_grad_0 - residual_tmp4*u2_grad_2 - residual_tmp5*u1_grad_1 + residual_tmp7 + residual_tmp8 + residual_tmp9;
      const s_t residual_tmp11 = pow_m1(residual_tmp10);
      const s_t residual_tmp12 = -residual_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
      const s_t residual_tmp13 = newmark_velocity_alpha*residual_tmp7;
      const s_t residual_tmp14 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
      const s_t residual_tmp15 = residual_tmp14*u2_grad_1;
      const s_t residual_tmp16 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
      const s_t residual_tmp17 = residual_tmp16*residual_tmp6;
      const s_t residual_tmp18 = residual_tmp15 - residual_tmp17;
      const s_t residual_tmp19 = -residual_tmp13 - residual_tmp18;
      const s_t residual_tmp20 = -residual_tmp1 + u0_grad_2*u1_grad_1 + u0_grad_2;
      const s_t residual_tmp21 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
      const s_t residual_tmp22 = residual_tmp21*residual_tmp6;
      const s_t residual_tmp23 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
      const s_t residual_tmp24 = residual_tmp23*u2_grad_1;
      const s_t residual_tmp25 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
      const s_t residual_tmp26 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
      const s_t residual_tmp27 = residual_tmp25*u0_grad_1 - residual_tmp26*u0_grad_2;
      const s_t residual_tmp28 = residual_tmp22 - residual_tmp24 + residual_tmp27;
      const s_t residual_tmp29 = newmark_velocity_alpha*residual_tmp12;
      const s_t residual_tmp30 = residual_tmp23*u0_grad_1;
      const s_t residual_tmp31 = residual_tmp21*u0_grad_2;
      const s_t residual_tmp32 = residual_tmp29 + residual_tmp30 - residual_tmp31;
      const s_t residual_tmp33 = residual_tmp26*residual_tmp6;
      const s_t residual_tmp34 = residual_tmp25*u2_grad_1;
      const s_t residual_tmp35 = residual_tmp33 - residual_tmp34;
      const s_t residual_tmp36 = s_t(3)*eta_b;
      const s_t residual_tmp37 = residual_tmp36*(residual_tmp32 + residual_tmp35);
      const s_t residual_tmp38 = s_t(2)*eta_s;
      const s_t residual_tmp39 = residual_tmp37 + residual_tmp38*(s_t(2)*residual_tmp26*residual_tmp6 - residual_tmp32 - s_t(2)*residual_tmp34);
      const s_t residual_tmp40 = ((s_t(1) / s_t(3)))*residual_tmp7;
      const s_t residual_tmp41 = pow_m2(residual_tmp10);
      const s_t residual_tmp42 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
      const s_t residual_tmp43 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
      const s_t residual_tmp44 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
      const s_t residual_tmp45 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
      const s_t residual_tmp46 = u1_grad_1 + s_t(1);
      const s_t residual_tmp47 = residual_tmp46 + residual_tmp9 + u0_grad_0;
      const s_t residual_tmp48 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
      const s_t residual_tmp49 = residual_tmp20*residual_tmp42 + residual_tmp21*residual_tmp44 + residual_tmp23*residual_tmp45 - residual_tmp25*residual_tmp47 + residual_tmp26*residual_tmp43 - residual_tmp48*residual_tmp7;
      const s_t residual_tmp50 = u0_grad_0*u2_grad_1 - u0_grad_1*u2_grad_0 + u2_grad_1;
      const s_t residual_tmp51 = residual_tmp6 + residual_tmp8;
      const s_t residual_tmp52 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
      const s_t residual_tmp53 = residual_tmp12*residual_tmp42 + residual_tmp14*residual_tmp45 + residual_tmp16*residual_tmp44 + residual_tmp25*residual_tmp50 - residual_tmp26*residual_tmp51 - residual_tmp52*residual_tmp7;
      const s_t residual_tmp54 = residual_tmp12*residual_tmp52;
      const s_t residual_tmp55 = residual_tmp14*residual_tmp50;
      const s_t residual_tmp56 = residual_tmp20*residual_tmp48;
      const s_t residual_tmp57 = residual_tmp21*residual_tmp43;
      const s_t residual_tmp58 = residual_tmp16*residual_tmp51;
      const s_t residual_tmp59 = -residual_tmp58;
      const s_t residual_tmp60 = residual_tmp23*residual_tmp47;
      const s_t residual_tmp61 = -residual_tmp60;
      const s_t residual_tmp62 = residual_tmp54 + residual_tmp55 + residual_tmp56 + residual_tmp57 + residual_tmp59 + residual_tmp61;
      const s_t residual_tmp63 = residual_tmp42*residual_tmp7;
      const s_t residual_tmp64 = residual_tmp25*residual_tmp45 + residual_tmp26*residual_tmp44 - residual_tmp63;
      const s_t residual_tmp65 = residual_tmp36*(residual_tmp62 + residual_tmp64);
      const s_t residual_tmp66 = residual_tmp38*(s_t(2)*residual_tmp25*residual_tmp45 + s_t(2)*residual_tmp26*residual_tmp44 - residual_tmp62 - s_t(2)*residual_tmp63) + residual_tmp65;
      const s_t residual_tmp67 = -residual_tmp66;
      const s_t residual_tmp68 = residual_tmp41*(eta_s*(residual_tmp12*residual_tmp53 + residual_tmp20*residual_tmp49) + residual_tmp40*residual_tmp67);
      const s_t residual_tmp69 = residual_tmp21*u1_grad_2;
      const s_t residual_tmp70 = residual_tmp23*residual_tmp46;
      const s_t residual_tmp71 = residual_tmp69 - residual_tmp70;
      const s_t residual_tmp72 = -residual_tmp13 - residual_tmp71;
      const s_t residual_tmp73 = -residual_tmp14*residual_tmp46 + residual_tmp16*u1_grad_2;
      const s_t residual_tmp74 = -residual_tmp27 - residual_tmp73;
      const s_t residual_tmp75 = newmark_velocity_alpha*residual_tmp20;
      const s_t residual_tmp76 = residual_tmp16*u0_grad_2;
      const s_t residual_tmp77 = residual_tmp14*u0_grad_1;
      const s_t residual_tmp78 = residual_tmp75 + residual_tmp76 - residual_tmp77;
      const s_t residual_tmp79 = residual_tmp25*residual_tmp46;
      const s_t residual_tmp80 = residual_tmp26*u1_grad_2;
      const s_t residual_tmp81 = residual_tmp79 - residual_tmp80;
      const s_t residual_tmp82 = residual_tmp36*(residual_tmp78 + residual_tmp81);
      const s_t residual_tmp83 = residual_tmp38*(s_t(2)*residual_tmp25*residual_tmp46 - residual_tmp78 - s_t(2)*residual_tmp80) + residual_tmp82;
      const s_t residual_tmp84 = -residual_tmp79 + residual_tmp80;
      const s_t residual_tmp85 = residual_tmp75 + residual_tmp84;
      const s_t residual_tmp86 = -residual_tmp33 + residual_tmp34;
      const s_t residual_tmp87 = residual_tmp29 + residual_tmp86;
      const s_t residual_tmp88 = residual_tmp13 - residual_tmp69 + residual_tmp70;
      const s_t residual_tmp89 = -residual_tmp15 + residual_tmp17;
      const s_t residual_tmp90 = residual_tmp36*(-residual_tmp88 - residual_tmp89);
      const s_t residual_tmp91 = residual_tmp38*(-s_t(2)*residual_tmp13 - residual_tmp18 - residual_tmp71) + residual_tmp90;
      const s_t residual_tmp92 = -residual_tmp7;
      const s_t residual_tmp93 = newmark_velocity_alpha*residual_tmp44;
      const s_t residual_tmp94 = residual_tmp23*u1_grad_0;
      const s_t residual_tmp95 = residual_tmp48*u1_grad_2;
      const s_t residual_tmp96 = residual_tmp93 + residual_tmp94 - residual_tmp95;
      const s_t residual_tmp97 = residual_tmp52*residual_tmp6;
      const s_t residual_tmp98 = residual_tmp14*u2_grad_0;
      const s_t residual_tmp99 = residual_tmp97 - residual_tmp98;
      const s_t residual_tmp100 = residual_tmp36*(residual_tmp96 + residual_tmp99);
      const s_t residual_tmp101 = -residual_tmp97 + residual_tmp98;
      const s_t residual_tmp102 = -residual_tmp94 + residual_tmp95;
      const s_t residual_tmp103 = residual_tmp100 + residual_tmp38*(residual_tmp101 + residual_tmp102 + s_t(2)*residual_tmp93);
      const s_t residual_tmp104 = newmark_velocity_alpha*residual_tmp43;
      const s_t residual_tmp105 = residual_tmp25*u1_grad_0;
      const s_t residual_tmp106 = residual_tmp42*u1_grad_2;
      const s_t residual_tmp107 = residual_tmp105 - residual_tmp106;
      const s_t residual_tmp108 = residual_tmp104 + residual_tmp107;
      const s_t residual_tmp109 = newmark_velocity_alpha*residual_tmp51;
      const s_t residual_tmp110 = residual_tmp25*u2_grad_0;
      const s_t residual_tmp111 = residual_tmp42*residual_tmp6;
      const s_t residual_tmp112 = residual_tmp110 - residual_tmp111;
      const s_t residual_tmp113 = -residual_tmp109 - residual_tmp112;
      const s_t residual_tmp114 = residual_tmp49*u1_grad_2;
      const s_t residual_tmp115 = -residual_tmp114;
      const s_t residual_tmp116 = residual_tmp53*residual_tmp6;
      const s_t residual_tmp117 = newmark_velocity_alpha*residual_tmp45;
      const s_t residual_tmp118 = residual_tmp46*residual_tmp48;
      const s_t residual_tmp119 = residual_tmp21*u1_grad_0;
      const s_t residual_tmp120 = residual_tmp117 + residual_tmp118 - residual_tmp119;
      const s_t residual_tmp121 = residual_tmp16*u2_grad_0;
      const s_t residual_tmp122 = residual_tmp52*u2_grad_1;
      const s_t residual_tmp123 = residual_tmp121 - residual_tmp122;
      const s_t residual_tmp124 = residual_tmp36*(residual_tmp120 + residual_tmp123);
      const s_t residual_tmp125 = -residual_tmp121 + residual_tmp122;
      const s_t residual_tmp126 = -residual_tmp118 + residual_tmp119;
      const s_t residual_tmp127 = residual_tmp124 + residual_tmp38*(s_t(2)*residual_tmp117 + residual_tmp125 + residual_tmp126);
      const s_t residual_tmp128 = newmark_velocity_alpha*residual_tmp50;
      const s_t residual_tmp129 = residual_tmp26*u2_grad_0;
      const s_t residual_tmp130 = residual_tmp42*u2_grad_1;
      const s_t residual_tmp131 = residual_tmp129 - residual_tmp130;
      const s_t residual_tmp132 = residual_tmp128 + residual_tmp131;
      const s_t residual_tmp133 = newmark_velocity_alpha*residual_tmp47;
      const s_t residual_tmp134 = residual_tmp26*u1_grad_0;
      const s_t residual_tmp135 = residual_tmp42*residual_tmp46;
      const s_t residual_tmp136 = residual_tmp134 - residual_tmp135;
      const s_t residual_tmp137 = -residual_tmp133 - residual_tmp136;
      const s_t residual_tmp138 = residual_tmp53*u2_grad_1;
      const s_t residual_tmp139 = -residual_tmp138;
      const s_t residual_tmp140 = residual_tmp46*residual_tmp49;
      const s_t residual_tmp141 = u0_grad_0 + s_t(1);
      const s_t residual_tmp142 = residual_tmp141*residual_tmp21;
      const s_t residual_tmp143 = residual_tmp48*u0_grad_1;
      const s_t residual_tmp144 = residual_tmp128 + residual_tmp142 - residual_tmp143;
      const s_t residual_tmp145 = -residual_tmp129 + residual_tmp130;
      const s_t residual_tmp146 = residual_tmp36*(residual_tmp144 + residual_tmp145);
      const s_t residual_tmp147 = residual_tmp146 + residual_tmp38*(-s_t(2)*residual_tmp129 - residual_tmp144 + s_t(2)*residual_tmp42*u2_grad_1);
      const s_t residual_tmp148 = residual_tmp117 + residual_tmp125;
      const s_t residual_tmp149 = residual_tmp21*u2_grad_0;
      const s_t residual_tmp150 = residual_tmp48*u2_grad_1;
      const s_t residual_tmp151 = -residual_tmp141*residual_tmp26 + residual_tmp42*u0_grad_1;
      const s_t residual_tmp152 = -residual_tmp149 + residual_tmp150 - residual_tmp151;
      const s_t residual_tmp153 = residual_tmp49*u0_grad_1;
      const s_t residual_tmp154 = ((s_t(1) / s_t(3)))*residual_tmp67;
      const s_t residual_tmp155 = residual_tmp14*residual_tmp141;
      const s_t residual_tmp156 = residual_tmp52*u0_grad_2;
      const s_t residual_tmp157 = residual_tmp104 + residual_tmp155 - residual_tmp156;
      const s_t residual_tmp158 = -residual_tmp105 + residual_tmp106;
      const s_t residual_tmp159 = residual_tmp36*(residual_tmp157 + residual_tmp158);
      const s_t residual_tmp160 = residual_tmp159 + residual_tmp38*(-s_t(2)*residual_tmp105 - residual_tmp157 + s_t(2)*residual_tmp42*u1_grad_2);
      const s_t residual_tmp161 = residual_tmp102 + residual_tmp93;
      const s_t residual_tmp162 = -residual_tmp141*residual_tmp25 + residual_tmp42*u0_grad_2;
      const s_t residual_tmp163 = residual_tmp14*u1_grad_0 - residual_tmp52*u1_grad_2;
      const s_t residual_tmp164 = -residual_tmp162 - residual_tmp163;
      const s_t residual_tmp165 = residual_tmp53*u0_grad_2;
      const s_t residual_tmp166 = -residual_tmp51;
      const s_t residual_tmp167 = residual_tmp141*residual_tmp23;
      const s_t residual_tmp168 = residual_tmp48*u0_grad_2;
      const s_t residual_tmp169 = residual_tmp109 + residual_tmp167 - residual_tmp168;
      const s_t residual_tmp170 = -residual_tmp110 + residual_tmp111;
      const s_t residual_tmp171 = residual_tmp36*(-residual_tmp169 - residual_tmp170);
      const s_t residual_tmp172 = residual_tmp171 + residual_tmp38*(s_t(2)*residual_tmp110 - s_t(2)*residual_tmp111 + residual_tmp169);
      const s_t residual_tmp173 = residual_tmp101 + residual_tmp93;
      const s_t residual_tmp174 = residual_tmp23*u2_grad_0;
      const s_t residual_tmp175 = residual_tmp48*residual_tmp6;
      const s_t residual_tmp176 = residual_tmp162 + residual_tmp174 - residual_tmp175;
      const s_t residual_tmp177 = residual_tmp49*u0_grad_2;
      const s_t residual_tmp178 = -residual_tmp47;
      const s_t residual_tmp179 = residual_tmp141*residual_tmp16;
      const s_t residual_tmp180 = residual_tmp52*u0_grad_1;
      const s_t residual_tmp181 = residual_tmp133 + residual_tmp179 - residual_tmp180;
      const s_t residual_tmp182 = -residual_tmp134 + residual_tmp135;
      const s_t residual_tmp183 = residual_tmp36*(-residual_tmp181 - residual_tmp182);
      const s_t residual_tmp184 = residual_tmp183 + residual_tmp38*(s_t(2)*residual_tmp134 - s_t(2)*residual_tmp135 + residual_tmp181);
      const s_t residual_tmp185 = residual_tmp117 + residual_tmp126;
      const s_t residual_tmp186 = residual_tmp16*u1_grad_0 - residual_tmp46*residual_tmp52;
      const s_t residual_tmp187 = residual_tmp151 + residual_tmp186;
      const s_t residual_tmp188 = residual_tmp53*u0_grad_1;
      const s_t residual_tmp189 = residual_tmp41*(-eta_s*(-residual_tmp43*residual_tmp49 + residual_tmp51*residual_tmp53) + ((s_t(1) / s_t(3)))*residual_tmp44*residual_tmp66);
      const s_t residual_tmp190 = residual_tmp49*u1_grad_0;
      const s_t residual_tmp191 = residual_tmp53*u2_grad_0;
      const s_t residual_tmp192 = -residual_tmp191;
      const s_t residual_tmp193 = residual_tmp190 + residual_tmp192;
      const s_t residual_tmp194 = -residual_tmp53*residual_tmp6;
      const s_t residual_tmp195 = residual_tmp141*residual_tmp49;
      const s_t residual_tmp196 = ((s_t(1) / s_t(3)))*residual_tmp66;
      const s_t residual_tmp197 = residual_tmp196*u2_grad_0;
      const s_t residual_tmp198 = ((s_t(1) / s_t(3)))*residual_tmp44;
      const s_t residual_tmp199 = residual_tmp141*residual_tmp53;
      const s_t residual_tmp200 = residual_tmp196*u1_grad_0;
      const s_t residual_tmp201 = residual_tmp41*(-eta_s*(residual_tmp47*residual_tmp49 - residual_tmp50*residual_tmp53) + ((s_t(1) / s_t(3)))*residual_tmp45*residual_tmp66);
      const s_t residual_tmp202 = -residual_tmp46*residual_tmp49;
      const s_t residual_tmp203 = ((s_t(1) / s_t(3)))*residual_tmp45;
      const s_t residual_tmp204 = -residual_tmp76 + residual_tmp77;
      const s_t residual_tmp205 = residual_tmp204 + residual_tmp75;
      const s_t residual_tmp206 = -residual_tmp30 + residual_tmp31;
      const s_t residual_tmp207 = residual_tmp37 + residual_tmp38*(residual_tmp206 + s_t(2)*residual_tmp29 + residual_tmp86);
      const s_t residual_tmp208 = residual_tmp12*residual_tmp48 - residual_tmp14*residual_tmp47 + residual_tmp16*residual_tmp43 + residual_tmp20*residual_tmp52 - residual_tmp21*residual_tmp51 + residual_tmp23*residual_tmp50;
      const s_t residual_tmp209 = residual_tmp38*(s_t(2)*residual_tmp12*residual_tmp52 + s_t(2)*residual_tmp14*residual_tmp50 - residual_tmp56 - residual_tmp57 - s_t(2)*residual_tmp58 - residual_tmp61 - residual_tmp64) + residual_tmp65;
      const s_t residual_tmp210 = residual_tmp41*(-eta_s*(-residual_tmp20*residual_tmp208 + residual_tmp53*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp209);
      const s_t residual_tmp211 = residual_tmp206 + residual_tmp29;
      const s_t residual_tmp212 = residual_tmp38*(s_t(2)*residual_tmp16*u0_grad_2 - residual_tmp75 - s_t(2)*residual_tmp77 - residual_tmp81) + residual_tmp82;
      const s_t residual_tmp213 = -residual_tmp22 + residual_tmp24 + residual_tmp73;
      const s_t residual_tmp214 = residual_tmp38*(s_t(2)*residual_tmp15 - s_t(2)*residual_tmp17 + residual_tmp88) + residual_tmp90;
      const s_t residual_tmp215 = -residual_tmp142 + residual_tmp143;
      const s_t residual_tmp216 = residual_tmp146 + residual_tmp38*(s_t(2)*residual_tmp128 + residual_tmp131 + residual_tmp215);
      const s_t residual_tmp217 = -residual_tmp179 + residual_tmp180;
      const s_t residual_tmp218 = -residual_tmp133 - residual_tmp217;
      const s_t residual_tmp219 = residual_tmp208*u0_grad_1;
      const s_t residual_tmp220 = residual_tmp139 + residual_tmp219;
      const s_t residual_tmp221 = -residual_tmp167 + residual_tmp168;
      const s_t residual_tmp222 = residual_tmp171 + residual_tmp38*(-s_t(2)*residual_tmp109 - residual_tmp112 - residual_tmp221);
      const s_t residual_tmp223 = -residual_tmp155 + residual_tmp156;
      const s_t residual_tmp224 = residual_tmp104 + residual_tmp223;
      const s_t residual_tmp225 = residual_tmp208*u0_grad_2;
      const s_t residual_tmp226 = residual_tmp124 + residual_tmp38*(-residual_tmp120 - s_t(2)*residual_tmp122 + s_t(2)*residual_tmp16*u2_grad_0);
      const s_t residual_tmp227 = residual_tmp149 - residual_tmp150 - residual_tmp186;
      const s_t residual_tmp228 = residual_tmp208*residual_tmp46;
      const s_t residual_tmp229 = ((s_t(1) / s_t(3)))*residual_tmp209;
      const s_t residual_tmp230 = residual_tmp229*u2_grad_1;
      const s_t residual_tmp231 = residual_tmp159 + residual_tmp38*(-residual_tmp104 + s_t(2)*residual_tmp14*residual_tmp141 - s_t(2)*residual_tmp156 - residual_tmp158);
      const s_t residual_tmp232 = -residual_tmp109 - residual_tmp221;
      const s_t residual_tmp233 = residual_tmp53*u1_grad_2;
      const s_t residual_tmp234 = residual_tmp100 + residual_tmp38*(s_t(2)*residual_tmp52*residual_tmp6 - residual_tmp96 - s_t(2)*residual_tmp98);
      const s_t residual_tmp235 = ((s_t(1) / s_t(3)))*residual_tmp12;
      const s_t residual_tmp236 = residual_tmp163 - residual_tmp174 + residual_tmp175;
      const s_t residual_tmp237 = residual_tmp208*u1_grad_2;
      const s_t residual_tmp238 = residual_tmp183 + residual_tmp38*(residual_tmp133 - s_t(2)*residual_tmp179 + s_t(2)*residual_tmp180 + residual_tmp182);
      const s_t residual_tmp239 = residual_tmp128 + residual_tmp215;
      const s_t residual_tmp240 = residual_tmp46*residual_tmp53;
      const s_t residual_tmp241 = residual_tmp229*u0_grad_1;
      const s_t residual_tmp242 = ((s_t(1) / s_t(3)))*residual_tmp51;
      const s_t residual_tmp243 = -(s_t(1) / s_t(3))*residual_tmp209;
      const s_t residual_tmp244 = residual_tmp41*(eta_s*(residual_tmp208*residual_tmp43 + residual_tmp44*residual_tmp53) + residual_tmp243*residual_tmp51);
      const s_t residual_tmp245 = residual_tmp141*residual_tmp208;
      const s_t residual_tmp246 = residual_tmp208*u1_grad_0;
      const s_t residual_tmp247 = residual_tmp53*u1_grad_0;
      const s_t residual_tmp248 = residual_tmp41*(-eta_s*(residual_tmp208*residual_tmp47 - residual_tmp45*residual_tmp53) + ((s_t(1) / s_t(3)))*residual_tmp209*residual_tmp50);
      const s_t residual_tmp249 = -residual_tmp141*residual_tmp208;
      const s_t residual_tmp250 = ((s_t(1) / s_t(3)))*residual_tmp50;
      const s_t residual_tmp251 = residual_tmp38*(residual_tmp204 + s_t(2)*residual_tmp75 + residual_tmp84) + residual_tmp82;
      const s_t residual_tmp252 = residual_tmp38*(s_t(2)*residual_tmp20*residual_tmp48 + s_t(2)*residual_tmp21*residual_tmp43 - residual_tmp54 - residual_tmp55 - residual_tmp59 - s_t(2)*residual_tmp60 - residual_tmp64) + residual_tmp65;
      const s_t residual_tmp253 = residual_tmp41*(-eta_s*(-residual_tmp12*residual_tmp208 + residual_tmp49*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp252);
      const s_t residual_tmp254 = residual_tmp37 + residual_tmp38*(s_t(2)*residual_tmp23*u0_grad_1 - residual_tmp29 - s_t(2)*residual_tmp31 - residual_tmp35);
      const s_t residual_tmp255 = residual_tmp38*(residual_tmp13 + s_t(2)*residual_tmp69 - s_t(2)*residual_tmp70 + residual_tmp89) + residual_tmp90;
      const s_t residual_tmp256 = residual_tmp159 + residual_tmp38*(s_t(2)*residual_tmp104 + residual_tmp107 + residual_tmp223);
      const s_t residual_tmp257 = residual_tmp115 + residual_tmp225;
      const s_t residual_tmp258 = residual_tmp183 + residual_tmp38*(-s_t(2)*residual_tmp133 - residual_tmp136 - residual_tmp217);
      const s_t residual_tmp259 = residual_tmp100 + residual_tmp38*(s_t(2)*residual_tmp23*u1_grad_0 - residual_tmp93 - s_t(2)*residual_tmp95 - residual_tmp99);
      const s_t residual_tmp260 = residual_tmp208*residual_tmp6;
      const s_t residual_tmp261 = ((s_t(1) / s_t(3)))*residual_tmp252;
      const s_t residual_tmp262 = residual_tmp261*u1_grad_2;
      const s_t residual_tmp263 = residual_tmp146 + residual_tmp38*(-residual_tmp128 + s_t(2)*residual_tmp141*residual_tmp21 - s_t(2)*residual_tmp143 - residual_tmp145);
      const s_t residual_tmp264 = residual_tmp49*u2_grad_1;
      const s_t residual_tmp265 = residual_tmp124 + residual_tmp38*(-residual_tmp117 - s_t(2)*residual_tmp119 - residual_tmp123 + s_t(2)*residual_tmp46*residual_tmp48);
      const s_t residual_tmp266 = ((s_t(1) / s_t(3)))*residual_tmp20;
      const s_t residual_tmp267 = residual_tmp208*u2_grad_1;
      const s_t residual_tmp268 = residual_tmp171 + residual_tmp38*(residual_tmp109 - s_t(2)*residual_tmp167 + s_t(2)*residual_tmp168 + residual_tmp170);
      const s_t residual_tmp269 = residual_tmp49*residual_tmp6;
      const s_t residual_tmp270 = residual_tmp261*u0_grad_2;
      const s_t residual_tmp271 = residual_tmp41*(-eta_s*(residual_tmp208*residual_tmp51 - residual_tmp44*residual_tmp49) + ((s_t(1) / s_t(3)))*residual_tmp252*residual_tmp43);
      const s_t residual_tmp272 = residual_tmp208*u2_grad_0;
      const s_t residual_tmp273 = ((s_t(1) / s_t(3)))*residual_tmp43;
      const s_t residual_tmp274 = residual_tmp49*u2_grad_0;
      const s_t residual_tmp275 = ((s_t(1) / s_t(3)))*residual_tmp47;
      const s_t residual_tmp276 = -(s_t(1) / s_t(3))*residual_tmp252;
      const s_t residual_tmp277 = residual_tmp41*(eta_s*(residual_tmp208*residual_tmp50 + residual_tmp45*residual_tmp49) + residual_tmp276*residual_tmp47);
      const s_t grad_coeff0_0 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp87 + residual_tmp20*residual_tmp85) - residual_tmp40*residual_tmp91) + residual_tmp68*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp108*residual_tmp20 + residual_tmp113*residual_tmp12 + residual_tmp115 + residual_tmp116) - residual_tmp103*residual_tmp40) + residual_tmp44*residual_tmp68) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp132 + residual_tmp137*residual_tmp20 + residual_tmp139 + residual_tmp140) - residual_tmp127*residual_tmp40) + residual_tmp45*residual_tmp68) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp19 + residual_tmp20*residual_tmp28) - residual_tmp39*residual_tmp40) + residual_tmp12*residual_tmp68) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp173 + residual_tmp176*residual_tmp20 + residual_tmp177) + residual_tmp154*residual_tmp6 - residual_tmp172*residual_tmp40) + residual_tmp166*residual_tmp68) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp148 + residual_tmp152*residual_tmp20 - residual_tmp153) - residual_tmp147*residual_tmp40 - residual_tmp154*u2_grad_1) + residual_tmp50*residual_tmp68) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp74 + residual_tmp20*residual_tmp72) - residual_tmp40*residual_tmp83) + residual_tmp20*residual_tmp68) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp164 + residual_tmp161*residual_tmp20 - residual_tmp165) - residual_tmp154*u1_grad_2 - residual_tmp160*residual_tmp40) + residual_tmp43*residual_tmp68) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp187 + residual_tmp185*residual_tmp20 + residual_tmp188) + residual_tmp154*residual_tmp46 - residual_tmp184*residual_tmp40) + residual_tmp178*residual_tmp68);
      const s_t grad_coeff0_1 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp114 - residual_tmp194 - residual_tmp43*residual_tmp85 + residual_tmp51*residual_tmp87) + ((s_t(1) / s_t(3)))*residual_tmp44*residual_tmp91) + residual_tmp189*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp108*residual_tmp43 + residual_tmp113*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp103*residual_tmp44) + residual_tmp189*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp132*residual_tmp51 - residual_tmp137*residual_tmp43 + residual_tmp193) + ((s_t(1) / s_t(3)))*residual_tmp127*residual_tmp44) + residual_tmp189*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp177 + residual_tmp19*residual_tmp51 - residual_tmp28*residual_tmp43) + residual_tmp196*residual_tmp6 + residual_tmp198*residual_tmp39) + residual_tmp12*residual_tmp189) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp173*residual_tmp51 - residual_tmp176*residual_tmp43) + ((s_t(1) / s_t(3)))*residual_tmp172*residual_tmp44) + residual_tmp166*residual_tmp189) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp148*residual_tmp51 - residual_tmp152*residual_tmp43 - residual_tmp195) + ((s_t(1) / s_t(3)))*residual_tmp147*residual_tmp44 - residual_tmp197) + residual_tmp189*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp165 - residual_tmp43*residual_tmp72 + residual_tmp51*residual_tmp74) - residual_tmp196*u1_grad_2 + ((s_t(1) / s_t(3)))*residual_tmp44*residual_tmp83) + residual_tmp189*residual_tmp20) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp161*residual_tmp43 + residual_tmp164*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp160*residual_tmp44) + residual_tmp189*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp185*residual_tmp43 + residual_tmp187*residual_tmp51 + residual_tmp199) + residual_tmp184*residual_tmp198 + residual_tmp200) + residual_tmp178*residual_tmp189);
      const s_t grad_coeff0_2 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp138 - residual_tmp202 + residual_tmp47*residual_tmp85 - residual_tmp50*residual_tmp87) + ((s_t(1) / s_t(3)))*residual_tmp45*residual_tmp91) + residual_tmp201*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp108*residual_tmp47 - residual_tmp113*residual_tmp50 - residual_tmp193) + ((s_t(1) / s_t(3)))*residual_tmp103*residual_tmp45) + residual_tmp201*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp132*residual_tmp50 + residual_tmp137*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp127*residual_tmp45) + residual_tmp201*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp153 - residual_tmp19*residual_tmp50 + residual_tmp28*residual_tmp47) - residual_tmp196*u2_grad_1 + ((s_t(1) / s_t(3)))*residual_tmp39*residual_tmp45) + residual_tmp12*residual_tmp201) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp173*residual_tmp50 + residual_tmp176*residual_tmp47 + residual_tmp195) + residual_tmp172*residual_tmp203 + residual_tmp197) + residual_tmp166*residual_tmp201) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp148*residual_tmp50 + residual_tmp152*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp147*residual_tmp45) + residual_tmp201*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp188 + residual_tmp47*residual_tmp72 - residual_tmp50*residual_tmp74) + residual_tmp196*residual_tmp46 + residual_tmp203*residual_tmp83) + residual_tmp20*residual_tmp201) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp161*residual_tmp47 - residual_tmp164*residual_tmp50 - residual_tmp199) + ((s_t(1) / s_t(3)))*residual_tmp160*residual_tmp45 - residual_tmp200) + residual_tmp201*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp185*residual_tmp47 - residual_tmp187*residual_tmp50) + ((s_t(1) / s_t(3)))*residual_tmp184*residual_tmp45) + residual_tmp178*residual_tmp201);
      const s_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp20*residual_tmp213 + residual_tmp7*residual_tmp87) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp214) + residual_tmp210*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp113*residual_tmp7 - residual_tmp20*residual_tmp236 + residual_tmp237) + residual_tmp229*residual_tmp6 + residual_tmp234*residual_tmp235) + residual_tmp210*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp132*residual_tmp7 - residual_tmp20*residual_tmp227 - residual_tmp228) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp226 - residual_tmp230) + residual_tmp210*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp19*residual_tmp7 - residual_tmp20*residual_tmp205) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp207) + residual_tmp12*residual_tmp210) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp173*residual_tmp7 - residual_tmp194 - residual_tmp20*residual_tmp224 - residual_tmp225) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp222) + residual_tmp166*residual_tmp210) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp148*residual_tmp7 - residual_tmp20*residual_tmp218 + residual_tmp220) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp216) + residual_tmp210*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp20*residual_tmp211 + residual_tmp7*residual_tmp74) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp212) + residual_tmp20*residual_tmp210) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp164*residual_tmp7 - residual_tmp20*residual_tmp232 - residual_tmp233) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp231 - residual_tmp229*u0_grad_2) + residual_tmp210*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp187*residual_tmp7 - residual_tmp20*residual_tmp239 + residual_tmp240) + residual_tmp235*residual_tmp238 + residual_tmp241) + residual_tmp178*residual_tmp210);
      const s_t grad_coeff1_1 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp213*residual_tmp43 + residual_tmp237 + residual_tmp44*residual_tmp87) - residual_tmp214*residual_tmp242 + residual_tmp243*residual_tmp6) + residual_tmp244*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp113*residual_tmp44 + residual_tmp236*residual_tmp43) - residual_tmp234*residual_tmp242) + residual_tmp244*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp132*residual_tmp44 + residual_tmp227*residual_tmp43 - residual_tmp246) - residual_tmp226*residual_tmp242 - residual_tmp243*u2_grad_0) + residual_tmp244*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp116 + residual_tmp19*residual_tmp44 + residual_tmp205*residual_tmp43 - residual_tmp225) - residual_tmp207*residual_tmp242) + residual_tmp12*residual_tmp244) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp173*residual_tmp44 + residual_tmp224*residual_tmp43) - residual_tmp222*residual_tmp242) + residual_tmp166*residual_tmp244) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp148*residual_tmp44 + residual_tmp192 + residual_tmp218*residual_tmp43 + residual_tmp245) - residual_tmp216*residual_tmp242) + residual_tmp244*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp211*residual_tmp43 - residual_tmp233 + residual_tmp44*residual_tmp74) - residual_tmp212*residual_tmp242 - residual_tmp243*u0_grad_2) + residual_tmp20*residual_tmp244) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp164*residual_tmp44 + residual_tmp232*residual_tmp43) - residual_tmp231*residual_tmp242) + residual_tmp244*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp187*residual_tmp44 + residual_tmp239*residual_tmp43 + residual_tmp247) + residual_tmp141*residual_tmp243 - residual_tmp238*residual_tmp242) + residual_tmp178*residual_tmp244);
      const s_t grad_coeff1_2 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp213*residual_tmp47 + residual_tmp228 - residual_tmp45*residual_tmp87) + residual_tmp214*residual_tmp250 + residual_tmp230) + residual_tmp248*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp113*residual_tmp45 + residual_tmp236*residual_tmp47 - residual_tmp246) - residual_tmp229*u2_grad_0 + ((s_t(1) / s_t(3)))*residual_tmp234*residual_tmp50) + residual_tmp248*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp132*residual_tmp45 + residual_tmp227*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp226*residual_tmp50) + residual_tmp248*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp19*residual_tmp45 + residual_tmp205*residual_tmp47 - residual_tmp220) + ((s_t(1) / s_t(3)))*residual_tmp207*residual_tmp50) + residual_tmp12*residual_tmp248) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp173*residual_tmp45 - residual_tmp191 + residual_tmp224*residual_tmp47 - residual_tmp249) + ((s_t(1) / s_t(3)))*residual_tmp222*residual_tmp50) + residual_tmp166*residual_tmp248) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp148*residual_tmp45 + residual_tmp218*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp216*residual_tmp50) + residual_tmp248*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp211*residual_tmp47 - residual_tmp240 - residual_tmp45*residual_tmp74) + ((s_t(1) / s_t(3)))*residual_tmp212*residual_tmp50 - residual_tmp241) + residual_tmp20*residual_tmp248) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp164*residual_tmp45 + residual_tmp232*residual_tmp47 + residual_tmp247) + residual_tmp141*residual_tmp229 + residual_tmp231*residual_tmp250) + residual_tmp248*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp187*residual_tmp45 + residual_tmp239*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp238*residual_tmp50) + residual_tmp178*residual_tmp248);
      const s_t grad_coeff2_0 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp213 + residual_tmp7*residual_tmp85) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp255) + residual_tmp253*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp108*residual_tmp7 - residual_tmp12*residual_tmp236 - residual_tmp260) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp259 - residual_tmp262) + residual_tmp253*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp227 + residual_tmp137*residual_tmp7 + residual_tmp267) + residual_tmp261*residual_tmp46 + residual_tmp265*residual_tmp266) + residual_tmp253*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp205 + residual_tmp28*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp254) + residual_tmp12*residual_tmp253) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp224 + residual_tmp176*residual_tmp7 + residual_tmp269) + residual_tmp266*residual_tmp268 + residual_tmp270) + residual_tmp166*residual_tmp253) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp218 + residual_tmp152*residual_tmp7 - residual_tmp264) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp263 - residual_tmp261*u0_grad_1) + residual_tmp253*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp211 + residual_tmp7*residual_tmp72) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp251) + residual_tmp20*residual_tmp253) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp232 + residual_tmp161*residual_tmp7 + residual_tmp257) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp256) + residual_tmp253*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp239 + residual_tmp185*residual_tmp7 - residual_tmp202 - residual_tmp219) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp258) + residual_tmp178*residual_tmp253);
      const s_t grad_coeff2_1 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp213*residual_tmp51 + residual_tmp260 - residual_tmp44*residual_tmp85) + residual_tmp255*residual_tmp273 + residual_tmp262) + residual_tmp271*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp108*residual_tmp44 + residual_tmp236*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp259*residual_tmp43) + residual_tmp271*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp137*residual_tmp44 + residual_tmp227*residual_tmp51 - residual_tmp272) - residual_tmp261*u1_grad_0 + ((s_t(1) / s_t(3)))*residual_tmp265*residual_tmp43) + residual_tmp271*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp205*residual_tmp51 - residual_tmp269 - residual_tmp28*residual_tmp44) + ((s_t(1) / s_t(3)))*residual_tmp254*residual_tmp43 - residual_tmp270) + residual_tmp12*residual_tmp271) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp176*residual_tmp44 + residual_tmp224*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp268*residual_tmp43) + residual_tmp166*residual_tmp271) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp152*residual_tmp44 + residual_tmp218*residual_tmp51 + residual_tmp274) + residual_tmp141*residual_tmp261 + residual_tmp263*residual_tmp273) + residual_tmp271*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp211*residual_tmp51 - residual_tmp257 - residual_tmp44*residual_tmp72) + ((s_t(1) / s_t(3)))*residual_tmp251*residual_tmp43) + residual_tmp20*residual_tmp271) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp161*residual_tmp44 + residual_tmp232*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp256*residual_tmp43) + residual_tmp271*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp185*residual_tmp44 - residual_tmp190 + residual_tmp239*residual_tmp51 - residual_tmp249) + ((s_t(1) / s_t(3)))*residual_tmp258*residual_tmp43) + residual_tmp178*residual_tmp271);
      const s_t grad_coeff2_2 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp213*residual_tmp50 + residual_tmp267 + residual_tmp45*residual_tmp85) - residual_tmp255*residual_tmp275 + residual_tmp276*residual_tmp46) + residual_tmp277*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp108*residual_tmp45 + residual_tmp236*residual_tmp50 - residual_tmp272) - residual_tmp259*residual_tmp275 - residual_tmp276*u1_grad_0) + residual_tmp277*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp137*residual_tmp45 + residual_tmp227*residual_tmp50) - residual_tmp265*residual_tmp275) + residual_tmp277*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp205*residual_tmp50 - residual_tmp264 + residual_tmp28*residual_tmp45) - residual_tmp254*residual_tmp275 - residual_tmp276*u0_grad_1) + residual_tmp12*residual_tmp277) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp176*residual_tmp45 + residual_tmp224*residual_tmp50 + residual_tmp274) + residual_tmp141*residual_tmp276 - residual_tmp268*residual_tmp275) + residual_tmp166*residual_tmp277) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp152*residual_tmp45 + residual_tmp218*residual_tmp50) - residual_tmp263*residual_tmp275) + residual_tmp277*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp140 + residual_tmp211*residual_tmp50 - residual_tmp219 + residual_tmp45*residual_tmp72) - residual_tmp251*residual_tmp275) + residual_tmp20*residual_tmp277) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp161*residual_tmp45 - residual_tmp190 + residual_tmp232*residual_tmp50 + residual_tmp245) - residual_tmp256*residual_tmp275) + residual_tmp277*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp185*residual_tmp45 + residual_tmp239*residual_tmp50) - residual_tmp258*residual_tmp275) + residual_tmp178*residual_tmp277);
      grad_coeff0_0_values[0] = grad_coeff0_0;
      grad_coeff0_1_values[0] = grad_coeff0_1;
      grad_coeff0_2_values[0] = grad_coeff0_2;
      grad_coeff1_0_values[0] = grad_coeff1_0;
      grad_coeff1_1_values[0] = grad_coeff1_1;
      grad_coeff1_2_values[0] = grad_coeff1_2;
      grad_coeff2_0_values[0] = grad_coeff2_0;
      grad_coeff2_1_values[0] = grad_coeff2_1;
      grad_coeff2_2_values[0] = grad_coeff2_2;
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
        const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj3 + grad_ref_z[q * NS + test] * adj6) / det;
        const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj4 + grad_ref_z[q * NS + test] * adj7) / det;
        const s_t test_grad2 = (grad_ref_x[q * NS + test] * adj2 + grad_ref_y[q * NS + test] * adj5 + grad_ref_z[q * NS + test] * adj8) / det;
        output[test * NC][0] += q_weight[q] * det * (grad_coeff0_0_values[0] * test_grad0 + grad_coeff0_1_values[0] * test_grad1 + grad_coeff0_2_values[0] * test_grad2);
        output[test * NC + 1][0] += q_weight[q] * det * (grad_coeff1_0_values[0] * test_grad0 + grad_coeff1_1_values[0] * test_grad1 + grad_coeff1_2_values[0] * test_grad2);
        output[test * NC + 2][0] += q_weight[q] * det * (grad_coeff2_0_values[0] * test_grad0 + grad_coeff2_1_values[0] * test_grad1 + grad_coeff2_2_values[0] * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_jacobian_action_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t current[3 * NS][VS],
    const s_t previous[3 * NS][VS],
    const s_t direction[3 * NS][VS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t newmark_velocity_alpha,
    s_t output[3 * NS][VS]
) {
  static constexpr int NC = 3;
  for (int q = 0; q < NQ; ++q) {
    s_t u0_grad_0_ref_values[VS];
    s_t u0_grad_1_ref_values[VS];
    s_t u0_grad_2_ref_values[VS];
    s_t u0_old_grad_0_ref_values[VS];
    s_t u0_old_grad_1_ref_values[VS];
    s_t u0_old_grad_2_ref_values[VS];
    s_t u0_direction_grad_0_ref_values[VS];
    s_t u0_direction_grad_1_ref_values[VS];
    s_t u0_direction_grad_2_ref_values[VS];
    s_t u1_grad_0_ref_values[VS];
    s_t u1_grad_1_ref_values[VS];
    s_t u1_grad_2_ref_values[VS];
    s_t u1_old_grad_0_ref_values[VS];
    s_t u1_old_grad_1_ref_values[VS];
    s_t u1_old_grad_2_ref_values[VS];
    s_t u1_direction_grad_0_ref_values[VS];
    s_t u1_direction_grad_1_ref_values[VS];
    s_t u1_direction_grad_2_ref_values[VS];
    s_t u2_grad_0_ref_values[VS];
    s_t u2_grad_1_ref_values[VS];
    s_t u2_grad_2_ref_values[VS];
    s_t u2_old_grad_0_ref_values[VS];
    s_t u2_old_grad_1_ref_values[VS];
    s_t u2_old_grad_2_ref_values[VS];
    s_t u2_direction_grad_0_ref_values[VS];
    s_t u2_direction_grad_1_ref_values[VS];
    s_t u2_direction_grad_2_ref_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff0_2_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    s_t grad_coeff1_2_values[VS];
    s_t grad_coeff2_0_values[VS];
    s_t grad_coeff2_1_values[VS];
    s_t grad_coeff2_2_values[VS];
    {
      u0_grad_0_ref_values[0] = s_t(0);
      u0_grad_1_ref_values[0] = s_t(0);
      u0_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC][0];
        u0_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u0_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u0_old_grad_0_ref_values[0] = s_t(0);
      u0_old_grad_1_ref_values[0] = s_t(0);
      u0_old_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC][0];
        u0_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u0_old_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u0_direction_grad_0_ref_values[0] = s_t(0);
      u0_direction_grad_1_ref_values[0] = s_t(0);
      u0_direction_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = direction[trial * NC][0];
        u0_direction_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_direction_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u0_direction_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u1_grad_0_ref_values[0] = s_t(0);
      u1_grad_1_ref_values[0] = s_t(0);
      u1_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 1][0];
        u1_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u1_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u1_old_grad_0_ref_values[0] = s_t(0);
      u1_old_grad_1_ref_values[0] = s_t(0);
      u1_old_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC + 1][0];
        u1_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u1_old_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u1_direction_grad_0_ref_values[0] = s_t(0);
      u1_direction_grad_1_ref_values[0] = s_t(0);
      u1_direction_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = direction[trial * NC + 1][0];
        u1_direction_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_direction_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u1_direction_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u2_grad_0_ref_values[0] = s_t(0);
      u2_grad_1_ref_values[0] = s_t(0);
      u2_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 2][0];
        u2_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u2_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u2_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u2_old_grad_0_ref_values[0] = s_t(0);
      u2_old_grad_1_ref_values[0] = s_t(0);
      u2_old_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC + 2][0];
        u2_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u2_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u2_old_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u2_direction_grad_0_ref_values[0] = s_t(0);
      u2_direction_grad_1_ref_values[0] = s_t(0);
      u2_direction_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = direction[trial * NC + 2][0];
        u2_direction_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u2_direction_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u2_direction_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
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
      const s_t u0_grad_0_ref = u0_grad_0_ref_values[0];
      const s_t u0_grad_1_ref = u0_grad_1_ref_values[0];
      const s_t u0_grad_2_ref = u0_grad_2_ref_values[0];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
      const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
      const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[0];
      const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[0];
      const s_t u0_old_grad_2_ref = u0_old_grad_2_ref_values[0];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
      const s_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
      const s_t u0_direction_grad_0_ref = u0_direction_grad_0_ref_values[0];
      const s_t u0_direction_grad_1_ref = u0_direction_grad_1_ref_values[0];
      const s_t u0_direction_grad_2_ref = u0_direction_grad_2_ref_values[0];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
      const s_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
      const s_t u1_grad_0_ref = u1_grad_0_ref_values[0];
      const s_t u1_grad_1_ref = u1_grad_1_ref_values[0];
      const s_t u1_grad_2_ref = u1_grad_2_ref_values[0];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
      const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
      const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[0];
      const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[0];
      const s_t u1_old_grad_2_ref = u1_old_grad_2_ref_values[0];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
      const s_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
      const s_t u1_direction_grad_0_ref = u1_direction_grad_0_ref_values[0];
      const s_t u1_direction_grad_1_ref = u1_direction_grad_1_ref_values[0];
      const s_t u1_direction_grad_2_ref = u1_direction_grad_2_ref_values[0];
      const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
      const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
      const s_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
      const s_t u2_grad_0_ref = u2_grad_0_ref_values[0];
      const s_t u2_grad_1_ref = u2_grad_1_ref_values[0];
      const s_t u2_grad_2_ref = u2_grad_2_ref_values[0];
      const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
      const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
      const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
      const s_t u2_old_grad_0_ref = u2_old_grad_0_ref_values[0];
      const s_t u2_old_grad_1_ref = u2_old_grad_1_ref_values[0];
      const s_t u2_old_grad_2_ref = u2_old_grad_2_ref_values[0];
      const s_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
      const s_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
      const s_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
      const s_t u2_direction_grad_0_ref = u2_direction_grad_0_ref_values[0];
      const s_t u2_direction_grad_1_ref = u2_direction_grad_1_ref_values[0];
      const s_t u2_direction_grad_2_ref = u2_direction_grad_2_ref_values[0];
      const s_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
      const s_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
      const s_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = u0_grad_0*u1_grad_1;
      const s_t residual_tmp1 = u0_grad_1*u1_grad_2;
      const s_t residual_tmp2 = u0_grad_2*u2_grad_1;
      const s_t residual_tmp3 = u1_grad_2*u2_grad_1;
      const s_t residual_tmp4 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp5 = u0_grad_2*u2_grad_0;
      const s_t residual_tmp6 = u2_grad_2 + s_t(1);
      const s_t residual_tmp7 = -residual_tmp3 + residual_tmp6 + u1_grad_1*u2_grad_2 + u1_grad_1;
      const s_t residual_tmp8 = -residual_tmp5 + u0_grad_0*u2_grad_2 + u0_grad_0;
      const s_t residual_tmp9 = residual_tmp0 - residual_tmp4;
      const s_t residual_tmp10 = residual_tmp0*u2_grad_2 + residual_tmp1*u2_grad_0 + residual_tmp2*u1_grad_0 - residual_tmp3*u0_grad_0 - residual_tmp4*u2_grad_2 - residual_tmp5*u1_grad_1 + residual_tmp7 + residual_tmp8 + residual_tmp9;
      const s_t residual_tmp11 = pow_m1(residual_tmp10);
      const s_t residual_tmp12 = -residual_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
      const s_t residual_tmp13 = newmark_velocity_alpha*residual_tmp7;
      const s_t residual_tmp14 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
      const s_t residual_tmp15 = residual_tmp14*u2_grad_1;
      const s_t residual_tmp16 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
      const s_t residual_tmp17 = residual_tmp16*residual_tmp6;
      const s_t residual_tmp18 = residual_tmp15 - residual_tmp17;
      const s_t residual_tmp19 = -residual_tmp13 - residual_tmp18;
      const s_t residual_tmp20 = -residual_tmp1 + u0_grad_2*u1_grad_1 + u0_grad_2;
      const s_t residual_tmp21 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
      const s_t residual_tmp22 = residual_tmp21*residual_tmp6;
      const s_t residual_tmp23 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
      const s_t residual_tmp24 = residual_tmp23*u2_grad_1;
      const s_t residual_tmp25 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
      const s_t residual_tmp26 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
      const s_t residual_tmp27 = residual_tmp25*u0_grad_1 - residual_tmp26*u0_grad_2;
      const s_t residual_tmp28 = residual_tmp22 - residual_tmp24 + residual_tmp27;
      const s_t residual_tmp29 = newmark_velocity_alpha*residual_tmp12;
      const s_t residual_tmp30 = residual_tmp23*u0_grad_1;
      const s_t residual_tmp31 = residual_tmp21*u0_grad_2;
      const s_t residual_tmp32 = residual_tmp29 + residual_tmp30 - residual_tmp31;
      const s_t residual_tmp33 = residual_tmp26*residual_tmp6;
      const s_t residual_tmp34 = residual_tmp25*u2_grad_1;
      const s_t residual_tmp35 = residual_tmp33 - residual_tmp34;
      const s_t residual_tmp36 = s_t(3)*eta_b;
      const s_t residual_tmp37 = residual_tmp36*(residual_tmp32 + residual_tmp35);
      const s_t residual_tmp38 = s_t(2)*eta_s;
      const s_t residual_tmp39 = residual_tmp37 + residual_tmp38*(s_t(2)*residual_tmp26*residual_tmp6 - residual_tmp32 - s_t(2)*residual_tmp34);
      const s_t residual_tmp40 = ((s_t(1) / s_t(3)))*residual_tmp7;
      const s_t residual_tmp41 = pow_m2(residual_tmp10);
      const s_t residual_tmp42 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
      const s_t residual_tmp43 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
      const s_t residual_tmp44 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
      const s_t residual_tmp45 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
      const s_t residual_tmp46 = u1_grad_1 + s_t(1);
      const s_t residual_tmp47 = residual_tmp46 + residual_tmp9 + u0_grad_0;
      const s_t residual_tmp48 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
      const s_t residual_tmp49 = residual_tmp20*residual_tmp42 + residual_tmp21*residual_tmp44 + residual_tmp23*residual_tmp45 - residual_tmp25*residual_tmp47 + residual_tmp26*residual_tmp43 - residual_tmp48*residual_tmp7;
      const s_t residual_tmp50 = u0_grad_0*u2_grad_1 - u0_grad_1*u2_grad_0 + u2_grad_1;
      const s_t residual_tmp51 = residual_tmp6 + residual_tmp8;
      const s_t residual_tmp52 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
      const s_t residual_tmp53 = residual_tmp12*residual_tmp42 + residual_tmp14*residual_tmp45 + residual_tmp16*residual_tmp44 + residual_tmp25*residual_tmp50 - residual_tmp26*residual_tmp51 - residual_tmp52*residual_tmp7;
      const s_t residual_tmp54 = residual_tmp12*residual_tmp52;
      const s_t residual_tmp55 = residual_tmp14*residual_tmp50;
      const s_t residual_tmp56 = residual_tmp20*residual_tmp48;
      const s_t residual_tmp57 = residual_tmp21*residual_tmp43;
      const s_t residual_tmp58 = residual_tmp16*residual_tmp51;
      const s_t residual_tmp59 = -residual_tmp58;
      const s_t residual_tmp60 = residual_tmp23*residual_tmp47;
      const s_t residual_tmp61 = -residual_tmp60;
      const s_t residual_tmp62 = residual_tmp54 + residual_tmp55 + residual_tmp56 + residual_tmp57 + residual_tmp59 + residual_tmp61;
      const s_t residual_tmp63 = residual_tmp42*residual_tmp7;
      const s_t residual_tmp64 = residual_tmp25*residual_tmp45 + residual_tmp26*residual_tmp44 - residual_tmp63;
      const s_t residual_tmp65 = residual_tmp36*(residual_tmp62 + residual_tmp64);
      const s_t residual_tmp66 = residual_tmp38*(s_t(2)*residual_tmp25*residual_tmp45 + s_t(2)*residual_tmp26*residual_tmp44 - residual_tmp62 - s_t(2)*residual_tmp63) + residual_tmp65;
      const s_t residual_tmp67 = -residual_tmp66;
      const s_t residual_tmp68 = residual_tmp41*(eta_s*(residual_tmp12*residual_tmp53 + residual_tmp20*residual_tmp49) + residual_tmp40*residual_tmp67);
      const s_t residual_tmp69 = residual_tmp21*u1_grad_2;
      const s_t residual_tmp70 = residual_tmp23*residual_tmp46;
      const s_t residual_tmp71 = residual_tmp69 - residual_tmp70;
      const s_t residual_tmp72 = -residual_tmp13 - residual_tmp71;
      const s_t residual_tmp73 = -residual_tmp14*residual_tmp46 + residual_tmp16*u1_grad_2;
      const s_t residual_tmp74 = -residual_tmp27 - residual_tmp73;
      const s_t residual_tmp75 = newmark_velocity_alpha*residual_tmp20;
      const s_t residual_tmp76 = residual_tmp16*u0_grad_2;
      const s_t residual_tmp77 = residual_tmp14*u0_grad_1;
      const s_t residual_tmp78 = residual_tmp75 + residual_tmp76 - residual_tmp77;
      const s_t residual_tmp79 = residual_tmp25*residual_tmp46;
      const s_t residual_tmp80 = residual_tmp26*u1_grad_2;
      const s_t residual_tmp81 = residual_tmp79 - residual_tmp80;
      const s_t residual_tmp82 = residual_tmp36*(residual_tmp78 + residual_tmp81);
      const s_t residual_tmp83 = residual_tmp38*(s_t(2)*residual_tmp25*residual_tmp46 - residual_tmp78 - s_t(2)*residual_tmp80) + residual_tmp82;
      const s_t residual_tmp84 = -residual_tmp79 + residual_tmp80;
      const s_t residual_tmp85 = residual_tmp75 + residual_tmp84;
      const s_t residual_tmp86 = -residual_tmp33 + residual_tmp34;
      const s_t residual_tmp87 = residual_tmp29 + residual_tmp86;
      const s_t residual_tmp88 = residual_tmp13 - residual_tmp69 + residual_tmp70;
      const s_t residual_tmp89 = -residual_tmp15 + residual_tmp17;
      const s_t residual_tmp90 = residual_tmp36*(-residual_tmp88 - residual_tmp89);
      const s_t residual_tmp91 = residual_tmp38*(-s_t(2)*residual_tmp13 - residual_tmp18 - residual_tmp71) + residual_tmp90;
      const s_t residual_tmp92 = -residual_tmp7;
      const s_t residual_tmp93 = newmark_velocity_alpha*residual_tmp44;
      const s_t residual_tmp94 = residual_tmp23*u1_grad_0;
      const s_t residual_tmp95 = residual_tmp48*u1_grad_2;
      const s_t residual_tmp96 = residual_tmp93 + residual_tmp94 - residual_tmp95;
      const s_t residual_tmp97 = residual_tmp52*residual_tmp6;
      const s_t residual_tmp98 = residual_tmp14*u2_grad_0;
      const s_t residual_tmp99 = residual_tmp97 - residual_tmp98;
      const s_t residual_tmp100 = residual_tmp36*(residual_tmp96 + residual_tmp99);
      const s_t residual_tmp101 = -residual_tmp97 + residual_tmp98;
      const s_t residual_tmp102 = -residual_tmp94 + residual_tmp95;
      const s_t residual_tmp103 = residual_tmp100 + residual_tmp38*(residual_tmp101 + residual_tmp102 + s_t(2)*residual_tmp93);
      const s_t residual_tmp104 = newmark_velocity_alpha*residual_tmp43;
      const s_t residual_tmp105 = residual_tmp25*u1_grad_0;
      const s_t residual_tmp106 = residual_tmp42*u1_grad_2;
      const s_t residual_tmp107 = residual_tmp105 - residual_tmp106;
      const s_t residual_tmp108 = residual_tmp104 + residual_tmp107;
      const s_t residual_tmp109 = newmark_velocity_alpha*residual_tmp51;
      const s_t residual_tmp110 = residual_tmp25*u2_grad_0;
      const s_t residual_tmp111 = residual_tmp42*residual_tmp6;
      const s_t residual_tmp112 = residual_tmp110 - residual_tmp111;
      const s_t residual_tmp113 = -residual_tmp109 - residual_tmp112;
      const s_t residual_tmp114 = residual_tmp49*u1_grad_2;
      const s_t residual_tmp115 = -residual_tmp114;
      const s_t residual_tmp116 = residual_tmp53*residual_tmp6;
      const s_t residual_tmp117 = newmark_velocity_alpha*residual_tmp45;
      const s_t residual_tmp118 = residual_tmp46*residual_tmp48;
      const s_t residual_tmp119 = residual_tmp21*u1_grad_0;
      const s_t residual_tmp120 = residual_tmp117 + residual_tmp118 - residual_tmp119;
      const s_t residual_tmp121 = residual_tmp16*u2_grad_0;
      const s_t residual_tmp122 = residual_tmp52*u2_grad_1;
      const s_t residual_tmp123 = residual_tmp121 - residual_tmp122;
      const s_t residual_tmp124 = residual_tmp36*(residual_tmp120 + residual_tmp123);
      const s_t residual_tmp125 = -residual_tmp121 + residual_tmp122;
      const s_t residual_tmp126 = -residual_tmp118 + residual_tmp119;
      const s_t residual_tmp127 = residual_tmp124 + residual_tmp38*(s_t(2)*residual_tmp117 + residual_tmp125 + residual_tmp126);
      const s_t residual_tmp128 = newmark_velocity_alpha*residual_tmp50;
      const s_t residual_tmp129 = residual_tmp26*u2_grad_0;
      const s_t residual_tmp130 = residual_tmp42*u2_grad_1;
      const s_t residual_tmp131 = residual_tmp129 - residual_tmp130;
      const s_t residual_tmp132 = residual_tmp128 + residual_tmp131;
      const s_t residual_tmp133 = newmark_velocity_alpha*residual_tmp47;
      const s_t residual_tmp134 = residual_tmp26*u1_grad_0;
      const s_t residual_tmp135 = residual_tmp42*residual_tmp46;
      const s_t residual_tmp136 = residual_tmp134 - residual_tmp135;
      const s_t residual_tmp137 = -residual_tmp133 - residual_tmp136;
      const s_t residual_tmp138 = residual_tmp53*u2_grad_1;
      const s_t residual_tmp139 = -residual_tmp138;
      const s_t residual_tmp140 = residual_tmp46*residual_tmp49;
      const s_t residual_tmp141 = u0_grad_0 + s_t(1);
      const s_t residual_tmp142 = residual_tmp141*residual_tmp21;
      const s_t residual_tmp143 = residual_tmp48*u0_grad_1;
      const s_t residual_tmp144 = residual_tmp128 + residual_tmp142 - residual_tmp143;
      const s_t residual_tmp145 = -residual_tmp129 + residual_tmp130;
      const s_t residual_tmp146 = residual_tmp36*(residual_tmp144 + residual_tmp145);
      const s_t residual_tmp147 = residual_tmp146 + residual_tmp38*(-s_t(2)*residual_tmp129 - residual_tmp144 + s_t(2)*residual_tmp42*u2_grad_1);
      const s_t residual_tmp148 = residual_tmp117 + residual_tmp125;
      const s_t residual_tmp149 = residual_tmp21*u2_grad_0;
      const s_t residual_tmp150 = residual_tmp48*u2_grad_1;
      const s_t residual_tmp151 = -residual_tmp141*residual_tmp26 + residual_tmp42*u0_grad_1;
      const s_t residual_tmp152 = -residual_tmp149 + residual_tmp150 - residual_tmp151;
      const s_t residual_tmp153 = residual_tmp49*u0_grad_1;
      const s_t residual_tmp154 = ((s_t(1) / s_t(3)))*residual_tmp67;
      const s_t residual_tmp155 = residual_tmp14*residual_tmp141;
      const s_t residual_tmp156 = residual_tmp52*u0_grad_2;
      const s_t residual_tmp157 = residual_tmp104 + residual_tmp155 - residual_tmp156;
      const s_t residual_tmp158 = -residual_tmp105 + residual_tmp106;
      const s_t residual_tmp159 = residual_tmp36*(residual_tmp157 + residual_tmp158);
      const s_t residual_tmp160 = residual_tmp159 + residual_tmp38*(-s_t(2)*residual_tmp105 - residual_tmp157 + s_t(2)*residual_tmp42*u1_grad_2);
      const s_t residual_tmp161 = residual_tmp102 + residual_tmp93;
      const s_t residual_tmp162 = -residual_tmp141*residual_tmp25 + residual_tmp42*u0_grad_2;
      const s_t residual_tmp163 = residual_tmp14*u1_grad_0 - residual_tmp52*u1_grad_2;
      const s_t residual_tmp164 = -residual_tmp162 - residual_tmp163;
      const s_t residual_tmp165 = residual_tmp53*u0_grad_2;
      const s_t residual_tmp166 = -residual_tmp51;
      const s_t residual_tmp167 = residual_tmp141*residual_tmp23;
      const s_t residual_tmp168 = residual_tmp48*u0_grad_2;
      const s_t residual_tmp169 = residual_tmp109 + residual_tmp167 - residual_tmp168;
      const s_t residual_tmp170 = -residual_tmp110 + residual_tmp111;
      const s_t residual_tmp171 = residual_tmp36*(-residual_tmp169 - residual_tmp170);
      const s_t residual_tmp172 = residual_tmp171 + residual_tmp38*(s_t(2)*residual_tmp110 - s_t(2)*residual_tmp111 + residual_tmp169);
      const s_t residual_tmp173 = residual_tmp101 + residual_tmp93;
      const s_t residual_tmp174 = residual_tmp23*u2_grad_0;
      const s_t residual_tmp175 = residual_tmp48*residual_tmp6;
      const s_t residual_tmp176 = residual_tmp162 + residual_tmp174 - residual_tmp175;
      const s_t residual_tmp177 = residual_tmp49*u0_grad_2;
      const s_t residual_tmp178 = -residual_tmp47;
      const s_t residual_tmp179 = residual_tmp141*residual_tmp16;
      const s_t residual_tmp180 = residual_tmp52*u0_grad_1;
      const s_t residual_tmp181 = residual_tmp133 + residual_tmp179 - residual_tmp180;
      const s_t residual_tmp182 = -residual_tmp134 + residual_tmp135;
      const s_t residual_tmp183 = residual_tmp36*(-residual_tmp181 - residual_tmp182);
      const s_t residual_tmp184 = residual_tmp183 + residual_tmp38*(s_t(2)*residual_tmp134 - s_t(2)*residual_tmp135 + residual_tmp181);
      const s_t residual_tmp185 = residual_tmp117 + residual_tmp126;
      const s_t residual_tmp186 = residual_tmp16*u1_grad_0 - residual_tmp46*residual_tmp52;
      const s_t residual_tmp187 = residual_tmp151 + residual_tmp186;
      const s_t residual_tmp188 = residual_tmp53*u0_grad_1;
      const s_t residual_tmp189 = residual_tmp41*(-eta_s*(-residual_tmp43*residual_tmp49 + residual_tmp51*residual_tmp53) + ((s_t(1) / s_t(3)))*residual_tmp44*residual_tmp66);
      const s_t residual_tmp190 = residual_tmp49*u1_grad_0;
      const s_t residual_tmp191 = residual_tmp53*u2_grad_0;
      const s_t residual_tmp192 = -residual_tmp191;
      const s_t residual_tmp193 = residual_tmp190 + residual_tmp192;
      const s_t residual_tmp194 = -residual_tmp53*residual_tmp6;
      const s_t residual_tmp195 = residual_tmp141*residual_tmp49;
      const s_t residual_tmp196 = ((s_t(1) / s_t(3)))*residual_tmp66;
      const s_t residual_tmp197 = residual_tmp196*u2_grad_0;
      const s_t residual_tmp198 = ((s_t(1) / s_t(3)))*residual_tmp44;
      const s_t residual_tmp199 = residual_tmp141*residual_tmp53;
      const s_t residual_tmp200 = residual_tmp196*u1_grad_0;
      const s_t residual_tmp201 = residual_tmp41*(-eta_s*(residual_tmp47*residual_tmp49 - residual_tmp50*residual_tmp53) + ((s_t(1) / s_t(3)))*residual_tmp45*residual_tmp66);
      const s_t residual_tmp202 = -residual_tmp46*residual_tmp49;
      const s_t residual_tmp203 = ((s_t(1) / s_t(3)))*residual_tmp45;
      const s_t residual_tmp204 = -residual_tmp76 + residual_tmp77;
      const s_t residual_tmp205 = residual_tmp204 + residual_tmp75;
      const s_t residual_tmp206 = -residual_tmp30 + residual_tmp31;
      const s_t residual_tmp207 = residual_tmp37 + residual_tmp38*(residual_tmp206 + s_t(2)*residual_tmp29 + residual_tmp86);
      const s_t residual_tmp208 = residual_tmp12*residual_tmp48 - residual_tmp14*residual_tmp47 + residual_tmp16*residual_tmp43 + residual_tmp20*residual_tmp52 - residual_tmp21*residual_tmp51 + residual_tmp23*residual_tmp50;
      const s_t residual_tmp209 = residual_tmp38*(s_t(2)*residual_tmp12*residual_tmp52 + s_t(2)*residual_tmp14*residual_tmp50 - residual_tmp56 - residual_tmp57 - s_t(2)*residual_tmp58 - residual_tmp61 - residual_tmp64) + residual_tmp65;
      const s_t residual_tmp210 = residual_tmp41*(-eta_s*(-residual_tmp20*residual_tmp208 + residual_tmp53*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp209);
      const s_t residual_tmp211 = residual_tmp206 + residual_tmp29;
      const s_t residual_tmp212 = residual_tmp38*(s_t(2)*residual_tmp16*u0_grad_2 - residual_tmp75 - s_t(2)*residual_tmp77 - residual_tmp81) + residual_tmp82;
      const s_t residual_tmp213 = -residual_tmp22 + residual_tmp24 + residual_tmp73;
      const s_t residual_tmp214 = residual_tmp38*(s_t(2)*residual_tmp15 - s_t(2)*residual_tmp17 + residual_tmp88) + residual_tmp90;
      const s_t residual_tmp215 = -residual_tmp142 + residual_tmp143;
      const s_t residual_tmp216 = residual_tmp146 + residual_tmp38*(s_t(2)*residual_tmp128 + residual_tmp131 + residual_tmp215);
      const s_t residual_tmp217 = -residual_tmp179 + residual_tmp180;
      const s_t residual_tmp218 = -residual_tmp133 - residual_tmp217;
      const s_t residual_tmp219 = residual_tmp208*u0_grad_1;
      const s_t residual_tmp220 = residual_tmp139 + residual_tmp219;
      const s_t residual_tmp221 = -residual_tmp167 + residual_tmp168;
      const s_t residual_tmp222 = residual_tmp171 + residual_tmp38*(-s_t(2)*residual_tmp109 - residual_tmp112 - residual_tmp221);
      const s_t residual_tmp223 = -residual_tmp155 + residual_tmp156;
      const s_t residual_tmp224 = residual_tmp104 + residual_tmp223;
      const s_t residual_tmp225 = residual_tmp208*u0_grad_2;
      const s_t residual_tmp226 = residual_tmp124 + residual_tmp38*(-residual_tmp120 - s_t(2)*residual_tmp122 + s_t(2)*residual_tmp16*u2_grad_0);
      const s_t residual_tmp227 = residual_tmp149 - residual_tmp150 - residual_tmp186;
      const s_t residual_tmp228 = residual_tmp208*residual_tmp46;
      const s_t residual_tmp229 = ((s_t(1) / s_t(3)))*residual_tmp209;
      const s_t residual_tmp230 = residual_tmp229*u2_grad_1;
      const s_t residual_tmp231 = residual_tmp159 + residual_tmp38*(-residual_tmp104 + s_t(2)*residual_tmp14*residual_tmp141 - s_t(2)*residual_tmp156 - residual_tmp158);
      const s_t residual_tmp232 = -residual_tmp109 - residual_tmp221;
      const s_t residual_tmp233 = residual_tmp53*u1_grad_2;
      const s_t residual_tmp234 = residual_tmp100 + residual_tmp38*(s_t(2)*residual_tmp52*residual_tmp6 - residual_tmp96 - s_t(2)*residual_tmp98);
      const s_t residual_tmp235 = ((s_t(1) / s_t(3)))*residual_tmp12;
      const s_t residual_tmp236 = residual_tmp163 - residual_tmp174 + residual_tmp175;
      const s_t residual_tmp237 = residual_tmp208*u1_grad_2;
      const s_t residual_tmp238 = residual_tmp183 + residual_tmp38*(residual_tmp133 - s_t(2)*residual_tmp179 + s_t(2)*residual_tmp180 + residual_tmp182);
      const s_t residual_tmp239 = residual_tmp128 + residual_tmp215;
      const s_t residual_tmp240 = residual_tmp46*residual_tmp53;
      const s_t residual_tmp241 = residual_tmp229*u0_grad_1;
      const s_t residual_tmp242 = ((s_t(1) / s_t(3)))*residual_tmp51;
      const s_t residual_tmp243 = -(s_t(1) / s_t(3))*residual_tmp209;
      const s_t residual_tmp244 = residual_tmp41*(eta_s*(residual_tmp208*residual_tmp43 + residual_tmp44*residual_tmp53) + residual_tmp243*residual_tmp51);
      const s_t residual_tmp245 = residual_tmp141*residual_tmp208;
      const s_t residual_tmp246 = residual_tmp208*u1_grad_0;
      const s_t residual_tmp247 = residual_tmp53*u1_grad_0;
      const s_t residual_tmp248 = residual_tmp41*(-eta_s*(residual_tmp208*residual_tmp47 - residual_tmp45*residual_tmp53) + ((s_t(1) / s_t(3)))*residual_tmp209*residual_tmp50);
      const s_t residual_tmp249 = -residual_tmp141*residual_tmp208;
      const s_t residual_tmp250 = ((s_t(1) / s_t(3)))*residual_tmp50;
      const s_t residual_tmp251 = residual_tmp38*(residual_tmp204 + s_t(2)*residual_tmp75 + residual_tmp84) + residual_tmp82;
      const s_t residual_tmp252 = residual_tmp38*(s_t(2)*residual_tmp20*residual_tmp48 + s_t(2)*residual_tmp21*residual_tmp43 - residual_tmp54 - residual_tmp55 - residual_tmp59 - s_t(2)*residual_tmp60 - residual_tmp64) + residual_tmp65;
      const s_t residual_tmp253 = residual_tmp41*(-eta_s*(-residual_tmp12*residual_tmp208 + residual_tmp49*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp252);
      const s_t residual_tmp254 = residual_tmp37 + residual_tmp38*(s_t(2)*residual_tmp23*u0_grad_1 - residual_tmp29 - s_t(2)*residual_tmp31 - residual_tmp35);
      const s_t residual_tmp255 = residual_tmp38*(residual_tmp13 + s_t(2)*residual_tmp69 - s_t(2)*residual_tmp70 + residual_tmp89) + residual_tmp90;
      const s_t residual_tmp256 = residual_tmp159 + residual_tmp38*(s_t(2)*residual_tmp104 + residual_tmp107 + residual_tmp223);
      const s_t residual_tmp257 = residual_tmp115 + residual_tmp225;
      const s_t residual_tmp258 = residual_tmp183 + residual_tmp38*(-s_t(2)*residual_tmp133 - residual_tmp136 - residual_tmp217);
      const s_t residual_tmp259 = residual_tmp100 + residual_tmp38*(s_t(2)*residual_tmp23*u1_grad_0 - residual_tmp93 - s_t(2)*residual_tmp95 - residual_tmp99);
      const s_t residual_tmp260 = residual_tmp208*residual_tmp6;
      const s_t residual_tmp261 = ((s_t(1) / s_t(3)))*residual_tmp252;
      const s_t residual_tmp262 = residual_tmp261*u1_grad_2;
      const s_t residual_tmp263 = residual_tmp146 + residual_tmp38*(-residual_tmp128 + s_t(2)*residual_tmp141*residual_tmp21 - s_t(2)*residual_tmp143 - residual_tmp145);
      const s_t residual_tmp264 = residual_tmp49*u2_grad_1;
      const s_t residual_tmp265 = residual_tmp124 + residual_tmp38*(-residual_tmp117 - s_t(2)*residual_tmp119 - residual_tmp123 + s_t(2)*residual_tmp46*residual_tmp48);
      const s_t residual_tmp266 = ((s_t(1) / s_t(3)))*residual_tmp20;
      const s_t residual_tmp267 = residual_tmp208*u2_grad_1;
      const s_t residual_tmp268 = residual_tmp171 + residual_tmp38*(residual_tmp109 - s_t(2)*residual_tmp167 + s_t(2)*residual_tmp168 + residual_tmp170);
      const s_t residual_tmp269 = residual_tmp49*residual_tmp6;
      const s_t residual_tmp270 = residual_tmp261*u0_grad_2;
      const s_t residual_tmp271 = residual_tmp41*(-eta_s*(residual_tmp208*residual_tmp51 - residual_tmp44*residual_tmp49) + ((s_t(1) / s_t(3)))*residual_tmp252*residual_tmp43);
      const s_t residual_tmp272 = residual_tmp208*u2_grad_0;
      const s_t residual_tmp273 = ((s_t(1) / s_t(3)))*residual_tmp43;
      const s_t residual_tmp274 = residual_tmp49*u2_grad_0;
      const s_t residual_tmp275 = ((s_t(1) / s_t(3)))*residual_tmp47;
      const s_t residual_tmp276 = -(s_t(1) / s_t(3))*residual_tmp252;
      const s_t residual_tmp277 = residual_tmp41*(eta_s*(residual_tmp208*residual_tmp50 + residual_tmp45*residual_tmp49) + residual_tmp276*residual_tmp47);
      const s_t grad_coeff0_0 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp87 + residual_tmp20*residual_tmp85) - residual_tmp40*residual_tmp91) + residual_tmp68*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp108*residual_tmp20 + residual_tmp113*residual_tmp12 + residual_tmp115 + residual_tmp116) - residual_tmp103*residual_tmp40) + residual_tmp44*residual_tmp68) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp132 + residual_tmp137*residual_tmp20 + residual_tmp139 + residual_tmp140) - residual_tmp127*residual_tmp40) + residual_tmp45*residual_tmp68) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp19 + residual_tmp20*residual_tmp28) - residual_tmp39*residual_tmp40) + residual_tmp12*residual_tmp68) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp173 + residual_tmp176*residual_tmp20 + residual_tmp177) + residual_tmp154*residual_tmp6 - residual_tmp172*residual_tmp40) + residual_tmp166*residual_tmp68) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp148 + residual_tmp152*residual_tmp20 - residual_tmp153) - residual_tmp147*residual_tmp40 - residual_tmp154*u2_grad_1) + residual_tmp50*residual_tmp68) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp74 + residual_tmp20*residual_tmp72) - residual_tmp40*residual_tmp83) + residual_tmp20*residual_tmp68) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp164 + residual_tmp161*residual_tmp20 - residual_tmp165) - residual_tmp154*u1_grad_2 - residual_tmp160*residual_tmp40) + residual_tmp43*residual_tmp68) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp187 + residual_tmp185*residual_tmp20 + residual_tmp188) + residual_tmp154*residual_tmp46 - residual_tmp184*residual_tmp40) + residual_tmp178*residual_tmp68);
      const s_t grad_coeff0_1 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp114 - residual_tmp194 - residual_tmp43*residual_tmp85 + residual_tmp51*residual_tmp87) + ((s_t(1) / s_t(3)))*residual_tmp44*residual_tmp91) + residual_tmp189*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp108*residual_tmp43 + residual_tmp113*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp103*residual_tmp44) + residual_tmp189*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp132*residual_tmp51 - residual_tmp137*residual_tmp43 + residual_tmp193) + ((s_t(1) / s_t(3)))*residual_tmp127*residual_tmp44) + residual_tmp189*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp177 + residual_tmp19*residual_tmp51 - residual_tmp28*residual_tmp43) + residual_tmp196*residual_tmp6 + residual_tmp198*residual_tmp39) + residual_tmp12*residual_tmp189) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp173*residual_tmp51 - residual_tmp176*residual_tmp43) + ((s_t(1) / s_t(3)))*residual_tmp172*residual_tmp44) + residual_tmp166*residual_tmp189) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp148*residual_tmp51 - residual_tmp152*residual_tmp43 - residual_tmp195) + ((s_t(1) / s_t(3)))*residual_tmp147*residual_tmp44 - residual_tmp197) + residual_tmp189*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp165 - residual_tmp43*residual_tmp72 + residual_tmp51*residual_tmp74) - residual_tmp196*u1_grad_2 + ((s_t(1) / s_t(3)))*residual_tmp44*residual_tmp83) + residual_tmp189*residual_tmp20) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp161*residual_tmp43 + residual_tmp164*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp160*residual_tmp44) + residual_tmp189*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp185*residual_tmp43 + residual_tmp187*residual_tmp51 + residual_tmp199) + residual_tmp184*residual_tmp198 + residual_tmp200) + residual_tmp178*residual_tmp189);
      const s_t grad_coeff0_2 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp138 - residual_tmp202 + residual_tmp47*residual_tmp85 - residual_tmp50*residual_tmp87) + ((s_t(1) / s_t(3)))*residual_tmp45*residual_tmp91) + residual_tmp201*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp108*residual_tmp47 - residual_tmp113*residual_tmp50 - residual_tmp193) + ((s_t(1) / s_t(3)))*residual_tmp103*residual_tmp45) + residual_tmp201*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp132*residual_tmp50 + residual_tmp137*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp127*residual_tmp45) + residual_tmp201*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp153 - residual_tmp19*residual_tmp50 + residual_tmp28*residual_tmp47) - residual_tmp196*u2_grad_1 + ((s_t(1) / s_t(3)))*residual_tmp39*residual_tmp45) + residual_tmp12*residual_tmp201) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp173*residual_tmp50 + residual_tmp176*residual_tmp47 + residual_tmp195) + residual_tmp172*residual_tmp203 + residual_tmp197) + residual_tmp166*residual_tmp201) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp148*residual_tmp50 + residual_tmp152*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp147*residual_tmp45) + residual_tmp201*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp188 + residual_tmp47*residual_tmp72 - residual_tmp50*residual_tmp74) + residual_tmp196*residual_tmp46 + residual_tmp203*residual_tmp83) + residual_tmp20*residual_tmp201) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp161*residual_tmp47 - residual_tmp164*residual_tmp50 - residual_tmp199) + ((s_t(1) / s_t(3)))*residual_tmp160*residual_tmp45 - residual_tmp200) + residual_tmp201*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp185*residual_tmp47 - residual_tmp187*residual_tmp50) + ((s_t(1) / s_t(3)))*residual_tmp184*residual_tmp45) + residual_tmp178*residual_tmp201);
      const s_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp20*residual_tmp213 + residual_tmp7*residual_tmp87) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp214) + residual_tmp210*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp113*residual_tmp7 - residual_tmp20*residual_tmp236 + residual_tmp237) + residual_tmp229*residual_tmp6 + residual_tmp234*residual_tmp235) + residual_tmp210*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp132*residual_tmp7 - residual_tmp20*residual_tmp227 - residual_tmp228) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp226 - residual_tmp230) + residual_tmp210*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp19*residual_tmp7 - residual_tmp20*residual_tmp205) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp207) + residual_tmp12*residual_tmp210) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp173*residual_tmp7 - residual_tmp194 - residual_tmp20*residual_tmp224 - residual_tmp225) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp222) + residual_tmp166*residual_tmp210) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp148*residual_tmp7 - residual_tmp20*residual_tmp218 + residual_tmp220) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp216) + residual_tmp210*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp20*residual_tmp211 + residual_tmp7*residual_tmp74) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp212) + residual_tmp20*residual_tmp210) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp164*residual_tmp7 - residual_tmp20*residual_tmp232 - residual_tmp233) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp231 - residual_tmp229*u0_grad_2) + residual_tmp210*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp187*residual_tmp7 - residual_tmp20*residual_tmp239 + residual_tmp240) + residual_tmp235*residual_tmp238 + residual_tmp241) + residual_tmp178*residual_tmp210);
      const s_t grad_coeff1_1 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp213*residual_tmp43 + residual_tmp237 + residual_tmp44*residual_tmp87) - residual_tmp214*residual_tmp242 + residual_tmp243*residual_tmp6) + residual_tmp244*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp113*residual_tmp44 + residual_tmp236*residual_tmp43) - residual_tmp234*residual_tmp242) + residual_tmp244*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp132*residual_tmp44 + residual_tmp227*residual_tmp43 - residual_tmp246) - residual_tmp226*residual_tmp242 - residual_tmp243*u2_grad_0) + residual_tmp244*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp116 + residual_tmp19*residual_tmp44 + residual_tmp205*residual_tmp43 - residual_tmp225) - residual_tmp207*residual_tmp242) + residual_tmp12*residual_tmp244) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp173*residual_tmp44 + residual_tmp224*residual_tmp43) - residual_tmp222*residual_tmp242) + residual_tmp166*residual_tmp244) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp148*residual_tmp44 + residual_tmp192 + residual_tmp218*residual_tmp43 + residual_tmp245) - residual_tmp216*residual_tmp242) + residual_tmp244*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp211*residual_tmp43 - residual_tmp233 + residual_tmp44*residual_tmp74) - residual_tmp212*residual_tmp242 - residual_tmp243*u0_grad_2) + residual_tmp20*residual_tmp244) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp164*residual_tmp44 + residual_tmp232*residual_tmp43) - residual_tmp231*residual_tmp242) + residual_tmp244*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp187*residual_tmp44 + residual_tmp239*residual_tmp43 + residual_tmp247) + residual_tmp141*residual_tmp243 - residual_tmp238*residual_tmp242) + residual_tmp178*residual_tmp244);
      const s_t grad_coeff1_2 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp213*residual_tmp47 + residual_tmp228 - residual_tmp45*residual_tmp87) + residual_tmp214*residual_tmp250 + residual_tmp230) + residual_tmp248*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp113*residual_tmp45 + residual_tmp236*residual_tmp47 - residual_tmp246) - residual_tmp229*u2_grad_0 + ((s_t(1) / s_t(3)))*residual_tmp234*residual_tmp50) + residual_tmp248*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp132*residual_tmp45 + residual_tmp227*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp226*residual_tmp50) + residual_tmp248*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp19*residual_tmp45 + residual_tmp205*residual_tmp47 - residual_tmp220) + ((s_t(1) / s_t(3)))*residual_tmp207*residual_tmp50) + residual_tmp12*residual_tmp248) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp173*residual_tmp45 - residual_tmp191 + residual_tmp224*residual_tmp47 - residual_tmp249) + ((s_t(1) / s_t(3)))*residual_tmp222*residual_tmp50) + residual_tmp166*residual_tmp248) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp148*residual_tmp45 + residual_tmp218*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp216*residual_tmp50) + residual_tmp248*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp211*residual_tmp47 - residual_tmp240 - residual_tmp45*residual_tmp74) + ((s_t(1) / s_t(3)))*residual_tmp212*residual_tmp50 - residual_tmp241) + residual_tmp20*residual_tmp248) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp164*residual_tmp45 + residual_tmp232*residual_tmp47 + residual_tmp247) + residual_tmp141*residual_tmp229 + residual_tmp231*residual_tmp250) + residual_tmp248*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp187*residual_tmp45 + residual_tmp239*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp238*residual_tmp50) + residual_tmp178*residual_tmp248);
      const s_t grad_coeff2_0 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp213 + residual_tmp7*residual_tmp85) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp255) + residual_tmp253*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp108*residual_tmp7 - residual_tmp12*residual_tmp236 - residual_tmp260) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp259 - residual_tmp262) + residual_tmp253*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp227 + residual_tmp137*residual_tmp7 + residual_tmp267) + residual_tmp261*residual_tmp46 + residual_tmp265*residual_tmp266) + residual_tmp253*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp205 + residual_tmp28*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp254) + residual_tmp12*residual_tmp253) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp224 + residual_tmp176*residual_tmp7 + residual_tmp269) + residual_tmp266*residual_tmp268 + residual_tmp270) + residual_tmp166*residual_tmp253) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp218 + residual_tmp152*residual_tmp7 - residual_tmp264) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp263 - residual_tmp261*u0_grad_1) + residual_tmp253*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp211 + residual_tmp7*residual_tmp72) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp251) + residual_tmp20*residual_tmp253) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp232 + residual_tmp161*residual_tmp7 + residual_tmp257) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp256) + residual_tmp253*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp239 + residual_tmp185*residual_tmp7 - residual_tmp202 - residual_tmp219) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp258) + residual_tmp178*residual_tmp253);
      const s_t grad_coeff2_1 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp213*residual_tmp51 + residual_tmp260 - residual_tmp44*residual_tmp85) + residual_tmp255*residual_tmp273 + residual_tmp262) + residual_tmp271*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp108*residual_tmp44 + residual_tmp236*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp259*residual_tmp43) + residual_tmp271*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp137*residual_tmp44 + residual_tmp227*residual_tmp51 - residual_tmp272) - residual_tmp261*u1_grad_0 + ((s_t(1) / s_t(3)))*residual_tmp265*residual_tmp43) + residual_tmp271*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp205*residual_tmp51 - residual_tmp269 - residual_tmp28*residual_tmp44) + ((s_t(1) / s_t(3)))*residual_tmp254*residual_tmp43 - residual_tmp270) + residual_tmp12*residual_tmp271) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp176*residual_tmp44 + residual_tmp224*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp268*residual_tmp43) + residual_tmp166*residual_tmp271) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp152*residual_tmp44 + residual_tmp218*residual_tmp51 + residual_tmp274) + residual_tmp141*residual_tmp261 + residual_tmp263*residual_tmp273) + residual_tmp271*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp211*residual_tmp51 - residual_tmp257 - residual_tmp44*residual_tmp72) + ((s_t(1) / s_t(3)))*residual_tmp251*residual_tmp43) + residual_tmp20*residual_tmp271) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp161*residual_tmp44 + residual_tmp232*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp256*residual_tmp43) + residual_tmp271*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp185*residual_tmp44 - residual_tmp190 + residual_tmp239*residual_tmp51 - residual_tmp249) + ((s_t(1) / s_t(3)))*residual_tmp258*residual_tmp43) + residual_tmp178*residual_tmp271);
      const s_t grad_coeff2_2 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp213*residual_tmp50 + residual_tmp267 + residual_tmp45*residual_tmp85) - residual_tmp255*residual_tmp275 + residual_tmp276*residual_tmp46) + residual_tmp277*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp108*residual_tmp45 + residual_tmp236*residual_tmp50 - residual_tmp272) - residual_tmp259*residual_tmp275 - residual_tmp276*u1_grad_0) + residual_tmp277*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp137*residual_tmp45 + residual_tmp227*residual_tmp50) - residual_tmp265*residual_tmp275) + residual_tmp277*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp205*residual_tmp50 - residual_tmp264 + residual_tmp28*residual_tmp45) - residual_tmp254*residual_tmp275 - residual_tmp276*u0_grad_1) + residual_tmp12*residual_tmp277) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp176*residual_tmp45 + residual_tmp224*residual_tmp50 + residual_tmp274) + residual_tmp141*residual_tmp276 - residual_tmp268*residual_tmp275) + residual_tmp166*residual_tmp277) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp152*residual_tmp45 + residual_tmp218*residual_tmp50) - residual_tmp263*residual_tmp275) + residual_tmp277*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp140 + residual_tmp211*residual_tmp50 - residual_tmp219 + residual_tmp45*residual_tmp72) - residual_tmp251*residual_tmp275) + residual_tmp20*residual_tmp277) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp161*residual_tmp45 - residual_tmp190 + residual_tmp232*residual_tmp50 + residual_tmp245) - residual_tmp256*residual_tmp275) + residual_tmp277*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp185*residual_tmp45 + residual_tmp239*residual_tmp50) - residual_tmp258*residual_tmp275) + residual_tmp178*residual_tmp277);
      grad_coeff0_0_values[0] = grad_coeff0_0;
      grad_coeff0_1_values[0] = grad_coeff0_1;
      grad_coeff0_2_values[0] = grad_coeff0_2;
      grad_coeff1_0_values[0] = grad_coeff1_0;
      grad_coeff1_1_values[0] = grad_coeff1_1;
      grad_coeff1_2_values[0] = grad_coeff1_2;
      grad_coeff2_0_values[0] = grad_coeff2_0;
      grad_coeff2_1_values[0] = grad_coeff2_1;
      grad_coeff2_2_values[0] = grad_coeff2_2;
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
        const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj3 + grad_ref_z[q * NS + test] * adj6) / det;
        const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj4 + grad_ref_z[q * NS + test] * adj7) / det;
        const s_t test_grad2 = (grad_ref_x[q * NS + test] * adj2 + grad_ref_y[q * NS + test] * adj5 + grad_ref_z[q * NS + test] * adj8) / det;
        output[test * NC][0] += q_weight[q] * det * (grad_coeff0_0_values[0] * test_grad0 + grad_coeff0_1_values[0] * test_grad1 + grad_coeff0_2_values[0] * test_grad2);
        output[test * NC + 1][0] += q_weight[q] * det * (grad_coeff1_0_values[0] * test_grad0 + grad_coeff1_1_values[0] * test_grad1 + grad_coeff1_2_values[0] * test_grad2);
        output[test * NC + 2][0] += q_weight[q] * det * (grad_coeff2_0_values[0] * test_grad0 + grad_coeff2_1_values[0] * test_grad1 + grad_coeff2_2_values[0] * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_tet4_jacobian_action_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[3 * NS],
    const s_t *const RSTR previous[3 * NS],
    const s_t *const RSTR direction[3 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t newmark_velocity_alpha,
    s_t *const RSTR output[3 * NS]
) {
  for (int q = 0; q < NQ; ++q) {
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
      const s_t u0_direction_grad_0_ref = -(direction[0][0]) + direction[3][0];
      const s_t u0_direction_grad_1_ref = -(direction[0][0]) + direction[6][0];
      const s_t u0_direction_grad_2_ref = -(direction[0][0]) + direction[9][0];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
      const s_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
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
      const s_t u1_direction_grad_0_ref = -(direction[1][0]) + direction[4][0];
      const s_t u1_direction_grad_1_ref = -(direction[1][0]) + direction[7][0];
      const s_t u1_direction_grad_2_ref = -(direction[1][0]) + direction[10][0];
      const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
      const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
      const s_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
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
      const s_t u2_direction_grad_0_ref = -(direction[2][0]) + direction[5][0];
      const s_t u2_direction_grad_1_ref = -(direction[2][0]) + direction[8][0];
      const s_t u2_direction_grad_2_ref = -(direction[2][0]) + direction[11][0];
      const s_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
      const s_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
      const s_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = u0_grad_0*u1_grad_1;
      const s_t residual_tmp1 = u0_grad_1*u1_grad_2;
      const s_t residual_tmp2 = u0_grad_2*u2_grad_1;
      const s_t residual_tmp3 = u1_grad_2*u2_grad_1;
      const s_t residual_tmp4 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp5 = u0_grad_2*u2_grad_0;
      const s_t residual_tmp6 = u2_grad_2 + s_t(1);
      const s_t residual_tmp7 = -residual_tmp3 + residual_tmp6 + u1_grad_1*u2_grad_2 + u1_grad_1;
      const s_t residual_tmp8 = -residual_tmp5 + u0_grad_0*u2_grad_2 + u0_grad_0;
      const s_t residual_tmp9 = residual_tmp0 - residual_tmp4;
      const s_t residual_tmp10 = residual_tmp0*u2_grad_2 + residual_tmp1*u2_grad_0 + residual_tmp2*u1_grad_0 - residual_tmp3*u0_grad_0 - residual_tmp4*u2_grad_2 - residual_tmp5*u1_grad_1 + residual_tmp7 + residual_tmp8 + residual_tmp9;
      const s_t residual_tmp11 = pow_m1(residual_tmp10);
      const s_t residual_tmp12 = -residual_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
      const s_t residual_tmp13 = newmark_velocity_alpha*residual_tmp7;
      const s_t residual_tmp14 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
      const s_t residual_tmp15 = residual_tmp14*u2_grad_1;
      const s_t residual_tmp16 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
      const s_t residual_tmp17 = residual_tmp16*residual_tmp6;
      const s_t residual_tmp18 = residual_tmp15 - residual_tmp17;
      const s_t residual_tmp19 = -residual_tmp13 - residual_tmp18;
      const s_t residual_tmp20 = -residual_tmp1 + u0_grad_2*u1_grad_1 + u0_grad_2;
      const s_t residual_tmp21 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
      const s_t residual_tmp22 = residual_tmp21*residual_tmp6;
      const s_t residual_tmp23 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
      const s_t residual_tmp24 = residual_tmp23*u2_grad_1;
      const s_t residual_tmp25 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
      const s_t residual_tmp26 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
      const s_t residual_tmp27 = residual_tmp25*u0_grad_1 - residual_tmp26*u0_grad_2;
      const s_t residual_tmp28 = residual_tmp22 - residual_tmp24 + residual_tmp27;
      const s_t residual_tmp29 = newmark_velocity_alpha*residual_tmp12;
      const s_t residual_tmp30 = residual_tmp23*u0_grad_1;
      const s_t residual_tmp31 = residual_tmp21*u0_grad_2;
      const s_t residual_tmp32 = residual_tmp29 + residual_tmp30 - residual_tmp31;
      const s_t residual_tmp33 = residual_tmp26*residual_tmp6;
      const s_t residual_tmp34 = residual_tmp25*u2_grad_1;
      const s_t residual_tmp35 = residual_tmp33 - residual_tmp34;
      const s_t residual_tmp36 = s_t(3)*eta_b;
      const s_t residual_tmp37 = residual_tmp36*(residual_tmp32 + residual_tmp35);
      const s_t residual_tmp38 = s_t(2)*eta_s;
      const s_t residual_tmp39 = residual_tmp37 + residual_tmp38*(s_t(2)*residual_tmp26*residual_tmp6 - residual_tmp32 - s_t(2)*residual_tmp34);
      const s_t residual_tmp40 = ((s_t(1) / s_t(3)))*residual_tmp7;
      const s_t residual_tmp41 = pow_m2(residual_tmp10);
      const s_t residual_tmp42 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
      const s_t residual_tmp43 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
      const s_t residual_tmp44 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
      const s_t residual_tmp45 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
      const s_t residual_tmp46 = u1_grad_1 + s_t(1);
      const s_t residual_tmp47 = residual_tmp46 + residual_tmp9 + u0_grad_0;
      const s_t residual_tmp48 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
      const s_t residual_tmp49 = residual_tmp20*residual_tmp42 + residual_tmp21*residual_tmp44 + residual_tmp23*residual_tmp45 - residual_tmp25*residual_tmp47 + residual_tmp26*residual_tmp43 - residual_tmp48*residual_tmp7;
      const s_t residual_tmp50 = u0_grad_0*u2_grad_1 - u0_grad_1*u2_grad_0 + u2_grad_1;
      const s_t residual_tmp51 = residual_tmp6 + residual_tmp8;
      const s_t residual_tmp52 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
      const s_t residual_tmp53 = residual_tmp12*residual_tmp42 + residual_tmp14*residual_tmp45 + residual_tmp16*residual_tmp44 + residual_tmp25*residual_tmp50 - residual_tmp26*residual_tmp51 - residual_tmp52*residual_tmp7;
      const s_t residual_tmp54 = residual_tmp12*residual_tmp52;
      const s_t residual_tmp55 = residual_tmp14*residual_tmp50;
      const s_t residual_tmp56 = residual_tmp20*residual_tmp48;
      const s_t residual_tmp57 = residual_tmp21*residual_tmp43;
      const s_t residual_tmp58 = residual_tmp16*residual_tmp51;
      const s_t residual_tmp59 = -residual_tmp58;
      const s_t residual_tmp60 = residual_tmp23*residual_tmp47;
      const s_t residual_tmp61 = -residual_tmp60;
      const s_t residual_tmp62 = residual_tmp54 + residual_tmp55 + residual_tmp56 + residual_tmp57 + residual_tmp59 + residual_tmp61;
      const s_t residual_tmp63 = residual_tmp42*residual_tmp7;
      const s_t residual_tmp64 = residual_tmp25*residual_tmp45 + residual_tmp26*residual_tmp44 - residual_tmp63;
      const s_t residual_tmp65 = residual_tmp36*(residual_tmp62 + residual_tmp64);
      const s_t residual_tmp66 = residual_tmp38*(s_t(2)*residual_tmp25*residual_tmp45 + s_t(2)*residual_tmp26*residual_tmp44 - residual_tmp62 - s_t(2)*residual_tmp63) + residual_tmp65;
      const s_t residual_tmp67 = -residual_tmp66;
      const s_t residual_tmp68 = residual_tmp41*(eta_s*(residual_tmp12*residual_tmp53 + residual_tmp20*residual_tmp49) + residual_tmp40*residual_tmp67);
      const s_t residual_tmp69 = residual_tmp21*u1_grad_2;
      const s_t residual_tmp70 = residual_tmp23*residual_tmp46;
      const s_t residual_tmp71 = residual_tmp69 - residual_tmp70;
      const s_t residual_tmp72 = -residual_tmp13 - residual_tmp71;
      const s_t residual_tmp73 = -residual_tmp14*residual_tmp46 + residual_tmp16*u1_grad_2;
      const s_t residual_tmp74 = -residual_tmp27 - residual_tmp73;
      const s_t residual_tmp75 = newmark_velocity_alpha*residual_tmp20;
      const s_t residual_tmp76 = residual_tmp16*u0_grad_2;
      const s_t residual_tmp77 = residual_tmp14*u0_grad_1;
      const s_t residual_tmp78 = residual_tmp75 + residual_tmp76 - residual_tmp77;
      const s_t residual_tmp79 = residual_tmp25*residual_tmp46;
      const s_t residual_tmp80 = residual_tmp26*u1_grad_2;
      const s_t residual_tmp81 = residual_tmp79 - residual_tmp80;
      const s_t residual_tmp82 = residual_tmp36*(residual_tmp78 + residual_tmp81);
      const s_t residual_tmp83 = residual_tmp38*(s_t(2)*residual_tmp25*residual_tmp46 - residual_tmp78 - s_t(2)*residual_tmp80) + residual_tmp82;
      const s_t residual_tmp84 = -residual_tmp79 + residual_tmp80;
      const s_t residual_tmp85 = residual_tmp75 + residual_tmp84;
      const s_t residual_tmp86 = -residual_tmp33 + residual_tmp34;
      const s_t residual_tmp87 = residual_tmp29 + residual_tmp86;
      const s_t residual_tmp88 = residual_tmp13 - residual_tmp69 + residual_tmp70;
      const s_t residual_tmp89 = -residual_tmp15 + residual_tmp17;
      const s_t residual_tmp90 = residual_tmp36*(-residual_tmp88 - residual_tmp89);
      const s_t residual_tmp91 = residual_tmp38*(-s_t(2)*residual_tmp13 - residual_tmp18 - residual_tmp71) + residual_tmp90;
      const s_t residual_tmp92 = -residual_tmp7;
      const s_t residual_tmp93 = newmark_velocity_alpha*residual_tmp44;
      const s_t residual_tmp94 = residual_tmp23*u1_grad_0;
      const s_t residual_tmp95 = residual_tmp48*u1_grad_2;
      const s_t residual_tmp96 = residual_tmp93 + residual_tmp94 - residual_tmp95;
      const s_t residual_tmp97 = residual_tmp52*residual_tmp6;
      const s_t residual_tmp98 = residual_tmp14*u2_grad_0;
      const s_t residual_tmp99 = residual_tmp97 - residual_tmp98;
      const s_t residual_tmp100 = residual_tmp36*(residual_tmp96 + residual_tmp99);
      const s_t residual_tmp101 = -residual_tmp97 + residual_tmp98;
      const s_t residual_tmp102 = -residual_tmp94 + residual_tmp95;
      const s_t residual_tmp103 = residual_tmp100 + residual_tmp38*(residual_tmp101 + residual_tmp102 + s_t(2)*residual_tmp93);
      const s_t residual_tmp104 = newmark_velocity_alpha*residual_tmp43;
      const s_t residual_tmp105 = residual_tmp25*u1_grad_0;
      const s_t residual_tmp106 = residual_tmp42*u1_grad_2;
      const s_t residual_tmp107 = residual_tmp105 - residual_tmp106;
      const s_t residual_tmp108 = residual_tmp104 + residual_tmp107;
      const s_t residual_tmp109 = newmark_velocity_alpha*residual_tmp51;
      const s_t residual_tmp110 = residual_tmp25*u2_grad_0;
      const s_t residual_tmp111 = residual_tmp42*residual_tmp6;
      const s_t residual_tmp112 = residual_tmp110 - residual_tmp111;
      const s_t residual_tmp113 = -residual_tmp109 - residual_tmp112;
      const s_t residual_tmp114 = residual_tmp49*u1_grad_2;
      const s_t residual_tmp115 = -residual_tmp114;
      const s_t residual_tmp116 = residual_tmp53*residual_tmp6;
      const s_t residual_tmp117 = newmark_velocity_alpha*residual_tmp45;
      const s_t residual_tmp118 = residual_tmp46*residual_tmp48;
      const s_t residual_tmp119 = residual_tmp21*u1_grad_0;
      const s_t residual_tmp120 = residual_tmp117 + residual_tmp118 - residual_tmp119;
      const s_t residual_tmp121 = residual_tmp16*u2_grad_0;
      const s_t residual_tmp122 = residual_tmp52*u2_grad_1;
      const s_t residual_tmp123 = residual_tmp121 - residual_tmp122;
      const s_t residual_tmp124 = residual_tmp36*(residual_tmp120 + residual_tmp123);
      const s_t residual_tmp125 = -residual_tmp121 + residual_tmp122;
      const s_t residual_tmp126 = -residual_tmp118 + residual_tmp119;
      const s_t residual_tmp127 = residual_tmp124 + residual_tmp38*(s_t(2)*residual_tmp117 + residual_tmp125 + residual_tmp126);
      const s_t residual_tmp128 = newmark_velocity_alpha*residual_tmp50;
      const s_t residual_tmp129 = residual_tmp26*u2_grad_0;
      const s_t residual_tmp130 = residual_tmp42*u2_grad_1;
      const s_t residual_tmp131 = residual_tmp129 - residual_tmp130;
      const s_t residual_tmp132 = residual_tmp128 + residual_tmp131;
      const s_t residual_tmp133 = newmark_velocity_alpha*residual_tmp47;
      const s_t residual_tmp134 = residual_tmp26*u1_grad_0;
      const s_t residual_tmp135 = residual_tmp42*residual_tmp46;
      const s_t residual_tmp136 = residual_tmp134 - residual_tmp135;
      const s_t residual_tmp137 = -residual_tmp133 - residual_tmp136;
      const s_t residual_tmp138 = residual_tmp53*u2_grad_1;
      const s_t residual_tmp139 = -residual_tmp138;
      const s_t residual_tmp140 = residual_tmp46*residual_tmp49;
      const s_t residual_tmp141 = u0_grad_0 + s_t(1);
      const s_t residual_tmp142 = residual_tmp141*residual_tmp21;
      const s_t residual_tmp143 = residual_tmp48*u0_grad_1;
      const s_t residual_tmp144 = residual_tmp128 + residual_tmp142 - residual_tmp143;
      const s_t residual_tmp145 = -residual_tmp129 + residual_tmp130;
      const s_t residual_tmp146 = residual_tmp36*(residual_tmp144 + residual_tmp145);
      const s_t residual_tmp147 = residual_tmp146 + residual_tmp38*(-s_t(2)*residual_tmp129 - residual_tmp144 + s_t(2)*residual_tmp42*u2_grad_1);
      const s_t residual_tmp148 = residual_tmp117 + residual_tmp125;
      const s_t residual_tmp149 = residual_tmp21*u2_grad_0;
      const s_t residual_tmp150 = residual_tmp48*u2_grad_1;
      const s_t residual_tmp151 = -residual_tmp141*residual_tmp26 + residual_tmp42*u0_grad_1;
      const s_t residual_tmp152 = -residual_tmp149 + residual_tmp150 - residual_tmp151;
      const s_t residual_tmp153 = residual_tmp49*u0_grad_1;
      const s_t residual_tmp154 = ((s_t(1) / s_t(3)))*residual_tmp67;
      const s_t residual_tmp155 = residual_tmp14*residual_tmp141;
      const s_t residual_tmp156 = residual_tmp52*u0_grad_2;
      const s_t residual_tmp157 = residual_tmp104 + residual_tmp155 - residual_tmp156;
      const s_t residual_tmp158 = -residual_tmp105 + residual_tmp106;
      const s_t residual_tmp159 = residual_tmp36*(residual_tmp157 + residual_tmp158);
      const s_t residual_tmp160 = residual_tmp159 + residual_tmp38*(-s_t(2)*residual_tmp105 - residual_tmp157 + s_t(2)*residual_tmp42*u1_grad_2);
      const s_t residual_tmp161 = residual_tmp102 + residual_tmp93;
      const s_t residual_tmp162 = -residual_tmp141*residual_tmp25 + residual_tmp42*u0_grad_2;
      const s_t residual_tmp163 = residual_tmp14*u1_grad_0 - residual_tmp52*u1_grad_2;
      const s_t residual_tmp164 = -residual_tmp162 - residual_tmp163;
      const s_t residual_tmp165 = residual_tmp53*u0_grad_2;
      const s_t residual_tmp166 = -residual_tmp51;
      const s_t residual_tmp167 = residual_tmp141*residual_tmp23;
      const s_t residual_tmp168 = residual_tmp48*u0_grad_2;
      const s_t residual_tmp169 = residual_tmp109 + residual_tmp167 - residual_tmp168;
      const s_t residual_tmp170 = -residual_tmp110 + residual_tmp111;
      const s_t residual_tmp171 = residual_tmp36*(-residual_tmp169 - residual_tmp170);
      const s_t residual_tmp172 = residual_tmp171 + residual_tmp38*(s_t(2)*residual_tmp110 - s_t(2)*residual_tmp111 + residual_tmp169);
      const s_t residual_tmp173 = residual_tmp101 + residual_tmp93;
      const s_t residual_tmp174 = residual_tmp23*u2_grad_0;
      const s_t residual_tmp175 = residual_tmp48*residual_tmp6;
      const s_t residual_tmp176 = residual_tmp162 + residual_tmp174 - residual_tmp175;
      const s_t residual_tmp177 = residual_tmp49*u0_grad_2;
      const s_t residual_tmp178 = -residual_tmp47;
      const s_t residual_tmp179 = residual_tmp141*residual_tmp16;
      const s_t residual_tmp180 = residual_tmp52*u0_grad_1;
      const s_t residual_tmp181 = residual_tmp133 + residual_tmp179 - residual_tmp180;
      const s_t residual_tmp182 = -residual_tmp134 + residual_tmp135;
      const s_t residual_tmp183 = residual_tmp36*(-residual_tmp181 - residual_tmp182);
      const s_t residual_tmp184 = residual_tmp183 + residual_tmp38*(s_t(2)*residual_tmp134 - s_t(2)*residual_tmp135 + residual_tmp181);
      const s_t residual_tmp185 = residual_tmp117 + residual_tmp126;
      const s_t residual_tmp186 = residual_tmp16*u1_grad_0 - residual_tmp46*residual_tmp52;
      const s_t residual_tmp187 = residual_tmp151 + residual_tmp186;
      const s_t residual_tmp188 = residual_tmp53*u0_grad_1;
      const s_t residual_tmp189 = residual_tmp41*(-eta_s*(-residual_tmp43*residual_tmp49 + residual_tmp51*residual_tmp53) + ((s_t(1) / s_t(3)))*residual_tmp44*residual_tmp66);
      const s_t residual_tmp190 = residual_tmp49*u1_grad_0;
      const s_t residual_tmp191 = residual_tmp53*u2_grad_0;
      const s_t residual_tmp192 = -residual_tmp191;
      const s_t residual_tmp193 = residual_tmp190 + residual_tmp192;
      const s_t residual_tmp194 = -residual_tmp53*residual_tmp6;
      const s_t residual_tmp195 = residual_tmp141*residual_tmp49;
      const s_t residual_tmp196 = ((s_t(1) / s_t(3)))*residual_tmp66;
      const s_t residual_tmp197 = residual_tmp196*u2_grad_0;
      const s_t residual_tmp198 = ((s_t(1) / s_t(3)))*residual_tmp44;
      const s_t residual_tmp199 = residual_tmp141*residual_tmp53;
      const s_t residual_tmp200 = residual_tmp196*u1_grad_0;
      const s_t residual_tmp201 = residual_tmp41*(-eta_s*(residual_tmp47*residual_tmp49 - residual_tmp50*residual_tmp53) + ((s_t(1) / s_t(3)))*residual_tmp45*residual_tmp66);
      const s_t residual_tmp202 = -residual_tmp46*residual_tmp49;
      const s_t residual_tmp203 = ((s_t(1) / s_t(3)))*residual_tmp45;
      const s_t residual_tmp204 = -residual_tmp76 + residual_tmp77;
      const s_t residual_tmp205 = residual_tmp204 + residual_tmp75;
      const s_t residual_tmp206 = -residual_tmp30 + residual_tmp31;
      const s_t residual_tmp207 = residual_tmp37 + residual_tmp38*(residual_tmp206 + s_t(2)*residual_tmp29 + residual_tmp86);
      const s_t residual_tmp208 = residual_tmp12*residual_tmp48 - residual_tmp14*residual_tmp47 + residual_tmp16*residual_tmp43 + residual_tmp20*residual_tmp52 - residual_tmp21*residual_tmp51 + residual_tmp23*residual_tmp50;
      const s_t residual_tmp209 = residual_tmp38*(s_t(2)*residual_tmp12*residual_tmp52 + s_t(2)*residual_tmp14*residual_tmp50 - residual_tmp56 - residual_tmp57 - s_t(2)*residual_tmp58 - residual_tmp61 - residual_tmp64) + residual_tmp65;
      const s_t residual_tmp210 = residual_tmp41*(-eta_s*(-residual_tmp20*residual_tmp208 + residual_tmp53*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp209);
      const s_t residual_tmp211 = residual_tmp206 + residual_tmp29;
      const s_t residual_tmp212 = residual_tmp38*(s_t(2)*residual_tmp16*u0_grad_2 - residual_tmp75 - s_t(2)*residual_tmp77 - residual_tmp81) + residual_tmp82;
      const s_t residual_tmp213 = -residual_tmp22 + residual_tmp24 + residual_tmp73;
      const s_t residual_tmp214 = residual_tmp38*(s_t(2)*residual_tmp15 - s_t(2)*residual_tmp17 + residual_tmp88) + residual_tmp90;
      const s_t residual_tmp215 = -residual_tmp142 + residual_tmp143;
      const s_t residual_tmp216 = residual_tmp146 + residual_tmp38*(s_t(2)*residual_tmp128 + residual_tmp131 + residual_tmp215);
      const s_t residual_tmp217 = -residual_tmp179 + residual_tmp180;
      const s_t residual_tmp218 = -residual_tmp133 - residual_tmp217;
      const s_t residual_tmp219 = residual_tmp208*u0_grad_1;
      const s_t residual_tmp220 = residual_tmp139 + residual_tmp219;
      const s_t residual_tmp221 = -residual_tmp167 + residual_tmp168;
      const s_t residual_tmp222 = residual_tmp171 + residual_tmp38*(-s_t(2)*residual_tmp109 - residual_tmp112 - residual_tmp221);
      const s_t residual_tmp223 = -residual_tmp155 + residual_tmp156;
      const s_t residual_tmp224 = residual_tmp104 + residual_tmp223;
      const s_t residual_tmp225 = residual_tmp208*u0_grad_2;
      const s_t residual_tmp226 = residual_tmp124 + residual_tmp38*(-residual_tmp120 - s_t(2)*residual_tmp122 + s_t(2)*residual_tmp16*u2_grad_0);
      const s_t residual_tmp227 = residual_tmp149 - residual_tmp150 - residual_tmp186;
      const s_t residual_tmp228 = residual_tmp208*residual_tmp46;
      const s_t residual_tmp229 = ((s_t(1) / s_t(3)))*residual_tmp209;
      const s_t residual_tmp230 = residual_tmp229*u2_grad_1;
      const s_t residual_tmp231 = residual_tmp159 + residual_tmp38*(-residual_tmp104 + s_t(2)*residual_tmp14*residual_tmp141 - s_t(2)*residual_tmp156 - residual_tmp158);
      const s_t residual_tmp232 = -residual_tmp109 - residual_tmp221;
      const s_t residual_tmp233 = residual_tmp53*u1_grad_2;
      const s_t residual_tmp234 = residual_tmp100 + residual_tmp38*(s_t(2)*residual_tmp52*residual_tmp6 - residual_tmp96 - s_t(2)*residual_tmp98);
      const s_t residual_tmp235 = ((s_t(1) / s_t(3)))*residual_tmp12;
      const s_t residual_tmp236 = residual_tmp163 - residual_tmp174 + residual_tmp175;
      const s_t residual_tmp237 = residual_tmp208*u1_grad_2;
      const s_t residual_tmp238 = residual_tmp183 + residual_tmp38*(residual_tmp133 - s_t(2)*residual_tmp179 + s_t(2)*residual_tmp180 + residual_tmp182);
      const s_t residual_tmp239 = residual_tmp128 + residual_tmp215;
      const s_t residual_tmp240 = residual_tmp46*residual_tmp53;
      const s_t residual_tmp241 = residual_tmp229*u0_grad_1;
      const s_t residual_tmp242 = ((s_t(1) / s_t(3)))*residual_tmp51;
      const s_t residual_tmp243 = -(s_t(1) / s_t(3))*residual_tmp209;
      const s_t residual_tmp244 = residual_tmp41*(eta_s*(residual_tmp208*residual_tmp43 + residual_tmp44*residual_tmp53) + residual_tmp243*residual_tmp51);
      const s_t residual_tmp245 = residual_tmp141*residual_tmp208;
      const s_t residual_tmp246 = residual_tmp208*u1_grad_0;
      const s_t residual_tmp247 = residual_tmp53*u1_grad_0;
      const s_t residual_tmp248 = residual_tmp41*(-eta_s*(residual_tmp208*residual_tmp47 - residual_tmp45*residual_tmp53) + ((s_t(1) / s_t(3)))*residual_tmp209*residual_tmp50);
      const s_t residual_tmp249 = -residual_tmp141*residual_tmp208;
      const s_t residual_tmp250 = ((s_t(1) / s_t(3)))*residual_tmp50;
      const s_t residual_tmp251 = residual_tmp38*(residual_tmp204 + s_t(2)*residual_tmp75 + residual_tmp84) + residual_tmp82;
      const s_t residual_tmp252 = residual_tmp38*(s_t(2)*residual_tmp20*residual_tmp48 + s_t(2)*residual_tmp21*residual_tmp43 - residual_tmp54 - residual_tmp55 - residual_tmp59 - s_t(2)*residual_tmp60 - residual_tmp64) + residual_tmp65;
      const s_t residual_tmp253 = residual_tmp41*(-eta_s*(-residual_tmp12*residual_tmp208 + residual_tmp49*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp252);
      const s_t residual_tmp254 = residual_tmp37 + residual_tmp38*(s_t(2)*residual_tmp23*u0_grad_1 - residual_tmp29 - s_t(2)*residual_tmp31 - residual_tmp35);
      const s_t residual_tmp255 = residual_tmp38*(residual_tmp13 + s_t(2)*residual_tmp69 - s_t(2)*residual_tmp70 + residual_tmp89) + residual_tmp90;
      const s_t residual_tmp256 = residual_tmp159 + residual_tmp38*(s_t(2)*residual_tmp104 + residual_tmp107 + residual_tmp223);
      const s_t residual_tmp257 = residual_tmp115 + residual_tmp225;
      const s_t residual_tmp258 = residual_tmp183 + residual_tmp38*(-s_t(2)*residual_tmp133 - residual_tmp136 - residual_tmp217);
      const s_t residual_tmp259 = residual_tmp100 + residual_tmp38*(s_t(2)*residual_tmp23*u1_grad_0 - residual_tmp93 - s_t(2)*residual_tmp95 - residual_tmp99);
      const s_t residual_tmp260 = residual_tmp208*residual_tmp6;
      const s_t residual_tmp261 = ((s_t(1) / s_t(3)))*residual_tmp252;
      const s_t residual_tmp262 = residual_tmp261*u1_grad_2;
      const s_t residual_tmp263 = residual_tmp146 + residual_tmp38*(-residual_tmp128 + s_t(2)*residual_tmp141*residual_tmp21 - s_t(2)*residual_tmp143 - residual_tmp145);
      const s_t residual_tmp264 = residual_tmp49*u2_grad_1;
      const s_t residual_tmp265 = residual_tmp124 + residual_tmp38*(-residual_tmp117 - s_t(2)*residual_tmp119 - residual_tmp123 + s_t(2)*residual_tmp46*residual_tmp48);
      const s_t residual_tmp266 = ((s_t(1) / s_t(3)))*residual_tmp20;
      const s_t residual_tmp267 = residual_tmp208*u2_grad_1;
      const s_t residual_tmp268 = residual_tmp171 + residual_tmp38*(residual_tmp109 - s_t(2)*residual_tmp167 + s_t(2)*residual_tmp168 + residual_tmp170);
      const s_t residual_tmp269 = residual_tmp49*residual_tmp6;
      const s_t residual_tmp270 = residual_tmp261*u0_grad_2;
      const s_t residual_tmp271 = residual_tmp41*(-eta_s*(residual_tmp208*residual_tmp51 - residual_tmp44*residual_tmp49) + ((s_t(1) / s_t(3)))*residual_tmp252*residual_tmp43);
      const s_t residual_tmp272 = residual_tmp208*u2_grad_0;
      const s_t residual_tmp273 = ((s_t(1) / s_t(3)))*residual_tmp43;
      const s_t residual_tmp274 = residual_tmp49*u2_grad_0;
      const s_t residual_tmp275 = ((s_t(1) / s_t(3)))*residual_tmp47;
      const s_t residual_tmp276 = -(s_t(1) / s_t(3))*residual_tmp252;
      const s_t residual_tmp277 = residual_tmp41*(eta_s*(residual_tmp208*residual_tmp50 + residual_tmp45*residual_tmp49) + residual_tmp276*residual_tmp47);
      const s_t grad_coeff0_0 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp87 + residual_tmp20*residual_tmp85) - residual_tmp40*residual_tmp91) + residual_tmp68*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp108*residual_tmp20 + residual_tmp113*residual_tmp12 + residual_tmp115 + residual_tmp116) - residual_tmp103*residual_tmp40) + residual_tmp44*residual_tmp68) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp132 + residual_tmp137*residual_tmp20 + residual_tmp139 + residual_tmp140) - residual_tmp127*residual_tmp40) + residual_tmp45*residual_tmp68) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp19 + residual_tmp20*residual_tmp28) - residual_tmp39*residual_tmp40) + residual_tmp12*residual_tmp68) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp173 + residual_tmp176*residual_tmp20 + residual_tmp177) + residual_tmp154*residual_tmp6 - residual_tmp172*residual_tmp40) + residual_tmp166*residual_tmp68) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp148 + residual_tmp152*residual_tmp20 - residual_tmp153) - residual_tmp147*residual_tmp40 - residual_tmp154*u2_grad_1) + residual_tmp50*residual_tmp68) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp74 + residual_tmp20*residual_tmp72) - residual_tmp40*residual_tmp83) + residual_tmp20*residual_tmp68) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp164 + residual_tmp161*residual_tmp20 - residual_tmp165) - residual_tmp154*u1_grad_2 - residual_tmp160*residual_tmp40) + residual_tmp43*residual_tmp68) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp187 + residual_tmp185*residual_tmp20 + residual_tmp188) + residual_tmp154*residual_tmp46 - residual_tmp184*residual_tmp40) + residual_tmp178*residual_tmp68);
      const s_t grad_coeff0_1 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp114 - residual_tmp194 - residual_tmp43*residual_tmp85 + residual_tmp51*residual_tmp87) + ((s_t(1) / s_t(3)))*residual_tmp44*residual_tmp91) + residual_tmp189*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp108*residual_tmp43 + residual_tmp113*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp103*residual_tmp44) + residual_tmp189*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp132*residual_tmp51 - residual_tmp137*residual_tmp43 + residual_tmp193) + ((s_t(1) / s_t(3)))*residual_tmp127*residual_tmp44) + residual_tmp189*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp177 + residual_tmp19*residual_tmp51 - residual_tmp28*residual_tmp43) + residual_tmp196*residual_tmp6 + residual_tmp198*residual_tmp39) + residual_tmp12*residual_tmp189) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp173*residual_tmp51 - residual_tmp176*residual_tmp43) + ((s_t(1) / s_t(3)))*residual_tmp172*residual_tmp44) + residual_tmp166*residual_tmp189) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp148*residual_tmp51 - residual_tmp152*residual_tmp43 - residual_tmp195) + ((s_t(1) / s_t(3)))*residual_tmp147*residual_tmp44 - residual_tmp197) + residual_tmp189*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp165 - residual_tmp43*residual_tmp72 + residual_tmp51*residual_tmp74) - residual_tmp196*u1_grad_2 + ((s_t(1) / s_t(3)))*residual_tmp44*residual_tmp83) + residual_tmp189*residual_tmp20) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp161*residual_tmp43 + residual_tmp164*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp160*residual_tmp44) + residual_tmp189*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp185*residual_tmp43 + residual_tmp187*residual_tmp51 + residual_tmp199) + residual_tmp184*residual_tmp198 + residual_tmp200) + residual_tmp178*residual_tmp189);
      const s_t grad_coeff0_2 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp138 - residual_tmp202 + residual_tmp47*residual_tmp85 - residual_tmp50*residual_tmp87) + ((s_t(1) / s_t(3)))*residual_tmp45*residual_tmp91) + residual_tmp201*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp108*residual_tmp47 - residual_tmp113*residual_tmp50 - residual_tmp193) + ((s_t(1) / s_t(3)))*residual_tmp103*residual_tmp45) + residual_tmp201*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp132*residual_tmp50 + residual_tmp137*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp127*residual_tmp45) + residual_tmp201*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp153 - residual_tmp19*residual_tmp50 + residual_tmp28*residual_tmp47) - residual_tmp196*u2_grad_1 + ((s_t(1) / s_t(3)))*residual_tmp39*residual_tmp45) + residual_tmp12*residual_tmp201) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp173*residual_tmp50 + residual_tmp176*residual_tmp47 + residual_tmp195) + residual_tmp172*residual_tmp203 + residual_tmp197) + residual_tmp166*residual_tmp201) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp148*residual_tmp50 + residual_tmp152*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp147*residual_tmp45) + residual_tmp201*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp188 + residual_tmp47*residual_tmp72 - residual_tmp50*residual_tmp74) + residual_tmp196*residual_tmp46 + residual_tmp203*residual_tmp83) + residual_tmp20*residual_tmp201) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp161*residual_tmp47 - residual_tmp164*residual_tmp50 - residual_tmp199) + ((s_t(1) / s_t(3)))*residual_tmp160*residual_tmp45 - residual_tmp200) + residual_tmp201*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp185*residual_tmp47 - residual_tmp187*residual_tmp50) + ((s_t(1) / s_t(3)))*residual_tmp184*residual_tmp45) + residual_tmp178*residual_tmp201);
      const s_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp20*residual_tmp213 + residual_tmp7*residual_tmp87) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp214) + residual_tmp210*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp113*residual_tmp7 - residual_tmp20*residual_tmp236 + residual_tmp237) + residual_tmp229*residual_tmp6 + residual_tmp234*residual_tmp235) + residual_tmp210*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp132*residual_tmp7 - residual_tmp20*residual_tmp227 - residual_tmp228) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp226 - residual_tmp230) + residual_tmp210*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp19*residual_tmp7 - residual_tmp20*residual_tmp205) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp207) + residual_tmp12*residual_tmp210) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp173*residual_tmp7 - residual_tmp194 - residual_tmp20*residual_tmp224 - residual_tmp225) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp222) + residual_tmp166*residual_tmp210) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp148*residual_tmp7 - residual_tmp20*residual_tmp218 + residual_tmp220) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp216) + residual_tmp210*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp20*residual_tmp211 + residual_tmp7*residual_tmp74) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp212) + residual_tmp20*residual_tmp210) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp164*residual_tmp7 - residual_tmp20*residual_tmp232 - residual_tmp233) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp231 - residual_tmp229*u0_grad_2) + residual_tmp210*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp187*residual_tmp7 - residual_tmp20*residual_tmp239 + residual_tmp240) + residual_tmp235*residual_tmp238 + residual_tmp241) + residual_tmp178*residual_tmp210);
      const s_t grad_coeff1_1 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp213*residual_tmp43 + residual_tmp237 + residual_tmp44*residual_tmp87) - residual_tmp214*residual_tmp242 + residual_tmp243*residual_tmp6) + residual_tmp244*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp113*residual_tmp44 + residual_tmp236*residual_tmp43) - residual_tmp234*residual_tmp242) + residual_tmp244*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp132*residual_tmp44 + residual_tmp227*residual_tmp43 - residual_tmp246) - residual_tmp226*residual_tmp242 - residual_tmp243*u2_grad_0) + residual_tmp244*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp116 + residual_tmp19*residual_tmp44 + residual_tmp205*residual_tmp43 - residual_tmp225) - residual_tmp207*residual_tmp242) + residual_tmp12*residual_tmp244) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp173*residual_tmp44 + residual_tmp224*residual_tmp43) - residual_tmp222*residual_tmp242) + residual_tmp166*residual_tmp244) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp148*residual_tmp44 + residual_tmp192 + residual_tmp218*residual_tmp43 + residual_tmp245) - residual_tmp216*residual_tmp242) + residual_tmp244*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp211*residual_tmp43 - residual_tmp233 + residual_tmp44*residual_tmp74) - residual_tmp212*residual_tmp242 - residual_tmp243*u0_grad_2) + residual_tmp20*residual_tmp244) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp164*residual_tmp44 + residual_tmp232*residual_tmp43) - residual_tmp231*residual_tmp242) + residual_tmp244*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp187*residual_tmp44 + residual_tmp239*residual_tmp43 + residual_tmp247) + residual_tmp141*residual_tmp243 - residual_tmp238*residual_tmp242) + residual_tmp178*residual_tmp244);
      const s_t grad_coeff1_2 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp213*residual_tmp47 + residual_tmp228 - residual_tmp45*residual_tmp87) + residual_tmp214*residual_tmp250 + residual_tmp230) + residual_tmp248*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp113*residual_tmp45 + residual_tmp236*residual_tmp47 - residual_tmp246) - residual_tmp229*u2_grad_0 + ((s_t(1) / s_t(3)))*residual_tmp234*residual_tmp50) + residual_tmp248*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp132*residual_tmp45 + residual_tmp227*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp226*residual_tmp50) + residual_tmp248*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp19*residual_tmp45 + residual_tmp205*residual_tmp47 - residual_tmp220) + ((s_t(1) / s_t(3)))*residual_tmp207*residual_tmp50) + residual_tmp12*residual_tmp248) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp173*residual_tmp45 - residual_tmp191 + residual_tmp224*residual_tmp47 - residual_tmp249) + ((s_t(1) / s_t(3)))*residual_tmp222*residual_tmp50) + residual_tmp166*residual_tmp248) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp148*residual_tmp45 + residual_tmp218*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp216*residual_tmp50) + residual_tmp248*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp211*residual_tmp47 - residual_tmp240 - residual_tmp45*residual_tmp74) + ((s_t(1) / s_t(3)))*residual_tmp212*residual_tmp50 - residual_tmp241) + residual_tmp20*residual_tmp248) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp164*residual_tmp45 + residual_tmp232*residual_tmp47 + residual_tmp247) + residual_tmp141*residual_tmp229 + residual_tmp231*residual_tmp250) + residual_tmp248*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp187*residual_tmp45 + residual_tmp239*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp238*residual_tmp50) + residual_tmp178*residual_tmp248);
      const s_t grad_coeff2_0 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp213 + residual_tmp7*residual_tmp85) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp255) + residual_tmp253*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp108*residual_tmp7 - residual_tmp12*residual_tmp236 - residual_tmp260) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp259 - residual_tmp262) + residual_tmp253*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp227 + residual_tmp137*residual_tmp7 + residual_tmp267) + residual_tmp261*residual_tmp46 + residual_tmp265*residual_tmp266) + residual_tmp253*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp205 + residual_tmp28*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp254) + residual_tmp12*residual_tmp253) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp224 + residual_tmp176*residual_tmp7 + residual_tmp269) + residual_tmp266*residual_tmp268 + residual_tmp270) + residual_tmp166*residual_tmp253) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp218 + residual_tmp152*residual_tmp7 - residual_tmp264) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp263 - residual_tmp261*u0_grad_1) + residual_tmp253*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp211 + residual_tmp7*residual_tmp72) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp251) + residual_tmp20*residual_tmp253) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp232 + residual_tmp161*residual_tmp7 + residual_tmp257) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp256) + residual_tmp253*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp239 + residual_tmp185*residual_tmp7 - residual_tmp202 - residual_tmp219) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp258) + residual_tmp178*residual_tmp253);
      const s_t grad_coeff2_1 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp213*residual_tmp51 + residual_tmp260 - residual_tmp44*residual_tmp85) + residual_tmp255*residual_tmp273 + residual_tmp262) + residual_tmp271*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp108*residual_tmp44 + residual_tmp236*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp259*residual_tmp43) + residual_tmp271*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp137*residual_tmp44 + residual_tmp227*residual_tmp51 - residual_tmp272) - residual_tmp261*u1_grad_0 + ((s_t(1) / s_t(3)))*residual_tmp265*residual_tmp43) + residual_tmp271*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp205*residual_tmp51 - residual_tmp269 - residual_tmp28*residual_tmp44) + ((s_t(1) / s_t(3)))*residual_tmp254*residual_tmp43 - residual_tmp270) + residual_tmp12*residual_tmp271) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp176*residual_tmp44 + residual_tmp224*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp268*residual_tmp43) + residual_tmp166*residual_tmp271) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp152*residual_tmp44 + residual_tmp218*residual_tmp51 + residual_tmp274) + residual_tmp141*residual_tmp261 + residual_tmp263*residual_tmp273) + residual_tmp271*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp211*residual_tmp51 - residual_tmp257 - residual_tmp44*residual_tmp72) + ((s_t(1) / s_t(3)))*residual_tmp251*residual_tmp43) + residual_tmp20*residual_tmp271) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp161*residual_tmp44 + residual_tmp232*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp256*residual_tmp43) + residual_tmp271*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp185*residual_tmp44 - residual_tmp190 + residual_tmp239*residual_tmp51 - residual_tmp249) + ((s_t(1) / s_t(3)))*residual_tmp258*residual_tmp43) + residual_tmp178*residual_tmp271);
      const s_t grad_coeff2_2 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp213*residual_tmp50 + residual_tmp267 + residual_tmp45*residual_tmp85) - residual_tmp255*residual_tmp275 + residual_tmp276*residual_tmp46) + residual_tmp277*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp108*residual_tmp45 + residual_tmp236*residual_tmp50 - residual_tmp272) - residual_tmp259*residual_tmp275 - residual_tmp276*u1_grad_0) + residual_tmp277*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp137*residual_tmp45 + residual_tmp227*residual_tmp50) - residual_tmp265*residual_tmp275) + residual_tmp277*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp205*residual_tmp50 - residual_tmp264 + residual_tmp28*residual_tmp45) - residual_tmp254*residual_tmp275 - residual_tmp276*u0_grad_1) + residual_tmp12*residual_tmp277) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp176*residual_tmp45 + residual_tmp224*residual_tmp50 + residual_tmp274) + residual_tmp141*residual_tmp276 - residual_tmp268*residual_tmp275) + residual_tmp166*residual_tmp277) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp152*residual_tmp45 + residual_tmp218*residual_tmp50) - residual_tmp263*residual_tmp275) + residual_tmp277*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp140 + residual_tmp211*residual_tmp50 - residual_tmp219 + residual_tmp45*residual_tmp72) - residual_tmp251*residual_tmp275) + residual_tmp20*residual_tmp277) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp161*residual_tmp45 - residual_tmp190 + residual_tmp232*residual_tmp50 + residual_tmp245) - residual_tmp256*residual_tmp275) + residual_tmp277*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp185*residual_tmp45 + residual_tmp239*residual_tmp50) - residual_tmp258*residual_tmp275) + residual_tmp178*residual_tmp277);
      const s_t grad_coeff0_0_value = grad_coeff0_0;
      const s_t grad_coeff0_1_value = grad_coeff0_1;
      const s_t grad_coeff0_2_value = grad_coeff0_2;
      const s_t grad_coeff1_0_value = grad_coeff1_0;
      const s_t grad_coeff1_1_value = grad_coeff1_1;
      const s_t grad_coeff1_2_value = grad_coeff1_2;
      const s_t grad_coeff2_0_value = grad_coeff2_0;
      const s_t grad_coeff2_1_value = grad_coeff2_1;
      const s_t grad_coeff2_2_value = grad_coeff2_2;
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
      output[0][0] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1 + grad_coeff0_2_value * test0_grad2);
      output[1][0] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1 + grad_coeff1_2_value * test0_grad2);
      output[2][0] += q_weight[q] * det * (grad_coeff2_0_value * test0_grad0 + grad_coeff2_1_value * test0_grad1 + grad_coeff2_2_value * test0_grad2);
      output[3][0] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1 + grad_coeff0_2_value * test1_grad2);
      output[4][0] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1 + grad_coeff1_2_value * test1_grad2);
      output[5][0] += q_weight[q] * det * (grad_coeff2_0_value * test1_grad0 + grad_coeff2_1_value * test1_grad1 + grad_coeff2_2_value * test1_grad2);
      output[6][0] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1 + grad_coeff0_2_value * test2_grad2);
      output[7][0] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1 + grad_coeff1_2_value * test2_grad2);
      output[8][0] += q_weight[q] * det * (grad_coeff2_0_value * test2_grad0 + grad_coeff2_1_value * test2_grad1 + grad_coeff2_2_value * test2_grad2);
      output[9][0] += q_weight[q] * det * (grad_coeff0_0_value * test3_grad0 + grad_coeff0_1_value * test3_grad1 + grad_coeff0_2_value * test3_grad2);
      output[10][0] += q_weight[q] * det * (grad_coeff1_0_value * test3_grad0 + grad_coeff1_1_value * test3_grad1 + grad_coeff1_2_value * test3_grad2);
      output[11][0] += q_weight[q] * det * (grad_coeff2_0_value * test3_grad0 + grad_coeff2_1_value * test3_grad1 + grad_coeff2_2_value * test3_grad2);
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_tet4_jacobian_action_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t current[3 * NS][VS],
    const s_t previous[3 * NS][VS],
    const s_t direction[3 * NS][VS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t newmark_velocity_alpha,
    s_t output[3 * NS][VS]
) {
  for (int q = 0; q < NQ; ++q) {
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
      const s_t u0_direction_grad_0_ref = -(direction[0][0]) + direction[3][0];
      const s_t u0_direction_grad_1_ref = -(direction[0][0]) + direction[6][0];
      const s_t u0_direction_grad_2_ref = -(direction[0][0]) + direction[9][0];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
      const s_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
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
      const s_t u1_direction_grad_0_ref = -(direction[1][0]) + direction[4][0];
      const s_t u1_direction_grad_1_ref = -(direction[1][0]) + direction[7][0];
      const s_t u1_direction_grad_2_ref = -(direction[1][0]) + direction[10][0];
      const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
      const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
      const s_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
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
      const s_t u2_direction_grad_0_ref = -(direction[2][0]) + direction[5][0];
      const s_t u2_direction_grad_1_ref = -(direction[2][0]) + direction[8][0];
      const s_t u2_direction_grad_2_ref = -(direction[2][0]) + direction[11][0];
      const s_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
      const s_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
      const s_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = u0_grad_0*u1_grad_1;
      const s_t residual_tmp1 = u0_grad_1*u1_grad_2;
      const s_t residual_tmp2 = u0_grad_2*u2_grad_1;
      const s_t residual_tmp3 = u1_grad_2*u2_grad_1;
      const s_t residual_tmp4 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp5 = u0_grad_2*u2_grad_0;
      const s_t residual_tmp6 = u2_grad_2 + s_t(1);
      const s_t residual_tmp7 = -residual_tmp3 + residual_tmp6 + u1_grad_1*u2_grad_2 + u1_grad_1;
      const s_t residual_tmp8 = -residual_tmp5 + u0_grad_0*u2_grad_2 + u0_grad_0;
      const s_t residual_tmp9 = residual_tmp0 - residual_tmp4;
      const s_t residual_tmp10 = residual_tmp0*u2_grad_2 + residual_tmp1*u2_grad_0 + residual_tmp2*u1_grad_0 - residual_tmp3*u0_grad_0 - residual_tmp4*u2_grad_2 - residual_tmp5*u1_grad_1 + residual_tmp7 + residual_tmp8 + residual_tmp9;
      const s_t residual_tmp11 = pow_m1(residual_tmp10);
      const s_t residual_tmp12 = -residual_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
      const s_t residual_tmp13 = newmark_velocity_alpha*residual_tmp7;
      const s_t residual_tmp14 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
      const s_t residual_tmp15 = residual_tmp14*u2_grad_1;
      const s_t residual_tmp16 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
      const s_t residual_tmp17 = residual_tmp16*residual_tmp6;
      const s_t residual_tmp18 = residual_tmp15 - residual_tmp17;
      const s_t residual_tmp19 = -residual_tmp13 - residual_tmp18;
      const s_t residual_tmp20 = -residual_tmp1 + u0_grad_2*u1_grad_1 + u0_grad_2;
      const s_t residual_tmp21 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
      const s_t residual_tmp22 = residual_tmp21*residual_tmp6;
      const s_t residual_tmp23 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
      const s_t residual_tmp24 = residual_tmp23*u2_grad_1;
      const s_t residual_tmp25 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
      const s_t residual_tmp26 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
      const s_t residual_tmp27 = residual_tmp25*u0_grad_1 - residual_tmp26*u0_grad_2;
      const s_t residual_tmp28 = residual_tmp22 - residual_tmp24 + residual_tmp27;
      const s_t residual_tmp29 = newmark_velocity_alpha*residual_tmp12;
      const s_t residual_tmp30 = residual_tmp23*u0_grad_1;
      const s_t residual_tmp31 = residual_tmp21*u0_grad_2;
      const s_t residual_tmp32 = residual_tmp29 + residual_tmp30 - residual_tmp31;
      const s_t residual_tmp33 = residual_tmp26*residual_tmp6;
      const s_t residual_tmp34 = residual_tmp25*u2_grad_1;
      const s_t residual_tmp35 = residual_tmp33 - residual_tmp34;
      const s_t residual_tmp36 = s_t(3)*eta_b;
      const s_t residual_tmp37 = residual_tmp36*(residual_tmp32 + residual_tmp35);
      const s_t residual_tmp38 = s_t(2)*eta_s;
      const s_t residual_tmp39 = residual_tmp37 + residual_tmp38*(s_t(2)*residual_tmp26*residual_tmp6 - residual_tmp32 - s_t(2)*residual_tmp34);
      const s_t residual_tmp40 = ((s_t(1) / s_t(3)))*residual_tmp7;
      const s_t residual_tmp41 = pow_m2(residual_tmp10);
      const s_t residual_tmp42 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
      const s_t residual_tmp43 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
      const s_t residual_tmp44 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
      const s_t residual_tmp45 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
      const s_t residual_tmp46 = u1_grad_1 + s_t(1);
      const s_t residual_tmp47 = residual_tmp46 + residual_tmp9 + u0_grad_0;
      const s_t residual_tmp48 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
      const s_t residual_tmp49 = residual_tmp20*residual_tmp42 + residual_tmp21*residual_tmp44 + residual_tmp23*residual_tmp45 - residual_tmp25*residual_tmp47 + residual_tmp26*residual_tmp43 - residual_tmp48*residual_tmp7;
      const s_t residual_tmp50 = u0_grad_0*u2_grad_1 - u0_grad_1*u2_grad_0 + u2_grad_1;
      const s_t residual_tmp51 = residual_tmp6 + residual_tmp8;
      const s_t residual_tmp52 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
      const s_t residual_tmp53 = residual_tmp12*residual_tmp42 + residual_tmp14*residual_tmp45 + residual_tmp16*residual_tmp44 + residual_tmp25*residual_tmp50 - residual_tmp26*residual_tmp51 - residual_tmp52*residual_tmp7;
      const s_t residual_tmp54 = residual_tmp12*residual_tmp52;
      const s_t residual_tmp55 = residual_tmp14*residual_tmp50;
      const s_t residual_tmp56 = residual_tmp20*residual_tmp48;
      const s_t residual_tmp57 = residual_tmp21*residual_tmp43;
      const s_t residual_tmp58 = residual_tmp16*residual_tmp51;
      const s_t residual_tmp59 = -residual_tmp58;
      const s_t residual_tmp60 = residual_tmp23*residual_tmp47;
      const s_t residual_tmp61 = -residual_tmp60;
      const s_t residual_tmp62 = residual_tmp54 + residual_tmp55 + residual_tmp56 + residual_tmp57 + residual_tmp59 + residual_tmp61;
      const s_t residual_tmp63 = residual_tmp42*residual_tmp7;
      const s_t residual_tmp64 = residual_tmp25*residual_tmp45 + residual_tmp26*residual_tmp44 - residual_tmp63;
      const s_t residual_tmp65 = residual_tmp36*(residual_tmp62 + residual_tmp64);
      const s_t residual_tmp66 = residual_tmp38*(s_t(2)*residual_tmp25*residual_tmp45 + s_t(2)*residual_tmp26*residual_tmp44 - residual_tmp62 - s_t(2)*residual_tmp63) + residual_tmp65;
      const s_t residual_tmp67 = -residual_tmp66;
      const s_t residual_tmp68 = residual_tmp41*(eta_s*(residual_tmp12*residual_tmp53 + residual_tmp20*residual_tmp49) + residual_tmp40*residual_tmp67);
      const s_t residual_tmp69 = residual_tmp21*u1_grad_2;
      const s_t residual_tmp70 = residual_tmp23*residual_tmp46;
      const s_t residual_tmp71 = residual_tmp69 - residual_tmp70;
      const s_t residual_tmp72 = -residual_tmp13 - residual_tmp71;
      const s_t residual_tmp73 = -residual_tmp14*residual_tmp46 + residual_tmp16*u1_grad_2;
      const s_t residual_tmp74 = -residual_tmp27 - residual_tmp73;
      const s_t residual_tmp75 = newmark_velocity_alpha*residual_tmp20;
      const s_t residual_tmp76 = residual_tmp16*u0_grad_2;
      const s_t residual_tmp77 = residual_tmp14*u0_grad_1;
      const s_t residual_tmp78 = residual_tmp75 + residual_tmp76 - residual_tmp77;
      const s_t residual_tmp79 = residual_tmp25*residual_tmp46;
      const s_t residual_tmp80 = residual_tmp26*u1_grad_2;
      const s_t residual_tmp81 = residual_tmp79 - residual_tmp80;
      const s_t residual_tmp82 = residual_tmp36*(residual_tmp78 + residual_tmp81);
      const s_t residual_tmp83 = residual_tmp38*(s_t(2)*residual_tmp25*residual_tmp46 - residual_tmp78 - s_t(2)*residual_tmp80) + residual_tmp82;
      const s_t residual_tmp84 = -residual_tmp79 + residual_tmp80;
      const s_t residual_tmp85 = residual_tmp75 + residual_tmp84;
      const s_t residual_tmp86 = -residual_tmp33 + residual_tmp34;
      const s_t residual_tmp87 = residual_tmp29 + residual_tmp86;
      const s_t residual_tmp88 = residual_tmp13 - residual_tmp69 + residual_tmp70;
      const s_t residual_tmp89 = -residual_tmp15 + residual_tmp17;
      const s_t residual_tmp90 = residual_tmp36*(-residual_tmp88 - residual_tmp89);
      const s_t residual_tmp91 = residual_tmp38*(-s_t(2)*residual_tmp13 - residual_tmp18 - residual_tmp71) + residual_tmp90;
      const s_t residual_tmp92 = -residual_tmp7;
      const s_t residual_tmp93 = newmark_velocity_alpha*residual_tmp44;
      const s_t residual_tmp94 = residual_tmp23*u1_grad_0;
      const s_t residual_tmp95 = residual_tmp48*u1_grad_2;
      const s_t residual_tmp96 = residual_tmp93 + residual_tmp94 - residual_tmp95;
      const s_t residual_tmp97 = residual_tmp52*residual_tmp6;
      const s_t residual_tmp98 = residual_tmp14*u2_grad_0;
      const s_t residual_tmp99 = residual_tmp97 - residual_tmp98;
      const s_t residual_tmp100 = residual_tmp36*(residual_tmp96 + residual_tmp99);
      const s_t residual_tmp101 = -residual_tmp97 + residual_tmp98;
      const s_t residual_tmp102 = -residual_tmp94 + residual_tmp95;
      const s_t residual_tmp103 = residual_tmp100 + residual_tmp38*(residual_tmp101 + residual_tmp102 + s_t(2)*residual_tmp93);
      const s_t residual_tmp104 = newmark_velocity_alpha*residual_tmp43;
      const s_t residual_tmp105 = residual_tmp25*u1_grad_0;
      const s_t residual_tmp106 = residual_tmp42*u1_grad_2;
      const s_t residual_tmp107 = residual_tmp105 - residual_tmp106;
      const s_t residual_tmp108 = residual_tmp104 + residual_tmp107;
      const s_t residual_tmp109 = newmark_velocity_alpha*residual_tmp51;
      const s_t residual_tmp110 = residual_tmp25*u2_grad_0;
      const s_t residual_tmp111 = residual_tmp42*residual_tmp6;
      const s_t residual_tmp112 = residual_tmp110 - residual_tmp111;
      const s_t residual_tmp113 = -residual_tmp109 - residual_tmp112;
      const s_t residual_tmp114 = residual_tmp49*u1_grad_2;
      const s_t residual_tmp115 = -residual_tmp114;
      const s_t residual_tmp116 = residual_tmp53*residual_tmp6;
      const s_t residual_tmp117 = newmark_velocity_alpha*residual_tmp45;
      const s_t residual_tmp118 = residual_tmp46*residual_tmp48;
      const s_t residual_tmp119 = residual_tmp21*u1_grad_0;
      const s_t residual_tmp120 = residual_tmp117 + residual_tmp118 - residual_tmp119;
      const s_t residual_tmp121 = residual_tmp16*u2_grad_0;
      const s_t residual_tmp122 = residual_tmp52*u2_grad_1;
      const s_t residual_tmp123 = residual_tmp121 - residual_tmp122;
      const s_t residual_tmp124 = residual_tmp36*(residual_tmp120 + residual_tmp123);
      const s_t residual_tmp125 = -residual_tmp121 + residual_tmp122;
      const s_t residual_tmp126 = -residual_tmp118 + residual_tmp119;
      const s_t residual_tmp127 = residual_tmp124 + residual_tmp38*(s_t(2)*residual_tmp117 + residual_tmp125 + residual_tmp126);
      const s_t residual_tmp128 = newmark_velocity_alpha*residual_tmp50;
      const s_t residual_tmp129 = residual_tmp26*u2_grad_0;
      const s_t residual_tmp130 = residual_tmp42*u2_grad_1;
      const s_t residual_tmp131 = residual_tmp129 - residual_tmp130;
      const s_t residual_tmp132 = residual_tmp128 + residual_tmp131;
      const s_t residual_tmp133 = newmark_velocity_alpha*residual_tmp47;
      const s_t residual_tmp134 = residual_tmp26*u1_grad_0;
      const s_t residual_tmp135 = residual_tmp42*residual_tmp46;
      const s_t residual_tmp136 = residual_tmp134 - residual_tmp135;
      const s_t residual_tmp137 = -residual_tmp133 - residual_tmp136;
      const s_t residual_tmp138 = residual_tmp53*u2_grad_1;
      const s_t residual_tmp139 = -residual_tmp138;
      const s_t residual_tmp140 = residual_tmp46*residual_tmp49;
      const s_t residual_tmp141 = u0_grad_0 + s_t(1);
      const s_t residual_tmp142 = residual_tmp141*residual_tmp21;
      const s_t residual_tmp143 = residual_tmp48*u0_grad_1;
      const s_t residual_tmp144 = residual_tmp128 + residual_tmp142 - residual_tmp143;
      const s_t residual_tmp145 = -residual_tmp129 + residual_tmp130;
      const s_t residual_tmp146 = residual_tmp36*(residual_tmp144 + residual_tmp145);
      const s_t residual_tmp147 = residual_tmp146 + residual_tmp38*(-s_t(2)*residual_tmp129 - residual_tmp144 + s_t(2)*residual_tmp42*u2_grad_1);
      const s_t residual_tmp148 = residual_tmp117 + residual_tmp125;
      const s_t residual_tmp149 = residual_tmp21*u2_grad_0;
      const s_t residual_tmp150 = residual_tmp48*u2_grad_1;
      const s_t residual_tmp151 = -residual_tmp141*residual_tmp26 + residual_tmp42*u0_grad_1;
      const s_t residual_tmp152 = -residual_tmp149 + residual_tmp150 - residual_tmp151;
      const s_t residual_tmp153 = residual_tmp49*u0_grad_1;
      const s_t residual_tmp154 = ((s_t(1) / s_t(3)))*residual_tmp67;
      const s_t residual_tmp155 = residual_tmp14*residual_tmp141;
      const s_t residual_tmp156 = residual_tmp52*u0_grad_2;
      const s_t residual_tmp157 = residual_tmp104 + residual_tmp155 - residual_tmp156;
      const s_t residual_tmp158 = -residual_tmp105 + residual_tmp106;
      const s_t residual_tmp159 = residual_tmp36*(residual_tmp157 + residual_tmp158);
      const s_t residual_tmp160 = residual_tmp159 + residual_tmp38*(-s_t(2)*residual_tmp105 - residual_tmp157 + s_t(2)*residual_tmp42*u1_grad_2);
      const s_t residual_tmp161 = residual_tmp102 + residual_tmp93;
      const s_t residual_tmp162 = -residual_tmp141*residual_tmp25 + residual_tmp42*u0_grad_2;
      const s_t residual_tmp163 = residual_tmp14*u1_grad_0 - residual_tmp52*u1_grad_2;
      const s_t residual_tmp164 = -residual_tmp162 - residual_tmp163;
      const s_t residual_tmp165 = residual_tmp53*u0_grad_2;
      const s_t residual_tmp166 = -residual_tmp51;
      const s_t residual_tmp167 = residual_tmp141*residual_tmp23;
      const s_t residual_tmp168 = residual_tmp48*u0_grad_2;
      const s_t residual_tmp169 = residual_tmp109 + residual_tmp167 - residual_tmp168;
      const s_t residual_tmp170 = -residual_tmp110 + residual_tmp111;
      const s_t residual_tmp171 = residual_tmp36*(-residual_tmp169 - residual_tmp170);
      const s_t residual_tmp172 = residual_tmp171 + residual_tmp38*(s_t(2)*residual_tmp110 - s_t(2)*residual_tmp111 + residual_tmp169);
      const s_t residual_tmp173 = residual_tmp101 + residual_tmp93;
      const s_t residual_tmp174 = residual_tmp23*u2_grad_0;
      const s_t residual_tmp175 = residual_tmp48*residual_tmp6;
      const s_t residual_tmp176 = residual_tmp162 + residual_tmp174 - residual_tmp175;
      const s_t residual_tmp177 = residual_tmp49*u0_grad_2;
      const s_t residual_tmp178 = -residual_tmp47;
      const s_t residual_tmp179 = residual_tmp141*residual_tmp16;
      const s_t residual_tmp180 = residual_tmp52*u0_grad_1;
      const s_t residual_tmp181 = residual_tmp133 + residual_tmp179 - residual_tmp180;
      const s_t residual_tmp182 = -residual_tmp134 + residual_tmp135;
      const s_t residual_tmp183 = residual_tmp36*(-residual_tmp181 - residual_tmp182);
      const s_t residual_tmp184 = residual_tmp183 + residual_tmp38*(s_t(2)*residual_tmp134 - s_t(2)*residual_tmp135 + residual_tmp181);
      const s_t residual_tmp185 = residual_tmp117 + residual_tmp126;
      const s_t residual_tmp186 = residual_tmp16*u1_grad_0 - residual_tmp46*residual_tmp52;
      const s_t residual_tmp187 = residual_tmp151 + residual_tmp186;
      const s_t residual_tmp188 = residual_tmp53*u0_grad_1;
      const s_t residual_tmp189 = residual_tmp41*(-eta_s*(-residual_tmp43*residual_tmp49 + residual_tmp51*residual_tmp53) + ((s_t(1) / s_t(3)))*residual_tmp44*residual_tmp66);
      const s_t residual_tmp190 = residual_tmp49*u1_grad_0;
      const s_t residual_tmp191 = residual_tmp53*u2_grad_0;
      const s_t residual_tmp192 = -residual_tmp191;
      const s_t residual_tmp193 = residual_tmp190 + residual_tmp192;
      const s_t residual_tmp194 = -residual_tmp53*residual_tmp6;
      const s_t residual_tmp195 = residual_tmp141*residual_tmp49;
      const s_t residual_tmp196 = ((s_t(1) / s_t(3)))*residual_tmp66;
      const s_t residual_tmp197 = residual_tmp196*u2_grad_0;
      const s_t residual_tmp198 = ((s_t(1) / s_t(3)))*residual_tmp44;
      const s_t residual_tmp199 = residual_tmp141*residual_tmp53;
      const s_t residual_tmp200 = residual_tmp196*u1_grad_0;
      const s_t residual_tmp201 = residual_tmp41*(-eta_s*(residual_tmp47*residual_tmp49 - residual_tmp50*residual_tmp53) + ((s_t(1) / s_t(3)))*residual_tmp45*residual_tmp66);
      const s_t residual_tmp202 = -residual_tmp46*residual_tmp49;
      const s_t residual_tmp203 = ((s_t(1) / s_t(3)))*residual_tmp45;
      const s_t residual_tmp204 = -residual_tmp76 + residual_tmp77;
      const s_t residual_tmp205 = residual_tmp204 + residual_tmp75;
      const s_t residual_tmp206 = -residual_tmp30 + residual_tmp31;
      const s_t residual_tmp207 = residual_tmp37 + residual_tmp38*(residual_tmp206 + s_t(2)*residual_tmp29 + residual_tmp86);
      const s_t residual_tmp208 = residual_tmp12*residual_tmp48 - residual_tmp14*residual_tmp47 + residual_tmp16*residual_tmp43 + residual_tmp20*residual_tmp52 - residual_tmp21*residual_tmp51 + residual_tmp23*residual_tmp50;
      const s_t residual_tmp209 = residual_tmp38*(s_t(2)*residual_tmp12*residual_tmp52 + s_t(2)*residual_tmp14*residual_tmp50 - residual_tmp56 - residual_tmp57 - s_t(2)*residual_tmp58 - residual_tmp61 - residual_tmp64) + residual_tmp65;
      const s_t residual_tmp210 = residual_tmp41*(-eta_s*(-residual_tmp20*residual_tmp208 + residual_tmp53*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp209);
      const s_t residual_tmp211 = residual_tmp206 + residual_tmp29;
      const s_t residual_tmp212 = residual_tmp38*(s_t(2)*residual_tmp16*u0_grad_2 - residual_tmp75 - s_t(2)*residual_tmp77 - residual_tmp81) + residual_tmp82;
      const s_t residual_tmp213 = -residual_tmp22 + residual_tmp24 + residual_tmp73;
      const s_t residual_tmp214 = residual_tmp38*(s_t(2)*residual_tmp15 - s_t(2)*residual_tmp17 + residual_tmp88) + residual_tmp90;
      const s_t residual_tmp215 = -residual_tmp142 + residual_tmp143;
      const s_t residual_tmp216 = residual_tmp146 + residual_tmp38*(s_t(2)*residual_tmp128 + residual_tmp131 + residual_tmp215);
      const s_t residual_tmp217 = -residual_tmp179 + residual_tmp180;
      const s_t residual_tmp218 = -residual_tmp133 - residual_tmp217;
      const s_t residual_tmp219 = residual_tmp208*u0_grad_1;
      const s_t residual_tmp220 = residual_tmp139 + residual_tmp219;
      const s_t residual_tmp221 = -residual_tmp167 + residual_tmp168;
      const s_t residual_tmp222 = residual_tmp171 + residual_tmp38*(-s_t(2)*residual_tmp109 - residual_tmp112 - residual_tmp221);
      const s_t residual_tmp223 = -residual_tmp155 + residual_tmp156;
      const s_t residual_tmp224 = residual_tmp104 + residual_tmp223;
      const s_t residual_tmp225 = residual_tmp208*u0_grad_2;
      const s_t residual_tmp226 = residual_tmp124 + residual_tmp38*(-residual_tmp120 - s_t(2)*residual_tmp122 + s_t(2)*residual_tmp16*u2_grad_0);
      const s_t residual_tmp227 = residual_tmp149 - residual_tmp150 - residual_tmp186;
      const s_t residual_tmp228 = residual_tmp208*residual_tmp46;
      const s_t residual_tmp229 = ((s_t(1) / s_t(3)))*residual_tmp209;
      const s_t residual_tmp230 = residual_tmp229*u2_grad_1;
      const s_t residual_tmp231 = residual_tmp159 + residual_tmp38*(-residual_tmp104 + s_t(2)*residual_tmp14*residual_tmp141 - s_t(2)*residual_tmp156 - residual_tmp158);
      const s_t residual_tmp232 = -residual_tmp109 - residual_tmp221;
      const s_t residual_tmp233 = residual_tmp53*u1_grad_2;
      const s_t residual_tmp234 = residual_tmp100 + residual_tmp38*(s_t(2)*residual_tmp52*residual_tmp6 - residual_tmp96 - s_t(2)*residual_tmp98);
      const s_t residual_tmp235 = ((s_t(1) / s_t(3)))*residual_tmp12;
      const s_t residual_tmp236 = residual_tmp163 - residual_tmp174 + residual_tmp175;
      const s_t residual_tmp237 = residual_tmp208*u1_grad_2;
      const s_t residual_tmp238 = residual_tmp183 + residual_tmp38*(residual_tmp133 - s_t(2)*residual_tmp179 + s_t(2)*residual_tmp180 + residual_tmp182);
      const s_t residual_tmp239 = residual_tmp128 + residual_tmp215;
      const s_t residual_tmp240 = residual_tmp46*residual_tmp53;
      const s_t residual_tmp241 = residual_tmp229*u0_grad_1;
      const s_t residual_tmp242 = ((s_t(1) / s_t(3)))*residual_tmp51;
      const s_t residual_tmp243 = -(s_t(1) / s_t(3))*residual_tmp209;
      const s_t residual_tmp244 = residual_tmp41*(eta_s*(residual_tmp208*residual_tmp43 + residual_tmp44*residual_tmp53) + residual_tmp243*residual_tmp51);
      const s_t residual_tmp245 = residual_tmp141*residual_tmp208;
      const s_t residual_tmp246 = residual_tmp208*u1_grad_0;
      const s_t residual_tmp247 = residual_tmp53*u1_grad_0;
      const s_t residual_tmp248 = residual_tmp41*(-eta_s*(residual_tmp208*residual_tmp47 - residual_tmp45*residual_tmp53) + ((s_t(1) / s_t(3)))*residual_tmp209*residual_tmp50);
      const s_t residual_tmp249 = -residual_tmp141*residual_tmp208;
      const s_t residual_tmp250 = ((s_t(1) / s_t(3)))*residual_tmp50;
      const s_t residual_tmp251 = residual_tmp38*(residual_tmp204 + s_t(2)*residual_tmp75 + residual_tmp84) + residual_tmp82;
      const s_t residual_tmp252 = residual_tmp38*(s_t(2)*residual_tmp20*residual_tmp48 + s_t(2)*residual_tmp21*residual_tmp43 - residual_tmp54 - residual_tmp55 - residual_tmp59 - s_t(2)*residual_tmp60 - residual_tmp64) + residual_tmp65;
      const s_t residual_tmp253 = residual_tmp41*(-eta_s*(-residual_tmp12*residual_tmp208 + residual_tmp49*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp252);
      const s_t residual_tmp254 = residual_tmp37 + residual_tmp38*(s_t(2)*residual_tmp23*u0_grad_1 - residual_tmp29 - s_t(2)*residual_tmp31 - residual_tmp35);
      const s_t residual_tmp255 = residual_tmp38*(residual_tmp13 + s_t(2)*residual_tmp69 - s_t(2)*residual_tmp70 + residual_tmp89) + residual_tmp90;
      const s_t residual_tmp256 = residual_tmp159 + residual_tmp38*(s_t(2)*residual_tmp104 + residual_tmp107 + residual_tmp223);
      const s_t residual_tmp257 = residual_tmp115 + residual_tmp225;
      const s_t residual_tmp258 = residual_tmp183 + residual_tmp38*(-s_t(2)*residual_tmp133 - residual_tmp136 - residual_tmp217);
      const s_t residual_tmp259 = residual_tmp100 + residual_tmp38*(s_t(2)*residual_tmp23*u1_grad_0 - residual_tmp93 - s_t(2)*residual_tmp95 - residual_tmp99);
      const s_t residual_tmp260 = residual_tmp208*residual_tmp6;
      const s_t residual_tmp261 = ((s_t(1) / s_t(3)))*residual_tmp252;
      const s_t residual_tmp262 = residual_tmp261*u1_grad_2;
      const s_t residual_tmp263 = residual_tmp146 + residual_tmp38*(-residual_tmp128 + s_t(2)*residual_tmp141*residual_tmp21 - s_t(2)*residual_tmp143 - residual_tmp145);
      const s_t residual_tmp264 = residual_tmp49*u2_grad_1;
      const s_t residual_tmp265 = residual_tmp124 + residual_tmp38*(-residual_tmp117 - s_t(2)*residual_tmp119 - residual_tmp123 + s_t(2)*residual_tmp46*residual_tmp48);
      const s_t residual_tmp266 = ((s_t(1) / s_t(3)))*residual_tmp20;
      const s_t residual_tmp267 = residual_tmp208*u2_grad_1;
      const s_t residual_tmp268 = residual_tmp171 + residual_tmp38*(residual_tmp109 - s_t(2)*residual_tmp167 + s_t(2)*residual_tmp168 + residual_tmp170);
      const s_t residual_tmp269 = residual_tmp49*residual_tmp6;
      const s_t residual_tmp270 = residual_tmp261*u0_grad_2;
      const s_t residual_tmp271 = residual_tmp41*(-eta_s*(residual_tmp208*residual_tmp51 - residual_tmp44*residual_tmp49) + ((s_t(1) / s_t(3)))*residual_tmp252*residual_tmp43);
      const s_t residual_tmp272 = residual_tmp208*u2_grad_0;
      const s_t residual_tmp273 = ((s_t(1) / s_t(3)))*residual_tmp43;
      const s_t residual_tmp274 = residual_tmp49*u2_grad_0;
      const s_t residual_tmp275 = ((s_t(1) / s_t(3)))*residual_tmp47;
      const s_t residual_tmp276 = -(s_t(1) / s_t(3))*residual_tmp252;
      const s_t residual_tmp277 = residual_tmp41*(eta_s*(residual_tmp208*residual_tmp50 + residual_tmp45*residual_tmp49) + residual_tmp276*residual_tmp47);
      const s_t grad_coeff0_0 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp87 + residual_tmp20*residual_tmp85) - residual_tmp40*residual_tmp91) + residual_tmp68*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp108*residual_tmp20 + residual_tmp113*residual_tmp12 + residual_tmp115 + residual_tmp116) - residual_tmp103*residual_tmp40) + residual_tmp44*residual_tmp68) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp132 + residual_tmp137*residual_tmp20 + residual_tmp139 + residual_tmp140) - residual_tmp127*residual_tmp40) + residual_tmp45*residual_tmp68) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp19 + residual_tmp20*residual_tmp28) - residual_tmp39*residual_tmp40) + residual_tmp12*residual_tmp68) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp173 + residual_tmp176*residual_tmp20 + residual_tmp177) + residual_tmp154*residual_tmp6 - residual_tmp172*residual_tmp40) + residual_tmp166*residual_tmp68) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp148 + residual_tmp152*residual_tmp20 - residual_tmp153) - residual_tmp147*residual_tmp40 - residual_tmp154*u2_grad_1) + residual_tmp50*residual_tmp68) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp74 + residual_tmp20*residual_tmp72) - residual_tmp40*residual_tmp83) + residual_tmp20*residual_tmp68) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp164 + residual_tmp161*residual_tmp20 - residual_tmp165) - residual_tmp154*u1_grad_2 - residual_tmp160*residual_tmp40) + residual_tmp43*residual_tmp68) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp187 + residual_tmp185*residual_tmp20 + residual_tmp188) + residual_tmp154*residual_tmp46 - residual_tmp184*residual_tmp40) + residual_tmp178*residual_tmp68);
      const s_t grad_coeff0_1 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp114 - residual_tmp194 - residual_tmp43*residual_tmp85 + residual_tmp51*residual_tmp87) + ((s_t(1) / s_t(3)))*residual_tmp44*residual_tmp91) + residual_tmp189*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp108*residual_tmp43 + residual_tmp113*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp103*residual_tmp44) + residual_tmp189*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp132*residual_tmp51 - residual_tmp137*residual_tmp43 + residual_tmp193) + ((s_t(1) / s_t(3)))*residual_tmp127*residual_tmp44) + residual_tmp189*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp177 + residual_tmp19*residual_tmp51 - residual_tmp28*residual_tmp43) + residual_tmp196*residual_tmp6 + residual_tmp198*residual_tmp39) + residual_tmp12*residual_tmp189) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp173*residual_tmp51 - residual_tmp176*residual_tmp43) + ((s_t(1) / s_t(3)))*residual_tmp172*residual_tmp44) + residual_tmp166*residual_tmp189) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp148*residual_tmp51 - residual_tmp152*residual_tmp43 - residual_tmp195) + ((s_t(1) / s_t(3)))*residual_tmp147*residual_tmp44 - residual_tmp197) + residual_tmp189*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp165 - residual_tmp43*residual_tmp72 + residual_tmp51*residual_tmp74) - residual_tmp196*u1_grad_2 + ((s_t(1) / s_t(3)))*residual_tmp44*residual_tmp83) + residual_tmp189*residual_tmp20) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp161*residual_tmp43 + residual_tmp164*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp160*residual_tmp44) + residual_tmp189*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp185*residual_tmp43 + residual_tmp187*residual_tmp51 + residual_tmp199) + residual_tmp184*residual_tmp198 + residual_tmp200) + residual_tmp178*residual_tmp189);
      const s_t grad_coeff0_2 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp138 - residual_tmp202 + residual_tmp47*residual_tmp85 - residual_tmp50*residual_tmp87) + ((s_t(1) / s_t(3)))*residual_tmp45*residual_tmp91) + residual_tmp201*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp108*residual_tmp47 - residual_tmp113*residual_tmp50 - residual_tmp193) + ((s_t(1) / s_t(3)))*residual_tmp103*residual_tmp45) + residual_tmp201*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp132*residual_tmp50 + residual_tmp137*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp127*residual_tmp45) + residual_tmp201*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp153 - residual_tmp19*residual_tmp50 + residual_tmp28*residual_tmp47) - residual_tmp196*u2_grad_1 + ((s_t(1) / s_t(3)))*residual_tmp39*residual_tmp45) + residual_tmp12*residual_tmp201) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp173*residual_tmp50 + residual_tmp176*residual_tmp47 + residual_tmp195) + residual_tmp172*residual_tmp203 + residual_tmp197) + residual_tmp166*residual_tmp201) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp148*residual_tmp50 + residual_tmp152*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp147*residual_tmp45) + residual_tmp201*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp188 + residual_tmp47*residual_tmp72 - residual_tmp50*residual_tmp74) + residual_tmp196*residual_tmp46 + residual_tmp203*residual_tmp83) + residual_tmp20*residual_tmp201) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp161*residual_tmp47 - residual_tmp164*residual_tmp50 - residual_tmp199) + ((s_t(1) / s_t(3)))*residual_tmp160*residual_tmp45 - residual_tmp200) + residual_tmp201*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp185*residual_tmp47 - residual_tmp187*residual_tmp50) + ((s_t(1) / s_t(3)))*residual_tmp184*residual_tmp45) + residual_tmp178*residual_tmp201);
      const s_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp20*residual_tmp213 + residual_tmp7*residual_tmp87) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp214) + residual_tmp210*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp113*residual_tmp7 - residual_tmp20*residual_tmp236 + residual_tmp237) + residual_tmp229*residual_tmp6 + residual_tmp234*residual_tmp235) + residual_tmp210*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp132*residual_tmp7 - residual_tmp20*residual_tmp227 - residual_tmp228) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp226 - residual_tmp230) + residual_tmp210*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp19*residual_tmp7 - residual_tmp20*residual_tmp205) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp207) + residual_tmp12*residual_tmp210) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp173*residual_tmp7 - residual_tmp194 - residual_tmp20*residual_tmp224 - residual_tmp225) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp222) + residual_tmp166*residual_tmp210) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp148*residual_tmp7 - residual_tmp20*residual_tmp218 + residual_tmp220) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp216) + residual_tmp210*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp20*residual_tmp211 + residual_tmp7*residual_tmp74) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp212) + residual_tmp20*residual_tmp210) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp164*residual_tmp7 - residual_tmp20*residual_tmp232 - residual_tmp233) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp231 - residual_tmp229*u0_grad_2) + residual_tmp210*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp187*residual_tmp7 - residual_tmp20*residual_tmp239 + residual_tmp240) + residual_tmp235*residual_tmp238 + residual_tmp241) + residual_tmp178*residual_tmp210);
      const s_t grad_coeff1_1 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp213*residual_tmp43 + residual_tmp237 + residual_tmp44*residual_tmp87) - residual_tmp214*residual_tmp242 + residual_tmp243*residual_tmp6) + residual_tmp244*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp113*residual_tmp44 + residual_tmp236*residual_tmp43) - residual_tmp234*residual_tmp242) + residual_tmp244*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp132*residual_tmp44 + residual_tmp227*residual_tmp43 - residual_tmp246) - residual_tmp226*residual_tmp242 - residual_tmp243*u2_grad_0) + residual_tmp244*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp116 + residual_tmp19*residual_tmp44 + residual_tmp205*residual_tmp43 - residual_tmp225) - residual_tmp207*residual_tmp242) + residual_tmp12*residual_tmp244) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp173*residual_tmp44 + residual_tmp224*residual_tmp43) - residual_tmp222*residual_tmp242) + residual_tmp166*residual_tmp244) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp148*residual_tmp44 + residual_tmp192 + residual_tmp218*residual_tmp43 + residual_tmp245) - residual_tmp216*residual_tmp242) + residual_tmp244*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp211*residual_tmp43 - residual_tmp233 + residual_tmp44*residual_tmp74) - residual_tmp212*residual_tmp242 - residual_tmp243*u0_grad_2) + residual_tmp20*residual_tmp244) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp164*residual_tmp44 + residual_tmp232*residual_tmp43) - residual_tmp231*residual_tmp242) + residual_tmp244*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp187*residual_tmp44 + residual_tmp239*residual_tmp43 + residual_tmp247) + residual_tmp141*residual_tmp243 - residual_tmp238*residual_tmp242) + residual_tmp178*residual_tmp244);
      const s_t grad_coeff1_2 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp213*residual_tmp47 + residual_tmp228 - residual_tmp45*residual_tmp87) + residual_tmp214*residual_tmp250 + residual_tmp230) + residual_tmp248*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp113*residual_tmp45 + residual_tmp236*residual_tmp47 - residual_tmp246) - residual_tmp229*u2_grad_0 + ((s_t(1) / s_t(3)))*residual_tmp234*residual_tmp50) + residual_tmp248*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp132*residual_tmp45 + residual_tmp227*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp226*residual_tmp50) + residual_tmp248*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp19*residual_tmp45 + residual_tmp205*residual_tmp47 - residual_tmp220) + ((s_t(1) / s_t(3)))*residual_tmp207*residual_tmp50) + residual_tmp12*residual_tmp248) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp173*residual_tmp45 - residual_tmp191 + residual_tmp224*residual_tmp47 - residual_tmp249) + ((s_t(1) / s_t(3)))*residual_tmp222*residual_tmp50) + residual_tmp166*residual_tmp248) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp148*residual_tmp45 + residual_tmp218*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp216*residual_tmp50) + residual_tmp248*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp211*residual_tmp47 - residual_tmp240 - residual_tmp45*residual_tmp74) + ((s_t(1) / s_t(3)))*residual_tmp212*residual_tmp50 - residual_tmp241) + residual_tmp20*residual_tmp248) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp164*residual_tmp45 + residual_tmp232*residual_tmp47 + residual_tmp247) + residual_tmp141*residual_tmp229 + residual_tmp231*residual_tmp250) + residual_tmp248*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp187*residual_tmp45 + residual_tmp239*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp238*residual_tmp50) + residual_tmp178*residual_tmp248);
      const s_t grad_coeff2_0 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp213 + residual_tmp7*residual_tmp85) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp255) + residual_tmp253*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp108*residual_tmp7 - residual_tmp12*residual_tmp236 - residual_tmp260) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp259 - residual_tmp262) + residual_tmp253*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp227 + residual_tmp137*residual_tmp7 + residual_tmp267) + residual_tmp261*residual_tmp46 + residual_tmp265*residual_tmp266) + residual_tmp253*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp205 + residual_tmp28*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp254) + residual_tmp12*residual_tmp253) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp224 + residual_tmp176*residual_tmp7 + residual_tmp269) + residual_tmp266*residual_tmp268 + residual_tmp270) + residual_tmp166*residual_tmp253) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp218 + residual_tmp152*residual_tmp7 - residual_tmp264) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp263 - residual_tmp261*u0_grad_1) + residual_tmp253*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp211 + residual_tmp7*residual_tmp72) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp251) + residual_tmp20*residual_tmp253) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp232 + residual_tmp161*residual_tmp7 + residual_tmp257) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp256) + residual_tmp253*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp239 + residual_tmp185*residual_tmp7 - residual_tmp202 - residual_tmp219) + ((s_t(1) / s_t(3)))*residual_tmp20*residual_tmp258) + residual_tmp178*residual_tmp253);
      const s_t grad_coeff2_1 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp213*residual_tmp51 + residual_tmp260 - residual_tmp44*residual_tmp85) + residual_tmp255*residual_tmp273 + residual_tmp262) + residual_tmp271*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp108*residual_tmp44 + residual_tmp236*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp259*residual_tmp43) + residual_tmp271*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp137*residual_tmp44 + residual_tmp227*residual_tmp51 - residual_tmp272) - residual_tmp261*u1_grad_0 + ((s_t(1) / s_t(3)))*residual_tmp265*residual_tmp43) + residual_tmp271*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp205*residual_tmp51 - residual_tmp269 - residual_tmp28*residual_tmp44) + ((s_t(1) / s_t(3)))*residual_tmp254*residual_tmp43 - residual_tmp270) + residual_tmp12*residual_tmp271) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp176*residual_tmp44 + residual_tmp224*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp268*residual_tmp43) + residual_tmp166*residual_tmp271) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp152*residual_tmp44 + residual_tmp218*residual_tmp51 + residual_tmp274) + residual_tmp141*residual_tmp261 + residual_tmp263*residual_tmp273) + residual_tmp271*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp211*residual_tmp51 - residual_tmp257 - residual_tmp44*residual_tmp72) + ((s_t(1) / s_t(3)))*residual_tmp251*residual_tmp43) + residual_tmp20*residual_tmp271) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp161*residual_tmp44 + residual_tmp232*residual_tmp51) + ((s_t(1) / s_t(3)))*residual_tmp256*residual_tmp43) + residual_tmp271*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp185*residual_tmp44 - residual_tmp190 + residual_tmp239*residual_tmp51 - residual_tmp249) + ((s_t(1) / s_t(3)))*residual_tmp258*residual_tmp43) + residual_tmp178*residual_tmp271);
      const s_t grad_coeff2_2 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp213*residual_tmp50 + residual_tmp267 + residual_tmp45*residual_tmp85) - residual_tmp255*residual_tmp275 + residual_tmp276*residual_tmp46) + residual_tmp277*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp108*residual_tmp45 + residual_tmp236*residual_tmp50 - residual_tmp272) - residual_tmp259*residual_tmp275 - residual_tmp276*u1_grad_0) + residual_tmp277*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp137*residual_tmp45 + residual_tmp227*residual_tmp50) - residual_tmp265*residual_tmp275) + residual_tmp277*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp205*residual_tmp50 - residual_tmp264 + residual_tmp28*residual_tmp45) - residual_tmp254*residual_tmp275 - residual_tmp276*u0_grad_1) + residual_tmp12*residual_tmp277) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp176*residual_tmp45 + residual_tmp224*residual_tmp50 + residual_tmp274) + residual_tmp141*residual_tmp276 - residual_tmp268*residual_tmp275) + residual_tmp166*residual_tmp277) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp152*residual_tmp45 + residual_tmp218*residual_tmp50) - residual_tmp263*residual_tmp275) + residual_tmp277*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp140 + residual_tmp211*residual_tmp50 - residual_tmp219 + residual_tmp45*residual_tmp72) - residual_tmp251*residual_tmp275) + residual_tmp20*residual_tmp277) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp161*residual_tmp45 - residual_tmp190 + residual_tmp232*residual_tmp50 + residual_tmp245) - residual_tmp256*residual_tmp275) + residual_tmp277*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp185*residual_tmp45 + residual_tmp239*residual_tmp50) - residual_tmp258*residual_tmp275) + residual_tmp178*residual_tmp277);
      const s_t grad_coeff0_0_value = grad_coeff0_0;
      const s_t grad_coeff0_1_value = grad_coeff0_1;
      const s_t grad_coeff0_2_value = grad_coeff0_2;
      const s_t grad_coeff1_0_value = grad_coeff1_0;
      const s_t grad_coeff1_1_value = grad_coeff1_1;
      const s_t grad_coeff1_2_value = grad_coeff1_2;
      const s_t grad_coeff2_0_value = grad_coeff2_0;
      const s_t grad_coeff2_1_value = grad_coeff2_1;
      const s_t grad_coeff2_2_value = grad_coeff2_2;
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
      output[0][0] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1 + grad_coeff0_2_value * test0_grad2);
      output[1][0] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1 + grad_coeff1_2_value * test0_grad2);
      output[2][0] += q_weight[q] * det * (grad_coeff2_0_value * test0_grad0 + grad_coeff2_1_value * test0_grad1 + grad_coeff2_2_value * test0_grad2);
      output[3][0] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1 + grad_coeff0_2_value * test1_grad2);
      output[4][0] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1 + grad_coeff1_2_value * test1_grad2);
      output[5][0] += q_weight[q] * det * (grad_coeff2_0_value * test1_grad0 + grad_coeff2_1_value * test1_grad1 + grad_coeff2_2_value * test1_grad2);
      output[6][0] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1 + grad_coeff0_2_value * test2_grad2);
      output[7][0] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1 + grad_coeff1_2_value * test2_grad2);
      output[8][0] += q_weight[q] * det * (grad_coeff2_0_value * test2_grad0 + grad_coeff2_1_value * test2_grad1 + grad_coeff2_2_value * test2_grad2);
      output[9][0] += q_weight[q] * det * (grad_coeff0_0_value * test3_grad0 + grad_coeff0_1_value * test3_grad1 + grad_coeff0_2_value * test3_grad2);
      output[10][0] += q_weight[q] * det * (grad_coeff1_0_value * test3_grad0 + grad_coeff1_1_value * test3_grad1 + grad_coeff1_2_value * test3_grad2);
      output[11][0] += q_weight[q] * det * (grad_coeff2_0_value * test3_grad0 + grad_coeff2_1_value * test3_grad1 + grad_coeff2_2_value * test3_grad2);
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_tet4_hessian_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t current[3 * NS][VS],
    const s_t previous[3 * NS][VS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t newmark_velocity_alpha,
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
    const s_t element_matrix_tmp0 = u0_grad_0*u1_grad_1;
    const s_t element_matrix_tmp1 = u0_grad_1*u1_grad_2;
    const s_t element_matrix_tmp2 = u0_grad_2*u2_grad_1;
    const s_t element_matrix_tmp3 = u1_grad_2*u2_grad_1;
    const s_t element_matrix_tmp4 = u0_grad_1*u1_grad_0;
    const s_t element_matrix_tmp5 = u0_grad_2*u2_grad_0;
    const s_t element_matrix_tmp6 = u2_grad_2 + s_t(1);
    const s_t element_matrix_tmp7 = -element_matrix_tmp3 + element_matrix_tmp6 + u1_grad_1*u2_grad_2 + u1_grad_1;
    const s_t element_matrix_tmp8 = -element_matrix_tmp5 + u0_grad_0*u2_grad_2 + u0_grad_0;
    const s_t element_matrix_tmp9 = element_matrix_tmp0 - element_matrix_tmp4;
    const s_t element_matrix_tmp10 = element_matrix_tmp0*u2_grad_2 + element_matrix_tmp1*u2_grad_0 + element_matrix_tmp2*u1_grad_0 - element_matrix_tmp3*u0_grad_0 - element_matrix_tmp4*u2_grad_2 - element_matrix_tmp5*u1_grad_1 + element_matrix_tmp7 + element_matrix_tmp8 + element_matrix_tmp9;
    const s_t element_matrix_tmp11 = pow_m1(element_matrix_tmp10);
    const s_t element_matrix_tmp12 = -element_matrix_tmp1 + u0_grad_2*u1_grad_1 + u0_grad_2;
    const s_t element_matrix_tmp13 = element_matrix_tmp12*newmark_velocity_alpha;
    const s_t element_matrix_tmp14 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
    const s_t element_matrix_tmp15 = element_matrix_tmp14*u1_grad_2;
    const s_t element_matrix_tmp16 = u1_grad_1 + s_t(1);
    const s_t element_matrix_tmp17 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
    const s_t element_matrix_tmp18 = element_matrix_tmp16*element_matrix_tmp17;
    const s_t element_matrix_tmp19 = element_matrix_tmp15 - element_matrix_tmp18;
    const s_t element_matrix_tmp20 = element_matrix_tmp13 + element_matrix_tmp19;
    const s_t element_matrix_tmp21 = -element_matrix_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
    const s_t element_matrix_tmp22 = element_matrix_tmp21*newmark_velocity_alpha;
    const s_t element_matrix_tmp23 = element_matrix_tmp17*u2_grad_1;
    const s_t element_matrix_tmp24 = element_matrix_tmp14*element_matrix_tmp6;
    const s_t element_matrix_tmp25 = element_matrix_tmp23 - element_matrix_tmp24;
    const s_t element_matrix_tmp26 = element_matrix_tmp22 + element_matrix_tmp25;
    const s_t element_matrix_tmp27 = element_matrix_tmp7*newmark_velocity_alpha;
    const s_t element_matrix_tmp28 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
    const s_t element_matrix_tmp29 = element_matrix_tmp16*element_matrix_tmp28;
    const s_t element_matrix_tmp30 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
    const s_t element_matrix_tmp31 = element_matrix_tmp30*u1_grad_2;
    const s_t element_matrix_tmp32 = element_matrix_tmp27 + element_matrix_tmp29 - element_matrix_tmp31;
    const s_t element_matrix_tmp33 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
    const s_t element_matrix_tmp34 = element_matrix_tmp33*element_matrix_tmp6;
    const s_t element_matrix_tmp35 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
    const s_t element_matrix_tmp36 = element_matrix_tmp35*u2_grad_1;
    const s_t element_matrix_tmp37 = element_matrix_tmp34 - element_matrix_tmp36;
    const s_t element_matrix_tmp38 = s_t(3)*eta_b;
    const s_t element_matrix_tmp39 = element_matrix_tmp38*(-element_matrix_tmp32 - element_matrix_tmp37);
    const s_t element_matrix_tmp40 = -element_matrix_tmp34 + element_matrix_tmp36;
    const s_t element_matrix_tmp41 = -element_matrix_tmp29 + element_matrix_tmp31;
    const s_t element_matrix_tmp42 = s_t(2)*eta_s;
    const s_t element_matrix_tmp43 = element_matrix_tmp39 + element_matrix_tmp42*(-s_t(2)*element_matrix_tmp27 - element_matrix_tmp40 - element_matrix_tmp41);
    const s_t element_matrix_tmp44 = ((s_t(1) / s_t(3)))*element_matrix_tmp7;
    const s_t element_matrix_tmp45 = -element_matrix_tmp7;
    const s_t element_matrix_tmp46 = pow_m2(element_matrix_tmp10);
    const s_t element_matrix_tmp47 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
    const s_t element_matrix_tmp48 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
    const s_t element_matrix_tmp49 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
    const s_t element_matrix_tmp50 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
    const s_t element_matrix_tmp51 = element_matrix_tmp16 + element_matrix_tmp9 + u0_grad_0;
    const s_t element_matrix_tmp52 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
    const s_t element_matrix_tmp53 = element_matrix_tmp12*element_matrix_tmp47 + element_matrix_tmp14*element_matrix_tmp48 - element_matrix_tmp17*element_matrix_tmp51 + element_matrix_tmp28*element_matrix_tmp50 + element_matrix_tmp30*element_matrix_tmp49 - element_matrix_tmp52*element_matrix_tmp7;
    const s_t element_matrix_tmp54 = u0_grad_0*u2_grad_1 - u0_grad_1*u2_grad_0 + u2_grad_1;
    const s_t element_matrix_tmp55 = element_matrix_tmp6 + element_matrix_tmp8;
    const s_t element_matrix_tmp56 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
    const s_t element_matrix_tmp57 = -element_matrix_tmp14*element_matrix_tmp55 + element_matrix_tmp17*element_matrix_tmp54 + element_matrix_tmp21*element_matrix_tmp47 + element_matrix_tmp33*element_matrix_tmp49 + element_matrix_tmp35*element_matrix_tmp50 - element_matrix_tmp56*element_matrix_tmp7;
    const s_t element_matrix_tmp58 = element_matrix_tmp21*element_matrix_tmp56;
    const s_t element_matrix_tmp59 = element_matrix_tmp35*element_matrix_tmp54;
    const s_t element_matrix_tmp60 = element_matrix_tmp12*element_matrix_tmp52;
    const s_t element_matrix_tmp61 = element_matrix_tmp30*element_matrix_tmp48;
    const s_t element_matrix_tmp62 = element_matrix_tmp33*element_matrix_tmp55;
    const s_t element_matrix_tmp63 = -element_matrix_tmp62;
    const s_t element_matrix_tmp64 = element_matrix_tmp28*element_matrix_tmp51;
    const s_t element_matrix_tmp65 = -element_matrix_tmp64;
    const s_t element_matrix_tmp66 = element_matrix_tmp58 + element_matrix_tmp59 + element_matrix_tmp60 + element_matrix_tmp61 + element_matrix_tmp63 + element_matrix_tmp65;
    const s_t element_matrix_tmp67 = element_matrix_tmp47*element_matrix_tmp7;
    const s_t element_matrix_tmp68 = element_matrix_tmp14*element_matrix_tmp49 + element_matrix_tmp17*element_matrix_tmp50 - element_matrix_tmp67;
    const s_t element_matrix_tmp69 = element_matrix_tmp38*(element_matrix_tmp66 + element_matrix_tmp68);
    const s_t element_matrix_tmp70 = element_matrix_tmp42*(s_t(2)*element_matrix_tmp14*element_matrix_tmp49 + s_t(2)*element_matrix_tmp17*element_matrix_tmp50 - element_matrix_tmp66 - s_t(2)*element_matrix_tmp67) + element_matrix_tmp69;
    const s_t element_matrix_tmp71 = -element_matrix_tmp70;
    const s_t element_matrix_tmp72 = element_matrix_tmp46*(element_matrix_tmp44*element_matrix_tmp71 + eta_s*(element_matrix_tmp12*element_matrix_tmp53 + element_matrix_tmp21*element_matrix_tmp57));
    const s_t element_matrix_tmp73 = element_matrix_tmp11*(-element_matrix_tmp43*element_matrix_tmp44 + eta_s*(element_matrix_tmp12*element_matrix_tmp20 + element_matrix_tmp21*element_matrix_tmp26)) + element_matrix_tmp45*element_matrix_tmp72;
    const s_t element_matrix_tmp74 = element_matrix_tmp49*newmark_velocity_alpha;
    const s_t element_matrix_tmp75 = element_matrix_tmp28*u1_grad_0;
    const s_t element_matrix_tmp76 = element_matrix_tmp52*u1_grad_2;
    const s_t element_matrix_tmp77 = element_matrix_tmp74 + element_matrix_tmp75 - element_matrix_tmp76;
    const s_t element_matrix_tmp78 = element_matrix_tmp56*element_matrix_tmp6;
    const s_t element_matrix_tmp79 = element_matrix_tmp35*u2_grad_0;
    const s_t element_matrix_tmp80 = element_matrix_tmp78 - element_matrix_tmp79;
    const s_t element_matrix_tmp81 = element_matrix_tmp38*(element_matrix_tmp77 + element_matrix_tmp80);
    const s_t element_matrix_tmp82 = -element_matrix_tmp78 + element_matrix_tmp79;
    const s_t element_matrix_tmp83 = -element_matrix_tmp75 + element_matrix_tmp76;
    const s_t element_matrix_tmp84 = element_matrix_tmp42*(s_t(2)*element_matrix_tmp74 + element_matrix_tmp82 + element_matrix_tmp83) + element_matrix_tmp81;
    const s_t element_matrix_tmp85 = element_matrix_tmp48*newmark_velocity_alpha;
    const s_t element_matrix_tmp86 = element_matrix_tmp17*u1_grad_0;
    const s_t element_matrix_tmp87 = element_matrix_tmp47*u1_grad_2;
    const s_t element_matrix_tmp88 = element_matrix_tmp86 - element_matrix_tmp87;
    const s_t element_matrix_tmp89 = element_matrix_tmp85 + element_matrix_tmp88;
    const s_t element_matrix_tmp90 = element_matrix_tmp55*newmark_velocity_alpha;
    const s_t element_matrix_tmp91 = element_matrix_tmp17*u2_grad_0;
    const s_t element_matrix_tmp92 = element_matrix_tmp47*element_matrix_tmp6;
    const s_t element_matrix_tmp93 = element_matrix_tmp91 - element_matrix_tmp92;
    const s_t element_matrix_tmp94 = -element_matrix_tmp90 - element_matrix_tmp93;
    const s_t element_matrix_tmp95 = element_matrix_tmp53*u1_grad_2;
    const s_t element_matrix_tmp96 = -element_matrix_tmp95;
    const s_t element_matrix_tmp97 = element_matrix_tmp57*element_matrix_tmp6;
    const s_t element_matrix_tmp98 = element_matrix_tmp11*(-element_matrix_tmp44*element_matrix_tmp84 + eta_s*(element_matrix_tmp12*element_matrix_tmp89 + element_matrix_tmp21*element_matrix_tmp94 + element_matrix_tmp96 + element_matrix_tmp97)) + element_matrix_tmp49*element_matrix_tmp72;
    const s_t element_matrix_tmp99 = element_matrix_tmp50*newmark_velocity_alpha;
    const s_t element_matrix_tmp100 = element_matrix_tmp16*element_matrix_tmp52;
    const s_t element_matrix_tmp101 = element_matrix_tmp30*u1_grad_0;
    const s_t element_matrix_tmp102 = element_matrix_tmp100 - element_matrix_tmp101 + element_matrix_tmp99;
    const s_t element_matrix_tmp103 = element_matrix_tmp33*u2_grad_0;
    const s_t element_matrix_tmp104 = element_matrix_tmp56*u2_grad_1;
    const s_t element_matrix_tmp105 = element_matrix_tmp103 - element_matrix_tmp104;
    const s_t element_matrix_tmp106 = element_matrix_tmp38*(element_matrix_tmp102 + element_matrix_tmp105);
    const s_t element_matrix_tmp107 = -element_matrix_tmp103 + element_matrix_tmp104;
    const s_t element_matrix_tmp108 = -element_matrix_tmp100 + element_matrix_tmp101;
    const s_t element_matrix_tmp109 = element_matrix_tmp106 + element_matrix_tmp42*(element_matrix_tmp107 + element_matrix_tmp108 + s_t(2)*element_matrix_tmp99);
    const s_t element_matrix_tmp110 = element_matrix_tmp54*newmark_velocity_alpha;
    const s_t element_matrix_tmp111 = element_matrix_tmp14*u2_grad_0;
    const s_t element_matrix_tmp112 = element_matrix_tmp47*u2_grad_1;
    const s_t element_matrix_tmp113 = element_matrix_tmp111 - element_matrix_tmp112;
    const s_t element_matrix_tmp114 = element_matrix_tmp110 + element_matrix_tmp113;
    const s_t element_matrix_tmp115 = element_matrix_tmp51*newmark_velocity_alpha;
    const s_t element_matrix_tmp116 = element_matrix_tmp14*u1_grad_0;
    const s_t element_matrix_tmp117 = element_matrix_tmp16*element_matrix_tmp47;
    const s_t element_matrix_tmp118 = element_matrix_tmp116 - element_matrix_tmp117;
    const s_t element_matrix_tmp119 = -element_matrix_tmp115 - element_matrix_tmp118;
    const s_t element_matrix_tmp120 = element_matrix_tmp57*u2_grad_1;
    const s_t element_matrix_tmp121 = -element_matrix_tmp120;
    const s_t element_matrix_tmp122 = element_matrix_tmp16*element_matrix_tmp53;
    const s_t element_matrix_tmp123 = element_matrix_tmp11*(-element_matrix_tmp109*element_matrix_tmp44 + eta_s*(element_matrix_tmp114*element_matrix_tmp21 + element_matrix_tmp119*element_matrix_tmp12 + element_matrix_tmp121 + element_matrix_tmp122)) + element_matrix_tmp50*element_matrix_tmp72;
    const s_t element_matrix_tmp124 = basis0_grad0*element_matrix_tmp73 + basis0_grad1*element_matrix_tmp98 + basis0_grad2*element_matrix_tmp123;
    const s_t element_matrix_tmp125 = element_matrix_tmp46*(((s_t(1) / s_t(3)))*element_matrix_tmp49*element_matrix_tmp70 - eta_s*(-element_matrix_tmp48*element_matrix_tmp53 + element_matrix_tmp55*element_matrix_tmp57));
    const s_t element_matrix_tmp126 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp49*element_matrix_tmp84 - eta_s*(-element_matrix_tmp48*element_matrix_tmp89 + element_matrix_tmp55*element_matrix_tmp94)) + element_matrix_tmp125*element_matrix_tmp49;
    const s_t element_matrix_tmp127 = element_matrix_tmp53*u1_grad_0;
    const s_t element_matrix_tmp128 = element_matrix_tmp57*u2_grad_0;
    const s_t element_matrix_tmp129 = -element_matrix_tmp128;
    const s_t element_matrix_tmp130 = element_matrix_tmp127 + element_matrix_tmp129;
    const s_t element_matrix_tmp131 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp109*element_matrix_tmp49 - eta_s*(element_matrix_tmp114*element_matrix_tmp55 - element_matrix_tmp119*element_matrix_tmp48 + element_matrix_tmp130)) + element_matrix_tmp125*element_matrix_tmp50;
    const s_t element_matrix_tmp132 = -element_matrix_tmp57*element_matrix_tmp6;
    const s_t element_matrix_tmp133 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp43*element_matrix_tmp49 - eta_s*(-element_matrix_tmp132 - element_matrix_tmp20*element_matrix_tmp48 + element_matrix_tmp26*element_matrix_tmp55 - element_matrix_tmp95)) + element_matrix_tmp125*element_matrix_tmp45;
    const s_t element_matrix_tmp134 = basis0_grad0*element_matrix_tmp133 + basis0_grad1*element_matrix_tmp126 + basis0_grad2*element_matrix_tmp131;
    const s_t element_matrix_tmp135 = element_matrix_tmp46*(((s_t(1) / s_t(3)))*element_matrix_tmp50*element_matrix_tmp70 - eta_s*(element_matrix_tmp51*element_matrix_tmp53 - element_matrix_tmp54*element_matrix_tmp57));
    const s_t element_matrix_tmp136 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp109*element_matrix_tmp50 - eta_s*(-element_matrix_tmp114*element_matrix_tmp54 + element_matrix_tmp119*element_matrix_tmp51)) + element_matrix_tmp135*element_matrix_tmp50;
    const s_t element_matrix_tmp137 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp50*element_matrix_tmp84 - eta_s*(-element_matrix_tmp130 + element_matrix_tmp51*element_matrix_tmp89 - element_matrix_tmp54*element_matrix_tmp94)) + element_matrix_tmp135*element_matrix_tmp49;
    const s_t element_matrix_tmp138 = -element_matrix_tmp16*element_matrix_tmp53;
    const s_t element_matrix_tmp139 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp43*element_matrix_tmp50 - eta_s*(-element_matrix_tmp120 - element_matrix_tmp138 + element_matrix_tmp20*element_matrix_tmp51 - element_matrix_tmp26*element_matrix_tmp54)) + element_matrix_tmp135*element_matrix_tmp45;
    const s_t element_matrix_tmp140 = basis0_grad0*element_matrix_tmp139 + basis0_grad1*element_matrix_tmp137 + basis0_grad2*element_matrix_tmp136;
    const s_t element_matrix_tmp141 = ((s_t(1) / s_t(6)))*det;
    const s_t element_matrix_tmp142 = element_matrix_tmp52*element_matrix_tmp6;
    const s_t element_matrix_tmp143 = element_matrix_tmp28*u2_grad_0;
    const s_t element_matrix_tmp144 = element_matrix_tmp35*u1_grad_0 - element_matrix_tmp56*u1_grad_2;
    const s_t element_matrix_tmp145 = element_matrix_tmp142 - element_matrix_tmp143 + element_matrix_tmp144;
    const s_t element_matrix_tmp146 = element_matrix_tmp42*(s_t(2)*element_matrix_tmp56*element_matrix_tmp6 - element_matrix_tmp77 - s_t(2)*element_matrix_tmp79) + element_matrix_tmp81;
    const s_t element_matrix_tmp147 = ((s_t(1) / s_t(3)))*element_matrix_tmp55;
    const s_t element_matrix_tmp148 = element_matrix_tmp12*element_matrix_tmp56 + element_matrix_tmp21*element_matrix_tmp52 + element_matrix_tmp28*element_matrix_tmp54 - element_matrix_tmp30*element_matrix_tmp55 + element_matrix_tmp33*element_matrix_tmp48 - element_matrix_tmp35*element_matrix_tmp51;
    const s_t element_matrix_tmp149 = element_matrix_tmp42*(s_t(2)*element_matrix_tmp21*element_matrix_tmp56 + s_t(2)*element_matrix_tmp35*element_matrix_tmp54 - element_matrix_tmp60 - element_matrix_tmp61 - s_t(2)*element_matrix_tmp62 - element_matrix_tmp65 - element_matrix_tmp68) + element_matrix_tmp69;
    const s_t element_matrix_tmp150 = -(s_t(1) / s_t(3))*element_matrix_tmp149;
    const s_t element_matrix_tmp151 = element_matrix_tmp46*(element_matrix_tmp150*element_matrix_tmp55 + eta_s*(element_matrix_tmp148*element_matrix_tmp48 + element_matrix_tmp49*element_matrix_tmp57));
    const s_t element_matrix_tmp152 = element_matrix_tmp11*(-element_matrix_tmp146*element_matrix_tmp147 + eta_s*(element_matrix_tmp145*element_matrix_tmp48 + element_matrix_tmp49*element_matrix_tmp94)) + element_matrix_tmp151*element_matrix_tmp49;
    const s_t element_matrix_tmp153 = element_matrix_tmp106 + element_matrix_tmp42*(-element_matrix_tmp102 - s_t(2)*element_matrix_tmp104 + s_t(2)*element_matrix_tmp33*u2_grad_0);
    const s_t element_matrix_tmp154 = element_matrix_tmp52*u2_grad_1;
    const s_t element_matrix_tmp155 = element_matrix_tmp30*u2_grad_0;
    const s_t element_matrix_tmp156 = -element_matrix_tmp16*element_matrix_tmp56 + element_matrix_tmp33*u1_grad_0;
    const s_t element_matrix_tmp157 = -element_matrix_tmp154 + element_matrix_tmp155 - element_matrix_tmp156;
    const s_t element_matrix_tmp158 = element_matrix_tmp148*u1_grad_0;
    const s_t element_matrix_tmp159 = element_matrix_tmp11*(-element_matrix_tmp147*element_matrix_tmp153 - element_matrix_tmp150*u2_grad_0 + eta_s*(element_matrix_tmp114*element_matrix_tmp49 + element_matrix_tmp157*element_matrix_tmp48 - element_matrix_tmp158)) + element_matrix_tmp151*element_matrix_tmp50;
    const s_t element_matrix_tmp160 = element_matrix_tmp39 + element_matrix_tmp42*(element_matrix_tmp32 - s_t(2)*element_matrix_tmp34 + s_t(2)*element_matrix_tmp36);
    const s_t element_matrix_tmp161 = element_matrix_tmp28*u2_grad_1;
    const s_t element_matrix_tmp162 = element_matrix_tmp30*element_matrix_tmp6;
    const s_t element_matrix_tmp163 = -element_matrix_tmp16*element_matrix_tmp35 + element_matrix_tmp33*u1_grad_2;
    const s_t element_matrix_tmp164 = element_matrix_tmp161 - element_matrix_tmp162 + element_matrix_tmp163;
    const s_t element_matrix_tmp165 = element_matrix_tmp148*u1_grad_2;
    const s_t element_matrix_tmp166 = element_matrix_tmp11*(-element_matrix_tmp147*element_matrix_tmp160 + element_matrix_tmp150*element_matrix_tmp6 + eta_s*(element_matrix_tmp164*element_matrix_tmp48 + element_matrix_tmp165 + element_matrix_tmp26*element_matrix_tmp49)) + element_matrix_tmp151*element_matrix_tmp45;
    const s_t element_matrix_tmp167 = basis0_grad0*element_matrix_tmp166 + basis0_grad1*element_matrix_tmp152 + basis0_grad2*element_matrix_tmp159;
    const s_t element_matrix_tmp168 = element_matrix_tmp46*(((s_t(1) / s_t(3)))*element_matrix_tmp149*element_matrix_tmp54 - eta_s*(element_matrix_tmp148*element_matrix_tmp51 - element_matrix_tmp50*element_matrix_tmp57));
    const s_t element_matrix_tmp169 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp153*element_matrix_tmp54 - eta_s*(-element_matrix_tmp114*element_matrix_tmp50 + element_matrix_tmp157*element_matrix_tmp51)) + element_matrix_tmp168*element_matrix_tmp50;
    const s_t element_matrix_tmp170 = ((s_t(1) / s_t(3)))*element_matrix_tmp149;
    const s_t element_matrix_tmp171 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp146*element_matrix_tmp54 - element_matrix_tmp170*u2_grad_0 - eta_s*(element_matrix_tmp145*element_matrix_tmp51 - element_matrix_tmp158 - element_matrix_tmp50*element_matrix_tmp94)) + element_matrix_tmp168*element_matrix_tmp49;
    const s_t element_matrix_tmp172 = ((s_t(1) / s_t(3)))*element_matrix_tmp54;
    const s_t element_matrix_tmp173 = element_matrix_tmp148*element_matrix_tmp16;
    const s_t element_matrix_tmp174 = element_matrix_tmp170*u2_grad_1;
    const s_t element_matrix_tmp175 = element_matrix_tmp11*(element_matrix_tmp160*element_matrix_tmp172 + element_matrix_tmp174 - eta_s*(element_matrix_tmp164*element_matrix_tmp51 + element_matrix_tmp173 - element_matrix_tmp26*element_matrix_tmp50)) + element_matrix_tmp168*element_matrix_tmp45;
    const s_t element_matrix_tmp176 = basis0_grad0*element_matrix_tmp175 + basis0_grad1*element_matrix_tmp171 + basis0_grad2*element_matrix_tmp169;
    const s_t element_matrix_tmp177 = element_matrix_tmp46*(((s_t(1) / s_t(3)))*element_matrix_tmp149*element_matrix_tmp21 - eta_s*(-element_matrix_tmp12*element_matrix_tmp148 + element_matrix_tmp57*element_matrix_tmp7));
    const s_t element_matrix_tmp178 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp160*element_matrix_tmp21 - eta_s*(-element_matrix_tmp12*element_matrix_tmp164 + element_matrix_tmp26*element_matrix_tmp7)) + element_matrix_tmp177*element_matrix_tmp45;
    const s_t element_matrix_tmp179 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp153*element_matrix_tmp21 - element_matrix_tmp174 - eta_s*(element_matrix_tmp114*element_matrix_tmp7 - element_matrix_tmp12*element_matrix_tmp157 - element_matrix_tmp173)) + element_matrix_tmp177*element_matrix_tmp50;
    const s_t element_matrix_tmp180 = ((s_t(1) / s_t(3)))*element_matrix_tmp21;
    const s_t element_matrix_tmp181 = element_matrix_tmp11*(element_matrix_tmp146*element_matrix_tmp180 + element_matrix_tmp170*element_matrix_tmp6 - eta_s*(-element_matrix_tmp12*element_matrix_tmp145 + element_matrix_tmp165 + element_matrix_tmp7*element_matrix_tmp94)) + element_matrix_tmp177*element_matrix_tmp49;
    const s_t element_matrix_tmp182 = basis0_grad0*element_matrix_tmp178 + basis0_grad1*element_matrix_tmp181 + basis0_grad2*element_matrix_tmp179;
    const s_t element_matrix_tmp183 = element_matrix_tmp106 + element_matrix_tmp42*(-s_t(2)*element_matrix_tmp101 - element_matrix_tmp105 + s_t(2)*element_matrix_tmp16*element_matrix_tmp52 - element_matrix_tmp99);
    const s_t element_matrix_tmp184 = ((s_t(1) / s_t(3)))*element_matrix_tmp51;
    const s_t element_matrix_tmp185 = element_matrix_tmp42*(s_t(2)*element_matrix_tmp12*element_matrix_tmp52 + s_t(2)*element_matrix_tmp30*element_matrix_tmp48 - element_matrix_tmp58 - element_matrix_tmp59 - element_matrix_tmp63 - s_t(2)*element_matrix_tmp64 - element_matrix_tmp68) + element_matrix_tmp69;
    const s_t element_matrix_tmp186 = -(s_t(1) / s_t(3))*element_matrix_tmp185;
    const s_t element_matrix_tmp187 = element_matrix_tmp46*(element_matrix_tmp186*element_matrix_tmp51 + eta_s*(element_matrix_tmp148*element_matrix_tmp54 + element_matrix_tmp50*element_matrix_tmp53));
    const s_t element_matrix_tmp188 = element_matrix_tmp11*(-element_matrix_tmp183*element_matrix_tmp184 + eta_s*(element_matrix_tmp119*element_matrix_tmp50 + element_matrix_tmp157*element_matrix_tmp54)) + element_matrix_tmp187*element_matrix_tmp50;
    const s_t element_matrix_tmp189 = element_matrix_tmp42*(s_t(2)*element_matrix_tmp28*u1_grad_0 - element_matrix_tmp74 - s_t(2)*element_matrix_tmp76 - element_matrix_tmp80) + element_matrix_tmp81;
    const s_t element_matrix_tmp190 = element_matrix_tmp148*u2_grad_0;
    const s_t element_matrix_tmp191 = element_matrix_tmp11*(-element_matrix_tmp184*element_matrix_tmp189 - element_matrix_tmp186*u1_grad_0 + eta_s*(element_matrix_tmp145*element_matrix_tmp54 - element_matrix_tmp190 + element_matrix_tmp50*element_matrix_tmp89)) + element_matrix_tmp187*element_matrix_tmp49;
    const s_t element_matrix_tmp192 = element_matrix_tmp39 + element_matrix_tmp42*(element_matrix_tmp27 - s_t(2)*element_matrix_tmp29 + s_t(2)*element_matrix_tmp31 + element_matrix_tmp37);
    const s_t element_matrix_tmp193 = element_matrix_tmp148*u2_grad_1;
    const s_t element_matrix_tmp194 = element_matrix_tmp11*(element_matrix_tmp16*element_matrix_tmp186 - element_matrix_tmp184*element_matrix_tmp192 + eta_s*(element_matrix_tmp164*element_matrix_tmp54 + element_matrix_tmp193 + element_matrix_tmp20*element_matrix_tmp50)) + element_matrix_tmp187*element_matrix_tmp45;
    const s_t element_matrix_tmp195 = basis0_grad0*element_matrix_tmp194 + basis0_grad1*element_matrix_tmp191 + basis0_grad2*element_matrix_tmp188;
    const s_t element_matrix_tmp196 = element_matrix_tmp46*(((s_t(1) / s_t(3)))*element_matrix_tmp185*element_matrix_tmp48 - eta_s*(element_matrix_tmp148*element_matrix_tmp55 - element_matrix_tmp49*element_matrix_tmp53));
    const s_t element_matrix_tmp197 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp189*element_matrix_tmp48 - eta_s*(element_matrix_tmp145*element_matrix_tmp55 - element_matrix_tmp49*element_matrix_tmp89)) + element_matrix_tmp196*element_matrix_tmp49;
    const s_t element_matrix_tmp198 = ((s_t(1) / s_t(3)))*element_matrix_tmp185;
    const s_t element_matrix_tmp199 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp183*element_matrix_tmp48 - element_matrix_tmp198*u1_grad_0 - eta_s*(-element_matrix_tmp119*element_matrix_tmp49 + element_matrix_tmp157*element_matrix_tmp55 - element_matrix_tmp190)) + element_matrix_tmp196*element_matrix_tmp50;
    const s_t element_matrix_tmp200 = ((s_t(1) / s_t(3)))*element_matrix_tmp48;
    const s_t element_matrix_tmp201 = element_matrix_tmp148*element_matrix_tmp6;
    const s_t element_matrix_tmp202 = element_matrix_tmp198*u1_grad_2;
    const s_t element_matrix_tmp203 = element_matrix_tmp11*(element_matrix_tmp192*element_matrix_tmp200 + element_matrix_tmp202 - eta_s*(element_matrix_tmp164*element_matrix_tmp55 - element_matrix_tmp20*element_matrix_tmp49 + element_matrix_tmp201)) + element_matrix_tmp196*element_matrix_tmp45;
    const s_t element_matrix_tmp204 = basis0_grad0*element_matrix_tmp203 + basis0_grad1*element_matrix_tmp197 + basis0_grad2*element_matrix_tmp199;
    const s_t element_matrix_tmp205 = element_matrix_tmp46*(((s_t(1) / s_t(3)))*element_matrix_tmp12*element_matrix_tmp185 - eta_s*(-element_matrix_tmp148*element_matrix_tmp21 + element_matrix_tmp53*element_matrix_tmp7));
    const s_t element_matrix_tmp206 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp12*element_matrix_tmp192 - eta_s*(-element_matrix_tmp164*element_matrix_tmp21 + element_matrix_tmp20*element_matrix_tmp7)) + element_matrix_tmp205*element_matrix_tmp45;
    const s_t element_matrix_tmp207 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp12*element_matrix_tmp189 - element_matrix_tmp202 - eta_s*(-element_matrix_tmp145*element_matrix_tmp21 - element_matrix_tmp201 + element_matrix_tmp7*element_matrix_tmp89)) + element_matrix_tmp205*element_matrix_tmp49;
    const s_t element_matrix_tmp208 = ((s_t(1) / s_t(3)))*element_matrix_tmp12;
    const s_t element_matrix_tmp209 = element_matrix_tmp11*(element_matrix_tmp16*element_matrix_tmp198 + element_matrix_tmp183*element_matrix_tmp208 - eta_s*(element_matrix_tmp119*element_matrix_tmp7 - element_matrix_tmp157*element_matrix_tmp21 + element_matrix_tmp193)) + element_matrix_tmp205*element_matrix_tmp50;
    const s_t element_matrix_tmp210 = basis0_grad0*element_matrix_tmp206 + basis0_grad1*element_matrix_tmp207 + basis0_grad2*element_matrix_tmp209;
    const s_t element_matrix_tmp211 = basis1_grad0*element_matrix_tmp73 + basis1_grad1*element_matrix_tmp98 + basis1_grad2*element_matrix_tmp123;
    const s_t element_matrix_tmp212 = basis1_grad0*element_matrix_tmp133 + basis1_grad1*element_matrix_tmp126 + basis1_grad2*element_matrix_tmp131;
    const s_t element_matrix_tmp213 = basis1_grad0*element_matrix_tmp139 + basis1_grad1*element_matrix_tmp137 + basis1_grad2*element_matrix_tmp136;
    const s_t element_matrix_tmp214 = basis1_grad0*element_matrix_tmp166 + basis1_grad1*element_matrix_tmp152 + basis1_grad2*element_matrix_tmp159;
    const s_t element_matrix_tmp215 = basis1_grad0*element_matrix_tmp175 + basis1_grad1*element_matrix_tmp171 + basis1_grad2*element_matrix_tmp169;
    const s_t element_matrix_tmp216 = basis1_grad0*element_matrix_tmp178 + basis1_grad1*element_matrix_tmp181 + basis1_grad2*element_matrix_tmp179;
    const s_t element_matrix_tmp217 = basis1_grad0*element_matrix_tmp194 + basis1_grad1*element_matrix_tmp191 + basis1_grad2*element_matrix_tmp188;
    const s_t element_matrix_tmp218 = basis1_grad0*element_matrix_tmp203 + basis1_grad1*element_matrix_tmp197 + basis1_grad2*element_matrix_tmp199;
    const s_t element_matrix_tmp219 = basis1_grad0*element_matrix_tmp206 + basis1_grad1*element_matrix_tmp207 + basis1_grad2*element_matrix_tmp209;
    const s_t element_matrix_tmp220 = basis2_grad0*element_matrix_tmp73 + basis2_grad1*element_matrix_tmp98 + basis2_grad2*element_matrix_tmp123;
    const s_t element_matrix_tmp221 = basis2_grad0*element_matrix_tmp133 + basis2_grad1*element_matrix_tmp126 + basis2_grad2*element_matrix_tmp131;
    const s_t element_matrix_tmp222 = basis2_grad0*element_matrix_tmp139 + basis2_grad1*element_matrix_tmp137 + basis2_grad2*element_matrix_tmp136;
    const s_t element_matrix_tmp223 = basis2_grad0*element_matrix_tmp166 + basis2_grad1*element_matrix_tmp152 + basis2_grad2*element_matrix_tmp159;
    const s_t element_matrix_tmp224 = basis2_grad0*element_matrix_tmp175 + basis2_grad1*element_matrix_tmp171 + basis2_grad2*element_matrix_tmp169;
    const s_t element_matrix_tmp225 = basis2_grad0*element_matrix_tmp178 + basis2_grad1*element_matrix_tmp181 + basis2_grad2*element_matrix_tmp179;
    const s_t element_matrix_tmp226 = basis2_grad0*element_matrix_tmp194 + basis2_grad1*element_matrix_tmp191 + basis2_grad2*element_matrix_tmp188;
    const s_t element_matrix_tmp227 = basis2_grad0*element_matrix_tmp203 + basis2_grad1*element_matrix_tmp197 + basis2_grad2*element_matrix_tmp199;
    const s_t element_matrix_tmp228 = basis2_grad0*element_matrix_tmp206 + basis2_grad1*element_matrix_tmp207 + basis2_grad2*element_matrix_tmp209;
    const s_t element_matrix_tmp229 = basis3_grad0*element_matrix_tmp73 + basis3_grad1*element_matrix_tmp98 + basis3_grad2*element_matrix_tmp123;
    const s_t element_matrix_tmp230 = basis3_grad0*element_matrix_tmp133 + basis3_grad1*element_matrix_tmp126 + basis3_grad2*element_matrix_tmp131;
    const s_t element_matrix_tmp231 = basis3_grad0*element_matrix_tmp139 + basis3_grad1*element_matrix_tmp137 + basis3_grad2*element_matrix_tmp136;
    const s_t element_matrix_tmp232 = basis3_grad0*element_matrix_tmp166 + basis3_grad1*element_matrix_tmp152 + basis3_grad2*element_matrix_tmp159;
    const s_t element_matrix_tmp233 = basis3_grad0*element_matrix_tmp175 + basis3_grad1*element_matrix_tmp171 + basis3_grad2*element_matrix_tmp169;
    const s_t element_matrix_tmp234 = basis3_grad0*element_matrix_tmp178 + basis3_grad1*element_matrix_tmp181 + basis3_grad2*element_matrix_tmp179;
    const s_t element_matrix_tmp235 = basis3_grad0*element_matrix_tmp194 + basis3_grad1*element_matrix_tmp191 + basis3_grad2*element_matrix_tmp188;
    const s_t element_matrix_tmp236 = basis3_grad0*element_matrix_tmp203 + basis3_grad1*element_matrix_tmp197 + basis3_grad2*element_matrix_tmp199;
    const s_t element_matrix_tmp237 = basis3_grad0*element_matrix_tmp206 + basis3_grad1*element_matrix_tmp207 + basis3_grad2*element_matrix_tmp209;
    const s_t element_matrix_tmp238 = -element_matrix_tmp27 - element_matrix_tmp40;
    const s_t element_matrix_tmp239 = -element_matrix_tmp14*u0_grad_2 + element_matrix_tmp17*u0_grad_1;
    const s_t element_matrix_tmp240 = -element_matrix_tmp161 + element_matrix_tmp162 + element_matrix_tmp239;
    const s_t element_matrix_tmp241 = element_matrix_tmp28*u0_grad_1;
    const s_t element_matrix_tmp242 = element_matrix_tmp30*u0_grad_2;
    const s_t element_matrix_tmp243 = element_matrix_tmp22 + element_matrix_tmp241 - element_matrix_tmp242;
    const s_t element_matrix_tmp244 = -element_matrix_tmp23 + element_matrix_tmp24;
    const s_t element_matrix_tmp245 = element_matrix_tmp38*(element_matrix_tmp243 + element_matrix_tmp244);
    const s_t element_matrix_tmp246 = element_matrix_tmp245 + element_matrix_tmp42*(s_t(2)*element_matrix_tmp14*element_matrix_tmp6 - s_t(2)*element_matrix_tmp23 - element_matrix_tmp243);
    const s_t element_matrix_tmp247 = element_matrix_tmp11*(-element_matrix_tmp246*element_matrix_tmp44 + eta_s*(element_matrix_tmp12*element_matrix_tmp240 + element_matrix_tmp21*element_matrix_tmp238)) + element_matrix_tmp21*element_matrix_tmp72;
    const s_t element_matrix_tmp248 = u0_grad_0 + s_t(1);
    const s_t element_matrix_tmp249 = element_matrix_tmp248*element_matrix_tmp30;
    const s_t element_matrix_tmp250 = element_matrix_tmp52*u0_grad_1;
    const s_t element_matrix_tmp251 = element_matrix_tmp110 + element_matrix_tmp249 - element_matrix_tmp250;
    const s_t element_matrix_tmp252 = -element_matrix_tmp111 + element_matrix_tmp112;
    const s_t element_matrix_tmp253 = element_matrix_tmp38*(element_matrix_tmp251 + element_matrix_tmp252);
    const s_t element_matrix_tmp254 = element_matrix_tmp253 + element_matrix_tmp42*(-s_t(2)*element_matrix_tmp111 - element_matrix_tmp251 + s_t(2)*element_matrix_tmp47*u2_grad_1);
    const s_t element_matrix_tmp255 = element_matrix_tmp107 + element_matrix_tmp99;
    const s_t element_matrix_tmp256 = -element_matrix_tmp14*element_matrix_tmp248 + element_matrix_tmp47*u0_grad_1;
    const s_t element_matrix_tmp257 = element_matrix_tmp154 - element_matrix_tmp155 - element_matrix_tmp256;
    const s_t element_matrix_tmp258 = element_matrix_tmp53*u0_grad_1;
    const s_t element_matrix_tmp259 = ((s_t(1) / s_t(3)))*element_matrix_tmp71;
    const s_t element_matrix_tmp260 = element_matrix_tmp11*(-element_matrix_tmp254*element_matrix_tmp44 - element_matrix_tmp259*u2_grad_1 + eta_s*(element_matrix_tmp12*element_matrix_tmp257 + element_matrix_tmp21*element_matrix_tmp255 - element_matrix_tmp258)) + element_matrix_tmp54*element_matrix_tmp72;
    const s_t element_matrix_tmp261 = -element_matrix_tmp55;
    const s_t element_matrix_tmp262 = element_matrix_tmp248*element_matrix_tmp28;
    const s_t element_matrix_tmp263 = element_matrix_tmp52*u0_grad_2;
    const s_t element_matrix_tmp264 = element_matrix_tmp262 - element_matrix_tmp263 + element_matrix_tmp90;
    const s_t element_matrix_tmp265 = -element_matrix_tmp91 + element_matrix_tmp92;
    const s_t element_matrix_tmp266 = element_matrix_tmp38*(-element_matrix_tmp264 - element_matrix_tmp265);
    const s_t element_matrix_tmp267 = element_matrix_tmp266 + element_matrix_tmp42*(element_matrix_tmp264 + s_t(2)*element_matrix_tmp91 - s_t(2)*element_matrix_tmp92);
    const s_t element_matrix_tmp268 = element_matrix_tmp74 + element_matrix_tmp82;
    const s_t element_matrix_tmp269 = -element_matrix_tmp17*element_matrix_tmp248 + element_matrix_tmp47*u0_grad_2;
    const s_t element_matrix_tmp270 = -element_matrix_tmp142 + element_matrix_tmp143 + element_matrix_tmp269;
    const s_t element_matrix_tmp271 = element_matrix_tmp53*u0_grad_2;
    const s_t element_matrix_tmp272 = element_matrix_tmp11*(element_matrix_tmp259*element_matrix_tmp6 - element_matrix_tmp267*element_matrix_tmp44 + eta_s*(element_matrix_tmp12*element_matrix_tmp270 + element_matrix_tmp21*element_matrix_tmp268 + element_matrix_tmp271)) + element_matrix_tmp261*element_matrix_tmp72;
    const s_t element_matrix_tmp273 = basis0_grad0*element_matrix_tmp247 + basis0_grad1*element_matrix_tmp272 + basis0_grad2*element_matrix_tmp260;
    const s_t element_matrix_tmp274 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp254*element_matrix_tmp50 - eta_s*(-element_matrix_tmp255*element_matrix_tmp54 + element_matrix_tmp257*element_matrix_tmp51)) + element_matrix_tmp135*element_matrix_tmp54;
    const s_t element_matrix_tmp275 = ((s_t(1) / s_t(3)))*element_matrix_tmp70;
    const s_t element_matrix_tmp276 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp246*element_matrix_tmp50 - element_matrix_tmp275*u2_grad_1 - eta_s*(-element_matrix_tmp238*element_matrix_tmp54 + element_matrix_tmp240*element_matrix_tmp51 - element_matrix_tmp258)) + element_matrix_tmp135*element_matrix_tmp21;
    const s_t element_matrix_tmp277 = ((s_t(1) / s_t(3)))*element_matrix_tmp50;
    const s_t element_matrix_tmp278 = element_matrix_tmp248*element_matrix_tmp53;
    const s_t element_matrix_tmp279 = element_matrix_tmp275*u2_grad_0;
    const s_t element_matrix_tmp280 = element_matrix_tmp11*(element_matrix_tmp267*element_matrix_tmp277 + element_matrix_tmp279 - eta_s*(-element_matrix_tmp268*element_matrix_tmp54 + element_matrix_tmp270*element_matrix_tmp51 + element_matrix_tmp278)) + element_matrix_tmp135*element_matrix_tmp261;
    const s_t element_matrix_tmp281 = basis0_grad0*element_matrix_tmp276 + basis0_grad1*element_matrix_tmp280 + basis0_grad2*element_matrix_tmp274;
    const s_t element_matrix_tmp282 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp267*element_matrix_tmp49 - eta_s*(element_matrix_tmp268*element_matrix_tmp55 - element_matrix_tmp270*element_matrix_tmp48)) + element_matrix_tmp125*element_matrix_tmp261;
    const s_t element_matrix_tmp283 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp254*element_matrix_tmp49 - element_matrix_tmp279 - eta_s*(element_matrix_tmp255*element_matrix_tmp55 - element_matrix_tmp257*element_matrix_tmp48 - element_matrix_tmp278)) + element_matrix_tmp125*element_matrix_tmp54;
    const s_t element_matrix_tmp284 = ((s_t(1) / s_t(3)))*element_matrix_tmp49;
    const s_t element_matrix_tmp285 = element_matrix_tmp11*(element_matrix_tmp246*element_matrix_tmp284 + element_matrix_tmp275*element_matrix_tmp6 - eta_s*(element_matrix_tmp238*element_matrix_tmp55 - element_matrix_tmp240*element_matrix_tmp48 + element_matrix_tmp271)) + element_matrix_tmp125*element_matrix_tmp21;
    const s_t element_matrix_tmp286 = basis0_grad0*element_matrix_tmp285 + basis0_grad1*element_matrix_tmp282 + basis0_grad2*element_matrix_tmp283;
    const s_t element_matrix_tmp287 = element_matrix_tmp56*u0_grad_2;
    const s_t element_matrix_tmp288 = element_matrix_tmp248*element_matrix_tmp35;
    const s_t element_matrix_tmp289 = element_matrix_tmp287 - element_matrix_tmp288;
    const s_t element_matrix_tmp290 = element_matrix_tmp289 + element_matrix_tmp85;
    const s_t element_matrix_tmp291 = -element_matrix_tmp262 + element_matrix_tmp263;
    const s_t element_matrix_tmp292 = element_matrix_tmp266 + element_matrix_tmp42*(-element_matrix_tmp291 - s_t(2)*element_matrix_tmp90 - element_matrix_tmp93);
    const s_t element_matrix_tmp293 = element_matrix_tmp11*(-element_matrix_tmp147*element_matrix_tmp292 + eta_s*(element_matrix_tmp268*element_matrix_tmp49 + element_matrix_tmp290*element_matrix_tmp48)) + element_matrix_tmp151*element_matrix_tmp261;
    const s_t element_matrix_tmp294 = -element_matrix_tmp241 + element_matrix_tmp242;
    const s_t element_matrix_tmp295 = element_matrix_tmp245 + element_matrix_tmp42*(s_t(2)*element_matrix_tmp22 + element_matrix_tmp25 + element_matrix_tmp294);
    const s_t element_matrix_tmp296 = element_matrix_tmp35*u0_grad_1;
    const s_t element_matrix_tmp297 = element_matrix_tmp33*u0_grad_2;
    const s_t element_matrix_tmp298 = element_matrix_tmp296 - element_matrix_tmp297;
    const s_t element_matrix_tmp299 = element_matrix_tmp13 + element_matrix_tmp298;
    const s_t element_matrix_tmp300 = element_matrix_tmp148*u0_grad_2;
    const s_t element_matrix_tmp301 = element_matrix_tmp11*(-element_matrix_tmp147*element_matrix_tmp295 + eta_s*(element_matrix_tmp238*element_matrix_tmp49 + element_matrix_tmp299*element_matrix_tmp48 - element_matrix_tmp300 + element_matrix_tmp97)) + element_matrix_tmp151*element_matrix_tmp21;
    const s_t element_matrix_tmp302 = -element_matrix_tmp249 + element_matrix_tmp250;
    const s_t element_matrix_tmp303 = element_matrix_tmp253 + element_matrix_tmp42*(s_t(2)*element_matrix_tmp110 + element_matrix_tmp113 + element_matrix_tmp302);
    const s_t element_matrix_tmp304 = element_matrix_tmp56*u0_grad_1;
    const s_t element_matrix_tmp305 = element_matrix_tmp248*element_matrix_tmp33;
    const s_t element_matrix_tmp306 = element_matrix_tmp304 - element_matrix_tmp305;
    const s_t element_matrix_tmp307 = -element_matrix_tmp115 - element_matrix_tmp306;
    const s_t element_matrix_tmp308 = element_matrix_tmp148*element_matrix_tmp248;
    const s_t element_matrix_tmp309 = element_matrix_tmp11*(-element_matrix_tmp147*element_matrix_tmp303 + eta_s*(element_matrix_tmp129 + element_matrix_tmp255*element_matrix_tmp49 + element_matrix_tmp307*element_matrix_tmp48 + element_matrix_tmp308)) + element_matrix_tmp151*element_matrix_tmp54;
    const s_t element_matrix_tmp310 = basis0_grad0*element_matrix_tmp301 + basis0_grad1*element_matrix_tmp293 + basis0_grad2*element_matrix_tmp309;
    const s_t element_matrix_tmp311 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp21*element_matrix_tmp295 - eta_s*(-element_matrix_tmp12*element_matrix_tmp299 + element_matrix_tmp238*element_matrix_tmp7)) + element_matrix_tmp177*element_matrix_tmp21;
    const s_t element_matrix_tmp312 = element_matrix_tmp148*u0_grad_1;
    const s_t element_matrix_tmp313 = element_matrix_tmp121 + element_matrix_tmp312;
    const s_t element_matrix_tmp314 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp21*element_matrix_tmp303 - eta_s*(-element_matrix_tmp12*element_matrix_tmp307 + element_matrix_tmp255*element_matrix_tmp7 + element_matrix_tmp313)) + element_matrix_tmp177*element_matrix_tmp54;
    const s_t element_matrix_tmp315 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp21*element_matrix_tmp292 - eta_s*(-element_matrix_tmp12*element_matrix_tmp290 - element_matrix_tmp132 + element_matrix_tmp268*element_matrix_tmp7 - element_matrix_tmp300)) + element_matrix_tmp177*element_matrix_tmp261;
    const s_t element_matrix_tmp316 = basis0_grad0*element_matrix_tmp311 + basis0_grad1*element_matrix_tmp315 + basis0_grad2*element_matrix_tmp314;
    const s_t element_matrix_tmp317 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp303*element_matrix_tmp54 - eta_s*(-element_matrix_tmp255*element_matrix_tmp50 + element_matrix_tmp307*element_matrix_tmp51)) + element_matrix_tmp168*element_matrix_tmp54;
    const s_t element_matrix_tmp318 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp295*element_matrix_tmp54 - eta_s*(-element_matrix_tmp238*element_matrix_tmp50 + element_matrix_tmp299*element_matrix_tmp51 - element_matrix_tmp313)) + element_matrix_tmp168*element_matrix_tmp21;
    const s_t element_matrix_tmp319 = -element_matrix_tmp148*element_matrix_tmp248;
    const s_t element_matrix_tmp320 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp292*element_matrix_tmp54 - eta_s*(-element_matrix_tmp128 - element_matrix_tmp268*element_matrix_tmp50 + element_matrix_tmp290*element_matrix_tmp51 - element_matrix_tmp319)) + element_matrix_tmp168*element_matrix_tmp261;
    const s_t element_matrix_tmp321 = basis0_grad0*element_matrix_tmp318 + basis0_grad1*element_matrix_tmp320 + basis0_grad2*element_matrix_tmp317;
    const s_t element_matrix_tmp322 = element_matrix_tmp253 + element_matrix_tmp42*(-element_matrix_tmp110 + s_t(2)*element_matrix_tmp248*element_matrix_tmp30 - s_t(2)*element_matrix_tmp250 - element_matrix_tmp252);
    const s_t element_matrix_tmp323 = element_matrix_tmp11*(-element_matrix_tmp184*element_matrix_tmp322 + eta_s*(element_matrix_tmp257*element_matrix_tmp50 + element_matrix_tmp307*element_matrix_tmp54)) + element_matrix_tmp187*element_matrix_tmp54;
    const s_t element_matrix_tmp324 = element_matrix_tmp245 + element_matrix_tmp42*(-element_matrix_tmp22 - s_t(2)*element_matrix_tmp242 - element_matrix_tmp244 + s_t(2)*element_matrix_tmp28*u0_grad_1);
    const s_t element_matrix_tmp325 = element_matrix_tmp53*u2_grad_1;
    const s_t element_matrix_tmp326 = element_matrix_tmp11*(-element_matrix_tmp184*element_matrix_tmp324 - element_matrix_tmp186*u0_grad_1 + eta_s*(element_matrix_tmp240*element_matrix_tmp50 + element_matrix_tmp299*element_matrix_tmp54 - element_matrix_tmp325)) + element_matrix_tmp187*element_matrix_tmp21;
    const s_t element_matrix_tmp327 = element_matrix_tmp266 + element_matrix_tmp42*(-s_t(2)*element_matrix_tmp262 + s_t(2)*element_matrix_tmp263 + element_matrix_tmp265 + element_matrix_tmp90);
    const s_t element_matrix_tmp328 = element_matrix_tmp53*u2_grad_0;
    const s_t element_matrix_tmp329 = element_matrix_tmp11*(-element_matrix_tmp184*element_matrix_tmp327 + element_matrix_tmp186*element_matrix_tmp248 + eta_s*(element_matrix_tmp270*element_matrix_tmp50 + element_matrix_tmp290*element_matrix_tmp54 + element_matrix_tmp328)) + element_matrix_tmp187*element_matrix_tmp261;
    const s_t element_matrix_tmp330 = basis0_grad0*element_matrix_tmp326 + basis0_grad1*element_matrix_tmp329 + basis0_grad2*element_matrix_tmp323;
    const s_t element_matrix_tmp331 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp12*element_matrix_tmp324 - eta_s*(-element_matrix_tmp21*element_matrix_tmp299 + element_matrix_tmp240*element_matrix_tmp7)) + element_matrix_tmp205*element_matrix_tmp21;
    const s_t element_matrix_tmp332 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp12*element_matrix_tmp322 - element_matrix_tmp198*u0_grad_1 - eta_s*(-element_matrix_tmp21*element_matrix_tmp307 + element_matrix_tmp257*element_matrix_tmp7 - element_matrix_tmp325)) + element_matrix_tmp205*element_matrix_tmp54;
    const s_t element_matrix_tmp333 = element_matrix_tmp53*element_matrix_tmp6;
    const s_t element_matrix_tmp334 = element_matrix_tmp198*u0_grad_2;
    const s_t element_matrix_tmp335 = element_matrix_tmp11*(element_matrix_tmp208*element_matrix_tmp327 + element_matrix_tmp334 - eta_s*(-element_matrix_tmp21*element_matrix_tmp290 + element_matrix_tmp270*element_matrix_tmp7 + element_matrix_tmp333)) + element_matrix_tmp205*element_matrix_tmp261;
    const s_t element_matrix_tmp336 = basis0_grad0*element_matrix_tmp331 + basis0_grad1*element_matrix_tmp335 + basis0_grad2*element_matrix_tmp332;
    const s_t element_matrix_tmp337 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp327*element_matrix_tmp48 - eta_s*(-element_matrix_tmp270*element_matrix_tmp49 + element_matrix_tmp290*element_matrix_tmp55)) + element_matrix_tmp196*element_matrix_tmp261;
    const s_t element_matrix_tmp338 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp324*element_matrix_tmp48 - element_matrix_tmp334 - eta_s*(-element_matrix_tmp240*element_matrix_tmp49 + element_matrix_tmp299*element_matrix_tmp55 - element_matrix_tmp333)) + element_matrix_tmp196*element_matrix_tmp21;
    const s_t element_matrix_tmp339 = element_matrix_tmp11*(element_matrix_tmp198*element_matrix_tmp248 + element_matrix_tmp200*element_matrix_tmp322 - eta_s*(-element_matrix_tmp257*element_matrix_tmp49 + element_matrix_tmp307*element_matrix_tmp55 + element_matrix_tmp328)) + element_matrix_tmp196*element_matrix_tmp54;
    const s_t element_matrix_tmp340 = basis0_grad0*element_matrix_tmp338 + basis0_grad1*element_matrix_tmp337 + basis0_grad2*element_matrix_tmp339;
    const s_t element_matrix_tmp341 = basis1_grad0*element_matrix_tmp247 + basis1_grad1*element_matrix_tmp272 + basis1_grad2*element_matrix_tmp260;
    const s_t element_matrix_tmp342 = basis1_grad0*element_matrix_tmp276 + basis1_grad1*element_matrix_tmp280 + basis1_grad2*element_matrix_tmp274;
    const s_t element_matrix_tmp343 = basis1_grad0*element_matrix_tmp285 + basis1_grad1*element_matrix_tmp282 + basis1_grad2*element_matrix_tmp283;
    const s_t element_matrix_tmp344 = basis1_grad0*element_matrix_tmp301 + basis1_grad1*element_matrix_tmp293 + basis1_grad2*element_matrix_tmp309;
    const s_t element_matrix_tmp345 = basis1_grad0*element_matrix_tmp311 + basis1_grad1*element_matrix_tmp315 + basis1_grad2*element_matrix_tmp314;
    const s_t element_matrix_tmp346 = basis1_grad0*element_matrix_tmp318 + basis1_grad1*element_matrix_tmp320 + basis1_grad2*element_matrix_tmp317;
    const s_t element_matrix_tmp347 = basis1_grad0*element_matrix_tmp326 + basis1_grad1*element_matrix_tmp329 + basis1_grad2*element_matrix_tmp323;
    const s_t element_matrix_tmp348 = basis1_grad0*element_matrix_tmp331 + basis1_grad1*element_matrix_tmp335 + basis1_grad2*element_matrix_tmp332;
    const s_t element_matrix_tmp349 = basis1_grad0*element_matrix_tmp338 + basis1_grad1*element_matrix_tmp337 + basis1_grad2*element_matrix_tmp339;
    const s_t element_matrix_tmp350 = basis2_grad0*element_matrix_tmp247 + basis2_grad1*element_matrix_tmp272 + basis2_grad2*element_matrix_tmp260;
    const s_t element_matrix_tmp351 = basis2_grad0*element_matrix_tmp276 + basis2_grad1*element_matrix_tmp280 + basis2_grad2*element_matrix_tmp274;
    const s_t element_matrix_tmp352 = basis2_grad0*element_matrix_tmp285 + basis2_grad1*element_matrix_tmp282 + basis2_grad2*element_matrix_tmp283;
    const s_t element_matrix_tmp353 = basis2_grad0*element_matrix_tmp301 + basis2_grad1*element_matrix_tmp293 + basis2_grad2*element_matrix_tmp309;
    const s_t element_matrix_tmp354 = basis2_grad0*element_matrix_tmp311 + basis2_grad1*element_matrix_tmp315 + basis2_grad2*element_matrix_tmp314;
    const s_t element_matrix_tmp355 = basis2_grad0*element_matrix_tmp318 + basis2_grad1*element_matrix_tmp320 + basis2_grad2*element_matrix_tmp317;
    const s_t element_matrix_tmp356 = basis2_grad0*element_matrix_tmp326 + basis2_grad1*element_matrix_tmp329 + basis2_grad2*element_matrix_tmp323;
    const s_t element_matrix_tmp357 = basis2_grad0*element_matrix_tmp331 + basis2_grad1*element_matrix_tmp335 + basis2_grad2*element_matrix_tmp332;
    const s_t element_matrix_tmp358 = basis2_grad0*element_matrix_tmp338 + basis2_grad1*element_matrix_tmp337 + basis2_grad2*element_matrix_tmp339;
    const s_t element_matrix_tmp359 = basis3_grad0*element_matrix_tmp247 + basis3_grad1*element_matrix_tmp272 + basis3_grad2*element_matrix_tmp260;
    const s_t element_matrix_tmp360 = basis3_grad0*element_matrix_tmp276 + basis3_grad1*element_matrix_tmp280 + basis3_grad2*element_matrix_tmp274;
    const s_t element_matrix_tmp361 = basis3_grad0*element_matrix_tmp285 + basis3_grad1*element_matrix_tmp282 + basis3_grad2*element_matrix_tmp283;
    const s_t element_matrix_tmp362 = basis3_grad0*element_matrix_tmp301 + basis3_grad1*element_matrix_tmp293 + basis3_grad2*element_matrix_tmp309;
    const s_t element_matrix_tmp363 = basis3_grad0*element_matrix_tmp311 + basis3_grad1*element_matrix_tmp315 + basis3_grad2*element_matrix_tmp314;
    const s_t element_matrix_tmp364 = basis3_grad0*element_matrix_tmp318 + basis3_grad1*element_matrix_tmp320 + basis3_grad2*element_matrix_tmp317;
    const s_t element_matrix_tmp365 = basis3_grad0*element_matrix_tmp326 + basis3_grad1*element_matrix_tmp329 + basis3_grad2*element_matrix_tmp323;
    const s_t element_matrix_tmp366 = basis3_grad0*element_matrix_tmp331 + basis3_grad1*element_matrix_tmp335 + basis3_grad2*element_matrix_tmp332;
    const s_t element_matrix_tmp367 = basis3_grad0*element_matrix_tmp338 + basis3_grad1*element_matrix_tmp337 + basis3_grad2*element_matrix_tmp339;
    const s_t element_matrix_tmp368 = -element_matrix_tmp27 - element_matrix_tmp41;
    const s_t element_matrix_tmp369 = -element_matrix_tmp163 - element_matrix_tmp239;
    const s_t element_matrix_tmp370 = element_matrix_tmp13 - element_matrix_tmp296 + element_matrix_tmp297;
    const s_t element_matrix_tmp371 = -element_matrix_tmp15 + element_matrix_tmp18;
    const s_t element_matrix_tmp372 = element_matrix_tmp38*(element_matrix_tmp370 + element_matrix_tmp371);
    const s_t element_matrix_tmp373 = element_matrix_tmp372 + element_matrix_tmp42*(-s_t(2)*element_matrix_tmp15 + s_t(2)*element_matrix_tmp16*element_matrix_tmp17 - element_matrix_tmp370);
    const s_t element_matrix_tmp374 = element_matrix_tmp11*(-element_matrix_tmp373*element_matrix_tmp44 + eta_s*(element_matrix_tmp12*element_matrix_tmp368 + element_matrix_tmp21*element_matrix_tmp369)) + element_matrix_tmp12*element_matrix_tmp72;
    const s_t element_matrix_tmp375 = -element_matrix_tmp287 + element_matrix_tmp288 + element_matrix_tmp85;
    const s_t element_matrix_tmp376 = -element_matrix_tmp86 + element_matrix_tmp87;
    const s_t element_matrix_tmp377 = element_matrix_tmp38*(element_matrix_tmp375 + element_matrix_tmp376);
    const s_t element_matrix_tmp378 = element_matrix_tmp377 + element_matrix_tmp42*(-element_matrix_tmp375 + s_t(2)*element_matrix_tmp47*u1_grad_2 - s_t(2)*element_matrix_tmp86);
    const s_t element_matrix_tmp379 = element_matrix_tmp74 + element_matrix_tmp83;
    const s_t element_matrix_tmp380 = -element_matrix_tmp144 - element_matrix_tmp269;
    const s_t element_matrix_tmp381 = element_matrix_tmp57*u0_grad_2;
    const s_t element_matrix_tmp382 = element_matrix_tmp11*(-element_matrix_tmp259*u1_grad_2 - element_matrix_tmp378*element_matrix_tmp44 + eta_s*(element_matrix_tmp12*element_matrix_tmp379 + element_matrix_tmp21*element_matrix_tmp380 - element_matrix_tmp381)) + element_matrix_tmp48*element_matrix_tmp72;
    const s_t element_matrix_tmp383 = -element_matrix_tmp51;
    const s_t element_matrix_tmp384 = element_matrix_tmp115 - element_matrix_tmp304 + element_matrix_tmp305;
    const s_t element_matrix_tmp385 = -element_matrix_tmp116 + element_matrix_tmp117;
    const s_t element_matrix_tmp386 = element_matrix_tmp38*(-element_matrix_tmp384 - element_matrix_tmp385);
    const s_t element_matrix_tmp387 = element_matrix_tmp386 + element_matrix_tmp42*(s_t(2)*element_matrix_tmp116 - s_t(2)*element_matrix_tmp117 + element_matrix_tmp384);
    const s_t element_matrix_tmp388 = element_matrix_tmp108 + element_matrix_tmp99;
    const s_t element_matrix_tmp389 = element_matrix_tmp156 + element_matrix_tmp256;
    const s_t element_matrix_tmp390 = element_matrix_tmp57*u0_grad_1;
    const s_t element_matrix_tmp391 = element_matrix_tmp11*(element_matrix_tmp16*element_matrix_tmp259 - element_matrix_tmp387*element_matrix_tmp44 + eta_s*(element_matrix_tmp12*element_matrix_tmp388 + element_matrix_tmp21*element_matrix_tmp389 + element_matrix_tmp390)) + element_matrix_tmp383*element_matrix_tmp72;
    const s_t element_matrix_tmp392 = basis0_grad0*element_matrix_tmp374 + basis0_grad1*element_matrix_tmp382 + basis0_grad2*element_matrix_tmp391;
    const s_t element_matrix_tmp393 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp378*element_matrix_tmp49 - eta_s*(-element_matrix_tmp379*element_matrix_tmp48 + element_matrix_tmp380*element_matrix_tmp55)) + element_matrix_tmp125*element_matrix_tmp48;
    const s_t element_matrix_tmp394 = element_matrix_tmp11*(-element_matrix_tmp275*u1_grad_2 + ((s_t(1) / s_t(3)))*element_matrix_tmp373*element_matrix_tmp49 - eta_s*(-element_matrix_tmp368*element_matrix_tmp48 + element_matrix_tmp369*element_matrix_tmp55 - element_matrix_tmp381)) + element_matrix_tmp12*element_matrix_tmp125;
    const s_t element_matrix_tmp395 = element_matrix_tmp248*element_matrix_tmp57;
    const s_t element_matrix_tmp396 = element_matrix_tmp275*u1_grad_0;
    const s_t element_matrix_tmp397 = element_matrix_tmp11*(element_matrix_tmp284*element_matrix_tmp387 + element_matrix_tmp396 - eta_s*(-element_matrix_tmp388*element_matrix_tmp48 + element_matrix_tmp389*element_matrix_tmp55 + element_matrix_tmp395)) + element_matrix_tmp125*element_matrix_tmp383;
    const s_t element_matrix_tmp398 = basis0_grad0*element_matrix_tmp394 + basis0_grad1*element_matrix_tmp393 + basis0_grad2*element_matrix_tmp397;
    const s_t element_matrix_tmp399 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp387*element_matrix_tmp50 - eta_s*(element_matrix_tmp388*element_matrix_tmp51 - element_matrix_tmp389*element_matrix_tmp54)) + element_matrix_tmp135*element_matrix_tmp383;
    const s_t element_matrix_tmp400 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp378*element_matrix_tmp50 - element_matrix_tmp396 - eta_s*(element_matrix_tmp379*element_matrix_tmp51 - element_matrix_tmp380*element_matrix_tmp54 - element_matrix_tmp395)) + element_matrix_tmp135*element_matrix_tmp48;
    const s_t element_matrix_tmp401 = element_matrix_tmp11*(element_matrix_tmp16*element_matrix_tmp275 + element_matrix_tmp277*element_matrix_tmp373 - eta_s*(element_matrix_tmp368*element_matrix_tmp51 - element_matrix_tmp369*element_matrix_tmp54 + element_matrix_tmp390)) + element_matrix_tmp12*element_matrix_tmp135;
    const s_t element_matrix_tmp402 = basis0_grad0*element_matrix_tmp401 + basis0_grad1*element_matrix_tmp400 + basis0_grad2*element_matrix_tmp399;
    const s_t element_matrix_tmp403 = -element_matrix_tmp291 - element_matrix_tmp90;
    const s_t element_matrix_tmp404 = element_matrix_tmp377 + element_matrix_tmp42*(s_t(2)*element_matrix_tmp248*element_matrix_tmp35 - s_t(2)*element_matrix_tmp287 - element_matrix_tmp376 - element_matrix_tmp85);
    const s_t element_matrix_tmp405 = element_matrix_tmp11*(-element_matrix_tmp147*element_matrix_tmp404 + eta_s*(element_matrix_tmp380*element_matrix_tmp49 + element_matrix_tmp403*element_matrix_tmp48)) + element_matrix_tmp151*element_matrix_tmp48;
    const s_t element_matrix_tmp406 = element_matrix_tmp372 + element_matrix_tmp42*(-element_matrix_tmp13 - s_t(2)*element_matrix_tmp296 + s_t(2)*element_matrix_tmp33*u0_grad_2 - element_matrix_tmp371);
    const s_t element_matrix_tmp407 = element_matrix_tmp22 + element_matrix_tmp294;
    const s_t element_matrix_tmp408 = element_matrix_tmp57*u1_grad_2;
    const s_t element_matrix_tmp409 = element_matrix_tmp11*(-element_matrix_tmp147*element_matrix_tmp406 - element_matrix_tmp150*u0_grad_2 + eta_s*(element_matrix_tmp369*element_matrix_tmp49 + element_matrix_tmp407*element_matrix_tmp48 - element_matrix_tmp408)) + element_matrix_tmp12*element_matrix_tmp151;
    const s_t element_matrix_tmp410 = element_matrix_tmp386 + element_matrix_tmp42*(element_matrix_tmp115 + s_t(2)*element_matrix_tmp304 - s_t(2)*element_matrix_tmp305 + element_matrix_tmp385);
    const s_t element_matrix_tmp411 = element_matrix_tmp110 + element_matrix_tmp302;
    const s_t element_matrix_tmp412 = element_matrix_tmp57*u1_grad_0;
    const s_t element_matrix_tmp413 = element_matrix_tmp11*(-element_matrix_tmp147*element_matrix_tmp410 + element_matrix_tmp150*element_matrix_tmp248 + eta_s*(element_matrix_tmp389*element_matrix_tmp49 + element_matrix_tmp411*element_matrix_tmp48 + element_matrix_tmp412)) + element_matrix_tmp151*element_matrix_tmp383;
    const s_t element_matrix_tmp414 = basis0_grad0*element_matrix_tmp409 + basis0_grad1*element_matrix_tmp405 + basis0_grad2*element_matrix_tmp413;
    const s_t element_matrix_tmp415 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp21*element_matrix_tmp406 - eta_s*(-element_matrix_tmp12*element_matrix_tmp407 + element_matrix_tmp369*element_matrix_tmp7)) + element_matrix_tmp12*element_matrix_tmp177;
    const s_t element_matrix_tmp416 = element_matrix_tmp11*(-element_matrix_tmp170*u0_grad_2 + ((s_t(1) / s_t(3)))*element_matrix_tmp21*element_matrix_tmp404 - eta_s*(-element_matrix_tmp12*element_matrix_tmp403 + element_matrix_tmp380*element_matrix_tmp7 - element_matrix_tmp408)) + element_matrix_tmp177*element_matrix_tmp48;
    const s_t element_matrix_tmp417 = element_matrix_tmp16*element_matrix_tmp57;
    const s_t element_matrix_tmp418 = element_matrix_tmp170*u0_grad_1;
    const s_t element_matrix_tmp419 = element_matrix_tmp11*(element_matrix_tmp180*element_matrix_tmp410 + element_matrix_tmp418 - eta_s*(-element_matrix_tmp12*element_matrix_tmp411 + element_matrix_tmp389*element_matrix_tmp7 + element_matrix_tmp417)) + element_matrix_tmp177*element_matrix_tmp383;
    const s_t element_matrix_tmp420 = basis0_grad0*element_matrix_tmp415 + basis0_grad1*element_matrix_tmp416 + basis0_grad2*element_matrix_tmp419;
    const s_t element_matrix_tmp421 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp410*element_matrix_tmp54 - eta_s*(-element_matrix_tmp389*element_matrix_tmp50 + element_matrix_tmp411*element_matrix_tmp51)) + element_matrix_tmp168*element_matrix_tmp383;
    const s_t element_matrix_tmp422 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp406*element_matrix_tmp54 - element_matrix_tmp418 - eta_s*(-element_matrix_tmp369*element_matrix_tmp50 + element_matrix_tmp407*element_matrix_tmp51 - element_matrix_tmp417)) + element_matrix_tmp12*element_matrix_tmp168;
    const s_t element_matrix_tmp423 = element_matrix_tmp11*(element_matrix_tmp170*element_matrix_tmp248 + element_matrix_tmp172*element_matrix_tmp404 - eta_s*(-element_matrix_tmp380*element_matrix_tmp50 + element_matrix_tmp403*element_matrix_tmp51 + element_matrix_tmp412)) + element_matrix_tmp168*element_matrix_tmp48;
    const s_t element_matrix_tmp424 = basis0_grad0*element_matrix_tmp422 + basis0_grad1*element_matrix_tmp423 + basis0_grad2*element_matrix_tmp421;
    const s_t element_matrix_tmp425 = element_matrix_tmp386 + element_matrix_tmp42*(-s_t(2)*element_matrix_tmp115 - element_matrix_tmp118 - element_matrix_tmp306);
    const s_t element_matrix_tmp426 = element_matrix_tmp11*(-element_matrix_tmp184*element_matrix_tmp425 + eta_s*(element_matrix_tmp388*element_matrix_tmp50 + element_matrix_tmp411*element_matrix_tmp54)) + element_matrix_tmp187*element_matrix_tmp383;
    const s_t element_matrix_tmp427 = element_matrix_tmp372 + element_matrix_tmp42*(s_t(2)*element_matrix_tmp13 + element_matrix_tmp19 + element_matrix_tmp298);
    const s_t element_matrix_tmp428 = element_matrix_tmp11*(-element_matrix_tmp184*element_matrix_tmp427 + eta_s*(element_matrix_tmp122 - element_matrix_tmp312 + element_matrix_tmp368*element_matrix_tmp50 + element_matrix_tmp407*element_matrix_tmp54)) + element_matrix_tmp12*element_matrix_tmp187;
    const s_t element_matrix_tmp429 = element_matrix_tmp377 + element_matrix_tmp42*(element_matrix_tmp289 + s_t(2)*element_matrix_tmp85 + element_matrix_tmp88);
    const s_t element_matrix_tmp430 = element_matrix_tmp11*(-element_matrix_tmp184*element_matrix_tmp429 + eta_s*(-element_matrix_tmp127 + element_matrix_tmp308 + element_matrix_tmp379*element_matrix_tmp50 + element_matrix_tmp403*element_matrix_tmp54)) + element_matrix_tmp187*element_matrix_tmp48;
    const s_t element_matrix_tmp431 = basis0_grad0*element_matrix_tmp428 + basis0_grad1*element_matrix_tmp430 + basis0_grad2*element_matrix_tmp426;
    const s_t element_matrix_tmp432 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp12*element_matrix_tmp427 - eta_s*(-element_matrix_tmp21*element_matrix_tmp407 + element_matrix_tmp368*element_matrix_tmp7)) + element_matrix_tmp12*element_matrix_tmp205;
    const s_t element_matrix_tmp433 = element_matrix_tmp300 + element_matrix_tmp96;
    const s_t element_matrix_tmp434 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp12*element_matrix_tmp429 - eta_s*(-element_matrix_tmp21*element_matrix_tmp403 + element_matrix_tmp379*element_matrix_tmp7 + element_matrix_tmp433)) + element_matrix_tmp205*element_matrix_tmp48;
    const s_t element_matrix_tmp435 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp12*element_matrix_tmp425 - eta_s*(-element_matrix_tmp138 - element_matrix_tmp21*element_matrix_tmp411 - element_matrix_tmp312 + element_matrix_tmp388*element_matrix_tmp7)) + element_matrix_tmp205*element_matrix_tmp383;
    const s_t element_matrix_tmp436 = basis0_grad0*element_matrix_tmp432 + basis0_grad1*element_matrix_tmp434 + basis0_grad2*element_matrix_tmp435;
    const s_t element_matrix_tmp437 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp429*element_matrix_tmp48 - eta_s*(-element_matrix_tmp379*element_matrix_tmp49 + element_matrix_tmp403*element_matrix_tmp55)) + element_matrix_tmp196*element_matrix_tmp48;
    const s_t element_matrix_tmp438 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp427*element_matrix_tmp48 - eta_s*(-element_matrix_tmp368*element_matrix_tmp49 + element_matrix_tmp407*element_matrix_tmp55 - element_matrix_tmp433)) + element_matrix_tmp12*element_matrix_tmp196;
    const s_t element_matrix_tmp439 = element_matrix_tmp11*(((s_t(1) / s_t(3)))*element_matrix_tmp425*element_matrix_tmp48 - eta_s*(-element_matrix_tmp127 - element_matrix_tmp319 - element_matrix_tmp388*element_matrix_tmp49 + element_matrix_tmp411*element_matrix_tmp55)) + element_matrix_tmp196*element_matrix_tmp383;
    const s_t element_matrix_tmp440 = basis0_grad0*element_matrix_tmp438 + basis0_grad1*element_matrix_tmp437 + basis0_grad2*element_matrix_tmp439;
    const s_t element_matrix_tmp441 = basis1_grad0*element_matrix_tmp374 + basis1_grad1*element_matrix_tmp382 + basis1_grad2*element_matrix_tmp391;
    const s_t element_matrix_tmp442 = basis1_grad0*element_matrix_tmp394 + basis1_grad1*element_matrix_tmp393 + basis1_grad2*element_matrix_tmp397;
    const s_t element_matrix_tmp443 = basis1_grad0*element_matrix_tmp401 + basis1_grad1*element_matrix_tmp400 + basis1_grad2*element_matrix_tmp399;
    const s_t element_matrix_tmp444 = basis1_grad0*element_matrix_tmp409 + basis1_grad1*element_matrix_tmp405 + basis1_grad2*element_matrix_tmp413;
    const s_t element_matrix_tmp445 = basis1_grad0*element_matrix_tmp415 + basis1_grad1*element_matrix_tmp416 + basis1_grad2*element_matrix_tmp419;
    const s_t element_matrix_tmp446 = basis1_grad0*element_matrix_tmp422 + basis1_grad1*element_matrix_tmp423 + basis1_grad2*element_matrix_tmp421;
    const s_t element_matrix_tmp447 = basis1_grad0*element_matrix_tmp428 + basis1_grad1*element_matrix_tmp430 + basis1_grad2*element_matrix_tmp426;
    const s_t element_matrix_tmp448 = basis1_grad0*element_matrix_tmp432 + basis1_grad1*element_matrix_tmp434 + basis1_grad2*element_matrix_tmp435;
    const s_t element_matrix_tmp449 = basis1_grad0*element_matrix_tmp438 + basis1_grad1*element_matrix_tmp437 + basis1_grad2*element_matrix_tmp439;
    const s_t element_matrix_tmp450 = basis2_grad0*element_matrix_tmp374 + basis2_grad1*element_matrix_tmp382 + basis2_grad2*element_matrix_tmp391;
    const s_t element_matrix_tmp451 = basis2_grad0*element_matrix_tmp394 + basis2_grad1*element_matrix_tmp393 + basis2_grad2*element_matrix_tmp397;
    const s_t element_matrix_tmp452 = basis2_grad0*element_matrix_tmp401 + basis2_grad1*element_matrix_tmp400 + basis2_grad2*element_matrix_tmp399;
    const s_t element_matrix_tmp453 = basis2_grad0*element_matrix_tmp409 + basis2_grad1*element_matrix_tmp405 + basis2_grad2*element_matrix_tmp413;
    const s_t element_matrix_tmp454 = basis2_grad0*element_matrix_tmp415 + basis2_grad1*element_matrix_tmp416 + basis2_grad2*element_matrix_tmp419;
    const s_t element_matrix_tmp455 = basis2_grad0*element_matrix_tmp422 + basis2_grad1*element_matrix_tmp423 + basis2_grad2*element_matrix_tmp421;
    const s_t element_matrix_tmp456 = basis2_grad0*element_matrix_tmp428 + basis2_grad1*element_matrix_tmp430 + basis2_grad2*element_matrix_tmp426;
    const s_t element_matrix_tmp457 = basis2_grad0*element_matrix_tmp432 + basis2_grad1*element_matrix_tmp434 + basis2_grad2*element_matrix_tmp435;
    const s_t element_matrix_tmp458 = basis2_grad0*element_matrix_tmp438 + basis2_grad1*element_matrix_tmp437 + basis2_grad2*element_matrix_tmp439;
    const s_t element_matrix_tmp459 = basis3_grad0*element_matrix_tmp374 + basis3_grad1*element_matrix_tmp382 + basis3_grad2*element_matrix_tmp391;
    const s_t element_matrix_tmp460 = basis3_grad0*element_matrix_tmp394 + basis3_grad1*element_matrix_tmp393 + basis3_grad2*element_matrix_tmp397;
    const s_t element_matrix_tmp461 = basis3_grad0*element_matrix_tmp401 + basis3_grad1*element_matrix_tmp400 + basis3_grad2*element_matrix_tmp399;
    const s_t element_matrix_tmp462 = basis3_grad0*element_matrix_tmp409 + basis3_grad1*element_matrix_tmp405 + basis3_grad2*element_matrix_tmp413;
    const s_t element_matrix_tmp463 = basis3_grad0*element_matrix_tmp415 + basis3_grad1*element_matrix_tmp416 + basis3_grad2*element_matrix_tmp419;
    const s_t element_matrix_tmp464 = basis3_grad0*element_matrix_tmp422 + basis3_grad1*element_matrix_tmp423 + basis3_grad2*element_matrix_tmp421;
    const s_t element_matrix_tmp465 = basis3_grad0*element_matrix_tmp428 + basis3_grad1*element_matrix_tmp430 + basis3_grad2*element_matrix_tmp426;
    const s_t element_matrix_tmp466 = basis3_grad0*element_matrix_tmp432 + basis3_grad1*element_matrix_tmp434 + basis3_grad2*element_matrix_tmp435;
    const s_t element_matrix_tmp467 = basis3_grad0*element_matrix_tmp438 + basis3_grad1*element_matrix_tmp437 + basis3_grad2*element_matrix_tmp439;
    element_matrix[0] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp124 + basis0_grad1*element_matrix_tmp134 + basis0_grad2*element_matrix_tmp140);
    element_matrix[12] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp124 + basis1_grad1*element_matrix_tmp134 + basis1_grad2*element_matrix_tmp140);
    element_matrix[24] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp124 + basis2_grad1*element_matrix_tmp134 + basis2_grad2*element_matrix_tmp140);
    element_matrix[36] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp124 + basis3_grad1*element_matrix_tmp134 + basis3_grad2*element_matrix_tmp140);
    element_matrix[48] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp182 + basis0_grad1*element_matrix_tmp167 + basis0_grad2*element_matrix_tmp176);
    element_matrix[60] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp182 + basis1_grad1*element_matrix_tmp167 + basis1_grad2*element_matrix_tmp176);
    element_matrix[72] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp182 + basis2_grad1*element_matrix_tmp167 + basis2_grad2*element_matrix_tmp176);
    element_matrix[84] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp182 + basis3_grad1*element_matrix_tmp167 + basis3_grad2*element_matrix_tmp176);
    element_matrix[96] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp210 + basis0_grad1*element_matrix_tmp204 + basis0_grad2*element_matrix_tmp195);
    element_matrix[108] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp210 + basis1_grad1*element_matrix_tmp204 + basis1_grad2*element_matrix_tmp195);
    element_matrix[120] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp210 + basis2_grad1*element_matrix_tmp204 + basis2_grad2*element_matrix_tmp195);
    element_matrix[132] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp210 + basis3_grad1*element_matrix_tmp204 + basis3_grad2*element_matrix_tmp195);
    element_matrix[1] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp211 + basis0_grad1*element_matrix_tmp212 + basis0_grad2*element_matrix_tmp213);
    element_matrix[13] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp211 + basis1_grad1*element_matrix_tmp212 + basis1_grad2*element_matrix_tmp213);
    element_matrix[25] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp211 + basis2_grad1*element_matrix_tmp212 + basis2_grad2*element_matrix_tmp213);
    element_matrix[37] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp211 + basis3_grad1*element_matrix_tmp212 + basis3_grad2*element_matrix_tmp213);
    element_matrix[49] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp216 + basis0_grad1*element_matrix_tmp214 + basis0_grad2*element_matrix_tmp215);
    element_matrix[61] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp216 + basis1_grad1*element_matrix_tmp214 + basis1_grad2*element_matrix_tmp215);
    element_matrix[73] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp216 + basis2_grad1*element_matrix_tmp214 + basis2_grad2*element_matrix_tmp215);
    element_matrix[85] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp216 + basis3_grad1*element_matrix_tmp214 + basis3_grad2*element_matrix_tmp215);
    element_matrix[97] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp219 + basis0_grad1*element_matrix_tmp218 + basis0_grad2*element_matrix_tmp217);
    element_matrix[109] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp219 + basis1_grad1*element_matrix_tmp218 + basis1_grad2*element_matrix_tmp217);
    element_matrix[121] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp219 + basis2_grad1*element_matrix_tmp218 + basis2_grad2*element_matrix_tmp217);
    element_matrix[133] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp219 + basis3_grad1*element_matrix_tmp218 + basis3_grad2*element_matrix_tmp217);
    element_matrix[2] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp220 + basis0_grad1*element_matrix_tmp221 + basis0_grad2*element_matrix_tmp222);
    element_matrix[14] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp220 + basis1_grad1*element_matrix_tmp221 + basis1_grad2*element_matrix_tmp222);
    element_matrix[26] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp220 + basis2_grad1*element_matrix_tmp221 + basis2_grad2*element_matrix_tmp222);
    element_matrix[38] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp220 + basis3_grad1*element_matrix_tmp221 + basis3_grad2*element_matrix_tmp222);
    element_matrix[50] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp225 + basis0_grad1*element_matrix_tmp223 + basis0_grad2*element_matrix_tmp224);
    element_matrix[62] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp225 + basis1_grad1*element_matrix_tmp223 + basis1_grad2*element_matrix_tmp224);
    element_matrix[74] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp225 + basis2_grad1*element_matrix_tmp223 + basis2_grad2*element_matrix_tmp224);
    element_matrix[86] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp225 + basis3_grad1*element_matrix_tmp223 + basis3_grad2*element_matrix_tmp224);
    element_matrix[98] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp228 + basis0_grad1*element_matrix_tmp227 + basis0_grad2*element_matrix_tmp226);
    element_matrix[110] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp228 + basis1_grad1*element_matrix_tmp227 + basis1_grad2*element_matrix_tmp226);
    element_matrix[122] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp228 + basis2_grad1*element_matrix_tmp227 + basis2_grad2*element_matrix_tmp226);
    element_matrix[134] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp228 + basis3_grad1*element_matrix_tmp227 + basis3_grad2*element_matrix_tmp226);
    element_matrix[3] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp229 + basis0_grad1*element_matrix_tmp230 + basis0_grad2*element_matrix_tmp231);
    element_matrix[15] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp229 + basis1_grad1*element_matrix_tmp230 + basis1_grad2*element_matrix_tmp231);
    element_matrix[27] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp229 + basis2_grad1*element_matrix_tmp230 + basis2_grad2*element_matrix_tmp231);
    element_matrix[39] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp229 + basis3_grad1*element_matrix_tmp230 + basis3_grad2*element_matrix_tmp231);
    element_matrix[51] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp234 + basis0_grad1*element_matrix_tmp232 + basis0_grad2*element_matrix_tmp233);
    element_matrix[63] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp234 + basis1_grad1*element_matrix_tmp232 + basis1_grad2*element_matrix_tmp233);
    element_matrix[75] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp234 + basis2_grad1*element_matrix_tmp232 + basis2_grad2*element_matrix_tmp233);
    element_matrix[87] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp234 + basis3_grad1*element_matrix_tmp232 + basis3_grad2*element_matrix_tmp233);
    element_matrix[99] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp237 + basis0_grad1*element_matrix_tmp236 + basis0_grad2*element_matrix_tmp235);
    element_matrix[111] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp237 + basis1_grad1*element_matrix_tmp236 + basis1_grad2*element_matrix_tmp235);
    element_matrix[123] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp237 + basis2_grad1*element_matrix_tmp236 + basis2_grad2*element_matrix_tmp235);
    element_matrix[135] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp237 + basis3_grad1*element_matrix_tmp236 + basis3_grad2*element_matrix_tmp235);
    element_matrix[4] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp273 + basis0_grad1*element_matrix_tmp286 + basis0_grad2*element_matrix_tmp281);
    element_matrix[16] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp273 + basis1_grad1*element_matrix_tmp286 + basis1_grad2*element_matrix_tmp281);
    element_matrix[28] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp273 + basis2_grad1*element_matrix_tmp286 + basis2_grad2*element_matrix_tmp281);
    element_matrix[40] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp273 + basis3_grad1*element_matrix_tmp286 + basis3_grad2*element_matrix_tmp281);
    element_matrix[52] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp316 + basis0_grad1*element_matrix_tmp310 + basis0_grad2*element_matrix_tmp321);
    element_matrix[64] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp316 + basis1_grad1*element_matrix_tmp310 + basis1_grad2*element_matrix_tmp321);
    element_matrix[76] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp316 + basis2_grad1*element_matrix_tmp310 + basis2_grad2*element_matrix_tmp321);
    element_matrix[88] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp316 + basis3_grad1*element_matrix_tmp310 + basis3_grad2*element_matrix_tmp321);
    element_matrix[100] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp336 + basis0_grad1*element_matrix_tmp340 + basis0_grad2*element_matrix_tmp330);
    element_matrix[112] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp336 + basis1_grad1*element_matrix_tmp340 + basis1_grad2*element_matrix_tmp330);
    element_matrix[124] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp336 + basis2_grad1*element_matrix_tmp340 + basis2_grad2*element_matrix_tmp330);
    element_matrix[136] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp336 + basis3_grad1*element_matrix_tmp340 + basis3_grad2*element_matrix_tmp330);
    element_matrix[5] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp341 + basis0_grad1*element_matrix_tmp343 + basis0_grad2*element_matrix_tmp342);
    element_matrix[17] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp341 + basis1_grad1*element_matrix_tmp343 + basis1_grad2*element_matrix_tmp342);
    element_matrix[29] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp341 + basis2_grad1*element_matrix_tmp343 + basis2_grad2*element_matrix_tmp342);
    element_matrix[41] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp341 + basis3_grad1*element_matrix_tmp343 + basis3_grad2*element_matrix_tmp342);
    element_matrix[53] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp345 + basis0_grad1*element_matrix_tmp344 + basis0_grad2*element_matrix_tmp346);
    element_matrix[65] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp345 + basis1_grad1*element_matrix_tmp344 + basis1_grad2*element_matrix_tmp346);
    element_matrix[77] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp345 + basis2_grad1*element_matrix_tmp344 + basis2_grad2*element_matrix_tmp346);
    element_matrix[89] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp345 + basis3_grad1*element_matrix_tmp344 + basis3_grad2*element_matrix_tmp346);
    element_matrix[101] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp348 + basis0_grad1*element_matrix_tmp349 + basis0_grad2*element_matrix_tmp347);
    element_matrix[113] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp348 + basis1_grad1*element_matrix_tmp349 + basis1_grad2*element_matrix_tmp347);
    element_matrix[125] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp348 + basis2_grad1*element_matrix_tmp349 + basis2_grad2*element_matrix_tmp347);
    element_matrix[137] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp348 + basis3_grad1*element_matrix_tmp349 + basis3_grad2*element_matrix_tmp347);
    element_matrix[6] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp350 + basis0_grad1*element_matrix_tmp352 + basis0_grad2*element_matrix_tmp351);
    element_matrix[18] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp350 + basis1_grad1*element_matrix_tmp352 + basis1_grad2*element_matrix_tmp351);
    element_matrix[30] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp350 + basis2_grad1*element_matrix_tmp352 + basis2_grad2*element_matrix_tmp351);
    element_matrix[42] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp350 + basis3_grad1*element_matrix_tmp352 + basis3_grad2*element_matrix_tmp351);
    element_matrix[54] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp354 + basis0_grad1*element_matrix_tmp353 + basis0_grad2*element_matrix_tmp355);
    element_matrix[66] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp354 + basis1_grad1*element_matrix_tmp353 + basis1_grad2*element_matrix_tmp355);
    element_matrix[78] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp354 + basis2_grad1*element_matrix_tmp353 + basis2_grad2*element_matrix_tmp355);
    element_matrix[90] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp354 + basis3_grad1*element_matrix_tmp353 + basis3_grad2*element_matrix_tmp355);
    element_matrix[102] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp357 + basis0_grad1*element_matrix_tmp358 + basis0_grad2*element_matrix_tmp356);
    element_matrix[114] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp357 + basis1_grad1*element_matrix_tmp358 + basis1_grad2*element_matrix_tmp356);
    element_matrix[126] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp357 + basis2_grad1*element_matrix_tmp358 + basis2_grad2*element_matrix_tmp356);
    element_matrix[138] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp357 + basis3_grad1*element_matrix_tmp358 + basis3_grad2*element_matrix_tmp356);
    element_matrix[7] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp359 + basis0_grad1*element_matrix_tmp361 + basis0_grad2*element_matrix_tmp360);
    element_matrix[19] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp359 + basis1_grad1*element_matrix_tmp361 + basis1_grad2*element_matrix_tmp360);
    element_matrix[31] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp359 + basis2_grad1*element_matrix_tmp361 + basis2_grad2*element_matrix_tmp360);
    element_matrix[43] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp359 + basis3_grad1*element_matrix_tmp361 + basis3_grad2*element_matrix_tmp360);
    element_matrix[55] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp363 + basis0_grad1*element_matrix_tmp362 + basis0_grad2*element_matrix_tmp364);
    element_matrix[67] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp363 + basis1_grad1*element_matrix_tmp362 + basis1_grad2*element_matrix_tmp364);
    element_matrix[79] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp363 + basis2_grad1*element_matrix_tmp362 + basis2_grad2*element_matrix_tmp364);
    element_matrix[91] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp363 + basis3_grad1*element_matrix_tmp362 + basis3_grad2*element_matrix_tmp364);
    element_matrix[103] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp366 + basis0_grad1*element_matrix_tmp367 + basis0_grad2*element_matrix_tmp365);
    element_matrix[115] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp366 + basis1_grad1*element_matrix_tmp367 + basis1_grad2*element_matrix_tmp365);
    element_matrix[127] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp366 + basis2_grad1*element_matrix_tmp367 + basis2_grad2*element_matrix_tmp365);
    element_matrix[139] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp366 + basis3_grad1*element_matrix_tmp367 + basis3_grad2*element_matrix_tmp365);
    element_matrix[8] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp392 + basis0_grad1*element_matrix_tmp398 + basis0_grad2*element_matrix_tmp402);
    element_matrix[20] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp392 + basis1_grad1*element_matrix_tmp398 + basis1_grad2*element_matrix_tmp402);
    element_matrix[32] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp392 + basis2_grad1*element_matrix_tmp398 + basis2_grad2*element_matrix_tmp402);
    element_matrix[44] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp392 + basis3_grad1*element_matrix_tmp398 + basis3_grad2*element_matrix_tmp402);
    element_matrix[56] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp420 + basis0_grad1*element_matrix_tmp414 + basis0_grad2*element_matrix_tmp424);
    element_matrix[68] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp420 + basis1_grad1*element_matrix_tmp414 + basis1_grad2*element_matrix_tmp424);
    element_matrix[80] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp420 + basis2_grad1*element_matrix_tmp414 + basis2_grad2*element_matrix_tmp424);
    element_matrix[92] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp420 + basis3_grad1*element_matrix_tmp414 + basis3_grad2*element_matrix_tmp424);
    element_matrix[104] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp436 + basis0_grad1*element_matrix_tmp440 + basis0_grad2*element_matrix_tmp431);
    element_matrix[116] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp436 + basis1_grad1*element_matrix_tmp440 + basis1_grad2*element_matrix_tmp431);
    element_matrix[128] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp436 + basis2_grad1*element_matrix_tmp440 + basis2_grad2*element_matrix_tmp431);
    element_matrix[140] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp436 + basis3_grad1*element_matrix_tmp440 + basis3_grad2*element_matrix_tmp431);
    element_matrix[9] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp441 + basis0_grad1*element_matrix_tmp442 + basis0_grad2*element_matrix_tmp443);
    element_matrix[21] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp441 + basis1_grad1*element_matrix_tmp442 + basis1_grad2*element_matrix_tmp443);
    element_matrix[33] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp441 + basis2_grad1*element_matrix_tmp442 + basis2_grad2*element_matrix_tmp443);
    element_matrix[45] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp441 + basis3_grad1*element_matrix_tmp442 + basis3_grad2*element_matrix_tmp443);
    element_matrix[57] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp445 + basis0_grad1*element_matrix_tmp444 + basis0_grad2*element_matrix_tmp446);
    element_matrix[69] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp445 + basis1_grad1*element_matrix_tmp444 + basis1_grad2*element_matrix_tmp446);
    element_matrix[81] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp445 + basis2_grad1*element_matrix_tmp444 + basis2_grad2*element_matrix_tmp446);
    element_matrix[93] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp445 + basis3_grad1*element_matrix_tmp444 + basis3_grad2*element_matrix_tmp446);
    element_matrix[105] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp448 + basis0_grad1*element_matrix_tmp449 + basis0_grad2*element_matrix_tmp447);
    element_matrix[117] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp448 + basis1_grad1*element_matrix_tmp449 + basis1_grad2*element_matrix_tmp447);
    element_matrix[129] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp448 + basis2_grad1*element_matrix_tmp449 + basis2_grad2*element_matrix_tmp447);
    element_matrix[141] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp448 + basis3_grad1*element_matrix_tmp449 + basis3_grad2*element_matrix_tmp447);
    element_matrix[10] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp450 + basis0_grad1*element_matrix_tmp451 + basis0_grad2*element_matrix_tmp452);
    element_matrix[22] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp450 + basis1_grad1*element_matrix_tmp451 + basis1_grad2*element_matrix_tmp452);
    element_matrix[34] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp450 + basis2_grad1*element_matrix_tmp451 + basis2_grad2*element_matrix_tmp452);
    element_matrix[46] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp450 + basis3_grad1*element_matrix_tmp451 + basis3_grad2*element_matrix_tmp452);
    element_matrix[58] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp454 + basis0_grad1*element_matrix_tmp453 + basis0_grad2*element_matrix_tmp455);
    element_matrix[70] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp454 + basis1_grad1*element_matrix_tmp453 + basis1_grad2*element_matrix_tmp455);
    element_matrix[82] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp454 + basis2_grad1*element_matrix_tmp453 + basis2_grad2*element_matrix_tmp455);
    element_matrix[94] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp454 + basis3_grad1*element_matrix_tmp453 + basis3_grad2*element_matrix_tmp455);
    element_matrix[106] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp457 + basis0_grad1*element_matrix_tmp458 + basis0_grad2*element_matrix_tmp456);
    element_matrix[118] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp457 + basis1_grad1*element_matrix_tmp458 + basis1_grad2*element_matrix_tmp456);
    element_matrix[130] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp457 + basis2_grad1*element_matrix_tmp458 + basis2_grad2*element_matrix_tmp456);
    element_matrix[142] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp457 + basis3_grad1*element_matrix_tmp458 + basis3_grad2*element_matrix_tmp456);
    element_matrix[11] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp459 + basis0_grad1*element_matrix_tmp460 + basis0_grad2*element_matrix_tmp461);
    element_matrix[23] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp459 + basis1_grad1*element_matrix_tmp460 + basis1_grad2*element_matrix_tmp461);
    element_matrix[35] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp459 + basis2_grad1*element_matrix_tmp460 + basis2_grad2*element_matrix_tmp461);
    element_matrix[47] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp459 + basis3_grad1*element_matrix_tmp460 + basis3_grad2*element_matrix_tmp461);
    element_matrix[59] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp463 + basis0_grad1*element_matrix_tmp462 + basis0_grad2*element_matrix_tmp464);
    element_matrix[71] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp463 + basis1_grad1*element_matrix_tmp462 + basis1_grad2*element_matrix_tmp464);
    element_matrix[83] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp463 + basis2_grad1*element_matrix_tmp462 + basis2_grad2*element_matrix_tmp464);
    element_matrix[95] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp463 + basis3_grad1*element_matrix_tmp462 + basis3_grad2*element_matrix_tmp464);
    element_matrix[107] = element_matrix_tmp141*(basis0_grad0*element_matrix_tmp466 + basis0_grad1*element_matrix_tmp467 + basis0_grad2*element_matrix_tmp465);
    element_matrix[119] = element_matrix_tmp141*(basis1_grad0*element_matrix_tmp466 + basis1_grad1*element_matrix_tmp467 + basis1_grad2*element_matrix_tmp465);
    element_matrix[131] = element_matrix_tmp141*(basis2_grad0*element_matrix_tmp466 + basis2_grad1*element_matrix_tmp467 + basis2_grad2*element_matrix_tmp465);
    element_matrix[143] = element_matrix_tmp141*(basis3_grad0*element_matrix_tmp466 + basis3_grad1*element_matrix_tmp467 + basis3_grad2*element_matrix_tmp465);
  }
}

template <typename s_t, int NQ, int NS, int VS>
__host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_hessian_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t current[3 * NS][VS],
    const s_t previous[3 * NS][VS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t newmark_velocity_alpha,
    s_t *const RSTR element_matrix
) {
  static constexpr int NC = 3;
  for (int entry = 0; entry < 3 * NS * 3 * NS; ++entry) {
    element_matrix[entry] = s_t(0);
  }
  s_t u0_grad_0_ref_values[VS];
  s_t u0_grad_1_ref_values[VS];
  s_t u0_grad_2_ref_values[VS];
  s_t u0_old_grad_0_ref_values[VS];
  s_t u0_old_grad_1_ref_values[VS];
  s_t u0_old_grad_2_ref_values[VS];
  s_t u1_grad_0_ref_values[VS];
  s_t u1_grad_1_ref_values[VS];
  s_t u1_grad_2_ref_values[VS];
  s_t u1_old_grad_0_ref_values[VS];
  s_t u1_old_grad_1_ref_values[VS];
  s_t u1_old_grad_2_ref_values[VS];
  s_t u2_grad_0_ref_values[VS];
  s_t u2_grad_1_ref_values[VS];
  s_t u2_grad_2_ref_values[VS];
  s_t u2_old_grad_0_ref_values[VS];
  s_t u2_old_grad_1_ref_values[VS];
  s_t u2_old_grad_2_ref_values[VS];
  s_t grad_coeff0_0_values[VS];
  s_t grad_coeff0_1_values[VS];
  s_t grad_coeff0_2_values[VS];
  s_t grad_coeff1_0_values[VS];
  s_t grad_coeff1_1_values[VS];
  s_t grad_coeff1_2_values[VS];
  s_t grad_coeff2_0_values[VS];
  s_t grad_coeff2_1_values[VS];
  s_t grad_coeff2_2_values[VS];
  for (int q = 0; q < NQ; ++q) {
    {
      u0_grad_0_ref_values[0] = s_t(0);
      u0_grad_1_ref_values[0] = s_t(0);
      u0_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC][0];
        u0_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u0_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u0_old_grad_0_ref_values[0] = s_t(0);
      u0_old_grad_1_ref_values[0] = s_t(0);
      u0_old_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC][0];
        u0_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u0_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u0_old_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u1_grad_0_ref_values[0] = s_t(0);
      u1_grad_1_ref_values[0] = s_t(0);
      u1_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 1][0];
        u1_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u1_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u1_old_grad_0_ref_values[0] = s_t(0);
      u1_old_grad_1_ref_values[0] = s_t(0);
      u1_old_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC + 1][0];
        u1_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u1_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u1_old_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u2_grad_0_ref_values[0] = s_t(0);
      u2_grad_1_ref_values[0] = s_t(0);
      u2_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = current[trial * NC + 2][0];
        u2_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u2_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u2_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    {
      u2_old_grad_0_ref_values[0] = s_t(0);
      u2_old_grad_1_ref_values[0] = s_t(0);
      u2_old_grad_2_ref_values[0] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      {
        const s_t coeff = previous[trial * NC + 2][0];
        u2_old_grad_0_ref_values[0] += coeff * grad_ref_x[q * NS + trial];
        u2_old_grad_1_ref_values[0] += coeff * grad_ref_y[q * NS + trial];
        u2_old_grad_2_ref_values[0] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    for (int trial = 0; trial < NS; ++trial) {
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
        const s_t u0_grad_0_ref = u0_grad_0_ref_values[0];
        const s_t u0_grad_1_ref = u0_grad_1_ref_values[0];
        const s_t u0_grad_2_ref = u0_grad_2_ref_values[0];
        const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
        const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
        const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
        const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[0];
        const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[0];
        const s_t u0_old_grad_2_ref = u0_old_grad_2_ref_values[0];
        const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
        const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
        const s_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
        const s_t u1_grad_0_ref = u1_grad_0_ref_values[0];
        const s_t u1_grad_1_ref = u1_grad_1_ref_values[0];
        const s_t u1_grad_2_ref = u1_grad_2_ref_values[0];
        const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
        const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
        const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
        const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[0];
        const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[0];
        const s_t u1_old_grad_2_ref = u1_old_grad_2_ref_values[0];
        const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
        const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
        const s_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
        const s_t u2_grad_0_ref = u2_grad_0_ref_values[0];
        const s_t u2_grad_1_ref = u2_grad_1_ref_values[0];
        const s_t u2_grad_2_ref = u2_grad_2_ref_values[0];
        const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
        const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
        const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
        const s_t u2_old_grad_0_ref = u2_old_grad_0_ref_values[0];
        const s_t u2_old_grad_1_ref = u2_old_grad_1_ref_values[0];
        const s_t u2_old_grad_2_ref = u2_old_grad_2_ref_values[0];
        const s_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
        const s_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
        const s_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
        const s_t trial_grad0 = (grad_ref_x[q * NS + trial] * adj0 + grad_ref_y[q * NS + trial] * adj3 + grad_ref_z[q * NS + trial] * adj6) / det;
        const s_t trial_grad1 = (grad_ref_x[q * NS + trial] * adj1 + grad_ref_y[q * NS + trial] * adj4 + grad_ref_z[q * NS + trial] * adj7) / det;
        const s_t trial_grad2 = (grad_ref_x[q * NS + trial] * adj2 + grad_ref_y[q * NS + trial] * adj5 + grad_ref_z[q * NS + trial] * adj8) / det;
        const s_t residual_tmp0 = u0_grad_0*u1_grad_1;
        const s_t residual_tmp1 = u0_grad_1*u1_grad_2;
        const s_t residual_tmp2 = u0_grad_2*u2_grad_1;
        const s_t residual_tmp3 = u1_grad_2*u2_grad_1;
        const s_t residual_tmp4 = u0_grad_1*u1_grad_0;
        const s_t residual_tmp5 = u0_grad_2*u2_grad_0;
        const s_t residual_tmp6 = u2_grad_2 + s_t(1);
        const s_t residual_tmp7 = -residual_tmp3 + residual_tmp6 + u1_grad_1*u2_grad_2 + u1_grad_1;
        const s_t residual_tmp8 = -residual_tmp5 + u0_grad_0*u2_grad_2 + u0_grad_0;
        const s_t residual_tmp9 = residual_tmp0 - residual_tmp4;
        const s_t residual_tmp10 = residual_tmp0*u2_grad_2 + residual_tmp1*u2_grad_0 + residual_tmp2*u1_grad_0 - residual_tmp3*u0_grad_0 - residual_tmp4*u2_grad_2 - residual_tmp5*u1_grad_1 + residual_tmp7 + residual_tmp8 + residual_tmp9;
        const s_t residual_tmp11 = pow_m1(residual_tmp10);
        const s_t residual_tmp12 = -residual_tmp1 + u0_grad_2*u1_grad_1 + u0_grad_2;
        const s_t residual_tmp13 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
        const s_t residual_tmp14 = u1_grad_1 + s_t(1);
        const s_t residual_tmp15 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
        const s_t residual_tmp16 = newmark_velocity_alpha*residual_tmp12 + residual_tmp13*u1_grad_2 - residual_tmp14*residual_tmp15;
        const s_t residual_tmp17 = -residual_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
        const s_t residual_tmp18 = newmark_velocity_alpha*residual_tmp17 - residual_tmp13*residual_tmp6 + residual_tmp15*u2_grad_1;
        const s_t residual_tmp19 = newmark_velocity_alpha*residual_tmp7;
        const s_t residual_tmp20 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
        const s_t residual_tmp21 = residual_tmp14*residual_tmp20;
        const s_t residual_tmp22 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
        const s_t residual_tmp23 = residual_tmp22*u1_grad_2;
        const s_t residual_tmp24 = residual_tmp19 + residual_tmp21 - residual_tmp23;
        const s_t residual_tmp25 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
        const s_t residual_tmp26 = residual_tmp25*residual_tmp6;
        const s_t residual_tmp27 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
        const s_t residual_tmp28 = residual_tmp27*u2_grad_1;
        const s_t residual_tmp29 = residual_tmp26 - residual_tmp28;
        const s_t residual_tmp30 = s_t(3)*eta_b;
        const s_t residual_tmp31 = residual_tmp30*(-residual_tmp24 - residual_tmp29);
        const s_t residual_tmp32 = s_t(2)*eta_s;
        const s_t residual_tmp33 = residual_tmp31 + residual_tmp32*(-s_t(2)*residual_tmp19 + residual_tmp21 - residual_tmp23 + residual_tmp26 - residual_tmp28);
        const s_t residual_tmp34 = ((s_t(1) / s_t(3)))*residual_tmp7;
        const s_t residual_tmp35 = -residual_tmp7;
        const s_t residual_tmp36 = pow_m2(residual_tmp10);
        const s_t residual_tmp37 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
        const s_t residual_tmp38 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
        const s_t residual_tmp39 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
        const s_t residual_tmp40 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
        const s_t residual_tmp41 = residual_tmp14 + residual_tmp9 + u0_grad_0;
        const s_t residual_tmp42 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
        const s_t residual_tmp43 = residual_tmp12*residual_tmp37 + residual_tmp13*residual_tmp38 - residual_tmp15*residual_tmp41 + residual_tmp20*residual_tmp40 + residual_tmp22*residual_tmp39 - residual_tmp42*residual_tmp7;
        const s_t residual_tmp44 = u0_grad_0*u2_grad_1 - u0_grad_1*u2_grad_0 + u2_grad_1;
        const s_t residual_tmp45 = residual_tmp6 + residual_tmp8;
        const s_t residual_tmp46 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
        const s_t residual_tmp47 = -residual_tmp13*residual_tmp45 + residual_tmp15*residual_tmp44 + residual_tmp17*residual_tmp37 + residual_tmp25*residual_tmp39 + residual_tmp27*residual_tmp40 - residual_tmp46*residual_tmp7;
        const s_t residual_tmp48 = residual_tmp17*residual_tmp46;
        const s_t residual_tmp49 = residual_tmp27*residual_tmp44;
        const s_t residual_tmp50 = residual_tmp12*residual_tmp42;
        const s_t residual_tmp51 = residual_tmp22*residual_tmp38;
        const s_t residual_tmp52 = residual_tmp25*residual_tmp45;
        const s_t residual_tmp53 = -residual_tmp52;
        const s_t residual_tmp54 = residual_tmp20*residual_tmp41;
        const s_t residual_tmp55 = -residual_tmp54;
        const s_t residual_tmp56 = residual_tmp48 + residual_tmp49 + residual_tmp50 + residual_tmp51 + residual_tmp53 + residual_tmp55;
        const s_t residual_tmp57 = residual_tmp37*residual_tmp7;
        const s_t residual_tmp58 = residual_tmp13*residual_tmp39 + residual_tmp15*residual_tmp40 - residual_tmp57;
        const s_t residual_tmp59 = residual_tmp30*(residual_tmp56 + residual_tmp58);
        const s_t residual_tmp60 = residual_tmp32*(s_t(2)*residual_tmp13*residual_tmp39 + s_t(2)*residual_tmp15*residual_tmp40 - residual_tmp56 - s_t(2)*residual_tmp57) + residual_tmp59;
        const s_t residual_tmp61 = residual_tmp36*(eta_s*(residual_tmp12*residual_tmp43 + residual_tmp17*residual_tmp47) - residual_tmp34*residual_tmp60);
        const s_t residual_tmp62 = newmark_velocity_alpha*residual_tmp39;
        const s_t residual_tmp63 = residual_tmp20*u1_grad_0;
        const s_t residual_tmp64 = residual_tmp42*u1_grad_2;
        const s_t residual_tmp65 = residual_tmp62 + residual_tmp63 - residual_tmp64;
        const s_t residual_tmp66 = residual_tmp46*residual_tmp6;
        const s_t residual_tmp67 = residual_tmp27*u2_grad_0;
        const s_t residual_tmp68 = residual_tmp66 - residual_tmp67;
        const s_t residual_tmp69 = residual_tmp30*(residual_tmp65 + residual_tmp68);
        const s_t residual_tmp70 = residual_tmp32*(s_t(2)*residual_tmp62 - residual_tmp63 + residual_tmp64 - residual_tmp66 + residual_tmp67) + residual_tmp69;
        const s_t residual_tmp71 = newmark_velocity_alpha*residual_tmp38 + residual_tmp15*u1_grad_0 - residual_tmp37*u1_grad_2;
        const s_t residual_tmp72 = -newmark_velocity_alpha*residual_tmp45 - residual_tmp15*u2_grad_0 + residual_tmp37*residual_tmp6;
        const s_t residual_tmp73 = residual_tmp43*u1_grad_2;
        const s_t residual_tmp74 = newmark_velocity_alpha*residual_tmp40;
        const s_t residual_tmp75 = residual_tmp14*residual_tmp42;
        const s_t residual_tmp76 = residual_tmp22*u1_grad_0;
        const s_t residual_tmp77 = residual_tmp74 + residual_tmp75 - residual_tmp76;
        const s_t residual_tmp78 = residual_tmp25*u2_grad_0;
        const s_t residual_tmp79 = residual_tmp46*u2_grad_1;
        const s_t residual_tmp80 = residual_tmp78 - residual_tmp79;
        const s_t residual_tmp81 = residual_tmp30*(residual_tmp77 + residual_tmp80);
        const s_t residual_tmp82 = residual_tmp32*(s_t(2)*residual_tmp74 - residual_tmp75 + residual_tmp76 - residual_tmp78 + residual_tmp79) + residual_tmp81;
        const s_t residual_tmp83 = newmark_velocity_alpha*residual_tmp44 + residual_tmp13*u2_grad_0 - residual_tmp37*u2_grad_1;
        const s_t residual_tmp84 = -newmark_velocity_alpha*residual_tmp41 - residual_tmp13*u1_grad_0 + residual_tmp14*residual_tmp37;
        const s_t residual_tmp85 = residual_tmp47*u2_grad_1;
        const s_t residual_tmp86 = residual_tmp36*(-eta_s*(-residual_tmp38*residual_tmp43 + residual_tmp45*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp39*residual_tmp60);
        const s_t residual_tmp87 = residual_tmp43*u1_grad_0 - residual_tmp47*u2_grad_0;
        const s_t residual_tmp88 = residual_tmp36*(-eta_s*(residual_tmp41*residual_tmp43 - residual_tmp44*residual_tmp47) + ((s_t(1) / s_t(3)))*residual_tmp40*residual_tmp60);
        const s_t residual_tmp89 = -residual_tmp14*residual_tmp27 + residual_tmp20*u2_grad_1 - residual_tmp22*residual_tmp6 + residual_tmp25*u1_grad_2;
        const s_t residual_tmp90 = residual_tmp31 + residual_tmp32*(residual_tmp24 - s_t(2)*residual_tmp26 + s_t(2)*residual_tmp28);
        const s_t residual_tmp91 = residual_tmp12*residual_tmp46 + residual_tmp17*residual_tmp42 + residual_tmp20*residual_tmp44 - residual_tmp22*residual_tmp45 + residual_tmp25*residual_tmp38 - residual_tmp27*residual_tmp41;
        const s_t residual_tmp92 = residual_tmp32*(s_t(2)*residual_tmp17*residual_tmp46 + s_t(2)*residual_tmp27*residual_tmp44 - residual_tmp50 - residual_tmp51 - s_t(2)*residual_tmp52 - residual_tmp55 - residual_tmp58) + residual_tmp59;
        const s_t residual_tmp93 = residual_tmp36*(-eta_s*(-residual_tmp12*residual_tmp91 + residual_tmp47*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp17*residual_tmp92);
        const s_t residual_tmp94 = residual_tmp32*(s_t(2)*residual_tmp25*u2_grad_0 - residual_tmp77 - s_t(2)*residual_tmp79) + residual_tmp81;
        const s_t residual_tmp95 = residual_tmp14*residual_tmp46 + residual_tmp22*u2_grad_0 - residual_tmp25*u1_grad_0 - residual_tmp42*u2_grad_1;
        const s_t residual_tmp96 = residual_tmp14*residual_tmp91;
        const s_t residual_tmp97 = ((s_t(1) / s_t(3)))*residual_tmp92;
        const s_t residual_tmp98 = residual_tmp97*u2_grad_1;
        const s_t residual_tmp99 = residual_tmp32*(s_t(2)*residual_tmp46*residual_tmp6 - residual_tmp65 - s_t(2)*residual_tmp67) + residual_tmp69;
        const s_t residual_tmp100 = -residual_tmp20*u2_grad_0 + residual_tmp27*u1_grad_0 + residual_tmp42*residual_tmp6 - residual_tmp46*u1_grad_2;
        const s_t residual_tmp101 = residual_tmp91*u1_grad_2;
        const s_t residual_tmp102 = ((s_t(1) / s_t(3)))*residual_tmp45;
        const s_t residual_tmp103 = -(s_t(1) / s_t(3))*residual_tmp92;
        const s_t residual_tmp104 = residual_tmp36*(eta_s*(residual_tmp38*residual_tmp91 + residual_tmp39*residual_tmp47) + residual_tmp103*residual_tmp45);
        const s_t residual_tmp105 = residual_tmp91*u1_grad_0;
        const s_t residual_tmp106 = residual_tmp36*(-eta_s*(-residual_tmp40*residual_tmp47 + residual_tmp41*residual_tmp91) + ((s_t(1) / s_t(3)))*residual_tmp44*residual_tmp92);
        const s_t residual_tmp107 = residual_tmp31 + residual_tmp32*(residual_tmp19 - s_t(2)*residual_tmp21 + s_t(2)*residual_tmp23 + residual_tmp29);
        const s_t residual_tmp108 = residual_tmp32*(s_t(2)*residual_tmp12*residual_tmp42 + s_t(2)*residual_tmp22*residual_tmp38 - residual_tmp48 - residual_tmp49 - residual_tmp53 - s_t(2)*residual_tmp54 - residual_tmp58) + residual_tmp59;
        const s_t residual_tmp109 = residual_tmp36*(-eta_s*(-residual_tmp17*residual_tmp91 + residual_tmp43*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp108*residual_tmp12);
        const s_t residual_tmp110 = residual_tmp32*(s_t(2)*residual_tmp20*u1_grad_0 - residual_tmp62 - s_t(2)*residual_tmp64 - residual_tmp68) + residual_tmp69;
        const s_t residual_tmp111 = residual_tmp6*residual_tmp91;
        const s_t residual_tmp112 = ((s_t(1) / s_t(3)))*residual_tmp108;
        const s_t residual_tmp113 = residual_tmp112*u1_grad_2;
        const s_t residual_tmp114 = residual_tmp32*(s_t(2)*residual_tmp14*residual_tmp42 - residual_tmp74 - s_t(2)*residual_tmp76 - residual_tmp80) + residual_tmp81;
        const s_t residual_tmp115 = residual_tmp91*u2_grad_1;
        const s_t residual_tmp116 = residual_tmp36*(-eta_s*(-residual_tmp39*residual_tmp43 + residual_tmp45*residual_tmp91) + ((s_t(1) / s_t(3)))*residual_tmp108*residual_tmp38);
        const s_t residual_tmp117 = residual_tmp91*u2_grad_0;
        const s_t residual_tmp118 = ((s_t(1) / s_t(3)))*residual_tmp41;
        const s_t residual_tmp119 = -(s_t(1) / s_t(3))*residual_tmp108;
        const s_t residual_tmp120 = residual_tmp36*(eta_s*(residual_tmp40*residual_tmp43 + residual_tmp44*residual_tmp91) + residual_tmp119*residual_tmp41);
        const s_t grad_coeff0_0 = trial_grad0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp16 + residual_tmp17*residual_tmp18) - residual_tmp33*residual_tmp34) + residual_tmp35*residual_tmp61) + trial_grad1*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp71 + residual_tmp17*residual_tmp72 + residual_tmp47*residual_tmp6 - residual_tmp73) - residual_tmp34*residual_tmp70) + residual_tmp39*residual_tmp61) + trial_grad2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp84 + residual_tmp14*residual_tmp43 + residual_tmp17*residual_tmp83 - residual_tmp85) - residual_tmp34*residual_tmp82) + residual_tmp40*residual_tmp61);
        const s_t grad_coeff0_1 = trial_grad0*(residual_tmp11*(-eta_s*(-residual_tmp16*residual_tmp38 + residual_tmp18*residual_tmp45 + residual_tmp47*residual_tmp6 - residual_tmp73) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp39) + residual_tmp35*residual_tmp86) + trial_grad1*(residual_tmp11*(-eta_s*(-residual_tmp38*residual_tmp71 + residual_tmp45*residual_tmp72) + ((s_t(1) / s_t(3)))*residual_tmp39*residual_tmp70) + residual_tmp39*residual_tmp86) + trial_grad2*(residual_tmp11*(-eta_s*(-residual_tmp38*residual_tmp84 + residual_tmp45*residual_tmp83 + residual_tmp87) + ((s_t(1) / s_t(3)))*residual_tmp39*residual_tmp82) + residual_tmp40*residual_tmp86);
        const s_t grad_coeff0_2 = trial_grad0*(residual_tmp11*(-eta_s*(residual_tmp14*residual_tmp43 + residual_tmp16*residual_tmp41 - residual_tmp18*residual_tmp44 - residual_tmp85) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp40) + residual_tmp35*residual_tmp88) + trial_grad1*(residual_tmp11*(-eta_s*(residual_tmp41*residual_tmp71 - residual_tmp44*residual_tmp72 - residual_tmp87) + ((s_t(1) / s_t(3)))*residual_tmp40*residual_tmp70) + residual_tmp39*residual_tmp88) + trial_grad2*(residual_tmp11*(-eta_s*(residual_tmp41*residual_tmp84 - residual_tmp44*residual_tmp83) + ((s_t(1) / s_t(3)))*residual_tmp40*residual_tmp82) + residual_tmp40*residual_tmp88);
        const s_t grad_coeff1_0 = trial_grad0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp89 + residual_tmp18*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp17*residual_tmp90) + residual_tmp35*residual_tmp93) + trial_grad1*(residual_tmp11*(-eta_s*(-residual_tmp100*residual_tmp12 + residual_tmp101 + residual_tmp7*residual_tmp72) + ((s_t(1) / s_t(3)))*residual_tmp17*residual_tmp99 + residual_tmp6*residual_tmp97) + residual_tmp39*residual_tmp93) + trial_grad2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp95 + residual_tmp7*residual_tmp83 - residual_tmp96) + ((s_t(1) / s_t(3)))*residual_tmp17*residual_tmp94 - residual_tmp98) + residual_tmp40*residual_tmp93);
        const s_t grad_coeff1_1 = trial_grad0*(residual_tmp104*residual_tmp35 + residual_tmp11*(eta_s*(residual_tmp101 + residual_tmp18*residual_tmp39 + residual_tmp38*residual_tmp89) - residual_tmp102*residual_tmp90 + residual_tmp103*residual_tmp6)) + trial_grad1*(residual_tmp104*residual_tmp39 + residual_tmp11*(eta_s*(residual_tmp100*residual_tmp38 + residual_tmp39*residual_tmp72) - residual_tmp102*residual_tmp99)) + trial_grad2*(residual_tmp104*residual_tmp40 + residual_tmp11*(eta_s*(-residual_tmp105 + residual_tmp38*residual_tmp95 + residual_tmp39*residual_tmp83) - residual_tmp102*residual_tmp94 - residual_tmp103*u2_grad_0));
        const s_t grad_coeff1_2 = trial_grad0*(residual_tmp106*residual_tmp35 + residual_tmp11*(-eta_s*(-residual_tmp18*residual_tmp40 + residual_tmp41*residual_tmp89 + residual_tmp96) + ((s_t(1) / s_t(3)))*residual_tmp44*residual_tmp90 + residual_tmp98)) + trial_grad1*(residual_tmp106*residual_tmp39 + residual_tmp11*(-eta_s*(residual_tmp100*residual_tmp41 - residual_tmp105 - residual_tmp40*residual_tmp72) + ((s_t(1) / s_t(3)))*residual_tmp44*residual_tmp99 - residual_tmp97*u2_grad_0)) + trial_grad2*(residual_tmp106*residual_tmp40 + residual_tmp11*(-eta_s*(-residual_tmp40*residual_tmp83 + residual_tmp41*residual_tmp95) + ((s_t(1) / s_t(3)))*residual_tmp44*residual_tmp94));
        const s_t grad_coeff2_0 = trial_grad0*(residual_tmp109*residual_tmp35 + residual_tmp11*(-eta_s*(residual_tmp16*residual_tmp7 - residual_tmp17*residual_tmp89) + ((s_t(1) / s_t(3)))*residual_tmp107*residual_tmp12)) + trial_grad1*(residual_tmp109*residual_tmp39 + residual_tmp11*(-eta_s*(-residual_tmp100*residual_tmp17 - residual_tmp111 + residual_tmp7*residual_tmp71) + ((s_t(1) / s_t(3)))*residual_tmp110*residual_tmp12 - residual_tmp113)) + trial_grad2*(residual_tmp109*residual_tmp40 + residual_tmp11*(-eta_s*(residual_tmp115 - residual_tmp17*residual_tmp95 + residual_tmp7*residual_tmp84) + residual_tmp112*residual_tmp14 + ((s_t(1) / s_t(3)))*residual_tmp114*residual_tmp12));
        const s_t grad_coeff2_1 = trial_grad0*(residual_tmp11*(-eta_s*(residual_tmp111 - residual_tmp16*residual_tmp39 + residual_tmp45*residual_tmp89) + ((s_t(1) / s_t(3)))*residual_tmp107*residual_tmp38 + residual_tmp113) + residual_tmp116*residual_tmp35) + trial_grad1*(residual_tmp11*(-eta_s*(residual_tmp100*residual_tmp45 - residual_tmp39*residual_tmp71) + ((s_t(1) / s_t(3)))*residual_tmp110*residual_tmp38) + residual_tmp116*residual_tmp39) + trial_grad2*(residual_tmp11*(-eta_s*(-residual_tmp117 - residual_tmp39*residual_tmp84 + residual_tmp45*residual_tmp95) - residual_tmp112*u1_grad_0 + ((s_t(1) / s_t(3)))*residual_tmp114*residual_tmp38) + residual_tmp116*residual_tmp40);
        const s_t grad_coeff2_2 = trial_grad0*(residual_tmp11*(eta_s*(residual_tmp115 + residual_tmp16*residual_tmp40 + residual_tmp44*residual_tmp89) - residual_tmp107*residual_tmp118 + residual_tmp119*residual_tmp14) + residual_tmp120*residual_tmp35) + trial_grad1*(residual_tmp11*(eta_s*(residual_tmp100*residual_tmp44 - residual_tmp117 + residual_tmp40*residual_tmp71) - residual_tmp110*residual_tmp118 - residual_tmp119*u1_grad_0) + residual_tmp120*residual_tmp39) + trial_grad2*(residual_tmp11*(eta_s*(residual_tmp40*residual_tmp84 + residual_tmp44*residual_tmp95) - residual_tmp114*residual_tmp118) + residual_tmp120*residual_tmp40);
        grad_coeff0_0_values[0] = grad_coeff0_0;
        grad_coeff0_1_values[0] = grad_coeff0_1;
        grad_coeff0_2_values[0] = grad_coeff0_2;
        grad_coeff1_0_values[0] = grad_coeff1_0;
        grad_coeff1_1_values[0] = grad_coeff1_1;
        grad_coeff1_2_values[0] = grad_coeff1_2;
        grad_coeff2_0_values[0] = grad_coeff2_0;
        grad_coeff2_1_values[0] = grad_coeff2_1;
        grad_coeff2_2_values[0] = grad_coeff2_2;
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
          const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj3 + grad_ref_z[q * NS + test] * adj6) / det;
          const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj4 + grad_ref_z[q * NS + test] * adj7) / det;
          const s_t test_grad2 = (grad_ref_x[q * NS + test] * adj2 + grad_ref_y[q * NS + test] * adj5 + grad_ref_z[q * NS + test] * adj8) / det;
          element_matrix[(0 * NS + test) * 3 * NS + 0 * NS + trial] += q_weight[q] * det * (grad_coeff0_0_values[0] * test_grad0 + grad_coeff0_1_values[0] * test_grad1 + grad_coeff0_2_values[0] * test_grad2);
          element_matrix[(1 * NS + test) * 3 * NS + 0 * NS + trial] += q_weight[q] * det * (grad_coeff1_0_values[0] * test_grad0 + grad_coeff1_1_values[0] * test_grad1 + grad_coeff1_2_values[0] * test_grad2);
          element_matrix[(2 * NS + test) * 3 * NS + 0 * NS + trial] += q_weight[q] * det * (grad_coeff2_0_values[0] * test_grad0 + grad_coeff2_1_values[0] * test_grad1 + grad_coeff2_2_values[0] * test_grad2);
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
        const s_t u0_grad_0_ref = u0_grad_0_ref_values[0];
        const s_t u0_grad_1_ref = u0_grad_1_ref_values[0];
        const s_t u0_grad_2_ref = u0_grad_2_ref_values[0];
        const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
        const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
        const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
        const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[0];
        const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[0];
        const s_t u0_old_grad_2_ref = u0_old_grad_2_ref_values[0];
        const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
        const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
        const s_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
        const s_t u1_grad_0_ref = u1_grad_0_ref_values[0];
        const s_t u1_grad_1_ref = u1_grad_1_ref_values[0];
        const s_t u1_grad_2_ref = u1_grad_2_ref_values[0];
        const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
        const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
        const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
        const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[0];
        const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[0];
        const s_t u1_old_grad_2_ref = u1_old_grad_2_ref_values[0];
        const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
        const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
        const s_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
        const s_t u2_grad_0_ref = u2_grad_0_ref_values[0];
        const s_t u2_grad_1_ref = u2_grad_1_ref_values[0];
        const s_t u2_grad_2_ref = u2_grad_2_ref_values[0];
        const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
        const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
        const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
        const s_t u2_old_grad_0_ref = u2_old_grad_0_ref_values[0];
        const s_t u2_old_grad_1_ref = u2_old_grad_1_ref_values[0];
        const s_t u2_old_grad_2_ref = u2_old_grad_2_ref_values[0];
        const s_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
        const s_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
        const s_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
        const s_t trial_grad0 = (grad_ref_x[q * NS + trial] * adj0 + grad_ref_y[q * NS + trial] * adj3 + grad_ref_z[q * NS + trial] * adj6) / det;
        const s_t trial_grad1 = (grad_ref_x[q * NS + trial] * adj1 + grad_ref_y[q * NS + trial] * adj4 + grad_ref_z[q * NS + trial] * adj7) / det;
        const s_t trial_grad2 = (grad_ref_x[q * NS + trial] * adj2 + grad_ref_y[q * NS + trial] * adj5 + grad_ref_z[q * NS + trial] * adj8) / det;
        const s_t residual_tmp0 = u0_grad_0*u1_grad_1;
        const s_t residual_tmp1 = u0_grad_1*u1_grad_2;
        const s_t residual_tmp2 = u0_grad_2*u2_grad_1;
        const s_t residual_tmp3 = u1_grad_2*u2_grad_1;
        const s_t residual_tmp4 = u0_grad_1*u1_grad_0;
        const s_t residual_tmp5 = u0_grad_2*u2_grad_0;
        const s_t residual_tmp6 = u2_grad_2 + s_t(1);
        const s_t residual_tmp7 = -residual_tmp3 + residual_tmp6 + u1_grad_1*u2_grad_2 + u1_grad_1;
        const s_t residual_tmp8 = -residual_tmp5 + u0_grad_0*u2_grad_2 + u0_grad_0;
        const s_t residual_tmp9 = residual_tmp0 - residual_tmp4;
        const s_t residual_tmp10 = residual_tmp0*u2_grad_2 + residual_tmp1*u2_grad_0 + residual_tmp2*u1_grad_0 - residual_tmp3*u0_grad_0 - residual_tmp4*u2_grad_2 - residual_tmp5*u1_grad_1 + residual_tmp7 + residual_tmp8 + residual_tmp9;
        const s_t residual_tmp11 = pow_m1(residual_tmp10);
        const s_t residual_tmp12 = -residual_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
        const s_t residual_tmp13 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
        const s_t residual_tmp14 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
        const s_t residual_tmp15 = -newmark_velocity_alpha*residual_tmp7 - residual_tmp13*u2_grad_1 + residual_tmp14*residual_tmp6;
        const s_t residual_tmp16 = -residual_tmp1 + u0_grad_2*u1_grad_1 + u0_grad_2;
        const s_t residual_tmp17 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
        const s_t residual_tmp18 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
        const s_t residual_tmp19 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
        const s_t residual_tmp20 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
        const s_t residual_tmp21 = residual_tmp17*u0_grad_1 - residual_tmp18*u0_grad_2 - residual_tmp19*u2_grad_1 + residual_tmp20*residual_tmp6;
        const s_t residual_tmp22 = newmark_velocity_alpha*residual_tmp12;
        const s_t residual_tmp23 = residual_tmp19*u0_grad_1;
        const s_t residual_tmp24 = residual_tmp20*u0_grad_2;
        const s_t residual_tmp25 = residual_tmp22 + residual_tmp23 - residual_tmp24;
        const s_t residual_tmp26 = residual_tmp18*residual_tmp6;
        const s_t residual_tmp27 = residual_tmp17*u2_grad_1;
        const s_t residual_tmp28 = residual_tmp26 - residual_tmp27;
        const s_t residual_tmp29 = s_t(3)*eta_b;
        const s_t residual_tmp30 = residual_tmp29*(residual_tmp25 + residual_tmp28);
        const s_t residual_tmp31 = s_t(2)*eta_s;
        const s_t residual_tmp32 = residual_tmp30 + residual_tmp31*(s_t(2)*residual_tmp18*residual_tmp6 - residual_tmp25 - s_t(2)*residual_tmp27);
        const s_t residual_tmp33 = ((s_t(1) / s_t(3)))*residual_tmp7;
        const s_t residual_tmp34 = pow_m2(residual_tmp10);
        const s_t residual_tmp35 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
        const s_t residual_tmp36 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
        const s_t residual_tmp37 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
        const s_t residual_tmp38 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
        const s_t residual_tmp39 = u0_grad_0 + s_t(1);
        const s_t residual_tmp40 = residual_tmp39 + residual_tmp9 + u1_grad_1;
        const s_t residual_tmp41 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
        const s_t residual_tmp42 = residual_tmp16*residual_tmp35 - residual_tmp17*residual_tmp40 + residual_tmp18*residual_tmp36 + residual_tmp19*residual_tmp38 + residual_tmp20*residual_tmp37 - residual_tmp41*residual_tmp7;
        const s_t residual_tmp43 = u0_grad_0*u2_grad_1 - u0_grad_1*u2_grad_0 + u2_grad_1;
        const s_t residual_tmp44 = residual_tmp6 + residual_tmp8;
        const s_t residual_tmp45 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
        const s_t residual_tmp46 = residual_tmp12*residual_tmp35 + residual_tmp13*residual_tmp38 + residual_tmp14*residual_tmp37 + residual_tmp17*residual_tmp43 - residual_tmp18*residual_tmp44 - residual_tmp45*residual_tmp7;
        const s_t residual_tmp47 = residual_tmp12*residual_tmp45;
        const s_t residual_tmp48 = residual_tmp13*residual_tmp43;
        const s_t residual_tmp49 = residual_tmp16*residual_tmp41;
        const s_t residual_tmp50 = residual_tmp20*residual_tmp36;
        const s_t residual_tmp51 = residual_tmp14*residual_tmp44;
        const s_t residual_tmp52 = -residual_tmp51;
        const s_t residual_tmp53 = residual_tmp19*residual_tmp40;
        const s_t residual_tmp54 = -residual_tmp53;
        const s_t residual_tmp55 = residual_tmp47 + residual_tmp48 + residual_tmp49 + residual_tmp50 + residual_tmp52 + residual_tmp54;
        const s_t residual_tmp56 = residual_tmp35*residual_tmp7;
        const s_t residual_tmp57 = residual_tmp17*residual_tmp38 + residual_tmp18*residual_tmp37 - residual_tmp56;
        const s_t residual_tmp58 = residual_tmp29*(residual_tmp55 + residual_tmp57);
        const s_t residual_tmp59 = residual_tmp31*(s_t(2)*residual_tmp17*residual_tmp38 + s_t(2)*residual_tmp18*residual_tmp37 - residual_tmp55 - s_t(2)*residual_tmp56) + residual_tmp58;
        const s_t residual_tmp60 = -residual_tmp59;
        const s_t residual_tmp61 = residual_tmp34*(eta_s*(residual_tmp12*residual_tmp46 + residual_tmp16*residual_tmp42) + residual_tmp33*residual_tmp60);
        const s_t residual_tmp62 = newmark_velocity_alpha*residual_tmp43;
        const s_t residual_tmp63 = residual_tmp20*residual_tmp39;
        const s_t residual_tmp64 = residual_tmp41*u0_grad_1;
        const s_t residual_tmp65 = residual_tmp62 + residual_tmp63 - residual_tmp64;
        const s_t residual_tmp66 = residual_tmp35*u2_grad_1;
        const s_t residual_tmp67 = residual_tmp18*u2_grad_0;
        const s_t residual_tmp68 = residual_tmp66 - residual_tmp67;
        const s_t residual_tmp69 = residual_tmp29*(residual_tmp65 + residual_tmp68);
        const s_t residual_tmp70 = residual_tmp31*(s_t(2)*residual_tmp35*u2_grad_1 - residual_tmp65 - s_t(2)*residual_tmp67) + residual_tmp69;
        const s_t residual_tmp71 = newmark_velocity_alpha*residual_tmp38 - residual_tmp14*u2_grad_0 + residual_tmp45*u2_grad_1;
        const s_t residual_tmp72 = residual_tmp18*residual_tmp39 - residual_tmp20*u2_grad_0 - residual_tmp35*u0_grad_1 + residual_tmp41*u2_grad_1;
        const s_t residual_tmp73 = residual_tmp42*u0_grad_1;
        const s_t residual_tmp74 = ((s_t(1) / s_t(3)))*residual_tmp60;
        const s_t residual_tmp75 = -residual_tmp44;
        const s_t residual_tmp76 = newmark_velocity_alpha*residual_tmp44;
        const s_t residual_tmp77 = residual_tmp19*residual_tmp39;
        const s_t residual_tmp78 = residual_tmp41*u0_grad_2;
        const s_t residual_tmp79 = residual_tmp76 + residual_tmp77 - residual_tmp78;
        const s_t residual_tmp80 = residual_tmp35*residual_tmp6;
        const s_t residual_tmp81 = residual_tmp17*u2_grad_0;
        const s_t residual_tmp82 = residual_tmp80 - residual_tmp81;
        const s_t residual_tmp83 = residual_tmp29*(-residual_tmp79 - residual_tmp82);
        const s_t residual_tmp84 = residual_tmp31*(residual_tmp79 - s_t(2)*residual_tmp80 + s_t(2)*residual_tmp81) + residual_tmp83;
        const s_t residual_tmp85 = newmark_velocity_alpha*residual_tmp37 + residual_tmp13*u2_grad_0 - residual_tmp45*residual_tmp6;
        const s_t residual_tmp86 = -residual_tmp17*residual_tmp39 + residual_tmp19*u2_grad_0 + residual_tmp35*u0_grad_2 - residual_tmp41*residual_tmp6;
        const s_t residual_tmp87 = residual_tmp42*u0_grad_2;
        const s_t residual_tmp88 = residual_tmp34*(-eta_s*(-residual_tmp36*residual_tmp42 + residual_tmp44*residual_tmp46) + ((s_t(1) / s_t(3)))*residual_tmp37*residual_tmp59);
        const s_t residual_tmp89 = residual_tmp39*residual_tmp42;
        const s_t residual_tmp90 = ((s_t(1) / s_t(3)))*residual_tmp59;
        const s_t residual_tmp91 = residual_tmp90*u2_grad_0;
        const s_t residual_tmp92 = residual_tmp34*(-eta_s*(residual_tmp40*residual_tmp42 - residual_tmp43*residual_tmp46) + ((s_t(1) / s_t(3)))*residual_tmp38*residual_tmp59);
        const s_t residual_tmp93 = newmark_velocity_alpha*residual_tmp16 + residual_tmp13*u0_grad_1 - residual_tmp14*u0_grad_2;
        const s_t residual_tmp94 = residual_tmp30 + residual_tmp31*(s_t(2)*residual_tmp22 - residual_tmp23 + residual_tmp24 - residual_tmp26 + residual_tmp27);
        const s_t residual_tmp95 = residual_tmp12*residual_tmp41 - residual_tmp13*residual_tmp40 + residual_tmp14*residual_tmp36 + residual_tmp16*residual_tmp45 + residual_tmp19*residual_tmp43 - residual_tmp20*residual_tmp44;
        const s_t residual_tmp96 = residual_tmp31*(s_t(2)*residual_tmp12*residual_tmp45 + s_t(2)*residual_tmp13*residual_tmp43 - residual_tmp49 - residual_tmp50 - s_t(2)*residual_tmp51 - residual_tmp54 - residual_tmp57) + residual_tmp58;
        const s_t residual_tmp97 = residual_tmp34*(-eta_s*(-residual_tmp16*residual_tmp95 + residual_tmp46*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp96);
        const s_t residual_tmp98 = residual_tmp31*(s_t(2)*residual_tmp62 - residual_tmp63 + residual_tmp64 - residual_tmp66 + residual_tmp67) + residual_tmp69;
        const s_t residual_tmp99 = -newmark_velocity_alpha*residual_tmp40 + residual_tmp14*residual_tmp39 - residual_tmp45*u0_grad_1;
        const s_t residual_tmp100 = -residual_tmp46*u2_grad_1 + residual_tmp95*u0_grad_1;
        const s_t residual_tmp101 = residual_tmp31*(-s_t(2)*residual_tmp76 + residual_tmp77 - residual_tmp78 + residual_tmp80 - residual_tmp81) + residual_tmp83;
        const s_t residual_tmp102 = newmark_velocity_alpha*residual_tmp36 - residual_tmp13*residual_tmp39 + residual_tmp45*u0_grad_2;
        const s_t residual_tmp103 = residual_tmp95*u0_grad_2;
        const s_t residual_tmp104 = ((s_t(1) / s_t(3)))*residual_tmp44;
        const s_t residual_tmp105 = residual_tmp34*(eta_s*(residual_tmp36*residual_tmp95 + residual_tmp37*residual_tmp46) - residual_tmp104*residual_tmp96);
        const s_t residual_tmp106 = residual_tmp46*u2_grad_0;
        const s_t residual_tmp107 = residual_tmp34*(-eta_s*(-residual_tmp38*residual_tmp46 + residual_tmp40*residual_tmp95) + ((s_t(1) / s_t(3)))*residual_tmp43*residual_tmp96);
        const s_t residual_tmp108 = residual_tmp30 + residual_tmp31*(s_t(2)*residual_tmp19*u0_grad_1 - residual_tmp22 - s_t(2)*residual_tmp24 - residual_tmp28);
        const s_t residual_tmp109 = residual_tmp31*(s_t(2)*residual_tmp16*residual_tmp41 + s_t(2)*residual_tmp20*residual_tmp36 - residual_tmp47 - residual_tmp48 - residual_tmp52 - s_t(2)*residual_tmp53 - residual_tmp57) + residual_tmp58;
        const s_t residual_tmp110 = residual_tmp34*(-eta_s*(-residual_tmp12*residual_tmp95 + residual_tmp42*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp109*residual_tmp16);
        const s_t residual_tmp111 = residual_tmp31*(s_t(2)*residual_tmp20*residual_tmp39 - residual_tmp62 - s_t(2)*residual_tmp64 - residual_tmp68) + residual_tmp69;
        const s_t residual_tmp112 = residual_tmp42*u2_grad_1;
        const s_t residual_tmp113 = ((s_t(1) / s_t(3)))*residual_tmp109;
        const s_t residual_tmp114 = residual_tmp31*(residual_tmp76 - s_t(2)*residual_tmp77 + s_t(2)*residual_tmp78 + residual_tmp82) + residual_tmp83;
        const s_t residual_tmp115 = residual_tmp42*residual_tmp6;
        const s_t residual_tmp116 = residual_tmp113*u0_grad_2;
        const s_t residual_tmp117 = residual_tmp34*(-eta_s*(-residual_tmp37*residual_tmp42 + residual_tmp44*residual_tmp95) + ((s_t(1) / s_t(3)))*residual_tmp109*residual_tmp36);
        const s_t residual_tmp118 = residual_tmp42*u2_grad_0;
        const s_t residual_tmp119 = ((s_t(1) / s_t(3)))*residual_tmp40;
        const s_t residual_tmp120 = -(s_t(1) / s_t(3))*residual_tmp109;
        const s_t residual_tmp121 = residual_tmp34*(eta_s*(residual_tmp38*residual_tmp42 + residual_tmp43*residual_tmp95) + residual_tmp120*residual_tmp40);
        const s_t grad_coeff0_0 = trial_grad0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp15 + residual_tmp16*residual_tmp21) - residual_tmp32*residual_tmp33) + residual_tmp12*residual_tmp61) + trial_grad1*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp85 + residual_tmp16*residual_tmp86 + residual_tmp87) - residual_tmp33*residual_tmp84 + residual_tmp6*residual_tmp74) + residual_tmp61*residual_tmp75) + trial_grad2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp71 + residual_tmp16*residual_tmp72 - residual_tmp73) - residual_tmp33*residual_tmp70 - residual_tmp74*u2_grad_1) + residual_tmp43*residual_tmp61);
        const s_t grad_coeff0_1 = trial_grad0*(residual_tmp11*(-eta_s*(residual_tmp15*residual_tmp44 - residual_tmp21*residual_tmp36 + residual_tmp87) + ((s_t(1) / s_t(3)))*residual_tmp32*residual_tmp37 + residual_tmp6*residual_tmp90) + residual_tmp12*residual_tmp88) + trial_grad1*(residual_tmp11*(-eta_s*(-residual_tmp36*residual_tmp86 + residual_tmp44*residual_tmp85) + ((s_t(1) / s_t(3)))*residual_tmp37*residual_tmp84) + residual_tmp75*residual_tmp88) + trial_grad2*(residual_tmp11*(-eta_s*(-residual_tmp36*residual_tmp72 + residual_tmp44*residual_tmp71 - residual_tmp89) + ((s_t(1) / s_t(3)))*residual_tmp37*residual_tmp70 - residual_tmp91) + residual_tmp43*residual_tmp88);
        const s_t grad_coeff0_2 = trial_grad0*(residual_tmp11*(-eta_s*(-residual_tmp15*residual_tmp43 + residual_tmp21*residual_tmp40 - residual_tmp73) + ((s_t(1) / s_t(3)))*residual_tmp32*residual_tmp38 - residual_tmp90*u2_grad_1) + residual_tmp12*residual_tmp92) + trial_grad1*(residual_tmp11*(-eta_s*(residual_tmp40*residual_tmp86 - residual_tmp43*residual_tmp85 + residual_tmp89) + ((s_t(1) / s_t(3)))*residual_tmp38*residual_tmp84 + residual_tmp91) + residual_tmp75*residual_tmp92) + trial_grad2*(residual_tmp11*(-eta_s*(residual_tmp40*residual_tmp72 - residual_tmp43*residual_tmp71) + ((s_t(1) / s_t(3)))*residual_tmp38*residual_tmp70) + residual_tmp43*residual_tmp92);
        const s_t grad_coeff1_0 = trial_grad0*(residual_tmp11*(-eta_s*(residual_tmp15*residual_tmp7 - residual_tmp16*residual_tmp93) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp94) + residual_tmp12*residual_tmp97) + trial_grad1*(residual_tmp11*(-eta_s*(-residual_tmp102*residual_tmp16 - residual_tmp103 + residual_tmp46*residual_tmp6 + residual_tmp7*residual_tmp85) + ((s_t(1) / s_t(3)))*residual_tmp101*residual_tmp12) + residual_tmp75*residual_tmp97) + trial_grad2*(residual_tmp11*(-eta_s*(residual_tmp100 - residual_tmp16*residual_tmp99 + residual_tmp7*residual_tmp71) + ((s_t(1) / s_t(3)))*residual_tmp12*residual_tmp98) + residual_tmp43*residual_tmp97);
        const s_t grad_coeff1_1 = trial_grad0*(residual_tmp105*residual_tmp12 + residual_tmp11*(eta_s*(-residual_tmp103 + residual_tmp15*residual_tmp37 + residual_tmp36*residual_tmp93 + residual_tmp46*residual_tmp6) - residual_tmp104*residual_tmp94)) + trial_grad1*(residual_tmp105*residual_tmp75 + residual_tmp11*(eta_s*(residual_tmp102*residual_tmp36 + residual_tmp37*residual_tmp85) - residual_tmp101*residual_tmp104)) + trial_grad2*(residual_tmp105*residual_tmp43 + residual_tmp11*(eta_s*(-residual_tmp106 + residual_tmp36*residual_tmp99 + residual_tmp37*residual_tmp71 + residual_tmp39*residual_tmp95) - residual_tmp104*residual_tmp98));
        const s_t grad_coeff1_2 = trial_grad0*(residual_tmp107*residual_tmp12 + residual_tmp11*(-eta_s*(-residual_tmp100 - residual_tmp15*residual_tmp38 + residual_tmp40*residual_tmp93) + ((s_t(1) / s_t(3)))*residual_tmp43*residual_tmp94)) + trial_grad1*(residual_tmp107*residual_tmp75 + residual_tmp11*(-eta_s*(residual_tmp102*residual_tmp40 - residual_tmp106 - residual_tmp38*residual_tmp85 + residual_tmp39*residual_tmp95) + ((s_t(1) / s_t(3)))*residual_tmp101*residual_tmp43)) + trial_grad2*(residual_tmp107*residual_tmp43 + residual_tmp11*(-eta_s*(-residual_tmp38*residual_tmp71 + residual_tmp40*residual_tmp99) + ((s_t(1) / s_t(3)))*residual_tmp43*residual_tmp98));
        const s_t grad_coeff2_0 = trial_grad0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp93 + residual_tmp21*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp108*residual_tmp16) + residual_tmp110*residual_tmp12) + trial_grad1*(residual_tmp11*(-eta_s*(-residual_tmp102*residual_tmp12 + residual_tmp115 + residual_tmp7*residual_tmp86) + ((s_t(1) / s_t(3)))*residual_tmp114*residual_tmp16 + residual_tmp116) + residual_tmp110*residual_tmp75) + trial_grad2*(residual_tmp11*(-eta_s*(-residual_tmp112 - residual_tmp12*residual_tmp99 + residual_tmp7*residual_tmp72) + ((s_t(1) / s_t(3)))*residual_tmp111*residual_tmp16 - residual_tmp113*u0_grad_1) + residual_tmp110*residual_tmp43);
        const s_t grad_coeff2_1 = trial_grad0*(residual_tmp11*(-eta_s*(-residual_tmp115 - residual_tmp21*residual_tmp37 + residual_tmp44*residual_tmp93) + ((s_t(1) / s_t(3)))*residual_tmp108*residual_tmp36 - residual_tmp116) + residual_tmp117*residual_tmp12) + trial_grad1*(residual_tmp11*(-eta_s*(residual_tmp102*residual_tmp44 - residual_tmp37*residual_tmp86) + ((s_t(1) / s_t(3)))*residual_tmp114*residual_tmp36) + residual_tmp117*residual_tmp75) + trial_grad2*(residual_tmp11*(-eta_s*(residual_tmp118 - residual_tmp37*residual_tmp72 + residual_tmp44*residual_tmp99) + ((s_t(1) / s_t(3)))*residual_tmp111*residual_tmp36 + residual_tmp113*residual_tmp39) + residual_tmp117*residual_tmp43);
        const s_t grad_coeff2_2 = trial_grad0*(residual_tmp11*(eta_s*(-residual_tmp112 + residual_tmp21*residual_tmp38 + residual_tmp43*residual_tmp93) - residual_tmp108*residual_tmp119 - residual_tmp120*u0_grad_1) + residual_tmp12*residual_tmp121) + trial_grad1*(residual_tmp11*(eta_s*(residual_tmp102*residual_tmp43 + residual_tmp118 + residual_tmp38*residual_tmp86) - residual_tmp114*residual_tmp119 + residual_tmp120*residual_tmp39) + residual_tmp121*residual_tmp75) + trial_grad2*(residual_tmp11*(eta_s*(residual_tmp38*residual_tmp72 + residual_tmp43*residual_tmp99) - residual_tmp111*residual_tmp119) + residual_tmp121*residual_tmp43);
        grad_coeff0_0_values[0] = grad_coeff0_0;
        grad_coeff0_1_values[0] = grad_coeff0_1;
        grad_coeff0_2_values[0] = grad_coeff0_2;
        grad_coeff1_0_values[0] = grad_coeff1_0;
        grad_coeff1_1_values[0] = grad_coeff1_1;
        grad_coeff1_2_values[0] = grad_coeff1_2;
        grad_coeff2_0_values[0] = grad_coeff2_0;
        grad_coeff2_1_values[0] = grad_coeff2_1;
        grad_coeff2_2_values[0] = grad_coeff2_2;
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
          const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj3 + grad_ref_z[q * NS + test] * adj6) / det;
          const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj4 + grad_ref_z[q * NS + test] * adj7) / det;
          const s_t test_grad2 = (grad_ref_x[q * NS + test] * adj2 + grad_ref_y[q * NS + test] * adj5 + grad_ref_z[q * NS + test] * adj8) / det;
          element_matrix[(0 * NS + test) * 3 * NS + 1 * NS + trial] += q_weight[q] * det * (grad_coeff0_0_values[0] * test_grad0 + grad_coeff0_1_values[0] * test_grad1 + grad_coeff0_2_values[0] * test_grad2);
          element_matrix[(1 * NS + test) * 3 * NS + 1 * NS + trial] += q_weight[q] * det * (grad_coeff1_0_values[0] * test_grad0 + grad_coeff1_1_values[0] * test_grad1 + grad_coeff1_2_values[0] * test_grad2);
          element_matrix[(2 * NS + test) * 3 * NS + 1 * NS + trial] += q_weight[q] * det * (grad_coeff2_0_values[0] * test_grad0 + grad_coeff2_1_values[0] * test_grad1 + grad_coeff2_2_values[0] * test_grad2);
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
        const s_t u0_grad_0_ref = u0_grad_0_ref_values[0];
        const s_t u0_grad_1_ref = u0_grad_1_ref_values[0];
        const s_t u0_grad_2_ref = u0_grad_2_ref_values[0];
        const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
        const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
        const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
        const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[0];
        const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[0];
        const s_t u0_old_grad_2_ref = u0_old_grad_2_ref_values[0];
        const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
        const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
        const s_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
        const s_t u1_grad_0_ref = u1_grad_0_ref_values[0];
        const s_t u1_grad_1_ref = u1_grad_1_ref_values[0];
        const s_t u1_grad_2_ref = u1_grad_2_ref_values[0];
        const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
        const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
        const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
        const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[0];
        const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[0];
        const s_t u1_old_grad_2_ref = u1_old_grad_2_ref_values[0];
        const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
        const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
        const s_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
        const s_t u2_grad_0_ref = u2_grad_0_ref_values[0];
        const s_t u2_grad_1_ref = u2_grad_1_ref_values[0];
        const s_t u2_grad_2_ref = u2_grad_2_ref_values[0];
        const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
        const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
        const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
        const s_t u2_old_grad_0_ref = u2_old_grad_0_ref_values[0];
        const s_t u2_old_grad_1_ref = u2_old_grad_1_ref_values[0];
        const s_t u2_old_grad_2_ref = u2_old_grad_2_ref_values[0];
        const s_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
        const s_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
        const s_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
        const s_t trial_grad0 = (grad_ref_x[q * NS + trial] * adj0 + grad_ref_y[q * NS + trial] * adj3 + grad_ref_z[q * NS + trial] * adj6) / det;
        const s_t trial_grad1 = (grad_ref_x[q * NS + trial] * adj1 + grad_ref_y[q * NS + trial] * adj4 + grad_ref_z[q * NS + trial] * adj7) / det;
        const s_t trial_grad2 = (grad_ref_x[q * NS + trial] * adj2 + grad_ref_y[q * NS + trial] * adj5 + grad_ref_z[q * NS + trial] * adj8) / det;
        const s_t residual_tmp0 = u0_grad_0*u1_grad_1;
        const s_t residual_tmp1 = u0_grad_1*u1_grad_2;
        const s_t residual_tmp2 = u0_grad_2*u2_grad_1;
        const s_t residual_tmp3 = u1_grad_2*u2_grad_1;
        const s_t residual_tmp4 = u0_grad_1*u1_grad_0;
        const s_t residual_tmp5 = u0_grad_2*u2_grad_0;
        const s_t residual_tmp6 = u1_grad_1 + s_t(1);
        const s_t residual_tmp7 = -residual_tmp3 + residual_tmp6 + u1_grad_1*u2_grad_2 + u2_grad_2;
        const s_t residual_tmp8 = -residual_tmp5 + u0_grad_0*u2_grad_2;
        const s_t residual_tmp9 = residual_tmp0 - residual_tmp4 + u0_grad_0;
        const s_t residual_tmp10 = residual_tmp0*u2_grad_2 + residual_tmp1*u2_grad_0 + residual_tmp2*u1_grad_0 - residual_tmp3*u0_grad_0 - residual_tmp4*u2_grad_2 - residual_tmp5*u1_grad_1 + residual_tmp7 + residual_tmp8 + residual_tmp9;
        const s_t residual_tmp11 = pow_m1(residual_tmp10);
        const s_t residual_tmp12 = -residual_tmp1 + u0_grad_2*u1_grad_1 + u0_grad_2;
        const s_t residual_tmp13 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
        const s_t residual_tmp14 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
        const s_t residual_tmp15 = -newmark_velocity_alpha*residual_tmp7 - residual_tmp13*u1_grad_2 + residual_tmp14*residual_tmp6;
        const s_t residual_tmp16 = -residual_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
        const s_t residual_tmp17 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
        const s_t residual_tmp18 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
        const s_t residual_tmp19 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
        const s_t residual_tmp20 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
        const s_t residual_tmp21 = -residual_tmp17*u0_grad_1 - residual_tmp18*u1_grad_2 + residual_tmp19*u0_grad_2 + residual_tmp20*residual_tmp6;
        const s_t residual_tmp22 = newmark_velocity_alpha*residual_tmp12;
        const s_t residual_tmp23 = residual_tmp18*u0_grad_2;
        const s_t residual_tmp24 = residual_tmp20*u0_grad_1;
        const s_t residual_tmp25 = residual_tmp22 + residual_tmp23 - residual_tmp24;
        const s_t residual_tmp26 = residual_tmp17*residual_tmp6;
        const s_t residual_tmp27 = residual_tmp19*u1_grad_2;
        const s_t residual_tmp28 = residual_tmp26 - residual_tmp27;
        const s_t residual_tmp29 = s_t(3)*eta_b;
        const s_t residual_tmp30 = residual_tmp29*(residual_tmp25 + residual_tmp28);
        const s_t residual_tmp31 = s_t(2)*eta_s;
        const s_t residual_tmp32 = residual_tmp30 + residual_tmp31*(s_t(2)*residual_tmp17*residual_tmp6 - residual_tmp25 - s_t(2)*residual_tmp27);
        const s_t residual_tmp33 = ((s_t(1) / s_t(3)))*residual_tmp7;
        const s_t residual_tmp34 = pow_m2(residual_tmp10);
        const s_t residual_tmp35 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
        const s_t residual_tmp36 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
        const s_t residual_tmp37 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
        const s_t residual_tmp38 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
        const s_t residual_tmp39 = residual_tmp6 + residual_tmp9;
        const s_t residual_tmp40 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
        const s_t residual_tmp41 = residual_tmp12*residual_tmp35 + residual_tmp13*residual_tmp37 + residual_tmp14*residual_tmp38 - residual_tmp17*residual_tmp39 + residual_tmp19*residual_tmp36 - residual_tmp40*residual_tmp7;
        const s_t residual_tmp42 = u0_grad_0*u2_grad_1 - u0_grad_1*u2_grad_0 + u2_grad_1;
        const s_t residual_tmp43 = u0_grad_0 + s_t(1);
        const s_t residual_tmp44 = residual_tmp43 + residual_tmp8 + u2_grad_2;
        const s_t residual_tmp45 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
        const s_t residual_tmp46 = residual_tmp16*residual_tmp35 + residual_tmp17*residual_tmp42 + residual_tmp18*residual_tmp37 - residual_tmp19*residual_tmp44 + residual_tmp20*residual_tmp38 - residual_tmp45*residual_tmp7;
        const s_t residual_tmp47 = residual_tmp16*residual_tmp45;
        const s_t residual_tmp48 = residual_tmp20*residual_tmp42;
        const s_t residual_tmp49 = residual_tmp12*residual_tmp40;
        const s_t residual_tmp50 = residual_tmp13*residual_tmp36;
        const s_t residual_tmp51 = residual_tmp18*residual_tmp44;
        const s_t residual_tmp52 = -residual_tmp51;
        const s_t residual_tmp53 = residual_tmp14*residual_tmp39;
        const s_t residual_tmp54 = -residual_tmp53;
        const s_t residual_tmp55 = residual_tmp47 + residual_tmp48 + residual_tmp49 + residual_tmp50 + residual_tmp52 + residual_tmp54;
        const s_t residual_tmp56 = residual_tmp35*residual_tmp7;
        const s_t residual_tmp57 = residual_tmp17*residual_tmp38 + residual_tmp19*residual_tmp37 - residual_tmp56;
        const s_t residual_tmp58 = residual_tmp29*(residual_tmp55 + residual_tmp57);
        const s_t residual_tmp59 = residual_tmp31*(s_t(2)*residual_tmp17*residual_tmp38 + s_t(2)*residual_tmp19*residual_tmp37 - residual_tmp55 - s_t(2)*residual_tmp56) + residual_tmp58;
        const s_t residual_tmp60 = -residual_tmp59;
        const s_t residual_tmp61 = residual_tmp34*(eta_s*(residual_tmp12*residual_tmp41 + residual_tmp16*residual_tmp46) + residual_tmp33*residual_tmp60);
        const s_t residual_tmp62 = newmark_velocity_alpha*residual_tmp36;
        const s_t residual_tmp63 = residual_tmp20*residual_tmp43;
        const s_t residual_tmp64 = residual_tmp45*u0_grad_2;
        const s_t residual_tmp65 = residual_tmp62 + residual_tmp63 - residual_tmp64;
        const s_t residual_tmp66 = residual_tmp35*u1_grad_2;
        const s_t residual_tmp67 = residual_tmp17*u1_grad_0;
        const s_t residual_tmp68 = residual_tmp66 - residual_tmp67;
        const s_t residual_tmp69 = residual_tmp29*(residual_tmp65 + residual_tmp68);
        const s_t residual_tmp70 = residual_tmp31*(s_t(2)*residual_tmp35*u1_grad_2 - residual_tmp65 - s_t(2)*residual_tmp67) + residual_tmp69;
        const s_t residual_tmp71 = newmark_velocity_alpha*residual_tmp37 - residual_tmp14*u1_grad_0 + residual_tmp40*u1_grad_2;
        const s_t residual_tmp72 = residual_tmp17*residual_tmp43 - residual_tmp20*u1_grad_0 - residual_tmp35*u0_grad_2 + residual_tmp45*u1_grad_2;
        const s_t residual_tmp73 = residual_tmp46*u0_grad_2;
        const s_t residual_tmp74 = ((s_t(1) / s_t(3)))*residual_tmp60;
        const s_t residual_tmp75 = -residual_tmp39;
        const s_t residual_tmp76 = newmark_velocity_alpha*residual_tmp39;
        const s_t residual_tmp77 = residual_tmp18*residual_tmp43;
        const s_t residual_tmp78 = residual_tmp45*u0_grad_1;
        const s_t residual_tmp79 = residual_tmp76 + residual_tmp77 - residual_tmp78;
        const s_t residual_tmp80 = residual_tmp35*residual_tmp6;
        const s_t residual_tmp81 = residual_tmp19*u1_grad_0;
        const s_t residual_tmp82 = residual_tmp80 - residual_tmp81;
        const s_t residual_tmp83 = residual_tmp29*(-residual_tmp79 - residual_tmp82);
        const s_t residual_tmp84 = residual_tmp31*(residual_tmp79 - s_t(2)*residual_tmp80 + s_t(2)*residual_tmp81) + residual_tmp83;
        const s_t residual_tmp85 = newmark_velocity_alpha*residual_tmp38 + residual_tmp13*u1_grad_0 - residual_tmp40*residual_tmp6;
        const s_t residual_tmp86 = residual_tmp18*u1_grad_0 - residual_tmp19*residual_tmp43 + residual_tmp35*u0_grad_1 - residual_tmp45*residual_tmp6;
        const s_t residual_tmp87 = residual_tmp46*u0_grad_1;
        const s_t residual_tmp88 = residual_tmp34*(-eta_s*(-residual_tmp36*residual_tmp41 + residual_tmp44*residual_tmp46) + ((s_t(1) / s_t(3)))*residual_tmp37*residual_tmp59);
        const s_t residual_tmp89 = ((s_t(1) / s_t(3)))*residual_tmp59;
        const s_t residual_tmp90 = residual_tmp43*residual_tmp46;
        const s_t residual_tmp91 = residual_tmp89*u1_grad_0;
        const s_t residual_tmp92 = residual_tmp34*(-eta_s*(residual_tmp39*residual_tmp41 - residual_tmp42*residual_tmp46) + ((s_t(1) / s_t(3)))*residual_tmp38*residual_tmp59);
        const s_t residual_tmp93 = newmark_velocity_alpha*residual_tmp16 + residual_tmp13*u0_grad_2 - residual_tmp14*u0_grad_1;
        const s_t residual_tmp94 = residual_tmp30 + residual_tmp31*(s_t(2)*residual_tmp18*u0_grad_2 - residual_tmp22 - s_t(2)*residual_tmp24 - residual_tmp28);
        const s_t residual_tmp95 = residual_tmp12*residual_tmp45 - residual_tmp13*residual_tmp44 + residual_tmp14*residual_tmp42 + residual_tmp16*residual_tmp40 + residual_tmp18*residual_tmp36 - residual_tmp20*residual_tmp39;
        const s_t residual_tmp96 = residual_tmp31*(s_t(2)*residual_tmp16*residual_tmp45 + s_t(2)*residual_tmp20*residual_tmp42 - residual_tmp49 - residual_tmp50 - s_t(2)*residual_tmp51 - residual_tmp54 - residual_tmp57) + residual_tmp58;
        const s_t residual_tmp97 = residual_tmp34*(-eta_s*(-residual_tmp12*residual_tmp95 + residual_tmp46*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp16*residual_tmp96);
        const s_t residual_tmp98 = residual_tmp31*(s_t(2)*residual_tmp20*residual_tmp43 - residual_tmp62 - s_t(2)*residual_tmp64 - residual_tmp68) + residual_tmp69;
        const s_t residual_tmp99 = -newmark_velocity_alpha*residual_tmp44 + residual_tmp14*residual_tmp43 - residual_tmp40*u0_grad_2;
        const s_t residual_tmp100 = residual_tmp46*u1_grad_2;
        const s_t residual_tmp101 = ((s_t(1) / s_t(3)))*residual_tmp96;
        const s_t residual_tmp102 = residual_tmp31*(residual_tmp76 - s_t(2)*residual_tmp77 + s_t(2)*residual_tmp78 + residual_tmp82) + residual_tmp83;
        const s_t residual_tmp103 = newmark_velocity_alpha*residual_tmp42 - residual_tmp13*residual_tmp43 + residual_tmp40*u0_grad_1;
        const s_t residual_tmp104 = residual_tmp46*residual_tmp6;
        const s_t residual_tmp105 = residual_tmp101*u0_grad_1;
        const s_t residual_tmp106 = ((s_t(1) / s_t(3)))*residual_tmp44;
        const s_t residual_tmp107 = -(s_t(1) / s_t(3))*residual_tmp96;
        const s_t residual_tmp108 = residual_tmp34*(eta_s*(residual_tmp36*residual_tmp95 + residual_tmp37*residual_tmp46) + residual_tmp107*residual_tmp44);
        const s_t residual_tmp109 = residual_tmp46*u1_grad_0;
        const s_t residual_tmp110 = residual_tmp34*(-eta_s*(-residual_tmp38*residual_tmp46 + residual_tmp39*residual_tmp95) + ((s_t(1) / s_t(3)))*residual_tmp42*residual_tmp96);
        const s_t residual_tmp111 = residual_tmp30 + residual_tmp31*(s_t(2)*residual_tmp22 - residual_tmp23 + residual_tmp24 - residual_tmp26 + residual_tmp27);
        const s_t residual_tmp112 = residual_tmp31*(s_t(2)*residual_tmp12*residual_tmp40 + s_t(2)*residual_tmp13*residual_tmp36 - residual_tmp47 - residual_tmp48 - residual_tmp52 - s_t(2)*residual_tmp53 - residual_tmp57) + residual_tmp58;
        const s_t residual_tmp113 = residual_tmp34*(-eta_s*(-residual_tmp16*residual_tmp95 + residual_tmp41*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp112*residual_tmp12);
        const s_t residual_tmp114 = residual_tmp31*(s_t(2)*residual_tmp62 - residual_tmp63 + residual_tmp64 - residual_tmp66 + residual_tmp67) + residual_tmp69;
        const s_t residual_tmp115 = -residual_tmp41*u1_grad_2 + residual_tmp95*u0_grad_2;
        const s_t residual_tmp116 = residual_tmp31*(-s_t(2)*residual_tmp76 + residual_tmp77 - residual_tmp78 + residual_tmp80 - residual_tmp81) + residual_tmp83;
        const s_t residual_tmp117 = residual_tmp95*u0_grad_1;
        const s_t residual_tmp118 = residual_tmp34*(-eta_s*(-residual_tmp37*residual_tmp41 + residual_tmp44*residual_tmp95) + ((s_t(1) / s_t(3)))*residual_tmp112*residual_tmp36);
        const s_t residual_tmp119 = residual_tmp41*u1_grad_0;
        const s_t residual_tmp120 = ((s_t(1) / s_t(3)))*residual_tmp39;
        const s_t residual_tmp121 = residual_tmp34*(eta_s*(residual_tmp38*residual_tmp41 + residual_tmp42*residual_tmp95) - residual_tmp112*residual_tmp120);
        const s_t grad_coeff0_0 = trial_grad0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp15 + residual_tmp16*residual_tmp21) - residual_tmp32*residual_tmp33) + residual_tmp12*residual_tmp61) + trial_grad1*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp71 + residual_tmp16*residual_tmp72 - residual_tmp73) - residual_tmp33*residual_tmp70 - residual_tmp74*u1_grad_2) + residual_tmp36*residual_tmp61) + trial_grad2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp85 + residual_tmp16*residual_tmp86 + residual_tmp87) - residual_tmp33*residual_tmp84 + residual_tmp6*residual_tmp74) + residual_tmp61*residual_tmp75);
        const s_t grad_coeff0_1 = trial_grad0*(residual_tmp11*(-eta_s*(-residual_tmp15*residual_tmp36 + residual_tmp21*residual_tmp44 - residual_tmp73) + ((s_t(1) / s_t(3)))*residual_tmp32*residual_tmp37 - residual_tmp89*u1_grad_2) + residual_tmp12*residual_tmp88) + trial_grad1*(residual_tmp11*(-eta_s*(-residual_tmp36*residual_tmp71 + residual_tmp44*residual_tmp72) + ((s_t(1) / s_t(3)))*residual_tmp37*residual_tmp70) + residual_tmp36*residual_tmp88) + trial_grad2*(residual_tmp11*(-eta_s*(-residual_tmp36*residual_tmp85 + residual_tmp44*residual_tmp86 + residual_tmp90) + ((s_t(1) / s_t(3)))*residual_tmp37*residual_tmp84 + residual_tmp91) + residual_tmp75*residual_tmp88);
        const s_t grad_coeff0_2 = trial_grad0*(residual_tmp11*(-eta_s*(residual_tmp15*residual_tmp39 - residual_tmp21*residual_tmp42 + residual_tmp87) + ((s_t(1) / s_t(3)))*residual_tmp32*residual_tmp38 + residual_tmp6*residual_tmp89) + residual_tmp12*residual_tmp92) + trial_grad1*(residual_tmp11*(-eta_s*(residual_tmp39*residual_tmp71 - residual_tmp42*residual_tmp72 - residual_tmp90) + ((s_t(1) / s_t(3)))*residual_tmp38*residual_tmp70 - residual_tmp91) + residual_tmp36*residual_tmp92) + trial_grad2*(residual_tmp11*(-eta_s*(residual_tmp39*residual_tmp85 - residual_tmp42*residual_tmp86) + ((s_t(1) / s_t(3)))*residual_tmp38*residual_tmp84) + residual_tmp75*residual_tmp92);
        const s_t grad_coeff1_0 = trial_grad0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp93 + residual_tmp21*residual_tmp7) + ((s_t(1) / s_t(3)))*residual_tmp16*residual_tmp94) + residual_tmp12*residual_tmp97) + trial_grad1*(residual_tmp11*(-eta_s*(-residual_tmp100 - residual_tmp12*residual_tmp99 + residual_tmp7*residual_tmp72) - residual_tmp101*u0_grad_2 + ((s_t(1) / s_t(3)))*residual_tmp16*residual_tmp98) + residual_tmp36*residual_tmp97) + trial_grad2*(residual_tmp11*(-eta_s*(-residual_tmp103*residual_tmp12 + residual_tmp104 + residual_tmp7*residual_tmp86) + ((s_t(1) / s_t(3)))*residual_tmp102*residual_tmp16 + residual_tmp105) + residual_tmp75*residual_tmp97);
        const s_t grad_coeff1_1 = trial_grad0*(residual_tmp108*residual_tmp12 + residual_tmp11*(eta_s*(-residual_tmp100 + residual_tmp21*residual_tmp37 + residual_tmp36*residual_tmp93) - residual_tmp106*residual_tmp94 - residual_tmp107*u0_grad_2)) + trial_grad1*(residual_tmp108*residual_tmp36 + residual_tmp11*(eta_s*(residual_tmp36*residual_tmp99 + residual_tmp37*residual_tmp72) - residual_tmp106*residual_tmp98)) + trial_grad2*(residual_tmp108*residual_tmp75 + residual_tmp11*(eta_s*(residual_tmp103*residual_tmp36 + residual_tmp109 + residual_tmp37*residual_tmp86) - residual_tmp102*residual_tmp106 + residual_tmp107*residual_tmp43));
        const s_t grad_coeff1_2 = trial_grad0*(residual_tmp11*(-eta_s*(-residual_tmp104 - residual_tmp21*residual_tmp38 + residual_tmp39*residual_tmp93) - residual_tmp105 + ((s_t(1) / s_t(3)))*residual_tmp42*residual_tmp94) + residual_tmp110*residual_tmp12) + trial_grad1*(residual_tmp11*(-eta_s*(residual_tmp109 - residual_tmp38*residual_tmp72 + residual_tmp39*residual_tmp99) + residual_tmp101*residual_tmp43 + ((s_t(1) / s_t(3)))*residual_tmp42*residual_tmp98) + residual_tmp110*residual_tmp36) + trial_grad2*(residual_tmp11*(-eta_s*(residual_tmp103*residual_tmp39 - residual_tmp38*residual_tmp86) + ((s_t(1) / s_t(3)))*residual_tmp102*residual_tmp42) + residual_tmp110*residual_tmp75);
        const s_t grad_coeff2_0 = trial_grad0*(residual_tmp11*(-eta_s*(residual_tmp15*residual_tmp7 - residual_tmp16*residual_tmp93) + ((s_t(1) / s_t(3)))*residual_tmp111*residual_tmp12) + residual_tmp113*residual_tmp12) + trial_grad1*(residual_tmp11*(-eta_s*(residual_tmp115 - residual_tmp16*residual_tmp99 + residual_tmp7*residual_tmp71) + ((s_t(1) / s_t(3)))*residual_tmp114*residual_tmp12) + residual_tmp113*residual_tmp36) + trial_grad2*(residual_tmp11*(-eta_s*(-residual_tmp103*residual_tmp16 - residual_tmp117 + residual_tmp41*residual_tmp6 + residual_tmp7*residual_tmp85) + ((s_t(1) / s_t(3)))*residual_tmp116*residual_tmp12) + residual_tmp113*residual_tmp75);
        const s_t grad_coeff2_1 = trial_grad0*(residual_tmp11*(-eta_s*(-residual_tmp115 - residual_tmp15*residual_tmp37 + residual_tmp44*residual_tmp93) + ((s_t(1) / s_t(3)))*residual_tmp111*residual_tmp36) + residual_tmp118*residual_tmp12) + trial_grad1*(residual_tmp11*(-eta_s*(-residual_tmp37*residual_tmp71 + residual_tmp44*residual_tmp99) + ((s_t(1) / s_t(3)))*residual_tmp114*residual_tmp36) + residual_tmp118*residual_tmp36) + trial_grad2*(residual_tmp11*(-eta_s*(residual_tmp103*residual_tmp44 - residual_tmp119 - residual_tmp37*residual_tmp85 + residual_tmp43*residual_tmp95) + ((s_t(1) / s_t(3)))*residual_tmp116*residual_tmp36) + residual_tmp118*residual_tmp75);
        const s_t grad_coeff2_2 = trial_grad0*(residual_tmp11*(eta_s*(-residual_tmp117 + residual_tmp15*residual_tmp38 + residual_tmp41*residual_tmp6 + residual_tmp42*residual_tmp93) - residual_tmp111*residual_tmp120) + residual_tmp12*residual_tmp121) + trial_grad1*(residual_tmp11*(eta_s*(-residual_tmp119 + residual_tmp38*residual_tmp71 + residual_tmp42*residual_tmp99 + residual_tmp43*residual_tmp95) - residual_tmp114*residual_tmp120) + residual_tmp121*residual_tmp36) + trial_grad2*(residual_tmp11*(eta_s*(residual_tmp103*residual_tmp42 + residual_tmp38*residual_tmp85) - residual_tmp116*residual_tmp120) + residual_tmp121*residual_tmp75);
        grad_coeff0_0_values[0] = grad_coeff0_0;
        grad_coeff0_1_values[0] = grad_coeff0_1;
        grad_coeff0_2_values[0] = grad_coeff0_2;
        grad_coeff1_0_values[0] = grad_coeff1_0;
        grad_coeff1_1_values[0] = grad_coeff1_1;
        grad_coeff1_2_values[0] = grad_coeff1_2;
        grad_coeff2_0_values[0] = grad_coeff2_0;
        grad_coeff2_1_values[0] = grad_coeff2_1;
        grad_coeff2_2_values[0] = grad_coeff2_2;
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
          const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj3 + grad_ref_z[q * NS + test] * adj6) / det;
          const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj4 + grad_ref_z[q * NS + test] * adj7) / det;
          const s_t test_grad2 = (grad_ref_x[q * NS + test] * adj2 + grad_ref_y[q * NS + test] * adj5 + grad_ref_z[q * NS + test] * adj8) / det;
          element_matrix[(0 * NS + test) * 3 * NS + 2 * NS + trial] += q_weight[q] * det * (grad_coeff0_0_values[0] * test_grad0 + grad_coeff0_1_values[0] * test_grad1 + grad_coeff0_2_values[0] * test_grad2);
          element_matrix[(1 * NS + test) * 3 * NS + 2 * NS + trial] += q_weight[q] * det * (grad_coeff1_0_values[0] * test_grad0 + grad_coeff1_1_values[0] * test_grad1 + grad_coeff1_2_values[0] * test_grad2);
          element_matrix[(2 * NS + test) * 3 * NS + 2 * NS + trial] += q_weight[q] * det * (grad_coeff2_0_values[0] * test_grad0 + grad_coeff2_1_values[0] * test_grad1 + grad_coeff2_2_values[0] * test_grad2);
        }
      }
    }
  }
}

} // namespace codegen
} // namespace sfem

#endif
