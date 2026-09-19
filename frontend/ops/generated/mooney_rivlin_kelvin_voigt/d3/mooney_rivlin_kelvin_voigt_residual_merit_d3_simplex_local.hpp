#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_RESIDUAL_MERIT_D3_SIMPLEX_LOCAL_HPP
#define MOONEY_RIVLIN_KELVIN_VOIGT_RESIDUAL_MERIT_D3_SIMPLEX_LOCAL_HPP

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
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_residual_merit_d3_simplex_residual_block(
    const int ne,
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
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u0_grad_0_ref_values[lane] = s_t(0);
      u0_grad_1_ref_values[lane] = s_t(0);
      u0_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC][lane];
        u0_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u0_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u0_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u0_old_grad_0_ref_values[lane] = s_t(0);
      u0_old_grad_1_ref_values[lane] = s_t(0);
      u0_old_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC][lane];
        u0_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u0_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u0_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u1_grad_0_ref_values[lane] = s_t(0);
      u1_grad_1_ref_values[lane] = s_t(0);
      u1_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC + 1][lane];
        u1_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u1_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u1_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u1_old_grad_0_ref_values[lane] = s_t(0);
      u1_old_grad_1_ref_values[lane] = s_t(0);
      u1_old_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC + 1][lane];
        u1_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u1_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u1_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u2_grad_0_ref_values[lane] = s_t(0);
      u2_grad_1_ref_values[lane] = s_t(0);
      u2_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC + 2][lane];
        u2_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u2_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u2_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u2_old_grad_0_ref_values[lane] = s_t(0);
      u2_old_grad_1_ref_values[lane] = s_t(0);
      u2_old_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC + 2][lane];
        u2_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u2_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u2_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
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
      const s_t u0_grad_0_ref = u0_grad_0_ref_values[lane];
      const s_t u0_grad_1_ref = u0_grad_1_ref_values[lane];
      const s_t u0_grad_2_ref = u0_grad_2_ref_values[lane];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
      const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
      const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[lane];
      const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[lane];
      const s_t u0_old_grad_2_ref = u0_old_grad_2_ref_values[lane];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
      const s_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
      const s_t u1_grad_0_ref = u1_grad_0_ref_values[lane];
      const s_t u1_grad_1_ref = u1_grad_1_ref_values[lane];
      const s_t u1_grad_2_ref = u1_grad_2_ref_values[lane];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
      const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
      const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[lane];
      const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[lane];
      const s_t u1_old_grad_2_ref = u1_old_grad_2_ref_values[lane];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
      const s_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
      const s_t u2_grad_0_ref = u2_grad_0_ref_values[lane];
      const s_t u2_grad_1_ref = u2_grad_1_ref_values[lane];
      const s_t u2_grad_2_ref = u2_grad_2_ref_values[lane];
      const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
      const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
      const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
      const s_t u2_old_grad_0_ref = u2_old_grad_0_ref_values[lane];
      const s_t u2_old_grad_1_ref = u2_old_grad_1_ref_values[lane];
      const s_t u2_old_grad_2_ref = u2_old_grad_2_ref_values[lane];
      const s_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
      const s_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
      const s_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = u1_grad_2*u2_grad_1;
      const s_t residual_tmp1 = u1_grad_1 + s_t(1);
      const s_t residual_tmp2 = u2_grad_2 + s_t(1);
      const s_t residual_tmp3 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp4 = u0_grad_2*u2_grad_0;
      const s_t residual_tmp5 = u0_grad_0 + s_t(1);
      const s_t residual_tmp6 = ((s_t(1) / s_t(2)))*lmbda*(-residual_tmp0*residual_tmp5 + residual_tmp1*residual_tmp2*residual_tmp5 - residual_tmp1*residual_tmp4 - residual_tmp2*residual_tmp3 + u0_grad_1*u1_grad_2*u2_grad_0 + u0_grad_2*u1_grad_0*u2_grad_1 + s_t(-1));
      const s_t residual_tmp7 = residual_tmp1*u1_grad_0 + residual_tmp5*u0_grad_1 + u2_grad_0*u2_grad_1;
      const s_t residual_tmp8 = s_t(2)*u0_grad_1;
      const s_t residual_tmp9 = residual_tmp2*u2_grad_0 + residual_tmp5*u0_grad_2 + u1_grad_0*u1_grad_2;
      const s_t residual_tmp10 = s_t(2)*u0_grad_2;
      const s_t residual_tmp11 = pow_2(residual_tmp5) + pow_2(u1_grad_0) + pow_2(u2_grad_0);
      const s_t residual_tmp12 = s_t(2)*residual_tmp5;
      const s_t residual_tmp13 = pow_2(residual_tmp1) + pow_2(u0_grad_1) + pow_2(u2_grad_1);
      const s_t residual_tmp14 = pow_2(residual_tmp2) + pow_2(u0_grad_2) + pow_2(u1_grad_2);
      const s_t residual_tmp15 = residual_tmp11 + residual_tmp13 + residual_tmp14;
      const s_t residual_tmp16 = u0_grad_0*u1_grad_1;
      const s_t residual_tmp17 = u0_grad_1*u2_grad_0;
      const s_t residual_tmp18 = u0_grad_2*u2_grad_1;
      const s_t residual_tmp19 = -residual_tmp4 + u0_grad_0*u2_grad_2 + u0_grad_0;
      const s_t residual_tmp20 = -residual_tmp0 + residual_tmp1 + u1_grad_1*u2_grad_2 + u2_grad_2;
      const s_t residual_tmp21 = residual_tmp16 - residual_tmp3;
      const s_t residual_tmp22 = pow_m1(-residual_tmp0*u0_grad_0 + residual_tmp16*u2_grad_2 + residual_tmp17*u1_grad_2 + residual_tmp18*u1_grad_0 + residual_tmp19 + residual_tmp20 + residual_tmp21 - residual_tmp3*u2_grad_2 - residual_tmp4*u1_grad_1);
      const s_t residual_tmp23 = u0_grad_1*u1_grad_2;
      const s_t residual_tmp24 = -residual_tmp23 + u0_grad_2*u1_grad_1 + u0_grad_2;
      const s_t residual_tmp25 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp26 = u0_grad_1*u_dt_shift + u0_old_grad_1;
      const s_t residual_tmp27 = u0_grad_2*u1_grad_0;
      const s_t residual_tmp28 = -residual_tmp27 + u0_grad_0*u1_grad_2 + u1_grad_2;
      const s_t residual_tmp29 = u2_grad_1*u_dt_shift + u2_old_grad_1;
      const s_t residual_tmp30 = u1_grad_2*u2_grad_0;
      const s_t residual_tmp31 = -residual_tmp30 + u1_grad_0*u2_grad_2 + u1_grad_0;
      const s_t residual_tmp32 = u2_grad_2*u_dt_shift + u2_old_grad_2;
      const s_t residual_tmp33 = u1_grad_0*u2_grad_1;
      const s_t residual_tmp34 = -residual_tmp33 + u1_grad_1*u2_grad_0 + u2_grad_0;
      const s_t residual_tmp35 = u0_grad_2*u_dt_shift + u0_old_grad_2;
      const s_t residual_tmp36 = residual_tmp1 + residual_tmp21 + u0_grad_0;
      const s_t residual_tmp37 = u2_grad_0*u_dt_shift + u2_old_grad_0;
      const s_t residual_tmp38 = -residual_tmp20*residual_tmp37 + residual_tmp24*residual_tmp25 + residual_tmp26*residual_tmp28 + residual_tmp29*residual_tmp31 + residual_tmp32*residual_tmp34 - residual_tmp35*residual_tmp36;
      const s_t residual_tmp39 = -residual_tmp18 + u0_grad_1*u2_grad_2 + u0_grad_1;
      const s_t residual_tmp40 = -residual_tmp17 + u0_grad_0*u2_grad_1 + u2_grad_1;
      const s_t residual_tmp41 = u1_grad_1*u_dt_shift + u1_old_grad_1;
      const s_t residual_tmp42 = u1_grad_2*u_dt_shift + u1_old_grad_2;
      const s_t residual_tmp43 = residual_tmp19 + residual_tmp2;
      const s_t residual_tmp44 = u1_grad_0*u_dt_shift + u1_old_grad_0;
      const s_t residual_tmp45 = -residual_tmp20*residual_tmp44 + residual_tmp25*residual_tmp39 - residual_tmp26*residual_tmp43 + residual_tmp31*residual_tmp41 + residual_tmp34*residual_tmp42 + residual_tmp35*residual_tmp40;
      const s_t residual_tmp46 = residual_tmp39*residual_tmp44;
      const s_t residual_tmp47 = residual_tmp40*residual_tmp42;
      const s_t residual_tmp48 = residual_tmp24*residual_tmp37;
      const s_t residual_tmp49 = residual_tmp28*residual_tmp29;
      const s_t residual_tmp50 = residual_tmp41*residual_tmp43;
      const s_t residual_tmp51 = -residual_tmp50;
      const s_t residual_tmp52 = residual_tmp32*residual_tmp36;
      const s_t residual_tmp53 = -residual_tmp52;
      const s_t residual_tmp54 = residual_tmp46 + residual_tmp47 + residual_tmp48 + residual_tmp49 + residual_tmp51 + residual_tmp53;
      const s_t residual_tmp55 = residual_tmp20*residual_tmp25;
      const s_t residual_tmp56 = residual_tmp26*residual_tmp31 + residual_tmp34*residual_tmp35 - residual_tmp55;
      const s_t residual_tmp57 = s_t(3)*eta_b*(residual_tmp54 + residual_tmp56);
      const s_t residual_tmp58 = s_t(2)*eta_s;
      const s_t residual_tmp59 = residual_tmp57 + residual_tmp58*(s_t(2)*residual_tmp26*residual_tmp31 + s_t(2)*residual_tmp34*residual_tmp35 - residual_tmp54 - s_t(2)*residual_tmp55);
      const s_t residual_tmp60 = s_t(2)*u1_grad_0;
      const s_t residual_tmp61 = residual_tmp1*u1_grad_2 + residual_tmp2*u2_grad_1 + u0_grad_1*u0_grad_2;
      const s_t residual_tmp62 = s_t(2)*u2_grad_0;
      const s_t residual_tmp63 = s_t(2)*u1_grad_2;
      const s_t residual_tmp64 = s_t(2)*residual_tmp1;
      const s_t residual_tmp65 = residual_tmp24*residual_tmp44 + residual_tmp28*residual_tmp41 - residual_tmp29*residual_tmp43 + residual_tmp32*residual_tmp40 - residual_tmp36*residual_tmp42 + residual_tmp37*residual_tmp39;
      const s_t residual_tmp66 = residual_tmp57 + residual_tmp58*(s_t(2)*residual_tmp39*residual_tmp44 + s_t(2)*residual_tmp40*residual_tmp42 - residual_tmp48 - residual_tmp49 - s_t(2)*residual_tmp50 - residual_tmp53 - residual_tmp56);
      const s_t residual_tmp67 = s_t(2)*u2_grad_1;
      const s_t residual_tmp68 = s_t(2)*residual_tmp2;
      const s_t residual_tmp69 = residual_tmp57 + residual_tmp58*(s_t(2)*residual_tmp24*residual_tmp37 + s_t(2)*residual_tmp28*residual_tmp29 - residual_tmp46 - residual_tmp47 - residual_tmp51 - s_t(2)*residual_tmp52 - residual_tmp56);
      const s_t grad_coeff0_0 = mu*(s_t(6)*residual_tmp0 - s_t(6)*residual_tmp1*residual_tmp2 - residual_tmp10*residual_tmp9 - residual_tmp11*residual_tmp12 + residual_tmp12*residual_tmp15 - residual_tmp7*residual_tmp8 + s_t(2)*u0_grad_0 + s_t(2)) + residual_tmp22*(eta_s*(residual_tmp24*residual_tmp38 + residual_tmp39*residual_tmp45) - (s_t(1) / s_t(3))*residual_tmp20*residual_tmp59) + residual_tmp6*(-s_t(2)*residual_tmp0 + s_t(2)*residual_tmp1*residual_tmp2);
      const s_t grad_coeff0_1 = mu*(-residual_tmp10*residual_tmp61 - residual_tmp12*residual_tmp7 - residual_tmp13*residual_tmp8 + s_t(2)*residual_tmp15*u0_grad_1 + s_t(6)*residual_tmp2*u1_grad_0 - s_t(6)*residual_tmp30 + s_t(2)*u0_grad_1) + residual_tmp22*(-eta_s*(-residual_tmp28*residual_tmp38 + residual_tmp43*residual_tmp45) + ((s_t(1) / s_t(3)))*residual_tmp31*residual_tmp59) + residual_tmp6*(-residual_tmp2*residual_tmp60 + s_t(2)*u1_grad_2*u2_grad_0);
      const s_t grad_coeff0_2 = mu*(s_t(6)*residual_tmp1*u2_grad_0 - residual_tmp10*residual_tmp14 - residual_tmp12*residual_tmp9 + s_t(2)*residual_tmp15*u0_grad_2 - s_t(6)*residual_tmp33 - residual_tmp61*residual_tmp8 + s_t(2)*u0_grad_2) + residual_tmp22*(-eta_s*(residual_tmp36*residual_tmp38 - residual_tmp40*residual_tmp45) + ((s_t(1) / s_t(3)))*residual_tmp34*residual_tmp59) + residual_tmp6*(-residual_tmp1*residual_tmp62 + s_t(2)*residual_tmp33);
      const s_t grad_coeff1_0 = mu*(-residual_tmp11*residual_tmp60 + s_t(2)*residual_tmp15*u1_grad_0 - s_t(6)*residual_tmp18 + s_t(6)*residual_tmp2*u0_grad_1 - residual_tmp63*residual_tmp9 - residual_tmp64*residual_tmp7 + s_t(2)*u1_grad_0) + residual_tmp22*(-eta_s*(residual_tmp20*residual_tmp45 - residual_tmp24*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp39*residual_tmp66) + residual_tmp6*(-residual_tmp2*residual_tmp8 + s_t(2)*u0_grad_2*u2_grad_1);
      const s_t grad_coeff1_1 = mu*(-residual_tmp13*residual_tmp64 + residual_tmp15*residual_tmp64 - s_t(6)*residual_tmp2*residual_tmp5 + s_t(6)*residual_tmp4 - residual_tmp60*residual_tmp7 - residual_tmp61*residual_tmp63 + s_t(2)*u1_grad_1 + s_t(2)) + residual_tmp22*(eta_s*(residual_tmp28*residual_tmp65 + residual_tmp31*residual_tmp45) - (s_t(1) / s_t(3))*residual_tmp43*residual_tmp66) + residual_tmp6*(s_t(2)*residual_tmp2*residual_tmp5 - s_t(2)*residual_tmp4);
      const s_t grad_coeff1_2 = mu*(-residual_tmp14*residual_tmp63 + s_t(2)*residual_tmp15*u1_grad_2 - s_t(6)*residual_tmp17 + s_t(6)*residual_tmp5*u2_grad_1 - residual_tmp60*residual_tmp9 - residual_tmp61*residual_tmp64 + s_t(2)*u1_grad_2) + residual_tmp22*(-eta_s*(-residual_tmp34*residual_tmp45 + residual_tmp36*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp40*residual_tmp66) + residual_tmp6*(s_t(2)*residual_tmp17 - residual_tmp5*residual_tmp67);
      const s_t grad_coeff2_0 = mu*(s_t(6)*residual_tmp1*u0_grad_2 - residual_tmp11*residual_tmp62 + s_t(2)*residual_tmp15*u2_grad_0 - s_t(6)*residual_tmp23 - residual_tmp67*residual_tmp7 - residual_tmp68*residual_tmp9 + s_t(2)*u2_grad_0) + residual_tmp22*(-eta_s*(residual_tmp20*residual_tmp38 - residual_tmp39*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp24*residual_tmp69) + residual_tmp6*(-residual_tmp1*residual_tmp10 + s_t(2)*residual_tmp23);
      const s_t grad_coeff2_1 = mu*(-residual_tmp13*residual_tmp67 + s_t(2)*residual_tmp15*u2_grad_1 - s_t(6)*residual_tmp27 + s_t(6)*residual_tmp5*u1_grad_2 - residual_tmp61*residual_tmp68 - residual_tmp62*residual_tmp7 + s_t(2)*u2_grad_1) + residual_tmp22*(-eta_s*(-residual_tmp31*residual_tmp38 + residual_tmp43*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp28*residual_tmp69) + residual_tmp6*(s_t(2)*residual_tmp27 - residual_tmp5*residual_tmp63);
      const s_t grad_coeff2_2 = mu*(-s_t(6)*residual_tmp1*residual_tmp5 - residual_tmp14*residual_tmp68 + residual_tmp15*residual_tmp68 + s_t(6)*residual_tmp3 - residual_tmp61*residual_tmp67 - residual_tmp62*residual_tmp9 + s_t(2)*u2_grad_2 + s_t(2)) + residual_tmp22*(eta_s*(residual_tmp34*residual_tmp38 + residual_tmp40*residual_tmp65) - (s_t(1) / s_t(3))*residual_tmp36*residual_tmp69) + residual_tmp6*(s_t(2)*residual_tmp1*residual_tmp5 - s_t(2)*residual_tmp3);
      grad_coeff0_0_values[lane] = grad_coeff0_0;
      grad_coeff0_1_values[lane] = grad_coeff0_1;
      grad_coeff0_2_values[lane] = grad_coeff0_2;
      grad_coeff1_0_values[lane] = grad_coeff1_0;
      grad_coeff1_1_values[lane] = grad_coeff1_1;
      grad_coeff1_2_values[lane] = grad_coeff1_2;
      grad_coeff2_0_values[lane] = grad_coeff2_0;
      grad_coeff2_1_values[lane] = grad_coeff2_1;
      grad_coeff2_2_values[lane] = grad_coeff2_2;
    }
    for (int test = 0; test < NS; ++test) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
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
        output[test * NC][lane] += q_weight[q] * det * (grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
        output[test * NC + 1][lane] += q_weight[q] * det * (grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
        output[test * NC + 2][lane] += q_weight[q] * det * (grad_coeff2_0_values[lane] * test_grad0 + grad_coeff2_1_values[lane] * test_grad1 + grad_coeff2_2_values[lane] * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_residual_merit_d3_simplex_residual_block_contiguous(
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
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
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
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u0_grad_0_ref_values[lane] = s_t(0);
      u0_grad_1_ref_values[lane] = s_t(0);
      u0_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC][lane];
        u0_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u0_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u0_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u0_old_grad_0_ref_values[lane] = s_t(0);
      u0_old_grad_1_ref_values[lane] = s_t(0);
      u0_old_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC][lane];
        u0_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u0_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u0_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u1_grad_0_ref_values[lane] = s_t(0);
      u1_grad_1_ref_values[lane] = s_t(0);
      u1_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC + 1][lane];
        u1_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u1_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u1_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u1_old_grad_0_ref_values[lane] = s_t(0);
      u1_old_grad_1_ref_values[lane] = s_t(0);
      u1_old_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC + 1][lane];
        u1_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u1_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u1_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u2_grad_0_ref_values[lane] = s_t(0);
      u2_grad_1_ref_values[lane] = s_t(0);
      u2_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC + 2][lane];
        u2_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u2_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u2_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u2_old_grad_0_ref_values[lane] = s_t(0);
      u2_old_grad_1_ref_values[lane] = s_t(0);
      u2_old_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC + 2][lane];
        u2_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u2_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u2_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
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
      const s_t u0_grad_0_ref = u0_grad_0_ref_values[lane];
      const s_t u0_grad_1_ref = u0_grad_1_ref_values[lane];
      const s_t u0_grad_2_ref = u0_grad_2_ref_values[lane];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
      const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
      const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[lane];
      const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[lane];
      const s_t u0_old_grad_2_ref = u0_old_grad_2_ref_values[lane];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
      const s_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
      const s_t u1_grad_0_ref = u1_grad_0_ref_values[lane];
      const s_t u1_grad_1_ref = u1_grad_1_ref_values[lane];
      const s_t u1_grad_2_ref = u1_grad_2_ref_values[lane];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
      const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
      const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[lane];
      const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[lane];
      const s_t u1_old_grad_2_ref = u1_old_grad_2_ref_values[lane];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
      const s_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
      const s_t u2_grad_0_ref = u2_grad_0_ref_values[lane];
      const s_t u2_grad_1_ref = u2_grad_1_ref_values[lane];
      const s_t u2_grad_2_ref = u2_grad_2_ref_values[lane];
      const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
      const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
      const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
      const s_t u2_old_grad_0_ref = u2_old_grad_0_ref_values[lane];
      const s_t u2_old_grad_1_ref = u2_old_grad_1_ref_values[lane];
      const s_t u2_old_grad_2_ref = u2_old_grad_2_ref_values[lane];
      const s_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
      const s_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
      const s_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = u1_grad_2*u2_grad_1;
      const s_t residual_tmp1 = u1_grad_1 + s_t(1);
      const s_t residual_tmp2 = u2_grad_2 + s_t(1);
      const s_t residual_tmp3 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp4 = u0_grad_2*u2_grad_0;
      const s_t residual_tmp5 = u0_grad_0 + s_t(1);
      const s_t residual_tmp6 = ((s_t(1) / s_t(2)))*lmbda*(-residual_tmp0*residual_tmp5 + residual_tmp1*residual_tmp2*residual_tmp5 - residual_tmp1*residual_tmp4 - residual_tmp2*residual_tmp3 + u0_grad_1*u1_grad_2*u2_grad_0 + u0_grad_2*u1_grad_0*u2_grad_1 + s_t(-1));
      const s_t residual_tmp7 = residual_tmp1*u1_grad_0 + residual_tmp5*u0_grad_1 + u2_grad_0*u2_grad_1;
      const s_t residual_tmp8 = s_t(2)*u0_grad_1;
      const s_t residual_tmp9 = residual_tmp2*u2_grad_0 + residual_tmp5*u0_grad_2 + u1_grad_0*u1_grad_2;
      const s_t residual_tmp10 = s_t(2)*u0_grad_2;
      const s_t residual_tmp11 = pow_2(residual_tmp5) + pow_2(u1_grad_0) + pow_2(u2_grad_0);
      const s_t residual_tmp12 = s_t(2)*residual_tmp5;
      const s_t residual_tmp13 = pow_2(residual_tmp1) + pow_2(u0_grad_1) + pow_2(u2_grad_1);
      const s_t residual_tmp14 = pow_2(residual_tmp2) + pow_2(u0_grad_2) + pow_2(u1_grad_2);
      const s_t residual_tmp15 = residual_tmp11 + residual_tmp13 + residual_tmp14;
      const s_t residual_tmp16 = u0_grad_0*u1_grad_1;
      const s_t residual_tmp17 = u0_grad_1*u2_grad_0;
      const s_t residual_tmp18 = u0_grad_2*u2_grad_1;
      const s_t residual_tmp19 = -residual_tmp4 + u0_grad_0*u2_grad_2 + u0_grad_0;
      const s_t residual_tmp20 = -residual_tmp0 + residual_tmp1 + u1_grad_1*u2_grad_2 + u2_grad_2;
      const s_t residual_tmp21 = residual_tmp16 - residual_tmp3;
      const s_t residual_tmp22 = pow_m1(-residual_tmp0*u0_grad_0 + residual_tmp16*u2_grad_2 + residual_tmp17*u1_grad_2 + residual_tmp18*u1_grad_0 + residual_tmp19 + residual_tmp20 + residual_tmp21 - residual_tmp3*u2_grad_2 - residual_tmp4*u1_grad_1);
      const s_t residual_tmp23 = u0_grad_1*u1_grad_2;
      const s_t residual_tmp24 = -residual_tmp23 + u0_grad_2*u1_grad_1 + u0_grad_2;
      const s_t residual_tmp25 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp26 = u0_grad_1*u_dt_shift + u0_old_grad_1;
      const s_t residual_tmp27 = u0_grad_2*u1_grad_0;
      const s_t residual_tmp28 = -residual_tmp27 + u0_grad_0*u1_grad_2 + u1_grad_2;
      const s_t residual_tmp29 = u2_grad_1*u_dt_shift + u2_old_grad_1;
      const s_t residual_tmp30 = u1_grad_2*u2_grad_0;
      const s_t residual_tmp31 = -residual_tmp30 + u1_grad_0*u2_grad_2 + u1_grad_0;
      const s_t residual_tmp32 = u2_grad_2*u_dt_shift + u2_old_grad_2;
      const s_t residual_tmp33 = u1_grad_0*u2_grad_1;
      const s_t residual_tmp34 = -residual_tmp33 + u1_grad_1*u2_grad_0 + u2_grad_0;
      const s_t residual_tmp35 = u0_grad_2*u_dt_shift + u0_old_grad_2;
      const s_t residual_tmp36 = residual_tmp1 + residual_tmp21 + u0_grad_0;
      const s_t residual_tmp37 = u2_grad_0*u_dt_shift + u2_old_grad_0;
      const s_t residual_tmp38 = -residual_tmp20*residual_tmp37 + residual_tmp24*residual_tmp25 + residual_tmp26*residual_tmp28 + residual_tmp29*residual_tmp31 + residual_tmp32*residual_tmp34 - residual_tmp35*residual_tmp36;
      const s_t residual_tmp39 = -residual_tmp18 + u0_grad_1*u2_grad_2 + u0_grad_1;
      const s_t residual_tmp40 = -residual_tmp17 + u0_grad_0*u2_grad_1 + u2_grad_1;
      const s_t residual_tmp41 = u1_grad_1*u_dt_shift + u1_old_grad_1;
      const s_t residual_tmp42 = u1_grad_2*u_dt_shift + u1_old_grad_2;
      const s_t residual_tmp43 = residual_tmp19 + residual_tmp2;
      const s_t residual_tmp44 = u1_grad_0*u_dt_shift + u1_old_grad_0;
      const s_t residual_tmp45 = -residual_tmp20*residual_tmp44 + residual_tmp25*residual_tmp39 - residual_tmp26*residual_tmp43 + residual_tmp31*residual_tmp41 + residual_tmp34*residual_tmp42 + residual_tmp35*residual_tmp40;
      const s_t residual_tmp46 = residual_tmp39*residual_tmp44;
      const s_t residual_tmp47 = residual_tmp40*residual_tmp42;
      const s_t residual_tmp48 = residual_tmp24*residual_tmp37;
      const s_t residual_tmp49 = residual_tmp28*residual_tmp29;
      const s_t residual_tmp50 = residual_tmp41*residual_tmp43;
      const s_t residual_tmp51 = -residual_tmp50;
      const s_t residual_tmp52 = residual_tmp32*residual_tmp36;
      const s_t residual_tmp53 = -residual_tmp52;
      const s_t residual_tmp54 = residual_tmp46 + residual_tmp47 + residual_tmp48 + residual_tmp49 + residual_tmp51 + residual_tmp53;
      const s_t residual_tmp55 = residual_tmp20*residual_tmp25;
      const s_t residual_tmp56 = residual_tmp26*residual_tmp31 + residual_tmp34*residual_tmp35 - residual_tmp55;
      const s_t residual_tmp57 = s_t(3)*eta_b*(residual_tmp54 + residual_tmp56);
      const s_t residual_tmp58 = s_t(2)*eta_s;
      const s_t residual_tmp59 = residual_tmp57 + residual_tmp58*(s_t(2)*residual_tmp26*residual_tmp31 + s_t(2)*residual_tmp34*residual_tmp35 - residual_tmp54 - s_t(2)*residual_tmp55);
      const s_t residual_tmp60 = s_t(2)*u1_grad_0;
      const s_t residual_tmp61 = residual_tmp1*u1_grad_2 + residual_tmp2*u2_grad_1 + u0_grad_1*u0_grad_2;
      const s_t residual_tmp62 = s_t(2)*u2_grad_0;
      const s_t residual_tmp63 = s_t(2)*u1_grad_2;
      const s_t residual_tmp64 = s_t(2)*residual_tmp1;
      const s_t residual_tmp65 = residual_tmp24*residual_tmp44 + residual_tmp28*residual_tmp41 - residual_tmp29*residual_tmp43 + residual_tmp32*residual_tmp40 - residual_tmp36*residual_tmp42 + residual_tmp37*residual_tmp39;
      const s_t residual_tmp66 = residual_tmp57 + residual_tmp58*(s_t(2)*residual_tmp39*residual_tmp44 + s_t(2)*residual_tmp40*residual_tmp42 - residual_tmp48 - residual_tmp49 - s_t(2)*residual_tmp50 - residual_tmp53 - residual_tmp56);
      const s_t residual_tmp67 = s_t(2)*u2_grad_1;
      const s_t residual_tmp68 = s_t(2)*residual_tmp2;
      const s_t residual_tmp69 = residual_tmp57 + residual_tmp58*(s_t(2)*residual_tmp24*residual_tmp37 + s_t(2)*residual_tmp28*residual_tmp29 - residual_tmp46 - residual_tmp47 - residual_tmp51 - s_t(2)*residual_tmp52 - residual_tmp56);
      const s_t grad_coeff0_0 = mu*(s_t(6)*residual_tmp0 - s_t(6)*residual_tmp1*residual_tmp2 - residual_tmp10*residual_tmp9 - residual_tmp11*residual_tmp12 + residual_tmp12*residual_tmp15 - residual_tmp7*residual_tmp8 + s_t(2)*u0_grad_0 + s_t(2)) + residual_tmp22*(eta_s*(residual_tmp24*residual_tmp38 + residual_tmp39*residual_tmp45) - (s_t(1) / s_t(3))*residual_tmp20*residual_tmp59) + residual_tmp6*(-s_t(2)*residual_tmp0 + s_t(2)*residual_tmp1*residual_tmp2);
      const s_t grad_coeff0_1 = mu*(-residual_tmp10*residual_tmp61 - residual_tmp12*residual_tmp7 - residual_tmp13*residual_tmp8 + s_t(2)*residual_tmp15*u0_grad_1 + s_t(6)*residual_tmp2*u1_grad_0 - s_t(6)*residual_tmp30 + s_t(2)*u0_grad_1) + residual_tmp22*(-eta_s*(-residual_tmp28*residual_tmp38 + residual_tmp43*residual_tmp45) + ((s_t(1) / s_t(3)))*residual_tmp31*residual_tmp59) + residual_tmp6*(-residual_tmp2*residual_tmp60 + s_t(2)*u1_grad_2*u2_grad_0);
      const s_t grad_coeff0_2 = mu*(s_t(6)*residual_tmp1*u2_grad_0 - residual_tmp10*residual_tmp14 - residual_tmp12*residual_tmp9 + s_t(2)*residual_tmp15*u0_grad_2 - s_t(6)*residual_tmp33 - residual_tmp61*residual_tmp8 + s_t(2)*u0_grad_2) + residual_tmp22*(-eta_s*(residual_tmp36*residual_tmp38 - residual_tmp40*residual_tmp45) + ((s_t(1) / s_t(3)))*residual_tmp34*residual_tmp59) + residual_tmp6*(-residual_tmp1*residual_tmp62 + s_t(2)*residual_tmp33);
      const s_t grad_coeff1_0 = mu*(-residual_tmp11*residual_tmp60 + s_t(2)*residual_tmp15*u1_grad_0 - s_t(6)*residual_tmp18 + s_t(6)*residual_tmp2*u0_grad_1 - residual_tmp63*residual_tmp9 - residual_tmp64*residual_tmp7 + s_t(2)*u1_grad_0) + residual_tmp22*(-eta_s*(residual_tmp20*residual_tmp45 - residual_tmp24*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp39*residual_tmp66) + residual_tmp6*(-residual_tmp2*residual_tmp8 + s_t(2)*u0_grad_2*u2_grad_1);
      const s_t grad_coeff1_1 = mu*(-residual_tmp13*residual_tmp64 + residual_tmp15*residual_tmp64 - s_t(6)*residual_tmp2*residual_tmp5 + s_t(6)*residual_tmp4 - residual_tmp60*residual_tmp7 - residual_tmp61*residual_tmp63 + s_t(2)*u1_grad_1 + s_t(2)) + residual_tmp22*(eta_s*(residual_tmp28*residual_tmp65 + residual_tmp31*residual_tmp45) - (s_t(1) / s_t(3))*residual_tmp43*residual_tmp66) + residual_tmp6*(s_t(2)*residual_tmp2*residual_tmp5 - s_t(2)*residual_tmp4);
      const s_t grad_coeff1_2 = mu*(-residual_tmp14*residual_tmp63 + s_t(2)*residual_tmp15*u1_grad_2 - s_t(6)*residual_tmp17 + s_t(6)*residual_tmp5*u2_grad_1 - residual_tmp60*residual_tmp9 - residual_tmp61*residual_tmp64 + s_t(2)*u1_grad_2) + residual_tmp22*(-eta_s*(-residual_tmp34*residual_tmp45 + residual_tmp36*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp40*residual_tmp66) + residual_tmp6*(s_t(2)*residual_tmp17 - residual_tmp5*residual_tmp67);
      const s_t grad_coeff2_0 = mu*(s_t(6)*residual_tmp1*u0_grad_2 - residual_tmp11*residual_tmp62 + s_t(2)*residual_tmp15*u2_grad_0 - s_t(6)*residual_tmp23 - residual_tmp67*residual_tmp7 - residual_tmp68*residual_tmp9 + s_t(2)*u2_grad_0) + residual_tmp22*(-eta_s*(residual_tmp20*residual_tmp38 - residual_tmp39*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp24*residual_tmp69) + residual_tmp6*(-residual_tmp1*residual_tmp10 + s_t(2)*residual_tmp23);
      const s_t grad_coeff2_1 = mu*(-residual_tmp13*residual_tmp67 + s_t(2)*residual_tmp15*u2_grad_1 - s_t(6)*residual_tmp27 + s_t(6)*residual_tmp5*u1_grad_2 - residual_tmp61*residual_tmp68 - residual_tmp62*residual_tmp7 + s_t(2)*u2_grad_1) + residual_tmp22*(-eta_s*(-residual_tmp31*residual_tmp38 + residual_tmp43*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp28*residual_tmp69) + residual_tmp6*(s_t(2)*residual_tmp27 - residual_tmp5*residual_tmp63);
      const s_t grad_coeff2_2 = mu*(-s_t(6)*residual_tmp1*residual_tmp5 - residual_tmp14*residual_tmp68 + residual_tmp15*residual_tmp68 + s_t(6)*residual_tmp3 - residual_tmp61*residual_tmp67 - residual_tmp62*residual_tmp9 + s_t(2)*u2_grad_2 + s_t(2)) + residual_tmp22*(eta_s*(residual_tmp34*residual_tmp38 + residual_tmp40*residual_tmp65) - (s_t(1) / s_t(3))*residual_tmp36*residual_tmp69) + residual_tmp6*(s_t(2)*residual_tmp1*residual_tmp5 - s_t(2)*residual_tmp3);
      grad_coeff0_0_values[lane] = grad_coeff0_0;
      grad_coeff0_1_values[lane] = grad_coeff0_1;
      grad_coeff0_2_values[lane] = grad_coeff0_2;
      grad_coeff1_0_values[lane] = grad_coeff1_0;
      grad_coeff1_1_values[lane] = grad_coeff1_1;
      grad_coeff1_2_values[lane] = grad_coeff1_2;
      grad_coeff2_0_values[lane] = grad_coeff2_0;
      grad_coeff2_1_values[lane] = grad_coeff2_1;
      grad_coeff2_2_values[lane] = grad_coeff2_2;
    }
    for (int test = 0; test < NS; ++test) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
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
        output[test * NC][lane] += q_weight[q] * det * (grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
        output[test * NC + 1][lane] += q_weight[q] * det * (grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
        output[test * NC + 2][lane] += q_weight[q] * det * (grad_coeff2_0_values[lane] * test_grad0 + grad_coeff2_1_values[lane] * test_grad1 + grad_coeff2_2_values[lane] * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_residual_merit_d3_simplex_tet4_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
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
  for (int q = 0; q < NQ; ++q) {
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
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
      const s_t u0_grad_0_ref = -(current[0][lane]) + current[3][lane];
      const s_t u0_grad_1_ref = -(current[0][lane]) + current[6][lane];
      const s_t u0_grad_2_ref = -(current[0][lane]) + current[9][lane];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
      const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
      const s_t u0_old_grad_0_ref = -(previous[0][lane]) + previous[3][lane];
      const s_t u0_old_grad_1_ref = -(previous[0][lane]) + previous[6][lane];
      const s_t u0_old_grad_2_ref = -(previous[0][lane]) + previous[9][lane];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
      const s_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
      const s_t u1_grad_0_ref = -(current[1][lane]) + current[4][lane];
      const s_t u1_grad_1_ref = -(current[1][lane]) + current[7][lane];
      const s_t u1_grad_2_ref = -(current[1][lane]) + current[10][lane];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
      const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
      const s_t u1_old_grad_0_ref = -(previous[1][lane]) + previous[4][lane];
      const s_t u1_old_grad_1_ref = -(previous[1][lane]) + previous[7][lane];
      const s_t u1_old_grad_2_ref = -(previous[1][lane]) + previous[10][lane];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
      const s_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
      const s_t u2_grad_0_ref = -(current[2][lane]) + current[5][lane];
      const s_t u2_grad_1_ref = -(current[2][lane]) + current[8][lane];
      const s_t u2_grad_2_ref = -(current[2][lane]) + current[11][lane];
      const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
      const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
      const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
      const s_t u2_old_grad_0_ref = -(previous[2][lane]) + previous[5][lane];
      const s_t u2_old_grad_1_ref = -(previous[2][lane]) + previous[8][lane];
      const s_t u2_old_grad_2_ref = -(previous[2][lane]) + previous[11][lane];
      const s_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
      const s_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
      const s_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = u1_grad_2*u2_grad_1;
      const s_t residual_tmp1 = u1_grad_1 + s_t(1);
      const s_t residual_tmp2 = u2_grad_2 + s_t(1);
      const s_t residual_tmp3 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp4 = u0_grad_2*u2_grad_0;
      const s_t residual_tmp5 = u0_grad_0 + s_t(1);
      const s_t residual_tmp6 = ((s_t(1) / s_t(2)))*lmbda*(-residual_tmp0*residual_tmp5 + residual_tmp1*residual_tmp2*residual_tmp5 - residual_tmp1*residual_tmp4 - residual_tmp2*residual_tmp3 + u0_grad_1*u1_grad_2*u2_grad_0 + u0_grad_2*u1_grad_0*u2_grad_1 + s_t(-1));
      const s_t residual_tmp7 = residual_tmp1*u1_grad_0 + residual_tmp5*u0_grad_1 + u2_grad_0*u2_grad_1;
      const s_t residual_tmp8 = s_t(2)*u0_grad_1;
      const s_t residual_tmp9 = residual_tmp2*u2_grad_0 + residual_tmp5*u0_grad_2 + u1_grad_0*u1_grad_2;
      const s_t residual_tmp10 = s_t(2)*u0_grad_2;
      const s_t residual_tmp11 = pow_2(residual_tmp5) + pow_2(u1_grad_0) + pow_2(u2_grad_0);
      const s_t residual_tmp12 = s_t(2)*residual_tmp5;
      const s_t residual_tmp13 = pow_2(residual_tmp1) + pow_2(u0_grad_1) + pow_2(u2_grad_1);
      const s_t residual_tmp14 = pow_2(residual_tmp2) + pow_2(u0_grad_2) + pow_2(u1_grad_2);
      const s_t residual_tmp15 = residual_tmp11 + residual_tmp13 + residual_tmp14;
      const s_t residual_tmp16 = u0_grad_0*u1_grad_1;
      const s_t residual_tmp17 = u0_grad_1*u2_grad_0;
      const s_t residual_tmp18 = u0_grad_2*u2_grad_1;
      const s_t residual_tmp19 = -residual_tmp4 + u0_grad_0*u2_grad_2 + u0_grad_0;
      const s_t residual_tmp20 = -residual_tmp0 + residual_tmp1 + u1_grad_1*u2_grad_2 + u2_grad_2;
      const s_t residual_tmp21 = residual_tmp16 - residual_tmp3;
      const s_t residual_tmp22 = pow_m1(-residual_tmp0*u0_grad_0 + residual_tmp16*u2_grad_2 + residual_tmp17*u1_grad_2 + residual_tmp18*u1_grad_0 + residual_tmp19 + residual_tmp20 + residual_tmp21 - residual_tmp3*u2_grad_2 - residual_tmp4*u1_grad_1);
      const s_t residual_tmp23 = u0_grad_1*u1_grad_2;
      const s_t residual_tmp24 = -residual_tmp23 + u0_grad_2*u1_grad_1 + u0_grad_2;
      const s_t residual_tmp25 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp26 = u0_grad_1*u_dt_shift + u0_old_grad_1;
      const s_t residual_tmp27 = u0_grad_2*u1_grad_0;
      const s_t residual_tmp28 = -residual_tmp27 + u0_grad_0*u1_grad_2 + u1_grad_2;
      const s_t residual_tmp29 = u2_grad_1*u_dt_shift + u2_old_grad_1;
      const s_t residual_tmp30 = u1_grad_2*u2_grad_0;
      const s_t residual_tmp31 = -residual_tmp30 + u1_grad_0*u2_grad_2 + u1_grad_0;
      const s_t residual_tmp32 = u2_grad_2*u_dt_shift + u2_old_grad_2;
      const s_t residual_tmp33 = u1_grad_0*u2_grad_1;
      const s_t residual_tmp34 = -residual_tmp33 + u1_grad_1*u2_grad_0 + u2_grad_0;
      const s_t residual_tmp35 = u0_grad_2*u_dt_shift + u0_old_grad_2;
      const s_t residual_tmp36 = residual_tmp1 + residual_tmp21 + u0_grad_0;
      const s_t residual_tmp37 = u2_grad_0*u_dt_shift + u2_old_grad_0;
      const s_t residual_tmp38 = -residual_tmp20*residual_tmp37 + residual_tmp24*residual_tmp25 + residual_tmp26*residual_tmp28 + residual_tmp29*residual_tmp31 + residual_tmp32*residual_tmp34 - residual_tmp35*residual_tmp36;
      const s_t residual_tmp39 = -residual_tmp18 + u0_grad_1*u2_grad_2 + u0_grad_1;
      const s_t residual_tmp40 = -residual_tmp17 + u0_grad_0*u2_grad_1 + u2_grad_1;
      const s_t residual_tmp41 = u1_grad_1*u_dt_shift + u1_old_grad_1;
      const s_t residual_tmp42 = u1_grad_2*u_dt_shift + u1_old_grad_2;
      const s_t residual_tmp43 = residual_tmp19 + residual_tmp2;
      const s_t residual_tmp44 = u1_grad_0*u_dt_shift + u1_old_grad_0;
      const s_t residual_tmp45 = -residual_tmp20*residual_tmp44 + residual_tmp25*residual_tmp39 - residual_tmp26*residual_tmp43 + residual_tmp31*residual_tmp41 + residual_tmp34*residual_tmp42 + residual_tmp35*residual_tmp40;
      const s_t residual_tmp46 = residual_tmp39*residual_tmp44;
      const s_t residual_tmp47 = residual_tmp40*residual_tmp42;
      const s_t residual_tmp48 = residual_tmp24*residual_tmp37;
      const s_t residual_tmp49 = residual_tmp28*residual_tmp29;
      const s_t residual_tmp50 = residual_tmp41*residual_tmp43;
      const s_t residual_tmp51 = -residual_tmp50;
      const s_t residual_tmp52 = residual_tmp32*residual_tmp36;
      const s_t residual_tmp53 = -residual_tmp52;
      const s_t residual_tmp54 = residual_tmp46 + residual_tmp47 + residual_tmp48 + residual_tmp49 + residual_tmp51 + residual_tmp53;
      const s_t residual_tmp55 = residual_tmp20*residual_tmp25;
      const s_t residual_tmp56 = residual_tmp26*residual_tmp31 + residual_tmp34*residual_tmp35 - residual_tmp55;
      const s_t residual_tmp57 = s_t(3)*eta_b*(residual_tmp54 + residual_tmp56);
      const s_t residual_tmp58 = s_t(2)*eta_s;
      const s_t residual_tmp59 = residual_tmp57 + residual_tmp58*(s_t(2)*residual_tmp26*residual_tmp31 + s_t(2)*residual_tmp34*residual_tmp35 - residual_tmp54 - s_t(2)*residual_tmp55);
      const s_t residual_tmp60 = s_t(2)*u1_grad_0;
      const s_t residual_tmp61 = residual_tmp1*u1_grad_2 + residual_tmp2*u2_grad_1 + u0_grad_1*u0_grad_2;
      const s_t residual_tmp62 = s_t(2)*u2_grad_0;
      const s_t residual_tmp63 = s_t(2)*u1_grad_2;
      const s_t residual_tmp64 = s_t(2)*residual_tmp1;
      const s_t residual_tmp65 = residual_tmp24*residual_tmp44 + residual_tmp28*residual_tmp41 - residual_tmp29*residual_tmp43 + residual_tmp32*residual_tmp40 - residual_tmp36*residual_tmp42 + residual_tmp37*residual_tmp39;
      const s_t residual_tmp66 = residual_tmp57 + residual_tmp58*(s_t(2)*residual_tmp39*residual_tmp44 + s_t(2)*residual_tmp40*residual_tmp42 - residual_tmp48 - residual_tmp49 - s_t(2)*residual_tmp50 - residual_tmp53 - residual_tmp56);
      const s_t residual_tmp67 = s_t(2)*u2_grad_1;
      const s_t residual_tmp68 = s_t(2)*residual_tmp2;
      const s_t residual_tmp69 = residual_tmp57 + residual_tmp58*(s_t(2)*residual_tmp24*residual_tmp37 + s_t(2)*residual_tmp28*residual_tmp29 - residual_tmp46 - residual_tmp47 - residual_tmp51 - s_t(2)*residual_tmp52 - residual_tmp56);
      const s_t grad_coeff0_0 = mu*(s_t(6)*residual_tmp0 - s_t(6)*residual_tmp1*residual_tmp2 - residual_tmp10*residual_tmp9 - residual_tmp11*residual_tmp12 + residual_tmp12*residual_tmp15 - residual_tmp7*residual_tmp8 + s_t(2)*u0_grad_0 + s_t(2)) + residual_tmp22*(eta_s*(residual_tmp24*residual_tmp38 + residual_tmp39*residual_tmp45) - (s_t(1) / s_t(3))*residual_tmp20*residual_tmp59) + residual_tmp6*(-s_t(2)*residual_tmp0 + s_t(2)*residual_tmp1*residual_tmp2);
      const s_t grad_coeff0_1 = mu*(-residual_tmp10*residual_tmp61 - residual_tmp12*residual_tmp7 - residual_tmp13*residual_tmp8 + s_t(2)*residual_tmp15*u0_grad_1 + s_t(6)*residual_tmp2*u1_grad_0 - s_t(6)*residual_tmp30 + s_t(2)*u0_grad_1) + residual_tmp22*(-eta_s*(-residual_tmp28*residual_tmp38 + residual_tmp43*residual_tmp45) + ((s_t(1) / s_t(3)))*residual_tmp31*residual_tmp59) + residual_tmp6*(-residual_tmp2*residual_tmp60 + s_t(2)*u1_grad_2*u2_grad_0);
      const s_t grad_coeff0_2 = mu*(s_t(6)*residual_tmp1*u2_grad_0 - residual_tmp10*residual_tmp14 - residual_tmp12*residual_tmp9 + s_t(2)*residual_tmp15*u0_grad_2 - s_t(6)*residual_tmp33 - residual_tmp61*residual_tmp8 + s_t(2)*u0_grad_2) + residual_tmp22*(-eta_s*(residual_tmp36*residual_tmp38 - residual_tmp40*residual_tmp45) + ((s_t(1) / s_t(3)))*residual_tmp34*residual_tmp59) + residual_tmp6*(-residual_tmp1*residual_tmp62 + s_t(2)*residual_tmp33);
      const s_t grad_coeff1_0 = mu*(-residual_tmp11*residual_tmp60 + s_t(2)*residual_tmp15*u1_grad_0 - s_t(6)*residual_tmp18 + s_t(6)*residual_tmp2*u0_grad_1 - residual_tmp63*residual_tmp9 - residual_tmp64*residual_tmp7 + s_t(2)*u1_grad_0) + residual_tmp22*(-eta_s*(residual_tmp20*residual_tmp45 - residual_tmp24*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp39*residual_tmp66) + residual_tmp6*(-residual_tmp2*residual_tmp8 + s_t(2)*u0_grad_2*u2_grad_1);
      const s_t grad_coeff1_1 = mu*(-residual_tmp13*residual_tmp64 + residual_tmp15*residual_tmp64 - s_t(6)*residual_tmp2*residual_tmp5 + s_t(6)*residual_tmp4 - residual_tmp60*residual_tmp7 - residual_tmp61*residual_tmp63 + s_t(2)*u1_grad_1 + s_t(2)) + residual_tmp22*(eta_s*(residual_tmp28*residual_tmp65 + residual_tmp31*residual_tmp45) - (s_t(1) / s_t(3))*residual_tmp43*residual_tmp66) + residual_tmp6*(s_t(2)*residual_tmp2*residual_tmp5 - s_t(2)*residual_tmp4);
      const s_t grad_coeff1_2 = mu*(-residual_tmp14*residual_tmp63 + s_t(2)*residual_tmp15*u1_grad_2 - s_t(6)*residual_tmp17 + s_t(6)*residual_tmp5*u2_grad_1 - residual_tmp60*residual_tmp9 - residual_tmp61*residual_tmp64 + s_t(2)*u1_grad_2) + residual_tmp22*(-eta_s*(-residual_tmp34*residual_tmp45 + residual_tmp36*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp40*residual_tmp66) + residual_tmp6*(s_t(2)*residual_tmp17 - residual_tmp5*residual_tmp67);
      const s_t grad_coeff2_0 = mu*(s_t(6)*residual_tmp1*u0_grad_2 - residual_tmp11*residual_tmp62 + s_t(2)*residual_tmp15*u2_grad_0 - s_t(6)*residual_tmp23 - residual_tmp67*residual_tmp7 - residual_tmp68*residual_tmp9 + s_t(2)*u2_grad_0) + residual_tmp22*(-eta_s*(residual_tmp20*residual_tmp38 - residual_tmp39*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp24*residual_tmp69) + residual_tmp6*(-residual_tmp1*residual_tmp10 + s_t(2)*residual_tmp23);
      const s_t grad_coeff2_1 = mu*(-residual_tmp13*residual_tmp67 + s_t(2)*residual_tmp15*u2_grad_1 - s_t(6)*residual_tmp27 + s_t(6)*residual_tmp5*u1_grad_2 - residual_tmp61*residual_tmp68 - residual_tmp62*residual_tmp7 + s_t(2)*u2_grad_1) + residual_tmp22*(-eta_s*(-residual_tmp31*residual_tmp38 + residual_tmp43*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp28*residual_tmp69) + residual_tmp6*(s_t(2)*residual_tmp27 - residual_tmp5*residual_tmp63);
      const s_t grad_coeff2_2 = mu*(-s_t(6)*residual_tmp1*residual_tmp5 - residual_tmp14*residual_tmp68 + residual_tmp15*residual_tmp68 + s_t(6)*residual_tmp3 - residual_tmp61*residual_tmp67 - residual_tmp62*residual_tmp9 + s_t(2)*u2_grad_2 + s_t(2)) + residual_tmp22*(eta_s*(residual_tmp34*residual_tmp38 + residual_tmp40*residual_tmp65) - (s_t(1) / s_t(3))*residual_tmp36*residual_tmp69) + residual_tmp6*(s_t(2)*residual_tmp1*residual_tmp5 - s_t(2)*residual_tmp3);
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
      output[0][lane] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1 + grad_coeff0_2_value * test0_grad2);
      output[1][lane] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1 + grad_coeff1_2_value * test0_grad2);
      output[2][lane] += q_weight[q] * det * (grad_coeff2_0_value * test0_grad0 + grad_coeff2_1_value * test0_grad1 + grad_coeff2_2_value * test0_grad2);
      output[3][lane] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1 + grad_coeff0_2_value * test1_grad2);
      output[4][lane] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1 + grad_coeff1_2_value * test1_grad2);
      output[5][lane] += q_weight[q] * det * (grad_coeff2_0_value * test1_grad0 + grad_coeff2_1_value * test1_grad1 + grad_coeff2_2_value * test1_grad2);
      output[6][lane] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1 + grad_coeff0_2_value * test2_grad2);
      output[7][lane] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1 + grad_coeff1_2_value * test2_grad2);
      output[8][lane] += q_weight[q] * det * (grad_coeff2_0_value * test2_grad0 + grad_coeff2_1_value * test2_grad1 + grad_coeff2_2_value * test2_grad2);
      output[9][lane] += q_weight[q] * det * (grad_coeff0_0_value * test3_grad0 + grad_coeff0_1_value * test3_grad1 + grad_coeff0_2_value * test3_grad2);
      output[10][lane] += q_weight[q] * det * (grad_coeff1_0_value * test3_grad0 + grad_coeff1_1_value * test3_grad1 + grad_coeff1_2_value * test3_grad2);
      output[11][lane] += q_weight[q] * det * (grad_coeff2_0_value * test3_grad0 + grad_coeff2_1_value * test3_grad1 + grad_coeff2_2_value * test3_grad2);
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_residual_merit_d3_simplex_tet4_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR q_weight,
    const s_t current[3 * NS][VS],
    const s_t previous[3 * NS][VS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t output[3 * NS][VS]
) {
  for (int q = 0; q < NQ; ++q) {
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
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
      const s_t u0_grad_0_ref = -(current[0][lane]) + current[3][lane];
      const s_t u0_grad_1_ref = -(current[0][lane]) + current[6][lane];
      const s_t u0_grad_2_ref = -(current[0][lane]) + current[9][lane];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
      const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
      const s_t u0_old_grad_0_ref = -(previous[0][lane]) + previous[3][lane];
      const s_t u0_old_grad_1_ref = -(previous[0][lane]) + previous[6][lane];
      const s_t u0_old_grad_2_ref = -(previous[0][lane]) + previous[9][lane];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
      const s_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
      const s_t u1_grad_0_ref = -(current[1][lane]) + current[4][lane];
      const s_t u1_grad_1_ref = -(current[1][lane]) + current[7][lane];
      const s_t u1_grad_2_ref = -(current[1][lane]) + current[10][lane];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
      const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
      const s_t u1_old_grad_0_ref = -(previous[1][lane]) + previous[4][lane];
      const s_t u1_old_grad_1_ref = -(previous[1][lane]) + previous[7][lane];
      const s_t u1_old_grad_2_ref = -(previous[1][lane]) + previous[10][lane];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
      const s_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
      const s_t u2_grad_0_ref = -(current[2][lane]) + current[5][lane];
      const s_t u2_grad_1_ref = -(current[2][lane]) + current[8][lane];
      const s_t u2_grad_2_ref = -(current[2][lane]) + current[11][lane];
      const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
      const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
      const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
      const s_t u2_old_grad_0_ref = -(previous[2][lane]) + previous[5][lane];
      const s_t u2_old_grad_1_ref = -(previous[2][lane]) + previous[8][lane];
      const s_t u2_old_grad_2_ref = -(previous[2][lane]) + previous[11][lane];
      const s_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
      const s_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
      const s_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = u1_grad_2*u2_grad_1;
      const s_t residual_tmp1 = u1_grad_1 + s_t(1);
      const s_t residual_tmp2 = u2_grad_2 + s_t(1);
      const s_t residual_tmp3 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp4 = u0_grad_2*u2_grad_0;
      const s_t residual_tmp5 = u0_grad_0 + s_t(1);
      const s_t residual_tmp6 = ((s_t(1) / s_t(2)))*lmbda*(-residual_tmp0*residual_tmp5 + residual_tmp1*residual_tmp2*residual_tmp5 - residual_tmp1*residual_tmp4 - residual_tmp2*residual_tmp3 + u0_grad_1*u1_grad_2*u2_grad_0 + u0_grad_2*u1_grad_0*u2_grad_1 + s_t(-1));
      const s_t residual_tmp7 = residual_tmp1*u1_grad_0 + residual_tmp5*u0_grad_1 + u2_grad_0*u2_grad_1;
      const s_t residual_tmp8 = s_t(2)*u0_grad_1;
      const s_t residual_tmp9 = residual_tmp2*u2_grad_0 + residual_tmp5*u0_grad_2 + u1_grad_0*u1_grad_2;
      const s_t residual_tmp10 = s_t(2)*u0_grad_2;
      const s_t residual_tmp11 = pow_2(residual_tmp5) + pow_2(u1_grad_0) + pow_2(u2_grad_0);
      const s_t residual_tmp12 = s_t(2)*residual_tmp5;
      const s_t residual_tmp13 = pow_2(residual_tmp1) + pow_2(u0_grad_1) + pow_2(u2_grad_1);
      const s_t residual_tmp14 = pow_2(residual_tmp2) + pow_2(u0_grad_2) + pow_2(u1_grad_2);
      const s_t residual_tmp15 = residual_tmp11 + residual_tmp13 + residual_tmp14;
      const s_t residual_tmp16 = u0_grad_0*u1_grad_1;
      const s_t residual_tmp17 = u0_grad_1*u2_grad_0;
      const s_t residual_tmp18 = u0_grad_2*u2_grad_1;
      const s_t residual_tmp19 = -residual_tmp4 + u0_grad_0*u2_grad_2 + u0_grad_0;
      const s_t residual_tmp20 = -residual_tmp0 + residual_tmp1 + u1_grad_1*u2_grad_2 + u2_grad_2;
      const s_t residual_tmp21 = residual_tmp16 - residual_tmp3;
      const s_t residual_tmp22 = pow_m1(-residual_tmp0*u0_grad_0 + residual_tmp16*u2_grad_2 + residual_tmp17*u1_grad_2 + residual_tmp18*u1_grad_0 + residual_tmp19 + residual_tmp20 + residual_tmp21 - residual_tmp3*u2_grad_2 - residual_tmp4*u1_grad_1);
      const s_t residual_tmp23 = u0_grad_1*u1_grad_2;
      const s_t residual_tmp24 = -residual_tmp23 + u0_grad_2*u1_grad_1 + u0_grad_2;
      const s_t residual_tmp25 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp26 = u0_grad_1*u_dt_shift + u0_old_grad_1;
      const s_t residual_tmp27 = u0_grad_2*u1_grad_0;
      const s_t residual_tmp28 = -residual_tmp27 + u0_grad_0*u1_grad_2 + u1_grad_2;
      const s_t residual_tmp29 = u2_grad_1*u_dt_shift + u2_old_grad_1;
      const s_t residual_tmp30 = u1_grad_2*u2_grad_0;
      const s_t residual_tmp31 = -residual_tmp30 + u1_grad_0*u2_grad_2 + u1_grad_0;
      const s_t residual_tmp32 = u2_grad_2*u_dt_shift + u2_old_grad_2;
      const s_t residual_tmp33 = u1_grad_0*u2_grad_1;
      const s_t residual_tmp34 = -residual_tmp33 + u1_grad_1*u2_grad_0 + u2_grad_0;
      const s_t residual_tmp35 = u0_grad_2*u_dt_shift + u0_old_grad_2;
      const s_t residual_tmp36 = residual_tmp1 + residual_tmp21 + u0_grad_0;
      const s_t residual_tmp37 = u2_grad_0*u_dt_shift + u2_old_grad_0;
      const s_t residual_tmp38 = -residual_tmp20*residual_tmp37 + residual_tmp24*residual_tmp25 + residual_tmp26*residual_tmp28 + residual_tmp29*residual_tmp31 + residual_tmp32*residual_tmp34 - residual_tmp35*residual_tmp36;
      const s_t residual_tmp39 = -residual_tmp18 + u0_grad_1*u2_grad_2 + u0_grad_1;
      const s_t residual_tmp40 = -residual_tmp17 + u0_grad_0*u2_grad_1 + u2_grad_1;
      const s_t residual_tmp41 = u1_grad_1*u_dt_shift + u1_old_grad_1;
      const s_t residual_tmp42 = u1_grad_2*u_dt_shift + u1_old_grad_2;
      const s_t residual_tmp43 = residual_tmp19 + residual_tmp2;
      const s_t residual_tmp44 = u1_grad_0*u_dt_shift + u1_old_grad_0;
      const s_t residual_tmp45 = -residual_tmp20*residual_tmp44 + residual_tmp25*residual_tmp39 - residual_tmp26*residual_tmp43 + residual_tmp31*residual_tmp41 + residual_tmp34*residual_tmp42 + residual_tmp35*residual_tmp40;
      const s_t residual_tmp46 = residual_tmp39*residual_tmp44;
      const s_t residual_tmp47 = residual_tmp40*residual_tmp42;
      const s_t residual_tmp48 = residual_tmp24*residual_tmp37;
      const s_t residual_tmp49 = residual_tmp28*residual_tmp29;
      const s_t residual_tmp50 = residual_tmp41*residual_tmp43;
      const s_t residual_tmp51 = -residual_tmp50;
      const s_t residual_tmp52 = residual_tmp32*residual_tmp36;
      const s_t residual_tmp53 = -residual_tmp52;
      const s_t residual_tmp54 = residual_tmp46 + residual_tmp47 + residual_tmp48 + residual_tmp49 + residual_tmp51 + residual_tmp53;
      const s_t residual_tmp55 = residual_tmp20*residual_tmp25;
      const s_t residual_tmp56 = residual_tmp26*residual_tmp31 + residual_tmp34*residual_tmp35 - residual_tmp55;
      const s_t residual_tmp57 = s_t(3)*eta_b*(residual_tmp54 + residual_tmp56);
      const s_t residual_tmp58 = s_t(2)*eta_s;
      const s_t residual_tmp59 = residual_tmp57 + residual_tmp58*(s_t(2)*residual_tmp26*residual_tmp31 + s_t(2)*residual_tmp34*residual_tmp35 - residual_tmp54 - s_t(2)*residual_tmp55);
      const s_t residual_tmp60 = s_t(2)*u1_grad_0;
      const s_t residual_tmp61 = residual_tmp1*u1_grad_2 + residual_tmp2*u2_grad_1 + u0_grad_1*u0_grad_2;
      const s_t residual_tmp62 = s_t(2)*u2_grad_0;
      const s_t residual_tmp63 = s_t(2)*u1_grad_2;
      const s_t residual_tmp64 = s_t(2)*residual_tmp1;
      const s_t residual_tmp65 = residual_tmp24*residual_tmp44 + residual_tmp28*residual_tmp41 - residual_tmp29*residual_tmp43 + residual_tmp32*residual_tmp40 - residual_tmp36*residual_tmp42 + residual_tmp37*residual_tmp39;
      const s_t residual_tmp66 = residual_tmp57 + residual_tmp58*(s_t(2)*residual_tmp39*residual_tmp44 + s_t(2)*residual_tmp40*residual_tmp42 - residual_tmp48 - residual_tmp49 - s_t(2)*residual_tmp50 - residual_tmp53 - residual_tmp56);
      const s_t residual_tmp67 = s_t(2)*u2_grad_1;
      const s_t residual_tmp68 = s_t(2)*residual_tmp2;
      const s_t residual_tmp69 = residual_tmp57 + residual_tmp58*(s_t(2)*residual_tmp24*residual_tmp37 + s_t(2)*residual_tmp28*residual_tmp29 - residual_tmp46 - residual_tmp47 - residual_tmp51 - s_t(2)*residual_tmp52 - residual_tmp56);
      const s_t grad_coeff0_0 = mu*(s_t(6)*residual_tmp0 - s_t(6)*residual_tmp1*residual_tmp2 - residual_tmp10*residual_tmp9 - residual_tmp11*residual_tmp12 + residual_tmp12*residual_tmp15 - residual_tmp7*residual_tmp8 + s_t(2)*u0_grad_0 + s_t(2)) + residual_tmp22*(eta_s*(residual_tmp24*residual_tmp38 + residual_tmp39*residual_tmp45) - (s_t(1) / s_t(3))*residual_tmp20*residual_tmp59) + residual_tmp6*(-s_t(2)*residual_tmp0 + s_t(2)*residual_tmp1*residual_tmp2);
      const s_t grad_coeff0_1 = mu*(-residual_tmp10*residual_tmp61 - residual_tmp12*residual_tmp7 - residual_tmp13*residual_tmp8 + s_t(2)*residual_tmp15*u0_grad_1 + s_t(6)*residual_tmp2*u1_grad_0 - s_t(6)*residual_tmp30 + s_t(2)*u0_grad_1) + residual_tmp22*(-eta_s*(-residual_tmp28*residual_tmp38 + residual_tmp43*residual_tmp45) + ((s_t(1) / s_t(3)))*residual_tmp31*residual_tmp59) + residual_tmp6*(-residual_tmp2*residual_tmp60 + s_t(2)*u1_grad_2*u2_grad_0);
      const s_t grad_coeff0_2 = mu*(s_t(6)*residual_tmp1*u2_grad_0 - residual_tmp10*residual_tmp14 - residual_tmp12*residual_tmp9 + s_t(2)*residual_tmp15*u0_grad_2 - s_t(6)*residual_tmp33 - residual_tmp61*residual_tmp8 + s_t(2)*u0_grad_2) + residual_tmp22*(-eta_s*(residual_tmp36*residual_tmp38 - residual_tmp40*residual_tmp45) + ((s_t(1) / s_t(3)))*residual_tmp34*residual_tmp59) + residual_tmp6*(-residual_tmp1*residual_tmp62 + s_t(2)*residual_tmp33);
      const s_t grad_coeff1_0 = mu*(-residual_tmp11*residual_tmp60 + s_t(2)*residual_tmp15*u1_grad_0 - s_t(6)*residual_tmp18 + s_t(6)*residual_tmp2*u0_grad_1 - residual_tmp63*residual_tmp9 - residual_tmp64*residual_tmp7 + s_t(2)*u1_grad_0) + residual_tmp22*(-eta_s*(residual_tmp20*residual_tmp45 - residual_tmp24*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp39*residual_tmp66) + residual_tmp6*(-residual_tmp2*residual_tmp8 + s_t(2)*u0_grad_2*u2_grad_1);
      const s_t grad_coeff1_1 = mu*(-residual_tmp13*residual_tmp64 + residual_tmp15*residual_tmp64 - s_t(6)*residual_tmp2*residual_tmp5 + s_t(6)*residual_tmp4 - residual_tmp60*residual_tmp7 - residual_tmp61*residual_tmp63 + s_t(2)*u1_grad_1 + s_t(2)) + residual_tmp22*(eta_s*(residual_tmp28*residual_tmp65 + residual_tmp31*residual_tmp45) - (s_t(1) / s_t(3))*residual_tmp43*residual_tmp66) + residual_tmp6*(s_t(2)*residual_tmp2*residual_tmp5 - s_t(2)*residual_tmp4);
      const s_t grad_coeff1_2 = mu*(-residual_tmp14*residual_tmp63 + s_t(2)*residual_tmp15*u1_grad_2 - s_t(6)*residual_tmp17 + s_t(6)*residual_tmp5*u2_grad_1 - residual_tmp60*residual_tmp9 - residual_tmp61*residual_tmp64 + s_t(2)*u1_grad_2) + residual_tmp22*(-eta_s*(-residual_tmp34*residual_tmp45 + residual_tmp36*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp40*residual_tmp66) + residual_tmp6*(s_t(2)*residual_tmp17 - residual_tmp5*residual_tmp67);
      const s_t grad_coeff2_0 = mu*(s_t(6)*residual_tmp1*u0_grad_2 - residual_tmp11*residual_tmp62 + s_t(2)*residual_tmp15*u2_grad_0 - s_t(6)*residual_tmp23 - residual_tmp67*residual_tmp7 - residual_tmp68*residual_tmp9 + s_t(2)*u2_grad_0) + residual_tmp22*(-eta_s*(residual_tmp20*residual_tmp38 - residual_tmp39*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp24*residual_tmp69) + residual_tmp6*(-residual_tmp1*residual_tmp10 + s_t(2)*residual_tmp23);
      const s_t grad_coeff2_1 = mu*(-residual_tmp13*residual_tmp67 + s_t(2)*residual_tmp15*u2_grad_1 - s_t(6)*residual_tmp27 + s_t(6)*residual_tmp5*u1_grad_2 - residual_tmp61*residual_tmp68 - residual_tmp62*residual_tmp7 + s_t(2)*u2_grad_1) + residual_tmp22*(-eta_s*(-residual_tmp31*residual_tmp38 + residual_tmp43*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp28*residual_tmp69) + residual_tmp6*(s_t(2)*residual_tmp27 - residual_tmp5*residual_tmp63);
      const s_t grad_coeff2_2 = mu*(-s_t(6)*residual_tmp1*residual_tmp5 - residual_tmp14*residual_tmp68 + residual_tmp15*residual_tmp68 + s_t(6)*residual_tmp3 - residual_tmp61*residual_tmp67 - residual_tmp62*residual_tmp9 + s_t(2)*u2_grad_2 + s_t(2)) + residual_tmp22*(eta_s*(residual_tmp34*residual_tmp38 + residual_tmp40*residual_tmp65) - (s_t(1) / s_t(3))*residual_tmp36*residual_tmp69) + residual_tmp6*(s_t(2)*residual_tmp1*residual_tmp5 - s_t(2)*residual_tmp3);
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
      output[0][lane] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1 + grad_coeff0_2_value * test0_grad2);
      output[1][lane] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1 + grad_coeff1_2_value * test0_grad2);
      output[2][lane] += q_weight[q] * det * (grad_coeff2_0_value * test0_grad0 + grad_coeff2_1_value * test0_grad1 + grad_coeff2_2_value * test0_grad2);
      output[3][lane] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1 + grad_coeff0_2_value * test1_grad2);
      output[4][lane] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1 + grad_coeff1_2_value * test1_grad2);
      output[5][lane] += q_weight[q] * det * (grad_coeff2_0_value * test1_grad0 + grad_coeff2_1_value * test1_grad1 + grad_coeff2_2_value * test1_grad2);
      output[6][lane] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1 + grad_coeff0_2_value * test2_grad2);
      output[7][lane] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1 + grad_coeff1_2_value * test2_grad2);
      output[8][lane] += q_weight[q] * det * (grad_coeff2_0_value * test2_grad0 + grad_coeff2_1_value * test2_grad1 + grad_coeff2_2_value * test2_grad2);
      output[9][lane] += q_weight[q] * det * (grad_coeff0_0_value * test3_grad0 + grad_coeff0_1_value * test3_grad1 + grad_coeff0_2_value * test3_grad2);
      output[10][lane] += q_weight[q] * det * (grad_coeff1_0_value * test3_grad0 + grad_coeff1_1_value * test3_grad1 + grad_coeff1_2_value * test3_grad2);
      output[11][lane] += q_weight[q] * det * (grad_coeff2_0_value * test3_grad0 + grad_coeff2_1_value * test3_grad1 + grad_coeff2_2_value * test3_grad2);
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_residual_merit_d3_simplex_jacobian_action_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[3 * NS],
    const s_t *const RSTR previous[3 * NS],
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
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u0_grad_0_ref_values[lane] = s_t(0);
      u0_grad_1_ref_values[lane] = s_t(0);
      u0_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC][lane];
        u0_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u0_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u0_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u0_old_grad_0_ref_values[lane] = s_t(0);
      u0_old_grad_1_ref_values[lane] = s_t(0);
      u0_old_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC][lane];
        u0_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u0_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u0_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u0_direction_grad_0_ref_values[lane] = s_t(0);
      u0_direction_grad_1_ref_values[lane] = s_t(0);
      u0_direction_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = direction[trial * NC][lane];
        u0_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u0_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u0_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u1_grad_0_ref_values[lane] = s_t(0);
      u1_grad_1_ref_values[lane] = s_t(0);
      u1_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC + 1][lane];
        u1_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u1_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u1_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u1_old_grad_0_ref_values[lane] = s_t(0);
      u1_old_grad_1_ref_values[lane] = s_t(0);
      u1_old_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC + 1][lane];
        u1_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u1_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u1_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u1_direction_grad_0_ref_values[lane] = s_t(0);
      u1_direction_grad_1_ref_values[lane] = s_t(0);
      u1_direction_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = direction[trial * NC + 1][lane];
        u1_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u1_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u1_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u2_grad_0_ref_values[lane] = s_t(0);
      u2_grad_1_ref_values[lane] = s_t(0);
      u2_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC + 2][lane];
        u2_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u2_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u2_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u2_old_grad_0_ref_values[lane] = s_t(0);
      u2_old_grad_1_ref_values[lane] = s_t(0);
      u2_old_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC + 2][lane];
        u2_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u2_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u2_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u2_direction_grad_0_ref_values[lane] = s_t(0);
      u2_direction_grad_1_ref_values[lane] = s_t(0);
      u2_direction_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = direction[trial * NC + 2][lane];
        u2_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u2_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u2_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
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
      const s_t u0_grad_0_ref = u0_grad_0_ref_values[lane];
      const s_t u0_grad_1_ref = u0_grad_1_ref_values[lane];
      const s_t u0_grad_2_ref = u0_grad_2_ref_values[lane];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
      const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
      const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[lane];
      const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[lane];
      const s_t u0_old_grad_2_ref = u0_old_grad_2_ref_values[lane];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
      const s_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
      const s_t u0_direction_grad_0_ref = u0_direction_grad_0_ref_values[lane];
      const s_t u0_direction_grad_1_ref = u0_direction_grad_1_ref_values[lane];
      const s_t u0_direction_grad_2_ref = u0_direction_grad_2_ref_values[lane];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
      const s_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
      const s_t u1_grad_0_ref = u1_grad_0_ref_values[lane];
      const s_t u1_grad_1_ref = u1_grad_1_ref_values[lane];
      const s_t u1_grad_2_ref = u1_grad_2_ref_values[lane];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
      const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
      const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[lane];
      const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[lane];
      const s_t u1_old_grad_2_ref = u1_old_grad_2_ref_values[lane];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
      const s_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
      const s_t u1_direction_grad_0_ref = u1_direction_grad_0_ref_values[lane];
      const s_t u1_direction_grad_1_ref = u1_direction_grad_1_ref_values[lane];
      const s_t u1_direction_grad_2_ref = u1_direction_grad_2_ref_values[lane];
      const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
      const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
      const s_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
      const s_t u2_grad_0_ref = u2_grad_0_ref_values[lane];
      const s_t u2_grad_1_ref = u2_grad_1_ref_values[lane];
      const s_t u2_grad_2_ref = u2_grad_2_ref_values[lane];
      const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
      const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
      const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
      const s_t u2_old_grad_0_ref = u2_old_grad_0_ref_values[lane];
      const s_t u2_old_grad_1_ref = u2_old_grad_1_ref_values[lane];
      const s_t u2_old_grad_2_ref = u2_old_grad_2_ref_values[lane];
      const s_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
      const s_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
      const s_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
      const s_t u2_direction_grad_0_ref = u2_direction_grad_0_ref_values[lane];
      const s_t u2_direction_grad_1_ref = u2_direction_grad_1_ref_values[lane];
      const s_t u2_direction_grad_2_ref = u2_direction_grad_2_ref_values[lane];
      const s_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
      const s_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
      const s_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = s_t(2)*u0_grad_2;
      const s_t residual_tmp1 = residual_tmp0*u1_grad_2;
      const s_t residual_tmp2 = u1_grad_1 + s_t(1);
      const s_t residual_tmp3 = s_t(2)*u0_grad_1;
      const s_t residual_tmp4 = residual_tmp2*residual_tmp3;
      const s_t residual_tmp5 = mu*(-residual_tmp1 - residual_tmp4);
      const s_t residual_tmp6 = u0_grad_2*u2_grad_1;
      const s_t residual_tmp7 = -residual_tmp6;
      const s_t residual_tmp8 = u2_grad_2 + s_t(1);
      const s_t residual_tmp9 = residual_tmp8*u0_grad_1;
      const s_t residual_tmp10 = -residual_tmp7 - residual_tmp9;
      const s_t residual_tmp11 = u1_grad_2*u2_grad_1;
      const s_t residual_tmp12 = s_t(2)*residual_tmp11;
      const s_t residual_tmp13 = -s_t(2)*residual_tmp2*residual_tmp8;
      const s_t residual_tmp14 = ((s_t(1) / s_t(2)))*lmbda;
      const s_t residual_tmp15 = residual_tmp14*(-residual_tmp12 - residual_tmp13);
      const s_t residual_tmp16 = u0_grad_0*u1_grad_1;
      const s_t residual_tmp17 = u0_grad_1*u1_grad_2;
      const s_t residual_tmp18 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp19 = u0_grad_2*u2_grad_0;
      const s_t residual_tmp20 = -residual_tmp11 + residual_tmp2 + u1_grad_1*u2_grad_2 + u2_grad_2;
      const s_t residual_tmp21 = -residual_tmp19 + u0_grad_0*u2_grad_2 + u0_grad_0;
      const s_t residual_tmp22 = residual_tmp16 - residual_tmp18;
      const s_t residual_tmp23 = -residual_tmp11*u0_grad_0 + residual_tmp16*u2_grad_2 + residual_tmp17*u2_grad_0 - residual_tmp18*u2_grad_2 - residual_tmp19*u1_grad_1 + residual_tmp20 + residual_tmp21 + residual_tmp22 + residual_tmp6*u1_grad_0;
      const s_t residual_tmp24 = pow_m1(residual_tmp23);
      const s_t residual_tmp25 = residual_tmp7 + u0_grad_1*u2_grad_2 + u0_grad_1;
      const s_t residual_tmp26 = residual_tmp20*u_dt_shift;
      const s_t residual_tmp27 = u1_grad_2*u_dt_shift + u1_old_grad_2;
      const s_t residual_tmp28 = residual_tmp27*u2_grad_1;
      const s_t residual_tmp29 = u1_grad_1*u_dt_shift + u1_old_grad_1;
      const s_t residual_tmp30 = residual_tmp29*residual_tmp8;
      const s_t residual_tmp31 = residual_tmp28 - residual_tmp30;
      const s_t residual_tmp32 = -residual_tmp26 - residual_tmp31;
      const s_t residual_tmp33 = -residual_tmp17 + u0_grad_2*u1_grad_1 + u0_grad_2;
      const s_t residual_tmp34 = u2_grad_1*u_dt_shift + u2_old_grad_1;
      const s_t residual_tmp35 = residual_tmp34*residual_tmp8;
      const s_t residual_tmp36 = u2_grad_2*u_dt_shift + u2_old_grad_2;
      const s_t residual_tmp37 = residual_tmp36*u2_grad_1;
      const s_t residual_tmp38 = u0_grad_2*u_dt_shift + u0_old_grad_2;
      const s_t residual_tmp39 = u0_grad_1*u_dt_shift + u0_old_grad_1;
      const s_t residual_tmp40 = residual_tmp38*u0_grad_1 - residual_tmp39*u0_grad_2;
      const s_t residual_tmp41 = residual_tmp35 - residual_tmp37 + residual_tmp40;
      const s_t residual_tmp42 = residual_tmp25*u_dt_shift;
      const s_t residual_tmp43 = residual_tmp36*u0_grad_1;
      const s_t residual_tmp44 = residual_tmp34*u0_grad_2;
      const s_t residual_tmp45 = residual_tmp42 + residual_tmp43 - residual_tmp44;
      const s_t residual_tmp46 = residual_tmp39*residual_tmp8;
      const s_t residual_tmp47 = residual_tmp38*u2_grad_1;
      const s_t residual_tmp48 = residual_tmp46 - residual_tmp47;
      const s_t residual_tmp49 = s_t(3)*eta_b;
      const s_t residual_tmp50 = residual_tmp49*(residual_tmp45 + residual_tmp48);
      const s_t residual_tmp51 = s_t(2)*eta_s;
      const s_t residual_tmp52 = residual_tmp50 + residual_tmp51*(s_t(2)*residual_tmp39*residual_tmp8 - residual_tmp45 - s_t(2)*residual_tmp47);
      const s_t residual_tmp53 = ((s_t(1) / s_t(3)))*residual_tmp20;
      const s_t residual_tmp54 = pow_m2(residual_tmp23);
      const s_t residual_tmp55 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp56 = u0_grad_2*u1_grad_0;
      const s_t residual_tmp57 = -residual_tmp56 + u0_grad_0*u1_grad_2 + u1_grad_2;
      const s_t residual_tmp58 = u1_grad_2*u2_grad_0;
      const s_t residual_tmp59 = -residual_tmp58;
      const s_t residual_tmp60 = residual_tmp59 + u1_grad_0*u2_grad_2 + u1_grad_0;
      const s_t residual_tmp61 = u1_grad_0*u2_grad_1;
      const s_t residual_tmp62 = -residual_tmp61 + u1_grad_1*u2_grad_0 + u2_grad_0;
      const s_t residual_tmp63 = residual_tmp2 + residual_tmp22 + u0_grad_0;
      const s_t residual_tmp64 = u2_grad_0*u_dt_shift + u2_old_grad_0;
      const s_t residual_tmp65 = -residual_tmp20*residual_tmp64 + residual_tmp33*residual_tmp55 + residual_tmp34*residual_tmp60 + residual_tmp36*residual_tmp62 - residual_tmp38*residual_tmp63 + residual_tmp39*residual_tmp57;
      const s_t residual_tmp66 = u0_grad_1*u2_grad_0;
      const s_t residual_tmp67 = -residual_tmp66 + u0_grad_0*u2_grad_1 + u2_grad_1;
      const s_t residual_tmp68 = residual_tmp21 + residual_tmp8;
      const s_t residual_tmp69 = u1_grad_0*u_dt_shift + u1_old_grad_0;
      const s_t residual_tmp70 = -residual_tmp20*residual_tmp69 + residual_tmp25*residual_tmp55 + residual_tmp27*residual_tmp62 + residual_tmp29*residual_tmp60 + residual_tmp38*residual_tmp67 - residual_tmp39*residual_tmp68;
      const s_t residual_tmp71 = residual_tmp25*residual_tmp69;
      const s_t residual_tmp72 = residual_tmp27*residual_tmp67;
      const s_t residual_tmp73 = residual_tmp33*residual_tmp64;
      const s_t residual_tmp74 = residual_tmp34*residual_tmp57;
      const s_t residual_tmp75 = residual_tmp29*residual_tmp68;
      const s_t residual_tmp76 = -residual_tmp75;
      const s_t residual_tmp77 = residual_tmp36*residual_tmp63;
      const s_t residual_tmp78 = -residual_tmp77;
      const s_t residual_tmp79 = residual_tmp71 + residual_tmp72 + residual_tmp73 + residual_tmp74 + residual_tmp76 + residual_tmp78;
      const s_t residual_tmp80 = residual_tmp20*residual_tmp55;
      const s_t residual_tmp81 = residual_tmp38*residual_tmp62 + residual_tmp39*residual_tmp60 - residual_tmp80;
      const s_t residual_tmp82 = residual_tmp49*(residual_tmp79 + residual_tmp81);
      const s_t residual_tmp83 = residual_tmp51*(s_t(2)*residual_tmp38*residual_tmp62 + s_t(2)*residual_tmp39*residual_tmp60 - residual_tmp79 - s_t(2)*residual_tmp80) + residual_tmp82;
      const s_t residual_tmp84 = -residual_tmp83;
      const s_t residual_tmp85 = residual_tmp54*(eta_s*(residual_tmp25*residual_tmp70 + residual_tmp33*residual_tmp65) + residual_tmp53*residual_tmp84);
      const s_t residual_tmp86 = residual_tmp3*u2_grad_1;
      const s_t residual_tmp87 = residual_tmp0*residual_tmp8;
      const s_t residual_tmp88 = mu*(-residual_tmp86 - residual_tmp87);
      const s_t residual_tmp89 = residual_tmp2*u0_grad_2;
      const s_t residual_tmp90 = residual_tmp17 - residual_tmp89;
      const s_t residual_tmp91 = residual_tmp34*u1_grad_2;
      const s_t residual_tmp92 = residual_tmp2*residual_tmp36;
      const s_t residual_tmp93 = residual_tmp91 - residual_tmp92;
      const s_t residual_tmp94 = -residual_tmp26 - residual_tmp93;
      const s_t residual_tmp95 = -residual_tmp2*residual_tmp27 + residual_tmp29*u1_grad_2;
      const s_t residual_tmp96 = -residual_tmp40 - residual_tmp95;
      const s_t residual_tmp97 = residual_tmp33*u_dt_shift;
      const s_t residual_tmp98 = residual_tmp29*u0_grad_2;
      const s_t residual_tmp99 = residual_tmp27*u0_grad_1;
      const s_t residual_tmp100 = residual_tmp97 + residual_tmp98 - residual_tmp99;
      const s_t residual_tmp101 = residual_tmp2*residual_tmp38;
      const s_t residual_tmp102 = residual_tmp39*u1_grad_2;
      const s_t residual_tmp103 = residual_tmp101 - residual_tmp102;
      const s_t residual_tmp104 = residual_tmp49*(residual_tmp100 + residual_tmp103);
      const s_t residual_tmp105 = residual_tmp104 + residual_tmp51*(-residual_tmp100 - s_t(2)*residual_tmp102 + s_t(2)*residual_tmp2*residual_tmp38);
      const s_t residual_tmp106 = s_t(2)*pow_2(u1_grad_2);
      const s_t residual_tmp107 = s_t(2)*pow_2(residual_tmp8) + s_t(2);
      const s_t residual_tmp108 = residual_tmp106 + residual_tmp107;
      const s_t residual_tmp109 = s_t(2)*pow_2(u2_grad_1);
      const s_t residual_tmp110 = s_t(2)*pow_2(residual_tmp2);
      const s_t residual_tmp111 = residual_tmp109 + residual_tmp110;
      const s_t residual_tmp112 = -residual_tmp11 + residual_tmp2*residual_tmp8;
      const s_t residual_tmp113 = -residual_tmp101 + residual_tmp102;
      const s_t residual_tmp114 = residual_tmp113 + residual_tmp97;
      const s_t residual_tmp115 = -residual_tmp46 + residual_tmp47;
      const s_t residual_tmp116 = residual_tmp115 + residual_tmp42;
      const s_t residual_tmp117 = residual_tmp26 - residual_tmp91 + residual_tmp92;
      const s_t residual_tmp118 = -residual_tmp28 + residual_tmp30;
      const s_t residual_tmp119 = residual_tmp49*(-residual_tmp117 - residual_tmp118);
      const s_t residual_tmp120 = residual_tmp119 + residual_tmp51*(-s_t(2)*residual_tmp26 - residual_tmp31 - residual_tmp93);
      const s_t residual_tmp121 = -residual_tmp20;
      const s_t residual_tmp122 = s_t(2)*u2_grad_0;
      const s_t residual_tmp123 = residual_tmp122*u2_grad_1;
      const s_t residual_tmp124 = s_t(2)*u1_grad_0;
      const s_t residual_tmp125 = residual_tmp124*residual_tmp2;
      const s_t residual_tmp126 = residual_tmp123 + residual_tmp125;
      const s_t residual_tmp127 = residual_tmp8*u1_grad_0;
      const s_t residual_tmp128 = -residual_tmp127 - residual_tmp59;
      const s_t residual_tmp129 = residual_tmp60*u_dt_shift;
      const s_t residual_tmp130 = residual_tmp36*u1_grad_0;
      const s_t residual_tmp131 = residual_tmp64*u1_grad_2;
      const s_t residual_tmp132 = residual_tmp129 + residual_tmp130 - residual_tmp131;
      const s_t residual_tmp133 = residual_tmp69*residual_tmp8;
      const s_t residual_tmp134 = residual_tmp27*u2_grad_0;
      const s_t residual_tmp135 = residual_tmp133 - residual_tmp134;
      const s_t residual_tmp136 = residual_tmp49*(residual_tmp132 + residual_tmp135);
      const s_t residual_tmp137 = -residual_tmp133 + residual_tmp134;
      const s_t residual_tmp138 = -residual_tmp130 + residual_tmp131;
      const s_t residual_tmp139 = residual_tmp136 + residual_tmp51*(s_t(2)*residual_tmp129 + residual_tmp137 + residual_tmp138);
      const s_t residual_tmp140 = residual_tmp57*u_dt_shift;
      const s_t residual_tmp141 = residual_tmp38*u1_grad_0;
      const s_t residual_tmp142 = residual_tmp55*u1_grad_2;
      const s_t residual_tmp143 = residual_tmp141 - residual_tmp142;
      const s_t residual_tmp144 = residual_tmp140 + residual_tmp143;
      const s_t residual_tmp145 = residual_tmp68*u_dt_shift;
      const s_t residual_tmp146 = residual_tmp38*u2_grad_0;
      const s_t residual_tmp147 = residual_tmp55*residual_tmp8;
      const s_t residual_tmp148 = residual_tmp146 - residual_tmp147;
      const s_t residual_tmp149 = -residual_tmp145 - residual_tmp148;
      const s_t residual_tmp150 = residual_tmp65*u1_grad_2;
      const s_t residual_tmp151 = -residual_tmp150;
      const s_t residual_tmp152 = residual_tmp70*residual_tmp8;
      const s_t residual_tmp153 = residual_tmp124*u1_grad_2;
      const s_t residual_tmp154 = residual_tmp122*residual_tmp8;
      const s_t residual_tmp155 = residual_tmp153 + residual_tmp154;
      const s_t residual_tmp156 = residual_tmp2*u2_grad_0;
      const s_t residual_tmp157 = -residual_tmp156 + residual_tmp61;
      const s_t residual_tmp158 = residual_tmp62*u_dt_shift;
      const s_t residual_tmp159 = residual_tmp2*residual_tmp64;
      const s_t residual_tmp160 = residual_tmp34*u1_grad_0;
      const s_t residual_tmp161 = residual_tmp158 + residual_tmp159 - residual_tmp160;
      const s_t residual_tmp162 = residual_tmp29*u2_grad_0;
      const s_t residual_tmp163 = residual_tmp69*u2_grad_1;
      const s_t residual_tmp164 = residual_tmp162 - residual_tmp163;
      const s_t residual_tmp165 = residual_tmp49*(residual_tmp161 + residual_tmp164);
      const s_t residual_tmp166 = -residual_tmp162 + residual_tmp163;
      const s_t residual_tmp167 = -residual_tmp159 + residual_tmp160;
      const s_t residual_tmp168 = residual_tmp165 + residual_tmp51*(s_t(2)*residual_tmp158 + residual_tmp166 + residual_tmp167);
      const s_t residual_tmp169 = residual_tmp67*u_dt_shift;
      const s_t residual_tmp170 = residual_tmp39*u2_grad_0;
      const s_t residual_tmp171 = residual_tmp55*u2_grad_1;
      const s_t residual_tmp172 = residual_tmp170 - residual_tmp171;
      const s_t residual_tmp173 = residual_tmp169 + residual_tmp172;
      const s_t residual_tmp174 = residual_tmp63*u_dt_shift;
      const s_t residual_tmp175 = residual_tmp39*u1_grad_0;
      const s_t residual_tmp176 = residual_tmp2*residual_tmp55;
      const s_t residual_tmp177 = residual_tmp175 - residual_tmp176;
      const s_t residual_tmp178 = -residual_tmp174 - residual_tmp177;
      const s_t residual_tmp179 = residual_tmp70*u2_grad_1;
      const s_t residual_tmp180 = -residual_tmp179;
      const s_t residual_tmp181 = residual_tmp2*residual_tmp65;
      const s_t residual_tmp182 = u0_grad_0 + s_t(1);
      const s_t residual_tmp183 = residual_tmp182*u1_grad_2;
      const s_t residual_tmp184 = s_t(6)*u2_grad_1;
      const s_t residual_tmp185 = s_t(2)*residual_tmp56;
      const s_t residual_tmp186 = residual_tmp184 - residual_tmp185;
      const s_t residual_tmp187 = residual_tmp182*u2_grad_1;
      const s_t residual_tmp188 = -residual_tmp187 + residual_tmp66;
      const s_t residual_tmp189 = lmbda*(-residual_tmp11*residual_tmp182 - residual_tmp18*residual_tmp8 + residual_tmp182*residual_tmp2*residual_tmp8 - residual_tmp19*residual_tmp2 + u0_grad_1*u1_grad_2*u2_grad_0 + u0_grad_2*u1_grad_0*u2_grad_1 + s_t(-1));
      const s_t residual_tmp190 = residual_tmp189*u2_grad_1;
      const s_t residual_tmp191 = -residual_tmp190;
      const s_t residual_tmp192 = residual_tmp182*residual_tmp34;
      const s_t residual_tmp193 = residual_tmp64*u0_grad_1;
      const s_t residual_tmp194 = residual_tmp169 + residual_tmp192 - residual_tmp193;
      const s_t residual_tmp195 = -residual_tmp170 + residual_tmp171;
      const s_t residual_tmp196 = residual_tmp49*(residual_tmp194 + residual_tmp195);
      const s_t residual_tmp197 = residual_tmp196 + residual_tmp51*(-s_t(2)*residual_tmp170 - residual_tmp194 + s_t(2)*residual_tmp55*u2_grad_1);
      const s_t residual_tmp198 = residual_tmp158 + residual_tmp166;
      const s_t residual_tmp199 = residual_tmp34*u2_grad_0;
      const s_t residual_tmp200 = residual_tmp64*u2_grad_1;
      const s_t residual_tmp201 = -residual_tmp182*residual_tmp39 + residual_tmp55*u0_grad_1;
      const s_t residual_tmp202 = -residual_tmp199 + residual_tmp200 - residual_tmp201;
      const s_t residual_tmp203 = residual_tmp65*u0_grad_1;
      const s_t residual_tmp204 = ((s_t(1) / s_t(3)))*residual_tmp84;
      const s_t residual_tmp205 = s_t(6)*u1_grad_2;
      const s_t residual_tmp206 = s_t(2)*residual_tmp66;
      const s_t residual_tmp207 = residual_tmp205 - residual_tmp206;
      const s_t residual_tmp208 = -residual_tmp183 + residual_tmp56;
      const s_t residual_tmp209 = residual_tmp189*u1_grad_2;
      const s_t residual_tmp210 = -residual_tmp209;
      const s_t residual_tmp211 = residual_tmp182*residual_tmp27;
      const s_t residual_tmp212 = residual_tmp69*u0_grad_2;
      const s_t residual_tmp213 = residual_tmp140 + residual_tmp211 - residual_tmp212;
      const s_t residual_tmp214 = -residual_tmp141 + residual_tmp142;
      const s_t residual_tmp215 = residual_tmp49*(residual_tmp213 + residual_tmp214);
      const s_t residual_tmp216 = residual_tmp215 + residual_tmp51*(-s_t(2)*residual_tmp141 - residual_tmp213 + s_t(2)*residual_tmp55*u1_grad_2);
      const s_t residual_tmp217 = residual_tmp129 + residual_tmp138;
      const s_t residual_tmp218 = -residual_tmp182*residual_tmp38 + residual_tmp55*u0_grad_2;
      const s_t residual_tmp219 = residual_tmp27*u1_grad_0 - residual_tmp69*u1_grad_2;
      const s_t residual_tmp220 = -residual_tmp218 - residual_tmp219;
      const s_t residual_tmp221 = residual_tmp70*u0_grad_2;
      const s_t residual_tmp222 = s_t(2)*u1_grad_1 + s_t(2);
      const s_t residual_tmp223 = s_t(2)*residual_tmp18;
      const s_t residual_tmp224 = s_t(6)*u2_grad_2 + s_t(6);
      const s_t residual_tmp225 = residual_tmp223 + residual_tmp224;
      const s_t residual_tmp226 = residual_tmp182*residual_tmp8 - residual_tmp19;
      const s_t residual_tmp227 = s_t(2)*u2_grad_2 + s_t(2);
      const s_t residual_tmp228 = ((s_t(1) / s_t(2)))*residual_tmp189;
      const s_t residual_tmp229 = residual_tmp227*residual_tmp228;
      const s_t residual_tmp230 = -residual_tmp68;
      const s_t residual_tmp231 = residual_tmp182*residual_tmp36;
      const s_t residual_tmp232 = residual_tmp64*u0_grad_2;
      const s_t residual_tmp233 = residual_tmp145 + residual_tmp231 - residual_tmp232;
      const s_t residual_tmp234 = -residual_tmp146 + residual_tmp147;
      const s_t residual_tmp235 = residual_tmp49*(-residual_tmp233 - residual_tmp234);
      const s_t residual_tmp236 = residual_tmp235 + residual_tmp51*(s_t(2)*residual_tmp146 - s_t(2)*residual_tmp147 + residual_tmp233);
      const s_t residual_tmp237 = residual_tmp129 + residual_tmp137;
      const s_t residual_tmp238 = residual_tmp36*u2_grad_0;
      const s_t residual_tmp239 = residual_tmp64*residual_tmp8;
      const s_t residual_tmp240 = residual_tmp218 + residual_tmp238 - residual_tmp239;
      const s_t residual_tmp241 = residual_tmp65*u0_grad_2;
      const s_t residual_tmp242 = s_t(2)*residual_tmp19;
      const s_t residual_tmp243 = s_t(6)*u1_grad_1 + s_t(6);
      const s_t residual_tmp244 = residual_tmp242 + residual_tmp243;
      const s_t residual_tmp245 = -residual_tmp18 + residual_tmp182*residual_tmp2;
      const s_t residual_tmp246 = residual_tmp222*residual_tmp228;
      const s_t residual_tmp247 = -residual_tmp63;
      const s_t residual_tmp248 = residual_tmp182*residual_tmp29;
      const s_t residual_tmp249 = residual_tmp69*u0_grad_1;
      const s_t residual_tmp250 = residual_tmp174 + residual_tmp248 - residual_tmp249;
      const s_t residual_tmp251 = -residual_tmp175 + residual_tmp176;
      const s_t residual_tmp252 = residual_tmp49*(-residual_tmp250 - residual_tmp251);
      const s_t residual_tmp253 = residual_tmp252 + residual_tmp51*(s_t(2)*residual_tmp175 - s_t(2)*residual_tmp176 + residual_tmp250);
      const s_t residual_tmp254 = residual_tmp158 + residual_tmp167;
      const s_t residual_tmp255 = -residual_tmp2*residual_tmp69 + residual_tmp29*u1_grad_0;
      const s_t residual_tmp256 = residual_tmp201 + residual_tmp255;
      const s_t residual_tmp257 = residual_tmp70*u0_grad_1;
      const s_t residual_tmp258 = residual_tmp122*residual_tmp182;
      const s_t residual_tmp259 = mu*(-residual_tmp258 - residual_tmp87);
      const s_t residual_tmp260 = -s_t(2)*residual_tmp58;
      const s_t residual_tmp261 = s_t(2)*residual_tmp127;
      const s_t residual_tmp262 = residual_tmp14*(-residual_tmp260 - residual_tmp261);
      const s_t residual_tmp263 = residual_tmp54*(-eta_s*(-residual_tmp57*residual_tmp65 + residual_tmp68*residual_tmp70) + ((s_t(1) / s_t(3)))*residual_tmp60*residual_tmp83);
      const s_t residual_tmp264 = s_t(2)*pow_2(u1_grad_0);
      const s_t residual_tmp265 = s_t(2)*pow_2(u2_grad_0);
      const s_t residual_tmp266 = residual_tmp264 + residual_tmp265;
      const s_t residual_tmp267 = residual_tmp124*residual_tmp182;
      const s_t residual_tmp268 = mu*(-residual_tmp1 - residual_tmp267);
      const s_t residual_tmp269 = s_t(2)*u1_grad_2;
      const s_t residual_tmp270 = residual_tmp2*residual_tmp269;
      const s_t residual_tmp271 = s_t(2)*u2_grad_1;
      const s_t residual_tmp272 = residual_tmp271*residual_tmp8;
      const s_t residual_tmp273 = mu*(-residual_tmp270 - residual_tmp272);
      const s_t residual_tmp274 = residual_tmp65*u1_grad_0;
      const s_t residual_tmp275 = residual_tmp70*u2_grad_0;
      const s_t residual_tmp276 = -residual_tmp275;
      const s_t residual_tmp277 = residual_tmp274 + residual_tmp276;
      const s_t residual_tmp278 = s_t(2)*u0_grad_0 + s_t(2);
      const s_t residual_tmp279 = s_t(4)*residual_tmp182;
      const s_t residual_tmp280 = -residual_tmp70*residual_tmp8;
      const s_t residual_tmp281 = residual_tmp182*residual_tmp65;
      const s_t residual_tmp282 = ((s_t(1) / s_t(3)))*residual_tmp83;
      const s_t residual_tmp283 = residual_tmp282*u2_grad_0;
      const s_t residual_tmp284 = s_t(6)*u2_grad_0;
      const s_t residual_tmp285 = s_t(2)*residual_tmp89;
      const s_t residual_tmp286 = residual_tmp189*u2_grad_0;
      const s_t residual_tmp287 = mu*(-residual_tmp284 - residual_tmp285 + s_t(4)*u0_grad_1*u1_grad_2) + residual_tmp286;
      const s_t residual_tmp288 = s_t(2)*residual_tmp187;
      const s_t residual_tmp289 = mu*(-residual_tmp205 - residual_tmp288 + s_t(4)*u0_grad_1*u2_grad_0) + residual_tmp209;
      const s_t residual_tmp290 = ((s_t(1) / s_t(3)))*residual_tmp60;
      const s_t residual_tmp291 = -s_t(2)*residual_tmp182*residual_tmp2;
      const s_t residual_tmp292 = mu*(s_t(4)*residual_tmp18 + residual_tmp224 + residual_tmp291) - residual_tmp227*residual_tmp228;
      const s_t residual_tmp293 = -s_t(2)*residual_tmp6;
      const s_t residual_tmp294 = s_t(6)*u1_grad_0;
      const s_t residual_tmp295 = residual_tmp293 + residual_tmp294;
      const s_t residual_tmp296 = residual_tmp189*u1_grad_0;
      const s_t residual_tmp297 = -residual_tmp296;
      const s_t residual_tmp298 = residual_tmp182*residual_tmp70;
      const s_t residual_tmp299 = residual_tmp282*u1_grad_0;
      const s_t residual_tmp300 = mu*(-residual_tmp267 - residual_tmp4);
      const s_t residual_tmp301 = s_t(2)*residual_tmp61;
      const s_t residual_tmp302 = s_t(2)*residual_tmp156;
      const s_t residual_tmp303 = residual_tmp14*(residual_tmp301 - residual_tmp302);
      const s_t residual_tmp304 = residual_tmp54*(-eta_s*(residual_tmp63*residual_tmp65 - residual_tmp67*residual_tmp70) + ((s_t(1) / s_t(3)))*residual_tmp62*residual_tmp83);
      const s_t residual_tmp305 = mu*(-residual_tmp258 - residual_tmp86);
      const s_t residual_tmp306 = -residual_tmp2*residual_tmp65;
      const s_t residual_tmp307 = s_t(2)*residual_tmp9;
      const s_t residual_tmp308 = mu*(-residual_tmp294 - residual_tmp307 + s_t(4)*u0_grad_2*u2_grad_1) + residual_tmp296;
      const s_t residual_tmp309 = s_t(2)*residual_tmp183;
      const s_t residual_tmp310 = mu*(-residual_tmp184 - residual_tmp309 + s_t(4)*u0_grad_2*u1_grad_0) + residual_tmp190;
      const s_t residual_tmp311 = ((s_t(1) / s_t(3)))*residual_tmp62;
      const s_t residual_tmp312 = -s_t(2)*residual_tmp182*residual_tmp8;
      const s_t residual_tmp313 = mu*(s_t(4)*residual_tmp19 + residual_tmp243 + residual_tmp312) - residual_tmp222*residual_tmp228;
      const s_t residual_tmp314 = s_t(2)*residual_tmp17;
      const s_t residual_tmp315 = residual_tmp284 - residual_tmp314;
      const s_t residual_tmp316 = -residual_tmp286;
      const s_t residual_tmp317 = residual_tmp269*residual_tmp8;
      const s_t residual_tmp318 = residual_tmp2*residual_tmp271;
      const s_t residual_tmp319 = mu*(-residual_tmp317 - residual_tmp318);
      const s_t residual_tmp320 = residual_tmp14*(-residual_tmp293 - residual_tmp307);
      const s_t residual_tmp321 = -residual_tmp43 + residual_tmp44;
      const s_t residual_tmp322 = residual_tmp321 + residual_tmp42;
      const s_t residual_tmp323 = residual_tmp104 + residual_tmp51*(-residual_tmp103 + s_t(2)*residual_tmp29*u0_grad_2 - residual_tmp97 - s_t(2)*residual_tmp99);
      const s_t residual_tmp324 = residual_tmp25*residual_tmp64 - residual_tmp27*residual_tmp63 + residual_tmp29*residual_tmp57 + residual_tmp33*residual_tmp69 - residual_tmp34*residual_tmp68 + residual_tmp36*residual_tmp67;
      const s_t residual_tmp325 = residual_tmp51*(s_t(2)*residual_tmp25*residual_tmp69 + s_t(2)*residual_tmp27*residual_tmp67 - residual_tmp73 - residual_tmp74 - s_t(2)*residual_tmp75 - residual_tmp78 - residual_tmp81) + residual_tmp82;
      const s_t residual_tmp326 = residual_tmp54*(-eta_s*(residual_tmp20*residual_tmp70 - residual_tmp324*residual_tmp33) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp325);
      const s_t residual_tmp327 = s_t(2)*pow_2(u0_grad_2);
      const s_t residual_tmp328 = residual_tmp107 + residual_tmp327;
      const s_t residual_tmp329 = s_t(2)*pow_2(u0_grad_1);
      const s_t residual_tmp330 = residual_tmp109 + residual_tmp329;
      const s_t residual_tmp331 = -residual_tmp98 + residual_tmp99;
      const s_t residual_tmp332 = residual_tmp331 + residual_tmp97;
      const s_t residual_tmp333 = residual_tmp50 + residual_tmp51*(residual_tmp115 + residual_tmp321 + s_t(2)*residual_tmp42);
      const s_t residual_tmp334 = -residual_tmp35 + residual_tmp37 + residual_tmp95;
      const s_t residual_tmp335 = residual_tmp119 + residual_tmp51*(residual_tmp117 + s_t(2)*residual_tmp28 - s_t(2)*residual_tmp30);
      const s_t residual_tmp336 = residual_tmp0*residual_tmp182;
      const s_t residual_tmp337 = mu*(-residual_tmp154 - residual_tmp336);
      const s_t residual_tmp338 = -residual_tmp192 + residual_tmp193;
      const s_t residual_tmp339 = residual_tmp196 + residual_tmp51*(s_t(2)*residual_tmp169 + residual_tmp172 + residual_tmp338);
      const s_t residual_tmp340 = -residual_tmp248 + residual_tmp249;
      const s_t residual_tmp341 = -residual_tmp174 - residual_tmp340;
      const s_t residual_tmp342 = residual_tmp324*u0_grad_1;
      const s_t residual_tmp343 = residual_tmp180 + residual_tmp342;
      const s_t residual_tmp344 = s_t(4)*residual_tmp2;
      const s_t residual_tmp345 = residual_tmp182*residual_tmp3;
      const s_t residual_tmp346 = residual_tmp123 + residual_tmp345;
      const s_t residual_tmp347 = -residual_tmp231 + residual_tmp232;
      const s_t residual_tmp348 = residual_tmp235 + residual_tmp51*(-s_t(2)*residual_tmp145 - residual_tmp148 - residual_tmp347);
      const s_t residual_tmp349 = -residual_tmp211 + residual_tmp212;
      const s_t residual_tmp350 = residual_tmp140 + residual_tmp349;
      const s_t residual_tmp351 = residual_tmp324*u0_grad_2;
      const s_t residual_tmp352 = residual_tmp165 + residual_tmp51*(-residual_tmp161 - s_t(2)*residual_tmp163 + s_t(2)*residual_tmp29*u2_grad_0);
      const s_t residual_tmp353 = residual_tmp199 - residual_tmp200 - residual_tmp255;
      const s_t residual_tmp354 = residual_tmp2*residual_tmp324;
      const s_t residual_tmp355 = ((s_t(1) / s_t(3)))*residual_tmp325;
      const s_t residual_tmp356 = residual_tmp355*u2_grad_1;
      const s_t residual_tmp357 = residual_tmp215 + residual_tmp51*(-residual_tmp140 + s_t(2)*residual_tmp182*residual_tmp27 - s_t(2)*residual_tmp212 - residual_tmp214);
      const s_t residual_tmp358 = -residual_tmp145 - residual_tmp347;
      const s_t residual_tmp359 = residual_tmp70*u1_grad_2;
      const s_t residual_tmp360 = s_t(6)*u0_grad_2;
      const s_t residual_tmp361 = residual_tmp189*u0_grad_2;
      const s_t residual_tmp362 = mu*(-residual_tmp302 - residual_tmp360 + s_t(4)*u1_grad_0*u2_grad_1) + residual_tmp361;
      const s_t residual_tmp363 = residual_tmp136 + residual_tmp51*(-residual_tmp132 - s_t(2)*residual_tmp134 + s_t(2)*residual_tmp69*residual_tmp8);
      const s_t residual_tmp364 = ((s_t(1) / s_t(3)))*residual_tmp25;
      const s_t residual_tmp365 = residual_tmp219 - residual_tmp238 + residual_tmp239;
      const s_t residual_tmp366 = residual_tmp324*u1_grad_2;
      const s_t residual_tmp367 = s_t(6)*u0_grad_1;
      const s_t residual_tmp368 = residual_tmp260 + residual_tmp367;
      const s_t residual_tmp369 = residual_tmp189*u0_grad_1;
      const s_t residual_tmp370 = -residual_tmp369;
      const s_t residual_tmp371 = residual_tmp252 + residual_tmp51*(residual_tmp174 - s_t(2)*residual_tmp248 + s_t(2)*residual_tmp249 + residual_tmp251);
      const s_t residual_tmp372 = residual_tmp169 + residual_tmp338;
      const s_t residual_tmp373 = residual_tmp2*residual_tmp70;
      const s_t residual_tmp374 = residual_tmp355*u0_grad_1;
      const s_t residual_tmp375 = residual_tmp14*(-residual_tmp242 - residual_tmp312);
      const s_t residual_tmp376 = ((s_t(1) / s_t(3)))*residual_tmp68;
      const s_t residual_tmp377 = -(s_t(1) / s_t(3))*residual_tmp325;
      const s_t residual_tmp378 = residual_tmp54*(eta_s*(residual_tmp324*residual_tmp57 + residual_tmp60*residual_tmp70) + residual_tmp377*residual_tmp68);
      const s_t residual_tmp379 = residual_tmp124*u2_grad_0;
      const s_t residual_tmp380 = mu*(-residual_tmp317 - residual_tmp379);
      const s_t residual_tmp381 = s_t(2)*pow_2(residual_tmp182);
      const s_t residual_tmp382 = residual_tmp265 + residual_tmp381;
      const s_t residual_tmp383 = residual_tmp3*u0_grad_2;
      const s_t residual_tmp384 = residual_tmp272 + residual_tmp383;
      const s_t residual_tmp385 = residual_tmp182*residual_tmp324;
      const s_t residual_tmp386 = residual_tmp324*u1_grad_0;
      const s_t residual_tmp387 = -residual_tmp301 + residual_tmp360;
      const s_t residual_tmp388 = -residual_tmp361;
      const s_t residual_tmp389 = s_t(6)*u0_grad_0 + s_t(6);
      const s_t residual_tmp390 = residual_tmp12 + residual_tmp389;
      const s_t residual_tmp391 = residual_tmp228*residual_tmp278;
      const s_t residual_tmp392 = residual_tmp70*u1_grad_0;
      const s_t residual_tmp393 = residual_tmp14*(residual_tmp206 - residual_tmp288);
      const s_t residual_tmp394 = residual_tmp54*(-eta_s*(residual_tmp324*residual_tmp63 - residual_tmp62*residual_tmp70) + ((s_t(1) / s_t(3)))*residual_tmp325*residual_tmp67);
      const s_t residual_tmp395 = mu*(-residual_tmp318 - residual_tmp379);
      const s_t residual_tmp396 = -residual_tmp182*residual_tmp324;
      const s_t residual_tmp397 = mu*(-residual_tmp261 - residual_tmp367 + s_t(4)*u1_grad_2*u2_grad_0) + residual_tmp369;
      const s_t residual_tmp398 = ((s_t(1) / s_t(3)))*residual_tmp67;
      const s_t residual_tmp399 = mu*(s_t(4)*residual_tmp11 + residual_tmp13 + residual_tmp389) - residual_tmp228*residual_tmp278;
      const s_t residual_tmp400 = residual_tmp14*(-residual_tmp285 + residual_tmp314);
      const s_t residual_tmp401 = residual_tmp50 + residual_tmp51*(s_t(2)*residual_tmp36*u0_grad_1 - residual_tmp42 - s_t(2)*residual_tmp44 - residual_tmp48);
      const s_t residual_tmp402 = residual_tmp51*(s_t(2)*residual_tmp33*residual_tmp64 + s_t(2)*residual_tmp34*residual_tmp57 - residual_tmp71 - residual_tmp72 - residual_tmp76 - s_t(2)*residual_tmp77 - residual_tmp81) + residual_tmp82;
      const s_t residual_tmp403 = residual_tmp54*(-eta_s*(residual_tmp20*residual_tmp65 - residual_tmp25*residual_tmp324) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp402);
      const s_t residual_tmp404 = residual_tmp106 + residual_tmp327 + s_t(2);
      const s_t residual_tmp405 = residual_tmp110 + residual_tmp329;
      const s_t residual_tmp406 = residual_tmp104 + residual_tmp51*(residual_tmp113 + residual_tmp331 + s_t(2)*residual_tmp97);
      const s_t residual_tmp407 = residual_tmp119 + residual_tmp51*(residual_tmp118 + residual_tmp26 + s_t(2)*residual_tmp91 - s_t(2)*residual_tmp92);
      const s_t residual_tmp408 = mu*(-residual_tmp125 - residual_tmp345);
      const s_t residual_tmp409 = residual_tmp215 + residual_tmp51*(s_t(2)*residual_tmp140 + residual_tmp143 + residual_tmp349);
      const s_t residual_tmp410 = residual_tmp151 + residual_tmp351;
      const s_t residual_tmp411 = s_t(4)*residual_tmp8;
      const s_t residual_tmp412 = residual_tmp153 + residual_tmp336;
      const s_t residual_tmp413 = residual_tmp252 + residual_tmp51*(-s_t(2)*residual_tmp174 - residual_tmp177 - residual_tmp340);
      const s_t residual_tmp414 = residual_tmp136 + residual_tmp51*(-residual_tmp129 - s_t(2)*residual_tmp131 - residual_tmp135 + s_t(2)*residual_tmp36*u1_grad_0);
      const s_t residual_tmp415 = residual_tmp324*residual_tmp8;
      const s_t residual_tmp416 = ((s_t(1) / s_t(3)))*residual_tmp402;
      const s_t residual_tmp417 = residual_tmp416*u1_grad_2;
      const s_t residual_tmp418 = residual_tmp196 + residual_tmp51*(-residual_tmp169 + s_t(2)*residual_tmp182*residual_tmp34 - s_t(2)*residual_tmp193 - residual_tmp195);
      const s_t residual_tmp419 = residual_tmp65*u2_grad_1;
      const s_t residual_tmp420 = residual_tmp165 + residual_tmp51*(-residual_tmp158 - s_t(2)*residual_tmp160 - residual_tmp164 + s_t(2)*residual_tmp2*residual_tmp64);
      const s_t residual_tmp421 = ((s_t(1) / s_t(3)))*residual_tmp33;
      const s_t residual_tmp422 = residual_tmp324*u2_grad_1;
      const s_t residual_tmp423 = residual_tmp235 + residual_tmp51*(residual_tmp145 - s_t(2)*residual_tmp231 + s_t(2)*residual_tmp232 + residual_tmp234);
      const s_t residual_tmp424 = residual_tmp65*residual_tmp8;
      const s_t residual_tmp425 = residual_tmp416*u0_grad_2;
      const s_t residual_tmp426 = residual_tmp14*(residual_tmp185 - residual_tmp309);
      const s_t residual_tmp427 = residual_tmp54*(-eta_s*(residual_tmp324*residual_tmp68 - residual_tmp60*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp402*residual_tmp57);
      const s_t residual_tmp428 = residual_tmp264 + residual_tmp381;
      const s_t residual_tmp429 = residual_tmp270 + residual_tmp383;
      const s_t residual_tmp430 = residual_tmp324*u2_grad_0;
      const s_t residual_tmp431 = ((s_t(1) / s_t(3)))*residual_tmp57;
      const s_t residual_tmp432 = residual_tmp65*u2_grad_0;
      const s_t residual_tmp433 = residual_tmp14*(-residual_tmp223 - residual_tmp291);
      const s_t residual_tmp434 = ((s_t(1) / s_t(3)))*residual_tmp63;
      const s_t residual_tmp435 = -(s_t(1) / s_t(3))*residual_tmp402;
      const s_t residual_tmp436 = residual_tmp54*(eta_s*(residual_tmp324*residual_tmp67 + residual_tmp62*residual_tmp65) + residual_tmp435*residual_tmp63);
      const s_t grad_coeff0_0 = u0_direction_grad_0*(mu*(residual_tmp108 + residual_tmp111) + residual_tmp112*residual_tmp15 + residual_tmp121*residual_tmp85 + residual_tmp24*(eta_s*(residual_tmp114*residual_tmp33 + residual_tmp116*residual_tmp25) - residual_tmp120*residual_tmp53)) + u0_direction_grad_1*(-mu*residual_tmp126 + residual_tmp128*residual_tmp15 + residual_tmp24*(eta_s*(residual_tmp144*residual_tmp33 + residual_tmp149*residual_tmp25 + residual_tmp151 + residual_tmp152) - residual_tmp139*residual_tmp53) + residual_tmp60*residual_tmp85) + u0_direction_grad_2*(-mu*residual_tmp155 + residual_tmp15*residual_tmp157 + residual_tmp24*(eta_s*(residual_tmp173*residual_tmp25 + residual_tmp178*residual_tmp33 + residual_tmp180 + residual_tmp181) - residual_tmp168*residual_tmp53) + residual_tmp62*residual_tmp85) + u1_direction_grad_0*(residual_tmp10*residual_tmp15 + residual_tmp24*(eta_s*(residual_tmp25*residual_tmp32 + residual_tmp33*residual_tmp41) - residual_tmp52*residual_tmp53) + residual_tmp25*residual_tmp85 + residual_tmp5) + u1_direction_grad_1*(mu*(s_t(2)*residual_tmp182*residual_tmp222 - residual_tmp225) + residual_tmp15*residual_tmp226 + residual_tmp229 + residual_tmp230*residual_tmp85 + residual_tmp24*(eta_s*(residual_tmp237*residual_tmp25 + residual_tmp240*residual_tmp33 + residual_tmp241) + residual_tmp204*residual_tmp8 - residual_tmp236*residual_tmp53)) + u1_direction_grad_2*(mu*(s_t(4)*residual_tmp183 + residual_tmp186) + residual_tmp15*residual_tmp188 + residual_tmp191 + residual_tmp24*(eta_s*(residual_tmp198*residual_tmp25 + residual_tmp202*residual_tmp33 - residual_tmp203) - residual_tmp197*residual_tmp53 - residual_tmp204*u2_grad_1) + residual_tmp67*residual_tmp85) + u2_direction_grad_0*(residual_tmp15*residual_tmp90 + residual_tmp24*(eta_s*(residual_tmp25*residual_tmp96 + residual_tmp33*residual_tmp94) - residual_tmp105*residual_tmp53) + residual_tmp33*residual_tmp85 + residual_tmp88) + u2_direction_grad_1*(mu*(s_t(4)*residual_tmp187 + residual_tmp207) + residual_tmp15*residual_tmp208 + residual_tmp210 + residual_tmp24*(eta_s*(residual_tmp217*residual_tmp33 + residual_tmp220*residual_tmp25 - residual_tmp221) - residual_tmp204*u1_grad_2 - residual_tmp216*residual_tmp53) + residual_tmp57*residual_tmp85) + u2_direction_grad_2*(mu*(s_t(2)*residual_tmp182*residual_tmp227 - residual_tmp244) + residual_tmp15*residual_tmp245 + residual_tmp24*(eta_s*(residual_tmp25*residual_tmp256 + residual_tmp254*residual_tmp33 + residual_tmp257) + residual_tmp2*residual_tmp204 - residual_tmp253*residual_tmp53) + residual_tmp246 + residual_tmp247*residual_tmp85);
      const s_t grad_coeff0_1 = u0_direction_grad_0*(mu*(-residual_tmp126 + s_t(2)*residual_tmp278*u0_grad_1 - residual_tmp279*u0_grad_1) + residual_tmp112*residual_tmp262 + residual_tmp121*residual_tmp263 + residual_tmp24*(-eta_s*(-residual_tmp114*residual_tmp57 + residual_tmp116*residual_tmp68 - residual_tmp150 - residual_tmp280) + ((s_t(1) / s_t(3)))*residual_tmp120*residual_tmp60)) + u0_direction_grad_1*(mu*(residual_tmp108 + residual_tmp266) + residual_tmp128*residual_tmp262 + residual_tmp24*(-eta_s*(-residual_tmp144*residual_tmp57 + residual_tmp149*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp139*residual_tmp60) + residual_tmp263*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp262 + residual_tmp24*(-eta_s*(residual_tmp173*residual_tmp68 - residual_tmp178*residual_tmp57 + residual_tmp277) + ((s_t(1) / s_t(3)))*residual_tmp168*residual_tmp60) + residual_tmp263*residual_tmp62 + residual_tmp273) + u1_direction_grad_0*(residual_tmp10*residual_tmp262 + residual_tmp24*(-eta_s*(residual_tmp241 + residual_tmp32*residual_tmp68 - residual_tmp41*residual_tmp57) + residual_tmp282*residual_tmp8 + residual_tmp290*residual_tmp52) + residual_tmp25*residual_tmp263 + residual_tmp292) + u1_direction_grad_1*(residual_tmp226*residual_tmp262 + residual_tmp230*residual_tmp263 + residual_tmp24*(-eta_s*(residual_tmp237*residual_tmp68 - residual_tmp240*residual_tmp57) + ((s_t(1) / s_t(3)))*residual_tmp236*residual_tmp60) + residual_tmp268) + u1_direction_grad_2*(residual_tmp188*residual_tmp262 + residual_tmp24*(-eta_s*(residual_tmp198*residual_tmp68 - residual_tmp202*residual_tmp57 - residual_tmp281) + ((s_t(1) / s_t(3)))*residual_tmp197*residual_tmp60 - residual_tmp283) + residual_tmp263*residual_tmp67 + residual_tmp287) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(-residual_tmp221 - residual_tmp57*residual_tmp94 + residual_tmp68*residual_tmp96) + ((s_t(1) / s_t(3)))*residual_tmp105*residual_tmp60 - residual_tmp282*u1_grad_2) + residual_tmp262*residual_tmp90 + residual_tmp263*residual_tmp33 + residual_tmp289) + u2_direction_grad_1*(residual_tmp208*residual_tmp262 + residual_tmp24*(-eta_s*(-residual_tmp217*residual_tmp57 + residual_tmp220*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp216*residual_tmp60) + residual_tmp259 + residual_tmp263*residual_tmp57) + u2_direction_grad_2*(mu*(residual_tmp227*residual_tmp3 + residual_tmp295) + residual_tmp24*(-eta_s*(-residual_tmp254*residual_tmp57 + residual_tmp256*residual_tmp68 + residual_tmp298) + residual_tmp253*residual_tmp290 + residual_tmp299) + residual_tmp245*residual_tmp262 + residual_tmp247*residual_tmp263 + residual_tmp297);
      const s_t grad_coeff0_2 = u0_direction_grad_0*(mu*(-residual_tmp155 + s_t(2)*residual_tmp278*u0_grad_2 - residual_tmp279*u0_grad_2) + residual_tmp112*residual_tmp303 + residual_tmp121*residual_tmp304 + residual_tmp24*(-eta_s*(residual_tmp114*residual_tmp63 - residual_tmp116*residual_tmp67 - residual_tmp179 - residual_tmp306) + ((s_t(1) / s_t(3)))*residual_tmp120*residual_tmp62)) + u0_direction_grad_1*(residual_tmp128*residual_tmp303 + residual_tmp24*(-eta_s*(residual_tmp144*residual_tmp63 - residual_tmp149*residual_tmp67 - residual_tmp277) + ((s_t(1) / s_t(3)))*residual_tmp139*residual_tmp62) + residual_tmp273 + residual_tmp304*residual_tmp60) + u0_direction_grad_2*(mu*(residual_tmp111 + residual_tmp266 + s_t(2)) + residual_tmp157*residual_tmp303 + residual_tmp24*(-eta_s*(-residual_tmp173*residual_tmp67 + residual_tmp178*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp168*residual_tmp62) + residual_tmp304*residual_tmp62) + u1_direction_grad_0*(residual_tmp10*residual_tmp303 + residual_tmp24*(-eta_s*(-residual_tmp203 - residual_tmp32*residual_tmp67 + residual_tmp41*residual_tmp63) - residual_tmp282*u2_grad_1 + ((s_t(1) / s_t(3)))*residual_tmp52*residual_tmp62) + residual_tmp25*residual_tmp304 + residual_tmp310) + u1_direction_grad_1*(mu*(residual_tmp0*residual_tmp222 + residual_tmp315) + residual_tmp226*residual_tmp303 + residual_tmp230*residual_tmp304 + residual_tmp24*(-eta_s*(-residual_tmp237*residual_tmp67 + residual_tmp240*residual_tmp63 + residual_tmp281) + residual_tmp236*residual_tmp311 + residual_tmp283) + residual_tmp316) + u1_direction_grad_2*(residual_tmp188*residual_tmp303 + residual_tmp24*(-eta_s*(-residual_tmp198*residual_tmp67 + residual_tmp202*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp197*residual_tmp62) + residual_tmp300 + residual_tmp304*residual_tmp67) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(residual_tmp257 + residual_tmp63*residual_tmp94 - residual_tmp67*residual_tmp96) + residual_tmp105*residual_tmp311 + residual_tmp2*residual_tmp282) + residual_tmp303*residual_tmp90 + residual_tmp304*residual_tmp33 + residual_tmp313) + u2_direction_grad_1*(residual_tmp208*residual_tmp303 + residual_tmp24*(-eta_s*(residual_tmp217*residual_tmp63 - residual_tmp220*residual_tmp67 - residual_tmp298) + ((s_t(1) / s_t(3)))*residual_tmp216*residual_tmp62 - residual_tmp299) + residual_tmp304*residual_tmp57 + residual_tmp308) + u2_direction_grad_2*(residual_tmp24*(-eta_s*(residual_tmp254*residual_tmp63 - residual_tmp256*residual_tmp67) + ((s_t(1) / s_t(3)))*residual_tmp253*residual_tmp62) + residual_tmp245*residual_tmp303 + residual_tmp247*residual_tmp304 + residual_tmp305);
      const s_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp112*residual_tmp320 + residual_tmp121*residual_tmp326 + residual_tmp24*(-eta_s*(residual_tmp116*residual_tmp20 - residual_tmp33*residual_tmp334) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp335) + residual_tmp5) + u0_direction_grad_1*(residual_tmp128*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp149*residual_tmp20 - residual_tmp33*residual_tmp365 + residual_tmp366) + residual_tmp355*residual_tmp8 + residual_tmp363*residual_tmp364) + residual_tmp292 + residual_tmp326*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp173*residual_tmp20 - residual_tmp33*residual_tmp353 - residual_tmp354) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp352 - residual_tmp356) + residual_tmp310 + residual_tmp326*residual_tmp62) + u1_direction_grad_0*(mu*(residual_tmp328 + residual_tmp330) + residual_tmp10*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp32 - residual_tmp33*residual_tmp332) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp333) + residual_tmp25*residual_tmp326) + u1_direction_grad_1*(mu*(s_t(2)*residual_tmp222*u1_grad_0 - residual_tmp344*u1_grad_0 - residual_tmp346) + residual_tmp226*residual_tmp320 + residual_tmp230*residual_tmp326 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp237 - residual_tmp280 - residual_tmp33*residual_tmp350 - residual_tmp351) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp348)) + u1_direction_grad_2*(residual_tmp188*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp198*residual_tmp20 - residual_tmp33*residual_tmp341 + residual_tmp343) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp339) + residual_tmp326*residual_tmp67 + residual_tmp337) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp96 - residual_tmp322*residual_tmp33) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp323) + residual_tmp319 + residual_tmp320*residual_tmp90 + residual_tmp326*residual_tmp33) + u2_direction_grad_1*(residual_tmp208*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp220 - residual_tmp33*residual_tmp358 - residual_tmp359) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp357 - residual_tmp355*u0_grad_2) + residual_tmp326*residual_tmp57 + residual_tmp362) + u2_direction_grad_2*(mu*(residual_tmp124*residual_tmp227 + residual_tmp368) + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp256 - residual_tmp33*residual_tmp372 + residual_tmp373) + residual_tmp364*residual_tmp371 + residual_tmp374) + residual_tmp245*residual_tmp320 + residual_tmp247*residual_tmp326 + residual_tmp370);
      const s_t grad_coeff1_1 = u0_direction_grad_0*(mu*(s_t(2)*residual_tmp2*residual_tmp278 - residual_tmp225) + residual_tmp112*residual_tmp375 + residual_tmp121*residual_tmp378 + residual_tmp229 + residual_tmp24*(eta_s*(residual_tmp116*residual_tmp60 + residual_tmp334*residual_tmp57 + residual_tmp366) - residual_tmp335*residual_tmp376 + residual_tmp377*residual_tmp8)) + u0_direction_grad_1*(residual_tmp128*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp149*residual_tmp60 + residual_tmp365*residual_tmp57) - residual_tmp363*residual_tmp376) + residual_tmp268 + residual_tmp378*residual_tmp60) + u0_direction_grad_2*(mu*(residual_tmp315 + s_t(4)*residual_tmp89) + residual_tmp157*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp173*residual_tmp60 + residual_tmp353*residual_tmp57 - residual_tmp386) - residual_tmp352*residual_tmp376 - residual_tmp377*u2_grad_0) + residual_tmp316 + residual_tmp378*residual_tmp62) + u1_direction_grad_0*(-mu*residual_tmp346 + residual_tmp10*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp152 + residual_tmp32*residual_tmp60 + residual_tmp332*residual_tmp57 - residual_tmp351) - residual_tmp333*residual_tmp376) + residual_tmp25*residual_tmp378) + u1_direction_grad_1*(mu*(residual_tmp328 + residual_tmp382) + residual_tmp226*residual_tmp375 + residual_tmp230*residual_tmp378 + residual_tmp24*(eta_s*(residual_tmp237*residual_tmp60 + residual_tmp350*residual_tmp57) - residual_tmp348*residual_tmp376)) + u1_direction_grad_2*(-mu*residual_tmp384 + residual_tmp188*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp198*residual_tmp60 + residual_tmp276 + residual_tmp341*residual_tmp57 + residual_tmp385) - residual_tmp339*residual_tmp376) + residual_tmp378*residual_tmp67) + u2_direction_grad_0*(mu*(s_t(4)*residual_tmp156 + residual_tmp387) + residual_tmp24*(eta_s*(residual_tmp322*residual_tmp57 - residual_tmp359 + residual_tmp60*residual_tmp96) - residual_tmp323*residual_tmp376 - residual_tmp377*u0_grad_2) + residual_tmp33*residual_tmp378 + residual_tmp375*residual_tmp90 + residual_tmp388) + u2_direction_grad_1*(residual_tmp208*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp220*residual_tmp60 + residual_tmp358*residual_tmp57) - residual_tmp357*residual_tmp376) + residual_tmp378*residual_tmp57 + residual_tmp380) + u2_direction_grad_2*(mu*(s_t(2)*residual_tmp2*residual_tmp227 - residual_tmp390) + residual_tmp24*(eta_s*(residual_tmp256*residual_tmp60 + residual_tmp372*residual_tmp57 + residual_tmp392) + residual_tmp182*residual_tmp377 - residual_tmp371*residual_tmp376) + residual_tmp245*residual_tmp375 + residual_tmp247*residual_tmp378 + residual_tmp391);
      const s_t grad_coeff1_2 = u0_direction_grad_0*(mu*(residual_tmp186 + residual_tmp269*residual_tmp278) + residual_tmp112*residual_tmp393 + residual_tmp121*residual_tmp394 + residual_tmp191 + residual_tmp24*(-eta_s*(-residual_tmp116*residual_tmp62 + residual_tmp334*residual_tmp63 + residual_tmp354) + residual_tmp335*residual_tmp398 + residual_tmp356)) + u0_direction_grad_1*(residual_tmp128*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp149*residual_tmp62 + residual_tmp365*residual_tmp63 - residual_tmp386) - residual_tmp355*u2_grad_0 + ((s_t(1) / s_t(3)))*residual_tmp363*residual_tmp67) + residual_tmp287 + residual_tmp394*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp173*residual_tmp62 + residual_tmp353*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp352*residual_tmp67) + residual_tmp300 + residual_tmp394*residual_tmp62) + u1_direction_grad_0*(residual_tmp10*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp32*residual_tmp62 + residual_tmp332*residual_tmp63 - residual_tmp343) + ((s_t(1) / s_t(3)))*residual_tmp333*residual_tmp67) + residual_tmp25*residual_tmp394 + residual_tmp337) + u1_direction_grad_1*(mu*(s_t(2)*residual_tmp222*u1_grad_2 - residual_tmp344*u1_grad_2 - residual_tmp384) + residual_tmp226*residual_tmp393 + residual_tmp230*residual_tmp394 + residual_tmp24*(-eta_s*(-residual_tmp237*residual_tmp62 - residual_tmp275 + residual_tmp350*residual_tmp63 - residual_tmp396) + ((s_t(1) / s_t(3)))*residual_tmp348*residual_tmp67)) + u1_direction_grad_2*(mu*(residual_tmp330 + residual_tmp382 + s_t(2)) + residual_tmp188*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp198*residual_tmp62 + residual_tmp341*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp339*residual_tmp67) + residual_tmp394*residual_tmp67) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(residual_tmp322*residual_tmp63 - residual_tmp373 - residual_tmp62*residual_tmp96) + ((s_t(1) / s_t(3)))*residual_tmp323*residual_tmp67 - residual_tmp374) + residual_tmp33*residual_tmp394 + residual_tmp393*residual_tmp90 + residual_tmp397) + u2_direction_grad_1*(residual_tmp208*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp220*residual_tmp62 + residual_tmp358*residual_tmp63 + residual_tmp392) + residual_tmp182*residual_tmp355 + residual_tmp357*residual_tmp398) + residual_tmp394*residual_tmp57 + residual_tmp399) + u2_direction_grad_2*(residual_tmp24*(-eta_s*(-residual_tmp256*residual_tmp62 + residual_tmp372*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp371*residual_tmp67) + residual_tmp245*residual_tmp393 + residual_tmp247*residual_tmp394 + residual_tmp395);
      const s_t grad_coeff2_0 = u0_direction_grad_0*(residual_tmp112*residual_tmp400 + residual_tmp121*residual_tmp403 + residual_tmp24*(-eta_s*(residual_tmp114*residual_tmp20 - residual_tmp25*residual_tmp334) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp407) + residual_tmp88) + u0_direction_grad_1*(residual_tmp128*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp144*residual_tmp20 - residual_tmp25*residual_tmp365 - residual_tmp415) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp414 - residual_tmp417) + residual_tmp289 + residual_tmp403*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp178*residual_tmp20 - residual_tmp25*residual_tmp353 + residual_tmp422) + residual_tmp2*residual_tmp416 + residual_tmp420*residual_tmp421) + residual_tmp313 + residual_tmp403*residual_tmp62) + u1_direction_grad_0*(residual_tmp10*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp41 - residual_tmp25*residual_tmp332) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp401) + residual_tmp25*residual_tmp403 + residual_tmp319) + u1_direction_grad_1*(mu*(residual_tmp122*residual_tmp222 + residual_tmp387) + residual_tmp226*residual_tmp400 + residual_tmp230*residual_tmp403 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp240 - residual_tmp25*residual_tmp350 + residual_tmp424) + residual_tmp421*residual_tmp423 + residual_tmp425) + residual_tmp388) + u1_direction_grad_2*(residual_tmp188*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp202 - residual_tmp25*residual_tmp341 - residual_tmp419) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp418 - residual_tmp416*u0_grad_1) + residual_tmp397 + residual_tmp403*residual_tmp67) + u2_direction_grad_0*(mu*(residual_tmp404 + residual_tmp405) + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp94 - residual_tmp25*residual_tmp322) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp406) + residual_tmp33*residual_tmp403 + residual_tmp400*residual_tmp90) + u2_direction_grad_1*(residual_tmp208*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp217 - residual_tmp25*residual_tmp358 + residual_tmp410) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp409) + residual_tmp403*residual_tmp57 + residual_tmp408) + u2_direction_grad_2*(mu*(s_t(2)*residual_tmp227*u2_grad_0 - residual_tmp411*u2_grad_0 - residual_tmp412) + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp254 - residual_tmp25*residual_tmp372 - residual_tmp306 - residual_tmp342) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp413) + residual_tmp245*residual_tmp400 + residual_tmp247*residual_tmp403);
      const s_t grad_coeff2_1 = u0_direction_grad_0*(mu*(residual_tmp207 + residual_tmp271*residual_tmp278) + residual_tmp112*residual_tmp426 + residual_tmp121*residual_tmp427 + residual_tmp210 + residual_tmp24*(-eta_s*(-residual_tmp114*residual_tmp60 + residual_tmp334*residual_tmp68 + residual_tmp415) + residual_tmp407*residual_tmp431 + residual_tmp417)) + u0_direction_grad_1*(residual_tmp128*residual_tmp426 + residual_tmp24*(-eta_s*(-residual_tmp144*residual_tmp60 + residual_tmp365*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp414*residual_tmp57) + residual_tmp259 + residual_tmp427*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp426 + residual_tmp24*(-eta_s*(-residual_tmp178*residual_tmp60 + residual_tmp353*residual_tmp68 - residual_tmp430) - residual_tmp416*u1_grad_0 + ((s_t(1) / s_t(3)))*residual_tmp420*residual_tmp57) + residual_tmp308 + residual_tmp427*residual_tmp62) + u1_direction_grad_0*(residual_tmp10*residual_tmp426 + residual_tmp24*(-eta_s*(residual_tmp332*residual_tmp68 - residual_tmp41*residual_tmp60 - residual_tmp424) + ((s_t(1) / s_t(3)))*residual_tmp401*residual_tmp57 - residual_tmp425) + residual_tmp25*residual_tmp427 + residual_tmp362) + u1_direction_grad_1*(residual_tmp226*residual_tmp426 + residual_tmp230*residual_tmp427 + residual_tmp24*(-eta_s*(-residual_tmp240*residual_tmp60 + residual_tmp350*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp423*residual_tmp57) + residual_tmp380) + u1_direction_grad_2*(residual_tmp188*residual_tmp426 + residual_tmp24*(-eta_s*(-residual_tmp202*residual_tmp60 + residual_tmp341*residual_tmp68 + residual_tmp432) + residual_tmp182*residual_tmp416 + residual_tmp418*residual_tmp431) + residual_tmp399 + residual_tmp427*residual_tmp67) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(residual_tmp322*residual_tmp68 - residual_tmp410 - residual_tmp60*residual_tmp94) + ((s_t(1) / s_t(3)))*residual_tmp406*residual_tmp57) + residual_tmp33*residual_tmp427 + residual_tmp408 + residual_tmp426*residual_tmp90) + u2_direction_grad_1*(mu*(residual_tmp404 + residual_tmp428) + residual_tmp208*residual_tmp426 + residual_tmp24*(-eta_s*(-residual_tmp217*residual_tmp60 + residual_tmp358*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp409*residual_tmp57) + residual_tmp427*residual_tmp57) + u2_direction_grad_2*(mu*(s_t(2)*residual_tmp227*u2_grad_1 - residual_tmp411*u2_grad_1 - residual_tmp429) + residual_tmp24*(-eta_s*(-residual_tmp254*residual_tmp60 - residual_tmp274 + residual_tmp372*residual_tmp68 - residual_tmp396) + ((s_t(1) / s_t(3)))*residual_tmp413*residual_tmp57) + residual_tmp245*residual_tmp426 + residual_tmp247*residual_tmp427);
      const s_t grad_coeff2_2 = u0_direction_grad_0*(mu*(-residual_tmp244 + s_t(2)*residual_tmp278*residual_tmp8) + residual_tmp112*residual_tmp433 + residual_tmp121*residual_tmp436 + residual_tmp24*(eta_s*(residual_tmp114*residual_tmp62 + residual_tmp334*residual_tmp67 + residual_tmp422) + residual_tmp2*residual_tmp435 - residual_tmp407*residual_tmp434) + residual_tmp246) + u0_direction_grad_1*(mu*(residual_tmp295 + s_t(4)*residual_tmp9) + residual_tmp128*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp144*residual_tmp62 + residual_tmp365*residual_tmp67 - residual_tmp430) - residual_tmp414*residual_tmp434 - residual_tmp435*u1_grad_0) + residual_tmp297 + residual_tmp436*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp178*residual_tmp62 + residual_tmp353*residual_tmp67) - residual_tmp420*residual_tmp434) + residual_tmp305 + residual_tmp436*residual_tmp62) + u1_direction_grad_0*(mu*(s_t(4)*residual_tmp127 + residual_tmp368) + residual_tmp10*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp332*residual_tmp67 + residual_tmp41*residual_tmp62 - residual_tmp419) - residual_tmp401*residual_tmp434 - residual_tmp435*u0_grad_1) + residual_tmp25*residual_tmp436 + residual_tmp370) + u1_direction_grad_1*(mu*(s_t(2)*residual_tmp222*residual_tmp8 - residual_tmp390) + residual_tmp226*residual_tmp433 + residual_tmp230*residual_tmp436 + residual_tmp24*(eta_s*(residual_tmp240*residual_tmp62 + residual_tmp350*residual_tmp67 + residual_tmp432) + residual_tmp182*residual_tmp435 - residual_tmp423*residual_tmp434) + residual_tmp391) + u1_direction_grad_2*(residual_tmp188*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp202*residual_tmp62 + residual_tmp341*residual_tmp67) - residual_tmp418*residual_tmp434) + residual_tmp395 + residual_tmp436*residual_tmp67) + u2_direction_grad_0*(-mu*residual_tmp412 + residual_tmp24*(eta_s*(residual_tmp181 + residual_tmp322*residual_tmp67 - residual_tmp342 + residual_tmp62*residual_tmp94) - residual_tmp406*residual_tmp434) + residual_tmp33*residual_tmp436 + residual_tmp433*residual_tmp90) + u2_direction_grad_1*(-mu*residual_tmp429 + residual_tmp208*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp217*residual_tmp62 - residual_tmp274 + residual_tmp358*residual_tmp67 + residual_tmp385) - residual_tmp409*residual_tmp434) + residual_tmp436*residual_tmp57) + u2_direction_grad_2*(mu*(residual_tmp405 + residual_tmp428 + s_t(2)) + residual_tmp24*(eta_s*(residual_tmp254*residual_tmp62 + residual_tmp372*residual_tmp67) - residual_tmp413*residual_tmp434) + residual_tmp245*residual_tmp433 + residual_tmp247*residual_tmp436);
      grad_coeff0_0_values[lane] = grad_coeff0_0;
      grad_coeff0_1_values[lane] = grad_coeff0_1;
      grad_coeff0_2_values[lane] = grad_coeff0_2;
      grad_coeff1_0_values[lane] = grad_coeff1_0;
      grad_coeff1_1_values[lane] = grad_coeff1_1;
      grad_coeff1_2_values[lane] = grad_coeff1_2;
      grad_coeff2_0_values[lane] = grad_coeff2_0;
      grad_coeff2_1_values[lane] = grad_coeff2_1;
      grad_coeff2_2_values[lane] = grad_coeff2_2;
    }
    for (int test = 0; test < NS; ++test) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
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
        output[test * NC][lane] += q_weight[q] * det * (grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
        output[test * NC + 1][lane] += q_weight[q] * det * (grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
        output[test * NC + 2][lane] += q_weight[q] * det * (grad_coeff2_0_values[lane] * test_grad0 + grad_coeff2_1_values[lane] * test_grad1 + grad_coeff2_2_values[lane] * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_residual_merit_d3_simplex_jacobian_action_block_contiguous(
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
    const s_t direction[3 * NS][VS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
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
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u0_grad_0_ref_values[lane] = s_t(0);
      u0_grad_1_ref_values[lane] = s_t(0);
      u0_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC][lane];
        u0_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u0_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u0_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u0_old_grad_0_ref_values[lane] = s_t(0);
      u0_old_grad_1_ref_values[lane] = s_t(0);
      u0_old_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC][lane];
        u0_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u0_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u0_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u0_direction_grad_0_ref_values[lane] = s_t(0);
      u0_direction_grad_1_ref_values[lane] = s_t(0);
      u0_direction_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = direction[trial * NC][lane];
        u0_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u0_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u0_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u1_grad_0_ref_values[lane] = s_t(0);
      u1_grad_1_ref_values[lane] = s_t(0);
      u1_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC + 1][lane];
        u1_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u1_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u1_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u1_old_grad_0_ref_values[lane] = s_t(0);
      u1_old_grad_1_ref_values[lane] = s_t(0);
      u1_old_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC + 1][lane];
        u1_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u1_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u1_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u1_direction_grad_0_ref_values[lane] = s_t(0);
      u1_direction_grad_1_ref_values[lane] = s_t(0);
      u1_direction_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = direction[trial * NC + 1][lane];
        u1_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u1_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u1_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u2_grad_0_ref_values[lane] = s_t(0);
      u2_grad_1_ref_values[lane] = s_t(0);
      u2_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC + 2][lane];
        u2_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u2_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u2_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u2_old_grad_0_ref_values[lane] = s_t(0);
      u2_old_grad_1_ref_values[lane] = s_t(0);
      u2_old_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC + 2][lane];
        u2_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u2_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u2_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      u2_direction_grad_0_ref_values[lane] = s_t(0);
      u2_direction_grad_1_ref_values[lane] = s_t(0);
      u2_direction_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = direction[trial * NC + 2][lane];
        u2_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        u2_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        u2_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
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
      const s_t u0_grad_0_ref = u0_grad_0_ref_values[lane];
      const s_t u0_grad_1_ref = u0_grad_1_ref_values[lane];
      const s_t u0_grad_2_ref = u0_grad_2_ref_values[lane];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
      const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
      const s_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[lane];
      const s_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[lane];
      const s_t u0_old_grad_2_ref = u0_old_grad_2_ref_values[lane];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
      const s_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
      const s_t u0_direction_grad_0_ref = u0_direction_grad_0_ref_values[lane];
      const s_t u0_direction_grad_1_ref = u0_direction_grad_1_ref_values[lane];
      const s_t u0_direction_grad_2_ref = u0_direction_grad_2_ref_values[lane];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
      const s_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
      const s_t u1_grad_0_ref = u1_grad_0_ref_values[lane];
      const s_t u1_grad_1_ref = u1_grad_1_ref_values[lane];
      const s_t u1_grad_2_ref = u1_grad_2_ref_values[lane];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
      const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
      const s_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[lane];
      const s_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[lane];
      const s_t u1_old_grad_2_ref = u1_old_grad_2_ref_values[lane];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
      const s_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
      const s_t u1_direction_grad_0_ref = u1_direction_grad_0_ref_values[lane];
      const s_t u1_direction_grad_1_ref = u1_direction_grad_1_ref_values[lane];
      const s_t u1_direction_grad_2_ref = u1_direction_grad_2_ref_values[lane];
      const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
      const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
      const s_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
      const s_t u2_grad_0_ref = u2_grad_0_ref_values[lane];
      const s_t u2_grad_1_ref = u2_grad_1_ref_values[lane];
      const s_t u2_grad_2_ref = u2_grad_2_ref_values[lane];
      const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
      const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
      const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
      const s_t u2_old_grad_0_ref = u2_old_grad_0_ref_values[lane];
      const s_t u2_old_grad_1_ref = u2_old_grad_1_ref_values[lane];
      const s_t u2_old_grad_2_ref = u2_old_grad_2_ref_values[lane];
      const s_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
      const s_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
      const s_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
      const s_t u2_direction_grad_0_ref = u2_direction_grad_0_ref_values[lane];
      const s_t u2_direction_grad_1_ref = u2_direction_grad_1_ref_values[lane];
      const s_t u2_direction_grad_2_ref = u2_direction_grad_2_ref_values[lane];
      const s_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
      const s_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
      const s_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = s_t(2)*u0_grad_2;
      const s_t residual_tmp1 = residual_tmp0*u1_grad_2;
      const s_t residual_tmp2 = u1_grad_1 + s_t(1);
      const s_t residual_tmp3 = s_t(2)*u0_grad_1;
      const s_t residual_tmp4 = residual_tmp2*residual_tmp3;
      const s_t residual_tmp5 = mu*(-residual_tmp1 - residual_tmp4);
      const s_t residual_tmp6 = u0_grad_2*u2_grad_1;
      const s_t residual_tmp7 = -residual_tmp6;
      const s_t residual_tmp8 = u2_grad_2 + s_t(1);
      const s_t residual_tmp9 = residual_tmp8*u0_grad_1;
      const s_t residual_tmp10 = -residual_tmp7 - residual_tmp9;
      const s_t residual_tmp11 = u1_grad_2*u2_grad_1;
      const s_t residual_tmp12 = s_t(2)*residual_tmp11;
      const s_t residual_tmp13 = -s_t(2)*residual_tmp2*residual_tmp8;
      const s_t residual_tmp14 = ((s_t(1) / s_t(2)))*lmbda;
      const s_t residual_tmp15 = residual_tmp14*(-residual_tmp12 - residual_tmp13);
      const s_t residual_tmp16 = u0_grad_0*u1_grad_1;
      const s_t residual_tmp17 = u0_grad_1*u1_grad_2;
      const s_t residual_tmp18 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp19 = u0_grad_2*u2_grad_0;
      const s_t residual_tmp20 = -residual_tmp11 + residual_tmp2 + u1_grad_1*u2_grad_2 + u2_grad_2;
      const s_t residual_tmp21 = -residual_tmp19 + u0_grad_0*u2_grad_2 + u0_grad_0;
      const s_t residual_tmp22 = residual_tmp16 - residual_tmp18;
      const s_t residual_tmp23 = -residual_tmp11*u0_grad_0 + residual_tmp16*u2_grad_2 + residual_tmp17*u2_grad_0 - residual_tmp18*u2_grad_2 - residual_tmp19*u1_grad_1 + residual_tmp20 + residual_tmp21 + residual_tmp22 + residual_tmp6*u1_grad_0;
      const s_t residual_tmp24 = pow_m1(residual_tmp23);
      const s_t residual_tmp25 = residual_tmp7 + u0_grad_1*u2_grad_2 + u0_grad_1;
      const s_t residual_tmp26 = residual_tmp20*u_dt_shift;
      const s_t residual_tmp27 = u1_grad_2*u_dt_shift + u1_old_grad_2;
      const s_t residual_tmp28 = residual_tmp27*u2_grad_1;
      const s_t residual_tmp29 = u1_grad_1*u_dt_shift + u1_old_grad_1;
      const s_t residual_tmp30 = residual_tmp29*residual_tmp8;
      const s_t residual_tmp31 = residual_tmp28 - residual_tmp30;
      const s_t residual_tmp32 = -residual_tmp26 - residual_tmp31;
      const s_t residual_tmp33 = -residual_tmp17 + u0_grad_2*u1_grad_1 + u0_grad_2;
      const s_t residual_tmp34 = u2_grad_1*u_dt_shift + u2_old_grad_1;
      const s_t residual_tmp35 = residual_tmp34*residual_tmp8;
      const s_t residual_tmp36 = u2_grad_2*u_dt_shift + u2_old_grad_2;
      const s_t residual_tmp37 = residual_tmp36*u2_grad_1;
      const s_t residual_tmp38 = u0_grad_2*u_dt_shift + u0_old_grad_2;
      const s_t residual_tmp39 = u0_grad_1*u_dt_shift + u0_old_grad_1;
      const s_t residual_tmp40 = residual_tmp38*u0_grad_1 - residual_tmp39*u0_grad_2;
      const s_t residual_tmp41 = residual_tmp35 - residual_tmp37 + residual_tmp40;
      const s_t residual_tmp42 = residual_tmp25*u_dt_shift;
      const s_t residual_tmp43 = residual_tmp36*u0_grad_1;
      const s_t residual_tmp44 = residual_tmp34*u0_grad_2;
      const s_t residual_tmp45 = residual_tmp42 + residual_tmp43 - residual_tmp44;
      const s_t residual_tmp46 = residual_tmp39*residual_tmp8;
      const s_t residual_tmp47 = residual_tmp38*u2_grad_1;
      const s_t residual_tmp48 = residual_tmp46 - residual_tmp47;
      const s_t residual_tmp49 = s_t(3)*eta_b;
      const s_t residual_tmp50 = residual_tmp49*(residual_tmp45 + residual_tmp48);
      const s_t residual_tmp51 = s_t(2)*eta_s;
      const s_t residual_tmp52 = residual_tmp50 + residual_tmp51*(s_t(2)*residual_tmp39*residual_tmp8 - residual_tmp45 - s_t(2)*residual_tmp47);
      const s_t residual_tmp53 = ((s_t(1) / s_t(3)))*residual_tmp20;
      const s_t residual_tmp54 = pow_m2(residual_tmp23);
      const s_t residual_tmp55 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp56 = u0_grad_2*u1_grad_0;
      const s_t residual_tmp57 = -residual_tmp56 + u0_grad_0*u1_grad_2 + u1_grad_2;
      const s_t residual_tmp58 = u1_grad_2*u2_grad_0;
      const s_t residual_tmp59 = -residual_tmp58;
      const s_t residual_tmp60 = residual_tmp59 + u1_grad_0*u2_grad_2 + u1_grad_0;
      const s_t residual_tmp61 = u1_grad_0*u2_grad_1;
      const s_t residual_tmp62 = -residual_tmp61 + u1_grad_1*u2_grad_0 + u2_grad_0;
      const s_t residual_tmp63 = residual_tmp2 + residual_tmp22 + u0_grad_0;
      const s_t residual_tmp64 = u2_grad_0*u_dt_shift + u2_old_grad_0;
      const s_t residual_tmp65 = -residual_tmp20*residual_tmp64 + residual_tmp33*residual_tmp55 + residual_tmp34*residual_tmp60 + residual_tmp36*residual_tmp62 - residual_tmp38*residual_tmp63 + residual_tmp39*residual_tmp57;
      const s_t residual_tmp66 = u0_grad_1*u2_grad_0;
      const s_t residual_tmp67 = -residual_tmp66 + u0_grad_0*u2_grad_1 + u2_grad_1;
      const s_t residual_tmp68 = residual_tmp21 + residual_tmp8;
      const s_t residual_tmp69 = u1_grad_0*u_dt_shift + u1_old_grad_0;
      const s_t residual_tmp70 = -residual_tmp20*residual_tmp69 + residual_tmp25*residual_tmp55 + residual_tmp27*residual_tmp62 + residual_tmp29*residual_tmp60 + residual_tmp38*residual_tmp67 - residual_tmp39*residual_tmp68;
      const s_t residual_tmp71 = residual_tmp25*residual_tmp69;
      const s_t residual_tmp72 = residual_tmp27*residual_tmp67;
      const s_t residual_tmp73 = residual_tmp33*residual_tmp64;
      const s_t residual_tmp74 = residual_tmp34*residual_tmp57;
      const s_t residual_tmp75 = residual_tmp29*residual_tmp68;
      const s_t residual_tmp76 = -residual_tmp75;
      const s_t residual_tmp77 = residual_tmp36*residual_tmp63;
      const s_t residual_tmp78 = -residual_tmp77;
      const s_t residual_tmp79 = residual_tmp71 + residual_tmp72 + residual_tmp73 + residual_tmp74 + residual_tmp76 + residual_tmp78;
      const s_t residual_tmp80 = residual_tmp20*residual_tmp55;
      const s_t residual_tmp81 = residual_tmp38*residual_tmp62 + residual_tmp39*residual_tmp60 - residual_tmp80;
      const s_t residual_tmp82 = residual_tmp49*(residual_tmp79 + residual_tmp81);
      const s_t residual_tmp83 = residual_tmp51*(s_t(2)*residual_tmp38*residual_tmp62 + s_t(2)*residual_tmp39*residual_tmp60 - residual_tmp79 - s_t(2)*residual_tmp80) + residual_tmp82;
      const s_t residual_tmp84 = -residual_tmp83;
      const s_t residual_tmp85 = residual_tmp54*(eta_s*(residual_tmp25*residual_tmp70 + residual_tmp33*residual_tmp65) + residual_tmp53*residual_tmp84);
      const s_t residual_tmp86 = residual_tmp3*u2_grad_1;
      const s_t residual_tmp87 = residual_tmp0*residual_tmp8;
      const s_t residual_tmp88 = mu*(-residual_tmp86 - residual_tmp87);
      const s_t residual_tmp89 = residual_tmp2*u0_grad_2;
      const s_t residual_tmp90 = residual_tmp17 - residual_tmp89;
      const s_t residual_tmp91 = residual_tmp34*u1_grad_2;
      const s_t residual_tmp92 = residual_tmp2*residual_tmp36;
      const s_t residual_tmp93 = residual_tmp91 - residual_tmp92;
      const s_t residual_tmp94 = -residual_tmp26 - residual_tmp93;
      const s_t residual_tmp95 = -residual_tmp2*residual_tmp27 + residual_tmp29*u1_grad_2;
      const s_t residual_tmp96 = -residual_tmp40 - residual_tmp95;
      const s_t residual_tmp97 = residual_tmp33*u_dt_shift;
      const s_t residual_tmp98 = residual_tmp29*u0_grad_2;
      const s_t residual_tmp99 = residual_tmp27*u0_grad_1;
      const s_t residual_tmp100 = residual_tmp97 + residual_tmp98 - residual_tmp99;
      const s_t residual_tmp101 = residual_tmp2*residual_tmp38;
      const s_t residual_tmp102 = residual_tmp39*u1_grad_2;
      const s_t residual_tmp103 = residual_tmp101 - residual_tmp102;
      const s_t residual_tmp104 = residual_tmp49*(residual_tmp100 + residual_tmp103);
      const s_t residual_tmp105 = residual_tmp104 + residual_tmp51*(-residual_tmp100 - s_t(2)*residual_tmp102 + s_t(2)*residual_tmp2*residual_tmp38);
      const s_t residual_tmp106 = s_t(2)*pow_2(u1_grad_2);
      const s_t residual_tmp107 = s_t(2)*pow_2(residual_tmp8) + s_t(2);
      const s_t residual_tmp108 = residual_tmp106 + residual_tmp107;
      const s_t residual_tmp109 = s_t(2)*pow_2(u2_grad_1);
      const s_t residual_tmp110 = s_t(2)*pow_2(residual_tmp2);
      const s_t residual_tmp111 = residual_tmp109 + residual_tmp110;
      const s_t residual_tmp112 = -residual_tmp11 + residual_tmp2*residual_tmp8;
      const s_t residual_tmp113 = -residual_tmp101 + residual_tmp102;
      const s_t residual_tmp114 = residual_tmp113 + residual_tmp97;
      const s_t residual_tmp115 = -residual_tmp46 + residual_tmp47;
      const s_t residual_tmp116 = residual_tmp115 + residual_tmp42;
      const s_t residual_tmp117 = residual_tmp26 - residual_tmp91 + residual_tmp92;
      const s_t residual_tmp118 = -residual_tmp28 + residual_tmp30;
      const s_t residual_tmp119 = residual_tmp49*(-residual_tmp117 - residual_tmp118);
      const s_t residual_tmp120 = residual_tmp119 + residual_tmp51*(-s_t(2)*residual_tmp26 - residual_tmp31 - residual_tmp93);
      const s_t residual_tmp121 = -residual_tmp20;
      const s_t residual_tmp122 = s_t(2)*u2_grad_0;
      const s_t residual_tmp123 = residual_tmp122*u2_grad_1;
      const s_t residual_tmp124 = s_t(2)*u1_grad_0;
      const s_t residual_tmp125 = residual_tmp124*residual_tmp2;
      const s_t residual_tmp126 = residual_tmp123 + residual_tmp125;
      const s_t residual_tmp127 = residual_tmp8*u1_grad_0;
      const s_t residual_tmp128 = -residual_tmp127 - residual_tmp59;
      const s_t residual_tmp129 = residual_tmp60*u_dt_shift;
      const s_t residual_tmp130 = residual_tmp36*u1_grad_0;
      const s_t residual_tmp131 = residual_tmp64*u1_grad_2;
      const s_t residual_tmp132 = residual_tmp129 + residual_tmp130 - residual_tmp131;
      const s_t residual_tmp133 = residual_tmp69*residual_tmp8;
      const s_t residual_tmp134 = residual_tmp27*u2_grad_0;
      const s_t residual_tmp135 = residual_tmp133 - residual_tmp134;
      const s_t residual_tmp136 = residual_tmp49*(residual_tmp132 + residual_tmp135);
      const s_t residual_tmp137 = -residual_tmp133 + residual_tmp134;
      const s_t residual_tmp138 = -residual_tmp130 + residual_tmp131;
      const s_t residual_tmp139 = residual_tmp136 + residual_tmp51*(s_t(2)*residual_tmp129 + residual_tmp137 + residual_tmp138);
      const s_t residual_tmp140 = residual_tmp57*u_dt_shift;
      const s_t residual_tmp141 = residual_tmp38*u1_grad_0;
      const s_t residual_tmp142 = residual_tmp55*u1_grad_2;
      const s_t residual_tmp143 = residual_tmp141 - residual_tmp142;
      const s_t residual_tmp144 = residual_tmp140 + residual_tmp143;
      const s_t residual_tmp145 = residual_tmp68*u_dt_shift;
      const s_t residual_tmp146 = residual_tmp38*u2_grad_0;
      const s_t residual_tmp147 = residual_tmp55*residual_tmp8;
      const s_t residual_tmp148 = residual_tmp146 - residual_tmp147;
      const s_t residual_tmp149 = -residual_tmp145 - residual_tmp148;
      const s_t residual_tmp150 = residual_tmp65*u1_grad_2;
      const s_t residual_tmp151 = -residual_tmp150;
      const s_t residual_tmp152 = residual_tmp70*residual_tmp8;
      const s_t residual_tmp153 = residual_tmp124*u1_grad_2;
      const s_t residual_tmp154 = residual_tmp122*residual_tmp8;
      const s_t residual_tmp155 = residual_tmp153 + residual_tmp154;
      const s_t residual_tmp156 = residual_tmp2*u2_grad_0;
      const s_t residual_tmp157 = -residual_tmp156 + residual_tmp61;
      const s_t residual_tmp158 = residual_tmp62*u_dt_shift;
      const s_t residual_tmp159 = residual_tmp2*residual_tmp64;
      const s_t residual_tmp160 = residual_tmp34*u1_grad_0;
      const s_t residual_tmp161 = residual_tmp158 + residual_tmp159 - residual_tmp160;
      const s_t residual_tmp162 = residual_tmp29*u2_grad_0;
      const s_t residual_tmp163 = residual_tmp69*u2_grad_1;
      const s_t residual_tmp164 = residual_tmp162 - residual_tmp163;
      const s_t residual_tmp165 = residual_tmp49*(residual_tmp161 + residual_tmp164);
      const s_t residual_tmp166 = -residual_tmp162 + residual_tmp163;
      const s_t residual_tmp167 = -residual_tmp159 + residual_tmp160;
      const s_t residual_tmp168 = residual_tmp165 + residual_tmp51*(s_t(2)*residual_tmp158 + residual_tmp166 + residual_tmp167);
      const s_t residual_tmp169 = residual_tmp67*u_dt_shift;
      const s_t residual_tmp170 = residual_tmp39*u2_grad_0;
      const s_t residual_tmp171 = residual_tmp55*u2_grad_1;
      const s_t residual_tmp172 = residual_tmp170 - residual_tmp171;
      const s_t residual_tmp173 = residual_tmp169 + residual_tmp172;
      const s_t residual_tmp174 = residual_tmp63*u_dt_shift;
      const s_t residual_tmp175 = residual_tmp39*u1_grad_0;
      const s_t residual_tmp176 = residual_tmp2*residual_tmp55;
      const s_t residual_tmp177 = residual_tmp175 - residual_tmp176;
      const s_t residual_tmp178 = -residual_tmp174 - residual_tmp177;
      const s_t residual_tmp179 = residual_tmp70*u2_grad_1;
      const s_t residual_tmp180 = -residual_tmp179;
      const s_t residual_tmp181 = residual_tmp2*residual_tmp65;
      const s_t residual_tmp182 = u0_grad_0 + s_t(1);
      const s_t residual_tmp183 = residual_tmp182*u1_grad_2;
      const s_t residual_tmp184 = s_t(6)*u2_grad_1;
      const s_t residual_tmp185 = s_t(2)*residual_tmp56;
      const s_t residual_tmp186 = residual_tmp184 - residual_tmp185;
      const s_t residual_tmp187 = residual_tmp182*u2_grad_1;
      const s_t residual_tmp188 = -residual_tmp187 + residual_tmp66;
      const s_t residual_tmp189 = lmbda*(-residual_tmp11*residual_tmp182 - residual_tmp18*residual_tmp8 + residual_tmp182*residual_tmp2*residual_tmp8 - residual_tmp19*residual_tmp2 + u0_grad_1*u1_grad_2*u2_grad_0 + u0_grad_2*u1_grad_0*u2_grad_1 + s_t(-1));
      const s_t residual_tmp190 = residual_tmp189*u2_grad_1;
      const s_t residual_tmp191 = -residual_tmp190;
      const s_t residual_tmp192 = residual_tmp182*residual_tmp34;
      const s_t residual_tmp193 = residual_tmp64*u0_grad_1;
      const s_t residual_tmp194 = residual_tmp169 + residual_tmp192 - residual_tmp193;
      const s_t residual_tmp195 = -residual_tmp170 + residual_tmp171;
      const s_t residual_tmp196 = residual_tmp49*(residual_tmp194 + residual_tmp195);
      const s_t residual_tmp197 = residual_tmp196 + residual_tmp51*(-s_t(2)*residual_tmp170 - residual_tmp194 + s_t(2)*residual_tmp55*u2_grad_1);
      const s_t residual_tmp198 = residual_tmp158 + residual_tmp166;
      const s_t residual_tmp199 = residual_tmp34*u2_grad_0;
      const s_t residual_tmp200 = residual_tmp64*u2_grad_1;
      const s_t residual_tmp201 = -residual_tmp182*residual_tmp39 + residual_tmp55*u0_grad_1;
      const s_t residual_tmp202 = -residual_tmp199 + residual_tmp200 - residual_tmp201;
      const s_t residual_tmp203 = residual_tmp65*u0_grad_1;
      const s_t residual_tmp204 = ((s_t(1) / s_t(3)))*residual_tmp84;
      const s_t residual_tmp205 = s_t(6)*u1_grad_2;
      const s_t residual_tmp206 = s_t(2)*residual_tmp66;
      const s_t residual_tmp207 = residual_tmp205 - residual_tmp206;
      const s_t residual_tmp208 = -residual_tmp183 + residual_tmp56;
      const s_t residual_tmp209 = residual_tmp189*u1_grad_2;
      const s_t residual_tmp210 = -residual_tmp209;
      const s_t residual_tmp211 = residual_tmp182*residual_tmp27;
      const s_t residual_tmp212 = residual_tmp69*u0_grad_2;
      const s_t residual_tmp213 = residual_tmp140 + residual_tmp211 - residual_tmp212;
      const s_t residual_tmp214 = -residual_tmp141 + residual_tmp142;
      const s_t residual_tmp215 = residual_tmp49*(residual_tmp213 + residual_tmp214);
      const s_t residual_tmp216 = residual_tmp215 + residual_tmp51*(-s_t(2)*residual_tmp141 - residual_tmp213 + s_t(2)*residual_tmp55*u1_grad_2);
      const s_t residual_tmp217 = residual_tmp129 + residual_tmp138;
      const s_t residual_tmp218 = -residual_tmp182*residual_tmp38 + residual_tmp55*u0_grad_2;
      const s_t residual_tmp219 = residual_tmp27*u1_grad_0 - residual_tmp69*u1_grad_2;
      const s_t residual_tmp220 = -residual_tmp218 - residual_tmp219;
      const s_t residual_tmp221 = residual_tmp70*u0_grad_2;
      const s_t residual_tmp222 = s_t(2)*u1_grad_1 + s_t(2);
      const s_t residual_tmp223 = s_t(2)*residual_tmp18;
      const s_t residual_tmp224 = s_t(6)*u2_grad_2 + s_t(6);
      const s_t residual_tmp225 = residual_tmp223 + residual_tmp224;
      const s_t residual_tmp226 = residual_tmp182*residual_tmp8 - residual_tmp19;
      const s_t residual_tmp227 = s_t(2)*u2_grad_2 + s_t(2);
      const s_t residual_tmp228 = ((s_t(1) / s_t(2)))*residual_tmp189;
      const s_t residual_tmp229 = residual_tmp227*residual_tmp228;
      const s_t residual_tmp230 = -residual_tmp68;
      const s_t residual_tmp231 = residual_tmp182*residual_tmp36;
      const s_t residual_tmp232 = residual_tmp64*u0_grad_2;
      const s_t residual_tmp233 = residual_tmp145 + residual_tmp231 - residual_tmp232;
      const s_t residual_tmp234 = -residual_tmp146 + residual_tmp147;
      const s_t residual_tmp235 = residual_tmp49*(-residual_tmp233 - residual_tmp234);
      const s_t residual_tmp236 = residual_tmp235 + residual_tmp51*(s_t(2)*residual_tmp146 - s_t(2)*residual_tmp147 + residual_tmp233);
      const s_t residual_tmp237 = residual_tmp129 + residual_tmp137;
      const s_t residual_tmp238 = residual_tmp36*u2_grad_0;
      const s_t residual_tmp239 = residual_tmp64*residual_tmp8;
      const s_t residual_tmp240 = residual_tmp218 + residual_tmp238 - residual_tmp239;
      const s_t residual_tmp241 = residual_tmp65*u0_grad_2;
      const s_t residual_tmp242 = s_t(2)*residual_tmp19;
      const s_t residual_tmp243 = s_t(6)*u1_grad_1 + s_t(6);
      const s_t residual_tmp244 = residual_tmp242 + residual_tmp243;
      const s_t residual_tmp245 = -residual_tmp18 + residual_tmp182*residual_tmp2;
      const s_t residual_tmp246 = residual_tmp222*residual_tmp228;
      const s_t residual_tmp247 = -residual_tmp63;
      const s_t residual_tmp248 = residual_tmp182*residual_tmp29;
      const s_t residual_tmp249 = residual_tmp69*u0_grad_1;
      const s_t residual_tmp250 = residual_tmp174 + residual_tmp248 - residual_tmp249;
      const s_t residual_tmp251 = -residual_tmp175 + residual_tmp176;
      const s_t residual_tmp252 = residual_tmp49*(-residual_tmp250 - residual_tmp251);
      const s_t residual_tmp253 = residual_tmp252 + residual_tmp51*(s_t(2)*residual_tmp175 - s_t(2)*residual_tmp176 + residual_tmp250);
      const s_t residual_tmp254 = residual_tmp158 + residual_tmp167;
      const s_t residual_tmp255 = -residual_tmp2*residual_tmp69 + residual_tmp29*u1_grad_0;
      const s_t residual_tmp256 = residual_tmp201 + residual_tmp255;
      const s_t residual_tmp257 = residual_tmp70*u0_grad_1;
      const s_t residual_tmp258 = residual_tmp122*residual_tmp182;
      const s_t residual_tmp259 = mu*(-residual_tmp258 - residual_tmp87);
      const s_t residual_tmp260 = -s_t(2)*residual_tmp58;
      const s_t residual_tmp261 = s_t(2)*residual_tmp127;
      const s_t residual_tmp262 = residual_tmp14*(-residual_tmp260 - residual_tmp261);
      const s_t residual_tmp263 = residual_tmp54*(-eta_s*(-residual_tmp57*residual_tmp65 + residual_tmp68*residual_tmp70) + ((s_t(1) / s_t(3)))*residual_tmp60*residual_tmp83);
      const s_t residual_tmp264 = s_t(2)*pow_2(u1_grad_0);
      const s_t residual_tmp265 = s_t(2)*pow_2(u2_grad_0);
      const s_t residual_tmp266 = residual_tmp264 + residual_tmp265;
      const s_t residual_tmp267 = residual_tmp124*residual_tmp182;
      const s_t residual_tmp268 = mu*(-residual_tmp1 - residual_tmp267);
      const s_t residual_tmp269 = s_t(2)*u1_grad_2;
      const s_t residual_tmp270 = residual_tmp2*residual_tmp269;
      const s_t residual_tmp271 = s_t(2)*u2_grad_1;
      const s_t residual_tmp272 = residual_tmp271*residual_tmp8;
      const s_t residual_tmp273 = mu*(-residual_tmp270 - residual_tmp272);
      const s_t residual_tmp274 = residual_tmp65*u1_grad_0;
      const s_t residual_tmp275 = residual_tmp70*u2_grad_0;
      const s_t residual_tmp276 = -residual_tmp275;
      const s_t residual_tmp277 = residual_tmp274 + residual_tmp276;
      const s_t residual_tmp278 = s_t(2)*u0_grad_0 + s_t(2);
      const s_t residual_tmp279 = s_t(4)*residual_tmp182;
      const s_t residual_tmp280 = -residual_tmp70*residual_tmp8;
      const s_t residual_tmp281 = residual_tmp182*residual_tmp65;
      const s_t residual_tmp282 = ((s_t(1) / s_t(3)))*residual_tmp83;
      const s_t residual_tmp283 = residual_tmp282*u2_grad_0;
      const s_t residual_tmp284 = s_t(6)*u2_grad_0;
      const s_t residual_tmp285 = s_t(2)*residual_tmp89;
      const s_t residual_tmp286 = residual_tmp189*u2_grad_0;
      const s_t residual_tmp287 = mu*(-residual_tmp284 - residual_tmp285 + s_t(4)*u0_grad_1*u1_grad_2) + residual_tmp286;
      const s_t residual_tmp288 = s_t(2)*residual_tmp187;
      const s_t residual_tmp289 = mu*(-residual_tmp205 - residual_tmp288 + s_t(4)*u0_grad_1*u2_grad_0) + residual_tmp209;
      const s_t residual_tmp290 = ((s_t(1) / s_t(3)))*residual_tmp60;
      const s_t residual_tmp291 = -s_t(2)*residual_tmp182*residual_tmp2;
      const s_t residual_tmp292 = mu*(s_t(4)*residual_tmp18 + residual_tmp224 + residual_tmp291) - residual_tmp227*residual_tmp228;
      const s_t residual_tmp293 = -s_t(2)*residual_tmp6;
      const s_t residual_tmp294 = s_t(6)*u1_grad_0;
      const s_t residual_tmp295 = residual_tmp293 + residual_tmp294;
      const s_t residual_tmp296 = residual_tmp189*u1_grad_0;
      const s_t residual_tmp297 = -residual_tmp296;
      const s_t residual_tmp298 = residual_tmp182*residual_tmp70;
      const s_t residual_tmp299 = residual_tmp282*u1_grad_0;
      const s_t residual_tmp300 = mu*(-residual_tmp267 - residual_tmp4);
      const s_t residual_tmp301 = s_t(2)*residual_tmp61;
      const s_t residual_tmp302 = s_t(2)*residual_tmp156;
      const s_t residual_tmp303 = residual_tmp14*(residual_tmp301 - residual_tmp302);
      const s_t residual_tmp304 = residual_tmp54*(-eta_s*(residual_tmp63*residual_tmp65 - residual_tmp67*residual_tmp70) + ((s_t(1) / s_t(3)))*residual_tmp62*residual_tmp83);
      const s_t residual_tmp305 = mu*(-residual_tmp258 - residual_tmp86);
      const s_t residual_tmp306 = -residual_tmp2*residual_tmp65;
      const s_t residual_tmp307 = s_t(2)*residual_tmp9;
      const s_t residual_tmp308 = mu*(-residual_tmp294 - residual_tmp307 + s_t(4)*u0_grad_2*u2_grad_1) + residual_tmp296;
      const s_t residual_tmp309 = s_t(2)*residual_tmp183;
      const s_t residual_tmp310 = mu*(-residual_tmp184 - residual_tmp309 + s_t(4)*u0_grad_2*u1_grad_0) + residual_tmp190;
      const s_t residual_tmp311 = ((s_t(1) / s_t(3)))*residual_tmp62;
      const s_t residual_tmp312 = -s_t(2)*residual_tmp182*residual_tmp8;
      const s_t residual_tmp313 = mu*(s_t(4)*residual_tmp19 + residual_tmp243 + residual_tmp312) - residual_tmp222*residual_tmp228;
      const s_t residual_tmp314 = s_t(2)*residual_tmp17;
      const s_t residual_tmp315 = residual_tmp284 - residual_tmp314;
      const s_t residual_tmp316 = -residual_tmp286;
      const s_t residual_tmp317 = residual_tmp269*residual_tmp8;
      const s_t residual_tmp318 = residual_tmp2*residual_tmp271;
      const s_t residual_tmp319 = mu*(-residual_tmp317 - residual_tmp318);
      const s_t residual_tmp320 = residual_tmp14*(-residual_tmp293 - residual_tmp307);
      const s_t residual_tmp321 = -residual_tmp43 + residual_tmp44;
      const s_t residual_tmp322 = residual_tmp321 + residual_tmp42;
      const s_t residual_tmp323 = residual_tmp104 + residual_tmp51*(-residual_tmp103 + s_t(2)*residual_tmp29*u0_grad_2 - residual_tmp97 - s_t(2)*residual_tmp99);
      const s_t residual_tmp324 = residual_tmp25*residual_tmp64 - residual_tmp27*residual_tmp63 + residual_tmp29*residual_tmp57 + residual_tmp33*residual_tmp69 - residual_tmp34*residual_tmp68 + residual_tmp36*residual_tmp67;
      const s_t residual_tmp325 = residual_tmp51*(s_t(2)*residual_tmp25*residual_tmp69 + s_t(2)*residual_tmp27*residual_tmp67 - residual_tmp73 - residual_tmp74 - s_t(2)*residual_tmp75 - residual_tmp78 - residual_tmp81) + residual_tmp82;
      const s_t residual_tmp326 = residual_tmp54*(-eta_s*(residual_tmp20*residual_tmp70 - residual_tmp324*residual_tmp33) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp325);
      const s_t residual_tmp327 = s_t(2)*pow_2(u0_grad_2);
      const s_t residual_tmp328 = residual_tmp107 + residual_tmp327;
      const s_t residual_tmp329 = s_t(2)*pow_2(u0_grad_1);
      const s_t residual_tmp330 = residual_tmp109 + residual_tmp329;
      const s_t residual_tmp331 = -residual_tmp98 + residual_tmp99;
      const s_t residual_tmp332 = residual_tmp331 + residual_tmp97;
      const s_t residual_tmp333 = residual_tmp50 + residual_tmp51*(residual_tmp115 + residual_tmp321 + s_t(2)*residual_tmp42);
      const s_t residual_tmp334 = -residual_tmp35 + residual_tmp37 + residual_tmp95;
      const s_t residual_tmp335 = residual_tmp119 + residual_tmp51*(residual_tmp117 + s_t(2)*residual_tmp28 - s_t(2)*residual_tmp30);
      const s_t residual_tmp336 = residual_tmp0*residual_tmp182;
      const s_t residual_tmp337 = mu*(-residual_tmp154 - residual_tmp336);
      const s_t residual_tmp338 = -residual_tmp192 + residual_tmp193;
      const s_t residual_tmp339 = residual_tmp196 + residual_tmp51*(s_t(2)*residual_tmp169 + residual_tmp172 + residual_tmp338);
      const s_t residual_tmp340 = -residual_tmp248 + residual_tmp249;
      const s_t residual_tmp341 = -residual_tmp174 - residual_tmp340;
      const s_t residual_tmp342 = residual_tmp324*u0_grad_1;
      const s_t residual_tmp343 = residual_tmp180 + residual_tmp342;
      const s_t residual_tmp344 = s_t(4)*residual_tmp2;
      const s_t residual_tmp345 = residual_tmp182*residual_tmp3;
      const s_t residual_tmp346 = residual_tmp123 + residual_tmp345;
      const s_t residual_tmp347 = -residual_tmp231 + residual_tmp232;
      const s_t residual_tmp348 = residual_tmp235 + residual_tmp51*(-s_t(2)*residual_tmp145 - residual_tmp148 - residual_tmp347);
      const s_t residual_tmp349 = -residual_tmp211 + residual_tmp212;
      const s_t residual_tmp350 = residual_tmp140 + residual_tmp349;
      const s_t residual_tmp351 = residual_tmp324*u0_grad_2;
      const s_t residual_tmp352 = residual_tmp165 + residual_tmp51*(-residual_tmp161 - s_t(2)*residual_tmp163 + s_t(2)*residual_tmp29*u2_grad_0);
      const s_t residual_tmp353 = residual_tmp199 - residual_tmp200 - residual_tmp255;
      const s_t residual_tmp354 = residual_tmp2*residual_tmp324;
      const s_t residual_tmp355 = ((s_t(1) / s_t(3)))*residual_tmp325;
      const s_t residual_tmp356 = residual_tmp355*u2_grad_1;
      const s_t residual_tmp357 = residual_tmp215 + residual_tmp51*(-residual_tmp140 + s_t(2)*residual_tmp182*residual_tmp27 - s_t(2)*residual_tmp212 - residual_tmp214);
      const s_t residual_tmp358 = -residual_tmp145 - residual_tmp347;
      const s_t residual_tmp359 = residual_tmp70*u1_grad_2;
      const s_t residual_tmp360 = s_t(6)*u0_grad_2;
      const s_t residual_tmp361 = residual_tmp189*u0_grad_2;
      const s_t residual_tmp362 = mu*(-residual_tmp302 - residual_tmp360 + s_t(4)*u1_grad_0*u2_grad_1) + residual_tmp361;
      const s_t residual_tmp363 = residual_tmp136 + residual_tmp51*(-residual_tmp132 - s_t(2)*residual_tmp134 + s_t(2)*residual_tmp69*residual_tmp8);
      const s_t residual_tmp364 = ((s_t(1) / s_t(3)))*residual_tmp25;
      const s_t residual_tmp365 = residual_tmp219 - residual_tmp238 + residual_tmp239;
      const s_t residual_tmp366 = residual_tmp324*u1_grad_2;
      const s_t residual_tmp367 = s_t(6)*u0_grad_1;
      const s_t residual_tmp368 = residual_tmp260 + residual_tmp367;
      const s_t residual_tmp369 = residual_tmp189*u0_grad_1;
      const s_t residual_tmp370 = -residual_tmp369;
      const s_t residual_tmp371 = residual_tmp252 + residual_tmp51*(residual_tmp174 - s_t(2)*residual_tmp248 + s_t(2)*residual_tmp249 + residual_tmp251);
      const s_t residual_tmp372 = residual_tmp169 + residual_tmp338;
      const s_t residual_tmp373 = residual_tmp2*residual_tmp70;
      const s_t residual_tmp374 = residual_tmp355*u0_grad_1;
      const s_t residual_tmp375 = residual_tmp14*(-residual_tmp242 - residual_tmp312);
      const s_t residual_tmp376 = ((s_t(1) / s_t(3)))*residual_tmp68;
      const s_t residual_tmp377 = -(s_t(1) / s_t(3))*residual_tmp325;
      const s_t residual_tmp378 = residual_tmp54*(eta_s*(residual_tmp324*residual_tmp57 + residual_tmp60*residual_tmp70) + residual_tmp377*residual_tmp68);
      const s_t residual_tmp379 = residual_tmp124*u2_grad_0;
      const s_t residual_tmp380 = mu*(-residual_tmp317 - residual_tmp379);
      const s_t residual_tmp381 = s_t(2)*pow_2(residual_tmp182);
      const s_t residual_tmp382 = residual_tmp265 + residual_tmp381;
      const s_t residual_tmp383 = residual_tmp3*u0_grad_2;
      const s_t residual_tmp384 = residual_tmp272 + residual_tmp383;
      const s_t residual_tmp385 = residual_tmp182*residual_tmp324;
      const s_t residual_tmp386 = residual_tmp324*u1_grad_0;
      const s_t residual_tmp387 = -residual_tmp301 + residual_tmp360;
      const s_t residual_tmp388 = -residual_tmp361;
      const s_t residual_tmp389 = s_t(6)*u0_grad_0 + s_t(6);
      const s_t residual_tmp390 = residual_tmp12 + residual_tmp389;
      const s_t residual_tmp391 = residual_tmp228*residual_tmp278;
      const s_t residual_tmp392 = residual_tmp70*u1_grad_0;
      const s_t residual_tmp393 = residual_tmp14*(residual_tmp206 - residual_tmp288);
      const s_t residual_tmp394 = residual_tmp54*(-eta_s*(residual_tmp324*residual_tmp63 - residual_tmp62*residual_tmp70) + ((s_t(1) / s_t(3)))*residual_tmp325*residual_tmp67);
      const s_t residual_tmp395 = mu*(-residual_tmp318 - residual_tmp379);
      const s_t residual_tmp396 = -residual_tmp182*residual_tmp324;
      const s_t residual_tmp397 = mu*(-residual_tmp261 - residual_tmp367 + s_t(4)*u1_grad_2*u2_grad_0) + residual_tmp369;
      const s_t residual_tmp398 = ((s_t(1) / s_t(3)))*residual_tmp67;
      const s_t residual_tmp399 = mu*(s_t(4)*residual_tmp11 + residual_tmp13 + residual_tmp389) - residual_tmp228*residual_tmp278;
      const s_t residual_tmp400 = residual_tmp14*(-residual_tmp285 + residual_tmp314);
      const s_t residual_tmp401 = residual_tmp50 + residual_tmp51*(s_t(2)*residual_tmp36*u0_grad_1 - residual_tmp42 - s_t(2)*residual_tmp44 - residual_tmp48);
      const s_t residual_tmp402 = residual_tmp51*(s_t(2)*residual_tmp33*residual_tmp64 + s_t(2)*residual_tmp34*residual_tmp57 - residual_tmp71 - residual_tmp72 - residual_tmp76 - s_t(2)*residual_tmp77 - residual_tmp81) + residual_tmp82;
      const s_t residual_tmp403 = residual_tmp54*(-eta_s*(residual_tmp20*residual_tmp65 - residual_tmp25*residual_tmp324) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp402);
      const s_t residual_tmp404 = residual_tmp106 + residual_tmp327 + s_t(2);
      const s_t residual_tmp405 = residual_tmp110 + residual_tmp329;
      const s_t residual_tmp406 = residual_tmp104 + residual_tmp51*(residual_tmp113 + residual_tmp331 + s_t(2)*residual_tmp97);
      const s_t residual_tmp407 = residual_tmp119 + residual_tmp51*(residual_tmp118 + residual_tmp26 + s_t(2)*residual_tmp91 - s_t(2)*residual_tmp92);
      const s_t residual_tmp408 = mu*(-residual_tmp125 - residual_tmp345);
      const s_t residual_tmp409 = residual_tmp215 + residual_tmp51*(s_t(2)*residual_tmp140 + residual_tmp143 + residual_tmp349);
      const s_t residual_tmp410 = residual_tmp151 + residual_tmp351;
      const s_t residual_tmp411 = s_t(4)*residual_tmp8;
      const s_t residual_tmp412 = residual_tmp153 + residual_tmp336;
      const s_t residual_tmp413 = residual_tmp252 + residual_tmp51*(-s_t(2)*residual_tmp174 - residual_tmp177 - residual_tmp340);
      const s_t residual_tmp414 = residual_tmp136 + residual_tmp51*(-residual_tmp129 - s_t(2)*residual_tmp131 - residual_tmp135 + s_t(2)*residual_tmp36*u1_grad_0);
      const s_t residual_tmp415 = residual_tmp324*residual_tmp8;
      const s_t residual_tmp416 = ((s_t(1) / s_t(3)))*residual_tmp402;
      const s_t residual_tmp417 = residual_tmp416*u1_grad_2;
      const s_t residual_tmp418 = residual_tmp196 + residual_tmp51*(-residual_tmp169 + s_t(2)*residual_tmp182*residual_tmp34 - s_t(2)*residual_tmp193 - residual_tmp195);
      const s_t residual_tmp419 = residual_tmp65*u2_grad_1;
      const s_t residual_tmp420 = residual_tmp165 + residual_tmp51*(-residual_tmp158 - s_t(2)*residual_tmp160 - residual_tmp164 + s_t(2)*residual_tmp2*residual_tmp64);
      const s_t residual_tmp421 = ((s_t(1) / s_t(3)))*residual_tmp33;
      const s_t residual_tmp422 = residual_tmp324*u2_grad_1;
      const s_t residual_tmp423 = residual_tmp235 + residual_tmp51*(residual_tmp145 - s_t(2)*residual_tmp231 + s_t(2)*residual_tmp232 + residual_tmp234);
      const s_t residual_tmp424 = residual_tmp65*residual_tmp8;
      const s_t residual_tmp425 = residual_tmp416*u0_grad_2;
      const s_t residual_tmp426 = residual_tmp14*(residual_tmp185 - residual_tmp309);
      const s_t residual_tmp427 = residual_tmp54*(-eta_s*(residual_tmp324*residual_tmp68 - residual_tmp60*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp402*residual_tmp57);
      const s_t residual_tmp428 = residual_tmp264 + residual_tmp381;
      const s_t residual_tmp429 = residual_tmp270 + residual_tmp383;
      const s_t residual_tmp430 = residual_tmp324*u2_grad_0;
      const s_t residual_tmp431 = ((s_t(1) / s_t(3)))*residual_tmp57;
      const s_t residual_tmp432 = residual_tmp65*u2_grad_0;
      const s_t residual_tmp433 = residual_tmp14*(-residual_tmp223 - residual_tmp291);
      const s_t residual_tmp434 = ((s_t(1) / s_t(3)))*residual_tmp63;
      const s_t residual_tmp435 = -(s_t(1) / s_t(3))*residual_tmp402;
      const s_t residual_tmp436 = residual_tmp54*(eta_s*(residual_tmp324*residual_tmp67 + residual_tmp62*residual_tmp65) + residual_tmp435*residual_tmp63);
      const s_t grad_coeff0_0 = u0_direction_grad_0*(mu*(residual_tmp108 + residual_tmp111) + residual_tmp112*residual_tmp15 + residual_tmp121*residual_tmp85 + residual_tmp24*(eta_s*(residual_tmp114*residual_tmp33 + residual_tmp116*residual_tmp25) - residual_tmp120*residual_tmp53)) + u0_direction_grad_1*(-mu*residual_tmp126 + residual_tmp128*residual_tmp15 + residual_tmp24*(eta_s*(residual_tmp144*residual_tmp33 + residual_tmp149*residual_tmp25 + residual_tmp151 + residual_tmp152) - residual_tmp139*residual_tmp53) + residual_tmp60*residual_tmp85) + u0_direction_grad_2*(-mu*residual_tmp155 + residual_tmp15*residual_tmp157 + residual_tmp24*(eta_s*(residual_tmp173*residual_tmp25 + residual_tmp178*residual_tmp33 + residual_tmp180 + residual_tmp181) - residual_tmp168*residual_tmp53) + residual_tmp62*residual_tmp85) + u1_direction_grad_0*(residual_tmp10*residual_tmp15 + residual_tmp24*(eta_s*(residual_tmp25*residual_tmp32 + residual_tmp33*residual_tmp41) - residual_tmp52*residual_tmp53) + residual_tmp25*residual_tmp85 + residual_tmp5) + u1_direction_grad_1*(mu*(s_t(2)*residual_tmp182*residual_tmp222 - residual_tmp225) + residual_tmp15*residual_tmp226 + residual_tmp229 + residual_tmp230*residual_tmp85 + residual_tmp24*(eta_s*(residual_tmp237*residual_tmp25 + residual_tmp240*residual_tmp33 + residual_tmp241) + residual_tmp204*residual_tmp8 - residual_tmp236*residual_tmp53)) + u1_direction_grad_2*(mu*(s_t(4)*residual_tmp183 + residual_tmp186) + residual_tmp15*residual_tmp188 + residual_tmp191 + residual_tmp24*(eta_s*(residual_tmp198*residual_tmp25 + residual_tmp202*residual_tmp33 - residual_tmp203) - residual_tmp197*residual_tmp53 - residual_tmp204*u2_grad_1) + residual_tmp67*residual_tmp85) + u2_direction_grad_0*(residual_tmp15*residual_tmp90 + residual_tmp24*(eta_s*(residual_tmp25*residual_tmp96 + residual_tmp33*residual_tmp94) - residual_tmp105*residual_tmp53) + residual_tmp33*residual_tmp85 + residual_tmp88) + u2_direction_grad_1*(mu*(s_t(4)*residual_tmp187 + residual_tmp207) + residual_tmp15*residual_tmp208 + residual_tmp210 + residual_tmp24*(eta_s*(residual_tmp217*residual_tmp33 + residual_tmp220*residual_tmp25 - residual_tmp221) - residual_tmp204*u1_grad_2 - residual_tmp216*residual_tmp53) + residual_tmp57*residual_tmp85) + u2_direction_grad_2*(mu*(s_t(2)*residual_tmp182*residual_tmp227 - residual_tmp244) + residual_tmp15*residual_tmp245 + residual_tmp24*(eta_s*(residual_tmp25*residual_tmp256 + residual_tmp254*residual_tmp33 + residual_tmp257) + residual_tmp2*residual_tmp204 - residual_tmp253*residual_tmp53) + residual_tmp246 + residual_tmp247*residual_tmp85);
      const s_t grad_coeff0_1 = u0_direction_grad_0*(mu*(-residual_tmp126 + s_t(2)*residual_tmp278*u0_grad_1 - residual_tmp279*u0_grad_1) + residual_tmp112*residual_tmp262 + residual_tmp121*residual_tmp263 + residual_tmp24*(-eta_s*(-residual_tmp114*residual_tmp57 + residual_tmp116*residual_tmp68 - residual_tmp150 - residual_tmp280) + ((s_t(1) / s_t(3)))*residual_tmp120*residual_tmp60)) + u0_direction_grad_1*(mu*(residual_tmp108 + residual_tmp266) + residual_tmp128*residual_tmp262 + residual_tmp24*(-eta_s*(-residual_tmp144*residual_tmp57 + residual_tmp149*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp139*residual_tmp60) + residual_tmp263*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp262 + residual_tmp24*(-eta_s*(residual_tmp173*residual_tmp68 - residual_tmp178*residual_tmp57 + residual_tmp277) + ((s_t(1) / s_t(3)))*residual_tmp168*residual_tmp60) + residual_tmp263*residual_tmp62 + residual_tmp273) + u1_direction_grad_0*(residual_tmp10*residual_tmp262 + residual_tmp24*(-eta_s*(residual_tmp241 + residual_tmp32*residual_tmp68 - residual_tmp41*residual_tmp57) + residual_tmp282*residual_tmp8 + residual_tmp290*residual_tmp52) + residual_tmp25*residual_tmp263 + residual_tmp292) + u1_direction_grad_1*(residual_tmp226*residual_tmp262 + residual_tmp230*residual_tmp263 + residual_tmp24*(-eta_s*(residual_tmp237*residual_tmp68 - residual_tmp240*residual_tmp57) + ((s_t(1) / s_t(3)))*residual_tmp236*residual_tmp60) + residual_tmp268) + u1_direction_grad_2*(residual_tmp188*residual_tmp262 + residual_tmp24*(-eta_s*(residual_tmp198*residual_tmp68 - residual_tmp202*residual_tmp57 - residual_tmp281) + ((s_t(1) / s_t(3)))*residual_tmp197*residual_tmp60 - residual_tmp283) + residual_tmp263*residual_tmp67 + residual_tmp287) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(-residual_tmp221 - residual_tmp57*residual_tmp94 + residual_tmp68*residual_tmp96) + ((s_t(1) / s_t(3)))*residual_tmp105*residual_tmp60 - residual_tmp282*u1_grad_2) + residual_tmp262*residual_tmp90 + residual_tmp263*residual_tmp33 + residual_tmp289) + u2_direction_grad_1*(residual_tmp208*residual_tmp262 + residual_tmp24*(-eta_s*(-residual_tmp217*residual_tmp57 + residual_tmp220*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp216*residual_tmp60) + residual_tmp259 + residual_tmp263*residual_tmp57) + u2_direction_grad_2*(mu*(residual_tmp227*residual_tmp3 + residual_tmp295) + residual_tmp24*(-eta_s*(-residual_tmp254*residual_tmp57 + residual_tmp256*residual_tmp68 + residual_tmp298) + residual_tmp253*residual_tmp290 + residual_tmp299) + residual_tmp245*residual_tmp262 + residual_tmp247*residual_tmp263 + residual_tmp297);
      const s_t grad_coeff0_2 = u0_direction_grad_0*(mu*(-residual_tmp155 + s_t(2)*residual_tmp278*u0_grad_2 - residual_tmp279*u0_grad_2) + residual_tmp112*residual_tmp303 + residual_tmp121*residual_tmp304 + residual_tmp24*(-eta_s*(residual_tmp114*residual_tmp63 - residual_tmp116*residual_tmp67 - residual_tmp179 - residual_tmp306) + ((s_t(1) / s_t(3)))*residual_tmp120*residual_tmp62)) + u0_direction_grad_1*(residual_tmp128*residual_tmp303 + residual_tmp24*(-eta_s*(residual_tmp144*residual_tmp63 - residual_tmp149*residual_tmp67 - residual_tmp277) + ((s_t(1) / s_t(3)))*residual_tmp139*residual_tmp62) + residual_tmp273 + residual_tmp304*residual_tmp60) + u0_direction_grad_2*(mu*(residual_tmp111 + residual_tmp266 + s_t(2)) + residual_tmp157*residual_tmp303 + residual_tmp24*(-eta_s*(-residual_tmp173*residual_tmp67 + residual_tmp178*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp168*residual_tmp62) + residual_tmp304*residual_tmp62) + u1_direction_grad_0*(residual_tmp10*residual_tmp303 + residual_tmp24*(-eta_s*(-residual_tmp203 - residual_tmp32*residual_tmp67 + residual_tmp41*residual_tmp63) - residual_tmp282*u2_grad_1 + ((s_t(1) / s_t(3)))*residual_tmp52*residual_tmp62) + residual_tmp25*residual_tmp304 + residual_tmp310) + u1_direction_grad_1*(mu*(residual_tmp0*residual_tmp222 + residual_tmp315) + residual_tmp226*residual_tmp303 + residual_tmp230*residual_tmp304 + residual_tmp24*(-eta_s*(-residual_tmp237*residual_tmp67 + residual_tmp240*residual_tmp63 + residual_tmp281) + residual_tmp236*residual_tmp311 + residual_tmp283) + residual_tmp316) + u1_direction_grad_2*(residual_tmp188*residual_tmp303 + residual_tmp24*(-eta_s*(-residual_tmp198*residual_tmp67 + residual_tmp202*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp197*residual_tmp62) + residual_tmp300 + residual_tmp304*residual_tmp67) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(residual_tmp257 + residual_tmp63*residual_tmp94 - residual_tmp67*residual_tmp96) + residual_tmp105*residual_tmp311 + residual_tmp2*residual_tmp282) + residual_tmp303*residual_tmp90 + residual_tmp304*residual_tmp33 + residual_tmp313) + u2_direction_grad_1*(residual_tmp208*residual_tmp303 + residual_tmp24*(-eta_s*(residual_tmp217*residual_tmp63 - residual_tmp220*residual_tmp67 - residual_tmp298) + ((s_t(1) / s_t(3)))*residual_tmp216*residual_tmp62 - residual_tmp299) + residual_tmp304*residual_tmp57 + residual_tmp308) + u2_direction_grad_2*(residual_tmp24*(-eta_s*(residual_tmp254*residual_tmp63 - residual_tmp256*residual_tmp67) + ((s_t(1) / s_t(3)))*residual_tmp253*residual_tmp62) + residual_tmp245*residual_tmp303 + residual_tmp247*residual_tmp304 + residual_tmp305);
      const s_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp112*residual_tmp320 + residual_tmp121*residual_tmp326 + residual_tmp24*(-eta_s*(residual_tmp116*residual_tmp20 - residual_tmp33*residual_tmp334) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp335) + residual_tmp5) + u0_direction_grad_1*(residual_tmp128*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp149*residual_tmp20 - residual_tmp33*residual_tmp365 + residual_tmp366) + residual_tmp355*residual_tmp8 + residual_tmp363*residual_tmp364) + residual_tmp292 + residual_tmp326*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp173*residual_tmp20 - residual_tmp33*residual_tmp353 - residual_tmp354) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp352 - residual_tmp356) + residual_tmp310 + residual_tmp326*residual_tmp62) + u1_direction_grad_0*(mu*(residual_tmp328 + residual_tmp330) + residual_tmp10*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp32 - residual_tmp33*residual_tmp332) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp333) + residual_tmp25*residual_tmp326) + u1_direction_grad_1*(mu*(s_t(2)*residual_tmp222*u1_grad_0 - residual_tmp344*u1_grad_0 - residual_tmp346) + residual_tmp226*residual_tmp320 + residual_tmp230*residual_tmp326 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp237 - residual_tmp280 - residual_tmp33*residual_tmp350 - residual_tmp351) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp348)) + u1_direction_grad_2*(residual_tmp188*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp198*residual_tmp20 - residual_tmp33*residual_tmp341 + residual_tmp343) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp339) + residual_tmp326*residual_tmp67 + residual_tmp337) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp96 - residual_tmp322*residual_tmp33) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp323) + residual_tmp319 + residual_tmp320*residual_tmp90 + residual_tmp326*residual_tmp33) + u2_direction_grad_1*(residual_tmp208*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp220 - residual_tmp33*residual_tmp358 - residual_tmp359) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp357 - residual_tmp355*u0_grad_2) + residual_tmp326*residual_tmp57 + residual_tmp362) + u2_direction_grad_2*(mu*(residual_tmp124*residual_tmp227 + residual_tmp368) + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp256 - residual_tmp33*residual_tmp372 + residual_tmp373) + residual_tmp364*residual_tmp371 + residual_tmp374) + residual_tmp245*residual_tmp320 + residual_tmp247*residual_tmp326 + residual_tmp370);
      const s_t grad_coeff1_1 = u0_direction_grad_0*(mu*(s_t(2)*residual_tmp2*residual_tmp278 - residual_tmp225) + residual_tmp112*residual_tmp375 + residual_tmp121*residual_tmp378 + residual_tmp229 + residual_tmp24*(eta_s*(residual_tmp116*residual_tmp60 + residual_tmp334*residual_tmp57 + residual_tmp366) - residual_tmp335*residual_tmp376 + residual_tmp377*residual_tmp8)) + u0_direction_grad_1*(residual_tmp128*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp149*residual_tmp60 + residual_tmp365*residual_tmp57) - residual_tmp363*residual_tmp376) + residual_tmp268 + residual_tmp378*residual_tmp60) + u0_direction_grad_2*(mu*(residual_tmp315 + s_t(4)*residual_tmp89) + residual_tmp157*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp173*residual_tmp60 + residual_tmp353*residual_tmp57 - residual_tmp386) - residual_tmp352*residual_tmp376 - residual_tmp377*u2_grad_0) + residual_tmp316 + residual_tmp378*residual_tmp62) + u1_direction_grad_0*(-mu*residual_tmp346 + residual_tmp10*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp152 + residual_tmp32*residual_tmp60 + residual_tmp332*residual_tmp57 - residual_tmp351) - residual_tmp333*residual_tmp376) + residual_tmp25*residual_tmp378) + u1_direction_grad_1*(mu*(residual_tmp328 + residual_tmp382) + residual_tmp226*residual_tmp375 + residual_tmp230*residual_tmp378 + residual_tmp24*(eta_s*(residual_tmp237*residual_tmp60 + residual_tmp350*residual_tmp57) - residual_tmp348*residual_tmp376)) + u1_direction_grad_2*(-mu*residual_tmp384 + residual_tmp188*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp198*residual_tmp60 + residual_tmp276 + residual_tmp341*residual_tmp57 + residual_tmp385) - residual_tmp339*residual_tmp376) + residual_tmp378*residual_tmp67) + u2_direction_grad_0*(mu*(s_t(4)*residual_tmp156 + residual_tmp387) + residual_tmp24*(eta_s*(residual_tmp322*residual_tmp57 - residual_tmp359 + residual_tmp60*residual_tmp96) - residual_tmp323*residual_tmp376 - residual_tmp377*u0_grad_2) + residual_tmp33*residual_tmp378 + residual_tmp375*residual_tmp90 + residual_tmp388) + u2_direction_grad_1*(residual_tmp208*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp220*residual_tmp60 + residual_tmp358*residual_tmp57) - residual_tmp357*residual_tmp376) + residual_tmp378*residual_tmp57 + residual_tmp380) + u2_direction_grad_2*(mu*(s_t(2)*residual_tmp2*residual_tmp227 - residual_tmp390) + residual_tmp24*(eta_s*(residual_tmp256*residual_tmp60 + residual_tmp372*residual_tmp57 + residual_tmp392) + residual_tmp182*residual_tmp377 - residual_tmp371*residual_tmp376) + residual_tmp245*residual_tmp375 + residual_tmp247*residual_tmp378 + residual_tmp391);
      const s_t grad_coeff1_2 = u0_direction_grad_0*(mu*(residual_tmp186 + residual_tmp269*residual_tmp278) + residual_tmp112*residual_tmp393 + residual_tmp121*residual_tmp394 + residual_tmp191 + residual_tmp24*(-eta_s*(-residual_tmp116*residual_tmp62 + residual_tmp334*residual_tmp63 + residual_tmp354) + residual_tmp335*residual_tmp398 + residual_tmp356)) + u0_direction_grad_1*(residual_tmp128*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp149*residual_tmp62 + residual_tmp365*residual_tmp63 - residual_tmp386) - residual_tmp355*u2_grad_0 + ((s_t(1) / s_t(3)))*residual_tmp363*residual_tmp67) + residual_tmp287 + residual_tmp394*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp173*residual_tmp62 + residual_tmp353*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp352*residual_tmp67) + residual_tmp300 + residual_tmp394*residual_tmp62) + u1_direction_grad_0*(residual_tmp10*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp32*residual_tmp62 + residual_tmp332*residual_tmp63 - residual_tmp343) + ((s_t(1) / s_t(3)))*residual_tmp333*residual_tmp67) + residual_tmp25*residual_tmp394 + residual_tmp337) + u1_direction_grad_1*(mu*(s_t(2)*residual_tmp222*u1_grad_2 - residual_tmp344*u1_grad_2 - residual_tmp384) + residual_tmp226*residual_tmp393 + residual_tmp230*residual_tmp394 + residual_tmp24*(-eta_s*(-residual_tmp237*residual_tmp62 - residual_tmp275 + residual_tmp350*residual_tmp63 - residual_tmp396) + ((s_t(1) / s_t(3)))*residual_tmp348*residual_tmp67)) + u1_direction_grad_2*(mu*(residual_tmp330 + residual_tmp382 + s_t(2)) + residual_tmp188*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp198*residual_tmp62 + residual_tmp341*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp339*residual_tmp67) + residual_tmp394*residual_tmp67) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(residual_tmp322*residual_tmp63 - residual_tmp373 - residual_tmp62*residual_tmp96) + ((s_t(1) / s_t(3)))*residual_tmp323*residual_tmp67 - residual_tmp374) + residual_tmp33*residual_tmp394 + residual_tmp393*residual_tmp90 + residual_tmp397) + u2_direction_grad_1*(residual_tmp208*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp220*residual_tmp62 + residual_tmp358*residual_tmp63 + residual_tmp392) + residual_tmp182*residual_tmp355 + residual_tmp357*residual_tmp398) + residual_tmp394*residual_tmp57 + residual_tmp399) + u2_direction_grad_2*(residual_tmp24*(-eta_s*(-residual_tmp256*residual_tmp62 + residual_tmp372*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp371*residual_tmp67) + residual_tmp245*residual_tmp393 + residual_tmp247*residual_tmp394 + residual_tmp395);
      const s_t grad_coeff2_0 = u0_direction_grad_0*(residual_tmp112*residual_tmp400 + residual_tmp121*residual_tmp403 + residual_tmp24*(-eta_s*(residual_tmp114*residual_tmp20 - residual_tmp25*residual_tmp334) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp407) + residual_tmp88) + u0_direction_grad_1*(residual_tmp128*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp144*residual_tmp20 - residual_tmp25*residual_tmp365 - residual_tmp415) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp414 - residual_tmp417) + residual_tmp289 + residual_tmp403*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp178*residual_tmp20 - residual_tmp25*residual_tmp353 + residual_tmp422) + residual_tmp2*residual_tmp416 + residual_tmp420*residual_tmp421) + residual_tmp313 + residual_tmp403*residual_tmp62) + u1_direction_grad_0*(residual_tmp10*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp41 - residual_tmp25*residual_tmp332) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp401) + residual_tmp25*residual_tmp403 + residual_tmp319) + u1_direction_grad_1*(mu*(residual_tmp122*residual_tmp222 + residual_tmp387) + residual_tmp226*residual_tmp400 + residual_tmp230*residual_tmp403 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp240 - residual_tmp25*residual_tmp350 + residual_tmp424) + residual_tmp421*residual_tmp423 + residual_tmp425) + residual_tmp388) + u1_direction_grad_2*(residual_tmp188*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp202 - residual_tmp25*residual_tmp341 - residual_tmp419) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp418 - residual_tmp416*u0_grad_1) + residual_tmp397 + residual_tmp403*residual_tmp67) + u2_direction_grad_0*(mu*(residual_tmp404 + residual_tmp405) + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp94 - residual_tmp25*residual_tmp322) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp406) + residual_tmp33*residual_tmp403 + residual_tmp400*residual_tmp90) + u2_direction_grad_1*(residual_tmp208*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp217 - residual_tmp25*residual_tmp358 + residual_tmp410) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp409) + residual_tmp403*residual_tmp57 + residual_tmp408) + u2_direction_grad_2*(mu*(s_t(2)*residual_tmp227*u2_grad_0 - residual_tmp411*u2_grad_0 - residual_tmp412) + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp254 - residual_tmp25*residual_tmp372 - residual_tmp306 - residual_tmp342) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp413) + residual_tmp245*residual_tmp400 + residual_tmp247*residual_tmp403);
      const s_t grad_coeff2_1 = u0_direction_grad_0*(mu*(residual_tmp207 + residual_tmp271*residual_tmp278) + residual_tmp112*residual_tmp426 + residual_tmp121*residual_tmp427 + residual_tmp210 + residual_tmp24*(-eta_s*(-residual_tmp114*residual_tmp60 + residual_tmp334*residual_tmp68 + residual_tmp415) + residual_tmp407*residual_tmp431 + residual_tmp417)) + u0_direction_grad_1*(residual_tmp128*residual_tmp426 + residual_tmp24*(-eta_s*(-residual_tmp144*residual_tmp60 + residual_tmp365*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp414*residual_tmp57) + residual_tmp259 + residual_tmp427*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp426 + residual_tmp24*(-eta_s*(-residual_tmp178*residual_tmp60 + residual_tmp353*residual_tmp68 - residual_tmp430) - residual_tmp416*u1_grad_0 + ((s_t(1) / s_t(3)))*residual_tmp420*residual_tmp57) + residual_tmp308 + residual_tmp427*residual_tmp62) + u1_direction_grad_0*(residual_tmp10*residual_tmp426 + residual_tmp24*(-eta_s*(residual_tmp332*residual_tmp68 - residual_tmp41*residual_tmp60 - residual_tmp424) + ((s_t(1) / s_t(3)))*residual_tmp401*residual_tmp57 - residual_tmp425) + residual_tmp25*residual_tmp427 + residual_tmp362) + u1_direction_grad_1*(residual_tmp226*residual_tmp426 + residual_tmp230*residual_tmp427 + residual_tmp24*(-eta_s*(-residual_tmp240*residual_tmp60 + residual_tmp350*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp423*residual_tmp57) + residual_tmp380) + u1_direction_grad_2*(residual_tmp188*residual_tmp426 + residual_tmp24*(-eta_s*(-residual_tmp202*residual_tmp60 + residual_tmp341*residual_tmp68 + residual_tmp432) + residual_tmp182*residual_tmp416 + residual_tmp418*residual_tmp431) + residual_tmp399 + residual_tmp427*residual_tmp67) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(residual_tmp322*residual_tmp68 - residual_tmp410 - residual_tmp60*residual_tmp94) + ((s_t(1) / s_t(3)))*residual_tmp406*residual_tmp57) + residual_tmp33*residual_tmp427 + residual_tmp408 + residual_tmp426*residual_tmp90) + u2_direction_grad_1*(mu*(residual_tmp404 + residual_tmp428) + residual_tmp208*residual_tmp426 + residual_tmp24*(-eta_s*(-residual_tmp217*residual_tmp60 + residual_tmp358*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp409*residual_tmp57) + residual_tmp427*residual_tmp57) + u2_direction_grad_2*(mu*(s_t(2)*residual_tmp227*u2_grad_1 - residual_tmp411*u2_grad_1 - residual_tmp429) + residual_tmp24*(-eta_s*(-residual_tmp254*residual_tmp60 - residual_tmp274 + residual_tmp372*residual_tmp68 - residual_tmp396) + ((s_t(1) / s_t(3)))*residual_tmp413*residual_tmp57) + residual_tmp245*residual_tmp426 + residual_tmp247*residual_tmp427);
      const s_t grad_coeff2_2 = u0_direction_grad_0*(mu*(-residual_tmp244 + s_t(2)*residual_tmp278*residual_tmp8) + residual_tmp112*residual_tmp433 + residual_tmp121*residual_tmp436 + residual_tmp24*(eta_s*(residual_tmp114*residual_tmp62 + residual_tmp334*residual_tmp67 + residual_tmp422) + residual_tmp2*residual_tmp435 - residual_tmp407*residual_tmp434) + residual_tmp246) + u0_direction_grad_1*(mu*(residual_tmp295 + s_t(4)*residual_tmp9) + residual_tmp128*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp144*residual_tmp62 + residual_tmp365*residual_tmp67 - residual_tmp430) - residual_tmp414*residual_tmp434 - residual_tmp435*u1_grad_0) + residual_tmp297 + residual_tmp436*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp178*residual_tmp62 + residual_tmp353*residual_tmp67) - residual_tmp420*residual_tmp434) + residual_tmp305 + residual_tmp436*residual_tmp62) + u1_direction_grad_0*(mu*(s_t(4)*residual_tmp127 + residual_tmp368) + residual_tmp10*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp332*residual_tmp67 + residual_tmp41*residual_tmp62 - residual_tmp419) - residual_tmp401*residual_tmp434 - residual_tmp435*u0_grad_1) + residual_tmp25*residual_tmp436 + residual_tmp370) + u1_direction_grad_1*(mu*(s_t(2)*residual_tmp222*residual_tmp8 - residual_tmp390) + residual_tmp226*residual_tmp433 + residual_tmp230*residual_tmp436 + residual_tmp24*(eta_s*(residual_tmp240*residual_tmp62 + residual_tmp350*residual_tmp67 + residual_tmp432) + residual_tmp182*residual_tmp435 - residual_tmp423*residual_tmp434) + residual_tmp391) + u1_direction_grad_2*(residual_tmp188*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp202*residual_tmp62 + residual_tmp341*residual_tmp67) - residual_tmp418*residual_tmp434) + residual_tmp395 + residual_tmp436*residual_tmp67) + u2_direction_grad_0*(-mu*residual_tmp412 + residual_tmp24*(eta_s*(residual_tmp181 + residual_tmp322*residual_tmp67 - residual_tmp342 + residual_tmp62*residual_tmp94) - residual_tmp406*residual_tmp434) + residual_tmp33*residual_tmp436 + residual_tmp433*residual_tmp90) + u2_direction_grad_1*(-mu*residual_tmp429 + residual_tmp208*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp217*residual_tmp62 - residual_tmp274 + residual_tmp358*residual_tmp67 + residual_tmp385) - residual_tmp409*residual_tmp434) + residual_tmp436*residual_tmp57) + u2_direction_grad_2*(mu*(residual_tmp405 + residual_tmp428 + s_t(2)) + residual_tmp24*(eta_s*(residual_tmp254*residual_tmp62 + residual_tmp372*residual_tmp67) - residual_tmp413*residual_tmp434) + residual_tmp245*residual_tmp433 + residual_tmp247*residual_tmp436);
      grad_coeff0_0_values[lane] = grad_coeff0_0;
      grad_coeff0_1_values[lane] = grad_coeff0_1;
      grad_coeff0_2_values[lane] = grad_coeff0_2;
      grad_coeff1_0_values[lane] = grad_coeff1_0;
      grad_coeff1_1_values[lane] = grad_coeff1_1;
      grad_coeff1_2_values[lane] = grad_coeff1_2;
      grad_coeff2_0_values[lane] = grad_coeff2_0;
      grad_coeff2_1_values[lane] = grad_coeff2_1;
      grad_coeff2_2_values[lane] = grad_coeff2_2;
    }
    for (int test = 0; test < NS; ++test) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
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
        output[test * NC][lane] += q_weight[q] * det * (grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
        output[test * NC + 1][lane] += q_weight[q] * det * (grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
        output[test * NC + 2][lane] += q_weight[q] * det * (grad_coeff2_0_values[lane] * test_grad0 + grad_coeff2_1_values[lane] * test_grad1 + grad_coeff2_2_values[lane] * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_residual_merit_d3_simplex_tet4_jacobian_action_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[3 * NS],
    const s_t *const RSTR previous[3 * NS],
    const s_t *const RSTR direction[3 * NS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t *const RSTR output[3 * NS]
) {
  for (int q = 0; q < NQ; ++q) {
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
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
      const s_t u0_grad_0_ref = -(current[0][lane]) + current[3][lane];
      const s_t u0_grad_1_ref = -(current[0][lane]) + current[6][lane];
      const s_t u0_grad_2_ref = -(current[0][lane]) + current[9][lane];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
      const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
      const s_t u0_old_grad_0_ref = -(previous[0][lane]) + previous[3][lane];
      const s_t u0_old_grad_1_ref = -(previous[0][lane]) + previous[6][lane];
      const s_t u0_old_grad_2_ref = -(previous[0][lane]) + previous[9][lane];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
      const s_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
      const s_t u0_direction_grad_0_ref = -(direction[0][lane]) + direction[3][lane];
      const s_t u0_direction_grad_1_ref = -(direction[0][lane]) + direction[6][lane];
      const s_t u0_direction_grad_2_ref = -(direction[0][lane]) + direction[9][lane];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
      const s_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
      const s_t u1_grad_0_ref = -(current[1][lane]) + current[4][lane];
      const s_t u1_grad_1_ref = -(current[1][lane]) + current[7][lane];
      const s_t u1_grad_2_ref = -(current[1][lane]) + current[10][lane];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
      const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
      const s_t u1_old_grad_0_ref = -(previous[1][lane]) + previous[4][lane];
      const s_t u1_old_grad_1_ref = -(previous[1][lane]) + previous[7][lane];
      const s_t u1_old_grad_2_ref = -(previous[1][lane]) + previous[10][lane];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
      const s_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
      const s_t u1_direction_grad_0_ref = -(direction[1][lane]) + direction[4][lane];
      const s_t u1_direction_grad_1_ref = -(direction[1][lane]) + direction[7][lane];
      const s_t u1_direction_grad_2_ref = -(direction[1][lane]) + direction[10][lane];
      const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
      const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
      const s_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
      const s_t u2_grad_0_ref = -(current[2][lane]) + current[5][lane];
      const s_t u2_grad_1_ref = -(current[2][lane]) + current[8][lane];
      const s_t u2_grad_2_ref = -(current[2][lane]) + current[11][lane];
      const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
      const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
      const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
      const s_t u2_old_grad_0_ref = -(previous[2][lane]) + previous[5][lane];
      const s_t u2_old_grad_1_ref = -(previous[2][lane]) + previous[8][lane];
      const s_t u2_old_grad_2_ref = -(previous[2][lane]) + previous[11][lane];
      const s_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
      const s_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
      const s_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
      const s_t u2_direction_grad_0_ref = -(direction[2][lane]) + direction[5][lane];
      const s_t u2_direction_grad_1_ref = -(direction[2][lane]) + direction[8][lane];
      const s_t u2_direction_grad_2_ref = -(direction[2][lane]) + direction[11][lane];
      const s_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
      const s_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
      const s_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = s_t(2)*u0_grad_2;
      const s_t residual_tmp1 = residual_tmp0*u1_grad_2;
      const s_t residual_tmp2 = u1_grad_1 + s_t(1);
      const s_t residual_tmp3 = s_t(2)*u0_grad_1;
      const s_t residual_tmp4 = residual_tmp2*residual_tmp3;
      const s_t residual_tmp5 = mu*(-residual_tmp1 - residual_tmp4);
      const s_t residual_tmp6 = u0_grad_2*u2_grad_1;
      const s_t residual_tmp7 = -residual_tmp6;
      const s_t residual_tmp8 = u2_grad_2 + s_t(1);
      const s_t residual_tmp9 = residual_tmp8*u0_grad_1;
      const s_t residual_tmp10 = -residual_tmp7 - residual_tmp9;
      const s_t residual_tmp11 = u1_grad_2*u2_grad_1;
      const s_t residual_tmp12 = s_t(2)*residual_tmp11;
      const s_t residual_tmp13 = -s_t(2)*residual_tmp2*residual_tmp8;
      const s_t residual_tmp14 = ((s_t(1) / s_t(2)))*lmbda;
      const s_t residual_tmp15 = residual_tmp14*(-residual_tmp12 - residual_tmp13);
      const s_t residual_tmp16 = u0_grad_0*u1_grad_1;
      const s_t residual_tmp17 = u0_grad_1*u1_grad_2;
      const s_t residual_tmp18 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp19 = u0_grad_2*u2_grad_0;
      const s_t residual_tmp20 = -residual_tmp11 + residual_tmp2 + u1_grad_1*u2_grad_2 + u2_grad_2;
      const s_t residual_tmp21 = -residual_tmp19 + u0_grad_0*u2_grad_2 + u0_grad_0;
      const s_t residual_tmp22 = residual_tmp16 - residual_tmp18;
      const s_t residual_tmp23 = -residual_tmp11*u0_grad_0 + residual_tmp16*u2_grad_2 + residual_tmp17*u2_grad_0 - residual_tmp18*u2_grad_2 - residual_tmp19*u1_grad_1 + residual_tmp20 + residual_tmp21 + residual_tmp22 + residual_tmp6*u1_grad_0;
      const s_t residual_tmp24 = pow_m1(residual_tmp23);
      const s_t residual_tmp25 = residual_tmp7 + u0_grad_1*u2_grad_2 + u0_grad_1;
      const s_t residual_tmp26 = residual_tmp20*u_dt_shift;
      const s_t residual_tmp27 = u1_grad_2*u_dt_shift + u1_old_grad_2;
      const s_t residual_tmp28 = residual_tmp27*u2_grad_1;
      const s_t residual_tmp29 = u1_grad_1*u_dt_shift + u1_old_grad_1;
      const s_t residual_tmp30 = residual_tmp29*residual_tmp8;
      const s_t residual_tmp31 = residual_tmp28 - residual_tmp30;
      const s_t residual_tmp32 = -residual_tmp26 - residual_tmp31;
      const s_t residual_tmp33 = -residual_tmp17 + u0_grad_2*u1_grad_1 + u0_grad_2;
      const s_t residual_tmp34 = u2_grad_1*u_dt_shift + u2_old_grad_1;
      const s_t residual_tmp35 = residual_tmp34*residual_tmp8;
      const s_t residual_tmp36 = u2_grad_2*u_dt_shift + u2_old_grad_2;
      const s_t residual_tmp37 = residual_tmp36*u2_grad_1;
      const s_t residual_tmp38 = u0_grad_2*u_dt_shift + u0_old_grad_2;
      const s_t residual_tmp39 = u0_grad_1*u_dt_shift + u0_old_grad_1;
      const s_t residual_tmp40 = residual_tmp38*u0_grad_1 - residual_tmp39*u0_grad_2;
      const s_t residual_tmp41 = residual_tmp35 - residual_tmp37 + residual_tmp40;
      const s_t residual_tmp42 = residual_tmp25*u_dt_shift;
      const s_t residual_tmp43 = residual_tmp36*u0_grad_1;
      const s_t residual_tmp44 = residual_tmp34*u0_grad_2;
      const s_t residual_tmp45 = residual_tmp42 + residual_tmp43 - residual_tmp44;
      const s_t residual_tmp46 = residual_tmp39*residual_tmp8;
      const s_t residual_tmp47 = residual_tmp38*u2_grad_1;
      const s_t residual_tmp48 = residual_tmp46 - residual_tmp47;
      const s_t residual_tmp49 = s_t(3)*eta_b;
      const s_t residual_tmp50 = residual_tmp49*(residual_tmp45 + residual_tmp48);
      const s_t residual_tmp51 = s_t(2)*eta_s;
      const s_t residual_tmp52 = residual_tmp50 + residual_tmp51*(s_t(2)*residual_tmp39*residual_tmp8 - residual_tmp45 - s_t(2)*residual_tmp47);
      const s_t residual_tmp53 = ((s_t(1) / s_t(3)))*residual_tmp20;
      const s_t residual_tmp54 = pow_m2(residual_tmp23);
      const s_t residual_tmp55 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp56 = u0_grad_2*u1_grad_0;
      const s_t residual_tmp57 = -residual_tmp56 + u0_grad_0*u1_grad_2 + u1_grad_2;
      const s_t residual_tmp58 = u1_grad_2*u2_grad_0;
      const s_t residual_tmp59 = -residual_tmp58;
      const s_t residual_tmp60 = residual_tmp59 + u1_grad_0*u2_grad_2 + u1_grad_0;
      const s_t residual_tmp61 = u1_grad_0*u2_grad_1;
      const s_t residual_tmp62 = -residual_tmp61 + u1_grad_1*u2_grad_0 + u2_grad_0;
      const s_t residual_tmp63 = residual_tmp2 + residual_tmp22 + u0_grad_0;
      const s_t residual_tmp64 = u2_grad_0*u_dt_shift + u2_old_grad_0;
      const s_t residual_tmp65 = -residual_tmp20*residual_tmp64 + residual_tmp33*residual_tmp55 + residual_tmp34*residual_tmp60 + residual_tmp36*residual_tmp62 - residual_tmp38*residual_tmp63 + residual_tmp39*residual_tmp57;
      const s_t residual_tmp66 = u0_grad_1*u2_grad_0;
      const s_t residual_tmp67 = -residual_tmp66 + u0_grad_0*u2_grad_1 + u2_grad_1;
      const s_t residual_tmp68 = residual_tmp21 + residual_tmp8;
      const s_t residual_tmp69 = u1_grad_0*u_dt_shift + u1_old_grad_0;
      const s_t residual_tmp70 = -residual_tmp20*residual_tmp69 + residual_tmp25*residual_tmp55 + residual_tmp27*residual_tmp62 + residual_tmp29*residual_tmp60 + residual_tmp38*residual_tmp67 - residual_tmp39*residual_tmp68;
      const s_t residual_tmp71 = residual_tmp25*residual_tmp69;
      const s_t residual_tmp72 = residual_tmp27*residual_tmp67;
      const s_t residual_tmp73 = residual_tmp33*residual_tmp64;
      const s_t residual_tmp74 = residual_tmp34*residual_tmp57;
      const s_t residual_tmp75 = residual_tmp29*residual_tmp68;
      const s_t residual_tmp76 = -residual_tmp75;
      const s_t residual_tmp77 = residual_tmp36*residual_tmp63;
      const s_t residual_tmp78 = -residual_tmp77;
      const s_t residual_tmp79 = residual_tmp71 + residual_tmp72 + residual_tmp73 + residual_tmp74 + residual_tmp76 + residual_tmp78;
      const s_t residual_tmp80 = residual_tmp20*residual_tmp55;
      const s_t residual_tmp81 = residual_tmp38*residual_tmp62 + residual_tmp39*residual_tmp60 - residual_tmp80;
      const s_t residual_tmp82 = residual_tmp49*(residual_tmp79 + residual_tmp81);
      const s_t residual_tmp83 = residual_tmp51*(s_t(2)*residual_tmp38*residual_tmp62 + s_t(2)*residual_tmp39*residual_tmp60 - residual_tmp79 - s_t(2)*residual_tmp80) + residual_tmp82;
      const s_t residual_tmp84 = -residual_tmp83;
      const s_t residual_tmp85 = residual_tmp54*(eta_s*(residual_tmp25*residual_tmp70 + residual_tmp33*residual_tmp65) + residual_tmp53*residual_tmp84);
      const s_t residual_tmp86 = residual_tmp3*u2_grad_1;
      const s_t residual_tmp87 = residual_tmp0*residual_tmp8;
      const s_t residual_tmp88 = mu*(-residual_tmp86 - residual_tmp87);
      const s_t residual_tmp89 = residual_tmp2*u0_grad_2;
      const s_t residual_tmp90 = residual_tmp17 - residual_tmp89;
      const s_t residual_tmp91 = residual_tmp34*u1_grad_2;
      const s_t residual_tmp92 = residual_tmp2*residual_tmp36;
      const s_t residual_tmp93 = residual_tmp91 - residual_tmp92;
      const s_t residual_tmp94 = -residual_tmp26 - residual_tmp93;
      const s_t residual_tmp95 = -residual_tmp2*residual_tmp27 + residual_tmp29*u1_grad_2;
      const s_t residual_tmp96 = -residual_tmp40 - residual_tmp95;
      const s_t residual_tmp97 = residual_tmp33*u_dt_shift;
      const s_t residual_tmp98 = residual_tmp29*u0_grad_2;
      const s_t residual_tmp99 = residual_tmp27*u0_grad_1;
      const s_t residual_tmp100 = residual_tmp97 + residual_tmp98 - residual_tmp99;
      const s_t residual_tmp101 = residual_tmp2*residual_tmp38;
      const s_t residual_tmp102 = residual_tmp39*u1_grad_2;
      const s_t residual_tmp103 = residual_tmp101 - residual_tmp102;
      const s_t residual_tmp104 = residual_tmp49*(residual_tmp100 + residual_tmp103);
      const s_t residual_tmp105 = residual_tmp104 + residual_tmp51*(-residual_tmp100 - s_t(2)*residual_tmp102 + s_t(2)*residual_tmp2*residual_tmp38);
      const s_t residual_tmp106 = s_t(2)*pow_2(u1_grad_2);
      const s_t residual_tmp107 = s_t(2)*pow_2(residual_tmp8) + s_t(2);
      const s_t residual_tmp108 = residual_tmp106 + residual_tmp107;
      const s_t residual_tmp109 = s_t(2)*pow_2(u2_grad_1);
      const s_t residual_tmp110 = s_t(2)*pow_2(residual_tmp2);
      const s_t residual_tmp111 = residual_tmp109 + residual_tmp110;
      const s_t residual_tmp112 = -residual_tmp11 + residual_tmp2*residual_tmp8;
      const s_t residual_tmp113 = -residual_tmp101 + residual_tmp102;
      const s_t residual_tmp114 = residual_tmp113 + residual_tmp97;
      const s_t residual_tmp115 = -residual_tmp46 + residual_tmp47;
      const s_t residual_tmp116 = residual_tmp115 + residual_tmp42;
      const s_t residual_tmp117 = residual_tmp26 - residual_tmp91 + residual_tmp92;
      const s_t residual_tmp118 = -residual_tmp28 + residual_tmp30;
      const s_t residual_tmp119 = residual_tmp49*(-residual_tmp117 - residual_tmp118);
      const s_t residual_tmp120 = residual_tmp119 + residual_tmp51*(-s_t(2)*residual_tmp26 - residual_tmp31 - residual_tmp93);
      const s_t residual_tmp121 = -residual_tmp20;
      const s_t residual_tmp122 = s_t(2)*u2_grad_0;
      const s_t residual_tmp123 = residual_tmp122*u2_grad_1;
      const s_t residual_tmp124 = s_t(2)*u1_grad_0;
      const s_t residual_tmp125 = residual_tmp124*residual_tmp2;
      const s_t residual_tmp126 = residual_tmp123 + residual_tmp125;
      const s_t residual_tmp127 = residual_tmp8*u1_grad_0;
      const s_t residual_tmp128 = -residual_tmp127 - residual_tmp59;
      const s_t residual_tmp129 = residual_tmp60*u_dt_shift;
      const s_t residual_tmp130 = residual_tmp36*u1_grad_0;
      const s_t residual_tmp131 = residual_tmp64*u1_grad_2;
      const s_t residual_tmp132 = residual_tmp129 + residual_tmp130 - residual_tmp131;
      const s_t residual_tmp133 = residual_tmp69*residual_tmp8;
      const s_t residual_tmp134 = residual_tmp27*u2_grad_0;
      const s_t residual_tmp135 = residual_tmp133 - residual_tmp134;
      const s_t residual_tmp136 = residual_tmp49*(residual_tmp132 + residual_tmp135);
      const s_t residual_tmp137 = -residual_tmp133 + residual_tmp134;
      const s_t residual_tmp138 = -residual_tmp130 + residual_tmp131;
      const s_t residual_tmp139 = residual_tmp136 + residual_tmp51*(s_t(2)*residual_tmp129 + residual_tmp137 + residual_tmp138);
      const s_t residual_tmp140 = residual_tmp57*u_dt_shift;
      const s_t residual_tmp141 = residual_tmp38*u1_grad_0;
      const s_t residual_tmp142 = residual_tmp55*u1_grad_2;
      const s_t residual_tmp143 = residual_tmp141 - residual_tmp142;
      const s_t residual_tmp144 = residual_tmp140 + residual_tmp143;
      const s_t residual_tmp145 = residual_tmp68*u_dt_shift;
      const s_t residual_tmp146 = residual_tmp38*u2_grad_0;
      const s_t residual_tmp147 = residual_tmp55*residual_tmp8;
      const s_t residual_tmp148 = residual_tmp146 - residual_tmp147;
      const s_t residual_tmp149 = -residual_tmp145 - residual_tmp148;
      const s_t residual_tmp150 = residual_tmp65*u1_grad_2;
      const s_t residual_tmp151 = -residual_tmp150;
      const s_t residual_tmp152 = residual_tmp70*residual_tmp8;
      const s_t residual_tmp153 = residual_tmp124*u1_grad_2;
      const s_t residual_tmp154 = residual_tmp122*residual_tmp8;
      const s_t residual_tmp155 = residual_tmp153 + residual_tmp154;
      const s_t residual_tmp156 = residual_tmp2*u2_grad_0;
      const s_t residual_tmp157 = -residual_tmp156 + residual_tmp61;
      const s_t residual_tmp158 = residual_tmp62*u_dt_shift;
      const s_t residual_tmp159 = residual_tmp2*residual_tmp64;
      const s_t residual_tmp160 = residual_tmp34*u1_grad_0;
      const s_t residual_tmp161 = residual_tmp158 + residual_tmp159 - residual_tmp160;
      const s_t residual_tmp162 = residual_tmp29*u2_grad_0;
      const s_t residual_tmp163 = residual_tmp69*u2_grad_1;
      const s_t residual_tmp164 = residual_tmp162 - residual_tmp163;
      const s_t residual_tmp165 = residual_tmp49*(residual_tmp161 + residual_tmp164);
      const s_t residual_tmp166 = -residual_tmp162 + residual_tmp163;
      const s_t residual_tmp167 = -residual_tmp159 + residual_tmp160;
      const s_t residual_tmp168 = residual_tmp165 + residual_tmp51*(s_t(2)*residual_tmp158 + residual_tmp166 + residual_tmp167);
      const s_t residual_tmp169 = residual_tmp67*u_dt_shift;
      const s_t residual_tmp170 = residual_tmp39*u2_grad_0;
      const s_t residual_tmp171 = residual_tmp55*u2_grad_1;
      const s_t residual_tmp172 = residual_tmp170 - residual_tmp171;
      const s_t residual_tmp173 = residual_tmp169 + residual_tmp172;
      const s_t residual_tmp174 = residual_tmp63*u_dt_shift;
      const s_t residual_tmp175 = residual_tmp39*u1_grad_0;
      const s_t residual_tmp176 = residual_tmp2*residual_tmp55;
      const s_t residual_tmp177 = residual_tmp175 - residual_tmp176;
      const s_t residual_tmp178 = -residual_tmp174 - residual_tmp177;
      const s_t residual_tmp179 = residual_tmp70*u2_grad_1;
      const s_t residual_tmp180 = -residual_tmp179;
      const s_t residual_tmp181 = residual_tmp2*residual_tmp65;
      const s_t residual_tmp182 = u0_grad_0 + s_t(1);
      const s_t residual_tmp183 = residual_tmp182*u1_grad_2;
      const s_t residual_tmp184 = s_t(6)*u2_grad_1;
      const s_t residual_tmp185 = s_t(2)*residual_tmp56;
      const s_t residual_tmp186 = residual_tmp184 - residual_tmp185;
      const s_t residual_tmp187 = residual_tmp182*u2_grad_1;
      const s_t residual_tmp188 = -residual_tmp187 + residual_tmp66;
      const s_t residual_tmp189 = lmbda*(-residual_tmp11*residual_tmp182 - residual_tmp18*residual_tmp8 + residual_tmp182*residual_tmp2*residual_tmp8 - residual_tmp19*residual_tmp2 + u0_grad_1*u1_grad_2*u2_grad_0 + u0_grad_2*u1_grad_0*u2_grad_1 + s_t(-1));
      const s_t residual_tmp190 = residual_tmp189*u2_grad_1;
      const s_t residual_tmp191 = -residual_tmp190;
      const s_t residual_tmp192 = residual_tmp182*residual_tmp34;
      const s_t residual_tmp193 = residual_tmp64*u0_grad_1;
      const s_t residual_tmp194 = residual_tmp169 + residual_tmp192 - residual_tmp193;
      const s_t residual_tmp195 = -residual_tmp170 + residual_tmp171;
      const s_t residual_tmp196 = residual_tmp49*(residual_tmp194 + residual_tmp195);
      const s_t residual_tmp197 = residual_tmp196 + residual_tmp51*(-s_t(2)*residual_tmp170 - residual_tmp194 + s_t(2)*residual_tmp55*u2_grad_1);
      const s_t residual_tmp198 = residual_tmp158 + residual_tmp166;
      const s_t residual_tmp199 = residual_tmp34*u2_grad_0;
      const s_t residual_tmp200 = residual_tmp64*u2_grad_1;
      const s_t residual_tmp201 = -residual_tmp182*residual_tmp39 + residual_tmp55*u0_grad_1;
      const s_t residual_tmp202 = -residual_tmp199 + residual_tmp200 - residual_tmp201;
      const s_t residual_tmp203 = residual_tmp65*u0_grad_1;
      const s_t residual_tmp204 = ((s_t(1) / s_t(3)))*residual_tmp84;
      const s_t residual_tmp205 = s_t(6)*u1_grad_2;
      const s_t residual_tmp206 = s_t(2)*residual_tmp66;
      const s_t residual_tmp207 = residual_tmp205 - residual_tmp206;
      const s_t residual_tmp208 = -residual_tmp183 + residual_tmp56;
      const s_t residual_tmp209 = residual_tmp189*u1_grad_2;
      const s_t residual_tmp210 = -residual_tmp209;
      const s_t residual_tmp211 = residual_tmp182*residual_tmp27;
      const s_t residual_tmp212 = residual_tmp69*u0_grad_2;
      const s_t residual_tmp213 = residual_tmp140 + residual_tmp211 - residual_tmp212;
      const s_t residual_tmp214 = -residual_tmp141 + residual_tmp142;
      const s_t residual_tmp215 = residual_tmp49*(residual_tmp213 + residual_tmp214);
      const s_t residual_tmp216 = residual_tmp215 + residual_tmp51*(-s_t(2)*residual_tmp141 - residual_tmp213 + s_t(2)*residual_tmp55*u1_grad_2);
      const s_t residual_tmp217 = residual_tmp129 + residual_tmp138;
      const s_t residual_tmp218 = -residual_tmp182*residual_tmp38 + residual_tmp55*u0_grad_2;
      const s_t residual_tmp219 = residual_tmp27*u1_grad_0 - residual_tmp69*u1_grad_2;
      const s_t residual_tmp220 = -residual_tmp218 - residual_tmp219;
      const s_t residual_tmp221 = residual_tmp70*u0_grad_2;
      const s_t residual_tmp222 = s_t(2)*u1_grad_1 + s_t(2);
      const s_t residual_tmp223 = s_t(2)*residual_tmp18;
      const s_t residual_tmp224 = s_t(6)*u2_grad_2 + s_t(6);
      const s_t residual_tmp225 = residual_tmp223 + residual_tmp224;
      const s_t residual_tmp226 = residual_tmp182*residual_tmp8 - residual_tmp19;
      const s_t residual_tmp227 = s_t(2)*u2_grad_2 + s_t(2);
      const s_t residual_tmp228 = ((s_t(1) / s_t(2)))*residual_tmp189;
      const s_t residual_tmp229 = residual_tmp227*residual_tmp228;
      const s_t residual_tmp230 = -residual_tmp68;
      const s_t residual_tmp231 = residual_tmp182*residual_tmp36;
      const s_t residual_tmp232 = residual_tmp64*u0_grad_2;
      const s_t residual_tmp233 = residual_tmp145 + residual_tmp231 - residual_tmp232;
      const s_t residual_tmp234 = -residual_tmp146 + residual_tmp147;
      const s_t residual_tmp235 = residual_tmp49*(-residual_tmp233 - residual_tmp234);
      const s_t residual_tmp236 = residual_tmp235 + residual_tmp51*(s_t(2)*residual_tmp146 - s_t(2)*residual_tmp147 + residual_tmp233);
      const s_t residual_tmp237 = residual_tmp129 + residual_tmp137;
      const s_t residual_tmp238 = residual_tmp36*u2_grad_0;
      const s_t residual_tmp239 = residual_tmp64*residual_tmp8;
      const s_t residual_tmp240 = residual_tmp218 + residual_tmp238 - residual_tmp239;
      const s_t residual_tmp241 = residual_tmp65*u0_grad_2;
      const s_t residual_tmp242 = s_t(2)*residual_tmp19;
      const s_t residual_tmp243 = s_t(6)*u1_grad_1 + s_t(6);
      const s_t residual_tmp244 = residual_tmp242 + residual_tmp243;
      const s_t residual_tmp245 = -residual_tmp18 + residual_tmp182*residual_tmp2;
      const s_t residual_tmp246 = residual_tmp222*residual_tmp228;
      const s_t residual_tmp247 = -residual_tmp63;
      const s_t residual_tmp248 = residual_tmp182*residual_tmp29;
      const s_t residual_tmp249 = residual_tmp69*u0_grad_1;
      const s_t residual_tmp250 = residual_tmp174 + residual_tmp248 - residual_tmp249;
      const s_t residual_tmp251 = -residual_tmp175 + residual_tmp176;
      const s_t residual_tmp252 = residual_tmp49*(-residual_tmp250 - residual_tmp251);
      const s_t residual_tmp253 = residual_tmp252 + residual_tmp51*(s_t(2)*residual_tmp175 - s_t(2)*residual_tmp176 + residual_tmp250);
      const s_t residual_tmp254 = residual_tmp158 + residual_tmp167;
      const s_t residual_tmp255 = -residual_tmp2*residual_tmp69 + residual_tmp29*u1_grad_0;
      const s_t residual_tmp256 = residual_tmp201 + residual_tmp255;
      const s_t residual_tmp257 = residual_tmp70*u0_grad_1;
      const s_t residual_tmp258 = residual_tmp122*residual_tmp182;
      const s_t residual_tmp259 = mu*(-residual_tmp258 - residual_tmp87);
      const s_t residual_tmp260 = -s_t(2)*residual_tmp58;
      const s_t residual_tmp261 = s_t(2)*residual_tmp127;
      const s_t residual_tmp262 = residual_tmp14*(-residual_tmp260 - residual_tmp261);
      const s_t residual_tmp263 = residual_tmp54*(-eta_s*(-residual_tmp57*residual_tmp65 + residual_tmp68*residual_tmp70) + ((s_t(1) / s_t(3)))*residual_tmp60*residual_tmp83);
      const s_t residual_tmp264 = s_t(2)*pow_2(u1_grad_0);
      const s_t residual_tmp265 = s_t(2)*pow_2(u2_grad_0);
      const s_t residual_tmp266 = residual_tmp264 + residual_tmp265;
      const s_t residual_tmp267 = residual_tmp124*residual_tmp182;
      const s_t residual_tmp268 = mu*(-residual_tmp1 - residual_tmp267);
      const s_t residual_tmp269 = s_t(2)*u1_grad_2;
      const s_t residual_tmp270 = residual_tmp2*residual_tmp269;
      const s_t residual_tmp271 = s_t(2)*u2_grad_1;
      const s_t residual_tmp272 = residual_tmp271*residual_tmp8;
      const s_t residual_tmp273 = mu*(-residual_tmp270 - residual_tmp272);
      const s_t residual_tmp274 = residual_tmp65*u1_grad_0;
      const s_t residual_tmp275 = residual_tmp70*u2_grad_0;
      const s_t residual_tmp276 = -residual_tmp275;
      const s_t residual_tmp277 = residual_tmp274 + residual_tmp276;
      const s_t residual_tmp278 = s_t(2)*u0_grad_0 + s_t(2);
      const s_t residual_tmp279 = s_t(4)*residual_tmp182;
      const s_t residual_tmp280 = -residual_tmp70*residual_tmp8;
      const s_t residual_tmp281 = residual_tmp182*residual_tmp65;
      const s_t residual_tmp282 = ((s_t(1) / s_t(3)))*residual_tmp83;
      const s_t residual_tmp283 = residual_tmp282*u2_grad_0;
      const s_t residual_tmp284 = s_t(6)*u2_grad_0;
      const s_t residual_tmp285 = s_t(2)*residual_tmp89;
      const s_t residual_tmp286 = residual_tmp189*u2_grad_0;
      const s_t residual_tmp287 = mu*(-residual_tmp284 - residual_tmp285 + s_t(4)*u0_grad_1*u1_grad_2) + residual_tmp286;
      const s_t residual_tmp288 = s_t(2)*residual_tmp187;
      const s_t residual_tmp289 = mu*(-residual_tmp205 - residual_tmp288 + s_t(4)*u0_grad_1*u2_grad_0) + residual_tmp209;
      const s_t residual_tmp290 = ((s_t(1) / s_t(3)))*residual_tmp60;
      const s_t residual_tmp291 = -s_t(2)*residual_tmp182*residual_tmp2;
      const s_t residual_tmp292 = mu*(s_t(4)*residual_tmp18 + residual_tmp224 + residual_tmp291) - residual_tmp227*residual_tmp228;
      const s_t residual_tmp293 = -s_t(2)*residual_tmp6;
      const s_t residual_tmp294 = s_t(6)*u1_grad_0;
      const s_t residual_tmp295 = residual_tmp293 + residual_tmp294;
      const s_t residual_tmp296 = residual_tmp189*u1_grad_0;
      const s_t residual_tmp297 = -residual_tmp296;
      const s_t residual_tmp298 = residual_tmp182*residual_tmp70;
      const s_t residual_tmp299 = residual_tmp282*u1_grad_0;
      const s_t residual_tmp300 = mu*(-residual_tmp267 - residual_tmp4);
      const s_t residual_tmp301 = s_t(2)*residual_tmp61;
      const s_t residual_tmp302 = s_t(2)*residual_tmp156;
      const s_t residual_tmp303 = residual_tmp14*(residual_tmp301 - residual_tmp302);
      const s_t residual_tmp304 = residual_tmp54*(-eta_s*(residual_tmp63*residual_tmp65 - residual_tmp67*residual_tmp70) + ((s_t(1) / s_t(3)))*residual_tmp62*residual_tmp83);
      const s_t residual_tmp305 = mu*(-residual_tmp258 - residual_tmp86);
      const s_t residual_tmp306 = -residual_tmp2*residual_tmp65;
      const s_t residual_tmp307 = s_t(2)*residual_tmp9;
      const s_t residual_tmp308 = mu*(-residual_tmp294 - residual_tmp307 + s_t(4)*u0_grad_2*u2_grad_1) + residual_tmp296;
      const s_t residual_tmp309 = s_t(2)*residual_tmp183;
      const s_t residual_tmp310 = mu*(-residual_tmp184 - residual_tmp309 + s_t(4)*u0_grad_2*u1_grad_0) + residual_tmp190;
      const s_t residual_tmp311 = ((s_t(1) / s_t(3)))*residual_tmp62;
      const s_t residual_tmp312 = -s_t(2)*residual_tmp182*residual_tmp8;
      const s_t residual_tmp313 = mu*(s_t(4)*residual_tmp19 + residual_tmp243 + residual_tmp312) - residual_tmp222*residual_tmp228;
      const s_t residual_tmp314 = s_t(2)*residual_tmp17;
      const s_t residual_tmp315 = residual_tmp284 - residual_tmp314;
      const s_t residual_tmp316 = -residual_tmp286;
      const s_t residual_tmp317 = residual_tmp269*residual_tmp8;
      const s_t residual_tmp318 = residual_tmp2*residual_tmp271;
      const s_t residual_tmp319 = mu*(-residual_tmp317 - residual_tmp318);
      const s_t residual_tmp320 = residual_tmp14*(-residual_tmp293 - residual_tmp307);
      const s_t residual_tmp321 = -residual_tmp43 + residual_tmp44;
      const s_t residual_tmp322 = residual_tmp321 + residual_tmp42;
      const s_t residual_tmp323 = residual_tmp104 + residual_tmp51*(-residual_tmp103 + s_t(2)*residual_tmp29*u0_grad_2 - residual_tmp97 - s_t(2)*residual_tmp99);
      const s_t residual_tmp324 = residual_tmp25*residual_tmp64 - residual_tmp27*residual_tmp63 + residual_tmp29*residual_tmp57 + residual_tmp33*residual_tmp69 - residual_tmp34*residual_tmp68 + residual_tmp36*residual_tmp67;
      const s_t residual_tmp325 = residual_tmp51*(s_t(2)*residual_tmp25*residual_tmp69 + s_t(2)*residual_tmp27*residual_tmp67 - residual_tmp73 - residual_tmp74 - s_t(2)*residual_tmp75 - residual_tmp78 - residual_tmp81) + residual_tmp82;
      const s_t residual_tmp326 = residual_tmp54*(-eta_s*(residual_tmp20*residual_tmp70 - residual_tmp324*residual_tmp33) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp325);
      const s_t residual_tmp327 = s_t(2)*pow_2(u0_grad_2);
      const s_t residual_tmp328 = residual_tmp107 + residual_tmp327;
      const s_t residual_tmp329 = s_t(2)*pow_2(u0_grad_1);
      const s_t residual_tmp330 = residual_tmp109 + residual_tmp329;
      const s_t residual_tmp331 = -residual_tmp98 + residual_tmp99;
      const s_t residual_tmp332 = residual_tmp331 + residual_tmp97;
      const s_t residual_tmp333 = residual_tmp50 + residual_tmp51*(residual_tmp115 + residual_tmp321 + s_t(2)*residual_tmp42);
      const s_t residual_tmp334 = -residual_tmp35 + residual_tmp37 + residual_tmp95;
      const s_t residual_tmp335 = residual_tmp119 + residual_tmp51*(residual_tmp117 + s_t(2)*residual_tmp28 - s_t(2)*residual_tmp30);
      const s_t residual_tmp336 = residual_tmp0*residual_tmp182;
      const s_t residual_tmp337 = mu*(-residual_tmp154 - residual_tmp336);
      const s_t residual_tmp338 = -residual_tmp192 + residual_tmp193;
      const s_t residual_tmp339 = residual_tmp196 + residual_tmp51*(s_t(2)*residual_tmp169 + residual_tmp172 + residual_tmp338);
      const s_t residual_tmp340 = -residual_tmp248 + residual_tmp249;
      const s_t residual_tmp341 = -residual_tmp174 - residual_tmp340;
      const s_t residual_tmp342 = residual_tmp324*u0_grad_1;
      const s_t residual_tmp343 = residual_tmp180 + residual_tmp342;
      const s_t residual_tmp344 = s_t(4)*residual_tmp2;
      const s_t residual_tmp345 = residual_tmp182*residual_tmp3;
      const s_t residual_tmp346 = residual_tmp123 + residual_tmp345;
      const s_t residual_tmp347 = -residual_tmp231 + residual_tmp232;
      const s_t residual_tmp348 = residual_tmp235 + residual_tmp51*(-s_t(2)*residual_tmp145 - residual_tmp148 - residual_tmp347);
      const s_t residual_tmp349 = -residual_tmp211 + residual_tmp212;
      const s_t residual_tmp350 = residual_tmp140 + residual_tmp349;
      const s_t residual_tmp351 = residual_tmp324*u0_grad_2;
      const s_t residual_tmp352 = residual_tmp165 + residual_tmp51*(-residual_tmp161 - s_t(2)*residual_tmp163 + s_t(2)*residual_tmp29*u2_grad_0);
      const s_t residual_tmp353 = residual_tmp199 - residual_tmp200 - residual_tmp255;
      const s_t residual_tmp354 = residual_tmp2*residual_tmp324;
      const s_t residual_tmp355 = ((s_t(1) / s_t(3)))*residual_tmp325;
      const s_t residual_tmp356 = residual_tmp355*u2_grad_1;
      const s_t residual_tmp357 = residual_tmp215 + residual_tmp51*(-residual_tmp140 + s_t(2)*residual_tmp182*residual_tmp27 - s_t(2)*residual_tmp212 - residual_tmp214);
      const s_t residual_tmp358 = -residual_tmp145 - residual_tmp347;
      const s_t residual_tmp359 = residual_tmp70*u1_grad_2;
      const s_t residual_tmp360 = s_t(6)*u0_grad_2;
      const s_t residual_tmp361 = residual_tmp189*u0_grad_2;
      const s_t residual_tmp362 = mu*(-residual_tmp302 - residual_tmp360 + s_t(4)*u1_grad_0*u2_grad_1) + residual_tmp361;
      const s_t residual_tmp363 = residual_tmp136 + residual_tmp51*(-residual_tmp132 - s_t(2)*residual_tmp134 + s_t(2)*residual_tmp69*residual_tmp8);
      const s_t residual_tmp364 = ((s_t(1) / s_t(3)))*residual_tmp25;
      const s_t residual_tmp365 = residual_tmp219 - residual_tmp238 + residual_tmp239;
      const s_t residual_tmp366 = residual_tmp324*u1_grad_2;
      const s_t residual_tmp367 = s_t(6)*u0_grad_1;
      const s_t residual_tmp368 = residual_tmp260 + residual_tmp367;
      const s_t residual_tmp369 = residual_tmp189*u0_grad_1;
      const s_t residual_tmp370 = -residual_tmp369;
      const s_t residual_tmp371 = residual_tmp252 + residual_tmp51*(residual_tmp174 - s_t(2)*residual_tmp248 + s_t(2)*residual_tmp249 + residual_tmp251);
      const s_t residual_tmp372 = residual_tmp169 + residual_tmp338;
      const s_t residual_tmp373 = residual_tmp2*residual_tmp70;
      const s_t residual_tmp374 = residual_tmp355*u0_grad_1;
      const s_t residual_tmp375 = residual_tmp14*(-residual_tmp242 - residual_tmp312);
      const s_t residual_tmp376 = ((s_t(1) / s_t(3)))*residual_tmp68;
      const s_t residual_tmp377 = -(s_t(1) / s_t(3))*residual_tmp325;
      const s_t residual_tmp378 = residual_tmp54*(eta_s*(residual_tmp324*residual_tmp57 + residual_tmp60*residual_tmp70) + residual_tmp377*residual_tmp68);
      const s_t residual_tmp379 = residual_tmp124*u2_grad_0;
      const s_t residual_tmp380 = mu*(-residual_tmp317 - residual_tmp379);
      const s_t residual_tmp381 = s_t(2)*pow_2(residual_tmp182);
      const s_t residual_tmp382 = residual_tmp265 + residual_tmp381;
      const s_t residual_tmp383 = residual_tmp3*u0_grad_2;
      const s_t residual_tmp384 = residual_tmp272 + residual_tmp383;
      const s_t residual_tmp385 = residual_tmp182*residual_tmp324;
      const s_t residual_tmp386 = residual_tmp324*u1_grad_0;
      const s_t residual_tmp387 = -residual_tmp301 + residual_tmp360;
      const s_t residual_tmp388 = -residual_tmp361;
      const s_t residual_tmp389 = s_t(6)*u0_grad_0 + s_t(6);
      const s_t residual_tmp390 = residual_tmp12 + residual_tmp389;
      const s_t residual_tmp391 = residual_tmp228*residual_tmp278;
      const s_t residual_tmp392 = residual_tmp70*u1_grad_0;
      const s_t residual_tmp393 = residual_tmp14*(residual_tmp206 - residual_tmp288);
      const s_t residual_tmp394 = residual_tmp54*(-eta_s*(residual_tmp324*residual_tmp63 - residual_tmp62*residual_tmp70) + ((s_t(1) / s_t(3)))*residual_tmp325*residual_tmp67);
      const s_t residual_tmp395 = mu*(-residual_tmp318 - residual_tmp379);
      const s_t residual_tmp396 = -residual_tmp182*residual_tmp324;
      const s_t residual_tmp397 = mu*(-residual_tmp261 - residual_tmp367 + s_t(4)*u1_grad_2*u2_grad_0) + residual_tmp369;
      const s_t residual_tmp398 = ((s_t(1) / s_t(3)))*residual_tmp67;
      const s_t residual_tmp399 = mu*(s_t(4)*residual_tmp11 + residual_tmp13 + residual_tmp389) - residual_tmp228*residual_tmp278;
      const s_t residual_tmp400 = residual_tmp14*(-residual_tmp285 + residual_tmp314);
      const s_t residual_tmp401 = residual_tmp50 + residual_tmp51*(s_t(2)*residual_tmp36*u0_grad_1 - residual_tmp42 - s_t(2)*residual_tmp44 - residual_tmp48);
      const s_t residual_tmp402 = residual_tmp51*(s_t(2)*residual_tmp33*residual_tmp64 + s_t(2)*residual_tmp34*residual_tmp57 - residual_tmp71 - residual_tmp72 - residual_tmp76 - s_t(2)*residual_tmp77 - residual_tmp81) + residual_tmp82;
      const s_t residual_tmp403 = residual_tmp54*(-eta_s*(residual_tmp20*residual_tmp65 - residual_tmp25*residual_tmp324) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp402);
      const s_t residual_tmp404 = residual_tmp106 + residual_tmp327 + s_t(2);
      const s_t residual_tmp405 = residual_tmp110 + residual_tmp329;
      const s_t residual_tmp406 = residual_tmp104 + residual_tmp51*(residual_tmp113 + residual_tmp331 + s_t(2)*residual_tmp97);
      const s_t residual_tmp407 = residual_tmp119 + residual_tmp51*(residual_tmp118 + residual_tmp26 + s_t(2)*residual_tmp91 - s_t(2)*residual_tmp92);
      const s_t residual_tmp408 = mu*(-residual_tmp125 - residual_tmp345);
      const s_t residual_tmp409 = residual_tmp215 + residual_tmp51*(s_t(2)*residual_tmp140 + residual_tmp143 + residual_tmp349);
      const s_t residual_tmp410 = residual_tmp151 + residual_tmp351;
      const s_t residual_tmp411 = s_t(4)*residual_tmp8;
      const s_t residual_tmp412 = residual_tmp153 + residual_tmp336;
      const s_t residual_tmp413 = residual_tmp252 + residual_tmp51*(-s_t(2)*residual_tmp174 - residual_tmp177 - residual_tmp340);
      const s_t residual_tmp414 = residual_tmp136 + residual_tmp51*(-residual_tmp129 - s_t(2)*residual_tmp131 - residual_tmp135 + s_t(2)*residual_tmp36*u1_grad_0);
      const s_t residual_tmp415 = residual_tmp324*residual_tmp8;
      const s_t residual_tmp416 = ((s_t(1) / s_t(3)))*residual_tmp402;
      const s_t residual_tmp417 = residual_tmp416*u1_grad_2;
      const s_t residual_tmp418 = residual_tmp196 + residual_tmp51*(-residual_tmp169 + s_t(2)*residual_tmp182*residual_tmp34 - s_t(2)*residual_tmp193 - residual_tmp195);
      const s_t residual_tmp419 = residual_tmp65*u2_grad_1;
      const s_t residual_tmp420 = residual_tmp165 + residual_tmp51*(-residual_tmp158 - s_t(2)*residual_tmp160 - residual_tmp164 + s_t(2)*residual_tmp2*residual_tmp64);
      const s_t residual_tmp421 = ((s_t(1) / s_t(3)))*residual_tmp33;
      const s_t residual_tmp422 = residual_tmp324*u2_grad_1;
      const s_t residual_tmp423 = residual_tmp235 + residual_tmp51*(residual_tmp145 - s_t(2)*residual_tmp231 + s_t(2)*residual_tmp232 + residual_tmp234);
      const s_t residual_tmp424 = residual_tmp65*residual_tmp8;
      const s_t residual_tmp425 = residual_tmp416*u0_grad_2;
      const s_t residual_tmp426 = residual_tmp14*(residual_tmp185 - residual_tmp309);
      const s_t residual_tmp427 = residual_tmp54*(-eta_s*(residual_tmp324*residual_tmp68 - residual_tmp60*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp402*residual_tmp57);
      const s_t residual_tmp428 = residual_tmp264 + residual_tmp381;
      const s_t residual_tmp429 = residual_tmp270 + residual_tmp383;
      const s_t residual_tmp430 = residual_tmp324*u2_grad_0;
      const s_t residual_tmp431 = ((s_t(1) / s_t(3)))*residual_tmp57;
      const s_t residual_tmp432 = residual_tmp65*u2_grad_0;
      const s_t residual_tmp433 = residual_tmp14*(-residual_tmp223 - residual_tmp291);
      const s_t residual_tmp434 = ((s_t(1) / s_t(3)))*residual_tmp63;
      const s_t residual_tmp435 = -(s_t(1) / s_t(3))*residual_tmp402;
      const s_t residual_tmp436 = residual_tmp54*(eta_s*(residual_tmp324*residual_tmp67 + residual_tmp62*residual_tmp65) + residual_tmp435*residual_tmp63);
      const s_t grad_coeff0_0 = u0_direction_grad_0*(mu*(residual_tmp108 + residual_tmp111) + residual_tmp112*residual_tmp15 + residual_tmp121*residual_tmp85 + residual_tmp24*(eta_s*(residual_tmp114*residual_tmp33 + residual_tmp116*residual_tmp25) - residual_tmp120*residual_tmp53)) + u0_direction_grad_1*(-mu*residual_tmp126 + residual_tmp128*residual_tmp15 + residual_tmp24*(eta_s*(residual_tmp144*residual_tmp33 + residual_tmp149*residual_tmp25 + residual_tmp151 + residual_tmp152) - residual_tmp139*residual_tmp53) + residual_tmp60*residual_tmp85) + u0_direction_grad_2*(-mu*residual_tmp155 + residual_tmp15*residual_tmp157 + residual_tmp24*(eta_s*(residual_tmp173*residual_tmp25 + residual_tmp178*residual_tmp33 + residual_tmp180 + residual_tmp181) - residual_tmp168*residual_tmp53) + residual_tmp62*residual_tmp85) + u1_direction_grad_0*(residual_tmp10*residual_tmp15 + residual_tmp24*(eta_s*(residual_tmp25*residual_tmp32 + residual_tmp33*residual_tmp41) - residual_tmp52*residual_tmp53) + residual_tmp25*residual_tmp85 + residual_tmp5) + u1_direction_grad_1*(mu*(s_t(2)*residual_tmp182*residual_tmp222 - residual_tmp225) + residual_tmp15*residual_tmp226 + residual_tmp229 + residual_tmp230*residual_tmp85 + residual_tmp24*(eta_s*(residual_tmp237*residual_tmp25 + residual_tmp240*residual_tmp33 + residual_tmp241) + residual_tmp204*residual_tmp8 - residual_tmp236*residual_tmp53)) + u1_direction_grad_2*(mu*(s_t(4)*residual_tmp183 + residual_tmp186) + residual_tmp15*residual_tmp188 + residual_tmp191 + residual_tmp24*(eta_s*(residual_tmp198*residual_tmp25 + residual_tmp202*residual_tmp33 - residual_tmp203) - residual_tmp197*residual_tmp53 - residual_tmp204*u2_grad_1) + residual_tmp67*residual_tmp85) + u2_direction_grad_0*(residual_tmp15*residual_tmp90 + residual_tmp24*(eta_s*(residual_tmp25*residual_tmp96 + residual_tmp33*residual_tmp94) - residual_tmp105*residual_tmp53) + residual_tmp33*residual_tmp85 + residual_tmp88) + u2_direction_grad_1*(mu*(s_t(4)*residual_tmp187 + residual_tmp207) + residual_tmp15*residual_tmp208 + residual_tmp210 + residual_tmp24*(eta_s*(residual_tmp217*residual_tmp33 + residual_tmp220*residual_tmp25 - residual_tmp221) - residual_tmp204*u1_grad_2 - residual_tmp216*residual_tmp53) + residual_tmp57*residual_tmp85) + u2_direction_grad_2*(mu*(s_t(2)*residual_tmp182*residual_tmp227 - residual_tmp244) + residual_tmp15*residual_tmp245 + residual_tmp24*(eta_s*(residual_tmp25*residual_tmp256 + residual_tmp254*residual_tmp33 + residual_tmp257) + residual_tmp2*residual_tmp204 - residual_tmp253*residual_tmp53) + residual_tmp246 + residual_tmp247*residual_tmp85);
      const s_t grad_coeff0_1 = u0_direction_grad_0*(mu*(-residual_tmp126 + s_t(2)*residual_tmp278*u0_grad_1 - residual_tmp279*u0_grad_1) + residual_tmp112*residual_tmp262 + residual_tmp121*residual_tmp263 + residual_tmp24*(-eta_s*(-residual_tmp114*residual_tmp57 + residual_tmp116*residual_tmp68 - residual_tmp150 - residual_tmp280) + ((s_t(1) / s_t(3)))*residual_tmp120*residual_tmp60)) + u0_direction_grad_1*(mu*(residual_tmp108 + residual_tmp266) + residual_tmp128*residual_tmp262 + residual_tmp24*(-eta_s*(-residual_tmp144*residual_tmp57 + residual_tmp149*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp139*residual_tmp60) + residual_tmp263*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp262 + residual_tmp24*(-eta_s*(residual_tmp173*residual_tmp68 - residual_tmp178*residual_tmp57 + residual_tmp277) + ((s_t(1) / s_t(3)))*residual_tmp168*residual_tmp60) + residual_tmp263*residual_tmp62 + residual_tmp273) + u1_direction_grad_0*(residual_tmp10*residual_tmp262 + residual_tmp24*(-eta_s*(residual_tmp241 + residual_tmp32*residual_tmp68 - residual_tmp41*residual_tmp57) + residual_tmp282*residual_tmp8 + residual_tmp290*residual_tmp52) + residual_tmp25*residual_tmp263 + residual_tmp292) + u1_direction_grad_1*(residual_tmp226*residual_tmp262 + residual_tmp230*residual_tmp263 + residual_tmp24*(-eta_s*(residual_tmp237*residual_tmp68 - residual_tmp240*residual_tmp57) + ((s_t(1) / s_t(3)))*residual_tmp236*residual_tmp60) + residual_tmp268) + u1_direction_grad_2*(residual_tmp188*residual_tmp262 + residual_tmp24*(-eta_s*(residual_tmp198*residual_tmp68 - residual_tmp202*residual_tmp57 - residual_tmp281) + ((s_t(1) / s_t(3)))*residual_tmp197*residual_tmp60 - residual_tmp283) + residual_tmp263*residual_tmp67 + residual_tmp287) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(-residual_tmp221 - residual_tmp57*residual_tmp94 + residual_tmp68*residual_tmp96) + ((s_t(1) / s_t(3)))*residual_tmp105*residual_tmp60 - residual_tmp282*u1_grad_2) + residual_tmp262*residual_tmp90 + residual_tmp263*residual_tmp33 + residual_tmp289) + u2_direction_grad_1*(residual_tmp208*residual_tmp262 + residual_tmp24*(-eta_s*(-residual_tmp217*residual_tmp57 + residual_tmp220*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp216*residual_tmp60) + residual_tmp259 + residual_tmp263*residual_tmp57) + u2_direction_grad_2*(mu*(residual_tmp227*residual_tmp3 + residual_tmp295) + residual_tmp24*(-eta_s*(-residual_tmp254*residual_tmp57 + residual_tmp256*residual_tmp68 + residual_tmp298) + residual_tmp253*residual_tmp290 + residual_tmp299) + residual_tmp245*residual_tmp262 + residual_tmp247*residual_tmp263 + residual_tmp297);
      const s_t grad_coeff0_2 = u0_direction_grad_0*(mu*(-residual_tmp155 + s_t(2)*residual_tmp278*u0_grad_2 - residual_tmp279*u0_grad_2) + residual_tmp112*residual_tmp303 + residual_tmp121*residual_tmp304 + residual_tmp24*(-eta_s*(residual_tmp114*residual_tmp63 - residual_tmp116*residual_tmp67 - residual_tmp179 - residual_tmp306) + ((s_t(1) / s_t(3)))*residual_tmp120*residual_tmp62)) + u0_direction_grad_1*(residual_tmp128*residual_tmp303 + residual_tmp24*(-eta_s*(residual_tmp144*residual_tmp63 - residual_tmp149*residual_tmp67 - residual_tmp277) + ((s_t(1) / s_t(3)))*residual_tmp139*residual_tmp62) + residual_tmp273 + residual_tmp304*residual_tmp60) + u0_direction_grad_2*(mu*(residual_tmp111 + residual_tmp266 + s_t(2)) + residual_tmp157*residual_tmp303 + residual_tmp24*(-eta_s*(-residual_tmp173*residual_tmp67 + residual_tmp178*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp168*residual_tmp62) + residual_tmp304*residual_tmp62) + u1_direction_grad_0*(residual_tmp10*residual_tmp303 + residual_tmp24*(-eta_s*(-residual_tmp203 - residual_tmp32*residual_tmp67 + residual_tmp41*residual_tmp63) - residual_tmp282*u2_grad_1 + ((s_t(1) / s_t(3)))*residual_tmp52*residual_tmp62) + residual_tmp25*residual_tmp304 + residual_tmp310) + u1_direction_grad_1*(mu*(residual_tmp0*residual_tmp222 + residual_tmp315) + residual_tmp226*residual_tmp303 + residual_tmp230*residual_tmp304 + residual_tmp24*(-eta_s*(-residual_tmp237*residual_tmp67 + residual_tmp240*residual_tmp63 + residual_tmp281) + residual_tmp236*residual_tmp311 + residual_tmp283) + residual_tmp316) + u1_direction_grad_2*(residual_tmp188*residual_tmp303 + residual_tmp24*(-eta_s*(-residual_tmp198*residual_tmp67 + residual_tmp202*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp197*residual_tmp62) + residual_tmp300 + residual_tmp304*residual_tmp67) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(residual_tmp257 + residual_tmp63*residual_tmp94 - residual_tmp67*residual_tmp96) + residual_tmp105*residual_tmp311 + residual_tmp2*residual_tmp282) + residual_tmp303*residual_tmp90 + residual_tmp304*residual_tmp33 + residual_tmp313) + u2_direction_grad_1*(residual_tmp208*residual_tmp303 + residual_tmp24*(-eta_s*(residual_tmp217*residual_tmp63 - residual_tmp220*residual_tmp67 - residual_tmp298) + ((s_t(1) / s_t(3)))*residual_tmp216*residual_tmp62 - residual_tmp299) + residual_tmp304*residual_tmp57 + residual_tmp308) + u2_direction_grad_2*(residual_tmp24*(-eta_s*(residual_tmp254*residual_tmp63 - residual_tmp256*residual_tmp67) + ((s_t(1) / s_t(3)))*residual_tmp253*residual_tmp62) + residual_tmp245*residual_tmp303 + residual_tmp247*residual_tmp304 + residual_tmp305);
      const s_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp112*residual_tmp320 + residual_tmp121*residual_tmp326 + residual_tmp24*(-eta_s*(residual_tmp116*residual_tmp20 - residual_tmp33*residual_tmp334) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp335) + residual_tmp5) + u0_direction_grad_1*(residual_tmp128*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp149*residual_tmp20 - residual_tmp33*residual_tmp365 + residual_tmp366) + residual_tmp355*residual_tmp8 + residual_tmp363*residual_tmp364) + residual_tmp292 + residual_tmp326*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp173*residual_tmp20 - residual_tmp33*residual_tmp353 - residual_tmp354) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp352 - residual_tmp356) + residual_tmp310 + residual_tmp326*residual_tmp62) + u1_direction_grad_0*(mu*(residual_tmp328 + residual_tmp330) + residual_tmp10*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp32 - residual_tmp33*residual_tmp332) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp333) + residual_tmp25*residual_tmp326) + u1_direction_grad_1*(mu*(s_t(2)*residual_tmp222*u1_grad_0 - residual_tmp344*u1_grad_0 - residual_tmp346) + residual_tmp226*residual_tmp320 + residual_tmp230*residual_tmp326 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp237 - residual_tmp280 - residual_tmp33*residual_tmp350 - residual_tmp351) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp348)) + u1_direction_grad_2*(residual_tmp188*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp198*residual_tmp20 - residual_tmp33*residual_tmp341 + residual_tmp343) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp339) + residual_tmp326*residual_tmp67 + residual_tmp337) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp96 - residual_tmp322*residual_tmp33) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp323) + residual_tmp319 + residual_tmp320*residual_tmp90 + residual_tmp326*residual_tmp33) + u2_direction_grad_1*(residual_tmp208*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp220 - residual_tmp33*residual_tmp358 - residual_tmp359) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp357 - residual_tmp355*u0_grad_2) + residual_tmp326*residual_tmp57 + residual_tmp362) + u2_direction_grad_2*(mu*(residual_tmp124*residual_tmp227 + residual_tmp368) + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp256 - residual_tmp33*residual_tmp372 + residual_tmp373) + residual_tmp364*residual_tmp371 + residual_tmp374) + residual_tmp245*residual_tmp320 + residual_tmp247*residual_tmp326 + residual_tmp370);
      const s_t grad_coeff1_1 = u0_direction_grad_0*(mu*(s_t(2)*residual_tmp2*residual_tmp278 - residual_tmp225) + residual_tmp112*residual_tmp375 + residual_tmp121*residual_tmp378 + residual_tmp229 + residual_tmp24*(eta_s*(residual_tmp116*residual_tmp60 + residual_tmp334*residual_tmp57 + residual_tmp366) - residual_tmp335*residual_tmp376 + residual_tmp377*residual_tmp8)) + u0_direction_grad_1*(residual_tmp128*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp149*residual_tmp60 + residual_tmp365*residual_tmp57) - residual_tmp363*residual_tmp376) + residual_tmp268 + residual_tmp378*residual_tmp60) + u0_direction_grad_2*(mu*(residual_tmp315 + s_t(4)*residual_tmp89) + residual_tmp157*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp173*residual_tmp60 + residual_tmp353*residual_tmp57 - residual_tmp386) - residual_tmp352*residual_tmp376 - residual_tmp377*u2_grad_0) + residual_tmp316 + residual_tmp378*residual_tmp62) + u1_direction_grad_0*(-mu*residual_tmp346 + residual_tmp10*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp152 + residual_tmp32*residual_tmp60 + residual_tmp332*residual_tmp57 - residual_tmp351) - residual_tmp333*residual_tmp376) + residual_tmp25*residual_tmp378) + u1_direction_grad_1*(mu*(residual_tmp328 + residual_tmp382) + residual_tmp226*residual_tmp375 + residual_tmp230*residual_tmp378 + residual_tmp24*(eta_s*(residual_tmp237*residual_tmp60 + residual_tmp350*residual_tmp57) - residual_tmp348*residual_tmp376)) + u1_direction_grad_2*(-mu*residual_tmp384 + residual_tmp188*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp198*residual_tmp60 + residual_tmp276 + residual_tmp341*residual_tmp57 + residual_tmp385) - residual_tmp339*residual_tmp376) + residual_tmp378*residual_tmp67) + u2_direction_grad_0*(mu*(s_t(4)*residual_tmp156 + residual_tmp387) + residual_tmp24*(eta_s*(residual_tmp322*residual_tmp57 - residual_tmp359 + residual_tmp60*residual_tmp96) - residual_tmp323*residual_tmp376 - residual_tmp377*u0_grad_2) + residual_tmp33*residual_tmp378 + residual_tmp375*residual_tmp90 + residual_tmp388) + u2_direction_grad_1*(residual_tmp208*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp220*residual_tmp60 + residual_tmp358*residual_tmp57) - residual_tmp357*residual_tmp376) + residual_tmp378*residual_tmp57 + residual_tmp380) + u2_direction_grad_2*(mu*(s_t(2)*residual_tmp2*residual_tmp227 - residual_tmp390) + residual_tmp24*(eta_s*(residual_tmp256*residual_tmp60 + residual_tmp372*residual_tmp57 + residual_tmp392) + residual_tmp182*residual_tmp377 - residual_tmp371*residual_tmp376) + residual_tmp245*residual_tmp375 + residual_tmp247*residual_tmp378 + residual_tmp391);
      const s_t grad_coeff1_2 = u0_direction_grad_0*(mu*(residual_tmp186 + residual_tmp269*residual_tmp278) + residual_tmp112*residual_tmp393 + residual_tmp121*residual_tmp394 + residual_tmp191 + residual_tmp24*(-eta_s*(-residual_tmp116*residual_tmp62 + residual_tmp334*residual_tmp63 + residual_tmp354) + residual_tmp335*residual_tmp398 + residual_tmp356)) + u0_direction_grad_1*(residual_tmp128*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp149*residual_tmp62 + residual_tmp365*residual_tmp63 - residual_tmp386) - residual_tmp355*u2_grad_0 + ((s_t(1) / s_t(3)))*residual_tmp363*residual_tmp67) + residual_tmp287 + residual_tmp394*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp173*residual_tmp62 + residual_tmp353*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp352*residual_tmp67) + residual_tmp300 + residual_tmp394*residual_tmp62) + u1_direction_grad_0*(residual_tmp10*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp32*residual_tmp62 + residual_tmp332*residual_tmp63 - residual_tmp343) + ((s_t(1) / s_t(3)))*residual_tmp333*residual_tmp67) + residual_tmp25*residual_tmp394 + residual_tmp337) + u1_direction_grad_1*(mu*(s_t(2)*residual_tmp222*u1_grad_2 - residual_tmp344*u1_grad_2 - residual_tmp384) + residual_tmp226*residual_tmp393 + residual_tmp230*residual_tmp394 + residual_tmp24*(-eta_s*(-residual_tmp237*residual_tmp62 - residual_tmp275 + residual_tmp350*residual_tmp63 - residual_tmp396) + ((s_t(1) / s_t(3)))*residual_tmp348*residual_tmp67)) + u1_direction_grad_2*(mu*(residual_tmp330 + residual_tmp382 + s_t(2)) + residual_tmp188*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp198*residual_tmp62 + residual_tmp341*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp339*residual_tmp67) + residual_tmp394*residual_tmp67) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(residual_tmp322*residual_tmp63 - residual_tmp373 - residual_tmp62*residual_tmp96) + ((s_t(1) / s_t(3)))*residual_tmp323*residual_tmp67 - residual_tmp374) + residual_tmp33*residual_tmp394 + residual_tmp393*residual_tmp90 + residual_tmp397) + u2_direction_grad_1*(residual_tmp208*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp220*residual_tmp62 + residual_tmp358*residual_tmp63 + residual_tmp392) + residual_tmp182*residual_tmp355 + residual_tmp357*residual_tmp398) + residual_tmp394*residual_tmp57 + residual_tmp399) + u2_direction_grad_2*(residual_tmp24*(-eta_s*(-residual_tmp256*residual_tmp62 + residual_tmp372*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp371*residual_tmp67) + residual_tmp245*residual_tmp393 + residual_tmp247*residual_tmp394 + residual_tmp395);
      const s_t grad_coeff2_0 = u0_direction_grad_0*(residual_tmp112*residual_tmp400 + residual_tmp121*residual_tmp403 + residual_tmp24*(-eta_s*(residual_tmp114*residual_tmp20 - residual_tmp25*residual_tmp334) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp407) + residual_tmp88) + u0_direction_grad_1*(residual_tmp128*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp144*residual_tmp20 - residual_tmp25*residual_tmp365 - residual_tmp415) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp414 - residual_tmp417) + residual_tmp289 + residual_tmp403*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp178*residual_tmp20 - residual_tmp25*residual_tmp353 + residual_tmp422) + residual_tmp2*residual_tmp416 + residual_tmp420*residual_tmp421) + residual_tmp313 + residual_tmp403*residual_tmp62) + u1_direction_grad_0*(residual_tmp10*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp41 - residual_tmp25*residual_tmp332) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp401) + residual_tmp25*residual_tmp403 + residual_tmp319) + u1_direction_grad_1*(mu*(residual_tmp122*residual_tmp222 + residual_tmp387) + residual_tmp226*residual_tmp400 + residual_tmp230*residual_tmp403 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp240 - residual_tmp25*residual_tmp350 + residual_tmp424) + residual_tmp421*residual_tmp423 + residual_tmp425) + residual_tmp388) + u1_direction_grad_2*(residual_tmp188*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp202 - residual_tmp25*residual_tmp341 - residual_tmp419) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp418 - residual_tmp416*u0_grad_1) + residual_tmp397 + residual_tmp403*residual_tmp67) + u2_direction_grad_0*(mu*(residual_tmp404 + residual_tmp405) + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp94 - residual_tmp25*residual_tmp322) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp406) + residual_tmp33*residual_tmp403 + residual_tmp400*residual_tmp90) + u2_direction_grad_1*(residual_tmp208*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp217 - residual_tmp25*residual_tmp358 + residual_tmp410) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp409) + residual_tmp403*residual_tmp57 + residual_tmp408) + u2_direction_grad_2*(mu*(s_t(2)*residual_tmp227*u2_grad_0 - residual_tmp411*u2_grad_0 - residual_tmp412) + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp254 - residual_tmp25*residual_tmp372 - residual_tmp306 - residual_tmp342) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp413) + residual_tmp245*residual_tmp400 + residual_tmp247*residual_tmp403);
      const s_t grad_coeff2_1 = u0_direction_grad_0*(mu*(residual_tmp207 + residual_tmp271*residual_tmp278) + residual_tmp112*residual_tmp426 + residual_tmp121*residual_tmp427 + residual_tmp210 + residual_tmp24*(-eta_s*(-residual_tmp114*residual_tmp60 + residual_tmp334*residual_tmp68 + residual_tmp415) + residual_tmp407*residual_tmp431 + residual_tmp417)) + u0_direction_grad_1*(residual_tmp128*residual_tmp426 + residual_tmp24*(-eta_s*(-residual_tmp144*residual_tmp60 + residual_tmp365*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp414*residual_tmp57) + residual_tmp259 + residual_tmp427*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp426 + residual_tmp24*(-eta_s*(-residual_tmp178*residual_tmp60 + residual_tmp353*residual_tmp68 - residual_tmp430) - residual_tmp416*u1_grad_0 + ((s_t(1) / s_t(3)))*residual_tmp420*residual_tmp57) + residual_tmp308 + residual_tmp427*residual_tmp62) + u1_direction_grad_0*(residual_tmp10*residual_tmp426 + residual_tmp24*(-eta_s*(residual_tmp332*residual_tmp68 - residual_tmp41*residual_tmp60 - residual_tmp424) + ((s_t(1) / s_t(3)))*residual_tmp401*residual_tmp57 - residual_tmp425) + residual_tmp25*residual_tmp427 + residual_tmp362) + u1_direction_grad_1*(residual_tmp226*residual_tmp426 + residual_tmp230*residual_tmp427 + residual_tmp24*(-eta_s*(-residual_tmp240*residual_tmp60 + residual_tmp350*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp423*residual_tmp57) + residual_tmp380) + u1_direction_grad_2*(residual_tmp188*residual_tmp426 + residual_tmp24*(-eta_s*(-residual_tmp202*residual_tmp60 + residual_tmp341*residual_tmp68 + residual_tmp432) + residual_tmp182*residual_tmp416 + residual_tmp418*residual_tmp431) + residual_tmp399 + residual_tmp427*residual_tmp67) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(residual_tmp322*residual_tmp68 - residual_tmp410 - residual_tmp60*residual_tmp94) + ((s_t(1) / s_t(3)))*residual_tmp406*residual_tmp57) + residual_tmp33*residual_tmp427 + residual_tmp408 + residual_tmp426*residual_tmp90) + u2_direction_grad_1*(mu*(residual_tmp404 + residual_tmp428) + residual_tmp208*residual_tmp426 + residual_tmp24*(-eta_s*(-residual_tmp217*residual_tmp60 + residual_tmp358*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp409*residual_tmp57) + residual_tmp427*residual_tmp57) + u2_direction_grad_2*(mu*(s_t(2)*residual_tmp227*u2_grad_1 - residual_tmp411*u2_grad_1 - residual_tmp429) + residual_tmp24*(-eta_s*(-residual_tmp254*residual_tmp60 - residual_tmp274 + residual_tmp372*residual_tmp68 - residual_tmp396) + ((s_t(1) / s_t(3)))*residual_tmp413*residual_tmp57) + residual_tmp245*residual_tmp426 + residual_tmp247*residual_tmp427);
      const s_t grad_coeff2_2 = u0_direction_grad_0*(mu*(-residual_tmp244 + s_t(2)*residual_tmp278*residual_tmp8) + residual_tmp112*residual_tmp433 + residual_tmp121*residual_tmp436 + residual_tmp24*(eta_s*(residual_tmp114*residual_tmp62 + residual_tmp334*residual_tmp67 + residual_tmp422) + residual_tmp2*residual_tmp435 - residual_tmp407*residual_tmp434) + residual_tmp246) + u0_direction_grad_1*(mu*(residual_tmp295 + s_t(4)*residual_tmp9) + residual_tmp128*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp144*residual_tmp62 + residual_tmp365*residual_tmp67 - residual_tmp430) - residual_tmp414*residual_tmp434 - residual_tmp435*u1_grad_0) + residual_tmp297 + residual_tmp436*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp178*residual_tmp62 + residual_tmp353*residual_tmp67) - residual_tmp420*residual_tmp434) + residual_tmp305 + residual_tmp436*residual_tmp62) + u1_direction_grad_0*(mu*(s_t(4)*residual_tmp127 + residual_tmp368) + residual_tmp10*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp332*residual_tmp67 + residual_tmp41*residual_tmp62 - residual_tmp419) - residual_tmp401*residual_tmp434 - residual_tmp435*u0_grad_1) + residual_tmp25*residual_tmp436 + residual_tmp370) + u1_direction_grad_1*(mu*(s_t(2)*residual_tmp222*residual_tmp8 - residual_tmp390) + residual_tmp226*residual_tmp433 + residual_tmp230*residual_tmp436 + residual_tmp24*(eta_s*(residual_tmp240*residual_tmp62 + residual_tmp350*residual_tmp67 + residual_tmp432) + residual_tmp182*residual_tmp435 - residual_tmp423*residual_tmp434) + residual_tmp391) + u1_direction_grad_2*(residual_tmp188*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp202*residual_tmp62 + residual_tmp341*residual_tmp67) - residual_tmp418*residual_tmp434) + residual_tmp395 + residual_tmp436*residual_tmp67) + u2_direction_grad_0*(-mu*residual_tmp412 + residual_tmp24*(eta_s*(residual_tmp181 + residual_tmp322*residual_tmp67 - residual_tmp342 + residual_tmp62*residual_tmp94) - residual_tmp406*residual_tmp434) + residual_tmp33*residual_tmp436 + residual_tmp433*residual_tmp90) + u2_direction_grad_1*(-mu*residual_tmp429 + residual_tmp208*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp217*residual_tmp62 - residual_tmp274 + residual_tmp358*residual_tmp67 + residual_tmp385) - residual_tmp409*residual_tmp434) + residual_tmp436*residual_tmp57) + u2_direction_grad_2*(mu*(residual_tmp405 + residual_tmp428 + s_t(2)) + residual_tmp24*(eta_s*(residual_tmp254*residual_tmp62 + residual_tmp372*residual_tmp67) - residual_tmp413*residual_tmp434) + residual_tmp245*residual_tmp433 + residual_tmp247*residual_tmp436);
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
      output[0][lane] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1 + grad_coeff0_2_value * test0_grad2);
      output[1][lane] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1 + grad_coeff1_2_value * test0_grad2);
      output[2][lane] += q_weight[q] * det * (grad_coeff2_0_value * test0_grad0 + grad_coeff2_1_value * test0_grad1 + grad_coeff2_2_value * test0_grad2);
      output[3][lane] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1 + grad_coeff0_2_value * test1_grad2);
      output[4][lane] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1 + grad_coeff1_2_value * test1_grad2);
      output[5][lane] += q_weight[q] * det * (grad_coeff2_0_value * test1_grad0 + grad_coeff2_1_value * test1_grad1 + grad_coeff2_2_value * test1_grad2);
      output[6][lane] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1 + grad_coeff0_2_value * test2_grad2);
      output[7][lane] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1 + grad_coeff1_2_value * test2_grad2);
      output[8][lane] += q_weight[q] * det * (grad_coeff2_0_value * test2_grad0 + grad_coeff2_1_value * test2_grad1 + grad_coeff2_2_value * test2_grad2);
      output[9][lane] += q_weight[q] * det * (grad_coeff0_0_value * test3_grad0 + grad_coeff0_1_value * test3_grad1 + grad_coeff0_2_value * test3_grad2);
      output[10][lane] += q_weight[q] * det * (grad_coeff1_0_value * test3_grad0 + grad_coeff1_1_value * test3_grad1 + grad_coeff1_2_value * test3_grad2);
      output[11][lane] += q_weight[q] * det * (grad_coeff2_0_value * test3_grad0 + grad_coeff2_1_value * test3_grad1 + grad_coeff2_2_value * test3_grad2);
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_residual_merit_d3_simplex_tet4_jacobian_action_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR q_weight,
    const s_t current[3 * NS][VS],
    const s_t previous[3 * NS][VS],
    const s_t direction[3 * NS][VS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t output[3 * NS][VS]
) {
  for (int q = 0; q < NQ; ++q) {
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
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
      const s_t u0_grad_0_ref = -(current[0][lane]) + current[3][lane];
      const s_t u0_grad_1_ref = -(current[0][lane]) + current[6][lane];
      const s_t u0_grad_2_ref = -(current[0][lane]) + current[9][lane];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
      const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
      const s_t u0_old_grad_0_ref = -(previous[0][lane]) + previous[3][lane];
      const s_t u0_old_grad_1_ref = -(previous[0][lane]) + previous[6][lane];
      const s_t u0_old_grad_2_ref = -(previous[0][lane]) + previous[9][lane];
      const s_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
      const s_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
      const s_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
      const s_t u0_direction_grad_0_ref = -(direction[0][lane]) + direction[3][lane];
      const s_t u0_direction_grad_1_ref = -(direction[0][lane]) + direction[6][lane];
      const s_t u0_direction_grad_2_ref = -(direction[0][lane]) + direction[9][lane];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
      const s_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
      const s_t u1_grad_0_ref = -(current[1][lane]) + current[4][lane];
      const s_t u1_grad_1_ref = -(current[1][lane]) + current[7][lane];
      const s_t u1_grad_2_ref = -(current[1][lane]) + current[10][lane];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
      const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
      const s_t u1_old_grad_0_ref = -(previous[1][lane]) + previous[4][lane];
      const s_t u1_old_grad_1_ref = -(previous[1][lane]) + previous[7][lane];
      const s_t u1_old_grad_2_ref = -(previous[1][lane]) + previous[10][lane];
      const s_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
      const s_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
      const s_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
      const s_t u1_direction_grad_0_ref = -(direction[1][lane]) + direction[4][lane];
      const s_t u1_direction_grad_1_ref = -(direction[1][lane]) + direction[7][lane];
      const s_t u1_direction_grad_2_ref = -(direction[1][lane]) + direction[10][lane];
      const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
      const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
      const s_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
      const s_t u2_grad_0_ref = -(current[2][lane]) + current[5][lane];
      const s_t u2_grad_1_ref = -(current[2][lane]) + current[8][lane];
      const s_t u2_grad_2_ref = -(current[2][lane]) + current[11][lane];
      const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
      const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
      const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
      const s_t u2_old_grad_0_ref = -(previous[2][lane]) + previous[5][lane];
      const s_t u2_old_grad_1_ref = -(previous[2][lane]) + previous[8][lane];
      const s_t u2_old_grad_2_ref = -(previous[2][lane]) + previous[11][lane];
      const s_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
      const s_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
      const s_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
      const s_t u2_direction_grad_0_ref = -(direction[2][lane]) + direction[5][lane];
      const s_t u2_direction_grad_1_ref = -(direction[2][lane]) + direction[8][lane];
      const s_t u2_direction_grad_2_ref = -(direction[2][lane]) + direction[11][lane];
      const s_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
      const s_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
      const s_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = s_t(2)*u0_grad_2;
      const s_t residual_tmp1 = residual_tmp0*u1_grad_2;
      const s_t residual_tmp2 = u1_grad_1 + s_t(1);
      const s_t residual_tmp3 = s_t(2)*u0_grad_1;
      const s_t residual_tmp4 = residual_tmp2*residual_tmp3;
      const s_t residual_tmp5 = mu*(-residual_tmp1 - residual_tmp4);
      const s_t residual_tmp6 = u0_grad_2*u2_grad_1;
      const s_t residual_tmp7 = -residual_tmp6;
      const s_t residual_tmp8 = u2_grad_2 + s_t(1);
      const s_t residual_tmp9 = residual_tmp8*u0_grad_1;
      const s_t residual_tmp10 = -residual_tmp7 - residual_tmp9;
      const s_t residual_tmp11 = u1_grad_2*u2_grad_1;
      const s_t residual_tmp12 = s_t(2)*residual_tmp11;
      const s_t residual_tmp13 = -s_t(2)*residual_tmp2*residual_tmp8;
      const s_t residual_tmp14 = ((s_t(1) / s_t(2)))*lmbda;
      const s_t residual_tmp15 = residual_tmp14*(-residual_tmp12 - residual_tmp13);
      const s_t residual_tmp16 = u0_grad_0*u1_grad_1;
      const s_t residual_tmp17 = u0_grad_1*u1_grad_2;
      const s_t residual_tmp18 = u0_grad_1*u1_grad_0;
      const s_t residual_tmp19 = u0_grad_2*u2_grad_0;
      const s_t residual_tmp20 = -residual_tmp11 + residual_tmp2 + u1_grad_1*u2_grad_2 + u2_grad_2;
      const s_t residual_tmp21 = -residual_tmp19 + u0_grad_0*u2_grad_2 + u0_grad_0;
      const s_t residual_tmp22 = residual_tmp16 - residual_tmp18;
      const s_t residual_tmp23 = -residual_tmp11*u0_grad_0 + residual_tmp16*u2_grad_2 + residual_tmp17*u2_grad_0 - residual_tmp18*u2_grad_2 - residual_tmp19*u1_grad_1 + residual_tmp20 + residual_tmp21 + residual_tmp22 + residual_tmp6*u1_grad_0;
      const s_t residual_tmp24 = pow_m1(residual_tmp23);
      const s_t residual_tmp25 = residual_tmp7 + u0_grad_1*u2_grad_2 + u0_grad_1;
      const s_t residual_tmp26 = residual_tmp20*u_dt_shift;
      const s_t residual_tmp27 = u1_grad_2*u_dt_shift + u1_old_grad_2;
      const s_t residual_tmp28 = residual_tmp27*u2_grad_1;
      const s_t residual_tmp29 = u1_grad_1*u_dt_shift + u1_old_grad_1;
      const s_t residual_tmp30 = residual_tmp29*residual_tmp8;
      const s_t residual_tmp31 = residual_tmp28 - residual_tmp30;
      const s_t residual_tmp32 = -residual_tmp26 - residual_tmp31;
      const s_t residual_tmp33 = -residual_tmp17 + u0_grad_2*u1_grad_1 + u0_grad_2;
      const s_t residual_tmp34 = u2_grad_1*u_dt_shift + u2_old_grad_1;
      const s_t residual_tmp35 = residual_tmp34*residual_tmp8;
      const s_t residual_tmp36 = u2_grad_2*u_dt_shift + u2_old_grad_2;
      const s_t residual_tmp37 = residual_tmp36*u2_grad_1;
      const s_t residual_tmp38 = u0_grad_2*u_dt_shift + u0_old_grad_2;
      const s_t residual_tmp39 = u0_grad_1*u_dt_shift + u0_old_grad_1;
      const s_t residual_tmp40 = residual_tmp38*u0_grad_1 - residual_tmp39*u0_grad_2;
      const s_t residual_tmp41 = residual_tmp35 - residual_tmp37 + residual_tmp40;
      const s_t residual_tmp42 = residual_tmp25*u_dt_shift;
      const s_t residual_tmp43 = residual_tmp36*u0_grad_1;
      const s_t residual_tmp44 = residual_tmp34*u0_grad_2;
      const s_t residual_tmp45 = residual_tmp42 + residual_tmp43 - residual_tmp44;
      const s_t residual_tmp46 = residual_tmp39*residual_tmp8;
      const s_t residual_tmp47 = residual_tmp38*u2_grad_1;
      const s_t residual_tmp48 = residual_tmp46 - residual_tmp47;
      const s_t residual_tmp49 = s_t(3)*eta_b;
      const s_t residual_tmp50 = residual_tmp49*(residual_tmp45 + residual_tmp48);
      const s_t residual_tmp51 = s_t(2)*eta_s;
      const s_t residual_tmp52 = residual_tmp50 + residual_tmp51*(s_t(2)*residual_tmp39*residual_tmp8 - residual_tmp45 - s_t(2)*residual_tmp47);
      const s_t residual_tmp53 = ((s_t(1) / s_t(3)))*residual_tmp20;
      const s_t residual_tmp54 = pow_m2(residual_tmp23);
      const s_t residual_tmp55 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t residual_tmp56 = u0_grad_2*u1_grad_0;
      const s_t residual_tmp57 = -residual_tmp56 + u0_grad_0*u1_grad_2 + u1_grad_2;
      const s_t residual_tmp58 = u1_grad_2*u2_grad_0;
      const s_t residual_tmp59 = -residual_tmp58;
      const s_t residual_tmp60 = residual_tmp59 + u1_grad_0*u2_grad_2 + u1_grad_0;
      const s_t residual_tmp61 = u1_grad_0*u2_grad_1;
      const s_t residual_tmp62 = -residual_tmp61 + u1_grad_1*u2_grad_0 + u2_grad_0;
      const s_t residual_tmp63 = residual_tmp2 + residual_tmp22 + u0_grad_0;
      const s_t residual_tmp64 = u2_grad_0*u_dt_shift + u2_old_grad_0;
      const s_t residual_tmp65 = -residual_tmp20*residual_tmp64 + residual_tmp33*residual_tmp55 + residual_tmp34*residual_tmp60 + residual_tmp36*residual_tmp62 - residual_tmp38*residual_tmp63 + residual_tmp39*residual_tmp57;
      const s_t residual_tmp66 = u0_grad_1*u2_grad_0;
      const s_t residual_tmp67 = -residual_tmp66 + u0_grad_0*u2_grad_1 + u2_grad_1;
      const s_t residual_tmp68 = residual_tmp21 + residual_tmp8;
      const s_t residual_tmp69 = u1_grad_0*u_dt_shift + u1_old_grad_0;
      const s_t residual_tmp70 = -residual_tmp20*residual_tmp69 + residual_tmp25*residual_tmp55 + residual_tmp27*residual_tmp62 + residual_tmp29*residual_tmp60 + residual_tmp38*residual_tmp67 - residual_tmp39*residual_tmp68;
      const s_t residual_tmp71 = residual_tmp25*residual_tmp69;
      const s_t residual_tmp72 = residual_tmp27*residual_tmp67;
      const s_t residual_tmp73 = residual_tmp33*residual_tmp64;
      const s_t residual_tmp74 = residual_tmp34*residual_tmp57;
      const s_t residual_tmp75 = residual_tmp29*residual_tmp68;
      const s_t residual_tmp76 = -residual_tmp75;
      const s_t residual_tmp77 = residual_tmp36*residual_tmp63;
      const s_t residual_tmp78 = -residual_tmp77;
      const s_t residual_tmp79 = residual_tmp71 + residual_tmp72 + residual_tmp73 + residual_tmp74 + residual_tmp76 + residual_tmp78;
      const s_t residual_tmp80 = residual_tmp20*residual_tmp55;
      const s_t residual_tmp81 = residual_tmp38*residual_tmp62 + residual_tmp39*residual_tmp60 - residual_tmp80;
      const s_t residual_tmp82 = residual_tmp49*(residual_tmp79 + residual_tmp81);
      const s_t residual_tmp83 = residual_tmp51*(s_t(2)*residual_tmp38*residual_tmp62 + s_t(2)*residual_tmp39*residual_tmp60 - residual_tmp79 - s_t(2)*residual_tmp80) + residual_tmp82;
      const s_t residual_tmp84 = -residual_tmp83;
      const s_t residual_tmp85 = residual_tmp54*(eta_s*(residual_tmp25*residual_tmp70 + residual_tmp33*residual_tmp65) + residual_tmp53*residual_tmp84);
      const s_t residual_tmp86 = residual_tmp3*u2_grad_1;
      const s_t residual_tmp87 = residual_tmp0*residual_tmp8;
      const s_t residual_tmp88 = mu*(-residual_tmp86 - residual_tmp87);
      const s_t residual_tmp89 = residual_tmp2*u0_grad_2;
      const s_t residual_tmp90 = residual_tmp17 - residual_tmp89;
      const s_t residual_tmp91 = residual_tmp34*u1_grad_2;
      const s_t residual_tmp92 = residual_tmp2*residual_tmp36;
      const s_t residual_tmp93 = residual_tmp91 - residual_tmp92;
      const s_t residual_tmp94 = -residual_tmp26 - residual_tmp93;
      const s_t residual_tmp95 = -residual_tmp2*residual_tmp27 + residual_tmp29*u1_grad_2;
      const s_t residual_tmp96 = -residual_tmp40 - residual_tmp95;
      const s_t residual_tmp97 = residual_tmp33*u_dt_shift;
      const s_t residual_tmp98 = residual_tmp29*u0_grad_2;
      const s_t residual_tmp99 = residual_tmp27*u0_grad_1;
      const s_t residual_tmp100 = residual_tmp97 + residual_tmp98 - residual_tmp99;
      const s_t residual_tmp101 = residual_tmp2*residual_tmp38;
      const s_t residual_tmp102 = residual_tmp39*u1_grad_2;
      const s_t residual_tmp103 = residual_tmp101 - residual_tmp102;
      const s_t residual_tmp104 = residual_tmp49*(residual_tmp100 + residual_tmp103);
      const s_t residual_tmp105 = residual_tmp104 + residual_tmp51*(-residual_tmp100 - s_t(2)*residual_tmp102 + s_t(2)*residual_tmp2*residual_tmp38);
      const s_t residual_tmp106 = s_t(2)*pow_2(u1_grad_2);
      const s_t residual_tmp107 = s_t(2)*pow_2(residual_tmp8) + s_t(2);
      const s_t residual_tmp108 = residual_tmp106 + residual_tmp107;
      const s_t residual_tmp109 = s_t(2)*pow_2(u2_grad_1);
      const s_t residual_tmp110 = s_t(2)*pow_2(residual_tmp2);
      const s_t residual_tmp111 = residual_tmp109 + residual_tmp110;
      const s_t residual_tmp112 = -residual_tmp11 + residual_tmp2*residual_tmp8;
      const s_t residual_tmp113 = -residual_tmp101 + residual_tmp102;
      const s_t residual_tmp114 = residual_tmp113 + residual_tmp97;
      const s_t residual_tmp115 = -residual_tmp46 + residual_tmp47;
      const s_t residual_tmp116 = residual_tmp115 + residual_tmp42;
      const s_t residual_tmp117 = residual_tmp26 - residual_tmp91 + residual_tmp92;
      const s_t residual_tmp118 = -residual_tmp28 + residual_tmp30;
      const s_t residual_tmp119 = residual_tmp49*(-residual_tmp117 - residual_tmp118);
      const s_t residual_tmp120 = residual_tmp119 + residual_tmp51*(-s_t(2)*residual_tmp26 - residual_tmp31 - residual_tmp93);
      const s_t residual_tmp121 = -residual_tmp20;
      const s_t residual_tmp122 = s_t(2)*u2_grad_0;
      const s_t residual_tmp123 = residual_tmp122*u2_grad_1;
      const s_t residual_tmp124 = s_t(2)*u1_grad_0;
      const s_t residual_tmp125 = residual_tmp124*residual_tmp2;
      const s_t residual_tmp126 = residual_tmp123 + residual_tmp125;
      const s_t residual_tmp127 = residual_tmp8*u1_grad_0;
      const s_t residual_tmp128 = -residual_tmp127 - residual_tmp59;
      const s_t residual_tmp129 = residual_tmp60*u_dt_shift;
      const s_t residual_tmp130 = residual_tmp36*u1_grad_0;
      const s_t residual_tmp131 = residual_tmp64*u1_grad_2;
      const s_t residual_tmp132 = residual_tmp129 + residual_tmp130 - residual_tmp131;
      const s_t residual_tmp133 = residual_tmp69*residual_tmp8;
      const s_t residual_tmp134 = residual_tmp27*u2_grad_0;
      const s_t residual_tmp135 = residual_tmp133 - residual_tmp134;
      const s_t residual_tmp136 = residual_tmp49*(residual_tmp132 + residual_tmp135);
      const s_t residual_tmp137 = -residual_tmp133 + residual_tmp134;
      const s_t residual_tmp138 = -residual_tmp130 + residual_tmp131;
      const s_t residual_tmp139 = residual_tmp136 + residual_tmp51*(s_t(2)*residual_tmp129 + residual_tmp137 + residual_tmp138);
      const s_t residual_tmp140 = residual_tmp57*u_dt_shift;
      const s_t residual_tmp141 = residual_tmp38*u1_grad_0;
      const s_t residual_tmp142 = residual_tmp55*u1_grad_2;
      const s_t residual_tmp143 = residual_tmp141 - residual_tmp142;
      const s_t residual_tmp144 = residual_tmp140 + residual_tmp143;
      const s_t residual_tmp145 = residual_tmp68*u_dt_shift;
      const s_t residual_tmp146 = residual_tmp38*u2_grad_0;
      const s_t residual_tmp147 = residual_tmp55*residual_tmp8;
      const s_t residual_tmp148 = residual_tmp146 - residual_tmp147;
      const s_t residual_tmp149 = -residual_tmp145 - residual_tmp148;
      const s_t residual_tmp150 = residual_tmp65*u1_grad_2;
      const s_t residual_tmp151 = -residual_tmp150;
      const s_t residual_tmp152 = residual_tmp70*residual_tmp8;
      const s_t residual_tmp153 = residual_tmp124*u1_grad_2;
      const s_t residual_tmp154 = residual_tmp122*residual_tmp8;
      const s_t residual_tmp155 = residual_tmp153 + residual_tmp154;
      const s_t residual_tmp156 = residual_tmp2*u2_grad_0;
      const s_t residual_tmp157 = -residual_tmp156 + residual_tmp61;
      const s_t residual_tmp158 = residual_tmp62*u_dt_shift;
      const s_t residual_tmp159 = residual_tmp2*residual_tmp64;
      const s_t residual_tmp160 = residual_tmp34*u1_grad_0;
      const s_t residual_tmp161 = residual_tmp158 + residual_tmp159 - residual_tmp160;
      const s_t residual_tmp162 = residual_tmp29*u2_grad_0;
      const s_t residual_tmp163 = residual_tmp69*u2_grad_1;
      const s_t residual_tmp164 = residual_tmp162 - residual_tmp163;
      const s_t residual_tmp165 = residual_tmp49*(residual_tmp161 + residual_tmp164);
      const s_t residual_tmp166 = -residual_tmp162 + residual_tmp163;
      const s_t residual_tmp167 = -residual_tmp159 + residual_tmp160;
      const s_t residual_tmp168 = residual_tmp165 + residual_tmp51*(s_t(2)*residual_tmp158 + residual_tmp166 + residual_tmp167);
      const s_t residual_tmp169 = residual_tmp67*u_dt_shift;
      const s_t residual_tmp170 = residual_tmp39*u2_grad_0;
      const s_t residual_tmp171 = residual_tmp55*u2_grad_1;
      const s_t residual_tmp172 = residual_tmp170 - residual_tmp171;
      const s_t residual_tmp173 = residual_tmp169 + residual_tmp172;
      const s_t residual_tmp174 = residual_tmp63*u_dt_shift;
      const s_t residual_tmp175 = residual_tmp39*u1_grad_0;
      const s_t residual_tmp176 = residual_tmp2*residual_tmp55;
      const s_t residual_tmp177 = residual_tmp175 - residual_tmp176;
      const s_t residual_tmp178 = -residual_tmp174 - residual_tmp177;
      const s_t residual_tmp179 = residual_tmp70*u2_grad_1;
      const s_t residual_tmp180 = -residual_tmp179;
      const s_t residual_tmp181 = residual_tmp2*residual_tmp65;
      const s_t residual_tmp182 = u0_grad_0 + s_t(1);
      const s_t residual_tmp183 = residual_tmp182*u1_grad_2;
      const s_t residual_tmp184 = s_t(6)*u2_grad_1;
      const s_t residual_tmp185 = s_t(2)*residual_tmp56;
      const s_t residual_tmp186 = residual_tmp184 - residual_tmp185;
      const s_t residual_tmp187 = residual_tmp182*u2_grad_1;
      const s_t residual_tmp188 = -residual_tmp187 + residual_tmp66;
      const s_t residual_tmp189 = lmbda*(-residual_tmp11*residual_tmp182 - residual_tmp18*residual_tmp8 + residual_tmp182*residual_tmp2*residual_tmp8 - residual_tmp19*residual_tmp2 + u0_grad_1*u1_grad_2*u2_grad_0 + u0_grad_2*u1_grad_0*u2_grad_1 + s_t(-1));
      const s_t residual_tmp190 = residual_tmp189*u2_grad_1;
      const s_t residual_tmp191 = -residual_tmp190;
      const s_t residual_tmp192 = residual_tmp182*residual_tmp34;
      const s_t residual_tmp193 = residual_tmp64*u0_grad_1;
      const s_t residual_tmp194 = residual_tmp169 + residual_tmp192 - residual_tmp193;
      const s_t residual_tmp195 = -residual_tmp170 + residual_tmp171;
      const s_t residual_tmp196 = residual_tmp49*(residual_tmp194 + residual_tmp195);
      const s_t residual_tmp197 = residual_tmp196 + residual_tmp51*(-s_t(2)*residual_tmp170 - residual_tmp194 + s_t(2)*residual_tmp55*u2_grad_1);
      const s_t residual_tmp198 = residual_tmp158 + residual_tmp166;
      const s_t residual_tmp199 = residual_tmp34*u2_grad_0;
      const s_t residual_tmp200 = residual_tmp64*u2_grad_1;
      const s_t residual_tmp201 = -residual_tmp182*residual_tmp39 + residual_tmp55*u0_grad_1;
      const s_t residual_tmp202 = -residual_tmp199 + residual_tmp200 - residual_tmp201;
      const s_t residual_tmp203 = residual_tmp65*u0_grad_1;
      const s_t residual_tmp204 = ((s_t(1) / s_t(3)))*residual_tmp84;
      const s_t residual_tmp205 = s_t(6)*u1_grad_2;
      const s_t residual_tmp206 = s_t(2)*residual_tmp66;
      const s_t residual_tmp207 = residual_tmp205 - residual_tmp206;
      const s_t residual_tmp208 = -residual_tmp183 + residual_tmp56;
      const s_t residual_tmp209 = residual_tmp189*u1_grad_2;
      const s_t residual_tmp210 = -residual_tmp209;
      const s_t residual_tmp211 = residual_tmp182*residual_tmp27;
      const s_t residual_tmp212 = residual_tmp69*u0_grad_2;
      const s_t residual_tmp213 = residual_tmp140 + residual_tmp211 - residual_tmp212;
      const s_t residual_tmp214 = -residual_tmp141 + residual_tmp142;
      const s_t residual_tmp215 = residual_tmp49*(residual_tmp213 + residual_tmp214);
      const s_t residual_tmp216 = residual_tmp215 + residual_tmp51*(-s_t(2)*residual_tmp141 - residual_tmp213 + s_t(2)*residual_tmp55*u1_grad_2);
      const s_t residual_tmp217 = residual_tmp129 + residual_tmp138;
      const s_t residual_tmp218 = -residual_tmp182*residual_tmp38 + residual_tmp55*u0_grad_2;
      const s_t residual_tmp219 = residual_tmp27*u1_grad_0 - residual_tmp69*u1_grad_2;
      const s_t residual_tmp220 = -residual_tmp218 - residual_tmp219;
      const s_t residual_tmp221 = residual_tmp70*u0_grad_2;
      const s_t residual_tmp222 = s_t(2)*u1_grad_1 + s_t(2);
      const s_t residual_tmp223 = s_t(2)*residual_tmp18;
      const s_t residual_tmp224 = s_t(6)*u2_grad_2 + s_t(6);
      const s_t residual_tmp225 = residual_tmp223 + residual_tmp224;
      const s_t residual_tmp226 = residual_tmp182*residual_tmp8 - residual_tmp19;
      const s_t residual_tmp227 = s_t(2)*u2_grad_2 + s_t(2);
      const s_t residual_tmp228 = ((s_t(1) / s_t(2)))*residual_tmp189;
      const s_t residual_tmp229 = residual_tmp227*residual_tmp228;
      const s_t residual_tmp230 = -residual_tmp68;
      const s_t residual_tmp231 = residual_tmp182*residual_tmp36;
      const s_t residual_tmp232 = residual_tmp64*u0_grad_2;
      const s_t residual_tmp233 = residual_tmp145 + residual_tmp231 - residual_tmp232;
      const s_t residual_tmp234 = -residual_tmp146 + residual_tmp147;
      const s_t residual_tmp235 = residual_tmp49*(-residual_tmp233 - residual_tmp234);
      const s_t residual_tmp236 = residual_tmp235 + residual_tmp51*(s_t(2)*residual_tmp146 - s_t(2)*residual_tmp147 + residual_tmp233);
      const s_t residual_tmp237 = residual_tmp129 + residual_tmp137;
      const s_t residual_tmp238 = residual_tmp36*u2_grad_0;
      const s_t residual_tmp239 = residual_tmp64*residual_tmp8;
      const s_t residual_tmp240 = residual_tmp218 + residual_tmp238 - residual_tmp239;
      const s_t residual_tmp241 = residual_tmp65*u0_grad_2;
      const s_t residual_tmp242 = s_t(2)*residual_tmp19;
      const s_t residual_tmp243 = s_t(6)*u1_grad_1 + s_t(6);
      const s_t residual_tmp244 = residual_tmp242 + residual_tmp243;
      const s_t residual_tmp245 = -residual_tmp18 + residual_tmp182*residual_tmp2;
      const s_t residual_tmp246 = residual_tmp222*residual_tmp228;
      const s_t residual_tmp247 = -residual_tmp63;
      const s_t residual_tmp248 = residual_tmp182*residual_tmp29;
      const s_t residual_tmp249 = residual_tmp69*u0_grad_1;
      const s_t residual_tmp250 = residual_tmp174 + residual_tmp248 - residual_tmp249;
      const s_t residual_tmp251 = -residual_tmp175 + residual_tmp176;
      const s_t residual_tmp252 = residual_tmp49*(-residual_tmp250 - residual_tmp251);
      const s_t residual_tmp253 = residual_tmp252 + residual_tmp51*(s_t(2)*residual_tmp175 - s_t(2)*residual_tmp176 + residual_tmp250);
      const s_t residual_tmp254 = residual_tmp158 + residual_tmp167;
      const s_t residual_tmp255 = -residual_tmp2*residual_tmp69 + residual_tmp29*u1_grad_0;
      const s_t residual_tmp256 = residual_tmp201 + residual_tmp255;
      const s_t residual_tmp257 = residual_tmp70*u0_grad_1;
      const s_t residual_tmp258 = residual_tmp122*residual_tmp182;
      const s_t residual_tmp259 = mu*(-residual_tmp258 - residual_tmp87);
      const s_t residual_tmp260 = -s_t(2)*residual_tmp58;
      const s_t residual_tmp261 = s_t(2)*residual_tmp127;
      const s_t residual_tmp262 = residual_tmp14*(-residual_tmp260 - residual_tmp261);
      const s_t residual_tmp263 = residual_tmp54*(-eta_s*(-residual_tmp57*residual_tmp65 + residual_tmp68*residual_tmp70) + ((s_t(1) / s_t(3)))*residual_tmp60*residual_tmp83);
      const s_t residual_tmp264 = s_t(2)*pow_2(u1_grad_0);
      const s_t residual_tmp265 = s_t(2)*pow_2(u2_grad_0);
      const s_t residual_tmp266 = residual_tmp264 + residual_tmp265;
      const s_t residual_tmp267 = residual_tmp124*residual_tmp182;
      const s_t residual_tmp268 = mu*(-residual_tmp1 - residual_tmp267);
      const s_t residual_tmp269 = s_t(2)*u1_grad_2;
      const s_t residual_tmp270 = residual_tmp2*residual_tmp269;
      const s_t residual_tmp271 = s_t(2)*u2_grad_1;
      const s_t residual_tmp272 = residual_tmp271*residual_tmp8;
      const s_t residual_tmp273 = mu*(-residual_tmp270 - residual_tmp272);
      const s_t residual_tmp274 = residual_tmp65*u1_grad_0;
      const s_t residual_tmp275 = residual_tmp70*u2_grad_0;
      const s_t residual_tmp276 = -residual_tmp275;
      const s_t residual_tmp277 = residual_tmp274 + residual_tmp276;
      const s_t residual_tmp278 = s_t(2)*u0_grad_0 + s_t(2);
      const s_t residual_tmp279 = s_t(4)*residual_tmp182;
      const s_t residual_tmp280 = -residual_tmp70*residual_tmp8;
      const s_t residual_tmp281 = residual_tmp182*residual_tmp65;
      const s_t residual_tmp282 = ((s_t(1) / s_t(3)))*residual_tmp83;
      const s_t residual_tmp283 = residual_tmp282*u2_grad_0;
      const s_t residual_tmp284 = s_t(6)*u2_grad_0;
      const s_t residual_tmp285 = s_t(2)*residual_tmp89;
      const s_t residual_tmp286 = residual_tmp189*u2_grad_0;
      const s_t residual_tmp287 = mu*(-residual_tmp284 - residual_tmp285 + s_t(4)*u0_grad_1*u1_grad_2) + residual_tmp286;
      const s_t residual_tmp288 = s_t(2)*residual_tmp187;
      const s_t residual_tmp289 = mu*(-residual_tmp205 - residual_tmp288 + s_t(4)*u0_grad_1*u2_grad_0) + residual_tmp209;
      const s_t residual_tmp290 = ((s_t(1) / s_t(3)))*residual_tmp60;
      const s_t residual_tmp291 = -s_t(2)*residual_tmp182*residual_tmp2;
      const s_t residual_tmp292 = mu*(s_t(4)*residual_tmp18 + residual_tmp224 + residual_tmp291) - residual_tmp227*residual_tmp228;
      const s_t residual_tmp293 = -s_t(2)*residual_tmp6;
      const s_t residual_tmp294 = s_t(6)*u1_grad_0;
      const s_t residual_tmp295 = residual_tmp293 + residual_tmp294;
      const s_t residual_tmp296 = residual_tmp189*u1_grad_0;
      const s_t residual_tmp297 = -residual_tmp296;
      const s_t residual_tmp298 = residual_tmp182*residual_tmp70;
      const s_t residual_tmp299 = residual_tmp282*u1_grad_0;
      const s_t residual_tmp300 = mu*(-residual_tmp267 - residual_tmp4);
      const s_t residual_tmp301 = s_t(2)*residual_tmp61;
      const s_t residual_tmp302 = s_t(2)*residual_tmp156;
      const s_t residual_tmp303 = residual_tmp14*(residual_tmp301 - residual_tmp302);
      const s_t residual_tmp304 = residual_tmp54*(-eta_s*(residual_tmp63*residual_tmp65 - residual_tmp67*residual_tmp70) + ((s_t(1) / s_t(3)))*residual_tmp62*residual_tmp83);
      const s_t residual_tmp305 = mu*(-residual_tmp258 - residual_tmp86);
      const s_t residual_tmp306 = -residual_tmp2*residual_tmp65;
      const s_t residual_tmp307 = s_t(2)*residual_tmp9;
      const s_t residual_tmp308 = mu*(-residual_tmp294 - residual_tmp307 + s_t(4)*u0_grad_2*u2_grad_1) + residual_tmp296;
      const s_t residual_tmp309 = s_t(2)*residual_tmp183;
      const s_t residual_tmp310 = mu*(-residual_tmp184 - residual_tmp309 + s_t(4)*u0_grad_2*u1_grad_0) + residual_tmp190;
      const s_t residual_tmp311 = ((s_t(1) / s_t(3)))*residual_tmp62;
      const s_t residual_tmp312 = -s_t(2)*residual_tmp182*residual_tmp8;
      const s_t residual_tmp313 = mu*(s_t(4)*residual_tmp19 + residual_tmp243 + residual_tmp312) - residual_tmp222*residual_tmp228;
      const s_t residual_tmp314 = s_t(2)*residual_tmp17;
      const s_t residual_tmp315 = residual_tmp284 - residual_tmp314;
      const s_t residual_tmp316 = -residual_tmp286;
      const s_t residual_tmp317 = residual_tmp269*residual_tmp8;
      const s_t residual_tmp318 = residual_tmp2*residual_tmp271;
      const s_t residual_tmp319 = mu*(-residual_tmp317 - residual_tmp318);
      const s_t residual_tmp320 = residual_tmp14*(-residual_tmp293 - residual_tmp307);
      const s_t residual_tmp321 = -residual_tmp43 + residual_tmp44;
      const s_t residual_tmp322 = residual_tmp321 + residual_tmp42;
      const s_t residual_tmp323 = residual_tmp104 + residual_tmp51*(-residual_tmp103 + s_t(2)*residual_tmp29*u0_grad_2 - residual_tmp97 - s_t(2)*residual_tmp99);
      const s_t residual_tmp324 = residual_tmp25*residual_tmp64 - residual_tmp27*residual_tmp63 + residual_tmp29*residual_tmp57 + residual_tmp33*residual_tmp69 - residual_tmp34*residual_tmp68 + residual_tmp36*residual_tmp67;
      const s_t residual_tmp325 = residual_tmp51*(s_t(2)*residual_tmp25*residual_tmp69 + s_t(2)*residual_tmp27*residual_tmp67 - residual_tmp73 - residual_tmp74 - s_t(2)*residual_tmp75 - residual_tmp78 - residual_tmp81) + residual_tmp82;
      const s_t residual_tmp326 = residual_tmp54*(-eta_s*(residual_tmp20*residual_tmp70 - residual_tmp324*residual_tmp33) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp325);
      const s_t residual_tmp327 = s_t(2)*pow_2(u0_grad_2);
      const s_t residual_tmp328 = residual_tmp107 + residual_tmp327;
      const s_t residual_tmp329 = s_t(2)*pow_2(u0_grad_1);
      const s_t residual_tmp330 = residual_tmp109 + residual_tmp329;
      const s_t residual_tmp331 = -residual_tmp98 + residual_tmp99;
      const s_t residual_tmp332 = residual_tmp331 + residual_tmp97;
      const s_t residual_tmp333 = residual_tmp50 + residual_tmp51*(residual_tmp115 + residual_tmp321 + s_t(2)*residual_tmp42);
      const s_t residual_tmp334 = -residual_tmp35 + residual_tmp37 + residual_tmp95;
      const s_t residual_tmp335 = residual_tmp119 + residual_tmp51*(residual_tmp117 + s_t(2)*residual_tmp28 - s_t(2)*residual_tmp30);
      const s_t residual_tmp336 = residual_tmp0*residual_tmp182;
      const s_t residual_tmp337 = mu*(-residual_tmp154 - residual_tmp336);
      const s_t residual_tmp338 = -residual_tmp192 + residual_tmp193;
      const s_t residual_tmp339 = residual_tmp196 + residual_tmp51*(s_t(2)*residual_tmp169 + residual_tmp172 + residual_tmp338);
      const s_t residual_tmp340 = -residual_tmp248 + residual_tmp249;
      const s_t residual_tmp341 = -residual_tmp174 - residual_tmp340;
      const s_t residual_tmp342 = residual_tmp324*u0_grad_1;
      const s_t residual_tmp343 = residual_tmp180 + residual_tmp342;
      const s_t residual_tmp344 = s_t(4)*residual_tmp2;
      const s_t residual_tmp345 = residual_tmp182*residual_tmp3;
      const s_t residual_tmp346 = residual_tmp123 + residual_tmp345;
      const s_t residual_tmp347 = -residual_tmp231 + residual_tmp232;
      const s_t residual_tmp348 = residual_tmp235 + residual_tmp51*(-s_t(2)*residual_tmp145 - residual_tmp148 - residual_tmp347);
      const s_t residual_tmp349 = -residual_tmp211 + residual_tmp212;
      const s_t residual_tmp350 = residual_tmp140 + residual_tmp349;
      const s_t residual_tmp351 = residual_tmp324*u0_grad_2;
      const s_t residual_tmp352 = residual_tmp165 + residual_tmp51*(-residual_tmp161 - s_t(2)*residual_tmp163 + s_t(2)*residual_tmp29*u2_grad_0);
      const s_t residual_tmp353 = residual_tmp199 - residual_tmp200 - residual_tmp255;
      const s_t residual_tmp354 = residual_tmp2*residual_tmp324;
      const s_t residual_tmp355 = ((s_t(1) / s_t(3)))*residual_tmp325;
      const s_t residual_tmp356 = residual_tmp355*u2_grad_1;
      const s_t residual_tmp357 = residual_tmp215 + residual_tmp51*(-residual_tmp140 + s_t(2)*residual_tmp182*residual_tmp27 - s_t(2)*residual_tmp212 - residual_tmp214);
      const s_t residual_tmp358 = -residual_tmp145 - residual_tmp347;
      const s_t residual_tmp359 = residual_tmp70*u1_grad_2;
      const s_t residual_tmp360 = s_t(6)*u0_grad_2;
      const s_t residual_tmp361 = residual_tmp189*u0_grad_2;
      const s_t residual_tmp362 = mu*(-residual_tmp302 - residual_tmp360 + s_t(4)*u1_grad_0*u2_grad_1) + residual_tmp361;
      const s_t residual_tmp363 = residual_tmp136 + residual_tmp51*(-residual_tmp132 - s_t(2)*residual_tmp134 + s_t(2)*residual_tmp69*residual_tmp8);
      const s_t residual_tmp364 = ((s_t(1) / s_t(3)))*residual_tmp25;
      const s_t residual_tmp365 = residual_tmp219 - residual_tmp238 + residual_tmp239;
      const s_t residual_tmp366 = residual_tmp324*u1_grad_2;
      const s_t residual_tmp367 = s_t(6)*u0_grad_1;
      const s_t residual_tmp368 = residual_tmp260 + residual_tmp367;
      const s_t residual_tmp369 = residual_tmp189*u0_grad_1;
      const s_t residual_tmp370 = -residual_tmp369;
      const s_t residual_tmp371 = residual_tmp252 + residual_tmp51*(residual_tmp174 - s_t(2)*residual_tmp248 + s_t(2)*residual_tmp249 + residual_tmp251);
      const s_t residual_tmp372 = residual_tmp169 + residual_tmp338;
      const s_t residual_tmp373 = residual_tmp2*residual_tmp70;
      const s_t residual_tmp374 = residual_tmp355*u0_grad_1;
      const s_t residual_tmp375 = residual_tmp14*(-residual_tmp242 - residual_tmp312);
      const s_t residual_tmp376 = ((s_t(1) / s_t(3)))*residual_tmp68;
      const s_t residual_tmp377 = -(s_t(1) / s_t(3))*residual_tmp325;
      const s_t residual_tmp378 = residual_tmp54*(eta_s*(residual_tmp324*residual_tmp57 + residual_tmp60*residual_tmp70) + residual_tmp377*residual_tmp68);
      const s_t residual_tmp379 = residual_tmp124*u2_grad_0;
      const s_t residual_tmp380 = mu*(-residual_tmp317 - residual_tmp379);
      const s_t residual_tmp381 = s_t(2)*pow_2(residual_tmp182);
      const s_t residual_tmp382 = residual_tmp265 + residual_tmp381;
      const s_t residual_tmp383 = residual_tmp3*u0_grad_2;
      const s_t residual_tmp384 = residual_tmp272 + residual_tmp383;
      const s_t residual_tmp385 = residual_tmp182*residual_tmp324;
      const s_t residual_tmp386 = residual_tmp324*u1_grad_0;
      const s_t residual_tmp387 = -residual_tmp301 + residual_tmp360;
      const s_t residual_tmp388 = -residual_tmp361;
      const s_t residual_tmp389 = s_t(6)*u0_grad_0 + s_t(6);
      const s_t residual_tmp390 = residual_tmp12 + residual_tmp389;
      const s_t residual_tmp391 = residual_tmp228*residual_tmp278;
      const s_t residual_tmp392 = residual_tmp70*u1_grad_0;
      const s_t residual_tmp393 = residual_tmp14*(residual_tmp206 - residual_tmp288);
      const s_t residual_tmp394 = residual_tmp54*(-eta_s*(residual_tmp324*residual_tmp63 - residual_tmp62*residual_tmp70) + ((s_t(1) / s_t(3)))*residual_tmp325*residual_tmp67);
      const s_t residual_tmp395 = mu*(-residual_tmp318 - residual_tmp379);
      const s_t residual_tmp396 = -residual_tmp182*residual_tmp324;
      const s_t residual_tmp397 = mu*(-residual_tmp261 - residual_tmp367 + s_t(4)*u1_grad_2*u2_grad_0) + residual_tmp369;
      const s_t residual_tmp398 = ((s_t(1) / s_t(3)))*residual_tmp67;
      const s_t residual_tmp399 = mu*(s_t(4)*residual_tmp11 + residual_tmp13 + residual_tmp389) - residual_tmp228*residual_tmp278;
      const s_t residual_tmp400 = residual_tmp14*(-residual_tmp285 + residual_tmp314);
      const s_t residual_tmp401 = residual_tmp50 + residual_tmp51*(s_t(2)*residual_tmp36*u0_grad_1 - residual_tmp42 - s_t(2)*residual_tmp44 - residual_tmp48);
      const s_t residual_tmp402 = residual_tmp51*(s_t(2)*residual_tmp33*residual_tmp64 + s_t(2)*residual_tmp34*residual_tmp57 - residual_tmp71 - residual_tmp72 - residual_tmp76 - s_t(2)*residual_tmp77 - residual_tmp81) + residual_tmp82;
      const s_t residual_tmp403 = residual_tmp54*(-eta_s*(residual_tmp20*residual_tmp65 - residual_tmp25*residual_tmp324) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp402);
      const s_t residual_tmp404 = residual_tmp106 + residual_tmp327 + s_t(2);
      const s_t residual_tmp405 = residual_tmp110 + residual_tmp329;
      const s_t residual_tmp406 = residual_tmp104 + residual_tmp51*(residual_tmp113 + residual_tmp331 + s_t(2)*residual_tmp97);
      const s_t residual_tmp407 = residual_tmp119 + residual_tmp51*(residual_tmp118 + residual_tmp26 + s_t(2)*residual_tmp91 - s_t(2)*residual_tmp92);
      const s_t residual_tmp408 = mu*(-residual_tmp125 - residual_tmp345);
      const s_t residual_tmp409 = residual_tmp215 + residual_tmp51*(s_t(2)*residual_tmp140 + residual_tmp143 + residual_tmp349);
      const s_t residual_tmp410 = residual_tmp151 + residual_tmp351;
      const s_t residual_tmp411 = s_t(4)*residual_tmp8;
      const s_t residual_tmp412 = residual_tmp153 + residual_tmp336;
      const s_t residual_tmp413 = residual_tmp252 + residual_tmp51*(-s_t(2)*residual_tmp174 - residual_tmp177 - residual_tmp340);
      const s_t residual_tmp414 = residual_tmp136 + residual_tmp51*(-residual_tmp129 - s_t(2)*residual_tmp131 - residual_tmp135 + s_t(2)*residual_tmp36*u1_grad_0);
      const s_t residual_tmp415 = residual_tmp324*residual_tmp8;
      const s_t residual_tmp416 = ((s_t(1) / s_t(3)))*residual_tmp402;
      const s_t residual_tmp417 = residual_tmp416*u1_grad_2;
      const s_t residual_tmp418 = residual_tmp196 + residual_tmp51*(-residual_tmp169 + s_t(2)*residual_tmp182*residual_tmp34 - s_t(2)*residual_tmp193 - residual_tmp195);
      const s_t residual_tmp419 = residual_tmp65*u2_grad_1;
      const s_t residual_tmp420 = residual_tmp165 + residual_tmp51*(-residual_tmp158 - s_t(2)*residual_tmp160 - residual_tmp164 + s_t(2)*residual_tmp2*residual_tmp64);
      const s_t residual_tmp421 = ((s_t(1) / s_t(3)))*residual_tmp33;
      const s_t residual_tmp422 = residual_tmp324*u2_grad_1;
      const s_t residual_tmp423 = residual_tmp235 + residual_tmp51*(residual_tmp145 - s_t(2)*residual_tmp231 + s_t(2)*residual_tmp232 + residual_tmp234);
      const s_t residual_tmp424 = residual_tmp65*residual_tmp8;
      const s_t residual_tmp425 = residual_tmp416*u0_grad_2;
      const s_t residual_tmp426 = residual_tmp14*(residual_tmp185 - residual_tmp309);
      const s_t residual_tmp427 = residual_tmp54*(-eta_s*(residual_tmp324*residual_tmp68 - residual_tmp60*residual_tmp65) + ((s_t(1) / s_t(3)))*residual_tmp402*residual_tmp57);
      const s_t residual_tmp428 = residual_tmp264 + residual_tmp381;
      const s_t residual_tmp429 = residual_tmp270 + residual_tmp383;
      const s_t residual_tmp430 = residual_tmp324*u2_grad_0;
      const s_t residual_tmp431 = ((s_t(1) / s_t(3)))*residual_tmp57;
      const s_t residual_tmp432 = residual_tmp65*u2_grad_0;
      const s_t residual_tmp433 = residual_tmp14*(-residual_tmp223 - residual_tmp291);
      const s_t residual_tmp434 = ((s_t(1) / s_t(3)))*residual_tmp63;
      const s_t residual_tmp435 = -(s_t(1) / s_t(3))*residual_tmp402;
      const s_t residual_tmp436 = residual_tmp54*(eta_s*(residual_tmp324*residual_tmp67 + residual_tmp62*residual_tmp65) + residual_tmp435*residual_tmp63);
      const s_t grad_coeff0_0 = u0_direction_grad_0*(mu*(residual_tmp108 + residual_tmp111) + residual_tmp112*residual_tmp15 + residual_tmp121*residual_tmp85 + residual_tmp24*(eta_s*(residual_tmp114*residual_tmp33 + residual_tmp116*residual_tmp25) - residual_tmp120*residual_tmp53)) + u0_direction_grad_1*(-mu*residual_tmp126 + residual_tmp128*residual_tmp15 + residual_tmp24*(eta_s*(residual_tmp144*residual_tmp33 + residual_tmp149*residual_tmp25 + residual_tmp151 + residual_tmp152) - residual_tmp139*residual_tmp53) + residual_tmp60*residual_tmp85) + u0_direction_grad_2*(-mu*residual_tmp155 + residual_tmp15*residual_tmp157 + residual_tmp24*(eta_s*(residual_tmp173*residual_tmp25 + residual_tmp178*residual_tmp33 + residual_tmp180 + residual_tmp181) - residual_tmp168*residual_tmp53) + residual_tmp62*residual_tmp85) + u1_direction_grad_0*(residual_tmp10*residual_tmp15 + residual_tmp24*(eta_s*(residual_tmp25*residual_tmp32 + residual_tmp33*residual_tmp41) - residual_tmp52*residual_tmp53) + residual_tmp25*residual_tmp85 + residual_tmp5) + u1_direction_grad_1*(mu*(s_t(2)*residual_tmp182*residual_tmp222 - residual_tmp225) + residual_tmp15*residual_tmp226 + residual_tmp229 + residual_tmp230*residual_tmp85 + residual_tmp24*(eta_s*(residual_tmp237*residual_tmp25 + residual_tmp240*residual_tmp33 + residual_tmp241) + residual_tmp204*residual_tmp8 - residual_tmp236*residual_tmp53)) + u1_direction_grad_2*(mu*(s_t(4)*residual_tmp183 + residual_tmp186) + residual_tmp15*residual_tmp188 + residual_tmp191 + residual_tmp24*(eta_s*(residual_tmp198*residual_tmp25 + residual_tmp202*residual_tmp33 - residual_tmp203) - residual_tmp197*residual_tmp53 - residual_tmp204*u2_grad_1) + residual_tmp67*residual_tmp85) + u2_direction_grad_0*(residual_tmp15*residual_tmp90 + residual_tmp24*(eta_s*(residual_tmp25*residual_tmp96 + residual_tmp33*residual_tmp94) - residual_tmp105*residual_tmp53) + residual_tmp33*residual_tmp85 + residual_tmp88) + u2_direction_grad_1*(mu*(s_t(4)*residual_tmp187 + residual_tmp207) + residual_tmp15*residual_tmp208 + residual_tmp210 + residual_tmp24*(eta_s*(residual_tmp217*residual_tmp33 + residual_tmp220*residual_tmp25 - residual_tmp221) - residual_tmp204*u1_grad_2 - residual_tmp216*residual_tmp53) + residual_tmp57*residual_tmp85) + u2_direction_grad_2*(mu*(s_t(2)*residual_tmp182*residual_tmp227 - residual_tmp244) + residual_tmp15*residual_tmp245 + residual_tmp24*(eta_s*(residual_tmp25*residual_tmp256 + residual_tmp254*residual_tmp33 + residual_tmp257) + residual_tmp2*residual_tmp204 - residual_tmp253*residual_tmp53) + residual_tmp246 + residual_tmp247*residual_tmp85);
      const s_t grad_coeff0_1 = u0_direction_grad_0*(mu*(-residual_tmp126 + s_t(2)*residual_tmp278*u0_grad_1 - residual_tmp279*u0_grad_1) + residual_tmp112*residual_tmp262 + residual_tmp121*residual_tmp263 + residual_tmp24*(-eta_s*(-residual_tmp114*residual_tmp57 + residual_tmp116*residual_tmp68 - residual_tmp150 - residual_tmp280) + ((s_t(1) / s_t(3)))*residual_tmp120*residual_tmp60)) + u0_direction_grad_1*(mu*(residual_tmp108 + residual_tmp266) + residual_tmp128*residual_tmp262 + residual_tmp24*(-eta_s*(-residual_tmp144*residual_tmp57 + residual_tmp149*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp139*residual_tmp60) + residual_tmp263*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp262 + residual_tmp24*(-eta_s*(residual_tmp173*residual_tmp68 - residual_tmp178*residual_tmp57 + residual_tmp277) + ((s_t(1) / s_t(3)))*residual_tmp168*residual_tmp60) + residual_tmp263*residual_tmp62 + residual_tmp273) + u1_direction_grad_0*(residual_tmp10*residual_tmp262 + residual_tmp24*(-eta_s*(residual_tmp241 + residual_tmp32*residual_tmp68 - residual_tmp41*residual_tmp57) + residual_tmp282*residual_tmp8 + residual_tmp290*residual_tmp52) + residual_tmp25*residual_tmp263 + residual_tmp292) + u1_direction_grad_1*(residual_tmp226*residual_tmp262 + residual_tmp230*residual_tmp263 + residual_tmp24*(-eta_s*(residual_tmp237*residual_tmp68 - residual_tmp240*residual_tmp57) + ((s_t(1) / s_t(3)))*residual_tmp236*residual_tmp60) + residual_tmp268) + u1_direction_grad_2*(residual_tmp188*residual_tmp262 + residual_tmp24*(-eta_s*(residual_tmp198*residual_tmp68 - residual_tmp202*residual_tmp57 - residual_tmp281) + ((s_t(1) / s_t(3)))*residual_tmp197*residual_tmp60 - residual_tmp283) + residual_tmp263*residual_tmp67 + residual_tmp287) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(-residual_tmp221 - residual_tmp57*residual_tmp94 + residual_tmp68*residual_tmp96) + ((s_t(1) / s_t(3)))*residual_tmp105*residual_tmp60 - residual_tmp282*u1_grad_2) + residual_tmp262*residual_tmp90 + residual_tmp263*residual_tmp33 + residual_tmp289) + u2_direction_grad_1*(residual_tmp208*residual_tmp262 + residual_tmp24*(-eta_s*(-residual_tmp217*residual_tmp57 + residual_tmp220*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp216*residual_tmp60) + residual_tmp259 + residual_tmp263*residual_tmp57) + u2_direction_grad_2*(mu*(residual_tmp227*residual_tmp3 + residual_tmp295) + residual_tmp24*(-eta_s*(-residual_tmp254*residual_tmp57 + residual_tmp256*residual_tmp68 + residual_tmp298) + residual_tmp253*residual_tmp290 + residual_tmp299) + residual_tmp245*residual_tmp262 + residual_tmp247*residual_tmp263 + residual_tmp297);
      const s_t grad_coeff0_2 = u0_direction_grad_0*(mu*(-residual_tmp155 + s_t(2)*residual_tmp278*u0_grad_2 - residual_tmp279*u0_grad_2) + residual_tmp112*residual_tmp303 + residual_tmp121*residual_tmp304 + residual_tmp24*(-eta_s*(residual_tmp114*residual_tmp63 - residual_tmp116*residual_tmp67 - residual_tmp179 - residual_tmp306) + ((s_t(1) / s_t(3)))*residual_tmp120*residual_tmp62)) + u0_direction_grad_1*(residual_tmp128*residual_tmp303 + residual_tmp24*(-eta_s*(residual_tmp144*residual_tmp63 - residual_tmp149*residual_tmp67 - residual_tmp277) + ((s_t(1) / s_t(3)))*residual_tmp139*residual_tmp62) + residual_tmp273 + residual_tmp304*residual_tmp60) + u0_direction_grad_2*(mu*(residual_tmp111 + residual_tmp266 + s_t(2)) + residual_tmp157*residual_tmp303 + residual_tmp24*(-eta_s*(-residual_tmp173*residual_tmp67 + residual_tmp178*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp168*residual_tmp62) + residual_tmp304*residual_tmp62) + u1_direction_grad_0*(residual_tmp10*residual_tmp303 + residual_tmp24*(-eta_s*(-residual_tmp203 - residual_tmp32*residual_tmp67 + residual_tmp41*residual_tmp63) - residual_tmp282*u2_grad_1 + ((s_t(1) / s_t(3)))*residual_tmp52*residual_tmp62) + residual_tmp25*residual_tmp304 + residual_tmp310) + u1_direction_grad_1*(mu*(residual_tmp0*residual_tmp222 + residual_tmp315) + residual_tmp226*residual_tmp303 + residual_tmp230*residual_tmp304 + residual_tmp24*(-eta_s*(-residual_tmp237*residual_tmp67 + residual_tmp240*residual_tmp63 + residual_tmp281) + residual_tmp236*residual_tmp311 + residual_tmp283) + residual_tmp316) + u1_direction_grad_2*(residual_tmp188*residual_tmp303 + residual_tmp24*(-eta_s*(-residual_tmp198*residual_tmp67 + residual_tmp202*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp197*residual_tmp62) + residual_tmp300 + residual_tmp304*residual_tmp67) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(residual_tmp257 + residual_tmp63*residual_tmp94 - residual_tmp67*residual_tmp96) + residual_tmp105*residual_tmp311 + residual_tmp2*residual_tmp282) + residual_tmp303*residual_tmp90 + residual_tmp304*residual_tmp33 + residual_tmp313) + u2_direction_grad_1*(residual_tmp208*residual_tmp303 + residual_tmp24*(-eta_s*(residual_tmp217*residual_tmp63 - residual_tmp220*residual_tmp67 - residual_tmp298) + ((s_t(1) / s_t(3)))*residual_tmp216*residual_tmp62 - residual_tmp299) + residual_tmp304*residual_tmp57 + residual_tmp308) + u2_direction_grad_2*(residual_tmp24*(-eta_s*(residual_tmp254*residual_tmp63 - residual_tmp256*residual_tmp67) + ((s_t(1) / s_t(3)))*residual_tmp253*residual_tmp62) + residual_tmp245*residual_tmp303 + residual_tmp247*residual_tmp304 + residual_tmp305);
      const s_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp112*residual_tmp320 + residual_tmp121*residual_tmp326 + residual_tmp24*(-eta_s*(residual_tmp116*residual_tmp20 - residual_tmp33*residual_tmp334) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp335) + residual_tmp5) + u0_direction_grad_1*(residual_tmp128*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp149*residual_tmp20 - residual_tmp33*residual_tmp365 + residual_tmp366) + residual_tmp355*residual_tmp8 + residual_tmp363*residual_tmp364) + residual_tmp292 + residual_tmp326*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp173*residual_tmp20 - residual_tmp33*residual_tmp353 - residual_tmp354) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp352 - residual_tmp356) + residual_tmp310 + residual_tmp326*residual_tmp62) + u1_direction_grad_0*(mu*(residual_tmp328 + residual_tmp330) + residual_tmp10*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp32 - residual_tmp33*residual_tmp332) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp333) + residual_tmp25*residual_tmp326) + u1_direction_grad_1*(mu*(s_t(2)*residual_tmp222*u1_grad_0 - residual_tmp344*u1_grad_0 - residual_tmp346) + residual_tmp226*residual_tmp320 + residual_tmp230*residual_tmp326 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp237 - residual_tmp280 - residual_tmp33*residual_tmp350 - residual_tmp351) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp348)) + u1_direction_grad_2*(residual_tmp188*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp198*residual_tmp20 - residual_tmp33*residual_tmp341 + residual_tmp343) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp339) + residual_tmp326*residual_tmp67 + residual_tmp337) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp96 - residual_tmp322*residual_tmp33) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp323) + residual_tmp319 + residual_tmp320*residual_tmp90 + residual_tmp326*residual_tmp33) + u2_direction_grad_1*(residual_tmp208*residual_tmp320 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp220 - residual_tmp33*residual_tmp358 - residual_tmp359) + ((s_t(1) / s_t(3)))*residual_tmp25*residual_tmp357 - residual_tmp355*u0_grad_2) + residual_tmp326*residual_tmp57 + residual_tmp362) + u2_direction_grad_2*(mu*(residual_tmp124*residual_tmp227 + residual_tmp368) + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp256 - residual_tmp33*residual_tmp372 + residual_tmp373) + residual_tmp364*residual_tmp371 + residual_tmp374) + residual_tmp245*residual_tmp320 + residual_tmp247*residual_tmp326 + residual_tmp370);
      const s_t grad_coeff1_1 = u0_direction_grad_0*(mu*(s_t(2)*residual_tmp2*residual_tmp278 - residual_tmp225) + residual_tmp112*residual_tmp375 + residual_tmp121*residual_tmp378 + residual_tmp229 + residual_tmp24*(eta_s*(residual_tmp116*residual_tmp60 + residual_tmp334*residual_tmp57 + residual_tmp366) - residual_tmp335*residual_tmp376 + residual_tmp377*residual_tmp8)) + u0_direction_grad_1*(residual_tmp128*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp149*residual_tmp60 + residual_tmp365*residual_tmp57) - residual_tmp363*residual_tmp376) + residual_tmp268 + residual_tmp378*residual_tmp60) + u0_direction_grad_2*(mu*(residual_tmp315 + s_t(4)*residual_tmp89) + residual_tmp157*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp173*residual_tmp60 + residual_tmp353*residual_tmp57 - residual_tmp386) - residual_tmp352*residual_tmp376 - residual_tmp377*u2_grad_0) + residual_tmp316 + residual_tmp378*residual_tmp62) + u1_direction_grad_0*(-mu*residual_tmp346 + residual_tmp10*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp152 + residual_tmp32*residual_tmp60 + residual_tmp332*residual_tmp57 - residual_tmp351) - residual_tmp333*residual_tmp376) + residual_tmp25*residual_tmp378) + u1_direction_grad_1*(mu*(residual_tmp328 + residual_tmp382) + residual_tmp226*residual_tmp375 + residual_tmp230*residual_tmp378 + residual_tmp24*(eta_s*(residual_tmp237*residual_tmp60 + residual_tmp350*residual_tmp57) - residual_tmp348*residual_tmp376)) + u1_direction_grad_2*(-mu*residual_tmp384 + residual_tmp188*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp198*residual_tmp60 + residual_tmp276 + residual_tmp341*residual_tmp57 + residual_tmp385) - residual_tmp339*residual_tmp376) + residual_tmp378*residual_tmp67) + u2_direction_grad_0*(mu*(s_t(4)*residual_tmp156 + residual_tmp387) + residual_tmp24*(eta_s*(residual_tmp322*residual_tmp57 - residual_tmp359 + residual_tmp60*residual_tmp96) - residual_tmp323*residual_tmp376 - residual_tmp377*u0_grad_2) + residual_tmp33*residual_tmp378 + residual_tmp375*residual_tmp90 + residual_tmp388) + u2_direction_grad_1*(residual_tmp208*residual_tmp375 + residual_tmp24*(eta_s*(residual_tmp220*residual_tmp60 + residual_tmp358*residual_tmp57) - residual_tmp357*residual_tmp376) + residual_tmp378*residual_tmp57 + residual_tmp380) + u2_direction_grad_2*(mu*(s_t(2)*residual_tmp2*residual_tmp227 - residual_tmp390) + residual_tmp24*(eta_s*(residual_tmp256*residual_tmp60 + residual_tmp372*residual_tmp57 + residual_tmp392) + residual_tmp182*residual_tmp377 - residual_tmp371*residual_tmp376) + residual_tmp245*residual_tmp375 + residual_tmp247*residual_tmp378 + residual_tmp391);
      const s_t grad_coeff1_2 = u0_direction_grad_0*(mu*(residual_tmp186 + residual_tmp269*residual_tmp278) + residual_tmp112*residual_tmp393 + residual_tmp121*residual_tmp394 + residual_tmp191 + residual_tmp24*(-eta_s*(-residual_tmp116*residual_tmp62 + residual_tmp334*residual_tmp63 + residual_tmp354) + residual_tmp335*residual_tmp398 + residual_tmp356)) + u0_direction_grad_1*(residual_tmp128*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp149*residual_tmp62 + residual_tmp365*residual_tmp63 - residual_tmp386) - residual_tmp355*u2_grad_0 + ((s_t(1) / s_t(3)))*residual_tmp363*residual_tmp67) + residual_tmp287 + residual_tmp394*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp173*residual_tmp62 + residual_tmp353*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp352*residual_tmp67) + residual_tmp300 + residual_tmp394*residual_tmp62) + u1_direction_grad_0*(residual_tmp10*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp32*residual_tmp62 + residual_tmp332*residual_tmp63 - residual_tmp343) + ((s_t(1) / s_t(3)))*residual_tmp333*residual_tmp67) + residual_tmp25*residual_tmp394 + residual_tmp337) + u1_direction_grad_1*(mu*(s_t(2)*residual_tmp222*u1_grad_2 - residual_tmp344*u1_grad_2 - residual_tmp384) + residual_tmp226*residual_tmp393 + residual_tmp230*residual_tmp394 + residual_tmp24*(-eta_s*(-residual_tmp237*residual_tmp62 - residual_tmp275 + residual_tmp350*residual_tmp63 - residual_tmp396) + ((s_t(1) / s_t(3)))*residual_tmp348*residual_tmp67)) + u1_direction_grad_2*(mu*(residual_tmp330 + residual_tmp382 + s_t(2)) + residual_tmp188*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp198*residual_tmp62 + residual_tmp341*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp339*residual_tmp67) + residual_tmp394*residual_tmp67) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(residual_tmp322*residual_tmp63 - residual_tmp373 - residual_tmp62*residual_tmp96) + ((s_t(1) / s_t(3)))*residual_tmp323*residual_tmp67 - residual_tmp374) + residual_tmp33*residual_tmp394 + residual_tmp393*residual_tmp90 + residual_tmp397) + u2_direction_grad_1*(residual_tmp208*residual_tmp393 + residual_tmp24*(-eta_s*(-residual_tmp220*residual_tmp62 + residual_tmp358*residual_tmp63 + residual_tmp392) + residual_tmp182*residual_tmp355 + residual_tmp357*residual_tmp398) + residual_tmp394*residual_tmp57 + residual_tmp399) + u2_direction_grad_2*(residual_tmp24*(-eta_s*(-residual_tmp256*residual_tmp62 + residual_tmp372*residual_tmp63) + ((s_t(1) / s_t(3)))*residual_tmp371*residual_tmp67) + residual_tmp245*residual_tmp393 + residual_tmp247*residual_tmp394 + residual_tmp395);
      const s_t grad_coeff2_0 = u0_direction_grad_0*(residual_tmp112*residual_tmp400 + residual_tmp121*residual_tmp403 + residual_tmp24*(-eta_s*(residual_tmp114*residual_tmp20 - residual_tmp25*residual_tmp334) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp407) + residual_tmp88) + u0_direction_grad_1*(residual_tmp128*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp144*residual_tmp20 - residual_tmp25*residual_tmp365 - residual_tmp415) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp414 - residual_tmp417) + residual_tmp289 + residual_tmp403*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp178*residual_tmp20 - residual_tmp25*residual_tmp353 + residual_tmp422) + residual_tmp2*residual_tmp416 + residual_tmp420*residual_tmp421) + residual_tmp313 + residual_tmp403*residual_tmp62) + u1_direction_grad_0*(residual_tmp10*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp41 - residual_tmp25*residual_tmp332) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp401) + residual_tmp25*residual_tmp403 + residual_tmp319) + u1_direction_grad_1*(mu*(residual_tmp122*residual_tmp222 + residual_tmp387) + residual_tmp226*residual_tmp400 + residual_tmp230*residual_tmp403 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp240 - residual_tmp25*residual_tmp350 + residual_tmp424) + residual_tmp421*residual_tmp423 + residual_tmp425) + residual_tmp388) + u1_direction_grad_2*(residual_tmp188*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp202 - residual_tmp25*residual_tmp341 - residual_tmp419) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp418 - residual_tmp416*u0_grad_1) + residual_tmp397 + residual_tmp403*residual_tmp67) + u2_direction_grad_0*(mu*(residual_tmp404 + residual_tmp405) + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp94 - residual_tmp25*residual_tmp322) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp406) + residual_tmp33*residual_tmp403 + residual_tmp400*residual_tmp90) + u2_direction_grad_1*(residual_tmp208*residual_tmp400 + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp217 - residual_tmp25*residual_tmp358 + residual_tmp410) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp409) + residual_tmp403*residual_tmp57 + residual_tmp408) + u2_direction_grad_2*(mu*(s_t(2)*residual_tmp227*u2_grad_0 - residual_tmp411*u2_grad_0 - residual_tmp412) + residual_tmp24*(-eta_s*(residual_tmp20*residual_tmp254 - residual_tmp25*residual_tmp372 - residual_tmp306 - residual_tmp342) + ((s_t(1) / s_t(3)))*residual_tmp33*residual_tmp413) + residual_tmp245*residual_tmp400 + residual_tmp247*residual_tmp403);
      const s_t grad_coeff2_1 = u0_direction_grad_0*(mu*(residual_tmp207 + residual_tmp271*residual_tmp278) + residual_tmp112*residual_tmp426 + residual_tmp121*residual_tmp427 + residual_tmp210 + residual_tmp24*(-eta_s*(-residual_tmp114*residual_tmp60 + residual_tmp334*residual_tmp68 + residual_tmp415) + residual_tmp407*residual_tmp431 + residual_tmp417)) + u0_direction_grad_1*(residual_tmp128*residual_tmp426 + residual_tmp24*(-eta_s*(-residual_tmp144*residual_tmp60 + residual_tmp365*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp414*residual_tmp57) + residual_tmp259 + residual_tmp427*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp426 + residual_tmp24*(-eta_s*(-residual_tmp178*residual_tmp60 + residual_tmp353*residual_tmp68 - residual_tmp430) - residual_tmp416*u1_grad_0 + ((s_t(1) / s_t(3)))*residual_tmp420*residual_tmp57) + residual_tmp308 + residual_tmp427*residual_tmp62) + u1_direction_grad_0*(residual_tmp10*residual_tmp426 + residual_tmp24*(-eta_s*(residual_tmp332*residual_tmp68 - residual_tmp41*residual_tmp60 - residual_tmp424) + ((s_t(1) / s_t(3)))*residual_tmp401*residual_tmp57 - residual_tmp425) + residual_tmp25*residual_tmp427 + residual_tmp362) + u1_direction_grad_1*(residual_tmp226*residual_tmp426 + residual_tmp230*residual_tmp427 + residual_tmp24*(-eta_s*(-residual_tmp240*residual_tmp60 + residual_tmp350*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp423*residual_tmp57) + residual_tmp380) + u1_direction_grad_2*(residual_tmp188*residual_tmp426 + residual_tmp24*(-eta_s*(-residual_tmp202*residual_tmp60 + residual_tmp341*residual_tmp68 + residual_tmp432) + residual_tmp182*residual_tmp416 + residual_tmp418*residual_tmp431) + residual_tmp399 + residual_tmp427*residual_tmp67) + u2_direction_grad_0*(residual_tmp24*(-eta_s*(residual_tmp322*residual_tmp68 - residual_tmp410 - residual_tmp60*residual_tmp94) + ((s_t(1) / s_t(3)))*residual_tmp406*residual_tmp57) + residual_tmp33*residual_tmp427 + residual_tmp408 + residual_tmp426*residual_tmp90) + u2_direction_grad_1*(mu*(residual_tmp404 + residual_tmp428) + residual_tmp208*residual_tmp426 + residual_tmp24*(-eta_s*(-residual_tmp217*residual_tmp60 + residual_tmp358*residual_tmp68) + ((s_t(1) / s_t(3)))*residual_tmp409*residual_tmp57) + residual_tmp427*residual_tmp57) + u2_direction_grad_2*(mu*(s_t(2)*residual_tmp227*u2_grad_1 - residual_tmp411*u2_grad_1 - residual_tmp429) + residual_tmp24*(-eta_s*(-residual_tmp254*residual_tmp60 - residual_tmp274 + residual_tmp372*residual_tmp68 - residual_tmp396) + ((s_t(1) / s_t(3)))*residual_tmp413*residual_tmp57) + residual_tmp245*residual_tmp426 + residual_tmp247*residual_tmp427);
      const s_t grad_coeff2_2 = u0_direction_grad_0*(mu*(-residual_tmp244 + s_t(2)*residual_tmp278*residual_tmp8) + residual_tmp112*residual_tmp433 + residual_tmp121*residual_tmp436 + residual_tmp24*(eta_s*(residual_tmp114*residual_tmp62 + residual_tmp334*residual_tmp67 + residual_tmp422) + residual_tmp2*residual_tmp435 - residual_tmp407*residual_tmp434) + residual_tmp246) + u0_direction_grad_1*(mu*(residual_tmp295 + s_t(4)*residual_tmp9) + residual_tmp128*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp144*residual_tmp62 + residual_tmp365*residual_tmp67 - residual_tmp430) - residual_tmp414*residual_tmp434 - residual_tmp435*u1_grad_0) + residual_tmp297 + residual_tmp436*residual_tmp60) + u0_direction_grad_2*(residual_tmp157*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp178*residual_tmp62 + residual_tmp353*residual_tmp67) - residual_tmp420*residual_tmp434) + residual_tmp305 + residual_tmp436*residual_tmp62) + u1_direction_grad_0*(mu*(s_t(4)*residual_tmp127 + residual_tmp368) + residual_tmp10*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp332*residual_tmp67 + residual_tmp41*residual_tmp62 - residual_tmp419) - residual_tmp401*residual_tmp434 - residual_tmp435*u0_grad_1) + residual_tmp25*residual_tmp436 + residual_tmp370) + u1_direction_grad_1*(mu*(s_t(2)*residual_tmp222*residual_tmp8 - residual_tmp390) + residual_tmp226*residual_tmp433 + residual_tmp230*residual_tmp436 + residual_tmp24*(eta_s*(residual_tmp240*residual_tmp62 + residual_tmp350*residual_tmp67 + residual_tmp432) + residual_tmp182*residual_tmp435 - residual_tmp423*residual_tmp434) + residual_tmp391) + u1_direction_grad_2*(residual_tmp188*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp202*residual_tmp62 + residual_tmp341*residual_tmp67) - residual_tmp418*residual_tmp434) + residual_tmp395 + residual_tmp436*residual_tmp67) + u2_direction_grad_0*(-mu*residual_tmp412 + residual_tmp24*(eta_s*(residual_tmp181 + residual_tmp322*residual_tmp67 - residual_tmp342 + residual_tmp62*residual_tmp94) - residual_tmp406*residual_tmp434) + residual_tmp33*residual_tmp436 + residual_tmp433*residual_tmp90) + u2_direction_grad_1*(-mu*residual_tmp429 + residual_tmp208*residual_tmp433 + residual_tmp24*(eta_s*(residual_tmp217*residual_tmp62 - residual_tmp274 + residual_tmp358*residual_tmp67 + residual_tmp385) - residual_tmp409*residual_tmp434) + residual_tmp436*residual_tmp57) + u2_direction_grad_2*(mu*(residual_tmp405 + residual_tmp428 + s_t(2)) + residual_tmp24*(eta_s*(residual_tmp254*residual_tmp62 + residual_tmp372*residual_tmp67) - residual_tmp413*residual_tmp434) + residual_tmp245*residual_tmp433 + residual_tmp247*residual_tmp436);
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
      output[0][lane] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1 + grad_coeff0_2_value * test0_grad2);
      output[1][lane] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1 + grad_coeff1_2_value * test0_grad2);
      output[2][lane] += q_weight[q] * det * (grad_coeff2_0_value * test0_grad0 + grad_coeff2_1_value * test0_grad1 + grad_coeff2_2_value * test0_grad2);
      output[3][lane] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1 + grad_coeff0_2_value * test1_grad2);
      output[4][lane] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1 + grad_coeff1_2_value * test1_grad2);
      output[5][lane] += q_weight[q] * det * (grad_coeff2_0_value * test1_grad0 + grad_coeff2_1_value * test1_grad1 + grad_coeff2_2_value * test1_grad2);
      output[6][lane] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1 + grad_coeff0_2_value * test2_grad2);
      output[7][lane] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1 + grad_coeff1_2_value * test2_grad2);
      output[8][lane] += q_weight[q] * det * (grad_coeff2_0_value * test2_grad0 + grad_coeff2_1_value * test2_grad1 + grad_coeff2_2_value * test2_grad2);
      output[9][lane] += q_weight[q] * det * (grad_coeff0_0_value * test3_grad0 + grad_coeff0_1_value * test3_grad1 + grad_coeff0_2_value * test3_grad2);
      output[10][lane] += q_weight[q] * det * (grad_coeff1_0_value * test3_grad0 + grad_coeff1_1_value * test3_grad1 + grad_coeff1_2_value * test3_grad2);
      output[11][lane] += q_weight[q] * det * (grad_coeff2_0_value * test3_grad0 + grad_coeff2_1_value * test3_grad1 + grad_coeff2_2_value * test3_grad2);
    }
  }
}

} // namespace codegen
} // namespace sfem

#endif
