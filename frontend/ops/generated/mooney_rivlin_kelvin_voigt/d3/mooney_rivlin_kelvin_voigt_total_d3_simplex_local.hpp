#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_TOTAL_D3_SIMPLEX_LOCAL_HPP
#define MOONEY_RIVLIN_KELVIN_VOIGT_TOTAL_D3_SIMPLEX_LOCAL_HPP

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
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_total_d3_simplex_residual_block(
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
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_total_d3_simplex_residual_block_contiguous(
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
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_total_d3_simplex_tet4_residual_block(
    const int ne,
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
  #pragma omp simd
  for (int lane = 0; lane < ne; ++lane) {
    const ptrdiff_t goff = lane;
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
    output[0][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test0_grad0 + grad_coeff0_1 * test0_grad1 + grad_coeff0_2 * test0_grad2);
    output[1][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test0_grad0 + grad_coeff1_1 * test0_grad1 + grad_coeff1_2 * test0_grad2);
    output[2][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test0_grad0 + grad_coeff2_1 * test0_grad1 + grad_coeff2_2 * test0_grad2);
    output[3][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test1_grad0 + grad_coeff0_1 * test1_grad1 + grad_coeff0_2 * test1_grad2);
    output[4][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test1_grad0 + grad_coeff1_1 * test1_grad1 + grad_coeff1_2 * test1_grad2);
    output[5][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test1_grad0 + grad_coeff2_1 * test1_grad1 + grad_coeff2_2 * test1_grad2);
    output[6][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test2_grad0 + grad_coeff0_1 * test2_grad1 + grad_coeff0_2 * test2_grad2);
    output[7][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test2_grad0 + grad_coeff1_1 * test2_grad1 + grad_coeff1_2 * test2_grad2);
    output[8][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test2_grad0 + grad_coeff2_1 * test2_grad1 + grad_coeff2_2 * test2_grad2);
    output[9][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test3_grad0 + grad_coeff0_1 * test3_grad1 + grad_coeff0_2 * test3_grad2);
    output[10][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test3_grad0 + grad_coeff1_1 * test3_grad1 + grad_coeff1_2 * test3_grad2);
    output[11][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test3_grad0 + grad_coeff2_1 * test3_grad1 + grad_coeff2_2 * test3_grad2);
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_total_d3_simplex_tet4_residual_block_contiguous(
    const int ne,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t current[3 * NS][VS],
    const s_t previous[3 * NS][VS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t output[3 * NS][VS]
) {
  #pragma omp simd
  for (int lane = 0; lane < ne; ++lane) {
    const ptrdiff_t goff = lane;
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
    output[0][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test0_grad0 + grad_coeff0_1 * test0_grad1 + grad_coeff0_2 * test0_grad2);
    output[1][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test0_grad0 + grad_coeff1_1 * test0_grad1 + grad_coeff1_2 * test0_grad2);
    output[2][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test0_grad0 + grad_coeff2_1 * test0_grad1 + grad_coeff2_2 * test0_grad2);
    output[3][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test1_grad0 + grad_coeff0_1 * test1_grad1 + grad_coeff0_2 * test1_grad2);
    output[4][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test1_grad0 + grad_coeff1_1 * test1_grad1 + grad_coeff1_2 * test1_grad2);
    output[5][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test1_grad0 + grad_coeff2_1 * test1_grad1 + grad_coeff2_2 * test1_grad2);
    output[6][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test2_grad0 + grad_coeff0_1 * test2_grad1 + grad_coeff0_2 * test2_grad2);
    output[7][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test2_grad0 + grad_coeff1_1 * test2_grad1 + grad_coeff1_2 * test2_grad2);
    output[8][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test2_grad0 + grad_coeff2_1 * test2_grad1 + grad_coeff2_2 * test2_grad2);
    output[9][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test3_grad0 + grad_coeff0_1 * test3_grad1 + grad_coeff0_2 * test3_grad2);
    output[10][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test3_grad0 + grad_coeff1_1 * test3_grad1 + grad_coeff1_2 * test3_grad2);
    output[11][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test3_grad0 + grad_coeff2_1 * test3_grad1 + grad_coeff2_2 * test3_grad2);
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_total_d3_simplex_jacobian_action_block(
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
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_total_d3_simplex_jacobian_action_block_contiguous(
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
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_total_d3_simplex_tet4_jacobian_action_block(
    const int ne,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
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
  #pragma omp simd
  for (int lane = 0; lane < ne; ++lane) {
    const ptrdiff_t goff = lane;
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
    output[0][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test0_grad0 + grad_coeff0_1 * test0_grad1 + grad_coeff0_2 * test0_grad2);
    output[1][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test0_grad0 + grad_coeff1_1 * test0_grad1 + grad_coeff1_2 * test0_grad2);
    output[2][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test0_grad0 + grad_coeff2_1 * test0_grad1 + grad_coeff2_2 * test0_grad2);
    output[3][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test1_grad0 + grad_coeff0_1 * test1_grad1 + grad_coeff0_2 * test1_grad2);
    output[4][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test1_grad0 + grad_coeff1_1 * test1_grad1 + grad_coeff1_2 * test1_grad2);
    output[5][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test1_grad0 + grad_coeff2_1 * test1_grad1 + grad_coeff2_2 * test1_grad2);
    output[6][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test2_grad0 + grad_coeff0_1 * test2_grad1 + grad_coeff0_2 * test2_grad2);
    output[7][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test2_grad0 + grad_coeff1_1 * test2_grad1 + grad_coeff1_2 * test2_grad2);
    output[8][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test2_grad0 + grad_coeff2_1 * test2_grad1 + grad_coeff2_2 * test2_grad2);
    output[9][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test3_grad0 + grad_coeff0_1 * test3_grad1 + grad_coeff0_2 * test3_grad2);
    output[10][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test3_grad0 + grad_coeff1_1 * test3_grad1 + grad_coeff1_2 * test3_grad2);
    output[11][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test3_grad0 + grad_coeff2_1 * test3_grad1 + grad_coeff2_2 * test3_grad2);
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_total_d3_simplex_tet4_jacobian_action_block_contiguous(
    const int ne,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
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
  #pragma omp simd
  for (int lane = 0; lane < ne; ++lane) {
    const ptrdiff_t goff = lane;
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
    output[0][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test0_grad0 + grad_coeff0_1 * test0_grad1 + grad_coeff0_2 * test0_grad2);
    output[1][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test0_grad0 + grad_coeff1_1 * test0_grad1 + grad_coeff1_2 * test0_grad2);
    output[2][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test0_grad0 + grad_coeff2_1 * test0_grad1 + grad_coeff2_2 * test0_grad2);
    output[3][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test1_grad0 + grad_coeff0_1 * test1_grad1 + grad_coeff0_2 * test1_grad2);
    output[4][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test1_grad0 + grad_coeff1_1 * test1_grad1 + grad_coeff1_2 * test1_grad2);
    output[5][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test1_grad0 + grad_coeff2_1 * test1_grad1 + grad_coeff2_2 * test1_grad2);
    output[6][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test2_grad0 + grad_coeff0_1 * test2_grad1 + grad_coeff0_2 * test2_grad2);
    output[7][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test2_grad0 + grad_coeff1_1 * test2_grad1 + grad_coeff1_2 * test2_grad2);
    output[8][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test2_grad0 + grad_coeff2_1 * test2_grad1 + grad_coeff2_2 * test2_grad2);
    output[9][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff0_0 * test3_grad0 + grad_coeff0_1 * test3_grad1 + grad_coeff0_2 * test3_grad2);
    output[10][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff1_0 * test3_grad0 + grad_coeff1_1 * test3_grad1 + grad_coeff1_2 * test3_grad2);
    output[11][lane] += ((s_t(1) / s_t(6))) * (det) * (grad_coeff2_0 * test3_grad0 + grad_coeff2_1 * test3_grad1 + grad_coeff2_2 * test3_grad2);
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_total_d3_simplex_tet4_hessian_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t current[3 * NS][VS],
    const s_t previous[3 * NS][VS],
    const s_t eta_b,
    const s_t eta_s,
    const s_t lmbda,
    const s_t mu,
    const s_t u_dt_shift,
    s_t *const RSTR element_matrix
) {
  const int q = 0;
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
    const s_t element_matrix_tmp0 = s_t(2)*pow_2(u1_grad_2);
    const s_t element_matrix_tmp1 = u2_grad_2 + s_t(1);
    const s_t element_matrix_tmp2 = s_t(2)*pow_2(element_matrix_tmp1) + s_t(2);
    const s_t element_matrix_tmp3 = element_matrix_tmp0 + element_matrix_tmp2;
    const s_t element_matrix_tmp4 = s_t(2)*pow_2(u2_grad_1);
    const s_t element_matrix_tmp5 = u1_grad_1 + s_t(1);
    const s_t element_matrix_tmp6 = s_t(2)*pow_2(element_matrix_tmp5);
    const s_t element_matrix_tmp7 = element_matrix_tmp4 + element_matrix_tmp6;
    const s_t element_matrix_tmp8 = u1_grad_2*u2_grad_1;
    const s_t element_matrix_tmp9 = element_matrix_tmp1*element_matrix_tmp5 - element_matrix_tmp8;
    const s_t element_matrix_tmp10 = s_t(2)*element_matrix_tmp8;
    const s_t element_matrix_tmp11 = -s_t(2)*element_matrix_tmp1*element_matrix_tmp5;
    const s_t element_matrix_tmp12 = ((s_t(1) / s_t(2)))*lmbda;
    const s_t element_matrix_tmp13 = element_matrix_tmp12*(-element_matrix_tmp10 - element_matrix_tmp11);
    const s_t element_matrix_tmp14 = u0_grad_0*u1_grad_1;
    const s_t element_matrix_tmp15 = u0_grad_1*u1_grad_2;
    const s_t element_matrix_tmp16 = u0_grad_2*u2_grad_1;
    const s_t element_matrix_tmp17 = u0_grad_1*u1_grad_0;
    const s_t element_matrix_tmp18 = u0_grad_2*u2_grad_0;
    const s_t element_matrix_tmp19 = element_matrix_tmp5 - element_matrix_tmp8 + u1_grad_1*u2_grad_2 + u2_grad_2;
    const s_t element_matrix_tmp20 = -element_matrix_tmp18 + u0_grad_0*u2_grad_2 + u0_grad_0;
    const s_t element_matrix_tmp21 = element_matrix_tmp14 - element_matrix_tmp17;
    const s_t element_matrix_tmp22 = element_matrix_tmp14*u2_grad_2 + element_matrix_tmp15*u2_grad_0 + element_matrix_tmp16*u1_grad_0 - element_matrix_tmp17*u2_grad_2 - element_matrix_tmp18*u1_grad_1 + element_matrix_tmp19 + element_matrix_tmp20 + element_matrix_tmp21 - element_matrix_tmp8*u0_grad_0;
    const s_t element_matrix_tmp23 = pow_m1(element_matrix_tmp22);
    const s_t element_matrix_tmp24 = -element_matrix_tmp15 + u0_grad_2*u1_grad_1 + u0_grad_2;
    const s_t element_matrix_tmp25 = element_matrix_tmp24*u_dt_shift;
    const s_t element_matrix_tmp26 = u0_grad_1*u_dt_shift + u0_old_grad_1;
    const s_t element_matrix_tmp27 = element_matrix_tmp26*u1_grad_2;
    const s_t element_matrix_tmp28 = u0_grad_2*u_dt_shift + u0_old_grad_2;
    const s_t element_matrix_tmp29 = element_matrix_tmp28*element_matrix_tmp5;
    const s_t element_matrix_tmp30 = element_matrix_tmp27 - element_matrix_tmp29;
    const s_t element_matrix_tmp31 = element_matrix_tmp25 + element_matrix_tmp30;
    const s_t element_matrix_tmp32 = -element_matrix_tmp16;
    const s_t element_matrix_tmp33 = element_matrix_tmp32 + u0_grad_1*u2_grad_2 + u0_grad_1;
    const s_t element_matrix_tmp34 = element_matrix_tmp33*u_dt_shift;
    const s_t element_matrix_tmp35 = element_matrix_tmp28*u2_grad_1;
    const s_t element_matrix_tmp36 = element_matrix_tmp1*element_matrix_tmp26;
    const s_t element_matrix_tmp37 = element_matrix_tmp35 - element_matrix_tmp36;
    const s_t element_matrix_tmp38 = element_matrix_tmp34 + element_matrix_tmp37;
    const s_t element_matrix_tmp39 = element_matrix_tmp19*u_dt_shift;
    const s_t element_matrix_tmp40 = u2_grad_2*u_dt_shift + u2_old_grad_2;
    const s_t element_matrix_tmp41 = element_matrix_tmp40*element_matrix_tmp5;
    const s_t element_matrix_tmp42 = u2_grad_1*u_dt_shift + u2_old_grad_1;
    const s_t element_matrix_tmp43 = element_matrix_tmp42*u1_grad_2;
    const s_t element_matrix_tmp44 = element_matrix_tmp39 + element_matrix_tmp41 - element_matrix_tmp43;
    const s_t element_matrix_tmp45 = u1_grad_1*u_dt_shift + u1_old_grad_1;
    const s_t element_matrix_tmp46 = element_matrix_tmp1*element_matrix_tmp45;
    const s_t element_matrix_tmp47 = u1_grad_2*u_dt_shift + u1_old_grad_2;
    const s_t element_matrix_tmp48 = element_matrix_tmp47*u2_grad_1;
    const s_t element_matrix_tmp49 = element_matrix_tmp46 - element_matrix_tmp48;
    const s_t element_matrix_tmp50 = s_t(3)*eta_b;
    const s_t element_matrix_tmp51 = element_matrix_tmp50*(-element_matrix_tmp44 - element_matrix_tmp49);
    const s_t element_matrix_tmp52 = -element_matrix_tmp46 + element_matrix_tmp48;
    const s_t element_matrix_tmp53 = -element_matrix_tmp41 + element_matrix_tmp43;
    const s_t element_matrix_tmp54 = s_t(2)*eta_s;
    const s_t element_matrix_tmp55 = element_matrix_tmp51 + element_matrix_tmp54*(-s_t(2)*element_matrix_tmp39 - element_matrix_tmp52 - element_matrix_tmp53);
    const s_t element_matrix_tmp56 = ((s_t(1) / s_t(3)))*element_matrix_tmp19;
    const s_t element_matrix_tmp57 = -element_matrix_tmp19;
    const s_t element_matrix_tmp58 = pow_m2(element_matrix_tmp22);
    const s_t element_matrix_tmp59 = u0_grad_0*u_dt_shift + u0_old_grad_0;
    const s_t element_matrix_tmp60 = u0_grad_2*u1_grad_0;
    const s_t element_matrix_tmp61 = -element_matrix_tmp60 + u0_grad_0*u1_grad_2 + u1_grad_2;
    const s_t element_matrix_tmp62 = u1_grad_2*u2_grad_0;
    const s_t element_matrix_tmp63 = -element_matrix_tmp62;
    const s_t element_matrix_tmp64 = element_matrix_tmp63 + u1_grad_0*u2_grad_2 + u1_grad_0;
    const s_t element_matrix_tmp65 = u1_grad_0*u2_grad_1;
    const s_t element_matrix_tmp66 = -element_matrix_tmp65 + u1_grad_1*u2_grad_0 + u2_grad_0;
    const s_t element_matrix_tmp67 = element_matrix_tmp21 + element_matrix_tmp5 + u0_grad_0;
    const s_t element_matrix_tmp68 = u2_grad_0*u_dt_shift + u2_old_grad_0;
    const s_t element_matrix_tmp69 = -element_matrix_tmp19*element_matrix_tmp68 + element_matrix_tmp24*element_matrix_tmp59 + element_matrix_tmp26*element_matrix_tmp61 - element_matrix_tmp28*element_matrix_tmp67 + element_matrix_tmp40*element_matrix_tmp66 + element_matrix_tmp42*element_matrix_tmp64;
    const s_t element_matrix_tmp70 = u0_grad_1*u2_grad_0;
    const s_t element_matrix_tmp71 = -element_matrix_tmp70 + u0_grad_0*u2_grad_1 + u2_grad_1;
    const s_t element_matrix_tmp72 = element_matrix_tmp1 + element_matrix_tmp20;
    const s_t element_matrix_tmp73 = u1_grad_0*u_dt_shift + u1_old_grad_0;
    const s_t element_matrix_tmp74 = -element_matrix_tmp19*element_matrix_tmp73 - element_matrix_tmp26*element_matrix_tmp72 + element_matrix_tmp28*element_matrix_tmp71 + element_matrix_tmp33*element_matrix_tmp59 + element_matrix_tmp45*element_matrix_tmp64 + element_matrix_tmp47*element_matrix_tmp66;
    const s_t element_matrix_tmp75 = element_matrix_tmp33*element_matrix_tmp73;
    const s_t element_matrix_tmp76 = element_matrix_tmp47*element_matrix_tmp71;
    const s_t element_matrix_tmp77 = element_matrix_tmp24*element_matrix_tmp68;
    const s_t element_matrix_tmp78 = element_matrix_tmp42*element_matrix_tmp61;
    const s_t element_matrix_tmp79 = element_matrix_tmp45*element_matrix_tmp72;
    const s_t element_matrix_tmp80 = -element_matrix_tmp79;
    const s_t element_matrix_tmp81 = element_matrix_tmp40*element_matrix_tmp67;
    const s_t element_matrix_tmp82 = -element_matrix_tmp81;
    const s_t element_matrix_tmp83 = element_matrix_tmp75 + element_matrix_tmp76 + element_matrix_tmp77 + element_matrix_tmp78 + element_matrix_tmp80 + element_matrix_tmp82;
    const s_t element_matrix_tmp84 = element_matrix_tmp19*element_matrix_tmp59;
    const s_t element_matrix_tmp85 = element_matrix_tmp26*element_matrix_tmp64 + element_matrix_tmp28*element_matrix_tmp66 - element_matrix_tmp84;
    const s_t element_matrix_tmp86 = element_matrix_tmp50*(element_matrix_tmp83 + element_matrix_tmp85);
    const s_t element_matrix_tmp87 = element_matrix_tmp54*(s_t(2)*element_matrix_tmp26*element_matrix_tmp64 + s_t(2)*element_matrix_tmp28*element_matrix_tmp66 - element_matrix_tmp83 - s_t(2)*element_matrix_tmp84) + element_matrix_tmp86;
    const s_t element_matrix_tmp88 = -element_matrix_tmp87;
    const s_t element_matrix_tmp89 = element_matrix_tmp58*(element_matrix_tmp56*element_matrix_tmp88 + eta_s*(element_matrix_tmp24*element_matrix_tmp69 + element_matrix_tmp33*element_matrix_tmp74));
    const s_t element_matrix_tmp90 = element_matrix_tmp13*element_matrix_tmp9 + element_matrix_tmp23*(-element_matrix_tmp55*element_matrix_tmp56 + eta_s*(element_matrix_tmp24*element_matrix_tmp31 + element_matrix_tmp33*element_matrix_tmp38)) + element_matrix_tmp57*element_matrix_tmp89 + mu*(element_matrix_tmp3 + element_matrix_tmp7);
    const s_t element_matrix_tmp91 = s_t(2)*u2_grad_0;
    const s_t element_matrix_tmp92 = element_matrix_tmp91*u2_grad_1;
    const s_t element_matrix_tmp93 = s_t(2)*u1_grad_0;
    const s_t element_matrix_tmp94 = element_matrix_tmp5*element_matrix_tmp93;
    const s_t element_matrix_tmp95 = element_matrix_tmp92 + element_matrix_tmp94;
    const s_t element_matrix_tmp96 = element_matrix_tmp1*u1_grad_0;
    const s_t element_matrix_tmp97 = -element_matrix_tmp63 - element_matrix_tmp96;
    const s_t element_matrix_tmp98 = element_matrix_tmp64*u_dt_shift;
    const s_t element_matrix_tmp99 = element_matrix_tmp40*u1_grad_0;
    const s_t element_matrix_tmp100 = element_matrix_tmp68*u1_grad_2;
    const s_t element_matrix_tmp101 = -element_matrix_tmp100 + element_matrix_tmp98 + element_matrix_tmp99;
    const s_t element_matrix_tmp102 = element_matrix_tmp1*element_matrix_tmp73;
    const s_t element_matrix_tmp103 = element_matrix_tmp47*u2_grad_0;
    const s_t element_matrix_tmp104 = element_matrix_tmp102 - element_matrix_tmp103;
    const s_t element_matrix_tmp105 = element_matrix_tmp50*(element_matrix_tmp101 + element_matrix_tmp104);
    const s_t element_matrix_tmp106 = -element_matrix_tmp102 + element_matrix_tmp103;
    const s_t element_matrix_tmp107 = element_matrix_tmp100 - element_matrix_tmp99;
    const s_t element_matrix_tmp108 = element_matrix_tmp105 + element_matrix_tmp54*(element_matrix_tmp106 + element_matrix_tmp107 + s_t(2)*element_matrix_tmp98);
    const s_t element_matrix_tmp109 = element_matrix_tmp61*u_dt_shift;
    const s_t element_matrix_tmp110 = element_matrix_tmp28*u1_grad_0;
    const s_t element_matrix_tmp111 = element_matrix_tmp59*u1_grad_2;
    const s_t element_matrix_tmp112 = element_matrix_tmp110 - element_matrix_tmp111;
    const s_t element_matrix_tmp113 = element_matrix_tmp109 + element_matrix_tmp112;
    const s_t element_matrix_tmp114 = element_matrix_tmp72*u_dt_shift;
    const s_t element_matrix_tmp115 = element_matrix_tmp28*u2_grad_0;
    const s_t element_matrix_tmp116 = element_matrix_tmp1*element_matrix_tmp59;
    const s_t element_matrix_tmp117 = element_matrix_tmp115 - element_matrix_tmp116;
    const s_t element_matrix_tmp118 = -element_matrix_tmp114 - element_matrix_tmp117;
    const s_t element_matrix_tmp119 = element_matrix_tmp69*u1_grad_2;
    const s_t element_matrix_tmp120 = -element_matrix_tmp119;
    const s_t element_matrix_tmp121 = element_matrix_tmp1*element_matrix_tmp74;
    const s_t element_matrix_tmp122 = element_matrix_tmp13*element_matrix_tmp97 + element_matrix_tmp23*(-element_matrix_tmp108*element_matrix_tmp56 + eta_s*(element_matrix_tmp113*element_matrix_tmp24 + element_matrix_tmp118*element_matrix_tmp33 + element_matrix_tmp120 + element_matrix_tmp121)) + element_matrix_tmp64*element_matrix_tmp89 - element_matrix_tmp95*mu;
    const s_t element_matrix_tmp123 = element_matrix_tmp93*u1_grad_2;
    const s_t element_matrix_tmp124 = element_matrix_tmp1*element_matrix_tmp91;
    const s_t element_matrix_tmp125 = element_matrix_tmp123 + element_matrix_tmp124;
    const s_t element_matrix_tmp126 = element_matrix_tmp5*u2_grad_0;
    const s_t element_matrix_tmp127 = -element_matrix_tmp126 + element_matrix_tmp65;
    const s_t element_matrix_tmp128 = element_matrix_tmp66*u_dt_shift;
    const s_t element_matrix_tmp129 = element_matrix_tmp5*element_matrix_tmp68;
    const s_t element_matrix_tmp130 = element_matrix_tmp42*u1_grad_0;
    const s_t element_matrix_tmp131 = element_matrix_tmp128 + element_matrix_tmp129 - element_matrix_tmp130;
    const s_t element_matrix_tmp132 = element_matrix_tmp45*u2_grad_0;
    const s_t element_matrix_tmp133 = element_matrix_tmp73*u2_grad_1;
    const s_t element_matrix_tmp134 = element_matrix_tmp132 - element_matrix_tmp133;
    const s_t element_matrix_tmp135 = element_matrix_tmp50*(element_matrix_tmp131 + element_matrix_tmp134);
    const s_t element_matrix_tmp136 = -element_matrix_tmp132 + element_matrix_tmp133;
    const s_t element_matrix_tmp137 = -element_matrix_tmp129 + element_matrix_tmp130;
    const s_t element_matrix_tmp138 = element_matrix_tmp135 + element_matrix_tmp54*(s_t(2)*element_matrix_tmp128 + element_matrix_tmp136 + element_matrix_tmp137);
    const s_t element_matrix_tmp139 = element_matrix_tmp71*u_dt_shift;
    const s_t element_matrix_tmp140 = element_matrix_tmp26*u2_grad_0;
    const s_t element_matrix_tmp141 = element_matrix_tmp59*u2_grad_1;
    const s_t element_matrix_tmp142 = element_matrix_tmp140 - element_matrix_tmp141;
    const s_t element_matrix_tmp143 = element_matrix_tmp139 + element_matrix_tmp142;
    const s_t element_matrix_tmp144 = element_matrix_tmp67*u_dt_shift;
    const s_t element_matrix_tmp145 = element_matrix_tmp26*u1_grad_0;
    const s_t element_matrix_tmp146 = element_matrix_tmp5*element_matrix_tmp59;
    const s_t element_matrix_tmp147 = element_matrix_tmp145 - element_matrix_tmp146;
    const s_t element_matrix_tmp148 = -element_matrix_tmp144 - element_matrix_tmp147;
    const s_t element_matrix_tmp149 = element_matrix_tmp74*u2_grad_1;
    const s_t element_matrix_tmp150 = -element_matrix_tmp149;
    const s_t element_matrix_tmp151 = element_matrix_tmp5*element_matrix_tmp69;
    const s_t element_matrix_tmp152 = -element_matrix_tmp125*mu + element_matrix_tmp127*element_matrix_tmp13 + element_matrix_tmp23*(-element_matrix_tmp138*element_matrix_tmp56 + eta_s*(element_matrix_tmp143*element_matrix_tmp33 + element_matrix_tmp148*element_matrix_tmp24 + element_matrix_tmp150 + element_matrix_tmp151)) + element_matrix_tmp66*element_matrix_tmp89;
    const s_t element_matrix_tmp153 = basis0_grad0*element_matrix_tmp90 + basis0_grad1*element_matrix_tmp122 + basis0_grad2*element_matrix_tmp152;
    const s_t element_matrix_tmp154 = -s_t(2)*element_matrix_tmp62;
    const s_t element_matrix_tmp155 = s_t(2)*element_matrix_tmp96;
    const s_t element_matrix_tmp156 = element_matrix_tmp12*(-element_matrix_tmp154 - element_matrix_tmp155);
    const s_t element_matrix_tmp157 = s_t(2)*pow_2(u1_grad_0);
    const s_t element_matrix_tmp158 = s_t(2)*pow_2(u2_grad_0);
    const s_t element_matrix_tmp159 = element_matrix_tmp157 + element_matrix_tmp158;
    const s_t element_matrix_tmp160 = element_matrix_tmp58*(((s_t(1) / s_t(3)))*element_matrix_tmp64*element_matrix_tmp87 - eta_s*(-element_matrix_tmp61*element_matrix_tmp69 + element_matrix_tmp72*element_matrix_tmp74));
    const s_t element_matrix_tmp161 = element_matrix_tmp156*element_matrix_tmp97 + element_matrix_tmp160*element_matrix_tmp64 + element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp108*element_matrix_tmp64 - eta_s*(-element_matrix_tmp113*element_matrix_tmp61 + element_matrix_tmp118*element_matrix_tmp72)) + mu*(element_matrix_tmp159 + element_matrix_tmp3);
    const s_t element_matrix_tmp162 = s_t(2)*element_matrix_tmp5;
    const s_t element_matrix_tmp163 = element_matrix_tmp162*u1_grad_2;
    const s_t element_matrix_tmp164 = s_t(2)*u2_grad_1;
    const s_t element_matrix_tmp165 = element_matrix_tmp1*element_matrix_tmp164;
    const s_t element_matrix_tmp166 = mu*(-element_matrix_tmp163 - element_matrix_tmp165);
    const s_t element_matrix_tmp167 = element_matrix_tmp69*u1_grad_0;
    const s_t element_matrix_tmp168 = element_matrix_tmp74*u2_grad_0;
    const s_t element_matrix_tmp169 = -element_matrix_tmp168;
    const s_t element_matrix_tmp170 = element_matrix_tmp167 + element_matrix_tmp169;
    const s_t element_matrix_tmp171 = element_matrix_tmp127*element_matrix_tmp156 + element_matrix_tmp160*element_matrix_tmp66 + element_matrix_tmp166 + element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp138*element_matrix_tmp64 - eta_s*(element_matrix_tmp143*element_matrix_tmp72 - element_matrix_tmp148*element_matrix_tmp61 + element_matrix_tmp170));
    const s_t element_matrix_tmp172 = s_t(2)*u0_grad_0 + s_t(2);
    const s_t element_matrix_tmp173 = u0_grad_0 + s_t(1);
    const s_t element_matrix_tmp174 = s_t(4)*element_matrix_tmp173;
    const s_t element_matrix_tmp175 = -element_matrix_tmp1*element_matrix_tmp74;
    const s_t element_matrix_tmp176 = element_matrix_tmp156*element_matrix_tmp9 + element_matrix_tmp160*element_matrix_tmp57 + element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp55*element_matrix_tmp64 - eta_s*(-element_matrix_tmp119 - element_matrix_tmp175 - element_matrix_tmp31*element_matrix_tmp61 + element_matrix_tmp38*element_matrix_tmp72)) + mu*(s_t(2)*element_matrix_tmp172*u0_grad_1 - element_matrix_tmp174*u0_grad_1 - element_matrix_tmp95);
    const s_t element_matrix_tmp177 = basis0_grad0*element_matrix_tmp176 + basis0_grad1*element_matrix_tmp161 + basis0_grad2*element_matrix_tmp171;
    const s_t element_matrix_tmp178 = s_t(2)*element_matrix_tmp65;
    const s_t element_matrix_tmp179 = s_t(2)*element_matrix_tmp126;
    const s_t element_matrix_tmp180 = element_matrix_tmp12*(element_matrix_tmp178 - element_matrix_tmp179);
    const s_t element_matrix_tmp181 = element_matrix_tmp58*(((s_t(1) / s_t(3)))*element_matrix_tmp66*element_matrix_tmp87 - eta_s*(element_matrix_tmp67*element_matrix_tmp69 - element_matrix_tmp71*element_matrix_tmp74));
    const s_t element_matrix_tmp182 = element_matrix_tmp127*element_matrix_tmp180 + element_matrix_tmp181*element_matrix_tmp66 + element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp138*element_matrix_tmp66 - eta_s*(-element_matrix_tmp143*element_matrix_tmp71 + element_matrix_tmp148*element_matrix_tmp67)) + mu*(element_matrix_tmp159 + element_matrix_tmp7 + s_t(2));
    const s_t element_matrix_tmp183 = element_matrix_tmp166 + element_matrix_tmp180*element_matrix_tmp97 + element_matrix_tmp181*element_matrix_tmp64 + element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp108*element_matrix_tmp66 - eta_s*(element_matrix_tmp113*element_matrix_tmp67 - element_matrix_tmp118*element_matrix_tmp71 - element_matrix_tmp170));
    const s_t element_matrix_tmp184 = -element_matrix_tmp5*element_matrix_tmp69;
    const s_t element_matrix_tmp185 = element_matrix_tmp180*element_matrix_tmp9 + element_matrix_tmp181*element_matrix_tmp57 + element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp55*element_matrix_tmp66 - eta_s*(-element_matrix_tmp149 - element_matrix_tmp184 + element_matrix_tmp31*element_matrix_tmp67 - element_matrix_tmp38*element_matrix_tmp71)) + mu*(-element_matrix_tmp125 + s_t(2)*element_matrix_tmp172*u0_grad_2 - element_matrix_tmp174*u0_grad_2);
    const s_t element_matrix_tmp186 = basis0_grad0*element_matrix_tmp185 + basis0_grad1*element_matrix_tmp183 + basis0_grad2*element_matrix_tmp182;
    const s_t element_matrix_tmp187 = ((s_t(1) / s_t(6)))*det;
    const s_t element_matrix_tmp188 = s_t(2)*u0_grad_2;
    const s_t element_matrix_tmp189 = element_matrix_tmp188*u1_grad_2;
    const s_t element_matrix_tmp190 = element_matrix_tmp173*element_matrix_tmp93;
    const s_t element_matrix_tmp191 = mu*(-element_matrix_tmp189 - element_matrix_tmp190);
    const s_t element_matrix_tmp192 = s_t(2)*element_matrix_tmp18;
    const s_t element_matrix_tmp193 = -s_t(2)*element_matrix_tmp1*element_matrix_tmp173;
    const s_t element_matrix_tmp194 = element_matrix_tmp12*(-element_matrix_tmp192 - element_matrix_tmp193);
    const s_t element_matrix_tmp195 = element_matrix_tmp1*element_matrix_tmp68;
    const s_t element_matrix_tmp196 = element_matrix_tmp40*u2_grad_0;
    const s_t element_matrix_tmp197 = element_matrix_tmp47*u1_grad_0 - element_matrix_tmp73*u1_grad_2;
    const s_t element_matrix_tmp198 = element_matrix_tmp195 - element_matrix_tmp196 + element_matrix_tmp197;
    const s_t element_matrix_tmp199 = element_matrix_tmp105 + element_matrix_tmp54*(s_t(2)*element_matrix_tmp1*element_matrix_tmp73 - element_matrix_tmp101 - s_t(2)*element_matrix_tmp103);
    const s_t element_matrix_tmp200 = ((s_t(1) / s_t(3)))*element_matrix_tmp72;
    const s_t element_matrix_tmp201 = element_matrix_tmp24*element_matrix_tmp73 + element_matrix_tmp33*element_matrix_tmp68 + element_matrix_tmp40*element_matrix_tmp71 - element_matrix_tmp42*element_matrix_tmp72 + element_matrix_tmp45*element_matrix_tmp61 - element_matrix_tmp47*element_matrix_tmp67;
    const s_t element_matrix_tmp202 = element_matrix_tmp54*(s_t(2)*element_matrix_tmp33*element_matrix_tmp73 + s_t(2)*element_matrix_tmp47*element_matrix_tmp71 - element_matrix_tmp77 - element_matrix_tmp78 - s_t(2)*element_matrix_tmp79 - element_matrix_tmp82 - element_matrix_tmp85) + element_matrix_tmp86;
    const s_t element_matrix_tmp203 = -(s_t(1) / s_t(3))*element_matrix_tmp202;
    const s_t element_matrix_tmp204 = element_matrix_tmp58*(element_matrix_tmp203*element_matrix_tmp72 + eta_s*(element_matrix_tmp201*element_matrix_tmp61 + element_matrix_tmp64*element_matrix_tmp74));
    const s_t element_matrix_tmp205 = element_matrix_tmp191 + element_matrix_tmp194*element_matrix_tmp97 + element_matrix_tmp204*element_matrix_tmp64 + element_matrix_tmp23*(-element_matrix_tmp199*element_matrix_tmp200 + eta_s*(element_matrix_tmp118*element_matrix_tmp64 + element_matrix_tmp198*element_matrix_tmp61));
    const s_t element_matrix_tmp206 = element_matrix_tmp5*u0_grad_2;
    const s_t element_matrix_tmp207 = s_t(6)*u2_grad_0;
    const s_t element_matrix_tmp208 = s_t(2)*element_matrix_tmp15;
    const s_t element_matrix_tmp209 = element_matrix_tmp207 - element_matrix_tmp208;
    const s_t element_matrix_tmp210 = lmbda*(-element_matrix_tmp1*element_matrix_tmp17 + element_matrix_tmp1*element_matrix_tmp173*element_matrix_tmp5 - element_matrix_tmp173*element_matrix_tmp8 - element_matrix_tmp18*element_matrix_tmp5 + u0_grad_1*u1_grad_2*u2_grad_0 + u0_grad_2*u1_grad_0*u2_grad_1 + s_t(-1));
    const s_t element_matrix_tmp211 = element_matrix_tmp210*u2_grad_0;
    const s_t element_matrix_tmp212 = -element_matrix_tmp211;
    const s_t element_matrix_tmp213 = element_matrix_tmp135 + element_matrix_tmp54*(-element_matrix_tmp131 - s_t(2)*element_matrix_tmp133 + s_t(2)*element_matrix_tmp45*u2_grad_0);
    const s_t element_matrix_tmp214 = element_matrix_tmp68*u2_grad_1;
    const s_t element_matrix_tmp215 = element_matrix_tmp42*u2_grad_0;
    const s_t element_matrix_tmp216 = element_matrix_tmp45*u1_grad_0 - element_matrix_tmp5*element_matrix_tmp73;
    const s_t element_matrix_tmp217 = -element_matrix_tmp214 + element_matrix_tmp215 - element_matrix_tmp216;
    const s_t element_matrix_tmp218 = element_matrix_tmp201*u1_grad_0;
    const s_t element_matrix_tmp219 = element_matrix_tmp127*element_matrix_tmp194 + element_matrix_tmp204*element_matrix_tmp66 + element_matrix_tmp212 + element_matrix_tmp23*(-element_matrix_tmp200*element_matrix_tmp213 - element_matrix_tmp203*u2_grad_0 + eta_s*(element_matrix_tmp143*element_matrix_tmp64 + element_matrix_tmp217*element_matrix_tmp61 - element_matrix_tmp218)) + mu*(s_t(4)*element_matrix_tmp206 + element_matrix_tmp209);
    const s_t element_matrix_tmp220 = s_t(2)*element_matrix_tmp17;
    const s_t element_matrix_tmp221 = s_t(6)*u2_grad_2 + s_t(6);
    const s_t element_matrix_tmp222 = element_matrix_tmp220 + element_matrix_tmp221;
    const s_t element_matrix_tmp223 = s_t(2)*u2_grad_2 + s_t(2);
    const s_t element_matrix_tmp224 = ((s_t(1) / s_t(2)))*element_matrix_tmp210;
    const s_t element_matrix_tmp225 = element_matrix_tmp223*element_matrix_tmp224;
    const s_t element_matrix_tmp226 = element_matrix_tmp51 + element_matrix_tmp54*(element_matrix_tmp44 - s_t(2)*element_matrix_tmp46 + s_t(2)*element_matrix_tmp48);
    const s_t element_matrix_tmp227 = element_matrix_tmp40*u2_grad_1;
    const s_t element_matrix_tmp228 = element_matrix_tmp1*element_matrix_tmp42;
    const s_t element_matrix_tmp229 = element_matrix_tmp45*u1_grad_2 - element_matrix_tmp47*element_matrix_tmp5;
    const s_t element_matrix_tmp230 = element_matrix_tmp227 - element_matrix_tmp228 + element_matrix_tmp229;
    const s_t element_matrix_tmp231 = element_matrix_tmp201*u1_grad_2;
    const s_t element_matrix_tmp232 = element_matrix_tmp194*element_matrix_tmp9 + element_matrix_tmp204*element_matrix_tmp57 + element_matrix_tmp225 + element_matrix_tmp23*(element_matrix_tmp1*element_matrix_tmp203 - element_matrix_tmp200*element_matrix_tmp226 + eta_s*(element_matrix_tmp230*element_matrix_tmp61 + element_matrix_tmp231 + element_matrix_tmp38*element_matrix_tmp64)) + mu*(s_t(2)*element_matrix_tmp172*element_matrix_tmp5 - element_matrix_tmp222);
    const s_t element_matrix_tmp233 = basis0_grad0*element_matrix_tmp232 + basis0_grad1*element_matrix_tmp205 + basis0_grad2*element_matrix_tmp219;
    const s_t element_matrix_tmp234 = element_matrix_tmp162*u0_grad_1;
    const s_t element_matrix_tmp235 = mu*(-element_matrix_tmp190 - element_matrix_tmp234);
    const s_t element_matrix_tmp236 = s_t(2)*element_matrix_tmp70;
    const s_t element_matrix_tmp237 = element_matrix_tmp173*u2_grad_1;
    const s_t element_matrix_tmp238 = s_t(2)*element_matrix_tmp237;
    const s_t element_matrix_tmp239 = element_matrix_tmp12*(element_matrix_tmp236 - element_matrix_tmp238);
    const s_t element_matrix_tmp240 = element_matrix_tmp58*(((s_t(1) / s_t(3)))*element_matrix_tmp202*element_matrix_tmp71 - eta_s*(element_matrix_tmp201*element_matrix_tmp67 - element_matrix_tmp66*element_matrix_tmp74));
    const s_t element_matrix_tmp241 = element_matrix_tmp127*element_matrix_tmp239 + element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp213*element_matrix_tmp71 - eta_s*(-element_matrix_tmp143*element_matrix_tmp66 + element_matrix_tmp217*element_matrix_tmp67)) + element_matrix_tmp235 + element_matrix_tmp240*element_matrix_tmp66;
    const s_t element_matrix_tmp242 = ((s_t(1) / s_t(3)))*element_matrix_tmp202;
    const s_t element_matrix_tmp243 = s_t(2)*element_matrix_tmp206;
    const s_t element_matrix_tmp244 = element_matrix_tmp211 + mu*(-element_matrix_tmp207 - element_matrix_tmp243 + s_t(4)*u0_grad_1*u1_grad_2);
    const s_t element_matrix_tmp245 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp199*element_matrix_tmp71 - element_matrix_tmp242*u2_grad_0 - eta_s*(-element_matrix_tmp118*element_matrix_tmp66 + element_matrix_tmp198*element_matrix_tmp67 - element_matrix_tmp218)) + element_matrix_tmp239*element_matrix_tmp97 + element_matrix_tmp240*element_matrix_tmp64 + element_matrix_tmp244;
    const s_t element_matrix_tmp246 = s_t(2)*element_matrix_tmp172;
    const s_t element_matrix_tmp247 = s_t(6)*u2_grad_1;
    const s_t element_matrix_tmp248 = s_t(2)*element_matrix_tmp60;
    const s_t element_matrix_tmp249 = element_matrix_tmp247 - element_matrix_tmp248;
    const s_t element_matrix_tmp250 = element_matrix_tmp210*u2_grad_1;
    const s_t element_matrix_tmp251 = -element_matrix_tmp250;
    const s_t element_matrix_tmp252 = ((s_t(1) / s_t(3)))*element_matrix_tmp71;
    const s_t element_matrix_tmp253 = element_matrix_tmp201*element_matrix_tmp5;
    const s_t element_matrix_tmp254 = element_matrix_tmp242*u2_grad_1;
    const s_t element_matrix_tmp255 = element_matrix_tmp23*(element_matrix_tmp226*element_matrix_tmp252 + element_matrix_tmp254 - eta_s*(element_matrix_tmp230*element_matrix_tmp67 + element_matrix_tmp253 - element_matrix_tmp38*element_matrix_tmp66)) + element_matrix_tmp239*element_matrix_tmp9 + element_matrix_tmp240*element_matrix_tmp57 + element_matrix_tmp251 + mu*(element_matrix_tmp246*u1_grad_2 + element_matrix_tmp249);
    const s_t element_matrix_tmp256 = basis0_grad0*element_matrix_tmp255 + basis0_grad1*element_matrix_tmp245 + basis0_grad2*element_matrix_tmp241;
    const s_t element_matrix_tmp257 = mu*(-element_matrix_tmp189 - element_matrix_tmp234);
    const s_t element_matrix_tmp258 = -s_t(2)*element_matrix_tmp16;
    const s_t element_matrix_tmp259 = element_matrix_tmp1*u0_grad_1;
    const s_t element_matrix_tmp260 = s_t(2)*element_matrix_tmp259;
    const s_t element_matrix_tmp261 = element_matrix_tmp12*(-element_matrix_tmp258 - element_matrix_tmp260);
    const s_t element_matrix_tmp262 = element_matrix_tmp58*(((s_t(1) / s_t(3)))*element_matrix_tmp202*element_matrix_tmp33 - eta_s*(element_matrix_tmp19*element_matrix_tmp74 - element_matrix_tmp201*element_matrix_tmp24));
    const s_t element_matrix_tmp263 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp226*element_matrix_tmp33 - eta_s*(element_matrix_tmp19*element_matrix_tmp38 - element_matrix_tmp230*element_matrix_tmp24)) + element_matrix_tmp257 + element_matrix_tmp261*element_matrix_tmp9 + element_matrix_tmp262*element_matrix_tmp57;
    const s_t element_matrix_tmp264 = element_matrix_tmp173*u1_grad_2;
    const s_t element_matrix_tmp265 = s_t(2)*element_matrix_tmp264;
    const s_t element_matrix_tmp266 = element_matrix_tmp250 + mu*(-element_matrix_tmp247 - element_matrix_tmp265 + s_t(4)*u0_grad_2*u1_grad_0);
    const s_t element_matrix_tmp267 = element_matrix_tmp127*element_matrix_tmp261 + element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp213*element_matrix_tmp33 - element_matrix_tmp254 - eta_s*(element_matrix_tmp143*element_matrix_tmp19 - element_matrix_tmp217*element_matrix_tmp24 - element_matrix_tmp253)) + element_matrix_tmp262*element_matrix_tmp66 + element_matrix_tmp266;
    const s_t element_matrix_tmp268 = ((s_t(1) / s_t(3)))*element_matrix_tmp33;
    const s_t element_matrix_tmp269 = -s_t(2)*element_matrix_tmp173*element_matrix_tmp5;
    const s_t element_matrix_tmp270 = -element_matrix_tmp223*element_matrix_tmp224 + mu*(s_t(4)*element_matrix_tmp17 + element_matrix_tmp221 + element_matrix_tmp269);
    const s_t element_matrix_tmp271 = element_matrix_tmp23*(element_matrix_tmp1*element_matrix_tmp242 + element_matrix_tmp199*element_matrix_tmp268 - eta_s*(element_matrix_tmp118*element_matrix_tmp19 - element_matrix_tmp198*element_matrix_tmp24 + element_matrix_tmp231)) + element_matrix_tmp261*element_matrix_tmp97 + element_matrix_tmp262*element_matrix_tmp64 + element_matrix_tmp270;
    const s_t element_matrix_tmp272 = basis0_grad0*element_matrix_tmp263 + basis0_grad1*element_matrix_tmp271 + basis0_grad2*element_matrix_tmp267;
    const s_t element_matrix_tmp273 = element_matrix_tmp1*element_matrix_tmp188;
    const s_t element_matrix_tmp274 = element_matrix_tmp173*element_matrix_tmp91;
    const s_t element_matrix_tmp275 = mu*(-element_matrix_tmp273 - element_matrix_tmp274);
    const s_t element_matrix_tmp276 = element_matrix_tmp12*(element_matrix_tmp248 - element_matrix_tmp265);
    const s_t element_matrix_tmp277 = element_matrix_tmp105 + element_matrix_tmp54*(-s_t(2)*element_matrix_tmp100 - element_matrix_tmp104 + s_t(2)*element_matrix_tmp40*u1_grad_0 - element_matrix_tmp98);
    const s_t element_matrix_tmp278 = element_matrix_tmp54*(s_t(2)*element_matrix_tmp24*element_matrix_tmp68 + s_t(2)*element_matrix_tmp42*element_matrix_tmp61 - element_matrix_tmp75 - element_matrix_tmp76 - element_matrix_tmp80 - s_t(2)*element_matrix_tmp81 - element_matrix_tmp85) + element_matrix_tmp86;
    const s_t element_matrix_tmp279 = element_matrix_tmp58*(((s_t(1) / s_t(3)))*element_matrix_tmp278*element_matrix_tmp61 - eta_s*(element_matrix_tmp201*element_matrix_tmp72 - element_matrix_tmp64*element_matrix_tmp69));
    const s_t element_matrix_tmp280 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp277*element_matrix_tmp61 - eta_s*(-element_matrix_tmp113*element_matrix_tmp64 + element_matrix_tmp198*element_matrix_tmp72)) + element_matrix_tmp275 + element_matrix_tmp276*element_matrix_tmp97 + element_matrix_tmp279*element_matrix_tmp64;
    const s_t element_matrix_tmp281 = element_matrix_tmp135 + element_matrix_tmp54*(-element_matrix_tmp128 - s_t(2)*element_matrix_tmp130 - element_matrix_tmp134 + s_t(2)*element_matrix_tmp5*element_matrix_tmp68);
    const s_t element_matrix_tmp282 = element_matrix_tmp201*u2_grad_0;
    const s_t element_matrix_tmp283 = ((s_t(1) / s_t(3)))*element_matrix_tmp278;
    const s_t element_matrix_tmp284 = s_t(6)*u1_grad_0;
    const s_t element_matrix_tmp285 = element_matrix_tmp210*u1_grad_0;
    const s_t element_matrix_tmp286 = element_matrix_tmp285 + mu*(-element_matrix_tmp260 - element_matrix_tmp284 + s_t(4)*u0_grad_2*u2_grad_1);
    const s_t element_matrix_tmp287 = element_matrix_tmp127*element_matrix_tmp276 + element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp281*element_matrix_tmp61 - element_matrix_tmp283*u1_grad_0 - eta_s*(-element_matrix_tmp148*element_matrix_tmp64 + element_matrix_tmp217*element_matrix_tmp72 - element_matrix_tmp282)) + element_matrix_tmp279*element_matrix_tmp66 + element_matrix_tmp286;
    const s_t element_matrix_tmp288 = s_t(6)*u1_grad_2;
    const s_t element_matrix_tmp289 = -element_matrix_tmp236 + element_matrix_tmp288;
    const s_t element_matrix_tmp290 = element_matrix_tmp210*u1_grad_2;
    const s_t element_matrix_tmp291 = -element_matrix_tmp290;
    const s_t element_matrix_tmp292 = element_matrix_tmp51 + element_matrix_tmp54*(element_matrix_tmp39 - s_t(2)*element_matrix_tmp41 + s_t(2)*element_matrix_tmp43 + element_matrix_tmp49);
    const s_t element_matrix_tmp293 = ((s_t(1) / s_t(3)))*element_matrix_tmp61;
    const s_t element_matrix_tmp294 = element_matrix_tmp1*element_matrix_tmp201;
    const s_t element_matrix_tmp295 = element_matrix_tmp283*u1_grad_2;
    const s_t element_matrix_tmp296 = element_matrix_tmp23*(element_matrix_tmp292*element_matrix_tmp293 + element_matrix_tmp295 - eta_s*(element_matrix_tmp230*element_matrix_tmp72 + element_matrix_tmp294 - element_matrix_tmp31*element_matrix_tmp64)) + element_matrix_tmp276*element_matrix_tmp9 + element_matrix_tmp279*element_matrix_tmp57 + element_matrix_tmp291 + mu*(element_matrix_tmp246*u2_grad_1 + element_matrix_tmp289);
    const s_t element_matrix_tmp297 = basis0_grad0*element_matrix_tmp296 + basis0_grad1*element_matrix_tmp280 + basis0_grad2*element_matrix_tmp287;
    const s_t element_matrix_tmp298 = element_matrix_tmp164*u0_grad_1;
    const s_t element_matrix_tmp299 = mu*(-element_matrix_tmp274 - element_matrix_tmp298);
    const s_t element_matrix_tmp300 = element_matrix_tmp12*(-element_matrix_tmp220 - element_matrix_tmp269);
    const s_t element_matrix_tmp301 = ((s_t(1) / s_t(3)))*element_matrix_tmp67;
    const s_t element_matrix_tmp302 = -(s_t(1) / s_t(3))*element_matrix_tmp278;
    const s_t element_matrix_tmp303 = element_matrix_tmp58*(element_matrix_tmp302*element_matrix_tmp67 + eta_s*(element_matrix_tmp201*element_matrix_tmp71 + element_matrix_tmp66*element_matrix_tmp69));
    const s_t element_matrix_tmp304 = element_matrix_tmp127*element_matrix_tmp300 + element_matrix_tmp23*(-element_matrix_tmp281*element_matrix_tmp301 + eta_s*(element_matrix_tmp148*element_matrix_tmp66 + element_matrix_tmp217*element_matrix_tmp71)) + element_matrix_tmp299 + element_matrix_tmp303*element_matrix_tmp66;
    const s_t element_matrix_tmp305 = element_matrix_tmp258 + element_matrix_tmp284;
    const s_t element_matrix_tmp306 = -element_matrix_tmp285;
    const s_t element_matrix_tmp307 = element_matrix_tmp23*(-element_matrix_tmp277*element_matrix_tmp301 - element_matrix_tmp302*u1_grad_0 + eta_s*(element_matrix_tmp113*element_matrix_tmp66 + element_matrix_tmp198*element_matrix_tmp71 - element_matrix_tmp282)) + element_matrix_tmp300*element_matrix_tmp97 + element_matrix_tmp303*element_matrix_tmp64 + element_matrix_tmp306 + mu*(s_t(4)*element_matrix_tmp259 + element_matrix_tmp305);
    const s_t element_matrix_tmp308 = s_t(6)*u1_grad_1 + s_t(6);
    const s_t element_matrix_tmp309 = element_matrix_tmp192 + element_matrix_tmp308;
    const s_t element_matrix_tmp310 = s_t(2)*u1_grad_1 + s_t(2);
    const s_t element_matrix_tmp311 = element_matrix_tmp224*element_matrix_tmp310;
    const s_t element_matrix_tmp312 = element_matrix_tmp201*u2_grad_1;
    const s_t element_matrix_tmp313 = element_matrix_tmp23*(-element_matrix_tmp292*element_matrix_tmp301 + element_matrix_tmp302*element_matrix_tmp5 + eta_s*(element_matrix_tmp230*element_matrix_tmp71 + element_matrix_tmp31*element_matrix_tmp66 + element_matrix_tmp312)) + element_matrix_tmp300*element_matrix_tmp9 + element_matrix_tmp303*element_matrix_tmp57 + element_matrix_tmp311 + mu*(s_t(2)*element_matrix_tmp1*element_matrix_tmp172 - element_matrix_tmp309);
    const s_t element_matrix_tmp314 = basis0_grad0*element_matrix_tmp313 + basis0_grad1*element_matrix_tmp307 + basis0_grad2*element_matrix_tmp304;
    const s_t element_matrix_tmp315 = mu*(-element_matrix_tmp273 - element_matrix_tmp298);
    const s_t element_matrix_tmp316 = element_matrix_tmp12*(element_matrix_tmp208 - element_matrix_tmp243);
    const s_t element_matrix_tmp317 = element_matrix_tmp58*(((s_t(1) / s_t(3)))*element_matrix_tmp24*element_matrix_tmp278 - eta_s*(element_matrix_tmp19*element_matrix_tmp69 - element_matrix_tmp201*element_matrix_tmp33));
    const s_t element_matrix_tmp318 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp24*element_matrix_tmp292 - eta_s*(element_matrix_tmp19*element_matrix_tmp31 - element_matrix_tmp230*element_matrix_tmp33)) + element_matrix_tmp315 + element_matrix_tmp316*element_matrix_tmp9 + element_matrix_tmp317*element_matrix_tmp57;
    const s_t element_matrix_tmp319 = element_matrix_tmp290 + mu*(-element_matrix_tmp238 - element_matrix_tmp288 + s_t(4)*u0_grad_1*u2_grad_0);
    const s_t element_matrix_tmp320 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp24*element_matrix_tmp277 - element_matrix_tmp295 - eta_s*(element_matrix_tmp113*element_matrix_tmp19 - element_matrix_tmp198*element_matrix_tmp33 - element_matrix_tmp294)) + element_matrix_tmp316*element_matrix_tmp97 + element_matrix_tmp317*element_matrix_tmp64 + element_matrix_tmp319;
    const s_t element_matrix_tmp321 = ((s_t(1) / s_t(3)))*element_matrix_tmp24;
    const s_t element_matrix_tmp322 = -element_matrix_tmp224*element_matrix_tmp310 + mu*(s_t(4)*element_matrix_tmp18 + element_matrix_tmp193 + element_matrix_tmp308);
    const s_t element_matrix_tmp323 = element_matrix_tmp127*element_matrix_tmp316 + element_matrix_tmp23*(element_matrix_tmp281*element_matrix_tmp321 + element_matrix_tmp283*element_matrix_tmp5 - eta_s*(element_matrix_tmp148*element_matrix_tmp19 - element_matrix_tmp217*element_matrix_tmp33 + element_matrix_tmp312)) + element_matrix_tmp317*element_matrix_tmp66 + element_matrix_tmp322;
    const s_t element_matrix_tmp324 = basis0_grad0*element_matrix_tmp318 + basis0_grad1*element_matrix_tmp320 + basis0_grad2*element_matrix_tmp323;
    const s_t element_matrix_tmp325 = basis1_grad0*element_matrix_tmp90 + basis1_grad1*element_matrix_tmp122 + basis1_grad2*element_matrix_tmp152;
    const s_t element_matrix_tmp326 = basis1_grad0*element_matrix_tmp176 + basis1_grad1*element_matrix_tmp161 + basis1_grad2*element_matrix_tmp171;
    const s_t element_matrix_tmp327 = basis1_grad0*element_matrix_tmp185 + basis1_grad1*element_matrix_tmp183 + basis1_grad2*element_matrix_tmp182;
    const s_t element_matrix_tmp328 = basis1_grad0*element_matrix_tmp232 + basis1_grad1*element_matrix_tmp205 + basis1_grad2*element_matrix_tmp219;
    const s_t element_matrix_tmp329 = basis1_grad0*element_matrix_tmp255 + basis1_grad1*element_matrix_tmp245 + basis1_grad2*element_matrix_tmp241;
    const s_t element_matrix_tmp330 = basis1_grad0*element_matrix_tmp263 + basis1_grad1*element_matrix_tmp271 + basis1_grad2*element_matrix_tmp267;
    const s_t element_matrix_tmp331 = basis1_grad0*element_matrix_tmp296 + basis1_grad1*element_matrix_tmp280 + basis1_grad2*element_matrix_tmp287;
    const s_t element_matrix_tmp332 = basis1_grad0*element_matrix_tmp313 + basis1_grad1*element_matrix_tmp307 + basis1_grad2*element_matrix_tmp304;
    const s_t element_matrix_tmp333 = basis1_grad0*element_matrix_tmp318 + basis1_grad1*element_matrix_tmp320 + basis1_grad2*element_matrix_tmp323;
    const s_t element_matrix_tmp334 = basis2_grad0*element_matrix_tmp90 + basis2_grad1*element_matrix_tmp122 + basis2_grad2*element_matrix_tmp152;
    const s_t element_matrix_tmp335 = basis2_grad0*element_matrix_tmp176 + basis2_grad1*element_matrix_tmp161 + basis2_grad2*element_matrix_tmp171;
    const s_t element_matrix_tmp336 = basis2_grad0*element_matrix_tmp185 + basis2_grad1*element_matrix_tmp183 + basis2_grad2*element_matrix_tmp182;
    const s_t element_matrix_tmp337 = basis2_grad0*element_matrix_tmp232 + basis2_grad1*element_matrix_tmp205 + basis2_grad2*element_matrix_tmp219;
    const s_t element_matrix_tmp338 = basis2_grad0*element_matrix_tmp255 + basis2_grad1*element_matrix_tmp245 + basis2_grad2*element_matrix_tmp241;
    const s_t element_matrix_tmp339 = basis2_grad0*element_matrix_tmp263 + basis2_grad1*element_matrix_tmp271 + basis2_grad2*element_matrix_tmp267;
    const s_t element_matrix_tmp340 = basis2_grad0*element_matrix_tmp296 + basis2_grad1*element_matrix_tmp280 + basis2_grad2*element_matrix_tmp287;
    const s_t element_matrix_tmp341 = basis2_grad0*element_matrix_tmp313 + basis2_grad1*element_matrix_tmp307 + basis2_grad2*element_matrix_tmp304;
    const s_t element_matrix_tmp342 = basis2_grad0*element_matrix_tmp318 + basis2_grad1*element_matrix_tmp320 + basis2_grad2*element_matrix_tmp323;
    const s_t element_matrix_tmp343 = basis3_grad0*element_matrix_tmp90 + basis3_grad1*element_matrix_tmp122 + basis3_grad2*element_matrix_tmp152;
    const s_t element_matrix_tmp344 = basis3_grad0*element_matrix_tmp176 + basis3_grad1*element_matrix_tmp161 + basis3_grad2*element_matrix_tmp171;
    const s_t element_matrix_tmp345 = basis3_grad0*element_matrix_tmp185 + basis3_grad1*element_matrix_tmp183 + basis3_grad2*element_matrix_tmp182;
    const s_t element_matrix_tmp346 = basis3_grad0*element_matrix_tmp232 + basis3_grad1*element_matrix_tmp205 + basis3_grad2*element_matrix_tmp219;
    const s_t element_matrix_tmp347 = basis3_grad0*element_matrix_tmp255 + basis3_grad1*element_matrix_tmp245 + basis3_grad2*element_matrix_tmp241;
    const s_t element_matrix_tmp348 = basis3_grad0*element_matrix_tmp263 + basis3_grad1*element_matrix_tmp271 + basis3_grad2*element_matrix_tmp267;
    const s_t element_matrix_tmp349 = basis3_grad0*element_matrix_tmp296 + basis3_grad1*element_matrix_tmp280 + basis3_grad2*element_matrix_tmp287;
    const s_t element_matrix_tmp350 = basis3_grad0*element_matrix_tmp313 + basis3_grad1*element_matrix_tmp307 + basis3_grad2*element_matrix_tmp304;
    const s_t element_matrix_tmp351 = basis3_grad0*element_matrix_tmp318 + basis3_grad1*element_matrix_tmp320 + basis3_grad2*element_matrix_tmp323;
    const s_t element_matrix_tmp352 = -element_matrix_tmp259 - element_matrix_tmp32;
    const s_t element_matrix_tmp353 = -element_matrix_tmp39 - element_matrix_tmp52;
    const s_t element_matrix_tmp354 = -element_matrix_tmp26*u0_grad_2 + element_matrix_tmp28*u0_grad_1;
    const s_t element_matrix_tmp355 = -element_matrix_tmp227 + element_matrix_tmp228 + element_matrix_tmp354;
    const s_t element_matrix_tmp356 = element_matrix_tmp40*u0_grad_1;
    const s_t element_matrix_tmp357 = element_matrix_tmp42*u0_grad_2;
    const s_t element_matrix_tmp358 = element_matrix_tmp34 + element_matrix_tmp356 - element_matrix_tmp357;
    const s_t element_matrix_tmp359 = -element_matrix_tmp35 + element_matrix_tmp36;
    const s_t element_matrix_tmp360 = element_matrix_tmp50*(element_matrix_tmp358 + element_matrix_tmp359);
    const s_t element_matrix_tmp361 = element_matrix_tmp360 + element_matrix_tmp54*(s_t(2)*element_matrix_tmp1*element_matrix_tmp26 - s_t(2)*element_matrix_tmp35 - element_matrix_tmp358);
    const s_t element_matrix_tmp362 = element_matrix_tmp13*element_matrix_tmp352 + element_matrix_tmp23*(-element_matrix_tmp361*element_matrix_tmp56 + eta_s*(element_matrix_tmp24*element_matrix_tmp355 + element_matrix_tmp33*element_matrix_tmp353)) + element_matrix_tmp257 + element_matrix_tmp33*element_matrix_tmp89;
    const s_t element_matrix_tmp363 = -element_matrix_tmp237 + element_matrix_tmp70;
    const s_t element_matrix_tmp364 = element_matrix_tmp173*element_matrix_tmp42;
    const s_t element_matrix_tmp365 = element_matrix_tmp68*u0_grad_1;
    const s_t element_matrix_tmp366 = element_matrix_tmp139 + element_matrix_tmp364 - element_matrix_tmp365;
    const s_t element_matrix_tmp367 = -element_matrix_tmp140 + element_matrix_tmp141;
    const s_t element_matrix_tmp368 = element_matrix_tmp50*(element_matrix_tmp366 + element_matrix_tmp367);
    const s_t element_matrix_tmp369 = element_matrix_tmp368 + element_matrix_tmp54*(-s_t(2)*element_matrix_tmp140 - element_matrix_tmp366 + s_t(2)*element_matrix_tmp59*u2_grad_1);
    const s_t element_matrix_tmp370 = element_matrix_tmp128 + element_matrix_tmp136;
    const s_t element_matrix_tmp371 = -element_matrix_tmp173*element_matrix_tmp26 + element_matrix_tmp59*u0_grad_1;
    const s_t element_matrix_tmp372 = element_matrix_tmp214 - element_matrix_tmp215 - element_matrix_tmp371;
    const s_t element_matrix_tmp373 = element_matrix_tmp69*u0_grad_1;
    const s_t element_matrix_tmp374 = ((s_t(1) / s_t(3)))*element_matrix_tmp88;
    const s_t element_matrix_tmp375 = element_matrix_tmp13*element_matrix_tmp363 + element_matrix_tmp23*(-element_matrix_tmp369*element_matrix_tmp56 - element_matrix_tmp374*u2_grad_1 + eta_s*(element_matrix_tmp24*element_matrix_tmp372 + element_matrix_tmp33*element_matrix_tmp370 - element_matrix_tmp373)) + element_matrix_tmp251 + element_matrix_tmp71*element_matrix_tmp89 + mu*(element_matrix_tmp249 + s_t(4)*element_matrix_tmp264);
    const s_t element_matrix_tmp376 = element_matrix_tmp1*element_matrix_tmp173 - element_matrix_tmp18;
    const s_t element_matrix_tmp377 = -element_matrix_tmp72;
    const s_t element_matrix_tmp378 = element_matrix_tmp173*element_matrix_tmp40;
    const s_t element_matrix_tmp379 = element_matrix_tmp68*u0_grad_2;
    const s_t element_matrix_tmp380 = element_matrix_tmp114 + element_matrix_tmp378 - element_matrix_tmp379;
    const s_t element_matrix_tmp381 = -element_matrix_tmp115 + element_matrix_tmp116;
    const s_t element_matrix_tmp382 = element_matrix_tmp50*(-element_matrix_tmp380 - element_matrix_tmp381);
    const s_t element_matrix_tmp383 = element_matrix_tmp382 + element_matrix_tmp54*(s_t(2)*element_matrix_tmp115 - s_t(2)*element_matrix_tmp116 + element_matrix_tmp380);
    const s_t element_matrix_tmp384 = element_matrix_tmp106 + element_matrix_tmp98;
    const s_t element_matrix_tmp385 = -element_matrix_tmp173*element_matrix_tmp28 + element_matrix_tmp59*u0_grad_2;
    const s_t element_matrix_tmp386 = -element_matrix_tmp195 + element_matrix_tmp196 + element_matrix_tmp385;
    const s_t element_matrix_tmp387 = element_matrix_tmp69*u0_grad_2;
    const s_t element_matrix_tmp388 = element_matrix_tmp13*element_matrix_tmp376 + element_matrix_tmp225 + element_matrix_tmp23*(element_matrix_tmp1*element_matrix_tmp374 - element_matrix_tmp383*element_matrix_tmp56 + eta_s*(element_matrix_tmp24*element_matrix_tmp386 + element_matrix_tmp33*element_matrix_tmp384 + element_matrix_tmp387)) + element_matrix_tmp377*element_matrix_tmp89 + mu*(s_t(2)*element_matrix_tmp173*element_matrix_tmp310 - element_matrix_tmp222);
    const s_t element_matrix_tmp389 = basis0_grad0*element_matrix_tmp362 + basis0_grad1*element_matrix_tmp388 + basis0_grad2*element_matrix_tmp375;
    const s_t element_matrix_tmp390 = element_matrix_tmp180*element_matrix_tmp363 + element_matrix_tmp181*element_matrix_tmp71 + element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp369*element_matrix_tmp66 - eta_s*(-element_matrix_tmp370*element_matrix_tmp71 + element_matrix_tmp372*element_matrix_tmp67)) + element_matrix_tmp235;
    const s_t element_matrix_tmp391 = ((s_t(1) / s_t(3)))*element_matrix_tmp87;
    const s_t element_matrix_tmp392 = element_matrix_tmp180*element_matrix_tmp352 + element_matrix_tmp181*element_matrix_tmp33 + element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp361*element_matrix_tmp66 - element_matrix_tmp391*u2_grad_1 - eta_s*(-element_matrix_tmp353*element_matrix_tmp71 + element_matrix_tmp355*element_matrix_tmp67 - element_matrix_tmp373)) + element_matrix_tmp266;
    const s_t element_matrix_tmp393 = ((s_t(1) / s_t(3)))*element_matrix_tmp66;
    const s_t element_matrix_tmp394 = element_matrix_tmp173*element_matrix_tmp69;
    const s_t element_matrix_tmp395 = element_matrix_tmp391*u2_grad_0;
    const s_t element_matrix_tmp396 = element_matrix_tmp180*element_matrix_tmp376 + element_matrix_tmp181*element_matrix_tmp377 + element_matrix_tmp212 + element_matrix_tmp23*(element_matrix_tmp383*element_matrix_tmp393 + element_matrix_tmp395 - eta_s*(-element_matrix_tmp384*element_matrix_tmp71 + element_matrix_tmp386*element_matrix_tmp67 + element_matrix_tmp394)) + mu*(element_matrix_tmp188*element_matrix_tmp310 + element_matrix_tmp209);
    const s_t element_matrix_tmp397 = basis0_grad0*element_matrix_tmp392 + basis0_grad1*element_matrix_tmp396 + basis0_grad2*element_matrix_tmp390;
    const s_t element_matrix_tmp398 = element_matrix_tmp156*element_matrix_tmp376 + element_matrix_tmp160*element_matrix_tmp377 + element_matrix_tmp191 + element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp383*element_matrix_tmp64 - eta_s*(element_matrix_tmp384*element_matrix_tmp72 - element_matrix_tmp386*element_matrix_tmp61));
    const s_t element_matrix_tmp399 = element_matrix_tmp156*element_matrix_tmp363 + element_matrix_tmp160*element_matrix_tmp71 + element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp369*element_matrix_tmp64 - element_matrix_tmp395 - eta_s*(element_matrix_tmp370*element_matrix_tmp72 - element_matrix_tmp372*element_matrix_tmp61 - element_matrix_tmp394)) + element_matrix_tmp244;
    const s_t element_matrix_tmp400 = ((s_t(1) / s_t(3)))*element_matrix_tmp64;
    const s_t element_matrix_tmp401 = element_matrix_tmp156*element_matrix_tmp352 + element_matrix_tmp160*element_matrix_tmp33 + element_matrix_tmp23*(element_matrix_tmp1*element_matrix_tmp391 + element_matrix_tmp361*element_matrix_tmp400 - eta_s*(element_matrix_tmp353*element_matrix_tmp72 - element_matrix_tmp355*element_matrix_tmp61 + element_matrix_tmp387)) + element_matrix_tmp270;
    const s_t element_matrix_tmp402 = basis0_grad0*element_matrix_tmp401 + basis0_grad1*element_matrix_tmp398 + basis0_grad2*element_matrix_tmp399;
    const s_t element_matrix_tmp403 = s_t(2)*pow_2(u0_grad_2);
    const s_t element_matrix_tmp404 = element_matrix_tmp2 + element_matrix_tmp403;
    const s_t element_matrix_tmp405 = s_t(2)*pow_2(element_matrix_tmp173);
    const s_t element_matrix_tmp406 = element_matrix_tmp158 + element_matrix_tmp405;
    const s_t element_matrix_tmp407 = element_matrix_tmp73*u0_grad_2;
    const s_t element_matrix_tmp408 = element_matrix_tmp173*element_matrix_tmp47;
    const s_t element_matrix_tmp409 = element_matrix_tmp407 - element_matrix_tmp408;
    const s_t element_matrix_tmp410 = element_matrix_tmp109 + element_matrix_tmp409;
    const s_t element_matrix_tmp411 = -element_matrix_tmp378 + element_matrix_tmp379;
    const s_t element_matrix_tmp412 = element_matrix_tmp382 + element_matrix_tmp54*(-s_t(2)*element_matrix_tmp114 - element_matrix_tmp117 - element_matrix_tmp411);
    const s_t element_matrix_tmp413 = element_matrix_tmp194*element_matrix_tmp376 + element_matrix_tmp204*element_matrix_tmp377 + element_matrix_tmp23*(-element_matrix_tmp200*element_matrix_tmp412 + eta_s*(element_matrix_tmp384*element_matrix_tmp64 + element_matrix_tmp410*element_matrix_tmp61)) + mu*(element_matrix_tmp404 + element_matrix_tmp406);
    const s_t element_matrix_tmp414 = s_t(2)*element_matrix_tmp173*u0_grad_1;
    const s_t element_matrix_tmp415 = element_matrix_tmp414 + element_matrix_tmp92;
    const s_t element_matrix_tmp416 = -element_matrix_tmp356 + element_matrix_tmp357;
    const s_t element_matrix_tmp417 = element_matrix_tmp360 + element_matrix_tmp54*(s_t(2)*element_matrix_tmp34 + element_matrix_tmp37 + element_matrix_tmp416);
    const s_t element_matrix_tmp418 = element_matrix_tmp47*u0_grad_1;
    const s_t element_matrix_tmp419 = element_matrix_tmp45*u0_grad_2;
    const s_t element_matrix_tmp420 = element_matrix_tmp418 - element_matrix_tmp419;
    const s_t element_matrix_tmp421 = element_matrix_tmp25 + element_matrix_tmp420;
    const s_t element_matrix_tmp422 = element_matrix_tmp201*u0_grad_2;
    const s_t element_matrix_tmp423 = element_matrix_tmp194*element_matrix_tmp352 + element_matrix_tmp204*element_matrix_tmp33 + element_matrix_tmp23*(-element_matrix_tmp200*element_matrix_tmp417 + eta_s*(element_matrix_tmp121 + element_matrix_tmp353*element_matrix_tmp64 + element_matrix_tmp421*element_matrix_tmp61 - element_matrix_tmp422)) - element_matrix_tmp415*mu;
    const s_t element_matrix_tmp424 = element_matrix_tmp188*u0_grad_1;
    const s_t element_matrix_tmp425 = element_matrix_tmp165 + element_matrix_tmp424;
    const s_t element_matrix_tmp426 = -element_matrix_tmp364 + element_matrix_tmp365;
    const s_t element_matrix_tmp427 = element_matrix_tmp368 + element_matrix_tmp54*(s_t(2)*element_matrix_tmp139 + element_matrix_tmp142 + element_matrix_tmp426);
    const s_t element_matrix_tmp428 = element_matrix_tmp73*u0_grad_1;
    const s_t element_matrix_tmp429 = element_matrix_tmp173*element_matrix_tmp45;
    const s_t element_matrix_tmp430 = element_matrix_tmp428 - element_matrix_tmp429;
    const s_t element_matrix_tmp431 = -element_matrix_tmp144 - element_matrix_tmp430;
    const s_t element_matrix_tmp432 = element_matrix_tmp173*element_matrix_tmp201;
    const s_t element_matrix_tmp433 = element_matrix_tmp194*element_matrix_tmp363 + element_matrix_tmp204*element_matrix_tmp71 + element_matrix_tmp23*(-element_matrix_tmp200*element_matrix_tmp427 + eta_s*(element_matrix_tmp169 + element_matrix_tmp370*element_matrix_tmp64 + element_matrix_tmp431*element_matrix_tmp61 + element_matrix_tmp432)) - element_matrix_tmp425*mu;
    const s_t element_matrix_tmp434 = basis0_grad0*element_matrix_tmp423 + basis0_grad1*element_matrix_tmp413 + basis0_grad2*element_matrix_tmp433;
    const s_t element_matrix_tmp435 = s_t(2)*pow_2(u0_grad_1);
    const s_t element_matrix_tmp436 = element_matrix_tmp4 + element_matrix_tmp435;
    const s_t element_matrix_tmp437 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp33*element_matrix_tmp417 - eta_s*(element_matrix_tmp19*element_matrix_tmp353 - element_matrix_tmp24*element_matrix_tmp421)) + element_matrix_tmp261*element_matrix_tmp352 + element_matrix_tmp262*element_matrix_tmp33 + mu*(element_matrix_tmp404 + element_matrix_tmp436);
    const s_t element_matrix_tmp438 = element_matrix_tmp173*element_matrix_tmp188;
    const s_t element_matrix_tmp439 = mu*(-element_matrix_tmp124 - element_matrix_tmp438);
    const s_t element_matrix_tmp440 = element_matrix_tmp201*u0_grad_1;
    const s_t element_matrix_tmp441 = element_matrix_tmp150 + element_matrix_tmp440;
    const s_t element_matrix_tmp442 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp33*element_matrix_tmp427 - eta_s*(element_matrix_tmp19*element_matrix_tmp370 - element_matrix_tmp24*element_matrix_tmp431 + element_matrix_tmp441)) + element_matrix_tmp261*element_matrix_tmp363 + element_matrix_tmp262*element_matrix_tmp71 + element_matrix_tmp439;
    const s_t element_matrix_tmp443 = s_t(4)*element_matrix_tmp5;
    const s_t element_matrix_tmp444 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp33*element_matrix_tmp412 - eta_s*(-element_matrix_tmp175 + element_matrix_tmp19*element_matrix_tmp384 - element_matrix_tmp24*element_matrix_tmp410 - element_matrix_tmp422)) + element_matrix_tmp261*element_matrix_tmp376 + element_matrix_tmp262*element_matrix_tmp377 + mu*(s_t(2)*element_matrix_tmp310*u1_grad_0 - element_matrix_tmp415 - element_matrix_tmp443*u1_grad_0);
    const s_t element_matrix_tmp445 = basis0_grad0*element_matrix_tmp437 + basis0_grad1*element_matrix_tmp444 + basis0_grad2*element_matrix_tmp442;
    const s_t element_matrix_tmp446 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp427*element_matrix_tmp71 - eta_s*(-element_matrix_tmp370*element_matrix_tmp66 + element_matrix_tmp431*element_matrix_tmp67)) + element_matrix_tmp239*element_matrix_tmp363 + element_matrix_tmp240*element_matrix_tmp71 + mu*(element_matrix_tmp406 + element_matrix_tmp436 + s_t(2));
    const s_t element_matrix_tmp447 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp417*element_matrix_tmp71 - eta_s*(-element_matrix_tmp353*element_matrix_tmp66 + element_matrix_tmp421*element_matrix_tmp67 - element_matrix_tmp441)) + element_matrix_tmp239*element_matrix_tmp352 + element_matrix_tmp240*element_matrix_tmp33 + element_matrix_tmp439;
    const s_t element_matrix_tmp448 = -element_matrix_tmp173*element_matrix_tmp201;
    const s_t element_matrix_tmp449 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp412*element_matrix_tmp71 - eta_s*(-element_matrix_tmp168 - element_matrix_tmp384*element_matrix_tmp66 + element_matrix_tmp410*element_matrix_tmp67 - element_matrix_tmp448)) + element_matrix_tmp239*element_matrix_tmp376 + element_matrix_tmp240*element_matrix_tmp377 + mu*(s_t(2)*element_matrix_tmp310*u1_grad_2 - element_matrix_tmp425 - element_matrix_tmp443*u1_grad_2);
    const s_t element_matrix_tmp450 = basis0_grad0*element_matrix_tmp447 + basis0_grad1*element_matrix_tmp449 + basis0_grad2*element_matrix_tmp446;
    const s_t element_matrix_tmp451 = s_t(2)*element_matrix_tmp1*u1_grad_2;
    const s_t element_matrix_tmp452 = element_matrix_tmp162*u2_grad_1;
    const s_t element_matrix_tmp453 = mu*(-element_matrix_tmp451 - element_matrix_tmp452);
    const s_t element_matrix_tmp454 = element_matrix_tmp360 + element_matrix_tmp54*(-element_matrix_tmp34 - s_t(2)*element_matrix_tmp357 - element_matrix_tmp359 + s_t(2)*element_matrix_tmp40*u0_grad_1);
    const s_t element_matrix_tmp455 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp24*element_matrix_tmp454 - eta_s*(element_matrix_tmp19*element_matrix_tmp355 - element_matrix_tmp33*element_matrix_tmp421)) + element_matrix_tmp316*element_matrix_tmp352 + element_matrix_tmp317*element_matrix_tmp33 + element_matrix_tmp453;
    const s_t element_matrix_tmp456 = element_matrix_tmp368 + element_matrix_tmp54*(-element_matrix_tmp139 + s_t(2)*element_matrix_tmp173*element_matrix_tmp42 - s_t(2)*element_matrix_tmp365 - element_matrix_tmp367);
    const s_t element_matrix_tmp457 = element_matrix_tmp69*u2_grad_1;
    const s_t element_matrix_tmp458 = s_t(6)*u0_grad_1;
    const s_t element_matrix_tmp459 = element_matrix_tmp210*u0_grad_1;
    const s_t element_matrix_tmp460 = element_matrix_tmp459 + mu*(-element_matrix_tmp155 - element_matrix_tmp458 + s_t(4)*u1_grad_2*u2_grad_0);
    const s_t element_matrix_tmp461 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp24*element_matrix_tmp456 - element_matrix_tmp283*u0_grad_1 - eta_s*(element_matrix_tmp19*element_matrix_tmp372 - element_matrix_tmp33*element_matrix_tmp431 - element_matrix_tmp457)) + element_matrix_tmp316*element_matrix_tmp363 + element_matrix_tmp317*element_matrix_tmp71 + element_matrix_tmp460;
    const s_t element_matrix_tmp462 = s_t(6)*u0_grad_2;
    const s_t element_matrix_tmp463 = -element_matrix_tmp178 + element_matrix_tmp462;
    const s_t element_matrix_tmp464 = element_matrix_tmp210*u0_grad_2;
    const s_t element_matrix_tmp465 = -element_matrix_tmp464;
    const s_t element_matrix_tmp466 = element_matrix_tmp382 + element_matrix_tmp54*(element_matrix_tmp114 - s_t(2)*element_matrix_tmp378 + s_t(2)*element_matrix_tmp379 + element_matrix_tmp381);
    const s_t element_matrix_tmp467 = element_matrix_tmp1*element_matrix_tmp69;
    const s_t element_matrix_tmp468 = element_matrix_tmp283*u0_grad_2;
    const s_t element_matrix_tmp469 = element_matrix_tmp23*(element_matrix_tmp321*element_matrix_tmp466 + element_matrix_tmp468 - eta_s*(element_matrix_tmp19*element_matrix_tmp386 - element_matrix_tmp33*element_matrix_tmp410 + element_matrix_tmp467)) + element_matrix_tmp316*element_matrix_tmp376 + element_matrix_tmp317*element_matrix_tmp377 + element_matrix_tmp465 + mu*(element_matrix_tmp310*element_matrix_tmp91 + element_matrix_tmp463);
    const s_t element_matrix_tmp470 = basis0_grad0*element_matrix_tmp455 + basis0_grad1*element_matrix_tmp469 + basis0_grad2*element_matrix_tmp461;
    const s_t element_matrix_tmp471 = element_matrix_tmp93*u2_grad_0;
    const s_t element_matrix_tmp472 = mu*(-element_matrix_tmp452 - element_matrix_tmp471);
    const s_t element_matrix_tmp473 = element_matrix_tmp23*(-element_matrix_tmp301*element_matrix_tmp456 + eta_s*(element_matrix_tmp372*element_matrix_tmp66 + element_matrix_tmp431*element_matrix_tmp71)) + element_matrix_tmp300*element_matrix_tmp363 + element_matrix_tmp303*element_matrix_tmp71 + element_matrix_tmp472;
    const s_t element_matrix_tmp474 = element_matrix_tmp154 + element_matrix_tmp458;
    const s_t element_matrix_tmp475 = -element_matrix_tmp459;
    const s_t element_matrix_tmp476 = element_matrix_tmp23*(-element_matrix_tmp301*element_matrix_tmp454 - element_matrix_tmp302*u0_grad_1 + eta_s*(element_matrix_tmp355*element_matrix_tmp66 + element_matrix_tmp421*element_matrix_tmp71 - element_matrix_tmp457)) + element_matrix_tmp300*element_matrix_tmp352 + element_matrix_tmp303*element_matrix_tmp33 + element_matrix_tmp475 + mu*(element_matrix_tmp474 + s_t(4)*element_matrix_tmp96);
    const s_t element_matrix_tmp477 = s_t(6)*u0_grad_0 + s_t(6);
    const s_t element_matrix_tmp478 = element_matrix_tmp10 + element_matrix_tmp477;
    const s_t element_matrix_tmp479 = element_matrix_tmp172*element_matrix_tmp224;
    const s_t element_matrix_tmp480 = element_matrix_tmp69*u2_grad_0;
    const s_t element_matrix_tmp481 = element_matrix_tmp23*(element_matrix_tmp173*element_matrix_tmp302 - element_matrix_tmp301*element_matrix_tmp466 + eta_s*(element_matrix_tmp386*element_matrix_tmp66 + element_matrix_tmp410*element_matrix_tmp71 + element_matrix_tmp480)) + element_matrix_tmp300*element_matrix_tmp376 + element_matrix_tmp303*element_matrix_tmp377 + element_matrix_tmp479 + mu*(s_t(2)*element_matrix_tmp1*element_matrix_tmp310 - element_matrix_tmp478);
    const s_t element_matrix_tmp482 = basis0_grad0*element_matrix_tmp476 + basis0_grad1*element_matrix_tmp481 + basis0_grad2*element_matrix_tmp473;
    const s_t element_matrix_tmp483 = mu*(-element_matrix_tmp451 - element_matrix_tmp471);
    const s_t element_matrix_tmp484 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp466*element_matrix_tmp61 - eta_s*(-element_matrix_tmp386*element_matrix_tmp64 + element_matrix_tmp410*element_matrix_tmp72)) + element_matrix_tmp276*element_matrix_tmp376 + element_matrix_tmp279*element_matrix_tmp377 + element_matrix_tmp483;
    const s_t element_matrix_tmp485 = element_matrix_tmp464 + mu*(-element_matrix_tmp179 - element_matrix_tmp462 + s_t(4)*u1_grad_0*u2_grad_1);
    const s_t element_matrix_tmp486 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp454*element_matrix_tmp61 - element_matrix_tmp468 - eta_s*(-element_matrix_tmp355*element_matrix_tmp64 + element_matrix_tmp421*element_matrix_tmp72 - element_matrix_tmp467)) + element_matrix_tmp276*element_matrix_tmp352 + element_matrix_tmp279*element_matrix_tmp33 + element_matrix_tmp485;
    const s_t element_matrix_tmp487 = -element_matrix_tmp172*element_matrix_tmp224 + mu*(element_matrix_tmp11 + element_matrix_tmp477 + s_t(4)*element_matrix_tmp8);
    const s_t element_matrix_tmp488 = element_matrix_tmp23*(element_matrix_tmp173*element_matrix_tmp283 + element_matrix_tmp293*element_matrix_tmp456 - eta_s*(-element_matrix_tmp372*element_matrix_tmp64 + element_matrix_tmp431*element_matrix_tmp72 + element_matrix_tmp480)) + element_matrix_tmp276*element_matrix_tmp363 + element_matrix_tmp279*element_matrix_tmp71 + element_matrix_tmp487;
    const s_t element_matrix_tmp489 = basis0_grad0*element_matrix_tmp486 + basis0_grad1*element_matrix_tmp484 + basis0_grad2*element_matrix_tmp488;
    const s_t element_matrix_tmp490 = basis1_grad0*element_matrix_tmp362 + basis1_grad1*element_matrix_tmp388 + basis1_grad2*element_matrix_tmp375;
    const s_t element_matrix_tmp491 = basis1_grad0*element_matrix_tmp392 + basis1_grad1*element_matrix_tmp396 + basis1_grad2*element_matrix_tmp390;
    const s_t element_matrix_tmp492 = basis1_grad0*element_matrix_tmp401 + basis1_grad1*element_matrix_tmp398 + basis1_grad2*element_matrix_tmp399;
    const s_t element_matrix_tmp493 = basis1_grad0*element_matrix_tmp423 + basis1_grad1*element_matrix_tmp413 + basis1_grad2*element_matrix_tmp433;
    const s_t element_matrix_tmp494 = basis1_grad0*element_matrix_tmp437 + basis1_grad1*element_matrix_tmp444 + basis1_grad2*element_matrix_tmp442;
    const s_t element_matrix_tmp495 = basis1_grad0*element_matrix_tmp447 + basis1_grad1*element_matrix_tmp449 + basis1_grad2*element_matrix_tmp446;
    const s_t element_matrix_tmp496 = basis1_grad0*element_matrix_tmp455 + basis1_grad1*element_matrix_tmp469 + basis1_grad2*element_matrix_tmp461;
    const s_t element_matrix_tmp497 = basis1_grad0*element_matrix_tmp476 + basis1_grad1*element_matrix_tmp481 + basis1_grad2*element_matrix_tmp473;
    const s_t element_matrix_tmp498 = basis1_grad0*element_matrix_tmp486 + basis1_grad1*element_matrix_tmp484 + basis1_grad2*element_matrix_tmp488;
    const s_t element_matrix_tmp499 = basis2_grad0*element_matrix_tmp362 + basis2_grad1*element_matrix_tmp388 + basis2_grad2*element_matrix_tmp375;
    const s_t element_matrix_tmp500 = basis2_grad0*element_matrix_tmp392 + basis2_grad1*element_matrix_tmp396 + basis2_grad2*element_matrix_tmp390;
    const s_t element_matrix_tmp501 = basis2_grad0*element_matrix_tmp401 + basis2_grad1*element_matrix_tmp398 + basis2_grad2*element_matrix_tmp399;
    const s_t element_matrix_tmp502 = basis2_grad0*element_matrix_tmp423 + basis2_grad1*element_matrix_tmp413 + basis2_grad2*element_matrix_tmp433;
    const s_t element_matrix_tmp503 = basis2_grad0*element_matrix_tmp437 + basis2_grad1*element_matrix_tmp444 + basis2_grad2*element_matrix_tmp442;
    const s_t element_matrix_tmp504 = basis2_grad0*element_matrix_tmp447 + basis2_grad1*element_matrix_tmp449 + basis2_grad2*element_matrix_tmp446;
    const s_t element_matrix_tmp505 = basis2_grad0*element_matrix_tmp455 + basis2_grad1*element_matrix_tmp469 + basis2_grad2*element_matrix_tmp461;
    const s_t element_matrix_tmp506 = basis2_grad0*element_matrix_tmp476 + basis2_grad1*element_matrix_tmp481 + basis2_grad2*element_matrix_tmp473;
    const s_t element_matrix_tmp507 = basis2_grad0*element_matrix_tmp486 + basis2_grad1*element_matrix_tmp484 + basis2_grad2*element_matrix_tmp488;
    const s_t element_matrix_tmp508 = basis3_grad0*element_matrix_tmp362 + basis3_grad1*element_matrix_tmp388 + basis3_grad2*element_matrix_tmp375;
    const s_t element_matrix_tmp509 = basis3_grad0*element_matrix_tmp392 + basis3_grad1*element_matrix_tmp396 + basis3_grad2*element_matrix_tmp390;
    const s_t element_matrix_tmp510 = basis3_grad0*element_matrix_tmp401 + basis3_grad1*element_matrix_tmp398 + basis3_grad2*element_matrix_tmp399;
    const s_t element_matrix_tmp511 = basis3_grad0*element_matrix_tmp423 + basis3_grad1*element_matrix_tmp413 + basis3_grad2*element_matrix_tmp433;
    const s_t element_matrix_tmp512 = basis3_grad0*element_matrix_tmp437 + basis3_grad1*element_matrix_tmp444 + basis3_grad2*element_matrix_tmp442;
    const s_t element_matrix_tmp513 = basis3_grad0*element_matrix_tmp447 + basis3_grad1*element_matrix_tmp449 + basis3_grad2*element_matrix_tmp446;
    const s_t element_matrix_tmp514 = basis3_grad0*element_matrix_tmp455 + basis3_grad1*element_matrix_tmp469 + basis3_grad2*element_matrix_tmp461;
    const s_t element_matrix_tmp515 = basis3_grad0*element_matrix_tmp476 + basis3_grad1*element_matrix_tmp481 + basis3_grad2*element_matrix_tmp473;
    const s_t element_matrix_tmp516 = basis3_grad0*element_matrix_tmp486 + basis3_grad1*element_matrix_tmp484 + basis3_grad2*element_matrix_tmp488;
    const s_t element_matrix_tmp517 = element_matrix_tmp15 - element_matrix_tmp206;
    const s_t element_matrix_tmp518 = -element_matrix_tmp39 - element_matrix_tmp53;
    const s_t element_matrix_tmp519 = -element_matrix_tmp229 - element_matrix_tmp354;
    const s_t element_matrix_tmp520 = element_matrix_tmp25 - element_matrix_tmp418 + element_matrix_tmp419;
    const s_t element_matrix_tmp521 = -element_matrix_tmp27 + element_matrix_tmp29;
    const s_t element_matrix_tmp522 = element_matrix_tmp50*(element_matrix_tmp520 + element_matrix_tmp521);
    const s_t element_matrix_tmp523 = element_matrix_tmp522 + element_matrix_tmp54*(-s_t(2)*element_matrix_tmp27 + s_t(2)*element_matrix_tmp28*element_matrix_tmp5 - element_matrix_tmp520);
    const s_t element_matrix_tmp524 = element_matrix_tmp13*element_matrix_tmp517 + element_matrix_tmp23*(-element_matrix_tmp523*element_matrix_tmp56 + eta_s*(element_matrix_tmp24*element_matrix_tmp518 + element_matrix_tmp33*element_matrix_tmp519)) + element_matrix_tmp24*element_matrix_tmp89 + element_matrix_tmp315;
    const s_t element_matrix_tmp525 = -element_matrix_tmp264 + element_matrix_tmp60;
    const s_t element_matrix_tmp526 = element_matrix_tmp109 - element_matrix_tmp407 + element_matrix_tmp408;
    const s_t element_matrix_tmp527 = -element_matrix_tmp110 + element_matrix_tmp111;
    const s_t element_matrix_tmp528 = element_matrix_tmp50*(element_matrix_tmp526 + element_matrix_tmp527);
    const s_t element_matrix_tmp529 = element_matrix_tmp528 + element_matrix_tmp54*(-s_t(2)*element_matrix_tmp110 - element_matrix_tmp526 + s_t(2)*element_matrix_tmp59*u1_grad_2);
    const s_t element_matrix_tmp530 = element_matrix_tmp107 + element_matrix_tmp98;
    const s_t element_matrix_tmp531 = -element_matrix_tmp197 - element_matrix_tmp385;
    const s_t element_matrix_tmp532 = element_matrix_tmp74*u0_grad_2;
    const s_t element_matrix_tmp533 = element_matrix_tmp13*element_matrix_tmp525 + element_matrix_tmp23*(-element_matrix_tmp374*u1_grad_2 - element_matrix_tmp529*element_matrix_tmp56 + eta_s*(element_matrix_tmp24*element_matrix_tmp530 + element_matrix_tmp33*element_matrix_tmp531 - element_matrix_tmp532)) + element_matrix_tmp291 + element_matrix_tmp61*element_matrix_tmp89 + mu*(s_t(4)*element_matrix_tmp237 + element_matrix_tmp289);
    const s_t element_matrix_tmp534 = -element_matrix_tmp17 + element_matrix_tmp173*element_matrix_tmp5;
    const s_t element_matrix_tmp535 = -element_matrix_tmp67;
    const s_t element_matrix_tmp536 = element_matrix_tmp144 - element_matrix_tmp428 + element_matrix_tmp429;
    const s_t element_matrix_tmp537 = -element_matrix_tmp145 + element_matrix_tmp146;
    const s_t element_matrix_tmp538 = element_matrix_tmp50*(-element_matrix_tmp536 - element_matrix_tmp537);
    const s_t element_matrix_tmp539 = element_matrix_tmp538 + element_matrix_tmp54*(s_t(2)*element_matrix_tmp145 - s_t(2)*element_matrix_tmp146 + element_matrix_tmp536);
    const s_t element_matrix_tmp540 = element_matrix_tmp128 + element_matrix_tmp137;
    const s_t element_matrix_tmp541 = element_matrix_tmp216 + element_matrix_tmp371;
    const s_t element_matrix_tmp542 = element_matrix_tmp74*u0_grad_1;
    const s_t element_matrix_tmp543 = element_matrix_tmp13*element_matrix_tmp534 + element_matrix_tmp23*(element_matrix_tmp374*element_matrix_tmp5 - element_matrix_tmp539*element_matrix_tmp56 + eta_s*(element_matrix_tmp24*element_matrix_tmp540 + element_matrix_tmp33*element_matrix_tmp541 + element_matrix_tmp542)) + element_matrix_tmp311 + element_matrix_tmp535*element_matrix_tmp89 + mu*(s_t(2)*element_matrix_tmp173*element_matrix_tmp223 - element_matrix_tmp309);
    const s_t element_matrix_tmp544 = basis0_grad0*element_matrix_tmp524 + basis0_grad1*element_matrix_tmp533 + basis0_grad2*element_matrix_tmp543;
    const s_t element_matrix_tmp545 = element_matrix_tmp156*element_matrix_tmp525 + element_matrix_tmp160*element_matrix_tmp61 + element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp529*element_matrix_tmp64 - eta_s*(-element_matrix_tmp530*element_matrix_tmp61 + element_matrix_tmp531*element_matrix_tmp72)) + element_matrix_tmp275;
    const s_t element_matrix_tmp546 = element_matrix_tmp156*element_matrix_tmp517 + element_matrix_tmp160*element_matrix_tmp24 + element_matrix_tmp23*(-element_matrix_tmp391*u1_grad_2 + ((s_t(1) / s_t(3)))*element_matrix_tmp523*element_matrix_tmp64 - eta_s*(-element_matrix_tmp518*element_matrix_tmp61 + element_matrix_tmp519*element_matrix_tmp72 - element_matrix_tmp532)) + element_matrix_tmp319;
    const s_t element_matrix_tmp547 = element_matrix_tmp173*element_matrix_tmp74;
    const s_t element_matrix_tmp548 = element_matrix_tmp391*u1_grad_0;
    const s_t element_matrix_tmp549 = element_matrix_tmp156*element_matrix_tmp534 + element_matrix_tmp160*element_matrix_tmp535 + element_matrix_tmp23*(element_matrix_tmp400*element_matrix_tmp539 + element_matrix_tmp548 - eta_s*(-element_matrix_tmp540*element_matrix_tmp61 + element_matrix_tmp541*element_matrix_tmp72 + element_matrix_tmp547)) + element_matrix_tmp306 + mu*(s_t(2)*element_matrix_tmp223*u0_grad_1 + element_matrix_tmp305);
    const s_t element_matrix_tmp550 = basis0_grad0*element_matrix_tmp546 + basis0_grad1*element_matrix_tmp545 + basis0_grad2*element_matrix_tmp549;
    const s_t element_matrix_tmp551 = element_matrix_tmp180*element_matrix_tmp534 + element_matrix_tmp181*element_matrix_tmp535 + element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp539*element_matrix_tmp66 - eta_s*(element_matrix_tmp540*element_matrix_tmp67 - element_matrix_tmp541*element_matrix_tmp71)) + element_matrix_tmp299;
    const s_t element_matrix_tmp552 = element_matrix_tmp180*element_matrix_tmp525 + element_matrix_tmp181*element_matrix_tmp61 + element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp529*element_matrix_tmp66 - element_matrix_tmp548 - eta_s*(element_matrix_tmp530*element_matrix_tmp67 - element_matrix_tmp531*element_matrix_tmp71 - element_matrix_tmp547)) + element_matrix_tmp286;
    const s_t element_matrix_tmp553 = element_matrix_tmp180*element_matrix_tmp517 + element_matrix_tmp181*element_matrix_tmp24 + element_matrix_tmp23*(element_matrix_tmp391*element_matrix_tmp5 + element_matrix_tmp393*element_matrix_tmp523 - eta_s*(element_matrix_tmp518*element_matrix_tmp67 - element_matrix_tmp519*element_matrix_tmp71 + element_matrix_tmp542)) + element_matrix_tmp322;
    const s_t element_matrix_tmp554 = basis0_grad0*element_matrix_tmp553 + basis0_grad1*element_matrix_tmp552 + basis0_grad2*element_matrix_tmp551;
    const s_t element_matrix_tmp555 = element_matrix_tmp34 + element_matrix_tmp416;
    const s_t element_matrix_tmp556 = element_matrix_tmp522 + element_matrix_tmp54*(-element_matrix_tmp25 - s_t(2)*element_matrix_tmp418 + s_t(2)*element_matrix_tmp45*u0_grad_2 - element_matrix_tmp521);
    const s_t element_matrix_tmp557 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp33*element_matrix_tmp556 - eta_s*(element_matrix_tmp19*element_matrix_tmp519 - element_matrix_tmp24*element_matrix_tmp555)) + element_matrix_tmp24*element_matrix_tmp262 + element_matrix_tmp261*element_matrix_tmp517 + element_matrix_tmp453;
    const s_t element_matrix_tmp558 = element_matrix_tmp528 + element_matrix_tmp54*(-element_matrix_tmp109 + s_t(2)*element_matrix_tmp173*element_matrix_tmp47 - s_t(2)*element_matrix_tmp407 - element_matrix_tmp527);
    const s_t element_matrix_tmp559 = -element_matrix_tmp114 - element_matrix_tmp411;
    const s_t element_matrix_tmp560 = element_matrix_tmp74*u1_grad_2;
    const s_t element_matrix_tmp561 = element_matrix_tmp23*(-element_matrix_tmp242*u0_grad_2 + ((s_t(1) / s_t(3)))*element_matrix_tmp33*element_matrix_tmp558 - eta_s*(element_matrix_tmp19*element_matrix_tmp531 - element_matrix_tmp24*element_matrix_tmp559 - element_matrix_tmp560)) + element_matrix_tmp261*element_matrix_tmp525 + element_matrix_tmp262*element_matrix_tmp61 + element_matrix_tmp485;
    const s_t element_matrix_tmp562 = element_matrix_tmp538 + element_matrix_tmp54*(element_matrix_tmp144 + s_t(2)*element_matrix_tmp428 - s_t(2)*element_matrix_tmp429 + element_matrix_tmp537);
    const s_t element_matrix_tmp563 = element_matrix_tmp139 + element_matrix_tmp426;
    const s_t element_matrix_tmp564 = element_matrix_tmp5*element_matrix_tmp74;
    const s_t element_matrix_tmp565 = element_matrix_tmp242*u0_grad_1;
    const s_t element_matrix_tmp566 = element_matrix_tmp23*(element_matrix_tmp268*element_matrix_tmp562 + element_matrix_tmp565 - eta_s*(element_matrix_tmp19*element_matrix_tmp541 - element_matrix_tmp24*element_matrix_tmp563 + element_matrix_tmp564)) + element_matrix_tmp261*element_matrix_tmp534 + element_matrix_tmp262*element_matrix_tmp535 + element_matrix_tmp475 + mu*(element_matrix_tmp223*element_matrix_tmp93 + element_matrix_tmp474);
    const s_t element_matrix_tmp567 = basis0_grad0*element_matrix_tmp557 + basis0_grad1*element_matrix_tmp561 + basis0_grad2*element_matrix_tmp566;
    const s_t element_matrix_tmp568 = element_matrix_tmp194*element_matrix_tmp525 + element_matrix_tmp204*element_matrix_tmp61 + element_matrix_tmp23*(-element_matrix_tmp200*element_matrix_tmp558 + eta_s*(element_matrix_tmp531*element_matrix_tmp64 + element_matrix_tmp559*element_matrix_tmp61)) + element_matrix_tmp483;
    const s_t element_matrix_tmp569 = element_matrix_tmp194*element_matrix_tmp517 + element_matrix_tmp204*element_matrix_tmp24 + element_matrix_tmp23*(-element_matrix_tmp200*element_matrix_tmp556 - element_matrix_tmp203*u0_grad_2 + eta_s*(element_matrix_tmp519*element_matrix_tmp64 + element_matrix_tmp555*element_matrix_tmp61 - element_matrix_tmp560)) + element_matrix_tmp465 + mu*(s_t(4)*element_matrix_tmp126 + element_matrix_tmp463);
    const s_t element_matrix_tmp570 = element_matrix_tmp74*u1_grad_0;
    const s_t element_matrix_tmp571 = element_matrix_tmp194*element_matrix_tmp534 + element_matrix_tmp204*element_matrix_tmp535 + element_matrix_tmp23*(element_matrix_tmp173*element_matrix_tmp203 - element_matrix_tmp200*element_matrix_tmp562 + eta_s*(element_matrix_tmp541*element_matrix_tmp64 + element_matrix_tmp563*element_matrix_tmp61 + element_matrix_tmp570)) + element_matrix_tmp479 + mu*(s_t(2)*element_matrix_tmp223*element_matrix_tmp5 - element_matrix_tmp478);
    const s_t element_matrix_tmp572 = basis0_grad0*element_matrix_tmp569 + basis0_grad1*element_matrix_tmp568 + basis0_grad2*element_matrix_tmp571;
    const s_t element_matrix_tmp573 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp562*element_matrix_tmp71 - eta_s*(-element_matrix_tmp541*element_matrix_tmp66 + element_matrix_tmp563*element_matrix_tmp67)) + element_matrix_tmp239*element_matrix_tmp534 + element_matrix_tmp240*element_matrix_tmp535 + element_matrix_tmp472;
    const s_t element_matrix_tmp574 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp556*element_matrix_tmp71 - element_matrix_tmp565 - eta_s*(-element_matrix_tmp519*element_matrix_tmp66 + element_matrix_tmp555*element_matrix_tmp67 - element_matrix_tmp564)) + element_matrix_tmp239*element_matrix_tmp517 + element_matrix_tmp24*element_matrix_tmp240 + element_matrix_tmp460;
    const s_t element_matrix_tmp575 = element_matrix_tmp23*(element_matrix_tmp173*element_matrix_tmp242 + element_matrix_tmp252*element_matrix_tmp558 - eta_s*(-element_matrix_tmp531*element_matrix_tmp66 + element_matrix_tmp559*element_matrix_tmp67 + element_matrix_tmp570)) + element_matrix_tmp239*element_matrix_tmp525 + element_matrix_tmp240*element_matrix_tmp61 + element_matrix_tmp487;
    const s_t element_matrix_tmp576 = basis0_grad0*element_matrix_tmp574 + basis0_grad1*element_matrix_tmp575 + basis0_grad2*element_matrix_tmp573;
    const s_t element_matrix_tmp577 = element_matrix_tmp435 + element_matrix_tmp6;
    const s_t element_matrix_tmp578 = element_matrix_tmp157 + element_matrix_tmp405;
    const s_t element_matrix_tmp579 = element_matrix_tmp538 + element_matrix_tmp54*(-s_t(2)*element_matrix_tmp144 - element_matrix_tmp147 - element_matrix_tmp430);
    const s_t element_matrix_tmp580 = element_matrix_tmp23*(-element_matrix_tmp301*element_matrix_tmp579 + eta_s*(element_matrix_tmp540*element_matrix_tmp66 + element_matrix_tmp563*element_matrix_tmp71)) + element_matrix_tmp300*element_matrix_tmp534 + element_matrix_tmp303*element_matrix_tmp535 + mu*(element_matrix_tmp577 + element_matrix_tmp578 + s_t(2));
    const s_t element_matrix_tmp581 = element_matrix_tmp123 + element_matrix_tmp438;
    const s_t element_matrix_tmp582 = element_matrix_tmp522 + element_matrix_tmp54*(s_t(2)*element_matrix_tmp25 + element_matrix_tmp30 + element_matrix_tmp420);
    const s_t element_matrix_tmp583 = element_matrix_tmp23*(-element_matrix_tmp301*element_matrix_tmp582 + eta_s*(element_matrix_tmp151 - element_matrix_tmp440 + element_matrix_tmp518*element_matrix_tmp66 + element_matrix_tmp555*element_matrix_tmp71)) + element_matrix_tmp24*element_matrix_tmp303 + element_matrix_tmp300*element_matrix_tmp517 - element_matrix_tmp581*mu;
    const s_t element_matrix_tmp584 = element_matrix_tmp163 + element_matrix_tmp424;
    const s_t element_matrix_tmp585 = element_matrix_tmp528 + element_matrix_tmp54*(s_t(2)*element_matrix_tmp109 + element_matrix_tmp112 + element_matrix_tmp409);
    const s_t element_matrix_tmp586 = element_matrix_tmp23*(-element_matrix_tmp301*element_matrix_tmp585 + eta_s*(-element_matrix_tmp167 + element_matrix_tmp432 + element_matrix_tmp530*element_matrix_tmp66 + element_matrix_tmp559*element_matrix_tmp71)) + element_matrix_tmp300*element_matrix_tmp525 + element_matrix_tmp303*element_matrix_tmp61 - element_matrix_tmp584*mu;
    const s_t element_matrix_tmp587 = basis0_grad0*element_matrix_tmp583 + basis0_grad1*element_matrix_tmp586 + basis0_grad2*element_matrix_tmp580;
    const s_t element_matrix_tmp588 = element_matrix_tmp0 + element_matrix_tmp403 + s_t(2);
    const s_t element_matrix_tmp589 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp24*element_matrix_tmp582 - eta_s*(element_matrix_tmp19*element_matrix_tmp518 - element_matrix_tmp33*element_matrix_tmp555)) + element_matrix_tmp24*element_matrix_tmp317 + element_matrix_tmp316*element_matrix_tmp517 + mu*(element_matrix_tmp577 + element_matrix_tmp588);
    const s_t element_matrix_tmp590 = mu*(-element_matrix_tmp414 - element_matrix_tmp94);
    const s_t element_matrix_tmp591 = element_matrix_tmp120 + element_matrix_tmp422;
    const s_t element_matrix_tmp592 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp24*element_matrix_tmp585 - eta_s*(element_matrix_tmp19*element_matrix_tmp530 - element_matrix_tmp33*element_matrix_tmp559 + element_matrix_tmp591)) + element_matrix_tmp316*element_matrix_tmp525 + element_matrix_tmp317*element_matrix_tmp61 + element_matrix_tmp590;
    const s_t element_matrix_tmp593 = s_t(4)*element_matrix_tmp1;
    const s_t element_matrix_tmp594 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp24*element_matrix_tmp579 - eta_s*(-element_matrix_tmp184 + element_matrix_tmp19*element_matrix_tmp540 - element_matrix_tmp33*element_matrix_tmp563 - element_matrix_tmp440)) + element_matrix_tmp316*element_matrix_tmp534 + element_matrix_tmp317*element_matrix_tmp535 + mu*(s_t(2)*element_matrix_tmp223*u2_grad_0 - element_matrix_tmp581 - element_matrix_tmp593*u2_grad_0);
    const s_t element_matrix_tmp595 = basis0_grad0*element_matrix_tmp589 + basis0_grad1*element_matrix_tmp592 + basis0_grad2*element_matrix_tmp594;
    const s_t element_matrix_tmp596 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp585*element_matrix_tmp61 - eta_s*(-element_matrix_tmp530*element_matrix_tmp64 + element_matrix_tmp559*element_matrix_tmp72)) + element_matrix_tmp276*element_matrix_tmp525 + element_matrix_tmp279*element_matrix_tmp61 + mu*(element_matrix_tmp578 + element_matrix_tmp588);
    const s_t element_matrix_tmp597 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp582*element_matrix_tmp61 - eta_s*(-element_matrix_tmp518*element_matrix_tmp64 + element_matrix_tmp555*element_matrix_tmp72 - element_matrix_tmp591)) + element_matrix_tmp24*element_matrix_tmp279 + element_matrix_tmp276*element_matrix_tmp517 + element_matrix_tmp590;
    const s_t element_matrix_tmp598 = element_matrix_tmp23*(((s_t(1) / s_t(3)))*element_matrix_tmp579*element_matrix_tmp61 - eta_s*(-element_matrix_tmp167 - element_matrix_tmp448 - element_matrix_tmp540*element_matrix_tmp64 + element_matrix_tmp563*element_matrix_tmp72)) + element_matrix_tmp276*element_matrix_tmp534 + element_matrix_tmp279*element_matrix_tmp535 + mu*(s_t(2)*element_matrix_tmp223*u2_grad_1 - element_matrix_tmp584 - element_matrix_tmp593*u2_grad_1);
    const s_t element_matrix_tmp599 = basis0_grad0*element_matrix_tmp597 + basis0_grad1*element_matrix_tmp596 + basis0_grad2*element_matrix_tmp598;
    const s_t element_matrix_tmp600 = basis1_grad0*element_matrix_tmp524 + basis1_grad1*element_matrix_tmp533 + basis1_grad2*element_matrix_tmp543;
    const s_t element_matrix_tmp601 = basis1_grad0*element_matrix_tmp546 + basis1_grad1*element_matrix_tmp545 + basis1_grad2*element_matrix_tmp549;
    const s_t element_matrix_tmp602 = basis1_grad0*element_matrix_tmp553 + basis1_grad1*element_matrix_tmp552 + basis1_grad2*element_matrix_tmp551;
    const s_t element_matrix_tmp603 = basis1_grad0*element_matrix_tmp557 + basis1_grad1*element_matrix_tmp561 + basis1_grad2*element_matrix_tmp566;
    const s_t element_matrix_tmp604 = basis1_grad0*element_matrix_tmp569 + basis1_grad1*element_matrix_tmp568 + basis1_grad2*element_matrix_tmp571;
    const s_t element_matrix_tmp605 = basis1_grad0*element_matrix_tmp574 + basis1_grad1*element_matrix_tmp575 + basis1_grad2*element_matrix_tmp573;
    const s_t element_matrix_tmp606 = basis1_grad0*element_matrix_tmp583 + basis1_grad1*element_matrix_tmp586 + basis1_grad2*element_matrix_tmp580;
    const s_t element_matrix_tmp607 = basis1_grad0*element_matrix_tmp589 + basis1_grad1*element_matrix_tmp592 + basis1_grad2*element_matrix_tmp594;
    const s_t element_matrix_tmp608 = basis1_grad0*element_matrix_tmp597 + basis1_grad1*element_matrix_tmp596 + basis1_grad2*element_matrix_tmp598;
    const s_t element_matrix_tmp609 = basis2_grad0*element_matrix_tmp524 + basis2_grad1*element_matrix_tmp533 + basis2_grad2*element_matrix_tmp543;
    const s_t element_matrix_tmp610 = basis2_grad0*element_matrix_tmp546 + basis2_grad1*element_matrix_tmp545 + basis2_grad2*element_matrix_tmp549;
    const s_t element_matrix_tmp611 = basis2_grad0*element_matrix_tmp553 + basis2_grad1*element_matrix_tmp552 + basis2_grad2*element_matrix_tmp551;
    const s_t element_matrix_tmp612 = basis2_grad0*element_matrix_tmp557 + basis2_grad1*element_matrix_tmp561 + basis2_grad2*element_matrix_tmp566;
    const s_t element_matrix_tmp613 = basis2_grad0*element_matrix_tmp569 + basis2_grad1*element_matrix_tmp568 + basis2_grad2*element_matrix_tmp571;
    const s_t element_matrix_tmp614 = basis2_grad0*element_matrix_tmp574 + basis2_grad1*element_matrix_tmp575 + basis2_grad2*element_matrix_tmp573;
    const s_t element_matrix_tmp615 = basis2_grad0*element_matrix_tmp583 + basis2_grad1*element_matrix_tmp586 + basis2_grad2*element_matrix_tmp580;
    const s_t element_matrix_tmp616 = basis2_grad0*element_matrix_tmp589 + basis2_grad1*element_matrix_tmp592 + basis2_grad2*element_matrix_tmp594;
    const s_t element_matrix_tmp617 = basis2_grad0*element_matrix_tmp597 + basis2_grad1*element_matrix_tmp596 + basis2_grad2*element_matrix_tmp598;
    const s_t element_matrix_tmp618 = basis3_grad0*element_matrix_tmp524 + basis3_grad1*element_matrix_tmp533 + basis3_grad2*element_matrix_tmp543;
    const s_t element_matrix_tmp619 = basis3_grad0*element_matrix_tmp546 + basis3_grad1*element_matrix_tmp545 + basis3_grad2*element_matrix_tmp549;
    const s_t element_matrix_tmp620 = basis3_grad0*element_matrix_tmp553 + basis3_grad1*element_matrix_tmp552 + basis3_grad2*element_matrix_tmp551;
    const s_t element_matrix_tmp621 = basis3_grad0*element_matrix_tmp557 + basis3_grad1*element_matrix_tmp561 + basis3_grad2*element_matrix_tmp566;
    const s_t element_matrix_tmp622 = basis3_grad0*element_matrix_tmp569 + basis3_grad1*element_matrix_tmp568 + basis3_grad2*element_matrix_tmp571;
    const s_t element_matrix_tmp623 = basis3_grad0*element_matrix_tmp574 + basis3_grad1*element_matrix_tmp575 + basis3_grad2*element_matrix_tmp573;
    const s_t element_matrix_tmp624 = basis3_grad0*element_matrix_tmp583 + basis3_grad1*element_matrix_tmp586 + basis3_grad2*element_matrix_tmp580;
    const s_t element_matrix_tmp625 = basis3_grad0*element_matrix_tmp589 + basis3_grad1*element_matrix_tmp592 + basis3_grad2*element_matrix_tmp594;
    const s_t element_matrix_tmp626 = basis3_grad0*element_matrix_tmp597 + basis3_grad1*element_matrix_tmp596 + basis3_grad2*element_matrix_tmp598;
    element_matrix[0] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp153 + basis0_grad1*element_matrix_tmp177 + basis0_grad2*element_matrix_tmp186);
    element_matrix[12] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp153 + basis1_grad1*element_matrix_tmp177 + basis1_grad2*element_matrix_tmp186);
    element_matrix[24] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp153 + basis2_grad1*element_matrix_tmp177 + basis2_grad2*element_matrix_tmp186);
    element_matrix[36] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp153 + basis3_grad1*element_matrix_tmp177 + basis3_grad2*element_matrix_tmp186);
    element_matrix[48] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp272 + basis0_grad1*element_matrix_tmp233 + basis0_grad2*element_matrix_tmp256);
    element_matrix[60] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp272 + basis1_grad1*element_matrix_tmp233 + basis1_grad2*element_matrix_tmp256);
    element_matrix[72] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp272 + basis2_grad1*element_matrix_tmp233 + basis2_grad2*element_matrix_tmp256);
    element_matrix[84] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp272 + basis3_grad1*element_matrix_tmp233 + basis3_grad2*element_matrix_tmp256);
    element_matrix[96] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp324 + basis0_grad1*element_matrix_tmp297 + basis0_grad2*element_matrix_tmp314);
    element_matrix[108] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp324 + basis1_grad1*element_matrix_tmp297 + basis1_grad2*element_matrix_tmp314);
    element_matrix[120] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp324 + basis2_grad1*element_matrix_tmp297 + basis2_grad2*element_matrix_tmp314);
    element_matrix[132] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp324 + basis3_grad1*element_matrix_tmp297 + basis3_grad2*element_matrix_tmp314);
    element_matrix[1] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp325 + basis0_grad1*element_matrix_tmp326 + basis0_grad2*element_matrix_tmp327);
    element_matrix[13] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp325 + basis1_grad1*element_matrix_tmp326 + basis1_grad2*element_matrix_tmp327);
    element_matrix[25] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp325 + basis2_grad1*element_matrix_tmp326 + basis2_grad2*element_matrix_tmp327);
    element_matrix[37] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp325 + basis3_grad1*element_matrix_tmp326 + basis3_grad2*element_matrix_tmp327);
    element_matrix[49] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp330 + basis0_grad1*element_matrix_tmp328 + basis0_grad2*element_matrix_tmp329);
    element_matrix[61] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp330 + basis1_grad1*element_matrix_tmp328 + basis1_grad2*element_matrix_tmp329);
    element_matrix[73] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp330 + basis2_grad1*element_matrix_tmp328 + basis2_grad2*element_matrix_tmp329);
    element_matrix[85] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp330 + basis3_grad1*element_matrix_tmp328 + basis3_grad2*element_matrix_tmp329);
    element_matrix[97] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp333 + basis0_grad1*element_matrix_tmp331 + basis0_grad2*element_matrix_tmp332);
    element_matrix[109] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp333 + basis1_grad1*element_matrix_tmp331 + basis1_grad2*element_matrix_tmp332);
    element_matrix[121] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp333 + basis2_grad1*element_matrix_tmp331 + basis2_grad2*element_matrix_tmp332);
    element_matrix[133] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp333 + basis3_grad1*element_matrix_tmp331 + basis3_grad2*element_matrix_tmp332);
    element_matrix[2] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp334 + basis0_grad1*element_matrix_tmp335 + basis0_grad2*element_matrix_tmp336);
    element_matrix[14] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp334 + basis1_grad1*element_matrix_tmp335 + basis1_grad2*element_matrix_tmp336);
    element_matrix[26] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp334 + basis2_grad1*element_matrix_tmp335 + basis2_grad2*element_matrix_tmp336);
    element_matrix[38] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp334 + basis3_grad1*element_matrix_tmp335 + basis3_grad2*element_matrix_tmp336);
    element_matrix[50] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp339 + basis0_grad1*element_matrix_tmp337 + basis0_grad2*element_matrix_tmp338);
    element_matrix[62] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp339 + basis1_grad1*element_matrix_tmp337 + basis1_grad2*element_matrix_tmp338);
    element_matrix[74] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp339 + basis2_grad1*element_matrix_tmp337 + basis2_grad2*element_matrix_tmp338);
    element_matrix[86] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp339 + basis3_grad1*element_matrix_tmp337 + basis3_grad2*element_matrix_tmp338);
    element_matrix[98] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp342 + basis0_grad1*element_matrix_tmp340 + basis0_grad2*element_matrix_tmp341);
    element_matrix[110] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp342 + basis1_grad1*element_matrix_tmp340 + basis1_grad2*element_matrix_tmp341);
    element_matrix[122] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp342 + basis2_grad1*element_matrix_tmp340 + basis2_grad2*element_matrix_tmp341);
    element_matrix[134] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp342 + basis3_grad1*element_matrix_tmp340 + basis3_grad2*element_matrix_tmp341);
    element_matrix[3] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp343 + basis0_grad1*element_matrix_tmp344 + basis0_grad2*element_matrix_tmp345);
    element_matrix[15] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp343 + basis1_grad1*element_matrix_tmp344 + basis1_grad2*element_matrix_tmp345);
    element_matrix[27] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp343 + basis2_grad1*element_matrix_tmp344 + basis2_grad2*element_matrix_tmp345);
    element_matrix[39] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp343 + basis3_grad1*element_matrix_tmp344 + basis3_grad2*element_matrix_tmp345);
    element_matrix[51] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp348 + basis0_grad1*element_matrix_tmp346 + basis0_grad2*element_matrix_tmp347);
    element_matrix[63] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp348 + basis1_grad1*element_matrix_tmp346 + basis1_grad2*element_matrix_tmp347);
    element_matrix[75] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp348 + basis2_grad1*element_matrix_tmp346 + basis2_grad2*element_matrix_tmp347);
    element_matrix[87] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp348 + basis3_grad1*element_matrix_tmp346 + basis3_grad2*element_matrix_tmp347);
    element_matrix[99] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp351 + basis0_grad1*element_matrix_tmp349 + basis0_grad2*element_matrix_tmp350);
    element_matrix[111] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp351 + basis1_grad1*element_matrix_tmp349 + basis1_grad2*element_matrix_tmp350);
    element_matrix[123] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp351 + basis2_grad1*element_matrix_tmp349 + basis2_grad2*element_matrix_tmp350);
    element_matrix[135] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp351 + basis3_grad1*element_matrix_tmp349 + basis3_grad2*element_matrix_tmp350);
    element_matrix[4] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp389 + basis0_grad1*element_matrix_tmp402 + basis0_grad2*element_matrix_tmp397);
    element_matrix[16] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp389 + basis1_grad1*element_matrix_tmp402 + basis1_grad2*element_matrix_tmp397);
    element_matrix[28] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp389 + basis2_grad1*element_matrix_tmp402 + basis2_grad2*element_matrix_tmp397);
    element_matrix[40] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp389 + basis3_grad1*element_matrix_tmp402 + basis3_grad2*element_matrix_tmp397);
    element_matrix[52] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp445 + basis0_grad1*element_matrix_tmp434 + basis0_grad2*element_matrix_tmp450);
    element_matrix[64] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp445 + basis1_grad1*element_matrix_tmp434 + basis1_grad2*element_matrix_tmp450);
    element_matrix[76] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp445 + basis2_grad1*element_matrix_tmp434 + basis2_grad2*element_matrix_tmp450);
    element_matrix[88] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp445 + basis3_grad1*element_matrix_tmp434 + basis3_grad2*element_matrix_tmp450);
    element_matrix[100] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp470 + basis0_grad1*element_matrix_tmp489 + basis0_grad2*element_matrix_tmp482);
    element_matrix[112] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp470 + basis1_grad1*element_matrix_tmp489 + basis1_grad2*element_matrix_tmp482);
    element_matrix[124] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp470 + basis2_grad1*element_matrix_tmp489 + basis2_grad2*element_matrix_tmp482);
    element_matrix[136] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp470 + basis3_grad1*element_matrix_tmp489 + basis3_grad2*element_matrix_tmp482);
    element_matrix[5] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp490 + basis0_grad1*element_matrix_tmp492 + basis0_grad2*element_matrix_tmp491);
    element_matrix[17] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp490 + basis1_grad1*element_matrix_tmp492 + basis1_grad2*element_matrix_tmp491);
    element_matrix[29] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp490 + basis2_grad1*element_matrix_tmp492 + basis2_grad2*element_matrix_tmp491);
    element_matrix[41] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp490 + basis3_grad1*element_matrix_tmp492 + basis3_grad2*element_matrix_tmp491);
    element_matrix[53] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp494 + basis0_grad1*element_matrix_tmp493 + basis0_grad2*element_matrix_tmp495);
    element_matrix[65] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp494 + basis1_grad1*element_matrix_tmp493 + basis1_grad2*element_matrix_tmp495);
    element_matrix[77] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp494 + basis2_grad1*element_matrix_tmp493 + basis2_grad2*element_matrix_tmp495);
    element_matrix[89] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp494 + basis3_grad1*element_matrix_tmp493 + basis3_grad2*element_matrix_tmp495);
    element_matrix[101] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp496 + basis0_grad1*element_matrix_tmp498 + basis0_grad2*element_matrix_tmp497);
    element_matrix[113] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp496 + basis1_grad1*element_matrix_tmp498 + basis1_grad2*element_matrix_tmp497);
    element_matrix[125] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp496 + basis2_grad1*element_matrix_tmp498 + basis2_grad2*element_matrix_tmp497);
    element_matrix[137] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp496 + basis3_grad1*element_matrix_tmp498 + basis3_grad2*element_matrix_tmp497);
    element_matrix[6] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp499 + basis0_grad1*element_matrix_tmp501 + basis0_grad2*element_matrix_tmp500);
    element_matrix[18] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp499 + basis1_grad1*element_matrix_tmp501 + basis1_grad2*element_matrix_tmp500);
    element_matrix[30] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp499 + basis2_grad1*element_matrix_tmp501 + basis2_grad2*element_matrix_tmp500);
    element_matrix[42] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp499 + basis3_grad1*element_matrix_tmp501 + basis3_grad2*element_matrix_tmp500);
    element_matrix[54] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp503 + basis0_grad1*element_matrix_tmp502 + basis0_grad2*element_matrix_tmp504);
    element_matrix[66] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp503 + basis1_grad1*element_matrix_tmp502 + basis1_grad2*element_matrix_tmp504);
    element_matrix[78] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp503 + basis2_grad1*element_matrix_tmp502 + basis2_grad2*element_matrix_tmp504);
    element_matrix[90] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp503 + basis3_grad1*element_matrix_tmp502 + basis3_grad2*element_matrix_tmp504);
    element_matrix[102] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp505 + basis0_grad1*element_matrix_tmp507 + basis0_grad2*element_matrix_tmp506);
    element_matrix[114] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp505 + basis1_grad1*element_matrix_tmp507 + basis1_grad2*element_matrix_tmp506);
    element_matrix[126] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp505 + basis2_grad1*element_matrix_tmp507 + basis2_grad2*element_matrix_tmp506);
    element_matrix[138] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp505 + basis3_grad1*element_matrix_tmp507 + basis3_grad2*element_matrix_tmp506);
    element_matrix[7] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp508 + basis0_grad1*element_matrix_tmp510 + basis0_grad2*element_matrix_tmp509);
    element_matrix[19] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp508 + basis1_grad1*element_matrix_tmp510 + basis1_grad2*element_matrix_tmp509);
    element_matrix[31] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp508 + basis2_grad1*element_matrix_tmp510 + basis2_grad2*element_matrix_tmp509);
    element_matrix[43] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp508 + basis3_grad1*element_matrix_tmp510 + basis3_grad2*element_matrix_tmp509);
    element_matrix[55] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp512 + basis0_grad1*element_matrix_tmp511 + basis0_grad2*element_matrix_tmp513);
    element_matrix[67] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp512 + basis1_grad1*element_matrix_tmp511 + basis1_grad2*element_matrix_tmp513);
    element_matrix[79] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp512 + basis2_grad1*element_matrix_tmp511 + basis2_grad2*element_matrix_tmp513);
    element_matrix[91] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp512 + basis3_grad1*element_matrix_tmp511 + basis3_grad2*element_matrix_tmp513);
    element_matrix[103] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp514 + basis0_grad1*element_matrix_tmp516 + basis0_grad2*element_matrix_tmp515);
    element_matrix[115] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp514 + basis1_grad1*element_matrix_tmp516 + basis1_grad2*element_matrix_tmp515);
    element_matrix[127] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp514 + basis2_grad1*element_matrix_tmp516 + basis2_grad2*element_matrix_tmp515);
    element_matrix[139] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp514 + basis3_grad1*element_matrix_tmp516 + basis3_grad2*element_matrix_tmp515);
    element_matrix[8] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp544 + basis0_grad1*element_matrix_tmp550 + basis0_grad2*element_matrix_tmp554);
    element_matrix[20] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp544 + basis1_grad1*element_matrix_tmp550 + basis1_grad2*element_matrix_tmp554);
    element_matrix[32] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp544 + basis2_grad1*element_matrix_tmp550 + basis2_grad2*element_matrix_tmp554);
    element_matrix[44] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp544 + basis3_grad1*element_matrix_tmp550 + basis3_grad2*element_matrix_tmp554);
    element_matrix[56] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp567 + basis0_grad1*element_matrix_tmp572 + basis0_grad2*element_matrix_tmp576);
    element_matrix[68] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp567 + basis1_grad1*element_matrix_tmp572 + basis1_grad2*element_matrix_tmp576);
    element_matrix[80] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp567 + basis2_grad1*element_matrix_tmp572 + basis2_grad2*element_matrix_tmp576);
    element_matrix[92] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp567 + basis3_grad1*element_matrix_tmp572 + basis3_grad2*element_matrix_tmp576);
    element_matrix[104] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp595 + basis0_grad1*element_matrix_tmp599 + basis0_grad2*element_matrix_tmp587);
    element_matrix[116] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp595 + basis1_grad1*element_matrix_tmp599 + basis1_grad2*element_matrix_tmp587);
    element_matrix[128] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp595 + basis2_grad1*element_matrix_tmp599 + basis2_grad2*element_matrix_tmp587);
    element_matrix[140] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp595 + basis3_grad1*element_matrix_tmp599 + basis3_grad2*element_matrix_tmp587);
    element_matrix[9] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp600 + basis0_grad1*element_matrix_tmp601 + basis0_grad2*element_matrix_tmp602);
    element_matrix[21] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp600 + basis1_grad1*element_matrix_tmp601 + basis1_grad2*element_matrix_tmp602);
    element_matrix[33] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp600 + basis2_grad1*element_matrix_tmp601 + basis2_grad2*element_matrix_tmp602);
    element_matrix[45] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp600 + basis3_grad1*element_matrix_tmp601 + basis3_grad2*element_matrix_tmp602);
    element_matrix[57] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp603 + basis0_grad1*element_matrix_tmp604 + basis0_grad2*element_matrix_tmp605);
    element_matrix[69] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp603 + basis1_grad1*element_matrix_tmp604 + basis1_grad2*element_matrix_tmp605);
    element_matrix[81] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp603 + basis2_grad1*element_matrix_tmp604 + basis2_grad2*element_matrix_tmp605);
    element_matrix[93] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp603 + basis3_grad1*element_matrix_tmp604 + basis3_grad2*element_matrix_tmp605);
    element_matrix[105] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp607 + basis0_grad1*element_matrix_tmp608 + basis0_grad2*element_matrix_tmp606);
    element_matrix[117] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp607 + basis1_grad1*element_matrix_tmp608 + basis1_grad2*element_matrix_tmp606);
    element_matrix[129] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp607 + basis2_grad1*element_matrix_tmp608 + basis2_grad2*element_matrix_tmp606);
    element_matrix[141] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp607 + basis3_grad1*element_matrix_tmp608 + basis3_grad2*element_matrix_tmp606);
    element_matrix[10] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp609 + basis0_grad1*element_matrix_tmp610 + basis0_grad2*element_matrix_tmp611);
    element_matrix[22] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp609 + basis1_grad1*element_matrix_tmp610 + basis1_grad2*element_matrix_tmp611);
    element_matrix[34] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp609 + basis2_grad1*element_matrix_tmp610 + basis2_grad2*element_matrix_tmp611);
    element_matrix[46] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp609 + basis3_grad1*element_matrix_tmp610 + basis3_grad2*element_matrix_tmp611);
    element_matrix[58] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp612 + basis0_grad1*element_matrix_tmp613 + basis0_grad2*element_matrix_tmp614);
    element_matrix[70] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp612 + basis1_grad1*element_matrix_tmp613 + basis1_grad2*element_matrix_tmp614);
    element_matrix[82] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp612 + basis2_grad1*element_matrix_tmp613 + basis2_grad2*element_matrix_tmp614);
    element_matrix[94] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp612 + basis3_grad1*element_matrix_tmp613 + basis3_grad2*element_matrix_tmp614);
    element_matrix[106] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp616 + basis0_grad1*element_matrix_tmp617 + basis0_grad2*element_matrix_tmp615);
    element_matrix[118] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp616 + basis1_grad1*element_matrix_tmp617 + basis1_grad2*element_matrix_tmp615);
    element_matrix[130] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp616 + basis2_grad1*element_matrix_tmp617 + basis2_grad2*element_matrix_tmp615);
    element_matrix[142] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp616 + basis3_grad1*element_matrix_tmp617 + basis3_grad2*element_matrix_tmp615);
    element_matrix[11] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp618 + basis0_grad1*element_matrix_tmp619 + basis0_grad2*element_matrix_tmp620);
    element_matrix[23] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp618 + basis1_grad1*element_matrix_tmp619 + basis1_grad2*element_matrix_tmp620);
    element_matrix[35] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp618 + basis2_grad1*element_matrix_tmp619 + basis2_grad2*element_matrix_tmp620);
    element_matrix[47] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp618 + basis3_grad1*element_matrix_tmp619 + basis3_grad2*element_matrix_tmp620);
    element_matrix[59] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp621 + basis0_grad1*element_matrix_tmp622 + basis0_grad2*element_matrix_tmp623);
    element_matrix[71] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp621 + basis1_grad1*element_matrix_tmp622 + basis1_grad2*element_matrix_tmp623);
    element_matrix[83] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp621 + basis2_grad1*element_matrix_tmp622 + basis2_grad2*element_matrix_tmp623);
    element_matrix[95] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp621 + basis3_grad1*element_matrix_tmp622 + basis3_grad2*element_matrix_tmp623);
    element_matrix[107] = element_matrix_tmp187*(basis0_grad0*element_matrix_tmp625 + basis0_grad1*element_matrix_tmp626 + basis0_grad2*element_matrix_tmp624);
    element_matrix[119] = element_matrix_tmp187*(basis1_grad0*element_matrix_tmp625 + basis1_grad1*element_matrix_tmp626 + basis1_grad2*element_matrix_tmp624);
    element_matrix[131] = element_matrix_tmp187*(basis2_grad0*element_matrix_tmp625 + basis2_grad1*element_matrix_tmp626 + basis2_grad2*element_matrix_tmp624);
    element_matrix[143] = element_matrix_tmp187*(basis3_grad0*element_matrix_tmp625 + basis3_grad1*element_matrix_tmp626 + basis3_grad2*element_matrix_tmp624);
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_total_d3_simplex_hessian_block(
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
  s_t tangent[81][VS];
  for (int q = 0; q < NQ; ++q) {
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
      const s_t tangent_tmp0 = s_t(2)*pow_2(u1_grad_2);
      const s_t tangent_tmp1 = u2_grad_2 + s_t(1);
      const s_t tangent_tmp2 = s_t(2)*pow_2(tangent_tmp1) + s_t(2);
      const s_t tangent_tmp3 = tangent_tmp0 + tangent_tmp2;
      const s_t tangent_tmp4 = s_t(2)*pow_2(u2_grad_1);
      const s_t tangent_tmp5 = u1_grad_1 + s_t(1);
      const s_t tangent_tmp6 = s_t(2)*pow_2(tangent_tmp5);
      const s_t tangent_tmp7 = tangent_tmp4 + tangent_tmp6;
      const s_t tangent_tmp8 = u1_grad_2*u2_grad_1;
      const s_t tangent_tmp9 = s_t(2)*tangent_tmp8;
      const s_t tangent_tmp10 = -s_t(2)*tangent_tmp1*tangent_tmp5;
      const s_t tangent_tmp11 = -tangent_tmp10 - tangent_tmp9;
      const s_t tangent_tmp12 = ((s_t(1) / s_t(2)))*lmbda;
      const s_t tangent_tmp13 = tangent_tmp12*(tangent_tmp1*tangent_tmp5 - tangent_tmp8);
      const s_t tangent_tmp14 = u0_grad_0*u1_grad_1;
      const s_t tangent_tmp15 = u0_grad_1*u1_grad_2;
      const s_t tangent_tmp16 = u0_grad_2*u2_grad_1;
      const s_t tangent_tmp17 = u0_grad_1*u1_grad_0;
      const s_t tangent_tmp18 = u0_grad_2*u2_grad_0;
      const s_t tangent_tmp19 = tangent_tmp5 - tangent_tmp8 + u1_grad_1*u2_grad_2 + u2_grad_2;
      const s_t tangent_tmp20 = -tangent_tmp18 + u0_grad_0*u2_grad_2 + u0_grad_0;
      const s_t tangent_tmp21 = tangent_tmp14 - tangent_tmp17;
      const s_t tangent_tmp22 = tangent_tmp14*u2_grad_2 + tangent_tmp15*u2_grad_0 + tangent_tmp16*u1_grad_0 - tangent_tmp17*u2_grad_2 - tangent_tmp18*u1_grad_1 + tangent_tmp19 + tangent_tmp20 + tangent_tmp21 - tangent_tmp8*u0_grad_0;
      const s_t tangent_tmp23 = pow_m1(tangent_tmp22);
      const s_t tangent_tmp24 = -tangent_tmp15 + u0_grad_2*u1_grad_1 + u0_grad_2;
      const s_t tangent_tmp25 = tangent_tmp24*u_dt_shift;
      const s_t tangent_tmp26 = u0_grad_1*u_dt_shift + u0_old_grad_1;
      const s_t tangent_tmp27 = tangent_tmp26*u1_grad_2;
      const s_t tangent_tmp28 = u0_grad_2*u_dt_shift + u0_old_grad_2;
      const s_t tangent_tmp29 = tangent_tmp28*tangent_tmp5;
      const s_t tangent_tmp30 = tangent_tmp27 - tangent_tmp29;
      const s_t tangent_tmp31 = tangent_tmp25 + tangent_tmp30;
      const s_t tangent_tmp32 = -tangent_tmp16;
      const s_t tangent_tmp33 = tangent_tmp32 + u0_grad_1*u2_grad_2 + u0_grad_1;
      const s_t tangent_tmp34 = tangent_tmp33*u_dt_shift;
      const s_t tangent_tmp35 = tangent_tmp28*u2_grad_1;
      const s_t tangent_tmp36 = tangent_tmp1*tangent_tmp26;
      const s_t tangent_tmp37 = tangent_tmp35 - tangent_tmp36;
      const s_t tangent_tmp38 = tangent_tmp34 + tangent_tmp37;
      const s_t tangent_tmp39 = tangent_tmp19*u_dt_shift;
      const s_t tangent_tmp40 = u2_grad_2*u_dt_shift + u2_old_grad_2;
      const s_t tangent_tmp41 = tangent_tmp40*tangent_tmp5;
      const s_t tangent_tmp42 = u2_grad_1*u_dt_shift + u2_old_grad_1;
      const s_t tangent_tmp43 = tangent_tmp42*u1_grad_2;
      const s_t tangent_tmp44 = tangent_tmp39 + tangent_tmp41 - tangent_tmp43;
      const s_t tangent_tmp45 = u1_grad_1*u_dt_shift + u1_old_grad_1;
      const s_t tangent_tmp46 = tangent_tmp1*tangent_tmp45;
      const s_t tangent_tmp47 = u1_grad_2*u_dt_shift + u1_old_grad_2;
      const s_t tangent_tmp48 = tangent_tmp47*u2_grad_1;
      const s_t tangent_tmp49 = tangent_tmp46 - tangent_tmp48;
      const s_t tangent_tmp50 = s_t(3)*eta_b;
      const s_t tangent_tmp51 = tangent_tmp50*(-tangent_tmp44 - tangent_tmp49);
      const s_t tangent_tmp52 = -tangent_tmp46 + tangent_tmp48;
      const s_t tangent_tmp53 = -tangent_tmp41 + tangent_tmp43;
      const s_t tangent_tmp54 = s_t(2)*eta_s;
      const s_t tangent_tmp55 = tangent_tmp51 + tangent_tmp54*(-s_t(2)*tangent_tmp39 - tangent_tmp52 - tangent_tmp53);
      const s_t tangent_tmp56 = ((s_t(1) / s_t(3)))*tangent_tmp19;
      const s_t tangent_tmp57 = u0_grad_0*u_dt_shift + u0_old_grad_0;
      const s_t tangent_tmp58 = u0_grad_2*u1_grad_0;
      const s_t tangent_tmp59 = -tangent_tmp58 + u0_grad_0*u1_grad_2 + u1_grad_2;
      const s_t tangent_tmp60 = u1_grad_2*u2_grad_0;
      const s_t tangent_tmp61 = -tangent_tmp60;
      const s_t tangent_tmp62 = tangent_tmp61 + u1_grad_0*u2_grad_2 + u1_grad_0;
      const s_t tangent_tmp63 = u1_grad_0*u2_grad_1;
      const s_t tangent_tmp64 = -tangent_tmp63 + u1_grad_1*u2_grad_0 + u2_grad_0;
      const s_t tangent_tmp65 = tangent_tmp21 + tangent_tmp5 + u0_grad_0;
      const s_t tangent_tmp66 = u2_grad_0*u_dt_shift + u2_old_grad_0;
      const s_t tangent_tmp67 = -tangent_tmp19*tangent_tmp66 + tangent_tmp24*tangent_tmp57 + tangent_tmp26*tangent_tmp59 - tangent_tmp28*tangent_tmp65 + tangent_tmp40*tangent_tmp64 + tangent_tmp42*tangent_tmp62;
      const s_t tangent_tmp68 = u0_grad_1*u2_grad_0;
      const s_t tangent_tmp69 = -tangent_tmp68 + u0_grad_0*u2_grad_1 + u2_grad_1;
      const s_t tangent_tmp70 = tangent_tmp1 + tangent_tmp20;
      const s_t tangent_tmp71 = u1_grad_0*u_dt_shift + u1_old_grad_0;
      const s_t tangent_tmp72 = -tangent_tmp19*tangent_tmp71 - tangent_tmp26*tangent_tmp70 + tangent_tmp28*tangent_tmp69 + tangent_tmp33*tangent_tmp57 + tangent_tmp45*tangent_tmp62 + tangent_tmp47*tangent_tmp64;
      const s_t tangent_tmp73 = tangent_tmp33*tangent_tmp71;
      const s_t tangent_tmp74 = tangent_tmp47*tangent_tmp69;
      const s_t tangent_tmp75 = tangent_tmp24*tangent_tmp66;
      const s_t tangent_tmp76 = tangent_tmp42*tangent_tmp59;
      const s_t tangent_tmp77 = tangent_tmp45*tangent_tmp70;
      const s_t tangent_tmp78 = -tangent_tmp77;
      const s_t tangent_tmp79 = tangent_tmp40*tangent_tmp65;
      const s_t tangent_tmp80 = -tangent_tmp79;
      const s_t tangent_tmp81 = tangent_tmp73 + tangent_tmp74 + tangent_tmp75 + tangent_tmp76 + tangent_tmp78 + tangent_tmp80;
      const s_t tangent_tmp82 = tangent_tmp19*tangent_tmp57;
      const s_t tangent_tmp83 = tangent_tmp26*tangent_tmp62 + tangent_tmp28*tangent_tmp64 - tangent_tmp82;
      const s_t tangent_tmp84 = tangent_tmp50*(tangent_tmp81 + tangent_tmp83);
      const s_t tangent_tmp85 = tangent_tmp54*(s_t(2)*tangent_tmp26*tangent_tmp62 + s_t(2)*tangent_tmp28*tangent_tmp64 - tangent_tmp81 - s_t(2)*tangent_tmp82) + tangent_tmp84;
      const s_t tangent_tmp86 = -tangent_tmp85;
      const s_t tangent_tmp87 = eta_s*(tangent_tmp24*tangent_tmp67 + tangent_tmp33*tangent_tmp72) + tangent_tmp56*tangent_tmp86;
      const s_t tangent_tmp88 = pow_m2(tangent_tmp22);
      const s_t tangent_tmp89 = -tangent_tmp19*tangent_tmp88;
      const s_t tangent_tmp90 = -s_t(2)*tangent_tmp60;
      const s_t tangent_tmp91 = tangent_tmp1*u1_grad_0;
      const s_t tangent_tmp92 = s_t(2)*tangent_tmp91;
      const s_t tangent_tmp93 = -tangent_tmp90 - tangent_tmp92;
      const s_t tangent_tmp94 = s_t(2)*u0_grad_0 + s_t(2);
      const s_t tangent_tmp95 = u0_grad_0 + s_t(1);
      const s_t tangent_tmp96 = s_t(4)*tangent_tmp95;
      const s_t tangent_tmp97 = s_t(2)*u2_grad_0;
      const s_t tangent_tmp98 = tangent_tmp97*u2_grad_1;
      const s_t tangent_tmp99 = s_t(2)*u1_grad_0;
      const s_t tangent_tmp100 = tangent_tmp5*tangent_tmp99;
      const s_t tangent_tmp101 = tangent_tmp100 + tangent_tmp98;
      const s_t tangent_tmp102 = tangent_tmp67*u1_grad_2;
      const s_t tangent_tmp103 = -tangent_tmp1*tangent_tmp72;
      const s_t tangent_tmp104 = -eta_s*(-tangent_tmp59*tangent_tmp67 + tangent_tmp70*tangent_tmp72) + ((s_t(1) / s_t(3)))*tangent_tmp62*tangent_tmp85;
      const s_t tangent_tmp105 = s_t(2)*tangent_tmp63;
      const s_t tangent_tmp106 = tangent_tmp5*u2_grad_0;
      const s_t tangent_tmp107 = s_t(2)*tangent_tmp106;
      const s_t tangent_tmp108 = tangent_tmp105 - tangent_tmp107;
      const s_t tangent_tmp109 = tangent_tmp99*u1_grad_2;
      const s_t tangent_tmp110 = tangent_tmp1*tangent_tmp97;
      const s_t tangent_tmp111 = tangent_tmp109 + tangent_tmp110;
      const s_t tangent_tmp112 = tangent_tmp72*u2_grad_1;
      const s_t tangent_tmp113 = -tangent_tmp5*tangent_tmp67;
      const s_t tangent_tmp114 = -eta_s*(tangent_tmp65*tangent_tmp67 - tangent_tmp69*tangent_tmp72) + ((s_t(1) / s_t(3)))*tangent_tmp64*tangent_tmp85;
      const s_t tangent_tmp115 = s_t(2)*u0_grad_2;
      const s_t tangent_tmp116 = tangent_tmp115*u1_grad_2;
      const s_t tangent_tmp117 = s_t(2)*u0_grad_1;
      const s_t tangent_tmp118 = tangent_tmp117*tangent_tmp5;
      const s_t tangent_tmp119 = mu*(-tangent_tmp116 - tangent_tmp118);
      const s_t tangent_tmp120 = -s_t(2)*tangent_tmp16;
      const s_t tangent_tmp121 = tangent_tmp1*u0_grad_1;
      const s_t tangent_tmp122 = s_t(2)*tangent_tmp121;
      const s_t tangent_tmp123 = -tangent_tmp120 - tangent_tmp122;
      const s_t tangent_tmp124 = tangent_tmp40*u2_grad_1;
      const s_t tangent_tmp125 = tangent_tmp1*tangent_tmp42;
      const s_t tangent_tmp126 = tangent_tmp45*u1_grad_2 - tangent_tmp47*tangent_tmp5;
      const s_t tangent_tmp127 = tangent_tmp124 - tangent_tmp125 + tangent_tmp126;
      const s_t tangent_tmp128 = tangent_tmp51 + tangent_tmp54*(tangent_tmp44 - s_t(2)*tangent_tmp46 + s_t(2)*tangent_tmp48);
      const s_t tangent_tmp129 = tangent_tmp24*tangent_tmp71 + tangent_tmp33*tangent_tmp66 + tangent_tmp40*tangent_tmp69 - tangent_tmp42*tangent_tmp70 + tangent_tmp45*tangent_tmp59 - tangent_tmp47*tangent_tmp65;
      const s_t tangent_tmp130 = tangent_tmp54*(s_t(2)*tangent_tmp33*tangent_tmp71 + s_t(2)*tangent_tmp47*tangent_tmp69 - tangent_tmp75 - tangent_tmp76 - s_t(2)*tangent_tmp77 - tangent_tmp80 - tangent_tmp83) + tangent_tmp84;
      const s_t tangent_tmp131 = -eta_s*(-tangent_tmp129*tangent_tmp24 + tangent_tmp19*tangent_tmp72) + ((s_t(1) / s_t(3)))*tangent_tmp130*tangent_tmp33;
      const s_t tangent_tmp132 = s_t(2)*tangent_tmp17;
      const s_t tangent_tmp133 = s_t(6)*u2_grad_2 + s_t(6);
      const s_t tangent_tmp134 = tangent_tmp132 + tangent_tmp133;
      const s_t tangent_tmp135 = s_t(2)*tangent_tmp18;
      const s_t tangent_tmp136 = -s_t(2)*tangent_tmp1*tangent_tmp95;
      const s_t tangent_tmp137 = -tangent_tmp135 - tangent_tmp136;
      const s_t tangent_tmp138 = s_t(2)*u2_grad_2 + s_t(2);
      const s_t tangent_tmp139 = lmbda*(-tangent_tmp1*tangent_tmp17 + tangent_tmp1*tangent_tmp5*tangent_tmp95 - tangent_tmp18*tangent_tmp5 - tangent_tmp8*tangent_tmp95 + u0_grad_1*u1_grad_2*u2_grad_0 + u0_grad_2*u1_grad_0*u2_grad_1 + s_t(-1));
      const s_t tangent_tmp140 = ((s_t(1) / s_t(2)))*tangent_tmp139;
      const s_t tangent_tmp141 = tangent_tmp138*tangent_tmp140;
      const s_t tangent_tmp142 = -(s_t(1) / s_t(3))*tangent_tmp130;
      const s_t tangent_tmp143 = eta_s*(tangent_tmp129*tangent_tmp59 + tangent_tmp62*tangent_tmp72) + tangent_tmp142*tangent_tmp70;
      const s_t tangent_tmp144 = ((s_t(1) / s_t(3)))*tangent_tmp70;
      const s_t tangent_tmp145 = tangent_tmp129*u1_grad_2;
      const s_t tangent_tmp146 = s_t(2)*tangent_tmp94;
      const s_t tangent_tmp147 = s_t(6)*u2_grad_1;
      const s_t tangent_tmp148 = s_t(2)*tangent_tmp58;
      const s_t tangent_tmp149 = tangent_tmp147 - tangent_tmp148;
      const s_t tangent_tmp150 = s_t(2)*tangent_tmp68;
      const s_t tangent_tmp151 = tangent_tmp95*u2_grad_1;
      const s_t tangent_tmp152 = s_t(2)*tangent_tmp151;
      const s_t tangent_tmp153 = tangent_tmp150 - tangent_tmp152;
      const s_t tangent_tmp154 = tangent_tmp139*u2_grad_1;
      const s_t tangent_tmp155 = -tangent_tmp154;
      const s_t tangent_tmp156 = -eta_s*(tangent_tmp129*tangent_tmp65 - tangent_tmp64*tangent_tmp72) + ((s_t(1) / s_t(3)))*tangent_tmp130*tangent_tmp69;
      const s_t tangent_tmp157 = ((s_t(1) / s_t(3)))*tangent_tmp69;
      const s_t tangent_tmp158 = tangent_tmp129*tangent_tmp5;
      const s_t tangent_tmp159 = ((s_t(1) / s_t(3)))*tangent_tmp130;
      const s_t tangent_tmp160 = tangent_tmp159*u2_grad_1;
      const s_t tangent_tmp161 = tangent_tmp117*u2_grad_1;
      const s_t tangent_tmp162 = tangent_tmp1*tangent_tmp115;
      const s_t tangent_tmp163 = mu*(-tangent_tmp161 - tangent_tmp162);
      const s_t tangent_tmp164 = s_t(2)*tangent_tmp15;
      const s_t tangent_tmp165 = tangent_tmp5*u0_grad_2;
      const s_t tangent_tmp166 = s_t(2)*tangent_tmp165;
      const s_t tangent_tmp167 = tangent_tmp164 - tangent_tmp166;
      const s_t tangent_tmp168 = tangent_tmp51 + tangent_tmp54*(tangent_tmp39 - s_t(2)*tangent_tmp41 + s_t(2)*tangent_tmp43 + tangent_tmp49);
      const s_t tangent_tmp169 = tangent_tmp54*(s_t(2)*tangent_tmp24*tangent_tmp66 + s_t(2)*tangent_tmp42*tangent_tmp59 - tangent_tmp73 - tangent_tmp74 - tangent_tmp78 - s_t(2)*tangent_tmp79 - tangent_tmp83) + tangent_tmp84;
      const s_t tangent_tmp170 = -eta_s*(-tangent_tmp129*tangent_tmp33 + tangent_tmp19*tangent_tmp67) + ((s_t(1) / s_t(3)))*tangent_tmp169*tangent_tmp24;
      const s_t tangent_tmp171 = s_t(6)*u1_grad_2;
      const s_t tangent_tmp172 = -tangent_tmp150 + tangent_tmp171;
      const s_t tangent_tmp173 = tangent_tmp95*u1_grad_2;
      const s_t tangent_tmp174 = s_t(2)*tangent_tmp173;
      const s_t tangent_tmp175 = tangent_tmp148 - tangent_tmp174;
      const s_t tangent_tmp176 = tangent_tmp139*u1_grad_2;
      const s_t tangent_tmp177 = -tangent_tmp176;
      const s_t tangent_tmp178 = -eta_s*(tangent_tmp129*tangent_tmp70 - tangent_tmp62*tangent_tmp67) + ((s_t(1) / s_t(3)))*tangent_tmp169*tangent_tmp59;
      const s_t tangent_tmp179 = ((s_t(1) / s_t(3)))*tangent_tmp59;
      const s_t tangent_tmp180 = tangent_tmp1*tangent_tmp129;
      const s_t tangent_tmp181 = ((s_t(1) / s_t(3)))*tangent_tmp169;
      const s_t tangent_tmp182 = tangent_tmp181*u1_grad_2;
      const s_t tangent_tmp183 = s_t(6)*u1_grad_1 + s_t(6);
      const s_t tangent_tmp184 = tangent_tmp135 + tangent_tmp183;
      const s_t tangent_tmp185 = -s_t(2)*tangent_tmp5*tangent_tmp95;
      const s_t tangent_tmp186 = -tangent_tmp132 - tangent_tmp185;
      const s_t tangent_tmp187 = s_t(2)*u1_grad_1 + s_t(2);
      const s_t tangent_tmp188 = tangent_tmp140*tangent_tmp187;
      const s_t tangent_tmp189 = -(s_t(1) / s_t(3))*tangent_tmp169;
      const s_t tangent_tmp190 = eta_s*(tangent_tmp129*tangent_tmp69 + tangent_tmp64*tangent_tmp67) + tangent_tmp189*tangent_tmp65;
      const s_t tangent_tmp191 = ((s_t(1) / s_t(3)))*tangent_tmp65;
      const s_t tangent_tmp192 = tangent_tmp129*u2_grad_1;
      const s_t tangent_tmp193 = tangent_tmp12*(-tangent_tmp61 - tangent_tmp91);
      const s_t tangent_tmp194 = tangent_tmp62*u_dt_shift;
      const s_t tangent_tmp195 = tangent_tmp40*u1_grad_0;
      const s_t tangent_tmp196 = tangent_tmp66*u1_grad_2;
      const s_t tangent_tmp197 = tangent_tmp194 + tangent_tmp195 - tangent_tmp196;
      const s_t tangent_tmp198 = tangent_tmp1*tangent_tmp71;
      const s_t tangent_tmp199 = tangent_tmp47*u2_grad_0;
      const s_t tangent_tmp200 = tangent_tmp198 - tangent_tmp199;
      const s_t tangent_tmp201 = tangent_tmp50*(tangent_tmp197 + tangent_tmp200);
      const s_t tangent_tmp202 = -tangent_tmp198 + tangent_tmp199;
      const s_t tangent_tmp203 = -tangent_tmp195 + tangent_tmp196;
      const s_t tangent_tmp204 = tangent_tmp201 + tangent_tmp54*(s_t(2)*tangent_tmp194 + tangent_tmp202 + tangent_tmp203);
      const s_t tangent_tmp205 = tangent_tmp59*u_dt_shift;
      const s_t tangent_tmp206 = tangent_tmp28*u1_grad_0;
      const s_t tangent_tmp207 = tangent_tmp57*u1_grad_2;
      const s_t tangent_tmp208 = tangent_tmp206 - tangent_tmp207;
      const s_t tangent_tmp209 = tangent_tmp205 + tangent_tmp208;
      const s_t tangent_tmp210 = tangent_tmp70*u_dt_shift;
      const s_t tangent_tmp211 = tangent_tmp28*u2_grad_0;
      const s_t tangent_tmp212 = tangent_tmp1*tangent_tmp57;
      const s_t tangent_tmp213 = tangent_tmp211 - tangent_tmp212;
      const s_t tangent_tmp214 = -tangent_tmp210 - tangent_tmp213;
      const s_t tangent_tmp215 = -tangent_tmp102;
      const s_t tangent_tmp216 = tangent_tmp1*tangent_tmp72;
      const s_t tangent_tmp217 = tangent_tmp62*tangent_tmp88;
      const s_t tangent_tmp218 = s_t(2)*pow_2(u1_grad_0);
      const s_t tangent_tmp219 = s_t(2)*pow_2(u2_grad_0);
      const s_t tangent_tmp220 = tangent_tmp218 + tangent_tmp219;
      const s_t tangent_tmp221 = s_t(2)*u1_grad_2;
      const s_t tangent_tmp222 = tangent_tmp221*tangent_tmp5;
      const s_t tangent_tmp223 = s_t(2)*u2_grad_1;
      const s_t tangent_tmp224 = tangent_tmp1*tangent_tmp223;
      const s_t tangent_tmp225 = mu*(-tangent_tmp222 - tangent_tmp224);
      const s_t tangent_tmp226 = tangent_tmp67*u1_grad_0;
      const s_t tangent_tmp227 = tangent_tmp72*u2_grad_0;
      const s_t tangent_tmp228 = -tangent_tmp227;
      const s_t tangent_tmp229 = tangent_tmp226 + tangent_tmp228;
      const s_t tangent_tmp230 = tangent_tmp201 + tangent_tmp54*(s_t(2)*tangent_tmp1*tangent_tmp71 - tangent_tmp197 - s_t(2)*tangent_tmp199);
      const s_t tangent_tmp231 = ((s_t(1) / s_t(3)))*tangent_tmp33;
      const s_t tangent_tmp232 = tangent_tmp1*tangent_tmp66;
      const s_t tangent_tmp233 = tangent_tmp40*u2_grad_0;
      const s_t tangent_tmp234 = tangent_tmp47*u1_grad_0 - tangent_tmp71*u1_grad_2;
      const s_t tangent_tmp235 = tangent_tmp232 - tangent_tmp233 + tangent_tmp234;
      const s_t tangent_tmp236 = mu*(tangent_tmp133 + s_t(4)*tangent_tmp17 + tangent_tmp185) - tangent_tmp138*tangent_tmp140;
      const s_t tangent_tmp237 = tangent_tmp95*tangent_tmp99;
      const s_t tangent_tmp238 = mu*(-tangent_tmp116 - tangent_tmp237);
      const s_t tangent_tmp239 = tangent_tmp129*u1_grad_0;
      const s_t tangent_tmp240 = s_t(6)*u2_grad_0;
      const s_t tangent_tmp241 = tangent_tmp139*u2_grad_0;
      const s_t tangent_tmp242 = mu*(-tangent_tmp166 - tangent_tmp240 + s_t(4)*u0_grad_1*u1_grad_2) + tangent_tmp241;
      const s_t tangent_tmp243 = tangent_tmp201 + tangent_tmp54*(-tangent_tmp194 - s_t(2)*tangent_tmp196 - tangent_tmp200 + s_t(2)*tangent_tmp40*u1_grad_0);
      const s_t tangent_tmp244 = mu*(-tangent_tmp152 - tangent_tmp171 + s_t(4)*u0_grad_1*u2_grad_0) + tangent_tmp176;
      const s_t tangent_tmp245 = tangent_tmp95*tangent_tmp97;
      const s_t tangent_tmp246 = mu*(-tangent_tmp162 - tangent_tmp245);
      const s_t tangent_tmp247 = s_t(6)*u1_grad_0;
      const s_t tangent_tmp248 = tangent_tmp120 + tangent_tmp247;
      const s_t tangent_tmp249 = tangent_tmp139*u1_grad_0;
      const s_t tangent_tmp250 = -tangent_tmp249;
      const s_t tangent_tmp251 = tangent_tmp129*u2_grad_0;
      const s_t tangent_tmp252 = tangent_tmp12*(-tangent_tmp106 + tangent_tmp63);
      const s_t tangent_tmp253 = tangent_tmp64*u_dt_shift;
      const s_t tangent_tmp254 = tangent_tmp5*tangent_tmp66;
      const s_t tangent_tmp255 = tangent_tmp42*u1_grad_0;
      const s_t tangent_tmp256 = tangent_tmp253 + tangent_tmp254 - tangent_tmp255;
      const s_t tangent_tmp257 = tangent_tmp45*u2_grad_0;
      const s_t tangent_tmp258 = tangent_tmp71*u2_grad_1;
      const s_t tangent_tmp259 = tangent_tmp257 - tangent_tmp258;
      const s_t tangent_tmp260 = tangent_tmp50*(tangent_tmp256 + tangent_tmp259);
      const s_t tangent_tmp261 = -tangent_tmp257 + tangent_tmp258;
      const s_t tangent_tmp262 = -tangent_tmp254 + tangent_tmp255;
      const s_t tangent_tmp263 = tangent_tmp260 + tangent_tmp54*(s_t(2)*tangent_tmp253 + tangent_tmp261 + tangent_tmp262);
      const s_t tangent_tmp264 = tangent_tmp69*u_dt_shift;
      const s_t tangent_tmp265 = tangent_tmp26*u2_grad_0;
      const s_t tangent_tmp266 = tangent_tmp57*u2_grad_1;
      const s_t tangent_tmp267 = tangent_tmp265 - tangent_tmp266;
      const s_t tangent_tmp268 = tangent_tmp264 + tangent_tmp267;
      const s_t tangent_tmp269 = tangent_tmp65*u_dt_shift;
      const s_t tangent_tmp270 = tangent_tmp26*u1_grad_0;
      const s_t tangent_tmp271 = tangent_tmp5*tangent_tmp57;
      const s_t tangent_tmp272 = tangent_tmp270 - tangent_tmp271;
      const s_t tangent_tmp273 = -tangent_tmp269 - tangent_tmp272;
      const s_t tangent_tmp274 = -tangent_tmp112;
      const s_t tangent_tmp275 = tangent_tmp5*tangent_tmp67;
      const s_t tangent_tmp276 = tangent_tmp64*tangent_tmp88;
      const s_t tangent_tmp277 = tangent_tmp260 + tangent_tmp54*(-tangent_tmp256 - s_t(2)*tangent_tmp258 + s_t(2)*tangent_tmp45*u2_grad_0);
      const s_t tangent_tmp278 = tangent_tmp66*u2_grad_1;
      const s_t tangent_tmp279 = tangent_tmp42*u2_grad_0;
      const s_t tangent_tmp280 = tangent_tmp45*u1_grad_0 - tangent_tmp5*tangent_tmp71;
      const s_t tangent_tmp281 = -tangent_tmp278 + tangent_tmp279 - tangent_tmp280;
      const s_t tangent_tmp282 = mu*(-tangent_tmp147 - tangent_tmp174 + s_t(4)*u0_grad_2*u1_grad_0) + tangent_tmp154;
      const s_t tangent_tmp283 = -tangent_tmp164 + tangent_tmp240;
      const s_t tangent_tmp284 = -tangent_tmp241;
      const s_t tangent_tmp285 = mu*(-tangent_tmp118 - tangent_tmp237);
      const s_t tangent_tmp286 = tangent_tmp260 + tangent_tmp54*(-tangent_tmp253 - s_t(2)*tangent_tmp255 - tangent_tmp259 + s_t(2)*tangent_tmp5*tangent_tmp66);
      const s_t tangent_tmp287 = ((s_t(1) / s_t(3)))*tangent_tmp24;
      const s_t tangent_tmp288 = mu*(tangent_tmp136 + s_t(4)*tangent_tmp18 + tangent_tmp183) - tangent_tmp140*tangent_tmp187;
      const s_t tangent_tmp289 = mu*(-tangent_tmp122 - tangent_tmp247 + s_t(4)*u0_grad_2*u2_grad_1) + tangent_tmp249;
      const s_t tangent_tmp290 = mu*(-tangent_tmp161 - tangent_tmp245);
      const s_t tangent_tmp291 = tangent_tmp12*(-tangent_tmp121 - tangent_tmp32);
      const s_t tangent_tmp292 = -tangent_tmp39 - tangent_tmp52;
      const s_t tangent_tmp293 = -tangent_tmp26*u0_grad_2 + tangent_tmp28*u0_grad_1;
      const s_t tangent_tmp294 = -tangent_tmp124 + tangent_tmp125 + tangent_tmp293;
      const s_t tangent_tmp295 = tangent_tmp40*u0_grad_1;
      const s_t tangent_tmp296 = tangent_tmp42*u0_grad_2;
      const s_t tangent_tmp297 = tangent_tmp295 - tangent_tmp296 + tangent_tmp34;
      const s_t tangent_tmp298 = -tangent_tmp35 + tangent_tmp36;
      const s_t tangent_tmp299 = tangent_tmp50*(tangent_tmp297 + tangent_tmp298);
      const s_t tangent_tmp300 = tangent_tmp299 + tangent_tmp54*(s_t(2)*tangent_tmp1*tangent_tmp26 - tangent_tmp297 - s_t(2)*tangent_tmp35);
      const s_t tangent_tmp301 = tangent_tmp33*tangent_tmp88;
      const s_t tangent_tmp302 = ((s_t(1) / s_t(3)))*tangent_tmp62;
      const s_t tangent_tmp303 = tangent_tmp67*u0_grad_2;
      const s_t tangent_tmp304 = ((s_t(1) / s_t(3)))*tangent_tmp85;
      const s_t tangent_tmp305 = tangent_tmp67*u0_grad_1;
      const s_t tangent_tmp306 = s_t(2)*pow_2(u0_grad_2);
      const s_t tangent_tmp307 = tangent_tmp2 + tangent_tmp306;
      const s_t tangent_tmp308 = s_t(2)*pow_2(u0_grad_1);
      const s_t tangent_tmp309 = tangent_tmp308 + tangent_tmp4;
      const s_t tangent_tmp310 = tangent_tmp47*u0_grad_1;
      const s_t tangent_tmp311 = tangent_tmp45*u0_grad_2;
      const s_t tangent_tmp312 = tangent_tmp310 - tangent_tmp311;
      const s_t tangent_tmp313 = tangent_tmp25 + tangent_tmp312;
      const s_t tangent_tmp314 = -tangent_tmp295 + tangent_tmp296;
      const s_t tangent_tmp315 = tangent_tmp299 + tangent_tmp54*(tangent_tmp314 + s_t(2)*tangent_tmp34 + tangent_tmp37);
      const s_t tangent_tmp316 = tangent_tmp117*tangent_tmp95;
      const s_t tangent_tmp317 = tangent_tmp316 + tangent_tmp98;
      const s_t tangent_tmp318 = tangent_tmp129*u0_grad_2;
      const s_t tangent_tmp319 = tangent_tmp115*tangent_tmp95;
      const s_t tangent_tmp320 = mu*(-tangent_tmp110 - tangent_tmp319);
      const s_t tangent_tmp321 = tangent_tmp129*u0_grad_1;
      const s_t tangent_tmp322 = tangent_tmp274 + tangent_tmp321;
      const s_t tangent_tmp323 = tangent_tmp1*tangent_tmp221;
      const s_t tangent_tmp324 = tangent_tmp223*tangent_tmp5;
      const s_t tangent_tmp325 = mu*(-tangent_tmp323 - tangent_tmp324);
      const s_t tangent_tmp326 = tangent_tmp299 + tangent_tmp54*(-s_t(2)*tangent_tmp296 - tangent_tmp298 - tangent_tmp34 + s_t(2)*tangent_tmp40*u0_grad_1);
      const s_t tangent_tmp327 = tangent_tmp1*tangent_tmp67;
      const s_t tangent_tmp328 = tangent_tmp181*u0_grad_2;
      const s_t tangent_tmp329 = s_t(6)*u0_grad_2;
      const s_t tangent_tmp330 = tangent_tmp139*u0_grad_2;
      const s_t tangent_tmp331 = mu*(-tangent_tmp107 - tangent_tmp329 + s_t(4)*u1_grad_0*u2_grad_1) + tangent_tmp330;
      const s_t tangent_tmp332 = s_t(6)*u0_grad_1;
      const s_t tangent_tmp333 = tangent_tmp332 + tangent_tmp90;
      const s_t tangent_tmp334 = tangent_tmp139*u0_grad_1;
      const s_t tangent_tmp335 = -tangent_tmp334;
      const s_t tangent_tmp336 = tangent_tmp67*u2_grad_1;
      const s_t tangent_tmp337 = tangent_tmp12*(tangent_tmp1*tangent_tmp95 - tangent_tmp18);
      const s_t tangent_tmp338 = -tangent_tmp70*tangent_tmp88;
      const s_t tangent_tmp339 = tangent_tmp40*tangent_tmp95;
      const s_t tangent_tmp340 = tangent_tmp66*u0_grad_2;
      const s_t tangent_tmp341 = tangent_tmp210 + tangent_tmp339 - tangent_tmp340;
      const s_t tangent_tmp342 = -tangent_tmp211 + tangent_tmp212;
      const s_t tangent_tmp343 = tangent_tmp50*(-tangent_tmp341 - tangent_tmp342);
      const s_t tangent_tmp344 = tangent_tmp343 + tangent_tmp54*(s_t(2)*tangent_tmp211 - s_t(2)*tangent_tmp212 + tangent_tmp341);
      const s_t tangent_tmp345 = tangent_tmp194 + tangent_tmp202;
      const s_t tangent_tmp346 = -tangent_tmp28*tangent_tmp95 + tangent_tmp57*u0_grad_2;
      const s_t tangent_tmp347 = -tangent_tmp232 + tangent_tmp233 + tangent_tmp346;
      const s_t tangent_tmp348 = ((s_t(1) / s_t(3)))*tangent_tmp86;
      const s_t tangent_tmp349 = ((s_t(1) / s_t(3)))*tangent_tmp64;
      const s_t tangent_tmp350 = tangent_tmp67*tangent_tmp95;
      const s_t tangent_tmp351 = tangent_tmp304*u2_grad_0;
      const s_t tangent_tmp352 = s_t(4)*tangent_tmp5;
      const s_t tangent_tmp353 = -tangent_tmp339 + tangent_tmp340;
      const s_t tangent_tmp354 = tangent_tmp343 + tangent_tmp54*(-s_t(2)*tangent_tmp210 - tangent_tmp213 - tangent_tmp353);
      const s_t tangent_tmp355 = tangent_tmp71*u0_grad_2;
      const s_t tangent_tmp356 = tangent_tmp47*tangent_tmp95;
      const s_t tangent_tmp357 = tangent_tmp355 - tangent_tmp356;
      const s_t tangent_tmp358 = tangent_tmp205 + tangent_tmp357;
      const s_t tangent_tmp359 = s_t(2)*pow_2(tangent_tmp95);
      const s_t tangent_tmp360 = tangent_tmp219 + tangent_tmp359;
      const s_t tangent_tmp361 = tangent_tmp117*u0_grad_2;
      const s_t tangent_tmp362 = tangent_tmp224 + tangent_tmp361;
      const s_t tangent_tmp363 = -tangent_tmp129*tangent_tmp95;
      const s_t tangent_tmp364 = -tangent_tmp105 + tangent_tmp329;
      const s_t tangent_tmp365 = -tangent_tmp330;
      const s_t tangent_tmp366 = tangent_tmp343 + tangent_tmp54*(tangent_tmp210 - s_t(2)*tangent_tmp339 + s_t(2)*tangent_tmp340 + tangent_tmp342);
      const s_t tangent_tmp367 = tangent_tmp99*u2_grad_0;
      const s_t tangent_tmp368 = mu*(-tangent_tmp323 - tangent_tmp367);
      const s_t tangent_tmp369 = s_t(6)*u0_grad_0 + s_t(6);
      const s_t tangent_tmp370 = tangent_tmp369 + tangent_tmp9;
      const s_t tangent_tmp371 = tangent_tmp140*tangent_tmp94;
      const s_t tangent_tmp372 = tangent_tmp67*u2_grad_0;
      const s_t tangent_tmp373 = tangent_tmp12*(-tangent_tmp151 + tangent_tmp68);
      const s_t tangent_tmp374 = tangent_tmp69*tangent_tmp88;
      const s_t tangent_tmp375 = tangent_tmp42*tangent_tmp95;
      const s_t tangent_tmp376 = tangent_tmp66*u0_grad_1;
      const s_t tangent_tmp377 = tangent_tmp264 + tangent_tmp375 - tangent_tmp376;
      const s_t tangent_tmp378 = -tangent_tmp265 + tangent_tmp266;
      const s_t tangent_tmp379 = tangent_tmp50*(tangent_tmp377 + tangent_tmp378);
      const s_t tangent_tmp380 = tangent_tmp379 + tangent_tmp54*(-s_t(2)*tangent_tmp265 - tangent_tmp377 + s_t(2)*tangent_tmp57*u2_grad_1);
      const s_t tangent_tmp381 = tangent_tmp253 + tangent_tmp261;
      const s_t tangent_tmp382 = -tangent_tmp26*tangent_tmp95 + tangent_tmp57*u0_grad_1;
      const s_t tangent_tmp383 = tangent_tmp278 - tangent_tmp279 - tangent_tmp382;
      const s_t tangent_tmp384 = -tangent_tmp375 + tangent_tmp376;
      const s_t tangent_tmp385 = tangent_tmp379 + tangent_tmp54*(s_t(2)*tangent_tmp264 + tangent_tmp267 + tangent_tmp384);
      const s_t tangent_tmp386 = tangent_tmp71*u0_grad_1;
      const s_t tangent_tmp387 = tangent_tmp45*tangent_tmp95;
      const s_t tangent_tmp388 = tangent_tmp386 - tangent_tmp387;
      const s_t tangent_tmp389 = -tangent_tmp269 - tangent_tmp388;
      const s_t tangent_tmp390 = tangent_tmp129*tangent_tmp95;
      const s_t tangent_tmp391 = tangent_tmp379 + tangent_tmp54*(-tangent_tmp264 - s_t(2)*tangent_tmp376 - tangent_tmp378 + s_t(2)*tangent_tmp42*tangent_tmp95);
      const s_t tangent_tmp392 = mu*(-tangent_tmp332 - tangent_tmp92 + s_t(4)*u1_grad_2*u2_grad_0) + tangent_tmp334;
      const s_t tangent_tmp393 = mu*(tangent_tmp10 + tangent_tmp369 + s_t(4)*tangent_tmp8) - tangent_tmp140*tangent_tmp94;
      const s_t tangent_tmp394 = mu*(-tangent_tmp324 - tangent_tmp367);
      const s_t tangent_tmp395 = tangent_tmp12*(tangent_tmp15 - tangent_tmp165);
      const s_t tangent_tmp396 = -tangent_tmp39 - tangent_tmp53;
      const s_t tangent_tmp397 = -tangent_tmp126 - tangent_tmp293;
      const s_t tangent_tmp398 = tangent_tmp25 - tangent_tmp310 + tangent_tmp311;
      const s_t tangent_tmp399 = -tangent_tmp27 + tangent_tmp29;
      const s_t tangent_tmp400 = tangent_tmp50*(tangent_tmp398 + tangent_tmp399);
      const s_t tangent_tmp401 = tangent_tmp400 + tangent_tmp54*(-s_t(2)*tangent_tmp27 + s_t(2)*tangent_tmp28*tangent_tmp5 - tangent_tmp398);
      const s_t tangent_tmp402 = tangent_tmp24*tangent_tmp88;
      const s_t tangent_tmp403 = tangent_tmp72*u0_grad_2;
      const s_t tangent_tmp404 = tangent_tmp72*u0_grad_1;
      const s_t tangent_tmp405 = tangent_tmp314 + tangent_tmp34;
      const s_t tangent_tmp406 = tangent_tmp400 + tangent_tmp54*(-tangent_tmp25 - s_t(2)*tangent_tmp310 - tangent_tmp399 + s_t(2)*tangent_tmp45*u0_grad_2);
      const s_t tangent_tmp407 = tangent_tmp72*u1_grad_2;
      const s_t tangent_tmp408 = tangent_tmp5*tangent_tmp72;
      const s_t tangent_tmp409 = tangent_tmp159*u0_grad_1;
      const s_t tangent_tmp410 = tangent_tmp0 + tangent_tmp306 + s_t(2);
      const s_t tangent_tmp411 = tangent_tmp308 + tangent_tmp6;
      const s_t tangent_tmp412 = tangent_tmp400 + tangent_tmp54*(s_t(2)*tangent_tmp25 + tangent_tmp30 + tangent_tmp312);
      const s_t tangent_tmp413 = mu*(-tangent_tmp100 - tangent_tmp316);
      const s_t tangent_tmp414 = tangent_tmp215 + tangent_tmp318;
      const s_t tangent_tmp415 = tangent_tmp109 + tangent_tmp319;
      const s_t tangent_tmp416 = tangent_tmp12*(-tangent_tmp173 + tangent_tmp58);
      const s_t tangent_tmp417 = tangent_tmp59*tangent_tmp88;
      const s_t tangent_tmp418 = tangent_tmp205 - tangent_tmp355 + tangent_tmp356;
      const s_t tangent_tmp419 = -tangent_tmp206 + tangent_tmp207;
      const s_t tangent_tmp420 = tangent_tmp50*(tangent_tmp418 + tangent_tmp419);
      const s_t tangent_tmp421 = tangent_tmp420 + tangent_tmp54*(-s_t(2)*tangent_tmp206 - tangent_tmp418 + s_t(2)*tangent_tmp57*u1_grad_2);
      const s_t tangent_tmp422 = tangent_tmp194 + tangent_tmp203;
      const s_t tangent_tmp423 = -tangent_tmp234 - tangent_tmp346;
      const s_t tangent_tmp424 = tangent_tmp72*tangent_tmp95;
      const s_t tangent_tmp425 = tangent_tmp304*u1_grad_0;
      const s_t tangent_tmp426 = tangent_tmp420 + tangent_tmp54*(-tangent_tmp205 - s_t(2)*tangent_tmp355 - tangent_tmp419 + s_t(2)*tangent_tmp47*tangent_tmp95);
      const s_t tangent_tmp427 = -tangent_tmp210 - tangent_tmp353;
      const s_t tangent_tmp428 = tangent_tmp72*u1_grad_0;
      const s_t tangent_tmp429 = tangent_tmp420 + tangent_tmp54*(s_t(2)*tangent_tmp205 + tangent_tmp208 + tangent_tmp357);
      const s_t tangent_tmp430 = tangent_tmp218 + tangent_tmp359;
      const s_t tangent_tmp431 = tangent_tmp222 + tangent_tmp361;
      const s_t tangent_tmp432 = tangent_tmp12*(-tangent_tmp17 + tangent_tmp5*tangent_tmp95);
      const s_t tangent_tmp433 = -tangent_tmp65*tangent_tmp88;
      const s_t tangent_tmp434 = tangent_tmp269 - tangent_tmp386 + tangent_tmp387;
      const s_t tangent_tmp435 = -tangent_tmp270 + tangent_tmp271;
      const s_t tangent_tmp436 = tangent_tmp50*(-tangent_tmp434 - tangent_tmp435);
      const s_t tangent_tmp437 = tangent_tmp436 + tangent_tmp54*(s_t(2)*tangent_tmp270 - s_t(2)*tangent_tmp271 + tangent_tmp434);
      const s_t tangent_tmp438 = tangent_tmp253 + tangent_tmp262;
      const s_t tangent_tmp439 = tangent_tmp280 + tangent_tmp382;
      const s_t tangent_tmp440 = tangent_tmp436 + tangent_tmp54*(tangent_tmp269 + s_t(2)*tangent_tmp386 - s_t(2)*tangent_tmp387 + tangent_tmp435);
      const s_t tangent_tmp441 = tangent_tmp264 + tangent_tmp384;
      const s_t tangent_tmp442 = s_t(4)*tangent_tmp1;
      const s_t tangent_tmp443 = tangent_tmp436 + tangent_tmp54*(-s_t(2)*tangent_tmp269 - tangent_tmp272 - tangent_tmp388);
      const s_t tangent_grad_d0_0_grad0_0 = mu*(tangent_tmp3 + tangent_tmp7) + tangent_tmp11*tangent_tmp13 + tangent_tmp23*(eta_s*(tangent_tmp24*tangent_tmp31 + tangent_tmp33*tangent_tmp38) - tangent_tmp55*tangent_tmp56) + tangent_tmp87*tangent_tmp89;
      const s_t tangent_grad_d0_0_grad0_1 = mu*(-tangent_tmp101 + s_t(2)*tangent_tmp94*u0_grad_1 - tangent_tmp96*u0_grad_1) + tangent_tmp104*tangent_tmp89 + tangent_tmp13*tangent_tmp93 + tangent_tmp23*(-eta_s*(-tangent_tmp102 - tangent_tmp103 - tangent_tmp31*tangent_tmp59 + tangent_tmp38*tangent_tmp70) + ((s_t(1) / s_t(3)))*tangent_tmp55*tangent_tmp62);
      const s_t tangent_grad_d0_0_grad0_2 = mu*(-tangent_tmp111 + s_t(2)*tangent_tmp94*u0_grad_2 - tangent_tmp96*u0_grad_2) + tangent_tmp108*tangent_tmp13 + tangent_tmp114*tangent_tmp89 + tangent_tmp23*(-eta_s*(-tangent_tmp112 - tangent_tmp113 + tangent_tmp31*tangent_tmp65 - tangent_tmp38*tangent_tmp69) + ((s_t(1) / s_t(3)))*tangent_tmp55*tangent_tmp64);
      const s_t tangent_grad_d0_0_grad1_0 = tangent_tmp119 + tangent_tmp123*tangent_tmp13 + tangent_tmp131*tangent_tmp89 + tangent_tmp23*(-eta_s*(-tangent_tmp127*tangent_tmp24 + tangent_tmp19*tangent_tmp38) + ((s_t(1) / s_t(3)))*tangent_tmp128*tangent_tmp33);
      const s_t tangent_grad_d0_0_grad1_1 = mu*(-tangent_tmp134 + s_t(2)*tangent_tmp5*tangent_tmp94) + tangent_tmp13*tangent_tmp137 + tangent_tmp141 + tangent_tmp143*tangent_tmp89 + tangent_tmp23*(eta_s*(tangent_tmp127*tangent_tmp59 + tangent_tmp145 + tangent_tmp38*tangent_tmp62) + tangent_tmp1*tangent_tmp142 - tangent_tmp128*tangent_tmp144);
      const s_t tangent_grad_d0_0_grad1_2 = mu*(tangent_tmp146*u1_grad_2 + tangent_tmp149) + tangent_tmp13*tangent_tmp153 + tangent_tmp155 + tangent_tmp156*tangent_tmp89 + tangent_tmp23*(-eta_s*(tangent_tmp127*tangent_tmp65 + tangent_tmp158 - tangent_tmp38*tangent_tmp64) + tangent_tmp128*tangent_tmp157 + tangent_tmp160);
      const s_t tangent_grad_d0_0_grad2_0 = tangent_tmp13*tangent_tmp167 + tangent_tmp163 + tangent_tmp170*tangent_tmp89 + tangent_tmp23*(-eta_s*(-tangent_tmp127*tangent_tmp33 + tangent_tmp19*tangent_tmp31) + ((s_t(1) / s_t(3)))*tangent_tmp168*tangent_tmp24);
      const s_t tangent_grad_d0_0_grad2_1 = mu*(tangent_tmp146*u2_grad_1 + tangent_tmp172) + tangent_tmp13*tangent_tmp175 + tangent_tmp177 + tangent_tmp178*tangent_tmp89 + tangent_tmp23*(-eta_s*(tangent_tmp127*tangent_tmp70 + tangent_tmp180 - tangent_tmp31*tangent_tmp62) + tangent_tmp168*tangent_tmp179 + tangent_tmp182);
      const s_t tangent_grad_d0_0_grad2_2 = mu*(s_t(2)*tangent_tmp1*tangent_tmp94 - tangent_tmp184) + tangent_tmp13*tangent_tmp186 + tangent_tmp188 + tangent_tmp190*tangent_tmp89 + tangent_tmp23*(eta_s*(tangent_tmp127*tangent_tmp69 + tangent_tmp192 + tangent_tmp31*tangent_tmp64) - tangent_tmp168*tangent_tmp191 + tangent_tmp189*tangent_tmp5);
      const s_t tangent_grad_d0_1_grad0_0 = -mu*tangent_tmp101 + tangent_tmp11*tangent_tmp193 + tangent_tmp217*tangent_tmp87 + tangent_tmp23*(eta_s*(tangent_tmp209*tangent_tmp24 + tangent_tmp214*tangent_tmp33 + tangent_tmp215 + tangent_tmp216) - tangent_tmp204*tangent_tmp56);
      const s_t tangent_grad_d0_1_grad0_1 = mu*(tangent_tmp220 + tangent_tmp3) + tangent_tmp104*tangent_tmp217 + tangent_tmp193*tangent_tmp93 + tangent_tmp23*(-eta_s*(-tangent_tmp209*tangent_tmp59 + tangent_tmp214*tangent_tmp70) + ((s_t(1) / s_t(3)))*tangent_tmp204*tangent_tmp62);
      const s_t tangent_grad_d0_1_grad0_2 = tangent_tmp108*tangent_tmp193 + tangent_tmp114*tangent_tmp217 + tangent_tmp225 + tangent_tmp23*(-eta_s*(tangent_tmp209*tangent_tmp65 - tangent_tmp214*tangent_tmp69 - tangent_tmp229) + ((s_t(1) / s_t(3)))*tangent_tmp204*tangent_tmp64);
      const s_t tangent_grad_d0_1_grad1_0 = tangent_tmp123*tangent_tmp193 + tangent_tmp131*tangent_tmp217 + tangent_tmp23*(-eta_s*(tangent_tmp145 + tangent_tmp19*tangent_tmp214 - tangent_tmp235*tangent_tmp24) + tangent_tmp1*tangent_tmp159 + tangent_tmp230*tangent_tmp231) + tangent_tmp236;
      const s_t tangent_grad_d0_1_grad1_1 = tangent_tmp137*tangent_tmp193 + tangent_tmp143*tangent_tmp217 + tangent_tmp23*(eta_s*(tangent_tmp214*tangent_tmp62 + tangent_tmp235*tangent_tmp59) - tangent_tmp144*tangent_tmp230) + tangent_tmp238;
      const s_t tangent_grad_d0_1_grad1_2 = tangent_tmp153*tangent_tmp193 + tangent_tmp156*tangent_tmp217 + tangent_tmp23*(-eta_s*(-tangent_tmp214*tangent_tmp64 + tangent_tmp235*tangent_tmp65 - tangent_tmp239) - tangent_tmp159*u2_grad_0 + ((s_t(1) / s_t(3)))*tangent_tmp230*tangent_tmp69) + tangent_tmp242;
      const s_t tangent_grad_d0_1_grad2_0 = tangent_tmp167*tangent_tmp193 + tangent_tmp170*tangent_tmp217 + tangent_tmp23*(-eta_s*(-tangent_tmp180 + tangent_tmp19*tangent_tmp209 - tangent_tmp235*tangent_tmp33) - tangent_tmp182 + ((s_t(1) / s_t(3)))*tangent_tmp24*tangent_tmp243) + tangent_tmp244;
      const s_t tangent_grad_d0_1_grad2_1 = tangent_tmp175*tangent_tmp193 + tangent_tmp178*tangent_tmp217 + tangent_tmp23*(-eta_s*(-tangent_tmp209*tangent_tmp62 + tangent_tmp235*tangent_tmp70) + ((s_t(1) / s_t(3)))*tangent_tmp243*tangent_tmp59) + tangent_tmp246;
      const s_t tangent_grad_d0_1_grad2_2 = mu*(s_t(4)*tangent_tmp121 + tangent_tmp248) + tangent_tmp186*tangent_tmp193 + tangent_tmp190*tangent_tmp217 + tangent_tmp23*(eta_s*(tangent_tmp209*tangent_tmp64 + tangent_tmp235*tangent_tmp69 - tangent_tmp251) - tangent_tmp189*u1_grad_0 - tangent_tmp191*tangent_tmp243) + tangent_tmp250;
      const s_t tangent_grad_d0_2_grad0_0 = -mu*tangent_tmp111 + tangent_tmp11*tangent_tmp252 + tangent_tmp23*(eta_s*(tangent_tmp24*tangent_tmp273 + tangent_tmp268*tangent_tmp33 + tangent_tmp274 + tangent_tmp275) - tangent_tmp263*tangent_tmp56) + tangent_tmp276*tangent_tmp87;
      const s_t tangent_grad_d0_2_grad0_1 = tangent_tmp104*tangent_tmp276 + tangent_tmp225 + tangent_tmp23*(-eta_s*(tangent_tmp229 + tangent_tmp268*tangent_tmp70 - tangent_tmp273*tangent_tmp59) + ((s_t(1) / s_t(3)))*tangent_tmp263*tangent_tmp62) + tangent_tmp252*tangent_tmp93;
      const s_t tangent_grad_d0_2_grad0_2 = mu*(tangent_tmp220 + tangent_tmp7 + s_t(2)) + tangent_tmp108*tangent_tmp252 + tangent_tmp114*tangent_tmp276 + tangent_tmp23*(-eta_s*(-tangent_tmp268*tangent_tmp69 + tangent_tmp273*tangent_tmp65) + ((s_t(1) / s_t(3)))*tangent_tmp263*tangent_tmp64);
      const s_t tangent_grad_d0_2_grad1_0 = tangent_tmp123*tangent_tmp252 + tangent_tmp131*tangent_tmp276 + tangent_tmp23*(-eta_s*(-tangent_tmp158 + tangent_tmp19*tangent_tmp268 - tangent_tmp24*tangent_tmp281) - tangent_tmp160 + ((s_t(1) / s_t(3)))*tangent_tmp277*tangent_tmp33) + tangent_tmp282;
      const s_t tangent_grad_d0_2_grad1_1 = mu*(s_t(4)*tangent_tmp165 + tangent_tmp283) + tangent_tmp137*tangent_tmp252 + tangent_tmp143*tangent_tmp276 + tangent_tmp23*(eta_s*(-tangent_tmp239 + tangent_tmp268*tangent_tmp62 + tangent_tmp281*tangent_tmp59) - tangent_tmp142*u2_grad_0 - tangent_tmp144*tangent_tmp277) + tangent_tmp284;
      const s_t tangent_grad_d0_2_grad1_2 = tangent_tmp153*tangent_tmp252 + tangent_tmp156*tangent_tmp276 + tangent_tmp23*(-eta_s*(-tangent_tmp268*tangent_tmp64 + tangent_tmp281*tangent_tmp65) + ((s_t(1) / s_t(3)))*tangent_tmp277*tangent_tmp69) + tangent_tmp285;
      const s_t tangent_grad_d0_2_grad2_0 = tangent_tmp167*tangent_tmp252 + tangent_tmp170*tangent_tmp276 + tangent_tmp23*(-eta_s*(tangent_tmp19*tangent_tmp273 + tangent_tmp192 - tangent_tmp281*tangent_tmp33) + tangent_tmp181*tangent_tmp5 + tangent_tmp286*tangent_tmp287) + tangent_tmp288;
      const s_t tangent_grad_d0_2_grad2_1 = tangent_tmp175*tangent_tmp252 + tangent_tmp178*tangent_tmp276 + tangent_tmp23*(-eta_s*(-tangent_tmp251 - tangent_tmp273*tangent_tmp62 + tangent_tmp281*tangent_tmp70) - tangent_tmp181*u1_grad_0 + ((s_t(1) / s_t(3)))*tangent_tmp286*tangent_tmp59) + tangent_tmp289;
      const s_t tangent_grad_d0_2_grad2_2 = tangent_tmp186*tangent_tmp252 + tangent_tmp190*tangent_tmp276 + tangent_tmp23*(eta_s*(tangent_tmp273*tangent_tmp64 + tangent_tmp281*tangent_tmp69) - tangent_tmp191*tangent_tmp286) + tangent_tmp290;
      const s_t tangent_grad_d1_0_grad0_0 = tangent_tmp11*tangent_tmp291 + tangent_tmp119 + tangent_tmp23*(eta_s*(tangent_tmp24*tangent_tmp294 + tangent_tmp292*tangent_tmp33) - tangent_tmp300*tangent_tmp56) + tangent_tmp301*tangent_tmp87;
      const s_t tangent_grad_d1_0_grad0_1 = tangent_tmp104*tangent_tmp301 + tangent_tmp23*(-eta_s*(tangent_tmp292*tangent_tmp70 - tangent_tmp294*tangent_tmp59 + tangent_tmp303) + tangent_tmp1*tangent_tmp304 + tangent_tmp300*tangent_tmp302) + tangent_tmp236 + tangent_tmp291*tangent_tmp93;
      const s_t tangent_grad_d1_0_grad0_2 = tangent_tmp108*tangent_tmp291 + tangent_tmp114*tangent_tmp301 + tangent_tmp23*(-eta_s*(-tangent_tmp292*tangent_tmp69 + tangent_tmp294*tangent_tmp65 - tangent_tmp305) + ((s_t(1) / s_t(3)))*tangent_tmp300*tangent_tmp64 - tangent_tmp304*u2_grad_1) + tangent_tmp282;
      const s_t tangent_grad_d1_0_grad1_0 = mu*(tangent_tmp307 + tangent_tmp309) + tangent_tmp123*tangent_tmp291 + tangent_tmp131*tangent_tmp301 + tangent_tmp23*(-eta_s*(tangent_tmp19*tangent_tmp292 - tangent_tmp24*tangent_tmp313) + ((s_t(1) / s_t(3)))*tangent_tmp315*tangent_tmp33);
      const s_t tangent_grad_d1_0_grad1_1 = -mu*tangent_tmp317 + tangent_tmp137*tangent_tmp291 + tangent_tmp143*tangent_tmp301 + tangent_tmp23*(eta_s*(tangent_tmp216 + tangent_tmp292*tangent_tmp62 + tangent_tmp313*tangent_tmp59 - tangent_tmp318) - tangent_tmp144*tangent_tmp315);
      const s_t tangent_grad_d1_0_grad1_2 = tangent_tmp153*tangent_tmp291 + tangent_tmp156*tangent_tmp301 + tangent_tmp23*(-eta_s*(-tangent_tmp292*tangent_tmp64 + tangent_tmp313*tangent_tmp65 - tangent_tmp322) + ((s_t(1) / s_t(3)))*tangent_tmp315*tangent_tmp69) + tangent_tmp320;
      const s_t tangent_grad_d1_0_grad2_0 = tangent_tmp167*tangent_tmp291 + tangent_tmp170*tangent_tmp301 + tangent_tmp23*(-eta_s*(tangent_tmp19*tangent_tmp294 - tangent_tmp313*tangent_tmp33) + ((s_t(1) / s_t(3)))*tangent_tmp24*tangent_tmp326) + tangent_tmp325;
      const s_t tangent_grad_d1_0_grad2_1 = tangent_tmp175*tangent_tmp291 + tangent_tmp178*tangent_tmp301 + tangent_tmp23*(-eta_s*(-tangent_tmp294*tangent_tmp62 + tangent_tmp313*tangent_tmp70 - tangent_tmp327) + ((s_t(1) / s_t(3)))*tangent_tmp326*tangent_tmp59 - tangent_tmp328) + tangent_tmp331;
      const s_t tangent_grad_d1_0_grad2_2 = mu*(tangent_tmp333 + s_t(4)*tangent_tmp91) + tangent_tmp186*tangent_tmp291 + tangent_tmp190*tangent_tmp301 + tangent_tmp23*(eta_s*(tangent_tmp294*tangent_tmp64 + tangent_tmp313*tangent_tmp69 - tangent_tmp336) - tangent_tmp189*u0_grad_1 - tangent_tmp191*tangent_tmp326) + tangent_tmp335;
      const s_t tangent_grad_d1_1_grad0_0 = mu*(-tangent_tmp134 + s_t(2)*tangent_tmp187*tangent_tmp95) + tangent_tmp11*tangent_tmp337 + tangent_tmp141 + tangent_tmp23*(eta_s*(tangent_tmp24*tangent_tmp347 + tangent_tmp303 + tangent_tmp33*tangent_tmp345) + tangent_tmp1*tangent_tmp348 - tangent_tmp344*tangent_tmp56) + tangent_tmp338*tangent_tmp87;
      const s_t tangent_grad_d1_1_grad0_1 = tangent_tmp104*tangent_tmp338 + tangent_tmp23*(-eta_s*(tangent_tmp345*tangent_tmp70 - tangent_tmp347*tangent_tmp59) + ((s_t(1) / s_t(3)))*tangent_tmp344*tangent_tmp62) + tangent_tmp238 + tangent_tmp337*tangent_tmp93;
      const s_t tangent_grad_d1_1_grad0_2 = mu*(tangent_tmp115*tangent_tmp187 + tangent_tmp283) + tangent_tmp108*tangent_tmp337 + tangent_tmp114*tangent_tmp338 + tangent_tmp23*(-eta_s*(-tangent_tmp345*tangent_tmp69 + tangent_tmp347*tangent_tmp65 + tangent_tmp350) + tangent_tmp344*tangent_tmp349 + tangent_tmp351) + tangent_tmp284;
      const s_t tangent_grad_d1_1_grad1_0 = mu*(s_t(2)*tangent_tmp187*u1_grad_0 - tangent_tmp317 - tangent_tmp352*u1_grad_0) + tangent_tmp123*tangent_tmp337 + tangent_tmp131*tangent_tmp338 + tangent_tmp23*(-eta_s*(-tangent_tmp103 + tangent_tmp19*tangent_tmp345 - tangent_tmp24*tangent_tmp358 - tangent_tmp318) + ((s_t(1) / s_t(3)))*tangent_tmp33*tangent_tmp354);
      const s_t tangent_grad_d1_1_grad1_1 = mu*(tangent_tmp307 + tangent_tmp360) + tangent_tmp137*tangent_tmp337 + tangent_tmp143*tangent_tmp338 + tangent_tmp23*(eta_s*(tangent_tmp345*tangent_tmp62 + tangent_tmp358*tangent_tmp59) - tangent_tmp144*tangent_tmp354);
      const s_t tangent_grad_d1_1_grad1_2 = mu*(s_t(2)*tangent_tmp187*u1_grad_2 - tangent_tmp352*u1_grad_2 - tangent_tmp362) + tangent_tmp153*tangent_tmp337 + tangent_tmp156*tangent_tmp338 + tangent_tmp23*(-eta_s*(-tangent_tmp227 - tangent_tmp345*tangent_tmp64 + tangent_tmp358*tangent_tmp65 - tangent_tmp363) + ((s_t(1) / s_t(3)))*tangent_tmp354*tangent_tmp69);
      const s_t tangent_grad_d1_1_grad2_0 = mu*(tangent_tmp187*tangent_tmp97 + tangent_tmp364) + tangent_tmp167*tangent_tmp337 + tangent_tmp170*tangent_tmp338 + tangent_tmp23*(-eta_s*(tangent_tmp19*tangent_tmp347 + tangent_tmp327 - tangent_tmp33*tangent_tmp358) + tangent_tmp287*tangent_tmp366 + tangent_tmp328) + tangent_tmp365;
      const s_t tangent_grad_d1_1_grad2_1 = tangent_tmp175*tangent_tmp337 + tangent_tmp178*tangent_tmp338 + tangent_tmp23*(-eta_s*(-tangent_tmp347*tangent_tmp62 + tangent_tmp358*tangent_tmp70) + ((s_t(1) / s_t(3)))*tangent_tmp366*tangent_tmp59) + tangent_tmp368;
      const s_t tangent_grad_d1_1_grad2_2 = mu*(s_t(2)*tangent_tmp1*tangent_tmp187 - tangent_tmp370) + tangent_tmp186*tangent_tmp337 + tangent_tmp190*tangent_tmp338 + tangent_tmp23*(eta_s*(tangent_tmp347*tangent_tmp64 + tangent_tmp358*tangent_tmp69 + tangent_tmp372) + tangent_tmp189*tangent_tmp95 - tangent_tmp191*tangent_tmp366) + tangent_tmp371;
      const s_t tangent_grad_d1_2_grad0_0 = mu*(tangent_tmp149 + s_t(4)*tangent_tmp173) + tangent_tmp11*tangent_tmp373 + tangent_tmp155 + tangent_tmp23*(eta_s*(tangent_tmp24*tangent_tmp383 - tangent_tmp305 + tangent_tmp33*tangent_tmp381) - tangent_tmp348*u2_grad_1 - tangent_tmp380*tangent_tmp56) + tangent_tmp374*tangent_tmp87;
      const s_t tangent_grad_d1_2_grad0_1 = tangent_tmp104*tangent_tmp374 + tangent_tmp23*(-eta_s*(-tangent_tmp350 + tangent_tmp381*tangent_tmp70 - tangent_tmp383*tangent_tmp59) - tangent_tmp351 + ((s_t(1) / s_t(3)))*tangent_tmp380*tangent_tmp62) + tangent_tmp242 + tangent_tmp373*tangent_tmp93;
      const s_t tangent_grad_d1_2_grad0_2 = tangent_tmp108*tangent_tmp373 + tangent_tmp114*tangent_tmp374 + tangent_tmp23*(-eta_s*(-tangent_tmp381*tangent_tmp69 + tangent_tmp383*tangent_tmp65) + ((s_t(1) / s_t(3)))*tangent_tmp380*tangent_tmp64) + tangent_tmp285;
      const s_t tangent_grad_d1_2_grad1_0 = tangent_tmp123*tangent_tmp373 + tangent_tmp131*tangent_tmp374 + tangent_tmp23*(-eta_s*(tangent_tmp19*tangent_tmp381 - tangent_tmp24*tangent_tmp389 + tangent_tmp322) + ((s_t(1) / s_t(3)))*tangent_tmp33*tangent_tmp385) + tangent_tmp320;
      const s_t tangent_grad_d1_2_grad1_1 = -mu*tangent_tmp362 + tangent_tmp137*tangent_tmp373 + tangent_tmp143*tangent_tmp374 + tangent_tmp23*(eta_s*(tangent_tmp228 + tangent_tmp381*tangent_tmp62 + tangent_tmp389*tangent_tmp59 + tangent_tmp390) - tangent_tmp144*tangent_tmp385);
      const s_t tangent_grad_d1_2_grad1_2 = mu*(tangent_tmp309 + tangent_tmp360 + s_t(2)) + tangent_tmp153*tangent_tmp373 + tangent_tmp156*tangent_tmp374 + tangent_tmp23*(-eta_s*(-tangent_tmp381*tangent_tmp64 + tangent_tmp389*tangent_tmp65) + ((s_t(1) / s_t(3)))*tangent_tmp385*tangent_tmp69);
      const s_t tangent_grad_d1_2_grad2_0 = tangent_tmp167*tangent_tmp373 + tangent_tmp170*tangent_tmp374 + tangent_tmp23*(-eta_s*(tangent_tmp19*tangent_tmp383 - tangent_tmp33*tangent_tmp389 - tangent_tmp336) - tangent_tmp181*u0_grad_1 + ((s_t(1) / s_t(3)))*tangent_tmp24*tangent_tmp391) + tangent_tmp392;
      const s_t tangent_grad_d1_2_grad2_1 = tangent_tmp175*tangent_tmp373 + tangent_tmp178*tangent_tmp374 + tangent_tmp23*(-eta_s*(tangent_tmp372 - tangent_tmp383*tangent_tmp62 + tangent_tmp389*tangent_tmp70) + tangent_tmp179*tangent_tmp391 + tangent_tmp181*tangent_tmp95) + tangent_tmp393;
      const s_t tangent_grad_d1_2_grad2_2 = tangent_tmp186*tangent_tmp373 + tangent_tmp190*tangent_tmp374 + tangent_tmp23*(eta_s*(tangent_tmp383*tangent_tmp64 + tangent_tmp389*tangent_tmp69) - tangent_tmp191*tangent_tmp391) + tangent_tmp394;
      const s_t tangent_grad_d2_0_grad0_0 = tangent_tmp11*tangent_tmp395 + tangent_tmp163 + tangent_tmp23*(eta_s*(tangent_tmp24*tangent_tmp396 + tangent_tmp33*tangent_tmp397) - tangent_tmp401*tangent_tmp56) + tangent_tmp402*tangent_tmp87;
      const s_t tangent_grad_d2_0_grad0_1 = tangent_tmp104*tangent_tmp402 + tangent_tmp23*(-eta_s*(-tangent_tmp396*tangent_tmp59 + tangent_tmp397*tangent_tmp70 - tangent_tmp403) - tangent_tmp304*u1_grad_2 + ((s_t(1) / s_t(3)))*tangent_tmp401*tangent_tmp62) + tangent_tmp244 + tangent_tmp395*tangent_tmp93;
      const s_t tangent_grad_d2_0_grad0_2 = tangent_tmp108*tangent_tmp395 + tangent_tmp114*tangent_tmp402 + tangent_tmp23*(-eta_s*(tangent_tmp396*tangent_tmp65 - tangent_tmp397*tangent_tmp69 + tangent_tmp404) + tangent_tmp304*tangent_tmp5 + tangent_tmp349*tangent_tmp401) + tangent_tmp288;
      const s_t tangent_grad_d2_0_grad1_0 = tangent_tmp123*tangent_tmp395 + tangent_tmp131*tangent_tmp402 + tangent_tmp23*(-eta_s*(tangent_tmp19*tangent_tmp397 - tangent_tmp24*tangent_tmp405) + ((s_t(1) / s_t(3)))*tangent_tmp33*tangent_tmp406) + tangent_tmp325;
      const s_t tangent_grad_d2_0_grad1_1 = mu*(s_t(4)*tangent_tmp106 + tangent_tmp364) + tangent_tmp137*tangent_tmp395 + tangent_tmp143*tangent_tmp402 + tangent_tmp23*(eta_s*(tangent_tmp397*tangent_tmp62 + tangent_tmp405*tangent_tmp59 - tangent_tmp407) - tangent_tmp142*u0_grad_2 - tangent_tmp144*tangent_tmp406) + tangent_tmp365;
      const s_t tangent_grad_d2_0_grad1_2 = tangent_tmp153*tangent_tmp395 + tangent_tmp156*tangent_tmp402 + tangent_tmp23*(-eta_s*(-tangent_tmp397*tangent_tmp64 + tangent_tmp405*tangent_tmp65 - tangent_tmp408) + ((s_t(1) / s_t(3)))*tangent_tmp406*tangent_tmp69 - tangent_tmp409) + tangent_tmp392;
      const s_t tangent_grad_d2_0_grad2_0 = mu*(tangent_tmp410 + tangent_tmp411) + tangent_tmp167*tangent_tmp395 + tangent_tmp170*tangent_tmp402 + tangent_tmp23*(-eta_s*(tangent_tmp19*tangent_tmp396 - tangent_tmp33*tangent_tmp405) + ((s_t(1) / s_t(3)))*tangent_tmp24*tangent_tmp412);
      const s_t tangent_grad_d2_0_grad2_1 = tangent_tmp175*tangent_tmp395 + tangent_tmp178*tangent_tmp402 + tangent_tmp23*(-eta_s*(-tangent_tmp396*tangent_tmp62 + tangent_tmp405*tangent_tmp70 - tangent_tmp414) + ((s_t(1) / s_t(3)))*tangent_tmp412*tangent_tmp59) + tangent_tmp413;
      const s_t tangent_grad_d2_0_grad2_2 = -mu*tangent_tmp415 + tangent_tmp186*tangent_tmp395 + tangent_tmp190*tangent_tmp402 + tangent_tmp23*(eta_s*(tangent_tmp275 - tangent_tmp321 + tangent_tmp396*tangent_tmp64 + tangent_tmp405*tangent_tmp69) - tangent_tmp191*tangent_tmp412);
      const s_t tangent_grad_d2_1_grad0_0 = mu*(s_t(4)*tangent_tmp151 + tangent_tmp172) + tangent_tmp11*tangent_tmp416 + tangent_tmp177 + tangent_tmp23*(eta_s*(tangent_tmp24*tangent_tmp422 + tangent_tmp33*tangent_tmp423 - tangent_tmp403) - tangent_tmp348*u1_grad_2 - tangent_tmp421*tangent_tmp56) + tangent_tmp417*tangent_tmp87;
      const s_t tangent_grad_d2_1_grad0_1 = tangent_tmp104*tangent_tmp417 + tangent_tmp23*(-eta_s*(-tangent_tmp422*tangent_tmp59 + tangent_tmp423*tangent_tmp70) + ((s_t(1) / s_t(3)))*tangent_tmp421*tangent_tmp62) + tangent_tmp246 + tangent_tmp416*tangent_tmp93;
      const s_t tangent_grad_d2_1_grad0_2 = tangent_tmp108*tangent_tmp416 + tangent_tmp114*tangent_tmp417 + tangent_tmp23*(-eta_s*(tangent_tmp422*tangent_tmp65 - tangent_tmp423*tangent_tmp69 - tangent_tmp424) + ((s_t(1) / s_t(3)))*tangent_tmp421*tangent_tmp64 - tangent_tmp425) + tangent_tmp289;
      const s_t tangent_grad_d2_1_grad1_0 = tangent_tmp123*tangent_tmp416 + tangent_tmp131*tangent_tmp417 + tangent_tmp23*(-eta_s*(tangent_tmp19*tangent_tmp423 - tangent_tmp24*tangent_tmp427 - tangent_tmp407) - tangent_tmp159*u0_grad_2 + ((s_t(1) / s_t(3)))*tangent_tmp33*tangent_tmp426) + tangent_tmp331;
      const s_t tangent_grad_d2_1_grad1_1 = tangent_tmp137*tangent_tmp416 + tangent_tmp143*tangent_tmp417 + tangent_tmp23*(eta_s*(tangent_tmp423*tangent_tmp62 + tangent_tmp427*tangent_tmp59) - tangent_tmp144*tangent_tmp426) + tangent_tmp368;
      const s_t tangent_grad_d2_1_grad1_2 = tangent_tmp153*tangent_tmp416 + tangent_tmp156*tangent_tmp417 + tangent_tmp23*(-eta_s*(-tangent_tmp423*tangent_tmp64 + tangent_tmp427*tangent_tmp65 + tangent_tmp428) + tangent_tmp157*tangent_tmp426 + tangent_tmp159*tangent_tmp95) + tangent_tmp393;
      const s_t tangent_grad_d2_1_grad2_0 = tangent_tmp167*tangent_tmp416 + tangent_tmp170*tangent_tmp417 + tangent_tmp23*(-eta_s*(tangent_tmp19*tangent_tmp422 - tangent_tmp33*tangent_tmp427 + tangent_tmp414) + ((s_t(1) / s_t(3)))*tangent_tmp24*tangent_tmp429) + tangent_tmp413;
      const s_t tangent_grad_d2_1_grad2_1 = mu*(tangent_tmp410 + tangent_tmp430) + tangent_tmp175*tangent_tmp416 + tangent_tmp178*tangent_tmp417 + tangent_tmp23*(-eta_s*(-tangent_tmp422*tangent_tmp62 + tangent_tmp427*tangent_tmp70) + ((s_t(1) / s_t(3)))*tangent_tmp429*tangent_tmp59);
      const s_t tangent_grad_d2_1_grad2_2 = -mu*tangent_tmp431 + tangent_tmp186*tangent_tmp416 + tangent_tmp190*tangent_tmp417 + tangent_tmp23*(eta_s*(-tangent_tmp226 + tangent_tmp390 + tangent_tmp422*tangent_tmp64 + tangent_tmp427*tangent_tmp69) - tangent_tmp191*tangent_tmp429);
      const s_t tangent_grad_d2_2_grad0_0 = mu*(s_t(2)*tangent_tmp138*tangent_tmp95 - tangent_tmp184) + tangent_tmp11*tangent_tmp432 + tangent_tmp188 + tangent_tmp23*(eta_s*(tangent_tmp24*tangent_tmp438 + tangent_tmp33*tangent_tmp439 + tangent_tmp404) + tangent_tmp348*tangent_tmp5 - tangent_tmp437*tangent_tmp56) + tangent_tmp433*tangent_tmp87;
      const s_t tangent_grad_d2_2_grad0_1 = mu*(tangent_tmp117*tangent_tmp138 + tangent_tmp248) + tangent_tmp104*tangent_tmp433 + tangent_tmp23*(-eta_s*(tangent_tmp424 - tangent_tmp438*tangent_tmp59 + tangent_tmp439*tangent_tmp70) + tangent_tmp302*tangent_tmp437 + tangent_tmp425) + tangent_tmp250 + tangent_tmp432*tangent_tmp93;
      const s_t tangent_grad_d2_2_grad0_2 = tangent_tmp108*tangent_tmp432 + tangent_tmp114*tangent_tmp433 + tangent_tmp23*(-eta_s*(tangent_tmp438*tangent_tmp65 - tangent_tmp439*tangent_tmp69) + ((s_t(1) / s_t(3)))*tangent_tmp437*tangent_tmp64) + tangent_tmp290;
      const s_t tangent_grad_d2_2_grad1_0 = mu*(tangent_tmp138*tangent_tmp99 + tangent_tmp333) + tangent_tmp123*tangent_tmp432 + tangent_tmp131*tangent_tmp433 + tangent_tmp23*(-eta_s*(tangent_tmp19*tangent_tmp439 - tangent_tmp24*tangent_tmp441 + tangent_tmp408) + tangent_tmp231*tangent_tmp440 + tangent_tmp409) + tangent_tmp335;
      const s_t tangent_grad_d2_2_grad1_1 = mu*(s_t(2)*tangent_tmp138*tangent_tmp5 - tangent_tmp370) + tangent_tmp137*tangent_tmp432 + tangent_tmp143*tangent_tmp433 + tangent_tmp23*(eta_s*(tangent_tmp428 + tangent_tmp439*tangent_tmp62 + tangent_tmp441*tangent_tmp59) + tangent_tmp142*tangent_tmp95 - tangent_tmp144*tangent_tmp440) + tangent_tmp371;
      const s_t tangent_grad_d2_2_grad1_2 = tangent_tmp153*tangent_tmp432 + tangent_tmp156*tangent_tmp433 + tangent_tmp23*(-eta_s*(-tangent_tmp439*tangent_tmp64 + tangent_tmp441*tangent_tmp65) + ((s_t(1) / s_t(3)))*tangent_tmp440*tangent_tmp69) + tangent_tmp394;
      const s_t tangent_grad_d2_2_grad2_0 = mu*(s_t(2)*tangent_tmp138*u2_grad_0 - tangent_tmp415 - tangent_tmp442*u2_grad_0) + tangent_tmp167*tangent_tmp432 + tangent_tmp170*tangent_tmp433 + tangent_tmp23*(-eta_s*(-tangent_tmp113 + tangent_tmp19*tangent_tmp438 - tangent_tmp321 - tangent_tmp33*tangent_tmp441) + ((s_t(1) / s_t(3)))*tangent_tmp24*tangent_tmp443);
      const s_t tangent_grad_d2_2_grad2_1 = mu*(s_t(2)*tangent_tmp138*u2_grad_1 - tangent_tmp431 - tangent_tmp442*u2_grad_1) + tangent_tmp175*tangent_tmp432 + tangent_tmp178*tangent_tmp433 + tangent_tmp23*(-eta_s*(-tangent_tmp226 - tangent_tmp363 - tangent_tmp438*tangent_tmp62 + tangent_tmp441*tangent_tmp70) + ((s_t(1) / s_t(3)))*tangent_tmp443*tangent_tmp59);
      const s_t tangent_grad_d2_2_grad2_2 = mu*(tangent_tmp411 + tangent_tmp430 + s_t(2)) + tangent_tmp186*tangent_tmp432 + tangent_tmp190*tangent_tmp433 + tangent_tmp23*(eta_s*(tangent_tmp438*tangent_tmp64 + tangent_tmp441*tangent_tmp69) - tangent_tmp191*tangent_tmp443);
      tangent[0][lane] = tangent_grad_d0_0_grad0_0;
      tangent[1][lane] = tangent_grad_d0_0_grad0_1;
      tangent[2][lane] = tangent_grad_d0_0_grad0_2;
      tangent[3][lane] = tangent_grad_d0_0_grad1_0;
      tangent[4][lane] = tangent_grad_d0_0_grad1_1;
      tangent[5][lane] = tangent_grad_d0_0_grad1_2;
      tangent[6][lane] = tangent_grad_d0_0_grad2_0;
      tangent[7][lane] = tangent_grad_d0_0_grad2_1;
      tangent[8][lane] = tangent_grad_d0_0_grad2_2;
      tangent[9][lane] = tangent_grad_d0_1_grad0_0;
      tangent[10][lane] = tangent_grad_d0_1_grad0_1;
      tangent[11][lane] = tangent_grad_d0_1_grad0_2;
      tangent[12][lane] = tangent_grad_d0_1_grad1_0;
      tangent[13][lane] = tangent_grad_d0_1_grad1_1;
      tangent[14][lane] = tangent_grad_d0_1_grad1_2;
      tangent[15][lane] = tangent_grad_d0_1_grad2_0;
      tangent[16][lane] = tangent_grad_d0_1_grad2_1;
      tangent[17][lane] = tangent_grad_d0_1_grad2_2;
      tangent[18][lane] = tangent_grad_d0_2_grad0_0;
      tangent[19][lane] = tangent_grad_d0_2_grad0_1;
      tangent[20][lane] = tangent_grad_d0_2_grad0_2;
      tangent[21][lane] = tangent_grad_d0_2_grad1_0;
      tangent[22][lane] = tangent_grad_d0_2_grad1_1;
      tangent[23][lane] = tangent_grad_d0_2_grad1_2;
      tangent[24][lane] = tangent_grad_d0_2_grad2_0;
      tangent[25][lane] = tangent_grad_d0_2_grad2_1;
      tangent[26][lane] = tangent_grad_d0_2_grad2_2;
      tangent[27][lane] = tangent_grad_d1_0_grad0_0;
      tangent[28][lane] = tangent_grad_d1_0_grad0_1;
      tangent[29][lane] = tangent_grad_d1_0_grad0_2;
      tangent[30][lane] = tangent_grad_d1_0_grad1_0;
      tangent[31][lane] = tangent_grad_d1_0_grad1_1;
      tangent[32][lane] = tangent_grad_d1_0_grad1_2;
      tangent[33][lane] = tangent_grad_d1_0_grad2_0;
      tangent[34][lane] = tangent_grad_d1_0_grad2_1;
      tangent[35][lane] = tangent_grad_d1_0_grad2_2;
      tangent[36][lane] = tangent_grad_d1_1_grad0_0;
      tangent[37][lane] = tangent_grad_d1_1_grad0_1;
      tangent[38][lane] = tangent_grad_d1_1_grad0_2;
      tangent[39][lane] = tangent_grad_d1_1_grad1_0;
      tangent[40][lane] = tangent_grad_d1_1_grad1_1;
      tangent[41][lane] = tangent_grad_d1_1_grad1_2;
      tangent[42][lane] = tangent_grad_d1_1_grad2_0;
      tangent[43][lane] = tangent_grad_d1_1_grad2_1;
      tangent[44][lane] = tangent_grad_d1_1_grad2_2;
      tangent[45][lane] = tangent_grad_d1_2_grad0_0;
      tangent[46][lane] = tangent_grad_d1_2_grad0_1;
      tangent[47][lane] = tangent_grad_d1_2_grad0_2;
      tangent[48][lane] = tangent_grad_d1_2_grad1_0;
      tangent[49][lane] = tangent_grad_d1_2_grad1_1;
      tangent[50][lane] = tangent_grad_d1_2_grad1_2;
      tangent[51][lane] = tangent_grad_d1_2_grad2_0;
      tangent[52][lane] = tangent_grad_d1_2_grad2_1;
      tangent[53][lane] = tangent_grad_d1_2_grad2_2;
      tangent[54][lane] = tangent_grad_d2_0_grad0_0;
      tangent[55][lane] = tangent_grad_d2_0_grad0_1;
      tangent[56][lane] = tangent_grad_d2_0_grad0_2;
      tangent[57][lane] = tangent_grad_d2_0_grad1_0;
      tangent[58][lane] = tangent_grad_d2_0_grad1_1;
      tangent[59][lane] = tangent_grad_d2_0_grad1_2;
      tangent[60][lane] = tangent_grad_d2_0_grad2_0;
      tangent[61][lane] = tangent_grad_d2_0_grad2_1;
      tangent[62][lane] = tangent_grad_d2_0_grad2_2;
      tangent[63][lane] = tangent_grad_d2_1_grad0_0;
      tangent[64][lane] = tangent_grad_d2_1_grad0_1;
      tangent[65][lane] = tangent_grad_d2_1_grad0_2;
      tangent[66][lane] = tangent_grad_d2_1_grad1_0;
      tangent[67][lane] = tangent_grad_d2_1_grad1_1;
      tangent[68][lane] = tangent_grad_d2_1_grad1_2;
      tangent[69][lane] = tangent_grad_d2_1_grad2_0;
      tangent[70][lane] = tangent_grad_d2_1_grad2_1;
      tangent[71][lane] = tangent_grad_d2_1_grad2_2;
      tangent[72][lane] = tangent_grad_d2_2_grad0_0;
      tangent[73][lane] = tangent_grad_d2_2_grad0_1;
      tangent[74][lane] = tangent_grad_d2_2_grad0_2;
      tangent[75][lane] = tangent_grad_d2_2_grad1_0;
      tangent[76][lane] = tangent_grad_d2_2_grad1_1;
      tangent[77][lane] = tangent_grad_d2_2_grad1_2;
      tangent[78][lane] = tangent_grad_d2_2_grad2_0;
      tangent[79][lane] = tangent_grad_d2_2_grad2_1;
      tangent[80][lane] = tangent_grad_d2_2_grad2_2;
    }
    for (int trial = 0; trial < NS; ++trial) {
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
        const s_t trial_grad0 = (grad_ref_x[q * NS + trial] * adj0 + grad_ref_y[q * NS + trial] * adj3 + grad_ref_z[q * NS + trial] * adj6) / det;
        const s_t trial_grad1 = (grad_ref_x[q * NS + trial] * adj1 + grad_ref_y[q * NS + trial] * adj4 + grad_ref_z[q * NS + trial] * adj7) / det;
        const s_t trial_grad2 = (grad_ref_x[q * NS + trial] * adj2 + grad_ref_y[q * NS + trial] * adj5 + grad_ref_z[q * NS + trial] * adj8) / det;
        const s_t grad_coeff0_0 = trial_grad0 * tangent[0][lane] + trial_grad1 * tangent[9][lane] + trial_grad2 * tangent[18][lane];
        const s_t grad_coeff0_1 = trial_grad0 * tangent[1][lane] + trial_grad1 * tangent[10][lane] + trial_grad2 * tangent[19][lane];
        const s_t grad_coeff0_2 = trial_grad0 * tangent[2][lane] + trial_grad1 * tangent[11][lane] + trial_grad2 * tangent[20][lane];
        const s_t grad_coeff1_0 = trial_grad0 * tangent[3][lane] + trial_grad1 * tangent[12][lane] + trial_grad2 * tangent[21][lane];
        const s_t grad_coeff1_1 = trial_grad0 * tangent[4][lane] + trial_grad1 * tangent[13][lane] + trial_grad2 * tangent[22][lane];
        const s_t grad_coeff1_2 = trial_grad0 * tangent[5][lane] + trial_grad1 * tangent[14][lane] + trial_grad2 * tangent[23][lane];
        const s_t grad_coeff2_0 = trial_grad0 * tangent[6][lane] + trial_grad1 * tangent[15][lane] + trial_grad2 * tangent[24][lane];
        const s_t grad_coeff2_1 = trial_grad0 * tangent[7][lane] + trial_grad1 * tangent[16][lane] + trial_grad2 * tangent[25][lane];
        const s_t grad_coeff2_2 = trial_grad0 * tangent[8][lane] + trial_grad1 * tangent[17][lane] + trial_grad2 * tangent[26][lane];
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
          element_matrix[(0 * NS + test) * 3 * NS + 0 * NS + trial] += q_weight[q] * det * (grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
          element_matrix[(1 * NS + test) * 3 * NS + 0 * NS + trial] += q_weight[q] * det * (grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
          element_matrix[(2 * NS + test) * 3 * NS + 0 * NS + trial] += q_weight[q] * det * (grad_coeff2_0_values[lane] * test_grad0 + grad_coeff2_1_values[lane] * test_grad1 + grad_coeff2_2_values[lane] * test_grad2);
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
        const s_t trial_grad0 = (grad_ref_x[q * NS + trial] * adj0 + grad_ref_y[q * NS + trial] * adj3 + grad_ref_z[q * NS + trial] * adj6) / det;
        const s_t trial_grad1 = (grad_ref_x[q * NS + trial] * adj1 + grad_ref_y[q * NS + trial] * adj4 + grad_ref_z[q * NS + trial] * adj7) / det;
        const s_t trial_grad2 = (grad_ref_x[q * NS + trial] * adj2 + grad_ref_y[q * NS + trial] * adj5 + grad_ref_z[q * NS + trial] * adj8) / det;
        const s_t grad_coeff0_0 = trial_grad0 * tangent[27][lane] + trial_grad1 * tangent[36][lane] + trial_grad2 * tangent[45][lane];
        const s_t grad_coeff0_1 = trial_grad0 * tangent[28][lane] + trial_grad1 * tangent[37][lane] + trial_grad2 * tangent[46][lane];
        const s_t grad_coeff0_2 = trial_grad0 * tangent[29][lane] + trial_grad1 * tangent[38][lane] + trial_grad2 * tangent[47][lane];
        const s_t grad_coeff1_0 = trial_grad0 * tangent[30][lane] + trial_grad1 * tangent[39][lane] + trial_grad2 * tangent[48][lane];
        const s_t grad_coeff1_1 = trial_grad0 * tangent[31][lane] + trial_grad1 * tangent[40][lane] + trial_grad2 * tangent[49][lane];
        const s_t grad_coeff1_2 = trial_grad0 * tangent[32][lane] + trial_grad1 * tangent[41][lane] + trial_grad2 * tangent[50][lane];
        const s_t grad_coeff2_0 = trial_grad0 * tangent[33][lane] + trial_grad1 * tangent[42][lane] + trial_grad2 * tangent[51][lane];
        const s_t grad_coeff2_1 = trial_grad0 * tangent[34][lane] + trial_grad1 * tangent[43][lane] + trial_grad2 * tangent[52][lane];
        const s_t grad_coeff2_2 = trial_grad0 * tangent[35][lane] + trial_grad1 * tangent[44][lane] + trial_grad2 * tangent[53][lane];
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
          element_matrix[(0 * NS + test) * 3 * NS + 1 * NS + trial] += q_weight[q] * det * (grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
          element_matrix[(1 * NS + test) * 3 * NS + 1 * NS + trial] += q_weight[q] * det * (grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
          element_matrix[(2 * NS + test) * 3 * NS + 1 * NS + trial] += q_weight[q] * det * (grad_coeff2_0_values[lane] * test_grad0 + grad_coeff2_1_values[lane] * test_grad1 + grad_coeff2_2_values[lane] * test_grad2);
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
        const s_t trial_grad0 = (grad_ref_x[q * NS + trial] * adj0 + grad_ref_y[q * NS + trial] * adj3 + grad_ref_z[q * NS + trial] * adj6) / det;
        const s_t trial_grad1 = (grad_ref_x[q * NS + trial] * adj1 + grad_ref_y[q * NS + trial] * adj4 + grad_ref_z[q * NS + trial] * adj7) / det;
        const s_t trial_grad2 = (grad_ref_x[q * NS + trial] * adj2 + grad_ref_y[q * NS + trial] * adj5 + grad_ref_z[q * NS + trial] * adj8) / det;
        const s_t grad_coeff0_0 = trial_grad0 * tangent[54][lane] + trial_grad1 * tangent[63][lane] + trial_grad2 * tangent[72][lane];
        const s_t grad_coeff0_1 = trial_grad0 * tangent[55][lane] + trial_grad1 * tangent[64][lane] + trial_grad2 * tangent[73][lane];
        const s_t grad_coeff0_2 = trial_grad0 * tangent[56][lane] + trial_grad1 * tangent[65][lane] + trial_grad2 * tangent[74][lane];
        const s_t grad_coeff1_0 = trial_grad0 * tangent[57][lane] + trial_grad1 * tangent[66][lane] + trial_grad2 * tangent[75][lane];
        const s_t grad_coeff1_1 = trial_grad0 * tangent[58][lane] + trial_grad1 * tangent[67][lane] + trial_grad2 * tangent[76][lane];
        const s_t grad_coeff1_2 = trial_grad0 * tangent[59][lane] + trial_grad1 * tangent[68][lane] + trial_grad2 * tangent[77][lane];
        const s_t grad_coeff2_0 = trial_grad0 * tangent[60][lane] + trial_grad1 * tangent[69][lane] + trial_grad2 * tangent[78][lane];
        const s_t grad_coeff2_1 = trial_grad0 * tangent[61][lane] + trial_grad1 * tangent[70][lane] + trial_grad2 * tangent[79][lane];
        const s_t grad_coeff2_2 = trial_grad0 * tangent[62][lane] + trial_grad1 * tangent[71][lane] + trial_grad2 * tangent[80][lane];
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
          element_matrix[(0 * NS + test) * 3 * NS + 2 * NS + trial] += q_weight[q] * det * (grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
          element_matrix[(1 * NS + test) * 3 * NS + 2 * NS + trial] += q_weight[q] * det * (grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
          element_matrix[(2 * NS + test) * 3 * NS + 2 * NS + trial] += q_weight[q] * det * (grad_coeff2_0_values[lane] * test_grad0 + grad_coeff2_1_values[lane] * test_grad1 + grad_coeff2_2_values[lane] * test_grad2);
        }
      }
    }
  }
}

} // namespace codegen
} // namespace sfem

#endif
