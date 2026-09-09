#ifndef TWO_PHASE_FLOW_D3_SIMPLEX_LOCAL_HPP
#define TWO_PHASE_FLOW_D3_SIMPLEX_LOCAL_HPP

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
typedef ptrdiff_t count_t;
typedef double geom_t;
#endif

namespace sfem {
namespace codegen {

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_d3_simplex_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[2 * NS],
    const s_t *const RSTR previous[2 * NS],
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t C_kw1,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t K_4,
    const s_t K_5,
    const s_t K_6,
    const s_t K_7,
    const s_t K_8,
    const s_t M_c,
    const s_t P_r,
    const s_t R,
    const s_t S_res,
    const s_t T,
    const s_t Z,
    const s_t dt,
    const s_t kappa_T,
    const s_t m,
    const s_t mu_c,
    const s_t mu_w,
    const s_t p_wr,
    const s_t porosity,
    const s_t rho_w0,
    s_t *const RSTR output[2 * NS]
) {
  static constexpr int ND = 3;
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t p_w_values[VS];
    s_t p_w_grad_0_ref_values[VS];
    s_t p_w_grad_1_ref_values[VS];
    s_t p_w_grad_2_ref_values[VS];
    s_t p_w_old_values[VS];
    s_t p_c_values[VS];
    s_t p_c_grad_0_ref_values[VS];
    s_t p_c_grad_1_ref_values[VS];
    s_t p_c_grad_2_ref_values[VS];
    s_t p_c_old_values[VS];
    s_t value_coeff0_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff0_2_values[VS];
    s_t value_coeff1_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    s_t grad_coeff1_2_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC][lane];
        p_w_values[lane] += coeff * shape[q * NS + trial];
        p_w_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_w_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_w_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_old_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC][lane];
        p_w_old_values[lane] += coeff * shape[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC + 1][lane];
        p_c_values[lane] += coeff * shape[q * NS + trial];
        p_c_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_c_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_c_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_old_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC + 1][lane];
        p_c_old_values[lane] += coeff * shape[q * NS + trial];
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
      const s_t p_w = p_w_values[lane];
      const s_t p_w_grad_0_ref = p_w_grad_0_ref_values[lane];
      const s_t p_w_grad_1_ref = p_w_grad_1_ref_values[lane];
      const s_t p_w_grad_2_ref = p_w_grad_2_ref_values[lane];
      const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
      const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
      const s_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
      const s_t p_w_old = p_w_old_values[lane];
      const s_t p_c = p_c_values[lane];
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
      const s_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
      const s_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
      const s_t p_c_old = p_c_old_values[lane];
      const s_t residual_tmp0 = -p_wr;
      const s_t residual_tmp1 = exp(kappa_T*(p_w + residual_tmp0));
      const s_t residual_tmp2 = S_res + s_t(-1);
      const s_t residual_tmp3 = -residual_tmp2;
      const s_t residual_tmp4 = pow_m1(P_r);
      const s_t residual_tmp5 = (s_t(1) - m)/m;
      const s_t residual_tmp6 = pow(pow(residual_tmp4*(p_c - p_w), m) + s_t(1), residual_tmp5);
      const s_t residual_tmp7 = pow(pow(residual_tmp4*(p_c_old - p_w_old), m) + s_t(1), residual_tmp5);
      const s_t residual_tmp8 = porosity/dt;
      const s_t residual_tmp9 = residual_tmp2*residual_tmp6;
      const s_t residual_tmp10 = S_res - residual_tmp9;
      const s_t residual_tmp11 = residual_tmp1*sqrt(residual_tmp10)*rho_w0*pow_2(pow(s_t(1) - pow(residual_tmp10, pow_m1(C_kw1)), C_kw1) + s_t(-1))/mu_w;
      const s_t residual_tmp12 = s_t(1) - S_res;
      const s_t residual_tmp13 = M_c/(R*T*Z);
      const s_t residual_tmp14 = p_c*residual_tmp13*pow(s_t(1) - residual_tmp6, C_ka1)*(pow(residual_tmp6, C_ka2) + s_t(-1))/mu_c;
      const s_t value_coeff0 = residual_tmp8*rho_w0*(residual_tmp1*(S_res + residual_tmp3*residual_tmp6) - (S_res + residual_tmp3*residual_tmp7)*exp(kappa_T*(p_w_old + residual_tmp0)));
      const s_t grad_coeff0_0 = residual_tmp11*(K_0*p_w_grad_0 + K_1*p_w_grad_1 + K_2*p_w_grad_2);
      const s_t grad_coeff0_1 = residual_tmp11*(K_3*p_w_grad_0 + K_4*p_w_grad_1 + K_5*p_w_grad_2);
      const s_t grad_coeff0_2 = residual_tmp11*(K_6*p_w_grad_0 + K_7*p_w_grad_1 + K_8*p_w_grad_2);
      const s_t value_coeff1 = -residual_tmp13*residual_tmp8*(-p_c*(residual_tmp12 + residual_tmp9) + p_c_old*(residual_tmp12 + residual_tmp2*residual_tmp7));
      const s_t grad_coeff1_0 = residual_tmp14*(-K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2);
      const s_t grad_coeff1_1 = residual_tmp14*(-K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2);
      const s_t grad_coeff1_2 = residual_tmp14*(-K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2);
      value_coeff0_values[lane] = value_coeff0;
      grad_coeff0_0_values[lane] = grad_coeff0_0;
      grad_coeff0_1_values[lane] = grad_coeff0_1;
      grad_coeff0_2_values[lane] = grad_coeff0_2;
      value_coeff1_values[lane] = value_coeff1;
      grad_coeff1_0_values[lane] = grad_coeff1_0;
      grad_coeff1_1_values[lane] = grad_coeff1_1;
      grad_coeff1_2_values[lane] = grad_coeff1_2;
    }
    for (int test = 0; test < NS; ++test) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        const s_t test_value = shape[q * NS + test];
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
        output[test * NC][lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value + grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
        output[test * NC + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_d3_simplex_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t current[2 * NS][VS],
    const s_t previous[2 * NS][VS],
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t C_kw1,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t K_4,
    const s_t K_5,
    const s_t K_6,
    const s_t K_7,
    const s_t K_8,
    const s_t M_c,
    const s_t P_r,
    const s_t R,
    const s_t S_res,
    const s_t T,
    const s_t Z,
    const s_t dt,
    const s_t kappa_T,
    const s_t m,
    const s_t mu_c,
    const s_t mu_w,
    const s_t p_wr,
    const s_t porosity,
    const s_t rho_w0,
    s_t output[2 * NS][VS]
) {
  static constexpr int ND = 3;
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t p_w_values[VS];
    s_t p_w_grad_0_ref_values[VS];
    s_t p_w_grad_1_ref_values[VS];
    s_t p_w_grad_2_ref_values[VS];
    s_t p_w_old_values[VS];
    s_t p_c_values[VS];
    s_t p_c_grad_0_ref_values[VS];
    s_t p_c_grad_1_ref_values[VS];
    s_t p_c_grad_2_ref_values[VS];
    s_t p_c_old_values[VS];
    s_t value_coeff0_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff0_2_values[VS];
    s_t value_coeff1_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    s_t grad_coeff1_2_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC][lane];
        p_w_values[lane] += coeff * shape[q * NS + trial];
        p_w_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_w_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_w_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_old_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC][lane];
        p_w_old_values[lane] += coeff * shape[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC + 1][lane];
        p_c_values[lane] += coeff * shape[q * NS + trial];
        p_c_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_c_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_c_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_old_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC + 1][lane];
        p_c_old_values[lane] += coeff * shape[q * NS + trial];
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
      const s_t p_w = p_w_values[lane];
      const s_t p_w_grad_0_ref = p_w_grad_0_ref_values[lane];
      const s_t p_w_grad_1_ref = p_w_grad_1_ref_values[lane];
      const s_t p_w_grad_2_ref = p_w_grad_2_ref_values[lane];
      const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
      const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
      const s_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
      const s_t p_w_old = p_w_old_values[lane];
      const s_t p_c = p_c_values[lane];
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
      const s_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
      const s_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
      const s_t p_c_old = p_c_old_values[lane];
      const s_t residual_tmp0 = -p_wr;
      const s_t residual_tmp1 = exp(kappa_T*(p_w + residual_tmp0));
      const s_t residual_tmp2 = S_res + s_t(-1);
      const s_t residual_tmp3 = -residual_tmp2;
      const s_t residual_tmp4 = pow_m1(P_r);
      const s_t residual_tmp5 = (s_t(1) - m)/m;
      const s_t residual_tmp6 = pow(pow(residual_tmp4*(p_c - p_w), m) + s_t(1), residual_tmp5);
      const s_t residual_tmp7 = pow(pow(residual_tmp4*(p_c_old - p_w_old), m) + s_t(1), residual_tmp5);
      const s_t residual_tmp8 = porosity/dt;
      const s_t residual_tmp9 = residual_tmp2*residual_tmp6;
      const s_t residual_tmp10 = S_res - residual_tmp9;
      const s_t residual_tmp11 = residual_tmp1*sqrt(residual_tmp10)*rho_w0*pow_2(pow(s_t(1) - pow(residual_tmp10, pow_m1(C_kw1)), C_kw1) + s_t(-1))/mu_w;
      const s_t residual_tmp12 = s_t(1) - S_res;
      const s_t residual_tmp13 = M_c/(R*T*Z);
      const s_t residual_tmp14 = p_c*residual_tmp13*pow(s_t(1) - residual_tmp6, C_ka1)*(pow(residual_tmp6, C_ka2) + s_t(-1))/mu_c;
      const s_t value_coeff0 = residual_tmp8*rho_w0*(residual_tmp1*(S_res + residual_tmp3*residual_tmp6) - (S_res + residual_tmp3*residual_tmp7)*exp(kappa_T*(p_w_old + residual_tmp0)));
      const s_t grad_coeff0_0 = residual_tmp11*(K_0*p_w_grad_0 + K_1*p_w_grad_1 + K_2*p_w_grad_2);
      const s_t grad_coeff0_1 = residual_tmp11*(K_3*p_w_grad_0 + K_4*p_w_grad_1 + K_5*p_w_grad_2);
      const s_t grad_coeff0_2 = residual_tmp11*(K_6*p_w_grad_0 + K_7*p_w_grad_1 + K_8*p_w_grad_2);
      const s_t value_coeff1 = -residual_tmp13*residual_tmp8*(-p_c*(residual_tmp12 + residual_tmp9) + p_c_old*(residual_tmp12 + residual_tmp2*residual_tmp7));
      const s_t grad_coeff1_0 = residual_tmp14*(-K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2);
      const s_t grad_coeff1_1 = residual_tmp14*(-K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2);
      const s_t grad_coeff1_2 = residual_tmp14*(-K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2);
      value_coeff0_values[lane] = value_coeff0;
      grad_coeff0_0_values[lane] = grad_coeff0_0;
      grad_coeff0_1_values[lane] = grad_coeff0_1;
      grad_coeff0_2_values[lane] = grad_coeff0_2;
      value_coeff1_values[lane] = value_coeff1;
      grad_coeff1_0_values[lane] = grad_coeff1_0;
      grad_coeff1_1_values[lane] = grad_coeff1_1;
      grad_coeff1_2_values[lane] = grad_coeff1_2;
    }
    for (int test = 0; test < NS; ++test) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        const s_t test_value = shape[q * NS + test];
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
        output[test * NC][lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value + grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
        output[test * NC + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_d3_simplex_tet4_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[2 * NS],
    const s_t *const RSTR previous[2 * NS],
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t C_kw1,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t K_4,
    const s_t K_5,
    const s_t K_6,
    const s_t K_7,
    const s_t K_8,
    const s_t M_c,
    const s_t P_r,
    const s_t R,
    const s_t S_res,
    const s_t T,
    const s_t Z,
    const s_t dt,
    const s_t kappa_T,
    const s_t m,
    const s_t mu_c,
    const s_t mu_w,
    const s_t p_wr,
    const s_t porosity,
    const s_t rho_w0,
    s_t *const RSTR output[2 * NS]
) {
  static constexpr int ND = 3;
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t p_w_values[VS];
    s_t p_w_grad_0_ref_values[VS];
    s_t p_w_grad_1_ref_values[VS];
    s_t p_w_grad_2_ref_values[VS];
    s_t p_w_old_values[VS];
    s_t p_c_values[VS];
    s_t p_c_grad_0_ref_values[VS];
    s_t p_c_grad_1_ref_values[VS];
    s_t p_c_grad_2_ref_values[VS];
    s_t p_c_old_values[VS];
    s_t value_coeff0_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff0_2_values[VS];
    s_t value_coeff1_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    s_t grad_coeff1_2_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC][lane];
        p_w_values[lane] += coeff * shape[q * NS + trial];
        p_w_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_w_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_w_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_old_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC][lane];
        p_w_old_values[lane] += coeff * shape[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC + 1][lane];
        p_c_values[lane] += coeff * shape[q * NS + trial];
        p_c_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_c_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_c_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_old_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC + 1][lane];
        p_c_old_values[lane] += coeff * shape[q * NS + trial];
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
      const s_t p_w = p_w_values[lane];
      const s_t p_w_grad_0_ref = p_w_grad_0_ref_values[lane];
      const s_t p_w_grad_1_ref = p_w_grad_1_ref_values[lane];
      const s_t p_w_grad_2_ref = p_w_grad_2_ref_values[lane];
      const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
      const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
      const s_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
      const s_t p_w_old = p_w_old_values[lane];
      const s_t p_c = p_c_values[lane];
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
      const s_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
      const s_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
      const s_t p_c_old = p_c_old_values[lane];
      const s_t residual_tmp0 = -p_wr;
      const s_t residual_tmp1 = exp(kappa_T*(p_w + residual_tmp0));
      const s_t residual_tmp2 = S_res + s_t(-1);
      const s_t residual_tmp3 = -residual_tmp2;
      const s_t residual_tmp4 = pow_m1(P_r);
      const s_t residual_tmp5 = (s_t(1) - m)/m;
      const s_t residual_tmp6 = pow(pow(residual_tmp4*(p_c - p_w), m) + s_t(1), residual_tmp5);
      const s_t residual_tmp7 = pow(pow(residual_tmp4*(p_c_old - p_w_old), m) + s_t(1), residual_tmp5);
      const s_t residual_tmp8 = porosity/dt;
      const s_t residual_tmp9 = residual_tmp2*residual_tmp6;
      const s_t residual_tmp10 = S_res - residual_tmp9;
      const s_t residual_tmp11 = residual_tmp1*sqrt(residual_tmp10)*rho_w0*pow_2(pow(s_t(1) - pow(residual_tmp10, pow_m1(C_kw1)), C_kw1) + s_t(-1))/mu_w;
      const s_t residual_tmp12 = s_t(1) - S_res;
      const s_t residual_tmp13 = M_c/(R*T*Z);
      const s_t residual_tmp14 = p_c*residual_tmp13*pow(s_t(1) - residual_tmp6, C_ka1)*(pow(residual_tmp6, C_ka2) + s_t(-1))/mu_c;
      const s_t value_coeff0 = residual_tmp8*rho_w0*(residual_tmp1*(S_res + residual_tmp3*residual_tmp6) - (S_res + residual_tmp3*residual_tmp7)*exp(kappa_T*(p_w_old + residual_tmp0)));
      const s_t grad_coeff0_0 = residual_tmp11*(K_0*p_w_grad_0 + K_1*p_w_grad_1 + K_2*p_w_grad_2);
      const s_t grad_coeff0_1 = residual_tmp11*(K_3*p_w_grad_0 + K_4*p_w_grad_1 + K_5*p_w_grad_2);
      const s_t grad_coeff0_2 = residual_tmp11*(K_6*p_w_grad_0 + K_7*p_w_grad_1 + K_8*p_w_grad_2);
      const s_t value_coeff1 = -residual_tmp13*residual_tmp8*(-p_c*(residual_tmp12 + residual_tmp9) + p_c_old*(residual_tmp12 + residual_tmp2*residual_tmp7));
      const s_t grad_coeff1_0 = residual_tmp14*(-K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2);
      const s_t grad_coeff1_1 = residual_tmp14*(-K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2);
      const s_t grad_coeff1_2 = residual_tmp14*(-K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2);
      value_coeff0_values[lane] = value_coeff0;
      grad_coeff0_0_values[lane] = grad_coeff0_0;
      grad_coeff0_1_values[lane] = grad_coeff0_1;
      grad_coeff0_2_values[lane] = grad_coeff0_2;
      value_coeff1_values[lane] = value_coeff1;
      grad_coeff1_0_values[lane] = grad_coeff1_0;
      grad_coeff1_1_values[lane] = grad_coeff1_1;
      grad_coeff1_2_values[lane] = grad_coeff1_2;
    }
    for (int test = 0; test < NS; ++test) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        const s_t test_value = shape[q * NS + test];
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
        output[test * NC][lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value + grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
        output[test * NC + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_d3_simplex_tet4_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t current[2 * NS][VS],
    const s_t previous[2 * NS][VS],
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t C_kw1,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t K_4,
    const s_t K_5,
    const s_t K_6,
    const s_t K_7,
    const s_t K_8,
    const s_t M_c,
    const s_t P_r,
    const s_t R,
    const s_t S_res,
    const s_t T,
    const s_t Z,
    const s_t dt,
    const s_t kappa_T,
    const s_t m,
    const s_t mu_c,
    const s_t mu_w,
    const s_t p_wr,
    const s_t porosity,
    const s_t rho_w0,
    s_t output[2 * NS][VS]
) {
  static constexpr int ND = 3;
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t p_w_values[VS];
    s_t p_w_grad_0_ref_values[VS];
    s_t p_w_grad_1_ref_values[VS];
    s_t p_w_grad_2_ref_values[VS];
    s_t p_w_old_values[VS];
    s_t p_c_values[VS];
    s_t p_c_grad_0_ref_values[VS];
    s_t p_c_grad_1_ref_values[VS];
    s_t p_c_grad_2_ref_values[VS];
    s_t p_c_old_values[VS];
    s_t value_coeff0_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff0_2_values[VS];
    s_t value_coeff1_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    s_t grad_coeff1_2_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC][lane];
        p_w_values[lane] += coeff * shape[q * NS + trial];
        p_w_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_w_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_w_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_old_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC][lane];
        p_w_old_values[lane] += coeff * shape[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC + 1][lane];
        p_c_values[lane] += coeff * shape[q * NS + trial];
        p_c_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_c_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_c_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_old_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = previous[trial * NC + 1][lane];
        p_c_old_values[lane] += coeff * shape[q * NS + trial];
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
      const s_t p_w = p_w_values[lane];
      const s_t p_w_grad_0_ref = p_w_grad_0_ref_values[lane];
      const s_t p_w_grad_1_ref = p_w_grad_1_ref_values[lane];
      const s_t p_w_grad_2_ref = p_w_grad_2_ref_values[lane];
      const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
      const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
      const s_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
      const s_t p_w_old = p_w_old_values[lane];
      const s_t p_c = p_c_values[lane];
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
      const s_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
      const s_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
      const s_t p_c_old = p_c_old_values[lane];
      const s_t residual_tmp0 = -p_wr;
      const s_t residual_tmp1 = exp(kappa_T*(p_w + residual_tmp0));
      const s_t residual_tmp2 = S_res + s_t(-1);
      const s_t residual_tmp3 = -residual_tmp2;
      const s_t residual_tmp4 = pow_m1(P_r);
      const s_t residual_tmp5 = (s_t(1) - m)/m;
      const s_t residual_tmp6 = pow(pow(residual_tmp4*(p_c - p_w), m) + s_t(1), residual_tmp5);
      const s_t residual_tmp7 = pow(pow(residual_tmp4*(p_c_old - p_w_old), m) + s_t(1), residual_tmp5);
      const s_t residual_tmp8 = porosity/dt;
      const s_t residual_tmp9 = residual_tmp2*residual_tmp6;
      const s_t residual_tmp10 = S_res - residual_tmp9;
      const s_t residual_tmp11 = residual_tmp1*sqrt(residual_tmp10)*rho_w0*pow_2(pow(s_t(1) - pow(residual_tmp10, pow_m1(C_kw1)), C_kw1) + s_t(-1))/mu_w;
      const s_t residual_tmp12 = s_t(1) - S_res;
      const s_t residual_tmp13 = M_c/(R*T*Z);
      const s_t residual_tmp14 = p_c*residual_tmp13*pow(s_t(1) - residual_tmp6, C_ka1)*(pow(residual_tmp6, C_ka2) + s_t(-1))/mu_c;
      const s_t value_coeff0 = residual_tmp8*rho_w0*(residual_tmp1*(S_res + residual_tmp3*residual_tmp6) - (S_res + residual_tmp3*residual_tmp7)*exp(kappa_T*(p_w_old + residual_tmp0)));
      const s_t grad_coeff0_0 = residual_tmp11*(K_0*p_w_grad_0 + K_1*p_w_grad_1 + K_2*p_w_grad_2);
      const s_t grad_coeff0_1 = residual_tmp11*(K_3*p_w_grad_0 + K_4*p_w_grad_1 + K_5*p_w_grad_2);
      const s_t grad_coeff0_2 = residual_tmp11*(K_6*p_w_grad_0 + K_7*p_w_grad_1 + K_8*p_w_grad_2);
      const s_t value_coeff1 = -residual_tmp13*residual_tmp8*(-p_c*(residual_tmp12 + residual_tmp9) + p_c_old*(residual_tmp12 + residual_tmp2*residual_tmp7));
      const s_t grad_coeff1_0 = residual_tmp14*(-K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2);
      const s_t grad_coeff1_1 = residual_tmp14*(-K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2);
      const s_t grad_coeff1_2 = residual_tmp14*(-K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2);
      value_coeff0_values[lane] = value_coeff0;
      grad_coeff0_0_values[lane] = grad_coeff0_0;
      grad_coeff0_1_values[lane] = grad_coeff0_1;
      grad_coeff0_2_values[lane] = grad_coeff0_2;
      value_coeff1_values[lane] = value_coeff1;
      grad_coeff1_0_values[lane] = grad_coeff1_0;
      grad_coeff1_1_values[lane] = grad_coeff1_1;
      grad_coeff1_2_values[lane] = grad_coeff1_2;
    }
    for (int test = 0; test < NS; ++test) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        const s_t test_value = shape[q * NS + test];
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
        output[test * NC][lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value + grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
        output[test * NC + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_d3_simplex_jacobian_action_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[2 * NS],
    const s_t *const RSTR direction[2 * NS],
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t C_kw1,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t K_4,
    const s_t K_5,
    const s_t K_6,
    const s_t K_7,
    const s_t K_8,
    const s_t M_c,
    const s_t P_r,
    const s_t R,
    const s_t S_res,
    const s_t T,
    const s_t Z,
    const s_t dt,
    const s_t kappa_T,
    const s_t m,
    const s_t mu_c,
    const s_t mu_w,
    const s_t p_wr,
    const s_t porosity,
    const s_t rho_w0,
    s_t *const RSTR output[2 * NS]
) {
  static constexpr int ND = 3;
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t p_w_values[VS];
    s_t p_w_grad_0_ref_values[VS];
    s_t p_w_grad_1_ref_values[VS];
    s_t p_w_grad_2_ref_values[VS];
    s_t p_w_direction_values[VS];
    s_t p_w_direction_grad_0_ref_values[VS];
    s_t p_w_direction_grad_1_ref_values[VS];
    s_t p_w_direction_grad_2_ref_values[VS];
    s_t p_c_values[VS];
    s_t p_c_grad_0_ref_values[VS];
    s_t p_c_grad_1_ref_values[VS];
    s_t p_c_grad_2_ref_values[VS];
    s_t p_c_direction_values[VS];
    s_t p_c_direction_grad_0_ref_values[VS];
    s_t p_c_direction_grad_1_ref_values[VS];
    s_t p_c_direction_grad_2_ref_values[VS];
    s_t value_coeff0_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff0_2_values[VS];
    s_t value_coeff1_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    s_t grad_coeff1_2_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC][lane];
        p_w_values[lane] += coeff * shape[q * NS + trial];
        p_w_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_w_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_w_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_direction_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_direction_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_direction_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_direction_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = direction[trial * NC][lane];
        p_w_direction_values[lane] += coeff * shape[q * NS + trial];
        p_w_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_w_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_w_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC + 1][lane];
        p_c_values[lane] += coeff * shape[q * NS + trial];
        p_c_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_c_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_c_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_direction_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_direction_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_direction_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_direction_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = direction[trial * NC + 1][lane];
        p_c_direction_values[lane] += coeff * shape[q * NS + trial];
        p_c_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_c_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_c_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
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
      const s_t p_w = p_w_values[lane];
      const s_t p_w_grad_0_ref = p_w_grad_0_ref_values[lane];
      const s_t p_w_grad_1_ref = p_w_grad_1_ref_values[lane];
      const s_t p_w_grad_2_ref = p_w_grad_2_ref_values[lane];
      const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
      const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
      const s_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
      const s_t p_w_direction = p_w_direction_values[lane];
      const s_t p_w_direction_grad_0_ref = p_w_direction_grad_0_ref_values[lane];
      const s_t p_w_direction_grad_1_ref = p_w_direction_grad_1_ref_values[lane];
      const s_t p_w_direction_grad_2_ref = p_w_direction_grad_2_ref_values[lane];
      const s_t p_w_direction_grad_0 = (p_w_direction_grad_0_ref * adj0 + p_w_direction_grad_1_ref * adj3 + p_w_direction_grad_2_ref * adj6) / det;
      const s_t p_w_direction_grad_1 = (p_w_direction_grad_0_ref * adj1 + p_w_direction_grad_1_ref * adj4 + p_w_direction_grad_2_ref * adj7) / det;
      const s_t p_w_direction_grad_2 = (p_w_direction_grad_0_ref * adj2 + p_w_direction_grad_1_ref * adj5 + p_w_direction_grad_2_ref * adj8) / det;
      const s_t p_c = p_c_values[lane];
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
      const s_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
      const s_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
      const s_t p_c_direction = p_c_direction_values[lane];
      const s_t p_c_direction_grad_0_ref = p_c_direction_grad_0_ref_values[lane];
      const s_t p_c_direction_grad_1_ref = p_c_direction_grad_1_ref_values[lane];
      const s_t p_c_direction_grad_2_ref = p_c_direction_grad_2_ref_values[lane];
      const s_t p_c_direction_grad_0 = (p_c_direction_grad_0_ref * adj0 + p_c_direction_grad_1_ref * adj3 + p_c_direction_grad_2_ref * adj6) / det;
      const s_t p_c_direction_grad_1 = (p_c_direction_grad_0_ref * adj1 + p_c_direction_grad_1_ref * adj4 + p_c_direction_grad_2_ref * adj7) / det;
      const s_t p_c_direction_grad_2 = (p_c_direction_grad_0_ref * adj2 + p_c_direction_grad_1_ref * adj5 + p_c_direction_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = pow_m1(dt);
      const s_t residual_tmp1 = residual_tmp0*rho_w0;
      const s_t residual_tmp2 = p_c_direction*residual_tmp1;
      const s_t residual_tmp3 = S_res + s_t(-1);
      const s_t residual_tmp4 = p_c - p_w;
      const s_t residual_tmp5 = pow(residual_tmp4/P_r, m);
      const s_t residual_tmp6 = residual_tmp5 + s_t(1);
      const s_t residual_tmp7 = s_t(1) - m;
      const s_t residual_tmp8 = pow(residual_tmp6, residual_tmp7/m);
      const s_t residual_tmp9 = -residual_tmp3*residual_tmp8;
      const s_t residual_tmp10 = exp(kappa_T*(p_w - p_wr));
      const s_t residual_tmp11 = residual_tmp5*residual_tmp7/(residual_tmp4*residual_tmp6);
      const s_t residual_tmp12 = residual_tmp10*residual_tmp11;
      const s_t residual_tmp13 = residual_tmp12*residual_tmp9;
      const s_t residual_tmp14 = kappa_T*residual_tmp10;
      const s_t residual_tmp15 = p_w_direction*residual_tmp1;
      const s_t residual_tmp16 = pow_m1(mu_w);
      const s_t residual_tmp17 = residual_tmp3*residual_tmp8;
      const s_t residual_tmp18 = S_res - residual_tmp17;
      const s_t residual_tmp19 = pow(residual_tmp18, pow_m1(C_kw1));
      const s_t residual_tmp20 = s_t(1) - residual_tmp19;
      const s_t residual_tmp21 = pow(residual_tmp20, C_kw1);
      const s_t residual_tmp22 = residual_tmp21 + s_t(-1);
      const s_t residual_tmp23 = pow_2(residual_tmp22);
      const s_t residual_tmp24 = sqrt(residual_tmp18);
      const s_t residual_tmp25 = residual_tmp23*residual_tmp24;
      const s_t residual_tmp26 = residual_tmp10*residual_tmp16*residual_tmp25*rho_w0;
      const s_t residual_tmp27 = p_w_direction_grad_0*residual_tmp26;
      const s_t residual_tmp28 = p_w_direction_grad_1*residual_tmp26;
      const s_t residual_tmp29 = p_w_direction_grad_2*residual_tmp26;
      const s_t residual_tmp30 = dt*(K_0*p_w_grad_0 + K_1*p_w_grad_1 + K_2*p_w_grad_2);
      const s_t residual_tmp31 = residual_tmp12*residual_tmp17/residual_tmp24;
      const s_t residual_tmp32 = residual_tmp30*residual_tmp31;
      const s_t residual_tmp33 = ((s_t(1) / s_t(2)))*residual_tmp23;
      const s_t residual_tmp34 = s_t(2)*residual_tmp19*residual_tmp21*residual_tmp22/residual_tmp20;
      const s_t residual_tmp35 = residual_tmp32*residual_tmp33 - residual_tmp32*residual_tmp34;
      const s_t residual_tmp36 = residual_tmp16*residual_tmp2;
      const s_t residual_tmp37 = residual_tmp14*residual_tmp25;
      const s_t residual_tmp38 = residual_tmp15*residual_tmp16;
      const s_t residual_tmp39 = K_3*p_w_grad_0 + K_4*p_w_grad_1 + K_5*p_w_grad_2;
      const s_t residual_tmp40 = dt*residual_tmp31;
      const s_t residual_tmp41 = residual_tmp33*residual_tmp40;
      const s_t residual_tmp42 = residual_tmp34*residual_tmp40;
      const s_t residual_tmp43 = residual_tmp39*residual_tmp41 - residual_tmp39*residual_tmp42;
      const s_t residual_tmp44 = dt*residual_tmp37;
      const s_t residual_tmp45 = K_6*p_w_grad_0 + K_7*p_w_grad_1 + K_8*p_w_grad_2;
      const s_t residual_tmp46 = residual_tmp41*residual_tmp45 - residual_tmp42*residual_tmp45;
      const s_t residual_tmp47 = p_c*residual_tmp11;
      const s_t residual_tmp48 = residual_tmp17*residual_tmp47;
      const s_t residual_tmp49 = pow_m1(R);
      const s_t residual_tmp50 = pow_m1(T);
      const s_t residual_tmp51 = pow_m1(Z);
      const s_t residual_tmp52 = M_c*residual_tmp49*residual_tmp50*residual_tmp51;
      const s_t residual_tmp53 = residual_tmp0*residual_tmp52;
      const s_t residual_tmp54 = pow_m1(mu_c);
      const s_t residual_tmp55 = s_t(1) - residual_tmp8;
      const s_t residual_tmp56 = pow(residual_tmp55, C_ka1);
      const s_t residual_tmp57 = pow(residual_tmp8, C_ka2);
      const s_t residual_tmp58 = residual_tmp56*(residual_tmp57 + s_t(-1));
      const s_t residual_tmp59 = p_c*residual_tmp52*residual_tmp54*residual_tmp58;
      const s_t residual_tmp60 = p_c_direction_grad_0*residual_tmp59;
      const s_t residual_tmp61 = p_c_direction_grad_1*residual_tmp59;
      const s_t residual_tmp62 = p_c_direction_grad_2*residual_tmp59;
      const s_t residual_tmp63 = -K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2;
      const s_t residual_tmp64 = C_ka2*dt*residual_tmp47*residual_tmp56*residual_tmp57;
      const s_t residual_tmp65 = residual_tmp63*residual_tmp64;
      const s_t residual_tmp66 = dt*residual_tmp58;
      const s_t residual_tmp67 = residual_tmp63*residual_tmp66;
      const s_t residual_tmp68 = C_ka1*residual_tmp47*residual_tmp8/residual_tmp55;
      const s_t residual_tmp69 = residual_tmp67*residual_tmp68;
      const s_t residual_tmp70 = -K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2;
      const s_t residual_tmp71 = residual_tmp64*residual_tmp70;
      const s_t residual_tmp72 = residual_tmp66*residual_tmp70;
      const s_t residual_tmp73 = residual_tmp68*residual_tmp72;
      const s_t residual_tmp74 = -K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2;
      const s_t residual_tmp75 = residual_tmp64*residual_tmp74;
      const s_t residual_tmp76 = residual_tmp66*residual_tmp74;
      const s_t residual_tmp77 = residual_tmp68*residual_tmp76;
      const s_t value_coeff0 = porosity*residual_tmp13*residual_tmp2 + porosity*residual_tmp15*(-residual_tmp13 + residual_tmp14*(S_res + residual_tmp9));
      const s_t grad_coeff0_0 = K_0*residual_tmp27 + K_1*residual_tmp28 + K_2*residual_tmp29 - residual_tmp35*residual_tmp36 + residual_tmp38*(residual_tmp30*residual_tmp37 + residual_tmp35);
      const s_t grad_coeff0_1 = K_3*residual_tmp27 + K_4*residual_tmp28 + K_5*residual_tmp29 - residual_tmp36*residual_tmp43 + residual_tmp38*(residual_tmp39*residual_tmp44 + residual_tmp43);
      const s_t grad_coeff0_2 = K_6*residual_tmp27 + K_7*residual_tmp28 + K_8*residual_tmp29 - residual_tmp36*residual_tmp46 + residual_tmp38*(residual_tmp44*residual_tmp45 + residual_tmp46);
      const s_t value_coeff1 = -p_c_direction*porosity*residual_tmp53*(S_res - residual_tmp17 - residual_tmp48 + s_t(-1)) - p_w_direction*porosity*residual_tmp48*residual_tmp53;
      const s_t grad_coeff1_0 = -K_0*residual_tmp60 - K_1*residual_tmp61 - K_2*residual_tmp62 + M_c*p_c_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(residual_tmp65 + residual_tmp67 - residual_tmp69) + M_c*p_w_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(-residual_tmp65 + residual_tmp69);
      const s_t grad_coeff1_1 = -K_3*residual_tmp60 - K_4*residual_tmp61 - K_5*residual_tmp62 + M_c*p_c_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(residual_tmp71 + residual_tmp72 - residual_tmp73) + M_c*p_w_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(-residual_tmp71 + residual_tmp73);
      const s_t grad_coeff1_2 = -K_6*residual_tmp60 - K_7*residual_tmp61 - K_8*residual_tmp62 + M_c*p_c_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(residual_tmp75 + residual_tmp76 - residual_tmp77) + M_c*p_w_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(-residual_tmp75 + residual_tmp77);
      value_coeff0_values[lane] = value_coeff0;
      grad_coeff0_0_values[lane] = grad_coeff0_0;
      grad_coeff0_1_values[lane] = grad_coeff0_1;
      grad_coeff0_2_values[lane] = grad_coeff0_2;
      value_coeff1_values[lane] = value_coeff1;
      grad_coeff1_0_values[lane] = grad_coeff1_0;
      grad_coeff1_1_values[lane] = grad_coeff1_1;
      grad_coeff1_2_values[lane] = grad_coeff1_2;
    }
    for (int test = 0; test < NS; ++test) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        const s_t test_value = shape[q * NS + test];
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
        output[test * NC][lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value + grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
        output[test * NC + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_d3_simplex_jacobian_action_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t current[2 * NS][VS],
    const s_t direction[2 * NS][VS],
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t C_kw1,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t K_4,
    const s_t K_5,
    const s_t K_6,
    const s_t K_7,
    const s_t K_8,
    const s_t M_c,
    const s_t P_r,
    const s_t R,
    const s_t S_res,
    const s_t T,
    const s_t Z,
    const s_t dt,
    const s_t kappa_T,
    const s_t m,
    const s_t mu_c,
    const s_t mu_w,
    const s_t p_wr,
    const s_t porosity,
    const s_t rho_w0,
    s_t output[2 * NS][VS]
) {
  static constexpr int ND = 3;
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t p_w_values[VS];
    s_t p_w_grad_0_ref_values[VS];
    s_t p_w_grad_1_ref_values[VS];
    s_t p_w_grad_2_ref_values[VS];
    s_t p_w_direction_values[VS];
    s_t p_w_direction_grad_0_ref_values[VS];
    s_t p_w_direction_grad_1_ref_values[VS];
    s_t p_w_direction_grad_2_ref_values[VS];
    s_t p_c_values[VS];
    s_t p_c_grad_0_ref_values[VS];
    s_t p_c_grad_1_ref_values[VS];
    s_t p_c_grad_2_ref_values[VS];
    s_t p_c_direction_values[VS];
    s_t p_c_direction_grad_0_ref_values[VS];
    s_t p_c_direction_grad_1_ref_values[VS];
    s_t p_c_direction_grad_2_ref_values[VS];
    s_t value_coeff0_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff0_2_values[VS];
    s_t value_coeff1_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    s_t grad_coeff1_2_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC][lane];
        p_w_values[lane] += coeff * shape[q * NS + trial];
        p_w_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_w_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_w_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_direction_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_direction_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_direction_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_direction_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = direction[trial * NC][lane];
        p_w_direction_values[lane] += coeff * shape[q * NS + trial];
        p_w_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_w_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_w_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC + 1][lane];
        p_c_values[lane] += coeff * shape[q * NS + trial];
        p_c_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_c_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_c_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_direction_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_direction_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_direction_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_direction_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = direction[trial * NC + 1][lane];
        p_c_direction_values[lane] += coeff * shape[q * NS + trial];
        p_c_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_c_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_c_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
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
      const s_t p_w = p_w_values[lane];
      const s_t p_w_grad_0_ref = p_w_grad_0_ref_values[lane];
      const s_t p_w_grad_1_ref = p_w_grad_1_ref_values[lane];
      const s_t p_w_grad_2_ref = p_w_grad_2_ref_values[lane];
      const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
      const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
      const s_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
      const s_t p_w_direction = p_w_direction_values[lane];
      const s_t p_w_direction_grad_0_ref = p_w_direction_grad_0_ref_values[lane];
      const s_t p_w_direction_grad_1_ref = p_w_direction_grad_1_ref_values[lane];
      const s_t p_w_direction_grad_2_ref = p_w_direction_grad_2_ref_values[lane];
      const s_t p_w_direction_grad_0 = (p_w_direction_grad_0_ref * adj0 + p_w_direction_grad_1_ref * adj3 + p_w_direction_grad_2_ref * adj6) / det;
      const s_t p_w_direction_grad_1 = (p_w_direction_grad_0_ref * adj1 + p_w_direction_grad_1_ref * adj4 + p_w_direction_grad_2_ref * adj7) / det;
      const s_t p_w_direction_grad_2 = (p_w_direction_grad_0_ref * adj2 + p_w_direction_grad_1_ref * adj5 + p_w_direction_grad_2_ref * adj8) / det;
      const s_t p_c = p_c_values[lane];
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
      const s_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
      const s_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
      const s_t p_c_direction = p_c_direction_values[lane];
      const s_t p_c_direction_grad_0_ref = p_c_direction_grad_0_ref_values[lane];
      const s_t p_c_direction_grad_1_ref = p_c_direction_grad_1_ref_values[lane];
      const s_t p_c_direction_grad_2_ref = p_c_direction_grad_2_ref_values[lane];
      const s_t p_c_direction_grad_0 = (p_c_direction_grad_0_ref * adj0 + p_c_direction_grad_1_ref * adj3 + p_c_direction_grad_2_ref * adj6) / det;
      const s_t p_c_direction_grad_1 = (p_c_direction_grad_0_ref * adj1 + p_c_direction_grad_1_ref * adj4 + p_c_direction_grad_2_ref * adj7) / det;
      const s_t p_c_direction_grad_2 = (p_c_direction_grad_0_ref * adj2 + p_c_direction_grad_1_ref * adj5 + p_c_direction_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = pow_m1(dt);
      const s_t residual_tmp1 = residual_tmp0*rho_w0;
      const s_t residual_tmp2 = p_c_direction*residual_tmp1;
      const s_t residual_tmp3 = S_res + s_t(-1);
      const s_t residual_tmp4 = p_c - p_w;
      const s_t residual_tmp5 = pow(residual_tmp4/P_r, m);
      const s_t residual_tmp6 = residual_tmp5 + s_t(1);
      const s_t residual_tmp7 = s_t(1) - m;
      const s_t residual_tmp8 = pow(residual_tmp6, residual_tmp7/m);
      const s_t residual_tmp9 = -residual_tmp3*residual_tmp8;
      const s_t residual_tmp10 = exp(kappa_T*(p_w - p_wr));
      const s_t residual_tmp11 = residual_tmp5*residual_tmp7/(residual_tmp4*residual_tmp6);
      const s_t residual_tmp12 = residual_tmp10*residual_tmp11;
      const s_t residual_tmp13 = residual_tmp12*residual_tmp9;
      const s_t residual_tmp14 = kappa_T*residual_tmp10;
      const s_t residual_tmp15 = p_w_direction*residual_tmp1;
      const s_t residual_tmp16 = pow_m1(mu_w);
      const s_t residual_tmp17 = residual_tmp3*residual_tmp8;
      const s_t residual_tmp18 = S_res - residual_tmp17;
      const s_t residual_tmp19 = pow(residual_tmp18, pow_m1(C_kw1));
      const s_t residual_tmp20 = s_t(1) - residual_tmp19;
      const s_t residual_tmp21 = pow(residual_tmp20, C_kw1);
      const s_t residual_tmp22 = residual_tmp21 + s_t(-1);
      const s_t residual_tmp23 = pow_2(residual_tmp22);
      const s_t residual_tmp24 = sqrt(residual_tmp18);
      const s_t residual_tmp25 = residual_tmp23*residual_tmp24;
      const s_t residual_tmp26 = residual_tmp10*residual_tmp16*residual_tmp25*rho_w0;
      const s_t residual_tmp27 = p_w_direction_grad_0*residual_tmp26;
      const s_t residual_tmp28 = p_w_direction_grad_1*residual_tmp26;
      const s_t residual_tmp29 = p_w_direction_grad_2*residual_tmp26;
      const s_t residual_tmp30 = dt*(K_0*p_w_grad_0 + K_1*p_w_grad_1 + K_2*p_w_grad_2);
      const s_t residual_tmp31 = residual_tmp12*residual_tmp17/residual_tmp24;
      const s_t residual_tmp32 = residual_tmp30*residual_tmp31;
      const s_t residual_tmp33 = ((s_t(1) / s_t(2)))*residual_tmp23;
      const s_t residual_tmp34 = s_t(2)*residual_tmp19*residual_tmp21*residual_tmp22/residual_tmp20;
      const s_t residual_tmp35 = residual_tmp32*residual_tmp33 - residual_tmp32*residual_tmp34;
      const s_t residual_tmp36 = residual_tmp16*residual_tmp2;
      const s_t residual_tmp37 = residual_tmp14*residual_tmp25;
      const s_t residual_tmp38 = residual_tmp15*residual_tmp16;
      const s_t residual_tmp39 = K_3*p_w_grad_0 + K_4*p_w_grad_1 + K_5*p_w_grad_2;
      const s_t residual_tmp40 = dt*residual_tmp31;
      const s_t residual_tmp41 = residual_tmp33*residual_tmp40;
      const s_t residual_tmp42 = residual_tmp34*residual_tmp40;
      const s_t residual_tmp43 = residual_tmp39*residual_tmp41 - residual_tmp39*residual_tmp42;
      const s_t residual_tmp44 = dt*residual_tmp37;
      const s_t residual_tmp45 = K_6*p_w_grad_0 + K_7*p_w_grad_1 + K_8*p_w_grad_2;
      const s_t residual_tmp46 = residual_tmp41*residual_tmp45 - residual_tmp42*residual_tmp45;
      const s_t residual_tmp47 = p_c*residual_tmp11;
      const s_t residual_tmp48 = residual_tmp17*residual_tmp47;
      const s_t residual_tmp49 = pow_m1(R);
      const s_t residual_tmp50 = pow_m1(T);
      const s_t residual_tmp51 = pow_m1(Z);
      const s_t residual_tmp52 = M_c*residual_tmp49*residual_tmp50*residual_tmp51;
      const s_t residual_tmp53 = residual_tmp0*residual_tmp52;
      const s_t residual_tmp54 = pow_m1(mu_c);
      const s_t residual_tmp55 = s_t(1) - residual_tmp8;
      const s_t residual_tmp56 = pow(residual_tmp55, C_ka1);
      const s_t residual_tmp57 = pow(residual_tmp8, C_ka2);
      const s_t residual_tmp58 = residual_tmp56*(residual_tmp57 + s_t(-1));
      const s_t residual_tmp59 = p_c*residual_tmp52*residual_tmp54*residual_tmp58;
      const s_t residual_tmp60 = p_c_direction_grad_0*residual_tmp59;
      const s_t residual_tmp61 = p_c_direction_grad_1*residual_tmp59;
      const s_t residual_tmp62 = p_c_direction_grad_2*residual_tmp59;
      const s_t residual_tmp63 = -K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2;
      const s_t residual_tmp64 = C_ka2*dt*residual_tmp47*residual_tmp56*residual_tmp57;
      const s_t residual_tmp65 = residual_tmp63*residual_tmp64;
      const s_t residual_tmp66 = dt*residual_tmp58;
      const s_t residual_tmp67 = residual_tmp63*residual_tmp66;
      const s_t residual_tmp68 = C_ka1*residual_tmp47*residual_tmp8/residual_tmp55;
      const s_t residual_tmp69 = residual_tmp67*residual_tmp68;
      const s_t residual_tmp70 = -K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2;
      const s_t residual_tmp71 = residual_tmp64*residual_tmp70;
      const s_t residual_tmp72 = residual_tmp66*residual_tmp70;
      const s_t residual_tmp73 = residual_tmp68*residual_tmp72;
      const s_t residual_tmp74 = -K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2;
      const s_t residual_tmp75 = residual_tmp64*residual_tmp74;
      const s_t residual_tmp76 = residual_tmp66*residual_tmp74;
      const s_t residual_tmp77 = residual_tmp68*residual_tmp76;
      const s_t value_coeff0 = porosity*residual_tmp13*residual_tmp2 + porosity*residual_tmp15*(-residual_tmp13 + residual_tmp14*(S_res + residual_tmp9));
      const s_t grad_coeff0_0 = K_0*residual_tmp27 + K_1*residual_tmp28 + K_2*residual_tmp29 - residual_tmp35*residual_tmp36 + residual_tmp38*(residual_tmp30*residual_tmp37 + residual_tmp35);
      const s_t grad_coeff0_1 = K_3*residual_tmp27 + K_4*residual_tmp28 + K_5*residual_tmp29 - residual_tmp36*residual_tmp43 + residual_tmp38*(residual_tmp39*residual_tmp44 + residual_tmp43);
      const s_t grad_coeff0_2 = K_6*residual_tmp27 + K_7*residual_tmp28 + K_8*residual_tmp29 - residual_tmp36*residual_tmp46 + residual_tmp38*(residual_tmp44*residual_tmp45 + residual_tmp46);
      const s_t value_coeff1 = -p_c_direction*porosity*residual_tmp53*(S_res - residual_tmp17 - residual_tmp48 + s_t(-1)) - p_w_direction*porosity*residual_tmp48*residual_tmp53;
      const s_t grad_coeff1_0 = -K_0*residual_tmp60 - K_1*residual_tmp61 - K_2*residual_tmp62 + M_c*p_c_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(residual_tmp65 + residual_tmp67 - residual_tmp69) + M_c*p_w_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(-residual_tmp65 + residual_tmp69);
      const s_t grad_coeff1_1 = -K_3*residual_tmp60 - K_4*residual_tmp61 - K_5*residual_tmp62 + M_c*p_c_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(residual_tmp71 + residual_tmp72 - residual_tmp73) + M_c*p_w_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(-residual_tmp71 + residual_tmp73);
      const s_t grad_coeff1_2 = -K_6*residual_tmp60 - K_7*residual_tmp61 - K_8*residual_tmp62 + M_c*p_c_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(residual_tmp75 + residual_tmp76 - residual_tmp77) + M_c*p_w_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(-residual_tmp75 + residual_tmp77);
      value_coeff0_values[lane] = value_coeff0;
      grad_coeff0_0_values[lane] = grad_coeff0_0;
      grad_coeff0_1_values[lane] = grad_coeff0_1;
      grad_coeff0_2_values[lane] = grad_coeff0_2;
      value_coeff1_values[lane] = value_coeff1;
      grad_coeff1_0_values[lane] = grad_coeff1_0;
      grad_coeff1_1_values[lane] = grad_coeff1_1;
      grad_coeff1_2_values[lane] = grad_coeff1_2;
    }
    for (int test = 0; test < NS; ++test) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        const s_t test_value = shape[q * NS + test];
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
        output[test * NC][lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value + grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
        output[test * NC + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_d3_simplex_tet4_jacobian_action_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[2 * NS],
    const s_t *const RSTR direction[2 * NS],
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t C_kw1,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t K_4,
    const s_t K_5,
    const s_t K_6,
    const s_t K_7,
    const s_t K_8,
    const s_t M_c,
    const s_t P_r,
    const s_t R,
    const s_t S_res,
    const s_t T,
    const s_t Z,
    const s_t dt,
    const s_t kappa_T,
    const s_t m,
    const s_t mu_c,
    const s_t mu_w,
    const s_t p_wr,
    const s_t porosity,
    const s_t rho_w0,
    s_t *const RSTR output[2 * NS]
) {
  static constexpr int ND = 3;
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t p_w_values[VS];
    s_t p_w_grad_0_ref_values[VS];
    s_t p_w_grad_1_ref_values[VS];
    s_t p_w_grad_2_ref_values[VS];
    s_t p_w_direction_values[VS];
    s_t p_w_direction_grad_0_ref_values[VS];
    s_t p_w_direction_grad_1_ref_values[VS];
    s_t p_w_direction_grad_2_ref_values[VS];
    s_t p_c_values[VS];
    s_t p_c_grad_0_ref_values[VS];
    s_t p_c_grad_1_ref_values[VS];
    s_t p_c_grad_2_ref_values[VS];
    s_t p_c_direction_values[VS];
    s_t p_c_direction_grad_0_ref_values[VS];
    s_t p_c_direction_grad_1_ref_values[VS];
    s_t p_c_direction_grad_2_ref_values[VS];
    s_t value_coeff0_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff0_2_values[VS];
    s_t value_coeff1_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    s_t grad_coeff1_2_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC][lane];
        p_w_values[lane] += coeff * shape[q * NS + trial];
        p_w_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_w_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_w_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_direction_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_direction_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_direction_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_direction_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = direction[trial * NC][lane];
        p_w_direction_values[lane] += coeff * shape[q * NS + trial];
        p_w_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_w_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_w_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC + 1][lane];
        p_c_values[lane] += coeff * shape[q * NS + trial];
        p_c_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_c_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_c_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_direction_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_direction_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_direction_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_direction_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = direction[trial * NC + 1][lane];
        p_c_direction_values[lane] += coeff * shape[q * NS + trial];
        p_c_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_c_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_c_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
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
      const s_t p_w = p_w_values[lane];
      const s_t p_w_grad_0_ref = p_w_grad_0_ref_values[lane];
      const s_t p_w_grad_1_ref = p_w_grad_1_ref_values[lane];
      const s_t p_w_grad_2_ref = p_w_grad_2_ref_values[lane];
      const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
      const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
      const s_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
      const s_t p_w_direction = p_w_direction_values[lane];
      const s_t p_w_direction_grad_0_ref = p_w_direction_grad_0_ref_values[lane];
      const s_t p_w_direction_grad_1_ref = p_w_direction_grad_1_ref_values[lane];
      const s_t p_w_direction_grad_2_ref = p_w_direction_grad_2_ref_values[lane];
      const s_t p_w_direction_grad_0 = (p_w_direction_grad_0_ref * adj0 + p_w_direction_grad_1_ref * adj3 + p_w_direction_grad_2_ref * adj6) / det;
      const s_t p_w_direction_grad_1 = (p_w_direction_grad_0_ref * adj1 + p_w_direction_grad_1_ref * adj4 + p_w_direction_grad_2_ref * adj7) / det;
      const s_t p_w_direction_grad_2 = (p_w_direction_grad_0_ref * adj2 + p_w_direction_grad_1_ref * adj5 + p_w_direction_grad_2_ref * adj8) / det;
      const s_t p_c = p_c_values[lane];
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
      const s_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
      const s_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
      const s_t p_c_direction = p_c_direction_values[lane];
      const s_t p_c_direction_grad_0_ref = p_c_direction_grad_0_ref_values[lane];
      const s_t p_c_direction_grad_1_ref = p_c_direction_grad_1_ref_values[lane];
      const s_t p_c_direction_grad_2_ref = p_c_direction_grad_2_ref_values[lane];
      const s_t p_c_direction_grad_0 = (p_c_direction_grad_0_ref * adj0 + p_c_direction_grad_1_ref * adj3 + p_c_direction_grad_2_ref * adj6) / det;
      const s_t p_c_direction_grad_1 = (p_c_direction_grad_0_ref * adj1 + p_c_direction_grad_1_ref * adj4 + p_c_direction_grad_2_ref * adj7) / det;
      const s_t p_c_direction_grad_2 = (p_c_direction_grad_0_ref * adj2 + p_c_direction_grad_1_ref * adj5 + p_c_direction_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = pow_m1(dt);
      const s_t residual_tmp1 = residual_tmp0*rho_w0;
      const s_t residual_tmp2 = p_c_direction*residual_tmp1;
      const s_t residual_tmp3 = S_res + s_t(-1);
      const s_t residual_tmp4 = p_c - p_w;
      const s_t residual_tmp5 = pow(residual_tmp4/P_r, m);
      const s_t residual_tmp6 = residual_tmp5 + s_t(1);
      const s_t residual_tmp7 = s_t(1) - m;
      const s_t residual_tmp8 = pow(residual_tmp6, residual_tmp7/m);
      const s_t residual_tmp9 = -residual_tmp3*residual_tmp8;
      const s_t residual_tmp10 = exp(kappa_T*(p_w - p_wr));
      const s_t residual_tmp11 = residual_tmp5*residual_tmp7/(residual_tmp4*residual_tmp6);
      const s_t residual_tmp12 = residual_tmp10*residual_tmp11;
      const s_t residual_tmp13 = residual_tmp12*residual_tmp9;
      const s_t residual_tmp14 = kappa_T*residual_tmp10;
      const s_t residual_tmp15 = p_w_direction*residual_tmp1;
      const s_t residual_tmp16 = pow_m1(mu_w);
      const s_t residual_tmp17 = residual_tmp3*residual_tmp8;
      const s_t residual_tmp18 = S_res - residual_tmp17;
      const s_t residual_tmp19 = pow(residual_tmp18, pow_m1(C_kw1));
      const s_t residual_tmp20 = s_t(1) - residual_tmp19;
      const s_t residual_tmp21 = pow(residual_tmp20, C_kw1);
      const s_t residual_tmp22 = residual_tmp21 + s_t(-1);
      const s_t residual_tmp23 = pow_2(residual_tmp22);
      const s_t residual_tmp24 = sqrt(residual_tmp18);
      const s_t residual_tmp25 = residual_tmp23*residual_tmp24;
      const s_t residual_tmp26 = residual_tmp10*residual_tmp16*residual_tmp25*rho_w0;
      const s_t residual_tmp27 = p_w_direction_grad_0*residual_tmp26;
      const s_t residual_tmp28 = p_w_direction_grad_1*residual_tmp26;
      const s_t residual_tmp29 = p_w_direction_grad_2*residual_tmp26;
      const s_t residual_tmp30 = dt*(K_0*p_w_grad_0 + K_1*p_w_grad_1 + K_2*p_w_grad_2);
      const s_t residual_tmp31 = residual_tmp12*residual_tmp17/residual_tmp24;
      const s_t residual_tmp32 = residual_tmp30*residual_tmp31;
      const s_t residual_tmp33 = ((s_t(1) / s_t(2)))*residual_tmp23;
      const s_t residual_tmp34 = s_t(2)*residual_tmp19*residual_tmp21*residual_tmp22/residual_tmp20;
      const s_t residual_tmp35 = residual_tmp32*residual_tmp33 - residual_tmp32*residual_tmp34;
      const s_t residual_tmp36 = residual_tmp16*residual_tmp2;
      const s_t residual_tmp37 = residual_tmp14*residual_tmp25;
      const s_t residual_tmp38 = residual_tmp15*residual_tmp16;
      const s_t residual_tmp39 = K_3*p_w_grad_0 + K_4*p_w_grad_1 + K_5*p_w_grad_2;
      const s_t residual_tmp40 = dt*residual_tmp31;
      const s_t residual_tmp41 = residual_tmp33*residual_tmp40;
      const s_t residual_tmp42 = residual_tmp34*residual_tmp40;
      const s_t residual_tmp43 = residual_tmp39*residual_tmp41 - residual_tmp39*residual_tmp42;
      const s_t residual_tmp44 = dt*residual_tmp37;
      const s_t residual_tmp45 = K_6*p_w_grad_0 + K_7*p_w_grad_1 + K_8*p_w_grad_2;
      const s_t residual_tmp46 = residual_tmp41*residual_tmp45 - residual_tmp42*residual_tmp45;
      const s_t residual_tmp47 = p_c*residual_tmp11;
      const s_t residual_tmp48 = residual_tmp17*residual_tmp47;
      const s_t residual_tmp49 = pow_m1(R);
      const s_t residual_tmp50 = pow_m1(T);
      const s_t residual_tmp51 = pow_m1(Z);
      const s_t residual_tmp52 = M_c*residual_tmp49*residual_tmp50*residual_tmp51;
      const s_t residual_tmp53 = residual_tmp0*residual_tmp52;
      const s_t residual_tmp54 = pow_m1(mu_c);
      const s_t residual_tmp55 = s_t(1) - residual_tmp8;
      const s_t residual_tmp56 = pow(residual_tmp55, C_ka1);
      const s_t residual_tmp57 = pow(residual_tmp8, C_ka2);
      const s_t residual_tmp58 = residual_tmp56*(residual_tmp57 + s_t(-1));
      const s_t residual_tmp59 = p_c*residual_tmp52*residual_tmp54*residual_tmp58;
      const s_t residual_tmp60 = p_c_direction_grad_0*residual_tmp59;
      const s_t residual_tmp61 = p_c_direction_grad_1*residual_tmp59;
      const s_t residual_tmp62 = p_c_direction_grad_2*residual_tmp59;
      const s_t residual_tmp63 = -K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2;
      const s_t residual_tmp64 = C_ka2*dt*residual_tmp47*residual_tmp56*residual_tmp57;
      const s_t residual_tmp65 = residual_tmp63*residual_tmp64;
      const s_t residual_tmp66 = dt*residual_tmp58;
      const s_t residual_tmp67 = residual_tmp63*residual_tmp66;
      const s_t residual_tmp68 = C_ka1*residual_tmp47*residual_tmp8/residual_tmp55;
      const s_t residual_tmp69 = residual_tmp67*residual_tmp68;
      const s_t residual_tmp70 = -K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2;
      const s_t residual_tmp71 = residual_tmp64*residual_tmp70;
      const s_t residual_tmp72 = residual_tmp66*residual_tmp70;
      const s_t residual_tmp73 = residual_tmp68*residual_tmp72;
      const s_t residual_tmp74 = -K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2;
      const s_t residual_tmp75 = residual_tmp64*residual_tmp74;
      const s_t residual_tmp76 = residual_tmp66*residual_tmp74;
      const s_t residual_tmp77 = residual_tmp68*residual_tmp76;
      const s_t value_coeff0 = porosity*residual_tmp13*residual_tmp2 + porosity*residual_tmp15*(-residual_tmp13 + residual_tmp14*(S_res + residual_tmp9));
      const s_t grad_coeff0_0 = K_0*residual_tmp27 + K_1*residual_tmp28 + K_2*residual_tmp29 - residual_tmp35*residual_tmp36 + residual_tmp38*(residual_tmp30*residual_tmp37 + residual_tmp35);
      const s_t grad_coeff0_1 = K_3*residual_tmp27 + K_4*residual_tmp28 + K_5*residual_tmp29 - residual_tmp36*residual_tmp43 + residual_tmp38*(residual_tmp39*residual_tmp44 + residual_tmp43);
      const s_t grad_coeff0_2 = K_6*residual_tmp27 + K_7*residual_tmp28 + K_8*residual_tmp29 - residual_tmp36*residual_tmp46 + residual_tmp38*(residual_tmp44*residual_tmp45 + residual_tmp46);
      const s_t value_coeff1 = -p_c_direction*porosity*residual_tmp53*(S_res - residual_tmp17 - residual_tmp48 + s_t(-1)) - p_w_direction*porosity*residual_tmp48*residual_tmp53;
      const s_t grad_coeff1_0 = -K_0*residual_tmp60 - K_1*residual_tmp61 - K_2*residual_tmp62 + M_c*p_c_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(residual_tmp65 + residual_tmp67 - residual_tmp69) + M_c*p_w_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(-residual_tmp65 + residual_tmp69);
      const s_t grad_coeff1_1 = -K_3*residual_tmp60 - K_4*residual_tmp61 - K_5*residual_tmp62 + M_c*p_c_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(residual_tmp71 + residual_tmp72 - residual_tmp73) + M_c*p_w_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(-residual_tmp71 + residual_tmp73);
      const s_t grad_coeff1_2 = -K_6*residual_tmp60 - K_7*residual_tmp61 - K_8*residual_tmp62 + M_c*p_c_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(residual_tmp75 + residual_tmp76 - residual_tmp77) + M_c*p_w_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(-residual_tmp75 + residual_tmp77);
      value_coeff0_values[lane] = value_coeff0;
      grad_coeff0_0_values[lane] = grad_coeff0_0;
      grad_coeff0_1_values[lane] = grad_coeff0_1;
      grad_coeff0_2_values[lane] = grad_coeff0_2;
      value_coeff1_values[lane] = value_coeff1;
      grad_coeff1_0_values[lane] = grad_coeff1_0;
      grad_coeff1_1_values[lane] = grad_coeff1_1;
      grad_coeff1_2_values[lane] = grad_coeff1_2;
    }
    for (int test = 0; test < NS; ++test) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        const s_t test_value = shape[q * NS + test];
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
        output[test * NC][lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value + grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
        output[test * NC + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_d3_simplex_tet4_jacobian_action_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR shape,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
    const s_t current[2 * NS][VS],
    const s_t direction[2 * NS][VS],
    const s_t C_ka1,
    const s_t C_ka2,
    const s_t C_kw1,
    const s_t K_0,
    const s_t K_1,
    const s_t K_2,
    const s_t K_3,
    const s_t K_4,
    const s_t K_5,
    const s_t K_6,
    const s_t K_7,
    const s_t K_8,
    const s_t M_c,
    const s_t P_r,
    const s_t R,
    const s_t S_res,
    const s_t T,
    const s_t Z,
    const s_t dt,
    const s_t kappa_T,
    const s_t m,
    const s_t mu_c,
    const s_t mu_w,
    const s_t p_wr,
    const s_t porosity,
    const s_t rho_w0,
    s_t output[2 * NS][VS]
) {
  static constexpr int ND = 3;
  static constexpr int NC = 2;
  for (int q = 0; q < NQ; ++q) {
    s_t p_w_values[VS];
    s_t p_w_grad_0_ref_values[VS];
    s_t p_w_grad_1_ref_values[VS];
    s_t p_w_grad_2_ref_values[VS];
    s_t p_w_direction_values[VS];
    s_t p_w_direction_grad_0_ref_values[VS];
    s_t p_w_direction_grad_1_ref_values[VS];
    s_t p_w_direction_grad_2_ref_values[VS];
    s_t p_c_values[VS];
    s_t p_c_grad_0_ref_values[VS];
    s_t p_c_grad_1_ref_values[VS];
    s_t p_c_grad_2_ref_values[VS];
    s_t p_c_direction_values[VS];
    s_t p_c_direction_grad_0_ref_values[VS];
    s_t p_c_direction_grad_1_ref_values[VS];
    s_t p_c_direction_grad_2_ref_values[VS];
    s_t value_coeff0_values[VS];
    s_t grad_coeff0_0_values[VS];
    s_t grad_coeff0_1_values[VS];
    s_t grad_coeff0_2_values[VS];
    s_t value_coeff1_values[VS];
    s_t grad_coeff1_0_values[VS];
    s_t grad_coeff1_1_values[VS];
    s_t grad_coeff1_2_values[VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC][lane];
        p_w_values[lane] += coeff * shape[q * NS + trial];
        p_w_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_w_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_w_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_direction_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_direction_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_direction_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_w_direction_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = direction[trial * NC][lane];
        p_w_direction_values[lane] += coeff * shape[q * NS + trial];
        p_w_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_w_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_w_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = current[trial * NC + 1][lane];
        p_c_values[lane] += coeff * shape[q * NS + trial];
        p_c_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_c_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_c_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
      }
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_direction_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_direction_grad_0_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_direction_grad_1_ref_values[lane] = s_t(0);
    }
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      p_c_direction_grad_2_ref_values[lane] = s_t(0);
    }
    for (int trial = 0; trial < NS; ++trial) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t coeff = direction[trial * NC + 1][lane];
        p_c_direction_values[lane] += coeff * shape[q * NS + trial];
        p_c_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
        p_c_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
        p_c_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
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
      const s_t p_w = p_w_values[lane];
      const s_t p_w_grad_0_ref = p_w_grad_0_ref_values[lane];
      const s_t p_w_grad_1_ref = p_w_grad_1_ref_values[lane];
      const s_t p_w_grad_2_ref = p_w_grad_2_ref_values[lane];
      const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
      const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
      const s_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
      const s_t p_w_direction = p_w_direction_values[lane];
      const s_t p_w_direction_grad_0_ref = p_w_direction_grad_0_ref_values[lane];
      const s_t p_w_direction_grad_1_ref = p_w_direction_grad_1_ref_values[lane];
      const s_t p_w_direction_grad_2_ref = p_w_direction_grad_2_ref_values[lane];
      const s_t p_w_direction_grad_0 = (p_w_direction_grad_0_ref * adj0 + p_w_direction_grad_1_ref * adj3 + p_w_direction_grad_2_ref * adj6) / det;
      const s_t p_w_direction_grad_1 = (p_w_direction_grad_0_ref * adj1 + p_w_direction_grad_1_ref * adj4 + p_w_direction_grad_2_ref * adj7) / det;
      const s_t p_w_direction_grad_2 = (p_w_direction_grad_0_ref * adj2 + p_w_direction_grad_1_ref * adj5 + p_w_direction_grad_2_ref * adj8) / det;
      const s_t p_c = p_c_values[lane];
      const s_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
      const s_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
      const s_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
      const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
      const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
      const s_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
      const s_t p_c_direction = p_c_direction_values[lane];
      const s_t p_c_direction_grad_0_ref = p_c_direction_grad_0_ref_values[lane];
      const s_t p_c_direction_grad_1_ref = p_c_direction_grad_1_ref_values[lane];
      const s_t p_c_direction_grad_2_ref = p_c_direction_grad_2_ref_values[lane];
      const s_t p_c_direction_grad_0 = (p_c_direction_grad_0_ref * adj0 + p_c_direction_grad_1_ref * adj3 + p_c_direction_grad_2_ref * adj6) / det;
      const s_t p_c_direction_grad_1 = (p_c_direction_grad_0_ref * adj1 + p_c_direction_grad_1_ref * adj4 + p_c_direction_grad_2_ref * adj7) / det;
      const s_t p_c_direction_grad_2 = (p_c_direction_grad_0_ref * adj2 + p_c_direction_grad_1_ref * adj5 + p_c_direction_grad_2_ref * adj8) / det;
      const s_t residual_tmp0 = pow_m1(dt);
      const s_t residual_tmp1 = residual_tmp0*rho_w0;
      const s_t residual_tmp2 = p_c_direction*residual_tmp1;
      const s_t residual_tmp3 = S_res + s_t(-1);
      const s_t residual_tmp4 = p_c - p_w;
      const s_t residual_tmp5 = pow(residual_tmp4/P_r, m);
      const s_t residual_tmp6 = residual_tmp5 + s_t(1);
      const s_t residual_tmp7 = s_t(1) - m;
      const s_t residual_tmp8 = pow(residual_tmp6, residual_tmp7/m);
      const s_t residual_tmp9 = -residual_tmp3*residual_tmp8;
      const s_t residual_tmp10 = exp(kappa_T*(p_w - p_wr));
      const s_t residual_tmp11 = residual_tmp5*residual_tmp7/(residual_tmp4*residual_tmp6);
      const s_t residual_tmp12 = residual_tmp10*residual_tmp11;
      const s_t residual_tmp13 = residual_tmp12*residual_tmp9;
      const s_t residual_tmp14 = kappa_T*residual_tmp10;
      const s_t residual_tmp15 = p_w_direction*residual_tmp1;
      const s_t residual_tmp16 = pow_m1(mu_w);
      const s_t residual_tmp17 = residual_tmp3*residual_tmp8;
      const s_t residual_tmp18 = S_res - residual_tmp17;
      const s_t residual_tmp19 = pow(residual_tmp18, pow_m1(C_kw1));
      const s_t residual_tmp20 = s_t(1) - residual_tmp19;
      const s_t residual_tmp21 = pow(residual_tmp20, C_kw1);
      const s_t residual_tmp22 = residual_tmp21 + s_t(-1);
      const s_t residual_tmp23 = pow_2(residual_tmp22);
      const s_t residual_tmp24 = sqrt(residual_tmp18);
      const s_t residual_tmp25 = residual_tmp23*residual_tmp24;
      const s_t residual_tmp26 = residual_tmp10*residual_tmp16*residual_tmp25*rho_w0;
      const s_t residual_tmp27 = p_w_direction_grad_0*residual_tmp26;
      const s_t residual_tmp28 = p_w_direction_grad_1*residual_tmp26;
      const s_t residual_tmp29 = p_w_direction_grad_2*residual_tmp26;
      const s_t residual_tmp30 = dt*(K_0*p_w_grad_0 + K_1*p_w_grad_1 + K_2*p_w_grad_2);
      const s_t residual_tmp31 = residual_tmp12*residual_tmp17/residual_tmp24;
      const s_t residual_tmp32 = residual_tmp30*residual_tmp31;
      const s_t residual_tmp33 = ((s_t(1) / s_t(2)))*residual_tmp23;
      const s_t residual_tmp34 = s_t(2)*residual_tmp19*residual_tmp21*residual_tmp22/residual_tmp20;
      const s_t residual_tmp35 = residual_tmp32*residual_tmp33 - residual_tmp32*residual_tmp34;
      const s_t residual_tmp36 = residual_tmp16*residual_tmp2;
      const s_t residual_tmp37 = residual_tmp14*residual_tmp25;
      const s_t residual_tmp38 = residual_tmp15*residual_tmp16;
      const s_t residual_tmp39 = K_3*p_w_grad_0 + K_4*p_w_grad_1 + K_5*p_w_grad_2;
      const s_t residual_tmp40 = dt*residual_tmp31;
      const s_t residual_tmp41 = residual_tmp33*residual_tmp40;
      const s_t residual_tmp42 = residual_tmp34*residual_tmp40;
      const s_t residual_tmp43 = residual_tmp39*residual_tmp41 - residual_tmp39*residual_tmp42;
      const s_t residual_tmp44 = dt*residual_tmp37;
      const s_t residual_tmp45 = K_6*p_w_grad_0 + K_7*p_w_grad_1 + K_8*p_w_grad_2;
      const s_t residual_tmp46 = residual_tmp41*residual_tmp45 - residual_tmp42*residual_tmp45;
      const s_t residual_tmp47 = p_c*residual_tmp11;
      const s_t residual_tmp48 = residual_tmp17*residual_tmp47;
      const s_t residual_tmp49 = pow_m1(R);
      const s_t residual_tmp50 = pow_m1(T);
      const s_t residual_tmp51 = pow_m1(Z);
      const s_t residual_tmp52 = M_c*residual_tmp49*residual_tmp50*residual_tmp51;
      const s_t residual_tmp53 = residual_tmp0*residual_tmp52;
      const s_t residual_tmp54 = pow_m1(mu_c);
      const s_t residual_tmp55 = s_t(1) - residual_tmp8;
      const s_t residual_tmp56 = pow(residual_tmp55, C_ka1);
      const s_t residual_tmp57 = pow(residual_tmp8, C_ka2);
      const s_t residual_tmp58 = residual_tmp56*(residual_tmp57 + s_t(-1));
      const s_t residual_tmp59 = p_c*residual_tmp52*residual_tmp54*residual_tmp58;
      const s_t residual_tmp60 = p_c_direction_grad_0*residual_tmp59;
      const s_t residual_tmp61 = p_c_direction_grad_1*residual_tmp59;
      const s_t residual_tmp62 = p_c_direction_grad_2*residual_tmp59;
      const s_t residual_tmp63 = -K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2;
      const s_t residual_tmp64 = C_ka2*dt*residual_tmp47*residual_tmp56*residual_tmp57;
      const s_t residual_tmp65 = residual_tmp63*residual_tmp64;
      const s_t residual_tmp66 = dt*residual_tmp58;
      const s_t residual_tmp67 = residual_tmp63*residual_tmp66;
      const s_t residual_tmp68 = C_ka1*residual_tmp47*residual_tmp8/residual_tmp55;
      const s_t residual_tmp69 = residual_tmp67*residual_tmp68;
      const s_t residual_tmp70 = -K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2;
      const s_t residual_tmp71 = residual_tmp64*residual_tmp70;
      const s_t residual_tmp72 = residual_tmp66*residual_tmp70;
      const s_t residual_tmp73 = residual_tmp68*residual_tmp72;
      const s_t residual_tmp74 = -K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2;
      const s_t residual_tmp75 = residual_tmp64*residual_tmp74;
      const s_t residual_tmp76 = residual_tmp66*residual_tmp74;
      const s_t residual_tmp77 = residual_tmp68*residual_tmp76;
      const s_t value_coeff0 = porosity*residual_tmp13*residual_tmp2 + porosity*residual_tmp15*(-residual_tmp13 + residual_tmp14*(S_res + residual_tmp9));
      const s_t grad_coeff0_0 = K_0*residual_tmp27 + K_1*residual_tmp28 + K_2*residual_tmp29 - residual_tmp35*residual_tmp36 + residual_tmp38*(residual_tmp30*residual_tmp37 + residual_tmp35);
      const s_t grad_coeff0_1 = K_3*residual_tmp27 + K_4*residual_tmp28 + K_5*residual_tmp29 - residual_tmp36*residual_tmp43 + residual_tmp38*(residual_tmp39*residual_tmp44 + residual_tmp43);
      const s_t grad_coeff0_2 = K_6*residual_tmp27 + K_7*residual_tmp28 + K_8*residual_tmp29 - residual_tmp36*residual_tmp46 + residual_tmp38*(residual_tmp44*residual_tmp45 + residual_tmp46);
      const s_t value_coeff1 = -p_c_direction*porosity*residual_tmp53*(S_res - residual_tmp17 - residual_tmp48 + s_t(-1)) - p_w_direction*porosity*residual_tmp48*residual_tmp53;
      const s_t grad_coeff1_0 = -K_0*residual_tmp60 - K_1*residual_tmp61 - K_2*residual_tmp62 + M_c*p_c_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(residual_tmp65 + residual_tmp67 - residual_tmp69) + M_c*p_w_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(-residual_tmp65 + residual_tmp69);
      const s_t grad_coeff1_1 = -K_3*residual_tmp60 - K_4*residual_tmp61 - K_5*residual_tmp62 + M_c*p_c_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(residual_tmp71 + residual_tmp72 - residual_tmp73) + M_c*p_w_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(-residual_tmp71 + residual_tmp73);
      const s_t grad_coeff1_2 = -K_6*residual_tmp60 - K_7*residual_tmp61 - K_8*residual_tmp62 + M_c*p_c_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(residual_tmp75 + residual_tmp76 - residual_tmp77) + M_c*p_w_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(-residual_tmp75 + residual_tmp77);
      value_coeff0_values[lane] = value_coeff0;
      grad_coeff0_0_values[lane] = grad_coeff0_0;
      grad_coeff0_1_values[lane] = grad_coeff0_1;
      grad_coeff0_2_values[lane] = grad_coeff0_2;
      value_coeff1_values[lane] = value_coeff1;
      grad_coeff1_0_values[lane] = grad_coeff1_0;
      grad_coeff1_1_values[lane] = grad_coeff1_1;
      grad_coeff1_2_values[lane] = grad_coeff1_2;
    }
    for (int test = 0; test < NS; ++test) {
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const ptrdiff_t goff = q * geometry_stride + lane;
        const s_t det = determinant[goff];
        const s_t test_value = shape[q * NS + test];
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
        output[test * NC][lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value + grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
        output[test * NC + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
      }
    }
  }
}

} // namespace codegen
} // namespace sfem

#endif
