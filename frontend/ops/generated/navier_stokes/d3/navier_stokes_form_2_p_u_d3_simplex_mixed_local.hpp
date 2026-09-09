#ifndef NAVIER_STOKES_FORM_2_P_U_D3_SIMPLEX_MIXED_LOCAL_HPP
#define NAVIER_STOKES_FORM_2_P_U_D3_SIMPLEX_MIXED_LOCAL_HPP

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

template <typename s_t, int NQ, int CELL_NS, int VS>
static SFEM_INLINE void navier_stokes_form_2_p_u_d3_simplex_mixed_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR field_shape[2],
    const s_t *const RSTR q_weight,
    s_t *const RSTR output[34]
) {
  static constexpr int ND = 3;
  static constexpr int NC = 2;
  static constexpr int N_FIELD_STREAMS = 34;
  (void)CELL_NS;
  (void)N_FIELD_STREAMS;
  static constexpr int U_NS = 10;
  static constexpr int P_NS = 4;
  for (int q = 0; q < NQ; ++q) {
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
      const s_t det = determinant[goff];
    }
  }
}

template <typename s_t, int NQ, int CELL_NS, int VS>
static SFEM_INLINE void navier_stokes_form_2_p_u_d3_simplex_mixed_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR field_shape[2],
    const s_t *const RSTR q_weight,
    s_t output[34][VS]
) {
  static constexpr int ND = 3;
  static constexpr int NC = 2;
  static constexpr int N_FIELD_STREAMS = 34;
  (void)CELL_NS;
  (void)N_FIELD_STREAMS;
  static constexpr int U_NS = 10;
  static constexpr int P_NS = 4;
  for (int q = 0; q < NQ; ++q) {
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
      const s_t det = determinant[goff];
    }
  }
}

template <typename s_t, int NQ, int CELL_NS, int VS>
static SFEM_INLINE void navier_stokes_form_2_p_u_d3_simplex_mixed_jacobian_action_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR field_shape[2],
    const s_t *const RSTR fgref[6],
    const s_t *const RSTR q_weight,
    const s_t *const RSTR direction[34],
    s_t *const RSTR output[34]
) {
  static constexpr int ND = 3;
  static constexpr int NC = 2;
  static constexpr int N_FIELD_STREAMS = 34;
  (void)CELL_NS;
  (void)N_FIELD_STREAMS;
  static constexpr int U_NS = 10;
  static constexpr int P_NS = 4;
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
      s_t u0_direction_grad_0_ref = s_t(0);
      s_t u0_direction_grad_1_ref = s_t(0);
      s_t u0_direction_grad_2_ref = s_t(0);
      const s_t coeff_direction_u0_0 = direction[0][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_0 * fgref[0][q * U_NS];
      u0_direction_grad_1_ref += coeff_direction_u0_0 * fgref[1][q * U_NS];
      u0_direction_grad_2_ref += coeff_direction_u0_0 * fgref[2][q * U_NS];
      const s_t coeff_direction_u0_1 = direction[1][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_1 * fgref[0][q * U_NS + 1];
      u0_direction_grad_1_ref += coeff_direction_u0_1 * fgref[1][q * U_NS + 1];
      u0_direction_grad_2_ref += coeff_direction_u0_1 * fgref[2][q * U_NS + 1];
      const s_t coeff_direction_u0_2 = direction[2][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_2 * fgref[0][q * U_NS + 2];
      u0_direction_grad_1_ref += coeff_direction_u0_2 * fgref[1][q * U_NS + 2];
      u0_direction_grad_2_ref += coeff_direction_u0_2 * fgref[2][q * U_NS + 2];
      const s_t coeff_direction_u0_3 = direction[3][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_3 * fgref[0][q * U_NS + 3];
      u0_direction_grad_1_ref += coeff_direction_u0_3 * fgref[1][q * U_NS + 3];
      u0_direction_grad_2_ref += coeff_direction_u0_3 * fgref[2][q * U_NS + 3];
      const s_t coeff_direction_u0_4 = direction[4][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_4 * fgref[0][q * U_NS + 4];
      u0_direction_grad_1_ref += coeff_direction_u0_4 * fgref[1][q * U_NS + 4];
      u0_direction_grad_2_ref += coeff_direction_u0_4 * fgref[2][q * U_NS + 4];
      const s_t coeff_direction_u0_5 = direction[5][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_5 * fgref[0][q * U_NS + 5];
      u0_direction_grad_1_ref += coeff_direction_u0_5 * fgref[1][q * U_NS + 5];
      u0_direction_grad_2_ref += coeff_direction_u0_5 * fgref[2][q * U_NS + 5];
      const s_t coeff_direction_u0_6 = direction[6][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_6 * fgref[0][q * U_NS + 6];
      u0_direction_grad_1_ref += coeff_direction_u0_6 * fgref[1][q * U_NS + 6];
      u0_direction_grad_2_ref += coeff_direction_u0_6 * fgref[2][q * U_NS + 6];
      const s_t coeff_direction_u0_7 = direction[7][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_7 * fgref[0][q * U_NS + 7];
      u0_direction_grad_1_ref += coeff_direction_u0_7 * fgref[1][q * U_NS + 7];
      u0_direction_grad_2_ref += coeff_direction_u0_7 * fgref[2][q * U_NS + 7];
      const s_t coeff_direction_u0_8 = direction[8][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_8 * fgref[0][q * U_NS + 8];
      u0_direction_grad_1_ref += coeff_direction_u0_8 * fgref[1][q * U_NS + 8];
      u0_direction_grad_2_ref += coeff_direction_u0_8 * fgref[2][q * U_NS + 8];
      const s_t coeff_direction_u0_9 = direction[9][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_9 * fgref[0][q * U_NS + 9];
      u0_direction_grad_1_ref += coeff_direction_u0_9 * fgref[1][q * U_NS + 9];
      u0_direction_grad_2_ref += coeff_direction_u0_9 * fgref[2][q * U_NS + 9];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
      const s_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
      s_t u1_direction_grad_0_ref = s_t(0);
      s_t u1_direction_grad_1_ref = s_t(0);
      s_t u1_direction_grad_2_ref = s_t(0);
      const s_t coeff_direction_u1_0 = direction[10][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_0 * fgref[0][q * U_NS];
      u1_direction_grad_1_ref += coeff_direction_u1_0 * fgref[1][q * U_NS];
      u1_direction_grad_2_ref += coeff_direction_u1_0 * fgref[2][q * U_NS];
      const s_t coeff_direction_u1_1 = direction[11][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_1 * fgref[0][q * U_NS + 1];
      u1_direction_grad_1_ref += coeff_direction_u1_1 * fgref[1][q * U_NS + 1];
      u1_direction_grad_2_ref += coeff_direction_u1_1 * fgref[2][q * U_NS + 1];
      const s_t coeff_direction_u1_2 = direction[12][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_2 * fgref[0][q * U_NS + 2];
      u1_direction_grad_1_ref += coeff_direction_u1_2 * fgref[1][q * U_NS + 2];
      u1_direction_grad_2_ref += coeff_direction_u1_2 * fgref[2][q * U_NS + 2];
      const s_t coeff_direction_u1_3 = direction[13][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_3 * fgref[0][q * U_NS + 3];
      u1_direction_grad_1_ref += coeff_direction_u1_3 * fgref[1][q * U_NS + 3];
      u1_direction_grad_2_ref += coeff_direction_u1_3 * fgref[2][q * U_NS + 3];
      const s_t coeff_direction_u1_4 = direction[14][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_4 * fgref[0][q * U_NS + 4];
      u1_direction_grad_1_ref += coeff_direction_u1_4 * fgref[1][q * U_NS + 4];
      u1_direction_grad_2_ref += coeff_direction_u1_4 * fgref[2][q * U_NS + 4];
      const s_t coeff_direction_u1_5 = direction[15][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_5 * fgref[0][q * U_NS + 5];
      u1_direction_grad_1_ref += coeff_direction_u1_5 * fgref[1][q * U_NS + 5];
      u1_direction_grad_2_ref += coeff_direction_u1_5 * fgref[2][q * U_NS + 5];
      const s_t coeff_direction_u1_6 = direction[16][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_6 * fgref[0][q * U_NS + 6];
      u1_direction_grad_1_ref += coeff_direction_u1_6 * fgref[1][q * U_NS + 6];
      u1_direction_grad_2_ref += coeff_direction_u1_6 * fgref[2][q * U_NS + 6];
      const s_t coeff_direction_u1_7 = direction[17][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_7 * fgref[0][q * U_NS + 7];
      u1_direction_grad_1_ref += coeff_direction_u1_7 * fgref[1][q * U_NS + 7];
      u1_direction_grad_2_ref += coeff_direction_u1_7 * fgref[2][q * U_NS + 7];
      const s_t coeff_direction_u1_8 = direction[18][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_8 * fgref[0][q * U_NS + 8];
      u1_direction_grad_1_ref += coeff_direction_u1_8 * fgref[1][q * U_NS + 8];
      u1_direction_grad_2_ref += coeff_direction_u1_8 * fgref[2][q * U_NS + 8];
      const s_t coeff_direction_u1_9 = direction[19][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_9 * fgref[0][q * U_NS + 9];
      u1_direction_grad_1_ref += coeff_direction_u1_9 * fgref[1][q * U_NS + 9];
      u1_direction_grad_2_ref += coeff_direction_u1_9 * fgref[2][q * U_NS + 9];
      const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
      const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
      const s_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
      s_t u2_direction_grad_0_ref = s_t(0);
      s_t u2_direction_grad_1_ref = s_t(0);
      s_t u2_direction_grad_2_ref = s_t(0);
      const s_t coeff_direction_u2_0 = direction[20][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_0 * fgref[0][q * U_NS];
      u2_direction_grad_1_ref += coeff_direction_u2_0 * fgref[1][q * U_NS];
      u2_direction_grad_2_ref += coeff_direction_u2_0 * fgref[2][q * U_NS];
      const s_t coeff_direction_u2_1 = direction[21][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_1 * fgref[0][q * U_NS + 1];
      u2_direction_grad_1_ref += coeff_direction_u2_1 * fgref[1][q * U_NS + 1];
      u2_direction_grad_2_ref += coeff_direction_u2_1 * fgref[2][q * U_NS + 1];
      const s_t coeff_direction_u2_2 = direction[22][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_2 * fgref[0][q * U_NS + 2];
      u2_direction_grad_1_ref += coeff_direction_u2_2 * fgref[1][q * U_NS + 2];
      u2_direction_grad_2_ref += coeff_direction_u2_2 * fgref[2][q * U_NS + 2];
      const s_t coeff_direction_u2_3 = direction[23][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_3 * fgref[0][q * U_NS + 3];
      u2_direction_grad_1_ref += coeff_direction_u2_3 * fgref[1][q * U_NS + 3];
      u2_direction_grad_2_ref += coeff_direction_u2_3 * fgref[2][q * U_NS + 3];
      const s_t coeff_direction_u2_4 = direction[24][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_4 * fgref[0][q * U_NS + 4];
      u2_direction_grad_1_ref += coeff_direction_u2_4 * fgref[1][q * U_NS + 4];
      u2_direction_grad_2_ref += coeff_direction_u2_4 * fgref[2][q * U_NS + 4];
      const s_t coeff_direction_u2_5 = direction[25][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_5 * fgref[0][q * U_NS + 5];
      u2_direction_grad_1_ref += coeff_direction_u2_5 * fgref[1][q * U_NS + 5];
      u2_direction_grad_2_ref += coeff_direction_u2_5 * fgref[2][q * U_NS + 5];
      const s_t coeff_direction_u2_6 = direction[26][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_6 * fgref[0][q * U_NS + 6];
      u2_direction_grad_1_ref += coeff_direction_u2_6 * fgref[1][q * U_NS + 6];
      u2_direction_grad_2_ref += coeff_direction_u2_6 * fgref[2][q * U_NS + 6];
      const s_t coeff_direction_u2_7 = direction[27][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_7 * fgref[0][q * U_NS + 7];
      u2_direction_grad_1_ref += coeff_direction_u2_7 * fgref[1][q * U_NS + 7];
      u2_direction_grad_2_ref += coeff_direction_u2_7 * fgref[2][q * U_NS + 7];
      const s_t coeff_direction_u2_8 = direction[28][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_8 * fgref[0][q * U_NS + 8];
      u2_direction_grad_1_ref += coeff_direction_u2_8 * fgref[1][q * U_NS + 8];
      u2_direction_grad_2_ref += coeff_direction_u2_8 * fgref[2][q * U_NS + 8];
      const s_t coeff_direction_u2_9 = direction[29][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_9 * fgref[0][q * U_NS + 9];
      u2_direction_grad_1_ref += coeff_direction_u2_9 * fgref[1][q * U_NS + 9];
      u2_direction_grad_2_ref += coeff_direction_u2_9 * fgref[2][q * U_NS + 9];
      const s_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
      const s_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
      const s_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
      s_t p_direction_grad_0_ref = s_t(0);
      s_t p_direction_grad_1_ref = s_t(0);
      s_t p_direction_grad_2_ref = s_t(0);
      const s_t coeff_direction_p_0 = direction[30][lane];
      p_direction_grad_0_ref += coeff_direction_p_0 * fgref[ND][q * P_NS];
      p_direction_grad_1_ref += coeff_direction_p_0 * fgref[ND + 1][q * P_NS];
      p_direction_grad_2_ref += coeff_direction_p_0 * fgref[ND + 2][q * P_NS];
      const s_t coeff_direction_p_1 = direction[31][lane];
      p_direction_grad_0_ref += coeff_direction_p_1 * fgref[ND][q * P_NS + 1];
      p_direction_grad_1_ref += coeff_direction_p_1 * fgref[ND + 1][q * P_NS + 1];
      p_direction_grad_2_ref += coeff_direction_p_1 * fgref[ND + 2][q * P_NS + 1];
      const s_t coeff_direction_p_2 = direction[32][lane];
      p_direction_grad_0_ref += coeff_direction_p_2 * fgref[ND][q * P_NS + 2];
      p_direction_grad_1_ref += coeff_direction_p_2 * fgref[ND + 1][q * P_NS + 2];
      p_direction_grad_2_ref += coeff_direction_p_2 * fgref[ND + 2][q * P_NS + 2];
      const s_t coeff_direction_p_3 = direction[33][lane];
      p_direction_grad_0_ref += coeff_direction_p_3 * fgref[ND][q * P_NS + 3];
      p_direction_grad_1_ref += coeff_direction_p_3 * fgref[ND + 1][q * P_NS + 3];
      p_direction_grad_2_ref += coeff_direction_p_3 * fgref[ND + 2][q * P_NS + 3];
      const s_t p_direction_grad_0 = (p_direction_grad_0_ref * adj0 + p_direction_grad_1_ref * adj3 + p_direction_grad_2_ref * adj6) / det;
      const s_t p_direction_grad_1 = (p_direction_grad_0_ref * adj1 + p_direction_grad_1_ref * adj4 + p_direction_grad_2_ref * adj7) / det;
      const s_t p_direction_grad_2 = (p_direction_grad_0_ref * adj2 + p_direction_grad_1_ref * adj5 + p_direction_grad_2_ref * adj8) / det;
      const s_t value_coeff3 = u0_direction_grad_0 + u1_direction_grad_1 + u2_direction_grad_2;
      const s_t test_value_p_0 = field_shape[1][q * P_NS];
      output[30][lane] += q_weight[q] * det * (value_coeff3 * test_value_p_0);
      const s_t test_value_p_1 = field_shape[1][q * P_NS + 1];
      output[31][lane] += q_weight[q] * det * (value_coeff3 * test_value_p_1);
      const s_t test_value_p_2 = field_shape[1][q * P_NS + 2];
      output[32][lane] += q_weight[q] * det * (value_coeff3 * test_value_p_2);
      const s_t test_value_p_3 = field_shape[1][q * P_NS + 3];
      output[33][lane] += q_weight[q] * det * (value_coeff3 * test_value_p_3);
    }
  }
}

template <typename s_t, int NQ, int CELL_NS, int VS>
static SFEM_INLINE void navier_stokes_form_2_p_u_d3_simplex_mixed_jacobian_action_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR field_shape[2],
    const s_t *const RSTR fgref[6],
    const s_t *const RSTR q_weight,
    const s_t direction[34][VS],
    s_t output[34][VS]
) {
  static constexpr int ND = 3;
  static constexpr int NC = 2;
  static constexpr int N_FIELD_STREAMS = 34;
  (void)CELL_NS;
  (void)N_FIELD_STREAMS;
  static constexpr int U_NS = 10;
  static constexpr int P_NS = 4;
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
      s_t u0_direction_grad_0_ref = s_t(0);
      s_t u0_direction_grad_1_ref = s_t(0);
      s_t u0_direction_grad_2_ref = s_t(0);
      const s_t coeff_direction_u0_0 = direction[0][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_0 * fgref[0][q * U_NS];
      u0_direction_grad_1_ref += coeff_direction_u0_0 * fgref[1][q * U_NS];
      u0_direction_grad_2_ref += coeff_direction_u0_0 * fgref[2][q * U_NS];
      const s_t coeff_direction_u0_1 = direction[1][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_1 * fgref[0][q * U_NS + 1];
      u0_direction_grad_1_ref += coeff_direction_u0_1 * fgref[1][q * U_NS + 1];
      u0_direction_grad_2_ref += coeff_direction_u0_1 * fgref[2][q * U_NS + 1];
      const s_t coeff_direction_u0_2 = direction[2][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_2 * fgref[0][q * U_NS + 2];
      u0_direction_grad_1_ref += coeff_direction_u0_2 * fgref[1][q * U_NS + 2];
      u0_direction_grad_2_ref += coeff_direction_u0_2 * fgref[2][q * U_NS + 2];
      const s_t coeff_direction_u0_3 = direction[3][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_3 * fgref[0][q * U_NS + 3];
      u0_direction_grad_1_ref += coeff_direction_u0_3 * fgref[1][q * U_NS + 3];
      u0_direction_grad_2_ref += coeff_direction_u0_3 * fgref[2][q * U_NS + 3];
      const s_t coeff_direction_u0_4 = direction[4][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_4 * fgref[0][q * U_NS + 4];
      u0_direction_grad_1_ref += coeff_direction_u0_4 * fgref[1][q * U_NS + 4];
      u0_direction_grad_2_ref += coeff_direction_u0_4 * fgref[2][q * U_NS + 4];
      const s_t coeff_direction_u0_5 = direction[5][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_5 * fgref[0][q * U_NS + 5];
      u0_direction_grad_1_ref += coeff_direction_u0_5 * fgref[1][q * U_NS + 5];
      u0_direction_grad_2_ref += coeff_direction_u0_5 * fgref[2][q * U_NS + 5];
      const s_t coeff_direction_u0_6 = direction[6][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_6 * fgref[0][q * U_NS + 6];
      u0_direction_grad_1_ref += coeff_direction_u0_6 * fgref[1][q * U_NS + 6];
      u0_direction_grad_2_ref += coeff_direction_u0_6 * fgref[2][q * U_NS + 6];
      const s_t coeff_direction_u0_7 = direction[7][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_7 * fgref[0][q * U_NS + 7];
      u0_direction_grad_1_ref += coeff_direction_u0_7 * fgref[1][q * U_NS + 7];
      u0_direction_grad_2_ref += coeff_direction_u0_7 * fgref[2][q * U_NS + 7];
      const s_t coeff_direction_u0_8 = direction[8][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_8 * fgref[0][q * U_NS + 8];
      u0_direction_grad_1_ref += coeff_direction_u0_8 * fgref[1][q * U_NS + 8];
      u0_direction_grad_2_ref += coeff_direction_u0_8 * fgref[2][q * U_NS + 8];
      const s_t coeff_direction_u0_9 = direction[9][lane];
      u0_direction_grad_0_ref += coeff_direction_u0_9 * fgref[0][q * U_NS + 9];
      u0_direction_grad_1_ref += coeff_direction_u0_9 * fgref[1][q * U_NS + 9];
      u0_direction_grad_2_ref += coeff_direction_u0_9 * fgref[2][q * U_NS + 9];
      const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
      const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
      const s_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
      s_t u1_direction_grad_0_ref = s_t(0);
      s_t u1_direction_grad_1_ref = s_t(0);
      s_t u1_direction_grad_2_ref = s_t(0);
      const s_t coeff_direction_u1_0 = direction[10][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_0 * fgref[0][q * U_NS];
      u1_direction_grad_1_ref += coeff_direction_u1_0 * fgref[1][q * U_NS];
      u1_direction_grad_2_ref += coeff_direction_u1_0 * fgref[2][q * U_NS];
      const s_t coeff_direction_u1_1 = direction[11][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_1 * fgref[0][q * U_NS + 1];
      u1_direction_grad_1_ref += coeff_direction_u1_1 * fgref[1][q * U_NS + 1];
      u1_direction_grad_2_ref += coeff_direction_u1_1 * fgref[2][q * U_NS + 1];
      const s_t coeff_direction_u1_2 = direction[12][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_2 * fgref[0][q * U_NS + 2];
      u1_direction_grad_1_ref += coeff_direction_u1_2 * fgref[1][q * U_NS + 2];
      u1_direction_grad_2_ref += coeff_direction_u1_2 * fgref[2][q * U_NS + 2];
      const s_t coeff_direction_u1_3 = direction[13][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_3 * fgref[0][q * U_NS + 3];
      u1_direction_grad_1_ref += coeff_direction_u1_3 * fgref[1][q * U_NS + 3];
      u1_direction_grad_2_ref += coeff_direction_u1_3 * fgref[2][q * U_NS + 3];
      const s_t coeff_direction_u1_4 = direction[14][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_4 * fgref[0][q * U_NS + 4];
      u1_direction_grad_1_ref += coeff_direction_u1_4 * fgref[1][q * U_NS + 4];
      u1_direction_grad_2_ref += coeff_direction_u1_4 * fgref[2][q * U_NS + 4];
      const s_t coeff_direction_u1_5 = direction[15][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_5 * fgref[0][q * U_NS + 5];
      u1_direction_grad_1_ref += coeff_direction_u1_5 * fgref[1][q * U_NS + 5];
      u1_direction_grad_2_ref += coeff_direction_u1_5 * fgref[2][q * U_NS + 5];
      const s_t coeff_direction_u1_6 = direction[16][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_6 * fgref[0][q * U_NS + 6];
      u1_direction_grad_1_ref += coeff_direction_u1_6 * fgref[1][q * U_NS + 6];
      u1_direction_grad_2_ref += coeff_direction_u1_6 * fgref[2][q * U_NS + 6];
      const s_t coeff_direction_u1_7 = direction[17][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_7 * fgref[0][q * U_NS + 7];
      u1_direction_grad_1_ref += coeff_direction_u1_7 * fgref[1][q * U_NS + 7];
      u1_direction_grad_2_ref += coeff_direction_u1_7 * fgref[2][q * U_NS + 7];
      const s_t coeff_direction_u1_8 = direction[18][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_8 * fgref[0][q * U_NS + 8];
      u1_direction_grad_1_ref += coeff_direction_u1_8 * fgref[1][q * U_NS + 8];
      u1_direction_grad_2_ref += coeff_direction_u1_8 * fgref[2][q * U_NS + 8];
      const s_t coeff_direction_u1_9 = direction[19][lane];
      u1_direction_grad_0_ref += coeff_direction_u1_9 * fgref[0][q * U_NS + 9];
      u1_direction_grad_1_ref += coeff_direction_u1_9 * fgref[1][q * U_NS + 9];
      u1_direction_grad_2_ref += coeff_direction_u1_9 * fgref[2][q * U_NS + 9];
      const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
      const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
      const s_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
      s_t u2_direction_grad_0_ref = s_t(0);
      s_t u2_direction_grad_1_ref = s_t(0);
      s_t u2_direction_grad_2_ref = s_t(0);
      const s_t coeff_direction_u2_0 = direction[20][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_0 * fgref[0][q * U_NS];
      u2_direction_grad_1_ref += coeff_direction_u2_0 * fgref[1][q * U_NS];
      u2_direction_grad_2_ref += coeff_direction_u2_0 * fgref[2][q * U_NS];
      const s_t coeff_direction_u2_1 = direction[21][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_1 * fgref[0][q * U_NS + 1];
      u2_direction_grad_1_ref += coeff_direction_u2_1 * fgref[1][q * U_NS + 1];
      u2_direction_grad_2_ref += coeff_direction_u2_1 * fgref[2][q * U_NS + 1];
      const s_t coeff_direction_u2_2 = direction[22][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_2 * fgref[0][q * U_NS + 2];
      u2_direction_grad_1_ref += coeff_direction_u2_2 * fgref[1][q * U_NS + 2];
      u2_direction_grad_2_ref += coeff_direction_u2_2 * fgref[2][q * U_NS + 2];
      const s_t coeff_direction_u2_3 = direction[23][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_3 * fgref[0][q * U_NS + 3];
      u2_direction_grad_1_ref += coeff_direction_u2_3 * fgref[1][q * U_NS + 3];
      u2_direction_grad_2_ref += coeff_direction_u2_3 * fgref[2][q * U_NS + 3];
      const s_t coeff_direction_u2_4 = direction[24][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_4 * fgref[0][q * U_NS + 4];
      u2_direction_grad_1_ref += coeff_direction_u2_4 * fgref[1][q * U_NS + 4];
      u2_direction_grad_2_ref += coeff_direction_u2_4 * fgref[2][q * U_NS + 4];
      const s_t coeff_direction_u2_5 = direction[25][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_5 * fgref[0][q * U_NS + 5];
      u2_direction_grad_1_ref += coeff_direction_u2_5 * fgref[1][q * U_NS + 5];
      u2_direction_grad_2_ref += coeff_direction_u2_5 * fgref[2][q * U_NS + 5];
      const s_t coeff_direction_u2_6 = direction[26][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_6 * fgref[0][q * U_NS + 6];
      u2_direction_grad_1_ref += coeff_direction_u2_6 * fgref[1][q * U_NS + 6];
      u2_direction_grad_2_ref += coeff_direction_u2_6 * fgref[2][q * U_NS + 6];
      const s_t coeff_direction_u2_7 = direction[27][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_7 * fgref[0][q * U_NS + 7];
      u2_direction_grad_1_ref += coeff_direction_u2_7 * fgref[1][q * U_NS + 7];
      u2_direction_grad_2_ref += coeff_direction_u2_7 * fgref[2][q * U_NS + 7];
      const s_t coeff_direction_u2_8 = direction[28][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_8 * fgref[0][q * U_NS + 8];
      u2_direction_grad_1_ref += coeff_direction_u2_8 * fgref[1][q * U_NS + 8];
      u2_direction_grad_2_ref += coeff_direction_u2_8 * fgref[2][q * U_NS + 8];
      const s_t coeff_direction_u2_9 = direction[29][lane];
      u2_direction_grad_0_ref += coeff_direction_u2_9 * fgref[0][q * U_NS + 9];
      u2_direction_grad_1_ref += coeff_direction_u2_9 * fgref[1][q * U_NS + 9];
      u2_direction_grad_2_ref += coeff_direction_u2_9 * fgref[2][q * U_NS + 9];
      const s_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
      const s_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
      const s_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
      s_t p_direction_grad_0_ref = s_t(0);
      s_t p_direction_grad_1_ref = s_t(0);
      s_t p_direction_grad_2_ref = s_t(0);
      const s_t coeff_direction_p_0 = direction[30][lane];
      p_direction_grad_0_ref += coeff_direction_p_0 * fgref[ND][q * P_NS];
      p_direction_grad_1_ref += coeff_direction_p_0 * fgref[ND + 1][q * P_NS];
      p_direction_grad_2_ref += coeff_direction_p_0 * fgref[ND + 2][q * P_NS];
      const s_t coeff_direction_p_1 = direction[31][lane];
      p_direction_grad_0_ref += coeff_direction_p_1 * fgref[ND][q * P_NS + 1];
      p_direction_grad_1_ref += coeff_direction_p_1 * fgref[ND + 1][q * P_NS + 1];
      p_direction_grad_2_ref += coeff_direction_p_1 * fgref[ND + 2][q * P_NS + 1];
      const s_t coeff_direction_p_2 = direction[32][lane];
      p_direction_grad_0_ref += coeff_direction_p_2 * fgref[ND][q * P_NS + 2];
      p_direction_grad_1_ref += coeff_direction_p_2 * fgref[ND + 1][q * P_NS + 2];
      p_direction_grad_2_ref += coeff_direction_p_2 * fgref[ND + 2][q * P_NS + 2];
      const s_t coeff_direction_p_3 = direction[33][lane];
      p_direction_grad_0_ref += coeff_direction_p_3 * fgref[ND][q * P_NS + 3];
      p_direction_grad_1_ref += coeff_direction_p_3 * fgref[ND + 1][q * P_NS + 3];
      p_direction_grad_2_ref += coeff_direction_p_3 * fgref[ND + 2][q * P_NS + 3];
      const s_t p_direction_grad_0 = (p_direction_grad_0_ref * adj0 + p_direction_grad_1_ref * adj3 + p_direction_grad_2_ref * adj6) / det;
      const s_t p_direction_grad_1 = (p_direction_grad_0_ref * adj1 + p_direction_grad_1_ref * adj4 + p_direction_grad_2_ref * adj7) / det;
      const s_t p_direction_grad_2 = (p_direction_grad_0_ref * adj2 + p_direction_grad_1_ref * adj5 + p_direction_grad_2_ref * adj8) / det;
      const s_t value_coeff3 = u0_direction_grad_0 + u1_direction_grad_1 + u2_direction_grad_2;
      const s_t test_value_p_0 = field_shape[1][q * P_NS];
      output[30][lane] += q_weight[q] * det * (value_coeff3 * test_value_p_0);
      const s_t test_value_p_1 = field_shape[1][q * P_NS + 1];
      output[31][lane] += q_weight[q] * det * (value_coeff3 * test_value_p_1);
      const s_t test_value_p_2 = field_shape[1][q * P_NS + 2];
      output[32][lane] += q_weight[q] * det * (value_coeff3 * test_value_p_2);
      const s_t test_value_p_3 = field_shape[1][q * P_NS + 3];
      output[33][lane] += q_weight[q] * det * (value_coeff3 * test_value_p_3);
    }
  }
}

} // namespace codegen
} // namespace sfem

#endif
