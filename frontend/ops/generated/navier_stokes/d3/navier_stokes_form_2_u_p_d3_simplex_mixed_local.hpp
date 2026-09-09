#ifndef NAVIER_STOKES_FORM_2_U_P_D3_SIMPLEX_MIXED_LOCAL_HPP
#define NAVIER_STOKES_FORM_2_U_P_D3_SIMPLEX_MIXED_LOCAL_HPP

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
static SFEM_INLINE void navier_stokes_form_2_u_p_d3_simplex_mixed_residual_block(
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
static SFEM_INLINE void navier_stokes_form_2_u_p_d3_simplex_mixed_residual_block_contiguous(
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
static SFEM_INLINE void navier_stokes_form_2_u_p_d3_simplex_mixed_jacobian_action_block(
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
      s_t p_direction = s_t(0);
      const s_t coeff_direction_p_0 = direction[30][lane];
      p_direction += coeff_direction_p_0 * field_shape[1][q * P_NS];
      const s_t coeff_direction_p_1 = direction[31][lane];
      p_direction += coeff_direction_p_1 * field_shape[1][q * P_NS + 1];
      const s_t coeff_direction_p_2 = direction[32][lane];
      p_direction += coeff_direction_p_2 * field_shape[1][q * P_NS + 2];
      const s_t coeff_direction_p_3 = direction[33][lane];
      p_direction += coeff_direction_p_3 * field_shape[1][q * P_NS + 3];
      const s_t residual_tmp0 = -p_direction;
      const s_t grad_coeff0_0 = residual_tmp0;
      const s_t grad_coeff1_1 = residual_tmp0;
      const s_t grad_coeff2_2 = residual_tmp0;
      const s_t test_grad0_u0_0 = (fgref[0][q * U_NS] * adj0 + fgref[1][q * U_NS] * adj3 + fgref[2][q * U_NS] * adj6) / det;
      output[0][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_0);
      const s_t test_grad0_u0_1 = (fgref[0][q * U_NS + 1] * adj0 + fgref[1][q * U_NS + 1] * adj3 + fgref[2][q * U_NS + 1] * adj6) / det;
      output[1][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_1);
      const s_t test_grad0_u0_2 = (fgref[0][q * U_NS + 2] * adj0 + fgref[1][q * U_NS + 2] * adj3 + fgref[2][q * U_NS + 2] * adj6) / det;
      output[2][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_2);
      const s_t test_grad0_u0_3 = (fgref[0][q * U_NS + 3] * adj0 + fgref[1][q * U_NS + 3] * adj3 + fgref[2][q * U_NS + 3] * adj6) / det;
      output[3][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_3);
      const s_t test_grad0_u0_4 = (fgref[0][q * U_NS + 4] * adj0 + fgref[1][q * U_NS + 4] * adj3 + fgref[2][q * U_NS + 4] * adj6) / det;
      output[4][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_4);
      const s_t test_grad0_u0_5 = (fgref[0][q * U_NS + 5] * adj0 + fgref[1][q * U_NS + 5] * adj3 + fgref[2][q * U_NS + 5] * adj6) / det;
      output[5][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_5);
      const s_t test_grad0_u0_6 = (fgref[0][q * U_NS + 6] * adj0 + fgref[1][q * U_NS + 6] * adj3 + fgref[2][q * U_NS + 6] * adj6) / det;
      output[6][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_6);
      const s_t test_grad0_u0_7 = (fgref[0][q * U_NS + 7] * adj0 + fgref[1][q * U_NS + 7] * adj3 + fgref[2][q * U_NS + 7] * adj6) / det;
      output[7][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_7);
      const s_t test_grad0_u0_8 = (fgref[0][q * U_NS + 8] * adj0 + fgref[1][q * U_NS + 8] * adj3 + fgref[2][q * U_NS + 8] * adj6) / det;
      output[8][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_8);
      const s_t test_grad0_u0_9 = (fgref[0][q * U_NS + 9] * adj0 + fgref[1][q * U_NS + 9] * adj3 + fgref[2][q * U_NS + 9] * adj6) / det;
      output[9][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_9);
      const s_t test_grad1_u1_0 = (fgref[0][q * U_NS] * adj1 + fgref[1][q * U_NS] * adj4 + fgref[2][q * U_NS] * adj7) / det;
      output[10][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_0);
      const s_t test_grad1_u1_1 = (fgref[0][q * U_NS + 1] * adj1 + fgref[1][q * U_NS + 1] * adj4 + fgref[2][q * U_NS + 1] * adj7) / det;
      output[11][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_1);
      const s_t test_grad1_u1_2 = (fgref[0][q * U_NS + 2] * adj1 + fgref[1][q * U_NS + 2] * adj4 + fgref[2][q * U_NS + 2] * adj7) / det;
      output[12][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_2);
      const s_t test_grad1_u1_3 = (fgref[0][q * U_NS + 3] * adj1 + fgref[1][q * U_NS + 3] * adj4 + fgref[2][q * U_NS + 3] * adj7) / det;
      output[13][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_3);
      const s_t test_grad1_u1_4 = (fgref[0][q * U_NS + 4] * adj1 + fgref[1][q * U_NS + 4] * adj4 + fgref[2][q * U_NS + 4] * adj7) / det;
      output[14][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_4);
      const s_t test_grad1_u1_5 = (fgref[0][q * U_NS + 5] * adj1 + fgref[1][q * U_NS + 5] * adj4 + fgref[2][q * U_NS + 5] * adj7) / det;
      output[15][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_5);
      const s_t test_grad1_u1_6 = (fgref[0][q * U_NS + 6] * adj1 + fgref[1][q * U_NS + 6] * adj4 + fgref[2][q * U_NS + 6] * adj7) / det;
      output[16][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_6);
      const s_t test_grad1_u1_7 = (fgref[0][q * U_NS + 7] * adj1 + fgref[1][q * U_NS + 7] * adj4 + fgref[2][q * U_NS + 7] * adj7) / det;
      output[17][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_7);
      const s_t test_grad1_u1_8 = (fgref[0][q * U_NS + 8] * adj1 + fgref[1][q * U_NS + 8] * adj4 + fgref[2][q * U_NS + 8] * adj7) / det;
      output[18][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_8);
      const s_t test_grad1_u1_9 = (fgref[0][q * U_NS + 9] * adj1 + fgref[1][q * U_NS + 9] * adj4 + fgref[2][q * U_NS + 9] * adj7) / det;
      output[19][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_9);
      const s_t test_grad2_u2_0 = (fgref[0][q * U_NS] * adj2 + fgref[1][q * U_NS] * adj5 + fgref[2][q * U_NS] * adj8) / det;
      output[20][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_0);
      const s_t test_grad2_u2_1 = (fgref[0][q * U_NS + 1] * adj2 + fgref[1][q * U_NS + 1] * adj5 + fgref[2][q * U_NS + 1] * adj8) / det;
      output[21][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_1);
      const s_t test_grad2_u2_2 = (fgref[0][q * U_NS + 2] * adj2 + fgref[1][q * U_NS + 2] * adj5 + fgref[2][q * U_NS + 2] * adj8) / det;
      output[22][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_2);
      const s_t test_grad2_u2_3 = (fgref[0][q * U_NS + 3] * adj2 + fgref[1][q * U_NS + 3] * adj5 + fgref[2][q * U_NS + 3] * adj8) / det;
      output[23][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_3);
      const s_t test_grad2_u2_4 = (fgref[0][q * U_NS + 4] * adj2 + fgref[1][q * U_NS + 4] * adj5 + fgref[2][q * U_NS + 4] * adj8) / det;
      output[24][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_4);
      const s_t test_grad2_u2_5 = (fgref[0][q * U_NS + 5] * adj2 + fgref[1][q * U_NS + 5] * adj5 + fgref[2][q * U_NS + 5] * adj8) / det;
      output[25][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_5);
      const s_t test_grad2_u2_6 = (fgref[0][q * U_NS + 6] * adj2 + fgref[1][q * U_NS + 6] * adj5 + fgref[2][q * U_NS + 6] * adj8) / det;
      output[26][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_6);
      const s_t test_grad2_u2_7 = (fgref[0][q * U_NS + 7] * adj2 + fgref[1][q * U_NS + 7] * adj5 + fgref[2][q * U_NS + 7] * adj8) / det;
      output[27][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_7);
      const s_t test_grad2_u2_8 = (fgref[0][q * U_NS + 8] * adj2 + fgref[1][q * U_NS + 8] * adj5 + fgref[2][q * U_NS + 8] * adj8) / det;
      output[28][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_8);
      const s_t test_grad2_u2_9 = (fgref[0][q * U_NS + 9] * adj2 + fgref[1][q * U_NS + 9] * adj5 + fgref[2][q * U_NS + 9] * adj8) / det;
      output[29][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_9);
    }
  }
}

template <typename s_t, int NQ, int CELL_NS, int VS>
static SFEM_INLINE void navier_stokes_form_2_u_p_d3_simplex_mixed_jacobian_action_block_contiguous(
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
      s_t p_direction = s_t(0);
      const s_t coeff_direction_p_0 = direction[30][lane];
      p_direction += coeff_direction_p_0 * field_shape[1][q * P_NS];
      const s_t coeff_direction_p_1 = direction[31][lane];
      p_direction += coeff_direction_p_1 * field_shape[1][q * P_NS + 1];
      const s_t coeff_direction_p_2 = direction[32][lane];
      p_direction += coeff_direction_p_2 * field_shape[1][q * P_NS + 2];
      const s_t coeff_direction_p_3 = direction[33][lane];
      p_direction += coeff_direction_p_3 * field_shape[1][q * P_NS + 3];
      const s_t residual_tmp0 = -p_direction;
      const s_t grad_coeff0_0 = residual_tmp0;
      const s_t grad_coeff1_1 = residual_tmp0;
      const s_t grad_coeff2_2 = residual_tmp0;
      const s_t test_grad0_u0_0 = (fgref[0][q * U_NS] * adj0 + fgref[1][q * U_NS] * adj3 + fgref[2][q * U_NS] * adj6) / det;
      output[0][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_0);
      const s_t test_grad0_u0_1 = (fgref[0][q * U_NS + 1] * adj0 + fgref[1][q * U_NS + 1] * adj3 + fgref[2][q * U_NS + 1] * adj6) / det;
      output[1][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_1);
      const s_t test_grad0_u0_2 = (fgref[0][q * U_NS + 2] * adj0 + fgref[1][q * U_NS + 2] * adj3 + fgref[2][q * U_NS + 2] * adj6) / det;
      output[2][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_2);
      const s_t test_grad0_u0_3 = (fgref[0][q * U_NS + 3] * adj0 + fgref[1][q * U_NS + 3] * adj3 + fgref[2][q * U_NS + 3] * adj6) / det;
      output[3][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_3);
      const s_t test_grad0_u0_4 = (fgref[0][q * U_NS + 4] * adj0 + fgref[1][q * U_NS + 4] * adj3 + fgref[2][q * U_NS + 4] * adj6) / det;
      output[4][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_4);
      const s_t test_grad0_u0_5 = (fgref[0][q * U_NS + 5] * adj0 + fgref[1][q * U_NS + 5] * adj3 + fgref[2][q * U_NS + 5] * adj6) / det;
      output[5][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_5);
      const s_t test_grad0_u0_6 = (fgref[0][q * U_NS + 6] * adj0 + fgref[1][q * U_NS + 6] * adj3 + fgref[2][q * U_NS + 6] * adj6) / det;
      output[6][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_6);
      const s_t test_grad0_u0_7 = (fgref[0][q * U_NS + 7] * adj0 + fgref[1][q * U_NS + 7] * adj3 + fgref[2][q * U_NS + 7] * adj6) / det;
      output[7][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_7);
      const s_t test_grad0_u0_8 = (fgref[0][q * U_NS + 8] * adj0 + fgref[1][q * U_NS + 8] * adj3 + fgref[2][q * U_NS + 8] * adj6) / det;
      output[8][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_8);
      const s_t test_grad0_u0_9 = (fgref[0][q * U_NS + 9] * adj0 + fgref[1][q * U_NS + 9] * adj3 + fgref[2][q * U_NS + 9] * adj6) / det;
      output[9][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_9);
      const s_t test_grad1_u1_0 = (fgref[0][q * U_NS] * adj1 + fgref[1][q * U_NS] * adj4 + fgref[2][q * U_NS] * adj7) / det;
      output[10][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_0);
      const s_t test_grad1_u1_1 = (fgref[0][q * U_NS + 1] * adj1 + fgref[1][q * U_NS + 1] * adj4 + fgref[2][q * U_NS + 1] * adj7) / det;
      output[11][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_1);
      const s_t test_grad1_u1_2 = (fgref[0][q * U_NS + 2] * adj1 + fgref[1][q * U_NS + 2] * adj4 + fgref[2][q * U_NS + 2] * adj7) / det;
      output[12][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_2);
      const s_t test_grad1_u1_3 = (fgref[0][q * U_NS + 3] * adj1 + fgref[1][q * U_NS + 3] * adj4 + fgref[2][q * U_NS + 3] * adj7) / det;
      output[13][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_3);
      const s_t test_grad1_u1_4 = (fgref[0][q * U_NS + 4] * adj1 + fgref[1][q * U_NS + 4] * adj4 + fgref[2][q * U_NS + 4] * adj7) / det;
      output[14][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_4);
      const s_t test_grad1_u1_5 = (fgref[0][q * U_NS + 5] * adj1 + fgref[1][q * U_NS + 5] * adj4 + fgref[2][q * U_NS + 5] * adj7) / det;
      output[15][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_5);
      const s_t test_grad1_u1_6 = (fgref[0][q * U_NS + 6] * adj1 + fgref[1][q * U_NS + 6] * adj4 + fgref[2][q * U_NS + 6] * adj7) / det;
      output[16][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_6);
      const s_t test_grad1_u1_7 = (fgref[0][q * U_NS + 7] * adj1 + fgref[1][q * U_NS + 7] * adj4 + fgref[2][q * U_NS + 7] * adj7) / det;
      output[17][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_7);
      const s_t test_grad1_u1_8 = (fgref[0][q * U_NS + 8] * adj1 + fgref[1][q * U_NS + 8] * adj4 + fgref[2][q * U_NS + 8] * adj7) / det;
      output[18][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_8);
      const s_t test_grad1_u1_9 = (fgref[0][q * U_NS + 9] * adj1 + fgref[1][q * U_NS + 9] * adj4 + fgref[2][q * U_NS + 9] * adj7) / det;
      output[19][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_9);
      const s_t test_grad2_u2_0 = (fgref[0][q * U_NS] * adj2 + fgref[1][q * U_NS] * adj5 + fgref[2][q * U_NS] * adj8) / det;
      output[20][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_0);
      const s_t test_grad2_u2_1 = (fgref[0][q * U_NS + 1] * adj2 + fgref[1][q * U_NS + 1] * adj5 + fgref[2][q * U_NS + 1] * adj8) / det;
      output[21][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_1);
      const s_t test_grad2_u2_2 = (fgref[0][q * U_NS + 2] * adj2 + fgref[1][q * U_NS + 2] * adj5 + fgref[2][q * U_NS + 2] * adj8) / det;
      output[22][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_2);
      const s_t test_grad2_u2_3 = (fgref[0][q * U_NS + 3] * adj2 + fgref[1][q * U_NS + 3] * adj5 + fgref[2][q * U_NS + 3] * adj8) / det;
      output[23][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_3);
      const s_t test_grad2_u2_4 = (fgref[0][q * U_NS + 4] * adj2 + fgref[1][q * U_NS + 4] * adj5 + fgref[2][q * U_NS + 4] * adj8) / det;
      output[24][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_4);
      const s_t test_grad2_u2_5 = (fgref[0][q * U_NS + 5] * adj2 + fgref[1][q * U_NS + 5] * adj5 + fgref[2][q * U_NS + 5] * adj8) / det;
      output[25][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_5);
      const s_t test_grad2_u2_6 = (fgref[0][q * U_NS + 6] * adj2 + fgref[1][q * U_NS + 6] * adj5 + fgref[2][q * U_NS + 6] * adj8) / det;
      output[26][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_6);
      const s_t test_grad2_u2_7 = (fgref[0][q * U_NS + 7] * adj2 + fgref[1][q * U_NS + 7] * adj5 + fgref[2][q * U_NS + 7] * adj8) / det;
      output[27][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_7);
      const s_t test_grad2_u2_8 = (fgref[0][q * U_NS + 8] * adj2 + fgref[1][q * U_NS + 8] * adj5 + fgref[2][q * U_NS + 8] * adj8) / det;
      output[28][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_8);
      const s_t test_grad2_u2_9 = (fgref[0][q * U_NS + 9] * adj2 + fgref[1][q * U_NS + 9] * adj5 + fgref[2][q * U_NS + 9] * adj8) / det;
      output[29][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_9);
    }
  }
}

} // namespace codegen
} // namespace sfem

#endif
