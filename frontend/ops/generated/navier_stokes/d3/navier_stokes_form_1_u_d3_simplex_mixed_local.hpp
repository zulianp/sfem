#ifndef NAVIER_STOKES_FORM_1_U_D3_SIMPLEX_MIXED_LOCAL_HPP
#define NAVIER_STOKES_FORM_1_U_D3_SIMPLEX_MIXED_LOCAL_HPP

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
static SFEM_INLINE void navier_stokes_form_1_u_d3_simplex_mixed_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR field_shape[2],
    const s_t *const RSTR fgref[6],
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[34],
    const s_t *const RSTR previous[34],
    const s_t convection_scale,
    const s_t dt,
    const s_t f0,
    const s_t f1,
    const s_t f2,
    const s_t nu,
    const s_t rho,
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
      s_t u0 = s_t(0);
      s_t u0_grad_0_ref = s_t(0);
      s_t u0_grad_1_ref = s_t(0);
      s_t u0_grad_2_ref = s_t(0);
      const s_t coeff_current_u0_0 = current[0][lane];
      u0 += coeff_current_u0_0 * field_shape[0][q * U_NS];
      u0_grad_0_ref += coeff_current_u0_0 * fgref[0][q * U_NS];
      u0_grad_1_ref += coeff_current_u0_0 * fgref[1][q * U_NS];
      u0_grad_2_ref += coeff_current_u0_0 * fgref[2][q * U_NS];
      const s_t coeff_current_u0_1 = current[1][lane];
      u0 += coeff_current_u0_1 * field_shape[0][q * U_NS + 1];
      u0_grad_0_ref += coeff_current_u0_1 * fgref[0][q * U_NS + 1];
      u0_grad_1_ref += coeff_current_u0_1 * fgref[1][q * U_NS + 1];
      u0_grad_2_ref += coeff_current_u0_1 * fgref[2][q * U_NS + 1];
      const s_t coeff_current_u0_2 = current[2][lane];
      u0 += coeff_current_u0_2 * field_shape[0][q * U_NS + 2];
      u0_grad_0_ref += coeff_current_u0_2 * fgref[0][q * U_NS + 2];
      u0_grad_1_ref += coeff_current_u0_2 * fgref[1][q * U_NS + 2];
      u0_grad_2_ref += coeff_current_u0_2 * fgref[2][q * U_NS + 2];
      const s_t coeff_current_u0_3 = current[3][lane];
      u0 += coeff_current_u0_3 * field_shape[0][q * U_NS + 3];
      u0_grad_0_ref += coeff_current_u0_3 * fgref[0][q * U_NS + 3];
      u0_grad_1_ref += coeff_current_u0_3 * fgref[1][q * U_NS + 3];
      u0_grad_2_ref += coeff_current_u0_3 * fgref[2][q * U_NS + 3];
      const s_t coeff_current_u0_4 = current[4][lane];
      u0 += coeff_current_u0_4 * field_shape[0][q * U_NS + 4];
      u0_grad_0_ref += coeff_current_u0_4 * fgref[0][q * U_NS + 4];
      u0_grad_1_ref += coeff_current_u0_4 * fgref[1][q * U_NS + 4];
      u0_grad_2_ref += coeff_current_u0_4 * fgref[2][q * U_NS + 4];
      const s_t coeff_current_u0_5 = current[5][lane];
      u0 += coeff_current_u0_5 * field_shape[0][q * U_NS + 5];
      u0_grad_0_ref += coeff_current_u0_5 * fgref[0][q * U_NS + 5];
      u0_grad_1_ref += coeff_current_u0_5 * fgref[1][q * U_NS + 5];
      u0_grad_2_ref += coeff_current_u0_5 * fgref[2][q * U_NS + 5];
      const s_t coeff_current_u0_6 = current[6][lane];
      u0 += coeff_current_u0_6 * field_shape[0][q * U_NS + 6];
      u0_grad_0_ref += coeff_current_u0_6 * fgref[0][q * U_NS + 6];
      u0_grad_1_ref += coeff_current_u0_6 * fgref[1][q * U_NS + 6];
      u0_grad_2_ref += coeff_current_u0_6 * fgref[2][q * U_NS + 6];
      const s_t coeff_current_u0_7 = current[7][lane];
      u0 += coeff_current_u0_7 * field_shape[0][q * U_NS + 7];
      u0_grad_0_ref += coeff_current_u0_7 * fgref[0][q * U_NS + 7];
      u0_grad_1_ref += coeff_current_u0_7 * fgref[1][q * U_NS + 7];
      u0_grad_2_ref += coeff_current_u0_7 * fgref[2][q * U_NS + 7];
      const s_t coeff_current_u0_8 = current[8][lane];
      u0 += coeff_current_u0_8 * field_shape[0][q * U_NS + 8];
      u0_grad_0_ref += coeff_current_u0_8 * fgref[0][q * U_NS + 8];
      u0_grad_1_ref += coeff_current_u0_8 * fgref[1][q * U_NS + 8];
      u0_grad_2_ref += coeff_current_u0_8 * fgref[2][q * U_NS + 8];
      const s_t coeff_current_u0_9 = current[9][lane];
      u0 += coeff_current_u0_9 * field_shape[0][q * U_NS + 9];
      u0_grad_0_ref += coeff_current_u0_9 * fgref[0][q * U_NS + 9];
      u0_grad_1_ref += coeff_current_u0_9 * fgref[1][q * U_NS + 9];
      u0_grad_2_ref += coeff_current_u0_9 * fgref[2][q * U_NS + 9];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
      const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
      s_t u0_old = s_t(0);
      const s_t coeff_previous_u0_0 = previous[0][lane];
      u0_old += coeff_previous_u0_0 * field_shape[0][q * U_NS];
      const s_t coeff_previous_u0_1 = previous[1][lane];
      u0_old += coeff_previous_u0_1 * field_shape[0][q * U_NS + 1];
      const s_t coeff_previous_u0_2 = previous[2][lane];
      u0_old += coeff_previous_u0_2 * field_shape[0][q * U_NS + 2];
      const s_t coeff_previous_u0_3 = previous[3][lane];
      u0_old += coeff_previous_u0_3 * field_shape[0][q * U_NS + 3];
      const s_t coeff_previous_u0_4 = previous[4][lane];
      u0_old += coeff_previous_u0_4 * field_shape[0][q * U_NS + 4];
      const s_t coeff_previous_u0_5 = previous[5][lane];
      u0_old += coeff_previous_u0_5 * field_shape[0][q * U_NS + 5];
      const s_t coeff_previous_u0_6 = previous[6][lane];
      u0_old += coeff_previous_u0_6 * field_shape[0][q * U_NS + 6];
      const s_t coeff_previous_u0_7 = previous[7][lane];
      u0_old += coeff_previous_u0_7 * field_shape[0][q * U_NS + 7];
      const s_t coeff_previous_u0_8 = previous[8][lane];
      u0_old += coeff_previous_u0_8 * field_shape[0][q * U_NS + 8];
      const s_t coeff_previous_u0_9 = previous[9][lane];
      u0_old += coeff_previous_u0_9 * field_shape[0][q * U_NS + 9];
      s_t u1 = s_t(0);
      s_t u1_grad_0_ref = s_t(0);
      s_t u1_grad_1_ref = s_t(0);
      s_t u1_grad_2_ref = s_t(0);
      const s_t coeff_current_u1_0 = current[10][lane];
      u1 += coeff_current_u1_0 * field_shape[0][q * U_NS];
      u1_grad_0_ref += coeff_current_u1_0 * fgref[0][q * U_NS];
      u1_grad_1_ref += coeff_current_u1_0 * fgref[1][q * U_NS];
      u1_grad_2_ref += coeff_current_u1_0 * fgref[2][q * U_NS];
      const s_t coeff_current_u1_1 = current[11][lane];
      u1 += coeff_current_u1_1 * field_shape[0][q * U_NS + 1];
      u1_grad_0_ref += coeff_current_u1_1 * fgref[0][q * U_NS + 1];
      u1_grad_1_ref += coeff_current_u1_1 * fgref[1][q * U_NS + 1];
      u1_grad_2_ref += coeff_current_u1_1 * fgref[2][q * U_NS + 1];
      const s_t coeff_current_u1_2 = current[12][lane];
      u1 += coeff_current_u1_2 * field_shape[0][q * U_NS + 2];
      u1_grad_0_ref += coeff_current_u1_2 * fgref[0][q * U_NS + 2];
      u1_grad_1_ref += coeff_current_u1_2 * fgref[1][q * U_NS + 2];
      u1_grad_2_ref += coeff_current_u1_2 * fgref[2][q * U_NS + 2];
      const s_t coeff_current_u1_3 = current[13][lane];
      u1 += coeff_current_u1_3 * field_shape[0][q * U_NS + 3];
      u1_grad_0_ref += coeff_current_u1_3 * fgref[0][q * U_NS + 3];
      u1_grad_1_ref += coeff_current_u1_3 * fgref[1][q * U_NS + 3];
      u1_grad_2_ref += coeff_current_u1_3 * fgref[2][q * U_NS + 3];
      const s_t coeff_current_u1_4 = current[14][lane];
      u1 += coeff_current_u1_4 * field_shape[0][q * U_NS + 4];
      u1_grad_0_ref += coeff_current_u1_4 * fgref[0][q * U_NS + 4];
      u1_grad_1_ref += coeff_current_u1_4 * fgref[1][q * U_NS + 4];
      u1_grad_2_ref += coeff_current_u1_4 * fgref[2][q * U_NS + 4];
      const s_t coeff_current_u1_5 = current[15][lane];
      u1 += coeff_current_u1_5 * field_shape[0][q * U_NS + 5];
      u1_grad_0_ref += coeff_current_u1_5 * fgref[0][q * U_NS + 5];
      u1_grad_1_ref += coeff_current_u1_5 * fgref[1][q * U_NS + 5];
      u1_grad_2_ref += coeff_current_u1_5 * fgref[2][q * U_NS + 5];
      const s_t coeff_current_u1_6 = current[16][lane];
      u1 += coeff_current_u1_6 * field_shape[0][q * U_NS + 6];
      u1_grad_0_ref += coeff_current_u1_6 * fgref[0][q * U_NS + 6];
      u1_grad_1_ref += coeff_current_u1_6 * fgref[1][q * U_NS + 6];
      u1_grad_2_ref += coeff_current_u1_6 * fgref[2][q * U_NS + 6];
      const s_t coeff_current_u1_7 = current[17][lane];
      u1 += coeff_current_u1_7 * field_shape[0][q * U_NS + 7];
      u1_grad_0_ref += coeff_current_u1_7 * fgref[0][q * U_NS + 7];
      u1_grad_1_ref += coeff_current_u1_7 * fgref[1][q * U_NS + 7];
      u1_grad_2_ref += coeff_current_u1_7 * fgref[2][q * U_NS + 7];
      const s_t coeff_current_u1_8 = current[18][lane];
      u1 += coeff_current_u1_8 * field_shape[0][q * U_NS + 8];
      u1_grad_0_ref += coeff_current_u1_8 * fgref[0][q * U_NS + 8];
      u1_grad_1_ref += coeff_current_u1_8 * fgref[1][q * U_NS + 8];
      u1_grad_2_ref += coeff_current_u1_8 * fgref[2][q * U_NS + 8];
      const s_t coeff_current_u1_9 = current[19][lane];
      u1 += coeff_current_u1_9 * field_shape[0][q * U_NS + 9];
      u1_grad_0_ref += coeff_current_u1_9 * fgref[0][q * U_NS + 9];
      u1_grad_1_ref += coeff_current_u1_9 * fgref[1][q * U_NS + 9];
      u1_grad_2_ref += coeff_current_u1_9 * fgref[2][q * U_NS + 9];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
      const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
      s_t u1_old = s_t(0);
      const s_t coeff_previous_u1_0 = previous[10][lane];
      u1_old += coeff_previous_u1_0 * field_shape[0][q * U_NS];
      const s_t coeff_previous_u1_1 = previous[11][lane];
      u1_old += coeff_previous_u1_1 * field_shape[0][q * U_NS + 1];
      const s_t coeff_previous_u1_2 = previous[12][lane];
      u1_old += coeff_previous_u1_2 * field_shape[0][q * U_NS + 2];
      const s_t coeff_previous_u1_3 = previous[13][lane];
      u1_old += coeff_previous_u1_3 * field_shape[0][q * U_NS + 3];
      const s_t coeff_previous_u1_4 = previous[14][lane];
      u1_old += coeff_previous_u1_4 * field_shape[0][q * U_NS + 4];
      const s_t coeff_previous_u1_5 = previous[15][lane];
      u1_old += coeff_previous_u1_5 * field_shape[0][q * U_NS + 5];
      const s_t coeff_previous_u1_6 = previous[16][lane];
      u1_old += coeff_previous_u1_6 * field_shape[0][q * U_NS + 6];
      const s_t coeff_previous_u1_7 = previous[17][lane];
      u1_old += coeff_previous_u1_7 * field_shape[0][q * U_NS + 7];
      const s_t coeff_previous_u1_8 = previous[18][lane];
      u1_old += coeff_previous_u1_8 * field_shape[0][q * U_NS + 8];
      const s_t coeff_previous_u1_9 = previous[19][lane];
      u1_old += coeff_previous_u1_9 * field_shape[0][q * U_NS + 9];
      s_t u2 = s_t(0);
      s_t u2_grad_0_ref = s_t(0);
      s_t u2_grad_1_ref = s_t(0);
      s_t u2_grad_2_ref = s_t(0);
      const s_t coeff_current_u2_0 = current[20][lane];
      u2 += coeff_current_u2_0 * field_shape[0][q * U_NS];
      u2_grad_0_ref += coeff_current_u2_0 * fgref[0][q * U_NS];
      u2_grad_1_ref += coeff_current_u2_0 * fgref[1][q * U_NS];
      u2_grad_2_ref += coeff_current_u2_0 * fgref[2][q * U_NS];
      const s_t coeff_current_u2_1 = current[21][lane];
      u2 += coeff_current_u2_1 * field_shape[0][q * U_NS + 1];
      u2_grad_0_ref += coeff_current_u2_1 * fgref[0][q * U_NS + 1];
      u2_grad_1_ref += coeff_current_u2_1 * fgref[1][q * U_NS + 1];
      u2_grad_2_ref += coeff_current_u2_1 * fgref[2][q * U_NS + 1];
      const s_t coeff_current_u2_2 = current[22][lane];
      u2 += coeff_current_u2_2 * field_shape[0][q * U_NS + 2];
      u2_grad_0_ref += coeff_current_u2_2 * fgref[0][q * U_NS + 2];
      u2_grad_1_ref += coeff_current_u2_2 * fgref[1][q * U_NS + 2];
      u2_grad_2_ref += coeff_current_u2_2 * fgref[2][q * U_NS + 2];
      const s_t coeff_current_u2_3 = current[23][lane];
      u2 += coeff_current_u2_3 * field_shape[0][q * U_NS + 3];
      u2_grad_0_ref += coeff_current_u2_3 * fgref[0][q * U_NS + 3];
      u2_grad_1_ref += coeff_current_u2_3 * fgref[1][q * U_NS + 3];
      u2_grad_2_ref += coeff_current_u2_3 * fgref[2][q * U_NS + 3];
      const s_t coeff_current_u2_4 = current[24][lane];
      u2 += coeff_current_u2_4 * field_shape[0][q * U_NS + 4];
      u2_grad_0_ref += coeff_current_u2_4 * fgref[0][q * U_NS + 4];
      u2_grad_1_ref += coeff_current_u2_4 * fgref[1][q * U_NS + 4];
      u2_grad_2_ref += coeff_current_u2_4 * fgref[2][q * U_NS + 4];
      const s_t coeff_current_u2_5 = current[25][lane];
      u2 += coeff_current_u2_5 * field_shape[0][q * U_NS + 5];
      u2_grad_0_ref += coeff_current_u2_5 * fgref[0][q * U_NS + 5];
      u2_grad_1_ref += coeff_current_u2_5 * fgref[1][q * U_NS + 5];
      u2_grad_2_ref += coeff_current_u2_5 * fgref[2][q * U_NS + 5];
      const s_t coeff_current_u2_6 = current[26][lane];
      u2 += coeff_current_u2_6 * field_shape[0][q * U_NS + 6];
      u2_grad_0_ref += coeff_current_u2_6 * fgref[0][q * U_NS + 6];
      u2_grad_1_ref += coeff_current_u2_6 * fgref[1][q * U_NS + 6];
      u2_grad_2_ref += coeff_current_u2_6 * fgref[2][q * U_NS + 6];
      const s_t coeff_current_u2_7 = current[27][lane];
      u2 += coeff_current_u2_7 * field_shape[0][q * U_NS + 7];
      u2_grad_0_ref += coeff_current_u2_7 * fgref[0][q * U_NS + 7];
      u2_grad_1_ref += coeff_current_u2_7 * fgref[1][q * U_NS + 7];
      u2_grad_2_ref += coeff_current_u2_7 * fgref[2][q * U_NS + 7];
      const s_t coeff_current_u2_8 = current[28][lane];
      u2 += coeff_current_u2_8 * field_shape[0][q * U_NS + 8];
      u2_grad_0_ref += coeff_current_u2_8 * fgref[0][q * U_NS + 8];
      u2_grad_1_ref += coeff_current_u2_8 * fgref[1][q * U_NS + 8];
      u2_grad_2_ref += coeff_current_u2_8 * fgref[2][q * U_NS + 8];
      const s_t coeff_current_u2_9 = current[29][lane];
      u2 += coeff_current_u2_9 * field_shape[0][q * U_NS + 9];
      u2_grad_0_ref += coeff_current_u2_9 * fgref[0][q * U_NS + 9];
      u2_grad_1_ref += coeff_current_u2_9 * fgref[1][q * U_NS + 9];
      u2_grad_2_ref += coeff_current_u2_9 * fgref[2][q * U_NS + 9];
      const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
      const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
      const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
      s_t u2_old = s_t(0);
      const s_t coeff_previous_u2_0 = previous[20][lane];
      u2_old += coeff_previous_u2_0 * field_shape[0][q * U_NS];
      const s_t coeff_previous_u2_1 = previous[21][lane];
      u2_old += coeff_previous_u2_1 * field_shape[0][q * U_NS + 1];
      const s_t coeff_previous_u2_2 = previous[22][lane];
      u2_old += coeff_previous_u2_2 * field_shape[0][q * U_NS + 2];
      const s_t coeff_previous_u2_3 = previous[23][lane];
      u2_old += coeff_previous_u2_3 * field_shape[0][q * U_NS + 3];
      const s_t coeff_previous_u2_4 = previous[24][lane];
      u2_old += coeff_previous_u2_4 * field_shape[0][q * U_NS + 4];
      const s_t coeff_previous_u2_5 = previous[25][lane];
      u2_old += coeff_previous_u2_5 * field_shape[0][q * U_NS + 5];
      const s_t coeff_previous_u2_6 = previous[26][lane];
      u2_old += coeff_previous_u2_6 * field_shape[0][q * U_NS + 6];
      const s_t coeff_previous_u2_7 = previous[27][lane];
      u2_old += coeff_previous_u2_7 * field_shape[0][q * U_NS + 7];
      const s_t coeff_previous_u2_8 = previous[28][lane];
      u2_old += coeff_previous_u2_8 * field_shape[0][q * U_NS + 8];
      const s_t coeff_previous_u2_9 = previous[29][lane];
      u2_old += coeff_previous_u2_9 * field_shape[0][q * U_NS + 9];
      s_t p = s_t(0);
      const s_t coeff_current_p_0 = current[30][lane];
      p += coeff_current_p_0 * field_shape[1][q * P_NS];
      const s_t coeff_current_p_1 = current[31][lane];
      p += coeff_current_p_1 * field_shape[1][q * P_NS + 1];
      const s_t coeff_current_p_2 = current[32][lane];
      p += coeff_current_p_2 * field_shape[1][q * P_NS + 2];
      const s_t coeff_current_p_3 = current[33][lane];
      p += coeff_current_p_3 * field_shape[1][q * P_NS + 3];
      const s_t residual_tmp0 = rho/dt;
      const s_t residual_tmp1 = -p;
      const s_t residual_tmp2 = nu*rho;
      const s_t residual_tmp3 = s_t(2)*residual_tmp2;
      const s_t residual_tmp4 = residual_tmp2*(u0_grad_1 + u1_grad_0);
      const s_t residual_tmp5 = residual_tmp2*(u0_grad_2 + u2_grad_0);
      const s_t residual_tmp6 = residual_tmp2*(u1_grad_2 + u2_grad_1);
      const s_t value_coeff0 = residual_tmp0*(dt*(convection_scale*(u0_grad_0*u0_old + u0_grad_1*u1_old + u0_grad_2*u2_old) - f0) + u0 - u0_old);
      const s_t grad_coeff0_0 = residual_tmp1 + residual_tmp3*u0_grad_0;
      const s_t grad_coeff0_1 = residual_tmp4;
      const s_t grad_coeff0_2 = residual_tmp5;
      const s_t value_coeff1 = residual_tmp0*(dt*(convection_scale*(u0_old*u1_grad_0 + u1_grad_1*u1_old + u1_grad_2*u2_old) - f1) + u1 - u1_old);
      const s_t grad_coeff1_0 = residual_tmp4;
      const s_t grad_coeff1_1 = residual_tmp1 + residual_tmp3*u1_grad_1;
      const s_t grad_coeff1_2 = residual_tmp6;
      const s_t value_coeff2 = residual_tmp0*(dt*(convection_scale*(u0_old*u2_grad_0 + u1_old*u2_grad_1 + u2_grad_2*u2_old) - f2) + u2 - u2_old);
      const s_t grad_coeff2_0 = residual_tmp5;
      const s_t grad_coeff2_1 = residual_tmp6;
      const s_t grad_coeff2_2 = residual_tmp1 + residual_tmp3*u2_grad_2;
      const s_t test_value_u0_0 = field_shape[0][q * U_NS];
      const s_t test_grad0_u0_0 = (fgref[0][q * U_NS] * adj0 + fgref[1][q * U_NS] * adj3 + fgref[2][q * U_NS] * adj6) / det;
      const s_t test_grad1_u0_0 = (fgref[0][q * U_NS] * adj1 + fgref[1][q * U_NS] * adj4 + fgref[2][q * U_NS] * adj7) / det;
      const s_t test_grad2_u0_0 = (fgref[0][q * U_NS] * adj2 + fgref[1][q * U_NS] * adj5 + fgref[2][q * U_NS] * adj8) / det;
      output[0][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_0 + grad_coeff0_0 * test_grad0_u0_0 + grad_coeff0_1 * test_grad1_u0_0 + grad_coeff0_2 * test_grad2_u0_0);
      const s_t test_value_u0_1 = field_shape[0][q * U_NS + 1];
      const s_t test_grad0_u0_1 = (fgref[0][q * U_NS + 1] * adj0 + fgref[1][q * U_NS + 1] * adj3 + fgref[2][q * U_NS + 1] * adj6) / det;
      const s_t test_grad1_u0_1 = (fgref[0][q * U_NS + 1] * adj1 + fgref[1][q * U_NS + 1] * adj4 + fgref[2][q * U_NS + 1] * adj7) / det;
      const s_t test_grad2_u0_1 = (fgref[0][q * U_NS + 1] * adj2 + fgref[1][q * U_NS + 1] * adj5 + fgref[2][q * U_NS + 1] * adj8) / det;
      output[1][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_1 + grad_coeff0_0 * test_grad0_u0_1 + grad_coeff0_1 * test_grad1_u0_1 + grad_coeff0_2 * test_grad2_u0_1);
      const s_t test_value_u0_2 = field_shape[0][q * U_NS + 2];
      const s_t test_grad0_u0_2 = (fgref[0][q * U_NS + 2] * adj0 + fgref[1][q * U_NS + 2] * adj3 + fgref[2][q * U_NS + 2] * adj6) / det;
      const s_t test_grad1_u0_2 = (fgref[0][q * U_NS + 2] * adj1 + fgref[1][q * U_NS + 2] * adj4 + fgref[2][q * U_NS + 2] * adj7) / det;
      const s_t test_grad2_u0_2 = (fgref[0][q * U_NS + 2] * adj2 + fgref[1][q * U_NS + 2] * adj5 + fgref[2][q * U_NS + 2] * adj8) / det;
      output[2][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_2 + grad_coeff0_0 * test_grad0_u0_2 + grad_coeff0_1 * test_grad1_u0_2 + grad_coeff0_2 * test_grad2_u0_2);
      const s_t test_value_u0_3 = field_shape[0][q * U_NS + 3];
      const s_t test_grad0_u0_3 = (fgref[0][q * U_NS + 3] * adj0 + fgref[1][q * U_NS + 3] * adj3 + fgref[2][q * U_NS + 3] * adj6) / det;
      const s_t test_grad1_u0_3 = (fgref[0][q * U_NS + 3] * adj1 + fgref[1][q * U_NS + 3] * adj4 + fgref[2][q * U_NS + 3] * adj7) / det;
      const s_t test_grad2_u0_3 = (fgref[0][q * U_NS + 3] * adj2 + fgref[1][q * U_NS + 3] * adj5 + fgref[2][q * U_NS + 3] * adj8) / det;
      output[3][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_3 + grad_coeff0_0 * test_grad0_u0_3 + grad_coeff0_1 * test_grad1_u0_3 + grad_coeff0_2 * test_grad2_u0_3);
      const s_t test_value_u0_4 = field_shape[0][q * U_NS + 4];
      const s_t test_grad0_u0_4 = (fgref[0][q * U_NS + 4] * adj0 + fgref[1][q * U_NS + 4] * adj3 + fgref[2][q * U_NS + 4] * adj6) / det;
      const s_t test_grad1_u0_4 = (fgref[0][q * U_NS + 4] * adj1 + fgref[1][q * U_NS + 4] * adj4 + fgref[2][q * U_NS + 4] * adj7) / det;
      const s_t test_grad2_u0_4 = (fgref[0][q * U_NS + 4] * adj2 + fgref[1][q * U_NS + 4] * adj5 + fgref[2][q * U_NS + 4] * adj8) / det;
      output[4][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_4 + grad_coeff0_0 * test_grad0_u0_4 + grad_coeff0_1 * test_grad1_u0_4 + grad_coeff0_2 * test_grad2_u0_4);
      const s_t test_value_u0_5 = field_shape[0][q * U_NS + 5];
      const s_t test_grad0_u0_5 = (fgref[0][q * U_NS + 5] * adj0 + fgref[1][q * U_NS + 5] * adj3 + fgref[2][q * U_NS + 5] * adj6) / det;
      const s_t test_grad1_u0_5 = (fgref[0][q * U_NS + 5] * adj1 + fgref[1][q * U_NS + 5] * adj4 + fgref[2][q * U_NS + 5] * adj7) / det;
      const s_t test_grad2_u0_5 = (fgref[0][q * U_NS + 5] * adj2 + fgref[1][q * U_NS + 5] * adj5 + fgref[2][q * U_NS + 5] * adj8) / det;
      output[5][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_5 + grad_coeff0_0 * test_grad0_u0_5 + grad_coeff0_1 * test_grad1_u0_5 + grad_coeff0_2 * test_grad2_u0_5);
      const s_t test_value_u0_6 = field_shape[0][q * U_NS + 6];
      const s_t test_grad0_u0_6 = (fgref[0][q * U_NS + 6] * adj0 + fgref[1][q * U_NS + 6] * adj3 + fgref[2][q * U_NS + 6] * adj6) / det;
      const s_t test_grad1_u0_6 = (fgref[0][q * U_NS + 6] * adj1 + fgref[1][q * U_NS + 6] * adj4 + fgref[2][q * U_NS + 6] * adj7) / det;
      const s_t test_grad2_u0_6 = (fgref[0][q * U_NS + 6] * adj2 + fgref[1][q * U_NS + 6] * adj5 + fgref[2][q * U_NS + 6] * adj8) / det;
      output[6][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_6 + grad_coeff0_0 * test_grad0_u0_6 + grad_coeff0_1 * test_grad1_u0_6 + grad_coeff0_2 * test_grad2_u0_6);
      const s_t test_value_u0_7 = field_shape[0][q * U_NS + 7];
      const s_t test_grad0_u0_7 = (fgref[0][q * U_NS + 7] * adj0 + fgref[1][q * U_NS + 7] * adj3 + fgref[2][q * U_NS + 7] * adj6) / det;
      const s_t test_grad1_u0_7 = (fgref[0][q * U_NS + 7] * adj1 + fgref[1][q * U_NS + 7] * adj4 + fgref[2][q * U_NS + 7] * adj7) / det;
      const s_t test_grad2_u0_7 = (fgref[0][q * U_NS + 7] * adj2 + fgref[1][q * U_NS + 7] * adj5 + fgref[2][q * U_NS + 7] * adj8) / det;
      output[7][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_7 + grad_coeff0_0 * test_grad0_u0_7 + grad_coeff0_1 * test_grad1_u0_7 + grad_coeff0_2 * test_grad2_u0_7);
      const s_t test_value_u0_8 = field_shape[0][q * U_NS + 8];
      const s_t test_grad0_u0_8 = (fgref[0][q * U_NS + 8] * adj0 + fgref[1][q * U_NS + 8] * adj3 + fgref[2][q * U_NS + 8] * adj6) / det;
      const s_t test_grad1_u0_8 = (fgref[0][q * U_NS + 8] * adj1 + fgref[1][q * U_NS + 8] * adj4 + fgref[2][q * U_NS + 8] * adj7) / det;
      const s_t test_grad2_u0_8 = (fgref[0][q * U_NS + 8] * adj2 + fgref[1][q * U_NS + 8] * adj5 + fgref[2][q * U_NS + 8] * adj8) / det;
      output[8][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_8 + grad_coeff0_0 * test_grad0_u0_8 + grad_coeff0_1 * test_grad1_u0_8 + grad_coeff0_2 * test_grad2_u0_8);
      const s_t test_value_u0_9 = field_shape[0][q * U_NS + 9];
      const s_t test_grad0_u0_9 = (fgref[0][q * U_NS + 9] * adj0 + fgref[1][q * U_NS + 9] * adj3 + fgref[2][q * U_NS + 9] * adj6) / det;
      const s_t test_grad1_u0_9 = (fgref[0][q * U_NS + 9] * adj1 + fgref[1][q * U_NS + 9] * adj4 + fgref[2][q * U_NS + 9] * adj7) / det;
      const s_t test_grad2_u0_9 = (fgref[0][q * U_NS + 9] * adj2 + fgref[1][q * U_NS + 9] * adj5 + fgref[2][q * U_NS + 9] * adj8) / det;
      output[9][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_9 + grad_coeff0_0 * test_grad0_u0_9 + grad_coeff0_1 * test_grad1_u0_9 + grad_coeff0_2 * test_grad2_u0_9);
      const s_t test_value_u1_0 = field_shape[0][q * U_NS];
      const s_t test_grad0_u1_0 = (fgref[0][q * U_NS] * adj0 + fgref[1][q * U_NS] * adj3 + fgref[2][q * U_NS] * adj6) / det;
      const s_t test_grad1_u1_0 = (fgref[0][q * U_NS] * adj1 + fgref[1][q * U_NS] * adj4 + fgref[2][q * U_NS] * adj7) / det;
      const s_t test_grad2_u1_0 = (fgref[0][q * U_NS] * adj2 + fgref[1][q * U_NS] * adj5 + fgref[2][q * U_NS] * adj8) / det;
      output[10][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_0 + grad_coeff1_0 * test_grad0_u1_0 + grad_coeff1_1 * test_grad1_u1_0 + grad_coeff1_2 * test_grad2_u1_0);
      const s_t test_value_u1_1 = field_shape[0][q * U_NS + 1];
      const s_t test_grad0_u1_1 = (fgref[0][q * U_NS + 1] * adj0 + fgref[1][q * U_NS + 1] * adj3 + fgref[2][q * U_NS + 1] * adj6) / det;
      const s_t test_grad1_u1_1 = (fgref[0][q * U_NS + 1] * adj1 + fgref[1][q * U_NS + 1] * adj4 + fgref[2][q * U_NS + 1] * adj7) / det;
      const s_t test_grad2_u1_1 = (fgref[0][q * U_NS + 1] * adj2 + fgref[1][q * U_NS + 1] * adj5 + fgref[2][q * U_NS + 1] * adj8) / det;
      output[11][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_1 + grad_coeff1_0 * test_grad0_u1_1 + grad_coeff1_1 * test_grad1_u1_1 + grad_coeff1_2 * test_grad2_u1_1);
      const s_t test_value_u1_2 = field_shape[0][q * U_NS + 2];
      const s_t test_grad0_u1_2 = (fgref[0][q * U_NS + 2] * adj0 + fgref[1][q * U_NS + 2] * adj3 + fgref[2][q * U_NS + 2] * adj6) / det;
      const s_t test_grad1_u1_2 = (fgref[0][q * U_NS + 2] * adj1 + fgref[1][q * U_NS + 2] * adj4 + fgref[2][q * U_NS + 2] * adj7) / det;
      const s_t test_grad2_u1_2 = (fgref[0][q * U_NS + 2] * adj2 + fgref[1][q * U_NS + 2] * adj5 + fgref[2][q * U_NS + 2] * adj8) / det;
      output[12][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_2 + grad_coeff1_0 * test_grad0_u1_2 + grad_coeff1_1 * test_grad1_u1_2 + grad_coeff1_2 * test_grad2_u1_2);
      const s_t test_value_u1_3 = field_shape[0][q * U_NS + 3];
      const s_t test_grad0_u1_3 = (fgref[0][q * U_NS + 3] * adj0 + fgref[1][q * U_NS + 3] * adj3 + fgref[2][q * U_NS + 3] * adj6) / det;
      const s_t test_grad1_u1_3 = (fgref[0][q * U_NS + 3] * adj1 + fgref[1][q * U_NS + 3] * adj4 + fgref[2][q * U_NS + 3] * adj7) / det;
      const s_t test_grad2_u1_3 = (fgref[0][q * U_NS + 3] * adj2 + fgref[1][q * U_NS + 3] * adj5 + fgref[2][q * U_NS + 3] * adj8) / det;
      output[13][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_3 + grad_coeff1_0 * test_grad0_u1_3 + grad_coeff1_1 * test_grad1_u1_3 + grad_coeff1_2 * test_grad2_u1_3);
      const s_t test_value_u1_4 = field_shape[0][q * U_NS + 4];
      const s_t test_grad0_u1_4 = (fgref[0][q * U_NS + 4] * adj0 + fgref[1][q * U_NS + 4] * adj3 + fgref[2][q * U_NS + 4] * adj6) / det;
      const s_t test_grad1_u1_4 = (fgref[0][q * U_NS + 4] * adj1 + fgref[1][q * U_NS + 4] * adj4 + fgref[2][q * U_NS + 4] * adj7) / det;
      const s_t test_grad2_u1_4 = (fgref[0][q * U_NS + 4] * adj2 + fgref[1][q * U_NS + 4] * adj5 + fgref[2][q * U_NS + 4] * adj8) / det;
      output[14][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_4 + grad_coeff1_0 * test_grad0_u1_4 + grad_coeff1_1 * test_grad1_u1_4 + grad_coeff1_2 * test_grad2_u1_4);
      const s_t test_value_u1_5 = field_shape[0][q * U_NS + 5];
      const s_t test_grad0_u1_5 = (fgref[0][q * U_NS + 5] * adj0 + fgref[1][q * U_NS + 5] * adj3 + fgref[2][q * U_NS + 5] * adj6) / det;
      const s_t test_grad1_u1_5 = (fgref[0][q * U_NS + 5] * adj1 + fgref[1][q * U_NS + 5] * adj4 + fgref[2][q * U_NS + 5] * adj7) / det;
      const s_t test_grad2_u1_5 = (fgref[0][q * U_NS + 5] * adj2 + fgref[1][q * U_NS + 5] * adj5 + fgref[2][q * U_NS + 5] * adj8) / det;
      output[15][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_5 + grad_coeff1_0 * test_grad0_u1_5 + grad_coeff1_1 * test_grad1_u1_5 + grad_coeff1_2 * test_grad2_u1_5);
      const s_t test_value_u1_6 = field_shape[0][q * U_NS + 6];
      const s_t test_grad0_u1_6 = (fgref[0][q * U_NS + 6] * adj0 + fgref[1][q * U_NS + 6] * adj3 + fgref[2][q * U_NS + 6] * adj6) / det;
      const s_t test_grad1_u1_6 = (fgref[0][q * U_NS + 6] * adj1 + fgref[1][q * U_NS + 6] * adj4 + fgref[2][q * U_NS + 6] * adj7) / det;
      const s_t test_grad2_u1_6 = (fgref[0][q * U_NS + 6] * adj2 + fgref[1][q * U_NS + 6] * adj5 + fgref[2][q * U_NS + 6] * adj8) / det;
      output[16][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_6 + grad_coeff1_0 * test_grad0_u1_6 + grad_coeff1_1 * test_grad1_u1_6 + grad_coeff1_2 * test_grad2_u1_6);
      const s_t test_value_u1_7 = field_shape[0][q * U_NS + 7];
      const s_t test_grad0_u1_7 = (fgref[0][q * U_NS + 7] * adj0 + fgref[1][q * U_NS + 7] * adj3 + fgref[2][q * U_NS + 7] * adj6) / det;
      const s_t test_grad1_u1_7 = (fgref[0][q * U_NS + 7] * adj1 + fgref[1][q * U_NS + 7] * adj4 + fgref[2][q * U_NS + 7] * adj7) / det;
      const s_t test_grad2_u1_7 = (fgref[0][q * U_NS + 7] * adj2 + fgref[1][q * U_NS + 7] * adj5 + fgref[2][q * U_NS + 7] * adj8) / det;
      output[17][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_7 + grad_coeff1_0 * test_grad0_u1_7 + grad_coeff1_1 * test_grad1_u1_7 + grad_coeff1_2 * test_grad2_u1_7);
      const s_t test_value_u1_8 = field_shape[0][q * U_NS + 8];
      const s_t test_grad0_u1_8 = (fgref[0][q * U_NS + 8] * adj0 + fgref[1][q * U_NS + 8] * adj3 + fgref[2][q * U_NS + 8] * adj6) / det;
      const s_t test_grad1_u1_8 = (fgref[0][q * U_NS + 8] * adj1 + fgref[1][q * U_NS + 8] * adj4 + fgref[2][q * U_NS + 8] * adj7) / det;
      const s_t test_grad2_u1_8 = (fgref[0][q * U_NS + 8] * adj2 + fgref[1][q * U_NS + 8] * adj5 + fgref[2][q * U_NS + 8] * adj8) / det;
      output[18][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_8 + grad_coeff1_0 * test_grad0_u1_8 + grad_coeff1_1 * test_grad1_u1_8 + grad_coeff1_2 * test_grad2_u1_8);
      const s_t test_value_u1_9 = field_shape[0][q * U_NS + 9];
      const s_t test_grad0_u1_9 = (fgref[0][q * U_NS + 9] * adj0 + fgref[1][q * U_NS + 9] * adj3 + fgref[2][q * U_NS + 9] * adj6) / det;
      const s_t test_grad1_u1_9 = (fgref[0][q * U_NS + 9] * adj1 + fgref[1][q * U_NS + 9] * adj4 + fgref[2][q * U_NS + 9] * adj7) / det;
      const s_t test_grad2_u1_9 = (fgref[0][q * U_NS + 9] * adj2 + fgref[1][q * U_NS + 9] * adj5 + fgref[2][q * U_NS + 9] * adj8) / det;
      output[19][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_9 + grad_coeff1_0 * test_grad0_u1_9 + grad_coeff1_1 * test_grad1_u1_9 + grad_coeff1_2 * test_grad2_u1_9);
      const s_t test_value_u2_0 = field_shape[0][q * U_NS];
      const s_t test_grad0_u2_0 = (fgref[0][q * U_NS] * adj0 + fgref[1][q * U_NS] * adj3 + fgref[2][q * U_NS] * adj6) / det;
      const s_t test_grad1_u2_0 = (fgref[0][q * U_NS] * adj1 + fgref[1][q * U_NS] * adj4 + fgref[2][q * U_NS] * adj7) / det;
      const s_t test_grad2_u2_0 = (fgref[0][q * U_NS] * adj2 + fgref[1][q * U_NS] * adj5 + fgref[2][q * U_NS] * adj8) / det;
      output[20][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_0 + grad_coeff2_0 * test_grad0_u2_0 + grad_coeff2_1 * test_grad1_u2_0 + grad_coeff2_2 * test_grad2_u2_0);
      const s_t test_value_u2_1 = field_shape[0][q * U_NS + 1];
      const s_t test_grad0_u2_1 = (fgref[0][q * U_NS + 1] * adj0 + fgref[1][q * U_NS + 1] * adj3 + fgref[2][q * U_NS + 1] * adj6) / det;
      const s_t test_grad1_u2_1 = (fgref[0][q * U_NS + 1] * adj1 + fgref[1][q * U_NS + 1] * adj4 + fgref[2][q * U_NS + 1] * adj7) / det;
      const s_t test_grad2_u2_1 = (fgref[0][q * U_NS + 1] * adj2 + fgref[1][q * U_NS + 1] * adj5 + fgref[2][q * U_NS + 1] * adj8) / det;
      output[21][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_1 + grad_coeff2_0 * test_grad0_u2_1 + grad_coeff2_1 * test_grad1_u2_1 + grad_coeff2_2 * test_grad2_u2_1);
      const s_t test_value_u2_2 = field_shape[0][q * U_NS + 2];
      const s_t test_grad0_u2_2 = (fgref[0][q * U_NS + 2] * adj0 + fgref[1][q * U_NS + 2] * adj3 + fgref[2][q * U_NS + 2] * adj6) / det;
      const s_t test_grad1_u2_2 = (fgref[0][q * U_NS + 2] * adj1 + fgref[1][q * U_NS + 2] * adj4 + fgref[2][q * U_NS + 2] * adj7) / det;
      const s_t test_grad2_u2_2 = (fgref[0][q * U_NS + 2] * adj2 + fgref[1][q * U_NS + 2] * adj5 + fgref[2][q * U_NS + 2] * adj8) / det;
      output[22][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_2 + grad_coeff2_0 * test_grad0_u2_2 + grad_coeff2_1 * test_grad1_u2_2 + grad_coeff2_2 * test_grad2_u2_2);
      const s_t test_value_u2_3 = field_shape[0][q * U_NS + 3];
      const s_t test_grad0_u2_3 = (fgref[0][q * U_NS + 3] * adj0 + fgref[1][q * U_NS + 3] * adj3 + fgref[2][q * U_NS + 3] * adj6) / det;
      const s_t test_grad1_u2_3 = (fgref[0][q * U_NS + 3] * adj1 + fgref[1][q * U_NS + 3] * adj4 + fgref[2][q * U_NS + 3] * adj7) / det;
      const s_t test_grad2_u2_3 = (fgref[0][q * U_NS + 3] * adj2 + fgref[1][q * U_NS + 3] * adj5 + fgref[2][q * U_NS + 3] * adj8) / det;
      output[23][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_3 + grad_coeff2_0 * test_grad0_u2_3 + grad_coeff2_1 * test_grad1_u2_3 + grad_coeff2_2 * test_grad2_u2_3);
      const s_t test_value_u2_4 = field_shape[0][q * U_NS + 4];
      const s_t test_grad0_u2_4 = (fgref[0][q * U_NS + 4] * adj0 + fgref[1][q * U_NS + 4] * adj3 + fgref[2][q * U_NS + 4] * adj6) / det;
      const s_t test_grad1_u2_4 = (fgref[0][q * U_NS + 4] * adj1 + fgref[1][q * U_NS + 4] * adj4 + fgref[2][q * U_NS + 4] * adj7) / det;
      const s_t test_grad2_u2_4 = (fgref[0][q * U_NS + 4] * adj2 + fgref[1][q * U_NS + 4] * adj5 + fgref[2][q * U_NS + 4] * adj8) / det;
      output[24][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_4 + grad_coeff2_0 * test_grad0_u2_4 + grad_coeff2_1 * test_grad1_u2_4 + grad_coeff2_2 * test_grad2_u2_4);
      const s_t test_value_u2_5 = field_shape[0][q * U_NS + 5];
      const s_t test_grad0_u2_5 = (fgref[0][q * U_NS + 5] * adj0 + fgref[1][q * U_NS + 5] * adj3 + fgref[2][q * U_NS + 5] * adj6) / det;
      const s_t test_grad1_u2_5 = (fgref[0][q * U_NS + 5] * adj1 + fgref[1][q * U_NS + 5] * adj4 + fgref[2][q * U_NS + 5] * adj7) / det;
      const s_t test_grad2_u2_5 = (fgref[0][q * U_NS + 5] * adj2 + fgref[1][q * U_NS + 5] * adj5 + fgref[2][q * U_NS + 5] * adj8) / det;
      output[25][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_5 + grad_coeff2_0 * test_grad0_u2_5 + grad_coeff2_1 * test_grad1_u2_5 + grad_coeff2_2 * test_grad2_u2_5);
      const s_t test_value_u2_6 = field_shape[0][q * U_NS + 6];
      const s_t test_grad0_u2_6 = (fgref[0][q * U_NS + 6] * adj0 + fgref[1][q * U_NS + 6] * adj3 + fgref[2][q * U_NS + 6] * adj6) / det;
      const s_t test_grad1_u2_6 = (fgref[0][q * U_NS + 6] * adj1 + fgref[1][q * U_NS + 6] * adj4 + fgref[2][q * U_NS + 6] * adj7) / det;
      const s_t test_grad2_u2_6 = (fgref[0][q * U_NS + 6] * adj2 + fgref[1][q * U_NS + 6] * adj5 + fgref[2][q * U_NS + 6] * adj8) / det;
      output[26][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_6 + grad_coeff2_0 * test_grad0_u2_6 + grad_coeff2_1 * test_grad1_u2_6 + grad_coeff2_2 * test_grad2_u2_6);
      const s_t test_value_u2_7 = field_shape[0][q * U_NS + 7];
      const s_t test_grad0_u2_7 = (fgref[0][q * U_NS + 7] * adj0 + fgref[1][q * U_NS + 7] * adj3 + fgref[2][q * U_NS + 7] * adj6) / det;
      const s_t test_grad1_u2_7 = (fgref[0][q * U_NS + 7] * adj1 + fgref[1][q * U_NS + 7] * adj4 + fgref[2][q * U_NS + 7] * adj7) / det;
      const s_t test_grad2_u2_7 = (fgref[0][q * U_NS + 7] * adj2 + fgref[1][q * U_NS + 7] * adj5 + fgref[2][q * U_NS + 7] * adj8) / det;
      output[27][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_7 + grad_coeff2_0 * test_grad0_u2_7 + grad_coeff2_1 * test_grad1_u2_7 + grad_coeff2_2 * test_grad2_u2_7);
      const s_t test_value_u2_8 = field_shape[0][q * U_NS + 8];
      const s_t test_grad0_u2_8 = (fgref[0][q * U_NS + 8] * adj0 + fgref[1][q * U_NS + 8] * adj3 + fgref[2][q * U_NS + 8] * adj6) / det;
      const s_t test_grad1_u2_8 = (fgref[0][q * U_NS + 8] * adj1 + fgref[1][q * U_NS + 8] * adj4 + fgref[2][q * U_NS + 8] * adj7) / det;
      const s_t test_grad2_u2_8 = (fgref[0][q * U_NS + 8] * adj2 + fgref[1][q * U_NS + 8] * adj5 + fgref[2][q * U_NS + 8] * adj8) / det;
      output[28][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_8 + grad_coeff2_0 * test_grad0_u2_8 + grad_coeff2_1 * test_grad1_u2_8 + grad_coeff2_2 * test_grad2_u2_8);
      const s_t test_value_u2_9 = field_shape[0][q * U_NS + 9];
      const s_t test_grad0_u2_9 = (fgref[0][q * U_NS + 9] * adj0 + fgref[1][q * U_NS + 9] * adj3 + fgref[2][q * U_NS + 9] * adj6) / det;
      const s_t test_grad1_u2_9 = (fgref[0][q * U_NS + 9] * adj1 + fgref[1][q * U_NS + 9] * adj4 + fgref[2][q * U_NS + 9] * adj7) / det;
      const s_t test_grad2_u2_9 = (fgref[0][q * U_NS + 9] * adj2 + fgref[1][q * U_NS + 9] * adj5 + fgref[2][q * U_NS + 9] * adj8) / det;
      output[29][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_9 + grad_coeff2_0 * test_grad0_u2_9 + grad_coeff2_1 * test_grad1_u2_9 + grad_coeff2_2 * test_grad2_u2_9);
    }
  }
}

template <typename s_t, int NQ, int CELL_NS, int VS>
static SFEM_INLINE void navier_stokes_form_1_u_d3_simplex_mixed_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR field_shape[2],
    const s_t *const RSTR fgref[6],
    const s_t *const RSTR q_weight,
    const s_t current[34][VS],
    const s_t previous[34][VS],
    const s_t convection_scale,
    const s_t dt,
    const s_t f0,
    const s_t f1,
    const s_t f2,
    const s_t nu,
    const s_t rho,
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
      s_t u0 = s_t(0);
      s_t u0_grad_0_ref = s_t(0);
      s_t u0_grad_1_ref = s_t(0);
      s_t u0_grad_2_ref = s_t(0);
      const s_t coeff_current_u0_0 = current[0][lane];
      u0 += coeff_current_u0_0 * field_shape[0][q * U_NS];
      u0_grad_0_ref += coeff_current_u0_0 * fgref[0][q * U_NS];
      u0_grad_1_ref += coeff_current_u0_0 * fgref[1][q * U_NS];
      u0_grad_2_ref += coeff_current_u0_0 * fgref[2][q * U_NS];
      const s_t coeff_current_u0_1 = current[1][lane];
      u0 += coeff_current_u0_1 * field_shape[0][q * U_NS + 1];
      u0_grad_0_ref += coeff_current_u0_1 * fgref[0][q * U_NS + 1];
      u0_grad_1_ref += coeff_current_u0_1 * fgref[1][q * U_NS + 1];
      u0_grad_2_ref += coeff_current_u0_1 * fgref[2][q * U_NS + 1];
      const s_t coeff_current_u0_2 = current[2][lane];
      u0 += coeff_current_u0_2 * field_shape[0][q * U_NS + 2];
      u0_grad_0_ref += coeff_current_u0_2 * fgref[0][q * U_NS + 2];
      u0_grad_1_ref += coeff_current_u0_2 * fgref[1][q * U_NS + 2];
      u0_grad_2_ref += coeff_current_u0_2 * fgref[2][q * U_NS + 2];
      const s_t coeff_current_u0_3 = current[3][lane];
      u0 += coeff_current_u0_3 * field_shape[0][q * U_NS + 3];
      u0_grad_0_ref += coeff_current_u0_3 * fgref[0][q * U_NS + 3];
      u0_grad_1_ref += coeff_current_u0_3 * fgref[1][q * U_NS + 3];
      u0_grad_2_ref += coeff_current_u0_3 * fgref[2][q * U_NS + 3];
      const s_t coeff_current_u0_4 = current[4][lane];
      u0 += coeff_current_u0_4 * field_shape[0][q * U_NS + 4];
      u0_grad_0_ref += coeff_current_u0_4 * fgref[0][q * U_NS + 4];
      u0_grad_1_ref += coeff_current_u0_4 * fgref[1][q * U_NS + 4];
      u0_grad_2_ref += coeff_current_u0_4 * fgref[2][q * U_NS + 4];
      const s_t coeff_current_u0_5 = current[5][lane];
      u0 += coeff_current_u0_5 * field_shape[0][q * U_NS + 5];
      u0_grad_0_ref += coeff_current_u0_5 * fgref[0][q * U_NS + 5];
      u0_grad_1_ref += coeff_current_u0_5 * fgref[1][q * U_NS + 5];
      u0_grad_2_ref += coeff_current_u0_5 * fgref[2][q * U_NS + 5];
      const s_t coeff_current_u0_6 = current[6][lane];
      u0 += coeff_current_u0_6 * field_shape[0][q * U_NS + 6];
      u0_grad_0_ref += coeff_current_u0_6 * fgref[0][q * U_NS + 6];
      u0_grad_1_ref += coeff_current_u0_6 * fgref[1][q * U_NS + 6];
      u0_grad_2_ref += coeff_current_u0_6 * fgref[2][q * U_NS + 6];
      const s_t coeff_current_u0_7 = current[7][lane];
      u0 += coeff_current_u0_7 * field_shape[0][q * U_NS + 7];
      u0_grad_0_ref += coeff_current_u0_7 * fgref[0][q * U_NS + 7];
      u0_grad_1_ref += coeff_current_u0_7 * fgref[1][q * U_NS + 7];
      u0_grad_2_ref += coeff_current_u0_7 * fgref[2][q * U_NS + 7];
      const s_t coeff_current_u0_8 = current[8][lane];
      u0 += coeff_current_u0_8 * field_shape[0][q * U_NS + 8];
      u0_grad_0_ref += coeff_current_u0_8 * fgref[0][q * U_NS + 8];
      u0_grad_1_ref += coeff_current_u0_8 * fgref[1][q * U_NS + 8];
      u0_grad_2_ref += coeff_current_u0_8 * fgref[2][q * U_NS + 8];
      const s_t coeff_current_u0_9 = current[9][lane];
      u0 += coeff_current_u0_9 * field_shape[0][q * U_NS + 9];
      u0_grad_0_ref += coeff_current_u0_9 * fgref[0][q * U_NS + 9];
      u0_grad_1_ref += coeff_current_u0_9 * fgref[1][q * U_NS + 9];
      u0_grad_2_ref += coeff_current_u0_9 * fgref[2][q * U_NS + 9];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
      const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
      const s_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
      s_t u0_old = s_t(0);
      const s_t coeff_previous_u0_0 = previous[0][lane];
      u0_old += coeff_previous_u0_0 * field_shape[0][q * U_NS];
      const s_t coeff_previous_u0_1 = previous[1][lane];
      u0_old += coeff_previous_u0_1 * field_shape[0][q * U_NS + 1];
      const s_t coeff_previous_u0_2 = previous[2][lane];
      u0_old += coeff_previous_u0_2 * field_shape[0][q * U_NS + 2];
      const s_t coeff_previous_u0_3 = previous[3][lane];
      u0_old += coeff_previous_u0_3 * field_shape[0][q * U_NS + 3];
      const s_t coeff_previous_u0_4 = previous[4][lane];
      u0_old += coeff_previous_u0_4 * field_shape[0][q * U_NS + 4];
      const s_t coeff_previous_u0_5 = previous[5][lane];
      u0_old += coeff_previous_u0_5 * field_shape[0][q * U_NS + 5];
      const s_t coeff_previous_u0_6 = previous[6][lane];
      u0_old += coeff_previous_u0_6 * field_shape[0][q * U_NS + 6];
      const s_t coeff_previous_u0_7 = previous[7][lane];
      u0_old += coeff_previous_u0_7 * field_shape[0][q * U_NS + 7];
      const s_t coeff_previous_u0_8 = previous[8][lane];
      u0_old += coeff_previous_u0_8 * field_shape[0][q * U_NS + 8];
      const s_t coeff_previous_u0_9 = previous[9][lane];
      u0_old += coeff_previous_u0_9 * field_shape[0][q * U_NS + 9];
      s_t u1 = s_t(0);
      s_t u1_grad_0_ref = s_t(0);
      s_t u1_grad_1_ref = s_t(0);
      s_t u1_grad_2_ref = s_t(0);
      const s_t coeff_current_u1_0 = current[10][lane];
      u1 += coeff_current_u1_0 * field_shape[0][q * U_NS];
      u1_grad_0_ref += coeff_current_u1_0 * fgref[0][q * U_NS];
      u1_grad_1_ref += coeff_current_u1_0 * fgref[1][q * U_NS];
      u1_grad_2_ref += coeff_current_u1_0 * fgref[2][q * U_NS];
      const s_t coeff_current_u1_1 = current[11][lane];
      u1 += coeff_current_u1_1 * field_shape[0][q * U_NS + 1];
      u1_grad_0_ref += coeff_current_u1_1 * fgref[0][q * U_NS + 1];
      u1_grad_1_ref += coeff_current_u1_1 * fgref[1][q * U_NS + 1];
      u1_grad_2_ref += coeff_current_u1_1 * fgref[2][q * U_NS + 1];
      const s_t coeff_current_u1_2 = current[12][lane];
      u1 += coeff_current_u1_2 * field_shape[0][q * U_NS + 2];
      u1_grad_0_ref += coeff_current_u1_2 * fgref[0][q * U_NS + 2];
      u1_grad_1_ref += coeff_current_u1_2 * fgref[1][q * U_NS + 2];
      u1_grad_2_ref += coeff_current_u1_2 * fgref[2][q * U_NS + 2];
      const s_t coeff_current_u1_3 = current[13][lane];
      u1 += coeff_current_u1_3 * field_shape[0][q * U_NS + 3];
      u1_grad_0_ref += coeff_current_u1_3 * fgref[0][q * U_NS + 3];
      u1_grad_1_ref += coeff_current_u1_3 * fgref[1][q * U_NS + 3];
      u1_grad_2_ref += coeff_current_u1_3 * fgref[2][q * U_NS + 3];
      const s_t coeff_current_u1_4 = current[14][lane];
      u1 += coeff_current_u1_4 * field_shape[0][q * U_NS + 4];
      u1_grad_0_ref += coeff_current_u1_4 * fgref[0][q * U_NS + 4];
      u1_grad_1_ref += coeff_current_u1_4 * fgref[1][q * U_NS + 4];
      u1_grad_2_ref += coeff_current_u1_4 * fgref[2][q * U_NS + 4];
      const s_t coeff_current_u1_5 = current[15][lane];
      u1 += coeff_current_u1_5 * field_shape[0][q * U_NS + 5];
      u1_grad_0_ref += coeff_current_u1_5 * fgref[0][q * U_NS + 5];
      u1_grad_1_ref += coeff_current_u1_5 * fgref[1][q * U_NS + 5];
      u1_grad_2_ref += coeff_current_u1_5 * fgref[2][q * U_NS + 5];
      const s_t coeff_current_u1_6 = current[16][lane];
      u1 += coeff_current_u1_6 * field_shape[0][q * U_NS + 6];
      u1_grad_0_ref += coeff_current_u1_6 * fgref[0][q * U_NS + 6];
      u1_grad_1_ref += coeff_current_u1_6 * fgref[1][q * U_NS + 6];
      u1_grad_2_ref += coeff_current_u1_6 * fgref[2][q * U_NS + 6];
      const s_t coeff_current_u1_7 = current[17][lane];
      u1 += coeff_current_u1_7 * field_shape[0][q * U_NS + 7];
      u1_grad_0_ref += coeff_current_u1_7 * fgref[0][q * U_NS + 7];
      u1_grad_1_ref += coeff_current_u1_7 * fgref[1][q * U_NS + 7];
      u1_grad_2_ref += coeff_current_u1_7 * fgref[2][q * U_NS + 7];
      const s_t coeff_current_u1_8 = current[18][lane];
      u1 += coeff_current_u1_8 * field_shape[0][q * U_NS + 8];
      u1_grad_0_ref += coeff_current_u1_8 * fgref[0][q * U_NS + 8];
      u1_grad_1_ref += coeff_current_u1_8 * fgref[1][q * U_NS + 8];
      u1_grad_2_ref += coeff_current_u1_8 * fgref[2][q * U_NS + 8];
      const s_t coeff_current_u1_9 = current[19][lane];
      u1 += coeff_current_u1_9 * field_shape[0][q * U_NS + 9];
      u1_grad_0_ref += coeff_current_u1_9 * fgref[0][q * U_NS + 9];
      u1_grad_1_ref += coeff_current_u1_9 * fgref[1][q * U_NS + 9];
      u1_grad_2_ref += coeff_current_u1_9 * fgref[2][q * U_NS + 9];
      const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
      const s_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
      s_t u1_old = s_t(0);
      const s_t coeff_previous_u1_0 = previous[10][lane];
      u1_old += coeff_previous_u1_0 * field_shape[0][q * U_NS];
      const s_t coeff_previous_u1_1 = previous[11][lane];
      u1_old += coeff_previous_u1_1 * field_shape[0][q * U_NS + 1];
      const s_t coeff_previous_u1_2 = previous[12][lane];
      u1_old += coeff_previous_u1_2 * field_shape[0][q * U_NS + 2];
      const s_t coeff_previous_u1_3 = previous[13][lane];
      u1_old += coeff_previous_u1_3 * field_shape[0][q * U_NS + 3];
      const s_t coeff_previous_u1_4 = previous[14][lane];
      u1_old += coeff_previous_u1_4 * field_shape[0][q * U_NS + 4];
      const s_t coeff_previous_u1_5 = previous[15][lane];
      u1_old += coeff_previous_u1_5 * field_shape[0][q * U_NS + 5];
      const s_t coeff_previous_u1_6 = previous[16][lane];
      u1_old += coeff_previous_u1_6 * field_shape[0][q * U_NS + 6];
      const s_t coeff_previous_u1_7 = previous[17][lane];
      u1_old += coeff_previous_u1_7 * field_shape[0][q * U_NS + 7];
      const s_t coeff_previous_u1_8 = previous[18][lane];
      u1_old += coeff_previous_u1_8 * field_shape[0][q * U_NS + 8];
      const s_t coeff_previous_u1_9 = previous[19][lane];
      u1_old += coeff_previous_u1_9 * field_shape[0][q * U_NS + 9];
      s_t u2 = s_t(0);
      s_t u2_grad_0_ref = s_t(0);
      s_t u2_grad_1_ref = s_t(0);
      s_t u2_grad_2_ref = s_t(0);
      const s_t coeff_current_u2_0 = current[20][lane];
      u2 += coeff_current_u2_0 * field_shape[0][q * U_NS];
      u2_grad_0_ref += coeff_current_u2_0 * fgref[0][q * U_NS];
      u2_grad_1_ref += coeff_current_u2_0 * fgref[1][q * U_NS];
      u2_grad_2_ref += coeff_current_u2_0 * fgref[2][q * U_NS];
      const s_t coeff_current_u2_1 = current[21][lane];
      u2 += coeff_current_u2_1 * field_shape[0][q * U_NS + 1];
      u2_grad_0_ref += coeff_current_u2_1 * fgref[0][q * U_NS + 1];
      u2_grad_1_ref += coeff_current_u2_1 * fgref[1][q * U_NS + 1];
      u2_grad_2_ref += coeff_current_u2_1 * fgref[2][q * U_NS + 1];
      const s_t coeff_current_u2_2 = current[22][lane];
      u2 += coeff_current_u2_2 * field_shape[0][q * U_NS + 2];
      u2_grad_0_ref += coeff_current_u2_2 * fgref[0][q * U_NS + 2];
      u2_grad_1_ref += coeff_current_u2_2 * fgref[1][q * U_NS + 2];
      u2_grad_2_ref += coeff_current_u2_2 * fgref[2][q * U_NS + 2];
      const s_t coeff_current_u2_3 = current[23][lane];
      u2 += coeff_current_u2_3 * field_shape[0][q * U_NS + 3];
      u2_grad_0_ref += coeff_current_u2_3 * fgref[0][q * U_NS + 3];
      u2_grad_1_ref += coeff_current_u2_3 * fgref[1][q * U_NS + 3];
      u2_grad_2_ref += coeff_current_u2_3 * fgref[2][q * U_NS + 3];
      const s_t coeff_current_u2_4 = current[24][lane];
      u2 += coeff_current_u2_4 * field_shape[0][q * U_NS + 4];
      u2_grad_0_ref += coeff_current_u2_4 * fgref[0][q * U_NS + 4];
      u2_grad_1_ref += coeff_current_u2_4 * fgref[1][q * U_NS + 4];
      u2_grad_2_ref += coeff_current_u2_4 * fgref[2][q * U_NS + 4];
      const s_t coeff_current_u2_5 = current[25][lane];
      u2 += coeff_current_u2_5 * field_shape[0][q * U_NS + 5];
      u2_grad_0_ref += coeff_current_u2_5 * fgref[0][q * U_NS + 5];
      u2_grad_1_ref += coeff_current_u2_5 * fgref[1][q * U_NS + 5];
      u2_grad_2_ref += coeff_current_u2_5 * fgref[2][q * U_NS + 5];
      const s_t coeff_current_u2_6 = current[26][lane];
      u2 += coeff_current_u2_6 * field_shape[0][q * U_NS + 6];
      u2_grad_0_ref += coeff_current_u2_6 * fgref[0][q * U_NS + 6];
      u2_grad_1_ref += coeff_current_u2_6 * fgref[1][q * U_NS + 6];
      u2_grad_2_ref += coeff_current_u2_6 * fgref[2][q * U_NS + 6];
      const s_t coeff_current_u2_7 = current[27][lane];
      u2 += coeff_current_u2_7 * field_shape[0][q * U_NS + 7];
      u2_grad_0_ref += coeff_current_u2_7 * fgref[0][q * U_NS + 7];
      u2_grad_1_ref += coeff_current_u2_7 * fgref[1][q * U_NS + 7];
      u2_grad_2_ref += coeff_current_u2_7 * fgref[2][q * U_NS + 7];
      const s_t coeff_current_u2_8 = current[28][lane];
      u2 += coeff_current_u2_8 * field_shape[0][q * U_NS + 8];
      u2_grad_0_ref += coeff_current_u2_8 * fgref[0][q * U_NS + 8];
      u2_grad_1_ref += coeff_current_u2_8 * fgref[1][q * U_NS + 8];
      u2_grad_2_ref += coeff_current_u2_8 * fgref[2][q * U_NS + 8];
      const s_t coeff_current_u2_9 = current[29][lane];
      u2 += coeff_current_u2_9 * field_shape[0][q * U_NS + 9];
      u2_grad_0_ref += coeff_current_u2_9 * fgref[0][q * U_NS + 9];
      u2_grad_1_ref += coeff_current_u2_9 * fgref[1][q * U_NS + 9];
      u2_grad_2_ref += coeff_current_u2_9 * fgref[2][q * U_NS + 9];
      const s_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
      const s_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
      const s_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
      s_t u2_old = s_t(0);
      const s_t coeff_previous_u2_0 = previous[20][lane];
      u2_old += coeff_previous_u2_0 * field_shape[0][q * U_NS];
      const s_t coeff_previous_u2_1 = previous[21][lane];
      u2_old += coeff_previous_u2_1 * field_shape[0][q * U_NS + 1];
      const s_t coeff_previous_u2_2 = previous[22][lane];
      u2_old += coeff_previous_u2_2 * field_shape[0][q * U_NS + 2];
      const s_t coeff_previous_u2_3 = previous[23][lane];
      u2_old += coeff_previous_u2_3 * field_shape[0][q * U_NS + 3];
      const s_t coeff_previous_u2_4 = previous[24][lane];
      u2_old += coeff_previous_u2_4 * field_shape[0][q * U_NS + 4];
      const s_t coeff_previous_u2_5 = previous[25][lane];
      u2_old += coeff_previous_u2_5 * field_shape[0][q * U_NS + 5];
      const s_t coeff_previous_u2_6 = previous[26][lane];
      u2_old += coeff_previous_u2_6 * field_shape[0][q * U_NS + 6];
      const s_t coeff_previous_u2_7 = previous[27][lane];
      u2_old += coeff_previous_u2_7 * field_shape[0][q * U_NS + 7];
      const s_t coeff_previous_u2_8 = previous[28][lane];
      u2_old += coeff_previous_u2_8 * field_shape[0][q * U_NS + 8];
      const s_t coeff_previous_u2_9 = previous[29][lane];
      u2_old += coeff_previous_u2_9 * field_shape[0][q * U_NS + 9];
      s_t p = s_t(0);
      const s_t coeff_current_p_0 = current[30][lane];
      p += coeff_current_p_0 * field_shape[1][q * P_NS];
      const s_t coeff_current_p_1 = current[31][lane];
      p += coeff_current_p_1 * field_shape[1][q * P_NS + 1];
      const s_t coeff_current_p_2 = current[32][lane];
      p += coeff_current_p_2 * field_shape[1][q * P_NS + 2];
      const s_t coeff_current_p_3 = current[33][lane];
      p += coeff_current_p_3 * field_shape[1][q * P_NS + 3];
      const s_t residual_tmp0 = rho/dt;
      const s_t residual_tmp1 = -p;
      const s_t residual_tmp2 = nu*rho;
      const s_t residual_tmp3 = s_t(2)*residual_tmp2;
      const s_t residual_tmp4 = residual_tmp2*(u0_grad_1 + u1_grad_0);
      const s_t residual_tmp5 = residual_tmp2*(u0_grad_2 + u2_grad_0);
      const s_t residual_tmp6 = residual_tmp2*(u1_grad_2 + u2_grad_1);
      const s_t value_coeff0 = residual_tmp0*(dt*(convection_scale*(u0_grad_0*u0_old + u0_grad_1*u1_old + u0_grad_2*u2_old) - f0) + u0 - u0_old);
      const s_t grad_coeff0_0 = residual_tmp1 + residual_tmp3*u0_grad_0;
      const s_t grad_coeff0_1 = residual_tmp4;
      const s_t grad_coeff0_2 = residual_tmp5;
      const s_t value_coeff1 = residual_tmp0*(dt*(convection_scale*(u0_old*u1_grad_0 + u1_grad_1*u1_old + u1_grad_2*u2_old) - f1) + u1 - u1_old);
      const s_t grad_coeff1_0 = residual_tmp4;
      const s_t grad_coeff1_1 = residual_tmp1 + residual_tmp3*u1_grad_1;
      const s_t grad_coeff1_2 = residual_tmp6;
      const s_t value_coeff2 = residual_tmp0*(dt*(convection_scale*(u0_old*u2_grad_0 + u1_old*u2_grad_1 + u2_grad_2*u2_old) - f2) + u2 - u2_old);
      const s_t grad_coeff2_0 = residual_tmp5;
      const s_t grad_coeff2_1 = residual_tmp6;
      const s_t grad_coeff2_2 = residual_tmp1 + residual_tmp3*u2_grad_2;
      const s_t test_value_u0_0 = field_shape[0][q * U_NS];
      const s_t test_grad0_u0_0 = (fgref[0][q * U_NS] * adj0 + fgref[1][q * U_NS] * adj3 + fgref[2][q * U_NS] * adj6) / det;
      const s_t test_grad1_u0_0 = (fgref[0][q * U_NS] * adj1 + fgref[1][q * U_NS] * adj4 + fgref[2][q * U_NS] * adj7) / det;
      const s_t test_grad2_u0_0 = (fgref[0][q * U_NS] * adj2 + fgref[1][q * U_NS] * adj5 + fgref[2][q * U_NS] * adj8) / det;
      output[0][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_0 + grad_coeff0_0 * test_grad0_u0_0 + grad_coeff0_1 * test_grad1_u0_0 + grad_coeff0_2 * test_grad2_u0_0);
      const s_t test_value_u0_1 = field_shape[0][q * U_NS + 1];
      const s_t test_grad0_u0_1 = (fgref[0][q * U_NS + 1] * adj0 + fgref[1][q * U_NS + 1] * adj3 + fgref[2][q * U_NS + 1] * adj6) / det;
      const s_t test_grad1_u0_1 = (fgref[0][q * U_NS + 1] * adj1 + fgref[1][q * U_NS + 1] * adj4 + fgref[2][q * U_NS + 1] * adj7) / det;
      const s_t test_grad2_u0_1 = (fgref[0][q * U_NS + 1] * adj2 + fgref[1][q * U_NS + 1] * adj5 + fgref[2][q * U_NS + 1] * adj8) / det;
      output[1][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_1 + grad_coeff0_0 * test_grad0_u0_1 + grad_coeff0_1 * test_grad1_u0_1 + grad_coeff0_2 * test_grad2_u0_1);
      const s_t test_value_u0_2 = field_shape[0][q * U_NS + 2];
      const s_t test_grad0_u0_2 = (fgref[0][q * U_NS + 2] * adj0 + fgref[1][q * U_NS + 2] * adj3 + fgref[2][q * U_NS + 2] * adj6) / det;
      const s_t test_grad1_u0_2 = (fgref[0][q * U_NS + 2] * adj1 + fgref[1][q * U_NS + 2] * adj4 + fgref[2][q * U_NS + 2] * adj7) / det;
      const s_t test_grad2_u0_2 = (fgref[0][q * U_NS + 2] * adj2 + fgref[1][q * U_NS + 2] * adj5 + fgref[2][q * U_NS + 2] * adj8) / det;
      output[2][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_2 + grad_coeff0_0 * test_grad0_u0_2 + grad_coeff0_1 * test_grad1_u0_2 + grad_coeff0_2 * test_grad2_u0_2);
      const s_t test_value_u0_3 = field_shape[0][q * U_NS + 3];
      const s_t test_grad0_u0_3 = (fgref[0][q * U_NS + 3] * adj0 + fgref[1][q * U_NS + 3] * adj3 + fgref[2][q * U_NS + 3] * adj6) / det;
      const s_t test_grad1_u0_3 = (fgref[0][q * U_NS + 3] * adj1 + fgref[1][q * U_NS + 3] * adj4 + fgref[2][q * U_NS + 3] * adj7) / det;
      const s_t test_grad2_u0_3 = (fgref[0][q * U_NS + 3] * adj2 + fgref[1][q * U_NS + 3] * adj5 + fgref[2][q * U_NS + 3] * adj8) / det;
      output[3][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_3 + grad_coeff0_0 * test_grad0_u0_3 + grad_coeff0_1 * test_grad1_u0_3 + grad_coeff0_2 * test_grad2_u0_3);
      const s_t test_value_u0_4 = field_shape[0][q * U_NS + 4];
      const s_t test_grad0_u0_4 = (fgref[0][q * U_NS + 4] * adj0 + fgref[1][q * U_NS + 4] * adj3 + fgref[2][q * U_NS + 4] * adj6) / det;
      const s_t test_grad1_u0_4 = (fgref[0][q * U_NS + 4] * adj1 + fgref[1][q * U_NS + 4] * adj4 + fgref[2][q * U_NS + 4] * adj7) / det;
      const s_t test_grad2_u0_4 = (fgref[0][q * U_NS + 4] * adj2 + fgref[1][q * U_NS + 4] * adj5 + fgref[2][q * U_NS + 4] * adj8) / det;
      output[4][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_4 + grad_coeff0_0 * test_grad0_u0_4 + grad_coeff0_1 * test_grad1_u0_4 + grad_coeff0_2 * test_grad2_u0_4);
      const s_t test_value_u0_5 = field_shape[0][q * U_NS + 5];
      const s_t test_grad0_u0_5 = (fgref[0][q * U_NS + 5] * adj0 + fgref[1][q * U_NS + 5] * adj3 + fgref[2][q * U_NS + 5] * adj6) / det;
      const s_t test_grad1_u0_5 = (fgref[0][q * U_NS + 5] * adj1 + fgref[1][q * U_NS + 5] * adj4 + fgref[2][q * U_NS + 5] * adj7) / det;
      const s_t test_grad2_u0_5 = (fgref[0][q * U_NS + 5] * adj2 + fgref[1][q * U_NS + 5] * adj5 + fgref[2][q * U_NS + 5] * adj8) / det;
      output[5][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_5 + grad_coeff0_0 * test_grad0_u0_5 + grad_coeff0_1 * test_grad1_u0_5 + grad_coeff0_2 * test_grad2_u0_5);
      const s_t test_value_u0_6 = field_shape[0][q * U_NS + 6];
      const s_t test_grad0_u0_6 = (fgref[0][q * U_NS + 6] * adj0 + fgref[1][q * U_NS + 6] * adj3 + fgref[2][q * U_NS + 6] * adj6) / det;
      const s_t test_grad1_u0_6 = (fgref[0][q * U_NS + 6] * adj1 + fgref[1][q * U_NS + 6] * adj4 + fgref[2][q * U_NS + 6] * adj7) / det;
      const s_t test_grad2_u0_6 = (fgref[0][q * U_NS + 6] * adj2 + fgref[1][q * U_NS + 6] * adj5 + fgref[2][q * U_NS + 6] * adj8) / det;
      output[6][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_6 + grad_coeff0_0 * test_grad0_u0_6 + grad_coeff0_1 * test_grad1_u0_6 + grad_coeff0_2 * test_grad2_u0_6);
      const s_t test_value_u0_7 = field_shape[0][q * U_NS + 7];
      const s_t test_grad0_u0_7 = (fgref[0][q * U_NS + 7] * adj0 + fgref[1][q * U_NS + 7] * adj3 + fgref[2][q * U_NS + 7] * adj6) / det;
      const s_t test_grad1_u0_7 = (fgref[0][q * U_NS + 7] * adj1 + fgref[1][q * U_NS + 7] * adj4 + fgref[2][q * U_NS + 7] * adj7) / det;
      const s_t test_grad2_u0_7 = (fgref[0][q * U_NS + 7] * adj2 + fgref[1][q * U_NS + 7] * adj5 + fgref[2][q * U_NS + 7] * adj8) / det;
      output[7][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_7 + grad_coeff0_0 * test_grad0_u0_7 + grad_coeff0_1 * test_grad1_u0_7 + grad_coeff0_2 * test_grad2_u0_7);
      const s_t test_value_u0_8 = field_shape[0][q * U_NS + 8];
      const s_t test_grad0_u0_8 = (fgref[0][q * U_NS + 8] * adj0 + fgref[1][q * U_NS + 8] * adj3 + fgref[2][q * U_NS + 8] * adj6) / det;
      const s_t test_grad1_u0_8 = (fgref[0][q * U_NS + 8] * adj1 + fgref[1][q * U_NS + 8] * adj4 + fgref[2][q * U_NS + 8] * adj7) / det;
      const s_t test_grad2_u0_8 = (fgref[0][q * U_NS + 8] * adj2 + fgref[1][q * U_NS + 8] * adj5 + fgref[2][q * U_NS + 8] * adj8) / det;
      output[8][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_8 + grad_coeff0_0 * test_grad0_u0_8 + grad_coeff0_1 * test_grad1_u0_8 + grad_coeff0_2 * test_grad2_u0_8);
      const s_t test_value_u0_9 = field_shape[0][q * U_NS + 9];
      const s_t test_grad0_u0_9 = (fgref[0][q * U_NS + 9] * adj0 + fgref[1][q * U_NS + 9] * adj3 + fgref[2][q * U_NS + 9] * adj6) / det;
      const s_t test_grad1_u0_9 = (fgref[0][q * U_NS + 9] * adj1 + fgref[1][q * U_NS + 9] * adj4 + fgref[2][q * U_NS + 9] * adj7) / det;
      const s_t test_grad2_u0_9 = (fgref[0][q * U_NS + 9] * adj2 + fgref[1][q * U_NS + 9] * adj5 + fgref[2][q * U_NS + 9] * adj8) / det;
      output[9][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_9 + grad_coeff0_0 * test_grad0_u0_9 + grad_coeff0_1 * test_grad1_u0_9 + grad_coeff0_2 * test_grad2_u0_9);
      const s_t test_value_u1_0 = field_shape[0][q * U_NS];
      const s_t test_grad0_u1_0 = (fgref[0][q * U_NS] * adj0 + fgref[1][q * U_NS] * adj3 + fgref[2][q * U_NS] * adj6) / det;
      const s_t test_grad1_u1_0 = (fgref[0][q * U_NS] * adj1 + fgref[1][q * U_NS] * adj4 + fgref[2][q * U_NS] * adj7) / det;
      const s_t test_grad2_u1_0 = (fgref[0][q * U_NS] * adj2 + fgref[1][q * U_NS] * adj5 + fgref[2][q * U_NS] * adj8) / det;
      output[10][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_0 + grad_coeff1_0 * test_grad0_u1_0 + grad_coeff1_1 * test_grad1_u1_0 + grad_coeff1_2 * test_grad2_u1_0);
      const s_t test_value_u1_1 = field_shape[0][q * U_NS + 1];
      const s_t test_grad0_u1_1 = (fgref[0][q * U_NS + 1] * adj0 + fgref[1][q * U_NS + 1] * adj3 + fgref[2][q * U_NS + 1] * adj6) / det;
      const s_t test_grad1_u1_1 = (fgref[0][q * U_NS + 1] * adj1 + fgref[1][q * U_NS + 1] * adj4 + fgref[2][q * U_NS + 1] * adj7) / det;
      const s_t test_grad2_u1_1 = (fgref[0][q * U_NS + 1] * adj2 + fgref[1][q * U_NS + 1] * adj5 + fgref[2][q * U_NS + 1] * adj8) / det;
      output[11][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_1 + grad_coeff1_0 * test_grad0_u1_1 + grad_coeff1_1 * test_grad1_u1_1 + grad_coeff1_2 * test_grad2_u1_1);
      const s_t test_value_u1_2 = field_shape[0][q * U_NS + 2];
      const s_t test_grad0_u1_2 = (fgref[0][q * U_NS + 2] * adj0 + fgref[1][q * U_NS + 2] * adj3 + fgref[2][q * U_NS + 2] * adj6) / det;
      const s_t test_grad1_u1_2 = (fgref[0][q * U_NS + 2] * adj1 + fgref[1][q * U_NS + 2] * adj4 + fgref[2][q * U_NS + 2] * adj7) / det;
      const s_t test_grad2_u1_2 = (fgref[0][q * U_NS + 2] * adj2 + fgref[1][q * U_NS + 2] * adj5 + fgref[2][q * U_NS + 2] * adj8) / det;
      output[12][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_2 + grad_coeff1_0 * test_grad0_u1_2 + grad_coeff1_1 * test_grad1_u1_2 + grad_coeff1_2 * test_grad2_u1_2);
      const s_t test_value_u1_3 = field_shape[0][q * U_NS + 3];
      const s_t test_grad0_u1_3 = (fgref[0][q * U_NS + 3] * adj0 + fgref[1][q * U_NS + 3] * adj3 + fgref[2][q * U_NS + 3] * adj6) / det;
      const s_t test_grad1_u1_3 = (fgref[0][q * U_NS + 3] * adj1 + fgref[1][q * U_NS + 3] * adj4 + fgref[2][q * U_NS + 3] * adj7) / det;
      const s_t test_grad2_u1_3 = (fgref[0][q * U_NS + 3] * adj2 + fgref[1][q * U_NS + 3] * adj5 + fgref[2][q * U_NS + 3] * adj8) / det;
      output[13][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_3 + grad_coeff1_0 * test_grad0_u1_3 + grad_coeff1_1 * test_grad1_u1_3 + grad_coeff1_2 * test_grad2_u1_3);
      const s_t test_value_u1_4 = field_shape[0][q * U_NS + 4];
      const s_t test_grad0_u1_4 = (fgref[0][q * U_NS + 4] * adj0 + fgref[1][q * U_NS + 4] * adj3 + fgref[2][q * U_NS + 4] * adj6) / det;
      const s_t test_grad1_u1_4 = (fgref[0][q * U_NS + 4] * adj1 + fgref[1][q * U_NS + 4] * adj4 + fgref[2][q * U_NS + 4] * adj7) / det;
      const s_t test_grad2_u1_4 = (fgref[0][q * U_NS + 4] * adj2 + fgref[1][q * U_NS + 4] * adj5 + fgref[2][q * U_NS + 4] * adj8) / det;
      output[14][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_4 + grad_coeff1_0 * test_grad0_u1_4 + grad_coeff1_1 * test_grad1_u1_4 + grad_coeff1_2 * test_grad2_u1_4);
      const s_t test_value_u1_5 = field_shape[0][q * U_NS + 5];
      const s_t test_grad0_u1_5 = (fgref[0][q * U_NS + 5] * adj0 + fgref[1][q * U_NS + 5] * adj3 + fgref[2][q * U_NS + 5] * adj6) / det;
      const s_t test_grad1_u1_5 = (fgref[0][q * U_NS + 5] * adj1 + fgref[1][q * U_NS + 5] * adj4 + fgref[2][q * U_NS + 5] * adj7) / det;
      const s_t test_grad2_u1_5 = (fgref[0][q * U_NS + 5] * adj2 + fgref[1][q * U_NS + 5] * adj5 + fgref[2][q * U_NS + 5] * adj8) / det;
      output[15][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_5 + grad_coeff1_0 * test_grad0_u1_5 + grad_coeff1_1 * test_grad1_u1_5 + grad_coeff1_2 * test_grad2_u1_5);
      const s_t test_value_u1_6 = field_shape[0][q * U_NS + 6];
      const s_t test_grad0_u1_6 = (fgref[0][q * U_NS + 6] * adj0 + fgref[1][q * U_NS + 6] * adj3 + fgref[2][q * U_NS + 6] * adj6) / det;
      const s_t test_grad1_u1_6 = (fgref[0][q * U_NS + 6] * adj1 + fgref[1][q * U_NS + 6] * adj4 + fgref[2][q * U_NS + 6] * adj7) / det;
      const s_t test_grad2_u1_6 = (fgref[0][q * U_NS + 6] * adj2 + fgref[1][q * U_NS + 6] * adj5 + fgref[2][q * U_NS + 6] * adj8) / det;
      output[16][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_6 + grad_coeff1_0 * test_grad0_u1_6 + grad_coeff1_1 * test_grad1_u1_6 + grad_coeff1_2 * test_grad2_u1_6);
      const s_t test_value_u1_7 = field_shape[0][q * U_NS + 7];
      const s_t test_grad0_u1_7 = (fgref[0][q * U_NS + 7] * adj0 + fgref[1][q * U_NS + 7] * adj3 + fgref[2][q * U_NS + 7] * adj6) / det;
      const s_t test_grad1_u1_7 = (fgref[0][q * U_NS + 7] * adj1 + fgref[1][q * U_NS + 7] * adj4 + fgref[2][q * U_NS + 7] * adj7) / det;
      const s_t test_grad2_u1_7 = (fgref[0][q * U_NS + 7] * adj2 + fgref[1][q * U_NS + 7] * adj5 + fgref[2][q * U_NS + 7] * adj8) / det;
      output[17][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_7 + grad_coeff1_0 * test_grad0_u1_7 + grad_coeff1_1 * test_grad1_u1_7 + grad_coeff1_2 * test_grad2_u1_7);
      const s_t test_value_u1_8 = field_shape[0][q * U_NS + 8];
      const s_t test_grad0_u1_8 = (fgref[0][q * U_NS + 8] * adj0 + fgref[1][q * U_NS + 8] * adj3 + fgref[2][q * U_NS + 8] * adj6) / det;
      const s_t test_grad1_u1_8 = (fgref[0][q * U_NS + 8] * adj1 + fgref[1][q * U_NS + 8] * adj4 + fgref[2][q * U_NS + 8] * adj7) / det;
      const s_t test_grad2_u1_8 = (fgref[0][q * U_NS + 8] * adj2 + fgref[1][q * U_NS + 8] * adj5 + fgref[2][q * U_NS + 8] * adj8) / det;
      output[18][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_8 + grad_coeff1_0 * test_grad0_u1_8 + grad_coeff1_1 * test_grad1_u1_8 + grad_coeff1_2 * test_grad2_u1_8);
      const s_t test_value_u1_9 = field_shape[0][q * U_NS + 9];
      const s_t test_grad0_u1_9 = (fgref[0][q * U_NS + 9] * adj0 + fgref[1][q * U_NS + 9] * adj3 + fgref[2][q * U_NS + 9] * adj6) / det;
      const s_t test_grad1_u1_9 = (fgref[0][q * U_NS + 9] * adj1 + fgref[1][q * U_NS + 9] * adj4 + fgref[2][q * U_NS + 9] * adj7) / det;
      const s_t test_grad2_u1_9 = (fgref[0][q * U_NS + 9] * adj2 + fgref[1][q * U_NS + 9] * adj5 + fgref[2][q * U_NS + 9] * adj8) / det;
      output[19][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_9 + grad_coeff1_0 * test_grad0_u1_9 + grad_coeff1_1 * test_grad1_u1_9 + grad_coeff1_2 * test_grad2_u1_9);
      const s_t test_value_u2_0 = field_shape[0][q * U_NS];
      const s_t test_grad0_u2_0 = (fgref[0][q * U_NS] * adj0 + fgref[1][q * U_NS] * adj3 + fgref[2][q * U_NS] * adj6) / det;
      const s_t test_grad1_u2_0 = (fgref[0][q * U_NS] * adj1 + fgref[1][q * U_NS] * adj4 + fgref[2][q * U_NS] * adj7) / det;
      const s_t test_grad2_u2_0 = (fgref[0][q * U_NS] * adj2 + fgref[1][q * U_NS] * adj5 + fgref[2][q * U_NS] * adj8) / det;
      output[20][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_0 + grad_coeff2_0 * test_grad0_u2_0 + grad_coeff2_1 * test_grad1_u2_0 + grad_coeff2_2 * test_grad2_u2_0);
      const s_t test_value_u2_1 = field_shape[0][q * U_NS + 1];
      const s_t test_grad0_u2_1 = (fgref[0][q * U_NS + 1] * adj0 + fgref[1][q * U_NS + 1] * adj3 + fgref[2][q * U_NS + 1] * adj6) / det;
      const s_t test_grad1_u2_1 = (fgref[0][q * U_NS + 1] * adj1 + fgref[1][q * U_NS + 1] * adj4 + fgref[2][q * U_NS + 1] * adj7) / det;
      const s_t test_grad2_u2_1 = (fgref[0][q * U_NS + 1] * adj2 + fgref[1][q * U_NS + 1] * adj5 + fgref[2][q * U_NS + 1] * adj8) / det;
      output[21][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_1 + grad_coeff2_0 * test_grad0_u2_1 + grad_coeff2_1 * test_grad1_u2_1 + grad_coeff2_2 * test_grad2_u2_1);
      const s_t test_value_u2_2 = field_shape[0][q * U_NS + 2];
      const s_t test_grad0_u2_2 = (fgref[0][q * U_NS + 2] * adj0 + fgref[1][q * U_NS + 2] * adj3 + fgref[2][q * U_NS + 2] * adj6) / det;
      const s_t test_grad1_u2_2 = (fgref[0][q * U_NS + 2] * adj1 + fgref[1][q * U_NS + 2] * adj4 + fgref[2][q * U_NS + 2] * adj7) / det;
      const s_t test_grad2_u2_2 = (fgref[0][q * U_NS + 2] * adj2 + fgref[1][q * U_NS + 2] * adj5 + fgref[2][q * U_NS + 2] * adj8) / det;
      output[22][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_2 + grad_coeff2_0 * test_grad0_u2_2 + grad_coeff2_1 * test_grad1_u2_2 + grad_coeff2_2 * test_grad2_u2_2);
      const s_t test_value_u2_3 = field_shape[0][q * U_NS + 3];
      const s_t test_grad0_u2_3 = (fgref[0][q * U_NS + 3] * adj0 + fgref[1][q * U_NS + 3] * adj3 + fgref[2][q * U_NS + 3] * adj6) / det;
      const s_t test_grad1_u2_3 = (fgref[0][q * U_NS + 3] * adj1 + fgref[1][q * U_NS + 3] * adj4 + fgref[2][q * U_NS + 3] * adj7) / det;
      const s_t test_grad2_u2_3 = (fgref[0][q * U_NS + 3] * adj2 + fgref[1][q * U_NS + 3] * adj5 + fgref[2][q * U_NS + 3] * adj8) / det;
      output[23][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_3 + grad_coeff2_0 * test_grad0_u2_3 + grad_coeff2_1 * test_grad1_u2_3 + grad_coeff2_2 * test_grad2_u2_3);
      const s_t test_value_u2_4 = field_shape[0][q * U_NS + 4];
      const s_t test_grad0_u2_4 = (fgref[0][q * U_NS + 4] * adj0 + fgref[1][q * U_NS + 4] * adj3 + fgref[2][q * U_NS + 4] * adj6) / det;
      const s_t test_grad1_u2_4 = (fgref[0][q * U_NS + 4] * adj1 + fgref[1][q * U_NS + 4] * adj4 + fgref[2][q * U_NS + 4] * adj7) / det;
      const s_t test_grad2_u2_4 = (fgref[0][q * U_NS + 4] * adj2 + fgref[1][q * U_NS + 4] * adj5 + fgref[2][q * U_NS + 4] * adj8) / det;
      output[24][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_4 + grad_coeff2_0 * test_grad0_u2_4 + grad_coeff2_1 * test_grad1_u2_4 + grad_coeff2_2 * test_grad2_u2_4);
      const s_t test_value_u2_5 = field_shape[0][q * U_NS + 5];
      const s_t test_grad0_u2_5 = (fgref[0][q * U_NS + 5] * adj0 + fgref[1][q * U_NS + 5] * adj3 + fgref[2][q * U_NS + 5] * adj6) / det;
      const s_t test_grad1_u2_5 = (fgref[0][q * U_NS + 5] * adj1 + fgref[1][q * U_NS + 5] * adj4 + fgref[2][q * U_NS + 5] * adj7) / det;
      const s_t test_grad2_u2_5 = (fgref[0][q * U_NS + 5] * adj2 + fgref[1][q * U_NS + 5] * adj5 + fgref[2][q * U_NS + 5] * adj8) / det;
      output[25][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_5 + grad_coeff2_0 * test_grad0_u2_5 + grad_coeff2_1 * test_grad1_u2_5 + grad_coeff2_2 * test_grad2_u2_5);
      const s_t test_value_u2_6 = field_shape[0][q * U_NS + 6];
      const s_t test_grad0_u2_6 = (fgref[0][q * U_NS + 6] * adj0 + fgref[1][q * U_NS + 6] * adj3 + fgref[2][q * U_NS + 6] * adj6) / det;
      const s_t test_grad1_u2_6 = (fgref[0][q * U_NS + 6] * adj1 + fgref[1][q * U_NS + 6] * adj4 + fgref[2][q * U_NS + 6] * adj7) / det;
      const s_t test_grad2_u2_6 = (fgref[0][q * U_NS + 6] * adj2 + fgref[1][q * U_NS + 6] * adj5 + fgref[2][q * U_NS + 6] * adj8) / det;
      output[26][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_6 + grad_coeff2_0 * test_grad0_u2_6 + grad_coeff2_1 * test_grad1_u2_6 + grad_coeff2_2 * test_grad2_u2_6);
      const s_t test_value_u2_7 = field_shape[0][q * U_NS + 7];
      const s_t test_grad0_u2_7 = (fgref[0][q * U_NS + 7] * adj0 + fgref[1][q * U_NS + 7] * adj3 + fgref[2][q * U_NS + 7] * adj6) / det;
      const s_t test_grad1_u2_7 = (fgref[0][q * U_NS + 7] * adj1 + fgref[1][q * U_NS + 7] * adj4 + fgref[2][q * U_NS + 7] * adj7) / det;
      const s_t test_grad2_u2_7 = (fgref[0][q * U_NS + 7] * adj2 + fgref[1][q * U_NS + 7] * adj5 + fgref[2][q * U_NS + 7] * adj8) / det;
      output[27][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_7 + grad_coeff2_0 * test_grad0_u2_7 + grad_coeff2_1 * test_grad1_u2_7 + grad_coeff2_2 * test_grad2_u2_7);
      const s_t test_value_u2_8 = field_shape[0][q * U_NS + 8];
      const s_t test_grad0_u2_8 = (fgref[0][q * U_NS + 8] * adj0 + fgref[1][q * U_NS + 8] * adj3 + fgref[2][q * U_NS + 8] * adj6) / det;
      const s_t test_grad1_u2_8 = (fgref[0][q * U_NS + 8] * adj1 + fgref[1][q * U_NS + 8] * adj4 + fgref[2][q * U_NS + 8] * adj7) / det;
      const s_t test_grad2_u2_8 = (fgref[0][q * U_NS + 8] * adj2 + fgref[1][q * U_NS + 8] * adj5 + fgref[2][q * U_NS + 8] * adj8) / det;
      output[28][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_8 + grad_coeff2_0 * test_grad0_u2_8 + grad_coeff2_1 * test_grad1_u2_8 + grad_coeff2_2 * test_grad2_u2_8);
      const s_t test_value_u2_9 = field_shape[0][q * U_NS + 9];
      const s_t test_grad0_u2_9 = (fgref[0][q * U_NS + 9] * adj0 + fgref[1][q * U_NS + 9] * adj3 + fgref[2][q * U_NS + 9] * adj6) / det;
      const s_t test_grad1_u2_9 = (fgref[0][q * U_NS + 9] * adj1 + fgref[1][q * U_NS + 9] * adj4 + fgref[2][q * U_NS + 9] * adj7) / det;
      const s_t test_grad2_u2_9 = (fgref[0][q * U_NS + 9] * adj2 + fgref[1][q * U_NS + 9] * adj5 + fgref[2][q * U_NS + 9] * adj8) / det;
      output[29][lane] += q_weight[q] * det * (value_coeff2 * test_value_u2_9 + grad_coeff2_0 * test_grad0_u2_9 + grad_coeff2_1 * test_grad1_u2_9 + grad_coeff2_2 * test_grad2_u2_9);
    }
  }
}

template <typename s_t, int NQ, int CELL_NS, int VS>
static SFEM_INLINE void navier_stokes_form_1_u_d3_simplex_mixed_jacobian_action_block(
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
static SFEM_INLINE void navier_stokes_form_1_u_d3_simplex_mixed_jacobian_action_block_contiguous(
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

} // namespace codegen
} // namespace sfem

#endif
