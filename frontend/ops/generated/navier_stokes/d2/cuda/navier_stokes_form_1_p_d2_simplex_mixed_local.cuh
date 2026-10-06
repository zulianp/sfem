#ifndef NAVIER_STOKES_FORM_1_P_D2_SIMPLEX_MIXED_LOCAL_HPP
#define NAVIER_STOKES_FORM_1_P_D2_SIMPLEX_MIXED_LOCAL_HPP

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

template <typename s_t, int NQ, int CELL_NS, int VS>
__host__ __device__ __forceinline__ void navier_stokes_form_1_p_d2_simplex_mixed_residual_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR field_shape[2],
    const s_t *const RSTR fgref[4],
    const s_t *const RSTR q_weight,
    const s_t *const RSTR current[15],
    s_t *const RSTR output[15]
) {
  static constexpr int U_NS = 6;
  static constexpr int P_NS = 3;
  for (int q = 0; q < NQ; ++q) {
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      s_t u0_grad_0_ref = s_t(0);
      s_t u0_grad_1_ref = s_t(0);
      const s_t coeff_current_u0_0 = current[0][0];
      u0_grad_0_ref += coeff_current_u0_0 * fgref[0][q * U_NS];
      u0_grad_1_ref += coeff_current_u0_0 * fgref[1][q * U_NS];
      const s_t coeff_current_u0_1 = current[1][0];
      u0_grad_0_ref += coeff_current_u0_1 * fgref[0][q * U_NS + 1];
      u0_grad_1_ref += coeff_current_u0_1 * fgref[1][q * U_NS + 1];
      const s_t coeff_current_u0_2 = current[2][0];
      u0_grad_0_ref += coeff_current_u0_2 * fgref[0][q * U_NS + 2];
      u0_grad_1_ref += coeff_current_u0_2 * fgref[1][q * U_NS + 2];
      const s_t coeff_current_u0_3 = current[3][0];
      u0_grad_0_ref += coeff_current_u0_3 * fgref[0][q * U_NS + 3];
      u0_grad_1_ref += coeff_current_u0_3 * fgref[1][q * U_NS + 3];
      const s_t coeff_current_u0_4 = current[4][0];
      u0_grad_0_ref += coeff_current_u0_4 * fgref[0][q * U_NS + 4];
      u0_grad_1_ref += coeff_current_u0_4 * fgref[1][q * U_NS + 4];
      const s_t coeff_current_u0_5 = current[5][0];
      u0_grad_0_ref += coeff_current_u0_5 * fgref[0][q * U_NS + 5];
      u0_grad_1_ref += coeff_current_u0_5 * fgref[1][q * U_NS + 5];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      s_t u1_grad_0_ref = s_t(0);
      s_t u1_grad_1_ref = s_t(0);
      const s_t coeff_current_u1_0 = current[6][0];
      u1_grad_0_ref += coeff_current_u1_0 * fgref[0][q * U_NS];
      u1_grad_1_ref += coeff_current_u1_0 * fgref[1][q * U_NS];
      const s_t coeff_current_u1_1 = current[7][0];
      u1_grad_0_ref += coeff_current_u1_1 * fgref[0][q * U_NS + 1];
      u1_grad_1_ref += coeff_current_u1_1 * fgref[1][q * U_NS + 1];
      const s_t coeff_current_u1_2 = current[8][0];
      u1_grad_0_ref += coeff_current_u1_2 * fgref[0][q * U_NS + 2];
      u1_grad_1_ref += coeff_current_u1_2 * fgref[1][q * U_NS + 2];
      const s_t coeff_current_u1_3 = current[9][0];
      u1_grad_0_ref += coeff_current_u1_3 * fgref[0][q * U_NS + 3];
      u1_grad_1_ref += coeff_current_u1_3 * fgref[1][q * U_NS + 3];
      const s_t coeff_current_u1_4 = current[10][0];
      u1_grad_0_ref += coeff_current_u1_4 * fgref[0][q * U_NS + 4];
      u1_grad_1_ref += coeff_current_u1_4 * fgref[1][q * U_NS + 4];
      const s_t coeff_current_u1_5 = current[11][0];
      u1_grad_0_ref += coeff_current_u1_5 * fgref[0][q * U_NS + 5];
      u1_grad_1_ref += coeff_current_u1_5 * fgref[1][q * U_NS + 5];
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t value_coeff2 = u0_grad_0 + u1_grad_1;
      const s_t test_value_p_0 = field_shape[1][q * P_NS];
      output[12][0] += q_weight[q] * det * (value_coeff2 * test_value_p_0);
      const s_t test_value_p_1 = field_shape[1][q * P_NS + 1];
      output[13][0] += q_weight[q] * det * (value_coeff2 * test_value_p_1);
      const s_t test_value_p_2 = field_shape[1][q * P_NS + 2];
      output[14][0] += q_weight[q] * det * (value_coeff2 * test_value_p_2);
    }
  }
}

template <typename s_t, int NQ, int CELL_NS, int VS>
__host__ __device__ __forceinline__ void navier_stokes_form_1_p_d2_simplex_mixed_residual_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[4],
    const s_t *const RSTR field_shape[2],
    const s_t *const RSTR fgref[4],
    const s_t *const RSTR q_weight,
    const s_t current[15][VS],
    s_t output[15][VS]
) {
  static constexpr int U_NS = 6;
  static constexpr int P_NS = 3;
  for (int q = 0; q < NQ; ++q) {
    {
      const ptrdiff_t goff = q * geometry_stride;
      const s_t det = determinant[goff];
      const s_t adj0 = adjugate[0][goff];
      const s_t adj1 = adjugate[1][goff];
      const s_t adj2 = adjugate[2][goff];
      const s_t adj3 = adjugate[3][goff];
      s_t u0_grad_0_ref = s_t(0);
      s_t u0_grad_1_ref = s_t(0);
      const s_t coeff_current_u0_0 = current[0][0];
      u0_grad_0_ref += coeff_current_u0_0 * fgref[0][q * U_NS];
      u0_grad_1_ref += coeff_current_u0_0 * fgref[1][q * U_NS];
      const s_t coeff_current_u0_1 = current[1][0];
      u0_grad_0_ref += coeff_current_u0_1 * fgref[0][q * U_NS + 1];
      u0_grad_1_ref += coeff_current_u0_1 * fgref[1][q * U_NS + 1];
      const s_t coeff_current_u0_2 = current[2][0];
      u0_grad_0_ref += coeff_current_u0_2 * fgref[0][q * U_NS + 2];
      u0_grad_1_ref += coeff_current_u0_2 * fgref[1][q * U_NS + 2];
      const s_t coeff_current_u0_3 = current[3][0];
      u0_grad_0_ref += coeff_current_u0_3 * fgref[0][q * U_NS + 3];
      u0_grad_1_ref += coeff_current_u0_3 * fgref[1][q * U_NS + 3];
      const s_t coeff_current_u0_4 = current[4][0];
      u0_grad_0_ref += coeff_current_u0_4 * fgref[0][q * U_NS + 4];
      u0_grad_1_ref += coeff_current_u0_4 * fgref[1][q * U_NS + 4];
      const s_t coeff_current_u0_5 = current[5][0];
      u0_grad_0_ref += coeff_current_u0_5 * fgref[0][q * U_NS + 5];
      u0_grad_1_ref += coeff_current_u0_5 * fgref[1][q * U_NS + 5];
      const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
      s_t u1_grad_0_ref = s_t(0);
      s_t u1_grad_1_ref = s_t(0);
      const s_t coeff_current_u1_0 = current[6][0];
      u1_grad_0_ref += coeff_current_u1_0 * fgref[0][q * U_NS];
      u1_grad_1_ref += coeff_current_u1_0 * fgref[1][q * U_NS];
      const s_t coeff_current_u1_1 = current[7][0];
      u1_grad_0_ref += coeff_current_u1_1 * fgref[0][q * U_NS + 1];
      u1_grad_1_ref += coeff_current_u1_1 * fgref[1][q * U_NS + 1];
      const s_t coeff_current_u1_2 = current[8][0];
      u1_grad_0_ref += coeff_current_u1_2 * fgref[0][q * U_NS + 2];
      u1_grad_1_ref += coeff_current_u1_2 * fgref[1][q * U_NS + 2];
      const s_t coeff_current_u1_3 = current[9][0];
      u1_grad_0_ref += coeff_current_u1_3 * fgref[0][q * U_NS + 3];
      u1_grad_1_ref += coeff_current_u1_3 * fgref[1][q * U_NS + 3];
      const s_t coeff_current_u1_4 = current[10][0];
      u1_grad_0_ref += coeff_current_u1_4 * fgref[0][q * U_NS + 4];
      u1_grad_1_ref += coeff_current_u1_4 * fgref[1][q * U_NS + 4];
      const s_t coeff_current_u1_5 = current[11][0];
      u1_grad_0_ref += coeff_current_u1_5 * fgref[0][q * U_NS + 5];
      u1_grad_1_ref += coeff_current_u1_5 * fgref[1][q * U_NS + 5];
      const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
      const s_t value_coeff2 = u0_grad_0 + u1_grad_1;
      const s_t test_value_p_0 = field_shape[1][q * P_NS];
      output[12][0] += q_weight[q] * det * (value_coeff2 * test_value_p_0);
      const s_t test_value_p_1 = field_shape[1][q * P_NS + 1];
      output[13][0] += q_weight[q] * det * (value_coeff2 * test_value_p_1);
      const s_t test_value_p_2 = field_shape[1][q * P_NS + 2];
      output[14][0] += q_weight[q] * det * (value_coeff2 * test_value_p_2);
    }
  }
}


} // namespace codegen
} // namespace sfem

#endif
