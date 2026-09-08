#ifndef NAVIER_STOKES_D2_SIMPLEX_MIXED_LOCAL_HPP
#define NAVIER_STOKES_D2_SIMPLEX_MIXED_LOCAL_HPP

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
static SFEM_INLINE void navier_stokes_d2_simplex_mixed_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR adjugate[4],
        const s_t *const RSTR field_shape[2],
        const s_t *const RSTR fgref[4],
        const s_t *const RSTR q_weight,
        const s_t *const RSTR current[15],
        const s_t *const RSTR previous[15],
        const s_t convection_scale,
        const s_t dt,
        const s_t f0,
        const s_t f1,
        const s_t nu,
        const s_t rho,
        s_t *const RSTR output[15]
) {
    static constexpr int ND = 2;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 15;
    (void)CELL_NS;
    (void)N_FIELD_STREAMS;
    static constexpr int U_NS = 6;
    static constexpr int P_NS = 3;
    for (int q = 0; q < NQ; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t det = determinant[goff];
            const s_t adj0 = adjugate[0][goff];
            const s_t adj1 = adjugate[1][goff];
            const s_t adj2 = adjugate[2][goff];
            const s_t adj3 = adjugate[3][goff];
            s_t u0 = s_t(0);
            s_t u0_grad_0_ref = s_t(0);
            s_t u0_grad_1_ref = s_t(0);
            const s_t coeff_current_u0_0 = current[0][lane];
            u0 += coeff_current_u0_0 * field_shape[0][q * U_NS + 0];
            u0_grad_0_ref += coeff_current_u0_0 * fgref[0 * ND + 0][q * U_NS + 0];
            u0_grad_1_ref += coeff_current_u0_0 * fgref[0 * ND + 1][q * U_NS + 0];
            const s_t coeff_current_u0_1 = current[1][lane];
            u0 += coeff_current_u0_1 * field_shape[0][q * U_NS + 1];
            u0_grad_0_ref += coeff_current_u0_1 * fgref[0 * ND + 0][q * U_NS + 1];
            u0_grad_1_ref += coeff_current_u0_1 * fgref[0 * ND + 1][q * U_NS + 1];
            const s_t coeff_current_u0_2 = current[2][lane];
            u0 += coeff_current_u0_2 * field_shape[0][q * U_NS + 2];
            u0_grad_0_ref += coeff_current_u0_2 * fgref[0 * ND + 0][q * U_NS + 2];
            u0_grad_1_ref += coeff_current_u0_2 * fgref[0 * ND + 1][q * U_NS + 2];
            const s_t coeff_current_u0_3 = current[3][lane];
            u0 += coeff_current_u0_3 * field_shape[0][q * U_NS + 3];
            u0_grad_0_ref += coeff_current_u0_3 * fgref[0 * ND + 0][q * U_NS + 3];
            u0_grad_1_ref += coeff_current_u0_3 * fgref[0 * ND + 1][q * U_NS + 3];
            const s_t coeff_current_u0_4 = current[4][lane];
            u0 += coeff_current_u0_4 * field_shape[0][q * U_NS + 4];
            u0_grad_0_ref += coeff_current_u0_4 * fgref[0 * ND + 0][q * U_NS + 4];
            u0_grad_1_ref += coeff_current_u0_4 * fgref[0 * ND + 1][q * U_NS + 4];
            const s_t coeff_current_u0_5 = current[5][lane];
            u0 += coeff_current_u0_5 * field_shape[0][q * U_NS + 5];
            u0_grad_0_ref += coeff_current_u0_5 * fgref[0 * ND + 0][q * U_NS + 5];
            u0_grad_1_ref += coeff_current_u0_5 * fgref[0 * ND + 1][q * U_NS + 5];
            const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
            const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
            s_t u0_old = s_t(0);
            const s_t coeff_previous_u0_0 = previous[0][lane];
            u0_old += coeff_previous_u0_0 * field_shape[0][q * U_NS + 0];
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
            s_t u1 = s_t(0);
            s_t u1_grad_0_ref = s_t(0);
            s_t u1_grad_1_ref = s_t(0);
            const s_t coeff_current_u1_0 = current[6][lane];
            u1 += coeff_current_u1_0 * field_shape[0][q * U_NS + 0];
            u1_grad_0_ref += coeff_current_u1_0 * fgref[0 * ND + 0][q * U_NS + 0];
            u1_grad_1_ref += coeff_current_u1_0 * fgref[0 * ND + 1][q * U_NS + 0];
            const s_t coeff_current_u1_1 = current[7][lane];
            u1 += coeff_current_u1_1 * field_shape[0][q * U_NS + 1];
            u1_grad_0_ref += coeff_current_u1_1 * fgref[0 * ND + 0][q * U_NS + 1];
            u1_grad_1_ref += coeff_current_u1_1 * fgref[0 * ND + 1][q * U_NS + 1];
            const s_t coeff_current_u1_2 = current[8][lane];
            u1 += coeff_current_u1_2 * field_shape[0][q * U_NS + 2];
            u1_grad_0_ref += coeff_current_u1_2 * fgref[0 * ND + 0][q * U_NS + 2];
            u1_grad_1_ref += coeff_current_u1_2 * fgref[0 * ND + 1][q * U_NS + 2];
            const s_t coeff_current_u1_3 = current[9][lane];
            u1 += coeff_current_u1_3 * field_shape[0][q * U_NS + 3];
            u1_grad_0_ref += coeff_current_u1_3 * fgref[0 * ND + 0][q * U_NS + 3];
            u1_grad_1_ref += coeff_current_u1_3 * fgref[0 * ND + 1][q * U_NS + 3];
            const s_t coeff_current_u1_4 = current[10][lane];
            u1 += coeff_current_u1_4 * field_shape[0][q * U_NS + 4];
            u1_grad_0_ref += coeff_current_u1_4 * fgref[0 * ND + 0][q * U_NS + 4];
            u1_grad_1_ref += coeff_current_u1_4 * fgref[0 * ND + 1][q * U_NS + 4];
            const s_t coeff_current_u1_5 = current[11][lane];
            u1 += coeff_current_u1_5 * field_shape[0][q * U_NS + 5];
            u1_grad_0_ref += coeff_current_u1_5 * fgref[0 * ND + 0][q * U_NS + 5];
            u1_grad_1_ref += coeff_current_u1_5 * fgref[0 * ND + 1][q * U_NS + 5];
            const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
            const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
            s_t u1_old = s_t(0);
            const s_t coeff_previous_u1_0 = previous[6][lane];
            u1_old += coeff_previous_u1_0 * field_shape[0][q * U_NS + 0];
            const s_t coeff_previous_u1_1 = previous[7][lane];
            u1_old += coeff_previous_u1_1 * field_shape[0][q * U_NS + 1];
            const s_t coeff_previous_u1_2 = previous[8][lane];
            u1_old += coeff_previous_u1_2 * field_shape[0][q * U_NS + 2];
            const s_t coeff_previous_u1_3 = previous[9][lane];
            u1_old += coeff_previous_u1_3 * field_shape[0][q * U_NS + 3];
            const s_t coeff_previous_u1_4 = previous[10][lane];
            u1_old += coeff_previous_u1_4 * field_shape[0][q * U_NS + 4];
            const s_t coeff_previous_u1_5 = previous[11][lane];
            u1_old += coeff_previous_u1_5 * field_shape[0][q * U_NS + 5];
            s_t p = s_t(0);
            s_t p_grad_0_ref = s_t(0);
            s_t p_grad_1_ref = s_t(0);
            const s_t coeff_current_p_0 = current[12][lane];
            p += coeff_current_p_0 * field_shape[1][q * P_NS + 0];
            p_grad_0_ref += coeff_current_p_0 * fgref[1 * ND + 0][q * P_NS + 0];
            p_grad_1_ref += coeff_current_p_0 * fgref[1 * ND + 1][q * P_NS + 0];
            const s_t coeff_current_p_1 = current[13][lane];
            p += coeff_current_p_1 * field_shape[1][q * P_NS + 1];
            p_grad_0_ref += coeff_current_p_1 * fgref[1 * ND + 0][q * P_NS + 1];
            p_grad_1_ref += coeff_current_p_1 * fgref[1 * ND + 1][q * P_NS + 1];
            const s_t coeff_current_p_2 = current[14][lane];
            p += coeff_current_p_2 * field_shape[1][q * P_NS + 2];
            p_grad_0_ref += coeff_current_p_2 * fgref[1 * ND + 0][q * P_NS + 2];
            p_grad_1_ref += coeff_current_p_2 * fgref[1 * ND + 1][q * P_NS + 2];
            const s_t p_grad_0 = (p_grad_0_ref * adj0 + p_grad_1_ref * adj2) / det;
            const s_t p_grad_1 = (p_grad_0_ref * adj1 + p_grad_1_ref * adj3) / det;
            s_t p_old = s_t(0);
            const s_t coeff_previous_p_0 = previous[12][lane];
            p_old += coeff_previous_p_0 * field_shape[1][q * P_NS + 0];
            const s_t coeff_previous_p_1 = previous[13][lane];
            p_old += coeff_previous_p_1 * field_shape[1][q * P_NS + 1];
            const s_t coeff_previous_p_2 = previous[14][lane];
            p_old += coeff_previous_p_2 * field_shape[1][q * P_NS + 2];
            const s_t residual_tmp0 = rho/dt;
            const s_t residual_tmp1 = -p;
            const s_t residual_tmp2 = nu*rho;
            const s_t residual_tmp3 = s_t(2)*residual_tmp2;
            const s_t residual_tmp4 = residual_tmp2*(u0_grad_1 + u1_grad_0);
            const s_t value_coeff0 = residual_tmp0*(dt*(convection_scale*(u0_grad_0*u0_old + u0_grad_1*u1_old) - f0) + u0 - u0_old);
            const s_t grad_coeff0_0 = residual_tmp1 + residual_tmp3*u0_grad_0;
            const s_t grad_coeff0_1 = residual_tmp4;
            const s_t value_coeff1 = residual_tmp0*(dt*(convection_scale*(u0_old*u1_grad_0 + u1_grad_1*u1_old) - f1) + u1 - u1_old);
            const s_t grad_coeff1_0 = residual_tmp4;
            const s_t grad_coeff1_1 = residual_tmp1 + residual_tmp3*u1_grad_1;
            const s_t value_coeff2 = u0_grad_0 + u1_grad_1;
            const s_t test_value_u0_0 = field_shape[0][q * U_NS + 0];
            const s_t test_grad0_u0_0 = (fgref[0 * ND + 0][q * U_NS + 0] * adj0 + fgref[0 * ND + 1][q * U_NS + 0] * adj2) / det;
            const s_t test_grad1_u0_0 = (fgref[0 * ND + 0][q * U_NS + 0] * adj1 + fgref[0 * ND + 1][q * U_NS + 0] * adj3) / det;
            output[0][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_0 + grad_coeff0_0 * test_grad0_u0_0 + grad_coeff0_1 * test_grad1_u0_0);
            const s_t test_value_u0_1 = field_shape[0][q * U_NS + 1];
            const s_t test_grad0_u0_1 = (fgref[0 * ND + 0][q * U_NS + 1] * adj0 + fgref[0 * ND + 1][q * U_NS + 1] * adj2) / det;
            const s_t test_grad1_u0_1 = (fgref[0 * ND + 0][q * U_NS + 1] * adj1 + fgref[0 * ND + 1][q * U_NS + 1] * adj3) / det;
            output[1][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_1 + grad_coeff0_0 * test_grad0_u0_1 + grad_coeff0_1 * test_grad1_u0_1);
            const s_t test_value_u0_2 = field_shape[0][q * U_NS + 2];
            const s_t test_grad0_u0_2 = (fgref[0 * ND + 0][q * U_NS + 2] * adj0 + fgref[0 * ND + 1][q * U_NS + 2] * adj2) / det;
            const s_t test_grad1_u0_2 = (fgref[0 * ND + 0][q * U_NS + 2] * adj1 + fgref[0 * ND + 1][q * U_NS + 2] * adj3) / det;
            output[2][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_2 + grad_coeff0_0 * test_grad0_u0_2 + grad_coeff0_1 * test_grad1_u0_2);
            const s_t test_value_u0_3 = field_shape[0][q * U_NS + 3];
            const s_t test_grad0_u0_3 = (fgref[0 * ND + 0][q * U_NS + 3] * adj0 + fgref[0 * ND + 1][q * U_NS + 3] * adj2) / det;
            const s_t test_grad1_u0_3 = (fgref[0 * ND + 0][q * U_NS + 3] * adj1 + fgref[0 * ND + 1][q * U_NS + 3] * adj3) / det;
            output[3][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_3 + grad_coeff0_0 * test_grad0_u0_3 + grad_coeff0_1 * test_grad1_u0_3);
            const s_t test_value_u0_4 = field_shape[0][q * U_NS + 4];
            const s_t test_grad0_u0_4 = (fgref[0 * ND + 0][q * U_NS + 4] * adj0 + fgref[0 * ND + 1][q * U_NS + 4] * adj2) / det;
            const s_t test_grad1_u0_4 = (fgref[0 * ND + 0][q * U_NS + 4] * adj1 + fgref[0 * ND + 1][q * U_NS + 4] * adj3) / det;
            output[4][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_4 + grad_coeff0_0 * test_grad0_u0_4 + grad_coeff0_1 * test_grad1_u0_4);
            const s_t test_value_u0_5 = field_shape[0][q * U_NS + 5];
            const s_t test_grad0_u0_5 = (fgref[0 * ND + 0][q * U_NS + 5] * adj0 + fgref[0 * ND + 1][q * U_NS + 5] * adj2) / det;
            const s_t test_grad1_u0_5 = (fgref[0 * ND + 0][q * U_NS + 5] * adj1 + fgref[0 * ND + 1][q * U_NS + 5] * adj3) / det;
            output[5][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_5 + grad_coeff0_0 * test_grad0_u0_5 + grad_coeff0_1 * test_grad1_u0_5);
            const s_t test_value_u1_0 = field_shape[0][q * U_NS + 0];
            const s_t test_grad0_u1_0 = (fgref[0 * ND + 0][q * U_NS + 0] * adj0 + fgref[0 * ND + 1][q * U_NS + 0] * adj2) / det;
            const s_t test_grad1_u1_0 = (fgref[0 * ND + 0][q * U_NS + 0] * adj1 + fgref[0 * ND + 1][q * U_NS + 0] * adj3) / det;
            output[6][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_0 + grad_coeff1_0 * test_grad0_u1_0 + grad_coeff1_1 * test_grad1_u1_0);
            const s_t test_value_u1_1 = field_shape[0][q * U_NS + 1];
            const s_t test_grad0_u1_1 = (fgref[0 * ND + 0][q * U_NS + 1] * adj0 + fgref[0 * ND + 1][q * U_NS + 1] * adj2) / det;
            const s_t test_grad1_u1_1 = (fgref[0 * ND + 0][q * U_NS + 1] * adj1 + fgref[0 * ND + 1][q * U_NS + 1] * adj3) / det;
            output[7][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_1 + grad_coeff1_0 * test_grad0_u1_1 + grad_coeff1_1 * test_grad1_u1_1);
            const s_t test_value_u1_2 = field_shape[0][q * U_NS + 2];
            const s_t test_grad0_u1_2 = (fgref[0 * ND + 0][q * U_NS + 2] * adj0 + fgref[0 * ND + 1][q * U_NS + 2] * adj2) / det;
            const s_t test_grad1_u1_2 = (fgref[0 * ND + 0][q * U_NS + 2] * adj1 + fgref[0 * ND + 1][q * U_NS + 2] * adj3) / det;
            output[8][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_2 + grad_coeff1_0 * test_grad0_u1_2 + grad_coeff1_1 * test_grad1_u1_2);
            const s_t test_value_u1_3 = field_shape[0][q * U_NS + 3];
            const s_t test_grad0_u1_3 = (fgref[0 * ND + 0][q * U_NS + 3] * adj0 + fgref[0 * ND + 1][q * U_NS + 3] * adj2) / det;
            const s_t test_grad1_u1_3 = (fgref[0 * ND + 0][q * U_NS + 3] * adj1 + fgref[0 * ND + 1][q * U_NS + 3] * adj3) / det;
            output[9][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_3 + grad_coeff1_0 * test_grad0_u1_3 + grad_coeff1_1 * test_grad1_u1_3);
            const s_t test_value_u1_4 = field_shape[0][q * U_NS + 4];
            const s_t test_grad0_u1_4 = (fgref[0 * ND + 0][q * U_NS + 4] * adj0 + fgref[0 * ND + 1][q * U_NS + 4] * adj2) / det;
            const s_t test_grad1_u1_4 = (fgref[0 * ND + 0][q * U_NS + 4] * adj1 + fgref[0 * ND + 1][q * U_NS + 4] * adj3) / det;
            output[10][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_4 + grad_coeff1_0 * test_grad0_u1_4 + grad_coeff1_1 * test_grad1_u1_4);
            const s_t test_value_u1_5 = field_shape[0][q * U_NS + 5];
            const s_t test_grad0_u1_5 = (fgref[0 * ND + 0][q * U_NS + 5] * adj0 + fgref[0 * ND + 1][q * U_NS + 5] * adj2) / det;
            const s_t test_grad1_u1_5 = (fgref[0 * ND + 0][q * U_NS + 5] * adj1 + fgref[0 * ND + 1][q * U_NS + 5] * adj3) / det;
            output[11][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_5 + grad_coeff1_0 * test_grad0_u1_5 + grad_coeff1_1 * test_grad1_u1_5);
            const s_t test_value_p_0 = field_shape[1][q * P_NS + 0];
            output[12][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_0);
            const s_t test_value_p_1 = field_shape[1][q * P_NS + 1];
            output[13][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_1);
            const s_t test_value_p_2 = field_shape[1][q * P_NS + 2];
            output[14][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_2);
        }
    }
}

template <typename s_t, int NQ, int CELL_NS, int VS>
static SFEM_INLINE void navier_stokes_d2_simplex_mixed_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR adjugate[4],
        const s_t *const RSTR field_shape[2],
        const s_t *const RSTR fgref[4],
        const s_t *const RSTR q_weight,
        const s_t current[15][VS],
        const s_t previous[15][VS],
        const s_t convection_scale,
        const s_t dt,
        const s_t f0,
        const s_t f1,
        const s_t nu,
        const s_t rho,
        s_t output[15][VS]
) {
    static constexpr int ND = 2;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 15;
    (void)CELL_NS;
    (void)N_FIELD_STREAMS;
    static constexpr int U_NS = 6;
    static constexpr int P_NS = 3;
    for (int q = 0; q < NQ; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t det = determinant[goff];
            const s_t adj0 = adjugate[0][goff];
            const s_t adj1 = adjugate[1][goff];
            const s_t adj2 = adjugate[2][goff];
            const s_t adj3 = adjugate[3][goff];
            s_t u0 = s_t(0);
            s_t u0_grad_0_ref = s_t(0);
            s_t u0_grad_1_ref = s_t(0);
            const s_t coeff_current_u0_0 = current[0][lane];
            u0 += coeff_current_u0_0 * field_shape[0][q * U_NS + 0];
            u0_grad_0_ref += coeff_current_u0_0 * fgref[0 * ND + 0][q * U_NS + 0];
            u0_grad_1_ref += coeff_current_u0_0 * fgref[0 * ND + 1][q * U_NS + 0];
            const s_t coeff_current_u0_1 = current[1][lane];
            u0 += coeff_current_u0_1 * field_shape[0][q * U_NS + 1];
            u0_grad_0_ref += coeff_current_u0_1 * fgref[0 * ND + 0][q * U_NS + 1];
            u0_grad_1_ref += coeff_current_u0_1 * fgref[0 * ND + 1][q * U_NS + 1];
            const s_t coeff_current_u0_2 = current[2][lane];
            u0 += coeff_current_u0_2 * field_shape[0][q * U_NS + 2];
            u0_grad_0_ref += coeff_current_u0_2 * fgref[0 * ND + 0][q * U_NS + 2];
            u0_grad_1_ref += coeff_current_u0_2 * fgref[0 * ND + 1][q * U_NS + 2];
            const s_t coeff_current_u0_3 = current[3][lane];
            u0 += coeff_current_u0_3 * field_shape[0][q * U_NS + 3];
            u0_grad_0_ref += coeff_current_u0_3 * fgref[0 * ND + 0][q * U_NS + 3];
            u0_grad_1_ref += coeff_current_u0_3 * fgref[0 * ND + 1][q * U_NS + 3];
            const s_t coeff_current_u0_4 = current[4][lane];
            u0 += coeff_current_u0_4 * field_shape[0][q * U_NS + 4];
            u0_grad_0_ref += coeff_current_u0_4 * fgref[0 * ND + 0][q * U_NS + 4];
            u0_grad_1_ref += coeff_current_u0_4 * fgref[0 * ND + 1][q * U_NS + 4];
            const s_t coeff_current_u0_5 = current[5][lane];
            u0 += coeff_current_u0_5 * field_shape[0][q * U_NS + 5];
            u0_grad_0_ref += coeff_current_u0_5 * fgref[0 * ND + 0][q * U_NS + 5];
            u0_grad_1_ref += coeff_current_u0_5 * fgref[0 * ND + 1][q * U_NS + 5];
            const s_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
            const s_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
            s_t u0_old = s_t(0);
            const s_t coeff_previous_u0_0 = previous[0][lane];
            u0_old += coeff_previous_u0_0 * field_shape[0][q * U_NS + 0];
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
            s_t u1 = s_t(0);
            s_t u1_grad_0_ref = s_t(0);
            s_t u1_grad_1_ref = s_t(0);
            const s_t coeff_current_u1_0 = current[6][lane];
            u1 += coeff_current_u1_0 * field_shape[0][q * U_NS + 0];
            u1_grad_0_ref += coeff_current_u1_0 * fgref[0 * ND + 0][q * U_NS + 0];
            u1_grad_1_ref += coeff_current_u1_0 * fgref[0 * ND + 1][q * U_NS + 0];
            const s_t coeff_current_u1_1 = current[7][lane];
            u1 += coeff_current_u1_1 * field_shape[0][q * U_NS + 1];
            u1_grad_0_ref += coeff_current_u1_1 * fgref[0 * ND + 0][q * U_NS + 1];
            u1_grad_1_ref += coeff_current_u1_1 * fgref[0 * ND + 1][q * U_NS + 1];
            const s_t coeff_current_u1_2 = current[8][lane];
            u1 += coeff_current_u1_2 * field_shape[0][q * U_NS + 2];
            u1_grad_0_ref += coeff_current_u1_2 * fgref[0 * ND + 0][q * U_NS + 2];
            u1_grad_1_ref += coeff_current_u1_2 * fgref[0 * ND + 1][q * U_NS + 2];
            const s_t coeff_current_u1_3 = current[9][lane];
            u1 += coeff_current_u1_3 * field_shape[0][q * U_NS + 3];
            u1_grad_0_ref += coeff_current_u1_3 * fgref[0 * ND + 0][q * U_NS + 3];
            u1_grad_1_ref += coeff_current_u1_3 * fgref[0 * ND + 1][q * U_NS + 3];
            const s_t coeff_current_u1_4 = current[10][lane];
            u1 += coeff_current_u1_4 * field_shape[0][q * U_NS + 4];
            u1_grad_0_ref += coeff_current_u1_4 * fgref[0 * ND + 0][q * U_NS + 4];
            u1_grad_1_ref += coeff_current_u1_4 * fgref[0 * ND + 1][q * U_NS + 4];
            const s_t coeff_current_u1_5 = current[11][lane];
            u1 += coeff_current_u1_5 * field_shape[0][q * U_NS + 5];
            u1_grad_0_ref += coeff_current_u1_5 * fgref[0 * ND + 0][q * U_NS + 5];
            u1_grad_1_ref += coeff_current_u1_5 * fgref[0 * ND + 1][q * U_NS + 5];
            const s_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
            const s_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
            s_t u1_old = s_t(0);
            const s_t coeff_previous_u1_0 = previous[6][lane];
            u1_old += coeff_previous_u1_0 * field_shape[0][q * U_NS + 0];
            const s_t coeff_previous_u1_1 = previous[7][lane];
            u1_old += coeff_previous_u1_1 * field_shape[0][q * U_NS + 1];
            const s_t coeff_previous_u1_2 = previous[8][lane];
            u1_old += coeff_previous_u1_2 * field_shape[0][q * U_NS + 2];
            const s_t coeff_previous_u1_3 = previous[9][lane];
            u1_old += coeff_previous_u1_3 * field_shape[0][q * U_NS + 3];
            const s_t coeff_previous_u1_4 = previous[10][lane];
            u1_old += coeff_previous_u1_4 * field_shape[0][q * U_NS + 4];
            const s_t coeff_previous_u1_5 = previous[11][lane];
            u1_old += coeff_previous_u1_5 * field_shape[0][q * U_NS + 5];
            s_t p = s_t(0);
            s_t p_grad_0_ref = s_t(0);
            s_t p_grad_1_ref = s_t(0);
            const s_t coeff_current_p_0 = current[12][lane];
            p += coeff_current_p_0 * field_shape[1][q * P_NS + 0];
            p_grad_0_ref += coeff_current_p_0 * fgref[1 * ND + 0][q * P_NS + 0];
            p_grad_1_ref += coeff_current_p_0 * fgref[1 * ND + 1][q * P_NS + 0];
            const s_t coeff_current_p_1 = current[13][lane];
            p += coeff_current_p_1 * field_shape[1][q * P_NS + 1];
            p_grad_0_ref += coeff_current_p_1 * fgref[1 * ND + 0][q * P_NS + 1];
            p_grad_1_ref += coeff_current_p_1 * fgref[1 * ND + 1][q * P_NS + 1];
            const s_t coeff_current_p_2 = current[14][lane];
            p += coeff_current_p_2 * field_shape[1][q * P_NS + 2];
            p_grad_0_ref += coeff_current_p_2 * fgref[1 * ND + 0][q * P_NS + 2];
            p_grad_1_ref += coeff_current_p_2 * fgref[1 * ND + 1][q * P_NS + 2];
            const s_t p_grad_0 = (p_grad_0_ref * adj0 + p_grad_1_ref * adj2) / det;
            const s_t p_grad_1 = (p_grad_0_ref * adj1 + p_grad_1_ref * adj3) / det;
            s_t p_old = s_t(0);
            const s_t coeff_previous_p_0 = previous[12][lane];
            p_old += coeff_previous_p_0 * field_shape[1][q * P_NS + 0];
            const s_t coeff_previous_p_1 = previous[13][lane];
            p_old += coeff_previous_p_1 * field_shape[1][q * P_NS + 1];
            const s_t coeff_previous_p_2 = previous[14][lane];
            p_old += coeff_previous_p_2 * field_shape[1][q * P_NS + 2];
            const s_t residual_tmp0 = rho/dt;
            const s_t residual_tmp1 = -p;
            const s_t residual_tmp2 = nu*rho;
            const s_t residual_tmp3 = s_t(2)*residual_tmp2;
            const s_t residual_tmp4 = residual_tmp2*(u0_grad_1 + u1_grad_0);
            const s_t value_coeff0 = residual_tmp0*(dt*(convection_scale*(u0_grad_0*u0_old + u0_grad_1*u1_old) - f0) + u0 - u0_old);
            const s_t grad_coeff0_0 = residual_tmp1 + residual_tmp3*u0_grad_0;
            const s_t grad_coeff0_1 = residual_tmp4;
            const s_t value_coeff1 = residual_tmp0*(dt*(convection_scale*(u0_old*u1_grad_0 + u1_grad_1*u1_old) - f1) + u1 - u1_old);
            const s_t grad_coeff1_0 = residual_tmp4;
            const s_t grad_coeff1_1 = residual_tmp1 + residual_tmp3*u1_grad_1;
            const s_t value_coeff2 = u0_grad_0 + u1_grad_1;
            const s_t test_value_u0_0 = field_shape[0][q * U_NS + 0];
            const s_t test_grad0_u0_0 = (fgref[0 * ND + 0][q * U_NS + 0] * adj0 + fgref[0 * ND + 1][q * U_NS + 0] * adj2) / det;
            const s_t test_grad1_u0_0 = (fgref[0 * ND + 0][q * U_NS + 0] * adj1 + fgref[0 * ND + 1][q * U_NS + 0] * adj3) / det;
            output[0][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_0 + grad_coeff0_0 * test_grad0_u0_0 + grad_coeff0_1 * test_grad1_u0_0);
            const s_t test_value_u0_1 = field_shape[0][q * U_NS + 1];
            const s_t test_grad0_u0_1 = (fgref[0 * ND + 0][q * U_NS + 1] * adj0 + fgref[0 * ND + 1][q * U_NS + 1] * adj2) / det;
            const s_t test_grad1_u0_1 = (fgref[0 * ND + 0][q * U_NS + 1] * adj1 + fgref[0 * ND + 1][q * U_NS + 1] * adj3) / det;
            output[1][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_1 + grad_coeff0_0 * test_grad0_u0_1 + grad_coeff0_1 * test_grad1_u0_1);
            const s_t test_value_u0_2 = field_shape[0][q * U_NS + 2];
            const s_t test_grad0_u0_2 = (fgref[0 * ND + 0][q * U_NS + 2] * adj0 + fgref[0 * ND + 1][q * U_NS + 2] * adj2) / det;
            const s_t test_grad1_u0_2 = (fgref[0 * ND + 0][q * U_NS + 2] * adj1 + fgref[0 * ND + 1][q * U_NS + 2] * adj3) / det;
            output[2][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_2 + grad_coeff0_0 * test_grad0_u0_2 + grad_coeff0_1 * test_grad1_u0_2);
            const s_t test_value_u0_3 = field_shape[0][q * U_NS + 3];
            const s_t test_grad0_u0_3 = (fgref[0 * ND + 0][q * U_NS + 3] * adj0 + fgref[0 * ND + 1][q * U_NS + 3] * adj2) / det;
            const s_t test_grad1_u0_3 = (fgref[0 * ND + 0][q * U_NS + 3] * adj1 + fgref[0 * ND + 1][q * U_NS + 3] * adj3) / det;
            output[3][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_3 + grad_coeff0_0 * test_grad0_u0_3 + grad_coeff0_1 * test_grad1_u0_3);
            const s_t test_value_u0_4 = field_shape[0][q * U_NS + 4];
            const s_t test_grad0_u0_4 = (fgref[0 * ND + 0][q * U_NS + 4] * adj0 + fgref[0 * ND + 1][q * U_NS + 4] * adj2) / det;
            const s_t test_grad1_u0_4 = (fgref[0 * ND + 0][q * U_NS + 4] * adj1 + fgref[0 * ND + 1][q * U_NS + 4] * adj3) / det;
            output[4][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_4 + grad_coeff0_0 * test_grad0_u0_4 + grad_coeff0_1 * test_grad1_u0_4);
            const s_t test_value_u0_5 = field_shape[0][q * U_NS + 5];
            const s_t test_grad0_u0_5 = (fgref[0 * ND + 0][q * U_NS + 5] * adj0 + fgref[0 * ND + 1][q * U_NS + 5] * adj2) / det;
            const s_t test_grad1_u0_5 = (fgref[0 * ND + 0][q * U_NS + 5] * adj1 + fgref[0 * ND + 1][q * U_NS + 5] * adj3) / det;
            output[5][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_5 + grad_coeff0_0 * test_grad0_u0_5 + grad_coeff0_1 * test_grad1_u0_5);
            const s_t test_value_u1_0 = field_shape[0][q * U_NS + 0];
            const s_t test_grad0_u1_0 = (fgref[0 * ND + 0][q * U_NS + 0] * adj0 + fgref[0 * ND + 1][q * U_NS + 0] * adj2) / det;
            const s_t test_grad1_u1_0 = (fgref[0 * ND + 0][q * U_NS + 0] * adj1 + fgref[0 * ND + 1][q * U_NS + 0] * adj3) / det;
            output[6][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_0 + grad_coeff1_0 * test_grad0_u1_0 + grad_coeff1_1 * test_grad1_u1_0);
            const s_t test_value_u1_1 = field_shape[0][q * U_NS + 1];
            const s_t test_grad0_u1_1 = (fgref[0 * ND + 0][q * U_NS + 1] * adj0 + fgref[0 * ND + 1][q * U_NS + 1] * adj2) / det;
            const s_t test_grad1_u1_1 = (fgref[0 * ND + 0][q * U_NS + 1] * adj1 + fgref[0 * ND + 1][q * U_NS + 1] * adj3) / det;
            output[7][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_1 + grad_coeff1_0 * test_grad0_u1_1 + grad_coeff1_1 * test_grad1_u1_1);
            const s_t test_value_u1_2 = field_shape[0][q * U_NS + 2];
            const s_t test_grad0_u1_2 = (fgref[0 * ND + 0][q * U_NS + 2] * adj0 + fgref[0 * ND + 1][q * U_NS + 2] * adj2) / det;
            const s_t test_grad1_u1_2 = (fgref[0 * ND + 0][q * U_NS + 2] * adj1 + fgref[0 * ND + 1][q * U_NS + 2] * adj3) / det;
            output[8][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_2 + grad_coeff1_0 * test_grad0_u1_2 + grad_coeff1_1 * test_grad1_u1_2);
            const s_t test_value_u1_3 = field_shape[0][q * U_NS + 3];
            const s_t test_grad0_u1_3 = (fgref[0 * ND + 0][q * U_NS + 3] * adj0 + fgref[0 * ND + 1][q * U_NS + 3] * adj2) / det;
            const s_t test_grad1_u1_3 = (fgref[0 * ND + 0][q * U_NS + 3] * adj1 + fgref[0 * ND + 1][q * U_NS + 3] * adj3) / det;
            output[9][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_3 + grad_coeff1_0 * test_grad0_u1_3 + grad_coeff1_1 * test_grad1_u1_3);
            const s_t test_value_u1_4 = field_shape[0][q * U_NS + 4];
            const s_t test_grad0_u1_4 = (fgref[0 * ND + 0][q * U_NS + 4] * adj0 + fgref[0 * ND + 1][q * U_NS + 4] * adj2) / det;
            const s_t test_grad1_u1_4 = (fgref[0 * ND + 0][q * U_NS + 4] * adj1 + fgref[0 * ND + 1][q * U_NS + 4] * adj3) / det;
            output[10][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_4 + grad_coeff1_0 * test_grad0_u1_4 + grad_coeff1_1 * test_grad1_u1_4);
            const s_t test_value_u1_5 = field_shape[0][q * U_NS + 5];
            const s_t test_grad0_u1_5 = (fgref[0 * ND + 0][q * U_NS + 5] * adj0 + fgref[0 * ND + 1][q * U_NS + 5] * adj2) / det;
            const s_t test_grad1_u1_5 = (fgref[0 * ND + 0][q * U_NS + 5] * adj1 + fgref[0 * ND + 1][q * U_NS + 5] * adj3) / det;
            output[11][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_5 + grad_coeff1_0 * test_grad0_u1_5 + grad_coeff1_1 * test_grad1_u1_5);
            const s_t test_value_p_0 = field_shape[1][q * P_NS + 0];
            output[12][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_0);
            const s_t test_value_p_1 = field_shape[1][q * P_NS + 1];
            output[13][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_1);
            const s_t test_value_p_2 = field_shape[1][q * P_NS + 2];
            output[14][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_2);
        }
    }
}

template <typename s_t, int NQ, int CELL_NS, int VS>
static SFEM_INLINE void navier_stokes_d2_simplex_mixed_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR adjugate[4],
        const s_t *const RSTR field_shape[2],
        const s_t *const RSTR fgref[4],
        const s_t *const RSTR q_weight,
        const s_t *const RSTR previous[15],
        const s_t *const RSTR direction[15],
        const s_t convection_scale,
        const s_t dt,
        const s_t nu,
        const s_t rho,
        s_t *const RSTR output[15]
) {
    static constexpr int ND = 2;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 15;
    (void)CELL_NS;
    (void)N_FIELD_STREAMS;
    static constexpr int U_NS = 6;
    static constexpr int P_NS = 3;
    for (int q = 0; q < NQ; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t det = determinant[goff];
            const s_t adj0 = adjugate[0][goff];
            const s_t adj1 = adjugate[1][goff];
            const s_t adj2 = adjugate[2][goff];
            const s_t adj3 = adjugate[3][goff];
            s_t u0_old = s_t(0);
            const s_t coeff_previous_u0_0 = previous[0][lane];
            u0_old += coeff_previous_u0_0 * field_shape[0][q * U_NS + 0];
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
            s_t u0_direction = s_t(0);
            s_t u0_direction_grad_0_ref = s_t(0);
            s_t u0_direction_grad_1_ref = s_t(0);
            const s_t coeff_direction_u0_0 = direction[0][lane];
            u0_direction += coeff_direction_u0_0 * field_shape[0][q * U_NS + 0];
            u0_direction_grad_0_ref += coeff_direction_u0_0 * fgref[0 * ND + 0][q * U_NS + 0];
            u0_direction_grad_1_ref += coeff_direction_u0_0 * fgref[0 * ND + 1][q * U_NS + 0];
            const s_t coeff_direction_u0_1 = direction[1][lane];
            u0_direction += coeff_direction_u0_1 * field_shape[0][q * U_NS + 1];
            u0_direction_grad_0_ref += coeff_direction_u0_1 * fgref[0 * ND + 0][q * U_NS + 1];
            u0_direction_grad_1_ref += coeff_direction_u0_1 * fgref[0 * ND + 1][q * U_NS + 1];
            const s_t coeff_direction_u0_2 = direction[2][lane];
            u0_direction += coeff_direction_u0_2 * field_shape[0][q * U_NS + 2];
            u0_direction_grad_0_ref += coeff_direction_u0_2 * fgref[0 * ND + 0][q * U_NS + 2];
            u0_direction_grad_1_ref += coeff_direction_u0_2 * fgref[0 * ND + 1][q * U_NS + 2];
            const s_t coeff_direction_u0_3 = direction[3][lane];
            u0_direction += coeff_direction_u0_3 * field_shape[0][q * U_NS + 3];
            u0_direction_grad_0_ref += coeff_direction_u0_3 * fgref[0 * ND + 0][q * U_NS + 3];
            u0_direction_grad_1_ref += coeff_direction_u0_3 * fgref[0 * ND + 1][q * U_NS + 3];
            const s_t coeff_direction_u0_4 = direction[4][lane];
            u0_direction += coeff_direction_u0_4 * field_shape[0][q * U_NS + 4];
            u0_direction_grad_0_ref += coeff_direction_u0_4 * fgref[0 * ND + 0][q * U_NS + 4];
            u0_direction_grad_1_ref += coeff_direction_u0_4 * fgref[0 * ND + 1][q * U_NS + 4];
            const s_t coeff_direction_u0_5 = direction[5][lane];
            u0_direction += coeff_direction_u0_5 * field_shape[0][q * U_NS + 5];
            u0_direction_grad_0_ref += coeff_direction_u0_5 * fgref[0 * ND + 0][q * U_NS + 5];
            u0_direction_grad_1_ref += coeff_direction_u0_5 * fgref[0 * ND + 1][q * U_NS + 5];
            const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
            const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
            s_t u1_old = s_t(0);
            const s_t coeff_previous_u1_0 = previous[6][lane];
            u1_old += coeff_previous_u1_0 * field_shape[0][q * U_NS + 0];
            const s_t coeff_previous_u1_1 = previous[7][lane];
            u1_old += coeff_previous_u1_1 * field_shape[0][q * U_NS + 1];
            const s_t coeff_previous_u1_2 = previous[8][lane];
            u1_old += coeff_previous_u1_2 * field_shape[0][q * U_NS + 2];
            const s_t coeff_previous_u1_3 = previous[9][lane];
            u1_old += coeff_previous_u1_3 * field_shape[0][q * U_NS + 3];
            const s_t coeff_previous_u1_4 = previous[10][lane];
            u1_old += coeff_previous_u1_4 * field_shape[0][q * U_NS + 4];
            const s_t coeff_previous_u1_5 = previous[11][lane];
            u1_old += coeff_previous_u1_5 * field_shape[0][q * U_NS + 5];
            s_t u1_direction = s_t(0);
            s_t u1_direction_grad_0_ref = s_t(0);
            s_t u1_direction_grad_1_ref = s_t(0);
            const s_t coeff_direction_u1_0 = direction[6][lane];
            u1_direction += coeff_direction_u1_0 * field_shape[0][q * U_NS + 0];
            u1_direction_grad_0_ref += coeff_direction_u1_0 * fgref[0 * ND + 0][q * U_NS + 0];
            u1_direction_grad_1_ref += coeff_direction_u1_0 * fgref[0 * ND + 1][q * U_NS + 0];
            const s_t coeff_direction_u1_1 = direction[7][lane];
            u1_direction += coeff_direction_u1_1 * field_shape[0][q * U_NS + 1];
            u1_direction_grad_0_ref += coeff_direction_u1_1 * fgref[0 * ND + 0][q * U_NS + 1];
            u1_direction_grad_1_ref += coeff_direction_u1_1 * fgref[0 * ND + 1][q * U_NS + 1];
            const s_t coeff_direction_u1_2 = direction[8][lane];
            u1_direction += coeff_direction_u1_2 * field_shape[0][q * U_NS + 2];
            u1_direction_grad_0_ref += coeff_direction_u1_2 * fgref[0 * ND + 0][q * U_NS + 2];
            u1_direction_grad_1_ref += coeff_direction_u1_2 * fgref[0 * ND + 1][q * U_NS + 2];
            const s_t coeff_direction_u1_3 = direction[9][lane];
            u1_direction += coeff_direction_u1_3 * field_shape[0][q * U_NS + 3];
            u1_direction_grad_0_ref += coeff_direction_u1_3 * fgref[0 * ND + 0][q * U_NS + 3];
            u1_direction_grad_1_ref += coeff_direction_u1_3 * fgref[0 * ND + 1][q * U_NS + 3];
            const s_t coeff_direction_u1_4 = direction[10][lane];
            u1_direction += coeff_direction_u1_4 * field_shape[0][q * U_NS + 4];
            u1_direction_grad_0_ref += coeff_direction_u1_4 * fgref[0 * ND + 0][q * U_NS + 4];
            u1_direction_grad_1_ref += coeff_direction_u1_4 * fgref[0 * ND + 1][q * U_NS + 4];
            const s_t coeff_direction_u1_5 = direction[11][lane];
            u1_direction += coeff_direction_u1_5 * field_shape[0][q * U_NS + 5];
            u1_direction_grad_0_ref += coeff_direction_u1_5 * fgref[0 * ND + 0][q * U_NS + 5];
            u1_direction_grad_1_ref += coeff_direction_u1_5 * fgref[0 * ND + 1][q * U_NS + 5];
            const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj2) / det;
            const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj3) / det;
            s_t p_old = s_t(0);
            const s_t coeff_previous_p_0 = previous[12][lane];
            p_old += coeff_previous_p_0 * field_shape[1][q * P_NS + 0];
            const s_t coeff_previous_p_1 = previous[13][lane];
            p_old += coeff_previous_p_1 * field_shape[1][q * P_NS + 1];
            const s_t coeff_previous_p_2 = previous[14][lane];
            p_old += coeff_previous_p_2 * field_shape[1][q * P_NS + 2];
            s_t p_direction = s_t(0);
            s_t p_direction_grad_0_ref = s_t(0);
            s_t p_direction_grad_1_ref = s_t(0);
            const s_t coeff_direction_p_0 = direction[12][lane];
            p_direction += coeff_direction_p_0 * field_shape[1][q * P_NS + 0];
            p_direction_grad_0_ref += coeff_direction_p_0 * fgref[1 * ND + 0][q * P_NS + 0];
            p_direction_grad_1_ref += coeff_direction_p_0 * fgref[1 * ND + 1][q * P_NS + 0];
            const s_t coeff_direction_p_1 = direction[13][lane];
            p_direction += coeff_direction_p_1 * field_shape[1][q * P_NS + 1];
            p_direction_grad_0_ref += coeff_direction_p_1 * fgref[1 * ND + 0][q * P_NS + 1];
            p_direction_grad_1_ref += coeff_direction_p_1 * fgref[1 * ND + 1][q * P_NS + 1];
            const s_t coeff_direction_p_2 = direction[14][lane];
            p_direction += coeff_direction_p_2 * field_shape[1][q * P_NS + 2];
            p_direction_grad_0_ref += coeff_direction_p_2 * fgref[1 * ND + 0][q * P_NS + 2];
            p_direction_grad_1_ref += coeff_direction_p_2 * fgref[1 * ND + 1][q * P_NS + 2];
            const s_t p_direction_grad_0 = (p_direction_grad_0_ref * adj0 + p_direction_grad_1_ref * adj2) / det;
            const s_t p_direction_grad_1 = (p_direction_grad_0_ref * adj1 + p_direction_grad_1_ref * adj3) / det;
            const s_t residual_tmp0 = convection_scale*rho;
            const s_t residual_tmp1 = residual_tmp0*u0_old;
            const s_t residual_tmp2 = residual_tmp0*u1_old;
            const s_t residual_tmp3 = rho/dt;
            const s_t residual_tmp4 = -p_direction;
            const s_t residual_tmp5 = nu*rho;
            const s_t residual_tmp6 = s_t(2)*residual_tmp5;
            const s_t residual_tmp7 = residual_tmp5*u0_direction_grad_1 + residual_tmp5*u1_direction_grad_0;
            const s_t value_coeff0 = residual_tmp1*u0_direction_grad_0 + residual_tmp2*u0_direction_grad_1 + residual_tmp3*u0_direction;
            const s_t grad_coeff0_0 = residual_tmp4 + residual_tmp6*u0_direction_grad_0;
            const s_t grad_coeff0_1 = residual_tmp7;
            const s_t value_coeff1 = residual_tmp1*u1_direction_grad_0 + residual_tmp2*u1_direction_grad_1 + residual_tmp3*u1_direction;
            const s_t grad_coeff1_0 = residual_tmp7;
            const s_t grad_coeff1_1 = residual_tmp4 + residual_tmp6*u1_direction_grad_1;
            const s_t value_coeff2 = u0_direction_grad_0 + u1_direction_grad_1;
            const s_t test_value_u0_0 = field_shape[0][q * U_NS + 0];
            const s_t test_grad0_u0_0 = (fgref[0 * ND + 0][q * U_NS + 0] * adj0 + fgref[0 * ND + 1][q * U_NS + 0] * adj2) / det;
            const s_t test_grad1_u0_0 = (fgref[0 * ND + 0][q * U_NS + 0] * adj1 + fgref[0 * ND + 1][q * U_NS + 0] * adj3) / det;
            output[0][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_0 + grad_coeff0_0 * test_grad0_u0_0 + grad_coeff0_1 * test_grad1_u0_0);
            const s_t test_value_u0_1 = field_shape[0][q * U_NS + 1];
            const s_t test_grad0_u0_1 = (fgref[0 * ND + 0][q * U_NS + 1] * adj0 + fgref[0 * ND + 1][q * U_NS + 1] * adj2) / det;
            const s_t test_grad1_u0_1 = (fgref[0 * ND + 0][q * U_NS + 1] * adj1 + fgref[0 * ND + 1][q * U_NS + 1] * adj3) / det;
            output[1][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_1 + grad_coeff0_0 * test_grad0_u0_1 + grad_coeff0_1 * test_grad1_u0_1);
            const s_t test_value_u0_2 = field_shape[0][q * U_NS + 2];
            const s_t test_grad0_u0_2 = (fgref[0 * ND + 0][q * U_NS + 2] * adj0 + fgref[0 * ND + 1][q * U_NS + 2] * adj2) / det;
            const s_t test_grad1_u0_2 = (fgref[0 * ND + 0][q * U_NS + 2] * adj1 + fgref[0 * ND + 1][q * U_NS + 2] * adj3) / det;
            output[2][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_2 + grad_coeff0_0 * test_grad0_u0_2 + grad_coeff0_1 * test_grad1_u0_2);
            const s_t test_value_u0_3 = field_shape[0][q * U_NS + 3];
            const s_t test_grad0_u0_3 = (fgref[0 * ND + 0][q * U_NS + 3] * adj0 + fgref[0 * ND + 1][q * U_NS + 3] * adj2) / det;
            const s_t test_grad1_u0_3 = (fgref[0 * ND + 0][q * U_NS + 3] * adj1 + fgref[0 * ND + 1][q * U_NS + 3] * adj3) / det;
            output[3][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_3 + grad_coeff0_0 * test_grad0_u0_3 + grad_coeff0_1 * test_grad1_u0_3);
            const s_t test_value_u0_4 = field_shape[0][q * U_NS + 4];
            const s_t test_grad0_u0_4 = (fgref[0 * ND + 0][q * U_NS + 4] * adj0 + fgref[0 * ND + 1][q * U_NS + 4] * adj2) / det;
            const s_t test_grad1_u0_4 = (fgref[0 * ND + 0][q * U_NS + 4] * adj1 + fgref[0 * ND + 1][q * U_NS + 4] * adj3) / det;
            output[4][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_4 + grad_coeff0_0 * test_grad0_u0_4 + grad_coeff0_1 * test_grad1_u0_4);
            const s_t test_value_u0_5 = field_shape[0][q * U_NS + 5];
            const s_t test_grad0_u0_5 = (fgref[0 * ND + 0][q * U_NS + 5] * adj0 + fgref[0 * ND + 1][q * U_NS + 5] * adj2) / det;
            const s_t test_grad1_u0_5 = (fgref[0 * ND + 0][q * U_NS + 5] * adj1 + fgref[0 * ND + 1][q * U_NS + 5] * adj3) / det;
            output[5][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_5 + grad_coeff0_0 * test_grad0_u0_5 + grad_coeff0_1 * test_grad1_u0_5);
            const s_t test_value_u1_0 = field_shape[0][q * U_NS + 0];
            const s_t test_grad0_u1_0 = (fgref[0 * ND + 0][q * U_NS + 0] * adj0 + fgref[0 * ND + 1][q * U_NS + 0] * adj2) / det;
            const s_t test_grad1_u1_0 = (fgref[0 * ND + 0][q * U_NS + 0] * adj1 + fgref[0 * ND + 1][q * U_NS + 0] * adj3) / det;
            output[6][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_0 + grad_coeff1_0 * test_grad0_u1_0 + grad_coeff1_1 * test_grad1_u1_0);
            const s_t test_value_u1_1 = field_shape[0][q * U_NS + 1];
            const s_t test_grad0_u1_1 = (fgref[0 * ND + 0][q * U_NS + 1] * adj0 + fgref[0 * ND + 1][q * U_NS + 1] * adj2) / det;
            const s_t test_grad1_u1_1 = (fgref[0 * ND + 0][q * U_NS + 1] * adj1 + fgref[0 * ND + 1][q * U_NS + 1] * adj3) / det;
            output[7][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_1 + grad_coeff1_0 * test_grad0_u1_1 + grad_coeff1_1 * test_grad1_u1_1);
            const s_t test_value_u1_2 = field_shape[0][q * U_NS + 2];
            const s_t test_grad0_u1_2 = (fgref[0 * ND + 0][q * U_NS + 2] * adj0 + fgref[0 * ND + 1][q * U_NS + 2] * adj2) / det;
            const s_t test_grad1_u1_2 = (fgref[0 * ND + 0][q * U_NS + 2] * adj1 + fgref[0 * ND + 1][q * U_NS + 2] * adj3) / det;
            output[8][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_2 + grad_coeff1_0 * test_grad0_u1_2 + grad_coeff1_1 * test_grad1_u1_2);
            const s_t test_value_u1_3 = field_shape[0][q * U_NS + 3];
            const s_t test_grad0_u1_3 = (fgref[0 * ND + 0][q * U_NS + 3] * adj0 + fgref[0 * ND + 1][q * U_NS + 3] * adj2) / det;
            const s_t test_grad1_u1_3 = (fgref[0 * ND + 0][q * U_NS + 3] * adj1 + fgref[0 * ND + 1][q * U_NS + 3] * adj3) / det;
            output[9][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_3 + grad_coeff1_0 * test_grad0_u1_3 + grad_coeff1_1 * test_grad1_u1_3);
            const s_t test_value_u1_4 = field_shape[0][q * U_NS + 4];
            const s_t test_grad0_u1_4 = (fgref[0 * ND + 0][q * U_NS + 4] * adj0 + fgref[0 * ND + 1][q * U_NS + 4] * adj2) / det;
            const s_t test_grad1_u1_4 = (fgref[0 * ND + 0][q * U_NS + 4] * adj1 + fgref[0 * ND + 1][q * U_NS + 4] * adj3) / det;
            output[10][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_4 + grad_coeff1_0 * test_grad0_u1_4 + grad_coeff1_1 * test_grad1_u1_4);
            const s_t test_value_u1_5 = field_shape[0][q * U_NS + 5];
            const s_t test_grad0_u1_5 = (fgref[0 * ND + 0][q * U_NS + 5] * adj0 + fgref[0 * ND + 1][q * U_NS + 5] * adj2) / det;
            const s_t test_grad1_u1_5 = (fgref[0 * ND + 0][q * U_NS + 5] * adj1 + fgref[0 * ND + 1][q * U_NS + 5] * adj3) / det;
            output[11][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_5 + grad_coeff1_0 * test_grad0_u1_5 + grad_coeff1_1 * test_grad1_u1_5);
            const s_t test_value_p_0 = field_shape[1][q * P_NS + 0];
            output[12][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_0);
            const s_t test_value_p_1 = field_shape[1][q * P_NS + 1];
            output[13][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_1);
            const s_t test_value_p_2 = field_shape[1][q * P_NS + 2];
            output[14][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_2);
        }
    }
}

template <typename s_t, int NQ, int CELL_NS, int VS>
static SFEM_INLINE void navier_stokes_d2_simplex_mixed_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR adjugate[4],
        const s_t *const RSTR field_shape[2],
        const s_t *const RSTR fgref[4],
        const s_t *const RSTR q_weight,
        const s_t previous[15][VS],
        const s_t direction[15][VS],
        const s_t convection_scale,
        const s_t dt,
        const s_t nu,
        const s_t rho,
        s_t output[15][VS]
) {
    static constexpr int ND = 2;
    static constexpr int NC = 2;
    static constexpr int N_FIELD_STREAMS = 15;
    (void)CELL_NS;
    (void)N_FIELD_STREAMS;
    static constexpr int U_NS = 6;
    static constexpr int P_NS = 3;
    for (int q = 0; q < NQ; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t det = determinant[goff];
            const s_t adj0 = adjugate[0][goff];
            const s_t adj1 = adjugate[1][goff];
            const s_t adj2 = adjugate[2][goff];
            const s_t adj3 = adjugate[3][goff];
            s_t u0_old = s_t(0);
            const s_t coeff_previous_u0_0 = previous[0][lane];
            u0_old += coeff_previous_u0_0 * field_shape[0][q * U_NS + 0];
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
            s_t u0_direction = s_t(0);
            s_t u0_direction_grad_0_ref = s_t(0);
            s_t u0_direction_grad_1_ref = s_t(0);
            const s_t coeff_direction_u0_0 = direction[0][lane];
            u0_direction += coeff_direction_u0_0 * field_shape[0][q * U_NS + 0];
            u0_direction_grad_0_ref += coeff_direction_u0_0 * fgref[0 * ND + 0][q * U_NS + 0];
            u0_direction_grad_1_ref += coeff_direction_u0_0 * fgref[0 * ND + 1][q * U_NS + 0];
            const s_t coeff_direction_u0_1 = direction[1][lane];
            u0_direction += coeff_direction_u0_1 * field_shape[0][q * U_NS + 1];
            u0_direction_grad_0_ref += coeff_direction_u0_1 * fgref[0 * ND + 0][q * U_NS + 1];
            u0_direction_grad_1_ref += coeff_direction_u0_1 * fgref[0 * ND + 1][q * U_NS + 1];
            const s_t coeff_direction_u0_2 = direction[2][lane];
            u0_direction += coeff_direction_u0_2 * field_shape[0][q * U_NS + 2];
            u0_direction_grad_0_ref += coeff_direction_u0_2 * fgref[0 * ND + 0][q * U_NS + 2];
            u0_direction_grad_1_ref += coeff_direction_u0_2 * fgref[0 * ND + 1][q * U_NS + 2];
            const s_t coeff_direction_u0_3 = direction[3][lane];
            u0_direction += coeff_direction_u0_3 * field_shape[0][q * U_NS + 3];
            u0_direction_grad_0_ref += coeff_direction_u0_3 * fgref[0 * ND + 0][q * U_NS + 3];
            u0_direction_grad_1_ref += coeff_direction_u0_3 * fgref[0 * ND + 1][q * U_NS + 3];
            const s_t coeff_direction_u0_4 = direction[4][lane];
            u0_direction += coeff_direction_u0_4 * field_shape[0][q * U_NS + 4];
            u0_direction_grad_0_ref += coeff_direction_u0_4 * fgref[0 * ND + 0][q * U_NS + 4];
            u0_direction_grad_1_ref += coeff_direction_u0_4 * fgref[0 * ND + 1][q * U_NS + 4];
            const s_t coeff_direction_u0_5 = direction[5][lane];
            u0_direction += coeff_direction_u0_5 * field_shape[0][q * U_NS + 5];
            u0_direction_grad_0_ref += coeff_direction_u0_5 * fgref[0 * ND + 0][q * U_NS + 5];
            u0_direction_grad_1_ref += coeff_direction_u0_5 * fgref[0 * ND + 1][q * U_NS + 5];
            const s_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
            const s_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
            s_t u1_old = s_t(0);
            const s_t coeff_previous_u1_0 = previous[6][lane];
            u1_old += coeff_previous_u1_0 * field_shape[0][q * U_NS + 0];
            const s_t coeff_previous_u1_1 = previous[7][lane];
            u1_old += coeff_previous_u1_1 * field_shape[0][q * U_NS + 1];
            const s_t coeff_previous_u1_2 = previous[8][lane];
            u1_old += coeff_previous_u1_2 * field_shape[0][q * U_NS + 2];
            const s_t coeff_previous_u1_3 = previous[9][lane];
            u1_old += coeff_previous_u1_3 * field_shape[0][q * U_NS + 3];
            const s_t coeff_previous_u1_4 = previous[10][lane];
            u1_old += coeff_previous_u1_4 * field_shape[0][q * U_NS + 4];
            const s_t coeff_previous_u1_5 = previous[11][lane];
            u1_old += coeff_previous_u1_5 * field_shape[0][q * U_NS + 5];
            s_t u1_direction = s_t(0);
            s_t u1_direction_grad_0_ref = s_t(0);
            s_t u1_direction_grad_1_ref = s_t(0);
            const s_t coeff_direction_u1_0 = direction[6][lane];
            u1_direction += coeff_direction_u1_0 * field_shape[0][q * U_NS + 0];
            u1_direction_grad_0_ref += coeff_direction_u1_0 * fgref[0 * ND + 0][q * U_NS + 0];
            u1_direction_grad_1_ref += coeff_direction_u1_0 * fgref[0 * ND + 1][q * U_NS + 0];
            const s_t coeff_direction_u1_1 = direction[7][lane];
            u1_direction += coeff_direction_u1_1 * field_shape[0][q * U_NS + 1];
            u1_direction_grad_0_ref += coeff_direction_u1_1 * fgref[0 * ND + 0][q * U_NS + 1];
            u1_direction_grad_1_ref += coeff_direction_u1_1 * fgref[0 * ND + 1][q * U_NS + 1];
            const s_t coeff_direction_u1_2 = direction[8][lane];
            u1_direction += coeff_direction_u1_2 * field_shape[0][q * U_NS + 2];
            u1_direction_grad_0_ref += coeff_direction_u1_2 * fgref[0 * ND + 0][q * U_NS + 2];
            u1_direction_grad_1_ref += coeff_direction_u1_2 * fgref[0 * ND + 1][q * U_NS + 2];
            const s_t coeff_direction_u1_3 = direction[9][lane];
            u1_direction += coeff_direction_u1_3 * field_shape[0][q * U_NS + 3];
            u1_direction_grad_0_ref += coeff_direction_u1_3 * fgref[0 * ND + 0][q * U_NS + 3];
            u1_direction_grad_1_ref += coeff_direction_u1_3 * fgref[0 * ND + 1][q * U_NS + 3];
            const s_t coeff_direction_u1_4 = direction[10][lane];
            u1_direction += coeff_direction_u1_4 * field_shape[0][q * U_NS + 4];
            u1_direction_grad_0_ref += coeff_direction_u1_4 * fgref[0 * ND + 0][q * U_NS + 4];
            u1_direction_grad_1_ref += coeff_direction_u1_4 * fgref[0 * ND + 1][q * U_NS + 4];
            const s_t coeff_direction_u1_5 = direction[11][lane];
            u1_direction += coeff_direction_u1_5 * field_shape[0][q * U_NS + 5];
            u1_direction_grad_0_ref += coeff_direction_u1_5 * fgref[0 * ND + 0][q * U_NS + 5];
            u1_direction_grad_1_ref += coeff_direction_u1_5 * fgref[0 * ND + 1][q * U_NS + 5];
            const s_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj2) / det;
            const s_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj3) / det;
            s_t p_old = s_t(0);
            const s_t coeff_previous_p_0 = previous[12][lane];
            p_old += coeff_previous_p_0 * field_shape[1][q * P_NS + 0];
            const s_t coeff_previous_p_1 = previous[13][lane];
            p_old += coeff_previous_p_1 * field_shape[1][q * P_NS + 1];
            const s_t coeff_previous_p_2 = previous[14][lane];
            p_old += coeff_previous_p_2 * field_shape[1][q * P_NS + 2];
            s_t p_direction = s_t(0);
            s_t p_direction_grad_0_ref = s_t(0);
            s_t p_direction_grad_1_ref = s_t(0);
            const s_t coeff_direction_p_0 = direction[12][lane];
            p_direction += coeff_direction_p_0 * field_shape[1][q * P_NS + 0];
            p_direction_grad_0_ref += coeff_direction_p_0 * fgref[1 * ND + 0][q * P_NS + 0];
            p_direction_grad_1_ref += coeff_direction_p_0 * fgref[1 * ND + 1][q * P_NS + 0];
            const s_t coeff_direction_p_1 = direction[13][lane];
            p_direction += coeff_direction_p_1 * field_shape[1][q * P_NS + 1];
            p_direction_grad_0_ref += coeff_direction_p_1 * fgref[1 * ND + 0][q * P_NS + 1];
            p_direction_grad_1_ref += coeff_direction_p_1 * fgref[1 * ND + 1][q * P_NS + 1];
            const s_t coeff_direction_p_2 = direction[14][lane];
            p_direction += coeff_direction_p_2 * field_shape[1][q * P_NS + 2];
            p_direction_grad_0_ref += coeff_direction_p_2 * fgref[1 * ND + 0][q * P_NS + 2];
            p_direction_grad_1_ref += coeff_direction_p_2 * fgref[1 * ND + 1][q * P_NS + 2];
            const s_t p_direction_grad_0 = (p_direction_grad_0_ref * adj0 + p_direction_grad_1_ref * adj2) / det;
            const s_t p_direction_grad_1 = (p_direction_grad_0_ref * adj1 + p_direction_grad_1_ref * adj3) / det;
            const s_t residual_tmp0 = convection_scale*rho;
            const s_t residual_tmp1 = residual_tmp0*u0_old;
            const s_t residual_tmp2 = residual_tmp0*u1_old;
            const s_t residual_tmp3 = rho/dt;
            const s_t residual_tmp4 = -p_direction;
            const s_t residual_tmp5 = nu*rho;
            const s_t residual_tmp6 = s_t(2)*residual_tmp5;
            const s_t residual_tmp7 = residual_tmp5*u0_direction_grad_1 + residual_tmp5*u1_direction_grad_0;
            const s_t value_coeff0 = residual_tmp1*u0_direction_grad_0 + residual_tmp2*u0_direction_grad_1 + residual_tmp3*u0_direction;
            const s_t grad_coeff0_0 = residual_tmp4 + residual_tmp6*u0_direction_grad_0;
            const s_t grad_coeff0_1 = residual_tmp7;
            const s_t value_coeff1 = residual_tmp1*u1_direction_grad_0 + residual_tmp2*u1_direction_grad_1 + residual_tmp3*u1_direction;
            const s_t grad_coeff1_0 = residual_tmp7;
            const s_t grad_coeff1_1 = residual_tmp4 + residual_tmp6*u1_direction_grad_1;
            const s_t value_coeff2 = u0_direction_grad_0 + u1_direction_grad_1;
            const s_t test_value_u0_0 = field_shape[0][q * U_NS + 0];
            const s_t test_grad0_u0_0 = (fgref[0 * ND + 0][q * U_NS + 0] * adj0 + fgref[0 * ND + 1][q * U_NS + 0] * adj2) / det;
            const s_t test_grad1_u0_0 = (fgref[0 * ND + 0][q * U_NS + 0] * adj1 + fgref[0 * ND + 1][q * U_NS + 0] * adj3) / det;
            output[0][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_0 + grad_coeff0_0 * test_grad0_u0_0 + grad_coeff0_1 * test_grad1_u0_0);
            const s_t test_value_u0_1 = field_shape[0][q * U_NS + 1];
            const s_t test_grad0_u0_1 = (fgref[0 * ND + 0][q * U_NS + 1] * adj0 + fgref[0 * ND + 1][q * U_NS + 1] * adj2) / det;
            const s_t test_grad1_u0_1 = (fgref[0 * ND + 0][q * U_NS + 1] * adj1 + fgref[0 * ND + 1][q * U_NS + 1] * adj3) / det;
            output[1][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_1 + grad_coeff0_0 * test_grad0_u0_1 + grad_coeff0_1 * test_grad1_u0_1);
            const s_t test_value_u0_2 = field_shape[0][q * U_NS + 2];
            const s_t test_grad0_u0_2 = (fgref[0 * ND + 0][q * U_NS + 2] * adj0 + fgref[0 * ND + 1][q * U_NS + 2] * adj2) / det;
            const s_t test_grad1_u0_2 = (fgref[0 * ND + 0][q * U_NS + 2] * adj1 + fgref[0 * ND + 1][q * U_NS + 2] * adj3) / det;
            output[2][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_2 + grad_coeff0_0 * test_grad0_u0_2 + grad_coeff0_1 * test_grad1_u0_2);
            const s_t test_value_u0_3 = field_shape[0][q * U_NS + 3];
            const s_t test_grad0_u0_3 = (fgref[0 * ND + 0][q * U_NS + 3] * adj0 + fgref[0 * ND + 1][q * U_NS + 3] * adj2) / det;
            const s_t test_grad1_u0_3 = (fgref[0 * ND + 0][q * U_NS + 3] * adj1 + fgref[0 * ND + 1][q * U_NS + 3] * adj3) / det;
            output[3][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_3 + grad_coeff0_0 * test_grad0_u0_3 + grad_coeff0_1 * test_grad1_u0_3);
            const s_t test_value_u0_4 = field_shape[0][q * U_NS + 4];
            const s_t test_grad0_u0_4 = (fgref[0 * ND + 0][q * U_NS + 4] * adj0 + fgref[0 * ND + 1][q * U_NS + 4] * adj2) / det;
            const s_t test_grad1_u0_4 = (fgref[0 * ND + 0][q * U_NS + 4] * adj1 + fgref[0 * ND + 1][q * U_NS + 4] * adj3) / det;
            output[4][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_4 + grad_coeff0_0 * test_grad0_u0_4 + grad_coeff0_1 * test_grad1_u0_4);
            const s_t test_value_u0_5 = field_shape[0][q * U_NS + 5];
            const s_t test_grad0_u0_5 = (fgref[0 * ND + 0][q * U_NS + 5] * adj0 + fgref[0 * ND + 1][q * U_NS + 5] * adj2) / det;
            const s_t test_grad1_u0_5 = (fgref[0 * ND + 0][q * U_NS + 5] * adj1 + fgref[0 * ND + 1][q * U_NS + 5] * adj3) / det;
            output[5][lane] += q_weight[q] * det * (value_coeff0 * test_value_u0_5 + grad_coeff0_0 * test_grad0_u0_5 + grad_coeff0_1 * test_grad1_u0_5);
            const s_t test_value_u1_0 = field_shape[0][q * U_NS + 0];
            const s_t test_grad0_u1_0 = (fgref[0 * ND + 0][q * U_NS + 0] * adj0 + fgref[0 * ND + 1][q * U_NS + 0] * adj2) / det;
            const s_t test_grad1_u1_0 = (fgref[0 * ND + 0][q * U_NS + 0] * adj1 + fgref[0 * ND + 1][q * U_NS + 0] * adj3) / det;
            output[6][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_0 + grad_coeff1_0 * test_grad0_u1_0 + grad_coeff1_1 * test_grad1_u1_0);
            const s_t test_value_u1_1 = field_shape[0][q * U_NS + 1];
            const s_t test_grad0_u1_1 = (fgref[0 * ND + 0][q * U_NS + 1] * adj0 + fgref[0 * ND + 1][q * U_NS + 1] * adj2) / det;
            const s_t test_grad1_u1_1 = (fgref[0 * ND + 0][q * U_NS + 1] * adj1 + fgref[0 * ND + 1][q * U_NS + 1] * adj3) / det;
            output[7][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_1 + grad_coeff1_0 * test_grad0_u1_1 + grad_coeff1_1 * test_grad1_u1_1);
            const s_t test_value_u1_2 = field_shape[0][q * U_NS + 2];
            const s_t test_grad0_u1_2 = (fgref[0 * ND + 0][q * U_NS + 2] * adj0 + fgref[0 * ND + 1][q * U_NS + 2] * adj2) / det;
            const s_t test_grad1_u1_2 = (fgref[0 * ND + 0][q * U_NS + 2] * adj1 + fgref[0 * ND + 1][q * U_NS + 2] * adj3) / det;
            output[8][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_2 + grad_coeff1_0 * test_grad0_u1_2 + grad_coeff1_1 * test_grad1_u1_2);
            const s_t test_value_u1_3 = field_shape[0][q * U_NS + 3];
            const s_t test_grad0_u1_3 = (fgref[0 * ND + 0][q * U_NS + 3] * adj0 + fgref[0 * ND + 1][q * U_NS + 3] * adj2) / det;
            const s_t test_grad1_u1_3 = (fgref[0 * ND + 0][q * U_NS + 3] * adj1 + fgref[0 * ND + 1][q * U_NS + 3] * adj3) / det;
            output[9][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_3 + grad_coeff1_0 * test_grad0_u1_3 + grad_coeff1_1 * test_grad1_u1_3);
            const s_t test_value_u1_4 = field_shape[0][q * U_NS + 4];
            const s_t test_grad0_u1_4 = (fgref[0 * ND + 0][q * U_NS + 4] * adj0 + fgref[0 * ND + 1][q * U_NS + 4] * adj2) / det;
            const s_t test_grad1_u1_4 = (fgref[0 * ND + 0][q * U_NS + 4] * adj1 + fgref[0 * ND + 1][q * U_NS + 4] * adj3) / det;
            output[10][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_4 + grad_coeff1_0 * test_grad0_u1_4 + grad_coeff1_1 * test_grad1_u1_4);
            const s_t test_value_u1_5 = field_shape[0][q * U_NS + 5];
            const s_t test_grad0_u1_5 = (fgref[0 * ND + 0][q * U_NS + 5] * adj0 + fgref[0 * ND + 1][q * U_NS + 5] * adj2) / det;
            const s_t test_grad1_u1_5 = (fgref[0 * ND + 0][q * U_NS + 5] * adj1 + fgref[0 * ND + 1][q * U_NS + 5] * adj3) / det;
            output[11][lane] += q_weight[q] * det * (value_coeff1 * test_value_u1_5 + grad_coeff1_0 * test_grad0_u1_5 + grad_coeff1_1 * test_grad1_u1_5);
            const s_t test_value_p_0 = field_shape[1][q * P_NS + 0];
            output[12][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_0);
            const s_t test_value_p_1 = field_shape[1][q * P_NS + 1];
            output[13][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_1);
            const s_t test_value_p_2 = field_shape[1][q * P_NS + 2];
            output[14][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_2);
        }
    }
}

} // namespace codegen
} // namespace sfem

#endif
