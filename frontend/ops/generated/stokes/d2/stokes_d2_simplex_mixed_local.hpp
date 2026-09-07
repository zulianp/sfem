#ifndef STOKES_D2_SIMPLEX_MIXED_LOCAL_HPP
#define STOKES_D2_SIMPLEX_MIXED_LOCAL_HPP

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
#ifndef SFEM_GENERATED_SCALAR_T
#define SFEM_GENERATED_SCALAR_T
typedef double real_t;
typedef ptrdiff_t idx_t;
typedef ptrdiff_t count_t;
typedef double geom_t;
#endif

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void stokes_d2_simplex_mixed_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT field_shape[2],
        const scalar_t *const SFEM_RESTRICT field_grad_ref[4],
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT current[15],
        const scalar_t mu,
        scalar_t *const SFEM_RESTRICT output[15]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 15;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int U_N_SHAPE = 6;
    static constexpr int P_N_SHAPE = 3;
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t det = determinant[geometry_offset];
            const scalar_t adj0 = adjugate[0][geometry_offset];
            const scalar_t adj1 = adjugate[1][geometry_offset];
            const scalar_t adj2 = adjugate[2][geometry_offset];
            const scalar_t adj3 = adjugate[3][geometry_offset];
            scalar_t u0 = scalar_t(0);
            scalar_t u0_grad_0_ref = scalar_t(0);
            scalar_t u0_grad_1_ref = scalar_t(0);
            const scalar_t coeff_current_u0_0 = current[0][lane];
            u0 += coeff_current_u0_0 * field_shape[0][q * U_N_SHAPE + 0];
            u0_grad_0_ref += coeff_current_u0_0 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0];
            u0_grad_1_ref += coeff_current_u0_0 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0];
            const scalar_t coeff_current_u0_1 = current[1][lane];
            u0 += coeff_current_u0_1 * field_shape[0][q * U_N_SHAPE + 1];
            u0_grad_0_ref += coeff_current_u0_1 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1];
            u0_grad_1_ref += coeff_current_u0_1 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1];
            const scalar_t coeff_current_u0_2 = current[2][lane];
            u0 += coeff_current_u0_2 * field_shape[0][q * U_N_SHAPE + 2];
            u0_grad_0_ref += coeff_current_u0_2 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2];
            u0_grad_1_ref += coeff_current_u0_2 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2];
            const scalar_t coeff_current_u0_3 = current[3][lane];
            u0 += coeff_current_u0_3 * field_shape[0][q * U_N_SHAPE + 3];
            u0_grad_0_ref += coeff_current_u0_3 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3];
            u0_grad_1_ref += coeff_current_u0_3 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3];
            const scalar_t coeff_current_u0_4 = current[4][lane];
            u0 += coeff_current_u0_4 * field_shape[0][q * U_N_SHAPE + 4];
            u0_grad_0_ref += coeff_current_u0_4 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4];
            u0_grad_1_ref += coeff_current_u0_4 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4];
            const scalar_t coeff_current_u0_5 = current[5][lane];
            u0 += coeff_current_u0_5 * field_shape[0][q * U_N_SHAPE + 5];
            u0_grad_0_ref += coeff_current_u0_5 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5];
            u0_grad_1_ref += coeff_current_u0_5 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
            scalar_t u1 = scalar_t(0);
            scalar_t u1_grad_0_ref = scalar_t(0);
            scalar_t u1_grad_1_ref = scalar_t(0);
            const scalar_t coeff_current_u1_0 = current[6][lane];
            u1 += coeff_current_u1_0 * field_shape[0][q * U_N_SHAPE + 0];
            u1_grad_0_ref += coeff_current_u1_0 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0];
            u1_grad_1_ref += coeff_current_u1_0 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0];
            const scalar_t coeff_current_u1_1 = current[7][lane];
            u1 += coeff_current_u1_1 * field_shape[0][q * U_N_SHAPE + 1];
            u1_grad_0_ref += coeff_current_u1_1 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1];
            u1_grad_1_ref += coeff_current_u1_1 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1];
            const scalar_t coeff_current_u1_2 = current[8][lane];
            u1 += coeff_current_u1_2 * field_shape[0][q * U_N_SHAPE + 2];
            u1_grad_0_ref += coeff_current_u1_2 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2];
            u1_grad_1_ref += coeff_current_u1_2 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2];
            const scalar_t coeff_current_u1_3 = current[9][lane];
            u1 += coeff_current_u1_3 * field_shape[0][q * U_N_SHAPE + 3];
            u1_grad_0_ref += coeff_current_u1_3 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3];
            u1_grad_1_ref += coeff_current_u1_3 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3];
            const scalar_t coeff_current_u1_4 = current[10][lane];
            u1 += coeff_current_u1_4 * field_shape[0][q * U_N_SHAPE + 4];
            u1_grad_0_ref += coeff_current_u1_4 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4];
            u1_grad_1_ref += coeff_current_u1_4 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4];
            const scalar_t coeff_current_u1_5 = current[11][lane];
            u1 += coeff_current_u1_5 * field_shape[0][q * U_N_SHAPE + 5];
            u1_grad_0_ref += coeff_current_u1_5 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5];
            u1_grad_1_ref += coeff_current_u1_5 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
            scalar_t p = scalar_t(0);
            scalar_t p_grad_0_ref = scalar_t(0);
            scalar_t p_grad_1_ref = scalar_t(0);
            const scalar_t coeff_current_p_0 = current[12][lane];
            p += coeff_current_p_0 * field_shape[1][q * P_N_SHAPE + 0];
            p_grad_0_ref += coeff_current_p_0 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 0];
            p_grad_1_ref += coeff_current_p_0 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 0];
            const scalar_t coeff_current_p_1 = current[13][lane];
            p += coeff_current_p_1 * field_shape[1][q * P_N_SHAPE + 1];
            p_grad_0_ref += coeff_current_p_1 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 1];
            p_grad_1_ref += coeff_current_p_1 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 1];
            const scalar_t coeff_current_p_2 = current[14][lane];
            p += coeff_current_p_2 * field_shape[1][q * P_N_SHAPE + 2];
            p_grad_0_ref += coeff_current_p_2 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 2];
            p_grad_1_ref += coeff_current_p_2 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 2];
            const scalar_t p_grad_0 = (p_grad_0_ref * adj0 + p_grad_1_ref * adj2) / det;
            const scalar_t p_grad_1 = (p_grad_0_ref * adj1 + p_grad_1_ref * adj3) / det;
            const scalar_t residual_tmp0 = -p;
            const scalar_t residual_tmp1 = scalar_t(2)*mu;
            const scalar_t residual_tmp2 = mu*(u0_grad_1 + u1_grad_0);
            const scalar_t grad_coeff0_0 = residual_tmp0 + residual_tmp1*u0_grad_0;
            const scalar_t grad_coeff0_1 = residual_tmp2;
            const scalar_t grad_coeff1_0 = residual_tmp2;
            const scalar_t grad_coeff1_1 = residual_tmp0 + residual_tmp1*u1_grad_1;
            const scalar_t value_coeff2 = u0_grad_0 + u1_grad_1;
            const scalar_t test_grad0_u0_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj2) / det;
            const scalar_t test_grad1_u0_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj3) / det;
            output[0][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_0 + grad_coeff0_1 * test_grad1_u0_0);
            const scalar_t test_grad0_u0_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj2) / det;
            const scalar_t test_grad1_u0_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj3) / det;
            output[1][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_1 + grad_coeff0_1 * test_grad1_u0_1);
            const scalar_t test_grad0_u0_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj2) / det;
            const scalar_t test_grad1_u0_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj3) / det;
            output[2][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_2 + grad_coeff0_1 * test_grad1_u0_2);
            const scalar_t test_grad0_u0_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj2) / det;
            const scalar_t test_grad1_u0_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj3) / det;
            output[3][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_3 + grad_coeff0_1 * test_grad1_u0_3);
            const scalar_t test_grad0_u0_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj2) / det;
            const scalar_t test_grad1_u0_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj3) / det;
            output[4][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_4 + grad_coeff0_1 * test_grad1_u0_4);
            const scalar_t test_grad0_u0_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj2) / det;
            const scalar_t test_grad1_u0_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj3) / det;
            output[5][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_5 + grad_coeff0_1 * test_grad1_u0_5);
            const scalar_t test_grad0_u1_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj2) / det;
            const scalar_t test_grad1_u1_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj3) / det;
            output[6][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_0 + grad_coeff1_1 * test_grad1_u1_0);
            const scalar_t test_grad0_u1_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj2) / det;
            const scalar_t test_grad1_u1_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj3) / det;
            output[7][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_1 + grad_coeff1_1 * test_grad1_u1_1);
            const scalar_t test_grad0_u1_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj2) / det;
            const scalar_t test_grad1_u1_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj3) / det;
            output[8][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_2 + grad_coeff1_1 * test_grad1_u1_2);
            const scalar_t test_grad0_u1_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj2) / det;
            const scalar_t test_grad1_u1_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj3) / det;
            output[9][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_3 + grad_coeff1_1 * test_grad1_u1_3);
            const scalar_t test_grad0_u1_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj2) / det;
            const scalar_t test_grad1_u1_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj3) / det;
            output[10][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_4 + grad_coeff1_1 * test_grad1_u1_4);
            const scalar_t test_grad0_u1_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj2) / det;
            const scalar_t test_grad1_u1_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj3) / det;
            output[11][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_5 + grad_coeff1_1 * test_grad1_u1_5);
            const scalar_t test_value_p_0 = field_shape[1][q * P_N_SHAPE + 0];
            output[12][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_0);
            const scalar_t test_value_p_1 = field_shape[1][q * P_N_SHAPE + 1];
            output[13][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_1);
            const scalar_t test_value_p_2 = field_shape[1][q * P_N_SHAPE + 2];
            output[14][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_2);
        }
    }
}

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void stokes_d2_simplex_mixed_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT field_shape[2],
        const scalar_t *const SFEM_RESTRICT field_grad_ref[4],
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t current[15][VECTOR_SIZE],
        const scalar_t mu,
        scalar_t output[15][VECTOR_SIZE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 15;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int U_N_SHAPE = 6;
    static constexpr int P_N_SHAPE = 3;
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t det = determinant[geometry_offset];
            const scalar_t adj0 = adjugate[0][geometry_offset];
            const scalar_t adj1 = adjugate[1][geometry_offset];
            const scalar_t adj2 = adjugate[2][geometry_offset];
            const scalar_t adj3 = adjugate[3][geometry_offset];
            scalar_t u0 = scalar_t(0);
            scalar_t u0_grad_0_ref = scalar_t(0);
            scalar_t u0_grad_1_ref = scalar_t(0);
            const scalar_t coeff_current_u0_0 = current[0][lane];
            u0 += coeff_current_u0_0 * field_shape[0][q * U_N_SHAPE + 0];
            u0_grad_0_ref += coeff_current_u0_0 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0];
            u0_grad_1_ref += coeff_current_u0_0 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0];
            const scalar_t coeff_current_u0_1 = current[1][lane];
            u0 += coeff_current_u0_1 * field_shape[0][q * U_N_SHAPE + 1];
            u0_grad_0_ref += coeff_current_u0_1 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1];
            u0_grad_1_ref += coeff_current_u0_1 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1];
            const scalar_t coeff_current_u0_2 = current[2][lane];
            u0 += coeff_current_u0_2 * field_shape[0][q * U_N_SHAPE + 2];
            u0_grad_0_ref += coeff_current_u0_2 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2];
            u0_grad_1_ref += coeff_current_u0_2 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2];
            const scalar_t coeff_current_u0_3 = current[3][lane];
            u0 += coeff_current_u0_3 * field_shape[0][q * U_N_SHAPE + 3];
            u0_grad_0_ref += coeff_current_u0_3 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3];
            u0_grad_1_ref += coeff_current_u0_3 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3];
            const scalar_t coeff_current_u0_4 = current[4][lane];
            u0 += coeff_current_u0_4 * field_shape[0][q * U_N_SHAPE + 4];
            u0_grad_0_ref += coeff_current_u0_4 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4];
            u0_grad_1_ref += coeff_current_u0_4 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4];
            const scalar_t coeff_current_u0_5 = current[5][lane];
            u0 += coeff_current_u0_5 * field_shape[0][q * U_N_SHAPE + 5];
            u0_grad_0_ref += coeff_current_u0_5 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5];
            u0_grad_1_ref += coeff_current_u0_5 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
            scalar_t u1 = scalar_t(0);
            scalar_t u1_grad_0_ref = scalar_t(0);
            scalar_t u1_grad_1_ref = scalar_t(0);
            const scalar_t coeff_current_u1_0 = current[6][lane];
            u1 += coeff_current_u1_0 * field_shape[0][q * U_N_SHAPE + 0];
            u1_grad_0_ref += coeff_current_u1_0 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0];
            u1_grad_1_ref += coeff_current_u1_0 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0];
            const scalar_t coeff_current_u1_1 = current[7][lane];
            u1 += coeff_current_u1_1 * field_shape[0][q * U_N_SHAPE + 1];
            u1_grad_0_ref += coeff_current_u1_1 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1];
            u1_grad_1_ref += coeff_current_u1_1 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1];
            const scalar_t coeff_current_u1_2 = current[8][lane];
            u1 += coeff_current_u1_2 * field_shape[0][q * U_N_SHAPE + 2];
            u1_grad_0_ref += coeff_current_u1_2 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2];
            u1_grad_1_ref += coeff_current_u1_2 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2];
            const scalar_t coeff_current_u1_3 = current[9][lane];
            u1 += coeff_current_u1_3 * field_shape[0][q * U_N_SHAPE + 3];
            u1_grad_0_ref += coeff_current_u1_3 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3];
            u1_grad_1_ref += coeff_current_u1_3 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3];
            const scalar_t coeff_current_u1_4 = current[10][lane];
            u1 += coeff_current_u1_4 * field_shape[0][q * U_N_SHAPE + 4];
            u1_grad_0_ref += coeff_current_u1_4 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4];
            u1_grad_1_ref += coeff_current_u1_4 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4];
            const scalar_t coeff_current_u1_5 = current[11][lane];
            u1 += coeff_current_u1_5 * field_shape[0][q * U_N_SHAPE + 5];
            u1_grad_0_ref += coeff_current_u1_5 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5];
            u1_grad_1_ref += coeff_current_u1_5 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
            scalar_t p = scalar_t(0);
            scalar_t p_grad_0_ref = scalar_t(0);
            scalar_t p_grad_1_ref = scalar_t(0);
            const scalar_t coeff_current_p_0 = current[12][lane];
            p += coeff_current_p_0 * field_shape[1][q * P_N_SHAPE + 0];
            p_grad_0_ref += coeff_current_p_0 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 0];
            p_grad_1_ref += coeff_current_p_0 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 0];
            const scalar_t coeff_current_p_1 = current[13][lane];
            p += coeff_current_p_1 * field_shape[1][q * P_N_SHAPE + 1];
            p_grad_0_ref += coeff_current_p_1 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 1];
            p_grad_1_ref += coeff_current_p_1 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 1];
            const scalar_t coeff_current_p_2 = current[14][lane];
            p += coeff_current_p_2 * field_shape[1][q * P_N_SHAPE + 2];
            p_grad_0_ref += coeff_current_p_2 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 2];
            p_grad_1_ref += coeff_current_p_2 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 2];
            const scalar_t p_grad_0 = (p_grad_0_ref * adj0 + p_grad_1_ref * adj2) / det;
            const scalar_t p_grad_1 = (p_grad_0_ref * adj1 + p_grad_1_ref * adj3) / det;
            const scalar_t residual_tmp0 = -p;
            const scalar_t residual_tmp1 = scalar_t(2)*mu;
            const scalar_t residual_tmp2 = mu*(u0_grad_1 + u1_grad_0);
            const scalar_t grad_coeff0_0 = residual_tmp0 + residual_tmp1*u0_grad_0;
            const scalar_t grad_coeff0_1 = residual_tmp2;
            const scalar_t grad_coeff1_0 = residual_tmp2;
            const scalar_t grad_coeff1_1 = residual_tmp0 + residual_tmp1*u1_grad_1;
            const scalar_t value_coeff2 = u0_grad_0 + u1_grad_1;
            const scalar_t test_grad0_u0_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj2) / det;
            const scalar_t test_grad1_u0_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj3) / det;
            output[0][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_0 + grad_coeff0_1 * test_grad1_u0_0);
            const scalar_t test_grad0_u0_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj2) / det;
            const scalar_t test_grad1_u0_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj3) / det;
            output[1][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_1 + grad_coeff0_1 * test_grad1_u0_1);
            const scalar_t test_grad0_u0_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj2) / det;
            const scalar_t test_grad1_u0_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj3) / det;
            output[2][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_2 + grad_coeff0_1 * test_grad1_u0_2);
            const scalar_t test_grad0_u0_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj2) / det;
            const scalar_t test_grad1_u0_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj3) / det;
            output[3][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_3 + grad_coeff0_1 * test_grad1_u0_3);
            const scalar_t test_grad0_u0_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj2) / det;
            const scalar_t test_grad1_u0_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj3) / det;
            output[4][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_4 + grad_coeff0_1 * test_grad1_u0_4);
            const scalar_t test_grad0_u0_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj2) / det;
            const scalar_t test_grad1_u0_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj3) / det;
            output[5][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_5 + grad_coeff0_1 * test_grad1_u0_5);
            const scalar_t test_grad0_u1_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj2) / det;
            const scalar_t test_grad1_u1_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj3) / det;
            output[6][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_0 + grad_coeff1_1 * test_grad1_u1_0);
            const scalar_t test_grad0_u1_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj2) / det;
            const scalar_t test_grad1_u1_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj3) / det;
            output[7][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_1 + grad_coeff1_1 * test_grad1_u1_1);
            const scalar_t test_grad0_u1_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj2) / det;
            const scalar_t test_grad1_u1_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj3) / det;
            output[8][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_2 + grad_coeff1_1 * test_grad1_u1_2);
            const scalar_t test_grad0_u1_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj2) / det;
            const scalar_t test_grad1_u1_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj3) / det;
            output[9][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_3 + grad_coeff1_1 * test_grad1_u1_3);
            const scalar_t test_grad0_u1_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj2) / det;
            const scalar_t test_grad1_u1_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj3) / det;
            output[10][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_4 + grad_coeff1_1 * test_grad1_u1_4);
            const scalar_t test_grad0_u1_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj2) / det;
            const scalar_t test_grad1_u1_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj3) / det;
            output[11][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_5 + grad_coeff1_1 * test_grad1_u1_5);
            const scalar_t test_value_p_0 = field_shape[1][q * P_N_SHAPE + 0];
            output[12][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_0);
            const scalar_t test_value_p_1 = field_shape[1][q * P_N_SHAPE + 1];
            output[13][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_1);
            const scalar_t test_value_p_2 = field_shape[1][q * P_N_SHAPE + 2];
            output[14][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_2);
        }
    }
}

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void stokes_d2_simplex_mixed_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT field_shape[2],
        const scalar_t *const SFEM_RESTRICT field_grad_ref[4],
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT direction[15],
        const scalar_t mu,
        scalar_t *const SFEM_RESTRICT output[15]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 15;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int U_N_SHAPE = 6;
    static constexpr int P_N_SHAPE = 3;
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t det = determinant[geometry_offset];
            const scalar_t adj0 = adjugate[0][geometry_offset];
            const scalar_t adj1 = adjugate[1][geometry_offset];
            const scalar_t adj2 = adjugate[2][geometry_offset];
            const scalar_t adj3 = adjugate[3][geometry_offset];
            scalar_t u0_direction = scalar_t(0);
            scalar_t u0_direction_grad_0_ref = scalar_t(0);
            scalar_t u0_direction_grad_1_ref = scalar_t(0);
            const scalar_t coeff_direction_u0_0 = direction[0][lane];
            u0_direction += coeff_direction_u0_0 * field_shape[0][q * U_N_SHAPE + 0];
            u0_direction_grad_0_ref += coeff_direction_u0_0 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0];
            u0_direction_grad_1_ref += coeff_direction_u0_0 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0];
            const scalar_t coeff_direction_u0_1 = direction[1][lane];
            u0_direction += coeff_direction_u0_1 * field_shape[0][q * U_N_SHAPE + 1];
            u0_direction_grad_0_ref += coeff_direction_u0_1 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1];
            u0_direction_grad_1_ref += coeff_direction_u0_1 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1];
            const scalar_t coeff_direction_u0_2 = direction[2][lane];
            u0_direction += coeff_direction_u0_2 * field_shape[0][q * U_N_SHAPE + 2];
            u0_direction_grad_0_ref += coeff_direction_u0_2 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2];
            u0_direction_grad_1_ref += coeff_direction_u0_2 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2];
            const scalar_t coeff_direction_u0_3 = direction[3][lane];
            u0_direction += coeff_direction_u0_3 * field_shape[0][q * U_N_SHAPE + 3];
            u0_direction_grad_0_ref += coeff_direction_u0_3 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3];
            u0_direction_grad_1_ref += coeff_direction_u0_3 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3];
            const scalar_t coeff_direction_u0_4 = direction[4][lane];
            u0_direction += coeff_direction_u0_4 * field_shape[0][q * U_N_SHAPE + 4];
            u0_direction_grad_0_ref += coeff_direction_u0_4 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4];
            u0_direction_grad_1_ref += coeff_direction_u0_4 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4];
            const scalar_t coeff_direction_u0_5 = direction[5][lane];
            u0_direction += coeff_direction_u0_5 * field_shape[0][q * U_N_SHAPE + 5];
            u0_direction_grad_0_ref += coeff_direction_u0_5 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5];
            u0_direction_grad_1_ref += coeff_direction_u0_5 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5];
            const scalar_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
            const scalar_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
            scalar_t u1_direction = scalar_t(0);
            scalar_t u1_direction_grad_0_ref = scalar_t(0);
            scalar_t u1_direction_grad_1_ref = scalar_t(0);
            const scalar_t coeff_direction_u1_0 = direction[6][lane];
            u1_direction += coeff_direction_u1_0 * field_shape[0][q * U_N_SHAPE + 0];
            u1_direction_grad_0_ref += coeff_direction_u1_0 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0];
            u1_direction_grad_1_ref += coeff_direction_u1_0 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0];
            const scalar_t coeff_direction_u1_1 = direction[7][lane];
            u1_direction += coeff_direction_u1_1 * field_shape[0][q * U_N_SHAPE + 1];
            u1_direction_grad_0_ref += coeff_direction_u1_1 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1];
            u1_direction_grad_1_ref += coeff_direction_u1_1 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1];
            const scalar_t coeff_direction_u1_2 = direction[8][lane];
            u1_direction += coeff_direction_u1_2 * field_shape[0][q * U_N_SHAPE + 2];
            u1_direction_grad_0_ref += coeff_direction_u1_2 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2];
            u1_direction_grad_1_ref += coeff_direction_u1_2 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2];
            const scalar_t coeff_direction_u1_3 = direction[9][lane];
            u1_direction += coeff_direction_u1_3 * field_shape[0][q * U_N_SHAPE + 3];
            u1_direction_grad_0_ref += coeff_direction_u1_3 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3];
            u1_direction_grad_1_ref += coeff_direction_u1_3 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3];
            const scalar_t coeff_direction_u1_4 = direction[10][lane];
            u1_direction += coeff_direction_u1_4 * field_shape[0][q * U_N_SHAPE + 4];
            u1_direction_grad_0_ref += coeff_direction_u1_4 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4];
            u1_direction_grad_1_ref += coeff_direction_u1_4 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4];
            const scalar_t coeff_direction_u1_5 = direction[11][lane];
            u1_direction += coeff_direction_u1_5 * field_shape[0][q * U_N_SHAPE + 5];
            u1_direction_grad_0_ref += coeff_direction_u1_5 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5];
            u1_direction_grad_1_ref += coeff_direction_u1_5 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5];
            const scalar_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj2) / det;
            const scalar_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj3) / det;
            scalar_t p_direction = scalar_t(0);
            scalar_t p_direction_grad_0_ref = scalar_t(0);
            scalar_t p_direction_grad_1_ref = scalar_t(0);
            const scalar_t coeff_direction_p_0 = direction[12][lane];
            p_direction += coeff_direction_p_0 * field_shape[1][q * P_N_SHAPE + 0];
            p_direction_grad_0_ref += coeff_direction_p_0 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 0];
            p_direction_grad_1_ref += coeff_direction_p_0 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 0];
            const scalar_t coeff_direction_p_1 = direction[13][lane];
            p_direction += coeff_direction_p_1 * field_shape[1][q * P_N_SHAPE + 1];
            p_direction_grad_0_ref += coeff_direction_p_1 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 1];
            p_direction_grad_1_ref += coeff_direction_p_1 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 1];
            const scalar_t coeff_direction_p_2 = direction[14][lane];
            p_direction += coeff_direction_p_2 * field_shape[1][q * P_N_SHAPE + 2];
            p_direction_grad_0_ref += coeff_direction_p_2 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 2];
            p_direction_grad_1_ref += coeff_direction_p_2 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 2];
            const scalar_t p_direction_grad_0 = (p_direction_grad_0_ref * adj0 + p_direction_grad_1_ref * adj2) / det;
            const scalar_t p_direction_grad_1 = (p_direction_grad_0_ref * adj1 + p_direction_grad_1_ref * adj3) / det;
            const scalar_t residual_tmp0 = -p_direction;
            const scalar_t residual_tmp1 = scalar_t(2)*mu;
            const scalar_t residual_tmp2 = mu*u0_direction_grad_1 + mu*u1_direction_grad_0;
            const scalar_t grad_coeff0_0 = residual_tmp0 + residual_tmp1*u0_direction_grad_0;
            const scalar_t grad_coeff0_1 = residual_tmp2;
            const scalar_t grad_coeff1_0 = residual_tmp2;
            const scalar_t grad_coeff1_1 = residual_tmp0 + residual_tmp1*u1_direction_grad_1;
            const scalar_t value_coeff2 = u0_direction_grad_0 + u1_direction_grad_1;
            const scalar_t test_grad0_u0_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj2) / det;
            const scalar_t test_grad1_u0_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj3) / det;
            output[0][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_0 + grad_coeff0_1 * test_grad1_u0_0);
            const scalar_t test_grad0_u0_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj2) / det;
            const scalar_t test_grad1_u0_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj3) / det;
            output[1][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_1 + grad_coeff0_1 * test_grad1_u0_1);
            const scalar_t test_grad0_u0_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj2) / det;
            const scalar_t test_grad1_u0_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj3) / det;
            output[2][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_2 + grad_coeff0_1 * test_grad1_u0_2);
            const scalar_t test_grad0_u0_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj2) / det;
            const scalar_t test_grad1_u0_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj3) / det;
            output[3][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_3 + grad_coeff0_1 * test_grad1_u0_3);
            const scalar_t test_grad0_u0_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj2) / det;
            const scalar_t test_grad1_u0_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj3) / det;
            output[4][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_4 + grad_coeff0_1 * test_grad1_u0_4);
            const scalar_t test_grad0_u0_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj2) / det;
            const scalar_t test_grad1_u0_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj3) / det;
            output[5][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_5 + grad_coeff0_1 * test_grad1_u0_5);
            const scalar_t test_grad0_u1_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj2) / det;
            const scalar_t test_grad1_u1_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj3) / det;
            output[6][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_0 + grad_coeff1_1 * test_grad1_u1_0);
            const scalar_t test_grad0_u1_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj2) / det;
            const scalar_t test_grad1_u1_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj3) / det;
            output[7][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_1 + grad_coeff1_1 * test_grad1_u1_1);
            const scalar_t test_grad0_u1_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj2) / det;
            const scalar_t test_grad1_u1_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj3) / det;
            output[8][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_2 + grad_coeff1_1 * test_grad1_u1_2);
            const scalar_t test_grad0_u1_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj2) / det;
            const scalar_t test_grad1_u1_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj3) / det;
            output[9][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_3 + grad_coeff1_1 * test_grad1_u1_3);
            const scalar_t test_grad0_u1_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj2) / det;
            const scalar_t test_grad1_u1_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj3) / det;
            output[10][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_4 + grad_coeff1_1 * test_grad1_u1_4);
            const scalar_t test_grad0_u1_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj2) / det;
            const scalar_t test_grad1_u1_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj3) / det;
            output[11][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_5 + grad_coeff1_1 * test_grad1_u1_5);
            const scalar_t test_value_p_0 = field_shape[1][q * P_N_SHAPE + 0];
            output[12][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_0);
            const scalar_t test_value_p_1 = field_shape[1][q * P_N_SHAPE + 1];
            output[13][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_1);
            const scalar_t test_value_p_2 = field_shape[1][q * P_N_SHAPE + 2];
            output[14][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_2);
        }
    }
}

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void stokes_d2_simplex_mixed_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT field_shape[2],
        const scalar_t *const SFEM_RESTRICT field_grad_ref[4],
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t direction[15][VECTOR_SIZE],
        const scalar_t mu,
        scalar_t output[15][VECTOR_SIZE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 15;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int U_N_SHAPE = 6;
    static constexpr int P_N_SHAPE = 3;
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t det = determinant[geometry_offset];
            const scalar_t adj0 = adjugate[0][geometry_offset];
            const scalar_t adj1 = adjugate[1][geometry_offset];
            const scalar_t adj2 = adjugate[2][geometry_offset];
            const scalar_t adj3 = adjugate[3][geometry_offset];
            scalar_t u0_direction = scalar_t(0);
            scalar_t u0_direction_grad_0_ref = scalar_t(0);
            scalar_t u0_direction_grad_1_ref = scalar_t(0);
            const scalar_t coeff_direction_u0_0 = direction[0][lane];
            u0_direction += coeff_direction_u0_0 * field_shape[0][q * U_N_SHAPE + 0];
            u0_direction_grad_0_ref += coeff_direction_u0_0 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0];
            u0_direction_grad_1_ref += coeff_direction_u0_0 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0];
            const scalar_t coeff_direction_u0_1 = direction[1][lane];
            u0_direction += coeff_direction_u0_1 * field_shape[0][q * U_N_SHAPE + 1];
            u0_direction_grad_0_ref += coeff_direction_u0_1 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1];
            u0_direction_grad_1_ref += coeff_direction_u0_1 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1];
            const scalar_t coeff_direction_u0_2 = direction[2][lane];
            u0_direction += coeff_direction_u0_2 * field_shape[0][q * U_N_SHAPE + 2];
            u0_direction_grad_0_ref += coeff_direction_u0_2 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2];
            u0_direction_grad_1_ref += coeff_direction_u0_2 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2];
            const scalar_t coeff_direction_u0_3 = direction[3][lane];
            u0_direction += coeff_direction_u0_3 * field_shape[0][q * U_N_SHAPE + 3];
            u0_direction_grad_0_ref += coeff_direction_u0_3 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3];
            u0_direction_grad_1_ref += coeff_direction_u0_3 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3];
            const scalar_t coeff_direction_u0_4 = direction[4][lane];
            u0_direction += coeff_direction_u0_4 * field_shape[0][q * U_N_SHAPE + 4];
            u0_direction_grad_0_ref += coeff_direction_u0_4 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4];
            u0_direction_grad_1_ref += coeff_direction_u0_4 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4];
            const scalar_t coeff_direction_u0_5 = direction[5][lane];
            u0_direction += coeff_direction_u0_5 * field_shape[0][q * U_N_SHAPE + 5];
            u0_direction_grad_0_ref += coeff_direction_u0_5 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5];
            u0_direction_grad_1_ref += coeff_direction_u0_5 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5];
            const scalar_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
            const scalar_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
            scalar_t u1_direction = scalar_t(0);
            scalar_t u1_direction_grad_0_ref = scalar_t(0);
            scalar_t u1_direction_grad_1_ref = scalar_t(0);
            const scalar_t coeff_direction_u1_0 = direction[6][lane];
            u1_direction += coeff_direction_u1_0 * field_shape[0][q * U_N_SHAPE + 0];
            u1_direction_grad_0_ref += coeff_direction_u1_0 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0];
            u1_direction_grad_1_ref += coeff_direction_u1_0 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0];
            const scalar_t coeff_direction_u1_1 = direction[7][lane];
            u1_direction += coeff_direction_u1_1 * field_shape[0][q * U_N_SHAPE + 1];
            u1_direction_grad_0_ref += coeff_direction_u1_1 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1];
            u1_direction_grad_1_ref += coeff_direction_u1_1 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1];
            const scalar_t coeff_direction_u1_2 = direction[8][lane];
            u1_direction += coeff_direction_u1_2 * field_shape[0][q * U_N_SHAPE + 2];
            u1_direction_grad_0_ref += coeff_direction_u1_2 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2];
            u1_direction_grad_1_ref += coeff_direction_u1_2 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2];
            const scalar_t coeff_direction_u1_3 = direction[9][lane];
            u1_direction += coeff_direction_u1_3 * field_shape[0][q * U_N_SHAPE + 3];
            u1_direction_grad_0_ref += coeff_direction_u1_3 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3];
            u1_direction_grad_1_ref += coeff_direction_u1_3 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3];
            const scalar_t coeff_direction_u1_4 = direction[10][lane];
            u1_direction += coeff_direction_u1_4 * field_shape[0][q * U_N_SHAPE + 4];
            u1_direction_grad_0_ref += coeff_direction_u1_4 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4];
            u1_direction_grad_1_ref += coeff_direction_u1_4 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4];
            const scalar_t coeff_direction_u1_5 = direction[11][lane];
            u1_direction += coeff_direction_u1_5 * field_shape[0][q * U_N_SHAPE + 5];
            u1_direction_grad_0_ref += coeff_direction_u1_5 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5];
            u1_direction_grad_1_ref += coeff_direction_u1_5 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5];
            const scalar_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj2) / det;
            const scalar_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj3) / det;
            scalar_t p_direction = scalar_t(0);
            scalar_t p_direction_grad_0_ref = scalar_t(0);
            scalar_t p_direction_grad_1_ref = scalar_t(0);
            const scalar_t coeff_direction_p_0 = direction[12][lane];
            p_direction += coeff_direction_p_0 * field_shape[1][q * P_N_SHAPE + 0];
            p_direction_grad_0_ref += coeff_direction_p_0 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 0];
            p_direction_grad_1_ref += coeff_direction_p_0 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 0];
            const scalar_t coeff_direction_p_1 = direction[13][lane];
            p_direction += coeff_direction_p_1 * field_shape[1][q * P_N_SHAPE + 1];
            p_direction_grad_0_ref += coeff_direction_p_1 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 1];
            p_direction_grad_1_ref += coeff_direction_p_1 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 1];
            const scalar_t coeff_direction_p_2 = direction[14][lane];
            p_direction += coeff_direction_p_2 * field_shape[1][q * P_N_SHAPE + 2];
            p_direction_grad_0_ref += coeff_direction_p_2 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 2];
            p_direction_grad_1_ref += coeff_direction_p_2 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 2];
            const scalar_t p_direction_grad_0 = (p_direction_grad_0_ref * adj0 + p_direction_grad_1_ref * adj2) / det;
            const scalar_t p_direction_grad_1 = (p_direction_grad_0_ref * adj1 + p_direction_grad_1_ref * adj3) / det;
            const scalar_t residual_tmp0 = -p_direction;
            const scalar_t residual_tmp1 = scalar_t(2)*mu;
            const scalar_t residual_tmp2 = mu*u0_direction_grad_1 + mu*u1_direction_grad_0;
            const scalar_t grad_coeff0_0 = residual_tmp0 + residual_tmp1*u0_direction_grad_0;
            const scalar_t grad_coeff0_1 = residual_tmp2;
            const scalar_t grad_coeff1_0 = residual_tmp2;
            const scalar_t grad_coeff1_1 = residual_tmp0 + residual_tmp1*u1_direction_grad_1;
            const scalar_t value_coeff2 = u0_direction_grad_0 + u1_direction_grad_1;
            const scalar_t test_grad0_u0_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj2) / det;
            const scalar_t test_grad1_u0_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj3) / det;
            output[0][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_0 + grad_coeff0_1 * test_grad1_u0_0);
            const scalar_t test_grad0_u0_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj2) / det;
            const scalar_t test_grad1_u0_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj3) / det;
            output[1][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_1 + grad_coeff0_1 * test_grad1_u0_1);
            const scalar_t test_grad0_u0_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj2) / det;
            const scalar_t test_grad1_u0_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj3) / det;
            output[2][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_2 + grad_coeff0_1 * test_grad1_u0_2);
            const scalar_t test_grad0_u0_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj2) / det;
            const scalar_t test_grad1_u0_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj3) / det;
            output[3][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_3 + grad_coeff0_1 * test_grad1_u0_3);
            const scalar_t test_grad0_u0_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj2) / det;
            const scalar_t test_grad1_u0_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj3) / det;
            output[4][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_4 + grad_coeff0_1 * test_grad1_u0_4);
            const scalar_t test_grad0_u0_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj2) / det;
            const scalar_t test_grad1_u0_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj3) / det;
            output[5][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_5 + grad_coeff0_1 * test_grad1_u0_5);
            const scalar_t test_grad0_u1_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj2) / det;
            const scalar_t test_grad1_u1_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj3) / det;
            output[6][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_0 + grad_coeff1_1 * test_grad1_u1_0);
            const scalar_t test_grad0_u1_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj2) / det;
            const scalar_t test_grad1_u1_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj3) / det;
            output[7][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_1 + grad_coeff1_1 * test_grad1_u1_1);
            const scalar_t test_grad0_u1_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj2) / det;
            const scalar_t test_grad1_u1_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj3) / det;
            output[8][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_2 + grad_coeff1_1 * test_grad1_u1_2);
            const scalar_t test_grad0_u1_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj2) / det;
            const scalar_t test_grad1_u1_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj3) / det;
            output[9][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_3 + grad_coeff1_1 * test_grad1_u1_3);
            const scalar_t test_grad0_u1_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj2) / det;
            const scalar_t test_grad1_u1_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj3) / det;
            output[10][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_4 + grad_coeff1_1 * test_grad1_u1_4);
            const scalar_t test_grad0_u1_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj2) / det;
            const scalar_t test_grad1_u1_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj3) / det;
            output[11][lane] += q_weight[q] * det * (grad_coeff1_0 * test_grad0_u1_5 + grad_coeff1_1 * test_grad1_u1_5);
            const scalar_t test_value_p_0 = field_shape[1][q * P_N_SHAPE + 0];
            output[12][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_0);
            const scalar_t test_value_p_1 = field_shape[1][q * P_N_SHAPE + 1];
            output[13][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_1);
            const scalar_t test_value_p_2 = field_shape[1][q * P_N_SHAPE + 2];
            output[14][lane] += q_weight[q] * det * (value_coeff2 * test_value_p_2);
        }
    }
}

} // namespace codegen
} // namespace sfem

#endif
