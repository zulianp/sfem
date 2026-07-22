#ifndef PORO_HYPERELASTICITY_PORO_FORM_2_U_P_D3_SIMPLEX_MIXED_LOCAL_HPP
#define PORO_HYPERELASTICITY_PORO_FORM_2_U_P_D3_SIMPLEX_MIXED_LOCAL_HPP

#include <math.h>
#include <stddef.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_GENERATED_SCALAR_T
#endif
#endif
#include "../kernel_math.hpp"
#include "../tensor_product_kernels.hpp"

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
typedef double geom_t;
#endif

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void poro_hyperelasticity_poro_form_2_u_p_d3_simplex_mixed_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT field_shape[2],
        const scalar_t *const SFEM_RESTRICT q_weight,
        scalar_t *const SFEM_RESTRICT output[34]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 34;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int U_N_SHAPE = 10;
    static constexpr int P_N_SHAPE = 4;
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t det = determinant[geometry_offset];
        }
    }
}

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void poro_hyperelasticity_poro_form_2_u_p_d3_simplex_mixed_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT field_shape[2],
        const scalar_t *const SFEM_RESTRICT q_weight,
        scalar_t output[34][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 34;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int U_N_SHAPE = 10;
    static constexpr int P_N_SHAPE = 4;
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t det = determinant[geometry_offset];
        }
    }
}

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void poro_hyperelasticity_poro_form_2_u_p_d3_simplex_mixed_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT field_shape[2],
        const scalar_t *const SFEM_RESTRICT field_grad_ref[6],
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT direction[34],
        const scalar_t alpha,
        scalar_t *const SFEM_RESTRICT output[34]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 34;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int U_N_SHAPE = 10;
    static constexpr int P_N_SHAPE = 4;
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t det = determinant[geometry_offset];
            const scalar_t adj0 = adjugate[0][geometry_offset];
            const scalar_t adj1 = adjugate[1][geometry_offset];
            const scalar_t adj2 = adjugate[2][geometry_offset];
            const scalar_t adj3 = adjugate[3][geometry_offset];
            const scalar_t adj4 = adjugate[4][geometry_offset];
            const scalar_t adj5 = adjugate[5][geometry_offset];
            const scalar_t adj6 = adjugate[6][geometry_offset];
            const scalar_t adj7 = adjugate[7][geometry_offset];
            const scalar_t adj8 = adjugate[8][geometry_offset];
            scalar_t u0_direction = scalar_t(0);
            const scalar_t coeff_direction_u0_0 = direction[0][lane];
            u0_direction += coeff_direction_u0_0 * field_shape[0][q * U_N_SHAPE + 0];
            const scalar_t coeff_direction_u0_1 = direction[1][lane];
            u0_direction += coeff_direction_u0_1 * field_shape[0][q * U_N_SHAPE + 1];
            const scalar_t coeff_direction_u0_2 = direction[2][lane];
            u0_direction += coeff_direction_u0_2 * field_shape[0][q * U_N_SHAPE + 2];
            const scalar_t coeff_direction_u0_3 = direction[3][lane];
            u0_direction += coeff_direction_u0_3 * field_shape[0][q * U_N_SHAPE + 3];
            const scalar_t coeff_direction_u0_4 = direction[4][lane];
            u0_direction += coeff_direction_u0_4 * field_shape[0][q * U_N_SHAPE + 4];
            const scalar_t coeff_direction_u0_5 = direction[5][lane];
            u0_direction += coeff_direction_u0_5 * field_shape[0][q * U_N_SHAPE + 5];
            const scalar_t coeff_direction_u0_6 = direction[6][lane];
            u0_direction += coeff_direction_u0_6 * field_shape[0][q * U_N_SHAPE + 6];
            const scalar_t coeff_direction_u0_7 = direction[7][lane];
            u0_direction += coeff_direction_u0_7 * field_shape[0][q * U_N_SHAPE + 7];
            const scalar_t coeff_direction_u0_8 = direction[8][lane];
            u0_direction += coeff_direction_u0_8 * field_shape[0][q * U_N_SHAPE + 8];
            const scalar_t coeff_direction_u0_9 = direction[9][lane];
            u0_direction += coeff_direction_u0_9 * field_shape[0][q * U_N_SHAPE + 9];
            scalar_t u1_direction = scalar_t(0);
            const scalar_t coeff_direction_u1_0 = direction[10][lane];
            u1_direction += coeff_direction_u1_0 * field_shape[0][q * U_N_SHAPE + 0];
            const scalar_t coeff_direction_u1_1 = direction[11][lane];
            u1_direction += coeff_direction_u1_1 * field_shape[0][q * U_N_SHAPE + 1];
            const scalar_t coeff_direction_u1_2 = direction[12][lane];
            u1_direction += coeff_direction_u1_2 * field_shape[0][q * U_N_SHAPE + 2];
            const scalar_t coeff_direction_u1_3 = direction[13][lane];
            u1_direction += coeff_direction_u1_3 * field_shape[0][q * U_N_SHAPE + 3];
            const scalar_t coeff_direction_u1_4 = direction[14][lane];
            u1_direction += coeff_direction_u1_4 * field_shape[0][q * U_N_SHAPE + 4];
            const scalar_t coeff_direction_u1_5 = direction[15][lane];
            u1_direction += coeff_direction_u1_5 * field_shape[0][q * U_N_SHAPE + 5];
            const scalar_t coeff_direction_u1_6 = direction[16][lane];
            u1_direction += coeff_direction_u1_6 * field_shape[0][q * U_N_SHAPE + 6];
            const scalar_t coeff_direction_u1_7 = direction[17][lane];
            u1_direction += coeff_direction_u1_7 * field_shape[0][q * U_N_SHAPE + 7];
            const scalar_t coeff_direction_u1_8 = direction[18][lane];
            u1_direction += coeff_direction_u1_8 * field_shape[0][q * U_N_SHAPE + 8];
            const scalar_t coeff_direction_u1_9 = direction[19][lane];
            u1_direction += coeff_direction_u1_9 * field_shape[0][q * U_N_SHAPE + 9];
            scalar_t u2_direction = scalar_t(0);
            const scalar_t coeff_direction_u2_0 = direction[20][lane];
            u2_direction += coeff_direction_u2_0 * field_shape[0][q * U_N_SHAPE + 0];
            const scalar_t coeff_direction_u2_1 = direction[21][lane];
            u2_direction += coeff_direction_u2_1 * field_shape[0][q * U_N_SHAPE + 1];
            const scalar_t coeff_direction_u2_2 = direction[22][lane];
            u2_direction += coeff_direction_u2_2 * field_shape[0][q * U_N_SHAPE + 2];
            const scalar_t coeff_direction_u2_3 = direction[23][lane];
            u2_direction += coeff_direction_u2_3 * field_shape[0][q * U_N_SHAPE + 3];
            const scalar_t coeff_direction_u2_4 = direction[24][lane];
            u2_direction += coeff_direction_u2_4 * field_shape[0][q * U_N_SHAPE + 4];
            const scalar_t coeff_direction_u2_5 = direction[25][lane];
            u2_direction += coeff_direction_u2_5 * field_shape[0][q * U_N_SHAPE + 5];
            const scalar_t coeff_direction_u2_6 = direction[26][lane];
            u2_direction += coeff_direction_u2_6 * field_shape[0][q * U_N_SHAPE + 6];
            const scalar_t coeff_direction_u2_7 = direction[27][lane];
            u2_direction += coeff_direction_u2_7 * field_shape[0][q * U_N_SHAPE + 7];
            const scalar_t coeff_direction_u2_8 = direction[28][lane];
            u2_direction += coeff_direction_u2_8 * field_shape[0][q * U_N_SHAPE + 8];
            const scalar_t coeff_direction_u2_9 = direction[29][lane];
            u2_direction += coeff_direction_u2_9 * field_shape[0][q * U_N_SHAPE + 9];
            scalar_t p_direction = scalar_t(0);
            const scalar_t coeff_direction_p_0 = direction[30][lane];
            p_direction += coeff_direction_p_0 * field_shape[1][q * P_N_SHAPE + 0];
            const scalar_t coeff_direction_p_1 = direction[31][lane];
            p_direction += coeff_direction_p_1 * field_shape[1][q * P_N_SHAPE + 1];
            const scalar_t coeff_direction_p_2 = direction[32][lane];
            p_direction += coeff_direction_p_2 * field_shape[1][q * P_N_SHAPE + 2];
            const scalar_t coeff_direction_p_3 = direction[33][lane];
            p_direction += coeff_direction_p_3 * field_shape[1][q * P_N_SHAPE + 3];
            const scalar_t residual_tmp0 = -alpha*p_direction;
            const scalar_t grad_coeff0_0 = residual_tmp0;
            const scalar_t grad_coeff1_1 = residual_tmp0;
            const scalar_t grad_coeff2_2 = residual_tmp0;
            const scalar_t test_grad0_u0_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 0] * adj6) / det;
            output[0][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_0);
            const scalar_t test_grad0_u0_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 1] * adj6) / det;
            output[1][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_1);
            const scalar_t test_grad0_u0_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 2] * adj6) / det;
            output[2][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_2);
            const scalar_t test_grad0_u0_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 3] * adj6) / det;
            output[3][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_3);
            const scalar_t test_grad0_u0_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 4] * adj6) / det;
            output[4][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_4);
            const scalar_t test_grad0_u0_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 5] * adj6) / det;
            output[5][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_5);
            const scalar_t test_grad0_u0_6 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 6] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 6] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 6] * adj6) / det;
            output[6][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_6);
            const scalar_t test_grad0_u0_7 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 7] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 7] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 7] * adj6) / det;
            output[7][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_7);
            const scalar_t test_grad0_u0_8 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 8] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 8] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 8] * adj6) / det;
            output[8][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_8);
            const scalar_t test_grad0_u0_9 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 9] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 9] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 9] * adj6) / det;
            output[9][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_9);
            const scalar_t test_grad1_u1_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 0] * adj7) / det;
            output[10][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_0);
            const scalar_t test_grad1_u1_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 1] * adj7) / det;
            output[11][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_1);
            const scalar_t test_grad1_u1_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 2] * adj7) / det;
            output[12][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_2);
            const scalar_t test_grad1_u1_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 3] * adj7) / det;
            output[13][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_3);
            const scalar_t test_grad1_u1_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 4] * adj7) / det;
            output[14][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_4);
            const scalar_t test_grad1_u1_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 5] * adj7) / det;
            output[15][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_5);
            const scalar_t test_grad1_u1_6 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 6] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 6] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 6] * adj7) / det;
            output[16][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_6);
            const scalar_t test_grad1_u1_7 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 7] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 7] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 7] * adj7) / det;
            output[17][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_7);
            const scalar_t test_grad1_u1_8 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 8] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 8] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 8] * adj7) / det;
            output[18][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_8);
            const scalar_t test_grad1_u1_9 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 9] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 9] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 9] * adj7) / det;
            output[19][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_9);
            const scalar_t test_grad2_u2_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 0] * adj8) / det;
            output[20][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_0);
            const scalar_t test_grad2_u2_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 1] * adj8) / det;
            output[21][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_1);
            const scalar_t test_grad2_u2_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 2] * adj8) / det;
            output[22][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_2);
            const scalar_t test_grad2_u2_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 3] * adj8) / det;
            output[23][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_3);
            const scalar_t test_grad2_u2_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 4] * adj8) / det;
            output[24][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_4);
            const scalar_t test_grad2_u2_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 5] * adj8) / det;
            output[25][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_5);
            const scalar_t test_grad2_u2_6 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 6] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 6] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 6] * adj8) / det;
            output[26][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_6);
            const scalar_t test_grad2_u2_7 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 7] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 7] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 7] * adj8) / det;
            output[27][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_7);
            const scalar_t test_grad2_u2_8 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 8] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 8] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 8] * adj8) / det;
            output[28][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_8);
            const scalar_t test_grad2_u2_9 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 9] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 9] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 9] * adj8) / det;
            output[29][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_9);
        }
    }
}

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void poro_hyperelasticity_poro_form_2_u_p_d3_simplex_mixed_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT field_shape[2],
        const scalar_t *const SFEM_RESTRICT field_grad_ref[6],
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t direction[34][VECTOR_SIZE],
        const scalar_t alpha,
        scalar_t output[34][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 34;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int U_N_SHAPE = 10;
    static constexpr int P_N_SHAPE = 4;
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t det = determinant[geometry_offset];
            const scalar_t adj0 = adjugate[0][geometry_offset];
            const scalar_t adj1 = adjugate[1][geometry_offset];
            const scalar_t adj2 = adjugate[2][geometry_offset];
            const scalar_t adj3 = adjugate[3][geometry_offset];
            const scalar_t adj4 = adjugate[4][geometry_offset];
            const scalar_t adj5 = adjugate[5][geometry_offset];
            const scalar_t adj6 = adjugate[6][geometry_offset];
            const scalar_t adj7 = adjugate[7][geometry_offset];
            const scalar_t adj8 = adjugate[8][geometry_offset];
            scalar_t u0_direction = scalar_t(0);
            const scalar_t coeff_direction_u0_0 = direction[0][lane];
            u0_direction += coeff_direction_u0_0 * field_shape[0][q * U_N_SHAPE + 0];
            const scalar_t coeff_direction_u0_1 = direction[1][lane];
            u0_direction += coeff_direction_u0_1 * field_shape[0][q * U_N_SHAPE + 1];
            const scalar_t coeff_direction_u0_2 = direction[2][lane];
            u0_direction += coeff_direction_u0_2 * field_shape[0][q * U_N_SHAPE + 2];
            const scalar_t coeff_direction_u0_3 = direction[3][lane];
            u0_direction += coeff_direction_u0_3 * field_shape[0][q * U_N_SHAPE + 3];
            const scalar_t coeff_direction_u0_4 = direction[4][lane];
            u0_direction += coeff_direction_u0_4 * field_shape[0][q * U_N_SHAPE + 4];
            const scalar_t coeff_direction_u0_5 = direction[5][lane];
            u0_direction += coeff_direction_u0_5 * field_shape[0][q * U_N_SHAPE + 5];
            const scalar_t coeff_direction_u0_6 = direction[6][lane];
            u0_direction += coeff_direction_u0_6 * field_shape[0][q * U_N_SHAPE + 6];
            const scalar_t coeff_direction_u0_7 = direction[7][lane];
            u0_direction += coeff_direction_u0_7 * field_shape[0][q * U_N_SHAPE + 7];
            const scalar_t coeff_direction_u0_8 = direction[8][lane];
            u0_direction += coeff_direction_u0_8 * field_shape[0][q * U_N_SHAPE + 8];
            const scalar_t coeff_direction_u0_9 = direction[9][lane];
            u0_direction += coeff_direction_u0_9 * field_shape[0][q * U_N_SHAPE + 9];
            scalar_t u1_direction = scalar_t(0);
            const scalar_t coeff_direction_u1_0 = direction[10][lane];
            u1_direction += coeff_direction_u1_0 * field_shape[0][q * U_N_SHAPE + 0];
            const scalar_t coeff_direction_u1_1 = direction[11][lane];
            u1_direction += coeff_direction_u1_1 * field_shape[0][q * U_N_SHAPE + 1];
            const scalar_t coeff_direction_u1_2 = direction[12][lane];
            u1_direction += coeff_direction_u1_2 * field_shape[0][q * U_N_SHAPE + 2];
            const scalar_t coeff_direction_u1_3 = direction[13][lane];
            u1_direction += coeff_direction_u1_3 * field_shape[0][q * U_N_SHAPE + 3];
            const scalar_t coeff_direction_u1_4 = direction[14][lane];
            u1_direction += coeff_direction_u1_4 * field_shape[0][q * U_N_SHAPE + 4];
            const scalar_t coeff_direction_u1_5 = direction[15][lane];
            u1_direction += coeff_direction_u1_5 * field_shape[0][q * U_N_SHAPE + 5];
            const scalar_t coeff_direction_u1_6 = direction[16][lane];
            u1_direction += coeff_direction_u1_6 * field_shape[0][q * U_N_SHAPE + 6];
            const scalar_t coeff_direction_u1_7 = direction[17][lane];
            u1_direction += coeff_direction_u1_7 * field_shape[0][q * U_N_SHAPE + 7];
            const scalar_t coeff_direction_u1_8 = direction[18][lane];
            u1_direction += coeff_direction_u1_8 * field_shape[0][q * U_N_SHAPE + 8];
            const scalar_t coeff_direction_u1_9 = direction[19][lane];
            u1_direction += coeff_direction_u1_9 * field_shape[0][q * U_N_SHAPE + 9];
            scalar_t u2_direction = scalar_t(0);
            const scalar_t coeff_direction_u2_0 = direction[20][lane];
            u2_direction += coeff_direction_u2_0 * field_shape[0][q * U_N_SHAPE + 0];
            const scalar_t coeff_direction_u2_1 = direction[21][lane];
            u2_direction += coeff_direction_u2_1 * field_shape[0][q * U_N_SHAPE + 1];
            const scalar_t coeff_direction_u2_2 = direction[22][lane];
            u2_direction += coeff_direction_u2_2 * field_shape[0][q * U_N_SHAPE + 2];
            const scalar_t coeff_direction_u2_3 = direction[23][lane];
            u2_direction += coeff_direction_u2_3 * field_shape[0][q * U_N_SHAPE + 3];
            const scalar_t coeff_direction_u2_4 = direction[24][lane];
            u2_direction += coeff_direction_u2_4 * field_shape[0][q * U_N_SHAPE + 4];
            const scalar_t coeff_direction_u2_5 = direction[25][lane];
            u2_direction += coeff_direction_u2_5 * field_shape[0][q * U_N_SHAPE + 5];
            const scalar_t coeff_direction_u2_6 = direction[26][lane];
            u2_direction += coeff_direction_u2_6 * field_shape[0][q * U_N_SHAPE + 6];
            const scalar_t coeff_direction_u2_7 = direction[27][lane];
            u2_direction += coeff_direction_u2_7 * field_shape[0][q * U_N_SHAPE + 7];
            const scalar_t coeff_direction_u2_8 = direction[28][lane];
            u2_direction += coeff_direction_u2_8 * field_shape[0][q * U_N_SHAPE + 8];
            const scalar_t coeff_direction_u2_9 = direction[29][lane];
            u2_direction += coeff_direction_u2_9 * field_shape[0][q * U_N_SHAPE + 9];
            scalar_t p_direction = scalar_t(0);
            const scalar_t coeff_direction_p_0 = direction[30][lane];
            p_direction += coeff_direction_p_0 * field_shape[1][q * P_N_SHAPE + 0];
            const scalar_t coeff_direction_p_1 = direction[31][lane];
            p_direction += coeff_direction_p_1 * field_shape[1][q * P_N_SHAPE + 1];
            const scalar_t coeff_direction_p_2 = direction[32][lane];
            p_direction += coeff_direction_p_2 * field_shape[1][q * P_N_SHAPE + 2];
            const scalar_t coeff_direction_p_3 = direction[33][lane];
            p_direction += coeff_direction_p_3 * field_shape[1][q * P_N_SHAPE + 3];
            const scalar_t residual_tmp0 = -alpha*p_direction;
            const scalar_t grad_coeff0_0 = residual_tmp0;
            const scalar_t grad_coeff1_1 = residual_tmp0;
            const scalar_t grad_coeff2_2 = residual_tmp0;
            const scalar_t test_grad0_u0_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 0] * adj6) / det;
            output[0][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_0);
            const scalar_t test_grad0_u0_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 1] * adj6) / det;
            output[1][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_1);
            const scalar_t test_grad0_u0_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 2] * adj6) / det;
            output[2][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_2);
            const scalar_t test_grad0_u0_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 3] * adj6) / det;
            output[3][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_3);
            const scalar_t test_grad0_u0_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 4] * adj6) / det;
            output[4][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_4);
            const scalar_t test_grad0_u0_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 5] * adj6) / det;
            output[5][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_5);
            const scalar_t test_grad0_u0_6 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 6] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 6] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 6] * adj6) / det;
            output[6][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_6);
            const scalar_t test_grad0_u0_7 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 7] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 7] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 7] * adj6) / det;
            output[7][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_7);
            const scalar_t test_grad0_u0_8 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 8] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 8] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 8] * adj6) / det;
            output[8][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_8);
            const scalar_t test_grad0_u0_9 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 9] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 9] * adj3 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 9] * adj6) / det;
            output[9][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_9);
            const scalar_t test_grad1_u1_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 0] * adj7) / det;
            output[10][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_0);
            const scalar_t test_grad1_u1_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 1] * adj7) / det;
            output[11][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_1);
            const scalar_t test_grad1_u1_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 2] * adj7) / det;
            output[12][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_2);
            const scalar_t test_grad1_u1_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 3] * adj7) / det;
            output[13][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_3);
            const scalar_t test_grad1_u1_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 4] * adj7) / det;
            output[14][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_4);
            const scalar_t test_grad1_u1_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 5] * adj7) / det;
            output[15][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_5);
            const scalar_t test_grad1_u1_6 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 6] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 6] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 6] * adj7) / det;
            output[16][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_6);
            const scalar_t test_grad1_u1_7 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 7] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 7] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 7] * adj7) / det;
            output[17][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_7);
            const scalar_t test_grad1_u1_8 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 8] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 8] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 8] * adj7) / det;
            output[18][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_8);
            const scalar_t test_grad1_u1_9 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 9] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 9] * adj4 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 9] * adj7) / det;
            output[19][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_9);
            const scalar_t test_grad2_u2_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 0] * adj8) / det;
            output[20][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_0);
            const scalar_t test_grad2_u2_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 1] * adj8) / det;
            output[21][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_1);
            const scalar_t test_grad2_u2_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 2] * adj8) / det;
            output[22][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_2);
            const scalar_t test_grad2_u2_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 3] * adj8) / det;
            output[23][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_3);
            const scalar_t test_grad2_u2_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 4] * adj8) / det;
            output[24][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_4);
            const scalar_t test_grad2_u2_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 5] * adj8) / det;
            output[25][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_5);
            const scalar_t test_grad2_u2_6 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 6] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 6] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 6] * adj8) / det;
            output[26][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_6);
            const scalar_t test_grad2_u2_7 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 7] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 7] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 7] * adj8) / det;
            output[27][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_7);
            const scalar_t test_grad2_u2_8 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 8] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 8] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 8] * adj8) / det;
            output[28][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_8);
            const scalar_t test_grad2_u2_9 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 9] * adj2 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 9] * adj5 + field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 9] * adj8) / det;
            output[29][lane] += q_weight[q] * det * (grad_coeff2_2 * test_grad2_u2_9);
        }
    }
}

} // namespace codegen
} // namespace sfem

#endif
