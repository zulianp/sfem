#ifndef NAVIER_STOKES_FORM_2_P_U_D2_SIMPLEX_MIXED_LOCAL_HPP
#define NAVIER_STOKES_FORM_2_P_U_D2_SIMPLEX_MIXED_LOCAL_HPP

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
typedef double geom_t;
#endif

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void navier_stokes_form_2_p_u_d2_simplex_mixed_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT field_shape[2],
        const scalar_t *const SFEM_RESTRICT q_weight,
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
        }
    }
}

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void navier_stokes_form_2_p_u_d2_simplex_mixed_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT field_shape[2],
        const scalar_t *const SFEM_RESTRICT q_weight,
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
        }
    }
}

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void navier_stokes_form_2_p_u_d2_simplex_mixed_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT field_shape[2],
        const scalar_t *const SFEM_RESTRICT field_grad_ref[4],
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT direction[15],
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
            scalar_t u0_direction_grad_0_ref = scalar_t(0);
            scalar_t u0_direction_grad_1_ref = scalar_t(0);
            const scalar_t coeff_direction_u0_0 = direction[0][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_0 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0];
            u0_direction_grad_1_ref += coeff_direction_u0_0 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0];
            const scalar_t coeff_direction_u0_1 = direction[1][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_1 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1];
            u0_direction_grad_1_ref += coeff_direction_u0_1 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1];
            const scalar_t coeff_direction_u0_2 = direction[2][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_2 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2];
            u0_direction_grad_1_ref += coeff_direction_u0_2 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2];
            const scalar_t coeff_direction_u0_3 = direction[3][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_3 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3];
            u0_direction_grad_1_ref += coeff_direction_u0_3 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3];
            const scalar_t coeff_direction_u0_4 = direction[4][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_4 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4];
            u0_direction_grad_1_ref += coeff_direction_u0_4 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4];
            const scalar_t coeff_direction_u0_5 = direction[5][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_5 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5];
            u0_direction_grad_1_ref += coeff_direction_u0_5 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5];
            const scalar_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
            const scalar_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
            scalar_t u1_direction_grad_0_ref = scalar_t(0);
            scalar_t u1_direction_grad_1_ref = scalar_t(0);
            const scalar_t coeff_direction_u1_0 = direction[6][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_0 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0];
            u1_direction_grad_1_ref += coeff_direction_u1_0 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0];
            const scalar_t coeff_direction_u1_1 = direction[7][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_1 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1];
            u1_direction_grad_1_ref += coeff_direction_u1_1 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1];
            const scalar_t coeff_direction_u1_2 = direction[8][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_2 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2];
            u1_direction_grad_1_ref += coeff_direction_u1_2 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2];
            const scalar_t coeff_direction_u1_3 = direction[9][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_3 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3];
            u1_direction_grad_1_ref += coeff_direction_u1_3 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3];
            const scalar_t coeff_direction_u1_4 = direction[10][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_4 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4];
            u1_direction_grad_1_ref += coeff_direction_u1_4 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4];
            const scalar_t coeff_direction_u1_5 = direction[11][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_5 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5];
            u1_direction_grad_1_ref += coeff_direction_u1_5 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5];
            const scalar_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj2) / det;
            const scalar_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj3) / det;
            scalar_t p_direction_grad_0_ref = scalar_t(0);
            scalar_t p_direction_grad_1_ref = scalar_t(0);
            const scalar_t coeff_direction_p_0 = direction[12][lane];
            p_direction_grad_0_ref += coeff_direction_p_0 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 0];
            p_direction_grad_1_ref += coeff_direction_p_0 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 0];
            const scalar_t coeff_direction_p_1 = direction[13][lane];
            p_direction_grad_0_ref += coeff_direction_p_1 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 1];
            p_direction_grad_1_ref += coeff_direction_p_1 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 1];
            const scalar_t coeff_direction_p_2 = direction[14][lane];
            p_direction_grad_0_ref += coeff_direction_p_2 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 2];
            p_direction_grad_1_ref += coeff_direction_p_2 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 2];
            const scalar_t p_direction_grad_0 = (p_direction_grad_0_ref * adj0 + p_direction_grad_1_ref * adj2) / det;
            const scalar_t p_direction_grad_1 = (p_direction_grad_0_ref * adj1 + p_direction_grad_1_ref * adj3) / det;
            const scalar_t value_coeff2 = u0_direction_grad_0 + u1_direction_grad_1;
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
static SFEM_INLINE void navier_stokes_form_2_p_u_d2_simplex_mixed_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT field_shape[2],
        const scalar_t *const SFEM_RESTRICT field_grad_ref[4],
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t direction[15][VECTOR_SIZE],
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
            scalar_t u0_direction_grad_0_ref = scalar_t(0);
            scalar_t u0_direction_grad_1_ref = scalar_t(0);
            const scalar_t coeff_direction_u0_0 = direction[0][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_0 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0];
            u0_direction_grad_1_ref += coeff_direction_u0_0 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0];
            const scalar_t coeff_direction_u0_1 = direction[1][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_1 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1];
            u0_direction_grad_1_ref += coeff_direction_u0_1 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1];
            const scalar_t coeff_direction_u0_2 = direction[2][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_2 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2];
            u0_direction_grad_1_ref += coeff_direction_u0_2 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2];
            const scalar_t coeff_direction_u0_3 = direction[3][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_3 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3];
            u0_direction_grad_1_ref += coeff_direction_u0_3 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3];
            const scalar_t coeff_direction_u0_4 = direction[4][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_4 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4];
            u0_direction_grad_1_ref += coeff_direction_u0_4 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4];
            const scalar_t coeff_direction_u0_5 = direction[5][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_5 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5];
            u0_direction_grad_1_ref += coeff_direction_u0_5 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5];
            const scalar_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
            const scalar_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
            scalar_t u1_direction_grad_0_ref = scalar_t(0);
            scalar_t u1_direction_grad_1_ref = scalar_t(0);
            const scalar_t coeff_direction_u1_0 = direction[6][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_0 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0];
            u1_direction_grad_1_ref += coeff_direction_u1_0 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0];
            const scalar_t coeff_direction_u1_1 = direction[7][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_1 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1];
            u1_direction_grad_1_ref += coeff_direction_u1_1 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1];
            const scalar_t coeff_direction_u1_2 = direction[8][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_2 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2];
            u1_direction_grad_1_ref += coeff_direction_u1_2 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2];
            const scalar_t coeff_direction_u1_3 = direction[9][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_3 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3];
            u1_direction_grad_1_ref += coeff_direction_u1_3 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3];
            const scalar_t coeff_direction_u1_4 = direction[10][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_4 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4];
            u1_direction_grad_1_ref += coeff_direction_u1_4 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4];
            const scalar_t coeff_direction_u1_5 = direction[11][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_5 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5];
            u1_direction_grad_1_ref += coeff_direction_u1_5 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5];
            const scalar_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj2) / det;
            const scalar_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj3) / det;
            scalar_t p_direction_grad_0_ref = scalar_t(0);
            scalar_t p_direction_grad_1_ref = scalar_t(0);
            const scalar_t coeff_direction_p_0 = direction[12][lane];
            p_direction_grad_0_ref += coeff_direction_p_0 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 0];
            p_direction_grad_1_ref += coeff_direction_p_0 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 0];
            const scalar_t coeff_direction_p_1 = direction[13][lane];
            p_direction_grad_0_ref += coeff_direction_p_1 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 1];
            p_direction_grad_1_ref += coeff_direction_p_1 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 1];
            const scalar_t coeff_direction_p_2 = direction[14][lane];
            p_direction_grad_0_ref += coeff_direction_p_2 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 2];
            p_direction_grad_1_ref += coeff_direction_p_2 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 2];
            const scalar_t p_direction_grad_0 = (p_direction_grad_0_ref * adj0 + p_direction_grad_1_ref * adj2) / det;
            const scalar_t p_direction_grad_1 = (p_direction_grad_0_ref * adj1 + p_direction_grad_1_ref * adj3) / det;
            const scalar_t value_coeff2 = u0_direction_grad_0 + u1_direction_grad_1;
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
