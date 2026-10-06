#ifndef PORO_HYPERELASTICITY_PORO_FORM_2_P_U_D3_SIMPLEX_MIXED_LOCAL_HPP
#define PORO_HYPERELASTICITY_PORO_FORM_2_P_U_D3_SIMPLEX_MIXED_LOCAL_HPP

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
static SFEM_INLINE void poro_hyperelasticity_poro_form_2_p_u_d3_simplex_mixed_residual_block(
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
static SFEM_INLINE void poro_hyperelasticity_poro_form_2_p_u_d3_simplex_mixed_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT field_shape[2],
        const scalar_t *const SFEM_RESTRICT field_grad_ref[6],
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT direction[34],
        const scalar_t alpha,
        const scalar_t dt,
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
            scalar_t u0_direction_grad_0_ref = scalar_t(0);
            scalar_t u0_direction_grad_1_ref = scalar_t(0);
            scalar_t u0_direction_grad_2_ref = scalar_t(0);
            const scalar_t coeff_direction_u0_0 = direction[0][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_0 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0];
            u0_direction_grad_1_ref += coeff_direction_u0_0 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0];
            u0_direction_grad_2_ref += coeff_direction_u0_0 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 0];
            const scalar_t coeff_direction_u0_1 = direction[1][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_1 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1];
            u0_direction_grad_1_ref += coeff_direction_u0_1 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1];
            u0_direction_grad_2_ref += coeff_direction_u0_1 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 1];
            const scalar_t coeff_direction_u0_2 = direction[2][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_2 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2];
            u0_direction_grad_1_ref += coeff_direction_u0_2 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2];
            u0_direction_grad_2_ref += coeff_direction_u0_2 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 2];
            const scalar_t coeff_direction_u0_3 = direction[3][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_3 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3];
            u0_direction_grad_1_ref += coeff_direction_u0_3 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3];
            u0_direction_grad_2_ref += coeff_direction_u0_3 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 3];
            const scalar_t coeff_direction_u0_4 = direction[4][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_4 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4];
            u0_direction_grad_1_ref += coeff_direction_u0_4 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4];
            u0_direction_grad_2_ref += coeff_direction_u0_4 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 4];
            const scalar_t coeff_direction_u0_5 = direction[5][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_5 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5];
            u0_direction_grad_1_ref += coeff_direction_u0_5 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5];
            u0_direction_grad_2_ref += coeff_direction_u0_5 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 5];
            const scalar_t coeff_direction_u0_6 = direction[6][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_6 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 6];
            u0_direction_grad_1_ref += coeff_direction_u0_6 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 6];
            u0_direction_grad_2_ref += coeff_direction_u0_6 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 6];
            const scalar_t coeff_direction_u0_7 = direction[7][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_7 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 7];
            u0_direction_grad_1_ref += coeff_direction_u0_7 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 7];
            u0_direction_grad_2_ref += coeff_direction_u0_7 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 7];
            const scalar_t coeff_direction_u0_8 = direction[8][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_8 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 8];
            u0_direction_grad_1_ref += coeff_direction_u0_8 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 8];
            u0_direction_grad_2_ref += coeff_direction_u0_8 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 8];
            const scalar_t coeff_direction_u0_9 = direction[9][lane];
            u0_direction_grad_0_ref += coeff_direction_u0_9 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 9];
            u0_direction_grad_1_ref += coeff_direction_u0_9 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 9];
            u0_direction_grad_2_ref += coeff_direction_u0_9 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 9];
            const scalar_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
            const scalar_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
            const scalar_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
            scalar_t u1_direction_grad_0_ref = scalar_t(0);
            scalar_t u1_direction_grad_1_ref = scalar_t(0);
            scalar_t u1_direction_grad_2_ref = scalar_t(0);
            const scalar_t coeff_direction_u1_0 = direction[10][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_0 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0];
            u1_direction_grad_1_ref += coeff_direction_u1_0 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0];
            u1_direction_grad_2_ref += coeff_direction_u1_0 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 0];
            const scalar_t coeff_direction_u1_1 = direction[11][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_1 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1];
            u1_direction_grad_1_ref += coeff_direction_u1_1 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1];
            u1_direction_grad_2_ref += coeff_direction_u1_1 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 1];
            const scalar_t coeff_direction_u1_2 = direction[12][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_2 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2];
            u1_direction_grad_1_ref += coeff_direction_u1_2 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2];
            u1_direction_grad_2_ref += coeff_direction_u1_2 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 2];
            const scalar_t coeff_direction_u1_3 = direction[13][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_3 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3];
            u1_direction_grad_1_ref += coeff_direction_u1_3 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3];
            u1_direction_grad_2_ref += coeff_direction_u1_3 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 3];
            const scalar_t coeff_direction_u1_4 = direction[14][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_4 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4];
            u1_direction_grad_1_ref += coeff_direction_u1_4 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4];
            u1_direction_grad_2_ref += coeff_direction_u1_4 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 4];
            const scalar_t coeff_direction_u1_5 = direction[15][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_5 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5];
            u1_direction_grad_1_ref += coeff_direction_u1_5 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5];
            u1_direction_grad_2_ref += coeff_direction_u1_5 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 5];
            const scalar_t coeff_direction_u1_6 = direction[16][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_6 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 6];
            u1_direction_grad_1_ref += coeff_direction_u1_6 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 6];
            u1_direction_grad_2_ref += coeff_direction_u1_6 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 6];
            const scalar_t coeff_direction_u1_7 = direction[17][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_7 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 7];
            u1_direction_grad_1_ref += coeff_direction_u1_7 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 7];
            u1_direction_grad_2_ref += coeff_direction_u1_7 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 7];
            const scalar_t coeff_direction_u1_8 = direction[18][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_8 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 8];
            u1_direction_grad_1_ref += coeff_direction_u1_8 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 8];
            u1_direction_grad_2_ref += coeff_direction_u1_8 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 8];
            const scalar_t coeff_direction_u1_9 = direction[19][lane];
            u1_direction_grad_0_ref += coeff_direction_u1_9 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 9];
            u1_direction_grad_1_ref += coeff_direction_u1_9 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 9];
            u1_direction_grad_2_ref += coeff_direction_u1_9 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 9];
            const scalar_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
            const scalar_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
            const scalar_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
            scalar_t u2_direction_grad_0_ref = scalar_t(0);
            scalar_t u2_direction_grad_1_ref = scalar_t(0);
            scalar_t u2_direction_grad_2_ref = scalar_t(0);
            const scalar_t coeff_direction_u2_0 = direction[20][lane];
            u2_direction_grad_0_ref += coeff_direction_u2_0 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0];
            u2_direction_grad_1_ref += coeff_direction_u2_0 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0];
            u2_direction_grad_2_ref += coeff_direction_u2_0 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 0];
            const scalar_t coeff_direction_u2_1 = direction[21][lane];
            u2_direction_grad_0_ref += coeff_direction_u2_1 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1];
            u2_direction_grad_1_ref += coeff_direction_u2_1 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1];
            u2_direction_grad_2_ref += coeff_direction_u2_1 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 1];
            const scalar_t coeff_direction_u2_2 = direction[22][lane];
            u2_direction_grad_0_ref += coeff_direction_u2_2 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2];
            u2_direction_grad_1_ref += coeff_direction_u2_2 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2];
            u2_direction_grad_2_ref += coeff_direction_u2_2 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 2];
            const scalar_t coeff_direction_u2_3 = direction[23][lane];
            u2_direction_grad_0_ref += coeff_direction_u2_3 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3];
            u2_direction_grad_1_ref += coeff_direction_u2_3 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3];
            u2_direction_grad_2_ref += coeff_direction_u2_3 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 3];
            const scalar_t coeff_direction_u2_4 = direction[24][lane];
            u2_direction_grad_0_ref += coeff_direction_u2_4 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4];
            u2_direction_grad_1_ref += coeff_direction_u2_4 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4];
            u2_direction_grad_2_ref += coeff_direction_u2_4 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 4];
            const scalar_t coeff_direction_u2_5 = direction[25][lane];
            u2_direction_grad_0_ref += coeff_direction_u2_5 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5];
            u2_direction_grad_1_ref += coeff_direction_u2_5 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5];
            u2_direction_grad_2_ref += coeff_direction_u2_5 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 5];
            const scalar_t coeff_direction_u2_6 = direction[26][lane];
            u2_direction_grad_0_ref += coeff_direction_u2_6 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 6];
            u2_direction_grad_1_ref += coeff_direction_u2_6 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 6];
            u2_direction_grad_2_ref += coeff_direction_u2_6 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 6];
            const scalar_t coeff_direction_u2_7 = direction[27][lane];
            u2_direction_grad_0_ref += coeff_direction_u2_7 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 7];
            u2_direction_grad_1_ref += coeff_direction_u2_7 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 7];
            u2_direction_grad_2_ref += coeff_direction_u2_7 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 7];
            const scalar_t coeff_direction_u2_8 = direction[28][lane];
            u2_direction_grad_0_ref += coeff_direction_u2_8 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 8];
            u2_direction_grad_1_ref += coeff_direction_u2_8 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 8];
            u2_direction_grad_2_ref += coeff_direction_u2_8 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 8];
            const scalar_t coeff_direction_u2_9 = direction[29][lane];
            u2_direction_grad_0_ref += coeff_direction_u2_9 * field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 9];
            u2_direction_grad_1_ref += coeff_direction_u2_9 * field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 9];
            u2_direction_grad_2_ref += coeff_direction_u2_9 * field_grad_ref[0 * DIM + 2][q * U_N_SHAPE + 9];
            const scalar_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
            const scalar_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
            const scalar_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
            scalar_t p_direction_grad_0_ref = scalar_t(0);
            scalar_t p_direction_grad_1_ref = scalar_t(0);
            scalar_t p_direction_grad_2_ref = scalar_t(0);
            const scalar_t coeff_direction_p_0 = direction[30][lane];
            p_direction_grad_0_ref += coeff_direction_p_0 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 0];
            p_direction_grad_1_ref += coeff_direction_p_0 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 0];
            p_direction_grad_2_ref += coeff_direction_p_0 * field_grad_ref[1 * DIM + 2][q * P_N_SHAPE + 0];
            const scalar_t coeff_direction_p_1 = direction[31][lane];
            p_direction_grad_0_ref += coeff_direction_p_1 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 1];
            p_direction_grad_1_ref += coeff_direction_p_1 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 1];
            p_direction_grad_2_ref += coeff_direction_p_1 * field_grad_ref[1 * DIM + 2][q * P_N_SHAPE + 1];
            const scalar_t coeff_direction_p_2 = direction[32][lane];
            p_direction_grad_0_ref += coeff_direction_p_2 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 2];
            p_direction_grad_1_ref += coeff_direction_p_2 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 2];
            p_direction_grad_2_ref += coeff_direction_p_2 * field_grad_ref[1 * DIM + 2][q * P_N_SHAPE + 2];
            const scalar_t coeff_direction_p_3 = direction[33][lane];
            p_direction_grad_0_ref += coeff_direction_p_3 * field_grad_ref[1 * DIM + 0][q * P_N_SHAPE + 3];
            p_direction_grad_1_ref += coeff_direction_p_3 * field_grad_ref[1 * DIM + 1][q * P_N_SHAPE + 3];
            p_direction_grad_2_ref += coeff_direction_p_3 * field_grad_ref[1 * DIM + 2][q * P_N_SHAPE + 3];
            const scalar_t p_direction_grad_0 = (p_direction_grad_0_ref * adj0 + p_direction_grad_1_ref * adj3 + p_direction_grad_2_ref * adj6) / det;
            const scalar_t p_direction_grad_1 = (p_direction_grad_0_ref * adj1 + p_direction_grad_1_ref * adj4 + p_direction_grad_2_ref * adj7) / det;
            const scalar_t p_direction_grad_2 = (p_direction_grad_0_ref * adj2 + p_direction_grad_1_ref * adj5 + p_direction_grad_2_ref * adj8) / det;
            const scalar_t residual_tmp0 = alpha/dt;
            const scalar_t value_coeff3 = residual_tmp0*u0_direction_grad_0 + residual_tmp0*u1_direction_grad_1 + residual_tmp0*u2_direction_grad_2;
            const scalar_t test_value_p_0 = field_shape[1][q * P_N_SHAPE + 0];
            output[30][lane] += q_weight[q] * det * (value_coeff3 * test_value_p_0);
            const scalar_t test_value_p_1 = field_shape[1][q * P_N_SHAPE + 1];
            output[31][lane] += q_weight[q] * det * (value_coeff3 * test_value_p_1);
            const scalar_t test_value_p_2 = field_shape[1][q * P_N_SHAPE + 2];
            output[32][lane] += q_weight[q] * det * (value_coeff3 * test_value_p_2);
            const scalar_t test_value_p_3 = field_shape[1][q * P_N_SHAPE + 3];
            output[33][lane] += q_weight[q] * det * (value_coeff3 * test_value_p_3);
        }
    }
}

} // namespace codegen
} // namespace sfem

#endif
