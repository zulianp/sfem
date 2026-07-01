#ifndef PORO_HYPERELASTICITY_PORO_FORM_1_U_D2_SIMPLEX_MIXED_LOCAL_HPP
#define PORO_HYPERELASTICITY_PORO_FORM_1_U_D2_SIMPLEX_MIXED_LOCAL_HPP

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
static SFEM_INLINE void poro_hyperelasticity_poro_form_1_u_d2_simplex_mixed_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT field_shape[2],
        const scalar_t *const SFEM_RESTRICT field_grad_ref[4],
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT current[15],
        const scalar_t alpha,
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
            const scalar_t coeff_current_u0_0 = current[0][lane];
            u0 += coeff_current_u0_0 * field_shape[0][q * U_N_SHAPE + 0];
            const scalar_t coeff_current_u0_1 = current[1][lane];
            u0 += coeff_current_u0_1 * field_shape[0][q * U_N_SHAPE + 1];
            const scalar_t coeff_current_u0_2 = current[2][lane];
            u0 += coeff_current_u0_2 * field_shape[0][q * U_N_SHAPE + 2];
            const scalar_t coeff_current_u0_3 = current[3][lane];
            u0 += coeff_current_u0_3 * field_shape[0][q * U_N_SHAPE + 3];
            const scalar_t coeff_current_u0_4 = current[4][lane];
            u0 += coeff_current_u0_4 * field_shape[0][q * U_N_SHAPE + 4];
            const scalar_t coeff_current_u0_5 = current[5][lane];
            u0 += coeff_current_u0_5 * field_shape[0][q * U_N_SHAPE + 5];
            scalar_t u1 = scalar_t(0);
            const scalar_t coeff_current_u1_0 = current[6][lane];
            u1 += coeff_current_u1_0 * field_shape[0][q * U_N_SHAPE + 0];
            const scalar_t coeff_current_u1_1 = current[7][lane];
            u1 += coeff_current_u1_1 * field_shape[0][q * U_N_SHAPE + 1];
            const scalar_t coeff_current_u1_2 = current[8][lane];
            u1 += coeff_current_u1_2 * field_shape[0][q * U_N_SHAPE + 2];
            const scalar_t coeff_current_u1_3 = current[9][lane];
            u1 += coeff_current_u1_3 * field_shape[0][q * U_N_SHAPE + 3];
            const scalar_t coeff_current_u1_4 = current[10][lane];
            u1 += coeff_current_u1_4 * field_shape[0][q * U_N_SHAPE + 4];
            const scalar_t coeff_current_u1_5 = current[11][lane];
            u1 += coeff_current_u1_5 * field_shape[0][q * U_N_SHAPE + 5];
            scalar_t p = scalar_t(0);
            const scalar_t coeff_current_p_0 = current[12][lane];
            p += coeff_current_p_0 * field_shape[1][q * P_N_SHAPE + 0];
            const scalar_t coeff_current_p_1 = current[13][lane];
            p += coeff_current_p_1 * field_shape[1][q * P_N_SHAPE + 1];
            const scalar_t coeff_current_p_2 = current[14][lane];
            p += coeff_current_p_2 * field_shape[1][q * P_N_SHAPE + 2];
            const scalar_t residual_tmp0 = -alpha*p;
            const scalar_t grad_coeff0_0 = residual_tmp0;
            const scalar_t grad_coeff1_1 = residual_tmp0;
            const scalar_t test_grad0_u0_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj2) / det;
            output[0][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_0);
            const scalar_t test_grad0_u0_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj2) / det;
            output[1][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_1);
            const scalar_t test_grad0_u0_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj2) / det;
            output[2][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_2);
            const scalar_t test_grad0_u0_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj2) / det;
            output[3][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_3);
            const scalar_t test_grad0_u0_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj2) / det;
            output[4][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_4);
            const scalar_t test_grad0_u0_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj0 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj2) / det;
            output[5][lane] += q_weight[q] * det * (grad_coeff0_0 * test_grad0_u0_5);
            const scalar_t test_grad1_u1_0 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 0] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 0] * adj3) / det;
            output[6][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_0);
            const scalar_t test_grad1_u1_1 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 1] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 1] * adj3) / det;
            output[7][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_1);
            const scalar_t test_grad1_u1_2 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 2] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 2] * adj3) / det;
            output[8][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_2);
            const scalar_t test_grad1_u1_3 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 3] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 3] * adj3) / det;
            output[9][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_3);
            const scalar_t test_grad1_u1_4 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 4] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 4] * adj3) / det;
            output[10][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_4);
            const scalar_t test_grad1_u1_5 = (field_grad_ref[0 * DIM + 0][q * U_N_SHAPE + 5] * adj1 + field_grad_ref[0 * DIM + 1][q * U_N_SHAPE + 5] * adj3) / det;
            output[11][lane] += q_weight[q] * det * (grad_coeff1_1 * test_grad1_u1_5);
        }
    }
}

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void poro_hyperelasticity_poro_form_1_u_d2_simplex_mixed_jacobian_action_block(
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

} // namespace codegen
} // namespace sfem

#endif
