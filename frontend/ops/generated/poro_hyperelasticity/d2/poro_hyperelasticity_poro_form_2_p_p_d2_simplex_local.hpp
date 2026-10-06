#ifndef PORO_HYPERELASTICITY_PORO_FORM_2_P_P_D2_SIMPLEX_LOCAL_HPP
#define PORO_HYPERELASTICITY_PORO_FORM_2_P_P_D2_SIMPLEX_LOCAL_HPP

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
static SFEM_INLINE void poro_hyperelasticity_poro_form_2_p_p_d2_simplex_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT field_shape[1],
        const scalar_t *const SFEM_RESTRICT q_weight,
        scalar_t *const SFEM_RESTRICT output[3]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 3;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
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
static SFEM_INLINE void poro_hyperelasticity_poro_form_2_p_p_d2_simplex_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT field_shape[1],
        const scalar_t *const SFEM_RESTRICT field_grad_ref[2],
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT direction[3],
        const scalar_t dt,
        const scalar_t hydraulic_conductivity,
        const scalar_t storage,
        scalar_t *const SFEM_RESTRICT output[3]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 3;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
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
            scalar_t p_direction = scalar_t(0);
            scalar_t p_direction_grad_0_ref = scalar_t(0);
            scalar_t p_direction_grad_1_ref = scalar_t(0);
            const scalar_t coeff_direction_p_0 = direction[0][lane];
            p_direction += coeff_direction_p_0 * field_shape[0][q * P_N_SHAPE + 0];
            p_direction_grad_0_ref += coeff_direction_p_0 * field_grad_ref[0 * DIM + 0][q * P_N_SHAPE + 0];
            p_direction_grad_1_ref += coeff_direction_p_0 * field_grad_ref[0 * DIM + 1][q * P_N_SHAPE + 0];
            const scalar_t coeff_direction_p_1 = direction[1][lane];
            p_direction += coeff_direction_p_1 * field_shape[0][q * P_N_SHAPE + 1];
            p_direction_grad_0_ref += coeff_direction_p_1 * field_grad_ref[0 * DIM + 0][q * P_N_SHAPE + 1];
            p_direction_grad_1_ref += coeff_direction_p_1 * field_grad_ref[0 * DIM + 1][q * P_N_SHAPE + 1];
            const scalar_t coeff_direction_p_2 = direction[2][lane];
            p_direction += coeff_direction_p_2 * field_shape[0][q * P_N_SHAPE + 2];
            p_direction_grad_0_ref += coeff_direction_p_2 * field_grad_ref[0 * DIM + 0][q * P_N_SHAPE + 2];
            p_direction_grad_1_ref += coeff_direction_p_2 * field_grad_ref[0 * DIM + 1][q * P_N_SHAPE + 2];
            const scalar_t p_direction_grad_0 = (p_direction_grad_0_ref * adj0 + p_direction_grad_1_ref * adj2) / det;
            const scalar_t p_direction_grad_1 = (p_direction_grad_0_ref * adj1 + p_direction_grad_1_ref * adj3) / det;
            const scalar_t value_coeff0 = p_direction*storage/dt;
            const scalar_t grad_coeff0_0 = hydraulic_conductivity*p_direction_grad_0;
            const scalar_t grad_coeff0_1 = hydraulic_conductivity*p_direction_grad_1;
            const scalar_t test_value_p_0 = field_shape[0][q * P_N_SHAPE + 0];
            const scalar_t test_grad0_p_0 = (field_grad_ref[0 * DIM + 0][q * P_N_SHAPE + 0] * adj0 + field_grad_ref[0 * DIM + 1][q * P_N_SHAPE + 0] * adj2) / det;
            const scalar_t test_grad1_p_0 = (field_grad_ref[0 * DIM + 0][q * P_N_SHAPE + 0] * adj1 + field_grad_ref[0 * DIM + 1][q * P_N_SHAPE + 0] * adj3) / det;
            output[0][lane] += q_weight[q] * det * (value_coeff0 * test_value_p_0 + grad_coeff0_0 * test_grad0_p_0 + grad_coeff0_1 * test_grad1_p_0);
            const scalar_t test_value_p_1 = field_shape[0][q * P_N_SHAPE + 1];
            const scalar_t test_grad0_p_1 = (field_grad_ref[0 * DIM + 0][q * P_N_SHAPE + 1] * adj0 + field_grad_ref[0 * DIM + 1][q * P_N_SHAPE + 1] * adj2) / det;
            const scalar_t test_grad1_p_1 = (field_grad_ref[0 * DIM + 0][q * P_N_SHAPE + 1] * adj1 + field_grad_ref[0 * DIM + 1][q * P_N_SHAPE + 1] * adj3) / det;
            output[1][lane] += q_weight[q] * det * (value_coeff0 * test_value_p_1 + grad_coeff0_0 * test_grad0_p_1 + grad_coeff0_1 * test_grad1_p_1);
            const scalar_t test_value_p_2 = field_shape[0][q * P_N_SHAPE + 2];
            const scalar_t test_grad0_p_2 = (field_grad_ref[0 * DIM + 0][q * P_N_SHAPE + 2] * adj0 + field_grad_ref[0 * DIM + 1][q * P_N_SHAPE + 2] * adj2) / det;
            const scalar_t test_grad1_p_2 = (field_grad_ref[0 * DIM + 0][q * P_N_SHAPE + 2] * adj1 + field_grad_ref[0 * DIM + 1][q * P_N_SHAPE + 2] * adj3) / det;
            output[2][lane] += q_weight[q] * det * (value_coeff0 * test_value_p_2 + grad_coeff0_0 * test_grad0_p_2 + grad_coeff0_1 * test_grad1_p_2);
        }
    }
}

} // namespace codegen
} // namespace sfem

#endif
