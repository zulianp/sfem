#ifndef PORO_HYPERELASTICITY_PORO_FORM_2_P_P_D3_TENSOR_PRODUCT_LOCAL_HPP
#define PORO_HYPERELASTICITY_PORO_FORM_2_P_P_D3_TENSOR_PRODUCT_LOCAL_HPP

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
static SFEM_INLINE void poro_hyperelasticity_poro_form_2_p_p_d3_tensor_product_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT field_shape_1d[1],
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        scalar_t *const SFEM_RESTRICT output[8]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 8;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int P_N_SHAPE = 8;
    static constexpr int N_QP_1D = integer_root(N_QP, DIM);
    static_assert(ipow(N_QP_1D, DIM) == N_QP, "N_QP must be tensor-product compatible");
    static constexpr int P_N_SHAPE_1D = integer_root(P_N_SHAPE, DIM);
    static_assert(ipow(P_N_SHAPE_1D, DIM) == P_N_SHAPE, "P_N_SHAPE must be tensor-product compatible");
}

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void poro_hyperelasticity_poro_form_2_p_p_d3_tensor_product_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT field_shape_1d[1],
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        scalar_t output[8][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 8;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int P_N_SHAPE = 8;
    static constexpr int N_QP_1D = integer_root(N_QP, DIM);
    static_assert(ipow(N_QP_1D, DIM) == N_QP, "N_QP must be tensor-product compatible");
    static constexpr int P_N_SHAPE_1D = integer_root(P_N_SHAPE, DIM);
    static_assert(ipow(P_N_SHAPE_1D, DIM) == P_N_SHAPE, "P_N_SHAPE must be tensor-product compatible");
}

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void poro_hyperelasticity_poro_form_2_p_p_d3_tensor_product_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT field_shape_1d[1],
        const scalar_t *const SFEM_RESTRICT field_grad_1d[1],
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t *const SFEM_RESTRICT direction[8],
        const scalar_t dt,
        const scalar_t hydraulic_conductivity,
        const scalar_t storage,
        scalar_t *const SFEM_RESTRICT output[8]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 8;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int P_N_SHAPE = 8;
    static constexpr int N_QP_1D = integer_root(N_QP, DIM);
    static_assert(ipow(N_QP_1D, DIM) == N_QP, "N_QP must be tensor-product compatible");
    static constexpr int P_N_SHAPE_1D = integer_root(P_N_SHAPE, DIM);
    static_assert(ipow(P_N_SHAPE_1D, DIM) == P_N_SHAPE, "P_N_SHAPE must be tensor-product compatible");
    scalar_t direction_p_value[N_QP * VECTOR_SIZE];
    scalar_t direction_p_grad_ref[N_QP * DIM * VECTOR_SIZE];
    const scalar_t *const direction_p_streams[P_N_SHAPE] = {direction[0], direction[1], direction[2], direction[3], direction[4], direction[5], direction[6], direction[7]};
    tensor_evaluate<scalar_t, N_QP, P_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], direction_p_streams, direction_p_value, direction_p_grad_ref);
    scalar_t p_value_coeff[N_QP * VECTOR_SIZE];
    scalar_t p_grad_coeff_ref[N_QP * DIM * VECTOR_SIZE];
    for (int q = 0; q < N_QP; ++q) {
        const int qx = q % N_QP_1D;
        const int qy = (q / N_QP_1D) % N_QP_1D;
        const int qz = q / (N_QP_1D * N_QP_1D);
        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
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
            const scalar_t p_direction = direction_p_value[q * VECTOR_SIZE + lane];
            const scalar_t p_direction_grad_0_ref = direction_p_grad_ref[(q * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t p_direction_grad_1_ref = direction_p_grad_ref[(q * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t p_direction_grad_2_ref = direction_p_grad_ref[(q * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t p_direction_grad_0 = (p_direction_grad_0_ref * adj0 + p_direction_grad_1_ref * adj3 + p_direction_grad_2_ref * adj6) / det;
            const scalar_t p_direction_grad_1 = (p_direction_grad_0_ref * adj1 + p_direction_grad_1_ref * adj4 + p_direction_grad_2_ref * adj7) / det;
            const scalar_t p_direction_grad_2 = (p_direction_grad_0_ref * adj2 + p_direction_grad_1_ref * adj5 + p_direction_grad_2_ref * adj8) / det;
            const scalar_t value_coeff0 = p_direction*storage/dt;
            const scalar_t grad_coeff0_0 = hydraulic_conductivity*p_direction_grad_0;
            const scalar_t grad_coeff0_1 = hydraulic_conductivity*p_direction_grad_1;
            const scalar_t grad_coeff0_2 = hydraulic_conductivity*p_direction_grad_2;
            p_value_coeff[q * VECTOR_SIZE + lane] = qw * det * value_coeff0;
            p_grad_coeff_ref[(q * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1 + adj2 * grad_coeff0_2);
            p_grad_coeff_ref[(q * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj3 * grad_coeff0_0 + adj4 * grad_coeff0_1 + adj5 * grad_coeff0_2);
            p_grad_coeff_ref[(q * DIM + 2) * VECTOR_SIZE + lane] = qw * (adj6 * grad_coeff0_0 + adj7 * grad_coeff0_1 + adj8 * grad_coeff0_2);
        }
    }
    scalar_t *const p_output_streams[P_N_SHAPE] = {output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7]};
    tensor_integrate<scalar_t, N_QP, P_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], p_value_coeff, p_grad_coeff_ref, p_output_streams);
}

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void poro_hyperelasticity_poro_form_2_p_p_d3_tensor_product_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT field_shape_1d[1],
        const scalar_t *const SFEM_RESTRICT field_grad_1d[1],
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t direction[8][VECTOR_SIZE],
        const scalar_t dt,
        const scalar_t hydraulic_conductivity,
        const scalar_t storage,
        scalar_t output[8][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 8;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int P_N_SHAPE = 8;
    static constexpr int N_QP_1D = integer_root(N_QP, DIM);
    static_assert(ipow(N_QP_1D, DIM) == N_QP, "N_QP must be tensor-product compatible");
    static constexpr int P_N_SHAPE_1D = integer_root(P_N_SHAPE, DIM);
    static_assert(ipow(P_N_SHAPE_1D, DIM) == P_N_SHAPE, "P_N_SHAPE must be tensor-product compatible");
    scalar_t direction_p_value[N_QP * VECTOR_SIZE];
    scalar_t direction_p_grad_ref[N_QP * DIM * VECTOR_SIZE];
    tensor_evaluate_contiguous<scalar_t, N_QP, P_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], direction + 0, direction_p_value, direction_p_grad_ref);
    scalar_t p_value_coeff[N_QP * VECTOR_SIZE];
    scalar_t p_grad_coeff_ref[N_QP * DIM * VECTOR_SIZE];
    for (int q = 0; q < N_QP; ++q) {
        const int qx = q % N_QP_1D;
        const int qy = (q / N_QP_1D) % N_QP_1D;
        const int qz = q / (N_QP_1D * N_QP_1D);
        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
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
            const scalar_t p_direction = direction_p_value[q * VECTOR_SIZE + lane];
            const scalar_t p_direction_grad_0_ref = direction_p_grad_ref[(q * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t p_direction_grad_1_ref = direction_p_grad_ref[(q * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t p_direction_grad_2_ref = direction_p_grad_ref[(q * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t p_direction_grad_0 = (p_direction_grad_0_ref * adj0 + p_direction_grad_1_ref * adj3 + p_direction_grad_2_ref * adj6) / det;
            const scalar_t p_direction_grad_1 = (p_direction_grad_0_ref * adj1 + p_direction_grad_1_ref * adj4 + p_direction_grad_2_ref * adj7) / det;
            const scalar_t p_direction_grad_2 = (p_direction_grad_0_ref * adj2 + p_direction_grad_1_ref * adj5 + p_direction_grad_2_ref * adj8) / det;
            const scalar_t value_coeff0 = p_direction*storage/dt;
            const scalar_t grad_coeff0_0 = hydraulic_conductivity*p_direction_grad_0;
            const scalar_t grad_coeff0_1 = hydraulic_conductivity*p_direction_grad_1;
            const scalar_t grad_coeff0_2 = hydraulic_conductivity*p_direction_grad_2;
            p_value_coeff[q * VECTOR_SIZE + lane] = qw * det * value_coeff0;
            p_grad_coeff_ref[(q * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1 + adj2 * grad_coeff0_2);
            p_grad_coeff_ref[(q * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj3 * grad_coeff0_0 + adj4 * grad_coeff0_1 + adj5 * grad_coeff0_2);
            p_grad_coeff_ref[(q * DIM + 2) * VECTOR_SIZE + lane] = qw * (adj6 * grad_coeff0_0 + adj7 * grad_coeff0_1 + adj8 * grad_coeff0_2);
        }
    }
    tensor_integrate_contiguous<scalar_t, N_QP, P_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], p_value_coeff, p_grad_coeff_ref, output + 0);
}

} // namespace codegen
} // namespace sfem

#endif
