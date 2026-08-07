#ifndef NAVIER_STOKES_FORM_2_U_P_D3_TENSOR_PRODUCT_MIXED_LOCAL_HPP
#define NAVIER_STOKES_FORM_2_U_P_D3_TENSOR_PRODUCT_MIXED_LOCAL_HPP

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
static SFEM_INLINE void navier_stokes_form_2_u_p_d3_tensor_product_mixed_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT field_shape_1d[2],
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        scalar_t *const SFEM_RESTRICT output[89]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 89;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int U_N_SHAPE = 27;
    static constexpr int P_N_SHAPE = 8;
    static constexpr int N_QP_1D = integer_root(N_QP, DIM);
    static_assert(ipow(N_QP_1D, DIM) == N_QP, "N_QP must be tensor-product compatible");
    static constexpr int U_N_SHAPE_1D = integer_root(U_N_SHAPE, DIM);
    static_assert(ipow(U_N_SHAPE_1D, DIM) == U_N_SHAPE, "U_N_SHAPE must be tensor-product compatible");
    static constexpr int P_N_SHAPE_1D = integer_root(P_N_SHAPE, DIM);
    static_assert(ipow(P_N_SHAPE_1D, DIM) == P_N_SHAPE, "P_N_SHAPE must be tensor-product compatible");
}

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void navier_stokes_form_2_u_p_d3_tensor_product_mixed_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT field_shape_1d[2],
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        scalar_t output[89][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 89;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int U_N_SHAPE = 27;
    static constexpr int P_N_SHAPE = 8;
    static constexpr int N_QP_1D = integer_root(N_QP, DIM);
    static_assert(ipow(N_QP_1D, DIM) == N_QP, "N_QP must be tensor-product compatible");
    static constexpr int U_N_SHAPE_1D = integer_root(U_N_SHAPE, DIM);
    static_assert(ipow(U_N_SHAPE_1D, DIM) == U_N_SHAPE, "U_N_SHAPE must be tensor-product compatible");
    static constexpr int P_N_SHAPE_1D = integer_root(P_N_SHAPE, DIM);
    static_assert(ipow(P_N_SHAPE_1D, DIM) == P_N_SHAPE, "P_N_SHAPE must be tensor-product compatible");
}

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void navier_stokes_form_2_u_p_d3_tensor_product_mixed_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT field_shape_1d[2],
        const scalar_t *const SFEM_RESTRICT field_grad_1d[2],
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t *const SFEM_RESTRICT direction[89],
        scalar_t *const SFEM_RESTRICT output[89]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 89;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int U_N_SHAPE = 27;
    static constexpr int P_N_SHAPE = 8;
    static constexpr int N_QP_1D = integer_root(N_QP, DIM);
    static_assert(ipow(N_QP_1D, DIM) == N_QP, "N_QP must be tensor-product compatible");
    static constexpr int U_N_SHAPE_1D = integer_root(U_N_SHAPE, DIM);
    static_assert(ipow(U_N_SHAPE_1D, DIM) == U_N_SHAPE, "U_N_SHAPE must be tensor-product compatible");
    static constexpr int P_N_SHAPE_1D = integer_root(P_N_SHAPE, DIM);
    static_assert(ipow(P_N_SHAPE_1D, DIM) == P_N_SHAPE, "P_N_SHAPE must be tensor-product compatible");
    scalar_t direction_u0_value[N_QP * VECTOR_SIZE];
    const scalar_t *const direction_u0_streams[U_N_SHAPE] = {direction[0], direction[1], direction[2], direction[3], direction[4], direction[5], direction[6], direction[7], direction[8], direction[9], direction[10], direction[11], direction[12], direction[13], direction[14], direction[15], direction[16], direction[17], direction[18], direction[19], direction[20], direction[21], direction[22], direction[23], direction[24], direction[25], direction[26]};
    tensor_evaluate_value<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], direction_u0_streams, direction_u0_value);
    scalar_t direction_u1_value[N_QP * VECTOR_SIZE];
    const scalar_t *const direction_u1_streams[U_N_SHAPE] = {direction[27], direction[28], direction[29], direction[30], direction[31], direction[32], direction[33], direction[34], direction[35], direction[36], direction[37], direction[38], direction[39], direction[40], direction[41], direction[42], direction[43], direction[44], direction[45], direction[46], direction[47], direction[48], direction[49], direction[50], direction[51], direction[52], direction[53]};
    tensor_evaluate_value<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], direction_u1_streams, direction_u1_value);
    scalar_t direction_u2_value[N_QP * VECTOR_SIZE];
    const scalar_t *const direction_u2_streams[U_N_SHAPE] = {direction[54], direction[55], direction[56], direction[57], direction[58], direction[59], direction[60], direction[61], direction[62], direction[63], direction[64], direction[65], direction[66], direction[67], direction[68], direction[69], direction[70], direction[71], direction[72], direction[73], direction[74], direction[75], direction[76], direction[77], direction[78], direction[79], direction[80]};
    tensor_evaluate_value<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], direction_u2_streams, direction_u2_value);
    scalar_t direction_p_value[N_QP * VECTOR_SIZE];
    const scalar_t *const direction_p_streams[P_N_SHAPE] = {direction[81], direction[82], direction[83], direction[84], direction[85], direction[86], direction[87], direction[88]};
    tensor_evaluate_value<scalar_t, N_QP, P_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[1], direction_p_streams, direction_p_value);
    scalar_t u0_value_coeff[N_QP * VECTOR_SIZE];
    scalar_t u0_grad_coeff_ref[N_QP * DIM * VECTOR_SIZE];
    scalar_t u1_value_coeff[N_QP * VECTOR_SIZE];
    scalar_t u1_grad_coeff_ref[N_QP * DIM * VECTOR_SIZE];
    scalar_t u2_value_coeff[N_QP * VECTOR_SIZE];
    scalar_t u2_grad_coeff_ref[N_QP * DIM * VECTOR_SIZE];
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
            const scalar_t u0_direction = direction_u0_value[q * VECTOR_SIZE + lane];
            const scalar_t u1_direction = direction_u1_value[q * VECTOR_SIZE + lane];
            const scalar_t u2_direction = direction_u2_value[q * VECTOR_SIZE + lane];
            const scalar_t p_direction = direction_p_value[q * VECTOR_SIZE + lane];
            const scalar_t residual_tmp0 = -p_direction;
            const scalar_t grad_coeff0_0 = residual_tmp0;
            const scalar_t grad_coeff1_1 = residual_tmp0;
            const scalar_t grad_coeff2_2 = residual_tmp0;
            u0_value_coeff[q * VECTOR_SIZE + lane] = scalar_t(0);
            u0_grad_coeff_ref[(q * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff0_0);
            u0_grad_coeff_ref[(q * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj3 * grad_coeff0_0);
            u0_grad_coeff_ref[(q * DIM + 2) * VECTOR_SIZE + lane] = qw * (adj6 * grad_coeff0_0);
            u1_value_coeff[q * VECTOR_SIZE + lane] = scalar_t(0);
            u1_grad_coeff_ref[(q * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj1 * grad_coeff1_1);
            u1_grad_coeff_ref[(q * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj4 * grad_coeff1_1);
            u1_grad_coeff_ref[(q * DIM + 2) * VECTOR_SIZE + lane] = qw * (adj7 * grad_coeff1_1);
            u2_value_coeff[q * VECTOR_SIZE + lane] = scalar_t(0);
            u2_grad_coeff_ref[(q * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj2 * grad_coeff2_2);
            u2_grad_coeff_ref[(q * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj5 * grad_coeff2_2);
            u2_grad_coeff_ref[(q * DIM + 2) * VECTOR_SIZE + lane] = qw * (adj8 * grad_coeff2_2);
            p_value_coeff[q * VECTOR_SIZE + lane] = scalar_t(0);
            p_grad_coeff_ref[(q * DIM + 0) * VECTOR_SIZE + lane] = scalar_t(0);
            p_grad_coeff_ref[(q * DIM + 1) * VECTOR_SIZE + lane] = scalar_t(0);
            p_grad_coeff_ref[(q * DIM + 2) * VECTOR_SIZE + lane] = scalar_t(0);
        }
    }
    scalar_t *const u0_output_streams[U_N_SHAPE] = {output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7], output[8], output[9], output[10], output[11], output[12], output[13], output[14], output[15], output[16], output[17], output[18], output[19], output[20], output[21], output[22], output[23], output[24], output[25], output[26]};
    tensor_integrate<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], u0_value_coeff, u0_grad_coeff_ref, u0_output_streams);
    scalar_t *const u1_output_streams[U_N_SHAPE] = {output[27], output[28], output[29], output[30], output[31], output[32], output[33], output[34], output[35], output[36], output[37], output[38], output[39], output[40], output[41], output[42], output[43], output[44], output[45], output[46], output[47], output[48], output[49], output[50], output[51], output[52], output[53]};
    tensor_integrate<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], u1_value_coeff, u1_grad_coeff_ref, u1_output_streams);
    scalar_t *const u2_output_streams[U_N_SHAPE] = {output[54], output[55], output[56], output[57], output[58], output[59], output[60], output[61], output[62], output[63], output[64], output[65], output[66], output[67], output[68], output[69], output[70], output[71], output[72], output[73], output[74], output[75], output[76], output[77], output[78], output[79], output[80]};
    tensor_integrate<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], u2_value_coeff, u2_grad_coeff_ref, u2_output_streams);
    scalar_t *const p_output_streams[P_N_SHAPE] = {output[81], output[82], output[83], output[84], output[85], output[86], output[87], output[88]};
    tensor_integrate<scalar_t, N_QP, P_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[1], field_grad_1d[1], p_value_coeff, p_grad_coeff_ref, p_output_streams);
}

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void navier_stokes_form_2_u_p_d3_tensor_product_mixed_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT field_shape_1d[2],
        const scalar_t *const SFEM_RESTRICT field_grad_1d[2],
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t direction[89][VECTOR_SIZE],
        scalar_t output[89][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
    static constexpr int N_FIELD_STREAMS = 89;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int U_N_SHAPE = 27;
    static constexpr int P_N_SHAPE = 8;
    static constexpr int N_QP_1D = integer_root(N_QP, DIM);
    static_assert(ipow(N_QP_1D, DIM) == N_QP, "N_QP must be tensor-product compatible");
    static constexpr int U_N_SHAPE_1D = integer_root(U_N_SHAPE, DIM);
    static_assert(ipow(U_N_SHAPE_1D, DIM) == U_N_SHAPE, "U_N_SHAPE must be tensor-product compatible");
    static constexpr int P_N_SHAPE_1D = integer_root(P_N_SHAPE, DIM);
    static_assert(ipow(P_N_SHAPE_1D, DIM) == P_N_SHAPE, "P_N_SHAPE must be tensor-product compatible");
    scalar_t direction_u0_value[N_QP * VECTOR_SIZE];
    tensor_evaluate_value_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], direction + 0, direction_u0_value);
    scalar_t direction_u1_value[N_QP * VECTOR_SIZE];
    tensor_evaluate_value_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], direction + 27, direction_u1_value);
    scalar_t direction_u2_value[N_QP * VECTOR_SIZE];
    tensor_evaluate_value_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], direction + 54, direction_u2_value);
    scalar_t direction_p_value[N_QP * VECTOR_SIZE];
    tensor_evaluate_value_contiguous<scalar_t, N_QP, P_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[1], direction + 81, direction_p_value);
    scalar_t u0_value_coeff[N_QP * VECTOR_SIZE];
    scalar_t u0_grad_coeff_ref[N_QP * DIM * VECTOR_SIZE];
    scalar_t u1_value_coeff[N_QP * VECTOR_SIZE];
    scalar_t u1_grad_coeff_ref[N_QP * DIM * VECTOR_SIZE];
    scalar_t u2_value_coeff[N_QP * VECTOR_SIZE];
    scalar_t u2_grad_coeff_ref[N_QP * DIM * VECTOR_SIZE];
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
            const scalar_t u0_direction = direction_u0_value[q * VECTOR_SIZE + lane];
            const scalar_t u1_direction = direction_u1_value[q * VECTOR_SIZE + lane];
            const scalar_t u2_direction = direction_u2_value[q * VECTOR_SIZE + lane];
            const scalar_t p_direction = direction_p_value[q * VECTOR_SIZE + lane];
            const scalar_t residual_tmp0 = -p_direction;
            const scalar_t grad_coeff0_0 = residual_tmp0;
            const scalar_t grad_coeff1_1 = residual_tmp0;
            const scalar_t grad_coeff2_2 = residual_tmp0;
            u0_value_coeff[q * VECTOR_SIZE + lane] = scalar_t(0);
            u0_grad_coeff_ref[(q * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff0_0);
            u0_grad_coeff_ref[(q * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj3 * grad_coeff0_0);
            u0_grad_coeff_ref[(q * DIM + 2) * VECTOR_SIZE + lane] = qw * (adj6 * grad_coeff0_0);
            u1_value_coeff[q * VECTOR_SIZE + lane] = scalar_t(0);
            u1_grad_coeff_ref[(q * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj1 * grad_coeff1_1);
            u1_grad_coeff_ref[(q * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj4 * grad_coeff1_1);
            u1_grad_coeff_ref[(q * DIM + 2) * VECTOR_SIZE + lane] = qw * (adj7 * grad_coeff1_1);
            u2_value_coeff[q * VECTOR_SIZE + lane] = scalar_t(0);
            u2_grad_coeff_ref[(q * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj2 * grad_coeff2_2);
            u2_grad_coeff_ref[(q * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj5 * grad_coeff2_2);
            u2_grad_coeff_ref[(q * DIM + 2) * VECTOR_SIZE + lane] = qw * (adj8 * grad_coeff2_2);
            p_value_coeff[q * VECTOR_SIZE + lane] = scalar_t(0);
            p_grad_coeff_ref[(q * DIM + 0) * VECTOR_SIZE + lane] = scalar_t(0);
            p_grad_coeff_ref[(q * DIM + 1) * VECTOR_SIZE + lane] = scalar_t(0);
            p_grad_coeff_ref[(q * DIM + 2) * VECTOR_SIZE + lane] = scalar_t(0);
        }
    }
    tensor_integrate_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], u0_value_coeff, u0_grad_coeff_ref, output + 0);
    tensor_integrate_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], u1_value_coeff, u1_grad_coeff_ref, output + 27);
    tensor_integrate_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], u2_value_coeff, u2_grad_coeff_ref, output + 54);
    tensor_integrate_contiguous<scalar_t, N_QP, P_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[1], field_grad_1d[1], p_value_coeff, p_grad_coeff_ref, output + 81);
}

} // namespace codegen
} // namespace sfem

#endif
