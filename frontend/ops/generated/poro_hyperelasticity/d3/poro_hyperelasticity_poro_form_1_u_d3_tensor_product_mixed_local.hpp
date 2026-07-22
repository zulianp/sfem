#ifndef PORO_HYPERELASTICITY_PORO_FORM_1_U_D3_TENSOR_PRODUCT_MIXED_LOCAL_HPP
#define PORO_HYPERELASTICITY_PORO_FORM_1_U_D3_TENSOR_PRODUCT_MIXED_LOCAL_HPP

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
static SFEM_INLINE void poro_hyperelasticity_poro_form_1_u_d3_tensor_product_mixed_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT field_shape_1d[2],
        const scalar_t *const SFEM_RESTRICT field_grad_1d[2],
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t *const SFEM_RESTRICT current[89],
        const scalar_t alpha,
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
    scalar_t current_u0_value[N_QP * VECTOR_SIZE];
    const scalar_t *const current_u0_streams[U_N_SHAPE] = {current[0], current[1], current[2], current[3], current[4], current[5], current[6], current[7], current[8], current[9], current[10], current[11], current[12], current[13], current[14], current[15], current[16], current[17], current[18], current[19], current[20], current[21], current[22], current[23], current[24], current[25], current[26]};
    tensor_evaluate_value<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], current_u0_streams, current_u0_value);
    scalar_t current_u1_value[N_QP * VECTOR_SIZE];
    const scalar_t *const current_u1_streams[U_N_SHAPE] = {current[27], current[28], current[29], current[30], current[31], current[32], current[33], current[34], current[35], current[36], current[37], current[38], current[39], current[40], current[41], current[42], current[43], current[44], current[45], current[46], current[47], current[48], current[49], current[50], current[51], current[52], current[53]};
    tensor_evaluate_value<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], current_u1_streams, current_u1_value);
    scalar_t current_u2_value[N_QP * VECTOR_SIZE];
    const scalar_t *const current_u2_streams[U_N_SHAPE] = {current[54], current[55], current[56], current[57], current[58], current[59], current[60], current[61], current[62], current[63], current[64], current[65], current[66], current[67], current[68], current[69], current[70], current[71], current[72], current[73], current[74], current[75], current[76], current[77], current[78], current[79], current[80]};
    tensor_evaluate_value<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], current_u2_streams, current_u2_value);
    scalar_t current_p_value[N_QP * VECTOR_SIZE];
    const scalar_t *const current_p_streams[P_N_SHAPE] = {current[81], current[82], current[83], current[84], current[85], current[86], current[87], current[88]};
    tensor_evaluate_value<scalar_t, N_QP, P_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[1], current_p_streams, current_p_value);
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
            const scalar_t u0 = current_u0_value[q * VECTOR_SIZE + lane];
            const scalar_t u1 = current_u1_value[q * VECTOR_SIZE + lane];
            const scalar_t u2 = current_u2_value[q * VECTOR_SIZE + lane];
            const scalar_t p = current_p_value[q * VECTOR_SIZE + lane];
            const scalar_t residual_tmp0 = -alpha*p;
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
static SFEM_INLINE void poro_hyperelasticity_poro_form_1_u_d3_tensor_product_mixed_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT field_shape_1d[2],
        const scalar_t *const SFEM_RESTRICT field_grad_1d[2],
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t current[89][VECTOR_SIZE],
        const scalar_t alpha,
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
    scalar_t current_u0_value[N_QP * VECTOR_SIZE];
    const scalar_t *const current_u0_streams[U_N_SHAPE] = {current[0], current[1], current[2], current[3], current[4], current[5], current[6], current[7], current[8], current[9], current[10], current[11], current[12], current[13], current[14], current[15], current[16], current[17], current[18], current[19], current[20], current[21], current[22], current[23], current[24], current[25], current[26]};
    tensor_evaluate_value_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], current_u0_streams, current_u0_value);
    scalar_t current_u1_value[N_QP * VECTOR_SIZE];
    const scalar_t *const current_u1_streams[U_N_SHAPE] = {current[27], current[28], current[29], current[30], current[31], current[32], current[33], current[34], current[35], current[36], current[37], current[38], current[39], current[40], current[41], current[42], current[43], current[44], current[45], current[46], current[47], current[48], current[49], current[50], current[51], current[52], current[53]};
    tensor_evaluate_value_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], current_u1_streams, current_u1_value);
    scalar_t current_u2_value[N_QP * VECTOR_SIZE];
    const scalar_t *const current_u2_streams[U_N_SHAPE] = {current[54], current[55], current[56], current[57], current[58], current[59], current[60], current[61], current[62], current[63], current[64], current[65], current[66], current[67], current[68], current[69], current[70], current[71], current[72], current[73], current[74], current[75], current[76], current[77], current[78], current[79], current[80]};
    tensor_evaluate_value_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], current_u2_streams, current_u2_value);
    scalar_t current_p_value[N_QP * VECTOR_SIZE];
    const scalar_t *const current_p_streams[P_N_SHAPE] = {current[81], current[82], current[83], current[84], current[85], current[86], current[87], current[88]};
    tensor_evaluate_value_contiguous<scalar_t, N_QP, P_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[1], current_p_streams, current_p_value);
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
            const scalar_t u0 = current_u0_value[q * VECTOR_SIZE + lane];
            const scalar_t u1 = current_u1_value[q * VECTOR_SIZE + lane];
            const scalar_t u2 = current_u2_value[q * VECTOR_SIZE + lane];
            const scalar_t p = current_p_value[q * VECTOR_SIZE + lane];
            const scalar_t residual_tmp0 = -alpha*p;
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
    tensor_integrate_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], u0_value_coeff, u0_grad_coeff_ref, u0_output_streams);
    scalar_t *const u1_output_streams[U_N_SHAPE] = {output[27], output[28], output[29], output[30], output[31], output[32], output[33], output[34], output[35], output[36], output[37], output[38], output[39], output[40], output[41], output[42], output[43], output[44], output[45], output[46], output[47], output[48], output[49], output[50], output[51], output[52], output[53]};
    tensor_integrate_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], u1_value_coeff, u1_grad_coeff_ref, u1_output_streams);
    scalar_t *const u2_output_streams[U_N_SHAPE] = {output[54], output[55], output[56], output[57], output[58], output[59], output[60], output[61], output[62], output[63], output[64], output[65], output[66], output[67], output[68], output[69], output[70], output[71], output[72], output[73], output[74], output[75], output[76], output[77], output[78], output[79], output[80]};
    tensor_integrate_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], u2_value_coeff, u2_grad_coeff_ref, u2_output_streams);
    scalar_t *const p_output_streams[P_N_SHAPE] = {output[81], output[82], output[83], output[84], output[85], output[86], output[87], output[88]};
    tensor_integrate_contiguous<scalar_t, N_QP, P_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[1], field_grad_1d[1], p_value_coeff, p_grad_coeff_ref, p_output_streams);
}

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void poro_hyperelasticity_poro_form_1_u_d3_tensor_product_mixed_jacobian_action_block(
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
static SFEM_INLINE void poro_hyperelasticity_poro_form_1_u_d3_tensor_product_mixed_jacobian_action_block_contiguous(
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

} // namespace codegen
} // namespace sfem

#endif
