#ifndef NAVIER_STOKES_FORM_2_U_U_D3_TENSOR_PRODUCT_LOCAL_HPP
#define NAVIER_STOKES_FORM_2_U_U_D3_TENSOR_PRODUCT_LOCAL_HPP

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
static SFEM_INLINE void navier_stokes_form_2_u_u_d3_tensor_product_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT field_shape_1d[1],
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        scalar_t *const SFEM_RESTRICT output[81]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 81;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int U_N_SHAPE = 27;
    static constexpr int N_QP_1D = integer_root(N_QP, DIM);
    static_assert(ipow(N_QP_1D, DIM) == N_QP, "N_QP must be tensor-product compatible");
    static constexpr int U_N_SHAPE_1D = integer_root(U_N_SHAPE, DIM);
    static_assert(ipow(U_N_SHAPE_1D, DIM) == U_N_SHAPE, "U_N_SHAPE must be tensor-product compatible");
}

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void navier_stokes_form_2_u_u_d3_tensor_product_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT field_shape_1d[1],
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        scalar_t output[81][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 81;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int U_N_SHAPE = 27;
    static constexpr int N_QP_1D = integer_root(N_QP, DIM);
    static_assert(ipow(N_QP_1D, DIM) == N_QP, "N_QP must be tensor-product compatible");
    static constexpr int U_N_SHAPE_1D = integer_root(U_N_SHAPE, DIM);
    static_assert(ipow(U_N_SHAPE_1D, DIM) == U_N_SHAPE, "U_N_SHAPE must be tensor-product compatible");
}

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void navier_stokes_form_2_u_u_d3_tensor_product_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT field_shape_1d[1],
        const scalar_t *const SFEM_RESTRICT field_grad_1d[1],
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t *const SFEM_RESTRICT previous[81],
        const scalar_t *const SFEM_RESTRICT direction[81],
        const scalar_t convection_scale,
        const scalar_t dt,
        const scalar_t nu,
        const scalar_t rho,
        scalar_t *const SFEM_RESTRICT output[81]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 81;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int U_N_SHAPE = 27;
    static constexpr int N_QP_1D = integer_root(N_QP, DIM);
    static_assert(ipow(N_QP_1D, DIM) == N_QP, "N_QP must be tensor-product compatible");
    static constexpr int U_N_SHAPE_1D = integer_root(U_N_SHAPE, DIM);
    static_assert(ipow(U_N_SHAPE_1D, DIM) == U_N_SHAPE, "U_N_SHAPE must be tensor-product compatible");
    scalar_t previous_u0_value[N_QP * VECTOR_SIZE];
    const scalar_t *const previous_u0_streams[U_N_SHAPE] = {previous[0], previous[1], previous[2], previous[3], previous[4], previous[5], previous[6], previous[7], previous[8], previous[9], previous[10], previous[11], previous[12], previous[13], previous[14], previous[15], previous[16], previous[17], previous[18], previous[19], previous[20], previous[21], previous[22], previous[23], previous[24], previous[25], previous[26]};
    tensor_evaluate_value<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], previous_u0_streams, previous_u0_value);
    scalar_t direction_u0_value[N_QP * VECTOR_SIZE];
    scalar_t direction_u0_grad_ref[N_QP * DIM * VECTOR_SIZE];
    const scalar_t *const direction_u0_streams[U_N_SHAPE] = {direction[0], direction[1], direction[2], direction[3], direction[4], direction[5], direction[6], direction[7], direction[8], direction[9], direction[10], direction[11], direction[12], direction[13], direction[14], direction[15], direction[16], direction[17], direction[18], direction[19], direction[20], direction[21], direction[22], direction[23], direction[24], direction[25], direction[26]};
    tensor_evaluate<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], direction_u0_streams, direction_u0_value, direction_u0_grad_ref);
    scalar_t previous_u1_value[N_QP * VECTOR_SIZE];
    const scalar_t *const previous_u1_streams[U_N_SHAPE] = {previous[27], previous[28], previous[29], previous[30], previous[31], previous[32], previous[33], previous[34], previous[35], previous[36], previous[37], previous[38], previous[39], previous[40], previous[41], previous[42], previous[43], previous[44], previous[45], previous[46], previous[47], previous[48], previous[49], previous[50], previous[51], previous[52], previous[53]};
    tensor_evaluate_value<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], previous_u1_streams, previous_u1_value);
    scalar_t direction_u1_value[N_QP * VECTOR_SIZE];
    scalar_t direction_u1_grad_ref[N_QP * DIM * VECTOR_SIZE];
    const scalar_t *const direction_u1_streams[U_N_SHAPE] = {direction[27], direction[28], direction[29], direction[30], direction[31], direction[32], direction[33], direction[34], direction[35], direction[36], direction[37], direction[38], direction[39], direction[40], direction[41], direction[42], direction[43], direction[44], direction[45], direction[46], direction[47], direction[48], direction[49], direction[50], direction[51], direction[52], direction[53]};
    tensor_evaluate<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], direction_u1_streams, direction_u1_value, direction_u1_grad_ref);
    scalar_t previous_u2_value[N_QP * VECTOR_SIZE];
    const scalar_t *const previous_u2_streams[U_N_SHAPE] = {previous[54], previous[55], previous[56], previous[57], previous[58], previous[59], previous[60], previous[61], previous[62], previous[63], previous[64], previous[65], previous[66], previous[67], previous[68], previous[69], previous[70], previous[71], previous[72], previous[73], previous[74], previous[75], previous[76], previous[77], previous[78], previous[79], previous[80]};
    tensor_evaluate_value<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], previous_u2_streams, previous_u2_value);
    scalar_t direction_u2_value[N_QP * VECTOR_SIZE];
    scalar_t direction_u2_grad_ref[N_QP * DIM * VECTOR_SIZE];
    const scalar_t *const direction_u2_streams[U_N_SHAPE] = {direction[54], direction[55], direction[56], direction[57], direction[58], direction[59], direction[60], direction[61], direction[62], direction[63], direction[64], direction[65], direction[66], direction[67], direction[68], direction[69], direction[70], direction[71], direction[72], direction[73], direction[74], direction[75], direction[76], direction[77], direction[78], direction[79], direction[80]};
    tensor_evaluate<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], direction_u2_streams, direction_u2_value, direction_u2_grad_ref);
    scalar_t u0_value_coeff[N_QP * VECTOR_SIZE];
    scalar_t u0_grad_coeff_ref[N_QP * DIM * VECTOR_SIZE];
    scalar_t u1_value_coeff[N_QP * VECTOR_SIZE];
    scalar_t u1_grad_coeff_ref[N_QP * DIM * VECTOR_SIZE];
    scalar_t u2_value_coeff[N_QP * VECTOR_SIZE];
    scalar_t u2_grad_coeff_ref[N_QP * DIM * VECTOR_SIZE];
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
            const scalar_t u0_old = previous_u0_value[q * VECTOR_SIZE + lane];
            const scalar_t u0_direction = direction_u0_value[q * VECTOR_SIZE + lane];
            const scalar_t u0_direction_grad_0_ref = direction_u0_grad_ref[(q * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u0_direction_grad_1_ref = direction_u0_grad_ref[(q * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u0_direction_grad_2_ref = direction_u0_grad_ref[(q * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
            const scalar_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
            const scalar_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
            const scalar_t u1_old = previous_u1_value[q * VECTOR_SIZE + lane];
            const scalar_t u1_direction = direction_u1_value[q * VECTOR_SIZE + lane];
            const scalar_t u1_direction_grad_0_ref = direction_u1_grad_ref[(q * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u1_direction_grad_1_ref = direction_u1_grad_ref[(q * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u1_direction_grad_2_ref = direction_u1_grad_ref[(q * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
            const scalar_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
            const scalar_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
            const scalar_t u2_old = previous_u2_value[q * VECTOR_SIZE + lane];
            const scalar_t u2_direction = direction_u2_value[q * VECTOR_SIZE + lane];
            const scalar_t u2_direction_grad_0_ref = direction_u2_grad_ref[(q * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u2_direction_grad_1_ref = direction_u2_grad_ref[(q * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u2_direction_grad_2_ref = direction_u2_grad_ref[(q * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
            const scalar_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
            const scalar_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
            const scalar_t residual_tmp0 = convection_scale*rho;
            const scalar_t residual_tmp1 = residual_tmp0*u0_old;
            const scalar_t residual_tmp2 = residual_tmp0*u1_old;
            const scalar_t residual_tmp3 = residual_tmp0*u2_old;
            const scalar_t residual_tmp4 = rho/dt;
            const scalar_t residual_tmp5 = nu*rho;
            const scalar_t residual_tmp6 = scalar_t(2)*residual_tmp5;
            const scalar_t residual_tmp7 = residual_tmp5*u0_direction_grad_1 + residual_tmp5*u1_direction_grad_0;
            const scalar_t residual_tmp8 = residual_tmp5*u0_direction_grad_2 + residual_tmp5*u2_direction_grad_0;
            const scalar_t residual_tmp9 = residual_tmp5*u1_direction_grad_2 + residual_tmp5*u2_direction_grad_1;
            const scalar_t value_coeff0 = residual_tmp1*u0_direction_grad_0 + residual_tmp2*u0_direction_grad_1 + residual_tmp3*u0_direction_grad_2 + residual_tmp4*u0_direction;
            const scalar_t grad_coeff0_0 = residual_tmp6*u0_direction_grad_0;
            const scalar_t grad_coeff0_1 = residual_tmp7;
            const scalar_t grad_coeff0_2 = residual_tmp8;
            const scalar_t value_coeff1 = residual_tmp1*u1_direction_grad_0 + residual_tmp2*u1_direction_grad_1 + residual_tmp3*u1_direction_grad_2 + residual_tmp4*u1_direction;
            const scalar_t grad_coeff1_0 = residual_tmp7;
            const scalar_t grad_coeff1_1 = residual_tmp6*u1_direction_grad_1;
            const scalar_t grad_coeff1_2 = residual_tmp9;
            const scalar_t value_coeff2 = residual_tmp1*u2_direction_grad_0 + residual_tmp2*u2_direction_grad_1 + residual_tmp3*u2_direction_grad_2 + residual_tmp4*u2_direction;
            const scalar_t grad_coeff2_0 = residual_tmp8;
            const scalar_t grad_coeff2_1 = residual_tmp9;
            const scalar_t grad_coeff2_2 = residual_tmp6*u2_direction_grad_2;
            u0_value_coeff[q * VECTOR_SIZE + lane] = qw * det * value_coeff0;
            u0_grad_coeff_ref[(q * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1 + adj2 * grad_coeff0_2);
            u0_grad_coeff_ref[(q * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj3 * grad_coeff0_0 + adj4 * grad_coeff0_1 + adj5 * grad_coeff0_2);
            u0_grad_coeff_ref[(q * DIM + 2) * VECTOR_SIZE + lane] = qw * (adj6 * grad_coeff0_0 + adj7 * grad_coeff0_1 + adj8 * grad_coeff0_2);
            u1_value_coeff[q * VECTOR_SIZE + lane] = qw * det * value_coeff1;
            u1_grad_coeff_ref[(q * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1 + adj2 * grad_coeff1_2);
            u1_grad_coeff_ref[(q * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj3 * grad_coeff1_0 + adj4 * grad_coeff1_1 + adj5 * grad_coeff1_2);
            u1_grad_coeff_ref[(q * DIM + 2) * VECTOR_SIZE + lane] = qw * (adj6 * grad_coeff1_0 + adj7 * grad_coeff1_1 + adj8 * grad_coeff1_2);
            u2_value_coeff[q * VECTOR_SIZE + lane] = qw * det * value_coeff2;
            u2_grad_coeff_ref[(q * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff2_0 + adj1 * grad_coeff2_1 + adj2 * grad_coeff2_2);
            u2_grad_coeff_ref[(q * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj3 * grad_coeff2_0 + adj4 * grad_coeff2_1 + adj5 * grad_coeff2_2);
            u2_grad_coeff_ref[(q * DIM + 2) * VECTOR_SIZE + lane] = qw * (adj6 * grad_coeff2_0 + adj7 * grad_coeff2_1 + adj8 * grad_coeff2_2);
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
}

template <typename scalar_t, int N_QP, int CELL_N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void navier_stokes_form_2_u_u_d3_tensor_product_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT field_shape_1d[1],
        const scalar_t *const SFEM_RESTRICT field_grad_1d[1],
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t previous[81][VECTOR_SIZE],
        const scalar_t direction[81][VECTOR_SIZE],
        const scalar_t convection_scale,
        const scalar_t dt,
        const scalar_t nu,
        const scalar_t rho,
        scalar_t output[81][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 1;
    static constexpr int N_FIELD_STREAMS = 81;
    (void)CELL_N_SHAPE;
    (void)N_FIELD_STREAMS;
    static constexpr int U_N_SHAPE = 27;
    static constexpr int N_QP_1D = integer_root(N_QP, DIM);
    static_assert(ipow(N_QP_1D, DIM) == N_QP, "N_QP must be tensor-product compatible");
    static constexpr int U_N_SHAPE_1D = integer_root(U_N_SHAPE, DIM);
    static_assert(ipow(U_N_SHAPE_1D, DIM) == U_N_SHAPE, "U_N_SHAPE must be tensor-product compatible");
    scalar_t previous_u0_value[N_QP * VECTOR_SIZE];
    tensor_evaluate_value_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], previous + 0, previous_u0_value);
    scalar_t direction_u0_value[N_QP * VECTOR_SIZE];
    scalar_t direction_u0_grad_ref[N_QP * DIM * VECTOR_SIZE];
    tensor_evaluate_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], direction + 0, direction_u0_value, direction_u0_grad_ref);
    scalar_t previous_u1_value[N_QP * VECTOR_SIZE];
    tensor_evaluate_value_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], previous + 27, previous_u1_value);
    scalar_t direction_u1_value[N_QP * VECTOR_SIZE];
    scalar_t direction_u1_grad_ref[N_QP * DIM * VECTOR_SIZE];
    tensor_evaluate_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], direction + 27, direction_u1_value, direction_u1_grad_ref);
    scalar_t previous_u2_value[N_QP * VECTOR_SIZE];
    tensor_evaluate_value_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], previous + 54, previous_u2_value);
    scalar_t direction_u2_value[N_QP * VECTOR_SIZE];
    scalar_t direction_u2_grad_ref[N_QP * DIM * VECTOR_SIZE];
    tensor_evaluate_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], direction + 54, direction_u2_value, direction_u2_grad_ref);
    scalar_t u0_value_coeff[N_QP * VECTOR_SIZE];
    scalar_t u0_grad_coeff_ref[N_QP * DIM * VECTOR_SIZE];
    scalar_t u1_value_coeff[N_QP * VECTOR_SIZE];
    scalar_t u1_grad_coeff_ref[N_QP * DIM * VECTOR_SIZE];
    scalar_t u2_value_coeff[N_QP * VECTOR_SIZE];
    scalar_t u2_grad_coeff_ref[N_QP * DIM * VECTOR_SIZE];
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
            const scalar_t u0_old = previous_u0_value[q * VECTOR_SIZE + lane];
            const scalar_t u0_direction = direction_u0_value[q * VECTOR_SIZE + lane];
            const scalar_t u0_direction_grad_0_ref = direction_u0_grad_ref[(q * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u0_direction_grad_1_ref = direction_u0_grad_ref[(q * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u0_direction_grad_2_ref = direction_u0_grad_ref[(q * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
            const scalar_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
            const scalar_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
            const scalar_t u1_old = previous_u1_value[q * VECTOR_SIZE + lane];
            const scalar_t u1_direction = direction_u1_value[q * VECTOR_SIZE + lane];
            const scalar_t u1_direction_grad_0_ref = direction_u1_grad_ref[(q * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u1_direction_grad_1_ref = direction_u1_grad_ref[(q * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u1_direction_grad_2_ref = direction_u1_grad_ref[(q * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
            const scalar_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
            const scalar_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
            const scalar_t u2_old = previous_u2_value[q * VECTOR_SIZE + lane];
            const scalar_t u2_direction = direction_u2_value[q * VECTOR_SIZE + lane];
            const scalar_t u2_direction_grad_0_ref = direction_u2_grad_ref[(q * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u2_direction_grad_1_ref = direction_u2_grad_ref[(q * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u2_direction_grad_2_ref = direction_u2_grad_ref[(q * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
            const scalar_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
            const scalar_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
            const scalar_t residual_tmp0 = convection_scale*rho;
            const scalar_t residual_tmp1 = residual_tmp0*u0_old;
            const scalar_t residual_tmp2 = residual_tmp0*u1_old;
            const scalar_t residual_tmp3 = residual_tmp0*u2_old;
            const scalar_t residual_tmp4 = rho/dt;
            const scalar_t residual_tmp5 = nu*rho;
            const scalar_t residual_tmp6 = scalar_t(2)*residual_tmp5;
            const scalar_t residual_tmp7 = residual_tmp5*u0_direction_grad_1 + residual_tmp5*u1_direction_grad_0;
            const scalar_t residual_tmp8 = residual_tmp5*u0_direction_grad_2 + residual_tmp5*u2_direction_grad_0;
            const scalar_t residual_tmp9 = residual_tmp5*u1_direction_grad_2 + residual_tmp5*u2_direction_grad_1;
            const scalar_t value_coeff0 = residual_tmp1*u0_direction_grad_0 + residual_tmp2*u0_direction_grad_1 + residual_tmp3*u0_direction_grad_2 + residual_tmp4*u0_direction;
            const scalar_t grad_coeff0_0 = residual_tmp6*u0_direction_grad_0;
            const scalar_t grad_coeff0_1 = residual_tmp7;
            const scalar_t grad_coeff0_2 = residual_tmp8;
            const scalar_t value_coeff1 = residual_tmp1*u1_direction_grad_0 + residual_tmp2*u1_direction_grad_1 + residual_tmp3*u1_direction_grad_2 + residual_tmp4*u1_direction;
            const scalar_t grad_coeff1_0 = residual_tmp7;
            const scalar_t grad_coeff1_1 = residual_tmp6*u1_direction_grad_1;
            const scalar_t grad_coeff1_2 = residual_tmp9;
            const scalar_t value_coeff2 = residual_tmp1*u2_direction_grad_0 + residual_tmp2*u2_direction_grad_1 + residual_tmp3*u2_direction_grad_2 + residual_tmp4*u2_direction;
            const scalar_t grad_coeff2_0 = residual_tmp8;
            const scalar_t grad_coeff2_1 = residual_tmp9;
            const scalar_t grad_coeff2_2 = residual_tmp6*u2_direction_grad_2;
            u0_value_coeff[q * VECTOR_SIZE + lane] = qw * det * value_coeff0;
            u0_grad_coeff_ref[(q * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1 + adj2 * grad_coeff0_2);
            u0_grad_coeff_ref[(q * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj3 * grad_coeff0_0 + adj4 * grad_coeff0_1 + adj5 * grad_coeff0_2);
            u0_grad_coeff_ref[(q * DIM + 2) * VECTOR_SIZE + lane] = qw * (adj6 * grad_coeff0_0 + adj7 * grad_coeff0_1 + adj8 * grad_coeff0_2);
            u1_value_coeff[q * VECTOR_SIZE + lane] = qw * det * value_coeff1;
            u1_grad_coeff_ref[(q * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1 + adj2 * grad_coeff1_2);
            u1_grad_coeff_ref[(q * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj3 * grad_coeff1_0 + adj4 * grad_coeff1_1 + adj5 * grad_coeff1_2);
            u1_grad_coeff_ref[(q * DIM + 2) * VECTOR_SIZE + lane] = qw * (adj6 * grad_coeff1_0 + adj7 * grad_coeff1_1 + adj8 * grad_coeff1_2);
            u2_value_coeff[q * VECTOR_SIZE + lane] = qw * det * value_coeff2;
            u2_grad_coeff_ref[(q * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff2_0 + adj1 * grad_coeff2_1 + adj2 * grad_coeff2_2);
            u2_grad_coeff_ref[(q * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj3 * grad_coeff2_0 + adj4 * grad_coeff2_1 + adj5 * grad_coeff2_2);
            u2_grad_coeff_ref[(q * DIM + 2) * VECTOR_SIZE + lane] = qw * (adj6 * grad_coeff2_0 + adj7 * grad_coeff2_1 + adj8 * grad_coeff2_2);
        }
    }
    tensor_integrate_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], u0_value_coeff, u0_grad_coeff_ref, output + 0);
    tensor_integrate_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], u1_value_coeff, u1_grad_coeff_ref, output + 27);
    tensor_integrate_contiguous<scalar_t, N_QP, U_N_SHAPE, VECTOR_SIZE, DIM, 1>(
            nelems, field_shape_1d[0], field_grad_1d[0], u2_value_coeff, u2_grad_coeff_ref, output + 54);
}

} // namespace codegen
} // namespace sfem

#endif
