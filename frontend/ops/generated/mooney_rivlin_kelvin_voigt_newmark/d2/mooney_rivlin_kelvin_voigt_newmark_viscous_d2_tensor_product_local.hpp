#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_VISCOUS_D2_TENSOR_PRODUCT_LOCAL_HPP
#define MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_VISCOUS_D2_TENSOR_PRODUCT_LOCAL_HPP

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

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d2_tensor_product_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t *const SFEM_RESTRICT current[2 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT previous[2 * N_SHAPE],
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
    scalar_t current_value[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t current_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, current, current_value, current_grad_ref);
    scalar_t previous_value[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t previous_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, previous, previous_value, previous_grad_ref);
    scalar_t value_coeff[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t grad_coeff_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    static constexpr int Q = integer_root(N_QP, DIM);
    for (int q = 0; q < N_QP; ++q) {
        const int qx = q % Q;
        const int qy = q / Q;
        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t det = determinant[geometry_offset];
            const scalar_t adj0 = adjugate[0][geometry_offset];
            const scalar_t adj1 = adjugate[1][geometry_offset];
            const scalar_t adj2 = adjugate[2][geometry_offset];
            const scalar_t adj3 = adjugate[3][geometry_offset];
            const scalar_t u0_grad_0_ref = current_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u0_grad_1_ref = current_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
            const scalar_t u0_old_grad_0_ref = previous_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u0_old_grad_1_ref = previous_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
            const scalar_t u1_grad_0_ref = current_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u1_grad_1_ref = current_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
            const scalar_t u1_old_grad_0_ref = previous_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u1_old_grad_1_ref = previous_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
            const scalar_t residual_tmp0 = u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp1 = pow_m1(residual_tmp0 + u0_grad_0*u1_grad_1 + u0_grad_0 - u0_grad_1*u1_grad_0);
            const scalar_t residual_tmp2 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp3 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp4 = u0_grad_0 + scalar_t(1);
            const scalar_t residual_tmp5 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
            const scalar_t residual_tmp6 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
            const scalar_t residual_tmp7 = eta_s*(-residual_tmp0*residual_tmp6 + residual_tmp2*u0_grad_1 + residual_tmp3*u1_grad_0 - residual_tmp4*residual_tmp5);
            const scalar_t residual_tmp8 = residual_tmp5*u1_grad_0;
            const scalar_t residual_tmp9 = residual_tmp0*residual_tmp2;
            const scalar_t residual_tmp10 = -residual_tmp3*residual_tmp4 + residual_tmp6*u0_grad_1;
            const scalar_t residual_tmp11 = eta_b*(residual_tmp10 + residual_tmp8 - residual_tmp9);
            const scalar_t residual_tmp12 = residual_tmp10 - residual_tmp8 + residual_tmp9;
            const scalar_t residual_tmp13 = -eta_s*residual_tmp12 + residual_tmp11;
            const scalar_t residual_tmp14 = eta_s*residual_tmp12 + residual_tmp11;
            const scalar_t grad_coeff0_0 = residual_tmp1*(-residual_tmp0*residual_tmp13 + residual_tmp7*u0_grad_1);
            const scalar_t grad_coeff0_1 = residual_tmp1*(residual_tmp13*u1_grad_0 - residual_tmp4*residual_tmp7);
            const scalar_t grad_coeff1_0 = residual_tmp1*(-residual_tmp0*residual_tmp7 + residual_tmp14*u0_grad_1);
            const scalar_t grad_coeff1_1 = residual_tmp1*(-residual_tmp14*residual_tmp4 + residual_tmp7*u1_grad_0);
            value_coeff[(0 * N_QP + q) * VECTOR_SIZE + lane] = scalar_t(0);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
            value_coeff[(1 * N_QP + q) * VECTOR_SIZE + lane] = scalar_t(0);
            grad_coeff_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
            grad_coeff_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
        }
    }
    tensor_integrate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d2_tensor_product_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t current[2 * N_SHAPE][VECTOR_SIZE],
        const scalar_t previous[2 * N_SHAPE][VECTOR_SIZE],
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t output[2 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
    scalar_t current_value[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t current_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    tensor_evaluate_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, current, current_value, current_grad_ref);
    scalar_t previous_value[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t previous_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    tensor_evaluate_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, previous, previous_value, previous_grad_ref);
    scalar_t value_coeff[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t grad_coeff_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    static constexpr int Q = integer_root(N_QP, DIM);
    for (int q = 0; q < N_QP; ++q) {
        const int qx = q % Q;
        const int qy = q / Q;
        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t det = determinant[geometry_offset];
            const scalar_t adj0 = adjugate[0][geometry_offset];
            const scalar_t adj1 = adjugate[1][geometry_offset];
            const scalar_t adj2 = adjugate[2][geometry_offset];
            const scalar_t adj3 = adjugate[3][geometry_offset];
            const scalar_t u0_grad_0_ref = current_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u0_grad_1_ref = current_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
            const scalar_t u0_old_grad_0_ref = previous_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u0_old_grad_1_ref = previous_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
            const scalar_t u1_grad_0_ref = current_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u1_grad_1_ref = current_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
            const scalar_t u1_old_grad_0_ref = previous_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u1_old_grad_1_ref = previous_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
            const scalar_t residual_tmp0 = u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp1 = pow_m1(residual_tmp0 + u0_grad_0*u1_grad_1 + u0_grad_0 - u0_grad_1*u1_grad_0);
            const scalar_t residual_tmp2 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp3 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp4 = u0_grad_0 + scalar_t(1);
            const scalar_t residual_tmp5 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
            const scalar_t residual_tmp6 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
            const scalar_t residual_tmp7 = eta_s*(-residual_tmp0*residual_tmp6 + residual_tmp2*u0_grad_1 + residual_tmp3*u1_grad_0 - residual_tmp4*residual_tmp5);
            const scalar_t residual_tmp8 = residual_tmp5*u1_grad_0;
            const scalar_t residual_tmp9 = residual_tmp0*residual_tmp2;
            const scalar_t residual_tmp10 = -residual_tmp3*residual_tmp4 + residual_tmp6*u0_grad_1;
            const scalar_t residual_tmp11 = eta_b*(residual_tmp10 + residual_tmp8 - residual_tmp9);
            const scalar_t residual_tmp12 = residual_tmp10 - residual_tmp8 + residual_tmp9;
            const scalar_t residual_tmp13 = -eta_s*residual_tmp12 + residual_tmp11;
            const scalar_t residual_tmp14 = eta_s*residual_tmp12 + residual_tmp11;
            const scalar_t grad_coeff0_0 = residual_tmp1*(-residual_tmp0*residual_tmp13 + residual_tmp7*u0_grad_1);
            const scalar_t grad_coeff0_1 = residual_tmp1*(residual_tmp13*u1_grad_0 - residual_tmp4*residual_tmp7);
            const scalar_t grad_coeff1_0 = residual_tmp1*(-residual_tmp0*residual_tmp7 + residual_tmp14*u0_grad_1);
            const scalar_t grad_coeff1_1 = residual_tmp1*(-residual_tmp14*residual_tmp4 + residual_tmp7*u1_grad_0);
            value_coeff[(0 * N_QP + q) * VECTOR_SIZE + lane] = scalar_t(0);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
            value_coeff[(1 * N_QP + q) * VECTOR_SIZE + lane] = scalar_t(0);
            grad_coeff_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
            grad_coeff_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
        }
    }
    tensor_integrate_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d2_tensor_product_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t *const SFEM_RESTRICT current[2 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT previous[2 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT direction[2 * N_SHAPE],
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
    scalar_t current_value[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t current_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, current, current_value, current_grad_ref);
    scalar_t previous_value[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t previous_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, previous, previous_value, previous_grad_ref);
    scalar_t direction_value[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t direction_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, direction, direction_value, direction_grad_ref);
    scalar_t value_coeff[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t grad_coeff_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    static constexpr int Q = integer_root(N_QP, DIM);
    for (int q = 0; q < N_QP; ++q) {
        const int qx = q % Q;
        const int qy = q / Q;
        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t det = determinant[geometry_offset];
            const scalar_t adj0 = adjugate[0][geometry_offset];
            const scalar_t adj1 = adjugate[1][geometry_offset];
            const scalar_t adj2 = adjugate[2][geometry_offset];
            const scalar_t adj3 = adjugate[3][geometry_offset];
            const scalar_t u0_grad_0_ref = current_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u0_grad_1_ref = current_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
            const scalar_t u0_old_grad_0_ref = previous_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u0_old_grad_1_ref = previous_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
            const scalar_t u0_direction_grad_0_ref = direction_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u0_direction_grad_1_ref = direction_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
            const scalar_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
            const scalar_t u1_grad_0_ref = current_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u1_grad_1_ref = current_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
            const scalar_t u1_old_grad_0_ref = previous_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u1_old_grad_1_ref = previous_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
            const scalar_t u1_direction_grad_0_ref = direction_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u1_direction_grad_1_ref = direction_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj2) / det;
            const scalar_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj3) / det;
            const scalar_t residual_tmp0 = u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp1 = residual_tmp0 + u0_grad_0*u1_grad_1 + u0_grad_0 - u0_grad_1*u1_grad_0;
            const scalar_t residual_tmp2 = pow_m1(residual_tmp1);
            const scalar_t residual_tmp3 = newmark_velocity_alpha*residual_tmp0;
            const scalar_t residual_tmp4 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp5 = -residual_tmp3 + residual_tmp4;
            const scalar_t residual_tmp6 = eta_s*residual_tmp5;
            const scalar_t residual_tmp7 = eta_s*u0_old_grad_1;
            const scalar_t residual_tmp8 = newmark_velocity_alpha*u0_grad_1;
            const scalar_t residual_tmp9 = eta_b*(scalar_t(2)*residual_tmp8 + u0_old_grad_1);
            const scalar_t residual_tmp10 = residual_tmp7 + residual_tmp9;
            const scalar_t residual_tmp11 = pow_m2(residual_tmp1);
            const scalar_t residual_tmp12 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp13 = u0_grad_0 + scalar_t(1);
            const scalar_t residual_tmp14 = residual_tmp8 + u0_old_grad_1;
            const scalar_t residual_tmp15 = newmark_velocity_alpha*u1_grad_0;
            const scalar_t residual_tmp16 = residual_tmp15 + u1_old_grad_0;
            const scalar_t residual_tmp17 = eta_s*(-residual_tmp0*residual_tmp16 + residual_tmp12*u0_grad_1 - residual_tmp13*residual_tmp14 + residual_tmp4*u1_grad_0);
            const scalar_t residual_tmp18 = residual_tmp14*u1_grad_0;
            const scalar_t residual_tmp19 = residual_tmp0*residual_tmp12;
            const scalar_t residual_tmp20 = -residual_tmp13*residual_tmp4 + residual_tmp16*u0_grad_1;
            const scalar_t residual_tmp21 = eta_b*(residual_tmp18 - residual_tmp19 + residual_tmp20);
            const scalar_t residual_tmp22 = eta_s*(-residual_tmp18 + residual_tmp19 + residual_tmp20);
            const scalar_t residual_tmp23 = residual_tmp21 - residual_tmp22;
            const scalar_t residual_tmp24 = residual_tmp11*(-residual_tmp0*residual_tmp23 + residual_tmp17*u0_grad_1);
            const scalar_t residual_tmp25 = eta_b*(-residual_tmp3 - residual_tmp4);
            const scalar_t residual_tmp26 = -residual_tmp5;
            const scalar_t residual_tmp27 = -eta_s*residual_tmp26 + residual_tmp25;
            const scalar_t residual_tmp28 = -residual_tmp0;
            const scalar_t residual_tmp29 = newmark_velocity_alpha*residual_tmp13;
            const scalar_t residual_tmp30 = residual_tmp12 - residual_tmp29;
            const scalar_t residual_tmp31 = eta_s*residual_tmp30;
            const scalar_t residual_tmp32 = eta_b*(scalar_t(2)*residual_tmp15 + u1_old_grad_0);
            const scalar_t residual_tmp33 = -eta_s*u1_old_grad_0 + residual_tmp32;
            const scalar_t residual_tmp34 = eta_b*(-residual_tmp12 - residual_tmp29);
            const scalar_t residual_tmp35 = -eta_s*residual_tmp30 + residual_tmp34;
            const scalar_t residual_tmp36 = eta_s*u1_old_grad_0;
            const scalar_t residual_tmp37 = -residual_tmp13;
            const scalar_t residual_tmp38 = residual_tmp11*(-residual_tmp13*residual_tmp17 + residual_tmp23*u1_grad_0);
            const scalar_t residual_tmp39 = -residual_tmp17;
            const scalar_t residual_tmp40 = -residual_tmp7 + residual_tmp9;
            const scalar_t residual_tmp41 = residual_tmp21 + residual_tmp22;
            const scalar_t residual_tmp42 = residual_tmp11*(-residual_tmp0*residual_tmp17 + residual_tmp41*u0_grad_1);
            const scalar_t residual_tmp43 = eta_s*residual_tmp26 + residual_tmp25;
            const scalar_t residual_tmp44 = residual_tmp31 + residual_tmp34;
            const scalar_t residual_tmp45 = residual_tmp32 + residual_tmp36;
            const scalar_t residual_tmp46 = residual_tmp11*(-residual_tmp13*residual_tmp41 + residual_tmp17*u1_grad_0);
            const scalar_t grad_coeff0_0 = u0_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp27 - residual_tmp7*u0_grad_1) + residual_tmp24*residual_tmp28) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp33 + residual_tmp17 + residual_tmp31*u0_grad_1) + residual_tmp24*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp10 + residual_tmp6*u0_grad_1) + residual_tmp24*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp35 - residual_tmp23 - residual_tmp36*u0_grad_1) + residual_tmp24*residual_tmp37);
            const scalar_t grad_coeff0_1 = u0_direction_grad_0*(residual_tmp2*(residual_tmp13*residual_tmp7 + residual_tmp27*u1_grad_0 + residual_tmp39) + residual_tmp28*residual_tmp38) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp31 + residual_tmp33*u1_grad_0) + residual_tmp38*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(residual_tmp10*u1_grad_0 - residual_tmp13*residual_tmp6 + residual_tmp23) + residual_tmp38*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(residual_tmp13*residual_tmp36 + residual_tmp35*u1_grad_0) + residual_tmp37*residual_tmp38);
            const scalar_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp2*(residual_tmp0*residual_tmp7 + residual_tmp43*u0_grad_1) + residual_tmp28*residual_tmp42) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp31 + residual_tmp41 + residual_tmp45*u0_grad_1) + residual_tmp42*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp6 + residual_tmp40*u0_grad_1) + residual_tmp42*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(residual_tmp0*residual_tmp36 + residual_tmp39 + residual_tmp44*u0_grad_1) + residual_tmp37*residual_tmp42);
            const scalar_t grad_coeff1_1 = u0_direction_grad_0*(residual_tmp2*(-residual_tmp13*residual_tmp43 - residual_tmp41 - residual_tmp7*u1_grad_0) + residual_tmp28*residual_tmp46) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp45 + residual_tmp31*u1_grad_0) + residual_tmp46*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp13*residual_tmp40 + residual_tmp17 + residual_tmp6*u1_grad_0) + residual_tmp46*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp44 - residual_tmp36*u1_grad_0) + residual_tmp37*residual_tmp46);
            value_coeff[(0 * N_QP + q) * VECTOR_SIZE + lane] = scalar_t(0);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
            value_coeff[(1 * N_QP + q) * VECTOR_SIZE + lane] = scalar_t(0);
            grad_coeff_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
            grad_coeff_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
        }
    }
    tensor_integrate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d2_tensor_product_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t current[2 * N_SHAPE][VECTOR_SIZE],
        const scalar_t previous[2 * N_SHAPE][VECTOR_SIZE],
        const scalar_t direction[2 * N_SHAPE][VECTOR_SIZE],
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t output[2 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
    scalar_t current_value[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t current_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    tensor_evaluate_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, current, current_value, current_grad_ref);
    scalar_t previous_value[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t previous_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    tensor_evaluate_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, previous, previous_value, previous_grad_ref);
    scalar_t direction_value[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t direction_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    tensor_evaluate_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, direction, direction_value, direction_grad_ref);
    scalar_t value_coeff[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t grad_coeff_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    static constexpr int Q = integer_root(N_QP, DIM);
    for (int q = 0; q < N_QP; ++q) {
        const int qx = q % Q;
        const int qy = q / Q;
        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t det = determinant[geometry_offset];
            const scalar_t adj0 = adjugate[0][geometry_offset];
            const scalar_t adj1 = adjugate[1][geometry_offset];
            const scalar_t adj2 = adjugate[2][geometry_offset];
            const scalar_t adj3 = adjugate[3][geometry_offset];
            const scalar_t u0_grad_0_ref = current_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u0_grad_1_ref = current_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
            const scalar_t u0_old_grad_0_ref = previous_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u0_old_grad_1_ref = previous_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
            const scalar_t u0_direction_grad_0_ref = direction_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u0_direction_grad_1_ref = direction_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
            const scalar_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
            const scalar_t u1_grad_0_ref = current_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u1_grad_1_ref = current_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
            const scalar_t u1_old_grad_0_ref = previous_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u1_old_grad_1_ref = previous_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
            const scalar_t u1_direction_grad_0_ref = direction_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u1_direction_grad_1_ref = direction_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj2) / det;
            const scalar_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj3) / det;
            const scalar_t residual_tmp0 = u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp1 = residual_tmp0 + u0_grad_0*u1_grad_1 + u0_grad_0 - u0_grad_1*u1_grad_0;
            const scalar_t residual_tmp2 = pow_m1(residual_tmp1);
            const scalar_t residual_tmp3 = newmark_velocity_alpha*residual_tmp0;
            const scalar_t residual_tmp4 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp5 = -residual_tmp3 + residual_tmp4;
            const scalar_t residual_tmp6 = eta_s*residual_tmp5;
            const scalar_t residual_tmp7 = eta_s*u0_old_grad_1;
            const scalar_t residual_tmp8 = newmark_velocity_alpha*u0_grad_1;
            const scalar_t residual_tmp9 = eta_b*(scalar_t(2)*residual_tmp8 + u0_old_grad_1);
            const scalar_t residual_tmp10 = residual_tmp7 + residual_tmp9;
            const scalar_t residual_tmp11 = pow_m2(residual_tmp1);
            const scalar_t residual_tmp12 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp13 = u0_grad_0 + scalar_t(1);
            const scalar_t residual_tmp14 = residual_tmp8 + u0_old_grad_1;
            const scalar_t residual_tmp15 = newmark_velocity_alpha*u1_grad_0;
            const scalar_t residual_tmp16 = residual_tmp15 + u1_old_grad_0;
            const scalar_t residual_tmp17 = eta_s*(-residual_tmp0*residual_tmp16 + residual_tmp12*u0_grad_1 - residual_tmp13*residual_tmp14 + residual_tmp4*u1_grad_0);
            const scalar_t residual_tmp18 = residual_tmp14*u1_grad_0;
            const scalar_t residual_tmp19 = residual_tmp0*residual_tmp12;
            const scalar_t residual_tmp20 = -residual_tmp13*residual_tmp4 + residual_tmp16*u0_grad_1;
            const scalar_t residual_tmp21 = eta_b*(residual_tmp18 - residual_tmp19 + residual_tmp20);
            const scalar_t residual_tmp22 = eta_s*(-residual_tmp18 + residual_tmp19 + residual_tmp20);
            const scalar_t residual_tmp23 = residual_tmp21 - residual_tmp22;
            const scalar_t residual_tmp24 = residual_tmp11*(-residual_tmp0*residual_tmp23 + residual_tmp17*u0_grad_1);
            const scalar_t residual_tmp25 = eta_b*(-residual_tmp3 - residual_tmp4);
            const scalar_t residual_tmp26 = -residual_tmp5;
            const scalar_t residual_tmp27 = -eta_s*residual_tmp26 + residual_tmp25;
            const scalar_t residual_tmp28 = -residual_tmp0;
            const scalar_t residual_tmp29 = newmark_velocity_alpha*residual_tmp13;
            const scalar_t residual_tmp30 = residual_tmp12 - residual_tmp29;
            const scalar_t residual_tmp31 = eta_s*residual_tmp30;
            const scalar_t residual_tmp32 = eta_b*(scalar_t(2)*residual_tmp15 + u1_old_grad_0);
            const scalar_t residual_tmp33 = -eta_s*u1_old_grad_0 + residual_tmp32;
            const scalar_t residual_tmp34 = eta_b*(-residual_tmp12 - residual_tmp29);
            const scalar_t residual_tmp35 = -eta_s*residual_tmp30 + residual_tmp34;
            const scalar_t residual_tmp36 = eta_s*u1_old_grad_0;
            const scalar_t residual_tmp37 = -residual_tmp13;
            const scalar_t residual_tmp38 = residual_tmp11*(-residual_tmp13*residual_tmp17 + residual_tmp23*u1_grad_0);
            const scalar_t residual_tmp39 = -residual_tmp17;
            const scalar_t residual_tmp40 = -residual_tmp7 + residual_tmp9;
            const scalar_t residual_tmp41 = residual_tmp21 + residual_tmp22;
            const scalar_t residual_tmp42 = residual_tmp11*(-residual_tmp0*residual_tmp17 + residual_tmp41*u0_grad_1);
            const scalar_t residual_tmp43 = eta_s*residual_tmp26 + residual_tmp25;
            const scalar_t residual_tmp44 = residual_tmp31 + residual_tmp34;
            const scalar_t residual_tmp45 = residual_tmp32 + residual_tmp36;
            const scalar_t residual_tmp46 = residual_tmp11*(-residual_tmp13*residual_tmp41 + residual_tmp17*u1_grad_0);
            const scalar_t grad_coeff0_0 = u0_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp27 - residual_tmp7*u0_grad_1) + residual_tmp24*residual_tmp28) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp33 + residual_tmp17 + residual_tmp31*u0_grad_1) + residual_tmp24*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp10 + residual_tmp6*u0_grad_1) + residual_tmp24*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp35 - residual_tmp23 - residual_tmp36*u0_grad_1) + residual_tmp24*residual_tmp37);
            const scalar_t grad_coeff0_1 = u0_direction_grad_0*(residual_tmp2*(residual_tmp13*residual_tmp7 + residual_tmp27*u1_grad_0 + residual_tmp39) + residual_tmp28*residual_tmp38) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp31 + residual_tmp33*u1_grad_0) + residual_tmp38*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(residual_tmp10*u1_grad_0 - residual_tmp13*residual_tmp6 + residual_tmp23) + residual_tmp38*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(residual_tmp13*residual_tmp36 + residual_tmp35*u1_grad_0) + residual_tmp37*residual_tmp38);
            const scalar_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp2*(residual_tmp0*residual_tmp7 + residual_tmp43*u0_grad_1) + residual_tmp28*residual_tmp42) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp31 + residual_tmp41 + residual_tmp45*u0_grad_1) + residual_tmp42*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp6 + residual_tmp40*u0_grad_1) + residual_tmp42*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(residual_tmp0*residual_tmp36 + residual_tmp39 + residual_tmp44*u0_grad_1) + residual_tmp37*residual_tmp42);
            const scalar_t grad_coeff1_1 = u0_direction_grad_0*(residual_tmp2*(-residual_tmp13*residual_tmp43 - residual_tmp41 - residual_tmp7*u1_grad_0) + residual_tmp28*residual_tmp46) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp45 + residual_tmp31*u1_grad_0) + residual_tmp46*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp13*residual_tmp40 + residual_tmp17 + residual_tmp6*u1_grad_0) + residual_tmp46*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp44 - residual_tmp36*u1_grad_0) + residual_tmp37*residual_tmp46);
            value_coeff[(0 * N_QP + q) * VECTOR_SIZE + lane] = scalar_t(0);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
            value_coeff[(1 * N_QP + q) * VECTOR_SIZE + lane] = scalar_t(0);
            grad_coeff_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
            grad_coeff_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
        }
    }
    tensor_integrate_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

} // namespace codegen
} // namespace sfem

#endif
