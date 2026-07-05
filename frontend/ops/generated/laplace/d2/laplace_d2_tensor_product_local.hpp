#ifndef LAPLACE_D2_TENSOR_PRODUCT_LOCAL_HPP
#define LAPLACE_D2_TENSOR_PRODUCT_LOCAL_HPP

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

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void laplace_d2_tensor_product_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t *const SFEM_RESTRICT current[1 * N_SHAPE],
        const scalar_t kappa,
        scalar_t *const SFEM_RESTRICT output[1 * N_SHAPE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 1;
    scalar_t current_value[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t current_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, current, current_value, current_grad_ref);
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
            const scalar_t u_grad_0_ref = current_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u_grad_1_ref = current_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u_grad_0 = (u_grad_0_ref * adj0 + u_grad_1_ref * adj2) / det;
            const scalar_t u_grad_1 = (u_grad_0_ref * adj1 + u_grad_1_ref * adj3) / det;
            const scalar_t grad_coeff0_0 = kappa*u_grad_0;
            const scalar_t grad_coeff0_1 = kappa*u_grad_1;
            value_coeff[(0 * N_QP + q) * VECTOR_SIZE + lane] = scalar_t(0);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
        }
    }
    tensor_integrate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void laplace_d2_tensor_product_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t current[1 * N_SHAPE][VECTOR_SIZE],
        const scalar_t kappa,
        scalar_t output[1 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 1;
    scalar_t current_value[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t current_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    tensor_evaluate_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, current, current_value, current_grad_ref);
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
            const scalar_t u_grad_0_ref = current_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u_grad_1_ref = current_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u_grad_0 = (u_grad_0_ref * adj0 + u_grad_1_ref * adj2) / det;
            const scalar_t u_grad_1 = (u_grad_0_ref * adj1 + u_grad_1_ref * adj3) / det;
            const scalar_t grad_coeff0_0 = kappa*u_grad_0;
            const scalar_t grad_coeff0_1 = kappa*u_grad_1;
            value_coeff[(0 * N_QP + q) * VECTOR_SIZE + lane] = scalar_t(0);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
        }
    }
    tensor_integrate_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void laplace_d2_tensor_product_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t *const SFEM_RESTRICT direction[1 * N_SHAPE],
        const scalar_t kappa,
        scalar_t *const SFEM_RESTRICT output[1 * N_SHAPE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 1;
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
            const scalar_t u_direction_grad_0_ref = direction_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u_direction_grad_1_ref = direction_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u_direction_grad_0 = (u_direction_grad_0_ref * adj0 + u_direction_grad_1_ref * adj2) / det;
            const scalar_t u_direction_grad_1 = (u_direction_grad_0_ref * adj1 + u_direction_grad_1_ref * adj3) / det;
            const scalar_t grad_coeff0_0 = kappa*u_direction_grad_0;
            const scalar_t grad_coeff0_1 = kappa*u_direction_grad_1;
            value_coeff[(0 * N_QP + q) * VECTOR_SIZE + lane] = scalar_t(0);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
        }
    }
    tensor_integrate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void laplace_d2_tensor_product_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t direction[1 * N_SHAPE][VECTOR_SIZE],
        const scalar_t kappa,
        scalar_t output[1 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 1;
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
            const scalar_t u_direction_grad_0_ref = direction_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t u_direction_grad_1_ref = direction_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t u_direction_grad_0 = (u_direction_grad_0_ref * adj0 + u_direction_grad_1_ref * adj2) / det;
            const scalar_t u_direction_grad_1 = (u_direction_grad_0_ref * adj1 + u_direction_grad_1_ref * adj3) / det;
            const scalar_t grad_coeff0_0 = kappa*u_direction_grad_0;
            const scalar_t grad_coeff0_1 = kappa*u_direction_grad_1;
            value_coeff[(0 * N_QP + q) * VECTOR_SIZE + lane] = scalar_t(0);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
        }
    }
    tensor_integrate_contiguous<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

} // namespace codegen
} // namespace sfem

#endif
