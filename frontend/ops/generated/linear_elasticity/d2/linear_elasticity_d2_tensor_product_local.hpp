#ifndef LINEAR_ELASTICITY_D2_TENSOR_PRODUCT_LOCAL_HPP
#define LINEAR_ELASTICITY_D2_TENSOR_PRODUCT_LOCAL_HPP
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
static SFEM_INLINE void linear_elasticity_d2_tensor_product_objective_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t mu,
        const scalar_t lmbda,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 2],
        scalar_t *const SFEM_RESTRICT value
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
    static constexpr int N_QP_1D = integer_root(N_QP, 2);
    static constexpr int N_SHAPE_1D = integer_root(N_SHAPE, 2);
    static_assert(ipow(N_QP_1D, 2) == N_QP, "N_QP must be tensor-product compatible");
    static_assert(ipow(N_SHAPE_1D, 2) == N_SHAPE, "N_SHAPE must be tensor-product compatible");
    scalar_t grad_u_ref_q[N_QP * 4 * VECTOR_SIZE];
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 2>(nelems, shape_1d, grad_1d, u_streams, 0, &grad_u_ref_q[0 * N_QP * 2 * VECTOR_SIZE]);
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 2>(nelems, shape_1d, grad_1d, u_streams, 1, &grad_u_ref_q[1 * N_QP * 2 * VECTOR_SIZE]);
    for (int q = 0; q < N_QP; ++q) {
        const int qx = q % N_QP_1D;
        const int qy = q / N_QP_1D;
        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const scalar_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const scalar_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const scalar_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            scalar_t grad_u_ref[4];
            grad_u_ref[0] = grad_u_ref_q[((0 * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[1] = grad_u_ref_q[((0 * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[2] = grad_u_ref_q[((1 * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[3] = grad_u_ref_q[((1 * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane];
            scalar_t grad_u[4];
            const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
            grad_u[0] = (grad_u_ref[0] * jacobian_adjugate_lane0 + grad_u_ref[1] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
            grad_u[1] = (grad_u_ref[0] * jacobian_adjugate_lane1 + grad_u_ref[1] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
            grad_u[2] = (grad_u_ref[2] * jacobian_adjugate_lane0 + grad_u_ref[3] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
            grad_u[3] = (grad_u_ref[2] * jacobian_adjugate_lane1 + grad_u_ref[3] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        value[lane] += qw * jacobian_determinant_lane0 * (((scalar_t(1) / scalar_t(2)))*lmbda*pow_2(grad_u[0] + grad_u[3]) + mu*(pow_2(grad_u[0]) + pow_2(grad_u[3]) + scalar_t(2)*pow_2(((scalar_t(1) / scalar_t(2)))*grad_u[1] + ((scalar_t(1) / scalar_t(2)))*grad_u[2])));
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void linear_elasticity_d2_tensor_product_gradient_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t mu,
        const scalar_t lmbda,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 2],
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 2]
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
    static constexpr int N_QP_1D = integer_root(N_QP, 2);
    static constexpr int N_SHAPE_1D = integer_root(N_SHAPE, 2);
    static_assert(ipow(N_QP_1D, 2) == N_QP, "N_QP must be tensor-product compatible");
    static_assert(ipow(N_SHAPE_1D, 2) == N_SHAPE, "N_SHAPE must be tensor-product compatible");
    scalar_t grad_u_ref_q[N_QP * 4 * VECTOR_SIZE];
    scalar_t loperand_q[N_QP * 4 * VECTOR_SIZE];
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 2>(nelems, shape_1d, grad_1d, u_streams, 0, &grad_u_ref_q[0 * N_QP * 2 * VECTOR_SIZE]);
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 2>(nelems, shape_1d, grad_1d, u_streams, 1, &grad_u_ref_q[1 * N_QP * 2 * VECTOR_SIZE]);
    for (int q = 0; q < N_QP; ++q) {
        const int qx = q % N_QP_1D;
        const int qy = q / N_QP_1D;
        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const scalar_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const scalar_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const scalar_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            scalar_t grad_u_ref[4];
            grad_u_ref[0] = grad_u_ref_q[((0 * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[1] = grad_u_ref_q[((0 * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[2] = grad_u_ref_q[((1 * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[3] = grad_u_ref_q[((1 * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane];
            scalar_t grad_u[4];
            const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
            grad_u[0] = (grad_u_ref[0] * jacobian_adjugate_lane0 + grad_u_ref[1] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
            grad_u[1] = (grad_u_ref[0] * jacobian_adjugate_lane1 + grad_u_ref[1] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
            grad_u[2] = (grad_u_ref[2] * jacobian_adjugate_lane0 + grad_u_ref[3] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
            grad_u[3] = (grad_u_ref[2] * jacobian_adjugate_lane1 + grad_u_ref[3] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
            scalar_t loperand[4];
        scalar_t material[4];
        const scalar_t weak_mat_tmp0 = scalar_t(2)*grad_u[0];
        const scalar_t weak_mat_tmp1 = scalar_t(2)*grad_u[3];
        const scalar_t weak_mat_tmp2 = ((scalar_t(1) / scalar_t(2)))*lmbda*(weak_mat_tmp0 + weak_mat_tmp1);
        const scalar_t weak_mat_tmp3 = mu*(grad_u[1] + grad_u[2]);
        material[0] = mu*weak_mat_tmp0 + weak_mat_tmp2;
        material[1] = weak_mat_tmp3;
        material[2] = weak_mat_tmp3;
        material[3] = mu*weak_mat_tmp1 + weak_mat_tmp2;
        loperand[0] = qw * (material[0] * jacobian_adjugate_lane0 + material[1] * jacobian_adjugate_lane1);
        loperand[1] = qw * (material[0] * jacobian_adjugate_lane2 + material[1] * jacobian_adjugate_lane3);
        loperand[2] = qw * (material[2] * jacobian_adjugate_lane0 + material[3] * jacobian_adjugate_lane1);
        loperand[3] = qw * (material[2] * jacobian_adjugate_lane2 + material[3] * jacobian_adjugate_lane3);
            loperand_q[((0 * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane] = loperand[0];
            loperand_q[((0 * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane] = loperand[1];
            loperand_q[((1 * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane] = loperand[2];
            loperand_q[((1 * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane] = loperand[3];
        }
    }
    tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 2>(nelems, shape_1d, grad_1d, &loperand_q[0 * N_QP * 2 * VECTOR_SIZE], out_streams, 0);
    tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 2>(nelems, shape_1d, grad_1d, &loperand_q[1 * N_QP * 2 * VECTOR_SIZE], out_streams, 1);
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void linear_elasticity_d2_tensor_product_apply_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t mu,
        const scalar_t lmbda,
        const scalar_t *const SFEM_RESTRICT h_streams[N_SHAPE * 2],
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 2]
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
    static constexpr int N_QP_1D = integer_root(N_QP, 2);
    static constexpr int N_SHAPE_1D = integer_root(N_SHAPE, 2);
    static_assert(ipow(N_QP_1D, 2) == N_QP, "N_QP must be tensor-product compatible");
    static_assert(ipow(N_SHAPE_1D, 2) == N_SHAPE, "N_SHAPE must be tensor-product compatible");
    scalar_t grad_h_ref_q[N_QP * 4 * VECTOR_SIZE];
    scalar_t loperand_q[N_QP * 4 * VECTOR_SIZE];
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 2>(nelems, shape_1d, grad_1d, h_streams, 0, &grad_h_ref_q[0 * N_QP * 2 * VECTOR_SIZE]);
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 2>(nelems, shape_1d, grad_1d, h_streams, 1, &grad_h_ref_q[1 * N_QP * 2 * VECTOR_SIZE]);
    for (int q = 0; q < N_QP; ++q) {
        const int qx = q % N_QP_1D;
        const int qy = q / N_QP_1D;
        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const scalar_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const scalar_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const scalar_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            scalar_t grad_h_ref[4];
            grad_h_ref[0] = grad_h_ref_q[((0 * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane];
            grad_h_ref[1] = grad_h_ref_q[((0 * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane];
            grad_h_ref[2] = grad_h_ref_q[((1 * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane];
            grad_h_ref[3] = grad_h_ref_q[((1 * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane];
            scalar_t trial_grad[4];
            const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
            trial_grad[0] = (grad_h_ref[0] * jacobian_adjugate_lane0 + grad_h_ref[1] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
            trial_grad[1] = (grad_h_ref[0] * jacobian_adjugate_lane1 + grad_h_ref[1] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
            trial_grad[2] = (grad_h_ref[2] * jacobian_adjugate_lane0 + grad_h_ref[3] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
            trial_grad[3] = (grad_h_ref[2] * jacobian_adjugate_lane1 + grad_h_ref[3] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
            scalar_t loperand[4];
        scalar_t material[4];
        const scalar_t weak_mat_tmp0 = scalar_t(2)*trial_grad[0];
        const scalar_t weak_mat_tmp1 = scalar_t(2)*trial_grad[3];
        const scalar_t weak_mat_tmp2 = ((scalar_t(1) / scalar_t(2)))*lmbda*(weak_mat_tmp0 + weak_mat_tmp1);
        const scalar_t weak_mat_tmp3 = mu*(trial_grad[1] + trial_grad[2]);
        material[0] = mu*weak_mat_tmp0 + weak_mat_tmp2;
        material[1] = weak_mat_tmp3;
        material[2] = weak_mat_tmp3;
        material[3] = mu*weak_mat_tmp1 + weak_mat_tmp2;
        loperand[0] = qw * (material[0] * jacobian_adjugate_lane0 + material[1] * jacobian_adjugate_lane1);
        loperand[1] = qw * (material[0] * jacobian_adjugate_lane2 + material[1] * jacobian_adjugate_lane3);
        loperand[2] = qw * (material[2] * jacobian_adjugate_lane0 + material[3] * jacobian_adjugate_lane1);
        loperand[3] = qw * (material[2] * jacobian_adjugate_lane2 + material[3] * jacobian_adjugate_lane3);
            loperand_q[((0 * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane] = loperand[0];
            loperand_q[((0 * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane] = loperand[1];
            loperand_q[((1 * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane] = loperand[2];
            loperand_q[((1 * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane] = loperand[3];
        }
    }
    tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 2>(nelems, shape_1d, grad_1d, &loperand_q[0 * N_QP * 2 * VECTOR_SIZE], out_streams, 0);
    tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 2>(nelems, shape_1d, grad_1d, &loperand_q[1 * N_QP * 2 * VECTOR_SIZE], out_streams, 1);
}

} // namespace codegen
} // namespace sfem

#endif
