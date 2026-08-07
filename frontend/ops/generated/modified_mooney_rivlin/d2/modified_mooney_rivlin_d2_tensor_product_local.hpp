#ifndef MODIFIED_MOONEY_RIVLIN_D2_TENSOR_PRODUCT_LOCAL_HPP
#define MODIFIED_MOONEY_RIVLIN_D2_TENSOR_PRODUCT_LOCAL_HPP
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
static SFEM_INLINE void modified_mooney_rivlin_d2_tensor_product_objective_block(
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
        const scalar_t c1,
        const scalar_t c2,
        const scalar_t kappa,
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
        const scalar_t weak_obj_tmp0 = grad_u[0] + scalar_t(1);
        const scalar_t weak_obj_tmp1 = grad_u[3] + scalar_t(1);
        const scalar_t weak_obj_tmp2 = -grad_u[1]*grad_u[2] + weak_obj_tmp0*weak_obj_tmp1;
        const scalar_t weak_obj_tmp3 = pow_2(grad_u[1]) + pow_2(weak_obj_tmp1);
        const scalar_t weak_obj_tmp4 = pow_2(grad_u[2]) + pow_2(weak_obj_tmp0);
        const scalar_t weak_obj_tmp5 = weak_obj_tmp3 + weak_obj_tmp4;
        value[lane] += qw * jacobian_determinant_lane0 * (c1*(scalar_t(-2) + weak_obj_tmp5/pow(weak_obj_tmp2, (scalar_t(2) / scalar_t(3)))) + c2*(scalar_t(-1) + (-(scalar_t(1) / scalar_t(2))*pow_2(weak_obj_tmp3) - (scalar_t(1) / scalar_t(2))*pow_2(weak_obj_tmp4) + ((scalar_t(1) / scalar_t(2)))*pow_2(weak_obj_tmp5) - pow_2(grad_u[1]*weak_obj_tmp0 + grad_u[2]*weak_obj_tmp1))/pow(weak_obj_tmp2, (scalar_t(4) / scalar_t(3)))) + ((scalar_t(1) / scalar_t(2)))*kappa*pow_2(log(weak_obj_tmp2)));
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void modified_mooney_rivlin_d2_tensor_product_gradient_block(
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
        const scalar_t c1,
        const scalar_t c2,
        const scalar_t kappa,
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
        const scalar_t weak_mat_tmp0 = grad_u[3] + scalar_t(1);
        const scalar_t weak_mat_tmp1 = grad_u[0] + scalar_t(1);
        const scalar_t weak_mat_tmp2 = -grad_u[1]*grad_u[2] + weak_mat_tmp0*weak_mat_tmp1;
        const scalar_t weak_mat_tmp3 = kappa*log(weak_mat_tmp2)/weak_mat_tmp2;
        const scalar_t weak_mat_tmp4 = pow(weak_mat_tmp2, (scalar_t(-2) / scalar_t(3)));
        const scalar_t weak_mat_tmp5 = scalar_t(2)*weak_mat_tmp1;
        const scalar_t weak_mat_tmp6 = pow_2(grad_u[2]) + pow_2(weak_mat_tmp1);
        const scalar_t weak_mat_tmp7 = pow_2(grad_u[1]) + pow_2(weak_mat_tmp0);
        const scalar_t weak_mat_tmp8 = weak_mat_tmp6 + weak_mat_tmp7;
        const scalar_t weak_mat_tmp9 = ((scalar_t(2) / scalar_t(3)))*weak_mat_tmp8/pow(weak_mat_tmp2, (scalar_t(5) / scalar_t(3)));
        const scalar_t weak_mat_tmp10 = pow(weak_mat_tmp2, (scalar_t(-4) / scalar_t(3)));
        const scalar_t weak_mat_tmp11 = grad_u[1]*weak_mat_tmp1 + grad_u[2]*weak_mat_tmp0;
        const scalar_t weak_mat_tmp12 = scalar_t(2)*grad_u[1];
        const scalar_t weak_mat_tmp13 = ((scalar_t(4) / scalar_t(3)))*(-pow_2(weak_mat_tmp11) - (scalar_t(1) / scalar_t(2))*pow_2(weak_mat_tmp6) - (scalar_t(1) / scalar_t(2))*pow_2(weak_mat_tmp7) + ((scalar_t(1) / scalar_t(2)))*pow_2(weak_mat_tmp8))/pow(weak_mat_tmp2, (scalar_t(7) / scalar_t(3)));
        const scalar_t weak_mat_tmp14 = scalar_t(2)*grad_u[2];
        const scalar_t weak_mat_tmp15 = scalar_t(2)*weak_mat_tmp0;
        material[0] = c1*(-weak_mat_tmp0*weak_mat_tmp9 + weak_mat_tmp4*weak_mat_tmp5) + c2*(-weak_mat_tmp0*weak_mat_tmp13 + weak_mat_tmp10*(scalar_t(2)*weak_mat_tmp1*weak_mat_tmp8 - weak_mat_tmp11*weak_mat_tmp12 - weak_mat_tmp5*weak_mat_tmp6)) + weak_mat_tmp0*weak_mat_tmp3;
        material[1] = c1*(grad_u[2]*weak_mat_tmp9 + weak_mat_tmp12*weak_mat_tmp4) + c2*(grad_u[2]*weak_mat_tmp13 + weak_mat_tmp10*(scalar_t(2)*grad_u[1]*weak_mat_tmp8 - weak_mat_tmp11*weak_mat_tmp5 - weak_mat_tmp12*weak_mat_tmp7)) - grad_u[2]*weak_mat_tmp3;
        material[2] = c1*(grad_u[1]*weak_mat_tmp9 + weak_mat_tmp14*weak_mat_tmp4) + c2*(grad_u[1]*weak_mat_tmp13 + weak_mat_tmp10*(scalar_t(2)*grad_u[2]*weak_mat_tmp8 - weak_mat_tmp11*weak_mat_tmp15 - weak_mat_tmp14*weak_mat_tmp6)) - grad_u[1]*weak_mat_tmp3;
        material[3] = c1*(scalar_t(2)*weak_mat_tmp0*weak_mat_tmp4 - weak_mat_tmp1*weak_mat_tmp9) + c2*(-weak_mat_tmp1*weak_mat_tmp13 + weak_mat_tmp10*(scalar_t(2)*weak_mat_tmp0*weak_mat_tmp8 - weak_mat_tmp11*weak_mat_tmp14 - weak_mat_tmp15*weak_mat_tmp7)) + weak_mat_tmp1*weak_mat_tmp3;
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
static SFEM_INLINE void modified_mooney_rivlin_d2_tensor_product_apply_block(
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
        const scalar_t c1,
        const scalar_t c2,
        const scalar_t kappa,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 2],
        const scalar_t *const SFEM_RESTRICT h_streams[N_SHAPE * 2],
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 2]
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
    static constexpr int N_QP_1D = integer_root(N_QP, 2);
    static constexpr int N_SHAPE_1D = integer_root(N_SHAPE, 2);
    static_assert(ipow(N_QP_1D, 2) == N_QP, "N_QP must be tensor-product compatible");
    static_assert(ipow(N_SHAPE_1D, 2) == N_SHAPE, "N_SHAPE must be tensor-product compatible");
    scalar_t grad_u_ref_q[N_QP * 4 * VECTOR_SIZE];
    scalar_t grad_h_ref_q[N_QP * 4 * VECTOR_SIZE];
    scalar_t loperand_q[N_QP * 4 * VECTOR_SIZE];
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 2>(nelems, shape_1d, grad_1d, u_streams, 0, &grad_u_ref_q[0 * N_QP * 2 * VECTOR_SIZE]);
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 2>(nelems, shape_1d, grad_1d, h_streams, 0, &grad_h_ref_q[0 * N_QP * 2 * VECTOR_SIZE]);
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 2>(nelems, shape_1d, grad_1d, u_streams, 1, &grad_u_ref_q[1 * N_QP * 2 * VECTOR_SIZE]);
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
            scalar_t grad_u_ref[4];
            grad_u_ref[0] = grad_u_ref_q[((0 * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[1] = grad_u_ref_q[((0 * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[2] = grad_u_ref_q[((1 * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[3] = grad_u_ref_q[((1 * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane];
            scalar_t grad_h_ref[4];
            grad_h_ref[0] = grad_h_ref_q[((0 * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane];
            grad_h_ref[1] = grad_h_ref_q[((0 * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane];
            grad_h_ref[2] = grad_h_ref_q[((1 * N_QP + q) * 2 + 0) * VECTOR_SIZE + lane];
            grad_h_ref[3] = grad_h_ref_q[((1 * N_QP + q) * 2 + 1) * VECTOR_SIZE + lane];
            scalar_t grad_u[4];
            scalar_t trial_grad[4];
            const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
            grad_u[0] = (grad_u_ref[0] * jacobian_adjugate_lane0 + grad_u_ref[1] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
            trial_grad[0] = (grad_h_ref[0] * jacobian_adjugate_lane0 + grad_h_ref[1] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
            grad_u[1] = (grad_u_ref[0] * jacobian_adjugate_lane1 + grad_u_ref[1] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
            trial_grad[1] = (grad_h_ref[0] * jacobian_adjugate_lane1 + grad_h_ref[1] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
            grad_u[2] = (grad_u_ref[2] * jacobian_adjugate_lane0 + grad_u_ref[3] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
            trial_grad[2] = (grad_h_ref[2] * jacobian_adjugate_lane0 + grad_h_ref[3] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
            grad_u[3] = (grad_u_ref[2] * jacobian_adjugate_lane1 + grad_u_ref[3] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
            trial_grad[3] = (grad_h_ref[2] * jacobian_adjugate_lane1 + grad_h_ref[3] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
            scalar_t loperand[4];
        scalar_t material[4];
        const scalar_t weak_mat_tmp0 = grad_u[3] + scalar_t(1);
        const scalar_t weak_mat_tmp1 = pow_2(weak_mat_tmp0);
        const scalar_t weak_mat_tmp2 = grad_u[1]*grad_u[2];
        const scalar_t weak_mat_tmp3 = grad_u[0] + scalar_t(1);
        const scalar_t weak_mat_tmp4 = weak_mat_tmp0*weak_mat_tmp3 - weak_mat_tmp2;
        const scalar_t weak_mat_tmp5 = kappa/pow_2(weak_mat_tmp4);
        const scalar_t weak_mat_tmp6 = weak_mat_tmp1*weak_mat_tmp5;
        const scalar_t weak_mat_tmp7 = log(weak_mat_tmp4);
        const scalar_t weak_mat_tmp8 = pow_2(grad_u[2]);
        const scalar_t weak_mat_tmp9 = pow_2(weak_mat_tmp3);
        const scalar_t weak_mat_tmp10 = weak_mat_tmp8 + weak_mat_tmp9;
        const scalar_t weak_mat_tmp11 = pow_2(grad_u[1]);
        const scalar_t weak_mat_tmp12 = weak_mat_tmp1 + weak_mat_tmp11;
        const scalar_t weak_mat_tmp13 = weak_mat_tmp10 + weak_mat_tmp12;
        const scalar_t weak_mat_tmp14 = pow(weak_mat_tmp4, (scalar_t(-8) / scalar_t(3)));
        const scalar_t weak_mat_tmp15 = ((scalar_t(10) / scalar_t(9)))*weak_mat_tmp13*weak_mat_tmp14;
        const scalar_t weak_mat_tmp16 = scalar_t(2)/pow(weak_mat_tmp4, (scalar_t(2) / scalar_t(3)));
        const scalar_t weak_mat_tmp17 = weak_mat_tmp0*weak_mat_tmp3;
        const scalar_t weak_mat_tmp18 = pow(weak_mat_tmp4, (scalar_t(-5) / scalar_t(3)));
        const scalar_t weak_mat_tmp19 = ((scalar_t(8) / scalar_t(3)))*weak_mat_tmp18;
        const scalar_t weak_mat_tmp20 = weak_mat_tmp16 - weak_mat_tmp17*weak_mat_tmp19;
        const scalar_t weak_mat_tmp21 = pow(weak_mat_tmp4, (scalar_t(-4) / scalar_t(3)));
        const scalar_t weak_mat_tmp22 = scalar_t(2)*weak_mat_tmp21;
        const scalar_t weak_mat_tmp23 = grad_u[1]*weak_mat_tmp3;
        const scalar_t weak_mat_tmp24 = grad_u[2]*weak_mat_tmp0;
        const scalar_t weak_mat_tmp25 = weak_mat_tmp23 + weak_mat_tmp24;
        const scalar_t weak_mat_tmp26 = scalar_t(2)*grad_u[1];
        const scalar_t weak_mat_tmp27 = scalar_t(2)*weak_mat_tmp3;
        const scalar_t weak_mat_tmp28 = -weak_mat_tmp10*weak_mat_tmp27 + scalar_t(2)*weak_mat_tmp13*weak_mat_tmp3 - weak_mat_tmp25*weak_mat_tmp26;
        const scalar_t weak_mat_tmp29 = pow(weak_mat_tmp4, (scalar_t(-7) / scalar_t(3)));
        const scalar_t weak_mat_tmp30 = ((scalar_t(8) / scalar_t(3)))*weak_mat_tmp29;
        const scalar_t weak_mat_tmp31 = -(scalar_t(1) / scalar_t(2))*pow_2(weak_mat_tmp10) - (scalar_t(1) / scalar_t(2))*pow_2(weak_mat_tmp12) + ((scalar_t(1) / scalar_t(2)))*pow_2(weak_mat_tmp13) - pow_2(weak_mat_tmp25);
        const scalar_t weak_mat_tmp32 = pow(weak_mat_tmp4, (scalar_t(-10) / scalar_t(3)));
        const scalar_t weak_mat_tmp33 = ((scalar_t(28) / scalar_t(9)))*weak_mat_tmp31*weak_mat_tmp32;
        const scalar_t weak_mat_tmp34 = weak_mat_tmp24*weak_mat_tmp5;
        const scalar_t weak_mat_tmp35 = ((scalar_t(4) / scalar_t(3)))*weak_mat_tmp18;
        const scalar_t weak_mat_tmp36 = grad_u[1]*weak_mat_tmp0;
        const scalar_t weak_mat_tmp37 = weak_mat_tmp35*weak_mat_tmp36;
        const scalar_t weak_mat_tmp38 = grad_u[2]*weak_mat_tmp3;
        const scalar_t weak_mat_tmp39 = weak_mat_tmp35*weak_mat_tmp38;
        const scalar_t weak_mat_tmp40 = scalar_t(2)*grad_u[1]*weak_mat_tmp13 - weak_mat_tmp12*weak_mat_tmp26 - weak_mat_tmp25*weak_mat_tmp27;
        const scalar_t weak_mat_tmp41 = ((scalar_t(4) / scalar_t(3)))*weak_mat_tmp29;
        const scalar_t weak_mat_tmp42 = weak_mat_tmp0*weak_mat_tmp41;
        const scalar_t weak_mat_tmp43 = c1*(-weak_mat_tmp15*weak_mat_tmp24 - weak_mat_tmp37 + weak_mat_tmp39) + c2*(((scalar_t(4) / scalar_t(3)))*grad_u[2]*weak_mat_tmp28*weak_mat_tmp29 - weak_mat_tmp22*weak_mat_tmp24 - weak_mat_tmp24*weak_mat_tmp33 - weak_mat_tmp40*weak_mat_tmp42) + weak_mat_tmp34*weak_mat_tmp7 - weak_mat_tmp34;
        const scalar_t weak_mat_tmp44 = weak_mat_tmp36*weak_mat_tmp5;
        const scalar_t weak_mat_tmp45 = weak_mat_tmp23*weak_mat_tmp35;
        const scalar_t weak_mat_tmp46 = weak_mat_tmp24*weak_mat_tmp35;
        const scalar_t weak_mat_tmp47 = scalar_t(2)*grad_u[2];
        const scalar_t weak_mat_tmp48 = scalar_t(2)*weak_mat_tmp0;
        const scalar_t weak_mat_tmp49 = scalar_t(2)*grad_u[2]*weak_mat_tmp13 - weak_mat_tmp10*weak_mat_tmp47 - weak_mat_tmp25*weak_mat_tmp48;
        const scalar_t weak_mat_tmp50 = c1*(-weak_mat_tmp15*weak_mat_tmp36 + weak_mat_tmp45 - weak_mat_tmp46) + c2*(((scalar_t(4) / scalar_t(3)))*grad_u[1]*weak_mat_tmp28*weak_mat_tmp29 - weak_mat_tmp22*weak_mat_tmp36 - weak_mat_tmp33*weak_mat_tmp36 - weak_mat_tmp42*weak_mat_tmp49) + weak_mat_tmp44*weak_mat_tmp7 - weak_mat_tmp44;
        const scalar_t weak_mat_tmp51 = weak_mat_tmp17*weak_mat_tmp5;
        const scalar_t weak_mat_tmp52 = kappa*weak_mat_tmp7/weak_mat_tmp4;
        const scalar_t weak_mat_tmp53 = ((scalar_t(2) / scalar_t(3)))*weak_mat_tmp13*weak_mat_tmp18;
        const scalar_t weak_mat_tmp54 = scalar_t(2)*weak_mat_tmp0*weak_mat_tmp13 - weak_mat_tmp12*weak_mat_tmp48 - weak_mat_tmp25*weak_mat_tmp47;
        const scalar_t weak_mat_tmp55 = weak_mat_tmp31*weak_mat_tmp41;
        const scalar_t weak_mat_tmp56 = c1*(((scalar_t(10) / scalar_t(9)))*weak_mat_tmp0*weak_mat_tmp13*weak_mat_tmp14*weak_mat_tmp3 - weak_mat_tmp1*weak_mat_tmp35 - weak_mat_tmp35*weak_mat_tmp9 - weak_mat_tmp53) + c2*(((scalar_t(28) / scalar_t(9)))*weak_mat_tmp0*weak_mat_tmp3*weak_mat_tmp31*weak_mat_tmp32 + weak_mat_tmp21*(scalar_t(4)*weak_mat_tmp0*weak_mat_tmp3 - scalar_t(2)*weak_mat_tmp2) - weak_mat_tmp28*weak_mat_tmp3*weak_mat_tmp41 - weak_mat_tmp42*weak_mat_tmp54 - weak_mat_tmp55) - weak_mat_tmp51*weak_mat_tmp7 + weak_mat_tmp51 + weak_mat_tmp52;
        const scalar_t weak_mat_tmp57 = weak_mat_tmp5*weak_mat_tmp8;
        const scalar_t weak_mat_tmp58 = weak_mat_tmp16 + weak_mat_tmp19*weak_mat_tmp2;
        const scalar_t weak_mat_tmp59 = weak_mat_tmp38*weak_mat_tmp5;
        const scalar_t weak_mat_tmp60 = weak_mat_tmp40*weak_mat_tmp41;
        const scalar_t weak_mat_tmp61 = c1*(-weak_mat_tmp15*weak_mat_tmp38 - weak_mat_tmp45 + weak_mat_tmp46) + c2*(((scalar_t(4) / scalar_t(3)))*grad_u[2]*weak_mat_tmp29*weak_mat_tmp54 - weak_mat_tmp22*weak_mat_tmp38 - weak_mat_tmp3*weak_mat_tmp60 - weak_mat_tmp33*weak_mat_tmp38) + weak_mat_tmp59*weak_mat_tmp7 - weak_mat_tmp59;
        const scalar_t weak_mat_tmp62 = weak_mat_tmp2*weak_mat_tmp5;
        const scalar_t weak_mat_tmp63 = c1*(weak_mat_tmp11*weak_mat_tmp35 + weak_mat_tmp15*weak_mat_tmp2 + weak_mat_tmp35*weak_mat_tmp8 + weak_mat_tmp53) + c2*(grad_u[1]*weak_mat_tmp60 + grad_u[2]*weak_mat_tmp41*weak_mat_tmp49 + weak_mat_tmp2*weak_mat_tmp33 + weak_mat_tmp21*(-scalar_t(2)*weak_mat_tmp17 + scalar_t(4)*weak_mat_tmp2) + weak_mat_tmp55) - weak_mat_tmp52 - weak_mat_tmp62*weak_mat_tmp7 + weak_mat_tmp62;
        const scalar_t weak_mat_tmp64 = weak_mat_tmp11*weak_mat_tmp5;
        const scalar_t weak_mat_tmp65 = weak_mat_tmp23*weak_mat_tmp5;
        const scalar_t weak_mat_tmp66 = c1*(-weak_mat_tmp15*weak_mat_tmp23 + weak_mat_tmp37 - weak_mat_tmp39) + c2*(((scalar_t(4) / scalar_t(3)))*grad_u[1]*weak_mat_tmp29*weak_mat_tmp54 - weak_mat_tmp22*weak_mat_tmp23 - weak_mat_tmp23*weak_mat_tmp33 - weak_mat_tmp3*weak_mat_tmp41*weak_mat_tmp49) + weak_mat_tmp65*weak_mat_tmp7 - weak_mat_tmp65;
        const scalar_t weak_mat_tmp67 = weak_mat_tmp5*weak_mat_tmp9;
        material[0] = trial_grad[0]*(c1*(weak_mat_tmp1*weak_mat_tmp15 + weak_mat_tmp20) + c2*(-weak_mat_tmp0*weak_mat_tmp28*weak_mat_tmp30 + weak_mat_tmp1*weak_mat_tmp22 + weak_mat_tmp1*weak_mat_tmp33) - weak_mat_tmp6*weak_mat_tmp7 + weak_mat_tmp6) + trial_grad[1]*weak_mat_tmp43 + trial_grad[2]*weak_mat_tmp50 + trial_grad[3]*weak_mat_tmp56;
        material[1] = trial_grad[0]*weak_mat_tmp43 + trial_grad[1]*(c1*(weak_mat_tmp15*weak_mat_tmp8 + weak_mat_tmp58) + c2*(grad_u[2]*weak_mat_tmp30*weak_mat_tmp40 + weak_mat_tmp22*weak_mat_tmp8 + weak_mat_tmp33*weak_mat_tmp8) - weak_mat_tmp57*weak_mat_tmp7 + weak_mat_tmp57) + trial_grad[2]*weak_mat_tmp63 + trial_grad[3]*weak_mat_tmp61;
        material[2] = trial_grad[0]*weak_mat_tmp50 + trial_grad[1]*weak_mat_tmp63 + trial_grad[2]*(c1*(weak_mat_tmp11*weak_mat_tmp15 + weak_mat_tmp58) + c2*(grad_u[1]*weak_mat_tmp30*weak_mat_tmp49 + weak_mat_tmp11*weak_mat_tmp22 + weak_mat_tmp11*weak_mat_tmp33) - weak_mat_tmp64*weak_mat_tmp7 + weak_mat_tmp64) + trial_grad[3]*weak_mat_tmp66;
        material[3] = trial_grad[0]*weak_mat_tmp56 + trial_grad[1]*weak_mat_tmp61 + trial_grad[2]*weak_mat_tmp66 + trial_grad[3]*(c1*(weak_mat_tmp15*weak_mat_tmp9 + weak_mat_tmp20) + c2*(weak_mat_tmp22*weak_mat_tmp9 - weak_mat_tmp3*weak_mat_tmp30*weak_mat_tmp54 + weak_mat_tmp33*weak_mat_tmp9) - weak_mat_tmp67*weak_mat_tmp7 + weak_mat_tmp67);
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
