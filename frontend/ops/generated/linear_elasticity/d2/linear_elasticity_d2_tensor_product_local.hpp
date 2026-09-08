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
typedef ptrdiff_t count_t;
typedef double geom_t;
#endif
namespace sfem {
namespace codegen {

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void linear_elasticity_d2_tensor_product_objective_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT jacobian_adjugate0,
        const s_t *const SFEM_RESTRICT jacobian_adjugate1,
        const s_t *const SFEM_RESTRICT jacobian_adjugate2,
        const s_t *const SFEM_RESTRICT jacobian_adjugate3,
        const s_t *const SFEM_RESTRICT jacobian_determinant0,
        const s_t *const SFEM_RESTRICT shape_1d,
        const s_t *const SFEM_RESTRICT grad_1d,
        const s_t *const SFEM_RESTRICT q_weight_1d,
        const s_t lmbda,
        const s_t mu,
        const s_t *const SFEM_RESTRICT u_streams[NS * 2],
        s_t *const SFEM_RESTRICT value
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    static_assert(ipow(NQ1, 2) == NQ, "NQ must be tensor-product compatible");
    static_assert(ipow(NS1, 2) == NS, "NS must be tensor-product compatible");
    s_t grad_u_ref_q[NQ * 4 * VS];
    tensor_gradient<s_t, NQ, NS, VS, 2, 2>(nelems, shape_1d, grad_1d, u_streams, 0, &grad_u_ref_q[0 * NQ * 2 * VS]);
    tensor_gradient<s_t, NQ, NS, VS, 2, 2>(nelems, shape_1d, grad_1d, u_streams, 1, &grad_u_ref_q[1 * NQ * 2 * VS]);
    for (int q = 0; q < NQ; ++q) {
        const int qx = q % NQ1;
        const int qy = q / NQ1;
        const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const s_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const s_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const s_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const s_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const s_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            s_t grad_u_ref[4];
            grad_u_ref[0] = grad_u_ref_q[((0 * NQ + q) * 2 + 0) * VS + lane];
            grad_u_ref[1] = grad_u_ref_q[((0 * NQ + q) * 2 + 1) * VS + lane];
            grad_u_ref[2] = grad_u_ref_q[((1 * NQ + q) * 2 + 0) * VS + lane];
            grad_u_ref[3] = grad_u_ref_q[((1 * NQ + q) * 2 + 1) * VS + lane];
            s_t grad_u[4];
            const s_t inv_jacobian_determinant = s_t(1) / jacobian_determinant_lane0;
            grad_u[0] = (grad_u_ref[0] * jacobian_adjugate_lane0 + grad_u_ref[1] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
            grad_u[1] = (grad_u_ref[0] * jacobian_adjugate_lane1 + grad_u_ref[1] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
            grad_u[2] = (grad_u_ref[2] * jacobian_adjugate_lane0 + grad_u_ref[3] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
            grad_u[3] = (grad_u_ref[2] * jacobian_adjugate_lane1 + grad_u_ref[3] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        value[lane] += qw * jacobian_determinant_lane0 * (((s_t(1) / s_t(2)))*lmbda*pow_2(grad_u[0] + grad_u[3]) + mu*(pow_2(grad_u[0]) + pow_2(grad_u[3]) + s_t(2)*pow_2(((s_t(1) / s_t(2)))*grad_u[1] + ((s_t(1) / s_t(2)))*grad_u[2])));
        }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void linear_elasticity_d2_tensor_product_gradient_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT jacobian_adjugate0,
        const s_t *const SFEM_RESTRICT jacobian_adjugate1,
        const s_t *const SFEM_RESTRICT jacobian_adjugate2,
        const s_t *const SFEM_RESTRICT jacobian_adjugate3,
        const s_t *const SFEM_RESTRICT jacobian_determinant0,
        const s_t *const SFEM_RESTRICT shape_1d,
        const s_t *const SFEM_RESTRICT grad_1d,
        const s_t *const SFEM_RESTRICT q_weight_1d,
        const s_t lmbda,
        const s_t mu,
        const s_t *const SFEM_RESTRICT u_streams[NS * 2],
        s_t *const SFEM_RESTRICT out_streams[NS * 2]
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    static_assert(ipow(NQ1, 2) == NQ, "NQ must be tensor-product compatible");
    static_assert(ipow(NS1, 2) == NS, "NS must be tensor-product compatible");
    s_t grad_u_ref_q[NQ * 4 * VS];
    s_t loperand_q[NQ * 4 * VS];
    tensor_gradient<s_t, NQ, NS, VS, 2, 2>(nelems, shape_1d, grad_1d, u_streams, 0, &grad_u_ref_q[0 * NQ * 2 * VS]);
    tensor_gradient<s_t, NQ, NS, VS, 2, 2>(nelems, shape_1d, grad_1d, u_streams, 1, &grad_u_ref_q[1 * NQ * 2 * VS]);
    for (int q = 0; q < NQ; ++q) {
        const int qx = q % NQ1;
        const int qy = q / NQ1;
        const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const s_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const s_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const s_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const s_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const s_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            s_t grad_u_ref[4];
            grad_u_ref[0] = grad_u_ref_q[((0 * NQ + q) * 2 + 0) * VS + lane];
            grad_u_ref[1] = grad_u_ref_q[((0 * NQ + q) * 2 + 1) * VS + lane];
            grad_u_ref[2] = grad_u_ref_q[((1 * NQ + q) * 2 + 0) * VS + lane];
            grad_u_ref[3] = grad_u_ref_q[((1 * NQ + q) * 2 + 1) * VS + lane];
            s_t grad_u[4];
            const s_t inv_jacobian_determinant = s_t(1) / jacobian_determinant_lane0;
            grad_u[0] = (grad_u_ref[0] * jacobian_adjugate_lane0 + grad_u_ref[1] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
            grad_u[1] = (grad_u_ref[0] * jacobian_adjugate_lane1 + grad_u_ref[1] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
            grad_u[2] = (grad_u_ref[2] * jacobian_adjugate_lane0 + grad_u_ref[3] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
            grad_u[3] = (grad_u_ref[2] * jacobian_adjugate_lane1 + grad_u_ref[3] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
            s_t loperand[4];
        s_t material[4];
        const s_t weak_mat_tmp0 = s_t(2)*grad_u[0];
        const s_t weak_mat_tmp1 = s_t(2)*grad_u[3];
        const s_t weak_mat_tmp2 = ((s_t(1) / s_t(2)))*lmbda*(weak_mat_tmp0 + weak_mat_tmp1);
        const s_t weak_mat_tmp3 = mu*(grad_u[1] + grad_u[2]);
        material[0] = mu*weak_mat_tmp0 + weak_mat_tmp2;
        material[1] = weak_mat_tmp3;
        material[2] = weak_mat_tmp3;
        material[3] = mu*weak_mat_tmp1 + weak_mat_tmp2;
        loperand[0] = qw * (material[0] * jacobian_adjugate_lane0 + material[1] * jacobian_adjugate_lane1);
        loperand[1] = qw * (material[0] * jacobian_adjugate_lane2 + material[1] * jacobian_adjugate_lane3);
        loperand[2] = qw * (material[2] * jacobian_adjugate_lane0 + material[3] * jacobian_adjugate_lane1);
        loperand[3] = qw * (material[2] * jacobian_adjugate_lane2 + material[3] * jacobian_adjugate_lane3);
            loperand_q[((0 * NQ + q) * 2 + 0) * VS + lane] = loperand[0];
            loperand_q[((0 * NQ + q) * 2 + 1) * VS + lane] = loperand[1];
            loperand_q[((1 * NQ + q) * 2 + 0) * VS + lane] = loperand[2];
            loperand_q[((1 * NQ + q) * 2 + 1) * VS + lane] = loperand[3];
        }
    }
    tensor_test<s_t, NQ, NS, VS, 2, 2>(nelems, shape_1d, grad_1d, &loperand_q[0 * NQ * 2 * VS], out_streams, 0);
    tensor_test<s_t, NQ, NS, VS, 2, 2>(nelems, shape_1d, grad_1d, &loperand_q[1 * NQ * 2 * VS], out_streams, 1);
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void linear_elasticity_d2_tensor_product_apply_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT jacobian_adjugate0,
        const s_t *const SFEM_RESTRICT jacobian_adjugate1,
        const s_t *const SFEM_RESTRICT jacobian_adjugate2,
        const s_t *const SFEM_RESTRICT jacobian_adjugate3,
        const s_t *const SFEM_RESTRICT jacobian_determinant0,
        const s_t *const SFEM_RESTRICT shape_1d,
        const s_t *const SFEM_RESTRICT grad_1d,
        const s_t *const SFEM_RESTRICT q_weight_1d,
        const s_t lmbda,
        const s_t mu,
        const s_t *const SFEM_RESTRICT h_streams[NS * 2],
        s_t *const SFEM_RESTRICT out_streams[NS * 2]
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
    static constexpr int NQ1 = integer_root(NQ, 2);
    static constexpr int NS1 = integer_root(NS, 2);
    static_assert(ipow(NQ1, 2) == NQ, "NQ must be tensor-product compatible");
    static_assert(ipow(NS1, 2) == NS, "NS must be tensor-product compatible");
    s_t grad_h_ref_q[NQ * 4 * VS];
    s_t loperand_q[NQ * 4 * VS];
    tensor_gradient<s_t, NQ, NS, VS, 2, 2>(nelems, shape_1d, grad_1d, h_streams, 0, &grad_h_ref_q[0 * NQ * 2 * VS]);
    tensor_gradient<s_t, NQ, NS, VS, 2, 2>(nelems, shape_1d, grad_1d, h_streams, 1, &grad_h_ref_q[1 * NQ * 2 * VS]);
    for (int q = 0; q < NQ; ++q) {
        const int qx = q % NQ1;
        const int qy = q / NQ1;
        const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const s_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const s_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const s_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const s_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const s_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            s_t grad_h_ref[4];
            grad_h_ref[0] = grad_h_ref_q[((0 * NQ + q) * 2 + 0) * VS + lane];
            grad_h_ref[1] = grad_h_ref_q[((0 * NQ + q) * 2 + 1) * VS + lane];
            grad_h_ref[2] = grad_h_ref_q[((1 * NQ + q) * 2 + 0) * VS + lane];
            grad_h_ref[3] = grad_h_ref_q[((1 * NQ + q) * 2 + 1) * VS + lane];
            s_t trial_grad[4];
            const s_t inv_jacobian_determinant = s_t(1) / jacobian_determinant_lane0;
            trial_grad[0] = (grad_h_ref[0] * jacobian_adjugate_lane0 + grad_h_ref[1] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
            trial_grad[1] = (grad_h_ref[0] * jacobian_adjugate_lane1 + grad_h_ref[1] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
            trial_grad[2] = (grad_h_ref[2] * jacobian_adjugate_lane0 + grad_h_ref[3] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
            trial_grad[3] = (grad_h_ref[2] * jacobian_adjugate_lane1 + grad_h_ref[3] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
            s_t loperand[4];
        s_t material[4];
        const s_t weak_mat_tmp0 = s_t(2)*trial_grad[0];
        const s_t weak_mat_tmp1 = s_t(2)*trial_grad[3];
        const s_t weak_mat_tmp2 = ((s_t(1) / s_t(2)))*lmbda*(weak_mat_tmp0 + weak_mat_tmp1);
        const s_t weak_mat_tmp3 = mu*(trial_grad[1] + trial_grad[2]);
        material[0] = mu*weak_mat_tmp0 + weak_mat_tmp2;
        material[1] = weak_mat_tmp3;
        material[2] = weak_mat_tmp3;
        material[3] = mu*weak_mat_tmp1 + weak_mat_tmp2;
        loperand[0] = qw * (material[0] * jacobian_adjugate_lane0 + material[1] * jacobian_adjugate_lane1);
        loperand[1] = qw * (material[0] * jacobian_adjugate_lane2 + material[1] * jacobian_adjugate_lane3);
        loperand[2] = qw * (material[2] * jacobian_adjugate_lane0 + material[3] * jacobian_adjugate_lane1);
        loperand[3] = qw * (material[2] * jacobian_adjugate_lane2 + material[3] * jacobian_adjugate_lane3);
            loperand_q[((0 * NQ + q) * 2 + 0) * VS + lane] = loperand[0];
            loperand_q[((0 * NQ + q) * 2 + 1) * VS + lane] = loperand[1];
            loperand_q[((1 * NQ + q) * 2 + 0) * VS + lane] = loperand[2];
            loperand_q[((1 * NQ + q) * 2 + 1) * VS + lane] = loperand[3];
        }
    }
    tensor_test<s_t, NQ, NS, VS, 2, 2>(nelems, shape_1d, grad_1d, &loperand_q[0 * NQ * 2 * VS], out_streams, 0);
    tensor_test<s_t, NQ, NS, VS, 2, 2>(nelems, shape_1d, grad_1d, &loperand_q[1 * NQ * 2 * VS], out_streams, 1);
}

} // namespace codegen
} // namespace sfem

#endif
