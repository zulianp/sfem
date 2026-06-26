#ifndef GENERATED_NEOHOOKEAN_OGDEN_D2_SIMPLEX_LOCAL_HPP
#define GENERATED_NEOHOOKEAN_OGDEN_D2_SIMPLEX_LOCAL_HPP

#include <math.h>
#include <stddef.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_GENERATED_SCALAR_T
#endif
#endif

#include "kernel_math.hpp"
#include "tensor_product_kernels.hpp"

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
static SFEM_INLINE void generated_neohookean_ogden_d2_simplex_objective_block(
        const ptrdiff_t nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t mu,
        const scalar_t lmbda,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 2],
        scalar_t *const SFEM_RESTRICT value
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const scalar_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const scalar_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const scalar_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const scalar_t qw = q_weight[q];
            scalar_t grad_u_ref0 = scalar_t(0);
            scalar_t grad_u_ref1 = scalar_t(0);
            scalar_t grad_u_ref2 = scalar_t(0);
            scalar_t grad_u_ref3 = scalar_t(0);
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                grad_u_ref0 += u_streams[shape * 2 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref1 += u_streams[shape * 2 + 0][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_u_ref2 += u_streams[shape * 2 + 1][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref3 += u_streams[shape * 2 + 1][lane] * grad_ref_y[q * N_SHAPE + shape];
            }
        const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
        const scalar_t grad_u0 = (grad_u_ref0 * jacobian_adjugate_lane0 + grad_u_ref1 * jacobian_adjugate_lane2) * inv_jacobian_determinant;
        const scalar_t grad_u1 = (grad_u_ref0 * jacobian_adjugate_lane1 + grad_u_ref1 * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        const scalar_t grad_u2 = (grad_u_ref2 * jacobian_adjugate_lane0 + grad_u_ref3 * jacobian_adjugate_lane2) * inv_jacobian_determinant;
        const scalar_t grad_u3 = (grad_u_ref2 * jacobian_adjugate_lane1 + grad_u_ref3 * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        const scalar_t weak_obj_tmp0 = grad_u0 + scalar_t(1);
        const scalar_t weak_obj_tmp1 = grad_u3 + scalar_t(1);
        const scalar_t weak_obj_tmp2 = log(-grad_u1*grad_u2 + weak_obj_tmp0*weak_obj_tmp1);
        value[lane] += qw * jacobian_determinant_lane0 * (((scalar_t(1) / scalar_t(2)))*lmbda*pow_2(weak_obj_tmp2) - mu*weak_obj_tmp2 + ((scalar_t(1) / scalar_t(2)))*mu*(pow_2(grad_u1) + pow_2(grad_u2) + pow_2(weak_obj_tmp0) + pow_2(weak_obj_tmp1) + scalar_t(-2)));
            }
        }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void generated_neohookean_ogden_d2_simplex_gradient_block(
        const ptrdiff_t nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t mu,
        const scalar_t lmbda,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 2],
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 2]
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const scalar_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const scalar_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const scalar_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const scalar_t qw = q_weight[q];
            scalar_t grad_u_ref0 = scalar_t(0);
            scalar_t grad_u_ref1 = scalar_t(0);
            scalar_t grad_u_ref2 = scalar_t(0);
            scalar_t grad_u_ref3 = scalar_t(0);
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                grad_u_ref0 += u_streams[shape * 2 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref1 += u_streams[shape * 2 + 0][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_u_ref2 += u_streams[shape * 2 + 1][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref3 += u_streams[shape * 2 + 1][lane] * grad_ref_y[q * N_SHAPE + shape];
            }
        const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
        const scalar_t grad_u0 = (grad_u_ref0 * jacobian_adjugate_lane0 + grad_u_ref1 * jacobian_adjugate_lane2) * inv_jacobian_determinant;
        const scalar_t grad_u1 = (grad_u_ref0 * jacobian_adjugate_lane1 + grad_u_ref1 * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        const scalar_t grad_u2 = (grad_u_ref2 * jacobian_adjugate_lane0 + grad_u_ref3 * jacobian_adjugate_lane2) * inv_jacobian_determinant;
        const scalar_t grad_u3 = (grad_u_ref2 * jacobian_adjugate_lane1 + grad_u_ref3 * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        const scalar_t weak_mat_tmp0 = grad_u0 + scalar_t(1);
        const scalar_t weak_mat_tmp1 = mu*weak_mat_tmp0;
        const scalar_t weak_mat_tmp2 = grad_u3 + scalar_t(1);
        const scalar_t weak_mat_tmp3 = -grad_u1*grad_u2 + weak_mat_tmp0*weak_mat_tmp2;
        const scalar_t weak_mat_tmp4 = pow_m1(weak_mat_tmp3);
        const scalar_t weak_mat_tmp5 = mu*weak_mat_tmp2;
        const scalar_t weak_mat_tmp6 = lmbda*weak_mat_tmp4*log(weak_mat_tmp3);
        const scalar_t weak_mat_tmp7 = grad_u1*mu;
        const scalar_t weak_mat_tmp8 = grad_u2*mu;
        const scalar_t material0 = weak_mat_tmp1 + weak_mat_tmp2*weak_mat_tmp6 - weak_mat_tmp4*weak_mat_tmp5;
        const scalar_t material1 = -grad_u2*weak_mat_tmp6 + weak_mat_tmp4*weak_mat_tmp8 + weak_mat_tmp7;
        const scalar_t material2 = -grad_u1*weak_mat_tmp6 + weak_mat_tmp4*weak_mat_tmp7 + weak_mat_tmp8;
        const scalar_t material3 = weak_mat_tmp0*weak_mat_tmp6 - weak_mat_tmp1*weak_mat_tmp4 + weak_mat_tmp5;
        const scalar_t loperand0 = qw * (material0 * jacobian_adjugate_lane0 + material1 * jacobian_adjugate_lane1);
        const scalar_t loperand1 = qw * (material0 * jacobian_adjugate_lane2 + material1 * jacobian_adjugate_lane3);
        const scalar_t loperand2 = qw * (material2 * jacobian_adjugate_lane0 + material3 * jacobian_adjugate_lane1);
        const scalar_t loperand3 = qw * (material2 * jacobian_adjugate_lane2 + material3 * jacobian_adjugate_lane3);
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                out_streams[shape * 2 + 0][lane] += loperand0 * grad_ref_x[q * N_SHAPE + shape] + loperand1 * grad_ref_y[q * N_SHAPE + shape];
                out_streams[shape * 2 + 1][lane] += loperand2 * grad_ref_x[q * N_SHAPE + shape] + loperand3 * grad_ref_y[q * N_SHAPE + shape];
            }
            }
        }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void generated_neohookean_ogden_d2_simplex_apply_block(
        const ptrdiff_t nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t mu,
        const scalar_t lmbda,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 2],
        const scalar_t *const SFEM_RESTRICT h_streams[N_SHAPE * 2],
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 2]
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
        for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
            for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const scalar_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const scalar_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const scalar_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const scalar_t qw = q_weight[q];
            scalar_t grad_u_ref0 = scalar_t(0);
            scalar_t grad_h_ref0 = scalar_t(0);
            scalar_t grad_u_ref1 = scalar_t(0);
            scalar_t grad_h_ref1 = scalar_t(0);
            scalar_t grad_u_ref2 = scalar_t(0);
            scalar_t grad_h_ref2 = scalar_t(0);
            scalar_t grad_u_ref3 = scalar_t(0);
            scalar_t grad_h_ref3 = scalar_t(0);
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                grad_u_ref0 += u_streams[shape * 2 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_h_ref0 += h_streams[shape * 2 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref1 += u_streams[shape * 2 + 0][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_h_ref1 += h_streams[shape * 2 + 0][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_u_ref2 += u_streams[shape * 2 + 1][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_h_ref2 += h_streams[shape * 2 + 1][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref3 += u_streams[shape * 2 + 1][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_h_ref3 += h_streams[shape * 2 + 1][lane] * grad_ref_y[q * N_SHAPE + shape];
            }
        const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
        const scalar_t grad_u0 = (grad_u_ref0 * jacobian_adjugate_lane0 + grad_u_ref1 * jacobian_adjugate_lane2) * inv_jacobian_determinant;
        const scalar_t trial_grad0 = (grad_h_ref0 * jacobian_adjugate_lane0 + grad_h_ref1 * jacobian_adjugate_lane2) * inv_jacobian_determinant;
        const scalar_t grad_u1 = (grad_u_ref0 * jacobian_adjugate_lane1 + grad_u_ref1 * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        const scalar_t trial_grad1 = (grad_h_ref0 * jacobian_adjugate_lane1 + grad_h_ref1 * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        const scalar_t grad_u2 = (grad_u_ref2 * jacobian_adjugate_lane0 + grad_u_ref3 * jacobian_adjugate_lane2) * inv_jacobian_determinant;
        const scalar_t trial_grad2 = (grad_h_ref2 * jacobian_adjugate_lane0 + grad_h_ref3 * jacobian_adjugate_lane2) * inv_jacobian_determinant;
        const scalar_t grad_u3 = (grad_u_ref2 * jacobian_adjugate_lane1 + grad_u_ref3 * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        const scalar_t trial_grad3 = (grad_h_ref2 * jacobian_adjugate_lane1 + grad_h_ref3 * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        const scalar_t weak_mat_tmp0 = grad_u3 + scalar_t(1);
        const scalar_t weak_mat_tmp1 = grad_u1*grad_u2;
        const scalar_t weak_mat_tmp2 = grad_u0 + scalar_t(1);
        const scalar_t weak_mat_tmp3 = weak_mat_tmp0*weak_mat_tmp2 - weak_mat_tmp1;
        const scalar_t weak_mat_tmp4 = pow_m2(weak_mat_tmp3);
        const scalar_t weak_mat_tmp5 = weak_mat_tmp0*weak_mat_tmp4;
        const scalar_t weak_mat_tmp6 = grad_u2*weak_mat_tmp5;
        const scalar_t weak_mat_tmp7 = log(weak_mat_tmp3);
        const scalar_t weak_mat_tmp8 = grad_u2*lmbda*weak_mat_tmp0*weak_mat_tmp4*weak_mat_tmp7 - lmbda*weak_mat_tmp6 - mu*weak_mat_tmp6;
        const scalar_t weak_mat_tmp9 = grad_u1*weak_mat_tmp5;
        const scalar_t weak_mat_tmp10 = grad_u1*lmbda*weak_mat_tmp0*weak_mat_tmp4*weak_mat_tmp7 - lmbda*weak_mat_tmp9 - mu*weak_mat_tmp9;
        const scalar_t weak_mat_tmp11 = pow_2(weak_mat_tmp0)*weak_mat_tmp4;
        const scalar_t weak_mat_tmp12 = lmbda*weak_mat_tmp11;
        const scalar_t weak_mat_tmp13 = pow_m1(weak_mat_tmp3);
        const scalar_t weak_mat_tmp14 = mu*weak_mat_tmp13;
        const scalar_t weak_mat_tmp15 = weak_mat_tmp0*weak_mat_tmp2*weak_mat_tmp4;
        const scalar_t weak_mat_tmp16 = lmbda*weak_mat_tmp7;
        const scalar_t weak_mat_tmp17 = weak_mat_tmp13*weak_mat_tmp16;
        const scalar_t weak_mat_tmp18 = lmbda*weak_mat_tmp15 + mu*weak_mat_tmp15 - weak_mat_tmp14 - weak_mat_tmp15*weak_mat_tmp16 + weak_mat_tmp17;
        const scalar_t weak_mat_tmp19 = pow_2(grad_u2)*weak_mat_tmp4;
        const scalar_t weak_mat_tmp20 = weak_mat_tmp2*weak_mat_tmp4;
        const scalar_t weak_mat_tmp21 = grad_u2*weak_mat_tmp20;
        const scalar_t weak_mat_tmp22 = grad_u2*lmbda*weak_mat_tmp2*weak_mat_tmp4*weak_mat_tmp7 - lmbda*weak_mat_tmp21 - mu*weak_mat_tmp21;
        const scalar_t weak_mat_tmp23 = weak_mat_tmp1*weak_mat_tmp4;
        const scalar_t weak_mat_tmp24 = lmbda*weak_mat_tmp23 + mu*weak_mat_tmp23 + weak_mat_tmp14 - weak_mat_tmp16*weak_mat_tmp23 - weak_mat_tmp17;
        const scalar_t weak_mat_tmp25 = pow_2(grad_u1)*weak_mat_tmp4;
        const scalar_t weak_mat_tmp26 = grad_u1*weak_mat_tmp20;
        const scalar_t weak_mat_tmp27 = grad_u1*lmbda*weak_mat_tmp2*weak_mat_tmp4*weak_mat_tmp7 - lmbda*weak_mat_tmp26 - mu*weak_mat_tmp26;
        const scalar_t weak_mat_tmp28 = pow_2(weak_mat_tmp2)*weak_mat_tmp4;
        const scalar_t material0 = trial_grad0*(mu*weak_mat_tmp11 + mu - weak_mat_tmp12*weak_mat_tmp7 + weak_mat_tmp12) + trial_grad1*weak_mat_tmp8 + trial_grad2*weak_mat_tmp10 + trial_grad3*weak_mat_tmp18;
        const scalar_t material1 = trial_grad0*weak_mat_tmp8 + trial_grad1*(lmbda*weak_mat_tmp19 + mu*weak_mat_tmp19 + mu - weak_mat_tmp16*weak_mat_tmp19) + trial_grad2*weak_mat_tmp24 + trial_grad3*weak_mat_tmp22;
        const scalar_t material2 = trial_grad0*weak_mat_tmp10 + trial_grad1*weak_mat_tmp24 + trial_grad2*(lmbda*weak_mat_tmp25 + mu*weak_mat_tmp25 + mu - weak_mat_tmp16*weak_mat_tmp25) + trial_grad3*weak_mat_tmp27;
        const scalar_t material3 = trial_grad0*weak_mat_tmp18 + trial_grad1*weak_mat_tmp22 + trial_grad2*weak_mat_tmp27 + trial_grad3*(lmbda*weak_mat_tmp28 + mu*weak_mat_tmp28 + mu - weak_mat_tmp16*weak_mat_tmp28);
        const scalar_t loperand0 = qw * (material0 * jacobian_adjugate_lane0 + material1 * jacobian_adjugate_lane1);
        const scalar_t loperand1 = qw * (material0 * jacobian_adjugate_lane2 + material1 * jacobian_adjugate_lane3);
        const scalar_t loperand2 = qw * (material2 * jacobian_adjugate_lane0 + material3 * jacobian_adjugate_lane1);
        const scalar_t loperand3 = qw * (material2 * jacobian_adjugate_lane2 + material3 * jacobian_adjugate_lane3);
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                out_streams[shape * 2 + 0][lane] += loperand0 * grad_ref_x[q * N_SHAPE + shape] + loperand1 * grad_ref_y[q * N_SHAPE + shape];
                out_streams[shape * 2 + 1][lane] += loperand2 * grad_ref_x[q * N_SHAPE + shape] + loperand3 * grad_ref_y[q * N_SHAPE + shape];
            }
            }
        }
}

} // namespace codegen
} // namespace sfem

#endif
