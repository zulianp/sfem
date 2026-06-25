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
#pragma omp simd
    for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
        for (int q = 0; q < N_QP; ++q) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const scalar_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const scalar_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const scalar_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const scalar_t qw = q_weight[q];
            scalar_t grad_u_ref[4];
            grad_u_ref[0] = 0.0;
            grad_u_ref[1] = 0.0;
            grad_u_ref[2] = 0.0;
            grad_u_ref[3] = 0.0;
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                grad_u_ref[0] += u_streams[shape * 2 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref[1] += u_streams[shape * 2 + 0][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_u_ref[2] += u_streams[shape * 2 + 1][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref[3] += u_streams[shape * 2 + 1][lane] * grad_ref_y[q * N_SHAPE + shape];
            }
            scalar_t grad_u[4];
        const scalar_t inv_jacobian_determinant = 1.0 / jacobian_determinant_lane0;
        grad_u[0] = (grad_u_ref[0] * jacobian_adjugate_lane0 + grad_u_ref[1] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
        grad_u[1] = (grad_u_ref[0] * jacobian_adjugate_lane1 + grad_u_ref[1] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        grad_u[2] = (grad_u_ref[2] * jacobian_adjugate_lane0 + grad_u_ref[3] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
        grad_u[3] = (grad_u_ref[2] * jacobian_adjugate_lane1 + grad_u_ref[3] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        const scalar_t weak_obj_tmp0 = grad_u[0] + 1;
        const scalar_t weak_obj_tmp1 = grad_u[3] + 1;
        const scalar_t weak_obj_tmp2 = log(-grad_u[1]*grad_u[2] + weak_obj_tmp0*weak_obj_tmp1);
        value[lane] += qw * jacobian_determinant_lane0 * ((1.0/2.0)*lmbda*pow_2(weak_obj_tmp2) - mu*weak_obj_tmp2 + (1.0/2.0)*mu*(pow_2(grad_u[1]) + pow_2(grad_u[2]) + pow_2(weak_obj_tmp0) + pow_2(weak_obj_tmp1) - 2));
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
#pragma omp simd
    for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
        for (int q = 0; q < N_QP; ++q) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const scalar_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const scalar_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const scalar_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const scalar_t qw = q_weight[q];
            scalar_t grad_u_ref[4];
            grad_u_ref[0] = 0.0;
            grad_u_ref[1] = 0.0;
            grad_u_ref[2] = 0.0;
            grad_u_ref[3] = 0.0;
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                grad_u_ref[0] += u_streams[shape * 2 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref[1] += u_streams[shape * 2 + 0][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_u_ref[2] += u_streams[shape * 2 + 1][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref[3] += u_streams[shape * 2 + 1][lane] * grad_ref_y[q * N_SHAPE + shape];
            }
            scalar_t grad_u[4];
        const scalar_t inv_jacobian_determinant = 1.0 / jacobian_determinant_lane0;
        grad_u[0] = (grad_u_ref[0] * jacobian_adjugate_lane0 + grad_u_ref[1] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
        grad_u[1] = (grad_u_ref[0] * jacobian_adjugate_lane1 + grad_u_ref[1] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        grad_u[2] = (grad_u_ref[2] * jacobian_adjugate_lane0 + grad_u_ref[3] * jacobian_adjugate_lane2) * inv_jacobian_determinant;
        grad_u[3] = (grad_u_ref[2] * jacobian_adjugate_lane1 + grad_u_ref[3] * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        scalar_t loperand[4];
        scalar_t material[4];
        const scalar_t weak_mat_tmp0 = grad_u[0] + 1;
        const scalar_t weak_mat_tmp1 = mu*weak_mat_tmp0;
        const scalar_t weak_mat_tmp2 = grad_u[3] + 1;
        const scalar_t weak_mat_tmp3 = -grad_u[1]*grad_u[2] + weak_mat_tmp0*weak_mat_tmp2;
        const scalar_t weak_mat_tmp4 = pow_m1(weak_mat_tmp3);
        const scalar_t weak_mat_tmp5 = mu*weak_mat_tmp2;
        const scalar_t weak_mat_tmp6 = lmbda*weak_mat_tmp4*log(weak_mat_tmp3);
        const scalar_t weak_mat_tmp7 = grad_u[1]*mu;
        const scalar_t weak_mat_tmp8 = grad_u[2]*mu;
        material[0] = weak_mat_tmp1 + weak_mat_tmp2*weak_mat_tmp6 - weak_mat_tmp4*weak_mat_tmp5;
        material[1] = -grad_u[2]*weak_mat_tmp6 + weak_mat_tmp4*weak_mat_tmp8 + weak_mat_tmp7;
        material[2] = -grad_u[1]*weak_mat_tmp6 + weak_mat_tmp4*weak_mat_tmp7 + weak_mat_tmp8;
        material[3] = weak_mat_tmp0*weak_mat_tmp6 - weak_mat_tmp1*weak_mat_tmp4 + weak_mat_tmp5;
        loperand[0] = qw * (material[0] * jacobian_adjugate_lane0 + material[1] * jacobian_adjugate_lane1);
        loperand[1] = qw * (material[0] * jacobian_adjugate_lane2 + material[1] * jacobian_adjugate_lane3);
        loperand[2] = qw * (material[2] * jacobian_adjugate_lane0 + material[3] * jacobian_adjugate_lane1);
        loperand[3] = qw * (material[2] * jacobian_adjugate_lane2 + material[3] * jacobian_adjugate_lane3);
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                out_streams[shape * 2 + 0][lane] += loperand[0] * grad_ref_x[q * N_SHAPE + shape] + loperand[1] * grad_ref_y[q * N_SHAPE + shape];
                out_streams[shape * 2 + 1][lane] += loperand[2] * grad_ref_x[q * N_SHAPE + shape] + loperand[3] * grad_ref_y[q * N_SHAPE + shape];
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
#pragma omp simd
    for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
        for (int q = 0; q < N_QP; ++q) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const scalar_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const scalar_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const scalar_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const scalar_t qw = q_weight[q];
            scalar_t grad_u_ref[4];
            scalar_t grad_h_ref[4];
            grad_u_ref[0] = 0.0;
            grad_h_ref[0] = 0.0;
            grad_u_ref[1] = 0.0;
            grad_h_ref[1] = 0.0;
            grad_u_ref[2] = 0.0;
            grad_h_ref[2] = 0.0;
            grad_u_ref[3] = 0.0;
            grad_h_ref[3] = 0.0;
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                grad_u_ref[0] += u_streams[shape * 2 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_h_ref[0] += h_streams[shape * 2 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref[1] += u_streams[shape * 2 + 0][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_h_ref[1] += h_streams[shape * 2 + 0][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_u_ref[2] += u_streams[shape * 2 + 1][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_h_ref[2] += h_streams[shape * 2 + 1][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref[3] += u_streams[shape * 2 + 1][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_h_ref[3] += h_streams[shape * 2 + 1][lane] * grad_ref_y[q * N_SHAPE + shape];
            }
            scalar_t grad_u[4];
            scalar_t trial_grad[4];
        const scalar_t inv_jacobian_determinant = 1.0 / jacobian_determinant_lane0;
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
        const scalar_t weak_mat_tmp0 = grad_u[3] + 1;
        const scalar_t weak_mat_tmp1 = grad_u[1]*grad_u[2];
        const scalar_t weak_mat_tmp2 = grad_u[0] + 1;
        const scalar_t weak_mat_tmp3 = weak_mat_tmp0*weak_mat_tmp2 - weak_mat_tmp1;
        const scalar_t weak_mat_tmp4 = pow_m2(weak_mat_tmp3);
        const scalar_t weak_mat_tmp5 = weak_mat_tmp0*weak_mat_tmp4;
        const scalar_t weak_mat_tmp6 = grad_u[2]*weak_mat_tmp5;
        const scalar_t weak_mat_tmp7 = log(weak_mat_tmp3);
        const scalar_t weak_mat_tmp8 = grad_u[2]*lmbda*weak_mat_tmp0*weak_mat_tmp4*weak_mat_tmp7 - lmbda*weak_mat_tmp6 - mu*weak_mat_tmp6;
        const scalar_t weak_mat_tmp9 = grad_u[1]*weak_mat_tmp5;
        const scalar_t weak_mat_tmp10 = grad_u[1]*lmbda*weak_mat_tmp0*weak_mat_tmp4*weak_mat_tmp7 - lmbda*weak_mat_tmp9 - mu*weak_mat_tmp9;
        const scalar_t weak_mat_tmp11 = pow_2(weak_mat_tmp0)*weak_mat_tmp4;
        const scalar_t weak_mat_tmp12 = lmbda*weak_mat_tmp11;
        const scalar_t weak_mat_tmp13 = pow_m1(weak_mat_tmp3);
        const scalar_t weak_mat_tmp14 = mu*weak_mat_tmp13;
        const scalar_t weak_mat_tmp15 = weak_mat_tmp0*weak_mat_tmp2*weak_mat_tmp4;
        const scalar_t weak_mat_tmp16 = lmbda*weak_mat_tmp7;
        const scalar_t weak_mat_tmp17 = weak_mat_tmp13*weak_mat_tmp16;
        const scalar_t weak_mat_tmp18 = lmbda*weak_mat_tmp15 + mu*weak_mat_tmp15 - weak_mat_tmp14 - weak_mat_tmp15*weak_mat_tmp16 + weak_mat_tmp17;
        const scalar_t weak_mat_tmp19 = pow_2(grad_u[2])*weak_mat_tmp4;
        const scalar_t weak_mat_tmp20 = weak_mat_tmp2*weak_mat_tmp4;
        const scalar_t weak_mat_tmp21 = grad_u[2]*weak_mat_tmp20;
        const scalar_t weak_mat_tmp22 = grad_u[2]*lmbda*weak_mat_tmp2*weak_mat_tmp4*weak_mat_tmp7 - lmbda*weak_mat_tmp21 - mu*weak_mat_tmp21;
        const scalar_t weak_mat_tmp23 = weak_mat_tmp1*weak_mat_tmp4;
        const scalar_t weak_mat_tmp24 = lmbda*weak_mat_tmp23 + mu*weak_mat_tmp23 + weak_mat_tmp14 - weak_mat_tmp16*weak_mat_tmp23 - weak_mat_tmp17;
        const scalar_t weak_mat_tmp25 = pow_2(grad_u[1])*weak_mat_tmp4;
        const scalar_t weak_mat_tmp26 = grad_u[1]*weak_mat_tmp20;
        const scalar_t weak_mat_tmp27 = grad_u[1]*lmbda*weak_mat_tmp2*weak_mat_tmp4*weak_mat_tmp7 - lmbda*weak_mat_tmp26 - mu*weak_mat_tmp26;
        const scalar_t weak_mat_tmp28 = pow_2(weak_mat_tmp2)*weak_mat_tmp4;
        material[0] = trial_grad[0]*(mu*weak_mat_tmp11 + mu - weak_mat_tmp12*weak_mat_tmp7 + weak_mat_tmp12) + trial_grad[1]*weak_mat_tmp8 + trial_grad[2]*weak_mat_tmp10 + trial_grad[3]*weak_mat_tmp18;
        material[1] = trial_grad[0]*weak_mat_tmp8 + trial_grad[1]*(lmbda*weak_mat_tmp19 + mu*weak_mat_tmp19 + mu - weak_mat_tmp16*weak_mat_tmp19) + trial_grad[2]*weak_mat_tmp24 + trial_grad[3]*weak_mat_tmp22;
        material[2] = trial_grad[0]*weak_mat_tmp10 + trial_grad[1]*weak_mat_tmp24 + trial_grad[2]*(lmbda*weak_mat_tmp25 + mu*weak_mat_tmp25 + mu - weak_mat_tmp16*weak_mat_tmp25) + trial_grad[3]*weak_mat_tmp27;
        material[3] = trial_grad[0]*weak_mat_tmp18 + trial_grad[1]*weak_mat_tmp22 + trial_grad[2]*weak_mat_tmp27 + trial_grad[3]*(lmbda*weak_mat_tmp28 + mu*weak_mat_tmp28 + mu - weak_mat_tmp16*weak_mat_tmp28);
        loperand[0] = qw * (material[0] * jacobian_adjugate_lane0 + material[1] * jacobian_adjugate_lane1);
        loperand[1] = qw * (material[0] * jacobian_adjugate_lane2 + material[1] * jacobian_adjugate_lane3);
        loperand[2] = qw * (material[2] * jacobian_adjugate_lane0 + material[3] * jacobian_adjugate_lane1);
        loperand[3] = qw * (material[2] * jacobian_adjugate_lane2 + material[3] * jacobian_adjugate_lane3);
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                out_streams[shape * 2 + 0][lane] += loperand[0] * grad_ref_x[q * N_SHAPE + shape] + loperand[1] * grad_ref_y[q * N_SHAPE + shape];
                out_streams[shape * 2 + 1][lane] += loperand[2] * grad_ref_x[q * N_SHAPE + shape] + loperand[3] * grad_ref_y[q * N_SHAPE + shape];
            }
        }
    }
}

} // namespace codegen
} // namespace sfem

#endif
