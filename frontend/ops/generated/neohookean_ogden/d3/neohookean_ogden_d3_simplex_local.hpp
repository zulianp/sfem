#ifndef NEOHOOKEAN_OGDEN_D3_SIMPLEX_LOCAL_HPP
#define NEOHOOKEAN_OGDEN_D3_SIMPLEX_LOCAL_HPP

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
static SFEM_INLINE void neohookean_ogden_d3_simplex_objective_block(
        const ptrdiff_t nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate4,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate5,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate6,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate7,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate8,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t mu,
        const scalar_t lmbda,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 3],
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
            const scalar_t jacobian_adjugate_lane4 = jacobian_adjugate4[geometry_offset];
            const scalar_t jacobian_adjugate_lane5 = jacobian_adjugate5[geometry_offset];
            const scalar_t jacobian_adjugate_lane6 = jacobian_adjugate6[geometry_offset];
            const scalar_t jacobian_adjugate_lane7 = jacobian_adjugate7[geometry_offset];
            const scalar_t jacobian_adjugate_lane8 = jacobian_adjugate8[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const scalar_t qw = q_weight[q];
            scalar_t grad_u_ref0 = scalar_t(0);
            scalar_t grad_u_ref1 = scalar_t(0);
            scalar_t grad_u_ref2 = scalar_t(0);
            scalar_t grad_u_ref3 = scalar_t(0);
            scalar_t grad_u_ref4 = scalar_t(0);
            scalar_t grad_u_ref5 = scalar_t(0);
            scalar_t grad_u_ref6 = scalar_t(0);
            scalar_t grad_u_ref7 = scalar_t(0);
            scalar_t grad_u_ref8 = scalar_t(0);
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                grad_u_ref0 += u_streams[shape * 3 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref1 += u_streams[shape * 3 + 0][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_u_ref2 += u_streams[shape * 3 + 0][lane] * grad_ref_z[q * N_SHAPE + shape];
                grad_u_ref3 += u_streams[shape * 3 + 1][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref4 += u_streams[shape * 3 + 1][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_u_ref5 += u_streams[shape * 3 + 1][lane] * grad_ref_z[q * N_SHAPE + shape];
                grad_u_ref6 += u_streams[shape * 3 + 2][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref7 += u_streams[shape * 3 + 2][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_u_ref8 += u_streams[shape * 3 + 2][lane] * grad_ref_z[q * N_SHAPE + shape];
            }
        const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
        const scalar_t grad_u0 = (grad_u_ref0 * jacobian_adjugate_lane0 + grad_u_ref1 * jacobian_adjugate_lane3 + grad_u_ref2 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t grad_u1 = (grad_u_ref0 * jacobian_adjugate_lane1 + grad_u_ref1 * jacobian_adjugate_lane4 + grad_u_ref2 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t grad_u2 = (grad_u_ref0 * jacobian_adjugate_lane2 + grad_u_ref1 * jacobian_adjugate_lane5 + grad_u_ref2 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t grad_u3 = (grad_u_ref3 * jacobian_adjugate_lane0 + grad_u_ref4 * jacobian_adjugate_lane3 + grad_u_ref5 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t grad_u4 = (grad_u_ref3 * jacobian_adjugate_lane1 + grad_u_ref4 * jacobian_adjugate_lane4 + grad_u_ref5 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t grad_u5 = (grad_u_ref3 * jacobian_adjugate_lane2 + grad_u_ref4 * jacobian_adjugate_lane5 + grad_u_ref5 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t grad_u6 = (grad_u_ref6 * jacobian_adjugate_lane0 + grad_u_ref7 * jacobian_adjugate_lane3 + grad_u_ref8 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t grad_u7 = (grad_u_ref6 * jacobian_adjugate_lane1 + grad_u_ref7 * jacobian_adjugate_lane4 + grad_u_ref8 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t grad_u8 = (grad_u_ref6 * jacobian_adjugate_lane2 + grad_u_ref7 * jacobian_adjugate_lane5 + grad_u_ref8 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t weak_obj_tmp0 = grad_u0 + scalar_t(1);
        const scalar_t weak_obj_tmp1 = grad_u4 + scalar_t(1);
        const scalar_t weak_obj_tmp2 = grad_u8 + scalar_t(1);
        const scalar_t weak_obj_tmp3 = log(-grad_u1*grad_u3*weak_obj_tmp2 + grad_u1*grad_u5*grad_u6 + grad_u2*grad_u3*grad_u7 - grad_u2*grad_u6*weak_obj_tmp1 - grad_u5*grad_u7*weak_obj_tmp0 + weak_obj_tmp0*weak_obj_tmp1*weak_obj_tmp2);
        value[lane] += qw * jacobian_determinant_lane0 * (((scalar_t(1) / scalar_t(2)))*lmbda*pow_2(weak_obj_tmp3) - mu*weak_obj_tmp3 + ((scalar_t(1) / scalar_t(2)))*mu*(pow_2(grad_u1) + pow_2(grad_u2) + pow_2(grad_u3) + pow_2(grad_u5) + pow_2(grad_u6) + pow_2(grad_u7) + pow_2(weak_obj_tmp0) + pow_2(weak_obj_tmp1) + pow_2(weak_obj_tmp2) + scalar_t(-3)));
            }
        }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void neohookean_ogden_d3_simplex_gradient_block(
        const ptrdiff_t nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate4,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate5,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate6,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate7,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate8,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t mu,
        const scalar_t lmbda,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 3],
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 3]
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
            const scalar_t jacobian_adjugate_lane4 = jacobian_adjugate4[geometry_offset];
            const scalar_t jacobian_adjugate_lane5 = jacobian_adjugate5[geometry_offset];
            const scalar_t jacobian_adjugate_lane6 = jacobian_adjugate6[geometry_offset];
            const scalar_t jacobian_adjugate_lane7 = jacobian_adjugate7[geometry_offset];
            const scalar_t jacobian_adjugate_lane8 = jacobian_adjugate8[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const scalar_t qw = q_weight[q];
            scalar_t grad_u_ref0 = scalar_t(0);
            scalar_t grad_u_ref1 = scalar_t(0);
            scalar_t grad_u_ref2 = scalar_t(0);
            scalar_t grad_u_ref3 = scalar_t(0);
            scalar_t grad_u_ref4 = scalar_t(0);
            scalar_t grad_u_ref5 = scalar_t(0);
            scalar_t grad_u_ref6 = scalar_t(0);
            scalar_t grad_u_ref7 = scalar_t(0);
            scalar_t grad_u_ref8 = scalar_t(0);
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                grad_u_ref0 += u_streams[shape * 3 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref1 += u_streams[shape * 3 + 0][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_u_ref2 += u_streams[shape * 3 + 0][lane] * grad_ref_z[q * N_SHAPE + shape];
                grad_u_ref3 += u_streams[shape * 3 + 1][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref4 += u_streams[shape * 3 + 1][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_u_ref5 += u_streams[shape * 3 + 1][lane] * grad_ref_z[q * N_SHAPE + shape];
                grad_u_ref6 += u_streams[shape * 3 + 2][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref7 += u_streams[shape * 3 + 2][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_u_ref8 += u_streams[shape * 3 + 2][lane] * grad_ref_z[q * N_SHAPE + shape];
            }
        const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
        const scalar_t grad_u0 = (grad_u_ref0 * jacobian_adjugate_lane0 + grad_u_ref1 * jacobian_adjugate_lane3 + grad_u_ref2 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t grad_u1 = (grad_u_ref0 * jacobian_adjugate_lane1 + grad_u_ref1 * jacobian_adjugate_lane4 + grad_u_ref2 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t grad_u2 = (grad_u_ref0 * jacobian_adjugate_lane2 + grad_u_ref1 * jacobian_adjugate_lane5 + grad_u_ref2 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t grad_u3 = (grad_u_ref3 * jacobian_adjugate_lane0 + grad_u_ref4 * jacobian_adjugate_lane3 + grad_u_ref5 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t grad_u4 = (grad_u_ref3 * jacobian_adjugate_lane1 + grad_u_ref4 * jacobian_adjugate_lane4 + grad_u_ref5 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t grad_u5 = (grad_u_ref3 * jacobian_adjugate_lane2 + grad_u_ref4 * jacobian_adjugate_lane5 + grad_u_ref5 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t grad_u6 = (grad_u_ref6 * jacobian_adjugate_lane0 + grad_u_ref7 * jacobian_adjugate_lane3 + grad_u_ref8 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t grad_u7 = (grad_u_ref6 * jacobian_adjugate_lane1 + grad_u_ref7 * jacobian_adjugate_lane4 + grad_u_ref8 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t grad_u8 = (grad_u_ref6 * jacobian_adjugate_lane2 + grad_u_ref7 * jacobian_adjugate_lane5 + grad_u_ref8 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t weak_mat_tmp0 = grad_u0 + scalar_t(1);
        const scalar_t weak_mat_tmp1 = grad_u5*grad_u7;
        const scalar_t weak_mat_tmp2 = grad_u4 + scalar_t(1);
        const scalar_t weak_mat_tmp3 = grad_u8 + scalar_t(1);
        const scalar_t weak_mat_tmp4 = -weak_mat_tmp1 + weak_mat_tmp2*weak_mat_tmp3;
        const scalar_t weak_mat_tmp5 = grad_u3*weak_mat_tmp3;
        const scalar_t weak_mat_tmp6 = grad_u6*weak_mat_tmp2;
        const scalar_t weak_mat_tmp7 = grad_u1*grad_u5*grad_u6 - grad_u1*weak_mat_tmp5 + grad_u2*grad_u3*grad_u7 - grad_u2*weak_mat_tmp6 - weak_mat_tmp0*weak_mat_tmp1 + weak_mat_tmp0*weak_mat_tmp2*weak_mat_tmp3;
        const scalar_t weak_mat_tmp8 = pow_m1(weak_mat_tmp7);
        const scalar_t weak_mat_tmp9 = mu*weak_mat_tmp8;
        const scalar_t weak_mat_tmp10 = lmbda*weak_mat_tmp8*log(weak_mat_tmp7);
        const scalar_t weak_mat_tmp11 = grad_u5*grad_u6 - weak_mat_tmp5;
        const scalar_t weak_mat_tmp12 = grad_u3*grad_u7 - weak_mat_tmp6;
        const scalar_t weak_mat_tmp13 = -grad_u1*weak_mat_tmp3 + grad_u2*grad_u7;
        const scalar_t weak_mat_tmp14 = -grad_u2*grad_u6 + weak_mat_tmp0*weak_mat_tmp3;
        const scalar_t weak_mat_tmp15 = grad_u1*grad_u6 - grad_u7*weak_mat_tmp0;
        const scalar_t weak_mat_tmp16 = grad_u1*grad_u5 - grad_u2*weak_mat_tmp2;
        const scalar_t weak_mat_tmp17 = grad_u2*grad_u3 - grad_u5*weak_mat_tmp0;
        const scalar_t weak_mat_tmp18 = -grad_u1*grad_u3 + weak_mat_tmp0*weak_mat_tmp2;
        const scalar_t material0 = mu*weak_mat_tmp0 + weak_mat_tmp10*weak_mat_tmp4 - weak_mat_tmp4*weak_mat_tmp9;
        const scalar_t material1 = grad_u1*mu + weak_mat_tmp10*weak_mat_tmp11 - weak_mat_tmp11*weak_mat_tmp9;
        const scalar_t material2 = grad_u2*mu + weak_mat_tmp10*weak_mat_tmp12 - weak_mat_tmp12*weak_mat_tmp9;
        const scalar_t material3 = grad_u3*mu + weak_mat_tmp10*weak_mat_tmp13 - weak_mat_tmp13*weak_mat_tmp9;
        const scalar_t material4 = mu*weak_mat_tmp2 + weak_mat_tmp10*weak_mat_tmp14 - weak_mat_tmp14*weak_mat_tmp9;
        const scalar_t material5 = grad_u5*mu + weak_mat_tmp10*weak_mat_tmp15 - weak_mat_tmp15*weak_mat_tmp9;
        const scalar_t material6 = grad_u6*mu + weak_mat_tmp10*weak_mat_tmp16 - weak_mat_tmp16*weak_mat_tmp9;
        const scalar_t material7 = grad_u7*mu + weak_mat_tmp10*weak_mat_tmp17 - weak_mat_tmp17*weak_mat_tmp9;
        const scalar_t material8 = mu*weak_mat_tmp3 + weak_mat_tmp10*weak_mat_tmp18 - weak_mat_tmp18*weak_mat_tmp9;
        const scalar_t loperand0 = qw * (material0 * jacobian_adjugate_lane0 + material1 * jacobian_adjugate_lane1 + material2 * jacobian_adjugate_lane2);
        const scalar_t loperand1 = qw * (material0 * jacobian_adjugate_lane3 + material1 * jacobian_adjugate_lane4 + material2 * jacobian_adjugate_lane5);
        const scalar_t loperand2 = qw * (material0 * jacobian_adjugate_lane6 + material1 * jacobian_adjugate_lane7 + material2 * jacobian_adjugate_lane8);
        const scalar_t loperand3 = qw * (material3 * jacobian_adjugate_lane0 + material4 * jacobian_adjugate_lane1 + material5 * jacobian_adjugate_lane2);
        const scalar_t loperand4 = qw * (material3 * jacobian_adjugate_lane3 + material4 * jacobian_adjugate_lane4 + material5 * jacobian_adjugate_lane5);
        const scalar_t loperand5 = qw * (material3 * jacobian_adjugate_lane6 + material4 * jacobian_adjugate_lane7 + material5 * jacobian_adjugate_lane8);
        const scalar_t loperand6 = qw * (material6 * jacobian_adjugate_lane0 + material7 * jacobian_adjugate_lane1 + material8 * jacobian_adjugate_lane2);
        const scalar_t loperand7 = qw * (material6 * jacobian_adjugate_lane3 + material7 * jacobian_adjugate_lane4 + material8 * jacobian_adjugate_lane5);
        const scalar_t loperand8 = qw * (material6 * jacobian_adjugate_lane6 + material7 * jacobian_adjugate_lane7 + material8 * jacobian_adjugate_lane8);
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                out_streams[shape * 3 + 0][lane] += loperand0 * grad_ref_x[q * N_SHAPE + shape] + loperand1 * grad_ref_y[q * N_SHAPE + shape] + loperand2 * grad_ref_z[q * N_SHAPE + shape];
                out_streams[shape * 3 + 1][lane] += loperand3 * grad_ref_x[q * N_SHAPE + shape] + loperand4 * grad_ref_y[q * N_SHAPE + shape] + loperand5 * grad_ref_z[q * N_SHAPE + shape];
                out_streams[shape * 3 + 2][lane] += loperand6 * grad_ref_x[q * N_SHAPE + shape] + loperand7 * grad_ref_y[q * N_SHAPE + shape] + loperand8 * grad_ref_z[q * N_SHAPE + shape];
            }
            }
        }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void neohookean_ogden_d3_simplex_apply_block(
        const ptrdiff_t nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate4,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate5,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate6,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate7,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate8,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t mu,
        const scalar_t lmbda,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 3],
        const scalar_t *const SFEM_RESTRICT h_streams[N_SHAPE * 3],
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 3]
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
            const scalar_t jacobian_adjugate_lane4 = jacobian_adjugate4[geometry_offset];
            const scalar_t jacobian_adjugate_lane5 = jacobian_adjugate5[geometry_offset];
            const scalar_t jacobian_adjugate_lane6 = jacobian_adjugate6[geometry_offset];
            const scalar_t jacobian_adjugate_lane7 = jacobian_adjugate7[geometry_offset];
            const scalar_t jacobian_adjugate_lane8 = jacobian_adjugate8[geometry_offset];
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
            scalar_t grad_u_ref4 = scalar_t(0);
            scalar_t grad_h_ref4 = scalar_t(0);
            scalar_t grad_u_ref5 = scalar_t(0);
            scalar_t grad_h_ref5 = scalar_t(0);
            scalar_t grad_u_ref6 = scalar_t(0);
            scalar_t grad_h_ref6 = scalar_t(0);
            scalar_t grad_u_ref7 = scalar_t(0);
            scalar_t grad_h_ref7 = scalar_t(0);
            scalar_t grad_u_ref8 = scalar_t(0);
            scalar_t grad_h_ref8 = scalar_t(0);
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                grad_u_ref0 += u_streams[shape * 3 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_h_ref0 += h_streams[shape * 3 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref1 += u_streams[shape * 3 + 0][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_h_ref1 += h_streams[shape * 3 + 0][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_u_ref2 += u_streams[shape * 3 + 0][lane] * grad_ref_z[q * N_SHAPE + shape];
                grad_h_ref2 += h_streams[shape * 3 + 0][lane] * grad_ref_z[q * N_SHAPE + shape];
                grad_u_ref3 += u_streams[shape * 3 + 1][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_h_ref3 += h_streams[shape * 3 + 1][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref4 += u_streams[shape * 3 + 1][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_h_ref4 += h_streams[shape * 3 + 1][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_u_ref5 += u_streams[shape * 3 + 1][lane] * grad_ref_z[q * N_SHAPE + shape];
                grad_h_ref5 += h_streams[shape * 3 + 1][lane] * grad_ref_z[q * N_SHAPE + shape];
                grad_u_ref6 += u_streams[shape * 3 + 2][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_h_ref6 += h_streams[shape * 3 + 2][lane] * grad_ref_x[q * N_SHAPE + shape];
                grad_u_ref7 += u_streams[shape * 3 + 2][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_h_ref7 += h_streams[shape * 3 + 2][lane] * grad_ref_y[q * N_SHAPE + shape];
                grad_u_ref8 += u_streams[shape * 3 + 2][lane] * grad_ref_z[q * N_SHAPE + shape];
                grad_h_ref8 += h_streams[shape * 3 + 2][lane] * grad_ref_z[q * N_SHAPE + shape];
            }
        const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
        const scalar_t grad_u0 = (grad_u_ref0 * jacobian_adjugate_lane0 + grad_u_ref1 * jacobian_adjugate_lane3 + grad_u_ref2 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t trial_grad0 = (grad_h_ref0 * jacobian_adjugate_lane0 + grad_h_ref1 * jacobian_adjugate_lane3 + grad_h_ref2 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t grad_u1 = (grad_u_ref0 * jacobian_adjugate_lane1 + grad_u_ref1 * jacobian_adjugate_lane4 + grad_u_ref2 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t trial_grad1 = (grad_h_ref0 * jacobian_adjugate_lane1 + grad_h_ref1 * jacobian_adjugate_lane4 + grad_h_ref2 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t grad_u2 = (grad_u_ref0 * jacobian_adjugate_lane2 + grad_u_ref1 * jacobian_adjugate_lane5 + grad_u_ref2 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t trial_grad2 = (grad_h_ref0 * jacobian_adjugate_lane2 + grad_h_ref1 * jacobian_adjugate_lane5 + grad_h_ref2 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t grad_u3 = (grad_u_ref3 * jacobian_adjugate_lane0 + grad_u_ref4 * jacobian_adjugate_lane3 + grad_u_ref5 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t trial_grad3 = (grad_h_ref3 * jacobian_adjugate_lane0 + grad_h_ref4 * jacobian_adjugate_lane3 + grad_h_ref5 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t grad_u4 = (grad_u_ref3 * jacobian_adjugate_lane1 + grad_u_ref4 * jacobian_adjugate_lane4 + grad_u_ref5 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t trial_grad4 = (grad_h_ref3 * jacobian_adjugate_lane1 + grad_h_ref4 * jacobian_adjugate_lane4 + grad_h_ref5 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t grad_u5 = (grad_u_ref3 * jacobian_adjugate_lane2 + grad_u_ref4 * jacobian_adjugate_lane5 + grad_u_ref5 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t trial_grad5 = (grad_h_ref3 * jacobian_adjugate_lane2 + grad_h_ref4 * jacobian_adjugate_lane5 + grad_h_ref5 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t grad_u6 = (grad_u_ref6 * jacobian_adjugate_lane0 + grad_u_ref7 * jacobian_adjugate_lane3 + grad_u_ref8 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t trial_grad6 = (grad_h_ref6 * jacobian_adjugate_lane0 + grad_h_ref7 * jacobian_adjugate_lane3 + grad_h_ref8 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t grad_u7 = (grad_u_ref6 * jacobian_adjugate_lane1 + grad_u_ref7 * jacobian_adjugate_lane4 + grad_u_ref8 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t trial_grad7 = (grad_h_ref6 * jacobian_adjugate_lane1 + grad_h_ref7 * jacobian_adjugate_lane4 + grad_h_ref8 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t grad_u8 = (grad_u_ref6 * jacobian_adjugate_lane2 + grad_u_ref7 * jacobian_adjugate_lane5 + grad_u_ref8 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t trial_grad8 = (grad_h_ref6 * jacobian_adjugate_lane2 + grad_h_ref7 * jacobian_adjugate_lane5 + grad_h_ref8 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t weak_mat_tmp0 = grad_u5*grad_u7;
        const scalar_t weak_mat_tmp1 = grad_u4 + scalar_t(1);
        const scalar_t weak_mat_tmp2 = grad_u8 + scalar_t(1);
        const scalar_t weak_mat_tmp3 = weak_mat_tmp0 - weak_mat_tmp1*weak_mat_tmp2;
        const scalar_t weak_mat_tmp4 = -weak_mat_tmp3;
        const scalar_t weak_mat_tmp5 = grad_u3*weak_mat_tmp2;
        const scalar_t weak_mat_tmp6 = grad_u6*weak_mat_tmp1;
        const scalar_t weak_mat_tmp7 = grad_u0 + scalar_t(1);
        const scalar_t weak_mat_tmp8 = grad_u1*grad_u5*grad_u6 - grad_u1*weak_mat_tmp5 + grad_u2*grad_u3*grad_u7 - grad_u2*weak_mat_tmp6 - weak_mat_tmp0*weak_mat_tmp7 + weak_mat_tmp1*weak_mat_tmp2*weak_mat_tmp7;
        const scalar_t weak_mat_tmp9 = pow_m2(weak_mat_tmp8);
        const scalar_t weak_mat_tmp10 = lmbda*weak_mat_tmp9;
        const scalar_t weak_mat_tmp11 = mu*weak_mat_tmp9;
        const scalar_t weak_mat_tmp12 = weak_mat_tmp3*weak_mat_tmp4;
        const scalar_t weak_mat_tmp13 = log(weak_mat_tmp8);
        const scalar_t weak_mat_tmp14 = weak_mat_tmp10*weak_mat_tmp13;
        const scalar_t weak_mat_tmp15 = -grad_u5*grad_u6 + weak_mat_tmp5;
        const scalar_t weak_mat_tmp16 = -weak_mat_tmp15;
        const scalar_t weak_mat_tmp17 = weak_mat_tmp10*weak_mat_tmp4;
        const scalar_t weak_mat_tmp18 = weak_mat_tmp16*weak_mat_tmp17;
        const scalar_t weak_mat_tmp19 = weak_mat_tmp11*weak_mat_tmp4;
        const scalar_t weak_mat_tmp20 = weak_mat_tmp13*weak_mat_tmp17;
        const scalar_t weak_mat_tmp21 = grad_u3*grad_u7 - weak_mat_tmp6;
        const scalar_t weak_mat_tmp22 = weak_mat_tmp17*weak_mat_tmp21;
        const scalar_t weak_mat_tmp23 = -weak_mat_tmp21;
        const scalar_t weak_mat_tmp24 = grad_u1*weak_mat_tmp2 - grad_u2*grad_u7;
        const scalar_t weak_mat_tmp25 = -weak_mat_tmp24;
        const scalar_t weak_mat_tmp26 = weak_mat_tmp17*weak_mat_tmp25;
        const scalar_t weak_mat_tmp27 = grad_u1*grad_u5 - grad_u2*weak_mat_tmp1;
        const scalar_t weak_mat_tmp28 = weak_mat_tmp17*weak_mat_tmp27;
        const scalar_t weak_mat_tmp29 = -weak_mat_tmp27;
        const scalar_t weak_mat_tmp30 = grad_u1*grad_u6 - grad_u7*weak_mat_tmp7;
        const scalar_t weak_mat_tmp31 = -weak_mat_tmp30;
        const scalar_t weak_mat_tmp32 = pow_m1(weak_mat_tmp8);
        const scalar_t weak_mat_tmp33 = mu*weak_mat_tmp32;
        const scalar_t weak_mat_tmp34 = grad_u7*weak_mat_tmp33;
        const scalar_t weak_mat_tmp35 = lmbda*weak_mat_tmp13*weak_mat_tmp32;
        const scalar_t weak_mat_tmp36 = grad_u7*weak_mat_tmp35;
        const scalar_t weak_mat_tmp37 = weak_mat_tmp17*weak_mat_tmp30 + weak_mat_tmp34 - weak_mat_tmp36;
        const scalar_t weak_mat_tmp38 = grad_u2*grad_u3 - grad_u5*weak_mat_tmp7;
        const scalar_t weak_mat_tmp39 = -weak_mat_tmp38;
        const scalar_t weak_mat_tmp40 = grad_u5*weak_mat_tmp33;
        const scalar_t weak_mat_tmp41 = grad_u5*weak_mat_tmp35;
        const scalar_t weak_mat_tmp42 = weak_mat_tmp17*weak_mat_tmp38 + weak_mat_tmp40 - weak_mat_tmp41;
        const scalar_t weak_mat_tmp43 = grad_u2*grad_u6 - weak_mat_tmp2*weak_mat_tmp7;
        const scalar_t weak_mat_tmp44 = weak_mat_tmp2*weak_mat_tmp33;
        const scalar_t weak_mat_tmp45 = weak_mat_tmp2*weak_mat_tmp35;
        const scalar_t weak_mat_tmp46 = -weak_mat_tmp43;
        const scalar_t weak_mat_tmp47 = weak_mat_tmp17*weak_mat_tmp46 - weak_mat_tmp44 + weak_mat_tmp45;
        const scalar_t weak_mat_tmp48 = grad_u1*grad_u3 - weak_mat_tmp1*weak_mat_tmp7;
        const scalar_t weak_mat_tmp49 = weak_mat_tmp1*weak_mat_tmp33;
        const scalar_t weak_mat_tmp50 = weak_mat_tmp1*weak_mat_tmp35;
        const scalar_t weak_mat_tmp51 = -weak_mat_tmp48;
        const scalar_t weak_mat_tmp52 = weak_mat_tmp17*weak_mat_tmp51 - weak_mat_tmp49 + weak_mat_tmp50;
        const scalar_t weak_mat_tmp53 = weak_mat_tmp11*weak_mat_tmp16;
        const scalar_t weak_mat_tmp54 = weak_mat_tmp10*weak_mat_tmp16;
        const scalar_t weak_mat_tmp55 = weak_mat_tmp13*weak_mat_tmp54;
        const scalar_t weak_mat_tmp56 = weak_mat_tmp21*weak_mat_tmp54;
        const scalar_t weak_mat_tmp57 = weak_mat_tmp38*weak_mat_tmp54;
        const scalar_t weak_mat_tmp58 = weak_mat_tmp46*weak_mat_tmp54;
        const scalar_t weak_mat_tmp59 = grad_u6*weak_mat_tmp33;
        const scalar_t weak_mat_tmp60 = grad_u6*weak_mat_tmp35;
        const scalar_t weak_mat_tmp61 = weak_mat_tmp30*weak_mat_tmp54 - weak_mat_tmp59 + weak_mat_tmp60;
        const scalar_t weak_mat_tmp62 = weak_mat_tmp27*weak_mat_tmp54 - weak_mat_tmp40 + weak_mat_tmp41;
        const scalar_t weak_mat_tmp63 = weak_mat_tmp25*weak_mat_tmp54 + weak_mat_tmp44 - weak_mat_tmp45;
        const scalar_t weak_mat_tmp64 = grad_u3*weak_mat_tmp33;
        const scalar_t weak_mat_tmp65 = grad_u3*weak_mat_tmp35;
        const scalar_t weak_mat_tmp66 = weak_mat_tmp51*weak_mat_tmp54 + weak_mat_tmp64 - weak_mat_tmp65;
        const scalar_t weak_mat_tmp67 = weak_mat_tmp11*weak_mat_tmp21;
        const scalar_t weak_mat_tmp68 = weak_mat_tmp10*weak_mat_tmp21;
        const scalar_t weak_mat_tmp69 = weak_mat_tmp13*weak_mat_tmp68;
        const scalar_t weak_mat_tmp70 = weak_mat_tmp30*weak_mat_tmp68;
        const scalar_t weak_mat_tmp71 = weak_mat_tmp51*weak_mat_tmp68;
        const scalar_t weak_mat_tmp72 = weak_mat_tmp25*weak_mat_tmp68 - weak_mat_tmp34 + weak_mat_tmp36;
        const scalar_t weak_mat_tmp73 = weak_mat_tmp38*weak_mat_tmp68 - weak_mat_tmp64 + weak_mat_tmp65;
        const scalar_t weak_mat_tmp74 = weak_mat_tmp27*weak_mat_tmp68 + weak_mat_tmp49 - weak_mat_tmp50;
        const scalar_t weak_mat_tmp75 = weak_mat_tmp46*weak_mat_tmp68 + weak_mat_tmp59 - weak_mat_tmp60;
        const scalar_t weak_mat_tmp76 = weak_mat_tmp11*weak_mat_tmp25;
        const scalar_t weak_mat_tmp77 = weak_mat_tmp10*weak_mat_tmp25;
        const scalar_t weak_mat_tmp78 = weak_mat_tmp13*weak_mat_tmp77;
        const scalar_t weak_mat_tmp79 = weak_mat_tmp30*weak_mat_tmp77;
        const scalar_t weak_mat_tmp80 = weak_mat_tmp27*weak_mat_tmp77;
        const scalar_t weak_mat_tmp81 = weak_mat_tmp46*weak_mat_tmp77;
        const scalar_t weak_mat_tmp82 = grad_u2*weak_mat_tmp33;
        const scalar_t weak_mat_tmp83 = grad_u2*weak_mat_tmp35;
        const scalar_t weak_mat_tmp84 = weak_mat_tmp38*weak_mat_tmp77 - weak_mat_tmp82 + weak_mat_tmp83;
        const scalar_t weak_mat_tmp85 = grad_u1*weak_mat_tmp33;
        const scalar_t weak_mat_tmp86 = grad_u1*weak_mat_tmp35;
        const scalar_t weak_mat_tmp87 = weak_mat_tmp51*weak_mat_tmp77 + weak_mat_tmp85 - weak_mat_tmp86;
        const scalar_t weak_mat_tmp88 = weak_mat_tmp11*weak_mat_tmp46;
        const scalar_t weak_mat_tmp89 = weak_mat_tmp10*weak_mat_tmp46;
        const scalar_t weak_mat_tmp90 = weak_mat_tmp13*weak_mat_tmp89;
        const scalar_t weak_mat_tmp91 = weak_mat_tmp30*weak_mat_tmp89;
        const scalar_t weak_mat_tmp92 = weak_mat_tmp38*weak_mat_tmp89;
        const scalar_t weak_mat_tmp93 = weak_mat_tmp27*weak_mat_tmp89 + weak_mat_tmp82 - weak_mat_tmp83;
        const scalar_t weak_mat_tmp94 = weak_mat_tmp33*weak_mat_tmp7;
        const scalar_t weak_mat_tmp95 = weak_mat_tmp35*weak_mat_tmp7;
        const scalar_t weak_mat_tmp96 = weak_mat_tmp51*weak_mat_tmp89 - weak_mat_tmp94 + weak_mat_tmp95;
        const scalar_t weak_mat_tmp97 = weak_mat_tmp11*weak_mat_tmp30;
        const scalar_t weak_mat_tmp98 = weak_mat_tmp10*weak_mat_tmp30;
        const scalar_t weak_mat_tmp99 = weak_mat_tmp13*weak_mat_tmp98;
        const scalar_t weak_mat_tmp100 = weak_mat_tmp51*weak_mat_tmp98;
        const scalar_t weak_mat_tmp101 = weak_mat_tmp27*weak_mat_tmp98 - weak_mat_tmp85 + weak_mat_tmp86;
        const scalar_t weak_mat_tmp102 = weak_mat_tmp38*weak_mat_tmp98 + weak_mat_tmp94 - weak_mat_tmp95;
        const scalar_t weak_mat_tmp103 = weak_mat_tmp11*weak_mat_tmp27;
        const scalar_t weak_mat_tmp104 = weak_mat_tmp10*weak_mat_tmp27;
        const scalar_t weak_mat_tmp105 = weak_mat_tmp104*weak_mat_tmp13;
        const scalar_t weak_mat_tmp106 = weak_mat_tmp104*weak_mat_tmp38;
        const scalar_t weak_mat_tmp107 = weak_mat_tmp104*weak_mat_tmp51;
        const scalar_t weak_mat_tmp108 = weak_mat_tmp11*weak_mat_tmp38;
        const scalar_t weak_mat_tmp109 = weak_mat_tmp10*weak_mat_tmp38;
        const scalar_t weak_mat_tmp110 = weak_mat_tmp109*weak_mat_tmp13;
        const scalar_t weak_mat_tmp111 = weak_mat_tmp109*weak_mat_tmp51;
        const scalar_t weak_mat_tmp112 = weak_mat_tmp11*weak_mat_tmp51;
        const scalar_t weak_mat_tmp113 = weak_mat_tmp14*weak_mat_tmp51;
        const scalar_t material0 = trial_grad0*(mu + weak_mat_tmp10*pow_2(weak_mat_tmp4) - weak_mat_tmp11*weak_mat_tmp12 + weak_mat_tmp12*weak_mat_tmp14) + trial_grad1*(-weak_mat_tmp15*weak_mat_tmp19 + weak_mat_tmp15*weak_mat_tmp20 + weak_mat_tmp18) + trial_grad2*(-weak_mat_tmp19*weak_mat_tmp23 + weak_mat_tmp20*weak_mat_tmp23 + weak_mat_tmp22) + trial_grad3*(-weak_mat_tmp19*weak_mat_tmp24 + weak_mat_tmp20*weak_mat_tmp24 + weak_mat_tmp26) + trial_grad4*(-weak_mat_tmp19*weak_mat_tmp43 + weak_mat_tmp20*weak_mat_tmp43 + weak_mat_tmp47) + trial_grad5*(-weak_mat_tmp19*weak_mat_tmp31 + weak_mat_tmp20*weak_mat_tmp31 + weak_mat_tmp37) + trial_grad6*(-weak_mat_tmp19*weak_mat_tmp29 + weak_mat_tmp20*weak_mat_tmp29 + weak_mat_tmp28) + trial_grad7*(-weak_mat_tmp19*weak_mat_tmp39 + weak_mat_tmp20*weak_mat_tmp39 + weak_mat_tmp42) + trial_grad8*(-weak_mat_tmp19*weak_mat_tmp48 + weak_mat_tmp20*weak_mat_tmp48 + weak_mat_tmp52);
        const scalar_t material1 = trial_grad0*(weak_mat_tmp18 - weak_mat_tmp3*weak_mat_tmp53 + weak_mat_tmp3*weak_mat_tmp55) + trial_grad1*(mu + weak_mat_tmp10*pow_2(weak_mat_tmp16) - weak_mat_tmp15*weak_mat_tmp53 + weak_mat_tmp15*weak_mat_tmp55) + trial_grad2*(-weak_mat_tmp23*weak_mat_tmp53 + weak_mat_tmp23*weak_mat_tmp55 + weak_mat_tmp56) + trial_grad3*(-weak_mat_tmp24*weak_mat_tmp53 + weak_mat_tmp24*weak_mat_tmp55 + weak_mat_tmp63) + trial_grad4*(-weak_mat_tmp43*weak_mat_tmp53 + weak_mat_tmp43*weak_mat_tmp55 + weak_mat_tmp58) + trial_grad5*(-weak_mat_tmp31*weak_mat_tmp53 + weak_mat_tmp31*weak_mat_tmp55 + weak_mat_tmp61) + trial_grad6*(-weak_mat_tmp29*weak_mat_tmp53 + weak_mat_tmp29*weak_mat_tmp55 + weak_mat_tmp62) + trial_grad7*(-weak_mat_tmp39*weak_mat_tmp53 + weak_mat_tmp39*weak_mat_tmp55 + weak_mat_tmp57) + trial_grad8*(-weak_mat_tmp48*weak_mat_tmp53 + weak_mat_tmp48*weak_mat_tmp55 + weak_mat_tmp66);
        const scalar_t material2 = trial_grad0*(weak_mat_tmp22 - weak_mat_tmp3*weak_mat_tmp67 + weak_mat_tmp3*weak_mat_tmp69) + trial_grad1*(-weak_mat_tmp15*weak_mat_tmp67 + weak_mat_tmp15*weak_mat_tmp69 + weak_mat_tmp56) + trial_grad2*(mu + weak_mat_tmp10*pow_2(weak_mat_tmp21) - weak_mat_tmp23*weak_mat_tmp67 + weak_mat_tmp23*weak_mat_tmp69) + trial_grad3*(-weak_mat_tmp24*weak_mat_tmp67 + weak_mat_tmp24*weak_mat_tmp69 + weak_mat_tmp72) + trial_grad4*(-weak_mat_tmp43*weak_mat_tmp67 + weak_mat_tmp43*weak_mat_tmp69 + weak_mat_tmp75) + trial_grad5*(-weak_mat_tmp31*weak_mat_tmp67 + weak_mat_tmp31*weak_mat_tmp69 + weak_mat_tmp70) + trial_grad6*(-weak_mat_tmp29*weak_mat_tmp67 + weak_mat_tmp29*weak_mat_tmp69 + weak_mat_tmp74) + trial_grad7*(-weak_mat_tmp39*weak_mat_tmp67 + weak_mat_tmp39*weak_mat_tmp69 + weak_mat_tmp73) + trial_grad8*(-weak_mat_tmp48*weak_mat_tmp67 + weak_mat_tmp48*weak_mat_tmp69 + weak_mat_tmp71);
        const scalar_t material3 = trial_grad0*(weak_mat_tmp26 - weak_mat_tmp3*weak_mat_tmp76 + weak_mat_tmp3*weak_mat_tmp78) + trial_grad1*(-weak_mat_tmp15*weak_mat_tmp76 + weak_mat_tmp15*weak_mat_tmp78 + weak_mat_tmp63) + trial_grad2*(-weak_mat_tmp23*weak_mat_tmp76 + weak_mat_tmp23*weak_mat_tmp78 + weak_mat_tmp72) + trial_grad3*(mu + weak_mat_tmp10*pow_2(weak_mat_tmp25) - weak_mat_tmp24*weak_mat_tmp76 + weak_mat_tmp24*weak_mat_tmp78) + trial_grad4*(-weak_mat_tmp43*weak_mat_tmp76 + weak_mat_tmp43*weak_mat_tmp78 + weak_mat_tmp81) + trial_grad5*(-weak_mat_tmp31*weak_mat_tmp76 + weak_mat_tmp31*weak_mat_tmp78 + weak_mat_tmp79) + trial_grad6*(-weak_mat_tmp29*weak_mat_tmp76 + weak_mat_tmp29*weak_mat_tmp78 + weak_mat_tmp80) + trial_grad7*(-weak_mat_tmp39*weak_mat_tmp76 + weak_mat_tmp39*weak_mat_tmp78 + weak_mat_tmp84) + trial_grad8*(-weak_mat_tmp48*weak_mat_tmp76 + weak_mat_tmp48*weak_mat_tmp78 + weak_mat_tmp87);
        const scalar_t material4 = trial_grad0*(-weak_mat_tmp3*weak_mat_tmp88 + weak_mat_tmp3*weak_mat_tmp90 + weak_mat_tmp47) + trial_grad1*(-weak_mat_tmp15*weak_mat_tmp88 + weak_mat_tmp15*weak_mat_tmp90 + weak_mat_tmp58) + trial_grad2*(-weak_mat_tmp23*weak_mat_tmp88 + weak_mat_tmp23*weak_mat_tmp90 + weak_mat_tmp75) + trial_grad3*(-weak_mat_tmp24*weak_mat_tmp88 + weak_mat_tmp24*weak_mat_tmp90 + weak_mat_tmp81) + trial_grad4*(mu + weak_mat_tmp10*pow_2(weak_mat_tmp46) - weak_mat_tmp43*weak_mat_tmp88 + weak_mat_tmp43*weak_mat_tmp90) + trial_grad5*(-weak_mat_tmp31*weak_mat_tmp88 + weak_mat_tmp31*weak_mat_tmp90 + weak_mat_tmp91) + trial_grad6*(-weak_mat_tmp29*weak_mat_tmp88 + weak_mat_tmp29*weak_mat_tmp90 + weak_mat_tmp93) + trial_grad7*(-weak_mat_tmp39*weak_mat_tmp88 + weak_mat_tmp39*weak_mat_tmp90 + weak_mat_tmp92) + trial_grad8*(-weak_mat_tmp48*weak_mat_tmp88 + weak_mat_tmp48*weak_mat_tmp90 + weak_mat_tmp96);
        const scalar_t material5 = trial_grad0*(-weak_mat_tmp3*weak_mat_tmp97 + weak_mat_tmp3*weak_mat_tmp99 + weak_mat_tmp37) + trial_grad1*(-weak_mat_tmp15*weak_mat_tmp97 + weak_mat_tmp15*weak_mat_tmp99 + weak_mat_tmp61) + trial_grad2*(-weak_mat_tmp23*weak_mat_tmp97 + weak_mat_tmp23*weak_mat_tmp99 + weak_mat_tmp70) + trial_grad3*(-weak_mat_tmp24*weak_mat_tmp97 + weak_mat_tmp24*weak_mat_tmp99 + weak_mat_tmp79) + trial_grad4*(-weak_mat_tmp43*weak_mat_tmp97 + weak_mat_tmp43*weak_mat_tmp99 + weak_mat_tmp91) + trial_grad5*(mu + weak_mat_tmp10*pow_2(weak_mat_tmp30) - weak_mat_tmp31*weak_mat_tmp97 + weak_mat_tmp31*weak_mat_tmp99) + trial_grad6*(weak_mat_tmp101 - weak_mat_tmp29*weak_mat_tmp97 + weak_mat_tmp29*weak_mat_tmp99) + trial_grad7*(weak_mat_tmp102 - weak_mat_tmp39*weak_mat_tmp97 + weak_mat_tmp39*weak_mat_tmp99) + trial_grad8*(weak_mat_tmp100 - weak_mat_tmp48*weak_mat_tmp97 + weak_mat_tmp48*weak_mat_tmp99);
        const scalar_t material6 = trial_grad0*(-weak_mat_tmp103*weak_mat_tmp3 + weak_mat_tmp105*weak_mat_tmp3 + weak_mat_tmp28) + trial_grad1*(-weak_mat_tmp103*weak_mat_tmp15 + weak_mat_tmp105*weak_mat_tmp15 + weak_mat_tmp62) + trial_grad2*(-weak_mat_tmp103*weak_mat_tmp23 + weak_mat_tmp105*weak_mat_tmp23 + weak_mat_tmp74) + trial_grad3*(-weak_mat_tmp103*weak_mat_tmp24 + weak_mat_tmp105*weak_mat_tmp24 + weak_mat_tmp80) + trial_grad4*(-weak_mat_tmp103*weak_mat_tmp43 + weak_mat_tmp105*weak_mat_tmp43 + weak_mat_tmp93) + trial_grad5*(weak_mat_tmp101 - weak_mat_tmp103*weak_mat_tmp31 + weak_mat_tmp105*weak_mat_tmp31) + trial_grad6*(mu + weak_mat_tmp10*pow_2(weak_mat_tmp27) - weak_mat_tmp103*weak_mat_tmp29 + weak_mat_tmp105*weak_mat_tmp29) + trial_grad7*(-weak_mat_tmp103*weak_mat_tmp39 + weak_mat_tmp105*weak_mat_tmp39 + weak_mat_tmp106) + trial_grad8*(-weak_mat_tmp103*weak_mat_tmp48 + weak_mat_tmp105*weak_mat_tmp48 + weak_mat_tmp107);
        const scalar_t material7 = trial_grad0*(-weak_mat_tmp108*weak_mat_tmp3 + weak_mat_tmp110*weak_mat_tmp3 + weak_mat_tmp42) + trial_grad1*(-weak_mat_tmp108*weak_mat_tmp15 + weak_mat_tmp110*weak_mat_tmp15 + weak_mat_tmp57) + trial_grad2*(-weak_mat_tmp108*weak_mat_tmp23 + weak_mat_tmp110*weak_mat_tmp23 + weak_mat_tmp73) + trial_grad3*(-weak_mat_tmp108*weak_mat_tmp24 + weak_mat_tmp110*weak_mat_tmp24 + weak_mat_tmp84) + trial_grad4*(-weak_mat_tmp108*weak_mat_tmp43 + weak_mat_tmp110*weak_mat_tmp43 + weak_mat_tmp92) + trial_grad5*(weak_mat_tmp102 - weak_mat_tmp108*weak_mat_tmp31 + weak_mat_tmp110*weak_mat_tmp31) + trial_grad6*(weak_mat_tmp106 - weak_mat_tmp108*weak_mat_tmp29 + weak_mat_tmp110*weak_mat_tmp29) + trial_grad7*(mu + weak_mat_tmp10*pow_2(weak_mat_tmp38) - weak_mat_tmp108*weak_mat_tmp39 + weak_mat_tmp110*weak_mat_tmp39) + trial_grad8*(-weak_mat_tmp108*weak_mat_tmp48 + weak_mat_tmp110*weak_mat_tmp48 + weak_mat_tmp111);
        const scalar_t material8 = trial_grad0*(-weak_mat_tmp112*weak_mat_tmp3 + weak_mat_tmp113*weak_mat_tmp3 + weak_mat_tmp52) + trial_grad1*(-weak_mat_tmp112*weak_mat_tmp15 + weak_mat_tmp113*weak_mat_tmp15 + weak_mat_tmp66) + trial_grad2*(-weak_mat_tmp112*weak_mat_tmp23 + weak_mat_tmp113*weak_mat_tmp23 + weak_mat_tmp71) + trial_grad3*(-weak_mat_tmp112*weak_mat_tmp24 + weak_mat_tmp113*weak_mat_tmp24 + weak_mat_tmp87) + trial_grad4*(-weak_mat_tmp112*weak_mat_tmp43 + weak_mat_tmp113*weak_mat_tmp43 + weak_mat_tmp96) + trial_grad5*(weak_mat_tmp100 - weak_mat_tmp112*weak_mat_tmp31 + weak_mat_tmp113*weak_mat_tmp31) + trial_grad6*(weak_mat_tmp107 - weak_mat_tmp112*weak_mat_tmp29 + weak_mat_tmp113*weak_mat_tmp29) + trial_grad7*(weak_mat_tmp111 - weak_mat_tmp112*weak_mat_tmp39 + weak_mat_tmp113*weak_mat_tmp39) + trial_grad8*(mu + weak_mat_tmp10*pow_2(weak_mat_tmp51) - weak_mat_tmp112*weak_mat_tmp48 + weak_mat_tmp113*weak_mat_tmp48);
        const scalar_t loperand0 = qw * (material0 * jacobian_adjugate_lane0 + material1 * jacobian_adjugate_lane1 + material2 * jacobian_adjugate_lane2);
        const scalar_t loperand1 = qw * (material0 * jacobian_adjugate_lane3 + material1 * jacobian_adjugate_lane4 + material2 * jacobian_adjugate_lane5);
        const scalar_t loperand2 = qw * (material0 * jacobian_adjugate_lane6 + material1 * jacobian_adjugate_lane7 + material2 * jacobian_adjugate_lane8);
        const scalar_t loperand3 = qw * (material3 * jacobian_adjugate_lane0 + material4 * jacobian_adjugate_lane1 + material5 * jacobian_adjugate_lane2);
        const scalar_t loperand4 = qw * (material3 * jacobian_adjugate_lane3 + material4 * jacobian_adjugate_lane4 + material5 * jacobian_adjugate_lane5);
        const scalar_t loperand5 = qw * (material3 * jacobian_adjugate_lane6 + material4 * jacobian_adjugate_lane7 + material5 * jacobian_adjugate_lane8);
        const scalar_t loperand6 = qw * (material6 * jacobian_adjugate_lane0 + material7 * jacobian_adjugate_lane1 + material8 * jacobian_adjugate_lane2);
        const scalar_t loperand7 = qw * (material6 * jacobian_adjugate_lane3 + material7 * jacobian_adjugate_lane4 + material8 * jacobian_adjugate_lane5);
        const scalar_t loperand8 = qw * (material6 * jacobian_adjugate_lane6 + material7 * jacobian_adjugate_lane7 + material8 * jacobian_adjugate_lane8);
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                out_streams[shape * 3 + 0][lane] += loperand0 * grad_ref_x[q * N_SHAPE + shape] + loperand1 * grad_ref_y[q * N_SHAPE + shape] + loperand2 * grad_ref_z[q * N_SHAPE + shape];
                out_streams[shape * 3 + 1][lane] += loperand3 * grad_ref_x[q * N_SHAPE + shape] + loperand4 * grad_ref_y[q * N_SHAPE + shape] + loperand5 * grad_ref_z[q * N_SHAPE + shape];
                out_streams[shape * 3 + 2][lane] += loperand6 * grad_ref_x[q * N_SHAPE + shape] + loperand7 * grad_ref_y[q * N_SHAPE + shape] + loperand8 * grad_ref_z[q * N_SHAPE + shape];
            }
            }
        }
}

} // namespace codegen
} // namespace sfem

#endif
