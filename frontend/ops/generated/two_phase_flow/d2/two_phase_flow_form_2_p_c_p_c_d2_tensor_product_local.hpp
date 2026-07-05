#ifndef TWO_PHASE_FLOW_FORM_2_P_C_P_C_D2_TENSOR_PRODUCT_LOCAL_HPP
#define TWO_PHASE_FLOW_FORM_2_P_C_P_C_D2_TENSOR_PRODUCT_LOCAL_HPP

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
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_c_d2_tensor_product_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_c_d2_tensor_product_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t *const SFEM_RESTRICT current[2 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT direction[2 * N_SHAPE],
        const scalar_t C_ka1,
        const scalar_t C_ka2,
        const scalar_t K_0,
        const scalar_t K_1,
        const scalar_t K_2,
        const scalar_t K_3,
        const scalar_t M_c,
        const scalar_t P_r,
        const scalar_t R,
        const scalar_t S_res,
        const scalar_t T,
        const scalar_t Z,
        const scalar_t dt,
        const scalar_t m,
        const scalar_t mu_c,
        const scalar_t porosity,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
    scalar_t current_value[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t current_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, current, current_value, current_grad_ref);
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
            const scalar_t p_w = current_value[(0 * N_QP + q) * VECTOR_SIZE + lane];
            const scalar_t p_w_grad_0_ref = current_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t p_w_grad_1_ref = current_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj2) / det;
            const scalar_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj3) / det;
            const scalar_t p_w_direction = direction_value[(0 * N_QP + q) * VECTOR_SIZE + lane];
            const scalar_t p_w_direction_grad_0_ref = direction_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t p_w_direction_grad_1_ref = direction_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t p_w_direction_grad_0 = (p_w_direction_grad_0_ref * adj0 + p_w_direction_grad_1_ref * adj2) / det;
            const scalar_t p_w_direction_grad_1 = (p_w_direction_grad_0_ref * adj1 + p_w_direction_grad_1_ref * adj3) / det;
            const scalar_t p_c = current_value[(1 * N_QP + q) * VECTOR_SIZE + lane];
            const scalar_t p_c_grad_0_ref = current_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t p_c_grad_1_ref = current_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj2) / det;
            const scalar_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj3) / det;
            const scalar_t p_c_direction = direction_value[(1 * N_QP + q) * VECTOR_SIZE + lane];
            const scalar_t p_c_direction_grad_0_ref = direction_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t p_c_direction_grad_1_ref = direction_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t p_c_direction_grad_0 = (p_c_direction_grad_0_ref * adj0 + p_c_direction_grad_1_ref * adj2) / det;
            const scalar_t p_c_direction_grad_1 = (p_c_direction_grad_0_ref * adj1 + p_c_direction_grad_1_ref * adj3) / det;
            const scalar_t residual_tmp0 = p_c - p_w;
            const scalar_t residual_tmp1 = pow(residual_tmp0/P_r, m);
            const scalar_t residual_tmp2 = residual_tmp1 + scalar_t(1);
            const scalar_t residual_tmp3 = scalar_t(1) - m;
            const scalar_t residual_tmp4 = pow(residual_tmp2, residual_tmp3/m);
            const scalar_t residual_tmp5 = residual_tmp4*(S_res + scalar_t(-1));
            const scalar_t residual_tmp6 = p_c*residual_tmp1*residual_tmp3/(residual_tmp0*residual_tmp2);
            const scalar_t residual_tmp7 = pow_m1(dt);
            const scalar_t residual_tmp8 = pow_m1(R);
            const scalar_t residual_tmp9 = pow_m1(T);
            const scalar_t residual_tmp10 = pow_m1(Z);
            const scalar_t residual_tmp11 = M_c*residual_tmp10*residual_tmp8*residual_tmp9;
            const scalar_t residual_tmp12 = pow_m1(mu_c);
            const scalar_t residual_tmp13 = scalar_t(1) - residual_tmp4;
            const scalar_t residual_tmp14 = pow(residual_tmp13, C_ka1);
            const scalar_t residual_tmp15 = pow(residual_tmp4, C_ka2);
            const scalar_t residual_tmp16 = residual_tmp14*(residual_tmp15 + scalar_t(-1));
            const scalar_t residual_tmp17 = p_c*residual_tmp11*residual_tmp12*residual_tmp16;
            const scalar_t residual_tmp18 = p_c_direction_grad_0*residual_tmp17;
            const scalar_t residual_tmp19 = p_c_direction_grad_1*residual_tmp17;
            const scalar_t residual_tmp20 = -K_0*p_c_grad_0 - K_1*p_c_grad_1;
            const scalar_t residual_tmp21 = dt*residual_tmp16;
            const scalar_t residual_tmp22 = residual_tmp20*residual_tmp21;
            const scalar_t residual_tmp23 = C_ka2*dt*residual_tmp14*residual_tmp15*residual_tmp6;
            const scalar_t residual_tmp24 = C_ka1*residual_tmp4*residual_tmp6/residual_tmp13;
            const scalar_t residual_tmp25 = -K_2*p_c_grad_0 - K_3*p_c_grad_1;
            const scalar_t residual_tmp26 = residual_tmp21*residual_tmp25;
            const scalar_t value_coeff1 = -p_c_direction*porosity*residual_tmp11*residual_tmp7*(S_res - residual_tmp5*residual_tmp6 - residual_tmp5 + scalar_t(-1));
            const scalar_t grad_coeff1_0 = -K_0*residual_tmp18 - K_1*residual_tmp19 + M_c*p_c_direction*residual_tmp10*residual_tmp12*residual_tmp7*residual_tmp8*residual_tmp9*(residual_tmp20*residual_tmp23 - residual_tmp22*residual_tmp24 + residual_tmp22);
            const scalar_t grad_coeff1_1 = -K_2*residual_tmp18 - K_3*residual_tmp19 + M_c*p_c_direction*residual_tmp10*residual_tmp12*residual_tmp7*residual_tmp8*residual_tmp9*(residual_tmp23*residual_tmp25 - residual_tmp24*residual_tmp26 + residual_tmp26);
            value_coeff[(0 * N_QP + q) * VECTOR_SIZE + lane] = scalar_t(0);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = scalar_t(0);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = scalar_t(0);
            value_coeff[(1 * N_QP + q) * VECTOR_SIZE + lane] = qw * det * value_coeff1;
            grad_coeff_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
            grad_coeff_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
        }
    }
    tensor_integrate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

} // namespace codegen
} // namespace sfem

#endif
