#ifndef TWO_PHASE_FLOW_FORM_1_P_W_D3_TENSOR_PRODUCT_LOCAL_HPP
#define TWO_PHASE_FLOW_FORM_1_P_W_D3_TENSOR_PRODUCT_LOCAL_HPP

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
static SFEM_INLINE void two_phase_flow_form_1_p_w_d3_tensor_product_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t *const SFEM_RESTRICT current[2 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT previous[2 * N_SHAPE],
        const scalar_t C_kw1,
        const scalar_t K_0,
        const scalar_t K_1,
        const scalar_t K_2,
        const scalar_t K_3,
        const scalar_t K_4,
        const scalar_t K_5,
        const scalar_t K_6,
        const scalar_t K_7,
        const scalar_t K_8,
        const scalar_t P_r,
        const scalar_t S_res,
        const scalar_t dt,
        const scalar_t kappa_T,
        const scalar_t m,
        const scalar_t mu_w,
        const scalar_t p_wr,
        const scalar_t porosity,
        const scalar_t rho_w0,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
    scalar_t current_value[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t current_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, current, current_value, current_grad_ref);
    scalar_t previous_value[N_FIELDS * N_QP * VECTOR_SIZE];
    tensor_evaluate_value<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, previous, previous_value);
    scalar_t value_coeff[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t grad_coeff_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    static constexpr int Q = integer_root(N_QP, DIM);
    for (int q = 0; q < N_QP; ++q) {
        const int qx = q % Q;
        const int qy = (q / Q) % Q;
        const int qz = q / (Q * Q);
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
            const scalar_t p_w = current_value[(0 * N_QP + q) * VECTOR_SIZE + lane];
            const scalar_t p_w_grad_0_ref = current_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t p_w_grad_1_ref = current_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t p_w_grad_2_ref = current_grad_ref[((0 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
            const scalar_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
            const scalar_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
            const scalar_t p_w_old = previous_value[(0 * N_QP + q) * VECTOR_SIZE + lane];
            const scalar_t p_c = current_value[(1 * N_QP + q) * VECTOR_SIZE + lane];
            const scalar_t p_c_grad_0_ref = current_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t p_c_grad_1_ref = current_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t p_c_grad_2_ref = current_grad_ref[((1 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
            const scalar_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
            const scalar_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
            const scalar_t p_c_old = previous_value[(1 * N_QP + q) * VECTOR_SIZE + lane];
            const scalar_t residual_tmp0 = -p_wr;
            const scalar_t residual_tmp1 = exp(kappa_T*(p_w + residual_tmp0));
            const scalar_t residual_tmp2 = S_res + scalar_t(-1);
            const scalar_t residual_tmp3 = -residual_tmp2;
            const scalar_t residual_tmp4 = pow_m1(P_r);
            const scalar_t residual_tmp5 = (scalar_t(1) - m)/m;
            const scalar_t residual_tmp6 = pow(pow(residual_tmp4*(p_c - p_w), m) + scalar_t(1), residual_tmp5);
            const scalar_t residual_tmp7 = S_res - residual_tmp2*residual_tmp6;
            const scalar_t residual_tmp8 = residual_tmp1*sqrt(residual_tmp7)*rho_w0*pow_2(pow(scalar_t(1) - pow(residual_tmp7, pow_m1(C_kw1)), C_kw1) + scalar_t(-1))/mu_w;
            const scalar_t value_coeff0 = porosity*rho_w0*(residual_tmp1*(S_res + residual_tmp3*residual_tmp6) - (S_res + residual_tmp3*pow(pow(residual_tmp4*(p_c_old - p_w_old), m) + scalar_t(1), residual_tmp5))*exp(kappa_T*(p_w_old + residual_tmp0)))/dt;
            const scalar_t grad_coeff0_0 = residual_tmp8*(K_0*p_w_grad_0 + K_1*p_w_grad_1 + K_2*p_w_grad_2);
            const scalar_t grad_coeff0_1 = residual_tmp8*(K_3*p_w_grad_0 + K_4*p_w_grad_1 + K_5*p_w_grad_2);
            const scalar_t grad_coeff0_2 = residual_tmp8*(K_6*p_w_grad_0 + K_7*p_w_grad_1 + K_8*p_w_grad_2);
            value_coeff[(0 * N_QP + q) * VECTOR_SIZE + lane] = qw * det * value_coeff0;
            grad_coeff_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1 + adj2 * grad_coeff0_2);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj3 * grad_coeff0_0 + adj4 * grad_coeff0_1 + adj5 * grad_coeff0_2);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane] = qw * (adj6 * grad_coeff0_0 + adj7 * grad_coeff0_1 + adj8 * grad_coeff0_2);
            value_coeff[(1 * N_QP + q) * VECTOR_SIZE + lane] = scalar_t(0);
            grad_coeff_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = scalar_t(0);
            grad_coeff_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = scalar_t(0);
            grad_coeff_ref[((1 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane] = scalar_t(0);
        }
    }
    tensor_integrate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, DIM, N_FIELDS>(
            nelems, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void two_phase_flow_form_1_p_w_d3_tensor_product_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
}

} // namespace codegen
} // namespace sfem

#endif
