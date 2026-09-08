#ifndef TWO_PHASE_FLOW_FORM_1_P_W_D2_TENSOR_PRODUCT_LOCAL_HPP
#define TWO_PHASE_FLOW_FORM_1_P_W_D2_TENSOR_PRODUCT_LOCAL_HPP

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
#ifndef RSTR
#define RSTR SFEM_RESTRICT
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
static SFEM_INLINE void two_phase_flow_form_1_p_w_d2_tensor_product_residual_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR adjugate[4],
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t *const RSTR current[2 * NS],
        const s_t *const RSTR previous[2 * NS],
        const s_t C_kw1,
        const s_t K_0,
        const s_t K_1,
        const s_t K_2,
        const s_t K_3,
        const s_t P_r,
        const s_t S_res,
        const s_t dt,
        const s_t kappa_T,
        const s_t m,
        const s_t mu_w,
        const s_t p_wr,
        const s_t porosity,
        const s_t rho_w0,
        s_t *const RSTR output[2 * NS]
) {
    static constexpr int ND = 2;
    static constexpr int NC = 2;
    s_t current_value[NC * NQ * VS];
    s_t current_grad_ref[NC * NQ * ND * VS];
    tensor_evaluate<s_t, NQ, NS, VS, ND, NC>(
            ne, shape_1d, grad_1d, current, current_value, current_grad_ref);
    s_t previous_value[NC * NQ * VS];
    tensor_evaluate_value<s_t, NQ, NS, VS, ND, NC>(
            ne, shape_1d, previous, previous_value);
    s_t value_coeff[NC * NQ * VS];
    s_t grad_coeff_ref[NC * NQ * ND * VS];
    static constexpr int NQ1 = integer_root(NQ, ND);
    for (int q = 0; q < NQ; ++q) {
        const int qx = q % NQ1;
        const int qy = q / NQ1;
        const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t det = determinant[goff];
            const s_t adj0 = adjugate[0][goff];
            const s_t adj1 = adjugate[1][goff];
            const s_t adj2 = adjugate[2][goff];
            const s_t adj3 = adjugate[3][goff];
            const s_t p_w = current_value[(0 * NQ + q) * VS + lane];
            const s_t p_w_grad_0_ref = current_grad_ref[((0 * NQ + q) * ND + 0) * VS + lane];
            const s_t p_w_grad_1_ref = current_grad_ref[((0 * NQ + q) * ND + 1) * VS + lane];
            const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj2) / det;
            const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj3) / det;
            const s_t p_w_old = previous_value[(0 * NQ + q) * VS + lane];
            const s_t p_c = current_value[(1 * NQ + q) * VS + lane];
            const s_t p_c_grad_0_ref = current_grad_ref[((1 * NQ + q) * ND + 0) * VS + lane];
            const s_t p_c_grad_1_ref = current_grad_ref[((1 * NQ + q) * ND + 1) * VS + lane];
            const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj2) / det;
            const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj3) / det;
            const s_t p_c_old = previous_value[(1 * NQ + q) * VS + lane];
            const s_t residual_tmp0 = -p_wr;
            const s_t residual_tmp1 = exp(kappa_T*(p_w + residual_tmp0));
            const s_t residual_tmp2 = S_res + s_t(-1);
            const s_t residual_tmp3 = -residual_tmp2;
            const s_t residual_tmp4 = pow_m1(P_r);
            const s_t residual_tmp5 = (s_t(1) - m)/m;
            const s_t residual_tmp6 = pow(pow(residual_tmp4*(p_c - p_w), m) + s_t(1), residual_tmp5);
            const s_t residual_tmp7 = S_res - residual_tmp2*residual_tmp6;
            const s_t residual_tmp8 = residual_tmp1*sqrt(residual_tmp7)*rho_w0*pow_2(pow(s_t(1) - pow(residual_tmp7, pow_m1(C_kw1)), C_kw1) + s_t(-1))/mu_w;
            const s_t value_coeff0 = porosity*rho_w0*(residual_tmp1*(S_res + residual_tmp3*residual_tmp6) - (S_res + residual_tmp3*pow(pow(residual_tmp4*(p_c_old - p_w_old), m) + s_t(1), residual_tmp5))*exp(kappa_T*(p_w_old + residual_tmp0)))/dt;
            const s_t grad_coeff0_0 = residual_tmp8*(K_0*p_w_grad_0 + K_1*p_w_grad_1);
            const s_t grad_coeff0_1 = residual_tmp8*(K_2*p_w_grad_0 + K_3*p_w_grad_1);
            value_coeff[(0 * NQ + q) * VS + lane] = qw * det * value_coeff0;
            grad_coeff_ref[((0 * NQ + q) * ND + 0) * VS + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
            grad_coeff_ref[((0 * NQ + q) * ND + 1) * VS + lane] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
            value_coeff[(1 * NQ + q) * VS + lane] = s_t(0);
            grad_coeff_ref[((1 * NQ + q) * ND + 0) * VS + lane] = s_t(0);
            grad_coeff_ref[((1 * NQ + q) * ND + 1) * VS + lane] = s_t(0);
        }
    }
    tensor_integrate<s_t, NQ, NS, VS, ND, NC>(
            ne, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_form_1_p_w_d2_tensor_product_residual_block_contiguous(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR adjugate[4],
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t current[2 * NS][VS],
        const s_t previous[2 * NS][VS],
        const s_t C_kw1,
        const s_t K_0,
        const s_t K_1,
        const s_t K_2,
        const s_t K_3,
        const s_t P_r,
        const s_t S_res,
        const s_t dt,
        const s_t kappa_T,
        const s_t m,
        const s_t mu_w,
        const s_t p_wr,
        const s_t porosity,
        const s_t rho_w0,
        s_t output[2 * NS][VS]
) {
    static constexpr int ND = 2;
    static constexpr int NC = 2;
    s_t current_value[NC * NQ * VS];
    s_t current_grad_ref[NC * NQ * ND * VS];
    tensor_evaluate_contiguous<s_t, NQ, NS, VS, ND, NC>(
            ne, shape_1d, grad_1d, current, current_value, current_grad_ref);
    s_t previous_value[NC * NQ * VS];
    tensor_evaluate_value_contiguous<s_t, NQ, NS, VS, ND, NC>(
            ne, shape_1d, previous, previous_value);
    s_t value_coeff[NC * NQ * VS];
    s_t grad_coeff_ref[NC * NQ * ND * VS];
    static constexpr int NQ1 = integer_root(NQ, ND);
    for (int q = 0; q < NQ; ++q) {
        const int qx = q % NQ1;
        const int qy = q / NQ1;
        const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t det = determinant[goff];
            const s_t adj0 = adjugate[0][goff];
            const s_t adj1 = adjugate[1][goff];
            const s_t adj2 = adjugate[2][goff];
            const s_t adj3 = adjugate[3][goff];
            const s_t p_w = current_value[(0 * NQ + q) * VS + lane];
            const s_t p_w_grad_0_ref = current_grad_ref[((0 * NQ + q) * ND + 0) * VS + lane];
            const s_t p_w_grad_1_ref = current_grad_ref[((0 * NQ + q) * ND + 1) * VS + lane];
            const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj2) / det;
            const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj3) / det;
            const s_t p_w_old = previous_value[(0 * NQ + q) * VS + lane];
            const s_t p_c = current_value[(1 * NQ + q) * VS + lane];
            const s_t p_c_grad_0_ref = current_grad_ref[((1 * NQ + q) * ND + 0) * VS + lane];
            const s_t p_c_grad_1_ref = current_grad_ref[((1 * NQ + q) * ND + 1) * VS + lane];
            const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj2) / det;
            const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj3) / det;
            const s_t p_c_old = previous_value[(1 * NQ + q) * VS + lane];
            const s_t residual_tmp0 = -p_wr;
            const s_t residual_tmp1 = exp(kappa_T*(p_w + residual_tmp0));
            const s_t residual_tmp2 = S_res + s_t(-1);
            const s_t residual_tmp3 = -residual_tmp2;
            const s_t residual_tmp4 = pow_m1(P_r);
            const s_t residual_tmp5 = (s_t(1) - m)/m;
            const s_t residual_tmp6 = pow(pow(residual_tmp4*(p_c - p_w), m) + s_t(1), residual_tmp5);
            const s_t residual_tmp7 = S_res - residual_tmp2*residual_tmp6;
            const s_t residual_tmp8 = residual_tmp1*sqrt(residual_tmp7)*rho_w0*pow_2(pow(s_t(1) - pow(residual_tmp7, pow_m1(C_kw1)), C_kw1) + s_t(-1))/mu_w;
            const s_t value_coeff0 = porosity*rho_w0*(residual_tmp1*(S_res + residual_tmp3*residual_tmp6) - (S_res + residual_tmp3*pow(pow(residual_tmp4*(p_c_old - p_w_old), m) + s_t(1), residual_tmp5))*exp(kappa_T*(p_w_old + residual_tmp0)))/dt;
            const s_t grad_coeff0_0 = residual_tmp8*(K_0*p_w_grad_0 + K_1*p_w_grad_1);
            const s_t grad_coeff0_1 = residual_tmp8*(K_2*p_w_grad_0 + K_3*p_w_grad_1);
            value_coeff[(0 * NQ + q) * VS + lane] = qw * det * value_coeff0;
            grad_coeff_ref[((0 * NQ + q) * ND + 0) * VS + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
            grad_coeff_ref[((0 * NQ + q) * ND + 1) * VS + lane] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
            value_coeff[(1 * NQ + q) * VS + lane] = s_t(0);
            grad_coeff_ref[((1 * NQ + q) * ND + 0) * VS + lane] = s_t(0);
            grad_coeff_ref[((1 * NQ + q) * ND + 1) * VS + lane] = s_t(0);
        }
    }
    tensor_integrate_contiguous<s_t, NQ, NS, VS, ND, NC>(
            ne, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_form_1_p_w_d2_tensor_product_jacobian_action_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR q_weight_1d,
        s_t *const RSTR output[2 * NS]
) {
    static constexpr int ND = 2;
    static constexpr int NC = 2;
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_form_1_p_w_d2_tensor_product_jacobian_action_block_contiguous(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR q_weight_1d,
        s_t output[2 * NS][VS]
) {
    static constexpr int ND = 2;
    static constexpr int NC = 2;
}

} // namespace codegen
} // namespace sfem

#endif
