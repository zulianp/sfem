#ifndef TWO_PHASE_FLOW_D2_TENSOR_PRODUCT_LOCAL_HPP
#define TWO_PHASE_FLOW_D2_TENSOR_PRODUCT_LOCAL_HPP

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
static SFEM_INLINE void two_phase_flow_d2_tensor_product_residual_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR adjugate[4],
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t *const RSTR current[2 * NS],
        const s_t *const RSTR previous[2 * NS],
        const s_t C_ka1,
        const s_t C_ka2,
        const s_t C_kw1,
        const s_t K_0,
        const s_t K_1,
        const s_t K_2,
        const s_t K_3,
        const s_t M_c,
        const s_t P_r,
        const s_t R,
        const s_t S_res,
        const s_t T,
        const s_t Z,
        const s_t dt,
        const s_t kappa_T,
        const s_t m,
        const s_t mu_c,
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
            const s_t residual_tmp7 = pow(pow(residual_tmp4*(p_c_old - p_w_old), m) + s_t(1), residual_tmp5);
            const s_t residual_tmp8 = porosity/dt;
            const s_t residual_tmp9 = residual_tmp2*residual_tmp6;
            const s_t residual_tmp10 = S_res - residual_tmp9;
            const s_t residual_tmp11 = residual_tmp1*sqrt(residual_tmp10)*rho_w0*pow_2(pow(s_t(1) - pow(residual_tmp10, pow_m1(C_kw1)), C_kw1) + s_t(-1))/mu_w;
            const s_t residual_tmp12 = s_t(1) - S_res;
            const s_t residual_tmp13 = M_c/(R*T*Z);
            const s_t residual_tmp14 = p_c*residual_tmp13*pow(s_t(1) - residual_tmp6, C_ka1)*(pow(residual_tmp6, C_ka2) + s_t(-1))/mu_c;
            const s_t value_coeff0 = residual_tmp8*rho_w0*(residual_tmp1*(S_res + residual_tmp3*residual_tmp6) - (S_res + residual_tmp3*residual_tmp7)*exp(kappa_T*(p_w_old + residual_tmp0)));
            const s_t grad_coeff0_0 = residual_tmp11*(K_0*p_w_grad_0 + K_1*p_w_grad_1);
            const s_t grad_coeff0_1 = residual_tmp11*(K_2*p_w_grad_0 + K_3*p_w_grad_1);
            const s_t value_coeff1 = -residual_tmp13*residual_tmp8*(-p_c*(residual_tmp12 + residual_tmp9) + p_c_old*(residual_tmp12 + residual_tmp2*residual_tmp7));
            const s_t grad_coeff1_0 = residual_tmp14*(-K_0*p_c_grad_0 - K_1*p_c_grad_1);
            const s_t grad_coeff1_1 = residual_tmp14*(-K_2*p_c_grad_0 - K_3*p_c_grad_1);
            value_coeff[(0 * NQ + q) * VS + lane] = qw * det * value_coeff0;
            grad_coeff_ref[((0 * NQ + q) * ND + 0) * VS + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
            grad_coeff_ref[((0 * NQ + q) * ND + 1) * VS + lane] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
            value_coeff[(1 * NQ + q) * VS + lane] = qw * det * value_coeff1;
            grad_coeff_ref[((1 * NQ + q) * ND + 0) * VS + lane] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
            grad_coeff_ref[((1 * NQ + q) * ND + 1) * VS + lane] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
        }
    }
    tensor_integrate<s_t, NQ, NS, VS, ND, NC>(
            ne, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_d2_tensor_product_residual_block_contiguous(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR adjugate[4],
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t current[2 * NS][VS],
        const s_t previous[2 * NS][VS],
        const s_t C_ka1,
        const s_t C_ka2,
        const s_t C_kw1,
        const s_t K_0,
        const s_t K_1,
        const s_t K_2,
        const s_t K_3,
        const s_t M_c,
        const s_t P_r,
        const s_t R,
        const s_t S_res,
        const s_t T,
        const s_t Z,
        const s_t dt,
        const s_t kappa_T,
        const s_t m,
        const s_t mu_c,
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
            const s_t residual_tmp7 = pow(pow(residual_tmp4*(p_c_old - p_w_old), m) + s_t(1), residual_tmp5);
            const s_t residual_tmp8 = porosity/dt;
            const s_t residual_tmp9 = residual_tmp2*residual_tmp6;
            const s_t residual_tmp10 = S_res - residual_tmp9;
            const s_t residual_tmp11 = residual_tmp1*sqrt(residual_tmp10)*rho_w0*pow_2(pow(s_t(1) - pow(residual_tmp10, pow_m1(C_kw1)), C_kw1) + s_t(-1))/mu_w;
            const s_t residual_tmp12 = s_t(1) - S_res;
            const s_t residual_tmp13 = M_c/(R*T*Z);
            const s_t residual_tmp14 = p_c*residual_tmp13*pow(s_t(1) - residual_tmp6, C_ka1)*(pow(residual_tmp6, C_ka2) + s_t(-1))/mu_c;
            const s_t value_coeff0 = residual_tmp8*rho_w0*(residual_tmp1*(S_res + residual_tmp3*residual_tmp6) - (S_res + residual_tmp3*residual_tmp7)*exp(kappa_T*(p_w_old + residual_tmp0)));
            const s_t grad_coeff0_0 = residual_tmp11*(K_0*p_w_grad_0 + K_1*p_w_grad_1);
            const s_t grad_coeff0_1 = residual_tmp11*(K_2*p_w_grad_0 + K_3*p_w_grad_1);
            const s_t value_coeff1 = -residual_tmp13*residual_tmp8*(-p_c*(residual_tmp12 + residual_tmp9) + p_c_old*(residual_tmp12 + residual_tmp2*residual_tmp7));
            const s_t grad_coeff1_0 = residual_tmp14*(-K_0*p_c_grad_0 - K_1*p_c_grad_1);
            const s_t grad_coeff1_1 = residual_tmp14*(-K_2*p_c_grad_0 - K_3*p_c_grad_1);
            value_coeff[(0 * NQ + q) * VS + lane] = qw * det * value_coeff0;
            grad_coeff_ref[((0 * NQ + q) * ND + 0) * VS + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
            grad_coeff_ref[((0 * NQ + q) * ND + 1) * VS + lane] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
            value_coeff[(1 * NQ + q) * VS + lane] = qw * det * value_coeff1;
            grad_coeff_ref[((1 * NQ + q) * ND + 0) * VS + lane] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
            grad_coeff_ref[((1 * NQ + q) * ND + 1) * VS + lane] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
        }
    }
    tensor_integrate_contiguous<s_t, NQ, NS, VS, ND, NC>(
            ne, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_d2_tensor_product_jacobian_action_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR adjugate[4],
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t *const RSTR current[2 * NS],
        const s_t *const RSTR direction[2 * NS],
        const s_t C_ka1,
        const s_t C_ka2,
        const s_t C_kw1,
        const s_t K_0,
        const s_t K_1,
        const s_t K_2,
        const s_t K_3,
        const s_t M_c,
        const s_t P_r,
        const s_t R,
        const s_t S_res,
        const s_t T,
        const s_t Z,
        const s_t dt,
        const s_t kappa_T,
        const s_t m,
        const s_t mu_c,
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
    s_t direction_value[NC * NQ * VS];
    s_t direction_grad_ref[NC * NQ * ND * VS];
    tensor_evaluate<s_t, NQ, NS, VS, ND, NC>(
            ne, shape_1d, grad_1d, direction, direction_value, direction_grad_ref);
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
            const s_t p_w_direction = direction_value[(0 * NQ + q) * VS + lane];
            const s_t p_w_direction_grad_0_ref = direction_grad_ref[((0 * NQ + q) * ND + 0) * VS + lane];
            const s_t p_w_direction_grad_1_ref = direction_grad_ref[((0 * NQ + q) * ND + 1) * VS + lane];
            const s_t p_w_direction_grad_0 = (p_w_direction_grad_0_ref * adj0 + p_w_direction_grad_1_ref * adj2) / det;
            const s_t p_w_direction_grad_1 = (p_w_direction_grad_0_ref * adj1 + p_w_direction_grad_1_ref * adj3) / det;
            const s_t p_c = current_value[(1 * NQ + q) * VS + lane];
            const s_t p_c_grad_0_ref = current_grad_ref[((1 * NQ + q) * ND + 0) * VS + lane];
            const s_t p_c_grad_1_ref = current_grad_ref[((1 * NQ + q) * ND + 1) * VS + lane];
            const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj2) / det;
            const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj3) / det;
            const s_t p_c_direction = direction_value[(1 * NQ + q) * VS + lane];
            const s_t p_c_direction_grad_0_ref = direction_grad_ref[((1 * NQ + q) * ND + 0) * VS + lane];
            const s_t p_c_direction_grad_1_ref = direction_grad_ref[((1 * NQ + q) * ND + 1) * VS + lane];
            const s_t p_c_direction_grad_0 = (p_c_direction_grad_0_ref * adj0 + p_c_direction_grad_1_ref * adj2) / det;
            const s_t p_c_direction_grad_1 = (p_c_direction_grad_0_ref * adj1 + p_c_direction_grad_1_ref * adj3) / det;
            const s_t residual_tmp0 = pow_m1(dt);
            const s_t residual_tmp1 = residual_tmp0*rho_w0;
            const s_t residual_tmp2 = p_c_direction*residual_tmp1;
            const s_t residual_tmp3 = S_res + s_t(-1);
            const s_t residual_tmp4 = p_c - p_w;
            const s_t residual_tmp5 = pow(residual_tmp4/P_r, m);
            const s_t residual_tmp6 = residual_tmp5 + s_t(1);
            const s_t residual_tmp7 = s_t(1) - m;
            const s_t residual_tmp8 = pow(residual_tmp6, residual_tmp7/m);
            const s_t residual_tmp9 = -residual_tmp3*residual_tmp8;
            const s_t residual_tmp10 = exp(kappa_T*(p_w - p_wr));
            const s_t residual_tmp11 = residual_tmp5*residual_tmp7/(residual_tmp4*residual_tmp6);
            const s_t residual_tmp12 = residual_tmp10*residual_tmp11;
            const s_t residual_tmp13 = residual_tmp12*residual_tmp9;
            const s_t residual_tmp14 = kappa_T*residual_tmp10;
            const s_t residual_tmp15 = p_w_direction*residual_tmp1;
            const s_t residual_tmp16 = pow_m1(mu_w);
            const s_t residual_tmp17 = residual_tmp3*residual_tmp8;
            const s_t residual_tmp18 = S_res - residual_tmp17;
            const s_t residual_tmp19 = pow(residual_tmp18, pow_m1(C_kw1));
            const s_t residual_tmp20 = s_t(1) - residual_tmp19;
            const s_t residual_tmp21 = pow(residual_tmp20, C_kw1);
            const s_t residual_tmp22 = residual_tmp21 + s_t(-1);
            const s_t residual_tmp23 = pow_2(residual_tmp22);
            const s_t residual_tmp24 = sqrt(residual_tmp18);
            const s_t residual_tmp25 = residual_tmp23*residual_tmp24;
            const s_t residual_tmp26 = residual_tmp10*residual_tmp16*residual_tmp25*rho_w0;
            const s_t residual_tmp27 = p_w_direction_grad_0*residual_tmp26;
            const s_t residual_tmp28 = p_w_direction_grad_1*residual_tmp26;
            const s_t residual_tmp29 = dt*(K_0*p_w_grad_0 + K_1*p_w_grad_1);
            const s_t residual_tmp30 = residual_tmp12*residual_tmp17/residual_tmp24;
            const s_t residual_tmp31 = residual_tmp29*residual_tmp30;
            const s_t residual_tmp32 = ((s_t(1) / s_t(2)))*residual_tmp23;
            const s_t residual_tmp33 = s_t(2)*residual_tmp19*residual_tmp21*residual_tmp22/residual_tmp20;
            const s_t residual_tmp34 = residual_tmp31*residual_tmp32 - residual_tmp31*residual_tmp33;
            const s_t residual_tmp35 = residual_tmp16*residual_tmp2;
            const s_t residual_tmp36 = residual_tmp14*residual_tmp25;
            const s_t residual_tmp37 = residual_tmp15*residual_tmp16;
            const s_t residual_tmp38 = dt*(K_2*p_w_grad_0 + K_3*p_w_grad_1);
            const s_t residual_tmp39 = residual_tmp30*residual_tmp38;
            const s_t residual_tmp40 = residual_tmp32*residual_tmp39 - residual_tmp33*residual_tmp39;
            const s_t residual_tmp41 = p_c*residual_tmp11;
            const s_t residual_tmp42 = residual_tmp17*residual_tmp41;
            const s_t residual_tmp43 = pow_m1(R);
            const s_t residual_tmp44 = pow_m1(T);
            const s_t residual_tmp45 = pow_m1(Z);
            const s_t residual_tmp46 = M_c*residual_tmp43*residual_tmp44*residual_tmp45;
            const s_t residual_tmp47 = residual_tmp0*residual_tmp46;
            const s_t residual_tmp48 = pow_m1(mu_c);
            const s_t residual_tmp49 = s_t(1) - residual_tmp8;
            const s_t residual_tmp50 = pow(residual_tmp49, C_ka1);
            const s_t residual_tmp51 = pow(residual_tmp8, C_ka2);
            const s_t residual_tmp52 = residual_tmp50*(residual_tmp51 + s_t(-1));
            const s_t residual_tmp53 = p_c*residual_tmp46*residual_tmp48*residual_tmp52;
            const s_t residual_tmp54 = p_c_direction_grad_0*residual_tmp53;
            const s_t residual_tmp55 = p_c_direction_grad_1*residual_tmp53;
            const s_t residual_tmp56 = -K_0*p_c_grad_0 - K_1*p_c_grad_1;
            const s_t residual_tmp57 = C_ka2*dt*residual_tmp41*residual_tmp50*residual_tmp51;
            const s_t residual_tmp58 = residual_tmp56*residual_tmp57;
            const s_t residual_tmp59 = dt*residual_tmp52;
            const s_t residual_tmp60 = residual_tmp56*residual_tmp59;
            const s_t residual_tmp61 = C_ka1*residual_tmp41*residual_tmp8/residual_tmp49;
            const s_t residual_tmp62 = residual_tmp60*residual_tmp61;
            const s_t residual_tmp63 = -K_2*p_c_grad_0 - K_3*p_c_grad_1;
            const s_t residual_tmp64 = residual_tmp57*residual_tmp63;
            const s_t residual_tmp65 = residual_tmp59*residual_tmp63;
            const s_t residual_tmp66 = residual_tmp61*residual_tmp65;
            const s_t value_coeff0 = porosity*residual_tmp13*residual_tmp2 + porosity*residual_tmp15*(-residual_tmp13 + residual_tmp14*(S_res + residual_tmp9));
            const s_t grad_coeff0_0 = K_0*residual_tmp27 + K_1*residual_tmp28 - residual_tmp34*residual_tmp35 + residual_tmp37*(residual_tmp29*residual_tmp36 + residual_tmp34);
            const s_t grad_coeff0_1 = K_2*residual_tmp27 + K_3*residual_tmp28 - residual_tmp35*residual_tmp40 + residual_tmp37*(residual_tmp36*residual_tmp38 + residual_tmp40);
            const s_t value_coeff1 = -p_c_direction*porosity*residual_tmp47*(S_res - residual_tmp17 - residual_tmp42 + s_t(-1)) - p_w_direction*porosity*residual_tmp42*residual_tmp47;
            const s_t grad_coeff1_0 = -K_0*residual_tmp54 - K_1*residual_tmp55 + M_c*p_c_direction*residual_tmp0*residual_tmp43*residual_tmp44*residual_tmp45*residual_tmp48*(residual_tmp58 + residual_tmp60 - residual_tmp62) + M_c*p_w_direction*residual_tmp0*residual_tmp43*residual_tmp44*residual_tmp45*residual_tmp48*(-residual_tmp58 + residual_tmp62);
            const s_t grad_coeff1_1 = -K_2*residual_tmp54 - K_3*residual_tmp55 + M_c*p_c_direction*residual_tmp0*residual_tmp43*residual_tmp44*residual_tmp45*residual_tmp48*(residual_tmp64 + residual_tmp65 - residual_tmp66) + M_c*p_w_direction*residual_tmp0*residual_tmp43*residual_tmp44*residual_tmp45*residual_tmp48*(-residual_tmp64 + residual_tmp66);
            value_coeff[(0 * NQ + q) * VS + lane] = qw * det * value_coeff0;
            grad_coeff_ref[((0 * NQ + q) * ND + 0) * VS + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
            grad_coeff_ref[((0 * NQ + q) * ND + 1) * VS + lane] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
            value_coeff[(1 * NQ + q) * VS + lane] = qw * det * value_coeff1;
            grad_coeff_ref[((1 * NQ + q) * ND + 0) * VS + lane] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
            grad_coeff_ref[((1 * NQ + q) * ND + 1) * VS + lane] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
        }
    }
    tensor_integrate<s_t, NQ, NS, VS, ND, NC>(
            ne, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_d2_tensor_product_jacobian_action_block_contiguous(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR adjugate[4],
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t current[2 * NS][VS],
        const s_t direction[2 * NS][VS],
        const s_t C_ka1,
        const s_t C_ka2,
        const s_t C_kw1,
        const s_t K_0,
        const s_t K_1,
        const s_t K_2,
        const s_t K_3,
        const s_t M_c,
        const s_t P_r,
        const s_t R,
        const s_t S_res,
        const s_t T,
        const s_t Z,
        const s_t dt,
        const s_t kappa_T,
        const s_t m,
        const s_t mu_c,
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
    s_t direction_value[NC * NQ * VS];
    s_t direction_grad_ref[NC * NQ * ND * VS];
    tensor_evaluate_contiguous<s_t, NQ, NS, VS, ND, NC>(
            ne, shape_1d, grad_1d, direction, direction_value, direction_grad_ref);
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
            const s_t p_w_direction = direction_value[(0 * NQ + q) * VS + lane];
            const s_t p_w_direction_grad_0_ref = direction_grad_ref[((0 * NQ + q) * ND + 0) * VS + lane];
            const s_t p_w_direction_grad_1_ref = direction_grad_ref[((0 * NQ + q) * ND + 1) * VS + lane];
            const s_t p_w_direction_grad_0 = (p_w_direction_grad_0_ref * adj0 + p_w_direction_grad_1_ref * adj2) / det;
            const s_t p_w_direction_grad_1 = (p_w_direction_grad_0_ref * adj1 + p_w_direction_grad_1_ref * adj3) / det;
            const s_t p_c = current_value[(1 * NQ + q) * VS + lane];
            const s_t p_c_grad_0_ref = current_grad_ref[((1 * NQ + q) * ND + 0) * VS + lane];
            const s_t p_c_grad_1_ref = current_grad_ref[((1 * NQ + q) * ND + 1) * VS + lane];
            const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj2) / det;
            const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj3) / det;
            const s_t p_c_direction = direction_value[(1 * NQ + q) * VS + lane];
            const s_t p_c_direction_grad_0_ref = direction_grad_ref[((1 * NQ + q) * ND + 0) * VS + lane];
            const s_t p_c_direction_grad_1_ref = direction_grad_ref[((1 * NQ + q) * ND + 1) * VS + lane];
            const s_t p_c_direction_grad_0 = (p_c_direction_grad_0_ref * adj0 + p_c_direction_grad_1_ref * adj2) / det;
            const s_t p_c_direction_grad_1 = (p_c_direction_grad_0_ref * adj1 + p_c_direction_grad_1_ref * adj3) / det;
            const s_t residual_tmp0 = pow_m1(dt);
            const s_t residual_tmp1 = residual_tmp0*rho_w0;
            const s_t residual_tmp2 = p_c_direction*residual_tmp1;
            const s_t residual_tmp3 = S_res + s_t(-1);
            const s_t residual_tmp4 = p_c - p_w;
            const s_t residual_tmp5 = pow(residual_tmp4/P_r, m);
            const s_t residual_tmp6 = residual_tmp5 + s_t(1);
            const s_t residual_tmp7 = s_t(1) - m;
            const s_t residual_tmp8 = pow(residual_tmp6, residual_tmp7/m);
            const s_t residual_tmp9 = -residual_tmp3*residual_tmp8;
            const s_t residual_tmp10 = exp(kappa_T*(p_w - p_wr));
            const s_t residual_tmp11 = residual_tmp5*residual_tmp7/(residual_tmp4*residual_tmp6);
            const s_t residual_tmp12 = residual_tmp10*residual_tmp11;
            const s_t residual_tmp13 = residual_tmp12*residual_tmp9;
            const s_t residual_tmp14 = kappa_T*residual_tmp10;
            const s_t residual_tmp15 = p_w_direction*residual_tmp1;
            const s_t residual_tmp16 = pow_m1(mu_w);
            const s_t residual_tmp17 = residual_tmp3*residual_tmp8;
            const s_t residual_tmp18 = S_res - residual_tmp17;
            const s_t residual_tmp19 = pow(residual_tmp18, pow_m1(C_kw1));
            const s_t residual_tmp20 = s_t(1) - residual_tmp19;
            const s_t residual_tmp21 = pow(residual_tmp20, C_kw1);
            const s_t residual_tmp22 = residual_tmp21 + s_t(-1);
            const s_t residual_tmp23 = pow_2(residual_tmp22);
            const s_t residual_tmp24 = sqrt(residual_tmp18);
            const s_t residual_tmp25 = residual_tmp23*residual_tmp24;
            const s_t residual_tmp26 = residual_tmp10*residual_tmp16*residual_tmp25*rho_w0;
            const s_t residual_tmp27 = p_w_direction_grad_0*residual_tmp26;
            const s_t residual_tmp28 = p_w_direction_grad_1*residual_tmp26;
            const s_t residual_tmp29 = dt*(K_0*p_w_grad_0 + K_1*p_w_grad_1);
            const s_t residual_tmp30 = residual_tmp12*residual_tmp17/residual_tmp24;
            const s_t residual_tmp31 = residual_tmp29*residual_tmp30;
            const s_t residual_tmp32 = ((s_t(1) / s_t(2)))*residual_tmp23;
            const s_t residual_tmp33 = s_t(2)*residual_tmp19*residual_tmp21*residual_tmp22/residual_tmp20;
            const s_t residual_tmp34 = residual_tmp31*residual_tmp32 - residual_tmp31*residual_tmp33;
            const s_t residual_tmp35 = residual_tmp16*residual_tmp2;
            const s_t residual_tmp36 = residual_tmp14*residual_tmp25;
            const s_t residual_tmp37 = residual_tmp15*residual_tmp16;
            const s_t residual_tmp38 = dt*(K_2*p_w_grad_0 + K_3*p_w_grad_1);
            const s_t residual_tmp39 = residual_tmp30*residual_tmp38;
            const s_t residual_tmp40 = residual_tmp32*residual_tmp39 - residual_tmp33*residual_tmp39;
            const s_t residual_tmp41 = p_c*residual_tmp11;
            const s_t residual_tmp42 = residual_tmp17*residual_tmp41;
            const s_t residual_tmp43 = pow_m1(R);
            const s_t residual_tmp44 = pow_m1(T);
            const s_t residual_tmp45 = pow_m1(Z);
            const s_t residual_tmp46 = M_c*residual_tmp43*residual_tmp44*residual_tmp45;
            const s_t residual_tmp47 = residual_tmp0*residual_tmp46;
            const s_t residual_tmp48 = pow_m1(mu_c);
            const s_t residual_tmp49 = s_t(1) - residual_tmp8;
            const s_t residual_tmp50 = pow(residual_tmp49, C_ka1);
            const s_t residual_tmp51 = pow(residual_tmp8, C_ka2);
            const s_t residual_tmp52 = residual_tmp50*(residual_tmp51 + s_t(-1));
            const s_t residual_tmp53 = p_c*residual_tmp46*residual_tmp48*residual_tmp52;
            const s_t residual_tmp54 = p_c_direction_grad_0*residual_tmp53;
            const s_t residual_tmp55 = p_c_direction_grad_1*residual_tmp53;
            const s_t residual_tmp56 = -K_0*p_c_grad_0 - K_1*p_c_grad_1;
            const s_t residual_tmp57 = C_ka2*dt*residual_tmp41*residual_tmp50*residual_tmp51;
            const s_t residual_tmp58 = residual_tmp56*residual_tmp57;
            const s_t residual_tmp59 = dt*residual_tmp52;
            const s_t residual_tmp60 = residual_tmp56*residual_tmp59;
            const s_t residual_tmp61 = C_ka1*residual_tmp41*residual_tmp8/residual_tmp49;
            const s_t residual_tmp62 = residual_tmp60*residual_tmp61;
            const s_t residual_tmp63 = -K_2*p_c_grad_0 - K_3*p_c_grad_1;
            const s_t residual_tmp64 = residual_tmp57*residual_tmp63;
            const s_t residual_tmp65 = residual_tmp59*residual_tmp63;
            const s_t residual_tmp66 = residual_tmp61*residual_tmp65;
            const s_t value_coeff0 = porosity*residual_tmp13*residual_tmp2 + porosity*residual_tmp15*(-residual_tmp13 + residual_tmp14*(S_res + residual_tmp9));
            const s_t grad_coeff0_0 = K_0*residual_tmp27 + K_1*residual_tmp28 - residual_tmp34*residual_tmp35 + residual_tmp37*(residual_tmp29*residual_tmp36 + residual_tmp34);
            const s_t grad_coeff0_1 = K_2*residual_tmp27 + K_3*residual_tmp28 - residual_tmp35*residual_tmp40 + residual_tmp37*(residual_tmp36*residual_tmp38 + residual_tmp40);
            const s_t value_coeff1 = -p_c_direction*porosity*residual_tmp47*(S_res - residual_tmp17 - residual_tmp42 + s_t(-1)) - p_w_direction*porosity*residual_tmp42*residual_tmp47;
            const s_t grad_coeff1_0 = -K_0*residual_tmp54 - K_1*residual_tmp55 + M_c*p_c_direction*residual_tmp0*residual_tmp43*residual_tmp44*residual_tmp45*residual_tmp48*(residual_tmp58 + residual_tmp60 - residual_tmp62) + M_c*p_w_direction*residual_tmp0*residual_tmp43*residual_tmp44*residual_tmp45*residual_tmp48*(-residual_tmp58 + residual_tmp62);
            const s_t grad_coeff1_1 = -K_2*residual_tmp54 - K_3*residual_tmp55 + M_c*p_c_direction*residual_tmp0*residual_tmp43*residual_tmp44*residual_tmp45*residual_tmp48*(residual_tmp64 + residual_tmp65 - residual_tmp66) + M_c*p_w_direction*residual_tmp0*residual_tmp43*residual_tmp44*residual_tmp45*residual_tmp48*(-residual_tmp64 + residual_tmp66);
            value_coeff[(0 * NQ + q) * VS + lane] = qw * det * value_coeff0;
            grad_coeff_ref[((0 * NQ + q) * ND + 0) * VS + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1);
            grad_coeff_ref[((0 * NQ + q) * ND + 1) * VS + lane] = qw * (adj2 * grad_coeff0_0 + adj3 * grad_coeff0_1);
            value_coeff[(1 * NQ + q) * VS + lane] = qw * det * value_coeff1;
            grad_coeff_ref[((1 * NQ + q) * ND + 0) * VS + lane] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1);
            grad_coeff_ref[((1 * NQ + q) * ND + 1) * VS + lane] = qw * (adj2 * grad_coeff1_0 + adj3 * grad_coeff1_1);
        }
    }
    tensor_integrate_contiguous<s_t, NQ, NS, VS, ND, NC>(
            ne, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

} // namespace codegen
} // namespace sfem

#endif
