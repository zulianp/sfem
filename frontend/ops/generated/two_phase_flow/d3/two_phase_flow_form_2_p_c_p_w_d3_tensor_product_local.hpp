#ifndef TWO_PHASE_FLOW_FORM_2_P_C_P_W_D3_TENSOR_PRODUCT_LOCAL_HPP
#define TWO_PHASE_FLOW_FORM_2_P_C_P_W_D3_TENSOR_PRODUCT_LOCAL_HPP

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
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_tensor_product_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT determinant,
        const s_t *const SFEM_RESTRICT shape_1d,
        const s_t *const SFEM_RESTRICT q_weight_1d,
        s_t *const SFEM_RESTRICT output[2 * NS]
) {
    static constexpr int ND = 3;
    static constexpr int NC = 2;
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_tensor_product_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT determinant,
        const s_t *const SFEM_RESTRICT shape_1d,
        const s_t *const SFEM_RESTRICT q_weight_1d,
        s_t output[2 * NS][VS]
) {
    static constexpr int ND = 3;
    static constexpr int NC = 2;
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_tensor_product_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT determinant,
        const s_t *const SFEM_RESTRICT adjugate[9],
        const s_t *const SFEM_RESTRICT shape_1d,
        const s_t *const SFEM_RESTRICT grad_1d,
        const s_t *const SFEM_RESTRICT q_weight_1d,
        const s_t *const SFEM_RESTRICT current[2 * NS],
        const s_t *const SFEM_RESTRICT direction[2 * NS],
        const s_t C_ka1,
        const s_t C_ka2,
        const s_t K_0,
        const s_t K_1,
        const s_t K_2,
        const s_t K_3,
        const s_t K_4,
        const s_t K_5,
        const s_t K_6,
        const s_t K_7,
        const s_t K_8,
        const s_t M_c,
        const s_t P_r,
        const s_t R,
        const s_t S_res,
        const s_t T,
        const s_t Z,
        const s_t dt,
        const s_t m,
        const s_t mu_c,
        const s_t porosity,
        s_t *const SFEM_RESTRICT output[2 * NS]
) {
    static constexpr int ND = 3;
    static constexpr int NC = 2;
    s_t current_value[NC * NQ * VS];
    s_t current_grad_ref[NC * NQ * ND * VS];
    tensor_evaluate<s_t, NQ, NS, VS, ND, NC>(
            nelems, shape_1d, grad_1d, current, current_value, current_grad_ref);
    s_t direction_value[NC * NQ * VS];
    tensor_evaluate_value<s_t, NQ, NS, VS, ND, NC>(
            nelems, shape_1d, direction, direction_value);
    s_t value_coeff[NC * NQ * VS];
    s_t grad_coeff_ref[NC * NQ * ND * VS];
    static constexpr int NQ1 = integer_root(NQ, ND);
    for (int q = 0; q < NQ; ++q) {
        const int qx = q % NQ1;
        const int qy = (q / NQ1) % NQ1;
        const int qz = q / (NQ1 * NQ1);
        const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t det = determinant[goff];
            const s_t adj0 = adjugate[0][goff];
            const s_t adj1 = adjugate[1][goff];
            const s_t adj2 = adjugate[2][goff];
            const s_t adj3 = adjugate[3][goff];
            const s_t adj4 = adjugate[4][goff];
            const s_t adj5 = adjugate[5][goff];
            const s_t adj6 = adjugate[6][goff];
            const s_t adj7 = adjugate[7][goff];
            const s_t adj8 = adjugate[8][goff];
            const s_t p_w = current_value[(0 * NQ + q) * VS + lane];
            const s_t p_w_grad_0_ref = current_grad_ref[((0 * NQ + q) * ND + 0) * VS + lane];
            const s_t p_w_grad_1_ref = current_grad_ref[((0 * NQ + q) * ND + 1) * VS + lane];
            const s_t p_w_grad_2_ref = current_grad_ref[((0 * NQ + q) * ND + 2) * VS + lane];
            const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
            const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
            const s_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
            const s_t p_w_direction = direction_value[(0 * NQ + q) * VS + lane];
            const s_t p_c = current_value[(1 * NQ + q) * VS + lane];
            const s_t p_c_grad_0_ref = current_grad_ref[((1 * NQ + q) * ND + 0) * VS + lane];
            const s_t p_c_grad_1_ref = current_grad_ref[((1 * NQ + q) * ND + 1) * VS + lane];
            const s_t p_c_grad_2_ref = current_grad_ref[((1 * NQ + q) * ND + 2) * VS + lane];
            const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
            const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
            const s_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
            const s_t p_c_direction = direction_value[(1 * NQ + q) * VS + lane];
            const s_t residual_tmp0 = p_c - p_w;
            const s_t residual_tmp1 = pow(residual_tmp0/P_r, m);
            const s_t residual_tmp2 = residual_tmp1 + s_t(1);
            const s_t residual_tmp3 = s_t(1) - m;
            const s_t residual_tmp4 = pow(residual_tmp2, residual_tmp3/m);
            const s_t residual_tmp5 = M_c*p_w_direction/(R*T*Z*dt);
            const s_t residual_tmp6 = p_c*residual_tmp1*residual_tmp3/(residual_tmp0*residual_tmp2);
            const s_t residual_tmp7 = s_t(1) - residual_tmp4;
            const s_t residual_tmp8 = dt*residual_tmp6*pow(residual_tmp7, C_ka1);
            const s_t residual_tmp9 = residual_tmp8*(-K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2);
            const s_t residual_tmp10 = pow(residual_tmp4, C_ka2);
            const s_t residual_tmp11 = C_ka2*residual_tmp10;
            const s_t residual_tmp12 = C_ka1*residual_tmp4*(residual_tmp10 + s_t(-1))/residual_tmp7;
            const s_t residual_tmp13 = residual_tmp5/mu_c;
            const s_t residual_tmp14 = -K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2;
            const s_t residual_tmp15 = residual_tmp11*residual_tmp8;
            const s_t residual_tmp16 = residual_tmp12*residual_tmp8;
            const s_t residual_tmp17 = -K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2;
            const s_t value_coeff1 = -porosity*residual_tmp4*residual_tmp5*residual_tmp6*(S_res + s_t(-1));
            const s_t grad_coeff1_0 = residual_tmp13*(-residual_tmp11*residual_tmp9 + residual_tmp12*residual_tmp9);
            const s_t grad_coeff1_1 = residual_tmp13*(-residual_tmp14*residual_tmp15 + residual_tmp14*residual_tmp16);
            const s_t grad_coeff1_2 = residual_tmp13*(-residual_tmp15*residual_tmp17 + residual_tmp16*residual_tmp17);
            value_coeff[(0 * NQ + q) * VS + lane] = s_t(0);
            grad_coeff_ref[((0 * NQ + q) * ND + 0) * VS + lane] = s_t(0);
            grad_coeff_ref[((0 * NQ + q) * ND + 1) * VS + lane] = s_t(0);
            grad_coeff_ref[((0 * NQ + q) * ND + 2) * VS + lane] = s_t(0);
            value_coeff[(1 * NQ + q) * VS + lane] = qw * det * value_coeff1;
            grad_coeff_ref[((1 * NQ + q) * ND + 0) * VS + lane] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1 + adj2 * grad_coeff1_2);
            grad_coeff_ref[((1 * NQ + q) * ND + 1) * VS + lane] = qw * (adj3 * grad_coeff1_0 + adj4 * grad_coeff1_1 + adj5 * grad_coeff1_2);
            grad_coeff_ref[((1 * NQ + q) * ND + 2) * VS + lane] = qw * (adj6 * grad_coeff1_0 + adj7 * grad_coeff1_1 + adj8 * grad_coeff1_2);
        }
    }
    tensor_integrate<s_t, NQ, NS, VS, ND, NC>(
            nelems, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_tensor_product_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT determinant,
        const s_t *const SFEM_RESTRICT adjugate[9],
        const s_t *const SFEM_RESTRICT shape_1d,
        const s_t *const SFEM_RESTRICT grad_1d,
        const s_t *const SFEM_RESTRICT q_weight_1d,
        const s_t current[2 * NS][VS],
        const s_t direction[2 * NS][VS],
        const s_t C_ka1,
        const s_t C_ka2,
        const s_t K_0,
        const s_t K_1,
        const s_t K_2,
        const s_t K_3,
        const s_t K_4,
        const s_t K_5,
        const s_t K_6,
        const s_t K_7,
        const s_t K_8,
        const s_t M_c,
        const s_t P_r,
        const s_t R,
        const s_t S_res,
        const s_t T,
        const s_t Z,
        const s_t dt,
        const s_t m,
        const s_t mu_c,
        const s_t porosity,
        s_t output[2 * NS][VS]
) {
    static constexpr int ND = 3;
    static constexpr int NC = 2;
    s_t current_value[NC * NQ * VS];
    s_t current_grad_ref[NC * NQ * ND * VS];
    tensor_evaluate_contiguous<s_t, NQ, NS, VS, ND, NC>(
            nelems, shape_1d, grad_1d, current, current_value, current_grad_ref);
    s_t direction_value[NC * NQ * VS];
    tensor_evaluate_value_contiguous<s_t, NQ, NS, VS, ND, NC>(
            nelems, shape_1d, direction, direction_value);
    s_t value_coeff[NC * NQ * VS];
    s_t grad_coeff_ref[NC * NQ * ND * VS];
    static constexpr int NQ1 = integer_root(NQ, ND);
    for (int q = 0; q < NQ; ++q) {
        const int qx = q % NQ1;
        const int qy = (q / NQ1) % NQ1;
        const int qz = q / (NQ1 * NQ1);
        const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t det = determinant[goff];
            const s_t adj0 = adjugate[0][goff];
            const s_t adj1 = adjugate[1][goff];
            const s_t adj2 = adjugate[2][goff];
            const s_t adj3 = adjugate[3][goff];
            const s_t adj4 = adjugate[4][goff];
            const s_t adj5 = adjugate[5][goff];
            const s_t adj6 = adjugate[6][goff];
            const s_t adj7 = adjugate[7][goff];
            const s_t adj8 = adjugate[8][goff];
            const s_t p_w = current_value[(0 * NQ + q) * VS + lane];
            const s_t p_w_grad_0_ref = current_grad_ref[((0 * NQ + q) * ND + 0) * VS + lane];
            const s_t p_w_grad_1_ref = current_grad_ref[((0 * NQ + q) * ND + 1) * VS + lane];
            const s_t p_w_grad_2_ref = current_grad_ref[((0 * NQ + q) * ND + 2) * VS + lane];
            const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
            const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
            const s_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
            const s_t p_w_direction = direction_value[(0 * NQ + q) * VS + lane];
            const s_t p_c = current_value[(1 * NQ + q) * VS + lane];
            const s_t p_c_grad_0_ref = current_grad_ref[((1 * NQ + q) * ND + 0) * VS + lane];
            const s_t p_c_grad_1_ref = current_grad_ref[((1 * NQ + q) * ND + 1) * VS + lane];
            const s_t p_c_grad_2_ref = current_grad_ref[((1 * NQ + q) * ND + 2) * VS + lane];
            const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
            const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
            const s_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
            const s_t p_c_direction = direction_value[(1 * NQ + q) * VS + lane];
            const s_t residual_tmp0 = p_c - p_w;
            const s_t residual_tmp1 = pow(residual_tmp0/P_r, m);
            const s_t residual_tmp2 = residual_tmp1 + s_t(1);
            const s_t residual_tmp3 = s_t(1) - m;
            const s_t residual_tmp4 = pow(residual_tmp2, residual_tmp3/m);
            const s_t residual_tmp5 = M_c*p_w_direction/(R*T*Z*dt);
            const s_t residual_tmp6 = p_c*residual_tmp1*residual_tmp3/(residual_tmp0*residual_tmp2);
            const s_t residual_tmp7 = s_t(1) - residual_tmp4;
            const s_t residual_tmp8 = dt*residual_tmp6*pow(residual_tmp7, C_ka1);
            const s_t residual_tmp9 = residual_tmp8*(-K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2);
            const s_t residual_tmp10 = pow(residual_tmp4, C_ka2);
            const s_t residual_tmp11 = C_ka2*residual_tmp10;
            const s_t residual_tmp12 = C_ka1*residual_tmp4*(residual_tmp10 + s_t(-1))/residual_tmp7;
            const s_t residual_tmp13 = residual_tmp5/mu_c;
            const s_t residual_tmp14 = -K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2;
            const s_t residual_tmp15 = residual_tmp11*residual_tmp8;
            const s_t residual_tmp16 = residual_tmp12*residual_tmp8;
            const s_t residual_tmp17 = -K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2;
            const s_t value_coeff1 = -porosity*residual_tmp4*residual_tmp5*residual_tmp6*(S_res + s_t(-1));
            const s_t grad_coeff1_0 = residual_tmp13*(-residual_tmp11*residual_tmp9 + residual_tmp12*residual_tmp9);
            const s_t grad_coeff1_1 = residual_tmp13*(-residual_tmp14*residual_tmp15 + residual_tmp14*residual_tmp16);
            const s_t grad_coeff1_2 = residual_tmp13*(-residual_tmp15*residual_tmp17 + residual_tmp16*residual_tmp17);
            value_coeff[(0 * NQ + q) * VS + lane] = s_t(0);
            grad_coeff_ref[((0 * NQ + q) * ND + 0) * VS + lane] = s_t(0);
            grad_coeff_ref[((0 * NQ + q) * ND + 1) * VS + lane] = s_t(0);
            grad_coeff_ref[((0 * NQ + q) * ND + 2) * VS + lane] = s_t(0);
            value_coeff[(1 * NQ + q) * VS + lane] = qw * det * value_coeff1;
            grad_coeff_ref[((1 * NQ + q) * ND + 0) * VS + lane] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1 + adj2 * grad_coeff1_2);
            grad_coeff_ref[((1 * NQ + q) * ND + 1) * VS + lane] = qw * (adj3 * grad_coeff1_0 + adj4 * grad_coeff1_1 + adj5 * grad_coeff1_2);
            grad_coeff_ref[((1 * NQ + q) * ND + 2) * VS + lane] = qw * (adj6 * grad_coeff1_0 + adj7 * grad_coeff1_1 + adj8 * grad_coeff1_2);
        }
    }
    tensor_integrate_contiguous<s_t, NQ, NS, VS, ND, NC>(
            nelems, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

} // namespace codegen
} // namespace sfem

#endif
