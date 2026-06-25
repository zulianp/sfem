#ifndef GENERATED_TWO_PHASE_FLOW_D3_SIMPLEX_LOCAL_HPP
#define GENERATED_TWO_PHASE_FLOW_D3_SIMPLEX_LOCAL_HPP

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
static SFEM_INLINE void generated_two_phase_flow_d3_simplex_residual_block(
        const ptrdiff_t nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT current[2 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT previous[2 * N_SHAPE],
        const scalar_t porosity,
        const scalar_t S_res,
        const scalar_t P_r,
        const scalar_t m,
        const scalar_t rho_w0,
        const scalar_t kappa_T,
        const scalar_t p_wr,
        const scalar_t M_c,
        const scalar_t Z,
        const scalar_t R,
        const scalar_t T,
        const scalar_t mu_w,
        const scalar_t mu_c,
        const scalar_t C_kw1,
        const scalar_t C_ka1,
        const scalar_t C_ka2,
        const scalar_t dt,
        const scalar_t K_0,
        const scalar_t K_1,
        const scalar_t K_2,
        const scalar_t K_3,
        const scalar_t K_4,
        const scalar_t K_5,
        const scalar_t K_6,
        const scalar_t K_7,
        const scalar_t K_8,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
    for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
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
            scalar_t p_w = 0;
            scalar_t p_w_grad_0_ref = 0;
            scalar_t p_w_grad_1_ref = 0;
            scalar_t p_w_grad_2_ref = 0;
            for (int trial = 0; trial < N_SHAPE; ++trial) {
                const scalar_t coeff = current[trial * N_FIELDS + 0][lane];
                p_w += coeff * shape[q * N_SHAPE + trial];
                p_w_grad_0_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 0];
                p_w_grad_1_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 1];
                p_w_grad_2_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 2];
            }
            const scalar_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
            const scalar_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
            const scalar_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
            scalar_t p_w_old = 0;
            scalar_t p_w_old_grad_0_ref = 0;
            scalar_t p_w_old_grad_1_ref = 0;
            scalar_t p_w_old_grad_2_ref = 0;
            for (int trial = 0; trial < N_SHAPE; ++trial) {
                const scalar_t coeff = previous[trial * N_FIELDS + 0][lane];
                p_w_old += coeff * shape[q * N_SHAPE + trial];
                p_w_old_grad_0_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 0];
                p_w_old_grad_1_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 1];
                p_w_old_grad_2_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 2];
            }
            const scalar_t p_w_old_grad_0 = (p_w_old_grad_0_ref * adj0 + p_w_old_grad_1_ref * adj3 + p_w_old_grad_2_ref * adj6) / det;
            const scalar_t p_w_old_grad_1 = (p_w_old_grad_0_ref * adj1 + p_w_old_grad_1_ref * adj4 + p_w_old_grad_2_ref * adj7) / det;
            const scalar_t p_w_old_grad_2 = (p_w_old_grad_0_ref * adj2 + p_w_old_grad_1_ref * adj5 + p_w_old_grad_2_ref * adj8) / det;
            scalar_t p_c = 0;
            scalar_t p_c_grad_0_ref = 0;
            scalar_t p_c_grad_1_ref = 0;
            scalar_t p_c_grad_2_ref = 0;
            for (int trial = 0; trial < N_SHAPE; ++trial) {
                const scalar_t coeff = current[trial * N_FIELDS + 1][lane];
                p_c += coeff * shape[q * N_SHAPE + trial];
                p_c_grad_0_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 0];
                p_c_grad_1_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 1];
                p_c_grad_2_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 2];
            }
            const scalar_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
            const scalar_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
            const scalar_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
            scalar_t p_c_old = 0;
            scalar_t p_c_old_grad_0_ref = 0;
            scalar_t p_c_old_grad_1_ref = 0;
            scalar_t p_c_old_grad_2_ref = 0;
            for (int trial = 0; trial < N_SHAPE; ++trial) {
                const scalar_t coeff = previous[trial * N_FIELDS + 1][lane];
                p_c_old += coeff * shape[q * N_SHAPE + trial];
                p_c_old_grad_0_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 0];
                p_c_old_grad_1_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 1];
                p_c_old_grad_2_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 2];
            }
            const scalar_t p_c_old_grad_0 = (p_c_old_grad_0_ref * adj0 + p_c_old_grad_1_ref * adj3 + p_c_old_grad_2_ref * adj6) / det;
            const scalar_t p_c_old_grad_1 = (p_c_old_grad_0_ref * adj1 + p_c_old_grad_1_ref * adj4 + p_c_old_grad_2_ref * adj7) / det;
            const scalar_t p_c_old_grad_2 = (p_c_old_grad_0_ref * adj2 + p_c_old_grad_1_ref * adj5 + p_c_old_grad_2_ref * adj8) / det;
            const scalar_t residual_tmp0 = S_res - 1;
            const scalar_t residual_tmp1 = -residual_tmp0;
            const scalar_t residual_tmp2 = pow_m1(P_r);
            const scalar_t residual_tmp3 = 1 - 1/m;
            const scalar_t residual_tmp4 = pow(pow(residual_tmp2*(p_c - p_w), m) + 1, -residual_tmp3);
            const scalar_t residual_tmp5 = residual_tmp1*residual_tmp4;
            const scalar_t residual_tmp6 = S_res + residual_tmp5;
            const scalar_t residual_tmp7 = -p_wr;
            const scalar_t residual_tmp8 = rho_w0*exp(kappa_T*(p_w + residual_tmp7));
            const scalar_t residual_tmp9 = residual_tmp1*pow(pow(residual_tmp2*(p_c_old - p_w_old), m) + 1, -residual_tmp3);
            const scalar_t residual_tmp10 = porosity/dt;
            const scalar_t residual_tmp11 = sqrt(residual_tmp6)*residual_tmp8*pow_2(1 - pow(1 - pow(residual_tmp6, pow_m1(C_kw1)), C_kw1))/mu_w;
            const scalar_t residual_tmp12 = p_w_grad_0*residual_tmp11;
            const scalar_t residual_tmp13 = p_w_grad_1*residual_tmp11;
            const scalar_t residual_tmp14 = p_w_grad_2*residual_tmp11;
            const scalar_t residual_tmp15 = M_c/(R*T*Z);
            const scalar_t residual_tmp16 = p_c*residual_tmp15;
            const scalar_t residual_tmp17 = residual_tmp16*pow(1 - residual_tmp4, C_ka1)*(1 - pow(residual_tmp4, C_ka2))/mu_c;
            const scalar_t residual_tmp18 = p_c_grad_0*residual_tmp17;
            const scalar_t residual_tmp19 = p_c_grad_1*residual_tmp17;
            const scalar_t residual_tmp20 = p_c_grad_2*residual_tmp17;
            const scalar_t value_coeff0 = residual_tmp10*(residual_tmp6*residual_tmp8 - rho_w0*(S_res + residual_tmp9)*exp(kappa_T*(p_w_old + residual_tmp7)));
            const scalar_t grad_coeff0_0 = K_0*residual_tmp12 + K_1*residual_tmp13 + K_2*residual_tmp14;
            const scalar_t grad_coeff0_1 = K_3*residual_tmp12 + K_4*residual_tmp13 + K_5*residual_tmp14;
            const scalar_t grad_coeff0_2 = K_6*residual_tmp12 + K_7*residual_tmp13 + K_8*residual_tmp14;
            const scalar_t value_coeff1 = residual_tmp10*(-p_c_old*residual_tmp15*(-residual_tmp0 - residual_tmp9) + residual_tmp16*(-residual_tmp0 - residual_tmp5));
            const scalar_t grad_coeff1_0 = K_0*residual_tmp18 + K_1*residual_tmp19 + K_2*residual_tmp20;
            const scalar_t grad_coeff1_1 = K_3*residual_tmp18 + K_4*residual_tmp19 + K_5*residual_tmp20;
            const scalar_t grad_coeff1_2 = K_6*residual_tmp18 + K_7*residual_tmp19 + K_8*residual_tmp20;
            for (int test = 0; test < N_SHAPE; ++test) {
                const scalar_t test_value = shape[q * N_SHAPE + test];
                const scalar_t test_grad0 = (grad_ref[(q * N_SHAPE + test) * DIM + 0] * adj0 + grad_ref[(q * N_SHAPE + test) * DIM + 1] * adj3 + grad_ref[(q * N_SHAPE + test) * DIM + 2] * adj6) / det;
                const scalar_t test_grad1 = (grad_ref[(q * N_SHAPE + test) * DIM + 0] * adj1 + grad_ref[(q * N_SHAPE + test) * DIM + 1] * adj4 + grad_ref[(q * N_SHAPE + test) * DIM + 2] * adj7) / det;
                const scalar_t test_grad2 = (grad_ref[(q * N_SHAPE + test) * DIM + 0] * adj2 + grad_ref[(q * N_SHAPE + test) * DIM + 1] * adj5 + grad_ref[(q * N_SHAPE + test) * DIM + 2] * adj8) / det;
                output[test * N_FIELDS + 0][lane] += q_weight[q] * det * (value_coeff0 * test_value + grad_coeff0_0 * test_grad0 + grad_coeff0_1 * test_grad1 + grad_coeff0_2 * test_grad2);
                output[test * N_FIELDS + 1][lane] += q_weight[q] * det * (value_coeff1 * test_value + grad_coeff1_0 * test_grad0 + grad_coeff1_1 * test_grad1 + grad_coeff1_2 * test_grad2);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void generated_two_phase_flow_d3_simplex_jacobian_action_block(
        const ptrdiff_t nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT current[2 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT previous[2 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT direction[2 * N_SHAPE],
        const scalar_t porosity,
        const scalar_t S_res,
        const scalar_t P_r,
        const scalar_t m,
        const scalar_t rho_w0,
        const scalar_t kappa_T,
        const scalar_t p_wr,
        const scalar_t M_c,
        const scalar_t Z,
        const scalar_t R,
        const scalar_t T,
        const scalar_t mu_w,
        const scalar_t mu_c,
        const scalar_t C_kw1,
        const scalar_t C_ka1,
        const scalar_t C_ka2,
        const scalar_t dt,
        const scalar_t K_0,
        const scalar_t K_1,
        const scalar_t K_2,
        const scalar_t K_3,
        const scalar_t K_4,
        const scalar_t K_5,
        const scalar_t K_6,
        const scalar_t K_7,
        const scalar_t K_8,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
    for (int q = 0; q < N_QP; ++q) {
#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
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
            scalar_t p_w = 0;
            scalar_t p_w_grad_0_ref = 0;
            scalar_t p_w_grad_1_ref = 0;
            scalar_t p_w_grad_2_ref = 0;
            for (int trial = 0; trial < N_SHAPE; ++trial) {
                const scalar_t coeff = current[trial * N_FIELDS + 0][lane];
                p_w += coeff * shape[q * N_SHAPE + trial];
                p_w_grad_0_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 0];
                p_w_grad_1_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 1];
                p_w_grad_2_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 2];
            }
            const scalar_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
            const scalar_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
            const scalar_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
            scalar_t p_w_old = 0;
            scalar_t p_w_old_grad_0_ref = 0;
            scalar_t p_w_old_grad_1_ref = 0;
            scalar_t p_w_old_grad_2_ref = 0;
            for (int trial = 0; trial < N_SHAPE; ++trial) {
                const scalar_t coeff = previous[trial * N_FIELDS + 0][lane];
                p_w_old += coeff * shape[q * N_SHAPE + trial];
                p_w_old_grad_0_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 0];
                p_w_old_grad_1_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 1];
                p_w_old_grad_2_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 2];
            }
            const scalar_t p_w_old_grad_0 = (p_w_old_grad_0_ref * adj0 + p_w_old_grad_1_ref * adj3 + p_w_old_grad_2_ref * adj6) / det;
            const scalar_t p_w_old_grad_1 = (p_w_old_grad_0_ref * adj1 + p_w_old_grad_1_ref * adj4 + p_w_old_grad_2_ref * adj7) / det;
            const scalar_t p_w_old_grad_2 = (p_w_old_grad_0_ref * adj2 + p_w_old_grad_1_ref * adj5 + p_w_old_grad_2_ref * adj8) / det;
            scalar_t p_w_direction = 0;
            scalar_t p_w_direction_grad_0_ref = 0;
            scalar_t p_w_direction_grad_1_ref = 0;
            scalar_t p_w_direction_grad_2_ref = 0;
            for (int trial = 0; trial < N_SHAPE; ++trial) {
                const scalar_t coeff = direction[trial * N_FIELDS + 0][lane];
                p_w_direction += coeff * shape[q * N_SHAPE + trial];
                p_w_direction_grad_0_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 0];
                p_w_direction_grad_1_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 1];
                p_w_direction_grad_2_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 2];
            }
            const scalar_t p_w_direction_grad_0 = (p_w_direction_grad_0_ref * adj0 + p_w_direction_grad_1_ref * adj3 + p_w_direction_grad_2_ref * adj6) / det;
            const scalar_t p_w_direction_grad_1 = (p_w_direction_grad_0_ref * adj1 + p_w_direction_grad_1_ref * adj4 + p_w_direction_grad_2_ref * adj7) / det;
            const scalar_t p_w_direction_grad_2 = (p_w_direction_grad_0_ref * adj2 + p_w_direction_grad_1_ref * adj5 + p_w_direction_grad_2_ref * adj8) / det;
            scalar_t p_c = 0;
            scalar_t p_c_grad_0_ref = 0;
            scalar_t p_c_grad_1_ref = 0;
            scalar_t p_c_grad_2_ref = 0;
            for (int trial = 0; trial < N_SHAPE; ++trial) {
                const scalar_t coeff = current[trial * N_FIELDS + 1][lane];
                p_c += coeff * shape[q * N_SHAPE + trial];
                p_c_grad_0_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 0];
                p_c_grad_1_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 1];
                p_c_grad_2_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 2];
            }
            const scalar_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
            const scalar_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
            const scalar_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
            scalar_t p_c_old = 0;
            scalar_t p_c_old_grad_0_ref = 0;
            scalar_t p_c_old_grad_1_ref = 0;
            scalar_t p_c_old_grad_2_ref = 0;
            for (int trial = 0; trial < N_SHAPE; ++trial) {
                const scalar_t coeff = previous[trial * N_FIELDS + 1][lane];
                p_c_old += coeff * shape[q * N_SHAPE + trial];
                p_c_old_grad_0_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 0];
                p_c_old_grad_1_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 1];
                p_c_old_grad_2_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 2];
            }
            const scalar_t p_c_old_grad_0 = (p_c_old_grad_0_ref * adj0 + p_c_old_grad_1_ref * adj3 + p_c_old_grad_2_ref * adj6) / det;
            const scalar_t p_c_old_grad_1 = (p_c_old_grad_0_ref * adj1 + p_c_old_grad_1_ref * adj4 + p_c_old_grad_2_ref * adj7) / det;
            const scalar_t p_c_old_grad_2 = (p_c_old_grad_0_ref * adj2 + p_c_old_grad_1_ref * adj5 + p_c_old_grad_2_ref * adj8) / det;
            scalar_t p_c_direction = 0;
            scalar_t p_c_direction_grad_0_ref = 0;
            scalar_t p_c_direction_grad_1_ref = 0;
            scalar_t p_c_direction_grad_2_ref = 0;
            for (int trial = 0; trial < N_SHAPE; ++trial) {
                const scalar_t coeff = direction[trial * N_FIELDS + 1][lane];
                p_c_direction += coeff * shape[q * N_SHAPE + trial];
                p_c_direction_grad_0_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 0];
                p_c_direction_grad_1_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 1];
                p_c_direction_grad_2_ref += coeff * grad_ref[(q * N_SHAPE + trial) * DIM + 2];
            }
            const scalar_t p_c_direction_grad_0 = (p_c_direction_grad_0_ref * adj0 + p_c_direction_grad_1_ref * adj3 + p_c_direction_grad_2_ref * adj6) / det;
            const scalar_t p_c_direction_grad_1 = (p_c_direction_grad_0_ref * adj1 + p_c_direction_grad_1_ref * adj4 + p_c_direction_grad_2_ref * adj7) / det;
            const scalar_t p_c_direction_grad_2 = (p_c_direction_grad_0_ref * adj2 + p_c_direction_grad_1_ref * adj5 + p_c_direction_grad_2_ref * adj8) / det;
            const scalar_t residual_tmp0 = porosity/dt;
            const scalar_t residual_tmp1 = p_c_direction*residual_tmp0;
            const scalar_t residual_tmp2 = exp(kappa_T*(p_w - p_wr));
            const scalar_t residual_tmp3 = residual_tmp2*rho_w0;
            const scalar_t residual_tmp4 = S_res - 1;
            const scalar_t residual_tmp5 = p_c - p_w;
            const scalar_t residual_tmp6 = pow(residual_tmp5/P_r, m);
            const scalar_t residual_tmp7 = residual_tmp6 + 1;
            const scalar_t residual_tmp8 = 1 - 1/m;
            const scalar_t residual_tmp9 = pow(residual_tmp7, residual_tmp8);
            const scalar_t residual_tmp10 = pow_m1(residual_tmp9);
            const scalar_t residual_tmp11 = -residual_tmp10*residual_tmp4;
            const scalar_t residual_tmp12 = -m*residual_tmp6*residual_tmp8/(residual_tmp5*residual_tmp7);
            const scalar_t residual_tmp13 = residual_tmp11*residual_tmp12;
            const scalar_t residual_tmp14 = residual_tmp13*residual_tmp3;
            const scalar_t residual_tmp15 = S_res + residual_tmp11;
            const scalar_t residual_tmp16 = p_w_direction*residual_tmp0;
            const scalar_t residual_tmp17 = pow_m1(mu_w);
            const scalar_t residual_tmp18 = residual_tmp17*residual_tmp3;
            const scalar_t residual_tmp19 = K_0*residual_tmp18;
            const scalar_t residual_tmp20 = pow(residual_tmp15, pow_m1(C_kw1));
            const scalar_t residual_tmp21 = 1 - residual_tmp20;
            const scalar_t residual_tmp22 = pow(residual_tmp21, C_kw1);
            const scalar_t residual_tmp23 = 1 - residual_tmp22;
            const scalar_t residual_tmp24 = pow_2(residual_tmp23);
            const scalar_t residual_tmp25 = sqrt(residual_tmp15);
            const scalar_t residual_tmp26 = residual_tmp24*residual_tmp25;
            const scalar_t residual_tmp27 = p_w_direction_grad_0*residual_tmp26;
            const scalar_t residual_tmp28 = residual_tmp18*residual_tmp26;
            const scalar_t residual_tmp29 = p_w_direction_grad_1*residual_tmp28;
            const scalar_t residual_tmp30 = p_w_direction_grad_2*residual_tmp28;
            const scalar_t residual_tmp31 = (1.0/2.0)*residual_tmp24;
            const scalar_t residual_tmp32 = residual_tmp13/residual_tmp25;
            const scalar_t residual_tmp33 = p_w_grad_0*residual_tmp32;
            const scalar_t residual_tmp34 = residual_tmp19*residual_tmp33;
            const scalar_t residual_tmp35 = residual_tmp18*residual_tmp31;
            const scalar_t residual_tmp36 = residual_tmp32*residual_tmp35;
            const scalar_t residual_tmp37 = p_w_grad_1*residual_tmp36;
            const scalar_t residual_tmp38 = p_w_grad_2*residual_tmp36;
            const scalar_t residual_tmp39 = 2*residual_tmp20*residual_tmp22*residual_tmp23/residual_tmp21;
            const scalar_t residual_tmp40 = residual_tmp18*residual_tmp39;
            const scalar_t residual_tmp41 = residual_tmp32*residual_tmp40;
            const scalar_t residual_tmp42 = p_w_grad_1*residual_tmp41;
            const scalar_t residual_tmp43 = p_w_grad_2*residual_tmp41;
            const scalar_t residual_tmp44 = K_1*residual_tmp37 + K_1*residual_tmp42 + K_2*residual_tmp38 + K_2*residual_tmp43 + residual_tmp31*residual_tmp34 + residual_tmp34*residual_tmp39;
            const scalar_t residual_tmp45 = residual_tmp18*residual_tmp27;
            const scalar_t residual_tmp46 = residual_tmp33*residual_tmp35;
            const scalar_t residual_tmp47 = residual_tmp33*residual_tmp40;
            const scalar_t residual_tmp48 = K_3*residual_tmp46 + K_3*residual_tmp47 + K_4*residual_tmp37 + K_4*residual_tmp42 + K_5*residual_tmp38 + K_5*residual_tmp43;
            const scalar_t residual_tmp49 = K_6*residual_tmp46 + K_6*residual_tmp47 + K_7*residual_tmp37 + K_7*residual_tmp42 + K_8*residual_tmp38 + K_8*residual_tmp43;
            const scalar_t residual_tmp50 = pow_m1(R);
            const scalar_t residual_tmp51 = pow_m1(T);
            const scalar_t residual_tmp52 = pow_m1(Z);
            const scalar_t residual_tmp53 = M_c*residual_tmp50*residual_tmp51*residual_tmp52;
            const scalar_t residual_tmp54 = p_c*residual_tmp13*residual_tmp53;
            const scalar_t residual_tmp55 = pow(residual_tmp10, C_ka2);
            const scalar_t residual_tmp56 = 1 - residual_tmp55;
            const scalar_t residual_tmp57 = pow_m1(mu_c);
            const scalar_t residual_tmp58 = 1 - residual_tmp10;
            const scalar_t residual_tmp59 = pow(residual_tmp58, C_ka1);
            const scalar_t residual_tmp60 = residual_tmp53*residual_tmp57*residual_tmp59;
            const scalar_t residual_tmp61 = residual_tmp56*residual_tmp60;
            const scalar_t residual_tmp62 = p_c*residual_tmp61;
            const scalar_t residual_tmp63 = p_c_direction_grad_0*residual_tmp62;
            const scalar_t residual_tmp64 = p_c_direction_grad_1*residual_tmp62;
            const scalar_t residual_tmp65 = p_c_direction_grad_2*residual_tmp62;
            const scalar_t residual_tmp66 = p_c*residual_tmp10*residual_tmp12;
            const scalar_t residual_tmp67 = C_ka2*residual_tmp55*residual_tmp60*residual_tmp66*residual_tmp9;
            const scalar_t residual_tmp68 = p_c_grad_0*residual_tmp67;
            const scalar_t residual_tmp69 = p_c_grad_1*residual_tmp67;
            const scalar_t residual_tmp70 = p_c_grad_2*residual_tmp67;
            const scalar_t residual_tmp71 = p_c_grad_0*residual_tmp61;
            const scalar_t residual_tmp72 = C_ka1*residual_tmp66/residual_tmp58;
            const scalar_t residual_tmp73 = p_c_grad_1*residual_tmp61;
            const scalar_t residual_tmp74 = p_c_grad_2*residual_tmp61;
            const scalar_t residual_tmp75 = K_0*residual_tmp68 + K_0*residual_tmp71*residual_tmp72 + K_1*residual_tmp69 + K_1*residual_tmp72*residual_tmp73 + K_2*residual_tmp70 + K_2*residual_tmp72*residual_tmp74;
            const scalar_t residual_tmp76 = K_3*residual_tmp68 + K_3*residual_tmp71*residual_tmp72 + K_4*residual_tmp69 + K_4*residual_tmp72*residual_tmp73 + K_5*residual_tmp70 + K_5*residual_tmp72*residual_tmp74;
            const scalar_t residual_tmp77 = K_6*residual_tmp68 + K_6*residual_tmp71*residual_tmp72 + K_7*residual_tmp69 + K_7*residual_tmp72*residual_tmp73 + K_8*residual_tmp70 + K_8*residual_tmp72*residual_tmp74;
            const scalar_t value_coeff0 = residual_tmp1*residual_tmp14 + residual_tmp16*(kappa_T*residual_tmp15*residual_tmp3 - residual_tmp14);
            const scalar_t grad_coeff0_0 = K_1*residual_tmp29 + K_2*residual_tmp30 + p_c_direction*residual_tmp44 + p_w_direction*(K_0*kappa_T*p_w_grad_0*residual_tmp17*residual_tmp2*residual_tmp24*residual_tmp25*rho_w0 + K_1*kappa_T*p_w_grad_1*residual_tmp17*residual_tmp2*residual_tmp24*residual_tmp25*rho_w0 + K_2*kappa_T*p_w_grad_2*residual_tmp17*residual_tmp2*residual_tmp24*residual_tmp25*rho_w0 - residual_tmp44) + residual_tmp19*residual_tmp27;
            const scalar_t grad_coeff0_1 = K_3*residual_tmp45 + K_4*residual_tmp29 + K_5*residual_tmp30 + p_c_direction*residual_tmp48 + p_w_direction*(K_3*kappa_T*p_w_grad_0*residual_tmp17*residual_tmp2*residual_tmp24*residual_tmp25*rho_w0 + K_4*kappa_T*p_w_grad_1*residual_tmp17*residual_tmp2*residual_tmp24*residual_tmp25*rho_w0 + K_5*kappa_T*p_w_grad_2*residual_tmp17*residual_tmp2*residual_tmp24*residual_tmp25*rho_w0 - residual_tmp48);
            const scalar_t grad_coeff0_2 = K_6*residual_tmp45 + K_7*residual_tmp29 + K_8*residual_tmp30 + p_c_direction*residual_tmp49 + p_w_direction*(K_6*kappa_T*p_w_grad_0*residual_tmp17*residual_tmp2*residual_tmp24*residual_tmp25*rho_w0 + K_7*kappa_T*p_w_grad_1*residual_tmp17*residual_tmp2*residual_tmp24*residual_tmp25*rho_w0 + K_8*kappa_T*p_w_grad_2*residual_tmp17*residual_tmp2*residual_tmp24*residual_tmp25*rho_w0 - residual_tmp49);
            const scalar_t value_coeff1 = residual_tmp1*(M_c*residual_tmp50*residual_tmp51*residual_tmp52*(-residual_tmp11 - residual_tmp4) - residual_tmp54) + residual_tmp16*residual_tmp54;
            const scalar_t grad_coeff1_0 = K_0*residual_tmp63 + K_1*residual_tmp64 + K_2*residual_tmp65 + p_c_direction*(K_0*M_c*p_c_grad_0*residual_tmp50*residual_tmp51*residual_tmp52*residual_tmp56*residual_tmp57*residual_tmp59 + K_1*M_c*p_c_grad_1*residual_tmp50*residual_tmp51*residual_tmp52*residual_tmp56*residual_tmp57*residual_tmp59 + K_2*M_c*p_c_grad_2*residual_tmp50*residual_tmp51*residual_tmp52*residual_tmp56*residual_tmp57*residual_tmp59 - residual_tmp75) + p_w_direction*residual_tmp75;
            const scalar_t grad_coeff1_1 = K_3*residual_tmp63 + K_4*residual_tmp64 + K_5*residual_tmp65 + p_c_direction*(K_3*M_c*p_c_grad_0*residual_tmp50*residual_tmp51*residual_tmp52*residual_tmp56*residual_tmp57*residual_tmp59 + K_4*M_c*p_c_grad_1*residual_tmp50*residual_tmp51*residual_tmp52*residual_tmp56*residual_tmp57*residual_tmp59 + K_5*M_c*p_c_grad_2*residual_tmp50*residual_tmp51*residual_tmp52*residual_tmp56*residual_tmp57*residual_tmp59 - residual_tmp76) + p_w_direction*residual_tmp76;
            const scalar_t grad_coeff1_2 = K_6*residual_tmp63 + K_7*residual_tmp64 + K_8*residual_tmp65 + p_c_direction*(K_6*M_c*p_c_grad_0*residual_tmp50*residual_tmp51*residual_tmp52*residual_tmp56*residual_tmp57*residual_tmp59 + K_7*M_c*p_c_grad_1*residual_tmp50*residual_tmp51*residual_tmp52*residual_tmp56*residual_tmp57*residual_tmp59 + K_8*M_c*p_c_grad_2*residual_tmp50*residual_tmp51*residual_tmp52*residual_tmp56*residual_tmp57*residual_tmp59 - residual_tmp77) + p_w_direction*residual_tmp77;
            for (int test = 0; test < N_SHAPE; ++test) {
                const scalar_t test_value = shape[q * N_SHAPE + test];
                const scalar_t test_grad0 = (grad_ref[(q * N_SHAPE + test) * DIM + 0] * adj0 + grad_ref[(q * N_SHAPE + test) * DIM + 1] * adj3 + grad_ref[(q * N_SHAPE + test) * DIM + 2] * adj6) / det;
                const scalar_t test_grad1 = (grad_ref[(q * N_SHAPE + test) * DIM + 0] * adj1 + grad_ref[(q * N_SHAPE + test) * DIM + 1] * adj4 + grad_ref[(q * N_SHAPE + test) * DIM + 2] * adj7) / det;
                const scalar_t test_grad2 = (grad_ref[(q * N_SHAPE + test) * DIM + 0] * adj2 + grad_ref[(q * N_SHAPE + test) * DIM + 1] * adj5 + grad_ref[(q * N_SHAPE + test) * DIM + 2] * adj8) / det;
                output[test * N_FIELDS + 0][lane] += q_weight[q] * det * (value_coeff0 * test_value + grad_coeff0_0 * test_grad0 + grad_coeff0_1 * test_grad1 + grad_coeff0_2 * test_grad2);
                output[test * N_FIELDS + 1][lane] += q_weight[q] * det * (value_coeff1 * test_value + grad_coeff1_0 * test_grad0 + grad_coeff1_1 * test_grad1 + grad_coeff1_2 * test_grad2);
            }
        }
    }
}

} // namespace codegen
} // namespace sfem

#endif
