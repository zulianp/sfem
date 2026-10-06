#ifndef TWO_PHASE_FLOW_D3_SIMPLEX_LOCAL_HPP
#define TWO_PHASE_FLOW_D3_SIMPLEX_LOCAL_HPP

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
static SFEM_INLINE void two_phase_flow_d3_simplex_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT current[2 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT previous[2 * N_SHAPE],
        const scalar_t C_ka1,
        const scalar_t C_ka2,
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
        const scalar_t M_c,
        const scalar_t P_r,
        const scalar_t R,
        const scalar_t S_res,
        const scalar_t T,
        const scalar_t Z,
        const scalar_t dt,
        const scalar_t kappa_T,
        const scalar_t m,
        const scalar_t mu_c,
        const scalar_t mu_w,
        const scalar_t p_wr,
        const scalar_t porosity,
        const scalar_t rho_w0,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
    for (int q = 0; q < N_QP; ++q) {
        scalar_t p_w_values[VECTOR_SIZE];
        scalar_t p_w_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_w_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_w_grad_2_ref_values[VECTOR_SIZE];
        scalar_t p_w_old_values[VECTOR_SIZE];
        scalar_t p_c_values[VECTOR_SIZE];
        scalar_t p_c_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_c_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_c_grad_2_ref_values[VECTOR_SIZE];
        scalar_t p_c_old_values[VECTOR_SIZE];
        scalar_t value_coeff0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_1_values[VECTOR_SIZE];
        scalar_t grad_coeff0_2_values[VECTOR_SIZE];
        scalar_t value_coeff1_values[VECTOR_SIZE];
        scalar_t grad_coeff1_0_values[VECTOR_SIZE];
        scalar_t grad_coeff1_1_values[VECTOR_SIZE];
        scalar_t grad_coeff1_2_values[VECTOR_SIZE];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 0][lane];
                p_w_values[lane] += coeff * shape[q * N_SHAPE + trial];
                p_w_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                p_w_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                p_w_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_old_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 0][lane];
                p_w_old_values[lane] += coeff * shape[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 1][lane];
                p_c_values[lane] += coeff * shape[q * N_SHAPE + trial];
                p_c_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                p_c_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                p_c_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_old_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 1][lane];
                p_c_old_values[lane] += coeff * shape[q * N_SHAPE + trial];
            }
        }
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
            const scalar_t p_w = p_w_values[lane];
            const scalar_t p_w_grad_0_ref = p_w_grad_0_ref_values[lane];
            const scalar_t p_w_grad_1_ref = p_w_grad_1_ref_values[lane];
            const scalar_t p_w_grad_2_ref = p_w_grad_2_ref_values[lane];
            const scalar_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
            const scalar_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
            const scalar_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
            const scalar_t p_w_old = p_w_old_values[lane];
            const scalar_t p_c = p_c_values[lane];
            const scalar_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
            const scalar_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
            const scalar_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
            const scalar_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
            const scalar_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
            const scalar_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
            const scalar_t p_c_old = p_c_old_values[lane];
            const scalar_t residual_tmp0 = -p_wr;
            const scalar_t residual_tmp1 = exp(kappa_T*(p_w + residual_tmp0));
            const scalar_t residual_tmp2 = S_res + scalar_t(-1);
            const scalar_t residual_tmp3 = -residual_tmp2;
            const scalar_t residual_tmp4 = pow_m1(P_r);
            const scalar_t residual_tmp5 = (scalar_t(1) - m)/m;
            const scalar_t residual_tmp6 = pow(pow(residual_tmp4*(p_c - p_w), m) + scalar_t(1), residual_tmp5);
            const scalar_t residual_tmp7 = pow(pow(residual_tmp4*(p_c_old - p_w_old), m) + scalar_t(1), residual_tmp5);
            const scalar_t residual_tmp8 = porosity/dt;
            const scalar_t residual_tmp9 = residual_tmp2*residual_tmp6;
            const scalar_t residual_tmp10 = S_res - residual_tmp9;
            const scalar_t residual_tmp11 = residual_tmp1*sqrt(residual_tmp10)*rho_w0*pow_2(pow(scalar_t(1) - pow(residual_tmp10, pow_m1(C_kw1)), C_kw1) + scalar_t(-1))/mu_w;
            const scalar_t residual_tmp12 = scalar_t(1) - S_res;
            const scalar_t residual_tmp13 = M_c/(R*T*Z);
            const scalar_t residual_tmp14 = p_c*residual_tmp13*pow(scalar_t(1) - residual_tmp6, C_ka1)*(pow(residual_tmp6, C_ka2) + scalar_t(-1))/mu_c;
            const scalar_t value_coeff0 = residual_tmp8*rho_w0*(residual_tmp1*(S_res + residual_tmp3*residual_tmp6) - (S_res + residual_tmp3*residual_tmp7)*exp(kappa_T*(p_w_old + residual_tmp0)));
            const scalar_t grad_coeff0_0 = residual_tmp11*(K_0*p_w_grad_0 + K_1*p_w_grad_1 + K_2*p_w_grad_2);
            const scalar_t grad_coeff0_1 = residual_tmp11*(K_3*p_w_grad_0 + K_4*p_w_grad_1 + K_5*p_w_grad_2);
            const scalar_t grad_coeff0_2 = residual_tmp11*(K_6*p_w_grad_0 + K_7*p_w_grad_1 + K_8*p_w_grad_2);
            const scalar_t value_coeff1 = -residual_tmp13*residual_tmp8*(-p_c*(residual_tmp12 + residual_tmp9) + p_c_old*(residual_tmp12 + residual_tmp2*residual_tmp7));
            const scalar_t grad_coeff1_0 = residual_tmp14*(-K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2);
            const scalar_t grad_coeff1_1 = residual_tmp14*(-K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2);
            const scalar_t grad_coeff1_2 = residual_tmp14*(-K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2);
            value_coeff0_values[lane] = value_coeff0;
            grad_coeff0_0_values[lane] = grad_coeff0_0;
            grad_coeff0_1_values[lane] = grad_coeff0_1;
            grad_coeff0_2_values[lane] = grad_coeff0_2;
            value_coeff1_values[lane] = value_coeff1;
            grad_coeff1_0_values[lane] = grad_coeff1_0;
            grad_coeff1_1_values[lane] = grad_coeff1_1;
            grad_coeff1_2_values[lane] = grad_coeff1_2;
        }
        for (int test = 0; test < N_SHAPE; ++test) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const ptrdiff_t geometry_offset = q * geometry_stride + lane;
                const scalar_t det = determinant[geometry_offset];
                const scalar_t test_value = shape[q * N_SHAPE + test];
                const scalar_t adj0 = adjugate[0][geometry_offset];
                const scalar_t adj1 = adjugate[1][geometry_offset];
                const scalar_t adj2 = adjugate[2][geometry_offset];
                const scalar_t adj3 = adjugate[3][geometry_offset];
                const scalar_t adj4 = adjugate[4][geometry_offset];
                const scalar_t adj5 = adjugate[5][geometry_offset];
                const scalar_t adj6 = adjugate[6][geometry_offset];
                const scalar_t adj7 = adjugate[7][geometry_offset];
                const scalar_t adj8 = adjugate[8][geometry_offset];
                const scalar_t test_grad0 = (grad_ref_x[q * N_SHAPE + test] * adj0 + grad_ref_y[q * N_SHAPE + test] * adj3 + grad_ref_z[q * N_SHAPE + test] * adj6) / det;
                const scalar_t test_grad1 = (grad_ref_x[q * N_SHAPE + test] * adj1 + grad_ref_y[q * N_SHAPE + test] * adj4 + grad_ref_z[q * N_SHAPE + test] * adj7) / det;
                const scalar_t test_grad2 = (grad_ref_x[q * N_SHAPE + test] * adj2 + grad_ref_y[q * N_SHAPE + test] * adj5 + grad_ref_z[q * N_SHAPE + test] * adj8) / det;
                output[test * N_FIELDS + 0][lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value + grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
                output[test * N_FIELDS + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void two_phase_flow_d3_simplex_tet4_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT current[2 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT previous[2 * N_SHAPE],
        const scalar_t C_ka1,
        const scalar_t C_ka2,
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
        const scalar_t M_c,
        const scalar_t P_r,
        const scalar_t R,
        const scalar_t S_res,
        const scalar_t T,
        const scalar_t Z,
        const scalar_t dt,
        const scalar_t kappa_T,
        const scalar_t m,
        const scalar_t mu_c,
        const scalar_t mu_w,
        const scalar_t p_wr,
        const scalar_t porosity,
        const scalar_t rho_w0,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
    for (int q = 0; q < N_QP; ++q) {
        scalar_t p_w_values[VECTOR_SIZE];
        scalar_t p_w_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_w_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_w_grad_2_ref_values[VECTOR_SIZE];
        scalar_t p_w_old_values[VECTOR_SIZE];
        scalar_t p_c_values[VECTOR_SIZE];
        scalar_t p_c_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_c_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_c_grad_2_ref_values[VECTOR_SIZE];
        scalar_t p_c_old_values[VECTOR_SIZE];
        scalar_t value_coeff0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_1_values[VECTOR_SIZE];
        scalar_t grad_coeff0_2_values[VECTOR_SIZE];
        scalar_t value_coeff1_values[VECTOR_SIZE];
        scalar_t grad_coeff1_0_values[VECTOR_SIZE];
        scalar_t grad_coeff1_1_values[VECTOR_SIZE];
        scalar_t grad_coeff1_2_values[VECTOR_SIZE];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 0][lane];
                p_w_values[lane] += coeff * shape[q * N_SHAPE + trial];
                p_w_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                p_w_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                p_w_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_old_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 0][lane];
                p_w_old_values[lane] += coeff * shape[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 1][lane];
                p_c_values[lane] += coeff * shape[q * N_SHAPE + trial];
                p_c_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                p_c_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                p_c_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_old_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 1][lane];
                p_c_old_values[lane] += coeff * shape[q * N_SHAPE + trial];
            }
        }
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
            const scalar_t p_w = p_w_values[lane];
            const scalar_t p_w_grad_0_ref = p_w_grad_0_ref_values[lane];
            const scalar_t p_w_grad_1_ref = p_w_grad_1_ref_values[lane];
            const scalar_t p_w_grad_2_ref = p_w_grad_2_ref_values[lane];
            const scalar_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
            const scalar_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
            const scalar_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
            const scalar_t p_w_old = p_w_old_values[lane];
            const scalar_t p_c = p_c_values[lane];
            const scalar_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
            const scalar_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
            const scalar_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
            const scalar_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
            const scalar_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
            const scalar_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
            const scalar_t p_c_old = p_c_old_values[lane];
            const scalar_t residual_tmp0 = -p_wr;
            const scalar_t residual_tmp1 = exp(kappa_T*(p_w + residual_tmp0));
            const scalar_t residual_tmp2 = S_res + scalar_t(-1);
            const scalar_t residual_tmp3 = -residual_tmp2;
            const scalar_t residual_tmp4 = pow_m1(P_r);
            const scalar_t residual_tmp5 = (scalar_t(1) - m)/m;
            const scalar_t residual_tmp6 = pow(pow(residual_tmp4*(p_c - p_w), m) + scalar_t(1), residual_tmp5);
            const scalar_t residual_tmp7 = pow(pow(residual_tmp4*(p_c_old - p_w_old), m) + scalar_t(1), residual_tmp5);
            const scalar_t residual_tmp8 = porosity/dt;
            const scalar_t residual_tmp9 = residual_tmp2*residual_tmp6;
            const scalar_t residual_tmp10 = S_res - residual_tmp9;
            const scalar_t residual_tmp11 = residual_tmp1*sqrt(residual_tmp10)*rho_w0*pow_2(pow(scalar_t(1) - pow(residual_tmp10, pow_m1(C_kw1)), C_kw1) + scalar_t(-1))/mu_w;
            const scalar_t residual_tmp12 = scalar_t(1) - S_res;
            const scalar_t residual_tmp13 = M_c/(R*T*Z);
            const scalar_t residual_tmp14 = p_c*residual_tmp13*pow(scalar_t(1) - residual_tmp6, C_ka1)*(pow(residual_tmp6, C_ka2) + scalar_t(-1))/mu_c;
            const scalar_t value_coeff0 = residual_tmp8*rho_w0*(residual_tmp1*(S_res + residual_tmp3*residual_tmp6) - (S_res + residual_tmp3*residual_tmp7)*exp(kappa_T*(p_w_old + residual_tmp0)));
            const scalar_t grad_coeff0_0 = residual_tmp11*(K_0*p_w_grad_0 + K_1*p_w_grad_1 + K_2*p_w_grad_2);
            const scalar_t grad_coeff0_1 = residual_tmp11*(K_3*p_w_grad_0 + K_4*p_w_grad_1 + K_5*p_w_grad_2);
            const scalar_t grad_coeff0_2 = residual_tmp11*(K_6*p_w_grad_0 + K_7*p_w_grad_1 + K_8*p_w_grad_2);
            const scalar_t value_coeff1 = -residual_tmp13*residual_tmp8*(-p_c*(residual_tmp12 + residual_tmp9) + p_c_old*(residual_tmp12 + residual_tmp2*residual_tmp7));
            const scalar_t grad_coeff1_0 = residual_tmp14*(-K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2);
            const scalar_t grad_coeff1_1 = residual_tmp14*(-K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2);
            const scalar_t grad_coeff1_2 = residual_tmp14*(-K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2);
            value_coeff0_values[lane] = value_coeff0;
            grad_coeff0_0_values[lane] = grad_coeff0_0;
            grad_coeff0_1_values[lane] = grad_coeff0_1;
            grad_coeff0_2_values[lane] = grad_coeff0_2;
            value_coeff1_values[lane] = value_coeff1;
            grad_coeff1_0_values[lane] = grad_coeff1_0;
            grad_coeff1_1_values[lane] = grad_coeff1_1;
            grad_coeff1_2_values[lane] = grad_coeff1_2;
        }
        for (int test = 0; test < N_SHAPE; ++test) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const ptrdiff_t geometry_offset = q * geometry_stride + lane;
                const scalar_t det = determinant[geometry_offset];
                const scalar_t test_value = shape[q * N_SHAPE + test];
                const scalar_t adj0 = adjugate[0][geometry_offset];
                const scalar_t adj1 = adjugate[1][geometry_offset];
                const scalar_t adj2 = adjugate[2][geometry_offset];
                const scalar_t adj3 = adjugate[3][geometry_offset];
                const scalar_t adj4 = adjugate[4][geometry_offset];
                const scalar_t adj5 = adjugate[5][geometry_offset];
                const scalar_t adj6 = adjugate[6][geometry_offset];
                const scalar_t adj7 = adjugate[7][geometry_offset];
                const scalar_t adj8 = adjugate[8][geometry_offset];
                const scalar_t test_grad0 = (grad_ref_x[q * N_SHAPE + test] * adj0 + grad_ref_y[q * N_SHAPE + test] * adj3 + grad_ref_z[q * N_SHAPE + test] * adj6) / det;
                const scalar_t test_grad1 = (grad_ref_x[q * N_SHAPE + test] * adj1 + grad_ref_y[q * N_SHAPE + test] * adj4 + grad_ref_z[q * N_SHAPE + test] * adj7) / det;
                const scalar_t test_grad2 = (grad_ref_x[q * N_SHAPE + test] * adj2 + grad_ref_y[q * N_SHAPE + test] * adj5 + grad_ref_z[q * N_SHAPE + test] * adj8) / det;
                output[test * N_FIELDS + 0][lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value + grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
                output[test * N_FIELDS + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void two_phase_flow_d3_simplex_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT current[2 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT direction[2 * N_SHAPE],
        const scalar_t C_ka1,
        const scalar_t C_ka2,
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
        const scalar_t M_c,
        const scalar_t P_r,
        const scalar_t R,
        const scalar_t S_res,
        const scalar_t T,
        const scalar_t Z,
        const scalar_t dt,
        const scalar_t kappa_T,
        const scalar_t m,
        const scalar_t mu_c,
        const scalar_t mu_w,
        const scalar_t p_wr,
        const scalar_t porosity,
        const scalar_t rho_w0,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
    for (int q = 0; q < N_QP; ++q) {
        scalar_t p_w_values[VECTOR_SIZE];
        scalar_t p_w_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_w_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_w_grad_2_ref_values[VECTOR_SIZE];
        scalar_t p_w_direction_values[VECTOR_SIZE];
        scalar_t p_w_direction_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_w_direction_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_w_direction_grad_2_ref_values[VECTOR_SIZE];
        scalar_t p_c_values[VECTOR_SIZE];
        scalar_t p_c_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_c_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_c_grad_2_ref_values[VECTOR_SIZE];
        scalar_t p_c_direction_values[VECTOR_SIZE];
        scalar_t p_c_direction_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_c_direction_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_c_direction_grad_2_ref_values[VECTOR_SIZE];
        scalar_t value_coeff0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_1_values[VECTOR_SIZE];
        scalar_t grad_coeff0_2_values[VECTOR_SIZE];
        scalar_t value_coeff1_values[VECTOR_SIZE];
        scalar_t grad_coeff1_0_values[VECTOR_SIZE];
        scalar_t grad_coeff1_1_values[VECTOR_SIZE];
        scalar_t grad_coeff1_2_values[VECTOR_SIZE];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 0][lane];
                p_w_values[lane] += coeff * shape[q * N_SHAPE + trial];
                p_w_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                p_w_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                p_w_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_direction_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_direction_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_direction_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_direction_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 0][lane];
                p_w_direction_values[lane] += coeff * shape[q * N_SHAPE + trial];
                p_w_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                p_w_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                p_w_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 1][lane];
                p_c_values[lane] += coeff * shape[q * N_SHAPE + trial];
                p_c_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                p_c_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                p_c_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_direction_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_direction_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_direction_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_direction_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 1][lane];
                p_c_direction_values[lane] += coeff * shape[q * N_SHAPE + trial];
                p_c_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                p_c_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                p_c_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
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
            const scalar_t p_w = p_w_values[lane];
            const scalar_t p_w_grad_0_ref = p_w_grad_0_ref_values[lane];
            const scalar_t p_w_grad_1_ref = p_w_grad_1_ref_values[lane];
            const scalar_t p_w_grad_2_ref = p_w_grad_2_ref_values[lane];
            const scalar_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
            const scalar_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
            const scalar_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
            const scalar_t p_w_direction = p_w_direction_values[lane];
            const scalar_t p_w_direction_grad_0_ref = p_w_direction_grad_0_ref_values[lane];
            const scalar_t p_w_direction_grad_1_ref = p_w_direction_grad_1_ref_values[lane];
            const scalar_t p_w_direction_grad_2_ref = p_w_direction_grad_2_ref_values[lane];
            const scalar_t p_w_direction_grad_0 = (p_w_direction_grad_0_ref * adj0 + p_w_direction_grad_1_ref * adj3 + p_w_direction_grad_2_ref * adj6) / det;
            const scalar_t p_w_direction_grad_1 = (p_w_direction_grad_0_ref * adj1 + p_w_direction_grad_1_ref * adj4 + p_w_direction_grad_2_ref * adj7) / det;
            const scalar_t p_w_direction_grad_2 = (p_w_direction_grad_0_ref * adj2 + p_w_direction_grad_1_ref * adj5 + p_w_direction_grad_2_ref * adj8) / det;
            const scalar_t p_c = p_c_values[lane];
            const scalar_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
            const scalar_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
            const scalar_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
            const scalar_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
            const scalar_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
            const scalar_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
            const scalar_t p_c_direction = p_c_direction_values[lane];
            const scalar_t p_c_direction_grad_0_ref = p_c_direction_grad_0_ref_values[lane];
            const scalar_t p_c_direction_grad_1_ref = p_c_direction_grad_1_ref_values[lane];
            const scalar_t p_c_direction_grad_2_ref = p_c_direction_grad_2_ref_values[lane];
            const scalar_t p_c_direction_grad_0 = (p_c_direction_grad_0_ref * adj0 + p_c_direction_grad_1_ref * adj3 + p_c_direction_grad_2_ref * adj6) / det;
            const scalar_t p_c_direction_grad_1 = (p_c_direction_grad_0_ref * adj1 + p_c_direction_grad_1_ref * adj4 + p_c_direction_grad_2_ref * adj7) / det;
            const scalar_t p_c_direction_grad_2 = (p_c_direction_grad_0_ref * adj2 + p_c_direction_grad_1_ref * adj5 + p_c_direction_grad_2_ref * adj8) / det;
            const scalar_t residual_tmp0 = pow_m1(dt);
            const scalar_t residual_tmp1 = residual_tmp0*rho_w0;
            const scalar_t residual_tmp2 = p_c_direction*residual_tmp1;
            const scalar_t residual_tmp3 = S_res + scalar_t(-1);
            const scalar_t residual_tmp4 = p_c - p_w;
            const scalar_t residual_tmp5 = pow(residual_tmp4/P_r, m);
            const scalar_t residual_tmp6 = residual_tmp5 + scalar_t(1);
            const scalar_t residual_tmp7 = scalar_t(1) - m;
            const scalar_t residual_tmp8 = pow(residual_tmp6, residual_tmp7/m);
            const scalar_t residual_tmp9 = -residual_tmp3*residual_tmp8;
            const scalar_t residual_tmp10 = exp(kappa_T*(p_w - p_wr));
            const scalar_t residual_tmp11 = residual_tmp5*residual_tmp7/(residual_tmp4*residual_tmp6);
            const scalar_t residual_tmp12 = residual_tmp10*residual_tmp11;
            const scalar_t residual_tmp13 = residual_tmp12*residual_tmp9;
            const scalar_t residual_tmp14 = kappa_T*residual_tmp10;
            const scalar_t residual_tmp15 = p_w_direction*residual_tmp1;
            const scalar_t residual_tmp16 = pow_m1(mu_w);
            const scalar_t residual_tmp17 = residual_tmp3*residual_tmp8;
            const scalar_t residual_tmp18 = S_res - residual_tmp17;
            const scalar_t residual_tmp19 = pow(residual_tmp18, pow_m1(C_kw1));
            const scalar_t residual_tmp20 = scalar_t(1) - residual_tmp19;
            const scalar_t residual_tmp21 = pow(residual_tmp20, C_kw1);
            const scalar_t residual_tmp22 = residual_tmp21 + scalar_t(-1);
            const scalar_t residual_tmp23 = pow_2(residual_tmp22);
            const scalar_t residual_tmp24 = sqrt(residual_tmp18);
            const scalar_t residual_tmp25 = residual_tmp23*residual_tmp24;
            const scalar_t residual_tmp26 = residual_tmp10*residual_tmp16*residual_tmp25*rho_w0;
            const scalar_t residual_tmp27 = p_w_direction_grad_0*residual_tmp26;
            const scalar_t residual_tmp28 = p_w_direction_grad_1*residual_tmp26;
            const scalar_t residual_tmp29 = p_w_direction_grad_2*residual_tmp26;
            const scalar_t residual_tmp30 = dt*(K_0*p_w_grad_0 + K_1*p_w_grad_1 + K_2*p_w_grad_2);
            const scalar_t residual_tmp31 = residual_tmp12*residual_tmp17/residual_tmp24;
            const scalar_t residual_tmp32 = residual_tmp30*residual_tmp31;
            const scalar_t residual_tmp33 = ((scalar_t(1) / scalar_t(2)))*residual_tmp23;
            const scalar_t residual_tmp34 = scalar_t(2)*residual_tmp19*residual_tmp21*residual_tmp22/residual_tmp20;
            const scalar_t residual_tmp35 = residual_tmp32*residual_tmp33 - residual_tmp32*residual_tmp34;
            const scalar_t residual_tmp36 = residual_tmp16*residual_tmp2;
            const scalar_t residual_tmp37 = residual_tmp14*residual_tmp25;
            const scalar_t residual_tmp38 = residual_tmp15*residual_tmp16;
            const scalar_t residual_tmp39 = K_3*p_w_grad_0 + K_4*p_w_grad_1 + K_5*p_w_grad_2;
            const scalar_t residual_tmp40 = dt*residual_tmp31;
            const scalar_t residual_tmp41 = residual_tmp33*residual_tmp40;
            const scalar_t residual_tmp42 = residual_tmp34*residual_tmp40;
            const scalar_t residual_tmp43 = residual_tmp39*residual_tmp41 - residual_tmp39*residual_tmp42;
            const scalar_t residual_tmp44 = dt*residual_tmp37;
            const scalar_t residual_tmp45 = K_6*p_w_grad_0 + K_7*p_w_grad_1 + K_8*p_w_grad_2;
            const scalar_t residual_tmp46 = residual_tmp41*residual_tmp45 - residual_tmp42*residual_tmp45;
            const scalar_t residual_tmp47 = p_c*residual_tmp11;
            const scalar_t residual_tmp48 = residual_tmp17*residual_tmp47;
            const scalar_t residual_tmp49 = pow_m1(R);
            const scalar_t residual_tmp50 = pow_m1(T);
            const scalar_t residual_tmp51 = pow_m1(Z);
            const scalar_t residual_tmp52 = M_c*residual_tmp49*residual_tmp50*residual_tmp51;
            const scalar_t residual_tmp53 = residual_tmp0*residual_tmp52;
            const scalar_t residual_tmp54 = pow_m1(mu_c);
            const scalar_t residual_tmp55 = scalar_t(1) - residual_tmp8;
            const scalar_t residual_tmp56 = pow(residual_tmp55, C_ka1);
            const scalar_t residual_tmp57 = pow(residual_tmp8, C_ka2);
            const scalar_t residual_tmp58 = residual_tmp56*(residual_tmp57 + scalar_t(-1));
            const scalar_t residual_tmp59 = p_c*residual_tmp52*residual_tmp54*residual_tmp58;
            const scalar_t residual_tmp60 = p_c_direction_grad_0*residual_tmp59;
            const scalar_t residual_tmp61 = p_c_direction_grad_1*residual_tmp59;
            const scalar_t residual_tmp62 = p_c_direction_grad_2*residual_tmp59;
            const scalar_t residual_tmp63 = -K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2;
            const scalar_t residual_tmp64 = C_ka2*dt*residual_tmp47*residual_tmp56*residual_tmp57;
            const scalar_t residual_tmp65 = residual_tmp63*residual_tmp64;
            const scalar_t residual_tmp66 = dt*residual_tmp58;
            const scalar_t residual_tmp67 = residual_tmp63*residual_tmp66;
            const scalar_t residual_tmp68 = C_ka1*residual_tmp47*residual_tmp8/residual_tmp55;
            const scalar_t residual_tmp69 = residual_tmp67*residual_tmp68;
            const scalar_t residual_tmp70 = -K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2;
            const scalar_t residual_tmp71 = residual_tmp64*residual_tmp70;
            const scalar_t residual_tmp72 = residual_tmp66*residual_tmp70;
            const scalar_t residual_tmp73 = residual_tmp68*residual_tmp72;
            const scalar_t residual_tmp74 = -K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2;
            const scalar_t residual_tmp75 = residual_tmp64*residual_tmp74;
            const scalar_t residual_tmp76 = residual_tmp66*residual_tmp74;
            const scalar_t residual_tmp77 = residual_tmp68*residual_tmp76;
            const scalar_t value_coeff0 = porosity*residual_tmp13*residual_tmp2 + porosity*residual_tmp15*(-residual_tmp13 + residual_tmp14*(S_res + residual_tmp9));
            const scalar_t grad_coeff0_0 = K_0*residual_tmp27 + K_1*residual_tmp28 + K_2*residual_tmp29 - residual_tmp35*residual_tmp36 + residual_tmp38*(residual_tmp30*residual_tmp37 + residual_tmp35);
            const scalar_t grad_coeff0_1 = K_3*residual_tmp27 + K_4*residual_tmp28 + K_5*residual_tmp29 - residual_tmp36*residual_tmp43 + residual_tmp38*(residual_tmp39*residual_tmp44 + residual_tmp43);
            const scalar_t grad_coeff0_2 = K_6*residual_tmp27 + K_7*residual_tmp28 + K_8*residual_tmp29 - residual_tmp36*residual_tmp46 + residual_tmp38*(residual_tmp44*residual_tmp45 + residual_tmp46);
            const scalar_t value_coeff1 = -p_c_direction*porosity*residual_tmp53*(S_res - residual_tmp17 - residual_tmp48 + scalar_t(-1)) - p_w_direction*porosity*residual_tmp48*residual_tmp53;
            const scalar_t grad_coeff1_0 = -K_0*residual_tmp60 - K_1*residual_tmp61 - K_2*residual_tmp62 + M_c*p_c_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(residual_tmp65 + residual_tmp67 - residual_tmp69) + M_c*p_w_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(-residual_tmp65 + residual_tmp69);
            const scalar_t grad_coeff1_1 = -K_3*residual_tmp60 - K_4*residual_tmp61 - K_5*residual_tmp62 + M_c*p_c_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(residual_tmp71 + residual_tmp72 - residual_tmp73) + M_c*p_w_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(-residual_tmp71 + residual_tmp73);
            const scalar_t grad_coeff1_2 = -K_6*residual_tmp60 - K_7*residual_tmp61 - K_8*residual_tmp62 + M_c*p_c_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(residual_tmp75 + residual_tmp76 - residual_tmp77) + M_c*p_w_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(-residual_tmp75 + residual_tmp77);
            value_coeff0_values[lane] = value_coeff0;
            grad_coeff0_0_values[lane] = grad_coeff0_0;
            grad_coeff0_1_values[lane] = grad_coeff0_1;
            grad_coeff0_2_values[lane] = grad_coeff0_2;
            value_coeff1_values[lane] = value_coeff1;
            grad_coeff1_0_values[lane] = grad_coeff1_0;
            grad_coeff1_1_values[lane] = grad_coeff1_1;
            grad_coeff1_2_values[lane] = grad_coeff1_2;
        }
        for (int test = 0; test < N_SHAPE; ++test) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const ptrdiff_t geometry_offset = q * geometry_stride + lane;
                const scalar_t det = determinant[geometry_offset];
                const scalar_t test_value = shape[q * N_SHAPE + test];
                const scalar_t adj0 = adjugate[0][geometry_offset];
                const scalar_t adj1 = adjugate[1][geometry_offset];
                const scalar_t adj2 = adjugate[2][geometry_offset];
                const scalar_t adj3 = adjugate[3][geometry_offset];
                const scalar_t adj4 = adjugate[4][geometry_offset];
                const scalar_t adj5 = adjugate[5][geometry_offset];
                const scalar_t adj6 = adjugate[6][geometry_offset];
                const scalar_t adj7 = adjugate[7][geometry_offset];
                const scalar_t adj8 = adjugate[8][geometry_offset];
                const scalar_t test_grad0 = (grad_ref_x[q * N_SHAPE + test] * adj0 + grad_ref_y[q * N_SHAPE + test] * adj3 + grad_ref_z[q * N_SHAPE + test] * adj6) / det;
                const scalar_t test_grad1 = (grad_ref_x[q * N_SHAPE + test] * adj1 + grad_ref_y[q * N_SHAPE + test] * adj4 + grad_ref_z[q * N_SHAPE + test] * adj7) / det;
                const scalar_t test_grad2 = (grad_ref_x[q * N_SHAPE + test] * adj2 + grad_ref_y[q * N_SHAPE + test] * adj5 + grad_ref_z[q * N_SHAPE + test] * adj8) / det;
                output[test * N_FIELDS + 0][lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value + grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
                output[test * N_FIELDS + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void two_phase_flow_d3_simplex_tet4_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT current[2 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT direction[2 * N_SHAPE],
        const scalar_t C_ka1,
        const scalar_t C_ka2,
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
        const scalar_t M_c,
        const scalar_t P_r,
        const scalar_t R,
        const scalar_t S_res,
        const scalar_t T,
        const scalar_t Z,
        const scalar_t dt,
        const scalar_t kappa_T,
        const scalar_t m,
        const scalar_t mu_c,
        const scalar_t mu_w,
        const scalar_t p_wr,
        const scalar_t porosity,
        const scalar_t rho_w0,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
    for (int q = 0; q < N_QP; ++q) {
        scalar_t p_w_values[VECTOR_SIZE];
        scalar_t p_w_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_w_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_w_grad_2_ref_values[VECTOR_SIZE];
        scalar_t p_w_direction_values[VECTOR_SIZE];
        scalar_t p_w_direction_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_w_direction_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_w_direction_grad_2_ref_values[VECTOR_SIZE];
        scalar_t p_c_values[VECTOR_SIZE];
        scalar_t p_c_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_c_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_c_grad_2_ref_values[VECTOR_SIZE];
        scalar_t p_c_direction_values[VECTOR_SIZE];
        scalar_t p_c_direction_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_c_direction_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_c_direction_grad_2_ref_values[VECTOR_SIZE];
        scalar_t value_coeff0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_1_values[VECTOR_SIZE];
        scalar_t grad_coeff0_2_values[VECTOR_SIZE];
        scalar_t value_coeff1_values[VECTOR_SIZE];
        scalar_t grad_coeff1_0_values[VECTOR_SIZE];
        scalar_t grad_coeff1_1_values[VECTOR_SIZE];
        scalar_t grad_coeff1_2_values[VECTOR_SIZE];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 0][lane];
                p_w_values[lane] += coeff * shape[q * N_SHAPE + trial];
                p_w_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                p_w_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                p_w_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_direction_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_direction_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_direction_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_direction_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 0][lane];
                p_w_direction_values[lane] += coeff * shape[q * N_SHAPE + trial];
                p_w_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                p_w_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                p_w_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 1][lane];
                p_c_values[lane] += coeff * shape[q * N_SHAPE + trial];
                p_c_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                p_c_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                p_c_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_direction_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_direction_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_direction_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_direction_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 1][lane];
                p_c_direction_values[lane] += coeff * shape[q * N_SHAPE + trial];
                p_c_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                p_c_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                p_c_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
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
            const scalar_t p_w = p_w_values[lane];
            const scalar_t p_w_grad_0_ref = p_w_grad_0_ref_values[lane];
            const scalar_t p_w_grad_1_ref = p_w_grad_1_ref_values[lane];
            const scalar_t p_w_grad_2_ref = p_w_grad_2_ref_values[lane];
            const scalar_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
            const scalar_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
            const scalar_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
            const scalar_t p_w_direction = p_w_direction_values[lane];
            const scalar_t p_w_direction_grad_0_ref = p_w_direction_grad_0_ref_values[lane];
            const scalar_t p_w_direction_grad_1_ref = p_w_direction_grad_1_ref_values[lane];
            const scalar_t p_w_direction_grad_2_ref = p_w_direction_grad_2_ref_values[lane];
            const scalar_t p_w_direction_grad_0 = (p_w_direction_grad_0_ref * adj0 + p_w_direction_grad_1_ref * adj3 + p_w_direction_grad_2_ref * adj6) / det;
            const scalar_t p_w_direction_grad_1 = (p_w_direction_grad_0_ref * adj1 + p_w_direction_grad_1_ref * adj4 + p_w_direction_grad_2_ref * adj7) / det;
            const scalar_t p_w_direction_grad_2 = (p_w_direction_grad_0_ref * adj2 + p_w_direction_grad_1_ref * adj5 + p_w_direction_grad_2_ref * adj8) / det;
            const scalar_t p_c = p_c_values[lane];
            const scalar_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
            const scalar_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
            const scalar_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
            const scalar_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
            const scalar_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
            const scalar_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
            const scalar_t p_c_direction = p_c_direction_values[lane];
            const scalar_t p_c_direction_grad_0_ref = p_c_direction_grad_0_ref_values[lane];
            const scalar_t p_c_direction_grad_1_ref = p_c_direction_grad_1_ref_values[lane];
            const scalar_t p_c_direction_grad_2_ref = p_c_direction_grad_2_ref_values[lane];
            const scalar_t p_c_direction_grad_0 = (p_c_direction_grad_0_ref * adj0 + p_c_direction_grad_1_ref * adj3 + p_c_direction_grad_2_ref * adj6) / det;
            const scalar_t p_c_direction_grad_1 = (p_c_direction_grad_0_ref * adj1 + p_c_direction_grad_1_ref * adj4 + p_c_direction_grad_2_ref * adj7) / det;
            const scalar_t p_c_direction_grad_2 = (p_c_direction_grad_0_ref * adj2 + p_c_direction_grad_1_ref * adj5 + p_c_direction_grad_2_ref * adj8) / det;
            const scalar_t residual_tmp0 = pow_m1(dt);
            const scalar_t residual_tmp1 = residual_tmp0*rho_w0;
            const scalar_t residual_tmp2 = p_c_direction*residual_tmp1;
            const scalar_t residual_tmp3 = S_res + scalar_t(-1);
            const scalar_t residual_tmp4 = p_c - p_w;
            const scalar_t residual_tmp5 = pow(residual_tmp4/P_r, m);
            const scalar_t residual_tmp6 = residual_tmp5 + scalar_t(1);
            const scalar_t residual_tmp7 = scalar_t(1) - m;
            const scalar_t residual_tmp8 = pow(residual_tmp6, residual_tmp7/m);
            const scalar_t residual_tmp9 = -residual_tmp3*residual_tmp8;
            const scalar_t residual_tmp10 = exp(kappa_T*(p_w - p_wr));
            const scalar_t residual_tmp11 = residual_tmp5*residual_tmp7/(residual_tmp4*residual_tmp6);
            const scalar_t residual_tmp12 = residual_tmp10*residual_tmp11;
            const scalar_t residual_tmp13 = residual_tmp12*residual_tmp9;
            const scalar_t residual_tmp14 = kappa_T*residual_tmp10;
            const scalar_t residual_tmp15 = p_w_direction*residual_tmp1;
            const scalar_t residual_tmp16 = pow_m1(mu_w);
            const scalar_t residual_tmp17 = residual_tmp3*residual_tmp8;
            const scalar_t residual_tmp18 = S_res - residual_tmp17;
            const scalar_t residual_tmp19 = pow(residual_tmp18, pow_m1(C_kw1));
            const scalar_t residual_tmp20 = scalar_t(1) - residual_tmp19;
            const scalar_t residual_tmp21 = pow(residual_tmp20, C_kw1);
            const scalar_t residual_tmp22 = residual_tmp21 + scalar_t(-1);
            const scalar_t residual_tmp23 = pow_2(residual_tmp22);
            const scalar_t residual_tmp24 = sqrt(residual_tmp18);
            const scalar_t residual_tmp25 = residual_tmp23*residual_tmp24;
            const scalar_t residual_tmp26 = residual_tmp10*residual_tmp16*residual_tmp25*rho_w0;
            const scalar_t residual_tmp27 = p_w_direction_grad_0*residual_tmp26;
            const scalar_t residual_tmp28 = p_w_direction_grad_1*residual_tmp26;
            const scalar_t residual_tmp29 = p_w_direction_grad_2*residual_tmp26;
            const scalar_t residual_tmp30 = dt*(K_0*p_w_grad_0 + K_1*p_w_grad_1 + K_2*p_w_grad_2);
            const scalar_t residual_tmp31 = residual_tmp12*residual_tmp17/residual_tmp24;
            const scalar_t residual_tmp32 = residual_tmp30*residual_tmp31;
            const scalar_t residual_tmp33 = ((scalar_t(1) / scalar_t(2)))*residual_tmp23;
            const scalar_t residual_tmp34 = scalar_t(2)*residual_tmp19*residual_tmp21*residual_tmp22/residual_tmp20;
            const scalar_t residual_tmp35 = residual_tmp32*residual_tmp33 - residual_tmp32*residual_tmp34;
            const scalar_t residual_tmp36 = residual_tmp16*residual_tmp2;
            const scalar_t residual_tmp37 = residual_tmp14*residual_tmp25;
            const scalar_t residual_tmp38 = residual_tmp15*residual_tmp16;
            const scalar_t residual_tmp39 = K_3*p_w_grad_0 + K_4*p_w_grad_1 + K_5*p_w_grad_2;
            const scalar_t residual_tmp40 = dt*residual_tmp31;
            const scalar_t residual_tmp41 = residual_tmp33*residual_tmp40;
            const scalar_t residual_tmp42 = residual_tmp34*residual_tmp40;
            const scalar_t residual_tmp43 = residual_tmp39*residual_tmp41 - residual_tmp39*residual_tmp42;
            const scalar_t residual_tmp44 = dt*residual_tmp37;
            const scalar_t residual_tmp45 = K_6*p_w_grad_0 + K_7*p_w_grad_1 + K_8*p_w_grad_2;
            const scalar_t residual_tmp46 = residual_tmp41*residual_tmp45 - residual_tmp42*residual_tmp45;
            const scalar_t residual_tmp47 = p_c*residual_tmp11;
            const scalar_t residual_tmp48 = residual_tmp17*residual_tmp47;
            const scalar_t residual_tmp49 = pow_m1(R);
            const scalar_t residual_tmp50 = pow_m1(T);
            const scalar_t residual_tmp51 = pow_m1(Z);
            const scalar_t residual_tmp52 = M_c*residual_tmp49*residual_tmp50*residual_tmp51;
            const scalar_t residual_tmp53 = residual_tmp0*residual_tmp52;
            const scalar_t residual_tmp54 = pow_m1(mu_c);
            const scalar_t residual_tmp55 = scalar_t(1) - residual_tmp8;
            const scalar_t residual_tmp56 = pow(residual_tmp55, C_ka1);
            const scalar_t residual_tmp57 = pow(residual_tmp8, C_ka2);
            const scalar_t residual_tmp58 = residual_tmp56*(residual_tmp57 + scalar_t(-1));
            const scalar_t residual_tmp59 = p_c*residual_tmp52*residual_tmp54*residual_tmp58;
            const scalar_t residual_tmp60 = p_c_direction_grad_0*residual_tmp59;
            const scalar_t residual_tmp61 = p_c_direction_grad_1*residual_tmp59;
            const scalar_t residual_tmp62 = p_c_direction_grad_2*residual_tmp59;
            const scalar_t residual_tmp63 = -K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2;
            const scalar_t residual_tmp64 = C_ka2*dt*residual_tmp47*residual_tmp56*residual_tmp57;
            const scalar_t residual_tmp65 = residual_tmp63*residual_tmp64;
            const scalar_t residual_tmp66 = dt*residual_tmp58;
            const scalar_t residual_tmp67 = residual_tmp63*residual_tmp66;
            const scalar_t residual_tmp68 = C_ka1*residual_tmp47*residual_tmp8/residual_tmp55;
            const scalar_t residual_tmp69 = residual_tmp67*residual_tmp68;
            const scalar_t residual_tmp70 = -K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2;
            const scalar_t residual_tmp71 = residual_tmp64*residual_tmp70;
            const scalar_t residual_tmp72 = residual_tmp66*residual_tmp70;
            const scalar_t residual_tmp73 = residual_tmp68*residual_tmp72;
            const scalar_t residual_tmp74 = -K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2;
            const scalar_t residual_tmp75 = residual_tmp64*residual_tmp74;
            const scalar_t residual_tmp76 = residual_tmp66*residual_tmp74;
            const scalar_t residual_tmp77 = residual_tmp68*residual_tmp76;
            const scalar_t value_coeff0 = porosity*residual_tmp13*residual_tmp2 + porosity*residual_tmp15*(-residual_tmp13 + residual_tmp14*(S_res + residual_tmp9));
            const scalar_t grad_coeff0_0 = K_0*residual_tmp27 + K_1*residual_tmp28 + K_2*residual_tmp29 - residual_tmp35*residual_tmp36 + residual_tmp38*(residual_tmp30*residual_tmp37 + residual_tmp35);
            const scalar_t grad_coeff0_1 = K_3*residual_tmp27 + K_4*residual_tmp28 + K_5*residual_tmp29 - residual_tmp36*residual_tmp43 + residual_tmp38*(residual_tmp39*residual_tmp44 + residual_tmp43);
            const scalar_t grad_coeff0_2 = K_6*residual_tmp27 + K_7*residual_tmp28 + K_8*residual_tmp29 - residual_tmp36*residual_tmp46 + residual_tmp38*(residual_tmp44*residual_tmp45 + residual_tmp46);
            const scalar_t value_coeff1 = -p_c_direction*porosity*residual_tmp53*(S_res - residual_tmp17 - residual_tmp48 + scalar_t(-1)) - p_w_direction*porosity*residual_tmp48*residual_tmp53;
            const scalar_t grad_coeff1_0 = -K_0*residual_tmp60 - K_1*residual_tmp61 - K_2*residual_tmp62 + M_c*p_c_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(residual_tmp65 + residual_tmp67 - residual_tmp69) + M_c*p_w_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(-residual_tmp65 + residual_tmp69);
            const scalar_t grad_coeff1_1 = -K_3*residual_tmp60 - K_4*residual_tmp61 - K_5*residual_tmp62 + M_c*p_c_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(residual_tmp71 + residual_tmp72 - residual_tmp73) + M_c*p_w_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(-residual_tmp71 + residual_tmp73);
            const scalar_t grad_coeff1_2 = -K_6*residual_tmp60 - K_7*residual_tmp61 - K_8*residual_tmp62 + M_c*p_c_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(residual_tmp75 + residual_tmp76 - residual_tmp77) + M_c*p_w_direction*residual_tmp0*residual_tmp49*residual_tmp50*residual_tmp51*residual_tmp54*(-residual_tmp75 + residual_tmp77);
            value_coeff0_values[lane] = value_coeff0;
            grad_coeff0_0_values[lane] = grad_coeff0_0;
            grad_coeff0_1_values[lane] = grad_coeff0_1;
            grad_coeff0_2_values[lane] = grad_coeff0_2;
            value_coeff1_values[lane] = value_coeff1;
            grad_coeff1_0_values[lane] = grad_coeff1_0;
            grad_coeff1_1_values[lane] = grad_coeff1_1;
            grad_coeff1_2_values[lane] = grad_coeff1_2;
        }
        for (int test = 0; test < N_SHAPE; ++test) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const ptrdiff_t geometry_offset = q * geometry_stride + lane;
                const scalar_t det = determinant[geometry_offset];
                const scalar_t test_value = shape[q * N_SHAPE + test];
                const scalar_t adj0 = adjugate[0][geometry_offset];
                const scalar_t adj1 = adjugate[1][geometry_offset];
                const scalar_t adj2 = adjugate[2][geometry_offset];
                const scalar_t adj3 = adjugate[3][geometry_offset];
                const scalar_t adj4 = adjugate[4][geometry_offset];
                const scalar_t adj5 = adjugate[5][geometry_offset];
                const scalar_t adj6 = adjugate[6][geometry_offset];
                const scalar_t adj7 = adjugate[7][geometry_offset];
                const scalar_t adj8 = adjugate[8][geometry_offset];
                const scalar_t test_grad0 = (grad_ref_x[q * N_SHAPE + test] * adj0 + grad_ref_y[q * N_SHAPE + test] * adj3 + grad_ref_z[q * N_SHAPE + test] * adj6) / det;
                const scalar_t test_grad1 = (grad_ref_x[q * N_SHAPE + test] * adj1 + grad_ref_y[q * N_SHAPE + test] * adj4 + grad_ref_z[q * N_SHAPE + test] * adj7) / det;
                const scalar_t test_grad2 = (grad_ref_x[q * N_SHAPE + test] * adj2 + grad_ref_y[q * N_SHAPE + test] * adj5 + grad_ref_z[q * N_SHAPE + test] * adj8) / det;
                output[test * N_FIELDS + 0][lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value + grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
                output[test * N_FIELDS + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
            }
        }
    }
}

} // namespace codegen
} // namespace sfem

#endif
