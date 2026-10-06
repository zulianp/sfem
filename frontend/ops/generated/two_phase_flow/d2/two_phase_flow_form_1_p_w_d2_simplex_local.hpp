#ifndef TWO_PHASE_FLOW_FORM_1_P_W_D2_SIMPLEX_LOCAL_HPP
#define TWO_PHASE_FLOW_FORM_1_P_W_D2_SIMPLEX_LOCAL_HPP

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
static SFEM_INLINE void two_phase_flow_form_1_p_w_d2_simplex_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT current[2 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT previous[2 * N_SHAPE],
        const scalar_t C_kw1,
        const scalar_t K_0,
        const scalar_t K_1,
        const scalar_t K_2,
        const scalar_t K_3,
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
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
    for (int q = 0; q < N_QP; ++q) {
        scalar_t p_w_values[VECTOR_SIZE];
        scalar_t p_w_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_w_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_w_old_values[VECTOR_SIZE];
        scalar_t p_c_values[VECTOR_SIZE];
        scalar_t p_c_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_c_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_c_old_values[VECTOR_SIZE];
        scalar_t value_coeff0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_1_values[VECTOR_SIZE];
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
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 0][lane];
                p_w_values[lane] += coeff * shape[q * N_SHAPE + trial];
                p_w_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                p_w_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
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
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 1][lane];
                p_c_values[lane] += coeff * shape[q * N_SHAPE + trial];
                p_c_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                p_c_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
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
            const scalar_t p_w = p_w_values[lane];
            const scalar_t p_w_grad_0_ref = p_w_grad_0_ref_values[lane];
            const scalar_t p_w_grad_1_ref = p_w_grad_1_ref_values[lane];
            const scalar_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj2) / det;
            const scalar_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj3) / det;
            const scalar_t p_w_old = p_w_old_values[lane];
            const scalar_t p_c = p_c_values[lane];
            const scalar_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
            const scalar_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
            const scalar_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj2) / det;
            const scalar_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj3) / det;
            const scalar_t p_c_old = p_c_old_values[lane];
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
            const scalar_t grad_coeff0_0 = residual_tmp8*(K_0*p_w_grad_0 + K_1*p_w_grad_1);
            const scalar_t grad_coeff0_1 = residual_tmp8*(K_2*p_w_grad_0 + K_3*p_w_grad_1);
            value_coeff0_values[lane] = value_coeff0;
            grad_coeff0_0_values[lane] = grad_coeff0_0;
            grad_coeff0_1_values[lane] = grad_coeff0_1;
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
                const scalar_t test_grad0 = (grad_ref_x[q * N_SHAPE + test] * adj0 + grad_ref_y[q * N_SHAPE + test] * adj2) / det;
                const scalar_t test_grad1 = (grad_ref_x[q * N_SHAPE + test] * adj1 + grad_ref_y[q * N_SHAPE + test] * adj3) / det;
                output[test * N_FIELDS + 0][lane] += q_weight[q] * det * (value_coeff0_values[lane] * test_value + grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void two_phase_flow_form_1_p_w_d2_simplex_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT q_weight,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t det = determinant[geometry_offset];
        }
        for (int test = 0; test < N_SHAPE; ++test) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const ptrdiff_t geometry_offset = q * geometry_stride + lane;
                const scalar_t det = determinant[geometry_offset];
                const scalar_t test_value = shape[q * N_SHAPE + test];
            }
        }
    }
}

} // namespace codegen
} // namespace sfem

#endif
