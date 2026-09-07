#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_VISCOUS_D2_SIMPLEX_LOCAL_HPP
#define MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_VISCOUS_D2_SIMPLEX_LOCAL_HPP

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

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d2_simplex_residual_block(
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
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
    for (int q = 0; q < N_QP; ++q) {
        scalar_t u0_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t grad_coeff0_0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_1_values[VECTOR_SIZE];
        scalar_t grad_coeff1_0_values[VECTOR_SIZE];
        scalar_t grad_coeff1_1_values[VECTOR_SIZE];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 0][lane];
                u0_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_old_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_old_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 0][lane];
                u0_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 1][lane];
                u1_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_old_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_old_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 1][lane];
                u1_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
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
            const scalar_t u0_grad_0_ref = u0_grad_0_ref_values[lane];
            const scalar_t u0_grad_1_ref = u0_grad_1_ref_values[lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
            const scalar_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[lane];
            const scalar_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
            const scalar_t u1_grad_0_ref = u1_grad_0_ref_values[lane];
            const scalar_t u1_grad_1_ref = u1_grad_1_ref_values[lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
            const scalar_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[lane];
            const scalar_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
            const scalar_t residual_tmp0 = u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp1 = pow_m1(residual_tmp0 + u0_grad_0*u1_grad_1 + u0_grad_0 - u0_grad_1*u1_grad_0);
            const scalar_t residual_tmp2 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp3 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp4 = u0_grad_0 + scalar_t(1);
            const scalar_t residual_tmp5 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
            const scalar_t residual_tmp6 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
            const scalar_t residual_tmp7 = eta_s*(-residual_tmp0*residual_tmp6 + residual_tmp2*u0_grad_1 + residual_tmp3*u1_grad_0 - residual_tmp4*residual_tmp5);
            const scalar_t residual_tmp8 = residual_tmp5*u1_grad_0;
            const scalar_t residual_tmp9 = residual_tmp0*residual_tmp2;
            const scalar_t residual_tmp10 = -residual_tmp3*residual_tmp4 + residual_tmp6*u0_grad_1;
            const scalar_t residual_tmp11 = eta_b*(residual_tmp10 + residual_tmp8 - residual_tmp9);
            const scalar_t residual_tmp12 = residual_tmp10 - residual_tmp8 + residual_tmp9;
            const scalar_t residual_tmp13 = -eta_s*residual_tmp12 + residual_tmp11;
            const scalar_t residual_tmp14 = eta_s*residual_tmp12 + residual_tmp11;
            const scalar_t grad_coeff0_0 = residual_tmp1*(-residual_tmp0*residual_tmp13 + residual_tmp7*u0_grad_1);
            const scalar_t grad_coeff0_1 = residual_tmp1*(residual_tmp13*u1_grad_0 - residual_tmp4*residual_tmp7);
            const scalar_t grad_coeff1_0 = residual_tmp1*(-residual_tmp0*residual_tmp7 + residual_tmp14*u0_grad_1);
            const scalar_t grad_coeff1_1 = residual_tmp1*(-residual_tmp14*residual_tmp4 + residual_tmp7*u1_grad_0);
            grad_coeff0_0_values[lane] = grad_coeff0_0;
            grad_coeff0_1_values[lane] = grad_coeff0_1;
            grad_coeff1_0_values[lane] = grad_coeff1_0;
            grad_coeff1_1_values[lane] = grad_coeff1_1;
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
                output[test * N_FIELDS + 0][lane] += q_weight[q] * det * (grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1);
                output[test * N_FIELDS + 1][lane] += q_weight[q] * det * (grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d2_simplex_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t current[2 * N_SHAPE][VECTOR_SIZE],
        const scalar_t previous[2 * N_SHAPE][VECTOR_SIZE],
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t output[2 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
    for (int q = 0; q < N_QP; ++q) {
        scalar_t u0_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t grad_coeff0_0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_1_values[VECTOR_SIZE];
        scalar_t grad_coeff1_0_values[VECTOR_SIZE];
        scalar_t grad_coeff1_1_values[VECTOR_SIZE];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 0][lane];
                u0_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_old_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_old_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 0][lane];
                u0_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 1][lane];
                u1_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_old_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_old_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 1][lane];
                u1_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
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
            const scalar_t u0_grad_0_ref = u0_grad_0_ref_values[lane];
            const scalar_t u0_grad_1_ref = u0_grad_1_ref_values[lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
            const scalar_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[lane];
            const scalar_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
            const scalar_t u1_grad_0_ref = u1_grad_0_ref_values[lane];
            const scalar_t u1_grad_1_ref = u1_grad_1_ref_values[lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
            const scalar_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[lane];
            const scalar_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
            const scalar_t residual_tmp0 = u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp1 = pow_m1(residual_tmp0 + u0_grad_0*u1_grad_1 + u0_grad_0 - u0_grad_1*u1_grad_0);
            const scalar_t residual_tmp2 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp3 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp4 = u0_grad_0 + scalar_t(1);
            const scalar_t residual_tmp5 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
            const scalar_t residual_tmp6 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
            const scalar_t residual_tmp7 = eta_s*(-residual_tmp0*residual_tmp6 + residual_tmp2*u0_grad_1 + residual_tmp3*u1_grad_0 - residual_tmp4*residual_tmp5);
            const scalar_t residual_tmp8 = residual_tmp5*u1_grad_0;
            const scalar_t residual_tmp9 = residual_tmp0*residual_tmp2;
            const scalar_t residual_tmp10 = -residual_tmp3*residual_tmp4 + residual_tmp6*u0_grad_1;
            const scalar_t residual_tmp11 = eta_b*(residual_tmp10 + residual_tmp8 - residual_tmp9);
            const scalar_t residual_tmp12 = residual_tmp10 - residual_tmp8 + residual_tmp9;
            const scalar_t residual_tmp13 = -eta_s*residual_tmp12 + residual_tmp11;
            const scalar_t residual_tmp14 = eta_s*residual_tmp12 + residual_tmp11;
            const scalar_t grad_coeff0_0 = residual_tmp1*(-residual_tmp0*residual_tmp13 + residual_tmp7*u0_grad_1);
            const scalar_t grad_coeff0_1 = residual_tmp1*(residual_tmp13*u1_grad_0 - residual_tmp4*residual_tmp7);
            const scalar_t grad_coeff1_0 = residual_tmp1*(-residual_tmp0*residual_tmp7 + residual_tmp14*u0_grad_1);
            const scalar_t grad_coeff1_1 = residual_tmp1*(-residual_tmp14*residual_tmp4 + residual_tmp7*u1_grad_0);
            grad_coeff0_0_values[lane] = grad_coeff0_0;
            grad_coeff0_1_values[lane] = grad_coeff0_1;
            grad_coeff1_0_values[lane] = grad_coeff1_0;
            grad_coeff1_1_values[lane] = grad_coeff1_1;
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
                output[test * N_FIELDS + 0][lane] += q_weight[q] * det * (grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1);
                output[test * N_FIELDS + 1][lane] += q_weight[q] * det * (grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d2_simplex_tri3_residual_block(
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
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t det = determinant[geometry_offset];
            const scalar_t adj0 = adjugate[0][geometry_offset];
            const scalar_t adj1 = adjugate[1][geometry_offset];
            const scalar_t adj2 = adjugate[2][geometry_offset];
            const scalar_t adj3 = adjugate[3][geometry_offset];
            const scalar_t u0_grad_0_ref = -(current[0][lane]) + current[2][lane];
            const scalar_t u0_grad_1_ref = -(current[0][lane]) + current[4][lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
            const scalar_t u0_old_grad_0_ref = -(previous[0][lane]) + previous[2][lane];
            const scalar_t u0_old_grad_1_ref = -(previous[0][lane]) + previous[4][lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
            const scalar_t u1_grad_0_ref = -(current[1][lane]) + current[3][lane];
            const scalar_t u1_grad_1_ref = -(current[1][lane]) + current[5][lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
            const scalar_t u1_old_grad_0_ref = -(previous[1][lane]) + previous[3][lane];
            const scalar_t u1_old_grad_1_ref = -(previous[1][lane]) + previous[5][lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
            const scalar_t residual_tmp0 = u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp1 = pow_m1(residual_tmp0 + u0_grad_0*u1_grad_1 + u0_grad_0 - u0_grad_1*u1_grad_0);
            const scalar_t residual_tmp2 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp3 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp4 = u0_grad_0 + scalar_t(1);
            const scalar_t residual_tmp5 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
            const scalar_t residual_tmp6 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
            const scalar_t residual_tmp7 = eta_s*(-residual_tmp0*residual_tmp6 + residual_tmp2*u0_grad_1 + residual_tmp3*u1_grad_0 - residual_tmp4*residual_tmp5);
            const scalar_t residual_tmp8 = residual_tmp5*u1_grad_0;
            const scalar_t residual_tmp9 = residual_tmp0*residual_tmp2;
            const scalar_t residual_tmp10 = -residual_tmp3*residual_tmp4 + residual_tmp6*u0_grad_1;
            const scalar_t residual_tmp11 = eta_b*(residual_tmp10 + residual_tmp8 - residual_tmp9);
            const scalar_t residual_tmp12 = residual_tmp10 - residual_tmp8 + residual_tmp9;
            const scalar_t residual_tmp13 = -eta_s*residual_tmp12 + residual_tmp11;
            const scalar_t residual_tmp14 = eta_s*residual_tmp12 + residual_tmp11;
            const scalar_t grad_coeff0_0 = residual_tmp1*(-residual_tmp0*residual_tmp13 + residual_tmp7*u0_grad_1);
            const scalar_t grad_coeff0_1 = residual_tmp1*(residual_tmp13*u1_grad_0 - residual_tmp4*residual_tmp7);
            const scalar_t grad_coeff1_0 = residual_tmp1*(-residual_tmp0*residual_tmp7 + residual_tmp14*u0_grad_1);
            const scalar_t grad_coeff1_1 = residual_tmp1*(-residual_tmp14*residual_tmp4 + residual_tmp7*u1_grad_0);
            const scalar_t grad_coeff0_0_value = grad_coeff0_0;
            const scalar_t grad_coeff0_1_value = grad_coeff0_1;
            const scalar_t grad_coeff1_0_value = grad_coeff1_0;
            const scalar_t grad_coeff1_1_value = grad_coeff1_1;
            const scalar_t test0_grad0 = (-(adj0) - adj2) / det;
            const scalar_t test0_grad1 = (-(adj1) - adj3) / det;
            const scalar_t test1_grad0 = (adj0) / det;
            const scalar_t test1_grad1 = (adj1) / det;
            const scalar_t test2_grad0 = (adj2) / det;
            const scalar_t test2_grad1 = (adj3) / det;
            output[0][lane] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1);
            output[1][lane] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1);
            output[2][lane] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1);
            output[3][lane] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1);
            output[4][lane] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1);
            output[5][lane] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1);
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d2_simplex_tri3_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t current[2 * N_SHAPE][VECTOR_SIZE],
        const scalar_t previous[2 * N_SHAPE][VECTOR_SIZE],
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t output[2 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t det = determinant[geometry_offset];
            const scalar_t adj0 = adjugate[0][geometry_offset];
            const scalar_t adj1 = adjugate[1][geometry_offset];
            const scalar_t adj2 = adjugate[2][geometry_offset];
            const scalar_t adj3 = adjugate[3][geometry_offset];
            const scalar_t u0_grad_0_ref = -(current[0][lane]) + current[2][lane];
            const scalar_t u0_grad_1_ref = -(current[0][lane]) + current[4][lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
            const scalar_t u0_old_grad_0_ref = -(previous[0][lane]) + previous[2][lane];
            const scalar_t u0_old_grad_1_ref = -(previous[0][lane]) + previous[4][lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
            const scalar_t u1_grad_0_ref = -(current[1][lane]) + current[3][lane];
            const scalar_t u1_grad_1_ref = -(current[1][lane]) + current[5][lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
            const scalar_t u1_old_grad_0_ref = -(previous[1][lane]) + previous[3][lane];
            const scalar_t u1_old_grad_1_ref = -(previous[1][lane]) + previous[5][lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
            const scalar_t residual_tmp0 = u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp1 = pow_m1(residual_tmp0 + u0_grad_0*u1_grad_1 + u0_grad_0 - u0_grad_1*u1_grad_0);
            const scalar_t residual_tmp2 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp3 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp4 = u0_grad_0 + scalar_t(1);
            const scalar_t residual_tmp5 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
            const scalar_t residual_tmp6 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
            const scalar_t residual_tmp7 = eta_s*(-residual_tmp0*residual_tmp6 + residual_tmp2*u0_grad_1 + residual_tmp3*u1_grad_0 - residual_tmp4*residual_tmp5);
            const scalar_t residual_tmp8 = residual_tmp5*u1_grad_0;
            const scalar_t residual_tmp9 = residual_tmp0*residual_tmp2;
            const scalar_t residual_tmp10 = -residual_tmp3*residual_tmp4 + residual_tmp6*u0_grad_1;
            const scalar_t residual_tmp11 = eta_b*(residual_tmp10 + residual_tmp8 - residual_tmp9);
            const scalar_t residual_tmp12 = residual_tmp10 - residual_tmp8 + residual_tmp9;
            const scalar_t residual_tmp13 = -eta_s*residual_tmp12 + residual_tmp11;
            const scalar_t residual_tmp14 = eta_s*residual_tmp12 + residual_tmp11;
            const scalar_t grad_coeff0_0 = residual_tmp1*(-residual_tmp0*residual_tmp13 + residual_tmp7*u0_grad_1);
            const scalar_t grad_coeff0_1 = residual_tmp1*(residual_tmp13*u1_grad_0 - residual_tmp4*residual_tmp7);
            const scalar_t grad_coeff1_0 = residual_tmp1*(-residual_tmp0*residual_tmp7 + residual_tmp14*u0_grad_1);
            const scalar_t grad_coeff1_1 = residual_tmp1*(-residual_tmp14*residual_tmp4 + residual_tmp7*u1_grad_0);
            const scalar_t grad_coeff0_0_value = grad_coeff0_0;
            const scalar_t grad_coeff0_1_value = grad_coeff0_1;
            const scalar_t grad_coeff1_0_value = grad_coeff1_0;
            const scalar_t grad_coeff1_1_value = grad_coeff1_1;
            const scalar_t test0_grad0 = (-(adj0) - adj2) / det;
            const scalar_t test0_grad1 = (-(adj1) - adj3) / det;
            const scalar_t test1_grad0 = (adj0) / det;
            const scalar_t test1_grad1 = (adj1) / det;
            const scalar_t test2_grad0 = (adj2) / det;
            const scalar_t test2_grad1 = (adj3) / det;
            output[0][lane] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1);
            output[1][lane] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1);
            output[2][lane] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1);
            output[3][lane] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1);
            output[4][lane] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1);
            output[5][lane] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1);
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d2_simplex_jacobian_action_block(
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
        const scalar_t *const SFEM_RESTRICT direction[2 * N_SHAPE],
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
    for (int q = 0; q < N_QP; ++q) {
        scalar_t u0_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u0_direction_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_direction_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_direction_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_direction_grad_1_ref_values[VECTOR_SIZE];
        scalar_t grad_coeff0_0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_1_values[VECTOR_SIZE];
        scalar_t grad_coeff1_0_values[VECTOR_SIZE];
        scalar_t grad_coeff1_1_values[VECTOR_SIZE];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 0][lane];
                u0_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_old_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_old_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 0][lane];
                u0_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_direction_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_direction_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 0][lane];
                u0_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 1][lane];
                u1_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_old_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_old_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 1][lane];
                u1_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_direction_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_direction_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 1][lane];
                u1_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
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
            const scalar_t u0_grad_0_ref = u0_grad_0_ref_values[lane];
            const scalar_t u0_grad_1_ref = u0_grad_1_ref_values[lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
            const scalar_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[lane];
            const scalar_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
            const scalar_t u0_direction_grad_0_ref = u0_direction_grad_0_ref_values[lane];
            const scalar_t u0_direction_grad_1_ref = u0_direction_grad_1_ref_values[lane];
            const scalar_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
            const scalar_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
            const scalar_t u1_grad_0_ref = u1_grad_0_ref_values[lane];
            const scalar_t u1_grad_1_ref = u1_grad_1_ref_values[lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
            const scalar_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[lane];
            const scalar_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
            const scalar_t u1_direction_grad_0_ref = u1_direction_grad_0_ref_values[lane];
            const scalar_t u1_direction_grad_1_ref = u1_direction_grad_1_ref_values[lane];
            const scalar_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj2) / det;
            const scalar_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj3) / det;
            const scalar_t residual_tmp0 = u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp1 = residual_tmp0 + u0_grad_0*u1_grad_1 + u0_grad_0 - u0_grad_1*u1_grad_0;
            const scalar_t residual_tmp2 = pow_m1(residual_tmp1);
            const scalar_t residual_tmp3 = newmark_velocity_alpha*residual_tmp0;
            const scalar_t residual_tmp4 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp5 = -residual_tmp3 + residual_tmp4;
            const scalar_t residual_tmp6 = eta_s*residual_tmp5;
            const scalar_t residual_tmp7 = eta_s*u0_old_grad_1;
            const scalar_t residual_tmp8 = newmark_velocity_alpha*u0_grad_1;
            const scalar_t residual_tmp9 = eta_b*(scalar_t(2)*residual_tmp8 + u0_old_grad_1);
            const scalar_t residual_tmp10 = residual_tmp7 + residual_tmp9;
            const scalar_t residual_tmp11 = pow_m2(residual_tmp1);
            const scalar_t residual_tmp12 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp13 = u0_grad_0 + scalar_t(1);
            const scalar_t residual_tmp14 = residual_tmp8 + u0_old_grad_1;
            const scalar_t residual_tmp15 = newmark_velocity_alpha*u1_grad_0;
            const scalar_t residual_tmp16 = residual_tmp15 + u1_old_grad_0;
            const scalar_t residual_tmp17 = eta_s*(-residual_tmp0*residual_tmp16 + residual_tmp12*u0_grad_1 - residual_tmp13*residual_tmp14 + residual_tmp4*u1_grad_0);
            const scalar_t residual_tmp18 = residual_tmp14*u1_grad_0;
            const scalar_t residual_tmp19 = residual_tmp0*residual_tmp12;
            const scalar_t residual_tmp20 = -residual_tmp13*residual_tmp4 + residual_tmp16*u0_grad_1;
            const scalar_t residual_tmp21 = eta_b*(residual_tmp18 - residual_tmp19 + residual_tmp20);
            const scalar_t residual_tmp22 = eta_s*(-residual_tmp18 + residual_tmp19 + residual_tmp20);
            const scalar_t residual_tmp23 = residual_tmp21 - residual_tmp22;
            const scalar_t residual_tmp24 = residual_tmp11*(-residual_tmp0*residual_tmp23 + residual_tmp17*u0_grad_1);
            const scalar_t residual_tmp25 = eta_b*(-residual_tmp3 - residual_tmp4);
            const scalar_t residual_tmp26 = -residual_tmp5;
            const scalar_t residual_tmp27 = -eta_s*residual_tmp26 + residual_tmp25;
            const scalar_t residual_tmp28 = -residual_tmp0;
            const scalar_t residual_tmp29 = newmark_velocity_alpha*residual_tmp13;
            const scalar_t residual_tmp30 = residual_tmp12 - residual_tmp29;
            const scalar_t residual_tmp31 = eta_s*residual_tmp30;
            const scalar_t residual_tmp32 = eta_b*(scalar_t(2)*residual_tmp15 + u1_old_grad_0);
            const scalar_t residual_tmp33 = -eta_s*u1_old_grad_0 + residual_tmp32;
            const scalar_t residual_tmp34 = eta_b*(-residual_tmp12 - residual_tmp29);
            const scalar_t residual_tmp35 = -eta_s*residual_tmp30 + residual_tmp34;
            const scalar_t residual_tmp36 = eta_s*u1_old_grad_0;
            const scalar_t residual_tmp37 = -residual_tmp13;
            const scalar_t residual_tmp38 = residual_tmp11*(-residual_tmp13*residual_tmp17 + residual_tmp23*u1_grad_0);
            const scalar_t residual_tmp39 = -residual_tmp17;
            const scalar_t residual_tmp40 = -residual_tmp7 + residual_tmp9;
            const scalar_t residual_tmp41 = residual_tmp21 + residual_tmp22;
            const scalar_t residual_tmp42 = residual_tmp11*(-residual_tmp0*residual_tmp17 + residual_tmp41*u0_grad_1);
            const scalar_t residual_tmp43 = eta_s*residual_tmp26 + residual_tmp25;
            const scalar_t residual_tmp44 = residual_tmp31 + residual_tmp34;
            const scalar_t residual_tmp45 = residual_tmp32 + residual_tmp36;
            const scalar_t residual_tmp46 = residual_tmp11*(-residual_tmp13*residual_tmp41 + residual_tmp17*u1_grad_0);
            const scalar_t grad_coeff0_0 = u0_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp27 - residual_tmp7*u0_grad_1) + residual_tmp24*residual_tmp28) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp33 + residual_tmp17 + residual_tmp31*u0_grad_1) + residual_tmp24*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp10 + residual_tmp6*u0_grad_1) + residual_tmp24*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp35 - residual_tmp23 - residual_tmp36*u0_grad_1) + residual_tmp24*residual_tmp37);
            const scalar_t grad_coeff0_1 = u0_direction_grad_0*(residual_tmp2*(residual_tmp13*residual_tmp7 + residual_tmp27*u1_grad_0 + residual_tmp39) + residual_tmp28*residual_tmp38) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp31 + residual_tmp33*u1_grad_0) + residual_tmp38*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(residual_tmp10*u1_grad_0 - residual_tmp13*residual_tmp6 + residual_tmp23) + residual_tmp38*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(residual_tmp13*residual_tmp36 + residual_tmp35*u1_grad_0) + residual_tmp37*residual_tmp38);
            const scalar_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp2*(residual_tmp0*residual_tmp7 + residual_tmp43*u0_grad_1) + residual_tmp28*residual_tmp42) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp31 + residual_tmp41 + residual_tmp45*u0_grad_1) + residual_tmp42*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp6 + residual_tmp40*u0_grad_1) + residual_tmp42*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(residual_tmp0*residual_tmp36 + residual_tmp39 + residual_tmp44*u0_grad_1) + residual_tmp37*residual_tmp42);
            const scalar_t grad_coeff1_1 = u0_direction_grad_0*(residual_tmp2*(-residual_tmp13*residual_tmp43 - residual_tmp41 - residual_tmp7*u1_grad_0) + residual_tmp28*residual_tmp46) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp45 + residual_tmp31*u1_grad_0) + residual_tmp46*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp13*residual_tmp40 + residual_tmp17 + residual_tmp6*u1_grad_0) + residual_tmp46*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp44 - residual_tmp36*u1_grad_0) + residual_tmp37*residual_tmp46);
            grad_coeff0_0_values[lane] = grad_coeff0_0;
            grad_coeff0_1_values[lane] = grad_coeff0_1;
            grad_coeff1_0_values[lane] = grad_coeff1_0;
            grad_coeff1_1_values[lane] = grad_coeff1_1;
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
                output[test * N_FIELDS + 0][lane] += q_weight[q] * det * (grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1);
                output[test * N_FIELDS + 1][lane] += q_weight[q] * det * (grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d2_simplex_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t current[2 * N_SHAPE][VECTOR_SIZE],
        const scalar_t previous[2 * N_SHAPE][VECTOR_SIZE],
        const scalar_t direction[2 * N_SHAPE][VECTOR_SIZE],
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t output[2 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
    for (int q = 0; q < N_QP; ++q) {
        scalar_t u0_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u0_direction_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_direction_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_direction_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_direction_grad_1_ref_values[VECTOR_SIZE];
        scalar_t grad_coeff0_0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_1_values[VECTOR_SIZE];
        scalar_t grad_coeff1_0_values[VECTOR_SIZE];
        scalar_t grad_coeff1_1_values[VECTOR_SIZE];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 0][lane];
                u0_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_old_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_old_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 0][lane];
                u0_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_direction_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_direction_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 0][lane];
                u0_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 1][lane];
                u1_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_old_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_old_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 1][lane];
                u1_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_direction_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_direction_grad_1_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 1][lane];
                u1_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
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
            const scalar_t u0_grad_0_ref = u0_grad_0_ref_values[lane];
            const scalar_t u0_grad_1_ref = u0_grad_1_ref_values[lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
            const scalar_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[lane];
            const scalar_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
            const scalar_t u0_direction_grad_0_ref = u0_direction_grad_0_ref_values[lane];
            const scalar_t u0_direction_grad_1_ref = u0_direction_grad_1_ref_values[lane];
            const scalar_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
            const scalar_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
            const scalar_t u1_grad_0_ref = u1_grad_0_ref_values[lane];
            const scalar_t u1_grad_1_ref = u1_grad_1_ref_values[lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
            const scalar_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[lane];
            const scalar_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
            const scalar_t u1_direction_grad_0_ref = u1_direction_grad_0_ref_values[lane];
            const scalar_t u1_direction_grad_1_ref = u1_direction_grad_1_ref_values[lane];
            const scalar_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj2) / det;
            const scalar_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj3) / det;
            const scalar_t residual_tmp0 = u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp1 = residual_tmp0 + u0_grad_0*u1_grad_1 + u0_grad_0 - u0_grad_1*u1_grad_0;
            const scalar_t residual_tmp2 = pow_m1(residual_tmp1);
            const scalar_t residual_tmp3 = newmark_velocity_alpha*residual_tmp0;
            const scalar_t residual_tmp4 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp5 = -residual_tmp3 + residual_tmp4;
            const scalar_t residual_tmp6 = eta_s*residual_tmp5;
            const scalar_t residual_tmp7 = eta_s*u0_old_grad_1;
            const scalar_t residual_tmp8 = newmark_velocity_alpha*u0_grad_1;
            const scalar_t residual_tmp9 = eta_b*(scalar_t(2)*residual_tmp8 + u0_old_grad_1);
            const scalar_t residual_tmp10 = residual_tmp7 + residual_tmp9;
            const scalar_t residual_tmp11 = pow_m2(residual_tmp1);
            const scalar_t residual_tmp12 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp13 = u0_grad_0 + scalar_t(1);
            const scalar_t residual_tmp14 = residual_tmp8 + u0_old_grad_1;
            const scalar_t residual_tmp15 = newmark_velocity_alpha*u1_grad_0;
            const scalar_t residual_tmp16 = residual_tmp15 + u1_old_grad_0;
            const scalar_t residual_tmp17 = eta_s*(-residual_tmp0*residual_tmp16 + residual_tmp12*u0_grad_1 - residual_tmp13*residual_tmp14 + residual_tmp4*u1_grad_0);
            const scalar_t residual_tmp18 = residual_tmp14*u1_grad_0;
            const scalar_t residual_tmp19 = residual_tmp0*residual_tmp12;
            const scalar_t residual_tmp20 = -residual_tmp13*residual_tmp4 + residual_tmp16*u0_grad_1;
            const scalar_t residual_tmp21 = eta_b*(residual_tmp18 - residual_tmp19 + residual_tmp20);
            const scalar_t residual_tmp22 = eta_s*(-residual_tmp18 + residual_tmp19 + residual_tmp20);
            const scalar_t residual_tmp23 = residual_tmp21 - residual_tmp22;
            const scalar_t residual_tmp24 = residual_tmp11*(-residual_tmp0*residual_tmp23 + residual_tmp17*u0_grad_1);
            const scalar_t residual_tmp25 = eta_b*(-residual_tmp3 - residual_tmp4);
            const scalar_t residual_tmp26 = -residual_tmp5;
            const scalar_t residual_tmp27 = -eta_s*residual_tmp26 + residual_tmp25;
            const scalar_t residual_tmp28 = -residual_tmp0;
            const scalar_t residual_tmp29 = newmark_velocity_alpha*residual_tmp13;
            const scalar_t residual_tmp30 = residual_tmp12 - residual_tmp29;
            const scalar_t residual_tmp31 = eta_s*residual_tmp30;
            const scalar_t residual_tmp32 = eta_b*(scalar_t(2)*residual_tmp15 + u1_old_grad_0);
            const scalar_t residual_tmp33 = -eta_s*u1_old_grad_0 + residual_tmp32;
            const scalar_t residual_tmp34 = eta_b*(-residual_tmp12 - residual_tmp29);
            const scalar_t residual_tmp35 = -eta_s*residual_tmp30 + residual_tmp34;
            const scalar_t residual_tmp36 = eta_s*u1_old_grad_0;
            const scalar_t residual_tmp37 = -residual_tmp13;
            const scalar_t residual_tmp38 = residual_tmp11*(-residual_tmp13*residual_tmp17 + residual_tmp23*u1_grad_0);
            const scalar_t residual_tmp39 = -residual_tmp17;
            const scalar_t residual_tmp40 = -residual_tmp7 + residual_tmp9;
            const scalar_t residual_tmp41 = residual_tmp21 + residual_tmp22;
            const scalar_t residual_tmp42 = residual_tmp11*(-residual_tmp0*residual_tmp17 + residual_tmp41*u0_grad_1);
            const scalar_t residual_tmp43 = eta_s*residual_tmp26 + residual_tmp25;
            const scalar_t residual_tmp44 = residual_tmp31 + residual_tmp34;
            const scalar_t residual_tmp45 = residual_tmp32 + residual_tmp36;
            const scalar_t residual_tmp46 = residual_tmp11*(-residual_tmp13*residual_tmp41 + residual_tmp17*u1_grad_0);
            const scalar_t grad_coeff0_0 = u0_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp27 - residual_tmp7*u0_grad_1) + residual_tmp24*residual_tmp28) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp33 + residual_tmp17 + residual_tmp31*u0_grad_1) + residual_tmp24*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp10 + residual_tmp6*u0_grad_1) + residual_tmp24*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp35 - residual_tmp23 - residual_tmp36*u0_grad_1) + residual_tmp24*residual_tmp37);
            const scalar_t grad_coeff0_1 = u0_direction_grad_0*(residual_tmp2*(residual_tmp13*residual_tmp7 + residual_tmp27*u1_grad_0 + residual_tmp39) + residual_tmp28*residual_tmp38) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp31 + residual_tmp33*u1_grad_0) + residual_tmp38*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(residual_tmp10*u1_grad_0 - residual_tmp13*residual_tmp6 + residual_tmp23) + residual_tmp38*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(residual_tmp13*residual_tmp36 + residual_tmp35*u1_grad_0) + residual_tmp37*residual_tmp38);
            const scalar_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp2*(residual_tmp0*residual_tmp7 + residual_tmp43*u0_grad_1) + residual_tmp28*residual_tmp42) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp31 + residual_tmp41 + residual_tmp45*u0_grad_1) + residual_tmp42*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp6 + residual_tmp40*u0_grad_1) + residual_tmp42*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(residual_tmp0*residual_tmp36 + residual_tmp39 + residual_tmp44*u0_grad_1) + residual_tmp37*residual_tmp42);
            const scalar_t grad_coeff1_1 = u0_direction_grad_0*(residual_tmp2*(-residual_tmp13*residual_tmp43 - residual_tmp41 - residual_tmp7*u1_grad_0) + residual_tmp28*residual_tmp46) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp45 + residual_tmp31*u1_grad_0) + residual_tmp46*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp13*residual_tmp40 + residual_tmp17 + residual_tmp6*u1_grad_0) + residual_tmp46*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp44 - residual_tmp36*u1_grad_0) + residual_tmp37*residual_tmp46);
            grad_coeff0_0_values[lane] = grad_coeff0_0;
            grad_coeff0_1_values[lane] = grad_coeff0_1;
            grad_coeff1_0_values[lane] = grad_coeff1_0;
            grad_coeff1_1_values[lane] = grad_coeff1_1;
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
                output[test * N_FIELDS + 0][lane] += q_weight[q] * det * (grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1);
                output[test * N_FIELDS + 1][lane] += q_weight[q] * det * (grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d2_simplex_tri3_jacobian_action_block(
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
        const scalar_t *const SFEM_RESTRICT direction[2 * N_SHAPE],
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t det = determinant[geometry_offset];
            const scalar_t adj0 = adjugate[0][geometry_offset];
            const scalar_t adj1 = adjugate[1][geometry_offset];
            const scalar_t adj2 = adjugate[2][geometry_offset];
            const scalar_t adj3 = adjugate[3][geometry_offset];
            const scalar_t u0_grad_0_ref = -(current[0][lane]) + current[2][lane];
            const scalar_t u0_grad_1_ref = -(current[0][lane]) + current[4][lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
            const scalar_t u0_old_grad_0_ref = -(previous[0][lane]) + previous[2][lane];
            const scalar_t u0_old_grad_1_ref = -(previous[0][lane]) + previous[4][lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
            const scalar_t u0_direction_grad_0_ref = -(direction[0][lane]) + direction[2][lane];
            const scalar_t u0_direction_grad_1_ref = -(direction[0][lane]) + direction[4][lane];
            const scalar_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
            const scalar_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
            const scalar_t u1_grad_0_ref = -(current[1][lane]) + current[3][lane];
            const scalar_t u1_grad_1_ref = -(current[1][lane]) + current[5][lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
            const scalar_t u1_old_grad_0_ref = -(previous[1][lane]) + previous[3][lane];
            const scalar_t u1_old_grad_1_ref = -(previous[1][lane]) + previous[5][lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
            const scalar_t u1_direction_grad_0_ref = -(direction[1][lane]) + direction[3][lane];
            const scalar_t u1_direction_grad_1_ref = -(direction[1][lane]) + direction[5][lane];
            const scalar_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj2) / det;
            const scalar_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj3) / det;
            const scalar_t residual_tmp0 = u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp1 = residual_tmp0 + u0_grad_0*u1_grad_1 + u0_grad_0 - u0_grad_1*u1_grad_0;
            const scalar_t residual_tmp2 = pow_m1(residual_tmp1);
            const scalar_t residual_tmp3 = newmark_velocity_alpha*residual_tmp0;
            const scalar_t residual_tmp4 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp5 = -residual_tmp3 + residual_tmp4;
            const scalar_t residual_tmp6 = eta_s*residual_tmp5;
            const scalar_t residual_tmp7 = eta_s*u0_old_grad_1;
            const scalar_t residual_tmp8 = newmark_velocity_alpha*u0_grad_1;
            const scalar_t residual_tmp9 = eta_b*(scalar_t(2)*residual_tmp8 + u0_old_grad_1);
            const scalar_t residual_tmp10 = residual_tmp7 + residual_tmp9;
            const scalar_t residual_tmp11 = pow_m2(residual_tmp1);
            const scalar_t residual_tmp12 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp13 = u0_grad_0 + scalar_t(1);
            const scalar_t residual_tmp14 = residual_tmp8 + u0_old_grad_1;
            const scalar_t residual_tmp15 = newmark_velocity_alpha*u1_grad_0;
            const scalar_t residual_tmp16 = residual_tmp15 + u1_old_grad_0;
            const scalar_t residual_tmp17 = eta_s*(-residual_tmp0*residual_tmp16 + residual_tmp12*u0_grad_1 - residual_tmp13*residual_tmp14 + residual_tmp4*u1_grad_0);
            const scalar_t residual_tmp18 = residual_tmp14*u1_grad_0;
            const scalar_t residual_tmp19 = residual_tmp0*residual_tmp12;
            const scalar_t residual_tmp20 = -residual_tmp13*residual_tmp4 + residual_tmp16*u0_grad_1;
            const scalar_t residual_tmp21 = eta_b*(residual_tmp18 - residual_tmp19 + residual_tmp20);
            const scalar_t residual_tmp22 = eta_s*(-residual_tmp18 + residual_tmp19 + residual_tmp20);
            const scalar_t residual_tmp23 = residual_tmp21 - residual_tmp22;
            const scalar_t residual_tmp24 = residual_tmp11*(-residual_tmp0*residual_tmp23 + residual_tmp17*u0_grad_1);
            const scalar_t residual_tmp25 = eta_b*(-residual_tmp3 - residual_tmp4);
            const scalar_t residual_tmp26 = -residual_tmp5;
            const scalar_t residual_tmp27 = -eta_s*residual_tmp26 + residual_tmp25;
            const scalar_t residual_tmp28 = -residual_tmp0;
            const scalar_t residual_tmp29 = newmark_velocity_alpha*residual_tmp13;
            const scalar_t residual_tmp30 = residual_tmp12 - residual_tmp29;
            const scalar_t residual_tmp31 = eta_s*residual_tmp30;
            const scalar_t residual_tmp32 = eta_b*(scalar_t(2)*residual_tmp15 + u1_old_grad_0);
            const scalar_t residual_tmp33 = -eta_s*u1_old_grad_0 + residual_tmp32;
            const scalar_t residual_tmp34 = eta_b*(-residual_tmp12 - residual_tmp29);
            const scalar_t residual_tmp35 = -eta_s*residual_tmp30 + residual_tmp34;
            const scalar_t residual_tmp36 = eta_s*u1_old_grad_0;
            const scalar_t residual_tmp37 = -residual_tmp13;
            const scalar_t residual_tmp38 = residual_tmp11*(-residual_tmp13*residual_tmp17 + residual_tmp23*u1_grad_0);
            const scalar_t residual_tmp39 = -residual_tmp17;
            const scalar_t residual_tmp40 = -residual_tmp7 + residual_tmp9;
            const scalar_t residual_tmp41 = residual_tmp21 + residual_tmp22;
            const scalar_t residual_tmp42 = residual_tmp11*(-residual_tmp0*residual_tmp17 + residual_tmp41*u0_grad_1);
            const scalar_t residual_tmp43 = eta_s*residual_tmp26 + residual_tmp25;
            const scalar_t residual_tmp44 = residual_tmp31 + residual_tmp34;
            const scalar_t residual_tmp45 = residual_tmp32 + residual_tmp36;
            const scalar_t residual_tmp46 = residual_tmp11*(-residual_tmp13*residual_tmp41 + residual_tmp17*u1_grad_0);
            const scalar_t grad_coeff0_0 = u0_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp27 - residual_tmp7*u0_grad_1) + residual_tmp24*residual_tmp28) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp33 + residual_tmp17 + residual_tmp31*u0_grad_1) + residual_tmp24*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp10 + residual_tmp6*u0_grad_1) + residual_tmp24*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp35 - residual_tmp23 - residual_tmp36*u0_grad_1) + residual_tmp24*residual_tmp37);
            const scalar_t grad_coeff0_1 = u0_direction_grad_0*(residual_tmp2*(residual_tmp13*residual_tmp7 + residual_tmp27*u1_grad_0 + residual_tmp39) + residual_tmp28*residual_tmp38) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp31 + residual_tmp33*u1_grad_0) + residual_tmp38*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(residual_tmp10*u1_grad_0 - residual_tmp13*residual_tmp6 + residual_tmp23) + residual_tmp38*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(residual_tmp13*residual_tmp36 + residual_tmp35*u1_grad_0) + residual_tmp37*residual_tmp38);
            const scalar_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp2*(residual_tmp0*residual_tmp7 + residual_tmp43*u0_grad_1) + residual_tmp28*residual_tmp42) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp31 + residual_tmp41 + residual_tmp45*u0_grad_1) + residual_tmp42*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp6 + residual_tmp40*u0_grad_1) + residual_tmp42*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(residual_tmp0*residual_tmp36 + residual_tmp39 + residual_tmp44*u0_grad_1) + residual_tmp37*residual_tmp42);
            const scalar_t grad_coeff1_1 = u0_direction_grad_0*(residual_tmp2*(-residual_tmp13*residual_tmp43 - residual_tmp41 - residual_tmp7*u1_grad_0) + residual_tmp28*residual_tmp46) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp45 + residual_tmp31*u1_grad_0) + residual_tmp46*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp13*residual_tmp40 + residual_tmp17 + residual_tmp6*u1_grad_0) + residual_tmp46*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp44 - residual_tmp36*u1_grad_0) + residual_tmp37*residual_tmp46);
            const scalar_t grad_coeff0_0_value = grad_coeff0_0;
            const scalar_t grad_coeff0_1_value = grad_coeff0_1;
            const scalar_t grad_coeff1_0_value = grad_coeff1_0;
            const scalar_t grad_coeff1_1_value = grad_coeff1_1;
            const scalar_t test0_grad0 = (-(adj0) - adj2) / det;
            const scalar_t test0_grad1 = (-(adj1) - adj3) / det;
            const scalar_t test1_grad0 = (adj0) / det;
            const scalar_t test1_grad1 = (adj1) / det;
            const scalar_t test2_grad0 = (adj2) / det;
            const scalar_t test2_grad1 = (adj3) / det;
            output[0][lane] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1);
            output[1][lane] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1);
            output[2][lane] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1);
            output[3][lane] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1);
            output[4][lane] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1);
            output[5][lane] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1);
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d2_simplex_tri3_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[4],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t current[2 * N_SHAPE][VECTOR_SIZE],
        const scalar_t previous[2 * N_SHAPE][VECTOR_SIZE],
        const scalar_t direction[2 * N_SHAPE][VECTOR_SIZE],
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t output[2 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 2;
    static constexpr int N_FIELDS = 2;
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t det = determinant[geometry_offset];
            const scalar_t adj0 = adjugate[0][geometry_offset];
            const scalar_t adj1 = adjugate[1][geometry_offset];
            const scalar_t adj2 = adjugate[2][geometry_offset];
            const scalar_t adj3 = adjugate[3][geometry_offset];
            const scalar_t u0_grad_0_ref = -(current[0][lane]) + current[2][lane];
            const scalar_t u0_grad_1_ref = -(current[0][lane]) + current[4][lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj2) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj3) / det;
            const scalar_t u0_old_grad_0_ref = -(previous[0][lane]) + previous[2][lane];
            const scalar_t u0_old_grad_1_ref = -(previous[0][lane]) + previous[4][lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj2) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj3) / det;
            const scalar_t u0_direction_grad_0_ref = -(direction[0][lane]) + direction[2][lane];
            const scalar_t u0_direction_grad_1_ref = -(direction[0][lane]) + direction[4][lane];
            const scalar_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj2) / det;
            const scalar_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj3) / det;
            const scalar_t u1_grad_0_ref = -(current[1][lane]) + current[3][lane];
            const scalar_t u1_grad_1_ref = -(current[1][lane]) + current[5][lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj2) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj3) / det;
            const scalar_t u1_old_grad_0_ref = -(previous[1][lane]) + previous[3][lane];
            const scalar_t u1_old_grad_1_ref = -(previous[1][lane]) + previous[5][lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj2) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj3) / det;
            const scalar_t u1_direction_grad_0_ref = -(direction[1][lane]) + direction[3][lane];
            const scalar_t u1_direction_grad_1_ref = -(direction[1][lane]) + direction[5][lane];
            const scalar_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj2) / det;
            const scalar_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj3) / det;
            const scalar_t residual_tmp0 = u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp1 = residual_tmp0 + u0_grad_0*u1_grad_1 + u0_grad_0 - u0_grad_1*u1_grad_0;
            const scalar_t residual_tmp2 = pow_m1(residual_tmp1);
            const scalar_t residual_tmp3 = newmark_velocity_alpha*residual_tmp0;
            const scalar_t residual_tmp4 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp5 = -residual_tmp3 + residual_tmp4;
            const scalar_t residual_tmp6 = eta_s*residual_tmp5;
            const scalar_t residual_tmp7 = eta_s*u0_old_grad_1;
            const scalar_t residual_tmp8 = newmark_velocity_alpha*u0_grad_1;
            const scalar_t residual_tmp9 = eta_b*(scalar_t(2)*residual_tmp8 + u0_old_grad_1);
            const scalar_t residual_tmp10 = residual_tmp7 + residual_tmp9;
            const scalar_t residual_tmp11 = pow_m2(residual_tmp1);
            const scalar_t residual_tmp12 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp13 = u0_grad_0 + scalar_t(1);
            const scalar_t residual_tmp14 = residual_tmp8 + u0_old_grad_1;
            const scalar_t residual_tmp15 = newmark_velocity_alpha*u1_grad_0;
            const scalar_t residual_tmp16 = residual_tmp15 + u1_old_grad_0;
            const scalar_t residual_tmp17 = eta_s*(-residual_tmp0*residual_tmp16 + residual_tmp12*u0_grad_1 - residual_tmp13*residual_tmp14 + residual_tmp4*u1_grad_0);
            const scalar_t residual_tmp18 = residual_tmp14*u1_grad_0;
            const scalar_t residual_tmp19 = residual_tmp0*residual_tmp12;
            const scalar_t residual_tmp20 = -residual_tmp13*residual_tmp4 + residual_tmp16*u0_grad_1;
            const scalar_t residual_tmp21 = eta_b*(residual_tmp18 - residual_tmp19 + residual_tmp20);
            const scalar_t residual_tmp22 = eta_s*(-residual_tmp18 + residual_tmp19 + residual_tmp20);
            const scalar_t residual_tmp23 = residual_tmp21 - residual_tmp22;
            const scalar_t residual_tmp24 = residual_tmp11*(-residual_tmp0*residual_tmp23 + residual_tmp17*u0_grad_1);
            const scalar_t residual_tmp25 = eta_b*(-residual_tmp3 - residual_tmp4);
            const scalar_t residual_tmp26 = -residual_tmp5;
            const scalar_t residual_tmp27 = -eta_s*residual_tmp26 + residual_tmp25;
            const scalar_t residual_tmp28 = -residual_tmp0;
            const scalar_t residual_tmp29 = newmark_velocity_alpha*residual_tmp13;
            const scalar_t residual_tmp30 = residual_tmp12 - residual_tmp29;
            const scalar_t residual_tmp31 = eta_s*residual_tmp30;
            const scalar_t residual_tmp32 = eta_b*(scalar_t(2)*residual_tmp15 + u1_old_grad_0);
            const scalar_t residual_tmp33 = -eta_s*u1_old_grad_0 + residual_tmp32;
            const scalar_t residual_tmp34 = eta_b*(-residual_tmp12 - residual_tmp29);
            const scalar_t residual_tmp35 = -eta_s*residual_tmp30 + residual_tmp34;
            const scalar_t residual_tmp36 = eta_s*u1_old_grad_0;
            const scalar_t residual_tmp37 = -residual_tmp13;
            const scalar_t residual_tmp38 = residual_tmp11*(-residual_tmp13*residual_tmp17 + residual_tmp23*u1_grad_0);
            const scalar_t residual_tmp39 = -residual_tmp17;
            const scalar_t residual_tmp40 = -residual_tmp7 + residual_tmp9;
            const scalar_t residual_tmp41 = residual_tmp21 + residual_tmp22;
            const scalar_t residual_tmp42 = residual_tmp11*(-residual_tmp0*residual_tmp17 + residual_tmp41*u0_grad_1);
            const scalar_t residual_tmp43 = eta_s*residual_tmp26 + residual_tmp25;
            const scalar_t residual_tmp44 = residual_tmp31 + residual_tmp34;
            const scalar_t residual_tmp45 = residual_tmp32 + residual_tmp36;
            const scalar_t residual_tmp46 = residual_tmp11*(-residual_tmp13*residual_tmp41 + residual_tmp17*u1_grad_0);
            const scalar_t grad_coeff0_0 = u0_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp27 - residual_tmp7*u0_grad_1) + residual_tmp24*residual_tmp28) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp33 + residual_tmp17 + residual_tmp31*u0_grad_1) + residual_tmp24*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp10 + residual_tmp6*u0_grad_1) + residual_tmp24*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp35 - residual_tmp23 - residual_tmp36*u0_grad_1) + residual_tmp24*residual_tmp37);
            const scalar_t grad_coeff0_1 = u0_direction_grad_0*(residual_tmp2*(residual_tmp13*residual_tmp7 + residual_tmp27*u1_grad_0 + residual_tmp39) + residual_tmp28*residual_tmp38) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp31 + residual_tmp33*u1_grad_0) + residual_tmp38*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(residual_tmp10*u1_grad_0 - residual_tmp13*residual_tmp6 + residual_tmp23) + residual_tmp38*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(residual_tmp13*residual_tmp36 + residual_tmp35*u1_grad_0) + residual_tmp37*residual_tmp38);
            const scalar_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp2*(residual_tmp0*residual_tmp7 + residual_tmp43*u0_grad_1) + residual_tmp28*residual_tmp42) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp0*residual_tmp31 + residual_tmp41 + residual_tmp45*u0_grad_1) + residual_tmp42*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp0*residual_tmp6 + residual_tmp40*u0_grad_1) + residual_tmp42*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(residual_tmp0*residual_tmp36 + residual_tmp39 + residual_tmp44*u0_grad_1) + residual_tmp37*residual_tmp42);
            const scalar_t grad_coeff1_1 = u0_direction_grad_0*(residual_tmp2*(-residual_tmp13*residual_tmp43 - residual_tmp41 - residual_tmp7*u1_grad_0) + residual_tmp28*residual_tmp46) + u0_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp45 + residual_tmp31*u1_grad_0) + residual_tmp46*u1_grad_0) + u1_direction_grad_0*(residual_tmp2*(-residual_tmp13*residual_tmp40 + residual_tmp17 + residual_tmp6*u1_grad_0) + residual_tmp46*u0_grad_1) + u1_direction_grad_1*(residual_tmp2*(-residual_tmp13*residual_tmp44 - residual_tmp36*u1_grad_0) + residual_tmp37*residual_tmp46);
            const scalar_t grad_coeff0_0_value = grad_coeff0_0;
            const scalar_t grad_coeff0_1_value = grad_coeff0_1;
            const scalar_t grad_coeff1_0_value = grad_coeff1_0;
            const scalar_t grad_coeff1_1_value = grad_coeff1_1;
            const scalar_t test0_grad0 = (-(adj0) - adj2) / det;
            const scalar_t test0_grad1 = (-(adj1) - adj3) / det;
            const scalar_t test1_grad0 = (adj0) / det;
            const scalar_t test1_grad1 = (adj1) / det;
            const scalar_t test2_grad0 = (adj2) / det;
            const scalar_t test2_grad1 = (adj3) / det;
            output[0][lane] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1);
            output[1][lane] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1);
            output[2][lane] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1);
            output[3][lane] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1);
            output[4][lane] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1);
            output[5][lane] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1);
        }
    }
}

} // namespace codegen
} // namespace sfem

#endif
