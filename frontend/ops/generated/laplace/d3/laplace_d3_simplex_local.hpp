#ifndef LAPLACE_D3_SIMPLEX_LOCAL_HPP
#define LAPLACE_D3_SIMPLEX_LOCAL_HPP

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
typedef double geom_t;
#endif

namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void laplace_d3_simplex_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT current[1 * N_SHAPE],
        const scalar_t kappa,
        scalar_t *const SFEM_RESTRICT output[1 * N_SHAPE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 1;
    for (int q = 0; q < N_QP; ++q) {
        scalar_t u_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u_grad_2_ref_values[VECTOR_SIZE];
        scalar_t grad_coeff0_0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_1_values[VECTOR_SIZE];
        scalar_t grad_coeff0_2_values[VECTOR_SIZE];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 0][lane];
                u_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
            const scalar_t u_grad_0_ref = u_grad_0_ref_values[lane];
            const scalar_t u_grad_1_ref = u_grad_1_ref_values[lane];
            const scalar_t u_grad_2_ref = u_grad_2_ref_values[lane];
            const scalar_t u_grad_0 = (u_grad_0_ref * adj0 + u_grad_1_ref * adj3 + u_grad_2_ref * adj6) / det;
            const scalar_t u_grad_1 = (u_grad_0_ref * adj1 + u_grad_1_ref * adj4 + u_grad_2_ref * adj7) / det;
            const scalar_t u_grad_2 = (u_grad_0_ref * adj2 + u_grad_1_ref * adj5 + u_grad_2_ref * adj8) / det;
            const scalar_t grad_coeff0_0 = kappa*u_grad_0;
            const scalar_t grad_coeff0_1 = kappa*u_grad_1;
            const scalar_t grad_coeff0_2 = kappa*u_grad_2;
            grad_coeff0_0_values[lane] = grad_coeff0_0;
            grad_coeff0_1_values[lane] = grad_coeff0_1;
            grad_coeff0_2_values[lane] = grad_coeff0_2;
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
                output[test * N_FIELDS + 0][lane] += q_weight[q] * det * (grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void laplace_d3_simplex_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t current[1 * N_SHAPE][VECTOR_SIZE],
        const scalar_t kappa,
        scalar_t output[1 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 1;
    for (int q = 0; q < N_QP; ++q) {
        scalar_t u_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u_grad_2_ref_values[VECTOR_SIZE];
        scalar_t grad_coeff0_0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_1_values[VECTOR_SIZE];
        scalar_t grad_coeff0_2_values[VECTOR_SIZE];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 0][lane];
                u_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
            const scalar_t u_grad_0_ref = u_grad_0_ref_values[lane];
            const scalar_t u_grad_1_ref = u_grad_1_ref_values[lane];
            const scalar_t u_grad_2_ref = u_grad_2_ref_values[lane];
            const scalar_t u_grad_0 = (u_grad_0_ref * adj0 + u_grad_1_ref * adj3 + u_grad_2_ref * adj6) / det;
            const scalar_t u_grad_1 = (u_grad_0_ref * adj1 + u_grad_1_ref * adj4 + u_grad_2_ref * adj7) / det;
            const scalar_t u_grad_2 = (u_grad_0_ref * adj2 + u_grad_1_ref * adj5 + u_grad_2_ref * adj8) / det;
            const scalar_t grad_coeff0_0 = kappa*u_grad_0;
            const scalar_t grad_coeff0_1 = kappa*u_grad_1;
            const scalar_t grad_coeff0_2 = kappa*u_grad_2;
            grad_coeff0_0_values[lane] = grad_coeff0_0;
            grad_coeff0_1_values[lane] = grad_coeff0_1;
            grad_coeff0_2_values[lane] = grad_coeff0_2;
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
                output[test * N_FIELDS + 0][lane] += q_weight[q] * det * (grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void laplace_d3_simplex_tet4_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT geom_metric[6],
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT current[1 * N_SHAPE],
        const scalar_t kappa,
        scalar_t *const SFEM_RESTRICT output[1 * N_SHAPE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 1;
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const scalar_t coeff_current_u_0 = current[0][lane];
            const scalar_t coeff_current_u_1 = current[1][lane];
            const scalar_t coeff_current_u_2 = current[2][lane];
            const scalar_t coeff_current_u_3 = current[3][lane];
            const scalar_t u_grad_0_ref_value = -(coeff_current_u_0) + coeff_current_u_1;
            const scalar_t u_grad_1_ref_value = -(coeff_current_u_0) + coeff_current_u_2;
            const scalar_t u_grad_2_ref_value = -(coeff_current_u_0) + coeff_current_u_3;
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t metric_factor = q_weight[q] * (kappa);
            const scalar_t geom_metric00 = metric_factor * geom_metric[0][geometry_offset];
            const scalar_t geom_metric01 = metric_factor * geom_metric[1][geometry_offset];
            const scalar_t geom_metric02 = metric_factor * geom_metric[3][geometry_offset];
            const scalar_t geom_metric11 = metric_factor * geom_metric[2][geometry_offset];
            const scalar_t geom_metric12 = metric_factor * geom_metric[4][geometry_offset];
            const scalar_t geom_metric22 = metric_factor * geom_metric[5][geometry_offset];
            const scalar_t u_metric_grad_0_ref_value = geom_metric00 * u_grad_0_ref_value + geom_metric01 * u_grad_1_ref_value + geom_metric02 * u_grad_2_ref_value;
            const scalar_t u_metric_grad_1_ref_value = geom_metric01 * u_grad_0_ref_value + geom_metric11 * u_grad_1_ref_value + geom_metric12 * u_grad_2_ref_value;
            const scalar_t u_metric_grad_2_ref_value = geom_metric02 * u_grad_0_ref_value + geom_metric12 * u_grad_1_ref_value + geom_metric22 * u_grad_2_ref_value;
            output[0][lane] += -(u_metric_grad_0_ref_value) - u_metric_grad_1_ref_value - u_metric_grad_2_ref_value;
            output[1][lane] += u_metric_grad_0_ref_value;
            output[2][lane] += u_metric_grad_1_ref_value;
            output[3][lane] += u_metric_grad_2_ref_value;
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void laplace_d3_simplex_tet4_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT geom_metric[6],
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t current[1 * N_SHAPE][VECTOR_SIZE],
        const scalar_t kappa,
        scalar_t output[1 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 1;
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const scalar_t coeff_current_u_0 = current[0][lane];
            const scalar_t coeff_current_u_1 = current[1][lane];
            const scalar_t coeff_current_u_2 = current[2][lane];
            const scalar_t coeff_current_u_3 = current[3][lane];
            const scalar_t u_grad_0_ref_value = -(coeff_current_u_0) + coeff_current_u_1;
            const scalar_t u_grad_1_ref_value = -(coeff_current_u_0) + coeff_current_u_2;
            const scalar_t u_grad_2_ref_value = -(coeff_current_u_0) + coeff_current_u_3;
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t metric_factor = q_weight[q] * (kappa);
            const scalar_t geom_metric00 = metric_factor * geom_metric[0][geometry_offset];
            const scalar_t geom_metric01 = metric_factor * geom_metric[1][geometry_offset];
            const scalar_t geom_metric02 = metric_factor * geom_metric[3][geometry_offset];
            const scalar_t geom_metric11 = metric_factor * geom_metric[2][geometry_offset];
            const scalar_t geom_metric12 = metric_factor * geom_metric[4][geometry_offset];
            const scalar_t geom_metric22 = metric_factor * geom_metric[5][geometry_offset];
            const scalar_t u_metric_grad_0_ref_value = geom_metric00 * u_grad_0_ref_value + geom_metric01 * u_grad_1_ref_value + geom_metric02 * u_grad_2_ref_value;
            const scalar_t u_metric_grad_1_ref_value = geom_metric01 * u_grad_0_ref_value + geom_metric11 * u_grad_1_ref_value + geom_metric12 * u_grad_2_ref_value;
            const scalar_t u_metric_grad_2_ref_value = geom_metric02 * u_grad_0_ref_value + geom_metric12 * u_grad_1_ref_value + geom_metric22 * u_grad_2_ref_value;
            output[0][lane] += -(u_metric_grad_0_ref_value) - u_metric_grad_1_ref_value - u_metric_grad_2_ref_value;
            output[1][lane] += u_metric_grad_0_ref_value;
            output[2][lane] += u_metric_grad_1_ref_value;
            output[3][lane] += u_metric_grad_2_ref_value;
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void laplace_d3_simplex_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT direction[1 * N_SHAPE],
        const scalar_t kappa,
        scalar_t *const SFEM_RESTRICT output[1 * N_SHAPE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 1;
    for (int q = 0; q < N_QP; ++q) {
        scalar_t u_direction_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u_direction_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u_direction_grad_2_ref_values[VECTOR_SIZE];
        scalar_t grad_coeff0_0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_1_values[VECTOR_SIZE];
        scalar_t grad_coeff0_2_values[VECTOR_SIZE];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u_direction_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u_direction_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u_direction_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 0][lane];
                u_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
            const scalar_t u_direction_grad_0_ref = u_direction_grad_0_ref_values[lane];
            const scalar_t u_direction_grad_1_ref = u_direction_grad_1_ref_values[lane];
            const scalar_t u_direction_grad_2_ref = u_direction_grad_2_ref_values[lane];
            const scalar_t u_direction_grad_0 = (u_direction_grad_0_ref * adj0 + u_direction_grad_1_ref * adj3 + u_direction_grad_2_ref * adj6) / det;
            const scalar_t u_direction_grad_1 = (u_direction_grad_0_ref * adj1 + u_direction_grad_1_ref * adj4 + u_direction_grad_2_ref * adj7) / det;
            const scalar_t u_direction_grad_2 = (u_direction_grad_0_ref * adj2 + u_direction_grad_1_ref * adj5 + u_direction_grad_2_ref * adj8) / det;
            const scalar_t grad_coeff0_0 = kappa*u_direction_grad_0;
            const scalar_t grad_coeff0_1 = kappa*u_direction_grad_1;
            const scalar_t grad_coeff0_2 = kappa*u_direction_grad_2;
            grad_coeff0_0_values[lane] = grad_coeff0_0;
            grad_coeff0_1_values[lane] = grad_coeff0_1;
            grad_coeff0_2_values[lane] = grad_coeff0_2;
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
                output[test * N_FIELDS + 0][lane] += q_weight[q] * det * (grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void laplace_d3_simplex_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t direction[1 * N_SHAPE][VECTOR_SIZE],
        const scalar_t kappa,
        scalar_t output[1 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 1;
    for (int q = 0; q < N_QP; ++q) {
        scalar_t u_direction_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u_direction_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u_direction_grad_2_ref_values[VECTOR_SIZE];
        scalar_t grad_coeff0_0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_1_values[VECTOR_SIZE];
        scalar_t grad_coeff0_2_values[VECTOR_SIZE];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u_direction_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u_direction_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u_direction_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 0][lane];
                u_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
            const scalar_t u_direction_grad_0_ref = u_direction_grad_0_ref_values[lane];
            const scalar_t u_direction_grad_1_ref = u_direction_grad_1_ref_values[lane];
            const scalar_t u_direction_grad_2_ref = u_direction_grad_2_ref_values[lane];
            const scalar_t u_direction_grad_0 = (u_direction_grad_0_ref * adj0 + u_direction_grad_1_ref * adj3 + u_direction_grad_2_ref * adj6) / det;
            const scalar_t u_direction_grad_1 = (u_direction_grad_0_ref * adj1 + u_direction_grad_1_ref * adj4 + u_direction_grad_2_ref * adj7) / det;
            const scalar_t u_direction_grad_2 = (u_direction_grad_0_ref * adj2 + u_direction_grad_1_ref * adj5 + u_direction_grad_2_ref * adj8) / det;
            const scalar_t grad_coeff0_0 = kappa*u_direction_grad_0;
            const scalar_t grad_coeff0_1 = kappa*u_direction_grad_1;
            const scalar_t grad_coeff0_2 = kappa*u_direction_grad_2;
            grad_coeff0_0_values[lane] = grad_coeff0_0;
            grad_coeff0_1_values[lane] = grad_coeff0_1;
            grad_coeff0_2_values[lane] = grad_coeff0_2;
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
                output[test * N_FIELDS + 0][lane] += q_weight[q] * det * (grad_coeff0_0_values[lane] * test_grad0 + grad_coeff0_1_values[lane] * test_grad1 + grad_coeff0_2_values[lane] * test_grad2);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void laplace_d3_simplex_tet4_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT geom_metric[6],
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT direction[1 * N_SHAPE],
        const scalar_t kappa,
        scalar_t *const SFEM_RESTRICT output[1 * N_SHAPE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 1;
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const scalar_t coeff_direction_u_0 = direction[0][lane];
            const scalar_t coeff_direction_u_1 = direction[1][lane];
            const scalar_t coeff_direction_u_2 = direction[2][lane];
            const scalar_t coeff_direction_u_3 = direction[3][lane];
            const scalar_t u_direction_grad_0_ref_value = -(coeff_direction_u_0) + coeff_direction_u_1;
            const scalar_t u_direction_grad_1_ref_value = -(coeff_direction_u_0) + coeff_direction_u_2;
            const scalar_t u_direction_grad_2_ref_value = -(coeff_direction_u_0) + coeff_direction_u_3;
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t metric_factor = q_weight[q] * (kappa);
            const scalar_t geom_metric00 = metric_factor * geom_metric[0][geometry_offset];
            const scalar_t geom_metric01 = metric_factor * geom_metric[1][geometry_offset];
            const scalar_t geom_metric02 = metric_factor * geom_metric[3][geometry_offset];
            const scalar_t geom_metric11 = metric_factor * geom_metric[2][geometry_offset];
            const scalar_t geom_metric12 = metric_factor * geom_metric[4][geometry_offset];
            const scalar_t geom_metric22 = metric_factor * geom_metric[5][geometry_offset];
            const scalar_t u_direction_metric_grad_0_ref_value = geom_metric00 * u_direction_grad_0_ref_value + geom_metric01 * u_direction_grad_1_ref_value + geom_metric02 * u_direction_grad_2_ref_value;
            const scalar_t u_direction_metric_grad_1_ref_value = geom_metric01 * u_direction_grad_0_ref_value + geom_metric11 * u_direction_grad_1_ref_value + geom_metric12 * u_direction_grad_2_ref_value;
            const scalar_t u_direction_metric_grad_2_ref_value = geom_metric02 * u_direction_grad_0_ref_value + geom_metric12 * u_direction_grad_1_ref_value + geom_metric22 * u_direction_grad_2_ref_value;
            output[0][lane] += -(u_direction_metric_grad_0_ref_value) - u_direction_metric_grad_1_ref_value - u_direction_metric_grad_2_ref_value;
            output[1][lane] += u_direction_metric_grad_0_ref_value;
            output[2][lane] += u_direction_metric_grad_1_ref_value;
            output[3][lane] += u_direction_metric_grad_2_ref_value;
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void laplace_d3_simplex_tet4_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT geom_metric[6],
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t direction[1 * N_SHAPE][VECTOR_SIZE],
        const scalar_t kappa,
        scalar_t output[1 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 1;
    for (int q = 0; q < N_QP; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const scalar_t coeff_direction_u_0 = direction[0][lane];
            const scalar_t coeff_direction_u_1 = direction[1][lane];
            const scalar_t coeff_direction_u_2 = direction[2][lane];
            const scalar_t coeff_direction_u_3 = direction[3][lane];
            const scalar_t u_direction_grad_0_ref_value = -(coeff_direction_u_0) + coeff_direction_u_1;
            const scalar_t u_direction_grad_1_ref_value = -(coeff_direction_u_0) + coeff_direction_u_2;
            const scalar_t u_direction_grad_2_ref_value = -(coeff_direction_u_0) + coeff_direction_u_3;
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t metric_factor = q_weight[q] * (kappa);
            const scalar_t geom_metric00 = metric_factor * geom_metric[0][geometry_offset];
            const scalar_t geom_metric01 = metric_factor * geom_metric[1][geometry_offset];
            const scalar_t geom_metric02 = metric_factor * geom_metric[3][geometry_offset];
            const scalar_t geom_metric11 = metric_factor * geom_metric[2][geometry_offset];
            const scalar_t geom_metric12 = metric_factor * geom_metric[4][geometry_offset];
            const scalar_t geom_metric22 = metric_factor * geom_metric[5][geometry_offset];
            const scalar_t u_direction_metric_grad_0_ref_value = geom_metric00 * u_direction_grad_0_ref_value + geom_metric01 * u_direction_grad_1_ref_value + geom_metric02 * u_direction_grad_2_ref_value;
            const scalar_t u_direction_metric_grad_1_ref_value = geom_metric01 * u_direction_grad_0_ref_value + geom_metric11 * u_direction_grad_1_ref_value + geom_metric12 * u_direction_grad_2_ref_value;
            const scalar_t u_direction_metric_grad_2_ref_value = geom_metric02 * u_direction_grad_0_ref_value + geom_metric12 * u_direction_grad_1_ref_value + geom_metric22 * u_direction_grad_2_ref_value;
            output[0][lane] += -(u_direction_metric_grad_0_ref_value) - u_direction_metric_grad_1_ref_value - u_direction_metric_grad_2_ref_value;
            output[1][lane] += u_direction_metric_grad_0_ref_value;
            output[2][lane] += u_direction_metric_grad_1_ref_value;
            output[3][lane] += u_direction_metric_grad_2_ref_value;
        }
    }
}

} // namespace codegen
} // namespace sfem

#endif
