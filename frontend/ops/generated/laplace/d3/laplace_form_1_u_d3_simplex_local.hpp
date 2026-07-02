#ifndef LAPLACE_FORM_1_U_D3_SIMPLEX_LOCAL_HPP
#define LAPLACE_FORM_1_U_D3_SIMPLEX_LOCAL_HPP

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
static SFEM_INLINE void laplace_form_1_u_d3_simplex_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT geom_metric[6],
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
            const scalar_t geom_metric00 = (q_weight[q] * (kappa)) * geom_metric[0][geometry_offset];
            const scalar_t geom_metric01 = (q_weight[q] * (kappa)) * geom_metric[1][geometry_offset];
            const scalar_t geom_metric02 = (q_weight[q] * (kappa)) * geom_metric[3][geometry_offset];
            const scalar_t geom_metric11 = (q_weight[q] * (kappa)) * geom_metric[2][geometry_offset];
            const scalar_t geom_metric12 = (q_weight[q] * (kappa)) * geom_metric[4][geometry_offset];
            const scalar_t geom_metric22 = (q_weight[q] * (kappa)) * geom_metric[5][geometry_offset];
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
static SFEM_INLINE void laplace_form_1_u_d3_simplex_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT q_weight,
        scalar_t *const SFEM_RESTRICT output[1 * N_SHAPE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 1;
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
