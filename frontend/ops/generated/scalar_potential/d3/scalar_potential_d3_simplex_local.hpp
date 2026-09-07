#ifndef SCALAR_POTENTIAL_D3_SIMPLEX_LOCAL_HPP
#define SCALAR_POTENTIAL_D3_SIMPLEX_LOCAL_HPP
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
static SFEM_INLINE void scalar_potential_d3_simplex_objective_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate4,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate5,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate6,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate7,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate8,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t kappa,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 1],
        scalar_t *const SFEM_RESTRICT value
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
        for (int q = 0; q < N_QP; ++q) {
            const scalar_t qw = q_weight[q];
            scalar_t grad_u_ref0_values[VECTOR_SIZE];
            scalar_t grad_u_ref1_values[VECTOR_SIZE];
            scalar_t grad_u_ref2_values[VECTOR_SIZE];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref0_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref1_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref2_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref0_values[lane] += u_streams[shape * 1 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref1_values[lane] += u_streams[shape * 1 + 0][lane] * grad_ref_y[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref2_values[lane] += u_streams[shape * 1 + 0][lane] * grad_ref_z[q * N_SHAPE + shape];
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const scalar_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const scalar_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const scalar_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const scalar_t jacobian_adjugate_lane4 = jacobian_adjugate4[geometry_offset];
            const scalar_t jacobian_adjugate_lane5 = jacobian_adjugate5[geometry_offset];
            const scalar_t jacobian_adjugate_lane6 = jacobian_adjugate6[geometry_offset];
            const scalar_t jacobian_adjugate_lane7 = jacobian_adjugate7[geometry_offset];
            const scalar_t jacobian_adjugate_lane8 = jacobian_adjugate8[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const scalar_t grad_u_ref0 = grad_u_ref0_values[lane];
            const scalar_t grad_u_ref1 = grad_u_ref1_values[lane];
            const scalar_t grad_u_ref2 = grad_u_ref2_values[lane];
        const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
        const scalar_t grad_u0 = (grad_u_ref0 * jacobian_adjugate_lane0 + grad_u_ref1 * jacobian_adjugate_lane3 + grad_u_ref2 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t grad_u1 = (grad_u_ref0 * jacobian_adjugate_lane1 + grad_u_ref1 * jacobian_adjugate_lane4 + grad_u_ref2 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t grad_u2 = (grad_u_ref0 * jacobian_adjugate_lane2 + grad_u_ref1 * jacobian_adjugate_lane5 + grad_u_ref2 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        value[lane] += qw * jacobian_determinant_lane0 * (((scalar_t(1) / scalar_t(2)))*kappa*(pow_2(grad_u0) + pow_2(grad_u1) + pow_2(grad_u2)));
            }
        }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void scalar_potential_d3_simplex_tet4_objective_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate4,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate5,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate6,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate7,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate8,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t kappa,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 1],
        scalar_t *const SFEM_RESTRICT value
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
        { const int q = 0;  // constant-P1 simplex
            const scalar_t qw = q_weight[q];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const scalar_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const scalar_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const scalar_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const scalar_t jacobian_adjugate_lane4 = jacobian_adjugate4[geometry_offset];
            const scalar_t jacobian_adjugate_lane5 = jacobian_adjugate5[geometry_offset];
            const scalar_t jacobian_adjugate_lane6 = jacobian_adjugate6[geometry_offset];
            const scalar_t jacobian_adjugate_lane7 = jacobian_adjugate7[geometry_offset];
            const scalar_t jacobian_adjugate_lane8 = jacobian_adjugate8[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const scalar_t grad_u_ref0 = -(u_streams[0 * 1 + 0][lane]) + u_streams[1 * 1 + 0][lane];
            const scalar_t grad_u_ref1 = -(u_streams[0 * 1 + 0][lane]) + u_streams[2 * 1 + 0][lane];
            const scalar_t grad_u_ref2 = -(u_streams[0 * 1 + 0][lane]) + u_streams[3 * 1 + 0][lane];
            const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
            const scalar_t grad_u0 = (grad_u_ref0 * jacobian_adjugate_lane0 + grad_u_ref1 * jacobian_adjugate_lane3 + grad_u_ref2 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            const scalar_t grad_u1 = (grad_u_ref0 * jacobian_adjugate_lane1 + grad_u_ref1 * jacobian_adjugate_lane4 + grad_u_ref2 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            const scalar_t grad_u2 = (grad_u_ref0 * jacobian_adjugate_lane2 + grad_u_ref1 * jacobian_adjugate_lane5 + grad_u_ref2 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        value[lane] += qw * jacobian_determinant_lane0 * (((scalar_t(1) / scalar_t(2)))*kappa*(pow_2(grad_u0) + pow_2(grad_u1) + pow_2(grad_u2)));
            }
        }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void scalar_potential_d3_simplex_tet4_metric_objective_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT geom_metric0,
        const scalar_t *const SFEM_RESTRICT geom_metric1,
        const scalar_t *const SFEM_RESTRICT geom_metric2,
        const scalar_t *const SFEM_RESTRICT geom_metric3,
        const scalar_t *const SFEM_RESTRICT geom_metric4,
        const scalar_t *const SFEM_RESTRICT geom_metric5,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t kappa,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 1],
        scalar_t *const SFEM_RESTRICT value
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = lane;
            const scalar_t geom_metric_lane0 = geom_metric0[geometry_offset];
            const scalar_t geom_metric_lane1 = geom_metric1[geometry_offset];
            const scalar_t geom_metric_lane2 = geom_metric2[geometry_offset];
            const scalar_t geom_metric_lane3 = geom_metric3[geometry_offset];
            const scalar_t geom_metric_lane4 = geom_metric4[geometry_offset];
            const scalar_t geom_metric_lane5 = geom_metric5[geometry_offset];
            const scalar_t t0 = -u_streams[0 * 1 + 0][lane] + u_streams[1 * 1 + 0][lane];
            const scalar_t t1 = -u_streams[0 * 1 + 0][lane] + u_streams[2 * 1 + 0][lane];
            const scalar_t t2 = -u_streams[0 * 1 + 0][lane] + u_streams[3 * 1 + 0][lane];
            const scalar_t t3 = geom_metric_lane0*t0 + geom_metric_lane1*t1 + geom_metric_lane2*t2;
            const scalar_t t4 = geom_metric_lane1*t0 + geom_metric_lane3*t1 + geom_metric_lane4*t2;
            const scalar_t t5 = geom_metric_lane2*t0 + geom_metric_lane4*t1 + geom_metric_lane5*t2;
            value[lane] += ((scalar_t(1) / scalar_t(2)))*kappa*(t3*(-u_streams[0 * 1 + 0][lane] + u_streams[1 * 1 + 0][lane]) + t4*(-u_streams[0 * 1 + 0][lane] + u_streams[2 * 1 + 0][lane]) + t5*(-u_streams[0 * 1 + 0][lane] + u_streams[3 * 1 + 0][lane]));
        }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void scalar_potential_d3_simplex_gradient_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate4,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate5,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate6,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate7,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate8,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t kappa,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 1],
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 1]
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
        for (int q = 0; q < N_QP; ++q) {
            const scalar_t qw = q_weight[q];
            scalar_t grad_u_ref0_values[VECTOR_SIZE];
            scalar_t grad_u_ref1_values[VECTOR_SIZE];
            scalar_t grad_u_ref2_values[VECTOR_SIZE];
            scalar_t loperand0_values[VECTOR_SIZE];
            scalar_t loperand1_values[VECTOR_SIZE];
            scalar_t loperand2_values[VECTOR_SIZE];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref0_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref1_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref2_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref0_values[lane] += u_streams[shape * 1 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref1_values[lane] += u_streams[shape * 1 + 0][lane] * grad_ref_y[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref2_values[lane] += u_streams[shape * 1 + 0][lane] * grad_ref_z[q * N_SHAPE + shape];
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const scalar_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const scalar_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const scalar_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const scalar_t jacobian_adjugate_lane4 = jacobian_adjugate4[geometry_offset];
            const scalar_t jacobian_adjugate_lane5 = jacobian_adjugate5[geometry_offset];
            const scalar_t jacobian_adjugate_lane6 = jacobian_adjugate6[geometry_offset];
            const scalar_t jacobian_adjugate_lane7 = jacobian_adjugate7[geometry_offset];
            const scalar_t jacobian_adjugate_lane8 = jacobian_adjugate8[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const scalar_t grad_u_ref0 = grad_u_ref0_values[lane];
            const scalar_t grad_u_ref1 = grad_u_ref1_values[lane];
            const scalar_t grad_u_ref2 = grad_u_ref2_values[lane];
        const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
        const scalar_t grad_u0 = (grad_u_ref0 * jacobian_adjugate_lane0 + grad_u_ref1 * jacobian_adjugate_lane3 + grad_u_ref2 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t grad_u1 = (grad_u_ref0 * jacobian_adjugate_lane1 + grad_u_ref1 * jacobian_adjugate_lane4 + grad_u_ref2 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t grad_u2 = (grad_u_ref0 * jacobian_adjugate_lane2 + grad_u_ref1 * jacobian_adjugate_lane5 + grad_u_ref2 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t material0 = grad_u0*kappa;
        const scalar_t material1 = grad_u1*kappa;
        const scalar_t material2 = grad_u2*kappa;
        const scalar_t loperand0 = qw * (material0 * jacobian_adjugate_lane0 + material1 * jacobian_adjugate_lane1 + material2 * jacobian_adjugate_lane2);
        const scalar_t loperand1 = qw * (material0 * jacobian_adjugate_lane3 + material1 * jacobian_adjugate_lane4 + material2 * jacobian_adjugate_lane5);
        const scalar_t loperand2 = qw * (material0 * jacobian_adjugate_lane6 + material1 * jacobian_adjugate_lane7 + material2 * jacobian_adjugate_lane8);
            loperand0_values[lane] = loperand0;
            loperand1_values[lane] = loperand1;
            loperand2_values[lane] = loperand2;
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    out_streams[shape * 1 + 0][lane] += loperand0_values[lane] * grad_ref_x[q * N_SHAPE + shape] + loperand1_values[lane] * grad_ref_y[q * N_SHAPE + shape] + loperand2_values[lane] * grad_ref_z[q * N_SHAPE + shape];
                }
            }
        }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void scalar_potential_d3_simplex_tet4_gradient_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate4,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate5,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate6,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate7,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate8,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t kappa,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 1],
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 1]
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
        { const int q = 0;  // constant-P1 simplex
            const scalar_t qw = q_weight[q];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const scalar_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const scalar_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const scalar_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const scalar_t jacobian_adjugate_lane4 = jacobian_adjugate4[geometry_offset];
            const scalar_t jacobian_adjugate_lane5 = jacobian_adjugate5[geometry_offset];
            const scalar_t jacobian_adjugate_lane6 = jacobian_adjugate6[geometry_offset];
            const scalar_t jacobian_adjugate_lane7 = jacobian_adjugate7[geometry_offset];
            const scalar_t jacobian_adjugate_lane8 = jacobian_adjugate8[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const scalar_t grad_u_ref0 = -(u_streams[0 * 1 + 0][lane]) + u_streams[1 * 1 + 0][lane];
            const scalar_t grad_u_ref1 = -(u_streams[0 * 1 + 0][lane]) + u_streams[2 * 1 + 0][lane];
            const scalar_t grad_u_ref2 = -(u_streams[0 * 1 + 0][lane]) + u_streams[3 * 1 + 0][lane];
            const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
            const scalar_t grad_u0 = (grad_u_ref0 * jacobian_adjugate_lane0 + grad_u_ref1 * jacobian_adjugate_lane3 + grad_u_ref2 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            const scalar_t grad_u1 = (grad_u_ref0 * jacobian_adjugate_lane1 + grad_u_ref1 * jacobian_adjugate_lane4 + grad_u_ref2 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            const scalar_t grad_u2 = (grad_u_ref0 * jacobian_adjugate_lane2 + grad_u_ref1 * jacobian_adjugate_lane5 + grad_u_ref2 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t material0 = grad_u0*kappa;
        const scalar_t material1 = grad_u1*kappa;
        const scalar_t material2 = grad_u2*kappa;
        const scalar_t loperand0 = qw * (material0 * jacobian_adjugate_lane0 + material1 * jacobian_adjugate_lane1 + material2 * jacobian_adjugate_lane2);
        const scalar_t loperand1 = qw * (material0 * jacobian_adjugate_lane3 + material1 * jacobian_adjugate_lane4 + material2 * jacobian_adjugate_lane5);
        const scalar_t loperand2 = qw * (material0 * jacobian_adjugate_lane6 + material1 * jacobian_adjugate_lane7 + material2 * jacobian_adjugate_lane8);
            out_streams[0 * 1 + 0][lane] += -(loperand0) - loperand1 - loperand2;
            out_streams[1 * 1 + 0][lane] += loperand0;
            out_streams[2 * 1 + 0][lane] += loperand1;
            out_streams[3 * 1 + 0][lane] += loperand2;
            }
        }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void scalar_potential_d3_simplex_tet4_metric_gradient_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT geom_metric0,
        const scalar_t *const SFEM_RESTRICT geom_metric1,
        const scalar_t *const SFEM_RESTRICT geom_metric2,
        const scalar_t *const SFEM_RESTRICT geom_metric3,
        const scalar_t *const SFEM_RESTRICT geom_metric4,
        const scalar_t *const SFEM_RESTRICT geom_metric5,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t kappa,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 1],
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 1]
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = lane;
            const scalar_t geom_metric_lane0 = geom_metric0[geometry_offset];
            const scalar_t geom_metric_lane1 = geom_metric1[geometry_offset];
            const scalar_t geom_metric_lane2 = geom_metric2[geometry_offset];
            const scalar_t geom_metric_lane3 = geom_metric3[geometry_offset];
            const scalar_t geom_metric_lane4 = geom_metric4[geometry_offset];
            const scalar_t geom_metric_lane5 = geom_metric5[geometry_offset];
            const scalar_t t0 = -u_streams[0 * 1 + 0][lane] + u_streams[1 * 1 + 0][lane];
            const scalar_t t1 = -u_streams[0 * 1 + 0][lane] + u_streams[2 * 1 + 0][lane];
            const scalar_t t2 = -u_streams[0 * 1 + 0][lane] + u_streams[3 * 1 + 0][lane];
            const scalar_t t3 = geom_metric_lane0*t0 + geom_metric_lane1*t1 + geom_metric_lane2*t2;
            const scalar_t t4 = geom_metric_lane1*t0 + geom_metric_lane3*t1 + geom_metric_lane4*t2;
            const scalar_t t5 = geom_metric_lane2*t0 + geom_metric_lane4*t1 + geom_metric_lane5*t2;
            out_streams[0 * 1 + 0][lane] += kappa*(-t3 - t4 - t5);
            out_streams[1 * 1 + 0][lane] += kappa*t3;
            out_streams[2 * 1 + 0][lane] += kappa*t4;
            out_streams[3 * 1 + 0][lane] += kappa*t5;
        }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void scalar_potential_d3_simplex_apply_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate4,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate5,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate6,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate7,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate8,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t kappa,
        const scalar_t *const SFEM_RESTRICT h_streams[N_SHAPE * 1],
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 1]
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
        for (int q = 0; q < N_QP; ++q) {
            const scalar_t qw = q_weight[q];
            scalar_t grad_h_ref0_values[VECTOR_SIZE];
            scalar_t grad_h_ref1_values[VECTOR_SIZE];
            scalar_t grad_h_ref2_values[VECTOR_SIZE];
            scalar_t loperand0_values[VECTOR_SIZE];
            scalar_t loperand1_values[VECTOR_SIZE];
            scalar_t loperand2_values[VECTOR_SIZE];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_h_ref0_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_h_ref1_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_h_ref2_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_h_ref0_values[lane] += h_streams[shape * 1 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_h_ref1_values[lane] += h_streams[shape * 1 + 0][lane] * grad_ref_y[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_h_ref2_values[lane] += h_streams[shape * 1 + 0][lane] * grad_ref_z[q * N_SHAPE + shape];
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const scalar_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const scalar_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const scalar_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const scalar_t jacobian_adjugate_lane4 = jacobian_adjugate4[geometry_offset];
            const scalar_t jacobian_adjugate_lane5 = jacobian_adjugate5[geometry_offset];
            const scalar_t jacobian_adjugate_lane6 = jacobian_adjugate6[geometry_offset];
            const scalar_t jacobian_adjugate_lane7 = jacobian_adjugate7[geometry_offset];
            const scalar_t jacobian_adjugate_lane8 = jacobian_adjugate8[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const scalar_t grad_h_ref0 = grad_h_ref0_values[lane];
            const scalar_t grad_h_ref1 = grad_h_ref1_values[lane];
            const scalar_t grad_h_ref2 = grad_h_ref2_values[lane];
        const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
        const scalar_t trial_grad0 = (grad_h_ref0 * jacobian_adjugate_lane0 + grad_h_ref1 * jacobian_adjugate_lane3 + grad_h_ref2 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t trial_grad1 = (grad_h_ref0 * jacobian_adjugate_lane1 + grad_h_ref1 * jacobian_adjugate_lane4 + grad_h_ref2 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t trial_grad2 = (grad_h_ref0 * jacobian_adjugate_lane2 + grad_h_ref1 * jacobian_adjugate_lane5 + grad_h_ref2 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t material0 = kappa*trial_grad0;
        const scalar_t material1 = kappa*trial_grad1;
        const scalar_t material2 = kappa*trial_grad2;
        const scalar_t loperand0 = qw * (material0 * jacobian_adjugate_lane0 + material1 * jacobian_adjugate_lane1 + material2 * jacobian_adjugate_lane2);
        const scalar_t loperand1 = qw * (material0 * jacobian_adjugate_lane3 + material1 * jacobian_adjugate_lane4 + material2 * jacobian_adjugate_lane5);
        const scalar_t loperand2 = qw * (material0 * jacobian_adjugate_lane6 + material1 * jacobian_adjugate_lane7 + material2 * jacobian_adjugate_lane8);
            loperand0_values[lane] = loperand0;
            loperand1_values[lane] = loperand1;
            loperand2_values[lane] = loperand2;
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    out_streams[shape * 1 + 0][lane] += loperand0_values[lane] * grad_ref_x[q * N_SHAPE + shape] + loperand1_values[lane] * grad_ref_y[q * N_SHAPE + shape] + loperand2_values[lane] * grad_ref_z[q * N_SHAPE + shape];
                }
            }
        }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void scalar_potential_d3_simplex_tet4_apply_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate4,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate5,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate6,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate7,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate8,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t kappa,
        const scalar_t *const SFEM_RESTRICT h_streams[N_SHAPE * 1],
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 1]
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
        { const int q = 0;  // constant-P1 simplex
            const scalar_t qw = q_weight[q];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const scalar_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const scalar_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const scalar_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const scalar_t jacobian_adjugate_lane4 = jacobian_adjugate4[geometry_offset];
            const scalar_t jacobian_adjugate_lane5 = jacobian_adjugate5[geometry_offset];
            const scalar_t jacobian_adjugate_lane6 = jacobian_adjugate6[geometry_offset];
            const scalar_t jacobian_adjugate_lane7 = jacobian_adjugate7[geometry_offset];
            const scalar_t jacobian_adjugate_lane8 = jacobian_adjugate8[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const scalar_t grad_h_ref0 = -(h_streams[0 * 1 + 0][lane]) + h_streams[1 * 1 + 0][lane];
            const scalar_t grad_h_ref1 = -(h_streams[0 * 1 + 0][lane]) + h_streams[2 * 1 + 0][lane];
            const scalar_t grad_h_ref2 = -(h_streams[0 * 1 + 0][lane]) + h_streams[3 * 1 + 0][lane];
            const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
            const scalar_t trial_grad0 = (grad_h_ref0 * jacobian_adjugate_lane0 + grad_h_ref1 * jacobian_adjugate_lane3 + grad_h_ref2 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            const scalar_t trial_grad1 = (grad_h_ref0 * jacobian_adjugate_lane1 + grad_h_ref1 * jacobian_adjugate_lane4 + grad_h_ref2 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            const scalar_t trial_grad2 = (grad_h_ref0 * jacobian_adjugate_lane2 + grad_h_ref1 * jacobian_adjugate_lane5 + grad_h_ref2 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t material0 = kappa*trial_grad0;
        const scalar_t material1 = kappa*trial_grad1;
        const scalar_t material2 = kappa*trial_grad2;
        const scalar_t loperand0 = qw * (material0 * jacobian_adjugate_lane0 + material1 * jacobian_adjugate_lane1 + material2 * jacobian_adjugate_lane2);
        const scalar_t loperand1 = qw * (material0 * jacobian_adjugate_lane3 + material1 * jacobian_adjugate_lane4 + material2 * jacobian_adjugate_lane5);
        const scalar_t loperand2 = qw * (material0 * jacobian_adjugate_lane6 + material1 * jacobian_adjugate_lane7 + material2 * jacobian_adjugate_lane8);
            out_streams[0 * 1 + 0][lane] += -(loperand0) - loperand1 - loperand2;
            out_streams[1 * 1 + 0][lane] += loperand0;
            out_streams[2 * 1 + 0][lane] += loperand1;
            out_streams[3 * 1 + 0][lane] += loperand2;
            }
        }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void scalar_potential_d3_simplex_tet4_metric_apply_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT geom_metric0,
        const scalar_t *const SFEM_RESTRICT geom_metric1,
        const scalar_t *const SFEM_RESTRICT geom_metric2,
        const scalar_t *const SFEM_RESTRICT geom_metric3,
        const scalar_t *const SFEM_RESTRICT geom_metric4,
        const scalar_t *const SFEM_RESTRICT geom_metric5,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t kappa,
        const scalar_t *const SFEM_RESTRICT h_streams[N_SHAPE * 1],
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 1]
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = lane;
            const scalar_t geom_metric_lane0 = geom_metric0[geometry_offset];
            const scalar_t geom_metric_lane1 = geom_metric1[geometry_offset];
            const scalar_t geom_metric_lane2 = geom_metric2[geometry_offset];
            const scalar_t geom_metric_lane3 = geom_metric3[geometry_offset];
            const scalar_t geom_metric_lane4 = geom_metric4[geometry_offset];
            const scalar_t geom_metric_lane5 = geom_metric5[geometry_offset];
            const scalar_t t0 = -h_streams[0 * 1 + 0][lane] + h_streams[1 * 1 + 0][lane];
            const scalar_t t1 = -h_streams[0 * 1 + 0][lane] + h_streams[2 * 1 + 0][lane];
            const scalar_t t2 = -h_streams[0 * 1 + 0][lane] + h_streams[3 * 1 + 0][lane];
            const scalar_t t3 = geom_metric_lane0*t0 + geom_metric_lane1*t1 + geom_metric_lane2*t2;
            const scalar_t t4 = geom_metric_lane1*t0 + geom_metric_lane3*t1 + geom_metric_lane4*t2;
            const scalar_t t5 = geom_metric_lane2*t0 + geom_metric_lane4*t1 + geom_metric_lane5*t2;
            out_streams[0 * 1 + 0][lane] += kappa*(-t3 - t4 - t5);
            out_streams[1 * 1 + 0][lane] += kappa*t3;
            out_streams[2 * 1 + 0][lane] += kappa*t4;
            out_streams[3 * 1 + 0][lane] += kappa*t5;
        }
}

} // namespace codegen
} // namespace sfem

#endif
