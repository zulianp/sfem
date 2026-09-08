#ifndef LAPLACE_D2_SIMPLEX_LOCAL_HPP
#define LAPLACE_D2_SIMPLEX_LOCAL_HPP
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
static SFEM_INLINE void laplace_d2_simplex_objective_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT jacobian_adjugate0,
        const s_t *const SFEM_RESTRICT jacobian_adjugate1,
        const s_t *const SFEM_RESTRICT jacobian_adjugate2,
        const s_t *const SFEM_RESTRICT jacobian_adjugate3,
        const s_t *const SFEM_RESTRICT jacobian_determinant0,
        const s_t *const SFEM_RESTRICT grad_ref_x,
        const s_t *const SFEM_RESTRICT grad_ref_y,
        const s_t *const SFEM_RESTRICT q_weight,
        const s_t kappa,
        const s_t *const SFEM_RESTRICT u_streams[NS * 1],
        s_t *const SFEM_RESTRICT value
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
        for (int q = 0; q < NQ; ++q) {
            const s_t qw = q_weight[q];
            s_t grad_u_ref0_values[VS];
            s_t grad_u_ref1_values[VS];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref0_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref1_values[lane] = s_t(0);
            }
            for (int shape = 0; shape < NS; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref0_values[lane] += u_streams[shape * 1 + 0][lane] * grad_ref_x[q * NS + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref1_values[lane] += u_streams[shape * 1 + 0][lane] * grad_ref_y[q * NS + shape];
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const s_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const s_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const s_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const s_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const s_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const s_t grad_u_ref0 = grad_u_ref0_values[lane];
            const s_t grad_u_ref1 = grad_u_ref1_values[lane];
        const s_t inv_jacobian_determinant = s_t(1) / jacobian_determinant_lane0;
        const s_t grad_u0 = (grad_u_ref0 * jacobian_adjugate_lane0 + grad_u_ref1 * jacobian_adjugate_lane2) * inv_jacobian_determinant;
        const s_t grad_u1 = (grad_u_ref0 * jacobian_adjugate_lane1 + grad_u_ref1 * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        value[lane] += qw * jacobian_determinant_lane0 * (((s_t(1) / s_t(2)))*kappa*(pow_2(grad_u0) + pow_2(grad_u1)));
            }
        }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void laplace_d2_simplex_tri3_objective_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT jacobian_adjugate0,
        const s_t *const SFEM_RESTRICT jacobian_adjugate1,
        const s_t *const SFEM_RESTRICT jacobian_adjugate2,
        const s_t *const SFEM_RESTRICT jacobian_adjugate3,
        const s_t *const SFEM_RESTRICT jacobian_determinant0,
        const s_t *const SFEM_RESTRICT q_weight,
        const s_t kappa,
        const s_t *const SFEM_RESTRICT u_streams[NS * 1],
        s_t *const SFEM_RESTRICT value
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
        { const int q = 0;  // constant-P1 simplex
            const s_t qw = q_weight[q];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const s_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const s_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const s_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const s_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const s_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const s_t grad_u_ref0 = -(u_streams[0 * 1 + 0][lane]) + u_streams[1 * 1 + 0][lane];
            const s_t grad_u_ref1 = -(u_streams[0 * 1 + 0][lane]) + u_streams[2 * 1 + 0][lane];
            const s_t inv_jacobian_determinant = s_t(1) / jacobian_determinant_lane0;
            const s_t grad_u0 = (grad_u_ref0 * jacobian_adjugate_lane0 + grad_u_ref1 * jacobian_adjugate_lane2) * inv_jacobian_determinant;
            const s_t grad_u1 = (grad_u_ref0 * jacobian_adjugate_lane1 + grad_u_ref1 * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        value[lane] += qw * jacobian_determinant_lane0 * (((s_t(1) / s_t(2)))*kappa*(pow_2(grad_u0) + pow_2(grad_u1)));
            }
        }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void laplace_d2_simplex_tri3_metric_objective_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT geom_metric0,
        const s_t *const SFEM_RESTRICT geom_metric1,
        const s_t *const SFEM_RESTRICT geom_metric2,
        const s_t *const SFEM_RESTRICT q_weight,
        const s_t kappa,
        const s_t *const SFEM_RESTRICT u_streams[NS * 1],
        s_t *const SFEM_RESTRICT value
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = lane;
            const s_t geom_metric_lane0 = geom_metric0[geometry_offset];
            const s_t geom_metric_lane1 = geom_metric1[geometry_offset];
            const s_t geom_metric_lane2 = geom_metric2[geometry_offset];
            const s_t t0 = -u_streams[0 * 1 + 0][lane] + u_streams[1 * 1 + 0][lane];
            const s_t t1 = -u_streams[0 * 1 + 0][lane] + u_streams[2 * 1 + 0][lane];
            const s_t t2 = geom_metric_lane0*t0 + geom_metric_lane1*t1;
            const s_t t3 = geom_metric_lane1*t0 + geom_metric_lane2*t1;
            value[lane] += ((s_t(1) / s_t(2)))*kappa*(t2*(-u_streams[0 * 1 + 0][lane] + u_streams[1 * 1 + 0][lane]) + t3*(-u_streams[0 * 1 + 0][lane] + u_streams[2 * 1 + 0][lane]));
        }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void laplace_d2_simplex_gradient_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT jacobian_adjugate0,
        const s_t *const SFEM_RESTRICT jacobian_adjugate1,
        const s_t *const SFEM_RESTRICT jacobian_adjugate2,
        const s_t *const SFEM_RESTRICT jacobian_adjugate3,
        const s_t *const SFEM_RESTRICT jacobian_determinant0,
        const s_t *const SFEM_RESTRICT grad_ref_x,
        const s_t *const SFEM_RESTRICT grad_ref_y,
        const s_t *const SFEM_RESTRICT q_weight,
        const s_t kappa,
        const s_t *const SFEM_RESTRICT u_streams[NS * 1],
        s_t *const SFEM_RESTRICT out_streams[NS * 1]
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
        for (int q = 0; q < NQ; ++q) {
            const s_t qw = q_weight[q];
            s_t grad_u_ref0_values[VS];
            s_t grad_u_ref1_values[VS];
            s_t loperand0_values[VS];
            s_t loperand1_values[VS];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref0_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref1_values[lane] = s_t(0);
            }
            for (int shape = 0; shape < NS; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref0_values[lane] += u_streams[shape * 1 + 0][lane] * grad_ref_x[q * NS + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref1_values[lane] += u_streams[shape * 1 + 0][lane] * grad_ref_y[q * NS + shape];
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const s_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const s_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const s_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const s_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const s_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const s_t grad_u_ref0 = grad_u_ref0_values[lane];
            const s_t grad_u_ref1 = grad_u_ref1_values[lane];
        const s_t inv_jacobian_determinant = s_t(1) / jacobian_determinant_lane0;
        const s_t grad_u0 = (grad_u_ref0 * jacobian_adjugate_lane0 + grad_u_ref1 * jacobian_adjugate_lane2) * inv_jacobian_determinant;
        const s_t grad_u1 = (grad_u_ref0 * jacobian_adjugate_lane1 + grad_u_ref1 * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        const s_t material0 = grad_u0*kappa;
        const s_t material1 = grad_u1*kappa;
        const s_t loperand0 = qw * (material0 * jacobian_adjugate_lane0 + material1 * jacobian_adjugate_lane1);
        const s_t loperand1 = qw * (material0 * jacobian_adjugate_lane2 + material1 * jacobian_adjugate_lane3);
            loperand0_values[lane] = loperand0;
            loperand1_values[lane] = loperand1;
            }
            for (int shape = 0; shape < NS; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    out_streams[shape * 1 + 0][lane] += loperand0_values[lane] * grad_ref_x[q * NS + shape] + loperand1_values[lane] * grad_ref_y[q * NS + shape];
                }
            }
        }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void laplace_d2_simplex_tri3_gradient_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT jacobian_adjugate0,
        const s_t *const SFEM_RESTRICT jacobian_adjugate1,
        const s_t *const SFEM_RESTRICT jacobian_adjugate2,
        const s_t *const SFEM_RESTRICT jacobian_adjugate3,
        const s_t *const SFEM_RESTRICT jacobian_determinant0,
        const s_t *const SFEM_RESTRICT q_weight,
        const s_t kappa,
        const s_t *const SFEM_RESTRICT u_streams[NS * 1],
        s_t *const SFEM_RESTRICT out_streams[NS * 1]
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
        { const int q = 0;  // constant-P1 simplex
            const s_t qw = q_weight[q];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const s_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const s_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const s_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const s_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const s_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const s_t grad_u_ref0 = -(u_streams[0 * 1 + 0][lane]) + u_streams[1 * 1 + 0][lane];
            const s_t grad_u_ref1 = -(u_streams[0 * 1 + 0][lane]) + u_streams[2 * 1 + 0][lane];
            const s_t inv_jacobian_determinant = s_t(1) / jacobian_determinant_lane0;
            const s_t grad_u0 = (grad_u_ref0 * jacobian_adjugate_lane0 + grad_u_ref1 * jacobian_adjugate_lane2) * inv_jacobian_determinant;
            const s_t grad_u1 = (grad_u_ref0 * jacobian_adjugate_lane1 + grad_u_ref1 * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        const s_t material0 = grad_u0*kappa;
        const s_t material1 = grad_u1*kappa;
        const s_t loperand0 = qw * (material0 * jacobian_adjugate_lane0 + material1 * jacobian_adjugate_lane1);
        const s_t loperand1 = qw * (material0 * jacobian_adjugate_lane2 + material1 * jacobian_adjugate_lane3);
            out_streams[0 * 1 + 0][lane] += -(loperand0) - loperand1;
            out_streams[1 * 1 + 0][lane] += loperand0;
            out_streams[2 * 1 + 0][lane] += loperand1;
            }
        }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void laplace_d2_simplex_tri3_metric_gradient_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT geom_metric0,
        const s_t *const SFEM_RESTRICT geom_metric1,
        const s_t *const SFEM_RESTRICT geom_metric2,
        const s_t *const SFEM_RESTRICT q_weight,
        const s_t kappa,
        const s_t *const SFEM_RESTRICT u_streams[NS * 1],
        s_t *const SFEM_RESTRICT out_streams[NS * 1]
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = lane;
            const s_t geom_metric_lane0 = geom_metric0[geometry_offset];
            const s_t geom_metric_lane1 = geom_metric1[geometry_offset];
            const s_t geom_metric_lane2 = geom_metric2[geometry_offset];
            const s_t t0 = -u_streams[0 * 1 + 0][lane] + u_streams[1 * 1 + 0][lane];
            const s_t t1 = -u_streams[0 * 1 + 0][lane] + u_streams[2 * 1 + 0][lane];
            const s_t t2 = geom_metric_lane0*t0 + geom_metric_lane1*t1;
            const s_t t3 = geom_metric_lane1*t0 + geom_metric_lane2*t1;
            out_streams[0 * 1 + 0][lane] += kappa*(-t2 - t3);
            out_streams[1 * 1 + 0][lane] += kappa*t2;
            out_streams[2 * 1 + 0][lane] += kappa*t3;
        }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void laplace_d2_simplex_apply_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT jacobian_adjugate0,
        const s_t *const SFEM_RESTRICT jacobian_adjugate1,
        const s_t *const SFEM_RESTRICT jacobian_adjugate2,
        const s_t *const SFEM_RESTRICT jacobian_adjugate3,
        const s_t *const SFEM_RESTRICT jacobian_determinant0,
        const s_t *const SFEM_RESTRICT grad_ref_x,
        const s_t *const SFEM_RESTRICT grad_ref_y,
        const s_t *const SFEM_RESTRICT q_weight,
        const s_t kappa,
        const s_t *const SFEM_RESTRICT h_streams[NS * 1],
        s_t *const SFEM_RESTRICT out_streams[NS * 1]
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
        for (int q = 0; q < NQ; ++q) {
            const s_t qw = q_weight[q];
            s_t grad_h_ref0_values[VS];
            s_t grad_h_ref1_values[VS];
            s_t loperand0_values[VS];
            s_t loperand1_values[VS];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_h_ref0_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_h_ref1_values[lane] = s_t(0);
            }
            for (int shape = 0; shape < NS; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_h_ref0_values[lane] += h_streams[shape * 1 + 0][lane] * grad_ref_x[q * NS + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_h_ref1_values[lane] += h_streams[shape * 1 + 0][lane] * grad_ref_y[q * NS + shape];
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const s_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const s_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const s_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const s_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const s_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const s_t grad_h_ref0 = grad_h_ref0_values[lane];
            const s_t grad_h_ref1 = grad_h_ref1_values[lane];
        const s_t inv_jacobian_determinant = s_t(1) / jacobian_determinant_lane0;
        const s_t trial_grad0 = (grad_h_ref0 * jacobian_adjugate_lane0 + grad_h_ref1 * jacobian_adjugate_lane2) * inv_jacobian_determinant;
        const s_t trial_grad1 = (grad_h_ref0 * jacobian_adjugate_lane1 + grad_h_ref1 * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        const s_t material0 = kappa*trial_grad0;
        const s_t material1 = kappa*trial_grad1;
        const s_t loperand0 = qw * (material0 * jacobian_adjugate_lane0 + material1 * jacobian_adjugate_lane1);
        const s_t loperand1 = qw * (material0 * jacobian_adjugate_lane2 + material1 * jacobian_adjugate_lane3);
            loperand0_values[lane] = loperand0;
            loperand1_values[lane] = loperand1;
            }
            for (int shape = 0; shape < NS; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    out_streams[shape * 1 + 0][lane] += loperand0_values[lane] * grad_ref_x[q * NS + shape] + loperand1_values[lane] * grad_ref_y[q * NS + shape];
                }
            }
        }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void laplace_d2_simplex_tri3_apply_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT jacobian_adjugate0,
        const s_t *const SFEM_RESTRICT jacobian_adjugate1,
        const s_t *const SFEM_RESTRICT jacobian_adjugate2,
        const s_t *const SFEM_RESTRICT jacobian_adjugate3,
        const s_t *const SFEM_RESTRICT jacobian_determinant0,
        const s_t *const SFEM_RESTRICT q_weight,
        const s_t kappa,
        const s_t *const SFEM_RESTRICT h_streams[NS * 1],
        s_t *const SFEM_RESTRICT out_streams[NS * 1]
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
        { const int q = 0;  // constant-P1 simplex
            const s_t qw = q_weight[q];
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const s_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const s_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const s_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const s_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const s_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            const s_t grad_h_ref0 = -(h_streams[0 * 1 + 0][lane]) + h_streams[1 * 1 + 0][lane];
            const s_t grad_h_ref1 = -(h_streams[0 * 1 + 0][lane]) + h_streams[2 * 1 + 0][lane];
            const s_t inv_jacobian_determinant = s_t(1) / jacobian_determinant_lane0;
            const s_t trial_grad0 = (grad_h_ref0 * jacobian_adjugate_lane0 + grad_h_ref1 * jacobian_adjugate_lane2) * inv_jacobian_determinant;
            const s_t trial_grad1 = (grad_h_ref0 * jacobian_adjugate_lane1 + grad_h_ref1 * jacobian_adjugate_lane3) * inv_jacobian_determinant;
        const s_t material0 = kappa*trial_grad0;
        const s_t material1 = kappa*trial_grad1;
        const s_t loperand0 = qw * (material0 * jacobian_adjugate_lane0 + material1 * jacobian_adjugate_lane1);
        const s_t loperand1 = qw * (material0 * jacobian_adjugate_lane2 + material1 * jacobian_adjugate_lane3);
            out_streams[0 * 1 + 0][lane] += -(loperand0) - loperand1;
            out_streams[1 * 1 + 0][lane] += loperand0;
            out_streams[2 * 1 + 0][lane] += loperand1;
            }
        }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void laplace_d2_simplex_tri3_metric_apply_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const SFEM_RESTRICT geom_metric0,
        const s_t *const SFEM_RESTRICT geom_metric1,
        const s_t *const SFEM_RESTRICT geom_metric2,
        const s_t *const SFEM_RESTRICT q_weight,
        const s_t kappa,
        const s_t *const SFEM_RESTRICT h_streams[NS * 1],
        s_t *const SFEM_RESTRICT out_streams[NS * 1]
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = lane;
            const s_t geom_metric_lane0 = geom_metric0[geometry_offset];
            const s_t geom_metric_lane1 = geom_metric1[geometry_offset];
            const s_t geom_metric_lane2 = geom_metric2[geometry_offset];
            const s_t t0 = -h_streams[0 * 1 + 0][lane] + h_streams[1 * 1 + 0][lane];
            const s_t t1 = -h_streams[0 * 1 + 0][lane] + h_streams[2 * 1 + 0][lane];
            const s_t t2 = geom_metric_lane0*t0 + geom_metric_lane1*t1;
            const s_t t3 = geom_metric_lane1*t0 + geom_metric_lane2*t1;
            out_streams[0 * 1 + 0][lane] += kappa*(-t2 - t3);
            out_streams[1 * 1 + 0][lane] += kappa*t2;
            out_streams[2 * 1 + 0][lane] += kappa*t3;
        }
}

} // namespace codegen
} // namespace sfem

#endif
