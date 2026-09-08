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
static SFEM_INLINE void laplace_d3_simplex_objective_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR adj4,
        const s_t *const RSTR adj5,
        const s_t *const RSTR adj6,
        const s_t *const RSTR adj7,
        const s_t *const RSTR adj8,
        const s_t *const RSTR det0,
        const s_t *const RSTR grad_ref_x,
        const s_t *const RSTR grad_ref_y,
        const s_t *const RSTR grad_ref_z,
        const s_t *const RSTR q_weight,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 1],
        s_t *const RSTR value
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
        for (int q = 0; q < NQ; ++q) {
            const s_t qw = q_weight[q];
            s_t gu_ref0_values[VS];
            s_t gu_ref1_values[VS];
            s_t gu_ref2_values[VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
                gu_ref0_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
                gu_ref1_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
                gu_ref2_values[lane] = s_t(0);
            }
            for (int shape = 0; shape < NS; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    gu_ref0_values[lane] += u_streams[shape * 1 + 0][lane] * grad_ref_x[q * NS + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    gu_ref1_values[lane] += u_streams[shape * 1 + 0][lane] * grad_ref_y[q * NS + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    gu_ref2_values[lane] += u_streams[shape * 1 + 0][lane] * grad_ref_z[q * NS + shape];
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t adj_lane0 = adj0[goff];
            const s_t adj_lane1 = adj1[goff];
            const s_t adj_lane2 = adj2[goff];
            const s_t adj_lane3 = adj3[goff];
            const s_t adj_lane4 = adj4[goff];
            const s_t adj_lane5 = adj5[goff];
            const s_t adj_lane6 = adj6[goff];
            const s_t adj_lane7 = adj7[goff];
            const s_t adj_lane8 = adj8[goff];
            const s_t det_lane0 = det0[goff];
            const s_t gu_ref0 = gu_ref0_values[lane];
            const s_t gu_ref1 = gu_ref1_values[lane];
            const s_t gu_ref2 = gu_ref2_values[lane];
        const s_t idet = s_t(1) / det_lane0;
        const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane3 + gu_ref2 * adj_lane6) * idet;
        const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane4 + gu_ref2 * adj_lane7) * idet;
        const s_t gu2 = (gu_ref0 * adj_lane2 + gu_ref1 * adj_lane5 + gu_ref2 * adj_lane8) * idet;
        value[lane] += qw * det_lane0 * (((s_t(1) / s_t(2)))*kappa*(pow_2(gu0) + pow_2(gu1) + pow_2(gu2)));
            }
        }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void laplace_d3_simplex_tet4_objective_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR adj4,
        const s_t *const RSTR adj5,
        const s_t *const RSTR adj6,
        const s_t *const RSTR adj7,
        const s_t *const RSTR adj8,
        const s_t *const RSTR det0,
        const s_t *const RSTR q_weight,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 1],
        s_t *const RSTR value
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
        { const int q = 0;  // constant-P1 simplex
            const s_t qw = q_weight[q];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t adj_lane0 = adj0[goff];
            const s_t adj_lane1 = adj1[goff];
            const s_t adj_lane2 = adj2[goff];
            const s_t adj_lane3 = adj3[goff];
            const s_t adj_lane4 = adj4[goff];
            const s_t adj_lane5 = adj5[goff];
            const s_t adj_lane6 = adj6[goff];
            const s_t adj_lane7 = adj7[goff];
            const s_t adj_lane8 = adj8[goff];
            const s_t det_lane0 = det0[goff];
            const s_t gu_ref0 = -(u_streams[0 * 1 + 0][lane]) + u_streams[1 * 1 + 0][lane];
            const s_t gu_ref1 = -(u_streams[0 * 1 + 0][lane]) + u_streams[2 * 1 + 0][lane];
            const s_t gu_ref2 = -(u_streams[0 * 1 + 0][lane]) + u_streams[3 * 1 + 0][lane];
            const s_t idet = s_t(1) / det_lane0;
            const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane3 + gu_ref2 * adj_lane6) * idet;
            const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane4 + gu_ref2 * adj_lane7) * idet;
            const s_t gu2 = (gu_ref0 * adj_lane2 + gu_ref1 * adj_lane5 + gu_ref2 * adj_lane8) * idet;
        value[lane] += qw * det_lane0 * (((s_t(1) / s_t(2)))*kappa*(pow_2(gu0) + pow_2(gu1) + pow_2(gu2)));
            }
        }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void laplace_d3_simplex_tet4_metric_objective_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR geom_metric0,
        const s_t *const RSTR geom_metric1,
        const s_t *const RSTR geom_metric2,
        const s_t *const RSTR geom_metric3,
        const s_t *const RSTR geom_metric4,
        const s_t *const RSTR geom_metric5,
        const s_t *const RSTR q_weight,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 1],
        s_t *const RSTR value
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
            const ptrdiff_t goff = lane;
            const s_t geom_metric_lane0 = geom_metric0[goff];
            const s_t geom_metric_lane1 = geom_metric1[goff];
            const s_t geom_metric_lane2 = geom_metric2[goff];
            const s_t geom_metric_lane3 = geom_metric3[goff];
            const s_t geom_metric_lane4 = geom_metric4[goff];
            const s_t geom_metric_lane5 = geom_metric5[goff];
            const s_t t0 = -u_streams[0 * 1 + 0][lane] + u_streams[1 * 1 + 0][lane];
            const s_t t1 = -u_streams[0 * 1 + 0][lane] + u_streams[2 * 1 + 0][lane];
            const s_t t2 = -u_streams[0 * 1 + 0][lane] + u_streams[3 * 1 + 0][lane];
            const s_t t3 = geom_metric_lane0*t0 + geom_metric_lane1*t1 + geom_metric_lane2*t2;
            const s_t t4 = geom_metric_lane1*t0 + geom_metric_lane3*t1 + geom_metric_lane4*t2;
            const s_t t5 = geom_metric_lane2*t0 + geom_metric_lane4*t1 + geom_metric_lane5*t2;
            value[lane] += ((s_t(1) / s_t(2)))*kappa*(t3*(-u_streams[0 * 1 + 0][lane] + u_streams[1 * 1 + 0][lane]) + t4*(-u_streams[0 * 1 + 0][lane] + u_streams[2 * 1 + 0][lane]) + t5*(-u_streams[0 * 1 + 0][lane] + u_streams[3 * 1 + 0][lane]));
        }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void laplace_d3_simplex_gradient_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR adj4,
        const s_t *const RSTR adj5,
        const s_t *const RSTR adj6,
        const s_t *const RSTR adj7,
        const s_t *const RSTR adj8,
        const s_t *const RSTR det0,
        const s_t *const RSTR grad_ref_x,
        const s_t *const RSTR grad_ref_y,
        const s_t *const RSTR grad_ref_z,
        const s_t *const RSTR q_weight,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 1],
        s_t *const RSTR out_streams[NS * 1]
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
        for (int q = 0; q < NQ; ++q) {
            const s_t qw = q_weight[q];
            s_t gu_ref0_values[VS];
            s_t gu_ref1_values[VS];
            s_t gu_ref2_values[VS];
            s_t loperand0_values[VS];
            s_t loperand1_values[VS];
            s_t loperand2_values[VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
                gu_ref0_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
                gu_ref1_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
                gu_ref2_values[lane] = s_t(0);
            }
            for (int shape = 0; shape < NS; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    gu_ref0_values[lane] += u_streams[shape * 1 + 0][lane] * grad_ref_x[q * NS + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    gu_ref1_values[lane] += u_streams[shape * 1 + 0][lane] * grad_ref_y[q * NS + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    gu_ref2_values[lane] += u_streams[shape * 1 + 0][lane] * grad_ref_z[q * NS + shape];
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t adj_lane0 = adj0[goff];
            const s_t adj_lane1 = adj1[goff];
            const s_t adj_lane2 = adj2[goff];
            const s_t adj_lane3 = adj3[goff];
            const s_t adj_lane4 = adj4[goff];
            const s_t adj_lane5 = adj5[goff];
            const s_t adj_lane6 = adj6[goff];
            const s_t adj_lane7 = adj7[goff];
            const s_t adj_lane8 = adj8[goff];
            const s_t det_lane0 = det0[goff];
            const s_t gu_ref0 = gu_ref0_values[lane];
            const s_t gu_ref1 = gu_ref1_values[lane];
            const s_t gu_ref2 = gu_ref2_values[lane];
        const s_t idet = s_t(1) / det_lane0;
        const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane3 + gu_ref2 * adj_lane6) * idet;
        const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane4 + gu_ref2 * adj_lane7) * idet;
        const s_t gu2 = (gu_ref0 * adj_lane2 + gu_ref1 * adj_lane5 + gu_ref2 * adj_lane8) * idet;
        const s_t material0 = gu0*kappa;
        const s_t material1 = gu1*kappa;
        const s_t material2 = gu2*kappa;
        const s_t loperand0 = qw * (material0 * adj_lane0 + material1 * adj_lane1 + material2 * adj_lane2);
        const s_t loperand1 = qw * (material0 * adj_lane3 + material1 * adj_lane4 + material2 * adj_lane5);
        const s_t loperand2 = qw * (material0 * adj_lane6 + material1 * adj_lane7 + material2 * adj_lane8);
            loperand0_values[lane] = loperand0;
            loperand1_values[lane] = loperand1;
            loperand2_values[lane] = loperand2;
            }
            for (int shape = 0; shape < NS; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    out_streams[shape * 1 + 0][lane] += loperand0_values[lane] * grad_ref_x[q * NS + shape] + loperand1_values[lane] * grad_ref_y[q * NS + shape] + loperand2_values[lane] * grad_ref_z[q * NS + shape];
                }
            }
        }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void laplace_d3_simplex_tet4_gradient_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR adj4,
        const s_t *const RSTR adj5,
        const s_t *const RSTR adj6,
        const s_t *const RSTR adj7,
        const s_t *const RSTR adj8,
        const s_t *const RSTR det0,
        const s_t *const RSTR q_weight,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 1],
        s_t *const RSTR out_streams[NS * 1]
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
        { const int q = 0;  // constant-P1 simplex
            const s_t qw = q_weight[q];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t adj_lane0 = adj0[goff];
            const s_t adj_lane1 = adj1[goff];
            const s_t adj_lane2 = adj2[goff];
            const s_t adj_lane3 = adj3[goff];
            const s_t adj_lane4 = adj4[goff];
            const s_t adj_lane5 = adj5[goff];
            const s_t adj_lane6 = adj6[goff];
            const s_t adj_lane7 = adj7[goff];
            const s_t adj_lane8 = adj8[goff];
            const s_t det_lane0 = det0[goff];
            const s_t gu_ref0 = -(u_streams[0 * 1 + 0][lane]) + u_streams[1 * 1 + 0][lane];
            const s_t gu_ref1 = -(u_streams[0 * 1 + 0][lane]) + u_streams[2 * 1 + 0][lane];
            const s_t gu_ref2 = -(u_streams[0 * 1 + 0][lane]) + u_streams[3 * 1 + 0][lane];
            const s_t idet = s_t(1) / det_lane0;
            const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane3 + gu_ref2 * adj_lane6) * idet;
            const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane4 + gu_ref2 * adj_lane7) * idet;
            const s_t gu2 = (gu_ref0 * adj_lane2 + gu_ref1 * adj_lane5 + gu_ref2 * adj_lane8) * idet;
        const s_t material0 = gu0*kappa;
        const s_t material1 = gu1*kappa;
        const s_t material2 = gu2*kappa;
        const s_t loperand0 = qw * (material0 * adj_lane0 + material1 * adj_lane1 + material2 * adj_lane2);
        const s_t loperand1 = qw * (material0 * adj_lane3 + material1 * adj_lane4 + material2 * adj_lane5);
        const s_t loperand2 = qw * (material0 * adj_lane6 + material1 * adj_lane7 + material2 * adj_lane8);
            out_streams[0 * 1 + 0][lane] += -(loperand0) - loperand1 - loperand2;
            out_streams[1 * 1 + 0][lane] += loperand0;
            out_streams[2 * 1 + 0][lane] += loperand1;
            out_streams[3 * 1 + 0][lane] += loperand2;
            }
        }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void laplace_d3_simplex_tet4_metric_gradient_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR geom_metric0,
        const s_t *const RSTR geom_metric1,
        const s_t *const RSTR geom_metric2,
        const s_t *const RSTR geom_metric3,
        const s_t *const RSTR geom_metric4,
        const s_t *const RSTR geom_metric5,
        const s_t *const RSTR q_weight,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 1],
        s_t *const RSTR out_streams[NS * 1]
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
            const ptrdiff_t goff = lane;
            const s_t geom_metric_lane0 = geom_metric0[goff];
            const s_t geom_metric_lane1 = geom_metric1[goff];
            const s_t geom_metric_lane2 = geom_metric2[goff];
            const s_t geom_metric_lane3 = geom_metric3[goff];
            const s_t geom_metric_lane4 = geom_metric4[goff];
            const s_t geom_metric_lane5 = geom_metric5[goff];
            const s_t t0 = -u_streams[0 * 1 + 0][lane] + u_streams[1 * 1 + 0][lane];
            const s_t t1 = -u_streams[0 * 1 + 0][lane] + u_streams[2 * 1 + 0][lane];
            const s_t t2 = -u_streams[0 * 1 + 0][lane] + u_streams[3 * 1 + 0][lane];
            const s_t t3 = geom_metric_lane0*t0 + geom_metric_lane1*t1 + geom_metric_lane2*t2;
            const s_t t4 = geom_metric_lane1*t0 + geom_metric_lane3*t1 + geom_metric_lane4*t2;
            const s_t t5 = geom_metric_lane2*t0 + geom_metric_lane4*t1 + geom_metric_lane5*t2;
            out_streams[0 * 1 + 0][lane] += kappa*(-t3 - t4 - t5);
            out_streams[1 * 1 + 0][lane] += kappa*t3;
            out_streams[2 * 1 + 0][lane] += kappa*t4;
            out_streams[3 * 1 + 0][lane] += kappa*t5;
        }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void laplace_d3_simplex_apply_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR adj4,
        const s_t *const RSTR adj5,
        const s_t *const RSTR adj6,
        const s_t *const RSTR adj7,
        const s_t *const RSTR adj8,
        const s_t *const RSTR det0,
        const s_t *const RSTR grad_ref_x,
        const s_t *const RSTR grad_ref_y,
        const s_t *const RSTR grad_ref_z,
        const s_t *const RSTR q_weight,
        const s_t kappa,
        const s_t *const RSTR h_streams[NS * 1],
        s_t *const RSTR out_streams[NS * 1]
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
        for (int q = 0; q < NQ; ++q) {
            const s_t qw = q_weight[q];
            s_t grad_h_ref0_values[VS];
            s_t grad_h_ref1_values[VS];
            s_t grad_h_ref2_values[VS];
            s_t loperand0_values[VS];
            s_t loperand1_values[VS];
            s_t loperand2_values[VS];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
                grad_h_ref0_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
                grad_h_ref1_values[lane] = s_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
                grad_h_ref2_values[lane] = s_t(0);
            }
            for (int shape = 0; shape < NS; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    grad_h_ref0_values[lane] += h_streams[shape * 1 + 0][lane] * grad_ref_x[q * NS + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    grad_h_ref1_values[lane] += h_streams[shape * 1 + 0][lane] * grad_ref_y[q * NS + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    grad_h_ref2_values[lane] += h_streams[shape * 1 + 0][lane] * grad_ref_z[q * NS + shape];
                }
            }
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t adj_lane0 = adj0[goff];
            const s_t adj_lane1 = adj1[goff];
            const s_t adj_lane2 = adj2[goff];
            const s_t adj_lane3 = adj3[goff];
            const s_t adj_lane4 = adj4[goff];
            const s_t adj_lane5 = adj5[goff];
            const s_t adj_lane6 = adj6[goff];
            const s_t adj_lane7 = adj7[goff];
            const s_t adj_lane8 = adj8[goff];
            const s_t det_lane0 = det0[goff];
            const s_t grad_h_ref0 = grad_h_ref0_values[lane];
            const s_t grad_h_ref1 = grad_h_ref1_values[lane];
            const s_t grad_h_ref2 = grad_h_ref2_values[lane];
        const s_t idet = s_t(1) / det_lane0;
        const s_t trial_grad0 = (grad_h_ref0 * adj_lane0 + grad_h_ref1 * adj_lane3 + grad_h_ref2 * adj_lane6) * idet;
        const s_t trial_grad1 = (grad_h_ref0 * adj_lane1 + grad_h_ref1 * adj_lane4 + grad_h_ref2 * adj_lane7) * idet;
        const s_t trial_grad2 = (grad_h_ref0 * adj_lane2 + grad_h_ref1 * adj_lane5 + grad_h_ref2 * adj_lane8) * idet;
        const s_t material0 = kappa*trial_grad0;
        const s_t material1 = kappa*trial_grad1;
        const s_t material2 = kappa*trial_grad2;
        const s_t loperand0 = qw * (material0 * adj_lane0 + material1 * adj_lane1 + material2 * adj_lane2);
        const s_t loperand1 = qw * (material0 * adj_lane3 + material1 * adj_lane4 + material2 * adj_lane5);
        const s_t loperand2 = qw * (material0 * adj_lane6 + material1 * adj_lane7 + material2 * adj_lane8);
            loperand0_values[lane] = loperand0;
            loperand1_values[lane] = loperand1;
            loperand2_values[lane] = loperand2;
            }
            for (int shape = 0; shape < NS; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < ne; ++lane) {
                    out_streams[shape * 1 + 0][lane] += loperand0_values[lane] * grad_ref_x[q * NS + shape] + loperand1_values[lane] * grad_ref_y[q * NS + shape] + loperand2_values[lane] * grad_ref_z[q * NS + shape];
                }
            }
        }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void laplace_d3_simplex_tet4_apply_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR adj4,
        const s_t *const RSTR adj5,
        const s_t *const RSTR adj6,
        const s_t *const RSTR adj7,
        const s_t *const RSTR adj8,
        const s_t *const RSTR det0,
        const s_t *const RSTR q_weight,
        const s_t kappa,
        const s_t *const RSTR h_streams[NS * 1],
        s_t *const RSTR out_streams[NS * 1]
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
        { const int q = 0;  // constant-P1 simplex
            const s_t qw = q_weight[q];
            #pragma omp simd
            for (int lane = 0; lane < ne; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t adj_lane0 = adj0[goff];
            const s_t adj_lane1 = adj1[goff];
            const s_t adj_lane2 = adj2[goff];
            const s_t adj_lane3 = adj3[goff];
            const s_t adj_lane4 = adj4[goff];
            const s_t adj_lane5 = adj5[goff];
            const s_t adj_lane6 = adj6[goff];
            const s_t adj_lane7 = adj7[goff];
            const s_t adj_lane8 = adj8[goff];
            const s_t det_lane0 = det0[goff];
            const s_t grad_h_ref0 = -(h_streams[0 * 1 + 0][lane]) + h_streams[1 * 1 + 0][lane];
            const s_t grad_h_ref1 = -(h_streams[0 * 1 + 0][lane]) + h_streams[2 * 1 + 0][lane];
            const s_t grad_h_ref2 = -(h_streams[0 * 1 + 0][lane]) + h_streams[3 * 1 + 0][lane];
            const s_t idet = s_t(1) / det_lane0;
            const s_t trial_grad0 = (grad_h_ref0 * adj_lane0 + grad_h_ref1 * adj_lane3 + grad_h_ref2 * adj_lane6) * idet;
            const s_t trial_grad1 = (grad_h_ref0 * adj_lane1 + grad_h_ref1 * adj_lane4 + grad_h_ref2 * adj_lane7) * idet;
            const s_t trial_grad2 = (grad_h_ref0 * adj_lane2 + grad_h_ref1 * adj_lane5 + grad_h_ref2 * adj_lane8) * idet;
        const s_t material0 = kappa*trial_grad0;
        const s_t material1 = kappa*trial_grad1;
        const s_t material2 = kappa*trial_grad2;
        const s_t loperand0 = qw * (material0 * adj_lane0 + material1 * adj_lane1 + material2 * adj_lane2);
        const s_t loperand1 = qw * (material0 * adj_lane3 + material1 * adj_lane4 + material2 * adj_lane5);
        const s_t loperand2 = qw * (material0 * adj_lane6 + material1 * adj_lane7 + material2 * adj_lane8);
            out_streams[0 * 1 + 0][lane] += -(loperand0) - loperand1 - loperand2;
            out_streams[1 * 1 + 0][lane] += loperand0;
            out_streams[2 * 1 + 0][lane] += loperand1;
            out_streams[3 * 1 + 0][lane] += loperand2;
            }
        }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void laplace_d3_simplex_tet4_metric_apply_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR geom_metric0,
        const s_t *const RSTR geom_metric1,
        const s_t *const RSTR geom_metric2,
        const s_t *const RSTR geom_metric3,
        const s_t *const RSTR geom_metric4,
        const s_t *const RSTR geom_metric5,
        const s_t *const RSTR q_weight,
        const s_t kappa,
        const s_t *const RSTR h_streams[NS * 1],
        s_t *const RSTR out_streams[NS * 1]
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(VS > 0, "VS must be positive");
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
            const ptrdiff_t goff = lane;
            const s_t geom_metric_lane0 = geom_metric0[goff];
            const s_t geom_metric_lane1 = geom_metric1[goff];
            const s_t geom_metric_lane2 = geom_metric2[goff];
            const s_t geom_metric_lane3 = geom_metric3[goff];
            const s_t geom_metric_lane4 = geom_metric4[goff];
            const s_t geom_metric_lane5 = geom_metric5[goff];
            const s_t t0 = -h_streams[0 * 1 + 0][lane] + h_streams[1 * 1 + 0][lane];
            const s_t t1 = -h_streams[0 * 1 + 0][lane] + h_streams[2 * 1 + 0][lane];
            const s_t t2 = -h_streams[0 * 1 + 0][lane] + h_streams[3 * 1 + 0][lane];
            const s_t t3 = geom_metric_lane0*t0 + geom_metric_lane1*t1 + geom_metric_lane2*t2;
            const s_t t4 = geom_metric_lane1*t0 + geom_metric_lane3*t1 + geom_metric_lane4*t2;
            const s_t t5 = geom_metric_lane2*t0 + geom_metric_lane4*t1 + geom_metric_lane5*t2;
            out_streams[0 * 1 + 0][lane] += kappa*(-t3 - t4 - t5);
            out_streams[1 * 1 + 0][lane] += kappa*t3;
            out_streams[2 * 1 + 0][lane] += kappa*t4;
            out_streams[3 * 1 + 0][lane] += kappa*t5;
        }
}

} // namespace codegen
} // namespace sfem

#endif
