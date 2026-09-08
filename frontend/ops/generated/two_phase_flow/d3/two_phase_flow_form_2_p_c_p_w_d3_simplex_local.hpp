#ifndef TWO_PHASE_FLOW_FORM_2_P_C_P_W_D3_SIMPLEX_LOCAL_HPP
#define TWO_PHASE_FLOW_FORM_2_P_C_P_W_D3_SIMPLEX_LOCAL_HPP

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
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_simplex_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR shape,
        const s_t *const RSTR q_weight,
        s_t *const RSTR output[2 * NS]
) {
    static constexpr int ND = 3;
    static constexpr int NC = 2;
    for (int q = 0; q < NQ; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t det = determinant[goff];
        }
        for (int test = 0; test < NS; ++test) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const ptrdiff_t goff = q * geometry_stride + lane;
                const s_t det = determinant[goff];
                const s_t test_value = shape[q * NS + test];
            }
        }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_simplex_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR shape,
        const s_t *const RSTR q_weight,
        s_t output[2 * NS][VS]
) {
    static constexpr int ND = 3;
    static constexpr int NC = 2;
    for (int q = 0; q < NQ; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t det = determinant[goff];
        }
        for (int test = 0; test < NS; ++test) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const ptrdiff_t goff = q * geometry_stride + lane;
                const s_t det = determinant[goff];
                const s_t test_value = shape[q * NS + test];
            }
        }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_simplex_tet4_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR shape,
        const s_t *const RSTR q_weight,
        s_t *const RSTR output[2 * NS]
) {
    static constexpr int ND = 3;
    static constexpr int NC = 2;
    for (int q = 0; q < NQ; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t det = determinant[goff];
        }
        for (int test = 0; test < NS; ++test) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const ptrdiff_t goff = q * geometry_stride + lane;
                const s_t det = determinant[goff];
                const s_t test_value = shape[q * NS + test];
            }
        }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_simplex_tet4_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR shape,
        const s_t *const RSTR q_weight,
        s_t output[2 * NS][VS]
) {
    static constexpr int ND = 3;
    static constexpr int NC = 2;
    for (int q = 0; q < NQ; ++q) {
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t goff = q * geometry_stride + lane;
            const s_t det = determinant[goff];
        }
        for (int test = 0; test < NS; ++test) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const ptrdiff_t goff = q * geometry_stride + lane;
                const s_t det = determinant[goff];
                const s_t test_value = shape[q * NS + test];
            }
        }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_simplex_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR adjugate[9],
        const s_t *const RSTR shape,
        const s_t *const RSTR grad_ref_x,
        const s_t *const RSTR grad_ref_y,
        const s_t *const RSTR grad_ref_z,
        const s_t *const RSTR q_weight,
        const s_t *const RSTR current[2 * NS],
        const s_t *const RSTR direction[2 * NS],
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
        s_t *const RSTR output[2 * NS]
) {
    static constexpr int ND = 3;
    static constexpr int NC = 2;
    for (int q = 0; q < NQ; ++q) {
        s_t p_w_values[VS];
        s_t p_w_grad_0_ref_values[VS];
        s_t p_w_grad_1_ref_values[VS];
        s_t p_w_grad_2_ref_values[VS];
        s_t p_w_direction_values[VS];
        s_t p_c_values[VS];
        s_t p_c_grad_0_ref_values[VS];
        s_t p_c_grad_1_ref_values[VS];
        s_t p_c_grad_2_ref_values[VS];
        s_t p_c_direction_values[VS];
        s_t value_coeff1_values[VS];
        s_t grad_coeff1_0_values[VS];
        s_t grad_coeff1_1_values[VS];
        s_t grad_coeff1_2_values[VS];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_0_ref_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_1_ref_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_2_ref_values[lane] = s_t(0);
        }
        for (int trial = 0; trial < NS; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t coeff = current[trial * NC + 0][lane];
                p_w_values[lane] += coeff * shape[q * NS + trial];
                p_w_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
                p_w_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
                p_w_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_direction_values[lane] = s_t(0);
        }
        for (int trial = 0; trial < NS; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t coeff = direction[trial * NC + 0][lane];
                p_w_direction_values[lane] += coeff * shape[q * NS + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_0_ref_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_1_ref_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_2_ref_values[lane] = s_t(0);
        }
        for (int trial = 0; trial < NS; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t coeff = current[trial * NC + 1][lane];
                p_c_values[lane] += coeff * shape[q * NS + trial];
                p_c_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
                p_c_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
                p_c_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_direction_values[lane] = s_t(0);
        }
        for (int trial = 0; trial < NS; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t coeff = direction[trial * NC + 1][lane];
                p_c_direction_values[lane] += coeff * shape[q * NS + trial];
            }
        }
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
            const s_t p_w = p_w_values[lane];
            const s_t p_w_grad_0_ref = p_w_grad_0_ref_values[lane];
            const s_t p_w_grad_1_ref = p_w_grad_1_ref_values[lane];
            const s_t p_w_grad_2_ref = p_w_grad_2_ref_values[lane];
            const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
            const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
            const s_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
            const s_t p_w_direction = p_w_direction_values[lane];
            const s_t p_c = p_c_values[lane];
            const s_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
            const s_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
            const s_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
            const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
            const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
            const s_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
            const s_t p_c_direction = p_c_direction_values[lane];
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
            value_coeff1_values[lane] = value_coeff1;
            grad_coeff1_0_values[lane] = grad_coeff1_0;
            grad_coeff1_1_values[lane] = grad_coeff1_1;
            grad_coeff1_2_values[lane] = grad_coeff1_2;
        }
        for (int test = 0; test < NS; ++test) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const ptrdiff_t goff = q * geometry_stride + lane;
                const s_t det = determinant[goff];
                const s_t test_value = shape[q * NS + test];
                const s_t adj0 = adjugate[0][goff];
                const s_t adj1 = adjugate[1][goff];
                const s_t adj2 = adjugate[2][goff];
                const s_t adj3 = adjugate[3][goff];
                const s_t adj4 = adjugate[4][goff];
                const s_t adj5 = adjugate[5][goff];
                const s_t adj6 = adjugate[6][goff];
                const s_t adj7 = adjugate[7][goff];
                const s_t adj8 = adjugate[8][goff];
                const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj3 + grad_ref_z[q * NS + test] * adj6) / det;
                const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj4 + grad_ref_z[q * NS + test] * adj7) / det;
                const s_t test_grad2 = (grad_ref_x[q * NS + test] * adj2 + grad_ref_y[q * NS + test] * adj5 + grad_ref_z[q * NS + test] * adj8) / det;
                output[test * NC + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
            }
        }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_simplex_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR adjugate[9],
        const s_t *const RSTR shape,
        const s_t *const RSTR grad_ref_x,
        const s_t *const RSTR grad_ref_y,
        const s_t *const RSTR grad_ref_z,
        const s_t *const RSTR q_weight,
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
    for (int q = 0; q < NQ; ++q) {
        s_t p_w_values[VS];
        s_t p_w_grad_0_ref_values[VS];
        s_t p_w_grad_1_ref_values[VS];
        s_t p_w_grad_2_ref_values[VS];
        s_t p_w_direction_values[VS];
        s_t p_c_values[VS];
        s_t p_c_grad_0_ref_values[VS];
        s_t p_c_grad_1_ref_values[VS];
        s_t p_c_grad_2_ref_values[VS];
        s_t p_c_direction_values[VS];
        s_t value_coeff1_values[VS];
        s_t grad_coeff1_0_values[VS];
        s_t grad_coeff1_1_values[VS];
        s_t grad_coeff1_2_values[VS];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_0_ref_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_1_ref_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_2_ref_values[lane] = s_t(0);
        }
        for (int trial = 0; trial < NS; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t coeff = current[trial * NC + 0][lane];
                p_w_values[lane] += coeff * shape[q * NS + trial];
                p_w_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
                p_w_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
                p_w_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_direction_values[lane] = s_t(0);
        }
        for (int trial = 0; trial < NS; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t coeff = direction[trial * NC + 0][lane];
                p_w_direction_values[lane] += coeff * shape[q * NS + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_0_ref_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_1_ref_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_2_ref_values[lane] = s_t(0);
        }
        for (int trial = 0; trial < NS; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t coeff = current[trial * NC + 1][lane];
                p_c_values[lane] += coeff * shape[q * NS + trial];
                p_c_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
                p_c_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
                p_c_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_direction_values[lane] = s_t(0);
        }
        for (int trial = 0; trial < NS; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t coeff = direction[trial * NC + 1][lane];
                p_c_direction_values[lane] += coeff * shape[q * NS + trial];
            }
        }
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
            const s_t p_w = p_w_values[lane];
            const s_t p_w_grad_0_ref = p_w_grad_0_ref_values[lane];
            const s_t p_w_grad_1_ref = p_w_grad_1_ref_values[lane];
            const s_t p_w_grad_2_ref = p_w_grad_2_ref_values[lane];
            const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
            const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
            const s_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
            const s_t p_w_direction = p_w_direction_values[lane];
            const s_t p_c = p_c_values[lane];
            const s_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
            const s_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
            const s_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
            const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
            const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
            const s_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
            const s_t p_c_direction = p_c_direction_values[lane];
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
            value_coeff1_values[lane] = value_coeff1;
            grad_coeff1_0_values[lane] = grad_coeff1_0;
            grad_coeff1_1_values[lane] = grad_coeff1_1;
            grad_coeff1_2_values[lane] = grad_coeff1_2;
        }
        for (int test = 0; test < NS; ++test) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const ptrdiff_t goff = q * geometry_stride + lane;
                const s_t det = determinant[goff];
                const s_t test_value = shape[q * NS + test];
                const s_t adj0 = adjugate[0][goff];
                const s_t adj1 = adjugate[1][goff];
                const s_t adj2 = adjugate[2][goff];
                const s_t adj3 = adjugate[3][goff];
                const s_t adj4 = adjugate[4][goff];
                const s_t adj5 = adjugate[5][goff];
                const s_t adj6 = adjugate[6][goff];
                const s_t adj7 = adjugate[7][goff];
                const s_t adj8 = adjugate[8][goff];
                const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj3 + grad_ref_z[q * NS + test] * adj6) / det;
                const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj4 + grad_ref_z[q * NS + test] * adj7) / det;
                const s_t test_grad2 = (grad_ref_x[q * NS + test] * adj2 + grad_ref_y[q * NS + test] * adj5 + grad_ref_z[q * NS + test] * adj8) / det;
                output[test * NC + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
            }
        }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_simplex_tet4_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR adjugate[9],
        const s_t *const RSTR shape,
        const s_t *const RSTR grad_ref_x,
        const s_t *const RSTR grad_ref_y,
        const s_t *const RSTR grad_ref_z,
        const s_t *const RSTR q_weight,
        const s_t *const RSTR current[2 * NS],
        const s_t *const RSTR direction[2 * NS],
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
        s_t *const RSTR output[2 * NS]
) {
    static constexpr int ND = 3;
    static constexpr int NC = 2;
    for (int q = 0; q < NQ; ++q) {
        s_t p_w_values[VS];
        s_t p_w_grad_0_ref_values[VS];
        s_t p_w_grad_1_ref_values[VS];
        s_t p_w_grad_2_ref_values[VS];
        s_t p_w_direction_values[VS];
        s_t p_c_values[VS];
        s_t p_c_grad_0_ref_values[VS];
        s_t p_c_grad_1_ref_values[VS];
        s_t p_c_grad_2_ref_values[VS];
        s_t p_c_direction_values[VS];
        s_t value_coeff1_values[VS];
        s_t grad_coeff1_0_values[VS];
        s_t grad_coeff1_1_values[VS];
        s_t grad_coeff1_2_values[VS];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_0_ref_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_1_ref_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_2_ref_values[lane] = s_t(0);
        }
        for (int trial = 0; trial < NS; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t coeff = current[trial * NC + 0][lane];
                p_w_values[lane] += coeff * shape[q * NS + trial];
                p_w_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
                p_w_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
                p_w_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_direction_values[lane] = s_t(0);
        }
        for (int trial = 0; trial < NS; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t coeff = direction[trial * NC + 0][lane];
                p_w_direction_values[lane] += coeff * shape[q * NS + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_0_ref_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_1_ref_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_2_ref_values[lane] = s_t(0);
        }
        for (int trial = 0; trial < NS; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t coeff = current[trial * NC + 1][lane];
                p_c_values[lane] += coeff * shape[q * NS + trial];
                p_c_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
                p_c_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
                p_c_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_direction_values[lane] = s_t(0);
        }
        for (int trial = 0; trial < NS; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t coeff = direction[trial * NC + 1][lane];
                p_c_direction_values[lane] += coeff * shape[q * NS + trial];
            }
        }
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
            const s_t p_w = p_w_values[lane];
            const s_t p_w_grad_0_ref = p_w_grad_0_ref_values[lane];
            const s_t p_w_grad_1_ref = p_w_grad_1_ref_values[lane];
            const s_t p_w_grad_2_ref = p_w_grad_2_ref_values[lane];
            const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
            const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
            const s_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
            const s_t p_w_direction = p_w_direction_values[lane];
            const s_t p_c = p_c_values[lane];
            const s_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
            const s_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
            const s_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
            const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
            const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
            const s_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
            const s_t p_c_direction = p_c_direction_values[lane];
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
            value_coeff1_values[lane] = value_coeff1;
            grad_coeff1_0_values[lane] = grad_coeff1_0;
            grad_coeff1_1_values[lane] = grad_coeff1_1;
            grad_coeff1_2_values[lane] = grad_coeff1_2;
        }
        for (int test = 0; test < NS; ++test) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const ptrdiff_t goff = q * geometry_stride + lane;
                const s_t det = determinant[goff];
                const s_t test_value = shape[q * NS + test];
                const s_t adj0 = adjugate[0][goff];
                const s_t adj1 = adjugate[1][goff];
                const s_t adj2 = adjugate[2][goff];
                const s_t adj3 = adjugate[3][goff];
                const s_t adj4 = adjugate[4][goff];
                const s_t adj5 = adjugate[5][goff];
                const s_t adj6 = adjugate[6][goff];
                const s_t adj7 = adjugate[7][goff];
                const s_t adj8 = adjugate[8][goff];
                const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj3 + grad_ref_z[q * NS + test] * adj6) / det;
                const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj4 + grad_ref_z[q * NS + test] * adj7) / det;
                const s_t test_grad2 = (grad_ref_x[q * NS + test] * adj2 + grad_ref_y[q * NS + test] * adj5 + grad_ref_z[q * NS + test] * adj8) / det;
                output[test * NC + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
            }
        }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_simplex_tet4_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR determinant,
        const s_t *const RSTR adjugate[9],
        const s_t *const RSTR shape,
        const s_t *const RSTR grad_ref_x,
        const s_t *const RSTR grad_ref_y,
        const s_t *const RSTR grad_ref_z,
        const s_t *const RSTR q_weight,
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
    for (int q = 0; q < NQ; ++q) {
        s_t p_w_values[VS];
        s_t p_w_grad_0_ref_values[VS];
        s_t p_w_grad_1_ref_values[VS];
        s_t p_w_grad_2_ref_values[VS];
        s_t p_w_direction_values[VS];
        s_t p_c_values[VS];
        s_t p_c_grad_0_ref_values[VS];
        s_t p_c_grad_1_ref_values[VS];
        s_t p_c_grad_2_ref_values[VS];
        s_t p_c_direction_values[VS];
        s_t value_coeff1_values[VS];
        s_t grad_coeff1_0_values[VS];
        s_t grad_coeff1_1_values[VS];
        s_t grad_coeff1_2_values[VS];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_0_ref_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_1_ref_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_grad_2_ref_values[lane] = s_t(0);
        }
        for (int trial = 0; trial < NS; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t coeff = current[trial * NC + 0][lane];
                p_w_values[lane] += coeff * shape[q * NS + trial];
                p_w_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
                p_w_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
                p_w_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_w_direction_values[lane] = s_t(0);
        }
        for (int trial = 0; trial < NS; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t coeff = direction[trial * NC + 0][lane];
                p_w_direction_values[lane] += coeff * shape[q * NS + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_0_ref_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_1_ref_values[lane] = s_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_grad_2_ref_values[lane] = s_t(0);
        }
        for (int trial = 0; trial < NS; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t coeff = current[trial * NC + 1][lane];
                p_c_values[lane] += coeff * shape[q * NS + trial];
                p_c_grad_0_ref_values[lane] += coeff * grad_ref_x[q * NS + trial];
                p_c_grad_1_ref_values[lane] += coeff * grad_ref_y[q * NS + trial];
                p_c_grad_2_ref_values[lane] += coeff * grad_ref_z[q * NS + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            p_c_direction_values[lane] = s_t(0);
        }
        for (int trial = 0; trial < NS; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const s_t coeff = direction[trial * NC + 1][lane];
                p_c_direction_values[lane] += coeff * shape[q * NS + trial];
            }
        }
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
            const s_t p_w = p_w_values[lane];
            const s_t p_w_grad_0_ref = p_w_grad_0_ref_values[lane];
            const s_t p_w_grad_1_ref = p_w_grad_1_ref_values[lane];
            const s_t p_w_grad_2_ref = p_w_grad_2_ref_values[lane];
            const s_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
            const s_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
            const s_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
            const s_t p_w_direction = p_w_direction_values[lane];
            const s_t p_c = p_c_values[lane];
            const s_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
            const s_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
            const s_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
            const s_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
            const s_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
            const s_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
            const s_t p_c_direction = p_c_direction_values[lane];
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
            value_coeff1_values[lane] = value_coeff1;
            grad_coeff1_0_values[lane] = grad_coeff1_0;
            grad_coeff1_1_values[lane] = grad_coeff1_1;
            grad_coeff1_2_values[lane] = grad_coeff1_2;
        }
        for (int test = 0; test < NS; ++test) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const ptrdiff_t goff = q * geometry_stride + lane;
                const s_t det = determinant[goff];
                const s_t test_value = shape[q * NS + test];
                const s_t adj0 = adjugate[0][goff];
                const s_t adj1 = adjugate[1][goff];
                const s_t adj2 = adjugate[2][goff];
                const s_t adj3 = adjugate[3][goff];
                const s_t adj4 = adjugate[4][goff];
                const s_t adj5 = adjugate[5][goff];
                const s_t adj6 = adjugate[6][goff];
                const s_t adj7 = adjugate[7][goff];
                const s_t adj8 = adjugate[8][goff];
                const s_t test_grad0 = (grad_ref_x[q * NS + test] * adj0 + grad_ref_y[q * NS + test] * adj3 + grad_ref_z[q * NS + test] * adj6) / det;
                const s_t test_grad1 = (grad_ref_x[q * NS + test] * adj1 + grad_ref_y[q * NS + test] * adj4 + grad_ref_z[q * NS + test] * adj7) / det;
                const s_t test_grad2 = (grad_ref_x[q * NS + test] * adj2 + grad_ref_y[q * NS + test] * adj5 + grad_ref_z[q * NS + test] * adj8) / det;
                output[test * NC + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
            }
        }
    }
}

} // namespace codegen
} // namespace sfem

#endif
