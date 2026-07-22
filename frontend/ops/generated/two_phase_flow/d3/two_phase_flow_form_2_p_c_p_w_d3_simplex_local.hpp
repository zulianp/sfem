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
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_simplex_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT q_weight,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 3;
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

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_simplex_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT q_weight,
        scalar_t output[2 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
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

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_simplex_tet4_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT q_weight,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 3;
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

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_simplex_tet4_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT q_weight,
        scalar_t output[2 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
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

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_simplex_jacobian_action_block(
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
        const scalar_t m,
        const scalar_t mu_c,
        const scalar_t porosity,
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
        scalar_t p_c_values[VECTOR_SIZE];
        scalar_t p_c_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_c_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_c_grad_2_ref_values[VECTOR_SIZE];
        scalar_t p_c_direction_values[VECTOR_SIZE];
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
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 0][lane];
                p_w_direction_values[lane] += coeff * shape[q * N_SHAPE + trial];
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
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 1][lane];
                p_c_direction_values[lane] += coeff * shape[q * N_SHAPE + trial];
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
            const scalar_t p_c = p_c_values[lane];
            const scalar_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
            const scalar_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
            const scalar_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
            const scalar_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
            const scalar_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
            const scalar_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
            const scalar_t p_c_direction = p_c_direction_values[lane];
            const scalar_t residual_tmp0 = p_c - p_w;
            const scalar_t residual_tmp1 = pow(residual_tmp0/P_r, m);
            const scalar_t residual_tmp2 = residual_tmp1 + scalar_t(1);
            const scalar_t residual_tmp3 = scalar_t(1) - m;
            const scalar_t residual_tmp4 = pow(residual_tmp2, residual_tmp3/m);
            const scalar_t residual_tmp5 = M_c*p_w_direction/(R*T*Z*dt);
            const scalar_t residual_tmp6 = p_c*residual_tmp1*residual_tmp3/(residual_tmp0*residual_tmp2);
            const scalar_t residual_tmp7 = scalar_t(1) - residual_tmp4;
            const scalar_t residual_tmp8 = dt*residual_tmp6*pow(residual_tmp7, C_ka1);
            const scalar_t residual_tmp9 = residual_tmp8*(-K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2);
            const scalar_t residual_tmp10 = pow(residual_tmp4, C_ka2);
            const scalar_t residual_tmp11 = C_ka2*residual_tmp10;
            const scalar_t residual_tmp12 = C_ka1*residual_tmp4*(residual_tmp10 + scalar_t(-1))/residual_tmp7;
            const scalar_t residual_tmp13 = residual_tmp5/mu_c;
            const scalar_t residual_tmp14 = -K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2;
            const scalar_t residual_tmp15 = residual_tmp11*residual_tmp8;
            const scalar_t residual_tmp16 = residual_tmp12*residual_tmp8;
            const scalar_t residual_tmp17 = -K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2;
            const scalar_t value_coeff1 = -porosity*residual_tmp4*residual_tmp5*residual_tmp6*(S_res + scalar_t(-1));
            const scalar_t grad_coeff1_0 = residual_tmp13*(-residual_tmp11*residual_tmp9 + residual_tmp12*residual_tmp9);
            const scalar_t grad_coeff1_1 = residual_tmp13*(-residual_tmp14*residual_tmp15 + residual_tmp14*residual_tmp16);
            const scalar_t grad_coeff1_2 = residual_tmp13*(-residual_tmp15*residual_tmp17 + residual_tmp16*residual_tmp17);
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
                output[test * N_FIELDS + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_simplex_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t current[2 * N_SHAPE][VECTOR_SIZE],
        const scalar_t direction[2 * N_SHAPE][VECTOR_SIZE],
        const scalar_t C_ka1,
        const scalar_t C_ka2,
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
        const scalar_t m,
        const scalar_t mu_c,
        const scalar_t porosity,
        scalar_t output[2 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
    for (int q = 0; q < N_QP; ++q) {
        scalar_t p_w_values[VECTOR_SIZE];
        scalar_t p_w_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_w_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_w_grad_2_ref_values[VECTOR_SIZE];
        scalar_t p_w_direction_values[VECTOR_SIZE];
        scalar_t p_c_values[VECTOR_SIZE];
        scalar_t p_c_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_c_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_c_grad_2_ref_values[VECTOR_SIZE];
        scalar_t p_c_direction_values[VECTOR_SIZE];
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
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 0][lane];
                p_w_direction_values[lane] += coeff * shape[q * N_SHAPE + trial];
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
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 1][lane];
                p_c_direction_values[lane] += coeff * shape[q * N_SHAPE + trial];
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
            const scalar_t p_c = p_c_values[lane];
            const scalar_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
            const scalar_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
            const scalar_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
            const scalar_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
            const scalar_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
            const scalar_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
            const scalar_t p_c_direction = p_c_direction_values[lane];
            const scalar_t residual_tmp0 = p_c - p_w;
            const scalar_t residual_tmp1 = pow(residual_tmp0/P_r, m);
            const scalar_t residual_tmp2 = residual_tmp1 + scalar_t(1);
            const scalar_t residual_tmp3 = scalar_t(1) - m;
            const scalar_t residual_tmp4 = pow(residual_tmp2, residual_tmp3/m);
            const scalar_t residual_tmp5 = M_c*p_w_direction/(R*T*Z*dt);
            const scalar_t residual_tmp6 = p_c*residual_tmp1*residual_tmp3/(residual_tmp0*residual_tmp2);
            const scalar_t residual_tmp7 = scalar_t(1) - residual_tmp4;
            const scalar_t residual_tmp8 = dt*residual_tmp6*pow(residual_tmp7, C_ka1);
            const scalar_t residual_tmp9 = residual_tmp8*(-K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2);
            const scalar_t residual_tmp10 = pow(residual_tmp4, C_ka2);
            const scalar_t residual_tmp11 = C_ka2*residual_tmp10;
            const scalar_t residual_tmp12 = C_ka1*residual_tmp4*(residual_tmp10 + scalar_t(-1))/residual_tmp7;
            const scalar_t residual_tmp13 = residual_tmp5/mu_c;
            const scalar_t residual_tmp14 = -K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2;
            const scalar_t residual_tmp15 = residual_tmp11*residual_tmp8;
            const scalar_t residual_tmp16 = residual_tmp12*residual_tmp8;
            const scalar_t residual_tmp17 = -K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2;
            const scalar_t value_coeff1 = -porosity*residual_tmp4*residual_tmp5*residual_tmp6*(S_res + scalar_t(-1));
            const scalar_t grad_coeff1_0 = residual_tmp13*(-residual_tmp11*residual_tmp9 + residual_tmp12*residual_tmp9);
            const scalar_t grad_coeff1_1 = residual_tmp13*(-residual_tmp14*residual_tmp15 + residual_tmp14*residual_tmp16);
            const scalar_t grad_coeff1_2 = residual_tmp13*(-residual_tmp15*residual_tmp17 + residual_tmp16*residual_tmp17);
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
                output[test * N_FIELDS + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_simplex_tet4_jacobian_action_block(
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
        const scalar_t m,
        const scalar_t mu_c,
        const scalar_t porosity,
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
        scalar_t p_c_values[VECTOR_SIZE];
        scalar_t p_c_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_c_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_c_grad_2_ref_values[VECTOR_SIZE];
        scalar_t p_c_direction_values[VECTOR_SIZE];
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
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 0][lane];
                p_w_direction_values[lane] += coeff * shape[q * N_SHAPE + trial];
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
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 1][lane];
                p_c_direction_values[lane] += coeff * shape[q * N_SHAPE + trial];
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
            const scalar_t p_c = p_c_values[lane];
            const scalar_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
            const scalar_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
            const scalar_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
            const scalar_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
            const scalar_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
            const scalar_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
            const scalar_t p_c_direction = p_c_direction_values[lane];
            const scalar_t residual_tmp0 = p_c - p_w;
            const scalar_t residual_tmp1 = pow(residual_tmp0/P_r, m);
            const scalar_t residual_tmp2 = residual_tmp1 + scalar_t(1);
            const scalar_t residual_tmp3 = scalar_t(1) - m;
            const scalar_t residual_tmp4 = pow(residual_tmp2, residual_tmp3/m);
            const scalar_t residual_tmp5 = M_c*p_w_direction/(R*T*Z*dt);
            const scalar_t residual_tmp6 = p_c*residual_tmp1*residual_tmp3/(residual_tmp0*residual_tmp2);
            const scalar_t residual_tmp7 = scalar_t(1) - residual_tmp4;
            const scalar_t residual_tmp8 = dt*residual_tmp6*pow(residual_tmp7, C_ka1);
            const scalar_t residual_tmp9 = residual_tmp8*(-K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2);
            const scalar_t residual_tmp10 = pow(residual_tmp4, C_ka2);
            const scalar_t residual_tmp11 = C_ka2*residual_tmp10;
            const scalar_t residual_tmp12 = C_ka1*residual_tmp4*(residual_tmp10 + scalar_t(-1))/residual_tmp7;
            const scalar_t residual_tmp13 = residual_tmp5/mu_c;
            const scalar_t residual_tmp14 = -K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2;
            const scalar_t residual_tmp15 = residual_tmp11*residual_tmp8;
            const scalar_t residual_tmp16 = residual_tmp12*residual_tmp8;
            const scalar_t residual_tmp17 = -K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2;
            const scalar_t value_coeff1 = -porosity*residual_tmp4*residual_tmp5*residual_tmp6*(S_res + scalar_t(-1));
            const scalar_t grad_coeff1_0 = residual_tmp13*(-residual_tmp11*residual_tmp9 + residual_tmp12*residual_tmp9);
            const scalar_t grad_coeff1_1 = residual_tmp13*(-residual_tmp14*residual_tmp15 + residual_tmp14*residual_tmp16);
            const scalar_t grad_coeff1_2 = residual_tmp13*(-residual_tmp15*residual_tmp17 + residual_tmp16*residual_tmp17);
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
                output[test * N_FIELDS + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void two_phase_flow_form_2_p_c_p_w_d3_simplex_tet4_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t current[2 * N_SHAPE][VECTOR_SIZE],
        const scalar_t direction[2 * N_SHAPE][VECTOR_SIZE],
        const scalar_t C_ka1,
        const scalar_t C_ka2,
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
        const scalar_t m,
        const scalar_t mu_c,
        const scalar_t porosity,
        scalar_t output[2 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
    for (int q = 0; q < N_QP; ++q) {
        scalar_t p_w_values[VECTOR_SIZE];
        scalar_t p_w_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_w_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_w_grad_2_ref_values[VECTOR_SIZE];
        scalar_t p_w_direction_values[VECTOR_SIZE];
        scalar_t p_c_values[VECTOR_SIZE];
        scalar_t p_c_grad_0_ref_values[VECTOR_SIZE];
        scalar_t p_c_grad_1_ref_values[VECTOR_SIZE];
        scalar_t p_c_grad_2_ref_values[VECTOR_SIZE];
        scalar_t p_c_direction_values[VECTOR_SIZE];
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
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 0][lane];
                p_w_direction_values[lane] += coeff * shape[q * N_SHAPE + trial];
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
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 1][lane];
                p_c_direction_values[lane] += coeff * shape[q * N_SHAPE + trial];
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
            const scalar_t p_c = p_c_values[lane];
            const scalar_t p_c_grad_0_ref = p_c_grad_0_ref_values[lane];
            const scalar_t p_c_grad_1_ref = p_c_grad_1_ref_values[lane];
            const scalar_t p_c_grad_2_ref = p_c_grad_2_ref_values[lane];
            const scalar_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
            const scalar_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
            const scalar_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
            const scalar_t p_c_direction = p_c_direction_values[lane];
            const scalar_t residual_tmp0 = p_c - p_w;
            const scalar_t residual_tmp1 = pow(residual_tmp0/P_r, m);
            const scalar_t residual_tmp2 = residual_tmp1 + scalar_t(1);
            const scalar_t residual_tmp3 = scalar_t(1) - m;
            const scalar_t residual_tmp4 = pow(residual_tmp2, residual_tmp3/m);
            const scalar_t residual_tmp5 = M_c*p_w_direction/(R*T*Z*dt);
            const scalar_t residual_tmp6 = p_c*residual_tmp1*residual_tmp3/(residual_tmp0*residual_tmp2);
            const scalar_t residual_tmp7 = scalar_t(1) - residual_tmp4;
            const scalar_t residual_tmp8 = dt*residual_tmp6*pow(residual_tmp7, C_ka1);
            const scalar_t residual_tmp9 = residual_tmp8*(-K_0*p_c_grad_0 - K_1*p_c_grad_1 - K_2*p_c_grad_2);
            const scalar_t residual_tmp10 = pow(residual_tmp4, C_ka2);
            const scalar_t residual_tmp11 = C_ka2*residual_tmp10;
            const scalar_t residual_tmp12 = C_ka1*residual_tmp4*(residual_tmp10 + scalar_t(-1))/residual_tmp7;
            const scalar_t residual_tmp13 = residual_tmp5/mu_c;
            const scalar_t residual_tmp14 = -K_3*p_c_grad_0 - K_4*p_c_grad_1 - K_5*p_c_grad_2;
            const scalar_t residual_tmp15 = residual_tmp11*residual_tmp8;
            const scalar_t residual_tmp16 = residual_tmp12*residual_tmp8;
            const scalar_t residual_tmp17 = -K_6*p_c_grad_0 - K_7*p_c_grad_1 - K_8*p_c_grad_2;
            const scalar_t value_coeff1 = -porosity*residual_tmp4*residual_tmp5*residual_tmp6*(S_res + scalar_t(-1));
            const scalar_t grad_coeff1_0 = residual_tmp13*(-residual_tmp11*residual_tmp9 + residual_tmp12*residual_tmp9);
            const scalar_t grad_coeff1_1 = residual_tmp13*(-residual_tmp14*residual_tmp15 + residual_tmp14*residual_tmp16);
            const scalar_t grad_coeff1_2 = residual_tmp13*(-residual_tmp15*residual_tmp17 + residual_tmp16*residual_tmp17);
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
                output[test * N_FIELDS + 1][lane] += q_weight[q] * det * (value_coeff1_values[lane] * test_value + grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
            }
        }
    }
}

} // namespace codegen
} // namespace sfem

#endif
