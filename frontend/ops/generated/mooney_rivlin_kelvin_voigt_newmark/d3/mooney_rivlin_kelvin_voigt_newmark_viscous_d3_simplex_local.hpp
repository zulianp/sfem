#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_VISCOUS_D3_SIMPLEX_LOCAL_HPP
#define MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_VISCOUS_D3_SIMPLEX_LOCAL_HPP

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
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT current[3 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT previous[3 * N_SHAPE],
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t *const SFEM_RESTRICT output[3 * N_SHAPE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 3;
    for (int q = 0; q < N_QP; ++q) {
        scalar_t u0_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u0_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u2_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u2_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u2_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u2_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u2_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u2_old_grad_2_ref_values[VECTOR_SIZE];
        scalar_t grad_coeff0_0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_1_values[VECTOR_SIZE];
        scalar_t grad_coeff0_2_values[VECTOR_SIZE];
        scalar_t grad_coeff1_0_values[VECTOR_SIZE];
        scalar_t grad_coeff1_1_values[VECTOR_SIZE];
        scalar_t grad_coeff1_2_values[VECTOR_SIZE];
        scalar_t grad_coeff2_0_values[VECTOR_SIZE];
        scalar_t grad_coeff2_1_values[VECTOR_SIZE];
        scalar_t grad_coeff2_2_values[VECTOR_SIZE];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 0][lane];
                u0_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u0_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_old_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 0][lane];
                u0_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u0_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 1][lane];
                u1_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u1_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_old_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 1][lane];
                u1_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u1_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 2][lane];
                u2_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u2_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u2_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_old_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_old_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_old_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 2][lane];
                u2_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u2_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u2_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
            const scalar_t u0_grad_0_ref = u0_grad_0_ref_values[lane];
            const scalar_t u0_grad_1_ref = u0_grad_1_ref_values[lane];
            const scalar_t u0_grad_2_ref = u0_grad_2_ref_values[lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
            const scalar_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
            const scalar_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[lane];
            const scalar_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[lane];
            const scalar_t u0_old_grad_2_ref = u0_old_grad_2_ref_values[lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
            const scalar_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
            const scalar_t u1_grad_0_ref = u1_grad_0_ref_values[lane];
            const scalar_t u1_grad_1_ref = u1_grad_1_ref_values[lane];
            const scalar_t u1_grad_2_ref = u1_grad_2_ref_values[lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
            const scalar_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
            const scalar_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[lane];
            const scalar_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[lane];
            const scalar_t u1_old_grad_2_ref = u1_old_grad_2_ref_values[lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
            const scalar_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
            const scalar_t u2_grad_0_ref = u2_grad_0_ref_values[lane];
            const scalar_t u2_grad_1_ref = u2_grad_1_ref_values[lane];
            const scalar_t u2_grad_2_ref = u2_grad_2_ref_values[lane];
            const scalar_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
            const scalar_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
            const scalar_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
            const scalar_t u2_old_grad_0_ref = u2_old_grad_0_ref_values[lane];
            const scalar_t u2_old_grad_1_ref = u2_old_grad_1_ref_values[lane];
            const scalar_t u2_old_grad_2_ref = u2_old_grad_2_ref_values[lane];
            const scalar_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
            const scalar_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
            const scalar_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
            const scalar_t residual_tmp0 = u0_grad_0*u1_grad_1;
            const scalar_t residual_tmp1 = u0_grad_1*u2_grad_0;
            const scalar_t residual_tmp2 = u0_grad_2*u2_grad_1;
            const scalar_t residual_tmp3 = u1_grad_2*u2_grad_1;
            const scalar_t residual_tmp4 = u0_grad_1*u1_grad_0;
            const scalar_t residual_tmp5 = u0_grad_2*u2_grad_0;
            const scalar_t residual_tmp6 = u2_grad_2 + scalar_t(1);
            const scalar_t residual_tmp7 = -residual_tmp5 + residual_tmp6 + u0_grad_0*u2_grad_2 + u0_grad_0;
            const scalar_t residual_tmp8 = -residual_tmp3 + u1_grad_1*u2_grad_2 + u1_grad_1;
            const scalar_t residual_tmp9 = residual_tmp0 - residual_tmp4;
            const scalar_t residual_tmp10 = pow_m1(residual_tmp0*u2_grad_2 + residual_tmp1*u1_grad_2 + residual_tmp2*u1_grad_0 - residual_tmp3*u0_grad_0 - residual_tmp4*u2_grad_2 - residual_tmp5*u1_grad_1 + residual_tmp7 + residual_tmp8 + residual_tmp9);
            const scalar_t residual_tmp11 = -u0_grad_1*u1_grad_2 + u0_grad_2*u1_grad_1 + u0_grad_2;
            const scalar_t residual_tmp12 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp13 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
            const scalar_t residual_tmp14 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
            const scalar_t residual_tmp15 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
            const scalar_t residual_tmp16 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
            const scalar_t residual_tmp17 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
            const scalar_t residual_tmp18 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
            const scalar_t residual_tmp19 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
            const scalar_t residual_tmp20 = residual_tmp9 + u0_grad_0 + u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp21 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
            const scalar_t residual_tmp22 = residual_tmp6 + residual_tmp8;
            const scalar_t residual_tmp23 = residual_tmp11*residual_tmp12 + residual_tmp13*residual_tmp14 + residual_tmp15*residual_tmp16 + residual_tmp17*residual_tmp18 - residual_tmp19*residual_tmp20 - residual_tmp21*residual_tmp22;
            const scalar_t residual_tmp24 = -residual_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
            const scalar_t residual_tmp25 = -residual_tmp1 + u0_grad_0*u2_grad_1 + u2_grad_1;
            const scalar_t residual_tmp26 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp27 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
            const scalar_t residual_tmp28 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
            const scalar_t residual_tmp29 = residual_tmp12*residual_tmp24 - residual_tmp13*residual_tmp7 + residual_tmp16*residual_tmp26 + residual_tmp18*residual_tmp27 + residual_tmp19*residual_tmp25 - residual_tmp22*residual_tmp28;
            const scalar_t residual_tmp30 = residual_tmp24*residual_tmp28;
            const scalar_t residual_tmp31 = residual_tmp25*residual_tmp27;
            const scalar_t residual_tmp32 = residual_tmp11*residual_tmp21;
            const scalar_t residual_tmp33 = residual_tmp14*residual_tmp15;
            const scalar_t residual_tmp34 = residual_tmp26*residual_tmp7;
            const scalar_t residual_tmp35 = -residual_tmp34;
            const scalar_t residual_tmp36 = residual_tmp17*residual_tmp20;
            const scalar_t residual_tmp37 = -residual_tmp36;
            const scalar_t residual_tmp38 = residual_tmp30 + residual_tmp31 + residual_tmp32 + residual_tmp33 + residual_tmp35 + residual_tmp37;
            const scalar_t residual_tmp39 = residual_tmp12*residual_tmp22;
            const scalar_t residual_tmp40 = residual_tmp13*residual_tmp16 + residual_tmp18*residual_tmp19 - residual_tmp39;
            const scalar_t residual_tmp41 = scalar_t(3)*eta_b*(residual_tmp38 + residual_tmp40);
            const scalar_t residual_tmp42 = scalar_t(2)*eta_s;
            const scalar_t residual_tmp43 = residual_tmp41 + residual_tmp42*(scalar_t(2)*residual_tmp13*residual_tmp16 + scalar_t(2)*residual_tmp18*residual_tmp19 - residual_tmp38 - scalar_t(2)*residual_tmp39);
            const scalar_t residual_tmp44 = residual_tmp11*residual_tmp28 + residual_tmp14*residual_tmp26 - residual_tmp15*residual_tmp7 + residual_tmp17*residual_tmp25 - residual_tmp20*residual_tmp27 + residual_tmp21*residual_tmp24;
            const scalar_t residual_tmp45 = residual_tmp41 + residual_tmp42*(scalar_t(2)*residual_tmp24*residual_tmp28 + scalar_t(2)*residual_tmp25*residual_tmp27 - residual_tmp32 - residual_tmp33 - scalar_t(2)*residual_tmp34 - residual_tmp37 - residual_tmp40);
            const scalar_t residual_tmp46 = residual_tmp41 + residual_tmp42*(scalar_t(2)*residual_tmp11*residual_tmp21 + scalar_t(2)*residual_tmp14*residual_tmp15 - residual_tmp30 - residual_tmp31 - residual_tmp35 - scalar_t(2)*residual_tmp36 - residual_tmp40);
            const scalar_t grad_coeff0_0 = residual_tmp10*(eta_s*(residual_tmp11*residual_tmp23 + residual_tmp24*residual_tmp29) - (scalar_t(1) / scalar_t(3))*residual_tmp22*residual_tmp43);
            const scalar_t grad_coeff0_1 = residual_tmp10*(-eta_s*(-residual_tmp14*residual_tmp23 + residual_tmp29*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp16*residual_tmp43);
            const scalar_t grad_coeff0_2 = residual_tmp10*(-eta_s*(residual_tmp20*residual_tmp23 - residual_tmp25*residual_tmp29) + ((scalar_t(1) / scalar_t(3)))*residual_tmp18*residual_tmp43);
            const scalar_t grad_coeff1_0 = residual_tmp10*(-eta_s*(-residual_tmp11*residual_tmp44 + residual_tmp22*residual_tmp29) + ((scalar_t(1) / scalar_t(3)))*residual_tmp24*residual_tmp45);
            const scalar_t grad_coeff1_1 = residual_tmp10*(eta_s*(residual_tmp14*residual_tmp44 + residual_tmp16*residual_tmp29) - (scalar_t(1) / scalar_t(3))*residual_tmp45*residual_tmp7);
            const scalar_t grad_coeff1_2 = residual_tmp10*(-eta_s*(-residual_tmp18*residual_tmp29 + residual_tmp20*residual_tmp44) + ((scalar_t(1) / scalar_t(3)))*residual_tmp25*residual_tmp45);
            const scalar_t grad_coeff2_0 = residual_tmp10*(-eta_s*(residual_tmp22*residual_tmp23 - residual_tmp24*residual_tmp44) + ((scalar_t(1) / scalar_t(3)))*residual_tmp11*residual_tmp46);
            const scalar_t grad_coeff2_1 = residual_tmp10*(-eta_s*(-residual_tmp16*residual_tmp23 + residual_tmp44*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp14*residual_tmp46);
            const scalar_t grad_coeff2_2 = residual_tmp10*(eta_s*(residual_tmp18*residual_tmp23 + residual_tmp25*residual_tmp44) - (scalar_t(1) / scalar_t(3))*residual_tmp20*residual_tmp46);
            grad_coeff0_0_values[lane] = grad_coeff0_0;
            grad_coeff0_1_values[lane] = grad_coeff0_1;
            grad_coeff0_2_values[lane] = grad_coeff0_2;
            grad_coeff1_0_values[lane] = grad_coeff1_0;
            grad_coeff1_1_values[lane] = grad_coeff1_1;
            grad_coeff1_2_values[lane] = grad_coeff1_2;
            grad_coeff2_0_values[lane] = grad_coeff2_0;
            grad_coeff2_1_values[lane] = grad_coeff2_1;
            grad_coeff2_2_values[lane] = grad_coeff2_2;
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
                output[test * N_FIELDS + 1][lane] += q_weight[q] * det * (grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
                output[test * N_FIELDS + 2][lane] += q_weight[q] * det * (grad_coeff2_0_values[lane] * test_grad0 + grad_coeff2_1_values[lane] * test_grad1 + grad_coeff2_2_values[lane] * test_grad2);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t current[3 * N_SHAPE][VECTOR_SIZE],
        const scalar_t previous[3 * N_SHAPE][VECTOR_SIZE],
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t output[3 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 3;
    for (int q = 0; q < N_QP; ++q) {
        scalar_t u0_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u0_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u2_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u2_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u2_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u2_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u2_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u2_old_grad_2_ref_values[VECTOR_SIZE];
        scalar_t grad_coeff0_0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_1_values[VECTOR_SIZE];
        scalar_t grad_coeff0_2_values[VECTOR_SIZE];
        scalar_t grad_coeff1_0_values[VECTOR_SIZE];
        scalar_t grad_coeff1_1_values[VECTOR_SIZE];
        scalar_t grad_coeff1_2_values[VECTOR_SIZE];
        scalar_t grad_coeff2_0_values[VECTOR_SIZE];
        scalar_t grad_coeff2_1_values[VECTOR_SIZE];
        scalar_t grad_coeff2_2_values[VECTOR_SIZE];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 0][lane];
                u0_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u0_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_old_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 0][lane];
                u0_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u0_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 1][lane];
                u1_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u1_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_old_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 1][lane];
                u1_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u1_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 2][lane];
                u2_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u2_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u2_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_old_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_old_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_old_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 2][lane];
                u2_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u2_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u2_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
            const scalar_t u0_grad_0_ref = u0_grad_0_ref_values[lane];
            const scalar_t u0_grad_1_ref = u0_grad_1_ref_values[lane];
            const scalar_t u0_grad_2_ref = u0_grad_2_ref_values[lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
            const scalar_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
            const scalar_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[lane];
            const scalar_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[lane];
            const scalar_t u0_old_grad_2_ref = u0_old_grad_2_ref_values[lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
            const scalar_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
            const scalar_t u1_grad_0_ref = u1_grad_0_ref_values[lane];
            const scalar_t u1_grad_1_ref = u1_grad_1_ref_values[lane];
            const scalar_t u1_grad_2_ref = u1_grad_2_ref_values[lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
            const scalar_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
            const scalar_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[lane];
            const scalar_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[lane];
            const scalar_t u1_old_grad_2_ref = u1_old_grad_2_ref_values[lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
            const scalar_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
            const scalar_t u2_grad_0_ref = u2_grad_0_ref_values[lane];
            const scalar_t u2_grad_1_ref = u2_grad_1_ref_values[lane];
            const scalar_t u2_grad_2_ref = u2_grad_2_ref_values[lane];
            const scalar_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
            const scalar_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
            const scalar_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
            const scalar_t u2_old_grad_0_ref = u2_old_grad_0_ref_values[lane];
            const scalar_t u2_old_grad_1_ref = u2_old_grad_1_ref_values[lane];
            const scalar_t u2_old_grad_2_ref = u2_old_grad_2_ref_values[lane];
            const scalar_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
            const scalar_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
            const scalar_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
            const scalar_t residual_tmp0 = u0_grad_0*u1_grad_1;
            const scalar_t residual_tmp1 = u0_grad_1*u2_grad_0;
            const scalar_t residual_tmp2 = u0_grad_2*u2_grad_1;
            const scalar_t residual_tmp3 = u1_grad_2*u2_grad_1;
            const scalar_t residual_tmp4 = u0_grad_1*u1_grad_0;
            const scalar_t residual_tmp5 = u0_grad_2*u2_grad_0;
            const scalar_t residual_tmp6 = u2_grad_2 + scalar_t(1);
            const scalar_t residual_tmp7 = -residual_tmp5 + residual_tmp6 + u0_grad_0*u2_grad_2 + u0_grad_0;
            const scalar_t residual_tmp8 = -residual_tmp3 + u1_grad_1*u2_grad_2 + u1_grad_1;
            const scalar_t residual_tmp9 = residual_tmp0 - residual_tmp4;
            const scalar_t residual_tmp10 = pow_m1(residual_tmp0*u2_grad_2 + residual_tmp1*u1_grad_2 + residual_tmp2*u1_grad_0 - residual_tmp3*u0_grad_0 - residual_tmp4*u2_grad_2 - residual_tmp5*u1_grad_1 + residual_tmp7 + residual_tmp8 + residual_tmp9);
            const scalar_t residual_tmp11 = -u0_grad_1*u1_grad_2 + u0_grad_2*u1_grad_1 + u0_grad_2;
            const scalar_t residual_tmp12 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp13 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
            const scalar_t residual_tmp14 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
            const scalar_t residual_tmp15 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
            const scalar_t residual_tmp16 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
            const scalar_t residual_tmp17 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
            const scalar_t residual_tmp18 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
            const scalar_t residual_tmp19 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
            const scalar_t residual_tmp20 = residual_tmp9 + u0_grad_0 + u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp21 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
            const scalar_t residual_tmp22 = residual_tmp6 + residual_tmp8;
            const scalar_t residual_tmp23 = residual_tmp11*residual_tmp12 + residual_tmp13*residual_tmp14 + residual_tmp15*residual_tmp16 + residual_tmp17*residual_tmp18 - residual_tmp19*residual_tmp20 - residual_tmp21*residual_tmp22;
            const scalar_t residual_tmp24 = -residual_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
            const scalar_t residual_tmp25 = -residual_tmp1 + u0_grad_0*u2_grad_1 + u2_grad_1;
            const scalar_t residual_tmp26 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp27 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
            const scalar_t residual_tmp28 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
            const scalar_t residual_tmp29 = residual_tmp12*residual_tmp24 - residual_tmp13*residual_tmp7 + residual_tmp16*residual_tmp26 + residual_tmp18*residual_tmp27 + residual_tmp19*residual_tmp25 - residual_tmp22*residual_tmp28;
            const scalar_t residual_tmp30 = residual_tmp24*residual_tmp28;
            const scalar_t residual_tmp31 = residual_tmp25*residual_tmp27;
            const scalar_t residual_tmp32 = residual_tmp11*residual_tmp21;
            const scalar_t residual_tmp33 = residual_tmp14*residual_tmp15;
            const scalar_t residual_tmp34 = residual_tmp26*residual_tmp7;
            const scalar_t residual_tmp35 = -residual_tmp34;
            const scalar_t residual_tmp36 = residual_tmp17*residual_tmp20;
            const scalar_t residual_tmp37 = -residual_tmp36;
            const scalar_t residual_tmp38 = residual_tmp30 + residual_tmp31 + residual_tmp32 + residual_tmp33 + residual_tmp35 + residual_tmp37;
            const scalar_t residual_tmp39 = residual_tmp12*residual_tmp22;
            const scalar_t residual_tmp40 = residual_tmp13*residual_tmp16 + residual_tmp18*residual_tmp19 - residual_tmp39;
            const scalar_t residual_tmp41 = scalar_t(3)*eta_b*(residual_tmp38 + residual_tmp40);
            const scalar_t residual_tmp42 = scalar_t(2)*eta_s;
            const scalar_t residual_tmp43 = residual_tmp41 + residual_tmp42*(scalar_t(2)*residual_tmp13*residual_tmp16 + scalar_t(2)*residual_tmp18*residual_tmp19 - residual_tmp38 - scalar_t(2)*residual_tmp39);
            const scalar_t residual_tmp44 = residual_tmp11*residual_tmp28 + residual_tmp14*residual_tmp26 - residual_tmp15*residual_tmp7 + residual_tmp17*residual_tmp25 - residual_tmp20*residual_tmp27 + residual_tmp21*residual_tmp24;
            const scalar_t residual_tmp45 = residual_tmp41 + residual_tmp42*(scalar_t(2)*residual_tmp24*residual_tmp28 + scalar_t(2)*residual_tmp25*residual_tmp27 - residual_tmp32 - residual_tmp33 - scalar_t(2)*residual_tmp34 - residual_tmp37 - residual_tmp40);
            const scalar_t residual_tmp46 = residual_tmp41 + residual_tmp42*(scalar_t(2)*residual_tmp11*residual_tmp21 + scalar_t(2)*residual_tmp14*residual_tmp15 - residual_tmp30 - residual_tmp31 - residual_tmp35 - scalar_t(2)*residual_tmp36 - residual_tmp40);
            const scalar_t grad_coeff0_0 = residual_tmp10*(eta_s*(residual_tmp11*residual_tmp23 + residual_tmp24*residual_tmp29) - (scalar_t(1) / scalar_t(3))*residual_tmp22*residual_tmp43);
            const scalar_t grad_coeff0_1 = residual_tmp10*(-eta_s*(-residual_tmp14*residual_tmp23 + residual_tmp29*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp16*residual_tmp43);
            const scalar_t grad_coeff0_2 = residual_tmp10*(-eta_s*(residual_tmp20*residual_tmp23 - residual_tmp25*residual_tmp29) + ((scalar_t(1) / scalar_t(3)))*residual_tmp18*residual_tmp43);
            const scalar_t grad_coeff1_0 = residual_tmp10*(-eta_s*(-residual_tmp11*residual_tmp44 + residual_tmp22*residual_tmp29) + ((scalar_t(1) / scalar_t(3)))*residual_tmp24*residual_tmp45);
            const scalar_t grad_coeff1_1 = residual_tmp10*(eta_s*(residual_tmp14*residual_tmp44 + residual_tmp16*residual_tmp29) - (scalar_t(1) / scalar_t(3))*residual_tmp45*residual_tmp7);
            const scalar_t grad_coeff1_2 = residual_tmp10*(-eta_s*(-residual_tmp18*residual_tmp29 + residual_tmp20*residual_tmp44) + ((scalar_t(1) / scalar_t(3)))*residual_tmp25*residual_tmp45);
            const scalar_t grad_coeff2_0 = residual_tmp10*(-eta_s*(residual_tmp22*residual_tmp23 - residual_tmp24*residual_tmp44) + ((scalar_t(1) / scalar_t(3)))*residual_tmp11*residual_tmp46);
            const scalar_t grad_coeff2_1 = residual_tmp10*(-eta_s*(-residual_tmp16*residual_tmp23 + residual_tmp44*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp14*residual_tmp46);
            const scalar_t grad_coeff2_2 = residual_tmp10*(eta_s*(residual_tmp18*residual_tmp23 + residual_tmp25*residual_tmp44) - (scalar_t(1) / scalar_t(3))*residual_tmp20*residual_tmp46);
            grad_coeff0_0_values[lane] = grad_coeff0_0;
            grad_coeff0_1_values[lane] = grad_coeff0_1;
            grad_coeff0_2_values[lane] = grad_coeff0_2;
            grad_coeff1_0_values[lane] = grad_coeff1_0;
            grad_coeff1_1_values[lane] = grad_coeff1_1;
            grad_coeff1_2_values[lane] = grad_coeff1_2;
            grad_coeff2_0_values[lane] = grad_coeff2_0;
            grad_coeff2_1_values[lane] = grad_coeff2_1;
            grad_coeff2_2_values[lane] = grad_coeff2_2;
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
                output[test * N_FIELDS + 1][lane] += q_weight[q] * det * (grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
                output[test * N_FIELDS + 2][lane] += q_weight[q] * det * (grad_coeff2_0_values[lane] * test_grad0 + grad_coeff2_1_values[lane] * test_grad1 + grad_coeff2_2_values[lane] * test_grad2);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_tet4_residual_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT current[3 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT previous[3 * N_SHAPE],
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t *const SFEM_RESTRICT output[3 * N_SHAPE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 3;
    for (int q = 0; q < N_QP; ++q) {
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
            const scalar_t u0_grad_0_ref = -(current[0][lane]) + current[3][lane];
            const scalar_t u0_grad_1_ref = -(current[0][lane]) + current[6][lane];
            const scalar_t u0_grad_2_ref = -(current[0][lane]) + current[9][lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
            const scalar_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
            const scalar_t u0_old_grad_0_ref = -(previous[0][lane]) + previous[3][lane];
            const scalar_t u0_old_grad_1_ref = -(previous[0][lane]) + previous[6][lane];
            const scalar_t u0_old_grad_2_ref = -(previous[0][lane]) + previous[9][lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
            const scalar_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
            const scalar_t u1_grad_0_ref = -(current[1][lane]) + current[4][lane];
            const scalar_t u1_grad_1_ref = -(current[1][lane]) + current[7][lane];
            const scalar_t u1_grad_2_ref = -(current[1][lane]) + current[10][lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
            const scalar_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
            const scalar_t u1_old_grad_0_ref = -(previous[1][lane]) + previous[4][lane];
            const scalar_t u1_old_grad_1_ref = -(previous[1][lane]) + previous[7][lane];
            const scalar_t u1_old_grad_2_ref = -(previous[1][lane]) + previous[10][lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
            const scalar_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
            const scalar_t u2_grad_0_ref = -(current[2][lane]) + current[5][lane];
            const scalar_t u2_grad_1_ref = -(current[2][lane]) + current[8][lane];
            const scalar_t u2_grad_2_ref = -(current[2][lane]) + current[11][lane];
            const scalar_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
            const scalar_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
            const scalar_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
            const scalar_t u2_old_grad_0_ref = -(previous[2][lane]) + previous[5][lane];
            const scalar_t u2_old_grad_1_ref = -(previous[2][lane]) + previous[8][lane];
            const scalar_t u2_old_grad_2_ref = -(previous[2][lane]) + previous[11][lane];
            const scalar_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
            const scalar_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
            const scalar_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
            const scalar_t residual_tmp0 = u0_grad_0*u1_grad_1;
            const scalar_t residual_tmp1 = u0_grad_1*u2_grad_0;
            const scalar_t residual_tmp2 = u0_grad_2*u2_grad_1;
            const scalar_t residual_tmp3 = u1_grad_2*u2_grad_1;
            const scalar_t residual_tmp4 = u0_grad_1*u1_grad_0;
            const scalar_t residual_tmp5 = u0_grad_2*u2_grad_0;
            const scalar_t residual_tmp6 = u2_grad_2 + scalar_t(1);
            const scalar_t residual_tmp7 = -residual_tmp5 + residual_tmp6 + u0_grad_0*u2_grad_2 + u0_grad_0;
            const scalar_t residual_tmp8 = -residual_tmp3 + u1_grad_1*u2_grad_2 + u1_grad_1;
            const scalar_t residual_tmp9 = residual_tmp0 - residual_tmp4;
            const scalar_t residual_tmp10 = pow_m1(residual_tmp0*u2_grad_2 + residual_tmp1*u1_grad_2 + residual_tmp2*u1_grad_0 - residual_tmp3*u0_grad_0 - residual_tmp4*u2_grad_2 - residual_tmp5*u1_grad_1 + residual_tmp7 + residual_tmp8 + residual_tmp9);
            const scalar_t residual_tmp11 = -u0_grad_1*u1_grad_2 + u0_grad_2*u1_grad_1 + u0_grad_2;
            const scalar_t residual_tmp12 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp13 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
            const scalar_t residual_tmp14 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
            const scalar_t residual_tmp15 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
            const scalar_t residual_tmp16 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
            const scalar_t residual_tmp17 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
            const scalar_t residual_tmp18 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
            const scalar_t residual_tmp19 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
            const scalar_t residual_tmp20 = residual_tmp9 + u0_grad_0 + u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp21 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
            const scalar_t residual_tmp22 = residual_tmp6 + residual_tmp8;
            const scalar_t residual_tmp23 = residual_tmp11*residual_tmp12 + residual_tmp13*residual_tmp14 + residual_tmp15*residual_tmp16 + residual_tmp17*residual_tmp18 - residual_tmp19*residual_tmp20 - residual_tmp21*residual_tmp22;
            const scalar_t residual_tmp24 = -residual_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
            const scalar_t residual_tmp25 = -residual_tmp1 + u0_grad_0*u2_grad_1 + u2_grad_1;
            const scalar_t residual_tmp26 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp27 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
            const scalar_t residual_tmp28 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
            const scalar_t residual_tmp29 = residual_tmp12*residual_tmp24 - residual_tmp13*residual_tmp7 + residual_tmp16*residual_tmp26 + residual_tmp18*residual_tmp27 + residual_tmp19*residual_tmp25 - residual_tmp22*residual_tmp28;
            const scalar_t residual_tmp30 = residual_tmp24*residual_tmp28;
            const scalar_t residual_tmp31 = residual_tmp25*residual_tmp27;
            const scalar_t residual_tmp32 = residual_tmp11*residual_tmp21;
            const scalar_t residual_tmp33 = residual_tmp14*residual_tmp15;
            const scalar_t residual_tmp34 = residual_tmp26*residual_tmp7;
            const scalar_t residual_tmp35 = -residual_tmp34;
            const scalar_t residual_tmp36 = residual_tmp17*residual_tmp20;
            const scalar_t residual_tmp37 = -residual_tmp36;
            const scalar_t residual_tmp38 = residual_tmp30 + residual_tmp31 + residual_tmp32 + residual_tmp33 + residual_tmp35 + residual_tmp37;
            const scalar_t residual_tmp39 = residual_tmp12*residual_tmp22;
            const scalar_t residual_tmp40 = residual_tmp13*residual_tmp16 + residual_tmp18*residual_tmp19 - residual_tmp39;
            const scalar_t residual_tmp41 = scalar_t(3)*eta_b*(residual_tmp38 + residual_tmp40);
            const scalar_t residual_tmp42 = scalar_t(2)*eta_s;
            const scalar_t residual_tmp43 = residual_tmp41 + residual_tmp42*(scalar_t(2)*residual_tmp13*residual_tmp16 + scalar_t(2)*residual_tmp18*residual_tmp19 - residual_tmp38 - scalar_t(2)*residual_tmp39);
            const scalar_t residual_tmp44 = residual_tmp11*residual_tmp28 + residual_tmp14*residual_tmp26 - residual_tmp15*residual_tmp7 + residual_tmp17*residual_tmp25 - residual_tmp20*residual_tmp27 + residual_tmp21*residual_tmp24;
            const scalar_t residual_tmp45 = residual_tmp41 + residual_tmp42*(scalar_t(2)*residual_tmp24*residual_tmp28 + scalar_t(2)*residual_tmp25*residual_tmp27 - residual_tmp32 - residual_tmp33 - scalar_t(2)*residual_tmp34 - residual_tmp37 - residual_tmp40);
            const scalar_t residual_tmp46 = residual_tmp41 + residual_tmp42*(scalar_t(2)*residual_tmp11*residual_tmp21 + scalar_t(2)*residual_tmp14*residual_tmp15 - residual_tmp30 - residual_tmp31 - residual_tmp35 - scalar_t(2)*residual_tmp36 - residual_tmp40);
            const scalar_t grad_coeff0_0 = residual_tmp10*(eta_s*(residual_tmp11*residual_tmp23 + residual_tmp24*residual_tmp29) - (scalar_t(1) / scalar_t(3))*residual_tmp22*residual_tmp43);
            const scalar_t grad_coeff0_1 = residual_tmp10*(-eta_s*(-residual_tmp14*residual_tmp23 + residual_tmp29*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp16*residual_tmp43);
            const scalar_t grad_coeff0_2 = residual_tmp10*(-eta_s*(residual_tmp20*residual_tmp23 - residual_tmp25*residual_tmp29) + ((scalar_t(1) / scalar_t(3)))*residual_tmp18*residual_tmp43);
            const scalar_t grad_coeff1_0 = residual_tmp10*(-eta_s*(-residual_tmp11*residual_tmp44 + residual_tmp22*residual_tmp29) + ((scalar_t(1) / scalar_t(3)))*residual_tmp24*residual_tmp45);
            const scalar_t grad_coeff1_1 = residual_tmp10*(eta_s*(residual_tmp14*residual_tmp44 + residual_tmp16*residual_tmp29) - (scalar_t(1) / scalar_t(3))*residual_tmp45*residual_tmp7);
            const scalar_t grad_coeff1_2 = residual_tmp10*(-eta_s*(-residual_tmp18*residual_tmp29 + residual_tmp20*residual_tmp44) + ((scalar_t(1) / scalar_t(3)))*residual_tmp25*residual_tmp45);
            const scalar_t grad_coeff2_0 = residual_tmp10*(-eta_s*(residual_tmp22*residual_tmp23 - residual_tmp24*residual_tmp44) + ((scalar_t(1) / scalar_t(3)))*residual_tmp11*residual_tmp46);
            const scalar_t grad_coeff2_1 = residual_tmp10*(-eta_s*(-residual_tmp16*residual_tmp23 + residual_tmp44*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp14*residual_tmp46);
            const scalar_t grad_coeff2_2 = residual_tmp10*(eta_s*(residual_tmp18*residual_tmp23 + residual_tmp25*residual_tmp44) - (scalar_t(1) / scalar_t(3))*residual_tmp20*residual_tmp46);
            const scalar_t grad_coeff0_0_value = grad_coeff0_0;
            const scalar_t grad_coeff0_1_value = grad_coeff0_1;
            const scalar_t grad_coeff0_2_value = grad_coeff0_2;
            const scalar_t grad_coeff1_0_value = grad_coeff1_0;
            const scalar_t grad_coeff1_1_value = grad_coeff1_1;
            const scalar_t grad_coeff1_2_value = grad_coeff1_2;
            const scalar_t grad_coeff2_0_value = grad_coeff2_0;
            const scalar_t grad_coeff2_1_value = grad_coeff2_1;
            const scalar_t grad_coeff2_2_value = grad_coeff2_2;
            const scalar_t test0_grad0 = (-(adj0) - adj3 - adj6) / det;
            const scalar_t test0_grad1 = (-(adj1) - adj4 - adj7) / det;
            const scalar_t test0_grad2 = (-(adj2) - adj5 - adj8) / det;
            const scalar_t test1_grad0 = (adj0) / det;
            const scalar_t test1_grad1 = (adj1) / det;
            const scalar_t test1_grad2 = (adj2) / det;
            const scalar_t test2_grad0 = (adj3) / det;
            const scalar_t test2_grad1 = (adj4) / det;
            const scalar_t test2_grad2 = (adj5) / det;
            const scalar_t test3_grad0 = (adj6) / det;
            const scalar_t test3_grad1 = (adj7) / det;
            const scalar_t test3_grad2 = (adj8) / det;
            output[0][lane] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1 + grad_coeff0_2_value * test0_grad2);
            output[1][lane] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1 + grad_coeff1_2_value * test0_grad2);
            output[2][lane] += q_weight[q] * det * (grad_coeff2_0_value * test0_grad0 + grad_coeff2_1_value * test0_grad1 + grad_coeff2_2_value * test0_grad2);
            output[3][lane] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1 + grad_coeff0_2_value * test1_grad2);
            output[4][lane] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1 + grad_coeff1_2_value * test1_grad2);
            output[5][lane] += q_weight[q] * det * (grad_coeff2_0_value * test1_grad0 + grad_coeff2_1_value * test1_grad1 + grad_coeff2_2_value * test1_grad2);
            output[6][lane] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1 + grad_coeff0_2_value * test2_grad2);
            output[7][lane] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1 + grad_coeff1_2_value * test2_grad2);
            output[8][lane] += q_weight[q] * det * (grad_coeff2_0_value * test2_grad0 + grad_coeff2_1_value * test2_grad1 + grad_coeff2_2_value * test2_grad2);
            output[9][lane] += q_weight[q] * det * (grad_coeff0_0_value * test3_grad0 + grad_coeff0_1_value * test3_grad1 + grad_coeff0_2_value * test3_grad2);
            output[10][lane] += q_weight[q] * det * (grad_coeff1_0_value * test3_grad0 + grad_coeff1_1_value * test3_grad1 + grad_coeff1_2_value * test3_grad2);
            output[11][lane] += q_weight[q] * det * (grad_coeff2_0_value * test3_grad0 + grad_coeff2_1_value * test3_grad1 + grad_coeff2_2_value * test3_grad2);
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_tet4_residual_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t current[3 * N_SHAPE][VECTOR_SIZE],
        const scalar_t previous[3 * N_SHAPE][VECTOR_SIZE],
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t output[3 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 3;
    for (int q = 0; q < N_QP; ++q) {
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
            const scalar_t u0_grad_0_ref = -(current[0][lane]) + current[3][lane];
            const scalar_t u0_grad_1_ref = -(current[0][lane]) + current[6][lane];
            const scalar_t u0_grad_2_ref = -(current[0][lane]) + current[9][lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
            const scalar_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
            const scalar_t u0_old_grad_0_ref = -(previous[0][lane]) + previous[3][lane];
            const scalar_t u0_old_grad_1_ref = -(previous[0][lane]) + previous[6][lane];
            const scalar_t u0_old_grad_2_ref = -(previous[0][lane]) + previous[9][lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
            const scalar_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
            const scalar_t u1_grad_0_ref = -(current[1][lane]) + current[4][lane];
            const scalar_t u1_grad_1_ref = -(current[1][lane]) + current[7][lane];
            const scalar_t u1_grad_2_ref = -(current[1][lane]) + current[10][lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
            const scalar_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
            const scalar_t u1_old_grad_0_ref = -(previous[1][lane]) + previous[4][lane];
            const scalar_t u1_old_grad_1_ref = -(previous[1][lane]) + previous[7][lane];
            const scalar_t u1_old_grad_2_ref = -(previous[1][lane]) + previous[10][lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
            const scalar_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
            const scalar_t u2_grad_0_ref = -(current[2][lane]) + current[5][lane];
            const scalar_t u2_grad_1_ref = -(current[2][lane]) + current[8][lane];
            const scalar_t u2_grad_2_ref = -(current[2][lane]) + current[11][lane];
            const scalar_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
            const scalar_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
            const scalar_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
            const scalar_t u2_old_grad_0_ref = -(previous[2][lane]) + previous[5][lane];
            const scalar_t u2_old_grad_1_ref = -(previous[2][lane]) + previous[8][lane];
            const scalar_t u2_old_grad_2_ref = -(previous[2][lane]) + previous[11][lane];
            const scalar_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
            const scalar_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
            const scalar_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
            const scalar_t residual_tmp0 = u0_grad_0*u1_grad_1;
            const scalar_t residual_tmp1 = u0_grad_1*u2_grad_0;
            const scalar_t residual_tmp2 = u0_grad_2*u2_grad_1;
            const scalar_t residual_tmp3 = u1_grad_2*u2_grad_1;
            const scalar_t residual_tmp4 = u0_grad_1*u1_grad_0;
            const scalar_t residual_tmp5 = u0_grad_2*u2_grad_0;
            const scalar_t residual_tmp6 = u2_grad_2 + scalar_t(1);
            const scalar_t residual_tmp7 = -residual_tmp5 + residual_tmp6 + u0_grad_0*u2_grad_2 + u0_grad_0;
            const scalar_t residual_tmp8 = -residual_tmp3 + u1_grad_1*u2_grad_2 + u1_grad_1;
            const scalar_t residual_tmp9 = residual_tmp0 - residual_tmp4;
            const scalar_t residual_tmp10 = pow_m1(residual_tmp0*u2_grad_2 + residual_tmp1*u1_grad_2 + residual_tmp2*u1_grad_0 - residual_tmp3*u0_grad_0 - residual_tmp4*u2_grad_2 - residual_tmp5*u1_grad_1 + residual_tmp7 + residual_tmp8 + residual_tmp9);
            const scalar_t residual_tmp11 = -u0_grad_1*u1_grad_2 + u0_grad_2*u1_grad_1 + u0_grad_2;
            const scalar_t residual_tmp12 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp13 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
            const scalar_t residual_tmp14 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
            const scalar_t residual_tmp15 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
            const scalar_t residual_tmp16 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
            const scalar_t residual_tmp17 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
            const scalar_t residual_tmp18 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
            const scalar_t residual_tmp19 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
            const scalar_t residual_tmp20 = residual_tmp9 + u0_grad_0 + u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp21 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
            const scalar_t residual_tmp22 = residual_tmp6 + residual_tmp8;
            const scalar_t residual_tmp23 = residual_tmp11*residual_tmp12 + residual_tmp13*residual_tmp14 + residual_tmp15*residual_tmp16 + residual_tmp17*residual_tmp18 - residual_tmp19*residual_tmp20 - residual_tmp21*residual_tmp22;
            const scalar_t residual_tmp24 = -residual_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
            const scalar_t residual_tmp25 = -residual_tmp1 + u0_grad_0*u2_grad_1 + u2_grad_1;
            const scalar_t residual_tmp26 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp27 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
            const scalar_t residual_tmp28 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
            const scalar_t residual_tmp29 = residual_tmp12*residual_tmp24 - residual_tmp13*residual_tmp7 + residual_tmp16*residual_tmp26 + residual_tmp18*residual_tmp27 + residual_tmp19*residual_tmp25 - residual_tmp22*residual_tmp28;
            const scalar_t residual_tmp30 = residual_tmp24*residual_tmp28;
            const scalar_t residual_tmp31 = residual_tmp25*residual_tmp27;
            const scalar_t residual_tmp32 = residual_tmp11*residual_tmp21;
            const scalar_t residual_tmp33 = residual_tmp14*residual_tmp15;
            const scalar_t residual_tmp34 = residual_tmp26*residual_tmp7;
            const scalar_t residual_tmp35 = -residual_tmp34;
            const scalar_t residual_tmp36 = residual_tmp17*residual_tmp20;
            const scalar_t residual_tmp37 = -residual_tmp36;
            const scalar_t residual_tmp38 = residual_tmp30 + residual_tmp31 + residual_tmp32 + residual_tmp33 + residual_tmp35 + residual_tmp37;
            const scalar_t residual_tmp39 = residual_tmp12*residual_tmp22;
            const scalar_t residual_tmp40 = residual_tmp13*residual_tmp16 + residual_tmp18*residual_tmp19 - residual_tmp39;
            const scalar_t residual_tmp41 = scalar_t(3)*eta_b*(residual_tmp38 + residual_tmp40);
            const scalar_t residual_tmp42 = scalar_t(2)*eta_s;
            const scalar_t residual_tmp43 = residual_tmp41 + residual_tmp42*(scalar_t(2)*residual_tmp13*residual_tmp16 + scalar_t(2)*residual_tmp18*residual_tmp19 - residual_tmp38 - scalar_t(2)*residual_tmp39);
            const scalar_t residual_tmp44 = residual_tmp11*residual_tmp28 + residual_tmp14*residual_tmp26 - residual_tmp15*residual_tmp7 + residual_tmp17*residual_tmp25 - residual_tmp20*residual_tmp27 + residual_tmp21*residual_tmp24;
            const scalar_t residual_tmp45 = residual_tmp41 + residual_tmp42*(scalar_t(2)*residual_tmp24*residual_tmp28 + scalar_t(2)*residual_tmp25*residual_tmp27 - residual_tmp32 - residual_tmp33 - scalar_t(2)*residual_tmp34 - residual_tmp37 - residual_tmp40);
            const scalar_t residual_tmp46 = residual_tmp41 + residual_tmp42*(scalar_t(2)*residual_tmp11*residual_tmp21 + scalar_t(2)*residual_tmp14*residual_tmp15 - residual_tmp30 - residual_tmp31 - residual_tmp35 - scalar_t(2)*residual_tmp36 - residual_tmp40);
            const scalar_t grad_coeff0_0 = residual_tmp10*(eta_s*(residual_tmp11*residual_tmp23 + residual_tmp24*residual_tmp29) - (scalar_t(1) / scalar_t(3))*residual_tmp22*residual_tmp43);
            const scalar_t grad_coeff0_1 = residual_tmp10*(-eta_s*(-residual_tmp14*residual_tmp23 + residual_tmp29*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp16*residual_tmp43);
            const scalar_t grad_coeff0_2 = residual_tmp10*(-eta_s*(residual_tmp20*residual_tmp23 - residual_tmp25*residual_tmp29) + ((scalar_t(1) / scalar_t(3)))*residual_tmp18*residual_tmp43);
            const scalar_t grad_coeff1_0 = residual_tmp10*(-eta_s*(-residual_tmp11*residual_tmp44 + residual_tmp22*residual_tmp29) + ((scalar_t(1) / scalar_t(3)))*residual_tmp24*residual_tmp45);
            const scalar_t grad_coeff1_1 = residual_tmp10*(eta_s*(residual_tmp14*residual_tmp44 + residual_tmp16*residual_tmp29) - (scalar_t(1) / scalar_t(3))*residual_tmp45*residual_tmp7);
            const scalar_t grad_coeff1_2 = residual_tmp10*(-eta_s*(-residual_tmp18*residual_tmp29 + residual_tmp20*residual_tmp44) + ((scalar_t(1) / scalar_t(3)))*residual_tmp25*residual_tmp45);
            const scalar_t grad_coeff2_0 = residual_tmp10*(-eta_s*(residual_tmp22*residual_tmp23 - residual_tmp24*residual_tmp44) + ((scalar_t(1) / scalar_t(3)))*residual_tmp11*residual_tmp46);
            const scalar_t grad_coeff2_1 = residual_tmp10*(-eta_s*(-residual_tmp16*residual_tmp23 + residual_tmp44*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp14*residual_tmp46);
            const scalar_t grad_coeff2_2 = residual_tmp10*(eta_s*(residual_tmp18*residual_tmp23 + residual_tmp25*residual_tmp44) - (scalar_t(1) / scalar_t(3))*residual_tmp20*residual_tmp46);
            const scalar_t grad_coeff0_0_value = grad_coeff0_0;
            const scalar_t grad_coeff0_1_value = grad_coeff0_1;
            const scalar_t grad_coeff0_2_value = grad_coeff0_2;
            const scalar_t grad_coeff1_0_value = grad_coeff1_0;
            const scalar_t grad_coeff1_1_value = grad_coeff1_1;
            const scalar_t grad_coeff1_2_value = grad_coeff1_2;
            const scalar_t grad_coeff2_0_value = grad_coeff2_0;
            const scalar_t grad_coeff2_1_value = grad_coeff2_1;
            const scalar_t grad_coeff2_2_value = grad_coeff2_2;
            const scalar_t test0_grad0 = (-(adj0) - adj3 - adj6) / det;
            const scalar_t test0_grad1 = (-(adj1) - adj4 - adj7) / det;
            const scalar_t test0_grad2 = (-(adj2) - adj5 - adj8) / det;
            const scalar_t test1_grad0 = (adj0) / det;
            const scalar_t test1_grad1 = (adj1) / det;
            const scalar_t test1_grad2 = (adj2) / det;
            const scalar_t test2_grad0 = (adj3) / det;
            const scalar_t test2_grad1 = (adj4) / det;
            const scalar_t test2_grad2 = (adj5) / det;
            const scalar_t test3_grad0 = (adj6) / det;
            const scalar_t test3_grad1 = (adj7) / det;
            const scalar_t test3_grad2 = (adj8) / det;
            output[0][lane] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1 + grad_coeff0_2_value * test0_grad2);
            output[1][lane] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1 + grad_coeff1_2_value * test0_grad2);
            output[2][lane] += q_weight[q] * det * (grad_coeff2_0_value * test0_grad0 + grad_coeff2_1_value * test0_grad1 + grad_coeff2_2_value * test0_grad2);
            output[3][lane] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1 + grad_coeff0_2_value * test1_grad2);
            output[4][lane] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1 + grad_coeff1_2_value * test1_grad2);
            output[5][lane] += q_weight[q] * det * (grad_coeff2_0_value * test1_grad0 + grad_coeff2_1_value * test1_grad1 + grad_coeff2_2_value * test1_grad2);
            output[6][lane] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1 + grad_coeff0_2_value * test2_grad2);
            output[7][lane] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1 + grad_coeff1_2_value * test2_grad2);
            output[8][lane] += q_weight[q] * det * (grad_coeff2_0_value * test2_grad0 + grad_coeff2_1_value * test2_grad1 + grad_coeff2_2_value * test2_grad2);
            output[9][lane] += q_weight[q] * det * (grad_coeff0_0_value * test3_grad0 + grad_coeff0_1_value * test3_grad1 + grad_coeff0_2_value * test3_grad2);
            output[10][lane] += q_weight[q] * det * (grad_coeff1_0_value * test3_grad0 + grad_coeff1_1_value * test3_grad1 + grad_coeff1_2_value * test3_grad2);
            output[11][lane] += q_weight[q] * det * (grad_coeff2_0_value * test3_grad0 + grad_coeff2_1_value * test3_grad1 + grad_coeff2_2_value * test3_grad2);
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT current[3 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT previous[3 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT direction[3 * N_SHAPE],
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t *const SFEM_RESTRICT output[3 * N_SHAPE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 3;
    for (int q = 0; q < N_QP; ++q) {
        scalar_t u0_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u0_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u0_direction_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_direction_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u0_direction_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u1_direction_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_direction_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_direction_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u2_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u2_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u2_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u2_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u2_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u2_old_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u2_direction_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u2_direction_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u2_direction_grad_2_ref_values[VECTOR_SIZE];
        scalar_t grad_coeff0_0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_1_values[VECTOR_SIZE];
        scalar_t grad_coeff0_2_values[VECTOR_SIZE];
        scalar_t grad_coeff1_0_values[VECTOR_SIZE];
        scalar_t grad_coeff1_1_values[VECTOR_SIZE];
        scalar_t grad_coeff1_2_values[VECTOR_SIZE];
        scalar_t grad_coeff2_0_values[VECTOR_SIZE];
        scalar_t grad_coeff2_1_values[VECTOR_SIZE];
        scalar_t grad_coeff2_2_values[VECTOR_SIZE];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 0][lane];
                u0_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u0_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_old_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 0][lane];
                u0_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u0_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_direction_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 0][lane];
                u0_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u0_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 1][lane];
                u1_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u1_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_old_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 1][lane];
                u1_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u1_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_direction_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 1][lane];
                u1_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u1_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 2][lane];
                u2_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u2_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u2_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_old_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_old_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_old_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 2][lane];
                u2_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u2_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u2_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_direction_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_direction_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_direction_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 2][lane];
                u2_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u2_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u2_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
            const scalar_t u0_grad_0_ref = u0_grad_0_ref_values[lane];
            const scalar_t u0_grad_1_ref = u0_grad_1_ref_values[lane];
            const scalar_t u0_grad_2_ref = u0_grad_2_ref_values[lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
            const scalar_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
            const scalar_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[lane];
            const scalar_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[lane];
            const scalar_t u0_old_grad_2_ref = u0_old_grad_2_ref_values[lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
            const scalar_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
            const scalar_t u0_direction_grad_0_ref = u0_direction_grad_0_ref_values[lane];
            const scalar_t u0_direction_grad_1_ref = u0_direction_grad_1_ref_values[lane];
            const scalar_t u0_direction_grad_2_ref = u0_direction_grad_2_ref_values[lane];
            const scalar_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
            const scalar_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
            const scalar_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
            const scalar_t u1_grad_0_ref = u1_grad_0_ref_values[lane];
            const scalar_t u1_grad_1_ref = u1_grad_1_ref_values[lane];
            const scalar_t u1_grad_2_ref = u1_grad_2_ref_values[lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
            const scalar_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
            const scalar_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[lane];
            const scalar_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[lane];
            const scalar_t u1_old_grad_2_ref = u1_old_grad_2_ref_values[lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
            const scalar_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
            const scalar_t u1_direction_grad_0_ref = u1_direction_grad_0_ref_values[lane];
            const scalar_t u1_direction_grad_1_ref = u1_direction_grad_1_ref_values[lane];
            const scalar_t u1_direction_grad_2_ref = u1_direction_grad_2_ref_values[lane];
            const scalar_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
            const scalar_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
            const scalar_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
            const scalar_t u2_grad_0_ref = u2_grad_0_ref_values[lane];
            const scalar_t u2_grad_1_ref = u2_grad_1_ref_values[lane];
            const scalar_t u2_grad_2_ref = u2_grad_2_ref_values[lane];
            const scalar_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
            const scalar_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
            const scalar_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
            const scalar_t u2_old_grad_0_ref = u2_old_grad_0_ref_values[lane];
            const scalar_t u2_old_grad_1_ref = u2_old_grad_1_ref_values[lane];
            const scalar_t u2_old_grad_2_ref = u2_old_grad_2_ref_values[lane];
            const scalar_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
            const scalar_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
            const scalar_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
            const scalar_t u2_direction_grad_0_ref = u2_direction_grad_0_ref_values[lane];
            const scalar_t u2_direction_grad_1_ref = u2_direction_grad_1_ref_values[lane];
            const scalar_t u2_direction_grad_2_ref = u2_direction_grad_2_ref_values[lane];
            const scalar_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
            const scalar_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
            const scalar_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
            const scalar_t residual_tmp0 = u0_grad_0*u1_grad_1;
            const scalar_t residual_tmp1 = u0_grad_1*u1_grad_2;
            const scalar_t residual_tmp2 = u0_grad_2*u2_grad_1;
            const scalar_t residual_tmp3 = u1_grad_2*u2_grad_1;
            const scalar_t residual_tmp4 = u0_grad_1*u1_grad_0;
            const scalar_t residual_tmp5 = u0_grad_2*u2_grad_0;
            const scalar_t residual_tmp6 = u2_grad_2 + scalar_t(1);
            const scalar_t residual_tmp7 = -residual_tmp3 + residual_tmp6 + u1_grad_1*u2_grad_2 + u1_grad_1;
            const scalar_t residual_tmp8 = -residual_tmp5 + u0_grad_0*u2_grad_2 + u0_grad_0;
            const scalar_t residual_tmp9 = residual_tmp0 - residual_tmp4;
            const scalar_t residual_tmp10 = residual_tmp0*u2_grad_2 + residual_tmp1*u2_grad_0 + residual_tmp2*u1_grad_0 - residual_tmp3*u0_grad_0 - residual_tmp4*u2_grad_2 - residual_tmp5*u1_grad_1 + residual_tmp7 + residual_tmp8 + residual_tmp9;
            const scalar_t residual_tmp11 = pow_m1(residual_tmp10);
            const scalar_t residual_tmp12 = -residual_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
            const scalar_t residual_tmp13 = newmark_velocity_alpha*residual_tmp7;
            const scalar_t residual_tmp14 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
            const scalar_t residual_tmp15 = residual_tmp14*u2_grad_1;
            const scalar_t residual_tmp16 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp17 = residual_tmp16*residual_tmp6;
            const scalar_t residual_tmp18 = residual_tmp15 - residual_tmp17;
            const scalar_t residual_tmp19 = -residual_tmp13 - residual_tmp18;
            const scalar_t residual_tmp20 = -residual_tmp1 + u0_grad_2*u1_grad_1 + u0_grad_2;
            const scalar_t residual_tmp21 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
            const scalar_t residual_tmp22 = residual_tmp21*residual_tmp6;
            const scalar_t residual_tmp23 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
            const scalar_t residual_tmp24 = residual_tmp23*u2_grad_1;
            const scalar_t residual_tmp25 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
            const scalar_t residual_tmp26 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
            const scalar_t residual_tmp27 = residual_tmp25*u0_grad_1 - residual_tmp26*u0_grad_2;
            const scalar_t residual_tmp28 = residual_tmp22 - residual_tmp24 + residual_tmp27;
            const scalar_t residual_tmp29 = newmark_velocity_alpha*residual_tmp12;
            const scalar_t residual_tmp30 = residual_tmp23*u0_grad_1;
            const scalar_t residual_tmp31 = residual_tmp21*u0_grad_2;
            const scalar_t residual_tmp32 = residual_tmp29 + residual_tmp30 - residual_tmp31;
            const scalar_t residual_tmp33 = residual_tmp26*residual_tmp6;
            const scalar_t residual_tmp34 = residual_tmp25*u2_grad_1;
            const scalar_t residual_tmp35 = residual_tmp33 - residual_tmp34;
            const scalar_t residual_tmp36 = scalar_t(3)*eta_b;
            const scalar_t residual_tmp37 = residual_tmp36*(residual_tmp32 + residual_tmp35);
            const scalar_t residual_tmp38 = scalar_t(2)*eta_s;
            const scalar_t residual_tmp39 = residual_tmp37 + residual_tmp38*(scalar_t(2)*residual_tmp26*residual_tmp6 - residual_tmp32 - scalar_t(2)*residual_tmp34);
            const scalar_t residual_tmp40 = ((scalar_t(1) / scalar_t(3)))*residual_tmp7;
            const scalar_t residual_tmp41 = pow_m2(residual_tmp10);
            const scalar_t residual_tmp42 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp43 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
            const scalar_t residual_tmp44 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
            const scalar_t residual_tmp45 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
            const scalar_t residual_tmp46 = u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp47 = residual_tmp46 + residual_tmp9 + u0_grad_0;
            const scalar_t residual_tmp48 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
            const scalar_t residual_tmp49 = residual_tmp20*residual_tmp42 + residual_tmp21*residual_tmp44 + residual_tmp23*residual_tmp45 - residual_tmp25*residual_tmp47 + residual_tmp26*residual_tmp43 - residual_tmp48*residual_tmp7;
            const scalar_t residual_tmp50 = u0_grad_0*u2_grad_1 - u0_grad_1*u2_grad_0 + u2_grad_1;
            const scalar_t residual_tmp51 = residual_tmp6 + residual_tmp8;
            const scalar_t residual_tmp52 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
            const scalar_t residual_tmp53 = residual_tmp12*residual_tmp42 + residual_tmp14*residual_tmp45 + residual_tmp16*residual_tmp44 + residual_tmp25*residual_tmp50 - residual_tmp26*residual_tmp51 - residual_tmp52*residual_tmp7;
            const scalar_t residual_tmp54 = residual_tmp12*residual_tmp52;
            const scalar_t residual_tmp55 = residual_tmp14*residual_tmp50;
            const scalar_t residual_tmp56 = residual_tmp20*residual_tmp48;
            const scalar_t residual_tmp57 = residual_tmp21*residual_tmp43;
            const scalar_t residual_tmp58 = residual_tmp16*residual_tmp51;
            const scalar_t residual_tmp59 = -residual_tmp58;
            const scalar_t residual_tmp60 = residual_tmp23*residual_tmp47;
            const scalar_t residual_tmp61 = -residual_tmp60;
            const scalar_t residual_tmp62 = residual_tmp54 + residual_tmp55 + residual_tmp56 + residual_tmp57 + residual_tmp59 + residual_tmp61;
            const scalar_t residual_tmp63 = residual_tmp42*residual_tmp7;
            const scalar_t residual_tmp64 = residual_tmp25*residual_tmp45 + residual_tmp26*residual_tmp44 - residual_tmp63;
            const scalar_t residual_tmp65 = residual_tmp36*(residual_tmp62 + residual_tmp64);
            const scalar_t residual_tmp66 = residual_tmp38*(scalar_t(2)*residual_tmp25*residual_tmp45 + scalar_t(2)*residual_tmp26*residual_tmp44 - residual_tmp62 - scalar_t(2)*residual_tmp63) + residual_tmp65;
            const scalar_t residual_tmp67 = -residual_tmp66;
            const scalar_t residual_tmp68 = residual_tmp41*(eta_s*(residual_tmp12*residual_tmp53 + residual_tmp20*residual_tmp49) + residual_tmp40*residual_tmp67);
            const scalar_t residual_tmp69 = residual_tmp21*u1_grad_2;
            const scalar_t residual_tmp70 = residual_tmp23*residual_tmp46;
            const scalar_t residual_tmp71 = residual_tmp69 - residual_tmp70;
            const scalar_t residual_tmp72 = -residual_tmp13 - residual_tmp71;
            const scalar_t residual_tmp73 = -residual_tmp14*residual_tmp46 + residual_tmp16*u1_grad_2;
            const scalar_t residual_tmp74 = -residual_tmp27 - residual_tmp73;
            const scalar_t residual_tmp75 = newmark_velocity_alpha*residual_tmp20;
            const scalar_t residual_tmp76 = residual_tmp16*u0_grad_2;
            const scalar_t residual_tmp77 = residual_tmp14*u0_grad_1;
            const scalar_t residual_tmp78 = residual_tmp75 + residual_tmp76 - residual_tmp77;
            const scalar_t residual_tmp79 = residual_tmp25*residual_tmp46;
            const scalar_t residual_tmp80 = residual_tmp26*u1_grad_2;
            const scalar_t residual_tmp81 = residual_tmp79 - residual_tmp80;
            const scalar_t residual_tmp82 = residual_tmp36*(residual_tmp78 + residual_tmp81);
            const scalar_t residual_tmp83 = residual_tmp38*(scalar_t(2)*residual_tmp25*residual_tmp46 - residual_tmp78 - scalar_t(2)*residual_tmp80) + residual_tmp82;
            const scalar_t residual_tmp84 = -residual_tmp79 + residual_tmp80;
            const scalar_t residual_tmp85 = residual_tmp75 + residual_tmp84;
            const scalar_t residual_tmp86 = -residual_tmp33 + residual_tmp34;
            const scalar_t residual_tmp87 = residual_tmp29 + residual_tmp86;
            const scalar_t residual_tmp88 = residual_tmp13 - residual_tmp69 + residual_tmp70;
            const scalar_t residual_tmp89 = -residual_tmp15 + residual_tmp17;
            const scalar_t residual_tmp90 = residual_tmp36*(-residual_tmp88 - residual_tmp89);
            const scalar_t residual_tmp91 = residual_tmp38*(-scalar_t(2)*residual_tmp13 - residual_tmp18 - residual_tmp71) + residual_tmp90;
            const scalar_t residual_tmp92 = -residual_tmp7;
            const scalar_t residual_tmp93 = newmark_velocity_alpha*residual_tmp44;
            const scalar_t residual_tmp94 = residual_tmp23*u1_grad_0;
            const scalar_t residual_tmp95 = residual_tmp48*u1_grad_2;
            const scalar_t residual_tmp96 = residual_tmp93 + residual_tmp94 - residual_tmp95;
            const scalar_t residual_tmp97 = residual_tmp52*residual_tmp6;
            const scalar_t residual_tmp98 = residual_tmp14*u2_grad_0;
            const scalar_t residual_tmp99 = residual_tmp97 - residual_tmp98;
            const scalar_t residual_tmp100 = residual_tmp36*(residual_tmp96 + residual_tmp99);
            const scalar_t residual_tmp101 = -residual_tmp97 + residual_tmp98;
            const scalar_t residual_tmp102 = -residual_tmp94 + residual_tmp95;
            const scalar_t residual_tmp103 = residual_tmp100 + residual_tmp38*(residual_tmp101 + residual_tmp102 + scalar_t(2)*residual_tmp93);
            const scalar_t residual_tmp104 = newmark_velocity_alpha*residual_tmp43;
            const scalar_t residual_tmp105 = residual_tmp25*u1_grad_0;
            const scalar_t residual_tmp106 = residual_tmp42*u1_grad_2;
            const scalar_t residual_tmp107 = residual_tmp105 - residual_tmp106;
            const scalar_t residual_tmp108 = residual_tmp104 + residual_tmp107;
            const scalar_t residual_tmp109 = newmark_velocity_alpha*residual_tmp51;
            const scalar_t residual_tmp110 = residual_tmp25*u2_grad_0;
            const scalar_t residual_tmp111 = residual_tmp42*residual_tmp6;
            const scalar_t residual_tmp112 = residual_tmp110 - residual_tmp111;
            const scalar_t residual_tmp113 = -residual_tmp109 - residual_tmp112;
            const scalar_t residual_tmp114 = residual_tmp49*u1_grad_2;
            const scalar_t residual_tmp115 = -residual_tmp114;
            const scalar_t residual_tmp116 = residual_tmp53*residual_tmp6;
            const scalar_t residual_tmp117 = newmark_velocity_alpha*residual_tmp45;
            const scalar_t residual_tmp118 = residual_tmp46*residual_tmp48;
            const scalar_t residual_tmp119 = residual_tmp21*u1_grad_0;
            const scalar_t residual_tmp120 = residual_tmp117 + residual_tmp118 - residual_tmp119;
            const scalar_t residual_tmp121 = residual_tmp16*u2_grad_0;
            const scalar_t residual_tmp122 = residual_tmp52*u2_grad_1;
            const scalar_t residual_tmp123 = residual_tmp121 - residual_tmp122;
            const scalar_t residual_tmp124 = residual_tmp36*(residual_tmp120 + residual_tmp123);
            const scalar_t residual_tmp125 = -residual_tmp121 + residual_tmp122;
            const scalar_t residual_tmp126 = -residual_tmp118 + residual_tmp119;
            const scalar_t residual_tmp127 = residual_tmp124 + residual_tmp38*(scalar_t(2)*residual_tmp117 + residual_tmp125 + residual_tmp126);
            const scalar_t residual_tmp128 = newmark_velocity_alpha*residual_tmp50;
            const scalar_t residual_tmp129 = residual_tmp26*u2_grad_0;
            const scalar_t residual_tmp130 = residual_tmp42*u2_grad_1;
            const scalar_t residual_tmp131 = residual_tmp129 - residual_tmp130;
            const scalar_t residual_tmp132 = residual_tmp128 + residual_tmp131;
            const scalar_t residual_tmp133 = newmark_velocity_alpha*residual_tmp47;
            const scalar_t residual_tmp134 = residual_tmp26*u1_grad_0;
            const scalar_t residual_tmp135 = residual_tmp42*residual_tmp46;
            const scalar_t residual_tmp136 = residual_tmp134 - residual_tmp135;
            const scalar_t residual_tmp137 = -residual_tmp133 - residual_tmp136;
            const scalar_t residual_tmp138 = residual_tmp53*u2_grad_1;
            const scalar_t residual_tmp139 = -residual_tmp138;
            const scalar_t residual_tmp140 = residual_tmp46*residual_tmp49;
            const scalar_t residual_tmp141 = u0_grad_0 + scalar_t(1);
            const scalar_t residual_tmp142 = residual_tmp141*residual_tmp21;
            const scalar_t residual_tmp143 = residual_tmp48*u0_grad_1;
            const scalar_t residual_tmp144 = residual_tmp128 + residual_tmp142 - residual_tmp143;
            const scalar_t residual_tmp145 = -residual_tmp129 + residual_tmp130;
            const scalar_t residual_tmp146 = residual_tmp36*(residual_tmp144 + residual_tmp145);
            const scalar_t residual_tmp147 = residual_tmp146 + residual_tmp38*(-scalar_t(2)*residual_tmp129 - residual_tmp144 + scalar_t(2)*residual_tmp42*u2_grad_1);
            const scalar_t residual_tmp148 = residual_tmp117 + residual_tmp125;
            const scalar_t residual_tmp149 = residual_tmp21*u2_grad_0;
            const scalar_t residual_tmp150 = residual_tmp48*u2_grad_1;
            const scalar_t residual_tmp151 = -residual_tmp141*residual_tmp26 + residual_tmp42*u0_grad_1;
            const scalar_t residual_tmp152 = -residual_tmp149 + residual_tmp150 - residual_tmp151;
            const scalar_t residual_tmp153 = residual_tmp49*u0_grad_1;
            const scalar_t residual_tmp154 = ((scalar_t(1) / scalar_t(3)))*residual_tmp67;
            const scalar_t residual_tmp155 = residual_tmp14*residual_tmp141;
            const scalar_t residual_tmp156 = residual_tmp52*u0_grad_2;
            const scalar_t residual_tmp157 = residual_tmp104 + residual_tmp155 - residual_tmp156;
            const scalar_t residual_tmp158 = -residual_tmp105 + residual_tmp106;
            const scalar_t residual_tmp159 = residual_tmp36*(residual_tmp157 + residual_tmp158);
            const scalar_t residual_tmp160 = residual_tmp159 + residual_tmp38*(-scalar_t(2)*residual_tmp105 - residual_tmp157 + scalar_t(2)*residual_tmp42*u1_grad_2);
            const scalar_t residual_tmp161 = residual_tmp102 + residual_tmp93;
            const scalar_t residual_tmp162 = -residual_tmp141*residual_tmp25 + residual_tmp42*u0_grad_2;
            const scalar_t residual_tmp163 = residual_tmp14*u1_grad_0 - residual_tmp52*u1_grad_2;
            const scalar_t residual_tmp164 = -residual_tmp162 - residual_tmp163;
            const scalar_t residual_tmp165 = residual_tmp53*u0_grad_2;
            const scalar_t residual_tmp166 = -residual_tmp51;
            const scalar_t residual_tmp167 = residual_tmp141*residual_tmp23;
            const scalar_t residual_tmp168 = residual_tmp48*u0_grad_2;
            const scalar_t residual_tmp169 = residual_tmp109 + residual_tmp167 - residual_tmp168;
            const scalar_t residual_tmp170 = -residual_tmp110 + residual_tmp111;
            const scalar_t residual_tmp171 = residual_tmp36*(-residual_tmp169 - residual_tmp170);
            const scalar_t residual_tmp172 = residual_tmp171 + residual_tmp38*(scalar_t(2)*residual_tmp110 - scalar_t(2)*residual_tmp111 + residual_tmp169);
            const scalar_t residual_tmp173 = residual_tmp101 + residual_tmp93;
            const scalar_t residual_tmp174 = residual_tmp23*u2_grad_0;
            const scalar_t residual_tmp175 = residual_tmp48*residual_tmp6;
            const scalar_t residual_tmp176 = residual_tmp162 + residual_tmp174 - residual_tmp175;
            const scalar_t residual_tmp177 = residual_tmp49*u0_grad_2;
            const scalar_t residual_tmp178 = -residual_tmp47;
            const scalar_t residual_tmp179 = residual_tmp141*residual_tmp16;
            const scalar_t residual_tmp180 = residual_tmp52*u0_grad_1;
            const scalar_t residual_tmp181 = residual_tmp133 + residual_tmp179 - residual_tmp180;
            const scalar_t residual_tmp182 = -residual_tmp134 + residual_tmp135;
            const scalar_t residual_tmp183 = residual_tmp36*(-residual_tmp181 - residual_tmp182);
            const scalar_t residual_tmp184 = residual_tmp183 + residual_tmp38*(scalar_t(2)*residual_tmp134 - scalar_t(2)*residual_tmp135 + residual_tmp181);
            const scalar_t residual_tmp185 = residual_tmp117 + residual_tmp126;
            const scalar_t residual_tmp186 = residual_tmp16*u1_grad_0 - residual_tmp46*residual_tmp52;
            const scalar_t residual_tmp187 = residual_tmp151 + residual_tmp186;
            const scalar_t residual_tmp188 = residual_tmp53*u0_grad_1;
            const scalar_t residual_tmp189 = residual_tmp41*(-eta_s*(-residual_tmp43*residual_tmp49 + residual_tmp51*residual_tmp53) + ((scalar_t(1) / scalar_t(3)))*residual_tmp44*residual_tmp66);
            const scalar_t residual_tmp190 = residual_tmp49*u1_grad_0;
            const scalar_t residual_tmp191 = residual_tmp53*u2_grad_0;
            const scalar_t residual_tmp192 = -residual_tmp191;
            const scalar_t residual_tmp193 = residual_tmp190 + residual_tmp192;
            const scalar_t residual_tmp194 = -residual_tmp53*residual_tmp6;
            const scalar_t residual_tmp195 = residual_tmp141*residual_tmp49;
            const scalar_t residual_tmp196 = ((scalar_t(1) / scalar_t(3)))*residual_tmp66;
            const scalar_t residual_tmp197 = residual_tmp196*u2_grad_0;
            const scalar_t residual_tmp198 = ((scalar_t(1) / scalar_t(3)))*residual_tmp44;
            const scalar_t residual_tmp199 = residual_tmp141*residual_tmp53;
            const scalar_t residual_tmp200 = residual_tmp196*u1_grad_0;
            const scalar_t residual_tmp201 = residual_tmp41*(-eta_s*(residual_tmp47*residual_tmp49 - residual_tmp50*residual_tmp53) + ((scalar_t(1) / scalar_t(3)))*residual_tmp45*residual_tmp66);
            const scalar_t residual_tmp202 = -residual_tmp46*residual_tmp49;
            const scalar_t residual_tmp203 = ((scalar_t(1) / scalar_t(3)))*residual_tmp45;
            const scalar_t residual_tmp204 = -residual_tmp76 + residual_tmp77;
            const scalar_t residual_tmp205 = residual_tmp204 + residual_tmp75;
            const scalar_t residual_tmp206 = -residual_tmp30 + residual_tmp31;
            const scalar_t residual_tmp207 = residual_tmp37 + residual_tmp38*(residual_tmp206 + scalar_t(2)*residual_tmp29 + residual_tmp86);
            const scalar_t residual_tmp208 = residual_tmp12*residual_tmp48 - residual_tmp14*residual_tmp47 + residual_tmp16*residual_tmp43 + residual_tmp20*residual_tmp52 - residual_tmp21*residual_tmp51 + residual_tmp23*residual_tmp50;
            const scalar_t residual_tmp209 = residual_tmp38*(scalar_t(2)*residual_tmp12*residual_tmp52 + scalar_t(2)*residual_tmp14*residual_tmp50 - residual_tmp56 - residual_tmp57 - scalar_t(2)*residual_tmp58 - residual_tmp61 - residual_tmp64) + residual_tmp65;
            const scalar_t residual_tmp210 = residual_tmp41*(-eta_s*(-residual_tmp20*residual_tmp208 + residual_tmp53*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp209);
            const scalar_t residual_tmp211 = residual_tmp206 + residual_tmp29;
            const scalar_t residual_tmp212 = residual_tmp38*(scalar_t(2)*residual_tmp16*u0_grad_2 - residual_tmp75 - scalar_t(2)*residual_tmp77 - residual_tmp81) + residual_tmp82;
            const scalar_t residual_tmp213 = -residual_tmp22 + residual_tmp24 + residual_tmp73;
            const scalar_t residual_tmp214 = residual_tmp38*(scalar_t(2)*residual_tmp15 - scalar_t(2)*residual_tmp17 + residual_tmp88) + residual_tmp90;
            const scalar_t residual_tmp215 = -residual_tmp142 + residual_tmp143;
            const scalar_t residual_tmp216 = residual_tmp146 + residual_tmp38*(scalar_t(2)*residual_tmp128 + residual_tmp131 + residual_tmp215);
            const scalar_t residual_tmp217 = -residual_tmp179 + residual_tmp180;
            const scalar_t residual_tmp218 = -residual_tmp133 - residual_tmp217;
            const scalar_t residual_tmp219 = residual_tmp208*u0_grad_1;
            const scalar_t residual_tmp220 = residual_tmp139 + residual_tmp219;
            const scalar_t residual_tmp221 = -residual_tmp167 + residual_tmp168;
            const scalar_t residual_tmp222 = residual_tmp171 + residual_tmp38*(-scalar_t(2)*residual_tmp109 - residual_tmp112 - residual_tmp221);
            const scalar_t residual_tmp223 = -residual_tmp155 + residual_tmp156;
            const scalar_t residual_tmp224 = residual_tmp104 + residual_tmp223;
            const scalar_t residual_tmp225 = residual_tmp208*u0_grad_2;
            const scalar_t residual_tmp226 = residual_tmp124 + residual_tmp38*(-residual_tmp120 - scalar_t(2)*residual_tmp122 + scalar_t(2)*residual_tmp16*u2_grad_0);
            const scalar_t residual_tmp227 = residual_tmp149 - residual_tmp150 - residual_tmp186;
            const scalar_t residual_tmp228 = residual_tmp208*residual_tmp46;
            const scalar_t residual_tmp229 = ((scalar_t(1) / scalar_t(3)))*residual_tmp209;
            const scalar_t residual_tmp230 = residual_tmp229*u2_grad_1;
            const scalar_t residual_tmp231 = residual_tmp159 + residual_tmp38*(-residual_tmp104 + scalar_t(2)*residual_tmp14*residual_tmp141 - scalar_t(2)*residual_tmp156 - residual_tmp158);
            const scalar_t residual_tmp232 = -residual_tmp109 - residual_tmp221;
            const scalar_t residual_tmp233 = residual_tmp53*u1_grad_2;
            const scalar_t residual_tmp234 = residual_tmp100 + residual_tmp38*(scalar_t(2)*residual_tmp52*residual_tmp6 - residual_tmp96 - scalar_t(2)*residual_tmp98);
            const scalar_t residual_tmp235 = ((scalar_t(1) / scalar_t(3)))*residual_tmp12;
            const scalar_t residual_tmp236 = residual_tmp163 - residual_tmp174 + residual_tmp175;
            const scalar_t residual_tmp237 = residual_tmp208*u1_grad_2;
            const scalar_t residual_tmp238 = residual_tmp183 + residual_tmp38*(residual_tmp133 - scalar_t(2)*residual_tmp179 + scalar_t(2)*residual_tmp180 + residual_tmp182);
            const scalar_t residual_tmp239 = residual_tmp128 + residual_tmp215;
            const scalar_t residual_tmp240 = residual_tmp46*residual_tmp53;
            const scalar_t residual_tmp241 = residual_tmp229*u0_grad_1;
            const scalar_t residual_tmp242 = ((scalar_t(1) / scalar_t(3)))*residual_tmp51;
            const scalar_t residual_tmp243 = -(scalar_t(1) / scalar_t(3))*residual_tmp209;
            const scalar_t residual_tmp244 = residual_tmp41*(eta_s*(residual_tmp208*residual_tmp43 + residual_tmp44*residual_tmp53) + residual_tmp243*residual_tmp51);
            const scalar_t residual_tmp245 = residual_tmp141*residual_tmp208;
            const scalar_t residual_tmp246 = residual_tmp208*u1_grad_0;
            const scalar_t residual_tmp247 = residual_tmp53*u1_grad_0;
            const scalar_t residual_tmp248 = residual_tmp41*(-eta_s*(residual_tmp208*residual_tmp47 - residual_tmp45*residual_tmp53) + ((scalar_t(1) / scalar_t(3)))*residual_tmp209*residual_tmp50);
            const scalar_t residual_tmp249 = -residual_tmp141*residual_tmp208;
            const scalar_t residual_tmp250 = ((scalar_t(1) / scalar_t(3)))*residual_tmp50;
            const scalar_t residual_tmp251 = residual_tmp38*(residual_tmp204 + scalar_t(2)*residual_tmp75 + residual_tmp84) + residual_tmp82;
            const scalar_t residual_tmp252 = residual_tmp38*(scalar_t(2)*residual_tmp20*residual_tmp48 + scalar_t(2)*residual_tmp21*residual_tmp43 - residual_tmp54 - residual_tmp55 - residual_tmp59 - scalar_t(2)*residual_tmp60 - residual_tmp64) + residual_tmp65;
            const scalar_t residual_tmp253 = residual_tmp41*(-eta_s*(-residual_tmp12*residual_tmp208 + residual_tmp49*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp252);
            const scalar_t residual_tmp254 = residual_tmp37 + residual_tmp38*(scalar_t(2)*residual_tmp23*u0_grad_1 - residual_tmp29 - scalar_t(2)*residual_tmp31 - residual_tmp35);
            const scalar_t residual_tmp255 = residual_tmp38*(residual_tmp13 + scalar_t(2)*residual_tmp69 - scalar_t(2)*residual_tmp70 + residual_tmp89) + residual_tmp90;
            const scalar_t residual_tmp256 = residual_tmp159 + residual_tmp38*(scalar_t(2)*residual_tmp104 + residual_tmp107 + residual_tmp223);
            const scalar_t residual_tmp257 = residual_tmp115 + residual_tmp225;
            const scalar_t residual_tmp258 = residual_tmp183 + residual_tmp38*(-scalar_t(2)*residual_tmp133 - residual_tmp136 - residual_tmp217);
            const scalar_t residual_tmp259 = residual_tmp100 + residual_tmp38*(scalar_t(2)*residual_tmp23*u1_grad_0 - residual_tmp93 - scalar_t(2)*residual_tmp95 - residual_tmp99);
            const scalar_t residual_tmp260 = residual_tmp208*residual_tmp6;
            const scalar_t residual_tmp261 = ((scalar_t(1) / scalar_t(3)))*residual_tmp252;
            const scalar_t residual_tmp262 = residual_tmp261*u1_grad_2;
            const scalar_t residual_tmp263 = residual_tmp146 + residual_tmp38*(-residual_tmp128 + scalar_t(2)*residual_tmp141*residual_tmp21 - scalar_t(2)*residual_tmp143 - residual_tmp145);
            const scalar_t residual_tmp264 = residual_tmp49*u2_grad_1;
            const scalar_t residual_tmp265 = residual_tmp124 + residual_tmp38*(-residual_tmp117 - scalar_t(2)*residual_tmp119 - residual_tmp123 + scalar_t(2)*residual_tmp46*residual_tmp48);
            const scalar_t residual_tmp266 = ((scalar_t(1) / scalar_t(3)))*residual_tmp20;
            const scalar_t residual_tmp267 = residual_tmp208*u2_grad_1;
            const scalar_t residual_tmp268 = residual_tmp171 + residual_tmp38*(residual_tmp109 - scalar_t(2)*residual_tmp167 + scalar_t(2)*residual_tmp168 + residual_tmp170);
            const scalar_t residual_tmp269 = residual_tmp49*residual_tmp6;
            const scalar_t residual_tmp270 = residual_tmp261*u0_grad_2;
            const scalar_t residual_tmp271 = residual_tmp41*(-eta_s*(residual_tmp208*residual_tmp51 - residual_tmp44*residual_tmp49) + ((scalar_t(1) / scalar_t(3)))*residual_tmp252*residual_tmp43);
            const scalar_t residual_tmp272 = residual_tmp208*u2_grad_0;
            const scalar_t residual_tmp273 = ((scalar_t(1) / scalar_t(3)))*residual_tmp43;
            const scalar_t residual_tmp274 = residual_tmp49*u2_grad_0;
            const scalar_t residual_tmp275 = ((scalar_t(1) / scalar_t(3)))*residual_tmp47;
            const scalar_t residual_tmp276 = -(scalar_t(1) / scalar_t(3))*residual_tmp252;
            const scalar_t residual_tmp277 = residual_tmp41*(eta_s*(residual_tmp208*residual_tmp50 + residual_tmp45*residual_tmp49) + residual_tmp276*residual_tmp47);
            const scalar_t grad_coeff0_0 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp87 + residual_tmp20*residual_tmp85) - residual_tmp40*residual_tmp91) + residual_tmp68*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp108*residual_tmp20 + residual_tmp113*residual_tmp12 + residual_tmp115 + residual_tmp116) - residual_tmp103*residual_tmp40) + residual_tmp44*residual_tmp68) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp132 + residual_tmp137*residual_tmp20 + residual_tmp139 + residual_tmp140) - residual_tmp127*residual_tmp40) + residual_tmp45*residual_tmp68) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp19 + residual_tmp20*residual_tmp28) - residual_tmp39*residual_tmp40) + residual_tmp12*residual_tmp68) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp173 + residual_tmp176*residual_tmp20 + residual_tmp177) + residual_tmp154*residual_tmp6 - residual_tmp172*residual_tmp40) + residual_tmp166*residual_tmp68) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp148 + residual_tmp152*residual_tmp20 - residual_tmp153) - residual_tmp147*residual_tmp40 - residual_tmp154*u2_grad_1) + residual_tmp50*residual_tmp68) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp74 + residual_tmp20*residual_tmp72) - residual_tmp40*residual_tmp83) + residual_tmp20*residual_tmp68) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp164 + residual_tmp161*residual_tmp20 - residual_tmp165) - residual_tmp154*u1_grad_2 - residual_tmp160*residual_tmp40) + residual_tmp43*residual_tmp68) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp187 + residual_tmp185*residual_tmp20 + residual_tmp188) + residual_tmp154*residual_tmp46 - residual_tmp184*residual_tmp40) + residual_tmp178*residual_tmp68);
            const scalar_t grad_coeff0_1 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp114 - residual_tmp194 - residual_tmp43*residual_tmp85 + residual_tmp51*residual_tmp87) + ((scalar_t(1) / scalar_t(3)))*residual_tmp44*residual_tmp91) + residual_tmp189*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp108*residual_tmp43 + residual_tmp113*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp103*residual_tmp44) + residual_tmp189*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp132*residual_tmp51 - residual_tmp137*residual_tmp43 + residual_tmp193) + ((scalar_t(1) / scalar_t(3)))*residual_tmp127*residual_tmp44) + residual_tmp189*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp177 + residual_tmp19*residual_tmp51 - residual_tmp28*residual_tmp43) + residual_tmp196*residual_tmp6 + residual_tmp198*residual_tmp39) + residual_tmp12*residual_tmp189) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp173*residual_tmp51 - residual_tmp176*residual_tmp43) + ((scalar_t(1) / scalar_t(3)))*residual_tmp172*residual_tmp44) + residual_tmp166*residual_tmp189) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp148*residual_tmp51 - residual_tmp152*residual_tmp43 - residual_tmp195) + ((scalar_t(1) / scalar_t(3)))*residual_tmp147*residual_tmp44 - residual_tmp197) + residual_tmp189*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp165 - residual_tmp43*residual_tmp72 + residual_tmp51*residual_tmp74) - residual_tmp196*u1_grad_2 + ((scalar_t(1) / scalar_t(3)))*residual_tmp44*residual_tmp83) + residual_tmp189*residual_tmp20) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp161*residual_tmp43 + residual_tmp164*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp160*residual_tmp44) + residual_tmp189*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp185*residual_tmp43 + residual_tmp187*residual_tmp51 + residual_tmp199) + residual_tmp184*residual_tmp198 + residual_tmp200) + residual_tmp178*residual_tmp189);
            const scalar_t grad_coeff0_2 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp138 - residual_tmp202 + residual_tmp47*residual_tmp85 - residual_tmp50*residual_tmp87) + ((scalar_t(1) / scalar_t(3)))*residual_tmp45*residual_tmp91) + residual_tmp201*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp108*residual_tmp47 - residual_tmp113*residual_tmp50 - residual_tmp193) + ((scalar_t(1) / scalar_t(3)))*residual_tmp103*residual_tmp45) + residual_tmp201*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp132*residual_tmp50 + residual_tmp137*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp127*residual_tmp45) + residual_tmp201*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp153 - residual_tmp19*residual_tmp50 + residual_tmp28*residual_tmp47) - residual_tmp196*u2_grad_1 + ((scalar_t(1) / scalar_t(3)))*residual_tmp39*residual_tmp45) + residual_tmp12*residual_tmp201) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp173*residual_tmp50 + residual_tmp176*residual_tmp47 + residual_tmp195) + residual_tmp172*residual_tmp203 + residual_tmp197) + residual_tmp166*residual_tmp201) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp148*residual_tmp50 + residual_tmp152*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp147*residual_tmp45) + residual_tmp201*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp188 + residual_tmp47*residual_tmp72 - residual_tmp50*residual_tmp74) + residual_tmp196*residual_tmp46 + residual_tmp203*residual_tmp83) + residual_tmp20*residual_tmp201) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp161*residual_tmp47 - residual_tmp164*residual_tmp50 - residual_tmp199) + ((scalar_t(1) / scalar_t(3)))*residual_tmp160*residual_tmp45 - residual_tmp200) + residual_tmp201*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp185*residual_tmp47 - residual_tmp187*residual_tmp50) + ((scalar_t(1) / scalar_t(3)))*residual_tmp184*residual_tmp45) + residual_tmp178*residual_tmp201);
            const scalar_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp20*residual_tmp213 + residual_tmp7*residual_tmp87) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp214) + residual_tmp210*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp113*residual_tmp7 - residual_tmp20*residual_tmp236 + residual_tmp237) + residual_tmp229*residual_tmp6 + residual_tmp234*residual_tmp235) + residual_tmp210*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp132*residual_tmp7 - residual_tmp20*residual_tmp227 - residual_tmp228) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp226 - residual_tmp230) + residual_tmp210*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp19*residual_tmp7 - residual_tmp20*residual_tmp205) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp207) + residual_tmp12*residual_tmp210) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp173*residual_tmp7 - residual_tmp194 - residual_tmp20*residual_tmp224 - residual_tmp225) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp222) + residual_tmp166*residual_tmp210) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp148*residual_tmp7 - residual_tmp20*residual_tmp218 + residual_tmp220) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp216) + residual_tmp210*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp20*residual_tmp211 + residual_tmp7*residual_tmp74) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp212) + residual_tmp20*residual_tmp210) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp164*residual_tmp7 - residual_tmp20*residual_tmp232 - residual_tmp233) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp231 - residual_tmp229*u0_grad_2) + residual_tmp210*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp187*residual_tmp7 - residual_tmp20*residual_tmp239 + residual_tmp240) + residual_tmp235*residual_tmp238 + residual_tmp241) + residual_tmp178*residual_tmp210);
            const scalar_t grad_coeff1_1 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp213*residual_tmp43 + residual_tmp237 + residual_tmp44*residual_tmp87) - residual_tmp214*residual_tmp242 + residual_tmp243*residual_tmp6) + residual_tmp244*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp113*residual_tmp44 + residual_tmp236*residual_tmp43) - residual_tmp234*residual_tmp242) + residual_tmp244*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp132*residual_tmp44 + residual_tmp227*residual_tmp43 - residual_tmp246) - residual_tmp226*residual_tmp242 - residual_tmp243*u2_grad_0) + residual_tmp244*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp116 + residual_tmp19*residual_tmp44 + residual_tmp205*residual_tmp43 - residual_tmp225) - residual_tmp207*residual_tmp242) + residual_tmp12*residual_tmp244) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp173*residual_tmp44 + residual_tmp224*residual_tmp43) - residual_tmp222*residual_tmp242) + residual_tmp166*residual_tmp244) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp148*residual_tmp44 + residual_tmp192 + residual_tmp218*residual_tmp43 + residual_tmp245) - residual_tmp216*residual_tmp242) + residual_tmp244*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp211*residual_tmp43 - residual_tmp233 + residual_tmp44*residual_tmp74) - residual_tmp212*residual_tmp242 - residual_tmp243*u0_grad_2) + residual_tmp20*residual_tmp244) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp164*residual_tmp44 + residual_tmp232*residual_tmp43) - residual_tmp231*residual_tmp242) + residual_tmp244*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp187*residual_tmp44 + residual_tmp239*residual_tmp43 + residual_tmp247) + residual_tmp141*residual_tmp243 - residual_tmp238*residual_tmp242) + residual_tmp178*residual_tmp244);
            const scalar_t grad_coeff1_2 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp213*residual_tmp47 + residual_tmp228 - residual_tmp45*residual_tmp87) + residual_tmp214*residual_tmp250 + residual_tmp230) + residual_tmp248*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp113*residual_tmp45 + residual_tmp236*residual_tmp47 - residual_tmp246) - residual_tmp229*u2_grad_0 + ((scalar_t(1) / scalar_t(3)))*residual_tmp234*residual_tmp50) + residual_tmp248*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp132*residual_tmp45 + residual_tmp227*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp226*residual_tmp50) + residual_tmp248*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp19*residual_tmp45 + residual_tmp205*residual_tmp47 - residual_tmp220) + ((scalar_t(1) / scalar_t(3)))*residual_tmp207*residual_tmp50) + residual_tmp12*residual_tmp248) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp173*residual_tmp45 - residual_tmp191 + residual_tmp224*residual_tmp47 - residual_tmp249) + ((scalar_t(1) / scalar_t(3)))*residual_tmp222*residual_tmp50) + residual_tmp166*residual_tmp248) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp148*residual_tmp45 + residual_tmp218*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp216*residual_tmp50) + residual_tmp248*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp211*residual_tmp47 - residual_tmp240 - residual_tmp45*residual_tmp74) + ((scalar_t(1) / scalar_t(3)))*residual_tmp212*residual_tmp50 - residual_tmp241) + residual_tmp20*residual_tmp248) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp164*residual_tmp45 + residual_tmp232*residual_tmp47 + residual_tmp247) + residual_tmp141*residual_tmp229 + residual_tmp231*residual_tmp250) + residual_tmp248*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp187*residual_tmp45 + residual_tmp239*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp238*residual_tmp50) + residual_tmp178*residual_tmp248);
            const scalar_t grad_coeff2_0 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp213 + residual_tmp7*residual_tmp85) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp255) + residual_tmp253*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp108*residual_tmp7 - residual_tmp12*residual_tmp236 - residual_tmp260) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp259 - residual_tmp262) + residual_tmp253*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp227 + residual_tmp137*residual_tmp7 + residual_tmp267) + residual_tmp261*residual_tmp46 + residual_tmp265*residual_tmp266) + residual_tmp253*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp205 + residual_tmp28*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp254) + residual_tmp12*residual_tmp253) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp224 + residual_tmp176*residual_tmp7 + residual_tmp269) + residual_tmp266*residual_tmp268 + residual_tmp270) + residual_tmp166*residual_tmp253) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp218 + residual_tmp152*residual_tmp7 - residual_tmp264) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp263 - residual_tmp261*u0_grad_1) + residual_tmp253*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp211 + residual_tmp7*residual_tmp72) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp251) + residual_tmp20*residual_tmp253) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp232 + residual_tmp161*residual_tmp7 + residual_tmp257) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp256) + residual_tmp253*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp239 + residual_tmp185*residual_tmp7 - residual_tmp202 - residual_tmp219) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp258) + residual_tmp178*residual_tmp253);
            const scalar_t grad_coeff2_1 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp213*residual_tmp51 + residual_tmp260 - residual_tmp44*residual_tmp85) + residual_tmp255*residual_tmp273 + residual_tmp262) + residual_tmp271*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp108*residual_tmp44 + residual_tmp236*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp259*residual_tmp43) + residual_tmp271*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp137*residual_tmp44 + residual_tmp227*residual_tmp51 - residual_tmp272) - residual_tmp261*u1_grad_0 + ((scalar_t(1) / scalar_t(3)))*residual_tmp265*residual_tmp43) + residual_tmp271*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp205*residual_tmp51 - residual_tmp269 - residual_tmp28*residual_tmp44) + ((scalar_t(1) / scalar_t(3)))*residual_tmp254*residual_tmp43 - residual_tmp270) + residual_tmp12*residual_tmp271) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp176*residual_tmp44 + residual_tmp224*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp268*residual_tmp43) + residual_tmp166*residual_tmp271) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp152*residual_tmp44 + residual_tmp218*residual_tmp51 + residual_tmp274) + residual_tmp141*residual_tmp261 + residual_tmp263*residual_tmp273) + residual_tmp271*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp211*residual_tmp51 - residual_tmp257 - residual_tmp44*residual_tmp72) + ((scalar_t(1) / scalar_t(3)))*residual_tmp251*residual_tmp43) + residual_tmp20*residual_tmp271) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp161*residual_tmp44 + residual_tmp232*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp256*residual_tmp43) + residual_tmp271*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp185*residual_tmp44 - residual_tmp190 + residual_tmp239*residual_tmp51 - residual_tmp249) + ((scalar_t(1) / scalar_t(3)))*residual_tmp258*residual_tmp43) + residual_tmp178*residual_tmp271);
            const scalar_t grad_coeff2_2 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp213*residual_tmp50 + residual_tmp267 + residual_tmp45*residual_tmp85) - residual_tmp255*residual_tmp275 + residual_tmp276*residual_tmp46) + residual_tmp277*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp108*residual_tmp45 + residual_tmp236*residual_tmp50 - residual_tmp272) - residual_tmp259*residual_tmp275 - residual_tmp276*u1_grad_0) + residual_tmp277*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp137*residual_tmp45 + residual_tmp227*residual_tmp50) - residual_tmp265*residual_tmp275) + residual_tmp277*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp205*residual_tmp50 - residual_tmp264 + residual_tmp28*residual_tmp45) - residual_tmp254*residual_tmp275 - residual_tmp276*u0_grad_1) + residual_tmp12*residual_tmp277) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp176*residual_tmp45 + residual_tmp224*residual_tmp50 + residual_tmp274) + residual_tmp141*residual_tmp276 - residual_tmp268*residual_tmp275) + residual_tmp166*residual_tmp277) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp152*residual_tmp45 + residual_tmp218*residual_tmp50) - residual_tmp263*residual_tmp275) + residual_tmp277*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp140 + residual_tmp211*residual_tmp50 - residual_tmp219 + residual_tmp45*residual_tmp72) - residual_tmp251*residual_tmp275) + residual_tmp20*residual_tmp277) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp161*residual_tmp45 - residual_tmp190 + residual_tmp232*residual_tmp50 + residual_tmp245) - residual_tmp256*residual_tmp275) + residual_tmp277*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp185*residual_tmp45 + residual_tmp239*residual_tmp50) - residual_tmp258*residual_tmp275) + residual_tmp178*residual_tmp277);
            grad_coeff0_0_values[lane] = grad_coeff0_0;
            grad_coeff0_1_values[lane] = grad_coeff0_1;
            grad_coeff0_2_values[lane] = grad_coeff0_2;
            grad_coeff1_0_values[lane] = grad_coeff1_0;
            grad_coeff1_1_values[lane] = grad_coeff1_1;
            grad_coeff1_2_values[lane] = grad_coeff1_2;
            grad_coeff2_0_values[lane] = grad_coeff2_0;
            grad_coeff2_1_values[lane] = grad_coeff2_1;
            grad_coeff2_2_values[lane] = grad_coeff2_2;
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
                output[test * N_FIELDS + 1][lane] += q_weight[q] * det * (grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
                output[test * N_FIELDS + 2][lane] += q_weight[q] * det * (grad_coeff2_0_values[lane] * test_grad0 + grad_coeff2_1_values[lane] * test_grad1 + grad_coeff2_2_values[lane] * test_grad2);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t current[3 * N_SHAPE][VECTOR_SIZE],
        const scalar_t previous[3 * N_SHAPE][VECTOR_SIZE],
        const scalar_t direction[3 * N_SHAPE][VECTOR_SIZE],
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t output[3 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 3;
    for (int q = 0; q < N_QP; ++q) {
        scalar_t u0_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u0_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u0_old_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u0_direction_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u0_direction_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u0_direction_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_old_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u1_direction_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u1_direction_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u1_direction_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u2_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u2_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u2_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u2_old_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u2_old_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u2_old_grad_2_ref_values[VECTOR_SIZE];
        scalar_t u2_direction_grad_0_ref_values[VECTOR_SIZE];
        scalar_t u2_direction_grad_1_ref_values[VECTOR_SIZE];
        scalar_t u2_direction_grad_2_ref_values[VECTOR_SIZE];
        scalar_t grad_coeff0_0_values[VECTOR_SIZE];
        scalar_t grad_coeff0_1_values[VECTOR_SIZE];
        scalar_t grad_coeff0_2_values[VECTOR_SIZE];
        scalar_t grad_coeff1_0_values[VECTOR_SIZE];
        scalar_t grad_coeff1_1_values[VECTOR_SIZE];
        scalar_t grad_coeff1_2_values[VECTOR_SIZE];
        scalar_t grad_coeff2_0_values[VECTOR_SIZE];
        scalar_t grad_coeff2_1_values[VECTOR_SIZE];
        scalar_t grad_coeff2_2_values[VECTOR_SIZE];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 0][lane];
                u0_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u0_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_old_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 0][lane];
                u0_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u0_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u0_direction_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 0][lane];
                u0_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u0_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u0_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 1][lane];
                u1_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u1_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_old_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 1][lane];
                u1_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u1_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u1_direction_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 1][lane];
                u1_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u1_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u1_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = current[trial * N_FIELDS + 2][lane];
                u2_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u2_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u2_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_old_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_old_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_old_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = previous[trial * N_FIELDS + 2][lane];
                u2_old_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u2_old_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u2_old_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
            }
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_direction_grad_0_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_direction_grad_1_ref_values[lane] = scalar_t(0);
        }
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            u2_direction_grad_2_ref_values[lane] = scalar_t(0);
        }
        for (int trial = 0; trial < N_SHAPE; ++trial) {
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                const scalar_t coeff = direction[trial * N_FIELDS + 2][lane];
                u2_direction_grad_0_ref_values[lane] += coeff * grad_ref_x[q * N_SHAPE + trial];
                u2_direction_grad_1_ref_values[lane] += coeff * grad_ref_y[q * N_SHAPE + trial];
                u2_direction_grad_2_ref_values[lane] += coeff * grad_ref_z[q * N_SHAPE + trial];
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
            const scalar_t u0_grad_0_ref = u0_grad_0_ref_values[lane];
            const scalar_t u0_grad_1_ref = u0_grad_1_ref_values[lane];
            const scalar_t u0_grad_2_ref = u0_grad_2_ref_values[lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
            const scalar_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
            const scalar_t u0_old_grad_0_ref = u0_old_grad_0_ref_values[lane];
            const scalar_t u0_old_grad_1_ref = u0_old_grad_1_ref_values[lane];
            const scalar_t u0_old_grad_2_ref = u0_old_grad_2_ref_values[lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
            const scalar_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
            const scalar_t u0_direction_grad_0_ref = u0_direction_grad_0_ref_values[lane];
            const scalar_t u0_direction_grad_1_ref = u0_direction_grad_1_ref_values[lane];
            const scalar_t u0_direction_grad_2_ref = u0_direction_grad_2_ref_values[lane];
            const scalar_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
            const scalar_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
            const scalar_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
            const scalar_t u1_grad_0_ref = u1_grad_0_ref_values[lane];
            const scalar_t u1_grad_1_ref = u1_grad_1_ref_values[lane];
            const scalar_t u1_grad_2_ref = u1_grad_2_ref_values[lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
            const scalar_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
            const scalar_t u1_old_grad_0_ref = u1_old_grad_0_ref_values[lane];
            const scalar_t u1_old_grad_1_ref = u1_old_grad_1_ref_values[lane];
            const scalar_t u1_old_grad_2_ref = u1_old_grad_2_ref_values[lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
            const scalar_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
            const scalar_t u1_direction_grad_0_ref = u1_direction_grad_0_ref_values[lane];
            const scalar_t u1_direction_grad_1_ref = u1_direction_grad_1_ref_values[lane];
            const scalar_t u1_direction_grad_2_ref = u1_direction_grad_2_ref_values[lane];
            const scalar_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
            const scalar_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
            const scalar_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
            const scalar_t u2_grad_0_ref = u2_grad_0_ref_values[lane];
            const scalar_t u2_grad_1_ref = u2_grad_1_ref_values[lane];
            const scalar_t u2_grad_2_ref = u2_grad_2_ref_values[lane];
            const scalar_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
            const scalar_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
            const scalar_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
            const scalar_t u2_old_grad_0_ref = u2_old_grad_0_ref_values[lane];
            const scalar_t u2_old_grad_1_ref = u2_old_grad_1_ref_values[lane];
            const scalar_t u2_old_grad_2_ref = u2_old_grad_2_ref_values[lane];
            const scalar_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
            const scalar_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
            const scalar_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
            const scalar_t u2_direction_grad_0_ref = u2_direction_grad_0_ref_values[lane];
            const scalar_t u2_direction_grad_1_ref = u2_direction_grad_1_ref_values[lane];
            const scalar_t u2_direction_grad_2_ref = u2_direction_grad_2_ref_values[lane];
            const scalar_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
            const scalar_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
            const scalar_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
            const scalar_t residual_tmp0 = u0_grad_0*u1_grad_1;
            const scalar_t residual_tmp1 = u0_grad_1*u1_grad_2;
            const scalar_t residual_tmp2 = u0_grad_2*u2_grad_1;
            const scalar_t residual_tmp3 = u1_grad_2*u2_grad_1;
            const scalar_t residual_tmp4 = u0_grad_1*u1_grad_0;
            const scalar_t residual_tmp5 = u0_grad_2*u2_grad_0;
            const scalar_t residual_tmp6 = u2_grad_2 + scalar_t(1);
            const scalar_t residual_tmp7 = -residual_tmp3 + residual_tmp6 + u1_grad_1*u2_grad_2 + u1_grad_1;
            const scalar_t residual_tmp8 = -residual_tmp5 + u0_grad_0*u2_grad_2 + u0_grad_0;
            const scalar_t residual_tmp9 = residual_tmp0 - residual_tmp4;
            const scalar_t residual_tmp10 = residual_tmp0*u2_grad_2 + residual_tmp1*u2_grad_0 + residual_tmp2*u1_grad_0 - residual_tmp3*u0_grad_0 - residual_tmp4*u2_grad_2 - residual_tmp5*u1_grad_1 + residual_tmp7 + residual_tmp8 + residual_tmp9;
            const scalar_t residual_tmp11 = pow_m1(residual_tmp10);
            const scalar_t residual_tmp12 = -residual_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
            const scalar_t residual_tmp13 = newmark_velocity_alpha*residual_tmp7;
            const scalar_t residual_tmp14 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
            const scalar_t residual_tmp15 = residual_tmp14*u2_grad_1;
            const scalar_t residual_tmp16 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp17 = residual_tmp16*residual_tmp6;
            const scalar_t residual_tmp18 = residual_tmp15 - residual_tmp17;
            const scalar_t residual_tmp19 = -residual_tmp13 - residual_tmp18;
            const scalar_t residual_tmp20 = -residual_tmp1 + u0_grad_2*u1_grad_1 + u0_grad_2;
            const scalar_t residual_tmp21 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
            const scalar_t residual_tmp22 = residual_tmp21*residual_tmp6;
            const scalar_t residual_tmp23 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
            const scalar_t residual_tmp24 = residual_tmp23*u2_grad_1;
            const scalar_t residual_tmp25 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
            const scalar_t residual_tmp26 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
            const scalar_t residual_tmp27 = residual_tmp25*u0_grad_1 - residual_tmp26*u0_grad_2;
            const scalar_t residual_tmp28 = residual_tmp22 - residual_tmp24 + residual_tmp27;
            const scalar_t residual_tmp29 = newmark_velocity_alpha*residual_tmp12;
            const scalar_t residual_tmp30 = residual_tmp23*u0_grad_1;
            const scalar_t residual_tmp31 = residual_tmp21*u0_grad_2;
            const scalar_t residual_tmp32 = residual_tmp29 + residual_tmp30 - residual_tmp31;
            const scalar_t residual_tmp33 = residual_tmp26*residual_tmp6;
            const scalar_t residual_tmp34 = residual_tmp25*u2_grad_1;
            const scalar_t residual_tmp35 = residual_tmp33 - residual_tmp34;
            const scalar_t residual_tmp36 = scalar_t(3)*eta_b;
            const scalar_t residual_tmp37 = residual_tmp36*(residual_tmp32 + residual_tmp35);
            const scalar_t residual_tmp38 = scalar_t(2)*eta_s;
            const scalar_t residual_tmp39 = residual_tmp37 + residual_tmp38*(scalar_t(2)*residual_tmp26*residual_tmp6 - residual_tmp32 - scalar_t(2)*residual_tmp34);
            const scalar_t residual_tmp40 = ((scalar_t(1) / scalar_t(3)))*residual_tmp7;
            const scalar_t residual_tmp41 = pow_m2(residual_tmp10);
            const scalar_t residual_tmp42 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp43 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
            const scalar_t residual_tmp44 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
            const scalar_t residual_tmp45 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
            const scalar_t residual_tmp46 = u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp47 = residual_tmp46 + residual_tmp9 + u0_grad_0;
            const scalar_t residual_tmp48 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
            const scalar_t residual_tmp49 = residual_tmp20*residual_tmp42 + residual_tmp21*residual_tmp44 + residual_tmp23*residual_tmp45 - residual_tmp25*residual_tmp47 + residual_tmp26*residual_tmp43 - residual_tmp48*residual_tmp7;
            const scalar_t residual_tmp50 = u0_grad_0*u2_grad_1 - u0_grad_1*u2_grad_0 + u2_grad_1;
            const scalar_t residual_tmp51 = residual_tmp6 + residual_tmp8;
            const scalar_t residual_tmp52 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
            const scalar_t residual_tmp53 = residual_tmp12*residual_tmp42 + residual_tmp14*residual_tmp45 + residual_tmp16*residual_tmp44 + residual_tmp25*residual_tmp50 - residual_tmp26*residual_tmp51 - residual_tmp52*residual_tmp7;
            const scalar_t residual_tmp54 = residual_tmp12*residual_tmp52;
            const scalar_t residual_tmp55 = residual_tmp14*residual_tmp50;
            const scalar_t residual_tmp56 = residual_tmp20*residual_tmp48;
            const scalar_t residual_tmp57 = residual_tmp21*residual_tmp43;
            const scalar_t residual_tmp58 = residual_tmp16*residual_tmp51;
            const scalar_t residual_tmp59 = -residual_tmp58;
            const scalar_t residual_tmp60 = residual_tmp23*residual_tmp47;
            const scalar_t residual_tmp61 = -residual_tmp60;
            const scalar_t residual_tmp62 = residual_tmp54 + residual_tmp55 + residual_tmp56 + residual_tmp57 + residual_tmp59 + residual_tmp61;
            const scalar_t residual_tmp63 = residual_tmp42*residual_tmp7;
            const scalar_t residual_tmp64 = residual_tmp25*residual_tmp45 + residual_tmp26*residual_tmp44 - residual_tmp63;
            const scalar_t residual_tmp65 = residual_tmp36*(residual_tmp62 + residual_tmp64);
            const scalar_t residual_tmp66 = residual_tmp38*(scalar_t(2)*residual_tmp25*residual_tmp45 + scalar_t(2)*residual_tmp26*residual_tmp44 - residual_tmp62 - scalar_t(2)*residual_tmp63) + residual_tmp65;
            const scalar_t residual_tmp67 = -residual_tmp66;
            const scalar_t residual_tmp68 = residual_tmp41*(eta_s*(residual_tmp12*residual_tmp53 + residual_tmp20*residual_tmp49) + residual_tmp40*residual_tmp67);
            const scalar_t residual_tmp69 = residual_tmp21*u1_grad_2;
            const scalar_t residual_tmp70 = residual_tmp23*residual_tmp46;
            const scalar_t residual_tmp71 = residual_tmp69 - residual_tmp70;
            const scalar_t residual_tmp72 = -residual_tmp13 - residual_tmp71;
            const scalar_t residual_tmp73 = -residual_tmp14*residual_tmp46 + residual_tmp16*u1_grad_2;
            const scalar_t residual_tmp74 = -residual_tmp27 - residual_tmp73;
            const scalar_t residual_tmp75 = newmark_velocity_alpha*residual_tmp20;
            const scalar_t residual_tmp76 = residual_tmp16*u0_grad_2;
            const scalar_t residual_tmp77 = residual_tmp14*u0_grad_1;
            const scalar_t residual_tmp78 = residual_tmp75 + residual_tmp76 - residual_tmp77;
            const scalar_t residual_tmp79 = residual_tmp25*residual_tmp46;
            const scalar_t residual_tmp80 = residual_tmp26*u1_grad_2;
            const scalar_t residual_tmp81 = residual_tmp79 - residual_tmp80;
            const scalar_t residual_tmp82 = residual_tmp36*(residual_tmp78 + residual_tmp81);
            const scalar_t residual_tmp83 = residual_tmp38*(scalar_t(2)*residual_tmp25*residual_tmp46 - residual_tmp78 - scalar_t(2)*residual_tmp80) + residual_tmp82;
            const scalar_t residual_tmp84 = -residual_tmp79 + residual_tmp80;
            const scalar_t residual_tmp85 = residual_tmp75 + residual_tmp84;
            const scalar_t residual_tmp86 = -residual_tmp33 + residual_tmp34;
            const scalar_t residual_tmp87 = residual_tmp29 + residual_tmp86;
            const scalar_t residual_tmp88 = residual_tmp13 - residual_tmp69 + residual_tmp70;
            const scalar_t residual_tmp89 = -residual_tmp15 + residual_tmp17;
            const scalar_t residual_tmp90 = residual_tmp36*(-residual_tmp88 - residual_tmp89);
            const scalar_t residual_tmp91 = residual_tmp38*(-scalar_t(2)*residual_tmp13 - residual_tmp18 - residual_tmp71) + residual_tmp90;
            const scalar_t residual_tmp92 = -residual_tmp7;
            const scalar_t residual_tmp93 = newmark_velocity_alpha*residual_tmp44;
            const scalar_t residual_tmp94 = residual_tmp23*u1_grad_0;
            const scalar_t residual_tmp95 = residual_tmp48*u1_grad_2;
            const scalar_t residual_tmp96 = residual_tmp93 + residual_tmp94 - residual_tmp95;
            const scalar_t residual_tmp97 = residual_tmp52*residual_tmp6;
            const scalar_t residual_tmp98 = residual_tmp14*u2_grad_0;
            const scalar_t residual_tmp99 = residual_tmp97 - residual_tmp98;
            const scalar_t residual_tmp100 = residual_tmp36*(residual_tmp96 + residual_tmp99);
            const scalar_t residual_tmp101 = -residual_tmp97 + residual_tmp98;
            const scalar_t residual_tmp102 = -residual_tmp94 + residual_tmp95;
            const scalar_t residual_tmp103 = residual_tmp100 + residual_tmp38*(residual_tmp101 + residual_tmp102 + scalar_t(2)*residual_tmp93);
            const scalar_t residual_tmp104 = newmark_velocity_alpha*residual_tmp43;
            const scalar_t residual_tmp105 = residual_tmp25*u1_grad_0;
            const scalar_t residual_tmp106 = residual_tmp42*u1_grad_2;
            const scalar_t residual_tmp107 = residual_tmp105 - residual_tmp106;
            const scalar_t residual_tmp108 = residual_tmp104 + residual_tmp107;
            const scalar_t residual_tmp109 = newmark_velocity_alpha*residual_tmp51;
            const scalar_t residual_tmp110 = residual_tmp25*u2_grad_0;
            const scalar_t residual_tmp111 = residual_tmp42*residual_tmp6;
            const scalar_t residual_tmp112 = residual_tmp110 - residual_tmp111;
            const scalar_t residual_tmp113 = -residual_tmp109 - residual_tmp112;
            const scalar_t residual_tmp114 = residual_tmp49*u1_grad_2;
            const scalar_t residual_tmp115 = -residual_tmp114;
            const scalar_t residual_tmp116 = residual_tmp53*residual_tmp6;
            const scalar_t residual_tmp117 = newmark_velocity_alpha*residual_tmp45;
            const scalar_t residual_tmp118 = residual_tmp46*residual_tmp48;
            const scalar_t residual_tmp119 = residual_tmp21*u1_grad_0;
            const scalar_t residual_tmp120 = residual_tmp117 + residual_tmp118 - residual_tmp119;
            const scalar_t residual_tmp121 = residual_tmp16*u2_grad_0;
            const scalar_t residual_tmp122 = residual_tmp52*u2_grad_1;
            const scalar_t residual_tmp123 = residual_tmp121 - residual_tmp122;
            const scalar_t residual_tmp124 = residual_tmp36*(residual_tmp120 + residual_tmp123);
            const scalar_t residual_tmp125 = -residual_tmp121 + residual_tmp122;
            const scalar_t residual_tmp126 = -residual_tmp118 + residual_tmp119;
            const scalar_t residual_tmp127 = residual_tmp124 + residual_tmp38*(scalar_t(2)*residual_tmp117 + residual_tmp125 + residual_tmp126);
            const scalar_t residual_tmp128 = newmark_velocity_alpha*residual_tmp50;
            const scalar_t residual_tmp129 = residual_tmp26*u2_grad_0;
            const scalar_t residual_tmp130 = residual_tmp42*u2_grad_1;
            const scalar_t residual_tmp131 = residual_tmp129 - residual_tmp130;
            const scalar_t residual_tmp132 = residual_tmp128 + residual_tmp131;
            const scalar_t residual_tmp133 = newmark_velocity_alpha*residual_tmp47;
            const scalar_t residual_tmp134 = residual_tmp26*u1_grad_0;
            const scalar_t residual_tmp135 = residual_tmp42*residual_tmp46;
            const scalar_t residual_tmp136 = residual_tmp134 - residual_tmp135;
            const scalar_t residual_tmp137 = -residual_tmp133 - residual_tmp136;
            const scalar_t residual_tmp138 = residual_tmp53*u2_grad_1;
            const scalar_t residual_tmp139 = -residual_tmp138;
            const scalar_t residual_tmp140 = residual_tmp46*residual_tmp49;
            const scalar_t residual_tmp141 = u0_grad_0 + scalar_t(1);
            const scalar_t residual_tmp142 = residual_tmp141*residual_tmp21;
            const scalar_t residual_tmp143 = residual_tmp48*u0_grad_1;
            const scalar_t residual_tmp144 = residual_tmp128 + residual_tmp142 - residual_tmp143;
            const scalar_t residual_tmp145 = -residual_tmp129 + residual_tmp130;
            const scalar_t residual_tmp146 = residual_tmp36*(residual_tmp144 + residual_tmp145);
            const scalar_t residual_tmp147 = residual_tmp146 + residual_tmp38*(-scalar_t(2)*residual_tmp129 - residual_tmp144 + scalar_t(2)*residual_tmp42*u2_grad_1);
            const scalar_t residual_tmp148 = residual_tmp117 + residual_tmp125;
            const scalar_t residual_tmp149 = residual_tmp21*u2_grad_0;
            const scalar_t residual_tmp150 = residual_tmp48*u2_grad_1;
            const scalar_t residual_tmp151 = -residual_tmp141*residual_tmp26 + residual_tmp42*u0_grad_1;
            const scalar_t residual_tmp152 = -residual_tmp149 + residual_tmp150 - residual_tmp151;
            const scalar_t residual_tmp153 = residual_tmp49*u0_grad_1;
            const scalar_t residual_tmp154 = ((scalar_t(1) / scalar_t(3)))*residual_tmp67;
            const scalar_t residual_tmp155 = residual_tmp14*residual_tmp141;
            const scalar_t residual_tmp156 = residual_tmp52*u0_grad_2;
            const scalar_t residual_tmp157 = residual_tmp104 + residual_tmp155 - residual_tmp156;
            const scalar_t residual_tmp158 = -residual_tmp105 + residual_tmp106;
            const scalar_t residual_tmp159 = residual_tmp36*(residual_tmp157 + residual_tmp158);
            const scalar_t residual_tmp160 = residual_tmp159 + residual_tmp38*(-scalar_t(2)*residual_tmp105 - residual_tmp157 + scalar_t(2)*residual_tmp42*u1_grad_2);
            const scalar_t residual_tmp161 = residual_tmp102 + residual_tmp93;
            const scalar_t residual_tmp162 = -residual_tmp141*residual_tmp25 + residual_tmp42*u0_grad_2;
            const scalar_t residual_tmp163 = residual_tmp14*u1_grad_0 - residual_tmp52*u1_grad_2;
            const scalar_t residual_tmp164 = -residual_tmp162 - residual_tmp163;
            const scalar_t residual_tmp165 = residual_tmp53*u0_grad_2;
            const scalar_t residual_tmp166 = -residual_tmp51;
            const scalar_t residual_tmp167 = residual_tmp141*residual_tmp23;
            const scalar_t residual_tmp168 = residual_tmp48*u0_grad_2;
            const scalar_t residual_tmp169 = residual_tmp109 + residual_tmp167 - residual_tmp168;
            const scalar_t residual_tmp170 = -residual_tmp110 + residual_tmp111;
            const scalar_t residual_tmp171 = residual_tmp36*(-residual_tmp169 - residual_tmp170);
            const scalar_t residual_tmp172 = residual_tmp171 + residual_tmp38*(scalar_t(2)*residual_tmp110 - scalar_t(2)*residual_tmp111 + residual_tmp169);
            const scalar_t residual_tmp173 = residual_tmp101 + residual_tmp93;
            const scalar_t residual_tmp174 = residual_tmp23*u2_grad_0;
            const scalar_t residual_tmp175 = residual_tmp48*residual_tmp6;
            const scalar_t residual_tmp176 = residual_tmp162 + residual_tmp174 - residual_tmp175;
            const scalar_t residual_tmp177 = residual_tmp49*u0_grad_2;
            const scalar_t residual_tmp178 = -residual_tmp47;
            const scalar_t residual_tmp179 = residual_tmp141*residual_tmp16;
            const scalar_t residual_tmp180 = residual_tmp52*u0_grad_1;
            const scalar_t residual_tmp181 = residual_tmp133 + residual_tmp179 - residual_tmp180;
            const scalar_t residual_tmp182 = -residual_tmp134 + residual_tmp135;
            const scalar_t residual_tmp183 = residual_tmp36*(-residual_tmp181 - residual_tmp182);
            const scalar_t residual_tmp184 = residual_tmp183 + residual_tmp38*(scalar_t(2)*residual_tmp134 - scalar_t(2)*residual_tmp135 + residual_tmp181);
            const scalar_t residual_tmp185 = residual_tmp117 + residual_tmp126;
            const scalar_t residual_tmp186 = residual_tmp16*u1_grad_0 - residual_tmp46*residual_tmp52;
            const scalar_t residual_tmp187 = residual_tmp151 + residual_tmp186;
            const scalar_t residual_tmp188 = residual_tmp53*u0_grad_1;
            const scalar_t residual_tmp189 = residual_tmp41*(-eta_s*(-residual_tmp43*residual_tmp49 + residual_tmp51*residual_tmp53) + ((scalar_t(1) / scalar_t(3)))*residual_tmp44*residual_tmp66);
            const scalar_t residual_tmp190 = residual_tmp49*u1_grad_0;
            const scalar_t residual_tmp191 = residual_tmp53*u2_grad_0;
            const scalar_t residual_tmp192 = -residual_tmp191;
            const scalar_t residual_tmp193 = residual_tmp190 + residual_tmp192;
            const scalar_t residual_tmp194 = -residual_tmp53*residual_tmp6;
            const scalar_t residual_tmp195 = residual_tmp141*residual_tmp49;
            const scalar_t residual_tmp196 = ((scalar_t(1) / scalar_t(3)))*residual_tmp66;
            const scalar_t residual_tmp197 = residual_tmp196*u2_grad_0;
            const scalar_t residual_tmp198 = ((scalar_t(1) / scalar_t(3)))*residual_tmp44;
            const scalar_t residual_tmp199 = residual_tmp141*residual_tmp53;
            const scalar_t residual_tmp200 = residual_tmp196*u1_grad_0;
            const scalar_t residual_tmp201 = residual_tmp41*(-eta_s*(residual_tmp47*residual_tmp49 - residual_tmp50*residual_tmp53) + ((scalar_t(1) / scalar_t(3)))*residual_tmp45*residual_tmp66);
            const scalar_t residual_tmp202 = -residual_tmp46*residual_tmp49;
            const scalar_t residual_tmp203 = ((scalar_t(1) / scalar_t(3)))*residual_tmp45;
            const scalar_t residual_tmp204 = -residual_tmp76 + residual_tmp77;
            const scalar_t residual_tmp205 = residual_tmp204 + residual_tmp75;
            const scalar_t residual_tmp206 = -residual_tmp30 + residual_tmp31;
            const scalar_t residual_tmp207 = residual_tmp37 + residual_tmp38*(residual_tmp206 + scalar_t(2)*residual_tmp29 + residual_tmp86);
            const scalar_t residual_tmp208 = residual_tmp12*residual_tmp48 - residual_tmp14*residual_tmp47 + residual_tmp16*residual_tmp43 + residual_tmp20*residual_tmp52 - residual_tmp21*residual_tmp51 + residual_tmp23*residual_tmp50;
            const scalar_t residual_tmp209 = residual_tmp38*(scalar_t(2)*residual_tmp12*residual_tmp52 + scalar_t(2)*residual_tmp14*residual_tmp50 - residual_tmp56 - residual_tmp57 - scalar_t(2)*residual_tmp58 - residual_tmp61 - residual_tmp64) + residual_tmp65;
            const scalar_t residual_tmp210 = residual_tmp41*(-eta_s*(-residual_tmp20*residual_tmp208 + residual_tmp53*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp209);
            const scalar_t residual_tmp211 = residual_tmp206 + residual_tmp29;
            const scalar_t residual_tmp212 = residual_tmp38*(scalar_t(2)*residual_tmp16*u0_grad_2 - residual_tmp75 - scalar_t(2)*residual_tmp77 - residual_tmp81) + residual_tmp82;
            const scalar_t residual_tmp213 = -residual_tmp22 + residual_tmp24 + residual_tmp73;
            const scalar_t residual_tmp214 = residual_tmp38*(scalar_t(2)*residual_tmp15 - scalar_t(2)*residual_tmp17 + residual_tmp88) + residual_tmp90;
            const scalar_t residual_tmp215 = -residual_tmp142 + residual_tmp143;
            const scalar_t residual_tmp216 = residual_tmp146 + residual_tmp38*(scalar_t(2)*residual_tmp128 + residual_tmp131 + residual_tmp215);
            const scalar_t residual_tmp217 = -residual_tmp179 + residual_tmp180;
            const scalar_t residual_tmp218 = -residual_tmp133 - residual_tmp217;
            const scalar_t residual_tmp219 = residual_tmp208*u0_grad_1;
            const scalar_t residual_tmp220 = residual_tmp139 + residual_tmp219;
            const scalar_t residual_tmp221 = -residual_tmp167 + residual_tmp168;
            const scalar_t residual_tmp222 = residual_tmp171 + residual_tmp38*(-scalar_t(2)*residual_tmp109 - residual_tmp112 - residual_tmp221);
            const scalar_t residual_tmp223 = -residual_tmp155 + residual_tmp156;
            const scalar_t residual_tmp224 = residual_tmp104 + residual_tmp223;
            const scalar_t residual_tmp225 = residual_tmp208*u0_grad_2;
            const scalar_t residual_tmp226 = residual_tmp124 + residual_tmp38*(-residual_tmp120 - scalar_t(2)*residual_tmp122 + scalar_t(2)*residual_tmp16*u2_grad_0);
            const scalar_t residual_tmp227 = residual_tmp149 - residual_tmp150 - residual_tmp186;
            const scalar_t residual_tmp228 = residual_tmp208*residual_tmp46;
            const scalar_t residual_tmp229 = ((scalar_t(1) / scalar_t(3)))*residual_tmp209;
            const scalar_t residual_tmp230 = residual_tmp229*u2_grad_1;
            const scalar_t residual_tmp231 = residual_tmp159 + residual_tmp38*(-residual_tmp104 + scalar_t(2)*residual_tmp14*residual_tmp141 - scalar_t(2)*residual_tmp156 - residual_tmp158);
            const scalar_t residual_tmp232 = -residual_tmp109 - residual_tmp221;
            const scalar_t residual_tmp233 = residual_tmp53*u1_grad_2;
            const scalar_t residual_tmp234 = residual_tmp100 + residual_tmp38*(scalar_t(2)*residual_tmp52*residual_tmp6 - residual_tmp96 - scalar_t(2)*residual_tmp98);
            const scalar_t residual_tmp235 = ((scalar_t(1) / scalar_t(3)))*residual_tmp12;
            const scalar_t residual_tmp236 = residual_tmp163 - residual_tmp174 + residual_tmp175;
            const scalar_t residual_tmp237 = residual_tmp208*u1_grad_2;
            const scalar_t residual_tmp238 = residual_tmp183 + residual_tmp38*(residual_tmp133 - scalar_t(2)*residual_tmp179 + scalar_t(2)*residual_tmp180 + residual_tmp182);
            const scalar_t residual_tmp239 = residual_tmp128 + residual_tmp215;
            const scalar_t residual_tmp240 = residual_tmp46*residual_tmp53;
            const scalar_t residual_tmp241 = residual_tmp229*u0_grad_1;
            const scalar_t residual_tmp242 = ((scalar_t(1) / scalar_t(3)))*residual_tmp51;
            const scalar_t residual_tmp243 = -(scalar_t(1) / scalar_t(3))*residual_tmp209;
            const scalar_t residual_tmp244 = residual_tmp41*(eta_s*(residual_tmp208*residual_tmp43 + residual_tmp44*residual_tmp53) + residual_tmp243*residual_tmp51);
            const scalar_t residual_tmp245 = residual_tmp141*residual_tmp208;
            const scalar_t residual_tmp246 = residual_tmp208*u1_grad_0;
            const scalar_t residual_tmp247 = residual_tmp53*u1_grad_0;
            const scalar_t residual_tmp248 = residual_tmp41*(-eta_s*(residual_tmp208*residual_tmp47 - residual_tmp45*residual_tmp53) + ((scalar_t(1) / scalar_t(3)))*residual_tmp209*residual_tmp50);
            const scalar_t residual_tmp249 = -residual_tmp141*residual_tmp208;
            const scalar_t residual_tmp250 = ((scalar_t(1) / scalar_t(3)))*residual_tmp50;
            const scalar_t residual_tmp251 = residual_tmp38*(residual_tmp204 + scalar_t(2)*residual_tmp75 + residual_tmp84) + residual_tmp82;
            const scalar_t residual_tmp252 = residual_tmp38*(scalar_t(2)*residual_tmp20*residual_tmp48 + scalar_t(2)*residual_tmp21*residual_tmp43 - residual_tmp54 - residual_tmp55 - residual_tmp59 - scalar_t(2)*residual_tmp60 - residual_tmp64) + residual_tmp65;
            const scalar_t residual_tmp253 = residual_tmp41*(-eta_s*(-residual_tmp12*residual_tmp208 + residual_tmp49*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp252);
            const scalar_t residual_tmp254 = residual_tmp37 + residual_tmp38*(scalar_t(2)*residual_tmp23*u0_grad_1 - residual_tmp29 - scalar_t(2)*residual_tmp31 - residual_tmp35);
            const scalar_t residual_tmp255 = residual_tmp38*(residual_tmp13 + scalar_t(2)*residual_tmp69 - scalar_t(2)*residual_tmp70 + residual_tmp89) + residual_tmp90;
            const scalar_t residual_tmp256 = residual_tmp159 + residual_tmp38*(scalar_t(2)*residual_tmp104 + residual_tmp107 + residual_tmp223);
            const scalar_t residual_tmp257 = residual_tmp115 + residual_tmp225;
            const scalar_t residual_tmp258 = residual_tmp183 + residual_tmp38*(-scalar_t(2)*residual_tmp133 - residual_tmp136 - residual_tmp217);
            const scalar_t residual_tmp259 = residual_tmp100 + residual_tmp38*(scalar_t(2)*residual_tmp23*u1_grad_0 - residual_tmp93 - scalar_t(2)*residual_tmp95 - residual_tmp99);
            const scalar_t residual_tmp260 = residual_tmp208*residual_tmp6;
            const scalar_t residual_tmp261 = ((scalar_t(1) / scalar_t(3)))*residual_tmp252;
            const scalar_t residual_tmp262 = residual_tmp261*u1_grad_2;
            const scalar_t residual_tmp263 = residual_tmp146 + residual_tmp38*(-residual_tmp128 + scalar_t(2)*residual_tmp141*residual_tmp21 - scalar_t(2)*residual_tmp143 - residual_tmp145);
            const scalar_t residual_tmp264 = residual_tmp49*u2_grad_1;
            const scalar_t residual_tmp265 = residual_tmp124 + residual_tmp38*(-residual_tmp117 - scalar_t(2)*residual_tmp119 - residual_tmp123 + scalar_t(2)*residual_tmp46*residual_tmp48);
            const scalar_t residual_tmp266 = ((scalar_t(1) / scalar_t(3)))*residual_tmp20;
            const scalar_t residual_tmp267 = residual_tmp208*u2_grad_1;
            const scalar_t residual_tmp268 = residual_tmp171 + residual_tmp38*(residual_tmp109 - scalar_t(2)*residual_tmp167 + scalar_t(2)*residual_tmp168 + residual_tmp170);
            const scalar_t residual_tmp269 = residual_tmp49*residual_tmp6;
            const scalar_t residual_tmp270 = residual_tmp261*u0_grad_2;
            const scalar_t residual_tmp271 = residual_tmp41*(-eta_s*(residual_tmp208*residual_tmp51 - residual_tmp44*residual_tmp49) + ((scalar_t(1) / scalar_t(3)))*residual_tmp252*residual_tmp43);
            const scalar_t residual_tmp272 = residual_tmp208*u2_grad_0;
            const scalar_t residual_tmp273 = ((scalar_t(1) / scalar_t(3)))*residual_tmp43;
            const scalar_t residual_tmp274 = residual_tmp49*u2_grad_0;
            const scalar_t residual_tmp275 = ((scalar_t(1) / scalar_t(3)))*residual_tmp47;
            const scalar_t residual_tmp276 = -(scalar_t(1) / scalar_t(3))*residual_tmp252;
            const scalar_t residual_tmp277 = residual_tmp41*(eta_s*(residual_tmp208*residual_tmp50 + residual_tmp45*residual_tmp49) + residual_tmp276*residual_tmp47);
            const scalar_t grad_coeff0_0 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp87 + residual_tmp20*residual_tmp85) - residual_tmp40*residual_tmp91) + residual_tmp68*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp108*residual_tmp20 + residual_tmp113*residual_tmp12 + residual_tmp115 + residual_tmp116) - residual_tmp103*residual_tmp40) + residual_tmp44*residual_tmp68) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp132 + residual_tmp137*residual_tmp20 + residual_tmp139 + residual_tmp140) - residual_tmp127*residual_tmp40) + residual_tmp45*residual_tmp68) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp19 + residual_tmp20*residual_tmp28) - residual_tmp39*residual_tmp40) + residual_tmp12*residual_tmp68) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp173 + residual_tmp176*residual_tmp20 + residual_tmp177) + residual_tmp154*residual_tmp6 - residual_tmp172*residual_tmp40) + residual_tmp166*residual_tmp68) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp148 + residual_tmp152*residual_tmp20 - residual_tmp153) - residual_tmp147*residual_tmp40 - residual_tmp154*u2_grad_1) + residual_tmp50*residual_tmp68) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp74 + residual_tmp20*residual_tmp72) - residual_tmp40*residual_tmp83) + residual_tmp20*residual_tmp68) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp164 + residual_tmp161*residual_tmp20 - residual_tmp165) - residual_tmp154*u1_grad_2 - residual_tmp160*residual_tmp40) + residual_tmp43*residual_tmp68) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp187 + residual_tmp185*residual_tmp20 + residual_tmp188) + residual_tmp154*residual_tmp46 - residual_tmp184*residual_tmp40) + residual_tmp178*residual_tmp68);
            const scalar_t grad_coeff0_1 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp114 - residual_tmp194 - residual_tmp43*residual_tmp85 + residual_tmp51*residual_tmp87) + ((scalar_t(1) / scalar_t(3)))*residual_tmp44*residual_tmp91) + residual_tmp189*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp108*residual_tmp43 + residual_tmp113*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp103*residual_tmp44) + residual_tmp189*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp132*residual_tmp51 - residual_tmp137*residual_tmp43 + residual_tmp193) + ((scalar_t(1) / scalar_t(3)))*residual_tmp127*residual_tmp44) + residual_tmp189*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp177 + residual_tmp19*residual_tmp51 - residual_tmp28*residual_tmp43) + residual_tmp196*residual_tmp6 + residual_tmp198*residual_tmp39) + residual_tmp12*residual_tmp189) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp173*residual_tmp51 - residual_tmp176*residual_tmp43) + ((scalar_t(1) / scalar_t(3)))*residual_tmp172*residual_tmp44) + residual_tmp166*residual_tmp189) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp148*residual_tmp51 - residual_tmp152*residual_tmp43 - residual_tmp195) + ((scalar_t(1) / scalar_t(3)))*residual_tmp147*residual_tmp44 - residual_tmp197) + residual_tmp189*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp165 - residual_tmp43*residual_tmp72 + residual_tmp51*residual_tmp74) - residual_tmp196*u1_grad_2 + ((scalar_t(1) / scalar_t(3)))*residual_tmp44*residual_tmp83) + residual_tmp189*residual_tmp20) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp161*residual_tmp43 + residual_tmp164*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp160*residual_tmp44) + residual_tmp189*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp185*residual_tmp43 + residual_tmp187*residual_tmp51 + residual_tmp199) + residual_tmp184*residual_tmp198 + residual_tmp200) + residual_tmp178*residual_tmp189);
            const scalar_t grad_coeff0_2 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp138 - residual_tmp202 + residual_tmp47*residual_tmp85 - residual_tmp50*residual_tmp87) + ((scalar_t(1) / scalar_t(3)))*residual_tmp45*residual_tmp91) + residual_tmp201*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp108*residual_tmp47 - residual_tmp113*residual_tmp50 - residual_tmp193) + ((scalar_t(1) / scalar_t(3)))*residual_tmp103*residual_tmp45) + residual_tmp201*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp132*residual_tmp50 + residual_tmp137*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp127*residual_tmp45) + residual_tmp201*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp153 - residual_tmp19*residual_tmp50 + residual_tmp28*residual_tmp47) - residual_tmp196*u2_grad_1 + ((scalar_t(1) / scalar_t(3)))*residual_tmp39*residual_tmp45) + residual_tmp12*residual_tmp201) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp173*residual_tmp50 + residual_tmp176*residual_tmp47 + residual_tmp195) + residual_tmp172*residual_tmp203 + residual_tmp197) + residual_tmp166*residual_tmp201) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp148*residual_tmp50 + residual_tmp152*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp147*residual_tmp45) + residual_tmp201*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp188 + residual_tmp47*residual_tmp72 - residual_tmp50*residual_tmp74) + residual_tmp196*residual_tmp46 + residual_tmp203*residual_tmp83) + residual_tmp20*residual_tmp201) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp161*residual_tmp47 - residual_tmp164*residual_tmp50 - residual_tmp199) + ((scalar_t(1) / scalar_t(3)))*residual_tmp160*residual_tmp45 - residual_tmp200) + residual_tmp201*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp185*residual_tmp47 - residual_tmp187*residual_tmp50) + ((scalar_t(1) / scalar_t(3)))*residual_tmp184*residual_tmp45) + residual_tmp178*residual_tmp201);
            const scalar_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp20*residual_tmp213 + residual_tmp7*residual_tmp87) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp214) + residual_tmp210*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp113*residual_tmp7 - residual_tmp20*residual_tmp236 + residual_tmp237) + residual_tmp229*residual_tmp6 + residual_tmp234*residual_tmp235) + residual_tmp210*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp132*residual_tmp7 - residual_tmp20*residual_tmp227 - residual_tmp228) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp226 - residual_tmp230) + residual_tmp210*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp19*residual_tmp7 - residual_tmp20*residual_tmp205) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp207) + residual_tmp12*residual_tmp210) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp173*residual_tmp7 - residual_tmp194 - residual_tmp20*residual_tmp224 - residual_tmp225) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp222) + residual_tmp166*residual_tmp210) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp148*residual_tmp7 - residual_tmp20*residual_tmp218 + residual_tmp220) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp216) + residual_tmp210*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp20*residual_tmp211 + residual_tmp7*residual_tmp74) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp212) + residual_tmp20*residual_tmp210) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp164*residual_tmp7 - residual_tmp20*residual_tmp232 - residual_tmp233) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp231 - residual_tmp229*u0_grad_2) + residual_tmp210*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp187*residual_tmp7 - residual_tmp20*residual_tmp239 + residual_tmp240) + residual_tmp235*residual_tmp238 + residual_tmp241) + residual_tmp178*residual_tmp210);
            const scalar_t grad_coeff1_1 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp213*residual_tmp43 + residual_tmp237 + residual_tmp44*residual_tmp87) - residual_tmp214*residual_tmp242 + residual_tmp243*residual_tmp6) + residual_tmp244*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp113*residual_tmp44 + residual_tmp236*residual_tmp43) - residual_tmp234*residual_tmp242) + residual_tmp244*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp132*residual_tmp44 + residual_tmp227*residual_tmp43 - residual_tmp246) - residual_tmp226*residual_tmp242 - residual_tmp243*u2_grad_0) + residual_tmp244*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp116 + residual_tmp19*residual_tmp44 + residual_tmp205*residual_tmp43 - residual_tmp225) - residual_tmp207*residual_tmp242) + residual_tmp12*residual_tmp244) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp173*residual_tmp44 + residual_tmp224*residual_tmp43) - residual_tmp222*residual_tmp242) + residual_tmp166*residual_tmp244) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp148*residual_tmp44 + residual_tmp192 + residual_tmp218*residual_tmp43 + residual_tmp245) - residual_tmp216*residual_tmp242) + residual_tmp244*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp211*residual_tmp43 - residual_tmp233 + residual_tmp44*residual_tmp74) - residual_tmp212*residual_tmp242 - residual_tmp243*u0_grad_2) + residual_tmp20*residual_tmp244) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp164*residual_tmp44 + residual_tmp232*residual_tmp43) - residual_tmp231*residual_tmp242) + residual_tmp244*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp187*residual_tmp44 + residual_tmp239*residual_tmp43 + residual_tmp247) + residual_tmp141*residual_tmp243 - residual_tmp238*residual_tmp242) + residual_tmp178*residual_tmp244);
            const scalar_t grad_coeff1_2 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp213*residual_tmp47 + residual_tmp228 - residual_tmp45*residual_tmp87) + residual_tmp214*residual_tmp250 + residual_tmp230) + residual_tmp248*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp113*residual_tmp45 + residual_tmp236*residual_tmp47 - residual_tmp246) - residual_tmp229*u2_grad_0 + ((scalar_t(1) / scalar_t(3)))*residual_tmp234*residual_tmp50) + residual_tmp248*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp132*residual_tmp45 + residual_tmp227*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp226*residual_tmp50) + residual_tmp248*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp19*residual_tmp45 + residual_tmp205*residual_tmp47 - residual_tmp220) + ((scalar_t(1) / scalar_t(3)))*residual_tmp207*residual_tmp50) + residual_tmp12*residual_tmp248) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp173*residual_tmp45 - residual_tmp191 + residual_tmp224*residual_tmp47 - residual_tmp249) + ((scalar_t(1) / scalar_t(3)))*residual_tmp222*residual_tmp50) + residual_tmp166*residual_tmp248) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp148*residual_tmp45 + residual_tmp218*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp216*residual_tmp50) + residual_tmp248*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp211*residual_tmp47 - residual_tmp240 - residual_tmp45*residual_tmp74) + ((scalar_t(1) / scalar_t(3)))*residual_tmp212*residual_tmp50 - residual_tmp241) + residual_tmp20*residual_tmp248) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp164*residual_tmp45 + residual_tmp232*residual_tmp47 + residual_tmp247) + residual_tmp141*residual_tmp229 + residual_tmp231*residual_tmp250) + residual_tmp248*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp187*residual_tmp45 + residual_tmp239*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp238*residual_tmp50) + residual_tmp178*residual_tmp248);
            const scalar_t grad_coeff2_0 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp213 + residual_tmp7*residual_tmp85) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp255) + residual_tmp253*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp108*residual_tmp7 - residual_tmp12*residual_tmp236 - residual_tmp260) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp259 - residual_tmp262) + residual_tmp253*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp227 + residual_tmp137*residual_tmp7 + residual_tmp267) + residual_tmp261*residual_tmp46 + residual_tmp265*residual_tmp266) + residual_tmp253*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp205 + residual_tmp28*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp254) + residual_tmp12*residual_tmp253) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp224 + residual_tmp176*residual_tmp7 + residual_tmp269) + residual_tmp266*residual_tmp268 + residual_tmp270) + residual_tmp166*residual_tmp253) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp218 + residual_tmp152*residual_tmp7 - residual_tmp264) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp263 - residual_tmp261*u0_grad_1) + residual_tmp253*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp211 + residual_tmp7*residual_tmp72) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp251) + residual_tmp20*residual_tmp253) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp232 + residual_tmp161*residual_tmp7 + residual_tmp257) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp256) + residual_tmp253*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp239 + residual_tmp185*residual_tmp7 - residual_tmp202 - residual_tmp219) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp258) + residual_tmp178*residual_tmp253);
            const scalar_t grad_coeff2_1 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp213*residual_tmp51 + residual_tmp260 - residual_tmp44*residual_tmp85) + residual_tmp255*residual_tmp273 + residual_tmp262) + residual_tmp271*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp108*residual_tmp44 + residual_tmp236*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp259*residual_tmp43) + residual_tmp271*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp137*residual_tmp44 + residual_tmp227*residual_tmp51 - residual_tmp272) - residual_tmp261*u1_grad_0 + ((scalar_t(1) / scalar_t(3)))*residual_tmp265*residual_tmp43) + residual_tmp271*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp205*residual_tmp51 - residual_tmp269 - residual_tmp28*residual_tmp44) + ((scalar_t(1) / scalar_t(3)))*residual_tmp254*residual_tmp43 - residual_tmp270) + residual_tmp12*residual_tmp271) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp176*residual_tmp44 + residual_tmp224*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp268*residual_tmp43) + residual_tmp166*residual_tmp271) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp152*residual_tmp44 + residual_tmp218*residual_tmp51 + residual_tmp274) + residual_tmp141*residual_tmp261 + residual_tmp263*residual_tmp273) + residual_tmp271*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp211*residual_tmp51 - residual_tmp257 - residual_tmp44*residual_tmp72) + ((scalar_t(1) / scalar_t(3)))*residual_tmp251*residual_tmp43) + residual_tmp20*residual_tmp271) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp161*residual_tmp44 + residual_tmp232*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp256*residual_tmp43) + residual_tmp271*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp185*residual_tmp44 - residual_tmp190 + residual_tmp239*residual_tmp51 - residual_tmp249) + ((scalar_t(1) / scalar_t(3)))*residual_tmp258*residual_tmp43) + residual_tmp178*residual_tmp271);
            const scalar_t grad_coeff2_2 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp213*residual_tmp50 + residual_tmp267 + residual_tmp45*residual_tmp85) - residual_tmp255*residual_tmp275 + residual_tmp276*residual_tmp46) + residual_tmp277*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp108*residual_tmp45 + residual_tmp236*residual_tmp50 - residual_tmp272) - residual_tmp259*residual_tmp275 - residual_tmp276*u1_grad_0) + residual_tmp277*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp137*residual_tmp45 + residual_tmp227*residual_tmp50) - residual_tmp265*residual_tmp275) + residual_tmp277*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp205*residual_tmp50 - residual_tmp264 + residual_tmp28*residual_tmp45) - residual_tmp254*residual_tmp275 - residual_tmp276*u0_grad_1) + residual_tmp12*residual_tmp277) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp176*residual_tmp45 + residual_tmp224*residual_tmp50 + residual_tmp274) + residual_tmp141*residual_tmp276 - residual_tmp268*residual_tmp275) + residual_tmp166*residual_tmp277) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp152*residual_tmp45 + residual_tmp218*residual_tmp50) - residual_tmp263*residual_tmp275) + residual_tmp277*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp140 + residual_tmp211*residual_tmp50 - residual_tmp219 + residual_tmp45*residual_tmp72) - residual_tmp251*residual_tmp275) + residual_tmp20*residual_tmp277) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp161*residual_tmp45 - residual_tmp190 + residual_tmp232*residual_tmp50 + residual_tmp245) - residual_tmp256*residual_tmp275) + residual_tmp277*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp185*residual_tmp45 + residual_tmp239*residual_tmp50) - residual_tmp258*residual_tmp275) + residual_tmp178*residual_tmp277);
            grad_coeff0_0_values[lane] = grad_coeff0_0;
            grad_coeff0_1_values[lane] = grad_coeff0_1;
            grad_coeff0_2_values[lane] = grad_coeff0_2;
            grad_coeff1_0_values[lane] = grad_coeff1_0;
            grad_coeff1_1_values[lane] = grad_coeff1_1;
            grad_coeff1_2_values[lane] = grad_coeff1_2;
            grad_coeff2_0_values[lane] = grad_coeff2_0;
            grad_coeff2_1_values[lane] = grad_coeff2_1;
            grad_coeff2_2_values[lane] = grad_coeff2_2;
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
                output[test * N_FIELDS + 1][lane] += q_weight[q] * det * (grad_coeff1_0_values[lane] * test_grad0 + grad_coeff1_1_values[lane] * test_grad1 + grad_coeff1_2_values[lane] * test_grad2);
                output[test * N_FIELDS + 2][lane] += q_weight[q] * det * (grad_coeff2_0_values[lane] * test_grad0 + grad_coeff2_1_values[lane] * test_grad1 + grad_coeff2_2_values[lane] * test_grad2);
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_tet4_jacobian_action_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t *const SFEM_RESTRICT current[3 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT previous[3 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT direction[3 * N_SHAPE],
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t *const SFEM_RESTRICT output[3 * N_SHAPE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 3;
    for (int q = 0; q < N_QP; ++q) {
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
            const scalar_t u0_grad_0_ref = -(current[0][lane]) + current[3][lane];
            const scalar_t u0_grad_1_ref = -(current[0][lane]) + current[6][lane];
            const scalar_t u0_grad_2_ref = -(current[0][lane]) + current[9][lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
            const scalar_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
            const scalar_t u0_old_grad_0_ref = -(previous[0][lane]) + previous[3][lane];
            const scalar_t u0_old_grad_1_ref = -(previous[0][lane]) + previous[6][lane];
            const scalar_t u0_old_grad_2_ref = -(previous[0][lane]) + previous[9][lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
            const scalar_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
            const scalar_t u0_direction_grad_0_ref = -(direction[0][lane]) + direction[3][lane];
            const scalar_t u0_direction_grad_1_ref = -(direction[0][lane]) + direction[6][lane];
            const scalar_t u0_direction_grad_2_ref = -(direction[0][lane]) + direction[9][lane];
            const scalar_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
            const scalar_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
            const scalar_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
            const scalar_t u1_grad_0_ref = -(current[1][lane]) + current[4][lane];
            const scalar_t u1_grad_1_ref = -(current[1][lane]) + current[7][lane];
            const scalar_t u1_grad_2_ref = -(current[1][lane]) + current[10][lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
            const scalar_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
            const scalar_t u1_old_grad_0_ref = -(previous[1][lane]) + previous[4][lane];
            const scalar_t u1_old_grad_1_ref = -(previous[1][lane]) + previous[7][lane];
            const scalar_t u1_old_grad_2_ref = -(previous[1][lane]) + previous[10][lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
            const scalar_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
            const scalar_t u1_direction_grad_0_ref = -(direction[1][lane]) + direction[4][lane];
            const scalar_t u1_direction_grad_1_ref = -(direction[1][lane]) + direction[7][lane];
            const scalar_t u1_direction_grad_2_ref = -(direction[1][lane]) + direction[10][lane];
            const scalar_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
            const scalar_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
            const scalar_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
            const scalar_t u2_grad_0_ref = -(current[2][lane]) + current[5][lane];
            const scalar_t u2_grad_1_ref = -(current[2][lane]) + current[8][lane];
            const scalar_t u2_grad_2_ref = -(current[2][lane]) + current[11][lane];
            const scalar_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
            const scalar_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
            const scalar_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
            const scalar_t u2_old_grad_0_ref = -(previous[2][lane]) + previous[5][lane];
            const scalar_t u2_old_grad_1_ref = -(previous[2][lane]) + previous[8][lane];
            const scalar_t u2_old_grad_2_ref = -(previous[2][lane]) + previous[11][lane];
            const scalar_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
            const scalar_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
            const scalar_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
            const scalar_t u2_direction_grad_0_ref = -(direction[2][lane]) + direction[5][lane];
            const scalar_t u2_direction_grad_1_ref = -(direction[2][lane]) + direction[8][lane];
            const scalar_t u2_direction_grad_2_ref = -(direction[2][lane]) + direction[11][lane];
            const scalar_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
            const scalar_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
            const scalar_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
            const scalar_t residual_tmp0 = u0_grad_0*u1_grad_1;
            const scalar_t residual_tmp1 = u0_grad_1*u1_grad_2;
            const scalar_t residual_tmp2 = u0_grad_2*u2_grad_1;
            const scalar_t residual_tmp3 = u1_grad_2*u2_grad_1;
            const scalar_t residual_tmp4 = u0_grad_1*u1_grad_0;
            const scalar_t residual_tmp5 = u0_grad_2*u2_grad_0;
            const scalar_t residual_tmp6 = u2_grad_2 + scalar_t(1);
            const scalar_t residual_tmp7 = -residual_tmp3 + residual_tmp6 + u1_grad_1*u2_grad_2 + u1_grad_1;
            const scalar_t residual_tmp8 = -residual_tmp5 + u0_grad_0*u2_grad_2 + u0_grad_0;
            const scalar_t residual_tmp9 = residual_tmp0 - residual_tmp4;
            const scalar_t residual_tmp10 = residual_tmp0*u2_grad_2 + residual_tmp1*u2_grad_0 + residual_tmp2*u1_grad_0 - residual_tmp3*u0_grad_0 - residual_tmp4*u2_grad_2 - residual_tmp5*u1_grad_1 + residual_tmp7 + residual_tmp8 + residual_tmp9;
            const scalar_t residual_tmp11 = pow_m1(residual_tmp10);
            const scalar_t residual_tmp12 = -residual_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
            const scalar_t residual_tmp13 = newmark_velocity_alpha*residual_tmp7;
            const scalar_t residual_tmp14 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
            const scalar_t residual_tmp15 = residual_tmp14*u2_grad_1;
            const scalar_t residual_tmp16 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp17 = residual_tmp16*residual_tmp6;
            const scalar_t residual_tmp18 = residual_tmp15 - residual_tmp17;
            const scalar_t residual_tmp19 = -residual_tmp13 - residual_tmp18;
            const scalar_t residual_tmp20 = -residual_tmp1 + u0_grad_2*u1_grad_1 + u0_grad_2;
            const scalar_t residual_tmp21 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
            const scalar_t residual_tmp22 = residual_tmp21*residual_tmp6;
            const scalar_t residual_tmp23 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
            const scalar_t residual_tmp24 = residual_tmp23*u2_grad_1;
            const scalar_t residual_tmp25 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
            const scalar_t residual_tmp26 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
            const scalar_t residual_tmp27 = residual_tmp25*u0_grad_1 - residual_tmp26*u0_grad_2;
            const scalar_t residual_tmp28 = residual_tmp22 - residual_tmp24 + residual_tmp27;
            const scalar_t residual_tmp29 = newmark_velocity_alpha*residual_tmp12;
            const scalar_t residual_tmp30 = residual_tmp23*u0_grad_1;
            const scalar_t residual_tmp31 = residual_tmp21*u0_grad_2;
            const scalar_t residual_tmp32 = residual_tmp29 + residual_tmp30 - residual_tmp31;
            const scalar_t residual_tmp33 = residual_tmp26*residual_tmp6;
            const scalar_t residual_tmp34 = residual_tmp25*u2_grad_1;
            const scalar_t residual_tmp35 = residual_tmp33 - residual_tmp34;
            const scalar_t residual_tmp36 = scalar_t(3)*eta_b;
            const scalar_t residual_tmp37 = residual_tmp36*(residual_tmp32 + residual_tmp35);
            const scalar_t residual_tmp38 = scalar_t(2)*eta_s;
            const scalar_t residual_tmp39 = residual_tmp37 + residual_tmp38*(scalar_t(2)*residual_tmp26*residual_tmp6 - residual_tmp32 - scalar_t(2)*residual_tmp34);
            const scalar_t residual_tmp40 = ((scalar_t(1) / scalar_t(3)))*residual_tmp7;
            const scalar_t residual_tmp41 = pow_m2(residual_tmp10);
            const scalar_t residual_tmp42 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp43 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
            const scalar_t residual_tmp44 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
            const scalar_t residual_tmp45 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
            const scalar_t residual_tmp46 = u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp47 = residual_tmp46 + residual_tmp9 + u0_grad_0;
            const scalar_t residual_tmp48 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
            const scalar_t residual_tmp49 = residual_tmp20*residual_tmp42 + residual_tmp21*residual_tmp44 + residual_tmp23*residual_tmp45 - residual_tmp25*residual_tmp47 + residual_tmp26*residual_tmp43 - residual_tmp48*residual_tmp7;
            const scalar_t residual_tmp50 = u0_grad_0*u2_grad_1 - u0_grad_1*u2_grad_0 + u2_grad_1;
            const scalar_t residual_tmp51 = residual_tmp6 + residual_tmp8;
            const scalar_t residual_tmp52 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
            const scalar_t residual_tmp53 = residual_tmp12*residual_tmp42 + residual_tmp14*residual_tmp45 + residual_tmp16*residual_tmp44 + residual_tmp25*residual_tmp50 - residual_tmp26*residual_tmp51 - residual_tmp52*residual_tmp7;
            const scalar_t residual_tmp54 = residual_tmp12*residual_tmp52;
            const scalar_t residual_tmp55 = residual_tmp14*residual_tmp50;
            const scalar_t residual_tmp56 = residual_tmp20*residual_tmp48;
            const scalar_t residual_tmp57 = residual_tmp21*residual_tmp43;
            const scalar_t residual_tmp58 = residual_tmp16*residual_tmp51;
            const scalar_t residual_tmp59 = -residual_tmp58;
            const scalar_t residual_tmp60 = residual_tmp23*residual_tmp47;
            const scalar_t residual_tmp61 = -residual_tmp60;
            const scalar_t residual_tmp62 = residual_tmp54 + residual_tmp55 + residual_tmp56 + residual_tmp57 + residual_tmp59 + residual_tmp61;
            const scalar_t residual_tmp63 = residual_tmp42*residual_tmp7;
            const scalar_t residual_tmp64 = residual_tmp25*residual_tmp45 + residual_tmp26*residual_tmp44 - residual_tmp63;
            const scalar_t residual_tmp65 = residual_tmp36*(residual_tmp62 + residual_tmp64);
            const scalar_t residual_tmp66 = residual_tmp38*(scalar_t(2)*residual_tmp25*residual_tmp45 + scalar_t(2)*residual_tmp26*residual_tmp44 - residual_tmp62 - scalar_t(2)*residual_tmp63) + residual_tmp65;
            const scalar_t residual_tmp67 = -residual_tmp66;
            const scalar_t residual_tmp68 = residual_tmp41*(eta_s*(residual_tmp12*residual_tmp53 + residual_tmp20*residual_tmp49) + residual_tmp40*residual_tmp67);
            const scalar_t residual_tmp69 = residual_tmp21*u1_grad_2;
            const scalar_t residual_tmp70 = residual_tmp23*residual_tmp46;
            const scalar_t residual_tmp71 = residual_tmp69 - residual_tmp70;
            const scalar_t residual_tmp72 = -residual_tmp13 - residual_tmp71;
            const scalar_t residual_tmp73 = -residual_tmp14*residual_tmp46 + residual_tmp16*u1_grad_2;
            const scalar_t residual_tmp74 = -residual_tmp27 - residual_tmp73;
            const scalar_t residual_tmp75 = newmark_velocity_alpha*residual_tmp20;
            const scalar_t residual_tmp76 = residual_tmp16*u0_grad_2;
            const scalar_t residual_tmp77 = residual_tmp14*u0_grad_1;
            const scalar_t residual_tmp78 = residual_tmp75 + residual_tmp76 - residual_tmp77;
            const scalar_t residual_tmp79 = residual_tmp25*residual_tmp46;
            const scalar_t residual_tmp80 = residual_tmp26*u1_grad_2;
            const scalar_t residual_tmp81 = residual_tmp79 - residual_tmp80;
            const scalar_t residual_tmp82 = residual_tmp36*(residual_tmp78 + residual_tmp81);
            const scalar_t residual_tmp83 = residual_tmp38*(scalar_t(2)*residual_tmp25*residual_tmp46 - residual_tmp78 - scalar_t(2)*residual_tmp80) + residual_tmp82;
            const scalar_t residual_tmp84 = -residual_tmp79 + residual_tmp80;
            const scalar_t residual_tmp85 = residual_tmp75 + residual_tmp84;
            const scalar_t residual_tmp86 = -residual_tmp33 + residual_tmp34;
            const scalar_t residual_tmp87 = residual_tmp29 + residual_tmp86;
            const scalar_t residual_tmp88 = residual_tmp13 - residual_tmp69 + residual_tmp70;
            const scalar_t residual_tmp89 = -residual_tmp15 + residual_tmp17;
            const scalar_t residual_tmp90 = residual_tmp36*(-residual_tmp88 - residual_tmp89);
            const scalar_t residual_tmp91 = residual_tmp38*(-scalar_t(2)*residual_tmp13 - residual_tmp18 - residual_tmp71) + residual_tmp90;
            const scalar_t residual_tmp92 = -residual_tmp7;
            const scalar_t residual_tmp93 = newmark_velocity_alpha*residual_tmp44;
            const scalar_t residual_tmp94 = residual_tmp23*u1_grad_0;
            const scalar_t residual_tmp95 = residual_tmp48*u1_grad_2;
            const scalar_t residual_tmp96 = residual_tmp93 + residual_tmp94 - residual_tmp95;
            const scalar_t residual_tmp97 = residual_tmp52*residual_tmp6;
            const scalar_t residual_tmp98 = residual_tmp14*u2_grad_0;
            const scalar_t residual_tmp99 = residual_tmp97 - residual_tmp98;
            const scalar_t residual_tmp100 = residual_tmp36*(residual_tmp96 + residual_tmp99);
            const scalar_t residual_tmp101 = -residual_tmp97 + residual_tmp98;
            const scalar_t residual_tmp102 = -residual_tmp94 + residual_tmp95;
            const scalar_t residual_tmp103 = residual_tmp100 + residual_tmp38*(residual_tmp101 + residual_tmp102 + scalar_t(2)*residual_tmp93);
            const scalar_t residual_tmp104 = newmark_velocity_alpha*residual_tmp43;
            const scalar_t residual_tmp105 = residual_tmp25*u1_grad_0;
            const scalar_t residual_tmp106 = residual_tmp42*u1_grad_2;
            const scalar_t residual_tmp107 = residual_tmp105 - residual_tmp106;
            const scalar_t residual_tmp108 = residual_tmp104 + residual_tmp107;
            const scalar_t residual_tmp109 = newmark_velocity_alpha*residual_tmp51;
            const scalar_t residual_tmp110 = residual_tmp25*u2_grad_0;
            const scalar_t residual_tmp111 = residual_tmp42*residual_tmp6;
            const scalar_t residual_tmp112 = residual_tmp110 - residual_tmp111;
            const scalar_t residual_tmp113 = -residual_tmp109 - residual_tmp112;
            const scalar_t residual_tmp114 = residual_tmp49*u1_grad_2;
            const scalar_t residual_tmp115 = -residual_tmp114;
            const scalar_t residual_tmp116 = residual_tmp53*residual_tmp6;
            const scalar_t residual_tmp117 = newmark_velocity_alpha*residual_tmp45;
            const scalar_t residual_tmp118 = residual_tmp46*residual_tmp48;
            const scalar_t residual_tmp119 = residual_tmp21*u1_grad_0;
            const scalar_t residual_tmp120 = residual_tmp117 + residual_tmp118 - residual_tmp119;
            const scalar_t residual_tmp121 = residual_tmp16*u2_grad_0;
            const scalar_t residual_tmp122 = residual_tmp52*u2_grad_1;
            const scalar_t residual_tmp123 = residual_tmp121 - residual_tmp122;
            const scalar_t residual_tmp124 = residual_tmp36*(residual_tmp120 + residual_tmp123);
            const scalar_t residual_tmp125 = -residual_tmp121 + residual_tmp122;
            const scalar_t residual_tmp126 = -residual_tmp118 + residual_tmp119;
            const scalar_t residual_tmp127 = residual_tmp124 + residual_tmp38*(scalar_t(2)*residual_tmp117 + residual_tmp125 + residual_tmp126);
            const scalar_t residual_tmp128 = newmark_velocity_alpha*residual_tmp50;
            const scalar_t residual_tmp129 = residual_tmp26*u2_grad_0;
            const scalar_t residual_tmp130 = residual_tmp42*u2_grad_1;
            const scalar_t residual_tmp131 = residual_tmp129 - residual_tmp130;
            const scalar_t residual_tmp132 = residual_tmp128 + residual_tmp131;
            const scalar_t residual_tmp133 = newmark_velocity_alpha*residual_tmp47;
            const scalar_t residual_tmp134 = residual_tmp26*u1_grad_0;
            const scalar_t residual_tmp135 = residual_tmp42*residual_tmp46;
            const scalar_t residual_tmp136 = residual_tmp134 - residual_tmp135;
            const scalar_t residual_tmp137 = -residual_tmp133 - residual_tmp136;
            const scalar_t residual_tmp138 = residual_tmp53*u2_grad_1;
            const scalar_t residual_tmp139 = -residual_tmp138;
            const scalar_t residual_tmp140 = residual_tmp46*residual_tmp49;
            const scalar_t residual_tmp141 = u0_grad_0 + scalar_t(1);
            const scalar_t residual_tmp142 = residual_tmp141*residual_tmp21;
            const scalar_t residual_tmp143 = residual_tmp48*u0_grad_1;
            const scalar_t residual_tmp144 = residual_tmp128 + residual_tmp142 - residual_tmp143;
            const scalar_t residual_tmp145 = -residual_tmp129 + residual_tmp130;
            const scalar_t residual_tmp146 = residual_tmp36*(residual_tmp144 + residual_tmp145);
            const scalar_t residual_tmp147 = residual_tmp146 + residual_tmp38*(-scalar_t(2)*residual_tmp129 - residual_tmp144 + scalar_t(2)*residual_tmp42*u2_grad_1);
            const scalar_t residual_tmp148 = residual_tmp117 + residual_tmp125;
            const scalar_t residual_tmp149 = residual_tmp21*u2_grad_0;
            const scalar_t residual_tmp150 = residual_tmp48*u2_grad_1;
            const scalar_t residual_tmp151 = -residual_tmp141*residual_tmp26 + residual_tmp42*u0_grad_1;
            const scalar_t residual_tmp152 = -residual_tmp149 + residual_tmp150 - residual_tmp151;
            const scalar_t residual_tmp153 = residual_tmp49*u0_grad_1;
            const scalar_t residual_tmp154 = ((scalar_t(1) / scalar_t(3)))*residual_tmp67;
            const scalar_t residual_tmp155 = residual_tmp14*residual_tmp141;
            const scalar_t residual_tmp156 = residual_tmp52*u0_grad_2;
            const scalar_t residual_tmp157 = residual_tmp104 + residual_tmp155 - residual_tmp156;
            const scalar_t residual_tmp158 = -residual_tmp105 + residual_tmp106;
            const scalar_t residual_tmp159 = residual_tmp36*(residual_tmp157 + residual_tmp158);
            const scalar_t residual_tmp160 = residual_tmp159 + residual_tmp38*(-scalar_t(2)*residual_tmp105 - residual_tmp157 + scalar_t(2)*residual_tmp42*u1_grad_2);
            const scalar_t residual_tmp161 = residual_tmp102 + residual_tmp93;
            const scalar_t residual_tmp162 = -residual_tmp141*residual_tmp25 + residual_tmp42*u0_grad_2;
            const scalar_t residual_tmp163 = residual_tmp14*u1_grad_0 - residual_tmp52*u1_grad_2;
            const scalar_t residual_tmp164 = -residual_tmp162 - residual_tmp163;
            const scalar_t residual_tmp165 = residual_tmp53*u0_grad_2;
            const scalar_t residual_tmp166 = -residual_tmp51;
            const scalar_t residual_tmp167 = residual_tmp141*residual_tmp23;
            const scalar_t residual_tmp168 = residual_tmp48*u0_grad_2;
            const scalar_t residual_tmp169 = residual_tmp109 + residual_tmp167 - residual_tmp168;
            const scalar_t residual_tmp170 = -residual_tmp110 + residual_tmp111;
            const scalar_t residual_tmp171 = residual_tmp36*(-residual_tmp169 - residual_tmp170);
            const scalar_t residual_tmp172 = residual_tmp171 + residual_tmp38*(scalar_t(2)*residual_tmp110 - scalar_t(2)*residual_tmp111 + residual_tmp169);
            const scalar_t residual_tmp173 = residual_tmp101 + residual_tmp93;
            const scalar_t residual_tmp174 = residual_tmp23*u2_grad_0;
            const scalar_t residual_tmp175 = residual_tmp48*residual_tmp6;
            const scalar_t residual_tmp176 = residual_tmp162 + residual_tmp174 - residual_tmp175;
            const scalar_t residual_tmp177 = residual_tmp49*u0_grad_2;
            const scalar_t residual_tmp178 = -residual_tmp47;
            const scalar_t residual_tmp179 = residual_tmp141*residual_tmp16;
            const scalar_t residual_tmp180 = residual_tmp52*u0_grad_1;
            const scalar_t residual_tmp181 = residual_tmp133 + residual_tmp179 - residual_tmp180;
            const scalar_t residual_tmp182 = -residual_tmp134 + residual_tmp135;
            const scalar_t residual_tmp183 = residual_tmp36*(-residual_tmp181 - residual_tmp182);
            const scalar_t residual_tmp184 = residual_tmp183 + residual_tmp38*(scalar_t(2)*residual_tmp134 - scalar_t(2)*residual_tmp135 + residual_tmp181);
            const scalar_t residual_tmp185 = residual_tmp117 + residual_tmp126;
            const scalar_t residual_tmp186 = residual_tmp16*u1_grad_0 - residual_tmp46*residual_tmp52;
            const scalar_t residual_tmp187 = residual_tmp151 + residual_tmp186;
            const scalar_t residual_tmp188 = residual_tmp53*u0_grad_1;
            const scalar_t residual_tmp189 = residual_tmp41*(-eta_s*(-residual_tmp43*residual_tmp49 + residual_tmp51*residual_tmp53) + ((scalar_t(1) / scalar_t(3)))*residual_tmp44*residual_tmp66);
            const scalar_t residual_tmp190 = residual_tmp49*u1_grad_0;
            const scalar_t residual_tmp191 = residual_tmp53*u2_grad_0;
            const scalar_t residual_tmp192 = -residual_tmp191;
            const scalar_t residual_tmp193 = residual_tmp190 + residual_tmp192;
            const scalar_t residual_tmp194 = -residual_tmp53*residual_tmp6;
            const scalar_t residual_tmp195 = residual_tmp141*residual_tmp49;
            const scalar_t residual_tmp196 = ((scalar_t(1) / scalar_t(3)))*residual_tmp66;
            const scalar_t residual_tmp197 = residual_tmp196*u2_grad_0;
            const scalar_t residual_tmp198 = ((scalar_t(1) / scalar_t(3)))*residual_tmp44;
            const scalar_t residual_tmp199 = residual_tmp141*residual_tmp53;
            const scalar_t residual_tmp200 = residual_tmp196*u1_grad_0;
            const scalar_t residual_tmp201 = residual_tmp41*(-eta_s*(residual_tmp47*residual_tmp49 - residual_tmp50*residual_tmp53) + ((scalar_t(1) / scalar_t(3)))*residual_tmp45*residual_tmp66);
            const scalar_t residual_tmp202 = -residual_tmp46*residual_tmp49;
            const scalar_t residual_tmp203 = ((scalar_t(1) / scalar_t(3)))*residual_tmp45;
            const scalar_t residual_tmp204 = -residual_tmp76 + residual_tmp77;
            const scalar_t residual_tmp205 = residual_tmp204 + residual_tmp75;
            const scalar_t residual_tmp206 = -residual_tmp30 + residual_tmp31;
            const scalar_t residual_tmp207 = residual_tmp37 + residual_tmp38*(residual_tmp206 + scalar_t(2)*residual_tmp29 + residual_tmp86);
            const scalar_t residual_tmp208 = residual_tmp12*residual_tmp48 - residual_tmp14*residual_tmp47 + residual_tmp16*residual_tmp43 + residual_tmp20*residual_tmp52 - residual_tmp21*residual_tmp51 + residual_tmp23*residual_tmp50;
            const scalar_t residual_tmp209 = residual_tmp38*(scalar_t(2)*residual_tmp12*residual_tmp52 + scalar_t(2)*residual_tmp14*residual_tmp50 - residual_tmp56 - residual_tmp57 - scalar_t(2)*residual_tmp58 - residual_tmp61 - residual_tmp64) + residual_tmp65;
            const scalar_t residual_tmp210 = residual_tmp41*(-eta_s*(-residual_tmp20*residual_tmp208 + residual_tmp53*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp209);
            const scalar_t residual_tmp211 = residual_tmp206 + residual_tmp29;
            const scalar_t residual_tmp212 = residual_tmp38*(scalar_t(2)*residual_tmp16*u0_grad_2 - residual_tmp75 - scalar_t(2)*residual_tmp77 - residual_tmp81) + residual_tmp82;
            const scalar_t residual_tmp213 = -residual_tmp22 + residual_tmp24 + residual_tmp73;
            const scalar_t residual_tmp214 = residual_tmp38*(scalar_t(2)*residual_tmp15 - scalar_t(2)*residual_tmp17 + residual_tmp88) + residual_tmp90;
            const scalar_t residual_tmp215 = -residual_tmp142 + residual_tmp143;
            const scalar_t residual_tmp216 = residual_tmp146 + residual_tmp38*(scalar_t(2)*residual_tmp128 + residual_tmp131 + residual_tmp215);
            const scalar_t residual_tmp217 = -residual_tmp179 + residual_tmp180;
            const scalar_t residual_tmp218 = -residual_tmp133 - residual_tmp217;
            const scalar_t residual_tmp219 = residual_tmp208*u0_grad_1;
            const scalar_t residual_tmp220 = residual_tmp139 + residual_tmp219;
            const scalar_t residual_tmp221 = -residual_tmp167 + residual_tmp168;
            const scalar_t residual_tmp222 = residual_tmp171 + residual_tmp38*(-scalar_t(2)*residual_tmp109 - residual_tmp112 - residual_tmp221);
            const scalar_t residual_tmp223 = -residual_tmp155 + residual_tmp156;
            const scalar_t residual_tmp224 = residual_tmp104 + residual_tmp223;
            const scalar_t residual_tmp225 = residual_tmp208*u0_grad_2;
            const scalar_t residual_tmp226 = residual_tmp124 + residual_tmp38*(-residual_tmp120 - scalar_t(2)*residual_tmp122 + scalar_t(2)*residual_tmp16*u2_grad_0);
            const scalar_t residual_tmp227 = residual_tmp149 - residual_tmp150 - residual_tmp186;
            const scalar_t residual_tmp228 = residual_tmp208*residual_tmp46;
            const scalar_t residual_tmp229 = ((scalar_t(1) / scalar_t(3)))*residual_tmp209;
            const scalar_t residual_tmp230 = residual_tmp229*u2_grad_1;
            const scalar_t residual_tmp231 = residual_tmp159 + residual_tmp38*(-residual_tmp104 + scalar_t(2)*residual_tmp14*residual_tmp141 - scalar_t(2)*residual_tmp156 - residual_tmp158);
            const scalar_t residual_tmp232 = -residual_tmp109 - residual_tmp221;
            const scalar_t residual_tmp233 = residual_tmp53*u1_grad_2;
            const scalar_t residual_tmp234 = residual_tmp100 + residual_tmp38*(scalar_t(2)*residual_tmp52*residual_tmp6 - residual_tmp96 - scalar_t(2)*residual_tmp98);
            const scalar_t residual_tmp235 = ((scalar_t(1) / scalar_t(3)))*residual_tmp12;
            const scalar_t residual_tmp236 = residual_tmp163 - residual_tmp174 + residual_tmp175;
            const scalar_t residual_tmp237 = residual_tmp208*u1_grad_2;
            const scalar_t residual_tmp238 = residual_tmp183 + residual_tmp38*(residual_tmp133 - scalar_t(2)*residual_tmp179 + scalar_t(2)*residual_tmp180 + residual_tmp182);
            const scalar_t residual_tmp239 = residual_tmp128 + residual_tmp215;
            const scalar_t residual_tmp240 = residual_tmp46*residual_tmp53;
            const scalar_t residual_tmp241 = residual_tmp229*u0_grad_1;
            const scalar_t residual_tmp242 = ((scalar_t(1) / scalar_t(3)))*residual_tmp51;
            const scalar_t residual_tmp243 = -(scalar_t(1) / scalar_t(3))*residual_tmp209;
            const scalar_t residual_tmp244 = residual_tmp41*(eta_s*(residual_tmp208*residual_tmp43 + residual_tmp44*residual_tmp53) + residual_tmp243*residual_tmp51);
            const scalar_t residual_tmp245 = residual_tmp141*residual_tmp208;
            const scalar_t residual_tmp246 = residual_tmp208*u1_grad_0;
            const scalar_t residual_tmp247 = residual_tmp53*u1_grad_0;
            const scalar_t residual_tmp248 = residual_tmp41*(-eta_s*(residual_tmp208*residual_tmp47 - residual_tmp45*residual_tmp53) + ((scalar_t(1) / scalar_t(3)))*residual_tmp209*residual_tmp50);
            const scalar_t residual_tmp249 = -residual_tmp141*residual_tmp208;
            const scalar_t residual_tmp250 = ((scalar_t(1) / scalar_t(3)))*residual_tmp50;
            const scalar_t residual_tmp251 = residual_tmp38*(residual_tmp204 + scalar_t(2)*residual_tmp75 + residual_tmp84) + residual_tmp82;
            const scalar_t residual_tmp252 = residual_tmp38*(scalar_t(2)*residual_tmp20*residual_tmp48 + scalar_t(2)*residual_tmp21*residual_tmp43 - residual_tmp54 - residual_tmp55 - residual_tmp59 - scalar_t(2)*residual_tmp60 - residual_tmp64) + residual_tmp65;
            const scalar_t residual_tmp253 = residual_tmp41*(-eta_s*(-residual_tmp12*residual_tmp208 + residual_tmp49*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp252);
            const scalar_t residual_tmp254 = residual_tmp37 + residual_tmp38*(scalar_t(2)*residual_tmp23*u0_grad_1 - residual_tmp29 - scalar_t(2)*residual_tmp31 - residual_tmp35);
            const scalar_t residual_tmp255 = residual_tmp38*(residual_tmp13 + scalar_t(2)*residual_tmp69 - scalar_t(2)*residual_tmp70 + residual_tmp89) + residual_tmp90;
            const scalar_t residual_tmp256 = residual_tmp159 + residual_tmp38*(scalar_t(2)*residual_tmp104 + residual_tmp107 + residual_tmp223);
            const scalar_t residual_tmp257 = residual_tmp115 + residual_tmp225;
            const scalar_t residual_tmp258 = residual_tmp183 + residual_tmp38*(-scalar_t(2)*residual_tmp133 - residual_tmp136 - residual_tmp217);
            const scalar_t residual_tmp259 = residual_tmp100 + residual_tmp38*(scalar_t(2)*residual_tmp23*u1_grad_0 - residual_tmp93 - scalar_t(2)*residual_tmp95 - residual_tmp99);
            const scalar_t residual_tmp260 = residual_tmp208*residual_tmp6;
            const scalar_t residual_tmp261 = ((scalar_t(1) / scalar_t(3)))*residual_tmp252;
            const scalar_t residual_tmp262 = residual_tmp261*u1_grad_2;
            const scalar_t residual_tmp263 = residual_tmp146 + residual_tmp38*(-residual_tmp128 + scalar_t(2)*residual_tmp141*residual_tmp21 - scalar_t(2)*residual_tmp143 - residual_tmp145);
            const scalar_t residual_tmp264 = residual_tmp49*u2_grad_1;
            const scalar_t residual_tmp265 = residual_tmp124 + residual_tmp38*(-residual_tmp117 - scalar_t(2)*residual_tmp119 - residual_tmp123 + scalar_t(2)*residual_tmp46*residual_tmp48);
            const scalar_t residual_tmp266 = ((scalar_t(1) / scalar_t(3)))*residual_tmp20;
            const scalar_t residual_tmp267 = residual_tmp208*u2_grad_1;
            const scalar_t residual_tmp268 = residual_tmp171 + residual_tmp38*(residual_tmp109 - scalar_t(2)*residual_tmp167 + scalar_t(2)*residual_tmp168 + residual_tmp170);
            const scalar_t residual_tmp269 = residual_tmp49*residual_tmp6;
            const scalar_t residual_tmp270 = residual_tmp261*u0_grad_2;
            const scalar_t residual_tmp271 = residual_tmp41*(-eta_s*(residual_tmp208*residual_tmp51 - residual_tmp44*residual_tmp49) + ((scalar_t(1) / scalar_t(3)))*residual_tmp252*residual_tmp43);
            const scalar_t residual_tmp272 = residual_tmp208*u2_grad_0;
            const scalar_t residual_tmp273 = ((scalar_t(1) / scalar_t(3)))*residual_tmp43;
            const scalar_t residual_tmp274 = residual_tmp49*u2_grad_0;
            const scalar_t residual_tmp275 = ((scalar_t(1) / scalar_t(3)))*residual_tmp47;
            const scalar_t residual_tmp276 = -(scalar_t(1) / scalar_t(3))*residual_tmp252;
            const scalar_t residual_tmp277 = residual_tmp41*(eta_s*(residual_tmp208*residual_tmp50 + residual_tmp45*residual_tmp49) + residual_tmp276*residual_tmp47);
            const scalar_t grad_coeff0_0 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp87 + residual_tmp20*residual_tmp85) - residual_tmp40*residual_tmp91) + residual_tmp68*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp108*residual_tmp20 + residual_tmp113*residual_tmp12 + residual_tmp115 + residual_tmp116) - residual_tmp103*residual_tmp40) + residual_tmp44*residual_tmp68) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp132 + residual_tmp137*residual_tmp20 + residual_tmp139 + residual_tmp140) - residual_tmp127*residual_tmp40) + residual_tmp45*residual_tmp68) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp19 + residual_tmp20*residual_tmp28) - residual_tmp39*residual_tmp40) + residual_tmp12*residual_tmp68) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp173 + residual_tmp176*residual_tmp20 + residual_tmp177) + residual_tmp154*residual_tmp6 - residual_tmp172*residual_tmp40) + residual_tmp166*residual_tmp68) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp148 + residual_tmp152*residual_tmp20 - residual_tmp153) - residual_tmp147*residual_tmp40 - residual_tmp154*u2_grad_1) + residual_tmp50*residual_tmp68) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp74 + residual_tmp20*residual_tmp72) - residual_tmp40*residual_tmp83) + residual_tmp20*residual_tmp68) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp164 + residual_tmp161*residual_tmp20 - residual_tmp165) - residual_tmp154*u1_grad_2 - residual_tmp160*residual_tmp40) + residual_tmp43*residual_tmp68) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp187 + residual_tmp185*residual_tmp20 + residual_tmp188) + residual_tmp154*residual_tmp46 - residual_tmp184*residual_tmp40) + residual_tmp178*residual_tmp68);
            const scalar_t grad_coeff0_1 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp114 - residual_tmp194 - residual_tmp43*residual_tmp85 + residual_tmp51*residual_tmp87) + ((scalar_t(1) / scalar_t(3)))*residual_tmp44*residual_tmp91) + residual_tmp189*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp108*residual_tmp43 + residual_tmp113*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp103*residual_tmp44) + residual_tmp189*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp132*residual_tmp51 - residual_tmp137*residual_tmp43 + residual_tmp193) + ((scalar_t(1) / scalar_t(3)))*residual_tmp127*residual_tmp44) + residual_tmp189*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp177 + residual_tmp19*residual_tmp51 - residual_tmp28*residual_tmp43) + residual_tmp196*residual_tmp6 + residual_tmp198*residual_tmp39) + residual_tmp12*residual_tmp189) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp173*residual_tmp51 - residual_tmp176*residual_tmp43) + ((scalar_t(1) / scalar_t(3)))*residual_tmp172*residual_tmp44) + residual_tmp166*residual_tmp189) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp148*residual_tmp51 - residual_tmp152*residual_tmp43 - residual_tmp195) + ((scalar_t(1) / scalar_t(3)))*residual_tmp147*residual_tmp44 - residual_tmp197) + residual_tmp189*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp165 - residual_tmp43*residual_tmp72 + residual_tmp51*residual_tmp74) - residual_tmp196*u1_grad_2 + ((scalar_t(1) / scalar_t(3)))*residual_tmp44*residual_tmp83) + residual_tmp189*residual_tmp20) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp161*residual_tmp43 + residual_tmp164*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp160*residual_tmp44) + residual_tmp189*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp185*residual_tmp43 + residual_tmp187*residual_tmp51 + residual_tmp199) + residual_tmp184*residual_tmp198 + residual_tmp200) + residual_tmp178*residual_tmp189);
            const scalar_t grad_coeff0_2 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp138 - residual_tmp202 + residual_tmp47*residual_tmp85 - residual_tmp50*residual_tmp87) + ((scalar_t(1) / scalar_t(3)))*residual_tmp45*residual_tmp91) + residual_tmp201*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp108*residual_tmp47 - residual_tmp113*residual_tmp50 - residual_tmp193) + ((scalar_t(1) / scalar_t(3)))*residual_tmp103*residual_tmp45) + residual_tmp201*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp132*residual_tmp50 + residual_tmp137*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp127*residual_tmp45) + residual_tmp201*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp153 - residual_tmp19*residual_tmp50 + residual_tmp28*residual_tmp47) - residual_tmp196*u2_grad_1 + ((scalar_t(1) / scalar_t(3)))*residual_tmp39*residual_tmp45) + residual_tmp12*residual_tmp201) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp173*residual_tmp50 + residual_tmp176*residual_tmp47 + residual_tmp195) + residual_tmp172*residual_tmp203 + residual_tmp197) + residual_tmp166*residual_tmp201) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp148*residual_tmp50 + residual_tmp152*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp147*residual_tmp45) + residual_tmp201*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp188 + residual_tmp47*residual_tmp72 - residual_tmp50*residual_tmp74) + residual_tmp196*residual_tmp46 + residual_tmp203*residual_tmp83) + residual_tmp20*residual_tmp201) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp161*residual_tmp47 - residual_tmp164*residual_tmp50 - residual_tmp199) + ((scalar_t(1) / scalar_t(3)))*residual_tmp160*residual_tmp45 - residual_tmp200) + residual_tmp201*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp185*residual_tmp47 - residual_tmp187*residual_tmp50) + ((scalar_t(1) / scalar_t(3)))*residual_tmp184*residual_tmp45) + residual_tmp178*residual_tmp201);
            const scalar_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp20*residual_tmp213 + residual_tmp7*residual_tmp87) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp214) + residual_tmp210*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp113*residual_tmp7 - residual_tmp20*residual_tmp236 + residual_tmp237) + residual_tmp229*residual_tmp6 + residual_tmp234*residual_tmp235) + residual_tmp210*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp132*residual_tmp7 - residual_tmp20*residual_tmp227 - residual_tmp228) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp226 - residual_tmp230) + residual_tmp210*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp19*residual_tmp7 - residual_tmp20*residual_tmp205) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp207) + residual_tmp12*residual_tmp210) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp173*residual_tmp7 - residual_tmp194 - residual_tmp20*residual_tmp224 - residual_tmp225) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp222) + residual_tmp166*residual_tmp210) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp148*residual_tmp7 - residual_tmp20*residual_tmp218 + residual_tmp220) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp216) + residual_tmp210*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp20*residual_tmp211 + residual_tmp7*residual_tmp74) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp212) + residual_tmp20*residual_tmp210) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp164*residual_tmp7 - residual_tmp20*residual_tmp232 - residual_tmp233) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp231 - residual_tmp229*u0_grad_2) + residual_tmp210*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp187*residual_tmp7 - residual_tmp20*residual_tmp239 + residual_tmp240) + residual_tmp235*residual_tmp238 + residual_tmp241) + residual_tmp178*residual_tmp210);
            const scalar_t grad_coeff1_1 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp213*residual_tmp43 + residual_tmp237 + residual_tmp44*residual_tmp87) - residual_tmp214*residual_tmp242 + residual_tmp243*residual_tmp6) + residual_tmp244*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp113*residual_tmp44 + residual_tmp236*residual_tmp43) - residual_tmp234*residual_tmp242) + residual_tmp244*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp132*residual_tmp44 + residual_tmp227*residual_tmp43 - residual_tmp246) - residual_tmp226*residual_tmp242 - residual_tmp243*u2_grad_0) + residual_tmp244*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp116 + residual_tmp19*residual_tmp44 + residual_tmp205*residual_tmp43 - residual_tmp225) - residual_tmp207*residual_tmp242) + residual_tmp12*residual_tmp244) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp173*residual_tmp44 + residual_tmp224*residual_tmp43) - residual_tmp222*residual_tmp242) + residual_tmp166*residual_tmp244) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp148*residual_tmp44 + residual_tmp192 + residual_tmp218*residual_tmp43 + residual_tmp245) - residual_tmp216*residual_tmp242) + residual_tmp244*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp211*residual_tmp43 - residual_tmp233 + residual_tmp44*residual_tmp74) - residual_tmp212*residual_tmp242 - residual_tmp243*u0_grad_2) + residual_tmp20*residual_tmp244) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp164*residual_tmp44 + residual_tmp232*residual_tmp43) - residual_tmp231*residual_tmp242) + residual_tmp244*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp187*residual_tmp44 + residual_tmp239*residual_tmp43 + residual_tmp247) + residual_tmp141*residual_tmp243 - residual_tmp238*residual_tmp242) + residual_tmp178*residual_tmp244);
            const scalar_t grad_coeff1_2 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp213*residual_tmp47 + residual_tmp228 - residual_tmp45*residual_tmp87) + residual_tmp214*residual_tmp250 + residual_tmp230) + residual_tmp248*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp113*residual_tmp45 + residual_tmp236*residual_tmp47 - residual_tmp246) - residual_tmp229*u2_grad_0 + ((scalar_t(1) / scalar_t(3)))*residual_tmp234*residual_tmp50) + residual_tmp248*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp132*residual_tmp45 + residual_tmp227*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp226*residual_tmp50) + residual_tmp248*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp19*residual_tmp45 + residual_tmp205*residual_tmp47 - residual_tmp220) + ((scalar_t(1) / scalar_t(3)))*residual_tmp207*residual_tmp50) + residual_tmp12*residual_tmp248) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp173*residual_tmp45 - residual_tmp191 + residual_tmp224*residual_tmp47 - residual_tmp249) + ((scalar_t(1) / scalar_t(3)))*residual_tmp222*residual_tmp50) + residual_tmp166*residual_tmp248) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp148*residual_tmp45 + residual_tmp218*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp216*residual_tmp50) + residual_tmp248*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp211*residual_tmp47 - residual_tmp240 - residual_tmp45*residual_tmp74) + ((scalar_t(1) / scalar_t(3)))*residual_tmp212*residual_tmp50 - residual_tmp241) + residual_tmp20*residual_tmp248) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp164*residual_tmp45 + residual_tmp232*residual_tmp47 + residual_tmp247) + residual_tmp141*residual_tmp229 + residual_tmp231*residual_tmp250) + residual_tmp248*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp187*residual_tmp45 + residual_tmp239*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp238*residual_tmp50) + residual_tmp178*residual_tmp248);
            const scalar_t grad_coeff2_0 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp213 + residual_tmp7*residual_tmp85) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp255) + residual_tmp253*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp108*residual_tmp7 - residual_tmp12*residual_tmp236 - residual_tmp260) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp259 - residual_tmp262) + residual_tmp253*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp227 + residual_tmp137*residual_tmp7 + residual_tmp267) + residual_tmp261*residual_tmp46 + residual_tmp265*residual_tmp266) + residual_tmp253*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp205 + residual_tmp28*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp254) + residual_tmp12*residual_tmp253) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp224 + residual_tmp176*residual_tmp7 + residual_tmp269) + residual_tmp266*residual_tmp268 + residual_tmp270) + residual_tmp166*residual_tmp253) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp218 + residual_tmp152*residual_tmp7 - residual_tmp264) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp263 - residual_tmp261*u0_grad_1) + residual_tmp253*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp211 + residual_tmp7*residual_tmp72) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp251) + residual_tmp20*residual_tmp253) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp232 + residual_tmp161*residual_tmp7 + residual_tmp257) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp256) + residual_tmp253*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp239 + residual_tmp185*residual_tmp7 - residual_tmp202 - residual_tmp219) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp258) + residual_tmp178*residual_tmp253);
            const scalar_t grad_coeff2_1 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp213*residual_tmp51 + residual_tmp260 - residual_tmp44*residual_tmp85) + residual_tmp255*residual_tmp273 + residual_tmp262) + residual_tmp271*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp108*residual_tmp44 + residual_tmp236*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp259*residual_tmp43) + residual_tmp271*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp137*residual_tmp44 + residual_tmp227*residual_tmp51 - residual_tmp272) - residual_tmp261*u1_grad_0 + ((scalar_t(1) / scalar_t(3)))*residual_tmp265*residual_tmp43) + residual_tmp271*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp205*residual_tmp51 - residual_tmp269 - residual_tmp28*residual_tmp44) + ((scalar_t(1) / scalar_t(3)))*residual_tmp254*residual_tmp43 - residual_tmp270) + residual_tmp12*residual_tmp271) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp176*residual_tmp44 + residual_tmp224*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp268*residual_tmp43) + residual_tmp166*residual_tmp271) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp152*residual_tmp44 + residual_tmp218*residual_tmp51 + residual_tmp274) + residual_tmp141*residual_tmp261 + residual_tmp263*residual_tmp273) + residual_tmp271*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp211*residual_tmp51 - residual_tmp257 - residual_tmp44*residual_tmp72) + ((scalar_t(1) / scalar_t(3)))*residual_tmp251*residual_tmp43) + residual_tmp20*residual_tmp271) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp161*residual_tmp44 + residual_tmp232*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp256*residual_tmp43) + residual_tmp271*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp185*residual_tmp44 - residual_tmp190 + residual_tmp239*residual_tmp51 - residual_tmp249) + ((scalar_t(1) / scalar_t(3)))*residual_tmp258*residual_tmp43) + residual_tmp178*residual_tmp271);
            const scalar_t grad_coeff2_2 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp213*residual_tmp50 + residual_tmp267 + residual_tmp45*residual_tmp85) - residual_tmp255*residual_tmp275 + residual_tmp276*residual_tmp46) + residual_tmp277*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp108*residual_tmp45 + residual_tmp236*residual_tmp50 - residual_tmp272) - residual_tmp259*residual_tmp275 - residual_tmp276*u1_grad_0) + residual_tmp277*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp137*residual_tmp45 + residual_tmp227*residual_tmp50) - residual_tmp265*residual_tmp275) + residual_tmp277*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp205*residual_tmp50 - residual_tmp264 + residual_tmp28*residual_tmp45) - residual_tmp254*residual_tmp275 - residual_tmp276*u0_grad_1) + residual_tmp12*residual_tmp277) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp176*residual_tmp45 + residual_tmp224*residual_tmp50 + residual_tmp274) + residual_tmp141*residual_tmp276 - residual_tmp268*residual_tmp275) + residual_tmp166*residual_tmp277) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp152*residual_tmp45 + residual_tmp218*residual_tmp50) - residual_tmp263*residual_tmp275) + residual_tmp277*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp140 + residual_tmp211*residual_tmp50 - residual_tmp219 + residual_tmp45*residual_tmp72) - residual_tmp251*residual_tmp275) + residual_tmp20*residual_tmp277) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp161*residual_tmp45 - residual_tmp190 + residual_tmp232*residual_tmp50 + residual_tmp245) - residual_tmp256*residual_tmp275) + residual_tmp277*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp185*residual_tmp45 + residual_tmp239*residual_tmp50) - residual_tmp258*residual_tmp275) + residual_tmp178*residual_tmp277);
            const scalar_t grad_coeff0_0_value = grad_coeff0_0;
            const scalar_t grad_coeff0_1_value = grad_coeff0_1;
            const scalar_t grad_coeff0_2_value = grad_coeff0_2;
            const scalar_t grad_coeff1_0_value = grad_coeff1_0;
            const scalar_t grad_coeff1_1_value = grad_coeff1_1;
            const scalar_t grad_coeff1_2_value = grad_coeff1_2;
            const scalar_t grad_coeff2_0_value = grad_coeff2_0;
            const scalar_t grad_coeff2_1_value = grad_coeff2_1;
            const scalar_t grad_coeff2_2_value = grad_coeff2_2;
            const scalar_t test0_grad0 = (-(adj0) - adj3 - adj6) / det;
            const scalar_t test0_grad1 = (-(adj1) - adj4 - adj7) / det;
            const scalar_t test0_grad2 = (-(adj2) - adj5 - adj8) / det;
            const scalar_t test1_grad0 = (adj0) / det;
            const scalar_t test1_grad1 = (adj1) / det;
            const scalar_t test1_grad2 = (adj2) / det;
            const scalar_t test2_grad0 = (adj3) / det;
            const scalar_t test2_grad1 = (adj4) / det;
            const scalar_t test2_grad2 = (adj5) / det;
            const scalar_t test3_grad0 = (adj6) / det;
            const scalar_t test3_grad1 = (adj7) / det;
            const scalar_t test3_grad2 = (adj8) / det;
            output[0][lane] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1 + grad_coeff0_2_value * test0_grad2);
            output[1][lane] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1 + grad_coeff1_2_value * test0_grad2);
            output[2][lane] += q_weight[q] * det * (grad_coeff2_0_value * test0_grad0 + grad_coeff2_1_value * test0_grad1 + grad_coeff2_2_value * test0_grad2);
            output[3][lane] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1 + grad_coeff0_2_value * test1_grad2);
            output[4][lane] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1 + grad_coeff1_2_value * test1_grad2);
            output[5][lane] += q_weight[q] * det * (grad_coeff2_0_value * test1_grad0 + grad_coeff2_1_value * test1_grad1 + grad_coeff2_2_value * test1_grad2);
            output[6][lane] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1 + grad_coeff0_2_value * test2_grad2);
            output[7][lane] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1 + grad_coeff1_2_value * test2_grad2);
            output[8][lane] += q_weight[q] * det * (grad_coeff2_0_value * test2_grad0 + grad_coeff2_1_value * test2_grad1 + grad_coeff2_2_value * test2_grad2);
            output[9][lane] += q_weight[q] * det * (grad_coeff0_0_value * test3_grad0 + grad_coeff0_1_value * test3_grad1 + grad_coeff0_2_value * test3_grad2);
            output[10][lane] += q_weight[q] * det * (grad_coeff1_0_value * test3_grad0 + grad_coeff1_1_value * test3_grad1 + grad_coeff1_2_value * test3_grad2);
            output[11][lane] += q_weight[q] * det * (grad_coeff2_0_value * test3_grad0 + grad_coeff2_1_value * test3_grad1 + grad_coeff2_2_value * test3_grad2);
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_viscous_d3_simplex_tet4_jacobian_action_block_contiguous(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT shape,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT grad_ref_z,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t current[3 * N_SHAPE][VECTOR_SIZE],
        const scalar_t previous[3 * N_SHAPE][VECTOR_SIZE],
        const scalar_t direction[3 * N_SHAPE][VECTOR_SIZE],
        const scalar_t eta_b,
        const scalar_t eta_s,
        const scalar_t newmark_velocity_alpha,
        scalar_t output[3 * N_SHAPE][VECTOR_SIZE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 3;
    for (int q = 0; q < N_QP; ++q) {
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
            const scalar_t u0_grad_0_ref = -(current[0][lane]) + current[3][lane];
            const scalar_t u0_grad_1_ref = -(current[0][lane]) + current[6][lane];
            const scalar_t u0_grad_2_ref = -(current[0][lane]) + current[9][lane];
            const scalar_t u0_grad_0 = (u0_grad_0_ref * adj0 + u0_grad_1_ref * adj3 + u0_grad_2_ref * adj6) / det;
            const scalar_t u0_grad_1 = (u0_grad_0_ref * adj1 + u0_grad_1_ref * adj4 + u0_grad_2_ref * adj7) / det;
            const scalar_t u0_grad_2 = (u0_grad_0_ref * adj2 + u0_grad_1_ref * adj5 + u0_grad_2_ref * adj8) / det;
            const scalar_t u0_old_grad_0_ref = -(previous[0][lane]) + previous[3][lane];
            const scalar_t u0_old_grad_1_ref = -(previous[0][lane]) + previous[6][lane];
            const scalar_t u0_old_grad_2_ref = -(previous[0][lane]) + previous[9][lane];
            const scalar_t u0_old_grad_0 = (u0_old_grad_0_ref * adj0 + u0_old_grad_1_ref * adj3 + u0_old_grad_2_ref * adj6) / det;
            const scalar_t u0_old_grad_1 = (u0_old_grad_0_ref * adj1 + u0_old_grad_1_ref * adj4 + u0_old_grad_2_ref * adj7) / det;
            const scalar_t u0_old_grad_2 = (u0_old_grad_0_ref * adj2 + u0_old_grad_1_ref * adj5 + u0_old_grad_2_ref * adj8) / det;
            const scalar_t u0_direction_grad_0_ref = -(direction[0][lane]) + direction[3][lane];
            const scalar_t u0_direction_grad_1_ref = -(direction[0][lane]) + direction[6][lane];
            const scalar_t u0_direction_grad_2_ref = -(direction[0][lane]) + direction[9][lane];
            const scalar_t u0_direction_grad_0 = (u0_direction_grad_0_ref * adj0 + u0_direction_grad_1_ref * adj3 + u0_direction_grad_2_ref * adj6) / det;
            const scalar_t u0_direction_grad_1 = (u0_direction_grad_0_ref * adj1 + u0_direction_grad_1_ref * adj4 + u0_direction_grad_2_ref * adj7) / det;
            const scalar_t u0_direction_grad_2 = (u0_direction_grad_0_ref * adj2 + u0_direction_grad_1_ref * adj5 + u0_direction_grad_2_ref * adj8) / det;
            const scalar_t u1_grad_0_ref = -(current[1][lane]) + current[4][lane];
            const scalar_t u1_grad_1_ref = -(current[1][lane]) + current[7][lane];
            const scalar_t u1_grad_2_ref = -(current[1][lane]) + current[10][lane];
            const scalar_t u1_grad_0 = (u1_grad_0_ref * adj0 + u1_grad_1_ref * adj3 + u1_grad_2_ref * adj6) / det;
            const scalar_t u1_grad_1 = (u1_grad_0_ref * adj1 + u1_grad_1_ref * adj4 + u1_grad_2_ref * adj7) / det;
            const scalar_t u1_grad_2 = (u1_grad_0_ref * adj2 + u1_grad_1_ref * adj5 + u1_grad_2_ref * adj8) / det;
            const scalar_t u1_old_grad_0_ref = -(previous[1][lane]) + previous[4][lane];
            const scalar_t u1_old_grad_1_ref = -(previous[1][lane]) + previous[7][lane];
            const scalar_t u1_old_grad_2_ref = -(previous[1][lane]) + previous[10][lane];
            const scalar_t u1_old_grad_0 = (u1_old_grad_0_ref * adj0 + u1_old_grad_1_ref * adj3 + u1_old_grad_2_ref * adj6) / det;
            const scalar_t u1_old_grad_1 = (u1_old_grad_0_ref * adj1 + u1_old_grad_1_ref * adj4 + u1_old_grad_2_ref * adj7) / det;
            const scalar_t u1_old_grad_2 = (u1_old_grad_0_ref * adj2 + u1_old_grad_1_ref * adj5 + u1_old_grad_2_ref * adj8) / det;
            const scalar_t u1_direction_grad_0_ref = -(direction[1][lane]) + direction[4][lane];
            const scalar_t u1_direction_grad_1_ref = -(direction[1][lane]) + direction[7][lane];
            const scalar_t u1_direction_grad_2_ref = -(direction[1][lane]) + direction[10][lane];
            const scalar_t u1_direction_grad_0 = (u1_direction_grad_0_ref * adj0 + u1_direction_grad_1_ref * adj3 + u1_direction_grad_2_ref * adj6) / det;
            const scalar_t u1_direction_grad_1 = (u1_direction_grad_0_ref * adj1 + u1_direction_grad_1_ref * adj4 + u1_direction_grad_2_ref * adj7) / det;
            const scalar_t u1_direction_grad_2 = (u1_direction_grad_0_ref * adj2 + u1_direction_grad_1_ref * adj5 + u1_direction_grad_2_ref * adj8) / det;
            const scalar_t u2_grad_0_ref = -(current[2][lane]) + current[5][lane];
            const scalar_t u2_grad_1_ref = -(current[2][lane]) + current[8][lane];
            const scalar_t u2_grad_2_ref = -(current[2][lane]) + current[11][lane];
            const scalar_t u2_grad_0 = (u2_grad_0_ref * adj0 + u2_grad_1_ref * adj3 + u2_grad_2_ref * adj6) / det;
            const scalar_t u2_grad_1 = (u2_grad_0_ref * adj1 + u2_grad_1_ref * adj4 + u2_grad_2_ref * adj7) / det;
            const scalar_t u2_grad_2 = (u2_grad_0_ref * adj2 + u2_grad_1_ref * adj5 + u2_grad_2_ref * adj8) / det;
            const scalar_t u2_old_grad_0_ref = -(previous[2][lane]) + previous[5][lane];
            const scalar_t u2_old_grad_1_ref = -(previous[2][lane]) + previous[8][lane];
            const scalar_t u2_old_grad_2_ref = -(previous[2][lane]) + previous[11][lane];
            const scalar_t u2_old_grad_0 = (u2_old_grad_0_ref * adj0 + u2_old_grad_1_ref * adj3 + u2_old_grad_2_ref * adj6) / det;
            const scalar_t u2_old_grad_1 = (u2_old_grad_0_ref * adj1 + u2_old_grad_1_ref * adj4 + u2_old_grad_2_ref * adj7) / det;
            const scalar_t u2_old_grad_2 = (u2_old_grad_0_ref * adj2 + u2_old_grad_1_ref * adj5 + u2_old_grad_2_ref * adj8) / det;
            const scalar_t u2_direction_grad_0_ref = -(direction[2][lane]) + direction[5][lane];
            const scalar_t u2_direction_grad_1_ref = -(direction[2][lane]) + direction[8][lane];
            const scalar_t u2_direction_grad_2_ref = -(direction[2][lane]) + direction[11][lane];
            const scalar_t u2_direction_grad_0 = (u2_direction_grad_0_ref * adj0 + u2_direction_grad_1_ref * adj3 + u2_direction_grad_2_ref * adj6) / det;
            const scalar_t u2_direction_grad_1 = (u2_direction_grad_0_ref * adj1 + u2_direction_grad_1_ref * adj4 + u2_direction_grad_2_ref * adj7) / det;
            const scalar_t u2_direction_grad_2 = (u2_direction_grad_0_ref * adj2 + u2_direction_grad_1_ref * adj5 + u2_direction_grad_2_ref * adj8) / det;
            const scalar_t residual_tmp0 = u0_grad_0*u1_grad_1;
            const scalar_t residual_tmp1 = u0_grad_1*u1_grad_2;
            const scalar_t residual_tmp2 = u0_grad_2*u2_grad_1;
            const scalar_t residual_tmp3 = u1_grad_2*u2_grad_1;
            const scalar_t residual_tmp4 = u0_grad_1*u1_grad_0;
            const scalar_t residual_tmp5 = u0_grad_2*u2_grad_0;
            const scalar_t residual_tmp6 = u2_grad_2 + scalar_t(1);
            const scalar_t residual_tmp7 = -residual_tmp3 + residual_tmp6 + u1_grad_1*u2_grad_2 + u1_grad_1;
            const scalar_t residual_tmp8 = -residual_tmp5 + u0_grad_0*u2_grad_2 + u0_grad_0;
            const scalar_t residual_tmp9 = residual_tmp0 - residual_tmp4;
            const scalar_t residual_tmp10 = residual_tmp0*u2_grad_2 + residual_tmp1*u2_grad_0 + residual_tmp2*u1_grad_0 - residual_tmp3*u0_grad_0 - residual_tmp4*u2_grad_2 - residual_tmp5*u1_grad_1 + residual_tmp7 + residual_tmp8 + residual_tmp9;
            const scalar_t residual_tmp11 = pow_m1(residual_tmp10);
            const scalar_t residual_tmp12 = -residual_tmp2 + u0_grad_1*u2_grad_2 + u0_grad_1;
            const scalar_t residual_tmp13 = newmark_velocity_alpha*residual_tmp7;
            const scalar_t residual_tmp14 = newmark_velocity_alpha*u1_grad_2 + u1_old_grad_2;
            const scalar_t residual_tmp15 = residual_tmp14*u2_grad_1;
            const scalar_t residual_tmp16 = newmark_velocity_alpha*u1_grad_1 + u1_old_grad_1;
            const scalar_t residual_tmp17 = residual_tmp16*residual_tmp6;
            const scalar_t residual_tmp18 = residual_tmp15 - residual_tmp17;
            const scalar_t residual_tmp19 = -residual_tmp13 - residual_tmp18;
            const scalar_t residual_tmp20 = -residual_tmp1 + u0_grad_2*u1_grad_1 + u0_grad_2;
            const scalar_t residual_tmp21 = newmark_velocity_alpha*u2_grad_1 + u2_old_grad_1;
            const scalar_t residual_tmp22 = residual_tmp21*residual_tmp6;
            const scalar_t residual_tmp23 = newmark_velocity_alpha*u2_grad_2 + u2_old_grad_2;
            const scalar_t residual_tmp24 = residual_tmp23*u2_grad_1;
            const scalar_t residual_tmp25 = newmark_velocity_alpha*u0_grad_2 + u0_old_grad_2;
            const scalar_t residual_tmp26 = newmark_velocity_alpha*u0_grad_1 + u0_old_grad_1;
            const scalar_t residual_tmp27 = residual_tmp25*u0_grad_1 - residual_tmp26*u0_grad_2;
            const scalar_t residual_tmp28 = residual_tmp22 - residual_tmp24 + residual_tmp27;
            const scalar_t residual_tmp29 = newmark_velocity_alpha*residual_tmp12;
            const scalar_t residual_tmp30 = residual_tmp23*u0_grad_1;
            const scalar_t residual_tmp31 = residual_tmp21*u0_grad_2;
            const scalar_t residual_tmp32 = residual_tmp29 + residual_tmp30 - residual_tmp31;
            const scalar_t residual_tmp33 = residual_tmp26*residual_tmp6;
            const scalar_t residual_tmp34 = residual_tmp25*u2_grad_1;
            const scalar_t residual_tmp35 = residual_tmp33 - residual_tmp34;
            const scalar_t residual_tmp36 = scalar_t(3)*eta_b;
            const scalar_t residual_tmp37 = residual_tmp36*(residual_tmp32 + residual_tmp35);
            const scalar_t residual_tmp38 = scalar_t(2)*eta_s;
            const scalar_t residual_tmp39 = residual_tmp37 + residual_tmp38*(scalar_t(2)*residual_tmp26*residual_tmp6 - residual_tmp32 - scalar_t(2)*residual_tmp34);
            const scalar_t residual_tmp40 = ((scalar_t(1) / scalar_t(3)))*residual_tmp7;
            const scalar_t residual_tmp41 = pow_m2(residual_tmp10);
            const scalar_t residual_tmp42 = newmark_velocity_alpha*u0_grad_0 + u0_old_grad_0;
            const scalar_t residual_tmp43 = u0_grad_0*u1_grad_2 - u0_grad_2*u1_grad_0 + u1_grad_2;
            const scalar_t residual_tmp44 = u1_grad_0*u2_grad_2 + u1_grad_0 - u1_grad_2*u2_grad_0;
            const scalar_t residual_tmp45 = -u1_grad_0*u2_grad_1 + u1_grad_1*u2_grad_0 + u2_grad_0;
            const scalar_t residual_tmp46 = u1_grad_1 + scalar_t(1);
            const scalar_t residual_tmp47 = residual_tmp46 + residual_tmp9 + u0_grad_0;
            const scalar_t residual_tmp48 = newmark_velocity_alpha*u2_grad_0 + u2_old_grad_0;
            const scalar_t residual_tmp49 = residual_tmp20*residual_tmp42 + residual_tmp21*residual_tmp44 + residual_tmp23*residual_tmp45 - residual_tmp25*residual_tmp47 + residual_tmp26*residual_tmp43 - residual_tmp48*residual_tmp7;
            const scalar_t residual_tmp50 = u0_grad_0*u2_grad_1 - u0_grad_1*u2_grad_0 + u2_grad_1;
            const scalar_t residual_tmp51 = residual_tmp6 + residual_tmp8;
            const scalar_t residual_tmp52 = newmark_velocity_alpha*u1_grad_0 + u1_old_grad_0;
            const scalar_t residual_tmp53 = residual_tmp12*residual_tmp42 + residual_tmp14*residual_tmp45 + residual_tmp16*residual_tmp44 + residual_tmp25*residual_tmp50 - residual_tmp26*residual_tmp51 - residual_tmp52*residual_tmp7;
            const scalar_t residual_tmp54 = residual_tmp12*residual_tmp52;
            const scalar_t residual_tmp55 = residual_tmp14*residual_tmp50;
            const scalar_t residual_tmp56 = residual_tmp20*residual_tmp48;
            const scalar_t residual_tmp57 = residual_tmp21*residual_tmp43;
            const scalar_t residual_tmp58 = residual_tmp16*residual_tmp51;
            const scalar_t residual_tmp59 = -residual_tmp58;
            const scalar_t residual_tmp60 = residual_tmp23*residual_tmp47;
            const scalar_t residual_tmp61 = -residual_tmp60;
            const scalar_t residual_tmp62 = residual_tmp54 + residual_tmp55 + residual_tmp56 + residual_tmp57 + residual_tmp59 + residual_tmp61;
            const scalar_t residual_tmp63 = residual_tmp42*residual_tmp7;
            const scalar_t residual_tmp64 = residual_tmp25*residual_tmp45 + residual_tmp26*residual_tmp44 - residual_tmp63;
            const scalar_t residual_tmp65 = residual_tmp36*(residual_tmp62 + residual_tmp64);
            const scalar_t residual_tmp66 = residual_tmp38*(scalar_t(2)*residual_tmp25*residual_tmp45 + scalar_t(2)*residual_tmp26*residual_tmp44 - residual_tmp62 - scalar_t(2)*residual_tmp63) + residual_tmp65;
            const scalar_t residual_tmp67 = -residual_tmp66;
            const scalar_t residual_tmp68 = residual_tmp41*(eta_s*(residual_tmp12*residual_tmp53 + residual_tmp20*residual_tmp49) + residual_tmp40*residual_tmp67);
            const scalar_t residual_tmp69 = residual_tmp21*u1_grad_2;
            const scalar_t residual_tmp70 = residual_tmp23*residual_tmp46;
            const scalar_t residual_tmp71 = residual_tmp69 - residual_tmp70;
            const scalar_t residual_tmp72 = -residual_tmp13 - residual_tmp71;
            const scalar_t residual_tmp73 = -residual_tmp14*residual_tmp46 + residual_tmp16*u1_grad_2;
            const scalar_t residual_tmp74 = -residual_tmp27 - residual_tmp73;
            const scalar_t residual_tmp75 = newmark_velocity_alpha*residual_tmp20;
            const scalar_t residual_tmp76 = residual_tmp16*u0_grad_2;
            const scalar_t residual_tmp77 = residual_tmp14*u0_grad_1;
            const scalar_t residual_tmp78 = residual_tmp75 + residual_tmp76 - residual_tmp77;
            const scalar_t residual_tmp79 = residual_tmp25*residual_tmp46;
            const scalar_t residual_tmp80 = residual_tmp26*u1_grad_2;
            const scalar_t residual_tmp81 = residual_tmp79 - residual_tmp80;
            const scalar_t residual_tmp82 = residual_tmp36*(residual_tmp78 + residual_tmp81);
            const scalar_t residual_tmp83 = residual_tmp38*(scalar_t(2)*residual_tmp25*residual_tmp46 - residual_tmp78 - scalar_t(2)*residual_tmp80) + residual_tmp82;
            const scalar_t residual_tmp84 = -residual_tmp79 + residual_tmp80;
            const scalar_t residual_tmp85 = residual_tmp75 + residual_tmp84;
            const scalar_t residual_tmp86 = -residual_tmp33 + residual_tmp34;
            const scalar_t residual_tmp87 = residual_tmp29 + residual_tmp86;
            const scalar_t residual_tmp88 = residual_tmp13 - residual_tmp69 + residual_tmp70;
            const scalar_t residual_tmp89 = -residual_tmp15 + residual_tmp17;
            const scalar_t residual_tmp90 = residual_tmp36*(-residual_tmp88 - residual_tmp89);
            const scalar_t residual_tmp91 = residual_tmp38*(-scalar_t(2)*residual_tmp13 - residual_tmp18 - residual_tmp71) + residual_tmp90;
            const scalar_t residual_tmp92 = -residual_tmp7;
            const scalar_t residual_tmp93 = newmark_velocity_alpha*residual_tmp44;
            const scalar_t residual_tmp94 = residual_tmp23*u1_grad_0;
            const scalar_t residual_tmp95 = residual_tmp48*u1_grad_2;
            const scalar_t residual_tmp96 = residual_tmp93 + residual_tmp94 - residual_tmp95;
            const scalar_t residual_tmp97 = residual_tmp52*residual_tmp6;
            const scalar_t residual_tmp98 = residual_tmp14*u2_grad_0;
            const scalar_t residual_tmp99 = residual_tmp97 - residual_tmp98;
            const scalar_t residual_tmp100 = residual_tmp36*(residual_tmp96 + residual_tmp99);
            const scalar_t residual_tmp101 = -residual_tmp97 + residual_tmp98;
            const scalar_t residual_tmp102 = -residual_tmp94 + residual_tmp95;
            const scalar_t residual_tmp103 = residual_tmp100 + residual_tmp38*(residual_tmp101 + residual_tmp102 + scalar_t(2)*residual_tmp93);
            const scalar_t residual_tmp104 = newmark_velocity_alpha*residual_tmp43;
            const scalar_t residual_tmp105 = residual_tmp25*u1_grad_0;
            const scalar_t residual_tmp106 = residual_tmp42*u1_grad_2;
            const scalar_t residual_tmp107 = residual_tmp105 - residual_tmp106;
            const scalar_t residual_tmp108 = residual_tmp104 + residual_tmp107;
            const scalar_t residual_tmp109 = newmark_velocity_alpha*residual_tmp51;
            const scalar_t residual_tmp110 = residual_tmp25*u2_grad_0;
            const scalar_t residual_tmp111 = residual_tmp42*residual_tmp6;
            const scalar_t residual_tmp112 = residual_tmp110 - residual_tmp111;
            const scalar_t residual_tmp113 = -residual_tmp109 - residual_tmp112;
            const scalar_t residual_tmp114 = residual_tmp49*u1_grad_2;
            const scalar_t residual_tmp115 = -residual_tmp114;
            const scalar_t residual_tmp116 = residual_tmp53*residual_tmp6;
            const scalar_t residual_tmp117 = newmark_velocity_alpha*residual_tmp45;
            const scalar_t residual_tmp118 = residual_tmp46*residual_tmp48;
            const scalar_t residual_tmp119 = residual_tmp21*u1_grad_0;
            const scalar_t residual_tmp120 = residual_tmp117 + residual_tmp118 - residual_tmp119;
            const scalar_t residual_tmp121 = residual_tmp16*u2_grad_0;
            const scalar_t residual_tmp122 = residual_tmp52*u2_grad_1;
            const scalar_t residual_tmp123 = residual_tmp121 - residual_tmp122;
            const scalar_t residual_tmp124 = residual_tmp36*(residual_tmp120 + residual_tmp123);
            const scalar_t residual_tmp125 = -residual_tmp121 + residual_tmp122;
            const scalar_t residual_tmp126 = -residual_tmp118 + residual_tmp119;
            const scalar_t residual_tmp127 = residual_tmp124 + residual_tmp38*(scalar_t(2)*residual_tmp117 + residual_tmp125 + residual_tmp126);
            const scalar_t residual_tmp128 = newmark_velocity_alpha*residual_tmp50;
            const scalar_t residual_tmp129 = residual_tmp26*u2_grad_0;
            const scalar_t residual_tmp130 = residual_tmp42*u2_grad_1;
            const scalar_t residual_tmp131 = residual_tmp129 - residual_tmp130;
            const scalar_t residual_tmp132 = residual_tmp128 + residual_tmp131;
            const scalar_t residual_tmp133 = newmark_velocity_alpha*residual_tmp47;
            const scalar_t residual_tmp134 = residual_tmp26*u1_grad_0;
            const scalar_t residual_tmp135 = residual_tmp42*residual_tmp46;
            const scalar_t residual_tmp136 = residual_tmp134 - residual_tmp135;
            const scalar_t residual_tmp137 = -residual_tmp133 - residual_tmp136;
            const scalar_t residual_tmp138 = residual_tmp53*u2_grad_1;
            const scalar_t residual_tmp139 = -residual_tmp138;
            const scalar_t residual_tmp140 = residual_tmp46*residual_tmp49;
            const scalar_t residual_tmp141 = u0_grad_0 + scalar_t(1);
            const scalar_t residual_tmp142 = residual_tmp141*residual_tmp21;
            const scalar_t residual_tmp143 = residual_tmp48*u0_grad_1;
            const scalar_t residual_tmp144 = residual_tmp128 + residual_tmp142 - residual_tmp143;
            const scalar_t residual_tmp145 = -residual_tmp129 + residual_tmp130;
            const scalar_t residual_tmp146 = residual_tmp36*(residual_tmp144 + residual_tmp145);
            const scalar_t residual_tmp147 = residual_tmp146 + residual_tmp38*(-scalar_t(2)*residual_tmp129 - residual_tmp144 + scalar_t(2)*residual_tmp42*u2_grad_1);
            const scalar_t residual_tmp148 = residual_tmp117 + residual_tmp125;
            const scalar_t residual_tmp149 = residual_tmp21*u2_grad_0;
            const scalar_t residual_tmp150 = residual_tmp48*u2_grad_1;
            const scalar_t residual_tmp151 = -residual_tmp141*residual_tmp26 + residual_tmp42*u0_grad_1;
            const scalar_t residual_tmp152 = -residual_tmp149 + residual_tmp150 - residual_tmp151;
            const scalar_t residual_tmp153 = residual_tmp49*u0_grad_1;
            const scalar_t residual_tmp154 = ((scalar_t(1) / scalar_t(3)))*residual_tmp67;
            const scalar_t residual_tmp155 = residual_tmp14*residual_tmp141;
            const scalar_t residual_tmp156 = residual_tmp52*u0_grad_2;
            const scalar_t residual_tmp157 = residual_tmp104 + residual_tmp155 - residual_tmp156;
            const scalar_t residual_tmp158 = -residual_tmp105 + residual_tmp106;
            const scalar_t residual_tmp159 = residual_tmp36*(residual_tmp157 + residual_tmp158);
            const scalar_t residual_tmp160 = residual_tmp159 + residual_tmp38*(-scalar_t(2)*residual_tmp105 - residual_tmp157 + scalar_t(2)*residual_tmp42*u1_grad_2);
            const scalar_t residual_tmp161 = residual_tmp102 + residual_tmp93;
            const scalar_t residual_tmp162 = -residual_tmp141*residual_tmp25 + residual_tmp42*u0_grad_2;
            const scalar_t residual_tmp163 = residual_tmp14*u1_grad_0 - residual_tmp52*u1_grad_2;
            const scalar_t residual_tmp164 = -residual_tmp162 - residual_tmp163;
            const scalar_t residual_tmp165 = residual_tmp53*u0_grad_2;
            const scalar_t residual_tmp166 = -residual_tmp51;
            const scalar_t residual_tmp167 = residual_tmp141*residual_tmp23;
            const scalar_t residual_tmp168 = residual_tmp48*u0_grad_2;
            const scalar_t residual_tmp169 = residual_tmp109 + residual_tmp167 - residual_tmp168;
            const scalar_t residual_tmp170 = -residual_tmp110 + residual_tmp111;
            const scalar_t residual_tmp171 = residual_tmp36*(-residual_tmp169 - residual_tmp170);
            const scalar_t residual_tmp172 = residual_tmp171 + residual_tmp38*(scalar_t(2)*residual_tmp110 - scalar_t(2)*residual_tmp111 + residual_tmp169);
            const scalar_t residual_tmp173 = residual_tmp101 + residual_tmp93;
            const scalar_t residual_tmp174 = residual_tmp23*u2_grad_0;
            const scalar_t residual_tmp175 = residual_tmp48*residual_tmp6;
            const scalar_t residual_tmp176 = residual_tmp162 + residual_tmp174 - residual_tmp175;
            const scalar_t residual_tmp177 = residual_tmp49*u0_grad_2;
            const scalar_t residual_tmp178 = -residual_tmp47;
            const scalar_t residual_tmp179 = residual_tmp141*residual_tmp16;
            const scalar_t residual_tmp180 = residual_tmp52*u0_grad_1;
            const scalar_t residual_tmp181 = residual_tmp133 + residual_tmp179 - residual_tmp180;
            const scalar_t residual_tmp182 = -residual_tmp134 + residual_tmp135;
            const scalar_t residual_tmp183 = residual_tmp36*(-residual_tmp181 - residual_tmp182);
            const scalar_t residual_tmp184 = residual_tmp183 + residual_tmp38*(scalar_t(2)*residual_tmp134 - scalar_t(2)*residual_tmp135 + residual_tmp181);
            const scalar_t residual_tmp185 = residual_tmp117 + residual_tmp126;
            const scalar_t residual_tmp186 = residual_tmp16*u1_grad_0 - residual_tmp46*residual_tmp52;
            const scalar_t residual_tmp187 = residual_tmp151 + residual_tmp186;
            const scalar_t residual_tmp188 = residual_tmp53*u0_grad_1;
            const scalar_t residual_tmp189 = residual_tmp41*(-eta_s*(-residual_tmp43*residual_tmp49 + residual_tmp51*residual_tmp53) + ((scalar_t(1) / scalar_t(3)))*residual_tmp44*residual_tmp66);
            const scalar_t residual_tmp190 = residual_tmp49*u1_grad_0;
            const scalar_t residual_tmp191 = residual_tmp53*u2_grad_0;
            const scalar_t residual_tmp192 = -residual_tmp191;
            const scalar_t residual_tmp193 = residual_tmp190 + residual_tmp192;
            const scalar_t residual_tmp194 = -residual_tmp53*residual_tmp6;
            const scalar_t residual_tmp195 = residual_tmp141*residual_tmp49;
            const scalar_t residual_tmp196 = ((scalar_t(1) / scalar_t(3)))*residual_tmp66;
            const scalar_t residual_tmp197 = residual_tmp196*u2_grad_0;
            const scalar_t residual_tmp198 = ((scalar_t(1) / scalar_t(3)))*residual_tmp44;
            const scalar_t residual_tmp199 = residual_tmp141*residual_tmp53;
            const scalar_t residual_tmp200 = residual_tmp196*u1_grad_0;
            const scalar_t residual_tmp201 = residual_tmp41*(-eta_s*(residual_tmp47*residual_tmp49 - residual_tmp50*residual_tmp53) + ((scalar_t(1) / scalar_t(3)))*residual_tmp45*residual_tmp66);
            const scalar_t residual_tmp202 = -residual_tmp46*residual_tmp49;
            const scalar_t residual_tmp203 = ((scalar_t(1) / scalar_t(3)))*residual_tmp45;
            const scalar_t residual_tmp204 = -residual_tmp76 + residual_tmp77;
            const scalar_t residual_tmp205 = residual_tmp204 + residual_tmp75;
            const scalar_t residual_tmp206 = -residual_tmp30 + residual_tmp31;
            const scalar_t residual_tmp207 = residual_tmp37 + residual_tmp38*(residual_tmp206 + scalar_t(2)*residual_tmp29 + residual_tmp86);
            const scalar_t residual_tmp208 = residual_tmp12*residual_tmp48 - residual_tmp14*residual_tmp47 + residual_tmp16*residual_tmp43 + residual_tmp20*residual_tmp52 - residual_tmp21*residual_tmp51 + residual_tmp23*residual_tmp50;
            const scalar_t residual_tmp209 = residual_tmp38*(scalar_t(2)*residual_tmp12*residual_tmp52 + scalar_t(2)*residual_tmp14*residual_tmp50 - residual_tmp56 - residual_tmp57 - scalar_t(2)*residual_tmp58 - residual_tmp61 - residual_tmp64) + residual_tmp65;
            const scalar_t residual_tmp210 = residual_tmp41*(-eta_s*(-residual_tmp20*residual_tmp208 + residual_tmp53*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp209);
            const scalar_t residual_tmp211 = residual_tmp206 + residual_tmp29;
            const scalar_t residual_tmp212 = residual_tmp38*(scalar_t(2)*residual_tmp16*u0_grad_2 - residual_tmp75 - scalar_t(2)*residual_tmp77 - residual_tmp81) + residual_tmp82;
            const scalar_t residual_tmp213 = -residual_tmp22 + residual_tmp24 + residual_tmp73;
            const scalar_t residual_tmp214 = residual_tmp38*(scalar_t(2)*residual_tmp15 - scalar_t(2)*residual_tmp17 + residual_tmp88) + residual_tmp90;
            const scalar_t residual_tmp215 = -residual_tmp142 + residual_tmp143;
            const scalar_t residual_tmp216 = residual_tmp146 + residual_tmp38*(scalar_t(2)*residual_tmp128 + residual_tmp131 + residual_tmp215);
            const scalar_t residual_tmp217 = -residual_tmp179 + residual_tmp180;
            const scalar_t residual_tmp218 = -residual_tmp133 - residual_tmp217;
            const scalar_t residual_tmp219 = residual_tmp208*u0_grad_1;
            const scalar_t residual_tmp220 = residual_tmp139 + residual_tmp219;
            const scalar_t residual_tmp221 = -residual_tmp167 + residual_tmp168;
            const scalar_t residual_tmp222 = residual_tmp171 + residual_tmp38*(-scalar_t(2)*residual_tmp109 - residual_tmp112 - residual_tmp221);
            const scalar_t residual_tmp223 = -residual_tmp155 + residual_tmp156;
            const scalar_t residual_tmp224 = residual_tmp104 + residual_tmp223;
            const scalar_t residual_tmp225 = residual_tmp208*u0_grad_2;
            const scalar_t residual_tmp226 = residual_tmp124 + residual_tmp38*(-residual_tmp120 - scalar_t(2)*residual_tmp122 + scalar_t(2)*residual_tmp16*u2_grad_0);
            const scalar_t residual_tmp227 = residual_tmp149 - residual_tmp150 - residual_tmp186;
            const scalar_t residual_tmp228 = residual_tmp208*residual_tmp46;
            const scalar_t residual_tmp229 = ((scalar_t(1) / scalar_t(3)))*residual_tmp209;
            const scalar_t residual_tmp230 = residual_tmp229*u2_grad_1;
            const scalar_t residual_tmp231 = residual_tmp159 + residual_tmp38*(-residual_tmp104 + scalar_t(2)*residual_tmp14*residual_tmp141 - scalar_t(2)*residual_tmp156 - residual_tmp158);
            const scalar_t residual_tmp232 = -residual_tmp109 - residual_tmp221;
            const scalar_t residual_tmp233 = residual_tmp53*u1_grad_2;
            const scalar_t residual_tmp234 = residual_tmp100 + residual_tmp38*(scalar_t(2)*residual_tmp52*residual_tmp6 - residual_tmp96 - scalar_t(2)*residual_tmp98);
            const scalar_t residual_tmp235 = ((scalar_t(1) / scalar_t(3)))*residual_tmp12;
            const scalar_t residual_tmp236 = residual_tmp163 - residual_tmp174 + residual_tmp175;
            const scalar_t residual_tmp237 = residual_tmp208*u1_grad_2;
            const scalar_t residual_tmp238 = residual_tmp183 + residual_tmp38*(residual_tmp133 - scalar_t(2)*residual_tmp179 + scalar_t(2)*residual_tmp180 + residual_tmp182);
            const scalar_t residual_tmp239 = residual_tmp128 + residual_tmp215;
            const scalar_t residual_tmp240 = residual_tmp46*residual_tmp53;
            const scalar_t residual_tmp241 = residual_tmp229*u0_grad_1;
            const scalar_t residual_tmp242 = ((scalar_t(1) / scalar_t(3)))*residual_tmp51;
            const scalar_t residual_tmp243 = -(scalar_t(1) / scalar_t(3))*residual_tmp209;
            const scalar_t residual_tmp244 = residual_tmp41*(eta_s*(residual_tmp208*residual_tmp43 + residual_tmp44*residual_tmp53) + residual_tmp243*residual_tmp51);
            const scalar_t residual_tmp245 = residual_tmp141*residual_tmp208;
            const scalar_t residual_tmp246 = residual_tmp208*u1_grad_0;
            const scalar_t residual_tmp247 = residual_tmp53*u1_grad_0;
            const scalar_t residual_tmp248 = residual_tmp41*(-eta_s*(residual_tmp208*residual_tmp47 - residual_tmp45*residual_tmp53) + ((scalar_t(1) / scalar_t(3)))*residual_tmp209*residual_tmp50);
            const scalar_t residual_tmp249 = -residual_tmp141*residual_tmp208;
            const scalar_t residual_tmp250 = ((scalar_t(1) / scalar_t(3)))*residual_tmp50;
            const scalar_t residual_tmp251 = residual_tmp38*(residual_tmp204 + scalar_t(2)*residual_tmp75 + residual_tmp84) + residual_tmp82;
            const scalar_t residual_tmp252 = residual_tmp38*(scalar_t(2)*residual_tmp20*residual_tmp48 + scalar_t(2)*residual_tmp21*residual_tmp43 - residual_tmp54 - residual_tmp55 - residual_tmp59 - scalar_t(2)*residual_tmp60 - residual_tmp64) + residual_tmp65;
            const scalar_t residual_tmp253 = residual_tmp41*(-eta_s*(-residual_tmp12*residual_tmp208 + residual_tmp49*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp252);
            const scalar_t residual_tmp254 = residual_tmp37 + residual_tmp38*(scalar_t(2)*residual_tmp23*u0_grad_1 - residual_tmp29 - scalar_t(2)*residual_tmp31 - residual_tmp35);
            const scalar_t residual_tmp255 = residual_tmp38*(residual_tmp13 + scalar_t(2)*residual_tmp69 - scalar_t(2)*residual_tmp70 + residual_tmp89) + residual_tmp90;
            const scalar_t residual_tmp256 = residual_tmp159 + residual_tmp38*(scalar_t(2)*residual_tmp104 + residual_tmp107 + residual_tmp223);
            const scalar_t residual_tmp257 = residual_tmp115 + residual_tmp225;
            const scalar_t residual_tmp258 = residual_tmp183 + residual_tmp38*(-scalar_t(2)*residual_tmp133 - residual_tmp136 - residual_tmp217);
            const scalar_t residual_tmp259 = residual_tmp100 + residual_tmp38*(scalar_t(2)*residual_tmp23*u1_grad_0 - residual_tmp93 - scalar_t(2)*residual_tmp95 - residual_tmp99);
            const scalar_t residual_tmp260 = residual_tmp208*residual_tmp6;
            const scalar_t residual_tmp261 = ((scalar_t(1) / scalar_t(3)))*residual_tmp252;
            const scalar_t residual_tmp262 = residual_tmp261*u1_grad_2;
            const scalar_t residual_tmp263 = residual_tmp146 + residual_tmp38*(-residual_tmp128 + scalar_t(2)*residual_tmp141*residual_tmp21 - scalar_t(2)*residual_tmp143 - residual_tmp145);
            const scalar_t residual_tmp264 = residual_tmp49*u2_grad_1;
            const scalar_t residual_tmp265 = residual_tmp124 + residual_tmp38*(-residual_tmp117 - scalar_t(2)*residual_tmp119 - residual_tmp123 + scalar_t(2)*residual_tmp46*residual_tmp48);
            const scalar_t residual_tmp266 = ((scalar_t(1) / scalar_t(3)))*residual_tmp20;
            const scalar_t residual_tmp267 = residual_tmp208*u2_grad_1;
            const scalar_t residual_tmp268 = residual_tmp171 + residual_tmp38*(residual_tmp109 - scalar_t(2)*residual_tmp167 + scalar_t(2)*residual_tmp168 + residual_tmp170);
            const scalar_t residual_tmp269 = residual_tmp49*residual_tmp6;
            const scalar_t residual_tmp270 = residual_tmp261*u0_grad_2;
            const scalar_t residual_tmp271 = residual_tmp41*(-eta_s*(residual_tmp208*residual_tmp51 - residual_tmp44*residual_tmp49) + ((scalar_t(1) / scalar_t(3)))*residual_tmp252*residual_tmp43);
            const scalar_t residual_tmp272 = residual_tmp208*u2_grad_0;
            const scalar_t residual_tmp273 = ((scalar_t(1) / scalar_t(3)))*residual_tmp43;
            const scalar_t residual_tmp274 = residual_tmp49*u2_grad_0;
            const scalar_t residual_tmp275 = ((scalar_t(1) / scalar_t(3)))*residual_tmp47;
            const scalar_t residual_tmp276 = -(scalar_t(1) / scalar_t(3))*residual_tmp252;
            const scalar_t residual_tmp277 = residual_tmp41*(eta_s*(residual_tmp208*residual_tmp50 + residual_tmp45*residual_tmp49) + residual_tmp276*residual_tmp47);
            const scalar_t grad_coeff0_0 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp87 + residual_tmp20*residual_tmp85) - residual_tmp40*residual_tmp91) + residual_tmp68*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp108*residual_tmp20 + residual_tmp113*residual_tmp12 + residual_tmp115 + residual_tmp116) - residual_tmp103*residual_tmp40) + residual_tmp44*residual_tmp68) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp132 + residual_tmp137*residual_tmp20 + residual_tmp139 + residual_tmp140) - residual_tmp127*residual_tmp40) + residual_tmp45*residual_tmp68) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp19 + residual_tmp20*residual_tmp28) - residual_tmp39*residual_tmp40) + residual_tmp12*residual_tmp68) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp173 + residual_tmp176*residual_tmp20 + residual_tmp177) + residual_tmp154*residual_tmp6 - residual_tmp172*residual_tmp40) + residual_tmp166*residual_tmp68) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp148 + residual_tmp152*residual_tmp20 - residual_tmp153) - residual_tmp147*residual_tmp40 - residual_tmp154*u2_grad_1) + residual_tmp50*residual_tmp68) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp74 + residual_tmp20*residual_tmp72) - residual_tmp40*residual_tmp83) + residual_tmp20*residual_tmp68) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp164 + residual_tmp161*residual_tmp20 - residual_tmp165) - residual_tmp154*u1_grad_2 - residual_tmp160*residual_tmp40) + residual_tmp43*residual_tmp68) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp12*residual_tmp187 + residual_tmp185*residual_tmp20 + residual_tmp188) + residual_tmp154*residual_tmp46 - residual_tmp184*residual_tmp40) + residual_tmp178*residual_tmp68);
            const scalar_t grad_coeff0_1 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp114 - residual_tmp194 - residual_tmp43*residual_tmp85 + residual_tmp51*residual_tmp87) + ((scalar_t(1) / scalar_t(3)))*residual_tmp44*residual_tmp91) + residual_tmp189*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp108*residual_tmp43 + residual_tmp113*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp103*residual_tmp44) + residual_tmp189*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp132*residual_tmp51 - residual_tmp137*residual_tmp43 + residual_tmp193) + ((scalar_t(1) / scalar_t(3)))*residual_tmp127*residual_tmp44) + residual_tmp189*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp177 + residual_tmp19*residual_tmp51 - residual_tmp28*residual_tmp43) + residual_tmp196*residual_tmp6 + residual_tmp198*residual_tmp39) + residual_tmp12*residual_tmp189) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp173*residual_tmp51 - residual_tmp176*residual_tmp43) + ((scalar_t(1) / scalar_t(3)))*residual_tmp172*residual_tmp44) + residual_tmp166*residual_tmp189) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp148*residual_tmp51 - residual_tmp152*residual_tmp43 - residual_tmp195) + ((scalar_t(1) / scalar_t(3)))*residual_tmp147*residual_tmp44 - residual_tmp197) + residual_tmp189*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp165 - residual_tmp43*residual_tmp72 + residual_tmp51*residual_tmp74) - residual_tmp196*u1_grad_2 + ((scalar_t(1) / scalar_t(3)))*residual_tmp44*residual_tmp83) + residual_tmp189*residual_tmp20) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp161*residual_tmp43 + residual_tmp164*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp160*residual_tmp44) + residual_tmp189*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp185*residual_tmp43 + residual_tmp187*residual_tmp51 + residual_tmp199) + residual_tmp184*residual_tmp198 + residual_tmp200) + residual_tmp178*residual_tmp189);
            const scalar_t grad_coeff0_2 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp138 - residual_tmp202 + residual_tmp47*residual_tmp85 - residual_tmp50*residual_tmp87) + ((scalar_t(1) / scalar_t(3)))*residual_tmp45*residual_tmp91) + residual_tmp201*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp108*residual_tmp47 - residual_tmp113*residual_tmp50 - residual_tmp193) + ((scalar_t(1) / scalar_t(3)))*residual_tmp103*residual_tmp45) + residual_tmp201*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp132*residual_tmp50 + residual_tmp137*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp127*residual_tmp45) + residual_tmp201*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp153 - residual_tmp19*residual_tmp50 + residual_tmp28*residual_tmp47) - residual_tmp196*u2_grad_1 + ((scalar_t(1) / scalar_t(3)))*residual_tmp39*residual_tmp45) + residual_tmp12*residual_tmp201) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp173*residual_tmp50 + residual_tmp176*residual_tmp47 + residual_tmp195) + residual_tmp172*residual_tmp203 + residual_tmp197) + residual_tmp166*residual_tmp201) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp148*residual_tmp50 + residual_tmp152*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp147*residual_tmp45) + residual_tmp201*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp188 + residual_tmp47*residual_tmp72 - residual_tmp50*residual_tmp74) + residual_tmp196*residual_tmp46 + residual_tmp203*residual_tmp83) + residual_tmp20*residual_tmp201) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp161*residual_tmp47 - residual_tmp164*residual_tmp50 - residual_tmp199) + ((scalar_t(1) / scalar_t(3)))*residual_tmp160*residual_tmp45 - residual_tmp200) + residual_tmp201*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp185*residual_tmp47 - residual_tmp187*residual_tmp50) + ((scalar_t(1) / scalar_t(3)))*residual_tmp184*residual_tmp45) + residual_tmp178*residual_tmp201);
            const scalar_t grad_coeff1_0 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp20*residual_tmp213 + residual_tmp7*residual_tmp87) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp214) + residual_tmp210*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp113*residual_tmp7 - residual_tmp20*residual_tmp236 + residual_tmp237) + residual_tmp229*residual_tmp6 + residual_tmp234*residual_tmp235) + residual_tmp210*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp132*residual_tmp7 - residual_tmp20*residual_tmp227 - residual_tmp228) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp226 - residual_tmp230) + residual_tmp210*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp19*residual_tmp7 - residual_tmp20*residual_tmp205) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp207) + residual_tmp12*residual_tmp210) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp173*residual_tmp7 - residual_tmp194 - residual_tmp20*residual_tmp224 - residual_tmp225) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp222) + residual_tmp166*residual_tmp210) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp148*residual_tmp7 - residual_tmp20*residual_tmp218 + residual_tmp220) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp216) + residual_tmp210*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp20*residual_tmp211 + residual_tmp7*residual_tmp74) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp212) + residual_tmp20*residual_tmp210) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp164*residual_tmp7 - residual_tmp20*residual_tmp232 - residual_tmp233) + ((scalar_t(1) / scalar_t(3)))*residual_tmp12*residual_tmp231 - residual_tmp229*u0_grad_2) + residual_tmp210*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(residual_tmp187*residual_tmp7 - residual_tmp20*residual_tmp239 + residual_tmp240) + residual_tmp235*residual_tmp238 + residual_tmp241) + residual_tmp178*residual_tmp210);
            const scalar_t grad_coeff1_1 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp213*residual_tmp43 + residual_tmp237 + residual_tmp44*residual_tmp87) - residual_tmp214*residual_tmp242 + residual_tmp243*residual_tmp6) + residual_tmp244*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp113*residual_tmp44 + residual_tmp236*residual_tmp43) - residual_tmp234*residual_tmp242) + residual_tmp244*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp132*residual_tmp44 + residual_tmp227*residual_tmp43 - residual_tmp246) - residual_tmp226*residual_tmp242 - residual_tmp243*u2_grad_0) + residual_tmp244*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp116 + residual_tmp19*residual_tmp44 + residual_tmp205*residual_tmp43 - residual_tmp225) - residual_tmp207*residual_tmp242) + residual_tmp12*residual_tmp244) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp173*residual_tmp44 + residual_tmp224*residual_tmp43) - residual_tmp222*residual_tmp242) + residual_tmp166*residual_tmp244) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp148*residual_tmp44 + residual_tmp192 + residual_tmp218*residual_tmp43 + residual_tmp245) - residual_tmp216*residual_tmp242) + residual_tmp244*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp211*residual_tmp43 - residual_tmp233 + residual_tmp44*residual_tmp74) - residual_tmp212*residual_tmp242 - residual_tmp243*u0_grad_2) + residual_tmp20*residual_tmp244) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp164*residual_tmp44 + residual_tmp232*residual_tmp43) - residual_tmp231*residual_tmp242) + residual_tmp244*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp187*residual_tmp44 + residual_tmp239*residual_tmp43 + residual_tmp247) + residual_tmp141*residual_tmp243 - residual_tmp238*residual_tmp242) + residual_tmp178*residual_tmp244);
            const scalar_t grad_coeff1_2 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp213*residual_tmp47 + residual_tmp228 - residual_tmp45*residual_tmp87) + residual_tmp214*residual_tmp250 + residual_tmp230) + residual_tmp248*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp113*residual_tmp45 + residual_tmp236*residual_tmp47 - residual_tmp246) - residual_tmp229*u2_grad_0 + ((scalar_t(1) / scalar_t(3)))*residual_tmp234*residual_tmp50) + residual_tmp248*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp132*residual_tmp45 + residual_tmp227*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp226*residual_tmp50) + residual_tmp248*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp19*residual_tmp45 + residual_tmp205*residual_tmp47 - residual_tmp220) + ((scalar_t(1) / scalar_t(3)))*residual_tmp207*residual_tmp50) + residual_tmp12*residual_tmp248) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp173*residual_tmp45 - residual_tmp191 + residual_tmp224*residual_tmp47 - residual_tmp249) + ((scalar_t(1) / scalar_t(3)))*residual_tmp222*residual_tmp50) + residual_tmp166*residual_tmp248) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp148*residual_tmp45 + residual_tmp218*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp216*residual_tmp50) + residual_tmp248*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp211*residual_tmp47 - residual_tmp240 - residual_tmp45*residual_tmp74) + ((scalar_t(1) / scalar_t(3)))*residual_tmp212*residual_tmp50 - residual_tmp241) + residual_tmp20*residual_tmp248) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp164*residual_tmp45 + residual_tmp232*residual_tmp47 + residual_tmp247) + residual_tmp141*residual_tmp229 + residual_tmp231*residual_tmp250) + residual_tmp248*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp187*residual_tmp45 + residual_tmp239*residual_tmp47) + ((scalar_t(1) / scalar_t(3)))*residual_tmp238*residual_tmp50) + residual_tmp178*residual_tmp248);
            const scalar_t grad_coeff2_0 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp213 + residual_tmp7*residual_tmp85) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp255) + residual_tmp253*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(residual_tmp108*residual_tmp7 - residual_tmp12*residual_tmp236 - residual_tmp260) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp259 - residual_tmp262) + residual_tmp253*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp227 + residual_tmp137*residual_tmp7 + residual_tmp267) + residual_tmp261*residual_tmp46 + residual_tmp265*residual_tmp266) + residual_tmp253*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp205 + residual_tmp28*residual_tmp7) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp254) + residual_tmp12*residual_tmp253) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp224 + residual_tmp176*residual_tmp7 + residual_tmp269) + residual_tmp266*residual_tmp268 + residual_tmp270) + residual_tmp166*residual_tmp253) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp218 + residual_tmp152*residual_tmp7 - residual_tmp264) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp263 - residual_tmp261*u0_grad_1) + residual_tmp253*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp211 + residual_tmp7*residual_tmp72) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp251) + residual_tmp20*residual_tmp253) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp232 + residual_tmp161*residual_tmp7 + residual_tmp257) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp256) + residual_tmp253*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp12*residual_tmp239 + residual_tmp185*residual_tmp7 - residual_tmp202 - residual_tmp219) + ((scalar_t(1) / scalar_t(3)))*residual_tmp20*residual_tmp258) + residual_tmp178*residual_tmp253);
            const scalar_t grad_coeff2_1 = u0_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp213*residual_tmp51 + residual_tmp260 - residual_tmp44*residual_tmp85) + residual_tmp255*residual_tmp273 + residual_tmp262) + residual_tmp271*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp108*residual_tmp44 + residual_tmp236*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp259*residual_tmp43) + residual_tmp271*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp137*residual_tmp44 + residual_tmp227*residual_tmp51 - residual_tmp272) - residual_tmp261*u1_grad_0 + ((scalar_t(1) / scalar_t(3)))*residual_tmp265*residual_tmp43) + residual_tmp271*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp205*residual_tmp51 - residual_tmp269 - residual_tmp28*residual_tmp44) + ((scalar_t(1) / scalar_t(3)))*residual_tmp254*residual_tmp43 - residual_tmp270) + residual_tmp12*residual_tmp271) + u1_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp176*residual_tmp44 + residual_tmp224*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp268*residual_tmp43) + residual_tmp166*residual_tmp271) + u1_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp152*residual_tmp44 + residual_tmp218*residual_tmp51 + residual_tmp274) + residual_tmp141*residual_tmp261 + residual_tmp263*residual_tmp273) + residual_tmp271*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(-eta_s*(residual_tmp211*residual_tmp51 - residual_tmp257 - residual_tmp44*residual_tmp72) + ((scalar_t(1) / scalar_t(3)))*residual_tmp251*residual_tmp43) + residual_tmp20*residual_tmp271) + u2_direction_grad_1*(residual_tmp11*(-eta_s*(-residual_tmp161*residual_tmp44 + residual_tmp232*residual_tmp51) + ((scalar_t(1) / scalar_t(3)))*residual_tmp256*residual_tmp43) + residual_tmp271*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(-eta_s*(-residual_tmp185*residual_tmp44 - residual_tmp190 + residual_tmp239*residual_tmp51 - residual_tmp249) + ((scalar_t(1) / scalar_t(3)))*residual_tmp258*residual_tmp43) + residual_tmp178*residual_tmp271);
            const scalar_t grad_coeff2_2 = u0_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp213*residual_tmp50 + residual_tmp267 + residual_tmp45*residual_tmp85) - residual_tmp255*residual_tmp275 + residual_tmp276*residual_tmp46) + residual_tmp277*residual_tmp92) + u0_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp108*residual_tmp45 + residual_tmp236*residual_tmp50 - residual_tmp272) - residual_tmp259*residual_tmp275 - residual_tmp276*u1_grad_0) + residual_tmp277*residual_tmp44) + u0_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp137*residual_tmp45 + residual_tmp227*residual_tmp50) - residual_tmp265*residual_tmp275) + residual_tmp277*residual_tmp45) + u1_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp205*residual_tmp50 - residual_tmp264 + residual_tmp28*residual_tmp45) - residual_tmp254*residual_tmp275 - residual_tmp276*u0_grad_1) + residual_tmp12*residual_tmp277) + u1_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp176*residual_tmp45 + residual_tmp224*residual_tmp50 + residual_tmp274) + residual_tmp141*residual_tmp276 - residual_tmp268*residual_tmp275) + residual_tmp166*residual_tmp277) + u1_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp152*residual_tmp45 + residual_tmp218*residual_tmp50) - residual_tmp263*residual_tmp275) + residual_tmp277*residual_tmp50) + u2_direction_grad_0*(residual_tmp11*(eta_s*(residual_tmp140 + residual_tmp211*residual_tmp50 - residual_tmp219 + residual_tmp45*residual_tmp72) - residual_tmp251*residual_tmp275) + residual_tmp20*residual_tmp277) + u2_direction_grad_1*(residual_tmp11*(eta_s*(residual_tmp161*residual_tmp45 - residual_tmp190 + residual_tmp232*residual_tmp50 + residual_tmp245) - residual_tmp256*residual_tmp275) + residual_tmp277*residual_tmp43) + u2_direction_grad_2*(residual_tmp11*(eta_s*(residual_tmp185*residual_tmp45 + residual_tmp239*residual_tmp50) - residual_tmp258*residual_tmp275) + residual_tmp178*residual_tmp277);
            const scalar_t grad_coeff0_0_value = grad_coeff0_0;
            const scalar_t grad_coeff0_1_value = grad_coeff0_1;
            const scalar_t grad_coeff0_2_value = grad_coeff0_2;
            const scalar_t grad_coeff1_0_value = grad_coeff1_0;
            const scalar_t grad_coeff1_1_value = grad_coeff1_1;
            const scalar_t grad_coeff1_2_value = grad_coeff1_2;
            const scalar_t grad_coeff2_0_value = grad_coeff2_0;
            const scalar_t grad_coeff2_1_value = grad_coeff2_1;
            const scalar_t grad_coeff2_2_value = grad_coeff2_2;
            const scalar_t test0_grad0 = (-(adj0) - adj3 - adj6) / det;
            const scalar_t test0_grad1 = (-(adj1) - adj4 - adj7) / det;
            const scalar_t test0_grad2 = (-(adj2) - adj5 - adj8) / det;
            const scalar_t test1_grad0 = (adj0) / det;
            const scalar_t test1_grad1 = (adj1) / det;
            const scalar_t test1_grad2 = (adj2) / det;
            const scalar_t test2_grad0 = (adj3) / det;
            const scalar_t test2_grad1 = (adj4) / det;
            const scalar_t test2_grad2 = (adj5) / det;
            const scalar_t test3_grad0 = (adj6) / det;
            const scalar_t test3_grad1 = (adj7) / det;
            const scalar_t test3_grad2 = (adj8) / det;
            output[0][lane] += q_weight[q] * det * (grad_coeff0_0_value * test0_grad0 + grad_coeff0_1_value * test0_grad1 + grad_coeff0_2_value * test0_grad2);
            output[1][lane] += q_weight[q] * det * (grad_coeff1_0_value * test0_grad0 + grad_coeff1_1_value * test0_grad1 + grad_coeff1_2_value * test0_grad2);
            output[2][lane] += q_weight[q] * det * (grad_coeff2_0_value * test0_grad0 + grad_coeff2_1_value * test0_grad1 + grad_coeff2_2_value * test0_grad2);
            output[3][lane] += q_weight[q] * det * (grad_coeff0_0_value * test1_grad0 + grad_coeff0_1_value * test1_grad1 + grad_coeff0_2_value * test1_grad2);
            output[4][lane] += q_weight[q] * det * (grad_coeff1_0_value * test1_grad0 + grad_coeff1_1_value * test1_grad1 + grad_coeff1_2_value * test1_grad2);
            output[5][lane] += q_weight[q] * det * (grad_coeff2_0_value * test1_grad0 + grad_coeff2_1_value * test1_grad1 + grad_coeff2_2_value * test1_grad2);
            output[6][lane] += q_weight[q] * det * (grad_coeff0_0_value * test2_grad0 + grad_coeff0_1_value * test2_grad1 + grad_coeff0_2_value * test2_grad2);
            output[7][lane] += q_weight[q] * det * (grad_coeff1_0_value * test2_grad0 + grad_coeff1_1_value * test2_grad1 + grad_coeff1_2_value * test2_grad2);
            output[8][lane] += q_weight[q] * det * (grad_coeff2_0_value * test2_grad0 + grad_coeff2_1_value * test2_grad1 + grad_coeff2_2_value * test2_grad2);
            output[9][lane] += q_weight[q] * det * (grad_coeff0_0_value * test3_grad0 + grad_coeff0_1_value * test3_grad1 + grad_coeff0_2_value * test3_grad2);
            output[10][lane] += q_weight[q] * det * (grad_coeff1_0_value * test3_grad0 + grad_coeff1_1_value * test3_grad1 + grad_coeff1_2_value * test3_grad2);
            output[11][lane] += q_weight[q] * det * (grad_coeff2_0_value * test3_grad0 + grad_coeff2_1_value * test3_grad1 + grad_coeff2_2_value * test3_grad2);
        }
    }
}

} // namespace codegen
} // namespace sfem

#endif
