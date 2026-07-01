#ifndef LINEAR_ELASTICITY_D3_SIMPLEX_LOCAL_HPP
#define LINEAR_ELASTICITY_D3_SIMPLEX_LOCAL_HPP
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
static SFEM_INLINE void linear_elasticity_d3_simplex_objective_block(
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
        const scalar_t mu,
        const scalar_t lmbda,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 3],
        scalar_t *const SFEM_RESTRICT value
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
        for (int q = 0; q < N_QP; ++q) {
            const scalar_t qw = q_weight[q];
            scalar_t grad_u_ref0_values[VECTOR_SIZE];
            scalar_t grad_u_ref1_values[VECTOR_SIZE];
            scalar_t grad_u_ref2_values[VECTOR_SIZE];
            scalar_t grad_u_ref3_values[VECTOR_SIZE];
            scalar_t grad_u_ref4_values[VECTOR_SIZE];
            scalar_t grad_u_ref5_values[VECTOR_SIZE];
            scalar_t grad_u_ref6_values[VECTOR_SIZE];
            scalar_t grad_u_ref7_values[VECTOR_SIZE];
            scalar_t grad_u_ref8_values[VECTOR_SIZE];
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
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref3_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref4_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref5_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref6_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref7_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref8_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref0_values[lane] += u_streams[shape * 3 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref1_values[lane] += u_streams[shape * 3 + 0][lane] * grad_ref_y[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref2_values[lane] += u_streams[shape * 3 + 0][lane] * grad_ref_z[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref3_values[lane] += u_streams[shape * 3 + 1][lane] * grad_ref_x[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref4_values[lane] += u_streams[shape * 3 + 1][lane] * grad_ref_y[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref5_values[lane] += u_streams[shape * 3 + 1][lane] * grad_ref_z[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref6_values[lane] += u_streams[shape * 3 + 2][lane] * grad_ref_x[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref7_values[lane] += u_streams[shape * 3 + 2][lane] * grad_ref_y[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref8_values[lane] += u_streams[shape * 3 + 2][lane] * grad_ref_z[q * N_SHAPE + shape];
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
            const scalar_t grad_u_ref3 = grad_u_ref3_values[lane];
            const scalar_t grad_u_ref4 = grad_u_ref4_values[lane];
            const scalar_t grad_u_ref5 = grad_u_ref5_values[lane];
            const scalar_t grad_u_ref6 = grad_u_ref6_values[lane];
            const scalar_t grad_u_ref7 = grad_u_ref7_values[lane];
            const scalar_t grad_u_ref8 = grad_u_ref8_values[lane];
        const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
        const scalar_t grad_u0 = (grad_u_ref0 * jacobian_adjugate_lane0 + grad_u_ref1 * jacobian_adjugate_lane3 + grad_u_ref2 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t grad_u1 = (grad_u_ref0 * jacobian_adjugate_lane1 + grad_u_ref1 * jacobian_adjugate_lane4 + grad_u_ref2 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t grad_u2 = (grad_u_ref0 * jacobian_adjugate_lane2 + grad_u_ref1 * jacobian_adjugate_lane5 + grad_u_ref2 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t grad_u3 = (grad_u_ref3 * jacobian_adjugate_lane0 + grad_u_ref4 * jacobian_adjugate_lane3 + grad_u_ref5 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t grad_u4 = (grad_u_ref3 * jacobian_adjugate_lane1 + grad_u_ref4 * jacobian_adjugate_lane4 + grad_u_ref5 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t grad_u5 = (grad_u_ref3 * jacobian_adjugate_lane2 + grad_u_ref4 * jacobian_adjugate_lane5 + grad_u_ref5 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t grad_u6 = (grad_u_ref6 * jacobian_adjugate_lane0 + grad_u_ref7 * jacobian_adjugate_lane3 + grad_u_ref8 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t grad_u7 = (grad_u_ref6 * jacobian_adjugate_lane1 + grad_u_ref7 * jacobian_adjugate_lane4 + grad_u_ref8 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t grad_u8 = (grad_u_ref6 * jacobian_adjugate_lane2 + grad_u_ref7 * jacobian_adjugate_lane5 + grad_u_ref8 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        value[lane] += qw * jacobian_determinant_lane0 * (((scalar_t(1) / scalar_t(2)))*lmbda*pow_2(grad_u0 + grad_u4 + grad_u8) + mu*(pow_2(grad_u0) + pow_2(grad_u4) + pow_2(grad_u8) + scalar_t(2)*pow_2(((scalar_t(1) / scalar_t(2)))*grad_u1 + ((scalar_t(1) / scalar_t(2)))*grad_u3) + scalar_t(2)*pow_2(((scalar_t(1) / scalar_t(2)))*grad_u2 + ((scalar_t(1) / scalar_t(2)))*grad_u6) + scalar_t(2)*pow_2(((scalar_t(1) / scalar_t(2)))*grad_u5 + ((scalar_t(1) / scalar_t(2)))*grad_u7)));
            }
        }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void linear_elasticity_d3_simplex_gradient_block(
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
        const scalar_t mu,
        const scalar_t lmbda,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 3],
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 3]
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
        for (int q = 0; q < N_QP; ++q) {
            const scalar_t qw = q_weight[q];
            scalar_t grad_u_ref0_values[VECTOR_SIZE];
            scalar_t grad_u_ref1_values[VECTOR_SIZE];
            scalar_t grad_u_ref2_values[VECTOR_SIZE];
            scalar_t grad_u_ref3_values[VECTOR_SIZE];
            scalar_t grad_u_ref4_values[VECTOR_SIZE];
            scalar_t grad_u_ref5_values[VECTOR_SIZE];
            scalar_t grad_u_ref6_values[VECTOR_SIZE];
            scalar_t grad_u_ref7_values[VECTOR_SIZE];
            scalar_t grad_u_ref8_values[VECTOR_SIZE];
            scalar_t loperand0_values[VECTOR_SIZE];
            scalar_t loperand1_values[VECTOR_SIZE];
            scalar_t loperand2_values[VECTOR_SIZE];
            scalar_t loperand3_values[VECTOR_SIZE];
            scalar_t loperand4_values[VECTOR_SIZE];
            scalar_t loperand5_values[VECTOR_SIZE];
            scalar_t loperand6_values[VECTOR_SIZE];
            scalar_t loperand7_values[VECTOR_SIZE];
            scalar_t loperand8_values[VECTOR_SIZE];
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
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref3_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref4_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref5_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref6_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref7_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_u_ref8_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref0_values[lane] += u_streams[shape * 3 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref1_values[lane] += u_streams[shape * 3 + 0][lane] * grad_ref_y[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref2_values[lane] += u_streams[shape * 3 + 0][lane] * grad_ref_z[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref3_values[lane] += u_streams[shape * 3 + 1][lane] * grad_ref_x[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref4_values[lane] += u_streams[shape * 3 + 1][lane] * grad_ref_y[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref5_values[lane] += u_streams[shape * 3 + 1][lane] * grad_ref_z[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref6_values[lane] += u_streams[shape * 3 + 2][lane] * grad_ref_x[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref7_values[lane] += u_streams[shape * 3 + 2][lane] * grad_ref_y[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_u_ref8_values[lane] += u_streams[shape * 3 + 2][lane] * grad_ref_z[q * N_SHAPE + shape];
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
            const scalar_t grad_u_ref3 = grad_u_ref3_values[lane];
            const scalar_t grad_u_ref4 = grad_u_ref4_values[lane];
            const scalar_t grad_u_ref5 = grad_u_ref5_values[lane];
            const scalar_t grad_u_ref6 = grad_u_ref6_values[lane];
            const scalar_t grad_u_ref7 = grad_u_ref7_values[lane];
            const scalar_t grad_u_ref8 = grad_u_ref8_values[lane];
        const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
        const scalar_t grad_u0 = (grad_u_ref0 * jacobian_adjugate_lane0 + grad_u_ref1 * jacobian_adjugate_lane3 + grad_u_ref2 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t grad_u1 = (grad_u_ref0 * jacobian_adjugate_lane1 + grad_u_ref1 * jacobian_adjugate_lane4 + grad_u_ref2 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t grad_u2 = (grad_u_ref0 * jacobian_adjugate_lane2 + grad_u_ref1 * jacobian_adjugate_lane5 + grad_u_ref2 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t grad_u3 = (grad_u_ref3 * jacobian_adjugate_lane0 + grad_u_ref4 * jacobian_adjugate_lane3 + grad_u_ref5 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t grad_u4 = (grad_u_ref3 * jacobian_adjugate_lane1 + grad_u_ref4 * jacobian_adjugate_lane4 + grad_u_ref5 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t grad_u5 = (grad_u_ref3 * jacobian_adjugate_lane2 + grad_u_ref4 * jacobian_adjugate_lane5 + grad_u_ref5 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t grad_u6 = (grad_u_ref6 * jacobian_adjugate_lane0 + grad_u_ref7 * jacobian_adjugate_lane3 + grad_u_ref8 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t grad_u7 = (grad_u_ref6 * jacobian_adjugate_lane1 + grad_u_ref7 * jacobian_adjugate_lane4 + grad_u_ref8 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t grad_u8 = (grad_u_ref6 * jacobian_adjugate_lane2 + grad_u_ref7 * jacobian_adjugate_lane5 + grad_u_ref8 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t weak_mat_tmp0 = scalar_t(2)*grad_u0;
        const scalar_t weak_mat_tmp1 = scalar_t(2)*grad_u4;
        const scalar_t weak_mat_tmp2 = scalar_t(2)*grad_u8;
        const scalar_t weak_mat_tmp3 = ((scalar_t(1) / scalar_t(2)))*lmbda*(weak_mat_tmp0 + weak_mat_tmp1 + weak_mat_tmp2);
        const scalar_t weak_mat_tmp4 = mu*(grad_u1 + grad_u3);
        const scalar_t weak_mat_tmp5 = mu*(grad_u2 + grad_u6);
        const scalar_t weak_mat_tmp6 = mu*(grad_u5 + grad_u7);
        const scalar_t material0 = mu*weak_mat_tmp0 + weak_mat_tmp3;
        const scalar_t material1 = weak_mat_tmp4;
        const scalar_t material2 = weak_mat_tmp5;
        const scalar_t material3 = weak_mat_tmp4;
        const scalar_t material4 = mu*weak_mat_tmp1 + weak_mat_tmp3;
        const scalar_t material5 = weak_mat_tmp6;
        const scalar_t material6 = weak_mat_tmp5;
        const scalar_t material7 = weak_mat_tmp6;
        const scalar_t material8 = mu*weak_mat_tmp2 + weak_mat_tmp3;
        const scalar_t loperand0 = qw * (material0 * jacobian_adjugate_lane0 + material1 * jacobian_adjugate_lane1 + material2 * jacobian_adjugate_lane2);
        const scalar_t loperand1 = qw * (material0 * jacobian_adjugate_lane3 + material1 * jacobian_adjugate_lane4 + material2 * jacobian_adjugate_lane5);
        const scalar_t loperand2 = qw * (material0 * jacobian_adjugate_lane6 + material1 * jacobian_adjugate_lane7 + material2 * jacobian_adjugate_lane8);
        const scalar_t loperand3 = qw * (material3 * jacobian_adjugate_lane0 + material4 * jacobian_adjugate_lane1 + material5 * jacobian_adjugate_lane2);
        const scalar_t loperand4 = qw * (material3 * jacobian_adjugate_lane3 + material4 * jacobian_adjugate_lane4 + material5 * jacobian_adjugate_lane5);
        const scalar_t loperand5 = qw * (material3 * jacobian_adjugate_lane6 + material4 * jacobian_adjugate_lane7 + material5 * jacobian_adjugate_lane8);
        const scalar_t loperand6 = qw * (material6 * jacobian_adjugate_lane0 + material7 * jacobian_adjugate_lane1 + material8 * jacobian_adjugate_lane2);
        const scalar_t loperand7 = qw * (material6 * jacobian_adjugate_lane3 + material7 * jacobian_adjugate_lane4 + material8 * jacobian_adjugate_lane5);
        const scalar_t loperand8 = qw * (material6 * jacobian_adjugate_lane6 + material7 * jacobian_adjugate_lane7 + material8 * jacobian_adjugate_lane8);
            loperand0_values[lane] = loperand0;
            loperand1_values[lane] = loperand1;
            loperand2_values[lane] = loperand2;
            loperand3_values[lane] = loperand3;
            loperand4_values[lane] = loperand4;
            loperand5_values[lane] = loperand5;
            loperand6_values[lane] = loperand6;
            loperand7_values[lane] = loperand7;
            loperand8_values[lane] = loperand8;
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    out_streams[shape * 3 + 0][lane] += loperand0_values[lane] * grad_ref_x[q * N_SHAPE + shape] + loperand1_values[lane] * grad_ref_y[q * N_SHAPE + shape] + loperand2_values[lane] * grad_ref_z[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    out_streams[shape * 3 + 1][lane] += loperand3_values[lane] * grad_ref_x[q * N_SHAPE + shape] + loperand4_values[lane] * grad_ref_y[q * N_SHAPE + shape] + loperand5_values[lane] * grad_ref_z[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    out_streams[shape * 3 + 2][lane] += loperand6_values[lane] * grad_ref_x[q * N_SHAPE + shape] + loperand7_values[lane] * grad_ref_y[q * N_SHAPE + shape] + loperand8_values[lane] * grad_ref_z[q * N_SHAPE + shape];
                }
            }
        }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void linear_elasticity_d3_simplex_apply_block(
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
        const scalar_t mu,
        const scalar_t lmbda,
        const scalar_t *const SFEM_RESTRICT h_streams[N_SHAPE * 3],
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 3]
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
        for (int q = 0; q < N_QP; ++q) {
            const scalar_t qw = q_weight[q];
            scalar_t grad_h_ref0_values[VECTOR_SIZE];
            scalar_t grad_h_ref1_values[VECTOR_SIZE];
            scalar_t grad_h_ref2_values[VECTOR_SIZE];
            scalar_t grad_h_ref3_values[VECTOR_SIZE];
            scalar_t grad_h_ref4_values[VECTOR_SIZE];
            scalar_t grad_h_ref5_values[VECTOR_SIZE];
            scalar_t grad_h_ref6_values[VECTOR_SIZE];
            scalar_t grad_h_ref7_values[VECTOR_SIZE];
            scalar_t grad_h_ref8_values[VECTOR_SIZE];
            scalar_t loperand0_values[VECTOR_SIZE];
            scalar_t loperand1_values[VECTOR_SIZE];
            scalar_t loperand2_values[VECTOR_SIZE];
            scalar_t loperand3_values[VECTOR_SIZE];
            scalar_t loperand4_values[VECTOR_SIZE];
            scalar_t loperand5_values[VECTOR_SIZE];
            scalar_t loperand6_values[VECTOR_SIZE];
            scalar_t loperand7_values[VECTOR_SIZE];
            scalar_t loperand8_values[VECTOR_SIZE];
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
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_h_ref3_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_h_ref4_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_h_ref5_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_h_ref6_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_h_ref7_values[lane] = scalar_t(0);
            }
            #pragma omp simd
            for (int lane = 0; lane < nelems; ++lane) {
                grad_h_ref8_values[lane] = scalar_t(0);
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_h_ref0_values[lane] += h_streams[shape * 3 + 0][lane] * grad_ref_x[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_h_ref1_values[lane] += h_streams[shape * 3 + 0][lane] * grad_ref_y[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_h_ref2_values[lane] += h_streams[shape * 3 + 0][lane] * grad_ref_z[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_h_ref3_values[lane] += h_streams[shape * 3 + 1][lane] * grad_ref_x[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_h_ref4_values[lane] += h_streams[shape * 3 + 1][lane] * grad_ref_y[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_h_ref5_values[lane] += h_streams[shape * 3 + 1][lane] * grad_ref_z[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_h_ref6_values[lane] += h_streams[shape * 3 + 2][lane] * grad_ref_x[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_h_ref7_values[lane] += h_streams[shape * 3 + 2][lane] * grad_ref_y[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    grad_h_ref8_values[lane] += h_streams[shape * 3 + 2][lane] * grad_ref_z[q * N_SHAPE + shape];
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
            const scalar_t grad_h_ref3 = grad_h_ref3_values[lane];
            const scalar_t grad_h_ref4 = grad_h_ref4_values[lane];
            const scalar_t grad_h_ref5 = grad_h_ref5_values[lane];
            const scalar_t grad_h_ref6 = grad_h_ref6_values[lane];
            const scalar_t grad_h_ref7 = grad_h_ref7_values[lane];
            const scalar_t grad_h_ref8 = grad_h_ref8_values[lane];
        const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
        const scalar_t trial_grad0 = (grad_h_ref0 * jacobian_adjugate_lane0 + grad_h_ref1 * jacobian_adjugate_lane3 + grad_h_ref2 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t trial_grad1 = (grad_h_ref0 * jacobian_adjugate_lane1 + grad_h_ref1 * jacobian_adjugate_lane4 + grad_h_ref2 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t trial_grad2 = (grad_h_ref0 * jacobian_adjugate_lane2 + grad_h_ref1 * jacobian_adjugate_lane5 + grad_h_ref2 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t trial_grad3 = (grad_h_ref3 * jacobian_adjugate_lane0 + grad_h_ref4 * jacobian_adjugate_lane3 + grad_h_ref5 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t trial_grad4 = (grad_h_ref3 * jacobian_adjugate_lane1 + grad_h_ref4 * jacobian_adjugate_lane4 + grad_h_ref5 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t trial_grad5 = (grad_h_ref3 * jacobian_adjugate_lane2 + grad_h_ref4 * jacobian_adjugate_lane5 + grad_h_ref5 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t trial_grad6 = (grad_h_ref6 * jacobian_adjugate_lane0 + grad_h_ref7 * jacobian_adjugate_lane3 + grad_h_ref8 * jacobian_adjugate_lane6) * inv_jacobian_determinant;
        const scalar_t trial_grad7 = (grad_h_ref6 * jacobian_adjugate_lane1 + grad_h_ref7 * jacobian_adjugate_lane4 + grad_h_ref8 * jacobian_adjugate_lane7) * inv_jacobian_determinant;
        const scalar_t trial_grad8 = (grad_h_ref6 * jacobian_adjugate_lane2 + grad_h_ref7 * jacobian_adjugate_lane5 + grad_h_ref8 * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t weak_mat_tmp0 = lmbda*trial_grad4;
        const scalar_t weak_mat_tmp1 = lmbda*trial_grad8;
        const scalar_t weak_mat_tmp2 = lmbda + scalar_t(2)*mu;
        const scalar_t weak_mat_tmp3 = mu*trial_grad1 + mu*trial_grad3;
        const scalar_t weak_mat_tmp4 = mu*trial_grad2 + mu*trial_grad6;
        const scalar_t weak_mat_tmp5 = lmbda*trial_grad0;
        const scalar_t weak_mat_tmp6 = mu*trial_grad5 + mu*trial_grad7;
        const scalar_t material0 = trial_grad0*weak_mat_tmp2 + weak_mat_tmp0 + weak_mat_tmp1;
        const scalar_t material1 = weak_mat_tmp3;
        const scalar_t material2 = weak_mat_tmp4;
        const scalar_t material3 = weak_mat_tmp3;
        const scalar_t material4 = trial_grad4*weak_mat_tmp2 + weak_mat_tmp1 + weak_mat_tmp5;
        const scalar_t material5 = weak_mat_tmp6;
        const scalar_t material6 = weak_mat_tmp4;
        const scalar_t material7 = weak_mat_tmp6;
        const scalar_t material8 = trial_grad8*weak_mat_tmp2 + weak_mat_tmp0 + weak_mat_tmp5;
        const scalar_t loperand0 = qw * (material0 * jacobian_adjugate_lane0 + material1 * jacobian_adjugate_lane1 + material2 * jacobian_adjugate_lane2);
        const scalar_t loperand1 = qw * (material0 * jacobian_adjugate_lane3 + material1 * jacobian_adjugate_lane4 + material2 * jacobian_adjugate_lane5);
        const scalar_t loperand2 = qw * (material0 * jacobian_adjugate_lane6 + material1 * jacobian_adjugate_lane7 + material2 * jacobian_adjugate_lane8);
        const scalar_t loperand3 = qw * (material3 * jacobian_adjugate_lane0 + material4 * jacobian_adjugate_lane1 + material5 * jacobian_adjugate_lane2);
        const scalar_t loperand4 = qw * (material3 * jacobian_adjugate_lane3 + material4 * jacobian_adjugate_lane4 + material5 * jacobian_adjugate_lane5);
        const scalar_t loperand5 = qw * (material3 * jacobian_adjugate_lane6 + material4 * jacobian_adjugate_lane7 + material5 * jacobian_adjugate_lane8);
        const scalar_t loperand6 = qw * (material6 * jacobian_adjugate_lane0 + material7 * jacobian_adjugate_lane1 + material8 * jacobian_adjugate_lane2);
        const scalar_t loperand7 = qw * (material6 * jacobian_adjugate_lane3 + material7 * jacobian_adjugate_lane4 + material8 * jacobian_adjugate_lane5);
        const scalar_t loperand8 = qw * (material6 * jacobian_adjugate_lane6 + material7 * jacobian_adjugate_lane7 + material8 * jacobian_adjugate_lane8);
            loperand0_values[lane] = loperand0;
            loperand1_values[lane] = loperand1;
            loperand2_values[lane] = loperand2;
            loperand3_values[lane] = loperand3;
            loperand4_values[lane] = loperand4;
            loperand5_values[lane] = loperand5;
            loperand6_values[lane] = loperand6;
            loperand7_values[lane] = loperand7;
            loperand8_values[lane] = loperand8;
            }
            for (int shape = 0; shape < N_SHAPE; ++shape) {
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    out_streams[shape * 3 + 0][lane] += loperand0_values[lane] * grad_ref_x[q * N_SHAPE + shape] + loperand1_values[lane] * grad_ref_y[q * N_SHAPE + shape] + loperand2_values[lane] * grad_ref_z[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    out_streams[shape * 3 + 1][lane] += loperand3_values[lane] * grad_ref_x[q * N_SHAPE + shape] + loperand4_values[lane] * grad_ref_y[q * N_SHAPE + shape] + loperand5_values[lane] * grad_ref_z[q * N_SHAPE + shape];
                }
                #pragma omp simd
                for (int lane = 0; lane < nelems; ++lane) {
                    out_streams[shape * 3 + 2][lane] += loperand6_values[lane] * grad_ref_x[q * N_SHAPE + shape] + loperand7_values[lane] * grad_ref_y[q * N_SHAPE + shape] + loperand8_values[lane] * grad_ref_z[q * N_SHAPE + shape];
                }
            }
        }
}

} // namespace codegen
} // namespace sfem

#endif
