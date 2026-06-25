#ifndef GENERATED_NEOHOOKEAN_OGDEN_D3_TENSOR_PRODUCT_LOCAL_HPP
#define GENERATED_NEOHOOKEAN_OGDEN_D3_TENSOR_PRODUCT_LOCAL_HPP

#include <math.h>
#include <stddef.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_GENERATED_SCALAR_T
#endif
#endif

#include "kernel_math.hpp"

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

#ifndef SFEM_GENERATED_INTEGER_ROOT
#define SFEM_GENERATED_INTEGER_ROOT
static constexpr int sfem_generated_ipow(const int base, const int exponent) {
    return exponent == 0 ? 1 : base * sfem_generated_ipow(base, exponent - 1);
}
static constexpr int sfem_generated_integer_root_search(const int value, const int exponent, const int candidate) {
    return sfem_generated_ipow(candidate, exponent) >= value ? candidate : sfem_generated_integer_root_search(value, exponent, candidate + 1);
}
static constexpr int sfem_generated_integer_root(const int value, const int exponent) {
    return sfem_generated_integer_root_search(value, exponent, 1);
}
#endif

namespace sfem {
namespace codegen {

template <int N_SHAPE_1D>
static SFEM_INLINE int generated_neohookean_ogden_d3_tensor_product_tensor_shape_index(
        const int sx,
        const int sy,
        const int sz) {
    return N_SHAPE_1D == 2
               ? (sy == 0 ? sx : (sx == 0 ? 3 : 2)) + 4 * sz
               : sx + N_SHAPE_1D * (sy + N_SHAPE_1D * sz);
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void generated_neohookean_ogden_d3_tensor_product_tensor_gradient(
        const ptrdiff_t nelems,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT streams[N_SHAPE * 3],
        const int component,
        scalar_t *const SFEM_RESTRICT gradient) {
    static constexpr int Q = sfem_generated_integer_root(N_QP, 3);
    static constexpr int S = sfem_generated_integer_root(N_SHAPE, 3);
    scalar_t value_x[Q * S * S * VECTOR_SIZE];
    scalar_t grad_x[Q * S * S * VECTOR_SIZE];
    scalar_t value_xy[Q * Q * S * VECTOR_SIZE];
    scalar_t grad_x_xy[Q * Q * S * VECTOR_SIZE];
    scalar_t grad_y_xy[Q * Q * S * VECTOR_SIZE];
    for (int qx = 0; qx < Q; ++qx) {
        for (int sy = 0; sy < S; ++sy) {
            for (int sz = 0; sz < S; ++sz) {
#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    scalar_t v = 0; scalar_t gx = 0;
                    for (int sx = 0; sx < S; ++sx) {
                        const int shape = generated_neohookean_ogden_d3_tensor_product_tensor_shape_index<S>(sx, sy, sz);
                        const scalar_t u = streams[shape * 3 + component][lane];
                        v += u * shape_1d[qx * S + sx];
                        gx += u * grad_1d[qx * S + sx];
                    }
                    const int i = ((qx * S + sy) * S + sz) * VECTOR_SIZE + lane;
                    value_x[i] = v; grad_x[i] = gx;
                }
            }
        }
    }
    for (int qx = 0; qx < Q; ++qx) {
        for (int qy = 0; qy < Q; ++qy) {
            for (int sz = 0; sz < S; ++sz) {
#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    scalar_t v = 0; scalar_t gx = 0; scalar_t gy = 0;
                    for (int sy = 0; sy < S; ++sy) {
                        const int i = ((qx * S + sy) * S + sz) * VECTOR_SIZE + lane;
                        v += value_x[i] * shape_1d[qy * S + sy];
                        gx += grad_x[i] * shape_1d[qy * S + sy];
                        gy += value_x[i] * grad_1d[qy * S + sy];
                    }
                    const int j = ((qx * Q + qy) * S + sz) * VECTOR_SIZE + lane;
                    value_xy[j] = v; grad_x_xy[j] = gx; grad_y_xy[j] = gy;
                }
            }
        }
    }
    for (int qz = 0; qz < Q; ++qz) {
        for (int qy = 0; qy < Q; ++qy) {
            for (int qx = 0; qx < Q; ++qx) {
                const int q = qx + Q * (qy + Q * qz);
#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    scalar_t gx = 0; scalar_t gy = 0; scalar_t gz = 0;
                    for (int sz = 0; sz < S; ++sz) {
                        const int j = ((qx * Q + qy) * S + sz) * VECTOR_SIZE + lane;
                        gx += grad_x_xy[j] * shape_1d[qz * S + sz];
                        gy += grad_y_xy[j] * shape_1d[qz * S + sz];
                        gz += value_xy[j] * grad_1d[qz * S + sz];
                    }
                    gradient[(q * 3 + 0) * VECTOR_SIZE + lane] = gx;
                    gradient[(q * 3 + 1) * VECTOR_SIZE + lane] = gy;
                    gradient[(q * 3 + 2) * VECTOR_SIZE + lane] = gz;
                }
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void generated_neohookean_ogden_d3_tensor_product_tensor_test(
        const ptrdiff_t nelems,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT flux,
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 3],
        const int component) {
    static constexpr int Q = sfem_generated_integer_root(N_QP, 3);
    static constexpr int S = sfem_generated_integer_root(N_SHAPE, 3);
    scalar_t stage_x[Q * Q * S * VECTOR_SIZE];
    scalar_t stage_y[Q * Q * S * VECTOR_SIZE];
    scalar_t stage_z[Q * Q * S * VECTOR_SIZE];
    scalar_t stage_xy_x[Q * S * S * VECTOR_SIZE];
    scalar_t stage_xy_y[Q * S * S * VECTOR_SIZE];
    scalar_t stage_xy_z[Q * S * S * VECTOR_SIZE];
    for (int qx = 0; qx < Q; ++qx) {
        for (int qy = 0; qy < Q; ++qy) {
            for (int sz = 0; sz < S; ++sz) {
#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    scalar_t tx = 0; scalar_t ty = 0; scalar_t tz = 0;
                    for (int qz = 0; qz < Q; ++qz) {
                        const int q = qx + Q * (qy + Q * qz);
                        tx += flux[(q * 3 + 0) * VECTOR_SIZE + lane] * shape_1d[qz * S + sz];
                        ty += flux[(q * 3 + 1) * VECTOR_SIZE + lane] * shape_1d[qz * S + sz];
                        tz += flux[(q * 3 + 2) * VECTOR_SIZE + lane] * grad_1d[qz * S + sz];
                    }
                    const int i = ((qx * Q + qy) * S + sz) * VECTOR_SIZE + lane;
                    stage_x[i] = tx; stage_y[i] = ty; stage_z[i] = tz;
                }
            }
        }
    }
    for (int qx = 0; qx < Q; ++qx) {
        for (int sy = 0; sy < S; ++sy) {
            for (int sz = 0; sz < S; ++sz) {
#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    scalar_t tx = 0; scalar_t ty = 0; scalar_t tz = 0;
                    for (int qy = 0; qy < Q; ++qy) {
                        const int i = ((qx * Q + qy) * S + sz) * VECTOR_SIZE + lane;
                        tx += stage_x[i] * shape_1d[qy * S + sy];
                        ty += stage_y[i] * grad_1d[qy * S + sy];
                        tz += stage_z[i] * shape_1d[qy * S + sy];
                    }
                    const int j = ((qx * S + sy) * S + sz) * VECTOR_SIZE + lane;
                    stage_xy_x[j] = tx; stage_xy_y[j] = ty; stage_xy_z[j] = tz;
                }
            }
        }
    }
    for (int sz = 0; sz < S; ++sz) {
        for (int sy = 0; sy < S; ++sy) {
            for (int sx = 0; sx < S; ++sx) {
                const int shape = generated_neohookean_ogden_d3_tensor_product_tensor_shape_index<S>(sx, sy, sz);
#pragma omp simd
                for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
                    scalar_t value = 0;
                    for (int qx = 0; qx < Q; ++qx) {
                        const int j = ((qx * S + sy) * S + sz) * VECTOR_SIZE + lane;
                        value += stage_xy_x[j] * grad_1d[qx * S + sx]
                               + (stage_xy_y[j] + stage_xy_z[j]) * shape_1d[qx * S + sx];
                    }
                    out_streams[shape * 3 + component][lane] += value;
                }
            }
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void generated_neohookean_ogden_d3_tensor_product_objective_block(
        const ptrdiff_t nelems,
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
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t mu,
        const scalar_t lmbda,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 3],
        scalar_t *const SFEM_RESTRICT value
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
    static constexpr int N_QP_1D = sfem_generated_integer_root(N_QP, 3);
    static constexpr int N_SHAPE_1D = sfem_generated_integer_root(N_SHAPE, 3);
    static_assert(sfem_generated_ipow(N_QP_1D, 3) == N_QP, "N_QP must be tensor-product compatible");
    static_assert(sfem_generated_ipow(N_SHAPE_1D, 3) == N_SHAPE, "N_SHAPE must be tensor-product compatible");
    scalar_t grad_u_ref_q[N_QP * 9 * VECTOR_SIZE];
    generated_neohookean_ogden_d3_tensor_product_tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, shape_1d, grad_1d, u_streams, 0, &grad_u_ref_q[0 * N_QP * 3 * VECTOR_SIZE]);
    generated_neohookean_ogden_d3_tensor_product_tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, shape_1d, grad_1d, u_streams, 1, &grad_u_ref_q[1 * N_QP * 3 * VECTOR_SIZE]);
    generated_neohookean_ogden_d3_tensor_product_tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, shape_1d, grad_1d, u_streams, 2, &grad_u_ref_q[2 * N_QP * 3 * VECTOR_SIZE]);
    for (int q = 0; q < N_QP; ++q) {
        const int qx = q % N_QP_1D;
        const int qy = (q / N_QP_1D) % N_QP_1D;
        const int qz = q / (N_QP_1D * N_QP_1D);
        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
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
            scalar_t grad_u_ref[9];
            grad_u_ref[0] = grad_u_ref_q[((0 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[1] = grad_u_ref_q[((0 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[2] = grad_u_ref_q[((0 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            grad_u_ref[3] = grad_u_ref_q[((1 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[4] = grad_u_ref_q[((1 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[5] = grad_u_ref_q[((1 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            grad_u_ref[6] = grad_u_ref_q[((2 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[7] = grad_u_ref_q[((2 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[8] = grad_u_ref_q[((2 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            scalar_t grad_u[9];
            const scalar_t inv_jacobian_determinant = 1.0 / jacobian_determinant_lane0;
            grad_u[0] = (grad_u_ref[0] * jacobian_adjugate_lane0 + grad_u_ref[1] * jacobian_adjugate_lane3 + grad_u_ref[2] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            grad_u[1] = (grad_u_ref[0] * jacobian_adjugate_lane1 + grad_u_ref[1] * jacobian_adjugate_lane4 + grad_u_ref[2] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            grad_u[2] = (grad_u_ref[0] * jacobian_adjugate_lane2 + grad_u_ref[1] * jacobian_adjugate_lane5 + grad_u_ref[2] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            grad_u[3] = (grad_u_ref[3] * jacobian_adjugate_lane0 + grad_u_ref[4] * jacobian_adjugate_lane3 + grad_u_ref[5] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            grad_u[4] = (grad_u_ref[3] * jacobian_adjugate_lane1 + grad_u_ref[4] * jacobian_adjugate_lane4 + grad_u_ref[5] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            grad_u[5] = (grad_u_ref[3] * jacobian_adjugate_lane2 + grad_u_ref[4] * jacobian_adjugate_lane5 + grad_u_ref[5] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            grad_u[6] = (grad_u_ref[6] * jacobian_adjugate_lane0 + grad_u_ref[7] * jacobian_adjugate_lane3 + grad_u_ref[8] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            grad_u[7] = (grad_u_ref[6] * jacobian_adjugate_lane1 + grad_u_ref[7] * jacobian_adjugate_lane4 + grad_u_ref[8] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            grad_u[8] = (grad_u_ref[6] * jacobian_adjugate_lane2 + grad_u_ref[7] * jacobian_adjugate_lane5 + grad_u_ref[8] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t weak_obj_tmp0 = grad_u[0] + 1;
        const scalar_t weak_obj_tmp1 = grad_u[4] + 1;
        const scalar_t weak_obj_tmp2 = grad_u[8] + 1;
        const scalar_t weak_obj_tmp3 = log(-grad_u[1]*grad_u[3]*weak_obj_tmp2 + grad_u[1]*grad_u[5]*grad_u[6] + grad_u[2]*grad_u[3]*grad_u[7] - grad_u[2]*grad_u[6]*weak_obj_tmp1 - grad_u[5]*grad_u[7]*weak_obj_tmp0 + weak_obj_tmp0*weak_obj_tmp1*weak_obj_tmp2);
        value[lane] += qw * jacobian_determinant_lane0 * ((1.0/2.0)*lmbda*pow_2(weak_obj_tmp3) - mu*weak_obj_tmp3 + (1.0/2.0)*mu*(pow_2(grad_u[1]) + pow_2(grad_u[2]) + pow_2(grad_u[3]) + pow_2(grad_u[5]) + pow_2(grad_u[6]) + pow_2(grad_u[7]) + pow_2(weak_obj_tmp0) + pow_2(weak_obj_tmp1) + pow_2(weak_obj_tmp2) - 3));
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void generated_neohookean_ogden_d3_tensor_product_gradient_block(
        const ptrdiff_t nelems,
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
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t mu,
        const scalar_t lmbda,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 3],
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 3]
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
    static constexpr int N_QP_1D = sfem_generated_integer_root(N_QP, 3);
    static constexpr int N_SHAPE_1D = sfem_generated_integer_root(N_SHAPE, 3);
    static_assert(sfem_generated_ipow(N_QP_1D, 3) == N_QP, "N_QP must be tensor-product compatible");
    static_assert(sfem_generated_ipow(N_SHAPE_1D, 3) == N_SHAPE, "N_SHAPE must be tensor-product compatible");
    scalar_t grad_u_ref_q[N_QP * 9 * VECTOR_SIZE];
    scalar_t loperand_q[N_QP * 9 * VECTOR_SIZE];
    generated_neohookean_ogden_d3_tensor_product_tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, shape_1d, grad_1d, u_streams, 0, &grad_u_ref_q[0 * N_QP * 3 * VECTOR_SIZE]);
    generated_neohookean_ogden_d3_tensor_product_tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, shape_1d, grad_1d, u_streams, 1, &grad_u_ref_q[1 * N_QP * 3 * VECTOR_SIZE]);
    generated_neohookean_ogden_d3_tensor_product_tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, shape_1d, grad_1d, u_streams, 2, &grad_u_ref_q[2 * N_QP * 3 * VECTOR_SIZE]);
    for (int q = 0; q < N_QP; ++q) {
        const int qx = q % N_QP_1D;
        const int qy = (q / N_QP_1D) % N_QP_1D;
        const int qz = q / (N_QP_1D * N_QP_1D);
        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
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
            scalar_t grad_u_ref[9];
            grad_u_ref[0] = grad_u_ref_q[((0 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[1] = grad_u_ref_q[((0 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[2] = grad_u_ref_q[((0 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            grad_u_ref[3] = grad_u_ref_q[((1 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[4] = grad_u_ref_q[((1 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[5] = grad_u_ref_q[((1 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            grad_u_ref[6] = grad_u_ref_q[((2 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[7] = grad_u_ref_q[((2 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[8] = grad_u_ref_q[((2 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            scalar_t grad_u[9];
            const scalar_t inv_jacobian_determinant = 1.0 / jacobian_determinant_lane0;
            grad_u[0] = (grad_u_ref[0] * jacobian_adjugate_lane0 + grad_u_ref[1] * jacobian_adjugate_lane3 + grad_u_ref[2] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            grad_u[1] = (grad_u_ref[0] * jacobian_adjugate_lane1 + grad_u_ref[1] * jacobian_adjugate_lane4 + grad_u_ref[2] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            grad_u[2] = (grad_u_ref[0] * jacobian_adjugate_lane2 + grad_u_ref[1] * jacobian_adjugate_lane5 + grad_u_ref[2] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            grad_u[3] = (grad_u_ref[3] * jacobian_adjugate_lane0 + grad_u_ref[4] * jacobian_adjugate_lane3 + grad_u_ref[5] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            grad_u[4] = (grad_u_ref[3] * jacobian_adjugate_lane1 + grad_u_ref[4] * jacobian_adjugate_lane4 + grad_u_ref[5] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            grad_u[5] = (grad_u_ref[3] * jacobian_adjugate_lane2 + grad_u_ref[4] * jacobian_adjugate_lane5 + grad_u_ref[5] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            grad_u[6] = (grad_u_ref[6] * jacobian_adjugate_lane0 + grad_u_ref[7] * jacobian_adjugate_lane3 + grad_u_ref[8] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            grad_u[7] = (grad_u_ref[6] * jacobian_adjugate_lane1 + grad_u_ref[7] * jacobian_adjugate_lane4 + grad_u_ref[8] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            grad_u[8] = (grad_u_ref[6] * jacobian_adjugate_lane2 + grad_u_ref[7] * jacobian_adjugate_lane5 + grad_u_ref[8] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            scalar_t loperand[9];
        scalar_t material[9];
        const scalar_t weak_mat_tmp0 = grad_u[0] + 1;
        const scalar_t weak_mat_tmp1 = grad_u[5]*grad_u[7];
        const scalar_t weak_mat_tmp2 = grad_u[4] + 1;
        const scalar_t weak_mat_tmp3 = grad_u[8] + 1;
        const scalar_t weak_mat_tmp4 = -weak_mat_tmp1 + weak_mat_tmp2*weak_mat_tmp3;
        const scalar_t weak_mat_tmp5 = grad_u[3]*weak_mat_tmp3;
        const scalar_t weak_mat_tmp6 = grad_u[6]*weak_mat_tmp2;
        const scalar_t weak_mat_tmp7 = grad_u[1]*grad_u[5]*grad_u[6] - grad_u[1]*weak_mat_tmp5 + grad_u[2]*grad_u[3]*grad_u[7] - grad_u[2]*weak_mat_tmp6 - weak_mat_tmp0*weak_mat_tmp1 + weak_mat_tmp0*weak_mat_tmp2*weak_mat_tmp3;
        const scalar_t weak_mat_tmp8 = pow_m1(weak_mat_tmp7);
        const scalar_t weak_mat_tmp9 = mu*weak_mat_tmp8;
        const scalar_t weak_mat_tmp10 = lmbda*weak_mat_tmp8*log(weak_mat_tmp7);
        const scalar_t weak_mat_tmp11 = grad_u[5]*grad_u[6] - weak_mat_tmp5;
        const scalar_t weak_mat_tmp12 = grad_u[3]*grad_u[7] - weak_mat_tmp6;
        const scalar_t weak_mat_tmp13 = -grad_u[1]*weak_mat_tmp3 + grad_u[2]*grad_u[7];
        const scalar_t weak_mat_tmp14 = -grad_u[2]*grad_u[6] + weak_mat_tmp0*weak_mat_tmp3;
        const scalar_t weak_mat_tmp15 = grad_u[1]*grad_u[6] - grad_u[7]*weak_mat_tmp0;
        const scalar_t weak_mat_tmp16 = grad_u[1]*grad_u[5] - grad_u[2]*weak_mat_tmp2;
        const scalar_t weak_mat_tmp17 = grad_u[2]*grad_u[3] - grad_u[5]*weak_mat_tmp0;
        const scalar_t weak_mat_tmp18 = -grad_u[1]*grad_u[3] + weak_mat_tmp0*weak_mat_tmp2;
        material[0] = mu*weak_mat_tmp0 + weak_mat_tmp10*weak_mat_tmp4 - weak_mat_tmp4*weak_mat_tmp9;
        material[1] = grad_u[1]*mu + weak_mat_tmp10*weak_mat_tmp11 - weak_mat_tmp11*weak_mat_tmp9;
        material[2] = grad_u[2]*mu + weak_mat_tmp10*weak_mat_tmp12 - weak_mat_tmp12*weak_mat_tmp9;
        material[3] = grad_u[3]*mu + weak_mat_tmp10*weak_mat_tmp13 - weak_mat_tmp13*weak_mat_tmp9;
        material[4] = mu*weak_mat_tmp2 + weak_mat_tmp10*weak_mat_tmp14 - weak_mat_tmp14*weak_mat_tmp9;
        material[5] = grad_u[5]*mu + weak_mat_tmp10*weak_mat_tmp15 - weak_mat_tmp15*weak_mat_tmp9;
        material[6] = grad_u[6]*mu + weak_mat_tmp10*weak_mat_tmp16 - weak_mat_tmp16*weak_mat_tmp9;
        material[7] = grad_u[7]*mu + weak_mat_tmp10*weak_mat_tmp17 - weak_mat_tmp17*weak_mat_tmp9;
        material[8] = mu*weak_mat_tmp3 + weak_mat_tmp10*weak_mat_tmp18 - weak_mat_tmp18*weak_mat_tmp9;
        loperand[0] = qw * (material[0] * jacobian_adjugate_lane0 + material[1] * jacobian_adjugate_lane1 + material[2] * jacobian_adjugate_lane2);
        loperand[1] = qw * (material[0] * jacobian_adjugate_lane3 + material[1] * jacobian_adjugate_lane4 + material[2] * jacobian_adjugate_lane5);
        loperand[2] = qw * (material[0] * jacobian_adjugate_lane6 + material[1] * jacobian_adjugate_lane7 + material[2] * jacobian_adjugate_lane8);
        loperand[3] = qw * (material[3] * jacobian_adjugate_lane0 + material[4] * jacobian_adjugate_lane1 + material[5] * jacobian_adjugate_lane2);
        loperand[4] = qw * (material[3] * jacobian_adjugate_lane3 + material[4] * jacobian_adjugate_lane4 + material[5] * jacobian_adjugate_lane5);
        loperand[5] = qw * (material[3] * jacobian_adjugate_lane6 + material[4] * jacobian_adjugate_lane7 + material[5] * jacobian_adjugate_lane8);
        loperand[6] = qw * (material[6] * jacobian_adjugate_lane0 + material[7] * jacobian_adjugate_lane1 + material[8] * jacobian_adjugate_lane2);
        loperand[7] = qw * (material[6] * jacobian_adjugate_lane3 + material[7] * jacobian_adjugate_lane4 + material[8] * jacobian_adjugate_lane5);
        loperand[8] = qw * (material[6] * jacobian_adjugate_lane6 + material[7] * jacobian_adjugate_lane7 + material[8] * jacobian_adjugate_lane8);
            loperand_q[((0 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane] = loperand[0];
            loperand_q[((0 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane] = loperand[1];
            loperand_q[((0 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane] = loperand[2];
            loperand_q[((1 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane] = loperand[3];
            loperand_q[((1 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane] = loperand[4];
            loperand_q[((1 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane] = loperand[5];
            loperand_q[((2 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane] = loperand[6];
            loperand_q[((2 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane] = loperand[7];
            loperand_q[((2 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane] = loperand[8];
        }
    }
    generated_neohookean_ogden_d3_tensor_product_tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, shape_1d, grad_1d, &loperand_q[0 * N_QP * 3 * VECTOR_SIZE], out_streams, 0);
    generated_neohookean_ogden_d3_tensor_product_tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, shape_1d, grad_1d, &loperand_q[1 * N_QP * 3 * VECTOR_SIZE], out_streams, 1);
    generated_neohookean_ogden_d3_tensor_product_tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, shape_1d, grad_1d, &loperand_q[2 * N_QP * 3 * VECTOR_SIZE], out_streams, 2);
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void generated_neohookean_ogden_d3_tensor_product_apply_block(
        const ptrdiff_t nelems,
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
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t mu,
        const scalar_t lmbda,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 3],
        const scalar_t *const SFEM_RESTRICT h_streams[N_SHAPE * 3],
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 3]
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
    static constexpr int N_QP_1D = sfem_generated_integer_root(N_QP, 3);
    static constexpr int N_SHAPE_1D = sfem_generated_integer_root(N_SHAPE, 3);
    static_assert(sfem_generated_ipow(N_QP_1D, 3) == N_QP, "N_QP must be tensor-product compatible");
    static_assert(sfem_generated_ipow(N_SHAPE_1D, 3) == N_SHAPE, "N_SHAPE must be tensor-product compatible");
    scalar_t grad_u_ref_q[N_QP * 9 * VECTOR_SIZE];
    scalar_t grad_h_ref_q[N_QP * 9 * VECTOR_SIZE];
    scalar_t loperand_q[N_QP * 9 * VECTOR_SIZE];
    generated_neohookean_ogden_d3_tensor_product_tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, shape_1d, grad_1d, u_streams, 0, &grad_u_ref_q[0 * N_QP * 3 * VECTOR_SIZE]);
    generated_neohookean_ogden_d3_tensor_product_tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, shape_1d, grad_1d, h_streams, 0, &grad_h_ref_q[0 * N_QP * 3 * VECTOR_SIZE]);
    generated_neohookean_ogden_d3_tensor_product_tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, shape_1d, grad_1d, u_streams, 1, &grad_u_ref_q[1 * N_QP * 3 * VECTOR_SIZE]);
    generated_neohookean_ogden_d3_tensor_product_tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, shape_1d, grad_1d, h_streams, 1, &grad_h_ref_q[1 * N_QP * 3 * VECTOR_SIZE]);
    generated_neohookean_ogden_d3_tensor_product_tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, shape_1d, grad_1d, u_streams, 2, &grad_u_ref_q[2 * N_QP * 3 * VECTOR_SIZE]);
    generated_neohookean_ogden_d3_tensor_product_tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, shape_1d, grad_1d, h_streams, 2, &grad_h_ref_q[2 * N_QP * 3 * VECTOR_SIZE]);
    for (int q = 0; q < N_QP; ++q) {
        const int qx = q % N_QP_1D;
        const int qy = (q / N_QP_1D) % N_QP_1D;
        const int qz = q / (N_QP_1D * N_QP_1D);
        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
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
            scalar_t grad_u_ref[9];
            grad_u_ref[0] = grad_u_ref_q[((0 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[1] = grad_u_ref_q[((0 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[2] = grad_u_ref_q[((0 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            grad_u_ref[3] = grad_u_ref_q[((1 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[4] = grad_u_ref_q[((1 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[5] = grad_u_ref_q[((1 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            grad_u_ref[6] = grad_u_ref_q[((2 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[7] = grad_u_ref_q[((2 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[8] = grad_u_ref_q[((2 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            scalar_t grad_h_ref[9];
            grad_h_ref[0] = grad_h_ref_q[((0 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_h_ref[1] = grad_h_ref_q[((0 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_h_ref[2] = grad_h_ref_q[((0 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            grad_h_ref[3] = grad_h_ref_q[((1 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_h_ref[4] = grad_h_ref_q[((1 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_h_ref[5] = grad_h_ref_q[((1 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            grad_h_ref[6] = grad_h_ref_q[((2 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_h_ref[7] = grad_h_ref_q[((2 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_h_ref[8] = grad_h_ref_q[((2 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            scalar_t grad_u[9];
            scalar_t trial_grad[9];
            const scalar_t inv_jacobian_determinant = 1.0 / jacobian_determinant_lane0;
            grad_u[0] = (grad_u_ref[0] * jacobian_adjugate_lane0 + grad_u_ref[1] * jacobian_adjugate_lane3 + grad_u_ref[2] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            trial_grad[0] = (grad_h_ref[0] * jacobian_adjugate_lane0 + grad_h_ref[1] * jacobian_adjugate_lane3 + grad_h_ref[2] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            grad_u[1] = (grad_u_ref[0] * jacobian_adjugate_lane1 + grad_u_ref[1] * jacobian_adjugate_lane4 + grad_u_ref[2] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            trial_grad[1] = (grad_h_ref[0] * jacobian_adjugate_lane1 + grad_h_ref[1] * jacobian_adjugate_lane4 + grad_h_ref[2] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            grad_u[2] = (grad_u_ref[0] * jacobian_adjugate_lane2 + grad_u_ref[1] * jacobian_adjugate_lane5 + grad_u_ref[2] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            trial_grad[2] = (grad_h_ref[0] * jacobian_adjugate_lane2 + grad_h_ref[1] * jacobian_adjugate_lane5 + grad_h_ref[2] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            grad_u[3] = (grad_u_ref[3] * jacobian_adjugate_lane0 + grad_u_ref[4] * jacobian_adjugate_lane3 + grad_u_ref[5] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            trial_grad[3] = (grad_h_ref[3] * jacobian_adjugate_lane0 + grad_h_ref[4] * jacobian_adjugate_lane3 + grad_h_ref[5] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            grad_u[4] = (grad_u_ref[3] * jacobian_adjugate_lane1 + grad_u_ref[4] * jacobian_adjugate_lane4 + grad_u_ref[5] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            trial_grad[4] = (grad_h_ref[3] * jacobian_adjugate_lane1 + grad_h_ref[4] * jacobian_adjugate_lane4 + grad_h_ref[5] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            grad_u[5] = (grad_u_ref[3] * jacobian_adjugate_lane2 + grad_u_ref[4] * jacobian_adjugate_lane5 + grad_u_ref[5] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            trial_grad[5] = (grad_h_ref[3] * jacobian_adjugate_lane2 + grad_h_ref[4] * jacobian_adjugate_lane5 + grad_h_ref[5] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            grad_u[6] = (grad_u_ref[6] * jacobian_adjugate_lane0 + grad_u_ref[7] * jacobian_adjugate_lane3 + grad_u_ref[8] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            trial_grad[6] = (grad_h_ref[6] * jacobian_adjugate_lane0 + grad_h_ref[7] * jacobian_adjugate_lane3 + grad_h_ref[8] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            grad_u[7] = (grad_u_ref[6] * jacobian_adjugate_lane1 + grad_u_ref[7] * jacobian_adjugate_lane4 + grad_u_ref[8] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            trial_grad[7] = (grad_h_ref[6] * jacobian_adjugate_lane1 + grad_h_ref[7] * jacobian_adjugate_lane4 + grad_h_ref[8] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            grad_u[8] = (grad_u_ref[6] * jacobian_adjugate_lane2 + grad_u_ref[7] * jacobian_adjugate_lane5 + grad_u_ref[8] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            trial_grad[8] = (grad_h_ref[6] * jacobian_adjugate_lane2 + grad_h_ref[7] * jacobian_adjugate_lane5 + grad_h_ref[8] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            scalar_t loperand[9];
        scalar_t material[9];
        const scalar_t weak_mat_tmp0 = grad_u[5]*grad_u[7];
        const scalar_t weak_mat_tmp1 = grad_u[4] + 1;
        const scalar_t weak_mat_tmp2 = grad_u[8] + 1;
        const scalar_t weak_mat_tmp3 = weak_mat_tmp0 - weak_mat_tmp1*weak_mat_tmp2;
        const scalar_t weak_mat_tmp4 = -weak_mat_tmp3;
        const scalar_t weak_mat_tmp5 = grad_u[3]*weak_mat_tmp2;
        const scalar_t weak_mat_tmp6 = grad_u[6]*weak_mat_tmp1;
        const scalar_t weak_mat_tmp7 = grad_u[0] + 1;
        const scalar_t weak_mat_tmp8 = grad_u[1]*grad_u[5]*grad_u[6] - grad_u[1]*weak_mat_tmp5 + grad_u[2]*grad_u[3]*grad_u[7] - grad_u[2]*weak_mat_tmp6 - weak_mat_tmp0*weak_mat_tmp7 + weak_mat_tmp1*weak_mat_tmp2*weak_mat_tmp7;
        const scalar_t weak_mat_tmp9 = pow_m2(weak_mat_tmp8);
        const scalar_t weak_mat_tmp10 = lmbda*weak_mat_tmp9;
        const scalar_t weak_mat_tmp11 = mu*weak_mat_tmp9;
        const scalar_t weak_mat_tmp12 = weak_mat_tmp3*weak_mat_tmp4;
        const scalar_t weak_mat_tmp13 = log(weak_mat_tmp8);
        const scalar_t weak_mat_tmp14 = weak_mat_tmp10*weak_mat_tmp13;
        const scalar_t weak_mat_tmp15 = -grad_u[5]*grad_u[6] + weak_mat_tmp5;
        const scalar_t weak_mat_tmp16 = -weak_mat_tmp15;
        const scalar_t weak_mat_tmp17 = weak_mat_tmp10*weak_mat_tmp4;
        const scalar_t weak_mat_tmp18 = weak_mat_tmp16*weak_mat_tmp17;
        const scalar_t weak_mat_tmp19 = weak_mat_tmp11*weak_mat_tmp4;
        const scalar_t weak_mat_tmp20 = weak_mat_tmp13*weak_mat_tmp17;
        const scalar_t weak_mat_tmp21 = grad_u[3]*grad_u[7] - weak_mat_tmp6;
        const scalar_t weak_mat_tmp22 = weak_mat_tmp17*weak_mat_tmp21;
        const scalar_t weak_mat_tmp23 = -weak_mat_tmp21;
        const scalar_t weak_mat_tmp24 = grad_u[1]*weak_mat_tmp2 - grad_u[2]*grad_u[7];
        const scalar_t weak_mat_tmp25 = -weak_mat_tmp24;
        const scalar_t weak_mat_tmp26 = weak_mat_tmp17*weak_mat_tmp25;
        const scalar_t weak_mat_tmp27 = grad_u[1]*grad_u[5] - grad_u[2]*weak_mat_tmp1;
        const scalar_t weak_mat_tmp28 = weak_mat_tmp17*weak_mat_tmp27;
        const scalar_t weak_mat_tmp29 = -weak_mat_tmp27;
        const scalar_t weak_mat_tmp30 = grad_u[1]*grad_u[6] - grad_u[7]*weak_mat_tmp7;
        const scalar_t weak_mat_tmp31 = -weak_mat_tmp30;
        const scalar_t weak_mat_tmp32 = pow_m1(weak_mat_tmp8);
        const scalar_t weak_mat_tmp33 = mu*weak_mat_tmp32;
        const scalar_t weak_mat_tmp34 = grad_u[7]*weak_mat_tmp33;
        const scalar_t weak_mat_tmp35 = lmbda*weak_mat_tmp13*weak_mat_tmp32;
        const scalar_t weak_mat_tmp36 = grad_u[7]*weak_mat_tmp35;
        const scalar_t weak_mat_tmp37 = weak_mat_tmp17*weak_mat_tmp30 + weak_mat_tmp34 - weak_mat_tmp36;
        const scalar_t weak_mat_tmp38 = grad_u[2]*grad_u[3] - grad_u[5]*weak_mat_tmp7;
        const scalar_t weak_mat_tmp39 = -weak_mat_tmp38;
        const scalar_t weak_mat_tmp40 = grad_u[5]*weak_mat_tmp33;
        const scalar_t weak_mat_tmp41 = grad_u[5]*weak_mat_tmp35;
        const scalar_t weak_mat_tmp42 = weak_mat_tmp17*weak_mat_tmp38 + weak_mat_tmp40 - weak_mat_tmp41;
        const scalar_t weak_mat_tmp43 = grad_u[2]*grad_u[6] - weak_mat_tmp2*weak_mat_tmp7;
        const scalar_t weak_mat_tmp44 = weak_mat_tmp2*weak_mat_tmp33;
        const scalar_t weak_mat_tmp45 = weak_mat_tmp2*weak_mat_tmp35;
        const scalar_t weak_mat_tmp46 = -weak_mat_tmp43;
        const scalar_t weak_mat_tmp47 = weak_mat_tmp17*weak_mat_tmp46 - weak_mat_tmp44 + weak_mat_tmp45;
        const scalar_t weak_mat_tmp48 = grad_u[1]*grad_u[3] - weak_mat_tmp1*weak_mat_tmp7;
        const scalar_t weak_mat_tmp49 = weak_mat_tmp1*weak_mat_tmp33;
        const scalar_t weak_mat_tmp50 = weak_mat_tmp1*weak_mat_tmp35;
        const scalar_t weak_mat_tmp51 = -weak_mat_tmp48;
        const scalar_t weak_mat_tmp52 = weak_mat_tmp17*weak_mat_tmp51 - weak_mat_tmp49 + weak_mat_tmp50;
        const scalar_t weak_mat_tmp53 = weak_mat_tmp11*weak_mat_tmp16;
        const scalar_t weak_mat_tmp54 = weak_mat_tmp10*weak_mat_tmp16;
        const scalar_t weak_mat_tmp55 = weak_mat_tmp13*weak_mat_tmp54;
        const scalar_t weak_mat_tmp56 = weak_mat_tmp21*weak_mat_tmp54;
        const scalar_t weak_mat_tmp57 = weak_mat_tmp38*weak_mat_tmp54;
        const scalar_t weak_mat_tmp58 = weak_mat_tmp46*weak_mat_tmp54;
        const scalar_t weak_mat_tmp59 = grad_u[6]*weak_mat_tmp33;
        const scalar_t weak_mat_tmp60 = grad_u[6]*weak_mat_tmp35;
        const scalar_t weak_mat_tmp61 = weak_mat_tmp30*weak_mat_tmp54 - weak_mat_tmp59 + weak_mat_tmp60;
        const scalar_t weak_mat_tmp62 = weak_mat_tmp27*weak_mat_tmp54 - weak_mat_tmp40 + weak_mat_tmp41;
        const scalar_t weak_mat_tmp63 = weak_mat_tmp25*weak_mat_tmp54 + weak_mat_tmp44 - weak_mat_tmp45;
        const scalar_t weak_mat_tmp64 = grad_u[3]*weak_mat_tmp33;
        const scalar_t weak_mat_tmp65 = grad_u[3]*weak_mat_tmp35;
        const scalar_t weak_mat_tmp66 = weak_mat_tmp51*weak_mat_tmp54 + weak_mat_tmp64 - weak_mat_tmp65;
        const scalar_t weak_mat_tmp67 = weak_mat_tmp11*weak_mat_tmp21;
        const scalar_t weak_mat_tmp68 = weak_mat_tmp10*weak_mat_tmp21;
        const scalar_t weak_mat_tmp69 = weak_mat_tmp13*weak_mat_tmp68;
        const scalar_t weak_mat_tmp70 = weak_mat_tmp30*weak_mat_tmp68;
        const scalar_t weak_mat_tmp71 = weak_mat_tmp51*weak_mat_tmp68;
        const scalar_t weak_mat_tmp72 = weak_mat_tmp25*weak_mat_tmp68 - weak_mat_tmp34 + weak_mat_tmp36;
        const scalar_t weak_mat_tmp73 = weak_mat_tmp38*weak_mat_tmp68 - weak_mat_tmp64 + weak_mat_tmp65;
        const scalar_t weak_mat_tmp74 = weak_mat_tmp27*weak_mat_tmp68 + weak_mat_tmp49 - weak_mat_tmp50;
        const scalar_t weak_mat_tmp75 = weak_mat_tmp46*weak_mat_tmp68 + weak_mat_tmp59 - weak_mat_tmp60;
        const scalar_t weak_mat_tmp76 = weak_mat_tmp11*weak_mat_tmp25;
        const scalar_t weak_mat_tmp77 = weak_mat_tmp10*weak_mat_tmp25;
        const scalar_t weak_mat_tmp78 = weak_mat_tmp13*weak_mat_tmp77;
        const scalar_t weak_mat_tmp79 = weak_mat_tmp30*weak_mat_tmp77;
        const scalar_t weak_mat_tmp80 = weak_mat_tmp27*weak_mat_tmp77;
        const scalar_t weak_mat_tmp81 = weak_mat_tmp46*weak_mat_tmp77;
        const scalar_t weak_mat_tmp82 = grad_u[2]*weak_mat_tmp33;
        const scalar_t weak_mat_tmp83 = grad_u[2]*weak_mat_tmp35;
        const scalar_t weak_mat_tmp84 = weak_mat_tmp38*weak_mat_tmp77 - weak_mat_tmp82 + weak_mat_tmp83;
        const scalar_t weak_mat_tmp85 = grad_u[1]*weak_mat_tmp33;
        const scalar_t weak_mat_tmp86 = grad_u[1]*weak_mat_tmp35;
        const scalar_t weak_mat_tmp87 = weak_mat_tmp51*weak_mat_tmp77 + weak_mat_tmp85 - weak_mat_tmp86;
        const scalar_t weak_mat_tmp88 = weak_mat_tmp11*weak_mat_tmp46;
        const scalar_t weak_mat_tmp89 = weak_mat_tmp10*weak_mat_tmp46;
        const scalar_t weak_mat_tmp90 = weak_mat_tmp13*weak_mat_tmp89;
        const scalar_t weak_mat_tmp91 = weak_mat_tmp30*weak_mat_tmp89;
        const scalar_t weak_mat_tmp92 = weak_mat_tmp38*weak_mat_tmp89;
        const scalar_t weak_mat_tmp93 = weak_mat_tmp27*weak_mat_tmp89 + weak_mat_tmp82 - weak_mat_tmp83;
        const scalar_t weak_mat_tmp94 = weak_mat_tmp33*weak_mat_tmp7;
        const scalar_t weak_mat_tmp95 = weak_mat_tmp35*weak_mat_tmp7;
        const scalar_t weak_mat_tmp96 = weak_mat_tmp51*weak_mat_tmp89 - weak_mat_tmp94 + weak_mat_tmp95;
        const scalar_t weak_mat_tmp97 = weak_mat_tmp11*weak_mat_tmp30;
        const scalar_t weak_mat_tmp98 = weak_mat_tmp10*weak_mat_tmp30;
        const scalar_t weak_mat_tmp99 = weak_mat_tmp13*weak_mat_tmp98;
        const scalar_t weak_mat_tmp100 = weak_mat_tmp51*weak_mat_tmp98;
        const scalar_t weak_mat_tmp101 = weak_mat_tmp27*weak_mat_tmp98 - weak_mat_tmp85 + weak_mat_tmp86;
        const scalar_t weak_mat_tmp102 = weak_mat_tmp38*weak_mat_tmp98 + weak_mat_tmp94 - weak_mat_tmp95;
        const scalar_t weak_mat_tmp103 = weak_mat_tmp11*weak_mat_tmp27;
        const scalar_t weak_mat_tmp104 = weak_mat_tmp10*weak_mat_tmp27;
        const scalar_t weak_mat_tmp105 = weak_mat_tmp104*weak_mat_tmp13;
        const scalar_t weak_mat_tmp106 = weak_mat_tmp104*weak_mat_tmp38;
        const scalar_t weak_mat_tmp107 = weak_mat_tmp104*weak_mat_tmp51;
        const scalar_t weak_mat_tmp108 = weak_mat_tmp11*weak_mat_tmp38;
        const scalar_t weak_mat_tmp109 = weak_mat_tmp10*weak_mat_tmp38;
        const scalar_t weak_mat_tmp110 = weak_mat_tmp109*weak_mat_tmp13;
        const scalar_t weak_mat_tmp111 = weak_mat_tmp109*weak_mat_tmp51;
        const scalar_t weak_mat_tmp112 = weak_mat_tmp11*weak_mat_tmp51;
        const scalar_t weak_mat_tmp113 = weak_mat_tmp14*weak_mat_tmp51;
        material[0] = trial_grad[0]*(mu + weak_mat_tmp10*pow_2(weak_mat_tmp4) - weak_mat_tmp11*weak_mat_tmp12 + weak_mat_tmp12*weak_mat_tmp14) + trial_grad[1]*(-weak_mat_tmp15*weak_mat_tmp19 + weak_mat_tmp15*weak_mat_tmp20 + weak_mat_tmp18) + trial_grad[2]*(-weak_mat_tmp19*weak_mat_tmp23 + weak_mat_tmp20*weak_mat_tmp23 + weak_mat_tmp22) + trial_grad[3]*(-weak_mat_tmp19*weak_mat_tmp24 + weak_mat_tmp20*weak_mat_tmp24 + weak_mat_tmp26) + trial_grad[4]*(-weak_mat_tmp19*weak_mat_tmp43 + weak_mat_tmp20*weak_mat_tmp43 + weak_mat_tmp47) + trial_grad[5]*(-weak_mat_tmp19*weak_mat_tmp31 + weak_mat_tmp20*weak_mat_tmp31 + weak_mat_tmp37) + trial_grad[6]*(-weak_mat_tmp19*weak_mat_tmp29 + weak_mat_tmp20*weak_mat_tmp29 + weak_mat_tmp28) + trial_grad[7]*(-weak_mat_tmp19*weak_mat_tmp39 + weak_mat_tmp20*weak_mat_tmp39 + weak_mat_tmp42) + trial_grad[8]*(-weak_mat_tmp19*weak_mat_tmp48 + weak_mat_tmp20*weak_mat_tmp48 + weak_mat_tmp52);
        material[1] = trial_grad[0]*(weak_mat_tmp18 - weak_mat_tmp3*weak_mat_tmp53 + weak_mat_tmp3*weak_mat_tmp55) + trial_grad[1]*(mu + weak_mat_tmp10*pow_2(weak_mat_tmp16) - weak_mat_tmp15*weak_mat_tmp53 + weak_mat_tmp15*weak_mat_tmp55) + trial_grad[2]*(-weak_mat_tmp23*weak_mat_tmp53 + weak_mat_tmp23*weak_mat_tmp55 + weak_mat_tmp56) + trial_grad[3]*(-weak_mat_tmp24*weak_mat_tmp53 + weak_mat_tmp24*weak_mat_tmp55 + weak_mat_tmp63) + trial_grad[4]*(-weak_mat_tmp43*weak_mat_tmp53 + weak_mat_tmp43*weak_mat_tmp55 + weak_mat_tmp58) + trial_grad[5]*(-weak_mat_tmp31*weak_mat_tmp53 + weak_mat_tmp31*weak_mat_tmp55 + weak_mat_tmp61) + trial_grad[6]*(-weak_mat_tmp29*weak_mat_tmp53 + weak_mat_tmp29*weak_mat_tmp55 + weak_mat_tmp62) + trial_grad[7]*(-weak_mat_tmp39*weak_mat_tmp53 + weak_mat_tmp39*weak_mat_tmp55 + weak_mat_tmp57) + trial_grad[8]*(-weak_mat_tmp48*weak_mat_tmp53 + weak_mat_tmp48*weak_mat_tmp55 + weak_mat_tmp66);
        material[2] = trial_grad[0]*(weak_mat_tmp22 - weak_mat_tmp3*weak_mat_tmp67 + weak_mat_tmp3*weak_mat_tmp69) + trial_grad[1]*(-weak_mat_tmp15*weak_mat_tmp67 + weak_mat_tmp15*weak_mat_tmp69 + weak_mat_tmp56) + trial_grad[2]*(mu + weak_mat_tmp10*pow_2(weak_mat_tmp21) - weak_mat_tmp23*weak_mat_tmp67 + weak_mat_tmp23*weak_mat_tmp69) + trial_grad[3]*(-weak_mat_tmp24*weak_mat_tmp67 + weak_mat_tmp24*weak_mat_tmp69 + weak_mat_tmp72) + trial_grad[4]*(-weak_mat_tmp43*weak_mat_tmp67 + weak_mat_tmp43*weak_mat_tmp69 + weak_mat_tmp75) + trial_grad[5]*(-weak_mat_tmp31*weak_mat_tmp67 + weak_mat_tmp31*weak_mat_tmp69 + weak_mat_tmp70) + trial_grad[6]*(-weak_mat_tmp29*weak_mat_tmp67 + weak_mat_tmp29*weak_mat_tmp69 + weak_mat_tmp74) + trial_grad[7]*(-weak_mat_tmp39*weak_mat_tmp67 + weak_mat_tmp39*weak_mat_tmp69 + weak_mat_tmp73) + trial_grad[8]*(-weak_mat_tmp48*weak_mat_tmp67 + weak_mat_tmp48*weak_mat_tmp69 + weak_mat_tmp71);
        material[3] = trial_grad[0]*(weak_mat_tmp26 - weak_mat_tmp3*weak_mat_tmp76 + weak_mat_tmp3*weak_mat_tmp78) + trial_grad[1]*(-weak_mat_tmp15*weak_mat_tmp76 + weak_mat_tmp15*weak_mat_tmp78 + weak_mat_tmp63) + trial_grad[2]*(-weak_mat_tmp23*weak_mat_tmp76 + weak_mat_tmp23*weak_mat_tmp78 + weak_mat_tmp72) + trial_grad[3]*(mu + weak_mat_tmp10*pow_2(weak_mat_tmp25) - weak_mat_tmp24*weak_mat_tmp76 + weak_mat_tmp24*weak_mat_tmp78) + trial_grad[4]*(-weak_mat_tmp43*weak_mat_tmp76 + weak_mat_tmp43*weak_mat_tmp78 + weak_mat_tmp81) + trial_grad[5]*(-weak_mat_tmp31*weak_mat_tmp76 + weak_mat_tmp31*weak_mat_tmp78 + weak_mat_tmp79) + trial_grad[6]*(-weak_mat_tmp29*weak_mat_tmp76 + weak_mat_tmp29*weak_mat_tmp78 + weak_mat_tmp80) + trial_grad[7]*(-weak_mat_tmp39*weak_mat_tmp76 + weak_mat_tmp39*weak_mat_tmp78 + weak_mat_tmp84) + trial_grad[8]*(-weak_mat_tmp48*weak_mat_tmp76 + weak_mat_tmp48*weak_mat_tmp78 + weak_mat_tmp87);
        material[4] = trial_grad[0]*(-weak_mat_tmp3*weak_mat_tmp88 + weak_mat_tmp3*weak_mat_tmp90 + weak_mat_tmp47) + trial_grad[1]*(-weak_mat_tmp15*weak_mat_tmp88 + weak_mat_tmp15*weak_mat_tmp90 + weak_mat_tmp58) + trial_grad[2]*(-weak_mat_tmp23*weak_mat_tmp88 + weak_mat_tmp23*weak_mat_tmp90 + weak_mat_tmp75) + trial_grad[3]*(-weak_mat_tmp24*weak_mat_tmp88 + weak_mat_tmp24*weak_mat_tmp90 + weak_mat_tmp81) + trial_grad[4]*(mu + weak_mat_tmp10*pow_2(weak_mat_tmp46) - weak_mat_tmp43*weak_mat_tmp88 + weak_mat_tmp43*weak_mat_tmp90) + trial_grad[5]*(-weak_mat_tmp31*weak_mat_tmp88 + weak_mat_tmp31*weak_mat_tmp90 + weak_mat_tmp91) + trial_grad[6]*(-weak_mat_tmp29*weak_mat_tmp88 + weak_mat_tmp29*weak_mat_tmp90 + weak_mat_tmp93) + trial_grad[7]*(-weak_mat_tmp39*weak_mat_tmp88 + weak_mat_tmp39*weak_mat_tmp90 + weak_mat_tmp92) + trial_grad[8]*(-weak_mat_tmp48*weak_mat_tmp88 + weak_mat_tmp48*weak_mat_tmp90 + weak_mat_tmp96);
        material[5] = trial_grad[0]*(-weak_mat_tmp3*weak_mat_tmp97 + weak_mat_tmp3*weak_mat_tmp99 + weak_mat_tmp37) + trial_grad[1]*(-weak_mat_tmp15*weak_mat_tmp97 + weak_mat_tmp15*weak_mat_tmp99 + weak_mat_tmp61) + trial_grad[2]*(-weak_mat_tmp23*weak_mat_tmp97 + weak_mat_tmp23*weak_mat_tmp99 + weak_mat_tmp70) + trial_grad[3]*(-weak_mat_tmp24*weak_mat_tmp97 + weak_mat_tmp24*weak_mat_tmp99 + weak_mat_tmp79) + trial_grad[4]*(-weak_mat_tmp43*weak_mat_tmp97 + weak_mat_tmp43*weak_mat_tmp99 + weak_mat_tmp91) + trial_grad[5]*(mu + weak_mat_tmp10*pow_2(weak_mat_tmp30) - weak_mat_tmp31*weak_mat_tmp97 + weak_mat_tmp31*weak_mat_tmp99) + trial_grad[6]*(weak_mat_tmp101 - weak_mat_tmp29*weak_mat_tmp97 + weak_mat_tmp29*weak_mat_tmp99) + trial_grad[7]*(weak_mat_tmp102 - weak_mat_tmp39*weak_mat_tmp97 + weak_mat_tmp39*weak_mat_tmp99) + trial_grad[8]*(weak_mat_tmp100 - weak_mat_tmp48*weak_mat_tmp97 + weak_mat_tmp48*weak_mat_tmp99);
        material[6] = trial_grad[0]*(-weak_mat_tmp103*weak_mat_tmp3 + weak_mat_tmp105*weak_mat_tmp3 + weak_mat_tmp28) + trial_grad[1]*(-weak_mat_tmp103*weak_mat_tmp15 + weak_mat_tmp105*weak_mat_tmp15 + weak_mat_tmp62) + trial_grad[2]*(-weak_mat_tmp103*weak_mat_tmp23 + weak_mat_tmp105*weak_mat_tmp23 + weak_mat_tmp74) + trial_grad[3]*(-weak_mat_tmp103*weak_mat_tmp24 + weak_mat_tmp105*weak_mat_tmp24 + weak_mat_tmp80) + trial_grad[4]*(-weak_mat_tmp103*weak_mat_tmp43 + weak_mat_tmp105*weak_mat_tmp43 + weak_mat_tmp93) + trial_grad[5]*(weak_mat_tmp101 - weak_mat_tmp103*weak_mat_tmp31 + weak_mat_tmp105*weak_mat_tmp31) + trial_grad[6]*(mu + weak_mat_tmp10*pow_2(weak_mat_tmp27) - weak_mat_tmp103*weak_mat_tmp29 + weak_mat_tmp105*weak_mat_tmp29) + trial_grad[7]*(-weak_mat_tmp103*weak_mat_tmp39 + weak_mat_tmp105*weak_mat_tmp39 + weak_mat_tmp106) + trial_grad[8]*(-weak_mat_tmp103*weak_mat_tmp48 + weak_mat_tmp105*weak_mat_tmp48 + weak_mat_tmp107);
        material[7] = trial_grad[0]*(-weak_mat_tmp108*weak_mat_tmp3 + weak_mat_tmp110*weak_mat_tmp3 + weak_mat_tmp42) + trial_grad[1]*(-weak_mat_tmp108*weak_mat_tmp15 + weak_mat_tmp110*weak_mat_tmp15 + weak_mat_tmp57) + trial_grad[2]*(-weak_mat_tmp108*weak_mat_tmp23 + weak_mat_tmp110*weak_mat_tmp23 + weak_mat_tmp73) + trial_grad[3]*(-weak_mat_tmp108*weak_mat_tmp24 + weak_mat_tmp110*weak_mat_tmp24 + weak_mat_tmp84) + trial_grad[4]*(-weak_mat_tmp108*weak_mat_tmp43 + weak_mat_tmp110*weak_mat_tmp43 + weak_mat_tmp92) + trial_grad[5]*(weak_mat_tmp102 - weak_mat_tmp108*weak_mat_tmp31 + weak_mat_tmp110*weak_mat_tmp31) + trial_grad[6]*(weak_mat_tmp106 - weak_mat_tmp108*weak_mat_tmp29 + weak_mat_tmp110*weak_mat_tmp29) + trial_grad[7]*(mu + weak_mat_tmp10*pow_2(weak_mat_tmp38) - weak_mat_tmp108*weak_mat_tmp39 + weak_mat_tmp110*weak_mat_tmp39) + trial_grad[8]*(-weak_mat_tmp108*weak_mat_tmp48 + weak_mat_tmp110*weak_mat_tmp48 + weak_mat_tmp111);
        material[8] = trial_grad[0]*(-weak_mat_tmp112*weak_mat_tmp3 + weak_mat_tmp113*weak_mat_tmp3 + weak_mat_tmp52) + trial_grad[1]*(-weak_mat_tmp112*weak_mat_tmp15 + weak_mat_tmp113*weak_mat_tmp15 + weak_mat_tmp66) + trial_grad[2]*(-weak_mat_tmp112*weak_mat_tmp23 + weak_mat_tmp113*weak_mat_tmp23 + weak_mat_tmp71) + trial_grad[3]*(-weak_mat_tmp112*weak_mat_tmp24 + weak_mat_tmp113*weak_mat_tmp24 + weak_mat_tmp87) + trial_grad[4]*(-weak_mat_tmp112*weak_mat_tmp43 + weak_mat_tmp113*weak_mat_tmp43 + weak_mat_tmp96) + trial_grad[5]*(weak_mat_tmp100 - weak_mat_tmp112*weak_mat_tmp31 + weak_mat_tmp113*weak_mat_tmp31) + trial_grad[6]*(weak_mat_tmp107 - weak_mat_tmp112*weak_mat_tmp29 + weak_mat_tmp113*weak_mat_tmp29) + trial_grad[7]*(weak_mat_tmp111 - weak_mat_tmp112*weak_mat_tmp39 + weak_mat_tmp113*weak_mat_tmp39) + trial_grad[8]*(mu + weak_mat_tmp10*pow_2(weak_mat_tmp51) - weak_mat_tmp112*weak_mat_tmp48 + weak_mat_tmp113*weak_mat_tmp48);
        loperand[0] = qw * (material[0] * jacobian_adjugate_lane0 + material[1] * jacobian_adjugate_lane1 + material[2] * jacobian_adjugate_lane2);
        loperand[1] = qw * (material[0] * jacobian_adjugate_lane3 + material[1] * jacobian_adjugate_lane4 + material[2] * jacobian_adjugate_lane5);
        loperand[2] = qw * (material[0] * jacobian_adjugate_lane6 + material[1] * jacobian_adjugate_lane7 + material[2] * jacobian_adjugate_lane8);
        loperand[3] = qw * (material[3] * jacobian_adjugate_lane0 + material[4] * jacobian_adjugate_lane1 + material[5] * jacobian_adjugate_lane2);
        loperand[4] = qw * (material[3] * jacobian_adjugate_lane3 + material[4] * jacobian_adjugate_lane4 + material[5] * jacobian_adjugate_lane5);
        loperand[5] = qw * (material[3] * jacobian_adjugate_lane6 + material[4] * jacobian_adjugate_lane7 + material[5] * jacobian_adjugate_lane8);
        loperand[6] = qw * (material[6] * jacobian_adjugate_lane0 + material[7] * jacobian_adjugate_lane1 + material[8] * jacobian_adjugate_lane2);
        loperand[7] = qw * (material[6] * jacobian_adjugate_lane3 + material[7] * jacobian_adjugate_lane4 + material[8] * jacobian_adjugate_lane5);
        loperand[8] = qw * (material[6] * jacobian_adjugate_lane6 + material[7] * jacobian_adjugate_lane7 + material[8] * jacobian_adjugate_lane8);
            loperand_q[((0 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane] = loperand[0];
            loperand_q[((0 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane] = loperand[1];
            loperand_q[((0 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane] = loperand[2];
            loperand_q[((1 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane] = loperand[3];
            loperand_q[((1 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane] = loperand[4];
            loperand_q[((1 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane] = loperand[5];
            loperand_q[((2 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane] = loperand[6];
            loperand_q[((2 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane] = loperand[7];
            loperand_q[((2 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane] = loperand[8];
        }
    }
    generated_neohookean_ogden_d3_tensor_product_tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, shape_1d, grad_1d, &loperand_q[0 * N_QP * 3 * VECTOR_SIZE], out_streams, 0);
    generated_neohookean_ogden_d3_tensor_product_tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, shape_1d, grad_1d, &loperand_q[1 * N_QP * 3 * VECTOR_SIZE], out_streams, 1);
    generated_neohookean_ogden_d3_tensor_product_tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE>(nelems, shape_1d, grad_1d, &loperand_q[2 * N_QP * 3 * VECTOR_SIZE], out_streams, 2);
}

} // namespace codegen
} // namespace sfem

#endif
