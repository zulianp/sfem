#ifndef GENERATED_TWO_PHASE_FLOW_D3_TENSOR_PRODUCT_LOCAL_HPP
#define GENERATED_TWO_PHASE_FLOW_D3_TENSOR_PRODUCT_LOCAL_HPP

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

namespace sfem {
namespace codegen {

static constexpr int generated_two_phase_flow_d3_tensor_product_ipow(const int b, const int e) {
    return e == 0 ? 1 : b * generated_two_phase_flow_d3_tensor_product_ipow(b, e - 1);
}
static constexpr int generated_two_phase_flow_d3_tensor_product_integer_root_search(const int v, const int e, const int c) {
    return generated_two_phase_flow_d3_tensor_product_ipow(c, e) >= v ? c : generated_two_phase_flow_d3_tensor_product_integer_root_search(v, e, c + 1);
}
static constexpr int generated_two_phase_flow_d3_tensor_product_integer_root(const int v, const int e) {
    return generated_two_phase_flow_d3_tensor_product_integer_root_search(v, e, 1);
}

template <int S>
static SFEM_INLINE int generated_two_phase_flow_d3_tensor_product_tensor_index(const int sx, const int sy, const int sz = 0) {
    return S == 2 ? sx + sy * (3 - 2 * sx) + 4 * sz : sx + S * (sy + S * sz);
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int N_FIELDS>
static SFEM_INLINE void generated_two_phase_flow_d3_tensor_product_tensor_evaluate(
        const ptrdiff_t nelems, const scalar_t *const shape_1d, const scalar_t *const grad_1d,
        const scalar_t *const SFEM_RESTRICT streams[N_FIELDS * N_SHAPE], scalar_t *const value, scalar_t *const gradient) {
    static constexpr int Q = generated_two_phase_flow_d3_tensor_product_integer_root(N_QP, 3);
    static constexpr int S = generated_two_phase_flow_d3_tensor_product_integer_root(N_SHAPE, 3);
    scalar_t vx[N_FIELDS * Q * S * S * VECTOR_SIZE], gx[N_FIELDS * Q * S * S * VECTOR_SIZE];
    scalar_t vxy[N_FIELDS * Q * Q * S * VECTOR_SIZE], g0xy[N_FIELDS * Q * Q * S * VECTOR_SIZE], g1xy[N_FIELDS * Q * Q * S * VECTOR_SIZE];
    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) for (int sz = 0; sz < S; ++sz) {
#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            scalar_t v = 0, g = 0;
            for (int sx = 0; sx < S; ++sx) {
                const int s = generated_two_phase_flow_d3_tensor_product_tensor_index<S>(sx, sy, sz);
                const scalar_t u = streams[s * N_FIELDS + f][lane];
                v += u * shape_1d[qx * S + sx]; g += u * grad_1d[qx * S + sx];
            }
            const int i = (((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane;
            vx[i] = v; gx[i] = g;
        }
    }
    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int qy = 0; qy < Q; ++qy) for (int sz = 0; sz < S; ++sz) {
#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            scalar_t v = 0, g0 = 0, g1 = 0;
            for (int sy = 0; sy < S; ++sy) {
                const int i = (((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane;
                v += vx[i] * shape_1d[qy * S + sy]; g0 += gx[i] * shape_1d[qy * S + sy]; g1 += vx[i] * grad_1d[qy * S + sy];
            }
            const int j = (((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane;
            vxy[j] = v; g0xy[j] = g0; g1xy[j] = g1;
        }
    }
    for (int f = 0; f < N_FIELDS; ++f) for (int qz = 0; qz < Q; ++qz) for (int qy = 0; qy < Q; ++qy) for (int qx = 0; qx < Q; ++qx) {
        const int q = qx + Q * (qy + Q * qz);
#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
            scalar_t v = 0, g0 = 0, g1 = 0, g2 = 0;
            for (int sz = 0; sz < S; ++sz) {
                const int j = (((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane;
                v += vxy[j] * shape_1d[qz * S + sz]; g0 += g0xy[j] * shape_1d[qz * S + sz];
                g1 += g1xy[j] * shape_1d[qz * S + sz]; g2 += vxy[j] * grad_1d[qz * S + sz];
            }
            value[(f * N_QP + q) * VECTOR_SIZE + lane] = v;
            gradient[((f * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane] = g0;
            gradient[((f * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane] = g1;
            gradient[((f * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane] = g2;
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE, int N_FIELDS>
static SFEM_INLINE void generated_two_phase_flow_d3_tensor_product_tensor_integrate(
        const ptrdiff_t nelems, const scalar_t *const shape_1d, const scalar_t *const grad_1d,
        const scalar_t *const value_coeff, const scalar_t *const grad_coeff, scalar_t *const SFEM_RESTRICT output[N_FIELDS * N_SHAPE]) {
    static constexpr int Q = generated_two_phase_flow_d3_tensor_product_integer_root(N_QP, 3), S = generated_two_phase_flow_d3_tensor_product_integer_root(N_SHAPE, 3);
    scalar_t z0[N_FIELDS * Q * Q * S * VECTOR_SIZE], z1[N_FIELDS * Q * Q * S * VECTOR_SIZE], z2[N_FIELDS * Q * Q * S * VECTOR_SIZE];
    scalar_t yz0[N_FIELDS * Q * S * S * VECTOR_SIZE], yz1[N_FIELDS * Q * S * S * VECTOR_SIZE];
    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int qy = 0; qy < Q; ++qy) for (int sz = 0; sz < S; ++sz) {
#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) { scalar_t a = 0, b = 0, c = 0;
            for (int qz = 0; qz < Q; ++qz) { const int q = qx + Q * (qy + Q * qz);
                a += value_coeff[(f * N_QP + q) * VECTOR_SIZE + lane] * shape_1d[qz * S + sz]
                   + grad_coeff[((f * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane] * grad_1d[qz * S + sz];
                b += grad_coeff[((f * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane] * shape_1d[qz * S + sz];
                c += grad_coeff[((f * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane] * shape_1d[qz * S + sz]; }
            const int i = (((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane; z0[i] = a; z1[i] = b; z2[i] = c;
        }
    }
    for (int f = 0; f < N_FIELDS; ++f) for (int qx = 0; qx < Q; ++qx) for (int sy = 0; sy < S; ++sy) for (int sz = 0; sz < S; ++sz) {
#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) { scalar_t a = 0, b = 0;
            for (int qy = 0; qy < Q; ++qy) { const int i = (((f * Q + qx) * Q + qy) * S + sz) * VECTOR_SIZE + lane;
                a += z0[i] * shape_1d[qy * S + sy] + z2[i] * grad_1d[qy * S + sy];
                b += z1[i] * shape_1d[qy * S + sy]; }
            const int j = (((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane; yz0[j] = a; yz1[j] = b;
        }
    }
    for (int f = 0; f < N_FIELDS; ++f) for (int sz = 0; sz < S; ++sz) for (int sy = 0; sy < S; ++sy) for (int sx = 0; sx < S; ++sx) {
        const int s = generated_two_phase_flow_d3_tensor_product_tensor_index<S>(sx, sy, sz);
#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) { scalar_t v = 0;
            for (int qx = 0; qx < Q; ++qx) { const int j = (((f * Q + qx) * S + sy) * S + sz) * VECTOR_SIZE + lane;
                v += yz0[j] * shape_1d[qx * S + sx] + yz1[j] * grad_1d[qx * S + sx]; }
            output[s * N_FIELDS + f][lane] += v;
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void generated_two_phase_flow_d3_tensor_product_residual_block(
        const ptrdiff_t nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t *const SFEM_RESTRICT current[2 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT previous[2 * N_SHAPE],
        const scalar_t porosity,
        const scalar_t S_res,
        const scalar_t P_r,
        const scalar_t m,
        const scalar_t rho_w0,
        const scalar_t kappa_T,
        const scalar_t p_wr,
        const scalar_t M_c,
        const scalar_t Z,
        const scalar_t R,
        const scalar_t T,
        const scalar_t mu_w,
        const scalar_t mu_c,
        const scalar_t C_kw1,
        const scalar_t C_ka1,
        const scalar_t C_ka2,
        const scalar_t dt,
        const scalar_t K_0,
        const scalar_t K_1,
        const scalar_t K_2,
        const scalar_t K_3,
        const scalar_t K_4,
        const scalar_t K_5,
        const scalar_t K_6,
        const scalar_t K_7,
        const scalar_t K_8,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
    scalar_t current_value[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t current_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    scalar_t previous_value[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t previous_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    generated_two_phase_flow_d3_tensor_product_tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, N_FIELDS>(
            nelems, shape_1d, grad_1d, current, current_value, current_grad_ref);
    generated_two_phase_flow_d3_tensor_product_tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, N_FIELDS>(
            nelems, shape_1d, grad_1d, previous, previous_value, previous_grad_ref);
    scalar_t value_coeff[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t grad_coeff_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    static constexpr int Q = generated_two_phase_flow_d3_tensor_product_integer_root(N_QP, DIM);
    for (int q = 0; q < N_QP; ++q) {
        const int qx = q % Q;
        const int qy = (q / Q) % Q;
        const int qz = q / (Q * Q);
        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
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
            const scalar_t p_w = current_value[(0 * N_QP + q) * VECTOR_SIZE + lane];
            const scalar_t p_w_grad_0_ref = current_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t p_w_grad_1_ref = current_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t p_w_grad_2_ref = current_grad_ref[((0 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
            const scalar_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
            const scalar_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
            const scalar_t p_w_old = previous_value[(0 * N_QP + q) * VECTOR_SIZE + lane];
            const scalar_t p_w_old_grad_0_ref = previous_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t p_w_old_grad_1_ref = previous_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t p_w_old_grad_2_ref = previous_grad_ref[((0 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t p_w_old_grad_0 = (p_w_old_grad_0_ref * adj0 + p_w_old_grad_1_ref * adj3 + p_w_old_grad_2_ref * adj6) / det;
            const scalar_t p_w_old_grad_1 = (p_w_old_grad_0_ref * adj1 + p_w_old_grad_1_ref * adj4 + p_w_old_grad_2_ref * adj7) / det;
            const scalar_t p_w_old_grad_2 = (p_w_old_grad_0_ref * adj2 + p_w_old_grad_1_ref * adj5 + p_w_old_grad_2_ref * adj8) / det;
            const scalar_t p_c = current_value[(1 * N_QP + q) * VECTOR_SIZE + lane];
            const scalar_t p_c_grad_0_ref = current_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t p_c_grad_1_ref = current_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t p_c_grad_2_ref = current_grad_ref[((1 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
            const scalar_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
            const scalar_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
            const scalar_t p_c_old = previous_value[(1 * N_QP + q) * VECTOR_SIZE + lane];
            const scalar_t p_c_old_grad_0_ref = previous_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t p_c_old_grad_1_ref = previous_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t p_c_old_grad_2_ref = previous_grad_ref[((1 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t p_c_old_grad_0 = (p_c_old_grad_0_ref * adj0 + p_c_old_grad_1_ref * adj3 + p_c_old_grad_2_ref * adj6) / det;
            const scalar_t p_c_old_grad_1 = (p_c_old_grad_0_ref * adj1 + p_c_old_grad_1_ref * adj4 + p_c_old_grad_2_ref * adj7) / det;
            const scalar_t p_c_old_grad_2 = (p_c_old_grad_0_ref * adj2 + p_c_old_grad_1_ref * adj5 + p_c_old_grad_2_ref * adj8) / det;
            const scalar_t residual_tmp0 = S_res - 1;
            const scalar_t residual_tmp1 = -residual_tmp0;
            const scalar_t residual_tmp2 = pow_m1(P_r);
            const scalar_t residual_tmp3 = 1 - 1/m;
            const scalar_t residual_tmp4 = pow(pow(residual_tmp2*(p_c - p_w), m) + 1, -residual_tmp3);
            const scalar_t residual_tmp5 = residual_tmp1*residual_tmp4;
            const scalar_t residual_tmp6 = S_res + residual_tmp5;
            const scalar_t residual_tmp7 = -p_wr;
            const scalar_t residual_tmp8 = rho_w0*exp(kappa_T*(p_w + residual_tmp7));
            const scalar_t residual_tmp9 = residual_tmp1*pow(pow(residual_tmp2*(p_c_old - p_w_old), m) + 1, -residual_tmp3);
            const scalar_t residual_tmp10 = porosity/dt;
            const scalar_t residual_tmp11 = sqrt(residual_tmp6)*residual_tmp8*pow_2(1 - pow(1 - pow(residual_tmp6, pow_m1(C_kw1)), C_kw1))/mu_w;
            const scalar_t residual_tmp12 = p_w_grad_0*residual_tmp11;
            const scalar_t residual_tmp13 = p_w_grad_1*residual_tmp11;
            const scalar_t residual_tmp14 = p_w_grad_2*residual_tmp11;
            const scalar_t residual_tmp15 = M_c/(R*T*Z);
            const scalar_t residual_tmp16 = p_c*residual_tmp15;
            const scalar_t residual_tmp17 = residual_tmp16*pow(1 - residual_tmp4, C_ka1)*(1 - pow(residual_tmp4, C_ka2))/mu_c;
            const scalar_t residual_tmp18 = p_c_grad_0*residual_tmp17;
            const scalar_t residual_tmp19 = p_c_grad_1*residual_tmp17;
            const scalar_t residual_tmp20 = p_c_grad_2*residual_tmp17;
            const scalar_t value_coeff0 = residual_tmp10*(residual_tmp6*residual_tmp8 - rho_w0*(S_res + residual_tmp9)*exp(kappa_T*(p_w_old + residual_tmp7)));
            const scalar_t grad_coeff0_0 = K_0*residual_tmp12 + K_1*residual_tmp13 + K_2*residual_tmp14;
            const scalar_t grad_coeff0_1 = K_3*residual_tmp12 + K_4*residual_tmp13 + K_5*residual_tmp14;
            const scalar_t grad_coeff0_2 = K_6*residual_tmp12 + K_7*residual_tmp13 + K_8*residual_tmp14;
            const scalar_t value_coeff1 = residual_tmp10*(-p_c_old*residual_tmp15*(-residual_tmp0 - residual_tmp9) + residual_tmp16*(-residual_tmp0 - residual_tmp5));
            const scalar_t grad_coeff1_0 = K_0*residual_tmp18 + K_1*residual_tmp19 + K_2*residual_tmp20;
            const scalar_t grad_coeff1_1 = K_3*residual_tmp18 + K_4*residual_tmp19 + K_5*residual_tmp20;
            const scalar_t grad_coeff1_2 = K_6*residual_tmp18 + K_7*residual_tmp19 + K_8*residual_tmp20;
            value_coeff[(0 * N_QP + q) * VECTOR_SIZE + lane] = qw * det * value_coeff0;
            grad_coeff_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1 + adj2 * grad_coeff0_2);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj3 * grad_coeff0_0 + adj4 * grad_coeff0_1 + adj5 * grad_coeff0_2);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane] = qw * (adj6 * grad_coeff0_0 + adj7 * grad_coeff0_1 + adj8 * grad_coeff0_2);
            value_coeff[(1 * N_QP + q) * VECTOR_SIZE + lane] = qw * det * value_coeff1;
            grad_coeff_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1 + adj2 * grad_coeff1_2);
            grad_coeff_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj3 * grad_coeff1_0 + adj4 * grad_coeff1_1 + adj5 * grad_coeff1_2);
            grad_coeff_ref[((1 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane] = qw * (adj6 * grad_coeff1_0 + adj7 * grad_coeff1_1 + adj8 * grad_coeff1_2);
        }
    }
    generated_two_phase_flow_d3_tensor_product_tensor_integrate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, N_FIELDS>(
            nelems, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void generated_two_phase_flow_d3_tensor_product_jacobian_action_block(
        const ptrdiff_t nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT adjugate[9],
        const scalar_t *const SFEM_RESTRICT determinant,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t *const SFEM_RESTRICT current[2 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT previous[2 * N_SHAPE],
        const scalar_t *const SFEM_RESTRICT direction[2 * N_SHAPE],
        const scalar_t porosity,
        const scalar_t S_res,
        const scalar_t P_r,
        const scalar_t m,
        const scalar_t rho_w0,
        const scalar_t kappa_T,
        const scalar_t p_wr,
        const scalar_t M_c,
        const scalar_t Z,
        const scalar_t R,
        const scalar_t T,
        const scalar_t mu_w,
        const scalar_t mu_c,
        const scalar_t C_kw1,
        const scalar_t C_ka1,
        const scalar_t C_ka2,
        const scalar_t dt,
        const scalar_t K_0,
        const scalar_t K_1,
        const scalar_t K_2,
        const scalar_t K_3,
        const scalar_t K_4,
        const scalar_t K_5,
        const scalar_t K_6,
        const scalar_t K_7,
        const scalar_t K_8,
        scalar_t *const SFEM_RESTRICT output[2 * N_SHAPE]
) {
    static constexpr int DIM = 3;
    static constexpr int N_FIELDS = 2;
    scalar_t current_value[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t current_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    scalar_t previous_value[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t previous_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    generated_two_phase_flow_d3_tensor_product_tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, N_FIELDS>(
            nelems, shape_1d, grad_1d, current, current_value, current_grad_ref);
    generated_two_phase_flow_d3_tensor_product_tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, N_FIELDS>(
            nelems, shape_1d, grad_1d, previous, previous_value, previous_grad_ref);
    scalar_t direction_value[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t direction_grad_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    generated_two_phase_flow_d3_tensor_product_tensor_evaluate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, N_FIELDS>(
            nelems, shape_1d, grad_1d, direction, direction_value, direction_grad_ref);
    scalar_t value_coeff[N_FIELDS * N_QP * VECTOR_SIZE];
    scalar_t grad_coeff_ref[N_FIELDS * N_QP * DIM * VECTOR_SIZE];
    static constexpr int Q = generated_two_phase_flow_d3_tensor_product_integer_root(N_QP, DIM);
    for (int q = 0; q < N_QP; ++q) {
        const int qx = q % Q;
        const int qy = (q / Q) % Q;
        const int qz = q / (Q * Q);
        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
#pragma omp simd
        for (ptrdiff_t lane = 0; lane < nelems; ++lane) {
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
            const scalar_t p_w = current_value[(0 * N_QP + q) * VECTOR_SIZE + lane];
            const scalar_t p_w_grad_0_ref = current_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t p_w_grad_1_ref = current_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t p_w_grad_2_ref = current_grad_ref[((0 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t p_w_grad_0 = (p_w_grad_0_ref * adj0 + p_w_grad_1_ref * adj3 + p_w_grad_2_ref * adj6) / det;
            const scalar_t p_w_grad_1 = (p_w_grad_0_ref * adj1 + p_w_grad_1_ref * adj4 + p_w_grad_2_ref * adj7) / det;
            const scalar_t p_w_grad_2 = (p_w_grad_0_ref * adj2 + p_w_grad_1_ref * adj5 + p_w_grad_2_ref * adj8) / det;
            const scalar_t p_w_old = previous_value[(0 * N_QP + q) * VECTOR_SIZE + lane];
            const scalar_t p_w_old_grad_0_ref = previous_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t p_w_old_grad_1_ref = previous_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t p_w_old_grad_2_ref = previous_grad_ref[((0 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t p_w_old_grad_0 = (p_w_old_grad_0_ref * adj0 + p_w_old_grad_1_ref * adj3 + p_w_old_grad_2_ref * adj6) / det;
            const scalar_t p_w_old_grad_1 = (p_w_old_grad_0_ref * adj1 + p_w_old_grad_1_ref * adj4 + p_w_old_grad_2_ref * adj7) / det;
            const scalar_t p_w_old_grad_2 = (p_w_old_grad_0_ref * adj2 + p_w_old_grad_1_ref * adj5 + p_w_old_grad_2_ref * adj8) / det;
            const scalar_t p_w_direction = direction_value[(0 * N_QP + q) * VECTOR_SIZE + lane];
            const scalar_t p_w_direction_grad_0_ref = direction_grad_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t p_w_direction_grad_1_ref = direction_grad_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t p_w_direction_grad_2_ref = direction_grad_ref[((0 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t p_w_direction_grad_0 = (p_w_direction_grad_0_ref * adj0 + p_w_direction_grad_1_ref * adj3 + p_w_direction_grad_2_ref * adj6) / det;
            const scalar_t p_w_direction_grad_1 = (p_w_direction_grad_0_ref * adj1 + p_w_direction_grad_1_ref * adj4 + p_w_direction_grad_2_ref * adj7) / det;
            const scalar_t p_w_direction_grad_2 = (p_w_direction_grad_0_ref * adj2 + p_w_direction_grad_1_ref * adj5 + p_w_direction_grad_2_ref * adj8) / det;
            const scalar_t p_c = current_value[(1 * N_QP + q) * VECTOR_SIZE + lane];
            const scalar_t p_c_grad_0_ref = current_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t p_c_grad_1_ref = current_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t p_c_grad_2_ref = current_grad_ref[((1 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t p_c_grad_0 = (p_c_grad_0_ref * adj0 + p_c_grad_1_ref * adj3 + p_c_grad_2_ref * adj6) / det;
            const scalar_t p_c_grad_1 = (p_c_grad_0_ref * adj1 + p_c_grad_1_ref * adj4 + p_c_grad_2_ref * adj7) / det;
            const scalar_t p_c_grad_2 = (p_c_grad_0_ref * adj2 + p_c_grad_1_ref * adj5 + p_c_grad_2_ref * adj8) / det;
            const scalar_t p_c_old = previous_value[(1 * N_QP + q) * VECTOR_SIZE + lane];
            const scalar_t p_c_old_grad_0_ref = previous_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t p_c_old_grad_1_ref = previous_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t p_c_old_grad_2_ref = previous_grad_ref[((1 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t p_c_old_grad_0 = (p_c_old_grad_0_ref * adj0 + p_c_old_grad_1_ref * adj3 + p_c_old_grad_2_ref * adj6) / det;
            const scalar_t p_c_old_grad_1 = (p_c_old_grad_0_ref * adj1 + p_c_old_grad_1_ref * adj4 + p_c_old_grad_2_ref * adj7) / det;
            const scalar_t p_c_old_grad_2 = (p_c_old_grad_0_ref * adj2 + p_c_old_grad_1_ref * adj5 + p_c_old_grad_2_ref * adj8) / det;
            const scalar_t p_c_direction = direction_value[(1 * N_QP + q) * VECTOR_SIZE + lane];
            const scalar_t p_c_direction_grad_0_ref = direction_grad_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane];
            const scalar_t p_c_direction_grad_1_ref = direction_grad_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane];
            const scalar_t p_c_direction_grad_2_ref = direction_grad_ref[((1 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane];
            const scalar_t p_c_direction_grad_0 = (p_c_direction_grad_0_ref * adj0 + p_c_direction_grad_1_ref * adj3 + p_c_direction_grad_2_ref * adj6) / det;
            const scalar_t p_c_direction_grad_1 = (p_c_direction_grad_0_ref * adj1 + p_c_direction_grad_1_ref * adj4 + p_c_direction_grad_2_ref * adj7) / det;
            const scalar_t p_c_direction_grad_2 = (p_c_direction_grad_0_ref * adj2 + p_c_direction_grad_1_ref * adj5 + p_c_direction_grad_2_ref * adj8) / det;
            const scalar_t residual_tmp0 = porosity/dt;
            const scalar_t residual_tmp1 = p_c_direction*residual_tmp0;
            const scalar_t residual_tmp2 = exp(kappa_T*(p_w - p_wr));
            const scalar_t residual_tmp3 = residual_tmp2*rho_w0;
            const scalar_t residual_tmp4 = S_res - 1;
            const scalar_t residual_tmp5 = p_c - p_w;
            const scalar_t residual_tmp6 = pow(residual_tmp5/P_r, m);
            const scalar_t residual_tmp7 = residual_tmp6 + 1;
            const scalar_t residual_tmp8 = 1 - 1/m;
            const scalar_t residual_tmp9 = pow(residual_tmp7, residual_tmp8);
            const scalar_t residual_tmp10 = pow_m1(residual_tmp9);
            const scalar_t residual_tmp11 = -residual_tmp10*residual_tmp4;
            const scalar_t residual_tmp12 = -m*residual_tmp6*residual_tmp8/(residual_tmp5*residual_tmp7);
            const scalar_t residual_tmp13 = residual_tmp11*residual_tmp12;
            const scalar_t residual_tmp14 = residual_tmp13*residual_tmp3;
            const scalar_t residual_tmp15 = S_res + residual_tmp11;
            const scalar_t residual_tmp16 = p_w_direction*residual_tmp0;
            const scalar_t residual_tmp17 = pow_m1(mu_w);
            const scalar_t residual_tmp18 = residual_tmp17*residual_tmp3;
            const scalar_t residual_tmp19 = K_0*residual_tmp18;
            const scalar_t residual_tmp20 = pow(residual_tmp15, pow_m1(C_kw1));
            const scalar_t residual_tmp21 = 1 - residual_tmp20;
            const scalar_t residual_tmp22 = pow(residual_tmp21, C_kw1);
            const scalar_t residual_tmp23 = 1 - residual_tmp22;
            const scalar_t residual_tmp24 = pow_2(residual_tmp23);
            const scalar_t residual_tmp25 = sqrt(residual_tmp15);
            const scalar_t residual_tmp26 = residual_tmp24*residual_tmp25;
            const scalar_t residual_tmp27 = p_w_direction_grad_0*residual_tmp26;
            const scalar_t residual_tmp28 = residual_tmp18*residual_tmp26;
            const scalar_t residual_tmp29 = p_w_direction_grad_1*residual_tmp28;
            const scalar_t residual_tmp30 = p_w_direction_grad_2*residual_tmp28;
            const scalar_t residual_tmp31 = (1.0/2.0)*residual_tmp24;
            const scalar_t residual_tmp32 = residual_tmp13/residual_tmp25;
            const scalar_t residual_tmp33 = p_w_grad_0*residual_tmp32;
            const scalar_t residual_tmp34 = residual_tmp19*residual_tmp33;
            const scalar_t residual_tmp35 = residual_tmp18*residual_tmp31;
            const scalar_t residual_tmp36 = residual_tmp32*residual_tmp35;
            const scalar_t residual_tmp37 = p_w_grad_1*residual_tmp36;
            const scalar_t residual_tmp38 = p_w_grad_2*residual_tmp36;
            const scalar_t residual_tmp39 = 2*residual_tmp20*residual_tmp22*residual_tmp23/residual_tmp21;
            const scalar_t residual_tmp40 = residual_tmp18*residual_tmp39;
            const scalar_t residual_tmp41 = residual_tmp32*residual_tmp40;
            const scalar_t residual_tmp42 = p_w_grad_1*residual_tmp41;
            const scalar_t residual_tmp43 = p_w_grad_2*residual_tmp41;
            const scalar_t residual_tmp44 = K_1*residual_tmp37 + K_1*residual_tmp42 + K_2*residual_tmp38 + K_2*residual_tmp43 + residual_tmp31*residual_tmp34 + residual_tmp34*residual_tmp39;
            const scalar_t residual_tmp45 = residual_tmp18*residual_tmp27;
            const scalar_t residual_tmp46 = residual_tmp33*residual_tmp35;
            const scalar_t residual_tmp47 = residual_tmp33*residual_tmp40;
            const scalar_t residual_tmp48 = K_3*residual_tmp46 + K_3*residual_tmp47 + K_4*residual_tmp37 + K_4*residual_tmp42 + K_5*residual_tmp38 + K_5*residual_tmp43;
            const scalar_t residual_tmp49 = K_6*residual_tmp46 + K_6*residual_tmp47 + K_7*residual_tmp37 + K_7*residual_tmp42 + K_8*residual_tmp38 + K_8*residual_tmp43;
            const scalar_t residual_tmp50 = pow_m1(R);
            const scalar_t residual_tmp51 = pow_m1(T);
            const scalar_t residual_tmp52 = pow_m1(Z);
            const scalar_t residual_tmp53 = M_c*residual_tmp50*residual_tmp51*residual_tmp52;
            const scalar_t residual_tmp54 = p_c*residual_tmp13*residual_tmp53;
            const scalar_t residual_tmp55 = pow(residual_tmp10, C_ka2);
            const scalar_t residual_tmp56 = 1 - residual_tmp55;
            const scalar_t residual_tmp57 = pow_m1(mu_c);
            const scalar_t residual_tmp58 = 1 - residual_tmp10;
            const scalar_t residual_tmp59 = pow(residual_tmp58, C_ka1);
            const scalar_t residual_tmp60 = residual_tmp53*residual_tmp57*residual_tmp59;
            const scalar_t residual_tmp61 = residual_tmp56*residual_tmp60;
            const scalar_t residual_tmp62 = p_c*residual_tmp61;
            const scalar_t residual_tmp63 = p_c_direction_grad_0*residual_tmp62;
            const scalar_t residual_tmp64 = p_c_direction_grad_1*residual_tmp62;
            const scalar_t residual_tmp65 = p_c_direction_grad_2*residual_tmp62;
            const scalar_t residual_tmp66 = p_c*residual_tmp10*residual_tmp12;
            const scalar_t residual_tmp67 = C_ka2*residual_tmp55*residual_tmp60*residual_tmp66*residual_tmp9;
            const scalar_t residual_tmp68 = p_c_grad_0*residual_tmp67;
            const scalar_t residual_tmp69 = p_c_grad_1*residual_tmp67;
            const scalar_t residual_tmp70 = p_c_grad_2*residual_tmp67;
            const scalar_t residual_tmp71 = p_c_grad_0*residual_tmp61;
            const scalar_t residual_tmp72 = C_ka1*residual_tmp66/residual_tmp58;
            const scalar_t residual_tmp73 = p_c_grad_1*residual_tmp61;
            const scalar_t residual_tmp74 = p_c_grad_2*residual_tmp61;
            const scalar_t residual_tmp75 = K_0*residual_tmp68 + K_0*residual_tmp71*residual_tmp72 + K_1*residual_tmp69 + K_1*residual_tmp72*residual_tmp73 + K_2*residual_tmp70 + K_2*residual_tmp72*residual_tmp74;
            const scalar_t residual_tmp76 = K_3*residual_tmp68 + K_3*residual_tmp71*residual_tmp72 + K_4*residual_tmp69 + K_4*residual_tmp72*residual_tmp73 + K_5*residual_tmp70 + K_5*residual_tmp72*residual_tmp74;
            const scalar_t residual_tmp77 = K_6*residual_tmp68 + K_6*residual_tmp71*residual_tmp72 + K_7*residual_tmp69 + K_7*residual_tmp72*residual_tmp73 + K_8*residual_tmp70 + K_8*residual_tmp72*residual_tmp74;
            const scalar_t value_coeff0 = residual_tmp1*residual_tmp14 + residual_tmp16*(kappa_T*residual_tmp15*residual_tmp3 - residual_tmp14);
            const scalar_t grad_coeff0_0 = K_1*residual_tmp29 + K_2*residual_tmp30 + p_c_direction*residual_tmp44 + p_w_direction*(K_0*kappa_T*p_w_grad_0*residual_tmp17*residual_tmp2*residual_tmp24*residual_tmp25*rho_w0 + K_1*kappa_T*p_w_grad_1*residual_tmp17*residual_tmp2*residual_tmp24*residual_tmp25*rho_w0 + K_2*kappa_T*p_w_grad_2*residual_tmp17*residual_tmp2*residual_tmp24*residual_tmp25*rho_w0 - residual_tmp44) + residual_tmp19*residual_tmp27;
            const scalar_t grad_coeff0_1 = K_3*residual_tmp45 + K_4*residual_tmp29 + K_5*residual_tmp30 + p_c_direction*residual_tmp48 + p_w_direction*(K_3*kappa_T*p_w_grad_0*residual_tmp17*residual_tmp2*residual_tmp24*residual_tmp25*rho_w0 + K_4*kappa_T*p_w_grad_1*residual_tmp17*residual_tmp2*residual_tmp24*residual_tmp25*rho_w0 + K_5*kappa_T*p_w_grad_2*residual_tmp17*residual_tmp2*residual_tmp24*residual_tmp25*rho_w0 - residual_tmp48);
            const scalar_t grad_coeff0_2 = K_6*residual_tmp45 + K_7*residual_tmp29 + K_8*residual_tmp30 + p_c_direction*residual_tmp49 + p_w_direction*(K_6*kappa_T*p_w_grad_0*residual_tmp17*residual_tmp2*residual_tmp24*residual_tmp25*rho_w0 + K_7*kappa_T*p_w_grad_1*residual_tmp17*residual_tmp2*residual_tmp24*residual_tmp25*rho_w0 + K_8*kappa_T*p_w_grad_2*residual_tmp17*residual_tmp2*residual_tmp24*residual_tmp25*rho_w0 - residual_tmp49);
            const scalar_t value_coeff1 = residual_tmp1*(M_c*residual_tmp50*residual_tmp51*residual_tmp52*(-residual_tmp11 - residual_tmp4) - residual_tmp54) + residual_tmp16*residual_tmp54;
            const scalar_t grad_coeff1_0 = K_0*residual_tmp63 + K_1*residual_tmp64 + K_2*residual_tmp65 + p_c_direction*(K_0*M_c*p_c_grad_0*residual_tmp50*residual_tmp51*residual_tmp52*residual_tmp56*residual_tmp57*residual_tmp59 + K_1*M_c*p_c_grad_1*residual_tmp50*residual_tmp51*residual_tmp52*residual_tmp56*residual_tmp57*residual_tmp59 + K_2*M_c*p_c_grad_2*residual_tmp50*residual_tmp51*residual_tmp52*residual_tmp56*residual_tmp57*residual_tmp59 - residual_tmp75) + p_w_direction*residual_tmp75;
            const scalar_t grad_coeff1_1 = K_3*residual_tmp63 + K_4*residual_tmp64 + K_5*residual_tmp65 + p_c_direction*(K_3*M_c*p_c_grad_0*residual_tmp50*residual_tmp51*residual_tmp52*residual_tmp56*residual_tmp57*residual_tmp59 + K_4*M_c*p_c_grad_1*residual_tmp50*residual_tmp51*residual_tmp52*residual_tmp56*residual_tmp57*residual_tmp59 + K_5*M_c*p_c_grad_2*residual_tmp50*residual_tmp51*residual_tmp52*residual_tmp56*residual_tmp57*residual_tmp59 - residual_tmp76) + p_w_direction*residual_tmp76;
            const scalar_t grad_coeff1_2 = K_6*residual_tmp63 + K_7*residual_tmp64 + K_8*residual_tmp65 + p_c_direction*(K_6*M_c*p_c_grad_0*residual_tmp50*residual_tmp51*residual_tmp52*residual_tmp56*residual_tmp57*residual_tmp59 + K_7*M_c*p_c_grad_1*residual_tmp50*residual_tmp51*residual_tmp52*residual_tmp56*residual_tmp57*residual_tmp59 + K_8*M_c*p_c_grad_2*residual_tmp50*residual_tmp51*residual_tmp52*residual_tmp56*residual_tmp57*residual_tmp59 - residual_tmp77) + p_w_direction*residual_tmp77;
            value_coeff[(0 * N_QP + q) * VECTOR_SIZE + lane] = qw * det * value_coeff0;
            grad_coeff_ref[((0 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff0_0 + adj1 * grad_coeff0_1 + adj2 * grad_coeff0_2);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj3 * grad_coeff0_0 + adj4 * grad_coeff0_1 + adj5 * grad_coeff0_2);
            grad_coeff_ref[((0 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane] = qw * (adj6 * grad_coeff0_0 + adj7 * grad_coeff0_1 + adj8 * grad_coeff0_2);
            value_coeff[(1 * N_QP + q) * VECTOR_SIZE + lane] = qw * det * value_coeff1;
            grad_coeff_ref[((1 * N_QP + q) * DIM + 0) * VECTOR_SIZE + lane] = qw * (adj0 * grad_coeff1_0 + adj1 * grad_coeff1_1 + adj2 * grad_coeff1_2);
            grad_coeff_ref[((1 * N_QP + q) * DIM + 1) * VECTOR_SIZE + lane] = qw * (adj3 * grad_coeff1_0 + adj4 * grad_coeff1_1 + adj5 * grad_coeff1_2);
            grad_coeff_ref[((1 * N_QP + q) * DIM + 2) * VECTOR_SIZE + lane] = qw * (adj6 * grad_coeff1_0 + adj7 * grad_coeff1_1 + adj8 * grad_coeff1_2);
        }
    }
    generated_two_phase_flow_d3_tensor_product_tensor_integrate<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, N_FIELDS>(
            nelems, shape_1d, grad_1d, value_coeff, grad_coeff_ref, output);
}

} // namespace codegen
} // namespace sfem

#endif
