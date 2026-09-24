#ifndef LINEAR_ELASTICITY_D3_TENSOR_PRODUCT_LOCAL_HPP
#define LINEAR_ELASTICITY_D3_TENSOR_PRODUCT_LOCAL_HPP
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
typedef ptrdiff_t element_idx_t;
typedef ptrdiff_t count_t;
typedef double geom_t;
#endif
namespace sfem {
namespace codegen {

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void linear_elasticity_d3_tensor_product_objective_block(
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
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t lmbda,
        const s_t mu,
        const s_t *const RSTR u_streams[NS * 3],
        const s_t *const RSTR h_streams[NS * 3],
        const int nsteps,
        const s_t *const RSTR steps,
        const ptrdiff_t value_stride,
        s_t *const RSTR value
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NQ1 = integer_root(NQ, 3);
  static constexpr int NS1 = integer_root(NS, 3);
  static_assert(ipow(NQ1, 3) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 3) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[NQ * 9 * VS];
  s_t grad_h_ref_q[NQ * 9 * VS];
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, h_streams, 0, &grad_h_ref_q[0]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 1, &gu_ref_q[3 * NQ * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, h_streams, 1, &grad_h_ref_q[3 * NQ * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 2, &gu_ref_q[6 * NQ * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, h_streams, 2, &grad_h_ref_q[6 * NQ * VS]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
    const s_t *const RSTR gu_ref0 = &gu_ref_q[(3 * q) * VS];
    const s_t *const RSTR gu_ref1 = &gu_ref_q[(3 * q + 1) * VS];
    const s_t *const RSTR gu_ref2 = &gu_ref_q[(3 * q + 2) * VS];
    const s_t *const RSTR gu_ref3 = &gu_ref_q[(3 * (NQ + q)) * VS];
    const s_t *const RSTR gu_ref4 = &gu_ref_q[(3 * (NQ + q) + 1) * VS];
    const s_t *const RSTR gu_ref5 = &gu_ref_q[(3 * (NQ + q) + 2) * VS];
    const s_t *const RSTR gu_ref6 = &gu_ref_q[(3 * (2 * NQ + q)) * VS];
    const s_t *const RSTR gu_ref7 = &gu_ref_q[(3 * (2 * NQ + q) + 1) * VS];
    const s_t *const RSTR gu_ref8 = &gu_ref_q[(3 * (2 * NQ + q) + 2) * VS];
    const s_t *const RSTR grad_h_ref0 = &grad_h_ref_q[(3 * q) * VS];
    const s_t *const RSTR grad_h_ref1 = &grad_h_ref_q[(3 * q + 1) * VS];
    const s_t *const RSTR grad_h_ref2 = &grad_h_ref_q[(3 * q + 2) * VS];
    const s_t *const RSTR grad_h_ref3 = &grad_h_ref_q[(3 * (NQ + q)) * VS];
    const s_t *const RSTR grad_h_ref4 = &grad_h_ref_q[(3 * (NQ + q) + 1) * VS];
    const s_t *const RSTR grad_h_ref5 = &grad_h_ref_q[(3 * (NQ + q) + 2) * VS];
    const s_t *const RSTR grad_h_ref6 = &grad_h_ref_q[(3 * (2 * NQ + q)) * VS];
    const s_t *const RSTR grad_h_ref7 = &grad_h_ref_q[(3 * (2 * NQ + q) + 1) * VS];
    const s_t *const RSTR grad_h_ref8 = &grad_h_ref_q[(3 * (2 * NQ + q) + 2) * VS];
    const s_t *const RSTR adj_q0 = adj0 + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adj1 + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adj2 + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adj3 + q * geometry_stride;
    const s_t *const RSTR adj_q4 = adj4 + q * geometry_stride;
    const s_t *const RSTR adj_q5 = adj5 + q * geometry_stride;
    const s_t *const RSTR adj_q6 = adj6 + q * geometry_stride;
    const s_t *const RSTR adj_q7 = adj7 + q * geometry_stride;
    const s_t *const RSTR adj_q8 = adj8 + q * geometry_stride;
    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;
    s_t gu_base_v[9 * VS];
    s_t trial_grad_v[9 * VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t adj_lane0 = adj_q0[lane];
      const s_t adj_lane1 = adj_q1[lane];
      const s_t adj_lane2 = adj_q2[lane];
      const s_t adj_lane3 = adj_q3[lane];
      const s_t adj_lane4 = adj_q4[lane];
      const s_t adj_lane5 = adj_q5[lane];
      const s_t adj_lane6 = adj_q6[lane];
      const s_t adj_lane7 = adj_q7[lane];
      const s_t adj_lane8 = adj_q8[lane];
      const s_t det_lane0 = det_q0[lane];
      const s_t idet = s_t(1) / det_lane0;
      gu_base_v[0 * VS + lane] = (gu_ref0[lane] * adj_lane0 + gu_ref1[lane] * adj_lane3 + gu_ref2[lane] * adj_lane6) * idet;
      trial_grad_v[0 * VS + lane] = (grad_h_ref0[lane] * adj_lane0 + grad_h_ref1[lane] * adj_lane3 + grad_h_ref2[lane] * adj_lane6) * idet;
      gu_base_v[1 * VS + lane] = (gu_ref0[lane] * adj_lane1 + gu_ref1[lane] * adj_lane4 + gu_ref2[lane] * adj_lane7) * idet;
      trial_grad_v[1 * VS + lane] = (grad_h_ref0[lane] * adj_lane1 + grad_h_ref1[lane] * adj_lane4 + grad_h_ref2[lane] * adj_lane7) * idet;
      gu_base_v[2 * VS + lane] = (gu_ref0[lane] * adj_lane2 + gu_ref1[lane] * adj_lane5 + gu_ref2[lane] * adj_lane8) * idet;
      trial_grad_v[2 * VS + lane] = (grad_h_ref0[lane] * adj_lane2 + grad_h_ref1[lane] * adj_lane5 + grad_h_ref2[lane] * adj_lane8) * idet;
      gu_base_v[3 * VS + lane] = (gu_ref3[lane] * adj_lane0 + gu_ref4[lane] * adj_lane3 + gu_ref5[lane] * adj_lane6) * idet;
      trial_grad_v[3 * VS + lane] = (grad_h_ref3[lane] * adj_lane0 + grad_h_ref4[lane] * adj_lane3 + grad_h_ref5[lane] * adj_lane6) * idet;
      gu_base_v[4 * VS + lane] = (gu_ref3[lane] * adj_lane1 + gu_ref4[lane] * adj_lane4 + gu_ref5[lane] * adj_lane7) * idet;
      trial_grad_v[4 * VS + lane] = (grad_h_ref3[lane] * adj_lane1 + grad_h_ref4[lane] * adj_lane4 + grad_h_ref5[lane] * adj_lane7) * idet;
      gu_base_v[5 * VS + lane] = (gu_ref3[lane] * adj_lane2 + gu_ref4[lane] * adj_lane5 + gu_ref5[lane] * adj_lane8) * idet;
      trial_grad_v[5 * VS + lane] = (grad_h_ref3[lane] * adj_lane2 + grad_h_ref4[lane] * adj_lane5 + grad_h_ref5[lane] * adj_lane8) * idet;
      gu_base_v[6 * VS + lane] = (gu_ref6[lane] * adj_lane0 + gu_ref7[lane] * adj_lane3 + gu_ref8[lane] * adj_lane6) * idet;
      trial_grad_v[6 * VS + lane] = (grad_h_ref6[lane] * adj_lane0 + grad_h_ref7[lane] * adj_lane3 + grad_h_ref8[lane] * adj_lane6) * idet;
      gu_base_v[7 * VS + lane] = (gu_ref6[lane] * adj_lane1 + gu_ref7[lane] * adj_lane4 + gu_ref8[lane] * adj_lane7) * idet;
      trial_grad_v[7 * VS + lane] = (grad_h_ref6[lane] * adj_lane1 + grad_h_ref7[lane] * adj_lane4 + grad_h_ref8[lane] * adj_lane7) * idet;
      gu_base_v[8 * VS + lane] = (gu_ref6[lane] * adj_lane2 + gu_ref7[lane] * adj_lane5 + gu_ref8[lane] * adj_lane8) * idet;
      trial_grad_v[8 * VS + lane] = (grad_h_ref6[lane] * adj_lane2 + grad_h_ref7[lane] * adj_lane5 + grad_h_ref8[lane] * adj_lane8) * idet;
    }
    for (int step = 0; step < nsteps; ++step) {
      const s_t alpha = steps[step];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t det_lane0 = det_q0[lane];
        s_t gu[9];
        gu[0] = gu_base_v[0 * VS + lane] + alpha * trial_grad_v[0 * VS + lane];
        gu[1] = gu_base_v[1 * VS + lane] + alpha * trial_grad_v[1 * VS + lane];
        gu[2] = gu_base_v[2 * VS + lane] + alpha * trial_grad_v[2 * VS + lane];
        gu[3] = gu_base_v[3 * VS + lane] + alpha * trial_grad_v[3 * VS + lane];
        gu[4] = gu_base_v[4 * VS + lane] + alpha * trial_grad_v[4 * VS + lane];
        gu[5] = gu_base_v[5 * VS + lane] + alpha * trial_grad_v[5 * VS + lane];
        gu[6] = gu_base_v[6 * VS + lane] + alpha * trial_grad_v[6 * VS + lane];
        gu[7] = gu_base_v[7 * VS + lane] + alpha * trial_grad_v[7 * VS + lane];
        gu[8] = gu_base_v[8 * VS + lane] + alpha * trial_grad_v[8 * VS + lane];
    value[step * value_stride + lane] += qw * det_lane0 * (((s_t(1) / s_t(2)))*lmbda*pow_2(gu[0] + gu[4] + gu[8]) + mu*(pow_2(gu[0]) + pow_2(gu[4]) + pow_2(gu[8]) + s_t(2)*pow_2(((s_t(1) / s_t(2)))*gu[1] + ((s_t(1) / s_t(2)))*gu[3]) + s_t(2)*pow_2(((s_t(1) / s_t(2)))*gu[2] + ((s_t(1) / s_t(2)))*gu[6]) + s_t(2)*pow_2(((s_t(1) / s_t(2)))*gu[5] + ((s_t(1) / s_t(2)))*gu[7])));
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void linear_elasticity_d3_tensor_product_gradient_block(
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
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t lmbda,
        const s_t mu,
        const s_t *const RSTR u_streams[NS * 3],
        s_t *const RSTR out_streams[NS * 3]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NQ1 = integer_root(NQ, 3);
  static constexpr int NS1 = integer_root(NS, 3);
  static_assert(ipow(NQ1, 3) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 3) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[NQ * 9 * VS];
  s_t loperand_q[NQ * 9 * VS];
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 1, &gu_ref_q[3 * NQ * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 2, &gu_ref_q[6 * NQ * VS]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
    const s_t *const RSTR gu_ref0 = &gu_ref_q[(3 * q) * VS];
    const s_t *const RSTR gu_ref1 = &gu_ref_q[(3 * q + 1) * VS];
    const s_t *const RSTR gu_ref2 = &gu_ref_q[(3 * q + 2) * VS];
    const s_t *const RSTR gu_ref3 = &gu_ref_q[(3 * (NQ + q)) * VS];
    const s_t *const RSTR gu_ref4 = &gu_ref_q[(3 * (NQ + q) + 1) * VS];
    const s_t *const RSTR gu_ref5 = &gu_ref_q[(3 * (NQ + q) + 2) * VS];
    const s_t *const RSTR gu_ref6 = &gu_ref_q[(3 * (2 * NQ + q)) * VS];
    const s_t *const RSTR gu_ref7 = &gu_ref_q[(3 * (2 * NQ + q) + 1) * VS];
    const s_t *const RSTR gu_ref8 = &gu_ref_q[(3 * (2 * NQ + q) + 2) * VS];
    s_t *const RSTR loperand0 = &loperand_q[(3 * q) * VS];
    s_t *const RSTR loperand1 = &loperand_q[(3 * q + 1) * VS];
    s_t *const RSTR loperand2 = &loperand_q[(3 * q + 2) * VS];
    s_t *const RSTR loperand3 = &loperand_q[(3 * (NQ + q)) * VS];
    s_t *const RSTR loperand4 = &loperand_q[(3 * (NQ + q) + 1) * VS];
    s_t *const RSTR loperand5 = &loperand_q[(3 * (NQ + q) + 2) * VS];
    s_t *const RSTR loperand6 = &loperand_q[(3 * (2 * NQ + q)) * VS];
    s_t *const RSTR loperand7 = &loperand_q[(3 * (2 * NQ + q) + 1) * VS];
    s_t *const RSTR loperand8 = &loperand_q[(3 * (2 * NQ + q) + 2) * VS];
    const s_t *const RSTR adj_q0 = adj0 + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adj1 + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adj2 + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adj3 + q * geometry_stride;
    const s_t *const RSTR adj_q4 = adj4 + q * geometry_stride;
    const s_t *const RSTR adj_q5 = adj5 + q * geometry_stride;
    const s_t *const RSTR adj_q6 = adj6 + q * geometry_stride;
    const s_t *const RSTR adj_q7 = adj7 + q * geometry_stride;
    const s_t *const RSTR adj_q8 = adj8 + q * geometry_stride;
    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t adj_lane0 = adj_q0[lane];
      const s_t adj_lane1 = adj_q1[lane];
      const s_t adj_lane2 = adj_q2[lane];
      const s_t adj_lane3 = adj_q3[lane];
      const s_t adj_lane4 = adj_q4[lane];
      const s_t adj_lane5 = adj_q5[lane];
      const s_t adj_lane6 = adj_q6[lane];
      const s_t adj_lane7 = adj_q7[lane];
      const s_t adj_lane8 = adj_q8[lane];
      const s_t det_lane0 = det_q0[lane];
      s_t gu[9];
      const s_t idet = s_t(1) / det_lane0;
      gu[0] = (gu_ref0[lane] * adj_lane0 + gu_ref1[lane] * adj_lane3 + gu_ref2[lane] * adj_lane6) * idet;
      gu[1] = (gu_ref0[lane] * adj_lane1 + gu_ref1[lane] * adj_lane4 + gu_ref2[lane] * adj_lane7) * idet;
      gu[2] = (gu_ref0[lane] * adj_lane2 + gu_ref1[lane] * adj_lane5 + gu_ref2[lane] * adj_lane8) * idet;
      gu[3] = (gu_ref3[lane] * adj_lane0 + gu_ref4[lane] * adj_lane3 + gu_ref5[lane] * adj_lane6) * idet;
      gu[4] = (gu_ref3[lane] * adj_lane1 + gu_ref4[lane] * adj_lane4 + gu_ref5[lane] * adj_lane7) * idet;
      gu[5] = (gu_ref3[lane] * adj_lane2 + gu_ref4[lane] * adj_lane5 + gu_ref5[lane] * adj_lane8) * idet;
      gu[6] = (gu_ref6[lane] * adj_lane0 + gu_ref7[lane] * adj_lane3 + gu_ref8[lane] * adj_lane6) * idet;
      gu[7] = (gu_ref6[lane] * adj_lane1 + gu_ref7[lane] * adj_lane4 + gu_ref8[lane] * adj_lane7) * idet;
      gu[8] = (gu_ref6[lane] * adj_lane2 + gu_ref7[lane] * adj_lane5 + gu_ref8[lane] * adj_lane8) * idet;
    const s_t weak_mat_tmp0 = s_t(2)*gu[0];
    const s_t weak_mat_tmp1 = s_t(2)*gu[4];
    const s_t weak_mat_tmp2 = s_t(2)*gu[8];
    const s_t weak_mat_tmp3 = ((s_t(1) / s_t(2)))*lmbda*(weak_mat_tmp0 + weak_mat_tmp1 + weak_mat_tmp2);
    const s_t weak_mat_tmp4 = mu*(gu[1] + gu[3]);
    const s_t weak_mat_tmp5 = mu*(gu[2] + gu[6]);
    const s_t weak_mat_tmp6 = mu*(gu[5] + gu[7]);
    const s_t material0 = mu*weak_mat_tmp0 + weak_mat_tmp3;
    const s_t material1 = weak_mat_tmp4;
    const s_t material2 = weak_mat_tmp5;
    const s_t material3 = weak_mat_tmp4;
    const s_t material4 = mu*weak_mat_tmp1 + weak_mat_tmp3;
    const s_t material5 = weak_mat_tmp6;
    const s_t material6 = weak_mat_tmp5;
    const s_t material7 = weak_mat_tmp6;
    const s_t material8 = mu*weak_mat_tmp2 + weak_mat_tmp3;
      loperand0[lane] = qw * (material0 * adj_lane0 + material1 * adj_lane1 + material2 * adj_lane2);
      loperand1[lane] = qw * (material0 * adj_lane3 + material1 * adj_lane4 + material2 * adj_lane5);
      loperand2[lane] = qw * (material0 * adj_lane6 + material1 * adj_lane7 + material2 * adj_lane8);
      loperand3[lane] = qw * (material3 * adj_lane0 + material4 * adj_lane1 + material5 * adj_lane2);
      loperand4[lane] = qw * (material3 * adj_lane3 + material4 * adj_lane4 + material5 * adj_lane5);
      loperand5[lane] = qw * (material3 * adj_lane6 + material4 * adj_lane7 + material5 * adj_lane8);
      loperand6[lane] = qw * (material6 * adj_lane0 + material7 * adj_lane1 + material8 * adj_lane2);
      loperand7[lane] = qw * (material6 * adj_lane3 + material7 * adj_lane4 + material8 * adj_lane5);
      loperand8[lane] = qw * (material6 * adj_lane6 + material7 * adj_lane7 + material8 * adj_lane8);
    }
  }
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[0], out_streams, 0);
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[3 * NQ * VS], out_streams, 1);
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[6 * NQ * VS], out_streams, 2);
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void linear_elasticity_d3_tensor_product_apply_block(
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
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t lmbda,
        const s_t mu,
        const s_t *const RSTR h_streams[NS * 3],
        s_t *const RSTR out_streams[NS * 3]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NQ1 = integer_root(NQ, 3);
  static constexpr int NS1 = integer_root(NS, 3);
  static_assert(ipow(NQ1, 3) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 3) == NS, "NS must be tensor-product compatible");
  s_t grad_h_ref_q[NQ * 9 * VS];
  s_t loperand_q[NQ * 9 * VS];
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, h_streams, 0, &grad_h_ref_q[0]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, h_streams, 1, &grad_h_ref_q[3 * NQ * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, h_streams, 2, &grad_h_ref_q[6 * NQ * VS]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
    const s_t *const RSTR grad_h_ref0 = &grad_h_ref_q[(3 * q) * VS];
    const s_t *const RSTR grad_h_ref1 = &grad_h_ref_q[(3 * q + 1) * VS];
    const s_t *const RSTR grad_h_ref2 = &grad_h_ref_q[(3 * q + 2) * VS];
    const s_t *const RSTR grad_h_ref3 = &grad_h_ref_q[(3 * (NQ + q)) * VS];
    const s_t *const RSTR grad_h_ref4 = &grad_h_ref_q[(3 * (NQ + q) + 1) * VS];
    const s_t *const RSTR grad_h_ref5 = &grad_h_ref_q[(3 * (NQ + q) + 2) * VS];
    const s_t *const RSTR grad_h_ref6 = &grad_h_ref_q[(3 * (2 * NQ + q)) * VS];
    const s_t *const RSTR grad_h_ref7 = &grad_h_ref_q[(3 * (2 * NQ + q) + 1) * VS];
    const s_t *const RSTR grad_h_ref8 = &grad_h_ref_q[(3 * (2 * NQ + q) + 2) * VS];
    s_t *const RSTR loperand0 = &loperand_q[(3 * q) * VS];
    s_t *const RSTR loperand1 = &loperand_q[(3 * q + 1) * VS];
    s_t *const RSTR loperand2 = &loperand_q[(3 * q + 2) * VS];
    s_t *const RSTR loperand3 = &loperand_q[(3 * (NQ + q)) * VS];
    s_t *const RSTR loperand4 = &loperand_q[(3 * (NQ + q) + 1) * VS];
    s_t *const RSTR loperand5 = &loperand_q[(3 * (NQ + q) + 2) * VS];
    s_t *const RSTR loperand6 = &loperand_q[(3 * (2 * NQ + q)) * VS];
    s_t *const RSTR loperand7 = &loperand_q[(3 * (2 * NQ + q) + 1) * VS];
    s_t *const RSTR loperand8 = &loperand_q[(3 * (2 * NQ + q) + 2) * VS];
    const s_t *const RSTR adj_q0 = adj0 + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adj1 + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adj2 + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adj3 + q * geometry_stride;
    const s_t *const RSTR adj_q4 = adj4 + q * geometry_stride;
    const s_t *const RSTR adj_q5 = adj5 + q * geometry_stride;
    const s_t *const RSTR adj_q6 = adj6 + q * geometry_stride;
    const s_t *const RSTR adj_q7 = adj7 + q * geometry_stride;
    const s_t *const RSTR adj_q8 = adj8 + q * geometry_stride;
    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t adj_lane0 = adj_q0[lane];
      const s_t adj_lane1 = adj_q1[lane];
      const s_t adj_lane2 = adj_q2[lane];
      const s_t adj_lane3 = adj_q3[lane];
      const s_t adj_lane4 = adj_q4[lane];
      const s_t adj_lane5 = adj_q5[lane];
      const s_t adj_lane6 = adj_q6[lane];
      const s_t adj_lane7 = adj_q7[lane];
      const s_t adj_lane8 = adj_q8[lane];
      const s_t det_lane0 = det_q0[lane];
      s_t trial_grad[9];
      const s_t idet = s_t(1) / det_lane0;
      trial_grad[0] = (grad_h_ref0[lane] * adj_lane0 + grad_h_ref1[lane] * adj_lane3 + grad_h_ref2[lane] * adj_lane6) * idet;
      trial_grad[1] = (grad_h_ref0[lane] * adj_lane1 + grad_h_ref1[lane] * adj_lane4 + grad_h_ref2[lane] * adj_lane7) * idet;
      trial_grad[2] = (grad_h_ref0[lane] * adj_lane2 + grad_h_ref1[lane] * adj_lane5 + grad_h_ref2[lane] * adj_lane8) * idet;
      trial_grad[3] = (grad_h_ref3[lane] * adj_lane0 + grad_h_ref4[lane] * adj_lane3 + grad_h_ref5[lane] * adj_lane6) * idet;
      trial_grad[4] = (grad_h_ref3[lane] * adj_lane1 + grad_h_ref4[lane] * adj_lane4 + grad_h_ref5[lane] * adj_lane7) * idet;
      trial_grad[5] = (grad_h_ref3[lane] * adj_lane2 + grad_h_ref4[lane] * adj_lane5 + grad_h_ref5[lane] * adj_lane8) * idet;
      trial_grad[6] = (grad_h_ref6[lane] * adj_lane0 + grad_h_ref7[lane] * adj_lane3 + grad_h_ref8[lane] * adj_lane6) * idet;
      trial_grad[7] = (grad_h_ref6[lane] * adj_lane1 + grad_h_ref7[lane] * adj_lane4 + grad_h_ref8[lane] * adj_lane7) * idet;
      trial_grad[8] = (grad_h_ref6[lane] * adj_lane2 + grad_h_ref7[lane] * adj_lane5 + grad_h_ref8[lane] * adj_lane8) * idet;
    const s_t weak_mat_tmp0 = s_t(2)*trial_grad[0];
    const s_t weak_mat_tmp1 = s_t(2)*trial_grad[4];
    const s_t weak_mat_tmp2 = s_t(2)*trial_grad[8];
    const s_t weak_mat_tmp3 = ((s_t(1) / s_t(2)))*lmbda*(weak_mat_tmp0 + weak_mat_tmp1 + weak_mat_tmp2);
    const s_t weak_mat_tmp4 = mu*(trial_grad[1] + trial_grad[3]);
    const s_t weak_mat_tmp5 = mu*(trial_grad[2] + trial_grad[6]);
    const s_t weak_mat_tmp6 = mu*(trial_grad[5] + trial_grad[7]);
    const s_t material0 = mu*weak_mat_tmp0 + weak_mat_tmp3;
    const s_t material1 = weak_mat_tmp4;
    const s_t material2 = weak_mat_tmp5;
    const s_t material3 = weak_mat_tmp4;
    const s_t material4 = mu*weak_mat_tmp1 + weak_mat_tmp3;
    const s_t material5 = weak_mat_tmp6;
    const s_t material6 = weak_mat_tmp5;
    const s_t material7 = weak_mat_tmp6;
    const s_t material8 = mu*weak_mat_tmp2 + weak_mat_tmp3;
      loperand0[lane] = qw * (material0 * adj_lane0 + material1 * adj_lane1 + material2 * adj_lane2);
      loperand1[lane] = qw * (material0 * adj_lane3 + material1 * adj_lane4 + material2 * adj_lane5);
      loperand2[lane] = qw * (material0 * adj_lane6 + material1 * adj_lane7 + material2 * adj_lane8);
      loperand3[lane] = qw * (material3 * adj_lane0 + material4 * adj_lane1 + material5 * adj_lane2);
      loperand4[lane] = qw * (material3 * adj_lane3 + material4 * adj_lane4 + material5 * adj_lane5);
      loperand5[lane] = qw * (material3 * adj_lane6 + material4 * adj_lane7 + material5 * adj_lane8);
      loperand6[lane] = qw * (material6 * adj_lane0 + material7 * adj_lane1 + material8 * adj_lane2);
      loperand7[lane] = qw * (material6 * adj_lane3 + material7 * adj_lane4 + material8 * adj_lane5);
      loperand8[lane] = qw * (material6 * adj_lane6 + material7 * adj_lane7 + material8 * adj_lane8);
    }
  }
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[0], out_streams, 0);
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[3 * NQ * VS], out_streams, 1);
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[6 * NQ * VS], out_streams, 2);
}

} // namespace codegen
} // namespace sfem

#endif
