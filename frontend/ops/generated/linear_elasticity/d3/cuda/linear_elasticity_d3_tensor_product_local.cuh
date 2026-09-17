#ifndef LINEAR_ELASTICITY_D3_TENSOR_PRODUCT_LOCAL_CUH
#define LINEAR_ELASTICITY_D3_TENSOR_PRODUCT_LOCAL_CUH
#include <math.h>
#include <stddef.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_GENERATED_SCALAR_T
#endif
#endif
#include "../../../cuda/kernel_math.cuh"
#include "../../../cuda/tensor_product_kernels.cuh"
#ifndef SFEM_RESTRICT
#define SFEM_RESTRICT __restrict__
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
static __host__ __device__ __forceinline__ void linear_elasticity_d3_tensor_product_objective_block(
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
        s_t *const RSTR value
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NQ1 = integer_root(NQ, 3);
  static constexpr int NS1 = integer_root(NS, 3);
  static_assert(ipow(NQ1, 3) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 3) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[NQ * 9 * VS];
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
    {
      const s_t adj_value0 = adj_q0[0];
      const s_t adj_value1 = adj_q1[0];
      const s_t adj_value2 = adj_q2[0];
      const s_t adj_value3 = adj_q3[0];
      const s_t adj_value4 = adj_q4[0];
      const s_t adj_value5 = adj_q5[0];
      const s_t adj_value6 = adj_q6[0];
      const s_t adj_value7 = adj_q7[0];
      const s_t adj_value8 = adj_q8[0];
      const s_t det_value0 = det_q0[0];
      s_t gu[9];
      const s_t idet = s_t(1) / det_value0;
      gu[0] = (gu_ref0[0] * adj_value0 + gu_ref1[0] * adj_value3 + gu_ref2[0] * adj_value6) * idet;
      gu[1] = (gu_ref0[0] * adj_value1 + gu_ref1[0] * adj_value4 + gu_ref2[0] * adj_value7) * idet;
      gu[2] = (gu_ref0[0] * adj_value2 + gu_ref1[0] * adj_value5 + gu_ref2[0] * adj_value8) * idet;
      gu[3] = (gu_ref3[0] * adj_value0 + gu_ref4[0] * adj_value3 + gu_ref5[0] * adj_value6) * idet;
      gu[4] = (gu_ref3[0] * adj_value1 + gu_ref4[0] * adj_value4 + gu_ref5[0] * adj_value7) * idet;
      gu[5] = (gu_ref3[0] * adj_value2 + gu_ref4[0] * adj_value5 + gu_ref5[0] * adj_value8) * idet;
      gu[6] = (gu_ref6[0] * adj_value0 + gu_ref7[0] * adj_value3 + gu_ref8[0] * adj_value6) * idet;
      gu[7] = (gu_ref6[0] * adj_value1 + gu_ref7[0] * adj_value4 + gu_ref8[0] * adj_value7) * idet;
      gu[8] = (gu_ref6[0] * adj_value2 + gu_ref7[0] * adj_value5 + gu_ref8[0] * adj_value8) * idet;
    value[0] += qw * det_value0 * (((s_t(1) / s_t(2)))*lmbda*pow_2(gu[0] + gu[4] + gu[8]) + mu*(pow_2(gu[0]) + pow_2(gu[4]) + pow_2(gu[8]) + s_t(2)*pow_2(((s_t(1) / s_t(2)))*gu[1] + ((s_t(1) / s_t(2)))*gu[3]) + s_t(2)*pow_2(((s_t(1) / s_t(2)))*gu[2] + ((s_t(1) / s_t(2)))*gu[6]) + s_t(2)*pow_2(((s_t(1) / s_t(2)))*gu[5] + ((s_t(1) / s_t(2)))*gu[7])));
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void linear_elasticity_d3_tensor_product_gradient_block(
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
    {
      const s_t adj_value0 = adj_q0[0];
      const s_t adj_value1 = adj_q1[0];
      const s_t adj_value2 = adj_q2[0];
      const s_t adj_value3 = adj_q3[0];
      const s_t adj_value4 = adj_q4[0];
      const s_t adj_value5 = adj_q5[0];
      const s_t adj_value6 = adj_q6[0];
      const s_t adj_value7 = adj_q7[0];
      const s_t adj_value8 = adj_q8[0];
      const s_t det_value0 = det_q0[0];
      s_t gu[9];
      const s_t idet = s_t(1) / det_value0;
      gu[0] = (gu_ref0[0] * adj_value0 + gu_ref1[0] * adj_value3 + gu_ref2[0] * adj_value6) * idet;
      gu[1] = (gu_ref0[0] * adj_value1 + gu_ref1[0] * adj_value4 + gu_ref2[0] * adj_value7) * idet;
      gu[2] = (gu_ref0[0] * adj_value2 + gu_ref1[0] * adj_value5 + gu_ref2[0] * adj_value8) * idet;
      gu[3] = (gu_ref3[0] * adj_value0 + gu_ref4[0] * adj_value3 + gu_ref5[0] * adj_value6) * idet;
      gu[4] = (gu_ref3[0] * adj_value1 + gu_ref4[0] * adj_value4 + gu_ref5[0] * adj_value7) * idet;
      gu[5] = (gu_ref3[0] * adj_value2 + gu_ref4[0] * adj_value5 + gu_ref5[0] * adj_value8) * idet;
      gu[6] = (gu_ref6[0] * adj_value0 + gu_ref7[0] * adj_value3 + gu_ref8[0] * adj_value6) * idet;
      gu[7] = (gu_ref6[0] * adj_value1 + gu_ref7[0] * adj_value4 + gu_ref8[0] * adj_value7) * idet;
      gu[8] = (gu_ref6[0] * adj_value2 + gu_ref7[0] * adj_value5 + gu_ref8[0] * adj_value8) * idet;
      s_t loperand[9];
    s_t material[9];
    const s_t weak_mat_tmp0 = s_t(2)*gu[0];
    const s_t weak_mat_tmp1 = s_t(2)*gu[4];
    const s_t weak_mat_tmp2 = s_t(2)*gu[8];
    const s_t weak_mat_tmp3 = ((s_t(1) / s_t(2)))*lmbda*(weak_mat_tmp0 + weak_mat_tmp1 + weak_mat_tmp2);
    const s_t weak_mat_tmp4 = mu*(gu[1] + gu[3]);
    const s_t weak_mat_tmp5 = mu*(gu[2] + gu[6]);
    const s_t weak_mat_tmp6 = mu*(gu[5] + gu[7]);
    material[0] = mu*weak_mat_tmp0 + weak_mat_tmp3;
    material[1] = weak_mat_tmp4;
    material[2] = weak_mat_tmp5;
    material[3] = weak_mat_tmp4;
    material[4] = mu*weak_mat_tmp1 + weak_mat_tmp3;
    material[5] = weak_mat_tmp6;
    material[6] = weak_mat_tmp5;
    material[7] = weak_mat_tmp6;
    material[8] = mu*weak_mat_tmp2 + weak_mat_tmp3;
    loperand[0] = qw * (material[0] * adj_value0 + material[1] * adj_value1 + material[2] * adj_value2);
    loperand[1] = qw * (material[0] * adj_value3 + material[1] * adj_value4 + material[2] * adj_value5);
    loperand[2] = qw * (material[0] * adj_value6 + material[1] * adj_value7 + material[2] * adj_value8);
    loperand[3] = qw * (material[3] * adj_value0 + material[4] * adj_value1 + material[5] * adj_value2);
    loperand[4] = qw * (material[3] * adj_value3 + material[4] * adj_value4 + material[5] * adj_value5);
    loperand[5] = qw * (material[3] * adj_value6 + material[4] * adj_value7 + material[5] * adj_value8);
    loperand[6] = qw * (material[6] * adj_value0 + material[7] * adj_value1 + material[8] * adj_value2);
    loperand[7] = qw * (material[6] * adj_value3 + material[7] * adj_value4 + material[8] * adj_value5);
    loperand[8] = qw * (material[6] * adj_value6 + material[7] * adj_value7 + material[8] * adj_value8);
      loperand0[0] = loperand[0];
      loperand1[0] = loperand[1];
      loperand2[0] = loperand[2];
      loperand3[0] = loperand[3];
      loperand4[0] = loperand[4];
      loperand5[0] = loperand[5];
      loperand6[0] = loperand[6];
      loperand7[0] = loperand[7];
      loperand8[0] = loperand[8];
    }
  }
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[0], out_streams, 0);
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[3 * NQ * VS], out_streams, 1);
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[6 * NQ * VS], out_streams, 2);
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void linear_elasticity_d3_tensor_product_apply_block(
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
    {
      const s_t adj_value0 = adj_q0[0];
      const s_t adj_value1 = adj_q1[0];
      const s_t adj_value2 = adj_q2[0];
      const s_t adj_value3 = adj_q3[0];
      const s_t adj_value4 = adj_q4[0];
      const s_t adj_value5 = adj_q5[0];
      const s_t adj_value6 = adj_q6[0];
      const s_t adj_value7 = adj_q7[0];
      const s_t adj_value8 = adj_q8[0];
      const s_t det_value0 = det_q0[0];
      s_t trial_grad[9];
      const s_t idet = s_t(1) / det_value0;
      trial_grad[0] = (grad_h_ref0[0] * adj_value0 + grad_h_ref1[0] * adj_value3 + grad_h_ref2[0] * adj_value6) * idet;
      trial_grad[1] = (grad_h_ref0[0] * adj_value1 + grad_h_ref1[0] * adj_value4 + grad_h_ref2[0] * adj_value7) * idet;
      trial_grad[2] = (grad_h_ref0[0] * adj_value2 + grad_h_ref1[0] * adj_value5 + grad_h_ref2[0] * adj_value8) * idet;
      trial_grad[3] = (grad_h_ref3[0] * adj_value0 + grad_h_ref4[0] * adj_value3 + grad_h_ref5[0] * adj_value6) * idet;
      trial_grad[4] = (grad_h_ref3[0] * adj_value1 + grad_h_ref4[0] * adj_value4 + grad_h_ref5[0] * adj_value7) * idet;
      trial_grad[5] = (grad_h_ref3[0] * adj_value2 + grad_h_ref4[0] * adj_value5 + grad_h_ref5[0] * adj_value8) * idet;
      trial_grad[6] = (grad_h_ref6[0] * adj_value0 + grad_h_ref7[0] * adj_value3 + grad_h_ref8[0] * adj_value6) * idet;
      trial_grad[7] = (grad_h_ref6[0] * adj_value1 + grad_h_ref7[0] * adj_value4 + grad_h_ref8[0] * adj_value7) * idet;
      trial_grad[8] = (grad_h_ref6[0] * adj_value2 + grad_h_ref7[0] * adj_value5 + grad_h_ref8[0] * adj_value8) * idet;
      s_t loperand[9];
    s_t material[9];
    const s_t weak_mat_tmp0 = s_t(2)*trial_grad[0];
    const s_t weak_mat_tmp1 = s_t(2)*trial_grad[4];
    const s_t weak_mat_tmp2 = s_t(2)*trial_grad[8];
    const s_t weak_mat_tmp3 = ((s_t(1) / s_t(2)))*lmbda*(weak_mat_tmp0 + weak_mat_tmp1 + weak_mat_tmp2);
    const s_t weak_mat_tmp4 = mu*(trial_grad[1] + trial_grad[3]);
    const s_t weak_mat_tmp5 = mu*(trial_grad[2] + trial_grad[6]);
    const s_t weak_mat_tmp6 = mu*(trial_grad[5] + trial_grad[7]);
    material[0] = mu*weak_mat_tmp0 + weak_mat_tmp3;
    material[1] = weak_mat_tmp4;
    material[2] = weak_mat_tmp5;
    material[3] = weak_mat_tmp4;
    material[4] = mu*weak_mat_tmp1 + weak_mat_tmp3;
    material[5] = weak_mat_tmp6;
    material[6] = weak_mat_tmp5;
    material[7] = weak_mat_tmp6;
    material[8] = mu*weak_mat_tmp2 + weak_mat_tmp3;
    loperand[0] = qw * (material[0] * adj_value0 + material[1] * adj_value1 + material[2] * adj_value2);
    loperand[1] = qw * (material[0] * adj_value3 + material[1] * adj_value4 + material[2] * adj_value5);
    loperand[2] = qw * (material[0] * adj_value6 + material[1] * adj_value7 + material[2] * adj_value8);
    loperand[3] = qw * (material[3] * adj_value0 + material[4] * adj_value1 + material[5] * adj_value2);
    loperand[4] = qw * (material[3] * adj_value3 + material[4] * adj_value4 + material[5] * adj_value5);
    loperand[5] = qw * (material[3] * adj_value6 + material[4] * adj_value7 + material[5] * adj_value8);
    loperand[6] = qw * (material[6] * adj_value0 + material[7] * adj_value1 + material[8] * adj_value2);
    loperand[7] = qw * (material[6] * adj_value3 + material[7] * adj_value4 + material[8] * adj_value5);
    loperand[8] = qw * (material[6] * adj_value6 + material[7] * adj_value7 + material[8] * adj_value8);
      loperand0[0] = loperand[0];
      loperand1[0] = loperand[1];
      loperand2[0] = loperand[2];
      loperand3[0] = loperand[3];
      loperand4[0] = loperand[4];
      loperand5[0] = loperand[5];
      loperand6[0] = loperand[6];
      loperand7[0] = loperand[7];
      loperand8[0] = loperand[8];
    }
  }
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[0], out_streams, 0);
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[3 * NQ * VS], out_streams, 1);
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[6 * NQ * VS], out_streams, 2);
}

} // namespace codegen
} // namespace sfem

#endif
