#ifndef SAINT_VENANT_KIRCHHOFF_D3_TENSOR_PRODUCT_LOCAL_CUH
#define SAINT_VENANT_KIRCHHOFF_D3_TENSOR_PRODUCT_LOCAL_CUH
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
static __host__ __device__ __forceinline__ void saint_venant_kirchhoff_d3_tensor_product_objective_block(
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
      const s_t idet = s_t(1) / det_value0;
      gu_base_v[0 * VS + 0] = (gu_ref0[0] * adj_value0 + gu_ref1[0] * adj_value3 + gu_ref2[0] * adj_value6) * idet;
      trial_grad_v[0 * VS + 0] = (grad_h_ref0[0] * adj_value0 + grad_h_ref1[0] * adj_value3 + grad_h_ref2[0] * adj_value6) * idet;
      gu_base_v[1 * VS + 0] = (gu_ref0[0] * adj_value1 + gu_ref1[0] * adj_value4 + gu_ref2[0] * adj_value7) * idet;
      trial_grad_v[1 * VS + 0] = (grad_h_ref0[0] * adj_value1 + grad_h_ref1[0] * adj_value4 + grad_h_ref2[0] * adj_value7) * idet;
      gu_base_v[2 * VS + 0] = (gu_ref0[0] * adj_value2 + gu_ref1[0] * adj_value5 + gu_ref2[0] * adj_value8) * idet;
      trial_grad_v[2 * VS + 0] = (grad_h_ref0[0] * adj_value2 + grad_h_ref1[0] * adj_value5 + grad_h_ref2[0] * adj_value8) * idet;
      gu_base_v[3 * VS + 0] = (gu_ref3[0] * adj_value0 + gu_ref4[0] * adj_value3 + gu_ref5[0] * adj_value6) * idet;
      trial_grad_v[3 * VS + 0] = (grad_h_ref3[0] * adj_value0 + grad_h_ref4[0] * adj_value3 + grad_h_ref5[0] * adj_value6) * idet;
      gu_base_v[4 * VS + 0] = (gu_ref3[0] * adj_value1 + gu_ref4[0] * adj_value4 + gu_ref5[0] * adj_value7) * idet;
      trial_grad_v[4 * VS + 0] = (grad_h_ref3[0] * adj_value1 + grad_h_ref4[0] * adj_value4 + grad_h_ref5[0] * adj_value7) * idet;
      gu_base_v[5 * VS + 0] = (gu_ref3[0] * adj_value2 + gu_ref4[0] * adj_value5 + gu_ref5[0] * adj_value8) * idet;
      trial_grad_v[5 * VS + 0] = (grad_h_ref3[0] * adj_value2 + grad_h_ref4[0] * adj_value5 + grad_h_ref5[0] * adj_value8) * idet;
      gu_base_v[6 * VS + 0] = (gu_ref6[0] * adj_value0 + gu_ref7[0] * adj_value3 + gu_ref8[0] * adj_value6) * idet;
      trial_grad_v[6 * VS + 0] = (grad_h_ref6[0] * adj_value0 + grad_h_ref7[0] * adj_value3 + grad_h_ref8[0] * adj_value6) * idet;
      gu_base_v[7 * VS + 0] = (gu_ref6[0] * adj_value1 + gu_ref7[0] * adj_value4 + gu_ref8[0] * adj_value7) * idet;
      trial_grad_v[7 * VS + 0] = (grad_h_ref6[0] * adj_value1 + grad_h_ref7[0] * adj_value4 + grad_h_ref8[0] * adj_value7) * idet;
      gu_base_v[8 * VS + 0] = (gu_ref6[0] * adj_value2 + gu_ref7[0] * adj_value5 + gu_ref8[0] * adj_value8) * idet;
      trial_grad_v[8 * VS + 0] = (grad_h_ref6[0] * adj_value2 + grad_h_ref7[0] * adj_value5 + grad_h_ref8[0] * adj_value8) * idet;
    }
    for (int step = 0; step < nsteps; ++step) {
      const s_t alpha = steps[step];
      {
        const s_t det_value0 = det_q0[0];
        s_t gu[9];
        gu[0] = gu_base_v[0 * VS + 0] + alpha * trial_grad_v[0 * VS + 0];
        gu[1] = gu_base_v[1 * VS + 0] + alpha * trial_grad_v[1 * VS + 0];
        gu[2] = gu_base_v[2 * VS + 0] + alpha * trial_grad_v[2 * VS + 0];
        gu[3] = gu_base_v[3 * VS + 0] + alpha * trial_grad_v[3 * VS + 0];
        gu[4] = gu_base_v[4 * VS + 0] + alpha * trial_grad_v[4 * VS + 0];
        gu[5] = gu_base_v[5 * VS + 0] + alpha * trial_grad_v[5 * VS + 0];
        gu[6] = gu_base_v[6 * VS + 0] + alpha * trial_grad_v[6 * VS + 0];
        gu[7] = gu_base_v[7 * VS + 0] + alpha * trial_grad_v[7 * VS + 0];
        gu[8] = gu_base_v[8 * VS + 0] + alpha * trial_grad_v[8 * VS + 0];
    const s_t weak_obj_tmp0 = ((s_t(1) / s_t(2)))*pow_2(gu[0]) + gu[0] + ((s_t(1) / s_t(2)))*pow_2(gu[3]) + ((s_t(1) / s_t(2)))*pow_2(gu[6]);
    const s_t weak_obj_tmp1 = ((s_t(1) / s_t(2)))*pow_2(gu[1]) + ((s_t(1) / s_t(2)))*pow_2(gu[4]) + gu[4] + ((s_t(1) / s_t(2)))*pow_2(gu[7]);
    const s_t weak_obj_tmp2 = ((s_t(1) / s_t(2)))*pow_2(gu[2]) + ((s_t(1) / s_t(2)))*pow_2(gu[5]) + ((s_t(1) / s_t(2)))*pow_2(gu[8]) + gu[8];
    const s_t weak_obj_tmp3 = ((s_t(1) / s_t(2)))*gu[1];
    const s_t weak_obj_tmp4 = ((s_t(1) / s_t(2)))*gu[4] + (s_t(1) / s_t(2));
    const s_t weak_obj_tmp5 = gu[8] + s_t(1);
    const s_t weak_obj_tmp6 = ((s_t(1) / s_t(2)))*gu[7];
    const s_t weak_obj_tmp7 = gu[0] + s_t(1);
    value[step * value_stride + 0] += qw * det_value0 * (((s_t(1) / s_t(2)))*lmbda*pow_2(weak_obj_tmp0 + weak_obj_tmp1 + weak_obj_tmp2) + mu*(pow_2(weak_obj_tmp0) + pow_2(weak_obj_tmp1) + pow_2(weak_obj_tmp2) + s_t(2)*pow_2(gu[2]*weak_obj_tmp3 + gu[5]*weak_obj_tmp4 + weak_obj_tmp5*weak_obj_tmp6) + s_t(2)*pow_2(((s_t(1) / s_t(2)))*gu[2]*weak_obj_tmp7 + ((s_t(1) / s_t(2)))*gu[3]*gu[5] + ((s_t(1) / s_t(2)))*gu[6]*weak_obj_tmp5) + s_t(2)*pow_2(gu[3]*weak_obj_tmp4 + gu[6]*weak_obj_tmp6 + weak_obj_tmp3*weak_obj_tmp7)));
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void saint_venant_kirchhoff_d3_tensor_product_gradient_block(
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
    const s_t weak_mat_tmp0 = gu[0] + s_t(1);
    const s_t weak_mat_tmp1 = ((s_t(1) / s_t(2)))*pow_2(gu[0]) + gu[0] + ((s_t(1) / s_t(2)))*pow_2(gu[3]) + ((s_t(1) / s_t(2)))*pow_2(gu[6]);
    const s_t weak_mat_tmp2 = ((s_t(1) / s_t(2)))*pow_2(gu[1]) + ((s_t(1) / s_t(2)))*pow_2(gu[4]) + gu[4] + ((s_t(1) / s_t(2)))*pow_2(gu[7]);
    const s_t weak_mat_tmp3 = ((s_t(1) / s_t(2)))*pow_2(gu[2]) + ((s_t(1) / s_t(2)))*pow_2(gu[5]) + ((s_t(1) / s_t(2)))*pow_2(gu[8]) + gu[8];
    const s_t weak_mat_tmp4 = lmbda*(weak_mat_tmp1 + weak_mat_tmp2 + weak_mat_tmp3);
    const s_t weak_mat_tmp5 = ((s_t(1) / s_t(2)))*gu[6];
    const s_t weak_mat_tmp6 = ((s_t(1) / s_t(2)))*weak_mat_tmp0;
    const s_t weak_mat_tmp7 = gu[4] + s_t(1);
    const s_t weak_mat_tmp8 = ((s_t(1) / s_t(2)))*gu[3];
    const s_t weak_mat_tmp9 = gu[1]*weak_mat_tmp6 + gu[7]*weak_mat_tmp5 + weak_mat_tmp7*weak_mat_tmp8;
    const s_t weak_mat_tmp10 = s_t(2)*gu[1];
    const s_t weak_mat_tmp11 = gu[8] + s_t(1);
    const s_t weak_mat_tmp12 = gu[2]*weak_mat_tmp6 + gu[5]*weak_mat_tmp8 + weak_mat_tmp11*weak_mat_tmp5;
    const s_t weak_mat_tmp13 = s_t(2)*gu[2];
    const s_t weak_mat_tmp14 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp15 = ((s_t(1) / s_t(2)))*gu[1]*gu[2] + ((s_t(1) / s_t(2)))*gu[5]*weak_mat_tmp7 + ((s_t(1) / s_t(2)))*gu[7]*weak_mat_tmp11;
    const s_t weak_mat_tmp16 = s_t(2)*gu[3];
    const s_t weak_mat_tmp17 = s_t(2)*gu[5];
    const s_t weak_mat_tmp18 = s_t(2)*weak_mat_tmp7;
    const s_t weak_mat_tmp19 = s_t(2)*gu[6];
    const s_t weak_mat_tmp20 = s_t(2)*gu[7];
    const s_t weak_mat_tmp21 = s_t(2)*weak_mat_tmp11;
    material[0] = mu*(weak_mat_tmp1*weak_mat_tmp14 + weak_mat_tmp10*weak_mat_tmp9 + weak_mat_tmp12*weak_mat_tmp13) + weak_mat_tmp0*weak_mat_tmp4;
    material[1] = gu[1]*weak_mat_tmp4 + mu*(weak_mat_tmp10*weak_mat_tmp2 + weak_mat_tmp13*weak_mat_tmp15 + weak_mat_tmp14*weak_mat_tmp9);
    material[2] = gu[2]*weak_mat_tmp4 + mu*(weak_mat_tmp10*weak_mat_tmp15 + weak_mat_tmp12*weak_mat_tmp14 + weak_mat_tmp13*weak_mat_tmp3);
    material[3] = gu[3]*weak_mat_tmp4 + mu*(weak_mat_tmp1*weak_mat_tmp16 + weak_mat_tmp12*weak_mat_tmp17 + weak_mat_tmp18*weak_mat_tmp9);
    material[4] = mu*(weak_mat_tmp15*weak_mat_tmp17 + weak_mat_tmp16*weak_mat_tmp9 + weak_mat_tmp18*weak_mat_tmp2) + weak_mat_tmp4*weak_mat_tmp7;
    material[5] = gu[5]*weak_mat_tmp4 + mu*(weak_mat_tmp12*weak_mat_tmp16 + weak_mat_tmp15*weak_mat_tmp18 + weak_mat_tmp17*weak_mat_tmp3);
    material[6] = gu[6]*weak_mat_tmp4 + mu*(weak_mat_tmp1*weak_mat_tmp19 + weak_mat_tmp12*weak_mat_tmp21 + weak_mat_tmp20*weak_mat_tmp9);
    material[7] = gu[7]*weak_mat_tmp4 + mu*(weak_mat_tmp15*weak_mat_tmp21 + weak_mat_tmp19*weak_mat_tmp9 + weak_mat_tmp2*weak_mat_tmp20);
    material[8] = mu*(weak_mat_tmp12*weak_mat_tmp19 + weak_mat_tmp15*weak_mat_tmp20 + weak_mat_tmp21*weak_mat_tmp3) + weak_mat_tmp11*weak_mat_tmp4;
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
static __host__ __device__ __forceinline__ void saint_venant_kirchhoff_d3_tensor_product_apply_block(
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
        s_t *const RSTR out_streams[NS * 3]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NQ1 = integer_root(NQ, 3);
  static constexpr int NS1 = integer_root(NS, 3);
  static_assert(ipow(NQ1, 3) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 3) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[NQ * 9 * VS];
  s_t grad_h_ref_q[NQ * 9 * VS];
  s_t loperand_q[NQ * 9 * VS];
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
      s_t trial_grad[9];
      const s_t idet = s_t(1) / det_value0;
      gu[0] = (gu_ref0[0] * adj_value0 + gu_ref1[0] * adj_value3 + gu_ref2[0] * adj_value6) * idet;
      trial_grad[0] = (grad_h_ref0[0] * adj_value0 + grad_h_ref1[0] * adj_value3 + grad_h_ref2[0] * adj_value6) * idet;
      gu[1] = (gu_ref0[0] * adj_value1 + gu_ref1[0] * adj_value4 + gu_ref2[0] * adj_value7) * idet;
      trial_grad[1] = (grad_h_ref0[0] * adj_value1 + grad_h_ref1[0] * adj_value4 + grad_h_ref2[0] * adj_value7) * idet;
      gu[2] = (gu_ref0[0] * adj_value2 + gu_ref1[0] * adj_value5 + gu_ref2[0] * adj_value8) * idet;
      trial_grad[2] = (grad_h_ref0[0] * adj_value2 + grad_h_ref1[0] * adj_value5 + grad_h_ref2[0] * adj_value8) * idet;
      gu[3] = (gu_ref3[0] * adj_value0 + gu_ref4[0] * adj_value3 + gu_ref5[0] * adj_value6) * idet;
      trial_grad[3] = (grad_h_ref3[0] * adj_value0 + grad_h_ref4[0] * adj_value3 + grad_h_ref5[0] * adj_value6) * idet;
      gu[4] = (gu_ref3[0] * adj_value1 + gu_ref4[0] * adj_value4 + gu_ref5[0] * adj_value7) * idet;
      trial_grad[4] = (grad_h_ref3[0] * adj_value1 + grad_h_ref4[0] * adj_value4 + grad_h_ref5[0] * adj_value7) * idet;
      gu[5] = (gu_ref3[0] * adj_value2 + gu_ref4[0] * adj_value5 + gu_ref5[0] * adj_value8) * idet;
      trial_grad[5] = (grad_h_ref3[0] * adj_value2 + grad_h_ref4[0] * adj_value5 + grad_h_ref5[0] * adj_value8) * idet;
      gu[6] = (gu_ref6[0] * adj_value0 + gu_ref7[0] * adj_value3 + gu_ref8[0] * adj_value6) * idet;
      trial_grad[6] = (grad_h_ref6[0] * adj_value0 + grad_h_ref7[0] * adj_value3 + grad_h_ref8[0] * adj_value6) * idet;
      gu[7] = (gu_ref6[0] * adj_value1 + gu_ref7[0] * adj_value4 + gu_ref8[0] * adj_value7) * idet;
      trial_grad[7] = (grad_h_ref6[0] * adj_value1 + grad_h_ref7[0] * adj_value4 + grad_h_ref8[0] * adj_value7) * idet;
      gu[8] = (gu_ref6[0] * adj_value2 + gu_ref7[0] * adj_value5 + gu_ref8[0] * adj_value8) * idet;
      trial_grad[8] = (grad_h_ref6[0] * adj_value2 + grad_h_ref7[0] * adj_value5 + grad_h_ref8[0] * adj_value8) * idet;
      s_t loperand[9];
    s_t material[9];
    const s_t weak_mat_tmp0 = gu[3]*mu;
    const s_t weak_mat_tmp1 = gu[0] + s_t(1);
    const s_t weak_mat_tmp2 = lmbda*weak_mat_tmp1;
    const s_t weak_mat_tmp3 = gu[2]*weak_mat_tmp0 + gu[5]*weak_mat_tmp2;
    const s_t weak_mat_tmp4 = gu[6]*mu;
    const s_t weak_mat_tmp5 = gu[1]*weak_mat_tmp4 + gu[7]*weak_mat_tmp2;
    const s_t weak_mat_tmp6 = gu[4] + s_t(1);
    const s_t weak_mat_tmp7 = gu[1]*weak_mat_tmp0 + weak_mat_tmp2*weak_mat_tmp6;
    const s_t weak_mat_tmp8 = gu[8] + s_t(1);
    const s_t weak_mat_tmp9 = gu[2]*weak_mat_tmp4 + weak_mat_tmp2*weak_mat_tmp8;
    const s_t weak_mat_tmp10 = gu[1]*weak_mat_tmp1;
    const s_t weak_mat_tmp11 = gu[6]*gu[7];
    const s_t weak_mat_tmp12 = gu[3]*weak_mat_tmp6;
    const s_t weak_mat_tmp13 = lmbda*weak_mat_tmp10 + mu*(s_t(2)*weak_mat_tmp10 + weak_mat_tmp11 + weak_mat_tmp12);
    const s_t weak_mat_tmp14 = gu[2]*weak_mat_tmp1;
    const s_t weak_mat_tmp15 = gu[3]*gu[5];
    const s_t weak_mat_tmp16 = gu[6]*weak_mat_tmp8;
    const s_t weak_mat_tmp17 = lmbda*weak_mat_tmp14 + mu*(s_t(2)*weak_mat_tmp14 + weak_mat_tmp15 + weak_mat_tmp16);
    const s_t weak_mat_tmp18 = gu[3]*weak_mat_tmp1;
    const s_t weak_mat_tmp19 = gu[2]*gu[5];
    const s_t weak_mat_tmp20 = gu[1]*weak_mat_tmp6;
    const s_t weak_mat_tmp21 = lmbda*weak_mat_tmp18 + mu*(s_t(2)*weak_mat_tmp18 + weak_mat_tmp19 + weak_mat_tmp20);
    const s_t weak_mat_tmp22 = gu[6]*weak_mat_tmp1;
    const s_t weak_mat_tmp23 = gu[1]*gu[7];
    const s_t weak_mat_tmp24 = gu[2]*weak_mat_tmp8;
    const s_t weak_mat_tmp25 = lmbda*weak_mat_tmp22 + mu*(s_t(2)*weak_mat_tmp22 + weak_mat_tmp23 + weak_mat_tmp24);
    const s_t weak_mat_tmp26 = pow_2(weak_mat_tmp1);
    const s_t weak_mat_tmp27 = pow_2(gu[1]);
    const s_t weak_mat_tmp28 = pow_2(gu[3]);
    const s_t weak_mat_tmp29 = weak_mat_tmp27 + weak_mat_tmp28;
    const s_t weak_mat_tmp30 = pow_2(gu[6]);
    const s_t weak_mat_tmp31 = pow_2(gu[2]);
    const s_t weak_mat_tmp32 = weak_mat_tmp31 + s_t(-1);
    const s_t weak_mat_tmp33 = weak_mat_tmp30 + weak_mat_tmp32;
    const s_t weak_mat_tmp34 = pow_2(gu[5]);
    const s_t weak_mat_tmp35 = pow_2(gu[7]);
    const s_t weak_mat_tmp36 = lmbda*(((s_t(1) / s_t(2)))*pow_2(gu[0]) + gu[0] + ((s_t(1) / s_t(2)))*pow_2(gu[4]) + gu[4] + ((s_t(1) / s_t(2)))*pow_2(gu[8]) + gu[8] + ((s_t(1) / s_t(2)))*weak_mat_tmp27 + ((s_t(1) / s_t(2)))*weak_mat_tmp28 + ((s_t(1) / s_t(2)))*weak_mat_tmp30 + ((s_t(1) / s_t(2)))*weak_mat_tmp31 + ((s_t(1) / s_t(2)))*weak_mat_tmp34 + ((s_t(1) / s_t(2)))*weak_mat_tmp35);
    const s_t weak_mat_tmp37 = gu[1]*lmbda;
    const s_t weak_mat_tmp38 = mu*weak_mat_tmp6;
    const s_t weak_mat_tmp39 = gu[2]*weak_mat_tmp38 + gu[5]*weak_mat_tmp37;
    const s_t weak_mat_tmp40 = gu[7]*mu;
    const s_t weak_mat_tmp41 = gu[6]*weak_mat_tmp37 + weak_mat_tmp1*weak_mat_tmp40;
    const s_t weak_mat_tmp42 = gu[2]*weak_mat_tmp40 + weak_mat_tmp37*weak_mat_tmp8;
    const s_t weak_mat_tmp43 = gu[3]*weak_mat_tmp37 + weak_mat_tmp1*weak_mat_tmp38;
    const s_t weak_mat_tmp44 = gu[1]*gu[2];
    const s_t weak_mat_tmp45 = gu[5]*weak_mat_tmp6;
    const s_t weak_mat_tmp46 = gu[7]*weak_mat_tmp8;
    const s_t weak_mat_tmp47 = lmbda*weak_mat_tmp44 + mu*(s_t(2)*weak_mat_tmp44 + weak_mat_tmp45 + weak_mat_tmp46);
    const s_t weak_mat_tmp48 = lmbda*weak_mat_tmp23 + mu*(weak_mat_tmp22 + s_t(2)*weak_mat_tmp23 + weak_mat_tmp24);
    const s_t weak_mat_tmp49 = lmbda*weak_mat_tmp20 + mu*(weak_mat_tmp18 + weak_mat_tmp19 + s_t(2)*weak_mat_tmp20);
    const s_t weak_mat_tmp50 = pow_2(weak_mat_tmp6);
    const s_t weak_mat_tmp51 = weak_mat_tmp26 + weak_mat_tmp50;
    const s_t weak_mat_tmp52 = gu[2]*lmbda;
    const s_t weak_mat_tmp53 = gu[5]*mu;
    const s_t weak_mat_tmp54 = gu[3]*weak_mat_tmp52 + weak_mat_tmp1*weak_mat_tmp53;
    const s_t weak_mat_tmp55 = gu[1]*weak_mat_tmp53 + weak_mat_tmp52*weak_mat_tmp6;
    const s_t weak_mat_tmp56 = mu*weak_mat_tmp8;
    const s_t weak_mat_tmp57 = gu[1]*weak_mat_tmp56 + gu[7]*weak_mat_tmp52;
    const s_t weak_mat_tmp58 = gu[6]*weak_mat_tmp52 + weak_mat_tmp1*weak_mat_tmp56;
    const s_t weak_mat_tmp59 = lmbda*weak_mat_tmp19 + mu*(weak_mat_tmp18 + s_t(2)*weak_mat_tmp19 + weak_mat_tmp20);
    const s_t weak_mat_tmp60 = lmbda*weak_mat_tmp24 + mu*(weak_mat_tmp22 + weak_mat_tmp23 + s_t(2)*weak_mat_tmp24);
    const s_t weak_mat_tmp61 = weak_mat_tmp34 + s_t(-1);
    const s_t weak_mat_tmp62 = pow_2(weak_mat_tmp8);
    const s_t weak_mat_tmp63 = weak_mat_tmp26 + weak_mat_tmp62;
    const s_t weak_mat_tmp64 = gu[3]*lmbda;
    const s_t weak_mat_tmp65 = gu[7]*weak_mat_tmp64 + weak_mat_tmp4*weak_mat_tmp6;
    const s_t weak_mat_tmp66 = gu[5]*weak_mat_tmp4 + weak_mat_tmp64*weak_mat_tmp8;
    const s_t weak_mat_tmp67 = lmbda*weak_mat_tmp15 + mu*(weak_mat_tmp14 + s_t(2)*weak_mat_tmp15 + weak_mat_tmp16);
    const s_t weak_mat_tmp68 = gu[3]*gu[6];
    const s_t weak_mat_tmp69 = gu[5]*weak_mat_tmp8;
    const s_t weak_mat_tmp70 = gu[7]*weak_mat_tmp6;
    const s_t weak_mat_tmp71 = lmbda*weak_mat_tmp68 + mu*(s_t(2)*weak_mat_tmp68 + weak_mat_tmp69 + weak_mat_tmp70);
    const s_t weak_mat_tmp72 = lmbda*weak_mat_tmp12 + mu*(weak_mat_tmp10 + weak_mat_tmp11 + s_t(2)*weak_mat_tmp12);
    const s_t weak_mat_tmp73 = lmbda*weak_mat_tmp6;
    const s_t weak_mat_tmp74 = gu[6]*weak_mat_tmp73 + gu[7]*weak_mat_tmp0;
    const s_t weak_mat_tmp75 = gu[5]*weak_mat_tmp40 + weak_mat_tmp73*weak_mat_tmp8;
    const s_t weak_mat_tmp76 = lmbda*weak_mat_tmp45 + mu*(weak_mat_tmp44 + s_t(2)*weak_mat_tmp45 + weak_mat_tmp46);
    const s_t weak_mat_tmp77 = lmbda*weak_mat_tmp70 + mu*(weak_mat_tmp68 + weak_mat_tmp69 + s_t(2)*weak_mat_tmp70);
    const s_t weak_mat_tmp78 = gu[5]*lmbda;
    const s_t weak_mat_tmp79 = gu[6]*weak_mat_tmp78 + weak_mat_tmp0*weak_mat_tmp8;
    const s_t weak_mat_tmp80 = gu[7]*weak_mat_tmp78 + weak_mat_tmp38*weak_mat_tmp8;
    const s_t weak_mat_tmp81 = lmbda*weak_mat_tmp69 + mu*(weak_mat_tmp68 + s_t(2)*weak_mat_tmp69 + weak_mat_tmp70);
    const s_t weak_mat_tmp82 = weak_mat_tmp50 + weak_mat_tmp62;
    const s_t weak_mat_tmp83 = lmbda*weak_mat_tmp11 + mu*(weak_mat_tmp10 + s_t(2)*weak_mat_tmp11 + weak_mat_tmp12);
    const s_t weak_mat_tmp84 = lmbda*weak_mat_tmp16 + mu*(weak_mat_tmp14 + weak_mat_tmp15 + s_t(2)*weak_mat_tmp16);
    const s_t weak_mat_tmp85 = lmbda*weak_mat_tmp46 + mu*(weak_mat_tmp44 + weak_mat_tmp45 + s_t(2)*weak_mat_tmp46);
    material[0] = trial_grad[0]*(lmbda*weak_mat_tmp26 + mu*(s_t(3)*weak_mat_tmp26 + weak_mat_tmp29 + weak_mat_tmp33) + weak_mat_tmp36) + trial_grad[1]*weak_mat_tmp13 + trial_grad[2]*weak_mat_tmp17 + trial_grad[3]*weak_mat_tmp21 + trial_grad[4]*weak_mat_tmp7 + trial_grad[5]*weak_mat_tmp3 + trial_grad[6]*weak_mat_tmp25 + trial_grad[7]*weak_mat_tmp5 + trial_grad[8]*weak_mat_tmp9;
    material[1] = trial_grad[0]*weak_mat_tmp13 + trial_grad[1]*(lmbda*weak_mat_tmp27 + mu*(s_t(3)*weak_mat_tmp27 + weak_mat_tmp32 + weak_mat_tmp35 + weak_mat_tmp51) + weak_mat_tmp36) + trial_grad[2]*weak_mat_tmp47 + trial_grad[3]*weak_mat_tmp43 + trial_grad[4]*weak_mat_tmp49 + trial_grad[5]*weak_mat_tmp39 + trial_grad[6]*weak_mat_tmp41 + trial_grad[7]*weak_mat_tmp48 + trial_grad[8]*weak_mat_tmp42;
    material[2] = trial_grad[0]*weak_mat_tmp17 + trial_grad[1]*weak_mat_tmp47 + trial_grad[2]*(lmbda*weak_mat_tmp31 + mu*(weak_mat_tmp27 + s_t(3)*weak_mat_tmp31 + weak_mat_tmp61 + weak_mat_tmp63) + weak_mat_tmp36) + trial_grad[3]*weak_mat_tmp54 + trial_grad[4]*weak_mat_tmp55 + trial_grad[5]*weak_mat_tmp59 + trial_grad[6]*weak_mat_tmp58 + trial_grad[7]*weak_mat_tmp57 + trial_grad[8]*weak_mat_tmp60;
    material[3] = trial_grad[0]*weak_mat_tmp21 + trial_grad[1]*weak_mat_tmp43 + trial_grad[2]*weak_mat_tmp54 + trial_grad[3]*(lmbda*weak_mat_tmp28 + mu*(s_t(3)*weak_mat_tmp28 + weak_mat_tmp30 + weak_mat_tmp51 + weak_mat_tmp61) + weak_mat_tmp36) + trial_grad[4]*weak_mat_tmp72 + trial_grad[5]*weak_mat_tmp67 + trial_grad[6]*weak_mat_tmp71 + trial_grad[7]*weak_mat_tmp65 + trial_grad[8]*weak_mat_tmp66;
    material[4] = trial_grad[0]*weak_mat_tmp7 + trial_grad[1]*weak_mat_tmp49 + trial_grad[2]*weak_mat_tmp55 + trial_grad[3]*weak_mat_tmp72 + trial_grad[4]*(lmbda*weak_mat_tmp50 + mu*(weak_mat_tmp29 + weak_mat_tmp35 + s_t(3)*weak_mat_tmp50 + weak_mat_tmp61) + weak_mat_tmp36) + trial_grad[5]*weak_mat_tmp76 + trial_grad[6]*weak_mat_tmp74 + trial_grad[7]*weak_mat_tmp77 + trial_grad[8]*weak_mat_tmp75;
    material[5] = trial_grad[0]*weak_mat_tmp3 + trial_grad[1]*weak_mat_tmp39 + trial_grad[2]*weak_mat_tmp59 + trial_grad[3]*weak_mat_tmp67 + trial_grad[4]*weak_mat_tmp76 + trial_grad[5]*(lmbda*weak_mat_tmp34 + mu*(weak_mat_tmp28 + weak_mat_tmp32 + s_t(3)*weak_mat_tmp34 + weak_mat_tmp82) + weak_mat_tmp36) + trial_grad[6]*weak_mat_tmp79 + trial_grad[7]*weak_mat_tmp80 + trial_grad[8]*weak_mat_tmp81;
    material[6] = trial_grad[0]*weak_mat_tmp25 + trial_grad[1]*weak_mat_tmp41 + trial_grad[2]*weak_mat_tmp58 + trial_grad[3]*weak_mat_tmp71 + trial_grad[4]*weak_mat_tmp74 + trial_grad[5]*weak_mat_tmp79 + trial_grad[6]*(lmbda*weak_mat_tmp30 + mu*(weak_mat_tmp28 + s_t(3)*weak_mat_tmp30 + weak_mat_tmp35 + weak_mat_tmp63 + s_t(-1)) + weak_mat_tmp36) + trial_grad[7]*weak_mat_tmp83 + trial_grad[8]*weak_mat_tmp84;
    material[7] = trial_grad[0]*weak_mat_tmp5 + trial_grad[1]*weak_mat_tmp48 + trial_grad[2]*weak_mat_tmp57 + trial_grad[3]*weak_mat_tmp65 + trial_grad[4]*weak_mat_tmp77 + trial_grad[5]*weak_mat_tmp80 + trial_grad[6]*weak_mat_tmp83 + trial_grad[7]*(lmbda*weak_mat_tmp35 + mu*(weak_mat_tmp27 + weak_mat_tmp30 + s_t(3)*weak_mat_tmp35 + weak_mat_tmp82 + s_t(-1)) + weak_mat_tmp36) + trial_grad[8]*weak_mat_tmp85;
    material[8] = trial_grad[0]*weak_mat_tmp9 + trial_grad[1]*weak_mat_tmp42 + trial_grad[2]*weak_mat_tmp60 + trial_grad[3]*weak_mat_tmp66 + trial_grad[4]*weak_mat_tmp75 + trial_grad[5]*weak_mat_tmp81 + trial_grad[6]*weak_mat_tmp84 + trial_grad[7]*weak_mat_tmp85 + trial_grad[8]*(lmbda*weak_mat_tmp62 + mu*(weak_mat_tmp33 + weak_mat_tmp34 + weak_mat_tmp35 + s_t(3)*weak_mat_tmp62) + weak_mat_tmp36);
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
