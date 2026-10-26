#ifndef MOONEY_RIVLIN_PRONY_D2_TENSOR_PRODUCT_LOCAL_CUH
#define MOONEY_RIVLIN_PRONY_D2_TENSOR_PRODUCT_LOCAL_CUH
#include <math.h>
#include <stddef.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_GENERATED_SCALAR_T
#endif
#endif
#include "kernel_math.cuh"
#include "tensor_product_kernels.cuh"
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

template <typename s_t, int NQ, int NS>
static __host__ __device__ __forceinline__ void mooney_rivlin_prony_d2_tensor_product_objective_block(
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const ptrdiff_t qp_stride,
        const s_t *const RSTR u_streams[NS * 2],
        const s_t *const RSTR h_streams[NS * 2],
        const int nsteps,
        const s_t *const RSTR steps,
        const ptrdiff_t value_stride,
        s_t *const RSTR value
) {
  static_assert(NQ > 0, "NQ must be positive");
  static constexpr int NQ1 = integer_root(NQ, 2);
  static constexpr int NS1 = integer_root(NS, 2);
  static_assert(ipow(NQ1, 2) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 2) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[4 * NQ];
  s_t grad_h_ref_q[4 * NQ];
  tensor_gradient_scalar<s_t, NQ, NS, 2, 2>(shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0]);
  tensor_gradient_scalar<s_t, NQ, NS, 2, 2>(shape_1d, grad_1d, h_streams, 0, &grad_h_ref_q[0]);
  tensor_gradient_scalar<s_t, NQ, NS, 2, 2>(shape_1d, grad_1d, u_streams, 1, &gu_ref_q[2 * NQ]);
  tensor_gradient_scalar<s_t, NQ, NS, 2, 2>(shape_1d, grad_1d, h_streams, 1, &grad_h_ref_q[2 * NQ]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = q / NQ1;
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
    const s_t *const RSTR gu_ref0 = &gu_ref_q[(2 * q)];
    const s_t *const RSTR gu_ref1 = &gu_ref_q[(2 * q + 1)];
    const s_t *const RSTR gu_ref2 = &gu_ref_q[(2 * (NQ + q))];
    const s_t *const RSTR gu_ref3 = &gu_ref_q[(2 * (NQ + q) + 1)];
    const s_t *const RSTR grad_h_ref0 = &grad_h_ref_q[(2 * q)];
    const s_t *const RSTR grad_h_ref1 = &grad_h_ref_q[(2 * q + 1)];
    const s_t *const RSTR grad_h_ref2 = &grad_h_ref_q[(2 * (NQ + q))];
    const s_t *const RSTR grad_h_ref3 = &grad_h_ref_q[(2 * (NQ + q) + 1)];
    const s_t *const RSTR adj_q0 = adj0 + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adj1 + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adj2 + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adj3 + q * geometry_stride;
    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;
    const s_t *const RSTR iso_history_q = iso_history + q * qp_stride;
    s_t gu_base_v[4];
    s_t trial_grad_v[4];
    {
      const s_t adj_value0 = adj_q0[0];
      const s_t adj_value1 = adj_q1[0];
      const s_t adj_value2 = adj_q2[0];
      const s_t adj_value3 = adj_q3[0];
      const s_t det_value0 = det_q0[0];
      const s_t idet = s_t(1) / det_value0;
      gu_base_v[0] = (gu_ref0[0] * adj_value0 + gu_ref1[0] * adj_value2) * idet;
      trial_grad_v[0] = (grad_h_ref0[0] * adj_value0 + grad_h_ref1[0] * adj_value2) * idet;
      gu_base_v[1] = (gu_ref0[0] * adj_value1 + gu_ref1[0] * adj_value3) * idet;
      trial_grad_v[1] = (grad_h_ref0[0] * adj_value1 + grad_h_ref1[0] * adj_value3) * idet;
      gu_base_v[2] = (gu_ref2[0] * adj_value0 + gu_ref3[0] * adj_value2) * idet;
      trial_grad_v[2] = (grad_h_ref2[0] * adj_value0 + grad_h_ref3[0] * adj_value2) * idet;
      gu_base_v[3] = (gu_ref2[0] * adj_value1 + gu_ref3[0] * adj_value3) * idet;
      trial_grad_v[3] = (grad_h_ref2[0] * adj_value1 + grad_h_ref3[0] * adj_value3) * idet;
    }
    for (int step = 0; step < nsteps; ++step) {
      const s_t alpha = steps[step];
      {
        const s_t det_value0 = det_q0[0];
        const s_t iso_history_qp[3] = {iso_history_q[0], iso_history_q[1 * NQ * qp_stride + 0], iso_history_q[2 * NQ * qp_stride + 0]};
        s_t gu[4];
        gu[0] = gu_base_v[0] + alpha * trial_grad_v[0];
        gu[1] = gu_base_v[1] + alpha * trial_grad_v[1];
        gu[2] = gu_base_v[2] + alpha * trial_grad_v[2];
        gu[3] = gu_base_v[3] + alpha * trial_grad_v[3];
    const s_t weak_obj_tmp0 = pow_2(gu[2]);
    const s_t weak_obj_tmp1 = pow_2(gu[1]);
    const s_t weak_obj_tmp2 = gu[0] + s_t(1);
    const s_t weak_obj_tmp3 = gu[3] + s_t(1);
    const s_t weak_obj_tmp4 = gu[1]*weak_obj_tmp2 + gu[2]*weak_obj_tmp3;
    const s_t weak_obj_tmp5 = gu[1]*gu[2];
    const s_t weak_obj_tmp6 = weak_obj_tmp2*weak_obj_tmp3 - weak_obj_tmp5;
    const s_t weak_obj_tmp7 = weak_obj_tmp1 + pow_2(weak_obj_tmp3);
    const s_t weak_obj_tmp8 = weak_obj_tmp0 + pow_2(weak_obj_tmp2);
    const s_t weak_obj_tmp9 = weak_obj_tmp7 + weak_obj_tmp8;
    value[step * value_stride + 0] += qw * det_value0 * (((s_t(1) / s_t(2)))*K*pow_2(gu[0]*gu[3] + gu[0] + gu[3] - weak_obj_tmp5) + ((s_t(1) / s_t(2)))*iso_history_qp[0]*(pow_2(gu[0]) + s_t(2)*gu[0] + weak_obj_tmp0) + ((s_t(1) / s_t(2)))*iso_history_qp[1]*(pow_2(gu[3]) + s_t(2)*gu[3] + weak_obj_tmp1) + iso_history_qp[2]*weak_obj_tmp4 + iso_relaxation_gamma*(C01*(s_t(-3) + (-pow_2(weak_obj_tmp4) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp7) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp8) + ((s_t(1) / s_t(2)))*pow_2(weak_obj_tmp9) + weak_obj_tmp9)/pow(weak_obj_tmp6, (s_t(4) / s_t(3)))) + C10*(s_t(-3) + (weak_obj_tmp9 + s_t(1))/pow(weak_obj_tmp6, (s_t(2) / s_t(3))))));
      }
    }
  }
}

template <typename s_t, int NQ, int NS>
static __host__ __device__ __forceinline__ void mooney_rivlin_prony_d2_tensor_product_gradient_block(
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const ptrdiff_t qp_stride,
        const s_t *const RSTR u_streams[NS * 2],
        s_t *const RSTR out_streams[NS * 2]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static constexpr int NQ1 = integer_root(NQ, 2);
  static constexpr int NS1 = integer_root(NS, 2);
  static_assert(ipow(NQ1, 2) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 2) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[4 * NQ];
  s_t loperand_q[4 * NQ];
  tensor_gradient_scalar<s_t, NQ, NS, 2, 2>(shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0]);
  tensor_gradient_scalar<s_t, NQ, NS, 2, 2>(shape_1d, grad_1d, u_streams, 1, &gu_ref_q[2 * NQ]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = q / NQ1;
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
    const s_t *const RSTR gu_ref0 = &gu_ref_q[(2 * q)];
    const s_t *const RSTR gu_ref1 = &gu_ref_q[(2 * q + 1)];
    const s_t *const RSTR gu_ref2 = &gu_ref_q[(2 * (NQ + q))];
    const s_t *const RSTR gu_ref3 = &gu_ref_q[(2 * (NQ + q) + 1)];
    s_t *const RSTR loperand0 = &loperand_q[(2 * q)];
    s_t *const RSTR loperand1 = &loperand_q[(2 * q + 1)];
    s_t *const RSTR loperand2 = &loperand_q[(2 * (NQ + q))];
    s_t *const RSTR loperand3 = &loperand_q[(2 * (NQ + q) + 1)];
    const s_t *const RSTR adj_q0 = adj0 + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adj1 + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adj2 + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adj3 + q * geometry_stride;
    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;
    const s_t *const RSTR iso_history_q = iso_history + q * qp_stride;
    {
      const s_t adj_value0 = adj_q0[0];
      const s_t adj_value1 = adj_q1[0];
      const s_t adj_value2 = adj_q2[0];
      const s_t adj_value3 = adj_q3[0];
      const s_t det_value0 = det_q0[0];
      const s_t iso_history_qp[3] = {iso_history_q[0], iso_history_q[1 * NQ * qp_stride + 0], iso_history_q[2 * NQ * qp_stride + 0]};
      s_t gu[4];
      const s_t idet = s_t(1) / det_value0;
      gu[0] = (gu_ref0[0] * adj_value0 + gu_ref1[0] * adj_value2) * idet;
      gu[1] = (gu_ref0[0] * adj_value1 + gu_ref1[0] * adj_value3) * idet;
      gu[2] = (gu_ref2[0] * adj_value0 + gu_ref3[0] * adj_value2) * idet;
      gu[3] = (gu_ref2[0] * adj_value1 + gu_ref3[0] * adj_value3) * idet;
    const s_t weak_mat_tmp0 = gu[0] + s_t(1);
    const s_t weak_mat_tmp1 = gu[3] + s_t(1);
    const s_t weak_mat_tmp2 = gu[1]*gu[2];
    const s_t weak_mat_tmp3 = K*(gu[0]*gu[3] + gu[0] + gu[3] - weak_mat_tmp2);
    const s_t weak_mat_tmp4 = weak_mat_tmp0*weak_mat_tmp1 - weak_mat_tmp2;
    const s_t weak_mat_tmp5 = pow(weak_mat_tmp4, (s_t(-2) / s_t(3)));
    const s_t weak_mat_tmp6 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp7 = pow_2(gu[2]) + pow_2(weak_mat_tmp0);
    const s_t weak_mat_tmp8 = pow_2(gu[1]) + pow_2(weak_mat_tmp1);
    const s_t weak_mat_tmp9 = weak_mat_tmp7 + weak_mat_tmp8;
    const s_t weak_mat_tmp10 = ((s_t(2) / s_t(3)))*(weak_mat_tmp9 + s_t(1))/pow(weak_mat_tmp4, (s_t(5) / s_t(3)));
    const s_t weak_mat_tmp11 = pow(weak_mat_tmp4, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp12 = gu[1]*weak_mat_tmp0 + gu[2]*weak_mat_tmp1;
    const s_t weak_mat_tmp13 = s_t(2)*gu[1];
    const s_t weak_mat_tmp14 = ((s_t(4) / s_t(3)))*(-pow_2(weak_mat_tmp12) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp7) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp8) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp9) + weak_mat_tmp9)/pow(weak_mat_tmp4, (s_t(7) / s_t(3)));
    const s_t weak_mat_tmp15 = s_t(2)*gu[2];
    const s_t weak_mat_tmp16 = s_t(2)*weak_mat_tmp1;
    const s_t material0 = gu[1]*iso_history_qp[2] + iso_history_qp[0]*weak_mat_tmp0 + iso_relaxation_gamma*(C01*(-weak_mat_tmp1*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu[0] - weak_mat_tmp12*weak_mat_tmp13 - weak_mat_tmp6*weak_mat_tmp7 + weak_mat_tmp6*weak_mat_tmp9 + s_t(2))) + C10*(-weak_mat_tmp1*weak_mat_tmp10 + weak_mat_tmp5*weak_mat_tmp6)) + weak_mat_tmp1*weak_mat_tmp3;
    const s_t material1 = gu[1]*iso_history_qp[1] - gu[2]*weak_mat_tmp3 + iso_history_qp[2]*weak_mat_tmp0 + iso_relaxation_gamma*(C01*(gu[2]*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu[1]*weak_mat_tmp9 + s_t(2)*gu[1] - weak_mat_tmp12*weak_mat_tmp6 - weak_mat_tmp13*weak_mat_tmp8)) + C10*(gu[2]*weak_mat_tmp10 + weak_mat_tmp13*weak_mat_tmp5));
    const s_t material2 = -gu[1]*weak_mat_tmp3 + gu[2]*iso_history_qp[0] + iso_history_qp[2]*weak_mat_tmp1 + iso_relaxation_gamma*(C01*(gu[1]*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu[2]*weak_mat_tmp9 + s_t(2)*gu[2] - weak_mat_tmp12*weak_mat_tmp16 - weak_mat_tmp15*weak_mat_tmp7)) + C10*(gu[1]*weak_mat_tmp10 + weak_mat_tmp15*weak_mat_tmp5));
    const s_t material3 = gu[2]*iso_history_qp[2] + iso_history_qp[1]*weak_mat_tmp1 + iso_relaxation_gamma*(C01*(-weak_mat_tmp0*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu[3] - weak_mat_tmp12*weak_mat_tmp15 - weak_mat_tmp16*weak_mat_tmp8 + weak_mat_tmp16*weak_mat_tmp9 + s_t(2))) + C10*(-weak_mat_tmp0*weak_mat_tmp10 + s_t(2)*weak_mat_tmp1*weak_mat_tmp5)) + weak_mat_tmp0*weak_mat_tmp3;
      loperand0[0] = qw * (material0 * adj_value0 + material1 * adj_value1);
      loperand1[0] = qw * (material0 * adj_value2 + material1 * adj_value3);
      loperand2[0] = qw * (material2 * adj_value0 + material3 * adj_value1);
      loperand3[0] = qw * (material2 * adj_value2 + material3 * adj_value3);
    }
  }
  tensor_test_scalar<s_t, NQ, NS, 2, 2>(shape_1d, grad_1d, &loperand_q[0], out_streams, 0);
  tensor_test_scalar<s_t, NQ, NS, 2, 2>(shape_1d, grad_1d, &loperand_q[2 * NQ], out_streams, 1);
}

template <typename s_t, int NQ, int NS>
static __host__ __device__ __forceinline__ void mooney_rivlin_prony_d2_tensor_product_apply_block(
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const ptrdiff_t qp_stride,
        const s_t *const RSTR u_streams[NS * 2],
        const s_t *const RSTR h_streams[NS * 2],
        s_t *const RSTR out_streams[NS * 2]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static constexpr int NQ1 = integer_root(NQ, 2);
  static constexpr int NS1 = integer_root(NS, 2);
  static_assert(ipow(NQ1, 2) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 2) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[4 * NQ];
  s_t grad_h_ref_q[4 * NQ];
  s_t loperand_q[4 * NQ];
  tensor_gradient_scalar<s_t, NQ, NS, 2, 2>(shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0]);
  tensor_gradient_scalar<s_t, NQ, NS, 2, 2>(shape_1d, grad_1d, h_streams, 0, &grad_h_ref_q[0]);
  tensor_gradient_scalar<s_t, NQ, NS, 2, 2>(shape_1d, grad_1d, u_streams, 1, &gu_ref_q[2 * NQ]);
  tensor_gradient_scalar<s_t, NQ, NS, 2, 2>(shape_1d, grad_1d, h_streams, 1, &grad_h_ref_q[2 * NQ]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = q / NQ1;
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
    const s_t *const RSTR gu_ref0 = &gu_ref_q[(2 * q)];
    const s_t *const RSTR gu_ref1 = &gu_ref_q[(2 * q + 1)];
    const s_t *const RSTR gu_ref2 = &gu_ref_q[(2 * (NQ + q))];
    const s_t *const RSTR gu_ref3 = &gu_ref_q[(2 * (NQ + q) + 1)];
    const s_t *const RSTR grad_h_ref0 = &grad_h_ref_q[(2 * q)];
    const s_t *const RSTR grad_h_ref1 = &grad_h_ref_q[(2 * q + 1)];
    const s_t *const RSTR grad_h_ref2 = &grad_h_ref_q[(2 * (NQ + q))];
    const s_t *const RSTR grad_h_ref3 = &grad_h_ref_q[(2 * (NQ + q) + 1)];
    s_t *const RSTR loperand0 = &loperand_q[(2 * q)];
    s_t *const RSTR loperand1 = &loperand_q[(2 * q + 1)];
    s_t *const RSTR loperand2 = &loperand_q[(2 * (NQ + q))];
    s_t *const RSTR loperand3 = &loperand_q[(2 * (NQ + q) + 1)];
    const s_t *const RSTR adj_q0 = adj0 + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adj1 + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adj2 + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adj3 + q * geometry_stride;
    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;
    const s_t *const RSTR iso_history_q = iso_history + q * qp_stride;
    {
      const s_t adj_value0 = adj_q0[0];
      const s_t adj_value1 = adj_q1[0];
      const s_t adj_value2 = adj_q2[0];
      const s_t adj_value3 = adj_q3[0];
      const s_t det_value0 = det_q0[0];
      const s_t iso_history_qp[3] = {iso_history_q[0], iso_history_q[1 * NQ * qp_stride + 0], iso_history_q[2 * NQ * qp_stride + 0]};
      s_t gu[4];
      s_t trial_grad[4];
      const s_t idet = s_t(1) / det_value0;
      gu[0] = (gu_ref0[0] * adj_value0 + gu_ref1[0] * adj_value2) * idet;
      trial_grad[0] = (grad_h_ref0[0] * adj_value0 + grad_h_ref1[0] * adj_value2) * idet;
      gu[1] = (gu_ref0[0] * adj_value1 + gu_ref1[0] * adj_value3) * idet;
      trial_grad[1] = (grad_h_ref0[0] * adj_value1 + grad_h_ref1[0] * adj_value3) * idet;
      gu[2] = (gu_ref2[0] * adj_value0 + gu_ref3[0] * adj_value2) * idet;
      trial_grad[2] = (grad_h_ref2[0] * adj_value0 + grad_h_ref3[0] * adj_value2) * idet;
      gu[3] = (gu_ref2[0] * adj_value1 + gu_ref3[0] * adj_value3) * idet;
      trial_grad[3] = (grad_h_ref2[0] * adj_value1 + grad_h_ref3[0] * adj_value3) * idet;
    const s_t weak_mat_tmp0 = gu[3] + s_t(1);
    const s_t weak_mat_tmp1 = pow_2(weak_mat_tmp0);
    const s_t weak_mat_tmp2 = pow_2(gu[2]);
    const s_t weak_mat_tmp3 = gu[0] + s_t(1);
    const s_t weak_mat_tmp4 = pow_2(weak_mat_tmp3);
    const s_t weak_mat_tmp5 = weak_mat_tmp2 + weak_mat_tmp4;
    const s_t weak_mat_tmp6 = pow_2(gu[1]);
    const s_t weak_mat_tmp7 = weak_mat_tmp1 + weak_mat_tmp6;
    const s_t weak_mat_tmp8 = weak_mat_tmp5 + weak_mat_tmp7;
    const s_t weak_mat_tmp9 = weak_mat_tmp8 + s_t(1);
    const s_t weak_mat_tmp10 = gu[1]*gu[2];
    const s_t weak_mat_tmp11 = weak_mat_tmp0*weak_mat_tmp3 - weak_mat_tmp10;
    const s_t weak_mat_tmp12 = pow(weak_mat_tmp11, (s_t(-8) / s_t(3)));
    const s_t weak_mat_tmp13 = ((s_t(10) / s_t(9)))*weak_mat_tmp12*weak_mat_tmp9;
    const s_t weak_mat_tmp14 = s_t(2)/pow(weak_mat_tmp11, (s_t(2) / s_t(3)));
    const s_t weak_mat_tmp15 = weak_mat_tmp0*weak_mat_tmp3;
    const s_t weak_mat_tmp16 = pow(weak_mat_tmp11, (s_t(-5) / s_t(3)));
    const s_t weak_mat_tmp17 = ((s_t(8) / s_t(3)))*weak_mat_tmp16;
    const s_t weak_mat_tmp18 = weak_mat_tmp14 - weak_mat_tmp15*weak_mat_tmp17;
    const s_t weak_mat_tmp19 = pow(weak_mat_tmp11, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp20 = gu[1]*weak_mat_tmp3;
    const s_t weak_mat_tmp21 = gu[2]*weak_mat_tmp0;
    const s_t weak_mat_tmp22 = weak_mat_tmp20 + weak_mat_tmp21;
    const s_t weak_mat_tmp23 = s_t(2)*gu[1];
    const s_t weak_mat_tmp24 = s_t(2)*weak_mat_tmp3;
    const s_t weak_mat_tmp25 = s_t(2)*gu[0] - weak_mat_tmp22*weak_mat_tmp23 - weak_mat_tmp24*weak_mat_tmp5 + weak_mat_tmp24*weak_mat_tmp8 + s_t(2);
    const s_t weak_mat_tmp26 = pow(weak_mat_tmp11, (s_t(-7) / s_t(3)));
    const s_t weak_mat_tmp27 = ((s_t(8) / s_t(3)))*weak_mat_tmp26;
    const s_t weak_mat_tmp28 = -pow_2(weak_mat_tmp22) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp5) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp7) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp8) + weak_mat_tmp8;
    const s_t weak_mat_tmp29 = pow(weak_mat_tmp11, (s_t(-10) / s_t(3)));
    const s_t weak_mat_tmp30 = ((s_t(28) / s_t(9)))*weak_mat_tmp28*weak_mat_tmp29;
    const s_t weak_mat_tmp31 = gu[1]*weak_mat_tmp0;
    const s_t weak_mat_tmp32 = ((s_t(4) / s_t(3)))*weak_mat_tmp16;
    const s_t weak_mat_tmp33 = weak_mat_tmp20*weak_mat_tmp32;
    const s_t weak_mat_tmp34 = weak_mat_tmp21*weak_mat_tmp32;
    const s_t weak_mat_tmp35 = s_t(2)*gu[2];
    const s_t weak_mat_tmp36 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp37 = s_t(2)*gu[2]*weak_mat_tmp8 + s_t(2)*gu[2] - weak_mat_tmp22*weak_mat_tmp36 - weak_mat_tmp35*weak_mat_tmp5;
    const s_t weak_mat_tmp38 = ((s_t(4) / s_t(3)))*weak_mat_tmp26;
    const s_t weak_mat_tmp39 = weak_mat_tmp0*weak_mat_tmp38;
    const s_t weak_mat_tmp40 = -K*weak_mat_tmp31 + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu[1]*weak_mat_tmp25*weak_mat_tmp26 - weak_mat_tmp0*weak_mat_tmp19*weak_mat_tmp23 - weak_mat_tmp30*weak_mat_tmp31 - weak_mat_tmp37*weak_mat_tmp39) + C10*(-weak_mat_tmp13*weak_mat_tmp31 + weak_mat_tmp33 - weak_mat_tmp34));
    const s_t weak_mat_tmp41 = weak_mat_tmp31*weak_mat_tmp32;
    const s_t weak_mat_tmp42 = gu[2]*weak_mat_tmp3;
    const s_t weak_mat_tmp43 = weak_mat_tmp32*weak_mat_tmp42;
    const s_t weak_mat_tmp44 = s_t(2)*weak_mat_tmp19;
    const s_t weak_mat_tmp45 = s_t(2)*gu[1]*weak_mat_tmp8 + s_t(2)*gu[1] - weak_mat_tmp22*weak_mat_tmp24 - weak_mat_tmp23*weak_mat_tmp7;
    const s_t weak_mat_tmp46 = -K*weak_mat_tmp21 + iso_history_qp[2] + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu[2]*weak_mat_tmp25*weak_mat_tmp26 - weak_mat_tmp21*weak_mat_tmp30 - weak_mat_tmp21*weak_mat_tmp44 - weak_mat_tmp39*weak_mat_tmp45) + C10*(-weak_mat_tmp13*weak_mat_tmp21 - weak_mat_tmp41 + weak_mat_tmp43));
    const s_t weak_mat_tmp47 = K*(gu[0]*gu[3] + gu[0] + gu[3] - weak_mat_tmp10);
    const s_t weak_mat_tmp48 = ((s_t(2) / s_t(3)))*weak_mat_tmp16*weak_mat_tmp9;
    const s_t weak_mat_tmp49 = s_t(2)*gu[3] - weak_mat_tmp22*weak_mat_tmp35 - weak_mat_tmp36*weak_mat_tmp7 + weak_mat_tmp36*weak_mat_tmp8 + s_t(2);
    const s_t weak_mat_tmp50 = weak_mat_tmp28*weak_mat_tmp38;
    const s_t weak_mat_tmp51 = K*weak_mat_tmp15 + iso_relaxation_gamma*(C01*(((s_t(28) / s_t(9)))*weak_mat_tmp0*weak_mat_tmp28*weak_mat_tmp29*weak_mat_tmp3 + weak_mat_tmp19*(s_t(4)*weak_mat_tmp0*weak_mat_tmp3 - s_t(2)*weak_mat_tmp10) - weak_mat_tmp25*weak_mat_tmp3*weak_mat_tmp38 - weak_mat_tmp39*weak_mat_tmp49 - weak_mat_tmp50) + C10*(((s_t(10) / s_t(9)))*weak_mat_tmp0*weak_mat_tmp12*weak_mat_tmp3*weak_mat_tmp9 - weak_mat_tmp1*weak_mat_tmp32 - weak_mat_tmp32*weak_mat_tmp4 - weak_mat_tmp48)) + weak_mat_tmp47;
    const s_t weak_mat_tmp52 = weak_mat_tmp10*weak_mat_tmp17 + weak_mat_tmp14;
    const s_t weak_mat_tmp53 = weak_mat_tmp38*weak_mat_tmp45;
    const s_t weak_mat_tmp54 = -K*weak_mat_tmp42 + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu[2]*weak_mat_tmp26*weak_mat_tmp49 - weak_mat_tmp19*weak_mat_tmp3*weak_mat_tmp35 - weak_mat_tmp3*weak_mat_tmp53 - weak_mat_tmp30*weak_mat_tmp42) + C10*(-weak_mat_tmp13*weak_mat_tmp42 - weak_mat_tmp33 + weak_mat_tmp34));
    const s_t weak_mat_tmp55 = K*weak_mat_tmp10 + iso_relaxation_gamma*(C01*(gu[1]*weak_mat_tmp53 + gu[2]*weak_mat_tmp37*weak_mat_tmp38 + weak_mat_tmp10*weak_mat_tmp30 + weak_mat_tmp19*(s_t(4)*weak_mat_tmp10 - s_t(2)*weak_mat_tmp15) + weak_mat_tmp50) + C10*(weak_mat_tmp10*weak_mat_tmp13 + weak_mat_tmp2*weak_mat_tmp32 + weak_mat_tmp32*weak_mat_tmp6 + weak_mat_tmp48)) - weak_mat_tmp47;
    const s_t weak_mat_tmp56 = -K*weak_mat_tmp20 + iso_history_qp[2] + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu[1]*weak_mat_tmp26*weak_mat_tmp49 - weak_mat_tmp20*weak_mat_tmp30 - weak_mat_tmp20*weak_mat_tmp44 - weak_mat_tmp3*weak_mat_tmp37*weak_mat_tmp38) + C10*(-weak_mat_tmp13*weak_mat_tmp20 + weak_mat_tmp41 - weak_mat_tmp43));
    const s_t material0 = trial_grad[0]*(K*weak_mat_tmp1 + iso_history_qp[0] + iso_relaxation_gamma*(C01*(-weak_mat_tmp0*weak_mat_tmp25*weak_mat_tmp27 + weak_mat_tmp1*weak_mat_tmp30 + weak_mat_tmp19*(s_t(2)*weak_mat_tmp1 + s_t(2))) + C10*(weak_mat_tmp1*weak_mat_tmp13 + weak_mat_tmp18))) + trial_grad[1]*weak_mat_tmp46 + trial_grad[2]*weak_mat_tmp40 + trial_grad[3]*weak_mat_tmp51;
    const s_t material1 = trial_grad[0]*weak_mat_tmp46 + trial_grad[1]*(K*weak_mat_tmp2 + iso_history_qp[1] + iso_relaxation_gamma*(C01*(gu[2]*weak_mat_tmp27*weak_mat_tmp45 + weak_mat_tmp19*(s_t(2)*weak_mat_tmp2 + s_t(2)) + weak_mat_tmp2*weak_mat_tmp30) + C10*(weak_mat_tmp13*weak_mat_tmp2 + weak_mat_tmp52))) + trial_grad[2]*weak_mat_tmp55 + trial_grad[3]*weak_mat_tmp54;
    const s_t material2 = trial_grad[0]*weak_mat_tmp40 + trial_grad[1]*weak_mat_tmp55 + trial_grad[2]*(K*weak_mat_tmp6 + iso_history_qp[0] + iso_relaxation_gamma*(C01*(gu[1]*weak_mat_tmp27*weak_mat_tmp37 + weak_mat_tmp19*(s_t(2)*weak_mat_tmp6 + s_t(2)) + weak_mat_tmp30*weak_mat_tmp6) + C10*(weak_mat_tmp13*weak_mat_tmp6 + weak_mat_tmp52))) + trial_grad[3]*weak_mat_tmp56;
    const s_t material3 = trial_grad[0]*weak_mat_tmp51 + trial_grad[1]*weak_mat_tmp54 + trial_grad[2]*weak_mat_tmp56 + trial_grad[3]*(K*weak_mat_tmp4 + iso_history_qp[1] + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(s_t(2)*weak_mat_tmp4 + s_t(2)) - weak_mat_tmp27*weak_mat_tmp3*weak_mat_tmp49 + weak_mat_tmp30*weak_mat_tmp4) + C10*(weak_mat_tmp13*weak_mat_tmp4 + weak_mat_tmp18)));
      loperand0[0] = qw * (material0 * adj_value0 + material1 * adj_value1);
      loperand1[0] = qw * (material0 * adj_value2 + material1 * adj_value3);
      loperand2[0] = qw * (material2 * adj_value0 + material3 * adj_value1);
      loperand3[0] = qw * (material2 * adj_value2 + material3 * adj_value3);
    }
  }
  tensor_test_scalar<s_t, NQ, NS, 2, 2>(shape_1d, grad_1d, &loperand_q[0], out_streams, 0);
  tensor_test_scalar<s_t, NQ, NS, 2, 2>(shape_1d, grad_1d, &loperand_q[2 * NQ], out_streams, 1);
}

template <typename s_t, int NQ, int NS>
static __host__ __device__ __forceinline__ void mooney_rivlin_prony_d2_tensor_product_quadrature_output_block(
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t C01,
        const s_t C10,
        s_t *const RSTR iso_stress,
        const ptrdiff_t qp_stride,
        const s_t *const RSTR u_streams[NS * 2]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static constexpr int NQ1 = integer_root(NQ, 2);
  static constexpr int NS1 = integer_root(NS, 2);
  static_assert(ipow(NQ1, 2) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 2) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[4 * NQ];
  tensor_gradient_scalar<s_t, NQ, NS, 2, 2>(shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0]);
  tensor_gradient_scalar<s_t, NQ, NS, 2, 2>(shape_1d, grad_1d, u_streams, 1, &gu_ref_q[2 * NQ]);
  for (int q = 0; q < NQ; ++q) {
    const s_t *const RSTR gu_ref0 = &gu_ref_q[(2 * q)];
    const s_t *const RSTR gu_ref1 = &gu_ref_q[(2 * q + 1)];
    const s_t *const RSTR gu_ref2 = &gu_ref_q[(2 * (NQ + q))];
    const s_t *const RSTR gu_ref3 = &gu_ref_q[(2 * (NQ + q) + 1)];
    const s_t *const RSTR adj_q0 = adj0 + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adj1 + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adj2 + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adj3 + q * geometry_stride;
    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;
    s_t *const RSTR iso_stress_q = iso_stress + q * qp_stride;
    {
      const s_t adj_value0 = adj_q0[0];
      const s_t adj_value1 = adj_q1[0];
      const s_t adj_value2 = adj_q2[0];
      const s_t adj_value3 = adj_q3[0];
      const s_t det_value0 = det_q0[0];
      s_t gu[4];
      const s_t idet = s_t(1) / det_value0;
      gu[0] = (gu_ref0[0] * adj_value0 + gu_ref1[0] * adj_value2) * idet;
      gu[1] = (gu_ref0[0] * adj_value1 + gu_ref1[0] * adj_value3) * idet;
      gu[2] = (gu_ref2[0] * adj_value0 + gu_ref3[0] * adj_value2) * idet;
      gu[3] = (gu_ref2[0] * adj_value1 + gu_ref3[0] * adj_value3) * idet;
    const s_t qp_out_tmp0 = gu[0] + s_t(1);
    const s_t qp_out_tmp1 = gu[3] + s_t(1);
    const s_t qp_out_tmp2 = -gu[1]*gu[2] + qp_out_tmp0*qp_out_tmp1;
    const s_t qp_out_tmp3 = pow_m1(qp_out_tmp2);
    const s_t qp_out_tmp4 = pow(qp_out_tmp2, (s_t(-2) / s_t(3)));
    const s_t qp_out_tmp5 = s_t(2)*gu[2];
    const s_t qp_out_tmp6 = pow_2(gu[2]) + pow_2(qp_out_tmp0);
    const s_t qp_out_tmp7 = pow_2(gu[1]) + pow_2(qp_out_tmp1);
    const s_t qp_out_tmp8 = qp_out_tmp6 + qp_out_tmp7;
    const s_t qp_out_tmp9 = ((s_t(2) / s_t(3)))*(qp_out_tmp8 + s_t(1))/pow(qp_out_tmp2, (s_t(5) / s_t(3)));
    const s_t qp_out_tmp10 = pow(qp_out_tmp2, (s_t(-4) / s_t(3)));
    const s_t qp_out_tmp11 = gu[1]*qp_out_tmp0 + gu[2]*qp_out_tmp1;
    const s_t qp_out_tmp12 = s_t(2)*qp_out_tmp1;
    const s_t qp_out_tmp13 = ((s_t(4) / s_t(3)))*(-pow_2(qp_out_tmp11) - (s_t(1) / s_t(2))*pow_2(qp_out_tmp6) - (s_t(1) / s_t(2))*pow_2(qp_out_tmp7) + ((s_t(1) / s_t(2)))*pow_2(qp_out_tmp8) + qp_out_tmp8)/pow(qp_out_tmp2, (s_t(7) / s_t(3)));
    const s_t qp_out_tmp14 = s_t(2)*qp_out_tmp0;
    const s_t qp_out_tmp15 = s_t(2)*gu[1];
    const s_t qp_out_tmp16 = C01*(gu[2]*qp_out_tmp13 + qp_out_tmp10*(s_t(2)*gu[1]*qp_out_tmp8 + s_t(2)*gu[1] - qp_out_tmp11*qp_out_tmp14 - qp_out_tmp15*qp_out_tmp7)) + C10*(gu[2]*qp_out_tmp9 + qp_out_tmp15*qp_out_tmp4);
    const s_t qp_out_tmp17 = C01*(-qp_out_tmp0*qp_out_tmp13 + qp_out_tmp10*(s_t(2)*gu[3] - qp_out_tmp11*qp_out_tmp5 - qp_out_tmp12*qp_out_tmp7 + qp_out_tmp12*qp_out_tmp8 + s_t(2))) + C10*(-qp_out_tmp0*qp_out_tmp9 + s_t(2)*qp_out_tmp1*qp_out_tmp4);
    iso_stress_q[0] = qp_out_tmp3*(-gu[1]*(C01*(gu[1]*qp_out_tmp13 + qp_out_tmp10*(s_t(2)*gu[2]*qp_out_tmp8 + s_t(2)*gu[2] - qp_out_tmp11*qp_out_tmp12 - qp_out_tmp5*qp_out_tmp6)) + C10*(gu[1]*qp_out_tmp9 + qp_out_tmp4*qp_out_tmp5)) + qp_out_tmp1*(C01*(-qp_out_tmp1*qp_out_tmp13 + qp_out_tmp10*(s_t(2)*gu[0] - qp_out_tmp11*qp_out_tmp15 - qp_out_tmp14*qp_out_tmp6 + qp_out_tmp14*qp_out_tmp8 + s_t(2))) + C10*(-qp_out_tmp1*qp_out_tmp9 + qp_out_tmp14*qp_out_tmp4)));
    iso_stress_q[1 * NQ * qp_stride + 0] = qp_out_tmp3*(-gu[2]*qp_out_tmp16 + qp_out_tmp0*qp_out_tmp17);
    iso_stress_q[2 * NQ * qp_stride + 0] = qp_out_tmp3*(-gu[1]*qp_out_tmp17 + qp_out_tmp1*qp_out_tmp16);
    }
  }
}

} // namespace codegen
} // namespace sfem

#endif
