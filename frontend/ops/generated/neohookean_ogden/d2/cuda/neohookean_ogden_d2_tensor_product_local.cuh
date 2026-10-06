#ifndef NEOHOOKEAN_OGDEN_D2_TENSOR_PRODUCT_LOCAL_CUH
#define NEOHOOKEAN_OGDEN_D2_TENSOR_PRODUCT_LOCAL_CUH
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
static __host__ __device__ __forceinline__ void neohookean_ogden_d2_tensor_product_objective_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t lmbda,
        const s_t mu,
        const s_t *const RSTR u_streams[NS * 2],
        const s_t *const RSTR h_streams[NS * 2],
        const int nsteps,
        const s_t *const RSTR steps,
        const ptrdiff_t value_stride,
        s_t *const RSTR value
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NQ1 = integer_root(NQ, 2);
  static constexpr int NS1 = integer_root(NS, 2);
  static_assert(ipow(NQ1, 2) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 2) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[NQ * 4 * VS];
  s_t grad_h_ref_q[NQ * 4 * VS];
  tensor_gradient<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0]);
  tensor_gradient<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, h_streams, 0, &grad_h_ref_q[0]);
  tensor_gradient<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, u_streams, 1, &gu_ref_q[2 * NQ * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, h_streams, 1, &grad_h_ref_q[2 * NQ * VS]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = q / NQ1;
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
    const s_t *const RSTR gu_ref0 = &gu_ref_q[(2 * q) * VS];
    const s_t *const RSTR gu_ref1 = &gu_ref_q[(2 * q + 1) * VS];
    const s_t *const RSTR gu_ref2 = &gu_ref_q[(2 * (NQ + q)) * VS];
    const s_t *const RSTR gu_ref3 = &gu_ref_q[(2 * (NQ + q) + 1) * VS];
    const s_t *const RSTR grad_h_ref0 = &grad_h_ref_q[(2 * q) * VS];
    const s_t *const RSTR grad_h_ref1 = &grad_h_ref_q[(2 * q + 1) * VS];
    const s_t *const RSTR grad_h_ref2 = &grad_h_ref_q[(2 * (NQ + q)) * VS];
    const s_t *const RSTR grad_h_ref3 = &grad_h_ref_q[(2 * (NQ + q) + 1) * VS];
    const s_t *const RSTR adj_q0 = adj0 + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adj1 + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adj2 + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adj3 + q * geometry_stride;
    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;
    s_t gu_base_v[4 * VS];
    s_t trial_grad_v[4 * VS];
    {
      const s_t adj_value0 = adj_q0[0];
      const s_t adj_value1 = adj_q1[0];
      const s_t adj_value2 = adj_q2[0];
      const s_t adj_value3 = adj_q3[0];
      const s_t det_value0 = det_q0[0];
      const s_t idet = s_t(1) / det_value0;
      gu_base_v[0 * VS + 0] = (gu_ref0[0] * adj_value0 + gu_ref1[0] * adj_value2) * idet;
      trial_grad_v[0 * VS + 0] = (grad_h_ref0[0] * adj_value0 + grad_h_ref1[0] * adj_value2) * idet;
      gu_base_v[1 * VS + 0] = (gu_ref0[0] * adj_value1 + gu_ref1[0] * adj_value3) * idet;
      trial_grad_v[1 * VS + 0] = (grad_h_ref0[0] * adj_value1 + grad_h_ref1[0] * adj_value3) * idet;
      gu_base_v[2 * VS + 0] = (gu_ref2[0] * adj_value0 + gu_ref3[0] * adj_value2) * idet;
      trial_grad_v[2 * VS + 0] = (grad_h_ref2[0] * adj_value0 + grad_h_ref3[0] * adj_value2) * idet;
      gu_base_v[3 * VS + 0] = (gu_ref2[0] * adj_value1 + gu_ref3[0] * adj_value3) * idet;
      trial_grad_v[3 * VS + 0] = (grad_h_ref2[0] * adj_value1 + grad_h_ref3[0] * adj_value3) * idet;
    }
    for (int step = 0; step < nsteps; ++step) {
      const s_t alpha = steps[step];
      {
        const s_t det_value0 = det_q0[0];
        s_t gu[4];
        gu[0] = gu_base_v[0 * VS + 0] + alpha * trial_grad_v[0 * VS + 0];
        gu[1] = gu_base_v[1 * VS + 0] + alpha * trial_grad_v[1 * VS + 0];
        gu[2] = gu_base_v[2 * VS + 0] + alpha * trial_grad_v[2 * VS + 0];
        gu[3] = gu_base_v[3 * VS + 0] + alpha * trial_grad_v[3 * VS + 0];
    const s_t weak_obj_tmp0 = sfem_log1p(gu[0]*gu[3] + gu[0] - gu[1]*gu[2] + gu[3]);
    value[step * value_stride + 0] += qw * det_value0 * (((s_t(1) / s_t(2)))*lmbda*pow_2(weak_obj_tmp0) - mu*weak_obj_tmp0 + ((s_t(1) / s_t(2)))*mu*(pow_2(gu[0]) + s_t(2)*gu[0] + pow_2(gu[1]) + pow_2(gu[2]) + pow_2(gu[3]) + s_t(2)*gu[3]));
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void neohookean_ogden_d2_tensor_product_gradient_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t lmbda,
        const s_t mu,
        const s_t *const RSTR u_streams[NS * 2],
        s_t *const RSTR out_streams[NS * 2]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NQ1 = integer_root(NQ, 2);
  static constexpr int NS1 = integer_root(NS, 2);
  static_assert(ipow(NQ1, 2) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 2) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[NQ * 4 * VS];
  s_t loperand_q[NQ * 4 * VS];
  tensor_gradient<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0]);
  tensor_gradient<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, u_streams, 1, &gu_ref_q[2 * NQ * VS]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = q / NQ1;
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
    const s_t *const RSTR gu_ref0 = &gu_ref_q[(2 * q) * VS];
    const s_t *const RSTR gu_ref1 = &gu_ref_q[(2 * q + 1) * VS];
    const s_t *const RSTR gu_ref2 = &gu_ref_q[(2 * (NQ + q)) * VS];
    const s_t *const RSTR gu_ref3 = &gu_ref_q[(2 * (NQ + q) + 1) * VS];
    s_t *const RSTR loperand0 = &loperand_q[(2 * q) * VS];
    s_t *const RSTR loperand1 = &loperand_q[(2 * q + 1) * VS];
    s_t *const RSTR loperand2 = &loperand_q[(2 * (NQ + q)) * VS];
    s_t *const RSTR loperand3 = &loperand_q[(2 * (NQ + q) + 1) * VS];
    const s_t *const RSTR adj_q0 = adj0 + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adj1 + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adj2 + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adj3 + q * geometry_stride;
    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;
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
      s_t loperand[4];
    s_t material[4];
    const s_t weak_mat_tmp0 = gu[0] + s_t(1);
    const s_t weak_mat_tmp1 = mu*weak_mat_tmp0;
    const s_t weak_mat_tmp2 = gu[1]*gu[2];
    const s_t weak_mat_tmp3 = gu[3] + s_t(1);
    const s_t weak_mat_tmp4 = pow_m1(weak_mat_tmp0*weak_mat_tmp3 - weak_mat_tmp2);
    const s_t weak_mat_tmp5 = mu*weak_mat_tmp3;
    const s_t weak_mat_tmp6 = lmbda*weak_mat_tmp4*sfem_log1p(gu[0]*gu[3] + gu[0] + gu[3] - weak_mat_tmp2);
    const s_t weak_mat_tmp7 = gu[1]*mu;
    const s_t weak_mat_tmp8 = gu[2]*mu;
    material[0] = weak_mat_tmp1 + weak_mat_tmp3*weak_mat_tmp6 - weak_mat_tmp4*weak_mat_tmp5;
    material[1] = -gu[2]*weak_mat_tmp6 + weak_mat_tmp4*weak_mat_tmp8 + weak_mat_tmp7;
    material[2] = -gu[1]*weak_mat_tmp6 + weak_mat_tmp4*weak_mat_tmp7 + weak_mat_tmp8;
    material[3] = weak_mat_tmp0*weak_mat_tmp6 - weak_mat_tmp1*weak_mat_tmp4 + weak_mat_tmp5;
    loperand[0] = qw * (material[0] * adj_value0 + material[1] * adj_value1);
    loperand[1] = qw * (material[0] * adj_value2 + material[1] * adj_value3);
    loperand[2] = qw * (material[2] * adj_value0 + material[3] * adj_value1);
    loperand[3] = qw * (material[2] * adj_value2 + material[3] * adj_value3);
      loperand0[0] = loperand[0];
      loperand1[0] = loperand[1];
      loperand2[0] = loperand[2];
      loperand3[0] = loperand[3];
    }
  }
  tensor_test<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, &loperand_q[0], out_streams, 0);
  tensor_test<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, &loperand_q[2 * NQ * VS], out_streams, 1);
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void neohookean_ogden_d2_tensor_product_apply_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t lmbda,
        const s_t mu,
        const s_t *const RSTR u_streams[NS * 2],
        const s_t *const RSTR h_streams[NS * 2],
        s_t *const RSTR out_streams[NS * 2]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NQ1 = integer_root(NQ, 2);
  static constexpr int NS1 = integer_root(NS, 2);
  static_assert(ipow(NQ1, 2) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 2) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[NQ * 4 * VS];
  s_t grad_h_ref_q[NQ * 4 * VS];
  s_t loperand_q[NQ * 4 * VS];
  tensor_gradient<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0]);
  tensor_gradient<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, h_streams, 0, &grad_h_ref_q[0]);
  tensor_gradient<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, u_streams, 1, &gu_ref_q[2 * NQ * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, h_streams, 1, &grad_h_ref_q[2 * NQ * VS]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = q / NQ1;
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
    const s_t *const RSTR gu_ref0 = &gu_ref_q[(2 * q) * VS];
    const s_t *const RSTR gu_ref1 = &gu_ref_q[(2 * q + 1) * VS];
    const s_t *const RSTR gu_ref2 = &gu_ref_q[(2 * (NQ + q)) * VS];
    const s_t *const RSTR gu_ref3 = &gu_ref_q[(2 * (NQ + q) + 1) * VS];
    const s_t *const RSTR grad_h_ref0 = &grad_h_ref_q[(2 * q) * VS];
    const s_t *const RSTR grad_h_ref1 = &grad_h_ref_q[(2 * q + 1) * VS];
    const s_t *const RSTR grad_h_ref2 = &grad_h_ref_q[(2 * (NQ + q)) * VS];
    const s_t *const RSTR grad_h_ref3 = &grad_h_ref_q[(2 * (NQ + q) + 1) * VS];
    s_t *const RSTR loperand0 = &loperand_q[(2 * q) * VS];
    s_t *const RSTR loperand1 = &loperand_q[(2 * q + 1) * VS];
    s_t *const RSTR loperand2 = &loperand_q[(2 * (NQ + q)) * VS];
    s_t *const RSTR loperand3 = &loperand_q[(2 * (NQ + q) + 1) * VS];
    const s_t *const RSTR adj_q0 = adj0 + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adj1 + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adj2 + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adj3 + q * geometry_stride;
    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;
    {
      const s_t adj_value0 = adj_q0[0];
      const s_t adj_value1 = adj_q1[0];
      const s_t adj_value2 = adj_q2[0];
      const s_t adj_value3 = adj_q3[0];
      const s_t det_value0 = det_q0[0];
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
      s_t loperand[4];
    s_t material[4];
    const s_t weak_mat_tmp0 = gu[3] + s_t(1);
    const s_t weak_mat_tmp1 = gu[1]*gu[2];
    const s_t weak_mat_tmp2 = gu[0] + s_t(1);
    const s_t weak_mat_tmp3 = weak_mat_tmp0*weak_mat_tmp2 - weak_mat_tmp1;
    const s_t weak_mat_tmp4 = pow_m2(weak_mat_tmp3);
    const s_t weak_mat_tmp5 = weak_mat_tmp0*weak_mat_tmp4;
    const s_t weak_mat_tmp6 = gu[2]*weak_mat_tmp5;
    const s_t weak_mat_tmp7 = sfem_log1p(gu[0]*gu[3] + gu[0] + gu[3] - weak_mat_tmp1);
    const s_t weak_mat_tmp8 = gu[2]*lmbda*weak_mat_tmp0*weak_mat_tmp4*weak_mat_tmp7 - lmbda*weak_mat_tmp6 - mu*weak_mat_tmp6;
    const s_t weak_mat_tmp9 = gu[1]*weak_mat_tmp5;
    const s_t weak_mat_tmp10 = gu[1]*lmbda*weak_mat_tmp0*weak_mat_tmp4*weak_mat_tmp7 - lmbda*weak_mat_tmp9 - mu*weak_mat_tmp9;
    const s_t weak_mat_tmp11 = pow_2(weak_mat_tmp0)*weak_mat_tmp4;
    const s_t weak_mat_tmp12 = lmbda*weak_mat_tmp11;
    const s_t weak_mat_tmp13 = pow_m1(weak_mat_tmp3);
    const s_t weak_mat_tmp14 = mu*weak_mat_tmp13;
    const s_t weak_mat_tmp15 = weak_mat_tmp0*weak_mat_tmp2*weak_mat_tmp4;
    const s_t weak_mat_tmp16 = lmbda*weak_mat_tmp7;
    const s_t weak_mat_tmp17 = weak_mat_tmp13*weak_mat_tmp16;
    const s_t weak_mat_tmp18 = lmbda*weak_mat_tmp15 + mu*weak_mat_tmp15 - weak_mat_tmp14 - weak_mat_tmp15*weak_mat_tmp16 + weak_mat_tmp17;
    const s_t weak_mat_tmp19 = pow_2(gu[2])*weak_mat_tmp4;
    const s_t weak_mat_tmp20 = weak_mat_tmp2*weak_mat_tmp4;
    const s_t weak_mat_tmp21 = gu[2]*weak_mat_tmp20;
    const s_t weak_mat_tmp22 = gu[2]*lmbda*weak_mat_tmp2*weak_mat_tmp4*weak_mat_tmp7 - lmbda*weak_mat_tmp21 - mu*weak_mat_tmp21;
    const s_t weak_mat_tmp23 = weak_mat_tmp1*weak_mat_tmp4;
    const s_t weak_mat_tmp24 = lmbda*weak_mat_tmp23 + mu*weak_mat_tmp23 + weak_mat_tmp14 - weak_mat_tmp16*weak_mat_tmp23 - weak_mat_tmp17;
    const s_t weak_mat_tmp25 = pow_2(gu[1])*weak_mat_tmp4;
    const s_t weak_mat_tmp26 = gu[1]*weak_mat_tmp20;
    const s_t weak_mat_tmp27 = gu[1]*lmbda*weak_mat_tmp2*weak_mat_tmp4*weak_mat_tmp7 - lmbda*weak_mat_tmp26 - mu*weak_mat_tmp26;
    const s_t weak_mat_tmp28 = pow_2(weak_mat_tmp2)*weak_mat_tmp4;
    material[0] = trial_grad[0]*(mu*weak_mat_tmp11 + mu - weak_mat_tmp12*weak_mat_tmp7 + weak_mat_tmp12) + trial_grad[1]*weak_mat_tmp8 + trial_grad[2]*weak_mat_tmp10 + trial_grad[3]*weak_mat_tmp18;
    material[1] = trial_grad[0]*weak_mat_tmp8 + trial_grad[1]*(lmbda*weak_mat_tmp19 + mu*weak_mat_tmp19 + mu - weak_mat_tmp16*weak_mat_tmp19) + trial_grad[2]*weak_mat_tmp24 + trial_grad[3]*weak_mat_tmp22;
    material[2] = trial_grad[0]*weak_mat_tmp10 + trial_grad[1]*weak_mat_tmp24 + trial_grad[2]*(lmbda*weak_mat_tmp25 + mu*weak_mat_tmp25 + mu - weak_mat_tmp16*weak_mat_tmp25) + trial_grad[3]*weak_mat_tmp27;
    material[3] = trial_grad[0]*weak_mat_tmp18 + trial_grad[1]*weak_mat_tmp22 + trial_grad[2]*weak_mat_tmp27 + trial_grad[3]*(lmbda*weak_mat_tmp28 + mu*weak_mat_tmp28 + mu - weak_mat_tmp16*weak_mat_tmp28);
    loperand[0] = qw * (material[0] * adj_value0 + material[1] * adj_value1);
    loperand[1] = qw * (material[0] * adj_value2 + material[1] * adj_value3);
    loperand[2] = qw * (material[2] * adj_value0 + material[3] * adj_value1);
    loperand[3] = qw * (material[2] * adj_value2 + material[3] * adj_value3);
      loperand0[0] = loperand[0];
      loperand1[0] = loperand[1];
      loperand2[0] = loperand[2];
      loperand3[0] = loperand[3];
    }
  }
  tensor_test<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, &loperand_q[0], out_streams, 0);
  tensor_test<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, &loperand_q[2 * NQ * VS], out_streams, 1);
}

} // namespace codegen
} // namespace sfem

#endif
