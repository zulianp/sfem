#ifndef MODIFIED_MOONEY_RIVLIN_D2_TENSOR_PRODUCT_LOCAL_HPP
#define MODIFIED_MOONEY_RIVLIN_D2_TENSOR_PRODUCT_LOCAL_HPP
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
static SFEM_INLINE void modified_mooney_rivlin_d2_tensor_product_objective_block(
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
        const s_t c1,
        const s_t c2,
        const s_t kappa,
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
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t adj_lane0 = adj_q0[lane];
      const s_t adj_lane1 = adj_q1[lane];
      const s_t adj_lane2 = adj_q2[lane];
      const s_t adj_lane3 = adj_q3[lane];
      const s_t det_lane0 = det_q0[lane];
      const s_t idet = s_t(1) / det_lane0;
      gu_base_v[0 * VS + lane] = (gu_ref0[lane] * adj_lane0 + gu_ref1[lane] * adj_lane2) * idet;
      trial_grad_v[0 * VS + lane] = (grad_h_ref0[lane] * adj_lane0 + grad_h_ref1[lane] * adj_lane2) * idet;
      gu_base_v[1 * VS + lane] = (gu_ref0[lane] * adj_lane1 + gu_ref1[lane] * adj_lane3) * idet;
      trial_grad_v[1 * VS + lane] = (grad_h_ref0[lane] * adj_lane1 + grad_h_ref1[lane] * adj_lane3) * idet;
      gu_base_v[2 * VS + lane] = (gu_ref2[lane] * adj_lane0 + gu_ref3[lane] * adj_lane2) * idet;
      trial_grad_v[2 * VS + lane] = (grad_h_ref2[lane] * adj_lane0 + grad_h_ref3[lane] * adj_lane2) * idet;
      gu_base_v[3 * VS + lane] = (gu_ref2[lane] * adj_lane1 + gu_ref3[lane] * adj_lane3) * idet;
      trial_grad_v[3 * VS + lane] = (grad_h_ref2[lane] * adj_lane1 + grad_h_ref3[lane] * adj_lane3) * idet;
    }
    for (int step = 0; step < nsteps; ++step) {
      const s_t alpha = steps[step];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t det_lane0 = det_q0[lane];
        s_t gu[4];
        gu[0] = gu_base_v[0 * VS + lane] + alpha * trial_grad_v[0 * VS + lane];
        gu[1] = gu_base_v[1 * VS + lane] + alpha * trial_grad_v[1 * VS + lane];
        gu[2] = gu_base_v[2 * VS + lane] + alpha * trial_grad_v[2 * VS + lane];
        gu[3] = gu_base_v[3 * VS + lane] + alpha * trial_grad_v[3 * VS + lane];
    const s_t weak_obj_tmp0 = gu[1]*gu[2];
    const s_t weak_obj_tmp1 = gu[0] + s_t(1);
    const s_t weak_obj_tmp2 = gu[3] + s_t(1);
    const s_t weak_obj_tmp3 = -weak_obj_tmp0 + weak_obj_tmp1*weak_obj_tmp2;
    const s_t weak_obj_tmp4 = pow_2(gu[1]) + pow_2(weak_obj_tmp2);
    const s_t weak_obj_tmp5 = pow_2(gu[2]) + pow_2(weak_obj_tmp1);
    const s_t weak_obj_tmp6 = weak_obj_tmp4 + weak_obj_tmp5;
    value[step * value_stride + lane] += qw * det_lane0 * (c1*(s_t(-3) + (weak_obj_tmp6 + s_t(1))/pow(weak_obj_tmp3, (s_t(2) / s_t(3)))) + c2*(s_t(-3) + (-(s_t(1) / s_t(2))*pow_2(weak_obj_tmp4) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp5) + ((s_t(1) / s_t(2)))*pow_2(weak_obj_tmp6) + weak_obj_tmp6 - pow_2(gu[1]*weak_obj_tmp1 + gu[2]*weak_obj_tmp2))/pow(weak_obj_tmp3, (s_t(4) / s_t(3)))) + ((s_t(1) / s_t(2)))*kappa*pow_2(sfem_log1p(gu[0]*gu[3] + gu[0] + gu[3] - weak_obj_tmp0)));
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void modified_mooney_rivlin_d2_tensor_product_gradient_block(
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
        const s_t c1,
        const s_t c2,
        const s_t kappa,
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
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t adj_lane0 = adj_q0[lane];
      const s_t adj_lane1 = adj_q1[lane];
      const s_t adj_lane2 = adj_q2[lane];
      const s_t adj_lane3 = adj_q3[lane];
      const s_t det_lane0 = det_q0[lane];
      s_t gu[4];
      const s_t idet = s_t(1) / det_lane0;
      gu[0] = (gu_ref0[lane] * adj_lane0 + gu_ref1[lane] * adj_lane2) * idet;
      gu[1] = (gu_ref0[lane] * adj_lane1 + gu_ref1[lane] * adj_lane3) * idet;
      gu[2] = (gu_ref2[lane] * adj_lane0 + gu_ref3[lane] * adj_lane2) * idet;
      gu[3] = (gu_ref2[lane] * adj_lane1 + gu_ref3[lane] * adj_lane3) * idet;
      s_t loperand[4];
    s_t material[4];
    const s_t weak_mat_tmp0 = gu[3] + s_t(1);
    const s_t weak_mat_tmp1 = gu[1]*gu[2];
    const s_t weak_mat_tmp2 = gu[0] + s_t(1);
    const s_t weak_mat_tmp3 = weak_mat_tmp0*weak_mat_tmp2 - weak_mat_tmp1;
    const s_t weak_mat_tmp4 = kappa*sfem_log1p(gu[0]*gu[3] + gu[0] + gu[3] - weak_mat_tmp1)/weak_mat_tmp3;
    const s_t weak_mat_tmp5 = pow(weak_mat_tmp3, (s_t(-2) / s_t(3)));
    const s_t weak_mat_tmp6 = s_t(2)*weak_mat_tmp2;
    const s_t weak_mat_tmp7 = pow_2(gu[2]) + pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp8 = pow_2(gu[1]) + pow_2(weak_mat_tmp0);
    const s_t weak_mat_tmp9 = weak_mat_tmp7 + weak_mat_tmp8;
    const s_t weak_mat_tmp10 = ((s_t(2) / s_t(3)))*(weak_mat_tmp9 + s_t(1))/pow(weak_mat_tmp3, (s_t(5) / s_t(3)));
    const s_t weak_mat_tmp11 = pow(weak_mat_tmp3, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp12 = gu[1]*weak_mat_tmp2 + gu[2]*weak_mat_tmp0;
    const s_t weak_mat_tmp13 = s_t(2)*gu[1];
    const s_t weak_mat_tmp14 = ((s_t(4) / s_t(3)))*(-pow_2(weak_mat_tmp12) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp7) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp8) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp9) + weak_mat_tmp9)/pow(weak_mat_tmp3, (s_t(7) / s_t(3)));
    const s_t weak_mat_tmp15 = s_t(2)*gu[2];
    const s_t weak_mat_tmp16 = s_t(2)*weak_mat_tmp0;
    material[0] = c1*(-weak_mat_tmp0*weak_mat_tmp10 + weak_mat_tmp5*weak_mat_tmp6) + c2*(-weak_mat_tmp0*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu[0] - weak_mat_tmp12*weak_mat_tmp13 - weak_mat_tmp6*weak_mat_tmp7 + weak_mat_tmp6*weak_mat_tmp9 + s_t(2))) + weak_mat_tmp0*weak_mat_tmp4;
    material[1] = c1*(gu[2]*weak_mat_tmp10 + weak_mat_tmp13*weak_mat_tmp5) + c2*(gu[2]*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu[1]*weak_mat_tmp9 + s_t(2)*gu[1] - weak_mat_tmp12*weak_mat_tmp6 - weak_mat_tmp13*weak_mat_tmp8)) - gu[2]*weak_mat_tmp4;
    material[2] = c1*(gu[1]*weak_mat_tmp10 + weak_mat_tmp15*weak_mat_tmp5) + c2*(gu[1]*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu[2]*weak_mat_tmp9 + s_t(2)*gu[2] - weak_mat_tmp12*weak_mat_tmp16 - weak_mat_tmp15*weak_mat_tmp7)) - gu[1]*weak_mat_tmp4;
    material[3] = c1*(s_t(2)*weak_mat_tmp0*weak_mat_tmp5 - weak_mat_tmp10*weak_mat_tmp2) + c2*(weak_mat_tmp11*(s_t(2)*gu[3] - weak_mat_tmp12*weak_mat_tmp15 - weak_mat_tmp16*weak_mat_tmp8 + weak_mat_tmp16*weak_mat_tmp9 + s_t(2)) - weak_mat_tmp14*weak_mat_tmp2) + weak_mat_tmp2*weak_mat_tmp4;
    loperand[0] = qw * (material[0] * adj_lane0 + material[1] * adj_lane1);
    loperand[1] = qw * (material[0] * adj_lane2 + material[1] * adj_lane3);
    loperand[2] = qw * (material[2] * adj_lane0 + material[3] * adj_lane1);
    loperand[3] = qw * (material[2] * adj_lane2 + material[3] * adj_lane3);
      loperand0[lane] = loperand[0];
      loperand1[lane] = loperand[1];
      loperand2[lane] = loperand[2];
      loperand3[lane] = loperand[3];
    }
  }
  tensor_test<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, &loperand_q[0], out_streams, 0);
  tensor_test<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, &loperand_q[2 * NQ * VS], out_streams, 1);
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void modified_mooney_rivlin_d2_tensor_product_apply_block(
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
        const s_t c1,
        const s_t c2,
        const s_t kappa,
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
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t adj_lane0 = adj_q0[lane];
      const s_t adj_lane1 = adj_q1[lane];
      const s_t adj_lane2 = adj_q2[lane];
      const s_t adj_lane3 = adj_q3[lane];
      const s_t det_lane0 = det_q0[lane];
      s_t gu[4];
      s_t trial_grad[4];
      const s_t idet = s_t(1) / det_lane0;
      gu[0] = (gu_ref0[lane] * adj_lane0 + gu_ref1[lane] * adj_lane2) * idet;
      trial_grad[0] = (grad_h_ref0[lane] * adj_lane0 + grad_h_ref1[lane] * adj_lane2) * idet;
      gu[1] = (gu_ref0[lane] * adj_lane1 + gu_ref1[lane] * adj_lane3) * idet;
      trial_grad[1] = (grad_h_ref0[lane] * adj_lane1 + grad_h_ref1[lane] * adj_lane3) * idet;
      gu[2] = (gu_ref2[lane] * adj_lane0 + gu_ref3[lane] * adj_lane2) * idet;
      trial_grad[2] = (grad_h_ref2[lane] * adj_lane0 + grad_h_ref3[lane] * adj_lane2) * idet;
      gu[3] = (gu_ref2[lane] * adj_lane1 + gu_ref3[lane] * adj_lane3) * idet;
      trial_grad[3] = (grad_h_ref2[lane] * adj_lane1 + grad_h_ref3[lane] * adj_lane3) * idet;
      s_t loperand[4];
    s_t material[4];
    const s_t weak_mat_tmp0 = gu[3] + s_t(1);
    const s_t weak_mat_tmp1 = pow_2(weak_mat_tmp0);
    const s_t weak_mat_tmp2 = gu[1]*gu[2];
    const s_t weak_mat_tmp3 = gu[0] + s_t(1);
    const s_t weak_mat_tmp4 = weak_mat_tmp0*weak_mat_tmp3 - weak_mat_tmp2;
    const s_t weak_mat_tmp5 = kappa/pow_2(weak_mat_tmp4);
    const s_t weak_mat_tmp6 = weak_mat_tmp1*weak_mat_tmp5;
    const s_t weak_mat_tmp7 = sfem_log1p(gu[0]*gu[3] + gu[0] + gu[3] - weak_mat_tmp2);
    const s_t weak_mat_tmp8 = pow_2(gu[2]);
    const s_t weak_mat_tmp9 = pow_2(weak_mat_tmp3);
    const s_t weak_mat_tmp10 = weak_mat_tmp8 + weak_mat_tmp9;
    const s_t weak_mat_tmp11 = pow_2(gu[1]);
    const s_t weak_mat_tmp12 = weak_mat_tmp1 + weak_mat_tmp11;
    const s_t weak_mat_tmp13 = weak_mat_tmp10 + weak_mat_tmp12;
    const s_t weak_mat_tmp14 = weak_mat_tmp13 + s_t(1);
    const s_t weak_mat_tmp15 = pow(weak_mat_tmp4, (s_t(-8) / s_t(3)));
    const s_t weak_mat_tmp16 = ((s_t(10) / s_t(9)))*weak_mat_tmp14*weak_mat_tmp15;
    const s_t weak_mat_tmp17 = s_t(2)/pow(weak_mat_tmp4, (s_t(2) / s_t(3)));
    const s_t weak_mat_tmp18 = weak_mat_tmp0*weak_mat_tmp3;
    const s_t weak_mat_tmp19 = pow(weak_mat_tmp4, (s_t(-5) / s_t(3)));
    const s_t weak_mat_tmp20 = ((s_t(8) / s_t(3)))*weak_mat_tmp19;
    const s_t weak_mat_tmp21 = weak_mat_tmp17 - weak_mat_tmp18*weak_mat_tmp20;
    const s_t weak_mat_tmp22 = pow(weak_mat_tmp4, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp23 = gu[1]*weak_mat_tmp3;
    const s_t weak_mat_tmp24 = gu[2]*weak_mat_tmp0;
    const s_t weak_mat_tmp25 = weak_mat_tmp23 + weak_mat_tmp24;
    const s_t weak_mat_tmp26 = s_t(2)*gu[1];
    const s_t weak_mat_tmp27 = s_t(2)*weak_mat_tmp3;
    const s_t weak_mat_tmp28 = s_t(2)*gu[0] - weak_mat_tmp10*weak_mat_tmp27 + weak_mat_tmp13*weak_mat_tmp27 - weak_mat_tmp25*weak_mat_tmp26 + s_t(2);
    const s_t weak_mat_tmp29 = pow(weak_mat_tmp4, (s_t(-7) / s_t(3)));
    const s_t weak_mat_tmp30 = ((s_t(8) / s_t(3)))*weak_mat_tmp29;
    const s_t weak_mat_tmp31 = -(s_t(1) / s_t(2))*pow_2(weak_mat_tmp10) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp12) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp13) + weak_mat_tmp13 - pow_2(weak_mat_tmp25);
    const s_t weak_mat_tmp32 = pow(weak_mat_tmp4, (s_t(-10) / s_t(3)));
    const s_t weak_mat_tmp33 = ((s_t(28) / s_t(9)))*weak_mat_tmp31*weak_mat_tmp32;
    const s_t weak_mat_tmp34 = weak_mat_tmp24*weak_mat_tmp5;
    const s_t weak_mat_tmp35 = ((s_t(4) / s_t(3)))*weak_mat_tmp19;
    const s_t weak_mat_tmp36 = gu[1]*weak_mat_tmp0;
    const s_t weak_mat_tmp37 = weak_mat_tmp35*weak_mat_tmp36;
    const s_t weak_mat_tmp38 = gu[2]*weak_mat_tmp3;
    const s_t weak_mat_tmp39 = weak_mat_tmp35*weak_mat_tmp38;
    const s_t weak_mat_tmp40 = s_t(2)*weak_mat_tmp22;
    const s_t weak_mat_tmp41 = s_t(2)*gu[1]*weak_mat_tmp13 + s_t(2)*gu[1] - weak_mat_tmp12*weak_mat_tmp26 - weak_mat_tmp25*weak_mat_tmp27;
    const s_t weak_mat_tmp42 = ((s_t(4) / s_t(3)))*weak_mat_tmp29;
    const s_t weak_mat_tmp43 = weak_mat_tmp0*weak_mat_tmp42;
    const s_t weak_mat_tmp44 = c1*(-weak_mat_tmp16*weak_mat_tmp24 - weak_mat_tmp37 + weak_mat_tmp39) + c2*(((s_t(4) / s_t(3)))*gu[2]*weak_mat_tmp28*weak_mat_tmp29 - weak_mat_tmp24*weak_mat_tmp33 - weak_mat_tmp24*weak_mat_tmp40 - weak_mat_tmp41*weak_mat_tmp43) + weak_mat_tmp34*weak_mat_tmp7 - weak_mat_tmp34;
    const s_t weak_mat_tmp45 = weak_mat_tmp36*weak_mat_tmp5;
    const s_t weak_mat_tmp46 = weak_mat_tmp23*weak_mat_tmp35;
    const s_t weak_mat_tmp47 = weak_mat_tmp24*weak_mat_tmp35;
    const s_t weak_mat_tmp48 = s_t(2)*gu[2];
    const s_t weak_mat_tmp49 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp50 = s_t(2)*gu[2]*weak_mat_tmp13 + s_t(2)*gu[2] - weak_mat_tmp10*weak_mat_tmp48 - weak_mat_tmp25*weak_mat_tmp49;
    const s_t weak_mat_tmp51 = c1*(-weak_mat_tmp16*weak_mat_tmp36 + weak_mat_tmp46 - weak_mat_tmp47) + c2*(((s_t(4) / s_t(3)))*gu[1]*weak_mat_tmp28*weak_mat_tmp29 - weak_mat_tmp0*weak_mat_tmp22*weak_mat_tmp26 - weak_mat_tmp33*weak_mat_tmp36 - weak_mat_tmp43*weak_mat_tmp50) + weak_mat_tmp45*weak_mat_tmp7 - weak_mat_tmp45;
    const s_t weak_mat_tmp52 = weak_mat_tmp18*weak_mat_tmp5;
    const s_t weak_mat_tmp53 = kappa*weak_mat_tmp7/weak_mat_tmp4;
    const s_t weak_mat_tmp54 = ((s_t(2) / s_t(3)))*weak_mat_tmp14*weak_mat_tmp19;
    const s_t weak_mat_tmp55 = s_t(2)*gu[3] - weak_mat_tmp12*weak_mat_tmp49 + weak_mat_tmp13*weak_mat_tmp49 - weak_mat_tmp25*weak_mat_tmp48 + s_t(2);
    const s_t weak_mat_tmp56 = weak_mat_tmp31*weak_mat_tmp42;
    const s_t weak_mat_tmp57 = c1*(((s_t(10) / s_t(9)))*weak_mat_tmp0*weak_mat_tmp14*weak_mat_tmp15*weak_mat_tmp3 - weak_mat_tmp1*weak_mat_tmp35 - weak_mat_tmp35*weak_mat_tmp9 - weak_mat_tmp54) + c2*(((s_t(28) / s_t(9)))*weak_mat_tmp0*weak_mat_tmp3*weak_mat_tmp31*weak_mat_tmp32 + weak_mat_tmp22*(s_t(4)*weak_mat_tmp0*weak_mat_tmp3 - s_t(2)*weak_mat_tmp2) - weak_mat_tmp28*weak_mat_tmp3*weak_mat_tmp42 - weak_mat_tmp43*weak_mat_tmp55 - weak_mat_tmp56) - weak_mat_tmp52*weak_mat_tmp7 + weak_mat_tmp52 + weak_mat_tmp53;
    const s_t weak_mat_tmp58 = weak_mat_tmp5*weak_mat_tmp8;
    const s_t weak_mat_tmp59 = weak_mat_tmp17 + weak_mat_tmp2*weak_mat_tmp20;
    const s_t weak_mat_tmp60 = weak_mat_tmp38*weak_mat_tmp5;
    const s_t weak_mat_tmp61 = weak_mat_tmp41*weak_mat_tmp42;
    const s_t weak_mat_tmp62 = c1*(-weak_mat_tmp16*weak_mat_tmp38 - weak_mat_tmp46 + weak_mat_tmp47) + c2*(((s_t(4) / s_t(3)))*gu[2]*weak_mat_tmp29*weak_mat_tmp55 - weak_mat_tmp22*weak_mat_tmp3*weak_mat_tmp48 - weak_mat_tmp3*weak_mat_tmp61 - weak_mat_tmp33*weak_mat_tmp38) + weak_mat_tmp60*weak_mat_tmp7 - weak_mat_tmp60;
    const s_t weak_mat_tmp63 = weak_mat_tmp2*weak_mat_tmp5;
    const s_t weak_mat_tmp64 = c1*(weak_mat_tmp11*weak_mat_tmp35 + weak_mat_tmp16*weak_mat_tmp2 + weak_mat_tmp35*weak_mat_tmp8 + weak_mat_tmp54) + c2*(gu[1]*weak_mat_tmp61 + gu[2]*weak_mat_tmp42*weak_mat_tmp50 + weak_mat_tmp2*weak_mat_tmp33 + weak_mat_tmp22*(-s_t(2)*weak_mat_tmp18 + s_t(4)*weak_mat_tmp2) + weak_mat_tmp56) - weak_mat_tmp53 - weak_mat_tmp63*weak_mat_tmp7 + weak_mat_tmp63;
    const s_t weak_mat_tmp65 = weak_mat_tmp11*weak_mat_tmp5;
    const s_t weak_mat_tmp66 = weak_mat_tmp23*weak_mat_tmp5;
    const s_t weak_mat_tmp67 = c1*(-weak_mat_tmp16*weak_mat_tmp23 + weak_mat_tmp37 - weak_mat_tmp39) + c2*(((s_t(4) / s_t(3)))*gu[1]*weak_mat_tmp29*weak_mat_tmp55 - weak_mat_tmp23*weak_mat_tmp33 - weak_mat_tmp23*weak_mat_tmp40 - weak_mat_tmp3*weak_mat_tmp42*weak_mat_tmp50) + weak_mat_tmp66*weak_mat_tmp7 - weak_mat_tmp66;
    const s_t weak_mat_tmp68 = weak_mat_tmp5*weak_mat_tmp9;
    material[0] = trial_grad[0]*(c1*(weak_mat_tmp1*weak_mat_tmp16 + weak_mat_tmp21) + c2*(-weak_mat_tmp0*weak_mat_tmp28*weak_mat_tmp30 + weak_mat_tmp1*weak_mat_tmp33 + weak_mat_tmp22*(s_t(2)*weak_mat_tmp1 + s_t(2))) - weak_mat_tmp6*weak_mat_tmp7 + weak_mat_tmp6) + trial_grad[1]*weak_mat_tmp44 + trial_grad[2]*weak_mat_tmp51 + trial_grad[3]*weak_mat_tmp57;
    material[1] = trial_grad[0]*weak_mat_tmp44 + trial_grad[1]*(c1*(weak_mat_tmp16*weak_mat_tmp8 + weak_mat_tmp59) + c2*(gu[2]*weak_mat_tmp30*weak_mat_tmp41 + weak_mat_tmp22*(s_t(2)*weak_mat_tmp8 + s_t(2)) + weak_mat_tmp33*weak_mat_tmp8) - weak_mat_tmp58*weak_mat_tmp7 + weak_mat_tmp58) + trial_grad[2]*weak_mat_tmp64 + trial_grad[3]*weak_mat_tmp62;
    material[2] = trial_grad[0]*weak_mat_tmp51 + trial_grad[1]*weak_mat_tmp64 + trial_grad[2]*(c1*(weak_mat_tmp11*weak_mat_tmp16 + weak_mat_tmp59) + c2*(gu[1]*weak_mat_tmp30*weak_mat_tmp50 + weak_mat_tmp11*weak_mat_tmp33 + weak_mat_tmp22*(s_t(2)*weak_mat_tmp11 + s_t(2))) - weak_mat_tmp65*weak_mat_tmp7 + weak_mat_tmp65) + trial_grad[3]*weak_mat_tmp67;
    material[3] = trial_grad[0]*weak_mat_tmp57 + trial_grad[1]*weak_mat_tmp62 + trial_grad[2]*weak_mat_tmp67 + trial_grad[3]*(c1*(weak_mat_tmp16*weak_mat_tmp9 + weak_mat_tmp21) + c2*(weak_mat_tmp22*(s_t(2)*weak_mat_tmp9 + s_t(2)) - weak_mat_tmp3*weak_mat_tmp30*weak_mat_tmp55 + weak_mat_tmp33*weak_mat_tmp9) - weak_mat_tmp68*weak_mat_tmp7 + weak_mat_tmp68);
    loperand[0] = qw * (material[0] * adj_lane0 + material[1] * adj_lane1);
    loperand[1] = qw * (material[0] * adj_lane2 + material[1] * adj_lane3);
    loperand[2] = qw * (material[2] * adj_lane0 + material[3] * adj_lane1);
    loperand[3] = qw * (material[2] * adj_lane2 + material[3] * adj_lane3);
      loperand0[lane] = loperand[0];
      loperand1[lane] = loperand[1];
      loperand2[lane] = loperand[2];
      loperand3[lane] = loperand[3];
    }
  }
  tensor_test<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, &loperand_q[0], out_streams, 0);
  tensor_test<s_t, NQ, NS, VS, 2, 2>(ne, shape_1d, grad_1d, &loperand_q[2 * NQ * VS], out_streams, 1);
}

} // namespace codegen
} // namespace sfem

#endif
