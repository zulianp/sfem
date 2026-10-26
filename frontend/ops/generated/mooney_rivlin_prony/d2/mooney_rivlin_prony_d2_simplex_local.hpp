#ifndef MOONEY_RIVLIN_PRONY_D2_SIMPLEX_LOCAL_HPP
#define MOONEY_RIVLIN_PRONY_D2_SIMPLEX_LOCAL_HPP
#include <math.h>
#include <stddef.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_GENERATED_SCALAR_T
#endif
#endif
#include "kernel_math.hpp"
#include "tensor_product_kernels.hpp"
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
static SFEM_INLINE void mooney_rivlin_prony_d2_simplex_objective_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR grad_ref_x,
        const s_t *const RSTR grad_ref_y,
        const s_t *const RSTR q_weight,
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
  static_assert(VS > 0, "VS must be positive");
    for (int q = 0; q < NQ; ++q) {
      const s_t qw = q_weight[q];
      const s_t *const RSTR iso_history_q = iso_history + q * qp_stride;
      s_t gu_ref0_values[VS];
      s_t grad_h_ref0_values[VS];
      s_t gu_ref1_values[VS];
      s_t grad_h_ref1_values[VS];
      s_t gu_ref2_values[VS];
      s_t grad_h_ref2_values[VS];
      s_t gu_ref3_values[VS];
      s_t grad_h_ref3_values[VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        gu_ref0_values[lane] = s_t(0);
        grad_h_ref0_values[lane] = s_t(0);
        gu_ref1_values[lane] = s_t(0);
        grad_h_ref1_values[lane] = s_t(0);
        gu_ref2_values[lane] = s_t(0);
        grad_h_ref2_values[lane] = s_t(0);
        gu_ref3_values[lane] = s_t(0);
        grad_h_ref3_values[lane] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t gref0 = grad_ref_x[q * NS + shape];
        const s_t gref1 = grad_ref_y[q * NS + shape];
        const s_t *const RSTR u_shape0 = u_streams[2 * shape];
        const s_t *const RSTR u_shape1 = u_streams[2 * shape + 1];
        const s_t *const RSTR h_shape0 = h_streams[2 * shape];
        const s_t *const RSTR h_shape1 = h_streams[2 * shape + 1];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref0_values[lane] += u_shape0[lane] * gref0;
          grad_h_ref0_values[lane] += h_shape0[lane] * gref0;
          gu_ref1_values[lane] += u_shape0[lane] * gref1;
          grad_h_ref1_values[lane] += h_shape0[lane] * gref1;
          gu_ref2_values[lane] += u_shape1[lane] * gref0;
          grad_h_ref2_values[lane] += h_shape1[lane] * gref0;
          gu_ref3_values[lane] += u_shape1[lane] * gref1;
          grad_h_ref3_values[lane] += h_shape1[lane] * gref1;
        }
      }
      s_t gu_base_v[4 * VS];
      s_t trial_grad_v[4 * VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
      const s_t adj_lane0 = adj0[goff];
      const s_t adj_lane1 = adj1[goff];
      const s_t adj_lane2 = adj2[goff];
      const s_t adj_lane3 = adj3[goff];
      const s_t det_lane0 = det0[goff];
      const s_t gu_ref0 = gu_ref0_values[lane];
      const s_t grad_h_ref0 = grad_h_ref0_values[lane];
      const s_t gu_ref1 = gu_ref1_values[lane];
      const s_t grad_h_ref1 = grad_h_ref1_values[lane];
      const s_t gu_ref2 = gu_ref2_values[lane];
      const s_t grad_h_ref2 = grad_h_ref2_values[lane];
      const s_t gu_ref3 = gu_ref3_values[lane];
      const s_t grad_h_ref3 = grad_h_ref3_values[lane];
    const s_t idet = s_t(1) / det_lane0;
    gu_base_v[0 * VS + lane] = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane2) * idet;
    trial_grad_v[0 * VS + lane] = (grad_h_ref0 * adj_lane0 + grad_h_ref1 * adj_lane2) * idet;
    gu_base_v[1 * VS + lane] = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane3) * idet;
    trial_grad_v[1 * VS + lane] = (grad_h_ref0 * adj_lane1 + grad_h_ref1 * adj_lane3) * idet;
    gu_base_v[2 * VS + lane] = (gu_ref2 * adj_lane0 + gu_ref3 * adj_lane2) * idet;
    trial_grad_v[2 * VS + lane] = (grad_h_ref2 * adj_lane0 + grad_h_ref3 * adj_lane2) * idet;
    gu_base_v[3 * VS + lane] = (gu_ref2 * adj_lane1 + gu_ref3 * adj_lane3) * idet;
    trial_grad_v[3 * VS + lane] = (grad_h_ref2 * adj_lane1 + grad_h_ref3 * adj_lane3) * idet;
      }
      for (int step = 0; step < nsteps; ++step) {
        const s_t alpha = steps[step];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const ptrdiff_t goff = q * geometry_stride + lane;
          const s_t det_lane0 = det0[goff];
          const s_t iso_history_qp[3] = {iso_history_q[lane], iso_history_q[1 * NQ * qp_stride + lane], iso_history_q[2 * NQ * qp_stride + lane]};
          const s_t gu0 = gu_base_v[0 * VS + lane] + alpha * trial_grad_v[0 * VS + lane];
          const s_t gu1 = gu_base_v[1 * VS + lane] + alpha * trial_grad_v[1 * VS + lane];
          const s_t gu2 = gu_base_v[2 * VS + lane] + alpha * trial_grad_v[2 * VS + lane];
          const s_t gu3 = gu_base_v[3 * VS + lane] + alpha * trial_grad_v[3 * VS + lane];
    const s_t weak_obj_tmp0 = pow_2(gu2);
    const s_t weak_obj_tmp1 = pow_2(gu1);
    const s_t weak_obj_tmp2 = gu0 + s_t(1);
    const s_t weak_obj_tmp3 = gu3 + s_t(1);
    const s_t weak_obj_tmp4 = gu1*weak_obj_tmp2 + gu2*weak_obj_tmp3;
    const s_t weak_obj_tmp5 = gu1*gu2;
    const s_t weak_obj_tmp6 = weak_obj_tmp2*weak_obj_tmp3 - weak_obj_tmp5;
    const s_t weak_obj_tmp7 = weak_obj_tmp1 + pow_2(weak_obj_tmp3);
    const s_t weak_obj_tmp8 = weak_obj_tmp0 + pow_2(weak_obj_tmp2);
    const s_t weak_obj_tmp9 = weak_obj_tmp7 + weak_obj_tmp8;
    value[step * value_stride + lane] += qw * det_lane0 * (((s_t(1) / s_t(2)))*K*pow_2(gu0*gu3 + gu0 + gu3 - weak_obj_tmp5) + ((s_t(1) / s_t(2)))*iso_history_qp[0]*(pow_2(gu0) + s_t(2)*gu0 + weak_obj_tmp0) + ((s_t(1) / s_t(2)))*iso_history_qp[1]*(pow_2(gu3) + s_t(2)*gu3 + weak_obj_tmp1) + iso_history_qp[2]*weak_obj_tmp4 + iso_relaxation_gamma*(C01*(s_t(-3) + (-pow_2(weak_obj_tmp4) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp7) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp8) + ((s_t(1) / s_t(2)))*pow_2(weak_obj_tmp9) + weak_obj_tmp9)/pow(weak_obj_tmp6, (s_t(4) / s_t(3)))) + C10*(s_t(-3) + (weak_obj_tmp9 + s_t(1))/pow(weak_obj_tmp6, (s_t(2) / s_t(3))))));
        }
      }
    }
}

template <typename s_t, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_prony_d2_simplex_tri3_objective_block(
        const int ne,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
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
  static_assert(VS > 0, "VS must be positive");
    {
      const s_t qw = (s_t(1) / s_t(2));
      const s_t *const RSTR iso_history_q = iso_history;
      s_t gu_base_v[4 * VS];
      s_t trial_grad_v[4 * VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = lane;
      const s_t adj_lane0 = adj0[goff];
      const s_t adj_lane1 = adj1[goff];
      const s_t adj_lane2 = adj2[goff];
      const s_t adj_lane3 = adj3[goff];
      const s_t det_lane0 = det0[goff];
      const s_t gu_ref0 = -(u_streams[0][lane]) + u_streams[2][lane];
      const s_t grad_h_ref0 = -(h_streams[0][lane]) + h_streams[2][lane];
      const s_t gu_ref1 = -(u_streams[0][lane]) + u_streams[4][lane];
      const s_t grad_h_ref1 = -(h_streams[0][lane]) + h_streams[4][lane];
      const s_t gu_ref2 = -(u_streams[1][lane]) + u_streams[3][lane];
      const s_t grad_h_ref2 = -(h_streams[1][lane]) + h_streams[3][lane];
      const s_t gu_ref3 = -(u_streams[1][lane]) + u_streams[5][lane];
      const s_t grad_h_ref3 = -(h_streams[1][lane]) + h_streams[5][lane];
      const s_t idet = s_t(1) / det_lane0;
      gu_base_v[0 * VS + lane] = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane2) * idet;
      trial_grad_v[0 * VS + lane] = (grad_h_ref0 * adj_lane0 + grad_h_ref1 * adj_lane2) * idet;
      gu_base_v[1 * VS + lane] = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane3) * idet;
      trial_grad_v[1 * VS + lane] = (grad_h_ref0 * adj_lane1 + grad_h_ref1 * adj_lane3) * idet;
      gu_base_v[2 * VS + lane] = (gu_ref2 * adj_lane0 + gu_ref3 * adj_lane2) * idet;
      trial_grad_v[2 * VS + lane] = (grad_h_ref2 * adj_lane0 + grad_h_ref3 * adj_lane2) * idet;
      gu_base_v[3 * VS + lane] = (gu_ref2 * adj_lane1 + gu_ref3 * adj_lane3) * idet;
      trial_grad_v[3 * VS + lane] = (grad_h_ref2 * adj_lane1 + grad_h_ref3 * adj_lane3) * idet;
      }
      for (int step = 0; step < nsteps; ++step) {
        const s_t alpha = steps[step];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const ptrdiff_t goff = lane;
          const s_t det_lane0 = det0[goff];
          const s_t iso_history_qp[3] = {iso_history_q[lane], iso_history_q[1 * qp_stride + lane], iso_history_q[2 * qp_stride + lane]};
          const s_t gu0 = gu_base_v[0 * VS + lane] + alpha * trial_grad_v[0 * VS + lane];
          const s_t gu1 = gu_base_v[1 * VS + lane] + alpha * trial_grad_v[1 * VS + lane];
          const s_t gu2 = gu_base_v[2 * VS + lane] + alpha * trial_grad_v[2 * VS + lane];
          const s_t gu3 = gu_base_v[3 * VS + lane] + alpha * trial_grad_v[3 * VS + lane];
    const s_t weak_obj_tmp0 = pow_2(gu2);
    const s_t weak_obj_tmp1 = pow_2(gu1);
    const s_t weak_obj_tmp2 = gu0 + s_t(1);
    const s_t weak_obj_tmp3 = gu3 + s_t(1);
    const s_t weak_obj_tmp4 = gu1*weak_obj_tmp2 + gu2*weak_obj_tmp3;
    const s_t weak_obj_tmp5 = gu1*gu2;
    const s_t weak_obj_tmp6 = weak_obj_tmp2*weak_obj_tmp3 - weak_obj_tmp5;
    const s_t weak_obj_tmp7 = weak_obj_tmp1 + pow_2(weak_obj_tmp3);
    const s_t weak_obj_tmp8 = weak_obj_tmp0 + pow_2(weak_obj_tmp2);
    const s_t weak_obj_tmp9 = weak_obj_tmp7 + weak_obj_tmp8;
    value[step * value_stride + lane] += qw * det_lane0 * (((s_t(1) / s_t(2)))*K*pow_2(gu0*gu3 + gu0 + gu3 - weak_obj_tmp5) + ((s_t(1) / s_t(2)))*iso_history_qp[0]*(pow_2(gu0) + s_t(2)*gu0 + weak_obj_tmp0) + ((s_t(1) / s_t(2)))*iso_history_qp[1]*(pow_2(gu3) + s_t(2)*gu3 + weak_obj_tmp1) + iso_history_qp[2]*weak_obj_tmp4 + iso_relaxation_gamma*(C01*(s_t(-3) + (-pow_2(weak_obj_tmp4) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp7) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp8) + ((s_t(1) / s_t(2)))*pow_2(weak_obj_tmp9) + weak_obj_tmp9)/pow(weak_obj_tmp6, (s_t(4) / s_t(3)))) + C10*(s_t(-3) + (weak_obj_tmp9 + s_t(1))/pow(weak_obj_tmp6, (s_t(2) / s_t(3))))));
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_prony_d2_simplex_gradient_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR grad_ref_x,
        const s_t *const RSTR grad_ref_y,
        const s_t *const RSTR q_weight,
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
  static_assert(VS > 0, "VS must be positive");
    for (int q = 0; q < NQ; ++q) {
      const s_t qw = q_weight[q];
      const s_t *const RSTR iso_history_q = iso_history + q * qp_stride;
      s_t gu_ref0_values[VS];
      s_t gu_ref1_values[VS];
      s_t gu_ref2_values[VS];
      s_t gu_ref3_values[VS];
      s_t loperand0_values[VS];
      s_t loperand1_values[VS];
      s_t loperand2_values[VS];
      s_t loperand3_values[VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        gu_ref0_values[lane] = s_t(0);
        gu_ref1_values[lane] = s_t(0);
        gu_ref2_values[lane] = s_t(0);
        gu_ref3_values[lane] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t gref0 = grad_ref_x[q * NS + shape];
        const s_t gref1 = grad_ref_y[q * NS + shape];
        const s_t *const RSTR u_shape0 = u_streams[2 * shape];
        const s_t *const RSTR u_shape1 = u_streams[2 * shape + 1];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref0_values[lane] += u_shape0[lane] * gref0;
          gu_ref1_values[lane] += u_shape0[lane] * gref1;
          gu_ref2_values[lane] += u_shape1[lane] * gref0;
          gu_ref3_values[lane] += u_shape1[lane] * gref1;
        }
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
      const s_t adj_lane0 = adj0[goff];
      const s_t adj_lane1 = adj1[goff];
      const s_t adj_lane2 = adj2[goff];
      const s_t adj_lane3 = adj3[goff];
      const s_t det_lane0 = det0[goff];
      const s_t iso_history_qp[3] = {iso_history_q[lane], iso_history_q[1 * NQ * qp_stride + lane], iso_history_q[2 * NQ * qp_stride + lane]};
      const s_t gu_ref0 = gu_ref0_values[lane];
      const s_t gu_ref1 = gu_ref1_values[lane];
      const s_t gu_ref2 = gu_ref2_values[lane];
      const s_t gu_ref3 = gu_ref3_values[lane];
    const s_t idet = s_t(1) / det_lane0;
    const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane2) * idet;
    const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane3) * idet;
    const s_t gu2 = (gu_ref2 * adj_lane0 + gu_ref3 * adj_lane2) * idet;
    const s_t gu3 = (gu_ref2 * adj_lane1 + gu_ref3 * adj_lane3) * idet;
    const s_t weak_mat_tmp0 = gu0 + s_t(1);
    const s_t weak_mat_tmp1 = gu3 + s_t(1);
    const s_t weak_mat_tmp2 = gu1*gu2;
    const s_t weak_mat_tmp3 = K*(gu0*gu3 + gu0 + gu3 - weak_mat_tmp2);
    const s_t weak_mat_tmp4 = weak_mat_tmp0*weak_mat_tmp1 - weak_mat_tmp2;
    const s_t weak_mat_tmp5 = pow(weak_mat_tmp4, (s_t(-2) / s_t(3)));
    const s_t weak_mat_tmp6 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp7 = pow_2(gu2) + pow_2(weak_mat_tmp0);
    const s_t weak_mat_tmp8 = pow_2(gu1) + pow_2(weak_mat_tmp1);
    const s_t weak_mat_tmp9 = weak_mat_tmp7 + weak_mat_tmp8;
    const s_t weak_mat_tmp10 = ((s_t(2) / s_t(3)))*(weak_mat_tmp9 + s_t(1))/pow(weak_mat_tmp4, (s_t(5) / s_t(3)));
    const s_t weak_mat_tmp11 = pow(weak_mat_tmp4, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp12 = gu1*weak_mat_tmp0 + gu2*weak_mat_tmp1;
    const s_t weak_mat_tmp13 = s_t(2)*gu1;
    const s_t weak_mat_tmp14 = ((s_t(4) / s_t(3)))*(-pow_2(weak_mat_tmp12) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp7) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp8) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp9) + weak_mat_tmp9)/pow(weak_mat_tmp4, (s_t(7) / s_t(3)));
    const s_t weak_mat_tmp15 = s_t(2)*gu2;
    const s_t weak_mat_tmp16 = s_t(2)*weak_mat_tmp1;
    const s_t material0 = gu1*iso_history_qp[2] + iso_history_qp[0]*weak_mat_tmp0 + iso_relaxation_gamma*(C01*(-weak_mat_tmp1*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu0 - weak_mat_tmp12*weak_mat_tmp13 - weak_mat_tmp6*weak_mat_tmp7 + weak_mat_tmp6*weak_mat_tmp9 + s_t(2))) + C10*(-weak_mat_tmp1*weak_mat_tmp10 + weak_mat_tmp5*weak_mat_tmp6)) + weak_mat_tmp1*weak_mat_tmp3;
    const s_t material1 = gu1*iso_history_qp[1] - gu2*weak_mat_tmp3 + iso_history_qp[2]*weak_mat_tmp0 + iso_relaxation_gamma*(C01*(gu2*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu1*weak_mat_tmp9 + s_t(2)*gu1 - weak_mat_tmp12*weak_mat_tmp6 - weak_mat_tmp13*weak_mat_tmp8)) + C10*(gu2*weak_mat_tmp10 + weak_mat_tmp13*weak_mat_tmp5));
    const s_t material2 = -gu1*weak_mat_tmp3 + gu2*iso_history_qp[0] + iso_history_qp[2]*weak_mat_tmp1 + iso_relaxation_gamma*(C01*(gu1*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu2*weak_mat_tmp9 + s_t(2)*gu2 - weak_mat_tmp12*weak_mat_tmp16 - weak_mat_tmp15*weak_mat_tmp7)) + C10*(gu1*weak_mat_tmp10 + weak_mat_tmp15*weak_mat_tmp5));
    const s_t material3 = gu2*iso_history_qp[2] + iso_history_qp[1]*weak_mat_tmp1 + iso_relaxation_gamma*(C01*(-weak_mat_tmp0*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu3 - weak_mat_tmp12*weak_mat_tmp15 - weak_mat_tmp16*weak_mat_tmp8 + weak_mat_tmp16*weak_mat_tmp9 + s_t(2))) + C10*(-weak_mat_tmp0*weak_mat_tmp10 + s_t(2)*weak_mat_tmp1*weak_mat_tmp5)) + weak_mat_tmp0*weak_mat_tmp3;
      loperand0_values[lane] = qw * (material0 * adj_lane0 + material1 * adj_lane1);
      loperand1_values[lane] = qw * (material0 * adj_lane2 + material1 * adj_lane3);
      loperand2_values[lane] = qw * (material2 * adj_lane0 + material3 * adj_lane1);
      loperand3_values[lane] = qw * (material2 * adj_lane2 + material3 * adj_lane3);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t tref0 = grad_ref_x[q * NS + shape];
        const s_t tref1 = grad_ref_y[q * NS + shape];
        s_t *const RSTR out_shape0 = out_streams[2 * shape];
        s_t *const RSTR out_shape1 = out_streams[2 * shape + 1];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          out_shape0[lane] += loperand0_values[lane] * tref0 + loperand1_values[lane] * tref1;
          out_shape1[lane] += loperand2_values[lane] * tref0 + loperand3_values[lane] * tref1;
        }
      }
    }
}

template <typename s_t, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_prony_d2_simplex_tri3_gradient_block(
        const int ne,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const ptrdiff_t qp_stride,
        const s_t *const RSTR u_streams[NS * 2],
        s_t *const RSTR out_streams[NS * 2]
) {
  static_assert(VS > 0, "VS must be positive");
    {
      const s_t qw = (s_t(1) / s_t(2));
      const s_t *const RSTR iso_history_q = iso_history;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = lane;
      const s_t adj_lane0 = adj0[goff];
      const s_t adj_lane1 = adj1[goff];
      const s_t adj_lane2 = adj2[goff];
      const s_t adj_lane3 = adj3[goff];
      const s_t det_lane0 = det0[goff];
      const s_t iso_history_qp[3] = {iso_history_q[lane], iso_history_q[1 * qp_stride + lane], iso_history_q[2 * qp_stride + lane]};
      const s_t gu_ref0 = -(u_streams[0][lane]) + u_streams[2][lane];
      const s_t gu_ref1 = -(u_streams[0][lane]) + u_streams[4][lane];
      const s_t gu_ref2 = -(u_streams[1][lane]) + u_streams[3][lane];
      const s_t gu_ref3 = -(u_streams[1][lane]) + u_streams[5][lane];
      const s_t idet = s_t(1) / det_lane0;
      const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane2) * idet;
      const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane3) * idet;
      const s_t gu2 = (gu_ref2 * adj_lane0 + gu_ref3 * adj_lane2) * idet;
      const s_t gu3 = (gu_ref2 * adj_lane1 + gu_ref3 * adj_lane3) * idet;
    const s_t weak_mat_tmp0 = gu0 + s_t(1);
    const s_t weak_mat_tmp1 = gu3 + s_t(1);
    const s_t weak_mat_tmp2 = gu1*gu2;
    const s_t weak_mat_tmp3 = K*(gu0*gu3 + gu0 + gu3 - weak_mat_tmp2);
    const s_t weak_mat_tmp4 = weak_mat_tmp0*weak_mat_tmp1 - weak_mat_tmp2;
    const s_t weak_mat_tmp5 = pow(weak_mat_tmp4, (s_t(-2) / s_t(3)));
    const s_t weak_mat_tmp6 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp7 = pow_2(gu2) + pow_2(weak_mat_tmp0);
    const s_t weak_mat_tmp8 = pow_2(gu1) + pow_2(weak_mat_tmp1);
    const s_t weak_mat_tmp9 = weak_mat_tmp7 + weak_mat_tmp8;
    const s_t weak_mat_tmp10 = ((s_t(2) / s_t(3)))*(weak_mat_tmp9 + s_t(1))/pow(weak_mat_tmp4, (s_t(5) / s_t(3)));
    const s_t weak_mat_tmp11 = pow(weak_mat_tmp4, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp12 = gu1*weak_mat_tmp0 + gu2*weak_mat_tmp1;
    const s_t weak_mat_tmp13 = s_t(2)*gu1;
    const s_t weak_mat_tmp14 = ((s_t(4) / s_t(3)))*(-pow_2(weak_mat_tmp12) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp7) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp8) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp9) + weak_mat_tmp9)/pow(weak_mat_tmp4, (s_t(7) / s_t(3)));
    const s_t weak_mat_tmp15 = s_t(2)*gu2;
    const s_t weak_mat_tmp16 = s_t(2)*weak_mat_tmp1;
    const s_t material0 = gu1*iso_history_qp[2] + iso_history_qp[0]*weak_mat_tmp0 + iso_relaxation_gamma*(C01*(-weak_mat_tmp1*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu0 - weak_mat_tmp12*weak_mat_tmp13 - weak_mat_tmp6*weak_mat_tmp7 + weak_mat_tmp6*weak_mat_tmp9 + s_t(2))) + C10*(-weak_mat_tmp1*weak_mat_tmp10 + weak_mat_tmp5*weak_mat_tmp6)) + weak_mat_tmp1*weak_mat_tmp3;
    const s_t material1 = gu1*iso_history_qp[1] - gu2*weak_mat_tmp3 + iso_history_qp[2]*weak_mat_tmp0 + iso_relaxation_gamma*(C01*(gu2*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu1*weak_mat_tmp9 + s_t(2)*gu1 - weak_mat_tmp12*weak_mat_tmp6 - weak_mat_tmp13*weak_mat_tmp8)) + C10*(gu2*weak_mat_tmp10 + weak_mat_tmp13*weak_mat_tmp5));
    const s_t material2 = -gu1*weak_mat_tmp3 + gu2*iso_history_qp[0] + iso_history_qp[2]*weak_mat_tmp1 + iso_relaxation_gamma*(C01*(gu1*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu2*weak_mat_tmp9 + s_t(2)*gu2 - weak_mat_tmp12*weak_mat_tmp16 - weak_mat_tmp15*weak_mat_tmp7)) + C10*(gu1*weak_mat_tmp10 + weak_mat_tmp15*weak_mat_tmp5));
    const s_t material3 = gu2*iso_history_qp[2] + iso_history_qp[1]*weak_mat_tmp1 + iso_relaxation_gamma*(C01*(-weak_mat_tmp0*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu3 - weak_mat_tmp12*weak_mat_tmp15 - weak_mat_tmp16*weak_mat_tmp8 + weak_mat_tmp16*weak_mat_tmp9 + s_t(2))) + C10*(-weak_mat_tmp0*weak_mat_tmp10 + s_t(2)*weak_mat_tmp1*weak_mat_tmp5)) + weak_mat_tmp0*weak_mat_tmp3;
    const s_t loperand0 = qw * (material0 * adj_lane0 + material1 * adj_lane1);
    const s_t loperand1 = qw * (material0 * adj_lane2 + material1 * adj_lane3);
    const s_t loperand2 = qw * (material2 * adj_lane0 + material3 * adj_lane1);
    const s_t loperand3 = qw * (material2 * adj_lane2 + material3 * adj_lane3);
      out_streams[0][lane] += -(loperand0) - loperand1;
      out_streams[1][lane] += -(loperand2) - loperand3;
      out_streams[2][lane] += loperand0;
      out_streams[3][lane] += loperand2;
      out_streams[4][lane] += loperand1;
      out_streams[5][lane] += loperand3;
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_prony_d2_simplex_apply_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR grad_ref_x,
        const s_t *const RSTR grad_ref_y,
        const s_t *const RSTR q_weight,
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
  static_assert(VS > 0, "VS must be positive");
    for (int q = 0; q < NQ; ++q) {
      const s_t qw = q_weight[q];
      const s_t *const RSTR iso_history_q = iso_history + q * qp_stride;
      s_t gu_ref0_values[VS];
      s_t grad_h_ref0_values[VS];
      s_t gu_ref1_values[VS];
      s_t grad_h_ref1_values[VS];
      s_t gu_ref2_values[VS];
      s_t grad_h_ref2_values[VS];
      s_t gu_ref3_values[VS];
      s_t grad_h_ref3_values[VS];
      s_t loperand0_values[VS];
      s_t loperand1_values[VS];
      s_t loperand2_values[VS];
      s_t loperand3_values[VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        gu_ref0_values[lane] = s_t(0);
        grad_h_ref0_values[lane] = s_t(0);
        gu_ref1_values[lane] = s_t(0);
        grad_h_ref1_values[lane] = s_t(0);
        gu_ref2_values[lane] = s_t(0);
        grad_h_ref2_values[lane] = s_t(0);
        gu_ref3_values[lane] = s_t(0);
        grad_h_ref3_values[lane] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t gref0 = grad_ref_x[q * NS + shape];
        const s_t gref1 = grad_ref_y[q * NS + shape];
        const s_t *const RSTR u_shape0 = u_streams[2 * shape];
        const s_t *const RSTR u_shape1 = u_streams[2 * shape + 1];
        const s_t *const RSTR h_shape0 = h_streams[2 * shape];
        const s_t *const RSTR h_shape1 = h_streams[2 * shape + 1];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref0_values[lane] += u_shape0[lane] * gref0;
          grad_h_ref0_values[lane] += h_shape0[lane] * gref0;
          gu_ref1_values[lane] += u_shape0[lane] * gref1;
          grad_h_ref1_values[lane] += h_shape0[lane] * gref1;
          gu_ref2_values[lane] += u_shape1[lane] * gref0;
          grad_h_ref2_values[lane] += h_shape1[lane] * gref0;
          gu_ref3_values[lane] += u_shape1[lane] * gref1;
          grad_h_ref3_values[lane] += h_shape1[lane] * gref1;
        }
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
      const s_t adj_lane0 = adj0[goff];
      const s_t adj_lane1 = adj1[goff];
      const s_t adj_lane2 = adj2[goff];
      const s_t adj_lane3 = adj3[goff];
      const s_t det_lane0 = det0[goff];
      const s_t iso_history_qp[3] = {iso_history_q[lane], iso_history_q[1 * NQ * qp_stride + lane], iso_history_q[2 * NQ * qp_stride + lane]};
      const s_t gu_ref0 = gu_ref0_values[lane];
      const s_t grad_h_ref0 = grad_h_ref0_values[lane];
      const s_t gu_ref1 = gu_ref1_values[lane];
      const s_t grad_h_ref1 = grad_h_ref1_values[lane];
      const s_t gu_ref2 = gu_ref2_values[lane];
      const s_t grad_h_ref2 = grad_h_ref2_values[lane];
      const s_t gu_ref3 = gu_ref3_values[lane];
      const s_t grad_h_ref3 = grad_h_ref3_values[lane];
    const s_t idet = s_t(1) / det_lane0;
    const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane2) * idet;
    const s_t trial_grad0 = (grad_h_ref0 * adj_lane0 + grad_h_ref1 * adj_lane2) * idet;
    const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane3) * idet;
    const s_t trial_grad1 = (grad_h_ref0 * adj_lane1 + grad_h_ref1 * adj_lane3) * idet;
    const s_t gu2 = (gu_ref2 * adj_lane0 + gu_ref3 * adj_lane2) * idet;
    const s_t trial_grad2 = (grad_h_ref2 * adj_lane0 + grad_h_ref3 * adj_lane2) * idet;
    const s_t gu3 = (gu_ref2 * adj_lane1 + gu_ref3 * adj_lane3) * idet;
    const s_t trial_grad3 = (grad_h_ref2 * adj_lane1 + grad_h_ref3 * adj_lane3) * idet;
    const s_t weak_mat_tmp0 = gu3 + s_t(1);
    const s_t weak_mat_tmp1 = pow_2(weak_mat_tmp0);
    const s_t weak_mat_tmp2 = pow_2(gu2);
    const s_t weak_mat_tmp3 = gu0 + s_t(1);
    const s_t weak_mat_tmp4 = pow_2(weak_mat_tmp3);
    const s_t weak_mat_tmp5 = weak_mat_tmp2 + weak_mat_tmp4;
    const s_t weak_mat_tmp6 = pow_2(gu1);
    const s_t weak_mat_tmp7 = weak_mat_tmp1 + weak_mat_tmp6;
    const s_t weak_mat_tmp8 = weak_mat_tmp5 + weak_mat_tmp7;
    const s_t weak_mat_tmp9 = weak_mat_tmp8 + s_t(1);
    const s_t weak_mat_tmp10 = gu1*gu2;
    const s_t weak_mat_tmp11 = weak_mat_tmp0*weak_mat_tmp3 - weak_mat_tmp10;
    const s_t weak_mat_tmp12 = pow(weak_mat_tmp11, (s_t(-8) / s_t(3)));
    const s_t weak_mat_tmp13 = ((s_t(10) / s_t(9)))*weak_mat_tmp12*weak_mat_tmp9;
    const s_t weak_mat_tmp14 = s_t(2)/pow(weak_mat_tmp11, (s_t(2) / s_t(3)));
    const s_t weak_mat_tmp15 = weak_mat_tmp0*weak_mat_tmp3;
    const s_t weak_mat_tmp16 = pow(weak_mat_tmp11, (s_t(-5) / s_t(3)));
    const s_t weak_mat_tmp17 = ((s_t(8) / s_t(3)))*weak_mat_tmp16;
    const s_t weak_mat_tmp18 = weak_mat_tmp14 - weak_mat_tmp15*weak_mat_tmp17;
    const s_t weak_mat_tmp19 = pow(weak_mat_tmp11, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp20 = gu1*weak_mat_tmp3;
    const s_t weak_mat_tmp21 = gu2*weak_mat_tmp0;
    const s_t weak_mat_tmp22 = weak_mat_tmp20 + weak_mat_tmp21;
    const s_t weak_mat_tmp23 = s_t(2)*gu1;
    const s_t weak_mat_tmp24 = s_t(2)*weak_mat_tmp3;
    const s_t weak_mat_tmp25 = s_t(2)*gu0 - weak_mat_tmp22*weak_mat_tmp23 - weak_mat_tmp24*weak_mat_tmp5 + weak_mat_tmp24*weak_mat_tmp8 + s_t(2);
    const s_t weak_mat_tmp26 = pow(weak_mat_tmp11, (s_t(-7) / s_t(3)));
    const s_t weak_mat_tmp27 = ((s_t(8) / s_t(3)))*weak_mat_tmp26;
    const s_t weak_mat_tmp28 = -pow_2(weak_mat_tmp22) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp5) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp7) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp8) + weak_mat_tmp8;
    const s_t weak_mat_tmp29 = pow(weak_mat_tmp11, (s_t(-10) / s_t(3)));
    const s_t weak_mat_tmp30 = ((s_t(28) / s_t(9)))*weak_mat_tmp28*weak_mat_tmp29;
    const s_t weak_mat_tmp31 = gu1*weak_mat_tmp0;
    const s_t weak_mat_tmp32 = ((s_t(4) / s_t(3)))*weak_mat_tmp16;
    const s_t weak_mat_tmp33 = weak_mat_tmp20*weak_mat_tmp32;
    const s_t weak_mat_tmp34 = weak_mat_tmp21*weak_mat_tmp32;
    const s_t weak_mat_tmp35 = s_t(2)*gu2;
    const s_t weak_mat_tmp36 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp37 = s_t(2)*gu2*weak_mat_tmp8 + s_t(2)*gu2 - weak_mat_tmp22*weak_mat_tmp36 - weak_mat_tmp35*weak_mat_tmp5;
    const s_t weak_mat_tmp38 = ((s_t(4) / s_t(3)))*weak_mat_tmp26;
    const s_t weak_mat_tmp39 = weak_mat_tmp0*weak_mat_tmp38;
    const s_t weak_mat_tmp40 = -K*weak_mat_tmp31 + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu1*weak_mat_tmp25*weak_mat_tmp26 - weak_mat_tmp0*weak_mat_tmp19*weak_mat_tmp23 - weak_mat_tmp30*weak_mat_tmp31 - weak_mat_tmp37*weak_mat_tmp39) + C10*(-weak_mat_tmp13*weak_mat_tmp31 + weak_mat_tmp33 - weak_mat_tmp34));
    const s_t weak_mat_tmp41 = weak_mat_tmp31*weak_mat_tmp32;
    const s_t weak_mat_tmp42 = gu2*weak_mat_tmp3;
    const s_t weak_mat_tmp43 = weak_mat_tmp32*weak_mat_tmp42;
    const s_t weak_mat_tmp44 = s_t(2)*weak_mat_tmp19;
    const s_t weak_mat_tmp45 = s_t(2)*gu1*weak_mat_tmp8 + s_t(2)*gu1 - weak_mat_tmp22*weak_mat_tmp24 - weak_mat_tmp23*weak_mat_tmp7;
    const s_t weak_mat_tmp46 = -K*weak_mat_tmp21 + iso_history_qp[2] + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu2*weak_mat_tmp25*weak_mat_tmp26 - weak_mat_tmp21*weak_mat_tmp30 - weak_mat_tmp21*weak_mat_tmp44 - weak_mat_tmp39*weak_mat_tmp45) + C10*(-weak_mat_tmp13*weak_mat_tmp21 - weak_mat_tmp41 + weak_mat_tmp43));
    const s_t weak_mat_tmp47 = K*(gu0*gu3 + gu0 + gu3 - weak_mat_tmp10);
    const s_t weak_mat_tmp48 = ((s_t(2) / s_t(3)))*weak_mat_tmp16*weak_mat_tmp9;
    const s_t weak_mat_tmp49 = s_t(2)*gu3 - weak_mat_tmp22*weak_mat_tmp35 - weak_mat_tmp36*weak_mat_tmp7 + weak_mat_tmp36*weak_mat_tmp8 + s_t(2);
    const s_t weak_mat_tmp50 = weak_mat_tmp28*weak_mat_tmp38;
    const s_t weak_mat_tmp51 = K*weak_mat_tmp15 + iso_relaxation_gamma*(C01*(((s_t(28) / s_t(9)))*weak_mat_tmp0*weak_mat_tmp28*weak_mat_tmp29*weak_mat_tmp3 + weak_mat_tmp19*(s_t(4)*weak_mat_tmp0*weak_mat_tmp3 - s_t(2)*weak_mat_tmp10) - weak_mat_tmp25*weak_mat_tmp3*weak_mat_tmp38 - weak_mat_tmp39*weak_mat_tmp49 - weak_mat_tmp50) + C10*(((s_t(10) / s_t(9)))*weak_mat_tmp0*weak_mat_tmp12*weak_mat_tmp3*weak_mat_tmp9 - weak_mat_tmp1*weak_mat_tmp32 - weak_mat_tmp32*weak_mat_tmp4 - weak_mat_tmp48)) + weak_mat_tmp47;
    const s_t weak_mat_tmp52 = weak_mat_tmp10*weak_mat_tmp17 + weak_mat_tmp14;
    const s_t weak_mat_tmp53 = weak_mat_tmp38*weak_mat_tmp45;
    const s_t weak_mat_tmp54 = -K*weak_mat_tmp42 + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu2*weak_mat_tmp26*weak_mat_tmp49 - weak_mat_tmp19*weak_mat_tmp3*weak_mat_tmp35 - weak_mat_tmp3*weak_mat_tmp53 - weak_mat_tmp30*weak_mat_tmp42) + C10*(-weak_mat_tmp13*weak_mat_tmp42 - weak_mat_tmp33 + weak_mat_tmp34));
    const s_t weak_mat_tmp55 = K*weak_mat_tmp10 + iso_relaxation_gamma*(C01*(gu1*weak_mat_tmp53 + gu2*weak_mat_tmp37*weak_mat_tmp38 + weak_mat_tmp10*weak_mat_tmp30 + weak_mat_tmp19*(s_t(4)*weak_mat_tmp10 - s_t(2)*weak_mat_tmp15) + weak_mat_tmp50) + C10*(weak_mat_tmp10*weak_mat_tmp13 + weak_mat_tmp2*weak_mat_tmp32 + weak_mat_tmp32*weak_mat_tmp6 + weak_mat_tmp48)) - weak_mat_tmp47;
    const s_t weak_mat_tmp56 = -K*weak_mat_tmp20 + iso_history_qp[2] + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu1*weak_mat_tmp26*weak_mat_tmp49 - weak_mat_tmp20*weak_mat_tmp30 - weak_mat_tmp20*weak_mat_tmp44 - weak_mat_tmp3*weak_mat_tmp37*weak_mat_tmp38) + C10*(-weak_mat_tmp13*weak_mat_tmp20 + weak_mat_tmp41 - weak_mat_tmp43));
    const s_t material0 = trial_grad0*(K*weak_mat_tmp1 + iso_history_qp[0] + iso_relaxation_gamma*(C01*(-weak_mat_tmp0*weak_mat_tmp25*weak_mat_tmp27 + weak_mat_tmp1*weak_mat_tmp30 + weak_mat_tmp19*(s_t(2)*weak_mat_tmp1 + s_t(2))) + C10*(weak_mat_tmp1*weak_mat_tmp13 + weak_mat_tmp18))) + trial_grad1*weak_mat_tmp46 + trial_grad2*weak_mat_tmp40 + trial_grad3*weak_mat_tmp51;
    const s_t material1 = trial_grad0*weak_mat_tmp46 + trial_grad1*(K*weak_mat_tmp2 + iso_history_qp[1] + iso_relaxation_gamma*(C01*(gu2*weak_mat_tmp27*weak_mat_tmp45 + weak_mat_tmp19*(s_t(2)*weak_mat_tmp2 + s_t(2)) + weak_mat_tmp2*weak_mat_tmp30) + C10*(weak_mat_tmp13*weak_mat_tmp2 + weak_mat_tmp52))) + trial_grad2*weak_mat_tmp55 + trial_grad3*weak_mat_tmp54;
    const s_t material2 = trial_grad0*weak_mat_tmp40 + trial_grad1*weak_mat_tmp55 + trial_grad2*(K*weak_mat_tmp6 + iso_history_qp[0] + iso_relaxation_gamma*(C01*(gu1*weak_mat_tmp27*weak_mat_tmp37 + weak_mat_tmp19*(s_t(2)*weak_mat_tmp6 + s_t(2)) + weak_mat_tmp30*weak_mat_tmp6) + C10*(weak_mat_tmp13*weak_mat_tmp6 + weak_mat_tmp52))) + trial_grad3*weak_mat_tmp56;
    const s_t material3 = trial_grad0*weak_mat_tmp51 + trial_grad1*weak_mat_tmp54 + trial_grad2*weak_mat_tmp56 + trial_grad3*(K*weak_mat_tmp4 + iso_history_qp[1] + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(s_t(2)*weak_mat_tmp4 + s_t(2)) - weak_mat_tmp27*weak_mat_tmp3*weak_mat_tmp49 + weak_mat_tmp30*weak_mat_tmp4) + C10*(weak_mat_tmp13*weak_mat_tmp4 + weak_mat_tmp18)));
      loperand0_values[lane] = qw * (material0 * adj_lane0 + material1 * adj_lane1);
      loperand1_values[lane] = qw * (material0 * adj_lane2 + material1 * adj_lane3);
      loperand2_values[lane] = qw * (material2 * adj_lane0 + material3 * adj_lane1);
      loperand3_values[lane] = qw * (material2 * adj_lane2 + material3 * adj_lane3);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t tref0 = grad_ref_x[q * NS + shape];
        const s_t tref1 = grad_ref_y[q * NS + shape];
        s_t *const RSTR out_shape0 = out_streams[2 * shape];
        s_t *const RSTR out_shape1 = out_streams[2 * shape + 1];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          out_shape0[lane] += loperand0_values[lane] * tref0 + loperand1_values[lane] * tref1;
          out_shape1[lane] += loperand2_values[lane] * tref0 + loperand3_values[lane] * tref1;
        }
      }
    }
}

template <typename s_t, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_prony_d2_simplex_tri3_apply_block(
        const int ne,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
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
  static_assert(VS > 0, "VS must be positive");
    {
      const s_t qw = (s_t(1) / s_t(2));
      const s_t *const RSTR iso_history_q = iso_history;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = lane;
      const s_t adj_lane0 = adj0[goff];
      const s_t adj_lane1 = adj1[goff];
      const s_t adj_lane2 = adj2[goff];
      const s_t adj_lane3 = adj3[goff];
      const s_t det_lane0 = det0[goff];
      const s_t iso_history_qp[3] = {iso_history_q[lane], iso_history_q[1 * qp_stride + lane], iso_history_q[2 * qp_stride + lane]};
      const s_t gu_ref0 = -(u_streams[0][lane]) + u_streams[2][lane];
      const s_t grad_h_ref0 = -(h_streams[0][lane]) + h_streams[2][lane];
      const s_t gu_ref1 = -(u_streams[0][lane]) + u_streams[4][lane];
      const s_t grad_h_ref1 = -(h_streams[0][lane]) + h_streams[4][lane];
      const s_t gu_ref2 = -(u_streams[1][lane]) + u_streams[3][lane];
      const s_t grad_h_ref2 = -(h_streams[1][lane]) + h_streams[3][lane];
      const s_t gu_ref3 = -(u_streams[1][lane]) + u_streams[5][lane];
      const s_t grad_h_ref3 = -(h_streams[1][lane]) + h_streams[5][lane];
      const s_t idet = s_t(1) / det_lane0;
      const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane2) * idet;
      const s_t trial_grad0 = (grad_h_ref0 * adj_lane0 + grad_h_ref1 * adj_lane2) * idet;
      const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane3) * idet;
      const s_t trial_grad1 = (grad_h_ref0 * adj_lane1 + grad_h_ref1 * adj_lane3) * idet;
      const s_t gu2 = (gu_ref2 * adj_lane0 + gu_ref3 * adj_lane2) * idet;
      const s_t trial_grad2 = (grad_h_ref2 * adj_lane0 + grad_h_ref3 * adj_lane2) * idet;
      const s_t gu3 = (gu_ref2 * adj_lane1 + gu_ref3 * adj_lane3) * idet;
      const s_t trial_grad3 = (grad_h_ref2 * adj_lane1 + grad_h_ref3 * adj_lane3) * idet;
    const s_t weak_mat_tmp0 = gu3 + s_t(1);
    const s_t weak_mat_tmp1 = pow_2(weak_mat_tmp0);
    const s_t weak_mat_tmp2 = pow_2(gu2);
    const s_t weak_mat_tmp3 = gu0 + s_t(1);
    const s_t weak_mat_tmp4 = pow_2(weak_mat_tmp3);
    const s_t weak_mat_tmp5 = weak_mat_tmp2 + weak_mat_tmp4;
    const s_t weak_mat_tmp6 = pow_2(gu1);
    const s_t weak_mat_tmp7 = weak_mat_tmp1 + weak_mat_tmp6;
    const s_t weak_mat_tmp8 = weak_mat_tmp5 + weak_mat_tmp7;
    const s_t weak_mat_tmp9 = weak_mat_tmp8 + s_t(1);
    const s_t weak_mat_tmp10 = gu1*gu2;
    const s_t weak_mat_tmp11 = weak_mat_tmp0*weak_mat_tmp3 - weak_mat_tmp10;
    const s_t weak_mat_tmp12 = pow(weak_mat_tmp11, (s_t(-8) / s_t(3)));
    const s_t weak_mat_tmp13 = ((s_t(10) / s_t(9)))*weak_mat_tmp12*weak_mat_tmp9;
    const s_t weak_mat_tmp14 = s_t(2)/pow(weak_mat_tmp11, (s_t(2) / s_t(3)));
    const s_t weak_mat_tmp15 = weak_mat_tmp0*weak_mat_tmp3;
    const s_t weak_mat_tmp16 = pow(weak_mat_tmp11, (s_t(-5) / s_t(3)));
    const s_t weak_mat_tmp17 = ((s_t(8) / s_t(3)))*weak_mat_tmp16;
    const s_t weak_mat_tmp18 = weak_mat_tmp14 - weak_mat_tmp15*weak_mat_tmp17;
    const s_t weak_mat_tmp19 = pow(weak_mat_tmp11, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp20 = gu1*weak_mat_tmp3;
    const s_t weak_mat_tmp21 = gu2*weak_mat_tmp0;
    const s_t weak_mat_tmp22 = weak_mat_tmp20 + weak_mat_tmp21;
    const s_t weak_mat_tmp23 = s_t(2)*gu1;
    const s_t weak_mat_tmp24 = s_t(2)*weak_mat_tmp3;
    const s_t weak_mat_tmp25 = s_t(2)*gu0 - weak_mat_tmp22*weak_mat_tmp23 - weak_mat_tmp24*weak_mat_tmp5 + weak_mat_tmp24*weak_mat_tmp8 + s_t(2);
    const s_t weak_mat_tmp26 = pow(weak_mat_tmp11, (s_t(-7) / s_t(3)));
    const s_t weak_mat_tmp27 = ((s_t(8) / s_t(3)))*weak_mat_tmp26;
    const s_t weak_mat_tmp28 = -pow_2(weak_mat_tmp22) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp5) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp7) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp8) + weak_mat_tmp8;
    const s_t weak_mat_tmp29 = pow(weak_mat_tmp11, (s_t(-10) / s_t(3)));
    const s_t weak_mat_tmp30 = ((s_t(28) / s_t(9)))*weak_mat_tmp28*weak_mat_tmp29;
    const s_t weak_mat_tmp31 = gu1*weak_mat_tmp0;
    const s_t weak_mat_tmp32 = ((s_t(4) / s_t(3)))*weak_mat_tmp16;
    const s_t weak_mat_tmp33 = weak_mat_tmp20*weak_mat_tmp32;
    const s_t weak_mat_tmp34 = weak_mat_tmp21*weak_mat_tmp32;
    const s_t weak_mat_tmp35 = s_t(2)*gu2;
    const s_t weak_mat_tmp36 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp37 = s_t(2)*gu2*weak_mat_tmp8 + s_t(2)*gu2 - weak_mat_tmp22*weak_mat_tmp36 - weak_mat_tmp35*weak_mat_tmp5;
    const s_t weak_mat_tmp38 = ((s_t(4) / s_t(3)))*weak_mat_tmp26;
    const s_t weak_mat_tmp39 = weak_mat_tmp0*weak_mat_tmp38;
    const s_t weak_mat_tmp40 = -K*weak_mat_tmp31 + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu1*weak_mat_tmp25*weak_mat_tmp26 - weak_mat_tmp0*weak_mat_tmp19*weak_mat_tmp23 - weak_mat_tmp30*weak_mat_tmp31 - weak_mat_tmp37*weak_mat_tmp39) + C10*(-weak_mat_tmp13*weak_mat_tmp31 + weak_mat_tmp33 - weak_mat_tmp34));
    const s_t weak_mat_tmp41 = weak_mat_tmp31*weak_mat_tmp32;
    const s_t weak_mat_tmp42 = gu2*weak_mat_tmp3;
    const s_t weak_mat_tmp43 = weak_mat_tmp32*weak_mat_tmp42;
    const s_t weak_mat_tmp44 = s_t(2)*weak_mat_tmp19;
    const s_t weak_mat_tmp45 = s_t(2)*gu1*weak_mat_tmp8 + s_t(2)*gu1 - weak_mat_tmp22*weak_mat_tmp24 - weak_mat_tmp23*weak_mat_tmp7;
    const s_t weak_mat_tmp46 = -K*weak_mat_tmp21 + iso_history_qp[2] + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu2*weak_mat_tmp25*weak_mat_tmp26 - weak_mat_tmp21*weak_mat_tmp30 - weak_mat_tmp21*weak_mat_tmp44 - weak_mat_tmp39*weak_mat_tmp45) + C10*(-weak_mat_tmp13*weak_mat_tmp21 - weak_mat_tmp41 + weak_mat_tmp43));
    const s_t weak_mat_tmp47 = K*(gu0*gu3 + gu0 + gu3 - weak_mat_tmp10);
    const s_t weak_mat_tmp48 = ((s_t(2) / s_t(3)))*weak_mat_tmp16*weak_mat_tmp9;
    const s_t weak_mat_tmp49 = s_t(2)*gu3 - weak_mat_tmp22*weak_mat_tmp35 - weak_mat_tmp36*weak_mat_tmp7 + weak_mat_tmp36*weak_mat_tmp8 + s_t(2);
    const s_t weak_mat_tmp50 = weak_mat_tmp28*weak_mat_tmp38;
    const s_t weak_mat_tmp51 = K*weak_mat_tmp15 + iso_relaxation_gamma*(C01*(((s_t(28) / s_t(9)))*weak_mat_tmp0*weak_mat_tmp28*weak_mat_tmp29*weak_mat_tmp3 + weak_mat_tmp19*(s_t(4)*weak_mat_tmp0*weak_mat_tmp3 - s_t(2)*weak_mat_tmp10) - weak_mat_tmp25*weak_mat_tmp3*weak_mat_tmp38 - weak_mat_tmp39*weak_mat_tmp49 - weak_mat_tmp50) + C10*(((s_t(10) / s_t(9)))*weak_mat_tmp0*weak_mat_tmp12*weak_mat_tmp3*weak_mat_tmp9 - weak_mat_tmp1*weak_mat_tmp32 - weak_mat_tmp32*weak_mat_tmp4 - weak_mat_tmp48)) + weak_mat_tmp47;
    const s_t weak_mat_tmp52 = weak_mat_tmp10*weak_mat_tmp17 + weak_mat_tmp14;
    const s_t weak_mat_tmp53 = weak_mat_tmp38*weak_mat_tmp45;
    const s_t weak_mat_tmp54 = -K*weak_mat_tmp42 + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu2*weak_mat_tmp26*weak_mat_tmp49 - weak_mat_tmp19*weak_mat_tmp3*weak_mat_tmp35 - weak_mat_tmp3*weak_mat_tmp53 - weak_mat_tmp30*weak_mat_tmp42) + C10*(-weak_mat_tmp13*weak_mat_tmp42 - weak_mat_tmp33 + weak_mat_tmp34));
    const s_t weak_mat_tmp55 = K*weak_mat_tmp10 + iso_relaxation_gamma*(C01*(gu1*weak_mat_tmp53 + gu2*weak_mat_tmp37*weak_mat_tmp38 + weak_mat_tmp10*weak_mat_tmp30 + weak_mat_tmp19*(s_t(4)*weak_mat_tmp10 - s_t(2)*weak_mat_tmp15) + weak_mat_tmp50) + C10*(weak_mat_tmp10*weak_mat_tmp13 + weak_mat_tmp2*weak_mat_tmp32 + weak_mat_tmp32*weak_mat_tmp6 + weak_mat_tmp48)) - weak_mat_tmp47;
    const s_t weak_mat_tmp56 = -K*weak_mat_tmp20 + iso_history_qp[2] + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu1*weak_mat_tmp26*weak_mat_tmp49 - weak_mat_tmp20*weak_mat_tmp30 - weak_mat_tmp20*weak_mat_tmp44 - weak_mat_tmp3*weak_mat_tmp37*weak_mat_tmp38) + C10*(-weak_mat_tmp13*weak_mat_tmp20 + weak_mat_tmp41 - weak_mat_tmp43));
    const s_t material0 = trial_grad0*(K*weak_mat_tmp1 + iso_history_qp[0] + iso_relaxation_gamma*(C01*(-weak_mat_tmp0*weak_mat_tmp25*weak_mat_tmp27 + weak_mat_tmp1*weak_mat_tmp30 + weak_mat_tmp19*(s_t(2)*weak_mat_tmp1 + s_t(2))) + C10*(weak_mat_tmp1*weak_mat_tmp13 + weak_mat_tmp18))) + trial_grad1*weak_mat_tmp46 + trial_grad2*weak_mat_tmp40 + trial_grad3*weak_mat_tmp51;
    const s_t material1 = trial_grad0*weak_mat_tmp46 + trial_grad1*(K*weak_mat_tmp2 + iso_history_qp[1] + iso_relaxation_gamma*(C01*(gu2*weak_mat_tmp27*weak_mat_tmp45 + weak_mat_tmp19*(s_t(2)*weak_mat_tmp2 + s_t(2)) + weak_mat_tmp2*weak_mat_tmp30) + C10*(weak_mat_tmp13*weak_mat_tmp2 + weak_mat_tmp52))) + trial_grad2*weak_mat_tmp55 + trial_grad3*weak_mat_tmp54;
    const s_t material2 = trial_grad0*weak_mat_tmp40 + trial_grad1*weak_mat_tmp55 + trial_grad2*(K*weak_mat_tmp6 + iso_history_qp[0] + iso_relaxation_gamma*(C01*(gu1*weak_mat_tmp27*weak_mat_tmp37 + weak_mat_tmp19*(s_t(2)*weak_mat_tmp6 + s_t(2)) + weak_mat_tmp30*weak_mat_tmp6) + C10*(weak_mat_tmp13*weak_mat_tmp6 + weak_mat_tmp52))) + trial_grad3*weak_mat_tmp56;
    const s_t material3 = trial_grad0*weak_mat_tmp51 + trial_grad1*weak_mat_tmp54 + trial_grad2*weak_mat_tmp56 + trial_grad3*(K*weak_mat_tmp4 + iso_history_qp[1] + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(s_t(2)*weak_mat_tmp4 + s_t(2)) - weak_mat_tmp27*weak_mat_tmp3*weak_mat_tmp49 + weak_mat_tmp30*weak_mat_tmp4) + C10*(weak_mat_tmp13*weak_mat_tmp4 + weak_mat_tmp18)));
    const s_t loperand0 = qw * (material0 * adj_lane0 + material1 * adj_lane1);
    const s_t loperand1 = qw * (material0 * adj_lane2 + material1 * adj_lane3);
    const s_t loperand2 = qw * (material2 * adj_lane0 + material3 * adj_lane1);
    const s_t loperand3 = qw * (material2 * adj_lane2 + material3 * adj_lane3);
      out_streams[0][lane] += -(loperand0) - loperand1;
      out_streams[1][lane] += -(loperand2) - loperand3;
      out_streams[2][lane] += loperand0;
      out_streams[3][lane] += loperand2;
      out_streams[4][lane] += loperand1;
      out_streams[5][lane] += loperand3;
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_prony_d2_simplex_quadrature_output_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR grad_ref_x,
        const s_t *const RSTR grad_ref_y,
        const s_t C01,
        const s_t C10,
        s_t *const RSTR iso_stress,
        const ptrdiff_t qp_stride,
        const s_t *const RSTR u_streams[NS * 2]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
    for (int q = 0; q < NQ; ++q) {
      s_t *const RSTR iso_stress_q = iso_stress + q * qp_stride;
      s_t gu_ref0_values[VS];
      s_t gu_ref1_values[VS];
      s_t gu_ref2_values[VS];
      s_t gu_ref3_values[VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        gu_ref0_values[lane] = s_t(0);
        gu_ref1_values[lane] = s_t(0);
        gu_ref2_values[lane] = s_t(0);
        gu_ref3_values[lane] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t gref0 = grad_ref_x[q * NS + shape];
        const s_t gref1 = grad_ref_y[q * NS + shape];
        const s_t *const RSTR u_shape0 = u_streams[2 * shape];
        const s_t *const RSTR u_shape1 = u_streams[2 * shape + 1];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref0_values[lane] += u_shape0[lane] * gref0;
          gu_ref1_values[lane] += u_shape0[lane] * gref1;
          gu_ref2_values[lane] += u_shape1[lane] * gref0;
          gu_ref3_values[lane] += u_shape1[lane] * gref1;
        }
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
      const s_t adj_lane0 = adj0[goff];
      const s_t adj_lane1 = adj1[goff];
      const s_t adj_lane2 = adj2[goff];
      const s_t adj_lane3 = adj3[goff];
      const s_t det_lane0 = det0[goff];
      const s_t gu_ref0 = gu_ref0_values[lane];
      const s_t gu_ref1 = gu_ref1_values[lane];
      const s_t gu_ref2 = gu_ref2_values[lane];
      const s_t gu_ref3 = gu_ref3_values[lane];
    const s_t idet = s_t(1) / det_lane0;
    const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane2) * idet;
    const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane3) * idet;
    const s_t gu2 = (gu_ref2 * adj_lane0 + gu_ref3 * adj_lane2) * idet;
    const s_t gu3 = (gu_ref2 * adj_lane1 + gu_ref3 * adj_lane3) * idet;
    const s_t qp_out_tmp0 = gu0 + s_t(1);
    const s_t qp_out_tmp1 = gu3 + s_t(1);
    const s_t qp_out_tmp2 = -gu1*gu2 + qp_out_tmp0*qp_out_tmp1;
    const s_t qp_out_tmp3 = pow_m1(qp_out_tmp2);
    const s_t qp_out_tmp4 = pow(qp_out_tmp2, (s_t(-2) / s_t(3)));
    const s_t qp_out_tmp5 = s_t(2)*gu2;
    const s_t qp_out_tmp6 = pow_2(gu2) + pow_2(qp_out_tmp0);
    const s_t qp_out_tmp7 = pow_2(gu1) + pow_2(qp_out_tmp1);
    const s_t qp_out_tmp8 = qp_out_tmp6 + qp_out_tmp7;
    const s_t qp_out_tmp9 = ((s_t(2) / s_t(3)))*(qp_out_tmp8 + s_t(1))/pow(qp_out_tmp2, (s_t(5) / s_t(3)));
    const s_t qp_out_tmp10 = pow(qp_out_tmp2, (s_t(-4) / s_t(3)));
    const s_t qp_out_tmp11 = gu1*qp_out_tmp0 + gu2*qp_out_tmp1;
    const s_t qp_out_tmp12 = s_t(2)*qp_out_tmp1;
    const s_t qp_out_tmp13 = ((s_t(4) / s_t(3)))*(-pow_2(qp_out_tmp11) - (s_t(1) / s_t(2))*pow_2(qp_out_tmp6) - (s_t(1) / s_t(2))*pow_2(qp_out_tmp7) + ((s_t(1) / s_t(2)))*pow_2(qp_out_tmp8) + qp_out_tmp8)/pow(qp_out_tmp2, (s_t(7) / s_t(3)));
    const s_t qp_out_tmp14 = s_t(2)*qp_out_tmp0;
    const s_t qp_out_tmp15 = s_t(2)*gu1;
    const s_t qp_out_tmp16 = C01*(gu2*qp_out_tmp13 + qp_out_tmp10*(s_t(2)*gu1*qp_out_tmp8 + s_t(2)*gu1 - qp_out_tmp11*qp_out_tmp14 - qp_out_tmp15*qp_out_tmp7)) + C10*(gu2*qp_out_tmp9 + qp_out_tmp15*qp_out_tmp4);
    const s_t qp_out_tmp17 = C01*(-qp_out_tmp0*qp_out_tmp13 + qp_out_tmp10*(s_t(2)*gu3 - qp_out_tmp11*qp_out_tmp5 - qp_out_tmp12*qp_out_tmp7 + qp_out_tmp12*qp_out_tmp8 + s_t(2))) + C10*(-qp_out_tmp0*qp_out_tmp9 + s_t(2)*qp_out_tmp1*qp_out_tmp4);
    iso_stress_q[lane] = qp_out_tmp3*(-gu1*(C01*(gu1*qp_out_tmp13 + qp_out_tmp10*(s_t(2)*gu2*qp_out_tmp8 + s_t(2)*gu2 - qp_out_tmp11*qp_out_tmp12 - qp_out_tmp5*qp_out_tmp6)) + C10*(gu1*qp_out_tmp9 + qp_out_tmp4*qp_out_tmp5)) + qp_out_tmp1*(C01*(-qp_out_tmp1*qp_out_tmp13 + qp_out_tmp10*(s_t(2)*gu0 - qp_out_tmp11*qp_out_tmp15 - qp_out_tmp14*qp_out_tmp6 + qp_out_tmp14*qp_out_tmp8 + s_t(2))) + C10*(-qp_out_tmp1*qp_out_tmp9 + qp_out_tmp14*qp_out_tmp4)));
    iso_stress_q[1 * NQ * qp_stride + lane] = qp_out_tmp3*(-gu2*qp_out_tmp16 + qp_out_tmp0*qp_out_tmp17);
    iso_stress_q[2 * NQ * qp_stride + lane] = qp_out_tmp3*(-gu1*qp_out_tmp17 + qp_out_tmp1*qp_out_tmp16);
      }
    }
}

template <typename s_t, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_prony_d2_simplex_tri3_quadrature_output_block(
        const int ne,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t C01,
        const s_t C10,
        s_t *const RSTR iso_stress,
        const ptrdiff_t qp_stride,
        const s_t *const RSTR u_streams[NS * 2]
) {
  static_assert(VS > 0, "VS must be positive");
    {
      s_t *const RSTR iso_stress_q = iso_stress;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = lane;
      const s_t adj_lane0 = adj0[goff];
      const s_t adj_lane1 = adj1[goff];
      const s_t adj_lane2 = adj2[goff];
      const s_t adj_lane3 = adj3[goff];
      const s_t det_lane0 = det0[goff];
      const s_t gu_ref0 = -(u_streams[0][lane]) + u_streams[2][lane];
      const s_t gu_ref1 = -(u_streams[0][lane]) + u_streams[4][lane];
      const s_t gu_ref2 = -(u_streams[1][lane]) + u_streams[3][lane];
      const s_t gu_ref3 = -(u_streams[1][lane]) + u_streams[5][lane];
      const s_t idet = s_t(1) / det_lane0;
      const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane2) * idet;
      const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane3) * idet;
      const s_t gu2 = (gu_ref2 * adj_lane0 + gu_ref3 * adj_lane2) * idet;
      const s_t gu3 = (gu_ref2 * adj_lane1 + gu_ref3 * adj_lane3) * idet;
    const s_t qp_out_tmp0 = gu0 + s_t(1);
    const s_t qp_out_tmp1 = gu3 + s_t(1);
    const s_t qp_out_tmp2 = -gu1*gu2 + qp_out_tmp0*qp_out_tmp1;
    const s_t qp_out_tmp3 = pow_m1(qp_out_tmp2);
    const s_t qp_out_tmp4 = pow(qp_out_tmp2, (s_t(-2) / s_t(3)));
    const s_t qp_out_tmp5 = s_t(2)*gu2;
    const s_t qp_out_tmp6 = pow_2(gu2) + pow_2(qp_out_tmp0);
    const s_t qp_out_tmp7 = pow_2(gu1) + pow_2(qp_out_tmp1);
    const s_t qp_out_tmp8 = qp_out_tmp6 + qp_out_tmp7;
    const s_t qp_out_tmp9 = ((s_t(2) / s_t(3)))*(qp_out_tmp8 + s_t(1))/pow(qp_out_tmp2, (s_t(5) / s_t(3)));
    const s_t qp_out_tmp10 = pow(qp_out_tmp2, (s_t(-4) / s_t(3)));
    const s_t qp_out_tmp11 = gu1*qp_out_tmp0 + gu2*qp_out_tmp1;
    const s_t qp_out_tmp12 = s_t(2)*qp_out_tmp1;
    const s_t qp_out_tmp13 = ((s_t(4) / s_t(3)))*(-pow_2(qp_out_tmp11) - (s_t(1) / s_t(2))*pow_2(qp_out_tmp6) - (s_t(1) / s_t(2))*pow_2(qp_out_tmp7) + ((s_t(1) / s_t(2)))*pow_2(qp_out_tmp8) + qp_out_tmp8)/pow(qp_out_tmp2, (s_t(7) / s_t(3)));
    const s_t qp_out_tmp14 = s_t(2)*qp_out_tmp0;
    const s_t qp_out_tmp15 = s_t(2)*gu1;
    const s_t qp_out_tmp16 = C01*(gu2*qp_out_tmp13 + qp_out_tmp10*(s_t(2)*gu1*qp_out_tmp8 + s_t(2)*gu1 - qp_out_tmp11*qp_out_tmp14 - qp_out_tmp15*qp_out_tmp7)) + C10*(gu2*qp_out_tmp9 + qp_out_tmp15*qp_out_tmp4);
    const s_t qp_out_tmp17 = C01*(-qp_out_tmp0*qp_out_tmp13 + qp_out_tmp10*(s_t(2)*gu3 - qp_out_tmp11*qp_out_tmp5 - qp_out_tmp12*qp_out_tmp7 + qp_out_tmp12*qp_out_tmp8 + s_t(2))) + C10*(-qp_out_tmp0*qp_out_tmp9 + s_t(2)*qp_out_tmp1*qp_out_tmp4);
    iso_stress_q[lane] = qp_out_tmp3*(-gu1*(C01*(gu1*qp_out_tmp13 + qp_out_tmp10*(s_t(2)*gu2*qp_out_tmp8 + s_t(2)*gu2 - qp_out_tmp11*qp_out_tmp12 - qp_out_tmp5*qp_out_tmp6)) + C10*(gu1*qp_out_tmp9 + qp_out_tmp4*qp_out_tmp5)) + qp_out_tmp1*(C01*(-qp_out_tmp1*qp_out_tmp13 + qp_out_tmp10*(s_t(2)*gu0 - qp_out_tmp11*qp_out_tmp15 - qp_out_tmp14*qp_out_tmp6 + qp_out_tmp14*qp_out_tmp8 + s_t(2))) + C10*(-qp_out_tmp1*qp_out_tmp9 + qp_out_tmp14*qp_out_tmp4)));
    iso_stress_q[1 * qp_stride + lane] = qp_out_tmp3*(-gu2*qp_out_tmp16 + qp_out_tmp0*qp_out_tmp17);
    iso_stress_q[2 * qp_stride + lane] = qp_out_tmp3*(-gu1*qp_out_tmp17 + qp_out_tmp1*qp_out_tmp16);
      }
    }
}

} // namespace codegen
} // namespace sfem

#endif
