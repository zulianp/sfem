#ifndef CONTINUED_MOONEY_RIVLIN_PRONY_D3_SIMPLEX_LOCAL_HPP
#define CONTINUED_MOONEY_RIVLIN_PRONY_D3_SIMPLEX_LOCAL_HPP
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
static SFEM_INLINE void continued_mooney_rivlin_prony_d3_simplex_objective_block(
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
        const s_t *const RSTR grad_ref_x,
        const s_t *const RSTR grad_ref_y,
        const s_t *const RSTR grad_ref_z,
        const s_t *const RSTR q_weight,
        const s_t C01,
        const s_t C10,
        const s_t Jc,
        const s_t Jmin,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const ptrdiff_t qp_stride,
        const s_t *const RSTR u_streams[NS * 3],
        const s_t *const RSTR h_streams[NS * 3],
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
      s_t gu_ref4_values[VS];
      s_t grad_h_ref4_values[VS];
      s_t gu_ref5_values[VS];
      s_t grad_h_ref5_values[VS];
      s_t gu_ref6_values[VS];
      s_t grad_h_ref6_values[VS];
      s_t gu_ref7_values[VS];
      s_t grad_h_ref7_values[VS];
      s_t gu_ref8_values[VS];
      s_t grad_h_ref8_values[VS];
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
        gu_ref4_values[lane] = s_t(0);
        grad_h_ref4_values[lane] = s_t(0);
        gu_ref5_values[lane] = s_t(0);
        grad_h_ref5_values[lane] = s_t(0);
        gu_ref6_values[lane] = s_t(0);
        grad_h_ref6_values[lane] = s_t(0);
        gu_ref7_values[lane] = s_t(0);
        grad_h_ref7_values[lane] = s_t(0);
        gu_ref8_values[lane] = s_t(0);
        grad_h_ref8_values[lane] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t gref0 = grad_ref_x[q * NS + shape];
        const s_t gref1 = grad_ref_y[q * NS + shape];
        const s_t gref2 = grad_ref_z[q * NS + shape];
        const s_t *const RSTR u_shape0 = u_streams[3 * shape];
        const s_t *const RSTR u_shape1 = u_streams[3 * shape + 1];
        const s_t *const RSTR u_shape2 = u_streams[3 * shape + 2];
        const s_t *const RSTR h_shape0 = h_streams[3 * shape];
        const s_t *const RSTR h_shape1 = h_streams[3 * shape + 1];
        const s_t *const RSTR h_shape2 = h_streams[3 * shape + 2];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref0_values[lane] += u_shape0[lane] * gref0;
          grad_h_ref0_values[lane] += h_shape0[lane] * gref0;
          gu_ref1_values[lane] += u_shape0[lane] * gref1;
          grad_h_ref1_values[lane] += h_shape0[lane] * gref1;
          gu_ref2_values[lane] += u_shape0[lane] * gref2;
          grad_h_ref2_values[lane] += h_shape0[lane] * gref2;
          gu_ref3_values[lane] += u_shape1[lane] * gref0;
          grad_h_ref3_values[lane] += h_shape1[lane] * gref0;
          gu_ref4_values[lane] += u_shape1[lane] * gref1;
          grad_h_ref4_values[lane] += h_shape1[lane] * gref1;
          gu_ref5_values[lane] += u_shape1[lane] * gref2;
          grad_h_ref5_values[lane] += h_shape1[lane] * gref2;
          gu_ref6_values[lane] += u_shape2[lane] * gref0;
          grad_h_ref6_values[lane] += h_shape2[lane] * gref0;
          gu_ref7_values[lane] += u_shape2[lane] * gref1;
          grad_h_ref7_values[lane] += h_shape2[lane] * gref1;
          gu_ref8_values[lane] += u_shape2[lane] * gref2;
          grad_h_ref8_values[lane] += h_shape2[lane] * gref2;
        }
      }
      s_t gu_base_v[9 * VS];
      s_t trial_grad_v[9 * VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
      const s_t adj_lane0 = adj0[goff];
      const s_t adj_lane1 = adj1[goff];
      const s_t adj_lane2 = adj2[goff];
      const s_t adj_lane3 = adj3[goff];
      const s_t adj_lane4 = adj4[goff];
      const s_t adj_lane5 = adj5[goff];
      const s_t adj_lane6 = adj6[goff];
      const s_t adj_lane7 = adj7[goff];
      const s_t adj_lane8 = adj8[goff];
      const s_t det_lane0 = det0[goff];
      const s_t gu_ref0 = gu_ref0_values[lane];
      const s_t grad_h_ref0 = grad_h_ref0_values[lane];
      const s_t gu_ref1 = gu_ref1_values[lane];
      const s_t grad_h_ref1 = grad_h_ref1_values[lane];
      const s_t gu_ref2 = gu_ref2_values[lane];
      const s_t grad_h_ref2 = grad_h_ref2_values[lane];
      const s_t gu_ref3 = gu_ref3_values[lane];
      const s_t grad_h_ref3 = grad_h_ref3_values[lane];
      const s_t gu_ref4 = gu_ref4_values[lane];
      const s_t grad_h_ref4 = grad_h_ref4_values[lane];
      const s_t gu_ref5 = gu_ref5_values[lane];
      const s_t grad_h_ref5 = grad_h_ref5_values[lane];
      const s_t gu_ref6 = gu_ref6_values[lane];
      const s_t grad_h_ref6 = grad_h_ref6_values[lane];
      const s_t gu_ref7 = gu_ref7_values[lane];
      const s_t grad_h_ref7 = grad_h_ref7_values[lane];
      const s_t gu_ref8 = gu_ref8_values[lane];
      const s_t grad_h_ref8 = grad_h_ref8_values[lane];
    const s_t idet = s_t(1) / det_lane0;
    gu_base_v[0 * VS + lane] = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane3 + gu_ref2 * adj_lane6) * idet;
    trial_grad_v[0 * VS + lane] = (grad_h_ref0 * adj_lane0 + grad_h_ref1 * adj_lane3 + grad_h_ref2 * adj_lane6) * idet;
    gu_base_v[1 * VS + lane] = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane4 + gu_ref2 * adj_lane7) * idet;
    trial_grad_v[1 * VS + lane] = (grad_h_ref0 * adj_lane1 + grad_h_ref1 * adj_lane4 + grad_h_ref2 * adj_lane7) * idet;
    gu_base_v[2 * VS + lane] = (gu_ref0 * adj_lane2 + gu_ref1 * adj_lane5 + gu_ref2 * adj_lane8) * idet;
    trial_grad_v[2 * VS + lane] = (grad_h_ref0 * adj_lane2 + grad_h_ref1 * adj_lane5 + grad_h_ref2 * adj_lane8) * idet;
    gu_base_v[3 * VS + lane] = (gu_ref3 * adj_lane0 + gu_ref4 * adj_lane3 + gu_ref5 * adj_lane6) * idet;
    trial_grad_v[3 * VS + lane] = (grad_h_ref3 * adj_lane0 + grad_h_ref4 * adj_lane3 + grad_h_ref5 * adj_lane6) * idet;
    gu_base_v[4 * VS + lane] = (gu_ref3 * adj_lane1 + gu_ref4 * adj_lane4 + gu_ref5 * adj_lane7) * idet;
    trial_grad_v[4 * VS + lane] = (grad_h_ref3 * adj_lane1 + grad_h_ref4 * adj_lane4 + grad_h_ref5 * adj_lane7) * idet;
    gu_base_v[5 * VS + lane] = (gu_ref3 * adj_lane2 + gu_ref4 * adj_lane5 + gu_ref5 * adj_lane8) * idet;
    trial_grad_v[5 * VS + lane] = (grad_h_ref3 * adj_lane2 + grad_h_ref4 * adj_lane5 + grad_h_ref5 * adj_lane8) * idet;
    gu_base_v[6 * VS + lane] = (gu_ref6 * adj_lane0 + gu_ref7 * adj_lane3 + gu_ref8 * adj_lane6) * idet;
    trial_grad_v[6 * VS + lane] = (grad_h_ref6 * adj_lane0 + grad_h_ref7 * adj_lane3 + grad_h_ref8 * adj_lane6) * idet;
    gu_base_v[7 * VS + lane] = (gu_ref6 * adj_lane1 + gu_ref7 * adj_lane4 + gu_ref8 * adj_lane7) * idet;
    trial_grad_v[7 * VS + lane] = (grad_h_ref6 * adj_lane1 + grad_h_ref7 * adj_lane4 + grad_h_ref8 * adj_lane7) * idet;
    gu_base_v[8 * VS + lane] = (gu_ref6 * adj_lane2 + gu_ref7 * adj_lane5 + gu_ref8 * adj_lane8) * idet;
    trial_grad_v[8 * VS + lane] = (grad_h_ref6 * adj_lane2 + grad_h_ref7 * adj_lane5 + grad_h_ref8 * adj_lane8) * idet;
      }
      for (int step = 0; step < nsteps; ++step) {
        const s_t alpha = steps[step];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const ptrdiff_t goff = q * geometry_stride + lane;
          const s_t det_lane0 = det0[goff];
          const s_t iso_history_qp[6] = {iso_history_q[lane], iso_history_q[1 * NQ * qp_stride + lane], iso_history_q[2 * NQ * qp_stride + lane], iso_history_q[3 * NQ * qp_stride + lane], iso_history_q[4 * NQ * qp_stride + lane], iso_history_q[5 * NQ * qp_stride + lane]};
          const s_t gu0 = gu_base_v[0 * VS + lane] + alpha * trial_grad_v[0 * VS + lane];
          const s_t gu1 = gu_base_v[1 * VS + lane] + alpha * trial_grad_v[1 * VS + lane];
          const s_t gu2 = gu_base_v[2 * VS + lane] + alpha * trial_grad_v[2 * VS + lane];
          const s_t gu3 = gu_base_v[3 * VS + lane] + alpha * trial_grad_v[3 * VS + lane];
          const s_t gu4 = gu_base_v[4 * VS + lane] + alpha * trial_grad_v[4 * VS + lane];
          const s_t gu5 = gu_base_v[5 * VS + lane] + alpha * trial_grad_v[5 * VS + lane];
          const s_t gu6 = gu_base_v[6 * VS + lane] + alpha * trial_grad_v[6 * VS + lane];
          const s_t gu7 = gu_base_v[7 * VS + lane] + alpha * trial_grad_v[7 * VS + lane];
          const s_t gu8 = gu_base_v[8 * VS + lane] + alpha * trial_grad_v[8 * VS + lane];
    const s_t weak_obj_tmp0 = pow_2(gu3) + pow_2(gu6);
    const s_t weak_obj_tmp1 = pow_2(gu1) + pow_2(gu7);
    const s_t weak_obj_tmp2 = pow_2(gu2) + pow_2(gu5);
    const s_t weak_obj_tmp3 = gu0 + s_t(1);
    const s_t weak_obj_tmp4 = gu4 + s_t(1);
    const s_t weak_obj_tmp5 = gu1*weak_obj_tmp3 + gu3*weak_obj_tmp4 + gu6*gu7;
    const s_t weak_obj_tmp6 = gu8 + s_t(1);
    const s_t weak_obj_tmp7 = gu2*weak_obj_tmp3 + gu3*gu5 + gu6*weak_obj_tmp6;
    const s_t weak_obj_tmp8 = gu1*gu2 + gu5*weak_obj_tmp4 + gu7*weak_obj_tmp6;
    const s_t weak_obj_tmp9 = gu0*gu4;
    const s_t weak_obj_tmp10 = gu1*gu3;
    const s_t weak_obj_tmp11 = gu2*gu6;
    const s_t weak_obj_tmp12 = gu5*gu7;
    const s_t weak_obj_tmp13 = weak_obj_tmp1 + pow_2(weak_obj_tmp4);
    const s_t weak_obj_tmp14 = weak_obj_tmp2 + pow_2(weak_obj_tmp6);
    const s_t weak_obj_tmp15 = weak_obj_tmp0 + pow_2(weak_obj_tmp3);
    const s_t weak_obj_tmp16 = weak_obj_tmp13 + weak_obj_tmp14 + weak_obj_tmp15;
    const s_t weak_obj_tmp17 = gu1*gu5*gu6 + gu2*gu3*gu7 - weak_obj_tmp10*weak_obj_tmp6 - weak_obj_tmp11*weak_obj_tmp4 - weak_obj_tmp12*weak_obj_tmp3 + weak_obj_tmp3*weak_obj_tmp4*weak_obj_tmp6;
    value[step * value_stride + lane] += qw * det_lane0 * (((s_t(1) / s_t(2)))*K*pow_2(gu0*gu8 - gu0*weak_obj_tmp12 + gu0 + gu1*gu5*gu6 + gu2*gu3*gu7 + gu4*gu8 - gu4*weak_obj_tmp11 + gu4 - gu8*weak_obj_tmp10 + gu8*weak_obj_tmp9 + gu8 - weak_obj_tmp10 - weak_obj_tmp11 - weak_obj_tmp12 + weak_obj_tmp9) + ((s_t(1) / s_t(2)))*iso_history_qp[0]*(pow_2(gu0) + s_t(2)*gu0 + weak_obj_tmp0) + ((s_t(1) / s_t(2)))*iso_history_qp[1]*(pow_2(gu4) + s_t(2)*gu4 + weak_obj_tmp1) + ((s_t(1) / s_t(2)))*iso_history_qp[2]*(pow_2(gu8) + s_t(2)*gu8 + weak_obj_tmp2) + iso_history_qp[3]*weak_obj_tmp5 + iso_history_qp[4]*weak_obj_tmp7 + iso_history_qp[5]*weak_obj_tmp8 + iso_relaxation_gamma*(C01*((-(s_t(1) / s_t(2))*pow_2(weak_obj_tmp13) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp14) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp15) + ((s_t(1) / s_t(2)))*pow_2(weak_obj_tmp16) - pow_2(weak_obj_tmp5) - pow_2(weak_obj_tmp7) - pow_2(weak_obj_tmp8))*sfem_continued_power<s_t>(weak_obj_tmp17, (s_t(4) / s_t(3)), Jc, Jmin) + s_t(-3)) + C10*(weak_obj_tmp16*sfem_continued_power<s_t>(weak_obj_tmp17, (s_t(2) / s_t(3)), Jc, Jmin) + s_t(-3))));
        }
      }
    }
}

template <typename s_t, int NS, int VS>
static SFEM_INLINE void continued_mooney_rivlin_prony_d3_simplex_tet4_objective_block(
        const int ne,
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
        const s_t C01,
        const s_t C10,
        const s_t Jc,
        const s_t Jmin,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const ptrdiff_t qp_stride,
        const s_t *const RSTR u_streams[NS * 3],
        const s_t *const RSTR h_streams[NS * 3],
        const int nsteps,
        const s_t *const RSTR steps,
        const ptrdiff_t value_stride,
        s_t *const RSTR value
) {
  static_assert(VS > 0, "VS must be positive");
    {
      const s_t qw = (s_t(1) / s_t(6));
      const s_t *const RSTR iso_history_q = iso_history;
      s_t gu_base_v[9 * VS];
      s_t trial_grad_v[9 * VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = lane;
      const s_t adj_lane0 = adj0[goff];
      const s_t adj_lane1 = adj1[goff];
      const s_t adj_lane2 = adj2[goff];
      const s_t adj_lane3 = adj3[goff];
      const s_t adj_lane4 = adj4[goff];
      const s_t adj_lane5 = adj5[goff];
      const s_t adj_lane6 = adj6[goff];
      const s_t adj_lane7 = adj7[goff];
      const s_t adj_lane8 = adj8[goff];
      const s_t det_lane0 = det0[goff];
      const s_t gu_ref0 = -(u_streams[0][lane]) + u_streams[3][lane];
      const s_t grad_h_ref0 = -(h_streams[0][lane]) + h_streams[3][lane];
      const s_t gu_ref1 = -(u_streams[0][lane]) + u_streams[6][lane];
      const s_t grad_h_ref1 = -(h_streams[0][lane]) + h_streams[6][lane];
      const s_t gu_ref2 = -(u_streams[0][lane]) + u_streams[9][lane];
      const s_t grad_h_ref2 = -(h_streams[0][lane]) + h_streams[9][lane];
      const s_t gu_ref3 = -(u_streams[1][lane]) + u_streams[4][lane];
      const s_t grad_h_ref3 = -(h_streams[1][lane]) + h_streams[4][lane];
      const s_t gu_ref4 = -(u_streams[1][lane]) + u_streams[7][lane];
      const s_t grad_h_ref4 = -(h_streams[1][lane]) + h_streams[7][lane];
      const s_t gu_ref5 = -(u_streams[1][lane]) + u_streams[10][lane];
      const s_t grad_h_ref5 = -(h_streams[1][lane]) + h_streams[10][lane];
      const s_t gu_ref6 = -(u_streams[2][lane]) + u_streams[5][lane];
      const s_t grad_h_ref6 = -(h_streams[2][lane]) + h_streams[5][lane];
      const s_t gu_ref7 = -(u_streams[2][lane]) + u_streams[8][lane];
      const s_t grad_h_ref7 = -(h_streams[2][lane]) + h_streams[8][lane];
      const s_t gu_ref8 = -(u_streams[2][lane]) + u_streams[11][lane];
      const s_t grad_h_ref8 = -(h_streams[2][lane]) + h_streams[11][lane];
      const s_t idet = s_t(1) / det_lane0;
      gu_base_v[0 * VS + lane] = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane3 + gu_ref2 * adj_lane6) * idet;
      trial_grad_v[0 * VS + lane] = (grad_h_ref0 * adj_lane0 + grad_h_ref1 * adj_lane3 + grad_h_ref2 * adj_lane6) * idet;
      gu_base_v[1 * VS + lane] = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane4 + gu_ref2 * adj_lane7) * idet;
      trial_grad_v[1 * VS + lane] = (grad_h_ref0 * adj_lane1 + grad_h_ref1 * adj_lane4 + grad_h_ref2 * adj_lane7) * idet;
      gu_base_v[2 * VS + lane] = (gu_ref0 * adj_lane2 + gu_ref1 * adj_lane5 + gu_ref2 * adj_lane8) * idet;
      trial_grad_v[2 * VS + lane] = (grad_h_ref0 * adj_lane2 + grad_h_ref1 * adj_lane5 + grad_h_ref2 * adj_lane8) * idet;
      gu_base_v[3 * VS + lane] = (gu_ref3 * adj_lane0 + gu_ref4 * adj_lane3 + gu_ref5 * adj_lane6) * idet;
      trial_grad_v[3 * VS + lane] = (grad_h_ref3 * adj_lane0 + grad_h_ref4 * adj_lane3 + grad_h_ref5 * adj_lane6) * idet;
      gu_base_v[4 * VS + lane] = (gu_ref3 * adj_lane1 + gu_ref4 * adj_lane4 + gu_ref5 * adj_lane7) * idet;
      trial_grad_v[4 * VS + lane] = (grad_h_ref3 * adj_lane1 + grad_h_ref4 * adj_lane4 + grad_h_ref5 * adj_lane7) * idet;
      gu_base_v[5 * VS + lane] = (gu_ref3 * adj_lane2 + gu_ref4 * adj_lane5 + gu_ref5 * adj_lane8) * idet;
      trial_grad_v[5 * VS + lane] = (grad_h_ref3 * adj_lane2 + grad_h_ref4 * adj_lane5 + grad_h_ref5 * adj_lane8) * idet;
      gu_base_v[6 * VS + lane] = (gu_ref6 * adj_lane0 + gu_ref7 * adj_lane3 + gu_ref8 * adj_lane6) * idet;
      trial_grad_v[6 * VS + lane] = (grad_h_ref6 * adj_lane0 + grad_h_ref7 * adj_lane3 + grad_h_ref8 * adj_lane6) * idet;
      gu_base_v[7 * VS + lane] = (gu_ref6 * adj_lane1 + gu_ref7 * adj_lane4 + gu_ref8 * adj_lane7) * idet;
      trial_grad_v[7 * VS + lane] = (grad_h_ref6 * adj_lane1 + grad_h_ref7 * adj_lane4 + grad_h_ref8 * adj_lane7) * idet;
      gu_base_v[8 * VS + lane] = (gu_ref6 * adj_lane2 + gu_ref7 * adj_lane5 + gu_ref8 * adj_lane8) * idet;
      trial_grad_v[8 * VS + lane] = (grad_h_ref6 * adj_lane2 + grad_h_ref7 * adj_lane5 + grad_h_ref8 * adj_lane8) * idet;
      }
      for (int step = 0; step < nsteps; ++step) {
        const s_t alpha = steps[step];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          const ptrdiff_t goff = lane;
          const s_t det_lane0 = det0[goff];
          const s_t iso_history_qp[6] = {iso_history_q[lane], iso_history_q[1 * qp_stride + lane], iso_history_q[2 * qp_stride + lane], iso_history_q[3 * qp_stride + lane], iso_history_q[4 * qp_stride + lane], iso_history_q[5 * qp_stride + lane]};
          const s_t gu0 = gu_base_v[0 * VS + lane] + alpha * trial_grad_v[0 * VS + lane];
          const s_t gu1 = gu_base_v[1 * VS + lane] + alpha * trial_grad_v[1 * VS + lane];
          const s_t gu2 = gu_base_v[2 * VS + lane] + alpha * trial_grad_v[2 * VS + lane];
          const s_t gu3 = gu_base_v[3 * VS + lane] + alpha * trial_grad_v[3 * VS + lane];
          const s_t gu4 = gu_base_v[4 * VS + lane] + alpha * trial_grad_v[4 * VS + lane];
          const s_t gu5 = gu_base_v[5 * VS + lane] + alpha * trial_grad_v[5 * VS + lane];
          const s_t gu6 = gu_base_v[6 * VS + lane] + alpha * trial_grad_v[6 * VS + lane];
          const s_t gu7 = gu_base_v[7 * VS + lane] + alpha * trial_grad_v[7 * VS + lane];
          const s_t gu8 = gu_base_v[8 * VS + lane] + alpha * trial_grad_v[8 * VS + lane];
    const s_t weak_obj_tmp0 = pow_2(gu3) + pow_2(gu6);
    const s_t weak_obj_tmp1 = pow_2(gu1) + pow_2(gu7);
    const s_t weak_obj_tmp2 = pow_2(gu2) + pow_2(gu5);
    const s_t weak_obj_tmp3 = gu0 + s_t(1);
    const s_t weak_obj_tmp4 = gu4 + s_t(1);
    const s_t weak_obj_tmp5 = gu1*weak_obj_tmp3 + gu3*weak_obj_tmp4 + gu6*gu7;
    const s_t weak_obj_tmp6 = gu8 + s_t(1);
    const s_t weak_obj_tmp7 = gu2*weak_obj_tmp3 + gu3*gu5 + gu6*weak_obj_tmp6;
    const s_t weak_obj_tmp8 = gu1*gu2 + gu5*weak_obj_tmp4 + gu7*weak_obj_tmp6;
    const s_t weak_obj_tmp9 = gu0*gu4;
    const s_t weak_obj_tmp10 = gu1*gu3;
    const s_t weak_obj_tmp11 = gu2*gu6;
    const s_t weak_obj_tmp12 = gu5*gu7;
    const s_t weak_obj_tmp13 = weak_obj_tmp1 + pow_2(weak_obj_tmp4);
    const s_t weak_obj_tmp14 = weak_obj_tmp2 + pow_2(weak_obj_tmp6);
    const s_t weak_obj_tmp15 = weak_obj_tmp0 + pow_2(weak_obj_tmp3);
    const s_t weak_obj_tmp16 = weak_obj_tmp13 + weak_obj_tmp14 + weak_obj_tmp15;
    const s_t weak_obj_tmp17 = gu1*gu5*gu6 + gu2*gu3*gu7 - weak_obj_tmp10*weak_obj_tmp6 - weak_obj_tmp11*weak_obj_tmp4 - weak_obj_tmp12*weak_obj_tmp3 + weak_obj_tmp3*weak_obj_tmp4*weak_obj_tmp6;
    value[step * value_stride + lane] += qw * det_lane0 * (((s_t(1) / s_t(2)))*K*pow_2(gu0*gu8 - gu0*weak_obj_tmp12 + gu0 + gu1*gu5*gu6 + gu2*gu3*gu7 + gu4*gu8 - gu4*weak_obj_tmp11 + gu4 - gu8*weak_obj_tmp10 + gu8*weak_obj_tmp9 + gu8 - weak_obj_tmp10 - weak_obj_tmp11 - weak_obj_tmp12 + weak_obj_tmp9) + ((s_t(1) / s_t(2)))*iso_history_qp[0]*(pow_2(gu0) + s_t(2)*gu0 + weak_obj_tmp0) + ((s_t(1) / s_t(2)))*iso_history_qp[1]*(pow_2(gu4) + s_t(2)*gu4 + weak_obj_tmp1) + ((s_t(1) / s_t(2)))*iso_history_qp[2]*(pow_2(gu8) + s_t(2)*gu8 + weak_obj_tmp2) + iso_history_qp[3]*weak_obj_tmp5 + iso_history_qp[4]*weak_obj_tmp7 + iso_history_qp[5]*weak_obj_tmp8 + iso_relaxation_gamma*(C01*((-(s_t(1) / s_t(2))*pow_2(weak_obj_tmp13) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp14) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp15) + ((s_t(1) / s_t(2)))*pow_2(weak_obj_tmp16) - pow_2(weak_obj_tmp5) - pow_2(weak_obj_tmp7) - pow_2(weak_obj_tmp8))*sfem_continued_power<s_t>(weak_obj_tmp17, (s_t(4) / s_t(3)), Jc, Jmin) + s_t(-3)) + C10*(weak_obj_tmp16*sfem_continued_power<s_t>(weak_obj_tmp17, (s_t(2) / s_t(3)), Jc, Jmin) + s_t(-3))));
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void continued_mooney_rivlin_prony_d3_simplex_gradient_block(
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
        const s_t *const RSTR grad_ref_x,
        const s_t *const RSTR grad_ref_y,
        const s_t *const RSTR grad_ref_z,
        const s_t *const RSTR q_weight,
        const s_t C01,
        const s_t C10,
        const s_t Jc,
        const s_t Jmin,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const ptrdiff_t qp_stride,
        const s_t *const RSTR u_streams[NS * 3],
        s_t *const RSTR out_streams[NS * 3]
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
      s_t gu_ref4_values[VS];
      s_t gu_ref5_values[VS];
      s_t gu_ref6_values[VS];
      s_t gu_ref7_values[VS];
      s_t gu_ref8_values[VS];
      s_t loperand0_values[VS];
      s_t loperand1_values[VS];
      s_t loperand2_values[VS];
      s_t loperand3_values[VS];
      s_t loperand4_values[VS];
      s_t loperand5_values[VS];
      s_t loperand6_values[VS];
      s_t loperand7_values[VS];
      s_t loperand8_values[VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        gu_ref0_values[lane] = s_t(0);
        gu_ref1_values[lane] = s_t(0);
        gu_ref2_values[lane] = s_t(0);
        gu_ref3_values[lane] = s_t(0);
        gu_ref4_values[lane] = s_t(0);
        gu_ref5_values[lane] = s_t(0);
        gu_ref6_values[lane] = s_t(0);
        gu_ref7_values[lane] = s_t(0);
        gu_ref8_values[lane] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t gref0 = grad_ref_x[q * NS + shape];
        const s_t gref1 = grad_ref_y[q * NS + shape];
        const s_t gref2 = grad_ref_z[q * NS + shape];
        const s_t *const RSTR u_shape0 = u_streams[3 * shape];
        const s_t *const RSTR u_shape1 = u_streams[3 * shape + 1];
        const s_t *const RSTR u_shape2 = u_streams[3 * shape + 2];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref0_values[lane] += u_shape0[lane] * gref0;
          gu_ref1_values[lane] += u_shape0[lane] * gref1;
          gu_ref2_values[lane] += u_shape0[lane] * gref2;
          gu_ref3_values[lane] += u_shape1[lane] * gref0;
          gu_ref4_values[lane] += u_shape1[lane] * gref1;
          gu_ref5_values[lane] += u_shape1[lane] * gref2;
          gu_ref6_values[lane] += u_shape2[lane] * gref0;
          gu_ref7_values[lane] += u_shape2[lane] * gref1;
          gu_ref8_values[lane] += u_shape2[lane] * gref2;
        }
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
      const s_t adj_lane0 = adj0[goff];
      const s_t adj_lane1 = adj1[goff];
      const s_t adj_lane2 = adj2[goff];
      const s_t adj_lane3 = adj3[goff];
      const s_t adj_lane4 = adj4[goff];
      const s_t adj_lane5 = adj5[goff];
      const s_t adj_lane6 = adj6[goff];
      const s_t adj_lane7 = adj7[goff];
      const s_t adj_lane8 = adj8[goff];
      const s_t det_lane0 = det0[goff];
      const s_t iso_history_qp[6] = {iso_history_q[lane], iso_history_q[1 * NQ * qp_stride + lane], iso_history_q[2 * NQ * qp_stride + lane], iso_history_q[3 * NQ * qp_stride + lane], iso_history_q[4 * NQ * qp_stride + lane], iso_history_q[5 * NQ * qp_stride + lane]};
      const s_t gu_ref0 = gu_ref0_values[lane];
      const s_t gu_ref1 = gu_ref1_values[lane];
      const s_t gu_ref2 = gu_ref2_values[lane];
      const s_t gu_ref3 = gu_ref3_values[lane];
      const s_t gu_ref4 = gu_ref4_values[lane];
      const s_t gu_ref5 = gu_ref5_values[lane];
      const s_t gu_ref6 = gu_ref6_values[lane];
      const s_t gu_ref7 = gu_ref7_values[lane];
      const s_t gu_ref8 = gu_ref8_values[lane];
    const s_t idet = s_t(1) / det_lane0;
    const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane3 + gu_ref2 * adj_lane6) * idet;
    const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane4 + gu_ref2 * adj_lane7) * idet;
    const s_t gu2 = (gu_ref0 * adj_lane2 + gu_ref1 * adj_lane5 + gu_ref2 * adj_lane8) * idet;
    const s_t gu3 = (gu_ref3 * adj_lane0 + gu_ref4 * adj_lane3 + gu_ref5 * adj_lane6) * idet;
    const s_t gu4 = (gu_ref3 * adj_lane1 + gu_ref4 * adj_lane4 + gu_ref5 * adj_lane7) * idet;
    const s_t gu5 = (gu_ref3 * adj_lane2 + gu_ref4 * adj_lane5 + gu_ref5 * adj_lane8) * idet;
    const s_t gu6 = (gu_ref6 * adj_lane0 + gu_ref7 * adj_lane3 + gu_ref8 * adj_lane6) * idet;
    const s_t gu7 = (gu_ref6 * adj_lane1 + gu_ref7 * adj_lane4 + gu_ref8 * adj_lane7) * idet;
    const s_t gu8 = (gu_ref6 * adj_lane2 + gu_ref7 * adj_lane5 + gu_ref8 * adj_lane8) * idet;
    const s_t weak_mat_tmp0 = gu0 + s_t(1);
    const s_t weak_mat_tmp1 = gu5*gu7;
    const s_t weak_mat_tmp2 = gu4 + s_t(1);
    const s_t weak_mat_tmp3 = gu8 + s_t(1);
    const s_t weak_mat_tmp4 = gu0*gu4;
    const s_t weak_mat_tmp5 = gu1*gu3;
    const s_t weak_mat_tmp6 = gu2*gu6;
    const s_t weak_mat_tmp7 = gu3*gu7;
    const s_t weak_mat_tmp8 = ((s_t(1) / s_t(2)))*K*(gu0*gu8 - gu0*weak_mat_tmp1 + gu0 + gu1*gu5*gu6 + gu2*weak_mat_tmp7 + gu4*gu8 - gu4*weak_mat_tmp6 + gu4 + gu8*weak_mat_tmp4 - gu8*weak_mat_tmp5 + gu8 - weak_mat_tmp1 + weak_mat_tmp4 - weak_mat_tmp5 - weak_mat_tmp6);
    const s_t weak_mat_tmp9 = gu1*gu5*gu6 + gu2*gu3*gu7 - weak_mat_tmp0*weak_mat_tmp1 + weak_mat_tmp0*weak_mat_tmp2*weak_mat_tmp3 - weak_mat_tmp2*weak_mat_tmp6 - weak_mat_tmp3*weak_mat_tmp5;
    const s_t weak_mat_tmp10 = sfem_continued_power<s_t>(weak_mat_tmp9, (s_t(2) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp11 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp12 = -weak_mat_tmp1 + weak_mat_tmp2*weak_mat_tmp3;
    const s_t weak_mat_tmp13 = pow_2(gu3) + pow_2(gu6) + pow_2(weak_mat_tmp0);
    const s_t weak_mat_tmp14 = pow_2(gu1) + pow_2(gu7) + pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp15 = pow_2(gu2) + pow_2(gu5) + pow_2(weak_mat_tmp3);
    const s_t weak_mat_tmp16 = weak_mat_tmp13 + weak_mat_tmp14 + weak_mat_tmp15;
    const s_t weak_mat_tmp17 = weak_mat_tmp16*sfem_continued_power_d1<s_t>(weak_mat_tmp9, (s_t(2) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp18 = sfem_continued_power<s_t>(weak_mat_tmp9, (s_t(4) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp19 = gu1*weak_mat_tmp0 + gu3*weak_mat_tmp2 + gu6*gu7;
    const s_t weak_mat_tmp20 = s_t(2)*gu1;
    const s_t weak_mat_tmp21 = gu2*weak_mat_tmp0 + gu3*gu5 + gu6*weak_mat_tmp3;
    const s_t weak_mat_tmp22 = s_t(2)*gu2;
    const s_t weak_mat_tmp23 = gu1*gu2 + gu5*weak_mat_tmp2 + gu7*weak_mat_tmp3;
    const s_t weak_mat_tmp24 = (-(s_t(1) / s_t(2))*pow_2(weak_mat_tmp13) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp14) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp15) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp16) - pow_2(weak_mat_tmp19) - pow_2(weak_mat_tmp21) - pow_2(weak_mat_tmp23))*sfem_continued_power_d1<s_t>(weak_mat_tmp9, (s_t(4) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp25 = gu3*weak_mat_tmp3;
    const s_t weak_mat_tmp26 = gu5*gu6 - weak_mat_tmp25;
    const s_t weak_mat_tmp27 = gu6*weak_mat_tmp2;
    const s_t weak_mat_tmp28 = -weak_mat_tmp27 + weak_mat_tmp7;
    const s_t weak_mat_tmp29 = gu1*weak_mat_tmp3;
    const s_t weak_mat_tmp30 = s_t(2)*gu3;
    const s_t weak_mat_tmp31 = gu2*gu7 - weak_mat_tmp29;
    const s_t weak_mat_tmp32 = s_t(2)*gu5;
    const s_t weak_mat_tmp33 = s_t(2)*weak_mat_tmp2;
    const s_t weak_mat_tmp34 = weak_mat_tmp0*weak_mat_tmp3 - weak_mat_tmp6;
    const s_t weak_mat_tmp35 = gu1*gu6;
    const s_t weak_mat_tmp36 = gu7*weak_mat_tmp0;
    const s_t weak_mat_tmp37 = weak_mat_tmp35 - weak_mat_tmp36;
    const s_t weak_mat_tmp38 = gu1*gu5;
    const s_t weak_mat_tmp39 = gu2*weak_mat_tmp2;
    const s_t weak_mat_tmp40 = s_t(2)*gu6;
    const s_t weak_mat_tmp41 = weak_mat_tmp38 - weak_mat_tmp39;
    const s_t weak_mat_tmp42 = s_t(2)*gu7;
    const s_t weak_mat_tmp43 = s_t(2)*weak_mat_tmp3;
    const s_t weak_mat_tmp44 = gu2*gu3;
    const s_t weak_mat_tmp45 = gu5*weak_mat_tmp0;
    const s_t weak_mat_tmp46 = weak_mat_tmp44 - weak_mat_tmp45;
    const s_t weak_mat_tmp47 = weak_mat_tmp0*weak_mat_tmp2 - weak_mat_tmp5;
    const s_t material0 = gu1*iso_history_qp[3] + gu2*iso_history_qp[4] + iso_history_qp[0]*weak_mat_tmp0 + iso_relaxation_gamma*(C01*(weak_mat_tmp12*weak_mat_tmp24 + weak_mat_tmp18*(s_t(2)*weak_mat_tmp0*weak_mat_tmp16 - weak_mat_tmp11*weak_mat_tmp13 - weak_mat_tmp19*weak_mat_tmp20 - weak_mat_tmp21*weak_mat_tmp22)) + C10*(weak_mat_tmp10*weak_mat_tmp11 + weak_mat_tmp12*weak_mat_tmp17)) + weak_mat_tmp8*(-s_t(2)*weak_mat_tmp1 + s_t(2)*weak_mat_tmp2*weak_mat_tmp3);
    const s_t material1 = gu1*iso_history_qp[1] + gu2*iso_history_qp[5] + iso_history_qp[3]*weak_mat_tmp0 + iso_relaxation_gamma*(C01*(weak_mat_tmp18*(s_t(2)*gu1*weak_mat_tmp16 - weak_mat_tmp11*weak_mat_tmp19 - weak_mat_tmp14*weak_mat_tmp20 - weak_mat_tmp22*weak_mat_tmp23) + weak_mat_tmp24*weak_mat_tmp26) + C10*(weak_mat_tmp10*weak_mat_tmp20 + weak_mat_tmp17*weak_mat_tmp26)) + weak_mat_tmp8*(s_t(2)*gu5*gu6 - s_t(2)*weak_mat_tmp25);
    const s_t material2 = gu1*iso_history_qp[5] + gu2*iso_history_qp[2] + iso_history_qp[4]*weak_mat_tmp0 + iso_relaxation_gamma*(C01*(weak_mat_tmp18*(s_t(2)*gu2*weak_mat_tmp16 - weak_mat_tmp11*weak_mat_tmp21 - weak_mat_tmp15*weak_mat_tmp22 - weak_mat_tmp20*weak_mat_tmp23) + weak_mat_tmp24*weak_mat_tmp28) + C10*(weak_mat_tmp10*weak_mat_tmp22 + weak_mat_tmp17*weak_mat_tmp28)) + weak_mat_tmp8*(-s_t(2)*weak_mat_tmp27 + s_t(2)*weak_mat_tmp7);
    const s_t material3 = gu3*iso_history_qp[0] + gu5*iso_history_qp[4] + iso_history_qp[3]*weak_mat_tmp2 + iso_relaxation_gamma*(C01*(weak_mat_tmp18*(s_t(2)*gu3*weak_mat_tmp16 - weak_mat_tmp13*weak_mat_tmp30 - weak_mat_tmp19*weak_mat_tmp33 - weak_mat_tmp21*weak_mat_tmp32) + weak_mat_tmp24*weak_mat_tmp31) + C10*(weak_mat_tmp10*weak_mat_tmp30 + weak_mat_tmp17*weak_mat_tmp31)) + weak_mat_tmp8*(s_t(2)*gu2*gu7 - s_t(2)*weak_mat_tmp29);
    const s_t material4 = gu3*iso_history_qp[3] + gu5*iso_history_qp[5] + iso_history_qp[1]*weak_mat_tmp2 + iso_relaxation_gamma*(C01*(weak_mat_tmp18*(-weak_mat_tmp14*weak_mat_tmp33 + s_t(2)*weak_mat_tmp16*weak_mat_tmp2 - weak_mat_tmp19*weak_mat_tmp30 - weak_mat_tmp23*weak_mat_tmp32) + weak_mat_tmp24*weak_mat_tmp34) + C10*(weak_mat_tmp10*weak_mat_tmp33 + weak_mat_tmp17*weak_mat_tmp34)) + weak_mat_tmp8*(s_t(2)*weak_mat_tmp0*weak_mat_tmp3 - s_t(2)*weak_mat_tmp6);
    const s_t material5 = gu3*iso_history_qp[4] + gu5*iso_history_qp[2] + iso_history_qp[5]*weak_mat_tmp2 + iso_relaxation_gamma*(C01*(weak_mat_tmp18*(s_t(2)*gu5*weak_mat_tmp16 - weak_mat_tmp15*weak_mat_tmp32 - weak_mat_tmp21*weak_mat_tmp30 - weak_mat_tmp23*weak_mat_tmp33) + weak_mat_tmp24*weak_mat_tmp37) + C10*(weak_mat_tmp10*weak_mat_tmp32 + weak_mat_tmp17*weak_mat_tmp37)) + weak_mat_tmp8*(s_t(2)*weak_mat_tmp35 - s_t(2)*weak_mat_tmp36);
    const s_t material6 = gu6*iso_history_qp[0] + gu7*iso_history_qp[3] + iso_history_qp[4]*weak_mat_tmp3 + iso_relaxation_gamma*(C01*(weak_mat_tmp18*(s_t(2)*gu6*weak_mat_tmp16 - weak_mat_tmp13*weak_mat_tmp40 - weak_mat_tmp19*weak_mat_tmp42 - weak_mat_tmp21*weak_mat_tmp43) + weak_mat_tmp24*weak_mat_tmp41) + C10*(weak_mat_tmp10*weak_mat_tmp40 + weak_mat_tmp17*weak_mat_tmp41)) + weak_mat_tmp8*(s_t(2)*weak_mat_tmp38 - s_t(2)*weak_mat_tmp39);
    const s_t material7 = gu6*iso_history_qp[3] + gu7*iso_history_qp[1] + iso_history_qp[5]*weak_mat_tmp3 + iso_relaxation_gamma*(C01*(weak_mat_tmp18*(s_t(2)*gu7*weak_mat_tmp16 - weak_mat_tmp14*weak_mat_tmp42 - weak_mat_tmp19*weak_mat_tmp40 - weak_mat_tmp23*weak_mat_tmp43) + weak_mat_tmp24*weak_mat_tmp46) + C10*(weak_mat_tmp10*weak_mat_tmp42 + weak_mat_tmp17*weak_mat_tmp46)) + weak_mat_tmp8*(s_t(2)*weak_mat_tmp44 - s_t(2)*weak_mat_tmp45);
    const s_t material8 = gu6*iso_history_qp[4] + gu7*iso_history_qp[5] + iso_history_qp[2]*weak_mat_tmp3 + iso_relaxation_gamma*(C01*(weak_mat_tmp18*(-weak_mat_tmp15*weak_mat_tmp43 + s_t(2)*weak_mat_tmp16*weak_mat_tmp3 - weak_mat_tmp21*weak_mat_tmp40 - weak_mat_tmp23*weak_mat_tmp42) + weak_mat_tmp24*weak_mat_tmp47) + C10*(weak_mat_tmp10*weak_mat_tmp43 + weak_mat_tmp17*weak_mat_tmp47)) + weak_mat_tmp8*(s_t(2)*weak_mat_tmp0*weak_mat_tmp2 - s_t(2)*weak_mat_tmp5);
      loperand0_values[lane] = qw * (material0 * adj_lane0 + material1 * adj_lane1 + material2 * adj_lane2);
      loperand1_values[lane] = qw * (material0 * adj_lane3 + material1 * adj_lane4 + material2 * adj_lane5);
      loperand2_values[lane] = qw * (material0 * adj_lane6 + material1 * adj_lane7 + material2 * adj_lane8);
      loperand3_values[lane] = qw * (material3 * adj_lane0 + material4 * adj_lane1 + material5 * adj_lane2);
      loperand4_values[lane] = qw * (material3 * adj_lane3 + material4 * adj_lane4 + material5 * adj_lane5);
      loperand5_values[lane] = qw * (material3 * adj_lane6 + material4 * adj_lane7 + material5 * adj_lane8);
      loperand6_values[lane] = qw * (material6 * adj_lane0 + material7 * adj_lane1 + material8 * adj_lane2);
      loperand7_values[lane] = qw * (material6 * adj_lane3 + material7 * adj_lane4 + material8 * adj_lane5);
      loperand8_values[lane] = qw * (material6 * adj_lane6 + material7 * adj_lane7 + material8 * adj_lane8);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t tref0 = grad_ref_x[q * NS + shape];
        const s_t tref1 = grad_ref_y[q * NS + shape];
        const s_t tref2 = grad_ref_z[q * NS + shape];
        s_t *const RSTR out_shape0 = out_streams[3 * shape];
        s_t *const RSTR out_shape1 = out_streams[3 * shape + 1];
        s_t *const RSTR out_shape2 = out_streams[3 * shape + 2];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          out_shape0[lane] += loperand0_values[lane] * tref0 + loperand1_values[lane] * tref1 + loperand2_values[lane] * tref2;
          out_shape1[lane] += loperand3_values[lane] * tref0 + loperand4_values[lane] * tref1 + loperand5_values[lane] * tref2;
          out_shape2[lane] += loperand6_values[lane] * tref0 + loperand7_values[lane] * tref1 + loperand8_values[lane] * tref2;
        }
      }
    }
}

template <typename s_t, int NS, int VS>
static SFEM_INLINE void continued_mooney_rivlin_prony_d3_simplex_tet4_gradient_block(
        const int ne,
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
        const s_t C01,
        const s_t C10,
        const s_t Jc,
        const s_t Jmin,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const ptrdiff_t qp_stride,
        const s_t *const RSTR u_streams[NS * 3],
        s_t *const RSTR out_streams[NS * 3]
) {
  static_assert(VS > 0, "VS must be positive");
    {
      const s_t qw = (s_t(1) / s_t(6));
      const s_t *const RSTR iso_history_q = iso_history;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = lane;
      const s_t adj_lane0 = adj0[goff];
      const s_t adj_lane1 = adj1[goff];
      const s_t adj_lane2 = adj2[goff];
      const s_t adj_lane3 = adj3[goff];
      const s_t adj_lane4 = adj4[goff];
      const s_t adj_lane5 = adj5[goff];
      const s_t adj_lane6 = adj6[goff];
      const s_t adj_lane7 = adj7[goff];
      const s_t adj_lane8 = adj8[goff];
      const s_t det_lane0 = det0[goff];
      const s_t iso_history_qp[6] = {iso_history_q[lane], iso_history_q[1 * qp_stride + lane], iso_history_q[2 * qp_stride + lane], iso_history_q[3 * qp_stride + lane], iso_history_q[4 * qp_stride + lane], iso_history_q[5 * qp_stride + lane]};
      const s_t gu_ref0 = -(u_streams[0][lane]) + u_streams[3][lane];
      const s_t gu_ref1 = -(u_streams[0][lane]) + u_streams[6][lane];
      const s_t gu_ref2 = -(u_streams[0][lane]) + u_streams[9][lane];
      const s_t gu_ref3 = -(u_streams[1][lane]) + u_streams[4][lane];
      const s_t gu_ref4 = -(u_streams[1][lane]) + u_streams[7][lane];
      const s_t gu_ref5 = -(u_streams[1][lane]) + u_streams[10][lane];
      const s_t gu_ref6 = -(u_streams[2][lane]) + u_streams[5][lane];
      const s_t gu_ref7 = -(u_streams[2][lane]) + u_streams[8][lane];
      const s_t gu_ref8 = -(u_streams[2][lane]) + u_streams[11][lane];
      const s_t idet = s_t(1) / det_lane0;
      const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane3 + gu_ref2 * adj_lane6) * idet;
      const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane4 + gu_ref2 * adj_lane7) * idet;
      const s_t gu2 = (gu_ref0 * adj_lane2 + gu_ref1 * adj_lane5 + gu_ref2 * adj_lane8) * idet;
      const s_t gu3 = (gu_ref3 * adj_lane0 + gu_ref4 * adj_lane3 + gu_ref5 * adj_lane6) * idet;
      const s_t gu4 = (gu_ref3 * adj_lane1 + gu_ref4 * adj_lane4 + gu_ref5 * adj_lane7) * idet;
      const s_t gu5 = (gu_ref3 * adj_lane2 + gu_ref4 * adj_lane5 + gu_ref5 * adj_lane8) * idet;
      const s_t gu6 = (gu_ref6 * adj_lane0 + gu_ref7 * adj_lane3 + gu_ref8 * adj_lane6) * idet;
      const s_t gu7 = (gu_ref6 * adj_lane1 + gu_ref7 * adj_lane4 + gu_ref8 * adj_lane7) * idet;
      const s_t gu8 = (gu_ref6 * adj_lane2 + gu_ref7 * adj_lane5 + gu_ref8 * adj_lane8) * idet;
    const s_t weak_mat_tmp0 = gu0 + s_t(1);
    const s_t weak_mat_tmp1 = gu5*gu7;
    const s_t weak_mat_tmp2 = gu4 + s_t(1);
    const s_t weak_mat_tmp3 = gu8 + s_t(1);
    const s_t weak_mat_tmp4 = gu0*gu4;
    const s_t weak_mat_tmp5 = gu1*gu3;
    const s_t weak_mat_tmp6 = gu2*gu6;
    const s_t weak_mat_tmp7 = gu3*gu7;
    const s_t weak_mat_tmp8 = ((s_t(1) / s_t(2)))*K*(gu0*gu8 - gu0*weak_mat_tmp1 + gu0 + gu1*gu5*gu6 + gu2*weak_mat_tmp7 + gu4*gu8 - gu4*weak_mat_tmp6 + gu4 + gu8*weak_mat_tmp4 - gu8*weak_mat_tmp5 + gu8 - weak_mat_tmp1 + weak_mat_tmp4 - weak_mat_tmp5 - weak_mat_tmp6);
    const s_t weak_mat_tmp9 = gu1*gu5*gu6 + gu2*gu3*gu7 - weak_mat_tmp0*weak_mat_tmp1 + weak_mat_tmp0*weak_mat_tmp2*weak_mat_tmp3 - weak_mat_tmp2*weak_mat_tmp6 - weak_mat_tmp3*weak_mat_tmp5;
    const s_t weak_mat_tmp10 = sfem_continued_power<s_t>(weak_mat_tmp9, (s_t(2) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp11 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp12 = -weak_mat_tmp1 + weak_mat_tmp2*weak_mat_tmp3;
    const s_t weak_mat_tmp13 = pow_2(gu3) + pow_2(gu6) + pow_2(weak_mat_tmp0);
    const s_t weak_mat_tmp14 = pow_2(gu1) + pow_2(gu7) + pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp15 = pow_2(gu2) + pow_2(gu5) + pow_2(weak_mat_tmp3);
    const s_t weak_mat_tmp16 = weak_mat_tmp13 + weak_mat_tmp14 + weak_mat_tmp15;
    const s_t weak_mat_tmp17 = weak_mat_tmp16*sfem_continued_power_d1<s_t>(weak_mat_tmp9, (s_t(2) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp18 = sfem_continued_power<s_t>(weak_mat_tmp9, (s_t(4) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp19 = gu1*weak_mat_tmp0 + gu3*weak_mat_tmp2 + gu6*gu7;
    const s_t weak_mat_tmp20 = s_t(2)*gu1;
    const s_t weak_mat_tmp21 = gu2*weak_mat_tmp0 + gu3*gu5 + gu6*weak_mat_tmp3;
    const s_t weak_mat_tmp22 = s_t(2)*gu2;
    const s_t weak_mat_tmp23 = gu1*gu2 + gu5*weak_mat_tmp2 + gu7*weak_mat_tmp3;
    const s_t weak_mat_tmp24 = (-(s_t(1) / s_t(2))*pow_2(weak_mat_tmp13) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp14) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp15) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp16) - pow_2(weak_mat_tmp19) - pow_2(weak_mat_tmp21) - pow_2(weak_mat_tmp23))*sfem_continued_power_d1<s_t>(weak_mat_tmp9, (s_t(4) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp25 = gu3*weak_mat_tmp3;
    const s_t weak_mat_tmp26 = gu5*gu6 - weak_mat_tmp25;
    const s_t weak_mat_tmp27 = gu6*weak_mat_tmp2;
    const s_t weak_mat_tmp28 = -weak_mat_tmp27 + weak_mat_tmp7;
    const s_t weak_mat_tmp29 = gu1*weak_mat_tmp3;
    const s_t weak_mat_tmp30 = s_t(2)*gu3;
    const s_t weak_mat_tmp31 = gu2*gu7 - weak_mat_tmp29;
    const s_t weak_mat_tmp32 = s_t(2)*gu5;
    const s_t weak_mat_tmp33 = s_t(2)*weak_mat_tmp2;
    const s_t weak_mat_tmp34 = weak_mat_tmp0*weak_mat_tmp3 - weak_mat_tmp6;
    const s_t weak_mat_tmp35 = gu1*gu6;
    const s_t weak_mat_tmp36 = gu7*weak_mat_tmp0;
    const s_t weak_mat_tmp37 = weak_mat_tmp35 - weak_mat_tmp36;
    const s_t weak_mat_tmp38 = gu1*gu5;
    const s_t weak_mat_tmp39 = gu2*weak_mat_tmp2;
    const s_t weak_mat_tmp40 = s_t(2)*gu6;
    const s_t weak_mat_tmp41 = weak_mat_tmp38 - weak_mat_tmp39;
    const s_t weak_mat_tmp42 = s_t(2)*gu7;
    const s_t weak_mat_tmp43 = s_t(2)*weak_mat_tmp3;
    const s_t weak_mat_tmp44 = gu2*gu3;
    const s_t weak_mat_tmp45 = gu5*weak_mat_tmp0;
    const s_t weak_mat_tmp46 = weak_mat_tmp44 - weak_mat_tmp45;
    const s_t weak_mat_tmp47 = weak_mat_tmp0*weak_mat_tmp2 - weak_mat_tmp5;
    const s_t material0 = gu1*iso_history_qp[3] + gu2*iso_history_qp[4] + iso_history_qp[0]*weak_mat_tmp0 + iso_relaxation_gamma*(C01*(weak_mat_tmp12*weak_mat_tmp24 + weak_mat_tmp18*(s_t(2)*weak_mat_tmp0*weak_mat_tmp16 - weak_mat_tmp11*weak_mat_tmp13 - weak_mat_tmp19*weak_mat_tmp20 - weak_mat_tmp21*weak_mat_tmp22)) + C10*(weak_mat_tmp10*weak_mat_tmp11 + weak_mat_tmp12*weak_mat_tmp17)) + weak_mat_tmp8*(-s_t(2)*weak_mat_tmp1 + s_t(2)*weak_mat_tmp2*weak_mat_tmp3);
    const s_t material1 = gu1*iso_history_qp[1] + gu2*iso_history_qp[5] + iso_history_qp[3]*weak_mat_tmp0 + iso_relaxation_gamma*(C01*(weak_mat_tmp18*(s_t(2)*gu1*weak_mat_tmp16 - weak_mat_tmp11*weak_mat_tmp19 - weak_mat_tmp14*weak_mat_tmp20 - weak_mat_tmp22*weak_mat_tmp23) + weak_mat_tmp24*weak_mat_tmp26) + C10*(weak_mat_tmp10*weak_mat_tmp20 + weak_mat_tmp17*weak_mat_tmp26)) + weak_mat_tmp8*(s_t(2)*gu5*gu6 - s_t(2)*weak_mat_tmp25);
    const s_t material2 = gu1*iso_history_qp[5] + gu2*iso_history_qp[2] + iso_history_qp[4]*weak_mat_tmp0 + iso_relaxation_gamma*(C01*(weak_mat_tmp18*(s_t(2)*gu2*weak_mat_tmp16 - weak_mat_tmp11*weak_mat_tmp21 - weak_mat_tmp15*weak_mat_tmp22 - weak_mat_tmp20*weak_mat_tmp23) + weak_mat_tmp24*weak_mat_tmp28) + C10*(weak_mat_tmp10*weak_mat_tmp22 + weak_mat_tmp17*weak_mat_tmp28)) + weak_mat_tmp8*(-s_t(2)*weak_mat_tmp27 + s_t(2)*weak_mat_tmp7);
    const s_t material3 = gu3*iso_history_qp[0] + gu5*iso_history_qp[4] + iso_history_qp[3]*weak_mat_tmp2 + iso_relaxation_gamma*(C01*(weak_mat_tmp18*(s_t(2)*gu3*weak_mat_tmp16 - weak_mat_tmp13*weak_mat_tmp30 - weak_mat_tmp19*weak_mat_tmp33 - weak_mat_tmp21*weak_mat_tmp32) + weak_mat_tmp24*weak_mat_tmp31) + C10*(weak_mat_tmp10*weak_mat_tmp30 + weak_mat_tmp17*weak_mat_tmp31)) + weak_mat_tmp8*(s_t(2)*gu2*gu7 - s_t(2)*weak_mat_tmp29);
    const s_t material4 = gu3*iso_history_qp[3] + gu5*iso_history_qp[5] + iso_history_qp[1]*weak_mat_tmp2 + iso_relaxation_gamma*(C01*(weak_mat_tmp18*(-weak_mat_tmp14*weak_mat_tmp33 + s_t(2)*weak_mat_tmp16*weak_mat_tmp2 - weak_mat_tmp19*weak_mat_tmp30 - weak_mat_tmp23*weak_mat_tmp32) + weak_mat_tmp24*weak_mat_tmp34) + C10*(weak_mat_tmp10*weak_mat_tmp33 + weak_mat_tmp17*weak_mat_tmp34)) + weak_mat_tmp8*(s_t(2)*weak_mat_tmp0*weak_mat_tmp3 - s_t(2)*weak_mat_tmp6);
    const s_t material5 = gu3*iso_history_qp[4] + gu5*iso_history_qp[2] + iso_history_qp[5]*weak_mat_tmp2 + iso_relaxation_gamma*(C01*(weak_mat_tmp18*(s_t(2)*gu5*weak_mat_tmp16 - weak_mat_tmp15*weak_mat_tmp32 - weak_mat_tmp21*weak_mat_tmp30 - weak_mat_tmp23*weak_mat_tmp33) + weak_mat_tmp24*weak_mat_tmp37) + C10*(weak_mat_tmp10*weak_mat_tmp32 + weak_mat_tmp17*weak_mat_tmp37)) + weak_mat_tmp8*(s_t(2)*weak_mat_tmp35 - s_t(2)*weak_mat_tmp36);
    const s_t material6 = gu6*iso_history_qp[0] + gu7*iso_history_qp[3] + iso_history_qp[4]*weak_mat_tmp3 + iso_relaxation_gamma*(C01*(weak_mat_tmp18*(s_t(2)*gu6*weak_mat_tmp16 - weak_mat_tmp13*weak_mat_tmp40 - weak_mat_tmp19*weak_mat_tmp42 - weak_mat_tmp21*weak_mat_tmp43) + weak_mat_tmp24*weak_mat_tmp41) + C10*(weak_mat_tmp10*weak_mat_tmp40 + weak_mat_tmp17*weak_mat_tmp41)) + weak_mat_tmp8*(s_t(2)*weak_mat_tmp38 - s_t(2)*weak_mat_tmp39);
    const s_t material7 = gu6*iso_history_qp[3] + gu7*iso_history_qp[1] + iso_history_qp[5]*weak_mat_tmp3 + iso_relaxation_gamma*(C01*(weak_mat_tmp18*(s_t(2)*gu7*weak_mat_tmp16 - weak_mat_tmp14*weak_mat_tmp42 - weak_mat_tmp19*weak_mat_tmp40 - weak_mat_tmp23*weak_mat_tmp43) + weak_mat_tmp24*weak_mat_tmp46) + C10*(weak_mat_tmp10*weak_mat_tmp42 + weak_mat_tmp17*weak_mat_tmp46)) + weak_mat_tmp8*(s_t(2)*weak_mat_tmp44 - s_t(2)*weak_mat_tmp45);
    const s_t material8 = gu6*iso_history_qp[4] + gu7*iso_history_qp[5] + iso_history_qp[2]*weak_mat_tmp3 + iso_relaxation_gamma*(C01*(weak_mat_tmp18*(-weak_mat_tmp15*weak_mat_tmp43 + s_t(2)*weak_mat_tmp16*weak_mat_tmp3 - weak_mat_tmp21*weak_mat_tmp40 - weak_mat_tmp23*weak_mat_tmp42) + weak_mat_tmp24*weak_mat_tmp47) + C10*(weak_mat_tmp10*weak_mat_tmp43 + weak_mat_tmp17*weak_mat_tmp47)) + weak_mat_tmp8*(s_t(2)*weak_mat_tmp0*weak_mat_tmp2 - s_t(2)*weak_mat_tmp5);
    const s_t loperand0 = qw * (material0 * adj_lane0 + material1 * adj_lane1 + material2 * adj_lane2);
    const s_t loperand1 = qw * (material0 * adj_lane3 + material1 * adj_lane4 + material2 * adj_lane5);
    const s_t loperand2 = qw * (material0 * adj_lane6 + material1 * adj_lane7 + material2 * adj_lane8);
    const s_t loperand3 = qw * (material3 * adj_lane0 + material4 * adj_lane1 + material5 * adj_lane2);
    const s_t loperand4 = qw * (material3 * adj_lane3 + material4 * adj_lane4 + material5 * adj_lane5);
    const s_t loperand5 = qw * (material3 * adj_lane6 + material4 * adj_lane7 + material5 * adj_lane8);
    const s_t loperand6 = qw * (material6 * adj_lane0 + material7 * adj_lane1 + material8 * adj_lane2);
    const s_t loperand7 = qw * (material6 * adj_lane3 + material7 * adj_lane4 + material8 * adj_lane5);
    const s_t loperand8 = qw * (material6 * adj_lane6 + material7 * adj_lane7 + material8 * adj_lane8);
      out_streams[0][lane] += -(loperand0) - loperand1 - loperand2;
      out_streams[1][lane] += -(loperand3) - loperand4 - loperand5;
      out_streams[2][lane] += -(loperand6) - loperand7 - loperand8;
      out_streams[3][lane] += loperand0;
      out_streams[4][lane] += loperand3;
      out_streams[5][lane] += loperand6;
      out_streams[6][lane] += loperand1;
      out_streams[7][lane] += loperand4;
      out_streams[8][lane] += loperand7;
      out_streams[9][lane] += loperand2;
      out_streams[10][lane] += loperand5;
      out_streams[11][lane] += loperand8;
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void continued_mooney_rivlin_prony_d3_simplex_apply_block(
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
        const s_t *const RSTR grad_ref_x,
        const s_t *const RSTR grad_ref_y,
        const s_t *const RSTR grad_ref_z,
        const s_t *const RSTR q_weight,
        const s_t C01,
        const s_t C10,
        const s_t Jc,
        const s_t Jmin,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const ptrdiff_t qp_stride,
        const s_t *const RSTR u_streams[NS * 3],
        const s_t *const RSTR h_streams[NS * 3],
        s_t *const RSTR out_streams[NS * 3]
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
      s_t gu_ref4_values[VS];
      s_t grad_h_ref4_values[VS];
      s_t gu_ref5_values[VS];
      s_t grad_h_ref5_values[VS];
      s_t gu_ref6_values[VS];
      s_t grad_h_ref6_values[VS];
      s_t gu_ref7_values[VS];
      s_t grad_h_ref7_values[VS];
      s_t gu_ref8_values[VS];
      s_t grad_h_ref8_values[VS];
      s_t loperand0_values[VS];
      s_t loperand1_values[VS];
      s_t loperand2_values[VS];
      s_t loperand3_values[VS];
      s_t loperand4_values[VS];
      s_t loperand5_values[VS];
      s_t loperand6_values[VS];
      s_t loperand7_values[VS];
      s_t loperand8_values[VS];
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
        gu_ref4_values[lane] = s_t(0);
        grad_h_ref4_values[lane] = s_t(0);
        gu_ref5_values[lane] = s_t(0);
        grad_h_ref5_values[lane] = s_t(0);
        gu_ref6_values[lane] = s_t(0);
        grad_h_ref6_values[lane] = s_t(0);
        gu_ref7_values[lane] = s_t(0);
        grad_h_ref7_values[lane] = s_t(0);
        gu_ref8_values[lane] = s_t(0);
        grad_h_ref8_values[lane] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t gref0 = grad_ref_x[q * NS + shape];
        const s_t gref1 = grad_ref_y[q * NS + shape];
        const s_t gref2 = grad_ref_z[q * NS + shape];
        const s_t *const RSTR u_shape0 = u_streams[3 * shape];
        const s_t *const RSTR u_shape1 = u_streams[3 * shape + 1];
        const s_t *const RSTR u_shape2 = u_streams[3 * shape + 2];
        const s_t *const RSTR h_shape0 = h_streams[3 * shape];
        const s_t *const RSTR h_shape1 = h_streams[3 * shape + 1];
        const s_t *const RSTR h_shape2 = h_streams[3 * shape + 2];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref0_values[lane] += u_shape0[lane] * gref0;
          grad_h_ref0_values[lane] += h_shape0[lane] * gref0;
          gu_ref1_values[lane] += u_shape0[lane] * gref1;
          grad_h_ref1_values[lane] += h_shape0[lane] * gref1;
          gu_ref2_values[lane] += u_shape0[lane] * gref2;
          grad_h_ref2_values[lane] += h_shape0[lane] * gref2;
          gu_ref3_values[lane] += u_shape1[lane] * gref0;
          grad_h_ref3_values[lane] += h_shape1[lane] * gref0;
          gu_ref4_values[lane] += u_shape1[lane] * gref1;
          grad_h_ref4_values[lane] += h_shape1[lane] * gref1;
          gu_ref5_values[lane] += u_shape1[lane] * gref2;
          grad_h_ref5_values[lane] += h_shape1[lane] * gref2;
          gu_ref6_values[lane] += u_shape2[lane] * gref0;
          grad_h_ref6_values[lane] += h_shape2[lane] * gref0;
          gu_ref7_values[lane] += u_shape2[lane] * gref1;
          grad_h_ref7_values[lane] += h_shape2[lane] * gref1;
          gu_ref8_values[lane] += u_shape2[lane] * gref2;
          grad_h_ref8_values[lane] += h_shape2[lane] * gref2;
        }
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
      const s_t adj_lane0 = adj0[goff];
      const s_t adj_lane1 = adj1[goff];
      const s_t adj_lane2 = adj2[goff];
      const s_t adj_lane3 = adj3[goff];
      const s_t adj_lane4 = adj4[goff];
      const s_t adj_lane5 = adj5[goff];
      const s_t adj_lane6 = adj6[goff];
      const s_t adj_lane7 = adj7[goff];
      const s_t adj_lane8 = adj8[goff];
      const s_t det_lane0 = det0[goff];
      const s_t iso_history_qp[6] = {iso_history_q[lane], iso_history_q[1 * NQ * qp_stride + lane], iso_history_q[2 * NQ * qp_stride + lane], iso_history_q[3 * NQ * qp_stride + lane], iso_history_q[4 * NQ * qp_stride + lane], iso_history_q[5 * NQ * qp_stride + lane]};
      const s_t gu_ref0 = gu_ref0_values[lane];
      const s_t grad_h_ref0 = grad_h_ref0_values[lane];
      const s_t gu_ref1 = gu_ref1_values[lane];
      const s_t grad_h_ref1 = grad_h_ref1_values[lane];
      const s_t gu_ref2 = gu_ref2_values[lane];
      const s_t grad_h_ref2 = grad_h_ref2_values[lane];
      const s_t gu_ref3 = gu_ref3_values[lane];
      const s_t grad_h_ref3 = grad_h_ref3_values[lane];
      const s_t gu_ref4 = gu_ref4_values[lane];
      const s_t grad_h_ref4 = grad_h_ref4_values[lane];
      const s_t gu_ref5 = gu_ref5_values[lane];
      const s_t grad_h_ref5 = grad_h_ref5_values[lane];
      const s_t gu_ref6 = gu_ref6_values[lane];
      const s_t grad_h_ref6 = grad_h_ref6_values[lane];
      const s_t gu_ref7 = gu_ref7_values[lane];
      const s_t grad_h_ref7 = grad_h_ref7_values[lane];
      const s_t gu_ref8 = gu_ref8_values[lane];
      const s_t grad_h_ref8 = grad_h_ref8_values[lane];
    const s_t idet = s_t(1) / det_lane0;
    const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane3 + gu_ref2 * adj_lane6) * idet;
    const s_t trial_grad0 = (grad_h_ref0 * adj_lane0 + grad_h_ref1 * adj_lane3 + grad_h_ref2 * adj_lane6) * idet;
    const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane4 + gu_ref2 * adj_lane7) * idet;
    const s_t trial_grad1 = (grad_h_ref0 * adj_lane1 + grad_h_ref1 * adj_lane4 + grad_h_ref2 * adj_lane7) * idet;
    const s_t gu2 = (gu_ref0 * adj_lane2 + gu_ref1 * adj_lane5 + gu_ref2 * adj_lane8) * idet;
    const s_t trial_grad2 = (grad_h_ref0 * adj_lane2 + grad_h_ref1 * adj_lane5 + grad_h_ref2 * adj_lane8) * idet;
    const s_t gu3 = (gu_ref3 * adj_lane0 + gu_ref4 * adj_lane3 + gu_ref5 * adj_lane6) * idet;
    const s_t trial_grad3 = (grad_h_ref3 * adj_lane0 + grad_h_ref4 * adj_lane3 + grad_h_ref5 * adj_lane6) * idet;
    const s_t gu4 = (gu_ref3 * adj_lane1 + gu_ref4 * adj_lane4 + gu_ref5 * adj_lane7) * idet;
    const s_t trial_grad4 = (grad_h_ref3 * adj_lane1 + grad_h_ref4 * adj_lane4 + grad_h_ref5 * adj_lane7) * idet;
    const s_t gu5 = (gu_ref3 * adj_lane2 + gu_ref4 * adj_lane5 + gu_ref5 * adj_lane8) * idet;
    const s_t trial_grad5 = (grad_h_ref3 * adj_lane2 + grad_h_ref4 * adj_lane5 + grad_h_ref5 * adj_lane8) * idet;
    const s_t gu6 = (gu_ref6 * adj_lane0 + gu_ref7 * adj_lane3 + gu_ref8 * adj_lane6) * idet;
    const s_t trial_grad6 = (grad_h_ref6 * adj_lane0 + grad_h_ref7 * adj_lane3 + grad_h_ref8 * adj_lane6) * idet;
    const s_t gu7 = (gu_ref6 * adj_lane1 + gu_ref7 * adj_lane4 + gu_ref8 * adj_lane7) * idet;
    const s_t trial_grad7 = (grad_h_ref6 * adj_lane1 + grad_h_ref7 * adj_lane4 + grad_h_ref8 * adj_lane7) * idet;
    const s_t gu8 = (gu_ref6 * adj_lane2 + gu_ref7 * adj_lane5 + gu_ref8 * adj_lane8) * idet;
    const s_t trial_grad8 = (grad_h_ref6 * adj_lane2 + grad_h_ref7 * adj_lane5 + grad_h_ref8 * adj_lane8) * idet;
    const s_t weak_mat_tmp0 = gu5*gu7;
    const s_t weak_mat_tmp1 = gu4 + s_t(1);
    const s_t weak_mat_tmp2 = gu8 + s_t(1);
    const s_t weak_mat_tmp3 = -weak_mat_tmp0 + weak_mat_tmp1*weak_mat_tmp2;
    const s_t weak_mat_tmp4 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp5 = -s_t(2)*weak_mat_tmp1*weak_mat_tmp2;
    const s_t weak_mat_tmp6 = ((s_t(1) / s_t(2)))*K;
    const s_t weak_mat_tmp7 = weak_mat_tmp6*(-weak_mat_tmp4 - weak_mat_tmp5);
    const s_t weak_mat_tmp8 = gu3*weak_mat_tmp2;
    const s_t weak_mat_tmp9 = gu6*weak_mat_tmp1;
    const s_t weak_mat_tmp10 = gu0 + s_t(1);
    const s_t weak_mat_tmp11 = gu1*gu5*gu6 - gu1*weak_mat_tmp8 + gu2*gu3*gu7 - gu2*weak_mat_tmp9 - weak_mat_tmp0*weak_mat_tmp10 + weak_mat_tmp1*weak_mat_tmp10*weak_mat_tmp2;
    const s_t weak_mat_tmp12 = s_t(2)*sfem_continued_power<s_t>(weak_mat_tmp11, (s_t(2) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp13 = sfem_continued_power_d1<s_t>(weak_mat_tmp11, (s_t(2) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp14 = weak_mat_tmp13*weak_mat_tmp3;
    const s_t weak_mat_tmp15 = pow_2(weak_mat_tmp3);
    const s_t weak_mat_tmp16 = pow_2(gu1);
    const s_t weak_mat_tmp17 = pow_2(gu7);
    const s_t weak_mat_tmp18 = pow_2(weak_mat_tmp1);
    const s_t weak_mat_tmp19 = weak_mat_tmp16 + weak_mat_tmp17 + weak_mat_tmp18;
    const s_t weak_mat_tmp20 = pow_2(gu2);
    const s_t weak_mat_tmp21 = pow_2(gu5);
    const s_t weak_mat_tmp22 = pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp23 = weak_mat_tmp20 + weak_mat_tmp21 + weak_mat_tmp22;
    const s_t weak_mat_tmp24 = pow_2(gu3);
    const s_t weak_mat_tmp25 = pow_2(gu6);
    const s_t weak_mat_tmp26 = pow_2(weak_mat_tmp10);
    const s_t weak_mat_tmp27 = weak_mat_tmp24 + weak_mat_tmp25 + weak_mat_tmp26;
    const s_t weak_mat_tmp28 = weak_mat_tmp19 + weak_mat_tmp23 + weak_mat_tmp27;
    const s_t weak_mat_tmp29 = weak_mat_tmp28*sfem_continued_power_d2<s_t>(weak_mat_tmp11, (s_t(2) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp30 = s_t(2)*weak_mat_tmp21;
    const s_t weak_mat_tmp31 = s_t(2)*weak_mat_tmp22;
    const s_t weak_mat_tmp32 = weak_mat_tmp30 + weak_mat_tmp31;
    const s_t weak_mat_tmp33 = s_t(2)*weak_mat_tmp17;
    const s_t weak_mat_tmp34 = s_t(2)*weak_mat_tmp18;
    const s_t weak_mat_tmp35 = weak_mat_tmp33 + weak_mat_tmp34;
    const s_t weak_mat_tmp36 = sfem_continued_power<s_t>(weak_mat_tmp11, (s_t(4) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp37 = gu6*gu7;
    const s_t weak_mat_tmp38 = gu1*weak_mat_tmp10;
    const s_t weak_mat_tmp39 = gu3*weak_mat_tmp1;
    const s_t weak_mat_tmp40 = weak_mat_tmp37 + weak_mat_tmp38 + weak_mat_tmp39;
    const s_t weak_mat_tmp41 = s_t(2)*gu1;
    const s_t weak_mat_tmp42 = gu3*gu5;
    const s_t weak_mat_tmp43 = gu2*weak_mat_tmp10;
    const s_t weak_mat_tmp44 = gu6*weak_mat_tmp2;
    const s_t weak_mat_tmp45 = weak_mat_tmp42 + weak_mat_tmp43 + weak_mat_tmp44;
    const s_t weak_mat_tmp46 = s_t(2)*gu2;
    const s_t weak_mat_tmp47 = s_t(2)*weak_mat_tmp10;
    const s_t weak_mat_tmp48 = sfem_continued_power_d1<s_t>(weak_mat_tmp11, (s_t(4) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp49 = weak_mat_tmp48*(s_t(2)*weak_mat_tmp10*weak_mat_tmp28 - weak_mat_tmp27*weak_mat_tmp47 - weak_mat_tmp40*weak_mat_tmp41 - weak_mat_tmp45*weak_mat_tmp46);
    const s_t weak_mat_tmp50 = gu1*gu2;
    const s_t weak_mat_tmp51 = gu5*weak_mat_tmp1;
    const s_t weak_mat_tmp52 = gu7*weak_mat_tmp2;
    const s_t weak_mat_tmp53 = weak_mat_tmp50 + weak_mat_tmp51 + weak_mat_tmp52;
    const s_t weak_mat_tmp54 = -(s_t(1) / s_t(2))*pow_2(weak_mat_tmp19) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp23) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp27) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp28) - pow_2(weak_mat_tmp40) - pow_2(weak_mat_tmp45) - pow_2(weak_mat_tmp53);
    const s_t weak_mat_tmp55 = weak_mat_tmp54*sfem_continued_power_d2<s_t>(weak_mat_tmp11, (s_t(4) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp56 = gu1*weak_mat_tmp2;
    const s_t weak_mat_tmp57 = gu2*gu7 - weak_mat_tmp56;
    const s_t weak_mat_tmp58 = s_t(2)*gu3;
    const s_t weak_mat_tmp59 = weak_mat_tmp13*weak_mat_tmp47;
    const s_t weak_mat_tmp60 = weak_mat_tmp3*weak_mat_tmp57;
    const s_t weak_mat_tmp61 = gu5*weak_mat_tmp46;
    const s_t weak_mat_tmp62 = weak_mat_tmp1*weak_mat_tmp41;
    const s_t weak_mat_tmp63 = s_t(2)*gu5;
    const s_t weak_mat_tmp64 = s_t(2)*weak_mat_tmp1;
    const s_t weak_mat_tmp65 = s_t(2)*gu3*weak_mat_tmp28 - weak_mat_tmp27*weak_mat_tmp58 - weak_mat_tmp40*weak_mat_tmp64 - weak_mat_tmp45*weak_mat_tmp63;
    const s_t weak_mat_tmp66 = weak_mat_tmp3*weak_mat_tmp48;
    const s_t weak_mat_tmp67 = iso_relaxation_gamma*(C01*(weak_mat_tmp36*(-weak_mat_tmp61 - weak_mat_tmp62) + weak_mat_tmp49*weak_mat_tmp57 + weak_mat_tmp55*weak_mat_tmp60 + weak_mat_tmp65*weak_mat_tmp66) + C10*(weak_mat_tmp14*weak_mat_tmp58 + weak_mat_tmp29*weak_mat_tmp60 + weak_mat_tmp57*weak_mat_tmp59));
    const s_t weak_mat_tmp68 = gu1*gu5;
    const s_t weak_mat_tmp69 = gu2*weak_mat_tmp1;
    const s_t weak_mat_tmp70 = weak_mat_tmp68 - weak_mat_tmp69;
    const s_t weak_mat_tmp71 = s_t(2)*gu6;
    const s_t weak_mat_tmp72 = weak_mat_tmp3*weak_mat_tmp70;
    const s_t weak_mat_tmp73 = gu7*weak_mat_tmp41;
    const s_t weak_mat_tmp74 = weak_mat_tmp2*weak_mat_tmp46;
    const s_t weak_mat_tmp75 = s_t(2)*gu7;
    const s_t weak_mat_tmp76 = s_t(2)*weak_mat_tmp2;
    const s_t weak_mat_tmp77 = s_t(2)*gu6*weak_mat_tmp28 - weak_mat_tmp27*weak_mat_tmp71 - weak_mat_tmp40*weak_mat_tmp75 - weak_mat_tmp45*weak_mat_tmp76;
    const s_t weak_mat_tmp78 = iso_relaxation_gamma*(C01*(weak_mat_tmp36*(-weak_mat_tmp73 - weak_mat_tmp74) + weak_mat_tmp49*weak_mat_tmp70 + weak_mat_tmp55*weak_mat_tmp72 + weak_mat_tmp66*weak_mat_tmp77) + C10*(weak_mat_tmp14*weak_mat_tmp71 + weak_mat_tmp29*weak_mat_tmp72 + weak_mat_tmp59*weak_mat_tmp70));
    const s_t weak_mat_tmp79 = gu5*gu6 - weak_mat_tmp8;
    const s_t weak_mat_tmp80 = weak_mat_tmp3*weak_mat_tmp79;
    const s_t weak_mat_tmp81 = s_t(2)*weak_mat_tmp37;
    const s_t weak_mat_tmp82 = s_t(2)*weak_mat_tmp39;
    const s_t weak_mat_tmp83 = s_t(2)*gu1*weak_mat_tmp28 - weak_mat_tmp19*weak_mat_tmp41 - weak_mat_tmp40*weak_mat_tmp47 - weak_mat_tmp46*weak_mat_tmp53;
    const s_t weak_mat_tmp84 = iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_mat_tmp36*(-weak_mat_tmp81 - weak_mat_tmp82) + weak_mat_tmp49*weak_mat_tmp79 + weak_mat_tmp55*weak_mat_tmp80 + weak_mat_tmp66*weak_mat_tmp83) + C10*(weak_mat_tmp14*weak_mat_tmp41 + weak_mat_tmp29*weak_mat_tmp80 + weak_mat_tmp59*weak_mat_tmp79));
    const s_t weak_mat_tmp85 = gu3*gu7;
    const s_t weak_mat_tmp86 = weak_mat_tmp85 - weak_mat_tmp9;
    const s_t weak_mat_tmp87 = weak_mat_tmp3*weak_mat_tmp86;
    const s_t weak_mat_tmp88 = s_t(2)*weak_mat_tmp42;
    const s_t weak_mat_tmp89 = s_t(2)*weak_mat_tmp44;
    const s_t weak_mat_tmp90 = s_t(2)*gu2*weak_mat_tmp28 - weak_mat_tmp23*weak_mat_tmp46 - weak_mat_tmp41*weak_mat_tmp53 - weak_mat_tmp45*weak_mat_tmp47;
    const s_t weak_mat_tmp91 = iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_mat_tmp36*(-weak_mat_tmp88 - weak_mat_tmp89) + weak_mat_tmp49*weak_mat_tmp86 + weak_mat_tmp55*weak_mat_tmp87 + weak_mat_tmp66*weak_mat_tmp90) + C10*(weak_mat_tmp14*weak_mat_tmp46 + weak_mat_tmp29*weak_mat_tmp87 + weak_mat_tmp59*weak_mat_tmp86));
    const s_t weak_mat_tmp92 = gu1*gu6;
    const s_t weak_mat_tmp93 = gu7*weak_mat_tmp10;
    const s_t weak_mat_tmp94 = weak_mat_tmp92 - weak_mat_tmp93;
    const s_t weak_mat_tmp95 = weak_mat_tmp13*weak_mat_tmp28;
    const s_t weak_mat_tmp96 = gu7*weak_mat_tmp95;
    const s_t weak_mat_tmp97 = weak_mat_tmp3*weak_mat_tmp94;
    const s_t weak_mat_tmp98 = gu2*gu3;
    const s_t weak_mat_tmp99 = s_t(2)*weak_mat_tmp98;
    const s_t weak_mat_tmp100 = s_t(2)*gu5*weak_mat_tmp28 - weak_mat_tmp23*weak_mat_tmp63 - weak_mat_tmp45*weak_mat_tmp58 - weak_mat_tmp53*weak_mat_tmp64;
    const s_t weak_mat_tmp101 = weak_mat_tmp48*weak_mat_tmp54;
    const s_t weak_mat_tmp102 = gu7*weak_mat_tmp101;
    const s_t weak_mat_tmp103 = gu0*gu4;
    const s_t weak_mat_tmp104 = gu1*gu3;
    const s_t weak_mat_tmp105 = gu2*gu6;
    const s_t weak_mat_tmp106 = gu5*gu6;
    const s_t weak_mat_tmp107 = K*(gu0*gu8 - gu0*weak_mat_tmp0 + gu0 + gu1*weak_mat_tmp106 + gu2*weak_mat_tmp85 + gu4*gu8 - gu4*weak_mat_tmp105 + gu4 + gu8*weak_mat_tmp103 - gu8*weak_mat_tmp104 + gu8 - weak_mat_tmp0 + weak_mat_tmp103 - weak_mat_tmp104 - weak_mat_tmp105);
    const s_t weak_mat_tmp108 = gu7*weak_mat_tmp107;
    const s_t weak_mat_tmp109 = iso_relaxation_gamma*(C01*(weak_mat_tmp100*weak_mat_tmp66 - weak_mat_tmp102 + weak_mat_tmp36*(s_t(4)*gu5*weak_mat_tmp10 - weak_mat_tmp99) + weak_mat_tmp49*weak_mat_tmp94 + weak_mat_tmp55*weak_mat_tmp97) + C10*(weak_mat_tmp14*weak_mat_tmp63 + weak_mat_tmp29*weak_mat_tmp97 + weak_mat_tmp59*weak_mat_tmp94 - weak_mat_tmp96)) - weak_mat_tmp108;
    const s_t weak_mat_tmp110 = gu5*weak_mat_tmp10;
    const s_t weak_mat_tmp111 = -weak_mat_tmp110 + weak_mat_tmp98;
    const s_t weak_mat_tmp112 = gu5*weak_mat_tmp95;
    const s_t weak_mat_tmp113 = weak_mat_tmp111*weak_mat_tmp3;
    const s_t weak_mat_tmp114 = s_t(2)*weak_mat_tmp92;
    const s_t weak_mat_tmp115 = s_t(2)*gu7*weak_mat_tmp28 - weak_mat_tmp19*weak_mat_tmp75 - weak_mat_tmp40*weak_mat_tmp71 - weak_mat_tmp53*weak_mat_tmp76;
    const s_t weak_mat_tmp116 = gu5*weak_mat_tmp101;
    const s_t weak_mat_tmp117 = gu5*weak_mat_tmp107;
    const s_t weak_mat_tmp118 = iso_relaxation_gamma*(C01*(weak_mat_tmp111*weak_mat_tmp49 + weak_mat_tmp113*weak_mat_tmp55 + weak_mat_tmp115*weak_mat_tmp66 - weak_mat_tmp116 + weak_mat_tmp36*(s_t(4)*gu7*weak_mat_tmp10 - weak_mat_tmp114)) + C10*(weak_mat_tmp111*weak_mat_tmp59 - weak_mat_tmp112 + weak_mat_tmp113*weak_mat_tmp29 + weak_mat_tmp14*weak_mat_tmp75)) - weak_mat_tmp117;
    const s_t weak_mat_tmp119 = weak_mat_tmp10*weak_mat_tmp2 - weak_mat_tmp105;
    const s_t weak_mat_tmp120 = weak_mat_tmp2*weak_mat_tmp95;
    const s_t weak_mat_tmp121 = weak_mat_tmp119*weak_mat_tmp3;
    const s_t weak_mat_tmp122 = s_t(2)*weak_mat_tmp104;
    const s_t weak_mat_tmp123 = s_t(2)*weak_mat_tmp1*weak_mat_tmp28 - weak_mat_tmp19*weak_mat_tmp64 - weak_mat_tmp40*weak_mat_tmp58 - weak_mat_tmp53*weak_mat_tmp63;
    const s_t weak_mat_tmp124 = weak_mat_tmp101*weak_mat_tmp2;
    const s_t weak_mat_tmp125 = weak_mat_tmp107*weak_mat_tmp2;
    const s_t weak_mat_tmp126 = iso_relaxation_gamma*(C01*(weak_mat_tmp119*weak_mat_tmp49 + weak_mat_tmp121*weak_mat_tmp55 + weak_mat_tmp123*weak_mat_tmp66 + weak_mat_tmp124 + weak_mat_tmp36*(s_t(4)*weak_mat_tmp1*weak_mat_tmp10 - weak_mat_tmp122)) + C10*(weak_mat_tmp119*weak_mat_tmp59 + weak_mat_tmp120 + weak_mat_tmp121*weak_mat_tmp29 + weak_mat_tmp14*weak_mat_tmp64)) + weak_mat_tmp125;
    const s_t weak_mat_tmp127 = weak_mat_tmp1*weak_mat_tmp10 - weak_mat_tmp104;
    const s_t weak_mat_tmp128 = weak_mat_tmp1*weak_mat_tmp95;
    const s_t weak_mat_tmp129 = weak_mat_tmp127*weak_mat_tmp3;
    const s_t weak_mat_tmp130 = s_t(2)*weak_mat_tmp105;
    const s_t weak_mat_tmp131 = s_t(2)*weak_mat_tmp2*weak_mat_tmp28 - weak_mat_tmp23*weak_mat_tmp76 - weak_mat_tmp45*weak_mat_tmp71 - weak_mat_tmp53*weak_mat_tmp75;
    const s_t weak_mat_tmp132 = weak_mat_tmp1*weak_mat_tmp101;
    const s_t weak_mat_tmp133 = weak_mat_tmp1*weak_mat_tmp107;
    const s_t weak_mat_tmp134 = iso_relaxation_gamma*(C01*(weak_mat_tmp127*weak_mat_tmp49 + weak_mat_tmp129*weak_mat_tmp55 + weak_mat_tmp131*weak_mat_tmp66 + weak_mat_tmp132 + weak_mat_tmp36*(s_t(4)*weak_mat_tmp10*weak_mat_tmp2 - weak_mat_tmp130)) + C10*(weak_mat_tmp127*weak_mat_tmp59 + weak_mat_tmp128 + weak_mat_tmp129*weak_mat_tmp29 + weak_mat_tmp14*weak_mat_tmp76)) + weak_mat_tmp133;
    const s_t weak_mat_tmp135 = -s_t(2)*weak_mat_tmp106;
    const s_t weak_mat_tmp136 = s_t(2)*weak_mat_tmp8;
    const s_t weak_mat_tmp137 = weak_mat_tmp6*(-weak_mat_tmp135 - weak_mat_tmp136);
    const s_t weak_mat_tmp138 = weak_mat_tmp13*weak_mat_tmp79;
    const s_t weak_mat_tmp139 = pow_2(weak_mat_tmp79);
    const s_t weak_mat_tmp140 = s_t(2)*weak_mat_tmp24;
    const s_t weak_mat_tmp141 = s_t(2)*weak_mat_tmp25;
    const s_t weak_mat_tmp142 = weak_mat_tmp140 + weak_mat_tmp141;
    const s_t weak_mat_tmp143 = weak_mat_tmp48*weak_mat_tmp83;
    const s_t weak_mat_tmp144 = weak_mat_tmp13*weak_mat_tmp41;
    const s_t weak_mat_tmp145 = weak_mat_tmp111*weak_mat_tmp79;
    const s_t weak_mat_tmp146 = gu6*weak_mat_tmp47;
    const s_t weak_mat_tmp147 = weak_mat_tmp48*weak_mat_tmp79;
    const s_t weak_mat_tmp148 = iso_relaxation_gamma*(C01*(weak_mat_tmp111*weak_mat_tmp143 + weak_mat_tmp115*weak_mat_tmp147 + weak_mat_tmp145*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp146 - weak_mat_tmp74)) + C10*(weak_mat_tmp111*weak_mat_tmp144 + weak_mat_tmp138*weak_mat_tmp75 + weak_mat_tmp145*weak_mat_tmp29));
    const s_t weak_mat_tmp149 = weak_mat_tmp79*weak_mat_tmp86;
    const s_t weak_mat_tmp150 = s_t(2)*weak_mat_tmp51;
    const s_t weak_mat_tmp151 = s_t(2)*weak_mat_tmp52;
    const s_t weak_mat_tmp152 = iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_mat_tmp143*weak_mat_tmp86 + weak_mat_tmp147*weak_mat_tmp90 + weak_mat_tmp149*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp150 - weak_mat_tmp151)) + C10*(weak_mat_tmp138*weak_mat_tmp46 + weak_mat_tmp144*weak_mat_tmp86 + weak_mat_tmp149*weak_mat_tmp29));
    const s_t weak_mat_tmp153 = weak_mat_tmp119*weak_mat_tmp79;
    const s_t weak_mat_tmp154 = gu3*weak_mat_tmp47;
    const s_t weak_mat_tmp155 = iso_relaxation_gamma*(C01*(weak_mat_tmp119*weak_mat_tmp143 + weak_mat_tmp123*weak_mat_tmp147 + weak_mat_tmp153*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp154 - weak_mat_tmp61)) + C10*(weak_mat_tmp119*weak_mat_tmp144 + weak_mat_tmp138*weak_mat_tmp64 + weak_mat_tmp153*weak_mat_tmp29));
    const s_t weak_mat_tmp156 = gu6*weak_mat_tmp95;
    const s_t weak_mat_tmp157 = weak_mat_tmp79*weak_mat_tmp94;
    const s_t weak_mat_tmp158 = -s_t(2)*weak_mat_tmp69;
    const s_t weak_mat_tmp159 = gu6*weak_mat_tmp101;
    const s_t weak_mat_tmp160 = gu6*weak_mat_tmp107;
    const s_t weak_mat_tmp161 = iso_relaxation_gamma*(C01*(weak_mat_tmp100*weak_mat_tmp147 + weak_mat_tmp143*weak_mat_tmp94 + weak_mat_tmp157*weak_mat_tmp55 + weak_mat_tmp159 + weak_mat_tmp36*(weak_mat_tmp158 + s_t(4)*weak_mat_tmp68)) + C10*(weak_mat_tmp138*weak_mat_tmp63 + weak_mat_tmp144*weak_mat_tmp94 + weak_mat_tmp156 + weak_mat_tmp157*weak_mat_tmp29)) + weak_mat_tmp160;
    const s_t weak_mat_tmp162 = weak_mat_tmp70*weak_mat_tmp79;
    const s_t weak_mat_tmp163 = -s_t(2)*weak_mat_tmp93;
    const s_t weak_mat_tmp164 = iso_relaxation_gamma*(C01*(weak_mat_tmp116 + weak_mat_tmp143*weak_mat_tmp70 + weak_mat_tmp147*weak_mat_tmp77 + weak_mat_tmp162*weak_mat_tmp55 + weak_mat_tmp36*(weak_mat_tmp163 + s_t(4)*weak_mat_tmp92)) + C10*(weak_mat_tmp112 + weak_mat_tmp138*weak_mat_tmp71 + weak_mat_tmp144*weak_mat_tmp70 + weak_mat_tmp162*weak_mat_tmp29)) + weak_mat_tmp117;
    const s_t weak_mat_tmp165 = weak_mat_tmp57*weak_mat_tmp79;
    const s_t weak_mat_tmp166 = -s_t(2)*weak_mat_tmp1*weak_mat_tmp10;
    const s_t weak_mat_tmp167 = iso_relaxation_gamma*(C01*(-weak_mat_tmp124 + weak_mat_tmp143*weak_mat_tmp57 + weak_mat_tmp147*weak_mat_tmp65 + weak_mat_tmp165*weak_mat_tmp55 + weak_mat_tmp36*(s_t(4)*weak_mat_tmp104 + weak_mat_tmp166)) + C10*(-weak_mat_tmp120 + weak_mat_tmp138*weak_mat_tmp58 + weak_mat_tmp144*weak_mat_tmp57 + weak_mat_tmp165*weak_mat_tmp29)) - weak_mat_tmp125;
    const s_t weak_mat_tmp168 = gu3*weak_mat_tmp95;
    const s_t weak_mat_tmp169 = weak_mat_tmp127*weak_mat_tmp79;
    const s_t weak_mat_tmp170 = -s_t(2)*gu2*gu7;
    const s_t weak_mat_tmp171 = gu3*weak_mat_tmp101;
    const s_t weak_mat_tmp172 = gu3*weak_mat_tmp107;
    const s_t weak_mat_tmp173 = iso_relaxation_gamma*(C01*(weak_mat_tmp127*weak_mat_tmp143 + weak_mat_tmp131*weak_mat_tmp147 + weak_mat_tmp169*weak_mat_tmp55 - weak_mat_tmp171 + weak_mat_tmp36*(weak_mat_tmp170 + s_t(4)*weak_mat_tmp56)) + C10*(weak_mat_tmp127*weak_mat_tmp144 + weak_mat_tmp138*weak_mat_tmp76 - weak_mat_tmp168 + weak_mat_tmp169*weak_mat_tmp29)) - weak_mat_tmp172;
    const s_t weak_mat_tmp174 = s_t(2)*weak_mat_tmp85;
    const s_t weak_mat_tmp175 = -s_t(2)*weak_mat_tmp9;
    const s_t weak_mat_tmp176 = weak_mat_tmp6*(weak_mat_tmp174 + weak_mat_tmp175);
    const s_t weak_mat_tmp177 = weak_mat_tmp13*weak_mat_tmp86;
    const s_t weak_mat_tmp178 = pow_2(weak_mat_tmp86);
    const s_t weak_mat_tmp179 = weak_mat_tmp48*weak_mat_tmp90;
    const s_t weak_mat_tmp180 = weak_mat_tmp13*weak_mat_tmp46;
    const s_t weak_mat_tmp181 = weak_mat_tmp86*weak_mat_tmp94;
    const s_t weak_mat_tmp182 = weak_mat_tmp48*weak_mat_tmp86;
    const s_t weak_mat_tmp183 = iso_relaxation_gamma*(C01*(weak_mat_tmp100*weak_mat_tmp182 + weak_mat_tmp179*weak_mat_tmp94 + weak_mat_tmp181*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp154 - weak_mat_tmp62)) + C10*(weak_mat_tmp177*weak_mat_tmp63 + weak_mat_tmp180*weak_mat_tmp94 + weak_mat_tmp181*weak_mat_tmp29));
    const s_t weak_mat_tmp184 = weak_mat_tmp127*weak_mat_tmp86;
    const s_t weak_mat_tmp185 = iso_relaxation_gamma*(C01*(weak_mat_tmp127*weak_mat_tmp179 + weak_mat_tmp131*weak_mat_tmp182 + weak_mat_tmp184*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp146 - weak_mat_tmp73)) + C10*(weak_mat_tmp127*weak_mat_tmp180 + weak_mat_tmp177*weak_mat_tmp76 + weak_mat_tmp184*weak_mat_tmp29));
    const s_t weak_mat_tmp186 = weak_mat_tmp57*weak_mat_tmp86;
    const s_t weak_mat_tmp187 = -s_t(2)*weak_mat_tmp110;
    const s_t weak_mat_tmp188 = iso_relaxation_gamma*(C01*(weak_mat_tmp102 + weak_mat_tmp179*weak_mat_tmp57 + weak_mat_tmp182*weak_mat_tmp65 + weak_mat_tmp186*weak_mat_tmp55 + weak_mat_tmp36*(weak_mat_tmp187 + s_t(4)*weak_mat_tmp98)) + C10*(weak_mat_tmp177*weak_mat_tmp58 + weak_mat_tmp180*weak_mat_tmp57 + weak_mat_tmp186*weak_mat_tmp29 + weak_mat_tmp96)) + weak_mat_tmp108;
    const s_t weak_mat_tmp189 = weak_mat_tmp111*weak_mat_tmp86;
    const s_t weak_mat_tmp190 = s_t(2)*weak_mat_tmp56;
    const s_t weak_mat_tmp191 = iso_relaxation_gamma*(C01*(weak_mat_tmp111*weak_mat_tmp179 + weak_mat_tmp115*weak_mat_tmp182 + weak_mat_tmp171 + weak_mat_tmp189*weak_mat_tmp55 + weak_mat_tmp36*(s_t(4)*gu2*gu7 - weak_mat_tmp190)) + C10*(weak_mat_tmp111*weak_mat_tmp180 + weak_mat_tmp168 + weak_mat_tmp177*weak_mat_tmp75 + weak_mat_tmp189*weak_mat_tmp29)) + weak_mat_tmp172;
    const s_t weak_mat_tmp192 = weak_mat_tmp70*weak_mat_tmp86;
    const s_t weak_mat_tmp193 = -s_t(2)*weak_mat_tmp10*weak_mat_tmp2;
    const s_t weak_mat_tmp194 = iso_relaxation_gamma*(C01*(-weak_mat_tmp132 + weak_mat_tmp179*weak_mat_tmp70 + weak_mat_tmp182*weak_mat_tmp77 + weak_mat_tmp192*weak_mat_tmp55 + weak_mat_tmp36*(s_t(4)*weak_mat_tmp105 + weak_mat_tmp193)) + C10*(-weak_mat_tmp128 + weak_mat_tmp177*weak_mat_tmp71 + weak_mat_tmp180*weak_mat_tmp70 + weak_mat_tmp192*weak_mat_tmp29)) - weak_mat_tmp133;
    const s_t weak_mat_tmp195 = weak_mat_tmp119*weak_mat_tmp86;
    const s_t weak_mat_tmp196 = s_t(2)*weak_mat_tmp68;
    const s_t weak_mat_tmp197 = iso_relaxation_gamma*(C01*(weak_mat_tmp119*weak_mat_tmp179 + weak_mat_tmp123*weak_mat_tmp182 - weak_mat_tmp159 + weak_mat_tmp195*weak_mat_tmp55 + weak_mat_tmp36*(s_t(4)*gu2*weak_mat_tmp1 - weak_mat_tmp196)) + C10*(weak_mat_tmp119*weak_mat_tmp180 - weak_mat_tmp156 + weak_mat_tmp177*weak_mat_tmp64 + weak_mat_tmp195*weak_mat_tmp29)) - weak_mat_tmp160;
    const s_t weak_mat_tmp198 = weak_mat_tmp6*(-weak_mat_tmp170 - weak_mat_tmp190);
    const s_t weak_mat_tmp199 = weak_mat_tmp13*weak_mat_tmp57;
    const s_t weak_mat_tmp200 = pow_2(weak_mat_tmp57);
    const s_t weak_mat_tmp201 = s_t(2)*weak_mat_tmp20;
    const s_t weak_mat_tmp202 = weak_mat_tmp201 + weak_mat_tmp31;
    const s_t weak_mat_tmp203 = s_t(2)*weak_mat_tmp16;
    const s_t weak_mat_tmp204 = weak_mat_tmp203 + weak_mat_tmp33;
    const s_t weak_mat_tmp205 = weak_mat_tmp48*weak_mat_tmp57;
    const s_t weak_mat_tmp206 = weak_mat_tmp13*weak_mat_tmp58;
    const s_t weak_mat_tmp207 = weak_mat_tmp57*weak_mat_tmp70;
    const s_t weak_mat_tmp208 = weak_mat_tmp2*weak_mat_tmp63;
    const s_t weak_mat_tmp209 = gu7*weak_mat_tmp64;
    const s_t weak_mat_tmp210 = weak_mat_tmp48*weak_mat_tmp65;
    const s_t weak_mat_tmp211 = iso_relaxation_gamma*(C01*(weak_mat_tmp205*weak_mat_tmp77 + weak_mat_tmp207*weak_mat_tmp55 + weak_mat_tmp210*weak_mat_tmp70 + weak_mat_tmp36*(-weak_mat_tmp208 - weak_mat_tmp209)) + C10*(weak_mat_tmp199*weak_mat_tmp71 + weak_mat_tmp206*weak_mat_tmp70 + weak_mat_tmp207*weak_mat_tmp29));
    const s_t weak_mat_tmp212 = weak_mat_tmp57*weak_mat_tmp94;
    const s_t weak_mat_tmp213 = s_t(2)*weak_mat_tmp43;
    const s_t weak_mat_tmp214 = iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_mat_tmp100*weak_mat_tmp205 + weak_mat_tmp210*weak_mat_tmp94 + weak_mat_tmp212*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp213 - weak_mat_tmp89)) + C10*(weak_mat_tmp199*weak_mat_tmp63 + weak_mat_tmp206*weak_mat_tmp94 + weak_mat_tmp212*weak_mat_tmp29));
    const s_t weak_mat_tmp215 = weak_mat_tmp119*weak_mat_tmp57;
    const s_t weak_mat_tmp216 = s_t(2)*weak_mat_tmp38;
    const s_t weak_mat_tmp217 = iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_mat_tmp119*weak_mat_tmp210 + weak_mat_tmp123*weak_mat_tmp205 + weak_mat_tmp215*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp216 - weak_mat_tmp81)) + C10*(weak_mat_tmp119*weak_mat_tmp206 + weak_mat_tmp199*weak_mat_tmp64 + weak_mat_tmp215*weak_mat_tmp29));
    const s_t weak_mat_tmp218 = gu2*weak_mat_tmp95;
    const s_t weak_mat_tmp219 = weak_mat_tmp111*weak_mat_tmp57;
    const s_t weak_mat_tmp220 = gu2*weak_mat_tmp101;
    const s_t weak_mat_tmp221 = gu2*weak_mat_tmp107;
    const s_t weak_mat_tmp222 = iso_relaxation_gamma*(C01*(weak_mat_tmp111*weak_mat_tmp210 + weak_mat_tmp115*weak_mat_tmp205 + weak_mat_tmp219*weak_mat_tmp55 + weak_mat_tmp220 + weak_mat_tmp36*(weak_mat_tmp175 + s_t(4)*weak_mat_tmp85)) + C10*(weak_mat_tmp111*weak_mat_tmp206 + weak_mat_tmp199*weak_mat_tmp75 + weak_mat_tmp218 + weak_mat_tmp219*weak_mat_tmp29)) + weak_mat_tmp221;
    const s_t weak_mat_tmp223 = gu1*weak_mat_tmp95;
    const s_t weak_mat_tmp224 = weak_mat_tmp127*weak_mat_tmp57;
    const s_t weak_mat_tmp225 = gu1*weak_mat_tmp101;
    const s_t weak_mat_tmp226 = gu1*weak_mat_tmp107;
    const s_t weak_mat_tmp227 = iso_relaxation_gamma*(C01*(weak_mat_tmp127*weak_mat_tmp210 + weak_mat_tmp131*weak_mat_tmp205 + weak_mat_tmp224*weak_mat_tmp55 - weak_mat_tmp225 + weak_mat_tmp36*(weak_mat_tmp135 + s_t(4)*weak_mat_tmp8)) + C10*(weak_mat_tmp127*weak_mat_tmp206 + weak_mat_tmp199*weak_mat_tmp76 - weak_mat_tmp223 + weak_mat_tmp224*weak_mat_tmp29)) - weak_mat_tmp226;
    const s_t weak_mat_tmp228 = weak_mat_tmp6*(-weak_mat_tmp130 - weak_mat_tmp193);
    const s_t weak_mat_tmp229 = weak_mat_tmp119*weak_mat_tmp13;
    const s_t weak_mat_tmp230 = pow_2(weak_mat_tmp119);
    const s_t weak_mat_tmp231 = s_t(2)*weak_mat_tmp26;
    const s_t weak_mat_tmp232 = weak_mat_tmp141 + weak_mat_tmp231;
    const s_t weak_mat_tmp233 = weak_mat_tmp123*weak_mat_tmp48;
    const s_t weak_mat_tmp234 = weak_mat_tmp13*weak_mat_tmp64;
    const s_t weak_mat_tmp235 = weak_mat_tmp111*weak_mat_tmp119;
    const s_t weak_mat_tmp236 = gu6*weak_mat_tmp58;
    const s_t weak_mat_tmp237 = weak_mat_tmp119*weak_mat_tmp48;
    const s_t weak_mat_tmp238 = iso_relaxation_gamma*(C01*(weak_mat_tmp111*weak_mat_tmp233 + weak_mat_tmp115*weak_mat_tmp237 + weak_mat_tmp235*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp208 - weak_mat_tmp236)) + C10*(weak_mat_tmp111*weak_mat_tmp234 + weak_mat_tmp229*weak_mat_tmp75 + weak_mat_tmp235*weak_mat_tmp29));
    const s_t weak_mat_tmp239 = weak_mat_tmp119*weak_mat_tmp94;
    const s_t weak_mat_tmp240 = s_t(2)*weak_mat_tmp50;
    const s_t weak_mat_tmp241 = iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_mat_tmp100*weak_mat_tmp237 + weak_mat_tmp233*weak_mat_tmp94 + weak_mat_tmp239*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp151 - weak_mat_tmp240)) + C10*(weak_mat_tmp229*weak_mat_tmp63 + weak_mat_tmp234*weak_mat_tmp94 + weak_mat_tmp239*weak_mat_tmp29));
    const s_t weak_mat_tmp242 = weak_mat_tmp119*weak_mat_tmp70;
    const s_t weak_mat_tmp243 = iso_relaxation_gamma*(C01*(-weak_mat_tmp220 + weak_mat_tmp233*weak_mat_tmp70 + weak_mat_tmp237*weak_mat_tmp77 + weak_mat_tmp242*weak_mat_tmp55 + weak_mat_tmp36*(s_t(4)*gu6*weak_mat_tmp1 - weak_mat_tmp174)) + C10*(-weak_mat_tmp218 + weak_mat_tmp229*weak_mat_tmp71 + weak_mat_tmp234*weak_mat_tmp70 + weak_mat_tmp242*weak_mat_tmp29)) - weak_mat_tmp221;
    const s_t weak_mat_tmp244 = weak_mat_tmp10*weak_mat_tmp95;
    const s_t weak_mat_tmp245 = weak_mat_tmp119*weak_mat_tmp127;
    const s_t weak_mat_tmp246 = weak_mat_tmp10*weak_mat_tmp101;
    const s_t weak_mat_tmp247 = weak_mat_tmp10*weak_mat_tmp107;
    const s_t weak_mat_tmp248 = iso_relaxation_gamma*(C01*(weak_mat_tmp127*weak_mat_tmp233 + weak_mat_tmp131*weak_mat_tmp237 + weak_mat_tmp245*weak_mat_tmp55 + weak_mat_tmp246 + weak_mat_tmp36*(s_t(4)*weak_mat_tmp1*weak_mat_tmp2 - weak_mat_tmp4)) + C10*(weak_mat_tmp127*weak_mat_tmp234 + weak_mat_tmp229*weak_mat_tmp76 + weak_mat_tmp244 + weak_mat_tmp245*weak_mat_tmp29)) + weak_mat_tmp247;
    const s_t weak_mat_tmp249 = weak_mat_tmp6*(weak_mat_tmp114 + weak_mat_tmp163);
    const s_t weak_mat_tmp250 = weak_mat_tmp13*weak_mat_tmp94;
    const s_t weak_mat_tmp251 = pow_2(weak_mat_tmp94);
    const s_t weak_mat_tmp252 = weak_mat_tmp100*weak_mat_tmp48;
    const s_t weak_mat_tmp253 = weak_mat_tmp13*weak_mat_tmp63;
    const s_t weak_mat_tmp254 = weak_mat_tmp127*weak_mat_tmp94;
    const s_t weak_mat_tmp255 = weak_mat_tmp48*weak_mat_tmp94;
    const s_t weak_mat_tmp256 = iso_relaxation_gamma*(C01*(weak_mat_tmp127*weak_mat_tmp252 + weak_mat_tmp131*weak_mat_tmp255 + weak_mat_tmp254*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp209 - weak_mat_tmp236)) + C10*(weak_mat_tmp127*weak_mat_tmp253 + weak_mat_tmp250*weak_mat_tmp76 + weak_mat_tmp254*weak_mat_tmp29));
    const s_t weak_mat_tmp257 = weak_mat_tmp70*weak_mat_tmp94;
    const s_t weak_mat_tmp258 = iso_relaxation_gamma*(C01*(weak_mat_tmp225 + weak_mat_tmp252*weak_mat_tmp70 + weak_mat_tmp255*weak_mat_tmp77 + weak_mat_tmp257*weak_mat_tmp55 + weak_mat_tmp36*(s_t(4)*gu5*gu6 - weak_mat_tmp136)) + C10*(weak_mat_tmp223 + weak_mat_tmp250*weak_mat_tmp71 + weak_mat_tmp253*weak_mat_tmp70 + weak_mat_tmp257*weak_mat_tmp29)) + weak_mat_tmp226;
    const s_t weak_mat_tmp259 = weak_mat_tmp111*weak_mat_tmp94;
    const s_t weak_mat_tmp260 = iso_relaxation_gamma*(C01*(weak_mat_tmp111*weak_mat_tmp252 + weak_mat_tmp115*weak_mat_tmp255 - weak_mat_tmp246 + weak_mat_tmp259*weak_mat_tmp55 + weak_mat_tmp36*(s_t(4)*weak_mat_tmp0 + weak_mat_tmp5)) + C10*(weak_mat_tmp111*weak_mat_tmp253 - weak_mat_tmp244 + weak_mat_tmp250*weak_mat_tmp75 + weak_mat_tmp259*weak_mat_tmp29)) - weak_mat_tmp247;
    const s_t weak_mat_tmp261 = weak_mat_tmp6*(weak_mat_tmp158 + weak_mat_tmp196);
    const s_t weak_mat_tmp262 = weak_mat_tmp13*weak_mat_tmp70;
    const s_t weak_mat_tmp263 = pow_2(weak_mat_tmp70);
    const s_t weak_mat_tmp264 = weak_mat_tmp201 + weak_mat_tmp30;
    const s_t weak_mat_tmp265 = weak_mat_tmp203 + weak_mat_tmp34;
    const s_t weak_mat_tmp266 = weak_mat_tmp48*weak_mat_tmp70;
    const s_t weak_mat_tmp267 = weak_mat_tmp13*weak_mat_tmp71;
    const s_t weak_mat_tmp268 = weak_mat_tmp111*weak_mat_tmp70;
    const s_t weak_mat_tmp269 = weak_mat_tmp48*weak_mat_tmp77;
    const s_t weak_mat_tmp270 = iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_mat_tmp111*weak_mat_tmp269 + weak_mat_tmp115*weak_mat_tmp266 + weak_mat_tmp268*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp216 - weak_mat_tmp82)) + C10*(weak_mat_tmp111*weak_mat_tmp267 + weak_mat_tmp262*weak_mat_tmp75 + weak_mat_tmp268*weak_mat_tmp29));
    const s_t weak_mat_tmp271 = weak_mat_tmp127*weak_mat_tmp70;
    const s_t weak_mat_tmp272 = iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_mat_tmp127*weak_mat_tmp269 + weak_mat_tmp131*weak_mat_tmp266 + weak_mat_tmp271*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp213 - weak_mat_tmp88)) + C10*(weak_mat_tmp127*weak_mat_tmp267 + weak_mat_tmp262*weak_mat_tmp76 + weak_mat_tmp271*weak_mat_tmp29));
    const s_t weak_mat_tmp273 = weak_mat_tmp6*(weak_mat_tmp187 + weak_mat_tmp99);
    const s_t weak_mat_tmp274 = weak_mat_tmp111*weak_mat_tmp13;
    const s_t weak_mat_tmp275 = pow_2(weak_mat_tmp111);
    const s_t weak_mat_tmp276 = weak_mat_tmp140 + weak_mat_tmp231;
    const s_t weak_mat_tmp277 = weak_mat_tmp111*weak_mat_tmp48;
    const s_t weak_mat_tmp278 = weak_mat_tmp127*weak_mat_tmp13;
    const s_t weak_mat_tmp279 = weak_mat_tmp111*weak_mat_tmp127;
    const s_t weak_mat_tmp280 = weak_mat_tmp127*weak_mat_tmp48;
    const s_t weak_mat_tmp281 = iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_mat_tmp115*weak_mat_tmp280 + weak_mat_tmp131*weak_mat_tmp277 + weak_mat_tmp279*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp150 - weak_mat_tmp240)) + C10*(weak_mat_tmp274*weak_mat_tmp76 + weak_mat_tmp278*weak_mat_tmp75 + weak_mat_tmp279*weak_mat_tmp29));
    const s_t weak_mat_tmp282 = weak_mat_tmp6*(-weak_mat_tmp122 - weak_mat_tmp166);
    const s_t weak_mat_tmp283 = pow_2(weak_mat_tmp127);
    const s_t material0 = trial_grad0*(iso_history_qp[0] + iso_relaxation_gamma*(C01*(weak_mat_tmp15*weak_mat_tmp55 + s_t(2)*weak_mat_tmp3*weak_mat_tmp49 + weak_mat_tmp36*(weak_mat_tmp32 + weak_mat_tmp35)) + C10*(s_t(4)*weak_mat_tmp10*weak_mat_tmp14 + weak_mat_tmp12 + weak_mat_tmp15*weak_mat_tmp29)) + weak_mat_tmp3*weak_mat_tmp7) + trial_grad1*(weak_mat_tmp7*weak_mat_tmp79 + weak_mat_tmp84) + trial_grad2*(weak_mat_tmp7*weak_mat_tmp86 + weak_mat_tmp91) + trial_grad3*(weak_mat_tmp57*weak_mat_tmp7 + weak_mat_tmp67) + trial_grad4*(weak_mat_tmp119*weak_mat_tmp7 + weak_mat_tmp126) + trial_grad5*(weak_mat_tmp109 + weak_mat_tmp7*weak_mat_tmp94) + trial_grad6*(weak_mat_tmp7*weak_mat_tmp70 + weak_mat_tmp78) + trial_grad7*(weak_mat_tmp111*weak_mat_tmp7 + weak_mat_tmp118) + trial_grad8*(weak_mat_tmp127*weak_mat_tmp7 + weak_mat_tmp134);
    const s_t material1 = trial_grad0*(weak_mat_tmp137*weak_mat_tmp3 + weak_mat_tmp84) + trial_grad1*(iso_history_qp[1] + iso_relaxation_gamma*(C01*(weak_mat_tmp139*weak_mat_tmp55 + s_t(2)*weak_mat_tmp143*weak_mat_tmp79 + weak_mat_tmp36*(weak_mat_tmp142 + weak_mat_tmp32)) + C10*(s_t(4)*gu1*weak_mat_tmp138 + weak_mat_tmp12 + weak_mat_tmp139*weak_mat_tmp29)) + weak_mat_tmp137*weak_mat_tmp79) + trial_grad2*(weak_mat_tmp137*weak_mat_tmp86 + weak_mat_tmp152) + trial_grad3*(weak_mat_tmp137*weak_mat_tmp57 + weak_mat_tmp167) + trial_grad4*(weak_mat_tmp119*weak_mat_tmp137 + weak_mat_tmp155) + trial_grad5*(weak_mat_tmp137*weak_mat_tmp94 + weak_mat_tmp161) + trial_grad6*(weak_mat_tmp137*weak_mat_tmp70 + weak_mat_tmp164) + trial_grad7*(weak_mat_tmp111*weak_mat_tmp137 + weak_mat_tmp148) + trial_grad8*(weak_mat_tmp127*weak_mat_tmp137 + weak_mat_tmp173);
    const s_t material2 = trial_grad0*(weak_mat_tmp176*weak_mat_tmp3 + weak_mat_tmp91) + trial_grad1*(weak_mat_tmp152 + weak_mat_tmp176*weak_mat_tmp79) + trial_grad2*(iso_history_qp[2] + iso_relaxation_gamma*(C01*(weak_mat_tmp178*weak_mat_tmp55 + s_t(2)*weak_mat_tmp179*weak_mat_tmp86 + weak_mat_tmp36*(weak_mat_tmp142 + weak_mat_tmp35)) + C10*(s_t(4)*gu2*weak_mat_tmp177 + weak_mat_tmp12 + weak_mat_tmp178*weak_mat_tmp29)) + weak_mat_tmp176*weak_mat_tmp86) + trial_grad3*(weak_mat_tmp176*weak_mat_tmp57 + weak_mat_tmp188) + trial_grad4*(weak_mat_tmp119*weak_mat_tmp176 + weak_mat_tmp197) + trial_grad5*(weak_mat_tmp176*weak_mat_tmp94 + weak_mat_tmp183) + trial_grad6*(weak_mat_tmp176*weak_mat_tmp70 + weak_mat_tmp194) + trial_grad7*(weak_mat_tmp111*weak_mat_tmp176 + weak_mat_tmp191) + trial_grad8*(weak_mat_tmp127*weak_mat_tmp176 + weak_mat_tmp185);
    const s_t material3 = trial_grad0*(weak_mat_tmp198*weak_mat_tmp3 + weak_mat_tmp67) + trial_grad1*(weak_mat_tmp167 + weak_mat_tmp198*weak_mat_tmp79) + trial_grad2*(weak_mat_tmp188 + weak_mat_tmp198*weak_mat_tmp86) + trial_grad3*(iso_history_qp[0] + iso_relaxation_gamma*(C01*(weak_mat_tmp200*weak_mat_tmp55 + s_t(2)*weak_mat_tmp205*weak_mat_tmp65 + weak_mat_tmp36*(weak_mat_tmp202 + weak_mat_tmp204)) + C10*(s_t(4)*gu3*weak_mat_tmp199 + weak_mat_tmp12 + weak_mat_tmp200*weak_mat_tmp29)) + weak_mat_tmp198*weak_mat_tmp57) + trial_grad4*(weak_mat_tmp119*weak_mat_tmp198 + weak_mat_tmp217) + trial_grad5*(weak_mat_tmp198*weak_mat_tmp94 + weak_mat_tmp214) + trial_grad6*(weak_mat_tmp198*weak_mat_tmp70 + weak_mat_tmp211) + trial_grad7*(weak_mat_tmp111*weak_mat_tmp198 + weak_mat_tmp222) + trial_grad8*(weak_mat_tmp127*weak_mat_tmp198 + weak_mat_tmp227);
    const s_t material4 = trial_grad0*(weak_mat_tmp126 + weak_mat_tmp228*weak_mat_tmp3) + trial_grad1*(weak_mat_tmp155 + weak_mat_tmp228*weak_mat_tmp79) + trial_grad2*(weak_mat_tmp197 + weak_mat_tmp228*weak_mat_tmp86) + trial_grad3*(weak_mat_tmp217 + weak_mat_tmp228*weak_mat_tmp57) + trial_grad4*(iso_history_qp[1] + iso_relaxation_gamma*(C01*(s_t(2)*weak_mat_tmp119*weak_mat_tmp233 + weak_mat_tmp230*weak_mat_tmp55 + weak_mat_tmp36*(weak_mat_tmp202 + weak_mat_tmp232)) + C10*(s_t(4)*weak_mat_tmp1*weak_mat_tmp229 + weak_mat_tmp12 + weak_mat_tmp230*weak_mat_tmp29)) + weak_mat_tmp119*weak_mat_tmp228) + trial_grad5*(weak_mat_tmp228*weak_mat_tmp94 + weak_mat_tmp241) + trial_grad6*(weak_mat_tmp228*weak_mat_tmp70 + weak_mat_tmp243) + trial_grad7*(weak_mat_tmp111*weak_mat_tmp228 + weak_mat_tmp238) + trial_grad8*(weak_mat_tmp127*weak_mat_tmp228 + weak_mat_tmp248);
    const s_t material5 = trial_grad0*(weak_mat_tmp109 + weak_mat_tmp249*weak_mat_tmp3) + trial_grad1*(weak_mat_tmp161 + weak_mat_tmp249*weak_mat_tmp79) + trial_grad2*(weak_mat_tmp183 + weak_mat_tmp249*weak_mat_tmp86) + trial_grad3*(weak_mat_tmp214 + weak_mat_tmp249*weak_mat_tmp57) + trial_grad4*(weak_mat_tmp119*weak_mat_tmp249 + weak_mat_tmp241) + trial_grad5*(iso_history_qp[2] + iso_relaxation_gamma*(C01*(weak_mat_tmp251*weak_mat_tmp55 + s_t(2)*weak_mat_tmp252*weak_mat_tmp94 + weak_mat_tmp36*(weak_mat_tmp204 + weak_mat_tmp232)) + C10*(s_t(4)*gu5*weak_mat_tmp250 + weak_mat_tmp12 + weak_mat_tmp251*weak_mat_tmp29)) + weak_mat_tmp249*weak_mat_tmp94) + trial_grad6*(weak_mat_tmp249*weak_mat_tmp70 + weak_mat_tmp258) + trial_grad7*(weak_mat_tmp111*weak_mat_tmp249 + weak_mat_tmp260) + trial_grad8*(weak_mat_tmp127*weak_mat_tmp249 + weak_mat_tmp256);
    const s_t material6 = trial_grad0*(weak_mat_tmp261*weak_mat_tmp3 + weak_mat_tmp78) + trial_grad1*(weak_mat_tmp164 + weak_mat_tmp261*weak_mat_tmp79) + trial_grad2*(weak_mat_tmp194 + weak_mat_tmp261*weak_mat_tmp86) + trial_grad3*(weak_mat_tmp211 + weak_mat_tmp261*weak_mat_tmp57) + trial_grad4*(weak_mat_tmp119*weak_mat_tmp261 + weak_mat_tmp243) + trial_grad5*(weak_mat_tmp258 + weak_mat_tmp261*weak_mat_tmp94) + trial_grad6*(iso_history_qp[0] + iso_relaxation_gamma*(C01*(weak_mat_tmp263*weak_mat_tmp55 + s_t(2)*weak_mat_tmp266*weak_mat_tmp77 + weak_mat_tmp36*(weak_mat_tmp264 + weak_mat_tmp265)) + C10*(s_t(4)*gu6*weak_mat_tmp262 + weak_mat_tmp12 + weak_mat_tmp263*weak_mat_tmp29)) + weak_mat_tmp261*weak_mat_tmp70) + trial_grad7*(weak_mat_tmp111*weak_mat_tmp261 + weak_mat_tmp270) + trial_grad8*(weak_mat_tmp127*weak_mat_tmp261 + weak_mat_tmp272);
    const s_t material7 = trial_grad0*(weak_mat_tmp118 + weak_mat_tmp273*weak_mat_tmp3) + trial_grad1*(weak_mat_tmp148 + weak_mat_tmp273*weak_mat_tmp79) + trial_grad2*(weak_mat_tmp191 + weak_mat_tmp273*weak_mat_tmp86) + trial_grad3*(weak_mat_tmp222 + weak_mat_tmp273*weak_mat_tmp57) + trial_grad4*(weak_mat_tmp119*weak_mat_tmp273 + weak_mat_tmp238) + trial_grad5*(weak_mat_tmp260 + weak_mat_tmp273*weak_mat_tmp94) + trial_grad6*(weak_mat_tmp270 + weak_mat_tmp273*weak_mat_tmp70) + trial_grad7*(iso_history_qp[1] + iso_relaxation_gamma*(C01*(s_t(2)*weak_mat_tmp115*weak_mat_tmp277 + weak_mat_tmp275*weak_mat_tmp55 + weak_mat_tmp36*(weak_mat_tmp264 + weak_mat_tmp276)) + C10*(s_t(4)*gu7*weak_mat_tmp274 + weak_mat_tmp12 + weak_mat_tmp275*weak_mat_tmp29)) + weak_mat_tmp111*weak_mat_tmp273) + trial_grad8*(weak_mat_tmp127*weak_mat_tmp273 + weak_mat_tmp281);
    const s_t material8 = trial_grad0*(weak_mat_tmp134 + weak_mat_tmp282*weak_mat_tmp3) + trial_grad1*(weak_mat_tmp173 + weak_mat_tmp282*weak_mat_tmp79) + trial_grad2*(weak_mat_tmp185 + weak_mat_tmp282*weak_mat_tmp86) + trial_grad3*(weak_mat_tmp227 + weak_mat_tmp282*weak_mat_tmp57) + trial_grad4*(weak_mat_tmp119*weak_mat_tmp282 + weak_mat_tmp248) + trial_grad5*(weak_mat_tmp256 + weak_mat_tmp282*weak_mat_tmp94) + trial_grad6*(weak_mat_tmp272 + weak_mat_tmp282*weak_mat_tmp70) + trial_grad7*(weak_mat_tmp111*weak_mat_tmp282 + weak_mat_tmp281) + trial_grad8*(iso_history_qp[2] + iso_relaxation_gamma*(C01*(s_t(2)*weak_mat_tmp131*weak_mat_tmp280 + weak_mat_tmp283*weak_mat_tmp55 + weak_mat_tmp36*(weak_mat_tmp265 + weak_mat_tmp276)) + C10*(weak_mat_tmp12 + s_t(4)*weak_mat_tmp2*weak_mat_tmp278 + weak_mat_tmp283*weak_mat_tmp29)) + weak_mat_tmp127*weak_mat_tmp282);
      loperand0_values[lane] = qw * (material0 * adj_lane0 + material1 * adj_lane1 + material2 * adj_lane2);
      loperand1_values[lane] = qw * (material0 * adj_lane3 + material1 * adj_lane4 + material2 * adj_lane5);
      loperand2_values[lane] = qw * (material0 * adj_lane6 + material1 * adj_lane7 + material2 * adj_lane8);
      loperand3_values[lane] = qw * (material3 * adj_lane0 + material4 * adj_lane1 + material5 * adj_lane2);
      loperand4_values[lane] = qw * (material3 * adj_lane3 + material4 * adj_lane4 + material5 * adj_lane5);
      loperand5_values[lane] = qw * (material3 * adj_lane6 + material4 * adj_lane7 + material5 * adj_lane8);
      loperand6_values[lane] = qw * (material6 * adj_lane0 + material7 * adj_lane1 + material8 * adj_lane2);
      loperand7_values[lane] = qw * (material6 * adj_lane3 + material7 * adj_lane4 + material8 * adj_lane5);
      loperand8_values[lane] = qw * (material6 * adj_lane6 + material7 * adj_lane7 + material8 * adj_lane8);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t tref0 = grad_ref_x[q * NS + shape];
        const s_t tref1 = grad_ref_y[q * NS + shape];
        const s_t tref2 = grad_ref_z[q * NS + shape];
        s_t *const RSTR out_shape0 = out_streams[3 * shape];
        s_t *const RSTR out_shape1 = out_streams[3 * shape + 1];
        s_t *const RSTR out_shape2 = out_streams[3 * shape + 2];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          out_shape0[lane] += loperand0_values[lane] * tref0 + loperand1_values[lane] * tref1 + loperand2_values[lane] * tref2;
          out_shape1[lane] += loperand3_values[lane] * tref0 + loperand4_values[lane] * tref1 + loperand5_values[lane] * tref2;
          out_shape2[lane] += loperand6_values[lane] * tref0 + loperand7_values[lane] * tref1 + loperand8_values[lane] * tref2;
        }
      }
    }
}

template <typename s_t, int NS, int VS>
static SFEM_INLINE void continued_mooney_rivlin_prony_d3_simplex_tet4_apply_block(
        const int ne,
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
        const s_t C01,
        const s_t C10,
        const s_t Jc,
        const s_t Jmin,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const ptrdiff_t qp_stride,
        const s_t *const RSTR u_streams[NS * 3],
        const s_t *const RSTR h_streams[NS * 3],
        s_t *const RSTR out_streams[NS * 3]
) {
  static_assert(VS > 0, "VS must be positive");
    {
      const s_t qw = (s_t(1) / s_t(6));
      const s_t *const RSTR iso_history_q = iso_history;
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = lane;
      const s_t adj_lane0 = adj0[goff];
      const s_t adj_lane1 = adj1[goff];
      const s_t adj_lane2 = adj2[goff];
      const s_t adj_lane3 = adj3[goff];
      const s_t adj_lane4 = adj4[goff];
      const s_t adj_lane5 = adj5[goff];
      const s_t adj_lane6 = adj6[goff];
      const s_t adj_lane7 = adj7[goff];
      const s_t adj_lane8 = adj8[goff];
      const s_t det_lane0 = det0[goff];
      const s_t iso_history_qp[6] = {iso_history_q[lane], iso_history_q[1 * qp_stride + lane], iso_history_q[2 * qp_stride + lane], iso_history_q[3 * qp_stride + lane], iso_history_q[4 * qp_stride + lane], iso_history_q[5 * qp_stride + lane]};
      const s_t gu_ref0 = -(u_streams[0][lane]) + u_streams[3][lane];
      const s_t grad_h_ref0 = -(h_streams[0][lane]) + h_streams[3][lane];
      const s_t gu_ref1 = -(u_streams[0][lane]) + u_streams[6][lane];
      const s_t grad_h_ref1 = -(h_streams[0][lane]) + h_streams[6][lane];
      const s_t gu_ref2 = -(u_streams[0][lane]) + u_streams[9][lane];
      const s_t grad_h_ref2 = -(h_streams[0][lane]) + h_streams[9][lane];
      const s_t gu_ref3 = -(u_streams[1][lane]) + u_streams[4][lane];
      const s_t grad_h_ref3 = -(h_streams[1][lane]) + h_streams[4][lane];
      const s_t gu_ref4 = -(u_streams[1][lane]) + u_streams[7][lane];
      const s_t grad_h_ref4 = -(h_streams[1][lane]) + h_streams[7][lane];
      const s_t gu_ref5 = -(u_streams[1][lane]) + u_streams[10][lane];
      const s_t grad_h_ref5 = -(h_streams[1][lane]) + h_streams[10][lane];
      const s_t gu_ref6 = -(u_streams[2][lane]) + u_streams[5][lane];
      const s_t grad_h_ref6 = -(h_streams[2][lane]) + h_streams[5][lane];
      const s_t gu_ref7 = -(u_streams[2][lane]) + u_streams[8][lane];
      const s_t grad_h_ref7 = -(h_streams[2][lane]) + h_streams[8][lane];
      const s_t gu_ref8 = -(u_streams[2][lane]) + u_streams[11][lane];
      const s_t grad_h_ref8 = -(h_streams[2][lane]) + h_streams[11][lane];
      const s_t idet = s_t(1) / det_lane0;
      const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane3 + gu_ref2 * adj_lane6) * idet;
      const s_t trial_grad0 = (grad_h_ref0 * adj_lane0 + grad_h_ref1 * adj_lane3 + grad_h_ref2 * adj_lane6) * idet;
      const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane4 + gu_ref2 * adj_lane7) * idet;
      const s_t trial_grad1 = (grad_h_ref0 * adj_lane1 + grad_h_ref1 * adj_lane4 + grad_h_ref2 * adj_lane7) * idet;
      const s_t gu2 = (gu_ref0 * adj_lane2 + gu_ref1 * adj_lane5 + gu_ref2 * adj_lane8) * idet;
      const s_t trial_grad2 = (grad_h_ref0 * adj_lane2 + grad_h_ref1 * adj_lane5 + grad_h_ref2 * adj_lane8) * idet;
      const s_t gu3 = (gu_ref3 * adj_lane0 + gu_ref4 * adj_lane3 + gu_ref5 * adj_lane6) * idet;
      const s_t trial_grad3 = (grad_h_ref3 * adj_lane0 + grad_h_ref4 * adj_lane3 + grad_h_ref5 * adj_lane6) * idet;
      const s_t gu4 = (gu_ref3 * adj_lane1 + gu_ref4 * adj_lane4 + gu_ref5 * adj_lane7) * idet;
      const s_t trial_grad4 = (grad_h_ref3 * adj_lane1 + grad_h_ref4 * adj_lane4 + grad_h_ref5 * adj_lane7) * idet;
      const s_t gu5 = (gu_ref3 * adj_lane2 + gu_ref4 * adj_lane5 + gu_ref5 * adj_lane8) * idet;
      const s_t trial_grad5 = (grad_h_ref3 * adj_lane2 + grad_h_ref4 * adj_lane5 + grad_h_ref5 * adj_lane8) * idet;
      const s_t gu6 = (gu_ref6 * adj_lane0 + gu_ref7 * adj_lane3 + gu_ref8 * adj_lane6) * idet;
      const s_t trial_grad6 = (grad_h_ref6 * adj_lane0 + grad_h_ref7 * adj_lane3 + grad_h_ref8 * adj_lane6) * idet;
      const s_t gu7 = (gu_ref6 * adj_lane1 + gu_ref7 * adj_lane4 + gu_ref8 * adj_lane7) * idet;
      const s_t trial_grad7 = (grad_h_ref6 * adj_lane1 + grad_h_ref7 * adj_lane4 + grad_h_ref8 * adj_lane7) * idet;
      const s_t gu8 = (gu_ref6 * adj_lane2 + gu_ref7 * adj_lane5 + gu_ref8 * adj_lane8) * idet;
      const s_t trial_grad8 = (grad_h_ref6 * adj_lane2 + grad_h_ref7 * adj_lane5 + grad_h_ref8 * adj_lane8) * idet;
    const s_t weak_mat_tmp0 = gu5*gu7;
    const s_t weak_mat_tmp1 = gu4 + s_t(1);
    const s_t weak_mat_tmp2 = gu8 + s_t(1);
    const s_t weak_mat_tmp3 = -weak_mat_tmp0 + weak_mat_tmp1*weak_mat_tmp2;
    const s_t weak_mat_tmp4 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp5 = -s_t(2)*weak_mat_tmp1*weak_mat_tmp2;
    const s_t weak_mat_tmp6 = ((s_t(1) / s_t(2)))*K;
    const s_t weak_mat_tmp7 = weak_mat_tmp6*(-weak_mat_tmp4 - weak_mat_tmp5);
    const s_t weak_mat_tmp8 = gu3*weak_mat_tmp2;
    const s_t weak_mat_tmp9 = gu6*weak_mat_tmp1;
    const s_t weak_mat_tmp10 = gu0 + s_t(1);
    const s_t weak_mat_tmp11 = gu1*gu5*gu6 - gu1*weak_mat_tmp8 + gu2*gu3*gu7 - gu2*weak_mat_tmp9 - weak_mat_tmp0*weak_mat_tmp10 + weak_mat_tmp1*weak_mat_tmp10*weak_mat_tmp2;
    const s_t weak_mat_tmp12 = s_t(2)*sfem_continued_power<s_t>(weak_mat_tmp11, (s_t(2) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp13 = sfem_continued_power_d1<s_t>(weak_mat_tmp11, (s_t(2) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp14 = weak_mat_tmp13*weak_mat_tmp3;
    const s_t weak_mat_tmp15 = pow_2(weak_mat_tmp3);
    const s_t weak_mat_tmp16 = pow_2(gu1);
    const s_t weak_mat_tmp17 = pow_2(gu7);
    const s_t weak_mat_tmp18 = pow_2(weak_mat_tmp1);
    const s_t weak_mat_tmp19 = weak_mat_tmp16 + weak_mat_tmp17 + weak_mat_tmp18;
    const s_t weak_mat_tmp20 = pow_2(gu2);
    const s_t weak_mat_tmp21 = pow_2(gu5);
    const s_t weak_mat_tmp22 = pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp23 = weak_mat_tmp20 + weak_mat_tmp21 + weak_mat_tmp22;
    const s_t weak_mat_tmp24 = pow_2(gu3);
    const s_t weak_mat_tmp25 = pow_2(gu6);
    const s_t weak_mat_tmp26 = pow_2(weak_mat_tmp10);
    const s_t weak_mat_tmp27 = weak_mat_tmp24 + weak_mat_tmp25 + weak_mat_tmp26;
    const s_t weak_mat_tmp28 = weak_mat_tmp19 + weak_mat_tmp23 + weak_mat_tmp27;
    const s_t weak_mat_tmp29 = weak_mat_tmp28*sfem_continued_power_d2<s_t>(weak_mat_tmp11, (s_t(2) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp30 = s_t(2)*weak_mat_tmp21;
    const s_t weak_mat_tmp31 = s_t(2)*weak_mat_tmp22;
    const s_t weak_mat_tmp32 = weak_mat_tmp30 + weak_mat_tmp31;
    const s_t weak_mat_tmp33 = s_t(2)*weak_mat_tmp17;
    const s_t weak_mat_tmp34 = s_t(2)*weak_mat_tmp18;
    const s_t weak_mat_tmp35 = weak_mat_tmp33 + weak_mat_tmp34;
    const s_t weak_mat_tmp36 = sfem_continued_power<s_t>(weak_mat_tmp11, (s_t(4) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp37 = gu6*gu7;
    const s_t weak_mat_tmp38 = gu1*weak_mat_tmp10;
    const s_t weak_mat_tmp39 = gu3*weak_mat_tmp1;
    const s_t weak_mat_tmp40 = weak_mat_tmp37 + weak_mat_tmp38 + weak_mat_tmp39;
    const s_t weak_mat_tmp41 = s_t(2)*gu1;
    const s_t weak_mat_tmp42 = gu3*gu5;
    const s_t weak_mat_tmp43 = gu2*weak_mat_tmp10;
    const s_t weak_mat_tmp44 = gu6*weak_mat_tmp2;
    const s_t weak_mat_tmp45 = weak_mat_tmp42 + weak_mat_tmp43 + weak_mat_tmp44;
    const s_t weak_mat_tmp46 = s_t(2)*gu2;
    const s_t weak_mat_tmp47 = s_t(2)*weak_mat_tmp10;
    const s_t weak_mat_tmp48 = sfem_continued_power_d1<s_t>(weak_mat_tmp11, (s_t(4) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp49 = weak_mat_tmp48*(s_t(2)*weak_mat_tmp10*weak_mat_tmp28 - weak_mat_tmp27*weak_mat_tmp47 - weak_mat_tmp40*weak_mat_tmp41 - weak_mat_tmp45*weak_mat_tmp46);
    const s_t weak_mat_tmp50 = gu1*gu2;
    const s_t weak_mat_tmp51 = gu5*weak_mat_tmp1;
    const s_t weak_mat_tmp52 = gu7*weak_mat_tmp2;
    const s_t weak_mat_tmp53 = weak_mat_tmp50 + weak_mat_tmp51 + weak_mat_tmp52;
    const s_t weak_mat_tmp54 = -(s_t(1) / s_t(2))*pow_2(weak_mat_tmp19) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp23) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp27) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp28) - pow_2(weak_mat_tmp40) - pow_2(weak_mat_tmp45) - pow_2(weak_mat_tmp53);
    const s_t weak_mat_tmp55 = weak_mat_tmp54*sfem_continued_power_d2<s_t>(weak_mat_tmp11, (s_t(4) / s_t(3)), Jc, Jmin);
    const s_t weak_mat_tmp56 = gu1*weak_mat_tmp2;
    const s_t weak_mat_tmp57 = gu2*gu7 - weak_mat_tmp56;
    const s_t weak_mat_tmp58 = s_t(2)*gu3;
    const s_t weak_mat_tmp59 = weak_mat_tmp13*weak_mat_tmp47;
    const s_t weak_mat_tmp60 = weak_mat_tmp3*weak_mat_tmp57;
    const s_t weak_mat_tmp61 = gu5*weak_mat_tmp46;
    const s_t weak_mat_tmp62 = weak_mat_tmp1*weak_mat_tmp41;
    const s_t weak_mat_tmp63 = s_t(2)*gu5;
    const s_t weak_mat_tmp64 = s_t(2)*weak_mat_tmp1;
    const s_t weak_mat_tmp65 = s_t(2)*gu3*weak_mat_tmp28 - weak_mat_tmp27*weak_mat_tmp58 - weak_mat_tmp40*weak_mat_tmp64 - weak_mat_tmp45*weak_mat_tmp63;
    const s_t weak_mat_tmp66 = weak_mat_tmp3*weak_mat_tmp48;
    const s_t weak_mat_tmp67 = iso_relaxation_gamma*(C01*(weak_mat_tmp36*(-weak_mat_tmp61 - weak_mat_tmp62) + weak_mat_tmp49*weak_mat_tmp57 + weak_mat_tmp55*weak_mat_tmp60 + weak_mat_tmp65*weak_mat_tmp66) + C10*(weak_mat_tmp14*weak_mat_tmp58 + weak_mat_tmp29*weak_mat_tmp60 + weak_mat_tmp57*weak_mat_tmp59));
    const s_t weak_mat_tmp68 = gu1*gu5;
    const s_t weak_mat_tmp69 = gu2*weak_mat_tmp1;
    const s_t weak_mat_tmp70 = weak_mat_tmp68 - weak_mat_tmp69;
    const s_t weak_mat_tmp71 = s_t(2)*gu6;
    const s_t weak_mat_tmp72 = weak_mat_tmp3*weak_mat_tmp70;
    const s_t weak_mat_tmp73 = gu7*weak_mat_tmp41;
    const s_t weak_mat_tmp74 = weak_mat_tmp2*weak_mat_tmp46;
    const s_t weak_mat_tmp75 = s_t(2)*gu7;
    const s_t weak_mat_tmp76 = s_t(2)*weak_mat_tmp2;
    const s_t weak_mat_tmp77 = s_t(2)*gu6*weak_mat_tmp28 - weak_mat_tmp27*weak_mat_tmp71 - weak_mat_tmp40*weak_mat_tmp75 - weak_mat_tmp45*weak_mat_tmp76;
    const s_t weak_mat_tmp78 = iso_relaxation_gamma*(C01*(weak_mat_tmp36*(-weak_mat_tmp73 - weak_mat_tmp74) + weak_mat_tmp49*weak_mat_tmp70 + weak_mat_tmp55*weak_mat_tmp72 + weak_mat_tmp66*weak_mat_tmp77) + C10*(weak_mat_tmp14*weak_mat_tmp71 + weak_mat_tmp29*weak_mat_tmp72 + weak_mat_tmp59*weak_mat_tmp70));
    const s_t weak_mat_tmp79 = gu5*gu6 - weak_mat_tmp8;
    const s_t weak_mat_tmp80 = weak_mat_tmp3*weak_mat_tmp79;
    const s_t weak_mat_tmp81 = s_t(2)*weak_mat_tmp37;
    const s_t weak_mat_tmp82 = s_t(2)*weak_mat_tmp39;
    const s_t weak_mat_tmp83 = s_t(2)*gu1*weak_mat_tmp28 - weak_mat_tmp19*weak_mat_tmp41 - weak_mat_tmp40*weak_mat_tmp47 - weak_mat_tmp46*weak_mat_tmp53;
    const s_t weak_mat_tmp84 = iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_mat_tmp36*(-weak_mat_tmp81 - weak_mat_tmp82) + weak_mat_tmp49*weak_mat_tmp79 + weak_mat_tmp55*weak_mat_tmp80 + weak_mat_tmp66*weak_mat_tmp83) + C10*(weak_mat_tmp14*weak_mat_tmp41 + weak_mat_tmp29*weak_mat_tmp80 + weak_mat_tmp59*weak_mat_tmp79));
    const s_t weak_mat_tmp85 = gu3*gu7;
    const s_t weak_mat_tmp86 = weak_mat_tmp85 - weak_mat_tmp9;
    const s_t weak_mat_tmp87 = weak_mat_tmp3*weak_mat_tmp86;
    const s_t weak_mat_tmp88 = s_t(2)*weak_mat_tmp42;
    const s_t weak_mat_tmp89 = s_t(2)*weak_mat_tmp44;
    const s_t weak_mat_tmp90 = s_t(2)*gu2*weak_mat_tmp28 - weak_mat_tmp23*weak_mat_tmp46 - weak_mat_tmp41*weak_mat_tmp53 - weak_mat_tmp45*weak_mat_tmp47;
    const s_t weak_mat_tmp91 = iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_mat_tmp36*(-weak_mat_tmp88 - weak_mat_tmp89) + weak_mat_tmp49*weak_mat_tmp86 + weak_mat_tmp55*weak_mat_tmp87 + weak_mat_tmp66*weak_mat_tmp90) + C10*(weak_mat_tmp14*weak_mat_tmp46 + weak_mat_tmp29*weak_mat_tmp87 + weak_mat_tmp59*weak_mat_tmp86));
    const s_t weak_mat_tmp92 = gu1*gu6;
    const s_t weak_mat_tmp93 = gu7*weak_mat_tmp10;
    const s_t weak_mat_tmp94 = weak_mat_tmp92 - weak_mat_tmp93;
    const s_t weak_mat_tmp95 = weak_mat_tmp13*weak_mat_tmp28;
    const s_t weak_mat_tmp96 = gu7*weak_mat_tmp95;
    const s_t weak_mat_tmp97 = weak_mat_tmp3*weak_mat_tmp94;
    const s_t weak_mat_tmp98 = gu2*gu3;
    const s_t weak_mat_tmp99 = s_t(2)*weak_mat_tmp98;
    const s_t weak_mat_tmp100 = s_t(2)*gu5*weak_mat_tmp28 - weak_mat_tmp23*weak_mat_tmp63 - weak_mat_tmp45*weak_mat_tmp58 - weak_mat_tmp53*weak_mat_tmp64;
    const s_t weak_mat_tmp101 = weak_mat_tmp48*weak_mat_tmp54;
    const s_t weak_mat_tmp102 = gu7*weak_mat_tmp101;
    const s_t weak_mat_tmp103 = gu0*gu4;
    const s_t weak_mat_tmp104 = gu1*gu3;
    const s_t weak_mat_tmp105 = gu2*gu6;
    const s_t weak_mat_tmp106 = gu5*gu6;
    const s_t weak_mat_tmp107 = K*(gu0*gu8 - gu0*weak_mat_tmp0 + gu0 + gu1*weak_mat_tmp106 + gu2*weak_mat_tmp85 + gu4*gu8 - gu4*weak_mat_tmp105 + gu4 + gu8*weak_mat_tmp103 - gu8*weak_mat_tmp104 + gu8 - weak_mat_tmp0 + weak_mat_tmp103 - weak_mat_tmp104 - weak_mat_tmp105);
    const s_t weak_mat_tmp108 = gu7*weak_mat_tmp107;
    const s_t weak_mat_tmp109 = iso_relaxation_gamma*(C01*(weak_mat_tmp100*weak_mat_tmp66 - weak_mat_tmp102 + weak_mat_tmp36*(s_t(4)*gu5*weak_mat_tmp10 - weak_mat_tmp99) + weak_mat_tmp49*weak_mat_tmp94 + weak_mat_tmp55*weak_mat_tmp97) + C10*(weak_mat_tmp14*weak_mat_tmp63 + weak_mat_tmp29*weak_mat_tmp97 + weak_mat_tmp59*weak_mat_tmp94 - weak_mat_tmp96)) - weak_mat_tmp108;
    const s_t weak_mat_tmp110 = gu5*weak_mat_tmp10;
    const s_t weak_mat_tmp111 = -weak_mat_tmp110 + weak_mat_tmp98;
    const s_t weak_mat_tmp112 = gu5*weak_mat_tmp95;
    const s_t weak_mat_tmp113 = weak_mat_tmp111*weak_mat_tmp3;
    const s_t weak_mat_tmp114 = s_t(2)*weak_mat_tmp92;
    const s_t weak_mat_tmp115 = s_t(2)*gu7*weak_mat_tmp28 - weak_mat_tmp19*weak_mat_tmp75 - weak_mat_tmp40*weak_mat_tmp71 - weak_mat_tmp53*weak_mat_tmp76;
    const s_t weak_mat_tmp116 = gu5*weak_mat_tmp101;
    const s_t weak_mat_tmp117 = gu5*weak_mat_tmp107;
    const s_t weak_mat_tmp118 = iso_relaxation_gamma*(C01*(weak_mat_tmp111*weak_mat_tmp49 + weak_mat_tmp113*weak_mat_tmp55 + weak_mat_tmp115*weak_mat_tmp66 - weak_mat_tmp116 + weak_mat_tmp36*(s_t(4)*gu7*weak_mat_tmp10 - weak_mat_tmp114)) + C10*(weak_mat_tmp111*weak_mat_tmp59 - weak_mat_tmp112 + weak_mat_tmp113*weak_mat_tmp29 + weak_mat_tmp14*weak_mat_tmp75)) - weak_mat_tmp117;
    const s_t weak_mat_tmp119 = weak_mat_tmp10*weak_mat_tmp2 - weak_mat_tmp105;
    const s_t weak_mat_tmp120 = weak_mat_tmp2*weak_mat_tmp95;
    const s_t weak_mat_tmp121 = weak_mat_tmp119*weak_mat_tmp3;
    const s_t weak_mat_tmp122 = s_t(2)*weak_mat_tmp104;
    const s_t weak_mat_tmp123 = s_t(2)*weak_mat_tmp1*weak_mat_tmp28 - weak_mat_tmp19*weak_mat_tmp64 - weak_mat_tmp40*weak_mat_tmp58 - weak_mat_tmp53*weak_mat_tmp63;
    const s_t weak_mat_tmp124 = weak_mat_tmp101*weak_mat_tmp2;
    const s_t weak_mat_tmp125 = weak_mat_tmp107*weak_mat_tmp2;
    const s_t weak_mat_tmp126 = iso_relaxation_gamma*(C01*(weak_mat_tmp119*weak_mat_tmp49 + weak_mat_tmp121*weak_mat_tmp55 + weak_mat_tmp123*weak_mat_tmp66 + weak_mat_tmp124 + weak_mat_tmp36*(s_t(4)*weak_mat_tmp1*weak_mat_tmp10 - weak_mat_tmp122)) + C10*(weak_mat_tmp119*weak_mat_tmp59 + weak_mat_tmp120 + weak_mat_tmp121*weak_mat_tmp29 + weak_mat_tmp14*weak_mat_tmp64)) + weak_mat_tmp125;
    const s_t weak_mat_tmp127 = weak_mat_tmp1*weak_mat_tmp10 - weak_mat_tmp104;
    const s_t weak_mat_tmp128 = weak_mat_tmp1*weak_mat_tmp95;
    const s_t weak_mat_tmp129 = weak_mat_tmp127*weak_mat_tmp3;
    const s_t weak_mat_tmp130 = s_t(2)*weak_mat_tmp105;
    const s_t weak_mat_tmp131 = s_t(2)*weak_mat_tmp2*weak_mat_tmp28 - weak_mat_tmp23*weak_mat_tmp76 - weak_mat_tmp45*weak_mat_tmp71 - weak_mat_tmp53*weak_mat_tmp75;
    const s_t weak_mat_tmp132 = weak_mat_tmp1*weak_mat_tmp101;
    const s_t weak_mat_tmp133 = weak_mat_tmp1*weak_mat_tmp107;
    const s_t weak_mat_tmp134 = iso_relaxation_gamma*(C01*(weak_mat_tmp127*weak_mat_tmp49 + weak_mat_tmp129*weak_mat_tmp55 + weak_mat_tmp131*weak_mat_tmp66 + weak_mat_tmp132 + weak_mat_tmp36*(s_t(4)*weak_mat_tmp10*weak_mat_tmp2 - weak_mat_tmp130)) + C10*(weak_mat_tmp127*weak_mat_tmp59 + weak_mat_tmp128 + weak_mat_tmp129*weak_mat_tmp29 + weak_mat_tmp14*weak_mat_tmp76)) + weak_mat_tmp133;
    const s_t weak_mat_tmp135 = -s_t(2)*weak_mat_tmp106;
    const s_t weak_mat_tmp136 = s_t(2)*weak_mat_tmp8;
    const s_t weak_mat_tmp137 = weak_mat_tmp6*(-weak_mat_tmp135 - weak_mat_tmp136);
    const s_t weak_mat_tmp138 = weak_mat_tmp13*weak_mat_tmp79;
    const s_t weak_mat_tmp139 = pow_2(weak_mat_tmp79);
    const s_t weak_mat_tmp140 = s_t(2)*weak_mat_tmp24;
    const s_t weak_mat_tmp141 = s_t(2)*weak_mat_tmp25;
    const s_t weak_mat_tmp142 = weak_mat_tmp140 + weak_mat_tmp141;
    const s_t weak_mat_tmp143 = weak_mat_tmp48*weak_mat_tmp83;
    const s_t weak_mat_tmp144 = weak_mat_tmp13*weak_mat_tmp41;
    const s_t weak_mat_tmp145 = weak_mat_tmp111*weak_mat_tmp79;
    const s_t weak_mat_tmp146 = gu6*weak_mat_tmp47;
    const s_t weak_mat_tmp147 = weak_mat_tmp48*weak_mat_tmp79;
    const s_t weak_mat_tmp148 = iso_relaxation_gamma*(C01*(weak_mat_tmp111*weak_mat_tmp143 + weak_mat_tmp115*weak_mat_tmp147 + weak_mat_tmp145*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp146 - weak_mat_tmp74)) + C10*(weak_mat_tmp111*weak_mat_tmp144 + weak_mat_tmp138*weak_mat_tmp75 + weak_mat_tmp145*weak_mat_tmp29));
    const s_t weak_mat_tmp149 = weak_mat_tmp79*weak_mat_tmp86;
    const s_t weak_mat_tmp150 = s_t(2)*weak_mat_tmp51;
    const s_t weak_mat_tmp151 = s_t(2)*weak_mat_tmp52;
    const s_t weak_mat_tmp152 = iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_mat_tmp143*weak_mat_tmp86 + weak_mat_tmp147*weak_mat_tmp90 + weak_mat_tmp149*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp150 - weak_mat_tmp151)) + C10*(weak_mat_tmp138*weak_mat_tmp46 + weak_mat_tmp144*weak_mat_tmp86 + weak_mat_tmp149*weak_mat_tmp29));
    const s_t weak_mat_tmp153 = weak_mat_tmp119*weak_mat_tmp79;
    const s_t weak_mat_tmp154 = gu3*weak_mat_tmp47;
    const s_t weak_mat_tmp155 = iso_relaxation_gamma*(C01*(weak_mat_tmp119*weak_mat_tmp143 + weak_mat_tmp123*weak_mat_tmp147 + weak_mat_tmp153*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp154 - weak_mat_tmp61)) + C10*(weak_mat_tmp119*weak_mat_tmp144 + weak_mat_tmp138*weak_mat_tmp64 + weak_mat_tmp153*weak_mat_tmp29));
    const s_t weak_mat_tmp156 = gu6*weak_mat_tmp95;
    const s_t weak_mat_tmp157 = weak_mat_tmp79*weak_mat_tmp94;
    const s_t weak_mat_tmp158 = -s_t(2)*weak_mat_tmp69;
    const s_t weak_mat_tmp159 = gu6*weak_mat_tmp101;
    const s_t weak_mat_tmp160 = gu6*weak_mat_tmp107;
    const s_t weak_mat_tmp161 = iso_relaxation_gamma*(C01*(weak_mat_tmp100*weak_mat_tmp147 + weak_mat_tmp143*weak_mat_tmp94 + weak_mat_tmp157*weak_mat_tmp55 + weak_mat_tmp159 + weak_mat_tmp36*(weak_mat_tmp158 + s_t(4)*weak_mat_tmp68)) + C10*(weak_mat_tmp138*weak_mat_tmp63 + weak_mat_tmp144*weak_mat_tmp94 + weak_mat_tmp156 + weak_mat_tmp157*weak_mat_tmp29)) + weak_mat_tmp160;
    const s_t weak_mat_tmp162 = weak_mat_tmp70*weak_mat_tmp79;
    const s_t weak_mat_tmp163 = -s_t(2)*weak_mat_tmp93;
    const s_t weak_mat_tmp164 = iso_relaxation_gamma*(C01*(weak_mat_tmp116 + weak_mat_tmp143*weak_mat_tmp70 + weak_mat_tmp147*weak_mat_tmp77 + weak_mat_tmp162*weak_mat_tmp55 + weak_mat_tmp36*(weak_mat_tmp163 + s_t(4)*weak_mat_tmp92)) + C10*(weak_mat_tmp112 + weak_mat_tmp138*weak_mat_tmp71 + weak_mat_tmp144*weak_mat_tmp70 + weak_mat_tmp162*weak_mat_tmp29)) + weak_mat_tmp117;
    const s_t weak_mat_tmp165 = weak_mat_tmp57*weak_mat_tmp79;
    const s_t weak_mat_tmp166 = -s_t(2)*weak_mat_tmp1*weak_mat_tmp10;
    const s_t weak_mat_tmp167 = iso_relaxation_gamma*(C01*(-weak_mat_tmp124 + weak_mat_tmp143*weak_mat_tmp57 + weak_mat_tmp147*weak_mat_tmp65 + weak_mat_tmp165*weak_mat_tmp55 + weak_mat_tmp36*(s_t(4)*weak_mat_tmp104 + weak_mat_tmp166)) + C10*(-weak_mat_tmp120 + weak_mat_tmp138*weak_mat_tmp58 + weak_mat_tmp144*weak_mat_tmp57 + weak_mat_tmp165*weak_mat_tmp29)) - weak_mat_tmp125;
    const s_t weak_mat_tmp168 = gu3*weak_mat_tmp95;
    const s_t weak_mat_tmp169 = weak_mat_tmp127*weak_mat_tmp79;
    const s_t weak_mat_tmp170 = -s_t(2)*gu2*gu7;
    const s_t weak_mat_tmp171 = gu3*weak_mat_tmp101;
    const s_t weak_mat_tmp172 = gu3*weak_mat_tmp107;
    const s_t weak_mat_tmp173 = iso_relaxation_gamma*(C01*(weak_mat_tmp127*weak_mat_tmp143 + weak_mat_tmp131*weak_mat_tmp147 + weak_mat_tmp169*weak_mat_tmp55 - weak_mat_tmp171 + weak_mat_tmp36*(weak_mat_tmp170 + s_t(4)*weak_mat_tmp56)) + C10*(weak_mat_tmp127*weak_mat_tmp144 + weak_mat_tmp138*weak_mat_tmp76 - weak_mat_tmp168 + weak_mat_tmp169*weak_mat_tmp29)) - weak_mat_tmp172;
    const s_t weak_mat_tmp174 = s_t(2)*weak_mat_tmp85;
    const s_t weak_mat_tmp175 = -s_t(2)*weak_mat_tmp9;
    const s_t weak_mat_tmp176 = weak_mat_tmp6*(weak_mat_tmp174 + weak_mat_tmp175);
    const s_t weak_mat_tmp177 = weak_mat_tmp13*weak_mat_tmp86;
    const s_t weak_mat_tmp178 = pow_2(weak_mat_tmp86);
    const s_t weak_mat_tmp179 = weak_mat_tmp48*weak_mat_tmp90;
    const s_t weak_mat_tmp180 = weak_mat_tmp13*weak_mat_tmp46;
    const s_t weak_mat_tmp181 = weak_mat_tmp86*weak_mat_tmp94;
    const s_t weak_mat_tmp182 = weak_mat_tmp48*weak_mat_tmp86;
    const s_t weak_mat_tmp183 = iso_relaxation_gamma*(C01*(weak_mat_tmp100*weak_mat_tmp182 + weak_mat_tmp179*weak_mat_tmp94 + weak_mat_tmp181*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp154 - weak_mat_tmp62)) + C10*(weak_mat_tmp177*weak_mat_tmp63 + weak_mat_tmp180*weak_mat_tmp94 + weak_mat_tmp181*weak_mat_tmp29));
    const s_t weak_mat_tmp184 = weak_mat_tmp127*weak_mat_tmp86;
    const s_t weak_mat_tmp185 = iso_relaxation_gamma*(C01*(weak_mat_tmp127*weak_mat_tmp179 + weak_mat_tmp131*weak_mat_tmp182 + weak_mat_tmp184*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp146 - weak_mat_tmp73)) + C10*(weak_mat_tmp127*weak_mat_tmp180 + weak_mat_tmp177*weak_mat_tmp76 + weak_mat_tmp184*weak_mat_tmp29));
    const s_t weak_mat_tmp186 = weak_mat_tmp57*weak_mat_tmp86;
    const s_t weak_mat_tmp187 = -s_t(2)*weak_mat_tmp110;
    const s_t weak_mat_tmp188 = iso_relaxation_gamma*(C01*(weak_mat_tmp102 + weak_mat_tmp179*weak_mat_tmp57 + weak_mat_tmp182*weak_mat_tmp65 + weak_mat_tmp186*weak_mat_tmp55 + weak_mat_tmp36*(weak_mat_tmp187 + s_t(4)*weak_mat_tmp98)) + C10*(weak_mat_tmp177*weak_mat_tmp58 + weak_mat_tmp180*weak_mat_tmp57 + weak_mat_tmp186*weak_mat_tmp29 + weak_mat_tmp96)) + weak_mat_tmp108;
    const s_t weak_mat_tmp189 = weak_mat_tmp111*weak_mat_tmp86;
    const s_t weak_mat_tmp190 = s_t(2)*weak_mat_tmp56;
    const s_t weak_mat_tmp191 = iso_relaxation_gamma*(C01*(weak_mat_tmp111*weak_mat_tmp179 + weak_mat_tmp115*weak_mat_tmp182 + weak_mat_tmp171 + weak_mat_tmp189*weak_mat_tmp55 + weak_mat_tmp36*(s_t(4)*gu2*gu7 - weak_mat_tmp190)) + C10*(weak_mat_tmp111*weak_mat_tmp180 + weak_mat_tmp168 + weak_mat_tmp177*weak_mat_tmp75 + weak_mat_tmp189*weak_mat_tmp29)) + weak_mat_tmp172;
    const s_t weak_mat_tmp192 = weak_mat_tmp70*weak_mat_tmp86;
    const s_t weak_mat_tmp193 = -s_t(2)*weak_mat_tmp10*weak_mat_tmp2;
    const s_t weak_mat_tmp194 = iso_relaxation_gamma*(C01*(-weak_mat_tmp132 + weak_mat_tmp179*weak_mat_tmp70 + weak_mat_tmp182*weak_mat_tmp77 + weak_mat_tmp192*weak_mat_tmp55 + weak_mat_tmp36*(s_t(4)*weak_mat_tmp105 + weak_mat_tmp193)) + C10*(-weak_mat_tmp128 + weak_mat_tmp177*weak_mat_tmp71 + weak_mat_tmp180*weak_mat_tmp70 + weak_mat_tmp192*weak_mat_tmp29)) - weak_mat_tmp133;
    const s_t weak_mat_tmp195 = weak_mat_tmp119*weak_mat_tmp86;
    const s_t weak_mat_tmp196 = s_t(2)*weak_mat_tmp68;
    const s_t weak_mat_tmp197 = iso_relaxation_gamma*(C01*(weak_mat_tmp119*weak_mat_tmp179 + weak_mat_tmp123*weak_mat_tmp182 - weak_mat_tmp159 + weak_mat_tmp195*weak_mat_tmp55 + weak_mat_tmp36*(s_t(4)*gu2*weak_mat_tmp1 - weak_mat_tmp196)) + C10*(weak_mat_tmp119*weak_mat_tmp180 - weak_mat_tmp156 + weak_mat_tmp177*weak_mat_tmp64 + weak_mat_tmp195*weak_mat_tmp29)) - weak_mat_tmp160;
    const s_t weak_mat_tmp198 = weak_mat_tmp6*(-weak_mat_tmp170 - weak_mat_tmp190);
    const s_t weak_mat_tmp199 = weak_mat_tmp13*weak_mat_tmp57;
    const s_t weak_mat_tmp200 = pow_2(weak_mat_tmp57);
    const s_t weak_mat_tmp201 = s_t(2)*weak_mat_tmp20;
    const s_t weak_mat_tmp202 = weak_mat_tmp201 + weak_mat_tmp31;
    const s_t weak_mat_tmp203 = s_t(2)*weak_mat_tmp16;
    const s_t weak_mat_tmp204 = weak_mat_tmp203 + weak_mat_tmp33;
    const s_t weak_mat_tmp205 = weak_mat_tmp48*weak_mat_tmp57;
    const s_t weak_mat_tmp206 = weak_mat_tmp13*weak_mat_tmp58;
    const s_t weak_mat_tmp207 = weak_mat_tmp57*weak_mat_tmp70;
    const s_t weak_mat_tmp208 = weak_mat_tmp2*weak_mat_tmp63;
    const s_t weak_mat_tmp209 = gu7*weak_mat_tmp64;
    const s_t weak_mat_tmp210 = weak_mat_tmp48*weak_mat_tmp65;
    const s_t weak_mat_tmp211 = iso_relaxation_gamma*(C01*(weak_mat_tmp205*weak_mat_tmp77 + weak_mat_tmp207*weak_mat_tmp55 + weak_mat_tmp210*weak_mat_tmp70 + weak_mat_tmp36*(-weak_mat_tmp208 - weak_mat_tmp209)) + C10*(weak_mat_tmp199*weak_mat_tmp71 + weak_mat_tmp206*weak_mat_tmp70 + weak_mat_tmp207*weak_mat_tmp29));
    const s_t weak_mat_tmp212 = weak_mat_tmp57*weak_mat_tmp94;
    const s_t weak_mat_tmp213 = s_t(2)*weak_mat_tmp43;
    const s_t weak_mat_tmp214 = iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_mat_tmp100*weak_mat_tmp205 + weak_mat_tmp210*weak_mat_tmp94 + weak_mat_tmp212*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp213 - weak_mat_tmp89)) + C10*(weak_mat_tmp199*weak_mat_tmp63 + weak_mat_tmp206*weak_mat_tmp94 + weak_mat_tmp212*weak_mat_tmp29));
    const s_t weak_mat_tmp215 = weak_mat_tmp119*weak_mat_tmp57;
    const s_t weak_mat_tmp216 = s_t(2)*weak_mat_tmp38;
    const s_t weak_mat_tmp217 = iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_mat_tmp119*weak_mat_tmp210 + weak_mat_tmp123*weak_mat_tmp205 + weak_mat_tmp215*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp216 - weak_mat_tmp81)) + C10*(weak_mat_tmp119*weak_mat_tmp206 + weak_mat_tmp199*weak_mat_tmp64 + weak_mat_tmp215*weak_mat_tmp29));
    const s_t weak_mat_tmp218 = gu2*weak_mat_tmp95;
    const s_t weak_mat_tmp219 = weak_mat_tmp111*weak_mat_tmp57;
    const s_t weak_mat_tmp220 = gu2*weak_mat_tmp101;
    const s_t weak_mat_tmp221 = gu2*weak_mat_tmp107;
    const s_t weak_mat_tmp222 = iso_relaxation_gamma*(C01*(weak_mat_tmp111*weak_mat_tmp210 + weak_mat_tmp115*weak_mat_tmp205 + weak_mat_tmp219*weak_mat_tmp55 + weak_mat_tmp220 + weak_mat_tmp36*(weak_mat_tmp175 + s_t(4)*weak_mat_tmp85)) + C10*(weak_mat_tmp111*weak_mat_tmp206 + weak_mat_tmp199*weak_mat_tmp75 + weak_mat_tmp218 + weak_mat_tmp219*weak_mat_tmp29)) + weak_mat_tmp221;
    const s_t weak_mat_tmp223 = gu1*weak_mat_tmp95;
    const s_t weak_mat_tmp224 = weak_mat_tmp127*weak_mat_tmp57;
    const s_t weak_mat_tmp225 = gu1*weak_mat_tmp101;
    const s_t weak_mat_tmp226 = gu1*weak_mat_tmp107;
    const s_t weak_mat_tmp227 = iso_relaxation_gamma*(C01*(weak_mat_tmp127*weak_mat_tmp210 + weak_mat_tmp131*weak_mat_tmp205 + weak_mat_tmp224*weak_mat_tmp55 - weak_mat_tmp225 + weak_mat_tmp36*(weak_mat_tmp135 + s_t(4)*weak_mat_tmp8)) + C10*(weak_mat_tmp127*weak_mat_tmp206 + weak_mat_tmp199*weak_mat_tmp76 - weak_mat_tmp223 + weak_mat_tmp224*weak_mat_tmp29)) - weak_mat_tmp226;
    const s_t weak_mat_tmp228 = weak_mat_tmp6*(-weak_mat_tmp130 - weak_mat_tmp193);
    const s_t weak_mat_tmp229 = weak_mat_tmp119*weak_mat_tmp13;
    const s_t weak_mat_tmp230 = pow_2(weak_mat_tmp119);
    const s_t weak_mat_tmp231 = s_t(2)*weak_mat_tmp26;
    const s_t weak_mat_tmp232 = weak_mat_tmp141 + weak_mat_tmp231;
    const s_t weak_mat_tmp233 = weak_mat_tmp123*weak_mat_tmp48;
    const s_t weak_mat_tmp234 = weak_mat_tmp13*weak_mat_tmp64;
    const s_t weak_mat_tmp235 = weak_mat_tmp111*weak_mat_tmp119;
    const s_t weak_mat_tmp236 = gu6*weak_mat_tmp58;
    const s_t weak_mat_tmp237 = weak_mat_tmp119*weak_mat_tmp48;
    const s_t weak_mat_tmp238 = iso_relaxation_gamma*(C01*(weak_mat_tmp111*weak_mat_tmp233 + weak_mat_tmp115*weak_mat_tmp237 + weak_mat_tmp235*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp208 - weak_mat_tmp236)) + C10*(weak_mat_tmp111*weak_mat_tmp234 + weak_mat_tmp229*weak_mat_tmp75 + weak_mat_tmp235*weak_mat_tmp29));
    const s_t weak_mat_tmp239 = weak_mat_tmp119*weak_mat_tmp94;
    const s_t weak_mat_tmp240 = s_t(2)*weak_mat_tmp50;
    const s_t weak_mat_tmp241 = iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_mat_tmp100*weak_mat_tmp237 + weak_mat_tmp233*weak_mat_tmp94 + weak_mat_tmp239*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp151 - weak_mat_tmp240)) + C10*(weak_mat_tmp229*weak_mat_tmp63 + weak_mat_tmp234*weak_mat_tmp94 + weak_mat_tmp239*weak_mat_tmp29));
    const s_t weak_mat_tmp242 = weak_mat_tmp119*weak_mat_tmp70;
    const s_t weak_mat_tmp243 = iso_relaxation_gamma*(C01*(-weak_mat_tmp220 + weak_mat_tmp233*weak_mat_tmp70 + weak_mat_tmp237*weak_mat_tmp77 + weak_mat_tmp242*weak_mat_tmp55 + weak_mat_tmp36*(s_t(4)*gu6*weak_mat_tmp1 - weak_mat_tmp174)) + C10*(-weak_mat_tmp218 + weak_mat_tmp229*weak_mat_tmp71 + weak_mat_tmp234*weak_mat_tmp70 + weak_mat_tmp242*weak_mat_tmp29)) - weak_mat_tmp221;
    const s_t weak_mat_tmp244 = weak_mat_tmp10*weak_mat_tmp95;
    const s_t weak_mat_tmp245 = weak_mat_tmp119*weak_mat_tmp127;
    const s_t weak_mat_tmp246 = weak_mat_tmp10*weak_mat_tmp101;
    const s_t weak_mat_tmp247 = weak_mat_tmp10*weak_mat_tmp107;
    const s_t weak_mat_tmp248 = iso_relaxation_gamma*(C01*(weak_mat_tmp127*weak_mat_tmp233 + weak_mat_tmp131*weak_mat_tmp237 + weak_mat_tmp245*weak_mat_tmp55 + weak_mat_tmp246 + weak_mat_tmp36*(s_t(4)*weak_mat_tmp1*weak_mat_tmp2 - weak_mat_tmp4)) + C10*(weak_mat_tmp127*weak_mat_tmp234 + weak_mat_tmp229*weak_mat_tmp76 + weak_mat_tmp244 + weak_mat_tmp245*weak_mat_tmp29)) + weak_mat_tmp247;
    const s_t weak_mat_tmp249 = weak_mat_tmp6*(weak_mat_tmp114 + weak_mat_tmp163);
    const s_t weak_mat_tmp250 = weak_mat_tmp13*weak_mat_tmp94;
    const s_t weak_mat_tmp251 = pow_2(weak_mat_tmp94);
    const s_t weak_mat_tmp252 = weak_mat_tmp100*weak_mat_tmp48;
    const s_t weak_mat_tmp253 = weak_mat_tmp13*weak_mat_tmp63;
    const s_t weak_mat_tmp254 = weak_mat_tmp127*weak_mat_tmp94;
    const s_t weak_mat_tmp255 = weak_mat_tmp48*weak_mat_tmp94;
    const s_t weak_mat_tmp256 = iso_relaxation_gamma*(C01*(weak_mat_tmp127*weak_mat_tmp252 + weak_mat_tmp131*weak_mat_tmp255 + weak_mat_tmp254*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp209 - weak_mat_tmp236)) + C10*(weak_mat_tmp127*weak_mat_tmp253 + weak_mat_tmp250*weak_mat_tmp76 + weak_mat_tmp254*weak_mat_tmp29));
    const s_t weak_mat_tmp257 = weak_mat_tmp70*weak_mat_tmp94;
    const s_t weak_mat_tmp258 = iso_relaxation_gamma*(C01*(weak_mat_tmp225 + weak_mat_tmp252*weak_mat_tmp70 + weak_mat_tmp255*weak_mat_tmp77 + weak_mat_tmp257*weak_mat_tmp55 + weak_mat_tmp36*(s_t(4)*gu5*gu6 - weak_mat_tmp136)) + C10*(weak_mat_tmp223 + weak_mat_tmp250*weak_mat_tmp71 + weak_mat_tmp253*weak_mat_tmp70 + weak_mat_tmp257*weak_mat_tmp29)) + weak_mat_tmp226;
    const s_t weak_mat_tmp259 = weak_mat_tmp111*weak_mat_tmp94;
    const s_t weak_mat_tmp260 = iso_relaxation_gamma*(C01*(weak_mat_tmp111*weak_mat_tmp252 + weak_mat_tmp115*weak_mat_tmp255 - weak_mat_tmp246 + weak_mat_tmp259*weak_mat_tmp55 + weak_mat_tmp36*(s_t(4)*weak_mat_tmp0 + weak_mat_tmp5)) + C10*(weak_mat_tmp111*weak_mat_tmp253 - weak_mat_tmp244 + weak_mat_tmp250*weak_mat_tmp75 + weak_mat_tmp259*weak_mat_tmp29)) - weak_mat_tmp247;
    const s_t weak_mat_tmp261 = weak_mat_tmp6*(weak_mat_tmp158 + weak_mat_tmp196);
    const s_t weak_mat_tmp262 = weak_mat_tmp13*weak_mat_tmp70;
    const s_t weak_mat_tmp263 = pow_2(weak_mat_tmp70);
    const s_t weak_mat_tmp264 = weak_mat_tmp201 + weak_mat_tmp30;
    const s_t weak_mat_tmp265 = weak_mat_tmp203 + weak_mat_tmp34;
    const s_t weak_mat_tmp266 = weak_mat_tmp48*weak_mat_tmp70;
    const s_t weak_mat_tmp267 = weak_mat_tmp13*weak_mat_tmp71;
    const s_t weak_mat_tmp268 = weak_mat_tmp111*weak_mat_tmp70;
    const s_t weak_mat_tmp269 = weak_mat_tmp48*weak_mat_tmp77;
    const s_t weak_mat_tmp270 = iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_mat_tmp111*weak_mat_tmp269 + weak_mat_tmp115*weak_mat_tmp266 + weak_mat_tmp268*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp216 - weak_mat_tmp82)) + C10*(weak_mat_tmp111*weak_mat_tmp267 + weak_mat_tmp262*weak_mat_tmp75 + weak_mat_tmp268*weak_mat_tmp29));
    const s_t weak_mat_tmp271 = weak_mat_tmp127*weak_mat_tmp70;
    const s_t weak_mat_tmp272 = iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_mat_tmp127*weak_mat_tmp269 + weak_mat_tmp131*weak_mat_tmp266 + weak_mat_tmp271*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp213 - weak_mat_tmp88)) + C10*(weak_mat_tmp127*weak_mat_tmp267 + weak_mat_tmp262*weak_mat_tmp76 + weak_mat_tmp271*weak_mat_tmp29));
    const s_t weak_mat_tmp273 = weak_mat_tmp6*(weak_mat_tmp187 + weak_mat_tmp99);
    const s_t weak_mat_tmp274 = weak_mat_tmp111*weak_mat_tmp13;
    const s_t weak_mat_tmp275 = pow_2(weak_mat_tmp111);
    const s_t weak_mat_tmp276 = weak_mat_tmp140 + weak_mat_tmp231;
    const s_t weak_mat_tmp277 = weak_mat_tmp111*weak_mat_tmp48;
    const s_t weak_mat_tmp278 = weak_mat_tmp127*weak_mat_tmp13;
    const s_t weak_mat_tmp279 = weak_mat_tmp111*weak_mat_tmp127;
    const s_t weak_mat_tmp280 = weak_mat_tmp127*weak_mat_tmp48;
    const s_t weak_mat_tmp281 = iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_mat_tmp115*weak_mat_tmp280 + weak_mat_tmp131*weak_mat_tmp277 + weak_mat_tmp279*weak_mat_tmp55 + weak_mat_tmp36*(-weak_mat_tmp150 - weak_mat_tmp240)) + C10*(weak_mat_tmp274*weak_mat_tmp76 + weak_mat_tmp278*weak_mat_tmp75 + weak_mat_tmp279*weak_mat_tmp29));
    const s_t weak_mat_tmp282 = weak_mat_tmp6*(-weak_mat_tmp122 - weak_mat_tmp166);
    const s_t weak_mat_tmp283 = pow_2(weak_mat_tmp127);
    const s_t material0 = trial_grad0*(iso_history_qp[0] + iso_relaxation_gamma*(C01*(weak_mat_tmp15*weak_mat_tmp55 + s_t(2)*weak_mat_tmp3*weak_mat_tmp49 + weak_mat_tmp36*(weak_mat_tmp32 + weak_mat_tmp35)) + C10*(s_t(4)*weak_mat_tmp10*weak_mat_tmp14 + weak_mat_tmp12 + weak_mat_tmp15*weak_mat_tmp29)) + weak_mat_tmp3*weak_mat_tmp7) + trial_grad1*(weak_mat_tmp7*weak_mat_tmp79 + weak_mat_tmp84) + trial_grad2*(weak_mat_tmp7*weak_mat_tmp86 + weak_mat_tmp91) + trial_grad3*(weak_mat_tmp57*weak_mat_tmp7 + weak_mat_tmp67) + trial_grad4*(weak_mat_tmp119*weak_mat_tmp7 + weak_mat_tmp126) + trial_grad5*(weak_mat_tmp109 + weak_mat_tmp7*weak_mat_tmp94) + trial_grad6*(weak_mat_tmp7*weak_mat_tmp70 + weak_mat_tmp78) + trial_grad7*(weak_mat_tmp111*weak_mat_tmp7 + weak_mat_tmp118) + trial_grad8*(weak_mat_tmp127*weak_mat_tmp7 + weak_mat_tmp134);
    const s_t material1 = trial_grad0*(weak_mat_tmp137*weak_mat_tmp3 + weak_mat_tmp84) + trial_grad1*(iso_history_qp[1] + iso_relaxation_gamma*(C01*(weak_mat_tmp139*weak_mat_tmp55 + s_t(2)*weak_mat_tmp143*weak_mat_tmp79 + weak_mat_tmp36*(weak_mat_tmp142 + weak_mat_tmp32)) + C10*(s_t(4)*gu1*weak_mat_tmp138 + weak_mat_tmp12 + weak_mat_tmp139*weak_mat_tmp29)) + weak_mat_tmp137*weak_mat_tmp79) + trial_grad2*(weak_mat_tmp137*weak_mat_tmp86 + weak_mat_tmp152) + trial_grad3*(weak_mat_tmp137*weak_mat_tmp57 + weak_mat_tmp167) + trial_grad4*(weak_mat_tmp119*weak_mat_tmp137 + weak_mat_tmp155) + trial_grad5*(weak_mat_tmp137*weak_mat_tmp94 + weak_mat_tmp161) + trial_grad6*(weak_mat_tmp137*weak_mat_tmp70 + weak_mat_tmp164) + trial_grad7*(weak_mat_tmp111*weak_mat_tmp137 + weak_mat_tmp148) + trial_grad8*(weak_mat_tmp127*weak_mat_tmp137 + weak_mat_tmp173);
    const s_t material2 = trial_grad0*(weak_mat_tmp176*weak_mat_tmp3 + weak_mat_tmp91) + trial_grad1*(weak_mat_tmp152 + weak_mat_tmp176*weak_mat_tmp79) + trial_grad2*(iso_history_qp[2] + iso_relaxation_gamma*(C01*(weak_mat_tmp178*weak_mat_tmp55 + s_t(2)*weak_mat_tmp179*weak_mat_tmp86 + weak_mat_tmp36*(weak_mat_tmp142 + weak_mat_tmp35)) + C10*(s_t(4)*gu2*weak_mat_tmp177 + weak_mat_tmp12 + weak_mat_tmp178*weak_mat_tmp29)) + weak_mat_tmp176*weak_mat_tmp86) + trial_grad3*(weak_mat_tmp176*weak_mat_tmp57 + weak_mat_tmp188) + trial_grad4*(weak_mat_tmp119*weak_mat_tmp176 + weak_mat_tmp197) + trial_grad5*(weak_mat_tmp176*weak_mat_tmp94 + weak_mat_tmp183) + trial_grad6*(weak_mat_tmp176*weak_mat_tmp70 + weak_mat_tmp194) + trial_grad7*(weak_mat_tmp111*weak_mat_tmp176 + weak_mat_tmp191) + trial_grad8*(weak_mat_tmp127*weak_mat_tmp176 + weak_mat_tmp185);
    const s_t material3 = trial_grad0*(weak_mat_tmp198*weak_mat_tmp3 + weak_mat_tmp67) + trial_grad1*(weak_mat_tmp167 + weak_mat_tmp198*weak_mat_tmp79) + trial_grad2*(weak_mat_tmp188 + weak_mat_tmp198*weak_mat_tmp86) + trial_grad3*(iso_history_qp[0] + iso_relaxation_gamma*(C01*(weak_mat_tmp200*weak_mat_tmp55 + s_t(2)*weak_mat_tmp205*weak_mat_tmp65 + weak_mat_tmp36*(weak_mat_tmp202 + weak_mat_tmp204)) + C10*(s_t(4)*gu3*weak_mat_tmp199 + weak_mat_tmp12 + weak_mat_tmp200*weak_mat_tmp29)) + weak_mat_tmp198*weak_mat_tmp57) + trial_grad4*(weak_mat_tmp119*weak_mat_tmp198 + weak_mat_tmp217) + trial_grad5*(weak_mat_tmp198*weak_mat_tmp94 + weak_mat_tmp214) + trial_grad6*(weak_mat_tmp198*weak_mat_tmp70 + weak_mat_tmp211) + trial_grad7*(weak_mat_tmp111*weak_mat_tmp198 + weak_mat_tmp222) + trial_grad8*(weak_mat_tmp127*weak_mat_tmp198 + weak_mat_tmp227);
    const s_t material4 = trial_grad0*(weak_mat_tmp126 + weak_mat_tmp228*weak_mat_tmp3) + trial_grad1*(weak_mat_tmp155 + weak_mat_tmp228*weak_mat_tmp79) + trial_grad2*(weak_mat_tmp197 + weak_mat_tmp228*weak_mat_tmp86) + trial_grad3*(weak_mat_tmp217 + weak_mat_tmp228*weak_mat_tmp57) + trial_grad4*(iso_history_qp[1] + iso_relaxation_gamma*(C01*(s_t(2)*weak_mat_tmp119*weak_mat_tmp233 + weak_mat_tmp230*weak_mat_tmp55 + weak_mat_tmp36*(weak_mat_tmp202 + weak_mat_tmp232)) + C10*(s_t(4)*weak_mat_tmp1*weak_mat_tmp229 + weak_mat_tmp12 + weak_mat_tmp230*weak_mat_tmp29)) + weak_mat_tmp119*weak_mat_tmp228) + trial_grad5*(weak_mat_tmp228*weak_mat_tmp94 + weak_mat_tmp241) + trial_grad6*(weak_mat_tmp228*weak_mat_tmp70 + weak_mat_tmp243) + trial_grad7*(weak_mat_tmp111*weak_mat_tmp228 + weak_mat_tmp238) + trial_grad8*(weak_mat_tmp127*weak_mat_tmp228 + weak_mat_tmp248);
    const s_t material5 = trial_grad0*(weak_mat_tmp109 + weak_mat_tmp249*weak_mat_tmp3) + trial_grad1*(weak_mat_tmp161 + weak_mat_tmp249*weak_mat_tmp79) + trial_grad2*(weak_mat_tmp183 + weak_mat_tmp249*weak_mat_tmp86) + trial_grad3*(weak_mat_tmp214 + weak_mat_tmp249*weak_mat_tmp57) + trial_grad4*(weak_mat_tmp119*weak_mat_tmp249 + weak_mat_tmp241) + trial_grad5*(iso_history_qp[2] + iso_relaxation_gamma*(C01*(weak_mat_tmp251*weak_mat_tmp55 + s_t(2)*weak_mat_tmp252*weak_mat_tmp94 + weak_mat_tmp36*(weak_mat_tmp204 + weak_mat_tmp232)) + C10*(s_t(4)*gu5*weak_mat_tmp250 + weak_mat_tmp12 + weak_mat_tmp251*weak_mat_tmp29)) + weak_mat_tmp249*weak_mat_tmp94) + trial_grad6*(weak_mat_tmp249*weak_mat_tmp70 + weak_mat_tmp258) + trial_grad7*(weak_mat_tmp111*weak_mat_tmp249 + weak_mat_tmp260) + trial_grad8*(weak_mat_tmp127*weak_mat_tmp249 + weak_mat_tmp256);
    const s_t material6 = trial_grad0*(weak_mat_tmp261*weak_mat_tmp3 + weak_mat_tmp78) + trial_grad1*(weak_mat_tmp164 + weak_mat_tmp261*weak_mat_tmp79) + trial_grad2*(weak_mat_tmp194 + weak_mat_tmp261*weak_mat_tmp86) + trial_grad3*(weak_mat_tmp211 + weak_mat_tmp261*weak_mat_tmp57) + trial_grad4*(weak_mat_tmp119*weak_mat_tmp261 + weak_mat_tmp243) + trial_grad5*(weak_mat_tmp258 + weak_mat_tmp261*weak_mat_tmp94) + trial_grad6*(iso_history_qp[0] + iso_relaxation_gamma*(C01*(weak_mat_tmp263*weak_mat_tmp55 + s_t(2)*weak_mat_tmp266*weak_mat_tmp77 + weak_mat_tmp36*(weak_mat_tmp264 + weak_mat_tmp265)) + C10*(s_t(4)*gu6*weak_mat_tmp262 + weak_mat_tmp12 + weak_mat_tmp263*weak_mat_tmp29)) + weak_mat_tmp261*weak_mat_tmp70) + trial_grad7*(weak_mat_tmp111*weak_mat_tmp261 + weak_mat_tmp270) + trial_grad8*(weak_mat_tmp127*weak_mat_tmp261 + weak_mat_tmp272);
    const s_t material7 = trial_grad0*(weak_mat_tmp118 + weak_mat_tmp273*weak_mat_tmp3) + trial_grad1*(weak_mat_tmp148 + weak_mat_tmp273*weak_mat_tmp79) + trial_grad2*(weak_mat_tmp191 + weak_mat_tmp273*weak_mat_tmp86) + trial_grad3*(weak_mat_tmp222 + weak_mat_tmp273*weak_mat_tmp57) + trial_grad4*(weak_mat_tmp119*weak_mat_tmp273 + weak_mat_tmp238) + trial_grad5*(weak_mat_tmp260 + weak_mat_tmp273*weak_mat_tmp94) + trial_grad6*(weak_mat_tmp270 + weak_mat_tmp273*weak_mat_tmp70) + trial_grad7*(iso_history_qp[1] + iso_relaxation_gamma*(C01*(s_t(2)*weak_mat_tmp115*weak_mat_tmp277 + weak_mat_tmp275*weak_mat_tmp55 + weak_mat_tmp36*(weak_mat_tmp264 + weak_mat_tmp276)) + C10*(s_t(4)*gu7*weak_mat_tmp274 + weak_mat_tmp12 + weak_mat_tmp275*weak_mat_tmp29)) + weak_mat_tmp111*weak_mat_tmp273) + trial_grad8*(weak_mat_tmp127*weak_mat_tmp273 + weak_mat_tmp281);
    const s_t material8 = trial_grad0*(weak_mat_tmp134 + weak_mat_tmp282*weak_mat_tmp3) + trial_grad1*(weak_mat_tmp173 + weak_mat_tmp282*weak_mat_tmp79) + trial_grad2*(weak_mat_tmp185 + weak_mat_tmp282*weak_mat_tmp86) + trial_grad3*(weak_mat_tmp227 + weak_mat_tmp282*weak_mat_tmp57) + trial_grad4*(weak_mat_tmp119*weak_mat_tmp282 + weak_mat_tmp248) + trial_grad5*(weak_mat_tmp256 + weak_mat_tmp282*weak_mat_tmp94) + trial_grad6*(weak_mat_tmp272 + weak_mat_tmp282*weak_mat_tmp70) + trial_grad7*(weak_mat_tmp111*weak_mat_tmp282 + weak_mat_tmp281) + trial_grad8*(iso_history_qp[2] + iso_relaxation_gamma*(C01*(s_t(2)*weak_mat_tmp131*weak_mat_tmp280 + weak_mat_tmp283*weak_mat_tmp55 + weak_mat_tmp36*(weak_mat_tmp265 + weak_mat_tmp276)) + C10*(weak_mat_tmp12 + s_t(4)*weak_mat_tmp2*weak_mat_tmp278 + weak_mat_tmp283*weak_mat_tmp29)) + weak_mat_tmp127*weak_mat_tmp282);
    const s_t loperand0 = qw * (material0 * adj_lane0 + material1 * adj_lane1 + material2 * adj_lane2);
    const s_t loperand1 = qw * (material0 * adj_lane3 + material1 * adj_lane4 + material2 * adj_lane5);
    const s_t loperand2 = qw * (material0 * adj_lane6 + material1 * adj_lane7 + material2 * adj_lane8);
    const s_t loperand3 = qw * (material3 * adj_lane0 + material4 * adj_lane1 + material5 * adj_lane2);
    const s_t loperand4 = qw * (material3 * adj_lane3 + material4 * adj_lane4 + material5 * adj_lane5);
    const s_t loperand5 = qw * (material3 * adj_lane6 + material4 * adj_lane7 + material5 * adj_lane8);
    const s_t loperand6 = qw * (material6 * adj_lane0 + material7 * adj_lane1 + material8 * adj_lane2);
    const s_t loperand7 = qw * (material6 * adj_lane3 + material7 * adj_lane4 + material8 * adj_lane5);
    const s_t loperand8 = qw * (material6 * adj_lane6 + material7 * adj_lane7 + material8 * adj_lane8);
      out_streams[0][lane] += -(loperand0) - loperand1 - loperand2;
      out_streams[1][lane] += -(loperand3) - loperand4 - loperand5;
      out_streams[2][lane] += -(loperand6) - loperand7 - loperand8;
      out_streams[3][lane] += loperand0;
      out_streams[4][lane] += loperand3;
      out_streams[5][lane] += loperand6;
      out_streams[6][lane] += loperand1;
      out_streams[7][lane] += loperand4;
      out_streams[8][lane] += loperand7;
      out_streams[9][lane] += loperand2;
      out_streams[10][lane] += loperand5;
      out_streams[11][lane] += loperand8;
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void continued_mooney_rivlin_prony_d3_simplex_quadrature_output_block(
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
        const s_t *const RSTR grad_ref_x,
        const s_t *const RSTR grad_ref_y,
        const s_t *const RSTR grad_ref_z,
        const s_t C01,
        const s_t C10,
        const s_t Jc,
        const s_t Jmin,
        s_t *const RSTR iso_stress,
        const ptrdiff_t qp_stride,
        const s_t *const RSTR u_streams[NS * 3]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
    for (int q = 0; q < NQ; ++q) {
      s_t *const RSTR iso_stress_q = iso_stress + q * qp_stride;
      s_t gu_ref0_values[VS];
      s_t gu_ref1_values[VS];
      s_t gu_ref2_values[VS];
      s_t gu_ref3_values[VS];
      s_t gu_ref4_values[VS];
      s_t gu_ref5_values[VS];
      s_t gu_ref6_values[VS];
      s_t gu_ref7_values[VS];
      s_t gu_ref8_values[VS];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        gu_ref0_values[lane] = s_t(0);
        gu_ref1_values[lane] = s_t(0);
        gu_ref2_values[lane] = s_t(0);
        gu_ref3_values[lane] = s_t(0);
        gu_ref4_values[lane] = s_t(0);
        gu_ref5_values[lane] = s_t(0);
        gu_ref6_values[lane] = s_t(0);
        gu_ref7_values[lane] = s_t(0);
        gu_ref8_values[lane] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        const s_t gref0 = grad_ref_x[q * NS + shape];
        const s_t gref1 = grad_ref_y[q * NS + shape];
        const s_t gref2 = grad_ref_z[q * NS + shape];
        const s_t *const RSTR u_shape0 = u_streams[3 * shape];
        const s_t *const RSTR u_shape1 = u_streams[3 * shape + 1];
        const s_t *const RSTR u_shape2 = u_streams[3 * shape + 2];
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref0_values[lane] += u_shape0[lane] * gref0;
          gu_ref1_values[lane] += u_shape0[lane] * gref1;
          gu_ref2_values[lane] += u_shape0[lane] * gref2;
          gu_ref3_values[lane] += u_shape1[lane] * gref0;
          gu_ref4_values[lane] += u_shape1[lane] * gref1;
          gu_ref5_values[lane] += u_shape1[lane] * gref2;
          gu_ref6_values[lane] += u_shape2[lane] * gref0;
          gu_ref7_values[lane] += u_shape2[lane] * gref1;
          gu_ref8_values[lane] += u_shape2[lane] * gref2;
        }
      }
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
      const s_t adj_lane0 = adj0[goff];
      const s_t adj_lane1 = adj1[goff];
      const s_t adj_lane2 = adj2[goff];
      const s_t adj_lane3 = adj3[goff];
      const s_t adj_lane4 = adj4[goff];
      const s_t adj_lane5 = adj5[goff];
      const s_t adj_lane6 = adj6[goff];
      const s_t adj_lane7 = adj7[goff];
      const s_t adj_lane8 = adj8[goff];
      const s_t det_lane0 = det0[goff];
      const s_t gu_ref0 = gu_ref0_values[lane];
      const s_t gu_ref1 = gu_ref1_values[lane];
      const s_t gu_ref2 = gu_ref2_values[lane];
      const s_t gu_ref3 = gu_ref3_values[lane];
      const s_t gu_ref4 = gu_ref4_values[lane];
      const s_t gu_ref5 = gu_ref5_values[lane];
      const s_t gu_ref6 = gu_ref6_values[lane];
      const s_t gu_ref7 = gu_ref7_values[lane];
      const s_t gu_ref8 = gu_ref8_values[lane];
    const s_t idet = s_t(1) / det_lane0;
    const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane3 + gu_ref2 * adj_lane6) * idet;
    const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane4 + gu_ref2 * adj_lane7) * idet;
    const s_t gu2 = (gu_ref0 * adj_lane2 + gu_ref1 * adj_lane5 + gu_ref2 * adj_lane8) * idet;
    const s_t gu3 = (gu_ref3 * adj_lane0 + gu_ref4 * adj_lane3 + gu_ref5 * adj_lane6) * idet;
    const s_t gu4 = (gu_ref3 * adj_lane1 + gu_ref4 * adj_lane4 + gu_ref5 * adj_lane7) * idet;
    const s_t gu5 = (gu_ref3 * adj_lane2 + gu_ref4 * adj_lane5 + gu_ref5 * adj_lane8) * idet;
    const s_t gu6 = (gu_ref6 * adj_lane0 + gu_ref7 * adj_lane3 + gu_ref8 * adj_lane6) * idet;
    const s_t gu7 = (gu_ref6 * adj_lane1 + gu_ref7 * adj_lane4 + gu_ref8 * adj_lane7) * idet;
    const s_t gu8 = (gu_ref6 * adj_lane2 + gu_ref7 * adj_lane5 + gu_ref8 * adj_lane8) * idet;
    const s_t qp_out_tmp0 = gu8 + s_t(1);
    const s_t qp_out_tmp1 = gu1*qp_out_tmp0;
    const s_t qp_out_tmp2 = gu4 + s_t(1);
    const s_t qp_out_tmp3 = gu2*qp_out_tmp2;
    const s_t qp_out_tmp4 = gu0 + s_t(1);
    const s_t qp_out_tmp5 = gu5*gu7;
    const s_t qp_out_tmp6 = gu1*gu5*gu6 + gu2*gu3*gu7 - gu3*qp_out_tmp1 - gu6*qp_out_tmp3 + qp_out_tmp0*qp_out_tmp2*qp_out_tmp4 - qp_out_tmp4*qp_out_tmp5;
    const s_t qp_out_tmp7 = pow_m1(qp_out_tmp6);
    const s_t qp_out_tmp8 = gu1*gu5 - qp_out_tmp3;
    const s_t qp_out_tmp9 = sfem_continued_power<s_t>(qp_out_tmp6, (s_t(2) / s_t(3)), Jc, Jmin);
    const s_t qp_out_tmp10 = s_t(2)*gu6;
    const s_t qp_out_tmp11 = pow_2(gu3) + pow_2(gu6) + pow_2(qp_out_tmp4);
    const s_t qp_out_tmp12 = pow_2(gu1) + pow_2(gu7) + pow_2(qp_out_tmp2);
    const s_t qp_out_tmp13 = pow_2(gu2) + pow_2(gu5) + pow_2(qp_out_tmp0);
    const s_t qp_out_tmp14 = qp_out_tmp11 + qp_out_tmp12 + qp_out_tmp13;
    const s_t qp_out_tmp15 = qp_out_tmp14*sfem_continued_power_d1<s_t>(qp_out_tmp6, (s_t(2) / s_t(3)), Jc, Jmin);
    const s_t qp_out_tmp16 = sfem_continued_power<s_t>(qp_out_tmp6, (s_t(4) / s_t(3)), Jc, Jmin);
    const s_t qp_out_tmp17 = gu1*qp_out_tmp4 + gu3*qp_out_tmp2 + gu6*gu7;
    const s_t qp_out_tmp18 = s_t(2)*qp_out_tmp17;
    const s_t qp_out_tmp19 = gu2*qp_out_tmp4 + gu3*gu5 + gu6*qp_out_tmp0;
    const s_t qp_out_tmp20 = s_t(2)*qp_out_tmp19;
    const s_t qp_out_tmp21 = gu1*gu2 + gu5*qp_out_tmp2 + gu7*qp_out_tmp0;
    const s_t qp_out_tmp22 = (-(s_t(1) / s_t(2))*pow_2(qp_out_tmp11) - (s_t(1) / s_t(2))*pow_2(qp_out_tmp12) - (s_t(1) / s_t(2))*pow_2(qp_out_tmp13) + ((s_t(1) / s_t(2)))*pow_2(qp_out_tmp14) - pow_2(qp_out_tmp17) - pow_2(qp_out_tmp19) - pow_2(qp_out_tmp21))*sfem_continued_power_d1<s_t>(qp_out_tmp6, (s_t(4) / s_t(3)), Jc, Jmin);
    const s_t qp_out_tmp23 = gu2*gu7 - qp_out_tmp1;
    const s_t qp_out_tmp24 = s_t(2)*gu3;
    const s_t qp_out_tmp25 = qp_out_tmp0*qp_out_tmp2 - qp_out_tmp5;
    const s_t qp_out_tmp26 = s_t(2)*qp_out_tmp4;
    const s_t qp_out_tmp27 = gu2*gu3 - gu5*qp_out_tmp4;
    const s_t qp_out_tmp28 = s_t(2)*gu7;
    const s_t qp_out_tmp29 = s_t(2)*qp_out_tmp21;
    const s_t qp_out_tmp30 = C01*(qp_out_tmp16*(-gu6*qp_out_tmp18 + s_t(2)*gu7*qp_out_tmp14 - qp_out_tmp0*qp_out_tmp29 - qp_out_tmp12*qp_out_tmp28) + qp_out_tmp22*qp_out_tmp27) + C10*(qp_out_tmp15*qp_out_tmp27 + qp_out_tmp28*qp_out_tmp9);
    const s_t qp_out_tmp31 = -gu3*qp_out_tmp0 + gu5*gu6;
    const s_t qp_out_tmp32 = s_t(2)*gu1;
    const s_t qp_out_tmp33 = C01*(qp_out_tmp16*(s_t(2)*gu1*qp_out_tmp14 - gu2*qp_out_tmp29 - qp_out_tmp12*qp_out_tmp32 - qp_out_tmp18*qp_out_tmp4) + qp_out_tmp22*qp_out_tmp31) + C10*(qp_out_tmp15*qp_out_tmp31 + qp_out_tmp32*qp_out_tmp9);
    const s_t qp_out_tmp34 = -gu2*gu6 + qp_out_tmp0*qp_out_tmp4;
    const s_t qp_out_tmp35 = s_t(2)*qp_out_tmp2;
    const s_t qp_out_tmp36 = C01*(qp_out_tmp16*(-gu3*qp_out_tmp18 - gu5*qp_out_tmp29 - qp_out_tmp12*qp_out_tmp35 + s_t(2)*qp_out_tmp14*qp_out_tmp2) + qp_out_tmp22*qp_out_tmp34) + C10*(qp_out_tmp15*qp_out_tmp34 + qp_out_tmp35*qp_out_tmp9);
    const s_t qp_out_tmp37 = gu1*gu6 - gu7*qp_out_tmp4;
    const s_t qp_out_tmp38 = s_t(2)*gu5;
    const s_t qp_out_tmp39 = C01*(qp_out_tmp16*(-gu3*qp_out_tmp20 + s_t(2)*gu5*qp_out_tmp14 - qp_out_tmp13*qp_out_tmp38 - qp_out_tmp2*qp_out_tmp29) + qp_out_tmp22*qp_out_tmp37) + C10*(qp_out_tmp15*qp_out_tmp37 + qp_out_tmp38*qp_out_tmp9);
    const s_t qp_out_tmp40 = gu3*gu7 - gu6*qp_out_tmp2;
    const s_t qp_out_tmp41 = s_t(2)*gu2;
    const s_t qp_out_tmp42 = C01*(qp_out_tmp16*(s_t(2)*gu2*qp_out_tmp14 - qp_out_tmp13*qp_out_tmp41 - qp_out_tmp20*qp_out_tmp4 - qp_out_tmp21*qp_out_tmp32) + qp_out_tmp22*qp_out_tmp40) + C10*(qp_out_tmp15*qp_out_tmp40 + qp_out_tmp41*qp_out_tmp9);
    const s_t qp_out_tmp43 = -gu1*gu3 + qp_out_tmp2*qp_out_tmp4;
    const s_t qp_out_tmp44 = s_t(2)*qp_out_tmp0;
    const s_t qp_out_tmp45 = C01*(qp_out_tmp16*(-gu6*qp_out_tmp20 - gu7*qp_out_tmp29 + s_t(2)*qp_out_tmp0*qp_out_tmp14 - qp_out_tmp13*qp_out_tmp44) + qp_out_tmp22*qp_out_tmp43) + C10*(qp_out_tmp15*qp_out_tmp43 + qp_out_tmp44*qp_out_tmp9);
    iso_stress_q[lane] = qp_out_tmp7*(qp_out_tmp23*(C01*(qp_out_tmp16*(s_t(2)*gu3*qp_out_tmp14 - gu5*qp_out_tmp20 - qp_out_tmp11*qp_out_tmp24 - qp_out_tmp18*qp_out_tmp2) + qp_out_tmp22*qp_out_tmp23) + C10*(qp_out_tmp15*qp_out_tmp23 + qp_out_tmp24*qp_out_tmp9)) + qp_out_tmp25*(C01*(qp_out_tmp16*(-gu1*qp_out_tmp18 - gu2*qp_out_tmp20 - qp_out_tmp11*qp_out_tmp26 + s_t(2)*qp_out_tmp14*qp_out_tmp4) + qp_out_tmp22*qp_out_tmp25) + C10*(qp_out_tmp15*qp_out_tmp25 + qp_out_tmp26*qp_out_tmp9)) + qp_out_tmp8*(C01*(qp_out_tmp16*(s_t(2)*gu6*qp_out_tmp14 - gu7*qp_out_tmp18 - qp_out_tmp0*qp_out_tmp20 - qp_out_tmp10*qp_out_tmp11) + qp_out_tmp22*qp_out_tmp8) + C10*(qp_out_tmp10*qp_out_tmp9 + qp_out_tmp15*qp_out_tmp8)));
    iso_stress_q[1 * NQ * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp27*qp_out_tmp30 + qp_out_tmp31*qp_out_tmp33 + qp_out_tmp34*qp_out_tmp36);
    iso_stress_q[2 * NQ * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp37*qp_out_tmp39 + qp_out_tmp40*qp_out_tmp42 + qp_out_tmp43*qp_out_tmp45);
    iso_stress_q[3 * NQ * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp23*qp_out_tmp36 + qp_out_tmp25*qp_out_tmp33 + qp_out_tmp30*qp_out_tmp8);
    iso_stress_q[4 * NQ * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp23*qp_out_tmp39 + qp_out_tmp25*qp_out_tmp42 + qp_out_tmp45*qp_out_tmp8);
    iso_stress_q[5 * NQ * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp27*qp_out_tmp45 + qp_out_tmp31*qp_out_tmp42 + qp_out_tmp34*qp_out_tmp39);
      }
    }
}

template <typename s_t, int NS, int VS>
static SFEM_INLINE void continued_mooney_rivlin_prony_d3_simplex_tet4_quadrature_output_block(
        const int ne,
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
        const s_t C01,
        const s_t C10,
        const s_t Jc,
        const s_t Jmin,
        s_t *const RSTR iso_stress,
        const ptrdiff_t qp_stride,
        const s_t *const RSTR u_streams[NS * 3]
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
      const s_t adj_lane4 = adj4[goff];
      const s_t adj_lane5 = adj5[goff];
      const s_t adj_lane6 = adj6[goff];
      const s_t adj_lane7 = adj7[goff];
      const s_t adj_lane8 = adj8[goff];
      const s_t det_lane0 = det0[goff];
      const s_t gu_ref0 = -(u_streams[0][lane]) + u_streams[3][lane];
      const s_t gu_ref1 = -(u_streams[0][lane]) + u_streams[6][lane];
      const s_t gu_ref2 = -(u_streams[0][lane]) + u_streams[9][lane];
      const s_t gu_ref3 = -(u_streams[1][lane]) + u_streams[4][lane];
      const s_t gu_ref4 = -(u_streams[1][lane]) + u_streams[7][lane];
      const s_t gu_ref5 = -(u_streams[1][lane]) + u_streams[10][lane];
      const s_t gu_ref6 = -(u_streams[2][lane]) + u_streams[5][lane];
      const s_t gu_ref7 = -(u_streams[2][lane]) + u_streams[8][lane];
      const s_t gu_ref8 = -(u_streams[2][lane]) + u_streams[11][lane];
      const s_t idet = s_t(1) / det_lane0;
      const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane3 + gu_ref2 * adj_lane6) * idet;
      const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane4 + gu_ref2 * adj_lane7) * idet;
      const s_t gu2 = (gu_ref0 * adj_lane2 + gu_ref1 * adj_lane5 + gu_ref2 * adj_lane8) * idet;
      const s_t gu3 = (gu_ref3 * adj_lane0 + gu_ref4 * adj_lane3 + gu_ref5 * adj_lane6) * idet;
      const s_t gu4 = (gu_ref3 * adj_lane1 + gu_ref4 * adj_lane4 + gu_ref5 * adj_lane7) * idet;
      const s_t gu5 = (gu_ref3 * adj_lane2 + gu_ref4 * adj_lane5 + gu_ref5 * adj_lane8) * idet;
      const s_t gu6 = (gu_ref6 * adj_lane0 + gu_ref7 * adj_lane3 + gu_ref8 * adj_lane6) * idet;
      const s_t gu7 = (gu_ref6 * adj_lane1 + gu_ref7 * adj_lane4 + gu_ref8 * adj_lane7) * idet;
      const s_t gu8 = (gu_ref6 * adj_lane2 + gu_ref7 * adj_lane5 + gu_ref8 * adj_lane8) * idet;
    const s_t qp_out_tmp0 = gu8 + s_t(1);
    const s_t qp_out_tmp1 = gu1*qp_out_tmp0;
    const s_t qp_out_tmp2 = gu4 + s_t(1);
    const s_t qp_out_tmp3 = gu2*qp_out_tmp2;
    const s_t qp_out_tmp4 = gu0 + s_t(1);
    const s_t qp_out_tmp5 = gu5*gu7;
    const s_t qp_out_tmp6 = gu1*gu5*gu6 + gu2*gu3*gu7 - gu3*qp_out_tmp1 - gu6*qp_out_tmp3 + qp_out_tmp0*qp_out_tmp2*qp_out_tmp4 - qp_out_tmp4*qp_out_tmp5;
    const s_t qp_out_tmp7 = pow_m1(qp_out_tmp6);
    const s_t qp_out_tmp8 = gu1*gu5 - qp_out_tmp3;
    const s_t qp_out_tmp9 = sfem_continued_power<s_t>(qp_out_tmp6, (s_t(2) / s_t(3)), Jc, Jmin);
    const s_t qp_out_tmp10 = s_t(2)*gu6;
    const s_t qp_out_tmp11 = pow_2(gu3) + pow_2(gu6) + pow_2(qp_out_tmp4);
    const s_t qp_out_tmp12 = pow_2(gu1) + pow_2(gu7) + pow_2(qp_out_tmp2);
    const s_t qp_out_tmp13 = pow_2(gu2) + pow_2(gu5) + pow_2(qp_out_tmp0);
    const s_t qp_out_tmp14 = qp_out_tmp11 + qp_out_tmp12 + qp_out_tmp13;
    const s_t qp_out_tmp15 = qp_out_tmp14*sfem_continued_power_d1<s_t>(qp_out_tmp6, (s_t(2) / s_t(3)), Jc, Jmin);
    const s_t qp_out_tmp16 = sfem_continued_power<s_t>(qp_out_tmp6, (s_t(4) / s_t(3)), Jc, Jmin);
    const s_t qp_out_tmp17 = gu1*qp_out_tmp4 + gu3*qp_out_tmp2 + gu6*gu7;
    const s_t qp_out_tmp18 = s_t(2)*qp_out_tmp17;
    const s_t qp_out_tmp19 = gu2*qp_out_tmp4 + gu3*gu5 + gu6*qp_out_tmp0;
    const s_t qp_out_tmp20 = s_t(2)*qp_out_tmp19;
    const s_t qp_out_tmp21 = gu1*gu2 + gu5*qp_out_tmp2 + gu7*qp_out_tmp0;
    const s_t qp_out_tmp22 = (-(s_t(1) / s_t(2))*pow_2(qp_out_tmp11) - (s_t(1) / s_t(2))*pow_2(qp_out_tmp12) - (s_t(1) / s_t(2))*pow_2(qp_out_tmp13) + ((s_t(1) / s_t(2)))*pow_2(qp_out_tmp14) - pow_2(qp_out_tmp17) - pow_2(qp_out_tmp19) - pow_2(qp_out_tmp21))*sfem_continued_power_d1<s_t>(qp_out_tmp6, (s_t(4) / s_t(3)), Jc, Jmin);
    const s_t qp_out_tmp23 = gu2*gu7 - qp_out_tmp1;
    const s_t qp_out_tmp24 = s_t(2)*gu3;
    const s_t qp_out_tmp25 = qp_out_tmp0*qp_out_tmp2 - qp_out_tmp5;
    const s_t qp_out_tmp26 = s_t(2)*qp_out_tmp4;
    const s_t qp_out_tmp27 = gu2*gu3 - gu5*qp_out_tmp4;
    const s_t qp_out_tmp28 = s_t(2)*gu7;
    const s_t qp_out_tmp29 = s_t(2)*qp_out_tmp21;
    const s_t qp_out_tmp30 = C01*(qp_out_tmp16*(-gu6*qp_out_tmp18 + s_t(2)*gu7*qp_out_tmp14 - qp_out_tmp0*qp_out_tmp29 - qp_out_tmp12*qp_out_tmp28) + qp_out_tmp22*qp_out_tmp27) + C10*(qp_out_tmp15*qp_out_tmp27 + qp_out_tmp28*qp_out_tmp9);
    const s_t qp_out_tmp31 = -gu3*qp_out_tmp0 + gu5*gu6;
    const s_t qp_out_tmp32 = s_t(2)*gu1;
    const s_t qp_out_tmp33 = C01*(qp_out_tmp16*(s_t(2)*gu1*qp_out_tmp14 - gu2*qp_out_tmp29 - qp_out_tmp12*qp_out_tmp32 - qp_out_tmp18*qp_out_tmp4) + qp_out_tmp22*qp_out_tmp31) + C10*(qp_out_tmp15*qp_out_tmp31 + qp_out_tmp32*qp_out_tmp9);
    const s_t qp_out_tmp34 = -gu2*gu6 + qp_out_tmp0*qp_out_tmp4;
    const s_t qp_out_tmp35 = s_t(2)*qp_out_tmp2;
    const s_t qp_out_tmp36 = C01*(qp_out_tmp16*(-gu3*qp_out_tmp18 - gu5*qp_out_tmp29 - qp_out_tmp12*qp_out_tmp35 + s_t(2)*qp_out_tmp14*qp_out_tmp2) + qp_out_tmp22*qp_out_tmp34) + C10*(qp_out_tmp15*qp_out_tmp34 + qp_out_tmp35*qp_out_tmp9);
    const s_t qp_out_tmp37 = gu1*gu6 - gu7*qp_out_tmp4;
    const s_t qp_out_tmp38 = s_t(2)*gu5;
    const s_t qp_out_tmp39 = C01*(qp_out_tmp16*(-gu3*qp_out_tmp20 + s_t(2)*gu5*qp_out_tmp14 - qp_out_tmp13*qp_out_tmp38 - qp_out_tmp2*qp_out_tmp29) + qp_out_tmp22*qp_out_tmp37) + C10*(qp_out_tmp15*qp_out_tmp37 + qp_out_tmp38*qp_out_tmp9);
    const s_t qp_out_tmp40 = gu3*gu7 - gu6*qp_out_tmp2;
    const s_t qp_out_tmp41 = s_t(2)*gu2;
    const s_t qp_out_tmp42 = C01*(qp_out_tmp16*(s_t(2)*gu2*qp_out_tmp14 - qp_out_tmp13*qp_out_tmp41 - qp_out_tmp20*qp_out_tmp4 - qp_out_tmp21*qp_out_tmp32) + qp_out_tmp22*qp_out_tmp40) + C10*(qp_out_tmp15*qp_out_tmp40 + qp_out_tmp41*qp_out_tmp9);
    const s_t qp_out_tmp43 = -gu1*gu3 + qp_out_tmp2*qp_out_tmp4;
    const s_t qp_out_tmp44 = s_t(2)*qp_out_tmp0;
    const s_t qp_out_tmp45 = C01*(qp_out_tmp16*(-gu6*qp_out_tmp20 - gu7*qp_out_tmp29 + s_t(2)*qp_out_tmp0*qp_out_tmp14 - qp_out_tmp13*qp_out_tmp44) + qp_out_tmp22*qp_out_tmp43) + C10*(qp_out_tmp15*qp_out_tmp43 + qp_out_tmp44*qp_out_tmp9);
    iso_stress_q[lane] = qp_out_tmp7*(qp_out_tmp23*(C01*(qp_out_tmp16*(s_t(2)*gu3*qp_out_tmp14 - gu5*qp_out_tmp20 - qp_out_tmp11*qp_out_tmp24 - qp_out_tmp18*qp_out_tmp2) + qp_out_tmp22*qp_out_tmp23) + C10*(qp_out_tmp15*qp_out_tmp23 + qp_out_tmp24*qp_out_tmp9)) + qp_out_tmp25*(C01*(qp_out_tmp16*(-gu1*qp_out_tmp18 - gu2*qp_out_tmp20 - qp_out_tmp11*qp_out_tmp26 + s_t(2)*qp_out_tmp14*qp_out_tmp4) + qp_out_tmp22*qp_out_tmp25) + C10*(qp_out_tmp15*qp_out_tmp25 + qp_out_tmp26*qp_out_tmp9)) + qp_out_tmp8*(C01*(qp_out_tmp16*(s_t(2)*gu6*qp_out_tmp14 - gu7*qp_out_tmp18 - qp_out_tmp0*qp_out_tmp20 - qp_out_tmp10*qp_out_tmp11) + qp_out_tmp22*qp_out_tmp8) + C10*(qp_out_tmp10*qp_out_tmp9 + qp_out_tmp15*qp_out_tmp8)));
    iso_stress_q[1 * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp27*qp_out_tmp30 + qp_out_tmp31*qp_out_tmp33 + qp_out_tmp34*qp_out_tmp36);
    iso_stress_q[2 * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp37*qp_out_tmp39 + qp_out_tmp40*qp_out_tmp42 + qp_out_tmp43*qp_out_tmp45);
    iso_stress_q[3 * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp23*qp_out_tmp36 + qp_out_tmp25*qp_out_tmp33 + qp_out_tmp30*qp_out_tmp8);
    iso_stress_q[4 * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp23*qp_out_tmp39 + qp_out_tmp25*qp_out_tmp42 + qp_out_tmp45*qp_out_tmp8);
    iso_stress_q[5 * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp27*qp_out_tmp45 + qp_out_tmp31*qp_out_tmp42 + qp_out_tmp34*qp_out_tmp39);
      }
    }
}

} // namespace codegen
} // namespace sfem

#endif
