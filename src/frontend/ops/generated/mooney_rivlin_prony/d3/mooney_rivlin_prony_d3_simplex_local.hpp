#ifndef MOONEY_RIVLIN_PRONY_D3_SIMPLEX_LOCAL_HPP
#define MOONEY_RIVLIN_PRONY_D3_SIMPLEX_LOCAL_HPP
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
static SFEM_INLINE void mooney_rivlin_prony_d3_simplex_objective_block(
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
    value[step * value_stride + lane] += qw * det_lane0 * (((s_t(1) / s_t(2)))*K*pow_2(gu0*gu8 - gu0*weak_obj_tmp12 + gu0 + gu1*gu5*gu6 + gu2*gu3*gu7 + gu4*gu8 - gu4*weak_obj_tmp11 + gu4 - gu8*weak_obj_tmp10 + gu8*weak_obj_tmp9 + gu8 - weak_obj_tmp10 - weak_obj_tmp11 - weak_obj_tmp12 + weak_obj_tmp9) + ((s_t(1) / s_t(2)))*iso_history_qp[0]*(pow_2(gu0) + s_t(2)*gu0 + weak_obj_tmp0) + ((s_t(1) / s_t(2)))*iso_history_qp[1]*(pow_2(gu4) + s_t(2)*gu4 + weak_obj_tmp1) + ((s_t(1) / s_t(2)))*iso_history_qp[2]*(pow_2(gu8) + s_t(2)*gu8 + weak_obj_tmp2) + iso_history_qp[3]*weak_obj_tmp5 + iso_history_qp[4]*weak_obj_tmp7 + iso_history_qp[5]*weak_obj_tmp8 + iso_relaxation_gamma*(C01*(s_t(-3) + (-(s_t(1) / s_t(2))*pow_2(weak_obj_tmp13) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp14) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp15) + ((s_t(1) / s_t(2)))*pow_2(weak_obj_tmp16) - pow_2(weak_obj_tmp5) - pow_2(weak_obj_tmp7) - pow_2(weak_obj_tmp8))/pow(weak_obj_tmp17, (s_t(4) / s_t(3)))) + C10*(weak_obj_tmp16/pow(weak_obj_tmp17, (s_t(2) / s_t(3))) + s_t(-3))));
        }
      }
    }
}

template <typename s_t, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_prony_d3_simplex_tet4_objective_block(
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
    value[step * value_stride + lane] += qw * det_lane0 * (((s_t(1) / s_t(2)))*K*pow_2(gu0*gu8 - gu0*weak_obj_tmp12 + gu0 + gu1*gu5*gu6 + gu2*gu3*gu7 + gu4*gu8 - gu4*weak_obj_tmp11 + gu4 - gu8*weak_obj_tmp10 + gu8*weak_obj_tmp9 + gu8 - weak_obj_tmp10 - weak_obj_tmp11 - weak_obj_tmp12 + weak_obj_tmp9) + ((s_t(1) / s_t(2)))*iso_history_qp[0]*(pow_2(gu0) + s_t(2)*gu0 + weak_obj_tmp0) + ((s_t(1) / s_t(2)))*iso_history_qp[1]*(pow_2(gu4) + s_t(2)*gu4 + weak_obj_tmp1) + ((s_t(1) / s_t(2)))*iso_history_qp[2]*(pow_2(gu8) + s_t(2)*gu8 + weak_obj_tmp2) + iso_history_qp[3]*weak_obj_tmp5 + iso_history_qp[4]*weak_obj_tmp7 + iso_history_qp[5]*weak_obj_tmp8 + iso_relaxation_gamma*(C01*(s_t(-3) + (-(s_t(1) / s_t(2))*pow_2(weak_obj_tmp13) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp14) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp15) + ((s_t(1) / s_t(2)))*pow_2(weak_obj_tmp16) - pow_2(weak_obj_tmp5) - pow_2(weak_obj_tmp7) - pow_2(weak_obj_tmp8))/pow(weak_obj_tmp17, (s_t(4) / s_t(3)))) + C10*(weak_obj_tmp16/pow(weak_obj_tmp17, (s_t(2) / s_t(3))) + s_t(-3))));
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_prony_d3_simplex_gradient_block(
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
    const s_t weak_mat_tmp7 = gu5*gu6;
    const s_t weak_mat_tmp8 = gu3*gu7;
    const s_t weak_mat_tmp9 = ((s_t(1) / s_t(2)))*K*(gu0*gu8 - gu0*weak_mat_tmp1 + gu0 + gu1*weak_mat_tmp7 + gu2*weak_mat_tmp8 + gu4*gu8 - gu4*weak_mat_tmp6 + gu4 + gu8*weak_mat_tmp4 - gu8*weak_mat_tmp5 + gu8 - weak_mat_tmp1 + weak_mat_tmp4 - weak_mat_tmp5 - weak_mat_tmp6);
    const s_t weak_mat_tmp10 = gu1*gu5*gu6 + gu2*gu3*gu7 - weak_mat_tmp0*weak_mat_tmp1 + weak_mat_tmp0*weak_mat_tmp2*weak_mat_tmp3 - weak_mat_tmp2*weak_mat_tmp6 - weak_mat_tmp3*weak_mat_tmp5;
    const s_t weak_mat_tmp11 = pow(weak_mat_tmp10, (s_t(-2) / s_t(3)));
    const s_t weak_mat_tmp12 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp13 = weak_mat_tmp2*weak_mat_tmp3;
    const s_t weak_mat_tmp14 = pow_2(gu3) + pow_2(gu6) + pow_2(weak_mat_tmp0);
    const s_t weak_mat_tmp15 = pow_2(gu1) + pow_2(gu7) + pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp16 = pow_2(gu2) + pow_2(gu5) + pow_2(weak_mat_tmp3);
    const s_t weak_mat_tmp17 = weak_mat_tmp14 + weak_mat_tmp15 + weak_mat_tmp16;
    const s_t weak_mat_tmp18 = weak_mat_tmp17/pow(weak_mat_tmp10, (s_t(5) / s_t(3)));
    const s_t weak_mat_tmp19 = pow(weak_mat_tmp10, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp20 = gu1*weak_mat_tmp0 + gu3*weak_mat_tmp2 + gu6*gu7;
    const s_t weak_mat_tmp21 = s_t(2)*gu1;
    const s_t weak_mat_tmp22 = gu2*weak_mat_tmp0 + gu3*gu5 + gu6*weak_mat_tmp3;
    const s_t weak_mat_tmp23 = s_t(2)*gu2;
    const s_t weak_mat_tmp24 = gu1*gu2 + gu5*weak_mat_tmp2 + gu7*weak_mat_tmp3;
    const s_t weak_mat_tmp25 = (-(s_t(1) / s_t(2))*pow_2(weak_mat_tmp14) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp15) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp16) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp17) - pow_2(weak_mat_tmp20) - pow_2(weak_mat_tmp22) - pow_2(weak_mat_tmp24))/pow(weak_mat_tmp10, (s_t(7) / s_t(3)));
    const s_t weak_mat_tmp26 = gu3*weak_mat_tmp3;
    const s_t weak_mat_tmp27 = s_t(2)*gu3;
    const s_t weak_mat_tmp28 = gu2*gu7;
    const s_t weak_mat_tmp29 = gu1*weak_mat_tmp3;
    const s_t weak_mat_tmp30 = s_t(2)*gu5;
    const s_t weak_mat_tmp31 = s_t(2)*weak_mat_tmp2;
    const s_t weak_mat_tmp32 = weak_mat_tmp0*weak_mat_tmp3;
    const s_t weak_mat_tmp33 = gu1*gu6;
    const s_t weak_mat_tmp34 = s_t(2)*gu6;
    const s_t weak_mat_tmp35 = gu1*gu5;
    const s_t weak_mat_tmp36 = s_t(2)*gu7;
    const s_t weak_mat_tmp37 = s_t(2)*weak_mat_tmp3;
    const s_t weak_mat_tmp38 = gu2*gu3;
    const s_t weak_mat_tmp39 = weak_mat_tmp0*weak_mat_tmp2;
    const s_t material0 = gu1*iso_history_qp[3] + gu2*iso_history_qp[4] + iso_history_qp[0]*weak_mat_tmp0 + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(s_t(2)*weak_mat_tmp0*weak_mat_tmp17 - weak_mat_tmp12*weak_mat_tmp14 - weak_mat_tmp20*weak_mat_tmp21 - weak_mat_tmp22*weak_mat_tmp23) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*weak_mat_tmp1 - (s_t(4) / s_t(3))*weak_mat_tmp13)) + C10*(weak_mat_tmp11*weak_mat_tmp12 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*weak_mat_tmp1 - (s_t(2) / s_t(3))*weak_mat_tmp13))) + weak_mat_tmp9*(-s_t(2)*weak_mat_tmp1 + s_t(2)*weak_mat_tmp2*weak_mat_tmp3);
    const s_t material1 = gu1*iso_history_qp[1] + gu2*iso_history_qp[5] + iso_history_qp[3]*weak_mat_tmp0 + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(s_t(2)*gu1*weak_mat_tmp17 - weak_mat_tmp12*weak_mat_tmp20 - weak_mat_tmp15*weak_mat_tmp21 - weak_mat_tmp23*weak_mat_tmp24) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*weak_mat_tmp26 - (s_t(4) / s_t(3))*weak_mat_tmp7)) + C10*(weak_mat_tmp11*weak_mat_tmp21 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*weak_mat_tmp26 - (s_t(2) / s_t(3))*weak_mat_tmp7))) + weak_mat_tmp9*(s_t(2)*gu5*gu6 - s_t(2)*weak_mat_tmp26);
    const s_t material2 = gu1*iso_history_qp[5] + gu2*iso_history_qp[2] + iso_history_qp[4]*weak_mat_tmp0 + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(s_t(2)*gu2*weak_mat_tmp17 - weak_mat_tmp12*weak_mat_tmp22 - weak_mat_tmp16*weak_mat_tmp23 - weak_mat_tmp21*weak_mat_tmp24) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu6*weak_mat_tmp2 - (s_t(4) / s_t(3))*weak_mat_tmp8)) + C10*(weak_mat_tmp11*weak_mat_tmp23 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu6*weak_mat_tmp2 - (s_t(2) / s_t(3))*weak_mat_tmp8))) + weak_mat_tmp9*(-s_t(2)*gu6*weak_mat_tmp2 + s_t(2)*weak_mat_tmp8);
    const s_t material3 = gu3*iso_history_qp[0] + gu5*iso_history_qp[4] + iso_history_qp[3]*weak_mat_tmp2 + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(s_t(2)*gu3*weak_mat_tmp17 - weak_mat_tmp14*weak_mat_tmp27 - weak_mat_tmp20*weak_mat_tmp31 - weak_mat_tmp22*weak_mat_tmp30) + weak_mat_tmp25*(-(s_t(4) / s_t(3))*weak_mat_tmp28 + ((s_t(4) / s_t(3)))*weak_mat_tmp29)) + C10*(weak_mat_tmp11*weak_mat_tmp27 + weak_mat_tmp18*(-(s_t(2) / s_t(3))*weak_mat_tmp28 + ((s_t(2) / s_t(3)))*weak_mat_tmp29))) + weak_mat_tmp9*(s_t(2)*gu2*gu7 - weak_mat_tmp21*weak_mat_tmp3);
    const s_t material4 = gu3*iso_history_qp[3] + gu5*iso_history_qp[5] + iso_history_qp[1]*weak_mat_tmp2 + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(-weak_mat_tmp15*weak_mat_tmp31 + s_t(2)*weak_mat_tmp17*weak_mat_tmp2 - weak_mat_tmp20*weak_mat_tmp27 - weak_mat_tmp24*weak_mat_tmp30) + weak_mat_tmp25*(-(s_t(4) / s_t(3))*weak_mat_tmp32 + ((s_t(4) / s_t(3)))*weak_mat_tmp6)) + C10*(weak_mat_tmp11*weak_mat_tmp31 + weak_mat_tmp18*(-(s_t(2) / s_t(3))*weak_mat_tmp32 + ((s_t(2) / s_t(3)))*weak_mat_tmp6))) + weak_mat_tmp9*(s_t(2)*weak_mat_tmp0*weak_mat_tmp3 - s_t(2)*weak_mat_tmp6);
    const s_t material5 = gu3*iso_history_qp[4] + gu5*iso_history_qp[2] + iso_history_qp[5]*weak_mat_tmp2 + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(s_t(2)*gu5*weak_mat_tmp17 - weak_mat_tmp16*weak_mat_tmp30 - weak_mat_tmp22*weak_mat_tmp27 - weak_mat_tmp24*weak_mat_tmp31) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu7*weak_mat_tmp0 - (s_t(4) / s_t(3))*weak_mat_tmp33)) + C10*(weak_mat_tmp11*weak_mat_tmp30 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu7*weak_mat_tmp0 - (s_t(2) / s_t(3))*weak_mat_tmp33))) + weak_mat_tmp9*(gu6*weak_mat_tmp21 - gu7*weak_mat_tmp12);
    const s_t material6 = gu6*iso_history_qp[0] + gu7*iso_history_qp[3] + iso_history_qp[4]*weak_mat_tmp3 + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(s_t(2)*gu6*weak_mat_tmp17 - weak_mat_tmp14*weak_mat_tmp34 - weak_mat_tmp20*weak_mat_tmp36 - weak_mat_tmp22*weak_mat_tmp37) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu2*weak_mat_tmp2 - (s_t(4) / s_t(3))*weak_mat_tmp35)) + C10*(weak_mat_tmp11*weak_mat_tmp34 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu2*weak_mat_tmp2 - (s_t(2) / s_t(3))*weak_mat_tmp35))) + weak_mat_tmp9*(gu5*weak_mat_tmp21 - weak_mat_tmp2*weak_mat_tmp23);
    const s_t material7 = gu6*iso_history_qp[3] + gu7*iso_history_qp[1] + iso_history_qp[5]*weak_mat_tmp3 + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(s_t(2)*gu7*weak_mat_tmp17 - weak_mat_tmp15*weak_mat_tmp36 - weak_mat_tmp20*weak_mat_tmp34 - weak_mat_tmp24*weak_mat_tmp37) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu5*weak_mat_tmp0 - (s_t(4) / s_t(3))*weak_mat_tmp38)) + C10*(weak_mat_tmp11*weak_mat_tmp36 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu5*weak_mat_tmp0 - (s_t(2) / s_t(3))*weak_mat_tmp38))) + weak_mat_tmp9*(gu3*weak_mat_tmp23 - gu5*weak_mat_tmp12);
    const s_t material8 = gu6*iso_history_qp[4] + gu7*iso_history_qp[5] + iso_history_qp[2]*weak_mat_tmp3 + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(-weak_mat_tmp16*weak_mat_tmp37 + s_t(2)*weak_mat_tmp17*weak_mat_tmp3 - weak_mat_tmp22*weak_mat_tmp34 - weak_mat_tmp24*weak_mat_tmp36) + weak_mat_tmp25*(-(s_t(4) / s_t(3))*weak_mat_tmp39 + ((s_t(4) / s_t(3)))*weak_mat_tmp5)) + C10*(weak_mat_tmp11*weak_mat_tmp37 + weak_mat_tmp18*(-(s_t(2) / s_t(3))*weak_mat_tmp39 + ((s_t(2) / s_t(3)))*weak_mat_tmp5))) + weak_mat_tmp9*(s_t(2)*weak_mat_tmp0*weak_mat_tmp2 - s_t(2)*weak_mat_tmp5);
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
static SFEM_INLINE void mooney_rivlin_prony_d3_simplex_tet4_gradient_block(
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
    const s_t weak_mat_tmp7 = gu5*gu6;
    const s_t weak_mat_tmp8 = gu3*gu7;
    const s_t weak_mat_tmp9 = ((s_t(1) / s_t(2)))*K*(gu0*gu8 - gu0*weak_mat_tmp1 + gu0 + gu1*weak_mat_tmp7 + gu2*weak_mat_tmp8 + gu4*gu8 - gu4*weak_mat_tmp6 + gu4 + gu8*weak_mat_tmp4 - gu8*weak_mat_tmp5 + gu8 - weak_mat_tmp1 + weak_mat_tmp4 - weak_mat_tmp5 - weak_mat_tmp6);
    const s_t weak_mat_tmp10 = gu1*gu5*gu6 + gu2*gu3*gu7 - weak_mat_tmp0*weak_mat_tmp1 + weak_mat_tmp0*weak_mat_tmp2*weak_mat_tmp3 - weak_mat_tmp2*weak_mat_tmp6 - weak_mat_tmp3*weak_mat_tmp5;
    const s_t weak_mat_tmp11 = pow(weak_mat_tmp10, (s_t(-2) / s_t(3)));
    const s_t weak_mat_tmp12 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp13 = weak_mat_tmp2*weak_mat_tmp3;
    const s_t weak_mat_tmp14 = pow_2(gu3) + pow_2(gu6) + pow_2(weak_mat_tmp0);
    const s_t weak_mat_tmp15 = pow_2(gu1) + pow_2(gu7) + pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp16 = pow_2(gu2) + pow_2(gu5) + pow_2(weak_mat_tmp3);
    const s_t weak_mat_tmp17 = weak_mat_tmp14 + weak_mat_tmp15 + weak_mat_tmp16;
    const s_t weak_mat_tmp18 = weak_mat_tmp17/pow(weak_mat_tmp10, (s_t(5) / s_t(3)));
    const s_t weak_mat_tmp19 = pow(weak_mat_tmp10, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp20 = gu1*weak_mat_tmp0 + gu3*weak_mat_tmp2 + gu6*gu7;
    const s_t weak_mat_tmp21 = s_t(2)*gu1;
    const s_t weak_mat_tmp22 = gu2*weak_mat_tmp0 + gu3*gu5 + gu6*weak_mat_tmp3;
    const s_t weak_mat_tmp23 = s_t(2)*gu2;
    const s_t weak_mat_tmp24 = gu1*gu2 + gu5*weak_mat_tmp2 + gu7*weak_mat_tmp3;
    const s_t weak_mat_tmp25 = (-(s_t(1) / s_t(2))*pow_2(weak_mat_tmp14) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp15) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp16) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp17) - pow_2(weak_mat_tmp20) - pow_2(weak_mat_tmp22) - pow_2(weak_mat_tmp24))/pow(weak_mat_tmp10, (s_t(7) / s_t(3)));
    const s_t weak_mat_tmp26 = gu3*weak_mat_tmp3;
    const s_t weak_mat_tmp27 = s_t(2)*gu3;
    const s_t weak_mat_tmp28 = gu2*gu7;
    const s_t weak_mat_tmp29 = gu1*weak_mat_tmp3;
    const s_t weak_mat_tmp30 = s_t(2)*gu5;
    const s_t weak_mat_tmp31 = s_t(2)*weak_mat_tmp2;
    const s_t weak_mat_tmp32 = weak_mat_tmp0*weak_mat_tmp3;
    const s_t weak_mat_tmp33 = gu1*gu6;
    const s_t weak_mat_tmp34 = s_t(2)*gu6;
    const s_t weak_mat_tmp35 = gu1*gu5;
    const s_t weak_mat_tmp36 = s_t(2)*gu7;
    const s_t weak_mat_tmp37 = s_t(2)*weak_mat_tmp3;
    const s_t weak_mat_tmp38 = gu2*gu3;
    const s_t weak_mat_tmp39 = weak_mat_tmp0*weak_mat_tmp2;
    const s_t material0 = gu1*iso_history_qp[3] + gu2*iso_history_qp[4] + iso_history_qp[0]*weak_mat_tmp0 + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(s_t(2)*weak_mat_tmp0*weak_mat_tmp17 - weak_mat_tmp12*weak_mat_tmp14 - weak_mat_tmp20*weak_mat_tmp21 - weak_mat_tmp22*weak_mat_tmp23) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*weak_mat_tmp1 - (s_t(4) / s_t(3))*weak_mat_tmp13)) + C10*(weak_mat_tmp11*weak_mat_tmp12 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*weak_mat_tmp1 - (s_t(2) / s_t(3))*weak_mat_tmp13))) + weak_mat_tmp9*(-s_t(2)*weak_mat_tmp1 + s_t(2)*weak_mat_tmp2*weak_mat_tmp3);
    const s_t material1 = gu1*iso_history_qp[1] + gu2*iso_history_qp[5] + iso_history_qp[3]*weak_mat_tmp0 + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(s_t(2)*gu1*weak_mat_tmp17 - weak_mat_tmp12*weak_mat_tmp20 - weak_mat_tmp15*weak_mat_tmp21 - weak_mat_tmp23*weak_mat_tmp24) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*weak_mat_tmp26 - (s_t(4) / s_t(3))*weak_mat_tmp7)) + C10*(weak_mat_tmp11*weak_mat_tmp21 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*weak_mat_tmp26 - (s_t(2) / s_t(3))*weak_mat_tmp7))) + weak_mat_tmp9*(s_t(2)*gu5*gu6 - s_t(2)*weak_mat_tmp26);
    const s_t material2 = gu1*iso_history_qp[5] + gu2*iso_history_qp[2] + iso_history_qp[4]*weak_mat_tmp0 + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(s_t(2)*gu2*weak_mat_tmp17 - weak_mat_tmp12*weak_mat_tmp22 - weak_mat_tmp16*weak_mat_tmp23 - weak_mat_tmp21*weak_mat_tmp24) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu6*weak_mat_tmp2 - (s_t(4) / s_t(3))*weak_mat_tmp8)) + C10*(weak_mat_tmp11*weak_mat_tmp23 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu6*weak_mat_tmp2 - (s_t(2) / s_t(3))*weak_mat_tmp8))) + weak_mat_tmp9*(-s_t(2)*gu6*weak_mat_tmp2 + s_t(2)*weak_mat_tmp8);
    const s_t material3 = gu3*iso_history_qp[0] + gu5*iso_history_qp[4] + iso_history_qp[3]*weak_mat_tmp2 + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(s_t(2)*gu3*weak_mat_tmp17 - weak_mat_tmp14*weak_mat_tmp27 - weak_mat_tmp20*weak_mat_tmp31 - weak_mat_tmp22*weak_mat_tmp30) + weak_mat_tmp25*(-(s_t(4) / s_t(3))*weak_mat_tmp28 + ((s_t(4) / s_t(3)))*weak_mat_tmp29)) + C10*(weak_mat_tmp11*weak_mat_tmp27 + weak_mat_tmp18*(-(s_t(2) / s_t(3))*weak_mat_tmp28 + ((s_t(2) / s_t(3)))*weak_mat_tmp29))) + weak_mat_tmp9*(s_t(2)*gu2*gu7 - weak_mat_tmp21*weak_mat_tmp3);
    const s_t material4 = gu3*iso_history_qp[3] + gu5*iso_history_qp[5] + iso_history_qp[1]*weak_mat_tmp2 + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(-weak_mat_tmp15*weak_mat_tmp31 + s_t(2)*weak_mat_tmp17*weak_mat_tmp2 - weak_mat_tmp20*weak_mat_tmp27 - weak_mat_tmp24*weak_mat_tmp30) + weak_mat_tmp25*(-(s_t(4) / s_t(3))*weak_mat_tmp32 + ((s_t(4) / s_t(3)))*weak_mat_tmp6)) + C10*(weak_mat_tmp11*weak_mat_tmp31 + weak_mat_tmp18*(-(s_t(2) / s_t(3))*weak_mat_tmp32 + ((s_t(2) / s_t(3)))*weak_mat_tmp6))) + weak_mat_tmp9*(s_t(2)*weak_mat_tmp0*weak_mat_tmp3 - s_t(2)*weak_mat_tmp6);
    const s_t material5 = gu3*iso_history_qp[4] + gu5*iso_history_qp[2] + iso_history_qp[5]*weak_mat_tmp2 + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(s_t(2)*gu5*weak_mat_tmp17 - weak_mat_tmp16*weak_mat_tmp30 - weak_mat_tmp22*weak_mat_tmp27 - weak_mat_tmp24*weak_mat_tmp31) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu7*weak_mat_tmp0 - (s_t(4) / s_t(3))*weak_mat_tmp33)) + C10*(weak_mat_tmp11*weak_mat_tmp30 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu7*weak_mat_tmp0 - (s_t(2) / s_t(3))*weak_mat_tmp33))) + weak_mat_tmp9*(gu6*weak_mat_tmp21 - gu7*weak_mat_tmp12);
    const s_t material6 = gu6*iso_history_qp[0] + gu7*iso_history_qp[3] + iso_history_qp[4]*weak_mat_tmp3 + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(s_t(2)*gu6*weak_mat_tmp17 - weak_mat_tmp14*weak_mat_tmp34 - weak_mat_tmp20*weak_mat_tmp36 - weak_mat_tmp22*weak_mat_tmp37) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu2*weak_mat_tmp2 - (s_t(4) / s_t(3))*weak_mat_tmp35)) + C10*(weak_mat_tmp11*weak_mat_tmp34 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu2*weak_mat_tmp2 - (s_t(2) / s_t(3))*weak_mat_tmp35))) + weak_mat_tmp9*(gu5*weak_mat_tmp21 - weak_mat_tmp2*weak_mat_tmp23);
    const s_t material7 = gu6*iso_history_qp[3] + gu7*iso_history_qp[1] + iso_history_qp[5]*weak_mat_tmp3 + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(s_t(2)*gu7*weak_mat_tmp17 - weak_mat_tmp15*weak_mat_tmp36 - weak_mat_tmp20*weak_mat_tmp34 - weak_mat_tmp24*weak_mat_tmp37) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu5*weak_mat_tmp0 - (s_t(4) / s_t(3))*weak_mat_tmp38)) + C10*(weak_mat_tmp11*weak_mat_tmp36 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu5*weak_mat_tmp0 - (s_t(2) / s_t(3))*weak_mat_tmp38))) + weak_mat_tmp9*(gu3*weak_mat_tmp23 - gu5*weak_mat_tmp12);
    const s_t material8 = gu6*iso_history_qp[4] + gu7*iso_history_qp[5] + iso_history_qp[2]*weak_mat_tmp3 + iso_relaxation_gamma*(C01*(weak_mat_tmp19*(-weak_mat_tmp16*weak_mat_tmp37 + s_t(2)*weak_mat_tmp17*weak_mat_tmp3 - weak_mat_tmp22*weak_mat_tmp34 - weak_mat_tmp24*weak_mat_tmp36) + weak_mat_tmp25*(-(s_t(4) / s_t(3))*weak_mat_tmp39 + ((s_t(4) / s_t(3)))*weak_mat_tmp5)) + C10*(weak_mat_tmp11*weak_mat_tmp37 + weak_mat_tmp18*(-(s_t(2) / s_t(3))*weak_mat_tmp39 + ((s_t(2) / s_t(3)))*weak_mat_tmp5))) + weak_mat_tmp9*(s_t(2)*weak_mat_tmp0*weak_mat_tmp2 - s_t(2)*weak_mat_tmp5);
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
static SFEM_INLINE void mooney_rivlin_prony_d3_simplex_apply_block(
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
    const s_t weak_mat_tmp5 = weak_mat_tmp1*weak_mat_tmp2;
    const s_t weak_mat_tmp6 = -s_t(2)*weak_mat_tmp5;
    const s_t weak_mat_tmp7 = ((s_t(1) / s_t(2)))*K;
    const s_t weak_mat_tmp8 = weak_mat_tmp7*(-weak_mat_tmp4 - weak_mat_tmp6);
    const s_t weak_mat_tmp9 = gu3*weak_mat_tmp2;
    const s_t weak_mat_tmp10 = gu6*weak_mat_tmp1;
    const s_t weak_mat_tmp11 = gu0 + s_t(1);
    const s_t weak_mat_tmp12 = gu1*gu5*gu6 - gu1*weak_mat_tmp9 + gu2*gu3*gu7 - gu2*weak_mat_tmp10 - weak_mat_tmp0*weak_mat_tmp11 + weak_mat_tmp1*weak_mat_tmp11*weak_mat_tmp2;
    const s_t weak_mat_tmp13 = s_t(2)/pow(weak_mat_tmp12, (s_t(2) / s_t(3)));
    const s_t weak_mat_tmp14 = pow(weak_mat_tmp12, (s_t(-5) / s_t(3)));
    const s_t weak_mat_tmp15 = ((s_t(2) / s_t(3)))*weak_mat_tmp0 - (s_t(2) / s_t(3))*weak_mat_tmp5;
    const s_t weak_mat_tmp16 = weak_mat_tmp14*weak_mat_tmp15;
    const s_t weak_mat_tmp17 = ((s_t(5) / s_t(3)))*weak_mat_tmp0 - (s_t(5) / s_t(3))*weak_mat_tmp5;
    const s_t weak_mat_tmp18 = pow_2(gu3);
    const s_t weak_mat_tmp19 = pow_2(gu6);
    const s_t weak_mat_tmp20 = pow_2(weak_mat_tmp11);
    const s_t weak_mat_tmp21 = weak_mat_tmp18 + weak_mat_tmp19 + weak_mat_tmp20;
    const s_t weak_mat_tmp22 = pow_2(gu1);
    const s_t weak_mat_tmp23 = pow_2(gu7);
    const s_t weak_mat_tmp24 = pow_2(weak_mat_tmp1);
    const s_t weak_mat_tmp25 = weak_mat_tmp22 + weak_mat_tmp23 + weak_mat_tmp24;
    const s_t weak_mat_tmp26 = pow_2(gu2);
    const s_t weak_mat_tmp27 = pow_2(gu5);
    const s_t weak_mat_tmp28 = pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp29 = weak_mat_tmp26 + weak_mat_tmp27 + weak_mat_tmp28;
    const s_t weak_mat_tmp30 = weak_mat_tmp21 + weak_mat_tmp25 + weak_mat_tmp29;
    const s_t weak_mat_tmp31 = weak_mat_tmp30/pow(weak_mat_tmp12, (s_t(8) / s_t(3)));
    const s_t weak_mat_tmp32 = weak_mat_tmp15*weak_mat_tmp31;
    const s_t weak_mat_tmp33 = s_t(2)*weak_mat_tmp27;
    const s_t weak_mat_tmp34 = s_t(2)*weak_mat_tmp28;
    const s_t weak_mat_tmp35 = weak_mat_tmp33 + weak_mat_tmp34;
    const s_t weak_mat_tmp36 = s_t(2)*weak_mat_tmp23;
    const s_t weak_mat_tmp37 = s_t(2)*weak_mat_tmp24;
    const s_t weak_mat_tmp38 = weak_mat_tmp36 + weak_mat_tmp37;
    const s_t weak_mat_tmp39 = pow(weak_mat_tmp12, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp40 = ((s_t(4) / s_t(3)))*weak_mat_tmp0 - (s_t(4) / s_t(3))*weak_mat_tmp5;
    const s_t weak_mat_tmp41 = pow(weak_mat_tmp12, (s_t(-7) / s_t(3)));
    const s_t weak_mat_tmp42 = gu6*gu7;
    const s_t weak_mat_tmp43 = gu1*weak_mat_tmp11;
    const s_t weak_mat_tmp44 = gu3*weak_mat_tmp1;
    const s_t weak_mat_tmp45 = weak_mat_tmp42 + weak_mat_tmp43 + weak_mat_tmp44;
    const s_t weak_mat_tmp46 = s_t(2)*gu1;
    const s_t weak_mat_tmp47 = gu3*gu5;
    const s_t weak_mat_tmp48 = gu2*weak_mat_tmp11;
    const s_t weak_mat_tmp49 = gu6*weak_mat_tmp2;
    const s_t weak_mat_tmp50 = weak_mat_tmp47 + weak_mat_tmp48 + weak_mat_tmp49;
    const s_t weak_mat_tmp51 = s_t(2)*gu2;
    const s_t weak_mat_tmp52 = s_t(2)*weak_mat_tmp11;
    const s_t weak_mat_tmp53 = weak_mat_tmp41*(s_t(2)*weak_mat_tmp11*weak_mat_tmp30 - weak_mat_tmp21*weak_mat_tmp52 - weak_mat_tmp45*weak_mat_tmp46 - weak_mat_tmp50*weak_mat_tmp51);
    const s_t weak_mat_tmp54 = ((s_t(7) / s_t(3)))*weak_mat_tmp0 - (s_t(7) / s_t(3))*weak_mat_tmp5;
    const s_t weak_mat_tmp55 = gu1*gu2;
    const s_t weak_mat_tmp56 = gu5*weak_mat_tmp1;
    const s_t weak_mat_tmp57 = gu7*weak_mat_tmp2;
    const s_t weak_mat_tmp58 = weak_mat_tmp55 + weak_mat_tmp56 + weak_mat_tmp57;
    const s_t weak_mat_tmp59 = -(s_t(1) / s_t(2))*pow_2(weak_mat_tmp21) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp25) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp29) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp30) - pow_2(weak_mat_tmp45) - pow_2(weak_mat_tmp50) - pow_2(weak_mat_tmp58);
    const s_t weak_mat_tmp60 = weak_mat_tmp59/pow(weak_mat_tmp12, (s_t(10) / s_t(3)));
    const s_t weak_mat_tmp61 = weak_mat_tmp40*weak_mat_tmp60;
    const s_t weak_mat_tmp62 = gu1*weak_mat_tmp2;
    const s_t weak_mat_tmp63 = gu2*gu7 - weak_mat_tmp62;
    const s_t weak_mat_tmp64 = gu2*gu7;
    const s_t weak_mat_tmp65 = ((s_t(5) / s_t(3)))*weak_mat_tmp62 - (s_t(5) / s_t(3))*weak_mat_tmp64;
    const s_t weak_mat_tmp66 = s_t(2)*gu3;
    const s_t weak_mat_tmp67 = ((s_t(2) / s_t(3)))*weak_mat_tmp62 - (s_t(2) / s_t(3))*weak_mat_tmp64;
    const s_t weak_mat_tmp68 = weak_mat_tmp14*weak_mat_tmp52;
    const s_t weak_mat_tmp69 = weak_mat_tmp16*weak_mat_tmp66 + weak_mat_tmp67*weak_mat_tmp68;
    const s_t weak_mat_tmp70 = ((s_t(7) / s_t(3)))*weak_mat_tmp62 - (s_t(7) / s_t(3))*weak_mat_tmp64;
    const s_t weak_mat_tmp71 = gu5*weak_mat_tmp51;
    const s_t weak_mat_tmp72 = weak_mat_tmp1*weak_mat_tmp46;
    const s_t weak_mat_tmp73 = ((s_t(4) / s_t(3)))*weak_mat_tmp62 - (s_t(4) / s_t(3))*weak_mat_tmp64;
    const s_t weak_mat_tmp74 = s_t(2)*gu5;
    const s_t weak_mat_tmp75 = s_t(2)*weak_mat_tmp1;
    const s_t weak_mat_tmp76 = s_t(2)*gu3*weak_mat_tmp30 - weak_mat_tmp21*weak_mat_tmp66 - weak_mat_tmp45*weak_mat_tmp75 - weak_mat_tmp50*weak_mat_tmp74;
    const s_t weak_mat_tmp77 = weak_mat_tmp40*weak_mat_tmp41;
    const s_t weak_mat_tmp78 = weak_mat_tmp39*(-weak_mat_tmp71 - weak_mat_tmp72) + weak_mat_tmp53*weak_mat_tmp73 + weak_mat_tmp76*weak_mat_tmp77;
    const s_t weak_mat_tmp79 = gu1*gu5;
    const s_t weak_mat_tmp80 = gu2*weak_mat_tmp1;
    const s_t weak_mat_tmp81 = weak_mat_tmp79 - weak_mat_tmp80;
    const s_t weak_mat_tmp82 = ((s_t(5) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(5) / s_t(3))*weak_mat_tmp79;
    const s_t weak_mat_tmp83 = s_t(2)*gu6;
    const s_t weak_mat_tmp84 = ((s_t(2) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(2) / s_t(3))*weak_mat_tmp79;
    const s_t weak_mat_tmp85 = weak_mat_tmp16*weak_mat_tmp83 + weak_mat_tmp68*weak_mat_tmp84;
    const s_t weak_mat_tmp86 = ((s_t(7) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(7) / s_t(3))*weak_mat_tmp79;
    const s_t weak_mat_tmp87 = gu7*weak_mat_tmp46;
    const s_t weak_mat_tmp88 = weak_mat_tmp2*weak_mat_tmp51;
    const s_t weak_mat_tmp89 = ((s_t(4) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(4) / s_t(3))*weak_mat_tmp79;
    const s_t weak_mat_tmp90 = s_t(2)*gu7;
    const s_t weak_mat_tmp91 = s_t(2)*weak_mat_tmp2;
    const s_t weak_mat_tmp92 = s_t(2)*gu6*weak_mat_tmp30 - weak_mat_tmp21*weak_mat_tmp83 - weak_mat_tmp45*weak_mat_tmp90 - weak_mat_tmp50*weak_mat_tmp91;
    const s_t weak_mat_tmp93 = weak_mat_tmp39*(-weak_mat_tmp87 - weak_mat_tmp88) + weak_mat_tmp53*weak_mat_tmp89 + weak_mat_tmp77*weak_mat_tmp92;
    const s_t weak_mat_tmp94 = gu5*gu6 - weak_mat_tmp9;
    const s_t weak_mat_tmp95 = gu5*gu6;
    const s_t weak_mat_tmp96 = ((s_t(5) / s_t(3)))*weak_mat_tmp9 - (s_t(5) / s_t(3))*weak_mat_tmp95;
    const s_t weak_mat_tmp97 = ((s_t(2) / s_t(3)))*weak_mat_tmp9 - (s_t(2) / s_t(3))*weak_mat_tmp95;
    const s_t weak_mat_tmp98 = weak_mat_tmp16*weak_mat_tmp46 + weak_mat_tmp68*weak_mat_tmp97;
    const s_t weak_mat_tmp99 = ((s_t(7) / s_t(3)))*weak_mat_tmp9 - (s_t(7) / s_t(3))*weak_mat_tmp95;
    const s_t weak_mat_tmp100 = s_t(2)*weak_mat_tmp42;
    const s_t weak_mat_tmp101 = s_t(2)*weak_mat_tmp44;
    const s_t weak_mat_tmp102 = ((s_t(4) / s_t(3)))*weak_mat_tmp9 - (s_t(4) / s_t(3))*weak_mat_tmp95;
    const s_t weak_mat_tmp103 = s_t(2)*gu1*weak_mat_tmp30 - weak_mat_tmp25*weak_mat_tmp46 - weak_mat_tmp45*weak_mat_tmp52 - weak_mat_tmp51*weak_mat_tmp58;
    const s_t weak_mat_tmp104 = weak_mat_tmp102*weak_mat_tmp53 + weak_mat_tmp103*weak_mat_tmp77 + weak_mat_tmp39*(-weak_mat_tmp100 - weak_mat_tmp101);
    const s_t weak_mat_tmp105 = gu3*gu7;
    const s_t weak_mat_tmp106 = -weak_mat_tmp10 + weak_mat_tmp105;
    const s_t weak_mat_tmp107 = ((s_t(5) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(5) / s_t(3))*weak_mat_tmp105;
    const s_t weak_mat_tmp108 = ((s_t(2) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(2) / s_t(3))*weak_mat_tmp105;
    const s_t weak_mat_tmp109 = weak_mat_tmp108*weak_mat_tmp68 + weak_mat_tmp16*weak_mat_tmp51;
    const s_t weak_mat_tmp110 = ((s_t(7) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(7) / s_t(3))*weak_mat_tmp105;
    const s_t weak_mat_tmp111 = s_t(2)*weak_mat_tmp47;
    const s_t weak_mat_tmp112 = s_t(2)*weak_mat_tmp49;
    const s_t weak_mat_tmp113 = ((s_t(4) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(4) / s_t(3))*weak_mat_tmp105;
    const s_t weak_mat_tmp114 = s_t(2)*gu2*weak_mat_tmp30 - weak_mat_tmp29*weak_mat_tmp51 - weak_mat_tmp46*weak_mat_tmp58 - weak_mat_tmp50*weak_mat_tmp52;
    const s_t weak_mat_tmp115 = weak_mat_tmp113*weak_mat_tmp53 + weak_mat_tmp114*weak_mat_tmp77 + weak_mat_tmp39*(-weak_mat_tmp111 - weak_mat_tmp112);
    const s_t weak_mat_tmp116 = gu1*gu6;
    const s_t weak_mat_tmp117 = gu7*weak_mat_tmp11;
    const s_t weak_mat_tmp118 = weak_mat_tmp116 - weak_mat_tmp117;
    const s_t weak_mat_tmp119 = gu0*gu4;
    const s_t weak_mat_tmp120 = gu1*gu3;
    const s_t weak_mat_tmp121 = gu2*gu6;
    const s_t weak_mat_tmp122 = K*(gu0*gu8 - gu0*weak_mat_tmp0 + gu0 + gu1*weak_mat_tmp95 + gu2*weak_mat_tmp105 + gu4*gu8 - gu4*weak_mat_tmp121 + gu4 + gu8*weak_mat_tmp119 - gu8*weak_mat_tmp120 + gu8 - weak_mat_tmp0 + weak_mat_tmp119 - weak_mat_tmp120 - weak_mat_tmp121);
    const s_t weak_mat_tmp123 = gu7*weak_mat_tmp122;
    const s_t weak_mat_tmp124 = -weak_mat_tmp123;
    const s_t weak_mat_tmp125 = ((s_t(5) / s_t(3)))*gu7*weak_mat_tmp11 - (s_t(5) / s_t(3))*weak_mat_tmp116;
    const s_t weak_mat_tmp126 = ((s_t(2) / s_t(3)))*gu7*weak_mat_tmp11 - (s_t(2) / s_t(3))*weak_mat_tmp116;
    const s_t weak_mat_tmp127 = ((s_t(2) / s_t(3)))*weak_mat_tmp30;
    const s_t weak_mat_tmp128 = weak_mat_tmp127*weak_mat_tmp14;
    const s_t weak_mat_tmp129 = gu7*weak_mat_tmp128;
    const s_t weak_mat_tmp130 = weak_mat_tmp126*weak_mat_tmp68 + weak_mat_tmp129 + weak_mat_tmp16*weak_mat_tmp74;
    const s_t weak_mat_tmp131 = ((s_t(7) / s_t(3)))*gu7*weak_mat_tmp11 - (s_t(7) / s_t(3))*weak_mat_tmp116;
    const s_t weak_mat_tmp132 = gu2*gu3;
    const s_t weak_mat_tmp133 = s_t(2)*weak_mat_tmp132;
    const s_t weak_mat_tmp134 = ((s_t(4) / s_t(3)))*gu7*weak_mat_tmp11 - (s_t(4) / s_t(3))*weak_mat_tmp116;
    const s_t weak_mat_tmp135 = s_t(2)*gu5*weak_mat_tmp30 - weak_mat_tmp29*weak_mat_tmp74 - weak_mat_tmp50*weak_mat_tmp66 - weak_mat_tmp58*weak_mat_tmp75;
    const s_t weak_mat_tmp136 = ((s_t(4) / s_t(3)))*weak_mat_tmp41*weak_mat_tmp59;
    const s_t weak_mat_tmp137 = gu7*weak_mat_tmp136;
    const s_t weak_mat_tmp138 = weak_mat_tmp134*weak_mat_tmp53 + weak_mat_tmp135*weak_mat_tmp77 + weak_mat_tmp137 + weak_mat_tmp39*(s_t(4)*gu5*weak_mat_tmp11 - weak_mat_tmp133);
    const s_t weak_mat_tmp139 = gu5*weak_mat_tmp11;
    const s_t weak_mat_tmp140 = weak_mat_tmp132 - weak_mat_tmp139;
    const s_t weak_mat_tmp141 = gu5*weak_mat_tmp122;
    const s_t weak_mat_tmp142 = -weak_mat_tmp141;
    const s_t weak_mat_tmp143 = ((s_t(5) / s_t(3)))*gu5*weak_mat_tmp11 - (s_t(5) / s_t(3))*weak_mat_tmp132;
    const s_t weak_mat_tmp144 = ((s_t(2) / s_t(3)))*gu5*weak_mat_tmp11 - (s_t(2) / s_t(3))*weak_mat_tmp132;
    const s_t weak_mat_tmp145 = gu5*weak_mat_tmp128;
    const s_t weak_mat_tmp146 = weak_mat_tmp144*weak_mat_tmp68 + weak_mat_tmp145 + weak_mat_tmp16*weak_mat_tmp90;
    const s_t weak_mat_tmp147 = ((s_t(7) / s_t(3)))*gu5*weak_mat_tmp11 - (s_t(7) / s_t(3))*weak_mat_tmp132;
    const s_t weak_mat_tmp148 = s_t(2)*weak_mat_tmp116;
    const s_t weak_mat_tmp149 = ((s_t(4) / s_t(3)))*gu5*weak_mat_tmp11 - (s_t(4) / s_t(3))*weak_mat_tmp132;
    const s_t weak_mat_tmp150 = s_t(2)*gu7*weak_mat_tmp30 - weak_mat_tmp25*weak_mat_tmp90 - weak_mat_tmp45*weak_mat_tmp83 - weak_mat_tmp58*weak_mat_tmp91;
    const s_t weak_mat_tmp151 = gu5*weak_mat_tmp136;
    const s_t weak_mat_tmp152 = weak_mat_tmp149*weak_mat_tmp53 + weak_mat_tmp150*weak_mat_tmp77 + weak_mat_tmp151 + weak_mat_tmp39*(s_t(4)*gu7*weak_mat_tmp11 - weak_mat_tmp148);
    const s_t weak_mat_tmp153 = weak_mat_tmp11*weak_mat_tmp2 - weak_mat_tmp121;
    const s_t weak_mat_tmp154 = weak_mat_tmp122*weak_mat_tmp2;
    const s_t weak_mat_tmp155 = weak_mat_tmp11*weak_mat_tmp2;
    const s_t weak_mat_tmp156 = ((s_t(5) / s_t(3)))*weak_mat_tmp121 - (s_t(5) / s_t(3))*weak_mat_tmp155;
    const s_t weak_mat_tmp157 = ((s_t(2) / s_t(3)))*weak_mat_tmp121 - (s_t(2) / s_t(3))*weak_mat_tmp155;
    const s_t weak_mat_tmp158 = weak_mat_tmp14*weak_mat_tmp2;
    const s_t weak_mat_tmp159 = weak_mat_tmp127*weak_mat_tmp158;
    const s_t weak_mat_tmp160 = weak_mat_tmp157*weak_mat_tmp68 - weak_mat_tmp159 + weak_mat_tmp16*weak_mat_tmp75;
    const s_t weak_mat_tmp161 = ((s_t(7) / s_t(3)))*weak_mat_tmp121 - (s_t(7) / s_t(3))*weak_mat_tmp155;
    const s_t weak_mat_tmp162 = s_t(2)*weak_mat_tmp120;
    const s_t weak_mat_tmp163 = ((s_t(4) / s_t(3)))*weak_mat_tmp121 - (s_t(4) / s_t(3))*weak_mat_tmp155;
    const s_t weak_mat_tmp164 = s_t(2)*weak_mat_tmp1*weak_mat_tmp30 - weak_mat_tmp25*weak_mat_tmp75 - weak_mat_tmp45*weak_mat_tmp66 - weak_mat_tmp58*weak_mat_tmp74;
    const s_t weak_mat_tmp165 = weak_mat_tmp136*weak_mat_tmp2;
    const s_t weak_mat_tmp166 = weak_mat_tmp163*weak_mat_tmp53 + weak_mat_tmp164*weak_mat_tmp77 - weak_mat_tmp165 + weak_mat_tmp39*(s_t(4)*weak_mat_tmp1*weak_mat_tmp11 - weak_mat_tmp162);
    const s_t weak_mat_tmp167 = weak_mat_tmp1*weak_mat_tmp11 - weak_mat_tmp120;
    const s_t weak_mat_tmp168 = weak_mat_tmp1*weak_mat_tmp122;
    const s_t weak_mat_tmp169 = weak_mat_tmp1*weak_mat_tmp11;
    const s_t weak_mat_tmp170 = ((s_t(5) / s_t(3)))*weak_mat_tmp120 - (s_t(5) / s_t(3))*weak_mat_tmp169;
    const s_t weak_mat_tmp171 = ((s_t(2) / s_t(3)))*weak_mat_tmp120 - (s_t(2) / s_t(3))*weak_mat_tmp169;
    const s_t weak_mat_tmp172 = weak_mat_tmp1*weak_mat_tmp128;
    const s_t weak_mat_tmp173 = weak_mat_tmp16*weak_mat_tmp91 + weak_mat_tmp171*weak_mat_tmp68 - weak_mat_tmp172;
    const s_t weak_mat_tmp174 = ((s_t(7) / s_t(3)))*weak_mat_tmp120 - (s_t(7) / s_t(3))*weak_mat_tmp169;
    const s_t weak_mat_tmp175 = s_t(2)*weak_mat_tmp121;
    const s_t weak_mat_tmp176 = ((s_t(4) / s_t(3)))*weak_mat_tmp120 - (s_t(4) / s_t(3))*weak_mat_tmp169;
    const s_t weak_mat_tmp177 = s_t(2)*weak_mat_tmp2*weak_mat_tmp30 - weak_mat_tmp29*weak_mat_tmp91 - weak_mat_tmp50*weak_mat_tmp83 - weak_mat_tmp58*weak_mat_tmp90;
    const s_t weak_mat_tmp178 = weak_mat_tmp1*weak_mat_tmp136;
    const s_t weak_mat_tmp179 = weak_mat_tmp176*weak_mat_tmp53 + weak_mat_tmp177*weak_mat_tmp77 - weak_mat_tmp178 + weak_mat_tmp39*(s_t(4)*weak_mat_tmp11*weak_mat_tmp2 - weak_mat_tmp175);
    const s_t weak_mat_tmp180 = -s_t(2)*weak_mat_tmp95;
    const s_t weak_mat_tmp181 = s_t(2)*weak_mat_tmp9;
    const s_t weak_mat_tmp182 = weak_mat_tmp7*(-weak_mat_tmp180 - weak_mat_tmp181);
    const s_t weak_mat_tmp183 = weak_mat_tmp14*weak_mat_tmp97;
    const s_t weak_mat_tmp184 = weak_mat_tmp31*weak_mat_tmp97;
    const s_t weak_mat_tmp185 = s_t(2)*weak_mat_tmp18;
    const s_t weak_mat_tmp186 = s_t(2)*weak_mat_tmp19;
    const s_t weak_mat_tmp187 = weak_mat_tmp185 + weak_mat_tmp186;
    const s_t weak_mat_tmp188 = weak_mat_tmp103*weak_mat_tmp41;
    const s_t weak_mat_tmp189 = weak_mat_tmp102*weak_mat_tmp60;
    const s_t weak_mat_tmp190 = weak_mat_tmp14*weak_mat_tmp46;
    const s_t weak_mat_tmp191 = weak_mat_tmp144*weak_mat_tmp190 + weak_mat_tmp183*weak_mat_tmp90;
    const s_t weak_mat_tmp192 = gu6*weak_mat_tmp52;
    const s_t weak_mat_tmp193 = weak_mat_tmp102*weak_mat_tmp41;
    const s_t weak_mat_tmp194 = weak_mat_tmp149*weak_mat_tmp188 + weak_mat_tmp150*weak_mat_tmp193 + weak_mat_tmp39*(-weak_mat_tmp192 - weak_mat_tmp88);
    const s_t weak_mat_tmp195 = weak_mat_tmp108*weak_mat_tmp190 + weak_mat_tmp183*weak_mat_tmp51;
    const s_t weak_mat_tmp196 = s_t(2)*weak_mat_tmp56;
    const s_t weak_mat_tmp197 = s_t(2)*weak_mat_tmp57;
    const s_t weak_mat_tmp198 = weak_mat_tmp113*weak_mat_tmp188 + weak_mat_tmp114*weak_mat_tmp193 + weak_mat_tmp39*(-weak_mat_tmp196 - weak_mat_tmp197);
    const s_t weak_mat_tmp199 = weak_mat_tmp157*weak_mat_tmp190 + weak_mat_tmp183*weak_mat_tmp75;
    const s_t weak_mat_tmp200 = gu3*weak_mat_tmp52;
    const s_t weak_mat_tmp201 = weak_mat_tmp163*weak_mat_tmp188 + weak_mat_tmp164*weak_mat_tmp193 + weak_mat_tmp39*(-weak_mat_tmp200 - weak_mat_tmp71);
    const s_t weak_mat_tmp202 = gu6*weak_mat_tmp122;
    const s_t weak_mat_tmp203 = gu6*weak_mat_tmp128;
    const s_t weak_mat_tmp204 = weak_mat_tmp126*weak_mat_tmp190 + weak_mat_tmp183*weak_mat_tmp74 - weak_mat_tmp203;
    const s_t weak_mat_tmp205 = -s_t(2)*weak_mat_tmp80;
    const s_t weak_mat_tmp206 = gu6*weak_mat_tmp136;
    const s_t weak_mat_tmp207 = weak_mat_tmp134*weak_mat_tmp188 + weak_mat_tmp135*weak_mat_tmp193 - weak_mat_tmp206 + weak_mat_tmp39*(weak_mat_tmp205 + s_t(4)*weak_mat_tmp79);
    const s_t weak_mat_tmp208 = -weak_mat_tmp145 + weak_mat_tmp183*weak_mat_tmp83 + weak_mat_tmp190*weak_mat_tmp84;
    const s_t weak_mat_tmp209 = -s_t(2)*weak_mat_tmp117;
    const s_t weak_mat_tmp210 = -weak_mat_tmp151 + weak_mat_tmp188*weak_mat_tmp89 + weak_mat_tmp193*weak_mat_tmp92 + weak_mat_tmp39*(s_t(4)*weak_mat_tmp116 + weak_mat_tmp209);
    const s_t weak_mat_tmp211 = -weak_mat_tmp154;
    const s_t weak_mat_tmp212 = weak_mat_tmp159 + weak_mat_tmp183*weak_mat_tmp66 + weak_mat_tmp190*weak_mat_tmp67;
    const s_t weak_mat_tmp213 = -s_t(2)*weak_mat_tmp169;
    const s_t weak_mat_tmp214 = weak_mat_tmp165 + weak_mat_tmp188*weak_mat_tmp73 + weak_mat_tmp193*weak_mat_tmp76 + weak_mat_tmp39*(s_t(4)*weak_mat_tmp120 + weak_mat_tmp213);
    const s_t weak_mat_tmp215 = gu3*weak_mat_tmp122;
    const s_t weak_mat_tmp216 = -weak_mat_tmp215;
    const s_t weak_mat_tmp217 = gu3*weak_mat_tmp128;
    const s_t weak_mat_tmp218 = weak_mat_tmp171*weak_mat_tmp190 + weak_mat_tmp183*weak_mat_tmp91 + weak_mat_tmp217;
    const s_t weak_mat_tmp219 = -s_t(2)*weak_mat_tmp64;
    const s_t weak_mat_tmp220 = gu3*weak_mat_tmp136;
    const s_t weak_mat_tmp221 = weak_mat_tmp176*weak_mat_tmp188 + weak_mat_tmp177*weak_mat_tmp193 + weak_mat_tmp220 + weak_mat_tmp39*(weak_mat_tmp219 + s_t(4)*weak_mat_tmp62);
    const s_t weak_mat_tmp222 = s_t(2)*weak_mat_tmp105;
    const s_t weak_mat_tmp223 = -s_t(2)*weak_mat_tmp10;
    const s_t weak_mat_tmp224 = weak_mat_tmp7*(weak_mat_tmp222 + weak_mat_tmp223);
    const s_t weak_mat_tmp225 = weak_mat_tmp108*weak_mat_tmp14;
    const s_t weak_mat_tmp226 = weak_mat_tmp108*weak_mat_tmp31;
    const s_t weak_mat_tmp227 = weak_mat_tmp114*weak_mat_tmp41;
    const s_t weak_mat_tmp228 = weak_mat_tmp113*weak_mat_tmp60;
    const s_t weak_mat_tmp229 = weak_mat_tmp14*weak_mat_tmp51;
    const s_t weak_mat_tmp230 = weak_mat_tmp126*weak_mat_tmp229 + weak_mat_tmp225*weak_mat_tmp74;
    const s_t weak_mat_tmp231 = weak_mat_tmp113*weak_mat_tmp41;
    const s_t weak_mat_tmp232 = weak_mat_tmp134*weak_mat_tmp227 + weak_mat_tmp135*weak_mat_tmp231 + weak_mat_tmp39*(-weak_mat_tmp200 - weak_mat_tmp72);
    const s_t weak_mat_tmp233 = weak_mat_tmp171*weak_mat_tmp229 + weak_mat_tmp225*weak_mat_tmp91;
    const s_t weak_mat_tmp234 = weak_mat_tmp176*weak_mat_tmp227 + weak_mat_tmp177*weak_mat_tmp231 + weak_mat_tmp39*(-weak_mat_tmp192 - weak_mat_tmp87);
    const s_t weak_mat_tmp235 = -weak_mat_tmp129 + weak_mat_tmp225*weak_mat_tmp66 + weak_mat_tmp229*weak_mat_tmp67;
    const s_t weak_mat_tmp236 = -s_t(2)*weak_mat_tmp139;
    const s_t weak_mat_tmp237 = -weak_mat_tmp137 + weak_mat_tmp227*weak_mat_tmp73 + weak_mat_tmp231*weak_mat_tmp76 + weak_mat_tmp39*(s_t(4)*weak_mat_tmp132 + weak_mat_tmp236);
    const s_t weak_mat_tmp238 = weak_mat_tmp144*weak_mat_tmp229 - weak_mat_tmp217 + weak_mat_tmp225*weak_mat_tmp90;
    const s_t weak_mat_tmp239 = s_t(2)*weak_mat_tmp62;
    const s_t weak_mat_tmp240 = weak_mat_tmp149*weak_mat_tmp227 + weak_mat_tmp150*weak_mat_tmp231 - weak_mat_tmp220 + weak_mat_tmp39*(s_t(4)*gu2*gu7 - weak_mat_tmp239);
    const s_t weak_mat_tmp241 = -weak_mat_tmp168;
    const s_t weak_mat_tmp242 = weak_mat_tmp172 + weak_mat_tmp225*weak_mat_tmp83 + weak_mat_tmp229*weak_mat_tmp84;
    const s_t weak_mat_tmp243 = -s_t(2)*weak_mat_tmp155;
    const s_t weak_mat_tmp244 = weak_mat_tmp178 + weak_mat_tmp227*weak_mat_tmp89 + weak_mat_tmp231*weak_mat_tmp92 + weak_mat_tmp39*(s_t(4)*weak_mat_tmp121 + weak_mat_tmp243);
    const s_t weak_mat_tmp245 = -weak_mat_tmp202;
    const s_t weak_mat_tmp246 = weak_mat_tmp157*weak_mat_tmp229 + weak_mat_tmp203 + weak_mat_tmp225*weak_mat_tmp75;
    const s_t weak_mat_tmp247 = s_t(2)*weak_mat_tmp79;
    const s_t weak_mat_tmp248 = weak_mat_tmp163*weak_mat_tmp227 + weak_mat_tmp164*weak_mat_tmp231 + weak_mat_tmp206 + weak_mat_tmp39*(s_t(4)*gu2*weak_mat_tmp1 - weak_mat_tmp247);
    const s_t weak_mat_tmp249 = weak_mat_tmp7*(-weak_mat_tmp219 - weak_mat_tmp239);
    const s_t weak_mat_tmp250 = weak_mat_tmp14*weak_mat_tmp67;
    const s_t weak_mat_tmp251 = weak_mat_tmp31*weak_mat_tmp67;
    const s_t weak_mat_tmp252 = s_t(2)*weak_mat_tmp26;
    const s_t weak_mat_tmp253 = weak_mat_tmp252 + weak_mat_tmp34;
    const s_t weak_mat_tmp254 = s_t(2)*weak_mat_tmp22;
    const s_t weak_mat_tmp255 = weak_mat_tmp254 + weak_mat_tmp36;
    const s_t weak_mat_tmp256 = weak_mat_tmp41*weak_mat_tmp73;
    const s_t weak_mat_tmp257 = weak_mat_tmp60*weak_mat_tmp73;
    const s_t weak_mat_tmp258 = weak_mat_tmp14*weak_mat_tmp66;
    const s_t weak_mat_tmp259 = weak_mat_tmp250*weak_mat_tmp83 + weak_mat_tmp258*weak_mat_tmp84;
    const s_t weak_mat_tmp260 = weak_mat_tmp2*weak_mat_tmp74;
    const s_t weak_mat_tmp261 = gu7*weak_mat_tmp75;
    const s_t weak_mat_tmp262 = weak_mat_tmp41*weak_mat_tmp76;
    const s_t weak_mat_tmp263 = weak_mat_tmp256*weak_mat_tmp92 + weak_mat_tmp262*weak_mat_tmp89 + weak_mat_tmp39*(-weak_mat_tmp260 - weak_mat_tmp261);
    const s_t weak_mat_tmp264 = weak_mat_tmp126*weak_mat_tmp258 + weak_mat_tmp250*weak_mat_tmp74;
    const s_t weak_mat_tmp265 = s_t(2)*weak_mat_tmp48;
    const s_t weak_mat_tmp266 = weak_mat_tmp134*weak_mat_tmp262 + weak_mat_tmp135*weak_mat_tmp256 + weak_mat_tmp39*(-weak_mat_tmp112 - weak_mat_tmp265);
    const s_t weak_mat_tmp267 = weak_mat_tmp157*weak_mat_tmp258 + weak_mat_tmp250*weak_mat_tmp75;
    const s_t weak_mat_tmp268 = s_t(2)*weak_mat_tmp43;
    const s_t weak_mat_tmp269 = weak_mat_tmp163*weak_mat_tmp262 + weak_mat_tmp164*weak_mat_tmp256 + weak_mat_tmp39*(-weak_mat_tmp100 - weak_mat_tmp268);
    const s_t weak_mat_tmp270 = gu2*weak_mat_tmp122;
    const s_t weak_mat_tmp271 = gu2*weak_mat_tmp128;
    const s_t weak_mat_tmp272 = weak_mat_tmp144*weak_mat_tmp258 + weak_mat_tmp250*weak_mat_tmp90 - weak_mat_tmp271;
    const s_t weak_mat_tmp273 = gu2*weak_mat_tmp136;
    const s_t weak_mat_tmp274 = weak_mat_tmp149*weak_mat_tmp262 + weak_mat_tmp150*weak_mat_tmp256 - weak_mat_tmp273 + weak_mat_tmp39*(s_t(4)*weak_mat_tmp105 + weak_mat_tmp223);
    const s_t weak_mat_tmp275 = gu1*weak_mat_tmp122;
    const s_t weak_mat_tmp276 = -weak_mat_tmp275;
    const s_t weak_mat_tmp277 = gu1*weak_mat_tmp128;
    const s_t weak_mat_tmp278 = weak_mat_tmp171*weak_mat_tmp258 + weak_mat_tmp250*weak_mat_tmp91 + weak_mat_tmp277;
    const s_t weak_mat_tmp279 = gu1*weak_mat_tmp136;
    const s_t weak_mat_tmp280 = weak_mat_tmp176*weak_mat_tmp262 + weak_mat_tmp177*weak_mat_tmp256 + weak_mat_tmp279 + weak_mat_tmp39*(weak_mat_tmp180 + s_t(4)*weak_mat_tmp9);
    const s_t weak_mat_tmp281 = weak_mat_tmp7*(-weak_mat_tmp175 - weak_mat_tmp243);
    const s_t weak_mat_tmp282 = weak_mat_tmp14*weak_mat_tmp157;
    const s_t weak_mat_tmp283 = weak_mat_tmp157*weak_mat_tmp31;
    const s_t weak_mat_tmp284 = s_t(2)*weak_mat_tmp20;
    const s_t weak_mat_tmp285 = weak_mat_tmp186 + weak_mat_tmp284;
    const s_t weak_mat_tmp286 = weak_mat_tmp164*weak_mat_tmp41;
    const s_t weak_mat_tmp287 = weak_mat_tmp163*weak_mat_tmp60;
    const s_t weak_mat_tmp288 = weak_mat_tmp14*weak_mat_tmp75;
    const s_t weak_mat_tmp289 = weak_mat_tmp144*weak_mat_tmp288 + weak_mat_tmp282*weak_mat_tmp90;
    const s_t weak_mat_tmp290 = gu6*weak_mat_tmp66;
    const s_t weak_mat_tmp291 = weak_mat_tmp163*weak_mat_tmp41;
    const s_t weak_mat_tmp292 = weak_mat_tmp149*weak_mat_tmp286 + weak_mat_tmp150*weak_mat_tmp291 + weak_mat_tmp39*(-weak_mat_tmp260 - weak_mat_tmp290);
    const s_t weak_mat_tmp293 = weak_mat_tmp126*weak_mat_tmp288 + weak_mat_tmp282*weak_mat_tmp74;
    const s_t weak_mat_tmp294 = s_t(2)*weak_mat_tmp55;
    const s_t weak_mat_tmp295 = weak_mat_tmp134*weak_mat_tmp286 + weak_mat_tmp135*weak_mat_tmp291 + weak_mat_tmp39*(-weak_mat_tmp197 - weak_mat_tmp294);
    const s_t weak_mat_tmp296 = -weak_mat_tmp270;
    const s_t weak_mat_tmp297 = weak_mat_tmp271 + weak_mat_tmp282*weak_mat_tmp83 + weak_mat_tmp288*weak_mat_tmp84;
    const s_t weak_mat_tmp298 = weak_mat_tmp273 + weak_mat_tmp286*weak_mat_tmp89 + weak_mat_tmp291*weak_mat_tmp92 + weak_mat_tmp39*(s_t(4)*gu6*weak_mat_tmp1 - weak_mat_tmp222);
    const s_t weak_mat_tmp299 = weak_mat_tmp11*weak_mat_tmp122;
    const s_t weak_mat_tmp300 = weak_mat_tmp11*weak_mat_tmp128;
    const s_t weak_mat_tmp301 = weak_mat_tmp171*weak_mat_tmp288 + weak_mat_tmp282*weak_mat_tmp91 - weak_mat_tmp300;
    const s_t weak_mat_tmp302 = weak_mat_tmp11*weak_mat_tmp136;
    const s_t weak_mat_tmp303 = weak_mat_tmp176*weak_mat_tmp286 + weak_mat_tmp177*weak_mat_tmp291 - weak_mat_tmp302 + weak_mat_tmp39*(s_t(4)*weak_mat_tmp1*weak_mat_tmp2 - weak_mat_tmp4);
    const s_t weak_mat_tmp304 = weak_mat_tmp7*(weak_mat_tmp148 + weak_mat_tmp209);
    const s_t weak_mat_tmp305 = weak_mat_tmp126*weak_mat_tmp14;
    const s_t weak_mat_tmp306 = weak_mat_tmp126*weak_mat_tmp31;
    const s_t weak_mat_tmp307 = weak_mat_tmp135*weak_mat_tmp41;
    const s_t weak_mat_tmp308 = weak_mat_tmp134*weak_mat_tmp60;
    const s_t weak_mat_tmp309 = weak_mat_tmp14*weak_mat_tmp74;
    const s_t weak_mat_tmp310 = weak_mat_tmp171*weak_mat_tmp309 + weak_mat_tmp305*weak_mat_tmp91;
    const s_t weak_mat_tmp311 = weak_mat_tmp134*weak_mat_tmp41;
    const s_t weak_mat_tmp312 = weak_mat_tmp176*weak_mat_tmp307 + weak_mat_tmp177*weak_mat_tmp311 + weak_mat_tmp39*(-weak_mat_tmp261 - weak_mat_tmp290);
    const s_t weak_mat_tmp313 = -weak_mat_tmp277 + weak_mat_tmp305*weak_mat_tmp83 + weak_mat_tmp309*weak_mat_tmp84;
    const s_t weak_mat_tmp314 = -weak_mat_tmp279 + weak_mat_tmp307*weak_mat_tmp89 + weak_mat_tmp311*weak_mat_tmp92 + weak_mat_tmp39*(s_t(4)*gu5*gu6 - weak_mat_tmp181);
    const s_t weak_mat_tmp315 = -weak_mat_tmp299;
    const s_t weak_mat_tmp316 = weak_mat_tmp144*weak_mat_tmp309 + weak_mat_tmp300 + weak_mat_tmp305*weak_mat_tmp90;
    const s_t weak_mat_tmp317 = weak_mat_tmp149*weak_mat_tmp307 + weak_mat_tmp150*weak_mat_tmp311 + weak_mat_tmp302 + weak_mat_tmp39*(s_t(4)*weak_mat_tmp0 + weak_mat_tmp6);
    const s_t weak_mat_tmp318 = weak_mat_tmp7*(weak_mat_tmp205 + weak_mat_tmp247);
    const s_t weak_mat_tmp319 = weak_mat_tmp14*weak_mat_tmp84;
    const s_t weak_mat_tmp320 = weak_mat_tmp31*weak_mat_tmp84;
    const s_t weak_mat_tmp321 = weak_mat_tmp252 + weak_mat_tmp33;
    const s_t weak_mat_tmp322 = weak_mat_tmp254 + weak_mat_tmp37;
    const s_t weak_mat_tmp323 = weak_mat_tmp41*weak_mat_tmp89;
    const s_t weak_mat_tmp324 = weak_mat_tmp60*weak_mat_tmp89;
    const s_t weak_mat_tmp325 = weak_mat_tmp14*weak_mat_tmp83;
    const s_t weak_mat_tmp326 = weak_mat_tmp144*weak_mat_tmp325 + weak_mat_tmp319*weak_mat_tmp90;
    const s_t weak_mat_tmp327 = weak_mat_tmp41*weak_mat_tmp92;
    const s_t weak_mat_tmp328 = weak_mat_tmp149*weak_mat_tmp327 + weak_mat_tmp150*weak_mat_tmp323 + weak_mat_tmp39*(-weak_mat_tmp101 - weak_mat_tmp268);
    const s_t weak_mat_tmp329 = weak_mat_tmp171*weak_mat_tmp325 + weak_mat_tmp319*weak_mat_tmp91;
    const s_t weak_mat_tmp330 = weak_mat_tmp176*weak_mat_tmp327 + weak_mat_tmp177*weak_mat_tmp323 + weak_mat_tmp39*(-weak_mat_tmp111 - weak_mat_tmp265);
    const s_t weak_mat_tmp331 = weak_mat_tmp7*(weak_mat_tmp133 + weak_mat_tmp236);
    const s_t weak_mat_tmp332 = weak_mat_tmp14*weak_mat_tmp144;
    const s_t weak_mat_tmp333 = weak_mat_tmp144*weak_mat_tmp31;
    const s_t weak_mat_tmp334 = weak_mat_tmp185 + weak_mat_tmp284;
    const s_t weak_mat_tmp335 = weak_mat_tmp149*weak_mat_tmp41;
    const s_t weak_mat_tmp336 = weak_mat_tmp149*weak_mat_tmp60;
    const s_t weak_mat_tmp337 = weak_mat_tmp14*weak_mat_tmp171*weak_mat_tmp90 + weak_mat_tmp332*weak_mat_tmp91;
    const s_t weak_mat_tmp338 = weak_mat_tmp176*weak_mat_tmp41;
    const s_t weak_mat_tmp339 = weak_mat_tmp150*weak_mat_tmp338 + weak_mat_tmp177*weak_mat_tmp335 + weak_mat_tmp39*(-weak_mat_tmp196 - weak_mat_tmp294);
    const s_t weak_mat_tmp340 = weak_mat_tmp7*(-weak_mat_tmp162 - weak_mat_tmp213);
    const s_t weak_mat_tmp341 = weak_mat_tmp171*weak_mat_tmp31;
    const s_t weak_mat_tmp342 = weak_mat_tmp176*weak_mat_tmp60;
    const s_t material0 = trial_grad0*(iso_history_qp[0] + iso_relaxation_gamma*(C01*(weak_mat_tmp39*(weak_mat_tmp35 + weak_mat_tmp38) + s_t(2)*weak_mat_tmp40*weak_mat_tmp53 + weak_mat_tmp54*weak_mat_tmp61) + C10*(s_t(4)*weak_mat_tmp11*weak_mat_tmp16 + weak_mat_tmp13 + weak_mat_tmp17*weak_mat_tmp32)) + weak_mat_tmp3*weak_mat_tmp8) + trial_grad1*(iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_mat_tmp104 + weak_mat_tmp61*weak_mat_tmp99) + C10*(weak_mat_tmp32*weak_mat_tmp96 + weak_mat_tmp98)) + weak_mat_tmp8*weak_mat_tmp94) + trial_grad2*(iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_mat_tmp110*weak_mat_tmp61 + weak_mat_tmp115) + C10*(weak_mat_tmp107*weak_mat_tmp32 + weak_mat_tmp109)) + weak_mat_tmp106*weak_mat_tmp8) + trial_grad3*(iso_relaxation_gamma*(C01*(weak_mat_tmp61*weak_mat_tmp70 + weak_mat_tmp78) + C10*(weak_mat_tmp32*weak_mat_tmp65 + weak_mat_tmp69)) + weak_mat_tmp63*weak_mat_tmp8) + trial_grad4*(iso_relaxation_gamma*(C01*(weak_mat_tmp161*weak_mat_tmp61 + weak_mat_tmp166) + C10*(weak_mat_tmp156*weak_mat_tmp32 + weak_mat_tmp160)) + weak_mat_tmp153*weak_mat_tmp8 + weak_mat_tmp154) + trial_grad5*(iso_relaxation_gamma*(C01*(weak_mat_tmp131*weak_mat_tmp61 + weak_mat_tmp138) + C10*(weak_mat_tmp125*weak_mat_tmp32 + weak_mat_tmp130)) + weak_mat_tmp118*weak_mat_tmp8 + weak_mat_tmp124) + trial_grad6*(iso_relaxation_gamma*(C01*(weak_mat_tmp61*weak_mat_tmp86 + weak_mat_tmp93) + C10*(weak_mat_tmp32*weak_mat_tmp82 + weak_mat_tmp85)) + weak_mat_tmp8*weak_mat_tmp81) + trial_grad7*(iso_relaxation_gamma*(C01*(weak_mat_tmp147*weak_mat_tmp61 + weak_mat_tmp152) + C10*(weak_mat_tmp143*weak_mat_tmp32 + weak_mat_tmp146)) + weak_mat_tmp140*weak_mat_tmp8 + weak_mat_tmp142) + trial_grad8*(iso_relaxation_gamma*(C01*(weak_mat_tmp174*weak_mat_tmp61 + weak_mat_tmp179) + C10*(weak_mat_tmp170*weak_mat_tmp32 + weak_mat_tmp173)) + weak_mat_tmp167*weak_mat_tmp8 + weak_mat_tmp168);
    const s_t material1 = trial_grad0*(iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_mat_tmp104 + weak_mat_tmp189*weak_mat_tmp54) + C10*(weak_mat_tmp17*weak_mat_tmp184 + weak_mat_tmp98)) + weak_mat_tmp182*weak_mat_tmp3) + trial_grad1*(iso_history_qp[1] + iso_relaxation_gamma*(C01*(s_t(2)*weak_mat_tmp102*weak_mat_tmp188 + weak_mat_tmp189*weak_mat_tmp99 + weak_mat_tmp39*(weak_mat_tmp187 + weak_mat_tmp35)) + C10*(s_t(4)*gu1*weak_mat_tmp183 + weak_mat_tmp13 + weak_mat_tmp184*weak_mat_tmp96)) + weak_mat_tmp182*weak_mat_tmp94) + trial_grad2*(iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_mat_tmp110*weak_mat_tmp189 + weak_mat_tmp198) + C10*(weak_mat_tmp107*weak_mat_tmp184 + weak_mat_tmp195)) + weak_mat_tmp106*weak_mat_tmp182) + trial_grad3*(iso_relaxation_gamma*(C01*(weak_mat_tmp189*weak_mat_tmp70 + weak_mat_tmp214) + C10*(weak_mat_tmp184*weak_mat_tmp65 + weak_mat_tmp212)) + weak_mat_tmp182*weak_mat_tmp63 + weak_mat_tmp211) + trial_grad4*(iso_relaxation_gamma*(C01*(weak_mat_tmp161*weak_mat_tmp189 + weak_mat_tmp201) + C10*(weak_mat_tmp156*weak_mat_tmp184 + weak_mat_tmp199)) + weak_mat_tmp153*weak_mat_tmp182) + trial_grad5*(iso_relaxation_gamma*(C01*(weak_mat_tmp131*weak_mat_tmp189 + weak_mat_tmp207) + C10*(weak_mat_tmp125*weak_mat_tmp184 + weak_mat_tmp204)) + weak_mat_tmp118*weak_mat_tmp182 + weak_mat_tmp202) + trial_grad6*(iso_relaxation_gamma*(C01*(weak_mat_tmp189*weak_mat_tmp86 + weak_mat_tmp210) + C10*(weak_mat_tmp184*weak_mat_tmp82 + weak_mat_tmp208)) + weak_mat_tmp141 + weak_mat_tmp182*weak_mat_tmp81) + trial_grad7*(iso_relaxation_gamma*(C01*(weak_mat_tmp147*weak_mat_tmp189 + weak_mat_tmp194) + C10*(weak_mat_tmp143*weak_mat_tmp184 + weak_mat_tmp191)) + weak_mat_tmp140*weak_mat_tmp182) + trial_grad8*(iso_relaxation_gamma*(C01*(weak_mat_tmp174*weak_mat_tmp189 + weak_mat_tmp221) + C10*(weak_mat_tmp170*weak_mat_tmp184 + weak_mat_tmp218)) + weak_mat_tmp167*weak_mat_tmp182 + weak_mat_tmp216);
    const s_t material2 = trial_grad0*(iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_mat_tmp115 + weak_mat_tmp228*weak_mat_tmp54) + C10*(weak_mat_tmp109 + weak_mat_tmp17*weak_mat_tmp226)) + weak_mat_tmp224*weak_mat_tmp3) + trial_grad1*(iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_mat_tmp198 + weak_mat_tmp228*weak_mat_tmp99) + C10*(weak_mat_tmp195 + weak_mat_tmp226*weak_mat_tmp96)) + weak_mat_tmp224*weak_mat_tmp94) + trial_grad2*(iso_history_qp[2] + iso_relaxation_gamma*(C01*(weak_mat_tmp110*weak_mat_tmp228 + s_t(2)*weak_mat_tmp113*weak_mat_tmp227 + weak_mat_tmp39*(weak_mat_tmp187 + weak_mat_tmp38)) + C10*(s_t(4)*gu2*weak_mat_tmp225 + weak_mat_tmp107*weak_mat_tmp226 + weak_mat_tmp13)) + weak_mat_tmp106*weak_mat_tmp224) + trial_grad3*(iso_relaxation_gamma*(C01*(weak_mat_tmp228*weak_mat_tmp70 + weak_mat_tmp237) + C10*(weak_mat_tmp226*weak_mat_tmp65 + weak_mat_tmp235)) + weak_mat_tmp123 + weak_mat_tmp224*weak_mat_tmp63) + trial_grad4*(iso_relaxation_gamma*(C01*(weak_mat_tmp161*weak_mat_tmp228 + weak_mat_tmp248) + C10*(weak_mat_tmp156*weak_mat_tmp226 + weak_mat_tmp246)) + weak_mat_tmp153*weak_mat_tmp224 + weak_mat_tmp245) + trial_grad5*(iso_relaxation_gamma*(C01*(weak_mat_tmp131*weak_mat_tmp228 + weak_mat_tmp232) + C10*(weak_mat_tmp125*weak_mat_tmp226 + weak_mat_tmp230)) + weak_mat_tmp118*weak_mat_tmp224) + trial_grad6*(iso_relaxation_gamma*(C01*(weak_mat_tmp228*weak_mat_tmp86 + weak_mat_tmp244) + C10*(weak_mat_tmp226*weak_mat_tmp82 + weak_mat_tmp242)) + weak_mat_tmp224*weak_mat_tmp81 + weak_mat_tmp241) + trial_grad7*(iso_relaxation_gamma*(C01*(weak_mat_tmp147*weak_mat_tmp228 + weak_mat_tmp240) + C10*(weak_mat_tmp143*weak_mat_tmp226 + weak_mat_tmp238)) + weak_mat_tmp140*weak_mat_tmp224 + weak_mat_tmp215) + trial_grad8*(iso_relaxation_gamma*(C01*(weak_mat_tmp174*weak_mat_tmp228 + weak_mat_tmp234) + C10*(weak_mat_tmp170*weak_mat_tmp226 + weak_mat_tmp233)) + weak_mat_tmp167*weak_mat_tmp224);
    const s_t material3 = trial_grad0*(iso_relaxation_gamma*(C01*(weak_mat_tmp257*weak_mat_tmp54 + weak_mat_tmp78) + C10*(weak_mat_tmp17*weak_mat_tmp251 + weak_mat_tmp69)) + weak_mat_tmp249*weak_mat_tmp3) + trial_grad1*(iso_relaxation_gamma*(C01*(weak_mat_tmp214 + weak_mat_tmp257*weak_mat_tmp99) + C10*(weak_mat_tmp212 + weak_mat_tmp251*weak_mat_tmp96)) + weak_mat_tmp211 + weak_mat_tmp249*weak_mat_tmp94) + trial_grad2*(iso_relaxation_gamma*(C01*(weak_mat_tmp110*weak_mat_tmp257 + weak_mat_tmp237) + C10*(weak_mat_tmp107*weak_mat_tmp251 + weak_mat_tmp235)) + weak_mat_tmp106*weak_mat_tmp249 + weak_mat_tmp123) + trial_grad3*(iso_history_qp[0] + iso_relaxation_gamma*(C01*(s_t(2)*weak_mat_tmp256*weak_mat_tmp76 + weak_mat_tmp257*weak_mat_tmp70 + weak_mat_tmp39*(weak_mat_tmp253 + weak_mat_tmp255)) + C10*(s_t(4)*gu3*weak_mat_tmp250 + weak_mat_tmp13 + weak_mat_tmp251*weak_mat_tmp65)) + weak_mat_tmp249*weak_mat_tmp63) + trial_grad4*(iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_mat_tmp161*weak_mat_tmp257 + weak_mat_tmp269) + C10*(weak_mat_tmp156*weak_mat_tmp251 + weak_mat_tmp267)) + weak_mat_tmp153*weak_mat_tmp249) + trial_grad5*(iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_mat_tmp131*weak_mat_tmp257 + weak_mat_tmp266) + C10*(weak_mat_tmp125*weak_mat_tmp251 + weak_mat_tmp264)) + weak_mat_tmp118*weak_mat_tmp249) + trial_grad6*(iso_relaxation_gamma*(C01*(weak_mat_tmp257*weak_mat_tmp86 + weak_mat_tmp263) + C10*(weak_mat_tmp251*weak_mat_tmp82 + weak_mat_tmp259)) + weak_mat_tmp249*weak_mat_tmp81) + trial_grad7*(iso_relaxation_gamma*(C01*(weak_mat_tmp147*weak_mat_tmp257 + weak_mat_tmp274) + C10*(weak_mat_tmp143*weak_mat_tmp251 + weak_mat_tmp272)) + weak_mat_tmp140*weak_mat_tmp249 + weak_mat_tmp270) + trial_grad8*(iso_relaxation_gamma*(C01*(weak_mat_tmp174*weak_mat_tmp257 + weak_mat_tmp280) + C10*(weak_mat_tmp170*weak_mat_tmp251 + weak_mat_tmp278)) + weak_mat_tmp167*weak_mat_tmp249 + weak_mat_tmp276);
    const s_t material4 = trial_grad0*(iso_relaxation_gamma*(C01*(weak_mat_tmp166 + weak_mat_tmp287*weak_mat_tmp54) + C10*(weak_mat_tmp160 + weak_mat_tmp17*weak_mat_tmp283)) + weak_mat_tmp154 + weak_mat_tmp281*weak_mat_tmp3) + trial_grad1*(iso_relaxation_gamma*(C01*(weak_mat_tmp201 + weak_mat_tmp287*weak_mat_tmp99) + C10*(weak_mat_tmp199 + weak_mat_tmp283*weak_mat_tmp96)) + weak_mat_tmp281*weak_mat_tmp94) + trial_grad2*(iso_relaxation_gamma*(C01*(weak_mat_tmp110*weak_mat_tmp287 + weak_mat_tmp248) + C10*(weak_mat_tmp107*weak_mat_tmp283 + weak_mat_tmp246)) + weak_mat_tmp106*weak_mat_tmp281 + weak_mat_tmp245) + trial_grad3*(iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_mat_tmp269 + weak_mat_tmp287*weak_mat_tmp70) + C10*(weak_mat_tmp267 + weak_mat_tmp283*weak_mat_tmp65)) + weak_mat_tmp281*weak_mat_tmp63) + trial_grad4*(iso_history_qp[1] + iso_relaxation_gamma*(C01*(weak_mat_tmp161*weak_mat_tmp287 + s_t(2)*weak_mat_tmp163*weak_mat_tmp286 + weak_mat_tmp39*(weak_mat_tmp253 + weak_mat_tmp285)) + C10*(s_t(4)*weak_mat_tmp1*weak_mat_tmp282 + weak_mat_tmp13 + weak_mat_tmp156*weak_mat_tmp283)) + weak_mat_tmp153*weak_mat_tmp281) + trial_grad5*(iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_mat_tmp131*weak_mat_tmp287 + weak_mat_tmp295) + C10*(weak_mat_tmp125*weak_mat_tmp283 + weak_mat_tmp293)) + weak_mat_tmp118*weak_mat_tmp281) + trial_grad6*(iso_relaxation_gamma*(C01*(weak_mat_tmp287*weak_mat_tmp86 + weak_mat_tmp298) + C10*(weak_mat_tmp283*weak_mat_tmp82 + weak_mat_tmp297)) + weak_mat_tmp281*weak_mat_tmp81 + weak_mat_tmp296) + trial_grad7*(iso_relaxation_gamma*(C01*(weak_mat_tmp147*weak_mat_tmp287 + weak_mat_tmp292) + C10*(weak_mat_tmp143*weak_mat_tmp283 + weak_mat_tmp289)) + weak_mat_tmp140*weak_mat_tmp281) + trial_grad8*(iso_relaxation_gamma*(C01*(weak_mat_tmp174*weak_mat_tmp287 + weak_mat_tmp303) + C10*(weak_mat_tmp170*weak_mat_tmp283 + weak_mat_tmp301)) + weak_mat_tmp167*weak_mat_tmp281 + weak_mat_tmp299);
    const s_t material5 = trial_grad0*(iso_relaxation_gamma*(C01*(weak_mat_tmp138 + weak_mat_tmp308*weak_mat_tmp54) + C10*(weak_mat_tmp130 + weak_mat_tmp17*weak_mat_tmp306)) + weak_mat_tmp124 + weak_mat_tmp3*weak_mat_tmp304) + trial_grad1*(iso_relaxation_gamma*(C01*(weak_mat_tmp207 + weak_mat_tmp308*weak_mat_tmp99) + C10*(weak_mat_tmp204 + weak_mat_tmp306*weak_mat_tmp96)) + weak_mat_tmp202 + weak_mat_tmp304*weak_mat_tmp94) + trial_grad2*(iso_relaxation_gamma*(C01*(weak_mat_tmp110*weak_mat_tmp308 + weak_mat_tmp232) + C10*(weak_mat_tmp107*weak_mat_tmp306 + weak_mat_tmp230)) + weak_mat_tmp106*weak_mat_tmp304) + trial_grad3*(iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_mat_tmp266 + weak_mat_tmp308*weak_mat_tmp70) + C10*(weak_mat_tmp264 + weak_mat_tmp306*weak_mat_tmp65)) + weak_mat_tmp304*weak_mat_tmp63) + trial_grad4*(iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_mat_tmp161*weak_mat_tmp308 + weak_mat_tmp295) + C10*(weak_mat_tmp156*weak_mat_tmp306 + weak_mat_tmp293)) + weak_mat_tmp153*weak_mat_tmp304) + trial_grad5*(iso_history_qp[2] + iso_relaxation_gamma*(C01*(weak_mat_tmp131*weak_mat_tmp308 + s_t(2)*weak_mat_tmp134*weak_mat_tmp307 + weak_mat_tmp39*(weak_mat_tmp255 + weak_mat_tmp285)) + C10*(s_t(4)*gu5*weak_mat_tmp305 + weak_mat_tmp125*weak_mat_tmp306 + weak_mat_tmp13)) + weak_mat_tmp118*weak_mat_tmp304) + trial_grad6*(iso_relaxation_gamma*(C01*(weak_mat_tmp308*weak_mat_tmp86 + weak_mat_tmp314) + C10*(weak_mat_tmp306*weak_mat_tmp82 + weak_mat_tmp313)) + weak_mat_tmp275 + weak_mat_tmp304*weak_mat_tmp81) + trial_grad7*(iso_relaxation_gamma*(C01*(weak_mat_tmp147*weak_mat_tmp308 + weak_mat_tmp317) + C10*(weak_mat_tmp143*weak_mat_tmp306 + weak_mat_tmp316)) + weak_mat_tmp140*weak_mat_tmp304 + weak_mat_tmp315) + trial_grad8*(iso_relaxation_gamma*(C01*(weak_mat_tmp174*weak_mat_tmp308 + weak_mat_tmp312) + C10*(weak_mat_tmp170*weak_mat_tmp306 + weak_mat_tmp310)) + weak_mat_tmp167*weak_mat_tmp304);
    const s_t material6 = trial_grad0*(iso_relaxation_gamma*(C01*(weak_mat_tmp324*weak_mat_tmp54 + weak_mat_tmp93) + C10*(weak_mat_tmp17*weak_mat_tmp320 + weak_mat_tmp85)) + weak_mat_tmp3*weak_mat_tmp318) + trial_grad1*(iso_relaxation_gamma*(C01*(weak_mat_tmp210 + weak_mat_tmp324*weak_mat_tmp99) + C10*(weak_mat_tmp208 + weak_mat_tmp320*weak_mat_tmp96)) + weak_mat_tmp141 + weak_mat_tmp318*weak_mat_tmp94) + trial_grad2*(iso_relaxation_gamma*(C01*(weak_mat_tmp110*weak_mat_tmp324 + weak_mat_tmp244) + C10*(weak_mat_tmp107*weak_mat_tmp320 + weak_mat_tmp242)) + weak_mat_tmp106*weak_mat_tmp318 + weak_mat_tmp241) + trial_grad3*(iso_relaxation_gamma*(C01*(weak_mat_tmp263 + weak_mat_tmp324*weak_mat_tmp70) + C10*(weak_mat_tmp259 + weak_mat_tmp320*weak_mat_tmp65)) + weak_mat_tmp318*weak_mat_tmp63) + trial_grad4*(iso_relaxation_gamma*(C01*(weak_mat_tmp161*weak_mat_tmp324 + weak_mat_tmp298) + C10*(weak_mat_tmp156*weak_mat_tmp320 + weak_mat_tmp297)) + weak_mat_tmp153*weak_mat_tmp318 + weak_mat_tmp296) + trial_grad5*(iso_relaxation_gamma*(C01*(weak_mat_tmp131*weak_mat_tmp324 + weak_mat_tmp314) + C10*(weak_mat_tmp125*weak_mat_tmp320 + weak_mat_tmp313)) + weak_mat_tmp118*weak_mat_tmp318 + weak_mat_tmp275) + trial_grad6*(iso_history_qp[0] + iso_relaxation_gamma*(C01*(s_t(2)*weak_mat_tmp323*weak_mat_tmp92 + weak_mat_tmp324*weak_mat_tmp86 + weak_mat_tmp39*(weak_mat_tmp321 + weak_mat_tmp322)) + C10*(s_t(4)*gu6*weak_mat_tmp319 + weak_mat_tmp13 + weak_mat_tmp320*weak_mat_tmp82)) + weak_mat_tmp318*weak_mat_tmp81) + trial_grad7*(iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_mat_tmp147*weak_mat_tmp324 + weak_mat_tmp328) + C10*(weak_mat_tmp143*weak_mat_tmp320 + weak_mat_tmp326)) + weak_mat_tmp140*weak_mat_tmp318) + trial_grad8*(iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_mat_tmp174*weak_mat_tmp324 + weak_mat_tmp330) + C10*(weak_mat_tmp170*weak_mat_tmp320 + weak_mat_tmp329)) + weak_mat_tmp167*weak_mat_tmp318);
    const s_t material7 = trial_grad0*(iso_relaxation_gamma*(C01*(weak_mat_tmp152 + weak_mat_tmp336*weak_mat_tmp54) + C10*(weak_mat_tmp146 + weak_mat_tmp17*weak_mat_tmp333)) + weak_mat_tmp142 + weak_mat_tmp3*weak_mat_tmp331) + trial_grad1*(iso_relaxation_gamma*(C01*(weak_mat_tmp194 + weak_mat_tmp336*weak_mat_tmp99) + C10*(weak_mat_tmp191 + weak_mat_tmp333*weak_mat_tmp96)) + weak_mat_tmp331*weak_mat_tmp94) + trial_grad2*(iso_relaxation_gamma*(C01*(weak_mat_tmp110*weak_mat_tmp336 + weak_mat_tmp240) + C10*(weak_mat_tmp107*weak_mat_tmp333 + weak_mat_tmp238)) + weak_mat_tmp106*weak_mat_tmp331 + weak_mat_tmp215) + trial_grad3*(iso_relaxation_gamma*(C01*(weak_mat_tmp274 + weak_mat_tmp336*weak_mat_tmp70) + C10*(weak_mat_tmp272 + weak_mat_tmp333*weak_mat_tmp65)) + weak_mat_tmp270 + weak_mat_tmp331*weak_mat_tmp63) + trial_grad4*(iso_relaxation_gamma*(C01*(weak_mat_tmp161*weak_mat_tmp336 + weak_mat_tmp292) + C10*(weak_mat_tmp156*weak_mat_tmp333 + weak_mat_tmp289)) + weak_mat_tmp153*weak_mat_tmp331) + trial_grad5*(iso_relaxation_gamma*(C01*(weak_mat_tmp131*weak_mat_tmp336 + weak_mat_tmp317) + C10*(weak_mat_tmp125*weak_mat_tmp333 + weak_mat_tmp316)) + weak_mat_tmp118*weak_mat_tmp331 + weak_mat_tmp315) + trial_grad6*(iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_mat_tmp328 + weak_mat_tmp336*weak_mat_tmp86) + C10*(weak_mat_tmp326 + weak_mat_tmp333*weak_mat_tmp82)) + weak_mat_tmp331*weak_mat_tmp81) + trial_grad7*(iso_history_qp[1] + iso_relaxation_gamma*(C01*(weak_mat_tmp147*weak_mat_tmp336 + s_t(2)*weak_mat_tmp150*weak_mat_tmp335 + weak_mat_tmp39*(weak_mat_tmp321 + weak_mat_tmp334)) + C10*(s_t(4)*gu7*weak_mat_tmp332 + weak_mat_tmp13 + weak_mat_tmp143*weak_mat_tmp333)) + weak_mat_tmp140*weak_mat_tmp331) + trial_grad8*(iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_mat_tmp174*weak_mat_tmp336 + weak_mat_tmp339) + C10*(weak_mat_tmp170*weak_mat_tmp333 + weak_mat_tmp337)) + weak_mat_tmp167*weak_mat_tmp331);
    const s_t material8 = trial_grad0*(iso_relaxation_gamma*(C01*(weak_mat_tmp179 + weak_mat_tmp342*weak_mat_tmp54) + C10*(weak_mat_tmp17*weak_mat_tmp341 + weak_mat_tmp173)) + weak_mat_tmp168 + weak_mat_tmp3*weak_mat_tmp340) + trial_grad1*(iso_relaxation_gamma*(C01*(weak_mat_tmp221 + weak_mat_tmp342*weak_mat_tmp99) + C10*(weak_mat_tmp218 + weak_mat_tmp341*weak_mat_tmp96)) + weak_mat_tmp216 + weak_mat_tmp340*weak_mat_tmp94) + trial_grad2*(iso_relaxation_gamma*(C01*(weak_mat_tmp110*weak_mat_tmp342 + weak_mat_tmp234) + C10*(weak_mat_tmp107*weak_mat_tmp341 + weak_mat_tmp233)) + weak_mat_tmp106*weak_mat_tmp340) + trial_grad3*(iso_relaxation_gamma*(C01*(weak_mat_tmp280 + weak_mat_tmp342*weak_mat_tmp70) + C10*(weak_mat_tmp278 + weak_mat_tmp341*weak_mat_tmp65)) + weak_mat_tmp276 + weak_mat_tmp340*weak_mat_tmp63) + trial_grad4*(iso_relaxation_gamma*(C01*(weak_mat_tmp161*weak_mat_tmp342 + weak_mat_tmp303) + C10*(weak_mat_tmp156*weak_mat_tmp341 + weak_mat_tmp301)) + weak_mat_tmp153*weak_mat_tmp340 + weak_mat_tmp299) + trial_grad5*(iso_relaxation_gamma*(C01*(weak_mat_tmp131*weak_mat_tmp342 + weak_mat_tmp312) + C10*(weak_mat_tmp125*weak_mat_tmp341 + weak_mat_tmp310)) + weak_mat_tmp118*weak_mat_tmp340) + trial_grad6*(iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_mat_tmp330 + weak_mat_tmp342*weak_mat_tmp86) + C10*(weak_mat_tmp329 + weak_mat_tmp341*weak_mat_tmp82)) + weak_mat_tmp340*weak_mat_tmp81) + trial_grad7*(iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_mat_tmp147*weak_mat_tmp342 + weak_mat_tmp339) + C10*(weak_mat_tmp143*weak_mat_tmp341 + weak_mat_tmp337)) + weak_mat_tmp140*weak_mat_tmp340) + trial_grad8*(iso_history_qp[2] + iso_relaxation_gamma*(C01*(weak_mat_tmp174*weak_mat_tmp342 + s_t(2)*weak_mat_tmp177*weak_mat_tmp338 + weak_mat_tmp39*(weak_mat_tmp322 + weak_mat_tmp334)) + C10*(weak_mat_tmp13 + s_t(4)*weak_mat_tmp158*weak_mat_tmp171 + weak_mat_tmp170*weak_mat_tmp341)) + weak_mat_tmp167*weak_mat_tmp340);
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
static SFEM_INLINE void mooney_rivlin_prony_d3_simplex_tet4_apply_block(
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
    const s_t weak_mat_tmp5 = weak_mat_tmp1*weak_mat_tmp2;
    const s_t weak_mat_tmp6 = -s_t(2)*weak_mat_tmp5;
    const s_t weak_mat_tmp7 = ((s_t(1) / s_t(2)))*K;
    const s_t weak_mat_tmp8 = weak_mat_tmp7*(-weak_mat_tmp4 - weak_mat_tmp6);
    const s_t weak_mat_tmp9 = gu3*weak_mat_tmp2;
    const s_t weak_mat_tmp10 = gu6*weak_mat_tmp1;
    const s_t weak_mat_tmp11 = gu0 + s_t(1);
    const s_t weak_mat_tmp12 = gu1*gu5*gu6 - gu1*weak_mat_tmp9 + gu2*gu3*gu7 - gu2*weak_mat_tmp10 - weak_mat_tmp0*weak_mat_tmp11 + weak_mat_tmp1*weak_mat_tmp11*weak_mat_tmp2;
    const s_t weak_mat_tmp13 = s_t(2)/pow(weak_mat_tmp12, (s_t(2) / s_t(3)));
    const s_t weak_mat_tmp14 = pow(weak_mat_tmp12, (s_t(-5) / s_t(3)));
    const s_t weak_mat_tmp15 = ((s_t(2) / s_t(3)))*weak_mat_tmp0 - (s_t(2) / s_t(3))*weak_mat_tmp5;
    const s_t weak_mat_tmp16 = weak_mat_tmp14*weak_mat_tmp15;
    const s_t weak_mat_tmp17 = ((s_t(5) / s_t(3)))*weak_mat_tmp0 - (s_t(5) / s_t(3))*weak_mat_tmp5;
    const s_t weak_mat_tmp18 = pow_2(gu3);
    const s_t weak_mat_tmp19 = pow_2(gu6);
    const s_t weak_mat_tmp20 = pow_2(weak_mat_tmp11);
    const s_t weak_mat_tmp21 = weak_mat_tmp18 + weak_mat_tmp19 + weak_mat_tmp20;
    const s_t weak_mat_tmp22 = pow_2(gu1);
    const s_t weak_mat_tmp23 = pow_2(gu7);
    const s_t weak_mat_tmp24 = pow_2(weak_mat_tmp1);
    const s_t weak_mat_tmp25 = weak_mat_tmp22 + weak_mat_tmp23 + weak_mat_tmp24;
    const s_t weak_mat_tmp26 = pow_2(gu2);
    const s_t weak_mat_tmp27 = pow_2(gu5);
    const s_t weak_mat_tmp28 = pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp29 = weak_mat_tmp26 + weak_mat_tmp27 + weak_mat_tmp28;
    const s_t weak_mat_tmp30 = weak_mat_tmp21 + weak_mat_tmp25 + weak_mat_tmp29;
    const s_t weak_mat_tmp31 = weak_mat_tmp30/pow(weak_mat_tmp12, (s_t(8) / s_t(3)));
    const s_t weak_mat_tmp32 = weak_mat_tmp15*weak_mat_tmp31;
    const s_t weak_mat_tmp33 = s_t(2)*weak_mat_tmp27;
    const s_t weak_mat_tmp34 = s_t(2)*weak_mat_tmp28;
    const s_t weak_mat_tmp35 = weak_mat_tmp33 + weak_mat_tmp34;
    const s_t weak_mat_tmp36 = s_t(2)*weak_mat_tmp23;
    const s_t weak_mat_tmp37 = s_t(2)*weak_mat_tmp24;
    const s_t weak_mat_tmp38 = weak_mat_tmp36 + weak_mat_tmp37;
    const s_t weak_mat_tmp39 = pow(weak_mat_tmp12, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp40 = ((s_t(4) / s_t(3)))*weak_mat_tmp0 - (s_t(4) / s_t(3))*weak_mat_tmp5;
    const s_t weak_mat_tmp41 = pow(weak_mat_tmp12, (s_t(-7) / s_t(3)));
    const s_t weak_mat_tmp42 = gu6*gu7;
    const s_t weak_mat_tmp43 = gu1*weak_mat_tmp11;
    const s_t weak_mat_tmp44 = gu3*weak_mat_tmp1;
    const s_t weak_mat_tmp45 = weak_mat_tmp42 + weak_mat_tmp43 + weak_mat_tmp44;
    const s_t weak_mat_tmp46 = s_t(2)*gu1;
    const s_t weak_mat_tmp47 = gu3*gu5;
    const s_t weak_mat_tmp48 = gu2*weak_mat_tmp11;
    const s_t weak_mat_tmp49 = gu6*weak_mat_tmp2;
    const s_t weak_mat_tmp50 = weak_mat_tmp47 + weak_mat_tmp48 + weak_mat_tmp49;
    const s_t weak_mat_tmp51 = s_t(2)*gu2;
    const s_t weak_mat_tmp52 = s_t(2)*weak_mat_tmp11;
    const s_t weak_mat_tmp53 = weak_mat_tmp41*(s_t(2)*weak_mat_tmp11*weak_mat_tmp30 - weak_mat_tmp21*weak_mat_tmp52 - weak_mat_tmp45*weak_mat_tmp46 - weak_mat_tmp50*weak_mat_tmp51);
    const s_t weak_mat_tmp54 = ((s_t(7) / s_t(3)))*weak_mat_tmp0 - (s_t(7) / s_t(3))*weak_mat_tmp5;
    const s_t weak_mat_tmp55 = gu1*gu2;
    const s_t weak_mat_tmp56 = gu5*weak_mat_tmp1;
    const s_t weak_mat_tmp57 = gu7*weak_mat_tmp2;
    const s_t weak_mat_tmp58 = weak_mat_tmp55 + weak_mat_tmp56 + weak_mat_tmp57;
    const s_t weak_mat_tmp59 = -(s_t(1) / s_t(2))*pow_2(weak_mat_tmp21) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp25) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp29) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp30) - pow_2(weak_mat_tmp45) - pow_2(weak_mat_tmp50) - pow_2(weak_mat_tmp58);
    const s_t weak_mat_tmp60 = weak_mat_tmp59/pow(weak_mat_tmp12, (s_t(10) / s_t(3)));
    const s_t weak_mat_tmp61 = weak_mat_tmp40*weak_mat_tmp60;
    const s_t weak_mat_tmp62 = gu1*weak_mat_tmp2;
    const s_t weak_mat_tmp63 = gu2*gu7 - weak_mat_tmp62;
    const s_t weak_mat_tmp64 = gu2*gu7;
    const s_t weak_mat_tmp65 = ((s_t(5) / s_t(3)))*weak_mat_tmp62 - (s_t(5) / s_t(3))*weak_mat_tmp64;
    const s_t weak_mat_tmp66 = s_t(2)*gu3;
    const s_t weak_mat_tmp67 = ((s_t(2) / s_t(3)))*weak_mat_tmp62 - (s_t(2) / s_t(3))*weak_mat_tmp64;
    const s_t weak_mat_tmp68 = weak_mat_tmp14*weak_mat_tmp52;
    const s_t weak_mat_tmp69 = weak_mat_tmp16*weak_mat_tmp66 + weak_mat_tmp67*weak_mat_tmp68;
    const s_t weak_mat_tmp70 = ((s_t(7) / s_t(3)))*weak_mat_tmp62 - (s_t(7) / s_t(3))*weak_mat_tmp64;
    const s_t weak_mat_tmp71 = gu5*weak_mat_tmp51;
    const s_t weak_mat_tmp72 = weak_mat_tmp1*weak_mat_tmp46;
    const s_t weak_mat_tmp73 = ((s_t(4) / s_t(3)))*weak_mat_tmp62 - (s_t(4) / s_t(3))*weak_mat_tmp64;
    const s_t weak_mat_tmp74 = s_t(2)*gu5;
    const s_t weak_mat_tmp75 = s_t(2)*weak_mat_tmp1;
    const s_t weak_mat_tmp76 = s_t(2)*gu3*weak_mat_tmp30 - weak_mat_tmp21*weak_mat_tmp66 - weak_mat_tmp45*weak_mat_tmp75 - weak_mat_tmp50*weak_mat_tmp74;
    const s_t weak_mat_tmp77 = weak_mat_tmp40*weak_mat_tmp41;
    const s_t weak_mat_tmp78 = weak_mat_tmp39*(-weak_mat_tmp71 - weak_mat_tmp72) + weak_mat_tmp53*weak_mat_tmp73 + weak_mat_tmp76*weak_mat_tmp77;
    const s_t weak_mat_tmp79 = gu1*gu5;
    const s_t weak_mat_tmp80 = gu2*weak_mat_tmp1;
    const s_t weak_mat_tmp81 = weak_mat_tmp79 - weak_mat_tmp80;
    const s_t weak_mat_tmp82 = ((s_t(5) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(5) / s_t(3))*weak_mat_tmp79;
    const s_t weak_mat_tmp83 = s_t(2)*gu6;
    const s_t weak_mat_tmp84 = ((s_t(2) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(2) / s_t(3))*weak_mat_tmp79;
    const s_t weak_mat_tmp85 = weak_mat_tmp16*weak_mat_tmp83 + weak_mat_tmp68*weak_mat_tmp84;
    const s_t weak_mat_tmp86 = ((s_t(7) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(7) / s_t(3))*weak_mat_tmp79;
    const s_t weak_mat_tmp87 = gu7*weak_mat_tmp46;
    const s_t weak_mat_tmp88 = weak_mat_tmp2*weak_mat_tmp51;
    const s_t weak_mat_tmp89 = ((s_t(4) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(4) / s_t(3))*weak_mat_tmp79;
    const s_t weak_mat_tmp90 = s_t(2)*gu7;
    const s_t weak_mat_tmp91 = s_t(2)*weak_mat_tmp2;
    const s_t weak_mat_tmp92 = s_t(2)*gu6*weak_mat_tmp30 - weak_mat_tmp21*weak_mat_tmp83 - weak_mat_tmp45*weak_mat_tmp90 - weak_mat_tmp50*weak_mat_tmp91;
    const s_t weak_mat_tmp93 = weak_mat_tmp39*(-weak_mat_tmp87 - weak_mat_tmp88) + weak_mat_tmp53*weak_mat_tmp89 + weak_mat_tmp77*weak_mat_tmp92;
    const s_t weak_mat_tmp94 = gu5*gu6 - weak_mat_tmp9;
    const s_t weak_mat_tmp95 = gu5*gu6;
    const s_t weak_mat_tmp96 = ((s_t(5) / s_t(3)))*weak_mat_tmp9 - (s_t(5) / s_t(3))*weak_mat_tmp95;
    const s_t weak_mat_tmp97 = ((s_t(2) / s_t(3)))*weak_mat_tmp9 - (s_t(2) / s_t(3))*weak_mat_tmp95;
    const s_t weak_mat_tmp98 = weak_mat_tmp16*weak_mat_tmp46 + weak_mat_tmp68*weak_mat_tmp97;
    const s_t weak_mat_tmp99 = ((s_t(7) / s_t(3)))*weak_mat_tmp9 - (s_t(7) / s_t(3))*weak_mat_tmp95;
    const s_t weak_mat_tmp100 = s_t(2)*weak_mat_tmp42;
    const s_t weak_mat_tmp101 = s_t(2)*weak_mat_tmp44;
    const s_t weak_mat_tmp102 = ((s_t(4) / s_t(3)))*weak_mat_tmp9 - (s_t(4) / s_t(3))*weak_mat_tmp95;
    const s_t weak_mat_tmp103 = s_t(2)*gu1*weak_mat_tmp30 - weak_mat_tmp25*weak_mat_tmp46 - weak_mat_tmp45*weak_mat_tmp52 - weak_mat_tmp51*weak_mat_tmp58;
    const s_t weak_mat_tmp104 = weak_mat_tmp102*weak_mat_tmp53 + weak_mat_tmp103*weak_mat_tmp77 + weak_mat_tmp39*(-weak_mat_tmp100 - weak_mat_tmp101);
    const s_t weak_mat_tmp105 = gu3*gu7;
    const s_t weak_mat_tmp106 = -weak_mat_tmp10 + weak_mat_tmp105;
    const s_t weak_mat_tmp107 = ((s_t(5) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(5) / s_t(3))*weak_mat_tmp105;
    const s_t weak_mat_tmp108 = ((s_t(2) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(2) / s_t(3))*weak_mat_tmp105;
    const s_t weak_mat_tmp109 = weak_mat_tmp108*weak_mat_tmp68 + weak_mat_tmp16*weak_mat_tmp51;
    const s_t weak_mat_tmp110 = ((s_t(7) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(7) / s_t(3))*weak_mat_tmp105;
    const s_t weak_mat_tmp111 = s_t(2)*weak_mat_tmp47;
    const s_t weak_mat_tmp112 = s_t(2)*weak_mat_tmp49;
    const s_t weak_mat_tmp113 = ((s_t(4) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(4) / s_t(3))*weak_mat_tmp105;
    const s_t weak_mat_tmp114 = s_t(2)*gu2*weak_mat_tmp30 - weak_mat_tmp29*weak_mat_tmp51 - weak_mat_tmp46*weak_mat_tmp58 - weak_mat_tmp50*weak_mat_tmp52;
    const s_t weak_mat_tmp115 = weak_mat_tmp113*weak_mat_tmp53 + weak_mat_tmp114*weak_mat_tmp77 + weak_mat_tmp39*(-weak_mat_tmp111 - weak_mat_tmp112);
    const s_t weak_mat_tmp116 = gu1*gu6;
    const s_t weak_mat_tmp117 = gu7*weak_mat_tmp11;
    const s_t weak_mat_tmp118 = weak_mat_tmp116 - weak_mat_tmp117;
    const s_t weak_mat_tmp119 = gu0*gu4;
    const s_t weak_mat_tmp120 = gu1*gu3;
    const s_t weak_mat_tmp121 = gu2*gu6;
    const s_t weak_mat_tmp122 = K*(gu0*gu8 - gu0*weak_mat_tmp0 + gu0 + gu1*weak_mat_tmp95 + gu2*weak_mat_tmp105 + gu4*gu8 - gu4*weak_mat_tmp121 + gu4 + gu8*weak_mat_tmp119 - gu8*weak_mat_tmp120 + gu8 - weak_mat_tmp0 + weak_mat_tmp119 - weak_mat_tmp120 - weak_mat_tmp121);
    const s_t weak_mat_tmp123 = gu7*weak_mat_tmp122;
    const s_t weak_mat_tmp124 = -weak_mat_tmp123;
    const s_t weak_mat_tmp125 = ((s_t(5) / s_t(3)))*gu7*weak_mat_tmp11 - (s_t(5) / s_t(3))*weak_mat_tmp116;
    const s_t weak_mat_tmp126 = ((s_t(2) / s_t(3)))*gu7*weak_mat_tmp11 - (s_t(2) / s_t(3))*weak_mat_tmp116;
    const s_t weak_mat_tmp127 = ((s_t(2) / s_t(3)))*weak_mat_tmp30;
    const s_t weak_mat_tmp128 = weak_mat_tmp127*weak_mat_tmp14;
    const s_t weak_mat_tmp129 = gu7*weak_mat_tmp128;
    const s_t weak_mat_tmp130 = weak_mat_tmp126*weak_mat_tmp68 + weak_mat_tmp129 + weak_mat_tmp16*weak_mat_tmp74;
    const s_t weak_mat_tmp131 = ((s_t(7) / s_t(3)))*gu7*weak_mat_tmp11 - (s_t(7) / s_t(3))*weak_mat_tmp116;
    const s_t weak_mat_tmp132 = gu2*gu3;
    const s_t weak_mat_tmp133 = s_t(2)*weak_mat_tmp132;
    const s_t weak_mat_tmp134 = ((s_t(4) / s_t(3)))*gu7*weak_mat_tmp11 - (s_t(4) / s_t(3))*weak_mat_tmp116;
    const s_t weak_mat_tmp135 = s_t(2)*gu5*weak_mat_tmp30 - weak_mat_tmp29*weak_mat_tmp74 - weak_mat_tmp50*weak_mat_tmp66 - weak_mat_tmp58*weak_mat_tmp75;
    const s_t weak_mat_tmp136 = ((s_t(4) / s_t(3)))*weak_mat_tmp41*weak_mat_tmp59;
    const s_t weak_mat_tmp137 = gu7*weak_mat_tmp136;
    const s_t weak_mat_tmp138 = weak_mat_tmp134*weak_mat_tmp53 + weak_mat_tmp135*weak_mat_tmp77 + weak_mat_tmp137 + weak_mat_tmp39*(s_t(4)*gu5*weak_mat_tmp11 - weak_mat_tmp133);
    const s_t weak_mat_tmp139 = gu5*weak_mat_tmp11;
    const s_t weak_mat_tmp140 = weak_mat_tmp132 - weak_mat_tmp139;
    const s_t weak_mat_tmp141 = gu5*weak_mat_tmp122;
    const s_t weak_mat_tmp142 = -weak_mat_tmp141;
    const s_t weak_mat_tmp143 = ((s_t(5) / s_t(3)))*gu5*weak_mat_tmp11 - (s_t(5) / s_t(3))*weak_mat_tmp132;
    const s_t weak_mat_tmp144 = ((s_t(2) / s_t(3)))*gu5*weak_mat_tmp11 - (s_t(2) / s_t(3))*weak_mat_tmp132;
    const s_t weak_mat_tmp145 = gu5*weak_mat_tmp128;
    const s_t weak_mat_tmp146 = weak_mat_tmp144*weak_mat_tmp68 + weak_mat_tmp145 + weak_mat_tmp16*weak_mat_tmp90;
    const s_t weak_mat_tmp147 = ((s_t(7) / s_t(3)))*gu5*weak_mat_tmp11 - (s_t(7) / s_t(3))*weak_mat_tmp132;
    const s_t weak_mat_tmp148 = s_t(2)*weak_mat_tmp116;
    const s_t weak_mat_tmp149 = ((s_t(4) / s_t(3)))*gu5*weak_mat_tmp11 - (s_t(4) / s_t(3))*weak_mat_tmp132;
    const s_t weak_mat_tmp150 = s_t(2)*gu7*weak_mat_tmp30 - weak_mat_tmp25*weak_mat_tmp90 - weak_mat_tmp45*weak_mat_tmp83 - weak_mat_tmp58*weak_mat_tmp91;
    const s_t weak_mat_tmp151 = gu5*weak_mat_tmp136;
    const s_t weak_mat_tmp152 = weak_mat_tmp149*weak_mat_tmp53 + weak_mat_tmp150*weak_mat_tmp77 + weak_mat_tmp151 + weak_mat_tmp39*(s_t(4)*gu7*weak_mat_tmp11 - weak_mat_tmp148);
    const s_t weak_mat_tmp153 = weak_mat_tmp11*weak_mat_tmp2 - weak_mat_tmp121;
    const s_t weak_mat_tmp154 = weak_mat_tmp122*weak_mat_tmp2;
    const s_t weak_mat_tmp155 = weak_mat_tmp11*weak_mat_tmp2;
    const s_t weak_mat_tmp156 = ((s_t(5) / s_t(3)))*weak_mat_tmp121 - (s_t(5) / s_t(3))*weak_mat_tmp155;
    const s_t weak_mat_tmp157 = ((s_t(2) / s_t(3)))*weak_mat_tmp121 - (s_t(2) / s_t(3))*weak_mat_tmp155;
    const s_t weak_mat_tmp158 = weak_mat_tmp14*weak_mat_tmp2;
    const s_t weak_mat_tmp159 = weak_mat_tmp127*weak_mat_tmp158;
    const s_t weak_mat_tmp160 = weak_mat_tmp157*weak_mat_tmp68 - weak_mat_tmp159 + weak_mat_tmp16*weak_mat_tmp75;
    const s_t weak_mat_tmp161 = ((s_t(7) / s_t(3)))*weak_mat_tmp121 - (s_t(7) / s_t(3))*weak_mat_tmp155;
    const s_t weak_mat_tmp162 = s_t(2)*weak_mat_tmp120;
    const s_t weak_mat_tmp163 = ((s_t(4) / s_t(3)))*weak_mat_tmp121 - (s_t(4) / s_t(3))*weak_mat_tmp155;
    const s_t weak_mat_tmp164 = s_t(2)*weak_mat_tmp1*weak_mat_tmp30 - weak_mat_tmp25*weak_mat_tmp75 - weak_mat_tmp45*weak_mat_tmp66 - weak_mat_tmp58*weak_mat_tmp74;
    const s_t weak_mat_tmp165 = weak_mat_tmp136*weak_mat_tmp2;
    const s_t weak_mat_tmp166 = weak_mat_tmp163*weak_mat_tmp53 + weak_mat_tmp164*weak_mat_tmp77 - weak_mat_tmp165 + weak_mat_tmp39*(s_t(4)*weak_mat_tmp1*weak_mat_tmp11 - weak_mat_tmp162);
    const s_t weak_mat_tmp167 = weak_mat_tmp1*weak_mat_tmp11 - weak_mat_tmp120;
    const s_t weak_mat_tmp168 = weak_mat_tmp1*weak_mat_tmp122;
    const s_t weak_mat_tmp169 = weak_mat_tmp1*weak_mat_tmp11;
    const s_t weak_mat_tmp170 = ((s_t(5) / s_t(3)))*weak_mat_tmp120 - (s_t(5) / s_t(3))*weak_mat_tmp169;
    const s_t weak_mat_tmp171 = ((s_t(2) / s_t(3)))*weak_mat_tmp120 - (s_t(2) / s_t(3))*weak_mat_tmp169;
    const s_t weak_mat_tmp172 = weak_mat_tmp1*weak_mat_tmp128;
    const s_t weak_mat_tmp173 = weak_mat_tmp16*weak_mat_tmp91 + weak_mat_tmp171*weak_mat_tmp68 - weak_mat_tmp172;
    const s_t weak_mat_tmp174 = ((s_t(7) / s_t(3)))*weak_mat_tmp120 - (s_t(7) / s_t(3))*weak_mat_tmp169;
    const s_t weak_mat_tmp175 = s_t(2)*weak_mat_tmp121;
    const s_t weak_mat_tmp176 = ((s_t(4) / s_t(3)))*weak_mat_tmp120 - (s_t(4) / s_t(3))*weak_mat_tmp169;
    const s_t weak_mat_tmp177 = s_t(2)*weak_mat_tmp2*weak_mat_tmp30 - weak_mat_tmp29*weak_mat_tmp91 - weak_mat_tmp50*weak_mat_tmp83 - weak_mat_tmp58*weak_mat_tmp90;
    const s_t weak_mat_tmp178 = weak_mat_tmp1*weak_mat_tmp136;
    const s_t weak_mat_tmp179 = weak_mat_tmp176*weak_mat_tmp53 + weak_mat_tmp177*weak_mat_tmp77 - weak_mat_tmp178 + weak_mat_tmp39*(s_t(4)*weak_mat_tmp11*weak_mat_tmp2 - weak_mat_tmp175);
    const s_t weak_mat_tmp180 = -s_t(2)*weak_mat_tmp95;
    const s_t weak_mat_tmp181 = s_t(2)*weak_mat_tmp9;
    const s_t weak_mat_tmp182 = weak_mat_tmp7*(-weak_mat_tmp180 - weak_mat_tmp181);
    const s_t weak_mat_tmp183 = weak_mat_tmp14*weak_mat_tmp97;
    const s_t weak_mat_tmp184 = weak_mat_tmp31*weak_mat_tmp97;
    const s_t weak_mat_tmp185 = s_t(2)*weak_mat_tmp18;
    const s_t weak_mat_tmp186 = s_t(2)*weak_mat_tmp19;
    const s_t weak_mat_tmp187 = weak_mat_tmp185 + weak_mat_tmp186;
    const s_t weak_mat_tmp188 = weak_mat_tmp103*weak_mat_tmp41;
    const s_t weak_mat_tmp189 = weak_mat_tmp102*weak_mat_tmp60;
    const s_t weak_mat_tmp190 = weak_mat_tmp14*weak_mat_tmp46;
    const s_t weak_mat_tmp191 = weak_mat_tmp144*weak_mat_tmp190 + weak_mat_tmp183*weak_mat_tmp90;
    const s_t weak_mat_tmp192 = gu6*weak_mat_tmp52;
    const s_t weak_mat_tmp193 = weak_mat_tmp102*weak_mat_tmp41;
    const s_t weak_mat_tmp194 = weak_mat_tmp149*weak_mat_tmp188 + weak_mat_tmp150*weak_mat_tmp193 + weak_mat_tmp39*(-weak_mat_tmp192 - weak_mat_tmp88);
    const s_t weak_mat_tmp195 = weak_mat_tmp108*weak_mat_tmp190 + weak_mat_tmp183*weak_mat_tmp51;
    const s_t weak_mat_tmp196 = s_t(2)*weak_mat_tmp56;
    const s_t weak_mat_tmp197 = s_t(2)*weak_mat_tmp57;
    const s_t weak_mat_tmp198 = weak_mat_tmp113*weak_mat_tmp188 + weak_mat_tmp114*weak_mat_tmp193 + weak_mat_tmp39*(-weak_mat_tmp196 - weak_mat_tmp197);
    const s_t weak_mat_tmp199 = weak_mat_tmp157*weak_mat_tmp190 + weak_mat_tmp183*weak_mat_tmp75;
    const s_t weak_mat_tmp200 = gu3*weak_mat_tmp52;
    const s_t weak_mat_tmp201 = weak_mat_tmp163*weak_mat_tmp188 + weak_mat_tmp164*weak_mat_tmp193 + weak_mat_tmp39*(-weak_mat_tmp200 - weak_mat_tmp71);
    const s_t weak_mat_tmp202 = gu6*weak_mat_tmp122;
    const s_t weak_mat_tmp203 = gu6*weak_mat_tmp128;
    const s_t weak_mat_tmp204 = weak_mat_tmp126*weak_mat_tmp190 + weak_mat_tmp183*weak_mat_tmp74 - weak_mat_tmp203;
    const s_t weak_mat_tmp205 = -s_t(2)*weak_mat_tmp80;
    const s_t weak_mat_tmp206 = gu6*weak_mat_tmp136;
    const s_t weak_mat_tmp207 = weak_mat_tmp134*weak_mat_tmp188 + weak_mat_tmp135*weak_mat_tmp193 - weak_mat_tmp206 + weak_mat_tmp39*(weak_mat_tmp205 + s_t(4)*weak_mat_tmp79);
    const s_t weak_mat_tmp208 = -weak_mat_tmp145 + weak_mat_tmp183*weak_mat_tmp83 + weak_mat_tmp190*weak_mat_tmp84;
    const s_t weak_mat_tmp209 = -s_t(2)*weak_mat_tmp117;
    const s_t weak_mat_tmp210 = -weak_mat_tmp151 + weak_mat_tmp188*weak_mat_tmp89 + weak_mat_tmp193*weak_mat_tmp92 + weak_mat_tmp39*(s_t(4)*weak_mat_tmp116 + weak_mat_tmp209);
    const s_t weak_mat_tmp211 = -weak_mat_tmp154;
    const s_t weak_mat_tmp212 = weak_mat_tmp159 + weak_mat_tmp183*weak_mat_tmp66 + weak_mat_tmp190*weak_mat_tmp67;
    const s_t weak_mat_tmp213 = -s_t(2)*weak_mat_tmp169;
    const s_t weak_mat_tmp214 = weak_mat_tmp165 + weak_mat_tmp188*weak_mat_tmp73 + weak_mat_tmp193*weak_mat_tmp76 + weak_mat_tmp39*(s_t(4)*weak_mat_tmp120 + weak_mat_tmp213);
    const s_t weak_mat_tmp215 = gu3*weak_mat_tmp122;
    const s_t weak_mat_tmp216 = -weak_mat_tmp215;
    const s_t weak_mat_tmp217 = gu3*weak_mat_tmp128;
    const s_t weak_mat_tmp218 = weak_mat_tmp171*weak_mat_tmp190 + weak_mat_tmp183*weak_mat_tmp91 + weak_mat_tmp217;
    const s_t weak_mat_tmp219 = -s_t(2)*weak_mat_tmp64;
    const s_t weak_mat_tmp220 = gu3*weak_mat_tmp136;
    const s_t weak_mat_tmp221 = weak_mat_tmp176*weak_mat_tmp188 + weak_mat_tmp177*weak_mat_tmp193 + weak_mat_tmp220 + weak_mat_tmp39*(weak_mat_tmp219 + s_t(4)*weak_mat_tmp62);
    const s_t weak_mat_tmp222 = s_t(2)*weak_mat_tmp105;
    const s_t weak_mat_tmp223 = -s_t(2)*weak_mat_tmp10;
    const s_t weak_mat_tmp224 = weak_mat_tmp7*(weak_mat_tmp222 + weak_mat_tmp223);
    const s_t weak_mat_tmp225 = weak_mat_tmp108*weak_mat_tmp14;
    const s_t weak_mat_tmp226 = weak_mat_tmp108*weak_mat_tmp31;
    const s_t weak_mat_tmp227 = weak_mat_tmp114*weak_mat_tmp41;
    const s_t weak_mat_tmp228 = weak_mat_tmp113*weak_mat_tmp60;
    const s_t weak_mat_tmp229 = weak_mat_tmp14*weak_mat_tmp51;
    const s_t weak_mat_tmp230 = weak_mat_tmp126*weak_mat_tmp229 + weak_mat_tmp225*weak_mat_tmp74;
    const s_t weak_mat_tmp231 = weak_mat_tmp113*weak_mat_tmp41;
    const s_t weak_mat_tmp232 = weak_mat_tmp134*weak_mat_tmp227 + weak_mat_tmp135*weak_mat_tmp231 + weak_mat_tmp39*(-weak_mat_tmp200 - weak_mat_tmp72);
    const s_t weak_mat_tmp233 = weak_mat_tmp171*weak_mat_tmp229 + weak_mat_tmp225*weak_mat_tmp91;
    const s_t weak_mat_tmp234 = weak_mat_tmp176*weak_mat_tmp227 + weak_mat_tmp177*weak_mat_tmp231 + weak_mat_tmp39*(-weak_mat_tmp192 - weak_mat_tmp87);
    const s_t weak_mat_tmp235 = -weak_mat_tmp129 + weak_mat_tmp225*weak_mat_tmp66 + weak_mat_tmp229*weak_mat_tmp67;
    const s_t weak_mat_tmp236 = -s_t(2)*weak_mat_tmp139;
    const s_t weak_mat_tmp237 = -weak_mat_tmp137 + weak_mat_tmp227*weak_mat_tmp73 + weak_mat_tmp231*weak_mat_tmp76 + weak_mat_tmp39*(s_t(4)*weak_mat_tmp132 + weak_mat_tmp236);
    const s_t weak_mat_tmp238 = weak_mat_tmp144*weak_mat_tmp229 - weak_mat_tmp217 + weak_mat_tmp225*weak_mat_tmp90;
    const s_t weak_mat_tmp239 = s_t(2)*weak_mat_tmp62;
    const s_t weak_mat_tmp240 = weak_mat_tmp149*weak_mat_tmp227 + weak_mat_tmp150*weak_mat_tmp231 - weak_mat_tmp220 + weak_mat_tmp39*(s_t(4)*gu2*gu7 - weak_mat_tmp239);
    const s_t weak_mat_tmp241 = -weak_mat_tmp168;
    const s_t weak_mat_tmp242 = weak_mat_tmp172 + weak_mat_tmp225*weak_mat_tmp83 + weak_mat_tmp229*weak_mat_tmp84;
    const s_t weak_mat_tmp243 = -s_t(2)*weak_mat_tmp155;
    const s_t weak_mat_tmp244 = weak_mat_tmp178 + weak_mat_tmp227*weak_mat_tmp89 + weak_mat_tmp231*weak_mat_tmp92 + weak_mat_tmp39*(s_t(4)*weak_mat_tmp121 + weak_mat_tmp243);
    const s_t weak_mat_tmp245 = -weak_mat_tmp202;
    const s_t weak_mat_tmp246 = weak_mat_tmp157*weak_mat_tmp229 + weak_mat_tmp203 + weak_mat_tmp225*weak_mat_tmp75;
    const s_t weak_mat_tmp247 = s_t(2)*weak_mat_tmp79;
    const s_t weak_mat_tmp248 = weak_mat_tmp163*weak_mat_tmp227 + weak_mat_tmp164*weak_mat_tmp231 + weak_mat_tmp206 + weak_mat_tmp39*(s_t(4)*gu2*weak_mat_tmp1 - weak_mat_tmp247);
    const s_t weak_mat_tmp249 = weak_mat_tmp7*(-weak_mat_tmp219 - weak_mat_tmp239);
    const s_t weak_mat_tmp250 = weak_mat_tmp14*weak_mat_tmp67;
    const s_t weak_mat_tmp251 = weak_mat_tmp31*weak_mat_tmp67;
    const s_t weak_mat_tmp252 = s_t(2)*weak_mat_tmp26;
    const s_t weak_mat_tmp253 = weak_mat_tmp252 + weak_mat_tmp34;
    const s_t weak_mat_tmp254 = s_t(2)*weak_mat_tmp22;
    const s_t weak_mat_tmp255 = weak_mat_tmp254 + weak_mat_tmp36;
    const s_t weak_mat_tmp256 = weak_mat_tmp41*weak_mat_tmp73;
    const s_t weak_mat_tmp257 = weak_mat_tmp60*weak_mat_tmp73;
    const s_t weak_mat_tmp258 = weak_mat_tmp14*weak_mat_tmp66;
    const s_t weak_mat_tmp259 = weak_mat_tmp250*weak_mat_tmp83 + weak_mat_tmp258*weak_mat_tmp84;
    const s_t weak_mat_tmp260 = weak_mat_tmp2*weak_mat_tmp74;
    const s_t weak_mat_tmp261 = gu7*weak_mat_tmp75;
    const s_t weak_mat_tmp262 = weak_mat_tmp41*weak_mat_tmp76;
    const s_t weak_mat_tmp263 = weak_mat_tmp256*weak_mat_tmp92 + weak_mat_tmp262*weak_mat_tmp89 + weak_mat_tmp39*(-weak_mat_tmp260 - weak_mat_tmp261);
    const s_t weak_mat_tmp264 = weak_mat_tmp126*weak_mat_tmp258 + weak_mat_tmp250*weak_mat_tmp74;
    const s_t weak_mat_tmp265 = s_t(2)*weak_mat_tmp48;
    const s_t weak_mat_tmp266 = weak_mat_tmp134*weak_mat_tmp262 + weak_mat_tmp135*weak_mat_tmp256 + weak_mat_tmp39*(-weak_mat_tmp112 - weak_mat_tmp265);
    const s_t weak_mat_tmp267 = weak_mat_tmp157*weak_mat_tmp258 + weak_mat_tmp250*weak_mat_tmp75;
    const s_t weak_mat_tmp268 = s_t(2)*weak_mat_tmp43;
    const s_t weak_mat_tmp269 = weak_mat_tmp163*weak_mat_tmp262 + weak_mat_tmp164*weak_mat_tmp256 + weak_mat_tmp39*(-weak_mat_tmp100 - weak_mat_tmp268);
    const s_t weak_mat_tmp270 = gu2*weak_mat_tmp122;
    const s_t weak_mat_tmp271 = gu2*weak_mat_tmp128;
    const s_t weak_mat_tmp272 = weak_mat_tmp144*weak_mat_tmp258 + weak_mat_tmp250*weak_mat_tmp90 - weak_mat_tmp271;
    const s_t weak_mat_tmp273 = gu2*weak_mat_tmp136;
    const s_t weak_mat_tmp274 = weak_mat_tmp149*weak_mat_tmp262 + weak_mat_tmp150*weak_mat_tmp256 - weak_mat_tmp273 + weak_mat_tmp39*(s_t(4)*weak_mat_tmp105 + weak_mat_tmp223);
    const s_t weak_mat_tmp275 = gu1*weak_mat_tmp122;
    const s_t weak_mat_tmp276 = -weak_mat_tmp275;
    const s_t weak_mat_tmp277 = gu1*weak_mat_tmp128;
    const s_t weak_mat_tmp278 = weak_mat_tmp171*weak_mat_tmp258 + weak_mat_tmp250*weak_mat_tmp91 + weak_mat_tmp277;
    const s_t weak_mat_tmp279 = gu1*weak_mat_tmp136;
    const s_t weak_mat_tmp280 = weak_mat_tmp176*weak_mat_tmp262 + weak_mat_tmp177*weak_mat_tmp256 + weak_mat_tmp279 + weak_mat_tmp39*(weak_mat_tmp180 + s_t(4)*weak_mat_tmp9);
    const s_t weak_mat_tmp281 = weak_mat_tmp7*(-weak_mat_tmp175 - weak_mat_tmp243);
    const s_t weak_mat_tmp282 = weak_mat_tmp14*weak_mat_tmp157;
    const s_t weak_mat_tmp283 = weak_mat_tmp157*weak_mat_tmp31;
    const s_t weak_mat_tmp284 = s_t(2)*weak_mat_tmp20;
    const s_t weak_mat_tmp285 = weak_mat_tmp186 + weak_mat_tmp284;
    const s_t weak_mat_tmp286 = weak_mat_tmp164*weak_mat_tmp41;
    const s_t weak_mat_tmp287 = weak_mat_tmp163*weak_mat_tmp60;
    const s_t weak_mat_tmp288 = weak_mat_tmp14*weak_mat_tmp75;
    const s_t weak_mat_tmp289 = weak_mat_tmp144*weak_mat_tmp288 + weak_mat_tmp282*weak_mat_tmp90;
    const s_t weak_mat_tmp290 = gu6*weak_mat_tmp66;
    const s_t weak_mat_tmp291 = weak_mat_tmp163*weak_mat_tmp41;
    const s_t weak_mat_tmp292 = weak_mat_tmp149*weak_mat_tmp286 + weak_mat_tmp150*weak_mat_tmp291 + weak_mat_tmp39*(-weak_mat_tmp260 - weak_mat_tmp290);
    const s_t weak_mat_tmp293 = weak_mat_tmp126*weak_mat_tmp288 + weak_mat_tmp282*weak_mat_tmp74;
    const s_t weak_mat_tmp294 = s_t(2)*weak_mat_tmp55;
    const s_t weak_mat_tmp295 = weak_mat_tmp134*weak_mat_tmp286 + weak_mat_tmp135*weak_mat_tmp291 + weak_mat_tmp39*(-weak_mat_tmp197 - weak_mat_tmp294);
    const s_t weak_mat_tmp296 = -weak_mat_tmp270;
    const s_t weak_mat_tmp297 = weak_mat_tmp271 + weak_mat_tmp282*weak_mat_tmp83 + weak_mat_tmp288*weak_mat_tmp84;
    const s_t weak_mat_tmp298 = weak_mat_tmp273 + weak_mat_tmp286*weak_mat_tmp89 + weak_mat_tmp291*weak_mat_tmp92 + weak_mat_tmp39*(s_t(4)*gu6*weak_mat_tmp1 - weak_mat_tmp222);
    const s_t weak_mat_tmp299 = weak_mat_tmp11*weak_mat_tmp122;
    const s_t weak_mat_tmp300 = weak_mat_tmp11*weak_mat_tmp128;
    const s_t weak_mat_tmp301 = weak_mat_tmp171*weak_mat_tmp288 + weak_mat_tmp282*weak_mat_tmp91 - weak_mat_tmp300;
    const s_t weak_mat_tmp302 = weak_mat_tmp11*weak_mat_tmp136;
    const s_t weak_mat_tmp303 = weak_mat_tmp176*weak_mat_tmp286 + weak_mat_tmp177*weak_mat_tmp291 - weak_mat_tmp302 + weak_mat_tmp39*(s_t(4)*weak_mat_tmp1*weak_mat_tmp2 - weak_mat_tmp4);
    const s_t weak_mat_tmp304 = weak_mat_tmp7*(weak_mat_tmp148 + weak_mat_tmp209);
    const s_t weak_mat_tmp305 = weak_mat_tmp126*weak_mat_tmp14;
    const s_t weak_mat_tmp306 = weak_mat_tmp126*weak_mat_tmp31;
    const s_t weak_mat_tmp307 = weak_mat_tmp135*weak_mat_tmp41;
    const s_t weak_mat_tmp308 = weak_mat_tmp134*weak_mat_tmp60;
    const s_t weak_mat_tmp309 = weak_mat_tmp14*weak_mat_tmp74;
    const s_t weak_mat_tmp310 = weak_mat_tmp171*weak_mat_tmp309 + weak_mat_tmp305*weak_mat_tmp91;
    const s_t weak_mat_tmp311 = weak_mat_tmp134*weak_mat_tmp41;
    const s_t weak_mat_tmp312 = weak_mat_tmp176*weak_mat_tmp307 + weak_mat_tmp177*weak_mat_tmp311 + weak_mat_tmp39*(-weak_mat_tmp261 - weak_mat_tmp290);
    const s_t weak_mat_tmp313 = -weak_mat_tmp277 + weak_mat_tmp305*weak_mat_tmp83 + weak_mat_tmp309*weak_mat_tmp84;
    const s_t weak_mat_tmp314 = -weak_mat_tmp279 + weak_mat_tmp307*weak_mat_tmp89 + weak_mat_tmp311*weak_mat_tmp92 + weak_mat_tmp39*(s_t(4)*gu5*gu6 - weak_mat_tmp181);
    const s_t weak_mat_tmp315 = -weak_mat_tmp299;
    const s_t weak_mat_tmp316 = weak_mat_tmp144*weak_mat_tmp309 + weak_mat_tmp300 + weak_mat_tmp305*weak_mat_tmp90;
    const s_t weak_mat_tmp317 = weak_mat_tmp149*weak_mat_tmp307 + weak_mat_tmp150*weak_mat_tmp311 + weak_mat_tmp302 + weak_mat_tmp39*(s_t(4)*weak_mat_tmp0 + weak_mat_tmp6);
    const s_t weak_mat_tmp318 = weak_mat_tmp7*(weak_mat_tmp205 + weak_mat_tmp247);
    const s_t weak_mat_tmp319 = weak_mat_tmp14*weak_mat_tmp84;
    const s_t weak_mat_tmp320 = weak_mat_tmp31*weak_mat_tmp84;
    const s_t weak_mat_tmp321 = weak_mat_tmp252 + weak_mat_tmp33;
    const s_t weak_mat_tmp322 = weak_mat_tmp254 + weak_mat_tmp37;
    const s_t weak_mat_tmp323 = weak_mat_tmp41*weak_mat_tmp89;
    const s_t weak_mat_tmp324 = weak_mat_tmp60*weak_mat_tmp89;
    const s_t weak_mat_tmp325 = weak_mat_tmp14*weak_mat_tmp83;
    const s_t weak_mat_tmp326 = weak_mat_tmp144*weak_mat_tmp325 + weak_mat_tmp319*weak_mat_tmp90;
    const s_t weak_mat_tmp327 = weak_mat_tmp41*weak_mat_tmp92;
    const s_t weak_mat_tmp328 = weak_mat_tmp149*weak_mat_tmp327 + weak_mat_tmp150*weak_mat_tmp323 + weak_mat_tmp39*(-weak_mat_tmp101 - weak_mat_tmp268);
    const s_t weak_mat_tmp329 = weak_mat_tmp171*weak_mat_tmp325 + weak_mat_tmp319*weak_mat_tmp91;
    const s_t weak_mat_tmp330 = weak_mat_tmp176*weak_mat_tmp327 + weak_mat_tmp177*weak_mat_tmp323 + weak_mat_tmp39*(-weak_mat_tmp111 - weak_mat_tmp265);
    const s_t weak_mat_tmp331 = weak_mat_tmp7*(weak_mat_tmp133 + weak_mat_tmp236);
    const s_t weak_mat_tmp332 = weak_mat_tmp14*weak_mat_tmp144;
    const s_t weak_mat_tmp333 = weak_mat_tmp144*weak_mat_tmp31;
    const s_t weak_mat_tmp334 = weak_mat_tmp185 + weak_mat_tmp284;
    const s_t weak_mat_tmp335 = weak_mat_tmp149*weak_mat_tmp41;
    const s_t weak_mat_tmp336 = weak_mat_tmp149*weak_mat_tmp60;
    const s_t weak_mat_tmp337 = weak_mat_tmp14*weak_mat_tmp171*weak_mat_tmp90 + weak_mat_tmp332*weak_mat_tmp91;
    const s_t weak_mat_tmp338 = weak_mat_tmp176*weak_mat_tmp41;
    const s_t weak_mat_tmp339 = weak_mat_tmp150*weak_mat_tmp338 + weak_mat_tmp177*weak_mat_tmp335 + weak_mat_tmp39*(-weak_mat_tmp196 - weak_mat_tmp294);
    const s_t weak_mat_tmp340 = weak_mat_tmp7*(-weak_mat_tmp162 - weak_mat_tmp213);
    const s_t weak_mat_tmp341 = weak_mat_tmp171*weak_mat_tmp31;
    const s_t weak_mat_tmp342 = weak_mat_tmp176*weak_mat_tmp60;
    const s_t material0 = trial_grad0*(iso_history_qp[0] + iso_relaxation_gamma*(C01*(weak_mat_tmp39*(weak_mat_tmp35 + weak_mat_tmp38) + s_t(2)*weak_mat_tmp40*weak_mat_tmp53 + weak_mat_tmp54*weak_mat_tmp61) + C10*(s_t(4)*weak_mat_tmp11*weak_mat_tmp16 + weak_mat_tmp13 + weak_mat_tmp17*weak_mat_tmp32)) + weak_mat_tmp3*weak_mat_tmp8) + trial_grad1*(iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_mat_tmp104 + weak_mat_tmp61*weak_mat_tmp99) + C10*(weak_mat_tmp32*weak_mat_tmp96 + weak_mat_tmp98)) + weak_mat_tmp8*weak_mat_tmp94) + trial_grad2*(iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_mat_tmp110*weak_mat_tmp61 + weak_mat_tmp115) + C10*(weak_mat_tmp107*weak_mat_tmp32 + weak_mat_tmp109)) + weak_mat_tmp106*weak_mat_tmp8) + trial_grad3*(iso_relaxation_gamma*(C01*(weak_mat_tmp61*weak_mat_tmp70 + weak_mat_tmp78) + C10*(weak_mat_tmp32*weak_mat_tmp65 + weak_mat_tmp69)) + weak_mat_tmp63*weak_mat_tmp8) + trial_grad4*(iso_relaxation_gamma*(C01*(weak_mat_tmp161*weak_mat_tmp61 + weak_mat_tmp166) + C10*(weak_mat_tmp156*weak_mat_tmp32 + weak_mat_tmp160)) + weak_mat_tmp153*weak_mat_tmp8 + weak_mat_tmp154) + trial_grad5*(iso_relaxation_gamma*(C01*(weak_mat_tmp131*weak_mat_tmp61 + weak_mat_tmp138) + C10*(weak_mat_tmp125*weak_mat_tmp32 + weak_mat_tmp130)) + weak_mat_tmp118*weak_mat_tmp8 + weak_mat_tmp124) + trial_grad6*(iso_relaxation_gamma*(C01*(weak_mat_tmp61*weak_mat_tmp86 + weak_mat_tmp93) + C10*(weak_mat_tmp32*weak_mat_tmp82 + weak_mat_tmp85)) + weak_mat_tmp8*weak_mat_tmp81) + trial_grad7*(iso_relaxation_gamma*(C01*(weak_mat_tmp147*weak_mat_tmp61 + weak_mat_tmp152) + C10*(weak_mat_tmp143*weak_mat_tmp32 + weak_mat_tmp146)) + weak_mat_tmp140*weak_mat_tmp8 + weak_mat_tmp142) + trial_grad8*(iso_relaxation_gamma*(C01*(weak_mat_tmp174*weak_mat_tmp61 + weak_mat_tmp179) + C10*(weak_mat_tmp170*weak_mat_tmp32 + weak_mat_tmp173)) + weak_mat_tmp167*weak_mat_tmp8 + weak_mat_tmp168);
    const s_t material1 = trial_grad0*(iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_mat_tmp104 + weak_mat_tmp189*weak_mat_tmp54) + C10*(weak_mat_tmp17*weak_mat_tmp184 + weak_mat_tmp98)) + weak_mat_tmp182*weak_mat_tmp3) + trial_grad1*(iso_history_qp[1] + iso_relaxation_gamma*(C01*(s_t(2)*weak_mat_tmp102*weak_mat_tmp188 + weak_mat_tmp189*weak_mat_tmp99 + weak_mat_tmp39*(weak_mat_tmp187 + weak_mat_tmp35)) + C10*(s_t(4)*gu1*weak_mat_tmp183 + weak_mat_tmp13 + weak_mat_tmp184*weak_mat_tmp96)) + weak_mat_tmp182*weak_mat_tmp94) + trial_grad2*(iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_mat_tmp110*weak_mat_tmp189 + weak_mat_tmp198) + C10*(weak_mat_tmp107*weak_mat_tmp184 + weak_mat_tmp195)) + weak_mat_tmp106*weak_mat_tmp182) + trial_grad3*(iso_relaxation_gamma*(C01*(weak_mat_tmp189*weak_mat_tmp70 + weak_mat_tmp214) + C10*(weak_mat_tmp184*weak_mat_tmp65 + weak_mat_tmp212)) + weak_mat_tmp182*weak_mat_tmp63 + weak_mat_tmp211) + trial_grad4*(iso_relaxation_gamma*(C01*(weak_mat_tmp161*weak_mat_tmp189 + weak_mat_tmp201) + C10*(weak_mat_tmp156*weak_mat_tmp184 + weak_mat_tmp199)) + weak_mat_tmp153*weak_mat_tmp182) + trial_grad5*(iso_relaxation_gamma*(C01*(weak_mat_tmp131*weak_mat_tmp189 + weak_mat_tmp207) + C10*(weak_mat_tmp125*weak_mat_tmp184 + weak_mat_tmp204)) + weak_mat_tmp118*weak_mat_tmp182 + weak_mat_tmp202) + trial_grad6*(iso_relaxation_gamma*(C01*(weak_mat_tmp189*weak_mat_tmp86 + weak_mat_tmp210) + C10*(weak_mat_tmp184*weak_mat_tmp82 + weak_mat_tmp208)) + weak_mat_tmp141 + weak_mat_tmp182*weak_mat_tmp81) + trial_grad7*(iso_relaxation_gamma*(C01*(weak_mat_tmp147*weak_mat_tmp189 + weak_mat_tmp194) + C10*(weak_mat_tmp143*weak_mat_tmp184 + weak_mat_tmp191)) + weak_mat_tmp140*weak_mat_tmp182) + trial_grad8*(iso_relaxation_gamma*(C01*(weak_mat_tmp174*weak_mat_tmp189 + weak_mat_tmp221) + C10*(weak_mat_tmp170*weak_mat_tmp184 + weak_mat_tmp218)) + weak_mat_tmp167*weak_mat_tmp182 + weak_mat_tmp216);
    const s_t material2 = trial_grad0*(iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_mat_tmp115 + weak_mat_tmp228*weak_mat_tmp54) + C10*(weak_mat_tmp109 + weak_mat_tmp17*weak_mat_tmp226)) + weak_mat_tmp224*weak_mat_tmp3) + trial_grad1*(iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_mat_tmp198 + weak_mat_tmp228*weak_mat_tmp99) + C10*(weak_mat_tmp195 + weak_mat_tmp226*weak_mat_tmp96)) + weak_mat_tmp224*weak_mat_tmp94) + trial_grad2*(iso_history_qp[2] + iso_relaxation_gamma*(C01*(weak_mat_tmp110*weak_mat_tmp228 + s_t(2)*weak_mat_tmp113*weak_mat_tmp227 + weak_mat_tmp39*(weak_mat_tmp187 + weak_mat_tmp38)) + C10*(s_t(4)*gu2*weak_mat_tmp225 + weak_mat_tmp107*weak_mat_tmp226 + weak_mat_tmp13)) + weak_mat_tmp106*weak_mat_tmp224) + trial_grad3*(iso_relaxation_gamma*(C01*(weak_mat_tmp228*weak_mat_tmp70 + weak_mat_tmp237) + C10*(weak_mat_tmp226*weak_mat_tmp65 + weak_mat_tmp235)) + weak_mat_tmp123 + weak_mat_tmp224*weak_mat_tmp63) + trial_grad4*(iso_relaxation_gamma*(C01*(weak_mat_tmp161*weak_mat_tmp228 + weak_mat_tmp248) + C10*(weak_mat_tmp156*weak_mat_tmp226 + weak_mat_tmp246)) + weak_mat_tmp153*weak_mat_tmp224 + weak_mat_tmp245) + trial_grad5*(iso_relaxation_gamma*(C01*(weak_mat_tmp131*weak_mat_tmp228 + weak_mat_tmp232) + C10*(weak_mat_tmp125*weak_mat_tmp226 + weak_mat_tmp230)) + weak_mat_tmp118*weak_mat_tmp224) + trial_grad6*(iso_relaxation_gamma*(C01*(weak_mat_tmp228*weak_mat_tmp86 + weak_mat_tmp244) + C10*(weak_mat_tmp226*weak_mat_tmp82 + weak_mat_tmp242)) + weak_mat_tmp224*weak_mat_tmp81 + weak_mat_tmp241) + trial_grad7*(iso_relaxation_gamma*(C01*(weak_mat_tmp147*weak_mat_tmp228 + weak_mat_tmp240) + C10*(weak_mat_tmp143*weak_mat_tmp226 + weak_mat_tmp238)) + weak_mat_tmp140*weak_mat_tmp224 + weak_mat_tmp215) + trial_grad8*(iso_relaxation_gamma*(C01*(weak_mat_tmp174*weak_mat_tmp228 + weak_mat_tmp234) + C10*(weak_mat_tmp170*weak_mat_tmp226 + weak_mat_tmp233)) + weak_mat_tmp167*weak_mat_tmp224);
    const s_t material3 = trial_grad0*(iso_relaxation_gamma*(C01*(weak_mat_tmp257*weak_mat_tmp54 + weak_mat_tmp78) + C10*(weak_mat_tmp17*weak_mat_tmp251 + weak_mat_tmp69)) + weak_mat_tmp249*weak_mat_tmp3) + trial_grad1*(iso_relaxation_gamma*(C01*(weak_mat_tmp214 + weak_mat_tmp257*weak_mat_tmp99) + C10*(weak_mat_tmp212 + weak_mat_tmp251*weak_mat_tmp96)) + weak_mat_tmp211 + weak_mat_tmp249*weak_mat_tmp94) + trial_grad2*(iso_relaxation_gamma*(C01*(weak_mat_tmp110*weak_mat_tmp257 + weak_mat_tmp237) + C10*(weak_mat_tmp107*weak_mat_tmp251 + weak_mat_tmp235)) + weak_mat_tmp106*weak_mat_tmp249 + weak_mat_tmp123) + trial_grad3*(iso_history_qp[0] + iso_relaxation_gamma*(C01*(s_t(2)*weak_mat_tmp256*weak_mat_tmp76 + weak_mat_tmp257*weak_mat_tmp70 + weak_mat_tmp39*(weak_mat_tmp253 + weak_mat_tmp255)) + C10*(s_t(4)*gu3*weak_mat_tmp250 + weak_mat_tmp13 + weak_mat_tmp251*weak_mat_tmp65)) + weak_mat_tmp249*weak_mat_tmp63) + trial_grad4*(iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_mat_tmp161*weak_mat_tmp257 + weak_mat_tmp269) + C10*(weak_mat_tmp156*weak_mat_tmp251 + weak_mat_tmp267)) + weak_mat_tmp153*weak_mat_tmp249) + trial_grad5*(iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_mat_tmp131*weak_mat_tmp257 + weak_mat_tmp266) + C10*(weak_mat_tmp125*weak_mat_tmp251 + weak_mat_tmp264)) + weak_mat_tmp118*weak_mat_tmp249) + trial_grad6*(iso_relaxation_gamma*(C01*(weak_mat_tmp257*weak_mat_tmp86 + weak_mat_tmp263) + C10*(weak_mat_tmp251*weak_mat_tmp82 + weak_mat_tmp259)) + weak_mat_tmp249*weak_mat_tmp81) + trial_grad7*(iso_relaxation_gamma*(C01*(weak_mat_tmp147*weak_mat_tmp257 + weak_mat_tmp274) + C10*(weak_mat_tmp143*weak_mat_tmp251 + weak_mat_tmp272)) + weak_mat_tmp140*weak_mat_tmp249 + weak_mat_tmp270) + trial_grad8*(iso_relaxation_gamma*(C01*(weak_mat_tmp174*weak_mat_tmp257 + weak_mat_tmp280) + C10*(weak_mat_tmp170*weak_mat_tmp251 + weak_mat_tmp278)) + weak_mat_tmp167*weak_mat_tmp249 + weak_mat_tmp276);
    const s_t material4 = trial_grad0*(iso_relaxation_gamma*(C01*(weak_mat_tmp166 + weak_mat_tmp287*weak_mat_tmp54) + C10*(weak_mat_tmp160 + weak_mat_tmp17*weak_mat_tmp283)) + weak_mat_tmp154 + weak_mat_tmp281*weak_mat_tmp3) + trial_grad1*(iso_relaxation_gamma*(C01*(weak_mat_tmp201 + weak_mat_tmp287*weak_mat_tmp99) + C10*(weak_mat_tmp199 + weak_mat_tmp283*weak_mat_tmp96)) + weak_mat_tmp281*weak_mat_tmp94) + trial_grad2*(iso_relaxation_gamma*(C01*(weak_mat_tmp110*weak_mat_tmp287 + weak_mat_tmp248) + C10*(weak_mat_tmp107*weak_mat_tmp283 + weak_mat_tmp246)) + weak_mat_tmp106*weak_mat_tmp281 + weak_mat_tmp245) + trial_grad3*(iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_mat_tmp269 + weak_mat_tmp287*weak_mat_tmp70) + C10*(weak_mat_tmp267 + weak_mat_tmp283*weak_mat_tmp65)) + weak_mat_tmp281*weak_mat_tmp63) + trial_grad4*(iso_history_qp[1] + iso_relaxation_gamma*(C01*(weak_mat_tmp161*weak_mat_tmp287 + s_t(2)*weak_mat_tmp163*weak_mat_tmp286 + weak_mat_tmp39*(weak_mat_tmp253 + weak_mat_tmp285)) + C10*(s_t(4)*weak_mat_tmp1*weak_mat_tmp282 + weak_mat_tmp13 + weak_mat_tmp156*weak_mat_tmp283)) + weak_mat_tmp153*weak_mat_tmp281) + trial_grad5*(iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_mat_tmp131*weak_mat_tmp287 + weak_mat_tmp295) + C10*(weak_mat_tmp125*weak_mat_tmp283 + weak_mat_tmp293)) + weak_mat_tmp118*weak_mat_tmp281) + trial_grad6*(iso_relaxation_gamma*(C01*(weak_mat_tmp287*weak_mat_tmp86 + weak_mat_tmp298) + C10*(weak_mat_tmp283*weak_mat_tmp82 + weak_mat_tmp297)) + weak_mat_tmp281*weak_mat_tmp81 + weak_mat_tmp296) + trial_grad7*(iso_relaxation_gamma*(C01*(weak_mat_tmp147*weak_mat_tmp287 + weak_mat_tmp292) + C10*(weak_mat_tmp143*weak_mat_tmp283 + weak_mat_tmp289)) + weak_mat_tmp140*weak_mat_tmp281) + trial_grad8*(iso_relaxation_gamma*(C01*(weak_mat_tmp174*weak_mat_tmp287 + weak_mat_tmp303) + C10*(weak_mat_tmp170*weak_mat_tmp283 + weak_mat_tmp301)) + weak_mat_tmp167*weak_mat_tmp281 + weak_mat_tmp299);
    const s_t material5 = trial_grad0*(iso_relaxation_gamma*(C01*(weak_mat_tmp138 + weak_mat_tmp308*weak_mat_tmp54) + C10*(weak_mat_tmp130 + weak_mat_tmp17*weak_mat_tmp306)) + weak_mat_tmp124 + weak_mat_tmp3*weak_mat_tmp304) + trial_grad1*(iso_relaxation_gamma*(C01*(weak_mat_tmp207 + weak_mat_tmp308*weak_mat_tmp99) + C10*(weak_mat_tmp204 + weak_mat_tmp306*weak_mat_tmp96)) + weak_mat_tmp202 + weak_mat_tmp304*weak_mat_tmp94) + trial_grad2*(iso_relaxation_gamma*(C01*(weak_mat_tmp110*weak_mat_tmp308 + weak_mat_tmp232) + C10*(weak_mat_tmp107*weak_mat_tmp306 + weak_mat_tmp230)) + weak_mat_tmp106*weak_mat_tmp304) + trial_grad3*(iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_mat_tmp266 + weak_mat_tmp308*weak_mat_tmp70) + C10*(weak_mat_tmp264 + weak_mat_tmp306*weak_mat_tmp65)) + weak_mat_tmp304*weak_mat_tmp63) + trial_grad4*(iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_mat_tmp161*weak_mat_tmp308 + weak_mat_tmp295) + C10*(weak_mat_tmp156*weak_mat_tmp306 + weak_mat_tmp293)) + weak_mat_tmp153*weak_mat_tmp304) + trial_grad5*(iso_history_qp[2] + iso_relaxation_gamma*(C01*(weak_mat_tmp131*weak_mat_tmp308 + s_t(2)*weak_mat_tmp134*weak_mat_tmp307 + weak_mat_tmp39*(weak_mat_tmp255 + weak_mat_tmp285)) + C10*(s_t(4)*gu5*weak_mat_tmp305 + weak_mat_tmp125*weak_mat_tmp306 + weak_mat_tmp13)) + weak_mat_tmp118*weak_mat_tmp304) + trial_grad6*(iso_relaxation_gamma*(C01*(weak_mat_tmp308*weak_mat_tmp86 + weak_mat_tmp314) + C10*(weak_mat_tmp306*weak_mat_tmp82 + weak_mat_tmp313)) + weak_mat_tmp275 + weak_mat_tmp304*weak_mat_tmp81) + trial_grad7*(iso_relaxation_gamma*(C01*(weak_mat_tmp147*weak_mat_tmp308 + weak_mat_tmp317) + C10*(weak_mat_tmp143*weak_mat_tmp306 + weak_mat_tmp316)) + weak_mat_tmp140*weak_mat_tmp304 + weak_mat_tmp315) + trial_grad8*(iso_relaxation_gamma*(C01*(weak_mat_tmp174*weak_mat_tmp308 + weak_mat_tmp312) + C10*(weak_mat_tmp170*weak_mat_tmp306 + weak_mat_tmp310)) + weak_mat_tmp167*weak_mat_tmp304);
    const s_t material6 = trial_grad0*(iso_relaxation_gamma*(C01*(weak_mat_tmp324*weak_mat_tmp54 + weak_mat_tmp93) + C10*(weak_mat_tmp17*weak_mat_tmp320 + weak_mat_tmp85)) + weak_mat_tmp3*weak_mat_tmp318) + trial_grad1*(iso_relaxation_gamma*(C01*(weak_mat_tmp210 + weak_mat_tmp324*weak_mat_tmp99) + C10*(weak_mat_tmp208 + weak_mat_tmp320*weak_mat_tmp96)) + weak_mat_tmp141 + weak_mat_tmp318*weak_mat_tmp94) + trial_grad2*(iso_relaxation_gamma*(C01*(weak_mat_tmp110*weak_mat_tmp324 + weak_mat_tmp244) + C10*(weak_mat_tmp107*weak_mat_tmp320 + weak_mat_tmp242)) + weak_mat_tmp106*weak_mat_tmp318 + weak_mat_tmp241) + trial_grad3*(iso_relaxation_gamma*(C01*(weak_mat_tmp263 + weak_mat_tmp324*weak_mat_tmp70) + C10*(weak_mat_tmp259 + weak_mat_tmp320*weak_mat_tmp65)) + weak_mat_tmp318*weak_mat_tmp63) + trial_grad4*(iso_relaxation_gamma*(C01*(weak_mat_tmp161*weak_mat_tmp324 + weak_mat_tmp298) + C10*(weak_mat_tmp156*weak_mat_tmp320 + weak_mat_tmp297)) + weak_mat_tmp153*weak_mat_tmp318 + weak_mat_tmp296) + trial_grad5*(iso_relaxation_gamma*(C01*(weak_mat_tmp131*weak_mat_tmp324 + weak_mat_tmp314) + C10*(weak_mat_tmp125*weak_mat_tmp320 + weak_mat_tmp313)) + weak_mat_tmp118*weak_mat_tmp318 + weak_mat_tmp275) + trial_grad6*(iso_history_qp[0] + iso_relaxation_gamma*(C01*(s_t(2)*weak_mat_tmp323*weak_mat_tmp92 + weak_mat_tmp324*weak_mat_tmp86 + weak_mat_tmp39*(weak_mat_tmp321 + weak_mat_tmp322)) + C10*(s_t(4)*gu6*weak_mat_tmp319 + weak_mat_tmp13 + weak_mat_tmp320*weak_mat_tmp82)) + weak_mat_tmp318*weak_mat_tmp81) + trial_grad7*(iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_mat_tmp147*weak_mat_tmp324 + weak_mat_tmp328) + C10*(weak_mat_tmp143*weak_mat_tmp320 + weak_mat_tmp326)) + weak_mat_tmp140*weak_mat_tmp318) + trial_grad8*(iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_mat_tmp174*weak_mat_tmp324 + weak_mat_tmp330) + C10*(weak_mat_tmp170*weak_mat_tmp320 + weak_mat_tmp329)) + weak_mat_tmp167*weak_mat_tmp318);
    const s_t material7 = trial_grad0*(iso_relaxation_gamma*(C01*(weak_mat_tmp152 + weak_mat_tmp336*weak_mat_tmp54) + C10*(weak_mat_tmp146 + weak_mat_tmp17*weak_mat_tmp333)) + weak_mat_tmp142 + weak_mat_tmp3*weak_mat_tmp331) + trial_grad1*(iso_relaxation_gamma*(C01*(weak_mat_tmp194 + weak_mat_tmp336*weak_mat_tmp99) + C10*(weak_mat_tmp191 + weak_mat_tmp333*weak_mat_tmp96)) + weak_mat_tmp331*weak_mat_tmp94) + trial_grad2*(iso_relaxation_gamma*(C01*(weak_mat_tmp110*weak_mat_tmp336 + weak_mat_tmp240) + C10*(weak_mat_tmp107*weak_mat_tmp333 + weak_mat_tmp238)) + weak_mat_tmp106*weak_mat_tmp331 + weak_mat_tmp215) + trial_grad3*(iso_relaxation_gamma*(C01*(weak_mat_tmp274 + weak_mat_tmp336*weak_mat_tmp70) + C10*(weak_mat_tmp272 + weak_mat_tmp333*weak_mat_tmp65)) + weak_mat_tmp270 + weak_mat_tmp331*weak_mat_tmp63) + trial_grad4*(iso_relaxation_gamma*(C01*(weak_mat_tmp161*weak_mat_tmp336 + weak_mat_tmp292) + C10*(weak_mat_tmp156*weak_mat_tmp333 + weak_mat_tmp289)) + weak_mat_tmp153*weak_mat_tmp331) + trial_grad5*(iso_relaxation_gamma*(C01*(weak_mat_tmp131*weak_mat_tmp336 + weak_mat_tmp317) + C10*(weak_mat_tmp125*weak_mat_tmp333 + weak_mat_tmp316)) + weak_mat_tmp118*weak_mat_tmp331 + weak_mat_tmp315) + trial_grad6*(iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_mat_tmp328 + weak_mat_tmp336*weak_mat_tmp86) + C10*(weak_mat_tmp326 + weak_mat_tmp333*weak_mat_tmp82)) + weak_mat_tmp331*weak_mat_tmp81) + trial_grad7*(iso_history_qp[1] + iso_relaxation_gamma*(C01*(weak_mat_tmp147*weak_mat_tmp336 + s_t(2)*weak_mat_tmp150*weak_mat_tmp335 + weak_mat_tmp39*(weak_mat_tmp321 + weak_mat_tmp334)) + C10*(s_t(4)*gu7*weak_mat_tmp332 + weak_mat_tmp13 + weak_mat_tmp143*weak_mat_tmp333)) + weak_mat_tmp140*weak_mat_tmp331) + trial_grad8*(iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_mat_tmp174*weak_mat_tmp336 + weak_mat_tmp339) + C10*(weak_mat_tmp170*weak_mat_tmp333 + weak_mat_tmp337)) + weak_mat_tmp167*weak_mat_tmp331);
    const s_t material8 = trial_grad0*(iso_relaxation_gamma*(C01*(weak_mat_tmp179 + weak_mat_tmp342*weak_mat_tmp54) + C10*(weak_mat_tmp17*weak_mat_tmp341 + weak_mat_tmp173)) + weak_mat_tmp168 + weak_mat_tmp3*weak_mat_tmp340) + trial_grad1*(iso_relaxation_gamma*(C01*(weak_mat_tmp221 + weak_mat_tmp342*weak_mat_tmp99) + C10*(weak_mat_tmp218 + weak_mat_tmp341*weak_mat_tmp96)) + weak_mat_tmp216 + weak_mat_tmp340*weak_mat_tmp94) + trial_grad2*(iso_relaxation_gamma*(C01*(weak_mat_tmp110*weak_mat_tmp342 + weak_mat_tmp234) + C10*(weak_mat_tmp107*weak_mat_tmp341 + weak_mat_tmp233)) + weak_mat_tmp106*weak_mat_tmp340) + trial_grad3*(iso_relaxation_gamma*(C01*(weak_mat_tmp280 + weak_mat_tmp342*weak_mat_tmp70) + C10*(weak_mat_tmp278 + weak_mat_tmp341*weak_mat_tmp65)) + weak_mat_tmp276 + weak_mat_tmp340*weak_mat_tmp63) + trial_grad4*(iso_relaxation_gamma*(C01*(weak_mat_tmp161*weak_mat_tmp342 + weak_mat_tmp303) + C10*(weak_mat_tmp156*weak_mat_tmp341 + weak_mat_tmp301)) + weak_mat_tmp153*weak_mat_tmp340 + weak_mat_tmp299) + trial_grad5*(iso_relaxation_gamma*(C01*(weak_mat_tmp131*weak_mat_tmp342 + weak_mat_tmp312) + C10*(weak_mat_tmp125*weak_mat_tmp341 + weak_mat_tmp310)) + weak_mat_tmp118*weak_mat_tmp340) + trial_grad6*(iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_mat_tmp330 + weak_mat_tmp342*weak_mat_tmp86) + C10*(weak_mat_tmp329 + weak_mat_tmp341*weak_mat_tmp82)) + weak_mat_tmp340*weak_mat_tmp81) + trial_grad7*(iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_mat_tmp147*weak_mat_tmp342 + weak_mat_tmp339) + C10*(weak_mat_tmp143*weak_mat_tmp341 + weak_mat_tmp337)) + weak_mat_tmp140*weak_mat_tmp340) + trial_grad8*(iso_history_qp[2] + iso_relaxation_gamma*(C01*(weak_mat_tmp174*weak_mat_tmp342 + s_t(2)*weak_mat_tmp177*weak_mat_tmp338 + weak_mat_tmp39*(weak_mat_tmp322 + weak_mat_tmp334)) + C10*(weak_mat_tmp13 + s_t(4)*weak_mat_tmp158*weak_mat_tmp171 + weak_mat_tmp170*weak_mat_tmp341)) + weak_mat_tmp167*weak_mat_tmp340);
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
static SFEM_INLINE void mooney_rivlin_prony_d3_simplex_quadrature_output_block(
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
    const s_t qp_out_tmp8 = gu1*gu5;
    const s_t qp_out_tmp9 = -qp_out_tmp3 + qp_out_tmp8;
    const s_t qp_out_tmp10 = pow(qp_out_tmp6, (s_t(-2) / s_t(3)));
    const s_t qp_out_tmp11 = s_t(2)*gu6;
    const s_t qp_out_tmp12 = pow_2(gu3) + pow_2(gu6) + pow_2(qp_out_tmp4);
    const s_t qp_out_tmp13 = pow_2(gu1) + pow_2(gu7) + pow_2(qp_out_tmp2);
    const s_t qp_out_tmp14 = pow_2(gu2) + pow_2(gu5) + pow_2(qp_out_tmp0);
    const s_t qp_out_tmp15 = qp_out_tmp12 + qp_out_tmp13 + qp_out_tmp14;
    const s_t qp_out_tmp16 = qp_out_tmp15/pow(qp_out_tmp6, (s_t(5) / s_t(3)));
    const s_t qp_out_tmp17 = pow(qp_out_tmp6, (s_t(-4) / s_t(3)));
    const s_t qp_out_tmp18 = gu1*qp_out_tmp4 + gu3*qp_out_tmp2 + gu6*gu7;
    const s_t qp_out_tmp19 = s_t(2)*qp_out_tmp18;
    const s_t qp_out_tmp20 = gu2*qp_out_tmp4 + gu3*gu5 + gu6*qp_out_tmp0;
    const s_t qp_out_tmp21 = s_t(2)*qp_out_tmp20;
    const s_t qp_out_tmp22 = gu1*gu2 + gu5*qp_out_tmp2 + gu7*qp_out_tmp0;
    const s_t qp_out_tmp23 = (-(s_t(1) / s_t(2))*pow_2(qp_out_tmp12) - (s_t(1) / s_t(2))*pow_2(qp_out_tmp13) - (s_t(1) / s_t(2))*pow_2(qp_out_tmp14) + ((s_t(1) / s_t(2)))*pow_2(qp_out_tmp15) - pow_2(qp_out_tmp18) - pow_2(qp_out_tmp20) - pow_2(qp_out_tmp22))/pow(qp_out_tmp6, (s_t(7) / s_t(3)));
    const s_t qp_out_tmp24 = gu2*gu7 - qp_out_tmp1;
    const s_t qp_out_tmp25 = s_t(2)*gu3;
    const s_t qp_out_tmp26 = gu2*gu7;
    const s_t qp_out_tmp27 = qp_out_tmp0*qp_out_tmp2 - qp_out_tmp5;
    const s_t qp_out_tmp28 = s_t(2)*qp_out_tmp4;
    const s_t qp_out_tmp29 = qp_out_tmp0*qp_out_tmp2;
    const s_t qp_out_tmp30 = gu2*gu3;
    const s_t qp_out_tmp31 = -gu5*qp_out_tmp4 + qp_out_tmp30;
    const s_t qp_out_tmp32 = s_t(2)*gu7;
    const s_t qp_out_tmp33 = s_t(2)*qp_out_tmp22;
    const s_t qp_out_tmp34 = C01*(qp_out_tmp17*(-gu6*qp_out_tmp19 + s_t(2)*gu7*qp_out_tmp15 - qp_out_tmp0*qp_out_tmp33 - qp_out_tmp13*qp_out_tmp32) + qp_out_tmp23*(((s_t(4) / s_t(3)))*gu5*qp_out_tmp4 - (s_t(4) / s_t(3))*qp_out_tmp30)) + C10*(qp_out_tmp10*qp_out_tmp32 + qp_out_tmp16*(((s_t(2) / s_t(3)))*gu5*qp_out_tmp4 - (s_t(2) / s_t(3))*qp_out_tmp30));
    const s_t qp_out_tmp35 = gu3*qp_out_tmp0;
    const s_t qp_out_tmp36 = gu5*gu6 - qp_out_tmp35;
    const s_t qp_out_tmp37 = s_t(2)*gu1;
    const s_t qp_out_tmp38 = gu5*gu6;
    const s_t qp_out_tmp39 = C01*(qp_out_tmp17*(s_t(2)*gu1*qp_out_tmp15 - gu2*qp_out_tmp33 - qp_out_tmp13*qp_out_tmp37 - qp_out_tmp19*qp_out_tmp4) + qp_out_tmp23*(((s_t(4) / s_t(3)))*qp_out_tmp35 - (s_t(4) / s_t(3))*qp_out_tmp38)) + C10*(qp_out_tmp10*qp_out_tmp37 + qp_out_tmp16*(((s_t(2) / s_t(3)))*qp_out_tmp35 - (s_t(2) / s_t(3))*qp_out_tmp38));
    const s_t qp_out_tmp40 = gu2*gu6;
    const s_t qp_out_tmp41 = qp_out_tmp0*qp_out_tmp4 - qp_out_tmp40;
    const s_t qp_out_tmp42 = s_t(2)*qp_out_tmp2;
    const s_t qp_out_tmp43 = qp_out_tmp0*qp_out_tmp4;
    const s_t qp_out_tmp44 = C01*(qp_out_tmp17*(-gu3*qp_out_tmp19 - gu5*qp_out_tmp33 - qp_out_tmp13*qp_out_tmp42 + s_t(2)*qp_out_tmp15*qp_out_tmp2) + qp_out_tmp23*(((s_t(4) / s_t(3)))*qp_out_tmp40 - (s_t(4) / s_t(3))*qp_out_tmp43)) + C10*(qp_out_tmp10*qp_out_tmp42 + qp_out_tmp16*(((s_t(2) / s_t(3)))*qp_out_tmp40 - (s_t(2) / s_t(3))*qp_out_tmp43));
    const s_t qp_out_tmp45 = gu1*gu6;
    const s_t qp_out_tmp46 = s_t(2)*gu5;
    const s_t qp_out_tmp47 = C01*(qp_out_tmp17*(-gu3*qp_out_tmp21 + s_t(2)*gu5*qp_out_tmp15 - qp_out_tmp14*qp_out_tmp46 - qp_out_tmp2*qp_out_tmp33) + qp_out_tmp23*(((s_t(4) / s_t(3)))*gu7*qp_out_tmp4 - (s_t(4) / s_t(3))*qp_out_tmp45)) + C10*(qp_out_tmp10*qp_out_tmp46 + qp_out_tmp16*(((s_t(2) / s_t(3)))*gu7*qp_out_tmp4 - (s_t(2) / s_t(3))*qp_out_tmp45));
    const s_t qp_out_tmp48 = gu3*gu7;
    const s_t qp_out_tmp49 = s_t(2)*gu2;
    const s_t qp_out_tmp50 = C01*(qp_out_tmp17*(s_t(2)*gu2*qp_out_tmp15 - qp_out_tmp14*qp_out_tmp49 - qp_out_tmp21*qp_out_tmp4 - qp_out_tmp22*qp_out_tmp37) + qp_out_tmp23*(((s_t(4) / s_t(3)))*gu6*qp_out_tmp2 - (s_t(4) / s_t(3))*qp_out_tmp48)) + C10*(qp_out_tmp10*qp_out_tmp49 + qp_out_tmp16*(((s_t(2) / s_t(3)))*gu6*qp_out_tmp2 - (s_t(2) / s_t(3))*qp_out_tmp48));
    const s_t qp_out_tmp51 = gu1*gu3;
    const s_t qp_out_tmp52 = s_t(2)*qp_out_tmp0;
    const s_t qp_out_tmp53 = qp_out_tmp2*qp_out_tmp4;
    const s_t qp_out_tmp54 = C01*(qp_out_tmp17*(-gu6*qp_out_tmp21 - gu7*qp_out_tmp33 + s_t(2)*qp_out_tmp0*qp_out_tmp15 - qp_out_tmp14*qp_out_tmp52) + qp_out_tmp23*(((s_t(4) / s_t(3)))*qp_out_tmp51 - (s_t(4) / s_t(3))*qp_out_tmp53)) + C10*(qp_out_tmp10*qp_out_tmp52 + qp_out_tmp16*(((s_t(2) / s_t(3)))*qp_out_tmp51 - (s_t(2) / s_t(3))*qp_out_tmp53));
    iso_stress_q[lane] = qp_out_tmp7*(qp_out_tmp24*(C01*(qp_out_tmp17*(s_t(2)*gu3*qp_out_tmp15 - gu5*qp_out_tmp21 - qp_out_tmp12*qp_out_tmp25 - qp_out_tmp19*qp_out_tmp2) + qp_out_tmp23*(((s_t(4) / s_t(3)))*qp_out_tmp1 - (s_t(4) / s_t(3))*qp_out_tmp26)) + C10*(qp_out_tmp10*qp_out_tmp25 + qp_out_tmp16*(((s_t(2) / s_t(3)))*qp_out_tmp1 - (s_t(2) / s_t(3))*qp_out_tmp26))) + qp_out_tmp27*(C01*(qp_out_tmp17*(-gu1*qp_out_tmp19 - gu2*qp_out_tmp21 - qp_out_tmp12*qp_out_tmp28 + s_t(2)*qp_out_tmp15*qp_out_tmp4) + qp_out_tmp23*(-(s_t(4) / s_t(3))*qp_out_tmp29 + ((s_t(4) / s_t(3)))*qp_out_tmp5)) + C10*(qp_out_tmp10*qp_out_tmp28 + qp_out_tmp16*(-(s_t(2) / s_t(3))*qp_out_tmp29 + ((s_t(2) / s_t(3)))*qp_out_tmp5))) + qp_out_tmp9*(C01*(qp_out_tmp17*(s_t(2)*gu6*qp_out_tmp15 - gu7*qp_out_tmp19 - qp_out_tmp0*qp_out_tmp21 - qp_out_tmp11*qp_out_tmp12) + qp_out_tmp23*(((s_t(4) / s_t(3)))*gu2*qp_out_tmp2 - (s_t(4) / s_t(3))*qp_out_tmp8)) + C10*(qp_out_tmp10*qp_out_tmp11 + qp_out_tmp16*(((s_t(2) / s_t(3)))*gu2*qp_out_tmp2 - (s_t(2) / s_t(3))*qp_out_tmp8))));
    iso_stress_q[1 * NQ * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp31*qp_out_tmp34 + qp_out_tmp36*qp_out_tmp39 + qp_out_tmp41*qp_out_tmp44);
    iso_stress_q[2 * NQ * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp47*(-gu7*qp_out_tmp4 + qp_out_tmp45) + qp_out_tmp50*(-gu6*qp_out_tmp2 + qp_out_tmp48) + qp_out_tmp54*(qp_out_tmp2*qp_out_tmp4 - qp_out_tmp51));
    iso_stress_q[3 * NQ * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp24*qp_out_tmp44 + qp_out_tmp27*qp_out_tmp39 + qp_out_tmp34*qp_out_tmp9);
    iso_stress_q[4 * NQ * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp24*qp_out_tmp47 + qp_out_tmp27*qp_out_tmp50 + qp_out_tmp54*qp_out_tmp9);
    iso_stress_q[5 * NQ * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp31*qp_out_tmp54 + qp_out_tmp36*qp_out_tmp50 + qp_out_tmp41*qp_out_tmp47);
      }
    }
}

template <typename s_t, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_prony_d3_simplex_tet4_quadrature_output_block(
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
    const s_t qp_out_tmp8 = gu1*gu5;
    const s_t qp_out_tmp9 = -qp_out_tmp3 + qp_out_tmp8;
    const s_t qp_out_tmp10 = pow(qp_out_tmp6, (s_t(-2) / s_t(3)));
    const s_t qp_out_tmp11 = s_t(2)*gu6;
    const s_t qp_out_tmp12 = pow_2(gu3) + pow_2(gu6) + pow_2(qp_out_tmp4);
    const s_t qp_out_tmp13 = pow_2(gu1) + pow_2(gu7) + pow_2(qp_out_tmp2);
    const s_t qp_out_tmp14 = pow_2(gu2) + pow_2(gu5) + pow_2(qp_out_tmp0);
    const s_t qp_out_tmp15 = qp_out_tmp12 + qp_out_tmp13 + qp_out_tmp14;
    const s_t qp_out_tmp16 = qp_out_tmp15/pow(qp_out_tmp6, (s_t(5) / s_t(3)));
    const s_t qp_out_tmp17 = pow(qp_out_tmp6, (s_t(-4) / s_t(3)));
    const s_t qp_out_tmp18 = gu1*qp_out_tmp4 + gu3*qp_out_tmp2 + gu6*gu7;
    const s_t qp_out_tmp19 = s_t(2)*qp_out_tmp18;
    const s_t qp_out_tmp20 = gu2*qp_out_tmp4 + gu3*gu5 + gu6*qp_out_tmp0;
    const s_t qp_out_tmp21 = s_t(2)*qp_out_tmp20;
    const s_t qp_out_tmp22 = gu1*gu2 + gu5*qp_out_tmp2 + gu7*qp_out_tmp0;
    const s_t qp_out_tmp23 = (-(s_t(1) / s_t(2))*pow_2(qp_out_tmp12) - (s_t(1) / s_t(2))*pow_2(qp_out_tmp13) - (s_t(1) / s_t(2))*pow_2(qp_out_tmp14) + ((s_t(1) / s_t(2)))*pow_2(qp_out_tmp15) - pow_2(qp_out_tmp18) - pow_2(qp_out_tmp20) - pow_2(qp_out_tmp22))/pow(qp_out_tmp6, (s_t(7) / s_t(3)));
    const s_t qp_out_tmp24 = gu2*gu7 - qp_out_tmp1;
    const s_t qp_out_tmp25 = s_t(2)*gu3;
    const s_t qp_out_tmp26 = gu2*gu7;
    const s_t qp_out_tmp27 = qp_out_tmp0*qp_out_tmp2 - qp_out_tmp5;
    const s_t qp_out_tmp28 = s_t(2)*qp_out_tmp4;
    const s_t qp_out_tmp29 = qp_out_tmp0*qp_out_tmp2;
    const s_t qp_out_tmp30 = gu2*gu3;
    const s_t qp_out_tmp31 = -gu5*qp_out_tmp4 + qp_out_tmp30;
    const s_t qp_out_tmp32 = s_t(2)*gu7;
    const s_t qp_out_tmp33 = s_t(2)*qp_out_tmp22;
    const s_t qp_out_tmp34 = C01*(qp_out_tmp17*(-gu6*qp_out_tmp19 + s_t(2)*gu7*qp_out_tmp15 - qp_out_tmp0*qp_out_tmp33 - qp_out_tmp13*qp_out_tmp32) + qp_out_tmp23*(((s_t(4) / s_t(3)))*gu5*qp_out_tmp4 - (s_t(4) / s_t(3))*qp_out_tmp30)) + C10*(qp_out_tmp10*qp_out_tmp32 + qp_out_tmp16*(((s_t(2) / s_t(3)))*gu5*qp_out_tmp4 - (s_t(2) / s_t(3))*qp_out_tmp30));
    const s_t qp_out_tmp35 = gu3*qp_out_tmp0;
    const s_t qp_out_tmp36 = gu5*gu6 - qp_out_tmp35;
    const s_t qp_out_tmp37 = s_t(2)*gu1;
    const s_t qp_out_tmp38 = gu5*gu6;
    const s_t qp_out_tmp39 = C01*(qp_out_tmp17*(s_t(2)*gu1*qp_out_tmp15 - gu2*qp_out_tmp33 - qp_out_tmp13*qp_out_tmp37 - qp_out_tmp19*qp_out_tmp4) + qp_out_tmp23*(((s_t(4) / s_t(3)))*qp_out_tmp35 - (s_t(4) / s_t(3))*qp_out_tmp38)) + C10*(qp_out_tmp10*qp_out_tmp37 + qp_out_tmp16*(((s_t(2) / s_t(3)))*qp_out_tmp35 - (s_t(2) / s_t(3))*qp_out_tmp38));
    const s_t qp_out_tmp40 = gu2*gu6;
    const s_t qp_out_tmp41 = qp_out_tmp0*qp_out_tmp4 - qp_out_tmp40;
    const s_t qp_out_tmp42 = s_t(2)*qp_out_tmp2;
    const s_t qp_out_tmp43 = qp_out_tmp0*qp_out_tmp4;
    const s_t qp_out_tmp44 = C01*(qp_out_tmp17*(-gu3*qp_out_tmp19 - gu5*qp_out_tmp33 - qp_out_tmp13*qp_out_tmp42 + s_t(2)*qp_out_tmp15*qp_out_tmp2) + qp_out_tmp23*(((s_t(4) / s_t(3)))*qp_out_tmp40 - (s_t(4) / s_t(3))*qp_out_tmp43)) + C10*(qp_out_tmp10*qp_out_tmp42 + qp_out_tmp16*(((s_t(2) / s_t(3)))*qp_out_tmp40 - (s_t(2) / s_t(3))*qp_out_tmp43));
    const s_t qp_out_tmp45 = gu1*gu6;
    const s_t qp_out_tmp46 = s_t(2)*gu5;
    const s_t qp_out_tmp47 = C01*(qp_out_tmp17*(-gu3*qp_out_tmp21 + s_t(2)*gu5*qp_out_tmp15 - qp_out_tmp14*qp_out_tmp46 - qp_out_tmp2*qp_out_tmp33) + qp_out_tmp23*(((s_t(4) / s_t(3)))*gu7*qp_out_tmp4 - (s_t(4) / s_t(3))*qp_out_tmp45)) + C10*(qp_out_tmp10*qp_out_tmp46 + qp_out_tmp16*(((s_t(2) / s_t(3)))*gu7*qp_out_tmp4 - (s_t(2) / s_t(3))*qp_out_tmp45));
    const s_t qp_out_tmp48 = gu3*gu7;
    const s_t qp_out_tmp49 = s_t(2)*gu2;
    const s_t qp_out_tmp50 = C01*(qp_out_tmp17*(s_t(2)*gu2*qp_out_tmp15 - qp_out_tmp14*qp_out_tmp49 - qp_out_tmp21*qp_out_tmp4 - qp_out_tmp22*qp_out_tmp37) + qp_out_tmp23*(((s_t(4) / s_t(3)))*gu6*qp_out_tmp2 - (s_t(4) / s_t(3))*qp_out_tmp48)) + C10*(qp_out_tmp10*qp_out_tmp49 + qp_out_tmp16*(((s_t(2) / s_t(3)))*gu6*qp_out_tmp2 - (s_t(2) / s_t(3))*qp_out_tmp48));
    const s_t qp_out_tmp51 = gu1*gu3;
    const s_t qp_out_tmp52 = s_t(2)*qp_out_tmp0;
    const s_t qp_out_tmp53 = qp_out_tmp2*qp_out_tmp4;
    const s_t qp_out_tmp54 = C01*(qp_out_tmp17*(-gu6*qp_out_tmp21 - gu7*qp_out_tmp33 + s_t(2)*qp_out_tmp0*qp_out_tmp15 - qp_out_tmp14*qp_out_tmp52) + qp_out_tmp23*(((s_t(4) / s_t(3)))*qp_out_tmp51 - (s_t(4) / s_t(3))*qp_out_tmp53)) + C10*(qp_out_tmp10*qp_out_tmp52 + qp_out_tmp16*(((s_t(2) / s_t(3)))*qp_out_tmp51 - (s_t(2) / s_t(3))*qp_out_tmp53));
    iso_stress_q[lane] = qp_out_tmp7*(qp_out_tmp24*(C01*(qp_out_tmp17*(s_t(2)*gu3*qp_out_tmp15 - gu5*qp_out_tmp21 - qp_out_tmp12*qp_out_tmp25 - qp_out_tmp19*qp_out_tmp2) + qp_out_tmp23*(((s_t(4) / s_t(3)))*qp_out_tmp1 - (s_t(4) / s_t(3))*qp_out_tmp26)) + C10*(qp_out_tmp10*qp_out_tmp25 + qp_out_tmp16*(((s_t(2) / s_t(3)))*qp_out_tmp1 - (s_t(2) / s_t(3))*qp_out_tmp26))) + qp_out_tmp27*(C01*(qp_out_tmp17*(-gu1*qp_out_tmp19 - gu2*qp_out_tmp21 - qp_out_tmp12*qp_out_tmp28 + s_t(2)*qp_out_tmp15*qp_out_tmp4) + qp_out_tmp23*(-(s_t(4) / s_t(3))*qp_out_tmp29 + ((s_t(4) / s_t(3)))*qp_out_tmp5)) + C10*(qp_out_tmp10*qp_out_tmp28 + qp_out_tmp16*(-(s_t(2) / s_t(3))*qp_out_tmp29 + ((s_t(2) / s_t(3)))*qp_out_tmp5))) + qp_out_tmp9*(C01*(qp_out_tmp17*(s_t(2)*gu6*qp_out_tmp15 - gu7*qp_out_tmp19 - qp_out_tmp0*qp_out_tmp21 - qp_out_tmp11*qp_out_tmp12) + qp_out_tmp23*(((s_t(4) / s_t(3)))*gu2*qp_out_tmp2 - (s_t(4) / s_t(3))*qp_out_tmp8)) + C10*(qp_out_tmp10*qp_out_tmp11 + qp_out_tmp16*(((s_t(2) / s_t(3)))*gu2*qp_out_tmp2 - (s_t(2) / s_t(3))*qp_out_tmp8))));
    iso_stress_q[1 * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp31*qp_out_tmp34 + qp_out_tmp36*qp_out_tmp39 + qp_out_tmp41*qp_out_tmp44);
    iso_stress_q[2 * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp47*(-gu7*qp_out_tmp4 + qp_out_tmp45) + qp_out_tmp50*(-gu6*qp_out_tmp2 + qp_out_tmp48) + qp_out_tmp54*(qp_out_tmp2*qp_out_tmp4 - qp_out_tmp51));
    iso_stress_q[3 * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp24*qp_out_tmp44 + qp_out_tmp27*qp_out_tmp39 + qp_out_tmp34*qp_out_tmp9);
    iso_stress_q[4 * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp24*qp_out_tmp47 + qp_out_tmp27*qp_out_tmp50 + qp_out_tmp54*qp_out_tmp9);
    iso_stress_q[5 * qp_stride + lane] = qp_out_tmp7*(qp_out_tmp31*qp_out_tmp54 + qp_out_tmp36*qp_out_tmp50 + qp_out_tmp41*qp_out_tmp47);
      }
    }
}

} // namespace codegen
} // namespace sfem

#endif
