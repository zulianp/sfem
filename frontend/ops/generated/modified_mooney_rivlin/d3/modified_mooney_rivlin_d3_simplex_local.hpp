#ifndef MODIFIED_MOONEY_RIVLIN_D3_SIMPLEX_LOCAL_HPP
#define MODIFIED_MOONEY_RIVLIN_D3_SIMPLEX_LOCAL_HPP
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
static SFEM_INLINE void modified_mooney_rivlin_d3_simplex_objective_block(
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
        const s_t c1,
        const s_t c2,
        const s_t kappa,
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
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref0_values[lane] += u_streams[3 * shape][lane] * grad_ref_x[q * NS + shape];
          grad_h_ref0_values[lane] += h_streams[3 * shape][lane] * grad_ref_x[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref1_values[lane] += u_streams[3 * shape][lane] * grad_ref_y[q * NS + shape];
          grad_h_ref1_values[lane] += h_streams[3 * shape][lane] * grad_ref_y[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref2_values[lane] += u_streams[3 * shape][lane] * grad_ref_z[q * NS + shape];
          grad_h_ref2_values[lane] += h_streams[3 * shape][lane] * grad_ref_z[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref3_values[lane] += u_streams[3 * shape + 1][lane] * grad_ref_x[q * NS + shape];
          grad_h_ref3_values[lane] += h_streams[3 * shape + 1][lane] * grad_ref_x[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref4_values[lane] += u_streams[3 * shape + 1][lane] * grad_ref_y[q * NS + shape];
          grad_h_ref4_values[lane] += h_streams[3 * shape + 1][lane] * grad_ref_y[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref5_values[lane] += u_streams[3 * shape + 1][lane] * grad_ref_z[q * NS + shape];
          grad_h_ref5_values[lane] += h_streams[3 * shape + 1][lane] * grad_ref_z[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref6_values[lane] += u_streams[3 * shape + 2][lane] * grad_ref_x[q * NS + shape];
          grad_h_ref6_values[lane] += h_streams[3 * shape + 2][lane] * grad_ref_x[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref7_values[lane] += u_streams[3 * shape + 2][lane] * grad_ref_y[q * NS + shape];
          grad_h_ref7_values[lane] += h_streams[3 * shape + 2][lane] * grad_ref_y[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref8_values[lane] += u_streams[3 * shape + 2][lane] * grad_ref_z[q * NS + shape];
          grad_h_ref8_values[lane] += h_streams[3 * shape + 2][lane] * grad_ref_z[q * NS + shape];
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
          const s_t gu0 = gu_base_v[0 * VS + lane] + alpha * trial_grad_v[0 * VS + lane];
          const s_t gu1 = gu_base_v[1 * VS + lane] + alpha * trial_grad_v[1 * VS + lane];
          const s_t gu2 = gu_base_v[2 * VS + lane] + alpha * trial_grad_v[2 * VS + lane];
          const s_t gu3 = gu_base_v[3 * VS + lane] + alpha * trial_grad_v[3 * VS + lane];
          const s_t gu4 = gu_base_v[4 * VS + lane] + alpha * trial_grad_v[4 * VS + lane];
          const s_t gu5 = gu_base_v[5 * VS + lane] + alpha * trial_grad_v[5 * VS + lane];
          const s_t gu6 = gu_base_v[6 * VS + lane] + alpha * trial_grad_v[6 * VS + lane];
          const s_t gu7 = gu_base_v[7 * VS + lane] + alpha * trial_grad_v[7 * VS + lane];
          const s_t gu8 = gu_base_v[8 * VS + lane] + alpha * trial_grad_v[8 * VS + lane];
    const s_t weak_obj_tmp0 = gu0*gu4;
    const s_t weak_obj_tmp1 = gu1*gu3;
    const s_t weak_obj_tmp2 = gu2*gu6;
    const s_t weak_obj_tmp3 = gu5*gu7;
    const s_t weak_obj_tmp4 = gu4 + s_t(1);
    const s_t weak_obj_tmp5 = pow_2(gu1) + pow_2(gu7) + pow_2(weak_obj_tmp4);
    const s_t weak_obj_tmp6 = gu8 + s_t(1);
    const s_t weak_obj_tmp7 = pow_2(gu2) + pow_2(gu5) + pow_2(weak_obj_tmp6);
    const s_t weak_obj_tmp8 = gu0 + s_t(1);
    const s_t weak_obj_tmp9 = pow_2(gu3) + pow_2(gu6) + pow_2(weak_obj_tmp8);
    const s_t weak_obj_tmp10 = weak_obj_tmp5 + weak_obj_tmp7 + weak_obj_tmp9;
    const s_t weak_obj_tmp11 = gu1*gu5*gu6 + gu2*gu3*gu7 - weak_obj_tmp1*weak_obj_tmp6 - weak_obj_tmp2*weak_obj_tmp4 - weak_obj_tmp3*weak_obj_tmp8 + weak_obj_tmp4*weak_obj_tmp6*weak_obj_tmp8;
    value[step * value_stride + lane] += qw * det_lane0 * (c1*(weak_obj_tmp10/pow(weak_obj_tmp11, (s_t(2) / s_t(3))) + s_t(-3)) + c2*(s_t(-3) + (((s_t(1) / s_t(2)))*pow_2(weak_obj_tmp10) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp5) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp7) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp9) - pow_2(gu1*gu2 + gu5*weak_obj_tmp4 + gu7*weak_obj_tmp6) - pow_2(gu1*weak_obj_tmp8 + gu3*weak_obj_tmp4 + gu6*gu7) - pow_2(gu2*weak_obj_tmp8 + gu3*gu5 + gu6*weak_obj_tmp6))/pow(weak_obj_tmp11, (s_t(4) / s_t(3)))) + ((s_t(1) / s_t(2)))*kappa*pow_2(sfem_log1p(gu0*gu8 - gu0*weak_obj_tmp3 + gu0 + gu1*gu5*gu6 + gu2*gu3*gu7 + gu4*gu8 - gu4*weak_obj_tmp2 + gu4 + gu8*weak_obj_tmp0 - gu8*weak_obj_tmp1 + gu8 + weak_obj_tmp0 - weak_obj_tmp1 - weak_obj_tmp2 - weak_obj_tmp3)));
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void modified_mooney_rivlin_d3_simplex_tet4_objective_block(
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
        const s_t *const RSTR q_weight,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 3],
        const s_t *const RSTR h_streams[NS * 3],
        const int nsteps,
        const s_t *const RSTR steps,
        const ptrdiff_t value_stride,
        s_t *const RSTR value
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
    { const int q = 0;  // constant-P1 simplex
      const s_t qw = q_weight[q];
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
          const ptrdiff_t goff = q * geometry_stride + lane;
          const s_t det_lane0 = det0[goff];
          const s_t gu0 = gu_base_v[0 * VS + lane] + alpha * trial_grad_v[0 * VS + lane];
          const s_t gu1 = gu_base_v[1 * VS + lane] + alpha * trial_grad_v[1 * VS + lane];
          const s_t gu2 = gu_base_v[2 * VS + lane] + alpha * trial_grad_v[2 * VS + lane];
          const s_t gu3 = gu_base_v[3 * VS + lane] + alpha * trial_grad_v[3 * VS + lane];
          const s_t gu4 = gu_base_v[4 * VS + lane] + alpha * trial_grad_v[4 * VS + lane];
          const s_t gu5 = gu_base_v[5 * VS + lane] + alpha * trial_grad_v[5 * VS + lane];
          const s_t gu6 = gu_base_v[6 * VS + lane] + alpha * trial_grad_v[6 * VS + lane];
          const s_t gu7 = gu_base_v[7 * VS + lane] + alpha * trial_grad_v[7 * VS + lane];
          const s_t gu8 = gu_base_v[8 * VS + lane] + alpha * trial_grad_v[8 * VS + lane];
    const s_t weak_obj_tmp0 = gu0*gu4;
    const s_t weak_obj_tmp1 = gu1*gu3;
    const s_t weak_obj_tmp2 = gu2*gu6;
    const s_t weak_obj_tmp3 = gu5*gu7;
    const s_t weak_obj_tmp4 = gu4 + s_t(1);
    const s_t weak_obj_tmp5 = pow_2(gu1) + pow_2(gu7) + pow_2(weak_obj_tmp4);
    const s_t weak_obj_tmp6 = gu8 + s_t(1);
    const s_t weak_obj_tmp7 = pow_2(gu2) + pow_2(gu5) + pow_2(weak_obj_tmp6);
    const s_t weak_obj_tmp8 = gu0 + s_t(1);
    const s_t weak_obj_tmp9 = pow_2(gu3) + pow_2(gu6) + pow_2(weak_obj_tmp8);
    const s_t weak_obj_tmp10 = weak_obj_tmp5 + weak_obj_tmp7 + weak_obj_tmp9;
    const s_t weak_obj_tmp11 = gu1*gu5*gu6 + gu2*gu3*gu7 - weak_obj_tmp1*weak_obj_tmp6 - weak_obj_tmp2*weak_obj_tmp4 - weak_obj_tmp3*weak_obj_tmp8 + weak_obj_tmp4*weak_obj_tmp6*weak_obj_tmp8;
    value[step * value_stride + lane] += qw * det_lane0 * (c1*(weak_obj_tmp10/pow(weak_obj_tmp11, (s_t(2) / s_t(3))) + s_t(-3)) + c2*(s_t(-3) + (((s_t(1) / s_t(2)))*pow_2(weak_obj_tmp10) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp5) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp7) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp9) - pow_2(gu1*gu2 + gu5*weak_obj_tmp4 + gu7*weak_obj_tmp6) - pow_2(gu1*weak_obj_tmp8 + gu3*weak_obj_tmp4 + gu6*gu7) - pow_2(gu2*weak_obj_tmp8 + gu3*gu5 + gu6*weak_obj_tmp6))/pow(weak_obj_tmp11, (s_t(4) / s_t(3)))) + ((s_t(1) / s_t(2)))*kappa*pow_2(sfem_log1p(gu0*gu8 - gu0*weak_obj_tmp3 + gu0 + gu1*gu5*gu6 + gu2*gu3*gu7 + gu4*gu8 - gu4*weak_obj_tmp2 + gu4 + gu8*weak_obj_tmp0 - gu8*weak_obj_tmp1 + gu8 + weak_obj_tmp0 - weak_obj_tmp1 - weak_obj_tmp2 - weak_obj_tmp3)));
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void modified_mooney_rivlin_d3_simplex_gradient_block(
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
        const s_t c1,
        const s_t c2,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 3],
        s_t *const RSTR out_streams[NS * 3]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
    for (int q = 0; q < NQ; ++q) {
      const s_t qw = q_weight[q];
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
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref0_values[lane] += u_streams[3 * shape][lane] * grad_ref_x[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref1_values[lane] += u_streams[3 * shape][lane] * grad_ref_y[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref2_values[lane] += u_streams[3 * shape][lane] * grad_ref_z[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref3_values[lane] += u_streams[3 * shape + 1][lane] * grad_ref_x[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref4_values[lane] += u_streams[3 * shape + 1][lane] * grad_ref_y[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref5_values[lane] += u_streams[3 * shape + 1][lane] * grad_ref_z[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref6_values[lane] += u_streams[3 * shape + 2][lane] * grad_ref_x[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref7_values[lane] += u_streams[3 * shape + 2][lane] * grad_ref_y[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref8_values[lane] += u_streams[3 * shape + 2][lane] * grad_ref_z[q * NS + shape];
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
    const s_t weak_mat_tmp0 = gu5*gu7;
    const s_t weak_mat_tmp1 = gu4 + s_t(1);
    const s_t weak_mat_tmp2 = gu8 + s_t(1);
    const s_t weak_mat_tmp3 = gu1*gu3;
    const s_t weak_mat_tmp4 = gu2*gu6;
    const s_t weak_mat_tmp5 = gu0 + s_t(1);
    const s_t weak_mat_tmp6 = gu1*gu5*gu6 + gu2*gu3*gu7 - weak_mat_tmp0*weak_mat_tmp5 + weak_mat_tmp1*weak_mat_tmp2*weak_mat_tmp5 - weak_mat_tmp1*weak_mat_tmp4 - weak_mat_tmp2*weak_mat_tmp3;
    const s_t weak_mat_tmp7 = gu0*gu4;
    const s_t weak_mat_tmp8 = gu5*gu6;
    const s_t weak_mat_tmp9 = gu3*gu7;
    const s_t weak_mat_tmp10 = kappa*sfem_log1p(gu0*gu8 - gu0*weak_mat_tmp0 + gu0 + gu1*weak_mat_tmp8 + gu2*weak_mat_tmp9 + gu4*gu8 - gu4*weak_mat_tmp4 + gu4 - gu8*weak_mat_tmp3 + gu8*weak_mat_tmp7 + gu8 - weak_mat_tmp0 - weak_mat_tmp3 - weak_mat_tmp4 + weak_mat_tmp7)/weak_mat_tmp6;
    const s_t weak_mat_tmp11 = pow(weak_mat_tmp6, (s_t(-2) / s_t(3)));
    const s_t weak_mat_tmp12 = s_t(2)*weak_mat_tmp5;
    const s_t weak_mat_tmp13 = weak_mat_tmp1*weak_mat_tmp2;
    const s_t weak_mat_tmp14 = pow_2(gu3) + pow_2(gu6) + pow_2(weak_mat_tmp5);
    const s_t weak_mat_tmp15 = pow_2(gu1) + pow_2(gu7) + pow_2(weak_mat_tmp1);
    const s_t weak_mat_tmp16 = pow_2(gu2) + pow_2(gu5) + pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp17 = weak_mat_tmp14 + weak_mat_tmp15 + weak_mat_tmp16;
    const s_t weak_mat_tmp18 = weak_mat_tmp17/pow(weak_mat_tmp6, (s_t(5) / s_t(3)));
    const s_t weak_mat_tmp19 = pow(weak_mat_tmp6, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp20 = gu1*weak_mat_tmp5 + gu3*weak_mat_tmp1 + gu6*gu7;
    const s_t weak_mat_tmp21 = s_t(2)*gu1;
    const s_t weak_mat_tmp22 = gu2*weak_mat_tmp5 + gu3*gu5 + gu6*weak_mat_tmp2;
    const s_t weak_mat_tmp23 = s_t(2)*gu2;
    const s_t weak_mat_tmp24 = gu1*gu2 + gu5*weak_mat_tmp1 + gu7*weak_mat_tmp2;
    const s_t weak_mat_tmp25 = (-(s_t(1) / s_t(2))*pow_2(weak_mat_tmp14) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp15) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp16) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp17) - pow_2(weak_mat_tmp20) - pow_2(weak_mat_tmp22) - pow_2(weak_mat_tmp24))/pow(weak_mat_tmp6, (s_t(7) / s_t(3)));
    const s_t weak_mat_tmp26 = gu3*weak_mat_tmp2;
    const s_t weak_mat_tmp27 = gu1*weak_mat_tmp2;
    const s_t weak_mat_tmp28 = s_t(2)*gu3;
    const s_t weak_mat_tmp29 = gu2*gu7;
    const s_t weak_mat_tmp30 = s_t(2)*gu5;
    const s_t weak_mat_tmp31 = s_t(2)*weak_mat_tmp1;
    const s_t weak_mat_tmp32 = weak_mat_tmp2*weak_mat_tmp5;
    const s_t weak_mat_tmp33 = gu1*gu6;
    const s_t weak_mat_tmp34 = gu1*gu5;
    const s_t weak_mat_tmp35 = s_t(2)*gu6;
    const s_t weak_mat_tmp36 = s_t(2)*gu7;
    const s_t weak_mat_tmp37 = s_t(2)*weak_mat_tmp2;
    const s_t weak_mat_tmp38 = gu2*gu3;
    const s_t weak_mat_tmp39 = weak_mat_tmp1*weak_mat_tmp5;
    const s_t material0 = c1*(weak_mat_tmp11*weak_mat_tmp12 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*weak_mat_tmp0 - (s_t(2) / s_t(3))*weak_mat_tmp13)) + c2*(weak_mat_tmp19*(-weak_mat_tmp12*weak_mat_tmp14 + s_t(2)*weak_mat_tmp17*weak_mat_tmp5 - weak_mat_tmp20*weak_mat_tmp21 - weak_mat_tmp22*weak_mat_tmp23) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*weak_mat_tmp0 - (s_t(4) / s_t(3))*weak_mat_tmp13)) + weak_mat_tmp10*(-weak_mat_tmp0 + weak_mat_tmp1*weak_mat_tmp2);
    const s_t material1 = c1*(weak_mat_tmp11*weak_mat_tmp21 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*weak_mat_tmp26 - (s_t(2) / s_t(3))*weak_mat_tmp8)) + c2*(weak_mat_tmp19*(s_t(2)*gu1*weak_mat_tmp17 - weak_mat_tmp12*weak_mat_tmp20 - weak_mat_tmp15*weak_mat_tmp21 - weak_mat_tmp23*weak_mat_tmp24) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*weak_mat_tmp26 - (s_t(4) / s_t(3))*weak_mat_tmp8)) + weak_mat_tmp10*(gu5*gu6 - weak_mat_tmp26);
    const s_t material2 = c1*(weak_mat_tmp11*weak_mat_tmp23 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(2) / s_t(3))*weak_mat_tmp9)) + c2*(weak_mat_tmp19*(s_t(2)*gu2*weak_mat_tmp17 - weak_mat_tmp12*weak_mat_tmp22 - weak_mat_tmp16*weak_mat_tmp23 - weak_mat_tmp21*weak_mat_tmp24) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(4) / s_t(3))*weak_mat_tmp9)) + weak_mat_tmp10*(-gu6*weak_mat_tmp1 + weak_mat_tmp9);
    const s_t material3 = c1*(weak_mat_tmp11*weak_mat_tmp28 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*weak_mat_tmp27 - (s_t(2) / s_t(3))*weak_mat_tmp29)) + c2*(weak_mat_tmp19*(s_t(2)*gu3*weak_mat_tmp17 - weak_mat_tmp14*weak_mat_tmp28 - weak_mat_tmp20*weak_mat_tmp31 - weak_mat_tmp22*weak_mat_tmp30) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*weak_mat_tmp27 - (s_t(4) / s_t(3))*weak_mat_tmp29)) + weak_mat_tmp10*(gu2*gu7 - weak_mat_tmp27);
    const s_t material4 = c1*(weak_mat_tmp11*weak_mat_tmp31 + weak_mat_tmp18*(-(s_t(2) / s_t(3))*weak_mat_tmp32 + ((s_t(2) / s_t(3)))*weak_mat_tmp4)) + c2*(weak_mat_tmp19*(s_t(2)*weak_mat_tmp1*weak_mat_tmp17 - weak_mat_tmp15*weak_mat_tmp31 - weak_mat_tmp20*weak_mat_tmp28 - weak_mat_tmp24*weak_mat_tmp30) + weak_mat_tmp25*(-(s_t(4) / s_t(3))*weak_mat_tmp32 + ((s_t(4) / s_t(3)))*weak_mat_tmp4)) + weak_mat_tmp10*(weak_mat_tmp2*weak_mat_tmp5 - weak_mat_tmp4);
    const s_t material5 = c1*(weak_mat_tmp11*weak_mat_tmp30 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu7*weak_mat_tmp5 - (s_t(2) / s_t(3))*weak_mat_tmp33)) + c2*(weak_mat_tmp19*(s_t(2)*gu5*weak_mat_tmp17 - weak_mat_tmp16*weak_mat_tmp30 - weak_mat_tmp22*weak_mat_tmp28 - weak_mat_tmp24*weak_mat_tmp31) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu7*weak_mat_tmp5 - (s_t(4) / s_t(3))*weak_mat_tmp33)) + weak_mat_tmp10*(-gu7*weak_mat_tmp5 + weak_mat_tmp33);
    const s_t material6 = c1*(weak_mat_tmp11*weak_mat_tmp35 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(2) / s_t(3))*weak_mat_tmp34)) + c2*(weak_mat_tmp19*(s_t(2)*gu6*weak_mat_tmp17 - weak_mat_tmp14*weak_mat_tmp35 - weak_mat_tmp20*weak_mat_tmp36 - weak_mat_tmp22*weak_mat_tmp37) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(4) / s_t(3))*weak_mat_tmp34)) + weak_mat_tmp10*(-gu2*weak_mat_tmp1 + weak_mat_tmp34);
    const s_t material7 = c1*(weak_mat_tmp11*weak_mat_tmp36 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu5*weak_mat_tmp5 - (s_t(2) / s_t(3))*weak_mat_tmp38)) + c2*(weak_mat_tmp19*(s_t(2)*gu7*weak_mat_tmp17 - weak_mat_tmp15*weak_mat_tmp36 - weak_mat_tmp20*weak_mat_tmp35 - weak_mat_tmp24*weak_mat_tmp37) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu5*weak_mat_tmp5 - (s_t(4) / s_t(3))*weak_mat_tmp38)) + weak_mat_tmp10*(-gu5*weak_mat_tmp5 + weak_mat_tmp38);
    const s_t material8 = c1*(weak_mat_tmp11*weak_mat_tmp37 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*weak_mat_tmp3 - (s_t(2) / s_t(3))*weak_mat_tmp39)) + c2*(weak_mat_tmp19*(-weak_mat_tmp16*weak_mat_tmp37 + s_t(2)*weak_mat_tmp17*weak_mat_tmp2 - weak_mat_tmp22*weak_mat_tmp35 - weak_mat_tmp24*weak_mat_tmp36) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*weak_mat_tmp3 - (s_t(4) / s_t(3))*weak_mat_tmp39)) + weak_mat_tmp10*(weak_mat_tmp1*weak_mat_tmp5 - weak_mat_tmp3);
    const s_t loperand0 = qw * (material0 * adj_lane0 + material1 * adj_lane1 + material2 * adj_lane2);
    const s_t loperand1 = qw * (material0 * adj_lane3 + material1 * adj_lane4 + material2 * adj_lane5);
    const s_t loperand2 = qw * (material0 * adj_lane6 + material1 * adj_lane7 + material2 * adj_lane8);
    const s_t loperand3 = qw * (material3 * adj_lane0 + material4 * adj_lane1 + material5 * adj_lane2);
    const s_t loperand4 = qw * (material3 * adj_lane3 + material4 * adj_lane4 + material5 * adj_lane5);
    const s_t loperand5 = qw * (material3 * adj_lane6 + material4 * adj_lane7 + material5 * adj_lane8);
    const s_t loperand6 = qw * (material6 * adj_lane0 + material7 * adj_lane1 + material8 * adj_lane2);
    const s_t loperand7 = qw * (material6 * adj_lane3 + material7 * adj_lane4 + material8 * adj_lane5);
    const s_t loperand8 = qw * (material6 * adj_lane6 + material7 * adj_lane7 + material8 * adj_lane8);
      loperand0_values[lane] = loperand0;
      loperand1_values[lane] = loperand1;
      loperand2_values[lane] = loperand2;
      loperand3_values[lane] = loperand3;
      loperand4_values[lane] = loperand4;
      loperand5_values[lane] = loperand5;
      loperand6_values[lane] = loperand6;
      loperand7_values[lane] = loperand7;
      loperand8_values[lane] = loperand8;
      }
      for (int shape = 0; shape < NS; ++shape) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          out_streams[3 * shape][lane] += loperand0_values[lane] * grad_ref_x[q * NS + shape] + loperand1_values[lane] * grad_ref_y[q * NS + shape] + loperand2_values[lane] * grad_ref_z[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          out_streams[3 * shape + 1][lane] += loperand3_values[lane] * grad_ref_x[q * NS + shape] + loperand4_values[lane] * grad_ref_y[q * NS + shape] + loperand5_values[lane] * grad_ref_z[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          out_streams[3 * shape + 2][lane] += loperand6_values[lane] * grad_ref_x[q * NS + shape] + loperand7_values[lane] * grad_ref_y[q * NS + shape] + loperand8_values[lane] * grad_ref_z[q * NS + shape];
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void modified_mooney_rivlin_d3_simplex_tet4_gradient_block(
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
        const s_t *const RSTR q_weight,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 3],
        s_t *const RSTR out_streams[NS * 3]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
    { const int q = 0;  // constant-P1 simplex
      const s_t qw = q_weight[q];
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
    const s_t weak_mat_tmp0 = gu5*gu7;
    const s_t weak_mat_tmp1 = gu4 + s_t(1);
    const s_t weak_mat_tmp2 = gu8 + s_t(1);
    const s_t weak_mat_tmp3 = gu1*gu3;
    const s_t weak_mat_tmp4 = gu2*gu6;
    const s_t weak_mat_tmp5 = gu0 + s_t(1);
    const s_t weak_mat_tmp6 = gu1*gu5*gu6 + gu2*gu3*gu7 - weak_mat_tmp0*weak_mat_tmp5 + weak_mat_tmp1*weak_mat_tmp2*weak_mat_tmp5 - weak_mat_tmp1*weak_mat_tmp4 - weak_mat_tmp2*weak_mat_tmp3;
    const s_t weak_mat_tmp7 = gu0*gu4;
    const s_t weak_mat_tmp8 = gu5*gu6;
    const s_t weak_mat_tmp9 = gu3*gu7;
    const s_t weak_mat_tmp10 = kappa*sfem_log1p(gu0*gu8 - gu0*weak_mat_tmp0 + gu0 + gu1*weak_mat_tmp8 + gu2*weak_mat_tmp9 + gu4*gu8 - gu4*weak_mat_tmp4 + gu4 - gu8*weak_mat_tmp3 + gu8*weak_mat_tmp7 + gu8 - weak_mat_tmp0 - weak_mat_tmp3 - weak_mat_tmp4 + weak_mat_tmp7)/weak_mat_tmp6;
    const s_t weak_mat_tmp11 = pow(weak_mat_tmp6, (s_t(-2) / s_t(3)));
    const s_t weak_mat_tmp12 = s_t(2)*weak_mat_tmp5;
    const s_t weak_mat_tmp13 = weak_mat_tmp1*weak_mat_tmp2;
    const s_t weak_mat_tmp14 = pow_2(gu3) + pow_2(gu6) + pow_2(weak_mat_tmp5);
    const s_t weak_mat_tmp15 = pow_2(gu1) + pow_2(gu7) + pow_2(weak_mat_tmp1);
    const s_t weak_mat_tmp16 = pow_2(gu2) + pow_2(gu5) + pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp17 = weak_mat_tmp14 + weak_mat_tmp15 + weak_mat_tmp16;
    const s_t weak_mat_tmp18 = weak_mat_tmp17/pow(weak_mat_tmp6, (s_t(5) / s_t(3)));
    const s_t weak_mat_tmp19 = pow(weak_mat_tmp6, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp20 = gu1*weak_mat_tmp5 + gu3*weak_mat_tmp1 + gu6*gu7;
    const s_t weak_mat_tmp21 = s_t(2)*gu1;
    const s_t weak_mat_tmp22 = gu2*weak_mat_tmp5 + gu3*gu5 + gu6*weak_mat_tmp2;
    const s_t weak_mat_tmp23 = s_t(2)*gu2;
    const s_t weak_mat_tmp24 = gu1*gu2 + gu5*weak_mat_tmp1 + gu7*weak_mat_tmp2;
    const s_t weak_mat_tmp25 = (-(s_t(1) / s_t(2))*pow_2(weak_mat_tmp14) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp15) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp16) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp17) - pow_2(weak_mat_tmp20) - pow_2(weak_mat_tmp22) - pow_2(weak_mat_tmp24))/pow(weak_mat_tmp6, (s_t(7) / s_t(3)));
    const s_t weak_mat_tmp26 = gu3*weak_mat_tmp2;
    const s_t weak_mat_tmp27 = gu1*weak_mat_tmp2;
    const s_t weak_mat_tmp28 = s_t(2)*gu3;
    const s_t weak_mat_tmp29 = gu2*gu7;
    const s_t weak_mat_tmp30 = s_t(2)*gu5;
    const s_t weak_mat_tmp31 = s_t(2)*weak_mat_tmp1;
    const s_t weak_mat_tmp32 = weak_mat_tmp2*weak_mat_tmp5;
    const s_t weak_mat_tmp33 = gu1*gu6;
    const s_t weak_mat_tmp34 = gu1*gu5;
    const s_t weak_mat_tmp35 = s_t(2)*gu6;
    const s_t weak_mat_tmp36 = s_t(2)*gu7;
    const s_t weak_mat_tmp37 = s_t(2)*weak_mat_tmp2;
    const s_t weak_mat_tmp38 = gu2*gu3;
    const s_t weak_mat_tmp39 = weak_mat_tmp1*weak_mat_tmp5;
    const s_t material0 = c1*(weak_mat_tmp11*weak_mat_tmp12 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*weak_mat_tmp0 - (s_t(2) / s_t(3))*weak_mat_tmp13)) + c2*(weak_mat_tmp19*(-weak_mat_tmp12*weak_mat_tmp14 + s_t(2)*weak_mat_tmp17*weak_mat_tmp5 - weak_mat_tmp20*weak_mat_tmp21 - weak_mat_tmp22*weak_mat_tmp23) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*weak_mat_tmp0 - (s_t(4) / s_t(3))*weak_mat_tmp13)) + weak_mat_tmp10*(-weak_mat_tmp0 + weak_mat_tmp1*weak_mat_tmp2);
    const s_t material1 = c1*(weak_mat_tmp11*weak_mat_tmp21 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*weak_mat_tmp26 - (s_t(2) / s_t(3))*weak_mat_tmp8)) + c2*(weak_mat_tmp19*(s_t(2)*gu1*weak_mat_tmp17 - weak_mat_tmp12*weak_mat_tmp20 - weak_mat_tmp15*weak_mat_tmp21 - weak_mat_tmp23*weak_mat_tmp24) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*weak_mat_tmp26 - (s_t(4) / s_t(3))*weak_mat_tmp8)) + weak_mat_tmp10*(gu5*gu6 - weak_mat_tmp26);
    const s_t material2 = c1*(weak_mat_tmp11*weak_mat_tmp23 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(2) / s_t(3))*weak_mat_tmp9)) + c2*(weak_mat_tmp19*(s_t(2)*gu2*weak_mat_tmp17 - weak_mat_tmp12*weak_mat_tmp22 - weak_mat_tmp16*weak_mat_tmp23 - weak_mat_tmp21*weak_mat_tmp24) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(4) / s_t(3))*weak_mat_tmp9)) + weak_mat_tmp10*(-gu6*weak_mat_tmp1 + weak_mat_tmp9);
    const s_t material3 = c1*(weak_mat_tmp11*weak_mat_tmp28 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*weak_mat_tmp27 - (s_t(2) / s_t(3))*weak_mat_tmp29)) + c2*(weak_mat_tmp19*(s_t(2)*gu3*weak_mat_tmp17 - weak_mat_tmp14*weak_mat_tmp28 - weak_mat_tmp20*weak_mat_tmp31 - weak_mat_tmp22*weak_mat_tmp30) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*weak_mat_tmp27 - (s_t(4) / s_t(3))*weak_mat_tmp29)) + weak_mat_tmp10*(gu2*gu7 - weak_mat_tmp27);
    const s_t material4 = c1*(weak_mat_tmp11*weak_mat_tmp31 + weak_mat_tmp18*(-(s_t(2) / s_t(3))*weak_mat_tmp32 + ((s_t(2) / s_t(3)))*weak_mat_tmp4)) + c2*(weak_mat_tmp19*(s_t(2)*weak_mat_tmp1*weak_mat_tmp17 - weak_mat_tmp15*weak_mat_tmp31 - weak_mat_tmp20*weak_mat_tmp28 - weak_mat_tmp24*weak_mat_tmp30) + weak_mat_tmp25*(-(s_t(4) / s_t(3))*weak_mat_tmp32 + ((s_t(4) / s_t(3)))*weak_mat_tmp4)) + weak_mat_tmp10*(weak_mat_tmp2*weak_mat_tmp5 - weak_mat_tmp4);
    const s_t material5 = c1*(weak_mat_tmp11*weak_mat_tmp30 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu7*weak_mat_tmp5 - (s_t(2) / s_t(3))*weak_mat_tmp33)) + c2*(weak_mat_tmp19*(s_t(2)*gu5*weak_mat_tmp17 - weak_mat_tmp16*weak_mat_tmp30 - weak_mat_tmp22*weak_mat_tmp28 - weak_mat_tmp24*weak_mat_tmp31) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu7*weak_mat_tmp5 - (s_t(4) / s_t(3))*weak_mat_tmp33)) + weak_mat_tmp10*(-gu7*weak_mat_tmp5 + weak_mat_tmp33);
    const s_t material6 = c1*(weak_mat_tmp11*weak_mat_tmp35 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(2) / s_t(3))*weak_mat_tmp34)) + c2*(weak_mat_tmp19*(s_t(2)*gu6*weak_mat_tmp17 - weak_mat_tmp14*weak_mat_tmp35 - weak_mat_tmp20*weak_mat_tmp36 - weak_mat_tmp22*weak_mat_tmp37) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(4) / s_t(3))*weak_mat_tmp34)) + weak_mat_tmp10*(-gu2*weak_mat_tmp1 + weak_mat_tmp34);
    const s_t material7 = c1*(weak_mat_tmp11*weak_mat_tmp36 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu5*weak_mat_tmp5 - (s_t(2) / s_t(3))*weak_mat_tmp38)) + c2*(weak_mat_tmp19*(s_t(2)*gu7*weak_mat_tmp17 - weak_mat_tmp15*weak_mat_tmp36 - weak_mat_tmp20*weak_mat_tmp35 - weak_mat_tmp24*weak_mat_tmp37) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu5*weak_mat_tmp5 - (s_t(4) / s_t(3))*weak_mat_tmp38)) + weak_mat_tmp10*(-gu5*weak_mat_tmp5 + weak_mat_tmp38);
    const s_t material8 = c1*(weak_mat_tmp11*weak_mat_tmp37 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*weak_mat_tmp3 - (s_t(2) / s_t(3))*weak_mat_tmp39)) + c2*(weak_mat_tmp19*(-weak_mat_tmp16*weak_mat_tmp37 + s_t(2)*weak_mat_tmp17*weak_mat_tmp2 - weak_mat_tmp22*weak_mat_tmp35 - weak_mat_tmp24*weak_mat_tmp36) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*weak_mat_tmp3 - (s_t(4) / s_t(3))*weak_mat_tmp39)) + weak_mat_tmp10*(weak_mat_tmp1*weak_mat_tmp5 - weak_mat_tmp3);
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
static SFEM_INLINE void modified_mooney_rivlin_d3_simplex_apply_block(
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
        const s_t c1,
        const s_t c2,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 3],
        const s_t *const RSTR h_streams[NS * 3],
        s_t *const RSTR out_streams[NS * 3]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
    for (int q = 0; q < NQ; ++q) {
      const s_t qw = q_weight[q];
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
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref0_values[lane] += u_streams[3 * shape][lane] * grad_ref_x[q * NS + shape];
          grad_h_ref0_values[lane] += h_streams[3 * shape][lane] * grad_ref_x[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref1_values[lane] += u_streams[3 * shape][lane] * grad_ref_y[q * NS + shape];
          grad_h_ref1_values[lane] += h_streams[3 * shape][lane] * grad_ref_y[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref2_values[lane] += u_streams[3 * shape][lane] * grad_ref_z[q * NS + shape];
          grad_h_ref2_values[lane] += h_streams[3 * shape][lane] * grad_ref_z[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref3_values[lane] += u_streams[3 * shape + 1][lane] * grad_ref_x[q * NS + shape];
          grad_h_ref3_values[lane] += h_streams[3 * shape + 1][lane] * grad_ref_x[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref4_values[lane] += u_streams[3 * shape + 1][lane] * grad_ref_y[q * NS + shape];
          grad_h_ref4_values[lane] += h_streams[3 * shape + 1][lane] * grad_ref_y[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref5_values[lane] += u_streams[3 * shape + 1][lane] * grad_ref_z[q * NS + shape];
          grad_h_ref5_values[lane] += h_streams[3 * shape + 1][lane] * grad_ref_z[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref6_values[lane] += u_streams[3 * shape + 2][lane] * grad_ref_x[q * NS + shape];
          grad_h_ref6_values[lane] += h_streams[3 * shape + 2][lane] * grad_ref_x[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref7_values[lane] += u_streams[3 * shape + 2][lane] * grad_ref_y[q * NS + shape];
          grad_h_ref7_values[lane] += h_streams[3 * shape + 2][lane] * grad_ref_y[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          gu_ref8_values[lane] += u_streams[3 * shape + 2][lane] * grad_ref_z[q * NS + shape];
          grad_h_ref8_values[lane] += h_streams[3 * shape + 2][lane] * grad_ref_z[q * NS + shape];
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
    const s_t weak_mat_tmp3 = weak_mat_tmp0 - weak_mat_tmp1*weak_mat_tmp2;
    const s_t weak_mat_tmp4 = -weak_mat_tmp3;
    const s_t weak_mat_tmp5 = gu1*gu3;
    const s_t weak_mat_tmp6 = gu2*gu6;
    const s_t weak_mat_tmp7 = gu0 + s_t(1);
    const s_t weak_mat_tmp8 = gu1*gu5*gu6 + gu2*gu3*gu7 - weak_mat_tmp0*weak_mat_tmp7 + weak_mat_tmp1*weak_mat_tmp2*weak_mat_tmp7 - weak_mat_tmp1*weak_mat_tmp6 - weak_mat_tmp2*weak_mat_tmp5;
    const s_t weak_mat_tmp9 = kappa/pow_2(weak_mat_tmp8);
    const s_t weak_mat_tmp10 = gu0*gu4;
    const s_t weak_mat_tmp11 = gu5*gu6;
    const s_t weak_mat_tmp12 = gu3*gu7;
    const s_t weak_mat_tmp13 = sfem_log1p(gu0*gu8 - gu0*weak_mat_tmp0 + gu0 + gu1*weak_mat_tmp11 + gu2*weak_mat_tmp12 + gu4*gu8 - gu4*weak_mat_tmp6 + gu4 + gu8*weak_mat_tmp10 - gu8*weak_mat_tmp5 + gu8 - weak_mat_tmp0 + weak_mat_tmp10 - weak_mat_tmp5 - weak_mat_tmp6);
    const s_t weak_mat_tmp14 = weak_mat_tmp4*weak_mat_tmp9;
    const s_t weak_mat_tmp15 = weak_mat_tmp13*weak_mat_tmp14;
    const s_t weak_mat_tmp16 = s_t(2)/pow(weak_mat_tmp8, (s_t(2) / s_t(3)));
    const s_t weak_mat_tmp17 = pow(weak_mat_tmp8, (s_t(-5) / s_t(3)));
    const s_t weak_mat_tmp18 = weak_mat_tmp1*weak_mat_tmp2;
    const s_t weak_mat_tmp19 = ((s_t(2) / s_t(3)))*weak_mat_tmp0 - (s_t(2) / s_t(3))*weak_mat_tmp18;
    const s_t weak_mat_tmp20 = weak_mat_tmp17*weak_mat_tmp19;
    const s_t weak_mat_tmp21 = ((s_t(5) / s_t(3)))*weak_mat_tmp0 - (s_t(5) / s_t(3))*weak_mat_tmp18;
    const s_t weak_mat_tmp22 = pow_2(gu3);
    const s_t weak_mat_tmp23 = pow_2(gu6);
    const s_t weak_mat_tmp24 = pow_2(weak_mat_tmp7);
    const s_t weak_mat_tmp25 = weak_mat_tmp22 + weak_mat_tmp23 + weak_mat_tmp24;
    const s_t weak_mat_tmp26 = pow_2(gu1);
    const s_t weak_mat_tmp27 = pow_2(gu7);
    const s_t weak_mat_tmp28 = pow_2(weak_mat_tmp1);
    const s_t weak_mat_tmp29 = weak_mat_tmp26 + weak_mat_tmp27 + weak_mat_tmp28;
    const s_t weak_mat_tmp30 = pow_2(gu2);
    const s_t weak_mat_tmp31 = pow_2(gu5);
    const s_t weak_mat_tmp32 = pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp33 = weak_mat_tmp30 + weak_mat_tmp31 + weak_mat_tmp32;
    const s_t weak_mat_tmp34 = weak_mat_tmp25 + weak_mat_tmp29 + weak_mat_tmp33;
    const s_t weak_mat_tmp35 = weak_mat_tmp34/pow(weak_mat_tmp8, (s_t(8) / s_t(3)));
    const s_t weak_mat_tmp36 = weak_mat_tmp19*weak_mat_tmp35;
    const s_t weak_mat_tmp37 = s_t(2)*weak_mat_tmp31;
    const s_t weak_mat_tmp38 = s_t(2)*weak_mat_tmp32;
    const s_t weak_mat_tmp39 = weak_mat_tmp37 + weak_mat_tmp38;
    const s_t weak_mat_tmp40 = s_t(2)*weak_mat_tmp27;
    const s_t weak_mat_tmp41 = s_t(2)*weak_mat_tmp28;
    const s_t weak_mat_tmp42 = weak_mat_tmp40 + weak_mat_tmp41;
    const s_t weak_mat_tmp43 = pow(weak_mat_tmp8, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp44 = ((s_t(4) / s_t(3)))*weak_mat_tmp0 - (s_t(4) / s_t(3))*weak_mat_tmp18;
    const s_t weak_mat_tmp45 = pow(weak_mat_tmp8, (s_t(-7) / s_t(3)));
    const s_t weak_mat_tmp46 = gu6*gu7;
    const s_t weak_mat_tmp47 = gu1*weak_mat_tmp7;
    const s_t weak_mat_tmp48 = gu3*weak_mat_tmp1;
    const s_t weak_mat_tmp49 = weak_mat_tmp46 + weak_mat_tmp47 + weak_mat_tmp48;
    const s_t weak_mat_tmp50 = s_t(2)*gu1;
    const s_t weak_mat_tmp51 = gu3*gu5;
    const s_t weak_mat_tmp52 = gu2*weak_mat_tmp7;
    const s_t weak_mat_tmp53 = gu6*weak_mat_tmp2;
    const s_t weak_mat_tmp54 = weak_mat_tmp51 + weak_mat_tmp52 + weak_mat_tmp53;
    const s_t weak_mat_tmp55 = s_t(2)*gu2;
    const s_t weak_mat_tmp56 = s_t(2)*weak_mat_tmp7;
    const s_t weak_mat_tmp57 = weak_mat_tmp45*(-weak_mat_tmp25*weak_mat_tmp56 + s_t(2)*weak_mat_tmp34*weak_mat_tmp7 - weak_mat_tmp49*weak_mat_tmp50 - weak_mat_tmp54*weak_mat_tmp55);
    const s_t weak_mat_tmp58 = ((s_t(7) / s_t(3)))*weak_mat_tmp0 - (s_t(7) / s_t(3))*weak_mat_tmp18;
    const s_t weak_mat_tmp59 = gu1*gu2;
    const s_t weak_mat_tmp60 = gu5*weak_mat_tmp1;
    const s_t weak_mat_tmp61 = gu7*weak_mat_tmp2;
    const s_t weak_mat_tmp62 = weak_mat_tmp59 + weak_mat_tmp60 + weak_mat_tmp61;
    const s_t weak_mat_tmp63 = -(s_t(1) / s_t(2))*pow_2(weak_mat_tmp25) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp29) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp33) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp34) - pow_2(weak_mat_tmp49) - pow_2(weak_mat_tmp54) - pow_2(weak_mat_tmp62);
    const s_t weak_mat_tmp64 = weak_mat_tmp63/pow(weak_mat_tmp8, (s_t(10) / s_t(3)));
    const s_t weak_mat_tmp65 = weak_mat_tmp44*weak_mat_tmp64;
    const s_t weak_mat_tmp66 = gu3*weak_mat_tmp2;
    const s_t weak_mat_tmp67 = -gu5*gu6 + weak_mat_tmp66;
    const s_t weak_mat_tmp68 = -weak_mat_tmp67;
    const s_t weak_mat_tmp69 = weak_mat_tmp14*weak_mat_tmp68;
    const s_t weak_mat_tmp70 = -(s_t(5) / s_t(3))*weak_mat_tmp11 + ((s_t(5) / s_t(3)))*weak_mat_tmp66;
    const s_t weak_mat_tmp71 = -(s_t(2) / s_t(3))*weak_mat_tmp11 + ((s_t(2) / s_t(3)))*weak_mat_tmp66;
    const s_t weak_mat_tmp72 = weak_mat_tmp17*weak_mat_tmp56;
    const s_t weak_mat_tmp73 = weak_mat_tmp20*weak_mat_tmp50 + weak_mat_tmp71*weak_mat_tmp72;
    const s_t weak_mat_tmp74 = -(s_t(7) / s_t(3))*weak_mat_tmp11 + ((s_t(7) / s_t(3)))*weak_mat_tmp66;
    const s_t weak_mat_tmp75 = s_t(2)*weak_mat_tmp46;
    const s_t weak_mat_tmp76 = s_t(2)*weak_mat_tmp48;
    const s_t weak_mat_tmp77 = -(s_t(4) / s_t(3))*weak_mat_tmp11 + ((s_t(4) / s_t(3)))*weak_mat_tmp66;
    const s_t weak_mat_tmp78 = s_t(2)*gu1*weak_mat_tmp34 - weak_mat_tmp29*weak_mat_tmp50 - weak_mat_tmp49*weak_mat_tmp56 - weak_mat_tmp55*weak_mat_tmp62;
    const s_t weak_mat_tmp79 = weak_mat_tmp44*weak_mat_tmp45;
    const s_t weak_mat_tmp80 = weak_mat_tmp43*(-weak_mat_tmp75 - weak_mat_tmp76) + weak_mat_tmp57*weak_mat_tmp77 + weak_mat_tmp78*weak_mat_tmp79;
    const s_t weak_mat_tmp81 = gu6*weak_mat_tmp1;
    const s_t weak_mat_tmp82 = weak_mat_tmp12 - weak_mat_tmp81;
    const s_t weak_mat_tmp83 = weak_mat_tmp14*weak_mat_tmp82;
    const s_t weak_mat_tmp84 = -weak_mat_tmp82;
    const s_t weak_mat_tmp85 = ((s_t(5) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(5) / s_t(3))*weak_mat_tmp12;
    const s_t weak_mat_tmp86 = ((s_t(2) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(2) / s_t(3))*weak_mat_tmp12;
    const s_t weak_mat_tmp87 = weak_mat_tmp20*weak_mat_tmp55 + weak_mat_tmp72*weak_mat_tmp86;
    const s_t weak_mat_tmp88 = ((s_t(7) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(7) / s_t(3))*weak_mat_tmp12;
    const s_t weak_mat_tmp89 = s_t(2)*weak_mat_tmp51;
    const s_t weak_mat_tmp90 = s_t(2)*weak_mat_tmp53;
    const s_t weak_mat_tmp91 = ((s_t(4) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(4) / s_t(3))*weak_mat_tmp12;
    const s_t weak_mat_tmp92 = s_t(2)*gu2*weak_mat_tmp34 - weak_mat_tmp33*weak_mat_tmp55 - weak_mat_tmp50*weak_mat_tmp62 - weak_mat_tmp54*weak_mat_tmp56;
    const s_t weak_mat_tmp93 = weak_mat_tmp43*(-weak_mat_tmp89 - weak_mat_tmp90) + weak_mat_tmp57*weak_mat_tmp91 + weak_mat_tmp79*weak_mat_tmp92;
    const s_t weak_mat_tmp94 = gu1*weak_mat_tmp2;
    const s_t weak_mat_tmp95 = -gu2*gu7 + weak_mat_tmp94;
    const s_t weak_mat_tmp96 = -weak_mat_tmp95;
    const s_t weak_mat_tmp97 = weak_mat_tmp14*weak_mat_tmp96;
    const s_t weak_mat_tmp98 = gu2*gu7;
    const s_t weak_mat_tmp99 = ((s_t(5) / s_t(3)))*weak_mat_tmp94 - (s_t(5) / s_t(3))*weak_mat_tmp98;
    const s_t weak_mat_tmp100 = s_t(2)*gu3;
    const s_t weak_mat_tmp101 = ((s_t(2) / s_t(3)))*weak_mat_tmp94 - (s_t(2) / s_t(3))*weak_mat_tmp98;
    const s_t weak_mat_tmp102 = weak_mat_tmp100*weak_mat_tmp20 + weak_mat_tmp101*weak_mat_tmp72;
    const s_t weak_mat_tmp103 = ((s_t(7) / s_t(3)))*weak_mat_tmp94 - (s_t(7) / s_t(3))*weak_mat_tmp98;
    const s_t weak_mat_tmp104 = gu5*weak_mat_tmp55;
    const s_t weak_mat_tmp105 = weak_mat_tmp1*weak_mat_tmp50;
    const s_t weak_mat_tmp106 = ((s_t(4) / s_t(3)))*weak_mat_tmp94 - (s_t(4) / s_t(3))*weak_mat_tmp98;
    const s_t weak_mat_tmp107 = s_t(2)*gu5;
    const s_t weak_mat_tmp108 = s_t(2)*weak_mat_tmp1;
    const s_t weak_mat_tmp109 = s_t(2)*gu3*weak_mat_tmp34 - weak_mat_tmp100*weak_mat_tmp25 - weak_mat_tmp107*weak_mat_tmp54 - weak_mat_tmp108*weak_mat_tmp49;
    const s_t weak_mat_tmp110 = weak_mat_tmp106*weak_mat_tmp57 + weak_mat_tmp109*weak_mat_tmp79 + weak_mat_tmp43*(-weak_mat_tmp104 - weak_mat_tmp105);
    const s_t weak_mat_tmp111 = gu1*gu5;
    const s_t weak_mat_tmp112 = gu2*weak_mat_tmp1;
    const s_t weak_mat_tmp113 = weak_mat_tmp111 - weak_mat_tmp112;
    const s_t weak_mat_tmp114 = weak_mat_tmp113*weak_mat_tmp14;
    const s_t weak_mat_tmp115 = -weak_mat_tmp113;
    const s_t weak_mat_tmp116 = ((s_t(5) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(5) / s_t(3))*weak_mat_tmp111;
    const s_t weak_mat_tmp117 = s_t(2)*gu6;
    const s_t weak_mat_tmp118 = ((s_t(2) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(2) / s_t(3))*weak_mat_tmp111;
    const s_t weak_mat_tmp119 = weak_mat_tmp117*weak_mat_tmp20 + weak_mat_tmp118*weak_mat_tmp72;
    const s_t weak_mat_tmp120 = ((s_t(7) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(7) / s_t(3))*weak_mat_tmp111;
    const s_t weak_mat_tmp121 = gu7*weak_mat_tmp50;
    const s_t weak_mat_tmp122 = weak_mat_tmp2*weak_mat_tmp55;
    const s_t weak_mat_tmp123 = ((s_t(4) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(4) / s_t(3))*weak_mat_tmp111;
    const s_t weak_mat_tmp124 = s_t(2)*gu7;
    const s_t weak_mat_tmp125 = s_t(2)*weak_mat_tmp2;
    const s_t weak_mat_tmp126 = s_t(2)*gu6*weak_mat_tmp34 - weak_mat_tmp117*weak_mat_tmp25 - weak_mat_tmp124*weak_mat_tmp49 - weak_mat_tmp125*weak_mat_tmp54;
    const s_t weak_mat_tmp127 = weak_mat_tmp123*weak_mat_tmp57 + weak_mat_tmp126*weak_mat_tmp79 + weak_mat_tmp43*(-weak_mat_tmp121 - weak_mat_tmp122);
    const s_t weak_mat_tmp128 = gu1*gu6;
    const s_t weak_mat_tmp129 = ((s_t(5) / s_t(3)))*gu7*weak_mat_tmp7 - (s_t(5) / s_t(3))*weak_mat_tmp128;
    const s_t weak_mat_tmp130 = ((s_t(2) / s_t(3)))*gu7*weak_mat_tmp7 - (s_t(2) / s_t(3))*weak_mat_tmp128;
    const s_t weak_mat_tmp131 = ((s_t(2) / s_t(3)))*weak_mat_tmp34;
    const s_t weak_mat_tmp132 = weak_mat_tmp131*weak_mat_tmp17;
    const s_t weak_mat_tmp133 = gu7*weak_mat_tmp132;
    const s_t weak_mat_tmp134 = weak_mat_tmp107*weak_mat_tmp20 + weak_mat_tmp130*weak_mat_tmp72 + weak_mat_tmp133;
    const s_t weak_mat_tmp135 = ((s_t(7) / s_t(3)))*gu7*weak_mat_tmp7 - (s_t(7) / s_t(3))*weak_mat_tmp128;
    const s_t weak_mat_tmp136 = gu2*gu3;
    const s_t weak_mat_tmp137 = ((s_t(4) / s_t(3)))*gu7*weak_mat_tmp7 - (s_t(4) / s_t(3))*weak_mat_tmp128;
    const s_t weak_mat_tmp138 = s_t(2)*gu5*weak_mat_tmp34 - weak_mat_tmp100*weak_mat_tmp54 - weak_mat_tmp107*weak_mat_tmp33 - weak_mat_tmp108*weak_mat_tmp62;
    const s_t weak_mat_tmp139 = ((s_t(4) / s_t(3)))*weak_mat_tmp45*weak_mat_tmp63;
    const s_t weak_mat_tmp140 = gu7*weak_mat_tmp139;
    const s_t weak_mat_tmp141 = weak_mat_tmp137*weak_mat_tmp57 + weak_mat_tmp138*weak_mat_tmp79 + weak_mat_tmp140 + weak_mat_tmp43*(s_t(4)*gu5*weak_mat_tmp7 - s_t(2)*weak_mat_tmp136);
    const s_t weak_mat_tmp142 = gu7*weak_mat_tmp7;
    const s_t weak_mat_tmp143 = weak_mat_tmp128 - weak_mat_tmp142;
    const s_t weak_mat_tmp144 = -weak_mat_tmp143;
    const s_t weak_mat_tmp145 = kappa*weak_mat_tmp13/weak_mat_tmp8;
    const s_t weak_mat_tmp146 = gu7*weak_mat_tmp145;
    const s_t weak_mat_tmp147 = weak_mat_tmp14*weak_mat_tmp143 - weak_mat_tmp146;
    const s_t weak_mat_tmp148 = ((s_t(5) / s_t(3)))*gu5*weak_mat_tmp7 - (s_t(5) / s_t(3))*weak_mat_tmp136;
    const s_t weak_mat_tmp149 = ((s_t(2) / s_t(3)))*gu5*weak_mat_tmp7 - (s_t(2) / s_t(3))*weak_mat_tmp136;
    const s_t weak_mat_tmp150 = gu5*weak_mat_tmp132;
    const s_t weak_mat_tmp151 = weak_mat_tmp124*weak_mat_tmp20 + weak_mat_tmp149*weak_mat_tmp72 + weak_mat_tmp150;
    const s_t weak_mat_tmp152 = ((s_t(7) / s_t(3)))*gu5*weak_mat_tmp7 - (s_t(7) / s_t(3))*weak_mat_tmp136;
    const s_t weak_mat_tmp153 = ((s_t(4) / s_t(3)))*gu5*weak_mat_tmp7 - (s_t(4) / s_t(3))*weak_mat_tmp136;
    const s_t weak_mat_tmp154 = s_t(2)*gu7*weak_mat_tmp34 - weak_mat_tmp117*weak_mat_tmp49 - weak_mat_tmp124*weak_mat_tmp29 - weak_mat_tmp125*weak_mat_tmp62;
    const s_t weak_mat_tmp155 = gu5*weak_mat_tmp139;
    const s_t weak_mat_tmp156 = weak_mat_tmp153*weak_mat_tmp57 + weak_mat_tmp154*weak_mat_tmp79 + weak_mat_tmp155 + weak_mat_tmp43*(s_t(4)*gu7*weak_mat_tmp7 - s_t(2)*weak_mat_tmp128);
    const s_t weak_mat_tmp157 = gu5*weak_mat_tmp7;
    const s_t weak_mat_tmp158 = weak_mat_tmp136 - weak_mat_tmp157;
    const s_t weak_mat_tmp159 = -weak_mat_tmp158;
    const s_t weak_mat_tmp160 = gu5*weak_mat_tmp145;
    const s_t weak_mat_tmp161 = weak_mat_tmp14*weak_mat_tmp158 - weak_mat_tmp160;
    const s_t weak_mat_tmp162 = weak_mat_tmp2*weak_mat_tmp7;
    const s_t weak_mat_tmp163 = -(s_t(5) / s_t(3))*weak_mat_tmp162 + ((s_t(5) / s_t(3)))*weak_mat_tmp6;
    const s_t weak_mat_tmp164 = -(s_t(2) / s_t(3))*weak_mat_tmp162 + ((s_t(2) / s_t(3)))*weak_mat_tmp6;
    const s_t weak_mat_tmp165 = weak_mat_tmp17*weak_mat_tmp2;
    const s_t weak_mat_tmp166 = weak_mat_tmp131*weak_mat_tmp165;
    const s_t weak_mat_tmp167 = weak_mat_tmp108*weak_mat_tmp20 + weak_mat_tmp164*weak_mat_tmp72 - weak_mat_tmp166;
    const s_t weak_mat_tmp168 = -(s_t(7) / s_t(3))*weak_mat_tmp162 + ((s_t(7) / s_t(3)))*weak_mat_tmp6;
    const s_t weak_mat_tmp169 = -(s_t(4) / s_t(3))*weak_mat_tmp162 + ((s_t(4) / s_t(3)))*weak_mat_tmp6;
    const s_t weak_mat_tmp170 = s_t(2)*weak_mat_tmp1*weak_mat_tmp34 - weak_mat_tmp100*weak_mat_tmp49 - weak_mat_tmp107*weak_mat_tmp62 - weak_mat_tmp108*weak_mat_tmp29;
    const s_t weak_mat_tmp171 = weak_mat_tmp139*weak_mat_tmp2;
    const s_t weak_mat_tmp172 = weak_mat_tmp169*weak_mat_tmp57 + weak_mat_tmp170*weak_mat_tmp79 - weak_mat_tmp171 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp1*weak_mat_tmp7 - s_t(2)*weak_mat_tmp5);
    const s_t weak_mat_tmp173 = -weak_mat_tmp2*weak_mat_tmp7 + weak_mat_tmp6;
    const s_t weak_mat_tmp174 = weak_mat_tmp145*weak_mat_tmp2;
    const s_t weak_mat_tmp175 = -weak_mat_tmp173;
    const s_t weak_mat_tmp176 = weak_mat_tmp14*weak_mat_tmp175 + weak_mat_tmp174;
    const s_t weak_mat_tmp177 = weak_mat_tmp1*weak_mat_tmp7;
    const s_t weak_mat_tmp178 = -(s_t(5) / s_t(3))*weak_mat_tmp177 + ((s_t(5) / s_t(3)))*weak_mat_tmp5;
    const s_t weak_mat_tmp179 = -(s_t(2) / s_t(3))*weak_mat_tmp177 + ((s_t(2) / s_t(3)))*weak_mat_tmp5;
    const s_t weak_mat_tmp180 = weak_mat_tmp1*weak_mat_tmp132;
    const s_t weak_mat_tmp181 = weak_mat_tmp125*weak_mat_tmp20 + weak_mat_tmp179*weak_mat_tmp72 - weak_mat_tmp180;
    const s_t weak_mat_tmp182 = -(s_t(7) / s_t(3))*weak_mat_tmp177 + ((s_t(7) / s_t(3)))*weak_mat_tmp5;
    const s_t weak_mat_tmp183 = -(s_t(4) / s_t(3))*weak_mat_tmp177 + ((s_t(4) / s_t(3)))*weak_mat_tmp5;
    const s_t weak_mat_tmp184 = -weak_mat_tmp117*weak_mat_tmp54 - weak_mat_tmp124*weak_mat_tmp62 - weak_mat_tmp125*weak_mat_tmp33 + s_t(2)*weak_mat_tmp2*weak_mat_tmp34;
    const s_t weak_mat_tmp185 = weak_mat_tmp1*weak_mat_tmp139;
    const s_t weak_mat_tmp186 = weak_mat_tmp183*weak_mat_tmp57 + weak_mat_tmp184*weak_mat_tmp79 - weak_mat_tmp185 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp2*weak_mat_tmp7 - s_t(2)*weak_mat_tmp6);
    const s_t weak_mat_tmp187 = -weak_mat_tmp1*weak_mat_tmp7 + weak_mat_tmp5;
    const s_t weak_mat_tmp188 = weak_mat_tmp1*weak_mat_tmp145;
    const s_t weak_mat_tmp189 = -weak_mat_tmp187;
    const s_t weak_mat_tmp190 = weak_mat_tmp14*weak_mat_tmp189 + weak_mat_tmp188;
    const s_t weak_mat_tmp191 = weak_mat_tmp68*weak_mat_tmp9;
    const s_t weak_mat_tmp192 = weak_mat_tmp13*weak_mat_tmp191;
    const s_t weak_mat_tmp193 = weak_mat_tmp17*weak_mat_tmp71;
    const s_t weak_mat_tmp194 = weak_mat_tmp35*weak_mat_tmp71;
    const s_t weak_mat_tmp195 = s_t(2)*weak_mat_tmp22;
    const s_t weak_mat_tmp196 = s_t(2)*weak_mat_tmp23;
    const s_t weak_mat_tmp197 = weak_mat_tmp195 + weak_mat_tmp196;
    const s_t weak_mat_tmp198 = weak_mat_tmp45*weak_mat_tmp78;
    const s_t weak_mat_tmp199 = weak_mat_tmp64*weak_mat_tmp77;
    const s_t weak_mat_tmp200 = weak_mat_tmp191*weak_mat_tmp82;
    const s_t weak_mat_tmp201 = weak_mat_tmp17*weak_mat_tmp50;
    const s_t weak_mat_tmp202 = weak_mat_tmp193*weak_mat_tmp55 + weak_mat_tmp201*weak_mat_tmp86;
    const s_t weak_mat_tmp203 = s_t(2)*weak_mat_tmp60;
    const s_t weak_mat_tmp204 = s_t(2)*weak_mat_tmp61;
    const s_t weak_mat_tmp205 = weak_mat_tmp45*weak_mat_tmp77;
    const s_t weak_mat_tmp206 = weak_mat_tmp198*weak_mat_tmp91 + weak_mat_tmp205*weak_mat_tmp92 + weak_mat_tmp43*(-weak_mat_tmp203 - weak_mat_tmp204);
    const s_t weak_mat_tmp207 = weak_mat_tmp158*weak_mat_tmp191;
    const s_t weak_mat_tmp208 = weak_mat_tmp124*weak_mat_tmp193 + weak_mat_tmp149*weak_mat_tmp201;
    const s_t weak_mat_tmp209 = gu6*weak_mat_tmp56;
    const s_t weak_mat_tmp210 = weak_mat_tmp153*weak_mat_tmp198 + weak_mat_tmp154*weak_mat_tmp205 + weak_mat_tmp43*(-weak_mat_tmp122 - weak_mat_tmp209);
    const s_t weak_mat_tmp211 = weak_mat_tmp175*weak_mat_tmp191;
    const s_t weak_mat_tmp212 = weak_mat_tmp108*weak_mat_tmp193 + weak_mat_tmp164*weak_mat_tmp201;
    const s_t weak_mat_tmp213 = gu3*weak_mat_tmp56;
    const s_t weak_mat_tmp214 = weak_mat_tmp169*weak_mat_tmp198 + weak_mat_tmp170*weak_mat_tmp205 + weak_mat_tmp43*(-weak_mat_tmp104 - weak_mat_tmp213);
    const s_t weak_mat_tmp215 = gu6*weak_mat_tmp132;
    const s_t weak_mat_tmp216 = weak_mat_tmp107*weak_mat_tmp193 + weak_mat_tmp130*weak_mat_tmp201 - weak_mat_tmp215;
    const s_t weak_mat_tmp217 = gu6*weak_mat_tmp139;
    const s_t weak_mat_tmp218 = weak_mat_tmp137*weak_mat_tmp198 + weak_mat_tmp138*weak_mat_tmp205 - weak_mat_tmp217 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp111 - s_t(2)*weak_mat_tmp112);
    const s_t weak_mat_tmp219 = gu6*weak_mat_tmp145;
    const s_t weak_mat_tmp220 = weak_mat_tmp143*weak_mat_tmp191 + weak_mat_tmp219;
    const s_t weak_mat_tmp221 = weak_mat_tmp117*weak_mat_tmp193 + weak_mat_tmp118*weak_mat_tmp201 - weak_mat_tmp150;
    const s_t weak_mat_tmp222 = weak_mat_tmp123*weak_mat_tmp198 + weak_mat_tmp126*weak_mat_tmp205 - weak_mat_tmp155 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp128 - s_t(2)*weak_mat_tmp142);
    const s_t weak_mat_tmp223 = weak_mat_tmp113*weak_mat_tmp191 + weak_mat_tmp160;
    const s_t weak_mat_tmp224 = weak_mat_tmp100*weak_mat_tmp193 + weak_mat_tmp101*weak_mat_tmp201 + weak_mat_tmp166;
    const s_t weak_mat_tmp225 = weak_mat_tmp106*weak_mat_tmp198 + weak_mat_tmp109*weak_mat_tmp205 + weak_mat_tmp171 + weak_mat_tmp43*(-s_t(2)*weak_mat_tmp177 + s_t(4)*weak_mat_tmp5);
    const s_t weak_mat_tmp226 = -weak_mat_tmp174 + weak_mat_tmp191*weak_mat_tmp96;
    const s_t weak_mat_tmp227 = gu3*weak_mat_tmp132;
    const s_t weak_mat_tmp228 = weak_mat_tmp125*weak_mat_tmp193 + weak_mat_tmp179*weak_mat_tmp201 + weak_mat_tmp227;
    const s_t weak_mat_tmp229 = gu3*weak_mat_tmp139;
    const s_t weak_mat_tmp230 = weak_mat_tmp183*weak_mat_tmp198 + weak_mat_tmp184*weak_mat_tmp205 + weak_mat_tmp229 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp94 - s_t(2)*weak_mat_tmp98);
    const s_t weak_mat_tmp231 = gu3*weak_mat_tmp145;
    const s_t weak_mat_tmp232 = weak_mat_tmp189*weak_mat_tmp191 - weak_mat_tmp231;
    const s_t weak_mat_tmp233 = weak_mat_tmp82*weak_mat_tmp9;
    const s_t weak_mat_tmp234 = weak_mat_tmp13*weak_mat_tmp233;
    const s_t weak_mat_tmp235 = weak_mat_tmp17*weak_mat_tmp86;
    const s_t weak_mat_tmp236 = weak_mat_tmp35*weak_mat_tmp86;
    const s_t weak_mat_tmp237 = weak_mat_tmp45*weak_mat_tmp92;
    const s_t weak_mat_tmp238 = weak_mat_tmp64*weak_mat_tmp91;
    const s_t weak_mat_tmp239 = weak_mat_tmp143*weak_mat_tmp233;
    const s_t weak_mat_tmp240 = weak_mat_tmp17*weak_mat_tmp55;
    const s_t weak_mat_tmp241 = weak_mat_tmp107*weak_mat_tmp235 + weak_mat_tmp130*weak_mat_tmp240;
    const s_t weak_mat_tmp242 = weak_mat_tmp45*weak_mat_tmp91;
    const s_t weak_mat_tmp243 = weak_mat_tmp137*weak_mat_tmp237 + weak_mat_tmp138*weak_mat_tmp242 + weak_mat_tmp43*(-weak_mat_tmp105 - weak_mat_tmp213);
    const s_t weak_mat_tmp244 = weak_mat_tmp189*weak_mat_tmp233;
    const s_t weak_mat_tmp245 = weak_mat_tmp125*weak_mat_tmp235 + weak_mat_tmp179*weak_mat_tmp240;
    const s_t weak_mat_tmp246 = weak_mat_tmp183*weak_mat_tmp237 + weak_mat_tmp184*weak_mat_tmp242 + weak_mat_tmp43*(-weak_mat_tmp121 - weak_mat_tmp209);
    const s_t weak_mat_tmp247 = weak_mat_tmp100*weak_mat_tmp235 + weak_mat_tmp101*weak_mat_tmp240 - weak_mat_tmp133;
    const s_t weak_mat_tmp248 = weak_mat_tmp106*weak_mat_tmp237 + weak_mat_tmp109*weak_mat_tmp242 - weak_mat_tmp140 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp136 - s_t(2)*weak_mat_tmp157);
    const s_t weak_mat_tmp249 = weak_mat_tmp146 + weak_mat_tmp233*weak_mat_tmp96;
    const s_t weak_mat_tmp250 = weak_mat_tmp124*weak_mat_tmp235 + weak_mat_tmp149*weak_mat_tmp240 - weak_mat_tmp227;
    const s_t weak_mat_tmp251 = weak_mat_tmp153*weak_mat_tmp237 + weak_mat_tmp154*weak_mat_tmp242 - weak_mat_tmp229 + weak_mat_tmp43*(s_t(4)*gu2*gu7 - s_t(2)*weak_mat_tmp94);
    const s_t weak_mat_tmp252 = weak_mat_tmp158*weak_mat_tmp233 + weak_mat_tmp231;
    const s_t weak_mat_tmp253 = weak_mat_tmp117*weak_mat_tmp235 + weak_mat_tmp118*weak_mat_tmp240 + weak_mat_tmp180;
    const s_t weak_mat_tmp254 = weak_mat_tmp123*weak_mat_tmp237 + weak_mat_tmp126*weak_mat_tmp242 + weak_mat_tmp185 + weak_mat_tmp43*(-s_t(2)*weak_mat_tmp162 + s_t(4)*weak_mat_tmp6);
    const s_t weak_mat_tmp255 = weak_mat_tmp113*weak_mat_tmp233 - weak_mat_tmp188;
    const s_t weak_mat_tmp256 = weak_mat_tmp108*weak_mat_tmp235 + weak_mat_tmp164*weak_mat_tmp240 + weak_mat_tmp215;
    const s_t weak_mat_tmp257 = weak_mat_tmp169*weak_mat_tmp237 + weak_mat_tmp170*weak_mat_tmp242 + weak_mat_tmp217 + weak_mat_tmp43*(s_t(4)*gu2*weak_mat_tmp1 - s_t(2)*weak_mat_tmp111);
    const s_t weak_mat_tmp258 = weak_mat_tmp175*weak_mat_tmp233 - weak_mat_tmp219;
    const s_t weak_mat_tmp259 = weak_mat_tmp9*weak_mat_tmp96;
    const s_t weak_mat_tmp260 = weak_mat_tmp13*weak_mat_tmp259;
    const s_t weak_mat_tmp261 = weak_mat_tmp101*weak_mat_tmp17;
    const s_t weak_mat_tmp262 = weak_mat_tmp101*weak_mat_tmp35;
    const s_t weak_mat_tmp263 = s_t(2)*weak_mat_tmp30;
    const s_t weak_mat_tmp264 = weak_mat_tmp263 + weak_mat_tmp38;
    const s_t weak_mat_tmp265 = s_t(2)*weak_mat_tmp26;
    const s_t weak_mat_tmp266 = weak_mat_tmp265 + weak_mat_tmp40;
    const s_t weak_mat_tmp267 = weak_mat_tmp106*weak_mat_tmp45;
    const s_t weak_mat_tmp268 = weak_mat_tmp106*weak_mat_tmp64;
    const s_t weak_mat_tmp269 = weak_mat_tmp143*weak_mat_tmp259;
    const s_t weak_mat_tmp270 = weak_mat_tmp100*weak_mat_tmp17;
    const s_t weak_mat_tmp271 = weak_mat_tmp107*weak_mat_tmp261 + weak_mat_tmp130*weak_mat_tmp270;
    const s_t weak_mat_tmp272 = s_t(2)*weak_mat_tmp52;
    const s_t weak_mat_tmp273 = weak_mat_tmp109*weak_mat_tmp45;
    const s_t weak_mat_tmp274 = weak_mat_tmp137*weak_mat_tmp273 + weak_mat_tmp138*weak_mat_tmp267 + weak_mat_tmp43*(-weak_mat_tmp272 - weak_mat_tmp90);
    const s_t weak_mat_tmp275 = weak_mat_tmp113*weak_mat_tmp259;
    const s_t weak_mat_tmp276 = weak_mat_tmp117*weak_mat_tmp261 + weak_mat_tmp118*weak_mat_tmp270;
    const s_t weak_mat_tmp277 = weak_mat_tmp107*weak_mat_tmp2;
    const s_t weak_mat_tmp278 = gu7*weak_mat_tmp108;
    const s_t weak_mat_tmp279 = weak_mat_tmp123*weak_mat_tmp273 + weak_mat_tmp126*weak_mat_tmp267 + weak_mat_tmp43*(-weak_mat_tmp277 - weak_mat_tmp278);
    const s_t weak_mat_tmp280 = weak_mat_tmp175*weak_mat_tmp259;
    const s_t weak_mat_tmp281 = weak_mat_tmp108*weak_mat_tmp261 + weak_mat_tmp164*weak_mat_tmp270;
    const s_t weak_mat_tmp282 = s_t(2)*weak_mat_tmp47;
    const s_t weak_mat_tmp283 = weak_mat_tmp169*weak_mat_tmp273 + weak_mat_tmp170*weak_mat_tmp267 + weak_mat_tmp43*(-weak_mat_tmp282 - weak_mat_tmp75);
    const s_t weak_mat_tmp284 = gu2*weak_mat_tmp132;
    const s_t weak_mat_tmp285 = weak_mat_tmp124*weak_mat_tmp261 + weak_mat_tmp149*weak_mat_tmp270 - weak_mat_tmp284;
    const s_t weak_mat_tmp286 = gu2*weak_mat_tmp139;
    const s_t weak_mat_tmp287 = weak_mat_tmp153*weak_mat_tmp273 + weak_mat_tmp154*weak_mat_tmp267 - weak_mat_tmp286 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp12 - s_t(2)*weak_mat_tmp81);
    const s_t weak_mat_tmp288 = gu2*weak_mat_tmp145;
    const s_t weak_mat_tmp289 = weak_mat_tmp158*weak_mat_tmp259 + weak_mat_tmp288;
    const s_t weak_mat_tmp290 = gu1*weak_mat_tmp132;
    const s_t weak_mat_tmp291 = weak_mat_tmp125*weak_mat_tmp261 + weak_mat_tmp179*weak_mat_tmp270 + weak_mat_tmp290;
    const s_t weak_mat_tmp292 = gu1*weak_mat_tmp139;
    const s_t weak_mat_tmp293 = weak_mat_tmp183*weak_mat_tmp273 + weak_mat_tmp184*weak_mat_tmp267 + weak_mat_tmp292 + weak_mat_tmp43*(-s_t(2)*weak_mat_tmp11 + s_t(4)*weak_mat_tmp66);
    const s_t weak_mat_tmp294 = gu1*weak_mat_tmp145;
    const s_t weak_mat_tmp295 = weak_mat_tmp189*weak_mat_tmp259 - weak_mat_tmp294;
    const s_t weak_mat_tmp296 = weak_mat_tmp175*weak_mat_tmp9;
    const s_t weak_mat_tmp297 = weak_mat_tmp13*weak_mat_tmp296;
    const s_t weak_mat_tmp298 = weak_mat_tmp164*weak_mat_tmp17;
    const s_t weak_mat_tmp299 = weak_mat_tmp164*weak_mat_tmp35;
    const s_t weak_mat_tmp300 = s_t(2)*weak_mat_tmp24;
    const s_t weak_mat_tmp301 = weak_mat_tmp196 + weak_mat_tmp300;
    const s_t weak_mat_tmp302 = weak_mat_tmp170*weak_mat_tmp45;
    const s_t weak_mat_tmp303 = weak_mat_tmp169*weak_mat_tmp64;
    const s_t weak_mat_tmp304 = weak_mat_tmp143*weak_mat_tmp296;
    const s_t weak_mat_tmp305 = weak_mat_tmp108*weak_mat_tmp17;
    const s_t weak_mat_tmp306 = weak_mat_tmp107*weak_mat_tmp298 + weak_mat_tmp130*weak_mat_tmp305;
    const s_t weak_mat_tmp307 = s_t(2)*weak_mat_tmp59;
    const s_t weak_mat_tmp308 = weak_mat_tmp169*weak_mat_tmp45;
    const s_t weak_mat_tmp309 = weak_mat_tmp137*weak_mat_tmp302 + weak_mat_tmp138*weak_mat_tmp308 + weak_mat_tmp43*(-weak_mat_tmp204 - weak_mat_tmp307);
    const s_t weak_mat_tmp310 = weak_mat_tmp158*weak_mat_tmp296;
    const s_t weak_mat_tmp311 = weak_mat_tmp124*weak_mat_tmp298 + weak_mat_tmp149*weak_mat_tmp305;
    const s_t weak_mat_tmp312 = gu6*weak_mat_tmp100;
    const s_t weak_mat_tmp313 = weak_mat_tmp153*weak_mat_tmp302 + weak_mat_tmp154*weak_mat_tmp308 + weak_mat_tmp43*(-weak_mat_tmp277 - weak_mat_tmp312);
    const s_t weak_mat_tmp314 = weak_mat_tmp117*weak_mat_tmp298 + weak_mat_tmp118*weak_mat_tmp305 + weak_mat_tmp284;
    const s_t weak_mat_tmp315 = weak_mat_tmp123*weak_mat_tmp302 + weak_mat_tmp126*weak_mat_tmp308 + weak_mat_tmp286 + weak_mat_tmp43*(s_t(4)*gu6*weak_mat_tmp1 - s_t(2)*weak_mat_tmp12);
    const s_t weak_mat_tmp316 = weak_mat_tmp113*weak_mat_tmp296 - weak_mat_tmp288;
    const s_t weak_mat_tmp317 = weak_mat_tmp132*weak_mat_tmp7;
    const s_t weak_mat_tmp318 = weak_mat_tmp125*weak_mat_tmp298 + weak_mat_tmp179*weak_mat_tmp305 - weak_mat_tmp317;
    const s_t weak_mat_tmp319 = weak_mat_tmp139*weak_mat_tmp7;
    const s_t weak_mat_tmp320 = weak_mat_tmp183*weak_mat_tmp302 + weak_mat_tmp184*weak_mat_tmp308 - weak_mat_tmp319 + weak_mat_tmp43*(-s_t(2)*weak_mat_tmp0 + s_t(4)*weak_mat_tmp1*weak_mat_tmp2);
    const s_t weak_mat_tmp321 = weak_mat_tmp145*weak_mat_tmp7;
    const s_t weak_mat_tmp322 = weak_mat_tmp189*weak_mat_tmp296 + weak_mat_tmp321;
    const s_t weak_mat_tmp323 = weak_mat_tmp143*weak_mat_tmp9;
    const s_t weak_mat_tmp324 = weak_mat_tmp13*weak_mat_tmp323;
    const s_t weak_mat_tmp325 = weak_mat_tmp130*weak_mat_tmp17;
    const s_t weak_mat_tmp326 = weak_mat_tmp130*weak_mat_tmp35;
    const s_t weak_mat_tmp327 = weak_mat_tmp138*weak_mat_tmp45;
    const s_t weak_mat_tmp328 = weak_mat_tmp137*weak_mat_tmp64;
    const s_t weak_mat_tmp329 = weak_mat_tmp189*weak_mat_tmp323;
    const s_t weak_mat_tmp330 = weak_mat_tmp107*weak_mat_tmp17;
    const s_t weak_mat_tmp331 = weak_mat_tmp125*weak_mat_tmp325 + weak_mat_tmp179*weak_mat_tmp330;
    const s_t weak_mat_tmp332 = weak_mat_tmp137*weak_mat_tmp45;
    const s_t weak_mat_tmp333 = weak_mat_tmp183*weak_mat_tmp327 + weak_mat_tmp184*weak_mat_tmp332 + weak_mat_tmp43*(-weak_mat_tmp278 - weak_mat_tmp312);
    const s_t weak_mat_tmp334 = weak_mat_tmp117*weak_mat_tmp325 + weak_mat_tmp118*weak_mat_tmp330 - weak_mat_tmp290;
    const s_t weak_mat_tmp335 = weak_mat_tmp123*weak_mat_tmp327 + weak_mat_tmp126*weak_mat_tmp332 - weak_mat_tmp292 + weak_mat_tmp43*(s_t(4)*gu5*gu6 - s_t(2)*weak_mat_tmp66);
    const s_t weak_mat_tmp336 = weak_mat_tmp113*weak_mat_tmp323 + weak_mat_tmp294;
    const s_t weak_mat_tmp337 = weak_mat_tmp124*weak_mat_tmp325 + weak_mat_tmp149*weak_mat_tmp330 + weak_mat_tmp317;
    const s_t weak_mat_tmp338 = weak_mat_tmp153*weak_mat_tmp327 + weak_mat_tmp154*weak_mat_tmp332 + weak_mat_tmp319 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp0 - s_t(2)*weak_mat_tmp18);
    const s_t weak_mat_tmp339 = weak_mat_tmp158*weak_mat_tmp323 - weak_mat_tmp321;
    const s_t weak_mat_tmp340 = weak_mat_tmp113*weak_mat_tmp9;
    const s_t weak_mat_tmp341 = weak_mat_tmp13*weak_mat_tmp340;
    const s_t weak_mat_tmp342 = weak_mat_tmp118*weak_mat_tmp17;
    const s_t weak_mat_tmp343 = weak_mat_tmp118*weak_mat_tmp35;
    const s_t weak_mat_tmp344 = weak_mat_tmp263 + weak_mat_tmp37;
    const s_t weak_mat_tmp345 = weak_mat_tmp265 + weak_mat_tmp41;
    const s_t weak_mat_tmp346 = weak_mat_tmp123*weak_mat_tmp45;
    const s_t weak_mat_tmp347 = weak_mat_tmp123*weak_mat_tmp64;
    const s_t weak_mat_tmp348 = weak_mat_tmp158*weak_mat_tmp340;
    const s_t weak_mat_tmp349 = weak_mat_tmp117*weak_mat_tmp17;
    const s_t weak_mat_tmp350 = weak_mat_tmp124*weak_mat_tmp342 + weak_mat_tmp149*weak_mat_tmp349;
    const s_t weak_mat_tmp351 = weak_mat_tmp126*weak_mat_tmp45;
    const s_t weak_mat_tmp352 = weak_mat_tmp153*weak_mat_tmp351 + weak_mat_tmp154*weak_mat_tmp346 + weak_mat_tmp43*(-weak_mat_tmp282 - weak_mat_tmp76);
    const s_t weak_mat_tmp353 = weak_mat_tmp189*weak_mat_tmp340;
    const s_t weak_mat_tmp354 = weak_mat_tmp125*weak_mat_tmp342 + weak_mat_tmp179*weak_mat_tmp349;
    const s_t weak_mat_tmp355 = weak_mat_tmp183*weak_mat_tmp351 + weak_mat_tmp184*weak_mat_tmp346 + weak_mat_tmp43*(-weak_mat_tmp272 - weak_mat_tmp89);
    const s_t weak_mat_tmp356 = weak_mat_tmp158*weak_mat_tmp9;
    const s_t weak_mat_tmp357 = weak_mat_tmp13*weak_mat_tmp356;
    const s_t weak_mat_tmp358 = weak_mat_tmp149*weak_mat_tmp17;
    const s_t weak_mat_tmp359 = weak_mat_tmp149*weak_mat_tmp35;
    const s_t weak_mat_tmp360 = weak_mat_tmp195 + weak_mat_tmp300;
    const s_t weak_mat_tmp361 = weak_mat_tmp153*weak_mat_tmp45;
    const s_t weak_mat_tmp362 = weak_mat_tmp153*weak_mat_tmp64;
    const s_t weak_mat_tmp363 = weak_mat_tmp189*weak_mat_tmp356;
    const s_t weak_mat_tmp364 = weak_mat_tmp124*weak_mat_tmp17*weak_mat_tmp179 + weak_mat_tmp125*weak_mat_tmp358;
    const s_t weak_mat_tmp365 = weak_mat_tmp183*weak_mat_tmp45;
    const s_t weak_mat_tmp366 = weak_mat_tmp154*weak_mat_tmp365 + weak_mat_tmp184*weak_mat_tmp361 + weak_mat_tmp43*(-weak_mat_tmp203 - weak_mat_tmp307);
    const s_t weak_mat_tmp367 = weak_mat_tmp13*weak_mat_tmp189*weak_mat_tmp9;
    const s_t weak_mat_tmp368 = weak_mat_tmp179*weak_mat_tmp35;
    const s_t weak_mat_tmp369 = weak_mat_tmp183*weak_mat_tmp64;
    const s_t material0 = trial_grad0*(c1*(weak_mat_tmp16 + s_t(4)*weak_mat_tmp20*weak_mat_tmp7 + weak_mat_tmp21*weak_mat_tmp36) + c2*(weak_mat_tmp43*(weak_mat_tmp39 + weak_mat_tmp42) + s_t(2)*weak_mat_tmp44*weak_mat_tmp57 + weak_mat_tmp58*weak_mat_tmp65) + weak_mat_tmp15*weak_mat_tmp3 + pow_2(weak_mat_tmp4)*weak_mat_tmp9) + trial_grad1*(c1*(weak_mat_tmp36*weak_mat_tmp70 + weak_mat_tmp73) + c2*(weak_mat_tmp65*weak_mat_tmp74 + weak_mat_tmp80) + weak_mat_tmp15*weak_mat_tmp67 + weak_mat_tmp69) + trial_grad2*(c1*(weak_mat_tmp36*weak_mat_tmp85 + weak_mat_tmp87) + c2*(weak_mat_tmp65*weak_mat_tmp88 + weak_mat_tmp93) + weak_mat_tmp15*weak_mat_tmp84 + weak_mat_tmp83) + trial_grad3*(c1*(weak_mat_tmp102 + weak_mat_tmp36*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp65 + weak_mat_tmp110) + weak_mat_tmp15*weak_mat_tmp95 + weak_mat_tmp97) + trial_grad4*(c1*(weak_mat_tmp163*weak_mat_tmp36 + weak_mat_tmp167) + c2*(weak_mat_tmp168*weak_mat_tmp65 + weak_mat_tmp172) + weak_mat_tmp15*weak_mat_tmp173 + weak_mat_tmp176) + trial_grad5*(c1*(weak_mat_tmp129*weak_mat_tmp36 + weak_mat_tmp134) + c2*(weak_mat_tmp135*weak_mat_tmp65 + weak_mat_tmp141) + weak_mat_tmp144*weak_mat_tmp15 + weak_mat_tmp147) + trial_grad6*(c1*(weak_mat_tmp116*weak_mat_tmp36 + weak_mat_tmp119) + c2*(weak_mat_tmp120*weak_mat_tmp65 + weak_mat_tmp127) + weak_mat_tmp114 + weak_mat_tmp115*weak_mat_tmp15) + trial_grad7*(c1*(weak_mat_tmp148*weak_mat_tmp36 + weak_mat_tmp151) + c2*(weak_mat_tmp152*weak_mat_tmp65 + weak_mat_tmp156) + weak_mat_tmp15*weak_mat_tmp159 + weak_mat_tmp161) + trial_grad8*(c1*(weak_mat_tmp178*weak_mat_tmp36 + weak_mat_tmp181) + c2*(weak_mat_tmp182*weak_mat_tmp65 + weak_mat_tmp186) + weak_mat_tmp15*weak_mat_tmp187 + weak_mat_tmp190);
    const s_t material1 = trial_grad0*(c1*(weak_mat_tmp194*weak_mat_tmp21 + weak_mat_tmp73) + c2*(weak_mat_tmp199*weak_mat_tmp58 + weak_mat_tmp80) + weak_mat_tmp192*weak_mat_tmp3 + weak_mat_tmp69) + trial_grad1*(c1*(s_t(4)*gu1*weak_mat_tmp193 + weak_mat_tmp16 + weak_mat_tmp194*weak_mat_tmp70) + c2*(s_t(2)*weak_mat_tmp198*weak_mat_tmp77 + weak_mat_tmp199*weak_mat_tmp74 + weak_mat_tmp43*(weak_mat_tmp197 + weak_mat_tmp39)) + weak_mat_tmp192*weak_mat_tmp67 + pow_2(weak_mat_tmp68)*weak_mat_tmp9) + trial_grad2*(c1*(weak_mat_tmp194*weak_mat_tmp85 + weak_mat_tmp202) + c2*(weak_mat_tmp199*weak_mat_tmp88 + weak_mat_tmp206) + weak_mat_tmp192*weak_mat_tmp84 + weak_mat_tmp200) + trial_grad3*(c1*(weak_mat_tmp194*weak_mat_tmp99 + weak_mat_tmp224) + c2*(weak_mat_tmp103*weak_mat_tmp199 + weak_mat_tmp225) + weak_mat_tmp192*weak_mat_tmp95 + weak_mat_tmp226) + trial_grad4*(c1*(weak_mat_tmp163*weak_mat_tmp194 + weak_mat_tmp212) + c2*(weak_mat_tmp168*weak_mat_tmp199 + weak_mat_tmp214) + weak_mat_tmp173*weak_mat_tmp192 + weak_mat_tmp211) + trial_grad5*(c1*(weak_mat_tmp129*weak_mat_tmp194 + weak_mat_tmp216) + c2*(weak_mat_tmp135*weak_mat_tmp199 + weak_mat_tmp218) + weak_mat_tmp144*weak_mat_tmp192 + weak_mat_tmp220) + trial_grad6*(c1*(weak_mat_tmp116*weak_mat_tmp194 + weak_mat_tmp221) + c2*(weak_mat_tmp120*weak_mat_tmp199 + weak_mat_tmp222) + weak_mat_tmp115*weak_mat_tmp192 + weak_mat_tmp223) + trial_grad7*(c1*(weak_mat_tmp148*weak_mat_tmp194 + weak_mat_tmp208) + c2*(weak_mat_tmp152*weak_mat_tmp199 + weak_mat_tmp210) + weak_mat_tmp159*weak_mat_tmp192 + weak_mat_tmp207) + trial_grad8*(c1*(weak_mat_tmp178*weak_mat_tmp194 + weak_mat_tmp228) + c2*(weak_mat_tmp182*weak_mat_tmp199 + weak_mat_tmp230) + weak_mat_tmp187*weak_mat_tmp192 + weak_mat_tmp232);
    const s_t material2 = trial_grad0*(c1*(weak_mat_tmp21*weak_mat_tmp236 + weak_mat_tmp87) + c2*(weak_mat_tmp238*weak_mat_tmp58 + weak_mat_tmp93) + weak_mat_tmp234*weak_mat_tmp3 + weak_mat_tmp83) + trial_grad1*(c1*(weak_mat_tmp202 + weak_mat_tmp236*weak_mat_tmp70) + c2*(weak_mat_tmp206 + weak_mat_tmp238*weak_mat_tmp74) + weak_mat_tmp200 + weak_mat_tmp234*weak_mat_tmp67) + trial_grad2*(c1*(s_t(4)*gu2*weak_mat_tmp235 + weak_mat_tmp16 + weak_mat_tmp236*weak_mat_tmp85) + c2*(s_t(2)*weak_mat_tmp237*weak_mat_tmp91 + weak_mat_tmp238*weak_mat_tmp88 + weak_mat_tmp43*(weak_mat_tmp197 + weak_mat_tmp42)) + weak_mat_tmp234*weak_mat_tmp84 + pow_2(weak_mat_tmp82)*weak_mat_tmp9) + trial_grad3*(c1*(weak_mat_tmp236*weak_mat_tmp99 + weak_mat_tmp247) + c2*(weak_mat_tmp103*weak_mat_tmp238 + weak_mat_tmp248) + weak_mat_tmp234*weak_mat_tmp95 + weak_mat_tmp249) + trial_grad4*(c1*(weak_mat_tmp163*weak_mat_tmp236 + weak_mat_tmp256) + c2*(weak_mat_tmp168*weak_mat_tmp238 + weak_mat_tmp257) + weak_mat_tmp173*weak_mat_tmp234 + weak_mat_tmp258) + trial_grad5*(c1*(weak_mat_tmp129*weak_mat_tmp236 + weak_mat_tmp241) + c2*(weak_mat_tmp135*weak_mat_tmp238 + weak_mat_tmp243) + weak_mat_tmp144*weak_mat_tmp234 + weak_mat_tmp239) + trial_grad6*(c1*(weak_mat_tmp116*weak_mat_tmp236 + weak_mat_tmp253) + c2*(weak_mat_tmp120*weak_mat_tmp238 + weak_mat_tmp254) + weak_mat_tmp115*weak_mat_tmp234 + weak_mat_tmp255) + trial_grad7*(c1*(weak_mat_tmp148*weak_mat_tmp236 + weak_mat_tmp250) + c2*(weak_mat_tmp152*weak_mat_tmp238 + weak_mat_tmp251) + weak_mat_tmp159*weak_mat_tmp234 + weak_mat_tmp252) + trial_grad8*(c1*(weak_mat_tmp178*weak_mat_tmp236 + weak_mat_tmp245) + c2*(weak_mat_tmp182*weak_mat_tmp238 + weak_mat_tmp246) + weak_mat_tmp187*weak_mat_tmp234 + weak_mat_tmp244);
    const s_t material3 = trial_grad0*(c1*(weak_mat_tmp102 + weak_mat_tmp21*weak_mat_tmp262) + c2*(weak_mat_tmp110 + weak_mat_tmp268*weak_mat_tmp58) + weak_mat_tmp260*weak_mat_tmp3 + weak_mat_tmp97) + trial_grad1*(c1*(weak_mat_tmp224 + weak_mat_tmp262*weak_mat_tmp70) + c2*(weak_mat_tmp225 + weak_mat_tmp268*weak_mat_tmp74) + weak_mat_tmp226 + weak_mat_tmp260*weak_mat_tmp67) + trial_grad2*(c1*(weak_mat_tmp247 + weak_mat_tmp262*weak_mat_tmp85) + c2*(weak_mat_tmp248 + weak_mat_tmp268*weak_mat_tmp88) + weak_mat_tmp249 + weak_mat_tmp260*weak_mat_tmp84) + trial_grad3*(c1*(s_t(4)*gu3*weak_mat_tmp261 + weak_mat_tmp16 + weak_mat_tmp262*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp268 + s_t(2)*weak_mat_tmp109*weak_mat_tmp267 + weak_mat_tmp43*(weak_mat_tmp264 + weak_mat_tmp266)) + weak_mat_tmp260*weak_mat_tmp95 + weak_mat_tmp9*pow_2(weak_mat_tmp96)) + trial_grad4*(c1*(weak_mat_tmp163*weak_mat_tmp262 + weak_mat_tmp281) + c2*(weak_mat_tmp168*weak_mat_tmp268 + weak_mat_tmp283) + weak_mat_tmp173*weak_mat_tmp260 + weak_mat_tmp280) + trial_grad5*(c1*(weak_mat_tmp129*weak_mat_tmp262 + weak_mat_tmp271) + c2*(weak_mat_tmp135*weak_mat_tmp268 + weak_mat_tmp274) + weak_mat_tmp144*weak_mat_tmp260 + weak_mat_tmp269) + trial_grad6*(c1*(weak_mat_tmp116*weak_mat_tmp262 + weak_mat_tmp276) + c2*(weak_mat_tmp120*weak_mat_tmp268 + weak_mat_tmp279) + weak_mat_tmp115*weak_mat_tmp260 + weak_mat_tmp275) + trial_grad7*(c1*(weak_mat_tmp148*weak_mat_tmp262 + weak_mat_tmp285) + c2*(weak_mat_tmp152*weak_mat_tmp268 + weak_mat_tmp287) + weak_mat_tmp159*weak_mat_tmp260 + weak_mat_tmp289) + trial_grad8*(c1*(weak_mat_tmp178*weak_mat_tmp262 + weak_mat_tmp291) + c2*(weak_mat_tmp182*weak_mat_tmp268 + weak_mat_tmp293) + weak_mat_tmp187*weak_mat_tmp260 + weak_mat_tmp295);
    const s_t material4 = trial_grad0*(c1*(weak_mat_tmp167 + weak_mat_tmp21*weak_mat_tmp299) + c2*(weak_mat_tmp172 + weak_mat_tmp303*weak_mat_tmp58) + weak_mat_tmp176 + weak_mat_tmp297*weak_mat_tmp3) + trial_grad1*(c1*(weak_mat_tmp212 + weak_mat_tmp299*weak_mat_tmp70) + c2*(weak_mat_tmp214 + weak_mat_tmp303*weak_mat_tmp74) + weak_mat_tmp211 + weak_mat_tmp297*weak_mat_tmp67) + trial_grad2*(c1*(weak_mat_tmp256 + weak_mat_tmp299*weak_mat_tmp85) + c2*(weak_mat_tmp257 + weak_mat_tmp303*weak_mat_tmp88) + weak_mat_tmp258 + weak_mat_tmp297*weak_mat_tmp84) + trial_grad3*(c1*(weak_mat_tmp281 + weak_mat_tmp299*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp303 + weak_mat_tmp283) + weak_mat_tmp280 + weak_mat_tmp297*weak_mat_tmp95) + trial_grad4*(c1*(s_t(4)*weak_mat_tmp1*weak_mat_tmp298 + weak_mat_tmp16 + weak_mat_tmp163*weak_mat_tmp299) + c2*(weak_mat_tmp168*weak_mat_tmp303 + s_t(2)*weak_mat_tmp169*weak_mat_tmp302 + weak_mat_tmp43*(weak_mat_tmp264 + weak_mat_tmp301)) + weak_mat_tmp173*weak_mat_tmp297 + pow_2(weak_mat_tmp175)*weak_mat_tmp9) + trial_grad5*(c1*(weak_mat_tmp129*weak_mat_tmp299 + weak_mat_tmp306) + c2*(weak_mat_tmp135*weak_mat_tmp303 + weak_mat_tmp309) + weak_mat_tmp144*weak_mat_tmp297 + weak_mat_tmp304) + trial_grad6*(c1*(weak_mat_tmp116*weak_mat_tmp299 + weak_mat_tmp314) + c2*(weak_mat_tmp120*weak_mat_tmp303 + weak_mat_tmp315) + weak_mat_tmp115*weak_mat_tmp297 + weak_mat_tmp316) + trial_grad7*(c1*(weak_mat_tmp148*weak_mat_tmp299 + weak_mat_tmp311) + c2*(weak_mat_tmp152*weak_mat_tmp303 + weak_mat_tmp313) + weak_mat_tmp159*weak_mat_tmp297 + weak_mat_tmp310) + trial_grad8*(c1*(weak_mat_tmp178*weak_mat_tmp299 + weak_mat_tmp318) + c2*(weak_mat_tmp182*weak_mat_tmp303 + weak_mat_tmp320) + weak_mat_tmp187*weak_mat_tmp297 + weak_mat_tmp322);
    const s_t material5 = trial_grad0*(c1*(weak_mat_tmp134 + weak_mat_tmp21*weak_mat_tmp326) + c2*(weak_mat_tmp141 + weak_mat_tmp328*weak_mat_tmp58) + weak_mat_tmp147 + weak_mat_tmp3*weak_mat_tmp324) + trial_grad1*(c1*(weak_mat_tmp216 + weak_mat_tmp326*weak_mat_tmp70) + c2*(weak_mat_tmp218 + weak_mat_tmp328*weak_mat_tmp74) + weak_mat_tmp220 + weak_mat_tmp324*weak_mat_tmp67) + trial_grad2*(c1*(weak_mat_tmp241 + weak_mat_tmp326*weak_mat_tmp85) + c2*(weak_mat_tmp243 + weak_mat_tmp328*weak_mat_tmp88) + weak_mat_tmp239 + weak_mat_tmp324*weak_mat_tmp84) + trial_grad3*(c1*(weak_mat_tmp271 + weak_mat_tmp326*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp328 + weak_mat_tmp274) + weak_mat_tmp269 + weak_mat_tmp324*weak_mat_tmp95) + trial_grad4*(c1*(weak_mat_tmp163*weak_mat_tmp326 + weak_mat_tmp306) + c2*(weak_mat_tmp168*weak_mat_tmp328 + weak_mat_tmp309) + weak_mat_tmp173*weak_mat_tmp324 + weak_mat_tmp304) + trial_grad5*(c1*(s_t(4)*gu5*weak_mat_tmp325 + weak_mat_tmp129*weak_mat_tmp326 + weak_mat_tmp16) + c2*(weak_mat_tmp135*weak_mat_tmp328 + s_t(2)*weak_mat_tmp137*weak_mat_tmp327 + weak_mat_tmp43*(weak_mat_tmp266 + weak_mat_tmp301)) + pow_2(weak_mat_tmp143)*weak_mat_tmp9 + weak_mat_tmp144*weak_mat_tmp324) + trial_grad6*(c1*(weak_mat_tmp116*weak_mat_tmp326 + weak_mat_tmp334) + c2*(weak_mat_tmp120*weak_mat_tmp328 + weak_mat_tmp335) + weak_mat_tmp115*weak_mat_tmp324 + weak_mat_tmp336) + trial_grad7*(c1*(weak_mat_tmp148*weak_mat_tmp326 + weak_mat_tmp337) + c2*(weak_mat_tmp152*weak_mat_tmp328 + weak_mat_tmp338) + weak_mat_tmp159*weak_mat_tmp324 + weak_mat_tmp339) + trial_grad8*(c1*(weak_mat_tmp178*weak_mat_tmp326 + weak_mat_tmp331) + c2*(weak_mat_tmp182*weak_mat_tmp328 + weak_mat_tmp333) + weak_mat_tmp187*weak_mat_tmp324 + weak_mat_tmp329);
    const s_t material6 = trial_grad0*(c1*(weak_mat_tmp119 + weak_mat_tmp21*weak_mat_tmp343) + c2*(weak_mat_tmp127 + weak_mat_tmp347*weak_mat_tmp58) + weak_mat_tmp114 + weak_mat_tmp3*weak_mat_tmp341) + trial_grad1*(c1*(weak_mat_tmp221 + weak_mat_tmp343*weak_mat_tmp70) + c2*(weak_mat_tmp222 + weak_mat_tmp347*weak_mat_tmp74) + weak_mat_tmp223 + weak_mat_tmp341*weak_mat_tmp67) + trial_grad2*(c1*(weak_mat_tmp253 + weak_mat_tmp343*weak_mat_tmp85) + c2*(weak_mat_tmp254 + weak_mat_tmp347*weak_mat_tmp88) + weak_mat_tmp255 + weak_mat_tmp341*weak_mat_tmp84) + trial_grad3*(c1*(weak_mat_tmp276 + weak_mat_tmp343*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp347 + weak_mat_tmp279) + weak_mat_tmp275 + weak_mat_tmp341*weak_mat_tmp95) + trial_grad4*(c1*(weak_mat_tmp163*weak_mat_tmp343 + weak_mat_tmp314) + c2*(weak_mat_tmp168*weak_mat_tmp347 + weak_mat_tmp315) + weak_mat_tmp173*weak_mat_tmp341 + weak_mat_tmp316) + trial_grad5*(c1*(weak_mat_tmp129*weak_mat_tmp343 + weak_mat_tmp334) + c2*(weak_mat_tmp135*weak_mat_tmp347 + weak_mat_tmp335) + weak_mat_tmp144*weak_mat_tmp341 + weak_mat_tmp336) + trial_grad6*(c1*(s_t(4)*gu6*weak_mat_tmp342 + weak_mat_tmp116*weak_mat_tmp343 + weak_mat_tmp16) + c2*(weak_mat_tmp120*weak_mat_tmp347 + s_t(2)*weak_mat_tmp126*weak_mat_tmp346 + weak_mat_tmp43*(weak_mat_tmp344 + weak_mat_tmp345)) + pow_2(weak_mat_tmp113)*weak_mat_tmp9 + weak_mat_tmp115*weak_mat_tmp341) + trial_grad7*(c1*(weak_mat_tmp148*weak_mat_tmp343 + weak_mat_tmp350) + c2*(weak_mat_tmp152*weak_mat_tmp347 + weak_mat_tmp352) + weak_mat_tmp159*weak_mat_tmp341 + weak_mat_tmp348) + trial_grad8*(c1*(weak_mat_tmp178*weak_mat_tmp343 + weak_mat_tmp354) + c2*(weak_mat_tmp182*weak_mat_tmp347 + weak_mat_tmp355) + weak_mat_tmp187*weak_mat_tmp341 + weak_mat_tmp353);
    const s_t material7 = trial_grad0*(c1*(weak_mat_tmp151 + weak_mat_tmp21*weak_mat_tmp359) + c2*(weak_mat_tmp156 + weak_mat_tmp362*weak_mat_tmp58) + weak_mat_tmp161 + weak_mat_tmp3*weak_mat_tmp357) + trial_grad1*(c1*(weak_mat_tmp208 + weak_mat_tmp359*weak_mat_tmp70) + c2*(weak_mat_tmp210 + weak_mat_tmp362*weak_mat_tmp74) + weak_mat_tmp207 + weak_mat_tmp357*weak_mat_tmp67) + trial_grad2*(c1*(weak_mat_tmp250 + weak_mat_tmp359*weak_mat_tmp85) + c2*(weak_mat_tmp251 + weak_mat_tmp362*weak_mat_tmp88) + weak_mat_tmp252 + weak_mat_tmp357*weak_mat_tmp84) + trial_grad3*(c1*(weak_mat_tmp285 + weak_mat_tmp359*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp362 + weak_mat_tmp287) + weak_mat_tmp289 + weak_mat_tmp357*weak_mat_tmp95) + trial_grad4*(c1*(weak_mat_tmp163*weak_mat_tmp359 + weak_mat_tmp311) + c2*(weak_mat_tmp168*weak_mat_tmp362 + weak_mat_tmp313) + weak_mat_tmp173*weak_mat_tmp357 + weak_mat_tmp310) + trial_grad5*(c1*(weak_mat_tmp129*weak_mat_tmp359 + weak_mat_tmp337) + c2*(weak_mat_tmp135*weak_mat_tmp362 + weak_mat_tmp338) + weak_mat_tmp144*weak_mat_tmp357 + weak_mat_tmp339) + trial_grad6*(c1*(weak_mat_tmp116*weak_mat_tmp359 + weak_mat_tmp350) + c2*(weak_mat_tmp120*weak_mat_tmp362 + weak_mat_tmp352) + weak_mat_tmp115*weak_mat_tmp357 + weak_mat_tmp348) + trial_grad7*(c1*(s_t(4)*gu7*weak_mat_tmp358 + weak_mat_tmp148*weak_mat_tmp359 + weak_mat_tmp16) + c2*(weak_mat_tmp152*weak_mat_tmp362 + s_t(2)*weak_mat_tmp154*weak_mat_tmp361 + weak_mat_tmp43*(weak_mat_tmp344 + weak_mat_tmp360)) + pow_2(weak_mat_tmp158)*weak_mat_tmp9 + weak_mat_tmp159*weak_mat_tmp357) + trial_grad8*(c1*(weak_mat_tmp178*weak_mat_tmp359 + weak_mat_tmp364) + c2*(weak_mat_tmp182*weak_mat_tmp362 + weak_mat_tmp366) + weak_mat_tmp187*weak_mat_tmp357 + weak_mat_tmp363);
    const s_t material8 = trial_grad0*(c1*(weak_mat_tmp181 + weak_mat_tmp21*weak_mat_tmp368) + c2*(weak_mat_tmp186 + weak_mat_tmp369*weak_mat_tmp58) + weak_mat_tmp190 + weak_mat_tmp3*weak_mat_tmp367) + trial_grad1*(c1*(weak_mat_tmp228 + weak_mat_tmp368*weak_mat_tmp70) + c2*(weak_mat_tmp230 + weak_mat_tmp369*weak_mat_tmp74) + weak_mat_tmp232 + weak_mat_tmp367*weak_mat_tmp67) + trial_grad2*(c1*(weak_mat_tmp245 + weak_mat_tmp368*weak_mat_tmp85) + c2*(weak_mat_tmp246 + weak_mat_tmp369*weak_mat_tmp88) + weak_mat_tmp244 + weak_mat_tmp367*weak_mat_tmp84) + trial_grad3*(c1*(weak_mat_tmp291 + weak_mat_tmp368*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp369 + weak_mat_tmp293) + weak_mat_tmp295 + weak_mat_tmp367*weak_mat_tmp95) + trial_grad4*(c1*(weak_mat_tmp163*weak_mat_tmp368 + weak_mat_tmp318) + c2*(weak_mat_tmp168*weak_mat_tmp369 + weak_mat_tmp320) + weak_mat_tmp173*weak_mat_tmp367 + weak_mat_tmp322) + trial_grad5*(c1*(weak_mat_tmp129*weak_mat_tmp368 + weak_mat_tmp331) + c2*(weak_mat_tmp135*weak_mat_tmp369 + weak_mat_tmp333) + weak_mat_tmp144*weak_mat_tmp367 + weak_mat_tmp329) + trial_grad6*(c1*(weak_mat_tmp116*weak_mat_tmp368 + weak_mat_tmp354) + c2*(weak_mat_tmp120*weak_mat_tmp369 + weak_mat_tmp355) + weak_mat_tmp115*weak_mat_tmp367 + weak_mat_tmp353) + trial_grad7*(c1*(weak_mat_tmp148*weak_mat_tmp368 + weak_mat_tmp364) + c2*(weak_mat_tmp152*weak_mat_tmp369 + weak_mat_tmp366) + weak_mat_tmp159*weak_mat_tmp367 + weak_mat_tmp363) + trial_grad8*(c1*(weak_mat_tmp16 + s_t(4)*weak_mat_tmp165*weak_mat_tmp179 + weak_mat_tmp178*weak_mat_tmp368) + c2*(weak_mat_tmp182*weak_mat_tmp369 + s_t(2)*weak_mat_tmp184*weak_mat_tmp365 + weak_mat_tmp43*(weak_mat_tmp345 + weak_mat_tmp360)) + weak_mat_tmp187*weak_mat_tmp367 + pow_2(weak_mat_tmp189)*weak_mat_tmp9);
    const s_t loperand0 = qw * (material0 * adj_lane0 + material1 * adj_lane1 + material2 * adj_lane2);
    const s_t loperand1 = qw * (material0 * adj_lane3 + material1 * adj_lane4 + material2 * adj_lane5);
    const s_t loperand2 = qw * (material0 * adj_lane6 + material1 * adj_lane7 + material2 * adj_lane8);
    const s_t loperand3 = qw * (material3 * adj_lane0 + material4 * adj_lane1 + material5 * adj_lane2);
    const s_t loperand4 = qw * (material3 * adj_lane3 + material4 * adj_lane4 + material5 * adj_lane5);
    const s_t loperand5 = qw * (material3 * adj_lane6 + material4 * adj_lane7 + material5 * adj_lane8);
    const s_t loperand6 = qw * (material6 * adj_lane0 + material7 * adj_lane1 + material8 * adj_lane2);
    const s_t loperand7 = qw * (material6 * adj_lane3 + material7 * adj_lane4 + material8 * adj_lane5);
    const s_t loperand8 = qw * (material6 * adj_lane6 + material7 * adj_lane7 + material8 * adj_lane8);
      loperand0_values[lane] = loperand0;
      loperand1_values[lane] = loperand1;
      loperand2_values[lane] = loperand2;
      loperand3_values[lane] = loperand3;
      loperand4_values[lane] = loperand4;
      loperand5_values[lane] = loperand5;
      loperand6_values[lane] = loperand6;
      loperand7_values[lane] = loperand7;
      loperand8_values[lane] = loperand8;
      }
      for (int shape = 0; shape < NS; ++shape) {
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          out_streams[3 * shape][lane] += loperand0_values[lane] * grad_ref_x[q * NS + shape] + loperand1_values[lane] * grad_ref_y[q * NS + shape] + loperand2_values[lane] * grad_ref_z[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          out_streams[3 * shape + 1][lane] += loperand3_values[lane] * grad_ref_x[q * NS + shape] + loperand4_values[lane] * grad_ref_y[q * NS + shape] + loperand5_values[lane] * grad_ref_z[q * NS + shape];
        }
        #pragma omp simd
        for (int lane = 0; lane < ne; ++lane) {
          out_streams[3 * shape + 2][lane] += loperand6_values[lane] * grad_ref_x[q * NS + shape] + loperand7_values[lane] * grad_ref_y[q * NS + shape] + loperand8_values[lane] * grad_ref_z[q * NS + shape];
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void modified_mooney_rivlin_d3_simplex_tet4_apply_block(
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
        const s_t *const RSTR q_weight,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 3],
        const s_t *const RSTR h_streams[NS * 3],
        s_t *const RSTR out_streams[NS * 3]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
    { const int q = 0;  // constant-P1 simplex
      const s_t qw = q_weight[q];
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
    const s_t weak_mat_tmp3 = weak_mat_tmp0 - weak_mat_tmp1*weak_mat_tmp2;
    const s_t weak_mat_tmp4 = -weak_mat_tmp3;
    const s_t weak_mat_tmp5 = gu1*gu3;
    const s_t weak_mat_tmp6 = gu2*gu6;
    const s_t weak_mat_tmp7 = gu0 + s_t(1);
    const s_t weak_mat_tmp8 = gu1*gu5*gu6 + gu2*gu3*gu7 - weak_mat_tmp0*weak_mat_tmp7 + weak_mat_tmp1*weak_mat_tmp2*weak_mat_tmp7 - weak_mat_tmp1*weak_mat_tmp6 - weak_mat_tmp2*weak_mat_tmp5;
    const s_t weak_mat_tmp9 = kappa/pow_2(weak_mat_tmp8);
    const s_t weak_mat_tmp10 = gu0*gu4;
    const s_t weak_mat_tmp11 = gu5*gu6;
    const s_t weak_mat_tmp12 = gu3*gu7;
    const s_t weak_mat_tmp13 = sfem_log1p(gu0*gu8 - gu0*weak_mat_tmp0 + gu0 + gu1*weak_mat_tmp11 + gu2*weak_mat_tmp12 + gu4*gu8 - gu4*weak_mat_tmp6 + gu4 + gu8*weak_mat_tmp10 - gu8*weak_mat_tmp5 + gu8 - weak_mat_tmp0 + weak_mat_tmp10 - weak_mat_tmp5 - weak_mat_tmp6);
    const s_t weak_mat_tmp14 = weak_mat_tmp4*weak_mat_tmp9;
    const s_t weak_mat_tmp15 = weak_mat_tmp13*weak_mat_tmp14;
    const s_t weak_mat_tmp16 = s_t(2)/pow(weak_mat_tmp8, (s_t(2) / s_t(3)));
    const s_t weak_mat_tmp17 = pow(weak_mat_tmp8, (s_t(-5) / s_t(3)));
    const s_t weak_mat_tmp18 = weak_mat_tmp1*weak_mat_tmp2;
    const s_t weak_mat_tmp19 = ((s_t(2) / s_t(3)))*weak_mat_tmp0 - (s_t(2) / s_t(3))*weak_mat_tmp18;
    const s_t weak_mat_tmp20 = weak_mat_tmp17*weak_mat_tmp19;
    const s_t weak_mat_tmp21 = ((s_t(5) / s_t(3)))*weak_mat_tmp0 - (s_t(5) / s_t(3))*weak_mat_tmp18;
    const s_t weak_mat_tmp22 = pow_2(gu3);
    const s_t weak_mat_tmp23 = pow_2(gu6);
    const s_t weak_mat_tmp24 = pow_2(weak_mat_tmp7);
    const s_t weak_mat_tmp25 = weak_mat_tmp22 + weak_mat_tmp23 + weak_mat_tmp24;
    const s_t weak_mat_tmp26 = pow_2(gu1);
    const s_t weak_mat_tmp27 = pow_2(gu7);
    const s_t weak_mat_tmp28 = pow_2(weak_mat_tmp1);
    const s_t weak_mat_tmp29 = weak_mat_tmp26 + weak_mat_tmp27 + weak_mat_tmp28;
    const s_t weak_mat_tmp30 = pow_2(gu2);
    const s_t weak_mat_tmp31 = pow_2(gu5);
    const s_t weak_mat_tmp32 = pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp33 = weak_mat_tmp30 + weak_mat_tmp31 + weak_mat_tmp32;
    const s_t weak_mat_tmp34 = weak_mat_tmp25 + weak_mat_tmp29 + weak_mat_tmp33;
    const s_t weak_mat_tmp35 = weak_mat_tmp34/pow(weak_mat_tmp8, (s_t(8) / s_t(3)));
    const s_t weak_mat_tmp36 = weak_mat_tmp19*weak_mat_tmp35;
    const s_t weak_mat_tmp37 = s_t(2)*weak_mat_tmp31;
    const s_t weak_mat_tmp38 = s_t(2)*weak_mat_tmp32;
    const s_t weak_mat_tmp39 = weak_mat_tmp37 + weak_mat_tmp38;
    const s_t weak_mat_tmp40 = s_t(2)*weak_mat_tmp27;
    const s_t weak_mat_tmp41 = s_t(2)*weak_mat_tmp28;
    const s_t weak_mat_tmp42 = weak_mat_tmp40 + weak_mat_tmp41;
    const s_t weak_mat_tmp43 = pow(weak_mat_tmp8, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp44 = ((s_t(4) / s_t(3)))*weak_mat_tmp0 - (s_t(4) / s_t(3))*weak_mat_tmp18;
    const s_t weak_mat_tmp45 = pow(weak_mat_tmp8, (s_t(-7) / s_t(3)));
    const s_t weak_mat_tmp46 = gu6*gu7;
    const s_t weak_mat_tmp47 = gu1*weak_mat_tmp7;
    const s_t weak_mat_tmp48 = gu3*weak_mat_tmp1;
    const s_t weak_mat_tmp49 = weak_mat_tmp46 + weak_mat_tmp47 + weak_mat_tmp48;
    const s_t weak_mat_tmp50 = s_t(2)*gu1;
    const s_t weak_mat_tmp51 = gu3*gu5;
    const s_t weak_mat_tmp52 = gu2*weak_mat_tmp7;
    const s_t weak_mat_tmp53 = gu6*weak_mat_tmp2;
    const s_t weak_mat_tmp54 = weak_mat_tmp51 + weak_mat_tmp52 + weak_mat_tmp53;
    const s_t weak_mat_tmp55 = s_t(2)*gu2;
    const s_t weak_mat_tmp56 = s_t(2)*weak_mat_tmp7;
    const s_t weak_mat_tmp57 = weak_mat_tmp45*(-weak_mat_tmp25*weak_mat_tmp56 + s_t(2)*weak_mat_tmp34*weak_mat_tmp7 - weak_mat_tmp49*weak_mat_tmp50 - weak_mat_tmp54*weak_mat_tmp55);
    const s_t weak_mat_tmp58 = ((s_t(7) / s_t(3)))*weak_mat_tmp0 - (s_t(7) / s_t(3))*weak_mat_tmp18;
    const s_t weak_mat_tmp59 = gu1*gu2;
    const s_t weak_mat_tmp60 = gu5*weak_mat_tmp1;
    const s_t weak_mat_tmp61 = gu7*weak_mat_tmp2;
    const s_t weak_mat_tmp62 = weak_mat_tmp59 + weak_mat_tmp60 + weak_mat_tmp61;
    const s_t weak_mat_tmp63 = -(s_t(1) / s_t(2))*pow_2(weak_mat_tmp25) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp29) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp33) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp34) - pow_2(weak_mat_tmp49) - pow_2(weak_mat_tmp54) - pow_2(weak_mat_tmp62);
    const s_t weak_mat_tmp64 = weak_mat_tmp63/pow(weak_mat_tmp8, (s_t(10) / s_t(3)));
    const s_t weak_mat_tmp65 = weak_mat_tmp44*weak_mat_tmp64;
    const s_t weak_mat_tmp66 = gu3*weak_mat_tmp2;
    const s_t weak_mat_tmp67 = -gu5*gu6 + weak_mat_tmp66;
    const s_t weak_mat_tmp68 = -weak_mat_tmp67;
    const s_t weak_mat_tmp69 = weak_mat_tmp14*weak_mat_tmp68;
    const s_t weak_mat_tmp70 = -(s_t(5) / s_t(3))*weak_mat_tmp11 + ((s_t(5) / s_t(3)))*weak_mat_tmp66;
    const s_t weak_mat_tmp71 = -(s_t(2) / s_t(3))*weak_mat_tmp11 + ((s_t(2) / s_t(3)))*weak_mat_tmp66;
    const s_t weak_mat_tmp72 = weak_mat_tmp17*weak_mat_tmp56;
    const s_t weak_mat_tmp73 = weak_mat_tmp20*weak_mat_tmp50 + weak_mat_tmp71*weak_mat_tmp72;
    const s_t weak_mat_tmp74 = -(s_t(7) / s_t(3))*weak_mat_tmp11 + ((s_t(7) / s_t(3)))*weak_mat_tmp66;
    const s_t weak_mat_tmp75 = s_t(2)*weak_mat_tmp46;
    const s_t weak_mat_tmp76 = s_t(2)*weak_mat_tmp48;
    const s_t weak_mat_tmp77 = -(s_t(4) / s_t(3))*weak_mat_tmp11 + ((s_t(4) / s_t(3)))*weak_mat_tmp66;
    const s_t weak_mat_tmp78 = s_t(2)*gu1*weak_mat_tmp34 - weak_mat_tmp29*weak_mat_tmp50 - weak_mat_tmp49*weak_mat_tmp56 - weak_mat_tmp55*weak_mat_tmp62;
    const s_t weak_mat_tmp79 = weak_mat_tmp44*weak_mat_tmp45;
    const s_t weak_mat_tmp80 = weak_mat_tmp43*(-weak_mat_tmp75 - weak_mat_tmp76) + weak_mat_tmp57*weak_mat_tmp77 + weak_mat_tmp78*weak_mat_tmp79;
    const s_t weak_mat_tmp81 = gu6*weak_mat_tmp1;
    const s_t weak_mat_tmp82 = weak_mat_tmp12 - weak_mat_tmp81;
    const s_t weak_mat_tmp83 = weak_mat_tmp14*weak_mat_tmp82;
    const s_t weak_mat_tmp84 = -weak_mat_tmp82;
    const s_t weak_mat_tmp85 = ((s_t(5) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(5) / s_t(3))*weak_mat_tmp12;
    const s_t weak_mat_tmp86 = ((s_t(2) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(2) / s_t(3))*weak_mat_tmp12;
    const s_t weak_mat_tmp87 = weak_mat_tmp20*weak_mat_tmp55 + weak_mat_tmp72*weak_mat_tmp86;
    const s_t weak_mat_tmp88 = ((s_t(7) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(7) / s_t(3))*weak_mat_tmp12;
    const s_t weak_mat_tmp89 = s_t(2)*weak_mat_tmp51;
    const s_t weak_mat_tmp90 = s_t(2)*weak_mat_tmp53;
    const s_t weak_mat_tmp91 = ((s_t(4) / s_t(3)))*gu6*weak_mat_tmp1 - (s_t(4) / s_t(3))*weak_mat_tmp12;
    const s_t weak_mat_tmp92 = s_t(2)*gu2*weak_mat_tmp34 - weak_mat_tmp33*weak_mat_tmp55 - weak_mat_tmp50*weak_mat_tmp62 - weak_mat_tmp54*weak_mat_tmp56;
    const s_t weak_mat_tmp93 = weak_mat_tmp43*(-weak_mat_tmp89 - weak_mat_tmp90) + weak_mat_tmp57*weak_mat_tmp91 + weak_mat_tmp79*weak_mat_tmp92;
    const s_t weak_mat_tmp94 = gu1*weak_mat_tmp2;
    const s_t weak_mat_tmp95 = -gu2*gu7 + weak_mat_tmp94;
    const s_t weak_mat_tmp96 = -weak_mat_tmp95;
    const s_t weak_mat_tmp97 = weak_mat_tmp14*weak_mat_tmp96;
    const s_t weak_mat_tmp98 = gu2*gu7;
    const s_t weak_mat_tmp99 = ((s_t(5) / s_t(3)))*weak_mat_tmp94 - (s_t(5) / s_t(3))*weak_mat_tmp98;
    const s_t weak_mat_tmp100 = s_t(2)*gu3;
    const s_t weak_mat_tmp101 = ((s_t(2) / s_t(3)))*weak_mat_tmp94 - (s_t(2) / s_t(3))*weak_mat_tmp98;
    const s_t weak_mat_tmp102 = weak_mat_tmp100*weak_mat_tmp20 + weak_mat_tmp101*weak_mat_tmp72;
    const s_t weak_mat_tmp103 = ((s_t(7) / s_t(3)))*weak_mat_tmp94 - (s_t(7) / s_t(3))*weak_mat_tmp98;
    const s_t weak_mat_tmp104 = gu5*weak_mat_tmp55;
    const s_t weak_mat_tmp105 = weak_mat_tmp1*weak_mat_tmp50;
    const s_t weak_mat_tmp106 = ((s_t(4) / s_t(3)))*weak_mat_tmp94 - (s_t(4) / s_t(3))*weak_mat_tmp98;
    const s_t weak_mat_tmp107 = s_t(2)*gu5;
    const s_t weak_mat_tmp108 = s_t(2)*weak_mat_tmp1;
    const s_t weak_mat_tmp109 = s_t(2)*gu3*weak_mat_tmp34 - weak_mat_tmp100*weak_mat_tmp25 - weak_mat_tmp107*weak_mat_tmp54 - weak_mat_tmp108*weak_mat_tmp49;
    const s_t weak_mat_tmp110 = weak_mat_tmp106*weak_mat_tmp57 + weak_mat_tmp109*weak_mat_tmp79 + weak_mat_tmp43*(-weak_mat_tmp104 - weak_mat_tmp105);
    const s_t weak_mat_tmp111 = gu1*gu5;
    const s_t weak_mat_tmp112 = gu2*weak_mat_tmp1;
    const s_t weak_mat_tmp113 = weak_mat_tmp111 - weak_mat_tmp112;
    const s_t weak_mat_tmp114 = weak_mat_tmp113*weak_mat_tmp14;
    const s_t weak_mat_tmp115 = -weak_mat_tmp113;
    const s_t weak_mat_tmp116 = ((s_t(5) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(5) / s_t(3))*weak_mat_tmp111;
    const s_t weak_mat_tmp117 = s_t(2)*gu6;
    const s_t weak_mat_tmp118 = ((s_t(2) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(2) / s_t(3))*weak_mat_tmp111;
    const s_t weak_mat_tmp119 = weak_mat_tmp117*weak_mat_tmp20 + weak_mat_tmp118*weak_mat_tmp72;
    const s_t weak_mat_tmp120 = ((s_t(7) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(7) / s_t(3))*weak_mat_tmp111;
    const s_t weak_mat_tmp121 = gu7*weak_mat_tmp50;
    const s_t weak_mat_tmp122 = weak_mat_tmp2*weak_mat_tmp55;
    const s_t weak_mat_tmp123 = ((s_t(4) / s_t(3)))*gu2*weak_mat_tmp1 - (s_t(4) / s_t(3))*weak_mat_tmp111;
    const s_t weak_mat_tmp124 = s_t(2)*gu7;
    const s_t weak_mat_tmp125 = s_t(2)*weak_mat_tmp2;
    const s_t weak_mat_tmp126 = s_t(2)*gu6*weak_mat_tmp34 - weak_mat_tmp117*weak_mat_tmp25 - weak_mat_tmp124*weak_mat_tmp49 - weak_mat_tmp125*weak_mat_tmp54;
    const s_t weak_mat_tmp127 = weak_mat_tmp123*weak_mat_tmp57 + weak_mat_tmp126*weak_mat_tmp79 + weak_mat_tmp43*(-weak_mat_tmp121 - weak_mat_tmp122);
    const s_t weak_mat_tmp128 = gu1*gu6;
    const s_t weak_mat_tmp129 = ((s_t(5) / s_t(3)))*gu7*weak_mat_tmp7 - (s_t(5) / s_t(3))*weak_mat_tmp128;
    const s_t weak_mat_tmp130 = ((s_t(2) / s_t(3)))*gu7*weak_mat_tmp7 - (s_t(2) / s_t(3))*weak_mat_tmp128;
    const s_t weak_mat_tmp131 = ((s_t(2) / s_t(3)))*weak_mat_tmp34;
    const s_t weak_mat_tmp132 = weak_mat_tmp131*weak_mat_tmp17;
    const s_t weak_mat_tmp133 = gu7*weak_mat_tmp132;
    const s_t weak_mat_tmp134 = weak_mat_tmp107*weak_mat_tmp20 + weak_mat_tmp130*weak_mat_tmp72 + weak_mat_tmp133;
    const s_t weak_mat_tmp135 = ((s_t(7) / s_t(3)))*gu7*weak_mat_tmp7 - (s_t(7) / s_t(3))*weak_mat_tmp128;
    const s_t weak_mat_tmp136 = gu2*gu3;
    const s_t weak_mat_tmp137 = ((s_t(4) / s_t(3)))*gu7*weak_mat_tmp7 - (s_t(4) / s_t(3))*weak_mat_tmp128;
    const s_t weak_mat_tmp138 = s_t(2)*gu5*weak_mat_tmp34 - weak_mat_tmp100*weak_mat_tmp54 - weak_mat_tmp107*weak_mat_tmp33 - weak_mat_tmp108*weak_mat_tmp62;
    const s_t weak_mat_tmp139 = ((s_t(4) / s_t(3)))*weak_mat_tmp45*weak_mat_tmp63;
    const s_t weak_mat_tmp140 = gu7*weak_mat_tmp139;
    const s_t weak_mat_tmp141 = weak_mat_tmp137*weak_mat_tmp57 + weak_mat_tmp138*weak_mat_tmp79 + weak_mat_tmp140 + weak_mat_tmp43*(s_t(4)*gu5*weak_mat_tmp7 - s_t(2)*weak_mat_tmp136);
    const s_t weak_mat_tmp142 = gu7*weak_mat_tmp7;
    const s_t weak_mat_tmp143 = weak_mat_tmp128 - weak_mat_tmp142;
    const s_t weak_mat_tmp144 = -weak_mat_tmp143;
    const s_t weak_mat_tmp145 = kappa*weak_mat_tmp13/weak_mat_tmp8;
    const s_t weak_mat_tmp146 = gu7*weak_mat_tmp145;
    const s_t weak_mat_tmp147 = weak_mat_tmp14*weak_mat_tmp143 - weak_mat_tmp146;
    const s_t weak_mat_tmp148 = ((s_t(5) / s_t(3)))*gu5*weak_mat_tmp7 - (s_t(5) / s_t(3))*weak_mat_tmp136;
    const s_t weak_mat_tmp149 = ((s_t(2) / s_t(3)))*gu5*weak_mat_tmp7 - (s_t(2) / s_t(3))*weak_mat_tmp136;
    const s_t weak_mat_tmp150 = gu5*weak_mat_tmp132;
    const s_t weak_mat_tmp151 = weak_mat_tmp124*weak_mat_tmp20 + weak_mat_tmp149*weak_mat_tmp72 + weak_mat_tmp150;
    const s_t weak_mat_tmp152 = ((s_t(7) / s_t(3)))*gu5*weak_mat_tmp7 - (s_t(7) / s_t(3))*weak_mat_tmp136;
    const s_t weak_mat_tmp153 = ((s_t(4) / s_t(3)))*gu5*weak_mat_tmp7 - (s_t(4) / s_t(3))*weak_mat_tmp136;
    const s_t weak_mat_tmp154 = s_t(2)*gu7*weak_mat_tmp34 - weak_mat_tmp117*weak_mat_tmp49 - weak_mat_tmp124*weak_mat_tmp29 - weak_mat_tmp125*weak_mat_tmp62;
    const s_t weak_mat_tmp155 = gu5*weak_mat_tmp139;
    const s_t weak_mat_tmp156 = weak_mat_tmp153*weak_mat_tmp57 + weak_mat_tmp154*weak_mat_tmp79 + weak_mat_tmp155 + weak_mat_tmp43*(s_t(4)*gu7*weak_mat_tmp7 - s_t(2)*weak_mat_tmp128);
    const s_t weak_mat_tmp157 = gu5*weak_mat_tmp7;
    const s_t weak_mat_tmp158 = weak_mat_tmp136 - weak_mat_tmp157;
    const s_t weak_mat_tmp159 = -weak_mat_tmp158;
    const s_t weak_mat_tmp160 = gu5*weak_mat_tmp145;
    const s_t weak_mat_tmp161 = weak_mat_tmp14*weak_mat_tmp158 - weak_mat_tmp160;
    const s_t weak_mat_tmp162 = weak_mat_tmp2*weak_mat_tmp7;
    const s_t weak_mat_tmp163 = -(s_t(5) / s_t(3))*weak_mat_tmp162 + ((s_t(5) / s_t(3)))*weak_mat_tmp6;
    const s_t weak_mat_tmp164 = -(s_t(2) / s_t(3))*weak_mat_tmp162 + ((s_t(2) / s_t(3)))*weak_mat_tmp6;
    const s_t weak_mat_tmp165 = weak_mat_tmp17*weak_mat_tmp2;
    const s_t weak_mat_tmp166 = weak_mat_tmp131*weak_mat_tmp165;
    const s_t weak_mat_tmp167 = weak_mat_tmp108*weak_mat_tmp20 + weak_mat_tmp164*weak_mat_tmp72 - weak_mat_tmp166;
    const s_t weak_mat_tmp168 = -(s_t(7) / s_t(3))*weak_mat_tmp162 + ((s_t(7) / s_t(3)))*weak_mat_tmp6;
    const s_t weak_mat_tmp169 = -(s_t(4) / s_t(3))*weak_mat_tmp162 + ((s_t(4) / s_t(3)))*weak_mat_tmp6;
    const s_t weak_mat_tmp170 = s_t(2)*weak_mat_tmp1*weak_mat_tmp34 - weak_mat_tmp100*weak_mat_tmp49 - weak_mat_tmp107*weak_mat_tmp62 - weak_mat_tmp108*weak_mat_tmp29;
    const s_t weak_mat_tmp171 = weak_mat_tmp139*weak_mat_tmp2;
    const s_t weak_mat_tmp172 = weak_mat_tmp169*weak_mat_tmp57 + weak_mat_tmp170*weak_mat_tmp79 - weak_mat_tmp171 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp1*weak_mat_tmp7 - s_t(2)*weak_mat_tmp5);
    const s_t weak_mat_tmp173 = -weak_mat_tmp2*weak_mat_tmp7 + weak_mat_tmp6;
    const s_t weak_mat_tmp174 = weak_mat_tmp145*weak_mat_tmp2;
    const s_t weak_mat_tmp175 = -weak_mat_tmp173;
    const s_t weak_mat_tmp176 = weak_mat_tmp14*weak_mat_tmp175 + weak_mat_tmp174;
    const s_t weak_mat_tmp177 = weak_mat_tmp1*weak_mat_tmp7;
    const s_t weak_mat_tmp178 = -(s_t(5) / s_t(3))*weak_mat_tmp177 + ((s_t(5) / s_t(3)))*weak_mat_tmp5;
    const s_t weak_mat_tmp179 = -(s_t(2) / s_t(3))*weak_mat_tmp177 + ((s_t(2) / s_t(3)))*weak_mat_tmp5;
    const s_t weak_mat_tmp180 = weak_mat_tmp1*weak_mat_tmp132;
    const s_t weak_mat_tmp181 = weak_mat_tmp125*weak_mat_tmp20 + weak_mat_tmp179*weak_mat_tmp72 - weak_mat_tmp180;
    const s_t weak_mat_tmp182 = -(s_t(7) / s_t(3))*weak_mat_tmp177 + ((s_t(7) / s_t(3)))*weak_mat_tmp5;
    const s_t weak_mat_tmp183 = -(s_t(4) / s_t(3))*weak_mat_tmp177 + ((s_t(4) / s_t(3)))*weak_mat_tmp5;
    const s_t weak_mat_tmp184 = -weak_mat_tmp117*weak_mat_tmp54 - weak_mat_tmp124*weak_mat_tmp62 - weak_mat_tmp125*weak_mat_tmp33 + s_t(2)*weak_mat_tmp2*weak_mat_tmp34;
    const s_t weak_mat_tmp185 = weak_mat_tmp1*weak_mat_tmp139;
    const s_t weak_mat_tmp186 = weak_mat_tmp183*weak_mat_tmp57 + weak_mat_tmp184*weak_mat_tmp79 - weak_mat_tmp185 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp2*weak_mat_tmp7 - s_t(2)*weak_mat_tmp6);
    const s_t weak_mat_tmp187 = -weak_mat_tmp1*weak_mat_tmp7 + weak_mat_tmp5;
    const s_t weak_mat_tmp188 = weak_mat_tmp1*weak_mat_tmp145;
    const s_t weak_mat_tmp189 = -weak_mat_tmp187;
    const s_t weak_mat_tmp190 = weak_mat_tmp14*weak_mat_tmp189 + weak_mat_tmp188;
    const s_t weak_mat_tmp191 = weak_mat_tmp68*weak_mat_tmp9;
    const s_t weak_mat_tmp192 = weak_mat_tmp13*weak_mat_tmp191;
    const s_t weak_mat_tmp193 = weak_mat_tmp17*weak_mat_tmp71;
    const s_t weak_mat_tmp194 = weak_mat_tmp35*weak_mat_tmp71;
    const s_t weak_mat_tmp195 = s_t(2)*weak_mat_tmp22;
    const s_t weak_mat_tmp196 = s_t(2)*weak_mat_tmp23;
    const s_t weak_mat_tmp197 = weak_mat_tmp195 + weak_mat_tmp196;
    const s_t weak_mat_tmp198 = weak_mat_tmp45*weak_mat_tmp78;
    const s_t weak_mat_tmp199 = weak_mat_tmp64*weak_mat_tmp77;
    const s_t weak_mat_tmp200 = weak_mat_tmp191*weak_mat_tmp82;
    const s_t weak_mat_tmp201 = weak_mat_tmp17*weak_mat_tmp50;
    const s_t weak_mat_tmp202 = weak_mat_tmp193*weak_mat_tmp55 + weak_mat_tmp201*weak_mat_tmp86;
    const s_t weak_mat_tmp203 = s_t(2)*weak_mat_tmp60;
    const s_t weak_mat_tmp204 = s_t(2)*weak_mat_tmp61;
    const s_t weak_mat_tmp205 = weak_mat_tmp45*weak_mat_tmp77;
    const s_t weak_mat_tmp206 = weak_mat_tmp198*weak_mat_tmp91 + weak_mat_tmp205*weak_mat_tmp92 + weak_mat_tmp43*(-weak_mat_tmp203 - weak_mat_tmp204);
    const s_t weak_mat_tmp207 = weak_mat_tmp158*weak_mat_tmp191;
    const s_t weak_mat_tmp208 = weak_mat_tmp124*weak_mat_tmp193 + weak_mat_tmp149*weak_mat_tmp201;
    const s_t weak_mat_tmp209 = gu6*weak_mat_tmp56;
    const s_t weak_mat_tmp210 = weak_mat_tmp153*weak_mat_tmp198 + weak_mat_tmp154*weak_mat_tmp205 + weak_mat_tmp43*(-weak_mat_tmp122 - weak_mat_tmp209);
    const s_t weak_mat_tmp211 = weak_mat_tmp175*weak_mat_tmp191;
    const s_t weak_mat_tmp212 = weak_mat_tmp108*weak_mat_tmp193 + weak_mat_tmp164*weak_mat_tmp201;
    const s_t weak_mat_tmp213 = gu3*weak_mat_tmp56;
    const s_t weak_mat_tmp214 = weak_mat_tmp169*weak_mat_tmp198 + weak_mat_tmp170*weak_mat_tmp205 + weak_mat_tmp43*(-weak_mat_tmp104 - weak_mat_tmp213);
    const s_t weak_mat_tmp215 = gu6*weak_mat_tmp132;
    const s_t weak_mat_tmp216 = weak_mat_tmp107*weak_mat_tmp193 + weak_mat_tmp130*weak_mat_tmp201 - weak_mat_tmp215;
    const s_t weak_mat_tmp217 = gu6*weak_mat_tmp139;
    const s_t weak_mat_tmp218 = weak_mat_tmp137*weak_mat_tmp198 + weak_mat_tmp138*weak_mat_tmp205 - weak_mat_tmp217 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp111 - s_t(2)*weak_mat_tmp112);
    const s_t weak_mat_tmp219 = gu6*weak_mat_tmp145;
    const s_t weak_mat_tmp220 = weak_mat_tmp143*weak_mat_tmp191 + weak_mat_tmp219;
    const s_t weak_mat_tmp221 = weak_mat_tmp117*weak_mat_tmp193 + weak_mat_tmp118*weak_mat_tmp201 - weak_mat_tmp150;
    const s_t weak_mat_tmp222 = weak_mat_tmp123*weak_mat_tmp198 + weak_mat_tmp126*weak_mat_tmp205 - weak_mat_tmp155 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp128 - s_t(2)*weak_mat_tmp142);
    const s_t weak_mat_tmp223 = weak_mat_tmp113*weak_mat_tmp191 + weak_mat_tmp160;
    const s_t weak_mat_tmp224 = weak_mat_tmp100*weak_mat_tmp193 + weak_mat_tmp101*weak_mat_tmp201 + weak_mat_tmp166;
    const s_t weak_mat_tmp225 = weak_mat_tmp106*weak_mat_tmp198 + weak_mat_tmp109*weak_mat_tmp205 + weak_mat_tmp171 + weak_mat_tmp43*(-s_t(2)*weak_mat_tmp177 + s_t(4)*weak_mat_tmp5);
    const s_t weak_mat_tmp226 = -weak_mat_tmp174 + weak_mat_tmp191*weak_mat_tmp96;
    const s_t weak_mat_tmp227 = gu3*weak_mat_tmp132;
    const s_t weak_mat_tmp228 = weak_mat_tmp125*weak_mat_tmp193 + weak_mat_tmp179*weak_mat_tmp201 + weak_mat_tmp227;
    const s_t weak_mat_tmp229 = gu3*weak_mat_tmp139;
    const s_t weak_mat_tmp230 = weak_mat_tmp183*weak_mat_tmp198 + weak_mat_tmp184*weak_mat_tmp205 + weak_mat_tmp229 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp94 - s_t(2)*weak_mat_tmp98);
    const s_t weak_mat_tmp231 = gu3*weak_mat_tmp145;
    const s_t weak_mat_tmp232 = weak_mat_tmp189*weak_mat_tmp191 - weak_mat_tmp231;
    const s_t weak_mat_tmp233 = weak_mat_tmp82*weak_mat_tmp9;
    const s_t weak_mat_tmp234 = weak_mat_tmp13*weak_mat_tmp233;
    const s_t weak_mat_tmp235 = weak_mat_tmp17*weak_mat_tmp86;
    const s_t weak_mat_tmp236 = weak_mat_tmp35*weak_mat_tmp86;
    const s_t weak_mat_tmp237 = weak_mat_tmp45*weak_mat_tmp92;
    const s_t weak_mat_tmp238 = weak_mat_tmp64*weak_mat_tmp91;
    const s_t weak_mat_tmp239 = weak_mat_tmp143*weak_mat_tmp233;
    const s_t weak_mat_tmp240 = weak_mat_tmp17*weak_mat_tmp55;
    const s_t weak_mat_tmp241 = weak_mat_tmp107*weak_mat_tmp235 + weak_mat_tmp130*weak_mat_tmp240;
    const s_t weak_mat_tmp242 = weak_mat_tmp45*weak_mat_tmp91;
    const s_t weak_mat_tmp243 = weak_mat_tmp137*weak_mat_tmp237 + weak_mat_tmp138*weak_mat_tmp242 + weak_mat_tmp43*(-weak_mat_tmp105 - weak_mat_tmp213);
    const s_t weak_mat_tmp244 = weak_mat_tmp189*weak_mat_tmp233;
    const s_t weak_mat_tmp245 = weak_mat_tmp125*weak_mat_tmp235 + weak_mat_tmp179*weak_mat_tmp240;
    const s_t weak_mat_tmp246 = weak_mat_tmp183*weak_mat_tmp237 + weak_mat_tmp184*weak_mat_tmp242 + weak_mat_tmp43*(-weak_mat_tmp121 - weak_mat_tmp209);
    const s_t weak_mat_tmp247 = weak_mat_tmp100*weak_mat_tmp235 + weak_mat_tmp101*weak_mat_tmp240 - weak_mat_tmp133;
    const s_t weak_mat_tmp248 = weak_mat_tmp106*weak_mat_tmp237 + weak_mat_tmp109*weak_mat_tmp242 - weak_mat_tmp140 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp136 - s_t(2)*weak_mat_tmp157);
    const s_t weak_mat_tmp249 = weak_mat_tmp146 + weak_mat_tmp233*weak_mat_tmp96;
    const s_t weak_mat_tmp250 = weak_mat_tmp124*weak_mat_tmp235 + weak_mat_tmp149*weak_mat_tmp240 - weak_mat_tmp227;
    const s_t weak_mat_tmp251 = weak_mat_tmp153*weak_mat_tmp237 + weak_mat_tmp154*weak_mat_tmp242 - weak_mat_tmp229 + weak_mat_tmp43*(s_t(4)*gu2*gu7 - s_t(2)*weak_mat_tmp94);
    const s_t weak_mat_tmp252 = weak_mat_tmp158*weak_mat_tmp233 + weak_mat_tmp231;
    const s_t weak_mat_tmp253 = weak_mat_tmp117*weak_mat_tmp235 + weak_mat_tmp118*weak_mat_tmp240 + weak_mat_tmp180;
    const s_t weak_mat_tmp254 = weak_mat_tmp123*weak_mat_tmp237 + weak_mat_tmp126*weak_mat_tmp242 + weak_mat_tmp185 + weak_mat_tmp43*(-s_t(2)*weak_mat_tmp162 + s_t(4)*weak_mat_tmp6);
    const s_t weak_mat_tmp255 = weak_mat_tmp113*weak_mat_tmp233 - weak_mat_tmp188;
    const s_t weak_mat_tmp256 = weak_mat_tmp108*weak_mat_tmp235 + weak_mat_tmp164*weak_mat_tmp240 + weak_mat_tmp215;
    const s_t weak_mat_tmp257 = weak_mat_tmp169*weak_mat_tmp237 + weak_mat_tmp170*weak_mat_tmp242 + weak_mat_tmp217 + weak_mat_tmp43*(s_t(4)*gu2*weak_mat_tmp1 - s_t(2)*weak_mat_tmp111);
    const s_t weak_mat_tmp258 = weak_mat_tmp175*weak_mat_tmp233 - weak_mat_tmp219;
    const s_t weak_mat_tmp259 = weak_mat_tmp9*weak_mat_tmp96;
    const s_t weak_mat_tmp260 = weak_mat_tmp13*weak_mat_tmp259;
    const s_t weak_mat_tmp261 = weak_mat_tmp101*weak_mat_tmp17;
    const s_t weak_mat_tmp262 = weak_mat_tmp101*weak_mat_tmp35;
    const s_t weak_mat_tmp263 = s_t(2)*weak_mat_tmp30;
    const s_t weak_mat_tmp264 = weak_mat_tmp263 + weak_mat_tmp38;
    const s_t weak_mat_tmp265 = s_t(2)*weak_mat_tmp26;
    const s_t weak_mat_tmp266 = weak_mat_tmp265 + weak_mat_tmp40;
    const s_t weak_mat_tmp267 = weak_mat_tmp106*weak_mat_tmp45;
    const s_t weak_mat_tmp268 = weak_mat_tmp106*weak_mat_tmp64;
    const s_t weak_mat_tmp269 = weak_mat_tmp143*weak_mat_tmp259;
    const s_t weak_mat_tmp270 = weak_mat_tmp100*weak_mat_tmp17;
    const s_t weak_mat_tmp271 = weak_mat_tmp107*weak_mat_tmp261 + weak_mat_tmp130*weak_mat_tmp270;
    const s_t weak_mat_tmp272 = s_t(2)*weak_mat_tmp52;
    const s_t weak_mat_tmp273 = weak_mat_tmp109*weak_mat_tmp45;
    const s_t weak_mat_tmp274 = weak_mat_tmp137*weak_mat_tmp273 + weak_mat_tmp138*weak_mat_tmp267 + weak_mat_tmp43*(-weak_mat_tmp272 - weak_mat_tmp90);
    const s_t weak_mat_tmp275 = weak_mat_tmp113*weak_mat_tmp259;
    const s_t weak_mat_tmp276 = weak_mat_tmp117*weak_mat_tmp261 + weak_mat_tmp118*weak_mat_tmp270;
    const s_t weak_mat_tmp277 = weak_mat_tmp107*weak_mat_tmp2;
    const s_t weak_mat_tmp278 = gu7*weak_mat_tmp108;
    const s_t weak_mat_tmp279 = weak_mat_tmp123*weak_mat_tmp273 + weak_mat_tmp126*weak_mat_tmp267 + weak_mat_tmp43*(-weak_mat_tmp277 - weak_mat_tmp278);
    const s_t weak_mat_tmp280 = weak_mat_tmp175*weak_mat_tmp259;
    const s_t weak_mat_tmp281 = weak_mat_tmp108*weak_mat_tmp261 + weak_mat_tmp164*weak_mat_tmp270;
    const s_t weak_mat_tmp282 = s_t(2)*weak_mat_tmp47;
    const s_t weak_mat_tmp283 = weak_mat_tmp169*weak_mat_tmp273 + weak_mat_tmp170*weak_mat_tmp267 + weak_mat_tmp43*(-weak_mat_tmp282 - weak_mat_tmp75);
    const s_t weak_mat_tmp284 = gu2*weak_mat_tmp132;
    const s_t weak_mat_tmp285 = weak_mat_tmp124*weak_mat_tmp261 + weak_mat_tmp149*weak_mat_tmp270 - weak_mat_tmp284;
    const s_t weak_mat_tmp286 = gu2*weak_mat_tmp139;
    const s_t weak_mat_tmp287 = weak_mat_tmp153*weak_mat_tmp273 + weak_mat_tmp154*weak_mat_tmp267 - weak_mat_tmp286 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp12 - s_t(2)*weak_mat_tmp81);
    const s_t weak_mat_tmp288 = gu2*weak_mat_tmp145;
    const s_t weak_mat_tmp289 = weak_mat_tmp158*weak_mat_tmp259 + weak_mat_tmp288;
    const s_t weak_mat_tmp290 = gu1*weak_mat_tmp132;
    const s_t weak_mat_tmp291 = weak_mat_tmp125*weak_mat_tmp261 + weak_mat_tmp179*weak_mat_tmp270 + weak_mat_tmp290;
    const s_t weak_mat_tmp292 = gu1*weak_mat_tmp139;
    const s_t weak_mat_tmp293 = weak_mat_tmp183*weak_mat_tmp273 + weak_mat_tmp184*weak_mat_tmp267 + weak_mat_tmp292 + weak_mat_tmp43*(-s_t(2)*weak_mat_tmp11 + s_t(4)*weak_mat_tmp66);
    const s_t weak_mat_tmp294 = gu1*weak_mat_tmp145;
    const s_t weak_mat_tmp295 = weak_mat_tmp189*weak_mat_tmp259 - weak_mat_tmp294;
    const s_t weak_mat_tmp296 = weak_mat_tmp175*weak_mat_tmp9;
    const s_t weak_mat_tmp297 = weak_mat_tmp13*weak_mat_tmp296;
    const s_t weak_mat_tmp298 = weak_mat_tmp164*weak_mat_tmp17;
    const s_t weak_mat_tmp299 = weak_mat_tmp164*weak_mat_tmp35;
    const s_t weak_mat_tmp300 = s_t(2)*weak_mat_tmp24;
    const s_t weak_mat_tmp301 = weak_mat_tmp196 + weak_mat_tmp300;
    const s_t weak_mat_tmp302 = weak_mat_tmp170*weak_mat_tmp45;
    const s_t weak_mat_tmp303 = weak_mat_tmp169*weak_mat_tmp64;
    const s_t weak_mat_tmp304 = weak_mat_tmp143*weak_mat_tmp296;
    const s_t weak_mat_tmp305 = weak_mat_tmp108*weak_mat_tmp17;
    const s_t weak_mat_tmp306 = weak_mat_tmp107*weak_mat_tmp298 + weak_mat_tmp130*weak_mat_tmp305;
    const s_t weak_mat_tmp307 = s_t(2)*weak_mat_tmp59;
    const s_t weak_mat_tmp308 = weak_mat_tmp169*weak_mat_tmp45;
    const s_t weak_mat_tmp309 = weak_mat_tmp137*weak_mat_tmp302 + weak_mat_tmp138*weak_mat_tmp308 + weak_mat_tmp43*(-weak_mat_tmp204 - weak_mat_tmp307);
    const s_t weak_mat_tmp310 = weak_mat_tmp158*weak_mat_tmp296;
    const s_t weak_mat_tmp311 = weak_mat_tmp124*weak_mat_tmp298 + weak_mat_tmp149*weak_mat_tmp305;
    const s_t weak_mat_tmp312 = gu6*weak_mat_tmp100;
    const s_t weak_mat_tmp313 = weak_mat_tmp153*weak_mat_tmp302 + weak_mat_tmp154*weak_mat_tmp308 + weak_mat_tmp43*(-weak_mat_tmp277 - weak_mat_tmp312);
    const s_t weak_mat_tmp314 = weak_mat_tmp117*weak_mat_tmp298 + weak_mat_tmp118*weak_mat_tmp305 + weak_mat_tmp284;
    const s_t weak_mat_tmp315 = weak_mat_tmp123*weak_mat_tmp302 + weak_mat_tmp126*weak_mat_tmp308 + weak_mat_tmp286 + weak_mat_tmp43*(s_t(4)*gu6*weak_mat_tmp1 - s_t(2)*weak_mat_tmp12);
    const s_t weak_mat_tmp316 = weak_mat_tmp113*weak_mat_tmp296 - weak_mat_tmp288;
    const s_t weak_mat_tmp317 = weak_mat_tmp132*weak_mat_tmp7;
    const s_t weak_mat_tmp318 = weak_mat_tmp125*weak_mat_tmp298 + weak_mat_tmp179*weak_mat_tmp305 - weak_mat_tmp317;
    const s_t weak_mat_tmp319 = weak_mat_tmp139*weak_mat_tmp7;
    const s_t weak_mat_tmp320 = weak_mat_tmp183*weak_mat_tmp302 + weak_mat_tmp184*weak_mat_tmp308 - weak_mat_tmp319 + weak_mat_tmp43*(-s_t(2)*weak_mat_tmp0 + s_t(4)*weak_mat_tmp1*weak_mat_tmp2);
    const s_t weak_mat_tmp321 = weak_mat_tmp145*weak_mat_tmp7;
    const s_t weak_mat_tmp322 = weak_mat_tmp189*weak_mat_tmp296 + weak_mat_tmp321;
    const s_t weak_mat_tmp323 = weak_mat_tmp143*weak_mat_tmp9;
    const s_t weak_mat_tmp324 = weak_mat_tmp13*weak_mat_tmp323;
    const s_t weak_mat_tmp325 = weak_mat_tmp130*weak_mat_tmp17;
    const s_t weak_mat_tmp326 = weak_mat_tmp130*weak_mat_tmp35;
    const s_t weak_mat_tmp327 = weak_mat_tmp138*weak_mat_tmp45;
    const s_t weak_mat_tmp328 = weak_mat_tmp137*weak_mat_tmp64;
    const s_t weak_mat_tmp329 = weak_mat_tmp189*weak_mat_tmp323;
    const s_t weak_mat_tmp330 = weak_mat_tmp107*weak_mat_tmp17;
    const s_t weak_mat_tmp331 = weak_mat_tmp125*weak_mat_tmp325 + weak_mat_tmp179*weak_mat_tmp330;
    const s_t weak_mat_tmp332 = weak_mat_tmp137*weak_mat_tmp45;
    const s_t weak_mat_tmp333 = weak_mat_tmp183*weak_mat_tmp327 + weak_mat_tmp184*weak_mat_tmp332 + weak_mat_tmp43*(-weak_mat_tmp278 - weak_mat_tmp312);
    const s_t weak_mat_tmp334 = weak_mat_tmp117*weak_mat_tmp325 + weak_mat_tmp118*weak_mat_tmp330 - weak_mat_tmp290;
    const s_t weak_mat_tmp335 = weak_mat_tmp123*weak_mat_tmp327 + weak_mat_tmp126*weak_mat_tmp332 - weak_mat_tmp292 + weak_mat_tmp43*(s_t(4)*gu5*gu6 - s_t(2)*weak_mat_tmp66);
    const s_t weak_mat_tmp336 = weak_mat_tmp113*weak_mat_tmp323 + weak_mat_tmp294;
    const s_t weak_mat_tmp337 = weak_mat_tmp124*weak_mat_tmp325 + weak_mat_tmp149*weak_mat_tmp330 + weak_mat_tmp317;
    const s_t weak_mat_tmp338 = weak_mat_tmp153*weak_mat_tmp327 + weak_mat_tmp154*weak_mat_tmp332 + weak_mat_tmp319 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp0 - s_t(2)*weak_mat_tmp18);
    const s_t weak_mat_tmp339 = weak_mat_tmp158*weak_mat_tmp323 - weak_mat_tmp321;
    const s_t weak_mat_tmp340 = weak_mat_tmp113*weak_mat_tmp9;
    const s_t weak_mat_tmp341 = weak_mat_tmp13*weak_mat_tmp340;
    const s_t weak_mat_tmp342 = weak_mat_tmp118*weak_mat_tmp17;
    const s_t weak_mat_tmp343 = weak_mat_tmp118*weak_mat_tmp35;
    const s_t weak_mat_tmp344 = weak_mat_tmp263 + weak_mat_tmp37;
    const s_t weak_mat_tmp345 = weak_mat_tmp265 + weak_mat_tmp41;
    const s_t weak_mat_tmp346 = weak_mat_tmp123*weak_mat_tmp45;
    const s_t weak_mat_tmp347 = weak_mat_tmp123*weak_mat_tmp64;
    const s_t weak_mat_tmp348 = weak_mat_tmp158*weak_mat_tmp340;
    const s_t weak_mat_tmp349 = weak_mat_tmp117*weak_mat_tmp17;
    const s_t weak_mat_tmp350 = weak_mat_tmp124*weak_mat_tmp342 + weak_mat_tmp149*weak_mat_tmp349;
    const s_t weak_mat_tmp351 = weak_mat_tmp126*weak_mat_tmp45;
    const s_t weak_mat_tmp352 = weak_mat_tmp153*weak_mat_tmp351 + weak_mat_tmp154*weak_mat_tmp346 + weak_mat_tmp43*(-weak_mat_tmp282 - weak_mat_tmp76);
    const s_t weak_mat_tmp353 = weak_mat_tmp189*weak_mat_tmp340;
    const s_t weak_mat_tmp354 = weak_mat_tmp125*weak_mat_tmp342 + weak_mat_tmp179*weak_mat_tmp349;
    const s_t weak_mat_tmp355 = weak_mat_tmp183*weak_mat_tmp351 + weak_mat_tmp184*weak_mat_tmp346 + weak_mat_tmp43*(-weak_mat_tmp272 - weak_mat_tmp89);
    const s_t weak_mat_tmp356 = weak_mat_tmp158*weak_mat_tmp9;
    const s_t weak_mat_tmp357 = weak_mat_tmp13*weak_mat_tmp356;
    const s_t weak_mat_tmp358 = weak_mat_tmp149*weak_mat_tmp17;
    const s_t weak_mat_tmp359 = weak_mat_tmp149*weak_mat_tmp35;
    const s_t weak_mat_tmp360 = weak_mat_tmp195 + weak_mat_tmp300;
    const s_t weak_mat_tmp361 = weak_mat_tmp153*weak_mat_tmp45;
    const s_t weak_mat_tmp362 = weak_mat_tmp153*weak_mat_tmp64;
    const s_t weak_mat_tmp363 = weak_mat_tmp189*weak_mat_tmp356;
    const s_t weak_mat_tmp364 = weak_mat_tmp124*weak_mat_tmp17*weak_mat_tmp179 + weak_mat_tmp125*weak_mat_tmp358;
    const s_t weak_mat_tmp365 = weak_mat_tmp183*weak_mat_tmp45;
    const s_t weak_mat_tmp366 = weak_mat_tmp154*weak_mat_tmp365 + weak_mat_tmp184*weak_mat_tmp361 + weak_mat_tmp43*(-weak_mat_tmp203 - weak_mat_tmp307);
    const s_t weak_mat_tmp367 = weak_mat_tmp13*weak_mat_tmp189*weak_mat_tmp9;
    const s_t weak_mat_tmp368 = weak_mat_tmp179*weak_mat_tmp35;
    const s_t weak_mat_tmp369 = weak_mat_tmp183*weak_mat_tmp64;
    const s_t material0 = trial_grad0*(c1*(weak_mat_tmp16 + s_t(4)*weak_mat_tmp20*weak_mat_tmp7 + weak_mat_tmp21*weak_mat_tmp36) + c2*(weak_mat_tmp43*(weak_mat_tmp39 + weak_mat_tmp42) + s_t(2)*weak_mat_tmp44*weak_mat_tmp57 + weak_mat_tmp58*weak_mat_tmp65) + weak_mat_tmp15*weak_mat_tmp3 + pow_2(weak_mat_tmp4)*weak_mat_tmp9) + trial_grad1*(c1*(weak_mat_tmp36*weak_mat_tmp70 + weak_mat_tmp73) + c2*(weak_mat_tmp65*weak_mat_tmp74 + weak_mat_tmp80) + weak_mat_tmp15*weak_mat_tmp67 + weak_mat_tmp69) + trial_grad2*(c1*(weak_mat_tmp36*weak_mat_tmp85 + weak_mat_tmp87) + c2*(weak_mat_tmp65*weak_mat_tmp88 + weak_mat_tmp93) + weak_mat_tmp15*weak_mat_tmp84 + weak_mat_tmp83) + trial_grad3*(c1*(weak_mat_tmp102 + weak_mat_tmp36*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp65 + weak_mat_tmp110) + weak_mat_tmp15*weak_mat_tmp95 + weak_mat_tmp97) + trial_grad4*(c1*(weak_mat_tmp163*weak_mat_tmp36 + weak_mat_tmp167) + c2*(weak_mat_tmp168*weak_mat_tmp65 + weak_mat_tmp172) + weak_mat_tmp15*weak_mat_tmp173 + weak_mat_tmp176) + trial_grad5*(c1*(weak_mat_tmp129*weak_mat_tmp36 + weak_mat_tmp134) + c2*(weak_mat_tmp135*weak_mat_tmp65 + weak_mat_tmp141) + weak_mat_tmp144*weak_mat_tmp15 + weak_mat_tmp147) + trial_grad6*(c1*(weak_mat_tmp116*weak_mat_tmp36 + weak_mat_tmp119) + c2*(weak_mat_tmp120*weak_mat_tmp65 + weak_mat_tmp127) + weak_mat_tmp114 + weak_mat_tmp115*weak_mat_tmp15) + trial_grad7*(c1*(weak_mat_tmp148*weak_mat_tmp36 + weak_mat_tmp151) + c2*(weak_mat_tmp152*weak_mat_tmp65 + weak_mat_tmp156) + weak_mat_tmp15*weak_mat_tmp159 + weak_mat_tmp161) + trial_grad8*(c1*(weak_mat_tmp178*weak_mat_tmp36 + weak_mat_tmp181) + c2*(weak_mat_tmp182*weak_mat_tmp65 + weak_mat_tmp186) + weak_mat_tmp15*weak_mat_tmp187 + weak_mat_tmp190);
    const s_t material1 = trial_grad0*(c1*(weak_mat_tmp194*weak_mat_tmp21 + weak_mat_tmp73) + c2*(weak_mat_tmp199*weak_mat_tmp58 + weak_mat_tmp80) + weak_mat_tmp192*weak_mat_tmp3 + weak_mat_tmp69) + trial_grad1*(c1*(s_t(4)*gu1*weak_mat_tmp193 + weak_mat_tmp16 + weak_mat_tmp194*weak_mat_tmp70) + c2*(s_t(2)*weak_mat_tmp198*weak_mat_tmp77 + weak_mat_tmp199*weak_mat_tmp74 + weak_mat_tmp43*(weak_mat_tmp197 + weak_mat_tmp39)) + weak_mat_tmp192*weak_mat_tmp67 + pow_2(weak_mat_tmp68)*weak_mat_tmp9) + trial_grad2*(c1*(weak_mat_tmp194*weak_mat_tmp85 + weak_mat_tmp202) + c2*(weak_mat_tmp199*weak_mat_tmp88 + weak_mat_tmp206) + weak_mat_tmp192*weak_mat_tmp84 + weak_mat_tmp200) + trial_grad3*(c1*(weak_mat_tmp194*weak_mat_tmp99 + weak_mat_tmp224) + c2*(weak_mat_tmp103*weak_mat_tmp199 + weak_mat_tmp225) + weak_mat_tmp192*weak_mat_tmp95 + weak_mat_tmp226) + trial_grad4*(c1*(weak_mat_tmp163*weak_mat_tmp194 + weak_mat_tmp212) + c2*(weak_mat_tmp168*weak_mat_tmp199 + weak_mat_tmp214) + weak_mat_tmp173*weak_mat_tmp192 + weak_mat_tmp211) + trial_grad5*(c1*(weak_mat_tmp129*weak_mat_tmp194 + weak_mat_tmp216) + c2*(weak_mat_tmp135*weak_mat_tmp199 + weak_mat_tmp218) + weak_mat_tmp144*weak_mat_tmp192 + weak_mat_tmp220) + trial_grad6*(c1*(weak_mat_tmp116*weak_mat_tmp194 + weak_mat_tmp221) + c2*(weak_mat_tmp120*weak_mat_tmp199 + weak_mat_tmp222) + weak_mat_tmp115*weak_mat_tmp192 + weak_mat_tmp223) + trial_grad7*(c1*(weak_mat_tmp148*weak_mat_tmp194 + weak_mat_tmp208) + c2*(weak_mat_tmp152*weak_mat_tmp199 + weak_mat_tmp210) + weak_mat_tmp159*weak_mat_tmp192 + weak_mat_tmp207) + trial_grad8*(c1*(weak_mat_tmp178*weak_mat_tmp194 + weak_mat_tmp228) + c2*(weak_mat_tmp182*weak_mat_tmp199 + weak_mat_tmp230) + weak_mat_tmp187*weak_mat_tmp192 + weak_mat_tmp232);
    const s_t material2 = trial_grad0*(c1*(weak_mat_tmp21*weak_mat_tmp236 + weak_mat_tmp87) + c2*(weak_mat_tmp238*weak_mat_tmp58 + weak_mat_tmp93) + weak_mat_tmp234*weak_mat_tmp3 + weak_mat_tmp83) + trial_grad1*(c1*(weak_mat_tmp202 + weak_mat_tmp236*weak_mat_tmp70) + c2*(weak_mat_tmp206 + weak_mat_tmp238*weak_mat_tmp74) + weak_mat_tmp200 + weak_mat_tmp234*weak_mat_tmp67) + trial_grad2*(c1*(s_t(4)*gu2*weak_mat_tmp235 + weak_mat_tmp16 + weak_mat_tmp236*weak_mat_tmp85) + c2*(s_t(2)*weak_mat_tmp237*weak_mat_tmp91 + weak_mat_tmp238*weak_mat_tmp88 + weak_mat_tmp43*(weak_mat_tmp197 + weak_mat_tmp42)) + weak_mat_tmp234*weak_mat_tmp84 + pow_2(weak_mat_tmp82)*weak_mat_tmp9) + trial_grad3*(c1*(weak_mat_tmp236*weak_mat_tmp99 + weak_mat_tmp247) + c2*(weak_mat_tmp103*weak_mat_tmp238 + weak_mat_tmp248) + weak_mat_tmp234*weak_mat_tmp95 + weak_mat_tmp249) + trial_grad4*(c1*(weak_mat_tmp163*weak_mat_tmp236 + weak_mat_tmp256) + c2*(weak_mat_tmp168*weak_mat_tmp238 + weak_mat_tmp257) + weak_mat_tmp173*weak_mat_tmp234 + weak_mat_tmp258) + trial_grad5*(c1*(weak_mat_tmp129*weak_mat_tmp236 + weak_mat_tmp241) + c2*(weak_mat_tmp135*weak_mat_tmp238 + weak_mat_tmp243) + weak_mat_tmp144*weak_mat_tmp234 + weak_mat_tmp239) + trial_grad6*(c1*(weak_mat_tmp116*weak_mat_tmp236 + weak_mat_tmp253) + c2*(weak_mat_tmp120*weak_mat_tmp238 + weak_mat_tmp254) + weak_mat_tmp115*weak_mat_tmp234 + weak_mat_tmp255) + trial_grad7*(c1*(weak_mat_tmp148*weak_mat_tmp236 + weak_mat_tmp250) + c2*(weak_mat_tmp152*weak_mat_tmp238 + weak_mat_tmp251) + weak_mat_tmp159*weak_mat_tmp234 + weak_mat_tmp252) + trial_grad8*(c1*(weak_mat_tmp178*weak_mat_tmp236 + weak_mat_tmp245) + c2*(weak_mat_tmp182*weak_mat_tmp238 + weak_mat_tmp246) + weak_mat_tmp187*weak_mat_tmp234 + weak_mat_tmp244);
    const s_t material3 = trial_grad0*(c1*(weak_mat_tmp102 + weak_mat_tmp21*weak_mat_tmp262) + c2*(weak_mat_tmp110 + weak_mat_tmp268*weak_mat_tmp58) + weak_mat_tmp260*weak_mat_tmp3 + weak_mat_tmp97) + trial_grad1*(c1*(weak_mat_tmp224 + weak_mat_tmp262*weak_mat_tmp70) + c2*(weak_mat_tmp225 + weak_mat_tmp268*weak_mat_tmp74) + weak_mat_tmp226 + weak_mat_tmp260*weak_mat_tmp67) + trial_grad2*(c1*(weak_mat_tmp247 + weak_mat_tmp262*weak_mat_tmp85) + c2*(weak_mat_tmp248 + weak_mat_tmp268*weak_mat_tmp88) + weak_mat_tmp249 + weak_mat_tmp260*weak_mat_tmp84) + trial_grad3*(c1*(s_t(4)*gu3*weak_mat_tmp261 + weak_mat_tmp16 + weak_mat_tmp262*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp268 + s_t(2)*weak_mat_tmp109*weak_mat_tmp267 + weak_mat_tmp43*(weak_mat_tmp264 + weak_mat_tmp266)) + weak_mat_tmp260*weak_mat_tmp95 + weak_mat_tmp9*pow_2(weak_mat_tmp96)) + trial_grad4*(c1*(weak_mat_tmp163*weak_mat_tmp262 + weak_mat_tmp281) + c2*(weak_mat_tmp168*weak_mat_tmp268 + weak_mat_tmp283) + weak_mat_tmp173*weak_mat_tmp260 + weak_mat_tmp280) + trial_grad5*(c1*(weak_mat_tmp129*weak_mat_tmp262 + weak_mat_tmp271) + c2*(weak_mat_tmp135*weak_mat_tmp268 + weak_mat_tmp274) + weak_mat_tmp144*weak_mat_tmp260 + weak_mat_tmp269) + trial_grad6*(c1*(weak_mat_tmp116*weak_mat_tmp262 + weak_mat_tmp276) + c2*(weak_mat_tmp120*weak_mat_tmp268 + weak_mat_tmp279) + weak_mat_tmp115*weak_mat_tmp260 + weak_mat_tmp275) + trial_grad7*(c1*(weak_mat_tmp148*weak_mat_tmp262 + weak_mat_tmp285) + c2*(weak_mat_tmp152*weak_mat_tmp268 + weak_mat_tmp287) + weak_mat_tmp159*weak_mat_tmp260 + weak_mat_tmp289) + trial_grad8*(c1*(weak_mat_tmp178*weak_mat_tmp262 + weak_mat_tmp291) + c2*(weak_mat_tmp182*weak_mat_tmp268 + weak_mat_tmp293) + weak_mat_tmp187*weak_mat_tmp260 + weak_mat_tmp295);
    const s_t material4 = trial_grad0*(c1*(weak_mat_tmp167 + weak_mat_tmp21*weak_mat_tmp299) + c2*(weak_mat_tmp172 + weak_mat_tmp303*weak_mat_tmp58) + weak_mat_tmp176 + weak_mat_tmp297*weak_mat_tmp3) + trial_grad1*(c1*(weak_mat_tmp212 + weak_mat_tmp299*weak_mat_tmp70) + c2*(weak_mat_tmp214 + weak_mat_tmp303*weak_mat_tmp74) + weak_mat_tmp211 + weak_mat_tmp297*weak_mat_tmp67) + trial_grad2*(c1*(weak_mat_tmp256 + weak_mat_tmp299*weak_mat_tmp85) + c2*(weak_mat_tmp257 + weak_mat_tmp303*weak_mat_tmp88) + weak_mat_tmp258 + weak_mat_tmp297*weak_mat_tmp84) + trial_grad3*(c1*(weak_mat_tmp281 + weak_mat_tmp299*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp303 + weak_mat_tmp283) + weak_mat_tmp280 + weak_mat_tmp297*weak_mat_tmp95) + trial_grad4*(c1*(s_t(4)*weak_mat_tmp1*weak_mat_tmp298 + weak_mat_tmp16 + weak_mat_tmp163*weak_mat_tmp299) + c2*(weak_mat_tmp168*weak_mat_tmp303 + s_t(2)*weak_mat_tmp169*weak_mat_tmp302 + weak_mat_tmp43*(weak_mat_tmp264 + weak_mat_tmp301)) + weak_mat_tmp173*weak_mat_tmp297 + pow_2(weak_mat_tmp175)*weak_mat_tmp9) + trial_grad5*(c1*(weak_mat_tmp129*weak_mat_tmp299 + weak_mat_tmp306) + c2*(weak_mat_tmp135*weak_mat_tmp303 + weak_mat_tmp309) + weak_mat_tmp144*weak_mat_tmp297 + weak_mat_tmp304) + trial_grad6*(c1*(weak_mat_tmp116*weak_mat_tmp299 + weak_mat_tmp314) + c2*(weak_mat_tmp120*weak_mat_tmp303 + weak_mat_tmp315) + weak_mat_tmp115*weak_mat_tmp297 + weak_mat_tmp316) + trial_grad7*(c1*(weak_mat_tmp148*weak_mat_tmp299 + weak_mat_tmp311) + c2*(weak_mat_tmp152*weak_mat_tmp303 + weak_mat_tmp313) + weak_mat_tmp159*weak_mat_tmp297 + weak_mat_tmp310) + trial_grad8*(c1*(weak_mat_tmp178*weak_mat_tmp299 + weak_mat_tmp318) + c2*(weak_mat_tmp182*weak_mat_tmp303 + weak_mat_tmp320) + weak_mat_tmp187*weak_mat_tmp297 + weak_mat_tmp322);
    const s_t material5 = trial_grad0*(c1*(weak_mat_tmp134 + weak_mat_tmp21*weak_mat_tmp326) + c2*(weak_mat_tmp141 + weak_mat_tmp328*weak_mat_tmp58) + weak_mat_tmp147 + weak_mat_tmp3*weak_mat_tmp324) + trial_grad1*(c1*(weak_mat_tmp216 + weak_mat_tmp326*weak_mat_tmp70) + c2*(weak_mat_tmp218 + weak_mat_tmp328*weak_mat_tmp74) + weak_mat_tmp220 + weak_mat_tmp324*weak_mat_tmp67) + trial_grad2*(c1*(weak_mat_tmp241 + weak_mat_tmp326*weak_mat_tmp85) + c2*(weak_mat_tmp243 + weak_mat_tmp328*weak_mat_tmp88) + weak_mat_tmp239 + weak_mat_tmp324*weak_mat_tmp84) + trial_grad3*(c1*(weak_mat_tmp271 + weak_mat_tmp326*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp328 + weak_mat_tmp274) + weak_mat_tmp269 + weak_mat_tmp324*weak_mat_tmp95) + trial_grad4*(c1*(weak_mat_tmp163*weak_mat_tmp326 + weak_mat_tmp306) + c2*(weak_mat_tmp168*weak_mat_tmp328 + weak_mat_tmp309) + weak_mat_tmp173*weak_mat_tmp324 + weak_mat_tmp304) + trial_grad5*(c1*(s_t(4)*gu5*weak_mat_tmp325 + weak_mat_tmp129*weak_mat_tmp326 + weak_mat_tmp16) + c2*(weak_mat_tmp135*weak_mat_tmp328 + s_t(2)*weak_mat_tmp137*weak_mat_tmp327 + weak_mat_tmp43*(weak_mat_tmp266 + weak_mat_tmp301)) + pow_2(weak_mat_tmp143)*weak_mat_tmp9 + weak_mat_tmp144*weak_mat_tmp324) + trial_grad6*(c1*(weak_mat_tmp116*weak_mat_tmp326 + weak_mat_tmp334) + c2*(weak_mat_tmp120*weak_mat_tmp328 + weak_mat_tmp335) + weak_mat_tmp115*weak_mat_tmp324 + weak_mat_tmp336) + trial_grad7*(c1*(weak_mat_tmp148*weak_mat_tmp326 + weak_mat_tmp337) + c2*(weak_mat_tmp152*weak_mat_tmp328 + weak_mat_tmp338) + weak_mat_tmp159*weak_mat_tmp324 + weak_mat_tmp339) + trial_grad8*(c1*(weak_mat_tmp178*weak_mat_tmp326 + weak_mat_tmp331) + c2*(weak_mat_tmp182*weak_mat_tmp328 + weak_mat_tmp333) + weak_mat_tmp187*weak_mat_tmp324 + weak_mat_tmp329);
    const s_t material6 = trial_grad0*(c1*(weak_mat_tmp119 + weak_mat_tmp21*weak_mat_tmp343) + c2*(weak_mat_tmp127 + weak_mat_tmp347*weak_mat_tmp58) + weak_mat_tmp114 + weak_mat_tmp3*weak_mat_tmp341) + trial_grad1*(c1*(weak_mat_tmp221 + weak_mat_tmp343*weak_mat_tmp70) + c2*(weak_mat_tmp222 + weak_mat_tmp347*weak_mat_tmp74) + weak_mat_tmp223 + weak_mat_tmp341*weak_mat_tmp67) + trial_grad2*(c1*(weak_mat_tmp253 + weak_mat_tmp343*weak_mat_tmp85) + c2*(weak_mat_tmp254 + weak_mat_tmp347*weak_mat_tmp88) + weak_mat_tmp255 + weak_mat_tmp341*weak_mat_tmp84) + trial_grad3*(c1*(weak_mat_tmp276 + weak_mat_tmp343*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp347 + weak_mat_tmp279) + weak_mat_tmp275 + weak_mat_tmp341*weak_mat_tmp95) + trial_grad4*(c1*(weak_mat_tmp163*weak_mat_tmp343 + weak_mat_tmp314) + c2*(weak_mat_tmp168*weak_mat_tmp347 + weak_mat_tmp315) + weak_mat_tmp173*weak_mat_tmp341 + weak_mat_tmp316) + trial_grad5*(c1*(weak_mat_tmp129*weak_mat_tmp343 + weak_mat_tmp334) + c2*(weak_mat_tmp135*weak_mat_tmp347 + weak_mat_tmp335) + weak_mat_tmp144*weak_mat_tmp341 + weak_mat_tmp336) + trial_grad6*(c1*(s_t(4)*gu6*weak_mat_tmp342 + weak_mat_tmp116*weak_mat_tmp343 + weak_mat_tmp16) + c2*(weak_mat_tmp120*weak_mat_tmp347 + s_t(2)*weak_mat_tmp126*weak_mat_tmp346 + weak_mat_tmp43*(weak_mat_tmp344 + weak_mat_tmp345)) + pow_2(weak_mat_tmp113)*weak_mat_tmp9 + weak_mat_tmp115*weak_mat_tmp341) + trial_grad7*(c1*(weak_mat_tmp148*weak_mat_tmp343 + weak_mat_tmp350) + c2*(weak_mat_tmp152*weak_mat_tmp347 + weak_mat_tmp352) + weak_mat_tmp159*weak_mat_tmp341 + weak_mat_tmp348) + trial_grad8*(c1*(weak_mat_tmp178*weak_mat_tmp343 + weak_mat_tmp354) + c2*(weak_mat_tmp182*weak_mat_tmp347 + weak_mat_tmp355) + weak_mat_tmp187*weak_mat_tmp341 + weak_mat_tmp353);
    const s_t material7 = trial_grad0*(c1*(weak_mat_tmp151 + weak_mat_tmp21*weak_mat_tmp359) + c2*(weak_mat_tmp156 + weak_mat_tmp362*weak_mat_tmp58) + weak_mat_tmp161 + weak_mat_tmp3*weak_mat_tmp357) + trial_grad1*(c1*(weak_mat_tmp208 + weak_mat_tmp359*weak_mat_tmp70) + c2*(weak_mat_tmp210 + weak_mat_tmp362*weak_mat_tmp74) + weak_mat_tmp207 + weak_mat_tmp357*weak_mat_tmp67) + trial_grad2*(c1*(weak_mat_tmp250 + weak_mat_tmp359*weak_mat_tmp85) + c2*(weak_mat_tmp251 + weak_mat_tmp362*weak_mat_tmp88) + weak_mat_tmp252 + weak_mat_tmp357*weak_mat_tmp84) + trial_grad3*(c1*(weak_mat_tmp285 + weak_mat_tmp359*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp362 + weak_mat_tmp287) + weak_mat_tmp289 + weak_mat_tmp357*weak_mat_tmp95) + trial_grad4*(c1*(weak_mat_tmp163*weak_mat_tmp359 + weak_mat_tmp311) + c2*(weak_mat_tmp168*weak_mat_tmp362 + weak_mat_tmp313) + weak_mat_tmp173*weak_mat_tmp357 + weak_mat_tmp310) + trial_grad5*(c1*(weak_mat_tmp129*weak_mat_tmp359 + weak_mat_tmp337) + c2*(weak_mat_tmp135*weak_mat_tmp362 + weak_mat_tmp338) + weak_mat_tmp144*weak_mat_tmp357 + weak_mat_tmp339) + trial_grad6*(c1*(weak_mat_tmp116*weak_mat_tmp359 + weak_mat_tmp350) + c2*(weak_mat_tmp120*weak_mat_tmp362 + weak_mat_tmp352) + weak_mat_tmp115*weak_mat_tmp357 + weak_mat_tmp348) + trial_grad7*(c1*(s_t(4)*gu7*weak_mat_tmp358 + weak_mat_tmp148*weak_mat_tmp359 + weak_mat_tmp16) + c2*(weak_mat_tmp152*weak_mat_tmp362 + s_t(2)*weak_mat_tmp154*weak_mat_tmp361 + weak_mat_tmp43*(weak_mat_tmp344 + weak_mat_tmp360)) + pow_2(weak_mat_tmp158)*weak_mat_tmp9 + weak_mat_tmp159*weak_mat_tmp357) + trial_grad8*(c1*(weak_mat_tmp178*weak_mat_tmp359 + weak_mat_tmp364) + c2*(weak_mat_tmp182*weak_mat_tmp362 + weak_mat_tmp366) + weak_mat_tmp187*weak_mat_tmp357 + weak_mat_tmp363);
    const s_t material8 = trial_grad0*(c1*(weak_mat_tmp181 + weak_mat_tmp21*weak_mat_tmp368) + c2*(weak_mat_tmp186 + weak_mat_tmp369*weak_mat_tmp58) + weak_mat_tmp190 + weak_mat_tmp3*weak_mat_tmp367) + trial_grad1*(c1*(weak_mat_tmp228 + weak_mat_tmp368*weak_mat_tmp70) + c2*(weak_mat_tmp230 + weak_mat_tmp369*weak_mat_tmp74) + weak_mat_tmp232 + weak_mat_tmp367*weak_mat_tmp67) + trial_grad2*(c1*(weak_mat_tmp245 + weak_mat_tmp368*weak_mat_tmp85) + c2*(weak_mat_tmp246 + weak_mat_tmp369*weak_mat_tmp88) + weak_mat_tmp244 + weak_mat_tmp367*weak_mat_tmp84) + trial_grad3*(c1*(weak_mat_tmp291 + weak_mat_tmp368*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp369 + weak_mat_tmp293) + weak_mat_tmp295 + weak_mat_tmp367*weak_mat_tmp95) + trial_grad4*(c1*(weak_mat_tmp163*weak_mat_tmp368 + weak_mat_tmp318) + c2*(weak_mat_tmp168*weak_mat_tmp369 + weak_mat_tmp320) + weak_mat_tmp173*weak_mat_tmp367 + weak_mat_tmp322) + trial_grad5*(c1*(weak_mat_tmp129*weak_mat_tmp368 + weak_mat_tmp331) + c2*(weak_mat_tmp135*weak_mat_tmp369 + weak_mat_tmp333) + weak_mat_tmp144*weak_mat_tmp367 + weak_mat_tmp329) + trial_grad6*(c1*(weak_mat_tmp116*weak_mat_tmp368 + weak_mat_tmp354) + c2*(weak_mat_tmp120*weak_mat_tmp369 + weak_mat_tmp355) + weak_mat_tmp115*weak_mat_tmp367 + weak_mat_tmp353) + trial_grad7*(c1*(weak_mat_tmp148*weak_mat_tmp368 + weak_mat_tmp364) + c2*(weak_mat_tmp152*weak_mat_tmp369 + weak_mat_tmp366) + weak_mat_tmp159*weak_mat_tmp367 + weak_mat_tmp363) + trial_grad8*(c1*(weak_mat_tmp16 + s_t(4)*weak_mat_tmp165*weak_mat_tmp179 + weak_mat_tmp178*weak_mat_tmp368) + c2*(weak_mat_tmp182*weak_mat_tmp369 + s_t(2)*weak_mat_tmp184*weak_mat_tmp365 + weak_mat_tmp43*(weak_mat_tmp345 + weak_mat_tmp360)) + weak_mat_tmp187*weak_mat_tmp367 + pow_2(weak_mat_tmp189)*weak_mat_tmp9);
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

} // namespace codegen
} // namespace sfem

#endif
