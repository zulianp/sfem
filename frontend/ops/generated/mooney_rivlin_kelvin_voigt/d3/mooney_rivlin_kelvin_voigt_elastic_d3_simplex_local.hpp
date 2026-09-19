#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_ELASTIC_D3_SIMPLEX_LOCAL_HPP
#define MOONEY_RIVLIN_KELVIN_VOIGT_ELASTIC_D3_SIMPLEX_LOCAL_HPP
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
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_elastic_d3_simplex_objective_block(
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
    const s_t weak_obj_tmp4 = gu1*gu5*gu6;
    const s_t weak_obj_tmp5 = gu2*gu3*gu7;
    const s_t weak_obj_tmp6 = gu4 + s_t(1);
    const s_t weak_obj_tmp7 = pow_2(gu1) + pow_2(gu7) + pow_2(weak_obj_tmp6);
    const s_t weak_obj_tmp8 = gu8 + s_t(1);
    const s_t weak_obj_tmp9 = pow_2(gu2) + pow_2(gu5) + pow_2(weak_obj_tmp8);
    const s_t weak_obj_tmp10 = gu0 + s_t(1);
    const s_t weak_obj_tmp11 = pow_2(gu3) + pow_2(gu6) + pow_2(weak_obj_tmp10);
    const s_t weak_obj_tmp12 = weak_obj_tmp11 + weak_obj_tmp7 + weak_obj_tmp9;
    const s_t weak_obj_tmp13 = s_t(6)*weak_obj_tmp8;
    value[step * value_stride + lane] += qw * det_lane0 * (((s_t(1) / s_t(2)))*lmbda*pow_2(gu0*gu8 - gu0*weak_obj_tmp3 + gu0 + gu4*gu8 - gu4*weak_obj_tmp2 + gu4 + gu8*weak_obj_tmp0 - gu8*weak_obj_tmp1 + gu8 + weak_obj_tmp0 - weak_obj_tmp1 - weak_obj_tmp2 - weak_obj_tmp3 + weak_obj_tmp4 + weak_obj_tmp5) + mu*(weak_obj_tmp1*weak_obj_tmp13 - weak_obj_tmp10*weak_obj_tmp13*weak_obj_tmp6 + s_t(6)*weak_obj_tmp10*weak_obj_tmp3 - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp11) + ((s_t(1) / s_t(2)))*pow_2(weak_obj_tmp12) + weak_obj_tmp12 + s_t(6)*weak_obj_tmp2*weak_obj_tmp6 - s_t(6)*weak_obj_tmp4 - s_t(6)*weak_obj_tmp5 - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp7) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp9) - pow_2(gu1*gu2 + gu5*weak_obj_tmp6 + gu7*weak_obj_tmp8) - pow_2(gu1*weak_obj_tmp10 + gu3*weak_obj_tmp6 + gu6*gu7) - pow_2(gu2*weak_obj_tmp10 + gu3*gu5 + gu6*weak_obj_tmp8)));
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_elastic_d3_simplex_tet4_objective_block(
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
    const s_t weak_obj_tmp4 = gu1*gu5*gu6;
    const s_t weak_obj_tmp5 = gu2*gu3*gu7;
    const s_t weak_obj_tmp6 = gu4 + s_t(1);
    const s_t weak_obj_tmp7 = pow_2(gu1) + pow_2(gu7) + pow_2(weak_obj_tmp6);
    const s_t weak_obj_tmp8 = gu8 + s_t(1);
    const s_t weak_obj_tmp9 = pow_2(gu2) + pow_2(gu5) + pow_2(weak_obj_tmp8);
    const s_t weak_obj_tmp10 = gu0 + s_t(1);
    const s_t weak_obj_tmp11 = pow_2(gu3) + pow_2(gu6) + pow_2(weak_obj_tmp10);
    const s_t weak_obj_tmp12 = weak_obj_tmp11 + weak_obj_tmp7 + weak_obj_tmp9;
    const s_t weak_obj_tmp13 = s_t(6)*weak_obj_tmp8;
    value[step * value_stride + lane] += qw * det_lane0 * (((s_t(1) / s_t(2)))*lmbda*pow_2(gu0*gu8 - gu0*weak_obj_tmp3 + gu0 + gu4*gu8 - gu4*weak_obj_tmp2 + gu4 + gu8*weak_obj_tmp0 - gu8*weak_obj_tmp1 + gu8 + weak_obj_tmp0 - weak_obj_tmp1 - weak_obj_tmp2 - weak_obj_tmp3 + weak_obj_tmp4 + weak_obj_tmp5) + mu*(weak_obj_tmp1*weak_obj_tmp13 - weak_obj_tmp10*weak_obj_tmp13*weak_obj_tmp6 + s_t(6)*weak_obj_tmp10*weak_obj_tmp3 - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp11) + ((s_t(1) / s_t(2)))*pow_2(weak_obj_tmp12) + weak_obj_tmp12 + s_t(6)*weak_obj_tmp2*weak_obj_tmp6 - s_t(6)*weak_obj_tmp4 - s_t(6)*weak_obj_tmp5 - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp7) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp9) - pow_2(gu1*gu2 + gu5*weak_obj_tmp6 + gu7*weak_obj_tmp8) - pow_2(gu1*weak_obj_tmp10 + gu3*weak_obj_tmp6 + gu6*gu7) - pow_2(gu2*weak_obj_tmp10 + gu3*gu5 + gu6*weak_obj_tmp8)));
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_elastic_d3_simplex_gradient_block(
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
        const s_t lmbda,
        const s_t mu,
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
    const s_t weak_mat_tmp1 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp2 = gu4 + s_t(1);
    const s_t weak_mat_tmp3 = gu8 + s_t(1);
    const s_t weak_mat_tmp4 = gu0*gu4;
    const s_t weak_mat_tmp5 = gu0*gu8;
    const s_t weak_mat_tmp6 = gu4*gu8;
    const s_t weak_mat_tmp7 = gu1*gu3;
    const s_t weak_mat_tmp8 = gu2*gu6;
    const s_t weak_mat_tmp9 = gu5*gu6;
    const s_t weak_mat_tmp10 = gu3*gu7;
    const s_t weak_mat_tmp11 = ((s_t(1) / s_t(2)))*lmbda*(-gu0*weak_mat_tmp0 + gu0 + gu1*weak_mat_tmp9 + gu2*weak_mat_tmp10 - gu4*weak_mat_tmp8 + gu4 + gu8*weak_mat_tmp4 - gu8*weak_mat_tmp7 + gu8 - weak_mat_tmp0 + weak_mat_tmp4 + weak_mat_tmp5 + weak_mat_tmp6 - weak_mat_tmp7 - weak_mat_tmp8);
    const s_t weak_mat_tmp12 = pow_2(gu5);
    const s_t weak_mat_tmp13 = s_t(2)*weak_mat_tmp12;
    const s_t weak_mat_tmp14 = pow_2(gu7);
    const s_t weak_mat_tmp15 = s_t(2)*weak_mat_tmp14;
    const s_t weak_mat_tmp16 = s_t(6)*gu0;
    const s_t weak_mat_tmp17 = pow_2(gu4);
    const s_t weak_mat_tmp18 = s_t(2)*weak_mat_tmp17;
    const s_t weak_mat_tmp19 = pow_2(gu8);
    const s_t weak_mat_tmp20 = s_t(2)*weak_mat_tmp19;
    const s_t weak_mat_tmp21 = s_t(2)*gu4;
    const s_t weak_mat_tmp22 = s_t(2)*gu1;
    const s_t weak_mat_tmp23 = gu6*gu7;
    const s_t weak_mat_tmp24 = s_t(2)*gu2;
    const s_t weak_mat_tmp25 = gu3*gu5;
    const s_t weak_mat_tmp26 = s_t(2)*gu8;
    const s_t weak_mat_tmp27 = s_t(2)*weak_mat_tmp7;
    const s_t weak_mat_tmp28 = weak_mat_tmp20 - weak_mat_tmp26 - weak_mat_tmp27 + s_t(4)*weak_mat_tmp4;
    const s_t weak_mat_tmp29 = s_t(2)*weak_mat_tmp8;
    const s_t weak_mat_tmp30 = weak_mat_tmp18 - weak_mat_tmp21 - weak_mat_tmp29 + s_t(4)*weak_mat_tmp5;
    const s_t weak_mat_tmp31 = s_t(2)*gu3;
    const s_t weak_mat_tmp32 = pow_2(gu1);
    const s_t weak_mat_tmp33 = weak_mat_tmp14 + pow_2(weak_mat_tmp2) + weak_mat_tmp32;
    const s_t weak_mat_tmp34 = gu1*gu2 + gu5*weak_mat_tmp2 + gu7*weak_mat_tmp3;
    const s_t weak_mat_tmp35 = gu0 + s_t(1);
    const s_t weak_mat_tmp36 = gu1*weak_mat_tmp35 + gu3*weak_mat_tmp2 + weak_mat_tmp23;
    const s_t weak_mat_tmp37 = s_t(2)*weak_mat_tmp35;
    const s_t weak_mat_tmp38 = pow_2(gu2);
    const s_t weak_mat_tmp39 = weak_mat_tmp12 + pow_2(weak_mat_tmp3) + weak_mat_tmp38;
    const s_t weak_mat_tmp40 = pow_2(gu3);
    const s_t weak_mat_tmp41 = pow_2(gu6);
    const s_t weak_mat_tmp42 = pow_2(weak_mat_tmp35) + weak_mat_tmp40 + weak_mat_tmp41;
    const s_t weak_mat_tmp43 = weak_mat_tmp33 + weak_mat_tmp39 + weak_mat_tmp42;
    const s_t weak_mat_tmp44 = s_t(2)*gu6;
    const s_t weak_mat_tmp45 = gu2*weak_mat_tmp35 + gu6*weak_mat_tmp3 + weak_mat_tmp25;
    const s_t weak_mat_tmp46 = s_t(2)*gu5;
    const s_t weak_mat_tmp47 = s_t(2)*weak_mat_tmp2;
    const s_t weak_mat_tmp48 = pow_2(gu0);
    const s_t weak_mat_tmp49 = s_t(2)*gu0;
    const s_t weak_mat_tmp50 = gu5*weak_mat_tmp22;
    const s_t weak_mat_tmp51 = -weak_mat_tmp1 + s_t(2)*weak_mat_tmp48 - weak_mat_tmp49 + s_t(4)*weak_mat_tmp6;
    const s_t weak_mat_tmp52 = s_t(2)*gu7;
    const s_t weak_mat_tmp53 = s_t(2)*weak_mat_tmp3;
    const s_t material0 = mu*(gu0*weak_mat_tmp13 + gu0*weak_mat_tmp15 + gu0*weak_mat_tmp18 + gu0*weak_mat_tmp20 + s_t(6)*weak_mat_tmp0 + weak_mat_tmp13 + weak_mat_tmp15 + weak_mat_tmp16 - weak_mat_tmp21*weak_mat_tmp7 - weak_mat_tmp22*weak_mat_tmp23 - weak_mat_tmp24*weak_mat_tmp25 - weak_mat_tmp26*weak_mat_tmp8 + weak_mat_tmp28 + weak_mat_tmp30 - s_t(6)*weak_mat_tmp6) + weak_mat_tmp11*(-weak_mat_tmp1 + s_t(2)*weak_mat_tmp2*weak_mat_tmp3);
    const s_t material1 = mu*(s_t(2)*gu1*weak_mat_tmp43 + s_t(2)*gu1 + s_t(6)*gu3*weak_mat_tmp3 - weak_mat_tmp22*weak_mat_tmp33 - weak_mat_tmp24*weak_mat_tmp34 - weak_mat_tmp36*weak_mat_tmp37 - s_t(6)*weak_mat_tmp9) + weak_mat_tmp11*(s_t(2)*gu5*gu6 - weak_mat_tmp3*weak_mat_tmp31);
    const s_t material2 = mu*(s_t(2)*gu2*weak_mat_tmp43 + s_t(2)*gu2 + s_t(6)*gu6*weak_mat_tmp2 - s_t(6)*weak_mat_tmp10 - weak_mat_tmp22*weak_mat_tmp34 - weak_mat_tmp24*weak_mat_tmp39 - weak_mat_tmp37*weak_mat_tmp45) + weak_mat_tmp11*(gu7*weak_mat_tmp31 - weak_mat_tmp2*weak_mat_tmp44);
    const s_t material3 = mu*(s_t(6)*gu1*weak_mat_tmp3 - s_t(6)*gu2*gu7 + s_t(2)*gu3*weak_mat_tmp43 + s_t(2)*gu3 - weak_mat_tmp31*weak_mat_tmp42 - weak_mat_tmp36*weak_mat_tmp47 - weak_mat_tmp45*weak_mat_tmp46) + weak_mat_tmp11*(s_t(2)*gu2*gu7 - weak_mat_tmp22*weak_mat_tmp3);
    const s_t material4 = mu*(-gu2*weak_mat_tmp50 + s_t(6)*gu4 - gu8*weak_mat_tmp16 - weak_mat_tmp0*weak_mat_tmp26 + weak_mat_tmp19*weak_mat_tmp21 + weak_mat_tmp21*weak_mat_tmp38 + weak_mat_tmp21*weak_mat_tmp41 + weak_mat_tmp21*weak_mat_tmp48 - weak_mat_tmp23*weak_mat_tmp31 + weak_mat_tmp28 + s_t(2)*weak_mat_tmp38 + s_t(2)*weak_mat_tmp41 - weak_mat_tmp49*weak_mat_tmp7 + weak_mat_tmp51 + s_t(6)*weak_mat_tmp8) + weak_mat_tmp11*(-weak_mat_tmp29 + s_t(2)*weak_mat_tmp3*weak_mat_tmp35);
    const s_t material5 = mu*(-s_t(6)*gu1*gu6 + s_t(2)*gu5*weak_mat_tmp43 + s_t(2)*gu5 + s_t(6)*gu7*weak_mat_tmp35 - weak_mat_tmp31*weak_mat_tmp45 - weak_mat_tmp34*weak_mat_tmp47 - weak_mat_tmp39*weak_mat_tmp46) + weak_mat_tmp11*(gu6*weak_mat_tmp22 - weak_mat_tmp35*weak_mat_tmp52);
    const s_t material6 = mu*(-s_t(6)*gu1*gu5 + s_t(6)*gu2*weak_mat_tmp2 + s_t(2)*gu6*weak_mat_tmp43 + s_t(2)*gu6 - weak_mat_tmp36*weak_mat_tmp52 - weak_mat_tmp42*weak_mat_tmp44 - weak_mat_tmp45*weak_mat_tmp53) + weak_mat_tmp11*(-weak_mat_tmp2*weak_mat_tmp24 + weak_mat_tmp50);
    const s_t material7 = mu*(-s_t(6)*gu2*gu3 + s_t(6)*gu5*weak_mat_tmp35 + s_t(2)*gu7*weak_mat_tmp43 + s_t(2)*gu7 - weak_mat_tmp33*weak_mat_tmp52 - weak_mat_tmp34*weak_mat_tmp53 - weak_mat_tmp36*weak_mat_tmp44) + weak_mat_tmp11*(gu3*weak_mat_tmp24 - weak_mat_tmp35*weak_mat_tmp46);
    const s_t material8 = mu*(-gu2*gu7*weak_mat_tmp22 - gu4*weak_mat_tmp16 + s_t(6)*gu8 - weak_mat_tmp0*weak_mat_tmp21 + weak_mat_tmp17*weak_mat_tmp26 - weak_mat_tmp25*weak_mat_tmp44 + weak_mat_tmp26*weak_mat_tmp32 + weak_mat_tmp26*weak_mat_tmp40 + weak_mat_tmp26*weak_mat_tmp48 + weak_mat_tmp30 + s_t(2)*weak_mat_tmp32 + s_t(2)*weak_mat_tmp40 - weak_mat_tmp49*weak_mat_tmp8 + weak_mat_tmp51 + s_t(6)*weak_mat_tmp7) + weak_mat_tmp11*(s_t(2)*weak_mat_tmp2*weak_mat_tmp35 - weak_mat_tmp27);
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
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_elastic_d3_simplex_tet4_gradient_block(
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
        const s_t lmbda,
        const s_t mu,
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
    const s_t weak_mat_tmp1 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp2 = gu4 + s_t(1);
    const s_t weak_mat_tmp3 = gu8 + s_t(1);
    const s_t weak_mat_tmp4 = gu0*gu4;
    const s_t weak_mat_tmp5 = gu0*gu8;
    const s_t weak_mat_tmp6 = gu4*gu8;
    const s_t weak_mat_tmp7 = gu1*gu3;
    const s_t weak_mat_tmp8 = gu2*gu6;
    const s_t weak_mat_tmp9 = gu5*gu6;
    const s_t weak_mat_tmp10 = gu3*gu7;
    const s_t weak_mat_tmp11 = ((s_t(1) / s_t(2)))*lmbda*(-gu0*weak_mat_tmp0 + gu0 + gu1*weak_mat_tmp9 + gu2*weak_mat_tmp10 - gu4*weak_mat_tmp8 + gu4 + gu8*weak_mat_tmp4 - gu8*weak_mat_tmp7 + gu8 - weak_mat_tmp0 + weak_mat_tmp4 + weak_mat_tmp5 + weak_mat_tmp6 - weak_mat_tmp7 - weak_mat_tmp8);
    const s_t weak_mat_tmp12 = pow_2(gu5);
    const s_t weak_mat_tmp13 = s_t(2)*weak_mat_tmp12;
    const s_t weak_mat_tmp14 = pow_2(gu7);
    const s_t weak_mat_tmp15 = s_t(2)*weak_mat_tmp14;
    const s_t weak_mat_tmp16 = s_t(6)*gu0;
    const s_t weak_mat_tmp17 = pow_2(gu4);
    const s_t weak_mat_tmp18 = s_t(2)*weak_mat_tmp17;
    const s_t weak_mat_tmp19 = pow_2(gu8);
    const s_t weak_mat_tmp20 = s_t(2)*weak_mat_tmp19;
    const s_t weak_mat_tmp21 = s_t(2)*gu4;
    const s_t weak_mat_tmp22 = s_t(2)*gu1;
    const s_t weak_mat_tmp23 = gu6*gu7;
    const s_t weak_mat_tmp24 = s_t(2)*gu2;
    const s_t weak_mat_tmp25 = gu3*gu5;
    const s_t weak_mat_tmp26 = s_t(2)*gu8;
    const s_t weak_mat_tmp27 = s_t(2)*weak_mat_tmp7;
    const s_t weak_mat_tmp28 = weak_mat_tmp20 - weak_mat_tmp26 - weak_mat_tmp27 + s_t(4)*weak_mat_tmp4;
    const s_t weak_mat_tmp29 = s_t(2)*weak_mat_tmp8;
    const s_t weak_mat_tmp30 = weak_mat_tmp18 - weak_mat_tmp21 - weak_mat_tmp29 + s_t(4)*weak_mat_tmp5;
    const s_t weak_mat_tmp31 = s_t(2)*gu3;
    const s_t weak_mat_tmp32 = pow_2(gu1);
    const s_t weak_mat_tmp33 = weak_mat_tmp14 + pow_2(weak_mat_tmp2) + weak_mat_tmp32;
    const s_t weak_mat_tmp34 = gu1*gu2 + gu5*weak_mat_tmp2 + gu7*weak_mat_tmp3;
    const s_t weak_mat_tmp35 = gu0 + s_t(1);
    const s_t weak_mat_tmp36 = gu1*weak_mat_tmp35 + gu3*weak_mat_tmp2 + weak_mat_tmp23;
    const s_t weak_mat_tmp37 = s_t(2)*weak_mat_tmp35;
    const s_t weak_mat_tmp38 = pow_2(gu2);
    const s_t weak_mat_tmp39 = weak_mat_tmp12 + pow_2(weak_mat_tmp3) + weak_mat_tmp38;
    const s_t weak_mat_tmp40 = pow_2(gu3);
    const s_t weak_mat_tmp41 = pow_2(gu6);
    const s_t weak_mat_tmp42 = pow_2(weak_mat_tmp35) + weak_mat_tmp40 + weak_mat_tmp41;
    const s_t weak_mat_tmp43 = weak_mat_tmp33 + weak_mat_tmp39 + weak_mat_tmp42;
    const s_t weak_mat_tmp44 = s_t(2)*gu6;
    const s_t weak_mat_tmp45 = gu2*weak_mat_tmp35 + gu6*weak_mat_tmp3 + weak_mat_tmp25;
    const s_t weak_mat_tmp46 = s_t(2)*gu5;
    const s_t weak_mat_tmp47 = s_t(2)*weak_mat_tmp2;
    const s_t weak_mat_tmp48 = pow_2(gu0);
    const s_t weak_mat_tmp49 = s_t(2)*gu0;
    const s_t weak_mat_tmp50 = gu5*weak_mat_tmp22;
    const s_t weak_mat_tmp51 = -weak_mat_tmp1 + s_t(2)*weak_mat_tmp48 - weak_mat_tmp49 + s_t(4)*weak_mat_tmp6;
    const s_t weak_mat_tmp52 = s_t(2)*gu7;
    const s_t weak_mat_tmp53 = s_t(2)*weak_mat_tmp3;
    const s_t material0 = mu*(gu0*weak_mat_tmp13 + gu0*weak_mat_tmp15 + gu0*weak_mat_tmp18 + gu0*weak_mat_tmp20 + s_t(6)*weak_mat_tmp0 + weak_mat_tmp13 + weak_mat_tmp15 + weak_mat_tmp16 - weak_mat_tmp21*weak_mat_tmp7 - weak_mat_tmp22*weak_mat_tmp23 - weak_mat_tmp24*weak_mat_tmp25 - weak_mat_tmp26*weak_mat_tmp8 + weak_mat_tmp28 + weak_mat_tmp30 - s_t(6)*weak_mat_tmp6) + weak_mat_tmp11*(-weak_mat_tmp1 + s_t(2)*weak_mat_tmp2*weak_mat_tmp3);
    const s_t material1 = mu*(s_t(2)*gu1*weak_mat_tmp43 + s_t(2)*gu1 + s_t(6)*gu3*weak_mat_tmp3 - weak_mat_tmp22*weak_mat_tmp33 - weak_mat_tmp24*weak_mat_tmp34 - weak_mat_tmp36*weak_mat_tmp37 - s_t(6)*weak_mat_tmp9) + weak_mat_tmp11*(s_t(2)*gu5*gu6 - weak_mat_tmp3*weak_mat_tmp31);
    const s_t material2 = mu*(s_t(2)*gu2*weak_mat_tmp43 + s_t(2)*gu2 + s_t(6)*gu6*weak_mat_tmp2 - s_t(6)*weak_mat_tmp10 - weak_mat_tmp22*weak_mat_tmp34 - weak_mat_tmp24*weak_mat_tmp39 - weak_mat_tmp37*weak_mat_tmp45) + weak_mat_tmp11*(gu7*weak_mat_tmp31 - weak_mat_tmp2*weak_mat_tmp44);
    const s_t material3 = mu*(s_t(6)*gu1*weak_mat_tmp3 - s_t(6)*gu2*gu7 + s_t(2)*gu3*weak_mat_tmp43 + s_t(2)*gu3 - weak_mat_tmp31*weak_mat_tmp42 - weak_mat_tmp36*weak_mat_tmp47 - weak_mat_tmp45*weak_mat_tmp46) + weak_mat_tmp11*(s_t(2)*gu2*gu7 - weak_mat_tmp22*weak_mat_tmp3);
    const s_t material4 = mu*(-gu2*weak_mat_tmp50 + s_t(6)*gu4 - gu8*weak_mat_tmp16 - weak_mat_tmp0*weak_mat_tmp26 + weak_mat_tmp19*weak_mat_tmp21 + weak_mat_tmp21*weak_mat_tmp38 + weak_mat_tmp21*weak_mat_tmp41 + weak_mat_tmp21*weak_mat_tmp48 - weak_mat_tmp23*weak_mat_tmp31 + weak_mat_tmp28 + s_t(2)*weak_mat_tmp38 + s_t(2)*weak_mat_tmp41 - weak_mat_tmp49*weak_mat_tmp7 + weak_mat_tmp51 + s_t(6)*weak_mat_tmp8) + weak_mat_tmp11*(-weak_mat_tmp29 + s_t(2)*weak_mat_tmp3*weak_mat_tmp35);
    const s_t material5 = mu*(-s_t(6)*gu1*gu6 + s_t(2)*gu5*weak_mat_tmp43 + s_t(2)*gu5 + s_t(6)*gu7*weak_mat_tmp35 - weak_mat_tmp31*weak_mat_tmp45 - weak_mat_tmp34*weak_mat_tmp47 - weak_mat_tmp39*weak_mat_tmp46) + weak_mat_tmp11*(gu6*weak_mat_tmp22 - weak_mat_tmp35*weak_mat_tmp52);
    const s_t material6 = mu*(-s_t(6)*gu1*gu5 + s_t(6)*gu2*weak_mat_tmp2 + s_t(2)*gu6*weak_mat_tmp43 + s_t(2)*gu6 - weak_mat_tmp36*weak_mat_tmp52 - weak_mat_tmp42*weak_mat_tmp44 - weak_mat_tmp45*weak_mat_tmp53) + weak_mat_tmp11*(-weak_mat_tmp2*weak_mat_tmp24 + weak_mat_tmp50);
    const s_t material7 = mu*(-s_t(6)*gu2*gu3 + s_t(6)*gu5*weak_mat_tmp35 + s_t(2)*gu7*weak_mat_tmp43 + s_t(2)*gu7 - weak_mat_tmp33*weak_mat_tmp52 - weak_mat_tmp34*weak_mat_tmp53 - weak_mat_tmp36*weak_mat_tmp44) + weak_mat_tmp11*(gu3*weak_mat_tmp24 - weak_mat_tmp35*weak_mat_tmp46);
    const s_t material8 = mu*(-gu2*gu7*weak_mat_tmp22 - gu4*weak_mat_tmp16 + s_t(6)*gu8 - weak_mat_tmp0*weak_mat_tmp21 + weak_mat_tmp17*weak_mat_tmp26 - weak_mat_tmp25*weak_mat_tmp44 + weak_mat_tmp26*weak_mat_tmp32 + weak_mat_tmp26*weak_mat_tmp40 + weak_mat_tmp26*weak_mat_tmp48 + weak_mat_tmp30 + s_t(2)*weak_mat_tmp32 + s_t(2)*weak_mat_tmp40 - weak_mat_tmp49*weak_mat_tmp8 + weak_mat_tmp51 + s_t(6)*weak_mat_tmp7) + weak_mat_tmp11*(s_t(2)*weak_mat_tmp2*weak_mat_tmp35 - weak_mat_tmp27);
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
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_elastic_d3_simplex_apply_block(
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
        const s_t lmbda,
        const s_t mu,
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
    const s_t weak_mat_tmp0 = s_t(2)*gu6;
    const s_t weak_mat_tmp1 = gu7*weak_mat_tmp0;
    const s_t weak_mat_tmp2 = gu4 + s_t(1);
    const s_t weak_mat_tmp3 = s_t(2)*gu3;
    const s_t weak_mat_tmp4 = weak_mat_tmp2*weak_mat_tmp3;
    const s_t weak_mat_tmp5 = mu*(-weak_mat_tmp1 - weak_mat_tmp4);
    const s_t weak_mat_tmp6 = gu8 + s_t(1);
    const s_t weak_mat_tmp7 = gu3*weak_mat_tmp6;
    const s_t weak_mat_tmp8 = gu5*gu6 - weak_mat_tmp7;
    const s_t weak_mat_tmp9 = gu5*gu7;
    const s_t weak_mat_tmp10 = s_t(2)*weak_mat_tmp9;
    const s_t weak_mat_tmp11 = ((s_t(1) / s_t(2)))*lmbda;
    const s_t weak_mat_tmp12 = weak_mat_tmp11*(-weak_mat_tmp10 + s_t(2)*weak_mat_tmp2*weak_mat_tmp6);
    const s_t weak_mat_tmp13 = gu5*weak_mat_tmp3;
    const s_t weak_mat_tmp14 = weak_mat_tmp0*weak_mat_tmp6;
    const s_t weak_mat_tmp15 = mu*(-weak_mat_tmp13 - weak_mat_tmp14);
    const s_t weak_mat_tmp16 = gu3*gu7;
    const s_t weak_mat_tmp17 = gu6*weak_mat_tmp2;
    const s_t weak_mat_tmp18 = weak_mat_tmp16 - weak_mat_tmp17;
    const s_t weak_mat_tmp19 = s_t(2)*gu2;
    const s_t weak_mat_tmp20 = gu5*weak_mat_tmp19;
    const s_t weak_mat_tmp21 = s_t(2)*gu1;
    const s_t weak_mat_tmp22 = weak_mat_tmp2*weak_mat_tmp21;
    const s_t weak_mat_tmp23 = mu*(-weak_mat_tmp20 - weak_mat_tmp22);
    const s_t weak_mat_tmp24 = gu1*weak_mat_tmp6;
    const s_t weak_mat_tmp25 = gu2*gu7 - weak_mat_tmp24;
    const s_t weak_mat_tmp26 = gu7*weak_mat_tmp21;
    const s_t weak_mat_tmp27 = weak_mat_tmp19*weak_mat_tmp6;
    const s_t weak_mat_tmp28 = mu*(-weak_mat_tmp26 - weak_mat_tmp27);
    const s_t weak_mat_tmp29 = gu1*gu5;
    const s_t weak_mat_tmp30 = gu2*weak_mat_tmp2;
    const s_t weak_mat_tmp31 = weak_mat_tmp29 - weak_mat_tmp30;
    const s_t weak_mat_tmp32 = s_t(2)*pow_2(gu5);
    const s_t weak_mat_tmp33 = s_t(2)*pow_2(weak_mat_tmp6) + s_t(2);
    const s_t weak_mat_tmp34 = weak_mat_tmp32 + weak_mat_tmp33;
    const s_t weak_mat_tmp35 = s_t(2)*pow_2(gu7);
    const s_t weak_mat_tmp36 = s_t(2)*pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp37 = weak_mat_tmp35 + weak_mat_tmp36;
    const s_t weak_mat_tmp38 = weak_mat_tmp2*weak_mat_tmp6 - weak_mat_tmp9;
    const s_t weak_mat_tmp39 = gu1*gu6;
    const s_t weak_mat_tmp40 = gu0 + s_t(1);
    const s_t weak_mat_tmp41 = gu7*weak_mat_tmp40;
    const s_t weak_mat_tmp42 = weak_mat_tmp39 - weak_mat_tmp41;
    const s_t weak_mat_tmp43 = s_t(6)*gu7;
    const s_t weak_mat_tmp44 = gu2*gu3;
    const s_t weak_mat_tmp45 = s_t(2)*weak_mat_tmp44;
    const s_t weak_mat_tmp46 = gu5*weak_mat_tmp40;
    const s_t weak_mat_tmp47 = gu0*gu4;
    const s_t weak_mat_tmp48 = gu0*gu8;
    const s_t weak_mat_tmp49 = gu4*gu8;
    const s_t weak_mat_tmp50 = gu1*gu3;
    const s_t weak_mat_tmp51 = gu2*gu6;
    const s_t weak_mat_tmp52 = gu5*gu6;
    const s_t weak_mat_tmp53 = lmbda*(-gu0*weak_mat_tmp9 + gu0 + gu1*weak_mat_tmp52 + gu2*weak_mat_tmp16 - gu4*weak_mat_tmp51 + gu4 + gu8*weak_mat_tmp47 - gu8*weak_mat_tmp50 + gu8 + weak_mat_tmp47 + weak_mat_tmp48 + weak_mat_tmp49 - weak_mat_tmp50 - weak_mat_tmp51 - weak_mat_tmp9);
    const s_t weak_mat_tmp54 = gu7*weak_mat_tmp53;
    const s_t weak_mat_tmp55 = mu*(weak_mat_tmp43 - weak_mat_tmp45 + s_t(4)*weak_mat_tmp46) - weak_mat_tmp54;
    const s_t weak_mat_tmp56 = weak_mat_tmp44 - weak_mat_tmp46;
    const s_t weak_mat_tmp57 = s_t(6)*gu5;
    const s_t weak_mat_tmp58 = s_t(2)*weak_mat_tmp39;
    const s_t weak_mat_tmp59 = gu5*weak_mat_tmp53;
    const s_t weak_mat_tmp60 = mu*(s_t(4)*weak_mat_tmp41 + weak_mat_tmp57 - weak_mat_tmp58) - weak_mat_tmp59;
    const s_t weak_mat_tmp61 = weak_mat_tmp40*weak_mat_tmp6 - weak_mat_tmp51;
    const s_t weak_mat_tmp62 = -s_t(6)*gu8;
    const s_t weak_mat_tmp63 = s_t(2)*weak_mat_tmp50;
    const s_t weak_mat_tmp64 = s_t(4)*gu0;
    const s_t weak_mat_tmp65 = weak_mat_tmp64 + s_t(-2);
    const s_t weak_mat_tmp66 = weak_mat_tmp53*weak_mat_tmp6;
    const s_t weak_mat_tmp67 = mu*(gu4*weak_mat_tmp64 + s_t(4)*gu4 + weak_mat_tmp62 - weak_mat_tmp63 + weak_mat_tmp65) + weak_mat_tmp66;
    const s_t weak_mat_tmp68 = weak_mat_tmp2*weak_mat_tmp40 - weak_mat_tmp50;
    const s_t weak_mat_tmp69 = -s_t(6)*gu4;
    const s_t weak_mat_tmp70 = s_t(2)*weak_mat_tmp51;
    const s_t weak_mat_tmp71 = weak_mat_tmp2*weak_mat_tmp53;
    const s_t weak_mat_tmp72 = mu*(gu8*weak_mat_tmp64 + s_t(4)*gu8 + weak_mat_tmp65 + weak_mat_tmp69 - weak_mat_tmp70) + weak_mat_tmp71;
    const s_t weak_mat_tmp73 = -s_t(2)*weak_mat_tmp52;
    const s_t weak_mat_tmp74 = s_t(2)*weak_mat_tmp7;
    const s_t weak_mat_tmp75 = weak_mat_tmp11*(-weak_mat_tmp73 - weak_mat_tmp74);
    const s_t weak_mat_tmp76 = s_t(2)*gu5;
    const s_t weak_mat_tmp77 = weak_mat_tmp2*weak_mat_tmp76;
    const s_t weak_mat_tmp78 = s_t(2)*gu7;
    const s_t weak_mat_tmp79 = weak_mat_tmp6*weak_mat_tmp78;
    const s_t weak_mat_tmp80 = mu*(-weak_mat_tmp77 - weak_mat_tmp79);
    const s_t weak_mat_tmp81 = weak_mat_tmp3*weak_mat_tmp40;
    const s_t weak_mat_tmp82 = mu*(-weak_mat_tmp20 - weak_mat_tmp81);
    const s_t weak_mat_tmp83 = weak_mat_tmp0*weak_mat_tmp40;
    const s_t weak_mat_tmp84 = mu*(-weak_mat_tmp27 - weak_mat_tmp83);
    const s_t weak_mat_tmp85 = s_t(2)*pow_2(gu3);
    const s_t weak_mat_tmp86 = s_t(2)*pow_2(gu6);
    const s_t weak_mat_tmp87 = weak_mat_tmp85 + weak_mat_tmp86;
    const s_t weak_mat_tmp88 = s_t(6)*gu6;
    const s_t weak_mat_tmp89 = s_t(2)*weak_mat_tmp30;
    const s_t weak_mat_tmp90 = gu6*weak_mat_tmp53;
    const s_t weak_mat_tmp91 = mu*(s_t(4)*gu1*gu5 - weak_mat_tmp88 - weak_mat_tmp89) + weak_mat_tmp90;
    const s_t weak_mat_tmp92 = s_t(2)*weak_mat_tmp41;
    const s_t weak_mat_tmp93 = mu*(s_t(4)*gu1*gu6 - weak_mat_tmp57 - weak_mat_tmp92) + weak_mat_tmp59;
    const s_t weak_mat_tmp94 = s_t(6)*gu3;
    const s_t weak_mat_tmp95 = -s_t(2)*gu2*gu7;
    const s_t weak_mat_tmp96 = gu3*weak_mat_tmp53;
    const s_t weak_mat_tmp97 = mu*(s_t(4)*weak_mat_tmp24 + weak_mat_tmp94 + weak_mat_tmp95) - weak_mat_tmp96;
    const s_t weak_mat_tmp98 = s_t(2)*gu4;
    const s_t weak_mat_tmp99 = s_t(2)*gu0 + s_t(-4);
    const s_t weak_mat_tmp100 = mu*(s_t(4)*gu1*gu3 - s_t(2)*weak_mat_tmp47 - weak_mat_tmp62 - weak_mat_tmp98 - weak_mat_tmp99) - weak_mat_tmp66;
    const s_t weak_mat_tmp101 = s_t(2)*weak_mat_tmp16;
    const s_t weak_mat_tmp102 = s_t(2)*weak_mat_tmp17;
    const s_t weak_mat_tmp103 = weak_mat_tmp11*(weak_mat_tmp101 - weak_mat_tmp102);
    const s_t weak_mat_tmp104 = mu*(-weak_mat_tmp22 - weak_mat_tmp81);
    const s_t weak_mat_tmp105 = mu*(-weak_mat_tmp26 - weak_mat_tmp83);
    const s_t weak_mat_tmp106 = s_t(2)*weak_mat_tmp46;
    const s_t weak_mat_tmp107 = mu*(s_t(4)*gu2*gu3 - weak_mat_tmp106 - weak_mat_tmp43) + weak_mat_tmp54;
    const s_t weak_mat_tmp108 = s_t(2)*weak_mat_tmp24;
    const s_t weak_mat_tmp109 = mu*(s_t(4)*gu2*gu7 - weak_mat_tmp108 - weak_mat_tmp94) + weak_mat_tmp96;
    const s_t weak_mat_tmp110 = s_t(2)*weak_mat_tmp29;
    const s_t weak_mat_tmp111 = mu*(-weak_mat_tmp110 + s_t(4)*weak_mat_tmp30 + weak_mat_tmp88) - weak_mat_tmp90;
    const s_t weak_mat_tmp112 = s_t(2)*gu8;
    const s_t weak_mat_tmp113 = mu*(s_t(4)*gu2*gu6 - weak_mat_tmp112 - s_t(2)*weak_mat_tmp48 - weak_mat_tmp69 - weak_mat_tmp99) - weak_mat_tmp71;
    const s_t weak_mat_tmp114 = weak_mat_tmp11*(-weak_mat_tmp108 - weak_mat_tmp95);
    const s_t weak_mat_tmp115 = weak_mat_tmp21*weak_mat_tmp40;
    const s_t weak_mat_tmp116 = mu*(-weak_mat_tmp1 - weak_mat_tmp115);
    const s_t weak_mat_tmp117 = weak_mat_tmp19*weak_mat_tmp40;
    const s_t weak_mat_tmp118 = mu*(-weak_mat_tmp117 - weak_mat_tmp14);
    const s_t weak_mat_tmp119 = weak_mat_tmp6*weak_mat_tmp76;
    const s_t weak_mat_tmp120 = weak_mat_tmp2*weak_mat_tmp78;
    const s_t weak_mat_tmp121 = mu*(-weak_mat_tmp119 - weak_mat_tmp120);
    const s_t weak_mat_tmp122 = s_t(2)*pow_2(gu2);
    const s_t weak_mat_tmp123 = weak_mat_tmp122 + weak_mat_tmp33;
    const s_t weak_mat_tmp124 = s_t(2)*pow_2(gu1);
    const s_t weak_mat_tmp125 = weak_mat_tmp124 + weak_mat_tmp35;
    const s_t weak_mat_tmp126 = s_t(6)*gu2;
    const s_t weak_mat_tmp127 = gu2*weak_mat_tmp53;
    const s_t weak_mat_tmp128 = mu*(s_t(4)*gu3*gu7 - weak_mat_tmp102 - weak_mat_tmp126) + weak_mat_tmp127;
    const s_t weak_mat_tmp129 = s_t(6)*gu1;
    const s_t weak_mat_tmp130 = gu1*weak_mat_tmp53;
    const s_t weak_mat_tmp131 = mu*(weak_mat_tmp129 + s_t(4)*weak_mat_tmp7 + weak_mat_tmp73) - weak_mat_tmp130;
    const s_t weak_mat_tmp132 = weak_mat_tmp11*(s_t(2)*weak_mat_tmp40*weak_mat_tmp6 - weak_mat_tmp70);
    const s_t weak_mat_tmp133 = gu2*weak_mat_tmp21;
    const s_t weak_mat_tmp134 = mu*(-weak_mat_tmp133 - weak_mat_tmp79);
    const s_t weak_mat_tmp135 = gu6*weak_mat_tmp3;
    const s_t weak_mat_tmp136 = mu*(-weak_mat_tmp119 - weak_mat_tmp135);
    const s_t weak_mat_tmp137 = s_t(2)*pow_2(weak_mat_tmp40);
    const s_t weak_mat_tmp138 = weak_mat_tmp137 + weak_mat_tmp86;
    const s_t weak_mat_tmp139 = mu*(-weak_mat_tmp101 + weak_mat_tmp126 + s_t(4)*weak_mat_tmp17) - weak_mat_tmp127;
    const s_t weak_mat_tmp140 = s_t(6)*gu0;
    const s_t weak_mat_tmp141 = weak_mat_tmp40*weak_mat_tmp53;
    const s_t weak_mat_tmp142 = mu*(s_t(4)*gu4*gu8 + s_t(4)*gu4 + s_t(4)*gu8 - weak_mat_tmp10 - weak_mat_tmp140 + s_t(-2)) + weak_mat_tmp141;
    const s_t weak_mat_tmp143 = weak_mat_tmp11*(weak_mat_tmp58 - weak_mat_tmp92);
    const s_t weak_mat_tmp144 = mu*(-weak_mat_tmp120 - weak_mat_tmp135);
    const s_t weak_mat_tmp145 = mu*(s_t(4)*gu5*gu6 - weak_mat_tmp129 - weak_mat_tmp74) + weak_mat_tmp130;
    const s_t weak_mat_tmp146 = mu*(-weak_mat_tmp112 + weak_mat_tmp140 - s_t(2)*weak_mat_tmp49 + s_t(4)*weak_mat_tmp9 - weak_mat_tmp98 + s_t(4)) - weak_mat_tmp141;
    const s_t weak_mat_tmp147 = weak_mat_tmp11*(weak_mat_tmp110 - weak_mat_tmp89);
    const s_t weak_mat_tmp148 = mu*(-weak_mat_tmp115 - weak_mat_tmp4);
    const s_t weak_mat_tmp149 = mu*(-weak_mat_tmp117 - weak_mat_tmp13);
    const s_t weak_mat_tmp150 = weak_mat_tmp122 + weak_mat_tmp32 + s_t(2);
    const s_t weak_mat_tmp151 = weak_mat_tmp124 + weak_mat_tmp36;
    const s_t weak_mat_tmp152 = weak_mat_tmp11*(-weak_mat_tmp106 + weak_mat_tmp45);
    const s_t weak_mat_tmp153 = mu*(-weak_mat_tmp133 - weak_mat_tmp77);
    const s_t weak_mat_tmp154 = weak_mat_tmp137 + weak_mat_tmp85;
    const s_t weak_mat_tmp155 = weak_mat_tmp11*(s_t(2)*weak_mat_tmp2*weak_mat_tmp40 - weak_mat_tmp63);
    const s_t material0 = trial_grad0*(mu*(weak_mat_tmp34 + weak_mat_tmp37) + weak_mat_tmp12*weak_mat_tmp38) + trial_grad1*(weak_mat_tmp12*weak_mat_tmp8 + weak_mat_tmp5) + trial_grad2*(weak_mat_tmp12*weak_mat_tmp18 + weak_mat_tmp15) + trial_grad3*(weak_mat_tmp12*weak_mat_tmp25 + weak_mat_tmp23) + trial_grad4*(weak_mat_tmp12*weak_mat_tmp61 + weak_mat_tmp67) + trial_grad5*(weak_mat_tmp12*weak_mat_tmp42 + weak_mat_tmp55) + trial_grad6*(weak_mat_tmp12*weak_mat_tmp31 + weak_mat_tmp28) + trial_grad7*(weak_mat_tmp12*weak_mat_tmp56 + weak_mat_tmp60) + trial_grad8*(weak_mat_tmp12*weak_mat_tmp68 + weak_mat_tmp72);
    const s_t material1 = trial_grad0*(weak_mat_tmp38*weak_mat_tmp75 + weak_mat_tmp5) + trial_grad1*(mu*(weak_mat_tmp34 + weak_mat_tmp87) + weak_mat_tmp75*weak_mat_tmp8) + trial_grad2*(weak_mat_tmp18*weak_mat_tmp75 + weak_mat_tmp80) + trial_grad3*(weak_mat_tmp100 + weak_mat_tmp25*weak_mat_tmp75) + trial_grad4*(weak_mat_tmp61*weak_mat_tmp75 + weak_mat_tmp82) + trial_grad5*(weak_mat_tmp42*weak_mat_tmp75 + weak_mat_tmp91) + trial_grad6*(weak_mat_tmp31*weak_mat_tmp75 + weak_mat_tmp93) + trial_grad7*(weak_mat_tmp56*weak_mat_tmp75 + weak_mat_tmp84) + trial_grad8*(weak_mat_tmp68*weak_mat_tmp75 + weak_mat_tmp97);
    const s_t material2 = trial_grad0*(weak_mat_tmp103*weak_mat_tmp38 + weak_mat_tmp15) + trial_grad1*(weak_mat_tmp103*weak_mat_tmp8 + weak_mat_tmp80) + trial_grad2*(mu*(weak_mat_tmp37 + weak_mat_tmp87 + s_t(2)) + weak_mat_tmp103*weak_mat_tmp18) + trial_grad3*(weak_mat_tmp103*weak_mat_tmp25 + weak_mat_tmp107) + trial_grad4*(weak_mat_tmp103*weak_mat_tmp61 + weak_mat_tmp111) + trial_grad5*(weak_mat_tmp103*weak_mat_tmp42 + weak_mat_tmp104) + trial_grad6*(weak_mat_tmp103*weak_mat_tmp31 + weak_mat_tmp113) + trial_grad7*(weak_mat_tmp103*weak_mat_tmp56 + weak_mat_tmp109) + trial_grad8*(weak_mat_tmp103*weak_mat_tmp68 + weak_mat_tmp105);
    const s_t material3 = trial_grad0*(weak_mat_tmp114*weak_mat_tmp38 + weak_mat_tmp23) + trial_grad1*(weak_mat_tmp100 + weak_mat_tmp114*weak_mat_tmp8) + trial_grad2*(weak_mat_tmp107 + weak_mat_tmp114*weak_mat_tmp18) + trial_grad3*(mu*(weak_mat_tmp123 + weak_mat_tmp125) + weak_mat_tmp114*weak_mat_tmp25) + trial_grad4*(weak_mat_tmp114*weak_mat_tmp61 + weak_mat_tmp116) + trial_grad5*(weak_mat_tmp114*weak_mat_tmp42 + weak_mat_tmp118) + trial_grad6*(weak_mat_tmp114*weak_mat_tmp31 + weak_mat_tmp121) + trial_grad7*(weak_mat_tmp114*weak_mat_tmp56 + weak_mat_tmp128) + trial_grad8*(weak_mat_tmp114*weak_mat_tmp68 + weak_mat_tmp131);
    const s_t material4 = trial_grad0*(weak_mat_tmp132*weak_mat_tmp38 + weak_mat_tmp67) + trial_grad1*(weak_mat_tmp132*weak_mat_tmp8 + weak_mat_tmp82) + trial_grad2*(weak_mat_tmp111 + weak_mat_tmp132*weak_mat_tmp18) + trial_grad3*(weak_mat_tmp116 + weak_mat_tmp132*weak_mat_tmp25) + trial_grad4*(mu*(weak_mat_tmp123 + weak_mat_tmp138) + weak_mat_tmp132*weak_mat_tmp61) + trial_grad5*(weak_mat_tmp132*weak_mat_tmp42 + weak_mat_tmp134) + trial_grad6*(weak_mat_tmp132*weak_mat_tmp31 + weak_mat_tmp139) + trial_grad7*(weak_mat_tmp132*weak_mat_tmp56 + weak_mat_tmp136) + trial_grad8*(weak_mat_tmp132*weak_mat_tmp68 + weak_mat_tmp142);
    const s_t material5 = trial_grad0*(weak_mat_tmp143*weak_mat_tmp38 + weak_mat_tmp55) + trial_grad1*(weak_mat_tmp143*weak_mat_tmp8 + weak_mat_tmp91) + trial_grad2*(weak_mat_tmp104 + weak_mat_tmp143*weak_mat_tmp18) + trial_grad3*(weak_mat_tmp118 + weak_mat_tmp143*weak_mat_tmp25) + trial_grad4*(weak_mat_tmp134 + weak_mat_tmp143*weak_mat_tmp61) + trial_grad5*(mu*(weak_mat_tmp125 + weak_mat_tmp138 + s_t(2)) + weak_mat_tmp143*weak_mat_tmp42) + trial_grad6*(weak_mat_tmp143*weak_mat_tmp31 + weak_mat_tmp145) + trial_grad7*(weak_mat_tmp143*weak_mat_tmp56 + weak_mat_tmp146) + trial_grad8*(weak_mat_tmp143*weak_mat_tmp68 + weak_mat_tmp144);
    const s_t material6 = trial_grad0*(weak_mat_tmp147*weak_mat_tmp38 + weak_mat_tmp28) + trial_grad1*(weak_mat_tmp147*weak_mat_tmp8 + weak_mat_tmp93) + trial_grad2*(weak_mat_tmp113 + weak_mat_tmp147*weak_mat_tmp18) + trial_grad3*(weak_mat_tmp121 + weak_mat_tmp147*weak_mat_tmp25) + trial_grad4*(weak_mat_tmp139 + weak_mat_tmp147*weak_mat_tmp61) + trial_grad5*(weak_mat_tmp145 + weak_mat_tmp147*weak_mat_tmp42) + trial_grad6*(mu*(weak_mat_tmp150 + weak_mat_tmp151) + weak_mat_tmp147*weak_mat_tmp31) + trial_grad7*(weak_mat_tmp147*weak_mat_tmp56 + weak_mat_tmp148) + trial_grad8*(weak_mat_tmp147*weak_mat_tmp68 + weak_mat_tmp149);
    const s_t material7 = trial_grad0*(weak_mat_tmp152*weak_mat_tmp38 + weak_mat_tmp60) + trial_grad1*(weak_mat_tmp152*weak_mat_tmp8 + weak_mat_tmp84) + trial_grad2*(weak_mat_tmp109 + weak_mat_tmp152*weak_mat_tmp18) + trial_grad3*(weak_mat_tmp128 + weak_mat_tmp152*weak_mat_tmp25) + trial_grad4*(weak_mat_tmp136 + weak_mat_tmp152*weak_mat_tmp61) + trial_grad5*(weak_mat_tmp146 + weak_mat_tmp152*weak_mat_tmp42) + trial_grad6*(weak_mat_tmp148 + weak_mat_tmp152*weak_mat_tmp31) + trial_grad7*(mu*(weak_mat_tmp150 + weak_mat_tmp154) + weak_mat_tmp152*weak_mat_tmp56) + trial_grad8*(weak_mat_tmp152*weak_mat_tmp68 + weak_mat_tmp153);
    const s_t material8 = trial_grad0*(weak_mat_tmp155*weak_mat_tmp38 + weak_mat_tmp72) + trial_grad1*(weak_mat_tmp155*weak_mat_tmp8 + weak_mat_tmp97) + trial_grad2*(weak_mat_tmp105 + weak_mat_tmp155*weak_mat_tmp18) + trial_grad3*(weak_mat_tmp131 + weak_mat_tmp155*weak_mat_tmp25) + trial_grad4*(weak_mat_tmp142 + weak_mat_tmp155*weak_mat_tmp61) + trial_grad5*(weak_mat_tmp144 + weak_mat_tmp155*weak_mat_tmp42) + trial_grad6*(weak_mat_tmp149 + weak_mat_tmp155*weak_mat_tmp31) + trial_grad7*(weak_mat_tmp153 + weak_mat_tmp155*weak_mat_tmp56) + trial_grad8*(mu*(weak_mat_tmp151 + weak_mat_tmp154 + s_t(2)) + weak_mat_tmp155*weak_mat_tmp68);
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
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_elastic_d3_simplex_tet4_apply_block(
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
        const s_t lmbda,
        const s_t mu,
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
    const s_t weak_mat_tmp0 = s_t(2)*gu6;
    const s_t weak_mat_tmp1 = gu7*weak_mat_tmp0;
    const s_t weak_mat_tmp2 = gu4 + s_t(1);
    const s_t weak_mat_tmp3 = s_t(2)*gu3;
    const s_t weak_mat_tmp4 = weak_mat_tmp2*weak_mat_tmp3;
    const s_t weak_mat_tmp5 = mu*(-weak_mat_tmp1 - weak_mat_tmp4);
    const s_t weak_mat_tmp6 = gu8 + s_t(1);
    const s_t weak_mat_tmp7 = gu3*weak_mat_tmp6;
    const s_t weak_mat_tmp8 = gu5*gu6 - weak_mat_tmp7;
    const s_t weak_mat_tmp9 = gu5*gu7;
    const s_t weak_mat_tmp10 = s_t(2)*weak_mat_tmp9;
    const s_t weak_mat_tmp11 = ((s_t(1) / s_t(2)))*lmbda;
    const s_t weak_mat_tmp12 = weak_mat_tmp11*(-weak_mat_tmp10 + s_t(2)*weak_mat_tmp2*weak_mat_tmp6);
    const s_t weak_mat_tmp13 = gu5*weak_mat_tmp3;
    const s_t weak_mat_tmp14 = weak_mat_tmp0*weak_mat_tmp6;
    const s_t weak_mat_tmp15 = mu*(-weak_mat_tmp13 - weak_mat_tmp14);
    const s_t weak_mat_tmp16 = gu3*gu7;
    const s_t weak_mat_tmp17 = gu6*weak_mat_tmp2;
    const s_t weak_mat_tmp18 = weak_mat_tmp16 - weak_mat_tmp17;
    const s_t weak_mat_tmp19 = s_t(2)*gu2;
    const s_t weak_mat_tmp20 = gu5*weak_mat_tmp19;
    const s_t weak_mat_tmp21 = s_t(2)*gu1;
    const s_t weak_mat_tmp22 = weak_mat_tmp2*weak_mat_tmp21;
    const s_t weak_mat_tmp23 = mu*(-weak_mat_tmp20 - weak_mat_tmp22);
    const s_t weak_mat_tmp24 = gu1*weak_mat_tmp6;
    const s_t weak_mat_tmp25 = gu2*gu7 - weak_mat_tmp24;
    const s_t weak_mat_tmp26 = gu7*weak_mat_tmp21;
    const s_t weak_mat_tmp27 = weak_mat_tmp19*weak_mat_tmp6;
    const s_t weak_mat_tmp28 = mu*(-weak_mat_tmp26 - weak_mat_tmp27);
    const s_t weak_mat_tmp29 = gu1*gu5;
    const s_t weak_mat_tmp30 = gu2*weak_mat_tmp2;
    const s_t weak_mat_tmp31 = weak_mat_tmp29 - weak_mat_tmp30;
    const s_t weak_mat_tmp32 = s_t(2)*pow_2(gu5);
    const s_t weak_mat_tmp33 = s_t(2)*pow_2(weak_mat_tmp6) + s_t(2);
    const s_t weak_mat_tmp34 = weak_mat_tmp32 + weak_mat_tmp33;
    const s_t weak_mat_tmp35 = s_t(2)*pow_2(gu7);
    const s_t weak_mat_tmp36 = s_t(2)*pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp37 = weak_mat_tmp35 + weak_mat_tmp36;
    const s_t weak_mat_tmp38 = weak_mat_tmp2*weak_mat_tmp6 - weak_mat_tmp9;
    const s_t weak_mat_tmp39 = gu1*gu6;
    const s_t weak_mat_tmp40 = gu0 + s_t(1);
    const s_t weak_mat_tmp41 = gu7*weak_mat_tmp40;
    const s_t weak_mat_tmp42 = weak_mat_tmp39 - weak_mat_tmp41;
    const s_t weak_mat_tmp43 = s_t(6)*gu7;
    const s_t weak_mat_tmp44 = gu2*gu3;
    const s_t weak_mat_tmp45 = s_t(2)*weak_mat_tmp44;
    const s_t weak_mat_tmp46 = gu5*weak_mat_tmp40;
    const s_t weak_mat_tmp47 = gu0*gu4;
    const s_t weak_mat_tmp48 = gu0*gu8;
    const s_t weak_mat_tmp49 = gu4*gu8;
    const s_t weak_mat_tmp50 = gu1*gu3;
    const s_t weak_mat_tmp51 = gu2*gu6;
    const s_t weak_mat_tmp52 = gu5*gu6;
    const s_t weak_mat_tmp53 = lmbda*(-gu0*weak_mat_tmp9 + gu0 + gu1*weak_mat_tmp52 + gu2*weak_mat_tmp16 - gu4*weak_mat_tmp51 + gu4 + gu8*weak_mat_tmp47 - gu8*weak_mat_tmp50 + gu8 + weak_mat_tmp47 + weak_mat_tmp48 + weak_mat_tmp49 - weak_mat_tmp50 - weak_mat_tmp51 - weak_mat_tmp9);
    const s_t weak_mat_tmp54 = gu7*weak_mat_tmp53;
    const s_t weak_mat_tmp55 = mu*(weak_mat_tmp43 - weak_mat_tmp45 + s_t(4)*weak_mat_tmp46) - weak_mat_tmp54;
    const s_t weak_mat_tmp56 = weak_mat_tmp44 - weak_mat_tmp46;
    const s_t weak_mat_tmp57 = s_t(6)*gu5;
    const s_t weak_mat_tmp58 = s_t(2)*weak_mat_tmp39;
    const s_t weak_mat_tmp59 = gu5*weak_mat_tmp53;
    const s_t weak_mat_tmp60 = mu*(s_t(4)*weak_mat_tmp41 + weak_mat_tmp57 - weak_mat_tmp58) - weak_mat_tmp59;
    const s_t weak_mat_tmp61 = weak_mat_tmp40*weak_mat_tmp6 - weak_mat_tmp51;
    const s_t weak_mat_tmp62 = -s_t(6)*gu8;
    const s_t weak_mat_tmp63 = s_t(2)*weak_mat_tmp50;
    const s_t weak_mat_tmp64 = s_t(4)*gu0;
    const s_t weak_mat_tmp65 = weak_mat_tmp64 + s_t(-2);
    const s_t weak_mat_tmp66 = weak_mat_tmp53*weak_mat_tmp6;
    const s_t weak_mat_tmp67 = mu*(gu4*weak_mat_tmp64 + s_t(4)*gu4 + weak_mat_tmp62 - weak_mat_tmp63 + weak_mat_tmp65) + weak_mat_tmp66;
    const s_t weak_mat_tmp68 = weak_mat_tmp2*weak_mat_tmp40 - weak_mat_tmp50;
    const s_t weak_mat_tmp69 = -s_t(6)*gu4;
    const s_t weak_mat_tmp70 = s_t(2)*weak_mat_tmp51;
    const s_t weak_mat_tmp71 = weak_mat_tmp2*weak_mat_tmp53;
    const s_t weak_mat_tmp72 = mu*(gu8*weak_mat_tmp64 + s_t(4)*gu8 + weak_mat_tmp65 + weak_mat_tmp69 - weak_mat_tmp70) + weak_mat_tmp71;
    const s_t weak_mat_tmp73 = -s_t(2)*weak_mat_tmp52;
    const s_t weak_mat_tmp74 = s_t(2)*weak_mat_tmp7;
    const s_t weak_mat_tmp75 = weak_mat_tmp11*(-weak_mat_tmp73 - weak_mat_tmp74);
    const s_t weak_mat_tmp76 = s_t(2)*gu5;
    const s_t weak_mat_tmp77 = weak_mat_tmp2*weak_mat_tmp76;
    const s_t weak_mat_tmp78 = s_t(2)*gu7;
    const s_t weak_mat_tmp79 = weak_mat_tmp6*weak_mat_tmp78;
    const s_t weak_mat_tmp80 = mu*(-weak_mat_tmp77 - weak_mat_tmp79);
    const s_t weak_mat_tmp81 = weak_mat_tmp3*weak_mat_tmp40;
    const s_t weak_mat_tmp82 = mu*(-weak_mat_tmp20 - weak_mat_tmp81);
    const s_t weak_mat_tmp83 = weak_mat_tmp0*weak_mat_tmp40;
    const s_t weak_mat_tmp84 = mu*(-weak_mat_tmp27 - weak_mat_tmp83);
    const s_t weak_mat_tmp85 = s_t(2)*pow_2(gu3);
    const s_t weak_mat_tmp86 = s_t(2)*pow_2(gu6);
    const s_t weak_mat_tmp87 = weak_mat_tmp85 + weak_mat_tmp86;
    const s_t weak_mat_tmp88 = s_t(6)*gu6;
    const s_t weak_mat_tmp89 = s_t(2)*weak_mat_tmp30;
    const s_t weak_mat_tmp90 = gu6*weak_mat_tmp53;
    const s_t weak_mat_tmp91 = mu*(s_t(4)*gu1*gu5 - weak_mat_tmp88 - weak_mat_tmp89) + weak_mat_tmp90;
    const s_t weak_mat_tmp92 = s_t(2)*weak_mat_tmp41;
    const s_t weak_mat_tmp93 = mu*(s_t(4)*gu1*gu6 - weak_mat_tmp57 - weak_mat_tmp92) + weak_mat_tmp59;
    const s_t weak_mat_tmp94 = s_t(6)*gu3;
    const s_t weak_mat_tmp95 = -s_t(2)*gu2*gu7;
    const s_t weak_mat_tmp96 = gu3*weak_mat_tmp53;
    const s_t weak_mat_tmp97 = mu*(s_t(4)*weak_mat_tmp24 + weak_mat_tmp94 + weak_mat_tmp95) - weak_mat_tmp96;
    const s_t weak_mat_tmp98 = s_t(2)*gu4;
    const s_t weak_mat_tmp99 = s_t(2)*gu0 + s_t(-4);
    const s_t weak_mat_tmp100 = mu*(s_t(4)*gu1*gu3 - s_t(2)*weak_mat_tmp47 - weak_mat_tmp62 - weak_mat_tmp98 - weak_mat_tmp99) - weak_mat_tmp66;
    const s_t weak_mat_tmp101 = s_t(2)*weak_mat_tmp16;
    const s_t weak_mat_tmp102 = s_t(2)*weak_mat_tmp17;
    const s_t weak_mat_tmp103 = weak_mat_tmp11*(weak_mat_tmp101 - weak_mat_tmp102);
    const s_t weak_mat_tmp104 = mu*(-weak_mat_tmp22 - weak_mat_tmp81);
    const s_t weak_mat_tmp105 = mu*(-weak_mat_tmp26 - weak_mat_tmp83);
    const s_t weak_mat_tmp106 = s_t(2)*weak_mat_tmp46;
    const s_t weak_mat_tmp107 = mu*(s_t(4)*gu2*gu3 - weak_mat_tmp106 - weak_mat_tmp43) + weak_mat_tmp54;
    const s_t weak_mat_tmp108 = s_t(2)*weak_mat_tmp24;
    const s_t weak_mat_tmp109 = mu*(s_t(4)*gu2*gu7 - weak_mat_tmp108 - weak_mat_tmp94) + weak_mat_tmp96;
    const s_t weak_mat_tmp110 = s_t(2)*weak_mat_tmp29;
    const s_t weak_mat_tmp111 = mu*(-weak_mat_tmp110 + s_t(4)*weak_mat_tmp30 + weak_mat_tmp88) - weak_mat_tmp90;
    const s_t weak_mat_tmp112 = s_t(2)*gu8;
    const s_t weak_mat_tmp113 = mu*(s_t(4)*gu2*gu6 - weak_mat_tmp112 - s_t(2)*weak_mat_tmp48 - weak_mat_tmp69 - weak_mat_tmp99) - weak_mat_tmp71;
    const s_t weak_mat_tmp114 = weak_mat_tmp11*(-weak_mat_tmp108 - weak_mat_tmp95);
    const s_t weak_mat_tmp115 = weak_mat_tmp21*weak_mat_tmp40;
    const s_t weak_mat_tmp116 = mu*(-weak_mat_tmp1 - weak_mat_tmp115);
    const s_t weak_mat_tmp117 = weak_mat_tmp19*weak_mat_tmp40;
    const s_t weak_mat_tmp118 = mu*(-weak_mat_tmp117 - weak_mat_tmp14);
    const s_t weak_mat_tmp119 = weak_mat_tmp6*weak_mat_tmp76;
    const s_t weak_mat_tmp120 = weak_mat_tmp2*weak_mat_tmp78;
    const s_t weak_mat_tmp121 = mu*(-weak_mat_tmp119 - weak_mat_tmp120);
    const s_t weak_mat_tmp122 = s_t(2)*pow_2(gu2);
    const s_t weak_mat_tmp123 = weak_mat_tmp122 + weak_mat_tmp33;
    const s_t weak_mat_tmp124 = s_t(2)*pow_2(gu1);
    const s_t weak_mat_tmp125 = weak_mat_tmp124 + weak_mat_tmp35;
    const s_t weak_mat_tmp126 = s_t(6)*gu2;
    const s_t weak_mat_tmp127 = gu2*weak_mat_tmp53;
    const s_t weak_mat_tmp128 = mu*(s_t(4)*gu3*gu7 - weak_mat_tmp102 - weak_mat_tmp126) + weak_mat_tmp127;
    const s_t weak_mat_tmp129 = s_t(6)*gu1;
    const s_t weak_mat_tmp130 = gu1*weak_mat_tmp53;
    const s_t weak_mat_tmp131 = mu*(weak_mat_tmp129 + s_t(4)*weak_mat_tmp7 + weak_mat_tmp73) - weak_mat_tmp130;
    const s_t weak_mat_tmp132 = weak_mat_tmp11*(s_t(2)*weak_mat_tmp40*weak_mat_tmp6 - weak_mat_tmp70);
    const s_t weak_mat_tmp133 = gu2*weak_mat_tmp21;
    const s_t weak_mat_tmp134 = mu*(-weak_mat_tmp133 - weak_mat_tmp79);
    const s_t weak_mat_tmp135 = gu6*weak_mat_tmp3;
    const s_t weak_mat_tmp136 = mu*(-weak_mat_tmp119 - weak_mat_tmp135);
    const s_t weak_mat_tmp137 = s_t(2)*pow_2(weak_mat_tmp40);
    const s_t weak_mat_tmp138 = weak_mat_tmp137 + weak_mat_tmp86;
    const s_t weak_mat_tmp139 = mu*(-weak_mat_tmp101 + weak_mat_tmp126 + s_t(4)*weak_mat_tmp17) - weak_mat_tmp127;
    const s_t weak_mat_tmp140 = s_t(6)*gu0;
    const s_t weak_mat_tmp141 = weak_mat_tmp40*weak_mat_tmp53;
    const s_t weak_mat_tmp142 = mu*(s_t(4)*gu4*gu8 + s_t(4)*gu4 + s_t(4)*gu8 - weak_mat_tmp10 - weak_mat_tmp140 + s_t(-2)) + weak_mat_tmp141;
    const s_t weak_mat_tmp143 = weak_mat_tmp11*(weak_mat_tmp58 - weak_mat_tmp92);
    const s_t weak_mat_tmp144 = mu*(-weak_mat_tmp120 - weak_mat_tmp135);
    const s_t weak_mat_tmp145 = mu*(s_t(4)*gu5*gu6 - weak_mat_tmp129 - weak_mat_tmp74) + weak_mat_tmp130;
    const s_t weak_mat_tmp146 = mu*(-weak_mat_tmp112 + weak_mat_tmp140 - s_t(2)*weak_mat_tmp49 + s_t(4)*weak_mat_tmp9 - weak_mat_tmp98 + s_t(4)) - weak_mat_tmp141;
    const s_t weak_mat_tmp147 = weak_mat_tmp11*(weak_mat_tmp110 - weak_mat_tmp89);
    const s_t weak_mat_tmp148 = mu*(-weak_mat_tmp115 - weak_mat_tmp4);
    const s_t weak_mat_tmp149 = mu*(-weak_mat_tmp117 - weak_mat_tmp13);
    const s_t weak_mat_tmp150 = weak_mat_tmp122 + weak_mat_tmp32 + s_t(2);
    const s_t weak_mat_tmp151 = weak_mat_tmp124 + weak_mat_tmp36;
    const s_t weak_mat_tmp152 = weak_mat_tmp11*(-weak_mat_tmp106 + weak_mat_tmp45);
    const s_t weak_mat_tmp153 = mu*(-weak_mat_tmp133 - weak_mat_tmp77);
    const s_t weak_mat_tmp154 = weak_mat_tmp137 + weak_mat_tmp85;
    const s_t weak_mat_tmp155 = weak_mat_tmp11*(s_t(2)*weak_mat_tmp2*weak_mat_tmp40 - weak_mat_tmp63);
    const s_t material0 = trial_grad0*(mu*(weak_mat_tmp34 + weak_mat_tmp37) + weak_mat_tmp12*weak_mat_tmp38) + trial_grad1*(weak_mat_tmp12*weak_mat_tmp8 + weak_mat_tmp5) + trial_grad2*(weak_mat_tmp12*weak_mat_tmp18 + weak_mat_tmp15) + trial_grad3*(weak_mat_tmp12*weak_mat_tmp25 + weak_mat_tmp23) + trial_grad4*(weak_mat_tmp12*weak_mat_tmp61 + weak_mat_tmp67) + trial_grad5*(weak_mat_tmp12*weak_mat_tmp42 + weak_mat_tmp55) + trial_grad6*(weak_mat_tmp12*weak_mat_tmp31 + weak_mat_tmp28) + trial_grad7*(weak_mat_tmp12*weak_mat_tmp56 + weak_mat_tmp60) + trial_grad8*(weak_mat_tmp12*weak_mat_tmp68 + weak_mat_tmp72);
    const s_t material1 = trial_grad0*(weak_mat_tmp38*weak_mat_tmp75 + weak_mat_tmp5) + trial_grad1*(mu*(weak_mat_tmp34 + weak_mat_tmp87) + weak_mat_tmp75*weak_mat_tmp8) + trial_grad2*(weak_mat_tmp18*weak_mat_tmp75 + weak_mat_tmp80) + trial_grad3*(weak_mat_tmp100 + weak_mat_tmp25*weak_mat_tmp75) + trial_grad4*(weak_mat_tmp61*weak_mat_tmp75 + weak_mat_tmp82) + trial_grad5*(weak_mat_tmp42*weak_mat_tmp75 + weak_mat_tmp91) + trial_grad6*(weak_mat_tmp31*weak_mat_tmp75 + weak_mat_tmp93) + trial_grad7*(weak_mat_tmp56*weak_mat_tmp75 + weak_mat_tmp84) + trial_grad8*(weak_mat_tmp68*weak_mat_tmp75 + weak_mat_tmp97);
    const s_t material2 = trial_grad0*(weak_mat_tmp103*weak_mat_tmp38 + weak_mat_tmp15) + trial_grad1*(weak_mat_tmp103*weak_mat_tmp8 + weak_mat_tmp80) + trial_grad2*(mu*(weak_mat_tmp37 + weak_mat_tmp87 + s_t(2)) + weak_mat_tmp103*weak_mat_tmp18) + trial_grad3*(weak_mat_tmp103*weak_mat_tmp25 + weak_mat_tmp107) + trial_grad4*(weak_mat_tmp103*weak_mat_tmp61 + weak_mat_tmp111) + trial_grad5*(weak_mat_tmp103*weak_mat_tmp42 + weak_mat_tmp104) + trial_grad6*(weak_mat_tmp103*weak_mat_tmp31 + weak_mat_tmp113) + trial_grad7*(weak_mat_tmp103*weak_mat_tmp56 + weak_mat_tmp109) + trial_grad8*(weak_mat_tmp103*weak_mat_tmp68 + weak_mat_tmp105);
    const s_t material3 = trial_grad0*(weak_mat_tmp114*weak_mat_tmp38 + weak_mat_tmp23) + trial_grad1*(weak_mat_tmp100 + weak_mat_tmp114*weak_mat_tmp8) + trial_grad2*(weak_mat_tmp107 + weak_mat_tmp114*weak_mat_tmp18) + trial_grad3*(mu*(weak_mat_tmp123 + weak_mat_tmp125) + weak_mat_tmp114*weak_mat_tmp25) + trial_grad4*(weak_mat_tmp114*weak_mat_tmp61 + weak_mat_tmp116) + trial_grad5*(weak_mat_tmp114*weak_mat_tmp42 + weak_mat_tmp118) + trial_grad6*(weak_mat_tmp114*weak_mat_tmp31 + weak_mat_tmp121) + trial_grad7*(weak_mat_tmp114*weak_mat_tmp56 + weak_mat_tmp128) + trial_grad8*(weak_mat_tmp114*weak_mat_tmp68 + weak_mat_tmp131);
    const s_t material4 = trial_grad0*(weak_mat_tmp132*weak_mat_tmp38 + weak_mat_tmp67) + trial_grad1*(weak_mat_tmp132*weak_mat_tmp8 + weak_mat_tmp82) + trial_grad2*(weak_mat_tmp111 + weak_mat_tmp132*weak_mat_tmp18) + trial_grad3*(weak_mat_tmp116 + weak_mat_tmp132*weak_mat_tmp25) + trial_grad4*(mu*(weak_mat_tmp123 + weak_mat_tmp138) + weak_mat_tmp132*weak_mat_tmp61) + trial_grad5*(weak_mat_tmp132*weak_mat_tmp42 + weak_mat_tmp134) + trial_grad6*(weak_mat_tmp132*weak_mat_tmp31 + weak_mat_tmp139) + trial_grad7*(weak_mat_tmp132*weak_mat_tmp56 + weak_mat_tmp136) + trial_grad8*(weak_mat_tmp132*weak_mat_tmp68 + weak_mat_tmp142);
    const s_t material5 = trial_grad0*(weak_mat_tmp143*weak_mat_tmp38 + weak_mat_tmp55) + trial_grad1*(weak_mat_tmp143*weak_mat_tmp8 + weak_mat_tmp91) + trial_grad2*(weak_mat_tmp104 + weak_mat_tmp143*weak_mat_tmp18) + trial_grad3*(weak_mat_tmp118 + weak_mat_tmp143*weak_mat_tmp25) + trial_grad4*(weak_mat_tmp134 + weak_mat_tmp143*weak_mat_tmp61) + trial_grad5*(mu*(weak_mat_tmp125 + weak_mat_tmp138 + s_t(2)) + weak_mat_tmp143*weak_mat_tmp42) + trial_grad6*(weak_mat_tmp143*weak_mat_tmp31 + weak_mat_tmp145) + trial_grad7*(weak_mat_tmp143*weak_mat_tmp56 + weak_mat_tmp146) + trial_grad8*(weak_mat_tmp143*weak_mat_tmp68 + weak_mat_tmp144);
    const s_t material6 = trial_grad0*(weak_mat_tmp147*weak_mat_tmp38 + weak_mat_tmp28) + trial_grad1*(weak_mat_tmp147*weak_mat_tmp8 + weak_mat_tmp93) + trial_grad2*(weak_mat_tmp113 + weak_mat_tmp147*weak_mat_tmp18) + trial_grad3*(weak_mat_tmp121 + weak_mat_tmp147*weak_mat_tmp25) + trial_grad4*(weak_mat_tmp139 + weak_mat_tmp147*weak_mat_tmp61) + trial_grad5*(weak_mat_tmp145 + weak_mat_tmp147*weak_mat_tmp42) + trial_grad6*(mu*(weak_mat_tmp150 + weak_mat_tmp151) + weak_mat_tmp147*weak_mat_tmp31) + trial_grad7*(weak_mat_tmp147*weak_mat_tmp56 + weak_mat_tmp148) + trial_grad8*(weak_mat_tmp147*weak_mat_tmp68 + weak_mat_tmp149);
    const s_t material7 = trial_grad0*(weak_mat_tmp152*weak_mat_tmp38 + weak_mat_tmp60) + trial_grad1*(weak_mat_tmp152*weak_mat_tmp8 + weak_mat_tmp84) + trial_grad2*(weak_mat_tmp109 + weak_mat_tmp152*weak_mat_tmp18) + trial_grad3*(weak_mat_tmp128 + weak_mat_tmp152*weak_mat_tmp25) + trial_grad4*(weak_mat_tmp136 + weak_mat_tmp152*weak_mat_tmp61) + trial_grad5*(weak_mat_tmp146 + weak_mat_tmp152*weak_mat_tmp42) + trial_grad6*(weak_mat_tmp148 + weak_mat_tmp152*weak_mat_tmp31) + trial_grad7*(mu*(weak_mat_tmp150 + weak_mat_tmp154) + weak_mat_tmp152*weak_mat_tmp56) + trial_grad8*(weak_mat_tmp152*weak_mat_tmp68 + weak_mat_tmp153);
    const s_t material8 = trial_grad0*(weak_mat_tmp155*weak_mat_tmp38 + weak_mat_tmp72) + trial_grad1*(weak_mat_tmp155*weak_mat_tmp8 + weak_mat_tmp97) + trial_grad2*(weak_mat_tmp105 + weak_mat_tmp155*weak_mat_tmp18) + trial_grad3*(weak_mat_tmp131 + weak_mat_tmp155*weak_mat_tmp25) + trial_grad4*(weak_mat_tmp142 + weak_mat_tmp155*weak_mat_tmp61) + trial_grad5*(weak_mat_tmp144 + weak_mat_tmp155*weak_mat_tmp42) + trial_grad6*(weak_mat_tmp149 + weak_mat_tmp155*weak_mat_tmp31) + trial_grad7*(weak_mat_tmp153 + weak_mat_tmp155*weak_mat_tmp56) + trial_grad8*(mu*(weak_mat_tmp151 + weak_mat_tmp154 + s_t(2)) + weak_mat_tmp155*weak_mat_tmp68);
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
