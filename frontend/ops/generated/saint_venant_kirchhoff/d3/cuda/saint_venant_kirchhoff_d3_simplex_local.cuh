#ifndef SAINT_VENANT_KIRCHHOFF_D3_SIMPLEX_LOCAL_CUH
#define SAINT_VENANT_KIRCHHOFF_D3_SIMPLEX_LOCAL_CUH
#include <math.h>
#include <stddef.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_GENERATED_SCALAR_T
#endif
#endif
#include "../../../cuda/kernel_math.cuh"
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
static __host__ __device__ __forceinline__ void saint_venant_kirchhoff_d3_simplex_objective_block(
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
      {
        gu_ref0_values[0] = s_t(0);
        grad_h_ref0_values[0] = s_t(0);
        gu_ref1_values[0] = s_t(0);
        grad_h_ref1_values[0] = s_t(0);
        gu_ref2_values[0] = s_t(0);
        grad_h_ref2_values[0] = s_t(0);
        gu_ref3_values[0] = s_t(0);
        grad_h_ref3_values[0] = s_t(0);
        gu_ref4_values[0] = s_t(0);
        grad_h_ref4_values[0] = s_t(0);
        gu_ref5_values[0] = s_t(0);
        grad_h_ref5_values[0] = s_t(0);
        gu_ref6_values[0] = s_t(0);
        grad_h_ref6_values[0] = s_t(0);
        gu_ref7_values[0] = s_t(0);
        grad_h_ref7_values[0] = s_t(0);
        gu_ref8_values[0] = s_t(0);
        grad_h_ref8_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        {
          gu_ref0_values[0] += u_streams[3 * shape][0] * grad_ref_x[q * NS + shape];
          grad_h_ref0_values[0] += h_streams[3 * shape][0] * grad_ref_x[q * NS + shape];
        }
        {
          gu_ref1_values[0] += u_streams[3 * shape][0] * grad_ref_y[q * NS + shape];
          grad_h_ref1_values[0] += h_streams[3 * shape][0] * grad_ref_y[q * NS + shape];
        }
        {
          gu_ref2_values[0] += u_streams[3 * shape][0] * grad_ref_z[q * NS + shape];
          grad_h_ref2_values[0] += h_streams[3 * shape][0] * grad_ref_z[q * NS + shape];
        }
        {
          gu_ref3_values[0] += u_streams[3 * shape + 1][0] * grad_ref_x[q * NS + shape];
          grad_h_ref3_values[0] += h_streams[3 * shape + 1][0] * grad_ref_x[q * NS + shape];
        }
        {
          gu_ref4_values[0] += u_streams[3 * shape + 1][0] * grad_ref_y[q * NS + shape];
          grad_h_ref4_values[0] += h_streams[3 * shape + 1][0] * grad_ref_y[q * NS + shape];
        }
        {
          gu_ref5_values[0] += u_streams[3 * shape + 1][0] * grad_ref_z[q * NS + shape];
          grad_h_ref5_values[0] += h_streams[3 * shape + 1][0] * grad_ref_z[q * NS + shape];
        }
        {
          gu_ref6_values[0] += u_streams[3 * shape + 2][0] * grad_ref_x[q * NS + shape];
          grad_h_ref6_values[0] += h_streams[3 * shape + 2][0] * grad_ref_x[q * NS + shape];
        }
        {
          gu_ref7_values[0] += u_streams[3 * shape + 2][0] * grad_ref_y[q * NS + shape];
          grad_h_ref7_values[0] += h_streams[3 * shape + 2][0] * grad_ref_y[q * NS + shape];
        }
        {
          gu_ref8_values[0] += u_streams[3 * shape + 2][0] * grad_ref_z[q * NS + shape];
          grad_h_ref8_values[0] += h_streams[3 * shape + 2][0] * grad_ref_z[q * NS + shape];
        }
      }
      s_t gu_base_v[9 * VS];
      s_t trial_grad_v[9 * VS];
      {
      const ptrdiff_t goff = q * geometry_stride + 0;
      const s_t adj_value0 = adj0[goff];
      const s_t adj_value1 = adj1[goff];
      const s_t adj_value2 = adj2[goff];
      const s_t adj_value3 = adj3[goff];
      const s_t adj_value4 = adj4[goff];
      const s_t adj_value5 = adj5[goff];
      const s_t adj_value6 = adj6[goff];
      const s_t adj_value7 = adj7[goff];
      const s_t adj_value8 = adj8[goff];
      const s_t det_value0 = det0[goff];
      const s_t gu_ref0 = gu_ref0_values[0];
      const s_t grad_h_ref0 = grad_h_ref0_values[0];
      const s_t gu_ref1 = gu_ref1_values[0];
      const s_t grad_h_ref1 = grad_h_ref1_values[0];
      const s_t gu_ref2 = gu_ref2_values[0];
      const s_t grad_h_ref2 = grad_h_ref2_values[0];
      const s_t gu_ref3 = gu_ref3_values[0];
      const s_t grad_h_ref3 = grad_h_ref3_values[0];
      const s_t gu_ref4 = gu_ref4_values[0];
      const s_t grad_h_ref4 = grad_h_ref4_values[0];
      const s_t gu_ref5 = gu_ref5_values[0];
      const s_t grad_h_ref5 = grad_h_ref5_values[0];
      const s_t gu_ref6 = gu_ref6_values[0];
      const s_t grad_h_ref6 = grad_h_ref6_values[0];
      const s_t gu_ref7 = gu_ref7_values[0];
      const s_t grad_h_ref7 = grad_h_ref7_values[0];
      const s_t gu_ref8 = gu_ref8_values[0];
      const s_t grad_h_ref8 = grad_h_ref8_values[0];
    const s_t idet = s_t(1) / det_value0;
    gu_base_v[0 * VS + 0] = (gu_ref0 * adj_value0 + gu_ref1 * adj_value3 + gu_ref2 * adj_value6) * idet;
    trial_grad_v[0 * VS + 0] = (grad_h_ref0 * adj_value0 + grad_h_ref1 * adj_value3 + grad_h_ref2 * adj_value6) * idet;
    gu_base_v[1 * VS + 0] = (gu_ref0 * adj_value1 + gu_ref1 * adj_value4 + gu_ref2 * adj_value7) * idet;
    trial_grad_v[1 * VS + 0] = (grad_h_ref0 * adj_value1 + grad_h_ref1 * adj_value4 + grad_h_ref2 * adj_value7) * idet;
    gu_base_v[2 * VS + 0] = (gu_ref0 * adj_value2 + gu_ref1 * adj_value5 + gu_ref2 * adj_value8) * idet;
    trial_grad_v[2 * VS + 0] = (grad_h_ref0 * adj_value2 + grad_h_ref1 * adj_value5 + grad_h_ref2 * adj_value8) * idet;
    gu_base_v[3 * VS + 0] = (gu_ref3 * adj_value0 + gu_ref4 * adj_value3 + gu_ref5 * adj_value6) * idet;
    trial_grad_v[3 * VS + 0] = (grad_h_ref3 * adj_value0 + grad_h_ref4 * adj_value3 + grad_h_ref5 * adj_value6) * idet;
    gu_base_v[4 * VS + 0] = (gu_ref3 * adj_value1 + gu_ref4 * adj_value4 + gu_ref5 * adj_value7) * idet;
    trial_grad_v[4 * VS + 0] = (grad_h_ref3 * adj_value1 + grad_h_ref4 * adj_value4 + grad_h_ref5 * adj_value7) * idet;
    gu_base_v[5 * VS + 0] = (gu_ref3 * adj_value2 + gu_ref4 * adj_value5 + gu_ref5 * adj_value8) * idet;
    trial_grad_v[5 * VS + 0] = (grad_h_ref3 * adj_value2 + grad_h_ref4 * adj_value5 + grad_h_ref5 * adj_value8) * idet;
    gu_base_v[6 * VS + 0] = (gu_ref6 * adj_value0 + gu_ref7 * adj_value3 + gu_ref8 * adj_value6) * idet;
    trial_grad_v[6 * VS + 0] = (grad_h_ref6 * adj_value0 + grad_h_ref7 * adj_value3 + grad_h_ref8 * adj_value6) * idet;
    gu_base_v[7 * VS + 0] = (gu_ref6 * adj_value1 + gu_ref7 * adj_value4 + gu_ref8 * adj_value7) * idet;
    trial_grad_v[7 * VS + 0] = (grad_h_ref6 * adj_value1 + grad_h_ref7 * adj_value4 + grad_h_ref8 * adj_value7) * idet;
    gu_base_v[8 * VS + 0] = (gu_ref6 * adj_value2 + gu_ref7 * adj_value5 + gu_ref8 * adj_value8) * idet;
    trial_grad_v[8 * VS + 0] = (grad_h_ref6 * adj_value2 + grad_h_ref7 * adj_value5 + grad_h_ref8 * adj_value8) * idet;
      }
      for (int step = 0; step < nsteps; ++step) {
        const s_t alpha = steps[step];
        {
          const ptrdiff_t goff = q * geometry_stride + 0;
          const s_t det_value0 = det0[goff];
          const s_t gu0 = gu_base_v[0 * VS + 0] + alpha * trial_grad_v[0 * VS + 0];
          const s_t gu1 = gu_base_v[1 * VS + 0] + alpha * trial_grad_v[1 * VS + 0];
          const s_t gu2 = gu_base_v[2 * VS + 0] + alpha * trial_grad_v[2 * VS + 0];
          const s_t gu3 = gu_base_v[3 * VS + 0] + alpha * trial_grad_v[3 * VS + 0];
          const s_t gu4 = gu_base_v[4 * VS + 0] + alpha * trial_grad_v[4 * VS + 0];
          const s_t gu5 = gu_base_v[5 * VS + 0] + alpha * trial_grad_v[5 * VS + 0];
          const s_t gu6 = gu_base_v[6 * VS + 0] + alpha * trial_grad_v[6 * VS + 0];
          const s_t gu7 = gu_base_v[7 * VS + 0] + alpha * trial_grad_v[7 * VS + 0];
          const s_t gu8 = gu_base_v[8 * VS + 0] + alpha * trial_grad_v[8 * VS + 0];
    const s_t weak_obj_tmp0 = ((s_t(1) / s_t(2)))*pow_2(gu0) + gu0 + ((s_t(1) / s_t(2)))*pow_2(gu3) + ((s_t(1) / s_t(2)))*pow_2(gu6);
    const s_t weak_obj_tmp1 = ((s_t(1) / s_t(2)))*pow_2(gu1) + ((s_t(1) / s_t(2)))*pow_2(gu4) + gu4 + ((s_t(1) / s_t(2)))*pow_2(gu7);
    const s_t weak_obj_tmp2 = ((s_t(1) / s_t(2)))*pow_2(gu2) + ((s_t(1) / s_t(2)))*pow_2(gu5) + ((s_t(1) / s_t(2)))*pow_2(gu8) + gu8;
    const s_t weak_obj_tmp3 = ((s_t(1) / s_t(2)))*gu1;
    const s_t weak_obj_tmp4 = ((s_t(1) / s_t(2)))*gu4 + (s_t(1) / s_t(2));
    const s_t weak_obj_tmp5 = gu8 + s_t(1);
    const s_t weak_obj_tmp6 = ((s_t(1) / s_t(2)))*gu7;
    const s_t weak_obj_tmp7 = gu0 + s_t(1);
    value[step * value_stride + 0] += qw * det_value0 * (((s_t(1) / s_t(2)))*lmbda*pow_2(weak_obj_tmp0 + weak_obj_tmp1 + weak_obj_tmp2) + mu*(pow_2(weak_obj_tmp0) + pow_2(weak_obj_tmp1) + pow_2(weak_obj_tmp2) + s_t(2)*pow_2(gu2*weak_obj_tmp3 + gu5*weak_obj_tmp4 + weak_obj_tmp5*weak_obj_tmp6) + s_t(2)*pow_2(((s_t(1) / s_t(2)))*gu2*weak_obj_tmp7 + ((s_t(1) / s_t(2)))*gu3*gu5 + ((s_t(1) / s_t(2)))*gu6*weak_obj_tmp5) + s_t(2)*pow_2(gu3*weak_obj_tmp4 + gu6*weak_obj_tmp6 + weak_obj_tmp3*weak_obj_tmp7)));
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void saint_venant_kirchhoff_d3_simplex_tet4_objective_block(
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
      {
      const ptrdiff_t goff = q * geometry_stride + 0;
      const s_t adj_value0 = adj0[goff];
      const s_t adj_value1 = adj1[goff];
      const s_t adj_value2 = adj2[goff];
      const s_t adj_value3 = adj3[goff];
      const s_t adj_value4 = adj4[goff];
      const s_t adj_value5 = adj5[goff];
      const s_t adj_value6 = adj6[goff];
      const s_t adj_value7 = adj7[goff];
      const s_t adj_value8 = adj8[goff];
      const s_t det_value0 = det0[goff];
      const s_t gu_ref0 = -(u_streams[0][0]) + u_streams[3][0];
      const s_t grad_h_ref0 = -(h_streams[0][0]) + h_streams[3][0];
      const s_t gu_ref1 = -(u_streams[0][0]) + u_streams[6][0];
      const s_t grad_h_ref1 = -(h_streams[0][0]) + h_streams[6][0];
      const s_t gu_ref2 = -(u_streams[0][0]) + u_streams[9][0];
      const s_t grad_h_ref2 = -(h_streams[0][0]) + h_streams[9][0];
      const s_t gu_ref3 = -(u_streams[1][0]) + u_streams[4][0];
      const s_t grad_h_ref3 = -(h_streams[1][0]) + h_streams[4][0];
      const s_t gu_ref4 = -(u_streams[1][0]) + u_streams[7][0];
      const s_t grad_h_ref4 = -(h_streams[1][0]) + h_streams[7][0];
      const s_t gu_ref5 = -(u_streams[1][0]) + u_streams[10][0];
      const s_t grad_h_ref5 = -(h_streams[1][0]) + h_streams[10][0];
      const s_t gu_ref6 = -(u_streams[2][0]) + u_streams[5][0];
      const s_t grad_h_ref6 = -(h_streams[2][0]) + h_streams[5][0];
      const s_t gu_ref7 = -(u_streams[2][0]) + u_streams[8][0];
      const s_t grad_h_ref7 = -(h_streams[2][0]) + h_streams[8][0];
      const s_t gu_ref8 = -(u_streams[2][0]) + u_streams[11][0];
      const s_t grad_h_ref8 = -(h_streams[2][0]) + h_streams[11][0];
      const s_t idet = s_t(1) / det_value0;
      gu_base_v[0 * VS + 0] = (gu_ref0 * adj_value0 + gu_ref1 * adj_value3 + gu_ref2 * adj_value6) * idet;
      trial_grad_v[0 * VS + 0] = (grad_h_ref0 * adj_value0 + grad_h_ref1 * adj_value3 + grad_h_ref2 * adj_value6) * idet;
      gu_base_v[1 * VS + 0] = (gu_ref0 * adj_value1 + gu_ref1 * adj_value4 + gu_ref2 * adj_value7) * idet;
      trial_grad_v[1 * VS + 0] = (grad_h_ref0 * adj_value1 + grad_h_ref1 * adj_value4 + grad_h_ref2 * adj_value7) * idet;
      gu_base_v[2 * VS + 0] = (gu_ref0 * adj_value2 + gu_ref1 * adj_value5 + gu_ref2 * adj_value8) * idet;
      trial_grad_v[2 * VS + 0] = (grad_h_ref0 * adj_value2 + grad_h_ref1 * adj_value5 + grad_h_ref2 * adj_value8) * idet;
      gu_base_v[3 * VS + 0] = (gu_ref3 * adj_value0 + gu_ref4 * adj_value3 + gu_ref5 * adj_value6) * idet;
      trial_grad_v[3 * VS + 0] = (grad_h_ref3 * adj_value0 + grad_h_ref4 * adj_value3 + grad_h_ref5 * adj_value6) * idet;
      gu_base_v[4 * VS + 0] = (gu_ref3 * adj_value1 + gu_ref4 * adj_value4 + gu_ref5 * adj_value7) * idet;
      trial_grad_v[4 * VS + 0] = (grad_h_ref3 * adj_value1 + grad_h_ref4 * adj_value4 + grad_h_ref5 * adj_value7) * idet;
      gu_base_v[5 * VS + 0] = (gu_ref3 * adj_value2 + gu_ref4 * adj_value5 + gu_ref5 * adj_value8) * idet;
      trial_grad_v[5 * VS + 0] = (grad_h_ref3 * adj_value2 + grad_h_ref4 * adj_value5 + grad_h_ref5 * adj_value8) * idet;
      gu_base_v[6 * VS + 0] = (gu_ref6 * adj_value0 + gu_ref7 * adj_value3 + gu_ref8 * adj_value6) * idet;
      trial_grad_v[6 * VS + 0] = (grad_h_ref6 * adj_value0 + grad_h_ref7 * adj_value3 + grad_h_ref8 * adj_value6) * idet;
      gu_base_v[7 * VS + 0] = (gu_ref6 * adj_value1 + gu_ref7 * adj_value4 + gu_ref8 * adj_value7) * idet;
      trial_grad_v[7 * VS + 0] = (grad_h_ref6 * adj_value1 + grad_h_ref7 * adj_value4 + grad_h_ref8 * adj_value7) * idet;
      gu_base_v[8 * VS + 0] = (gu_ref6 * adj_value2 + gu_ref7 * adj_value5 + gu_ref8 * adj_value8) * idet;
      trial_grad_v[8 * VS + 0] = (grad_h_ref6 * adj_value2 + grad_h_ref7 * adj_value5 + grad_h_ref8 * adj_value8) * idet;
      }
      for (int step = 0; step < nsteps; ++step) {
        const s_t alpha = steps[step];
        {
          const ptrdiff_t goff = q * geometry_stride + 0;
          const s_t det_value0 = det0[goff];
          const s_t gu0 = gu_base_v[0 * VS + 0] + alpha * trial_grad_v[0 * VS + 0];
          const s_t gu1 = gu_base_v[1 * VS + 0] + alpha * trial_grad_v[1 * VS + 0];
          const s_t gu2 = gu_base_v[2 * VS + 0] + alpha * trial_grad_v[2 * VS + 0];
          const s_t gu3 = gu_base_v[3 * VS + 0] + alpha * trial_grad_v[3 * VS + 0];
          const s_t gu4 = gu_base_v[4 * VS + 0] + alpha * trial_grad_v[4 * VS + 0];
          const s_t gu5 = gu_base_v[5 * VS + 0] + alpha * trial_grad_v[5 * VS + 0];
          const s_t gu6 = gu_base_v[6 * VS + 0] + alpha * trial_grad_v[6 * VS + 0];
          const s_t gu7 = gu_base_v[7 * VS + 0] + alpha * trial_grad_v[7 * VS + 0];
          const s_t gu8 = gu_base_v[8 * VS + 0] + alpha * trial_grad_v[8 * VS + 0];
    const s_t weak_obj_tmp0 = ((s_t(1) / s_t(2)))*pow_2(gu0) + gu0 + ((s_t(1) / s_t(2)))*pow_2(gu3) + ((s_t(1) / s_t(2)))*pow_2(gu6);
    const s_t weak_obj_tmp1 = ((s_t(1) / s_t(2)))*pow_2(gu1) + ((s_t(1) / s_t(2)))*pow_2(gu4) + gu4 + ((s_t(1) / s_t(2)))*pow_2(gu7);
    const s_t weak_obj_tmp2 = ((s_t(1) / s_t(2)))*pow_2(gu2) + ((s_t(1) / s_t(2)))*pow_2(gu5) + ((s_t(1) / s_t(2)))*pow_2(gu8) + gu8;
    const s_t weak_obj_tmp3 = ((s_t(1) / s_t(2)))*gu1;
    const s_t weak_obj_tmp4 = ((s_t(1) / s_t(2)))*gu4 + (s_t(1) / s_t(2));
    const s_t weak_obj_tmp5 = gu8 + s_t(1);
    const s_t weak_obj_tmp6 = ((s_t(1) / s_t(2)))*gu7;
    const s_t weak_obj_tmp7 = gu0 + s_t(1);
    value[step * value_stride + 0] += qw * det_value0 * (((s_t(1) / s_t(2)))*lmbda*pow_2(weak_obj_tmp0 + weak_obj_tmp1 + weak_obj_tmp2) + mu*(pow_2(weak_obj_tmp0) + pow_2(weak_obj_tmp1) + pow_2(weak_obj_tmp2) + s_t(2)*pow_2(gu2*weak_obj_tmp3 + gu5*weak_obj_tmp4 + weak_obj_tmp5*weak_obj_tmp6) + s_t(2)*pow_2(((s_t(1) / s_t(2)))*gu2*weak_obj_tmp7 + ((s_t(1) / s_t(2)))*gu3*gu5 + ((s_t(1) / s_t(2)))*gu6*weak_obj_tmp5) + s_t(2)*pow_2(gu3*weak_obj_tmp4 + gu6*weak_obj_tmp6 + weak_obj_tmp3*weak_obj_tmp7)));
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void saint_venant_kirchhoff_d3_simplex_gradient_block(
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
      {
        gu_ref0_values[0] = s_t(0);
        gu_ref1_values[0] = s_t(0);
        gu_ref2_values[0] = s_t(0);
        gu_ref3_values[0] = s_t(0);
        gu_ref4_values[0] = s_t(0);
        gu_ref5_values[0] = s_t(0);
        gu_ref6_values[0] = s_t(0);
        gu_ref7_values[0] = s_t(0);
        gu_ref8_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        {
          gu_ref0_values[0] += u_streams[3 * shape][0] * grad_ref_x[q * NS + shape];
        }
        {
          gu_ref1_values[0] += u_streams[3 * shape][0] * grad_ref_y[q * NS + shape];
        }
        {
          gu_ref2_values[0] += u_streams[3 * shape][0] * grad_ref_z[q * NS + shape];
        }
        {
          gu_ref3_values[0] += u_streams[3 * shape + 1][0] * grad_ref_x[q * NS + shape];
        }
        {
          gu_ref4_values[0] += u_streams[3 * shape + 1][0] * grad_ref_y[q * NS + shape];
        }
        {
          gu_ref5_values[0] += u_streams[3 * shape + 1][0] * grad_ref_z[q * NS + shape];
        }
        {
          gu_ref6_values[0] += u_streams[3 * shape + 2][0] * grad_ref_x[q * NS + shape];
        }
        {
          gu_ref7_values[0] += u_streams[3 * shape + 2][0] * grad_ref_y[q * NS + shape];
        }
        {
          gu_ref8_values[0] += u_streams[3 * shape + 2][0] * grad_ref_z[q * NS + shape];
        }
      }
      {
      const ptrdiff_t goff = q * geometry_stride + 0;
      const s_t adj_value0 = adj0[goff];
      const s_t adj_value1 = adj1[goff];
      const s_t adj_value2 = adj2[goff];
      const s_t adj_value3 = adj3[goff];
      const s_t adj_value4 = adj4[goff];
      const s_t adj_value5 = adj5[goff];
      const s_t adj_value6 = adj6[goff];
      const s_t adj_value7 = adj7[goff];
      const s_t adj_value8 = adj8[goff];
      const s_t det_value0 = det0[goff];
      const s_t gu_ref0 = gu_ref0_values[0];
      const s_t gu_ref1 = gu_ref1_values[0];
      const s_t gu_ref2 = gu_ref2_values[0];
      const s_t gu_ref3 = gu_ref3_values[0];
      const s_t gu_ref4 = gu_ref4_values[0];
      const s_t gu_ref5 = gu_ref5_values[0];
      const s_t gu_ref6 = gu_ref6_values[0];
      const s_t gu_ref7 = gu_ref7_values[0];
      const s_t gu_ref8 = gu_ref8_values[0];
    const s_t idet = s_t(1) / det_value0;
    const s_t gu0 = (gu_ref0 * adj_value0 + gu_ref1 * adj_value3 + gu_ref2 * adj_value6) * idet;
    const s_t gu1 = (gu_ref0 * adj_value1 + gu_ref1 * adj_value4 + gu_ref2 * adj_value7) * idet;
    const s_t gu2 = (gu_ref0 * adj_value2 + gu_ref1 * adj_value5 + gu_ref2 * adj_value8) * idet;
    const s_t gu3 = (gu_ref3 * adj_value0 + gu_ref4 * adj_value3 + gu_ref5 * adj_value6) * idet;
    const s_t gu4 = (gu_ref3 * adj_value1 + gu_ref4 * adj_value4 + gu_ref5 * adj_value7) * idet;
    const s_t gu5 = (gu_ref3 * adj_value2 + gu_ref4 * adj_value5 + gu_ref5 * adj_value8) * idet;
    const s_t gu6 = (gu_ref6 * adj_value0 + gu_ref7 * adj_value3 + gu_ref8 * adj_value6) * idet;
    const s_t gu7 = (gu_ref6 * adj_value1 + gu_ref7 * adj_value4 + gu_ref8 * adj_value7) * idet;
    const s_t gu8 = (gu_ref6 * adj_value2 + gu_ref7 * adj_value5 + gu_ref8 * adj_value8) * idet;
    const s_t weak_mat_tmp0 = gu0 + s_t(1);
    const s_t weak_mat_tmp1 = ((s_t(1) / s_t(2)))*pow_2(gu0) + gu0 + ((s_t(1) / s_t(2)))*pow_2(gu3) + ((s_t(1) / s_t(2)))*pow_2(gu6);
    const s_t weak_mat_tmp2 = ((s_t(1) / s_t(2)))*pow_2(gu1) + ((s_t(1) / s_t(2)))*pow_2(gu4) + gu4 + ((s_t(1) / s_t(2)))*pow_2(gu7);
    const s_t weak_mat_tmp3 = ((s_t(1) / s_t(2)))*pow_2(gu2) + ((s_t(1) / s_t(2)))*pow_2(gu5) + ((s_t(1) / s_t(2)))*pow_2(gu8) + gu8;
    const s_t weak_mat_tmp4 = lmbda*(weak_mat_tmp1 + weak_mat_tmp2 + weak_mat_tmp3);
    const s_t weak_mat_tmp5 = ((s_t(1) / s_t(2)))*gu6;
    const s_t weak_mat_tmp6 = ((s_t(1) / s_t(2)))*weak_mat_tmp0;
    const s_t weak_mat_tmp7 = gu4 + s_t(1);
    const s_t weak_mat_tmp8 = ((s_t(1) / s_t(2)))*gu3;
    const s_t weak_mat_tmp9 = gu1*weak_mat_tmp6 + gu7*weak_mat_tmp5 + weak_mat_tmp7*weak_mat_tmp8;
    const s_t weak_mat_tmp10 = s_t(2)*gu1;
    const s_t weak_mat_tmp11 = gu8 + s_t(1);
    const s_t weak_mat_tmp12 = gu2*weak_mat_tmp6 + gu5*weak_mat_tmp8 + weak_mat_tmp11*weak_mat_tmp5;
    const s_t weak_mat_tmp13 = s_t(2)*gu2;
    const s_t weak_mat_tmp14 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp15 = ((s_t(1) / s_t(2)))*gu1*gu2 + ((s_t(1) / s_t(2)))*gu5*weak_mat_tmp7 + ((s_t(1) / s_t(2)))*gu7*weak_mat_tmp11;
    const s_t weak_mat_tmp16 = s_t(2)*gu3;
    const s_t weak_mat_tmp17 = s_t(2)*gu5;
    const s_t weak_mat_tmp18 = s_t(2)*weak_mat_tmp7;
    const s_t weak_mat_tmp19 = s_t(2)*gu6;
    const s_t weak_mat_tmp20 = s_t(2)*gu7;
    const s_t weak_mat_tmp21 = s_t(2)*weak_mat_tmp11;
    const s_t material0 = mu*(weak_mat_tmp1*weak_mat_tmp14 + weak_mat_tmp10*weak_mat_tmp9 + weak_mat_tmp12*weak_mat_tmp13) + weak_mat_tmp0*weak_mat_tmp4;
    const s_t material1 = gu1*weak_mat_tmp4 + mu*(weak_mat_tmp10*weak_mat_tmp2 + weak_mat_tmp13*weak_mat_tmp15 + weak_mat_tmp14*weak_mat_tmp9);
    const s_t material2 = gu2*weak_mat_tmp4 + mu*(weak_mat_tmp10*weak_mat_tmp15 + weak_mat_tmp12*weak_mat_tmp14 + weak_mat_tmp13*weak_mat_tmp3);
    const s_t material3 = gu3*weak_mat_tmp4 + mu*(weak_mat_tmp1*weak_mat_tmp16 + weak_mat_tmp12*weak_mat_tmp17 + weak_mat_tmp18*weak_mat_tmp9);
    const s_t material4 = mu*(weak_mat_tmp15*weak_mat_tmp17 + weak_mat_tmp16*weak_mat_tmp9 + weak_mat_tmp18*weak_mat_tmp2) + weak_mat_tmp4*weak_mat_tmp7;
    const s_t material5 = gu5*weak_mat_tmp4 + mu*(weak_mat_tmp12*weak_mat_tmp16 + weak_mat_tmp15*weak_mat_tmp18 + weak_mat_tmp17*weak_mat_tmp3);
    const s_t material6 = gu6*weak_mat_tmp4 + mu*(weak_mat_tmp1*weak_mat_tmp19 + weak_mat_tmp12*weak_mat_tmp21 + weak_mat_tmp20*weak_mat_tmp9);
    const s_t material7 = gu7*weak_mat_tmp4 + mu*(weak_mat_tmp15*weak_mat_tmp21 + weak_mat_tmp19*weak_mat_tmp9 + weak_mat_tmp2*weak_mat_tmp20);
    const s_t material8 = mu*(weak_mat_tmp12*weak_mat_tmp19 + weak_mat_tmp15*weak_mat_tmp20 + weak_mat_tmp21*weak_mat_tmp3) + weak_mat_tmp11*weak_mat_tmp4;
    const s_t loperand0 = qw * (material0 * adj_value0 + material1 * adj_value1 + material2 * adj_value2);
    const s_t loperand1 = qw * (material0 * adj_value3 + material1 * adj_value4 + material2 * adj_value5);
    const s_t loperand2 = qw * (material0 * adj_value6 + material1 * adj_value7 + material2 * adj_value8);
    const s_t loperand3 = qw * (material3 * adj_value0 + material4 * adj_value1 + material5 * adj_value2);
    const s_t loperand4 = qw * (material3 * adj_value3 + material4 * adj_value4 + material5 * adj_value5);
    const s_t loperand5 = qw * (material3 * adj_value6 + material4 * adj_value7 + material5 * adj_value8);
    const s_t loperand6 = qw * (material6 * adj_value0 + material7 * adj_value1 + material8 * adj_value2);
    const s_t loperand7 = qw * (material6 * adj_value3 + material7 * adj_value4 + material8 * adj_value5);
    const s_t loperand8 = qw * (material6 * adj_value6 + material7 * adj_value7 + material8 * adj_value8);
      loperand0_values[0] = loperand0;
      loperand1_values[0] = loperand1;
      loperand2_values[0] = loperand2;
      loperand3_values[0] = loperand3;
      loperand4_values[0] = loperand4;
      loperand5_values[0] = loperand5;
      loperand6_values[0] = loperand6;
      loperand7_values[0] = loperand7;
      loperand8_values[0] = loperand8;
      }
      for (int shape = 0; shape < NS; ++shape) {
        {
          out_streams[3 * shape][0] += loperand0_values[0] * grad_ref_x[q * NS + shape] + loperand1_values[0] * grad_ref_y[q * NS + shape] + loperand2_values[0] * grad_ref_z[q * NS + shape];
        }
        {
          out_streams[3 * shape + 1][0] += loperand3_values[0] * grad_ref_x[q * NS + shape] + loperand4_values[0] * grad_ref_y[q * NS + shape] + loperand5_values[0] * grad_ref_z[q * NS + shape];
        }
        {
          out_streams[3 * shape + 2][0] += loperand6_values[0] * grad_ref_x[q * NS + shape] + loperand7_values[0] * grad_ref_y[q * NS + shape] + loperand8_values[0] * grad_ref_z[q * NS + shape];
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void saint_venant_kirchhoff_d3_simplex_tet4_gradient_block(
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
      {
      const ptrdiff_t goff = q * geometry_stride + 0;
      const s_t adj_value0 = adj0[goff];
      const s_t adj_value1 = adj1[goff];
      const s_t adj_value2 = adj2[goff];
      const s_t adj_value3 = adj3[goff];
      const s_t adj_value4 = adj4[goff];
      const s_t adj_value5 = adj5[goff];
      const s_t adj_value6 = adj6[goff];
      const s_t adj_value7 = adj7[goff];
      const s_t adj_value8 = adj8[goff];
      const s_t det_value0 = det0[goff];
      const s_t gu_ref0 = -(u_streams[0][0]) + u_streams[3][0];
      const s_t gu_ref1 = -(u_streams[0][0]) + u_streams[6][0];
      const s_t gu_ref2 = -(u_streams[0][0]) + u_streams[9][0];
      const s_t gu_ref3 = -(u_streams[1][0]) + u_streams[4][0];
      const s_t gu_ref4 = -(u_streams[1][0]) + u_streams[7][0];
      const s_t gu_ref5 = -(u_streams[1][0]) + u_streams[10][0];
      const s_t gu_ref6 = -(u_streams[2][0]) + u_streams[5][0];
      const s_t gu_ref7 = -(u_streams[2][0]) + u_streams[8][0];
      const s_t gu_ref8 = -(u_streams[2][0]) + u_streams[11][0];
      const s_t idet = s_t(1) / det_value0;
      const s_t gu0 = (gu_ref0 * adj_value0 + gu_ref1 * adj_value3 + gu_ref2 * adj_value6) * idet;
      const s_t gu1 = (gu_ref0 * adj_value1 + gu_ref1 * adj_value4 + gu_ref2 * adj_value7) * idet;
      const s_t gu2 = (gu_ref0 * adj_value2 + gu_ref1 * adj_value5 + gu_ref2 * adj_value8) * idet;
      const s_t gu3 = (gu_ref3 * adj_value0 + gu_ref4 * adj_value3 + gu_ref5 * adj_value6) * idet;
      const s_t gu4 = (gu_ref3 * adj_value1 + gu_ref4 * adj_value4 + gu_ref5 * adj_value7) * idet;
      const s_t gu5 = (gu_ref3 * adj_value2 + gu_ref4 * adj_value5 + gu_ref5 * adj_value8) * idet;
      const s_t gu6 = (gu_ref6 * adj_value0 + gu_ref7 * adj_value3 + gu_ref8 * adj_value6) * idet;
      const s_t gu7 = (gu_ref6 * adj_value1 + gu_ref7 * adj_value4 + gu_ref8 * adj_value7) * idet;
      const s_t gu8 = (gu_ref6 * adj_value2 + gu_ref7 * adj_value5 + gu_ref8 * adj_value8) * idet;
    const s_t weak_mat_tmp0 = gu0 + s_t(1);
    const s_t weak_mat_tmp1 = ((s_t(1) / s_t(2)))*pow_2(gu0) + gu0 + ((s_t(1) / s_t(2)))*pow_2(gu3) + ((s_t(1) / s_t(2)))*pow_2(gu6);
    const s_t weak_mat_tmp2 = ((s_t(1) / s_t(2)))*pow_2(gu1) + ((s_t(1) / s_t(2)))*pow_2(gu4) + gu4 + ((s_t(1) / s_t(2)))*pow_2(gu7);
    const s_t weak_mat_tmp3 = ((s_t(1) / s_t(2)))*pow_2(gu2) + ((s_t(1) / s_t(2)))*pow_2(gu5) + ((s_t(1) / s_t(2)))*pow_2(gu8) + gu8;
    const s_t weak_mat_tmp4 = lmbda*(weak_mat_tmp1 + weak_mat_tmp2 + weak_mat_tmp3);
    const s_t weak_mat_tmp5 = ((s_t(1) / s_t(2)))*gu6;
    const s_t weak_mat_tmp6 = ((s_t(1) / s_t(2)))*weak_mat_tmp0;
    const s_t weak_mat_tmp7 = gu4 + s_t(1);
    const s_t weak_mat_tmp8 = ((s_t(1) / s_t(2)))*gu3;
    const s_t weak_mat_tmp9 = gu1*weak_mat_tmp6 + gu7*weak_mat_tmp5 + weak_mat_tmp7*weak_mat_tmp8;
    const s_t weak_mat_tmp10 = s_t(2)*gu1;
    const s_t weak_mat_tmp11 = gu8 + s_t(1);
    const s_t weak_mat_tmp12 = gu2*weak_mat_tmp6 + gu5*weak_mat_tmp8 + weak_mat_tmp11*weak_mat_tmp5;
    const s_t weak_mat_tmp13 = s_t(2)*gu2;
    const s_t weak_mat_tmp14 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp15 = ((s_t(1) / s_t(2)))*gu1*gu2 + ((s_t(1) / s_t(2)))*gu5*weak_mat_tmp7 + ((s_t(1) / s_t(2)))*gu7*weak_mat_tmp11;
    const s_t weak_mat_tmp16 = s_t(2)*gu3;
    const s_t weak_mat_tmp17 = s_t(2)*gu5;
    const s_t weak_mat_tmp18 = s_t(2)*weak_mat_tmp7;
    const s_t weak_mat_tmp19 = s_t(2)*gu6;
    const s_t weak_mat_tmp20 = s_t(2)*gu7;
    const s_t weak_mat_tmp21 = s_t(2)*weak_mat_tmp11;
    const s_t material0 = mu*(weak_mat_tmp1*weak_mat_tmp14 + weak_mat_tmp10*weak_mat_tmp9 + weak_mat_tmp12*weak_mat_tmp13) + weak_mat_tmp0*weak_mat_tmp4;
    const s_t material1 = gu1*weak_mat_tmp4 + mu*(weak_mat_tmp10*weak_mat_tmp2 + weak_mat_tmp13*weak_mat_tmp15 + weak_mat_tmp14*weak_mat_tmp9);
    const s_t material2 = gu2*weak_mat_tmp4 + mu*(weak_mat_tmp10*weak_mat_tmp15 + weak_mat_tmp12*weak_mat_tmp14 + weak_mat_tmp13*weak_mat_tmp3);
    const s_t material3 = gu3*weak_mat_tmp4 + mu*(weak_mat_tmp1*weak_mat_tmp16 + weak_mat_tmp12*weak_mat_tmp17 + weak_mat_tmp18*weak_mat_tmp9);
    const s_t material4 = mu*(weak_mat_tmp15*weak_mat_tmp17 + weak_mat_tmp16*weak_mat_tmp9 + weak_mat_tmp18*weak_mat_tmp2) + weak_mat_tmp4*weak_mat_tmp7;
    const s_t material5 = gu5*weak_mat_tmp4 + mu*(weak_mat_tmp12*weak_mat_tmp16 + weak_mat_tmp15*weak_mat_tmp18 + weak_mat_tmp17*weak_mat_tmp3);
    const s_t material6 = gu6*weak_mat_tmp4 + mu*(weak_mat_tmp1*weak_mat_tmp19 + weak_mat_tmp12*weak_mat_tmp21 + weak_mat_tmp20*weak_mat_tmp9);
    const s_t material7 = gu7*weak_mat_tmp4 + mu*(weak_mat_tmp15*weak_mat_tmp21 + weak_mat_tmp19*weak_mat_tmp9 + weak_mat_tmp2*weak_mat_tmp20);
    const s_t material8 = mu*(weak_mat_tmp12*weak_mat_tmp19 + weak_mat_tmp15*weak_mat_tmp20 + weak_mat_tmp21*weak_mat_tmp3) + weak_mat_tmp11*weak_mat_tmp4;
    const s_t loperand0 = qw * (material0 * adj_value0 + material1 * adj_value1 + material2 * adj_value2);
    const s_t loperand1 = qw * (material0 * adj_value3 + material1 * adj_value4 + material2 * adj_value5);
    const s_t loperand2 = qw * (material0 * adj_value6 + material1 * adj_value7 + material2 * adj_value8);
    const s_t loperand3 = qw * (material3 * adj_value0 + material4 * adj_value1 + material5 * adj_value2);
    const s_t loperand4 = qw * (material3 * adj_value3 + material4 * adj_value4 + material5 * adj_value5);
    const s_t loperand5 = qw * (material3 * adj_value6 + material4 * adj_value7 + material5 * adj_value8);
    const s_t loperand6 = qw * (material6 * adj_value0 + material7 * adj_value1 + material8 * adj_value2);
    const s_t loperand7 = qw * (material6 * adj_value3 + material7 * adj_value4 + material8 * adj_value5);
    const s_t loperand8 = qw * (material6 * adj_value6 + material7 * adj_value7 + material8 * adj_value8);
      out_streams[0][0] += -(loperand0) - loperand1 - loperand2;
      out_streams[1][0] += -(loperand3) - loperand4 - loperand5;
      out_streams[2][0] += -(loperand6) - loperand7 - loperand8;
      out_streams[3][0] += loperand0;
      out_streams[4][0] += loperand3;
      out_streams[5][0] += loperand6;
      out_streams[6][0] += loperand1;
      out_streams[7][0] += loperand4;
      out_streams[8][0] += loperand7;
      out_streams[9][0] += loperand2;
      out_streams[10][0] += loperand5;
      out_streams[11][0] += loperand8;
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void saint_venant_kirchhoff_d3_simplex_apply_block(
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
      {
        gu_ref0_values[0] = s_t(0);
        grad_h_ref0_values[0] = s_t(0);
        gu_ref1_values[0] = s_t(0);
        grad_h_ref1_values[0] = s_t(0);
        gu_ref2_values[0] = s_t(0);
        grad_h_ref2_values[0] = s_t(0);
        gu_ref3_values[0] = s_t(0);
        grad_h_ref3_values[0] = s_t(0);
        gu_ref4_values[0] = s_t(0);
        grad_h_ref4_values[0] = s_t(0);
        gu_ref5_values[0] = s_t(0);
        grad_h_ref5_values[0] = s_t(0);
        gu_ref6_values[0] = s_t(0);
        grad_h_ref6_values[0] = s_t(0);
        gu_ref7_values[0] = s_t(0);
        grad_h_ref7_values[0] = s_t(0);
        gu_ref8_values[0] = s_t(0);
        grad_h_ref8_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        {
          gu_ref0_values[0] += u_streams[3 * shape][0] * grad_ref_x[q * NS + shape];
          grad_h_ref0_values[0] += h_streams[3 * shape][0] * grad_ref_x[q * NS + shape];
        }
        {
          gu_ref1_values[0] += u_streams[3 * shape][0] * grad_ref_y[q * NS + shape];
          grad_h_ref1_values[0] += h_streams[3 * shape][0] * grad_ref_y[q * NS + shape];
        }
        {
          gu_ref2_values[0] += u_streams[3 * shape][0] * grad_ref_z[q * NS + shape];
          grad_h_ref2_values[0] += h_streams[3 * shape][0] * grad_ref_z[q * NS + shape];
        }
        {
          gu_ref3_values[0] += u_streams[3 * shape + 1][0] * grad_ref_x[q * NS + shape];
          grad_h_ref3_values[0] += h_streams[3 * shape + 1][0] * grad_ref_x[q * NS + shape];
        }
        {
          gu_ref4_values[0] += u_streams[3 * shape + 1][0] * grad_ref_y[q * NS + shape];
          grad_h_ref4_values[0] += h_streams[3 * shape + 1][0] * grad_ref_y[q * NS + shape];
        }
        {
          gu_ref5_values[0] += u_streams[3 * shape + 1][0] * grad_ref_z[q * NS + shape];
          grad_h_ref5_values[0] += h_streams[3 * shape + 1][0] * grad_ref_z[q * NS + shape];
        }
        {
          gu_ref6_values[0] += u_streams[3 * shape + 2][0] * grad_ref_x[q * NS + shape];
          grad_h_ref6_values[0] += h_streams[3 * shape + 2][0] * grad_ref_x[q * NS + shape];
        }
        {
          gu_ref7_values[0] += u_streams[3 * shape + 2][0] * grad_ref_y[q * NS + shape];
          grad_h_ref7_values[0] += h_streams[3 * shape + 2][0] * grad_ref_y[q * NS + shape];
        }
        {
          gu_ref8_values[0] += u_streams[3 * shape + 2][0] * grad_ref_z[q * NS + shape];
          grad_h_ref8_values[0] += h_streams[3 * shape + 2][0] * grad_ref_z[q * NS + shape];
        }
      }
      {
      const ptrdiff_t goff = q * geometry_stride + 0;
      const s_t adj_value0 = adj0[goff];
      const s_t adj_value1 = adj1[goff];
      const s_t adj_value2 = adj2[goff];
      const s_t adj_value3 = adj3[goff];
      const s_t adj_value4 = adj4[goff];
      const s_t adj_value5 = adj5[goff];
      const s_t adj_value6 = adj6[goff];
      const s_t adj_value7 = adj7[goff];
      const s_t adj_value8 = adj8[goff];
      const s_t det_value0 = det0[goff];
      const s_t gu_ref0 = gu_ref0_values[0];
      const s_t grad_h_ref0 = grad_h_ref0_values[0];
      const s_t gu_ref1 = gu_ref1_values[0];
      const s_t grad_h_ref1 = grad_h_ref1_values[0];
      const s_t gu_ref2 = gu_ref2_values[0];
      const s_t grad_h_ref2 = grad_h_ref2_values[0];
      const s_t gu_ref3 = gu_ref3_values[0];
      const s_t grad_h_ref3 = grad_h_ref3_values[0];
      const s_t gu_ref4 = gu_ref4_values[0];
      const s_t grad_h_ref4 = grad_h_ref4_values[0];
      const s_t gu_ref5 = gu_ref5_values[0];
      const s_t grad_h_ref5 = grad_h_ref5_values[0];
      const s_t gu_ref6 = gu_ref6_values[0];
      const s_t grad_h_ref6 = grad_h_ref6_values[0];
      const s_t gu_ref7 = gu_ref7_values[0];
      const s_t grad_h_ref7 = grad_h_ref7_values[0];
      const s_t gu_ref8 = gu_ref8_values[0];
      const s_t grad_h_ref8 = grad_h_ref8_values[0];
    const s_t idet = s_t(1) / det_value0;
    const s_t gu0 = (gu_ref0 * adj_value0 + gu_ref1 * adj_value3 + gu_ref2 * adj_value6) * idet;
    const s_t trial_grad0 = (grad_h_ref0 * adj_value0 + grad_h_ref1 * adj_value3 + grad_h_ref2 * adj_value6) * idet;
    const s_t gu1 = (gu_ref0 * adj_value1 + gu_ref1 * adj_value4 + gu_ref2 * adj_value7) * idet;
    const s_t trial_grad1 = (grad_h_ref0 * adj_value1 + grad_h_ref1 * adj_value4 + grad_h_ref2 * adj_value7) * idet;
    const s_t gu2 = (gu_ref0 * adj_value2 + gu_ref1 * adj_value5 + gu_ref2 * adj_value8) * idet;
    const s_t trial_grad2 = (grad_h_ref0 * adj_value2 + grad_h_ref1 * adj_value5 + grad_h_ref2 * adj_value8) * idet;
    const s_t gu3 = (gu_ref3 * adj_value0 + gu_ref4 * adj_value3 + gu_ref5 * adj_value6) * idet;
    const s_t trial_grad3 = (grad_h_ref3 * adj_value0 + grad_h_ref4 * adj_value3 + grad_h_ref5 * adj_value6) * idet;
    const s_t gu4 = (gu_ref3 * adj_value1 + gu_ref4 * adj_value4 + gu_ref5 * adj_value7) * idet;
    const s_t trial_grad4 = (grad_h_ref3 * adj_value1 + grad_h_ref4 * adj_value4 + grad_h_ref5 * adj_value7) * idet;
    const s_t gu5 = (gu_ref3 * adj_value2 + gu_ref4 * adj_value5 + gu_ref5 * adj_value8) * idet;
    const s_t trial_grad5 = (grad_h_ref3 * adj_value2 + grad_h_ref4 * adj_value5 + grad_h_ref5 * adj_value8) * idet;
    const s_t gu6 = (gu_ref6 * adj_value0 + gu_ref7 * adj_value3 + gu_ref8 * adj_value6) * idet;
    const s_t trial_grad6 = (grad_h_ref6 * adj_value0 + grad_h_ref7 * adj_value3 + grad_h_ref8 * adj_value6) * idet;
    const s_t gu7 = (gu_ref6 * adj_value1 + gu_ref7 * adj_value4 + gu_ref8 * adj_value7) * idet;
    const s_t trial_grad7 = (grad_h_ref6 * adj_value1 + grad_h_ref7 * adj_value4 + grad_h_ref8 * adj_value7) * idet;
    const s_t gu8 = (gu_ref6 * adj_value2 + gu_ref7 * adj_value5 + gu_ref8 * adj_value8) * idet;
    const s_t trial_grad8 = (grad_h_ref6 * adj_value2 + grad_h_ref7 * adj_value5 + grad_h_ref8 * adj_value8) * idet;
    const s_t weak_mat_tmp0 = gu3*mu;
    const s_t weak_mat_tmp1 = gu0 + s_t(1);
    const s_t weak_mat_tmp2 = lmbda*weak_mat_tmp1;
    const s_t weak_mat_tmp3 = gu2*weak_mat_tmp0 + gu5*weak_mat_tmp2;
    const s_t weak_mat_tmp4 = gu6*mu;
    const s_t weak_mat_tmp5 = gu1*weak_mat_tmp4 + gu7*weak_mat_tmp2;
    const s_t weak_mat_tmp6 = gu4 + s_t(1);
    const s_t weak_mat_tmp7 = gu1*weak_mat_tmp0 + weak_mat_tmp2*weak_mat_tmp6;
    const s_t weak_mat_tmp8 = gu8 + s_t(1);
    const s_t weak_mat_tmp9 = gu2*weak_mat_tmp4 + weak_mat_tmp2*weak_mat_tmp8;
    const s_t weak_mat_tmp10 = gu1*weak_mat_tmp1;
    const s_t weak_mat_tmp11 = gu6*gu7;
    const s_t weak_mat_tmp12 = gu3*weak_mat_tmp6;
    const s_t weak_mat_tmp13 = lmbda*weak_mat_tmp10 + mu*(s_t(2)*weak_mat_tmp10 + weak_mat_tmp11 + weak_mat_tmp12);
    const s_t weak_mat_tmp14 = gu2*weak_mat_tmp1;
    const s_t weak_mat_tmp15 = gu3*gu5;
    const s_t weak_mat_tmp16 = gu6*weak_mat_tmp8;
    const s_t weak_mat_tmp17 = lmbda*weak_mat_tmp14 + mu*(s_t(2)*weak_mat_tmp14 + weak_mat_tmp15 + weak_mat_tmp16);
    const s_t weak_mat_tmp18 = gu3*weak_mat_tmp1;
    const s_t weak_mat_tmp19 = gu2*gu5;
    const s_t weak_mat_tmp20 = gu1*weak_mat_tmp6;
    const s_t weak_mat_tmp21 = lmbda*weak_mat_tmp18 + mu*(s_t(2)*weak_mat_tmp18 + weak_mat_tmp19 + weak_mat_tmp20);
    const s_t weak_mat_tmp22 = gu6*weak_mat_tmp1;
    const s_t weak_mat_tmp23 = gu1*gu7;
    const s_t weak_mat_tmp24 = gu2*weak_mat_tmp8;
    const s_t weak_mat_tmp25 = lmbda*weak_mat_tmp22 + mu*(s_t(2)*weak_mat_tmp22 + weak_mat_tmp23 + weak_mat_tmp24);
    const s_t weak_mat_tmp26 = pow_2(weak_mat_tmp1);
    const s_t weak_mat_tmp27 = pow_2(gu1);
    const s_t weak_mat_tmp28 = pow_2(gu3);
    const s_t weak_mat_tmp29 = weak_mat_tmp27 + weak_mat_tmp28;
    const s_t weak_mat_tmp30 = pow_2(gu6);
    const s_t weak_mat_tmp31 = pow_2(gu2);
    const s_t weak_mat_tmp32 = weak_mat_tmp31 + s_t(-1);
    const s_t weak_mat_tmp33 = weak_mat_tmp30 + weak_mat_tmp32;
    const s_t weak_mat_tmp34 = pow_2(gu5);
    const s_t weak_mat_tmp35 = pow_2(gu7);
    const s_t weak_mat_tmp36 = lmbda*(((s_t(1) / s_t(2)))*pow_2(gu0) + gu0 + ((s_t(1) / s_t(2)))*pow_2(gu4) + gu4 + ((s_t(1) / s_t(2)))*pow_2(gu8) + gu8 + ((s_t(1) / s_t(2)))*weak_mat_tmp27 + ((s_t(1) / s_t(2)))*weak_mat_tmp28 + ((s_t(1) / s_t(2)))*weak_mat_tmp30 + ((s_t(1) / s_t(2)))*weak_mat_tmp31 + ((s_t(1) / s_t(2)))*weak_mat_tmp34 + ((s_t(1) / s_t(2)))*weak_mat_tmp35);
    const s_t weak_mat_tmp37 = gu1*lmbda;
    const s_t weak_mat_tmp38 = mu*weak_mat_tmp6;
    const s_t weak_mat_tmp39 = gu2*weak_mat_tmp38 + gu5*weak_mat_tmp37;
    const s_t weak_mat_tmp40 = gu7*mu;
    const s_t weak_mat_tmp41 = gu6*weak_mat_tmp37 + weak_mat_tmp1*weak_mat_tmp40;
    const s_t weak_mat_tmp42 = gu2*weak_mat_tmp40 + weak_mat_tmp37*weak_mat_tmp8;
    const s_t weak_mat_tmp43 = gu3*weak_mat_tmp37 + weak_mat_tmp1*weak_mat_tmp38;
    const s_t weak_mat_tmp44 = gu1*gu2;
    const s_t weak_mat_tmp45 = gu5*weak_mat_tmp6;
    const s_t weak_mat_tmp46 = gu7*weak_mat_tmp8;
    const s_t weak_mat_tmp47 = lmbda*weak_mat_tmp44 + mu*(s_t(2)*weak_mat_tmp44 + weak_mat_tmp45 + weak_mat_tmp46);
    const s_t weak_mat_tmp48 = lmbda*weak_mat_tmp23 + mu*(weak_mat_tmp22 + s_t(2)*weak_mat_tmp23 + weak_mat_tmp24);
    const s_t weak_mat_tmp49 = lmbda*weak_mat_tmp20 + mu*(weak_mat_tmp18 + weak_mat_tmp19 + s_t(2)*weak_mat_tmp20);
    const s_t weak_mat_tmp50 = pow_2(weak_mat_tmp6);
    const s_t weak_mat_tmp51 = weak_mat_tmp26 + weak_mat_tmp50;
    const s_t weak_mat_tmp52 = gu2*lmbda;
    const s_t weak_mat_tmp53 = gu5*mu;
    const s_t weak_mat_tmp54 = gu3*weak_mat_tmp52 + weak_mat_tmp1*weak_mat_tmp53;
    const s_t weak_mat_tmp55 = gu1*weak_mat_tmp53 + weak_mat_tmp52*weak_mat_tmp6;
    const s_t weak_mat_tmp56 = mu*weak_mat_tmp8;
    const s_t weak_mat_tmp57 = gu1*weak_mat_tmp56 + gu7*weak_mat_tmp52;
    const s_t weak_mat_tmp58 = gu6*weak_mat_tmp52 + weak_mat_tmp1*weak_mat_tmp56;
    const s_t weak_mat_tmp59 = lmbda*weak_mat_tmp19 + mu*(weak_mat_tmp18 + s_t(2)*weak_mat_tmp19 + weak_mat_tmp20);
    const s_t weak_mat_tmp60 = lmbda*weak_mat_tmp24 + mu*(weak_mat_tmp22 + weak_mat_tmp23 + s_t(2)*weak_mat_tmp24);
    const s_t weak_mat_tmp61 = weak_mat_tmp34 + s_t(-1);
    const s_t weak_mat_tmp62 = pow_2(weak_mat_tmp8);
    const s_t weak_mat_tmp63 = weak_mat_tmp26 + weak_mat_tmp62;
    const s_t weak_mat_tmp64 = gu3*lmbda;
    const s_t weak_mat_tmp65 = gu7*weak_mat_tmp64 + weak_mat_tmp4*weak_mat_tmp6;
    const s_t weak_mat_tmp66 = gu5*weak_mat_tmp4 + weak_mat_tmp64*weak_mat_tmp8;
    const s_t weak_mat_tmp67 = lmbda*weak_mat_tmp15 + mu*(weak_mat_tmp14 + s_t(2)*weak_mat_tmp15 + weak_mat_tmp16);
    const s_t weak_mat_tmp68 = gu3*gu6;
    const s_t weak_mat_tmp69 = gu5*weak_mat_tmp8;
    const s_t weak_mat_tmp70 = gu7*weak_mat_tmp6;
    const s_t weak_mat_tmp71 = lmbda*weak_mat_tmp68 + mu*(s_t(2)*weak_mat_tmp68 + weak_mat_tmp69 + weak_mat_tmp70);
    const s_t weak_mat_tmp72 = lmbda*weak_mat_tmp12 + mu*(weak_mat_tmp10 + weak_mat_tmp11 + s_t(2)*weak_mat_tmp12);
    const s_t weak_mat_tmp73 = lmbda*weak_mat_tmp6;
    const s_t weak_mat_tmp74 = gu6*weak_mat_tmp73 + gu7*weak_mat_tmp0;
    const s_t weak_mat_tmp75 = gu5*weak_mat_tmp40 + weak_mat_tmp73*weak_mat_tmp8;
    const s_t weak_mat_tmp76 = lmbda*weak_mat_tmp45 + mu*(weak_mat_tmp44 + s_t(2)*weak_mat_tmp45 + weak_mat_tmp46);
    const s_t weak_mat_tmp77 = lmbda*weak_mat_tmp70 + mu*(weak_mat_tmp68 + weak_mat_tmp69 + s_t(2)*weak_mat_tmp70);
    const s_t weak_mat_tmp78 = gu5*lmbda;
    const s_t weak_mat_tmp79 = gu6*weak_mat_tmp78 + weak_mat_tmp0*weak_mat_tmp8;
    const s_t weak_mat_tmp80 = gu7*weak_mat_tmp78 + weak_mat_tmp38*weak_mat_tmp8;
    const s_t weak_mat_tmp81 = lmbda*weak_mat_tmp69 + mu*(weak_mat_tmp68 + s_t(2)*weak_mat_tmp69 + weak_mat_tmp70);
    const s_t weak_mat_tmp82 = weak_mat_tmp50 + weak_mat_tmp62;
    const s_t weak_mat_tmp83 = lmbda*weak_mat_tmp11 + mu*(weak_mat_tmp10 + s_t(2)*weak_mat_tmp11 + weak_mat_tmp12);
    const s_t weak_mat_tmp84 = lmbda*weak_mat_tmp16 + mu*(weak_mat_tmp14 + weak_mat_tmp15 + s_t(2)*weak_mat_tmp16);
    const s_t weak_mat_tmp85 = lmbda*weak_mat_tmp46 + mu*(weak_mat_tmp44 + weak_mat_tmp45 + s_t(2)*weak_mat_tmp46);
    const s_t material0 = trial_grad0*(lmbda*weak_mat_tmp26 + mu*(s_t(3)*weak_mat_tmp26 + weak_mat_tmp29 + weak_mat_tmp33) + weak_mat_tmp36) + trial_grad1*weak_mat_tmp13 + trial_grad2*weak_mat_tmp17 + trial_grad3*weak_mat_tmp21 + trial_grad4*weak_mat_tmp7 + trial_grad5*weak_mat_tmp3 + trial_grad6*weak_mat_tmp25 + trial_grad7*weak_mat_tmp5 + trial_grad8*weak_mat_tmp9;
    const s_t material1 = trial_grad0*weak_mat_tmp13 + trial_grad1*(lmbda*weak_mat_tmp27 + mu*(s_t(3)*weak_mat_tmp27 + weak_mat_tmp32 + weak_mat_tmp35 + weak_mat_tmp51) + weak_mat_tmp36) + trial_grad2*weak_mat_tmp47 + trial_grad3*weak_mat_tmp43 + trial_grad4*weak_mat_tmp49 + trial_grad5*weak_mat_tmp39 + trial_grad6*weak_mat_tmp41 + trial_grad7*weak_mat_tmp48 + trial_grad8*weak_mat_tmp42;
    const s_t material2 = trial_grad0*weak_mat_tmp17 + trial_grad1*weak_mat_tmp47 + trial_grad2*(lmbda*weak_mat_tmp31 + mu*(weak_mat_tmp27 + s_t(3)*weak_mat_tmp31 + weak_mat_tmp61 + weak_mat_tmp63) + weak_mat_tmp36) + trial_grad3*weak_mat_tmp54 + trial_grad4*weak_mat_tmp55 + trial_grad5*weak_mat_tmp59 + trial_grad6*weak_mat_tmp58 + trial_grad7*weak_mat_tmp57 + trial_grad8*weak_mat_tmp60;
    const s_t material3 = trial_grad0*weak_mat_tmp21 + trial_grad1*weak_mat_tmp43 + trial_grad2*weak_mat_tmp54 + trial_grad3*(lmbda*weak_mat_tmp28 + mu*(s_t(3)*weak_mat_tmp28 + weak_mat_tmp30 + weak_mat_tmp51 + weak_mat_tmp61) + weak_mat_tmp36) + trial_grad4*weak_mat_tmp72 + trial_grad5*weak_mat_tmp67 + trial_grad6*weak_mat_tmp71 + trial_grad7*weak_mat_tmp65 + trial_grad8*weak_mat_tmp66;
    const s_t material4 = trial_grad0*weak_mat_tmp7 + trial_grad1*weak_mat_tmp49 + trial_grad2*weak_mat_tmp55 + trial_grad3*weak_mat_tmp72 + trial_grad4*(lmbda*weak_mat_tmp50 + mu*(weak_mat_tmp29 + weak_mat_tmp35 + s_t(3)*weak_mat_tmp50 + weak_mat_tmp61) + weak_mat_tmp36) + trial_grad5*weak_mat_tmp76 + trial_grad6*weak_mat_tmp74 + trial_grad7*weak_mat_tmp77 + trial_grad8*weak_mat_tmp75;
    const s_t material5 = trial_grad0*weak_mat_tmp3 + trial_grad1*weak_mat_tmp39 + trial_grad2*weak_mat_tmp59 + trial_grad3*weak_mat_tmp67 + trial_grad4*weak_mat_tmp76 + trial_grad5*(lmbda*weak_mat_tmp34 + mu*(weak_mat_tmp28 + weak_mat_tmp32 + s_t(3)*weak_mat_tmp34 + weak_mat_tmp82) + weak_mat_tmp36) + trial_grad6*weak_mat_tmp79 + trial_grad7*weak_mat_tmp80 + trial_grad8*weak_mat_tmp81;
    const s_t material6 = trial_grad0*weak_mat_tmp25 + trial_grad1*weak_mat_tmp41 + trial_grad2*weak_mat_tmp58 + trial_grad3*weak_mat_tmp71 + trial_grad4*weak_mat_tmp74 + trial_grad5*weak_mat_tmp79 + trial_grad6*(lmbda*weak_mat_tmp30 + mu*(weak_mat_tmp28 + s_t(3)*weak_mat_tmp30 + weak_mat_tmp35 + weak_mat_tmp63 + s_t(-1)) + weak_mat_tmp36) + trial_grad7*weak_mat_tmp83 + trial_grad8*weak_mat_tmp84;
    const s_t material7 = trial_grad0*weak_mat_tmp5 + trial_grad1*weak_mat_tmp48 + trial_grad2*weak_mat_tmp57 + trial_grad3*weak_mat_tmp65 + trial_grad4*weak_mat_tmp77 + trial_grad5*weak_mat_tmp80 + trial_grad6*weak_mat_tmp83 + trial_grad7*(lmbda*weak_mat_tmp35 + mu*(weak_mat_tmp27 + weak_mat_tmp30 + s_t(3)*weak_mat_tmp35 + weak_mat_tmp82 + s_t(-1)) + weak_mat_tmp36) + trial_grad8*weak_mat_tmp85;
    const s_t material8 = trial_grad0*weak_mat_tmp9 + trial_grad1*weak_mat_tmp42 + trial_grad2*weak_mat_tmp60 + trial_grad3*weak_mat_tmp66 + trial_grad4*weak_mat_tmp75 + trial_grad5*weak_mat_tmp81 + trial_grad6*weak_mat_tmp84 + trial_grad7*weak_mat_tmp85 + trial_grad8*(lmbda*weak_mat_tmp62 + mu*(weak_mat_tmp33 + weak_mat_tmp34 + weak_mat_tmp35 + s_t(3)*weak_mat_tmp62) + weak_mat_tmp36);
    const s_t loperand0 = qw * (material0 * adj_value0 + material1 * adj_value1 + material2 * adj_value2);
    const s_t loperand1 = qw * (material0 * adj_value3 + material1 * adj_value4 + material2 * adj_value5);
    const s_t loperand2 = qw * (material0 * adj_value6 + material1 * adj_value7 + material2 * adj_value8);
    const s_t loperand3 = qw * (material3 * adj_value0 + material4 * adj_value1 + material5 * adj_value2);
    const s_t loperand4 = qw * (material3 * adj_value3 + material4 * adj_value4 + material5 * adj_value5);
    const s_t loperand5 = qw * (material3 * adj_value6 + material4 * adj_value7 + material5 * adj_value8);
    const s_t loperand6 = qw * (material6 * adj_value0 + material7 * adj_value1 + material8 * adj_value2);
    const s_t loperand7 = qw * (material6 * adj_value3 + material7 * adj_value4 + material8 * adj_value5);
    const s_t loperand8 = qw * (material6 * adj_value6 + material7 * adj_value7 + material8 * adj_value8);
      loperand0_values[0] = loperand0;
      loperand1_values[0] = loperand1;
      loperand2_values[0] = loperand2;
      loperand3_values[0] = loperand3;
      loperand4_values[0] = loperand4;
      loperand5_values[0] = loperand5;
      loperand6_values[0] = loperand6;
      loperand7_values[0] = loperand7;
      loperand8_values[0] = loperand8;
      }
      for (int shape = 0; shape < NS; ++shape) {
        {
          out_streams[3 * shape][0] += loperand0_values[0] * grad_ref_x[q * NS + shape] + loperand1_values[0] * grad_ref_y[q * NS + shape] + loperand2_values[0] * grad_ref_z[q * NS + shape];
        }
        {
          out_streams[3 * shape + 1][0] += loperand3_values[0] * grad_ref_x[q * NS + shape] + loperand4_values[0] * grad_ref_y[q * NS + shape] + loperand5_values[0] * grad_ref_z[q * NS + shape];
        }
        {
          out_streams[3 * shape + 2][0] += loperand6_values[0] * grad_ref_x[q * NS + shape] + loperand7_values[0] * grad_ref_y[q * NS + shape] + loperand8_values[0] * grad_ref_z[q * NS + shape];
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void saint_venant_kirchhoff_d3_simplex_tet4_apply_block(
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
      {
      const ptrdiff_t goff = q * geometry_stride + 0;
      const s_t adj_value0 = adj0[goff];
      const s_t adj_value1 = adj1[goff];
      const s_t adj_value2 = adj2[goff];
      const s_t adj_value3 = adj3[goff];
      const s_t adj_value4 = adj4[goff];
      const s_t adj_value5 = adj5[goff];
      const s_t adj_value6 = adj6[goff];
      const s_t adj_value7 = adj7[goff];
      const s_t adj_value8 = adj8[goff];
      const s_t det_value0 = det0[goff];
      const s_t gu_ref0 = -(u_streams[0][0]) + u_streams[3][0];
      const s_t grad_h_ref0 = -(h_streams[0][0]) + h_streams[3][0];
      const s_t gu_ref1 = -(u_streams[0][0]) + u_streams[6][0];
      const s_t grad_h_ref1 = -(h_streams[0][0]) + h_streams[6][0];
      const s_t gu_ref2 = -(u_streams[0][0]) + u_streams[9][0];
      const s_t grad_h_ref2 = -(h_streams[0][0]) + h_streams[9][0];
      const s_t gu_ref3 = -(u_streams[1][0]) + u_streams[4][0];
      const s_t grad_h_ref3 = -(h_streams[1][0]) + h_streams[4][0];
      const s_t gu_ref4 = -(u_streams[1][0]) + u_streams[7][0];
      const s_t grad_h_ref4 = -(h_streams[1][0]) + h_streams[7][0];
      const s_t gu_ref5 = -(u_streams[1][0]) + u_streams[10][0];
      const s_t grad_h_ref5 = -(h_streams[1][0]) + h_streams[10][0];
      const s_t gu_ref6 = -(u_streams[2][0]) + u_streams[5][0];
      const s_t grad_h_ref6 = -(h_streams[2][0]) + h_streams[5][0];
      const s_t gu_ref7 = -(u_streams[2][0]) + u_streams[8][0];
      const s_t grad_h_ref7 = -(h_streams[2][0]) + h_streams[8][0];
      const s_t gu_ref8 = -(u_streams[2][0]) + u_streams[11][0];
      const s_t grad_h_ref8 = -(h_streams[2][0]) + h_streams[11][0];
      const s_t idet = s_t(1) / det_value0;
      const s_t gu0 = (gu_ref0 * adj_value0 + gu_ref1 * adj_value3 + gu_ref2 * adj_value6) * idet;
      const s_t trial_grad0 = (grad_h_ref0 * adj_value0 + grad_h_ref1 * adj_value3 + grad_h_ref2 * adj_value6) * idet;
      const s_t gu1 = (gu_ref0 * adj_value1 + gu_ref1 * adj_value4 + gu_ref2 * adj_value7) * idet;
      const s_t trial_grad1 = (grad_h_ref0 * adj_value1 + grad_h_ref1 * adj_value4 + grad_h_ref2 * adj_value7) * idet;
      const s_t gu2 = (gu_ref0 * adj_value2 + gu_ref1 * adj_value5 + gu_ref2 * adj_value8) * idet;
      const s_t trial_grad2 = (grad_h_ref0 * adj_value2 + grad_h_ref1 * adj_value5 + grad_h_ref2 * adj_value8) * idet;
      const s_t gu3 = (gu_ref3 * adj_value0 + gu_ref4 * adj_value3 + gu_ref5 * adj_value6) * idet;
      const s_t trial_grad3 = (grad_h_ref3 * adj_value0 + grad_h_ref4 * adj_value3 + grad_h_ref5 * adj_value6) * idet;
      const s_t gu4 = (gu_ref3 * adj_value1 + gu_ref4 * adj_value4 + gu_ref5 * adj_value7) * idet;
      const s_t trial_grad4 = (grad_h_ref3 * adj_value1 + grad_h_ref4 * adj_value4 + grad_h_ref5 * adj_value7) * idet;
      const s_t gu5 = (gu_ref3 * adj_value2 + gu_ref4 * adj_value5 + gu_ref5 * adj_value8) * idet;
      const s_t trial_grad5 = (grad_h_ref3 * adj_value2 + grad_h_ref4 * adj_value5 + grad_h_ref5 * adj_value8) * idet;
      const s_t gu6 = (gu_ref6 * adj_value0 + gu_ref7 * adj_value3 + gu_ref8 * adj_value6) * idet;
      const s_t trial_grad6 = (grad_h_ref6 * adj_value0 + grad_h_ref7 * adj_value3 + grad_h_ref8 * adj_value6) * idet;
      const s_t gu7 = (gu_ref6 * adj_value1 + gu_ref7 * adj_value4 + gu_ref8 * adj_value7) * idet;
      const s_t trial_grad7 = (grad_h_ref6 * adj_value1 + grad_h_ref7 * adj_value4 + grad_h_ref8 * adj_value7) * idet;
      const s_t gu8 = (gu_ref6 * adj_value2 + gu_ref7 * adj_value5 + gu_ref8 * adj_value8) * idet;
      const s_t trial_grad8 = (grad_h_ref6 * adj_value2 + grad_h_ref7 * adj_value5 + grad_h_ref8 * adj_value8) * idet;
    const s_t weak_mat_tmp0 = gu3*mu;
    const s_t weak_mat_tmp1 = gu0 + s_t(1);
    const s_t weak_mat_tmp2 = lmbda*weak_mat_tmp1;
    const s_t weak_mat_tmp3 = gu2*weak_mat_tmp0 + gu5*weak_mat_tmp2;
    const s_t weak_mat_tmp4 = gu6*mu;
    const s_t weak_mat_tmp5 = gu1*weak_mat_tmp4 + gu7*weak_mat_tmp2;
    const s_t weak_mat_tmp6 = gu4 + s_t(1);
    const s_t weak_mat_tmp7 = gu1*weak_mat_tmp0 + weak_mat_tmp2*weak_mat_tmp6;
    const s_t weak_mat_tmp8 = gu8 + s_t(1);
    const s_t weak_mat_tmp9 = gu2*weak_mat_tmp4 + weak_mat_tmp2*weak_mat_tmp8;
    const s_t weak_mat_tmp10 = gu1*weak_mat_tmp1;
    const s_t weak_mat_tmp11 = gu6*gu7;
    const s_t weak_mat_tmp12 = gu3*weak_mat_tmp6;
    const s_t weak_mat_tmp13 = lmbda*weak_mat_tmp10 + mu*(s_t(2)*weak_mat_tmp10 + weak_mat_tmp11 + weak_mat_tmp12);
    const s_t weak_mat_tmp14 = gu2*weak_mat_tmp1;
    const s_t weak_mat_tmp15 = gu3*gu5;
    const s_t weak_mat_tmp16 = gu6*weak_mat_tmp8;
    const s_t weak_mat_tmp17 = lmbda*weak_mat_tmp14 + mu*(s_t(2)*weak_mat_tmp14 + weak_mat_tmp15 + weak_mat_tmp16);
    const s_t weak_mat_tmp18 = gu3*weak_mat_tmp1;
    const s_t weak_mat_tmp19 = gu2*gu5;
    const s_t weak_mat_tmp20 = gu1*weak_mat_tmp6;
    const s_t weak_mat_tmp21 = lmbda*weak_mat_tmp18 + mu*(s_t(2)*weak_mat_tmp18 + weak_mat_tmp19 + weak_mat_tmp20);
    const s_t weak_mat_tmp22 = gu6*weak_mat_tmp1;
    const s_t weak_mat_tmp23 = gu1*gu7;
    const s_t weak_mat_tmp24 = gu2*weak_mat_tmp8;
    const s_t weak_mat_tmp25 = lmbda*weak_mat_tmp22 + mu*(s_t(2)*weak_mat_tmp22 + weak_mat_tmp23 + weak_mat_tmp24);
    const s_t weak_mat_tmp26 = pow_2(weak_mat_tmp1);
    const s_t weak_mat_tmp27 = pow_2(gu1);
    const s_t weak_mat_tmp28 = pow_2(gu3);
    const s_t weak_mat_tmp29 = weak_mat_tmp27 + weak_mat_tmp28;
    const s_t weak_mat_tmp30 = pow_2(gu6);
    const s_t weak_mat_tmp31 = pow_2(gu2);
    const s_t weak_mat_tmp32 = weak_mat_tmp31 + s_t(-1);
    const s_t weak_mat_tmp33 = weak_mat_tmp30 + weak_mat_tmp32;
    const s_t weak_mat_tmp34 = pow_2(gu5);
    const s_t weak_mat_tmp35 = pow_2(gu7);
    const s_t weak_mat_tmp36 = lmbda*(((s_t(1) / s_t(2)))*pow_2(gu0) + gu0 + ((s_t(1) / s_t(2)))*pow_2(gu4) + gu4 + ((s_t(1) / s_t(2)))*pow_2(gu8) + gu8 + ((s_t(1) / s_t(2)))*weak_mat_tmp27 + ((s_t(1) / s_t(2)))*weak_mat_tmp28 + ((s_t(1) / s_t(2)))*weak_mat_tmp30 + ((s_t(1) / s_t(2)))*weak_mat_tmp31 + ((s_t(1) / s_t(2)))*weak_mat_tmp34 + ((s_t(1) / s_t(2)))*weak_mat_tmp35);
    const s_t weak_mat_tmp37 = gu1*lmbda;
    const s_t weak_mat_tmp38 = mu*weak_mat_tmp6;
    const s_t weak_mat_tmp39 = gu2*weak_mat_tmp38 + gu5*weak_mat_tmp37;
    const s_t weak_mat_tmp40 = gu7*mu;
    const s_t weak_mat_tmp41 = gu6*weak_mat_tmp37 + weak_mat_tmp1*weak_mat_tmp40;
    const s_t weak_mat_tmp42 = gu2*weak_mat_tmp40 + weak_mat_tmp37*weak_mat_tmp8;
    const s_t weak_mat_tmp43 = gu3*weak_mat_tmp37 + weak_mat_tmp1*weak_mat_tmp38;
    const s_t weak_mat_tmp44 = gu1*gu2;
    const s_t weak_mat_tmp45 = gu5*weak_mat_tmp6;
    const s_t weak_mat_tmp46 = gu7*weak_mat_tmp8;
    const s_t weak_mat_tmp47 = lmbda*weak_mat_tmp44 + mu*(s_t(2)*weak_mat_tmp44 + weak_mat_tmp45 + weak_mat_tmp46);
    const s_t weak_mat_tmp48 = lmbda*weak_mat_tmp23 + mu*(weak_mat_tmp22 + s_t(2)*weak_mat_tmp23 + weak_mat_tmp24);
    const s_t weak_mat_tmp49 = lmbda*weak_mat_tmp20 + mu*(weak_mat_tmp18 + weak_mat_tmp19 + s_t(2)*weak_mat_tmp20);
    const s_t weak_mat_tmp50 = pow_2(weak_mat_tmp6);
    const s_t weak_mat_tmp51 = weak_mat_tmp26 + weak_mat_tmp50;
    const s_t weak_mat_tmp52 = gu2*lmbda;
    const s_t weak_mat_tmp53 = gu5*mu;
    const s_t weak_mat_tmp54 = gu3*weak_mat_tmp52 + weak_mat_tmp1*weak_mat_tmp53;
    const s_t weak_mat_tmp55 = gu1*weak_mat_tmp53 + weak_mat_tmp52*weak_mat_tmp6;
    const s_t weak_mat_tmp56 = mu*weak_mat_tmp8;
    const s_t weak_mat_tmp57 = gu1*weak_mat_tmp56 + gu7*weak_mat_tmp52;
    const s_t weak_mat_tmp58 = gu6*weak_mat_tmp52 + weak_mat_tmp1*weak_mat_tmp56;
    const s_t weak_mat_tmp59 = lmbda*weak_mat_tmp19 + mu*(weak_mat_tmp18 + s_t(2)*weak_mat_tmp19 + weak_mat_tmp20);
    const s_t weak_mat_tmp60 = lmbda*weak_mat_tmp24 + mu*(weak_mat_tmp22 + weak_mat_tmp23 + s_t(2)*weak_mat_tmp24);
    const s_t weak_mat_tmp61 = weak_mat_tmp34 + s_t(-1);
    const s_t weak_mat_tmp62 = pow_2(weak_mat_tmp8);
    const s_t weak_mat_tmp63 = weak_mat_tmp26 + weak_mat_tmp62;
    const s_t weak_mat_tmp64 = gu3*lmbda;
    const s_t weak_mat_tmp65 = gu7*weak_mat_tmp64 + weak_mat_tmp4*weak_mat_tmp6;
    const s_t weak_mat_tmp66 = gu5*weak_mat_tmp4 + weak_mat_tmp64*weak_mat_tmp8;
    const s_t weak_mat_tmp67 = lmbda*weak_mat_tmp15 + mu*(weak_mat_tmp14 + s_t(2)*weak_mat_tmp15 + weak_mat_tmp16);
    const s_t weak_mat_tmp68 = gu3*gu6;
    const s_t weak_mat_tmp69 = gu5*weak_mat_tmp8;
    const s_t weak_mat_tmp70 = gu7*weak_mat_tmp6;
    const s_t weak_mat_tmp71 = lmbda*weak_mat_tmp68 + mu*(s_t(2)*weak_mat_tmp68 + weak_mat_tmp69 + weak_mat_tmp70);
    const s_t weak_mat_tmp72 = lmbda*weak_mat_tmp12 + mu*(weak_mat_tmp10 + weak_mat_tmp11 + s_t(2)*weak_mat_tmp12);
    const s_t weak_mat_tmp73 = lmbda*weak_mat_tmp6;
    const s_t weak_mat_tmp74 = gu6*weak_mat_tmp73 + gu7*weak_mat_tmp0;
    const s_t weak_mat_tmp75 = gu5*weak_mat_tmp40 + weak_mat_tmp73*weak_mat_tmp8;
    const s_t weak_mat_tmp76 = lmbda*weak_mat_tmp45 + mu*(weak_mat_tmp44 + s_t(2)*weak_mat_tmp45 + weak_mat_tmp46);
    const s_t weak_mat_tmp77 = lmbda*weak_mat_tmp70 + mu*(weak_mat_tmp68 + weak_mat_tmp69 + s_t(2)*weak_mat_tmp70);
    const s_t weak_mat_tmp78 = gu5*lmbda;
    const s_t weak_mat_tmp79 = gu6*weak_mat_tmp78 + weak_mat_tmp0*weak_mat_tmp8;
    const s_t weak_mat_tmp80 = gu7*weak_mat_tmp78 + weak_mat_tmp38*weak_mat_tmp8;
    const s_t weak_mat_tmp81 = lmbda*weak_mat_tmp69 + mu*(weak_mat_tmp68 + s_t(2)*weak_mat_tmp69 + weak_mat_tmp70);
    const s_t weak_mat_tmp82 = weak_mat_tmp50 + weak_mat_tmp62;
    const s_t weak_mat_tmp83 = lmbda*weak_mat_tmp11 + mu*(weak_mat_tmp10 + s_t(2)*weak_mat_tmp11 + weak_mat_tmp12);
    const s_t weak_mat_tmp84 = lmbda*weak_mat_tmp16 + mu*(weak_mat_tmp14 + weak_mat_tmp15 + s_t(2)*weak_mat_tmp16);
    const s_t weak_mat_tmp85 = lmbda*weak_mat_tmp46 + mu*(weak_mat_tmp44 + weak_mat_tmp45 + s_t(2)*weak_mat_tmp46);
    const s_t material0 = trial_grad0*(lmbda*weak_mat_tmp26 + mu*(s_t(3)*weak_mat_tmp26 + weak_mat_tmp29 + weak_mat_tmp33) + weak_mat_tmp36) + trial_grad1*weak_mat_tmp13 + trial_grad2*weak_mat_tmp17 + trial_grad3*weak_mat_tmp21 + trial_grad4*weak_mat_tmp7 + trial_grad5*weak_mat_tmp3 + trial_grad6*weak_mat_tmp25 + trial_grad7*weak_mat_tmp5 + trial_grad8*weak_mat_tmp9;
    const s_t material1 = trial_grad0*weak_mat_tmp13 + trial_grad1*(lmbda*weak_mat_tmp27 + mu*(s_t(3)*weak_mat_tmp27 + weak_mat_tmp32 + weak_mat_tmp35 + weak_mat_tmp51) + weak_mat_tmp36) + trial_grad2*weak_mat_tmp47 + trial_grad3*weak_mat_tmp43 + trial_grad4*weak_mat_tmp49 + trial_grad5*weak_mat_tmp39 + trial_grad6*weak_mat_tmp41 + trial_grad7*weak_mat_tmp48 + trial_grad8*weak_mat_tmp42;
    const s_t material2 = trial_grad0*weak_mat_tmp17 + trial_grad1*weak_mat_tmp47 + trial_grad2*(lmbda*weak_mat_tmp31 + mu*(weak_mat_tmp27 + s_t(3)*weak_mat_tmp31 + weak_mat_tmp61 + weak_mat_tmp63) + weak_mat_tmp36) + trial_grad3*weak_mat_tmp54 + trial_grad4*weak_mat_tmp55 + trial_grad5*weak_mat_tmp59 + trial_grad6*weak_mat_tmp58 + trial_grad7*weak_mat_tmp57 + trial_grad8*weak_mat_tmp60;
    const s_t material3 = trial_grad0*weak_mat_tmp21 + trial_grad1*weak_mat_tmp43 + trial_grad2*weak_mat_tmp54 + trial_grad3*(lmbda*weak_mat_tmp28 + mu*(s_t(3)*weak_mat_tmp28 + weak_mat_tmp30 + weak_mat_tmp51 + weak_mat_tmp61) + weak_mat_tmp36) + trial_grad4*weak_mat_tmp72 + trial_grad5*weak_mat_tmp67 + trial_grad6*weak_mat_tmp71 + trial_grad7*weak_mat_tmp65 + trial_grad8*weak_mat_tmp66;
    const s_t material4 = trial_grad0*weak_mat_tmp7 + trial_grad1*weak_mat_tmp49 + trial_grad2*weak_mat_tmp55 + trial_grad3*weak_mat_tmp72 + trial_grad4*(lmbda*weak_mat_tmp50 + mu*(weak_mat_tmp29 + weak_mat_tmp35 + s_t(3)*weak_mat_tmp50 + weak_mat_tmp61) + weak_mat_tmp36) + trial_grad5*weak_mat_tmp76 + trial_grad6*weak_mat_tmp74 + trial_grad7*weak_mat_tmp77 + trial_grad8*weak_mat_tmp75;
    const s_t material5 = trial_grad0*weak_mat_tmp3 + trial_grad1*weak_mat_tmp39 + trial_grad2*weak_mat_tmp59 + trial_grad3*weak_mat_tmp67 + trial_grad4*weak_mat_tmp76 + trial_grad5*(lmbda*weak_mat_tmp34 + mu*(weak_mat_tmp28 + weak_mat_tmp32 + s_t(3)*weak_mat_tmp34 + weak_mat_tmp82) + weak_mat_tmp36) + trial_grad6*weak_mat_tmp79 + trial_grad7*weak_mat_tmp80 + trial_grad8*weak_mat_tmp81;
    const s_t material6 = trial_grad0*weak_mat_tmp25 + trial_grad1*weak_mat_tmp41 + trial_grad2*weak_mat_tmp58 + trial_grad3*weak_mat_tmp71 + trial_grad4*weak_mat_tmp74 + trial_grad5*weak_mat_tmp79 + trial_grad6*(lmbda*weak_mat_tmp30 + mu*(weak_mat_tmp28 + s_t(3)*weak_mat_tmp30 + weak_mat_tmp35 + weak_mat_tmp63 + s_t(-1)) + weak_mat_tmp36) + trial_grad7*weak_mat_tmp83 + trial_grad8*weak_mat_tmp84;
    const s_t material7 = trial_grad0*weak_mat_tmp5 + trial_grad1*weak_mat_tmp48 + trial_grad2*weak_mat_tmp57 + trial_grad3*weak_mat_tmp65 + trial_grad4*weak_mat_tmp77 + trial_grad5*weak_mat_tmp80 + trial_grad6*weak_mat_tmp83 + trial_grad7*(lmbda*weak_mat_tmp35 + mu*(weak_mat_tmp27 + weak_mat_tmp30 + s_t(3)*weak_mat_tmp35 + weak_mat_tmp82 + s_t(-1)) + weak_mat_tmp36) + trial_grad8*weak_mat_tmp85;
    const s_t material8 = trial_grad0*weak_mat_tmp9 + trial_grad1*weak_mat_tmp42 + trial_grad2*weak_mat_tmp60 + trial_grad3*weak_mat_tmp66 + trial_grad4*weak_mat_tmp75 + trial_grad5*weak_mat_tmp81 + trial_grad6*weak_mat_tmp84 + trial_grad7*weak_mat_tmp85 + trial_grad8*(lmbda*weak_mat_tmp62 + mu*(weak_mat_tmp33 + weak_mat_tmp34 + weak_mat_tmp35 + s_t(3)*weak_mat_tmp62) + weak_mat_tmp36);
    const s_t loperand0 = qw * (material0 * adj_value0 + material1 * adj_value1 + material2 * adj_value2);
    const s_t loperand1 = qw * (material0 * adj_value3 + material1 * adj_value4 + material2 * adj_value5);
    const s_t loperand2 = qw * (material0 * adj_value6 + material1 * adj_value7 + material2 * adj_value8);
    const s_t loperand3 = qw * (material3 * adj_value0 + material4 * adj_value1 + material5 * adj_value2);
    const s_t loperand4 = qw * (material3 * adj_value3 + material4 * adj_value4 + material5 * adj_value5);
    const s_t loperand5 = qw * (material3 * adj_value6 + material4 * adj_value7 + material5 * adj_value8);
    const s_t loperand6 = qw * (material6 * adj_value0 + material7 * adj_value1 + material8 * adj_value2);
    const s_t loperand7 = qw * (material6 * adj_value3 + material7 * adj_value4 + material8 * adj_value5);
    const s_t loperand8 = qw * (material6 * adj_value6 + material7 * adj_value7 + material8 * adj_value8);
      out_streams[0][0] += -(loperand0) - loperand1 - loperand2;
      out_streams[1][0] += -(loperand3) - loperand4 - loperand5;
      out_streams[2][0] += -(loperand6) - loperand7 - loperand8;
      out_streams[3][0] += loperand0;
      out_streams[4][0] += loperand3;
      out_streams[5][0] += loperand6;
      out_streams[6][0] += loperand1;
      out_streams[7][0] += loperand4;
      out_streams[8][0] += loperand7;
      out_streams[9][0] += loperand2;
      out_streams[10][0] += loperand5;
      out_streams[11][0] += loperand8;
      }
    }
}

} // namespace codegen
} // namespace sfem

#endif
