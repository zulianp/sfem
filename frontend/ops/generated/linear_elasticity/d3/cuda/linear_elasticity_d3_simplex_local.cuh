#ifndef LINEAR_ELASTICITY_D3_SIMPLEX_LOCAL_CUH
#define LINEAR_ELASTICITY_D3_SIMPLEX_LOCAL_CUH
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
static __host__ __device__ __forceinline__ void linear_elasticity_d3_simplex_objective_block(
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
    value[step * value_stride + 0] += qw * det_value0 * (((s_t(1) / s_t(2)))*lmbda*pow_2(gu0 + gu4 + gu8) + mu*(pow_2(gu0) + pow_2(gu4) + pow_2(gu8) + s_t(2)*pow_2(((s_t(1) / s_t(2)))*gu1 + ((s_t(1) / s_t(2)))*gu3) + s_t(2)*pow_2(((s_t(1) / s_t(2)))*gu2 + ((s_t(1) / s_t(2)))*gu6) + s_t(2)*pow_2(((s_t(1) / s_t(2)))*gu5 + ((s_t(1) / s_t(2)))*gu7)));
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void linear_elasticity_d3_simplex_tet4_objective_block(
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
    value[step * value_stride + 0] += qw * det_value0 * (((s_t(1) / s_t(2)))*lmbda*pow_2(gu0 + gu4 + gu8) + mu*(pow_2(gu0) + pow_2(gu4) + pow_2(gu8) + s_t(2)*pow_2(((s_t(1) / s_t(2)))*gu1 + ((s_t(1) / s_t(2)))*gu3) + s_t(2)*pow_2(((s_t(1) / s_t(2)))*gu2 + ((s_t(1) / s_t(2)))*gu6) + s_t(2)*pow_2(((s_t(1) / s_t(2)))*gu5 + ((s_t(1) / s_t(2)))*gu7)));
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void linear_elasticity_d3_simplex_gradient_block(
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
    const s_t weak_mat_tmp0 = s_t(2)*gu0;
    const s_t weak_mat_tmp1 = s_t(2)*gu4;
    const s_t weak_mat_tmp2 = s_t(2)*gu8;
    const s_t weak_mat_tmp3 = ((s_t(1) / s_t(2)))*lmbda*(weak_mat_tmp0 + weak_mat_tmp1 + weak_mat_tmp2);
    const s_t weak_mat_tmp4 = mu*(gu1 + gu3);
    const s_t weak_mat_tmp5 = mu*(gu2 + gu6);
    const s_t weak_mat_tmp6 = mu*(gu5 + gu7);
    const s_t material0 = mu*weak_mat_tmp0 + weak_mat_tmp3;
    const s_t material1 = weak_mat_tmp4;
    const s_t material2 = weak_mat_tmp5;
    const s_t material3 = weak_mat_tmp4;
    const s_t material4 = mu*weak_mat_tmp1 + weak_mat_tmp3;
    const s_t material5 = weak_mat_tmp6;
    const s_t material6 = weak_mat_tmp5;
    const s_t material7 = weak_mat_tmp6;
    const s_t material8 = mu*weak_mat_tmp2 + weak_mat_tmp3;
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
static __host__ __device__ __forceinline__ void linear_elasticity_d3_simplex_tet4_gradient_block(
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
    const s_t weak_mat_tmp0 = s_t(2)*gu0;
    const s_t weak_mat_tmp1 = s_t(2)*gu4;
    const s_t weak_mat_tmp2 = s_t(2)*gu8;
    const s_t weak_mat_tmp3 = ((s_t(1) / s_t(2)))*lmbda*(weak_mat_tmp0 + weak_mat_tmp1 + weak_mat_tmp2);
    const s_t weak_mat_tmp4 = mu*(gu1 + gu3);
    const s_t weak_mat_tmp5 = mu*(gu2 + gu6);
    const s_t weak_mat_tmp6 = mu*(gu5 + gu7);
    const s_t material0 = mu*weak_mat_tmp0 + weak_mat_tmp3;
    const s_t material1 = weak_mat_tmp4;
    const s_t material2 = weak_mat_tmp5;
    const s_t material3 = weak_mat_tmp4;
    const s_t material4 = mu*weak_mat_tmp1 + weak_mat_tmp3;
    const s_t material5 = weak_mat_tmp6;
    const s_t material6 = weak_mat_tmp5;
    const s_t material7 = weak_mat_tmp6;
    const s_t material8 = mu*weak_mat_tmp2 + weak_mat_tmp3;
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
static __host__ __device__ __forceinline__ void linear_elasticity_d3_simplex_apply_block(
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
        const s_t *const RSTR h_streams[NS * 3],
        s_t *const RSTR out_streams[NS * 3]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
    for (int q = 0; q < NQ; ++q) {
      const s_t qw = q_weight[q];
      s_t grad_h_ref0_values[VS];
      s_t grad_h_ref1_values[VS];
      s_t grad_h_ref2_values[VS];
      s_t grad_h_ref3_values[VS];
      s_t grad_h_ref4_values[VS];
      s_t grad_h_ref5_values[VS];
      s_t grad_h_ref6_values[VS];
      s_t grad_h_ref7_values[VS];
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
        grad_h_ref0_values[0] = s_t(0);
        grad_h_ref1_values[0] = s_t(0);
        grad_h_ref2_values[0] = s_t(0);
        grad_h_ref3_values[0] = s_t(0);
        grad_h_ref4_values[0] = s_t(0);
        grad_h_ref5_values[0] = s_t(0);
        grad_h_ref6_values[0] = s_t(0);
        grad_h_ref7_values[0] = s_t(0);
        grad_h_ref8_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        {
          grad_h_ref0_values[0] += h_streams[3 * shape][0] * grad_ref_x[q * NS + shape];
        }
        {
          grad_h_ref1_values[0] += h_streams[3 * shape][0] * grad_ref_y[q * NS + shape];
        }
        {
          grad_h_ref2_values[0] += h_streams[3 * shape][0] * grad_ref_z[q * NS + shape];
        }
        {
          grad_h_ref3_values[0] += h_streams[3 * shape + 1][0] * grad_ref_x[q * NS + shape];
        }
        {
          grad_h_ref4_values[0] += h_streams[3 * shape + 1][0] * grad_ref_y[q * NS + shape];
        }
        {
          grad_h_ref5_values[0] += h_streams[3 * shape + 1][0] * grad_ref_z[q * NS + shape];
        }
        {
          grad_h_ref6_values[0] += h_streams[3 * shape + 2][0] * grad_ref_x[q * NS + shape];
        }
        {
          grad_h_ref7_values[0] += h_streams[3 * shape + 2][0] * grad_ref_y[q * NS + shape];
        }
        {
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
      const s_t grad_h_ref0 = grad_h_ref0_values[0];
      const s_t grad_h_ref1 = grad_h_ref1_values[0];
      const s_t grad_h_ref2 = grad_h_ref2_values[0];
      const s_t grad_h_ref3 = grad_h_ref3_values[0];
      const s_t grad_h_ref4 = grad_h_ref4_values[0];
      const s_t grad_h_ref5 = grad_h_ref5_values[0];
      const s_t grad_h_ref6 = grad_h_ref6_values[0];
      const s_t grad_h_ref7 = grad_h_ref7_values[0];
      const s_t grad_h_ref8 = grad_h_ref8_values[0];
    const s_t idet = s_t(1) / det_value0;
    const s_t trial_grad0 = (grad_h_ref0 * adj_value0 + grad_h_ref1 * adj_value3 + grad_h_ref2 * adj_value6) * idet;
    const s_t trial_grad1 = (grad_h_ref0 * adj_value1 + grad_h_ref1 * adj_value4 + grad_h_ref2 * adj_value7) * idet;
    const s_t trial_grad2 = (grad_h_ref0 * adj_value2 + grad_h_ref1 * adj_value5 + grad_h_ref2 * adj_value8) * idet;
    const s_t trial_grad3 = (grad_h_ref3 * adj_value0 + grad_h_ref4 * adj_value3 + grad_h_ref5 * adj_value6) * idet;
    const s_t trial_grad4 = (grad_h_ref3 * adj_value1 + grad_h_ref4 * adj_value4 + grad_h_ref5 * adj_value7) * idet;
    const s_t trial_grad5 = (grad_h_ref3 * adj_value2 + grad_h_ref4 * adj_value5 + grad_h_ref5 * adj_value8) * idet;
    const s_t trial_grad6 = (grad_h_ref6 * adj_value0 + grad_h_ref7 * adj_value3 + grad_h_ref8 * adj_value6) * idet;
    const s_t trial_grad7 = (grad_h_ref6 * adj_value1 + grad_h_ref7 * adj_value4 + grad_h_ref8 * adj_value7) * idet;
    const s_t trial_grad8 = (grad_h_ref6 * adj_value2 + grad_h_ref7 * adj_value5 + grad_h_ref8 * adj_value8) * idet;
    const s_t weak_mat_tmp0 = s_t(2)*trial_grad0;
    const s_t weak_mat_tmp1 = s_t(2)*trial_grad4;
    const s_t weak_mat_tmp2 = s_t(2)*trial_grad8;
    const s_t weak_mat_tmp3 = ((s_t(1) / s_t(2)))*lmbda*(weak_mat_tmp0 + weak_mat_tmp1 + weak_mat_tmp2);
    const s_t weak_mat_tmp4 = mu*(trial_grad1 + trial_grad3);
    const s_t weak_mat_tmp5 = mu*(trial_grad2 + trial_grad6);
    const s_t weak_mat_tmp6 = mu*(trial_grad5 + trial_grad7);
    const s_t material0 = mu*weak_mat_tmp0 + weak_mat_tmp3;
    const s_t material1 = weak_mat_tmp4;
    const s_t material2 = weak_mat_tmp5;
    const s_t material3 = weak_mat_tmp4;
    const s_t material4 = mu*weak_mat_tmp1 + weak_mat_tmp3;
    const s_t material5 = weak_mat_tmp6;
    const s_t material6 = weak_mat_tmp5;
    const s_t material7 = weak_mat_tmp6;
    const s_t material8 = mu*weak_mat_tmp2 + weak_mat_tmp3;
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
static __host__ __device__ __forceinline__ void linear_elasticity_d3_simplex_tet4_apply_block(
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
      const s_t grad_h_ref0 = -(h_streams[0][0]) + h_streams[3][0];
      const s_t grad_h_ref1 = -(h_streams[0][0]) + h_streams[6][0];
      const s_t grad_h_ref2 = -(h_streams[0][0]) + h_streams[9][0];
      const s_t grad_h_ref3 = -(h_streams[1][0]) + h_streams[4][0];
      const s_t grad_h_ref4 = -(h_streams[1][0]) + h_streams[7][0];
      const s_t grad_h_ref5 = -(h_streams[1][0]) + h_streams[10][0];
      const s_t grad_h_ref6 = -(h_streams[2][0]) + h_streams[5][0];
      const s_t grad_h_ref7 = -(h_streams[2][0]) + h_streams[8][0];
      const s_t grad_h_ref8 = -(h_streams[2][0]) + h_streams[11][0];
      const s_t idet = s_t(1) / det_value0;
      const s_t trial_grad0 = (grad_h_ref0 * adj_value0 + grad_h_ref1 * adj_value3 + grad_h_ref2 * adj_value6) * idet;
      const s_t trial_grad1 = (grad_h_ref0 * adj_value1 + grad_h_ref1 * adj_value4 + grad_h_ref2 * adj_value7) * idet;
      const s_t trial_grad2 = (grad_h_ref0 * adj_value2 + grad_h_ref1 * adj_value5 + grad_h_ref2 * adj_value8) * idet;
      const s_t trial_grad3 = (grad_h_ref3 * adj_value0 + grad_h_ref4 * adj_value3 + grad_h_ref5 * adj_value6) * idet;
      const s_t trial_grad4 = (grad_h_ref3 * adj_value1 + grad_h_ref4 * adj_value4 + grad_h_ref5 * adj_value7) * idet;
      const s_t trial_grad5 = (grad_h_ref3 * adj_value2 + grad_h_ref4 * adj_value5 + grad_h_ref5 * adj_value8) * idet;
      const s_t trial_grad6 = (grad_h_ref6 * adj_value0 + grad_h_ref7 * adj_value3 + grad_h_ref8 * adj_value6) * idet;
      const s_t trial_grad7 = (grad_h_ref6 * adj_value1 + grad_h_ref7 * adj_value4 + grad_h_ref8 * adj_value7) * idet;
      const s_t trial_grad8 = (grad_h_ref6 * adj_value2 + grad_h_ref7 * adj_value5 + grad_h_ref8 * adj_value8) * idet;
    const s_t weak_mat_tmp0 = s_t(2)*trial_grad0;
    const s_t weak_mat_tmp1 = s_t(2)*trial_grad4;
    const s_t weak_mat_tmp2 = s_t(2)*trial_grad8;
    const s_t weak_mat_tmp3 = ((s_t(1) / s_t(2)))*lmbda*(weak_mat_tmp0 + weak_mat_tmp1 + weak_mat_tmp2);
    const s_t weak_mat_tmp4 = mu*(trial_grad1 + trial_grad3);
    const s_t weak_mat_tmp5 = mu*(trial_grad2 + trial_grad6);
    const s_t weak_mat_tmp6 = mu*(trial_grad5 + trial_grad7);
    const s_t material0 = mu*weak_mat_tmp0 + weak_mat_tmp3;
    const s_t material1 = weak_mat_tmp4;
    const s_t material2 = weak_mat_tmp5;
    const s_t material3 = weak_mat_tmp4;
    const s_t material4 = mu*weak_mat_tmp1 + weak_mat_tmp3;
    const s_t material5 = weak_mat_tmp6;
    const s_t material6 = weak_mat_tmp5;
    const s_t material7 = weak_mat_tmp6;
    const s_t material8 = mu*weak_mat_tmp2 + weak_mat_tmp3;
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
