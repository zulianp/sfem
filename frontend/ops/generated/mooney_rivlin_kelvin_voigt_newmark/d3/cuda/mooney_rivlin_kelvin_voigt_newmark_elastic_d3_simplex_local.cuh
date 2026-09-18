#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_ELASTIC_D3_SIMPLEX_LOCAL_CUH
#define MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_ELASTIC_D3_SIMPLEX_LOCAL_CUH
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
static __host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_objective_block(
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
    const s_t weak_obj_tmp0 = gu8 + s_t(1);
    const s_t weak_obj_tmp1 = gu1*gu3*weak_obj_tmp0;
    const s_t weak_obj_tmp2 = gu4 + s_t(1);
    const s_t weak_obj_tmp3 = gu2*gu6*weak_obj_tmp2;
    const s_t weak_obj_tmp4 = gu0 + s_t(1);
    const s_t weak_obj_tmp5 = gu5*gu7*weak_obj_tmp4;
    const s_t weak_obj_tmp6 = pow_2(gu1) + pow_2(gu7) + pow_2(weak_obj_tmp2);
    const s_t weak_obj_tmp7 = pow_2(gu2) + pow_2(gu5) + pow_2(weak_obj_tmp0);
    const s_t weak_obj_tmp8 = pow_2(gu3) + pow_2(gu6) + pow_2(weak_obj_tmp4);
    const s_t weak_obj_tmp9 = weak_obj_tmp6 + weak_obj_tmp7 + weak_obj_tmp8;
    value[step * value_stride + 0] += qw * det_value0 * (((s_t(1) / s_t(2)))*lmbda*pow_2(gu1*gu5*gu6 + gu2*gu3*gu7 + weak_obj_tmp0*weak_obj_tmp2*weak_obj_tmp4 - weak_obj_tmp1 - weak_obj_tmp3 - weak_obj_tmp5 + s_t(-1)) + mu*(-s_t(6)*gu1*gu5*gu6 - s_t(6)*gu2*gu3*gu7 - s_t(6)*weak_obj_tmp0*weak_obj_tmp2*weak_obj_tmp4 + s_t(6)*weak_obj_tmp1 + s_t(6)*weak_obj_tmp3 + s_t(6)*weak_obj_tmp5 - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp6) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp7) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp8) + ((s_t(1) / s_t(2)))*pow_2(weak_obj_tmp9) + weak_obj_tmp9 - pow_2(gu1*gu2 + gu5*weak_obj_tmp2 + gu7*weak_obj_tmp0) - pow_2(gu1*weak_obj_tmp4 + gu3*weak_obj_tmp2 + gu6*gu7) - pow_2(gu2*weak_obj_tmp4 + gu3*gu5 + gu6*weak_obj_tmp0)));
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_tet4_objective_block(
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
    const s_t weak_obj_tmp0 = gu8 + s_t(1);
    const s_t weak_obj_tmp1 = gu1*gu3*weak_obj_tmp0;
    const s_t weak_obj_tmp2 = gu4 + s_t(1);
    const s_t weak_obj_tmp3 = gu2*gu6*weak_obj_tmp2;
    const s_t weak_obj_tmp4 = gu0 + s_t(1);
    const s_t weak_obj_tmp5 = gu5*gu7*weak_obj_tmp4;
    const s_t weak_obj_tmp6 = pow_2(gu1) + pow_2(gu7) + pow_2(weak_obj_tmp2);
    const s_t weak_obj_tmp7 = pow_2(gu2) + pow_2(gu5) + pow_2(weak_obj_tmp0);
    const s_t weak_obj_tmp8 = pow_2(gu3) + pow_2(gu6) + pow_2(weak_obj_tmp4);
    const s_t weak_obj_tmp9 = weak_obj_tmp6 + weak_obj_tmp7 + weak_obj_tmp8;
    value[step * value_stride + 0] += qw * det_value0 * (((s_t(1) / s_t(2)))*lmbda*pow_2(gu1*gu5*gu6 + gu2*gu3*gu7 + weak_obj_tmp0*weak_obj_tmp2*weak_obj_tmp4 - weak_obj_tmp1 - weak_obj_tmp3 - weak_obj_tmp5 + s_t(-1)) + mu*(-s_t(6)*gu1*gu5*gu6 - s_t(6)*gu2*gu3*gu7 - s_t(6)*weak_obj_tmp0*weak_obj_tmp2*weak_obj_tmp4 + s_t(6)*weak_obj_tmp1 + s_t(6)*weak_obj_tmp3 + s_t(6)*weak_obj_tmp5 - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp6) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp7) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp8) + ((s_t(1) / s_t(2)))*pow_2(weak_obj_tmp9) + weak_obj_tmp9 - pow_2(gu1*gu2 + gu5*weak_obj_tmp2 + gu7*weak_obj_tmp0) - pow_2(gu1*weak_obj_tmp4 + gu3*weak_obj_tmp2 + gu6*gu7) - pow_2(gu2*weak_obj_tmp4 + gu3*gu5 + gu6*weak_obj_tmp0)));
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_gradient_block(
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
    const s_t weak_mat_tmp0 = s_t(2)*gu5;
    const s_t weak_mat_tmp1 = gu4 + s_t(1);
    const s_t weak_mat_tmp2 = gu8 + s_t(1);
    const s_t weak_mat_tmp3 = gu0 + s_t(1);
    const s_t weak_mat_tmp4 = gu5*gu7;
    const s_t weak_mat_tmp5 = ((s_t(1) / s_t(2)))*lmbda*(-gu1*gu3*weak_mat_tmp2 + gu1*gu5*gu6 + gu2*gu3*gu7 - gu2*gu6*weak_mat_tmp1 + weak_mat_tmp1*weak_mat_tmp2*weak_mat_tmp3 - weak_mat_tmp3*weak_mat_tmp4 + s_t(-1));
    const s_t weak_mat_tmp6 = gu1*weak_mat_tmp3 + gu3*weak_mat_tmp1 + gu6*gu7;
    const s_t weak_mat_tmp7 = s_t(2)*gu1;
    const s_t weak_mat_tmp8 = gu2*weak_mat_tmp3 + gu3*gu5 + gu6*weak_mat_tmp2;
    const s_t weak_mat_tmp9 = s_t(2)*gu2;
    const s_t weak_mat_tmp10 = pow_2(gu3) + pow_2(gu6) + pow_2(weak_mat_tmp3);
    const s_t weak_mat_tmp11 = s_t(2)*weak_mat_tmp3;
    const s_t weak_mat_tmp12 = pow_2(gu1) + pow_2(gu7) + pow_2(weak_mat_tmp1);
    const s_t weak_mat_tmp13 = pow_2(gu2) + pow_2(gu5) + pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp14 = weak_mat_tmp10 + weak_mat_tmp12 + weak_mat_tmp13;
    const s_t weak_mat_tmp15 = s_t(2)*gu3;
    const s_t weak_mat_tmp16 = gu1*gu2 + gu5*weak_mat_tmp1 + gu7*weak_mat_tmp2;
    const s_t weak_mat_tmp17 = s_t(2)*gu6;
    const s_t weak_mat_tmp18 = s_t(6)*gu2;
    const s_t weak_mat_tmp19 = s_t(2)*weak_mat_tmp1;
    const s_t weak_mat_tmp20 = s_t(2)*gu7;
    const s_t weak_mat_tmp21 = s_t(6)*gu1;
    const s_t weak_mat_tmp22 = s_t(2)*weak_mat_tmp2;
    const s_t material0 = mu*(s_t(2)*gu0 - s_t(6)*weak_mat_tmp1*weak_mat_tmp2 - weak_mat_tmp10*weak_mat_tmp11 + weak_mat_tmp11*weak_mat_tmp14 + s_t(6)*weak_mat_tmp4 - weak_mat_tmp6*weak_mat_tmp7 - weak_mat_tmp8*weak_mat_tmp9 + s_t(2)) + weak_mat_tmp5*(-gu7*weak_mat_tmp0 + s_t(2)*weak_mat_tmp1*weak_mat_tmp2);
    const s_t material1 = mu*(s_t(2)*gu1*weak_mat_tmp14 + s_t(2)*gu1 + s_t(6)*gu3*weak_mat_tmp2 - s_t(6)*gu5*gu6 - weak_mat_tmp11*weak_mat_tmp6 - weak_mat_tmp12*weak_mat_tmp7 - weak_mat_tmp16*weak_mat_tmp9) + weak_mat_tmp5*(s_t(2)*gu5*gu6 - weak_mat_tmp15*weak_mat_tmp2);
    const s_t material2 = mu*(s_t(2)*gu2*weak_mat_tmp14 + s_t(2)*gu2 - s_t(6)*gu3*gu7 + s_t(6)*gu6*weak_mat_tmp1 - weak_mat_tmp11*weak_mat_tmp8 - weak_mat_tmp13*weak_mat_tmp9 - weak_mat_tmp16*weak_mat_tmp7) + weak_mat_tmp5*(gu7*weak_mat_tmp15 - weak_mat_tmp1*weak_mat_tmp17);
    const s_t material3 = mu*(s_t(6)*gu1*weak_mat_tmp2 + s_t(2)*gu3*weak_mat_tmp14 + s_t(2)*gu3 - gu7*weak_mat_tmp18 - weak_mat_tmp0*weak_mat_tmp8 - weak_mat_tmp10*weak_mat_tmp15 - weak_mat_tmp19*weak_mat_tmp6) + weak_mat_tmp5*(s_t(2)*gu2*gu7 - weak_mat_tmp2*weak_mat_tmp7);
    const s_t material4 = mu*(s_t(2)*gu4 + gu6*weak_mat_tmp18 - weak_mat_tmp0*weak_mat_tmp16 - weak_mat_tmp12*weak_mat_tmp19 + weak_mat_tmp14*weak_mat_tmp19 - weak_mat_tmp15*weak_mat_tmp6 - s_t(6)*weak_mat_tmp2*weak_mat_tmp3 + s_t(2)) + weak_mat_tmp5*(-gu6*weak_mat_tmp9 + s_t(2)*weak_mat_tmp2*weak_mat_tmp3);
    const s_t material5 = mu*(s_t(2)*gu5*weak_mat_tmp14 + s_t(2)*gu5 - gu6*weak_mat_tmp21 + s_t(6)*gu7*weak_mat_tmp3 - weak_mat_tmp0*weak_mat_tmp13 - weak_mat_tmp15*weak_mat_tmp8 - weak_mat_tmp16*weak_mat_tmp19) + weak_mat_tmp5*(gu6*weak_mat_tmp7 - weak_mat_tmp20*weak_mat_tmp3);
    const s_t material6 = mu*(s_t(6)*gu2*weak_mat_tmp1 - gu5*weak_mat_tmp21 + s_t(2)*gu6*weak_mat_tmp14 + s_t(2)*gu6 - weak_mat_tmp10*weak_mat_tmp17 - weak_mat_tmp20*weak_mat_tmp6 - weak_mat_tmp22*weak_mat_tmp8) + weak_mat_tmp5*(gu5*weak_mat_tmp7 - weak_mat_tmp1*weak_mat_tmp9);
    const s_t material7 = mu*(-gu3*weak_mat_tmp18 + s_t(6)*gu5*weak_mat_tmp3 + s_t(2)*gu7*weak_mat_tmp14 + s_t(2)*gu7 - weak_mat_tmp12*weak_mat_tmp20 - weak_mat_tmp16*weak_mat_tmp22 - weak_mat_tmp17*weak_mat_tmp6) + weak_mat_tmp5*(gu3*weak_mat_tmp9 - weak_mat_tmp0*weak_mat_tmp3);
    const s_t material8 = mu*(gu3*weak_mat_tmp21 + s_t(2)*gu8 - s_t(6)*weak_mat_tmp1*weak_mat_tmp3 - weak_mat_tmp13*weak_mat_tmp22 + weak_mat_tmp14*weak_mat_tmp22 - weak_mat_tmp16*weak_mat_tmp20 - weak_mat_tmp17*weak_mat_tmp8 + s_t(2)) + weak_mat_tmp5*(-gu3*weak_mat_tmp7 + s_t(2)*weak_mat_tmp1*weak_mat_tmp3);
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
static __host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_tet4_gradient_block(
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
    const s_t weak_mat_tmp0 = s_t(2)*gu5;
    const s_t weak_mat_tmp1 = gu4 + s_t(1);
    const s_t weak_mat_tmp2 = gu8 + s_t(1);
    const s_t weak_mat_tmp3 = gu0 + s_t(1);
    const s_t weak_mat_tmp4 = gu5*gu7;
    const s_t weak_mat_tmp5 = ((s_t(1) / s_t(2)))*lmbda*(-gu1*gu3*weak_mat_tmp2 + gu1*gu5*gu6 + gu2*gu3*gu7 - gu2*gu6*weak_mat_tmp1 + weak_mat_tmp1*weak_mat_tmp2*weak_mat_tmp3 - weak_mat_tmp3*weak_mat_tmp4 + s_t(-1));
    const s_t weak_mat_tmp6 = gu1*weak_mat_tmp3 + gu3*weak_mat_tmp1 + gu6*gu7;
    const s_t weak_mat_tmp7 = s_t(2)*gu1;
    const s_t weak_mat_tmp8 = gu2*weak_mat_tmp3 + gu3*gu5 + gu6*weak_mat_tmp2;
    const s_t weak_mat_tmp9 = s_t(2)*gu2;
    const s_t weak_mat_tmp10 = pow_2(gu3) + pow_2(gu6) + pow_2(weak_mat_tmp3);
    const s_t weak_mat_tmp11 = s_t(2)*weak_mat_tmp3;
    const s_t weak_mat_tmp12 = pow_2(gu1) + pow_2(gu7) + pow_2(weak_mat_tmp1);
    const s_t weak_mat_tmp13 = pow_2(gu2) + pow_2(gu5) + pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp14 = weak_mat_tmp10 + weak_mat_tmp12 + weak_mat_tmp13;
    const s_t weak_mat_tmp15 = s_t(2)*gu3;
    const s_t weak_mat_tmp16 = gu1*gu2 + gu5*weak_mat_tmp1 + gu7*weak_mat_tmp2;
    const s_t weak_mat_tmp17 = s_t(2)*gu6;
    const s_t weak_mat_tmp18 = s_t(6)*gu2;
    const s_t weak_mat_tmp19 = s_t(2)*weak_mat_tmp1;
    const s_t weak_mat_tmp20 = s_t(2)*gu7;
    const s_t weak_mat_tmp21 = s_t(6)*gu1;
    const s_t weak_mat_tmp22 = s_t(2)*weak_mat_tmp2;
    const s_t material0 = mu*(s_t(2)*gu0 - s_t(6)*weak_mat_tmp1*weak_mat_tmp2 - weak_mat_tmp10*weak_mat_tmp11 + weak_mat_tmp11*weak_mat_tmp14 + s_t(6)*weak_mat_tmp4 - weak_mat_tmp6*weak_mat_tmp7 - weak_mat_tmp8*weak_mat_tmp9 + s_t(2)) + weak_mat_tmp5*(-gu7*weak_mat_tmp0 + s_t(2)*weak_mat_tmp1*weak_mat_tmp2);
    const s_t material1 = mu*(s_t(2)*gu1*weak_mat_tmp14 + s_t(2)*gu1 + s_t(6)*gu3*weak_mat_tmp2 - s_t(6)*gu5*gu6 - weak_mat_tmp11*weak_mat_tmp6 - weak_mat_tmp12*weak_mat_tmp7 - weak_mat_tmp16*weak_mat_tmp9) + weak_mat_tmp5*(s_t(2)*gu5*gu6 - weak_mat_tmp15*weak_mat_tmp2);
    const s_t material2 = mu*(s_t(2)*gu2*weak_mat_tmp14 + s_t(2)*gu2 - s_t(6)*gu3*gu7 + s_t(6)*gu6*weak_mat_tmp1 - weak_mat_tmp11*weak_mat_tmp8 - weak_mat_tmp13*weak_mat_tmp9 - weak_mat_tmp16*weak_mat_tmp7) + weak_mat_tmp5*(gu7*weak_mat_tmp15 - weak_mat_tmp1*weak_mat_tmp17);
    const s_t material3 = mu*(s_t(6)*gu1*weak_mat_tmp2 + s_t(2)*gu3*weak_mat_tmp14 + s_t(2)*gu3 - gu7*weak_mat_tmp18 - weak_mat_tmp0*weak_mat_tmp8 - weak_mat_tmp10*weak_mat_tmp15 - weak_mat_tmp19*weak_mat_tmp6) + weak_mat_tmp5*(s_t(2)*gu2*gu7 - weak_mat_tmp2*weak_mat_tmp7);
    const s_t material4 = mu*(s_t(2)*gu4 + gu6*weak_mat_tmp18 - weak_mat_tmp0*weak_mat_tmp16 - weak_mat_tmp12*weak_mat_tmp19 + weak_mat_tmp14*weak_mat_tmp19 - weak_mat_tmp15*weak_mat_tmp6 - s_t(6)*weak_mat_tmp2*weak_mat_tmp3 + s_t(2)) + weak_mat_tmp5*(-gu6*weak_mat_tmp9 + s_t(2)*weak_mat_tmp2*weak_mat_tmp3);
    const s_t material5 = mu*(s_t(2)*gu5*weak_mat_tmp14 + s_t(2)*gu5 - gu6*weak_mat_tmp21 + s_t(6)*gu7*weak_mat_tmp3 - weak_mat_tmp0*weak_mat_tmp13 - weak_mat_tmp15*weak_mat_tmp8 - weak_mat_tmp16*weak_mat_tmp19) + weak_mat_tmp5*(gu6*weak_mat_tmp7 - weak_mat_tmp20*weak_mat_tmp3);
    const s_t material6 = mu*(s_t(6)*gu2*weak_mat_tmp1 - gu5*weak_mat_tmp21 + s_t(2)*gu6*weak_mat_tmp14 + s_t(2)*gu6 - weak_mat_tmp10*weak_mat_tmp17 - weak_mat_tmp20*weak_mat_tmp6 - weak_mat_tmp22*weak_mat_tmp8) + weak_mat_tmp5*(gu5*weak_mat_tmp7 - weak_mat_tmp1*weak_mat_tmp9);
    const s_t material7 = mu*(-gu3*weak_mat_tmp18 + s_t(6)*gu5*weak_mat_tmp3 + s_t(2)*gu7*weak_mat_tmp14 + s_t(2)*gu7 - weak_mat_tmp12*weak_mat_tmp20 - weak_mat_tmp16*weak_mat_tmp22 - weak_mat_tmp17*weak_mat_tmp6) + weak_mat_tmp5*(gu3*weak_mat_tmp9 - weak_mat_tmp0*weak_mat_tmp3);
    const s_t material8 = mu*(gu3*weak_mat_tmp21 + s_t(2)*gu8 - s_t(6)*weak_mat_tmp1*weak_mat_tmp3 - weak_mat_tmp13*weak_mat_tmp22 + weak_mat_tmp14*weak_mat_tmp22 - weak_mat_tmp16*weak_mat_tmp20 - weak_mat_tmp17*weak_mat_tmp8 + s_t(2)) + weak_mat_tmp5*(-gu3*weak_mat_tmp7 + s_t(2)*weak_mat_tmp1*weak_mat_tmp3);
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
static __host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_apply_block(
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
    const s_t weak_mat_tmp11 = -s_t(2)*weak_mat_tmp2*weak_mat_tmp6;
    const s_t weak_mat_tmp12 = ((s_t(1) / s_t(2)))*lmbda;
    const s_t weak_mat_tmp13 = weak_mat_tmp12*(-weak_mat_tmp10 - weak_mat_tmp11);
    const s_t weak_mat_tmp14 = gu5*weak_mat_tmp3;
    const s_t weak_mat_tmp15 = weak_mat_tmp0*weak_mat_tmp6;
    const s_t weak_mat_tmp16 = mu*(-weak_mat_tmp14 - weak_mat_tmp15);
    const s_t weak_mat_tmp17 = gu3*gu7;
    const s_t weak_mat_tmp18 = gu6*weak_mat_tmp2;
    const s_t weak_mat_tmp19 = weak_mat_tmp17 - weak_mat_tmp18;
    const s_t weak_mat_tmp20 = s_t(2)*gu2;
    const s_t weak_mat_tmp21 = gu5*weak_mat_tmp20;
    const s_t weak_mat_tmp22 = s_t(2)*gu1;
    const s_t weak_mat_tmp23 = weak_mat_tmp2*weak_mat_tmp22;
    const s_t weak_mat_tmp24 = mu*(-weak_mat_tmp21 - weak_mat_tmp23);
    const s_t weak_mat_tmp25 = gu1*weak_mat_tmp6;
    const s_t weak_mat_tmp26 = gu2*gu7 - weak_mat_tmp25;
    const s_t weak_mat_tmp27 = gu7*weak_mat_tmp22;
    const s_t weak_mat_tmp28 = weak_mat_tmp20*weak_mat_tmp6;
    const s_t weak_mat_tmp29 = mu*(-weak_mat_tmp27 - weak_mat_tmp28);
    const s_t weak_mat_tmp30 = gu1*gu5;
    const s_t weak_mat_tmp31 = gu2*weak_mat_tmp2;
    const s_t weak_mat_tmp32 = weak_mat_tmp30 - weak_mat_tmp31;
    const s_t weak_mat_tmp33 = s_t(2)*pow_2(gu5);
    const s_t weak_mat_tmp34 = s_t(2)*pow_2(weak_mat_tmp6) + s_t(2);
    const s_t weak_mat_tmp35 = weak_mat_tmp33 + weak_mat_tmp34;
    const s_t weak_mat_tmp36 = s_t(2)*pow_2(gu7);
    const s_t weak_mat_tmp37 = s_t(2)*pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp38 = weak_mat_tmp36 + weak_mat_tmp37;
    const s_t weak_mat_tmp39 = weak_mat_tmp2*weak_mat_tmp6 - weak_mat_tmp9;
    const s_t weak_mat_tmp40 = gu1*gu6;
    const s_t weak_mat_tmp41 = gu0 + s_t(1);
    const s_t weak_mat_tmp42 = gu7*weak_mat_tmp41;
    const s_t weak_mat_tmp43 = weak_mat_tmp40 - weak_mat_tmp42;
    const s_t weak_mat_tmp44 = s_t(6)*gu7;
    const s_t weak_mat_tmp45 = gu2*gu3;
    const s_t weak_mat_tmp46 = s_t(2)*weak_mat_tmp45;
    const s_t weak_mat_tmp47 = gu5*weak_mat_tmp41;
    const s_t weak_mat_tmp48 = lmbda*(gu1*gu5*gu6 - gu1*weak_mat_tmp7 + gu2*gu3*gu7 - gu2*weak_mat_tmp18 + weak_mat_tmp2*weak_mat_tmp41*weak_mat_tmp6 - weak_mat_tmp41*weak_mat_tmp9 + s_t(-1));
    const s_t weak_mat_tmp49 = gu7*weak_mat_tmp48;
    const s_t weak_mat_tmp50 = mu*(weak_mat_tmp44 - weak_mat_tmp46 + s_t(4)*weak_mat_tmp47) - weak_mat_tmp49;
    const s_t weak_mat_tmp51 = weak_mat_tmp45 - weak_mat_tmp47;
    const s_t weak_mat_tmp52 = s_t(6)*gu5;
    const s_t weak_mat_tmp53 = s_t(2)*weak_mat_tmp40;
    const s_t weak_mat_tmp54 = gu5*weak_mat_tmp48;
    const s_t weak_mat_tmp55 = mu*(s_t(4)*weak_mat_tmp42 + weak_mat_tmp52 - weak_mat_tmp53) - weak_mat_tmp54;
    const s_t weak_mat_tmp56 = gu2*gu6;
    const s_t weak_mat_tmp57 = weak_mat_tmp41*weak_mat_tmp6 - weak_mat_tmp56;
    const s_t weak_mat_tmp58 = gu1*gu3;
    const s_t weak_mat_tmp59 = s_t(2)*weak_mat_tmp58;
    const s_t weak_mat_tmp60 = s_t(6)*gu8 + s_t(6);
    const s_t weak_mat_tmp61 = weak_mat_tmp48*weak_mat_tmp6;
    const s_t weak_mat_tmp62 = mu*(s_t(4)*weak_mat_tmp2*weak_mat_tmp41 - weak_mat_tmp59 - weak_mat_tmp60) + weak_mat_tmp61;
    const s_t weak_mat_tmp63 = weak_mat_tmp2*weak_mat_tmp41 - weak_mat_tmp58;
    const s_t weak_mat_tmp64 = s_t(2)*weak_mat_tmp56;
    const s_t weak_mat_tmp65 = s_t(6)*gu4 + s_t(6);
    const s_t weak_mat_tmp66 = weak_mat_tmp2*weak_mat_tmp48;
    const s_t weak_mat_tmp67 = mu*(s_t(4)*weak_mat_tmp41*weak_mat_tmp6 - weak_mat_tmp64 - weak_mat_tmp65) + weak_mat_tmp66;
    const s_t weak_mat_tmp68 = -s_t(2)*gu5*gu6;
    const s_t weak_mat_tmp69 = s_t(2)*weak_mat_tmp7;
    const s_t weak_mat_tmp70 = weak_mat_tmp12*(-weak_mat_tmp68 - weak_mat_tmp69);
    const s_t weak_mat_tmp71 = s_t(2)*gu5;
    const s_t weak_mat_tmp72 = weak_mat_tmp2*weak_mat_tmp71;
    const s_t weak_mat_tmp73 = s_t(2)*gu7;
    const s_t weak_mat_tmp74 = weak_mat_tmp6*weak_mat_tmp73;
    const s_t weak_mat_tmp75 = mu*(-weak_mat_tmp72 - weak_mat_tmp74);
    const s_t weak_mat_tmp76 = weak_mat_tmp3*weak_mat_tmp41;
    const s_t weak_mat_tmp77 = mu*(-weak_mat_tmp21 - weak_mat_tmp76);
    const s_t weak_mat_tmp78 = weak_mat_tmp0*weak_mat_tmp41;
    const s_t weak_mat_tmp79 = mu*(-weak_mat_tmp28 - weak_mat_tmp78);
    const s_t weak_mat_tmp80 = s_t(2)*pow_2(gu3);
    const s_t weak_mat_tmp81 = s_t(2)*pow_2(gu6);
    const s_t weak_mat_tmp82 = weak_mat_tmp80 + weak_mat_tmp81;
    const s_t weak_mat_tmp83 = s_t(6)*gu6;
    const s_t weak_mat_tmp84 = s_t(2)*weak_mat_tmp31;
    const s_t weak_mat_tmp85 = gu6*weak_mat_tmp48;
    const s_t weak_mat_tmp86 = mu*(s_t(4)*gu1*gu5 - weak_mat_tmp83 - weak_mat_tmp84) + weak_mat_tmp85;
    const s_t weak_mat_tmp87 = s_t(2)*weak_mat_tmp42;
    const s_t weak_mat_tmp88 = mu*(s_t(4)*gu1*gu6 - weak_mat_tmp52 - weak_mat_tmp87) + weak_mat_tmp54;
    const s_t weak_mat_tmp89 = s_t(6)*gu3;
    const s_t weak_mat_tmp90 = -s_t(2)*gu2*gu7;
    const s_t weak_mat_tmp91 = gu3*weak_mat_tmp48;
    const s_t weak_mat_tmp92 = mu*(s_t(4)*weak_mat_tmp25 + weak_mat_tmp89 + weak_mat_tmp90) - weak_mat_tmp91;
    const s_t weak_mat_tmp93 = -s_t(2)*weak_mat_tmp2*weak_mat_tmp41;
    const s_t weak_mat_tmp94 = mu*(s_t(4)*weak_mat_tmp58 + weak_mat_tmp60 + weak_mat_tmp93) - weak_mat_tmp61;
    const s_t weak_mat_tmp95 = s_t(2)*weak_mat_tmp17;
    const s_t weak_mat_tmp96 = s_t(2)*weak_mat_tmp18;
    const s_t weak_mat_tmp97 = weak_mat_tmp12*(weak_mat_tmp95 - weak_mat_tmp96);
    const s_t weak_mat_tmp98 = mu*(-weak_mat_tmp23 - weak_mat_tmp76);
    const s_t weak_mat_tmp99 = mu*(-weak_mat_tmp27 - weak_mat_tmp78);
    const s_t weak_mat_tmp100 = s_t(2)*weak_mat_tmp47;
    const s_t weak_mat_tmp101 = mu*(s_t(4)*gu2*gu3 - weak_mat_tmp100 - weak_mat_tmp44) + weak_mat_tmp49;
    const s_t weak_mat_tmp102 = s_t(2)*weak_mat_tmp25;
    const s_t weak_mat_tmp103 = mu*(s_t(4)*gu2*gu7 - weak_mat_tmp102 - weak_mat_tmp89) + weak_mat_tmp91;
    const s_t weak_mat_tmp104 = s_t(2)*weak_mat_tmp30;
    const s_t weak_mat_tmp105 = mu*(-weak_mat_tmp104 + s_t(4)*weak_mat_tmp31 + weak_mat_tmp83) - weak_mat_tmp85;
    const s_t weak_mat_tmp106 = -s_t(2)*weak_mat_tmp41*weak_mat_tmp6;
    const s_t weak_mat_tmp107 = mu*(weak_mat_tmp106 + s_t(4)*weak_mat_tmp56 + weak_mat_tmp65) - weak_mat_tmp66;
    const s_t weak_mat_tmp108 = weak_mat_tmp12*(-weak_mat_tmp102 - weak_mat_tmp90);
    const s_t weak_mat_tmp109 = weak_mat_tmp22*weak_mat_tmp41;
    const s_t weak_mat_tmp110 = mu*(-weak_mat_tmp1 - weak_mat_tmp109);
    const s_t weak_mat_tmp111 = weak_mat_tmp20*weak_mat_tmp41;
    const s_t weak_mat_tmp112 = mu*(-weak_mat_tmp111 - weak_mat_tmp15);
    const s_t weak_mat_tmp113 = weak_mat_tmp6*weak_mat_tmp71;
    const s_t weak_mat_tmp114 = weak_mat_tmp2*weak_mat_tmp73;
    const s_t weak_mat_tmp115 = mu*(-weak_mat_tmp113 - weak_mat_tmp114);
    const s_t weak_mat_tmp116 = s_t(2)*pow_2(gu2);
    const s_t weak_mat_tmp117 = weak_mat_tmp116 + weak_mat_tmp34;
    const s_t weak_mat_tmp118 = s_t(2)*pow_2(gu1);
    const s_t weak_mat_tmp119 = weak_mat_tmp118 + weak_mat_tmp36;
    const s_t weak_mat_tmp120 = s_t(6)*gu2;
    const s_t weak_mat_tmp121 = gu2*weak_mat_tmp48;
    const s_t weak_mat_tmp122 = mu*(s_t(4)*gu3*gu7 - weak_mat_tmp120 - weak_mat_tmp96) + weak_mat_tmp121;
    const s_t weak_mat_tmp123 = s_t(6)*gu1;
    const s_t weak_mat_tmp124 = gu1*weak_mat_tmp48;
    const s_t weak_mat_tmp125 = mu*(weak_mat_tmp123 + weak_mat_tmp68 + s_t(4)*weak_mat_tmp7) - weak_mat_tmp124;
    const s_t weak_mat_tmp126 = weak_mat_tmp12*(-weak_mat_tmp106 - weak_mat_tmp64);
    const s_t weak_mat_tmp127 = gu2*weak_mat_tmp22;
    const s_t weak_mat_tmp128 = mu*(-weak_mat_tmp127 - weak_mat_tmp74);
    const s_t weak_mat_tmp129 = gu6*weak_mat_tmp3;
    const s_t weak_mat_tmp130 = mu*(-weak_mat_tmp113 - weak_mat_tmp129);
    const s_t weak_mat_tmp131 = s_t(2)*pow_2(weak_mat_tmp41);
    const s_t weak_mat_tmp132 = weak_mat_tmp131 + weak_mat_tmp81;
    const s_t weak_mat_tmp133 = mu*(weak_mat_tmp120 + s_t(4)*weak_mat_tmp18 - weak_mat_tmp95) - weak_mat_tmp121;
    const s_t weak_mat_tmp134 = s_t(6)*gu0 + s_t(6);
    const s_t weak_mat_tmp135 = weak_mat_tmp41*weak_mat_tmp48;
    const s_t weak_mat_tmp136 = mu*(-weak_mat_tmp10 - weak_mat_tmp134 + s_t(4)*weak_mat_tmp2*weak_mat_tmp6) + weak_mat_tmp135;
    const s_t weak_mat_tmp137 = weak_mat_tmp12*(weak_mat_tmp53 - weak_mat_tmp87);
    const s_t weak_mat_tmp138 = mu*(-weak_mat_tmp114 - weak_mat_tmp129);
    const s_t weak_mat_tmp139 = mu*(s_t(4)*gu5*gu6 - weak_mat_tmp123 - weak_mat_tmp69) + weak_mat_tmp124;
    const s_t weak_mat_tmp140 = mu*(weak_mat_tmp11 + weak_mat_tmp134 + s_t(4)*weak_mat_tmp9) - weak_mat_tmp135;
    const s_t weak_mat_tmp141 = weak_mat_tmp12*(weak_mat_tmp104 - weak_mat_tmp84);
    const s_t weak_mat_tmp142 = mu*(-weak_mat_tmp109 - weak_mat_tmp4);
    const s_t weak_mat_tmp143 = mu*(-weak_mat_tmp111 - weak_mat_tmp14);
    const s_t weak_mat_tmp144 = weak_mat_tmp116 + weak_mat_tmp33 + s_t(2);
    const s_t weak_mat_tmp145 = weak_mat_tmp118 + weak_mat_tmp37;
    const s_t weak_mat_tmp146 = weak_mat_tmp12*(-weak_mat_tmp100 + weak_mat_tmp46);
    const s_t weak_mat_tmp147 = mu*(-weak_mat_tmp127 - weak_mat_tmp72);
    const s_t weak_mat_tmp148 = weak_mat_tmp131 + weak_mat_tmp80;
    const s_t weak_mat_tmp149 = weak_mat_tmp12*(-weak_mat_tmp59 - weak_mat_tmp93);
    const s_t material0 = trial_grad0*(mu*(weak_mat_tmp35 + weak_mat_tmp38) + weak_mat_tmp13*weak_mat_tmp39) + trial_grad1*(weak_mat_tmp13*weak_mat_tmp8 + weak_mat_tmp5) + trial_grad2*(weak_mat_tmp13*weak_mat_tmp19 + weak_mat_tmp16) + trial_grad3*(weak_mat_tmp13*weak_mat_tmp26 + weak_mat_tmp24) + trial_grad4*(weak_mat_tmp13*weak_mat_tmp57 + weak_mat_tmp62) + trial_grad5*(weak_mat_tmp13*weak_mat_tmp43 + weak_mat_tmp50) + trial_grad6*(weak_mat_tmp13*weak_mat_tmp32 + weak_mat_tmp29) + trial_grad7*(weak_mat_tmp13*weak_mat_tmp51 + weak_mat_tmp55) + trial_grad8*(weak_mat_tmp13*weak_mat_tmp63 + weak_mat_tmp67);
    const s_t material1 = trial_grad0*(weak_mat_tmp39*weak_mat_tmp70 + weak_mat_tmp5) + trial_grad1*(mu*(weak_mat_tmp35 + weak_mat_tmp82) + weak_mat_tmp70*weak_mat_tmp8) + trial_grad2*(weak_mat_tmp19*weak_mat_tmp70 + weak_mat_tmp75) + trial_grad3*(weak_mat_tmp26*weak_mat_tmp70 + weak_mat_tmp94) + trial_grad4*(weak_mat_tmp57*weak_mat_tmp70 + weak_mat_tmp77) + trial_grad5*(weak_mat_tmp43*weak_mat_tmp70 + weak_mat_tmp86) + trial_grad6*(weak_mat_tmp32*weak_mat_tmp70 + weak_mat_tmp88) + trial_grad7*(weak_mat_tmp51*weak_mat_tmp70 + weak_mat_tmp79) + trial_grad8*(weak_mat_tmp63*weak_mat_tmp70 + weak_mat_tmp92);
    const s_t material2 = trial_grad0*(weak_mat_tmp16 + weak_mat_tmp39*weak_mat_tmp97) + trial_grad1*(weak_mat_tmp75 + weak_mat_tmp8*weak_mat_tmp97) + trial_grad2*(mu*(weak_mat_tmp38 + weak_mat_tmp82 + s_t(2)) + weak_mat_tmp19*weak_mat_tmp97) + trial_grad3*(weak_mat_tmp101 + weak_mat_tmp26*weak_mat_tmp97) + trial_grad4*(weak_mat_tmp105 + weak_mat_tmp57*weak_mat_tmp97) + trial_grad5*(weak_mat_tmp43*weak_mat_tmp97 + weak_mat_tmp98) + trial_grad6*(weak_mat_tmp107 + weak_mat_tmp32*weak_mat_tmp97) + trial_grad7*(weak_mat_tmp103 + weak_mat_tmp51*weak_mat_tmp97) + trial_grad8*(weak_mat_tmp63*weak_mat_tmp97 + weak_mat_tmp99);
    const s_t material3 = trial_grad0*(weak_mat_tmp108*weak_mat_tmp39 + weak_mat_tmp24) + trial_grad1*(weak_mat_tmp108*weak_mat_tmp8 + weak_mat_tmp94) + trial_grad2*(weak_mat_tmp101 + weak_mat_tmp108*weak_mat_tmp19) + trial_grad3*(mu*(weak_mat_tmp117 + weak_mat_tmp119) + weak_mat_tmp108*weak_mat_tmp26) + trial_grad4*(weak_mat_tmp108*weak_mat_tmp57 + weak_mat_tmp110) + trial_grad5*(weak_mat_tmp108*weak_mat_tmp43 + weak_mat_tmp112) + trial_grad6*(weak_mat_tmp108*weak_mat_tmp32 + weak_mat_tmp115) + trial_grad7*(weak_mat_tmp108*weak_mat_tmp51 + weak_mat_tmp122) + trial_grad8*(weak_mat_tmp108*weak_mat_tmp63 + weak_mat_tmp125);
    const s_t material4 = trial_grad0*(weak_mat_tmp126*weak_mat_tmp39 + weak_mat_tmp62) + trial_grad1*(weak_mat_tmp126*weak_mat_tmp8 + weak_mat_tmp77) + trial_grad2*(weak_mat_tmp105 + weak_mat_tmp126*weak_mat_tmp19) + trial_grad3*(weak_mat_tmp110 + weak_mat_tmp126*weak_mat_tmp26) + trial_grad4*(mu*(weak_mat_tmp117 + weak_mat_tmp132) + weak_mat_tmp126*weak_mat_tmp57) + trial_grad5*(weak_mat_tmp126*weak_mat_tmp43 + weak_mat_tmp128) + trial_grad6*(weak_mat_tmp126*weak_mat_tmp32 + weak_mat_tmp133) + trial_grad7*(weak_mat_tmp126*weak_mat_tmp51 + weak_mat_tmp130) + trial_grad8*(weak_mat_tmp126*weak_mat_tmp63 + weak_mat_tmp136);
    const s_t material5 = trial_grad0*(weak_mat_tmp137*weak_mat_tmp39 + weak_mat_tmp50) + trial_grad1*(weak_mat_tmp137*weak_mat_tmp8 + weak_mat_tmp86) + trial_grad2*(weak_mat_tmp137*weak_mat_tmp19 + weak_mat_tmp98) + trial_grad3*(weak_mat_tmp112 + weak_mat_tmp137*weak_mat_tmp26) + trial_grad4*(weak_mat_tmp128 + weak_mat_tmp137*weak_mat_tmp57) + trial_grad5*(mu*(weak_mat_tmp119 + weak_mat_tmp132 + s_t(2)) + weak_mat_tmp137*weak_mat_tmp43) + trial_grad6*(weak_mat_tmp137*weak_mat_tmp32 + weak_mat_tmp139) + trial_grad7*(weak_mat_tmp137*weak_mat_tmp51 + weak_mat_tmp140) + trial_grad8*(weak_mat_tmp137*weak_mat_tmp63 + weak_mat_tmp138);
    const s_t material6 = trial_grad0*(weak_mat_tmp141*weak_mat_tmp39 + weak_mat_tmp29) + trial_grad1*(weak_mat_tmp141*weak_mat_tmp8 + weak_mat_tmp88) + trial_grad2*(weak_mat_tmp107 + weak_mat_tmp141*weak_mat_tmp19) + trial_grad3*(weak_mat_tmp115 + weak_mat_tmp141*weak_mat_tmp26) + trial_grad4*(weak_mat_tmp133 + weak_mat_tmp141*weak_mat_tmp57) + trial_grad5*(weak_mat_tmp139 + weak_mat_tmp141*weak_mat_tmp43) + trial_grad6*(mu*(weak_mat_tmp144 + weak_mat_tmp145) + weak_mat_tmp141*weak_mat_tmp32) + trial_grad7*(weak_mat_tmp141*weak_mat_tmp51 + weak_mat_tmp142) + trial_grad8*(weak_mat_tmp141*weak_mat_tmp63 + weak_mat_tmp143);
    const s_t material7 = trial_grad0*(weak_mat_tmp146*weak_mat_tmp39 + weak_mat_tmp55) + trial_grad1*(weak_mat_tmp146*weak_mat_tmp8 + weak_mat_tmp79) + trial_grad2*(weak_mat_tmp103 + weak_mat_tmp146*weak_mat_tmp19) + trial_grad3*(weak_mat_tmp122 + weak_mat_tmp146*weak_mat_tmp26) + trial_grad4*(weak_mat_tmp130 + weak_mat_tmp146*weak_mat_tmp57) + trial_grad5*(weak_mat_tmp140 + weak_mat_tmp146*weak_mat_tmp43) + trial_grad6*(weak_mat_tmp142 + weak_mat_tmp146*weak_mat_tmp32) + trial_grad7*(mu*(weak_mat_tmp144 + weak_mat_tmp148) + weak_mat_tmp146*weak_mat_tmp51) + trial_grad8*(weak_mat_tmp146*weak_mat_tmp63 + weak_mat_tmp147);
    const s_t material8 = trial_grad0*(weak_mat_tmp149*weak_mat_tmp39 + weak_mat_tmp67) + trial_grad1*(weak_mat_tmp149*weak_mat_tmp8 + weak_mat_tmp92) + trial_grad2*(weak_mat_tmp149*weak_mat_tmp19 + weak_mat_tmp99) + trial_grad3*(weak_mat_tmp125 + weak_mat_tmp149*weak_mat_tmp26) + trial_grad4*(weak_mat_tmp136 + weak_mat_tmp149*weak_mat_tmp57) + trial_grad5*(weak_mat_tmp138 + weak_mat_tmp149*weak_mat_tmp43) + trial_grad6*(weak_mat_tmp143 + weak_mat_tmp149*weak_mat_tmp32) + trial_grad7*(weak_mat_tmp147 + weak_mat_tmp149*weak_mat_tmp51) + trial_grad8*(mu*(weak_mat_tmp145 + weak_mat_tmp148 + s_t(2)) + weak_mat_tmp149*weak_mat_tmp63);
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
static __host__ __device__ __forceinline__ void mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_tet4_apply_block(
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
    const s_t weak_mat_tmp11 = -s_t(2)*weak_mat_tmp2*weak_mat_tmp6;
    const s_t weak_mat_tmp12 = ((s_t(1) / s_t(2)))*lmbda;
    const s_t weak_mat_tmp13 = weak_mat_tmp12*(-weak_mat_tmp10 - weak_mat_tmp11);
    const s_t weak_mat_tmp14 = gu5*weak_mat_tmp3;
    const s_t weak_mat_tmp15 = weak_mat_tmp0*weak_mat_tmp6;
    const s_t weak_mat_tmp16 = mu*(-weak_mat_tmp14 - weak_mat_tmp15);
    const s_t weak_mat_tmp17 = gu3*gu7;
    const s_t weak_mat_tmp18 = gu6*weak_mat_tmp2;
    const s_t weak_mat_tmp19 = weak_mat_tmp17 - weak_mat_tmp18;
    const s_t weak_mat_tmp20 = s_t(2)*gu2;
    const s_t weak_mat_tmp21 = gu5*weak_mat_tmp20;
    const s_t weak_mat_tmp22 = s_t(2)*gu1;
    const s_t weak_mat_tmp23 = weak_mat_tmp2*weak_mat_tmp22;
    const s_t weak_mat_tmp24 = mu*(-weak_mat_tmp21 - weak_mat_tmp23);
    const s_t weak_mat_tmp25 = gu1*weak_mat_tmp6;
    const s_t weak_mat_tmp26 = gu2*gu7 - weak_mat_tmp25;
    const s_t weak_mat_tmp27 = gu7*weak_mat_tmp22;
    const s_t weak_mat_tmp28 = weak_mat_tmp20*weak_mat_tmp6;
    const s_t weak_mat_tmp29 = mu*(-weak_mat_tmp27 - weak_mat_tmp28);
    const s_t weak_mat_tmp30 = gu1*gu5;
    const s_t weak_mat_tmp31 = gu2*weak_mat_tmp2;
    const s_t weak_mat_tmp32 = weak_mat_tmp30 - weak_mat_tmp31;
    const s_t weak_mat_tmp33 = s_t(2)*pow_2(gu5);
    const s_t weak_mat_tmp34 = s_t(2)*pow_2(weak_mat_tmp6) + s_t(2);
    const s_t weak_mat_tmp35 = weak_mat_tmp33 + weak_mat_tmp34;
    const s_t weak_mat_tmp36 = s_t(2)*pow_2(gu7);
    const s_t weak_mat_tmp37 = s_t(2)*pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp38 = weak_mat_tmp36 + weak_mat_tmp37;
    const s_t weak_mat_tmp39 = weak_mat_tmp2*weak_mat_tmp6 - weak_mat_tmp9;
    const s_t weak_mat_tmp40 = gu1*gu6;
    const s_t weak_mat_tmp41 = gu0 + s_t(1);
    const s_t weak_mat_tmp42 = gu7*weak_mat_tmp41;
    const s_t weak_mat_tmp43 = weak_mat_tmp40 - weak_mat_tmp42;
    const s_t weak_mat_tmp44 = s_t(6)*gu7;
    const s_t weak_mat_tmp45 = gu2*gu3;
    const s_t weak_mat_tmp46 = s_t(2)*weak_mat_tmp45;
    const s_t weak_mat_tmp47 = gu5*weak_mat_tmp41;
    const s_t weak_mat_tmp48 = lmbda*(gu1*gu5*gu6 - gu1*weak_mat_tmp7 + gu2*gu3*gu7 - gu2*weak_mat_tmp18 + weak_mat_tmp2*weak_mat_tmp41*weak_mat_tmp6 - weak_mat_tmp41*weak_mat_tmp9 + s_t(-1));
    const s_t weak_mat_tmp49 = gu7*weak_mat_tmp48;
    const s_t weak_mat_tmp50 = mu*(weak_mat_tmp44 - weak_mat_tmp46 + s_t(4)*weak_mat_tmp47) - weak_mat_tmp49;
    const s_t weak_mat_tmp51 = weak_mat_tmp45 - weak_mat_tmp47;
    const s_t weak_mat_tmp52 = s_t(6)*gu5;
    const s_t weak_mat_tmp53 = s_t(2)*weak_mat_tmp40;
    const s_t weak_mat_tmp54 = gu5*weak_mat_tmp48;
    const s_t weak_mat_tmp55 = mu*(s_t(4)*weak_mat_tmp42 + weak_mat_tmp52 - weak_mat_tmp53) - weak_mat_tmp54;
    const s_t weak_mat_tmp56 = gu2*gu6;
    const s_t weak_mat_tmp57 = weak_mat_tmp41*weak_mat_tmp6 - weak_mat_tmp56;
    const s_t weak_mat_tmp58 = gu1*gu3;
    const s_t weak_mat_tmp59 = s_t(2)*weak_mat_tmp58;
    const s_t weak_mat_tmp60 = s_t(6)*gu8 + s_t(6);
    const s_t weak_mat_tmp61 = weak_mat_tmp48*weak_mat_tmp6;
    const s_t weak_mat_tmp62 = mu*(s_t(4)*weak_mat_tmp2*weak_mat_tmp41 - weak_mat_tmp59 - weak_mat_tmp60) + weak_mat_tmp61;
    const s_t weak_mat_tmp63 = weak_mat_tmp2*weak_mat_tmp41 - weak_mat_tmp58;
    const s_t weak_mat_tmp64 = s_t(2)*weak_mat_tmp56;
    const s_t weak_mat_tmp65 = s_t(6)*gu4 + s_t(6);
    const s_t weak_mat_tmp66 = weak_mat_tmp2*weak_mat_tmp48;
    const s_t weak_mat_tmp67 = mu*(s_t(4)*weak_mat_tmp41*weak_mat_tmp6 - weak_mat_tmp64 - weak_mat_tmp65) + weak_mat_tmp66;
    const s_t weak_mat_tmp68 = -s_t(2)*gu5*gu6;
    const s_t weak_mat_tmp69 = s_t(2)*weak_mat_tmp7;
    const s_t weak_mat_tmp70 = weak_mat_tmp12*(-weak_mat_tmp68 - weak_mat_tmp69);
    const s_t weak_mat_tmp71 = s_t(2)*gu5;
    const s_t weak_mat_tmp72 = weak_mat_tmp2*weak_mat_tmp71;
    const s_t weak_mat_tmp73 = s_t(2)*gu7;
    const s_t weak_mat_tmp74 = weak_mat_tmp6*weak_mat_tmp73;
    const s_t weak_mat_tmp75 = mu*(-weak_mat_tmp72 - weak_mat_tmp74);
    const s_t weak_mat_tmp76 = weak_mat_tmp3*weak_mat_tmp41;
    const s_t weak_mat_tmp77 = mu*(-weak_mat_tmp21 - weak_mat_tmp76);
    const s_t weak_mat_tmp78 = weak_mat_tmp0*weak_mat_tmp41;
    const s_t weak_mat_tmp79 = mu*(-weak_mat_tmp28 - weak_mat_tmp78);
    const s_t weak_mat_tmp80 = s_t(2)*pow_2(gu3);
    const s_t weak_mat_tmp81 = s_t(2)*pow_2(gu6);
    const s_t weak_mat_tmp82 = weak_mat_tmp80 + weak_mat_tmp81;
    const s_t weak_mat_tmp83 = s_t(6)*gu6;
    const s_t weak_mat_tmp84 = s_t(2)*weak_mat_tmp31;
    const s_t weak_mat_tmp85 = gu6*weak_mat_tmp48;
    const s_t weak_mat_tmp86 = mu*(s_t(4)*gu1*gu5 - weak_mat_tmp83 - weak_mat_tmp84) + weak_mat_tmp85;
    const s_t weak_mat_tmp87 = s_t(2)*weak_mat_tmp42;
    const s_t weak_mat_tmp88 = mu*(s_t(4)*gu1*gu6 - weak_mat_tmp52 - weak_mat_tmp87) + weak_mat_tmp54;
    const s_t weak_mat_tmp89 = s_t(6)*gu3;
    const s_t weak_mat_tmp90 = -s_t(2)*gu2*gu7;
    const s_t weak_mat_tmp91 = gu3*weak_mat_tmp48;
    const s_t weak_mat_tmp92 = mu*(s_t(4)*weak_mat_tmp25 + weak_mat_tmp89 + weak_mat_tmp90) - weak_mat_tmp91;
    const s_t weak_mat_tmp93 = -s_t(2)*weak_mat_tmp2*weak_mat_tmp41;
    const s_t weak_mat_tmp94 = mu*(s_t(4)*weak_mat_tmp58 + weak_mat_tmp60 + weak_mat_tmp93) - weak_mat_tmp61;
    const s_t weak_mat_tmp95 = s_t(2)*weak_mat_tmp17;
    const s_t weak_mat_tmp96 = s_t(2)*weak_mat_tmp18;
    const s_t weak_mat_tmp97 = weak_mat_tmp12*(weak_mat_tmp95 - weak_mat_tmp96);
    const s_t weak_mat_tmp98 = mu*(-weak_mat_tmp23 - weak_mat_tmp76);
    const s_t weak_mat_tmp99 = mu*(-weak_mat_tmp27 - weak_mat_tmp78);
    const s_t weak_mat_tmp100 = s_t(2)*weak_mat_tmp47;
    const s_t weak_mat_tmp101 = mu*(s_t(4)*gu2*gu3 - weak_mat_tmp100 - weak_mat_tmp44) + weak_mat_tmp49;
    const s_t weak_mat_tmp102 = s_t(2)*weak_mat_tmp25;
    const s_t weak_mat_tmp103 = mu*(s_t(4)*gu2*gu7 - weak_mat_tmp102 - weak_mat_tmp89) + weak_mat_tmp91;
    const s_t weak_mat_tmp104 = s_t(2)*weak_mat_tmp30;
    const s_t weak_mat_tmp105 = mu*(-weak_mat_tmp104 + s_t(4)*weak_mat_tmp31 + weak_mat_tmp83) - weak_mat_tmp85;
    const s_t weak_mat_tmp106 = -s_t(2)*weak_mat_tmp41*weak_mat_tmp6;
    const s_t weak_mat_tmp107 = mu*(weak_mat_tmp106 + s_t(4)*weak_mat_tmp56 + weak_mat_tmp65) - weak_mat_tmp66;
    const s_t weak_mat_tmp108 = weak_mat_tmp12*(-weak_mat_tmp102 - weak_mat_tmp90);
    const s_t weak_mat_tmp109 = weak_mat_tmp22*weak_mat_tmp41;
    const s_t weak_mat_tmp110 = mu*(-weak_mat_tmp1 - weak_mat_tmp109);
    const s_t weak_mat_tmp111 = weak_mat_tmp20*weak_mat_tmp41;
    const s_t weak_mat_tmp112 = mu*(-weak_mat_tmp111 - weak_mat_tmp15);
    const s_t weak_mat_tmp113 = weak_mat_tmp6*weak_mat_tmp71;
    const s_t weak_mat_tmp114 = weak_mat_tmp2*weak_mat_tmp73;
    const s_t weak_mat_tmp115 = mu*(-weak_mat_tmp113 - weak_mat_tmp114);
    const s_t weak_mat_tmp116 = s_t(2)*pow_2(gu2);
    const s_t weak_mat_tmp117 = weak_mat_tmp116 + weak_mat_tmp34;
    const s_t weak_mat_tmp118 = s_t(2)*pow_2(gu1);
    const s_t weak_mat_tmp119 = weak_mat_tmp118 + weak_mat_tmp36;
    const s_t weak_mat_tmp120 = s_t(6)*gu2;
    const s_t weak_mat_tmp121 = gu2*weak_mat_tmp48;
    const s_t weak_mat_tmp122 = mu*(s_t(4)*gu3*gu7 - weak_mat_tmp120 - weak_mat_tmp96) + weak_mat_tmp121;
    const s_t weak_mat_tmp123 = s_t(6)*gu1;
    const s_t weak_mat_tmp124 = gu1*weak_mat_tmp48;
    const s_t weak_mat_tmp125 = mu*(weak_mat_tmp123 + weak_mat_tmp68 + s_t(4)*weak_mat_tmp7) - weak_mat_tmp124;
    const s_t weak_mat_tmp126 = weak_mat_tmp12*(-weak_mat_tmp106 - weak_mat_tmp64);
    const s_t weak_mat_tmp127 = gu2*weak_mat_tmp22;
    const s_t weak_mat_tmp128 = mu*(-weak_mat_tmp127 - weak_mat_tmp74);
    const s_t weak_mat_tmp129 = gu6*weak_mat_tmp3;
    const s_t weak_mat_tmp130 = mu*(-weak_mat_tmp113 - weak_mat_tmp129);
    const s_t weak_mat_tmp131 = s_t(2)*pow_2(weak_mat_tmp41);
    const s_t weak_mat_tmp132 = weak_mat_tmp131 + weak_mat_tmp81;
    const s_t weak_mat_tmp133 = mu*(weak_mat_tmp120 + s_t(4)*weak_mat_tmp18 - weak_mat_tmp95) - weak_mat_tmp121;
    const s_t weak_mat_tmp134 = s_t(6)*gu0 + s_t(6);
    const s_t weak_mat_tmp135 = weak_mat_tmp41*weak_mat_tmp48;
    const s_t weak_mat_tmp136 = mu*(-weak_mat_tmp10 - weak_mat_tmp134 + s_t(4)*weak_mat_tmp2*weak_mat_tmp6) + weak_mat_tmp135;
    const s_t weak_mat_tmp137 = weak_mat_tmp12*(weak_mat_tmp53 - weak_mat_tmp87);
    const s_t weak_mat_tmp138 = mu*(-weak_mat_tmp114 - weak_mat_tmp129);
    const s_t weak_mat_tmp139 = mu*(s_t(4)*gu5*gu6 - weak_mat_tmp123 - weak_mat_tmp69) + weak_mat_tmp124;
    const s_t weak_mat_tmp140 = mu*(weak_mat_tmp11 + weak_mat_tmp134 + s_t(4)*weak_mat_tmp9) - weak_mat_tmp135;
    const s_t weak_mat_tmp141 = weak_mat_tmp12*(weak_mat_tmp104 - weak_mat_tmp84);
    const s_t weak_mat_tmp142 = mu*(-weak_mat_tmp109 - weak_mat_tmp4);
    const s_t weak_mat_tmp143 = mu*(-weak_mat_tmp111 - weak_mat_tmp14);
    const s_t weak_mat_tmp144 = weak_mat_tmp116 + weak_mat_tmp33 + s_t(2);
    const s_t weak_mat_tmp145 = weak_mat_tmp118 + weak_mat_tmp37;
    const s_t weak_mat_tmp146 = weak_mat_tmp12*(-weak_mat_tmp100 + weak_mat_tmp46);
    const s_t weak_mat_tmp147 = mu*(-weak_mat_tmp127 - weak_mat_tmp72);
    const s_t weak_mat_tmp148 = weak_mat_tmp131 + weak_mat_tmp80;
    const s_t weak_mat_tmp149 = weak_mat_tmp12*(-weak_mat_tmp59 - weak_mat_tmp93);
    const s_t material0 = trial_grad0*(mu*(weak_mat_tmp35 + weak_mat_tmp38) + weak_mat_tmp13*weak_mat_tmp39) + trial_grad1*(weak_mat_tmp13*weak_mat_tmp8 + weak_mat_tmp5) + trial_grad2*(weak_mat_tmp13*weak_mat_tmp19 + weak_mat_tmp16) + trial_grad3*(weak_mat_tmp13*weak_mat_tmp26 + weak_mat_tmp24) + trial_grad4*(weak_mat_tmp13*weak_mat_tmp57 + weak_mat_tmp62) + trial_grad5*(weak_mat_tmp13*weak_mat_tmp43 + weak_mat_tmp50) + trial_grad6*(weak_mat_tmp13*weak_mat_tmp32 + weak_mat_tmp29) + trial_grad7*(weak_mat_tmp13*weak_mat_tmp51 + weak_mat_tmp55) + trial_grad8*(weak_mat_tmp13*weak_mat_tmp63 + weak_mat_tmp67);
    const s_t material1 = trial_grad0*(weak_mat_tmp39*weak_mat_tmp70 + weak_mat_tmp5) + trial_grad1*(mu*(weak_mat_tmp35 + weak_mat_tmp82) + weak_mat_tmp70*weak_mat_tmp8) + trial_grad2*(weak_mat_tmp19*weak_mat_tmp70 + weak_mat_tmp75) + trial_grad3*(weak_mat_tmp26*weak_mat_tmp70 + weak_mat_tmp94) + trial_grad4*(weak_mat_tmp57*weak_mat_tmp70 + weak_mat_tmp77) + trial_grad5*(weak_mat_tmp43*weak_mat_tmp70 + weak_mat_tmp86) + trial_grad6*(weak_mat_tmp32*weak_mat_tmp70 + weak_mat_tmp88) + trial_grad7*(weak_mat_tmp51*weak_mat_tmp70 + weak_mat_tmp79) + trial_grad8*(weak_mat_tmp63*weak_mat_tmp70 + weak_mat_tmp92);
    const s_t material2 = trial_grad0*(weak_mat_tmp16 + weak_mat_tmp39*weak_mat_tmp97) + trial_grad1*(weak_mat_tmp75 + weak_mat_tmp8*weak_mat_tmp97) + trial_grad2*(mu*(weak_mat_tmp38 + weak_mat_tmp82 + s_t(2)) + weak_mat_tmp19*weak_mat_tmp97) + trial_grad3*(weak_mat_tmp101 + weak_mat_tmp26*weak_mat_tmp97) + trial_grad4*(weak_mat_tmp105 + weak_mat_tmp57*weak_mat_tmp97) + trial_grad5*(weak_mat_tmp43*weak_mat_tmp97 + weak_mat_tmp98) + trial_grad6*(weak_mat_tmp107 + weak_mat_tmp32*weak_mat_tmp97) + trial_grad7*(weak_mat_tmp103 + weak_mat_tmp51*weak_mat_tmp97) + trial_grad8*(weak_mat_tmp63*weak_mat_tmp97 + weak_mat_tmp99);
    const s_t material3 = trial_grad0*(weak_mat_tmp108*weak_mat_tmp39 + weak_mat_tmp24) + trial_grad1*(weak_mat_tmp108*weak_mat_tmp8 + weak_mat_tmp94) + trial_grad2*(weak_mat_tmp101 + weak_mat_tmp108*weak_mat_tmp19) + trial_grad3*(mu*(weak_mat_tmp117 + weak_mat_tmp119) + weak_mat_tmp108*weak_mat_tmp26) + trial_grad4*(weak_mat_tmp108*weak_mat_tmp57 + weak_mat_tmp110) + trial_grad5*(weak_mat_tmp108*weak_mat_tmp43 + weak_mat_tmp112) + trial_grad6*(weak_mat_tmp108*weak_mat_tmp32 + weak_mat_tmp115) + trial_grad7*(weak_mat_tmp108*weak_mat_tmp51 + weak_mat_tmp122) + trial_grad8*(weak_mat_tmp108*weak_mat_tmp63 + weak_mat_tmp125);
    const s_t material4 = trial_grad0*(weak_mat_tmp126*weak_mat_tmp39 + weak_mat_tmp62) + trial_grad1*(weak_mat_tmp126*weak_mat_tmp8 + weak_mat_tmp77) + trial_grad2*(weak_mat_tmp105 + weak_mat_tmp126*weak_mat_tmp19) + trial_grad3*(weak_mat_tmp110 + weak_mat_tmp126*weak_mat_tmp26) + trial_grad4*(mu*(weak_mat_tmp117 + weak_mat_tmp132) + weak_mat_tmp126*weak_mat_tmp57) + trial_grad5*(weak_mat_tmp126*weak_mat_tmp43 + weak_mat_tmp128) + trial_grad6*(weak_mat_tmp126*weak_mat_tmp32 + weak_mat_tmp133) + trial_grad7*(weak_mat_tmp126*weak_mat_tmp51 + weak_mat_tmp130) + trial_grad8*(weak_mat_tmp126*weak_mat_tmp63 + weak_mat_tmp136);
    const s_t material5 = trial_grad0*(weak_mat_tmp137*weak_mat_tmp39 + weak_mat_tmp50) + trial_grad1*(weak_mat_tmp137*weak_mat_tmp8 + weak_mat_tmp86) + trial_grad2*(weak_mat_tmp137*weak_mat_tmp19 + weak_mat_tmp98) + trial_grad3*(weak_mat_tmp112 + weak_mat_tmp137*weak_mat_tmp26) + trial_grad4*(weak_mat_tmp128 + weak_mat_tmp137*weak_mat_tmp57) + trial_grad5*(mu*(weak_mat_tmp119 + weak_mat_tmp132 + s_t(2)) + weak_mat_tmp137*weak_mat_tmp43) + trial_grad6*(weak_mat_tmp137*weak_mat_tmp32 + weak_mat_tmp139) + trial_grad7*(weak_mat_tmp137*weak_mat_tmp51 + weak_mat_tmp140) + trial_grad8*(weak_mat_tmp137*weak_mat_tmp63 + weak_mat_tmp138);
    const s_t material6 = trial_grad0*(weak_mat_tmp141*weak_mat_tmp39 + weak_mat_tmp29) + trial_grad1*(weak_mat_tmp141*weak_mat_tmp8 + weak_mat_tmp88) + trial_grad2*(weak_mat_tmp107 + weak_mat_tmp141*weak_mat_tmp19) + trial_grad3*(weak_mat_tmp115 + weak_mat_tmp141*weak_mat_tmp26) + trial_grad4*(weak_mat_tmp133 + weak_mat_tmp141*weak_mat_tmp57) + trial_grad5*(weak_mat_tmp139 + weak_mat_tmp141*weak_mat_tmp43) + trial_grad6*(mu*(weak_mat_tmp144 + weak_mat_tmp145) + weak_mat_tmp141*weak_mat_tmp32) + trial_grad7*(weak_mat_tmp141*weak_mat_tmp51 + weak_mat_tmp142) + trial_grad8*(weak_mat_tmp141*weak_mat_tmp63 + weak_mat_tmp143);
    const s_t material7 = trial_grad0*(weak_mat_tmp146*weak_mat_tmp39 + weak_mat_tmp55) + trial_grad1*(weak_mat_tmp146*weak_mat_tmp8 + weak_mat_tmp79) + trial_grad2*(weak_mat_tmp103 + weak_mat_tmp146*weak_mat_tmp19) + trial_grad3*(weak_mat_tmp122 + weak_mat_tmp146*weak_mat_tmp26) + trial_grad4*(weak_mat_tmp130 + weak_mat_tmp146*weak_mat_tmp57) + trial_grad5*(weak_mat_tmp140 + weak_mat_tmp146*weak_mat_tmp43) + trial_grad6*(weak_mat_tmp142 + weak_mat_tmp146*weak_mat_tmp32) + trial_grad7*(mu*(weak_mat_tmp144 + weak_mat_tmp148) + weak_mat_tmp146*weak_mat_tmp51) + trial_grad8*(weak_mat_tmp146*weak_mat_tmp63 + weak_mat_tmp147);
    const s_t material8 = trial_grad0*(weak_mat_tmp149*weak_mat_tmp39 + weak_mat_tmp67) + trial_grad1*(weak_mat_tmp149*weak_mat_tmp8 + weak_mat_tmp92) + trial_grad2*(weak_mat_tmp149*weak_mat_tmp19 + weak_mat_tmp99) + trial_grad3*(weak_mat_tmp125 + weak_mat_tmp149*weak_mat_tmp26) + trial_grad4*(weak_mat_tmp136 + weak_mat_tmp149*weak_mat_tmp57) + trial_grad5*(weak_mat_tmp138 + weak_mat_tmp149*weak_mat_tmp43) + trial_grad6*(weak_mat_tmp143 + weak_mat_tmp149*weak_mat_tmp32) + trial_grad7*(weak_mat_tmp147 + weak_mat_tmp149*weak_mat_tmp51) + trial_grad8*(mu*(weak_mat_tmp145 + weak_mat_tmp148 + s_t(2)) + weak_mat_tmp149*weak_mat_tmp63);
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
