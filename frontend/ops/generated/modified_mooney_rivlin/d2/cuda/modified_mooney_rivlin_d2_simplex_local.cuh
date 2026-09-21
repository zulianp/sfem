#ifndef MODIFIED_MOONEY_RIVLIN_D2_SIMPLEX_LOCAL_CUH
#define MODIFIED_MOONEY_RIVLIN_D2_SIMPLEX_LOCAL_CUH
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
static __host__ __device__ __forceinline__ void modified_mooney_rivlin_d2_simplex_objective_block(
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
      {
        gu_ref0_values[0] = s_t(0);
        grad_h_ref0_values[0] = s_t(0);
        gu_ref1_values[0] = s_t(0);
        grad_h_ref1_values[0] = s_t(0);
        gu_ref2_values[0] = s_t(0);
        grad_h_ref2_values[0] = s_t(0);
        gu_ref3_values[0] = s_t(0);
        grad_h_ref3_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        {
          gu_ref0_values[0] += u_streams[2 * shape][0] * grad_ref_x[q * NS + shape];
          grad_h_ref0_values[0] += h_streams[2 * shape][0] * grad_ref_x[q * NS + shape];
        }
        {
          gu_ref1_values[0] += u_streams[2 * shape][0] * grad_ref_y[q * NS + shape];
          grad_h_ref1_values[0] += h_streams[2 * shape][0] * grad_ref_y[q * NS + shape];
        }
        {
          gu_ref2_values[0] += u_streams[2 * shape + 1][0] * grad_ref_x[q * NS + shape];
          grad_h_ref2_values[0] += h_streams[2 * shape + 1][0] * grad_ref_x[q * NS + shape];
        }
        {
          gu_ref3_values[0] += u_streams[2 * shape + 1][0] * grad_ref_y[q * NS + shape];
          grad_h_ref3_values[0] += h_streams[2 * shape + 1][0] * grad_ref_y[q * NS + shape];
        }
      }
      s_t gu_base_v[4 * VS];
      s_t trial_grad_v[4 * VS];
      {
      const ptrdiff_t goff = q * geometry_stride + 0;
      const s_t adj_value0 = adj0[goff];
      const s_t adj_value1 = adj1[goff];
      const s_t adj_value2 = adj2[goff];
      const s_t adj_value3 = adj3[goff];
      const s_t det_value0 = det0[goff];
      const s_t gu_ref0 = gu_ref0_values[0];
      const s_t grad_h_ref0 = grad_h_ref0_values[0];
      const s_t gu_ref1 = gu_ref1_values[0];
      const s_t grad_h_ref1 = grad_h_ref1_values[0];
      const s_t gu_ref2 = gu_ref2_values[0];
      const s_t grad_h_ref2 = grad_h_ref2_values[0];
      const s_t gu_ref3 = gu_ref3_values[0];
      const s_t grad_h_ref3 = grad_h_ref3_values[0];
    const s_t idet = s_t(1) / det_value0;
    gu_base_v[0 * VS + 0] = (gu_ref0 * adj_value0 + gu_ref1 * adj_value2) * idet;
    trial_grad_v[0 * VS + 0] = (grad_h_ref0 * adj_value0 + grad_h_ref1 * adj_value2) * idet;
    gu_base_v[1 * VS + 0] = (gu_ref0 * adj_value1 + gu_ref1 * adj_value3) * idet;
    trial_grad_v[1 * VS + 0] = (grad_h_ref0 * adj_value1 + grad_h_ref1 * adj_value3) * idet;
    gu_base_v[2 * VS + 0] = (gu_ref2 * adj_value0 + gu_ref3 * adj_value2) * idet;
    trial_grad_v[2 * VS + 0] = (grad_h_ref2 * adj_value0 + grad_h_ref3 * adj_value2) * idet;
    gu_base_v[3 * VS + 0] = (gu_ref2 * adj_value1 + gu_ref3 * adj_value3) * idet;
    trial_grad_v[3 * VS + 0] = (grad_h_ref2 * adj_value1 + grad_h_ref3 * adj_value3) * idet;
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
    const s_t weak_obj_tmp0 = gu1*gu2;
    const s_t weak_obj_tmp1 = gu0 + s_t(1);
    const s_t weak_obj_tmp2 = gu3 + s_t(1);
    const s_t weak_obj_tmp3 = -weak_obj_tmp0 + weak_obj_tmp1*weak_obj_tmp2;
    const s_t weak_obj_tmp4 = pow_2(gu1) + pow_2(weak_obj_tmp2);
    const s_t weak_obj_tmp5 = pow_2(gu2) + pow_2(weak_obj_tmp1);
    const s_t weak_obj_tmp6 = weak_obj_tmp4 + weak_obj_tmp5;
    value[step * value_stride + 0] += qw * det_value0 * (c1*(s_t(-3) + (weak_obj_tmp6 + s_t(1))/pow(weak_obj_tmp3, (s_t(2) / s_t(3)))) + c2*(s_t(-3) + (-(s_t(1) / s_t(2))*pow_2(weak_obj_tmp4) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp5) + ((s_t(1) / s_t(2)))*pow_2(weak_obj_tmp6) + weak_obj_tmp6 - pow_2(gu1*weak_obj_tmp1 + gu2*weak_obj_tmp2))/pow(weak_obj_tmp3, (s_t(4) / s_t(3)))) + ((s_t(1) / s_t(2)))*kappa*pow_2(sfem_log1p(gu0*gu3 + gu0 + gu3 - weak_obj_tmp0)));
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void modified_mooney_rivlin_d2_simplex_tri3_objective_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR q_weight,
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
    { const int q = 0;  // constant-P1 simplex
      const s_t qw = q_weight[q];
      s_t gu_base_v[4 * VS];
      s_t trial_grad_v[4 * VS];
      {
      const ptrdiff_t goff = q * geometry_stride + 0;
      const s_t adj_value0 = adj0[goff];
      const s_t adj_value1 = adj1[goff];
      const s_t adj_value2 = adj2[goff];
      const s_t adj_value3 = adj3[goff];
      const s_t det_value0 = det0[goff];
      const s_t gu_ref0 = -(u_streams[0][0]) + u_streams[2][0];
      const s_t grad_h_ref0 = -(h_streams[0][0]) + h_streams[2][0];
      const s_t gu_ref1 = -(u_streams[0][0]) + u_streams[4][0];
      const s_t grad_h_ref1 = -(h_streams[0][0]) + h_streams[4][0];
      const s_t gu_ref2 = -(u_streams[1][0]) + u_streams[3][0];
      const s_t grad_h_ref2 = -(h_streams[1][0]) + h_streams[3][0];
      const s_t gu_ref3 = -(u_streams[1][0]) + u_streams[5][0];
      const s_t grad_h_ref3 = -(h_streams[1][0]) + h_streams[5][0];
      const s_t idet = s_t(1) / det_value0;
      gu_base_v[0 * VS + 0] = (gu_ref0 * adj_value0 + gu_ref1 * adj_value2) * idet;
      trial_grad_v[0 * VS + 0] = (grad_h_ref0 * adj_value0 + grad_h_ref1 * adj_value2) * idet;
      gu_base_v[1 * VS + 0] = (gu_ref0 * adj_value1 + gu_ref1 * adj_value3) * idet;
      trial_grad_v[1 * VS + 0] = (grad_h_ref0 * adj_value1 + grad_h_ref1 * adj_value3) * idet;
      gu_base_v[2 * VS + 0] = (gu_ref2 * adj_value0 + gu_ref3 * adj_value2) * idet;
      trial_grad_v[2 * VS + 0] = (grad_h_ref2 * adj_value0 + grad_h_ref3 * adj_value2) * idet;
      gu_base_v[3 * VS + 0] = (gu_ref2 * adj_value1 + gu_ref3 * adj_value3) * idet;
      trial_grad_v[3 * VS + 0] = (grad_h_ref2 * adj_value1 + grad_h_ref3 * adj_value3) * idet;
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
    const s_t weak_obj_tmp0 = gu1*gu2;
    const s_t weak_obj_tmp1 = gu0 + s_t(1);
    const s_t weak_obj_tmp2 = gu3 + s_t(1);
    const s_t weak_obj_tmp3 = -weak_obj_tmp0 + weak_obj_tmp1*weak_obj_tmp2;
    const s_t weak_obj_tmp4 = pow_2(gu1) + pow_2(weak_obj_tmp2);
    const s_t weak_obj_tmp5 = pow_2(gu2) + pow_2(weak_obj_tmp1);
    const s_t weak_obj_tmp6 = weak_obj_tmp4 + weak_obj_tmp5;
    value[step * value_stride + 0] += qw * det_value0 * (c1*(s_t(-3) + (weak_obj_tmp6 + s_t(1))/pow(weak_obj_tmp3, (s_t(2) / s_t(3)))) + c2*(s_t(-3) + (-(s_t(1) / s_t(2))*pow_2(weak_obj_tmp4) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp5) + ((s_t(1) / s_t(2)))*pow_2(weak_obj_tmp6) + weak_obj_tmp6 - pow_2(gu1*weak_obj_tmp1 + gu2*weak_obj_tmp2))/pow(weak_obj_tmp3, (s_t(4) / s_t(3)))) + ((s_t(1) / s_t(2)))*kappa*pow_2(sfem_log1p(gu0*gu3 + gu0 + gu3 - weak_obj_tmp0)));
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void modified_mooney_rivlin_d2_simplex_gradient_block(
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
        const s_t c1,
        const s_t c2,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 2],
        s_t *const RSTR out_streams[NS * 2]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
    for (int q = 0; q < NQ; ++q) {
      const s_t qw = q_weight[q];
      s_t gu_ref0_values[VS];
      s_t gu_ref1_values[VS];
      s_t gu_ref2_values[VS];
      s_t gu_ref3_values[VS];
      s_t loperand0_values[VS];
      s_t loperand1_values[VS];
      s_t loperand2_values[VS];
      s_t loperand3_values[VS];
      {
        gu_ref0_values[0] = s_t(0);
        gu_ref1_values[0] = s_t(0);
        gu_ref2_values[0] = s_t(0);
        gu_ref3_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        {
          gu_ref0_values[0] += u_streams[2 * shape][0] * grad_ref_x[q * NS + shape];
        }
        {
          gu_ref1_values[0] += u_streams[2 * shape][0] * grad_ref_y[q * NS + shape];
        }
        {
          gu_ref2_values[0] += u_streams[2 * shape + 1][0] * grad_ref_x[q * NS + shape];
        }
        {
          gu_ref3_values[0] += u_streams[2 * shape + 1][0] * grad_ref_y[q * NS + shape];
        }
      }
      {
      const ptrdiff_t goff = q * geometry_stride + 0;
      const s_t adj_value0 = adj0[goff];
      const s_t adj_value1 = adj1[goff];
      const s_t adj_value2 = adj2[goff];
      const s_t adj_value3 = adj3[goff];
      const s_t det_value0 = det0[goff];
      const s_t gu_ref0 = gu_ref0_values[0];
      const s_t gu_ref1 = gu_ref1_values[0];
      const s_t gu_ref2 = gu_ref2_values[0];
      const s_t gu_ref3 = gu_ref3_values[0];
    const s_t idet = s_t(1) / det_value0;
    const s_t gu0 = (gu_ref0 * adj_value0 + gu_ref1 * adj_value2) * idet;
    const s_t gu1 = (gu_ref0 * adj_value1 + gu_ref1 * adj_value3) * idet;
    const s_t gu2 = (gu_ref2 * adj_value0 + gu_ref3 * adj_value2) * idet;
    const s_t gu3 = (gu_ref2 * adj_value1 + gu_ref3 * adj_value3) * idet;
    const s_t weak_mat_tmp0 = gu3 + s_t(1);
    const s_t weak_mat_tmp1 = gu1*gu2;
    const s_t weak_mat_tmp2 = gu0 + s_t(1);
    const s_t weak_mat_tmp3 = weak_mat_tmp0*weak_mat_tmp2 - weak_mat_tmp1;
    const s_t weak_mat_tmp4 = kappa*sfem_log1p(gu0*gu3 + gu0 + gu3 - weak_mat_tmp1)/weak_mat_tmp3;
    const s_t weak_mat_tmp5 = pow(weak_mat_tmp3, (s_t(-2) / s_t(3)));
    const s_t weak_mat_tmp6 = s_t(2)*weak_mat_tmp2;
    const s_t weak_mat_tmp7 = pow_2(gu2) + pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp8 = pow_2(gu1) + pow_2(weak_mat_tmp0);
    const s_t weak_mat_tmp9 = weak_mat_tmp7 + weak_mat_tmp8;
    const s_t weak_mat_tmp10 = ((s_t(2) / s_t(3)))*(weak_mat_tmp9 + s_t(1))/pow(weak_mat_tmp3, (s_t(5) / s_t(3)));
    const s_t weak_mat_tmp11 = pow(weak_mat_tmp3, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp12 = gu1*weak_mat_tmp2 + gu2*weak_mat_tmp0;
    const s_t weak_mat_tmp13 = s_t(2)*gu1;
    const s_t weak_mat_tmp14 = ((s_t(4) / s_t(3)))*(-pow_2(weak_mat_tmp12) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp7) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp8) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp9) + weak_mat_tmp9)/pow(weak_mat_tmp3, (s_t(7) / s_t(3)));
    const s_t weak_mat_tmp15 = s_t(2)*gu2;
    const s_t weak_mat_tmp16 = s_t(2)*weak_mat_tmp0;
    const s_t material0 = c1*(-weak_mat_tmp0*weak_mat_tmp10 + weak_mat_tmp5*weak_mat_tmp6) + c2*(-weak_mat_tmp0*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu0 - weak_mat_tmp12*weak_mat_tmp13 - weak_mat_tmp6*weak_mat_tmp7 + weak_mat_tmp6*weak_mat_tmp9 + s_t(2))) + weak_mat_tmp0*weak_mat_tmp4;
    const s_t material1 = c1*(gu2*weak_mat_tmp10 + weak_mat_tmp13*weak_mat_tmp5) + c2*(gu2*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu1*weak_mat_tmp9 + s_t(2)*gu1 - weak_mat_tmp12*weak_mat_tmp6 - weak_mat_tmp13*weak_mat_tmp8)) - gu2*weak_mat_tmp4;
    const s_t material2 = c1*(gu1*weak_mat_tmp10 + weak_mat_tmp15*weak_mat_tmp5) + c2*(gu1*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu2*weak_mat_tmp9 + s_t(2)*gu2 - weak_mat_tmp12*weak_mat_tmp16 - weak_mat_tmp15*weak_mat_tmp7)) - gu1*weak_mat_tmp4;
    const s_t material3 = c1*(s_t(2)*weak_mat_tmp0*weak_mat_tmp5 - weak_mat_tmp10*weak_mat_tmp2) + c2*(weak_mat_tmp11*(s_t(2)*gu3 - weak_mat_tmp12*weak_mat_tmp15 - weak_mat_tmp16*weak_mat_tmp8 + weak_mat_tmp16*weak_mat_tmp9 + s_t(2)) - weak_mat_tmp14*weak_mat_tmp2) + weak_mat_tmp2*weak_mat_tmp4;
    const s_t loperand0 = qw * (material0 * adj_value0 + material1 * adj_value1);
    const s_t loperand1 = qw * (material0 * adj_value2 + material1 * adj_value3);
    const s_t loperand2 = qw * (material2 * adj_value0 + material3 * adj_value1);
    const s_t loperand3 = qw * (material2 * adj_value2 + material3 * adj_value3);
      loperand0_values[0] = loperand0;
      loperand1_values[0] = loperand1;
      loperand2_values[0] = loperand2;
      loperand3_values[0] = loperand3;
      }
      for (int shape = 0; shape < NS; ++shape) {
        {
          out_streams[2 * shape][0] += loperand0_values[0] * grad_ref_x[q * NS + shape] + loperand1_values[0] * grad_ref_y[q * NS + shape];
        }
        {
          out_streams[2 * shape + 1][0] += loperand2_values[0] * grad_ref_x[q * NS + shape] + loperand3_values[0] * grad_ref_y[q * NS + shape];
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void modified_mooney_rivlin_d2_simplex_tri3_gradient_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR q_weight,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 2],
        s_t *const RSTR out_streams[NS * 2]
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
      const s_t det_value0 = det0[goff];
      const s_t gu_ref0 = -(u_streams[0][0]) + u_streams[2][0];
      const s_t gu_ref1 = -(u_streams[0][0]) + u_streams[4][0];
      const s_t gu_ref2 = -(u_streams[1][0]) + u_streams[3][0];
      const s_t gu_ref3 = -(u_streams[1][0]) + u_streams[5][0];
      const s_t idet = s_t(1) / det_value0;
      const s_t gu0 = (gu_ref0 * adj_value0 + gu_ref1 * adj_value2) * idet;
      const s_t gu1 = (gu_ref0 * adj_value1 + gu_ref1 * adj_value3) * idet;
      const s_t gu2 = (gu_ref2 * adj_value0 + gu_ref3 * adj_value2) * idet;
      const s_t gu3 = (gu_ref2 * adj_value1 + gu_ref3 * adj_value3) * idet;
    const s_t weak_mat_tmp0 = gu3 + s_t(1);
    const s_t weak_mat_tmp1 = gu1*gu2;
    const s_t weak_mat_tmp2 = gu0 + s_t(1);
    const s_t weak_mat_tmp3 = weak_mat_tmp0*weak_mat_tmp2 - weak_mat_tmp1;
    const s_t weak_mat_tmp4 = kappa*sfem_log1p(gu0*gu3 + gu0 + gu3 - weak_mat_tmp1)/weak_mat_tmp3;
    const s_t weak_mat_tmp5 = pow(weak_mat_tmp3, (s_t(-2) / s_t(3)));
    const s_t weak_mat_tmp6 = s_t(2)*weak_mat_tmp2;
    const s_t weak_mat_tmp7 = pow_2(gu2) + pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp8 = pow_2(gu1) + pow_2(weak_mat_tmp0);
    const s_t weak_mat_tmp9 = weak_mat_tmp7 + weak_mat_tmp8;
    const s_t weak_mat_tmp10 = ((s_t(2) / s_t(3)))*(weak_mat_tmp9 + s_t(1))/pow(weak_mat_tmp3, (s_t(5) / s_t(3)));
    const s_t weak_mat_tmp11 = pow(weak_mat_tmp3, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp12 = gu1*weak_mat_tmp2 + gu2*weak_mat_tmp0;
    const s_t weak_mat_tmp13 = s_t(2)*gu1;
    const s_t weak_mat_tmp14 = ((s_t(4) / s_t(3)))*(-pow_2(weak_mat_tmp12) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp7) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp8) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp9) + weak_mat_tmp9)/pow(weak_mat_tmp3, (s_t(7) / s_t(3)));
    const s_t weak_mat_tmp15 = s_t(2)*gu2;
    const s_t weak_mat_tmp16 = s_t(2)*weak_mat_tmp0;
    const s_t material0 = c1*(-weak_mat_tmp0*weak_mat_tmp10 + weak_mat_tmp5*weak_mat_tmp6) + c2*(-weak_mat_tmp0*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu0 - weak_mat_tmp12*weak_mat_tmp13 - weak_mat_tmp6*weak_mat_tmp7 + weak_mat_tmp6*weak_mat_tmp9 + s_t(2))) + weak_mat_tmp0*weak_mat_tmp4;
    const s_t material1 = c1*(gu2*weak_mat_tmp10 + weak_mat_tmp13*weak_mat_tmp5) + c2*(gu2*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu1*weak_mat_tmp9 + s_t(2)*gu1 - weak_mat_tmp12*weak_mat_tmp6 - weak_mat_tmp13*weak_mat_tmp8)) - gu2*weak_mat_tmp4;
    const s_t material2 = c1*(gu1*weak_mat_tmp10 + weak_mat_tmp15*weak_mat_tmp5) + c2*(gu1*weak_mat_tmp14 + weak_mat_tmp11*(s_t(2)*gu2*weak_mat_tmp9 + s_t(2)*gu2 - weak_mat_tmp12*weak_mat_tmp16 - weak_mat_tmp15*weak_mat_tmp7)) - gu1*weak_mat_tmp4;
    const s_t material3 = c1*(s_t(2)*weak_mat_tmp0*weak_mat_tmp5 - weak_mat_tmp10*weak_mat_tmp2) + c2*(weak_mat_tmp11*(s_t(2)*gu3 - weak_mat_tmp12*weak_mat_tmp15 - weak_mat_tmp16*weak_mat_tmp8 + weak_mat_tmp16*weak_mat_tmp9 + s_t(2)) - weak_mat_tmp14*weak_mat_tmp2) + weak_mat_tmp2*weak_mat_tmp4;
    const s_t loperand0 = qw * (material0 * adj_value0 + material1 * adj_value1);
    const s_t loperand1 = qw * (material0 * adj_value2 + material1 * adj_value3);
    const s_t loperand2 = qw * (material2 * adj_value0 + material3 * adj_value1);
    const s_t loperand3 = qw * (material2 * adj_value2 + material3 * adj_value3);
      out_streams[0][0] += -(loperand0) - loperand1;
      out_streams[1][0] += -(loperand2) - loperand3;
      out_streams[2][0] += loperand0;
      out_streams[3][0] += loperand2;
      out_streams[4][0] += loperand1;
      out_streams[5][0] += loperand3;
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void modified_mooney_rivlin_d2_simplex_apply_block(
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
        const s_t c1,
        const s_t c2,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 2],
        const s_t *const RSTR h_streams[NS * 2],
        s_t *const RSTR out_streams[NS * 2]
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
      s_t loperand0_values[VS];
      s_t loperand1_values[VS];
      s_t loperand2_values[VS];
      s_t loperand3_values[VS];
      {
        gu_ref0_values[0] = s_t(0);
        grad_h_ref0_values[0] = s_t(0);
        gu_ref1_values[0] = s_t(0);
        grad_h_ref1_values[0] = s_t(0);
        gu_ref2_values[0] = s_t(0);
        grad_h_ref2_values[0] = s_t(0);
        gu_ref3_values[0] = s_t(0);
        grad_h_ref3_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        {
          gu_ref0_values[0] += u_streams[2 * shape][0] * grad_ref_x[q * NS + shape];
          grad_h_ref0_values[0] += h_streams[2 * shape][0] * grad_ref_x[q * NS + shape];
        }
        {
          gu_ref1_values[0] += u_streams[2 * shape][0] * grad_ref_y[q * NS + shape];
          grad_h_ref1_values[0] += h_streams[2 * shape][0] * grad_ref_y[q * NS + shape];
        }
        {
          gu_ref2_values[0] += u_streams[2 * shape + 1][0] * grad_ref_x[q * NS + shape];
          grad_h_ref2_values[0] += h_streams[2 * shape + 1][0] * grad_ref_x[q * NS + shape];
        }
        {
          gu_ref3_values[0] += u_streams[2 * shape + 1][0] * grad_ref_y[q * NS + shape];
          grad_h_ref3_values[0] += h_streams[2 * shape + 1][0] * grad_ref_y[q * NS + shape];
        }
      }
      {
      const ptrdiff_t goff = q * geometry_stride + 0;
      const s_t adj_value0 = adj0[goff];
      const s_t adj_value1 = adj1[goff];
      const s_t adj_value2 = adj2[goff];
      const s_t adj_value3 = adj3[goff];
      const s_t det_value0 = det0[goff];
      const s_t gu_ref0 = gu_ref0_values[0];
      const s_t grad_h_ref0 = grad_h_ref0_values[0];
      const s_t gu_ref1 = gu_ref1_values[0];
      const s_t grad_h_ref1 = grad_h_ref1_values[0];
      const s_t gu_ref2 = gu_ref2_values[0];
      const s_t grad_h_ref2 = grad_h_ref2_values[0];
      const s_t gu_ref3 = gu_ref3_values[0];
      const s_t grad_h_ref3 = grad_h_ref3_values[0];
    const s_t idet = s_t(1) / det_value0;
    const s_t gu0 = (gu_ref0 * adj_value0 + gu_ref1 * adj_value2) * idet;
    const s_t trial_grad0 = (grad_h_ref0 * adj_value0 + grad_h_ref1 * adj_value2) * idet;
    const s_t gu1 = (gu_ref0 * adj_value1 + gu_ref1 * adj_value3) * idet;
    const s_t trial_grad1 = (grad_h_ref0 * adj_value1 + grad_h_ref1 * adj_value3) * idet;
    const s_t gu2 = (gu_ref2 * adj_value0 + gu_ref3 * adj_value2) * idet;
    const s_t trial_grad2 = (grad_h_ref2 * adj_value0 + grad_h_ref3 * adj_value2) * idet;
    const s_t gu3 = (gu_ref2 * adj_value1 + gu_ref3 * adj_value3) * idet;
    const s_t trial_grad3 = (grad_h_ref2 * adj_value1 + grad_h_ref3 * adj_value3) * idet;
    const s_t weak_mat_tmp0 = gu3 + s_t(1);
    const s_t weak_mat_tmp1 = pow_2(weak_mat_tmp0);
    const s_t weak_mat_tmp2 = gu1*gu2;
    const s_t weak_mat_tmp3 = gu0 + s_t(1);
    const s_t weak_mat_tmp4 = weak_mat_tmp0*weak_mat_tmp3 - weak_mat_tmp2;
    const s_t weak_mat_tmp5 = kappa/pow_2(weak_mat_tmp4);
    const s_t weak_mat_tmp6 = weak_mat_tmp1*weak_mat_tmp5;
    const s_t weak_mat_tmp7 = sfem_log1p(gu0*gu3 + gu0 + gu3 - weak_mat_tmp2);
    const s_t weak_mat_tmp8 = pow_2(gu2);
    const s_t weak_mat_tmp9 = pow_2(weak_mat_tmp3);
    const s_t weak_mat_tmp10 = weak_mat_tmp8 + weak_mat_tmp9;
    const s_t weak_mat_tmp11 = pow_2(gu1);
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
    const s_t weak_mat_tmp23 = gu1*weak_mat_tmp3;
    const s_t weak_mat_tmp24 = gu2*weak_mat_tmp0;
    const s_t weak_mat_tmp25 = weak_mat_tmp23 + weak_mat_tmp24;
    const s_t weak_mat_tmp26 = s_t(2)*gu1;
    const s_t weak_mat_tmp27 = s_t(2)*weak_mat_tmp3;
    const s_t weak_mat_tmp28 = s_t(2)*gu0 - weak_mat_tmp10*weak_mat_tmp27 + weak_mat_tmp13*weak_mat_tmp27 - weak_mat_tmp25*weak_mat_tmp26 + s_t(2);
    const s_t weak_mat_tmp29 = pow(weak_mat_tmp4, (s_t(-7) / s_t(3)));
    const s_t weak_mat_tmp30 = ((s_t(8) / s_t(3)))*weak_mat_tmp29;
    const s_t weak_mat_tmp31 = -(s_t(1) / s_t(2))*pow_2(weak_mat_tmp10) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp12) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp13) + weak_mat_tmp13 - pow_2(weak_mat_tmp25);
    const s_t weak_mat_tmp32 = pow(weak_mat_tmp4, (s_t(-10) / s_t(3)));
    const s_t weak_mat_tmp33 = ((s_t(28) / s_t(9)))*weak_mat_tmp31*weak_mat_tmp32;
    const s_t weak_mat_tmp34 = weak_mat_tmp24*weak_mat_tmp5;
    const s_t weak_mat_tmp35 = ((s_t(4) / s_t(3)))*weak_mat_tmp19;
    const s_t weak_mat_tmp36 = gu1*weak_mat_tmp0;
    const s_t weak_mat_tmp37 = weak_mat_tmp35*weak_mat_tmp36;
    const s_t weak_mat_tmp38 = gu2*weak_mat_tmp3;
    const s_t weak_mat_tmp39 = weak_mat_tmp35*weak_mat_tmp38;
    const s_t weak_mat_tmp40 = s_t(2)*weak_mat_tmp22;
    const s_t weak_mat_tmp41 = s_t(2)*gu1*weak_mat_tmp13 + s_t(2)*gu1 - weak_mat_tmp12*weak_mat_tmp26 - weak_mat_tmp25*weak_mat_tmp27;
    const s_t weak_mat_tmp42 = ((s_t(4) / s_t(3)))*weak_mat_tmp29;
    const s_t weak_mat_tmp43 = weak_mat_tmp0*weak_mat_tmp42;
    const s_t weak_mat_tmp44 = c1*(-weak_mat_tmp16*weak_mat_tmp24 - weak_mat_tmp37 + weak_mat_tmp39) + c2*(((s_t(4) / s_t(3)))*gu2*weak_mat_tmp28*weak_mat_tmp29 - weak_mat_tmp24*weak_mat_tmp33 - weak_mat_tmp24*weak_mat_tmp40 - weak_mat_tmp41*weak_mat_tmp43) + weak_mat_tmp34*weak_mat_tmp7 - weak_mat_tmp34;
    const s_t weak_mat_tmp45 = weak_mat_tmp36*weak_mat_tmp5;
    const s_t weak_mat_tmp46 = weak_mat_tmp23*weak_mat_tmp35;
    const s_t weak_mat_tmp47 = weak_mat_tmp24*weak_mat_tmp35;
    const s_t weak_mat_tmp48 = s_t(2)*gu2;
    const s_t weak_mat_tmp49 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp50 = s_t(2)*gu2*weak_mat_tmp13 + s_t(2)*gu2 - weak_mat_tmp10*weak_mat_tmp48 - weak_mat_tmp25*weak_mat_tmp49;
    const s_t weak_mat_tmp51 = c1*(-weak_mat_tmp16*weak_mat_tmp36 + weak_mat_tmp46 - weak_mat_tmp47) + c2*(((s_t(4) / s_t(3)))*gu1*weak_mat_tmp28*weak_mat_tmp29 - weak_mat_tmp0*weak_mat_tmp22*weak_mat_tmp26 - weak_mat_tmp33*weak_mat_tmp36 - weak_mat_tmp43*weak_mat_tmp50) + weak_mat_tmp45*weak_mat_tmp7 - weak_mat_tmp45;
    const s_t weak_mat_tmp52 = weak_mat_tmp18*weak_mat_tmp5;
    const s_t weak_mat_tmp53 = kappa*weak_mat_tmp7/weak_mat_tmp4;
    const s_t weak_mat_tmp54 = ((s_t(2) / s_t(3)))*weak_mat_tmp14*weak_mat_tmp19;
    const s_t weak_mat_tmp55 = s_t(2)*gu3 - weak_mat_tmp12*weak_mat_tmp49 + weak_mat_tmp13*weak_mat_tmp49 - weak_mat_tmp25*weak_mat_tmp48 + s_t(2);
    const s_t weak_mat_tmp56 = weak_mat_tmp31*weak_mat_tmp42;
    const s_t weak_mat_tmp57 = c1*(((s_t(10) / s_t(9)))*weak_mat_tmp0*weak_mat_tmp14*weak_mat_tmp15*weak_mat_tmp3 - weak_mat_tmp1*weak_mat_tmp35 - weak_mat_tmp35*weak_mat_tmp9 - weak_mat_tmp54) + c2*(((s_t(28) / s_t(9)))*weak_mat_tmp0*weak_mat_tmp3*weak_mat_tmp31*weak_mat_tmp32 + weak_mat_tmp22*(s_t(4)*weak_mat_tmp0*weak_mat_tmp3 - s_t(2)*weak_mat_tmp2) - weak_mat_tmp28*weak_mat_tmp3*weak_mat_tmp42 - weak_mat_tmp43*weak_mat_tmp55 - weak_mat_tmp56) - weak_mat_tmp52*weak_mat_tmp7 + weak_mat_tmp52 + weak_mat_tmp53;
    const s_t weak_mat_tmp58 = weak_mat_tmp5*weak_mat_tmp8;
    const s_t weak_mat_tmp59 = weak_mat_tmp17 + weak_mat_tmp2*weak_mat_tmp20;
    const s_t weak_mat_tmp60 = weak_mat_tmp38*weak_mat_tmp5;
    const s_t weak_mat_tmp61 = weak_mat_tmp41*weak_mat_tmp42;
    const s_t weak_mat_tmp62 = c1*(-weak_mat_tmp16*weak_mat_tmp38 - weak_mat_tmp46 + weak_mat_tmp47) + c2*(((s_t(4) / s_t(3)))*gu2*weak_mat_tmp29*weak_mat_tmp55 - weak_mat_tmp22*weak_mat_tmp3*weak_mat_tmp48 - weak_mat_tmp3*weak_mat_tmp61 - weak_mat_tmp33*weak_mat_tmp38) + weak_mat_tmp60*weak_mat_tmp7 - weak_mat_tmp60;
    const s_t weak_mat_tmp63 = weak_mat_tmp2*weak_mat_tmp5;
    const s_t weak_mat_tmp64 = c1*(weak_mat_tmp11*weak_mat_tmp35 + weak_mat_tmp16*weak_mat_tmp2 + weak_mat_tmp35*weak_mat_tmp8 + weak_mat_tmp54) + c2*(gu1*weak_mat_tmp61 + gu2*weak_mat_tmp42*weak_mat_tmp50 + weak_mat_tmp2*weak_mat_tmp33 + weak_mat_tmp22*(-s_t(2)*weak_mat_tmp18 + s_t(4)*weak_mat_tmp2) + weak_mat_tmp56) - weak_mat_tmp53 - weak_mat_tmp63*weak_mat_tmp7 + weak_mat_tmp63;
    const s_t weak_mat_tmp65 = weak_mat_tmp11*weak_mat_tmp5;
    const s_t weak_mat_tmp66 = weak_mat_tmp23*weak_mat_tmp5;
    const s_t weak_mat_tmp67 = c1*(-weak_mat_tmp16*weak_mat_tmp23 + weak_mat_tmp37 - weak_mat_tmp39) + c2*(((s_t(4) / s_t(3)))*gu1*weak_mat_tmp29*weak_mat_tmp55 - weak_mat_tmp23*weak_mat_tmp33 - weak_mat_tmp23*weak_mat_tmp40 - weak_mat_tmp3*weak_mat_tmp42*weak_mat_tmp50) + weak_mat_tmp66*weak_mat_tmp7 - weak_mat_tmp66;
    const s_t weak_mat_tmp68 = weak_mat_tmp5*weak_mat_tmp9;
    const s_t material0 = trial_grad0*(c1*(weak_mat_tmp1*weak_mat_tmp16 + weak_mat_tmp21) + c2*(-weak_mat_tmp0*weak_mat_tmp28*weak_mat_tmp30 + weak_mat_tmp1*weak_mat_tmp33 + weak_mat_tmp22*(s_t(2)*weak_mat_tmp1 + s_t(2))) - weak_mat_tmp6*weak_mat_tmp7 + weak_mat_tmp6) + trial_grad1*weak_mat_tmp44 + trial_grad2*weak_mat_tmp51 + trial_grad3*weak_mat_tmp57;
    const s_t material1 = trial_grad0*weak_mat_tmp44 + trial_grad1*(c1*(weak_mat_tmp16*weak_mat_tmp8 + weak_mat_tmp59) + c2*(gu2*weak_mat_tmp30*weak_mat_tmp41 + weak_mat_tmp22*(s_t(2)*weak_mat_tmp8 + s_t(2)) + weak_mat_tmp33*weak_mat_tmp8) - weak_mat_tmp58*weak_mat_tmp7 + weak_mat_tmp58) + trial_grad2*weak_mat_tmp64 + trial_grad3*weak_mat_tmp62;
    const s_t material2 = trial_grad0*weak_mat_tmp51 + trial_grad1*weak_mat_tmp64 + trial_grad2*(c1*(weak_mat_tmp11*weak_mat_tmp16 + weak_mat_tmp59) + c2*(gu1*weak_mat_tmp30*weak_mat_tmp50 + weak_mat_tmp11*weak_mat_tmp33 + weak_mat_tmp22*(s_t(2)*weak_mat_tmp11 + s_t(2))) - weak_mat_tmp65*weak_mat_tmp7 + weak_mat_tmp65) + trial_grad3*weak_mat_tmp67;
    const s_t material3 = trial_grad0*weak_mat_tmp57 + trial_grad1*weak_mat_tmp62 + trial_grad2*weak_mat_tmp67 + trial_grad3*(c1*(weak_mat_tmp16*weak_mat_tmp9 + weak_mat_tmp21) + c2*(weak_mat_tmp22*(s_t(2)*weak_mat_tmp9 + s_t(2)) - weak_mat_tmp3*weak_mat_tmp30*weak_mat_tmp55 + weak_mat_tmp33*weak_mat_tmp9) - weak_mat_tmp68*weak_mat_tmp7 + weak_mat_tmp68);
    const s_t loperand0 = qw * (material0 * adj_value0 + material1 * adj_value1);
    const s_t loperand1 = qw * (material0 * adj_value2 + material1 * adj_value3);
    const s_t loperand2 = qw * (material2 * adj_value0 + material3 * adj_value1);
    const s_t loperand3 = qw * (material2 * adj_value2 + material3 * adj_value3);
      loperand0_values[0] = loperand0;
      loperand1_values[0] = loperand1;
      loperand2_values[0] = loperand2;
      loperand3_values[0] = loperand3;
      }
      for (int shape = 0; shape < NS; ++shape) {
        {
          out_streams[2 * shape][0] += loperand0_values[0] * grad_ref_x[q * NS + shape] + loperand1_values[0] * grad_ref_y[q * NS + shape];
        }
        {
          out_streams[2 * shape + 1][0] += loperand2_values[0] * grad_ref_x[q * NS + shape] + loperand3_values[0] * grad_ref_y[q * NS + shape];
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void modified_mooney_rivlin_d2_simplex_tri3_apply_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR q_weight,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 2],
        const s_t *const RSTR h_streams[NS * 2],
        s_t *const RSTR out_streams[NS * 2]
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
      const s_t det_value0 = det0[goff];
      const s_t gu_ref0 = -(u_streams[0][0]) + u_streams[2][0];
      const s_t grad_h_ref0 = -(h_streams[0][0]) + h_streams[2][0];
      const s_t gu_ref1 = -(u_streams[0][0]) + u_streams[4][0];
      const s_t grad_h_ref1 = -(h_streams[0][0]) + h_streams[4][0];
      const s_t gu_ref2 = -(u_streams[1][0]) + u_streams[3][0];
      const s_t grad_h_ref2 = -(h_streams[1][0]) + h_streams[3][0];
      const s_t gu_ref3 = -(u_streams[1][0]) + u_streams[5][0];
      const s_t grad_h_ref3 = -(h_streams[1][0]) + h_streams[5][0];
      const s_t idet = s_t(1) / det_value0;
      const s_t gu0 = (gu_ref0 * adj_value0 + gu_ref1 * adj_value2) * idet;
      const s_t trial_grad0 = (grad_h_ref0 * adj_value0 + grad_h_ref1 * adj_value2) * idet;
      const s_t gu1 = (gu_ref0 * adj_value1 + gu_ref1 * adj_value3) * idet;
      const s_t trial_grad1 = (grad_h_ref0 * adj_value1 + grad_h_ref1 * adj_value3) * idet;
      const s_t gu2 = (gu_ref2 * adj_value0 + gu_ref3 * adj_value2) * idet;
      const s_t trial_grad2 = (grad_h_ref2 * adj_value0 + grad_h_ref3 * adj_value2) * idet;
      const s_t gu3 = (gu_ref2 * adj_value1 + gu_ref3 * adj_value3) * idet;
      const s_t trial_grad3 = (grad_h_ref2 * adj_value1 + grad_h_ref3 * adj_value3) * idet;
    const s_t weak_mat_tmp0 = gu3 + s_t(1);
    const s_t weak_mat_tmp1 = pow_2(weak_mat_tmp0);
    const s_t weak_mat_tmp2 = gu1*gu2;
    const s_t weak_mat_tmp3 = gu0 + s_t(1);
    const s_t weak_mat_tmp4 = weak_mat_tmp0*weak_mat_tmp3 - weak_mat_tmp2;
    const s_t weak_mat_tmp5 = kappa/pow_2(weak_mat_tmp4);
    const s_t weak_mat_tmp6 = weak_mat_tmp1*weak_mat_tmp5;
    const s_t weak_mat_tmp7 = sfem_log1p(gu0*gu3 + gu0 + gu3 - weak_mat_tmp2);
    const s_t weak_mat_tmp8 = pow_2(gu2);
    const s_t weak_mat_tmp9 = pow_2(weak_mat_tmp3);
    const s_t weak_mat_tmp10 = weak_mat_tmp8 + weak_mat_tmp9;
    const s_t weak_mat_tmp11 = pow_2(gu1);
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
    const s_t weak_mat_tmp23 = gu1*weak_mat_tmp3;
    const s_t weak_mat_tmp24 = gu2*weak_mat_tmp0;
    const s_t weak_mat_tmp25 = weak_mat_tmp23 + weak_mat_tmp24;
    const s_t weak_mat_tmp26 = s_t(2)*gu1;
    const s_t weak_mat_tmp27 = s_t(2)*weak_mat_tmp3;
    const s_t weak_mat_tmp28 = s_t(2)*gu0 - weak_mat_tmp10*weak_mat_tmp27 + weak_mat_tmp13*weak_mat_tmp27 - weak_mat_tmp25*weak_mat_tmp26 + s_t(2);
    const s_t weak_mat_tmp29 = pow(weak_mat_tmp4, (s_t(-7) / s_t(3)));
    const s_t weak_mat_tmp30 = ((s_t(8) / s_t(3)))*weak_mat_tmp29;
    const s_t weak_mat_tmp31 = -(s_t(1) / s_t(2))*pow_2(weak_mat_tmp10) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp12) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp13) + weak_mat_tmp13 - pow_2(weak_mat_tmp25);
    const s_t weak_mat_tmp32 = pow(weak_mat_tmp4, (s_t(-10) / s_t(3)));
    const s_t weak_mat_tmp33 = ((s_t(28) / s_t(9)))*weak_mat_tmp31*weak_mat_tmp32;
    const s_t weak_mat_tmp34 = weak_mat_tmp24*weak_mat_tmp5;
    const s_t weak_mat_tmp35 = ((s_t(4) / s_t(3)))*weak_mat_tmp19;
    const s_t weak_mat_tmp36 = gu1*weak_mat_tmp0;
    const s_t weak_mat_tmp37 = weak_mat_tmp35*weak_mat_tmp36;
    const s_t weak_mat_tmp38 = gu2*weak_mat_tmp3;
    const s_t weak_mat_tmp39 = weak_mat_tmp35*weak_mat_tmp38;
    const s_t weak_mat_tmp40 = s_t(2)*weak_mat_tmp22;
    const s_t weak_mat_tmp41 = s_t(2)*gu1*weak_mat_tmp13 + s_t(2)*gu1 - weak_mat_tmp12*weak_mat_tmp26 - weak_mat_tmp25*weak_mat_tmp27;
    const s_t weak_mat_tmp42 = ((s_t(4) / s_t(3)))*weak_mat_tmp29;
    const s_t weak_mat_tmp43 = weak_mat_tmp0*weak_mat_tmp42;
    const s_t weak_mat_tmp44 = c1*(-weak_mat_tmp16*weak_mat_tmp24 - weak_mat_tmp37 + weak_mat_tmp39) + c2*(((s_t(4) / s_t(3)))*gu2*weak_mat_tmp28*weak_mat_tmp29 - weak_mat_tmp24*weak_mat_tmp33 - weak_mat_tmp24*weak_mat_tmp40 - weak_mat_tmp41*weak_mat_tmp43) + weak_mat_tmp34*weak_mat_tmp7 - weak_mat_tmp34;
    const s_t weak_mat_tmp45 = weak_mat_tmp36*weak_mat_tmp5;
    const s_t weak_mat_tmp46 = weak_mat_tmp23*weak_mat_tmp35;
    const s_t weak_mat_tmp47 = weak_mat_tmp24*weak_mat_tmp35;
    const s_t weak_mat_tmp48 = s_t(2)*gu2;
    const s_t weak_mat_tmp49 = s_t(2)*weak_mat_tmp0;
    const s_t weak_mat_tmp50 = s_t(2)*gu2*weak_mat_tmp13 + s_t(2)*gu2 - weak_mat_tmp10*weak_mat_tmp48 - weak_mat_tmp25*weak_mat_tmp49;
    const s_t weak_mat_tmp51 = c1*(-weak_mat_tmp16*weak_mat_tmp36 + weak_mat_tmp46 - weak_mat_tmp47) + c2*(((s_t(4) / s_t(3)))*gu1*weak_mat_tmp28*weak_mat_tmp29 - weak_mat_tmp0*weak_mat_tmp22*weak_mat_tmp26 - weak_mat_tmp33*weak_mat_tmp36 - weak_mat_tmp43*weak_mat_tmp50) + weak_mat_tmp45*weak_mat_tmp7 - weak_mat_tmp45;
    const s_t weak_mat_tmp52 = weak_mat_tmp18*weak_mat_tmp5;
    const s_t weak_mat_tmp53 = kappa*weak_mat_tmp7/weak_mat_tmp4;
    const s_t weak_mat_tmp54 = ((s_t(2) / s_t(3)))*weak_mat_tmp14*weak_mat_tmp19;
    const s_t weak_mat_tmp55 = s_t(2)*gu3 - weak_mat_tmp12*weak_mat_tmp49 + weak_mat_tmp13*weak_mat_tmp49 - weak_mat_tmp25*weak_mat_tmp48 + s_t(2);
    const s_t weak_mat_tmp56 = weak_mat_tmp31*weak_mat_tmp42;
    const s_t weak_mat_tmp57 = c1*(((s_t(10) / s_t(9)))*weak_mat_tmp0*weak_mat_tmp14*weak_mat_tmp15*weak_mat_tmp3 - weak_mat_tmp1*weak_mat_tmp35 - weak_mat_tmp35*weak_mat_tmp9 - weak_mat_tmp54) + c2*(((s_t(28) / s_t(9)))*weak_mat_tmp0*weak_mat_tmp3*weak_mat_tmp31*weak_mat_tmp32 + weak_mat_tmp22*(s_t(4)*weak_mat_tmp0*weak_mat_tmp3 - s_t(2)*weak_mat_tmp2) - weak_mat_tmp28*weak_mat_tmp3*weak_mat_tmp42 - weak_mat_tmp43*weak_mat_tmp55 - weak_mat_tmp56) - weak_mat_tmp52*weak_mat_tmp7 + weak_mat_tmp52 + weak_mat_tmp53;
    const s_t weak_mat_tmp58 = weak_mat_tmp5*weak_mat_tmp8;
    const s_t weak_mat_tmp59 = weak_mat_tmp17 + weak_mat_tmp2*weak_mat_tmp20;
    const s_t weak_mat_tmp60 = weak_mat_tmp38*weak_mat_tmp5;
    const s_t weak_mat_tmp61 = weak_mat_tmp41*weak_mat_tmp42;
    const s_t weak_mat_tmp62 = c1*(-weak_mat_tmp16*weak_mat_tmp38 - weak_mat_tmp46 + weak_mat_tmp47) + c2*(((s_t(4) / s_t(3)))*gu2*weak_mat_tmp29*weak_mat_tmp55 - weak_mat_tmp22*weak_mat_tmp3*weak_mat_tmp48 - weak_mat_tmp3*weak_mat_tmp61 - weak_mat_tmp33*weak_mat_tmp38) + weak_mat_tmp60*weak_mat_tmp7 - weak_mat_tmp60;
    const s_t weak_mat_tmp63 = weak_mat_tmp2*weak_mat_tmp5;
    const s_t weak_mat_tmp64 = c1*(weak_mat_tmp11*weak_mat_tmp35 + weak_mat_tmp16*weak_mat_tmp2 + weak_mat_tmp35*weak_mat_tmp8 + weak_mat_tmp54) + c2*(gu1*weak_mat_tmp61 + gu2*weak_mat_tmp42*weak_mat_tmp50 + weak_mat_tmp2*weak_mat_tmp33 + weak_mat_tmp22*(-s_t(2)*weak_mat_tmp18 + s_t(4)*weak_mat_tmp2) + weak_mat_tmp56) - weak_mat_tmp53 - weak_mat_tmp63*weak_mat_tmp7 + weak_mat_tmp63;
    const s_t weak_mat_tmp65 = weak_mat_tmp11*weak_mat_tmp5;
    const s_t weak_mat_tmp66 = weak_mat_tmp23*weak_mat_tmp5;
    const s_t weak_mat_tmp67 = c1*(-weak_mat_tmp16*weak_mat_tmp23 + weak_mat_tmp37 - weak_mat_tmp39) + c2*(((s_t(4) / s_t(3)))*gu1*weak_mat_tmp29*weak_mat_tmp55 - weak_mat_tmp23*weak_mat_tmp33 - weak_mat_tmp23*weak_mat_tmp40 - weak_mat_tmp3*weak_mat_tmp42*weak_mat_tmp50) + weak_mat_tmp66*weak_mat_tmp7 - weak_mat_tmp66;
    const s_t weak_mat_tmp68 = weak_mat_tmp5*weak_mat_tmp9;
    const s_t material0 = trial_grad0*(c1*(weak_mat_tmp1*weak_mat_tmp16 + weak_mat_tmp21) + c2*(-weak_mat_tmp0*weak_mat_tmp28*weak_mat_tmp30 + weak_mat_tmp1*weak_mat_tmp33 + weak_mat_tmp22*(s_t(2)*weak_mat_tmp1 + s_t(2))) - weak_mat_tmp6*weak_mat_tmp7 + weak_mat_tmp6) + trial_grad1*weak_mat_tmp44 + trial_grad2*weak_mat_tmp51 + trial_grad3*weak_mat_tmp57;
    const s_t material1 = trial_grad0*weak_mat_tmp44 + trial_grad1*(c1*(weak_mat_tmp16*weak_mat_tmp8 + weak_mat_tmp59) + c2*(gu2*weak_mat_tmp30*weak_mat_tmp41 + weak_mat_tmp22*(s_t(2)*weak_mat_tmp8 + s_t(2)) + weak_mat_tmp33*weak_mat_tmp8) - weak_mat_tmp58*weak_mat_tmp7 + weak_mat_tmp58) + trial_grad2*weak_mat_tmp64 + trial_grad3*weak_mat_tmp62;
    const s_t material2 = trial_grad0*weak_mat_tmp51 + trial_grad1*weak_mat_tmp64 + trial_grad2*(c1*(weak_mat_tmp11*weak_mat_tmp16 + weak_mat_tmp59) + c2*(gu1*weak_mat_tmp30*weak_mat_tmp50 + weak_mat_tmp11*weak_mat_tmp33 + weak_mat_tmp22*(s_t(2)*weak_mat_tmp11 + s_t(2))) - weak_mat_tmp65*weak_mat_tmp7 + weak_mat_tmp65) + trial_grad3*weak_mat_tmp67;
    const s_t material3 = trial_grad0*weak_mat_tmp57 + trial_grad1*weak_mat_tmp62 + trial_grad2*weak_mat_tmp67 + trial_grad3*(c1*(weak_mat_tmp16*weak_mat_tmp9 + weak_mat_tmp21) + c2*(weak_mat_tmp22*(s_t(2)*weak_mat_tmp9 + s_t(2)) - weak_mat_tmp3*weak_mat_tmp30*weak_mat_tmp55 + weak_mat_tmp33*weak_mat_tmp9) - weak_mat_tmp68*weak_mat_tmp7 + weak_mat_tmp68);
    const s_t loperand0 = qw * (material0 * adj_value0 + material1 * adj_value1);
    const s_t loperand1 = qw * (material0 * adj_value2 + material1 * adj_value3);
    const s_t loperand2 = qw * (material2 * adj_value0 + material3 * adj_value1);
    const s_t loperand3 = qw * (material2 * adj_value2 + material3 * adj_value3);
      out_streams[0][0] += -(loperand0) - loperand1;
      out_streams[1][0] += -(loperand2) - loperand3;
      out_streams[2][0] += loperand0;
      out_streams[3][0] += loperand2;
      out_streams[4][0] += loperand1;
      out_streams[5][0] += loperand3;
      }
    }
}

} // namespace codegen
} // namespace sfem

#endif
