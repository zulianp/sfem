#ifndef LAPLACE_D2_SIMPLEX_LOCAL_CUH
#define LAPLACE_D2_SIMPLEX_LOCAL_CUH
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
static __host__ __device__ __forceinline__ void laplace_d2_simplex_objective_block(
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
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 1],
        const s_t *const RSTR h_streams[NS * 1],
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
      {
        gu_ref0_values[0] = s_t(0);
        grad_h_ref0_values[0] = s_t(0);
        gu_ref1_values[0] = s_t(0);
        grad_h_ref1_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        {
          gu_ref0_values[0] += u_streams[shape][0] * grad_ref_x[q * NS + shape];
          grad_h_ref0_values[0] += h_streams[shape][0] * grad_ref_x[q * NS + shape];
        }
        {
          gu_ref1_values[0] += u_streams[shape][0] * grad_ref_y[q * NS + shape];
          grad_h_ref1_values[0] += h_streams[shape][0] * grad_ref_y[q * NS + shape];
        }
      }
      s_t gu_base_v[2 * VS];
      s_t trial_grad_v[2 * VS];
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
    const s_t idet = s_t(1) / det_value0;
    gu_base_v[0 * VS + 0] = (gu_ref0 * adj_value0 + gu_ref1 * adj_value2) * idet;
    trial_grad_v[0 * VS + 0] = (grad_h_ref0 * adj_value0 + grad_h_ref1 * adj_value2) * idet;
    gu_base_v[1 * VS + 0] = (gu_ref0 * adj_value1 + gu_ref1 * adj_value3) * idet;
    trial_grad_v[1 * VS + 0] = (grad_h_ref0 * adj_value1 + grad_h_ref1 * adj_value3) * idet;
      }
      for (int step = 0; step < nsteps; ++step) {
        const s_t alpha = steps[step];
        {
          const ptrdiff_t goff = q * geometry_stride + 0;
          const s_t det_value0 = det0[goff];
          const s_t gu0 = gu_base_v[0 * VS + 0] + alpha * trial_grad_v[0 * VS + 0];
          const s_t gu1 = gu_base_v[1 * VS + 0] + alpha * trial_grad_v[1 * VS + 0];
    value[step * value_stride + 0] += qw * det_value0 * (((s_t(1) / s_t(2)))*kappa*(pow_2(gu0) + pow_2(gu1)));
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void laplace_d2_simplex_tri3_objective_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR q_weight,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 1],
        const s_t *const RSTR h_streams[NS * 1],
        const int nsteps,
        const s_t *const RSTR steps,
        const ptrdiff_t value_stride,
        s_t *const RSTR value
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
    { const int q = 0;  // constant-P1 simplex
      const s_t qw = q_weight[q];
      s_t gu_base_v[2 * VS];
      s_t trial_grad_v[2 * VS];
      {
      const ptrdiff_t goff = q * geometry_stride + 0;
      const s_t adj_value0 = adj0[goff];
      const s_t adj_value1 = adj1[goff];
      const s_t adj_value2 = adj2[goff];
      const s_t adj_value3 = adj3[goff];
      const s_t det_value0 = det0[goff];
      const s_t gu_ref0 = -(u_streams[0][0]) + u_streams[1][0];
      const s_t grad_h_ref0 = -(h_streams[0][0]) + h_streams[1][0];
      const s_t gu_ref1 = -(u_streams[0][0]) + u_streams[2][0];
      const s_t grad_h_ref1 = -(h_streams[0][0]) + h_streams[2][0];
      const s_t idet = s_t(1) / det_value0;
      gu_base_v[0 * VS + 0] = (gu_ref0 * adj_value0 + gu_ref1 * adj_value2) * idet;
      trial_grad_v[0 * VS + 0] = (grad_h_ref0 * adj_value0 + grad_h_ref1 * adj_value2) * idet;
      gu_base_v[1 * VS + 0] = (gu_ref0 * adj_value1 + gu_ref1 * adj_value3) * idet;
      trial_grad_v[1 * VS + 0] = (grad_h_ref0 * adj_value1 + grad_h_ref1 * adj_value3) * idet;
      }
      for (int step = 0; step < nsteps; ++step) {
        const s_t alpha = steps[step];
        {
          const ptrdiff_t goff = q * geometry_stride + 0;
          const s_t det_value0 = det0[goff];
          const s_t gu0 = gu_base_v[0 * VS + 0] + alpha * trial_grad_v[0 * VS + 0];
          const s_t gu1 = gu_base_v[1 * VS + 0] + alpha * trial_grad_v[1 * VS + 0];
    value[step * value_stride + 0] += qw * det_value0 * (((s_t(1) / s_t(2)))*kappa*(pow_2(gu0) + pow_2(gu1)));
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void laplace_d2_simplex_tri3_metric_objective_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR geom_metric0,
        const s_t *const RSTR geom_metric1,
        const s_t *const RSTR geom_metric2,
        const s_t *const RSTR q_weight,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 1],
        const s_t *const RSTR h_streams[NS * 1],
        const int nsteps,
        const s_t *const RSTR steps,
        const ptrdiff_t value_stride,
        s_t *const RSTR value
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
    for (int step = 0; step < nsteps; ++step) {
      const s_t alpha = steps[step];
      {
        const ptrdiff_t goff = 0;
        const s_t geom_metric_value0 = geom_metric0[goff];
        const s_t geom_metric_value1 = geom_metric1[goff];
        const s_t geom_metric_value2 = geom_metric2[goff];
        const s_t u_step0 = u_streams[0][0] + alpha * h_streams[0][0];
        const s_t u_step1 = u_streams[1][0] + alpha * h_streams[1][0];
        const s_t u_step2 = u_streams[2][0] + alpha * h_streams[2][0];
        const s_t t0 = -u_step0 + u_step1;
        const s_t t1 = -u_step0 + u_step2;
        const s_t t2 = geom_metric_value0*t0 + geom_metric_value1*t1;
        const s_t t3 = geom_metric_value1*t0 + geom_metric_value2*t1;
        value[step * value_stride + 0] += ((s_t(1) / s_t(2)))*kappa*(t2*(-u_step0 + u_step1) + t3*(-u_step0 + u_step2));
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void laplace_d2_simplex_gradient_block(
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
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 1],
        s_t *const RSTR out_streams[NS * 1]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
    for (int q = 0; q < NQ; ++q) {
      const s_t qw = q_weight[q];
      s_t gu_ref0_values[VS];
      s_t gu_ref1_values[VS];
      s_t loperand0_values[VS];
      s_t loperand1_values[VS];
      {
        gu_ref0_values[0] = s_t(0);
        gu_ref1_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        {
          gu_ref0_values[0] += u_streams[shape][0] * grad_ref_x[q * NS + shape];
        }
        {
          gu_ref1_values[0] += u_streams[shape][0] * grad_ref_y[q * NS + shape];
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
    const s_t idet = s_t(1) / det_value0;
    const s_t gu0 = (gu_ref0 * adj_value0 + gu_ref1 * adj_value2) * idet;
    const s_t gu1 = (gu_ref0 * adj_value1 + gu_ref1 * adj_value3) * idet;
    const s_t material0 = gu0*kappa;
    const s_t material1 = gu1*kappa;
    const s_t loperand0 = qw * (material0 * adj_value0 + material1 * adj_value1);
    const s_t loperand1 = qw * (material0 * adj_value2 + material1 * adj_value3);
      loperand0_values[0] = loperand0;
      loperand1_values[0] = loperand1;
      }
      for (int shape = 0; shape < NS; ++shape) {
        {
          out_streams[shape][0] += loperand0_values[0] * grad_ref_x[q * NS + shape] + loperand1_values[0] * grad_ref_y[q * NS + shape];
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void laplace_d2_simplex_tri3_gradient_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR q_weight,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 1],
        s_t *const RSTR out_streams[NS * 1]
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
      const s_t gu_ref0 = -(u_streams[0][0]) + u_streams[1][0];
      const s_t gu_ref1 = -(u_streams[0][0]) + u_streams[2][0];
      const s_t idet = s_t(1) / det_value0;
      const s_t gu0 = (gu_ref0 * adj_value0 + gu_ref1 * adj_value2) * idet;
      const s_t gu1 = (gu_ref0 * adj_value1 + gu_ref1 * adj_value3) * idet;
    const s_t material0 = gu0*kappa;
    const s_t material1 = gu1*kappa;
    const s_t loperand0 = qw * (material0 * adj_value0 + material1 * adj_value1);
    const s_t loperand1 = qw * (material0 * adj_value2 + material1 * adj_value3);
      out_streams[0][0] += -(loperand0) - loperand1;
      out_streams[1][0] += loperand0;
      out_streams[2][0] += loperand1;
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void laplace_d2_simplex_tri3_metric_gradient_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR geom_metric0,
        const s_t *const RSTR geom_metric1,
        const s_t *const RSTR geom_metric2,
        const s_t *const RSTR q_weight,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 1],
        s_t *const RSTR out_streams[NS * 1]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
    {
      const ptrdiff_t goff = 0;
      const s_t geom_metric_value0 = geom_metric0[goff];
      const s_t geom_metric_value1 = geom_metric1[goff];
      const s_t geom_metric_value2 = geom_metric2[goff];
      const s_t t0 = -u_streams[0][0] + u_streams[1][0];
      const s_t t1 = -u_streams[0][0] + u_streams[2][0];
      const s_t t2 = geom_metric_value0*t0 + geom_metric_value1*t1;
      const s_t t3 = geom_metric_value1*t0 + geom_metric_value2*t1;
      out_streams[0][0] += kappa*(-t2 - t3);
      out_streams[1][0] += kappa*t2;
      out_streams[2][0] += kappa*t3;
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void laplace_d2_simplex_apply_block(
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
        const s_t kappa,
        const s_t *const RSTR h_streams[NS * 1],
        s_t *const RSTR out_streams[NS * 1]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
    for (int q = 0; q < NQ; ++q) {
      const s_t qw = q_weight[q];
      s_t grad_h_ref0_values[VS];
      s_t grad_h_ref1_values[VS];
      s_t loperand0_values[VS];
      s_t loperand1_values[VS];
      {
        grad_h_ref0_values[0] = s_t(0);
        grad_h_ref1_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        {
          grad_h_ref0_values[0] += h_streams[shape][0] * grad_ref_x[q * NS + shape];
        }
        {
          grad_h_ref1_values[0] += h_streams[shape][0] * grad_ref_y[q * NS + shape];
        }
      }
      {
      const ptrdiff_t goff = q * geometry_stride + 0;
      const s_t adj_value0 = adj0[goff];
      const s_t adj_value1 = adj1[goff];
      const s_t adj_value2 = adj2[goff];
      const s_t adj_value3 = adj3[goff];
      const s_t det_value0 = det0[goff];
      const s_t grad_h_ref0 = grad_h_ref0_values[0];
      const s_t grad_h_ref1 = grad_h_ref1_values[0];
    const s_t idet = s_t(1) / det_value0;
    const s_t trial_grad0 = (grad_h_ref0 * adj_value0 + grad_h_ref1 * adj_value2) * idet;
    const s_t trial_grad1 = (grad_h_ref0 * adj_value1 + grad_h_ref1 * adj_value3) * idet;
    const s_t material0 = kappa*trial_grad0;
    const s_t material1 = kappa*trial_grad1;
    const s_t loperand0 = qw * (material0 * adj_value0 + material1 * adj_value1);
    const s_t loperand1 = qw * (material0 * adj_value2 + material1 * adj_value3);
      loperand0_values[0] = loperand0;
      loperand1_values[0] = loperand1;
      }
      for (int shape = 0; shape < NS; ++shape) {
        {
          out_streams[shape][0] += loperand0_values[0] * grad_ref_x[q * NS + shape] + loperand1_values[0] * grad_ref_y[q * NS + shape];
        }
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void laplace_d2_simplex_tri3_apply_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR q_weight,
        const s_t kappa,
        const s_t *const RSTR h_streams[NS * 1],
        s_t *const RSTR out_streams[NS * 1]
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
      const s_t grad_h_ref0 = -(h_streams[0][0]) + h_streams[1][0];
      const s_t grad_h_ref1 = -(h_streams[0][0]) + h_streams[2][0];
      const s_t idet = s_t(1) / det_value0;
      const s_t trial_grad0 = (grad_h_ref0 * adj_value0 + grad_h_ref1 * adj_value2) * idet;
      const s_t trial_grad1 = (grad_h_ref0 * adj_value1 + grad_h_ref1 * adj_value3) * idet;
    const s_t material0 = kappa*trial_grad0;
    const s_t material1 = kappa*trial_grad1;
    const s_t loperand0 = qw * (material0 * adj_value0 + material1 * adj_value1);
    const s_t loperand1 = qw * (material0 * adj_value2 + material1 * adj_value3);
      out_streams[0][0] += -(loperand0) - loperand1;
      out_streams[1][0] += loperand0;
      out_streams[2][0] += loperand1;
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void laplace_d2_simplex_tri3_metric_apply_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR geom_metric0,
        const s_t *const RSTR geom_metric1,
        const s_t *const RSTR geom_metric2,
        const s_t *const RSTR q_weight,
        const s_t kappa,
        const s_t *const RSTR h_streams[NS * 1],
        s_t *const RSTR out_streams[NS * 1]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
    {
      const ptrdiff_t goff = 0;
      const s_t geom_metric_value0 = geom_metric0[goff];
      const s_t geom_metric_value1 = geom_metric1[goff];
      const s_t geom_metric_value2 = geom_metric2[goff];
      const s_t t0 = -h_streams[0][0] + h_streams[1][0];
      const s_t t1 = -h_streams[0][0] + h_streams[2][0];
      const s_t t2 = geom_metric_value0*t0 + geom_metric_value1*t1;
      const s_t t3 = geom_metric_value1*t0 + geom_metric_value2*t1;
      out_streams[0][0] += kappa*(-t2 - t3);
      out_streams[1][0] += kappa*t2;
      out_streams[2][0] += kappa*t3;
    }
}

} // namespace codegen
} // namespace sfem

#endif
