#ifndef LINEAR_ELASTICITY_D2_SIMPLEX_LOCAL_CUH
#define LINEAR_ELASTICITY_D2_SIMPLEX_LOCAL_CUH
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
static __host__ __device__ __forceinline__ void linear_elasticity_d2_simplex_objective_block(
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
        const s_t lmbda,
        const s_t mu,
        const s_t *const RSTR u_streams[NS * 2],
        s_t *const RSTR value
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
    for (int q = 0; q < NQ; ++q) {
      const s_t qw = q_weight[q];
      s_t gu_ref0_values[VS];
      s_t gu_ref1_values[VS];
      s_t gu_ref2_values[VS];
      s_t gu_ref3_values[VS];
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
    value[0] += qw * det_value0 * (((s_t(1) / s_t(2)))*lmbda*pow_2(gu0 + gu3) + mu*(pow_2(gu0) + pow_2(gu3) + s_t(2)*pow_2(((s_t(1) / s_t(2)))*gu1 + ((s_t(1) / s_t(2)))*gu2)));
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void linear_elasticity_d2_simplex_tri3_objective_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR q_weight,
        const s_t lmbda,
        const s_t mu,
        const s_t *const RSTR u_streams[NS * 2],
        s_t *const RSTR value
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
    value[0] += qw * det_value0 * (((s_t(1) / s_t(2)))*lmbda*pow_2(gu0 + gu3) + mu*(pow_2(gu0) + pow_2(gu3) + s_t(2)*pow_2(((s_t(1) / s_t(2)))*gu1 + ((s_t(1) / s_t(2)))*gu2)));
      }
    }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void linear_elasticity_d2_simplex_gradient_block(
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
        const s_t lmbda,
        const s_t mu,
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
    const s_t weak_mat_tmp0 = s_t(2)*gu0;
    const s_t weak_mat_tmp1 = s_t(2)*gu3;
    const s_t weak_mat_tmp2 = ((s_t(1) / s_t(2)))*lmbda*(weak_mat_tmp0 + weak_mat_tmp1);
    const s_t weak_mat_tmp3 = mu*(gu1 + gu2);
    const s_t material0 = mu*weak_mat_tmp0 + weak_mat_tmp2;
    const s_t material1 = weak_mat_tmp3;
    const s_t material2 = weak_mat_tmp3;
    const s_t material3 = mu*weak_mat_tmp1 + weak_mat_tmp2;
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
static __host__ __device__ __forceinline__ void linear_elasticity_d2_simplex_tri3_gradient_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR q_weight,
        const s_t lmbda,
        const s_t mu,
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
    const s_t weak_mat_tmp0 = s_t(2)*gu0;
    const s_t weak_mat_tmp1 = s_t(2)*gu3;
    const s_t weak_mat_tmp2 = ((s_t(1) / s_t(2)))*lmbda*(weak_mat_tmp0 + weak_mat_tmp1);
    const s_t weak_mat_tmp3 = mu*(gu1 + gu2);
    const s_t material0 = mu*weak_mat_tmp0 + weak_mat_tmp2;
    const s_t material1 = weak_mat_tmp3;
    const s_t material2 = weak_mat_tmp3;
    const s_t material3 = mu*weak_mat_tmp1 + weak_mat_tmp2;
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
static __host__ __device__ __forceinline__ void linear_elasticity_d2_simplex_apply_block(
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
        const s_t lmbda,
        const s_t mu,
        const s_t *const RSTR h_streams[NS * 2],
        s_t *const RSTR out_streams[NS * 2]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
    for (int q = 0; q < NQ; ++q) {
      const s_t qw = q_weight[q];
      s_t grad_h_ref0_values[VS];
      s_t grad_h_ref1_values[VS];
      s_t grad_h_ref2_values[VS];
      s_t grad_h_ref3_values[VS];
      s_t loperand0_values[VS];
      s_t loperand1_values[VS];
      s_t loperand2_values[VS];
      s_t loperand3_values[VS];
      {
        grad_h_ref0_values[0] = s_t(0);
        grad_h_ref1_values[0] = s_t(0);
        grad_h_ref2_values[0] = s_t(0);
        grad_h_ref3_values[0] = s_t(0);
      }
      for (int shape = 0; shape < NS; ++shape) {
        {
          grad_h_ref0_values[0] += h_streams[2 * shape][0] * grad_ref_x[q * NS + shape];
        }
        {
          grad_h_ref1_values[0] += h_streams[2 * shape][0] * grad_ref_y[q * NS + shape];
        }
        {
          grad_h_ref2_values[0] += h_streams[2 * shape + 1][0] * grad_ref_x[q * NS + shape];
        }
        {
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
      const s_t grad_h_ref0 = grad_h_ref0_values[0];
      const s_t grad_h_ref1 = grad_h_ref1_values[0];
      const s_t grad_h_ref2 = grad_h_ref2_values[0];
      const s_t grad_h_ref3 = grad_h_ref3_values[0];
    const s_t idet = s_t(1) / det_value0;
    const s_t trial_grad0 = (grad_h_ref0 * adj_value0 + grad_h_ref1 * adj_value2) * idet;
    const s_t trial_grad1 = (grad_h_ref0 * adj_value1 + grad_h_ref1 * adj_value3) * idet;
    const s_t trial_grad2 = (grad_h_ref2 * adj_value0 + grad_h_ref3 * adj_value2) * idet;
    const s_t trial_grad3 = (grad_h_ref2 * adj_value1 + grad_h_ref3 * adj_value3) * idet;
    const s_t weak_mat_tmp0 = s_t(2)*trial_grad0;
    const s_t weak_mat_tmp1 = s_t(2)*trial_grad3;
    const s_t weak_mat_tmp2 = ((s_t(1) / s_t(2)))*lmbda*(weak_mat_tmp0 + weak_mat_tmp1);
    const s_t weak_mat_tmp3 = mu*(trial_grad1 + trial_grad2);
    const s_t material0 = mu*weak_mat_tmp0 + weak_mat_tmp2;
    const s_t material1 = weak_mat_tmp3;
    const s_t material2 = weak_mat_tmp3;
    const s_t material3 = mu*weak_mat_tmp1 + weak_mat_tmp2;
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
static __host__ __device__ __forceinline__ void linear_elasticity_d2_simplex_tri3_apply_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR q_weight,
        const s_t lmbda,
        const s_t mu,
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
      const s_t grad_h_ref0 = -(h_streams[0][0]) + h_streams[2][0];
      const s_t grad_h_ref1 = -(h_streams[0][0]) + h_streams[4][0];
      const s_t grad_h_ref2 = -(h_streams[1][0]) + h_streams[3][0];
      const s_t grad_h_ref3 = -(h_streams[1][0]) + h_streams[5][0];
      const s_t idet = s_t(1) / det_value0;
      const s_t trial_grad0 = (grad_h_ref0 * adj_value0 + grad_h_ref1 * adj_value2) * idet;
      const s_t trial_grad1 = (grad_h_ref0 * adj_value1 + grad_h_ref1 * adj_value3) * idet;
      const s_t trial_grad2 = (grad_h_ref2 * adj_value0 + grad_h_ref3 * adj_value2) * idet;
      const s_t trial_grad3 = (grad_h_ref2 * adj_value1 + grad_h_ref3 * adj_value3) * idet;
    const s_t weak_mat_tmp0 = s_t(2)*trial_grad0;
    const s_t weak_mat_tmp1 = s_t(2)*trial_grad3;
    const s_t weak_mat_tmp2 = ((s_t(1) / s_t(2)))*lmbda*(weak_mat_tmp0 + weak_mat_tmp1);
    const s_t weak_mat_tmp3 = mu*(trial_grad1 + trial_grad2);
    const s_t material0 = mu*weak_mat_tmp0 + weak_mat_tmp2;
    const s_t material1 = weak_mat_tmp3;
    const s_t material2 = weak_mat_tmp3;
    const s_t material3 = mu*weak_mat_tmp1 + weak_mat_tmp2;
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
