#ifndef LAPLACE_D2_TENSOR_PRODUCT_LOCAL_CUH
#define LAPLACE_D2_TENSOR_PRODUCT_LOCAL_CUH
#include <math.h>
#include <stddef.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_GENERATED_SCALAR_T
#endif
#endif
#include "../../../cuda/kernel_math.cuh"
#include "../../../cuda/tensor_product_kernels.cuh"
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
static __host__ __device__ __forceinline__ void laplace_d2_tensor_product_objective_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 1],
        s_t *const RSTR value
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NQ1 = integer_root(NQ, 2);
  static constexpr int NS1 = integer_root(NS, 2);
  static_assert(ipow(NQ1, 2) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 2) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[NQ * 4 * VS];
  tensor_gradient<s_t, NQ, NS, VS, 2, 1>(ne, shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = q / NQ1;
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
    const s_t *const RSTR gu_ref0 = &gu_ref_q[(2 * q) * VS];
    const s_t *const RSTR gu_ref1 = &gu_ref_q[(2 * q + 1) * VS];
    const s_t *const RSTR adj_q0 = adj0 + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adj1 + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adj2 + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adj3 + q * geometry_stride;
    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;
    {
      const s_t adj_value0 = adj_q0[0];
      const s_t adj_value1 = adj_q1[0];
      const s_t adj_value2 = adj_q2[0];
      const s_t adj_value3 = adj_q3[0];
      const s_t det_value0 = det_q0[0];
      s_t gu[2];
      const s_t idet = s_t(1) / det_value0;
      gu[0] = (gu_ref0[0] * adj_value0 + gu_ref1[0] * adj_value2) * idet;
      gu[1] = (gu_ref0[0] * adj_value1 + gu_ref1[0] * adj_value3) * idet;
    value[0] += qw * det_value0 * (((s_t(1) / s_t(2)))*kappa*(pow_2(gu[0]) + pow_2(gu[1])));
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void laplace_d2_tensor_product_gradient_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 1],
        s_t *const RSTR out_streams[NS * 1]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NQ1 = integer_root(NQ, 2);
  static constexpr int NS1 = integer_root(NS, 2);
  static_assert(ipow(NQ1, 2) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 2) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[NQ * 4 * VS];
  s_t loperand_q[NQ * 4 * VS];
  tensor_gradient<s_t, NQ, NS, VS, 2, 1>(ne, shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = q / NQ1;
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
    const s_t *const RSTR gu_ref0 = &gu_ref_q[(2 * q) * VS];
    const s_t *const RSTR gu_ref1 = &gu_ref_q[(2 * q + 1) * VS];
    s_t *const RSTR loperand0 = &loperand_q[(2 * q) * VS];
    s_t *const RSTR loperand1 = &loperand_q[(2 * q + 1) * VS];
    const s_t *const RSTR adj_q0 = adj0 + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adj1 + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adj2 + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adj3 + q * geometry_stride;
    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;
    {
      const s_t adj_value0 = adj_q0[0];
      const s_t adj_value1 = adj_q1[0];
      const s_t adj_value2 = adj_q2[0];
      const s_t adj_value3 = adj_q3[0];
      const s_t det_value0 = det_q0[0];
      s_t gu[2];
      const s_t idet = s_t(1) / det_value0;
      gu[0] = (gu_ref0[0] * adj_value0 + gu_ref1[0] * adj_value2) * idet;
      gu[1] = (gu_ref0[0] * adj_value1 + gu_ref1[0] * adj_value3) * idet;
      s_t loperand[2];
    s_t material[2];
    material[0] = gu[0]*kappa;
    material[1] = gu[1]*kappa;
    loperand[0] = qw * (material[0] * adj_value0 + material[1] * adj_value1);
    loperand[1] = qw * (material[0] * adj_value2 + material[1] * adj_value3);
      loperand0[0] = loperand[0];
      loperand1[0] = loperand[1];
    }
  }
  tensor_test<s_t, NQ, NS, VS, 2, 1>(ne, shape_1d, grad_1d, &loperand_q[0], out_streams, 0);
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void laplace_d2_tensor_product_apply_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR det0,
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t kappa,
        const s_t *const RSTR h_streams[NS * 1],
        s_t *const RSTR out_streams[NS * 1]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NQ1 = integer_root(NQ, 2);
  static constexpr int NS1 = integer_root(NS, 2);
  static_assert(ipow(NQ1, 2) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 2) == NS, "NS must be tensor-product compatible");
  s_t grad_h_ref_q[NQ * 4 * VS];
  s_t loperand_q[NQ * 4 * VS];
  tensor_gradient<s_t, NQ, NS, VS, 2, 1>(ne, shape_1d, grad_1d, h_streams, 0, &grad_h_ref_q[0]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = q / NQ1;
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
    const s_t *const RSTR grad_h_ref0 = &grad_h_ref_q[(2 * q) * VS];
    const s_t *const RSTR grad_h_ref1 = &grad_h_ref_q[(2 * q + 1) * VS];
    s_t *const RSTR loperand0 = &loperand_q[(2 * q) * VS];
    s_t *const RSTR loperand1 = &loperand_q[(2 * q + 1) * VS];
    const s_t *const RSTR adj_q0 = adj0 + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adj1 + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adj2 + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adj3 + q * geometry_stride;
    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;
    {
      const s_t adj_value0 = adj_q0[0];
      const s_t adj_value1 = adj_q1[0];
      const s_t adj_value2 = adj_q2[0];
      const s_t adj_value3 = adj_q3[0];
      const s_t det_value0 = det_q0[0];
      s_t trial_grad[2];
      const s_t idet = s_t(1) / det_value0;
      trial_grad[0] = (grad_h_ref0[0] * adj_value0 + grad_h_ref1[0] * adj_value2) * idet;
      trial_grad[1] = (grad_h_ref0[0] * adj_value1 + grad_h_ref1[0] * adj_value3) * idet;
      s_t loperand[2];
    s_t material[2];
    material[0] = kappa*trial_grad[0];
    material[1] = kappa*trial_grad[1];
    loperand[0] = qw * (material[0] * adj_value0 + material[1] * adj_value1);
    loperand[1] = qw * (material[0] * adj_value2 + material[1] * adj_value3);
      loperand0[0] = loperand[0];
      loperand1[0] = loperand[1];
    }
  }
  tensor_test<s_t, NQ, NS, VS, 2, 1>(ne, shape_1d, grad_1d, &loperand_q[0], out_streams, 0);
}

} // namespace codegen
} // namespace sfem

#endif
