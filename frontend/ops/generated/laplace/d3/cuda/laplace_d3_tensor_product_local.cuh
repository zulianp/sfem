#ifndef LAPLACE_D3_TENSOR_PRODUCT_LOCAL_CUH
#define LAPLACE_D3_TENSOR_PRODUCT_LOCAL_CUH
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
static __host__ __device__ __forceinline__ void laplace_d3_tensor_product_objective_block(
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
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
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
  static constexpr int NQ1 = integer_root(NQ, 3);
  static constexpr int NS1 = integer_root(NS, 3);
  static_assert(ipow(NQ1, 3) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 3) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[NQ * 9 * VS];
  s_t grad_h_ref_q[NQ * 9 * VS];
  tensor_gradient<s_t, NQ, NS, VS, 3, 1>(ne, shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 1>(ne, shape_1d, grad_1d, h_streams, 0, &grad_h_ref_q[0]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
    const s_t *const RSTR gu_ref0 = &gu_ref_q[(3 * q) * VS];
    const s_t *const RSTR gu_ref1 = &gu_ref_q[(3 * q + 1) * VS];
    const s_t *const RSTR gu_ref2 = &gu_ref_q[(3 * q + 2) * VS];
    const s_t *const RSTR grad_h_ref0 = &grad_h_ref_q[(3 * q) * VS];
    const s_t *const RSTR grad_h_ref1 = &grad_h_ref_q[(3 * q + 1) * VS];
    const s_t *const RSTR grad_h_ref2 = &grad_h_ref_q[(3 * q + 2) * VS];
    const s_t *const RSTR adj_q0 = adj0 + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adj1 + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adj2 + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adj3 + q * geometry_stride;
    const s_t *const RSTR adj_q4 = adj4 + q * geometry_stride;
    const s_t *const RSTR adj_q5 = adj5 + q * geometry_stride;
    const s_t *const RSTR adj_q6 = adj6 + q * geometry_stride;
    const s_t *const RSTR adj_q7 = adj7 + q * geometry_stride;
    const s_t *const RSTR adj_q8 = adj8 + q * geometry_stride;
    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;
    s_t gu_base_v[3 * VS];
    s_t trial_grad_v[3 * VS];
    {
      const s_t adj_value0 = adj_q0[0];
      const s_t adj_value1 = adj_q1[0];
      const s_t adj_value2 = adj_q2[0];
      const s_t adj_value3 = adj_q3[0];
      const s_t adj_value4 = adj_q4[0];
      const s_t adj_value5 = adj_q5[0];
      const s_t adj_value6 = adj_q6[0];
      const s_t adj_value7 = adj_q7[0];
      const s_t adj_value8 = adj_q8[0];
      const s_t det_value0 = det_q0[0];
      const s_t idet = s_t(1) / det_value0;
      gu_base_v[0 * VS + 0] = (gu_ref0[0] * adj_value0 + gu_ref1[0] * adj_value3 + gu_ref2[0] * adj_value6) * idet;
      trial_grad_v[0 * VS + 0] = (grad_h_ref0[0] * adj_value0 + grad_h_ref1[0] * adj_value3 + grad_h_ref2[0] * adj_value6) * idet;
      gu_base_v[1 * VS + 0] = (gu_ref0[0] * adj_value1 + gu_ref1[0] * adj_value4 + gu_ref2[0] * adj_value7) * idet;
      trial_grad_v[1 * VS + 0] = (grad_h_ref0[0] * adj_value1 + grad_h_ref1[0] * adj_value4 + grad_h_ref2[0] * adj_value7) * idet;
      gu_base_v[2 * VS + 0] = (gu_ref0[0] * adj_value2 + gu_ref1[0] * adj_value5 + gu_ref2[0] * adj_value8) * idet;
      trial_grad_v[2 * VS + 0] = (grad_h_ref0[0] * adj_value2 + grad_h_ref1[0] * adj_value5 + grad_h_ref2[0] * adj_value8) * idet;
    }
    for (int step = 0; step < nsteps; ++step) {
      const s_t alpha = steps[step];
      {
        const s_t det_value0 = det_q0[0];
        s_t gu[3];
        gu[0] = gu_base_v[0 * VS + 0] + alpha * trial_grad_v[0 * VS + 0];
        gu[1] = gu_base_v[1 * VS + 0] + alpha * trial_grad_v[1 * VS + 0];
        gu[2] = gu_base_v[2 * VS + 0] + alpha * trial_grad_v[2 * VS + 0];
    value[step * value_stride + 0] += qw * det_value0 * (((s_t(1) / s_t(2)))*kappa*(pow_2(gu[0]) + pow_2(gu[1]) + pow_2(gu[2])));
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void laplace_d3_tensor_product_gradient_block(
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
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 1],
        s_t *const RSTR out_streams[NS * 1]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NQ1 = integer_root(NQ, 3);
  static constexpr int NS1 = integer_root(NS, 3);
  static_assert(ipow(NQ1, 3) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 3) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[NQ * 9 * VS];
  s_t loperand_q[NQ * 9 * VS];
  tensor_gradient<s_t, NQ, NS, VS, 3, 1>(ne, shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
    const s_t *const RSTR gu_ref0 = &gu_ref_q[(3 * q) * VS];
    const s_t *const RSTR gu_ref1 = &gu_ref_q[(3 * q + 1) * VS];
    const s_t *const RSTR gu_ref2 = &gu_ref_q[(3 * q + 2) * VS];
    s_t *const RSTR loperand0 = &loperand_q[(3 * q) * VS];
    s_t *const RSTR loperand1 = &loperand_q[(3 * q + 1) * VS];
    s_t *const RSTR loperand2 = &loperand_q[(3 * q + 2) * VS];
    const s_t *const RSTR adj_q0 = adj0 + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adj1 + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adj2 + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adj3 + q * geometry_stride;
    const s_t *const RSTR adj_q4 = adj4 + q * geometry_stride;
    const s_t *const RSTR adj_q5 = adj5 + q * geometry_stride;
    const s_t *const RSTR adj_q6 = adj6 + q * geometry_stride;
    const s_t *const RSTR adj_q7 = adj7 + q * geometry_stride;
    const s_t *const RSTR adj_q8 = adj8 + q * geometry_stride;
    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;
    {
      const s_t adj_value0 = adj_q0[0];
      const s_t adj_value1 = adj_q1[0];
      const s_t adj_value2 = adj_q2[0];
      const s_t adj_value3 = adj_q3[0];
      const s_t adj_value4 = adj_q4[0];
      const s_t adj_value5 = adj_q5[0];
      const s_t adj_value6 = adj_q6[0];
      const s_t adj_value7 = adj_q7[0];
      const s_t adj_value8 = adj_q8[0];
      const s_t det_value0 = det_q0[0];
      s_t gu[3];
      const s_t idet = s_t(1) / det_value0;
      gu[0] = (gu_ref0[0] * adj_value0 + gu_ref1[0] * adj_value3 + gu_ref2[0] * adj_value6) * idet;
      gu[1] = (gu_ref0[0] * adj_value1 + gu_ref1[0] * adj_value4 + gu_ref2[0] * adj_value7) * idet;
      gu[2] = (gu_ref0[0] * adj_value2 + gu_ref1[0] * adj_value5 + gu_ref2[0] * adj_value8) * idet;
      s_t loperand[3];
    s_t material[3];
    material[0] = gu[0]*kappa;
    material[1] = gu[1]*kappa;
    material[2] = gu[2]*kappa;
    loperand[0] = qw * (material[0] * adj_value0 + material[1] * adj_value1 + material[2] * adj_value2);
    loperand[1] = qw * (material[0] * adj_value3 + material[1] * adj_value4 + material[2] * adj_value5);
    loperand[2] = qw * (material[0] * adj_value6 + material[1] * adj_value7 + material[2] * adj_value8);
      loperand0[0] = loperand[0];
      loperand1[0] = loperand[1];
      loperand2[0] = loperand[2];
    }
  }
  tensor_test<s_t, NQ, NS, VS, 3, 1>(ne, shape_1d, grad_1d, &loperand_q[0], out_streams, 0);
}

template <typename s_t, int NQ, int NS, int VS>
static __host__ __device__ __forceinline__ void laplace_d3_tensor_product_apply_block(
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
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t kappa,
        const s_t *const RSTR h_streams[NS * 1],
        s_t *const RSTR out_streams[NS * 1]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NQ1 = integer_root(NQ, 3);
  static constexpr int NS1 = integer_root(NS, 3);
  static_assert(ipow(NQ1, 3) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 3) == NS, "NS must be tensor-product compatible");
  s_t grad_h_ref_q[NQ * 9 * VS];
  s_t loperand_q[NQ * 9 * VS];
  tensor_gradient<s_t, NQ, NS, VS, 3, 1>(ne, shape_1d, grad_1d, h_streams, 0, &grad_h_ref_q[0]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
    const s_t *const RSTR grad_h_ref0 = &grad_h_ref_q[(3 * q) * VS];
    const s_t *const RSTR grad_h_ref1 = &grad_h_ref_q[(3 * q + 1) * VS];
    const s_t *const RSTR grad_h_ref2 = &grad_h_ref_q[(3 * q + 2) * VS];
    s_t *const RSTR loperand0 = &loperand_q[(3 * q) * VS];
    s_t *const RSTR loperand1 = &loperand_q[(3 * q + 1) * VS];
    s_t *const RSTR loperand2 = &loperand_q[(3 * q + 2) * VS];
    const s_t *const RSTR adj_q0 = adj0 + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adj1 + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adj2 + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adj3 + q * geometry_stride;
    const s_t *const RSTR adj_q4 = adj4 + q * geometry_stride;
    const s_t *const RSTR adj_q5 = adj5 + q * geometry_stride;
    const s_t *const RSTR adj_q6 = adj6 + q * geometry_stride;
    const s_t *const RSTR adj_q7 = adj7 + q * geometry_stride;
    const s_t *const RSTR adj_q8 = adj8 + q * geometry_stride;
    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;
    {
      const s_t adj_value0 = adj_q0[0];
      const s_t adj_value1 = adj_q1[0];
      const s_t adj_value2 = adj_q2[0];
      const s_t adj_value3 = adj_q3[0];
      const s_t adj_value4 = adj_q4[0];
      const s_t adj_value5 = adj_q5[0];
      const s_t adj_value6 = adj_q6[0];
      const s_t adj_value7 = adj_q7[0];
      const s_t adj_value8 = adj_q8[0];
      const s_t det_value0 = det_q0[0];
      s_t trial_grad[3];
      const s_t idet = s_t(1) / det_value0;
      trial_grad[0] = (grad_h_ref0[0] * adj_value0 + grad_h_ref1[0] * adj_value3 + grad_h_ref2[0] * adj_value6) * idet;
      trial_grad[1] = (grad_h_ref0[0] * adj_value1 + grad_h_ref1[0] * adj_value4 + grad_h_ref2[0] * adj_value7) * idet;
      trial_grad[2] = (grad_h_ref0[0] * adj_value2 + grad_h_ref1[0] * adj_value5 + grad_h_ref2[0] * adj_value8) * idet;
      s_t loperand[3];
    s_t material[3];
    material[0] = kappa*trial_grad[0];
    material[1] = kappa*trial_grad[1];
    material[2] = kappa*trial_grad[2];
    loperand[0] = qw * (material[0] * adj_value0 + material[1] * adj_value1 + material[2] * adj_value2);
    loperand[1] = qw * (material[0] * adj_value3 + material[1] * adj_value4 + material[2] * adj_value5);
    loperand[2] = qw * (material[0] * adj_value6 + material[1] * adj_value7 + material[2] * adj_value8);
      loperand0[0] = loperand[0];
      loperand1[0] = loperand[1];
      loperand2[0] = loperand[2];
    }
  }
  tensor_test<s_t, NQ, NS, VS, 3, 1>(ne, shape_1d, grad_1d, &loperand_q[0], out_streams, 0);
}

} // namespace codegen
} // namespace sfem

#endif
