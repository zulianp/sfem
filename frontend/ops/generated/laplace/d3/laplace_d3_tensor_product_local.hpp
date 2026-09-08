#ifndef LAPLACE_D3_TENSOR_PRODUCT_LOCAL_HPP
#define LAPLACE_D3_TENSOR_PRODUCT_LOCAL_HPP
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
typedef ptrdiff_t count_t;
typedef double geom_t;
#endif
namespace sfem {
namespace codegen {

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void laplace_d3_tensor_product_objective_block(
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
        s_t *const RSTR value
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NQ1 = integer_root(NQ, 3);
  static constexpr int NS1 = integer_root(NS, 3);
  static_assert(ipow(NQ1, 3) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 3) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[NQ * 9 * VS];
  tensor_gradient<s_t, NQ, NS, VS, 3, 1>(ne, shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0 * NQ * 3 * VS]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
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
      s_t gu_ref[3];
      gu_ref[0] = gu_ref_q[((0 * NQ + q) * 3 + 0) * VS + lane];
      gu_ref[1] = gu_ref_q[((0 * NQ + q) * 3 + 1) * VS + lane];
      gu_ref[2] = gu_ref_q[((0 * NQ + q) * 3 + 2) * VS + lane];
      s_t gu[3];
      const s_t idet = s_t(1) / det_lane0;
      gu[0] = (gu_ref[0] * adj_lane0 + gu_ref[1] * adj_lane3 + gu_ref[2] * adj_lane6) * idet;
      gu[1] = (gu_ref[0] * adj_lane1 + gu_ref[1] * adj_lane4 + gu_ref[2] * adj_lane7) * idet;
      gu[2] = (gu_ref[0] * adj_lane2 + gu_ref[1] * adj_lane5 + gu_ref[2] * adj_lane8) * idet;
    value[lane] += qw * det_lane0 * (((s_t(1) / s_t(2)))*kappa*(pow_2(gu[0]) + pow_2(gu[1]) + pow_2(gu[2])));
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void laplace_d3_tensor_product_gradient_block(
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
  tensor_gradient<s_t, NQ, NS, VS, 3, 1>(ne, shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0 * NQ * 3 * VS]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
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
      s_t gu_ref[3];
      gu_ref[0] = gu_ref_q[((0 * NQ + q) * 3 + 0) * VS + lane];
      gu_ref[1] = gu_ref_q[((0 * NQ + q) * 3 + 1) * VS + lane];
      gu_ref[2] = gu_ref_q[((0 * NQ + q) * 3 + 2) * VS + lane];
      s_t gu[3];
      const s_t idet = s_t(1) / det_lane0;
      gu[0] = (gu_ref[0] * adj_lane0 + gu_ref[1] * adj_lane3 + gu_ref[2] * adj_lane6) * idet;
      gu[1] = (gu_ref[0] * adj_lane1 + gu_ref[1] * adj_lane4 + gu_ref[2] * adj_lane7) * idet;
      gu[2] = (gu_ref[0] * adj_lane2 + gu_ref[1] * adj_lane5 + gu_ref[2] * adj_lane8) * idet;
      s_t loperand[3];
    s_t material[3];
    material[0] = gu[0]*kappa;
    material[1] = gu[1]*kappa;
    material[2] = gu[2]*kappa;
    loperand[0] = qw * (material[0] * adj_lane0 + material[1] * adj_lane1 + material[2] * adj_lane2);
    loperand[1] = qw * (material[0] * adj_lane3 + material[1] * adj_lane4 + material[2] * adj_lane5);
    loperand[2] = qw * (material[0] * adj_lane6 + material[1] * adj_lane7 + material[2] * adj_lane8);
      loperand_q[((0 * NQ + q) * 3 + 0) * VS + lane] = loperand[0];
      loperand_q[((0 * NQ + q) * 3 + 1) * VS + lane] = loperand[1];
      loperand_q[((0 * NQ + q) * 3 + 2) * VS + lane] = loperand[2];
    }
  }
  tensor_test<s_t, NQ, NS, VS, 3, 1>(ne, shape_1d, grad_1d, &loperand_q[0 * NQ * 3 * VS], out_streams, 0);
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void laplace_d3_tensor_product_apply_block(
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
  tensor_gradient<s_t, NQ, NS, VS, 3, 1>(ne, shape_1d, grad_1d, h_streams, 0, &grad_h_ref_q[0 * NQ * 3 * VS]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
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
      s_t grad_h_ref[3];
      grad_h_ref[0] = grad_h_ref_q[((0 * NQ + q) * 3 + 0) * VS + lane];
      grad_h_ref[1] = grad_h_ref_q[((0 * NQ + q) * 3 + 1) * VS + lane];
      grad_h_ref[2] = grad_h_ref_q[((0 * NQ + q) * 3 + 2) * VS + lane];
      s_t trial_grad[3];
      const s_t idet = s_t(1) / det_lane0;
      trial_grad[0] = (grad_h_ref[0] * adj_lane0 + grad_h_ref[1] * adj_lane3 + grad_h_ref[2] * adj_lane6) * idet;
      trial_grad[1] = (grad_h_ref[0] * adj_lane1 + grad_h_ref[1] * adj_lane4 + grad_h_ref[2] * adj_lane7) * idet;
      trial_grad[2] = (grad_h_ref[0] * adj_lane2 + grad_h_ref[1] * adj_lane5 + grad_h_ref[2] * adj_lane8) * idet;
      s_t loperand[3];
    s_t material[3];
    material[0] = kappa*trial_grad[0];
    material[1] = kappa*trial_grad[1];
    material[2] = kappa*trial_grad[2];
    loperand[0] = qw * (material[0] * adj_lane0 + material[1] * adj_lane1 + material[2] * adj_lane2);
    loperand[1] = qw * (material[0] * adj_lane3 + material[1] * adj_lane4 + material[2] * adj_lane5);
    loperand[2] = qw * (material[0] * adj_lane6 + material[1] * adj_lane7 + material[2] * adj_lane8);
      loperand_q[((0 * NQ + q) * 3 + 0) * VS + lane] = loperand[0];
      loperand_q[((0 * NQ + q) * 3 + 1) * VS + lane] = loperand[1];
      loperand_q[((0 * NQ + q) * 3 + 2) * VS + lane] = loperand[2];
    }
  }
  tensor_test<s_t, NQ, NS, VS, 3, 1>(ne, shape_1d, grad_1d, &loperand_q[0 * NQ * 3 * VS], out_streams, 0);
}

} // namespace codegen
} // namespace sfem

#endif
