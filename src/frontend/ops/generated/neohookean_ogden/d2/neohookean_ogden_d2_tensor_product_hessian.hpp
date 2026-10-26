#ifndef NEOHOOKEAN_OGDEN_D2_TENSOR_PRODUCT_HESSIAN_HPP
#define NEOHOOKEAN_OGDEN_D2_TENSOR_PRODUCT_HESSIAN_HPP
#include <math.h>
#include <stddef.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_GENERATED_SCALAR_T
#endif
#endif
#include "kernel_math.hpp"
#include "tensor_product_kernels.hpp"
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

template <typename s_t, int NQ, int NS>
static SFEM_INLINE void neohookean_ogden_d2_tensor_product_direct_hessian_tensor_product_element_matrix(
    const s_t *const RSTR badj0,
    const s_t *const RSTR badj1,
    const s_t *const RSTR badj2,
    const s_t *const RSTR badj3,
    const s_t *const RSTR bdet0,
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR q_weight_1d,
    const s_t lmbda,
    const s_t mu,
    const s_t bu_data[NS * 2],
    s_t *const RSTR element_matrix
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(NS > 0, "NS must be positive");
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NDOFS = NC * NS;
  static constexpr int NQ1 = integer_root(NQ, 2);
  static constexpr int NS1 = integer_root(NS, 2);
  static_assert(ipow(NQ1, 2) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 2) == NS, "NS must be tensor-product compatible");
  for (int entry = 0; entry < NDOFS * NDOFS; ++entry) {
    element_matrix[entry] = s_t(0);
  }
  s_t state_gradient_ref[NC * NQ * ND];
  for (int component = 0; component < NC; ++component) {
    tensor_gradient_contiguous_scalar<s_t, NQ, NS, 2, NC>(
        shape_1d, grad_1d, bu_data, component,
        state_gradient_ref + component * NQ * ND);
  }
  auto point_tangent = [&](const int q, s_t *const RSTR D, s_t *const RSTR adj_q, s_t &weight) {
    const int qx = q % NQ1;
    const int qy = q / NQ1;
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
    const ptrdiff_t goff = q;
    const s_t adj_lane0 = badj0[goff];
    const s_t adj_lane1 = badj1[goff];
    const s_t adj_lane2 = badj2[goff];
    const s_t adj_lane3 = badj3[goff];
    const s_t det_lane0 = bdet0[goff];
    const s_t idet = s_t(1) / det_lane0;
    const s_t gu_ref0 = state_gradient_ref[q * ND];
    const s_t gu_ref1 = state_gradient_ref[q * ND + 1];
    const s_t gu_ref2 = state_gradient_ref[NQ * ND + q * ND];
    const s_t gu_ref3 = state_gradient_ref[NQ * ND + q * ND + 1];
    const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane2) * idet;
    const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane3) * idet;
    const s_t gu2 = (gu_ref2 * adj_lane0 + gu_ref3 * adj_lane2) * idet;
    const s_t gu3 = (gu_ref2 * adj_lane1 + gu_ref3 * adj_lane3) * idet;
    const s_t weak_hess_tmp0 = gu3 + s_t(1);
    const s_t weak_hess_tmp1 = gu1*gu2;
    const s_t weak_hess_tmp2 = gu0 + s_t(1);
    const s_t weak_hess_tmp3 = weak_hess_tmp0*weak_hess_tmp2 - weak_hess_tmp1;
    const s_t weak_hess_tmp4 = pow_m2(weak_hess_tmp3);
    const s_t weak_hess_tmp5 = pow_2(weak_hess_tmp0)*weak_hess_tmp4;
    const s_t weak_hess_tmp6 = lmbda*weak_hess_tmp5;
    const s_t weak_hess_tmp7 = sfem_log1p(gu0*gu3 + gu0 + gu3 - weak_hess_tmp1);
    const s_t weak_hess_tmp8 = weak_hess_tmp0*weak_hess_tmp4;
    const s_t weak_hess_tmp9 = gu2*weak_hess_tmp8;
    const s_t weak_hess_tmp10 = gu1*weak_hess_tmp8;
    const s_t weak_hess_tmp11 = pow_m1(weak_hess_tmp3);
    const s_t weak_hess_tmp12 = mu*weak_hess_tmp11;
    const s_t weak_hess_tmp13 = weak_hess_tmp0*weak_hess_tmp2*weak_hess_tmp4;
    const s_t weak_hess_tmp14 = lmbda*weak_hess_tmp7;
    const s_t weak_hess_tmp15 = weak_hess_tmp11*weak_hess_tmp14;
    const s_t weak_hess_tmp16 = pow_2(gu2)*weak_hess_tmp4;
    const s_t weak_hess_tmp17 = weak_hess_tmp1*weak_hess_tmp4;
    const s_t weak_hess_tmp18 = weak_hess_tmp2*weak_hess_tmp4;
    const s_t weak_hess_tmp19 = gu2*weak_hess_tmp18;
    const s_t weak_hess_tmp20 = pow_2(gu1)*weak_hess_tmp4;
    const s_t weak_hess_tmp21 = gu1*weak_hess_tmp18;
    const s_t weak_hess_tmp22 = pow_2(weak_hess_tmp2)*weak_hess_tmp4;
    D[0] = mu*weak_hess_tmp5 + mu - weak_hess_tmp6*weak_hess_tmp7 + weak_hess_tmp6;
    D[1] = gu2*lmbda*weak_hess_tmp0*weak_hess_tmp4*weak_hess_tmp7 - lmbda*weak_hess_tmp9 - mu*weak_hess_tmp9;
    D[2] = gu1*lmbda*weak_hess_tmp0*weak_hess_tmp4*weak_hess_tmp7 - lmbda*weak_hess_tmp10 - mu*weak_hess_tmp10;
    D[3] = lmbda*weak_hess_tmp13 + mu*weak_hess_tmp13 - weak_hess_tmp12 - weak_hess_tmp13*weak_hess_tmp14 + weak_hess_tmp15;
    D[5] = lmbda*weak_hess_tmp16 + mu*weak_hess_tmp16 + mu - weak_hess_tmp14*weak_hess_tmp16;
    D[6] = lmbda*weak_hess_tmp17 + mu*weak_hess_tmp17 + weak_hess_tmp12 - weak_hess_tmp14*weak_hess_tmp17 - weak_hess_tmp15;
    D[7] = gu2*lmbda*weak_hess_tmp2*weak_hess_tmp4*weak_hess_tmp7 - lmbda*weak_hess_tmp19 - mu*weak_hess_tmp19;
    D[10] = lmbda*weak_hess_tmp20 + mu*weak_hess_tmp20 + mu - weak_hess_tmp14*weak_hess_tmp20;
    D[11] = gu1*lmbda*weak_hess_tmp2*weak_hess_tmp4*weak_hess_tmp7 - lmbda*weak_hess_tmp21 - mu*weak_hess_tmp21;
    D[15] = lmbda*weak_hess_tmp22 + mu*weak_hess_tmp22 + mu - weak_hess_tmp14*weak_hess_tmp22;
    D[4] = D[1];
    D[8] = D[2];
    D[12] = D[3];
    D[9] = D[6];
    D[13] = D[7];
    D[14] = D[11];
    adj_q[0] = adj_lane0;
    adj_q[1] = adj_lane1;
    adj_q[2] = adj_lane2;
    adj_q[3] = adj_lane3;
    weight = qw * idet;
  };
  if constexpr (NS1 == 2) {
    for (int q = 0; q < NQ; ++q) {
      s_t D[16], adj_q[ND * ND], weight;
      point_tangent(q, D, adj_q, weight);
      s_t g_ref[ND * NS];
      tensor_point_reference_gradients_scalar<s_t, NQ, NS, ND>(shape_1d, grad_1d, q, g_ref);
      point_test_symmetric_scalar<s_t, NS, ND, NC>(g_ref, adj_q, weight, D, element_matrix);
    }
    mirror_upper_triangle_scalar<s_t, NDOFS>(element_matrix);
  } else {
    static constexpr int NP = NQ1;
    s_t slab_tangent[NP * 16];
    s_t flux[NC * ND * NP];
    for (int q_last = 0; q_last < NQ1; ++q_last) {
      for (int p = 0; p < NP; ++p) {
        s_t D[16], adj_q[ND * ND], weight;
        point_tangent(p + NP * q_last, D, adj_q, weight);
        const s_t pullback_weight = weight;
        const s_t pullback_0_0_0_0 = adj_q[0] * D[0] + adj_q[1] * D[4];
        const s_t pullback_0_0_0_1 = adj_q[0] * D[1] + adj_q[1] * D[5];
        const s_t pullback_0_0_1_0 = adj_q[2] * D[0] + adj_q[3] * D[4];
        const s_t pullback_0_0_1_1 = adj_q[2] * D[1] + adj_q[3] * D[5];
        slab_tangent[p * 16 + 0] = pullback_weight * (pullback_0_0_0_0 * adj_q[0] + pullback_0_0_0_1 * adj_q[1]);
        slab_tangent[p * 16 + 1] = pullback_weight * (pullback_0_0_0_0 * adj_q[2] + pullback_0_0_0_1 * adj_q[3]);
        slab_tangent[p * 16 + 4] = pullback_weight * (pullback_0_0_1_0 * adj_q[0] + pullback_0_0_1_1 * adj_q[1]);
        slab_tangent[p * 16 + 5] = pullback_weight * (pullback_0_0_1_0 * adj_q[2] + pullback_0_0_1_1 * adj_q[3]);
        const s_t pullback_0_1_0_0 = adj_q[0] * D[2] + adj_q[1] * D[6];
        const s_t pullback_0_1_0_1 = adj_q[0] * D[3] + adj_q[1] * D[7];
        const s_t pullback_0_1_1_0 = adj_q[2] * D[2] + adj_q[3] * D[6];
        const s_t pullback_0_1_1_1 = adj_q[2] * D[3] + adj_q[3] * D[7];
        slab_tangent[p * 16 + 2] = pullback_weight * (pullback_0_1_0_0 * adj_q[0] + pullback_0_1_0_1 * adj_q[1]);
        slab_tangent[p * 16 + 3] = pullback_weight * (pullback_0_1_0_0 * adj_q[2] + pullback_0_1_0_1 * adj_q[3]);
        slab_tangent[p * 16 + 6] = pullback_weight * (pullback_0_1_1_0 * adj_q[0] + pullback_0_1_1_1 * adj_q[1]);
        slab_tangent[p * 16 + 7] = pullback_weight * (pullback_0_1_1_0 * adj_q[2] + pullback_0_1_1_1 * adj_q[3]);
        const s_t pullback_1_0_0_0 = adj_q[0] * D[8] + adj_q[1] * D[12];
        const s_t pullback_1_0_0_1 = adj_q[0] * D[9] + adj_q[1] * D[13];
        const s_t pullback_1_0_1_0 = adj_q[2] * D[8] + adj_q[3] * D[12];
        const s_t pullback_1_0_1_1 = adj_q[2] * D[9] + adj_q[3] * D[13];
        slab_tangent[p * 16 + 8] = pullback_weight * (pullback_1_0_0_0 * adj_q[0] + pullback_1_0_0_1 * adj_q[1]);
        slab_tangent[p * 16 + 9] = pullback_weight * (pullback_1_0_0_0 * adj_q[2] + pullback_1_0_0_1 * adj_q[3]);
        slab_tangent[p * 16 + 12] = pullback_weight * (pullback_1_0_1_0 * adj_q[0] + pullback_1_0_1_1 * adj_q[1]);
        slab_tangent[p * 16 + 13] = pullback_weight * (pullback_1_0_1_0 * adj_q[2] + pullback_1_0_1_1 * adj_q[3]);
        const s_t pullback_1_1_0_0 = adj_q[0] * D[10] + adj_q[1] * D[14];
        const s_t pullback_1_1_0_1 = adj_q[0] * D[11] + adj_q[1] * D[15];
        const s_t pullback_1_1_1_0 = adj_q[2] * D[10] + adj_q[3] * D[14];
        const s_t pullback_1_1_1_1 = adj_q[2] * D[11] + adj_q[3] * D[15];
        slab_tangent[p * 16 + 10] = pullback_weight * (pullback_1_1_0_0 * adj_q[0] + pullback_1_1_0_1 * adj_q[1]);
        slab_tangent[p * 16 + 11] = pullback_weight * (pullback_1_1_0_0 * adj_q[2] + pullback_1_1_0_1 * adj_q[3]);
        slab_tangent[p * 16 + 14] = pullback_weight * (pullback_1_1_1_0 * adj_q[0] + pullback_1_1_1_1 * adj_q[1]);
        slab_tangent[p * 16 + 15] = pullback_weight * (pullback_1_1_1_0 * adj_q[2] + pullback_1_1_1_1 * adj_q[3]);
      }
      for (int trial_component = 0; trial_component < NC; ++trial_component) {
        for (int trial_shape = 0; trial_shape < NS; ++trial_shape) {
          const int trial_sx = trial_shape % NS1;
          const int trial_sy = trial_shape / NS1;
          const s_t b_last = shape_1d[q_last * NS1 + trial_sy], g_last = grad_1d[q_last * NS1 + trial_sy];
          for (int p = 0; p < NP; ++p) {
            const int qx = p;
            const s_t bx = shape_1d[qx * NS1 + trial_sx], gx = grad_1d[qx * NS1 + trial_sx];
            const s_t r0 = gx * b_last, r1 = bx * g_last;
            const s_t *const RSTR tangent_p = &slab_tangent[p * 16 + trial_component * ND];
            for (int row = 0; row < NC * ND; ++row) {
              flux[row * NP + p] = tangent_p[row * NC * ND + 0] * r0 + tangent_p[row * NC * ND + 1] * r1;
            }
          }
          for (int component = 0; component < NC; ++component) {
            tensor_slab_test_scalar<s_t, NQ1, NS1, ND>(shape_1d, grad_1d, q_last, &flux[component * ND * NP],
                &element_matrix[(trial_component * NS + trial_shape) * NDOFS + component * NS]);
          }
        }
      }
    }
  }
}

} // namespace codegen
} // namespace sfem

#endif
