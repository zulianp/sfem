#ifndef LAPLACE_D3_TENSOR_PRODUCT_HESSIAN_CUH
#define LAPLACE_D3_TENSOR_PRODUCT_HESSIAN_CUH
#include <math.h>
#include <stddef.h>
#if defined(__has_include)
#if __has_include("sfem_base.hpp")
#include "sfem_base.hpp"
#define SFEM_GENERATED_SCALAR_T
#endif
#endif
#include "kernel_math.cuh"
#include "tensor_product_kernels.cuh"
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

template <typename s_t, int NQ, int NS>
static __host__ __device__ __forceinline__ void laplace_d3_tensor_product_direct_hessian_tensor_product_element_matrix(
    const s_t *const RSTR badj0,
    const s_t *const RSTR badj1,
    const s_t *const RSTR badj2,
    const s_t *const RSTR badj3,
    const s_t *const RSTR badj4,
    const s_t *const RSTR badj5,
    const s_t *const RSTR badj6,
    const s_t *const RSTR badj7,
    const s_t *const RSTR badj8,
    const s_t *const RSTR bdet0,
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR q_weight_1d,
    const s_t kappa,
    s_t *const RSTR element_matrix
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(NS > 0, "NS must be positive");
  static constexpr int NC = 1;
  static constexpr int ND = 3;
  static constexpr int NDOFS = NC * NS;
  static constexpr int NQ1 = integer_root(NQ, 3);
  static constexpr int NS1 = integer_root(NS, 3);
  static_assert(ipow(NQ1, 3) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 3) == NS, "NS must be tensor-product compatible");
  for (int entry = 0; entry < NDOFS * NDOFS; ++entry) {
    element_matrix[entry] = s_t(0);
  }
  auto point_tangent = [&](const int q, s_t *const RSTR D, s_t *const RSTR adj_q, s_t &weight) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
    const ptrdiff_t goff = q;
    const s_t adj_value0 = badj0[goff];
    const s_t adj_value1 = badj1[goff];
    const s_t adj_value2 = badj2[goff];
    const s_t adj_value3 = badj3[goff];
    const s_t adj_value4 = badj4[goff];
    const s_t adj_value5 = badj5[goff];
    const s_t adj_value6 = badj6[goff];
    const s_t adj_value7 = badj7[goff];
    const s_t adj_value8 = badj8[goff];
    const s_t det_value0 = bdet0[goff];
    const s_t idet = s_t(1) / det_value0;
    D[0] = kappa;
    D[1] = 0;
    D[2] = 0;
    D[4] = kappa;
    D[5] = 0;
    D[8] = kappa;
    D[3] = D[1];
    D[6] = D[2];
    D[7] = D[5];
    adj_q[0] = adj_value0;
    adj_q[1] = adj_value1;
    adj_q[2] = adj_value2;
    adj_q[3] = adj_value3;
    adj_q[4] = adj_value4;
    adj_q[5] = adj_value5;
    adj_q[6] = adj_value6;
    adj_q[7] = adj_value7;
    adj_q[8] = adj_value8;
    weight = qw * idet;
  };
  if constexpr (NS1 == 2) {
    for (int q = 0; q < NQ; ++q) {
      s_t D[9], adj_q[ND * ND], weight;
      point_tangent(q, D, adj_q, weight);
      s_t g_ref[ND * NS];
      tensor_point_reference_gradients_scalar<s_t, NQ, NS, ND>(shape_1d, grad_1d, q, g_ref);
      point_test_symmetric_scalar<s_t, NS, ND, NC>(g_ref, adj_q, weight, D, element_matrix);
    }
    mirror_upper_triangle_scalar<s_t, NDOFS>(element_matrix);
  } else {
    static constexpr int NP = NQ1 * NQ1;
    s_t slab_tangent[NP * 9];
    s_t flux[NC * ND * NP];
    for (int q_last = 0; q_last < NQ1; ++q_last) {
      for (int p = 0; p < NP; ++p) {
        s_t D[9], adj_q[ND * ND], weight;
        point_tangent(p + NP * q_last, D, adj_q, weight);
        const s_t pullback_weight = weight;
        const s_t pullback_0_0_0_0 = adj_q[0] * D[0] + adj_q[1] * D[3] + adj_q[2] * D[6];
        const s_t pullback_0_0_0_1 = adj_q[0] * D[1] + adj_q[1] * D[4] + adj_q[2] * D[7];
        const s_t pullback_0_0_0_2 = adj_q[0] * D[2] + adj_q[1] * D[5] + adj_q[2] * D[8];
        const s_t pullback_0_0_1_0 = adj_q[3] * D[0] + adj_q[4] * D[3] + adj_q[5] * D[6];
        const s_t pullback_0_0_1_1 = adj_q[3] * D[1] + adj_q[4] * D[4] + adj_q[5] * D[7];
        const s_t pullback_0_0_1_2 = adj_q[3] * D[2] + adj_q[4] * D[5] + adj_q[5] * D[8];
        const s_t pullback_0_0_2_0 = adj_q[6] * D[0] + adj_q[7] * D[3] + adj_q[8] * D[6];
        const s_t pullback_0_0_2_1 = adj_q[6] * D[1] + adj_q[7] * D[4] + adj_q[8] * D[7];
        const s_t pullback_0_0_2_2 = adj_q[6] * D[2] + adj_q[7] * D[5] + adj_q[8] * D[8];
        slab_tangent[p * 9 + 0] = pullback_weight * (pullback_0_0_0_0 * adj_q[0] + pullback_0_0_0_1 * adj_q[1] + pullback_0_0_0_2 * adj_q[2]);
        slab_tangent[p * 9 + 1] = pullback_weight * (pullback_0_0_0_0 * adj_q[3] + pullback_0_0_0_1 * adj_q[4] + pullback_0_0_0_2 * adj_q[5]);
        slab_tangent[p * 9 + 2] = pullback_weight * (pullback_0_0_0_0 * adj_q[6] + pullback_0_0_0_1 * adj_q[7] + pullback_0_0_0_2 * adj_q[8]);
        slab_tangent[p * 9 + 3] = pullback_weight * (pullback_0_0_1_0 * adj_q[0] + pullback_0_0_1_1 * adj_q[1] + pullback_0_0_1_2 * adj_q[2]);
        slab_tangent[p * 9 + 4] = pullback_weight * (pullback_0_0_1_0 * adj_q[3] + pullback_0_0_1_1 * adj_q[4] + pullback_0_0_1_2 * adj_q[5]);
        slab_tangent[p * 9 + 5] = pullback_weight * (pullback_0_0_1_0 * adj_q[6] + pullback_0_0_1_1 * adj_q[7] + pullback_0_0_1_2 * adj_q[8]);
        slab_tangent[p * 9 + 6] = pullback_weight * (pullback_0_0_2_0 * adj_q[0] + pullback_0_0_2_1 * adj_q[1] + pullback_0_0_2_2 * adj_q[2]);
        slab_tangent[p * 9 + 7] = pullback_weight * (pullback_0_0_2_0 * adj_q[3] + pullback_0_0_2_1 * adj_q[4] + pullback_0_0_2_2 * adj_q[5]);
        slab_tangent[p * 9 + 8] = pullback_weight * (pullback_0_0_2_0 * adj_q[6] + pullback_0_0_2_1 * adj_q[7] + pullback_0_0_2_2 * adj_q[8]);
      }
      for (int trial_component = 0; trial_component < NC; ++trial_component) {
        for (int trial_shape = 0; trial_shape < NS; ++trial_shape) {
          const int trial_sx = trial_shape % NS1;
          const int trial_sy = (trial_shape / NS1) % NS1;
          const int trial_sz = trial_shape / (NS1 * NS1);
          const s_t b_last = shape_1d[q_last * NS1 + trial_sz], g_last = grad_1d[q_last * NS1 + trial_sz];
          for (int p = 0; p < NP; ++p) {
            const int qx = p % NQ1, qy = p / NQ1;
            const s_t bx = shape_1d[qx * NS1 + trial_sx], gx = grad_1d[qx * NS1 + trial_sx];
            const s_t by = shape_1d[qy * NS1 + trial_sy], gy = grad_1d[qy * NS1 + trial_sy];
            const s_t r0 = gx * by * b_last, r1 = bx * gy * b_last, r2 = bx * by * g_last;
            const s_t *const RSTR tangent_p = &slab_tangent[p * 9 + trial_component * ND];
            for (int row = 0; row < NC * ND; ++row) {
              flux[row * NP + p] = tangent_p[row * NC * ND + 0] * r0 + tangent_p[row * NC * ND + 1] * r1 + tangent_p[row * NC * ND + 2] * r2;
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
