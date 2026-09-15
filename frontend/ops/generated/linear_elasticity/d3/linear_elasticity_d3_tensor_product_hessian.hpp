#ifndef LINEAR_ELASTICITY_D3_TENSOR_PRODUCT_HESSIAN_HPP
#define LINEAR_ELASTICITY_D3_TENSOR_PRODUCT_HESSIAN_HPP
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
static SFEM_INLINE void linear_elasticity_d3_tensor_product_direct_hessian_tensor_product_element_matrix(
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
    const s_t lmbda,
    const s_t mu,
    s_t *const RSTR element_matrix
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(NS > 0, "NS must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NC = 3;
  static constexpr int ND = 3;
  static constexpr int NDOFS = NC * NS;
  static constexpr int NQ1 = integer_root(NQ, 3);
  static constexpr int NS1 = integer_root(NS, 3);
  static_assert(ipow(NQ1, 3) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 3) == NS, "NS must be tensor-product compatible");
  for (int entry = 0; entry < NDOFS * NDOFS; ++entry) {
    element_matrix[entry] = s_t(0);
  }
  s_t flux[NC * NQ * ND];
  s_t *column[NC * NS];
  for (int trial_component = 0; trial_component < NC; ++trial_component) {
    for (int trial_shape = 0; trial_shape < NS; ++trial_shape) {
      const int trial_sx = trial_shape % NS1;
      const int trial_sy = (trial_shape / NS1) % NS1;
      const int trial_sz = trial_shape / (NS1 * NS1);
      for (int q = 0; q < NQ; ++q) {
        const int qx = q % NQ1;
        const int qy = (q / NQ1) % NQ1;
        const int qz = q / (NQ1 * NQ1);
        const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
        const int lane = 0;
        const ptrdiff_t goff = q * VS + lane;
        const s_t adj_lane0 = badj0[goff];
        const s_t adj_lane1 = badj1[goff];
        const s_t adj_lane2 = badj2[goff];
        const s_t adj_lane3 = badj3[goff];
        const s_t adj_lane4 = badj4[goff];
        const s_t adj_lane5 = badj5[goff];
        const s_t adj_lane6 = badj6[goff];
        const s_t adj_lane7 = badj7[goff];
        const s_t adj_lane8 = badj8[goff];
        const s_t det_lane0 = bdet0[goff];
        const s_t idet = s_t(1) / det_lane0;
        const s_t trial_grad_ref0 = grad_1d[qx * NS1 + trial_sx] * shape_1d[qy * NS1 + trial_sy] * shape_1d[qz * NS1 + trial_sz];
        const s_t trial_grad_ref1 = shape_1d[qx * NS1 + trial_sx] * grad_1d[qy * NS1 + trial_sy] * shape_1d[qz * NS1 + trial_sz];
        const s_t trial_grad_ref2 = shape_1d[qx * NS1 + trial_sx] * shape_1d[qy * NS1 + trial_sy] * grad_1d[qz * NS1 + trial_sz];
        s_t trial_grad[NC * ND];
        for (int i = 0; i < NC * ND; ++i) {
          trial_grad[i] = s_t(0);
        }
        trial_grad[trial_component * ND + 0] = (trial_grad_ref0 * adj_lane0 + trial_grad_ref1 * adj_lane3 + trial_grad_ref2 * adj_lane6) * idet;
        trial_grad[trial_component * ND + 1] = (trial_grad_ref0 * adj_lane1 + trial_grad_ref1 * adj_lane4 + trial_grad_ref2 * adj_lane7) * idet;
        trial_grad[trial_component * ND + 2] = (trial_grad_ref0 * adj_lane2 + trial_grad_ref1 * adj_lane5 + trial_grad_ref2 * adj_lane8) * idet;
        s_t material[NC * ND];
        const s_t weak_hess_tmp0 = s_t(2)*trial_grad[0];
        const s_t weak_hess_tmp1 = s_t(2)*trial_grad[4];
        const s_t weak_hess_tmp2 = s_t(2)*trial_grad[8];
        const s_t weak_hess_tmp3 = ((s_t(1) / s_t(2)))*lmbda*(weak_hess_tmp0 + weak_hess_tmp1 + weak_hess_tmp2);
        const s_t weak_hess_tmp4 = mu*(trial_grad[1] + trial_grad[3]);
        const s_t weak_hess_tmp5 = mu*(trial_grad[2] + trial_grad[6]);
        const s_t weak_hess_tmp6 = mu*(trial_grad[5] + trial_grad[7]);
        material[0] = mu*weak_hess_tmp0 + weak_hess_tmp3;
        material[1] = weak_hess_tmp4;
        material[2] = weak_hess_tmp5;
        material[3] = weak_hess_tmp4;
        material[4] = mu*weak_hess_tmp1 + weak_hess_tmp3;
        material[5] = weak_hess_tmp6;
        material[6] = weak_hess_tmp5;
        material[7] = weak_hess_tmp6;
        material[8] = mu*weak_hess_tmp2 + weak_hess_tmp3;
        flux[q * ND] = qw * (material[0] * adj_lane0 + material[1] * adj_lane1 + material[2] * adj_lane2);
        flux[q * ND + 1] = qw * (material[0] * adj_lane3 + material[1] * adj_lane4 + material[2] * adj_lane5);
        flux[q * ND + 2] = qw * (material[0] * adj_lane6 + material[1] * adj_lane7 + material[2] * adj_lane8);
        flux[(NQ + q) * ND] = qw * (material[ND] * adj_lane0 + material[ND + 1] * adj_lane1 + material[ND + 2] * adj_lane2);
        flux[(NQ + q) * ND + 1] = qw * (material[ND] * adj_lane3 + material[ND + 1] * adj_lane4 + material[ND + 2] * adj_lane5);
        flux[(NQ + q) * ND + 2] = qw * (material[ND] * adj_lane6 + material[ND + 1] * adj_lane7 + material[ND + 2] * adj_lane8);
        flux[(2 * NQ + q) * ND] = qw * (material[2 * ND] * adj_lane0 + material[2 * ND + 1] * adj_lane1 + material[2 * ND + 2] * adj_lane2);
        flux[(2 * NQ + q) * ND + 1] = qw * (material[2 * ND] * adj_lane3 + material[2 * ND + 1] * adj_lane4 + material[2 * ND + 2] * adj_lane5);
        flux[(2 * NQ + q) * ND + 2] = qw * (material[2 * ND] * adj_lane6 + material[2 * ND + 1] * adj_lane7 + material[2 * ND + 2] * adj_lane8);
      }
      for (int out_shape = 0; out_shape < NS; ++out_shape) {
        column[out_shape * NC + 0] = &element_matrix[(0 * NS + out_shape) * NDOFS + trial_component * NS + trial_shape];
        column[out_shape * NC + 1] = &element_matrix[(1 * NS + out_shape) * NDOFS + trial_component * NS + trial_shape];
        column[out_shape * NC + 2] = &element_matrix[(2 * NS + out_shape) * NDOFS + trial_component * NS + trial_shape];
      }
      tensor_test_scalar<s_t, NQ, NS, VS, 3, NC>(
          shape_1d, grad_1d, flux + 0, column, 0);
      tensor_test_scalar<s_t, NQ, NS, VS, 3, NC>(
          shape_1d, grad_1d, flux + NQ * ND, column, 1);
      tensor_test_scalar<s_t, NQ, NS, VS, 3, NC>(
          shape_1d, grad_1d, flux + 2 * NQ * ND, column, 2);
    }
  }
}

} // namespace codegen
} // namespace sfem

#endif
