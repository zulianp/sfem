#ifndef LINEAR_ELASTICITY_D3_SIMPLEX_HESSIAN_HPP
#define LINEAR_ELASTICITY_D3_SIMPLEX_HESSIAN_HPP
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
static SFEM_INLINE void linear_elasticity_d3_simplex_direct_hessian_reference_element_matrix(
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
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR grad_ref_z,
    const s_t *const RSTR q_weight,
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
  for (int entry = 0; entry < NDOFS * NDOFS; ++entry) {
    element_matrix[entry] = s_t(0);
  }
  for (int q = 0; q < NQ; ++q) {
    const s_t qw = q_weight[q];
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
    for (int trial_component = 0; trial_component < NC; ++trial_component) {
      for (int trial_shape = 0; trial_shape < NS; ++trial_shape) {
        const s_t trial_grad_ref0 = grad_ref_x[q * NS + trial_shape];
        const s_t trial_grad_ref1 = grad_ref_y[q * NS + trial_shape];
        const s_t trial_grad_ref2 = grad_ref_z[q * NS + trial_shape];
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
        for (int test_component = 0; test_component < NC; ++test_component) {
          for (int test_shape = 0; test_shape < NS; ++test_shape) {
            const s_t test_grad_ref0 = grad_ref_x[q * NS + test_shape];
            const s_t test_grad_ref1 = grad_ref_y[q * NS + test_shape];
            const s_t test_grad_ref2 = grad_ref_z[q * NS + test_shape];
            s_t entry = s_t(0);
            entry += test_grad_ref0 * qw * (material[test_component * ND] * adj_lane0 + material[test_component * ND + 1] * adj_lane1 + material[test_component * ND + 2] * adj_lane2);
            entry += test_grad_ref1 * qw * (material[test_component * ND] * adj_lane3 + material[test_component * ND + 1] * adj_lane4 + material[test_component * ND + 2] * adj_lane5);
            entry += test_grad_ref2 * qw * (material[test_component * ND] * adj_lane6 + material[test_component * ND + 1] * adj_lane7 + material[test_component * ND + 2] * adj_lane8);
            const int row = test_component * NS + test_shape;
            const int col = trial_component * NS + trial_shape;
            element_matrix[row * NDOFS + col] += entry;
          }
        }
      }
    }
  }
}

} // namespace codegen
} // namespace sfem

#endif
