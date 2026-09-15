#ifndef LINEAR_ELASTICITY_D2_SIMPLEX_HESSIAN_HPP
#define LINEAR_ELASTICITY_D2_SIMPLEX_HESSIAN_HPP
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
static SFEM_INLINE void linear_elasticity_d2_simplex_direct_hessian_reference_element_matrix(
    const s_t *const RSTR badj0,
    const s_t *const RSTR badj1,
    const s_t *const RSTR badj2,
    const s_t *const RSTR badj3,
    const s_t *const RSTR bdet0,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
    const s_t lmbda,
    const s_t mu,
    s_t *const RSTR element_matrix
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(NS > 0, "NS must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NC = 2;
  static constexpr int ND = 2;
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
    const s_t det_lane0 = bdet0[goff];
    const s_t idet = s_t(1) / det_lane0;
    for (int trial_component = 0; trial_component < NC; ++trial_component) {
      for (int trial_shape = 0; trial_shape < NS; ++trial_shape) {
        const s_t trial_grad_ref0 = grad_ref_x[q * NS + trial_shape];
        const s_t trial_grad_ref1 = grad_ref_y[q * NS + trial_shape];
        s_t trial_grad[NC * ND];
        for (int i = 0; i < NC * ND; ++i) {
          trial_grad[i] = s_t(0);
        }
        trial_grad[trial_component * ND + 0] = (trial_grad_ref0 * adj_lane0 + trial_grad_ref1 * adj_lane2) * idet;
        trial_grad[trial_component * ND + 1] = (trial_grad_ref0 * adj_lane1 + trial_grad_ref1 * adj_lane3) * idet;
        s_t material[NC * ND];
        const s_t weak_hess_tmp0 = s_t(2)*trial_grad[0];
        const s_t weak_hess_tmp1 = s_t(2)*trial_grad[3];
        const s_t weak_hess_tmp2 = ((s_t(1) / s_t(2)))*lmbda*(weak_hess_tmp0 + weak_hess_tmp1);
        const s_t weak_hess_tmp3 = mu*(trial_grad[1] + trial_grad[2]);
        material[0] = mu*weak_hess_tmp0 + weak_hess_tmp2;
        material[1] = weak_hess_tmp3;
        material[2] = weak_hess_tmp3;
        material[3] = mu*weak_hess_tmp1 + weak_hess_tmp2;
        for (int test_component = 0; test_component < NC; ++test_component) {
          for (int test_shape = 0; test_shape < NS; ++test_shape) {
            const s_t test_grad_ref0 = grad_ref_x[q * NS + test_shape];
            const s_t test_grad_ref1 = grad_ref_y[q * NS + test_shape];
            s_t entry = s_t(0);
            entry += test_grad_ref0 * qw * (material[test_component * ND] * adj_lane0 + material[test_component * ND + 1] * adj_lane1);
            entry += test_grad_ref1 * qw * (material[test_component * ND] * adj_lane2 + material[test_component * ND + 1] * adj_lane3);
            const int row = test_component * NS + test_shape;
            const int col = trial_component * NS + trial_shape;
            element_matrix[row * NDOFS + col] += entry;
          }
        }
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void linear_elasticity_d2_simplex_tri3_direct_hessian_element_matrix(
    const s_t *const RSTR badj0,
    const s_t *const RSTR badj1,
    const s_t *const RSTR badj2,
    const s_t *const RSTR badj3,
    const s_t *const RSTR bdet0,
    const s_t lmbda,
    const s_t mu,
    s_t *const RSTR element_matrix
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(NS > 0, "NS must be positive");
  static_assert(VS > 0, "VS must be positive");
  const int lane = 0;
  const ptrdiff_t goff = 0 * VS + lane;
  const s_t adj_lane0 = badj0[goff];
  const s_t adj_lane1 = badj1[goff];
  const s_t adj_lane2 = badj2[goff];
  const s_t adj_lane3 = badj3[goff];
  const s_t det_lane0 = bdet0[goff];
  const s_t idet = s_t(1) / det_lane0;
  const s_t hessian_tmp0 = -adj_lane1 - adj_lane3;
  const s_t hessian_tmp1 = idet*mu;
  const s_t hessian_tmp2 = ((s_t(1) / s_t(2)))*hessian_tmp1;
  const s_t hessian_tmp3 = hessian_tmp0*hessian_tmp2;
  const s_t hessian_tmp4 = -adj_lane0 - adj_lane2;
  const s_t hessian_tmp5 = idet*lmbda;
  const s_t hessian_tmp6 = hessian_tmp4*hessian_tmp5;
  const s_t hessian_tmp7 = s_t(2)*hessian_tmp1;
  const s_t hessian_tmp8 = ((s_t(1) / s_t(2)))*hessian_tmp4*hessian_tmp7 + ((s_t(1) / s_t(2)))*hessian_tmp6;
  const s_t hessian_tmp9 = adj_lane0*hessian_tmp8 + adj_lane1*hessian_tmp3;
  const s_t hessian_tmp10 = adj_lane2*hessian_tmp8 + adj_lane3*hessian_tmp3;
  const s_t hessian_tmp11 = ((s_t(1) / s_t(2)))*hessian_tmp6;
  const s_t hessian_tmp12 = adj_lane0*hessian_tmp3 + adj_lane1*hessian_tmp11;
  const s_t hessian_tmp13 = adj_lane2*hessian_tmp3 + adj_lane3*hessian_tmp11;
  const s_t hessian_tmp14 = adj_lane0*hessian_tmp5;
  const s_t hessian_tmp15 = ((s_t(1) / s_t(2)))*adj_lane0*hessian_tmp7 + ((s_t(1) / s_t(2)))*hessian_tmp14;
  const s_t hessian_tmp16 = adj_lane0*hessian_tmp15 + pow_2(adj_lane1)*hessian_tmp2;
  const s_t hessian_tmp17 = adj_lane1*hessian_tmp2;
  const s_t hessian_tmp18 = adj_lane3*hessian_tmp17;
  const s_t hessian_tmp19 = adj_lane2*hessian_tmp15 + hessian_tmp18;
  const s_t hessian_tmp20 = ((s_t(1) / s_t(2)))*hessian_tmp14;
  const s_t hessian_tmp21 = adj_lane0*hessian_tmp17 + adj_lane1*hessian_tmp20;
  const s_t hessian_tmp22 = adj_lane2*hessian_tmp17 + adj_lane3*hessian_tmp20;
  const s_t hessian_tmp23 = adj_lane2*hessian_tmp5;
  const s_t hessian_tmp24 = ((s_t(1) / s_t(2)))*adj_lane2*hessian_tmp7 + ((s_t(1) / s_t(2)))*hessian_tmp23;
  const s_t hessian_tmp25 = adj_lane0*hessian_tmp24 + hessian_tmp18;
  const s_t hessian_tmp26 = adj_lane2*hessian_tmp24 + pow_2(adj_lane3)*hessian_tmp2;
  const s_t hessian_tmp27 = adj_lane3*hessian_tmp2;
  const s_t hessian_tmp28 = ((s_t(1) / s_t(2)))*hessian_tmp23;
  const s_t hessian_tmp29 = adj_lane0*hessian_tmp27 + adj_lane1*hessian_tmp28;
  const s_t hessian_tmp30 = adj_lane2*hessian_tmp27 + adj_lane3*hessian_tmp28;
  const s_t hessian_tmp31 = hessian_tmp0*hessian_tmp20 + hessian_tmp17*hessian_tmp4;
  const s_t hessian_tmp32 = hessian_tmp0*hessian_tmp28 + hessian_tmp27*hessian_tmp4;
  const s_t hessian_tmp33 = hessian_tmp2*hessian_tmp4;
  const s_t hessian_tmp34 = ((s_t(1) / s_t(2)))*hessian_tmp0*hessian_tmp5 + ((s_t(1) / s_t(2)))*hessian_tmp0*hessian_tmp7;
  const s_t hessian_tmp35 = adj_lane0*hessian_tmp33 + adj_lane1*hessian_tmp34;
  const s_t hessian_tmp36 = adj_lane2*hessian_tmp33 + adj_lane3*hessian_tmp34;
  const s_t hessian_tmp37 = ((s_t(1) / s_t(2)))*adj_lane1*hessian_tmp5 + ((s_t(1) / s_t(2)))*adj_lane1*hessian_tmp7;
  const s_t hessian_tmp38 = pow_2(adj_lane0)*hessian_tmp2 + adj_lane1*hessian_tmp37;
  const s_t hessian_tmp39 = adj_lane0*adj_lane2*hessian_tmp2;
  const s_t hessian_tmp40 = adj_lane3*hessian_tmp37 + hessian_tmp39;
  const s_t hessian_tmp41 = ((s_t(1) / s_t(2)))*adj_lane3*hessian_tmp5 + ((s_t(1) / s_t(2)))*adj_lane3*hessian_tmp7;
  const s_t hessian_tmp42 = adj_lane1*hessian_tmp41 + hessian_tmp39;
  const s_t hessian_tmp43 = pow_2(adj_lane2)*hessian_tmp2 + adj_lane3*hessian_tmp41;
  element_matrix[0] = -hessian_tmp10 - hessian_tmp9;
  element_matrix[6] = hessian_tmp9;
  element_matrix[12] = hessian_tmp10;
  element_matrix[18] = -hessian_tmp12 - hessian_tmp13;
  element_matrix[24] = hessian_tmp12;
  element_matrix[30] = hessian_tmp13;
  element_matrix[1] = -hessian_tmp16 - hessian_tmp19;
  element_matrix[7] = hessian_tmp16;
  element_matrix[13] = hessian_tmp19;
  element_matrix[19] = -hessian_tmp21 - hessian_tmp22;
  element_matrix[25] = hessian_tmp21;
  element_matrix[31] = hessian_tmp22;
  element_matrix[2] = -hessian_tmp25 - hessian_tmp26;
  element_matrix[8] = hessian_tmp25;
  element_matrix[14] = hessian_tmp26;
  element_matrix[20] = -hessian_tmp29 - hessian_tmp30;
  element_matrix[26] = hessian_tmp29;
  element_matrix[32] = hessian_tmp30;
  element_matrix[3] = -hessian_tmp31 - hessian_tmp32;
  element_matrix[9] = hessian_tmp31;
  element_matrix[15] = hessian_tmp32;
  element_matrix[21] = -hessian_tmp35 - hessian_tmp36;
  element_matrix[27] = hessian_tmp35;
  element_matrix[33] = hessian_tmp36;
  element_matrix[4] = -hessian_tmp21 - hessian_tmp29;
  element_matrix[10] = hessian_tmp21;
  element_matrix[16] = hessian_tmp29;
  element_matrix[22] = -hessian_tmp38 - hessian_tmp40;
  element_matrix[28] = hessian_tmp38;
  element_matrix[34] = hessian_tmp40;
  element_matrix[5] = -hessian_tmp22 - hessian_tmp30;
  element_matrix[11] = hessian_tmp22;
  element_matrix[17] = hessian_tmp30;
  element_matrix[23] = -hessian_tmp42 - hessian_tmp43;
  element_matrix[29] = hessian_tmp42;
  element_matrix[35] = hessian_tmp43;
}

} // namespace codegen
} // namespace sfem

#endif
