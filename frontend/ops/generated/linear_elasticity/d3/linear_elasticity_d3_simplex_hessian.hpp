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
typedef ptrdiff_t element_idx_t;
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

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void linear_elasticity_d3_simplex_tet4_direct_hessian_element_matrix(
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
  const s_t adj_lane4 = badj4[goff];
  const s_t adj_lane5 = badj5[goff];
  const s_t adj_lane6 = badj6[goff];
  const s_t adj_lane7 = badj7[goff];
  const s_t adj_lane8 = badj8[goff];
  const s_t det_lane0 = bdet0[goff];
  const s_t idet = s_t(1) / det_lane0;
  const s_t hessian_tmp0 = -adj_lane1 - adj_lane4 - adj_lane7;
  const s_t hessian_tmp1 = idet*mu;
  const s_t hessian_tmp2 = ((s_t(1) / s_t(6)))*hessian_tmp1;
  const s_t hessian_tmp3 = hessian_tmp0*hessian_tmp2;
  const s_t hessian_tmp4 = adj_lane1*hessian_tmp3;
  const s_t hessian_tmp5 = -adj_lane2 - adj_lane5 - adj_lane8;
  const s_t hessian_tmp6 = hessian_tmp2*hessian_tmp5;
  const s_t hessian_tmp7 = adj_lane2*hessian_tmp6;
  const s_t hessian_tmp8 = -adj_lane0 - adj_lane3 - adj_lane6;
  const s_t hessian_tmp9 = idet*lmbda;
  const s_t hessian_tmp10 = hessian_tmp8*hessian_tmp9;
  const s_t hessian_tmp11 = s_t(2)*hessian_tmp1;
  const s_t hessian_tmp12 = ((s_t(1) / s_t(6)))*hessian_tmp10 + ((s_t(1) / s_t(6)))*hessian_tmp11*hessian_tmp8;
  const s_t hessian_tmp13 = adj_lane0*hessian_tmp12 + hessian_tmp4 + hessian_tmp7;
  const s_t hessian_tmp14 = adj_lane4*hessian_tmp3;
  const s_t hessian_tmp15 = adj_lane5*hessian_tmp6;
  const s_t hessian_tmp16 = adj_lane3*hessian_tmp12 + hessian_tmp14 + hessian_tmp15;
  const s_t hessian_tmp17 = adj_lane7*hessian_tmp3;
  const s_t hessian_tmp18 = adj_lane8*hessian_tmp6;
  const s_t hessian_tmp19 = adj_lane6*hessian_tmp12 + hessian_tmp17 + hessian_tmp18;
  const s_t hessian_tmp20 = ((s_t(1) / s_t(6)))*hessian_tmp10;
  const s_t hessian_tmp21 = adj_lane0*hessian_tmp3 + adj_lane1*hessian_tmp20;
  const s_t hessian_tmp22 = adj_lane3*hessian_tmp3 + adj_lane4*hessian_tmp20;
  const s_t hessian_tmp23 = adj_lane6*hessian_tmp3 + adj_lane7*hessian_tmp20;
  const s_t hessian_tmp24 = adj_lane0*hessian_tmp6 + adj_lane2*hessian_tmp20;
  const s_t hessian_tmp25 = adj_lane3*hessian_tmp6 + adj_lane5*hessian_tmp20;
  const s_t hessian_tmp26 = adj_lane6*hessian_tmp6 + adj_lane8*hessian_tmp20;
  const s_t hessian_tmp27 = pow_2(adj_lane1)*hessian_tmp2;
  const s_t hessian_tmp28 = pow_2(adj_lane2)*hessian_tmp2;
  const s_t hessian_tmp29 = adj_lane0*hessian_tmp9;
  const s_t hessian_tmp30 = ((s_t(1) / s_t(6)))*adj_lane0*hessian_tmp11 + ((s_t(1) / s_t(6)))*hessian_tmp29;
  const s_t hessian_tmp31 = adj_lane0*hessian_tmp30 + hessian_tmp27 + hessian_tmp28;
  const s_t hessian_tmp32 = adj_lane1*hessian_tmp2;
  const s_t hessian_tmp33 = adj_lane4*hessian_tmp32;
  const s_t hessian_tmp34 = adj_lane2*hessian_tmp2;
  const s_t hessian_tmp35 = adj_lane5*hessian_tmp34;
  const s_t hessian_tmp36 = hessian_tmp33 + hessian_tmp35;
  const s_t hessian_tmp37 = adj_lane3*hessian_tmp30 + hessian_tmp36;
  const s_t hessian_tmp38 = adj_lane7*hessian_tmp32;
  const s_t hessian_tmp39 = adj_lane8*hessian_tmp34;
  const s_t hessian_tmp40 = hessian_tmp38 + hessian_tmp39;
  const s_t hessian_tmp41 = adj_lane6*hessian_tmp30 + hessian_tmp40;
  const s_t hessian_tmp42 = ((s_t(1) / s_t(6)))*hessian_tmp29;
  const s_t hessian_tmp43 = adj_lane0*hessian_tmp32 + adj_lane1*hessian_tmp42;
  const s_t hessian_tmp44 = adj_lane3*hessian_tmp32 + adj_lane4*hessian_tmp42;
  const s_t hessian_tmp45 = adj_lane6*hessian_tmp32 + adj_lane7*hessian_tmp42;
  const s_t hessian_tmp46 = adj_lane0*hessian_tmp34 + adj_lane2*hessian_tmp42;
  const s_t hessian_tmp47 = adj_lane3*hessian_tmp34 + adj_lane5*hessian_tmp42;
  const s_t hessian_tmp48 = adj_lane6*hessian_tmp34 + adj_lane8*hessian_tmp42;
  const s_t hessian_tmp49 = adj_lane3*hessian_tmp9;
  const s_t hessian_tmp50 = ((s_t(1) / s_t(6)))*adj_lane3*hessian_tmp11 + ((s_t(1) / s_t(6)))*hessian_tmp49;
  const s_t hessian_tmp51 = adj_lane0*hessian_tmp50 + hessian_tmp36;
  const s_t hessian_tmp52 = pow_2(adj_lane4)*hessian_tmp2;
  const s_t hessian_tmp53 = pow_2(adj_lane5)*hessian_tmp2;
  const s_t hessian_tmp54 = adj_lane3*hessian_tmp50 + hessian_tmp52 + hessian_tmp53;
  const s_t hessian_tmp55 = adj_lane4*hessian_tmp2;
  const s_t hessian_tmp56 = adj_lane7*hessian_tmp55;
  const s_t hessian_tmp57 = adj_lane5*hessian_tmp2;
  const s_t hessian_tmp58 = adj_lane8*hessian_tmp57;
  const s_t hessian_tmp59 = hessian_tmp56 + hessian_tmp58;
  const s_t hessian_tmp60 = adj_lane6*hessian_tmp50 + hessian_tmp59;
  const s_t hessian_tmp61 = ((s_t(1) / s_t(6)))*hessian_tmp49;
  const s_t hessian_tmp62 = adj_lane0*hessian_tmp55 + adj_lane1*hessian_tmp61;
  const s_t hessian_tmp63 = adj_lane3*hessian_tmp55 + adj_lane4*hessian_tmp61;
  const s_t hessian_tmp64 = adj_lane6*hessian_tmp55 + adj_lane7*hessian_tmp61;
  const s_t hessian_tmp65 = adj_lane0*hessian_tmp57 + adj_lane2*hessian_tmp61;
  const s_t hessian_tmp66 = adj_lane3*hessian_tmp57 + adj_lane5*hessian_tmp61;
  const s_t hessian_tmp67 = adj_lane6*hessian_tmp57 + adj_lane8*hessian_tmp61;
  const s_t hessian_tmp68 = adj_lane6*hessian_tmp9;
  const s_t hessian_tmp69 = ((s_t(1) / s_t(6)))*adj_lane6*hessian_tmp11 + ((s_t(1) / s_t(6)))*hessian_tmp68;
  const s_t hessian_tmp70 = adj_lane0*hessian_tmp69 + hessian_tmp40;
  const s_t hessian_tmp71 = adj_lane3*hessian_tmp69 + hessian_tmp59;
  const s_t hessian_tmp72 = pow_2(adj_lane7)*hessian_tmp2;
  const s_t hessian_tmp73 = pow_2(adj_lane8)*hessian_tmp2;
  const s_t hessian_tmp74 = adj_lane6*hessian_tmp69 + hessian_tmp72 + hessian_tmp73;
  const s_t hessian_tmp75 = adj_lane7*hessian_tmp2;
  const s_t hessian_tmp76 = ((s_t(1) / s_t(6)))*hessian_tmp68;
  const s_t hessian_tmp77 = adj_lane0*hessian_tmp75 + adj_lane1*hessian_tmp76;
  const s_t hessian_tmp78 = adj_lane3*hessian_tmp75 + adj_lane4*hessian_tmp76;
  const s_t hessian_tmp79 = adj_lane6*hessian_tmp75 + adj_lane7*hessian_tmp76;
  const s_t hessian_tmp80 = adj_lane8*hessian_tmp2;
  const s_t hessian_tmp81 = adj_lane0*hessian_tmp80 + adj_lane2*hessian_tmp76;
  const s_t hessian_tmp82 = adj_lane3*hessian_tmp80 + adj_lane5*hessian_tmp76;
  const s_t hessian_tmp83 = adj_lane6*hessian_tmp80 + adj_lane8*hessian_tmp76;
  const s_t hessian_tmp84 = hessian_tmp0*hessian_tmp42 + hessian_tmp32*hessian_tmp8;
  const s_t hessian_tmp85 = hessian_tmp0*hessian_tmp61 + hessian_tmp55*hessian_tmp8;
  const s_t hessian_tmp86 = hessian_tmp0*hessian_tmp76 + hessian_tmp75*hessian_tmp8;
  const s_t hessian_tmp87 = hessian_tmp2*hessian_tmp8;
  const s_t hessian_tmp88 = adj_lane0*hessian_tmp87;
  const s_t hessian_tmp89 = hessian_tmp0*hessian_tmp9;
  const s_t hessian_tmp90 = ((s_t(1) / s_t(6)))*hessian_tmp0*hessian_tmp11 + ((s_t(1) / s_t(6)))*hessian_tmp89;
  const s_t hessian_tmp91 = adj_lane1*hessian_tmp90 + hessian_tmp7 + hessian_tmp88;
  const s_t hessian_tmp92 = adj_lane3*hessian_tmp87;
  const s_t hessian_tmp93 = adj_lane4*hessian_tmp90 + hessian_tmp15 + hessian_tmp92;
  const s_t hessian_tmp94 = adj_lane6*hessian_tmp87;
  const s_t hessian_tmp95 = adj_lane7*hessian_tmp90 + hessian_tmp18 + hessian_tmp94;
  const s_t hessian_tmp96 = ((s_t(1) / s_t(6)))*hessian_tmp89;
  const s_t hessian_tmp97 = adj_lane1*hessian_tmp6 + adj_lane2*hessian_tmp96;
  const s_t hessian_tmp98 = adj_lane4*hessian_tmp6 + adj_lane5*hessian_tmp96;
  const s_t hessian_tmp99 = adj_lane7*hessian_tmp6 + adj_lane8*hessian_tmp96;
  const s_t hessian_tmp100 = pow_2(adj_lane0)*hessian_tmp2;
  const s_t hessian_tmp101 = adj_lane1*hessian_tmp9;
  const s_t hessian_tmp102 = ((s_t(1) / s_t(6)))*adj_lane1*hessian_tmp11 + ((s_t(1) / s_t(6)))*hessian_tmp101;
  const s_t hessian_tmp103 = adj_lane1*hessian_tmp102 + hessian_tmp100 + hessian_tmp28;
  const s_t hessian_tmp104 = adj_lane0*hessian_tmp2;
  const s_t hessian_tmp105 = adj_lane3*hessian_tmp104;
  const s_t hessian_tmp106 = hessian_tmp105 + hessian_tmp35;
  const s_t hessian_tmp107 = adj_lane4*hessian_tmp102 + hessian_tmp106;
  const s_t hessian_tmp108 = adj_lane6*hessian_tmp104;
  const s_t hessian_tmp109 = hessian_tmp108 + hessian_tmp39;
  const s_t hessian_tmp110 = adj_lane7*hessian_tmp102 + hessian_tmp109;
  const s_t hessian_tmp111 = ((s_t(1) / s_t(6)))*hessian_tmp101;
  const s_t hessian_tmp112 = adj_lane2*hessian_tmp111 + adj_lane2*hessian_tmp32;
  const s_t hessian_tmp113 = adj_lane4*hessian_tmp34 + adj_lane5*hessian_tmp111;
  const s_t hessian_tmp114 = adj_lane7*hessian_tmp34 + adj_lane8*hessian_tmp111;
  const s_t hessian_tmp115 = adj_lane4*hessian_tmp9;
  const s_t hessian_tmp116 = ((s_t(1) / s_t(6)))*adj_lane4*hessian_tmp11 + ((s_t(1) / s_t(6)))*hessian_tmp115;
  const s_t hessian_tmp117 = adj_lane1*hessian_tmp116 + hessian_tmp106;
  const s_t hessian_tmp118 = pow_2(adj_lane3)*hessian_tmp2;
  const s_t hessian_tmp119 = adj_lane4*hessian_tmp116 + hessian_tmp118 + hessian_tmp53;
  const s_t hessian_tmp120 = adj_lane3*adj_lane6*hessian_tmp2;
  const s_t hessian_tmp121 = hessian_tmp120 + hessian_tmp58;
  const s_t hessian_tmp122 = adj_lane7*hessian_tmp116 + hessian_tmp121;
  const s_t hessian_tmp123 = ((s_t(1) / s_t(6)))*hessian_tmp115;
  const s_t hessian_tmp124 = adj_lane2*hessian_tmp123 + adj_lane5*hessian_tmp32;
  const s_t hessian_tmp125 = adj_lane5*hessian_tmp123 + adj_lane5*hessian_tmp55;
  const s_t hessian_tmp126 = adj_lane7*hessian_tmp57 + adj_lane8*hessian_tmp123;
  const s_t hessian_tmp127 = adj_lane7*hessian_tmp9;
  const s_t hessian_tmp128 = ((s_t(1) / s_t(6)))*adj_lane7*hessian_tmp11 + ((s_t(1) / s_t(6)))*hessian_tmp127;
  const s_t hessian_tmp129 = adj_lane1*hessian_tmp128 + hessian_tmp109;
  const s_t hessian_tmp130 = adj_lane4*hessian_tmp128 + hessian_tmp121;
  const s_t hessian_tmp131 = pow_2(adj_lane6)*hessian_tmp2;
  const s_t hessian_tmp132 = adj_lane7*hessian_tmp128 + hessian_tmp131 + hessian_tmp73;
  const s_t hessian_tmp133 = ((s_t(1) / s_t(6)))*hessian_tmp127;
  const s_t hessian_tmp134 = adj_lane2*hessian_tmp133 + adj_lane8*hessian_tmp32;
  const s_t hessian_tmp135 = adj_lane5*hessian_tmp133 + adj_lane8*hessian_tmp55;
  const s_t hessian_tmp136 = adj_lane8*hessian_tmp133 + adj_lane8*hessian_tmp75;
  const s_t hessian_tmp137 = hessian_tmp34*hessian_tmp8 + hessian_tmp42*hessian_tmp5;
  const s_t hessian_tmp138 = hessian_tmp5*hessian_tmp61 + hessian_tmp57*hessian_tmp8;
  const s_t hessian_tmp139 = hessian_tmp5*hessian_tmp76 + hessian_tmp8*hessian_tmp80;
  const s_t hessian_tmp140 = adj_lane2*hessian_tmp3 + hessian_tmp111*hessian_tmp5;
  const s_t hessian_tmp141 = adj_lane5*hessian_tmp3 + hessian_tmp123*hessian_tmp5;
  const s_t hessian_tmp142 = adj_lane8*hessian_tmp3 + hessian_tmp133*hessian_tmp5;
  const s_t hessian_tmp143 = ((s_t(1) / s_t(6)))*hessian_tmp11*hessian_tmp5 + ((s_t(1) / s_t(6)))*hessian_tmp5*hessian_tmp9;
  const s_t hessian_tmp144 = adj_lane2*hessian_tmp143 + hessian_tmp4 + hessian_tmp88;
  const s_t hessian_tmp145 = adj_lane5*hessian_tmp143 + hessian_tmp14 + hessian_tmp92;
  const s_t hessian_tmp146 = adj_lane8*hessian_tmp143 + hessian_tmp17 + hessian_tmp94;
  const s_t hessian_tmp147 = ((s_t(1) / s_t(6)))*adj_lane2*hessian_tmp11 + ((s_t(1) / s_t(6)))*adj_lane2*hessian_tmp9;
  const s_t hessian_tmp148 = adj_lane2*hessian_tmp147 + hessian_tmp100 + hessian_tmp27;
  const s_t hessian_tmp149 = hessian_tmp105 + hessian_tmp33;
  const s_t hessian_tmp150 = adj_lane5*hessian_tmp147 + hessian_tmp149;
  const s_t hessian_tmp151 = hessian_tmp108 + hessian_tmp38;
  const s_t hessian_tmp152 = adj_lane8*hessian_tmp147 + hessian_tmp151;
  const s_t hessian_tmp153 = ((s_t(1) / s_t(6)))*adj_lane5*hessian_tmp11 + ((s_t(1) / s_t(6)))*adj_lane5*hessian_tmp9;
  const s_t hessian_tmp154 = adj_lane2*hessian_tmp153 + hessian_tmp149;
  const s_t hessian_tmp155 = adj_lane5*hessian_tmp153 + hessian_tmp118 + hessian_tmp52;
  const s_t hessian_tmp156 = hessian_tmp120 + hessian_tmp56;
  const s_t hessian_tmp157 = adj_lane8*hessian_tmp153 + hessian_tmp156;
  const s_t hessian_tmp158 = ((s_t(1) / s_t(6)))*adj_lane8*hessian_tmp11 + ((s_t(1) / s_t(6)))*adj_lane8*hessian_tmp9;
  const s_t hessian_tmp159 = adj_lane2*hessian_tmp158 + hessian_tmp151;
  const s_t hessian_tmp160 = adj_lane5*hessian_tmp158 + hessian_tmp156;
  const s_t hessian_tmp161 = adj_lane8*hessian_tmp158 + hessian_tmp131 + hessian_tmp72;
  element_matrix[0] = -hessian_tmp13 - hessian_tmp16 - hessian_tmp19;
  element_matrix[12] = hessian_tmp13;
  element_matrix[24] = hessian_tmp16;
  element_matrix[36] = hessian_tmp19;
  element_matrix[48] = -hessian_tmp21 - hessian_tmp22 - hessian_tmp23;
  element_matrix[60] = hessian_tmp21;
  element_matrix[72] = hessian_tmp22;
  element_matrix[84] = hessian_tmp23;
  element_matrix[96] = -hessian_tmp24 - hessian_tmp25 - hessian_tmp26;
  element_matrix[108] = hessian_tmp24;
  element_matrix[120] = hessian_tmp25;
  element_matrix[132] = hessian_tmp26;
  element_matrix[1] = -hessian_tmp31 - hessian_tmp37 - hessian_tmp41;
  element_matrix[13] = hessian_tmp31;
  element_matrix[25] = hessian_tmp37;
  element_matrix[37] = hessian_tmp41;
  element_matrix[49] = -hessian_tmp43 - hessian_tmp44 - hessian_tmp45;
  element_matrix[61] = hessian_tmp43;
  element_matrix[73] = hessian_tmp44;
  element_matrix[85] = hessian_tmp45;
  element_matrix[97] = -hessian_tmp46 - hessian_tmp47 - hessian_tmp48;
  element_matrix[109] = hessian_tmp46;
  element_matrix[121] = hessian_tmp47;
  element_matrix[133] = hessian_tmp48;
  element_matrix[2] = -hessian_tmp51 - hessian_tmp54 - hessian_tmp60;
  element_matrix[14] = hessian_tmp51;
  element_matrix[26] = hessian_tmp54;
  element_matrix[38] = hessian_tmp60;
  element_matrix[50] = -hessian_tmp62 - hessian_tmp63 - hessian_tmp64;
  element_matrix[62] = hessian_tmp62;
  element_matrix[74] = hessian_tmp63;
  element_matrix[86] = hessian_tmp64;
  element_matrix[98] = -hessian_tmp65 - hessian_tmp66 - hessian_tmp67;
  element_matrix[110] = hessian_tmp65;
  element_matrix[122] = hessian_tmp66;
  element_matrix[134] = hessian_tmp67;
  element_matrix[3] = -hessian_tmp70 - hessian_tmp71 - hessian_tmp74;
  element_matrix[15] = hessian_tmp70;
  element_matrix[27] = hessian_tmp71;
  element_matrix[39] = hessian_tmp74;
  element_matrix[51] = -hessian_tmp77 - hessian_tmp78 - hessian_tmp79;
  element_matrix[63] = hessian_tmp77;
  element_matrix[75] = hessian_tmp78;
  element_matrix[87] = hessian_tmp79;
  element_matrix[99] = -hessian_tmp81 - hessian_tmp82 - hessian_tmp83;
  element_matrix[111] = hessian_tmp81;
  element_matrix[123] = hessian_tmp82;
  element_matrix[135] = hessian_tmp83;
  element_matrix[4] = -hessian_tmp84 - hessian_tmp85 - hessian_tmp86;
  element_matrix[16] = hessian_tmp84;
  element_matrix[28] = hessian_tmp85;
  element_matrix[40] = hessian_tmp86;
  element_matrix[52] = -hessian_tmp91 - hessian_tmp93 - hessian_tmp95;
  element_matrix[64] = hessian_tmp91;
  element_matrix[76] = hessian_tmp93;
  element_matrix[88] = hessian_tmp95;
  element_matrix[100] = -hessian_tmp97 - hessian_tmp98 - hessian_tmp99;
  element_matrix[112] = hessian_tmp97;
  element_matrix[124] = hessian_tmp98;
  element_matrix[136] = hessian_tmp99;
  element_matrix[5] = -hessian_tmp43 - hessian_tmp62 - hessian_tmp77;
  element_matrix[17] = hessian_tmp43;
  element_matrix[29] = hessian_tmp62;
  element_matrix[41] = hessian_tmp77;
  element_matrix[53] = -hessian_tmp103 - hessian_tmp107 - hessian_tmp110;
  element_matrix[65] = hessian_tmp103;
  element_matrix[77] = hessian_tmp107;
  element_matrix[89] = hessian_tmp110;
  element_matrix[101] = -hessian_tmp112 - hessian_tmp113 - hessian_tmp114;
  element_matrix[113] = hessian_tmp112;
  element_matrix[125] = hessian_tmp113;
  element_matrix[137] = hessian_tmp114;
  element_matrix[6] = -hessian_tmp44 - hessian_tmp63 - hessian_tmp78;
  element_matrix[18] = hessian_tmp44;
  element_matrix[30] = hessian_tmp63;
  element_matrix[42] = hessian_tmp78;
  element_matrix[54] = -hessian_tmp117 - hessian_tmp119 - hessian_tmp122;
  element_matrix[66] = hessian_tmp117;
  element_matrix[78] = hessian_tmp119;
  element_matrix[90] = hessian_tmp122;
  element_matrix[102] = -hessian_tmp124 - hessian_tmp125 - hessian_tmp126;
  element_matrix[114] = hessian_tmp124;
  element_matrix[126] = hessian_tmp125;
  element_matrix[138] = hessian_tmp126;
  element_matrix[7] = -hessian_tmp45 - hessian_tmp64 - hessian_tmp79;
  element_matrix[19] = hessian_tmp45;
  element_matrix[31] = hessian_tmp64;
  element_matrix[43] = hessian_tmp79;
  element_matrix[55] = -hessian_tmp129 - hessian_tmp130 - hessian_tmp132;
  element_matrix[67] = hessian_tmp129;
  element_matrix[79] = hessian_tmp130;
  element_matrix[91] = hessian_tmp132;
  element_matrix[103] = -hessian_tmp134 - hessian_tmp135 - hessian_tmp136;
  element_matrix[115] = hessian_tmp134;
  element_matrix[127] = hessian_tmp135;
  element_matrix[139] = hessian_tmp136;
  element_matrix[8] = -hessian_tmp137 - hessian_tmp138 - hessian_tmp139;
  element_matrix[20] = hessian_tmp137;
  element_matrix[32] = hessian_tmp138;
  element_matrix[44] = hessian_tmp139;
  element_matrix[56] = -hessian_tmp140 - hessian_tmp141 - hessian_tmp142;
  element_matrix[68] = hessian_tmp140;
  element_matrix[80] = hessian_tmp141;
  element_matrix[92] = hessian_tmp142;
  element_matrix[104] = -hessian_tmp144 - hessian_tmp145 - hessian_tmp146;
  element_matrix[116] = hessian_tmp144;
  element_matrix[128] = hessian_tmp145;
  element_matrix[140] = hessian_tmp146;
  element_matrix[9] = -hessian_tmp46 - hessian_tmp65 - hessian_tmp81;
  element_matrix[21] = hessian_tmp46;
  element_matrix[33] = hessian_tmp65;
  element_matrix[45] = hessian_tmp81;
  element_matrix[57] = -hessian_tmp112 - hessian_tmp124 - hessian_tmp134;
  element_matrix[69] = hessian_tmp112;
  element_matrix[81] = hessian_tmp124;
  element_matrix[93] = hessian_tmp134;
  element_matrix[105] = -hessian_tmp148 - hessian_tmp150 - hessian_tmp152;
  element_matrix[117] = hessian_tmp148;
  element_matrix[129] = hessian_tmp150;
  element_matrix[141] = hessian_tmp152;
  element_matrix[10] = -hessian_tmp47 - hessian_tmp66 - hessian_tmp82;
  element_matrix[22] = hessian_tmp47;
  element_matrix[34] = hessian_tmp66;
  element_matrix[46] = hessian_tmp82;
  element_matrix[58] = -hessian_tmp113 - hessian_tmp125 - hessian_tmp135;
  element_matrix[70] = hessian_tmp113;
  element_matrix[82] = hessian_tmp125;
  element_matrix[94] = hessian_tmp135;
  element_matrix[106] = -hessian_tmp154 - hessian_tmp155 - hessian_tmp157;
  element_matrix[118] = hessian_tmp154;
  element_matrix[130] = hessian_tmp155;
  element_matrix[142] = hessian_tmp157;
  element_matrix[11] = -hessian_tmp48 - hessian_tmp67 - hessian_tmp83;
  element_matrix[23] = hessian_tmp48;
  element_matrix[35] = hessian_tmp67;
  element_matrix[47] = hessian_tmp83;
  element_matrix[59] = -hessian_tmp114 - hessian_tmp126 - hessian_tmp136;
  element_matrix[71] = hessian_tmp114;
  element_matrix[83] = hessian_tmp126;
  element_matrix[95] = hessian_tmp136;
  element_matrix[107] = -hessian_tmp159 - hessian_tmp160 - hessian_tmp161;
  element_matrix[119] = hessian_tmp159;
  element_matrix[131] = hessian_tmp160;
  element_matrix[143] = hessian_tmp161;
}

} // namespace codegen
} // namespace sfem

#endif
