#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_ELASTIC_D3_SIMPLEX_HESSIAN_HPP
#define MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_ELASTIC_D3_SIMPLEX_HESSIAN_HPP
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
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_elastic_d3_simplex_direct_hessian_reference_element_matrix(
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
    const s_t bu_data[NS * 3][VS],
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
    s_t gu_ref0 = s_t(0);
    s_t gu_ref1 = s_t(0);
    s_t gu_ref2 = s_t(0);
    s_t gu_ref3 = s_t(0);
    s_t gu_ref4 = s_t(0);
    s_t gu_ref5 = s_t(0);
    s_t gu_ref6 = s_t(0);
    s_t gu_ref7 = s_t(0);
    s_t gu_ref8 = s_t(0);
    for (int shape = 0; shape < NS; ++shape) {
      const s_t state_grad_ref0 = grad_ref_x[q * NS + shape];
      const s_t state_grad_ref1 = grad_ref_y[q * NS + shape];
      const s_t state_grad_ref2 = grad_ref_z[q * NS + shape];
      const s_t state_u0 = bu_data[shape * NC][lane];
      gu_ref0 += state_u0 * state_grad_ref0;
      gu_ref1 += state_u0 * state_grad_ref1;
      gu_ref2 += state_u0 * state_grad_ref2;
      const s_t state_u1 = bu_data[shape * NC + 1][lane];
      gu_ref3 += state_u1 * state_grad_ref0;
      gu_ref4 += state_u1 * state_grad_ref1;
      gu_ref5 += state_u1 * state_grad_ref2;
      const s_t state_u2 = bu_data[shape * NC + 2][lane];
      gu_ref6 += state_u2 * state_grad_ref0;
      gu_ref7 += state_u2 * state_grad_ref1;
      gu_ref8 += state_u2 * state_grad_ref2;
    }
    const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane3 + gu_ref2 * adj_lane6) * idet;
    const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane4 + gu_ref2 * adj_lane7) * idet;
    const s_t gu2 = (gu_ref0 * adj_lane2 + gu_ref1 * adj_lane5 + gu_ref2 * adj_lane8) * idet;
    const s_t gu3 = (gu_ref3 * adj_lane0 + gu_ref4 * adj_lane3 + gu_ref5 * adj_lane6) * idet;
    const s_t gu4 = (gu_ref3 * adj_lane1 + gu_ref4 * adj_lane4 + gu_ref5 * adj_lane7) * idet;
    const s_t gu5 = (gu_ref3 * adj_lane2 + gu_ref4 * adj_lane5 + gu_ref5 * adj_lane8) * idet;
    const s_t gu6 = (gu_ref6 * adj_lane0 + gu_ref7 * adj_lane3 + gu_ref8 * adj_lane6) * idet;
    const s_t gu7 = (gu_ref6 * adj_lane1 + gu_ref7 * adj_lane4 + gu_ref8 * adj_lane7) * idet;
    const s_t gu8 = (gu_ref6 * adj_lane2 + gu_ref7 * adj_lane5 + gu_ref8 * adj_lane8) * idet;
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
        const s_t weak_hess_tmp0 = s_t(2)*gu6;
        const s_t weak_hess_tmp1 = gu7*weak_hess_tmp0;
        const s_t weak_hess_tmp2 = gu4 + s_t(1);
        const s_t weak_hess_tmp3 = s_t(2)*gu3;
        const s_t weak_hess_tmp4 = weak_hess_tmp2*weak_hess_tmp3;
        const s_t weak_hess_tmp5 = mu*(-weak_hess_tmp1 - weak_hess_tmp4);
        const s_t weak_hess_tmp6 = gu8 + s_t(1);
        const s_t weak_hess_tmp7 = gu3*weak_hess_tmp6;
        const s_t weak_hess_tmp8 = gu5*gu6 - weak_hess_tmp7;
        const s_t weak_hess_tmp9 = gu5*gu7;
        const s_t weak_hess_tmp10 = s_t(2)*weak_hess_tmp9;
        const s_t weak_hess_tmp11 = -s_t(2)*weak_hess_tmp2*weak_hess_tmp6;
        const s_t weak_hess_tmp12 = ((s_t(1) / s_t(2)))*lmbda;
        const s_t weak_hess_tmp13 = weak_hess_tmp12*(-weak_hess_tmp10 - weak_hess_tmp11);
        const s_t weak_hess_tmp14 = gu5*weak_hess_tmp3;
        const s_t weak_hess_tmp15 = weak_hess_tmp0*weak_hess_tmp6;
        const s_t weak_hess_tmp16 = mu*(-weak_hess_tmp14 - weak_hess_tmp15);
        const s_t weak_hess_tmp17 = gu3*gu7;
        const s_t weak_hess_tmp18 = gu6*weak_hess_tmp2;
        const s_t weak_hess_tmp19 = weak_hess_tmp17 - weak_hess_tmp18;
        const s_t weak_hess_tmp20 = s_t(2)*gu2;
        const s_t weak_hess_tmp21 = gu5*weak_hess_tmp20;
        const s_t weak_hess_tmp22 = s_t(2)*gu1;
        const s_t weak_hess_tmp23 = weak_hess_tmp2*weak_hess_tmp22;
        const s_t weak_hess_tmp24 = mu*(-weak_hess_tmp21 - weak_hess_tmp23);
        const s_t weak_hess_tmp25 = gu1*weak_hess_tmp6;
        const s_t weak_hess_tmp26 = gu2*gu7 - weak_hess_tmp25;
        const s_t weak_hess_tmp27 = gu7*weak_hess_tmp22;
        const s_t weak_hess_tmp28 = weak_hess_tmp20*weak_hess_tmp6;
        const s_t weak_hess_tmp29 = mu*(-weak_hess_tmp27 - weak_hess_tmp28);
        const s_t weak_hess_tmp30 = gu1*gu5;
        const s_t weak_hess_tmp31 = gu2*weak_hess_tmp2;
        const s_t weak_hess_tmp32 = weak_hess_tmp30 - weak_hess_tmp31;
        const s_t weak_hess_tmp33 = s_t(2)*pow_2(gu5);
        const s_t weak_hess_tmp34 = s_t(2)*pow_2(weak_hess_tmp6) + s_t(2);
        const s_t weak_hess_tmp35 = weak_hess_tmp33 + weak_hess_tmp34;
        const s_t weak_hess_tmp36 = s_t(2)*pow_2(gu7);
        const s_t weak_hess_tmp37 = s_t(2)*pow_2(weak_hess_tmp2);
        const s_t weak_hess_tmp38 = weak_hess_tmp36 + weak_hess_tmp37;
        const s_t weak_hess_tmp39 = weak_hess_tmp2*weak_hess_tmp6 - weak_hess_tmp9;
        const s_t weak_hess_tmp40 = gu1*gu6;
        const s_t weak_hess_tmp41 = gu0 + s_t(1);
        const s_t weak_hess_tmp42 = gu7*weak_hess_tmp41;
        const s_t weak_hess_tmp43 = weak_hess_tmp40 - weak_hess_tmp42;
        const s_t weak_hess_tmp44 = s_t(6)*gu7;
        const s_t weak_hess_tmp45 = gu2*gu3;
        const s_t weak_hess_tmp46 = s_t(2)*weak_hess_tmp45;
        const s_t weak_hess_tmp47 = gu5*weak_hess_tmp41;
        const s_t weak_hess_tmp48 = lmbda*(gu1*gu5*gu6 - gu1*weak_hess_tmp7 + gu2*gu3*gu7 - gu2*weak_hess_tmp18 + weak_hess_tmp2*weak_hess_tmp41*weak_hess_tmp6 - weak_hess_tmp41*weak_hess_tmp9 + s_t(-1));
        const s_t weak_hess_tmp49 = gu7*weak_hess_tmp48;
        const s_t weak_hess_tmp50 = mu*(weak_hess_tmp44 - weak_hess_tmp46 + s_t(4)*weak_hess_tmp47) - weak_hess_tmp49;
        const s_t weak_hess_tmp51 = weak_hess_tmp45 - weak_hess_tmp47;
        const s_t weak_hess_tmp52 = s_t(6)*gu5;
        const s_t weak_hess_tmp53 = s_t(2)*weak_hess_tmp40;
        const s_t weak_hess_tmp54 = gu5*weak_hess_tmp48;
        const s_t weak_hess_tmp55 = mu*(s_t(4)*weak_hess_tmp42 + weak_hess_tmp52 - weak_hess_tmp53) - weak_hess_tmp54;
        const s_t weak_hess_tmp56 = gu2*gu6;
        const s_t weak_hess_tmp57 = weak_hess_tmp41*weak_hess_tmp6 - weak_hess_tmp56;
        const s_t weak_hess_tmp58 = gu1*gu3;
        const s_t weak_hess_tmp59 = s_t(2)*weak_hess_tmp58;
        const s_t weak_hess_tmp60 = s_t(6)*gu8 + s_t(6);
        const s_t weak_hess_tmp61 = weak_hess_tmp48*weak_hess_tmp6;
        const s_t weak_hess_tmp62 = mu*(s_t(4)*weak_hess_tmp2*weak_hess_tmp41 - weak_hess_tmp59 - weak_hess_tmp60) + weak_hess_tmp61;
        const s_t weak_hess_tmp63 = weak_hess_tmp2*weak_hess_tmp41 - weak_hess_tmp58;
        const s_t weak_hess_tmp64 = s_t(2)*weak_hess_tmp56;
        const s_t weak_hess_tmp65 = s_t(6)*gu4 + s_t(6);
        const s_t weak_hess_tmp66 = weak_hess_tmp2*weak_hess_tmp48;
        const s_t weak_hess_tmp67 = mu*(s_t(4)*weak_hess_tmp41*weak_hess_tmp6 - weak_hess_tmp64 - weak_hess_tmp65) + weak_hess_tmp66;
        const s_t weak_hess_tmp68 = -s_t(2)*gu5*gu6;
        const s_t weak_hess_tmp69 = s_t(2)*weak_hess_tmp7;
        const s_t weak_hess_tmp70 = weak_hess_tmp12*(-weak_hess_tmp68 - weak_hess_tmp69);
        const s_t weak_hess_tmp71 = s_t(2)*gu5;
        const s_t weak_hess_tmp72 = weak_hess_tmp2*weak_hess_tmp71;
        const s_t weak_hess_tmp73 = s_t(2)*gu7;
        const s_t weak_hess_tmp74 = weak_hess_tmp6*weak_hess_tmp73;
        const s_t weak_hess_tmp75 = mu*(-weak_hess_tmp72 - weak_hess_tmp74);
        const s_t weak_hess_tmp76 = weak_hess_tmp3*weak_hess_tmp41;
        const s_t weak_hess_tmp77 = mu*(-weak_hess_tmp21 - weak_hess_tmp76);
        const s_t weak_hess_tmp78 = weak_hess_tmp0*weak_hess_tmp41;
        const s_t weak_hess_tmp79 = mu*(-weak_hess_tmp28 - weak_hess_tmp78);
        const s_t weak_hess_tmp80 = s_t(2)*pow_2(gu3);
        const s_t weak_hess_tmp81 = s_t(2)*pow_2(gu6);
        const s_t weak_hess_tmp82 = weak_hess_tmp80 + weak_hess_tmp81;
        const s_t weak_hess_tmp83 = s_t(6)*gu6;
        const s_t weak_hess_tmp84 = s_t(2)*weak_hess_tmp31;
        const s_t weak_hess_tmp85 = gu6*weak_hess_tmp48;
        const s_t weak_hess_tmp86 = mu*(s_t(4)*gu1*gu5 - weak_hess_tmp83 - weak_hess_tmp84) + weak_hess_tmp85;
        const s_t weak_hess_tmp87 = s_t(2)*weak_hess_tmp42;
        const s_t weak_hess_tmp88 = mu*(s_t(4)*gu1*gu6 - weak_hess_tmp52 - weak_hess_tmp87) + weak_hess_tmp54;
        const s_t weak_hess_tmp89 = s_t(6)*gu3;
        const s_t weak_hess_tmp90 = -s_t(2)*gu2*gu7;
        const s_t weak_hess_tmp91 = gu3*weak_hess_tmp48;
        const s_t weak_hess_tmp92 = mu*(s_t(4)*weak_hess_tmp25 + weak_hess_tmp89 + weak_hess_tmp90) - weak_hess_tmp91;
        const s_t weak_hess_tmp93 = -s_t(2)*weak_hess_tmp2*weak_hess_tmp41;
        const s_t weak_hess_tmp94 = mu*(s_t(4)*weak_hess_tmp58 + weak_hess_tmp60 + weak_hess_tmp93) - weak_hess_tmp61;
        const s_t weak_hess_tmp95 = s_t(2)*weak_hess_tmp17;
        const s_t weak_hess_tmp96 = s_t(2)*weak_hess_tmp18;
        const s_t weak_hess_tmp97 = weak_hess_tmp12*(weak_hess_tmp95 - weak_hess_tmp96);
        const s_t weak_hess_tmp98 = mu*(-weak_hess_tmp23 - weak_hess_tmp76);
        const s_t weak_hess_tmp99 = mu*(-weak_hess_tmp27 - weak_hess_tmp78);
        const s_t weak_hess_tmp100 = s_t(2)*weak_hess_tmp47;
        const s_t weak_hess_tmp101 = mu*(s_t(4)*gu2*gu3 - weak_hess_tmp100 - weak_hess_tmp44) + weak_hess_tmp49;
        const s_t weak_hess_tmp102 = s_t(2)*weak_hess_tmp25;
        const s_t weak_hess_tmp103 = mu*(s_t(4)*gu2*gu7 - weak_hess_tmp102 - weak_hess_tmp89) + weak_hess_tmp91;
        const s_t weak_hess_tmp104 = s_t(2)*weak_hess_tmp30;
        const s_t weak_hess_tmp105 = mu*(-weak_hess_tmp104 + s_t(4)*weak_hess_tmp31 + weak_hess_tmp83) - weak_hess_tmp85;
        const s_t weak_hess_tmp106 = -s_t(2)*weak_hess_tmp41*weak_hess_tmp6;
        const s_t weak_hess_tmp107 = mu*(weak_hess_tmp106 + s_t(4)*weak_hess_tmp56 + weak_hess_tmp65) - weak_hess_tmp66;
        const s_t weak_hess_tmp108 = weak_hess_tmp12*(-weak_hess_tmp102 - weak_hess_tmp90);
        const s_t weak_hess_tmp109 = weak_hess_tmp22*weak_hess_tmp41;
        const s_t weak_hess_tmp110 = mu*(-weak_hess_tmp1 - weak_hess_tmp109);
        const s_t weak_hess_tmp111 = weak_hess_tmp20*weak_hess_tmp41;
        const s_t weak_hess_tmp112 = mu*(-weak_hess_tmp111 - weak_hess_tmp15);
        const s_t weak_hess_tmp113 = weak_hess_tmp6*weak_hess_tmp71;
        const s_t weak_hess_tmp114 = weak_hess_tmp2*weak_hess_tmp73;
        const s_t weak_hess_tmp115 = mu*(-weak_hess_tmp113 - weak_hess_tmp114);
        const s_t weak_hess_tmp116 = s_t(2)*pow_2(gu2);
        const s_t weak_hess_tmp117 = weak_hess_tmp116 + weak_hess_tmp34;
        const s_t weak_hess_tmp118 = s_t(2)*pow_2(gu1);
        const s_t weak_hess_tmp119 = weak_hess_tmp118 + weak_hess_tmp36;
        const s_t weak_hess_tmp120 = s_t(6)*gu2;
        const s_t weak_hess_tmp121 = gu2*weak_hess_tmp48;
        const s_t weak_hess_tmp122 = mu*(s_t(4)*gu3*gu7 - weak_hess_tmp120 - weak_hess_tmp96) + weak_hess_tmp121;
        const s_t weak_hess_tmp123 = s_t(6)*gu1;
        const s_t weak_hess_tmp124 = gu1*weak_hess_tmp48;
        const s_t weak_hess_tmp125 = mu*(weak_hess_tmp123 + weak_hess_tmp68 + s_t(4)*weak_hess_tmp7) - weak_hess_tmp124;
        const s_t weak_hess_tmp126 = weak_hess_tmp12*(-weak_hess_tmp106 - weak_hess_tmp64);
        const s_t weak_hess_tmp127 = gu2*weak_hess_tmp22;
        const s_t weak_hess_tmp128 = mu*(-weak_hess_tmp127 - weak_hess_tmp74);
        const s_t weak_hess_tmp129 = gu6*weak_hess_tmp3;
        const s_t weak_hess_tmp130 = mu*(-weak_hess_tmp113 - weak_hess_tmp129);
        const s_t weak_hess_tmp131 = s_t(2)*pow_2(weak_hess_tmp41);
        const s_t weak_hess_tmp132 = weak_hess_tmp131 + weak_hess_tmp81;
        const s_t weak_hess_tmp133 = mu*(weak_hess_tmp120 + s_t(4)*weak_hess_tmp18 - weak_hess_tmp95) - weak_hess_tmp121;
        const s_t weak_hess_tmp134 = s_t(6)*gu0 + s_t(6);
        const s_t weak_hess_tmp135 = weak_hess_tmp41*weak_hess_tmp48;
        const s_t weak_hess_tmp136 = mu*(-weak_hess_tmp10 - weak_hess_tmp134 + s_t(4)*weak_hess_tmp2*weak_hess_tmp6) + weak_hess_tmp135;
        const s_t weak_hess_tmp137 = weak_hess_tmp12*(weak_hess_tmp53 - weak_hess_tmp87);
        const s_t weak_hess_tmp138 = mu*(-weak_hess_tmp114 - weak_hess_tmp129);
        const s_t weak_hess_tmp139 = mu*(s_t(4)*gu5*gu6 - weak_hess_tmp123 - weak_hess_tmp69) + weak_hess_tmp124;
        const s_t weak_hess_tmp140 = mu*(weak_hess_tmp11 + weak_hess_tmp134 + s_t(4)*weak_hess_tmp9) - weak_hess_tmp135;
        const s_t weak_hess_tmp141 = weak_hess_tmp12*(weak_hess_tmp104 - weak_hess_tmp84);
        const s_t weak_hess_tmp142 = mu*(-weak_hess_tmp109 - weak_hess_tmp4);
        const s_t weak_hess_tmp143 = mu*(-weak_hess_tmp111 - weak_hess_tmp14);
        const s_t weak_hess_tmp144 = weak_hess_tmp116 + weak_hess_tmp33 + s_t(2);
        const s_t weak_hess_tmp145 = weak_hess_tmp118 + weak_hess_tmp37;
        const s_t weak_hess_tmp146 = weak_hess_tmp12*(-weak_hess_tmp100 + weak_hess_tmp46);
        const s_t weak_hess_tmp147 = mu*(-weak_hess_tmp127 - weak_hess_tmp72);
        const s_t weak_hess_tmp148 = weak_hess_tmp131 + weak_hess_tmp80;
        const s_t weak_hess_tmp149 = weak_hess_tmp12*(-weak_hess_tmp59 - weak_hess_tmp93);
        material[0] = trial_grad[0]*(mu*(weak_hess_tmp35 + weak_hess_tmp38) + weak_hess_tmp13*weak_hess_tmp39) + trial_grad[1]*(weak_hess_tmp13*weak_hess_tmp8 + weak_hess_tmp5) + trial_grad[2]*(weak_hess_tmp13*weak_hess_tmp19 + weak_hess_tmp16) + trial_grad[3]*(weak_hess_tmp13*weak_hess_tmp26 + weak_hess_tmp24) + trial_grad[4]*(weak_hess_tmp13*weak_hess_tmp57 + weak_hess_tmp62) + trial_grad[5]*(weak_hess_tmp13*weak_hess_tmp43 + weak_hess_tmp50) + trial_grad[6]*(weak_hess_tmp13*weak_hess_tmp32 + weak_hess_tmp29) + trial_grad[7]*(weak_hess_tmp13*weak_hess_tmp51 + weak_hess_tmp55) + trial_grad[8]*(weak_hess_tmp13*weak_hess_tmp63 + weak_hess_tmp67);
        material[1] = trial_grad[0]*(weak_hess_tmp39*weak_hess_tmp70 + weak_hess_tmp5) + trial_grad[1]*(mu*(weak_hess_tmp35 + weak_hess_tmp82) + weak_hess_tmp70*weak_hess_tmp8) + trial_grad[2]*(weak_hess_tmp19*weak_hess_tmp70 + weak_hess_tmp75) + trial_grad[3]*(weak_hess_tmp26*weak_hess_tmp70 + weak_hess_tmp94) + trial_grad[4]*(weak_hess_tmp57*weak_hess_tmp70 + weak_hess_tmp77) + trial_grad[5]*(weak_hess_tmp43*weak_hess_tmp70 + weak_hess_tmp86) + trial_grad[6]*(weak_hess_tmp32*weak_hess_tmp70 + weak_hess_tmp88) + trial_grad[7]*(weak_hess_tmp51*weak_hess_tmp70 + weak_hess_tmp79) + trial_grad[8]*(weak_hess_tmp63*weak_hess_tmp70 + weak_hess_tmp92);
        material[2] = trial_grad[0]*(weak_hess_tmp16 + weak_hess_tmp39*weak_hess_tmp97) + trial_grad[1]*(weak_hess_tmp75 + weak_hess_tmp8*weak_hess_tmp97) + trial_grad[2]*(mu*(weak_hess_tmp38 + weak_hess_tmp82 + s_t(2)) + weak_hess_tmp19*weak_hess_tmp97) + trial_grad[3]*(weak_hess_tmp101 + weak_hess_tmp26*weak_hess_tmp97) + trial_grad[4]*(weak_hess_tmp105 + weak_hess_tmp57*weak_hess_tmp97) + trial_grad[5]*(weak_hess_tmp43*weak_hess_tmp97 + weak_hess_tmp98) + trial_grad[6]*(weak_hess_tmp107 + weak_hess_tmp32*weak_hess_tmp97) + trial_grad[7]*(weak_hess_tmp103 + weak_hess_tmp51*weak_hess_tmp97) + trial_grad[8]*(weak_hess_tmp63*weak_hess_tmp97 + weak_hess_tmp99);
        material[3] = trial_grad[0]*(weak_hess_tmp108*weak_hess_tmp39 + weak_hess_tmp24) + trial_grad[1]*(weak_hess_tmp108*weak_hess_tmp8 + weak_hess_tmp94) + trial_grad[2]*(weak_hess_tmp101 + weak_hess_tmp108*weak_hess_tmp19) + trial_grad[3]*(mu*(weak_hess_tmp117 + weak_hess_tmp119) + weak_hess_tmp108*weak_hess_tmp26) + trial_grad[4]*(weak_hess_tmp108*weak_hess_tmp57 + weak_hess_tmp110) + trial_grad[5]*(weak_hess_tmp108*weak_hess_tmp43 + weak_hess_tmp112) + trial_grad[6]*(weak_hess_tmp108*weak_hess_tmp32 + weak_hess_tmp115) + trial_grad[7]*(weak_hess_tmp108*weak_hess_tmp51 + weak_hess_tmp122) + trial_grad[8]*(weak_hess_tmp108*weak_hess_tmp63 + weak_hess_tmp125);
        material[4] = trial_grad[0]*(weak_hess_tmp126*weak_hess_tmp39 + weak_hess_tmp62) + trial_grad[1]*(weak_hess_tmp126*weak_hess_tmp8 + weak_hess_tmp77) + trial_grad[2]*(weak_hess_tmp105 + weak_hess_tmp126*weak_hess_tmp19) + trial_grad[3]*(weak_hess_tmp110 + weak_hess_tmp126*weak_hess_tmp26) + trial_grad[4]*(mu*(weak_hess_tmp117 + weak_hess_tmp132) + weak_hess_tmp126*weak_hess_tmp57) + trial_grad[5]*(weak_hess_tmp126*weak_hess_tmp43 + weak_hess_tmp128) + trial_grad[6]*(weak_hess_tmp126*weak_hess_tmp32 + weak_hess_tmp133) + trial_grad[7]*(weak_hess_tmp126*weak_hess_tmp51 + weak_hess_tmp130) + trial_grad[8]*(weak_hess_tmp126*weak_hess_tmp63 + weak_hess_tmp136);
        material[5] = trial_grad[0]*(weak_hess_tmp137*weak_hess_tmp39 + weak_hess_tmp50) + trial_grad[1]*(weak_hess_tmp137*weak_hess_tmp8 + weak_hess_tmp86) + trial_grad[2]*(weak_hess_tmp137*weak_hess_tmp19 + weak_hess_tmp98) + trial_grad[3]*(weak_hess_tmp112 + weak_hess_tmp137*weak_hess_tmp26) + trial_grad[4]*(weak_hess_tmp128 + weak_hess_tmp137*weak_hess_tmp57) + trial_grad[5]*(mu*(weak_hess_tmp119 + weak_hess_tmp132 + s_t(2)) + weak_hess_tmp137*weak_hess_tmp43) + trial_grad[6]*(weak_hess_tmp137*weak_hess_tmp32 + weak_hess_tmp139) + trial_grad[7]*(weak_hess_tmp137*weak_hess_tmp51 + weak_hess_tmp140) + trial_grad[8]*(weak_hess_tmp137*weak_hess_tmp63 + weak_hess_tmp138);
        material[6] = trial_grad[0]*(weak_hess_tmp141*weak_hess_tmp39 + weak_hess_tmp29) + trial_grad[1]*(weak_hess_tmp141*weak_hess_tmp8 + weak_hess_tmp88) + trial_grad[2]*(weak_hess_tmp107 + weak_hess_tmp141*weak_hess_tmp19) + trial_grad[3]*(weak_hess_tmp115 + weak_hess_tmp141*weak_hess_tmp26) + trial_grad[4]*(weak_hess_tmp133 + weak_hess_tmp141*weak_hess_tmp57) + trial_grad[5]*(weak_hess_tmp139 + weak_hess_tmp141*weak_hess_tmp43) + trial_grad[6]*(mu*(weak_hess_tmp144 + weak_hess_tmp145) + weak_hess_tmp141*weak_hess_tmp32) + trial_grad[7]*(weak_hess_tmp141*weak_hess_tmp51 + weak_hess_tmp142) + trial_grad[8]*(weak_hess_tmp141*weak_hess_tmp63 + weak_hess_tmp143);
        material[7] = trial_grad[0]*(weak_hess_tmp146*weak_hess_tmp39 + weak_hess_tmp55) + trial_grad[1]*(weak_hess_tmp146*weak_hess_tmp8 + weak_hess_tmp79) + trial_grad[2]*(weak_hess_tmp103 + weak_hess_tmp146*weak_hess_tmp19) + trial_grad[3]*(weak_hess_tmp122 + weak_hess_tmp146*weak_hess_tmp26) + trial_grad[4]*(weak_hess_tmp130 + weak_hess_tmp146*weak_hess_tmp57) + trial_grad[5]*(weak_hess_tmp140 + weak_hess_tmp146*weak_hess_tmp43) + trial_grad[6]*(weak_hess_tmp142 + weak_hess_tmp146*weak_hess_tmp32) + trial_grad[7]*(mu*(weak_hess_tmp144 + weak_hess_tmp148) + weak_hess_tmp146*weak_hess_tmp51) + trial_grad[8]*(weak_hess_tmp146*weak_hess_tmp63 + weak_hess_tmp147);
        material[8] = trial_grad[0]*(weak_hess_tmp149*weak_hess_tmp39 + weak_hess_tmp67) + trial_grad[1]*(weak_hess_tmp149*weak_hess_tmp8 + weak_hess_tmp92) + trial_grad[2]*(weak_hess_tmp149*weak_hess_tmp19 + weak_hess_tmp99) + trial_grad[3]*(weak_hess_tmp125 + weak_hess_tmp149*weak_hess_tmp26) + trial_grad[4]*(weak_hess_tmp136 + weak_hess_tmp149*weak_hess_tmp57) + trial_grad[5]*(weak_hess_tmp138 + weak_hess_tmp149*weak_hess_tmp43) + trial_grad[6]*(weak_hess_tmp143 + weak_hess_tmp149*weak_hess_tmp32) + trial_grad[7]*(weak_hess_tmp147 + weak_hess_tmp149*weak_hess_tmp51) + trial_grad[8]*(mu*(weak_hess_tmp145 + weak_hess_tmp148 + s_t(2)) + weak_hess_tmp149*weak_hess_tmp63);
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
