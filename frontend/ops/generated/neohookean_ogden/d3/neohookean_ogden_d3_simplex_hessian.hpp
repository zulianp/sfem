#ifndef NEOHOOKEAN_OGDEN_D3_SIMPLEX_HESSIAN_HPP
#define NEOHOOKEAN_OGDEN_D3_SIMPLEX_HESSIAN_HPP
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
static SFEM_INLINE void neohookean_ogden_d3_simplex_direct_hessian_reference_element_matrix(
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
        const s_t weak_hess_tmp0 = gu5*gu7;
        const s_t weak_hess_tmp1 = gu4 + s_t(1);
        const s_t weak_hess_tmp2 = gu8 + s_t(1);
        const s_t weak_hess_tmp3 = weak_hess_tmp0 - weak_hess_tmp1*weak_hess_tmp2;
        const s_t weak_hess_tmp4 = -weak_hess_tmp3;
        const s_t weak_hess_tmp5 = gu3*weak_hess_tmp2;
        const s_t weak_hess_tmp6 = gu6*weak_hess_tmp1;
        const s_t weak_hess_tmp7 = gu0 + s_t(1);
        const s_t weak_hess_tmp8 = gu1*gu5*gu6 - gu1*weak_hess_tmp5 + gu2*gu3*gu7 - gu2*weak_hess_tmp6 - weak_hess_tmp0*weak_hess_tmp7 + weak_hess_tmp1*weak_hess_tmp2*weak_hess_tmp7;
        const s_t weak_hess_tmp9 = pow_m2(weak_hess_tmp8);
        const s_t weak_hess_tmp10 = lmbda*weak_hess_tmp9;
        const s_t weak_hess_tmp11 = mu*weak_hess_tmp9;
        const s_t weak_hess_tmp12 = weak_hess_tmp3*weak_hess_tmp4;
        const s_t weak_hess_tmp13 = log(weak_hess_tmp8);
        const s_t weak_hess_tmp14 = weak_hess_tmp10*weak_hess_tmp13;
        const s_t weak_hess_tmp15 = -gu5*gu6 + weak_hess_tmp5;
        const s_t weak_hess_tmp16 = -weak_hess_tmp15;
        const s_t weak_hess_tmp17 = weak_hess_tmp10*weak_hess_tmp4;
        const s_t weak_hess_tmp18 = weak_hess_tmp16*weak_hess_tmp17;
        const s_t weak_hess_tmp19 = weak_hess_tmp11*weak_hess_tmp4;
        const s_t weak_hess_tmp20 = weak_hess_tmp13*weak_hess_tmp17;
        const s_t weak_hess_tmp21 = gu3*gu7 - weak_hess_tmp6;
        const s_t weak_hess_tmp22 = weak_hess_tmp17*weak_hess_tmp21;
        const s_t weak_hess_tmp23 = -weak_hess_tmp21;
        const s_t weak_hess_tmp24 = gu1*weak_hess_tmp2 - gu2*gu7;
        const s_t weak_hess_tmp25 = -weak_hess_tmp24;
        const s_t weak_hess_tmp26 = weak_hess_tmp17*weak_hess_tmp25;
        const s_t weak_hess_tmp27 = gu1*gu5 - gu2*weak_hess_tmp1;
        const s_t weak_hess_tmp28 = weak_hess_tmp17*weak_hess_tmp27;
        const s_t weak_hess_tmp29 = -weak_hess_tmp27;
        const s_t weak_hess_tmp30 = gu1*gu6 - gu7*weak_hess_tmp7;
        const s_t weak_hess_tmp31 = -weak_hess_tmp30;
        const s_t weak_hess_tmp32 = pow_m1(weak_hess_tmp8);
        const s_t weak_hess_tmp33 = mu*weak_hess_tmp32;
        const s_t weak_hess_tmp34 = gu7*weak_hess_tmp33;
        const s_t weak_hess_tmp35 = lmbda*weak_hess_tmp13*weak_hess_tmp32;
        const s_t weak_hess_tmp36 = gu7*weak_hess_tmp35;
        const s_t weak_hess_tmp37 = weak_hess_tmp17*weak_hess_tmp30 + weak_hess_tmp34 - weak_hess_tmp36;
        const s_t weak_hess_tmp38 = gu2*gu3 - gu5*weak_hess_tmp7;
        const s_t weak_hess_tmp39 = -weak_hess_tmp38;
        const s_t weak_hess_tmp40 = gu5*weak_hess_tmp33;
        const s_t weak_hess_tmp41 = gu5*weak_hess_tmp35;
        const s_t weak_hess_tmp42 = weak_hess_tmp17*weak_hess_tmp38 + weak_hess_tmp40 - weak_hess_tmp41;
        const s_t weak_hess_tmp43 = gu2*gu6 - weak_hess_tmp2*weak_hess_tmp7;
        const s_t weak_hess_tmp44 = weak_hess_tmp2*weak_hess_tmp33;
        const s_t weak_hess_tmp45 = weak_hess_tmp2*weak_hess_tmp35;
        const s_t weak_hess_tmp46 = -weak_hess_tmp43;
        const s_t weak_hess_tmp47 = weak_hess_tmp17*weak_hess_tmp46 - weak_hess_tmp44 + weak_hess_tmp45;
        const s_t weak_hess_tmp48 = gu1*gu3 - weak_hess_tmp1*weak_hess_tmp7;
        const s_t weak_hess_tmp49 = weak_hess_tmp1*weak_hess_tmp33;
        const s_t weak_hess_tmp50 = weak_hess_tmp1*weak_hess_tmp35;
        const s_t weak_hess_tmp51 = -weak_hess_tmp48;
        const s_t weak_hess_tmp52 = weak_hess_tmp17*weak_hess_tmp51 - weak_hess_tmp49 + weak_hess_tmp50;
        const s_t weak_hess_tmp53 = weak_hess_tmp11*weak_hess_tmp16;
        const s_t weak_hess_tmp54 = weak_hess_tmp10*weak_hess_tmp16;
        const s_t weak_hess_tmp55 = weak_hess_tmp13*weak_hess_tmp54;
        const s_t weak_hess_tmp56 = weak_hess_tmp21*weak_hess_tmp54;
        const s_t weak_hess_tmp57 = weak_hess_tmp38*weak_hess_tmp54;
        const s_t weak_hess_tmp58 = weak_hess_tmp46*weak_hess_tmp54;
        const s_t weak_hess_tmp59 = gu6*weak_hess_tmp33;
        const s_t weak_hess_tmp60 = gu6*weak_hess_tmp35;
        const s_t weak_hess_tmp61 = weak_hess_tmp30*weak_hess_tmp54 - weak_hess_tmp59 + weak_hess_tmp60;
        const s_t weak_hess_tmp62 = weak_hess_tmp27*weak_hess_tmp54 - weak_hess_tmp40 + weak_hess_tmp41;
        const s_t weak_hess_tmp63 = weak_hess_tmp25*weak_hess_tmp54 + weak_hess_tmp44 - weak_hess_tmp45;
        const s_t weak_hess_tmp64 = gu3*weak_hess_tmp33;
        const s_t weak_hess_tmp65 = gu3*weak_hess_tmp35;
        const s_t weak_hess_tmp66 = weak_hess_tmp51*weak_hess_tmp54 + weak_hess_tmp64 - weak_hess_tmp65;
        const s_t weak_hess_tmp67 = weak_hess_tmp11*weak_hess_tmp21;
        const s_t weak_hess_tmp68 = weak_hess_tmp10*weak_hess_tmp21;
        const s_t weak_hess_tmp69 = weak_hess_tmp13*weak_hess_tmp68;
        const s_t weak_hess_tmp70 = weak_hess_tmp30*weak_hess_tmp68;
        const s_t weak_hess_tmp71 = weak_hess_tmp51*weak_hess_tmp68;
        const s_t weak_hess_tmp72 = weak_hess_tmp25*weak_hess_tmp68 - weak_hess_tmp34 + weak_hess_tmp36;
        const s_t weak_hess_tmp73 = weak_hess_tmp38*weak_hess_tmp68 - weak_hess_tmp64 + weak_hess_tmp65;
        const s_t weak_hess_tmp74 = weak_hess_tmp27*weak_hess_tmp68 + weak_hess_tmp49 - weak_hess_tmp50;
        const s_t weak_hess_tmp75 = weak_hess_tmp46*weak_hess_tmp68 + weak_hess_tmp59 - weak_hess_tmp60;
        const s_t weak_hess_tmp76 = weak_hess_tmp11*weak_hess_tmp25;
        const s_t weak_hess_tmp77 = weak_hess_tmp10*weak_hess_tmp25;
        const s_t weak_hess_tmp78 = weak_hess_tmp13*weak_hess_tmp77;
        const s_t weak_hess_tmp79 = weak_hess_tmp30*weak_hess_tmp77;
        const s_t weak_hess_tmp80 = weak_hess_tmp27*weak_hess_tmp77;
        const s_t weak_hess_tmp81 = weak_hess_tmp46*weak_hess_tmp77;
        const s_t weak_hess_tmp82 = gu2*weak_hess_tmp33;
        const s_t weak_hess_tmp83 = gu2*weak_hess_tmp35;
        const s_t weak_hess_tmp84 = weak_hess_tmp38*weak_hess_tmp77 - weak_hess_tmp82 + weak_hess_tmp83;
        const s_t weak_hess_tmp85 = gu1*weak_hess_tmp33;
        const s_t weak_hess_tmp86 = gu1*weak_hess_tmp35;
        const s_t weak_hess_tmp87 = weak_hess_tmp51*weak_hess_tmp77 + weak_hess_tmp85 - weak_hess_tmp86;
        const s_t weak_hess_tmp88 = weak_hess_tmp11*weak_hess_tmp46;
        const s_t weak_hess_tmp89 = weak_hess_tmp10*weak_hess_tmp46;
        const s_t weak_hess_tmp90 = weak_hess_tmp13*weak_hess_tmp89;
        const s_t weak_hess_tmp91 = weak_hess_tmp30*weak_hess_tmp89;
        const s_t weak_hess_tmp92 = weak_hess_tmp38*weak_hess_tmp89;
        const s_t weak_hess_tmp93 = weak_hess_tmp27*weak_hess_tmp89 + weak_hess_tmp82 - weak_hess_tmp83;
        const s_t weak_hess_tmp94 = weak_hess_tmp33*weak_hess_tmp7;
        const s_t weak_hess_tmp95 = weak_hess_tmp35*weak_hess_tmp7;
        const s_t weak_hess_tmp96 = weak_hess_tmp51*weak_hess_tmp89 - weak_hess_tmp94 + weak_hess_tmp95;
        const s_t weak_hess_tmp97 = weak_hess_tmp11*weak_hess_tmp30;
        const s_t weak_hess_tmp98 = weak_hess_tmp10*weak_hess_tmp30;
        const s_t weak_hess_tmp99 = weak_hess_tmp13*weak_hess_tmp98;
        const s_t weak_hess_tmp100 = weak_hess_tmp51*weak_hess_tmp98;
        const s_t weak_hess_tmp101 = weak_hess_tmp27*weak_hess_tmp98 - weak_hess_tmp85 + weak_hess_tmp86;
        const s_t weak_hess_tmp102 = weak_hess_tmp38*weak_hess_tmp98 + weak_hess_tmp94 - weak_hess_tmp95;
        const s_t weak_hess_tmp103 = weak_hess_tmp11*weak_hess_tmp27;
        const s_t weak_hess_tmp104 = weak_hess_tmp10*weak_hess_tmp27;
        const s_t weak_hess_tmp105 = weak_hess_tmp104*weak_hess_tmp13;
        const s_t weak_hess_tmp106 = weak_hess_tmp104*weak_hess_tmp38;
        const s_t weak_hess_tmp107 = weak_hess_tmp104*weak_hess_tmp51;
        const s_t weak_hess_tmp108 = weak_hess_tmp11*weak_hess_tmp38;
        const s_t weak_hess_tmp109 = weak_hess_tmp10*weak_hess_tmp38;
        const s_t weak_hess_tmp110 = weak_hess_tmp109*weak_hess_tmp13;
        const s_t weak_hess_tmp111 = weak_hess_tmp109*weak_hess_tmp51;
        const s_t weak_hess_tmp112 = weak_hess_tmp11*weak_hess_tmp51;
        const s_t weak_hess_tmp113 = weak_hess_tmp14*weak_hess_tmp51;
        material[0] = trial_grad[0]*(mu + weak_hess_tmp10*pow_2(weak_hess_tmp4) - weak_hess_tmp11*weak_hess_tmp12 + weak_hess_tmp12*weak_hess_tmp14) + trial_grad[1]*(-weak_hess_tmp15*weak_hess_tmp19 + weak_hess_tmp15*weak_hess_tmp20 + weak_hess_tmp18) + trial_grad[2]*(-weak_hess_tmp19*weak_hess_tmp23 + weak_hess_tmp20*weak_hess_tmp23 + weak_hess_tmp22) + trial_grad[3]*(-weak_hess_tmp19*weak_hess_tmp24 + weak_hess_tmp20*weak_hess_tmp24 + weak_hess_tmp26) + trial_grad[4]*(-weak_hess_tmp19*weak_hess_tmp43 + weak_hess_tmp20*weak_hess_tmp43 + weak_hess_tmp47) + trial_grad[5]*(-weak_hess_tmp19*weak_hess_tmp31 + weak_hess_tmp20*weak_hess_tmp31 + weak_hess_tmp37) + trial_grad[6]*(-weak_hess_tmp19*weak_hess_tmp29 + weak_hess_tmp20*weak_hess_tmp29 + weak_hess_tmp28) + trial_grad[7]*(-weak_hess_tmp19*weak_hess_tmp39 + weak_hess_tmp20*weak_hess_tmp39 + weak_hess_tmp42) + trial_grad[8]*(-weak_hess_tmp19*weak_hess_tmp48 + weak_hess_tmp20*weak_hess_tmp48 + weak_hess_tmp52);
        material[1] = trial_grad[0]*(weak_hess_tmp18 - weak_hess_tmp3*weak_hess_tmp53 + weak_hess_tmp3*weak_hess_tmp55) + trial_grad[1]*(mu + weak_hess_tmp10*pow_2(weak_hess_tmp16) - weak_hess_tmp15*weak_hess_tmp53 + weak_hess_tmp15*weak_hess_tmp55) + trial_grad[2]*(-weak_hess_tmp23*weak_hess_tmp53 + weak_hess_tmp23*weak_hess_tmp55 + weak_hess_tmp56) + trial_grad[3]*(-weak_hess_tmp24*weak_hess_tmp53 + weak_hess_tmp24*weak_hess_tmp55 + weak_hess_tmp63) + trial_grad[4]*(-weak_hess_tmp43*weak_hess_tmp53 + weak_hess_tmp43*weak_hess_tmp55 + weak_hess_tmp58) + trial_grad[5]*(-weak_hess_tmp31*weak_hess_tmp53 + weak_hess_tmp31*weak_hess_tmp55 + weak_hess_tmp61) + trial_grad[6]*(-weak_hess_tmp29*weak_hess_tmp53 + weak_hess_tmp29*weak_hess_tmp55 + weak_hess_tmp62) + trial_grad[7]*(-weak_hess_tmp39*weak_hess_tmp53 + weak_hess_tmp39*weak_hess_tmp55 + weak_hess_tmp57) + trial_grad[8]*(-weak_hess_tmp48*weak_hess_tmp53 + weak_hess_tmp48*weak_hess_tmp55 + weak_hess_tmp66);
        material[2] = trial_grad[0]*(weak_hess_tmp22 - weak_hess_tmp3*weak_hess_tmp67 + weak_hess_tmp3*weak_hess_tmp69) + trial_grad[1]*(-weak_hess_tmp15*weak_hess_tmp67 + weak_hess_tmp15*weak_hess_tmp69 + weak_hess_tmp56) + trial_grad[2]*(mu + weak_hess_tmp10*pow_2(weak_hess_tmp21) - weak_hess_tmp23*weak_hess_tmp67 + weak_hess_tmp23*weak_hess_tmp69) + trial_grad[3]*(-weak_hess_tmp24*weak_hess_tmp67 + weak_hess_tmp24*weak_hess_tmp69 + weak_hess_tmp72) + trial_grad[4]*(-weak_hess_tmp43*weak_hess_tmp67 + weak_hess_tmp43*weak_hess_tmp69 + weak_hess_tmp75) + trial_grad[5]*(-weak_hess_tmp31*weak_hess_tmp67 + weak_hess_tmp31*weak_hess_tmp69 + weak_hess_tmp70) + trial_grad[6]*(-weak_hess_tmp29*weak_hess_tmp67 + weak_hess_tmp29*weak_hess_tmp69 + weak_hess_tmp74) + trial_grad[7]*(-weak_hess_tmp39*weak_hess_tmp67 + weak_hess_tmp39*weak_hess_tmp69 + weak_hess_tmp73) + trial_grad[8]*(-weak_hess_tmp48*weak_hess_tmp67 + weak_hess_tmp48*weak_hess_tmp69 + weak_hess_tmp71);
        material[3] = trial_grad[0]*(weak_hess_tmp26 - weak_hess_tmp3*weak_hess_tmp76 + weak_hess_tmp3*weak_hess_tmp78) + trial_grad[1]*(-weak_hess_tmp15*weak_hess_tmp76 + weak_hess_tmp15*weak_hess_tmp78 + weak_hess_tmp63) + trial_grad[2]*(-weak_hess_tmp23*weak_hess_tmp76 + weak_hess_tmp23*weak_hess_tmp78 + weak_hess_tmp72) + trial_grad[3]*(mu + weak_hess_tmp10*pow_2(weak_hess_tmp25) - weak_hess_tmp24*weak_hess_tmp76 + weak_hess_tmp24*weak_hess_tmp78) + trial_grad[4]*(-weak_hess_tmp43*weak_hess_tmp76 + weak_hess_tmp43*weak_hess_tmp78 + weak_hess_tmp81) + trial_grad[5]*(-weak_hess_tmp31*weak_hess_tmp76 + weak_hess_tmp31*weak_hess_tmp78 + weak_hess_tmp79) + trial_grad[6]*(-weak_hess_tmp29*weak_hess_tmp76 + weak_hess_tmp29*weak_hess_tmp78 + weak_hess_tmp80) + trial_grad[7]*(-weak_hess_tmp39*weak_hess_tmp76 + weak_hess_tmp39*weak_hess_tmp78 + weak_hess_tmp84) + trial_grad[8]*(-weak_hess_tmp48*weak_hess_tmp76 + weak_hess_tmp48*weak_hess_tmp78 + weak_hess_tmp87);
        material[4] = trial_grad[0]*(-weak_hess_tmp3*weak_hess_tmp88 + weak_hess_tmp3*weak_hess_tmp90 + weak_hess_tmp47) + trial_grad[1]*(-weak_hess_tmp15*weak_hess_tmp88 + weak_hess_tmp15*weak_hess_tmp90 + weak_hess_tmp58) + trial_grad[2]*(-weak_hess_tmp23*weak_hess_tmp88 + weak_hess_tmp23*weak_hess_tmp90 + weak_hess_tmp75) + trial_grad[3]*(-weak_hess_tmp24*weak_hess_tmp88 + weak_hess_tmp24*weak_hess_tmp90 + weak_hess_tmp81) + trial_grad[4]*(mu + weak_hess_tmp10*pow_2(weak_hess_tmp46) - weak_hess_tmp43*weak_hess_tmp88 + weak_hess_tmp43*weak_hess_tmp90) + trial_grad[5]*(-weak_hess_tmp31*weak_hess_tmp88 + weak_hess_tmp31*weak_hess_tmp90 + weak_hess_tmp91) + trial_grad[6]*(-weak_hess_tmp29*weak_hess_tmp88 + weak_hess_tmp29*weak_hess_tmp90 + weak_hess_tmp93) + trial_grad[7]*(-weak_hess_tmp39*weak_hess_tmp88 + weak_hess_tmp39*weak_hess_tmp90 + weak_hess_tmp92) + trial_grad[8]*(-weak_hess_tmp48*weak_hess_tmp88 + weak_hess_tmp48*weak_hess_tmp90 + weak_hess_tmp96);
        material[5] = trial_grad[0]*(-weak_hess_tmp3*weak_hess_tmp97 + weak_hess_tmp3*weak_hess_tmp99 + weak_hess_tmp37) + trial_grad[1]*(-weak_hess_tmp15*weak_hess_tmp97 + weak_hess_tmp15*weak_hess_tmp99 + weak_hess_tmp61) + trial_grad[2]*(-weak_hess_tmp23*weak_hess_tmp97 + weak_hess_tmp23*weak_hess_tmp99 + weak_hess_tmp70) + trial_grad[3]*(-weak_hess_tmp24*weak_hess_tmp97 + weak_hess_tmp24*weak_hess_tmp99 + weak_hess_tmp79) + trial_grad[4]*(-weak_hess_tmp43*weak_hess_tmp97 + weak_hess_tmp43*weak_hess_tmp99 + weak_hess_tmp91) + trial_grad[5]*(mu + weak_hess_tmp10*pow_2(weak_hess_tmp30) - weak_hess_tmp31*weak_hess_tmp97 + weak_hess_tmp31*weak_hess_tmp99) + trial_grad[6]*(weak_hess_tmp101 - weak_hess_tmp29*weak_hess_tmp97 + weak_hess_tmp29*weak_hess_tmp99) + trial_grad[7]*(weak_hess_tmp102 - weak_hess_tmp39*weak_hess_tmp97 + weak_hess_tmp39*weak_hess_tmp99) + trial_grad[8]*(weak_hess_tmp100 - weak_hess_tmp48*weak_hess_tmp97 + weak_hess_tmp48*weak_hess_tmp99);
        material[6] = trial_grad[0]*(-weak_hess_tmp103*weak_hess_tmp3 + weak_hess_tmp105*weak_hess_tmp3 + weak_hess_tmp28) + trial_grad[1]*(-weak_hess_tmp103*weak_hess_tmp15 + weak_hess_tmp105*weak_hess_tmp15 + weak_hess_tmp62) + trial_grad[2]*(-weak_hess_tmp103*weak_hess_tmp23 + weak_hess_tmp105*weak_hess_tmp23 + weak_hess_tmp74) + trial_grad[3]*(-weak_hess_tmp103*weak_hess_tmp24 + weak_hess_tmp105*weak_hess_tmp24 + weak_hess_tmp80) + trial_grad[4]*(-weak_hess_tmp103*weak_hess_tmp43 + weak_hess_tmp105*weak_hess_tmp43 + weak_hess_tmp93) + trial_grad[5]*(weak_hess_tmp101 - weak_hess_tmp103*weak_hess_tmp31 + weak_hess_tmp105*weak_hess_tmp31) + trial_grad[6]*(mu + weak_hess_tmp10*pow_2(weak_hess_tmp27) - weak_hess_tmp103*weak_hess_tmp29 + weak_hess_tmp105*weak_hess_tmp29) + trial_grad[7]*(-weak_hess_tmp103*weak_hess_tmp39 + weak_hess_tmp105*weak_hess_tmp39 + weak_hess_tmp106) + trial_grad[8]*(-weak_hess_tmp103*weak_hess_tmp48 + weak_hess_tmp105*weak_hess_tmp48 + weak_hess_tmp107);
        material[7] = trial_grad[0]*(-weak_hess_tmp108*weak_hess_tmp3 + weak_hess_tmp110*weak_hess_tmp3 + weak_hess_tmp42) + trial_grad[1]*(-weak_hess_tmp108*weak_hess_tmp15 + weak_hess_tmp110*weak_hess_tmp15 + weak_hess_tmp57) + trial_grad[2]*(-weak_hess_tmp108*weak_hess_tmp23 + weak_hess_tmp110*weak_hess_tmp23 + weak_hess_tmp73) + trial_grad[3]*(-weak_hess_tmp108*weak_hess_tmp24 + weak_hess_tmp110*weak_hess_tmp24 + weak_hess_tmp84) + trial_grad[4]*(-weak_hess_tmp108*weak_hess_tmp43 + weak_hess_tmp110*weak_hess_tmp43 + weak_hess_tmp92) + trial_grad[5]*(weak_hess_tmp102 - weak_hess_tmp108*weak_hess_tmp31 + weak_hess_tmp110*weak_hess_tmp31) + trial_grad[6]*(weak_hess_tmp106 - weak_hess_tmp108*weak_hess_tmp29 + weak_hess_tmp110*weak_hess_tmp29) + trial_grad[7]*(mu + weak_hess_tmp10*pow_2(weak_hess_tmp38) - weak_hess_tmp108*weak_hess_tmp39 + weak_hess_tmp110*weak_hess_tmp39) + trial_grad[8]*(-weak_hess_tmp108*weak_hess_tmp48 + weak_hess_tmp110*weak_hess_tmp48 + weak_hess_tmp111);
        material[8] = trial_grad[0]*(-weak_hess_tmp112*weak_hess_tmp3 + weak_hess_tmp113*weak_hess_tmp3 + weak_hess_tmp52) + trial_grad[1]*(-weak_hess_tmp112*weak_hess_tmp15 + weak_hess_tmp113*weak_hess_tmp15 + weak_hess_tmp66) + trial_grad[2]*(-weak_hess_tmp112*weak_hess_tmp23 + weak_hess_tmp113*weak_hess_tmp23 + weak_hess_tmp71) + trial_grad[3]*(-weak_hess_tmp112*weak_hess_tmp24 + weak_hess_tmp113*weak_hess_tmp24 + weak_hess_tmp87) + trial_grad[4]*(-weak_hess_tmp112*weak_hess_tmp43 + weak_hess_tmp113*weak_hess_tmp43 + weak_hess_tmp96) + trial_grad[5]*(weak_hess_tmp100 - weak_hess_tmp112*weak_hess_tmp31 + weak_hess_tmp113*weak_hess_tmp31) + trial_grad[6]*(weak_hess_tmp107 - weak_hess_tmp112*weak_hess_tmp29 + weak_hess_tmp113*weak_hess_tmp29) + trial_grad[7]*(weak_hess_tmp111 - weak_hess_tmp112*weak_hess_tmp39 + weak_hess_tmp113*weak_hess_tmp39) + trial_grad[8]*(mu + weak_hess_tmp10*pow_2(weak_hess_tmp51) - weak_hess_tmp112*weak_hess_tmp48 + weak_hess_tmp113*weak_hess_tmp48);
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
