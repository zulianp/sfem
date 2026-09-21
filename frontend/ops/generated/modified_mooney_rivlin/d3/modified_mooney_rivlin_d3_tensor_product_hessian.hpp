#ifndef MODIFIED_MOONEY_RIVLIN_D3_TENSOR_PRODUCT_HESSIAN_HPP
#define MODIFIED_MOONEY_RIVLIN_D3_TENSOR_PRODUCT_HESSIAN_HPP
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
static SFEM_INLINE void modified_mooney_rivlin_d3_tensor_product_direct_hessian_tensor_product_element_matrix(
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
    const s_t c1,
    const s_t c2,
    const s_t kappa,
    const s_t bu_data[NS * 3][VS],
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
  s_t state_gradient_ref[NC * NQ * ND];
  for (int component = 0; component < NC; ++component) {
    tensor_gradient_contiguous_scalar<s_t, NQ, NS, VS, 3, NC>(
        shape_1d, grad_1d, bu_data, component,
        state_gradient_ref + component * NQ * ND);
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
        const s_t gu_ref0 = state_gradient_ref[q * ND];
        const s_t gu_ref1 = state_gradient_ref[q * ND + 1];
        const s_t gu_ref2 = state_gradient_ref[q * ND + 2];
        const s_t gu_ref3 = state_gradient_ref[NQ * ND + q * ND];
        const s_t gu_ref4 = state_gradient_ref[NQ * ND + q * ND + 1];
        const s_t gu_ref5 = state_gradient_ref[NQ * ND + q * ND + 2];
        const s_t gu_ref6 = state_gradient_ref[2 * NQ * ND + q * ND];
        const s_t gu_ref7 = state_gradient_ref[2 * NQ * ND + q * ND + 1];
        const s_t gu_ref8 = state_gradient_ref[2 * NQ * ND + q * ND + 2];
        const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane3 + gu_ref2 * adj_lane6) * idet;
        const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane4 + gu_ref2 * adj_lane7) * idet;
        const s_t gu2 = (gu_ref0 * adj_lane2 + gu_ref1 * adj_lane5 + gu_ref2 * adj_lane8) * idet;
        const s_t gu3 = (gu_ref3 * adj_lane0 + gu_ref4 * adj_lane3 + gu_ref5 * adj_lane6) * idet;
        const s_t gu4 = (gu_ref3 * adj_lane1 + gu_ref4 * adj_lane4 + gu_ref5 * adj_lane7) * idet;
        const s_t gu5 = (gu_ref3 * adj_lane2 + gu_ref4 * adj_lane5 + gu_ref5 * adj_lane8) * idet;
        const s_t gu6 = (gu_ref6 * adj_lane0 + gu_ref7 * adj_lane3 + gu_ref8 * adj_lane6) * idet;
        const s_t gu7 = (gu_ref6 * adj_lane1 + gu_ref7 * adj_lane4 + gu_ref8 * adj_lane7) * idet;
        const s_t gu8 = (gu_ref6 * adj_lane2 + gu_ref7 * adj_lane5 + gu_ref8 * adj_lane8) * idet;
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
        const s_t weak_hess_tmp0 = gu5*gu7;
        const s_t weak_hess_tmp1 = gu4 + s_t(1);
        const s_t weak_hess_tmp2 = gu8 + s_t(1);
        const s_t weak_hess_tmp3 = weak_hess_tmp0 - weak_hess_tmp1*weak_hess_tmp2;
        const s_t weak_hess_tmp4 = -weak_hess_tmp3;
        const s_t weak_hess_tmp5 = gu1*gu3;
        const s_t weak_hess_tmp6 = gu2*gu6;
        const s_t weak_hess_tmp7 = gu0 + s_t(1);
        const s_t weak_hess_tmp8 = gu1*gu5*gu6 + gu2*gu3*gu7 - weak_hess_tmp0*weak_hess_tmp7 + weak_hess_tmp1*weak_hess_tmp2*weak_hess_tmp7 - weak_hess_tmp1*weak_hess_tmp6 - weak_hess_tmp2*weak_hess_tmp5;
        const s_t weak_hess_tmp9 = kappa/pow_2(weak_hess_tmp8);
        const s_t weak_hess_tmp10 = gu0*gu4;
        const s_t weak_hess_tmp11 = gu5*gu6;
        const s_t weak_hess_tmp12 = gu3*gu7;
        const s_t weak_hess_tmp13 = sfem_log1p(gu0*gu8 - gu0*weak_hess_tmp0 + gu0 + gu1*weak_hess_tmp11 + gu2*weak_hess_tmp12 + gu4*gu8 - gu4*weak_hess_tmp6 + gu4 + gu8*weak_hess_tmp10 - gu8*weak_hess_tmp5 + gu8 - weak_hess_tmp0 + weak_hess_tmp10 - weak_hess_tmp5 - weak_hess_tmp6);
        const s_t weak_hess_tmp14 = weak_hess_tmp4*weak_hess_tmp9;
        const s_t weak_hess_tmp15 = weak_hess_tmp13*weak_hess_tmp14;
        const s_t weak_hess_tmp16 = s_t(2)/pow(weak_hess_tmp8, (s_t(2) / s_t(3)));
        const s_t weak_hess_tmp17 = pow(weak_hess_tmp8, (s_t(-5) / s_t(3)));
        const s_t weak_hess_tmp18 = weak_hess_tmp1*weak_hess_tmp2;
        const s_t weak_hess_tmp19 = ((s_t(2) / s_t(3)))*weak_hess_tmp0 - (s_t(2) / s_t(3))*weak_hess_tmp18;
        const s_t weak_hess_tmp20 = weak_hess_tmp17*weak_hess_tmp19;
        const s_t weak_hess_tmp21 = ((s_t(5) / s_t(3)))*weak_hess_tmp0 - (s_t(5) / s_t(3))*weak_hess_tmp18;
        const s_t weak_hess_tmp22 = pow_2(gu3);
        const s_t weak_hess_tmp23 = pow_2(gu6);
        const s_t weak_hess_tmp24 = pow_2(weak_hess_tmp7);
        const s_t weak_hess_tmp25 = weak_hess_tmp22 + weak_hess_tmp23 + weak_hess_tmp24;
        const s_t weak_hess_tmp26 = pow_2(gu1);
        const s_t weak_hess_tmp27 = pow_2(gu7);
        const s_t weak_hess_tmp28 = pow_2(weak_hess_tmp1);
        const s_t weak_hess_tmp29 = weak_hess_tmp26 + weak_hess_tmp27 + weak_hess_tmp28;
        const s_t weak_hess_tmp30 = pow_2(gu2);
        const s_t weak_hess_tmp31 = pow_2(gu5);
        const s_t weak_hess_tmp32 = pow_2(weak_hess_tmp2);
        const s_t weak_hess_tmp33 = weak_hess_tmp30 + weak_hess_tmp31 + weak_hess_tmp32;
        const s_t weak_hess_tmp34 = weak_hess_tmp25 + weak_hess_tmp29 + weak_hess_tmp33;
        const s_t weak_hess_tmp35 = weak_hess_tmp34/pow(weak_hess_tmp8, (s_t(8) / s_t(3)));
        const s_t weak_hess_tmp36 = weak_hess_tmp19*weak_hess_tmp35;
        const s_t weak_hess_tmp37 = s_t(2)*weak_hess_tmp31;
        const s_t weak_hess_tmp38 = s_t(2)*weak_hess_tmp32;
        const s_t weak_hess_tmp39 = weak_hess_tmp37 + weak_hess_tmp38;
        const s_t weak_hess_tmp40 = s_t(2)*weak_hess_tmp27;
        const s_t weak_hess_tmp41 = s_t(2)*weak_hess_tmp28;
        const s_t weak_hess_tmp42 = weak_hess_tmp40 + weak_hess_tmp41;
        const s_t weak_hess_tmp43 = pow(weak_hess_tmp8, (s_t(-4) / s_t(3)));
        const s_t weak_hess_tmp44 = ((s_t(4) / s_t(3)))*weak_hess_tmp0 - (s_t(4) / s_t(3))*weak_hess_tmp18;
        const s_t weak_hess_tmp45 = pow(weak_hess_tmp8, (s_t(-7) / s_t(3)));
        const s_t weak_hess_tmp46 = gu6*gu7;
        const s_t weak_hess_tmp47 = gu1*weak_hess_tmp7;
        const s_t weak_hess_tmp48 = gu3*weak_hess_tmp1;
        const s_t weak_hess_tmp49 = weak_hess_tmp46 + weak_hess_tmp47 + weak_hess_tmp48;
        const s_t weak_hess_tmp50 = s_t(2)*gu1;
        const s_t weak_hess_tmp51 = gu3*gu5;
        const s_t weak_hess_tmp52 = gu2*weak_hess_tmp7;
        const s_t weak_hess_tmp53 = gu6*weak_hess_tmp2;
        const s_t weak_hess_tmp54 = weak_hess_tmp51 + weak_hess_tmp52 + weak_hess_tmp53;
        const s_t weak_hess_tmp55 = s_t(2)*gu2;
        const s_t weak_hess_tmp56 = s_t(2)*weak_hess_tmp7;
        const s_t weak_hess_tmp57 = weak_hess_tmp45*(-weak_hess_tmp25*weak_hess_tmp56 + s_t(2)*weak_hess_tmp34*weak_hess_tmp7 - weak_hess_tmp49*weak_hess_tmp50 - weak_hess_tmp54*weak_hess_tmp55);
        const s_t weak_hess_tmp58 = ((s_t(7) / s_t(3)))*weak_hess_tmp0 - (s_t(7) / s_t(3))*weak_hess_tmp18;
        const s_t weak_hess_tmp59 = gu1*gu2;
        const s_t weak_hess_tmp60 = gu5*weak_hess_tmp1;
        const s_t weak_hess_tmp61 = gu7*weak_hess_tmp2;
        const s_t weak_hess_tmp62 = weak_hess_tmp59 + weak_hess_tmp60 + weak_hess_tmp61;
        const s_t weak_hess_tmp63 = -(s_t(1) / s_t(2))*pow_2(weak_hess_tmp25) - (s_t(1) / s_t(2))*pow_2(weak_hess_tmp29) - (s_t(1) / s_t(2))*pow_2(weak_hess_tmp33) + ((s_t(1) / s_t(2)))*pow_2(weak_hess_tmp34) - pow_2(weak_hess_tmp49) - pow_2(weak_hess_tmp54) - pow_2(weak_hess_tmp62);
        const s_t weak_hess_tmp64 = weak_hess_tmp63/pow(weak_hess_tmp8, (s_t(10) / s_t(3)));
        const s_t weak_hess_tmp65 = weak_hess_tmp44*weak_hess_tmp64;
        const s_t weak_hess_tmp66 = gu3*weak_hess_tmp2;
        const s_t weak_hess_tmp67 = -gu5*gu6 + weak_hess_tmp66;
        const s_t weak_hess_tmp68 = -weak_hess_tmp67;
        const s_t weak_hess_tmp69 = weak_hess_tmp14*weak_hess_tmp68;
        const s_t weak_hess_tmp70 = -(s_t(5) / s_t(3))*weak_hess_tmp11 + ((s_t(5) / s_t(3)))*weak_hess_tmp66;
        const s_t weak_hess_tmp71 = -(s_t(2) / s_t(3))*weak_hess_tmp11 + ((s_t(2) / s_t(3)))*weak_hess_tmp66;
        const s_t weak_hess_tmp72 = weak_hess_tmp17*weak_hess_tmp56;
        const s_t weak_hess_tmp73 = weak_hess_tmp20*weak_hess_tmp50 + weak_hess_tmp71*weak_hess_tmp72;
        const s_t weak_hess_tmp74 = -(s_t(7) / s_t(3))*weak_hess_tmp11 + ((s_t(7) / s_t(3)))*weak_hess_tmp66;
        const s_t weak_hess_tmp75 = s_t(2)*weak_hess_tmp46;
        const s_t weak_hess_tmp76 = s_t(2)*weak_hess_tmp48;
        const s_t weak_hess_tmp77 = -(s_t(4) / s_t(3))*weak_hess_tmp11 + ((s_t(4) / s_t(3)))*weak_hess_tmp66;
        const s_t weak_hess_tmp78 = s_t(2)*gu1*weak_hess_tmp34 - weak_hess_tmp29*weak_hess_tmp50 - weak_hess_tmp49*weak_hess_tmp56 - weak_hess_tmp55*weak_hess_tmp62;
        const s_t weak_hess_tmp79 = weak_hess_tmp44*weak_hess_tmp45;
        const s_t weak_hess_tmp80 = weak_hess_tmp43*(-weak_hess_tmp75 - weak_hess_tmp76) + weak_hess_tmp57*weak_hess_tmp77 + weak_hess_tmp78*weak_hess_tmp79;
        const s_t weak_hess_tmp81 = gu6*weak_hess_tmp1;
        const s_t weak_hess_tmp82 = weak_hess_tmp12 - weak_hess_tmp81;
        const s_t weak_hess_tmp83 = weak_hess_tmp14*weak_hess_tmp82;
        const s_t weak_hess_tmp84 = -weak_hess_tmp82;
        const s_t weak_hess_tmp85 = ((s_t(5) / s_t(3)))*gu6*weak_hess_tmp1 - (s_t(5) / s_t(3))*weak_hess_tmp12;
        const s_t weak_hess_tmp86 = ((s_t(2) / s_t(3)))*gu6*weak_hess_tmp1 - (s_t(2) / s_t(3))*weak_hess_tmp12;
        const s_t weak_hess_tmp87 = weak_hess_tmp20*weak_hess_tmp55 + weak_hess_tmp72*weak_hess_tmp86;
        const s_t weak_hess_tmp88 = ((s_t(7) / s_t(3)))*gu6*weak_hess_tmp1 - (s_t(7) / s_t(3))*weak_hess_tmp12;
        const s_t weak_hess_tmp89 = s_t(2)*weak_hess_tmp51;
        const s_t weak_hess_tmp90 = s_t(2)*weak_hess_tmp53;
        const s_t weak_hess_tmp91 = ((s_t(4) / s_t(3)))*gu6*weak_hess_tmp1 - (s_t(4) / s_t(3))*weak_hess_tmp12;
        const s_t weak_hess_tmp92 = s_t(2)*gu2*weak_hess_tmp34 - weak_hess_tmp33*weak_hess_tmp55 - weak_hess_tmp50*weak_hess_tmp62 - weak_hess_tmp54*weak_hess_tmp56;
        const s_t weak_hess_tmp93 = weak_hess_tmp43*(-weak_hess_tmp89 - weak_hess_tmp90) + weak_hess_tmp57*weak_hess_tmp91 + weak_hess_tmp79*weak_hess_tmp92;
        const s_t weak_hess_tmp94 = gu1*weak_hess_tmp2;
        const s_t weak_hess_tmp95 = -gu2*gu7 + weak_hess_tmp94;
        const s_t weak_hess_tmp96 = -weak_hess_tmp95;
        const s_t weak_hess_tmp97 = weak_hess_tmp14*weak_hess_tmp96;
        const s_t weak_hess_tmp98 = gu2*gu7;
        const s_t weak_hess_tmp99 = ((s_t(5) / s_t(3)))*weak_hess_tmp94 - (s_t(5) / s_t(3))*weak_hess_tmp98;
        const s_t weak_hess_tmp100 = s_t(2)*gu3;
        const s_t weak_hess_tmp101 = ((s_t(2) / s_t(3)))*weak_hess_tmp94 - (s_t(2) / s_t(3))*weak_hess_tmp98;
        const s_t weak_hess_tmp102 = weak_hess_tmp100*weak_hess_tmp20 + weak_hess_tmp101*weak_hess_tmp72;
        const s_t weak_hess_tmp103 = ((s_t(7) / s_t(3)))*weak_hess_tmp94 - (s_t(7) / s_t(3))*weak_hess_tmp98;
        const s_t weak_hess_tmp104 = gu5*weak_hess_tmp55;
        const s_t weak_hess_tmp105 = weak_hess_tmp1*weak_hess_tmp50;
        const s_t weak_hess_tmp106 = ((s_t(4) / s_t(3)))*weak_hess_tmp94 - (s_t(4) / s_t(3))*weak_hess_tmp98;
        const s_t weak_hess_tmp107 = s_t(2)*gu5;
        const s_t weak_hess_tmp108 = s_t(2)*weak_hess_tmp1;
        const s_t weak_hess_tmp109 = s_t(2)*gu3*weak_hess_tmp34 - weak_hess_tmp100*weak_hess_tmp25 - weak_hess_tmp107*weak_hess_tmp54 - weak_hess_tmp108*weak_hess_tmp49;
        const s_t weak_hess_tmp110 = weak_hess_tmp106*weak_hess_tmp57 + weak_hess_tmp109*weak_hess_tmp79 + weak_hess_tmp43*(-weak_hess_tmp104 - weak_hess_tmp105);
        const s_t weak_hess_tmp111 = gu1*gu5;
        const s_t weak_hess_tmp112 = gu2*weak_hess_tmp1;
        const s_t weak_hess_tmp113 = weak_hess_tmp111 - weak_hess_tmp112;
        const s_t weak_hess_tmp114 = weak_hess_tmp113*weak_hess_tmp14;
        const s_t weak_hess_tmp115 = -weak_hess_tmp113;
        const s_t weak_hess_tmp116 = ((s_t(5) / s_t(3)))*gu2*weak_hess_tmp1 - (s_t(5) / s_t(3))*weak_hess_tmp111;
        const s_t weak_hess_tmp117 = s_t(2)*gu6;
        const s_t weak_hess_tmp118 = ((s_t(2) / s_t(3)))*gu2*weak_hess_tmp1 - (s_t(2) / s_t(3))*weak_hess_tmp111;
        const s_t weak_hess_tmp119 = weak_hess_tmp117*weak_hess_tmp20 + weak_hess_tmp118*weak_hess_tmp72;
        const s_t weak_hess_tmp120 = ((s_t(7) / s_t(3)))*gu2*weak_hess_tmp1 - (s_t(7) / s_t(3))*weak_hess_tmp111;
        const s_t weak_hess_tmp121 = gu7*weak_hess_tmp50;
        const s_t weak_hess_tmp122 = weak_hess_tmp2*weak_hess_tmp55;
        const s_t weak_hess_tmp123 = ((s_t(4) / s_t(3)))*gu2*weak_hess_tmp1 - (s_t(4) / s_t(3))*weak_hess_tmp111;
        const s_t weak_hess_tmp124 = s_t(2)*gu7;
        const s_t weak_hess_tmp125 = s_t(2)*weak_hess_tmp2;
        const s_t weak_hess_tmp126 = s_t(2)*gu6*weak_hess_tmp34 - weak_hess_tmp117*weak_hess_tmp25 - weak_hess_tmp124*weak_hess_tmp49 - weak_hess_tmp125*weak_hess_tmp54;
        const s_t weak_hess_tmp127 = weak_hess_tmp123*weak_hess_tmp57 + weak_hess_tmp126*weak_hess_tmp79 + weak_hess_tmp43*(-weak_hess_tmp121 - weak_hess_tmp122);
        const s_t weak_hess_tmp128 = gu1*gu6;
        const s_t weak_hess_tmp129 = ((s_t(5) / s_t(3)))*gu7*weak_hess_tmp7 - (s_t(5) / s_t(3))*weak_hess_tmp128;
        const s_t weak_hess_tmp130 = ((s_t(2) / s_t(3)))*gu7*weak_hess_tmp7 - (s_t(2) / s_t(3))*weak_hess_tmp128;
        const s_t weak_hess_tmp131 = ((s_t(2) / s_t(3)))*weak_hess_tmp34;
        const s_t weak_hess_tmp132 = weak_hess_tmp131*weak_hess_tmp17;
        const s_t weak_hess_tmp133 = gu7*weak_hess_tmp132;
        const s_t weak_hess_tmp134 = weak_hess_tmp107*weak_hess_tmp20 + weak_hess_tmp130*weak_hess_tmp72 + weak_hess_tmp133;
        const s_t weak_hess_tmp135 = ((s_t(7) / s_t(3)))*gu7*weak_hess_tmp7 - (s_t(7) / s_t(3))*weak_hess_tmp128;
        const s_t weak_hess_tmp136 = gu2*gu3;
        const s_t weak_hess_tmp137 = ((s_t(4) / s_t(3)))*gu7*weak_hess_tmp7 - (s_t(4) / s_t(3))*weak_hess_tmp128;
        const s_t weak_hess_tmp138 = s_t(2)*gu5*weak_hess_tmp34 - weak_hess_tmp100*weak_hess_tmp54 - weak_hess_tmp107*weak_hess_tmp33 - weak_hess_tmp108*weak_hess_tmp62;
        const s_t weak_hess_tmp139 = ((s_t(4) / s_t(3)))*weak_hess_tmp45*weak_hess_tmp63;
        const s_t weak_hess_tmp140 = gu7*weak_hess_tmp139;
        const s_t weak_hess_tmp141 = weak_hess_tmp137*weak_hess_tmp57 + weak_hess_tmp138*weak_hess_tmp79 + weak_hess_tmp140 + weak_hess_tmp43*(s_t(4)*gu5*weak_hess_tmp7 - s_t(2)*weak_hess_tmp136);
        const s_t weak_hess_tmp142 = gu7*weak_hess_tmp7;
        const s_t weak_hess_tmp143 = weak_hess_tmp128 - weak_hess_tmp142;
        const s_t weak_hess_tmp144 = -weak_hess_tmp143;
        const s_t weak_hess_tmp145 = kappa*weak_hess_tmp13/weak_hess_tmp8;
        const s_t weak_hess_tmp146 = gu7*weak_hess_tmp145;
        const s_t weak_hess_tmp147 = weak_hess_tmp14*weak_hess_tmp143 - weak_hess_tmp146;
        const s_t weak_hess_tmp148 = ((s_t(5) / s_t(3)))*gu5*weak_hess_tmp7 - (s_t(5) / s_t(3))*weak_hess_tmp136;
        const s_t weak_hess_tmp149 = ((s_t(2) / s_t(3)))*gu5*weak_hess_tmp7 - (s_t(2) / s_t(3))*weak_hess_tmp136;
        const s_t weak_hess_tmp150 = gu5*weak_hess_tmp132;
        const s_t weak_hess_tmp151 = weak_hess_tmp124*weak_hess_tmp20 + weak_hess_tmp149*weak_hess_tmp72 + weak_hess_tmp150;
        const s_t weak_hess_tmp152 = ((s_t(7) / s_t(3)))*gu5*weak_hess_tmp7 - (s_t(7) / s_t(3))*weak_hess_tmp136;
        const s_t weak_hess_tmp153 = ((s_t(4) / s_t(3)))*gu5*weak_hess_tmp7 - (s_t(4) / s_t(3))*weak_hess_tmp136;
        const s_t weak_hess_tmp154 = s_t(2)*gu7*weak_hess_tmp34 - weak_hess_tmp117*weak_hess_tmp49 - weak_hess_tmp124*weak_hess_tmp29 - weak_hess_tmp125*weak_hess_tmp62;
        const s_t weak_hess_tmp155 = gu5*weak_hess_tmp139;
        const s_t weak_hess_tmp156 = weak_hess_tmp153*weak_hess_tmp57 + weak_hess_tmp154*weak_hess_tmp79 + weak_hess_tmp155 + weak_hess_tmp43*(s_t(4)*gu7*weak_hess_tmp7 - s_t(2)*weak_hess_tmp128);
        const s_t weak_hess_tmp157 = gu5*weak_hess_tmp7;
        const s_t weak_hess_tmp158 = weak_hess_tmp136 - weak_hess_tmp157;
        const s_t weak_hess_tmp159 = -weak_hess_tmp158;
        const s_t weak_hess_tmp160 = gu5*weak_hess_tmp145;
        const s_t weak_hess_tmp161 = weak_hess_tmp14*weak_hess_tmp158 - weak_hess_tmp160;
        const s_t weak_hess_tmp162 = weak_hess_tmp2*weak_hess_tmp7;
        const s_t weak_hess_tmp163 = -(s_t(5) / s_t(3))*weak_hess_tmp162 + ((s_t(5) / s_t(3)))*weak_hess_tmp6;
        const s_t weak_hess_tmp164 = -(s_t(2) / s_t(3))*weak_hess_tmp162 + ((s_t(2) / s_t(3)))*weak_hess_tmp6;
        const s_t weak_hess_tmp165 = weak_hess_tmp17*weak_hess_tmp2;
        const s_t weak_hess_tmp166 = weak_hess_tmp131*weak_hess_tmp165;
        const s_t weak_hess_tmp167 = weak_hess_tmp108*weak_hess_tmp20 + weak_hess_tmp164*weak_hess_tmp72 - weak_hess_tmp166;
        const s_t weak_hess_tmp168 = -(s_t(7) / s_t(3))*weak_hess_tmp162 + ((s_t(7) / s_t(3)))*weak_hess_tmp6;
        const s_t weak_hess_tmp169 = -(s_t(4) / s_t(3))*weak_hess_tmp162 + ((s_t(4) / s_t(3)))*weak_hess_tmp6;
        const s_t weak_hess_tmp170 = s_t(2)*weak_hess_tmp1*weak_hess_tmp34 - weak_hess_tmp100*weak_hess_tmp49 - weak_hess_tmp107*weak_hess_tmp62 - weak_hess_tmp108*weak_hess_tmp29;
        const s_t weak_hess_tmp171 = weak_hess_tmp139*weak_hess_tmp2;
        const s_t weak_hess_tmp172 = weak_hess_tmp169*weak_hess_tmp57 + weak_hess_tmp170*weak_hess_tmp79 - weak_hess_tmp171 + weak_hess_tmp43*(s_t(4)*weak_hess_tmp1*weak_hess_tmp7 - s_t(2)*weak_hess_tmp5);
        const s_t weak_hess_tmp173 = -weak_hess_tmp2*weak_hess_tmp7 + weak_hess_tmp6;
        const s_t weak_hess_tmp174 = weak_hess_tmp145*weak_hess_tmp2;
        const s_t weak_hess_tmp175 = -weak_hess_tmp173;
        const s_t weak_hess_tmp176 = weak_hess_tmp14*weak_hess_tmp175 + weak_hess_tmp174;
        const s_t weak_hess_tmp177 = weak_hess_tmp1*weak_hess_tmp7;
        const s_t weak_hess_tmp178 = -(s_t(5) / s_t(3))*weak_hess_tmp177 + ((s_t(5) / s_t(3)))*weak_hess_tmp5;
        const s_t weak_hess_tmp179 = -(s_t(2) / s_t(3))*weak_hess_tmp177 + ((s_t(2) / s_t(3)))*weak_hess_tmp5;
        const s_t weak_hess_tmp180 = weak_hess_tmp1*weak_hess_tmp132;
        const s_t weak_hess_tmp181 = weak_hess_tmp125*weak_hess_tmp20 + weak_hess_tmp179*weak_hess_tmp72 - weak_hess_tmp180;
        const s_t weak_hess_tmp182 = -(s_t(7) / s_t(3))*weak_hess_tmp177 + ((s_t(7) / s_t(3)))*weak_hess_tmp5;
        const s_t weak_hess_tmp183 = -(s_t(4) / s_t(3))*weak_hess_tmp177 + ((s_t(4) / s_t(3)))*weak_hess_tmp5;
        const s_t weak_hess_tmp184 = -weak_hess_tmp117*weak_hess_tmp54 - weak_hess_tmp124*weak_hess_tmp62 - weak_hess_tmp125*weak_hess_tmp33 + s_t(2)*weak_hess_tmp2*weak_hess_tmp34;
        const s_t weak_hess_tmp185 = weak_hess_tmp1*weak_hess_tmp139;
        const s_t weak_hess_tmp186 = weak_hess_tmp183*weak_hess_tmp57 + weak_hess_tmp184*weak_hess_tmp79 - weak_hess_tmp185 + weak_hess_tmp43*(s_t(4)*weak_hess_tmp2*weak_hess_tmp7 - s_t(2)*weak_hess_tmp6);
        const s_t weak_hess_tmp187 = -weak_hess_tmp1*weak_hess_tmp7 + weak_hess_tmp5;
        const s_t weak_hess_tmp188 = weak_hess_tmp1*weak_hess_tmp145;
        const s_t weak_hess_tmp189 = -weak_hess_tmp187;
        const s_t weak_hess_tmp190 = weak_hess_tmp14*weak_hess_tmp189 + weak_hess_tmp188;
        const s_t weak_hess_tmp191 = weak_hess_tmp68*weak_hess_tmp9;
        const s_t weak_hess_tmp192 = weak_hess_tmp13*weak_hess_tmp191;
        const s_t weak_hess_tmp193 = weak_hess_tmp17*weak_hess_tmp71;
        const s_t weak_hess_tmp194 = weak_hess_tmp35*weak_hess_tmp71;
        const s_t weak_hess_tmp195 = s_t(2)*weak_hess_tmp22;
        const s_t weak_hess_tmp196 = s_t(2)*weak_hess_tmp23;
        const s_t weak_hess_tmp197 = weak_hess_tmp195 + weak_hess_tmp196;
        const s_t weak_hess_tmp198 = weak_hess_tmp45*weak_hess_tmp78;
        const s_t weak_hess_tmp199 = weak_hess_tmp64*weak_hess_tmp77;
        const s_t weak_hess_tmp200 = weak_hess_tmp191*weak_hess_tmp82;
        const s_t weak_hess_tmp201 = weak_hess_tmp17*weak_hess_tmp50;
        const s_t weak_hess_tmp202 = weak_hess_tmp193*weak_hess_tmp55 + weak_hess_tmp201*weak_hess_tmp86;
        const s_t weak_hess_tmp203 = s_t(2)*weak_hess_tmp60;
        const s_t weak_hess_tmp204 = s_t(2)*weak_hess_tmp61;
        const s_t weak_hess_tmp205 = weak_hess_tmp45*weak_hess_tmp77;
        const s_t weak_hess_tmp206 = weak_hess_tmp198*weak_hess_tmp91 + weak_hess_tmp205*weak_hess_tmp92 + weak_hess_tmp43*(-weak_hess_tmp203 - weak_hess_tmp204);
        const s_t weak_hess_tmp207 = weak_hess_tmp158*weak_hess_tmp191;
        const s_t weak_hess_tmp208 = weak_hess_tmp124*weak_hess_tmp193 + weak_hess_tmp149*weak_hess_tmp201;
        const s_t weak_hess_tmp209 = gu6*weak_hess_tmp56;
        const s_t weak_hess_tmp210 = weak_hess_tmp153*weak_hess_tmp198 + weak_hess_tmp154*weak_hess_tmp205 + weak_hess_tmp43*(-weak_hess_tmp122 - weak_hess_tmp209);
        const s_t weak_hess_tmp211 = weak_hess_tmp175*weak_hess_tmp191;
        const s_t weak_hess_tmp212 = weak_hess_tmp108*weak_hess_tmp193 + weak_hess_tmp164*weak_hess_tmp201;
        const s_t weak_hess_tmp213 = gu3*weak_hess_tmp56;
        const s_t weak_hess_tmp214 = weak_hess_tmp169*weak_hess_tmp198 + weak_hess_tmp170*weak_hess_tmp205 + weak_hess_tmp43*(-weak_hess_tmp104 - weak_hess_tmp213);
        const s_t weak_hess_tmp215 = gu6*weak_hess_tmp132;
        const s_t weak_hess_tmp216 = weak_hess_tmp107*weak_hess_tmp193 + weak_hess_tmp130*weak_hess_tmp201 - weak_hess_tmp215;
        const s_t weak_hess_tmp217 = gu6*weak_hess_tmp139;
        const s_t weak_hess_tmp218 = weak_hess_tmp137*weak_hess_tmp198 + weak_hess_tmp138*weak_hess_tmp205 - weak_hess_tmp217 + weak_hess_tmp43*(s_t(4)*weak_hess_tmp111 - s_t(2)*weak_hess_tmp112);
        const s_t weak_hess_tmp219 = gu6*weak_hess_tmp145;
        const s_t weak_hess_tmp220 = weak_hess_tmp143*weak_hess_tmp191 + weak_hess_tmp219;
        const s_t weak_hess_tmp221 = weak_hess_tmp117*weak_hess_tmp193 + weak_hess_tmp118*weak_hess_tmp201 - weak_hess_tmp150;
        const s_t weak_hess_tmp222 = weak_hess_tmp123*weak_hess_tmp198 + weak_hess_tmp126*weak_hess_tmp205 - weak_hess_tmp155 + weak_hess_tmp43*(s_t(4)*weak_hess_tmp128 - s_t(2)*weak_hess_tmp142);
        const s_t weak_hess_tmp223 = weak_hess_tmp113*weak_hess_tmp191 + weak_hess_tmp160;
        const s_t weak_hess_tmp224 = weak_hess_tmp100*weak_hess_tmp193 + weak_hess_tmp101*weak_hess_tmp201 + weak_hess_tmp166;
        const s_t weak_hess_tmp225 = weak_hess_tmp106*weak_hess_tmp198 + weak_hess_tmp109*weak_hess_tmp205 + weak_hess_tmp171 + weak_hess_tmp43*(-s_t(2)*weak_hess_tmp177 + s_t(4)*weak_hess_tmp5);
        const s_t weak_hess_tmp226 = -weak_hess_tmp174 + weak_hess_tmp191*weak_hess_tmp96;
        const s_t weak_hess_tmp227 = gu3*weak_hess_tmp132;
        const s_t weak_hess_tmp228 = weak_hess_tmp125*weak_hess_tmp193 + weak_hess_tmp179*weak_hess_tmp201 + weak_hess_tmp227;
        const s_t weak_hess_tmp229 = gu3*weak_hess_tmp139;
        const s_t weak_hess_tmp230 = weak_hess_tmp183*weak_hess_tmp198 + weak_hess_tmp184*weak_hess_tmp205 + weak_hess_tmp229 + weak_hess_tmp43*(s_t(4)*weak_hess_tmp94 - s_t(2)*weak_hess_tmp98);
        const s_t weak_hess_tmp231 = gu3*weak_hess_tmp145;
        const s_t weak_hess_tmp232 = weak_hess_tmp189*weak_hess_tmp191 - weak_hess_tmp231;
        const s_t weak_hess_tmp233 = weak_hess_tmp82*weak_hess_tmp9;
        const s_t weak_hess_tmp234 = weak_hess_tmp13*weak_hess_tmp233;
        const s_t weak_hess_tmp235 = weak_hess_tmp17*weak_hess_tmp86;
        const s_t weak_hess_tmp236 = weak_hess_tmp35*weak_hess_tmp86;
        const s_t weak_hess_tmp237 = weak_hess_tmp45*weak_hess_tmp92;
        const s_t weak_hess_tmp238 = weak_hess_tmp64*weak_hess_tmp91;
        const s_t weak_hess_tmp239 = weak_hess_tmp143*weak_hess_tmp233;
        const s_t weak_hess_tmp240 = weak_hess_tmp17*weak_hess_tmp55;
        const s_t weak_hess_tmp241 = weak_hess_tmp107*weak_hess_tmp235 + weak_hess_tmp130*weak_hess_tmp240;
        const s_t weak_hess_tmp242 = weak_hess_tmp45*weak_hess_tmp91;
        const s_t weak_hess_tmp243 = weak_hess_tmp137*weak_hess_tmp237 + weak_hess_tmp138*weak_hess_tmp242 + weak_hess_tmp43*(-weak_hess_tmp105 - weak_hess_tmp213);
        const s_t weak_hess_tmp244 = weak_hess_tmp189*weak_hess_tmp233;
        const s_t weak_hess_tmp245 = weak_hess_tmp125*weak_hess_tmp235 + weak_hess_tmp179*weak_hess_tmp240;
        const s_t weak_hess_tmp246 = weak_hess_tmp183*weak_hess_tmp237 + weak_hess_tmp184*weak_hess_tmp242 + weak_hess_tmp43*(-weak_hess_tmp121 - weak_hess_tmp209);
        const s_t weak_hess_tmp247 = weak_hess_tmp100*weak_hess_tmp235 + weak_hess_tmp101*weak_hess_tmp240 - weak_hess_tmp133;
        const s_t weak_hess_tmp248 = weak_hess_tmp106*weak_hess_tmp237 + weak_hess_tmp109*weak_hess_tmp242 - weak_hess_tmp140 + weak_hess_tmp43*(s_t(4)*weak_hess_tmp136 - s_t(2)*weak_hess_tmp157);
        const s_t weak_hess_tmp249 = weak_hess_tmp146 + weak_hess_tmp233*weak_hess_tmp96;
        const s_t weak_hess_tmp250 = weak_hess_tmp124*weak_hess_tmp235 + weak_hess_tmp149*weak_hess_tmp240 - weak_hess_tmp227;
        const s_t weak_hess_tmp251 = weak_hess_tmp153*weak_hess_tmp237 + weak_hess_tmp154*weak_hess_tmp242 - weak_hess_tmp229 + weak_hess_tmp43*(s_t(4)*gu2*gu7 - s_t(2)*weak_hess_tmp94);
        const s_t weak_hess_tmp252 = weak_hess_tmp158*weak_hess_tmp233 + weak_hess_tmp231;
        const s_t weak_hess_tmp253 = weak_hess_tmp117*weak_hess_tmp235 + weak_hess_tmp118*weak_hess_tmp240 + weak_hess_tmp180;
        const s_t weak_hess_tmp254 = weak_hess_tmp123*weak_hess_tmp237 + weak_hess_tmp126*weak_hess_tmp242 + weak_hess_tmp185 + weak_hess_tmp43*(-s_t(2)*weak_hess_tmp162 + s_t(4)*weak_hess_tmp6);
        const s_t weak_hess_tmp255 = weak_hess_tmp113*weak_hess_tmp233 - weak_hess_tmp188;
        const s_t weak_hess_tmp256 = weak_hess_tmp108*weak_hess_tmp235 + weak_hess_tmp164*weak_hess_tmp240 + weak_hess_tmp215;
        const s_t weak_hess_tmp257 = weak_hess_tmp169*weak_hess_tmp237 + weak_hess_tmp170*weak_hess_tmp242 + weak_hess_tmp217 + weak_hess_tmp43*(s_t(4)*gu2*weak_hess_tmp1 - s_t(2)*weak_hess_tmp111);
        const s_t weak_hess_tmp258 = weak_hess_tmp175*weak_hess_tmp233 - weak_hess_tmp219;
        const s_t weak_hess_tmp259 = weak_hess_tmp9*weak_hess_tmp96;
        const s_t weak_hess_tmp260 = weak_hess_tmp13*weak_hess_tmp259;
        const s_t weak_hess_tmp261 = weak_hess_tmp101*weak_hess_tmp17;
        const s_t weak_hess_tmp262 = weak_hess_tmp101*weak_hess_tmp35;
        const s_t weak_hess_tmp263 = s_t(2)*weak_hess_tmp30;
        const s_t weak_hess_tmp264 = weak_hess_tmp263 + weak_hess_tmp38;
        const s_t weak_hess_tmp265 = s_t(2)*weak_hess_tmp26;
        const s_t weak_hess_tmp266 = weak_hess_tmp265 + weak_hess_tmp40;
        const s_t weak_hess_tmp267 = weak_hess_tmp106*weak_hess_tmp45;
        const s_t weak_hess_tmp268 = weak_hess_tmp106*weak_hess_tmp64;
        const s_t weak_hess_tmp269 = weak_hess_tmp143*weak_hess_tmp259;
        const s_t weak_hess_tmp270 = weak_hess_tmp100*weak_hess_tmp17;
        const s_t weak_hess_tmp271 = weak_hess_tmp107*weak_hess_tmp261 + weak_hess_tmp130*weak_hess_tmp270;
        const s_t weak_hess_tmp272 = s_t(2)*weak_hess_tmp52;
        const s_t weak_hess_tmp273 = weak_hess_tmp109*weak_hess_tmp45;
        const s_t weak_hess_tmp274 = weak_hess_tmp137*weak_hess_tmp273 + weak_hess_tmp138*weak_hess_tmp267 + weak_hess_tmp43*(-weak_hess_tmp272 - weak_hess_tmp90);
        const s_t weak_hess_tmp275 = weak_hess_tmp113*weak_hess_tmp259;
        const s_t weak_hess_tmp276 = weak_hess_tmp117*weak_hess_tmp261 + weak_hess_tmp118*weak_hess_tmp270;
        const s_t weak_hess_tmp277 = weak_hess_tmp107*weak_hess_tmp2;
        const s_t weak_hess_tmp278 = gu7*weak_hess_tmp108;
        const s_t weak_hess_tmp279 = weak_hess_tmp123*weak_hess_tmp273 + weak_hess_tmp126*weak_hess_tmp267 + weak_hess_tmp43*(-weak_hess_tmp277 - weak_hess_tmp278);
        const s_t weak_hess_tmp280 = weak_hess_tmp175*weak_hess_tmp259;
        const s_t weak_hess_tmp281 = weak_hess_tmp108*weak_hess_tmp261 + weak_hess_tmp164*weak_hess_tmp270;
        const s_t weak_hess_tmp282 = s_t(2)*weak_hess_tmp47;
        const s_t weak_hess_tmp283 = weak_hess_tmp169*weak_hess_tmp273 + weak_hess_tmp170*weak_hess_tmp267 + weak_hess_tmp43*(-weak_hess_tmp282 - weak_hess_tmp75);
        const s_t weak_hess_tmp284 = gu2*weak_hess_tmp132;
        const s_t weak_hess_tmp285 = weak_hess_tmp124*weak_hess_tmp261 + weak_hess_tmp149*weak_hess_tmp270 - weak_hess_tmp284;
        const s_t weak_hess_tmp286 = gu2*weak_hess_tmp139;
        const s_t weak_hess_tmp287 = weak_hess_tmp153*weak_hess_tmp273 + weak_hess_tmp154*weak_hess_tmp267 - weak_hess_tmp286 + weak_hess_tmp43*(s_t(4)*weak_hess_tmp12 - s_t(2)*weak_hess_tmp81);
        const s_t weak_hess_tmp288 = gu2*weak_hess_tmp145;
        const s_t weak_hess_tmp289 = weak_hess_tmp158*weak_hess_tmp259 + weak_hess_tmp288;
        const s_t weak_hess_tmp290 = gu1*weak_hess_tmp132;
        const s_t weak_hess_tmp291 = weak_hess_tmp125*weak_hess_tmp261 + weak_hess_tmp179*weak_hess_tmp270 + weak_hess_tmp290;
        const s_t weak_hess_tmp292 = gu1*weak_hess_tmp139;
        const s_t weak_hess_tmp293 = weak_hess_tmp183*weak_hess_tmp273 + weak_hess_tmp184*weak_hess_tmp267 + weak_hess_tmp292 + weak_hess_tmp43*(-s_t(2)*weak_hess_tmp11 + s_t(4)*weak_hess_tmp66);
        const s_t weak_hess_tmp294 = gu1*weak_hess_tmp145;
        const s_t weak_hess_tmp295 = weak_hess_tmp189*weak_hess_tmp259 - weak_hess_tmp294;
        const s_t weak_hess_tmp296 = weak_hess_tmp175*weak_hess_tmp9;
        const s_t weak_hess_tmp297 = weak_hess_tmp13*weak_hess_tmp296;
        const s_t weak_hess_tmp298 = weak_hess_tmp164*weak_hess_tmp17;
        const s_t weak_hess_tmp299 = weak_hess_tmp164*weak_hess_tmp35;
        const s_t weak_hess_tmp300 = s_t(2)*weak_hess_tmp24;
        const s_t weak_hess_tmp301 = weak_hess_tmp196 + weak_hess_tmp300;
        const s_t weak_hess_tmp302 = weak_hess_tmp170*weak_hess_tmp45;
        const s_t weak_hess_tmp303 = weak_hess_tmp169*weak_hess_tmp64;
        const s_t weak_hess_tmp304 = weak_hess_tmp143*weak_hess_tmp296;
        const s_t weak_hess_tmp305 = weak_hess_tmp108*weak_hess_tmp17;
        const s_t weak_hess_tmp306 = weak_hess_tmp107*weak_hess_tmp298 + weak_hess_tmp130*weak_hess_tmp305;
        const s_t weak_hess_tmp307 = s_t(2)*weak_hess_tmp59;
        const s_t weak_hess_tmp308 = weak_hess_tmp169*weak_hess_tmp45;
        const s_t weak_hess_tmp309 = weak_hess_tmp137*weak_hess_tmp302 + weak_hess_tmp138*weak_hess_tmp308 + weak_hess_tmp43*(-weak_hess_tmp204 - weak_hess_tmp307);
        const s_t weak_hess_tmp310 = weak_hess_tmp158*weak_hess_tmp296;
        const s_t weak_hess_tmp311 = weak_hess_tmp124*weak_hess_tmp298 + weak_hess_tmp149*weak_hess_tmp305;
        const s_t weak_hess_tmp312 = gu6*weak_hess_tmp100;
        const s_t weak_hess_tmp313 = weak_hess_tmp153*weak_hess_tmp302 + weak_hess_tmp154*weak_hess_tmp308 + weak_hess_tmp43*(-weak_hess_tmp277 - weak_hess_tmp312);
        const s_t weak_hess_tmp314 = weak_hess_tmp117*weak_hess_tmp298 + weak_hess_tmp118*weak_hess_tmp305 + weak_hess_tmp284;
        const s_t weak_hess_tmp315 = weak_hess_tmp123*weak_hess_tmp302 + weak_hess_tmp126*weak_hess_tmp308 + weak_hess_tmp286 + weak_hess_tmp43*(s_t(4)*gu6*weak_hess_tmp1 - s_t(2)*weak_hess_tmp12);
        const s_t weak_hess_tmp316 = weak_hess_tmp113*weak_hess_tmp296 - weak_hess_tmp288;
        const s_t weak_hess_tmp317 = weak_hess_tmp132*weak_hess_tmp7;
        const s_t weak_hess_tmp318 = weak_hess_tmp125*weak_hess_tmp298 + weak_hess_tmp179*weak_hess_tmp305 - weak_hess_tmp317;
        const s_t weak_hess_tmp319 = weak_hess_tmp139*weak_hess_tmp7;
        const s_t weak_hess_tmp320 = weak_hess_tmp183*weak_hess_tmp302 + weak_hess_tmp184*weak_hess_tmp308 - weak_hess_tmp319 + weak_hess_tmp43*(-s_t(2)*weak_hess_tmp0 + s_t(4)*weak_hess_tmp1*weak_hess_tmp2);
        const s_t weak_hess_tmp321 = weak_hess_tmp145*weak_hess_tmp7;
        const s_t weak_hess_tmp322 = weak_hess_tmp189*weak_hess_tmp296 + weak_hess_tmp321;
        const s_t weak_hess_tmp323 = weak_hess_tmp143*weak_hess_tmp9;
        const s_t weak_hess_tmp324 = weak_hess_tmp13*weak_hess_tmp323;
        const s_t weak_hess_tmp325 = weak_hess_tmp130*weak_hess_tmp17;
        const s_t weak_hess_tmp326 = weak_hess_tmp130*weak_hess_tmp35;
        const s_t weak_hess_tmp327 = weak_hess_tmp138*weak_hess_tmp45;
        const s_t weak_hess_tmp328 = weak_hess_tmp137*weak_hess_tmp64;
        const s_t weak_hess_tmp329 = weak_hess_tmp189*weak_hess_tmp323;
        const s_t weak_hess_tmp330 = weak_hess_tmp107*weak_hess_tmp17;
        const s_t weak_hess_tmp331 = weak_hess_tmp125*weak_hess_tmp325 + weak_hess_tmp179*weak_hess_tmp330;
        const s_t weak_hess_tmp332 = weak_hess_tmp137*weak_hess_tmp45;
        const s_t weak_hess_tmp333 = weak_hess_tmp183*weak_hess_tmp327 + weak_hess_tmp184*weak_hess_tmp332 + weak_hess_tmp43*(-weak_hess_tmp278 - weak_hess_tmp312);
        const s_t weak_hess_tmp334 = weak_hess_tmp117*weak_hess_tmp325 + weak_hess_tmp118*weak_hess_tmp330 - weak_hess_tmp290;
        const s_t weak_hess_tmp335 = weak_hess_tmp123*weak_hess_tmp327 + weak_hess_tmp126*weak_hess_tmp332 - weak_hess_tmp292 + weak_hess_tmp43*(s_t(4)*gu5*gu6 - s_t(2)*weak_hess_tmp66);
        const s_t weak_hess_tmp336 = weak_hess_tmp113*weak_hess_tmp323 + weak_hess_tmp294;
        const s_t weak_hess_tmp337 = weak_hess_tmp124*weak_hess_tmp325 + weak_hess_tmp149*weak_hess_tmp330 + weak_hess_tmp317;
        const s_t weak_hess_tmp338 = weak_hess_tmp153*weak_hess_tmp327 + weak_hess_tmp154*weak_hess_tmp332 + weak_hess_tmp319 + weak_hess_tmp43*(s_t(4)*weak_hess_tmp0 - s_t(2)*weak_hess_tmp18);
        const s_t weak_hess_tmp339 = weak_hess_tmp158*weak_hess_tmp323 - weak_hess_tmp321;
        const s_t weak_hess_tmp340 = weak_hess_tmp113*weak_hess_tmp9;
        const s_t weak_hess_tmp341 = weak_hess_tmp13*weak_hess_tmp340;
        const s_t weak_hess_tmp342 = weak_hess_tmp118*weak_hess_tmp17;
        const s_t weak_hess_tmp343 = weak_hess_tmp118*weak_hess_tmp35;
        const s_t weak_hess_tmp344 = weak_hess_tmp263 + weak_hess_tmp37;
        const s_t weak_hess_tmp345 = weak_hess_tmp265 + weak_hess_tmp41;
        const s_t weak_hess_tmp346 = weak_hess_tmp123*weak_hess_tmp45;
        const s_t weak_hess_tmp347 = weak_hess_tmp123*weak_hess_tmp64;
        const s_t weak_hess_tmp348 = weak_hess_tmp158*weak_hess_tmp340;
        const s_t weak_hess_tmp349 = weak_hess_tmp117*weak_hess_tmp17;
        const s_t weak_hess_tmp350 = weak_hess_tmp124*weak_hess_tmp342 + weak_hess_tmp149*weak_hess_tmp349;
        const s_t weak_hess_tmp351 = weak_hess_tmp126*weak_hess_tmp45;
        const s_t weak_hess_tmp352 = weak_hess_tmp153*weak_hess_tmp351 + weak_hess_tmp154*weak_hess_tmp346 + weak_hess_tmp43*(-weak_hess_tmp282 - weak_hess_tmp76);
        const s_t weak_hess_tmp353 = weak_hess_tmp189*weak_hess_tmp340;
        const s_t weak_hess_tmp354 = weak_hess_tmp125*weak_hess_tmp342 + weak_hess_tmp179*weak_hess_tmp349;
        const s_t weak_hess_tmp355 = weak_hess_tmp183*weak_hess_tmp351 + weak_hess_tmp184*weak_hess_tmp346 + weak_hess_tmp43*(-weak_hess_tmp272 - weak_hess_tmp89);
        const s_t weak_hess_tmp356 = weak_hess_tmp158*weak_hess_tmp9;
        const s_t weak_hess_tmp357 = weak_hess_tmp13*weak_hess_tmp356;
        const s_t weak_hess_tmp358 = weak_hess_tmp149*weak_hess_tmp17;
        const s_t weak_hess_tmp359 = weak_hess_tmp149*weak_hess_tmp35;
        const s_t weak_hess_tmp360 = weak_hess_tmp195 + weak_hess_tmp300;
        const s_t weak_hess_tmp361 = weak_hess_tmp153*weak_hess_tmp45;
        const s_t weak_hess_tmp362 = weak_hess_tmp153*weak_hess_tmp64;
        const s_t weak_hess_tmp363 = weak_hess_tmp189*weak_hess_tmp356;
        const s_t weak_hess_tmp364 = weak_hess_tmp124*weak_hess_tmp17*weak_hess_tmp179 + weak_hess_tmp125*weak_hess_tmp358;
        const s_t weak_hess_tmp365 = weak_hess_tmp183*weak_hess_tmp45;
        const s_t weak_hess_tmp366 = weak_hess_tmp154*weak_hess_tmp365 + weak_hess_tmp184*weak_hess_tmp361 + weak_hess_tmp43*(-weak_hess_tmp203 - weak_hess_tmp307);
        const s_t weak_hess_tmp367 = weak_hess_tmp13*weak_hess_tmp189*weak_hess_tmp9;
        const s_t weak_hess_tmp368 = weak_hess_tmp179*weak_hess_tmp35;
        const s_t weak_hess_tmp369 = weak_hess_tmp183*weak_hess_tmp64;
        material[0] = trial_grad[0]*(c1*(weak_hess_tmp16 + s_t(4)*weak_hess_tmp20*weak_hess_tmp7 + weak_hess_tmp21*weak_hess_tmp36) + c2*(weak_hess_tmp43*(weak_hess_tmp39 + weak_hess_tmp42) + s_t(2)*weak_hess_tmp44*weak_hess_tmp57 + weak_hess_tmp58*weak_hess_tmp65) + weak_hess_tmp15*weak_hess_tmp3 + pow_2(weak_hess_tmp4)*weak_hess_tmp9) + trial_grad[1]*(c1*(weak_hess_tmp36*weak_hess_tmp70 + weak_hess_tmp73) + c2*(weak_hess_tmp65*weak_hess_tmp74 + weak_hess_tmp80) + weak_hess_tmp15*weak_hess_tmp67 + weak_hess_tmp69) + trial_grad[2]*(c1*(weak_hess_tmp36*weak_hess_tmp85 + weak_hess_tmp87) + c2*(weak_hess_tmp65*weak_hess_tmp88 + weak_hess_tmp93) + weak_hess_tmp15*weak_hess_tmp84 + weak_hess_tmp83) + trial_grad[3]*(c1*(weak_hess_tmp102 + weak_hess_tmp36*weak_hess_tmp99) + c2*(weak_hess_tmp103*weak_hess_tmp65 + weak_hess_tmp110) + weak_hess_tmp15*weak_hess_tmp95 + weak_hess_tmp97) + trial_grad[4]*(c1*(weak_hess_tmp163*weak_hess_tmp36 + weak_hess_tmp167) + c2*(weak_hess_tmp168*weak_hess_tmp65 + weak_hess_tmp172) + weak_hess_tmp15*weak_hess_tmp173 + weak_hess_tmp176) + trial_grad[5]*(c1*(weak_hess_tmp129*weak_hess_tmp36 + weak_hess_tmp134) + c2*(weak_hess_tmp135*weak_hess_tmp65 + weak_hess_tmp141) + weak_hess_tmp144*weak_hess_tmp15 + weak_hess_tmp147) + trial_grad[6]*(c1*(weak_hess_tmp116*weak_hess_tmp36 + weak_hess_tmp119) + c2*(weak_hess_tmp120*weak_hess_tmp65 + weak_hess_tmp127) + weak_hess_tmp114 + weak_hess_tmp115*weak_hess_tmp15) + trial_grad[7]*(c1*(weak_hess_tmp148*weak_hess_tmp36 + weak_hess_tmp151) + c2*(weak_hess_tmp152*weak_hess_tmp65 + weak_hess_tmp156) + weak_hess_tmp15*weak_hess_tmp159 + weak_hess_tmp161) + trial_grad[8]*(c1*(weak_hess_tmp178*weak_hess_tmp36 + weak_hess_tmp181) + c2*(weak_hess_tmp182*weak_hess_tmp65 + weak_hess_tmp186) + weak_hess_tmp15*weak_hess_tmp187 + weak_hess_tmp190);
        material[1] = trial_grad[0]*(c1*(weak_hess_tmp194*weak_hess_tmp21 + weak_hess_tmp73) + c2*(weak_hess_tmp199*weak_hess_tmp58 + weak_hess_tmp80) + weak_hess_tmp192*weak_hess_tmp3 + weak_hess_tmp69) + trial_grad[1]*(c1*(s_t(4)*gu1*weak_hess_tmp193 + weak_hess_tmp16 + weak_hess_tmp194*weak_hess_tmp70) + c2*(s_t(2)*weak_hess_tmp198*weak_hess_tmp77 + weak_hess_tmp199*weak_hess_tmp74 + weak_hess_tmp43*(weak_hess_tmp197 + weak_hess_tmp39)) + weak_hess_tmp192*weak_hess_tmp67 + pow_2(weak_hess_tmp68)*weak_hess_tmp9) + trial_grad[2]*(c1*(weak_hess_tmp194*weak_hess_tmp85 + weak_hess_tmp202) + c2*(weak_hess_tmp199*weak_hess_tmp88 + weak_hess_tmp206) + weak_hess_tmp192*weak_hess_tmp84 + weak_hess_tmp200) + trial_grad[3]*(c1*(weak_hess_tmp194*weak_hess_tmp99 + weak_hess_tmp224) + c2*(weak_hess_tmp103*weak_hess_tmp199 + weak_hess_tmp225) + weak_hess_tmp192*weak_hess_tmp95 + weak_hess_tmp226) + trial_grad[4]*(c1*(weak_hess_tmp163*weak_hess_tmp194 + weak_hess_tmp212) + c2*(weak_hess_tmp168*weak_hess_tmp199 + weak_hess_tmp214) + weak_hess_tmp173*weak_hess_tmp192 + weak_hess_tmp211) + trial_grad[5]*(c1*(weak_hess_tmp129*weak_hess_tmp194 + weak_hess_tmp216) + c2*(weak_hess_tmp135*weak_hess_tmp199 + weak_hess_tmp218) + weak_hess_tmp144*weak_hess_tmp192 + weak_hess_tmp220) + trial_grad[6]*(c1*(weak_hess_tmp116*weak_hess_tmp194 + weak_hess_tmp221) + c2*(weak_hess_tmp120*weak_hess_tmp199 + weak_hess_tmp222) + weak_hess_tmp115*weak_hess_tmp192 + weak_hess_tmp223) + trial_grad[7]*(c1*(weak_hess_tmp148*weak_hess_tmp194 + weak_hess_tmp208) + c2*(weak_hess_tmp152*weak_hess_tmp199 + weak_hess_tmp210) + weak_hess_tmp159*weak_hess_tmp192 + weak_hess_tmp207) + trial_grad[8]*(c1*(weak_hess_tmp178*weak_hess_tmp194 + weak_hess_tmp228) + c2*(weak_hess_tmp182*weak_hess_tmp199 + weak_hess_tmp230) + weak_hess_tmp187*weak_hess_tmp192 + weak_hess_tmp232);
        material[2] = trial_grad[0]*(c1*(weak_hess_tmp21*weak_hess_tmp236 + weak_hess_tmp87) + c2*(weak_hess_tmp238*weak_hess_tmp58 + weak_hess_tmp93) + weak_hess_tmp234*weak_hess_tmp3 + weak_hess_tmp83) + trial_grad[1]*(c1*(weak_hess_tmp202 + weak_hess_tmp236*weak_hess_tmp70) + c2*(weak_hess_tmp206 + weak_hess_tmp238*weak_hess_tmp74) + weak_hess_tmp200 + weak_hess_tmp234*weak_hess_tmp67) + trial_grad[2]*(c1*(s_t(4)*gu2*weak_hess_tmp235 + weak_hess_tmp16 + weak_hess_tmp236*weak_hess_tmp85) + c2*(s_t(2)*weak_hess_tmp237*weak_hess_tmp91 + weak_hess_tmp238*weak_hess_tmp88 + weak_hess_tmp43*(weak_hess_tmp197 + weak_hess_tmp42)) + weak_hess_tmp234*weak_hess_tmp84 + pow_2(weak_hess_tmp82)*weak_hess_tmp9) + trial_grad[3]*(c1*(weak_hess_tmp236*weak_hess_tmp99 + weak_hess_tmp247) + c2*(weak_hess_tmp103*weak_hess_tmp238 + weak_hess_tmp248) + weak_hess_tmp234*weak_hess_tmp95 + weak_hess_tmp249) + trial_grad[4]*(c1*(weak_hess_tmp163*weak_hess_tmp236 + weak_hess_tmp256) + c2*(weak_hess_tmp168*weak_hess_tmp238 + weak_hess_tmp257) + weak_hess_tmp173*weak_hess_tmp234 + weak_hess_tmp258) + trial_grad[5]*(c1*(weak_hess_tmp129*weak_hess_tmp236 + weak_hess_tmp241) + c2*(weak_hess_tmp135*weak_hess_tmp238 + weak_hess_tmp243) + weak_hess_tmp144*weak_hess_tmp234 + weak_hess_tmp239) + trial_grad[6]*(c1*(weak_hess_tmp116*weak_hess_tmp236 + weak_hess_tmp253) + c2*(weak_hess_tmp120*weak_hess_tmp238 + weak_hess_tmp254) + weak_hess_tmp115*weak_hess_tmp234 + weak_hess_tmp255) + trial_grad[7]*(c1*(weak_hess_tmp148*weak_hess_tmp236 + weak_hess_tmp250) + c2*(weak_hess_tmp152*weak_hess_tmp238 + weak_hess_tmp251) + weak_hess_tmp159*weak_hess_tmp234 + weak_hess_tmp252) + trial_grad[8]*(c1*(weak_hess_tmp178*weak_hess_tmp236 + weak_hess_tmp245) + c2*(weak_hess_tmp182*weak_hess_tmp238 + weak_hess_tmp246) + weak_hess_tmp187*weak_hess_tmp234 + weak_hess_tmp244);
        material[3] = trial_grad[0]*(c1*(weak_hess_tmp102 + weak_hess_tmp21*weak_hess_tmp262) + c2*(weak_hess_tmp110 + weak_hess_tmp268*weak_hess_tmp58) + weak_hess_tmp260*weak_hess_tmp3 + weak_hess_tmp97) + trial_grad[1]*(c1*(weak_hess_tmp224 + weak_hess_tmp262*weak_hess_tmp70) + c2*(weak_hess_tmp225 + weak_hess_tmp268*weak_hess_tmp74) + weak_hess_tmp226 + weak_hess_tmp260*weak_hess_tmp67) + trial_grad[2]*(c1*(weak_hess_tmp247 + weak_hess_tmp262*weak_hess_tmp85) + c2*(weak_hess_tmp248 + weak_hess_tmp268*weak_hess_tmp88) + weak_hess_tmp249 + weak_hess_tmp260*weak_hess_tmp84) + trial_grad[3]*(c1*(s_t(4)*gu3*weak_hess_tmp261 + weak_hess_tmp16 + weak_hess_tmp262*weak_hess_tmp99) + c2*(weak_hess_tmp103*weak_hess_tmp268 + s_t(2)*weak_hess_tmp109*weak_hess_tmp267 + weak_hess_tmp43*(weak_hess_tmp264 + weak_hess_tmp266)) + weak_hess_tmp260*weak_hess_tmp95 + weak_hess_tmp9*pow_2(weak_hess_tmp96)) + trial_grad[4]*(c1*(weak_hess_tmp163*weak_hess_tmp262 + weak_hess_tmp281) + c2*(weak_hess_tmp168*weak_hess_tmp268 + weak_hess_tmp283) + weak_hess_tmp173*weak_hess_tmp260 + weak_hess_tmp280) + trial_grad[5]*(c1*(weak_hess_tmp129*weak_hess_tmp262 + weak_hess_tmp271) + c2*(weak_hess_tmp135*weak_hess_tmp268 + weak_hess_tmp274) + weak_hess_tmp144*weak_hess_tmp260 + weak_hess_tmp269) + trial_grad[6]*(c1*(weak_hess_tmp116*weak_hess_tmp262 + weak_hess_tmp276) + c2*(weak_hess_tmp120*weak_hess_tmp268 + weak_hess_tmp279) + weak_hess_tmp115*weak_hess_tmp260 + weak_hess_tmp275) + trial_grad[7]*(c1*(weak_hess_tmp148*weak_hess_tmp262 + weak_hess_tmp285) + c2*(weak_hess_tmp152*weak_hess_tmp268 + weak_hess_tmp287) + weak_hess_tmp159*weak_hess_tmp260 + weak_hess_tmp289) + trial_grad[8]*(c1*(weak_hess_tmp178*weak_hess_tmp262 + weak_hess_tmp291) + c2*(weak_hess_tmp182*weak_hess_tmp268 + weak_hess_tmp293) + weak_hess_tmp187*weak_hess_tmp260 + weak_hess_tmp295);
        material[4] = trial_grad[0]*(c1*(weak_hess_tmp167 + weak_hess_tmp21*weak_hess_tmp299) + c2*(weak_hess_tmp172 + weak_hess_tmp303*weak_hess_tmp58) + weak_hess_tmp176 + weak_hess_tmp297*weak_hess_tmp3) + trial_grad[1]*(c1*(weak_hess_tmp212 + weak_hess_tmp299*weak_hess_tmp70) + c2*(weak_hess_tmp214 + weak_hess_tmp303*weak_hess_tmp74) + weak_hess_tmp211 + weak_hess_tmp297*weak_hess_tmp67) + trial_grad[2]*(c1*(weak_hess_tmp256 + weak_hess_tmp299*weak_hess_tmp85) + c2*(weak_hess_tmp257 + weak_hess_tmp303*weak_hess_tmp88) + weak_hess_tmp258 + weak_hess_tmp297*weak_hess_tmp84) + trial_grad[3]*(c1*(weak_hess_tmp281 + weak_hess_tmp299*weak_hess_tmp99) + c2*(weak_hess_tmp103*weak_hess_tmp303 + weak_hess_tmp283) + weak_hess_tmp280 + weak_hess_tmp297*weak_hess_tmp95) + trial_grad[4]*(c1*(s_t(4)*weak_hess_tmp1*weak_hess_tmp298 + weak_hess_tmp16 + weak_hess_tmp163*weak_hess_tmp299) + c2*(weak_hess_tmp168*weak_hess_tmp303 + s_t(2)*weak_hess_tmp169*weak_hess_tmp302 + weak_hess_tmp43*(weak_hess_tmp264 + weak_hess_tmp301)) + weak_hess_tmp173*weak_hess_tmp297 + pow_2(weak_hess_tmp175)*weak_hess_tmp9) + trial_grad[5]*(c1*(weak_hess_tmp129*weak_hess_tmp299 + weak_hess_tmp306) + c2*(weak_hess_tmp135*weak_hess_tmp303 + weak_hess_tmp309) + weak_hess_tmp144*weak_hess_tmp297 + weak_hess_tmp304) + trial_grad[6]*(c1*(weak_hess_tmp116*weak_hess_tmp299 + weak_hess_tmp314) + c2*(weak_hess_tmp120*weak_hess_tmp303 + weak_hess_tmp315) + weak_hess_tmp115*weak_hess_tmp297 + weak_hess_tmp316) + trial_grad[7]*(c1*(weak_hess_tmp148*weak_hess_tmp299 + weak_hess_tmp311) + c2*(weak_hess_tmp152*weak_hess_tmp303 + weak_hess_tmp313) + weak_hess_tmp159*weak_hess_tmp297 + weak_hess_tmp310) + trial_grad[8]*(c1*(weak_hess_tmp178*weak_hess_tmp299 + weak_hess_tmp318) + c2*(weak_hess_tmp182*weak_hess_tmp303 + weak_hess_tmp320) + weak_hess_tmp187*weak_hess_tmp297 + weak_hess_tmp322);
        material[5] = trial_grad[0]*(c1*(weak_hess_tmp134 + weak_hess_tmp21*weak_hess_tmp326) + c2*(weak_hess_tmp141 + weak_hess_tmp328*weak_hess_tmp58) + weak_hess_tmp147 + weak_hess_tmp3*weak_hess_tmp324) + trial_grad[1]*(c1*(weak_hess_tmp216 + weak_hess_tmp326*weak_hess_tmp70) + c2*(weak_hess_tmp218 + weak_hess_tmp328*weak_hess_tmp74) + weak_hess_tmp220 + weak_hess_tmp324*weak_hess_tmp67) + trial_grad[2]*(c1*(weak_hess_tmp241 + weak_hess_tmp326*weak_hess_tmp85) + c2*(weak_hess_tmp243 + weak_hess_tmp328*weak_hess_tmp88) + weak_hess_tmp239 + weak_hess_tmp324*weak_hess_tmp84) + trial_grad[3]*(c1*(weak_hess_tmp271 + weak_hess_tmp326*weak_hess_tmp99) + c2*(weak_hess_tmp103*weak_hess_tmp328 + weak_hess_tmp274) + weak_hess_tmp269 + weak_hess_tmp324*weak_hess_tmp95) + trial_grad[4]*(c1*(weak_hess_tmp163*weak_hess_tmp326 + weak_hess_tmp306) + c2*(weak_hess_tmp168*weak_hess_tmp328 + weak_hess_tmp309) + weak_hess_tmp173*weak_hess_tmp324 + weak_hess_tmp304) + trial_grad[5]*(c1*(s_t(4)*gu5*weak_hess_tmp325 + weak_hess_tmp129*weak_hess_tmp326 + weak_hess_tmp16) + c2*(weak_hess_tmp135*weak_hess_tmp328 + s_t(2)*weak_hess_tmp137*weak_hess_tmp327 + weak_hess_tmp43*(weak_hess_tmp266 + weak_hess_tmp301)) + pow_2(weak_hess_tmp143)*weak_hess_tmp9 + weak_hess_tmp144*weak_hess_tmp324) + trial_grad[6]*(c1*(weak_hess_tmp116*weak_hess_tmp326 + weak_hess_tmp334) + c2*(weak_hess_tmp120*weak_hess_tmp328 + weak_hess_tmp335) + weak_hess_tmp115*weak_hess_tmp324 + weak_hess_tmp336) + trial_grad[7]*(c1*(weak_hess_tmp148*weak_hess_tmp326 + weak_hess_tmp337) + c2*(weak_hess_tmp152*weak_hess_tmp328 + weak_hess_tmp338) + weak_hess_tmp159*weak_hess_tmp324 + weak_hess_tmp339) + trial_grad[8]*(c1*(weak_hess_tmp178*weak_hess_tmp326 + weak_hess_tmp331) + c2*(weak_hess_tmp182*weak_hess_tmp328 + weak_hess_tmp333) + weak_hess_tmp187*weak_hess_tmp324 + weak_hess_tmp329);
        material[6] = trial_grad[0]*(c1*(weak_hess_tmp119 + weak_hess_tmp21*weak_hess_tmp343) + c2*(weak_hess_tmp127 + weak_hess_tmp347*weak_hess_tmp58) + weak_hess_tmp114 + weak_hess_tmp3*weak_hess_tmp341) + trial_grad[1]*(c1*(weak_hess_tmp221 + weak_hess_tmp343*weak_hess_tmp70) + c2*(weak_hess_tmp222 + weak_hess_tmp347*weak_hess_tmp74) + weak_hess_tmp223 + weak_hess_tmp341*weak_hess_tmp67) + trial_grad[2]*(c1*(weak_hess_tmp253 + weak_hess_tmp343*weak_hess_tmp85) + c2*(weak_hess_tmp254 + weak_hess_tmp347*weak_hess_tmp88) + weak_hess_tmp255 + weak_hess_tmp341*weak_hess_tmp84) + trial_grad[3]*(c1*(weak_hess_tmp276 + weak_hess_tmp343*weak_hess_tmp99) + c2*(weak_hess_tmp103*weak_hess_tmp347 + weak_hess_tmp279) + weak_hess_tmp275 + weak_hess_tmp341*weak_hess_tmp95) + trial_grad[4]*(c1*(weak_hess_tmp163*weak_hess_tmp343 + weak_hess_tmp314) + c2*(weak_hess_tmp168*weak_hess_tmp347 + weak_hess_tmp315) + weak_hess_tmp173*weak_hess_tmp341 + weak_hess_tmp316) + trial_grad[5]*(c1*(weak_hess_tmp129*weak_hess_tmp343 + weak_hess_tmp334) + c2*(weak_hess_tmp135*weak_hess_tmp347 + weak_hess_tmp335) + weak_hess_tmp144*weak_hess_tmp341 + weak_hess_tmp336) + trial_grad[6]*(c1*(s_t(4)*gu6*weak_hess_tmp342 + weak_hess_tmp116*weak_hess_tmp343 + weak_hess_tmp16) + c2*(weak_hess_tmp120*weak_hess_tmp347 + s_t(2)*weak_hess_tmp126*weak_hess_tmp346 + weak_hess_tmp43*(weak_hess_tmp344 + weak_hess_tmp345)) + pow_2(weak_hess_tmp113)*weak_hess_tmp9 + weak_hess_tmp115*weak_hess_tmp341) + trial_grad[7]*(c1*(weak_hess_tmp148*weak_hess_tmp343 + weak_hess_tmp350) + c2*(weak_hess_tmp152*weak_hess_tmp347 + weak_hess_tmp352) + weak_hess_tmp159*weak_hess_tmp341 + weak_hess_tmp348) + trial_grad[8]*(c1*(weak_hess_tmp178*weak_hess_tmp343 + weak_hess_tmp354) + c2*(weak_hess_tmp182*weak_hess_tmp347 + weak_hess_tmp355) + weak_hess_tmp187*weak_hess_tmp341 + weak_hess_tmp353);
        material[7] = trial_grad[0]*(c1*(weak_hess_tmp151 + weak_hess_tmp21*weak_hess_tmp359) + c2*(weak_hess_tmp156 + weak_hess_tmp362*weak_hess_tmp58) + weak_hess_tmp161 + weak_hess_tmp3*weak_hess_tmp357) + trial_grad[1]*(c1*(weak_hess_tmp208 + weak_hess_tmp359*weak_hess_tmp70) + c2*(weak_hess_tmp210 + weak_hess_tmp362*weak_hess_tmp74) + weak_hess_tmp207 + weak_hess_tmp357*weak_hess_tmp67) + trial_grad[2]*(c1*(weak_hess_tmp250 + weak_hess_tmp359*weak_hess_tmp85) + c2*(weak_hess_tmp251 + weak_hess_tmp362*weak_hess_tmp88) + weak_hess_tmp252 + weak_hess_tmp357*weak_hess_tmp84) + trial_grad[3]*(c1*(weak_hess_tmp285 + weak_hess_tmp359*weak_hess_tmp99) + c2*(weak_hess_tmp103*weak_hess_tmp362 + weak_hess_tmp287) + weak_hess_tmp289 + weak_hess_tmp357*weak_hess_tmp95) + trial_grad[4]*(c1*(weak_hess_tmp163*weak_hess_tmp359 + weak_hess_tmp311) + c2*(weak_hess_tmp168*weak_hess_tmp362 + weak_hess_tmp313) + weak_hess_tmp173*weak_hess_tmp357 + weak_hess_tmp310) + trial_grad[5]*(c1*(weak_hess_tmp129*weak_hess_tmp359 + weak_hess_tmp337) + c2*(weak_hess_tmp135*weak_hess_tmp362 + weak_hess_tmp338) + weak_hess_tmp144*weak_hess_tmp357 + weak_hess_tmp339) + trial_grad[6]*(c1*(weak_hess_tmp116*weak_hess_tmp359 + weak_hess_tmp350) + c2*(weak_hess_tmp120*weak_hess_tmp362 + weak_hess_tmp352) + weak_hess_tmp115*weak_hess_tmp357 + weak_hess_tmp348) + trial_grad[7]*(c1*(s_t(4)*gu7*weak_hess_tmp358 + weak_hess_tmp148*weak_hess_tmp359 + weak_hess_tmp16) + c2*(weak_hess_tmp152*weak_hess_tmp362 + s_t(2)*weak_hess_tmp154*weak_hess_tmp361 + weak_hess_tmp43*(weak_hess_tmp344 + weak_hess_tmp360)) + pow_2(weak_hess_tmp158)*weak_hess_tmp9 + weak_hess_tmp159*weak_hess_tmp357) + trial_grad[8]*(c1*(weak_hess_tmp178*weak_hess_tmp359 + weak_hess_tmp364) + c2*(weak_hess_tmp182*weak_hess_tmp362 + weak_hess_tmp366) + weak_hess_tmp187*weak_hess_tmp357 + weak_hess_tmp363);
        material[8] = trial_grad[0]*(c1*(weak_hess_tmp181 + weak_hess_tmp21*weak_hess_tmp368) + c2*(weak_hess_tmp186 + weak_hess_tmp369*weak_hess_tmp58) + weak_hess_tmp190 + weak_hess_tmp3*weak_hess_tmp367) + trial_grad[1]*(c1*(weak_hess_tmp228 + weak_hess_tmp368*weak_hess_tmp70) + c2*(weak_hess_tmp230 + weak_hess_tmp369*weak_hess_tmp74) + weak_hess_tmp232 + weak_hess_tmp367*weak_hess_tmp67) + trial_grad[2]*(c1*(weak_hess_tmp245 + weak_hess_tmp368*weak_hess_tmp85) + c2*(weak_hess_tmp246 + weak_hess_tmp369*weak_hess_tmp88) + weak_hess_tmp244 + weak_hess_tmp367*weak_hess_tmp84) + trial_grad[3]*(c1*(weak_hess_tmp291 + weak_hess_tmp368*weak_hess_tmp99) + c2*(weak_hess_tmp103*weak_hess_tmp369 + weak_hess_tmp293) + weak_hess_tmp295 + weak_hess_tmp367*weak_hess_tmp95) + trial_grad[4]*(c1*(weak_hess_tmp163*weak_hess_tmp368 + weak_hess_tmp318) + c2*(weak_hess_tmp168*weak_hess_tmp369 + weak_hess_tmp320) + weak_hess_tmp173*weak_hess_tmp367 + weak_hess_tmp322) + trial_grad[5]*(c1*(weak_hess_tmp129*weak_hess_tmp368 + weak_hess_tmp331) + c2*(weak_hess_tmp135*weak_hess_tmp369 + weak_hess_tmp333) + weak_hess_tmp144*weak_hess_tmp367 + weak_hess_tmp329) + trial_grad[6]*(c1*(weak_hess_tmp116*weak_hess_tmp368 + weak_hess_tmp354) + c2*(weak_hess_tmp120*weak_hess_tmp369 + weak_hess_tmp355) + weak_hess_tmp115*weak_hess_tmp367 + weak_hess_tmp353) + trial_grad[7]*(c1*(weak_hess_tmp148*weak_hess_tmp368 + weak_hess_tmp364) + c2*(weak_hess_tmp152*weak_hess_tmp369 + weak_hess_tmp366) + weak_hess_tmp159*weak_hess_tmp367 + weak_hess_tmp363) + trial_grad[8]*(c1*(weak_hess_tmp16 + s_t(4)*weak_hess_tmp165*weak_hess_tmp179 + weak_hess_tmp178*weak_hess_tmp368) + c2*(weak_hess_tmp182*weak_hess_tmp369 + s_t(2)*weak_hess_tmp184*weak_hess_tmp365 + weak_hess_tmp43*(weak_hess_tmp345 + weak_hess_tmp360)) + weak_hess_tmp187*weak_hess_tmp367 + pow_2(weak_hess_tmp189)*weak_hess_tmp9);
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
