#ifndef MOONEY_RIVLIN_PRONY_D3_TENSOR_PRODUCT_HESSIAN_HPP
#define MOONEY_RIVLIN_PRONY_D3_TENSOR_PRODUCT_HESSIAN_HPP
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
static SFEM_INLINE void mooney_rivlin_prony_d3_tensor_product_direct_hessian_tensor_product_element_matrix(
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
    const s_t C01,
    const s_t C10,
    const s_t K,
    const s_t iso_relaxation_gamma,
    const s_t *const RSTR iso_history,
    const ptrdiff_t qp_stride,
    const s_t bu_data[NS * 3],
    s_t *const RSTR element_matrix
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(NS > 0, "NS must be positive");
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
    tensor_gradient_contiguous_scalar<s_t, NQ, NS, 3, NC>(
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
        const ptrdiff_t goff = q;
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
        const s_t *const RSTR iso_history_q = iso_history + q * qp_stride;
        const s_t iso_history_qp[6] = {iso_history_q[0], iso_history_q[1 * NQ * qp_stride], iso_history_q[2 * NQ * qp_stride], iso_history_q[3 * NQ * qp_stride], iso_history_q[4 * NQ * qp_stride], iso_history_q[5 * NQ * qp_stride]};
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
        const s_t weak_hess_tmp3 = -weak_hess_tmp0 + weak_hess_tmp1*weak_hess_tmp2;
        const s_t weak_hess_tmp4 = s_t(2)*weak_hess_tmp0;
        const s_t weak_hess_tmp5 = weak_hess_tmp1*weak_hess_tmp2;
        const s_t weak_hess_tmp6 = -s_t(2)*weak_hess_tmp5;
        const s_t weak_hess_tmp7 = ((s_t(1) / s_t(2)))*K;
        const s_t weak_hess_tmp8 = weak_hess_tmp7*(-weak_hess_tmp4 - weak_hess_tmp6);
        const s_t weak_hess_tmp9 = gu3*weak_hess_tmp2;
        const s_t weak_hess_tmp10 = gu6*weak_hess_tmp1;
        const s_t weak_hess_tmp11 = gu0 + s_t(1);
        const s_t weak_hess_tmp12 = gu1*gu5*gu6 - gu1*weak_hess_tmp9 + gu2*gu3*gu7 - gu2*weak_hess_tmp10 - weak_hess_tmp0*weak_hess_tmp11 + weak_hess_tmp1*weak_hess_tmp11*weak_hess_tmp2;
        const s_t weak_hess_tmp13 = s_t(2)/pow(weak_hess_tmp12, (s_t(2) / s_t(3)));
        const s_t weak_hess_tmp14 = pow(weak_hess_tmp12, (s_t(-5) / s_t(3)));
        const s_t weak_hess_tmp15 = ((s_t(2) / s_t(3)))*weak_hess_tmp0 - (s_t(2) / s_t(3))*weak_hess_tmp5;
        const s_t weak_hess_tmp16 = weak_hess_tmp14*weak_hess_tmp15;
        const s_t weak_hess_tmp17 = ((s_t(5) / s_t(3)))*weak_hess_tmp0 - (s_t(5) / s_t(3))*weak_hess_tmp5;
        const s_t weak_hess_tmp18 = pow_2(gu3);
        const s_t weak_hess_tmp19 = pow_2(gu6);
        const s_t weak_hess_tmp20 = pow_2(weak_hess_tmp11);
        const s_t weak_hess_tmp21 = weak_hess_tmp18 + weak_hess_tmp19 + weak_hess_tmp20;
        const s_t weak_hess_tmp22 = pow_2(gu1);
        const s_t weak_hess_tmp23 = pow_2(gu7);
        const s_t weak_hess_tmp24 = pow_2(weak_hess_tmp1);
        const s_t weak_hess_tmp25 = weak_hess_tmp22 + weak_hess_tmp23 + weak_hess_tmp24;
        const s_t weak_hess_tmp26 = pow_2(gu2);
        const s_t weak_hess_tmp27 = pow_2(gu5);
        const s_t weak_hess_tmp28 = pow_2(weak_hess_tmp2);
        const s_t weak_hess_tmp29 = weak_hess_tmp26 + weak_hess_tmp27 + weak_hess_tmp28;
        const s_t weak_hess_tmp30 = weak_hess_tmp21 + weak_hess_tmp25 + weak_hess_tmp29;
        const s_t weak_hess_tmp31 = weak_hess_tmp30/pow(weak_hess_tmp12, (s_t(8) / s_t(3)));
        const s_t weak_hess_tmp32 = weak_hess_tmp15*weak_hess_tmp31;
        const s_t weak_hess_tmp33 = s_t(2)*weak_hess_tmp27;
        const s_t weak_hess_tmp34 = s_t(2)*weak_hess_tmp28;
        const s_t weak_hess_tmp35 = weak_hess_tmp33 + weak_hess_tmp34;
        const s_t weak_hess_tmp36 = s_t(2)*weak_hess_tmp23;
        const s_t weak_hess_tmp37 = s_t(2)*weak_hess_tmp24;
        const s_t weak_hess_tmp38 = weak_hess_tmp36 + weak_hess_tmp37;
        const s_t weak_hess_tmp39 = pow(weak_hess_tmp12, (s_t(-4) / s_t(3)));
        const s_t weak_hess_tmp40 = ((s_t(4) / s_t(3)))*weak_hess_tmp0 - (s_t(4) / s_t(3))*weak_hess_tmp5;
        const s_t weak_hess_tmp41 = pow(weak_hess_tmp12, (s_t(-7) / s_t(3)));
        const s_t weak_hess_tmp42 = gu6*gu7;
        const s_t weak_hess_tmp43 = gu1*weak_hess_tmp11;
        const s_t weak_hess_tmp44 = gu3*weak_hess_tmp1;
        const s_t weak_hess_tmp45 = weak_hess_tmp42 + weak_hess_tmp43 + weak_hess_tmp44;
        const s_t weak_hess_tmp46 = s_t(2)*gu1;
        const s_t weak_hess_tmp47 = gu3*gu5;
        const s_t weak_hess_tmp48 = gu2*weak_hess_tmp11;
        const s_t weak_hess_tmp49 = gu6*weak_hess_tmp2;
        const s_t weak_hess_tmp50 = weak_hess_tmp47 + weak_hess_tmp48 + weak_hess_tmp49;
        const s_t weak_hess_tmp51 = s_t(2)*gu2;
        const s_t weak_hess_tmp52 = s_t(2)*weak_hess_tmp11;
        const s_t weak_hess_tmp53 = weak_hess_tmp41*(s_t(2)*weak_hess_tmp11*weak_hess_tmp30 - weak_hess_tmp21*weak_hess_tmp52 - weak_hess_tmp45*weak_hess_tmp46 - weak_hess_tmp50*weak_hess_tmp51);
        const s_t weak_hess_tmp54 = ((s_t(7) / s_t(3)))*weak_hess_tmp0 - (s_t(7) / s_t(3))*weak_hess_tmp5;
        const s_t weak_hess_tmp55 = gu1*gu2;
        const s_t weak_hess_tmp56 = gu5*weak_hess_tmp1;
        const s_t weak_hess_tmp57 = gu7*weak_hess_tmp2;
        const s_t weak_hess_tmp58 = weak_hess_tmp55 + weak_hess_tmp56 + weak_hess_tmp57;
        const s_t weak_hess_tmp59 = -(s_t(1) / s_t(2))*pow_2(weak_hess_tmp21) - (s_t(1) / s_t(2))*pow_2(weak_hess_tmp25) - (s_t(1) / s_t(2))*pow_2(weak_hess_tmp29) + ((s_t(1) / s_t(2)))*pow_2(weak_hess_tmp30) - pow_2(weak_hess_tmp45) - pow_2(weak_hess_tmp50) - pow_2(weak_hess_tmp58);
        const s_t weak_hess_tmp60 = weak_hess_tmp59/pow(weak_hess_tmp12, (s_t(10) / s_t(3)));
        const s_t weak_hess_tmp61 = weak_hess_tmp40*weak_hess_tmp60;
        const s_t weak_hess_tmp62 = gu1*weak_hess_tmp2;
        const s_t weak_hess_tmp63 = gu2*gu7 - weak_hess_tmp62;
        const s_t weak_hess_tmp64 = gu2*gu7;
        const s_t weak_hess_tmp65 = ((s_t(5) / s_t(3)))*weak_hess_tmp62 - (s_t(5) / s_t(3))*weak_hess_tmp64;
        const s_t weak_hess_tmp66 = s_t(2)*gu3;
        const s_t weak_hess_tmp67 = ((s_t(2) / s_t(3)))*weak_hess_tmp62 - (s_t(2) / s_t(3))*weak_hess_tmp64;
        const s_t weak_hess_tmp68 = weak_hess_tmp14*weak_hess_tmp52;
        const s_t weak_hess_tmp69 = weak_hess_tmp16*weak_hess_tmp66 + weak_hess_tmp67*weak_hess_tmp68;
        const s_t weak_hess_tmp70 = ((s_t(7) / s_t(3)))*weak_hess_tmp62 - (s_t(7) / s_t(3))*weak_hess_tmp64;
        const s_t weak_hess_tmp71 = gu5*weak_hess_tmp51;
        const s_t weak_hess_tmp72 = weak_hess_tmp1*weak_hess_tmp46;
        const s_t weak_hess_tmp73 = ((s_t(4) / s_t(3)))*weak_hess_tmp62 - (s_t(4) / s_t(3))*weak_hess_tmp64;
        const s_t weak_hess_tmp74 = s_t(2)*gu5;
        const s_t weak_hess_tmp75 = s_t(2)*weak_hess_tmp1;
        const s_t weak_hess_tmp76 = s_t(2)*gu3*weak_hess_tmp30 - weak_hess_tmp21*weak_hess_tmp66 - weak_hess_tmp45*weak_hess_tmp75 - weak_hess_tmp50*weak_hess_tmp74;
        const s_t weak_hess_tmp77 = weak_hess_tmp40*weak_hess_tmp41;
        const s_t weak_hess_tmp78 = weak_hess_tmp39*(-weak_hess_tmp71 - weak_hess_tmp72) + weak_hess_tmp53*weak_hess_tmp73 + weak_hess_tmp76*weak_hess_tmp77;
        const s_t weak_hess_tmp79 = gu1*gu5;
        const s_t weak_hess_tmp80 = gu2*weak_hess_tmp1;
        const s_t weak_hess_tmp81 = weak_hess_tmp79 - weak_hess_tmp80;
        const s_t weak_hess_tmp82 = ((s_t(5) / s_t(3)))*gu2*weak_hess_tmp1 - (s_t(5) / s_t(3))*weak_hess_tmp79;
        const s_t weak_hess_tmp83 = s_t(2)*gu6;
        const s_t weak_hess_tmp84 = ((s_t(2) / s_t(3)))*gu2*weak_hess_tmp1 - (s_t(2) / s_t(3))*weak_hess_tmp79;
        const s_t weak_hess_tmp85 = weak_hess_tmp16*weak_hess_tmp83 + weak_hess_tmp68*weak_hess_tmp84;
        const s_t weak_hess_tmp86 = ((s_t(7) / s_t(3)))*gu2*weak_hess_tmp1 - (s_t(7) / s_t(3))*weak_hess_tmp79;
        const s_t weak_hess_tmp87 = gu7*weak_hess_tmp46;
        const s_t weak_hess_tmp88 = weak_hess_tmp2*weak_hess_tmp51;
        const s_t weak_hess_tmp89 = ((s_t(4) / s_t(3)))*gu2*weak_hess_tmp1 - (s_t(4) / s_t(3))*weak_hess_tmp79;
        const s_t weak_hess_tmp90 = s_t(2)*gu7;
        const s_t weak_hess_tmp91 = s_t(2)*weak_hess_tmp2;
        const s_t weak_hess_tmp92 = s_t(2)*gu6*weak_hess_tmp30 - weak_hess_tmp21*weak_hess_tmp83 - weak_hess_tmp45*weak_hess_tmp90 - weak_hess_tmp50*weak_hess_tmp91;
        const s_t weak_hess_tmp93 = weak_hess_tmp39*(-weak_hess_tmp87 - weak_hess_tmp88) + weak_hess_tmp53*weak_hess_tmp89 + weak_hess_tmp77*weak_hess_tmp92;
        const s_t weak_hess_tmp94 = gu5*gu6 - weak_hess_tmp9;
        const s_t weak_hess_tmp95 = gu5*gu6;
        const s_t weak_hess_tmp96 = ((s_t(5) / s_t(3)))*weak_hess_tmp9 - (s_t(5) / s_t(3))*weak_hess_tmp95;
        const s_t weak_hess_tmp97 = ((s_t(2) / s_t(3)))*weak_hess_tmp9 - (s_t(2) / s_t(3))*weak_hess_tmp95;
        const s_t weak_hess_tmp98 = weak_hess_tmp16*weak_hess_tmp46 + weak_hess_tmp68*weak_hess_tmp97;
        const s_t weak_hess_tmp99 = ((s_t(7) / s_t(3)))*weak_hess_tmp9 - (s_t(7) / s_t(3))*weak_hess_tmp95;
        const s_t weak_hess_tmp100 = s_t(2)*weak_hess_tmp42;
        const s_t weak_hess_tmp101 = s_t(2)*weak_hess_tmp44;
        const s_t weak_hess_tmp102 = ((s_t(4) / s_t(3)))*weak_hess_tmp9 - (s_t(4) / s_t(3))*weak_hess_tmp95;
        const s_t weak_hess_tmp103 = s_t(2)*gu1*weak_hess_tmp30 - weak_hess_tmp25*weak_hess_tmp46 - weak_hess_tmp45*weak_hess_tmp52 - weak_hess_tmp51*weak_hess_tmp58;
        const s_t weak_hess_tmp104 = weak_hess_tmp102*weak_hess_tmp53 + weak_hess_tmp103*weak_hess_tmp77 + weak_hess_tmp39*(-weak_hess_tmp100 - weak_hess_tmp101);
        const s_t weak_hess_tmp105 = gu3*gu7;
        const s_t weak_hess_tmp106 = -weak_hess_tmp10 + weak_hess_tmp105;
        const s_t weak_hess_tmp107 = ((s_t(5) / s_t(3)))*gu6*weak_hess_tmp1 - (s_t(5) / s_t(3))*weak_hess_tmp105;
        const s_t weak_hess_tmp108 = ((s_t(2) / s_t(3)))*gu6*weak_hess_tmp1 - (s_t(2) / s_t(3))*weak_hess_tmp105;
        const s_t weak_hess_tmp109 = weak_hess_tmp108*weak_hess_tmp68 + weak_hess_tmp16*weak_hess_tmp51;
        const s_t weak_hess_tmp110 = ((s_t(7) / s_t(3)))*gu6*weak_hess_tmp1 - (s_t(7) / s_t(3))*weak_hess_tmp105;
        const s_t weak_hess_tmp111 = s_t(2)*weak_hess_tmp47;
        const s_t weak_hess_tmp112 = s_t(2)*weak_hess_tmp49;
        const s_t weak_hess_tmp113 = ((s_t(4) / s_t(3)))*gu6*weak_hess_tmp1 - (s_t(4) / s_t(3))*weak_hess_tmp105;
        const s_t weak_hess_tmp114 = s_t(2)*gu2*weak_hess_tmp30 - weak_hess_tmp29*weak_hess_tmp51 - weak_hess_tmp46*weak_hess_tmp58 - weak_hess_tmp50*weak_hess_tmp52;
        const s_t weak_hess_tmp115 = weak_hess_tmp113*weak_hess_tmp53 + weak_hess_tmp114*weak_hess_tmp77 + weak_hess_tmp39*(-weak_hess_tmp111 - weak_hess_tmp112);
        const s_t weak_hess_tmp116 = gu1*gu6;
        const s_t weak_hess_tmp117 = gu7*weak_hess_tmp11;
        const s_t weak_hess_tmp118 = weak_hess_tmp116 - weak_hess_tmp117;
        const s_t weak_hess_tmp119 = gu0*gu4;
        const s_t weak_hess_tmp120 = gu1*gu3;
        const s_t weak_hess_tmp121 = gu2*gu6;
        const s_t weak_hess_tmp122 = K*(gu0*gu8 - gu0*weak_hess_tmp0 + gu0 + gu1*weak_hess_tmp95 + gu2*weak_hess_tmp105 + gu4*gu8 - gu4*weak_hess_tmp121 + gu4 + gu8*weak_hess_tmp119 - gu8*weak_hess_tmp120 + gu8 - weak_hess_tmp0 + weak_hess_tmp119 - weak_hess_tmp120 - weak_hess_tmp121);
        const s_t weak_hess_tmp123 = gu7*weak_hess_tmp122;
        const s_t weak_hess_tmp124 = -weak_hess_tmp123;
        const s_t weak_hess_tmp125 = ((s_t(5) / s_t(3)))*gu7*weak_hess_tmp11 - (s_t(5) / s_t(3))*weak_hess_tmp116;
        const s_t weak_hess_tmp126 = ((s_t(2) / s_t(3)))*gu7*weak_hess_tmp11 - (s_t(2) / s_t(3))*weak_hess_tmp116;
        const s_t weak_hess_tmp127 = ((s_t(2) / s_t(3)))*weak_hess_tmp30;
        const s_t weak_hess_tmp128 = weak_hess_tmp127*weak_hess_tmp14;
        const s_t weak_hess_tmp129 = gu7*weak_hess_tmp128;
        const s_t weak_hess_tmp130 = weak_hess_tmp126*weak_hess_tmp68 + weak_hess_tmp129 + weak_hess_tmp16*weak_hess_tmp74;
        const s_t weak_hess_tmp131 = ((s_t(7) / s_t(3)))*gu7*weak_hess_tmp11 - (s_t(7) / s_t(3))*weak_hess_tmp116;
        const s_t weak_hess_tmp132 = gu2*gu3;
        const s_t weak_hess_tmp133 = s_t(2)*weak_hess_tmp132;
        const s_t weak_hess_tmp134 = ((s_t(4) / s_t(3)))*gu7*weak_hess_tmp11 - (s_t(4) / s_t(3))*weak_hess_tmp116;
        const s_t weak_hess_tmp135 = s_t(2)*gu5*weak_hess_tmp30 - weak_hess_tmp29*weak_hess_tmp74 - weak_hess_tmp50*weak_hess_tmp66 - weak_hess_tmp58*weak_hess_tmp75;
        const s_t weak_hess_tmp136 = ((s_t(4) / s_t(3)))*weak_hess_tmp41*weak_hess_tmp59;
        const s_t weak_hess_tmp137 = gu7*weak_hess_tmp136;
        const s_t weak_hess_tmp138 = weak_hess_tmp134*weak_hess_tmp53 + weak_hess_tmp135*weak_hess_tmp77 + weak_hess_tmp137 + weak_hess_tmp39*(s_t(4)*gu5*weak_hess_tmp11 - weak_hess_tmp133);
        const s_t weak_hess_tmp139 = gu5*weak_hess_tmp11;
        const s_t weak_hess_tmp140 = weak_hess_tmp132 - weak_hess_tmp139;
        const s_t weak_hess_tmp141 = gu5*weak_hess_tmp122;
        const s_t weak_hess_tmp142 = -weak_hess_tmp141;
        const s_t weak_hess_tmp143 = ((s_t(5) / s_t(3)))*gu5*weak_hess_tmp11 - (s_t(5) / s_t(3))*weak_hess_tmp132;
        const s_t weak_hess_tmp144 = ((s_t(2) / s_t(3)))*gu5*weak_hess_tmp11 - (s_t(2) / s_t(3))*weak_hess_tmp132;
        const s_t weak_hess_tmp145 = gu5*weak_hess_tmp128;
        const s_t weak_hess_tmp146 = weak_hess_tmp144*weak_hess_tmp68 + weak_hess_tmp145 + weak_hess_tmp16*weak_hess_tmp90;
        const s_t weak_hess_tmp147 = ((s_t(7) / s_t(3)))*gu5*weak_hess_tmp11 - (s_t(7) / s_t(3))*weak_hess_tmp132;
        const s_t weak_hess_tmp148 = s_t(2)*weak_hess_tmp116;
        const s_t weak_hess_tmp149 = ((s_t(4) / s_t(3)))*gu5*weak_hess_tmp11 - (s_t(4) / s_t(3))*weak_hess_tmp132;
        const s_t weak_hess_tmp150 = s_t(2)*gu7*weak_hess_tmp30 - weak_hess_tmp25*weak_hess_tmp90 - weak_hess_tmp45*weak_hess_tmp83 - weak_hess_tmp58*weak_hess_tmp91;
        const s_t weak_hess_tmp151 = gu5*weak_hess_tmp136;
        const s_t weak_hess_tmp152 = weak_hess_tmp149*weak_hess_tmp53 + weak_hess_tmp150*weak_hess_tmp77 + weak_hess_tmp151 + weak_hess_tmp39*(s_t(4)*gu7*weak_hess_tmp11 - weak_hess_tmp148);
        const s_t weak_hess_tmp153 = weak_hess_tmp11*weak_hess_tmp2 - weak_hess_tmp121;
        const s_t weak_hess_tmp154 = weak_hess_tmp122*weak_hess_tmp2;
        const s_t weak_hess_tmp155 = weak_hess_tmp11*weak_hess_tmp2;
        const s_t weak_hess_tmp156 = ((s_t(5) / s_t(3)))*weak_hess_tmp121 - (s_t(5) / s_t(3))*weak_hess_tmp155;
        const s_t weak_hess_tmp157 = ((s_t(2) / s_t(3)))*weak_hess_tmp121 - (s_t(2) / s_t(3))*weak_hess_tmp155;
        const s_t weak_hess_tmp158 = weak_hess_tmp14*weak_hess_tmp2;
        const s_t weak_hess_tmp159 = weak_hess_tmp127*weak_hess_tmp158;
        const s_t weak_hess_tmp160 = weak_hess_tmp157*weak_hess_tmp68 - weak_hess_tmp159 + weak_hess_tmp16*weak_hess_tmp75;
        const s_t weak_hess_tmp161 = ((s_t(7) / s_t(3)))*weak_hess_tmp121 - (s_t(7) / s_t(3))*weak_hess_tmp155;
        const s_t weak_hess_tmp162 = s_t(2)*weak_hess_tmp120;
        const s_t weak_hess_tmp163 = ((s_t(4) / s_t(3)))*weak_hess_tmp121 - (s_t(4) / s_t(3))*weak_hess_tmp155;
        const s_t weak_hess_tmp164 = s_t(2)*weak_hess_tmp1*weak_hess_tmp30 - weak_hess_tmp25*weak_hess_tmp75 - weak_hess_tmp45*weak_hess_tmp66 - weak_hess_tmp58*weak_hess_tmp74;
        const s_t weak_hess_tmp165 = weak_hess_tmp136*weak_hess_tmp2;
        const s_t weak_hess_tmp166 = weak_hess_tmp163*weak_hess_tmp53 + weak_hess_tmp164*weak_hess_tmp77 - weak_hess_tmp165 + weak_hess_tmp39*(s_t(4)*weak_hess_tmp1*weak_hess_tmp11 - weak_hess_tmp162);
        const s_t weak_hess_tmp167 = weak_hess_tmp1*weak_hess_tmp11 - weak_hess_tmp120;
        const s_t weak_hess_tmp168 = weak_hess_tmp1*weak_hess_tmp122;
        const s_t weak_hess_tmp169 = weak_hess_tmp1*weak_hess_tmp11;
        const s_t weak_hess_tmp170 = ((s_t(5) / s_t(3)))*weak_hess_tmp120 - (s_t(5) / s_t(3))*weak_hess_tmp169;
        const s_t weak_hess_tmp171 = ((s_t(2) / s_t(3)))*weak_hess_tmp120 - (s_t(2) / s_t(3))*weak_hess_tmp169;
        const s_t weak_hess_tmp172 = weak_hess_tmp1*weak_hess_tmp128;
        const s_t weak_hess_tmp173 = weak_hess_tmp16*weak_hess_tmp91 + weak_hess_tmp171*weak_hess_tmp68 - weak_hess_tmp172;
        const s_t weak_hess_tmp174 = ((s_t(7) / s_t(3)))*weak_hess_tmp120 - (s_t(7) / s_t(3))*weak_hess_tmp169;
        const s_t weak_hess_tmp175 = s_t(2)*weak_hess_tmp121;
        const s_t weak_hess_tmp176 = ((s_t(4) / s_t(3)))*weak_hess_tmp120 - (s_t(4) / s_t(3))*weak_hess_tmp169;
        const s_t weak_hess_tmp177 = s_t(2)*weak_hess_tmp2*weak_hess_tmp30 - weak_hess_tmp29*weak_hess_tmp91 - weak_hess_tmp50*weak_hess_tmp83 - weak_hess_tmp58*weak_hess_tmp90;
        const s_t weak_hess_tmp178 = weak_hess_tmp1*weak_hess_tmp136;
        const s_t weak_hess_tmp179 = weak_hess_tmp176*weak_hess_tmp53 + weak_hess_tmp177*weak_hess_tmp77 - weak_hess_tmp178 + weak_hess_tmp39*(s_t(4)*weak_hess_tmp11*weak_hess_tmp2 - weak_hess_tmp175);
        const s_t weak_hess_tmp180 = -s_t(2)*weak_hess_tmp95;
        const s_t weak_hess_tmp181 = s_t(2)*weak_hess_tmp9;
        const s_t weak_hess_tmp182 = weak_hess_tmp7*(-weak_hess_tmp180 - weak_hess_tmp181);
        const s_t weak_hess_tmp183 = weak_hess_tmp14*weak_hess_tmp97;
        const s_t weak_hess_tmp184 = weak_hess_tmp31*weak_hess_tmp97;
        const s_t weak_hess_tmp185 = s_t(2)*weak_hess_tmp18;
        const s_t weak_hess_tmp186 = s_t(2)*weak_hess_tmp19;
        const s_t weak_hess_tmp187 = weak_hess_tmp185 + weak_hess_tmp186;
        const s_t weak_hess_tmp188 = weak_hess_tmp103*weak_hess_tmp41;
        const s_t weak_hess_tmp189 = weak_hess_tmp102*weak_hess_tmp60;
        const s_t weak_hess_tmp190 = weak_hess_tmp14*weak_hess_tmp46;
        const s_t weak_hess_tmp191 = weak_hess_tmp144*weak_hess_tmp190 + weak_hess_tmp183*weak_hess_tmp90;
        const s_t weak_hess_tmp192 = gu6*weak_hess_tmp52;
        const s_t weak_hess_tmp193 = weak_hess_tmp102*weak_hess_tmp41;
        const s_t weak_hess_tmp194 = weak_hess_tmp149*weak_hess_tmp188 + weak_hess_tmp150*weak_hess_tmp193 + weak_hess_tmp39*(-weak_hess_tmp192 - weak_hess_tmp88);
        const s_t weak_hess_tmp195 = weak_hess_tmp108*weak_hess_tmp190 + weak_hess_tmp183*weak_hess_tmp51;
        const s_t weak_hess_tmp196 = s_t(2)*weak_hess_tmp56;
        const s_t weak_hess_tmp197 = s_t(2)*weak_hess_tmp57;
        const s_t weak_hess_tmp198 = weak_hess_tmp113*weak_hess_tmp188 + weak_hess_tmp114*weak_hess_tmp193 + weak_hess_tmp39*(-weak_hess_tmp196 - weak_hess_tmp197);
        const s_t weak_hess_tmp199 = weak_hess_tmp157*weak_hess_tmp190 + weak_hess_tmp183*weak_hess_tmp75;
        const s_t weak_hess_tmp200 = gu3*weak_hess_tmp52;
        const s_t weak_hess_tmp201 = weak_hess_tmp163*weak_hess_tmp188 + weak_hess_tmp164*weak_hess_tmp193 + weak_hess_tmp39*(-weak_hess_tmp200 - weak_hess_tmp71);
        const s_t weak_hess_tmp202 = gu6*weak_hess_tmp122;
        const s_t weak_hess_tmp203 = gu6*weak_hess_tmp128;
        const s_t weak_hess_tmp204 = weak_hess_tmp126*weak_hess_tmp190 + weak_hess_tmp183*weak_hess_tmp74 - weak_hess_tmp203;
        const s_t weak_hess_tmp205 = -s_t(2)*weak_hess_tmp80;
        const s_t weak_hess_tmp206 = gu6*weak_hess_tmp136;
        const s_t weak_hess_tmp207 = weak_hess_tmp134*weak_hess_tmp188 + weak_hess_tmp135*weak_hess_tmp193 - weak_hess_tmp206 + weak_hess_tmp39*(weak_hess_tmp205 + s_t(4)*weak_hess_tmp79);
        const s_t weak_hess_tmp208 = -weak_hess_tmp145 + weak_hess_tmp183*weak_hess_tmp83 + weak_hess_tmp190*weak_hess_tmp84;
        const s_t weak_hess_tmp209 = -s_t(2)*weak_hess_tmp117;
        const s_t weak_hess_tmp210 = -weak_hess_tmp151 + weak_hess_tmp188*weak_hess_tmp89 + weak_hess_tmp193*weak_hess_tmp92 + weak_hess_tmp39*(s_t(4)*weak_hess_tmp116 + weak_hess_tmp209);
        const s_t weak_hess_tmp211 = -weak_hess_tmp154;
        const s_t weak_hess_tmp212 = weak_hess_tmp159 + weak_hess_tmp183*weak_hess_tmp66 + weak_hess_tmp190*weak_hess_tmp67;
        const s_t weak_hess_tmp213 = -s_t(2)*weak_hess_tmp169;
        const s_t weak_hess_tmp214 = weak_hess_tmp165 + weak_hess_tmp188*weak_hess_tmp73 + weak_hess_tmp193*weak_hess_tmp76 + weak_hess_tmp39*(s_t(4)*weak_hess_tmp120 + weak_hess_tmp213);
        const s_t weak_hess_tmp215 = gu3*weak_hess_tmp122;
        const s_t weak_hess_tmp216 = -weak_hess_tmp215;
        const s_t weak_hess_tmp217 = gu3*weak_hess_tmp128;
        const s_t weak_hess_tmp218 = weak_hess_tmp171*weak_hess_tmp190 + weak_hess_tmp183*weak_hess_tmp91 + weak_hess_tmp217;
        const s_t weak_hess_tmp219 = -s_t(2)*weak_hess_tmp64;
        const s_t weak_hess_tmp220 = gu3*weak_hess_tmp136;
        const s_t weak_hess_tmp221 = weak_hess_tmp176*weak_hess_tmp188 + weak_hess_tmp177*weak_hess_tmp193 + weak_hess_tmp220 + weak_hess_tmp39*(weak_hess_tmp219 + s_t(4)*weak_hess_tmp62);
        const s_t weak_hess_tmp222 = s_t(2)*weak_hess_tmp105;
        const s_t weak_hess_tmp223 = -s_t(2)*weak_hess_tmp10;
        const s_t weak_hess_tmp224 = weak_hess_tmp7*(weak_hess_tmp222 + weak_hess_tmp223);
        const s_t weak_hess_tmp225 = weak_hess_tmp108*weak_hess_tmp14;
        const s_t weak_hess_tmp226 = weak_hess_tmp108*weak_hess_tmp31;
        const s_t weak_hess_tmp227 = weak_hess_tmp114*weak_hess_tmp41;
        const s_t weak_hess_tmp228 = weak_hess_tmp113*weak_hess_tmp60;
        const s_t weak_hess_tmp229 = weak_hess_tmp14*weak_hess_tmp51;
        const s_t weak_hess_tmp230 = weak_hess_tmp126*weak_hess_tmp229 + weak_hess_tmp225*weak_hess_tmp74;
        const s_t weak_hess_tmp231 = weak_hess_tmp113*weak_hess_tmp41;
        const s_t weak_hess_tmp232 = weak_hess_tmp134*weak_hess_tmp227 + weak_hess_tmp135*weak_hess_tmp231 + weak_hess_tmp39*(-weak_hess_tmp200 - weak_hess_tmp72);
        const s_t weak_hess_tmp233 = weak_hess_tmp171*weak_hess_tmp229 + weak_hess_tmp225*weak_hess_tmp91;
        const s_t weak_hess_tmp234 = weak_hess_tmp176*weak_hess_tmp227 + weak_hess_tmp177*weak_hess_tmp231 + weak_hess_tmp39*(-weak_hess_tmp192 - weak_hess_tmp87);
        const s_t weak_hess_tmp235 = -weak_hess_tmp129 + weak_hess_tmp225*weak_hess_tmp66 + weak_hess_tmp229*weak_hess_tmp67;
        const s_t weak_hess_tmp236 = -s_t(2)*weak_hess_tmp139;
        const s_t weak_hess_tmp237 = -weak_hess_tmp137 + weak_hess_tmp227*weak_hess_tmp73 + weak_hess_tmp231*weak_hess_tmp76 + weak_hess_tmp39*(s_t(4)*weak_hess_tmp132 + weak_hess_tmp236);
        const s_t weak_hess_tmp238 = weak_hess_tmp144*weak_hess_tmp229 - weak_hess_tmp217 + weak_hess_tmp225*weak_hess_tmp90;
        const s_t weak_hess_tmp239 = s_t(2)*weak_hess_tmp62;
        const s_t weak_hess_tmp240 = weak_hess_tmp149*weak_hess_tmp227 + weak_hess_tmp150*weak_hess_tmp231 - weak_hess_tmp220 + weak_hess_tmp39*(s_t(4)*gu2*gu7 - weak_hess_tmp239);
        const s_t weak_hess_tmp241 = -weak_hess_tmp168;
        const s_t weak_hess_tmp242 = weak_hess_tmp172 + weak_hess_tmp225*weak_hess_tmp83 + weak_hess_tmp229*weak_hess_tmp84;
        const s_t weak_hess_tmp243 = -s_t(2)*weak_hess_tmp155;
        const s_t weak_hess_tmp244 = weak_hess_tmp178 + weak_hess_tmp227*weak_hess_tmp89 + weak_hess_tmp231*weak_hess_tmp92 + weak_hess_tmp39*(s_t(4)*weak_hess_tmp121 + weak_hess_tmp243);
        const s_t weak_hess_tmp245 = -weak_hess_tmp202;
        const s_t weak_hess_tmp246 = weak_hess_tmp157*weak_hess_tmp229 + weak_hess_tmp203 + weak_hess_tmp225*weak_hess_tmp75;
        const s_t weak_hess_tmp247 = s_t(2)*weak_hess_tmp79;
        const s_t weak_hess_tmp248 = weak_hess_tmp163*weak_hess_tmp227 + weak_hess_tmp164*weak_hess_tmp231 + weak_hess_tmp206 + weak_hess_tmp39*(s_t(4)*gu2*weak_hess_tmp1 - weak_hess_tmp247);
        const s_t weak_hess_tmp249 = weak_hess_tmp7*(-weak_hess_tmp219 - weak_hess_tmp239);
        const s_t weak_hess_tmp250 = weak_hess_tmp14*weak_hess_tmp67;
        const s_t weak_hess_tmp251 = weak_hess_tmp31*weak_hess_tmp67;
        const s_t weak_hess_tmp252 = s_t(2)*weak_hess_tmp26;
        const s_t weak_hess_tmp253 = weak_hess_tmp252 + weak_hess_tmp34;
        const s_t weak_hess_tmp254 = s_t(2)*weak_hess_tmp22;
        const s_t weak_hess_tmp255 = weak_hess_tmp254 + weak_hess_tmp36;
        const s_t weak_hess_tmp256 = weak_hess_tmp41*weak_hess_tmp73;
        const s_t weak_hess_tmp257 = weak_hess_tmp60*weak_hess_tmp73;
        const s_t weak_hess_tmp258 = weak_hess_tmp14*weak_hess_tmp66;
        const s_t weak_hess_tmp259 = weak_hess_tmp250*weak_hess_tmp83 + weak_hess_tmp258*weak_hess_tmp84;
        const s_t weak_hess_tmp260 = weak_hess_tmp2*weak_hess_tmp74;
        const s_t weak_hess_tmp261 = gu7*weak_hess_tmp75;
        const s_t weak_hess_tmp262 = weak_hess_tmp41*weak_hess_tmp76;
        const s_t weak_hess_tmp263 = weak_hess_tmp256*weak_hess_tmp92 + weak_hess_tmp262*weak_hess_tmp89 + weak_hess_tmp39*(-weak_hess_tmp260 - weak_hess_tmp261);
        const s_t weak_hess_tmp264 = weak_hess_tmp126*weak_hess_tmp258 + weak_hess_tmp250*weak_hess_tmp74;
        const s_t weak_hess_tmp265 = s_t(2)*weak_hess_tmp48;
        const s_t weak_hess_tmp266 = weak_hess_tmp134*weak_hess_tmp262 + weak_hess_tmp135*weak_hess_tmp256 + weak_hess_tmp39*(-weak_hess_tmp112 - weak_hess_tmp265);
        const s_t weak_hess_tmp267 = weak_hess_tmp157*weak_hess_tmp258 + weak_hess_tmp250*weak_hess_tmp75;
        const s_t weak_hess_tmp268 = s_t(2)*weak_hess_tmp43;
        const s_t weak_hess_tmp269 = weak_hess_tmp163*weak_hess_tmp262 + weak_hess_tmp164*weak_hess_tmp256 + weak_hess_tmp39*(-weak_hess_tmp100 - weak_hess_tmp268);
        const s_t weak_hess_tmp270 = gu2*weak_hess_tmp122;
        const s_t weak_hess_tmp271 = gu2*weak_hess_tmp128;
        const s_t weak_hess_tmp272 = weak_hess_tmp144*weak_hess_tmp258 + weak_hess_tmp250*weak_hess_tmp90 - weak_hess_tmp271;
        const s_t weak_hess_tmp273 = gu2*weak_hess_tmp136;
        const s_t weak_hess_tmp274 = weak_hess_tmp149*weak_hess_tmp262 + weak_hess_tmp150*weak_hess_tmp256 - weak_hess_tmp273 + weak_hess_tmp39*(s_t(4)*weak_hess_tmp105 + weak_hess_tmp223);
        const s_t weak_hess_tmp275 = gu1*weak_hess_tmp122;
        const s_t weak_hess_tmp276 = -weak_hess_tmp275;
        const s_t weak_hess_tmp277 = gu1*weak_hess_tmp128;
        const s_t weak_hess_tmp278 = weak_hess_tmp171*weak_hess_tmp258 + weak_hess_tmp250*weak_hess_tmp91 + weak_hess_tmp277;
        const s_t weak_hess_tmp279 = gu1*weak_hess_tmp136;
        const s_t weak_hess_tmp280 = weak_hess_tmp176*weak_hess_tmp262 + weak_hess_tmp177*weak_hess_tmp256 + weak_hess_tmp279 + weak_hess_tmp39*(weak_hess_tmp180 + s_t(4)*weak_hess_tmp9);
        const s_t weak_hess_tmp281 = weak_hess_tmp7*(-weak_hess_tmp175 - weak_hess_tmp243);
        const s_t weak_hess_tmp282 = weak_hess_tmp14*weak_hess_tmp157;
        const s_t weak_hess_tmp283 = weak_hess_tmp157*weak_hess_tmp31;
        const s_t weak_hess_tmp284 = s_t(2)*weak_hess_tmp20;
        const s_t weak_hess_tmp285 = weak_hess_tmp186 + weak_hess_tmp284;
        const s_t weak_hess_tmp286 = weak_hess_tmp164*weak_hess_tmp41;
        const s_t weak_hess_tmp287 = weak_hess_tmp163*weak_hess_tmp60;
        const s_t weak_hess_tmp288 = weak_hess_tmp14*weak_hess_tmp75;
        const s_t weak_hess_tmp289 = weak_hess_tmp144*weak_hess_tmp288 + weak_hess_tmp282*weak_hess_tmp90;
        const s_t weak_hess_tmp290 = gu6*weak_hess_tmp66;
        const s_t weak_hess_tmp291 = weak_hess_tmp163*weak_hess_tmp41;
        const s_t weak_hess_tmp292 = weak_hess_tmp149*weak_hess_tmp286 + weak_hess_tmp150*weak_hess_tmp291 + weak_hess_tmp39*(-weak_hess_tmp260 - weak_hess_tmp290);
        const s_t weak_hess_tmp293 = weak_hess_tmp126*weak_hess_tmp288 + weak_hess_tmp282*weak_hess_tmp74;
        const s_t weak_hess_tmp294 = s_t(2)*weak_hess_tmp55;
        const s_t weak_hess_tmp295 = weak_hess_tmp134*weak_hess_tmp286 + weak_hess_tmp135*weak_hess_tmp291 + weak_hess_tmp39*(-weak_hess_tmp197 - weak_hess_tmp294);
        const s_t weak_hess_tmp296 = -weak_hess_tmp270;
        const s_t weak_hess_tmp297 = weak_hess_tmp271 + weak_hess_tmp282*weak_hess_tmp83 + weak_hess_tmp288*weak_hess_tmp84;
        const s_t weak_hess_tmp298 = weak_hess_tmp273 + weak_hess_tmp286*weak_hess_tmp89 + weak_hess_tmp291*weak_hess_tmp92 + weak_hess_tmp39*(s_t(4)*gu6*weak_hess_tmp1 - weak_hess_tmp222);
        const s_t weak_hess_tmp299 = weak_hess_tmp11*weak_hess_tmp122;
        const s_t weak_hess_tmp300 = weak_hess_tmp11*weak_hess_tmp128;
        const s_t weak_hess_tmp301 = weak_hess_tmp171*weak_hess_tmp288 + weak_hess_tmp282*weak_hess_tmp91 - weak_hess_tmp300;
        const s_t weak_hess_tmp302 = weak_hess_tmp11*weak_hess_tmp136;
        const s_t weak_hess_tmp303 = weak_hess_tmp176*weak_hess_tmp286 + weak_hess_tmp177*weak_hess_tmp291 - weak_hess_tmp302 + weak_hess_tmp39*(s_t(4)*weak_hess_tmp1*weak_hess_tmp2 - weak_hess_tmp4);
        const s_t weak_hess_tmp304 = weak_hess_tmp7*(weak_hess_tmp148 + weak_hess_tmp209);
        const s_t weak_hess_tmp305 = weak_hess_tmp126*weak_hess_tmp14;
        const s_t weak_hess_tmp306 = weak_hess_tmp126*weak_hess_tmp31;
        const s_t weak_hess_tmp307 = weak_hess_tmp135*weak_hess_tmp41;
        const s_t weak_hess_tmp308 = weak_hess_tmp134*weak_hess_tmp60;
        const s_t weak_hess_tmp309 = weak_hess_tmp14*weak_hess_tmp74;
        const s_t weak_hess_tmp310 = weak_hess_tmp171*weak_hess_tmp309 + weak_hess_tmp305*weak_hess_tmp91;
        const s_t weak_hess_tmp311 = weak_hess_tmp134*weak_hess_tmp41;
        const s_t weak_hess_tmp312 = weak_hess_tmp176*weak_hess_tmp307 + weak_hess_tmp177*weak_hess_tmp311 + weak_hess_tmp39*(-weak_hess_tmp261 - weak_hess_tmp290);
        const s_t weak_hess_tmp313 = -weak_hess_tmp277 + weak_hess_tmp305*weak_hess_tmp83 + weak_hess_tmp309*weak_hess_tmp84;
        const s_t weak_hess_tmp314 = -weak_hess_tmp279 + weak_hess_tmp307*weak_hess_tmp89 + weak_hess_tmp311*weak_hess_tmp92 + weak_hess_tmp39*(s_t(4)*gu5*gu6 - weak_hess_tmp181);
        const s_t weak_hess_tmp315 = -weak_hess_tmp299;
        const s_t weak_hess_tmp316 = weak_hess_tmp144*weak_hess_tmp309 + weak_hess_tmp300 + weak_hess_tmp305*weak_hess_tmp90;
        const s_t weak_hess_tmp317 = weak_hess_tmp149*weak_hess_tmp307 + weak_hess_tmp150*weak_hess_tmp311 + weak_hess_tmp302 + weak_hess_tmp39*(s_t(4)*weak_hess_tmp0 + weak_hess_tmp6);
        const s_t weak_hess_tmp318 = weak_hess_tmp7*(weak_hess_tmp205 + weak_hess_tmp247);
        const s_t weak_hess_tmp319 = weak_hess_tmp14*weak_hess_tmp84;
        const s_t weak_hess_tmp320 = weak_hess_tmp31*weak_hess_tmp84;
        const s_t weak_hess_tmp321 = weak_hess_tmp252 + weak_hess_tmp33;
        const s_t weak_hess_tmp322 = weak_hess_tmp254 + weak_hess_tmp37;
        const s_t weak_hess_tmp323 = weak_hess_tmp41*weak_hess_tmp89;
        const s_t weak_hess_tmp324 = weak_hess_tmp60*weak_hess_tmp89;
        const s_t weak_hess_tmp325 = weak_hess_tmp14*weak_hess_tmp83;
        const s_t weak_hess_tmp326 = weak_hess_tmp144*weak_hess_tmp325 + weak_hess_tmp319*weak_hess_tmp90;
        const s_t weak_hess_tmp327 = weak_hess_tmp41*weak_hess_tmp92;
        const s_t weak_hess_tmp328 = weak_hess_tmp149*weak_hess_tmp327 + weak_hess_tmp150*weak_hess_tmp323 + weak_hess_tmp39*(-weak_hess_tmp101 - weak_hess_tmp268);
        const s_t weak_hess_tmp329 = weak_hess_tmp171*weak_hess_tmp325 + weak_hess_tmp319*weak_hess_tmp91;
        const s_t weak_hess_tmp330 = weak_hess_tmp176*weak_hess_tmp327 + weak_hess_tmp177*weak_hess_tmp323 + weak_hess_tmp39*(-weak_hess_tmp111 - weak_hess_tmp265);
        const s_t weak_hess_tmp331 = weak_hess_tmp7*(weak_hess_tmp133 + weak_hess_tmp236);
        const s_t weak_hess_tmp332 = weak_hess_tmp14*weak_hess_tmp144;
        const s_t weak_hess_tmp333 = weak_hess_tmp144*weak_hess_tmp31;
        const s_t weak_hess_tmp334 = weak_hess_tmp185 + weak_hess_tmp284;
        const s_t weak_hess_tmp335 = weak_hess_tmp149*weak_hess_tmp41;
        const s_t weak_hess_tmp336 = weak_hess_tmp149*weak_hess_tmp60;
        const s_t weak_hess_tmp337 = weak_hess_tmp14*weak_hess_tmp171*weak_hess_tmp90 + weak_hess_tmp332*weak_hess_tmp91;
        const s_t weak_hess_tmp338 = weak_hess_tmp176*weak_hess_tmp41;
        const s_t weak_hess_tmp339 = weak_hess_tmp150*weak_hess_tmp338 + weak_hess_tmp177*weak_hess_tmp335 + weak_hess_tmp39*(-weak_hess_tmp196 - weak_hess_tmp294);
        const s_t weak_hess_tmp340 = weak_hess_tmp7*(-weak_hess_tmp162 - weak_hess_tmp213);
        const s_t weak_hess_tmp341 = weak_hess_tmp171*weak_hess_tmp31;
        const s_t weak_hess_tmp342 = weak_hess_tmp176*weak_hess_tmp60;
        material[0] = trial_grad[0]*(iso_history_qp[0] + iso_relaxation_gamma*(C01*(weak_hess_tmp39*(weak_hess_tmp35 + weak_hess_tmp38) + s_t(2)*weak_hess_tmp40*weak_hess_tmp53 + weak_hess_tmp54*weak_hess_tmp61) + C10*(s_t(4)*weak_hess_tmp11*weak_hess_tmp16 + weak_hess_tmp13 + weak_hess_tmp17*weak_hess_tmp32)) + weak_hess_tmp3*weak_hess_tmp8) + trial_grad[1]*(iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_hess_tmp104 + weak_hess_tmp61*weak_hess_tmp99) + C10*(weak_hess_tmp32*weak_hess_tmp96 + weak_hess_tmp98)) + weak_hess_tmp8*weak_hess_tmp94) + trial_grad[2]*(iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_hess_tmp110*weak_hess_tmp61 + weak_hess_tmp115) + C10*(weak_hess_tmp107*weak_hess_tmp32 + weak_hess_tmp109)) + weak_hess_tmp106*weak_hess_tmp8) + trial_grad[3]*(iso_relaxation_gamma*(C01*(weak_hess_tmp61*weak_hess_tmp70 + weak_hess_tmp78) + C10*(weak_hess_tmp32*weak_hess_tmp65 + weak_hess_tmp69)) + weak_hess_tmp63*weak_hess_tmp8) + trial_grad[4]*(iso_relaxation_gamma*(C01*(weak_hess_tmp161*weak_hess_tmp61 + weak_hess_tmp166) + C10*(weak_hess_tmp156*weak_hess_tmp32 + weak_hess_tmp160)) + weak_hess_tmp153*weak_hess_tmp8 + weak_hess_tmp154) + trial_grad[5]*(iso_relaxation_gamma*(C01*(weak_hess_tmp131*weak_hess_tmp61 + weak_hess_tmp138) + C10*(weak_hess_tmp125*weak_hess_tmp32 + weak_hess_tmp130)) + weak_hess_tmp118*weak_hess_tmp8 + weak_hess_tmp124) + trial_grad[6]*(iso_relaxation_gamma*(C01*(weak_hess_tmp61*weak_hess_tmp86 + weak_hess_tmp93) + C10*(weak_hess_tmp32*weak_hess_tmp82 + weak_hess_tmp85)) + weak_hess_tmp8*weak_hess_tmp81) + trial_grad[7]*(iso_relaxation_gamma*(C01*(weak_hess_tmp147*weak_hess_tmp61 + weak_hess_tmp152) + C10*(weak_hess_tmp143*weak_hess_tmp32 + weak_hess_tmp146)) + weak_hess_tmp140*weak_hess_tmp8 + weak_hess_tmp142) + trial_grad[8]*(iso_relaxation_gamma*(C01*(weak_hess_tmp174*weak_hess_tmp61 + weak_hess_tmp179) + C10*(weak_hess_tmp170*weak_hess_tmp32 + weak_hess_tmp173)) + weak_hess_tmp167*weak_hess_tmp8 + weak_hess_tmp168);
        material[1] = trial_grad[0]*(iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_hess_tmp104 + weak_hess_tmp189*weak_hess_tmp54) + C10*(weak_hess_tmp17*weak_hess_tmp184 + weak_hess_tmp98)) + weak_hess_tmp182*weak_hess_tmp3) + trial_grad[1]*(iso_history_qp[1] + iso_relaxation_gamma*(C01*(s_t(2)*weak_hess_tmp102*weak_hess_tmp188 + weak_hess_tmp189*weak_hess_tmp99 + weak_hess_tmp39*(weak_hess_tmp187 + weak_hess_tmp35)) + C10*(s_t(4)*gu1*weak_hess_tmp183 + weak_hess_tmp13 + weak_hess_tmp184*weak_hess_tmp96)) + weak_hess_tmp182*weak_hess_tmp94) + trial_grad[2]*(iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_hess_tmp110*weak_hess_tmp189 + weak_hess_tmp198) + C10*(weak_hess_tmp107*weak_hess_tmp184 + weak_hess_tmp195)) + weak_hess_tmp106*weak_hess_tmp182) + trial_grad[3]*(iso_relaxation_gamma*(C01*(weak_hess_tmp189*weak_hess_tmp70 + weak_hess_tmp214) + C10*(weak_hess_tmp184*weak_hess_tmp65 + weak_hess_tmp212)) + weak_hess_tmp182*weak_hess_tmp63 + weak_hess_tmp211) + trial_grad[4]*(iso_relaxation_gamma*(C01*(weak_hess_tmp161*weak_hess_tmp189 + weak_hess_tmp201) + C10*(weak_hess_tmp156*weak_hess_tmp184 + weak_hess_tmp199)) + weak_hess_tmp153*weak_hess_tmp182) + trial_grad[5]*(iso_relaxation_gamma*(C01*(weak_hess_tmp131*weak_hess_tmp189 + weak_hess_tmp207) + C10*(weak_hess_tmp125*weak_hess_tmp184 + weak_hess_tmp204)) + weak_hess_tmp118*weak_hess_tmp182 + weak_hess_tmp202) + trial_grad[6]*(iso_relaxation_gamma*(C01*(weak_hess_tmp189*weak_hess_tmp86 + weak_hess_tmp210) + C10*(weak_hess_tmp184*weak_hess_tmp82 + weak_hess_tmp208)) + weak_hess_tmp141 + weak_hess_tmp182*weak_hess_tmp81) + trial_grad[7]*(iso_relaxation_gamma*(C01*(weak_hess_tmp147*weak_hess_tmp189 + weak_hess_tmp194) + C10*(weak_hess_tmp143*weak_hess_tmp184 + weak_hess_tmp191)) + weak_hess_tmp140*weak_hess_tmp182) + trial_grad[8]*(iso_relaxation_gamma*(C01*(weak_hess_tmp174*weak_hess_tmp189 + weak_hess_tmp221) + C10*(weak_hess_tmp170*weak_hess_tmp184 + weak_hess_tmp218)) + weak_hess_tmp167*weak_hess_tmp182 + weak_hess_tmp216);
        material[2] = trial_grad[0]*(iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_hess_tmp115 + weak_hess_tmp228*weak_hess_tmp54) + C10*(weak_hess_tmp109 + weak_hess_tmp17*weak_hess_tmp226)) + weak_hess_tmp224*weak_hess_tmp3) + trial_grad[1]*(iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_hess_tmp198 + weak_hess_tmp228*weak_hess_tmp99) + C10*(weak_hess_tmp195 + weak_hess_tmp226*weak_hess_tmp96)) + weak_hess_tmp224*weak_hess_tmp94) + trial_grad[2]*(iso_history_qp[2] + iso_relaxation_gamma*(C01*(weak_hess_tmp110*weak_hess_tmp228 + s_t(2)*weak_hess_tmp113*weak_hess_tmp227 + weak_hess_tmp39*(weak_hess_tmp187 + weak_hess_tmp38)) + C10*(s_t(4)*gu2*weak_hess_tmp225 + weak_hess_tmp107*weak_hess_tmp226 + weak_hess_tmp13)) + weak_hess_tmp106*weak_hess_tmp224) + trial_grad[3]*(iso_relaxation_gamma*(C01*(weak_hess_tmp228*weak_hess_tmp70 + weak_hess_tmp237) + C10*(weak_hess_tmp226*weak_hess_tmp65 + weak_hess_tmp235)) + weak_hess_tmp123 + weak_hess_tmp224*weak_hess_tmp63) + trial_grad[4]*(iso_relaxation_gamma*(C01*(weak_hess_tmp161*weak_hess_tmp228 + weak_hess_tmp248) + C10*(weak_hess_tmp156*weak_hess_tmp226 + weak_hess_tmp246)) + weak_hess_tmp153*weak_hess_tmp224 + weak_hess_tmp245) + trial_grad[5]*(iso_relaxation_gamma*(C01*(weak_hess_tmp131*weak_hess_tmp228 + weak_hess_tmp232) + C10*(weak_hess_tmp125*weak_hess_tmp226 + weak_hess_tmp230)) + weak_hess_tmp118*weak_hess_tmp224) + trial_grad[6]*(iso_relaxation_gamma*(C01*(weak_hess_tmp228*weak_hess_tmp86 + weak_hess_tmp244) + C10*(weak_hess_tmp226*weak_hess_tmp82 + weak_hess_tmp242)) + weak_hess_tmp224*weak_hess_tmp81 + weak_hess_tmp241) + trial_grad[7]*(iso_relaxation_gamma*(C01*(weak_hess_tmp147*weak_hess_tmp228 + weak_hess_tmp240) + C10*(weak_hess_tmp143*weak_hess_tmp226 + weak_hess_tmp238)) + weak_hess_tmp140*weak_hess_tmp224 + weak_hess_tmp215) + trial_grad[8]*(iso_relaxation_gamma*(C01*(weak_hess_tmp174*weak_hess_tmp228 + weak_hess_tmp234) + C10*(weak_hess_tmp170*weak_hess_tmp226 + weak_hess_tmp233)) + weak_hess_tmp167*weak_hess_tmp224);
        material[3] = trial_grad[0]*(iso_relaxation_gamma*(C01*(weak_hess_tmp257*weak_hess_tmp54 + weak_hess_tmp78) + C10*(weak_hess_tmp17*weak_hess_tmp251 + weak_hess_tmp69)) + weak_hess_tmp249*weak_hess_tmp3) + trial_grad[1]*(iso_relaxation_gamma*(C01*(weak_hess_tmp214 + weak_hess_tmp257*weak_hess_tmp99) + C10*(weak_hess_tmp212 + weak_hess_tmp251*weak_hess_tmp96)) + weak_hess_tmp211 + weak_hess_tmp249*weak_hess_tmp94) + trial_grad[2]*(iso_relaxation_gamma*(C01*(weak_hess_tmp110*weak_hess_tmp257 + weak_hess_tmp237) + C10*(weak_hess_tmp107*weak_hess_tmp251 + weak_hess_tmp235)) + weak_hess_tmp106*weak_hess_tmp249 + weak_hess_tmp123) + trial_grad[3]*(iso_history_qp[0] + iso_relaxation_gamma*(C01*(s_t(2)*weak_hess_tmp256*weak_hess_tmp76 + weak_hess_tmp257*weak_hess_tmp70 + weak_hess_tmp39*(weak_hess_tmp253 + weak_hess_tmp255)) + C10*(s_t(4)*gu3*weak_hess_tmp250 + weak_hess_tmp13 + weak_hess_tmp251*weak_hess_tmp65)) + weak_hess_tmp249*weak_hess_tmp63) + trial_grad[4]*(iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_hess_tmp161*weak_hess_tmp257 + weak_hess_tmp269) + C10*(weak_hess_tmp156*weak_hess_tmp251 + weak_hess_tmp267)) + weak_hess_tmp153*weak_hess_tmp249) + trial_grad[5]*(iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_hess_tmp131*weak_hess_tmp257 + weak_hess_tmp266) + C10*(weak_hess_tmp125*weak_hess_tmp251 + weak_hess_tmp264)) + weak_hess_tmp118*weak_hess_tmp249) + trial_grad[6]*(iso_relaxation_gamma*(C01*(weak_hess_tmp257*weak_hess_tmp86 + weak_hess_tmp263) + C10*(weak_hess_tmp251*weak_hess_tmp82 + weak_hess_tmp259)) + weak_hess_tmp249*weak_hess_tmp81) + trial_grad[7]*(iso_relaxation_gamma*(C01*(weak_hess_tmp147*weak_hess_tmp257 + weak_hess_tmp274) + C10*(weak_hess_tmp143*weak_hess_tmp251 + weak_hess_tmp272)) + weak_hess_tmp140*weak_hess_tmp249 + weak_hess_tmp270) + trial_grad[8]*(iso_relaxation_gamma*(C01*(weak_hess_tmp174*weak_hess_tmp257 + weak_hess_tmp280) + C10*(weak_hess_tmp170*weak_hess_tmp251 + weak_hess_tmp278)) + weak_hess_tmp167*weak_hess_tmp249 + weak_hess_tmp276);
        material[4] = trial_grad[0]*(iso_relaxation_gamma*(C01*(weak_hess_tmp166 + weak_hess_tmp287*weak_hess_tmp54) + C10*(weak_hess_tmp160 + weak_hess_tmp17*weak_hess_tmp283)) + weak_hess_tmp154 + weak_hess_tmp281*weak_hess_tmp3) + trial_grad[1]*(iso_relaxation_gamma*(C01*(weak_hess_tmp201 + weak_hess_tmp287*weak_hess_tmp99) + C10*(weak_hess_tmp199 + weak_hess_tmp283*weak_hess_tmp96)) + weak_hess_tmp281*weak_hess_tmp94) + trial_grad[2]*(iso_relaxation_gamma*(C01*(weak_hess_tmp110*weak_hess_tmp287 + weak_hess_tmp248) + C10*(weak_hess_tmp107*weak_hess_tmp283 + weak_hess_tmp246)) + weak_hess_tmp106*weak_hess_tmp281 + weak_hess_tmp245) + trial_grad[3]*(iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_hess_tmp269 + weak_hess_tmp287*weak_hess_tmp70) + C10*(weak_hess_tmp267 + weak_hess_tmp283*weak_hess_tmp65)) + weak_hess_tmp281*weak_hess_tmp63) + trial_grad[4]*(iso_history_qp[1] + iso_relaxation_gamma*(C01*(weak_hess_tmp161*weak_hess_tmp287 + s_t(2)*weak_hess_tmp163*weak_hess_tmp286 + weak_hess_tmp39*(weak_hess_tmp253 + weak_hess_tmp285)) + C10*(s_t(4)*weak_hess_tmp1*weak_hess_tmp282 + weak_hess_tmp13 + weak_hess_tmp156*weak_hess_tmp283)) + weak_hess_tmp153*weak_hess_tmp281) + trial_grad[5]*(iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_hess_tmp131*weak_hess_tmp287 + weak_hess_tmp295) + C10*(weak_hess_tmp125*weak_hess_tmp283 + weak_hess_tmp293)) + weak_hess_tmp118*weak_hess_tmp281) + trial_grad[6]*(iso_relaxation_gamma*(C01*(weak_hess_tmp287*weak_hess_tmp86 + weak_hess_tmp298) + C10*(weak_hess_tmp283*weak_hess_tmp82 + weak_hess_tmp297)) + weak_hess_tmp281*weak_hess_tmp81 + weak_hess_tmp296) + trial_grad[7]*(iso_relaxation_gamma*(C01*(weak_hess_tmp147*weak_hess_tmp287 + weak_hess_tmp292) + C10*(weak_hess_tmp143*weak_hess_tmp283 + weak_hess_tmp289)) + weak_hess_tmp140*weak_hess_tmp281) + trial_grad[8]*(iso_relaxation_gamma*(C01*(weak_hess_tmp174*weak_hess_tmp287 + weak_hess_tmp303) + C10*(weak_hess_tmp170*weak_hess_tmp283 + weak_hess_tmp301)) + weak_hess_tmp167*weak_hess_tmp281 + weak_hess_tmp299);
        material[5] = trial_grad[0]*(iso_relaxation_gamma*(C01*(weak_hess_tmp138 + weak_hess_tmp308*weak_hess_tmp54) + C10*(weak_hess_tmp130 + weak_hess_tmp17*weak_hess_tmp306)) + weak_hess_tmp124 + weak_hess_tmp3*weak_hess_tmp304) + trial_grad[1]*(iso_relaxation_gamma*(C01*(weak_hess_tmp207 + weak_hess_tmp308*weak_hess_tmp99) + C10*(weak_hess_tmp204 + weak_hess_tmp306*weak_hess_tmp96)) + weak_hess_tmp202 + weak_hess_tmp304*weak_hess_tmp94) + trial_grad[2]*(iso_relaxation_gamma*(C01*(weak_hess_tmp110*weak_hess_tmp308 + weak_hess_tmp232) + C10*(weak_hess_tmp107*weak_hess_tmp306 + weak_hess_tmp230)) + weak_hess_tmp106*weak_hess_tmp304) + trial_grad[3]*(iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_hess_tmp266 + weak_hess_tmp308*weak_hess_tmp70) + C10*(weak_hess_tmp264 + weak_hess_tmp306*weak_hess_tmp65)) + weak_hess_tmp304*weak_hess_tmp63) + trial_grad[4]*(iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_hess_tmp161*weak_hess_tmp308 + weak_hess_tmp295) + C10*(weak_hess_tmp156*weak_hess_tmp306 + weak_hess_tmp293)) + weak_hess_tmp153*weak_hess_tmp304) + trial_grad[5]*(iso_history_qp[2] + iso_relaxation_gamma*(C01*(weak_hess_tmp131*weak_hess_tmp308 + s_t(2)*weak_hess_tmp134*weak_hess_tmp307 + weak_hess_tmp39*(weak_hess_tmp255 + weak_hess_tmp285)) + C10*(s_t(4)*gu5*weak_hess_tmp305 + weak_hess_tmp125*weak_hess_tmp306 + weak_hess_tmp13)) + weak_hess_tmp118*weak_hess_tmp304) + trial_grad[6]*(iso_relaxation_gamma*(C01*(weak_hess_tmp308*weak_hess_tmp86 + weak_hess_tmp314) + C10*(weak_hess_tmp306*weak_hess_tmp82 + weak_hess_tmp313)) + weak_hess_tmp275 + weak_hess_tmp304*weak_hess_tmp81) + trial_grad[7]*(iso_relaxation_gamma*(C01*(weak_hess_tmp147*weak_hess_tmp308 + weak_hess_tmp317) + C10*(weak_hess_tmp143*weak_hess_tmp306 + weak_hess_tmp316)) + weak_hess_tmp140*weak_hess_tmp304 + weak_hess_tmp315) + trial_grad[8]*(iso_relaxation_gamma*(C01*(weak_hess_tmp174*weak_hess_tmp308 + weak_hess_tmp312) + C10*(weak_hess_tmp170*weak_hess_tmp306 + weak_hess_tmp310)) + weak_hess_tmp167*weak_hess_tmp304);
        material[6] = trial_grad[0]*(iso_relaxation_gamma*(C01*(weak_hess_tmp324*weak_hess_tmp54 + weak_hess_tmp93) + C10*(weak_hess_tmp17*weak_hess_tmp320 + weak_hess_tmp85)) + weak_hess_tmp3*weak_hess_tmp318) + trial_grad[1]*(iso_relaxation_gamma*(C01*(weak_hess_tmp210 + weak_hess_tmp324*weak_hess_tmp99) + C10*(weak_hess_tmp208 + weak_hess_tmp320*weak_hess_tmp96)) + weak_hess_tmp141 + weak_hess_tmp318*weak_hess_tmp94) + trial_grad[2]*(iso_relaxation_gamma*(C01*(weak_hess_tmp110*weak_hess_tmp324 + weak_hess_tmp244) + C10*(weak_hess_tmp107*weak_hess_tmp320 + weak_hess_tmp242)) + weak_hess_tmp106*weak_hess_tmp318 + weak_hess_tmp241) + trial_grad[3]*(iso_relaxation_gamma*(C01*(weak_hess_tmp263 + weak_hess_tmp324*weak_hess_tmp70) + C10*(weak_hess_tmp259 + weak_hess_tmp320*weak_hess_tmp65)) + weak_hess_tmp318*weak_hess_tmp63) + trial_grad[4]*(iso_relaxation_gamma*(C01*(weak_hess_tmp161*weak_hess_tmp324 + weak_hess_tmp298) + C10*(weak_hess_tmp156*weak_hess_tmp320 + weak_hess_tmp297)) + weak_hess_tmp153*weak_hess_tmp318 + weak_hess_tmp296) + trial_grad[5]*(iso_relaxation_gamma*(C01*(weak_hess_tmp131*weak_hess_tmp324 + weak_hess_tmp314) + C10*(weak_hess_tmp125*weak_hess_tmp320 + weak_hess_tmp313)) + weak_hess_tmp118*weak_hess_tmp318 + weak_hess_tmp275) + trial_grad[6]*(iso_history_qp[0] + iso_relaxation_gamma*(C01*(s_t(2)*weak_hess_tmp323*weak_hess_tmp92 + weak_hess_tmp324*weak_hess_tmp86 + weak_hess_tmp39*(weak_hess_tmp321 + weak_hess_tmp322)) + C10*(s_t(4)*gu6*weak_hess_tmp319 + weak_hess_tmp13 + weak_hess_tmp320*weak_hess_tmp82)) + weak_hess_tmp318*weak_hess_tmp81) + trial_grad[7]*(iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_hess_tmp147*weak_hess_tmp324 + weak_hess_tmp328) + C10*(weak_hess_tmp143*weak_hess_tmp320 + weak_hess_tmp326)) + weak_hess_tmp140*weak_hess_tmp318) + trial_grad[8]*(iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_hess_tmp174*weak_hess_tmp324 + weak_hess_tmp330) + C10*(weak_hess_tmp170*weak_hess_tmp320 + weak_hess_tmp329)) + weak_hess_tmp167*weak_hess_tmp318);
        material[7] = trial_grad[0]*(iso_relaxation_gamma*(C01*(weak_hess_tmp152 + weak_hess_tmp336*weak_hess_tmp54) + C10*(weak_hess_tmp146 + weak_hess_tmp17*weak_hess_tmp333)) + weak_hess_tmp142 + weak_hess_tmp3*weak_hess_tmp331) + trial_grad[1]*(iso_relaxation_gamma*(C01*(weak_hess_tmp194 + weak_hess_tmp336*weak_hess_tmp99) + C10*(weak_hess_tmp191 + weak_hess_tmp333*weak_hess_tmp96)) + weak_hess_tmp331*weak_hess_tmp94) + trial_grad[2]*(iso_relaxation_gamma*(C01*(weak_hess_tmp110*weak_hess_tmp336 + weak_hess_tmp240) + C10*(weak_hess_tmp107*weak_hess_tmp333 + weak_hess_tmp238)) + weak_hess_tmp106*weak_hess_tmp331 + weak_hess_tmp215) + trial_grad[3]*(iso_relaxation_gamma*(C01*(weak_hess_tmp274 + weak_hess_tmp336*weak_hess_tmp70) + C10*(weak_hess_tmp272 + weak_hess_tmp333*weak_hess_tmp65)) + weak_hess_tmp270 + weak_hess_tmp331*weak_hess_tmp63) + trial_grad[4]*(iso_relaxation_gamma*(C01*(weak_hess_tmp161*weak_hess_tmp336 + weak_hess_tmp292) + C10*(weak_hess_tmp156*weak_hess_tmp333 + weak_hess_tmp289)) + weak_hess_tmp153*weak_hess_tmp331) + trial_grad[5]*(iso_relaxation_gamma*(C01*(weak_hess_tmp131*weak_hess_tmp336 + weak_hess_tmp317) + C10*(weak_hess_tmp125*weak_hess_tmp333 + weak_hess_tmp316)) + weak_hess_tmp118*weak_hess_tmp331 + weak_hess_tmp315) + trial_grad[6]*(iso_history_qp[3] + iso_relaxation_gamma*(C01*(weak_hess_tmp328 + weak_hess_tmp336*weak_hess_tmp86) + C10*(weak_hess_tmp326 + weak_hess_tmp333*weak_hess_tmp82)) + weak_hess_tmp331*weak_hess_tmp81) + trial_grad[7]*(iso_history_qp[1] + iso_relaxation_gamma*(C01*(weak_hess_tmp147*weak_hess_tmp336 + s_t(2)*weak_hess_tmp150*weak_hess_tmp335 + weak_hess_tmp39*(weak_hess_tmp321 + weak_hess_tmp334)) + C10*(s_t(4)*gu7*weak_hess_tmp332 + weak_hess_tmp13 + weak_hess_tmp143*weak_hess_tmp333)) + weak_hess_tmp140*weak_hess_tmp331) + trial_grad[8]*(iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_hess_tmp174*weak_hess_tmp336 + weak_hess_tmp339) + C10*(weak_hess_tmp170*weak_hess_tmp333 + weak_hess_tmp337)) + weak_hess_tmp167*weak_hess_tmp331);
        material[8] = trial_grad[0]*(iso_relaxation_gamma*(C01*(weak_hess_tmp179 + weak_hess_tmp342*weak_hess_tmp54) + C10*(weak_hess_tmp17*weak_hess_tmp341 + weak_hess_tmp173)) + weak_hess_tmp168 + weak_hess_tmp3*weak_hess_tmp340) + trial_grad[1]*(iso_relaxation_gamma*(C01*(weak_hess_tmp221 + weak_hess_tmp342*weak_hess_tmp99) + C10*(weak_hess_tmp218 + weak_hess_tmp341*weak_hess_tmp96)) + weak_hess_tmp216 + weak_hess_tmp340*weak_hess_tmp94) + trial_grad[2]*(iso_relaxation_gamma*(C01*(weak_hess_tmp110*weak_hess_tmp342 + weak_hess_tmp234) + C10*(weak_hess_tmp107*weak_hess_tmp341 + weak_hess_tmp233)) + weak_hess_tmp106*weak_hess_tmp340) + trial_grad[3]*(iso_relaxation_gamma*(C01*(weak_hess_tmp280 + weak_hess_tmp342*weak_hess_tmp70) + C10*(weak_hess_tmp278 + weak_hess_tmp341*weak_hess_tmp65)) + weak_hess_tmp276 + weak_hess_tmp340*weak_hess_tmp63) + trial_grad[4]*(iso_relaxation_gamma*(C01*(weak_hess_tmp161*weak_hess_tmp342 + weak_hess_tmp303) + C10*(weak_hess_tmp156*weak_hess_tmp341 + weak_hess_tmp301)) + weak_hess_tmp153*weak_hess_tmp340 + weak_hess_tmp299) + trial_grad[5]*(iso_relaxation_gamma*(C01*(weak_hess_tmp131*weak_hess_tmp342 + weak_hess_tmp312) + C10*(weak_hess_tmp125*weak_hess_tmp341 + weak_hess_tmp310)) + weak_hess_tmp118*weak_hess_tmp340) + trial_grad[6]*(iso_history_qp[4] + iso_relaxation_gamma*(C01*(weak_hess_tmp330 + weak_hess_tmp342*weak_hess_tmp86) + C10*(weak_hess_tmp329 + weak_hess_tmp341*weak_hess_tmp82)) + weak_hess_tmp340*weak_hess_tmp81) + trial_grad[7]*(iso_history_qp[5] + iso_relaxation_gamma*(C01*(weak_hess_tmp147*weak_hess_tmp342 + weak_hess_tmp339) + C10*(weak_hess_tmp143*weak_hess_tmp341 + weak_hess_tmp337)) + weak_hess_tmp140*weak_hess_tmp340) + trial_grad[8]*(iso_history_qp[2] + iso_relaxation_gamma*(C01*(weak_hess_tmp174*weak_hess_tmp342 + s_t(2)*weak_hess_tmp177*weak_hess_tmp338 + weak_hess_tmp39*(weak_hess_tmp322 + weak_hess_tmp334)) + C10*(weak_hess_tmp13 + s_t(4)*weak_hess_tmp158*weak_hess_tmp171 + weak_hess_tmp170*weak_hess_tmp341)) + weak_hess_tmp167*weak_hess_tmp340);
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
      tensor_test_scalar<s_t, NQ, NS, 3, NC>(
          shape_1d, grad_1d, flux + 0, column, 0);
      tensor_test_scalar<s_t, NQ, NS, 3, NC>(
          shape_1d, grad_1d, flux + NQ * ND, column, 1);
      tensor_test_scalar<s_t, NQ, NS, 3, NC>(
          shape_1d, grad_1d, flux + 2 * NQ * ND, column, 2);
    }
  }
}

} // namespace codegen
} // namespace sfem

#endif
