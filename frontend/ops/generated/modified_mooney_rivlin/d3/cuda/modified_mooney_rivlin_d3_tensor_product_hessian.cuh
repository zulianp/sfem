#ifndef MODIFIED_MOONEY_RIVLIN_D3_TENSOR_PRODUCT_HESSIAN_CUH
#define MODIFIED_MOONEY_RIVLIN_D3_TENSOR_PRODUCT_HESSIAN_CUH
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
static __host__ __device__ __forceinline__ void modified_mooney_rivlin_d3_tensor_product_direct_hessian_tensor_product_element_matrix(
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
    const s_t gu_ref0 = state_gradient_ref[q * ND];
    const s_t gu_ref1 = state_gradient_ref[q * ND + 1];
    const s_t gu_ref2 = state_gradient_ref[q * ND + 2];
    const s_t gu_ref3 = state_gradient_ref[NQ * ND + q * ND];
    const s_t gu_ref4 = state_gradient_ref[NQ * ND + q * ND + 1];
    const s_t gu_ref5 = state_gradient_ref[NQ * ND + q * ND + 2];
    const s_t gu_ref6 = state_gradient_ref[2 * NQ * ND + q * ND];
    const s_t gu_ref7 = state_gradient_ref[2 * NQ * ND + q * ND + 1];
    const s_t gu_ref8 = state_gradient_ref[2 * NQ * ND + q * ND + 2];
    const s_t gu0 = (gu_ref0 * adj_value0 + gu_ref1 * adj_value3 + gu_ref2 * adj_value6) * idet;
    const s_t gu1 = (gu_ref0 * adj_value1 + gu_ref1 * adj_value4 + gu_ref2 * adj_value7) * idet;
    const s_t gu2 = (gu_ref0 * adj_value2 + gu_ref1 * adj_value5 + gu_ref2 * adj_value8) * idet;
    const s_t gu3 = (gu_ref3 * adj_value0 + gu_ref4 * adj_value3 + gu_ref5 * adj_value6) * idet;
    const s_t gu4 = (gu_ref3 * adj_value1 + gu_ref4 * adj_value4 + gu_ref5 * adj_value7) * idet;
    const s_t gu5 = (gu_ref3 * adj_value2 + gu_ref4 * adj_value5 + gu_ref5 * adj_value8) * idet;
    const s_t gu6 = (gu_ref6 * adj_value0 + gu_ref7 * adj_value3 + gu_ref8 * adj_value6) * idet;
    const s_t gu7 = (gu_ref6 * adj_value1 + gu_ref7 * adj_value4 + gu_ref8 * adj_value7) * idet;
    const s_t gu8 = (gu_ref6 * adj_value2 + gu_ref7 * adj_value5 + gu_ref8 * adj_value8) * idet;
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
    const s_t weak_hess_tmp21 = pow_2(gu3);
    const s_t weak_hess_tmp22 = pow_2(gu6);
    const s_t weak_hess_tmp23 = pow_2(weak_hess_tmp7);
    const s_t weak_hess_tmp24 = weak_hess_tmp21 + weak_hess_tmp22 + weak_hess_tmp23;
    const s_t weak_hess_tmp25 = pow_2(gu1);
    const s_t weak_hess_tmp26 = pow_2(gu7);
    const s_t weak_hess_tmp27 = pow_2(weak_hess_tmp1);
    const s_t weak_hess_tmp28 = weak_hess_tmp25 + weak_hess_tmp26 + weak_hess_tmp27;
    const s_t weak_hess_tmp29 = pow_2(gu2);
    const s_t weak_hess_tmp30 = pow_2(gu5);
    const s_t weak_hess_tmp31 = pow_2(weak_hess_tmp2);
    const s_t weak_hess_tmp32 = weak_hess_tmp29 + weak_hess_tmp30 + weak_hess_tmp31;
    const s_t weak_hess_tmp33 = weak_hess_tmp24 + weak_hess_tmp28 + weak_hess_tmp32;
    const s_t weak_hess_tmp34 = weak_hess_tmp33/pow(weak_hess_tmp8, (s_t(8) / s_t(3)));
    const s_t weak_hess_tmp35 = weak_hess_tmp19*weak_hess_tmp34;
    const s_t weak_hess_tmp36 = s_t(2)*weak_hess_tmp30;
    const s_t weak_hess_tmp37 = s_t(2)*weak_hess_tmp31;
    const s_t weak_hess_tmp38 = weak_hess_tmp36 + weak_hess_tmp37;
    const s_t weak_hess_tmp39 = s_t(2)*weak_hess_tmp26;
    const s_t weak_hess_tmp40 = s_t(2)*weak_hess_tmp27;
    const s_t weak_hess_tmp41 = weak_hess_tmp39 + weak_hess_tmp40;
    const s_t weak_hess_tmp42 = pow(weak_hess_tmp8, (s_t(-4) / s_t(3)));
    const s_t weak_hess_tmp43 = ((s_t(4) / s_t(3)))*weak_hess_tmp0 - (s_t(4) / s_t(3))*weak_hess_tmp18;
    const s_t weak_hess_tmp44 = pow(weak_hess_tmp8, (s_t(-7) / s_t(3)));
    const s_t weak_hess_tmp45 = gu6*gu7;
    const s_t weak_hess_tmp46 = gu1*weak_hess_tmp7;
    const s_t weak_hess_tmp47 = gu3*weak_hess_tmp1;
    const s_t weak_hess_tmp48 = weak_hess_tmp45 + weak_hess_tmp46 + weak_hess_tmp47;
    const s_t weak_hess_tmp49 = s_t(2)*gu1;
    const s_t weak_hess_tmp50 = gu3*gu5;
    const s_t weak_hess_tmp51 = gu2*weak_hess_tmp7;
    const s_t weak_hess_tmp52 = gu6*weak_hess_tmp2;
    const s_t weak_hess_tmp53 = weak_hess_tmp50 + weak_hess_tmp51 + weak_hess_tmp52;
    const s_t weak_hess_tmp54 = s_t(2)*gu2;
    const s_t weak_hess_tmp55 = s_t(2)*weak_hess_tmp7;
    const s_t weak_hess_tmp56 = weak_hess_tmp44*(-weak_hess_tmp24*weak_hess_tmp55 + s_t(2)*weak_hess_tmp33*weak_hess_tmp7 - weak_hess_tmp48*weak_hess_tmp49 - weak_hess_tmp53*weak_hess_tmp54);
    const s_t weak_hess_tmp57 = gu1*gu2;
    const s_t weak_hess_tmp58 = gu5*weak_hess_tmp1;
    const s_t weak_hess_tmp59 = gu7*weak_hess_tmp2;
    const s_t weak_hess_tmp60 = weak_hess_tmp57 + weak_hess_tmp58 + weak_hess_tmp59;
    const s_t weak_hess_tmp61 = -(s_t(1) / s_t(2))*pow_2(weak_hess_tmp24) - (s_t(1) / s_t(2))*pow_2(weak_hess_tmp28) - (s_t(1) / s_t(2))*pow_2(weak_hess_tmp32) + ((s_t(1) / s_t(2)))*pow_2(weak_hess_tmp33) - pow_2(weak_hess_tmp48) - pow_2(weak_hess_tmp53) - pow_2(weak_hess_tmp60);
    const s_t weak_hess_tmp62 = weak_hess_tmp61/pow(weak_hess_tmp8, (s_t(10) / s_t(3)));
    const s_t weak_hess_tmp63 = weak_hess_tmp43*weak_hess_tmp62;
    const s_t weak_hess_tmp64 = gu3*weak_hess_tmp2;
    const s_t weak_hess_tmp65 = -gu5*gu6 + weak_hess_tmp64;
    const s_t weak_hess_tmp66 = -weak_hess_tmp65;
    const s_t weak_hess_tmp67 = -(s_t(2) / s_t(3))*weak_hess_tmp11 + ((s_t(2) / s_t(3)))*weak_hess_tmp64;
    const s_t weak_hess_tmp68 = weak_hess_tmp17*weak_hess_tmp55;
    const s_t weak_hess_tmp69 = -(s_t(5) / s_t(3))*weak_hess_tmp11 + ((s_t(5) / s_t(3)))*weak_hess_tmp64;
    const s_t weak_hess_tmp70 = s_t(2)*weak_hess_tmp45;
    const s_t weak_hess_tmp71 = s_t(2)*weak_hess_tmp47;
    const s_t weak_hess_tmp72 = -(s_t(4) / s_t(3))*weak_hess_tmp11 + ((s_t(4) / s_t(3)))*weak_hess_tmp64;
    const s_t weak_hess_tmp73 = s_t(2)*gu1*weak_hess_tmp33 - weak_hess_tmp28*weak_hess_tmp49 - weak_hess_tmp48*weak_hess_tmp55 - weak_hess_tmp54*weak_hess_tmp60;
    const s_t weak_hess_tmp74 = weak_hess_tmp43*weak_hess_tmp44;
    const s_t weak_hess_tmp75 = -(s_t(7) / s_t(3))*weak_hess_tmp11 + ((s_t(7) / s_t(3)))*weak_hess_tmp64;
    const s_t weak_hess_tmp76 = gu6*weak_hess_tmp1;
    const s_t weak_hess_tmp77 = weak_hess_tmp12 - weak_hess_tmp76;
    const s_t weak_hess_tmp78 = -weak_hess_tmp77;
    const s_t weak_hess_tmp79 = ((s_t(2) / s_t(3)))*gu6*weak_hess_tmp1 - (s_t(2) / s_t(3))*weak_hess_tmp12;
    const s_t weak_hess_tmp80 = ((s_t(5) / s_t(3)))*gu6*weak_hess_tmp1 - (s_t(5) / s_t(3))*weak_hess_tmp12;
    const s_t weak_hess_tmp81 = s_t(2)*weak_hess_tmp50;
    const s_t weak_hess_tmp82 = s_t(2)*weak_hess_tmp52;
    const s_t weak_hess_tmp83 = ((s_t(4) / s_t(3)))*gu6*weak_hess_tmp1 - (s_t(4) / s_t(3))*weak_hess_tmp12;
    const s_t weak_hess_tmp84 = s_t(2)*gu2*weak_hess_tmp33 - weak_hess_tmp32*weak_hess_tmp54 - weak_hess_tmp49*weak_hess_tmp60 - weak_hess_tmp53*weak_hess_tmp55;
    const s_t weak_hess_tmp85 = ((s_t(7) / s_t(3)))*gu6*weak_hess_tmp1 - (s_t(7) / s_t(3))*weak_hess_tmp12;
    const s_t weak_hess_tmp86 = gu1*weak_hess_tmp2;
    const s_t weak_hess_tmp87 = -gu2*gu7 + weak_hess_tmp86;
    const s_t weak_hess_tmp88 = -weak_hess_tmp87;
    const s_t weak_hess_tmp89 = s_t(2)*gu3;
    const s_t weak_hess_tmp90 = gu2*gu7;
    const s_t weak_hess_tmp91 = ((s_t(2) / s_t(3)))*weak_hess_tmp86 - (s_t(2) / s_t(3))*weak_hess_tmp90;
    const s_t weak_hess_tmp92 = ((s_t(5) / s_t(3)))*weak_hess_tmp86 - (s_t(5) / s_t(3))*weak_hess_tmp90;
    const s_t weak_hess_tmp93 = gu5*weak_hess_tmp54;
    const s_t weak_hess_tmp94 = weak_hess_tmp1*weak_hess_tmp49;
    const s_t weak_hess_tmp95 = ((s_t(4) / s_t(3)))*weak_hess_tmp86 - (s_t(4) / s_t(3))*weak_hess_tmp90;
    const s_t weak_hess_tmp96 = s_t(2)*gu5;
    const s_t weak_hess_tmp97 = s_t(2)*weak_hess_tmp1;
    const s_t weak_hess_tmp98 = s_t(2)*gu3*weak_hess_tmp33 - weak_hess_tmp24*weak_hess_tmp89 - weak_hess_tmp48*weak_hess_tmp97 - weak_hess_tmp53*weak_hess_tmp96;
    const s_t weak_hess_tmp99 = ((s_t(7) / s_t(3)))*weak_hess_tmp86 - (s_t(7) / s_t(3))*weak_hess_tmp90;
    const s_t weak_hess_tmp100 = -weak_hess_tmp2*weak_hess_tmp7 + weak_hess_tmp6;
    const s_t weak_hess_tmp101 = -weak_hess_tmp100;
    const s_t weak_hess_tmp102 = kappa*weak_hess_tmp13/weak_hess_tmp8;
    const s_t weak_hess_tmp103 = weak_hess_tmp102*weak_hess_tmp2;
    const s_t weak_hess_tmp104 = weak_hess_tmp2*weak_hess_tmp7;
    const s_t weak_hess_tmp105 = -(s_t(2) / s_t(3))*weak_hess_tmp104 + ((s_t(2) / s_t(3)))*weak_hess_tmp6;
    const s_t weak_hess_tmp106 = weak_hess_tmp17*weak_hess_tmp2;
    const s_t weak_hess_tmp107 = ((s_t(2) / s_t(3)))*weak_hess_tmp33;
    const s_t weak_hess_tmp108 = weak_hess_tmp106*weak_hess_tmp107;
    const s_t weak_hess_tmp109 = -(s_t(5) / s_t(3))*weak_hess_tmp104 + ((s_t(5) / s_t(3)))*weak_hess_tmp6;
    const s_t weak_hess_tmp110 = -(s_t(4) / s_t(3))*weak_hess_tmp104 + ((s_t(4) / s_t(3)))*weak_hess_tmp6;
    const s_t weak_hess_tmp111 = s_t(2)*weak_hess_tmp1*weak_hess_tmp33 - weak_hess_tmp28*weak_hess_tmp97 - weak_hess_tmp48*weak_hess_tmp89 - weak_hess_tmp60*weak_hess_tmp96;
    const s_t weak_hess_tmp112 = ((s_t(4) / s_t(3)))*weak_hess_tmp44*weak_hess_tmp61;
    const s_t weak_hess_tmp113 = weak_hess_tmp112*weak_hess_tmp2;
    const s_t weak_hess_tmp114 = -(s_t(7) / s_t(3))*weak_hess_tmp104 + ((s_t(7) / s_t(3)))*weak_hess_tmp6;
    const s_t weak_hess_tmp115 = gu1*gu6;
    const s_t weak_hess_tmp116 = gu7*weak_hess_tmp7;
    const s_t weak_hess_tmp117 = weak_hess_tmp115 - weak_hess_tmp116;
    const s_t weak_hess_tmp118 = gu7*weak_hess_tmp102;
    const s_t weak_hess_tmp119 = -weak_hess_tmp117;
    const s_t weak_hess_tmp120 = ((s_t(2) / s_t(3)))*gu7*weak_hess_tmp7 - (s_t(2) / s_t(3))*weak_hess_tmp115;
    const s_t weak_hess_tmp121 = weak_hess_tmp107*weak_hess_tmp17;
    const s_t weak_hess_tmp122 = gu7*weak_hess_tmp121;
    const s_t weak_hess_tmp123 = ((s_t(5) / s_t(3)))*gu7*weak_hess_tmp7 - (s_t(5) / s_t(3))*weak_hess_tmp115;
    const s_t weak_hess_tmp124 = gu2*gu3;
    const s_t weak_hess_tmp125 = ((s_t(4) / s_t(3)))*gu7*weak_hess_tmp7 - (s_t(4) / s_t(3))*weak_hess_tmp115;
    const s_t weak_hess_tmp126 = s_t(2)*gu5*weak_hess_tmp33 - weak_hess_tmp32*weak_hess_tmp96 - weak_hess_tmp53*weak_hess_tmp89 - weak_hess_tmp60*weak_hess_tmp97;
    const s_t weak_hess_tmp127 = gu7*weak_hess_tmp112;
    const s_t weak_hess_tmp128 = ((s_t(7) / s_t(3)))*gu7*weak_hess_tmp7 - (s_t(7) / s_t(3))*weak_hess_tmp115;
    const s_t weak_hess_tmp129 = gu1*gu5;
    const s_t weak_hess_tmp130 = gu2*weak_hess_tmp1;
    const s_t weak_hess_tmp131 = weak_hess_tmp129 - weak_hess_tmp130;
    const s_t weak_hess_tmp132 = -weak_hess_tmp131;
    const s_t weak_hess_tmp133 = s_t(2)*gu6;
    const s_t weak_hess_tmp134 = ((s_t(2) / s_t(3)))*gu2*weak_hess_tmp1 - (s_t(2) / s_t(3))*weak_hess_tmp129;
    const s_t weak_hess_tmp135 = ((s_t(5) / s_t(3)))*gu2*weak_hess_tmp1 - (s_t(5) / s_t(3))*weak_hess_tmp129;
    const s_t weak_hess_tmp136 = gu7*weak_hess_tmp49;
    const s_t weak_hess_tmp137 = weak_hess_tmp2*weak_hess_tmp54;
    const s_t weak_hess_tmp138 = ((s_t(4) / s_t(3)))*gu2*weak_hess_tmp1 - (s_t(4) / s_t(3))*weak_hess_tmp129;
    const s_t weak_hess_tmp139 = s_t(2)*gu7;
    const s_t weak_hess_tmp140 = s_t(2)*weak_hess_tmp2;
    const s_t weak_hess_tmp141 = s_t(2)*gu6*weak_hess_tmp33 - weak_hess_tmp133*weak_hess_tmp24 - weak_hess_tmp139*weak_hess_tmp48 - weak_hess_tmp140*weak_hess_tmp53;
    const s_t weak_hess_tmp142 = ((s_t(7) / s_t(3)))*gu2*weak_hess_tmp1 - (s_t(7) / s_t(3))*weak_hess_tmp129;
    const s_t weak_hess_tmp143 = gu5*weak_hess_tmp7;
    const s_t weak_hess_tmp144 = weak_hess_tmp124 - weak_hess_tmp143;
    const s_t weak_hess_tmp145 = gu5*weak_hess_tmp102;
    const s_t weak_hess_tmp146 = -weak_hess_tmp144;
    const s_t weak_hess_tmp147 = ((s_t(2) / s_t(3)))*gu5*weak_hess_tmp7 - (s_t(2) / s_t(3))*weak_hess_tmp124;
    const s_t weak_hess_tmp148 = gu5*weak_hess_tmp121;
    const s_t weak_hess_tmp149 = ((s_t(5) / s_t(3)))*gu5*weak_hess_tmp7 - (s_t(5) / s_t(3))*weak_hess_tmp124;
    const s_t weak_hess_tmp150 = ((s_t(4) / s_t(3)))*gu5*weak_hess_tmp7 - (s_t(4) / s_t(3))*weak_hess_tmp124;
    const s_t weak_hess_tmp151 = s_t(2)*gu7*weak_hess_tmp33 - weak_hess_tmp133*weak_hess_tmp48 - weak_hess_tmp139*weak_hess_tmp28 - weak_hess_tmp140*weak_hess_tmp60;
    const s_t weak_hess_tmp152 = gu5*weak_hess_tmp112;
    const s_t weak_hess_tmp153 = ((s_t(7) / s_t(3)))*gu5*weak_hess_tmp7 - (s_t(7) / s_t(3))*weak_hess_tmp124;
    const s_t weak_hess_tmp154 = -weak_hess_tmp1*weak_hess_tmp7 + weak_hess_tmp5;
    const s_t weak_hess_tmp155 = -weak_hess_tmp154;
    const s_t weak_hess_tmp156 = weak_hess_tmp1*weak_hess_tmp102;
    const s_t weak_hess_tmp157 = weak_hess_tmp1*weak_hess_tmp7;
    const s_t weak_hess_tmp158 = -(s_t(2) / s_t(3))*weak_hess_tmp157 + ((s_t(2) / s_t(3)))*weak_hess_tmp5;
    const s_t weak_hess_tmp159 = weak_hess_tmp1*weak_hess_tmp121;
    const s_t weak_hess_tmp160 = -(s_t(5) / s_t(3))*weak_hess_tmp157 + ((s_t(5) / s_t(3)))*weak_hess_tmp5;
    const s_t weak_hess_tmp161 = -(s_t(4) / s_t(3))*weak_hess_tmp157 + ((s_t(4) / s_t(3)))*weak_hess_tmp5;
    const s_t weak_hess_tmp162 = -weak_hess_tmp133*weak_hess_tmp53 - weak_hess_tmp139*weak_hess_tmp60 - weak_hess_tmp140*weak_hess_tmp32 + s_t(2)*weak_hess_tmp2*weak_hess_tmp33;
    const s_t weak_hess_tmp163 = weak_hess_tmp1*weak_hess_tmp112;
    const s_t weak_hess_tmp164 = -(s_t(7) / s_t(3))*weak_hess_tmp157 + ((s_t(7) / s_t(3)))*weak_hess_tmp5;
    const s_t weak_hess_tmp165 = weak_hess_tmp66*weak_hess_tmp9;
    const s_t weak_hess_tmp166 = weak_hess_tmp13*weak_hess_tmp165;
    const s_t weak_hess_tmp167 = weak_hess_tmp17*weak_hess_tmp67;
    const s_t weak_hess_tmp168 = weak_hess_tmp34*weak_hess_tmp67;
    const s_t weak_hess_tmp169 = s_t(2)*weak_hess_tmp21;
    const s_t weak_hess_tmp170 = s_t(2)*weak_hess_tmp22;
    const s_t weak_hess_tmp171 = weak_hess_tmp169 + weak_hess_tmp170;
    const s_t weak_hess_tmp172 = weak_hess_tmp44*weak_hess_tmp73;
    const s_t weak_hess_tmp173 = weak_hess_tmp62*weak_hess_tmp72;
    const s_t weak_hess_tmp174 = weak_hess_tmp17*weak_hess_tmp49;
    const s_t weak_hess_tmp175 = s_t(2)*weak_hess_tmp58;
    const s_t weak_hess_tmp176 = s_t(2)*weak_hess_tmp59;
    const s_t weak_hess_tmp177 = weak_hess_tmp44*weak_hess_tmp72;
    const s_t weak_hess_tmp178 = gu3*weak_hess_tmp55;
    const s_t weak_hess_tmp179 = gu6*weak_hess_tmp102;
    const s_t weak_hess_tmp180 = gu6*weak_hess_tmp121;
    const s_t weak_hess_tmp181 = gu6*weak_hess_tmp112;
    const s_t weak_hess_tmp182 = gu6*weak_hess_tmp55;
    const s_t weak_hess_tmp183 = gu3*weak_hess_tmp102;
    const s_t weak_hess_tmp184 = gu3*weak_hess_tmp121;
    const s_t weak_hess_tmp185 = gu3*weak_hess_tmp112;
    const s_t weak_hess_tmp186 = weak_hess_tmp77*weak_hess_tmp9;
    const s_t weak_hess_tmp187 = weak_hess_tmp13*weak_hess_tmp186;
    const s_t weak_hess_tmp188 = weak_hess_tmp17*weak_hess_tmp79;
    const s_t weak_hess_tmp189 = weak_hess_tmp34*weak_hess_tmp79;
    const s_t weak_hess_tmp190 = weak_hess_tmp44*weak_hess_tmp84;
    const s_t weak_hess_tmp191 = weak_hess_tmp62*weak_hess_tmp83;
    const s_t weak_hess_tmp192 = weak_hess_tmp17*weak_hess_tmp54;
    const s_t weak_hess_tmp193 = weak_hess_tmp44*weak_hess_tmp83;
    const s_t weak_hess_tmp194 = weak_hess_tmp88*weak_hess_tmp9;
    const s_t weak_hess_tmp195 = weak_hess_tmp13*weak_hess_tmp194;
    const s_t weak_hess_tmp196 = weak_hess_tmp17*weak_hess_tmp91;
    const s_t weak_hess_tmp197 = weak_hess_tmp34*weak_hess_tmp91;
    const s_t weak_hess_tmp198 = s_t(2)*weak_hess_tmp29;
    const s_t weak_hess_tmp199 = weak_hess_tmp198 + weak_hess_tmp37;
    const s_t weak_hess_tmp200 = s_t(2)*weak_hess_tmp25;
    const s_t weak_hess_tmp201 = weak_hess_tmp200 + weak_hess_tmp39;
    const s_t weak_hess_tmp202 = weak_hess_tmp44*weak_hess_tmp95;
    const s_t weak_hess_tmp203 = weak_hess_tmp62*weak_hess_tmp95;
    const s_t weak_hess_tmp204 = weak_hess_tmp17*weak_hess_tmp89;
    const s_t weak_hess_tmp205 = s_t(2)*weak_hess_tmp46;
    const s_t weak_hess_tmp206 = weak_hess_tmp44*weak_hess_tmp98;
    const s_t weak_hess_tmp207 = s_t(2)*weak_hess_tmp51;
    const s_t weak_hess_tmp208 = weak_hess_tmp2*weak_hess_tmp96;
    const s_t weak_hess_tmp209 = gu7*weak_hess_tmp97;
    const s_t weak_hess_tmp210 = gu2*weak_hess_tmp102;
    const s_t weak_hess_tmp211 = gu2*weak_hess_tmp121;
    const s_t weak_hess_tmp212 = gu2*weak_hess_tmp112;
    const s_t weak_hess_tmp213 = gu1*weak_hess_tmp102;
    const s_t weak_hess_tmp214 = gu1*weak_hess_tmp121;
    const s_t weak_hess_tmp215 = gu1*weak_hess_tmp112;
    const s_t weak_hess_tmp216 = weak_hess_tmp101*weak_hess_tmp9;
    const s_t weak_hess_tmp217 = weak_hess_tmp13*weak_hess_tmp216;
    const s_t weak_hess_tmp218 = weak_hess_tmp105*weak_hess_tmp17;
    const s_t weak_hess_tmp219 = weak_hess_tmp105*weak_hess_tmp34;
    const s_t weak_hess_tmp220 = s_t(2)*weak_hess_tmp23;
    const s_t weak_hess_tmp221 = weak_hess_tmp170 + weak_hess_tmp220;
    const s_t weak_hess_tmp222 = weak_hess_tmp111*weak_hess_tmp44;
    const s_t weak_hess_tmp223 = weak_hess_tmp110*weak_hess_tmp62;
    const s_t weak_hess_tmp224 = weak_hess_tmp17*weak_hess_tmp97;
    const s_t weak_hess_tmp225 = s_t(2)*weak_hess_tmp57;
    const s_t weak_hess_tmp226 = weak_hess_tmp110*weak_hess_tmp44;
    const s_t weak_hess_tmp227 = gu6*weak_hess_tmp89;
    const s_t weak_hess_tmp228 = weak_hess_tmp102*weak_hess_tmp7;
    const s_t weak_hess_tmp229 = weak_hess_tmp121*weak_hess_tmp7;
    const s_t weak_hess_tmp230 = weak_hess_tmp112*weak_hess_tmp7;
    const s_t weak_hess_tmp231 = weak_hess_tmp117*weak_hess_tmp9;
    const s_t weak_hess_tmp232 = weak_hess_tmp13*weak_hess_tmp231;
    const s_t weak_hess_tmp233 = weak_hess_tmp120*weak_hess_tmp17;
    const s_t weak_hess_tmp234 = weak_hess_tmp120*weak_hess_tmp34;
    const s_t weak_hess_tmp235 = weak_hess_tmp126*weak_hess_tmp44;
    const s_t weak_hess_tmp236 = weak_hess_tmp125*weak_hess_tmp62;
    const s_t weak_hess_tmp237 = weak_hess_tmp17*weak_hess_tmp96;
    const s_t weak_hess_tmp238 = weak_hess_tmp125*weak_hess_tmp44;
    const s_t weak_hess_tmp239 = weak_hess_tmp131*weak_hess_tmp9;
    const s_t weak_hess_tmp240 = weak_hess_tmp13*weak_hess_tmp239;
    const s_t weak_hess_tmp241 = weak_hess_tmp134*weak_hess_tmp17;
    const s_t weak_hess_tmp242 = weak_hess_tmp134*weak_hess_tmp34;
    const s_t weak_hess_tmp243 = weak_hess_tmp198 + weak_hess_tmp36;
    const s_t weak_hess_tmp244 = weak_hess_tmp200 + weak_hess_tmp40;
    const s_t weak_hess_tmp245 = weak_hess_tmp138*weak_hess_tmp44;
    const s_t weak_hess_tmp246 = weak_hess_tmp138*weak_hess_tmp62;
    const s_t weak_hess_tmp247 = weak_hess_tmp133*weak_hess_tmp17;
    const s_t weak_hess_tmp248 = weak_hess_tmp141*weak_hess_tmp44;
    const s_t weak_hess_tmp249 = weak_hess_tmp144*weak_hess_tmp9;
    const s_t weak_hess_tmp250 = weak_hess_tmp13*weak_hess_tmp249;
    const s_t weak_hess_tmp251 = weak_hess_tmp147*weak_hess_tmp17;
    const s_t weak_hess_tmp252 = weak_hess_tmp147*weak_hess_tmp34;
    const s_t weak_hess_tmp253 = weak_hess_tmp169 + weak_hess_tmp220;
    const s_t weak_hess_tmp254 = weak_hess_tmp150*weak_hess_tmp44;
    const s_t weak_hess_tmp255 = weak_hess_tmp150*weak_hess_tmp62;
    const s_t weak_hess_tmp256 = weak_hess_tmp161*weak_hess_tmp44;
    D[0] = c1*(weak_hess_tmp16 + s_t(4)*weak_hess_tmp20*weak_hess_tmp7 + weak_hess_tmp35*(((s_t(5) / s_t(3)))*weak_hess_tmp0 - (s_t(5) / s_t(3))*weak_hess_tmp18)) + c2*(weak_hess_tmp42*(weak_hess_tmp38 + weak_hess_tmp41) + s_t(2)*weak_hess_tmp43*weak_hess_tmp56 + weak_hess_tmp63*(((s_t(7) / s_t(3)))*weak_hess_tmp0 - (s_t(7) / s_t(3))*weak_hess_tmp18)) + weak_hess_tmp15*weak_hess_tmp3 + pow_2(weak_hess_tmp4)*weak_hess_tmp9;
    D[1] = c1*(weak_hess_tmp20*weak_hess_tmp49 + weak_hess_tmp35*weak_hess_tmp69 + weak_hess_tmp67*weak_hess_tmp68) + c2*(weak_hess_tmp42*(-weak_hess_tmp70 - weak_hess_tmp71) + weak_hess_tmp56*weak_hess_tmp72 + weak_hess_tmp63*weak_hess_tmp75 + weak_hess_tmp73*weak_hess_tmp74) + weak_hess_tmp14*weak_hess_tmp66 + weak_hess_tmp15*weak_hess_tmp65;
    D[2] = c1*(weak_hess_tmp20*weak_hess_tmp54 + weak_hess_tmp35*weak_hess_tmp80 + weak_hess_tmp68*weak_hess_tmp79) + c2*(weak_hess_tmp42*(-weak_hess_tmp81 - weak_hess_tmp82) + weak_hess_tmp56*weak_hess_tmp83 + weak_hess_tmp63*weak_hess_tmp85 + weak_hess_tmp74*weak_hess_tmp84) + weak_hess_tmp14*weak_hess_tmp77 + weak_hess_tmp15*weak_hess_tmp78;
    D[3] = c1*(weak_hess_tmp20*weak_hess_tmp89 + weak_hess_tmp35*weak_hess_tmp92 + weak_hess_tmp68*weak_hess_tmp91) + c2*(weak_hess_tmp42*(-weak_hess_tmp93 - weak_hess_tmp94) + weak_hess_tmp56*weak_hess_tmp95 + weak_hess_tmp63*weak_hess_tmp99 + weak_hess_tmp74*weak_hess_tmp98) + weak_hess_tmp14*weak_hess_tmp88 + weak_hess_tmp15*weak_hess_tmp87;
    D[4] = c1*(weak_hess_tmp105*weak_hess_tmp68 - weak_hess_tmp108 + weak_hess_tmp109*weak_hess_tmp35 + weak_hess_tmp20*weak_hess_tmp97) + c2*(weak_hess_tmp110*weak_hess_tmp56 + weak_hess_tmp111*weak_hess_tmp74 - weak_hess_tmp113 + weak_hess_tmp114*weak_hess_tmp63 + weak_hess_tmp42*(s_t(4)*weak_hess_tmp1*weak_hess_tmp7 - s_t(2)*weak_hess_tmp5)) + weak_hess_tmp100*weak_hess_tmp15 + weak_hess_tmp101*weak_hess_tmp14 + weak_hess_tmp103;
    D[5] = c1*(weak_hess_tmp120*weak_hess_tmp68 + weak_hess_tmp122 + weak_hess_tmp123*weak_hess_tmp35 + weak_hess_tmp20*weak_hess_tmp96) + c2*(weak_hess_tmp125*weak_hess_tmp56 + weak_hess_tmp126*weak_hess_tmp74 + weak_hess_tmp127 + weak_hess_tmp128*weak_hess_tmp63 + weak_hess_tmp42*(s_t(4)*gu5*weak_hess_tmp7 - s_t(2)*weak_hess_tmp124)) + weak_hess_tmp117*weak_hess_tmp14 - weak_hess_tmp118 + weak_hess_tmp119*weak_hess_tmp15;
    D[6] = c1*(weak_hess_tmp133*weak_hess_tmp20 + weak_hess_tmp134*weak_hess_tmp68 + weak_hess_tmp135*weak_hess_tmp35) + c2*(weak_hess_tmp138*weak_hess_tmp56 + weak_hess_tmp141*weak_hess_tmp74 + weak_hess_tmp142*weak_hess_tmp63 + weak_hess_tmp42*(-weak_hess_tmp136 - weak_hess_tmp137)) + weak_hess_tmp131*weak_hess_tmp14 + weak_hess_tmp132*weak_hess_tmp15;
    D[7] = c1*(weak_hess_tmp139*weak_hess_tmp20 + weak_hess_tmp147*weak_hess_tmp68 + weak_hess_tmp148 + weak_hess_tmp149*weak_hess_tmp35) + c2*(weak_hess_tmp150*weak_hess_tmp56 + weak_hess_tmp151*weak_hess_tmp74 + weak_hess_tmp152 + weak_hess_tmp153*weak_hess_tmp63 + weak_hess_tmp42*(s_t(4)*gu7*weak_hess_tmp7 - s_t(2)*weak_hess_tmp115)) + weak_hess_tmp14*weak_hess_tmp144 - weak_hess_tmp145 + weak_hess_tmp146*weak_hess_tmp15;
    D[8] = c1*(weak_hess_tmp140*weak_hess_tmp20 + weak_hess_tmp158*weak_hess_tmp68 - weak_hess_tmp159 + weak_hess_tmp160*weak_hess_tmp35) + c2*(weak_hess_tmp161*weak_hess_tmp56 + weak_hess_tmp162*weak_hess_tmp74 - weak_hess_tmp163 + weak_hess_tmp164*weak_hess_tmp63 + weak_hess_tmp42*(s_t(4)*weak_hess_tmp2*weak_hess_tmp7 - s_t(2)*weak_hess_tmp6)) + weak_hess_tmp14*weak_hess_tmp155 + weak_hess_tmp15*weak_hess_tmp154 + weak_hess_tmp156;
    D[10] = c1*(s_t(4)*gu1*weak_hess_tmp167 + weak_hess_tmp16 + weak_hess_tmp168*weak_hess_tmp69) + c2*(s_t(2)*weak_hess_tmp172*weak_hess_tmp72 + weak_hess_tmp173*weak_hess_tmp75 + weak_hess_tmp42*(weak_hess_tmp171 + weak_hess_tmp38)) + weak_hess_tmp166*weak_hess_tmp65 + pow_2(weak_hess_tmp66)*weak_hess_tmp9;
    D[11] = c1*(weak_hess_tmp167*weak_hess_tmp54 + weak_hess_tmp168*weak_hess_tmp80 + weak_hess_tmp174*weak_hess_tmp79) + c2*(weak_hess_tmp172*weak_hess_tmp83 + weak_hess_tmp173*weak_hess_tmp85 + weak_hess_tmp177*weak_hess_tmp84 + weak_hess_tmp42*(-weak_hess_tmp175 - weak_hess_tmp176)) + weak_hess_tmp165*weak_hess_tmp77 + weak_hess_tmp166*weak_hess_tmp78;
    D[12] = c1*(weak_hess_tmp108 + weak_hess_tmp167*weak_hess_tmp89 + weak_hess_tmp168*weak_hess_tmp92 + weak_hess_tmp174*weak_hess_tmp91) + c2*(weak_hess_tmp113 + weak_hess_tmp172*weak_hess_tmp95 + weak_hess_tmp173*weak_hess_tmp99 + weak_hess_tmp177*weak_hess_tmp98 + weak_hess_tmp42*(-s_t(2)*weak_hess_tmp157 + s_t(4)*weak_hess_tmp5)) - weak_hess_tmp103 + weak_hess_tmp165*weak_hess_tmp88 + weak_hess_tmp166*weak_hess_tmp87;
    D[13] = c1*(weak_hess_tmp105*weak_hess_tmp174 + weak_hess_tmp109*weak_hess_tmp168 + weak_hess_tmp167*weak_hess_tmp97) + c2*(weak_hess_tmp110*weak_hess_tmp172 + weak_hess_tmp111*weak_hess_tmp177 + weak_hess_tmp114*weak_hess_tmp173 + weak_hess_tmp42*(-weak_hess_tmp178 - weak_hess_tmp93)) + weak_hess_tmp100*weak_hess_tmp166 + weak_hess_tmp101*weak_hess_tmp165;
    D[14] = c1*(weak_hess_tmp120*weak_hess_tmp174 + weak_hess_tmp123*weak_hess_tmp168 + weak_hess_tmp167*weak_hess_tmp96 - weak_hess_tmp180) + c2*(weak_hess_tmp125*weak_hess_tmp172 + weak_hess_tmp126*weak_hess_tmp177 + weak_hess_tmp128*weak_hess_tmp173 - weak_hess_tmp181 + weak_hess_tmp42*(s_t(4)*weak_hess_tmp129 - s_t(2)*weak_hess_tmp130)) + weak_hess_tmp117*weak_hess_tmp165 + weak_hess_tmp119*weak_hess_tmp166 + weak_hess_tmp179;
    D[15] = c1*(weak_hess_tmp133*weak_hess_tmp167 + weak_hess_tmp134*weak_hess_tmp174 + weak_hess_tmp135*weak_hess_tmp168 - weak_hess_tmp148) + c2*(weak_hess_tmp138*weak_hess_tmp172 + weak_hess_tmp141*weak_hess_tmp177 + weak_hess_tmp142*weak_hess_tmp173 - weak_hess_tmp152 + weak_hess_tmp42*(s_t(4)*weak_hess_tmp115 - s_t(2)*weak_hess_tmp116)) + weak_hess_tmp131*weak_hess_tmp165 + weak_hess_tmp132*weak_hess_tmp166 + weak_hess_tmp145;
    D[16] = c1*(weak_hess_tmp139*weak_hess_tmp167 + weak_hess_tmp147*weak_hess_tmp174 + weak_hess_tmp149*weak_hess_tmp168) + c2*(weak_hess_tmp150*weak_hess_tmp172 + weak_hess_tmp151*weak_hess_tmp177 + weak_hess_tmp153*weak_hess_tmp173 + weak_hess_tmp42*(-weak_hess_tmp137 - weak_hess_tmp182)) + weak_hess_tmp144*weak_hess_tmp165 + weak_hess_tmp146*weak_hess_tmp166;
    D[17] = c1*(weak_hess_tmp140*weak_hess_tmp167 + weak_hess_tmp158*weak_hess_tmp174 + weak_hess_tmp160*weak_hess_tmp168 + weak_hess_tmp184) + c2*(weak_hess_tmp161*weak_hess_tmp172 + weak_hess_tmp162*weak_hess_tmp177 + weak_hess_tmp164*weak_hess_tmp173 + weak_hess_tmp185 + weak_hess_tmp42*(s_t(4)*weak_hess_tmp86 - s_t(2)*weak_hess_tmp90)) + weak_hess_tmp154*weak_hess_tmp166 + weak_hess_tmp155*weak_hess_tmp165 - weak_hess_tmp183;
    D[20] = c1*(s_t(4)*gu2*weak_hess_tmp188 + weak_hess_tmp16 + weak_hess_tmp189*weak_hess_tmp80) + c2*(s_t(2)*weak_hess_tmp190*weak_hess_tmp83 + weak_hess_tmp191*weak_hess_tmp85 + weak_hess_tmp42*(weak_hess_tmp171 + weak_hess_tmp41)) + weak_hess_tmp187*weak_hess_tmp78 + pow_2(weak_hess_tmp77)*weak_hess_tmp9;
    D[21] = c1*(-weak_hess_tmp122 + weak_hess_tmp188*weak_hess_tmp89 + weak_hess_tmp189*weak_hess_tmp92 + weak_hess_tmp192*weak_hess_tmp91) + c2*(-weak_hess_tmp127 + weak_hess_tmp190*weak_hess_tmp95 + weak_hess_tmp191*weak_hess_tmp99 + weak_hess_tmp193*weak_hess_tmp98 + weak_hess_tmp42*(s_t(4)*weak_hess_tmp124 - s_t(2)*weak_hess_tmp143)) + weak_hess_tmp118 + weak_hess_tmp186*weak_hess_tmp88 + weak_hess_tmp187*weak_hess_tmp87;
    D[22] = c1*(weak_hess_tmp105*weak_hess_tmp192 + weak_hess_tmp109*weak_hess_tmp189 + weak_hess_tmp180 + weak_hess_tmp188*weak_hess_tmp97) + c2*(weak_hess_tmp110*weak_hess_tmp190 + weak_hess_tmp111*weak_hess_tmp193 + weak_hess_tmp114*weak_hess_tmp191 + weak_hess_tmp181 + weak_hess_tmp42*(s_t(4)*gu2*weak_hess_tmp1 - s_t(2)*weak_hess_tmp129)) + weak_hess_tmp100*weak_hess_tmp187 + weak_hess_tmp101*weak_hess_tmp186 - weak_hess_tmp179;
    D[23] = c1*(weak_hess_tmp120*weak_hess_tmp192 + weak_hess_tmp123*weak_hess_tmp189 + weak_hess_tmp188*weak_hess_tmp96) + c2*(weak_hess_tmp125*weak_hess_tmp190 + weak_hess_tmp126*weak_hess_tmp193 + weak_hess_tmp128*weak_hess_tmp191 + weak_hess_tmp42*(-weak_hess_tmp178 - weak_hess_tmp94)) + weak_hess_tmp117*weak_hess_tmp186 + weak_hess_tmp119*weak_hess_tmp187;
    D[24] = c1*(weak_hess_tmp133*weak_hess_tmp188 + weak_hess_tmp134*weak_hess_tmp192 + weak_hess_tmp135*weak_hess_tmp189 + weak_hess_tmp159) + c2*(weak_hess_tmp138*weak_hess_tmp190 + weak_hess_tmp141*weak_hess_tmp193 + weak_hess_tmp142*weak_hess_tmp191 + weak_hess_tmp163 + weak_hess_tmp42*(-s_t(2)*weak_hess_tmp104 + s_t(4)*weak_hess_tmp6)) + weak_hess_tmp131*weak_hess_tmp186 + weak_hess_tmp132*weak_hess_tmp187 - weak_hess_tmp156;
    D[25] = c1*(weak_hess_tmp139*weak_hess_tmp188 + weak_hess_tmp147*weak_hess_tmp192 + weak_hess_tmp149*weak_hess_tmp189 - weak_hess_tmp184) + c2*(weak_hess_tmp150*weak_hess_tmp190 + weak_hess_tmp151*weak_hess_tmp193 + weak_hess_tmp153*weak_hess_tmp191 - weak_hess_tmp185 + weak_hess_tmp42*(s_t(4)*gu2*gu7 - s_t(2)*weak_hess_tmp86)) + weak_hess_tmp144*weak_hess_tmp186 + weak_hess_tmp146*weak_hess_tmp187 + weak_hess_tmp183;
    D[26] = c1*(weak_hess_tmp140*weak_hess_tmp188 + weak_hess_tmp158*weak_hess_tmp192 + weak_hess_tmp160*weak_hess_tmp189) + c2*(weak_hess_tmp161*weak_hess_tmp190 + weak_hess_tmp162*weak_hess_tmp193 + weak_hess_tmp164*weak_hess_tmp191 + weak_hess_tmp42*(-weak_hess_tmp136 - weak_hess_tmp182)) + weak_hess_tmp154*weak_hess_tmp187 + weak_hess_tmp155*weak_hess_tmp186;
    D[30] = c1*(s_t(4)*gu3*weak_hess_tmp196 + weak_hess_tmp16 + weak_hess_tmp197*weak_hess_tmp92) + c2*(s_t(2)*weak_hess_tmp202*weak_hess_tmp98 + weak_hess_tmp203*weak_hess_tmp99 + weak_hess_tmp42*(weak_hess_tmp199 + weak_hess_tmp201)) + weak_hess_tmp195*weak_hess_tmp87 + pow_2(weak_hess_tmp88)*weak_hess_tmp9;
    D[31] = c1*(weak_hess_tmp105*weak_hess_tmp204 + weak_hess_tmp109*weak_hess_tmp197 + weak_hess_tmp196*weak_hess_tmp97) + c2*(weak_hess_tmp110*weak_hess_tmp206 + weak_hess_tmp111*weak_hess_tmp202 + weak_hess_tmp114*weak_hess_tmp203 + weak_hess_tmp42*(-weak_hess_tmp205 - weak_hess_tmp70)) + weak_hess_tmp100*weak_hess_tmp195 + weak_hess_tmp101*weak_hess_tmp194;
    D[32] = c1*(weak_hess_tmp120*weak_hess_tmp204 + weak_hess_tmp123*weak_hess_tmp197 + weak_hess_tmp196*weak_hess_tmp96) + c2*(weak_hess_tmp125*weak_hess_tmp206 + weak_hess_tmp126*weak_hess_tmp202 + weak_hess_tmp128*weak_hess_tmp203 + weak_hess_tmp42*(-weak_hess_tmp207 - weak_hess_tmp82)) + weak_hess_tmp117*weak_hess_tmp194 + weak_hess_tmp119*weak_hess_tmp195;
    D[33] = c1*(weak_hess_tmp133*weak_hess_tmp196 + weak_hess_tmp134*weak_hess_tmp204 + weak_hess_tmp135*weak_hess_tmp197) + c2*(weak_hess_tmp138*weak_hess_tmp206 + weak_hess_tmp141*weak_hess_tmp202 + weak_hess_tmp142*weak_hess_tmp203 + weak_hess_tmp42*(-weak_hess_tmp208 - weak_hess_tmp209)) + weak_hess_tmp131*weak_hess_tmp194 + weak_hess_tmp132*weak_hess_tmp195;
    D[34] = c1*(weak_hess_tmp139*weak_hess_tmp196 + weak_hess_tmp147*weak_hess_tmp204 + weak_hess_tmp149*weak_hess_tmp197 - weak_hess_tmp211) + c2*(weak_hess_tmp150*weak_hess_tmp206 + weak_hess_tmp151*weak_hess_tmp202 + weak_hess_tmp153*weak_hess_tmp203 - weak_hess_tmp212 + weak_hess_tmp42*(s_t(4)*weak_hess_tmp12 - s_t(2)*weak_hess_tmp76)) + weak_hess_tmp144*weak_hess_tmp194 + weak_hess_tmp146*weak_hess_tmp195 + weak_hess_tmp210;
    D[35] = c1*(weak_hess_tmp140*weak_hess_tmp196 + weak_hess_tmp158*weak_hess_tmp204 + weak_hess_tmp160*weak_hess_tmp197 + weak_hess_tmp214) + c2*(weak_hess_tmp161*weak_hess_tmp206 + weak_hess_tmp162*weak_hess_tmp202 + weak_hess_tmp164*weak_hess_tmp203 + weak_hess_tmp215 + weak_hess_tmp42*(-s_t(2)*weak_hess_tmp11 + s_t(4)*weak_hess_tmp64)) + weak_hess_tmp154*weak_hess_tmp195 + weak_hess_tmp155*weak_hess_tmp194 - weak_hess_tmp213;
    D[40] = c1*(s_t(4)*weak_hess_tmp1*weak_hess_tmp218 + weak_hess_tmp109*weak_hess_tmp219 + weak_hess_tmp16) + c2*(s_t(2)*weak_hess_tmp110*weak_hess_tmp222 + weak_hess_tmp114*weak_hess_tmp223 + weak_hess_tmp42*(weak_hess_tmp199 + weak_hess_tmp221)) + weak_hess_tmp100*weak_hess_tmp217 + pow_2(weak_hess_tmp101)*weak_hess_tmp9;
    D[41] = c1*(weak_hess_tmp120*weak_hess_tmp224 + weak_hess_tmp123*weak_hess_tmp219 + weak_hess_tmp218*weak_hess_tmp96) + c2*(weak_hess_tmp125*weak_hess_tmp222 + weak_hess_tmp126*weak_hess_tmp226 + weak_hess_tmp128*weak_hess_tmp223 + weak_hess_tmp42*(-weak_hess_tmp176 - weak_hess_tmp225)) + weak_hess_tmp117*weak_hess_tmp216 + weak_hess_tmp119*weak_hess_tmp217;
    D[42] = c1*(weak_hess_tmp133*weak_hess_tmp218 + weak_hess_tmp134*weak_hess_tmp224 + weak_hess_tmp135*weak_hess_tmp219 + weak_hess_tmp211) + c2*(weak_hess_tmp138*weak_hess_tmp222 + weak_hess_tmp141*weak_hess_tmp226 + weak_hess_tmp142*weak_hess_tmp223 + weak_hess_tmp212 + weak_hess_tmp42*(s_t(4)*gu6*weak_hess_tmp1 - s_t(2)*weak_hess_tmp12)) + weak_hess_tmp131*weak_hess_tmp216 + weak_hess_tmp132*weak_hess_tmp217 - weak_hess_tmp210;
    D[43] = c1*(weak_hess_tmp139*weak_hess_tmp218 + weak_hess_tmp147*weak_hess_tmp224 + weak_hess_tmp149*weak_hess_tmp219) + c2*(weak_hess_tmp150*weak_hess_tmp222 + weak_hess_tmp151*weak_hess_tmp226 + weak_hess_tmp153*weak_hess_tmp223 + weak_hess_tmp42*(-weak_hess_tmp208 - weak_hess_tmp227)) + weak_hess_tmp144*weak_hess_tmp216 + weak_hess_tmp146*weak_hess_tmp217;
    D[44] = c1*(weak_hess_tmp140*weak_hess_tmp218 + weak_hess_tmp158*weak_hess_tmp224 + weak_hess_tmp160*weak_hess_tmp219 - weak_hess_tmp229) + c2*(weak_hess_tmp161*weak_hess_tmp222 + weak_hess_tmp162*weak_hess_tmp226 + weak_hess_tmp164*weak_hess_tmp223 - weak_hess_tmp230 + weak_hess_tmp42*(-s_t(2)*weak_hess_tmp0 + s_t(4)*weak_hess_tmp1*weak_hess_tmp2)) + weak_hess_tmp154*weak_hess_tmp217 + weak_hess_tmp155*weak_hess_tmp216 + weak_hess_tmp228;
    D[50] = c1*(s_t(4)*gu5*weak_hess_tmp233 + weak_hess_tmp123*weak_hess_tmp234 + weak_hess_tmp16) + c2*(s_t(2)*weak_hess_tmp125*weak_hess_tmp235 + weak_hess_tmp128*weak_hess_tmp236 + weak_hess_tmp42*(weak_hess_tmp201 + weak_hess_tmp221)) + pow_2(weak_hess_tmp117)*weak_hess_tmp9 + weak_hess_tmp119*weak_hess_tmp232;
    D[51] = c1*(weak_hess_tmp133*weak_hess_tmp233 + weak_hess_tmp134*weak_hess_tmp237 + weak_hess_tmp135*weak_hess_tmp234 - weak_hess_tmp214) + c2*(weak_hess_tmp138*weak_hess_tmp235 + weak_hess_tmp141*weak_hess_tmp238 + weak_hess_tmp142*weak_hess_tmp236 - weak_hess_tmp215 + weak_hess_tmp42*(s_t(4)*gu5*gu6 - s_t(2)*weak_hess_tmp64)) + weak_hess_tmp131*weak_hess_tmp231 + weak_hess_tmp132*weak_hess_tmp232 + weak_hess_tmp213;
    D[52] = c1*(weak_hess_tmp139*weak_hess_tmp233 + weak_hess_tmp147*weak_hess_tmp237 + weak_hess_tmp149*weak_hess_tmp234 + weak_hess_tmp229) + c2*(weak_hess_tmp150*weak_hess_tmp235 + weak_hess_tmp151*weak_hess_tmp238 + weak_hess_tmp153*weak_hess_tmp236 + weak_hess_tmp230 + weak_hess_tmp42*(s_t(4)*weak_hess_tmp0 - s_t(2)*weak_hess_tmp18)) + weak_hess_tmp144*weak_hess_tmp231 + weak_hess_tmp146*weak_hess_tmp232 - weak_hess_tmp228;
    D[53] = c1*(weak_hess_tmp140*weak_hess_tmp233 + weak_hess_tmp158*weak_hess_tmp237 + weak_hess_tmp160*weak_hess_tmp234) + c2*(weak_hess_tmp161*weak_hess_tmp235 + weak_hess_tmp162*weak_hess_tmp238 + weak_hess_tmp164*weak_hess_tmp236 + weak_hess_tmp42*(-weak_hess_tmp209 - weak_hess_tmp227)) + weak_hess_tmp154*weak_hess_tmp232 + weak_hess_tmp155*weak_hess_tmp231;
    D[60] = c1*(s_t(4)*gu6*weak_hess_tmp241 + weak_hess_tmp135*weak_hess_tmp242 + weak_hess_tmp16) + c2*(s_t(2)*weak_hess_tmp141*weak_hess_tmp245 + weak_hess_tmp142*weak_hess_tmp246 + weak_hess_tmp42*(weak_hess_tmp243 + weak_hess_tmp244)) + pow_2(weak_hess_tmp131)*weak_hess_tmp9 + weak_hess_tmp132*weak_hess_tmp240;
    D[61] = c1*(weak_hess_tmp139*weak_hess_tmp241 + weak_hess_tmp147*weak_hess_tmp247 + weak_hess_tmp149*weak_hess_tmp242) + c2*(weak_hess_tmp150*weak_hess_tmp248 + weak_hess_tmp151*weak_hess_tmp245 + weak_hess_tmp153*weak_hess_tmp246 + weak_hess_tmp42*(-weak_hess_tmp205 - weak_hess_tmp71)) + weak_hess_tmp144*weak_hess_tmp239 + weak_hess_tmp146*weak_hess_tmp240;
    D[62] = c1*(weak_hess_tmp140*weak_hess_tmp241 + weak_hess_tmp158*weak_hess_tmp247 + weak_hess_tmp160*weak_hess_tmp242) + c2*(weak_hess_tmp161*weak_hess_tmp248 + weak_hess_tmp162*weak_hess_tmp245 + weak_hess_tmp164*weak_hess_tmp246 + weak_hess_tmp42*(-weak_hess_tmp207 - weak_hess_tmp81)) + weak_hess_tmp154*weak_hess_tmp240 + weak_hess_tmp155*weak_hess_tmp239;
    D[70] = c1*(s_t(4)*gu7*weak_hess_tmp251 + weak_hess_tmp149*weak_hess_tmp252 + weak_hess_tmp16) + c2*(s_t(2)*weak_hess_tmp151*weak_hess_tmp254 + weak_hess_tmp153*weak_hess_tmp255 + weak_hess_tmp42*(weak_hess_tmp243 + weak_hess_tmp253)) + pow_2(weak_hess_tmp144)*weak_hess_tmp9 + weak_hess_tmp146*weak_hess_tmp250;
    D[71] = c1*(weak_hess_tmp139*weak_hess_tmp158*weak_hess_tmp17 + weak_hess_tmp140*weak_hess_tmp251 + weak_hess_tmp160*weak_hess_tmp252) + c2*(weak_hess_tmp151*weak_hess_tmp256 + weak_hess_tmp162*weak_hess_tmp254 + weak_hess_tmp164*weak_hess_tmp255 + weak_hess_tmp42*(-weak_hess_tmp175 - weak_hess_tmp225)) + weak_hess_tmp154*weak_hess_tmp250 + weak_hess_tmp155*weak_hess_tmp249;
    D[80] = c1*(s_t(4)*weak_hess_tmp106*weak_hess_tmp158 + weak_hess_tmp158*weak_hess_tmp160*weak_hess_tmp34 + weak_hess_tmp16) + c2*(weak_hess_tmp161*weak_hess_tmp164*weak_hess_tmp62 + s_t(2)*weak_hess_tmp162*weak_hess_tmp256 + weak_hess_tmp42*(weak_hess_tmp244 + weak_hess_tmp253)) + weak_hess_tmp13*weak_hess_tmp154*weak_hess_tmp155*weak_hess_tmp9 + pow_2(weak_hess_tmp155)*weak_hess_tmp9;
    D[9] = D[1];
    D[18] = D[2];
    D[27] = D[3];
    D[36] = D[4];
    D[45] = D[5];
    D[54] = D[6];
    D[63] = D[7];
    D[72] = D[8];
    D[19] = D[11];
    D[28] = D[12];
    D[37] = D[13];
    D[46] = D[14];
    D[55] = D[15];
    D[64] = D[16];
    D[73] = D[17];
    D[29] = D[21];
    D[38] = D[22];
    D[47] = D[23];
    D[56] = D[24];
    D[65] = D[25];
    D[74] = D[26];
    D[39] = D[31];
    D[48] = D[32];
    D[57] = D[33];
    D[66] = D[34];
    D[75] = D[35];
    D[49] = D[41];
    D[58] = D[42];
    D[67] = D[43];
    D[76] = D[44];
    D[59] = D[51];
    D[68] = D[52];
    D[77] = D[53];
    D[69] = D[61];
    D[78] = D[62];
    D[79] = D[71];
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
      s_t D[81], adj_q[ND * ND], weight;
      point_tangent(q, D, adj_q, weight);
      s_t g_ref[ND * NS];
      tensor_point_reference_gradients_scalar<s_t, NQ, NS, ND>(shape_1d, grad_1d, q, g_ref);
      point_test_symmetric_scalar<s_t, NS, ND, NC>(g_ref, adj_q, weight, D, element_matrix);
    }
    mirror_upper_triangle_scalar<s_t, NDOFS>(element_matrix);
  } else {
    static constexpr int NP = NQ1 * NQ1;
    s_t slab_tangent[NP * 81];
    s_t flux[NC * ND * NP];
    for (int q_last = 0; q_last < NQ1; ++q_last) {
      for (int p = 0; p < NP; ++p) {
        s_t D[81], adj_q[ND * ND], weight;
        point_tangent(p + NP * q_last, D, adj_q, weight);
        const s_t pullback_weight = weight;
        const s_t pullback_0_0_0_0 = adj_q[0] * D[0] + adj_q[1] * D[9] + adj_q[2] * D[18];
        const s_t pullback_0_0_0_1 = adj_q[0] * D[1] + adj_q[1] * D[10] + adj_q[2] * D[19];
        const s_t pullback_0_0_0_2 = adj_q[0] * D[2] + adj_q[1] * D[11] + adj_q[2] * D[20];
        const s_t pullback_0_0_1_0 = adj_q[3] * D[0] + adj_q[4] * D[9] + adj_q[5] * D[18];
        const s_t pullback_0_0_1_1 = adj_q[3] * D[1] + adj_q[4] * D[10] + adj_q[5] * D[19];
        const s_t pullback_0_0_1_2 = adj_q[3] * D[2] + adj_q[4] * D[11] + adj_q[5] * D[20];
        const s_t pullback_0_0_2_0 = adj_q[6] * D[0] + adj_q[7] * D[9] + adj_q[8] * D[18];
        const s_t pullback_0_0_2_1 = adj_q[6] * D[1] + adj_q[7] * D[10] + adj_q[8] * D[19];
        const s_t pullback_0_0_2_2 = adj_q[6] * D[2] + adj_q[7] * D[11] + adj_q[8] * D[20];
        slab_tangent[p * 81 + 0] = pullback_weight * (pullback_0_0_0_0 * adj_q[0] + pullback_0_0_0_1 * adj_q[1] + pullback_0_0_0_2 * adj_q[2]);
        slab_tangent[p * 81 + 1] = pullback_weight * (pullback_0_0_0_0 * adj_q[3] + pullback_0_0_0_1 * adj_q[4] + pullback_0_0_0_2 * adj_q[5]);
        slab_tangent[p * 81 + 2] = pullback_weight * (pullback_0_0_0_0 * adj_q[6] + pullback_0_0_0_1 * adj_q[7] + pullback_0_0_0_2 * adj_q[8]);
        slab_tangent[p * 81 + 9] = pullback_weight * (pullback_0_0_1_0 * adj_q[0] + pullback_0_0_1_1 * adj_q[1] + pullback_0_0_1_2 * adj_q[2]);
        slab_tangent[p * 81 + 10] = pullback_weight * (pullback_0_0_1_0 * adj_q[3] + pullback_0_0_1_1 * adj_q[4] + pullback_0_0_1_2 * adj_q[5]);
        slab_tangent[p * 81 + 11] = pullback_weight * (pullback_0_0_1_0 * adj_q[6] + pullback_0_0_1_1 * adj_q[7] + pullback_0_0_1_2 * adj_q[8]);
        slab_tangent[p * 81 + 18] = pullback_weight * (pullback_0_0_2_0 * adj_q[0] + pullback_0_0_2_1 * adj_q[1] + pullback_0_0_2_2 * adj_q[2]);
        slab_tangent[p * 81 + 19] = pullback_weight * (pullback_0_0_2_0 * adj_q[3] + pullback_0_0_2_1 * adj_q[4] + pullback_0_0_2_2 * adj_q[5]);
        slab_tangent[p * 81 + 20] = pullback_weight * (pullback_0_0_2_0 * adj_q[6] + pullback_0_0_2_1 * adj_q[7] + pullback_0_0_2_2 * adj_q[8]);
        const s_t pullback_0_1_0_0 = adj_q[0] * D[3] + adj_q[1] * D[12] + adj_q[2] * D[21];
        const s_t pullback_0_1_0_1 = adj_q[0] * D[4] + adj_q[1] * D[13] + adj_q[2] * D[22];
        const s_t pullback_0_1_0_2 = adj_q[0] * D[5] + adj_q[1] * D[14] + adj_q[2] * D[23];
        const s_t pullback_0_1_1_0 = adj_q[3] * D[3] + adj_q[4] * D[12] + adj_q[5] * D[21];
        const s_t pullback_0_1_1_1 = adj_q[3] * D[4] + adj_q[4] * D[13] + adj_q[5] * D[22];
        const s_t pullback_0_1_1_2 = adj_q[3] * D[5] + adj_q[4] * D[14] + adj_q[5] * D[23];
        const s_t pullback_0_1_2_0 = adj_q[6] * D[3] + adj_q[7] * D[12] + adj_q[8] * D[21];
        const s_t pullback_0_1_2_1 = adj_q[6] * D[4] + adj_q[7] * D[13] + adj_q[8] * D[22];
        const s_t pullback_0_1_2_2 = adj_q[6] * D[5] + adj_q[7] * D[14] + adj_q[8] * D[23];
        slab_tangent[p * 81 + 3] = pullback_weight * (pullback_0_1_0_0 * adj_q[0] + pullback_0_1_0_1 * adj_q[1] + pullback_0_1_0_2 * adj_q[2]);
        slab_tangent[p * 81 + 4] = pullback_weight * (pullback_0_1_0_0 * adj_q[3] + pullback_0_1_0_1 * adj_q[4] + pullback_0_1_0_2 * adj_q[5]);
        slab_tangent[p * 81 + 5] = pullback_weight * (pullback_0_1_0_0 * adj_q[6] + pullback_0_1_0_1 * adj_q[7] + pullback_0_1_0_2 * adj_q[8]);
        slab_tangent[p * 81 + 12] = pullback_weight * (pullback_0_1_1_0 * adj_q[0] + pullback_0_1_1_1 * adj_q[1] + pullback_0_1_1_2 * adj_q[2]);
        slab_tangent[p * 81 + 13] = pullback_weight * (pullback_0_1_1_0 * adj_q[3] + pullback_0_1_1_1 * adj_q[4] + pullback_0_1_1_2 * adj_q[5]);
        slab_tangent[p * 81 + 14] = pullback_weight * (pullback_0_1_1_0 * adj_q[6] + pullback_0_1_1_1 * adj_q[7] + pullback_0_1_1_2 * adj_q[8]);
        slab_tangent[p * 81 + 21] = pullback_weight * (pullback_0_1_2_0 * adj_q[0] + pullback_0_1_2_1 * adj_q[1] + pullback_0_1_2_2 * adj_q[2]);
        slab_tangent[p * 81 + 22] = pullback_weight * (pullback_0_1_2_0 * adj_q[3] + pullback_0_1_2_1 * adj_q[4] + pullback_0_1_2_2 * adj_q[5]);
        slab_tangent[p * 81 + 23] = pullback_weight * (pullback_0_1_2_0 * adj_q[6] + pullback_0_1_2_1 * adj_q[7] + pullback_0_1_2_2 * adj_q[8]);
        const s_t pullback_0_2_0_0 = adj_q[0] * D[6] + adj_q[1] * D[15] + adj_q[2] * D[24];
        const s_t pullback_0_2_0_1 = adj_q[0] * D[7] + adj_q[1] * D[16] + adj_q[2] * D[25];
        const s_t pullback_0_2_0_2 = adj_q[0] * D[8] + adj_q[1] * D[17] + adj_q[2] * D[26];
        const s_t pullback_0_2_1_0 = adj_q[3] * D[6] + adj_q[4] * D[15] + adj_q[5] * D[24];
        const s_t pullback_0_2_1_1 = adj_q[3] * D[7] + adj_q[4] * D[16] + adj_q[5] * D[25];
        const s_t pullback_0_2_1_2 = adj_q[3] * D[8] + adj_q[4] * D[17] + adj_q[5] * D[26];
        const s_t pullback_0_2_2_0 = adj_q[6] * D[6] + adj_q[7] * D[15] + adj_q[8] * D[24];
        const s_t pullback_0_2_2_1 = adj_q[6] * D[7] + adj_q[7] * D[16] + adj_q[8] * D[25];
        const s_t pullback_0_2_2_2 = adj_q[6] * D[8] + adj_q[7] * D[17] + adj_q[8] * D[26];
        slab_tangent[p * 81 + 6] = pullback_weight * (pullback_0_2_0_0 * adj_q[0] + pullback_0_2_0_1 * adj_q[1] + pullback_0_2_0_2 * adj_q[2]);
        slab_tangent[p * 81 + 7] = pullback_weight * (pullback_0_2_0_0 * adj_q[3] + pullback_0_2_0_1 * adj_q[4] + pullback_0_2_0_2 * adj_q[5]);
        slab_tangent[p * 81 + 8] = pullback_weight * (pullback_0_2_0_0 * adj_q[6] + pullback_0_2_0_1 * adj_q[7] + pullback_0_2_0_2 * adj_q[8]);
        slab_tangent[p * 81 + 15] = pullback_weight * (pullback_0_2_1_0 * adj_q[0] + pullback_0_2_1_1 * adj_q[1] + pullback_0_2_1_2 * adj_q[2]);
        slab_tangent[p * 81 + 16] = pullback_weight * (pullback_0_2_1_0 * adj_q[3] + pullback_0_2_1_1 * adj_q[4] + pullback_0_2_1_2 * adj_q[5]);
        slab_tangent[p * 81 + 17] = pullback_weight * (pullback_0_2_1_0 * adj_q[6] + pullback_0_2_1_1 * adj_q[7] + pullback_0_2_1_2 * adj_q[8]);
        slab_tangent[p * 81 + 24] = pullback_weight * (pullback_0_2_2_0 * adj_q[0] + pullback_0_2_2_1 * adj_q[1] + pullback_0_2_2_2 * adj_q[2]);
        slab_tangent[p * 81 + 25] = pullback_weight * (pullback_0_2_2_0 * adj_q[3] + pullback_0_2_2_1 * adj_q[4] + pullback_0_2_2_2 * adj_q[5]);
        slab_tangent[p * 81 + 26] = pullback_weight * (pullback_0_2_2_0 * adj_q[6] + pullback_0_2_2_1 * adj_q[7] + pullback_0_2_2_2 * adj_q[8]);
        const s_t pullback_1_0_0_0 = adj_q[0] * D[27] + adj_q[1] * D[36] + adj_q[2] * D[45];
        const s_t pullback_1_0_0_1 = adj_q[0] * D[28] + adj_q[1] * D[37] + adj_q[2] * D[46];
        const s_t pullback_1_0_0_2 = adj_q[0] * D[29] + adj_q[1] * D[38] + adj_q[2] * D[47];
        const s_t pullback_1_0_1_0 = adj_q[3] * D[27] + adj_q[4] * D[36] + adj_q[5] * D[45];
        const s_t pullback_1_0_1_1 = adj_q[3] * D[28] + adj_q[4] * D[37] + adj_q[5] * D[46];
        const s_t pullback_1_0_1_2 = adj_q[3] * D[29] + adj_q[4] * D[38] + adj_q[5] * D[47];
        const s_t pullback_1_0_2_0 = adj_q[6] * D[27] + adj_q[7] * D[36] + adj_q[8] * D[45];
        const s_t pullback_1_0_2_1 = adj_q[6] * D[28] + adj_q[7] * D[37] + adj_q[8] * D[46];
        const s_t pullback_1_0_2_2 = adj_q[6] * D[29] + adj_q[7] * D[38] + adj_q[8] * D[47];
        slab_tangent[p * 81 + 27] = pullback_weight * (pullback_1_0_0_0 * adj_q[0] + pullback_1_0_0_1 * adj_q[1] + pullback_1_0_0_2 * adj_q[2]);
        slab_tangent[p * 81 + 28] = pullback_weight * (pullback_1_0_0_0 * adj_q[3] + pullback_1_0_0_1 * adj_q[4] + pullback_1_0_0_2 * adj_q[5]);
        slab_tangent[p * 81 + 29] = pullback_weight * (pullback_1_0_0_0 * adj_q[6] + pullback_1_0_0_1 * adj_q[7] + pullback_1_0_0_2 * adj_q[8]);
        slab_tangent[p * 81 + 36] = pullback_weight * (pullback_1_0_1_0 * adj_q[0] + pullback_1_0_1_1 * adj_q[1] + pullback_1_0_1_2 * adj_q[2]);
        slab_tangent[p * 81 + 37] = pullback_weight * (pullback_1_0_1_0 * adj_q[3] + pullback_1_0_1_1 * adj_q[4] + pullback_1_0_1_2 * adj_q[5]);
        slab_tangent[p * 81 + 38] = pullback_weight * (pullback_1_0_1_0 * adj_q[6] + pullback_1_0_1_1 * adj_q[7] + pullback_1_0_1_2 * adj_q[8]);
        slab_tangent[p * 81 + 45] = pullback_weight * (pullback_1_0_2_0 * adj_q[0] + pullback_1_0_2_1 * adj_q[1] + pullback_1_0_2_2 * adj_q[2]);
        slab_tangent[p * 81 + 46] = pullback_weight * (pullback_1_0_2_0 * adj_q[3] + pullback_1_0_2_1 * adj_q[4] + pullback_1_0_2_2 * adj_q[5]);
        slab_tangent[p * 81 + 47] = pullback_weight * (pullback_1_0_2_0 * adj_q[6] + pullback_1_0_2_1 * adj_q[7] + pullback_1_0_2_2 * adj_q[8]);
        const s_t pullback_1_1_0_0 = adj_q[0] * D[30] + adj_q[1] * D[39] + adj_q[2] * D[48];
        const s_t pullback_1_1_0_1 = adj_q[0] * D[31] + adj_q[1] * D[40] + adj_q[2] * D[49];
        const s_t pullback_1_1_0_2 = adj_q[0] * D[32] + adj_q[1] * D[41] + adj_q[2] * D[50];
        const s_t pullback_1_1_1_0 = adj_q[3] * D[30] + adj_q[4] * D[39] + adj_q[5] * D[48];
        const s_t pullback_1_1_1_1 = adj_q[3] * D[31] + adj_q[4] * D[40] + adj_q[5] * D[49];
        const s_t pullback_1_1_1_2 = adj_q[3] * D[32] + adj_q[4] * D[41] + adj_q[5] * D[50];
        const s_t pullback_1_1_2_0 = adj_q[6] * D[30] + adj_q[7] * D[39] + adj_q[8] * D[48];
        const s_t pullback_1_1_2_1 = adj_q[6] * D[31] + adj_q[7] * D[40] + adj_q[8] * D[49];
        const s_t pullback_1_1_2_2 = adj_q[6] * D[32] + adj_q[7] * D[41] + adj_q[8] * D[50];
        slab_tangent[p * 81 + 30] = pullback_weight * (pullback_1_1_0_0 * adj_q[0] + pullback_1_1_0_1 * adj_q[1] + pullback_1_1_0_2 * adj_q[2]);
        slab_tangent[p * 81 + 31] = pullback_weight * (pullback_1_1_0_0 * adj_q[3] + pullback_1_1_0_1 * adj_q[4] + pullback_1_1_0_2 * adj_q[5]);
        slab_tangent[p * 81 + 32] = pullback_weight * (pullback_1_1_0_0 * adj_q[6] + pullback_1_1_0_1 * adj_q[7] + pullback_1_1_0_2 * adj_q[8]);
        slab_tangent[p * 81 + 39] = pullback_weight * (pullback_1_1_1_0 * adj_q[0] + pullback_1_1_1_1 * adj_q[1] + pullback_1_1_1_2 * adj_q[2]);
        slab_tangent[p * 81 + 40] = pullback_weight * (pullback_1_1_1_0 * adj_q[3] + pullback_1_1_1_1 * adj_q[4] + pullback_1_1_1_2 * adj_q[5]);
        slab_tangent[p * 81 + 41] = pullback_weight * (pullback_1_1_1_0 * adj_q[6] + pullback_1_1_1_1 * adj_q[7] + pullback_1_1_1_2 * adj_q[8]);
        slab_tangent[p * 81 + 48] = pullback_weight * (pullback_1_1_2_0 * adj_q[0] + pullback_1_1_2_1 * adj_q[1] + pullback_1_1_2_2 * adj_q[2]);
        slab_tangent[p * 81 + 49] = pullback_weight * (pullback_1_1_2_0 * adj_q[3] + pullback_1_1_2_1 * adj_q[4] + pullback_1_1_2_2 * adj_q[5]);
        slab_tangent[p * 81 + 50] = pullback_weight * (pullback_1_1_2_0 * adj_q[6] + pullback_1_1_2_1 * adj_q[7] + pullback_1_1_2_2 * adj_q[8]);
        const s_t pullback_1_2_0_0 = adj_q[0] * D[33] + adj_q[1] * D[42] + adj_q[2] * D[51];
        const s_t pullback_1_2_0_1 = adj_q[0] * D[34] + adj_q[1] * D[43] + adj_q[2] * D[52];
        const s_t pullback_1_2_0_2 = adj_q[0] * D[35] + adj_q[1] * D[44] + adj_q[2] * D[53];
        const s_t pullback_1_2_1_0 = adj_q[3] * D[33] + adj_q[4] * D[42] + adj_q[5] * D[51];
        const s_t pullback_1_2_1_1 = adj_q[3] * D[34] + adj_q[4] * D[43] + adj_q[5] * D[52];
        const s_t pullback_1_2_1_2 = adj_q[3] * D[35] + adj_q[4] * D[44] + adj_q[5] * D[53];
        const s_t pullback_1_2_2_0 = adj_q[6] * D[33] + adj_q[7] * D[42] + adj_q[8] * D[51];
        const s_t pullback_1_2_2_1 = adj_q[6] * D[34] + adj_q[7] * D[43] + adj_q[8] * D[52];
        const s_t pullback_1_2_2_2 = adj_q[6] * D[35] + adj_q[7] * D[44] + adj_q[8] * D[53];
        slab_tangent[p * 81 + 33] = pullback_weight * (pullback_1_2_0_0 * adj_q[0] + pullback_1_2_0_1 * adj_q[1] + pullback_1_2_0_2 * adj_q[2]);
        slab_tangent[p * 81 + 34] = pullback_weight * (pullback_1_2_0_0 * adj_q[3] + pullback_1_2_0_1 * adj_q[4] + pullback_1_2_0_2 * adj_q[5]);
        slab_tangent[p * 81 + 35] = pullback_weight * (pullback_1_2_0_0 * adj_q[6] + pullback_1_2_0_1 * adj_q[7] + pullback_1_2_0_2 * adj_q[8]);
        slab_tangent[p * 81 + 42] = pullback_weight * (pullback_1_2_1_0 * adj_q[0] + pullback_1_2_1_1 * adj_q[1] + pullback_1_2_1_2 * adj_q[2]);
        slab_tangent[p * 81 + 43] = pullback_weight * (pullback_1_2_1_0 * adj_q[3] + pullback_1_2_1_1 * adj_q[4] + pullback_1_2_1_2 * adj_q[5]);
        slab_tangent[p * 81 + 44] = pullback_weight * (pullback_1_2_1_0 * adj_q[6] + pullback_1_2_1_1 * adj_q[7] + pullback_1_2_1_2 * adj_q[8]);
        slab_tangent[p * 81 + 51] = pullback_weight * (pullback_1_2_2_0 * adj_q[0] + pullback_1_2_2_1 * adj_q[1] + pullback_1_2_2_2 * adj_q[2]);
        slab_tangent[p * 81 + 52] = pullback_weight * (pullback_1_2_2_0 * adj_q[3] + pullback_1_2_2_1 * adj_q[4] + pullback_1_2_2_2 * adj_q[5]);
        slab_tangent[p * 81 + 53] = pullback_weight * (pullback_1_2_2_0 * adj_q[6] + pullback_1_2_2_1 * adj_q[7] + pullback_1_2_2_2 * adj_q[8]);
        const s_t pullback_2_0_0_0 = adj_q[0] * D[54] + adj_q[1] * D[63] + adj_q[2] * D[72];
        const s_t pullback_2_0_0_1 = adj_q[0] * D[55] + adj_q[1] * D[64] + adj_q[2] * D[73];
        const s_t pullback_2_0_0_2 = adj_q[0] * D[56] + adj_q[1] * D[65] + adj_q[2] * D[74];
        const s_t pullback_2_0_1_0 = adj_q[3] * D[54] + adj_q[4] * D[63] + adj_q[5] * D[72];
        const s_t pullback_2_0_1_1 = adj_q[3] * D[55] + adj_q[4] * D[64] + adj_q[5] * D[73];
        const s_t pullback_2_0_1_2 = adj_q[3] * D[56] + adj_q[4] * D[65] + adj_q[5] * D[74];
        const s_t pullback_2_0_2_0 = adj_q[6] * D[54] + adj_q[7] * D[63] + adj_q[8] * D[72];
        const s_t pullback_2_0_2_1 = adj_q[6] * D[55] + adj_q[7] * D[64] + adj_q[8] * D[73];
        const s_t pullback_2_0_2_2 = adj_q[6] * D[56] + adj_q[7] * D[65] + adj_q[8] * D[74];
        slab_tangent[p * 81 + 54] = pullback_weight * (pullback_2_0_0_0 * adj_q[0] + pullback_2_0_0_1 * adj_q[1] + pullback_2_0_0_2 * adj_q[2]);
        slab_tangent[p * 81 + 55] = pullback_weight * (pullback_2_0_0_0 * adj_q[3] + pullback_2_0_0_1 * adj_q[4] + pullback_2_0_0_2 * adj_q[5]);
        slab_tangent[p * 81 + 56] = pullback_weight * (pullback_2_0_0_0 * adj_q[6] + pullback_2_0_0_1 * adj_q[7] + pullback_2_0_0_2 * adj_q[8]);
        slab_tangent[p * 81 + 63] = pullback_weight * (pullback_2_0_1_0 * adj_q[0] + pullback_2_0_1_1 * adj_q[1] + pullback_2_0_1_2 * adj_q[2]);
        slab_tangent[p * 81 + 64] = pullback_weight * (pullback_2_0_1_0 * adj_q[3] + pullback_2_0_1_1 * adj_q[4] + pullback_2_0_1_2 * adj_q[5]);
        slab_tangent[p * 81 + 65] = pullback_weight * (pullback_2_0_1_0 * adj_q[6] + pullback_2_0_1_1 * adj_q[7] + pullback_2_0_1_2 * adj_q[8]);
        slab_tangent[p * 81 + 72] = pullback_weight * (pullback_2_0_2_0 * adj_q[0] + pullback_2_0_2_1 * adj_q[1] + pullback_2_0_2_2 * adj_q[2]);
        slab_tangent[p * 81 + 73] = pullback_weight * (pullback_2_0_2_0 * adj_q[3] + pullback_2_0_2_1 * adj_q[4] + pullback_2_0_2_2 * adj_q[5]);
        slab_tangent[p * 81 + 74] = pullback_weight * (pullback_2_0_2_0 * adj_q[6] + pullback_2_0_2_1 * adj_q[7] + pullback_2_0_2_2 * adj_q[8]);
        const s_t pullback_2_1_0_0 = adj_q[0] * D[57] + adj_q[1] * D[66] + adj_q[2] * D[75];
        const s_t pullback_2_1_0_1 = adj_q[0] * D[58] + adj_q[1] * D[67] + adj_q[2] * D[76];
        const s_t pullback_2_1_0_2 = adj_q[0] * D[59] + adj_q[1] * D[68] + adj_q[2] * D[77];
        const s_t pullback_2_1_1_0 = adj_q[3] * D[57] + adj_q[4] * D[66] + adj_q[5] * D[75];
        const s_t pullback_2_1_1_1 = adj_q[3] * D[58] + adj_q[4] * D[67] + adj_q[5] * D[76];
        const s_t pullback_2_1_1_2 = adj_q[3] * D[59] + adj_q[4] * D[68] + adj_q[5] * D[77];
        const s_t pullback_2_1_2_0 = adj_q[6] * D[57] + adj_q[7] * D[66] + adj_q[8] * D[75];
        const s_t pullback_2_1_2_1 = adj_q[6] * D[58] + adj_q[7] * D[67] + adj_q[8] * D[76];
        const s_t pullback_2_1_2_2 = adj_q[6] * D[59] + adj_q[7] * D[68] + adj_q[8] * D[77];
        slab_tangent[p * 81 + 57] = pullback_weight * (pullback_2_1_0_0 * adj_q[0] + pullback_2_1_0_1 * adj_q[1] + pullback_2_1_0_2 * adj_q[2]);
        slab_tangent[p * 81 + 58] = pullback_weight * (pullback_2_1_0_0 * adj_q[3] + pullback_2_1_0_1 * adj_q[4] + pullback_2_1_0_2 * adj_q[5]);
        slab_tangent[p * 81 + 59] = pullback_weight * (pullback_2_1_0_0 * adj_q[6] + pullback_2_1_0_1 * adj_q[7] + pullback_2_1_0_2 * adj_q[8]);
        slab_tangent[p * 81 + 66] = pullback_weight * (pullback_2_1_1_0 * adj_q[0] + pullback_2_1_1_1 * adj_q[1] + pullback_2_1_1_2 * adj_q[2]);
        slab_tangent[p * 81 + 67] = pullback_weight * (pullback_2_1_1_0 * adj_q[3] + pullback_2_1_1_1 * adj_q[4] + pullback_2_1_1_2 * adj_q[5]);
        slab_tangent[p * 81 + 68] = pullback_weight * (pullback_2_1_1_0 * adj_q[6] + pullback_2_1_1_1 * adj_q[7] + pullback_2_1_1_2 * adj_q[8]);
        slab_tangent[p * 81 + 75] = pullback_weight * (pullback_2_1_2_0 * adj_q[0] + pullback_2_1_2_1 * adj_q[1] + pullback_2_1_2_2 * adj_q[2]);
        slab_tangent[p * 81 + 76] = pullback_weight * (pullback_2_1_2_0 * adj_q[3] + pullback_2_1_2_1 * adj_q[4] + pullback_2_1_2_2 * adj_q[5]);
        slab_tangent[p * 81 + 77] = pullback_weight * (pullback_2_1_2_0 * adj_q[6] + pullback_2_1_2_1 * adj_q[7] + pullback_2_1_2_2 * adj_q[8]);
        const s_t pullback_2_2_0_0 = adj_q[0] * D[60] + adj_q[1] * D[69] + adj_q[2] * D[78];
        const s_t pullback_2_2_0_1 = adj_q[0] * D[61] + adj_q[1] * D[70] + adj_q[2] * D[79];
        const s_t pullback_2_2_0_2 = adj_q[0] * D[62] + adj_q[1] * D[71] + adj_q[2] * D[80];
        const s_t pullback_2_2_1_0 = adj_q[3] * D[60] + adj_q[4] * D[69] + adj_q[5] * D[78];
        const s_t pullback_2_2_1_1 = adj_q[3] * D[61] + adj_q[4] * D[70] + adj_q[5] * D[79];
        const s_t pullback_2_2_1_2 = adj_q[3] * D[62] + adj_q[4] * D[71] + adj_q[5] * D[80];
        const s_t pullback_2_2_2_0 = adj_q[6] * D[60] + adj_q[7] * D[69] + adj_q[8] * D[78];
        const s_t pullback_2_2_2_1 = adj_q[6] * D[61] + adj_q[7] * D[70] + adj_q[8] * D[79];
        const s_t pullback_2_2_2_2 = adj_q[6] * D[62] + adj_q[7] * D[71] + adj_q[8] * D[80];
        slab_tangent[p * 81 + 60] = pullback_weight * (pullback_2_2_0_0 * adj_q[0] + pullback_2_2_0_1 * adj_q[1] + pullback_2_2_0_2 * adj_q[2]);
        slab_tangent[p * 81 + 61] = pullback_weight * (pullback_2_2_0_0 * adj_q[3] + pullback_2_2_0_1 * adj_q[4] + pullback_2_2_0_2 * adj_q[5]);
        slab_tangent[p * 81 + 62] = pullback_weight * (pullback_2_2_0_0 * adj_q[6] + pullback_2_2_0_1 * adj_q[7] + pullback_2_2_0_2 * adj_q[8]);
        slab_tangent[p * 81 + 69] = pullback_weight * (pullback_2_2_1_0 * adj_q[0] + pullback_2_2_1_1 * adj_q[1] + pullback_2_2_1_2 * adj_q[2]);
        slab_tangent[p * 81 + 70] = pullback_weight * (pullback_2_2_1_0 * adj_q[3] + pullback_2_2_1_1 * adj_q[4] + pullback_2_2_1_2 * adj_q[5]);
        slab_tangent[p * 81 + 71] = pullback_weight * (pullback_2_2_1_0 * adj_q[6] + pullback_2_2_1_1 * adj_q[7] + pullback_2_2_1_2 * adj_q[8]);
        slab_tangent[p * 81 + 78] = pullback_weight * (pullback_2_2_2_0 * adj_q[0] + pullback_2_2_2_1 * adj_q[1] + pullback_2_2_2_2 * adj_q[2]);
        slab_tangent[p * 81 + 79] = pullback_weight * (pullback_2_2_2_0 * adj_q[3] + pullback_2_2_2_1 * adj_q[4] + pullback_2_2_2_2 * adj_q[5]);
        slab_tangent[p * 81 + 80] = pullback_weight * (pullback_2_2_2_0 * adj_q[6] + pullback_2_2_2_1 * adj_q[7] + pullback_2_2_2_2 * adj_q[8]);
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
            const s_t *const RSTR tangent_p = &slab_tangent[p * 81 + trial_component * ND];
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
