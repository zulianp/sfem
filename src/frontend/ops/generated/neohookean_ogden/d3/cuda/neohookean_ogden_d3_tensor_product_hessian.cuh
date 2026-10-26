#ifndef NEOHOOKEAN_OGDEN_D3_TENSOR_PRODUCT_HESSIAN_CUH
#define NEOHOOKEAN_OGDEN_D3_TENSOR_PRODUCT_HESSIAN_CUH
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
static __host__ __device__ __forceinline__ void neohookean_ogden_d3_tensor_product_direct_hessian_tensor_product_element_matrix(
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
    const s_t weak_hess_tmp9 = pow_m2(weak_hess_tmp8);
    const s_t weak_hess_tmp10 = lmbda*weak_hess_tmp9;
    const s_t weak_hess_tmp11 = mu*weak_hess_tmp9;
    const s_t weak_hess_tmp12 = weak_hess_tmp3*weak_hess_tmp4;
    const s_t weak_hess_tmp13 = gu0*gu4;
    const s_t weak_hess_tmp14 = gu3*gu7;
    const s_t weak_hess_tmp15 = sfem_log1p(gu0*gu8 - gu0*weak_hess_tmp0 + gu0 + gu1*gu5*gu6 + gu2*weak_hess_tmp14 + gu4*gu8 - gu4*weak_hess_tmp6 + gu4 + gu8*weak_hess_tmp13 - gu8*weak_hess_tmp5 + gu8 - weak_hess_tmp0 + weak_hess_tmp13 - weak_hess_tmp5 - weak_hess_tmp6);
    const s_t weak_hess_tmp16 = weak_hess_tmp10*weak_hess_tmp15;
    const s_t weak_hess_tmp17 = gu3*weak_hess_tmp2 - gu5*gu6;
    const s_t weak_hess_tmp18 = -weak_hess_tmp17;
    const s_t weak_hess_tmp19 = weak_hess_tmp10*weak_hess_tmp4;
    const s_t weak_hess_tmp20 = weak_hess_tmp11*weak_hess_tmp4;
    const s_t weak_hess_tmp21 = weak_hess_tmp15*weak_hess_tmp19;
    const s_t weak_hess_tmp22 = -gu6*weak_hess_tmp1 + weak_hess_tmp14;
    const s_t weak_hess_tmp23 = -weak_hess_tmp22;
    const s_t weak_hess_tmp24 = gu1*weak_hess_tmp2 - gu2*gu7;
    const s_t weak_hess_tmp25 = -weak_hess_tmp24;
    const s_t weak_hess_tmp26 = pow_m1(weak_hess_tmp8);
    const s_t weak_hess_tmp27 = mu*weak_hess_tmp26;
    const s_t weak_hess_tmp28 = weak_hess_tmp2*weak_hess_tmp27;
    const s_t weak_hess_tmp29 = -weak_hess_tmp2*weak_hess_tmp7 + weak_hess_tmp6;
    const s_t weak_hess_tmp30 = -weak_hess_tmp29;
    const s_t weak_hess_tmp31 = lmbda*weak_hess_tmp15*weak_hess_tmp26;
    const s_t weak_hess_tmp32 = weak_hess_tmp2*weak_hess_tmp31;
    const s_t weak_hess_tmp33 = gu7*weak_hess_tmp27;
    const s_t weak_hess_tmp34 = gu1*gu6 - gu7*weak_hess_tmp7;
    const s_t weak_hess_tmp35 = -weak_hess_tmp34;
    const s_t weak_hess_tmp36 = gu7*weak_hess_tmp31;
    const s_t weak_hess_tmp37 = gu1*gu5 - gu2*weak_hess_tmp1;
    const s_t weak_hess_tmp38 = -weak_hess_tmp37;
    const s_t weak_hess_tmp39 = gu5*weak_hess_tmp27;
    const s_t weak_hess_tmp40 = gu2*gu3 - gu5*weak_hess_tmp7;
    const s_t weak_hess_tmp41 = -weak_hess_tmp40;
    const s_t weak_hess_tmp42 = gu5*weak_hess_tmp31;
    const s_t weak_hess_tmp43 = weak_hess_tmp1*weak_hess_tmp27;
    const s_t weak_hess_tmp44 = -weak_hess_tmp1*weak_hess_tmp7 + weak_hess_tmp5;
    const s_t weak_hess_tmp45 = -weak_hess_tmp44;
    const s_t weak_hess_tmp46 = weak_hess_tmp1*weak_hess_tmp31;
    const s_t weak_hess_tmp47 = weak_hess_tmp11*weak_hess_tmp18;
    const s_t weak_hess_tmp48 = weak_hess_tmp10*weak_hess_tmp18;
    const s_t weak_hess_tmp49 = weak_hess_tmp15*weak_hess_tmp48;
    const s_t weak_hess_tmp50 = gu6*weak_hess_tmp27;
    const s_t weak_hess_tmp51 = gu6*weak_hess_tmp31;
    const s_t weak_hess_tmp52 = gu3*weak_hess_tmp27;
    const s_t weak_hess_tmp53 = gu3*weak_hess_tmp31;
    const s_t weak_hess_tmp54 = weak_hess_tmp11*weak_hess_tmp22;
    const s_t weak_hess_tmp55 = weak_hess_tmp10*weak_hess_tmp22;
    const s_t weak_hess_tmp56 = weak_hess_tmp15*weak_hess_tmp55;
    const s_t weak_hess_tmp57 = weak_hess_tmp11*weak_hess_tmp25;
    const s_t weak_hess_tmp58 = weak_hess_tmp10*weak_hess_tmp25;
    const s_t weak_hess_tmp59 = weak_hess_tmp15*weak_hess_tmp58;
    const s_t weak_hess_tmp60 = gu2*weak_hess_tmp27;
    const s_t weak_hess_tmp61 = gu2*weak_hess_tmp31;
    const s_t weak_hess_tmp62 = gu1*weak_hess_tmp27;
    const s_t weak_hess_tmp63 = gu1*weak_hess_tmp31;
    const s_t weak_hess_tmp64 = weak_hess_tmp11*weak_hess_tmp30;
    const s_t weak_hess_tmp65 = weak_hess_tmp10*weak_hess_tmp30;
    const s_t weak_hess_tmp66 = weak_hess_tmp15*weak_hess_tmp65;
    const s_t weak_hess_tmp67 = weak_hess_tmp27*weak_hess_tmp7;
    const s_t weak_hess_tmp68 = weak_hess_tmp31*weak_hess_tmp7;
    const s_t weak_hess_tmp69 = weak_hess_tmp11*weak_hess_tmp34;
    const s_t weak_hess_tmp70 = weak_hess_tmp10*weak_hess_tmp34;
    const s_t weak_hess_tmp71 = weak_hess_tmp15*weak_hess_tmp70;
    const s_t weak_hess_tmp72 = weak_hess_tmp11*weak_hess_tmp37;
    const s_t weak_hess_tmp73 = weak_hess_tmp10*weak_hess_tmp37;
    const s_t weak_hess_tmp74 = weak_hess_tmp15*weak_hess_tmp73;
    const s_t weak_hess_tmp75 = weak_hess_tmp11*weak_hess_tmp40;
    const s_t weak_hess_tmp76 = weak_hess_tmp10*weak_hess_tmp40;
    const s_t weak_hess_tmp77 = weak_hess_tmp15*weak_hess_tmp76;
    const s_t weak_hess_tmp78 = weak_hess_tmp44*weak_hess_tmp45;
    D[0] = mu + weak_hess_tmp10*pow_2(weak_hess_tmp4) - weak_hess_tmp11*weak_hess_tmp12 + weak_hess_tmp12*weak_hess_tmp16;
    D[1] = -weak_hess_tmp17*weak_hess_tmp20 + weak_hess_tmp17*weak_hess_tmp21 + weak_hess_tmp18*weak_hess_tmp19;
    D[2] = weak_hess_tmp19*weak_hess_tmp22 - weak_hess_tmp20*weak_hess_tmp23 + weak_hess_tmp21*weak_hess_tmp23;
    D[3] = weak_hess_tmp19*weak_hess_tmp25 - weak_hess_tmp20*weak_hess_tmp24 + weak_hess_tmp21*weak_hess_tmp24;
    D[4] = weak_hess_tmp19*weak_hess_tmp30 - weak_hess_tmp20*weak_hess_tmp29 + weak_hess_tmp21*weak_hess_tmp29 - weak_hess_tmp28 + weak_hess_tmp32;
    D[5] = weak_hess_tmp19*weak_hess_tmp34 - weak_hess_tmp20*weak_hess_tmp35 + weak_hess_tmp21*weak_hess_tmp35 + weak_hess_tmp33 - weak_hess_tmp36;
    D[6] = weak_hess_tmp19*weak_hess_tmp37 - weak_hess_tmp20*weak_hess_tmp38 + weak_hess_tmp21*weak_hess_tmp38;
    D[7] = weak_hess_tmp19*weak_hess_tmp40 - weak_hess_tmp20*weak_hess_tmp41 + weak_hess_tmp21*weak_hess_tmp41 + weak_hess_tmp39 - weak_hess_tmp42;
    D[8] = weak_hess_tmp19*weak_hess_tmp45 - weak_hess_tmp20*weak_hess_tmp44 + weak_hess_tmp21*weak_hess_tmp44 - weak_hess_tmp43 + weak_hess_tmp46;
    D[10] = mu + weak_hess_tmp10*pow_2(weak_hess_tmp18) - weak_hess_tmp17*weak_hess_tmp47 + weak_hess_tmp17*weak_hess_tmp49;
    D[11] = weak_hess_tmp22*weak_hess_tmp48 - weak_hess_tmp23*weak_hess_tmp47 + weak_hess_tmp23*weak_hess_tmp49;
    D[12] = -weak_hess_tmp24*weak_hess_tmp47 + weak_hess_tmp24*weak_hess_tmp49 + weak_hess_tmp25*weak_hess_tmp48 + weak_hess_tmp28 - weak_hess_tmp32;
    D[13] = -weak_hess_tmp29*weak_hess_tmp47 + weak_hess_tmp29*weak_hess_tmp49 + weak_hess_tmp30*weak_hess_tmp48;
    D[14] = weak_hess_tmp34*weak_hess_tmp48 - weak_hess_tmp35*weak_hess_tmp47 + weak_hess_tmp35*weak_hess_tmp49 - weak_hess_tmp50 + weak_hess_tmp51;
    D[15] = weak_hess_tmp37*weak_hess_tmp48 - weak_hess_tmp38*weak_hess_tmp47 + weak_hess_tmp38*weak_hess_tmp49 - weak_hess_tmp39 + weak_hess_tmp42;
    D[16] = weak_hess_tmp40*weak_hess_tmp48 - weak_hess_tmp41*weak_hess_tmp47 + weak_hess_tmp41*weak_hess_tmp49;
    D[17] = -weak_hess_tmp44*weak_hess_tmp47 + weak_hess_tmp44*weak_hess_tmp49 + weak_hess_tmp45*weak_hess_tmp48 + weak_hess_tmp52 - weak_hess_tmp53;
    D[20] = mu + weak_hess_tmp10*pow_2(weak_hess_tmp22) - weak_hess_tmp23*weak_hess_tmp54 + weak_hess_tmp23*weak_hess_tmp56;
    D[21] = -weak_hess_tmp24*weak_hess_tmp54 + weak_hess_tmp24*weak_hess_tmp56 + weak_hess_tmp25*weak_hess_tmp55 - weak_hess_tmp33 + weak_hess_tmp36;
    D[22] = -weak_hess_tmp29*weak_hess_tmp54 + weak_hess_tmp29*weak_hess_tmp56 + weak_hess_tmp30*weak_hess_tmp55 + weak_hess_tmp50 - weak_hess_tmp51;
    D[23] = weak_hess_tmp34*weak_hess_tmp55 - weak_hess_tmp35*weak_hess_tmp54 + weak_hess_tmp35*weak_hess_tmp56;
    D[24] = weak_hess_tmp37*weak_hess_tmp55 - weak_hess_tmp38*weak_hess_tmp54 + weak_hess_tmp38*weak_hess_tmp56 + weak_hess_tmp43 - weak_hess_tmp46;
    D[25] = weak_hess_tmp40*weak_hess_tmp55 - weak_hess_tmp41*weak_hess_tmp54 + weak_hess_tmp41*weak_hess_tmp56 - weak_hess_tmp52 + weak_hess_tmp53;
    D[26] = -weak_hess_tmp44*weak_hess_tmp54 + weak_hess_tmp44*weak_hess_tmp56 + weak_hess_tmp45*weak_hess_tmp55;
    D[30] = mu + weak_hess_tmp10*pow_2(weak_hess_tmp25) - weak_hess_tmp24*weak_hess_tmp57 + weak_hess_tmp24*weak_hess_tmp59;
    D[31] = -weak_hess_tmp29*weak_hess_tmp57 + weak_hess_tmp29*weak_hess_tmp59 + weak_hess_tmp30*weak_hess_tmp58;
    D[32] = weak_hess_tmp34*weak_hess_tmp58 - weak_hess_tmp35*weak_hess_tmp57 + weak_hess_tmp35*weak_hess_tmp59;
    D[33] = weak_hess_tmp37*weak_hess_tmp58 - weak_hess_tmp38*weak_hess_tmp57 + weak_hess_tmp38*weak_hess_tmp59;
    D[34] = weak_hess_tmp40*weak_hess_tmp58 - weak_hess_tmp41*weak_hess_tmp57 + weak_hess_tmp41*weak_hess_tmp59 - weak_hess_tmp60 + weak_hess_tmp61;
    D[35] = -weak_hess_tmp44*weak_hess_tmp57 + weak_hess_tmp44*weak_hess_tmp59 + weak_hess_tmp45*weak_hess_tmp58 + weak_hess_tmp62 - weak_hess_tmp63;
    D[40] = mu + weak_hess_tmp10*pow_2(weak_hess_tmp30) - weak_hess_tmp29*weak_hess_tmp64 + weak_hess_tmp29*weak_hess_tmp66;
    D[41] = weak_hess_tmp34*weak_hess_tmp65 - weak_hess_tmp35*weak_hess_tmp64 + weak_hess_tmp35*weak_hess_tmp66;
    D[42] = weak_hess_tmp37*weak_hess_tmp65 - weak_hess_tmp38*weak_hess_tmp64 + weak_hess_tmp38*weak_hess_tmp66 + weak_hess_tmp60 - weak_hess_tmp61;
    D[43] = weak_hess_tmp40*weak_hess_tmp65 - weak_hess_tmp41*weak_hess_tmp64 + weak_hess_tmp41*weak_hess_tmp66;
    D[44] = -weak_hess_tmp44*weak_hess_tmp64 + weak_hess_tmp44*weak_hess_tmp66 + weak_hess_tmp45*weak_hess_tmp65 - weak_hess_tmp67 + weak_hess_tmp68;
    D[50] = mu + weak_hess_tmp10*pow_2(weak_hess_tmp34) - weak_hess_tmp35*weak_hess_tmp69 + weak_hess_tmp35*weak_hess_tmp71;
    D[51] = weak_hess_tmp37*weak_hess_tmp70 - weak_hess_tmp38*weak_hess_tmp69 + weak_hess_tmp38*weak_hess_tmp71 - weak_hess_tmp62 + weak_hess_tmp63;
    D[52] = weak_hess_tmp40*weak_hess_tmp70 - weak_hess_tmp41*weak_hess_tmp69 + weak_hess_tmp41*weak_hess_tmp71 + weak_hess_tmp67 - weak_hess_tmp68;
    D[53] = -weak_hess_tmp44*weak_hess_tmp69 + weak_hess_tmp44*weak_hess_tmp71 + weak_hess_tmp45*weak_hess_tmp70;
    D[60] = mu + weak_hess_tmp10*pow_2(weak_hess_tmp37) - weak_hess_tmp38*weak_hess_tmp72 + weak_hess_tmp38*weak_hess_tmp74;
    D[61] = weak_hess_tmp40*weak_hess_tmp73 - weak_hess_tmp41*weak_hess_tmp72 + weak_hess_tmp41*weak_hess_tmp74;
    D[62] = -weak_hess_tmp44*weak_hess_tmp72 + weak_hess_tmp44*weak_hess_tmp74 + weak_hess_tmp45*weak_hess_tmp73;
    D[70] = mu + weak_hess_tmp10*pow_2(weak_hess_tmp40) - weak_hess_tmp41*weak_hess_tmp75 + weak_hess_tmp41*weak_hess_tmp77;
    D[71] = -weak_hess_tmp44*weak_hess_tmp75 + weak_hess_tmp44*weak_hess_tmp77 + weak_hess_tmp45*weak_hess_tmp76;
    D[80] = mu + weak_hess_tmp10*pow_2(weak_hess_tmp45) - weak_hess_tmp11*weak_hess_tmp78 + weak_hess_tmp16*weak_hess_tmp78;
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
