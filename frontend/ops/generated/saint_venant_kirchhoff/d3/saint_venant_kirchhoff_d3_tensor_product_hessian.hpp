#ifndef SAINT_VENANT_KIRCHHOFF_D3_TENSOR_PRODUCT_HESSIAN_HPP
#define SAINT_VENANT_KIRCHHOFF_D3_TENSOR_PRODUCT_HESSIAN_HPP
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
static SFEM_INLINE void saint_venant_kirchhoff_d3_tensor_product_direct_hessian_tensor_product_element_matrix(
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
    const s_t weak_hess_tmp0 = gu0 + s_t(1);
    const s_t weak_hess_tmp1 = pow_2(weak_hess_tmp0);
    const s_t weak_hess_tmp2 = pow_2(gu1);
    const s_t weak_hess_tmp3 = pow_2(gu3);
    const s_t weak_hess_tmp4 = weak_hess_tmp2 + weak_hess_tmp3;
    const s_t weak_hess_tmp5 = pow_2(gu6);
    const s_t weak_hess_tmp6 = pow_2(gu2);
    const s_t weak_hess_tmp7 = weak_hess_tmp6 + s_t(-1);
    const s_t weak_hess_tmp8 = weak_hess_tmp5 + weak_hess_tmp7;
    const s_t weak_hess_tmp9 = pow_2(gu5);
    const s_t weak_hess_tmp10 = pow_2(gu7);
    const s_t weak_hess_tmp11 = lmbda*(((s_t(1) / s_t(2)))*pow_2(gu0) + gu0 + ((s_t(1) / s_t(2)))*pow_2(gu4) + gu4 + ((s_t(1) / s_t(2)))*pow_2(gu8) + gu8 + ((s_t(1) / s_t(2)))*weak_hess_tmp10 + ((s_t(1) / s_t(2)))*weak_hess_tmp2 + ((s_t(1) / s_t(2)))*weak_hess_tmp3 + ((s_t(1) / s_t(2)))*weak_hess_tmp5 + ((s_t(1) / s_t(2)))*weak_hess_tmp6 + ((s_t(1) / s_t(2)))*weak_hess_tmp9);
    const s_t weak_hess_tmp12 = gu1*weak_hess_tmp0;
    const s_t weak_hess_tmp13 = gu6*gu7;
    const s_t weak_hess_tmp14 = gu4 + s_t(1);
    const s_t weak_hess_tmp15 = gu3*weak_hess_tmp14;
    const s_t weak_hess_tmp16 = gu2*weak_hess_tmp0;
    const s_t weak_hess_tmp17 = gu3*gu5;
    const s_t weak_hess_tmp18 = gu8 + s_t(1);
    const s_t weak_hess_tmp19 = gu6*weak_hess_tmp18;
    const s_t weak_hess_tmp20 = gu3*weak_hess_tmp0;
    const s_t weak_hess_tmp21 = gu2*gu5;
    const s_t weak_hess_tmp22 = gu1*weak_hess_tmp14;
    const s_t weak_hess_tmp23 = gu3*mu;
    const s_t weak_hess_tmp24 = lmbda*weak_hess_tmp0;
    const s_t weak_hess_tmp25 = gu6*weak_hess_tmp0;
    const s_t weak_hess_tmp26 = gu1*gu7;
    const s_t weak_hess_tmp27 = gu2*weak_hess_tmp18;
    const s_t weak_hess_tmp28 = gu6*mu;
    const s_t weak_hess_tmp29 = pow_2(weak_hess_tmp14);
    const s_t weak_hess_tmp30 = weak_hess_tmp1 + weak_hess_tmp29;
    const s_t weak_hess_tmp31 = gu1*gu2;
    const s_t weak_hess_tmp32 = gu5*weak_hess_tmp14;
    const s_t weak_hess_tmp33 = gu7*weak_hess_tmp18;
    const s_t weak_hess_tmp34 = gu1*lmbda;
    const s_t weak_hess_tmp35 = mu*weak_hess_tmp14;
    const s_t weak_hess_tmp36 = gu7*mu;
    const s_t weak_hess_tmp37 = weak_hess_tmp9 + s_t(-1);
    const s_t weak_hess_tmp38 = pow_2(weak_hess_tmp18);
    const s_t weak_hess_tmp39 = weak_hess_tmp1 + weak_hess_tmp38;
    const s_t weak_hess_tmp40 = gu2*lmbda;
    const s_t weak_hess_tmp41 = gu5*mu;
    const s_t weak_hess_tmp42 = mu*weak_hess_tmp18;
    const s_t weak_hess_tmp43 = gu3*gu6;
    const s_t weak_hess_tmp44 = gu5*weak_hess_tmp18;
    const s_t weak_hess_tmp45 = gu7*weak_hess_tmp14;
    const s_t weak_hess_tmp46 = gu3*lmbda;
    const s_t weak_hess_tmp47 = lmbda*weak_hess_tmp14;
    const s_t weak_hess_tmp48 = weak_hess_tmp29 + weak_hess_tmp38;
    const s_t weak_hess_tmp49 = gu5*lmbda;
    D[0] = lmbda*weak_hess_tmp1 + mu*(s_t(3)*weak_hess_tmp1 + weak_hess_tmp4 + weak_hess_tmp8) + weak_hess_tmp11;
    D[1] = lmbda*weak_hess_tmp12 + mu*(s_t(2)*weak_hess_tmp12 + weak_hess_tmp13 + weak_hess_tmp15);
    D[2] = lmbda*weak_hess_tmp16 + mu*(s_t(2)*weak_hess_tmp16 + weak_hess_tmp17 + weak_hess_tmp19);
    D[3] = lmbda*weak_hess_tmp20 + mu*(s_t(2)*weak_hess_tmp20 + weak_hess_tmp21 + weak_hess_tmp22);
    D[4] = gu1*weak_hess_tmp23 + weak_hess_tmp14*weak_hess_tmp24;
    D[5] = gu2*weak_hess_tmp23 + gu5*weak_hess_tmp24;
    D[6] = lmbda*weak_hess_tmp25 + mu*(s_t(2)*weak_hess_tmp25 + weak_hess_tmp26 + weak_hess_tmp27);
    D[7] = gu1*weak_hess_tmp28 + gu7*weak_hess_tmp24;
    D[8] = gu2*weak_hess_tmp28 + weak_hess_tmp18*weak_hess_tmp24;
    D[10] = lmbda*weak_hess_tmp2 + mu*(weak_hess_tmp10 + s_t(3)*weak_hess_tmp2 + weak_hess_tmp30 + weak_hess_tmp7) + weak_hess_tmp11;
    D[11] = lmbda*weak_hess_tmp31 + mu*(s_t(2)*weak_hess_tmp31 + weak_hess_tmp32 + weak_hess_tmp33);
    D[12] = gu3*weak_hess_tmp34 + weak_hess_tmp0*weak_hess_tmp35;
    D[13] = lmbda*weak_hess_tmp22 + mu*(weak_hess_tmp20 + weak_hess_tmp21 + s_t(2)*weak_hess_tmp22);
    D[14] = gu2*weak_hess_tmp35 + gu5*weak_hess_tmp34;
    D[15] = gu6*weak_hess_tmp34 + weak_hess_tmp0*weak_hess_tmp36;
    D[16] = lmbda*weak_hess_tmp26 + mu*(weak_hess_tmp25 + s_t(2)*weak_hess_tmp26 + weak_hess_tmp27);
    D[17] = gu2*weak_hess_tmp36 + weak_hess_tmp18*weak_hess_tmp34;
    D[20] = lmbda*weak_hess_tmp6 + mu*(weak_hess_tmp2 + weak_hess_tmp37 + weak_hess_tmp39 + s_t(3)*weak_hess_tmp6) + weak_hess_tmp11;
    D[21] = gu3*weak_hess_tmp40 + weak_hess_tmp0*weak_hess_tmp41;
    D[22] = gu1*weak_hess_tmp41 + weak_hess_tmp14*weak_hess_tmp40;
    D[23] = lmbda*weak_hess_tmp21 + mu*(weak_hess_tmp20 + s_t(2)*weak_hess_tmp21 + weak_hess_tmp22);
    D[24] = gu6*weak_hess_tmp40 + weak_hess_tmp0*weak_hess_tmp42;
    D[25] = gu1*weak_hess_tmp42 + gu7*weak_hess_tmp40;
    D[26] = lmbda*weak_hess_tmp27 + mu*(weak_hess_tmp25 + weak_hess_tmp26 + s_t(2)*weak_hess_tmp27);
    D[30] = lmbda*weak_hess_tmp3 + mu*(s_t(3)*weak_hess_tmp3 + weak_hess_tmp30 + weak_hess_tmp37 + weak_hess_tmp5) + weak_hess_tmp11;
    D[31] = lmbda*weak_hess_tmp15 + mu*(weak_hess_tmp12 + weak_hess_tmp13 + s_t(2)*weak_hess_tmp15);
    D[32] = lmbda*weak_hess_tmp17 + mu*(weak_hess_tmp16 + s_t(2)*weak_hess_tmp17 + weak_hess_tmp19);
    D[33] = lmbda*weak_hess_tmp43 + mu*(s_t(2)*weak_hess_tmp43 + weak_hess_tmp44 + weak_hess_tmp45);
    D[34] = gu7*weak_hess_tmp46 + weak_hess_tmp14*weak_hess_tmp28;
    D[35] = gu5*weak_hess_tmp28 + weak_hess_tmp18*weak_hess_tmp46;
    D[40] = lmbda*weak_hess_tmp29 + mu*(weak_hess_tmp10 + s_t(3)*weak_hess_tmp29 + weak_hess_tmp37 + weak_hess_tmp4) + weak_hess_tmp11;
    D[41] = lmbda*weak_hess_tmp32 + mu*(weak_hess_tmp31 + s_t(2)*weak_hess_tmp32 + weak_hess_tmp33);
    D[42] = gu6*weak_hess_tmp47 + gu7*weak_hess_tmp23;
    D[43] = lmbda*weak_hess_tmp45 + mu*(weak_hess_tmp43 + weak_hess_tmp44 + s_t(2)*weak_hess_tmp45);
    D[44] = gu5*weak_hess_tmp36 + weak_hess_tmp18*weak_hess_tmp47;
    D[50] = lmbda*weak_hess_tmp9 + mu*(weak_hess_tmp3 + weak_hess_tmp48 + weak_hess_tmp7 + s_t(3)*weak_hess_tmp9) + weak_hess_tmp11;
    D[51] = gu6*weak_hess_tmp49 + weak_hess_tmp18*weak_hess_tmp23;
    D[52] = gu7*weak_hess_tmp49 + weak_hess_tmp18*weak_hess_tmp35;
    D[53] = lmbda*weak_hess_tmp44 + mu*(weak_hess_tmp43 + s_t(2)*weak_hess_tmp44 + weak_hess_tmp45);
    D[60] = lmbda*weak_hess_tmp5 + mu*(weak_hess_tmp10 + weak_hess_tmp3 + weak_hess_tmp39 + s_t(3)*weak_hess_tmp5 + s_t(-1)) + weak_hess_tmp11;
    D[61] = lmbda*weak_hess_tmp13 + mu*(weak_hess_tmp12 + s_t(2)*weak_hess_tmp13 + weak_hess_tmp15);
    D[62] = lmbda*weak_hess_tmp19 + mu*(weak_hess_tmp16 + weak_hess_tmp17 + s_t(2)*weak_hess_tmp19);
    D[70] = lmbda*weak_hess_tmp10 + mu*(s_t(3)*weak_hess_tmp10 + weak_hess_tmp2 + weak_hess_tmp48 + weak_hess_tmp5 + s_t(-1)) + weak_hess_tmp11;
    D[71] = lmbda*weak_hess_tmp33 + mu*(weak_hess_tmp31 + weak_hess_tmp32 + s_t(2)*weak_hess_tmp33);
    D[80] = lmbda*weak_hess_tmp38 + mu*(weak_hess_tmp10 + s_t(3)*weak_hess_tmp38 + weak_hess_tmp8 + weak_hess_tmp9) + weak_hess_tmp11;
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
    adj_q[0] = adj_lane0;
    adj_q[1] = adj_lane1;
    adj_q[2] = adj_lane2;
    adj_q[3] = adj_lane3;
    adj_q[4] = adj_lane4;
    adj_q[5] = adj_lane5;
    adj_q[6] = adj_lane6;
    adj_q[7] = adj_lane7;
    adj_q[8] = adj_lane8;
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
