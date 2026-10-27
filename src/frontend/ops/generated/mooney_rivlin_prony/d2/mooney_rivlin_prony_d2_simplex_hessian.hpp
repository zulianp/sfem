#ifndef MOONEY_RIVLIN_PRONY_D2_SIMPLEX_HESSIAN_HPP
#define MOONEY_RIVLIN_PRONY_D2_SIMPLEX_HESSIAN_HPP
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
static SFEM_INLINE void mooney_rivlin_prony_d2_simplex_direct_hessian_reference_element_matrix(
    const s_t *const RSTR badj0,
    const s_t *const RSTR badj1,
    const s_t *const RSTR badj2,
    const s_t *const RSTR badj3,
    const s_t *const RSTR bdet0,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
    const s_t C01,
    const s_t C10,
    const s_t K,
    const s_t iso_relaxation_gamma,
    const s_t *const RSTR iso_history,
    const ptrdiff_t qp_stride,
    const s_t bu_data[NS * 2],
    s_t *const RSTR element_matrix
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(NS > 0, "NS must be positive");
  static constexpr int NC = 2;
  static constexpr int ND = 2;
  static constexpr int NDOFS = NC * NS;
  for (int entry = 0; entry < NDOFS * NDOFS; ++entry) {
    element_matrix[entry] = s_t(0);
  }
  for (int q = 0; q < NQ; ++q) {
    const s_t qw = q_weight[q];
    const ptrdiff_t goff = q;
    const s_t adj_lane0 = badj0[goff];
    const s_t adj_lane1 = badj1[goff];
    const s_t adj_lane2 = badj2[goff];
    const s_t adj_lane3 = badj3[goff];
    const s_t det_lane0 = bdet0[goff];
    const s_t idet = s_t(1) / det_lane0;
    const s_t *const RSTR iso_history_q = iso_history + q * qp_stride;
    const s_t iso_history_qp[3] = {iso_history_q[0], iso_history_q[1 * NQ * qp_stride], iso_history_q[2 * NQ * qp_stride]};
    s_t gu_ref0 = s_t(0);
    s_t gu_ref1 = s_t(0);
    s_t gu_ref2 = s_t(0);
    s_t gu_ref3 = s_t(0);
    for (int shape = 0; shape < NS; ++shape) {
      const s_t state_grad_ref0 = grad_ref_x[q * NS + shape];
      const s_t state_grad_ref1 = grad_ref_y[q * NS + shape];
      const s_t state_u0 = bu_data[shape * NC];
      gu_ref0 += state_u0 * state_grad_ref0;
      gu_ref1 += state_u0 * state_grad_ref1;
      const s_t state_u1 = bu_data[shape * NC + 1];
      gu_ref2 += state_u1 * state_grad_ref0;
      gu_ref3 += state_u1 * state_grad_ref1;
    }
    const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane2) * idet;
    const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane3) * idet;
    const s_t gu2 = (gu_ref2 * adj_lane0 + gu_ref3 * adj_lane2) * idet;
    const s_t gu3 = (gu_ref2 * adj_lane1 + gu_ref3 * adj_lane3) * idet;
    s_t D[16];
    const s_t weak_hess_tmp0 = gu3 + s_t(1);
    const s_t weak_hess_tmp1 = pow_2(weak_hess_tmp0);
    const s_t weak_hess_tmp2 = pow_2(gu2);
    const s_t weak_hess_tmp3 = gu0 + s_t(1);
    const s_t weak_hess_tmp4 = pow_2(weak_hess_tmp3);
    const s_t weak_hess_tmp5 = weak_hess_tmp2 + weak_hess_tmp4;
    const s_t weak_hess_tmp6 = pow_2(gu1);
    const s_t weak_hess_tmp7 = weak_hess_tmp1 + weak_hess_tmp6;
    const s_t weak_hess_tmp8 = weak_hess_tmp5 + weak_hess_tmp7;
    const s_t weak_hess_tmp9 = weak_hess_tmp8 + s_t(1);
    const s_t weak_hess_tmp10 = gu1*gu2;
    const s_t weak_hess_tmp11 = weak_hess_tmp0*weak_hess_tmp3 - weak_hess_tmp10;
    const s_t weak_hess_tmp12 = pow(weak_hess_tmp11, (s_t(-8) / s_t(3)));
    const s_t weak_hess_tmp13 = ((s_t(10) / s_t(9)))*weak_hess_tmp12*weak_hess_tmp9;
    const s_t weak_hess_tmp14 = s_t(2)/pow(weak_hess_tmp11, (s_t(2) / s_t(3)));
    const s_t weak_hess_tmp15 = weak_hess_tmp0*weak_hess_tmp3;
    const s_t weak_hess_tmp16 = pow(weak_hess_tmp11, (s_t(-5) / s_t(3)));
    const s_t weak_hess_tmp17 = ((s_t(8) / s_t(3)))*weak_hess_tmp16;
    const s_t weak_hess_tmp18 = weak_hess_tmp14 - weak_hess_tmp15*weak_hess_tmp17;
    const s_t weak_hess_tmp19 = pow(weak_hess_tmp11, (s_t(-4) / s_t(3)));
    const s_t weak_hess_tmp20 = gu1*weak_hess_tmp3;
    const s_t weak_hess_tmp21 = gu2*weak_hess_tmp0;
    const s_t weak_hess_tmp22 = weak_hess_tmp20 + weak_hess_tmp21;
    const s_t weak_hess_tmp23 = s_t(2)*gu1;
    const s_t weak_hess_tmp24 = s_t(2)*weak_hess_tmp3;
    const s_t weak_hess_tmp25 = s_t(2)*gu0 - weak_hess_tmp22*weak_hess_tmp23 - weak_hess_tmp24*weak_hess_tmp5 + weak_hess_tmp24*weak_hess_tmp8 + s_t(2);
    const s_t weak_hess_tmp26 = pow(weak_hess_tmp11, (s_t(-7) / s_t(3)));
    const s_t weak_hess_tmp27 = ((s_t(8) / s_t(3)))*weak_hess_tmp26;
    const s_t weak_hess_tmp28 = -pow_2(weak_hess_tmp22) - (s_t(1) / s_t(2))*pow_2(weak_hess_tmp5) - (s_t(1) / s_t(2))*pow_2(weak_hess_tmp7) + ((s_t(1) / s_t(2)))*pow_2(weak_hess_tmp8) + weak_hess_tmp8;
    const s_t weak_hess_tmp29 = pow(weak_hess_tmp11, (s_t(-10) / s_t(3)));
    const s_t weak_hess_tmp30 = ((s_t(28) / s_t(9)))*weak_hess_tmp28*weak_hess_tmp29;
    const s_t weak_hess_tmp31 = gu1*weak_hess_tmp0;
    const s_t weak_hess_tmp32 = ((s_t(4) / s_t(3)))*weak_hess_tmp16;
    const s_t weak_hess_tmp33 = weak_hess_tmp31*weak_hess_tmp32;
    const s_t weak_hess_tmp34 = gu2*weak_hess_tmp3;
    const s_t weak_hess_tmp35 = weak_hess_tmp32*weak_hess_tmp34;
    const s_t weak_hess_tmp36 = s_t(2)*weak_hess_tmp19;
    const s_t weak_hess_tmp37 = s_t(2)*gu1*weak_hess_tmp8 + s_t(2)*gu1 - weak_hess_tmp22*weak_hess_tmp24 - weak_hess_tmp23*weak_hess_tmp7;
    const s_t weak_hess_tmp38 = ((s_t(4) / s_t(3)))*weak_hess_tmp26;
    const s_t weak_hess_tmp39 = weak_hess_tmp0*weak_hess_tmp38;
    const s_t weak_hess_tmp40 = weak_hess_tmp20*weak_hess_tmp32;
    const s_t weak_hess_tmp41 = weak_hess_tmp21*weak_hess_tmp32;
    const s_t weak_hess_tmp42 = s_t(2)*gu2;
    const s_t weak_hess_tmp43 = s_t(2)*weak_hess_tmp0;
    const s_t weak_hess_tmp44 = s_t(2)*gu2*weak_hess_tmp8 + s_t(2)*gu2 - weak_hess_tmp22*weak_hess_tmp43 - weak_hess_tmp42*weak_hess_tmp5;
    const s_t weak_hess_tmp45 = K*(gu0*gu3 + gu0 + gu3 - weak_hess_tmp10);
    const s_t weak_hess_tmp46 = ((s_t(2) / s_t(3)))*weak_hess_tmp16*weak_hess_tmp9;
    const s_t weak_hess_tmp47 = s_t(2)*gu3 - weak_hess_tmp22*weak_hess_tmp42 - weak_hess_tmp43*weak_hess_tmp7 + weak_hess_tmp43*weak_hess_tmp8 + s_t(2);
    const s_t weak_hess_tmp48 = weak_hess_tmp28*weak_hess_tmp38;
    const s_t weak_hess_tmp49 = weak_hess_tmp10*weak_hess_tmp17 + weak_hess_tmp14;
    const s_t weak_hess_tmp50 = weak_hess_tmp37*weak_hess_tmp38;
    D[0] = K*weak_hess_tmp1 + iso_history_qp[0] + iso_relaxation_gamma*(C01*(-weak_hess_tmp0*weak_hess_tmp25*weak_hess_tmp27 + weak_hess_tmp1*weak_hess_tmp30 + weak_hess_tmp19*(s_t(2)*weak_hess_tmp1 + s_t(2))) + C10*(weak_hess_tmp1*weak_hess_tmp13 + weak_hess_tmp18));
    D[1] = -K*weak_hess_tmp21 + iso_history_qp[2] + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu2*weak_hess_tmp25*weak_hess_tmp26 - weak_hess_tmp21*weak_hess_tmp30 - weak_hess_tmp21*weak_hess_tmp36 - weak_hess_tmp37*weak_hess_tmp39) + C10*(-weak_hess_tmp13*weak_hess_tmp21 - weak_hess_tmp33 + weak_hess_tmp35));
    D[2] = -K*weak_hess_tmp31 + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu1*weak_hess_tmp25*weak_hess_tmp26 - weak_hess_tmp0*weak_hess_tmp19*weak_hess_tmp23 - weak_hess_tmp30*weak_hess_tmp31 - weak_hess_tmp39*weak_hess_tmp44) + C10*(-weak_hess_tmp13*weak_hess_tmp31 + weak_hess_tmp40 - weak_hess_tmp41));
    D[3] = K*weak_hess_tmp15 + iso_relaxation_gamma*(C01*(((s_t(28) / s_t(9)))*weak_hess_tmp0*weak_hess_tmp28*weak_hess_tmp29*weak_hess_tmp3 + weak_hess_tmp19*(s_t(4)*weak_hess_tmp0*weak_hess_tmp3 - s_t(2)*weak_hess_tmp10) - weak_hess_tmp25*weak_hess_tmp3*weak_hess_tmp38 - weak_hess_tmp39*weak_hess_tmp47 - weak_hess_tmp48) + C10*(((s_t(10) / s_t(9)))*weak_hess_tmp0*weak_hess_tmp12*weak_hess_tmp3*weak_hess_tmp9 - weak_hess_tmp1*weak_hess_tmp32 - weak_hess_tmp32*weak_hess_tmp4 - weak_hess_tmp46)) + weak_hess_tmp45;
    D[5] = K*weak_hess_tmp2 + iso_history_qp[1] + iso_relaxation_gamma*(C01*(gu2*weak_hess_tmp27*weak_hess_tmp37 + weak_hess_tmp19*(s_t(2)*weak_hess_tmp2 + s_t(2)) + weak_hess_tmp2*weak_hess_tmp30) + C10*(weak_hess_tmp13*weak_hess_tmp2 + weak_hess_tmp49));
    D[6] = K*weak_hess_tmp10 + iso_relaxation_gamma*(C01*(gu1*weak_hess_tmp50 + gu2*weak_hess_tmp38*weak_hess_tmp44 + weak_hess_tmp10*weak_hess_tmp30 + weak_hess_tmp19*(s_t(4)*weak_hess_tmp10 - s_t(2)*weak_hess_tmp15) + weak_hess_tmp48) + C10*(weak_hess_tmp10*weak_hess_tmp13 + weak_hess_tmp2*weak_hess_tmp32 + weak_hess_tmp32*weak_hess_tmp6 + weak_hess_tmp46)) - weak_hess_tmp45;
    D[7] = -K*weak_hess_tmp34 + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu2*weak_hess_tmp26*weak_hess_tmp47 - weak_hess_tmp19*weak_hess_tmp3*weak_hess_tmp42 - weak_hess_tmp3*weak_hess_tmp50 - weak_hess_tmp30*weak_hess_tmp34) + C10*(-weak_hess_tmp13*weak_hess_tmp34 - weak_hess_tmp40 + weak_hess_tmp41));
    D[10] = K*weak_hess_tmp6 + iso_history_qp[0] + iso_relaxation_gamma*(C01*(gu1*weak_hess_tmp27*weak_hess_tmp44 + weak_hess_tmp19*(s_t(2)*weak_hess_tmp6 + s_t(2)) + weak_hess_tmp30*weak_hess_tmp6) + C10*(weak_hess_tmp13*weak_hess_tmp6 + weak_hess_tmp49));
    D[11] = -K*weak_hess_tmp20 + iso_history_qp[2] + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu1*weak_hess_tmp26*weak_hess_tmp47 - weak_hess_tmp20*weak_hess_tmp30 - weak_hess_tmp20*weak_hess_tmp36 - weak_hess_tmp3*weak_hess_tmp38*weak_hess_tmp44) + C10*(-weak_hess_tmp13*weak_hess_tmp20 + weak_hess_tmp33 - weak_hess_tmp35));
    D[15] = K*weak_hess_tmp4 + iso_history_qp[1] + iso_relaxation_gamma*(C01*(weak_hess_tmp19*(s_t(2)*weak_hess_tmp4 + s_t(2)) - weak_hess_tmp27*weak_hess_tmp3*weak_hess_tmp47 + weak_hess_tmp30*weak_hess_tmp4) + C10*(weak_hess_tmp13*weak_hess_tmp4 + weak_hess_tmp18));
    D[4] = D[1];
    D[8] = D[2];
    D[12] = D[3];
    D[9] = D[6];
    D[13] = D[7];
    D[14] = D[11];
    s_t g_ref[ND * NS];
    for (int shape = 0; shape < NS; ++shape) {
      g_ref[0 * NS + shape] = grad_ref_x[q * NS + shape];
      g_ref[1 * NS + shape] = grad_ref_y[q * NS + shape];
    }
    const s_t adj_q[ND * ND] = {adj_lane0, adj_lane1, adj_lane2, adj_lane3};
    point_test_symmetric_scalar<s_t, NS, ND, NC>(g_ref, adj_q, qw * idet, D, element_matrix);
  }
  mirror_upper_triangle_scalar<s_t, NDOFS>(element_matrix);
}

template <typename s_t, int NS>
static SFEM_INLINE void mooney_rivlin_prony_d2_simplex_tri3_direct_hessian_element_matrix(
    const s_t *const RSTR badj0,
    const s_t *const RSTR badj1,
    const s_t *const RSTR badj2,
    const s_t *const RSTR badj3,
    const s_t *const RSTR bdet0,
    const s_t C01,
    const s_t C10,
    const s_t K,
    const s_t iso_relaxation_gamma,
    const s_t *const RSTR iso_history,
    const ptrdiff_t qp_stride,
    const s_t bu_data[NS * 2],
    s_t *const RSTR element_matrix
) {
  static_assert(NS > 0, "NS must be positive");
  const ptrdiff_t goff = 0;
  const s_t adj_lane0 = badj0[goff];
  const s_t adj_lane1 = badj1[goff];
  const s_t adj_lane2 = badj2[goff];
  const s_t adj_lane3 = badj3[goff];
  const s_t det_lane0 = bdet0[goff];
  const s_t idet = s_t(1) / det_lane0;
  const s_t *const RSTR iso_history_q = iso_history;
  const s_t iso_history_qp[3] = {iso_history_q[0], iso_history_q[1 * qp_stride], iso_history_q[2 * qp_stride]};
  const s_t gu_ref0 = -bu_data[0] + bu_data[2];
  const s_t gu_ref1 = -bu_data[0] + bu_data[4];
  const s_t gu_ref2 = -bu_data[1] + bu_data[3];
  const s_t gu_ref3 = -bu_data[1] + bu_data[5];
  const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane2) * idet;
  const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane3) * idet;
  const s_t gu2 = (gu_ref2 * adj_lane0 + gu_ref3 * adj_lane2) * idet;
  const s_t gu3 = (gu_ref2 * adj_lane1 + gu_ref3 * adj_lane3) * idet;
  const s_t hessian_tmp0 = pow_2(gu2);
  const s_t hessian_tmp1 = gu0 + s_t(1);
  const s_t hessian_tmp2 = pow_2(hessian_tmp1);
  const s_t hessian_tmp3 = hessian_tmp0 + hessian_tmp2;
  const s_t hessian_tmp4 = pow_2(gu1);
  const s_t hessian_tmp5 = gu3 + s_t(1);
  const s_t hessian_tmp6 = pow_2(hessian_tmp5);
  const s_t hessian_tmp7 = hessian_tmp4 + hessian_tmp6;
  const s_t hessian_tmp8 = hessian_tmp3 + hessian_tmp7;
  const s_t hessian_tmp9 = hessian_tmp8 + s_t(1);
  const s_t hessian_tmp10 = gu1*gu2;
  const s_t hessian_tmp11 = hessian_tmp1*hessian_tmp5 - hessian_tmp10;
  const s_t hessian_tmp12 = pow(hessian_tmp11, (s_t(-8) / s_t(3)));
  const s_t hessian_tmp13 = ((s_t(10) / s_t(9)))*hessian_tmp12*hessian_tmp9;
  const s_t hessian_tmp14 = s_t(2)/pow(hessian_tmp11, (s_t(2) / s_t(3)));
  const s_t hessian_tmp15 = pow(hessian_tmp11, (s_t(-5) / s_t(3)));
  const s_t hessian_tmp16 = ((s_t(8) / s_t(3)))*hessian_tmp15;
  const s_t hessian_tmp17 = hessian_tmp10*hessian_tmp16 + hessian_tmp14;
  const s_t hessian_tmp18 = pow(hessian_tmp11, (s_t(-4) / s_t(3)));
  const s_t hessian_tmp19 = s_t(2)*gu1;
  const s_t hessian_tmp20 = gu1*hessian_tmp1;
  const s_t hessian_tmp21 = gu2*hessian_tmp5;
  const s_t hessian_tmp22 = hessian_tmp20 + hessian_tmp21;
  const s_t hessian_tmp23 = s_t(2)*hessian_tmp1;
  const s_t hessian_tmp24 = s_t(2)*gu1*hessian_tmp8 + s_t(2)*gu1 - hessian_tmp19*hessian_tmp7 - hessian_tmp22*hessian_tmp23;
  const s_t hessian_tmp25 = pow(hessian_tmp11, (s_t(-7) / s_t(3)));
  const s_t hessian_tmp26 = ((s_t(8) / s_t(3)))*hessian_tmp25;
  const s_t hessian_tmp27 = -pow_2(hessian_tmp22) - (s_t(1) / s_t(2))*pow_2(hessian_tmp3) - (s_t(1) / s_t(2))*pow_2(hessian_tmp7) + ((s_t(1) / s_t(2)))*pow_2(hessian_tmp8) + hessian_tmp8;
  const s_t hessian_tmp28 = pow(hessian_tmp11, (s_t(-10) / s_t(3)));
  const s_t hessian_tmp29 = ((s_t(28) / s_t(9)))*hessian_tmp27*hessian_tmp28;
  const s_t hessian_tmp30 = K*hessian_tmp0 + iso_history_qp[1] + iso_relaxation_gamma*(C01*(gu2*hessian_tmp24*hessian_tmp26 + hessian_tmp0*hessian_tmp29 + hessian_tmp18*(s_t(2)*hessian_tmp0 + s_t(2))) + C10*(hessian_tmp0*hessian_tmp13 + hessian_tmp17));
  const s_t hessian_tmp31 = idet*(-adj_lane1 - adj_lane3);
  const s_t hessian_tmp32 = gu1*hessian_tmp5;
  const s_t hessian_tmp33 = ((s_t(4) / s_t(3)))*hessian_tmp15;
  const s_t hessian_tmp34 = hessian_tmp32*hessian_tmp33;
  const s_t hessian_tmp35 = gu2*hessian_tmp1;
  const s_t hessian_tmp36 = hessian_tmp33*hessian_tmp35;
  const s_t hessian_tmp37 = s_t(2)*hessian_tmp18;
  const s_t hessian_tmp38 = ((s_t(4) / s_t(3)))*hessian_tmp25;
  const s_t hessian_tmp39 = hessian_tmp38*hessian_tmp5;
  const s_t hessian_tmp40 = s_t(2)*gu0 - hessian_tmp19*hessian_tmp22 - hessian_tmp23*hessian_tmp3 + hessian_tmp23*hessian_tmp8 + s_t(2);
  const s_t hessian_tmp41 = -K*hessian_tmp21 + iso_history_qp[2] + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu2*hessian_tmp25*hessian_tmp40 - hessian_tmp21*hessian_tmp29 - hessian_tmp21*hessian_tmp37 - hessian_tmp24*hessian_tmp39) + C10*(-hessian_tmp13*hessian_tmp21 - hessian_tmp34 + hessian_tmp36));
  const s_t hessian_tmp42 = idet*(-adj_lane0 - adj_lane2);
  const s_t hessian_tmp43 = ((s_t(1) / s_t(2)))*hessian_tmp30*hessian_tmp31 + ((s_t(1) / s_t(2)))*hessian_tmp41*hessian_tmp42;
  const s_t hessian_tmp44 = hessian_tmp1*hessian_tmp5;
  const s_t hessian_tmp45 = hessian_tmp14 - hessian_tmp16*hessian_tmp44;
  const s_t hessian_tmp46 = K*hessian_tmp6 + iso_history_qp[0] + iso_relaxation_gamma*(C01*(hessian_tmp18*(s_t(2)*hessian_tmp6 + s_t(2)) - hessian_tmp26*hessian_tmp40*hessian_tmp5 + hessian_tmp29*hessian_tmp6) + C10*(hessian_tmp13*hessian_tmp6 + hessian_tmp45));
  const s_t hessian_tmp47 = ((s_t(1) / s_t(2)))*hessian_tmp31*hessian_tmp41 + ((s_t(1) / s_t(2)))*hessian_tmp42*hessian_tmp46;
  const s_t hessian_tmp48 = adj_lane0*hessian_tmp47 + adj_lane1*hessian_tmp43;
  const s_t hessian_tmp49 = adj_lane2*hessian_tmp47 + adj_lane3*hessian_tmp43;
  const s_t hessian_tmp50 = hessian_tmp20*hessian_tmp33;
  const s_t hessian_tmp51 = hessian_tmp21*hessian_tmp33;
  const s_t hessian_tmp52 = s_t(2)*gu2;
  const s_t hessian_tmp53 = s_t(2)*hessian_tmp5;
  const s_t hessian_tmp54 = s_t(2)*gu2*hessian_tmp8 + s_t(2)*gu2 - hessian_tmp22*hessian_tmp53 - hessian_tmp3*hessian_tmp52;
  const s_t hessian_tmp55 = -K*hessian_tmp32 + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu1*hessian_tmp25*hessian_tmp40 - hessian_tmp18*hessian_tmp19*hessian_tmp5 - hessian_tmp29*hessian_tmp32 - hessian_tmp39*hessian_tmp54) + C10*(-hessian_tmp13*hessian_tmp32 + hessian_tmp50 - hessian_tmp51));
  const s_t hessian_tmp56 = hessian_tmp42*hessian_tmp55;
  const s_t hessian_tmp57 = K*(gu0*gu3 + gu0 + gu3 - hessian_tmp10);
  const s_t hessian_tmp58 = ((s_t(2) / s_t(3)))*hessian_tmp15*hessian_tmp9;
  const s_t hessian_tmp59 = hessian_tmp24*hessian_tmp38;
  const s_t hessian_tmp60 = hessian_tmp27*hessian_tmp38;
  const s_t hessian_tmp61 = K*hessian_tmp10 - hessian_tmp57 + iso_relaxation_gamma*(C01*(gu1*hessian_tmp59 + gu2*hessian_tmp38*hessian_tmp54 + hessian_tmp10*hessian_tmp29 + hessian_tmp18*(s_t(4)*hessian_tmp10 - s_t(2)*hessian_tmp44) + hessian_tmp60) + C10*(hessian_tmp0*hessian_tmp33 + hessian_tmp10*hessian_tmp13 + hessian_tmp33*hessian_tmp4 + hessian_tmp58));
  const s_t hessian_tmp62 = ((s_t(1) / s_t(2)))*hessian_tmp31*hessian_tmp61 + ((s_t(1) / s_t(2)))*hessian_tmp56;
  const s_t hessian_tmp63 = s_t(2)*gu3 - hessian_tmp22*hessian_tmp52 - hessian_tmp53*hessian_tmp7 + hessian_tmp53*hessian_tmp8 + s_t(2);
  const s_t hessian_tmp64 = -K*hessian_tmp35 + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu2*hessian_tmp25*hessian_tmp63 - hessian_tmp1*hessian_tmp18*hessian_tmp52 - hessian_tmp1*hessian_tmp59 - hessian_tmp29*hessian_tmp35) + C10*(-hessian_tmp13*hessian_tmp35 - hessian_tmp50 + hessian_tmp51));
  const s_t hessian_tmp65 = hessian_tmp31*hessian_tmp64;
  const s_t hessian_tmp66 = K*hessian_tmp44 + hessian_tmp57 + iso_relaxation_gamma*(C01*(((s_t(28) / s_t(9)))*hessian_tmp1*hessian_tmp27*hessian_tmp28*hessian_tmp5 - hessian_tmp1*hessian_tmp38*hessian_tmp40 + hessian_tmp18*(s_t(4)*hessian_tmp1*hessian_tmp5 - s_t(2)*hessian_tmp10) - hessian_tmp39*hessian_tmp63 - hessian_tmp60) + C10*(((s_t(10) / s_t(9)))*hessian_tmp1*hessian_tmp12*hessian_tmp5*hessian_tmp9 - hessian_tmp2*hessian_tmp33 - hessian_tmp33*hessian_tmp6 - hessian_tmp58));
  const s_t hessian_tmp67 = ((s_t(1) / s_t(2)))*hessian_tmp42*hessian_tmp66 + ((s_t(1) / s_t(2)))*hessian_tmp65;
  const s_t hessian_tmp68 = adj_lane0*hessian_tmp62 + adj_lane1*hessian_tmp67;
  const s_t hessian_tmp69 = adj_lane2*hessian_tmp62 + adj_lane3*hessian_tmp67;
  const s_t hessian_tmp70 = adj_lane1*idet;
  const s_t hessian_tmp71 = adj_lane0*idet;
  const s_t hessian_tmp72 = ((s_t(1) / s_t(2)))*hessian_tmp30*hessian_tmp70 + ((s_t(1) / s_t(2)))*hessian_tmp41*hessian_tmp71;
  const s_t hessian_tmp73 = ((s_t(1) / s_t(2)))*hessian_tmp41*hessian_tmp70 + ((s_t(1) / s_t(2)))*hessian_tmp46*hessian_tmp71;
  const s_t hessian_tmp74 = adj_lane0*hessian_tmp73 + adj_lane1*hessian_tmp72;
  const s_t hessian_tmp75 = adj_lane2*hessian_tmp73 + adj_lane3*hessian_tmp72;
  const s_t hessian_tmp76 = hessian_tmp55*hessian_tmp71;
  const s_t hessian_tmp77 = ((s_t(1) / s_t(2)))*hessian_tmp61*hessian_tmp70 + ((s_t(1) / s_t(2)))*hessian_tmp76;
  const s_t hessian_tmp78 = hessian_tmp64*hessian_tmp70;
  const s_t hessian_tmp79 = ((s_t(1) / s_t(2)))*hessian_tmp66*hessian_tmp71 + ((s_t(1) / s_t(2)))*hessian_tmp78;
  const s_t hessian_tmp80 = adj_lane0*hessian_tmp77 + adj_lane1*hessian_tmp79;
  const s_t hessian_tmp81 = adj_lane2*hessian_tmp77 + adj_lane3*hessian_tmp79;
  const s_t hessian_tmp82 = adj_lane3*idet;
  const s_t hessian_tmp83 = adj_lane2*idet;
  const s_t hessian_tmp84 = ((s_t(1) / s_t(2)))*hessian_tmp30*hessian_tmp82 + ((s_t(1) / s_t(2)))*hessian_tmp41*hessian_tmp83;
  const s_t hessian_tmp85 = ((s_t(1) / s_t(2)))*hessian_tmp41*hessian_tmp82 + ((s_t(1) / s_t(2)))*hessian_tmp46*hessian_tmp83;
  const s_t hessian_tmp86 = adj_lane0*hessian_tmp85 + adj_lane1*hessian_tmp84;
  const s_t hessian_tmp87 = adj_lane2*hessian_tmp85 + adj_lane3*hessian_tmp84;
  const s_t hessian_tmp88 = hessian_tmp55*hessian_tmp83;
  const s_t hessian_tmp89 = ((s_t(1) / s_t(2)))*hessian_tmp61*hessian_tmp82 + ((s_t(1) / s_t(2)))*hessian_tmp88;
  const s_t hessian_tmp90 = hessian_tmp64*hessian_tmp82;
  const s_t hessian_tmp91 = ((s_t(1) / s_t(2)))*hessian_tmp66*hessian_tmp83 + ((s_t(1) / s_t(2)))*hessian_tmp90;
  const s_t hessian_tmp92 = adj_lane0*hessian_tmp89 + adj_lane1*hessian_tmp91;
  const s_t hessian_tmp93 = adj_lane2*hessian_tmp89 + adj_lane3*hessian_tmp91;
  const s_t hessian_tmp94 = ((s_t(1) / s_t(2)))*hessian_tmp42*hessian_tmp61 + ((s_t(1) / s_t(2)))*hessian_tmp65;
  const s_t hessian_tmp95 = ((s_t(1) / s_t(2)))*hessian_tmp31*hessian_tmp66 + ((s_t(1) / s_t(2)))*hessian_tmp56;
  const s_t hessian_tmp96 = adj_lane0*hessian_tmp95 + adj_lane1*hessian_tmp94;
  const s_t hessian_tmp97 = adj_lane2*hessian_tmp95 + adj_lane3*hessian_tmp94;
  const s_t hessian_tmp98 = K*hessian_tmp4 + iso_history_qp[0] + iso_relaxation_gamma*(C01*(gu1*hessian_tmp26*hessian_tmp54 + hessian_tmp18*(s_t(2)*hessian_tmp4 + s_t(2)) + hessian_tmp29*hessian_tmp4) + C10*(hessian_tmp13*hessian_tmp4 + hessian_tmp17));
  const s_t hessian_tmp99 = -K*hessian_tmp20 + iso_history_qp[2] + iso_relaxation_gamma*(C01*(((s_t(4) / s_t(3)))*gu1*hessian_tmp25*hessian_tmp63 - hessian_tmp1*hessian_tmp38*hessian_tmp54 - hessian_tmp20*hessian_tmp29 - hessian_tmp20*hessian_tmp37) + C10*(-hessian_tmp13*hessian_tmp20 + hessian_tmp34 - hessian_tmp36));
  const s_t hessian_tmp100 = ((s_t(1) / s_t(2)))*hessian_tmp31*hessian_tmp99 + ((s_t(1) / s_t(2)))*hessian_tmp42*hessian_tmp98;
  const s_t hessian_tmp101 = K*hessian_tmp2 + iso_history_qp[1] + iso_relaxation_gamma*(C01*(-hessian_tmp1*hessian_tmp26*hessian_tmp63 + hessian_tmp18*(s_t(2)*hessian_tmp2 + s_t(2)) + hessian_tmp2*hessian_tmp29) + C10*(hessian_tmp13*hessian_tmp2 + hessian_tmp45));
  const s_t hessian_tmp102 = ((s_t(1) / s_t(2)))*hessian_tmp101*hessian_tmp31 + ((s_t(1) / s_t(2)))*hessian_tmp42*hessian_tmp99;
  const s_t hessian_tmp103 = adj_lane0*hessian_tmp100 + adj_lane1*hessian_tmp102;
  const s_t hessian_tmp104 = adj_lane2*hessian_tmp100 + adj_lane3*hessian_tmp102;
  const s_t hessian_tmp105 = ((s_t(1) / s_t(2)))*hessian_tmp61*hessian_tmp71 + ((s_t(1) / s_t(2)))*hessian_tmp78;
  const s_t hessian_tmp106 = ((s_t(1) / s_t(2)))*hessian_tmp66*hessian_tmp70 + ((s_t(1) / s_t(2)))*hessian_tmp76;
  const s_t hessian_tmp107 = adj_lane0*hessian_tmp106 + adj_lane1*hessian_tmp105;
  const s_t hessian_tmp108 = adj_lane2*hessian_tmp106 + adj_lane3*hessian_tmp105;
  const s_t hessian_tmp109 = ((s_t(1) / s_t(2)))*hessian_tmp70*hessian_tmp99 + ((s_t(1) / s_t(2)))*hessian_tmp71*hessian_tmp98;
  const s_t hessian_tmp110 = ((s_t(1) / s_t(2)))*hessian_tmp101*hessian_tmp70 + ((s_t(1) / s_t(2)))*hessian_tmp71*hessian_tmp99;
  const s_t hessian_tmp111 = adj_lane0*hessian_tmp109 + adj_lane1*hessian_tmp110;
  const s_t hessian_tmp112 = adj_lane2*hessian_tmp109 + adj_lane3*hessian_tmp110;
  const s_t hessian_tmp113 = ((s_t(1) / s_t(2)))*hessian_tmp61*hessian_tmp83 + ((s_t(1) / s_t(2)))*hessian_tmp90;
  const s_t hessian_tmp114 = ((s_t(1) / s_t(2)))*hessian_tmp66*hessian_tmp82 + ((s_t(1) / s_t(2)))*hessian_tmp88;
  const s_t hessian_tmp115 = adj_lane0*hessian_tmp114 + adj_lane1*hessian_tmp113;
  const s_t hessian_tmp116 = adj_lane2*hessian_tmp114 + adj_lane3*hessian_tmp113;
  const s_t hessian_tmp117 = ((s_t(1) / s_t(2)))*hessian_tmp82*hessian_tmp99 + ((s_t(1) / s_t(2)))*hessian_tmp83*hessian_tmp98;
  const s_t hessian_tmp118 = ((s_t(1) / s_t(2)))*hessian_tmp101*hessian_tmp82 + ((s_t(1) / s_t(2)))*hessian_tmp83*hessian_tmp99;
  const s_t hessian_tmp119 = adj_lane0*hessian_tmp117 + adj_lane1*hessian_tmp118;
  const s_t hessian_tmp120 = adj_lane2*hessian_tmp117 + adj_lane3*hessian_tmp118;
  element_matrix[0] = -hessian_tmp48 - hessian_tmp49;
  element_matrix[6] = hessian_tmp48;
  element_matrix[12] = hessian_tmp49;
  element_matrix[18] = -hessian_tmp68 - hessian_tmp69;
  element_matrix[24] = hessian_tmp68;
  element_matrix[30] = hessian_tmp69;
  element_matrix[1] = -hessian_tmp74 - hessian_tmp75;
  element_matrix[7] = hessian_tmp74;
  element_matrix[13] = hessian_tmp75;
  element_matrix[19] = -hessian_tmp80 - hessian_tmp81;
  element_matrix[25] = hessian_tmp80;
  element_matrix[31] = hessian_tmp81;
  element_matrix[2] = -hessian_tmp86 - hessian_tmp87;
  element_matrix[8] = hessian_tmp86;
  element_matrix[14] = hessian_tmp87;
  element_matrix[20] = -hessian_tmp92 - hessian_tmp93;
  element_matrix[26] = hessian_tmp92;
  element_matrix[32] = hessian_tmp93;
  element_matrix[3] = -hessian_tmp96 - hessian_tmp97;
  element_matrix[9] = hessian_tmp96;
  element_matrix[15] = hessian_tmp97;
  element_matrix[21] = -hessian_tmp103 - hessian_tmp104;
  element_matrix[27] = hessian_tmp103;
  element_matrix[33] = hessian_tmp104;
  element_matrix[4] = -hessian_tmp107 - hessian_tmp108;
  element_matrix[10] = hessian_tmp107;
  element_matrix[16] = hessian_tmp108;
  element_matrix[22] = -hessian_tmp111 - hessian_tmp112;
  element_matrix[28] = hessian_tmp111;
  element_matrix[34] = hessian_tmp112;
  element_matrix[5] = -hessian_tmp115 - hessian_tmp116;
  element_matrix[11] = hessian_tmp115;
  element_matrix[17] = hessian_tmp116;
  element_matrix[23] = -hessian_tmp119 - hessian_tmp120;
  element_matrix[29] = hessian_tmp119;
  element_matrix[35] = hessian_tmp120;
}

} // namespace codegen
} // namespace sfem

#endif
