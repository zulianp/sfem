#ifndef MODIFIED_MOONEY_RIVLIN_D2_SIMPLEX_HESSIAN_HPP
#define MODIFIED_MOONEY_RIVLIN_D2_SIMPLEX_HESSIAN_HPP
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
static SFEM_INLINE void modified_mooney_rivlin_d2_simplex_direct_hessian_reference_element_matrix(
    const s_t *const RSTR badj0,
    const s_t *const RSTR badj1,
    const s_t *const RSTR badj2,
    const s_t *const RSTR badj3,
    const s_t *const RSTR bdet0,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
    const s_t c1,
    const s_t c2,
    const s_t kappa,
    const s_t bu_data[NS * 2][VS],
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
    s_t gu_ref0 = s_t(0);
    s_t gu_ref1 = s_t(0);
    s_t gu_ref2 = s_t(0);
    s_t gu_ref3 = s_t(0);
    for (int shape = 0; shape < NS; ++shape) {
      const s_t state_grad_ref0 = grad_ref_x[q * NS + shape];
      const s_t state_grad_ref1 = grad_ref_y[q * NS + shape];
      const s_t state_u0 = bu_data[shape * NC][lane];
      gu_ref0 += state_u0 * state_grad_ref0;
      gu_ref1 += state_u0 * state_grad_ref1;
      const s_t state_u1 = bu_data[shape * NC + 1][lane];
      gu_ref2 += state_u1 * state_grad_ref0;
      gu_ref3 += state_u1 * state_grad_ref1;
    }
    const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane2) * idet;
    const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane3) * idet;
    const s_t gu2 = (gu_ref2 * adj_lane0 + gu_ref3 * adj_lane2) * idet;
    const s_t gu3 = (gu_ref2 * adj_lane1 + gu_ref3 * adj_lane3) * idet;
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
        const s_t weak_hess_tmp0 = gu3 + s_t(1);
        const s_t weak_hess_tmp1 = pow_2(weak_hess_tmp0);
        const s_t weak_hess_tmp2 = gu1*gu2;
        const s_t weak_hess_tmp3 = gu0 + s_t(1);
        const s_t weak_hess_tmp4 = weak_hess_tmp0*weak_hess_tmp3 - weak_hess_tmp2;
        const s_t weak_hess_tmp5 = kappa/pow_2(weak_hess_tmp4);
        const s_t weak_hess_tmp6 = weak_hess_tmp1*weak_hess_tmp5;
        const s_t weak_hess_tmp7 = sfem_log1p(gu0*gu3 + gu0 + gu3 - weak_hess_tmp2);
        const s_t weak_hess_tmp8 = pow_2(gu2);
        const s_t weak_hess_tmp9 = pow_2(weak_hess_tmp3);
        const s_t weak_hess_tmp10 = weak_hess_tmp8 + weak_hess_tmp9;
        const s_t weak_hess_tmp11 = pow_2(gu1);
        const s_t weak_hess_tmp12 = weak_hess_tmp1 + weak_hess_tmp11;
        const s_t weak_hess_tmp13 = weak_hess_tmp10 + weak_hess_tmp12;
        const s_t weak_hess_tmp14 = weak_hess_tmp13 + s_t(1);
        const s_t weak_hess_tmp15 = pow(weak_hess_tmp4, (s_t(-8) / s_t(3)));
        const s_t weak_hess_tmp16 = ((s_t(10) / s_t(9)))*weak_hess_tmp14*weak_hess_tmp15;
        const s_t weak_hess_tmp17 = s_t(2)/pow(weak_hess_tmp4, (s_t(2) / s_t(3)));
        const s_t weak_hess_tmp18 = weak_hess_tmp0*weak_hess_tmp3;
        const s_t weak_hess_tmp19 = pow(weak_hess_tmp4, (s_t(-5) / s_t(3)));
        const s_t weak_hess_tmp20 = ((s_t(8) / s_t(3)))*weak_hess_tmp19;
        const s_t weak_hess_tmp21 = weak_hess_tmp17 - weak_hess_tmp18*weak_hess_tmp20;
        const s_t weak_hess_tmp22 = pow(weak_hess_tmp4, (s_t(-4) / s_t(3)));
        const s_t weak_hess_tmp23 = gu1*weak_hess_tmp3;
        const s_t weak_hess_tmp24 = gu2*weak_hess_tmp0;
        const s_t weak_hess_tmp25 = weak_hess_tmp23 + weak_hess_tmp24;
        const s_t weak_hess_tmp26 = s_t(2)*gu1;
        const s_t weak_hess_tmp27 = s_t(2)*weak_hess_tmp3;
        const s_t weak_hess_tmp28 = s_t(2)*gu0 - weak_hess_tmp10*weak_hess_tmp27 + weak_hess_tmp13*weak_hess_tmp27 - weak_hess_tmp25*weak_hess_tmp26 + s_t(2);
        const s_t weak_hess_tmp29 = pow(weak_hess_tmp4, (s_t(-7) / s_t(3)));
        const s_t weak_hess_tmp30 = ((s_t(8) / s_t(3)))*weak_hess_tmp29;
        const s_t weak_hess_tmp31 = -(s_t(1) / s_t(2))*pow_2(weak_hess_tmp10) - (s_t(1) / s_t(2))*pow_2(weak_hess_tmp12) + ((s_t(1) / s_t(2)))*pow_2(weak_hess_tmp13) + weak_hess_tmp13 - pow_2(weak_hess_tmp25);
        const s_t weak_hess_tmp32 = pow(weak_hess_tmp4, (s_t(-10) / s_t(3)));
        const s_t weak_hess_tmp33 = ((s_t(28) / s_t(9)))*weak_hess_tmp31*weak_hess_tmp32;
        const s_t weak_hess_tmp34 = weak_hess_tmp24*weak_hess_tmp5;
        const s_t weak_hess_tmp35 = ((s_t(4) / s_t(3)))*weak_hess_tmp19;
        const s_t weak_hess_tmp36 = gu1*weak_hess_tmp0;
        const s_t weak_hess_tmp37 = weak_hess_tmp35*weak_hess_tmp36;
        const s_t weak_hess_tmp38 = gu2*weak_hess_tmp3;
        const s_t weak_hess_tmp39 = weak_hess_tmp35*weak_hess_tmp38;
        const s_t weak_hess_tmp40 = s_t(2)*weak_hess_tmp22;
        const s_t weak_hess_tmp41 = s_t(2)*gu1*weak_hess_tmp13 + s_t(2)*gu1 - weak_hess_tmp12*weak_hess_tmp26 - weak_hess_tmp25*weak_hess_tmp27;
        const s_t weak_hess_tmp42 = ((s_t(4) / s_t(3)))*weak_hess_tmp29;
        const s_t weak_hess_tmp43 = weak_hess_tmp0*weak_hess_tmp42;
        const s_t weak_hess_tmp44 = c1*(-weak_hess_tmp16*weak_hess_tmp24 - weak_hess_tmp37 + weak_hess_tmp39) + c2*(((s_t(4) / s_t(3)))*gu2*weak_hess_tmp28*weak_hess_tmp29 - weak_hess_tmp24*weak_hess_tmp33 - weak_hess_tmp24*weak_hess_tmp40 - weak_hess_tmp41*weak_hess_tmp43) + weak_hess_tmp34*weak_hess_tmp7 - weak_hess_tmp34;
        const s_t weak_hess_tmp45 = weak_hess_tmp36*weak_hess_tmp5;
        const s_t weak_hess_tmp46 = weak_hess_tmp23*weak_hess_tmp35;
        const s_t weak_hess_tmp47 = weak_hess_tmp24*weak_hess_tmp35;
        const s_t weak_hess_tmp48 = s_t(2)*gu2;
        const s_t weak_hess_tmp49 = s_t(2)*weak_hess_tmp0;
        const s_t weak_hess_tmp50 = s_t(2)*gu2*weak_hess_tmp13 + s_t(2)*gu2 - weak_hess_tmp10*weak_hess_tmp48 - weak_hess_tmp25*weak_hess_tmp49;
        const s_t weak_hess_tmp51 = c1*(-weak_hess_tmp16*weak_hess_tmp36 + weak_hess_tmp46 - weak_hess_tmp47) + c2*(((s_t(4) / s_t(3)))*gu1*weak_hess_tmp28*weak_hess_tmp29 - weak_hess_tmp0*weak_hess_tmp22*weak_hess_tmp26 - weak_hess_tmp33*weak_hess_tmp36 - weak_hess_tmp43*weak_hess_tmp50) + weak_hess_tmp45*weak_hess_tmp7 - weak_hess_tmp45;
        const s_t weak_hess_tmp52 = weak_hess_tmp18*weak_hess_tmp5;
        const s_t weak_hess_tmp53 = kappa*weak_hess_tmp7/weak_hess_tmp4;
        const s_t weak_hess_tmp54 = ((s_t(2) / s_t(3)))*weak_hess_tmp14*weak_hess_tmp19;
        const s_t weak_hess_tmp55 = s_t(2)*gu3 - weak_hess_tmp12*weak_hess_tmp49 + weak_hess_tmp13*weak_hess_tmp49 - weak_hess_tmp25*weak_hess_tmp48 + s_t(2);
        const s_t weak_hess_tmp56 = weak_hess_tmp31*weak_hess_tmp42;
        const s_t weak_hess_tmp57 = c1*(((s_t(10) / s_t(9)))*weak_hess_tmp0*weak_hess_tmp14*weak_hess_tmp15*weak_hess_tmp3 - weak_hess_tmp1*weak_hess_tmp35 - weak_hess_tmp35*weak_hess_tmp9 - weak_hess_tmp54) + c2*(((s_t(28) / s_t(9)))*weak_hess_tmp0*weak_hess_tmp3*weak_hess_tmp31*weak_hess_tmp32 + weak_hess_tmp22*(s_t(4)*weak_hess_tmp0*weak_hess_tmp3 - s_t(2)*weak_hess_tmp2) - weak_hess_tmp28*weak_hess_tmp3*weak_hess_tmp42 - weak_hess_tmp43*weak_hess_tmp55 - weak_hess_tmp56) - weak_hess_tmp52*weak_hess_tmp7 + weak_hess_tmp52 + weak_hess_tmp53;
        const s_t weak_hess_tmp58 = weak_hess_tmp5*weak_hess_tmp8;
        const s_t weak_hess_tmp59 = weak_hess_tmp17 + weak_hess_tmp2*weak_hess_tmp20;
        const s_t weak_hess_tmp60 = weak_hess_tmp38*weak_hess_tmp5;
        const s_t weak_hess_tmp61 = weak_hess_tmp41*weak_hess_tmp42;
        const s_t weak_hess_tmp62 = c1*(-weak_hess_tmp16*weak_hess_tmp38 - weak_hess_tmp46 + weak_hess_tmp47) + c2*(((s_t(4) / s_t(3)))*gu2*weak_hess_tmp29*weak_hess_tmp55 - weak_hess_tmp22*weak_hess_tmp3*weak_hess_tmp48 - weak_hess_tmp3*weak_hess_tmp61 - weak_hess_tmp33*weak_hess_tmp38) + weak_hess_tmp60*weak_hess_tmp7 - weak_hess_tmp60;
        const s_t weak_hess_tmp63 = weak_hess_tmp2*weak_hess_tmp5;
        const s_t weak_hess_tmp64 = c1*(weak_hess_tmp11*weak_hess_tmp35 + weak_hess_tmp16*weak_hess_tmp2 + weak_hess_tmp35*weak_hess_tmp8 + weak_hess_tmp54) + c2*(gu1*weak_hess_tmp61 + gu2*weak_hess_tmp42*weak_hess_tmp50 + weak_hess_tmp2*weak_hess_tmp33 + weak_hess_tmp22*(-s_t(2)*weak_hess_tmp18 + s_t(4)*weak_hess_tmp2) + weak_hess_tmp56) - weak_hess_tmp53 - weak_hess_tmp63*weak_hess_tmp7 + weak_hess_tmp63;
        const s_t weak_hess_tmp65 = weak_hess_tmp11*weak_hess_tmp5;
        const s_t weak_hess_tmp66 = weak_hess_tmp23*weak_hess_tmp5;
        const s_t weak_hess_tmp67 = c1*(-weak_hess_tmp16*weak_hess_tmp23 + weak_hess_tmp37 - weak_hess_tmp39) + c2*(((s_t(4) / s_t(3)))*gu1*weak_hess_tmp29*weak_hess_tmp55 - weak_hess_tmp23*weak_hess_tmp33 - weak_hess_tmp23*weak_hess_tmp40 - weak_hess_tmp3*weak_hess_tmp42*weak_hess_tmp50) + weak_hess_tmp66*weak_hess_tmp7 - weak_hess_tmp66;
        const s_t weak_hess_tmp68 = weak_hess_tmp5*weak_hess_tmp9;
        material[0] = trial_grad[0]*(c1*(weak_hess_tmp1*weak_hess_tmp16 + weak_hess_tmp21) + c2*(-weak_hess_tmp0*weak_hess_tmp28*weak_hess_tmp30 + weak_hess_tmp1*weak_hess_tmp33 + weak_hess_tmp22*(s_t(2)*weak_hess_tmp1 + s_t(2))) - weak_hess_tmp6*weak_hess_tmp7 + weak_hess_tmp6) + trial_grad[1]*weak_hess_tmp44 + trial_grad[2]*weak_hess_tmp51 + trial_grad[3]*weak_hess_tmp57;
        material[1] = trial_grad[0]*weak_hess_tmp44 + trial_grad[1]*(c1*(weak_hess_tmp16*weak_hess_tmp8 + weak_hess_tmp59) + c2*(gu2*weak_hess_tmp30*weak_hess_tmp41 + weak_hess_tmp22*(s_t(2)*weak_hess_tmp8 + s_t(2)) + weak_hess_tmp33*weak_hess_tmp8) - weak_hess_tmp58*weak_hess_tmp7 + weak_hess_tmp58) + trial_grad[2]*weak_hess_tmp64 + trial_grad[3]*weak_hess_tmp62;
        material[2] = trial_grad[0]*weak_hess_tmp51 + trial_grad[1]*weak_hess_tmp64 + trial_grad[2]*(c1*(weak_hess_tmp11*weak_hess_tmp16 + weak_hess_tmp59) + c2*(gu1*weak_hess_tmp30*weak_hess_tmp50 + weak_hess_tmp11*weak_hess_tmp33 + weak_hess_tmp22*(s_t(2)*weak_hess_tmp11 + s_t(2))) - weak_hess_tmp65*weak_hess_tmp7 + weak_hess_tmp65) + trial_grad[3]*weak_hess_tmp67;
        material[3] = trial_grad[0]*weak_hess_tmp57 + trial_grad[1]*weak_hess_tmp62 + trial_grad[2]*weak_hess_tmp67 + trial_grad[3]*(c1*(weak_hess_tmp16*weak_hess_tmp9 + weak_hess_tmp21) + c2*(weak_hess_tmp22*(s_t(2)*weak_hess_tmp9 + s_t(2)) - weak_hess_tmp3*weak_hess_tmp30*weak_hess_tmp55 + weak_hess_tmp33*weak_hess_tmp9) - weak_hess_tmp68*weak_hess_tmp7 + weak_hess_tmp68);
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
static SFEM_INLINE void modified_mooney_rivlin_d2_simplex_tri3_direct_hessian_element_matrix(
    const s_t *const RSTR badj0,
    const s_t *const RSTR badj1,
    const s_t *const RSTR badj2,
    const s_t *const RSTR badj3,
    const s_t *const RSTR bdet0,
    const s_t c1,
    const s_t c2,
    const s_t kappa,
    const s_t bu_data[NS * 2][VS],
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
  const s_t gu_ref0 = -bu_data[0][lane] + bu_data[2][lane];
  const s_t gu_ref1 = -bu_data[0][lane] + bu_data[4][lane];
  const s_t gu_ref2 = -bu_data[1][lane] + bu_data[3][lane];
  const s_t gu_ref3 = -bu_data[1][lane] + bu_data[5][lane];
  const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane2) * idet;
  const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane3) * idet;
  const s_t gu2 = (gu_ref2 * adj_lane0 + gu_ref3 * adj_lane2) * idet;
  const s_t gu3 = (gu_ref2 * adj_lane1 + gu_ref3 * adj_lane3) * idet;
  const s_t hessian_tmp0 = pow_2(gu2);
  const s_t hessian_tmp1 = gu1*gu2;
  const s_t hessian_tmp2 = gu0 + s_t(1);
  const s_t hessian_tmp3 = gu3 + s_t(1);
  const s_t hessian_tmp4 = -hessian_tmp1 + hessian_tmp2*hessian_tmp3;
  const s_t hessian_tmp5 = kappa/pow_2(hessian_tmp4);
  const s_t hessian_tmp6 = hessian_tmp0*hessian_tmp5;
  const s_t hessian_tmp7 = sfem_log1p(gu0*gu3 + gu0 + gu3 - hessian_tmp1);
  const s_t hessian_tmp8 = pow_2(hessian_tmp2);
  const s_t hessian_tmp9 = hessian_tmp0 + hessian_tmp8;
  const s_t hessian_tmp10 = pow_2(gu1);
  const s_t hessian_tmp11 = pow_2(hessian_tmp3);
  const s_t hessian_tmp12 = hessian_tmp10 + hessian_tmp11;
  const s_t hessian_tmp13 = hessian_tmp12 + hessian_tmp9;
  const s_t hessian_tmp14 = hessian_tmp13 + s_t(1);
  const s_t hessian_tmp15 = pow(hessian_tmp4, (s_t(-8) / s_t(3)));
  const s_t hessian_tmp16 = ((s_t(10) / s_t(9)))*hessian_tmp14*hessian_tmp15;
  const s_t hessian_tmp17 = s_t(2)/pow(hessian_tmp4, (s_t(2) / s_t(3)));
  const s_t hessian_tmp18 = pow(hessian_tmp4, (s_t(-5) / s_t(3)));
  const s_t hessian_tmp19 = ((s_t(8) / s_t(3)))*hessian_tmp18;
  const s_t hessian_tmp20 = hessian_tmp1*hessian_tmp19 + hessian_tmp17;
  const s_t hessian_tmp21 = pow(hessian_tmp4, (s_t(-4) / s_t(3)));
  const s_t hessian_tmp22 = s_t(2)*gu1;
  const s_t hessian_tmp23 = gu1*hessian_tmp2;
  const s_t hessian_tmp24 = gu2*hessian_tmp3;
  const s_t hessian_tmp25 = hessian_tmp23 + hessian_tmp24;
  const s_t hessian_tmp26 = s_t(2)*hessian_tmp2;
  const s_t hessian_tmp27 = s_t(2)*gu1*hessian_tmp13 + s_t(2)*gu1 - hessian_tmp12*hessian_tmp22 - hessian_tmp25*hessian_tmp26;
  const s_t hessian_tmp28 = pow(hessian_tmp4, (s_t(-7) / s_t(3)));
  const s_t hessian_tmp29 = ((s_t(8) / s_t(3)))*hessian_tmp28;
  const s_t hessian_tmp30 = -(s_t(1) / s_t(2))*pow_2(hessian_tmp12) + ((s_t(1) / s_t(2)))*pow_2(hessian_tmp13) + hessian_tmp13 - pow_2(hessian_tmp25) - (s_t(1) / s_t(2))*pow_2(hessian_tmp9);
  const s_t hessian_tmp31 = pow(hessian_tmp4, (s_t(-10) / s_t(3)));
  const s_t hessian_tmp32 = ((s_t(28) / s_t(9)))*hessian_tmp30*hessian_tmp31;
  const s_t hessian_tmp33 = c1*(hessian_tmp0*hessian_tmp16 + hessian_tmp20) + c2*(gu2*hessian_tmp27*hessian_tmp29 + hessian_tmp0*hessian_tmp32 + hessian_tmp21*(s_t(2)*hessian_tmp0 + s_t(2))) - hessian_tmp6*hessian_tmp7 + hessian_tmp6;
  const s_t hessian_tmp34 = idet*(-adj_lane1 - adj_lane3);
  const s_t hessian_tmp35 = hessian_tmp24*hessian_tmp5;
  const s_t hessian_tmp36 = ((s_t(4) / s_t(3)))*hessian_tmp18;
  const s_t hessian_tmp37 = gu1*hessian_tmp3;
  const s_t hessian_tmp38 = hessian_tmp36*hessian_tmp37;
  const s_t hessian_tmp39 = gu2*hessian_tmp2;
  const s_t hessian_tmp40 = hessian_tmp36*hessian_tmp39;
  const s_t hessian_tmp41 = s_t(2)*hessian_tmp21;
  const s_t hessian_tmp42 = ((s_t(4) / s_t(3)))*hessian_tmp28;
  const s_t hessian_tmp43 = hessian_tmp3*hessian_tmp42;
  const s_t hessian_tmp44 = s_t(2)*gu0 + hessian_tmp13*hessian_tmp26 - hessian_tmp22*hessian_tmp25 - hessian_tmp26*hessian_tmp9 + s_t(2);
  const s_t hessian_tmp45 = c1*(-hessian_tmp16*hessian_tmp24 - hessian_tmp38 + hessian_tmp40) + c2*(((s_t(4) / s_t(3)))*gu2*hessian_tmp28*hessian_tmp44 - hessian_tmp24*hessian_tmp32 - hessian_tmp24*hessian_tmp41 - hessian_tmp27*hessian_tmp43) + hessian_tmp35*hessian_tmp7 - hessian_tmp35;
  const s_t hessian_tmp46 = idet*(-adj_lane0 - adj_lane2);
  const s_t hessian_tmp47 = ((s_t(1) / s_t(2)))*hessian_tmp33*hessian_tmp34 + ((s_t(1) / s_t(2)))*hessian_tmp45*hessian_tmp46;
  const s_t hessian_tmp48 = hessian_tmp11*hessian_tmp5;
  const s_t hessian_tmp49 = hessian_tmp2*hessian_tmp3;
  const s_t hessian_tmp50 = hessian_tmp17 - hessian_tmp19*hessian_tmp49;
  const s_t hessian_tmp51 = c1*(hessian_tmp11*hessian_tmp16 + hessian_tmp50) + c2*(hessian_tmp11*hessian_tmp32 + hessian_tmp21*(s_t(2)*hessian_tmp11 + s_t(2)) - hessian_tmp29*hessian_tmp3*hessian_tmp44) - hessian_tmp48*hessian_tmp7 + hessian_tmp48;
  const s_t hessian_tmp52 = ((s_t(1) / s_t(2)))*hessian_tmp34*hessian_tmp45 + ((s_t(1) / s_t(2)))*hessian_tmp46*hessian_tmp51;
  const s_t hessian_tmp53 = adj_lane0*hessian_tmp52 + adj_lane1*hessian_tmp47;
  const s_t hessian_tmp54 = adj_lane2*hessian_tmp52 + adj_lane3*hessian_tmp47;
  const s_t hessian_tmp55 = hessian_tmp37*hessian_tmp5;
  const s_t hessian_tmp56 = hessian_tmp23*hessian_tmp36;
  const s_t hessian_tmp57 = hessian_tmp24*hessian_tmp36;
  const s_t hessian_tmp58 = s_t(2)*gu2;
  const s_t hessian_tmp59 = s_t(2)*hessian_tmp3;
  const s_t hessian_tmp60 = s_t(2)*gu2*hessian_tmp13 + s_t(2)*gu2 - hessian_tmp25*hessian_tmp59 - hessian_tmp58*hessian_tmp9;
  const s_t hessian_tmp61 = c1*(-hessian_tmp16*hessian_tmp37 + hessian_tmp56 - hessian_tmp57) + c2*(((s_t(4) / s_t(3)))*gu1*hessian_tmp28*hessian_tmp44 - hessian_tmp21*hessian_tmp22*hessian_tmp3 - hessian_tmp32*hessian_tmp37 - hessian_tmp43*hessian_tmp60) + hessian_tmp55*hessian_tmp7 - hessian_tmp55;
  const s_t hessian_tmp62 = hessian_tmp46*hessian_tmp61;
  const s_t hessian_tmp63 = hessian_tmp1*hessian_tmp5;
  const s_t hessian_tmp64 = hessian_tmp7*kappa/hessian_tmp4;
  const s_t hessian_tmp65 = ((s_t(2) / s_t(3)))*hessian_tmp14*hessian_tmp18;
  const s_t hessian_tmp66 = hessian_tmp27*hessian_tmp42;
  const s_t hessian_tmp67 = hessian_tmp30*hessian_tmp42;
  const s_t hessian_tmp68 = c1*(hessian_tmp0*hessian_tmp36 + hessian_tmp1*hessian_tmp16 + hessian_tmp10*hessian_tmp36 + hessian_tmp65) + c2*(gu1*hessian_tmp66 + gu2*hessian_tmp42*hessian_tmp60 + hessian_tmp1*hessian_tmp32 + hessian_tmp21*(s_t(4)*hessian_tmp1 - s_t(2)*hessian_tmp49) + hessian_tmp67) - hessian_tmp63*hessian_tmp7 + hessian_tmp63 - hessian_tmp64;
  const s_t hessian_tmp69 = ((s_t(1) / s_t(2)))*hessian_tmp34*hessian_tmp68 + ((s_t(1) / s_t(2)))*hessian_tmp62;
  const s_t hessian_tmp70 = hessian_tmp39*hessian_tmp5;
  const s_t hessian_tmp71 = s_t(2)*gu3 - hessian_tmp12*hessian_tmp59 + hessian_tmp13*hessian_tmp59 - hessian_tmp25*hessian_tmp58 + s_t(2);
  const s_t hessian_tmp72 = c1*(-hessian_tmp16*hessian_tmp39 - hessian_tmp56 + hessian_tmp57) + c2*(((s_t(4) / s_t(3)))*gu2*hessian_tmp28*hessian_tmp71 - hessian_tmp2*hessian_tmp21*hessian_tmp58 - hessian_tmp2*hessian_tmp66 - hessian_tmp32*hessian_tmp39) + hessian_tmp7*hessian_tmp70 - hessian_tmp70;
  const s_t hessian_tmp73 = hessian_tmp34*hessian_tmp72;
  const s_t hessian_tmp74 = hessian_tmp49*hessian_tmp5;
  const s_t hessian_tmp75 = c1*(-hessian_tmp11*hessian_tmp36 + ((s_t(10) / s_t(9)))*hessian_tmp14*hessian_tmp15*hessian_tmp2*hessian_tmp3 - hessian_tmp36*hessian_tmp8 - hessian_tmp65) + c2*(((s_t(28) / s_t(9)))*hessian_tmp2*hessian_tmp3*hessian_tmp30*hessian_tmp31 - hessian_tmp2*hessian_tmp42*hessian_tmp44 + hessian_tmp21*(-s_t(2)*hessian_tmp1 + s_t(4)*hessian_tmp2*hessian_tmp3) - hessian_tmp43*hessian_tmp71 - hessian_tmp67) + hessian_tmp64 - hessian_tmp7*hessian_tmp74 + hessian_tmp74;
  const s_t hessian_tmp76 = ((s_t(1) / s_t(2)))*hessian_tmp46*hessian_tmp75 + ((s_t(1) / s_t(2)))*hessian_tmp73;
  const s_t hessian_tmp77 = adj_lane0*hessian_tmp69 + adj_lane1*hessian_tmp76;
  const s_t hessian_tmp78 = adj_lane2*hessian_tmp69 + adj_lane3*hessian_tmp76;
  const s_t hessian_tmp79 = adj_lane1*idet;
  const s_t hessian_tmp80 = adj_lane0*idet;
  const s_t hessian_tmp81 = ((s_t(1) / s_t(2)))*hessian_tmp33*hessian_tmp79 + ((s_t(1) / s_t(2)))*hessian_tmp45*hessian_tmp80;
  const s_t hessian_tmp82 = ((s_t(1) / s_t(2)))*hessian_tmp45*hessian_tmp79 + ((s_t(1) / s_t(2)))*hessian_tmp51*hessian_tmp80;
  const s_t hessian_tmp83 = adj_lane0*hessian_tmp82 + adj_lane1*hessian_tmp81;
  const s_t hessian_tmp84 = adj_lane2*hessian_tmp82 + adj_lane3*hessian_tmp81;
  const s_t hessian_tmp85 = hessian_tmp61*hessian_tmp80;
  const s_t hessian_tmp86 = ((s_t(1) / s_t(2)))*hessian_tmp68*hessian_tmp79 + ((s_t(1) / s_t(2)))*hessian_tmp85;
  const s_t hessian_tmp87 = hessian_tmp72*hessian_tmp79;
  const s_t hessian_tmp88 = ((s_t(1) / s_t(2)))*hessian_tmp75*hessian_tmp80 + ((s_t(1) / s_t(2)))*hessian_tmp87;
  const s_t hessian_tmp89 = adj_lane0*hessian_tmp86 + adj_lane1*hessian_tmp88;
  const s_t hessian_tmp90 = adj_lane2*hessian_tmp86 + adj_lane3*hessian_tmp88;
  const s_t hessian_tmp91 = adj_lane3*idet;
  const s_t hessian_tmp92 = adj_lane2*idet;
  const s_t hessian_tmp93 = ((s_t(1) / s_t(2)))*hessian_tmp33*hessian_tmp91 + ((s_t(1) / s_t(2)))*hessian_tmp45*hessian_tmp92;
  const s_t hessian_tmp94 = ((s_t(1) / s_t(2)))*hessian_tmp45*hessian_tmp91 + ((s_t(1) / s_t(2)))*hessian_tmp51*hessian_tmp92;
  const s_t hessian_tmp95 = adj_lane0*hessian_tmp94 + adj_lane1*hessian_tmp93;
  const s_t hessian_tmp96 = adj_lane2*hessian_tmp94 + adj_lane3*hessian_tmp93;
  const s_t hessian_tmp97 = hessian_tmp61*hessian_tmp92;
  const s_t hessian_tmp98 = ((s_t(1) / s_t(2)))*hessian_tmp68*hessian_tmp91 + ((s_t(1) / s_t(2)))*hessian_tmp97;
  const s_t hessian_tmp99 = hessian_tmp72*hessian_tmp91;
  const s_t hessian_tmp100 = ((s_t(1) / s_t(2)))*hessian_tmp75*hessian_tmp92 + ((s_t(1) / s_t(2)))*hessian_tmp99;
  const s_t hessian_tmp101 = adj_lane0*hessian_tmp98 + adj_lane1*hessian_tmp100;
  const s_t hessian_tmp102 = adj_lane2*hessian_tmp98 + adj_lane3*hessian_tmp100;
  const s_t hessian_tmp103 = ((s_t(1) / s_t(2)))*hessian_tmp46*hessian_tmp68 + ((s_t(1) / s_t(2)))*hessian_tmp73;
  const s_t hessian_tmp104 = ((s_t(1) / s_t(2)))*hessian_tmp34*hessian_tmp75 + ((s_t(1) / s_t(2)))*hessian_tmp62;
  const s_t hessian_tmp105 = adj_lane0*hessian_tmp104 + adj_lane1*hessian_tmp103;
  const s_t hessian_tmp106 = adj_lane2*hessian_tmp104 + adj_lane3*hessian_tmp103;
  const s_t hessian_tmp107 = hessian_tmp10*hessian_tmp5;
  const s_t hessian_tmp108 = c1*(hessian_tmp10*hessian_tmp16 + hessian_tmp20) + c2*(gu1*hessian_tmp29*hessian_tmp60 + hessian_tmp10*hessian_tmp32 + hessian_tmp21*(s_t(2)*hessian_tmp10 + s_t(2))) - hessian_tmp107*hessian_tmp7 + hessian_tmp107;
  const s_t hessian_tmp109 = hessian_tmp23*hessian_tmp5;
  const s_t hessian_tmp110 = c1*(-hessian_tmp16*hessian_tmp23 + hessian_tmp38 - hessian_tmp40) + c2*(((s_t(4) / s_t(3)))*gu1*hessian_tmp28*hessian_tmp71 - hessian_tmp2*hessian_tmp42*hessian_tmp60 - hessian_tmp23*hessian_tmp32 - hessian_tmp23*hessian_tmp41) + hessian_tmp109*hessian_tmp7 - hessian_tmp109;
  const s_t hessian_tmp111 = ((s_t(1) / s_t(2)))*hessian_tmp108*hessian_tmp46 + ((s_t(1) / s_t(2)))*hessian_tmp110*hessian_tmp34;
  const s_t hessian_tmp112 = hessian_tmp5*hessian_tmp8;
  const s_t hessian_tmp113 = c1*(hessian_tmp16*hessian_tmp8 + hessian_tmp50) + c2*(-hessian_tmp2*hessian_tmp29*hessian_tmp71 + hessian_tmp21*(s_t(2)*hessian_tmp8 + s_t(2)) + hessian_tmp32*hessian_tmp8) - hessian_tmp112*hessian_tmp7 + hessian_tmp112;
  const s_t hessian_tmp114 = ((s_t(1) / s_t(2)))*hessian_tmp110*hessian_tmp46 + ((s_t(1) / s_t(2)))*hessian_tmp113*hessian_tmp34;
  const s_t hessian_tmp115 = adj_lane0*hessian_tmp111 + adj_lane1*hessian_tmp114;
  const s_t hessian_tmp116 = adj_lane2*hessian_tmp111 + adj_lane3*hessian_tmp114;
  const s_t hessian_tmp117 = ((s_t(1) / s_t(2)))*hessian_tmp68*hessian_tmp80 + ((s_t(1) / s_t(2)))*hessian_tmp87;
  const s_t hessian_tmp118 = ((s_t(1) / s_t(2)))*hessian_tmp75*hessian_tmp79 + ((s_t(1) / s_t(2)))*hessian_tmp85;
  const s_t hessian_tmp119 = adj_lane0*hessian_tmp118 + adj_lane1*hessian_tmp117;
  const s_t hessian_tmp120 = adj_lane2*hessian_tmp118 + adj_lane3*hessian_tmp117;
  const s_t hessian_tmp121 = ((s_t(1) / s_t(2)))*hessian_tmp108*hessian_tmp80 + ((s_t(1) / s_t(2)))*hessian_tmp110*hessian_tmp79;
  const s_t hessian_tmp122 = ((s_t(1) / s_t(2)))*hessian_tmp110*hessian_tmp80 + ((s_t(1) / s_t(2)))*hessian_tmp113*hessian_tmp79;
  const s_t hessian_tmp123 = adj_lane0*hessian_tmp121 + adj_lane1*hessian_tmp122;
  const s_t hessian_tmp124 = adj_lane2*hessian_tmp121 + adj_lane3*hessian_tmp122;
  const s_t hessian_tmp125 = ((s_t(1) / s_t(2)))*hessian_tmp68*hessian_tmp92 + ((s_t(1) / s_t(2)))*hessian_tmp99;
  const s_t hessian_tmp126 = ((s_t(1) / s_t(2)))*hessian_tmp75*hessian_tmp91 + ((s_t(1) / s_t(2)))*hessian_tmp97;
  const s_t hessian_tmp127 = adj_lane0*hessian_tmp126 + adj_lane1*hessian_tmp125;
  const s_t hessian_tmp128 = adj_lane2*hessian_tmp126 + adj_lane3*hessian_tmp125;
  const s_t hessian_tmp129 = ((s_t(1) / s_t(2)))*hessian_tmp108*hessian_tmp92 + ((s_t(1) / s_t(2)))*hessian_tmp110*hessian_tmp91;
  const s_t hessian_tmp130 = ((s_t(1) / s_t(2)))*hessian_tmp110*hessian_tmp92 + ((s_t(1) / s_t(2)))*hessian_tmp113*hessian_tmp91;
  const s_t hessian_tmp131 = adj_lane0*hessian_tmp129 + adj_lane1*hessian_tmp130;
  const s_t hessian_tmp132 = adj_lane2*hessian_tmp129 + adj_lane3*hessian_tmp130;
  element_matrix[0] = -hessian_tmp53 - hessian_tmp54;
  element_matrix[6] = hessian_tmp53;
  element_matrix[12] = hessian_tmp54;
  element_matrix[18] = -hessian_tmp77 - hessian_tmp78;
  element_matrix[24] = hessian_tmp77;
  element_matrix[30] = hessian_tmp78;
  element_matrix[1] = -hessian_tmp83 - hessian_tmp84;
  element_matrix[7] = hessian_tmp83;
  element_matrix[13] = hessian_tmp84;
  element_matrix[19] = -hessian_tmp89 - hessian_tmp90;
  element_matrix[25] = hessian_tmp89;
  element_matrix[31] = hessian_tmp90;
  element_matrix[2] = -hessian_tmp95 - hessian_tmp96;
  element_matrix[8] = hessian_tmp95;
  element_matrix[14] = hessian_tmp96;
  element_matrix[20] = -hessian_tmp101 - hessian_tmp102;
  element_matrix[26] = hessian_tmp101;
  element_matrix[32] = hessian_tmp102;
  element_matrix[3] = -hessian_tmp105 - hessian_tmp106;
  element_matrix[9] = hessian_tmp105;
  element_matrix[15] = hessian_tmp106;
  element_matrix[21] = -hessian_tmp115 - hessian_tmp116;
  element_matrix[27] = hessian_tmp115;
  element_matrix[33] = hessian_tmp116;
  element_matrix[4] = -hessian_tmp119 - hessian_tmp120;
  element_matrix[10] = hessian_tmp119;
  element_matrix[16] = hessian_tmp120;
  element_matrix[22] = -hessian_tmp123 - hessian_tmp124;
  element_matrix[28] = hessian_tmp123;
  element_matrix[34] = hessian_tmp124;
  element_matrix[5] = -hessian_tmp127 - hessian_tmp128;
  element_matrix[11] = hessian_tmp127;
  element_matrix[17] = hessian_tmp128;
  element_matrix[23] = -hessian_tmp131 - hessian_tmp132;
  element_matrix[29] = hessian_tmp131;
  element_matrix[35] = hessian_tmp132;
}

} // namespace codegen
} // namespace sfem

#endif
