#ifndef NEOHOOKEAN_OGDEN_D2_SIMPLEX_HESSIAN_HPP
#define NEOHOOKEAN_OGDEN_D2_SIMPLEX_HESSIAN_HPP
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
static SFEM_INLINE void neohookean_ogden_d2_simplex_direct_hessian_reference_element_matrix(
    const s_t *const RSTR badj0,
    const s_t *const RSTR badj1,
    const s_t *const RSTR badj2,
    const s_t *const RSTR badj3,
    const s_t *const RSTR bdet0,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
    const s_t lmbda,
    const s_t mu,
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
        const s_t weak_hess_tmp1 = gu1*gu2;
        const s_t weak_hess_tmp2 = gu0 + s_t(1);
        const s_t weak_hess_tmp3 = weak_hess_tmp0*weak_hess_tmp2 - weak_hess_tmp1;
        const s_t weak_hess_tmp4 = pow_m2(weak_hess_tmp3);
        const s_t weak_hess_tmp5 = weak_hess_tmp0*weak_hess_tmp4;
        const s_t weak_hess_tmp6 = gu2*weak_hess_tmp5;
        const s_t weak_hess_tmp7 = log(weak_hess_tmp3);
        const s_t weak_hess_tmp8 = gu2*lmbda*weak_hess_tmp0*weak_hess_tmp4*weak_hess_tmp7 - lmbda*weak_hess_tmp6 - mu*weak_hess_tmp6;
        const s_t weak_hess_tmp9 = gu1*weak_hess_tmp5;
        const s_t weak_hess_tmp10 = gu1*lmbda*weak_hess_tmp0*weak_hess_tmp4*weak_hess_tmp7 - lmbda*weak_hess_tmp9 - mu*weak_hess_tmp9;
        const s_t weak_hess_tmp11 = pow_2(weak_hess_tmp0)*weak_hess_tmp4;
        const s_t weak_hess_tmp12 = lmbda*weak_hess_tmp11;
        const s_t weak_hess_tmp13 = pow_m1(weak_hess_tmp3);
        const s_t weak_hess_tmp14 = mu*weak_hess_tmp13;
        const s_t weak_hess_tmp15 = weak_hess_tmp0*weak_hess_tmp2*weak_hess_tmp4;
        const s_t weak_hess_tmp16 = lmbda*weak_hess_tmp7;
        const s_t weak_hess_tmp17 = weak_hess_tmp13*weak_hess_tmp16;
        const s_t weak_hess_tmp18 = lmbda*weak_hess_tmp15 + mu*weak_hess_tmp15 - weak_hess_tmp14 - weak_hess_tmp15*weak_hess_tmp16 + weak_hess_tmp17;
        const s_t weak_hess_tmp19 = pow_2(gu2)*weak_hess_tmp4;
        const s_t weak_hess_tmp20 = weak_hess_tmp2*weak_hess_tmp4;
        const s_t weak_hess_tmp21 = gu2*weak_hess_tmp20;
        const s_t weak_hess_tmp22 = gu2*lmbda*weak_hess_tmp2*weak_hess_tmp4*weak_hess_tmp7 - lmbda*weak_hess_tmp21 - mu*weak_hess_tmp21;
        const s_t weak_hess_tmp23 = weak_hess_tmp1*weak_hess_tmp4;
        const s_t weak_hess_tmp24 = lmbda*weak_hess_tmp23 + mu*weak_hess_tmp23 + weak_hess_tmp14 - weak_hess_tmp16*weak_hess_tmp23 - weak_hess_tmp17;
        const s_t weak_hess_tmp25 = pow_2(gu1)*weak_hess_tmp4;
        const s_t weak_hess_tmp26 = gu1*weak_hess_tmp20;
        const s_t weak_hess_tmp27 = gu1*lmbda*weak_hess_tmp2*weak_hess_tmp4*weak_hess_tmp7 - lmbda*weak_hess_tmp26 - mu*weak_hess_tmp26;
        const s_t weak_hess_tmp28 = pow_2(weak_hess_tmp2)*weak_hess_tmp4;
        material[0] = trial_grad[0]*(mu*weak_hess_tmp11 + mu - weak_hess_tmp12*weak_hess_tmp7 + weak_hess_tmp12) + trial_grad[1]*weak_hess_tmp8 + trial_grad[2]*weak_hess_tmp10 + trial_grad[3]*weak_hess_tmp18;
        material[1] = trial_grad[0]*weak_hess_tmp8 + trial_grad[1]*(lmbda*weak_hess_tmp19 + mu*weak_hess_tmp19 + mu - weak_hess_tmp16*weak_hess_tmp19) + trial_grad[2]*weak_hess_tmp24 + trial_grad[3]*weak_hess_tmp22;
        material[2] = trial_grad[0]*weak_hess_tmp10 + trial_grad[1]*weak_hess_tmp24 + trial_grad[2]*(lmbda*weak_hess_tmp25 + mu*weak_hess_tmp25 + mu - weak_hess_tmp16*weak_hess_tmp25) + trial_grad[3]*weak_hess_tmp27;
        material[3] = trial_grad[0]*weak_hess_tmp18 + trial_grad[1]*weak_hess_tmp22 + trial_grad[2]*weak_hess_tmp27 + trial_grad[3]*(lmbda*weak_hess_tmp28 + mu*weak_hess_tmp28 + mu - weak_hess_tmp16*weak_hess_tmp28);
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
static SFEM_INLINE void neohookean_ogden_d2_simplex_tri3_direct_hessian_element_matrix(
    const s_t *const RSTR badj0,
    const s_t *const RSTR badj1,
    const s_t *const RSTR badj2,
    const s_t *const RSTR badj3,
    const s_t *const RSTR bdet0,
    const s_t lmbda,
    const s_t mu,
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
  const s_t hessian_tmp0 = gu1*gu2;
  const s_t hessian_tmp1 = gu0 + s_t(1);
  const s_t hessian_tmp2 = gu3 + s_t(1);
  const s_t hessian_tmp3 = -hessian_tmp0 + hessian_tmp1*hessian_tmp2;
  const s_t hessian_tmp4 = pow_m2(hessian_tmp3);
  const s_t hessian_tmp5 = pow_2(gu2)*hessian_tmp4;
  const s_t hessian_tmp6 = hessian_tmp5*lmbda;
  const s_t hessian_tmp7 = log(hessian_tmp3);
  const s_t hessian_tmp8 = hessian_tmp5*mu - hessian_tmp6*hessian_tmp7 + hessian_tmp6 + mu;
  const s_t hessian_tmp9 = idet*(-adj_lane1 - adj_lane3);
  const s_t hessian_tmp10 = hessian_tmp2*hessian_tmp4;
  const s_t hessian_tmp11 = gu2*hessian_tmp10;
  const s_t hessian_tmp12 = gu2*hessian_tmp2*hessian_tmp4*hessian_tmp7*lmbda - hessian_tmp11*lmbda - hessian_tmp11*mu;
  const s_t hessian_tmp13 = idet*(-adj_lane0 - adj_lane2);
  const s_t hessian_tmp14 = ((s_t(1) / s_t(2)))*hessian_tmp12*hessian_tmp13 + ((s_t(1) / s_t(2)))*hessian_tmp8*hessian_tmp9;
  const s_t hessian_tmp15 = pow_2(hessian_tmp2)*hessian_tmp4;
  const s_t hessian_tmp16 = hessian_tmp15*lmbda;
  const s_t hessian_tmp17 = hessian_tmp15*mu - hessian_tmp16*hessian_tmp7 + hessian_tmp16 + mu;
  const s_t hessian_tmp18 = ((s_t(1) / s_t(2)))*hessian_tmp12*hessian_tmp9 + ((s_t(1) / s_t(2)))*hessian_tmp13*hessian_tmp17;
  const s_t hessian_tmp19 = adj_lane0*hessian_tmp18 + adj_lane1*hessian_tmp14;
  const s_t hessian_tmp20 = adj_lane2*hessian_tmp18 + adj_lane3*hessian_tmp14;
  const s_t hessian_tmp21 = gu1*hessian_tmp10;
  const s_t hessian_tmp22 = gu1*hessian_tmp2*hessian_tmp4*hessian_tmp7*lmbda - hessian_tmp21*lmbda - hessian_tmp21*mu;
  const s_t hessian_tmp23 = hessian_tmp13*hessian_tmp22;
  const s_t hessian_tmp24 = pow_m1(hessian_tmp3);
  const s_t hessian_tmp25 = hessian_tmp24*mu;
  const s_t hessian_tmp26 = hessian_tmp0*hessian_tmp4;
  const s_t hessian_tmp27 = hessian_tmp7*lmbda;
  const s_t hessian_tmp28 = hessian_tmp24*hessian_tmp27;
  const s_t hessian_tmp29 = hessian_tmp25 - hessian_tmp26*hessian_tmp27 + hessian_tmp26*lmbda + hessian_tmp26*mu - hessian_tmp28;
  const s_t hessian_tmp30 = ((s_t(1) / s_t(2)))*hessian_tmp23 + ((s_t(1) / s_t(2)))*hessian_tmp29*hessian_tmp9;
  const s_t hessian_tmp31 = hessian_tmp1*hessian_tmp4;
  const s_t hessian_tmp32 = gu2*hessian_tmp31;
  const s_t hessian_tmp33 = gu2*hessian_tmp1*hessian_tmp4*hessian_tmp7*lmbda - hessian_tmp32*lmbda - hessian_tmp32*mu;
  const s_t hessian_tmp34 = hessian_tmp33*hessian_tmp9;
  const s_t hessian_tmp35 = hessian_tmp1*hessian_tmp2*hessian_tmp4;
  const s_t hessian_tmp36 = -hessian_tmp25 - hessian_tmp27*hessian_tmp35 + hessian_tmp28 + hessian_tmp35*lmbda + hessian_tmp35*mu;
  const s_t hessian_tmp37 = ((s_t(1) / s_t(2)))*hessian_tmp13*hessian_tmp36 + ((s_t(1) / s_t(2)))*hessian_tmp34;
  const s_t hessian_tmp38 = adj_lane0*hessian_tmp30 + adj_lane1*hessian_tmp37;
  const s_t hessian_tmp39 = adj_lane2*hessian_tmp30 + adj_lane3*hessian_tmp37;
  const s_t hessian_tmp40 = adj_lane1*idet;
  const s_t hessian_tmp41 = adj_lane0*idet;
  const s_t hessian_tmp42 = ((s_t(1) / s_t(2)))*hessian_tmp12*hessian_tmp41 + ((s_t(1) / s_t(2)))*hessian_tmp40*hessian_tmp8;
  const s_t hessian_tmp43 = ((s_t(1) / s_t(2)))*hessian_tmp12*hessian_tmp40 + ((s_t(1) / s_t(2)))*hessian_tmp17*hessian_tmp41;
  const s_t hessian_tmp44 = adj_lane0*hessian_tmp43 + adj_lane1*hessian_tmp42;
  const s_t hessian_tmp45 = adj_lane2*hessian_tmp43 + adj_lane3*hessian_tmp42;
  const s_t hessian_tmp46 = hessian_tmp22*hessian_tmp41;
  const s_t hessian_tmp47 = ((s_t(1) / s_t(2)))*hessian_tmp29*hessian_tmp40 + ((s_t(1) / s_t(2)))*hessian_tmp46;
  const s_t hessian_tmp48 = hessian_tmp33*hessian_tmp40;
  const s_t hessian_tmp49 = ((s_t(1) / s_t(2)))*hessian_tmp36*hessian_tmp41 + ((s_t(1) / s_t(2)))*hessian_tmp48;
  const s_t hessian_tmp50 = adj_lane0*hessian_tmp47 + adj_lane1*hessian_tmp49;
  const s_t hessian_tmp51 = adj_lane2*hessian_tmp47 + adj_lane3*hessian_tmp49;
  const s_t hessian_tmp52 = adj_lane3*idet;
  const s_t hessian_tmp53 = adj_lane2*idet;
  const s_t hessian_tmp54 = ((s_t(1) / s_t(2)))*hessian_tmp12*hessian_tmp53 + ((s_t(1) / s_t(2)))*hessian_tmp52*hessian_tmp8;
  const s_t hessian_tmp55 = ((s_t(1) / s_t(2)))*hessian_tmp12*hessian_tmp52 + ((s_t(1) / s_t(2)))*hessian_tmp17*hessian_tmp53;
  const s_t hessian_tmp56 = adj_lane0*hessian_tmp55 + adj_lane1*hessian_tmp54;
  const s_t hessian_tmp57 = adj_lane2*hessian_tmp55 + adj_lane3*hessian_tmp54;
  const s_t hessian_tmp58 = hessian_tmp22*hessian_tmp53;
  const s_t hessian_tmp59 = ((s_t(1) / s_t(2)))*hessian_tmp29*hessian_tmp52 + ((s_t(1) / s_t(2)))*hessian_tmp58;
  const s_t hessian_tmp60 = hessian_tmp33*hessian_tmp52;
  const s_t hessian_tmp61 = ((s_t(1) / s_t(2)))*hessian_tmp36*hessian_tmp53 + ((s_t(1) / s_t(2)))*hessian_tmp60;
  const s_t hessian_tmp62 = adj_lane0*hessian_tmp59 + adj_lane1*hessian_tmp61;
  const s_t hessian_tmp63 = adj_lane2*hessian_tmp59 + adj_lane3*hessian_tmp61;
  const s_t hessian_tmp64 = ((s_t(1) / s_t(2)))*hessian_tmp13*hessian_tmp29 + ((s_t(1) / s_t(2)))*hessian_tmp34;
  const s_t hessian_tmp65 = ((s_t(1) / s_t(2)))*hessian_tmp23 + ((s_t(1) / s_t(2)))*hessian_tmp36*hessian_tmp9;
  const s_t hessian_tmp66 = adj_lane0*hessian_tmp65 + adj_lane1*hessian_tmp64;
  const s_t hessian_tmp67 = adj_lane2*hessian_tmp65 + adj_lane3*hessian_tmp64;
  const s_t hessian_tmp68 = pow_2(gu1)*hessian_tmp4;
  const s_t hessian_tmp69 = -hessian_tmp27*hessian_tmp68 + hessian_tmp68*lmbda + hessian_tmp68*mu + mu;
  const s_t hessian_tmp70 = gu1*hessian_tmp31;
  const s_t hessian_tmp71 = gu1*hessian_tmp1*hessian_tmp4*hessian_tmp7*lmbda - hessian_tmp70*lmbda - hessian_tmp70*mu;
  const s_t hessian_tmp72 = ((s_t(1) / s_t(2)))*hessian_tmp13*hessian_tmp69 + ((s_t(1) / s_t(2)))*hessian_tmp71*hessian_tmp9;
  const s_t hessian_tmp73 = pow_2(hessian_tmp1)*hessian_tmp4;
  const s_t hessian_tmp74 = -hessian_tmp27*hessian_tmp73 + hessian_tmp73*lmbda + hessian_tmp73*mu + mu;
  const s_t hessian_tmp75 = ((s_t(1) / s_t(2)))*hessian_tmp13*hessian_tmp71 + ((s_t(1) / s_t(2)))*hessian_tmp74*hessian_tmp9;
  const s_t hessian_tmp76 = adj_lane0*hessian_tmp72 + adj_lane1*hessian_tmp75;
  const s_t hessian_tmp77 = adj_lane2*hessian_tmp72 + adj_lane3*hessian_tmp75;
  const s_t hessian_tmp78 = ((s_t(1) / s_t(2)))*hessian_tmp29*hessian_tmp41 + ((s_t(1) / s_t(2)))*hessian_tmp48;
  const s_t hessian_tmp79 = ((s_t(1) / s_t(2)))*hessian_tmp36*hessian_tmp40 + ((s_t(1) / s_t(2)))*hessian_tmp46;
  const s_t hessian_tmp80 = adj_lane0*hessian_tmp79 + adj_lane1*hessian_tmp78;
  const s_t hessian_tmp81 = adj_lane2*hessian_tmp79 + adj_lane3*hessian_tmp78;
  const s_t hessian_tmp82 = ((s_t(1) / s_t(2)))*hessian_tmp40*hessian_tmp71 + ((s_t(1) / s_t(2)))*hessian_tmp41*hessian_tmp69;
  const s_t hessian_tmp83 = ((s_t(1) / s_t(2)))*hessian_tmp40*hessian_tmp74 + ((s_t(1) / s_t(2)))*hessian_tmp41*hessian_tmp71;
  const s_t hessian_tmp84 = adj_lane0*hessian_tmp82 + adj_lane1*hessian_tmp83;
  const s_t hessian_tmp85 = adj_lane2*hessian_tmp82 + adj_lane3*hessian_tmp83;
  const s_t hessian_tmp86 = ((s_t(1) / s_t(2)))*hessian_tmp29*hessian_tmp53 + ((s_t(1) / s_t(2)))*hessian_tmp60;
  const s_t hessian_tmp87 = ((s_t(1) / s_t(2)))*hessian_tmp36*hessian_tmp52 + ((s_t(1) / s_t(2)))*hessian_tmp58;
  const s_t hessian_tmp88 = adj_lane0*hessian_tmp87 + adj_lane1*hessian_tmp86;
  const s_t hessian_tmp89 = adj_lane2*hessian_tmp87 + adj_lane3*hessian_tmp86;
  const s_t hessian_tmp90 = ((s_t(1) / s_t(2)))*hessian_tmp52*hessian_tmp71 + ((s_t(1) / s_t(2)))*hessian_tmp53*hessian_tmp69;
  const s_t hessian_tmp91 = ((s_t(1) / s_t(2)))*hessian_tmp52*hessian_tmp74 + ((s_t(1) / s_t(2)))*hessian_tmp53*hessian_tmp71;
  const s_t hessian_tmp92 = adj_lane0*hessian_tmp90 + adj_lane1*hessian_tmp91;
  const s_t hessian_tmp93 = adj_lane2*hessian_tmp90 + adj_lane3*hessian_tmp91;
  element_matrix[0] = -hessian_tmp19 - hessian_tmp20;
  element_matrix[6] = hessian_tmp19;
  element_matrix[12] = hessian_tmp20;
  element_matrix[18] = -hessian_tmp38 - hessian_tmp39;
  element_matrix[24] = hessian_tmp38;
  element_matrix[30] = hessian_tmp39;
  element_matrix[1] = -hessian_tmp44 - hessian_tmp45;
  element_matrix[7] = hessian_tmp44;
  element_matrix[13] = hessian_tmp45;
  element_matrix[19] = -hessian_tmp50 - hessian_tmp51;
  element_matrix[25] = hessian_tmp50;
  element_matrix[31] = hessian_tmp51;
  element_matrix[2] = -hessian_tmp56 - hessian_tmp57;
  element_matrix[8] = hessian_tmp56;
  element_matrix[14] = hessian_tmp57;
  element_matrix[20] = -hessian_tmp62 - hessian_tmp63;
  element_matrix[26] = hessian_tmp62;
  element_matrix[32] = hessian_tmp63;
  element_matrix[3] = -hessian_tmp66 - hessian_tmp67;
  element_matrix[9] = hessian_tmp66;
  element_matrix[15] = hessian_tmp67;
  element_matrix[21] = -hessian_tmp76 - hessian_tmp77;
  element_matrix[27] = hessian_tmp76;
  element_matrix[33] = hessian_tmp77;
  element_matrix[4] = -hessian_tmp80 - hessian_tmp81;
  element_matrix[10] = hessian_tmp80;
  element_matrix[16] = hessian_tmp81;
  element_matrix[22] = -hessian_tmp84 - hessian_tmp85;
  element_matrix[28] = hessian_tmp84;
  element_matrix[34] = hessian_tmp85;
  element_matrix[5] = -hessian_tmp88 - hessian_tmp89;
  element_matrix[11] = hessian_tmp88;
  element_matrix[17] = hessian_tmp89;
  element_matrix[23] = -hessian_tmp92 - hessian_tmp93;
  element_matrix[29] = hessian_tmp92;
  element_matrix[35] = hessian_tmp93;
}

} // namespace codegen
} // namespace sfem

#endif
