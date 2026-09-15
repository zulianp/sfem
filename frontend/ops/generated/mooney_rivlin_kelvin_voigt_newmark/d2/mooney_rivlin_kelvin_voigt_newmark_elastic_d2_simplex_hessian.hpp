#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_ELASTIC_D2_SIMPLEX_HESSIAN_HPP
#define MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_ELASTIC_D2_SIMPLEX_HESSIAN_HPP
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
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_direct_hessian_reference_element_matrix(
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
        const s_t weak_hess_tmp1 = lmbda*weak_hess_tmp0;
        const s_t weak_hess_tmp2 = s_t(2)*mu;
        const s_t weak_hess_tmp3 = weak_hess_tmp0*weak_hess_tmp2;
        const s_t weak_hess_tmp4 = -gu2*weak_hess_tmp1 - gu2*weak_hess_tmp3;
        const s_t weak_hess_tmp5 = -gu1*weak_hess_tmp1 - gu1*weak_hess_tmp3;
        const s_t weak_hess_tmp6 = pow_2(weak_hess_tmp0);
        const s_t weak_hess_tmp7 = gu0 + s_t(1);
        const s_t weak_hess_tmp8 = weak_hess_tmp0*weak_hess_tmp7;
        const s_t weak_hess_tmp9 = gu1*gu2;
        const s_t weak_hess_tmp10 = lmbda*(weak_hess_tmp0*weak_hess_tmp7 - weak_hess_tmp9 + s_t(-1));
        const s_t weak_hess_tmp11 = lmbda*weak_hess_tmp8 + mu*(s_t(4)*weak_hess_tmp0*weak_hess_tmp7 - s_t(2)*weak_hess_tmp9 + s_t(-6)) + weak_hess_tmp10;
        const s_t weak_hess_tmp12 = pow_2(gu2);
        const s_t weak_hess_tmp13 = lmbda*weak_hess_tmp7;
        const s_t weak_hess_tmp14 = weak_hess_tmp2*weak_hess_tmp7;
        const s_t weak_hess_tmp15 = -gu2*weak_hess_tmp13 - gu2*weak_hess_tmp14;
        const s_t weak_hess_tmp16 = lmbda*weak_hess_tmp9 + mu*(-s_t(2)*weak_hess_tmp8 + s_t(4)*weak_hess_tmp9 + s_t(6)) - weak_hess_tmp10;
        const s_t weak_hess_tmp17 = pow_2(gu1);
        const s_t weak_hess_tmp18 = -gu1*weak_hess_tmp13 - gu1*weak_hess_tmp14;
        const s_t weak_hess_tmp19 = pow_2(weak_hess_tmp7);
        material[0] = trial_grad[0]*(lmbda*weak_hess_tmp6 + mu*(s_t(2)*weak_hess_tmp6 + s_t(4))) + trial_grad[1]*weak_hess_tmp4 + trial_grad[2]*weak_hess_tmp5 + trial_grad[3]*weak_hess_tmp11;
        material[1] = trial_grad[0]*weak_hess_tmp4 + trial_grad[1]*(lmbda*weak_hess_tmp12 + mu*(s_t(2)*weak_hess_tmp12 + s_t(4))) + trial_grad[2]*weak_hess_tmp16 + trial_grad[3]*weak_hess_tmp15;
        material[2] = trial_grad[0]*weak_hess_tmp5 + trial_grad[1]*weak_hess_tmp16 + trial_grad[2]*(lmbda*weak_hess_tmp17 + mu*(s_t(2)*weak_hess_tmp17 + s_t(4))) + trial_grad[3]*weak_hess_tmp18;
        material[3] = trial_grad[0]*weak_hess_tmp11 + trial_grad[1]*weak_hess_tmp15 + trial_grad[2]*weak_hess_tmp18 + trial_grad[3]*(lmbda*weak_hess_tmp19 + mu*(s_t(2)*weak_hess_tmp19 + s_t(4)));
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
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_elastic_d2_simplex_tri3_direct_hessian_element_matrix(
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
  const s_t hessian_tmp0 = gu3 + s_t(1);
  const s_t hessian_tmp1 = hessian_tmp0*lmbda;
  const s_t hessian_tmp2 = s_t(2)*mu;
  const s_t hessian_tmp3 = hessian_tmp0*hessian_tmp2;
  const s_t hessian_tmp4 = -gu2*hessian_tmp1 - gu2*hessian_tmp3;
  const s_t hessian_tmp5 = idet*(-adj_lane0 - adj_lane2);
  const s_t hessian_tmp6 = pow_2(gu2);
  const s_t hessian_tmp7 = hessian_tmp6*lmbda + mu*(s_t(2)*hessian_tmp6 + s_t(4));
  const s_t hessian_tmp8 = idet*(-adj_lane1 - adj_lane3);
  const s_t hessian_tmp9 = ((s_t(1) / s_t(2)))*hessian_tmp4*hessian_tmp5 + ((s_t(1) / s_t(2)))*hessian_tmp7*hessian_tmp8;
  const s_t hessian_tmp10 = pow_2(hessian_tmp0);
  const s_t hessian_tmp11 = hessian_tmp10*lmbda + mu*(s_t(2)*hessian_tmp10 + s_t(4));
  const s_t hessian_tmp12 = ((s_t(1) / s_t(2)))*hessian_tmp11*hessian_tmp5 + ((s_t(1) / s_t(2)))*hessian_tmp4*hessian_tmp8;
  const s_t hessian_tmp13 = adj_lane0*hessian_tmp12 + adj_lane1*hessian_tmp9;
  const s_t hessian_tmp14 = adj_lane2*hessian_tmp12 + adj_lane3*hessian_tmp9;
  const s_t hessian_tmp15 = -gu1*hessian_tmp1 - gu1*hessian_tmp3;
  const s_t hessian_tmp16 = hessian_tmp15*hessian_tmp5;
  const s_t hessian_tmp17 = gu1*gu2;
  const s_t hessian_tmp18 = gu0 + s_t(1);
  const s_t hessian_tmp19 = lmbda*(hessian_tmp0*hessian_tmp18 - hessian_tmp17 + s_t(-1));
  const s_t hessian_tmp20 = hessian_tmp0*hessian_tmp18;
  const s_t hessian_tmp21 = hessian_tmp17*lmbda - hessian_tmp19 + mu*(s_t(4)*hessian_tmp17 - s_t(2)*hessian_tmp20 + s_t(6));
  const s_t hessian_tmp22 = ((s_t(1) / s_t(2)))*hessian_tmp16 + ((s_t(1) / s_t(2)))*hessian_tmp21*hessian_tmp8;
  const s_t hessian_tmp23 = hessian_tmp18*lmbda;
  const s_t hessian_tmp24 = hessian_tmp18*hessian_tmp2;
  const s_t hessian_tmp25 = -gu2*hessian_tmp23 - gu2*hessian_tmp24;
  const s_t hessian_tmp26 = hessian_tmp25*hessian_tmp8;
  const s_t hessian_tmp27 = hessian_tmp19 + hessian_tmp20*lmbda + mu*(s_t(4)*hessian_tmp0*hessian_tmp18 - s_t(2)*hessian_tmp17 + s_t(-6));
  const s_t hessian_tmp28 = ((s_t(1) / s_t(2)))*hessian_tmp26 + ((s_t(1) / s_t(2)))*hessian_tmp27*hessian_tmp5;
  const s_t hessian_tmp29 = adj_lane0*hessian_tmp22 + adj_lane1*hessian_tmp28;
  const s_t hessian_tmp30 = adj_lane2*hessian_tmp22 + adj_lane3*hessian_tmp28;
  const s_t hessian_tmp31 = adj_lane0*idet;
  const s_t hessian_tmp32 = adj_lane1*idet;
  const s_t hessian_tmp33 = ((s_t(1) / s_t(2)))*hessian_tmp31*hessian_tmp4 + ((s_t(1) / s_t(2)))*hessian_tmp32*hessian_tmp7;
  const s_t hessian_tmp34 = ((s_t(1) / s_t(2)))*hessian_tmp11*hessian_tmp31 + ((s_t(1) / s_t(2)))*hessian_tmp32*hessian_tmp4;
  const s_t hessian_tmp35 = adj_lane0*hessian_tmp34 + adj_lane1*hessian_tmp33;
  const s_t hessian_tmp36 = adj_lane2*hessian_tmp34 + adj_lane3*hessian_tmp33;
  const s_t hessian_tmp37 = hessian_tmp15*hessian_tmp31;
  const s_t hessian_tmp38 = ((s_t(1) / s_t(2)))*hessian_tmp21*hessian_tmp32 + ((s_t(1) / s_t(2)))*hessian_tmp37;
  const s_t hessian_tmp39 = hessian_tmp25*hessian_tmp32;
  const s_t hessian_tmp40 = ((s_t(1) / s_t(2)))*hessian_tmp27*hessian_tmp31 + ((s_t(1) / s_t(2)))*hessian_tmp39;
  const s_t hessian_tmp41 = adj_lane0*hessian_tmp38 + adj_lane1*hessian_tmp40;
  const s_t hessian_tmp42 = adj_lane2*hessian_tmp38 + adj_lane3*hessian_tmp40;
  const s_t hessian_tmp43 = adj_lane2*idet;
  const s_t hessian_tmp44 = adj_lane3*idet;
  const s_t hessian_tmp45 = ((s_t(1) / s_t(2)))*hessian_tmp4*hessian_tmp43 + ((s_t(1) / s_t(2)))*hessian_tmp44*hessian_tmp7;
  const s_t hessian_tmp46 = ((s_t(1) / s_t(2)))*hessian_tmp11*hessian_tmp43 + ((s_t(1) / s_t(2)))*hessian_tmp4*hessian_tmp44;
  const s_t hessian_tmp47 = adj_lane0*hessian_tmp46 + adj_lane1*hessian_tmp45;
  const s_t hessian_tmp48 = adj_lane2*hessian_tmp46 + adj_lane3*hessian_tmp45;
  const s_t hessian_tmp49 = hessian_tmp15*hessian_tmp43;
  const s_t hessian_tmp50 = ((s_t(1) / s_t(2)))*hessian_tmp21*hessian_tmp44 + ((s_t(1) / s_t(2)))*hessian_tmp49;
  const s_t hessian_tmp51 = hessian_tmp25*hessian_tmp44;
  const s_t hessian_tmp52 = ((s_t(1) / s_t(2)))*hessian_tmp27*hessian_tmp43 + ((s_t(1) / s_t(2)))*hessian_tmp51;
  const s_t hessian_tmp53 = adj_lane0*hessian_tmp50 + adj_lane1*hessian_tmp52;
  const s_t hessian_tmp54 = adj_lane2*hessian_tmp50 + adj_lane3*hessian_tmp52;
  const s_t hessian_tmp55 = ((s_t(1) / s_t(2)))*hessian_tmp21*hessian_tmp5 + ((s_t(1) / s_t(2)))*hessian_tmp26;
  const s_t hessian_tmp56 = ((s_t(1) / s_t(2)))*hessian_tmp16 + ((s_t(1) / s_t(2)))*hessian_tmp27*hessian_tmp8;
  const s_t hessian_tmp57 = adj_lane0*hessian_tmp56 + adj_lane1*hessian_tmp55;
  const s_t hessian_tmp58 = adj_lane2*hessian_tmp56 + adj_lane3*hessian_tmp55;
  const s_t hessian_tmp59 = pow_2(gu1);
  const s_t hessian_tmp60 = hessian_tmp59*lmbda + mu*(s_t(2)*hessian_tmp59 + s_t(4));
  const s_t hessian_tmp61 = -gu1*hessian_tmp23 - gu1*hessian_tmp24;
  const s_t hessian_tmp62 = ((s_t(1) / s_t(2)))*hessian_tmp5*hessian_tmp60 + ((s_t(1) / s_t(2)))*hessian_tmp61*hessian_tmp8;
  const s_t hessian_tmp63 = pow_2(hessian_tmp18);
  const s_t hessian_tmp64 = hessian_tmp63*lmbda + mu*(s_t(2)*hessian_tmp63 + s_t(4));
  const s_t hessian_tmp65 = ((s_t(1) / s_t(2)))*hessian_tmp5*hessian_tmp61 + ((s_t(1) / s_t(2)))*hessian_tmp64*hessian_tmp8;
  const s_t hessian_tmp66 = adj_lane0*hessian_tmp62 + adj_lane1*hessian_tmp65;
  const s_t hessian_tmp67 = adj_lane2*hessian_tmp62 + adj_lane3*hessian_tmp65;
  const s_t hessian_tmp68 = ((s_t(1) / s_t(2)))*hessian_tmp21*hessian_tmp31 + ((s_t(1) / s_t(2)))*hessian_tmp39;
  const s_t hessian_tmp69 = ((s_t(1) / s_t(2)))*hessian_tmp27*hessian_tmp32 + ((s_t(1) / s_t(2)))*hessian_tmp37;
  const s_t hessian_tmp70 = adj_lane0*hessian_tmp69 + adj_lane1*hessian_tmp68;
  const s_t hessian_tmp71 = adj_lane2*hessian_tmp69 + adj_lane3*hessian_tmp68;
  const s_t hessian_tmp72 = ((s_t(1) / s_t(2)))*hessian_tmp31*hessian_tmp60 + ((s_t(1) / s_t(2)))*hessian_tmp32*hessian_tmp61;
  const s_t hessian_tmp73 = ((s_t(1) / s_t(2)))*hessian_tmp31*hessian_tmp61 + ((s_t(1) / s_t(2)))*hessian_tmp32*hessian_tmp64;
  const s_t hessian_tmp74 = adj_lane0*hessian_tmp72 + adj_lane1*hessian_tmp73;
  const s_t hessian_tmp75 = adj_lane2*hessian_tmp72 + adj_lane3*hessian_tmp73;
  const s_t hessian_tmp76 = ((s_t(1) / s_t(2)))*hessian_tmp21*hessian_tmp43 + ((s_t(1) / s_t(2)))*hessian_tmp51;
  const s_t hessian_tmp77 = ((s_t(1) / s_t(2)))*hessian_tmp27*hessian_tmp44 + ((s_t(1) / s_t(2)))*hessian_tmp49;
  const s_t hessian_tmp78 = adj_lane0*hessian_tmp77 + adj_lane1*hessian_tmp76;
  const s_t hessian_tmp79 = adj_lane2*hessian_tmp77 + adj_lane3*hessian_tmp76;
  const s_t hessian_tmp80 = ((s_t(1) / s_t(2)))*hessian_tmp43*hessian_tmp60 + ((s_t(1) / s_t(2)))*hessian_tmp44*hessian_tmp61;
  const s_t hessian_tmp81 = ((s_t(1) / s_t(2)))*hessian_tmp43*hessian_tmp61 + ((s_t(1) / s_t(2)))*hessian_tmp44*hessian_tmp64;
  const s_t hessian_tmp82 = adj_lane0*hessian_tmp80 + adj_lane1*hessian_tmp81;
  const s_t hessian_tmp83 = adj_lane2*hessian_tmp80 + adj_lane3*hessian_tmp81;
  element_matrix[0] = -hessian_tmp13 - hessian_tmp14;
  element_matrix[6] = hessian_tmp13;
  element_matrix[12] = hessian_tmp14;
  element_matrix[18] = -hessian_tmp29 - hessian_tmp30;
  element_matrix[24] = hessian_tmp29;
  element_matrix[30] = hessian_tmp30;
  element_matrix[1] = -hessian_tmp35 - hessian_tmp36;
  element_matrix[7] = hessian_tmp35;
  element_matrix[13] = hessian_tmp36;
  element_matrix[19] = -hessian_tmp41 - hessian_tmp42;
  element_matrix[25] = hessian_tmp41;
  element_matrix[31] = hessian_tmp42;
  element_matrix[2] = -hessian_tmp47 - hessian_tmp48;
  element_matrix[8] = hessian_tmp47;
  element_matrix[14] = hessian_tmp48;
  element_matrix[20] = -hessian_tmp53 - hessian_tmp54;
  element_matrix[26] = hessian_tmp53;
  element_matrix[32] = hessian_tmp54;
  element_matrix[3] = -hessian_tmp57 - hessian_tmp58;
  element_matrix[9] = hessian_tmp57;
  element_matrix[15] = hessian_tmp58;
  element_matrix[21] = -hessian_tmp66 - hessian_tmp67;
  element_matrix[27] = hessian_tmp66;
  element_matrix[33] = hessian_tmp67;
  element_matrix[4] = -hessian_tmp70 - hessian_tmp71;
  element_matrix[10] = hessian_tmp70;
  element_matrix[16] = hessian_tmp71;
  element_matrix[22] = -hessian_tmp74 - hessian_tmp75;
  element_matrix[28] = hessian_tmp74;
  element_matrix[34] = hessian_tmp75;
  element_matrix[5] = -hessian_tmp78 - hessian_tmp79;
  element_matrix[11] = hessian_tmp78;
  element_matrix[17] = hessian_tmp79;
  element_matrix[23] = -hessian_tmp82 - hessian_tmp83;
  element_matrix[29] = hessian_tmp82;
  element_matrix[35] = hessian_tmp83;
}

} // namespace codegen
} // namespace sfem

#endif
