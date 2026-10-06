#ifndef SAINT_VENANT_KIRCHHOFF_D3_SIMPLEX_HESSIAN_HPP
#define SAINT_VENANT_KIRCHHOFF_D3_SIMPLEX_HESSIAN_HPP
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
static SFEM_INLINE void saint_venant_kirchhoff_d3_simplex_direct_hessian_reference_element_matrix(
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
        const s_t weak_hess_tmp0 = gu3*mu;
        const s_t weak_hess_tmp1 = gu0 + s_t(1);
        const s_t weak_hess_tmp2 = lmbda*weak_hess_tmp1;
        const s_t weak_hess_tmp3 = gu2*weak_hess_tmp0 + gu5*weak_hess_tmp2;
        const s_t weak_hess_tmp4 = gu6*mu;
        const s_t weak_hess_tmp5 = gu1*weak_hess_tmp4 + gu7*weak_hess_tmp2;
        const s_t weak_hess_tmp6 = gu4 + s_t(1);
        const s_t weak_hess_tmp7 = gu1*weak_hess_tmp0 + weak_hess_tmp2*weak_hess_tmp6;
        const s_t weak_hess_tmp8 = gu8 + s_t(1);
        const s_t weak_hess_tmp9 = gu2*weak_hess_tmp4 + weak_hess_tmp2*weak_hess_tmp8;
        const s_t weak_hess_tmp10 = gu1*weak_hess_tmp1;
        const s_t weak_hess_tmp11 = gu6*gu7;
        const s_t weak_hess_tmp12 = gu3*weak_hess_tmp6;
        const s_t weak_hess_tmp13 = lmbda*weak_hess_tmp10 + mu*(s_t(2)*weak_hess_tmp10 + weak_hess_tmp11 + weak_hess_tmp12);
        const s_t weak_hess_tmp14 = gu2*weak_hess_tmp1;
        const s_t weak_hess_tmp15 = gu3*gu5;
        const s_t weak_hess_tmp16 = gu6*weak_hess_tmp8;
        const s_t weak_hess_tmp17 = lmbda*weak_hess_tmp14 + mu*(s_t(2)*weak_hess_tmp14 + weak_hess_tmp15 + weak_hess_tmp16);
        const s_t weak_hess_tmp18 = gu3*weak_hess_tmp1;
        const s_t weak_hess_tmp19 = gu2*gu5;
        const s_t weak_hess_tmp20 = gu1*weak_hess_tmp6;
        const s_t weak_hess_tmp21 = lmbda*weak_hess_tmp18 + mu*(s_t(2)*weak_hess_tmp18 + weak_hess_tmp19 + weak_hess_tmp20);
        const s_t weak_hess_tmp22 = gu6*weak_hess_tmp1;
        const s_t weak_hess_tmp23 = gu1*gu7;
        const s_t weak_hess_tmp24 = gu2*weak_hess_tmp8;
        const s_t weak_hess_tmp25 = lmbda*weak_hess_tmp22 + mu*(s_t(2)*weak_hess_tmp22 + weak_hess_tmp23 + weak_hess_tmp24);
        const s_t weak_hess_tmp26 = pow_2(weak_hess_tmp1);
        const s_t weak_hess_tmp27 = pow_2(gu1);
        const s_t weak_hess_tmp28 = pow_2(gu3);
        const s_t weak_hess_tmp29 = weak_hess_tmp27 + weak_hess_tmp28;
        const s_t weak_hess_tmp30 = pow_2(gu6);
        const s_t weak_hess_tmp31 = pow_2(gu2);
        const s_t weak_hess_tmp32 = weak_hess_tmp31 + s_t(-1);
        const s_t weak_hess_tmp33 = weak_hess_tmp30 + weak_hess_tmp32;
        const s_t weak_hess_tmp34 = pow_2(gu5);
        const s_t weak_hess_tmp35 = pow_2(gu7);
        const s_t weak_hess_tmp36 = lmbda*(((s_t(1) / s_t(2)))*pow_2(gu0) + gu0 + ((s_t(1) / s_t(2)))*pow_2(gu4) + gu4 + ((s_t(1) / s_t(2)))*pow_2(gu8) + gu8 + ((s_t(1) / s_t(2)))*weak_hess_tmp27 + ((s_t(1) / s_t(2)))*weak_hess_tmp28 + ((s_t(1) / s_t(2)))*weak_hess_tmp30 + ((s_t(1) / s_t(2)))*weak_hess_tmp31 + ((s_t(1) / s_t(2)))*weak_hess_tmp34 + ((s_t(1) / s_t(2)))*weak_hess_tmp35);
        const s_t weak_hess_tmp37 = gu1*lmbda;
        const s_t weak_hess_tmp38 = mu*weak_hess_tmp6;
        const s_t weak_hess_tmp39 = gu2*weak_hess_tmp38 + gu5*weak_hess_tmp37;
        const s_t weak_hess_tmp40 = gu7*mu;
        const s_t weak_hess_tmp41 = gu6*weak_hess_tmp37 + weak_hess_tmp1*weak_hess_tmp40;
        const s_t weak_hess_tmp42 = gu2*weak_hess_tmp40 + weak_hess_tmp37*weak_hess_tmp8;
        const s_t weak_hess_tmp43 = gu3*weak_hess_tmp37 + weak_hess_tmp1*weak_hess_tmp38;
        const s_t weak_hess_tmp44 = gu1*gu2;
        const s_t weak_hess_tmp45 = gu5*weak_hess_tmp6;
        const s_t weak_hess_tmp46 = gu7*weak_hess_tmp8;
        const s_t weak_hess_tmp47 = lmbda*weak_hess_tmp44 + mu*(s_t(2)*weak_hess_tmp44 + weak_hess_tmp45 + weak_hess_tmp46);
        const s_t weak_hess_tmp48 = lmbda*weak_hess_tmp23 + mu*(weak_hess_tmp22 + s_t(2)*weak_hess_tmp23 + weak_hess_tmp24);
        const s_t weak_hess_tmp49 = lmbda*weak_hess_tmp20 + mu*(weak_hess_tmp18 + weak_hess_tmp19 + s_t(2)*weak_hess_tmp20);
        const s_t weak_hess_tmp50 = pow_2(weak_hess_tmp6);
        const s_t weak_hess_tmp51 = weak_hess_tmp26 + weak_hess_tmp50;
        const s_t weak_hess_tmp52 = gu2*lmbda;
        const s_t weak_hess_tmp53 = gu5*mu;
        const s_t weak_hess_tmp54 = gu3*weak_hess_tmp52 + weak_hess_tmp1*weak_hess_tmp53;
        const s_t weak_hess_tmp55 = gu1*weak_hess_tmp53 + weak_hess_tmp52*weak_hess_tmp6;
        const s_t weak_hess_tmp56 = mu*weak_hess_tmp8;
        const s_t weak_hess_tmp57 = gu1*weak_hess_tmp56 + gu7*weak_hess_tmp52;
        const s_t weak_hess_tmp58 = gu6*weak_hess_tmp52 + weak_hess_tmp1*weak_hess_tmp56;
        const s_t weak_hess_tmp59 = lmbda*weak_hess_tmp19 + mu*(weak_hess_tmp18 + s_t(2)*weak_hess_tmp19 + weak_hess_tmp20);
        const s_t weak_hess_tmp60 = lmbda*weak_hess_tmp24 + mu*(weak_hess_tmp22 + weak_hess_tmp23 + s_t(2)*weak_hess_tmp24);
        const s_t weak_hess_tmp61 = weak_hess_tmp34 + s_t(-1);
        const s_t weak_hess_tmp62 = pow_2(weak_hess_tmp8);
        const s_t weak_hess_tmp63 = weak_hess_tmp26 + weak_hess_tmp62;
        const s_t weak_hess_tmp64 = gu3*lmbda;
        const s_t weak_hess_tmp65 = gu7*weak_hess_tmp64 + weak_hess_tmp4*weak_hess_tmp6;
        const s_t weak_hess_tmp66 = gu5*weak_hess_tmp4 + weak_hess_tmp64*weak_hess_tmp8;
        const s_t weak_hess_tmp67 = lmbda*weak_hess_tmp15 + mu*(weak_hess_tmp14 + s_t(2)*weak_hess_tmp15 + weak_hess_tmp16);
        const s_t weak_hess_tmp68 = gu3*gu6;
        const s_t weak_hess_tmp69 = gu5*weak_hess_tmp8;
        const s_t weak_hess_tmp70 = gu7*weak_hess_tmp6;
        const s_t weak_hess_tmp71 = lmbda*weak_hess_tmp68 + mu*(s_t(2)*weak_hess_tmp68 + weak_hess_tmp69 + weak_hess_tmp70);
        const s_t weak_hess_tmp72 = lmbda*weak_hess_tmp12 + mu*(weak_hess_tmp10 + weak_hess_tmp11 + s_t(2)*weak_hess_tmp12);
        const s_t weak_hess_tmp73 = lmbda*weak_hess_tmp6;
        const s_t weak_hess_tmp74 = gu6*weak_hess_tmp73 + gu7*weak_hess_tmp0;
        const s_t weak_hess_tmp75 = gu5*weak_hess_tmp40 + weak_hess_tmp73*weak_hess_tmp8;
        const s_t weak_hess_tmp76 = lmbda*weak_hess_tmp45 + mu*(weak_hess_tmp44 + s_t(2)*weak_hess_tmp45 + weak_hess_tmp46);
        const s_t weak_hess_tmp77 = lmbda*weak_hess_tmp70 + mu*(weak_hess_tmp68 + weak_hess_tmp69 + s_t(2)*weak_hess_tmp70);
        const s_t weak_hess_tmp78 = gu5*lmbda;
        const s_t weak_hess_tmp79 = gu6*weak_hess_tmp78 + weak_hess_tmp0*weak_hess_tmp8;
        const s_t weak_hess_tmp80 = gu7*weak_hess_tmp78 + weak_hess_tmp38*weak_hess_tmp8;
        const s_t weak_hess_tmp81 = lmbda*weak_hess_tmp69 + mu*(weak_hess_tmp68 + s_t(2)*weak_hess_tmp69 + weak_hess_tmp70);
        const s_t weak_hess_tmp82 = weak_hess_tmp50 + weak_hess_tmp62;
        const s_t weak_hess_tmp83 = lmbda*weak_hess_tmp11 + mu*(weak_hess_tmp10 + s_t(2)*weak_hess_tmp11 + weak_hess_tmp12);
        const s_t weak_hess_tmp84 = lmbda*weak_hess_tmp16 + mu*(weak_hess_tmp14 + weak_hess_tmp15 + s_t(2)*weak_hess_tmp16);
        const s_t weak_hess_tmp85 = lmbda*weak_hess_tmp46 + mu*(weak_hess_tmp44 + weak_hess_tmp45 + s_t(2)*weak_hess_tmp46);
        material[0] = trial_grad[0]*(lmbda*weak_hess_tmp26 + mu*(s_t(3)*weak_hess_tmp26 + weak_hess_tmp29 + weak_hess_tmp33) + weak_hess_tmp36) + trial_grad[1]*weak_hess_tmp13 + trial_grad[2]*weak_hess_tmp17 + trial_grad[3]*weak_hess_tmp21 + trial_grad[4]*weak_hess_tmp7 + trial_grad[5]*weak_hess_tmp3 + trial_grad[6]*weak_hess_tmp25 + trial_grad[7]*weak_hess_tmp5 + trial_grad[8]*weak_hess_tmp9;
        material[1] = trial_grad[0]*weak_hess_tmp13 + trial_grad[1]*(lmbda*weak_hess_tmp27 + mu*(s_t(3)*weak_hess_tmp27 + weak_hess_tmp32 + weak_hess_tmp35 + weak_hess_tmp51) + weak_hess_tmp36) + trial_grad[2]*weak_hess_tmp47 + trial_grad[3]*weak_hess_tmp43 + trial_grad[4]*weak_hess_tmp49 + trial_grad[5]*weak_hess_tmp39 + trial_grad[6]*weak_hess_tmp41 + trial_grad[7]*weak_hess_tmp48 + trial_grad[8]*weak_hess_tmp42;
        material[2] = trial_grad[0]*weak_hess_tmp17 + trial_grad[1]*weak_hess_tmp47 + trial_grad[2]*(lmbda*weak_hess_tmp31 + mu*(weak_hess_tmp27 + s_t(3)*weak_hess_tmp31 + weak_hess_tmp61 + weak_hess_tmp63) + weak_hess_tmp36) + trial_grad[3]*weak_hess_tmp54 + trial_grad[4]*weak_hess_tmp55 + trial_grad[5]*weak_hess_tmp59 + trial_grad[6]*weak_hess_tmp58 + trial_grad[7]*weak_hess_tmp57 + trial_grad[8]*weak_hess_tmp60;
        material[3] = trial_grad[0]*weak_hess_tmp21 + trial_grad[1]*weak_hess_tmp43 + trial_grad[2]*weak_hess_tmp54 + trial_grad[3]*(lmbda*weak_hess_tmp28 + mu*(s_t(3)*weak_hess_tmp28 + weak_hess_tmp30 + weak_hess_tmp51 + weak_hess_tmp61) + weak_hess_tmp36) + trial_grad[4]*weak_hess_tmp72 + trial_grad[5]*weak_hess_tmp67 + trial_grad[6]*weak_hess_tmp71 + trial_grad[7]*weak_hess_tmp65 + trial_grad[8]*weak_hess_tmp66;
        material[4] = trial_grad[0]*weak_hess_tmp7 + trial_grad[1]*weak_hess_tmp49 + trial_grad[2]*weak_hess_tmp55 + trial_grad[3]*weak_hess_tmp72 + trial_grad[4]*(lmbda*weak_hess_tmp50 + mu*(weak_hess_tmp29 + weak_hess_tmp35 + s_t(3)*weak_hess_tmp50 + weak_hess_tmp61) + weak_hess_tmp36) + trial_grad[5]*weak_hess_tmp76 + trial_grad[6]*weak_hess_tmp74 + trial_grad[7]*weak_hess_tmp77 + trial_grad[8]*weak_hess_tmp75;
        material[5] = trial_grad[0]*weak_hess_tmp3 + trial_grad[1]*weak_hess_tmp39 + trial_grad[2]*weak_hess_tmp59 + trial_grad[3]*weak_hess_tmp67 + trial_grad[4]*weak_hess_tmp76 + trial_grad[5]*(lmbda*weak_hess_tmp34 + mu*(weak_hess_tmp28 + weak_hess_tmp32 + s_t(3)*weak_hess_tmp34 + weak_hess_tmp82) + weak_hess_tmp36) + trial_grad[6]*weak_hess_tmp79 + trial_grad[7]*weak_hess_tmp80 + trial_grad[8]*weak_hess_tmp81;
        material[6] = trial_grad[0]*weak_hess_tmp25 + trial_grad[1]*weak_hess_tmp41 + trial_grad[2]*weak_hess_tmp58 + trial_grad[3]*weak_hess_tmp71 + trial_grad[4]*weak_hess_tmp74 + trial_grad[5]*weak_hess_tmp79 + trial_grad[6]*(lmbda*weak_hess_tmp30 + mu*(weak_hess_tmp28 + s_t(3)*weak_hess_tmp30 + weak_hess_tmp35 + weak_hess_tmp63 + s_t(-1)) + weak_hess_tmp36) + trial_grad[7]*weak_hess_tmp83 + trial_grad[8]*weak_hess_tmp84;
        material[7] = trial_grad[0]*weak_hess_tmp5 + trial_grad[1]*weak_hess_tmp48 + trial_grad[2]*weak_hess_tmp57 + trial_grad[3]*weak_hess_tmp65 + trial_grad[4]*weak_hess_tmp77 + trial_grad[5]*weak_hess_tmp80 + trial_grad[6]*weak_hess_tmp83 + trial_grad[7]*(lmbda*weak_hess_tmp35 + mu*(weak_hess_tmp27 + weak_hess_tmp30 + s_t(3)*weak_hess_tmp35 + weak_hess_tmp82 + s_t(-1)) + weak_hess_tmp36) + trial_grad[8]*weak_hess_tmp85;
        material[8] = trial_grad[0]*weak_hess_tmp9 + trial_grad[1]*weak_hess_tmp42 + trial_grad[2]*weak_hess_tmp60 + trial_grad[3]*weak_hess_tmp66 + trial_grad[4]*weak_hess_tmp75 + trial_grad[5]*weak_hess_tmp81 + trial_grad[6]*weak_hess_tmp84 + trial_grad[7]*weak_hess_tmp85 + trial_grad[8]*(lmbda*weak_hess_tmp62 + mu*(weak_hess_tmp33 + weak_hess_tmp34 + weak_hess_tmp35 + s_t(3)*weak_hess_tmp62) + weak_hess_tmp36);
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

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void saint_venant_kirchhoff_d3_simplex_tet4_direct_hessian_element_matrix(
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
    const s_t lmbda,
    const s_t mu,
    const s_t bu_data[NS * 3][VS],
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
  const s_t adj_lane4 = badj4[goff];
  const s_t adj_lane5 = badj5[goff];
  const s_t adj_lane6 = badj6[goff];
  const s_t adj_lane7 = badj7[goff];
  const s_t adj_lane8 = badj8[goff];
  const s_t det_lane0 = bdet0[goff];
  const s_t idet = s_t(1) / det_lane0;
  const s_t gu_ref0 = -bu_data[0][lane] + bu_data[3][lane];
  const s_t gu_ref1 = -bu_data[0][lane] + bu_data[6][lane];
  const s_t gu_ref2 = -bu_data[0][lane] + bu_data[9][lane];
  const s_t gu_ref3 = -bu_data[1][lane] + bu_data[4][lane];
  const s_t gu_ref4 = -bu_data[1][lane] + bu_data[7][lane];
  const s_t gu_ref5 = bu_data[10][lane] - bu_data[1][lane];
  const s_t gu_ref6 = -bu_data[2][lane] + bu_data[5][lane];
  const s_t gu_ref7 = -bu_data[2][lane] + bu_data[8][lane];
  const s_t gu_ref8 = bu_data[11][lane] - bu_data[2][lane];
  const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane3 + gu_ref2 * adj_lane6) * idet;
  const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane4 + gu_ref2 * adj_lane7) * idet;
  const s_t gu2 = (gu_ref0 * adj_lane2 + gu_ref1 * adj_lane5 + gu_ref2 * adj_lane8) * idet;
  const s_t gu3 = (gu_ref3 * adj_lane0 + gu_ref4 * adj_lane3 + gu_ref5 * adj_lane6) * idet;
  const s_t gu4 = (gu_ref3 * adj_lane1 + gu_ref4 * adj_lane4 + gu_ref5 * adj_lane7) * idet;
  const s_t gu5 = (gu_ref3 * adj_lane2 + gu_ref4 * adj_lane5 + gu_ref5 * adj_lane8) * idet;
  const s_t gu6 = (gu_ref6 * adj_lane0 + gu_ref7 * adj_lane3 + gu_ref8 * adj_lane6) * idet;
  const s_t gu7 = (gu_ref6 * adj_lane1 + gu_ref7 * adj_lane4 + gu_ref8 * adj_lane7) * idet;
  const s_t gu8 = (gu_ref6 * adj_lane2 + gu_ref7 * adj_lane5 + gu_ref8 * adj_lane8) * idet;
  const s_t hessian_tmp0 = gu1*gu2;
  const s_t hessian_tmp1 = gu4 + s_t(1);
  const s_t hessian_tmp2 = gu5*hessian_tmp1;
  const s_t hessian_tmp3 = gu8 + s_t(1);
  const s_t hessian_tmp4 = gu7*hessian_tmp3;
  const s_t hessian_tmp5 = hessian_tmp0*lmbda + mu*(s_t(2)*hessian_tmp0 + hessian_tmp2 + hessian_tmp4);
  const s_t hessian_tmp6 = idet*(-adj_lane2 - adj_lane5 - adj_lane8);
  const s_t hessian_tmp7 = -adj_lane0 - adj_lane3 - adj_lane6;
  const s_t hessian_tmp8 = gu0 + s_t(1);
  const s_t hessian_tmp9 = gu1*hessian_tmp8;
  const s_t hessian_tmp10 = gu6*gu7;
  const s_t hessian_tmp11 = gu3*hessian_tmp1;
  const s_t hessian_tmp12 = idet*(hessian_tmp9*lmbda + mu*(hessian_tmp10 + hessian_tmp11 + s_t(2)*hessian_tmp9));
  const s_t hessian_tmp13 = pow_2(gu1);
  const s_t hessian_tmp14 = pow_2(gu7);
  const s_t hessian_tmp15 = pow_2(gu2);
  const s_t hessian_tmp16 = hessian_tmp15 + s_t(-1);
  const s_t hessian_tmp17 = pow_2(hessian_tmp8);
  const s_t hessian_tmp18 = pow_2(hessian_tmp1);
  const s_t hessian_tmp19 = hessian_tmp17 + hessian_tmp18;
  const s_t hessian_tmp20 = pow_2(gu3);
  const s_t hessian_tmp21 = pow_2(gu5);
  const s_t hessian_tmp22 = pow_2(gu6);
  const s_t hessian_tmp23 = lmbda*(((s_t(1) / s_t(2)))*pow_2(gu0) + gu0 + ((s_t(1) / s_t(2)))*pow_2(gu4) + gu4 + ((s_t(1) / s_t(2)))*pow_2(gu8) + gu8 + ((s_t(1) / s_t(2)))*hessian_tmp13 + ((s_t(1) / s_t(2)))*hessian_tmp14 + ((s_t(1) / s_t(2)))*hessian_tmp15 + ((s_t(1) / s_t(2)))*hessian_tmp20 + ((s_t(1) / s_t(2)))*hessian_tmp21 + ((s_t(1) / s_t(2)))*hessian_tmp22);
  const s_t hessian_tmp24 = hessian_tmp13*lmbda + hessian_tmp23 + mu*(s_t(3)*hessian_tmp13 + hessian_tmp14 + hessian_tmp16 + hessian_tmp19);
  const s_t hessian_tmp25 = -adj_lane1 - adj_lane4 - adj_lane7;
  const s_t hessian_tmp26 = hessian_tmp25*idet;
  const s_t hessian_tmp27 = ((s_t(1) / s_t(6)))*hessian_tmp12*hessian_tmp7 + ((s_t(1) / s_t(6)))*hessian_tmp24*hessian_tmp26 + ((s_t(1) / s_t(6)))*hessian_tmp5*hessian_tmp6;
  const s_t hessian_tmp28 = gu2*hessian_tmp8;
  const s_t hessian_tmp29 = gu3*gu5;
  const s_t hessian_tmp30 = gu6*hessian_tmp3;
  const s_t hessian_tmp31 = hessian_tmp28*lmbda + mu*(s_t(2)*hessian_tmp28 + hessian_tmp29 + hessian_tmp30);
  const s_t hessian_tmp32 = hessian_tmp7*idet;
  const s_t hessian_tmp33 = hessian_tmp21 + s_t(-1);
  const s_t hessian_tmp34 = pow_2(hessian_tmp3);
  const s_t hessian_tmp35 = hessian_tmp17 + hessian_tmp34;
  const s_t hessian_tmp36 = hessian_tmp15*lmbda + hessian_tmp23 + mu*(hessian_tmp13 + s_t(3)*hessian_tmp15 + hessian_tmp33 + hessian_tmp35);
  const s_t hessian_tmp37 = ((s_t(1) / s_t(6)))*hessian_tmp26*hessian_tmp5 + ((s_t(1) / s_t(6)))*hessian_tmp31*hessian_tmp32 + ((s_t(1) / s_t(6)))*hessian_tmp36*hessian_tmp6;
  const s_t hessian_tmp38 = hessian_tmp13 + hessian_tmp20;
  const s_t hessian_tmp39 = hessian_tmp16 + hessian_tmp22;
  const s_t hessian_tmp40 = hessian_tmp17*lmbda + hessian_tmp23 + mu*(s_t(3)*hessian_tmp17 + hessian_tmp38 + hessian_tmp39);
  const s_t hessian_tmp41 = ((s_t(1) / s_t(6)))*hessian_tmp12*hessian_tmp25 + ((s_t(1) / s_t(6)))*hessian_tmp31*hessian_tmp6 + ((s_t(1) / s_t(6)))*hessian_tmp32*hessian_tmp40;
  const s_t hessian_tmp42 = adj_lane0*hessian_tmp41 + adj_lane1*hessian_tmp27 + adj_lane2*hessian_tmp37;
  const s_t hessian_tmp43 = adj_lane3*hessian_tmp41 + adj_lane4*hessian_tmp27 + adj_lane5*hessian_tmp37;
  const s_t hessian_tmp44 = adj_lane6*hessian_tmp41 + adj_lane7*hessian_tmp27 + adj_lane8*hessian_tmp37;
  const s_t hessian_tmp45 = gu5*lmbda;
  const s_t hessian_tmp46 = gu2*mu;
  const s_t hessian_tmp47 = gu1*hessian_tmp45 + hessian_tmp1*hessian_tmp46;
  const s_t hessian_tmp48 = gu3*hessian_tmp46 + hessian_tmp45*hessian_tmp8;
  const s_t hessian_tmp49 = gu2*gu5;
  const s_t hessian_tmp50 = gu1*hessian_tmp1;
  const s_t hessian_tmp51 = gu3*hessian_tmp8;
  const s_t hessian_tmp52 = hessian_tmp49*lmbda + mu*(s_t(2)*hessian_tmp49 + hessian_tmp50 + hessian_tmp51);
  const s_t hessian_tmp53 = hessian_tmp52*hessian_tmp6;
  const s_t hessian_tmp54 = ((s_t(1) / s_t(6)))*hessian_tmp26*hessian_tmp47 + ((s_t(1) / s_t(6)))*hessian_tmp32*hessian_tmp48 + ((s_t(1) / s_t(6)))*hessian_tmp53;
  const s_t hessian_tmp55 = gu3*lmbda;
  const s_t hessian_tmp56 = hessian_tmp8*mu;
  const s_t hessian_tmp57 = gu2*hessian_tmp55 + gu5*hessian_tmp56;
  const s_t hessian_tmp58 = gu1*hessian_tmp55 + hessian_tmp1*hessian_tmp56;
  const s_t hessian_tmp59 = hessian_tmp51*lmbda + mu*(hessian_tmp49 + hessian_tmp50 + s_t(2)*hessian_tmp51);
  const s_t hessian_tmp60 = hessian_tmp32*hessian_tmp59;
  const s_t hessian_tmp61 = ((s_t(1) / s_t(6)))*hessian_tmp26*hessian_tmp58 + ((s_t(1) / s_t(6)))*hessian_tmp57*hessian_tmp6 + ((s_t(1) / s_t(6)))*hessian_tmp60;
  const s_t hessian_tmp62 = gu1*mu;
  const s_t hessian_tmp63 = hessian_tmp1*lmbda;
  const s_t hessian_tmp64 = gu2*hessian_tmp63 + gu5*hessian_tmp62;
  const s_t hessian_tmp65 = gu3*hessian_tmp62 + hessian_tmp63*hessian_tmp8;
  const s_t hessian_tmp66 = hessian_tmp50*lmbda + mu*(hessian_tmp49 + s_t(2)*hessian_tmp50 + hessian_tmp51);
  const s_t hessian_tmp67 = hessian_tmp26*hessian_tmp66;
  const s_t hessian_tmp68 = ((s_t(1) / s_t(6)))*hessian_tmp32*hessian_tmp65 + ((s_t(1) / s_t(6)))*hessian_tmp6*hessian_tmp64 + ((s_t(1) / s_t(6)))*hessian_tmp67;
  const s_t hessian_tmp69 = adj_lane0*hessian_tmp61 + adj_lane1*hessian_tmp68 + adj_lane2*hessian_tmp54;
  const s_t hessian_tmp70 = adj_lane3*hessian_tmp61 + adj_lane4*hessian_tmp68 + adj_lane5*hessian_tmp54;
  const s_t hessian_tmp71 = adj_lane6*hessian_tmp61 + adj_lane7*hessian_tmp68 + adj_lane8*hessian_tmp54;
  const s_t hessian_tmp72 = gu7*lmbda;
  const s_t hessian_tmp73 = gu6*hessian_tmp62 + hessian_tmp72*hessian_tmp8;
  const s_t hessian_tmp74 = gu2*hessian_tmp72 + hessian_tmp3*hessian_tmp62;
  const s_t hessian_tmp75 = gu1*gu7;
  const s_t hessian_tmp76 = gu2*hessian_tmp3;
  const s_t hessian_tmp77 = gu6*hessian_tmp8;
  const s_t hessian_tmp78 = hessian_tmp75*lmbda + mu*(s_t(2)*hessian_tmp75 + hessian_tmp76 + hessian_tmp77);
  const s_t hessian_tmp79 = hessian_tmp26*hessian_tmp78;
  const s_t hessian_tmp80 = ((s_t(1) / s_t(6)))*hessian_tmp32*hessian_tmp73 + ((s_t(1) / s_t(6)))*hessian_tmp6*hessian_tmp74 + ((s_t(1) / s_t(6)))*hessian_tmp79;
  const s_t hessian_tmp81 = gu6*lmbda;
  const s_t hessian_tmp82 = gu1*hessian_tmp81 + gu7*hessian_tmp56;
  const s_t hessian_tmp83 = gu2*hessian_tmp81 + hessian_tmp3*hessian_tmp56;
  const s_t hessian_tmp84 = hessian_tmp77*lmbda + mu*(hessian_tmp75 + hessian_tmp76 + s_t(2)*hessian_tmp77);
  const s_t hessian_tmp85 = hessian_tmp32*hessian_tmp84;
  const s_t hessian_tmp86 = ((s_t(1) / s_t(6)))*hessian_tmp26*hessian_tmp82 + ((s_t(1) / s_t(6)))*hessian_tmp6*hessian_tmp83 + ((s_t(1) / s_t(6)))*hessian_tmp85;
  const s_t hessian_tmp87 = hessian_tmp3*lmbda;
  const s_t hessian_tmp88 = gu1*hessian_tmp87 + gu7*hessian_tmp46;
  const s_t hessian_tmp89 = gu6*hessian_tmp46 + hessian_tmp8*hessian_tmp87;
  const s_t hessian_tmp90 = hessian_tmp76*lmbda + mu*(hessian_tmp75 + s_t(2)*hessian_tmp76 + hessian_tmp77);
  const s_t hessian_tmp91 = hessian_tmp6*hessian_tmp90;
  const s_t hessian_tmp92 = ((s_t(1) / s_t(6)))*hessian_tmp26*hessian_tmp88 + ((s_t(1) / s_t(6)))*hessian_tmp32*hessian_tmp89 + ((s_t(1) / s_t(6)))*hessian_tmp91;
  const s_t hessian_tmp93 = adj_lane0*hessian_tmp86 + adj_lane1*hessian_tmp80 + adj_lane2*hessian_tmp92;
  const s_t hessian_tmp94 = adj_lane3*hessian_tmp86 + adj_lane4*hessian_tmp80 + adj_lane5*hessian_tmp92;
  const s_t hessian_tmp95 = adj_lane6*hessian_tmp86 + adj_lane7*hessian_tmp80 + adj_lane8*hessian_tmp92;
  const s_t hessian_tmp96 = adj_lane2*idet;
  const s_t hessian_tmp97 = adj_lane1*idet;
  const s_t hessian_tmp98 = ((s_t(1) / s_t(6)))*adj_lane0*hessian_tmp12 + ((s_t(1) / s_t(6)))*hessian_tmp24*hessian_tmp97 + ((s_t(1) / s_t(6)))*hessian_tmp5*hessian_tmp96;
  const s_t hessian_tmp99 = adj_lane0*idet;
  const s_t hessian_tmp100 = ((s_t(1) / s_t(6)))*hessian_tmp31*hessian_tmp99 + ((s_t(1) / s_t(6)))*hessian_tmp36*hessian_tmp96 + ((s_t(1) / s_t(6)))*hessian_tmp5*hessian_tmp97;
  const s_t hessian_tmp101 = ((s_t(1) / s_t(6)))*adj_lane1*hessian_tmp12 + ((s_t(1) / s_t(6)))*hessian_tmp31*hessian_tmp96 + ((s_t(1) / s_t(6)))*hessian_tmp40*hessian_tmp99;
  const s_t hessian_tmp102 = adj_lane0*hessian_tmp101 + adj_lane1*hessian_tmp98 + adj_lane2*hessian_tmp100;
  const s_t hessian_tmp103 = adj_lane3*hessian_tmp101 + adj_lane4*hessian_tmp98 + adj_lane5*hessian_tmp100;
  const s_t hessian_tmp104 = adj_lane6*hessian_tmp101 + adj_lane7*hessian_tmp98 + adj_lane8*hessian_tmp100;
  const s_t hessian_tmp105 = hessian_tmp52*hessian_tmp96;
  const s_t hessian_tmp106 = ((s_t(1) / s_t(6)))*hessian_tmp105 + ((s_t(1) / s_t(6)))*hessian_tmp47*hessian_tmp97 + ((s_t(1) / s_t(6)))*hessian_tmp48*hessian_tmp99;
  const s_t hessian_tmp107 = hessian_tmp59*hessian_tmp99;
  const s_t hessian_tmp108 = ((s_t(1) / s_t(6)))*hessian_tmp107 + ((s_t(1) / s_t(6)))*hessian_tmp57*hessian_tmp96 + ((s_t(1) / s_t(6)))*hessian_tmp58*hessian_tmp97;
  const s_t hessian_tmp109 = hessian_tmp66*hessian_tmp97;
  const s_t hessian_tmp110 = ((s_t(1) / s_t(6)))*hessian_tmp109 + ((s_t(1) / s_t(6)))*hessian_tmp64*hessian_tmp96 + ((s_t(1) / s_t(6)))*hessian_tmp65*hessian_tmp99;
  const s_t hessian_tmp111 = adj_lane0*hessian_tmp108 + adj_lane1*hessian_tmp110 + adj_lane2*hessian_tmp106;
  const s_t hessian_tmp112 = adj_lane3*hessian_tmp108 + adj_lane4*hessian_tmp110 + adj_lane5*hessian_tmp106;
  const s_t hessian_tmp113 = adj_lane6*hessian_tmp108 + adj_lane7*hessian_tmp110 + adj_lane8*hessian_tmp106;
  const s_t hessian_tmp114 = hessian_tmp78*hessian_tmp97;
  const s_t hessian_tmp115 = ((s_t(1) / s_t(6)))*hessian_tmp114 + ((s_t(1) / s_t(6)))*hessian_tmp73*hessian_tmp99 + ((s_t(1) / s_t(6)))*hessian_tmp74*hessian_tmp96;
  const s_t hessian_tmp116 = hessian_tmp84*hessian_tmp99;
  const s_t hessian_tmp117 = ((s_t(1) / s_t(6)))*hessian_tmp116 + ((s_t(1) / s_t(6)))*hessian_tmp82*hessian_tmp97 + ((s_t(1) / s_t(6)))*hessian_tmp83*hessian_tmp96;
  const s_t hessian_tmp118 = hessian_tmp90*hessian_tmp96;
  const s_t hessian_tmp119 = ((s_t(1) / s_t(6)))*hessian_tmp118 + ((s_t(1) / s_t(6)))*hessian_tmp88*hessian_tmp97 + ((s_t(1) / s_t(6)))*hessian_tmp89*hessian_tmp99;
  const s_t hessian_tmp120 = adj_lane0*hessian_tmp117 + adj_lane1*hessian_tmp115 + adj_lane2*hessian_tmp119;
  const s_t hessian_tmp121 = adj_lane3*hessian_tmp117 + adj_lane4*hessian_tmp115 + adj_lane5*hessian_tmp119;
  const s_t hessian_tmp122 = adj_lane6*hessian_tmp117 + adj_lane7*hessian_tmp115 + adj_lane8*hessian_tmp119;
  const s_t hessian_tmp123 = adj_lane5*idet;
  const s_t hessian_tmp124 = adj_lane4*idet;
  const s_t hessian_tmp125 = ((s_t(1) / s_t(6)))*adj_lane3*hessian_tmp12 + ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp5 + ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp24;
  const s_t hessian_tmp126 = adj_lane3*idet;
  const s_t hessian_tmp127 = ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp36 + ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp5 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp31;
  const s_t hessian_tmp128 = ((s_t(1) / s_t(6)))*adj_lane4*hessian_tmp12 + ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp31 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp40;
  const s_t hessian_tmp129 = adj_lane0*hessian_tmp128 + adj_lane1*hessian_tmp125 + adj_lane2*hessian_tmp127;
  const s_t hessian_tmp130 = adj_lane3*hessian_tmp128 + adj_lane4*hessian_tmp125 + adj_lane5*hessian_tmp127;
  const s_t hessian_tmp131 = adj_lane6*hessian_tmp128 + adj_lane7*hessian_tmp125 + adj_lane8*hessian_tmp127;
  const s_t hessian_tmp132 = hessian_tmp123*hessian_tmp52;
  const s_t hessian_tmp133 = ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp47 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp48 + ((s_t(1) / s_t(6)))*hessian_tmp132;
  const s_t hessian_tmp134 = hessian_tmp126*hessian_tmp59;
  const s_t hessian_tmp135 = ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp57 + ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp58 + ((s_t(1) / s_t(6)))*hessian_tmp134;
  const s_t hessian_tmp136 = hessian_tmp124*hessian_tmp66;
  const s_t hessian_tmp137 = ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp64 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp65 + ((s_t(1) / s_t(6)))*hessian_tmp136;
  const s_t hessian_tmp138 = adj_lane0*hessian_tmp135 + adj_lane1*hessian_tmp137 + adj_lane2*hessian_tmp133;
  const s_t hessian_tmp139 = adj_lane3*hessian_tmp135 + adj_lane4*hessian_tmp137 + adj_lane5*hessian_tmp133;
  const s_t hessian_tmp140 = adj_lane6*hessian_tmp135 + adj_lane7*hessian_tmp137 + adj_lane8*hessian_tmp133;
  const s_t hessian_tmp141 = hessian_tmp124*hessian_tmp78;
  const s_t hessian_tmp142 = ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp74 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp73 + ((s_t(1) / s_t(6)))*hessian_tmp141;
  const s_t hessian_tmp143 = hessian_tmp126*hessian_tmp84;
  const s_t hessian_tmp144 = ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp83 + ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp82 + ((s_t(1) / s_t(6)))*hessian_tmp143;
  const s_t hessian_tmp145 = hessian_tmp123*hessian_tmp90;
  const s_t hessian_tmp146 = ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp88 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp89 + ((s_t(1) / s_t(6)))*hessian_tmp145;
  const s_t hessian_tmp147 = adj_lane0*hessian_tmp144 + adj_lane1*hessian_tmp142 + adj_lane2*hessian_tmp146;
  const s_t hessian_tmp148 = adj_lane3*hessian_tmp144 + adj_lane4*hessian_tmp142 + adj_lane5*hessian_tmp146;
  const s_t hessian_tmp149 = adj_lane6*hessian_tmp144 + adj_lane7*hessian_tmp142 + adj_lane8*hessian_tmp146;
  const s_t hessian_tmp150 = adj_lane8*idet;
  const s_t hessian_tmp151 = adj_lane7*idet;
  const s_t hessian_tmp152 = ((s_t(1) / s_t(6)))*adj_lane6*hessian_tmp12 + ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp5 + ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp24;
  const s_t hessian_tmp153 = adj_lane6*idet;
  const s_t hessian_tmp154 = ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp36 + ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp5 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp31;
  const s_t hessian_tmp155 = ((s_t(1) / s_t(6)))*adj_lane7*hessian_tmp12 + ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp31 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp40;
  const s_t hessian_tmp156 = adj_lane0*hessian_tmp155 + adj_lane1*hessian_tmp152 + adj_lane2*hessian_tmp154;
  const s_t hessian_tmp157 = adj_lane3*hessian_tmp155 + adj_lane4*hessian_tmp152 + adj_lane5*hessian_tmp154;
  const s_t hessian_tmp158 = adj_lane6*hessian_tmp155 + adj_lane7*hessian_tmp152 + adj_lane8*hessian_tmp154;
  const s_t hessian_tmp159 = hessian_tmp150*hessian_tmp52;
  const s_t hessian_tmp160 = ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp47 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp48 + ((s_t(1) / s_t(6)))*hessian_tmp159;
  const s_t hessian_tmp161 = hessian_tmp153*hessian_tmp59;
  const s_t hessian_tmp162 = ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp57 + ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp58 + ((s_t(1) / s_t(6)))*hessian_tmp161;
  const s_t hessian_tmp163 = hessian_tmp151*hessian_tmp66;
  const s_t hessian_tmp164 = ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp64 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp65 + ((s_t(1) / s_t(6)))*hessian_tmp163;
  const s_t hessian_tmp165 = adj_lane0*hessian_tmp162 + adj_lane1*hessian_tmp164 + adj_lane2*hessian_tmp160;
  const s_t hessian_tmp166 = adj_lane3*hessian_tmp162 + adj_lane4*hessian_tmp164 + adj_lane5*hessian_tmp160;
  const s_t hessian_tmp167 = adj_lane6*hessian_tmp162 + adj_lane7*hessian_tmp164 + adj_lane8*hessian_tmp160;
  const s_t hessian_tmp168 = hessian_tmp151*hessian_tmp78;
  const s_t hessian_tmp169 = ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp74 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp73 + ((s_t(1) / s_t(6)))*hessian_tmp168;
  const s_t hessian_tmp170 = hessian_tmp153*hessian_tmp84;
  const s_t hessian_tmp171 = ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp83 + ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp82 + ((s_t(1) / s_t(6)))*hessian_tmp170;
  const s_t hessian_tmp172 = hessian_tmp150*hessian_tmp90;
  const s_t hessian_tmp173 = ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp88 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp89 + ((s_t(1) / s_t(6)))*hessian_tmp172;
  const s_t hessian_tmp174 = adj_lane0*hessian_tmp171 + adj_lane1*hessian_tmp169 + adj_lane2*hessian_tmp173;
  const s_t hessian_tmp175 = adj_lane3*hessian_tmp171 + adj_lane4*hessian_tmp169 + adj_lane5*hessian_tmp173;
  const s_t hessian_tmp176 = adj_lane6*hessian_tmp171 + adj_lane7*hessian_tmp169 + adj_lane8*hessian_tmp173;
  const s_t hessian_tmp177 = ((s_t(1) / s_t(6)))*hessian_tmp26*hessian_tmp64 + ((s_t(1) / s_t(6)))*hessian_tmp32*hessian_tmp57 + ((s_t(1) / s_t(6)))*hessian_tmp53;
  const s_t hessian_tmp178 = ((s_t(1) / s_t(6)))*hessian_tmp26*hessian_tmp65 + ((s_t(1) / s_t(6)))*hessian_tmp48*hessian_tmp6 + ((s_t(1) / s_t(6)))*hessian_tmp60;
  const s_t hessian_tmp179 = ((s_t(1) / s_t(6)))*hessian_tmp32*hessian_tmp58 + ((s_t(1) / s_t(6)))*hessian_tmp47*hessian_tmp6 + ((s_t(1) / s_t(6)))*hessian_tmp67;
  const s_t hessian_tmp180 = adj_lane0*hessian_tmp178 + adj_lane1*hessian_tmp179 + adj_lane2*hessian_tmp177;
  const s_t hessian_tmp181 = adj_lane3*hessian_tmp178 + adj_lane4*hessian_tmp179 + adj_lane5*hessian_tmp177;
  const s_t hessian_tmp182 = adj_lane6*hessian_tmp178 + adj_lane7*hessian_tmp179 + adj_lane8*hessian_tmp177;
  const s_t hessian_tmp183 = hessian_tmp29*lmbda + mu*(hessian_tmp28 + s_t(2)*hessian_tmp29 + hessian_tmp30);
  const s_t hessian_tmp184 = hessian_tmp11*lmbda + mu*(hessian_tmp10 + s_t(2)*hessian_tmp11 + hessian_tmp9);
  const s_t hessian_tmp185 = hessian_tmp20*lmbda + hessian_tmp23 + mu*(hessian_tmp19 + s_t(3)*hessian_tmp20 + hessian_tmp22 + hessian_tmp33);
  const s_t hessian_tmp186 = ((s_t(1) / s_t(6)))*hessian_tmp183*hessian_tmp6 + ((s_t(1) / s_t(6)))*hessian_tmp184*hessian_tmp26 + ((s_t(1) / s_t(6)))*hessian_tmp185*hessian_tmp32;
  const s_t hessian_tmp187 = hessian_tmp2*lmbda + mu*(hessian_tmp0 + s_t(2)*hessian_tmp2 + hessian_tmp4);
  const s_t hessian_tmp188 = hessian_tmp18 + hessian_tmp34;
  const s_t hessian_tmp189 = hessian_tmp21*lmbda + hessian_tmp23 + mu*(hessian_tmp16 + hessian_tmp188 + hessian_tmp20 + s_t(3)*hessian_tmp21);
  const s_t hessian_tmp190 = ((s_t(1) / s_t(6)))*hessian_tmp183*hessian_tmp32 + ((s_t(1) / s_t(6)))*hessian_tmp187*hessian_tmp26 + ((s_t(1) / s_t(6)))*hessian_tmp189*hessian_tmp6;
  const s_t hessian_tmp191 = hessian_tmp18*lmbda + hessian_tmp23 + mu*(hessian_tmp14 + s_t(3)*hessian_tmp18 + hessian_tmp33 + hessian_tmp38);
  const s_t hessian_tmp192 = ((s_t(1) / s_t(6)))*hessian_tmp184*hessian_tmp32 + ((s_t(1) / s_t(6)))*hessian_tmp187*hessian_tmp6 + ((s_t(1) / s_t(6)))*hessian_tmp191*hessian_tmp26;
  const s_t hessian_tmp193 = adj_lane0*hessian_tmp186 + adj_lane1*hessian_tmp192 + adj_lane2*hessian_tmp190;
  const s_t hessian_tmp194 = adj_lane3*hessian_tmp186 + adj_lane4*hessian_tmp192 + adj_lane5*hessian_tmp190;
  const s_t hessian_tmp195 = adj_lane6*hessian_tmp186 + adj_lane7*hessian_tmp192 + adj_lane8*hessian_tmp190;
  const s_t hessian_tmp196 = gu3*mu;
  const s_t hessian_tmp197 = gu6*hessian_tmp63 + gu7*hessian_tmp196;
  const s_t hessian_tmp198 = gu6*hessian_tmp45 + hessian_tmp196*hessian_tmp3;
  const s_t hessian_tmp199 = gu3*gu6;
  const s_t hessian_tmp200 = gu5*hessian_tmp3;
  const s_t hessian_tmp201 = gu7*hessian_tmp1;
  const s_t hessian_tmp202 = hessian_tmp199*lmbda + mu*(s_t(2)*hessian_tmp199 + hessian_tmp200 + hessian_tmp201);
  const s_t hessian_tmp203 = hessian_tmp202*hessian_tmp32;
  const s_t hessian_tmp204 = ((s_t(1) / s_t(6)))*hessian_tmp197*hessian_tmp26 + ((s_t(1) / s_t(6)))*hessian_tmp198*hessian_tmp6 + ((s_t(1) / s_t(6)))*hessian_tmp203;
  const s_t hessian_tmp205 = hessian_tmp1*mu;
  const s_t hessian_tmp206 = gu6*hessian_tmp205 + gu7*hessian_tmp55;
  const s_t hessian_tmp207 = gu7*hessian_tmp45 + hessian_tmp205*hessian_tmp3;
  const s_t hessian_tmp208 = hessian_tmp201*lmbda + mu*(hessian_tmp199 + hessian_tmp200 + s_t(2)*hessian_tmp201);
  const s_t hessian_tmp209 = hessian_tmp208*hessian_tmp26;
  const s_t hessian_tmp210 = ((s_t(1) / s_t(6)))*hessian_tmp206*hessian_tmp32 + ((s_t(1) / s_t(6)))*hessian_tmp207*hessian_tmp6 + ((s_t(1) / s_t(6)))*hessian_tmp209;
  const s_t hessian_tmp211 = gu5*mu;
  const s_t hessian_tmp212 = gu6*hessian_tmp211 + hessian_tmp3*hessian_tmp55;
  const s_t hessian_tmp213 = gu7*hessian_tmp211 + hessian_tmp3*hessian_tmp63;
  const s_t hessian_tmp214 = hessian_tmp200*lmbda + mu*(hessian_tmp199 + s_t(2)*hessian_tmp200 + hessian_tmp201);
  const s_t hessian_tmp215 = hessian_tmp214*hessian_tmp6;
  const s_t hessian_tmp216 = ((s_t(1) / s_t(6)))*hessian_tmp212*hessian_tmp32 + ((s_t(1) / s_t(6)))*hessian_tmp213*hessian_tmp26 + ((s_t(1) / s_t(6)))*hessian_tmp215;
  const s_t hessian_tmp217 = adj_lane0*hessian_tmp204 + adj_lane1*hessian_tmp210 + adj_lane2*hessian_tmp216;
  const s_t hessian_tmp218 = adj_lane3*hessian_tmp204 + adj_lane4*hessian_tmp210 + adj_lane5*hessian_tmp216;
  const s_t hessian_tmp219 = adj_lane6*hessian_tmp204 + adj_lane7*hessian_tmp210 + adj_lane8*hessian_tmp216;
  const s_t hessian_tmp220 = ((s_t(1) / s_t(6)))*hessian_tmp105 + ((s_t(1) / s_t(6)))*hessian_tmp57*hessian_tmp99 + ((s_t(1) / s_t(6)))*hessian_tmp64*hessian_tmp97;
  const s_t hessian_tmp221 = ((s_t(1) / s_t(6)))*hessian_tmp107 + ((s_t(1) / s_t(6)))*hessian_tmp48*hessian_tmp96 + ((s_t(1) / s_t(6)))*hessian_tmp65*hessian_tmp97;
  const s_t hessian_tmp222 = ((s_t(1) / s_t(6)))*hessian_tmp109 + ((s_t(1) / s_t(6)))*hessian_tmp47*hessian_tmp96 + ((s_t(1) / s_t(6)))*hessian_tmp58*hessian_tmp99;
  const s_t hessian_tmp223 = adj_lane0*hessian_tmp221 + adj_lane1*hessian_tmp222 + adj_lane2*hessian_tmp220;
  const s_t hessian_tmp224 = adj_lane3*hessian_tmp221 + adj_lane4*hessian_tmp222 + adj_lane5*hessian_tmp220;
  const s_t hessian_tmp225 = adj_lane6*hessian_tmp221 + adj_lane7*hessian_tmp222 + adj_lane8*hessian_tmp220;
  const s_t hessian_tmp226 = ((s_t(1) / s_t(6)))*hessian_tmp183*hessian_tmp96 + ((s_t(1) / s_t(6)))*hessian_tmp184*hessian_tmp97 + ((s_t(1) / s_t(6)))*hessian_tmp185*hessian_tmp99;
  const s_t hessian_tmp227 = ((s_t(1) / s_t(6)))*hessian_tmp183*hessian_tmp99 + ((s_t(1) / s_t(6)))*hessian_tmp187*hessian_tmp97 + ((s_t(1) / s_t(6)))*hessian_tmp189*hessian_tmp96;
  const s_t hessian_tmp228 = ((s_t(1) / s_t(6)))*hessian_tmp184*hessian_tmp99 + ((s_t(1) / s_t(6)))*hessian_tmp187*hessian_tmp96 + ((s_t(1) / s_t(6)))*hessian_tmp191*hessian_tmp97;
  const s_t hessian_tmp229 = adj_lane0*hessian_tmp226 + adj_lane1*hessian_tmp228 + adj_lane2*hessian_tmp227;
  const s_t hessian_tmp230 = adj_lane3*hessian_tmp226 + adj_lane4*hessian_tmp228 + adj_lane5*hessian_tmp227;
  const s_t hessian_tmp231 = adj_lane6*hessian_tmp226 + adj_lane7*hessian_tmp228 + adj_lane8*hessian_tmp227;
  const s_t hessian_tmp232 = hessian_tmp202*hessian_tmp99;
  const s_t hessian_tmp233 = ((s_t(1) / s_t(6)))*hessian_tmp197*hessian_tmp97 + ((s_t(1) / s_t(6)))*hessian_tmp198*hessian_tmp96 + ((s_t(1) / s_t(6)))*hessian_tmp232;
  const s_t hessian_tmp234 = hessian_tmp208*hessian_tmp97;
  const s_t hessian_tmp235 = ((s_t(1) / s_t(6)))*hessian_tmp206*hessian_tmp99 + ((s_t(1) / s_t(6)))*hessian_tmp207*hessian_tmp96 + ((s_t(1) / s_t(6)))*hessian_tmp234;
  const s_t hessian_tmp236 = hessian_tmp214*hessian_tmp96;
  const s_t hessian_tmp237 = ((s_t(1) / s_t(6)))*hessian_tmp212*hessian_tmp99 + ((s_t(1) / s_t(6)))*hessian_tmp213*hessian_tmp97 + ((s_t(1) / s_t(6)))*hessian_tmp236;
  const s_t hessian_tmp238 = adj_lane0*hessian_tmp233 + adj_lane1*hessian_tmp235 + adj_lane2*hessian_tmp237;
  const s_t hessian_tmp239 = adj_lane3*hessian_tmp233 + adj_lane4*hessian_tmp235 + adj_lane5*hessian_tmp237;
  const s_t hessian_tmp240 = adj_lane6*hessian_tmp233 + adj_lane7*hessian_tmp235 + adj_lane8*hessian_tmp237;
  const s_t hessian_tmp241 = ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp64 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp57 + ((s_t(1) / s_t(6)))*hessian_tmp132;
  const s_t hessian_tmp242 = ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp48 + ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp65 + ((s_t(1) / s_t(6)))*hessian_tmp134;
  const s_t hessian_tmp243 = ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp47 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp58 + ((s_t(1) / s_t(6)))*hessian_tmp136;
  const s_t hessian_tmp244 = adj_lane0*hessian_tmp242 + adj_lane1*hessian_tmp243 + adj_lane2*hessian_tmp241;
  const s_t hessian_tmp245 = adj_lane3*hessian_tmp242 + adj_lane4*hessian_tmp243 + adj_lane5*hessian_tmp241;
  const s_t hessian_tmp246 = adj_lane6*hessian_tmp242 + adj_lane7*hessian_tmp243 + adj_lane8*hessian_tmp241;
  const s_t hessian_tmp247 = ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp183 + ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp184 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp185;
  const s_t hessian_tmp248 = ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp189 + ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp187 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp183;
  const s_t hessian_tmp249 = ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp187 + ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp191 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp184;
  const s_t hessian_tmp250 = adj_lane0*hessian_tmp247 + adj_lane1*hessian_tmp249 + adj_lane2*hessian_tmp248;
  const s_t hessian_tmp251 = adj_lane3*hessian_tmp247 + adj_lane4*hessian_tmp249 + adj_lane5*hessian_tmp248;
  const s_t hessian_tmp252 = adj_lane6*hessian_tmp247 + adj_lane7*hessian_tmp249 + adj_lane8*hessian_tmp248;
  const s_t hessian_tmp253 = hessian_tmp126*hessian_tmp202;
  const s_t hessian_tmp254 = ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp198 + ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp197 + ((s_t(1) / s_t(6)))*hessian_tmp253;
  const s_t hessian_tmp255 = hessian_tmp124*hessian_tmp208;
  const s_t hessian_tmp256 = ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp207 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp206 + ((s_t(1) / s_t(6)))*hessian_tmp255;
  const s_t hessian_tmp257 = hessian_tmp123*hessian_tmp214;
  const s_t hessian_tmp258 = ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp213 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp212 + ((s_t(1) / s_t(6)))*hessian_tmp257;
  const s_t hessian_tmp259 = adj_lane0*hessian_tmp254 + adj_lane1*hessian_tmp256 + adj_lane2*hessian_tmp258;
  const s_t hessian_tmp260 = adj_lane3*hessian_tmp254 + adj_lane4*hessian_tmp256 + adj_lane5*hessian_tmp258;
  const s_t hessian_tmp261 = adj_lane6*hessian_tmp254 + adj_lane7*hessian_tmp256 + adj_lane8*hessian_tmp258;
  const s_t hessian_tmp262 = ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp64 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp57 + ((s_t(1) / s_t(6)))*hessian_tmp159;
  const s_t hessian_tmp263 = ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp48 + ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp65 + ((s_t(1) / s_t(6)))*hessian_tmp161;
  const s_t hessian_tmp264 = ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp47 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp58 + ((s_t(1) / s_t(6)))*hessian_tmp163;
  const s_t hessian_tmp265 = adj_lane0*hessian_tmp263 + adj_lane1*hessian_tmp264 + adj_lane2*hessian_tmp262;
  const s_t hessian_tmp266 = adj_lane3*hessian_tmp263 + adj_lane4*hessian_tmp264 + adj_lane5*hessian_tmp262;
  const s_t hessian_tmp267 = adj_lane6*hessian_tmp263 + adj_lane7*hessian_tmp264 + adj_lane8*hessian_tmp262;
  const s_t hessian_tmp268 = ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp183 + ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp184 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp185;
  const s_t hessian_tmp269 = ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp189 + ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp187 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp183;
  const s_t hessian_tmp270 = ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp187 + ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp191 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp184;
  const s_t hessian_tmp271 = adj_lane0*hessian_tmp268 + adj_lane1*hessian_tmp270 + adj_lane2*hessian_tmp269;
  const s_t hessian_tmp272 = adj_lane3*hessian_tmp268 + adj_lane4*hessian_tmp270 + adj_lane5*hessian_tmp269;
  const s_t hessian_tmp273 = adj_lane6*hessian_tmp268 + adj_lane7*hessian_tmp270 + adj_lane8*hessian_tmp269;
  const s_t hessian_tmp274 = hessian_tmp153*hessian_tmp202;
  const s_t hessian_tmp275 = ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp198 + ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp197 + ((s_t(1) / s_t(6)))*hessian_tmp274;
  const s_t hessian_tmp276 = hessian_tmp151*hessian_tmp208;
  const s_t hessian_tmp277 = ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp207 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp206 + ((s_t(1) / s_t(6)))*hessian_tmp276;
  const s_t hessian_tmp278 = hessian_tmp150*hessian_tmp214;
  const s_t hessian_tmp279 = ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp213 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp212 + ((s_t(1) / s_t(6)))*hessian_tmp278;
  const s_t hessian_tmp280 = adj_lane0*hessian_tmp275 + adj_lane1*hessian_tmp277 + adj_lane2*hessian_tmp279;
  const s_t hessian_tmp281 = adj_lane3*hessian_tmp275 + adj_lane4*hessian_tmp277 + adj_lane5*hessian_tmp279;
  const s_t hessian_tmp282 = adj_lane6*hessian_tmp275 + adj_lane7*hessian_tmp277 + adj_lane8*hessian_tmp279;
  const s_t hessian_tmp283 = ((s_t(1) / s_t(6)))*hessian_tmp32*hessian_tmp82 + ((s_t(1) / s_t(6)))*hessian_tmp6*hessian_tmp88 + ((s_t(1) / s_t(6)))*hessian_tmp79;
  const s_t hessian_tmp284 = ((s_t(1) / s_t(6)))*hessian_tmp26*hessian_tmp73 + ((s_t(1) / s_t(6)))*hessian_tmp6*hessian_tmp89 + ((s_t(1) / s_t(6)))*hessian_tmp85;
  const s_t hessian_tmp285 = ((s_t(1) / s_t(6)))*hessian_tmp26*hessian_tmp74 + ((s_t(1) / s_t(6)))*hessian_tmp32*hessian_tmp83 + ((s_t(1) / s_t(6)))*hessian_tmp91;
  const s_t hessian_tmp286 = adj_lane0*hessian_tmp284 + adj_lane1*hessian_tmp283 + adj_lane2*hessian_tmp285;
  const s_t hessian_tmp287 = adj_lane3*hessian_tmp284 + adj_lane4*hessian_tmp283 + adj_lane5*hessian_tmp285;
  const s_t hessian_tmp288 = adj_lane6*hessian_tmp284 + adj_lane7*hessian_tmp283 + adj_lane8*hessian_tmp285;
  const s_t hessian_tmp289 = ((s_t(1) / s_t(6)))*hessian_tmp203 + ((s_t(1) / s_t(6)))*hessian_tmp206*hessian_tmp26 + ((s_t(1) / s_t(6)))*hessian_tmp212*hessian_tmp6;
  const s_t hessian_tmp290 = ((s_t(1) / s_t(6)))*hessian_tmp197*hessian_tmp32 + ((s_t(1) / s_t(6)))*hessian_tmp209 + ((s_t(1) / s_t(6)))*hessian_tmp213*hessian_tmp6;
  const s_t hessian_tmp291 = ((s_t(1) / s_t(6)))*hessian_tmp198*hessian_tmp32 + ((s_t(1) / s_t(6)))*hessian_tmp207*hessian_tmp26 + ((s_t(1) / s_t(6)))*hessian_tmp215;
  const s_t hessian_tmp292 = adj_lane0*hessian_tmp289 + adj_lane1*hessian_tmp290 + adj_lane2*hessian_tmp291;
  const s_t hessian_tmp293 = adj_lane3*hessian_tmp289 + adj_lane4*hessian_tmp290 + adj_lane5*hessian_tmp291;
  const s_t hessian_tmp294 = adj_lane6*hessian_tmp289 + adj_lane7*hessian_tmp290 + adj_lane8*hessian_tmp291;
  const s_t hessian_tmp295 = hessian_tmp10*lmbda + mu*(s_t(2)*hessian_tmp10 + hessian_tmp11 + hessian_tmp9);
  const s_t hessian_tmp296 = hessian_tmp30*lmbda + mu*(hessian_tmp28 + hessian_tmp29 + s_t(2)*hessian_tmp30);
  const s_t hessian_tmp297 = hessian_tmp22*lmbda + hessian_tmp23 + mu*(hessian_tmp14 + hessian_tmp20 + s_t(3)*hessian_tmp22 + hessian_tmp35 + s_t(-1));
  const s_t hessian_tmp298 = ((s_t(1) / s_t(6)))*hessian_tmp26*hessian_tmp295 + ((s_t(1) / s_t(6)))*hessian_tmp296*hessian_tmp6 + ((s_t(1) / s_t(6)))*hessian_tmp297*hessian_tmp32;
  const s_t hessian_tmp299 = hessian_tmp4*lmbda + mu*(hessian_tmp0 + hessian_tmp2 + s_t(2)*hessian_tmp4);
  const s_t hessian_tmp300 = hessian_tmp14*lmbda + hessian_tmp23 + mu*(hessian_tmp13 + s_t(3)*hessian_tmp14 + hessian_tmp188 + hessian_tmp22 + s_t(-1));
  const s_t hessian_tmp301 = ((s_t(1) / s_t(6)))*hessian_tmp26*hessian_tmp300 + ((s_t(1) / s_t(6)))*hessian_tmp295*hessian_tmp32 + ((s_t(1) / s_t(6)))*hessian_tmp299*hessian_tmp6;
  const s_t hessian_tmp302 = hessian_tmp23 + hessian_tmp34*lmbda + mu*(hessian_tmp14 + hessian_tmp21 + s_t(3)*hessian_tmp34 + hessian_tmp39);
  const s_t hessian_tmp303 = ((s_t(1) / s_t(6)))*hessian_tmp26*hessian_tmp299 + ((s_t(1) / s_t(6)))*hessian_tmp296*hessian_tmp32 + ((s_t(1) / s_t(6)))*hessian_tmp302*hessian_tmp6;
  const s_t hessian_tmp304 = adj_lane0*hessian_tmp298 + adj_lane1*hessian_tmp301 + adj_lane2*hessian_tmp303;
  const s_t hessian_tmp305 = adj_lane3*hessian_tmp298 + adj_lane4*hessian_tmp301 + adj_lane5*hessian_tmp303;
  const s_t hessian_tmp306 = adj_lane6*hessian_tmp298 + adj_lane7*hessian_tmp301 + adj_lane8*hessian_tmp303;
  const s_t hessian_tmp307 = ((s_t(1) / s_t(6)))*hessian_tmp114 + ((s_t(1) / s_t(6)))*hessian_tmp82*hessian_tmp99 + ((s_t(1) / s_t(6)))*hessian_tmp88*hessian_tmp96;
  const s_t hessian_tmp308 = ((s_t(1) / s_t(6)))*hessian_tmp116 + ((s_t(1) / s_t(6)))*hessian_tmp73*hessian_tmp97 + ((s_t(1) / s_t(6)))*hessian_tmp89*hessian_tmp96;
  const s_t hessian_tmp309 = ((s_t(1) / s_t(6)))*hessian_tmp118 + ((s_t(1) / s_t(6)))*hessian_tmp74*hessian_tmp97 + ((s_t(1) / s_t(6)))*hessian_tmp83*hessian_tmp99;
  const s_t hessian_tmp310 = adj_lane0*hessian_tmp308 + adj_lane1*hessian_tmp307 + adj_lane2*hessian_tmp309;
  const s_t hessian_tmp311 = adj_lane3*hessian_tmp308 + adj_lane4*hessian_tmp307 + adj_lane5*hessian_tmp309;
  const s_t hessian_tmp312 = adj_lane6*hessian_tmp308 + adj_lane7*hessian_tmp307 + adj_lane8*hessian_tmp309;
  const s_t hessian_tmp313 = ((s_t(1) / s_t(6)))*hessian_tmp206*hessian_tmp97 + ((s_t(1) / s_t(6)))*hessian_tmp212*hessian_tmp96 + ((s_t(1) / s_t(6)))*hessian_tmp232;
  const s_t hessian_tmp314 = ((s_t(1) / s_t(6)))*hessian_tmp197*hessian_tmp99 + ((s_t(1) / s_t(6)))*hessian_tmp213*hessian_tmp96 + ((s_t(1) / s_t(6)))*hessian_tmp234;
  const s_t hessian_tmp315 = ((s_t(1) / s_t(6)))*hessian_tmp198*hessian_tmp99 + ((s_t(1) / s_t(6)))*hessian_tmp207*hessian_tmp97 + ((s_t(1) / s_t(6)))*hessian_tmp236;
  const s_t hessian_tmp316 = adj_lane0*hessian_tmp313 + adj_lane1*hessian_tmp314 + adj_lane2*hessian_tmp315;
  const s_t hessian_tmp317 = adj_lane3*hessian_tmp313 + adj_lane4*hessian_tmp314 + adj_lane5*hessian_tmp315;
  const s_t hessian_tmp318 = adj_lane6*hessian_tmp313 + adj_lane7*hessian_tmp314 + adj_lane8*hessian_tmp315;
  const s_t hessian_tmp319 = ((s_t(1) / s_t(6)))*hessian_tmp295*hessian_tmp97 + ((s_t(1) / s_t(6)))*hessian_tmp296*hessian_tmp96 + ((s_t(1) / s_t(6)))*hessian_tmp297*hessian_tmp99;
  const s_t hessian_tmp320 = ((s_t(1) / s_t(6)))*hessian_tmp295*hessian_tmp99 + ((s_t(1) / s_t(6)))*hessian_tmp299*hessian_tmp96 + ((s_t(1) / s_t(6)))*hessian_tmp300*hessian_tmp97;
  const s_t hessian_tmp321 = ((s_t(1) / s_t(6)))*hessian_tmp296*hessian_tmp99 + ((s_t(1) / s_t(6)))*hessian_tmp299*hessian_tmp97 + ((s_t(1) / s_t(6)))*hessian_tmp302*hessian_tmp96;
  const s_t hessian_tmp322 = adj_lane0*hessian_tmp319 + adj_lane1*hessian_tmp320 + adj_lane2*hessian_tmp321;
  const s_t hessian_tmp323 = adj_lane3*hessian_tmp319 + adj_lane4*hessian_tmp320 + adj_lane5*hessian_tmp321;
  const s_t hessian_tmp324 = adj_lane6*hessian_tmp319 + adj_lane7*hessian_tmp320 + adj_lane8*hessian_tmp321;
  const s_t hessian_tmp325 = ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp88 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp82 + ((s_t(1) / s_t(6)))*hessian_tmp141;
  const s_t hessian_tmp326 = ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp89 + ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp73 + ((s_t(1) / s_t(6)))*hessian_tmp143;
  const s_t hessian_tmp327 = ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp74 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp83 + ((s_t(1) / s_t(6)))*hessian_tmp145;
  const s_t hessian_tmp328 = adj_lane0*hessian_tmp326 + adj_lane1*hessian_tmp325 + adj_lane2*hessian_tmp327;
  const s_t hessian_tmp329 = adj_lane3*hessian_tmp326 + adj_lane4*hessian_tmp325 + adj_lane5*hessian_tmp327;
  const s_t hessian_tmp330 = adj_lane6*hessian_tmp326 + adj_lane7*hessian_tmp325 + adj_lane8*hessian_tmp327;
  const s_t hessian_tmp331 = ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp212 + ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp206 + ((s_t(1) / s_t(6)))*hessian_tmp253;
  const s_t hessian_tmp332 = ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp213 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp197 + ((s_t(1) / s_t(6)))*hessian_tmp255;
  const s_t hessian_tmp333 = ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp207 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp198 + ((s_t(1) / s_t(6)))*hessian_tmp257;
  const s_t hessian_tmp334 = adj_lane0*hessian_tmp331 + adj_lane1*hessian_tmp332 + adj_lane2*hessian_tmp333;
  const s_t hessian_tmp335 = adj_lane3*hessian_tmp331 + adj_lane4*hessian_tmp332 + adj_lane5*hessian_tmp333;
  const s_t hessian_tmp336 = adj_lane6*hessian_tmp331 + adj_lane7*hessian_tmp332 + adj_lane8*hessian_tmp333;
  const s_t hessian_tmp337 = ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp296 + ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp295 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp297;
  const s_t hessian_tmp338 = ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp299 + ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp300 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp295;
  const s_t hessian_tmp339 = ((s_t(1) / s_t(6)))*hessian_tmp123*hessian_tmp302 + ((s_t(1) / s_t(6)))*hessian_tmp124*hessian_tmp299 + ((s_t(1) / s_t(6)))*hessian_tmp126*hessian_tmp296;
  const s_t hessian_tmp340 = adj_lane0*hessian_tmp337 + adj_lane1*hessian_tmp338 + adj_lane2*hessian_tmp339;
  const s_t hessian_tmp341 = adj_lane3*hessian_tmp337 + adj_lane4*hessian_tmp338 + adj_lane5*hessian_tmp339;
  const s_t hessian_tmp342 = adj_lane6*hessian_tmp337 + adj_lane7*hessian_tmp338 + adj_lane8*hessian_tmp339;
  const s_t hessian_tmp343 = ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp88 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp82 + ((s_t(1) / s_t(6)))*hessian_tmp168;
  const s_t hessian_tmp344 = ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp89 + ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp73 + ((s_t(1) / s_t(6)))*hessian_tmp170;
  const s_t hessian_tmp345 = ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp74 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp83 + ((s_t(1) / s_t(6)))*hessian_tmp172;
  const s_t hessian_tmp346 = adj_lane0*hessian_tmp344 + adj_lane1*hessian_tmp343 + adj_lane2*hessian_tmp345;
  const s_t hessian_tmp347 = adj_lane3*hessian_tmp344 + adj_lane4*hessian_tmp343 + adj_lane5*hessian_tmp345;
  const s_t hessian_tmp348 = adj_lane6*hessian_tmp344 + adj_lane7*hessian_tmp343 + adj_lane8*hessian_tmp345;
  const s_t hessian_tmp349 = ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp212 + ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp206 + ((s_t(1) / s_t(6)))*hessian_tmp274;
  const s_t hessian_tmp350 = ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp213 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp197 + ((s_t(1) / s_t(6)))*hessian_tmp276;
  const s_t hessian_tmp351 = ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp207 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp198 + ((s_t(1) / s_t(6)))*hessian_tmp278;
  const s_t hessian_tmp352 = adj_lane0*hessian_tmp349 + adj_lane1*hessian_tmp350 + adj_lane2*hessian_tmp351;
  const s_t hessian_tmp353 = adj_lane3*hessian_tmp349 + adj_lane4*hessian_tmp350 + adj_lane5*hessian_tmp351;
  const s_t hessian_tmp354 = adj_lane6*hessian_tmp349 + adj_lane7*hessian_tmp350 + adj_lane8*hessian_tmp351;
  const s_t hessian_tmp355 = ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp296 + ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp295 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp297;
  const s_t hessian_tmp356 = ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp299 + ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp300 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp295;
  const s_t hessian_tmp357 = ((s_t(1) / s_t(6)))*hessian_tmp150*hessian_tmp302 + ((s_t(1) / s_t(6)))*hessian_tmp151*hessian_tmp299 + ((s_t(1) / s_t(6)))*hessian_tmp153*hessian_tmp296;
  const s_t hessian_tmp358 = adj_lane0*hessian_tmp355 + adj_lane1*hessian_tmp356 + adj_lane2*hessian_tmp357;
  const s_t hessian_tmp359 = adj_lane3*hessian_tmp355 + adj_lane4*hessian_tmp356 + adj_lane5*hessian_tmp357;
  const s_t hessian_tmp360 = adj_lane6*hessian_tmp355 + adj_lane7*hessian_tmp356 + adj_lane8*hessian_tmp357;
  element_matrix[0] = -hessian_tmp42 - hessian_tmp43 - hessian_tmp44;
  element_matrix[12] = hessian_tmp42;
  element_matrix[24] = hessian_tmp43;
  element_matrix[36] = hessian_tmp44;
  element_matrix[48] = -hessian_tmp69 - hessian_tmp70 - hessian_tmp71;
  element_matrix[60] = hessian_tmp69;
  element_matrix[72] = hessian_tmp70;
  element_matrix[84] = hessian_tmp71;
  element_matrix[96] = -hessian_tmp93 - hessian_tmp94 - hessian_tmp95;
  element_matrix[108] = hessian_tmp93;
  element_matrix[120] = hessian_tmp94;
  element_matrix[132] = hessian_tmp95;
  element_matrix[1] = -hessian_tmp102 - hessian_tmp103 - hessian_tmp104;
  element_matrix[13] = hessian_tmp102;
  element_matrix[25] = hessian_tmp103;
  element_matrix[37] = hessian_tmp104;
  element_matrix[49] = -hessian_tmp111 - hessian_tmp112 - hessian_tmp113;
  element_matrix[61] = hessian_tmp111;
  element_matrix[73] = hessian_tmp112;
  element_matrix[85] = hessian_tmp113;
  element_matrix[97] = -hessian_tmp120 - hessian_tmp121 - hessian_tmp122;
  element_matrix[109] = hessian_tmp120;
  element_matrix[121] = hessian_tmp121;
  element_matrix[133] = hessian_tmp122;
  element_matrix[2] = -hessian_tmp129 - hessian_tmp130 - hessian_tmp131;
  element_matrix[14] = hessian_tmp129;
  element_matrix[26] = hessian_tmp130;
  element_matrix[38] = hessian_tmp131;
  element_matrix[50] = -hessian_tmp138 - hessian_tmp139 - hessian_tmp140;
  element_matrix[62] = hessian_tmp138;
  element_matrix[74] = hessian_tmp139;
  element_matrix[86] = hessian_tmp140;
  element_matrix[98] = -hessian_tmp147 - hessian_tmp148 - hessian_tmp149;
  element_matrix[110] = hessian_tmp147;
  element_matrix[122] = hessian_tmp148;
  element_matrix[134] = hessian_tmp149;
  element_matrix[3] = -hessian_tmp156 - hessian_tmp157 - hessian_tmp158;
  element_matrix[15] = hessian_tmp156;
  element_matrix[27] = hessian_tmp157;
  element_matrix[39] = hessian_tmp158;
  element_matrix[51] = -hessian_tmp165 - hessian_tmp166 - hessian_tmp167;
  element_matrix[63] = hessian_tmp165;
  element_matrix[75] = hessian_tmp166;
  element_matrix[87] = hessian_tmp167;
  element_matrix[99] = -hessian_tmp174 - hessian_tmp175 - hessian_tmp176;
  element_matrix[111] = hessian_tmp174;
  element_matrix[123] = hessian_tmp175;
  element_matrix[135] = hessian_tmp176;
  element_matrix[4] = -hessian_tmp180 - hessian_tmp181 - hessian_tmp182;
  element_matrix[16] = hessian_tmp180;
  element_matrix[28] = hessian_tmp181;
  element_matrix[40] = hessian_tmp182;
  element_matrix[52] = -hessian_tmp193 - hessian_tmp194 - hessian_tmp195;
  element_matrix[64] = hessian_tmp193;
  element_matrix[76] = hessian_tmp194;
  element_matrix[88] = hessian_tmp195;
  element_matrix[100] = -hessian_tmp217 - hessian_tmp218 - hessian_tmp219;
  element_matrix[112] = hessian_tmp217;
  element_matrix[124] = hessian_tmp218;
  element_matrix[136] = hessian_tmp219;
  element_matrix[5] = -hessian_tmp223 - hessian_tmp224 - hessian_tmp225;
  element_matrix[17] = hessian_tmp223;
  element_matrix[29] = hessian_tmp224;
  element_matrix[41] = hessian_tmp225;
  element_matrix[53] = -hessian_tmp229 - hessian_tmp230 - hessian_tmp231;
  element_matrix[65] = hessian_tmp229;
  element_matrix[77] = hessian_tmp230;
  element_matrix[89] = hessian_tmp231;
  element_matrix[101] = -hessian_tmp238 - hessian_tmp239 - hessian_tmp240;
  element_matrix[113] = hessian_tmp238;
  element_matrix[125] = hessian_tmp239;
  element_matrix[137] = hessian_tmp240;
  element_matrix[6] = -hessian_tmp244 - hessian_tmp245 - hessian_tmp246;
  element_matrix[18] = hessian_tmp244;
  element_matrix[30] = hessian_tmp245;
  element_matrix[42] = hessian_tmp246;
  element_matrix[54] = -hessian_tmp250 - hessian_tmp251 - hessian_tmp252;
  element_matrix[66] = hessian_tmp250;
  element_matrix[78] = hessian_tmp251;
  element_matrix[90] = hessian_tmp252;
  element_matrix[102] = -hessian_tmp259 - hessian_tmp260 - hessian_tmp261;
  element_matrix[114] = hessian_tmp259;
  element_matrix[126] = hessian_tmp260;
  element_matrix[138] = hessian_tmp261;
  element_matrix[7] = -hessian_tmp265 - hessian_tmp266 - hessian_tmp267;
  element_matrix[19] = hessian_tmp265;
  element_matrix[31] = hessian_tmp266;
  element_matrix[43] = hessian_tmp267;
  element_matrix[55] = -hessian_tmp271 - hessian_tmp272 - hessian_tmp273;
  element_matrix[67] = hessian_tmp271;
  element_matrix[79] = hessian_tmp272;
  element_matrix[91] = hessian_tmp273;
  element_matrix[103] = -hessian_tmp280 - hessian_tmp281 - hessian_tmp282;
  element_matrix[115] = hessian_tmp280;
  element_matrix[127] = hessian_tmp281;
  element_matrix[139] = hessian_tmp282;
  element_matrix[8] = -hessian_tmp286 - hessian_tmp287 - hessian_tmp288;
  element_matrix[20] = hessian_tmp286;
  element_matrix[32] = hessian_tmp287;
  element_matrix[44] = hessian_tmp288;
  element_matrix[56] = -hessian_tmp292 - hessian_tmp293 - hessian_tmp294;
  element_matrix[68] = hessian_tmp292;
  element_matrix[80] = hessian_tmp293;
  element_matrix[92] = hessian_tmp294;
  element_matrix[104] = -hessian_tmp304 - hessian_tmp305 - hessian_tmp306;
  element_matrix[116] = hessian_tmp304;
  element_matrix[128] = hessian_tmp305;
  element_matrix[140] = hessian_tmp306;
  element_matrix[9] = -hessian_tmp310 - hessian_tmp311 - hessian_tmp312;
  element_matrix[21] = hessian_tmp310;
  element_matrix[33] = hessian_tmp311;
  element_matrix[45] = hessian_tmp312;
  element_matrix[57] = -hessian_tmp316 - hessian_tmp317 - hessian_tmp318;
  element_matrix[69] = hessian_tmp316;
  element_matrix[81] = hessian_tmp317;
  element_matrix[93] = hessian_tmp318;
  element_matrix[105] = -hessian_tmp322 - hessian_tmp323 - hessian_tmp324;
  element_matrix[117] = hessian_tmp322;
  element_matrix[129] = hessian_tmp323;
  element_matrix[141] = hessian_tmp324;
  element_matrix[10] = -hessian_tmp328 - hessian_tmp329 - hessian_tmp330;
  element_matrix[22] = hessian_tmp328;
  element_matrix[34] = hessian_tmp329;
  element_matrix[46] = hessian_tmp330;
  element_matrix[58] = -hessian_tmp334 - hessian_tmp335 - hessian_tmp336;
  element_matrix[70] = hessian_tmp334;
  element_matrix[82] = hessian_tmp335;
  element_matrix[94] = hessian_tmp336;
  element_matrix[106] = -hessian_tmp340 - hessian_tmp341 - hessian_tmp342;
  element_matrix[118] = hessian_tmp340;
  element_matrix[130] = hessian_tmp341;
  element_matrix[142] = hessian_tmp342;
  element_matrix[11] = -hessian_tmp346 - hessian_tmp347 - hessian_tmp348;
  element_matrix[23] = hessian_tmp346;
  element_matrix[35] = hessian_tmp347;
  element_matrix[47] = hessian_tmp348;
  element_matrix[59] = -hessian_tmp352 - hessian_tmp353 - hessian_tmp354;
  element_matrix[71] = hessian_tmp352;
  element_matrix[83] = hessian_tmp353;
  element_matrix[95] = hessian_tmp354;
  element_matrix[107] = -hessian_tmp358 - hessian_tmp359 - hessian_tmp360;
  element_matrix[119] = hessian_tmp358;
  element_matrix[131] = hessian_tmp359;
  element_matrix[143] = hessian_tmp360;
}

} // namespace codegen
} // namespace sfem

#endif
