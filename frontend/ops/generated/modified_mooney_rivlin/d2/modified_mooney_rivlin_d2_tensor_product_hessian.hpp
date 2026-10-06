#ifndef MODIFIED_MOONEY_RIVLIN_D2_TENSOR_PRODUCT_HESSIAN_HPP
#define MODIFIED_MOONEY_RIVLIN_D2_TENSOR_PRODUCT_HESSIAN_HPP
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
static SFEM_INLINE void modified_mooney_rivlin_d2_tensor_product_direct_hessian_tensor_product_element_matrix(
    const s_t *const RSTR badj0,
    const s_t *const RSTR badj1,
    const s_t *const RSTR badj2,
    const s_t *const RSTR badj3,
    const s_t *const RSTR bdet0,
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR q_weight_1d,
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
  static constexpr int NQ1 = integer_root(NQ, 2);
  static constexpr int NS1 = integer_root(NS, 2);
  static_assert(ipow(NQ1, 2) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 2) == NS, "NS must be tensor-product compatible");
  for (int entry = 0; entry < NDOFS * NDOFS; ++entry) {
    element_matrix[entry] = s_t(0);
  }
  s_t state_gradient_ref[NC * NQ * ND];
  for (int component = 0; component < NC; ++component) {
    tensor_gradient_contiguous_scalar<s_t, NQ, NS, VS, 2, NC>(
        shape_1d, grad_1d, bu_data, component,
        state_gradient_ref + component * NQ * ND);
  }
  s_t flux[NC * NQ * ND];
  s_t *column[NC * NS];
  for (int trial_component = 0; trial_component < NC; ++trial_component) {
    for (int trial_shape = 0; trial_shape < NS; ++trial_shape) {
      const int trial_sx = trial_shape % NS1;
      const int trial_sy = trial_shape / NS1;
      for (int q = 0; q < NQ; ++q) {
        const int qx = q % NQ1;
        const int qy = q / NQ1;
        const s_t qw = q_weight_1d[qx] * q_weight_1d[qy];
        const int lane = 0;
        const ptrdiff_t goff = q * VS + lane;
        const s_t adj_lane0 = badj0[goff];
        const s_t adj_lane1 = badj1[goff];
        const s_t adj_lane2 = badj2[goff];
        const s_t adj_lane3 = badj3[goff];
        const s_t det_lane0 = bdet0[goff];
        const s_t idet = s_t(1) / det_lane0;
        const s_t gu_ref0 = state_gradient_ref[q * ND];
        const s_t gu_ref1 = state_gradient_ref[q * ND + 1];
        const s_t gu_ref2 = state_gradient_ref[NQ * ND + q * ND];
        const s_t gu_ref3 = state_gradient_ref[NQ * ND + q * ND + 1];
        const s_t gu0 = (gu_ref0 * adj_lane0 + gu_ref1 * adj_lane2) * idet;
        const s_t gu1 = (gu_ref0 * adj_lane1 + gu_ref1 * adj_lane3) * idet;
        const s_t gu2 = (gu_ref2 * adj_lane0 + gu_ref3 * adj_lane2) * idet;
        const s_t gu3 = (gu_ref2 * adj_lane1 + gu_ref3 * adj_lane3) * idet;
        const s_t trial_grad_ref0 = grad_1d[qx * NS1 + trial_sx] * shape_1d[qy * NS1 + trial_sy];
        const s_t trial_grad_ref1 = shape_1d[qx * NS1 + trial_sx] * grad_1d[qy * NS1 + trial_sy];
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
        flux[q * ND] = qw * (material[0] * adj_lane0 + material[1] * adj_lane1);
        flux[q * ND + 1] = qw * (material[0] * adj_lane2 + material[1] * adj_lane3);
        flux[(NQ + q) * ND] = qw * (material[ND] * adj_lane0 + material[ND + 1] * adj_lane1);
        flux[(NQ + q) * ND + 1] = qw * (material[ND] * adj_lane2 + material[ND + 1] * adj_lane3);
      }
      for (int out_shape = 0; out_shape < NS; ++out_shape) {
        column[out_shape * NC + 0] = &element_matrix[(0 * NS + out_shape) * NDOFS + trial_component * NS + trial_shape];
        column[out_shape * NC + 1] = &element_matrix[(1 * NS + out_shape) * NDOFS + trial_component * NS + trial_shape];
      }
      tensor_test_scalar<s_t, NQ, NS, VS, 2, NC>(
          shape_1d, grad_1d, flux + 0, column, 0);
      tensor_test_scalar<s_t, NQ, NS, VS, 2, NC>(
          shape_1d, grad_1d, flux + NQ * ND, column, 1);
    }
  }
}

} // namespace codegen
} // namespace sfem

#endif
