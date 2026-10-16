#ifndef SAINT_VENANT_KIRCHHOFF_D2_SIMPLEX_HESSIAN_CUH
#define SAINT_VENANT_KIRCHHOFF_D2_SIMPLEX_HESSIAN_CUH
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
static __host__ __device__ __forceinline__ void saint_venant_kirchhoff_d2_simplex_direct_hessian_reference_element_matrix(
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
    const s_t adj_value0 = badj0[goff];
    const s_t adj_value1 = badj1[goff];
    const s_t adj_value2 = badj2[goff];
    const s_t adj_value3 = badj3[goff];
    const s_t det_value0 = bdet0[goff];
    const s_t idet = s_t(1) / det_value0;
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
    const s_t gu0 = (gu_ref0 * adj_value0 + gu_ref1 * adj_value2) * idet;
    const s_t gu1 = (gu_ref0 * adj_value1 + gu_ref1 * adj_value3) * idet;
    const s_t gu2 = (gu_ref2 * adj_value0 + gu_ref3 * adj_value2) * idet;
    const s_t gu3 = (gu_ref2 * adj_value1 + gu_ref3 * adj_value3) * idet;
    s_t D[16];
    const s_t weak_hess_tmp0 = gu0 + s_t(1);
    const s_t weak_hess_tmp1 = pow_2(weak_hess_tmp0);
    const s_t weak_hess_tmp2 = pow_2(gu1);
    const s_t weak_hess_tmp3 = pow_2(gu2);
    const s_t weak_hess_tmp4 = weak_hess_tmp2 + weak_hess_tmp3 + s_t(-1);
    const s_t weak_hess_tmp5 = lmbda*(((s_t(1) / s_t(2)))*pow_2(gu0) + gu0 + ((s_t(1) / s_t(2)))*pow_2(gu3) + gu3 + ((s_t(1) / s_t(2)))*weak_hess_tmp2 + ((s_t(1) / s_t(2)))*weak_hess_tmp3);
    const s_t weak_hess_tmp6 = gu1*weak_hess_tmp0;
    const s_t weak_hess_tmp7 = gu3 + s_t(1);
    const s_t weak_hess_tmp8 = gu2*weak_hess_tmp7;
    const s_t weak_hess_tmp9 = gu2*weak_hess_tmp0;
    const s_t weak_hess_tmp10 = gu1*weak_hess_tmp7;
    const s_t weak_hess_tmp11 = gu1*gu2;
    const s_t weak_hess_tmp12 = weak_hess_tmp0*weak_hess_tmp7;
    const s_t weak_hess_tmp13 = pow_2(weak_hess_tmp7);
    const s_t weak_hess_tmp14 = weak_hess_tmp1 + weak_hess_tmp13 + s_t(-1);
    D[0] = lmbda*weak_hess_tmp1 + mu*(s_t(3)*weak_hess_tmp1 + weak_hess_tmp4) + weak_hess_tmp5;
    D[1] = lmbda*weak_hess_tmp6 + mu*(s_t(2)*weak_hess_tmp6 + weak_hess_tmp8);
    D[2] = lmbda*weak_hess_tmp9 + mu*(weak_hess_tmp10 + s_t(2)*weak_hess_tmp9);
    D[3] = lmbda*weak_hess_tmp12 + mu*weak_hess_tmp11;
    D[5] = lmbda*weak_hess_tmp2 + mu*(weak_hess_tmp14 + s_t(3)*weak_hess_tmp2) + weak_hess_tmp5;
    D[6] = lmbda*weak_hess_tmp11 + mu*weak_hess_tmp12;
    D[7] = lmbda*weak_hess_tmp10 + mu*(s_t(2)*weak_hess_tmp10 + weak_hess_tmp9);
    D[10] = lmbda*weak_hess_tmp3 + mu*(weak_hess_tmp14 + s_t(3)*weak_hess_tmp3) + weak_hess_tmp5;
    D[11] = lmbda*weak_hess_tmp8 + mu*(weak_hess_tmp6 + s_t(2)*weak_hess_tmp8);
    D[15] = lmbda*weak_hess_tmp13 + mu*(s_t(3)*weak_hess_tmp13 + weak_hess_tmp4) + weak_hess_tmp5;
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
    const s_t adj_q[ND * ND] = {adj_value0, adj_value1, adj_value2, adj_value3};
    point_test_symmetric_scalar<s_t, NS, ND, NC>(g_ref, adj_q, qw * idet, D, element_matrix);
  }
  mirror_upper_triangle_scalar<s_t, NDOFS>(element_matrix);
}

template <typename s_t, int NS>
static __host__ __device__ __forceinline__ void saint_venant_kirchhoff_d2_simplex_tri3_direct_hessian_element_matrix(
    const s_t *const RSTR badj0,
    const s_t *const RSTR badj1,
    const s_t *const RSTR badj2,
    const s_t *const RSTR badj3,
    const s_t *const RSTR bdet0,
    const s_t lmbda,
    const s_t mu,
    const s_t bu_data[NS * 2],
    s_t *const RSTR element_matrix
) {
  static_assert(NS > 0, "NS must be positive");
  const ptrdiff_t goff = 0;
  const s_t adj_value0 = badj0[goff];
  const s_t adj_value1 = badj1[goff];
  const s_t adj_value2 = badj2[goff];
  const s_t adj_value3 = badj3[goff];
  const s_t det_value0 = bdet0[goff];
  const s_t idet = s_t(1) / det_value0;
  const s_t gu_ref0 = -bu_data[0] + bu_data[2];
  const s_t gu_ref1 = -bu_data[0] + bu_data[4];
  const s_t gu_ref2 = -bu_data[1] + bu_data[3];
  const s_t gu_ref3 = -bu_data[1] + bu_data[5];
  const s_t gu0 = (gu_ref0 * adj_value0 + gu_ref1 * adj_value2) * idet;
  const s_t gu1 = (gu_ref0 * adj_value1 + gu_ref1 * adj_value3) * idet;
  const s_t gu2 = (gu_ref2 * adj_value0 + gu_ref3 * adj_value2) * idet;
  const s_t gu3 = (gu_ref2 * adj_value1 + gu_ref3 * adj_value3) * idet;
  const s_t hessian_tmp0 = gu0 + s_t(1);
  const s_t hessian_tmp1 = gu1*hessian_tmp0;
  const s_t hessian_tmp2 = gu3 + s_t(1);
  const s_t hessian_tmp3 = gu2*hessian_tmp2;
  const s_t hessian_tmp4 = hessian_tmp1*lmbda + mu*(s_t(2)*hessian_tmp1 + hessian_tmp3);
  const s_t hessian_tmp5 = idet*(-adj_value1 - adj_value3);
  const s_t hessian_tmp6 = pow_2(hessian_tmp0);
  const s_t hessian_tmp7 = pow_2(gu1);
  const s_t hessian_tmp8 = pow_2(gu2);
  const s_t hessian_tmp9 = hessian_tmp7 + hessian_tmp8 + s_t(-1);
  const s_t hessian_tmp10 = lmbda*(((s_t(1) / s_t(2)))*pow_2(gu0) + gu0 + ((s_t(1) / s_t(2)))*pow_2(gu3) + gu3 + ((s_t(1) / s_t(2)))*hessian_tmp7 + ((s_t(1) / s_t(2)))*hessian_tmp8);
  const s_t hessian_tmp11 = hessian_tmp10 + hessian_tmp6*lmbda + mu*(s_t(3)*hessian_tmp6 + hessian_tmp9);
  const s_t hessian_tmp12 = idet*(-adj_value0 - adj_value2);
  const s_t hessian_tmp13 = ((s_t(1) / s_t(2)))*hessian_tmp11*hessian_tmp12 + ((s_t(1) / s_t(2)))*hessian_tmp4*hessian_tmp5;
  const s_t hessian_tmp14 = pow_2(hessian_tmp2);
  const s_t hessian_tmp15 = hessian_tmp14 + hessian_tmp6 + s_t(-1);
  const s_t hessian_tmp16 = hessian_tmp10 + hessian_tmp7*lmbda + mu*(hessian_tmp15 + s_t(3)*hessian_tmp7);
  const s_t hessian_tmp17 = ((s_t(1) / s_t(2)))*hessian_tmp12*hessian_tmp4 + ((s_t(1) / s_t(2)))*hessian_tmp16*hessian_tmp5;
  const s_t hessian_tmp18 = adj_value0*hessian_tmp13 + adj_value1*hessian_tmp17;
  const s_t hessian_tmp19 = adj_value2*hessian_tmp13 + adj_value3*hessian_tmp17;
  const s_t hessian_tmp20 = gu1*gu2;
  const s_t hessian_tmp21 = hessian_tmp0*hessian_tmp2;
  const s_t hessian_tmp22 = hessian_tmp20*lmbda + hessian_tmp21*mu;
  const s_t hessian_tmp23 = gu2*hessian_tmp0;
  const s_t hessian_tmp24 = gu1*hessian_tmp2;
  const s_t hessian_tmp25 = hessian_tmp23*lmbda + mu*(s_t(2)*hessian_tmp23 + hessian_tmp24);
  const s_t hessian_tmp26 = hessian_tmp12*hessian_tmp25;
  const s_t hessian_tmp27 = ((s_t(1) / s_t(2)))*hessian_tmp22*hessian_tmp5 + ((s_t(1) / s_t(2)))*hessian_tmp26;
  const s_t hessian_tmp28 = hessian_tmp20*mu + hessian_tmp21*lmbda;
  const s_t hessian_tmp29 = hessian_tmp24*lmbda + mu*(hessian_tmp23 + s_t(2)*hessian_tmp24);
  const s_t hessian_tmp30 = hessian_tmp29*hessian_tmp5;
  const s_t hessian_tmp31 = ((s_t(1) / s_t(2)))*hessian_tmp12*hessian_tmp28 + ((s_t(1) / s_t(2)))*hessian_tmp30;
  const s_t hessian_tmp32 = adj_value0*hessian_tmp27 + adj_value1*hessian_tmp31;
  const s_t hessian_tmp33 = adj_value2*hessian_tmp27 + adj_value3*hessian_tmp31;
  const s_t hessian_tmp34 = adj_value1*idet;
  const s_t hessian_tmp35 = adj_value0*idet;
  const s_t hessian_tmp36 = ((s_t(1) / s_t(2)))*hessian_tmp11*hessian_tmp35 + ((s_t(1) / s_t(2)))*hessian_tmp34*hessian_tmp4;
  const s_t hessian_tmp37 = ((s_t(1) / s_t(2)))*hessian_tmp16*hessian_tmp34 + ((s_t(1) / s_t(2)))*hessian_tmp35*hessian_tmp4;
  const s_t hessian_tmp38 = adj_value0*hessian_tmp36 + adj_value1*hessian_tmp37;
  const s_t hessian_tmp39 = adj_value2*hessian_tmp36 + adj_value3*hessian_tmp37;
  const s_t hessian_tmp40 = hessian_tmp25*hessian_tmp35;
  const s_t hessian_tmp41 = ((s_t(1) / s_t(2)))*hessian_tmp22*hessian_tmp34 + ((s_t(1) / s_t(2)))*hessian_tmp40;
  const s_t hessian_tmp42 = hessian_tmp29*hessian_tmp34;
  const s_t hessian_tmp43 = ((s_t(1) / s_t(2)))*hessian_tmp28*hessian_tmp35 + ((s_t(1) / s_t(2)))*hessian_tmp42;
  const s_t hessian_tmp44 = adj_value0*hessian_tmp41 + adj_value1*hessian_tmp43;
  const s_t hessian_tmp45 = adj_value2*hessian_tmp41 + adj_value3*hessian_tmp43;
  const s_t hessian_tmp46 = adj_value3*idet;
  const s_t hessian_tmp47 = adj_value2*idet;
  const s_t hessian_tmp48 = ((s_t(1) / s_t(2)))*hessian_tmp11*hessian_tmp47 + ((s_t(1) / s_t(2)))*hessian_tmp4*hessian_tmp46;
  const s_t hessian_tmp49 = ((s_t(1) / s_t(2)))*hessian_tmp16*hessian_tmp46 + ((s_t(1) / s_t(2)))*hessian_tmp4*hessian_tmp47;
  const s_t hessian_tmp50 = adj_value0*hessian_tmp48 + adj_value1*hessian_tmp49;
  const s_t hessian_tmp51 = adj_value2*hessian_tmp48 + adj_value3*hessian_tmp49;
  const s_t hessian_tmp52 = hessian_tmp25*hessian_tmp47;
  const s_t hessian_tmp53 = ((s_t(1) / s_t(2)))*hessian_tmp22*hessian_tmp46 + ((s_t(1) / s_t(2)))*hessian_tmp52;
  const s_t hessian_tmp54 = hessian_tmp29*hessian_tmp46;
  const s_t hessian_tmp55 = ((s_t(1) / s_t(2)))*hessian_tmp28*hessian_tmp47 + ((s_t(1) / s_t(2)))*hessian_tmp54;
  const s_t hessian_tmp56 = adj_value0*hessian_tmp53 + adj_value1*hessian_tmp55;
  const s_t hessian_tmp57 = adj_value2*hessian_tmp53 + adj_value3*hessian_tmp55;
  const s_t hessian_tmp58 = ((s_t(1) / s_t(2)))*hessian_tmp26 + ((s_t(1) / s_t(2)))*hessian_tmp28*hessian_tmp5;
  const s_t hessian_tmp59 = ((s_t(1) / s_t(2)))*hessian_tmp12*hessian_tmp22 + ((s_t(1) / s_t(2)))*hessian_tmp30;
  const s_t hessian_tmp60 = adj_value0*hessian_tmp58 + adj_value1*hessian_tmp59;
  const s_t hessian_tmp61 = adj_value2*hessian_tmp58 + adj_value3*hessian_tmp59;
  const s_t hessian_tmp62 = hessian_tmp3*lmbda + mu*(hessian_tmp1 + s_t(2)*hessian_tmp3);
  const s_t hessian_tmp63 = hessian_tmp10 + hessian_tmp8*lmbda + mu*(hessian_tmp15 + s_t(3)*hessian_tmp8);
  const s_t hessian_tmp64 = ((s_t(1) / s_t(2)))*hessian_tmp12*hessian_tmp63 + ((s_t(1) / s_t(2)))*hessian_tmp5*hessian_tmp62;
  const s_t hessian_tmp65 = hessian_tmp10 + hessian_tmp14*lmbda + mu*(s_t(3)*hessian_tmp14 + hessian_tmp9);
  const s_t hessian_tmp66 = ((s_t(1) / s_t(2)))*hessian_tmp12*hessian_tmp62 + ((s_t(1) / s_t(2)))*hessian_tmp5*hessian_tmp65;
  const s_t hessian_tmp67 = adj_value0*hessian_tmp64 + adj_value1*hessian_tmp66;
  const s_t hessian_tmp68 = adj_value2*hessian_tmp64 + adj_value3*hessian_tmp66;
  const s_t hessian_tmp69 = ((s_t(1) / s_t(2)))*hessian_tmp28*hessian_tmp34 + ((s_t(1) / s_t(2)))*hessian_tmp40;
  const s_t hessian_tmp70 = ((s_t(1) / s_t(2)))*hessian_tmp22*hessian_tmp35 + ((s_t(1) / s_t(2)))*hessian_tmp42;
  const s_t hessian_tmp71 = adj_value0*hessian_tmp69 + adj_value1*hessian_tmp70;
  const s_t hessian_tmp72 = adj_value2*hessian_tmp69 + adj_value3*hessian_tmp70;
  const s_t hessian_tmp73 = ((s_t(1) / s_t(2)))*hessian_tmp34*hessian_tmp62 + ((s_t(1) / s_t(2)))*hessian_tmp35*hessian_tmp63;
  const s_t hessian_tmp74 = ((s_t(1) / s_t(2)))*hessian_tmp34*hessian_tmp65 + ((s_t(1) / s_t(2)))*hessian_tmp35*hessian_tmp62;
  const s_t hessian_tmp75 = adj_value0*hessian_tmp73 + adj_value1*hessian_tmp74;
  const s_t hessian_tmp76 = adj_value2*hessian_tmp73 + adj_value3*hessian_tmp74;
  const s_t hessian_tmp77 = ((s_t(1) / s_t(2)))*hessian_tmp28*hessian_tmp46 + ((s_t(1) / s_t(2)))*hessian_tmp52;
  const s_t hessian_tmp78 = ((s_t(1) / s_t(2)))*hessian_tmp22*hessian_tmp47 + ((s_t(1) / s_t(2)))*hessian_tmp54;
  const s_t hessian_tmp79 = adj_value0*hessian_tmp77 + adj_value1*hessian_tmp78;
  const s_t hessian_tmp80 = adj_value2*hessian_tmp77 + adj_value3*hessian_tmp78;
  const s_t hessian_tmp81 = ((s_t(1) / s_t(2)))*hessian_tmp46*hessian_tmp62 + ((s_t(1) / s_t(2)))*hessian_tmp47*hessian_tmp63;
  const s_t hessian_tmp82 = ((s_t(1) / s_t(2)))*hessian_tmp46*hessian_tmp65 + ((s_t(1) / s_t(2)))*hessian_tmp47*hessian_tmp62;
  const s_t hessian_tmp83 = adj_value0*hessian_tmp81 + adj_value1*hessian_tmp82;
  const s_t hessian_tmp84 = adj_value2*hessian_tmp81 + adj_value3*hessian_tmp82;
  element_matrix[0] = -hessian_tmp18 - hessian_tmp19;
  element_matrix[6] = hessian_tmp18;
  element_matrix[12] = hessian_tmp19;
  element_matrix[18] = -hessian_tmp32 - hessian_tmp33;
  element_matrix[24] = hessian_tmp32;
  element_matrix[30] = hessian_tmp33;
  element_matrix[1] = -hessian_tmp38 - hessian_tmp39;
  element_matrix[7] = hessian_tmp38;
  element_matrix[13] = hessian_tmp39;
  element_matrix[19] = -hessian_tmp44 - hessian_tmp45;
  element_matrix[25] = hessian_tmp44;
  element_matrix[31] = hessian_tmp45;
  element_matrix[2] = -hessian_tmp50 - hessian_tmp51;
  element_matrix[8] = hessian_tmp50;
  element_matrix[14] = hessian_tmp51;
  element_matrix[20] = -hessian_tmp56 - hessian_tmp57;
  element_matrix[26] = hessian_tmp56;
  element_matrix[32] = hessian_tmp57;
  element_matrix[3] = -hessian_tmp60 - hessian_tmp61;
  element_matrix[9] = hessian_tmp60;
  element_matrix[15] = hessian_tmp61;
  element_matrix[21] = -hessian_tmp67 - hessian_tmp68;
  element_matrix[27] = hessian_tmp67;
  element_matrix[33] = hessian_tmp68;
  element_matrix[4] = -hessian_tmp71 - hessian_tmp72;
  element_matrix[10] = hessian_tmp71;
  element_matrix[16] = hessian_tmp72;
  element_matrix[22] = -hessian_tmp75 - hessian_tmp76;
  element_matrix[28] = hessian_tmp75;
  element_matrix[34] = hessian_tmp76;
  element_matrix[5] = -hessian_tmp79 - hessian_tmp80;
  element_matrix[11] = hessian_tmp79;
  element_matrix[17] = hessian_tmp80;
  element_matrix[23] = -hessian_tmp83 - hessian_tmp84;
  element_matrix[29] = hessian_tmp83;
  element_matrix[35] = hessian_tmp84;
}

} // namespace codegen
} // namespace sfem

#endif
