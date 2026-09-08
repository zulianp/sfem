#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_ELASTIC_D3_TENSOR_PRODUCT_LOCAL_HPP
#define MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_ELASTIC_D3_TENSOR_PRODUCT_LOCAL_HPP
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
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_objective_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR adj4,
        const s_t *const RSTR adj5,
        const s_t *const RSTR adj6,
        const s_t *const RSTR adj7,
        const s_t *const RSTR adj8,
        const s_t *const RSTR det0,
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t lmbda,
        const s_t mu,
        const s_t *const RSTR u_streams[NS * 3],
        s_t *const RSTR value
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NQ1 = integer_root(NQ, 3);
  static constexpr int NS1 = integer_root(NS, 3);
  static_assert(ipow(NQ1, 3) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 3) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[NQ * 9 * VS];
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0 * NQ * 3 * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 1, &gu_ref_q[1 * NQ * 3 * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 2, &gu_ref_q[2 * NQ * 3 * VS]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
      const s_t adj_lane0 = adj0[goff];
      const s_t adj_lane1 = adj1[goff];
      const s_t adj_lane2 = adj2[goff];
      const s_t adj_lane3 = adj3[goff];
      const s_t adj_lane4 = adj4[goff];
      const s_t adj_lane5 = adj5[goff];
      const s_t adj_lane6 = adj6[goff];
      const s_t adj_lane7 = adj7[goff];
      const s_t adj_lane8 = adj8[goff];
      const s_t det_lane0 = det0[goff];
      s_t gu_ref[9];
      gu_ref[0] = gu_ref_q[((0 * NQ + q) * 3 + 0) * VS + lane];
      gu_ref[1] = gu_ref_q[((0 * NQ + q) * 3 + 1) * VS + lane];
      gu_ref[2] = gu_ref_q[((0 * NQ + q) * 3 + 2) * VS + lane];
      gu_ref[3] = gu_ref_q[((1 * NQ + q) * 3 + 0) * VS + lane];
      gu_ref[4] = gu_ref_q[((1 * NQ + q) * 3 + 1) * VS + lane];
      gu_ref[5] = gu_ref_q[((1 * NQ + q) * 3 + 2) * VS + lane];
      gu_ref[6] = gu_ref_q[((2 * NQ + q) * 3 + 0) * VS + lane];
      gu_ref[7] = gu_ref_q[((2 * NQ + q) * 3 + 1) * VS + lane];
      gu_ref[8] = gu_ref_q[((2 * NQ + q) * 3 + 2) * VS + lane];
      s_t gu[9];
      const s_t idet = s_t(1) / det_lane0;
      gu[0] = (gu_ref[0] * adj_lane0 + gu_ref[1] * adj_lane3 + gu_ref[2] * adj_lane6) * idet;
      gu[1] = (gu_ref[0] * adj_lane1 + gu_ref[1] * adj_lane4 + gu_ref[2] * adj_lane7) * idet;
      gu[2] = (gu_ref[0] * adj_lane2 + gu_ref[1] * adj_lane5 + gu_ref[2] * adj_lane8) * idet;
      gu[3] = (gu_ref[3] * adj_lane0 + gu_ref[4] * adj_lane3 + gu_ref[5] * adj_lane6) * idet;
      gu[4] = (gu_ref[3] * adj_lane1 + gu_ref[4] * adj_lane4 + gu_ref[5] * adj_lane7) * idet;
      gu[5] = (gu_ref[3] * adj_lane2 + gu_ref[4] * adj_lane5 + gu_ref[5] * adj_lane8) * idet;
      gu[6] = (gu_ref[6] * adj_lane0 + gu_ref[7] * adj_lane3 + gu_ref[8] * adj_lane6) * idet;
      gu[7] = (gu_ref[6] * adj_lane1 + gu_ref[7] * adj_lane4 + gu_ref[8] * adj_lane7) * idet;
      gu[8] = (gu_ref[6] * adj_lane2 + gu_ref[7] * adj_lane5 + gu_ref[8] * adj_lane8) * idet;
    const s_t weak_obj_tmp0 = gu[8] + s_t(1);
    const s_t weak_obj_tmp1 = gu[1]*gu[3]*weak_obj_tmp0;
    const s_t weak_obj_tmp2 = gu[4] + s_t(1);
    const s_t weak_obj_tmp3 = gu[2]*gu[6]*weak_obj_tmp2;
    const s_t weak_obj_tmp4 = gu[0] + s_t(1);
    const s_t weak_obj_tmp5 = gu[5]*gu[7]*weak_obj_tmp4;
    const s_t weak_obj_tmp6 = pow_2(gu[1]) + pow_2(gu[7]) + pow_2(weak_obj_tmp2);
    const s_t weak_obj_tmp7 = pow_2(gu[2]) + pow_2(gu[5]) + pow_2(weak_obj_tmp0);
    const s_t weak_obj_tmp8 = pow_2(gu[3]) + pow_2(gu[6]) + pow_2(weak_obj_tmp4);
    const s_t weak_obj_tmp9 = weak_obj_tmp6 + weak_obj_tmp7 + weak_obj_tmp8;
    value[lane] += qw * det_lane0 * (((s_t(1) / s_t(2)))*lmbda*pow_2(gu[1]*gu[5]*gu[6] + gu[2]*gu[3]*gu[7] + weak_obj_tmp0*weak_obj_tmp2*weak_obj_tmp4 - weak_obj_tmp1 - weak_obj_tmp3 - weak_obj_tmp5 + s_t(-1)) + mu*(-s_t(6)*gu[1]*gu[5]*gu[6] - s_t(6)*gu[2]*gu[3]*gu[7] - s_t(6)*weak_obj_tmp0*weak_obj_tmp2*weak_obj_tmp4 + s_t(6)*weak_obj_tmp1 + s_t(6)*weak_obj_tmp3 + s_t(6)*weak_obj_tmp5 - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp6) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp7) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp8) + ((s_t(1) / s_t(2)))*pow_2(weak_obj_tmp9) + weak_obj_tmp9 - pow_2(gu[1]*gu[2] + gu[5]*weak_obj_tmp2 + gu[7]*weak_obj_tmp0) - pow_2(gu[1]*weak_obj_tmp4 + gu[3]*weak_obj_tmp2 + gu[6]*gu[7]) - pow_2(gu[2]*weak_obj_tmp4 + gu[3]*gu[5] + gu[6]*weak_obj_tmp0)));
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_gradient_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR adj4,
        const s_t *const RSTR adj5,
        const s_t *const RSTR adj6,
        const s_t *const RSTR adj7,
        const s_t *const RSTR adj8,
        const s_t *const RSTR det0,
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t lmbda,
        const s_t mu,
        const s_t *const RSTR u_streams[NS * 3],
        s_t *const RSTR out_streams[NS * 3]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NQ1 = integer_root(NQ, 3);
  static constexpr int NS1 = integer_root(NS, 3);
  static_assert(ipow(NQ1, 3) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 3) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[NQ * 9 * VS];
  s_t loperand_q[NQ * 9 * VS];
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0 * NQ * 3 * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 1, &gu_ref_q[1 * NQ * 3 * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 2, &gu_ref_q[2 * NQ * 3 * VS]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
      const s_t adj_lane0 = adj0[goff];
      const s_t adj_lane1 = adj1[goff];
      const s_t adj_lane2 = adj2[goff];
      const s_t adj_lane3 = adj3[goff];
      const s_t adj_lane4 = adj4[goff];
      const s_t adj_lane5 = adj5[goff];
      const s_t adj_lane6 = adj6[goff];
      const s_t adj_lane7 = adj7[goff];
      const s_t adj_lane8 = adj8[goff];
      const s_t det_lane0 = det0[goff];
      s_t gu_ref[9];
      gu_ref[0] = gu_ref_q[((0 * NQ + q) * 3 + 0) * VS + lane];
      gu_ref[1] = gu_ref_q[((0 * NQ + q) * 3 + 1) * VS + lane];
      gu_ref[2] = gu_ref_q[((0 * NQ + q) * 3 + 2) * VS + lane];
      gu_ref[3] = gu_ref_q[((1 * NQ + q) * 3 + 0) * VS + lane];
      gu_ref[4] = gu_ref_q[((1 * NQ + q) * 3 + 1) * VS + lane];
      gu_ref[5] = gu_ref_q[((1 * NQ + q) * 3 + 2) * VS + lane];
      gu_ref[6] = gu_ref_q[((2 * NQ + q) * 3 + 0) * VS + lane];
      gu_ref[7] = gu_ref_q[((2 * NQ + q) * 3 + 1) * VS + lane];
      gu_ref[8] = gu_ref_q[((2 * NQ + q) * 3 + 2) * VS + lane];
      s_t gu[9];
      const s_t idet = s_t(1) / det_lane0;
      gu[0] = (gu_ref[0] * adj_lane0 + gu_ref[1] * adj_lane3 + gu_ref[2] * adj_lane6) * idet;
      gu[1] = (gu_ref[0] * adj_lane1 + gu_ref[1] * adj_lane4 + gu_ref[2] * adj_lane7) * idet;
      gu[2] = (gu_ref[0] * adj_lane2 + gu_ref[1] * adj_lane5 + gu_ref[2] * adj_lane8) * idet;
      gu[3] = (gu_ref[3] * adj_lane0 + gu_ref[4] * adj_lane3 + gu_ref[5] * adj_lane6) * idet;
      gu[4] = (gu_ref[3] * adj_lane1 + gu_ref[4] * adj_lane4 + gu_ref[5] * adj_lane7) * idet;
      gu[5] = (gu_ref[3] * adj_lane2 + gu_ref[4] * adj_lane5 + gu_ref[5] * adj_lane8) * idet;
      gu[6] = (gu_ref[6] * adj_lane0 + gu_ref[7] * adj_lane3 + gu_ref[8] * adj_lane6) * idet;
      gu[7] = (gu_ref[6] * adj_lane1 + gu_ref[7] * adj_lane4 + gu_ref[8] * adj_lane7) * idet;
      gu[8] = (gu_ref[6] * adj_lane2 + gu_ref[7] * adj_lane5 + gu_ref[8] * adj_lane8) * idet;
      s_t loperand[9];
    s_t material[9];
    const s_t weak_mat_tmp0 = s_t(2)*gu[5];
    const s_t weak_mat_tmp1 = gu[4] + s_t(1);
    const s_t weak_mat_tmp2 = gu[8] + s_t(1);
    const s_t weak_mat_tmp3 = gu[0] + s_t(1);
    const s_t weak_mat_tmp4 = gu[5]*gu[7];
    const s_t weak_mat_tmp5 = ((s_t(1) / s_t(2)))*lmbda*(-gu[1]*gu[3]*weak_mat_tmp2 + gu[1]*gu[5]*gu[6] + gu[2]*gu[3]*gu[7] - gu[2]*gu[6]*weak_mat_tmp1 + weak_mat_tmp1*weak_mat_tmp2*weak_mat_tmp3 - weak_mat_tmp3*weak_mat_tmp4 + s_t(-1));
    const s_t weak_mat_tmp6 = gu[1]*weak_mat_tmp3 + gu[3]*weak_mat_tmp1 + gu[6]*gu[7];
    const s_t weak_mat_tmp7 = s_t(2)*gu[1];
    const s_t weak_mat_tmp8 = gu[2]*weak_mat_tmp3 + gu[3]*gu[5] + gu[6]*weak_mat_tmp2;
    const s_t weak_mat_tmp9 = s_t(2)*gu[2];
    const s_t weak_mat_tmp10 = pow_2(gu[3]) + pow_2(gu[6]) + pow_2(weak_mat_tmp3);
    const s_t weak_mat_tmp11 = s_t(2)*weak_mat_tmp3;
    const s_t weak_mat_tmp12 = pow_2(gu[1]) + pow_2(gu[7]) + pow_2(weak_mat_tmp1);
    const s_t weak_mat_tmp13 = pow_2(gu[2]) + pow_2(gu[5]) + pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp14 = weak_mat_tmp10 + weak_mat_tmp12 + weak_mat_tmp13;
    const s_t weak_mat_tmp15 = s_t(2)*gu[3];
    const s_t weak_mat_tmp16 = gu[1]*gu[2] + gu[5]*weak_mat_tmp1 + gu[7]*weak_mat_tmp2;
    const s_t weak_mat_tmp17 = s_t(2)*gu[6];
    const s_t weak_mat_tmp18 = s_t(6)*gu[2];
    const s_t weak_mat_tmp19 = s_t(2)*weak_mat_tmp1;
    const s_t weak_mat_tmp20 = s_t(2)*gu[7];
    const s_t weak_mat_tmp21 = s_t(6)*gu[1];
    const s_t weak_mat_tmp22 = s_t(2)*weak_mat_tmp2;
    material[0] = mu*(s_t(2)*gu[0] - s_t(6)*weak_mat_tmp1*weak_mat_tmp2 - weak_mat_tmp10*weak_mat_tmp11 + weak_mat_tmp11*weak_mat_tmp14 + s_t(6)*weak_mat_tmp4 - weak_mat_tmp6*weak_mat_tmp7 - weak_mat_tmp8*weak_mat_tmp9 + s_t(2)) + weak_mat_tmp5*(-gu[7]*weak_mat_tmp0 + s_t(2)*weak_mat_tmp1*weak_mat_tmp2);
    material[1] = mu*(s_t(2)*gu[1]*weak_mat_tmp14 + s_t(2)*gu[1] + s_t(6)*gu[3]*weak_mat_tmp2 - s_t(6)*gu[5]*gu[6] - weak_mat_tmp11*weak_mat_tmp6 - weak_mat_tmp12*weak_mat_tmp7 - weak_mat_tmp16*weak_mat_tmp9) + weak_mat_tmp5*(s_t(2)*gu[5]*gu[6] - weak_mat_tmp15*weak_mat_tmp2);
    material[2] = mu*(s_t(2)*gu[2]*weak_mat_tmp14 + s_t(2)*gu[2] - s_t(6)*gu[3]*gu[7] + s_t(6)*gu[6]*weak_mat_tmp1 - weak_mat_tmp11*weak_mat_tmp8 - weak_mat_tmp13*weak_mat_tmp9 - weak_mat_tmp16*weak_mat_tmp7) + weak_mat_tmp5*(gu[7]*weak_mat_tmp15 - weak_mat_tmp1*weak_mat_tmp17);
    material[3] = mu*(s_t(6)*gu[1]*weak_mat_tmp2 + s_t(2)*gu[3]*weak_mat_tmp14 + s_t(2)*gu[3] - gu[7]*weak_mat_tmp18 - weak_mat_tmp0*weak_mat_tmp8 - weak_mat_tmp10*weak_mat_tmp15 - weak_mat_tmp19*weak_mat_tmp6) + weak_mat_tmp5*(s_t(2)*gu[2]*gu[7] - weak_mat_tmp2*weak_mat_tmp7);
    material[4] = mu*(s_t(2)*gu[4] + gu[6]*weak_mat_tmp18 - weak_mat_tmp0*weak_mat_tmp16 - weak_mat_tmp12*weak_mat_tmp19 + weak_mat_tmp14*weak_mat_tmp19 - weak_mat_tmp15*weak_mat_tmp6 - s_t(6)*weak_mat_tmp2*weak_mat_tmp3 + s_t(2)) + weak_mat_tmp5*(-gu[6]*weak_mat_tmp9 + s_t(2)*weak_mat_tmp2*weak_mat_tmp3);
    material[5] = mu*(s_t(2)*gu[5]*weak_mat_tmp14 + s_t(2)*gu[5] - gu[6]*weak_mat_tmp21 + s_t(6)*gu[7]*weak_mat_tmp3 - weak_mat_tmp0*weak_mat_tmp13 - weak_mat_tmp15*weak_mat_tmp8 - weak_mat_tmp16*weak_mat_tmp19) + weak_mat_tmp5*(gu[6]*weak_mat_tmp7 - weak_mat_tmp20*weak_mat_tmp3);
    material[6] = mu*(s_t(6)*gu[2]*weak_mat_tmp1 - gu[5]*weak_mat_tmp21 + s_t(2)*gu[6]*weak_mat_tmp14 + s_t(2)*gu[6] - weak_mat_tmp10*weak_mat_tmp17 - weak_mat_tmp20*weak_mat_tmp6 - weak_mat_tmp22*weak_mat_tmp8) + weak_mat_tmp5*(gu[5]*weak_mat_tmp7 - weak_mat_tmp1*weak_mat_tmp9);
    material[7] = mu*(-gu[3]*weak_mat_tmp18 + s_t(6)*gu[5]*weak_mat_tmp3 + s_t(2)*gu[7]*weak_mat_tmp14 + s_t(2)*gu[7] - weak_mat_tmp12*weak_mat_tmp20 - weak_mat_tmp16*weak_mat_tmp22 - weak_mat_tmp17*weak_mat_tmp6) + weak_mat_tmp5*(gu[3]*weak_mat_tmp9 - weak_mat_tmp0*weak_mat_tmp3);
    material[8] = mu*(gu[3]*weak_mat_tmp21 + s_t(2)*gu[8] - s_t(6)*weak_mat_tmp1*weak_mat_tmp3 - weak_mat_tmp13*weak_mat_tmp22 + weak_mat_tmp14*weak_mat_tmp22 - weak_mat_tmp16*weak_mat_tmp20 - weak_mat_tmp17*weak_mat_tmp8 + s_t(2)) + weak_mat_tmp5*(-gu[3]*weak_mat_tmp7 + s_t(2)*weak_mat_tmp1*weak_mat_tmp3);
    loperand[0] = qw * (material[0] * adj_lane0 + material[1] * adj_lane1 + material[2] * adj_lane2);
    loperand[1] = qw * (material[0] * adj_lane3 + material[1] * adj_lane4 + material[2] * adj_lane5);
    loperand[2] = qw * (material[0] * adj_lane6 + material[1] * adj_lane7 + material[2] * adj_lane8);
    loperand[3] = qw * (material[3] * adj_lane0 + material[4] * adj_lane1 + material[5] * adj_lane2);
    loperand[4] = qw * (material[3] * adj_lane3 + material[4] * adj_lane4 + material[5] * adj_lane5);
    loperand[5] = qw * (material[3] * adj_lane6 + material[4] * adj_lane7 + material[5] * adj_lane8);
    loperand[6] = qw * (material[6] * adj_lane0 + material[7] * adj_lane1 + material[8] * adj_lane2);
    loperand[7] = qw * (material[6] * adj_lane3 + material[7] * adj_lane4 + material[8] * adj_lane5);
    loperand[8] = qw * (material[6] * adj_lane6 + material[7] * adj_lane7 + material[8] * adj_lane8);
      loperand_q[((0 * NQ + q) * 3 + 0) * VS + lane] = loperand[0];
      loperand_q[((0 * NQ + q) * 3 + 1) * VS + lane] = loperand[1];
      loperand_q[((0 * NQ + q) * 3 + 2) * VS + lane] = loperand[2];
      loperand_q[((1 * NQ + q) * 3 + 0) * VS + lane] = loperand[3];
      loperand_q[((1 * NQ + q) * 3 + 1) * VS + lane] = loperand[4];
      loperand_q[((1 * NQ + q) * 3 + 2) * VS + lane] = loperand[5];
      loperand_q[((2 * NQ + q) * 3 + 0) * VS + lane] = loperand[6];
      loperand_q[((2 * NQ + q) * 3 + 1) * VS + lane] = loperand[7];
      loperand_q[((2 * NQ + q) * 3 + 2) * VS + lane] = loperand[8];
    }
  }
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[0 * NQ * 3 * VS], out_streams, 0);
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[1 * NQ * 3 * VS], out_streams, 1);
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[2 * NQ * 3 * VS], out_streams, 2);
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_elastic_d3_tensor_product_apply_block(
        const int ne,
        const ptrdiff_t geometry_stride,
        const s_t *const RSTR adj0,
        const s_t *const RSTR adj1,
        const s_t *const RSTR adj2,
        const s_t *const RSTR adj3,
        const s_t *const RSTR adj4,
        const s_t *const RSTR adj5,
        const s_t *const RSTR adj6,
        const s_t *const RSTR adj7,
        const s_t *const RSTR adj8,
        const s_t *const RSTR det0,
        const s_t *const RSTR shape_1d,
        const s_t *const RSTR grad_1d,
        const s_t *const RSTR q_weight_1d,
        const s_t lmbda,
        const s_t mu,
        const s_t *const RSTR u_streams[NS * 3],
        const s_t *const RSTR h_streams[NS * 3],
        s_t *const RSTR out_streams[NS * 3]
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NQ1 = integer_root(NQ, 3);
  static constexpr int NS1 = integer_root(NS, 3);
  static_assert(ipow(NQ1, 3) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 3) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[NQ * 9 * VS];
  s_t grad_h_ref_q[NQ * 9 * VS];
  s_t loperand_q[NQ * 9 * VS];
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0 * NQ * 3 * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, h_streams, 0, &grad_h_ref_q[0 * NQ * 3 * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 1, &gu_ref_q[1 * NQ * 3 * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, h_streams, 1, &grad_h_ref_q[1 * NQ * 3 * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 2, &gu_ref_q[2 * NQ * 3 * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, h_streams, 2, &grad_h_ref_q[2 * NQ * 3 * VS]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const ptrdiff_t goff = q * geometry_stride + lane;
      const s_t adj_lane0 = adj0[goff];
      const s_t adj_lane1 = adj1[goff];
      const s_t adj_lane2 = adj2[goff];
      const s_t adj_lane3 = adj3[goff];
      const s_t adj_lane4 = adj4[goff];
      const s_t adj_lane5 = adj5[goff];
      const s_t adj_lane6 = adj6[goff];
      const s_t adj_lane7 = adj7[goff];
      const s_t adj_lane8 = adj8[goff];
      const s_t det_lane0 = det0[goff];
      s_t gu_ref[9];
      gu_ref[0] = gu_ref_q[((0 * NQ + q) * 3 + 0) * VS + lane];
      gu_ref[1] = gu_ref_q[((0 * NQ + q) * 3 + 1) * VS + lane];
      gu_ref[2] = gu_ref_q[((0 * NQ + q) * 3 + 2) * VS + lane];
      gu_ref[3] = gu_ref_q[((1 * NQ + q) * 3 + 0) * VS + lane];
      gu_ref[4] = gu_ref_q[((1 * NQ + q) * 3 + 1) * VS + lane];
      gu_ref[5] = gu_ref_q[((1 * NQ + q) * 3 + 2) * VS + lane];
      gu_ref[6] = gu_ref_q[((2 * NQ + q) * 3 + 0) * VS + lane];
      gu_ref[7] = gu_ref_q[((2 * NQ + q) * 3 + 1) * VS + lane];
      gu_ref[8] = gu_ref_q[((2 * NQ + q) * 3 + 2) * VS + lane];
      s_t grad_h_ref[9];
      grad_h_ref[0] = grad_h_ref_q[((0 * NQ + q) * 3 + 0) * VS + lane];
      grad_h_ref[1] = grad_h_ref_q[((0 * NQ + q) * 3 + 1) * VS + lane];
      grad_h_ref[2] = grad_h_ref_q[((0 * NQ + q) * 3 + 2) * VS + lane];
      grad_h_ref[3] = grad_h_ref_q[((1 * NQ + q) * 3 + 0) * VS + lane];
      grad_h_ref[4] = grad_h_ref_q[((1 * NQ + q) * 3 + 1) * VS + lane];
      grad_h_ref[5] = grad_h_ref_q[((1 * NQ + q) * 3 + 2) * VS + lane];
      grad_h_ref[6] = grad_h_ref_q[((2 * NQ + q) * 3 + 0) * VS + lane];
      grad_h_ref[7] = grad_h_ref_q[((2 * NQ + q) * 3 + 1) * VS + lane];
      grad_h_ref[8] = grad_h_ref_q[((2 * NQ + q) * 3 + 2) * VS + lane];
      s_t gu[9];
      s_t trial_grad[9];
      const s_t idet = s_t(1) / det_lane0;
      gu[0] = (gu_ref[0] * adj_lane0 + gu_ref[1] * adj_lane3 + gu_ref[2] * adj_lane6) * idet;
      trial_grad[0] = (grad_h_ref[0] * adj_lane0 + grad_h_ref[1] * adj_lane3 + grad_h_ref[2] * adj_lane6) * idet;
      gu[1] = (gu_ref[0] * adj_lane1 + gu_ref[1] * adj_lane4 + gu_ref[2] * adj_lane7) * idet;
      trial_grad[1] = (grad_h_ref[0] * adj_lane1 + grad_h_ref[1] * adj_lane4 + grad_h_ref[2] * adj_lane7) * idet;
      gu[2] = (gu_ref[0] * adj_lane2 + gu_ref[1] * adj_lane5 + gu_ref[2] * adj_lane8) * idet;
      trial_grad[2] = (grad_h_ref[0] * adj_lane2 + grad_h_ref[1] * adj_lane5 + grad_h_ref[2] * adj_lane8) * idet;
      gu[3] = (gu_ref[3] * adj_lane0 + gu_ref[4] * adj_lane3 + gu_ref[5] * adj_lane6) * idet;
      trial_grad[3] = (grad_h_ref[3] * adj_lane0 + grad_h_ref[4] * adj_lane3 + grad_h_ref[5] * adj_lane6) * idet;
      gu[4] = (gu_ref[3] * adj_lane1 + gu_ref[4] * adj_lane4 + gu_ref[5] * adj_lane7) * idet;
      trial_grad[4] = (grad_h_ref[3] * adj_lane1 + grad_h_ref[4] * adj_lane4 + grad_h_ref[5] * adj_lane7) * idet;
      gu[5] = (gu_ref[3] * adj_lane2 + gu_ref[4] * adj_lane5 + gu_ref[5] * adj_lane8) * idet;
      trial_grad[5] = (grad_h_ref[3] * adj_lane2 + grad_h_ref[4] * adj_lane5 + grad_h_ref[5] * adj_lane8) * idet;
      gu[6] = (gu_ref[6] * adj_lane0 + gu_ref[7] * adj_lane3 + gu_ref[8] * adj_lane6) * idet;
      trial_grad[6] = (grad_h_ref[6] * adj_lane0 + grad_h_ref[7] * adj_lane3 + grad_h_ref[8] * adj_lane6) * idet;
      gu[7] = (gu_ref[6] * adj_lane1 + gu_ref[7] * adj_lane4 + gu_ref[8] * adj_lane7) * idet;
      trial_grad[7] = (grad_h_ref[6] * adj_lane1 + grad_h_ref[7] * adj_lane4 + grad_h_ref[8] * adj_lane7) * idet;
      gu[8] = (gu_ref[6] * adj_lane2 + gu_ref[7] * adj_lane5 + gu_ref[8] * adj_lane8) * idet;
      trial_grad[8] = (grad_h_ref[6] * adj_lane2 + grad_h_ref[7] * adj_lane5 + grad_h_ref[8] * adj_lane8) * idet;
      s_t loperand[9];
    s_t material[9];
    const s_t weak_mat_tmp0 = s_t(2)*gu[6];
    const s_t weak_mat_tmp1 = gu[7]*weak_mat_tmp0;
    const s_t weak_mat_tmp2 = gu[4] + s_t(1);
    const s_t weak_mat_tmp3 = s_t(2)*gu[3];
    const s_t weak_mat_tmp4 = weak_mat_tmp2*weak_mat_tmp3;
    const s_t weak_mat_tmp5 = mu*(-weak_mat_tmp1 - weak_mat_tmp4);
    const s_t weak_mat_tmp6 = gu[8] + s_t(1);
    const s_t weak_mat_tmp7 = gu[3]*weak_mat_tmp6;
    const s_t weak_mat_tmp8 = gu[5]*gu[6] - weak_mat_tmp7;
    const s_t weak_mat_tmp9 = gu[5]*gu[7];
    const s_t weak_mat_tmp10 = s_t(2)*weak_mat_tmp9;
    const s_t weak_mat_tmp11 = -s_t(2)*weak_mat_tmp2*weak_mat_tmp6;
    const s_t weak_mat_tmp12 = ((s_t(1) / s_t(2)))*lmbda;
    const s_t weak_mat_tmp13 = weak_mat_tmp12*(-weak_mat_tmp10 - weak_mat_tmp11);
    const s_t weak_mat_tmp14 = gu[5]*weak_mat_tmp3;
    const s_t weak_mat_tmp15 = weak_mat_tmp0*weak_mat_tmp6;
    const s_t weak_mat_tmp16 = mu*(-weak_mat_tmp14 - weak_mat_tmp15);
    const s_t weak_mat_tmp17 = gu[3]*gu[7];
    const s_t weak_mat_tmp18 = gu[6]*weak_mat_tmp2;
    const s_t weak_mat_tmp19 = weak_mat_tmp17 - weak_mat_tmp18;
    const s_t weak_mat_tmp20 = s_t(2)*gu[2];
    const s_t weak_mat_tmp21 = gu[5]*weak_mat_tmp20;
    const s_t weak_mat_tmp22 = s_t(2)*gu[1];
    const s_t weak_mat_tmp23 = weak_mat_tmp2*weak_mat_tmp22;
    const s_t weak_mat_tmp24 = mu*(-weak_mat_tmp21 - weak_mat_tmp23);
    const s_t weak_mat_tmp25 = gu[1]*weak_mat_tmp6;
    const s_t weak_mat_tmp26 = gu[2]*gu[7] - weak_mat_tmp25;
    const s_t weak_mat_tmp27 = gu[7]*weak_mat_tmp22;
    const s_t weak_mat_tmp28 = weak_mat_tmp20*weak_mat_tmp6;
    const s_t weak_mat_tmp29 = mu*(-weak_mat_tmp27 - weak_mat_tmp28);
    const s_t weak_mat_tmp30 = gu[1]*gu[5];
    const s_t weak_mat_tmp31 = gu[2]*weak_mat_tmp2;
    const s_t weak_mat_tmp32 = weak_mat_tmp30 - weak_mat_tmp31;
    const s_t weak_mat_tmp33 = s_t(2)*pow_2(gu[5]);
    const s_t weak_mat_tmp34 = s_t(2)*pow_2(weak_mat_tmp6) + s_t(2);
    const s_t weak_mat_tmp35 = weak_mat_tmp33 + weak_mat_tmp34;
    const s_t weak_mat_tmp36 = s_t(2)*pow_2(gu[7]);
    const s_t weak_mat_tmp37 = s_t(2)*pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp38 = weak_mat_tmp36 + weak_mat_tmp37;
    const s_t weak_mat_tmp39 = weak_mat_tmp2*weak_mat_tmp6 - weak_mat_tmp9;
    const s_t weak_mat_tmp40 = gu[1]*gu[6];
    const s_t weak_mat_tmp41 = gu[0] + s_t(1);
    const s_t weak_mat_tmp42 = gu[7]*weak_mat_tmp41;
    const s_t weak_mat_tmp43 = weak_mat_tmp40 - weak_mat_tmp42;
    const s_t weak_mat_tmp44 = s_t(6)*gu[7];
    const s_t weak_mat_tmp45 = gu[2]*gu[3];
    const s_t weak_mat_tmp46 = s_t(2)*weak_mat_tmp45;
    const s_t weak_mat_tmp47 = gu[5]*weak_mat_tmp41;
    const s_t weak_mat_tmp48 = lmbda*(gu[1]*gu[5]*gu[6] - gu[1]*weak_mat_tmp7 + gu[2]*gu[3]*gu[7] - gu[2]*weak_mat_tmp18 + weak_mat_tmp2*weak_mat_tmp41*weak_mat_tmp6 - weak_mat_tmp41*weak_mat_tmp9 + s_t(-1));
    const s_t weak_mat_tmp49 = gu[7]*weak_mat_tmp48;
    const s_t weak_mat_tmp50 = mu*(weak_mat_tmp44 - weak_mat_tmp46 + s_t(4)*weak_mat_tmp47) - weak_mat_tmp49;
    const s_t weak_mat_tmp51 = weak_mat_tmp45 - weak_mat_tmp47;
    const s_t weak_mat_tmp52 = s_t(6)*gu[5];
    const s_t weak_mat_tmp53 = s_t(2)*weak_mat_tmp40;
    const s_t weak_mat_tmp54 = gu[5]*weak_mat_tmp48;
    const s_t weak_mat_tmp55 = mu*(s_t(4)*weak_mat_tmp42 + weak_mat_tmp52 - weak_mat_tmp53) - weak_mat_tmp54;
    const s_t weak_mat_tmp56 = gu[2]*gu[6];
    const s_t weak_mat_tmp57 = weak_mat_tmp41*weak_mat_tmp6 - weak_mat_tmp56;
    const s_t weak_mat_tmp58 = gu[1]*gu[3];
    const s_t weak_mat_tmp59 = s_t(2)*weak_mat_tmp58;
    const s_t weak_mat_tmp60 = s_t(6)*gu[8] + s_t(6);
    const s_t weak_mat_tmp61 = weak_mat_tmp48*weak_mat_tmp6;
    const s_t weak_mat_tmp62 = mu*(s_t(4)*weak_mat_tmp2*weak_mat_tmp41 - weak_mat_tmp59 - weak_mat_tmp60) + weak_mat_tmp61;
    const s_t weak_mat_tmp63 = weak_mat_tmp2*weak_mat_tmp41 - weak_mat_tmp58;
    const s_t weak_mat_tmp64 = s_t(2)*weak_mat_tmp56;
    const s_t weak_mat_tmp65 = s_t(6)*gu[4] + s_t(6);
    const s_t weak_mat_tmp66 = weak_mat_tmp2*weak_mat_tmp48;
    const s_t weak_mat_tmp67 = mu*(s_t(4)*weak_mat_tmp41*weak_mat_tmp6 - weak_mat_tmp64 - weak_mat_tmp65) + weak_mat_tmp66;
    const s_t weak_mat_tmp68 = -s_t(2)*gu[5]*gu[6];
    const s_t weak_mat_tmp69 = s_t(2)*weak_mat_tmp7;
    const s_t weak_mat_tmp70 = weak_mat_tmp12*(-weak_mat_tmp68 - weak_mat_tmp69);
    const s_t weak_mat_tmp71 = s_t(2)*gu[5];
    const s_t weak_mat_tmp72 = weak_mat_tmp2*weak_mat_tmp71;
    const s_t weak_mat_tmp73 = s_t(2)*gu[7];
    const s_t weak_mat_tmp74 = weak_mat_tmp6*weak_mat_tmp73;
    const s_t weak_mat_tmp75 = mu*(-weak_mat_tmp72 - weak_mat_tmp74);
    const s_t weak_mat_tmp76 = weak_mat_tmp3*weak_mat_tmp41;
    const s_t weak_mat_tmp77 = mu*(-weak_mat_tmp21 - weak_mat_tmp76);
    const s_t weak_mat_tmp78 = weak_mat_tmp0*weak_mat_tmp41;
    const s_t weak_mat_tmp79 = mu*(-weak_mat_tmp28 - weak_mat_tmp78);
    const s_t weak_mat_tmp80 = s_t(2)*pow_2(gu[3]);
    const s_t weak_mat_tmp81 = s_t(2)*pow_2(gu[6]);
    const s_t weak_mat_tmp82 = weak_mat_tmp80 + weak_mat_tmp81;
    const s_t weak_mat_tmp83 = s_t(6)*gu[6];
    const s_t weak_mat_tmp84 = s_t(2)*weak_mat_tmp31;
    const s_t weak_mat_tmp85 = gu[6]*weak_mat_tmp48;
    const s_t weak_mat_tmp86 = mu*(s_t(4)*gu[1]*gu[5] - weak_mat_tmp83 - weak_mat_tmp84) + weak_mat_tmp85;
    const s_t weak_mat_tmp87 = s_t(2)*weak_mat_tmp42;
    const s_t weak_mat_tmp88 = mu*(s_t(4)*gu[1]*gu[6] - weak_mat_tmp52 - weak_mat_tmp87) + weak_mat_tmp54;
    const s_t weak_mat_tmp89 = s_t(6)*gu[3];
    const s_t weak_mat_tmp90 = -s_t(2)*gu[2]*gu[7];
    const s_t weak_mat_tmp91 = gu[3]*weak_mat_tmp48;
    const s_t weak_mat_tmp92 = mu*(s_t(4)*weak_mat_tmp25 + weak_mat_tmp89 + weak_mat_tmp90) - weak_mat_tmp91;
    const s_t weak_mat_tmp93 = -s_t(2)*weak_mat_tmp2*weak_mat_tmp41;
    const s_t weak_mat_tmp94 = mu*(s_t(4)*weak_mat_tmp58 + weak_mat_tmp60 + weak_mat_tmp93) - weak_mat_tmp61;
    const s_t weak_mat_tmp95 = s_t(2)*weak_mat_tmp17;
    const s_t weak_mat_tmp96 = s_t(2)*weak_mat_tmp18;
    const s_t weak_mat_tmp97 = weak_mat_tmp12*(weak_mat_tmp95 - weak_mat_tmp96);
    const s_t weak_mat_tmp98 = mu*(-weak_mat_tmp23 - weak_mat_tmp76);
    const s_t weak_mat_tmp99 = mu*(-weak_mat_tmp27 - weak_mat_tmp78);
    const s_t weak_mat_tmp100 = s_t(2)*weak_mat_tmp47;
    const s_t weak_mat_tmp101 = mu*(s_t(4)*gu[2]*gu[3] - weak_mat_tmp100 - weak_mat_tmp44) + weak_mat_tmp49;
    const s_t weak_mat_tmp102 = s_t(2)*weak_mat_tmp25;
    const s_t weak_mat_tmp103 = mu*(s_t(4)*gu[2]*gu[7] - weak_mat_tmp102 - weak_mat_tmp89) + weak_mat_tmp91;
    const s_t weak_mat_tmp104 = s_t(2)*weak_mat_tmp30;
    const s_t weak_mat_tmp105 = mu*(-weak_mat_tmp104 + s_t(4)*weak_mat_tmp31 + weak_mat_tmp83) - weak_mat_tmp85;
    const s_t weak_mat_tmp106 = -s_t(2)*weak_mat_tmp41*weak_mat_tmp6;
    const s_t weak_mat_tmp107 = mu*(weak_mat_tmp106 + s_t(4)*weak_mat_tmp56 + weak_mat_tmp65) - weak_mat_tmp66;
    const s_t weak_mat_tmp108 = weak_mat_tmp12*(-weak_mat_tmp102 - weak_mat_tmp90);
    const s_t weak_mat_tmp109 = weak_mat_tmp22*weak_mat_tmp41;
    const s_t weak_mat_tmp110 = mu*(-weak_mat_tmp1 - weak_mat_tmp109);
    const s_t weak_mat_tmp111 = weak_mat_tmp20*weak_mat_tmp41;
    const s_t weak_mat_tmp112 = mu*(-weak_mat_tmp111 - weak_mat_tmp15);
    const s_t weak_mat_tmp113 = weak_mat_tmp6*weak_mat_tmp71;
    const s_t weak_mat_tmp114 = weak_mat_tmp2*weak_mat_tmp73;
    const s_t weak_mat_tmp115 = mu*(-weak_mat_tmp113 - weak_mat_tmp114);
    const s_t weak_mat_tmp116 = s_t(2)*pow_2(gu[2]);
    const s_t weak_mat_tmp117 = weak_mat_tmp116 + weak_mat_tmp34;
    const s_t weak_mat_tmp118 = s_t(2)*pow_2(gu[1]);
    const s_t weak_mat_tmp119 = weak_mat_tmp118 + weak_mat_tmp36;
    const s_t weak_mat_tmp120 = s_t(6)*gu[2];
    const s_t weak_mat_tmp121 = gu[2]*weak_mat_tmp48;
    const s_t weak_mat_tmp122 = mu*(s_t(4)*gu[3]*gu[7] - weak_mat_tmp120 - weak_mat_tmp96) + weak_mat_tmp121;
    const s_t weak_mat_tmp123 = s_t(6)*gu[1];
    const s_t weak_mat_tmp124 = gu[1]*weak_mat_tmp48;
    const s_t weak_mat_tmp125 = mu*(weak_mat_tmp123 + weak_mat_tmp68 + s_t(4)*weak_mat_tmp7) - weak_mat_tmp124;
    const s_t weak_mat_tmp126 = weak_mat_tmp12*(-weak_mat_tmp106 - weak_mat_tmp64);
    const s_t weak_mat_tmp127 = gu[2]*weak_mat_tmp22;
    const s_t weak_mat_tmp128 = mu*(-weak_mat_tmp127 - weak_mat_tmp74);
    const s_t weak_mat_tmp129 = gu[6]*weak_mat_tmp3;
    const s_t weak_mat_tmp130 = mu*(-weak_mat_tmp113 - weak_mat_tmp129);
    const s_t weak_mat_tmp131 = s_t(2)*pow_2(weak_mat_tmp41);
    const s_t weak_mat_tmp132 = weak_mat_tmp131 + weak_mat_tmp81;
    const s_t weak_mat_tmp133 = mu*(weak_mat_tmp120 + s_t(4)*weak_mat_tmp18 - weak_mat_tmp95) - weak_mat_tmp121;
    const s_t weak_mat_tmp134 = s_t(6)*gu[0] + s_t(6);
    const s_t weak_mat_tmp135 = weak_mat_tmp41*weak_mat_tmp48;
    const s_t weak_mat_tmp136 = mu*(-weak_mat_tmp10 - weak_mat_tmp134 + s_t(4)*weak_mat_tmp2*weak_mat_tmp6) + weak_mat_tmp135;
    const s_t weak_mat_tmp137 = weak_mat_tmp12*(weak_mat_tmp53 - weak_mat_tmp87);
    const s_t weak_mat_tmp138 = mu*(-weak_mat_tmp114 - weak_mat_tmp129);
    const s_t weak_mat_tmp139 = mu*(s_t(4)*gu[5]*gu[6] - weak_mat_tmp123 - weak_mat_tmp69) + weak_mat_tmp124;
    const s_t weak_mat_tmp140 = mu*(weak_mat_tmp11 + weak_mat_tmp134 + s_t(4)*weak_mat_tmp9) - weak_mat_tmp135;
    const s_t weak_mat_tmp141 = weak_mat_tmp12*(weak_mat_tmp104 - weak_mat_tmp84);
    const s_t weak_mat_tmp142 = mu*(-weak_mat_tmp109 - weak_mat_tmp4);
    const s_t weak_mat_tmp143 = mu*(-weak_mat_tmp111 - weak_mat_tmp14);
    const s_t weak_mat_tmp144 = weak_mat_tmp116 + weak_mat_tmp33 + s_t(2);
    const s_t weak_mat_tmp145 = weak_mat_tmp118 + weak_mat_tmp37;
    const s_t weak_mat_tmp146 = weak_mat_tmp12*(-weak_mat_tmp100 + weak_mat_tmp46);
    const s_t weak_mat_tmp147 = mu*(-weak_mat_tmp127 - weak_mat_tmp72);
    const s_t weak_mat_tmp148 = weak_mat_tmp131 + weak_mat_tmp80;
    const s_t weak_mat_tmp149 = weak_mat_tmp12*(-weak_mat_tmp59 - weak_mat_tmp93);
    material[0] = trial_grad[0]*(mu*(weak_mat_tmp35 + weak_mat_tmp38) + weak_mat_tmp13*weak_mat_tmp39) + trial_grad[1]*(weak_mat_tmp13*weak_mat_tmp8 + weak_mat_tmp5) + trial_grad[2]*(weak_mat_tmp13*weak_mat_tmp19 + weak_mat_tmp16) + trial_grad[3]*(weak_mat_tmp13*weak_mat_tmp26 + weak_mat_tmp24) + trial_grad[4]*(weak_mat_tmp13*weak_mat_tmp57 + weak_mat_tmp62) + trial_grad[5]*(weak_mat_tmp13*weak_mat_tmp43 + weak_mat_tmp50) + trial_grad[6]*(weak_mat_tmp13*weak_mat_tmp32 + weak_mat_tmp29) + trial_grad[7]*(weak_mat_tmp13*weak_mat_tmp51 + weak_mat_tmp55) + trial_grad[8]*(weak_mat_tmp13*weak_mat_tmp63 + weak_mat_tmp67);
    material[1] = trial_grad[0]*(weak_mat_tmp39*weak_mat_tmp70 + weak_mat_tmp5) + trial_grad[1]*(mu*(weak_mat_tmp35 + weak_mat_tmp82) + weak_mat_tmp70*weak_mat_tmp8) + trial_grad[2]*(weak_mat_tmp19*weak_mat_tmp70 + weak_mat_tmp75) + trial_grad[3]*(weak_mat_tmp26*weak_mat_tmp70 + weak_mat_tmp94) + trial_grad[4]*(weak_mat_tmp57*weak_mat_tmp70 + weak_mat_tmp77) + trial_grad[5]*(weak_mat_tmp43*weak_mat_tmp70 + weak_mat_tmp86) + trial_grad[6]*(weak_mat_tmp32*weak_mat_tmp70 + weak_mat_tmp88) + trial_grad[7]*(weak_mat_tmp51*weak_mat_tmp70 + weak_mat_tmp79) + trial_grad[8]*(weak_mat_tmp63*weak_mat_tmp70 + weak_mat_tmp92);
    material[2] = trial_grad[0]*(weak_mat_tmp16 + weak_mat_tmp39*weak_mat_tmp97) + trial_grad[1]*(weak_mat_tmp75 + weak_mat_tmp8*weak_mat_tmp97) + trial_grad[2]*(mu*(weak_mat_tmp38 + weak_mat_tmp82 + s_t(2)) + weak_mat_tmp19*weak_mat_tmp97) + trial_grad[3]*(weak_mat_tmp101 + weak_mat_tmp26*weak_mat_tmp97) + trial_grad[4]*(weak_mat_tmp105 + weak_mat_tmp57*weak_mat_tmp97) + trial_grad[5]*(weak_mat_tmp43*weak_mat_tmp97 + weak_mat_tmp98) + trial_grad[6]*(weak_mat_tmp107 + weak_mat_tmp32*weak_mat_tmp97) + trial_grad[7]*(weak_mat_tmp103 + weak_mat_tmp51*weak_mat_tmp97) + trial_grad[8]*(weak_mat_tmp63*weak_mat_tmp97 + weak_mat_tmp99);
    material[3] = trial_grad[0]*(weak_mat_tmp108*weak_mat_tmp39 + weak_mat_tmp24) + trial_grad[1]*(weak_mat_tmp108*weak_mat_tmp8 + weak_mat_tmp94) + trial_grad[2]*(weak_mat_tmp101 + weak_mat_tmp108*weak_mat_tmp19) + trial_grad[3]*(mu*(weak_mat_tmp117 + weak_mat_tmp119) + weak_mat_tmp108*weak_mat_tmp26) + trial_grad[4]*(weak_mat_tmp108*weak_mat_tmp57 + weak_mat_tmp110) + trial_grad[5]*(weak_mat_tmp108*weak_mat_tmp43 + weak_mat_tmp112) + trial_grad[6]*(weak_mat_tmp108*weak_mat_tmp32 + weak_mat_tmp115) + trial_grad[7]*(weak_mat_tmp108*weak_mat_tmp51 + weak_mat_tmp122) + trial_grad[8]*(weak_mat_tmp108*weak_mat_tmp63 + weak_mat_tmp125);
    material[4] = trial_grad[0]*(weak_mat_tmp126*weak_mat_tmp39 + weak_mat_tmp62) + trial_grad[1]*(weak_mat_tmp126*weak_mat_tmp8 + weak_mat_tmp77) + trial_grad[2]*(weak_mat_tmp105 + weak_mat_tmp126*weak_mat_tmp19) + trial_grad[3]*(weak_mat_tmp110 + weak_mat_tmp126*weak_mat_tmp26) + trial_grad[4]*(mu*(weak_mat_tmp117 + weak_mat_tmp132) + weak_mat_tmp126*weak_mat_tmp57) + trial_grad[5]*(weak_mat_tmp126*weak_mat_tmp43 + weak_mat_tmp128) + trial_grad[6]*(weak_mat_tmp126*weak_mat_tmp32 + weak_mat_tmp133) + trial_grad[7]*(weak_mat_tmp126*weak_mat_tmp51 + weak_mat_tmp130) + trial_grad[8]*(weak_mat_tmp126*weak_mat_tmp63 + weak_mat_tmp136);
    material[5] = trial_grad[0]*(weak_mat_tmp137*weak_mat_tmp39 + weak_mat_tmp50) + trial_grad[1]*(weak_mat_tmp137*weak_mat_tmp8 + weak_mat_tmp86) + trial_grad[2]*(weak_mat_tmp137*weak_mat_tmp19 + weak_mat_tmp98) + trial_grad[3]*(weak_mat_tmp112 + weak_mat_tmp137*weak_mat_tmp26) + trial_grad[4]*(weak_mat_tmp128 + weak_mat_tmp137*weak_mat_tmp57) + trial_grad[5]*(mu*(weak_mat_tmp119 + weak_mat_tmp132 + s_t(2)) + weak_mat_tmp137*weak_mat_tmp43) + trial_grad[6]*(weak_mat_tmp137*weak_mat_tmp32 + weak_mat_tmp139) + trial_grad[7]*(weak_mat_tmp137*weak_mat_tmp51 + weak_mat_tmp140) + trial_grad[8]*(weak_mat_tmp137*weak_mat_tmp63 + weak_mat_tmp138);
    material[6] = trial_grad[0]*(weak_mat_tmp141*weak_mat_tmp39 + weak_mat_tmp29) + trial_grad[1]*(weak_mat_tmp141*weak_mat_tmp8 + weak_mat_tmp88) + trial_grad[2]*(weak_mat_tmp107 + weak_mat_tmp141*weak_mat_tmp19) + trial_grad[3]*(weak_mat_tmp115 + weak_mat_tmp141*weak_mat_tmp26) + trial_grad[4]*(weak_mat_tmp133 + weak_mat_tmp141*weak_mat_tmp57) + trial_grad[5]*(weak_mat_tmp139 + weak_mat_tmp141*weak_mat_tmp43) + trial_grad[6]*(mu*(weak_mat_tmp144 + weak_mat_tmp145) + weak_mat_tmp141*weak_mat_tmp32) + trial_grad[7]*(weak_mat_tmp141*weak_mat_tmp51 + weak_mat_tmp142) + trial_grad[8]*(weak_mat_tmp141*weak_mat_tmp63 + weak_mat_tmp143);
    material[7] = trial_grad[0]*(weak_mat_tmp146*weak_mat_tmp39 + weak_mat_tmp55) + trial_grad[1]*(weak_mat_tmp146*weak_mat_tmp8 + weak_mat_tmp79) + trial_grad[2]*(weak_mat_tmp103 + weak_mat_tmp146*weak_mat_tmp19) + trial_grad[3]*(weak_mat_tmp122 + weak_mat_tmp146*weak_mat_tmp26) + trial_grad[4]*(weak_mat_tmp130 + weak_mat_tmp146*weak_mat_tmp57) + trial_grad[5]*(weak_mat_tmp140 + weak_mat_tmp146*weak_mat_tmp43) + trial_grad[6]*(weak_mat_tmp142 + weak_mat_tmp146*weak_mat_tmp32) + trial_grad[7]*(mu*(weak_mat_tmp144 + weak_mat_tmp148) + weak_mat_tmp146*weak_mat_tmp51) + trial_grad[8]*(weak_mat_tmp146*weak_mat_tmp63 + weak_mat_tmp147);
    material[8] = trial_grad[0]*(weak_mat_tmp149*weak_mat_tmp39 + weak_mat_tmp67) + trial_grad[1]*(weak_mat_tmp149*weak_mat_tmp8 + weak_mat_tmp92) + trial_grad[2]*(weak_mat_tmp149*weak_mat_tmp19 + weak_mat_tmp99) + trial_grad[3]*(weak_mat_tmp125 + weak_mat_tmp149*weak_mat_tmp26) + trial_grad[4]*(weak_mat_tmp136 + weak_mat_tmp149*weak_mat_tmp57) + trial_grad[5]*(weak_mat_tmp138 + weak_mat_tmp149*weak_mat_tmp43) + trial_grad[6]*(weak_mat_tmp143 + weak_mat_tmp149*weak_mat_tmp32) + trial_grad[7]*(weak_mat_tmp147 + weak_mat_tmp149*weak_mat_tmp51) + trial_grad[8]*(mu*(weak_mat_tmp145 + weak_mat_tmp148 + s_t(2)) + weak_mat_tmp149*weak_mat_tmp63);
    loperand[0] = qw * (material[0] * adj_lane0 + material[1] * adj_lane1 + material[2] * adj_lane2);
    loperand[1] = qw * (material[0] * adj_lane3 + material[1] * adj_lane4 + material[2] * adj_lane5);
    loperand[2] = qw * (material[0] * adj_lane6 + material[1] * adj_lane7 + material[2] * adj_lane8);
    loperand[3] = qw * (material[3] * adj_lane0 + material[4] * adj_lane1 + material[5] * adj_lane2);
    loperand[4] = qw * (material[3] * adj_lane3 + material[4] * adj_lane4 + material[5] * adj_lane5);
    loperand[5] = qw * (material[3] * adj_lane6 + material[4] * adj_lane7 + material[5] * adj_lane8);
    loperand[6] = qw * (material[6] * adj_lane0 + material[7] * adj_lane1 + material[8] * adj_lane2);
    loperand[7] = qw * (material[6] * adj_lane3 + material[7] * adj_lane4 + material[8] * adj_lane5);
    loperand[8] = qw * (material[6] * adj_lane6 + material[7] * adj_lane7 + material[8] * adj_lane8);
      loperand_q[((0 * NQ + q) * 3 + 0) * VS + lane] = loperand[0];
      loperand_q[((0 * NQ + q) * 3 + 1) * VS + lane] = loperand[1];
      loperand_q[((0 * NQ + q) * 3 + 2) * VS + lane] = loperand[2];
      loperand_q[((1 * NQ + q) * 3 + 0) * VS + lane] = loperand[3];
      loperand_q[((1 * NQ + q) * 3 + 1) * VS + lane] = loperand[4];
      loperand_q[((1 * NQ + q) * 3 + 2) * VS + lane] = loperand[5];
      loperand_q[((2 * NQ + q) * 3 + 0) * VS + lane] = loperand[6];
      loperand_q[((2 * NQ + q) * 3 + 1) * VS + lane] = loperand[7];
      loperand_q[((2 * NQ + q) * 3 + 2) * VS + lane] = loperand[8];
    }
  }
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[0 * NQ * 3 * VS], out_streams, 0);
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[1 * NQ * 3 * VS], out_streams, 1);
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[2 * NQ * 3 * VS], out_streams, 2);
}

} // namespace codegen
} // namespace sfem

#endif
