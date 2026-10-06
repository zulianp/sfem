#ifndef MODIFIED_MOONEY_RIVLIN_D3_TENSOR_PRODUCT_LOCAL_HPP
#define MODIFIED_MOONEY_RIVLIN_D3_TENSOR_PRODUCT_LOCAL_HPP
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
static SFEM_INLINE void modified_mooney_rivlin_d3_tensor_product_objective_block(
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
        const s_t c1,
        const s_t c2,
        const s_t kappa,
        const s_t *const RSTR u_streams[NS * 3],
        const s_t *const RSTR h_streams[NS * 3],
        const int nsteps,
        const s_t *const RSTR steps,
        const ptrdiff_t value_stride,
        s_t *const RSTR value
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(VS > 0, "VS must be positive");
  static constexpr int NQ1 = integer_root(NQ, 3);
  static constexpr int NS1 = integer_root(NS, 3);
  static_assert(ipow(NQ1, 3) == NQ, "NQ must be tensor-product compatible");
  static_assert(ipow(NS1, 3) == NS, "NS must be tensor-product compatible");
  s_t gu_ref_q[NQ * 9 * VS];
  s_t grad_h_ref_q[NQ * 9 * VS];
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, h_streams, 0, &grad_h_ref_q[0]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 1, &gu_ref_q[3 * NQ * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, h_streams, 1, &grad_h_ref_q[3 * NQ * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 2, &gu_ref_q[6 * NQ * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, h_streams, 2, &grad_h_ref_q[6 * NQ * VS]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
    const s_t *const RSTR gu_ref0 = &gu_ref_q[(3 * q) * VS];
    const s_t *const RSTR gu_ref1 = &gu_ref_q[(3 * q + 1) * VS];
    const s_t *const RSTR gu_ref2 = &gu_ref_q[(3 * q + 2) * VS];
    const s_t *const RSTR gu_ref3 = &gu_ref_q[(3 * (NQ + q)) * VS];
    const s_t *const RSTR gu_ref4 = &gu_ref_q[(3 * (NQ + q) + 1) * VS];
    const s_t *const RSTR gu_ref5 = &gu_ref_q[(3 * (NQ + q) + 2) * VS];
    const s_t *const RSTR gu_ref6 = &gu_ref_q[(3 * (2 * NQ + q)) * VS];
    const s_t *const RSTR gu_ref7 = &gu_ref_q[(3 * (2 * NQ + q) + 1) * VS];
    const s_t *const RSTR gu_ref8 = &gu_ref_q[(3 * (2 * NQ + q) + 2) * VS];
    const s_t *const RSTR grad_h_ref0 = &grad_h_ref_q[(3 * q) * VS];
    const s_t *const RSTR grad_h_ref1 = &grad_h_ref_q[(3 * q + 1) * VS];
    const s_t *const RSTR grad_h_ref2 = &grad_h_ref_q[(3 * q + 2) * VS];
    const s_t *const RSTR grad_h_ref3 = &grad_h_ref_q[(3 * (NQ + q)) * VS];
    const s_t *const RSTR grad_h_ref4 = &grad_h_ref_q[(3 * (NQ + q) + 1) * VS];
    const s_t *const RSTR grad_h_ref5 = &grad_h_ref_q[(3 * (NQ + q) + 2) * VS];
    const s_t *const RSTR grad_h_ref6 = &grad_h_ref_q[(3 * (2 * NQ + q)) * VS];
    const s_t *const RSTR grad_h_ref7 = &grad_h_ref_q[(3 * (2 * NQ + q) + 1) * VS];
    const s_t *const RSTR grad_h_ref8 = &grad_h_ref_q[(3 * (2 * NQ + q) + 2) * VS];
    const s_t *const RSTR adj_q0 = adj0 + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adj1 + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adj2 + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adj3 + q * geometry_stride;
    const s_t *const RSTR adj_q4 = adj4 + q * geometry_stride;
    const s_t *const RSTR adj_q5 = adj5 + q * geometry_stride;
    const s_t *const RSTR adj_q6 = adj6 + q * geometry_stride;
    const s_t *const RSTR adj_q7 = adj7 + q * geometry_stride;
    const s_t *const RSTR adj_q8 = adj8 + q * geometry_stride;
    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;
    s_t gu_base_v[9 * VS];
    s_t trial_grad_v[9 * VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t adj_lane0 = adj_q0[lane];
      const s_t adj_lane1 = adj_q1[lane];
      const s_t adj_lane2 = adj_q2[lane];
      const s_t adj_lane3 = adj_q3[lane];
      const s_t adj_lane4 = adj_q4[lane];
      const s_t adj_lane5 = adj_q5[lane];
      const s_t adj_lane6 = adj_q6[lane];
      const s_t adj_lane7 = adj_q7[lane];
      const s_t adj_lane8 = adj_q8[lane];
      const s_t det_lane0 = det_q0[lane];
      const s_t idet = s_t(1) / det_lane0;
      gu_base_v[0 * VS + lane] = (gu_ref0[lane] * adj_lane0 + gu_ref1[lane] * adj_lane3 + gu_ref2[lane] * adj_lane6) * idet;
      trial_grad_v[0 * VS + lane] = (grad_h_ref0[lane] * adj_lane0 + grad_h_ref1[lane] * adj_lane3 + grad_h_ref2[lane] * adj_lane6) * idet;
      gu_base_v[1 * VS + lane] = (gu_ref0[lane] * adj_lane1 + gu_ref1[lane] * adj_lane4 + gu_ref2[lane] * adj_lane7) * idet;
      trial_grad_v[1 * VS + lane] = (grad_h_ref0[lane] * adj_lane1 + grad_h_ref1[lane] * adj_lane4 + grad_h_ref2[lane] * adj_lane7) * idet;
      gu_base_v[2 * VS + lane] = (gu_ref0[lane] * adj_lane2 + gu_ref1[lane] * adj_lane5 + gu_ref2[lane] * adj_lane8) * idet;
      trial_grad_v[2 * VS + lane] = (grad_h_ref0[lane] * adj_lane2 + grad_h_ref1[lane] * adj_lane5 + grad_h_ref2[lane] * adj_lane8) * idet;
      gu_base_v[3 * VS + lane] = (gu_ref3[lane] * adj_lane0 + gu_ref4[lane] * adj_lane3 + gu_ref5[lane] * adj_lane6) * idet;
      trial_grad_v[3 * VS + lane] = (grad_h_ref3[lane] * adj_lane0 + grad_h_ref4[lane] * adj_lane3 + grad_h_ref5[lane] * adj_lane6) * idet;
      gu_base_v[4 * VS + lane] = (gu_ref3[lane] * adj_lane1 + gu_ref4[lane] * adj_lane4 + gu_ref5[lane] * adj_lane7) * idet;
      trial_grad_v[4 * VS + lane] = (grad_h_ref3[lane] * adj_lane1 + grad_h_ref4[lane] * adj_lane4 + grad_h_ref5[lane] * adj_lane7) * idet;
      gu_base_v[5 * VS + lane] = (gu_ref3[lane] * adj_lane2 + gu_ref4[lane] * adj_lane5 + gu_ref5[lane] * adj_lane8) * idet;
      trial_grad_v[5 * VS + lane] = (grad_h_ref3[lane] * adj_lane2 + grad_h_ref4[lane] * adj_lane5 + grad_h_ref5[lane] * adj_lane8) * idet;
      gu_base_v[6 * VS + lane] = (gu_ref6[lane] * adj_lane0 + gu_ref7[lane] * adj_lane3 + gu_ref8[lane] * adj_lane6) * idet;
      trial_grad_v[6 * VS + lane] = (grad_h_ref6[lane] * adj_lane0 + grad_h_ref7[lane] * adj_lane3 + grad_h_ref8[lane] * adj_lane6) * idet;
      gu_base_v[7 * VS + lane] = (gu_ref6[lane] * adj_lane1 + gu_ref7[lane] * adj_lane4 + gu_ref8[lane] * adj_lane7) * idet;
      trial_grad_v[7 * VS + lane] = (grad_h_ref6[lane] * adj_lane1 + grad_h_ref7[lane] * adj_lane4 + grad_h_ref8[lane] * adj_lane7) * idet;
      gu_base_v[8 * VS + lane] = (gu_ref6[lane] * adj_lane2 + gu_ref7[lane] * adj_lane5 + gu_ref8[lane] * adj_lane8) * idet;
      trial_grad_v[8 * VS + lane] = (grad_h_ref6[lane] * adj_lane2 + grad_h_ref7[lane] * adj_lane5 + grad_h_ref8[lane] * adj_lane8) * idet;
    }
    for (int step = 0; step < nsteps; ++step) {
      const s_t alpha = steps[step];
      #pragma omp simd
      for (int lane = 0; lane < ne; ++lane) {
        const s_t det_lane0 = det_q0[lane];
        s_t gu[9];
        gu[0] = gu_base_v[0 * VS + lane] + alpha * trial_grad_v[0 * VS + lane];
        gu[1] = gu_base_v[1 * VS + lane] + alpha * trial_grad_v[1 * VS + lane];
        gu[2] = gu_base_v[2 * VS + lane] + alpha * trial_grad_v[2 * VS + lane];
        gu[3] = gu_base_v[3 * VS + lane] + alpha * trial_grad_v[3 * VS + lane];
        gu[4] = gu_base_v[4 * VS + lane] + alpha * trial_grad_v[4 * VS + lane];
        gu[5] = gu_base_v[5 * VS + lane] + alpha * trial_grad_v[5 * VS + lane];
        gu[6] = gu_base_v[6 * VS + lane] + alpha * trial_grad_v[6 * VS + lane];
        gu[7] = gu_base_v[7 * VS + lane] + alpha * trial_grad_v[7 * VS + lane];
        gu[8] = gu_base_v[8 * VS + lane] + alpha * trial_grad_v[8 * VS + lane];
    const s_t weak_obj_tmp0 = gu[0]*gu[4];
    const s_t weak_obj_tmp1 = gu[1]*gu[3];
    const s_t weak_obj_tmp2 = gu[2]*gu[6];
    const s_t weak_obj_tmp3 = gu[5]*gu[7];
    const s_t weak_obj_tmp4 = gu[4] + s_t(1);
    const s_t weak_obj_tmp5 = pow_2(gu[1]) + pow_2(gu[7]) + pow_2(weak_obj_tmp4);
    const s_t weak_obj_tmp6 = gu[8] + s_t(1);
    const s_t weak_obj_tmp7 = pow_2(gu[2]) + pow_2(gu[5]) + pow_2(weak_obj_tmp6);
    const s_t weak_obj_tmp8 = gu[0] + s_t(1);
    const s_t weak_obj_tmp9 = pow_2(gu[3]) + pow_2(gu[6]) + pow_2(weak_obj_tmp8);
    const s_t weak_obj_tmp10 = weak_obj_tmp5 + weak_obj_tmp7 + weak_obj_tmp9;
    const s_t weak_obj_tmp11 = gu[1]*gu[5]*gu[6] + gu[2]*gu[3]*gu[7] - weak_obj_tmp1*weak_obj_tmp6 - weak_obj_tmp2*weak_obj_tmp4 - weak_obj_tmp3*weak_obj_tmp8 + weak_obj_tmp4*weak_obj_tmp6*weak_obj_tmp8;
    value[step * value_stride + lane] += qw * det_lane0 * (c1*(weak_obj_tmp10/pow(weak_obj_tmp11, (s_t(2) / s_t(3))) + s_t(-3)) + c2*(s_t(-3) + (((s_t(1) / s_t(2)))*pow_2(weak_obj_tmp10) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp5) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp7) - (s_t(1) / s_t(2))*pow_2(weak_obj_tmp9) - pow_2(gu[1]*gu[2] + gu[5]*weak_obj_tmp4 + gu[7]*weak_obj_tmp6) - pow_2(gu[1]*weak_obj_tmp8 + gu[3]*weak_obj_tmp4 + gu[6]*gu[7]) - pow_2(gu[2]*weak_obj_tmp8 + gu[3]*gu[5] + gu[6]*weak_obj_tmp6))/pow(weak_obj_tmp11, (s_t(4) / s_t(3)))) + ((s_t(1) / s_t(2)))*kappa*pow_2(sfem_log1p(gu[0]*gu[8] - gu[0]*weak_obj_tmp3 + gu[0] + gu[1]*gu[5]*gu[6] + gu[2]*gu[3]*gu[7] + gu[4]*gu[8] - gu[4]*weak_obj_tmp2 + gu[4] + gu[8]*weak_obj_tmp0 - gu[8]*weak_obj_tmp1 + gu[8] + weak_obj_tmp0 - weak_obj_tmp1 - weak_obj_tmp2 - weak_obj_tmp3)));
      }
    }
  }
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void modified_mooney_rivlin_d3_tensor_product_gradient_block(
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
        const s_t c1,
        const s_t c2,
        const s_t kappa,
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
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 1, &gu_ref_q[3 * NQ * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 2, &gu_ref_q[6 * NQ * VS]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
    const s_t *const RSTR gu_ref0 = &gu_ref_q[(3 * q) * VS];
    const s_t *const RSTR gu_ref1 = &gu_ref_q[(3 * q + 1) * VS];
    const s_t *const RSTR gu_ref2 = &gu_ref_q[(3 * q + 2) * VS];
    const s_t *const RSTR gu_ref3 = &gu_ref_q[(3 * (NQ + q)) * VS];
    const s_t *const RSTR gu_ref4 = &gu_ref_q[(3 * (NQ + q) + 1) * VS];
    const s_t *const RSTR gu_ref5 = &gu_ref_q[(3 * (NQ + q) + 2) * VS];
    const s_t *const RSTR gu_ref6 = &gu_ref_q[(3 * (2 * NQ + q)) * VS];
    const s_t *const RSTR gu_ref7 = &gu_ref_q[(3 * (2 * NQ + q) + 1) * VS];
    const s_t *const RSTR gu_ref8 = &gu_ref_q[(3 * (2 * NQ + q) + 2) * VS];
    s_t *const RSTR loperand0 = &loperand_q[(3 * q) * VS];
    s_t *const RSTR loperand1 = &loperand_q[(3 * q + 1) * VS];
    s_t *const RSTR loperand2 = &loperand_q[(3 * q + 2) * VS];
    s_t *const RSTR loperand3 = &loperand_q[(3 * (NQ + q)) * VS];
    s_t *const RSTR loperand4 = &loperand_q[(3 * (NQ + q) + 1) * VS];
    s_t *const RSTR loperand5 = &loperand_q[(3 * (NQ + q) + 2) * VS];
    s_t *const RSTR loperand6 = &loperand_q[(3 * (2 * NQ + q)) * VS];
    s_t *const RSTR loperand7 = &loperand_q[(3 * (2 * NQ + q) + 1) * VS];
    s_t *const RSTR loperand8 = &loperand_q[(3 * (2 * NQ + q) + 2) * VS];
    const s_t *const RSTR adj_q0 = adj0 + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adj1 + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adj2 + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adj3 + q * geometry_stride;
    const s_t *const RSTR adj_q4 = adj4 + q * geometry_stride;
    const s_t *const RSTR adj_q5 = adj5 + q * geometry_stride;
    const s_t *const RSTR adj_q6 = adj6 + q * geometry_stride;
    const s_t *const RSTR adj_q7 = adj7 + q * geometry_stride;
    const s_t *const RSTR adj_q8 = adj8 + q * geometry_stride;
    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t adj_lane0 = adj_q0[lane];
      const s_t adj_lane1 = adj_q1[lane];
      const s_t adj_lane2 = adj_q2[lane];
      const s_t adj_lane3 = adj_q3[lane];
      const s_t adj_lane4 = adj_q4[lane];
      const s_t adj_lane5 = adj_q5[lane];
      const s_t adj_lane6 = adj_q6[lane];
      const s_t adj_lane7 = adj_q7[lane];
      const s_t adj_lane8 = adj_q8[lane];
      const s_t det_lane0 = det_q0[lane];
      s_t gu[9];
      const s_t idet = s_t(1) / det_lane0;
      gu[0] = (gu_ref0[lane] * adj_lane0 + gu_ref1[lane] * adj_lane3 + gu_ref2[lane] * adj_lane6) * idet;
      gu[1] = (gu_ref0[lane] * adj_lane1 + gu_ref1[lane] * adj_lane4 + gu_ref2[lane] * adj_lane7) * idet;
      gu[2] = (gu_ref0[lane] * adj_lane2 + gu_ref1[lane] * adj_lane5 + gu_ref2[lane] * adj_lane8) * idet;
      gu[3] = (gu_ref3[lane] * adj_lane0 + gu_ref4[lane] * adj_lane3 + gu_ref5[lane] * adj_lane6) * idet;
      gu[4] = (gu_ref3[lane] * adj_lane1 + gu_ref4[lane] * adj_lane4 + gu_ref5[lane] * adj_lane7) * idet;
      gu[5] = (gu_ref3[lane] * adj_lane2 + gu_ref4[lane] * adj_lane5 + gu_ref5[lane] * adj_lane8) * idet;
      gu[6] = (gu_ref6[lane] * adj_lane0 + gu_ref7[lane] * adj_lane3 + gu_ref8[lane] * adj_lane6) * idet;
      gu[7] = (gu_ref6[lane] * adj_lane1 + gu_ref7[lane] * adj_lane4 + gu_ref8[lane] * adj_lane7) * idet;
      gu[8] = (gu_ref6[lane] * adj_lane2 + gu_ref7[lane] * adj_lane5 + gu_ref8[lane] * adj_lane8) * idet;
      s_t loperand[9];
    s_t material[9];
    const s_t weak_mat_tmp0 = gu[5]*gu[7];
    const s_t weak_mat_tmp1 = gu[4] + s_t(1);
    const s_t weak_mat_tmp2 = gu[8] + s_t(1);
    const s_t weak_mat_tmp3 = gu[1]*gu[3];
    const s_t weak_mat_tmp4 = gu[2]*gu[6];
    const s_t weak_mat_tmp5 = gu[0] + s_t(1);
    const s_t weak_mat_tmp6 = gu[1]*gu[5]*gu[6] + gu[2]*gu[3]*gu[7] - weak_mat_tmp0*weak_mat_tmp5 + weak_mat_tmp1*weak_mat_tmp2*weak_mat_tmp5 - weak_mat_tmp1*weak_mat_tmp4 - weak_mat_tmp2*weak_mat_tmp3;
    const s_t weak_mat_tmp7 = gu[0]*gu[4];
    const s_t weak_mat_tmp8 = gu[5]*gu[6];
    const s_t weak_mat_tmp9 = gu[3]*gu[7];
    const s_t weak_mat_tmp10 = kappa*sfem_log1p(gu[0]*gu[8] - gu[0]*weak_mat_tmp0 + gu[0] + gu[1]*weak_mat_tmp8 + gu[2]*weak_mat_tmp9 + gu[4]*gu[8] - gu[4]*weak_mat_tmp4 + gu[4] - gu[8]*weak_mat_tmp3 + gu[8]*weak_mat_tmp7 + gu[8] - weak_mat_tmp0 - weak_mat_tmp3 - weak_mat_tmp4 + weak_mat_tmp7)/weak_mat_tmp6;
    const s_t weak_mat_tmp11 = pow(weak_mat_tmp6, (s_t(-2) / s_t(3)));
    const s_t weak_mat_tmp12 = s_t(2)*weak_mat_tmp5;
    const s_t weak_mat_tmp13 = weak_mat_tmp1*weak_mat_tmp2;
    const s_t weak_mat_tmp14 = pow_2(gu[3]) + pow_2(gu[6]) + pow_2(weak_mat_tmp5);
    const s_t weak_mat_tmp15 = pow_2(gu[1]) + pow_2(gu[7]) + pow_2(weak_mat_tmp1);
    const s_t weak_mat_tmp16 = pow_2(gu[2]) + pow_2(gu[5]) + pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp17 = weak_mat_tmp14 + weak_mat_tmp15 + weak_mat_tmp16;
    const s_t weak_mat_tmp18 = weak_mat_tmp17/pow(weak_mat_tmp6, (s_t(5) / s_t(3)));
    const s_t weak_mat_tmp19 = pow(weak_mat_tmp6, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp20 = gu[1]*weak_mat_tmp5 + gu[3]*weak_mat_tmp1 + gu[6]*gu[7];
    const s_t weak_mat_tmp21 = s_t(2)*gu[1];
    const s_t weak_mat_tmp22 = gu[2]*weak_mat_tmp5 + gu[3]*gu[5] + gu[6]*weak_mat_tmp2;
    const s_t weak_mat_tmp23 = s_t(2)*gu[2];
    const s_t weak_mat_tmp24 = gu[1]*gu[2] + gu[5]*weak_mat_tmp1 + gu[7]*weak_mat_tmp2;
    const s_t weak_mat_tmp25 = (-(s_t(1) / s_t(2))*pow_2(weak_mat_tmp14) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp15) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp16) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp17) - pow_2(weak_mat_tmp20) - pow_2(weak_mat_tmp22) - pow_2(weak_mat_tmp24))/pow(weak_mat_tmp6, (s_t(7) / s_t(3)));
    const s_t weak_mat_tmp26 = gu[3]*weak_mat_tmp2;
    const s_t weak_mat_tmp27 = gu[1]*weak_mat_tmp2;
    const s_t weak_mat_tmp28 = s_t(2)*gu[3];
    const s_t weak_mat_tmp29 = gu[2]*gu[7];
    const s_t weak_mat_tmp30 = s_t(2)*gu[5];
    const s_t weak_mat_tmp31 = s_t(2)*weak_mat_tmp1;
    const s_t weak_mat_tmp32 = weak_mat_tmp2*weak_mat_tmp5;
    const s_t weak_mat_tmp33 = gu[1]*gu[6];
    const s_t weak_mat_tmp34 = gu[1]*gu[5];
    const s_t weak_mat_tmp35 = s_t(2)*gu[6];
    const s_t weak_mat_tmp36 = s_t(2)*gu[7];
    const s_t weak_mat_tmp37 = s_t(2)*weak_mat_tmp2;
    const s_t weak_mat_tmp38 = gu[2]*gu[3];
    const s_t weak_mat_tmp39 = weak_mat_tmp1*weak_mat_tmp5;
    material[0] = c1*(weak_mat_tmp11*weak_mat_tmp12 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*weak_mat_tmp0 - (s_t(2) / s_t(3))*weak_mat_tmp13)) + c2*(weak_mat_tmp19*(-weak_mat_tmp12*weak_mat_tmp14 + s_t(2)*weak_mat_tmp17*weak_mat_tmp5 - weak_mat_tmp20*weak_mat_tmp21 - weak_mat_tmp22*weak_mat_tmp23) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*weak_mat_tmp0 - (s_t(4) / s_t(3))*weak_mat_tmp13)) + weak_mat_tmp10*(-weak_mat_tmp0 + weak_mat_tmp1*weak_mat_tmp2);
    material[1] = c1*(weak_mat_tmp11*weak_mat_tmp21 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*weak_mat_tmp26 - (s_t(2) / s_t(3))*weak_mat_tmp8)) + c2*(weak_mat_tmp19*(s_t(2)*gu[1]*weak_mat_tmp17 - weak_mat_tmp12*weak_mat_tmp20 - weak_mat_tmp15*weak_mat_tmp21 - weak_mat_tmp23*weak_mat_tmp24) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*weak_mat_tmp26 - (s_t(4) / s_t(3))*weak_mat_tmp8)) + weak_mat_tmp10*(gu[5]*gu[6] - weak_mat_tmp26);
    material[2] = c1*(weak_mat_tmp11*weak_mat_tmp23 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu[6]*weak_mat_tmp1 - (s_t(2) / s_t(3))*weak_mat_tmp9)) + c2*(weak_mat_tmp19*(s_t(2)*gu[2]*weak_mat_tmp17 - weak_mat_tmp12*weak_mat_tmp22 - weak_mat_tmp16*weak_mat_tmp23 - weak_mat_tmp21*weak_mat_tmp24) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu[6]*weak_mat_tmp1 - (s_t(4) / s_t(3))*weak_mat_tmp9)) + weak_mat_tmp10*(-gu[6]*weak_mat_tmp1 + weak_mat_tmp9);
    material[3] = c1*(weak_mat_tmp11*weak_mat_tmp28 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*weak_mat_tmp27 - (s_t(2) / s_t(3))*weak_mat_tmp29)) + c2*(weak_mat_tmp19*(s_t(2)*gu[3]*weak_mat_tmp17 - weak_mat_tmp14*weak_mat_tmp28 - weak_mat_tmp20*weak_mat_tmp31 - weak_mat_tmp22*weak_mat_tmp30) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*weak_mat_tmp27 - (s_t(4) / s_t(3))*weak_mat_tmp29)) + weak_mat_tmp10*(gu[2]*gu[7] - weak_mat_tmp27);
    material[4] = c1*(weak_mat_tmp11*weak_mat_tmp31 + weak_mat_tmp18*(-(s_t(2) / s_t(3))*weak_mat_tmp32 + ((s_t(2) / s_t(3)))*weak_mat_tmp4)) + c2*(weak_mat_tmp19*(s_t(2)*weak_mat_tmp1*weak_mat_tmp17 - weak_mat_tmp15*weak_mat_tmp31 - weak_mat_tmp20*weak_mat_tmp28 - weak_mat_tmp24*weak_mat_tmp30) + weak_mat_tmp25*(-(s_t(4) / s_t(3))*weak_mat_tmp32 + ((s_t(4) / s_t(3)))*weak_mat_tmp4)) + weak_mat_tmp10*(weak_mat_tmp2*weak_mat_tmp5 - weak_mat_tmp4);
    material[5] = c1*(weak_mat_tmp11*weak_mat_tmp30 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu[7]*weak_mat_tmp5 - (s_t(2) / s_t(3))*weak_mat_tmp33)) + c2*(weak_mat_tmp19*(s_t(2)*gu[5]*weak_mat_tmp17 - weak_mat_tmp16*weak_mat_tmp30 - weak_mat_tmp22*weak_mat_tmp28 - weak_mat_tmp24*weak_mat_tmp31) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu[7]*weak_mat_tmp5 - (s_t(4) / s_t(3))*weak_mat_tmp33)) + weak_mat_tmp10*(-gu[7]*weak_mat_tmp5 + weak_mat_tmp33);
    material[6] = c1*(weak_mat_tmp11*weak_mat_tmp35 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu[2]*weak_mat_tmp1 - (s_t(2) / s_t(3))*weak_mat_tmp34)) + c2*(weak_mat_tmp19*(s_t(2)*gu[6]*weak_mat_tmp17 - weak_mat_tmp14*weak_mat_tmp35 - weak_mat_tmp20*weak_mat_tmp36 - weak_mat_tmp22*weak_mat_tmp37) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu[2]*weak_mat_tmp1 - (s_t(4) / s_t(3))*weak_mat_tmp34)) + weak_mat_tmp10*(-gu[2]*weak_mat_tmp1 + weak_mat_tmp34);
    material[7] = c1*(weak_mat_tmp11*weak_mat_tmp36 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*gu[5]*weak_mat_tmp5 - (s_t(2) / s_t(3))*weak_mat_tmp38)) + c2*(weak_mat_tmp19*(s_t(2)*gu[7]*weak_mat_tmp17 - weak_mat_tmp15*weak_mat_tmp36 - weak_mat_tmp20*weak_mat_tmp35 - weak_mat_tmp24*weak_mat_tmp37) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*gu[5]*weak_mat_tmp5 - (s_t(4) / s_t(3))*weak_mat_tmp38)) + weak_mat_tmp10*(-gu[5]*weak_mat_tmp5 + weak_mat_tmp38);
    material[8] = c1*(weak_mat_tmp11*weak_mat_tmp37 + weak_mat_tmp18*(((s_t(2) / s_t(3)))*weak_mat_tmp3 - (s_t(2) / s_t(3))*weak_mat_tmp39)) + c2*(weak_mat_tmp19*(-weak_mat_tmp16*weak_mat_tmp37 + s_t(2)*weak_mat_tmp17*weak_mat_tmp2 - weak_mat_tmp22*weak_mat_tmp35 - weak_mat_tmp24*weak_mat_tmp36) + weak_mat_tmp25*(((s_t(4) / s_t(3)))*weak_mat_tmp3 - (s_t(4) / s_t(3))*weak_mat_tmp39)) + weak_mat_tmp10*(weak_mat_tmp1*weak_mat_tmp5 - weak_mat_tmp3);
    loperand[0] = qw * (material[0] * adj_lane0 + material[1] * adj_lane1 + material[2] * adj_lane2);
    loperand[1] = qw * (material[0] * adj_lane3 + material[1] * adj_lane4 + material[2] * adj_lane5);
    loperand[2] = qw * (material[0] * adj_lane6 + material[1] * adj_lane7 + material[2] * adj_lane8);
    loperand[3] = qw * (material[3] * adj_lane0 + material[4] * adj_lane1 + material[5] * adj_lane2);
    loperand[4] = qw * (material[3] * adj_lane3 + material[4] * adj_lane4 + material[5] * adj_lane5);
    loperand[5] = qw * (material[3] * adj_lane6 + material[4] * adj_lane7 + material[5] * adj_lane8);
    loperand[6] = qw * (material[6] * adj_lane0 + material[7] * adj_lane1 + material[8] * adj_lane2);
    loperand[7] = qw * (material[6] * adj_lane3 + material[7] * adj_lane4 + material[8] * adj_lane5);
    loperand[8] = qw * (material[6] * adj_lane6 + material[7] * adj_lane7 + material[8] * adj_lane8);
      loperand0[lane] = loperand[0];
      loperand1[lane] = loperand[1];
      loperand2[lane] = loperand[2];
      loperand3[lane] = loperand[3];
      loperand4[lane] = loperand[4];
      loperand5[lane] = loperand[5];
      loperand6[lane] = loperand[6];
      loperand7[lane] = loperand[7];
      loperand8[lane] = loperand[8];
    }
  }
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[0], out_streams, 0);
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[3 * NQ * VS], out_streams, 1);
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[6 * NQ * VS], out_streams, 2);
}

template <typename s_t, int NQ, int NS, int VS>
static SFEM_INLINE void modified_mooney_rivlin_d3_tensor_product_apply_block(
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
        const s_t c1,
        const s_t c2,
        const s_t kappa,
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
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 0, &gu_ref_q[0]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, h_streams, 0, &grad_h_ref_q[0]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 1, &gu_ref_q[3 * NQ * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, h_streams, 1, &grad_h_ref_q[3 * NQ * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, u_streams, 2, &gu_ref_q[6 * NQ * VS]);
  tensor_gradient<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, h_streams, 2, &grad_h_ref_q[6 * NQ * VS]);
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
    const s_t *const RSTR gu_ref0 = &gu_ref_q[(3 * q) * VS];
    const s_t *const RSTR gu_ref1 = &gu_ref_q[(3 * q + 1) * VS];
    const s_t *const RSTR gu_ref2 = &gu_ref_q[(3 * q + 2) * VS];
    const s_t *const RSTR gu_ref3 = &gu_ref_q[(3 * (NQ + q)) * VS];
    const s_t *const RSTR gu_ref4 = &gu_ref_q[(3 * (NQ + q) + 1) * VS];
    const s_t *const RSTR gu_ref5 = &gu_ref_q[(3 * (NQ + q) + 2) * VS];
    const s_t *const RSTR gu_ref6 = &gu_ref_q[(3 * (2 * NQ + q)) * VS];
    const s_t *const RSTR gu_ref7 = &gu_ref_q[(3 * (2 * NQ + q) + 1) * VS];
    const s_t *const RSTR gu_ref8 = &gu_ref_q[(3 * (2 * NQ + q) + 2) * VS];
    const s_t *const RSTR grad_h_ref0 = &grad_h_ref_q[(3 * q) * VS];
    const s_t *const RSTR grad_h_ref1 = &grad_h_ref_q[(3 * q + 1) * VS];
    const s_t *const RSTR grad_h_ref2 = &grad_h_ref_q[(3 * q + 2) * VS];
    const s_t *const RSTR grad_h_ref3 = &grad_h_ref_q[(3 * (NQ + q)) * VS];
    const s_t *const RSTR grad_h_ref4 = &grad_h_ref_q[(3 * (NQ + q) + 1) * VS];
    const s_t *const RSTR grad_h_ref5 = &grad_h_ref_q[(3 * (NQ + q) + 2) * VS];
    const s_t *const RSTR grad_h_ref6 = &grad_h_ref_q[(3 * (2 * NQ + q)) * VS];
    const s_t *const RSTR grad_h_ref7 = &grad_h_ref_q[(3 * (2 * NQ + q) + 1) * VS];
    const s_t *const RSTR grad_h_ref8 = &grad_h_ref_q[(3 * (2 * NQ + q) + 2) * VS];
    s_t *const RSTR loperand0 = &loperand_q[(3 * q) * VS];
    s_t *const RSTR loperand1 = &loperand_q[(3 * q + 1) * VS];
    s_t *const RSTR loperand2 = &loperand_q[(3 * q + 2) * VS];
    s_t *const RSTR loperand3 = &loperand_q[(3 * (NQ + q)) * VS];
    s_t *const RSTR loperand4 = &loperand_q[(3 * (NQ + q) + 1) * VS];
    s_t *const RSTR loperand5 = &loperand_q[(3 * (NQ + q) + 2) * VS];
    s_t *const RSTR loperand6 = &loperand_q[(3 * (2 * NQ + q)) * VS];
    s_t *const RSTR loperand7 = &loperand_q[(3 * (2 * NQ + q) + 1) * VS];
    s_t *const RSTR loperand8 = &loperand_q[(3 * (2 * NQ + q) + 2) * VS];
    const s_t *const RSTR adj_q0 = adj0 + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adj1 + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adj2 + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adj3 + q * geometry_stride;
    const s_t *const RSTR adj_q4 = adj4 + q * geometry_stride;
    const s_t *const RSTR adj_q5 = adj5 + q * geometry_stride;
    const s_t *const RSTR adj_q6 = adj6 + q * geometry_stride;
    const s_t *const RSTR adj_q7 = adj7 + q * geometry_stride;
    const s_t *const RSTR adj_q8 = adj8 + q * geometry_stride;
    const s_t *const RSTR det_q0 = det0 + q * geometry_stride;
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t adj_lane0 = adj_q0[lane];
      const s_t adj_lane1 = adj_q1[lane];
      const s_t adj_lane2 = adj_q2[lane];
      const s_t adj_lane3 = adj_q3[lane];
      const s_t adj_lane4 = adj_q4[lane];
      const s_t adj_lane5 = adj_q5[lane];
      const s_t adj_lane6 = adj_q6[lane];
      const s_t adj_lane7 = adj_q7[lane];
      const s_t adj_lane8 = adj_q8[lane];
      const s_t det_lane0 = det_q0[lane];
      s_t gu[9];
      s_t trial_grad[9];
      const s_t idet = s_t(1) / det_lane0;
      gu[0] = (gu_ref0[lane] * adj_lane0 + gu_ref1[lane] * adj_lane3 + gu_ref2[lane] * adj_lane6) * idet;
      trial_grad[0] = (grad_h_ref0[lane] * adj_lane0 + grad_h_ref1[lane] * adj_lane3 + grad_h_ref2[lane] * adj_lane6) * idet;
      gu[1] = (gu_ref0[lane] * adj_lane1 + gu_ref1[lane] * adj_lane4 + gu_ref2[lane] * adj_lane7) * idet;
      trial_grad[1] = (grad_h_ref0[lane] * adj_lane1 + grad_h_ref1[lane] * adj_lane4 + grad_h_ref2[lane] * adj_lane7) * idet;
      gu[2] = (gu_ref0[lane] * adj_lane2 + gu_ref1[lane] * adj_lane5 + gu_ref2[lane] * adj_lane8) * idet;
      trial_grad[2] = (grad_h_ref0[lane] * adj_lane2 + grad_h_ref1[lane] * adj_lane5 + grad_h_ref2[lane] * adj_lane8) * idet;
      gu[3] = (gu_ref3[lane] * adj_lane0 + gu_ref4[lane] * adj_lane3 + gu_ref5[lane] * adj_lane6) * idet;
      trial_grad[3] = (grad_h_ref3[lane] * adj_lane0 + grad_h_ref4[lane] * adj_lane3 + grad_h_ref5[lane] * adj_lane6) * idet;
      gu[4] = (gu_ref3[lane] * adj_lane1 + gu_ref4[lane] * adj_lane4 + gu_ref5[lane] * adj_lane7) * idet;
      trial_grad[4] = (grad_h_ref3[lane] * adj_lane1 + grad_h_ref4[lane] * adj_lane4 + grad_h_ref5[lane] * adj_lane7) * idet;
      gu[5] = (gu_ref3[lane] * adj_lane2 + gu_ref4[lane] * adj_lane5 + gu_ref5[lane] * adj_lane8) * idet;
      trial_grad[5] = (grad_h_ref3[lane] * adj_lane2 + grad_h_ref4[lane] * adj_lane5 + grad_h_ref5[lane] * adj_lane8) * idet;
      gu[6] = (gu_ref6[lane] * adj_lane0 + gu_ref7[lane] * adj_lane3 + gu_ref8[lane] * adj_lane6) * idet;
      trial_grad[6] = (grad_h_ref6[lane] * adj_lane0 + grad_h_ref7[lane] * adj_lane3 + grad_h_ref8[lane] * adj_lane6) * idet;
      gu[7] = (gu_ref6[lane] * adj_lane1 + gu_ref7[lane] * adj_lane4 + gu_ref8[lane] * adj_lane7) * idet;
      trial_grad[7] = (grad_h_ref6[lane] * adj_lane1 + grad_h_ref7[lane] * adj_lane4 + grad_h_ref8[lane] * adj_lane7) * idet;
      gu[8] = (gu_ref6[lane] * adj_lane2 + gu_ref7[lane] * adj_lane5 + gu_ref8[lane] * adj_lane8) * idet;
      trial_grad[8] = (grad_h_ref6[lane] * adj_lane2 + grad_h_ref7[lane] * adj_lane5 + grad_h_ref8[lane] * adj_lane8) * idet;
      s_t loperand[9];
    s_t material[9];
    const s_t weak_mat_tmp0 = gu[5]*gu[7];
    const s_t weak_mat_tmp1 = gu[4] + s_t(1);
    const s_t weak_mat_tmp2 = gu[8] + s_t(1);
    const s_t weak_mat_tmp3 = weak_mat_tmp0 - weak_mat_tmp1*weak_mat_tmp2;
    const s_t weak_mat_tmp4 = -weak_mat_tmp3;
    const s_t weak_mat_tmp5 = gu[1]*gu[3];
    const s_t weak_mat_tmp6 = gu[2]*gu[6];
    const s_t weak_mat_tmp7 = gu[0] + s_t(1);
    const s_t weak_mat_tmp8 = gu[1]*gu[5]*gu[6] + gu[2]*gu[3]*gu[7] - weak_mat_tmp0*weak_mat_tmp7 + weak_mat_tmp1*weak_mat_tmp2*weak_mat_tmp7 - weak_mat_tmp1*weak_mat_tmp6 - weak_mat_tmp2*weak_mat_tmp5;
    const s_t weak_mat_tmp9 = kappa/pow_2(weak_mat_tmp8);
    const s_t weak_mat_tmp10 = gu[0]*gu[4];
    const s_t weak_mat_tmp11 = gu[5]*gu[6];
    const s_t weak_mat_tmp12 = gu[3]*gu[7];
    const s_t weak_mat_tmp13 = sfem_log1p(gu[0]*gu[8] - gu[0]*weak_mat_tmp0 + gu[0] + gu[1]*weak_mat_tmp11 + gu[2]*weak_mat_tmp12 + gu[4]*gu[8] - gu[4]*weak_mat_tmp6 + gu[4] + gu[8]*weak_mat_tmp10 - gu[8]*weak_mat_tmp5 + gu[8] - weak_mat_tmp0 + weak_mat_tmp10 - weak_mat_tmp5 - weak_mat_tmp6);
    const s_t weak_mat_tmp14 = weak_mat_tmp4*weak_mat_tmp9;
    const s_t weak_mat_tmp15 = weak_mat_tmp13*weak_mat_tmp14;
    const s_t weak_mat_tmp16 = s_t(2)/pow(weak_mat_tmp8, (s_t(2) / s_t(3)));
    const s_t weak_mat_tmp17 = pow(weak_mat_tmp8, (s_t(-5) / s_t(3)));
    const s_t weak_mat_tmp18 = weak_mat_tmp1*weak_mat_tmp2;
    const s_t weak_mat_tmp19 = ((s_t(2) / s_t(3)))*weak_mat_tmp0 - (s_t(2) / s_t(3))*weak_mat_tmp18;
    const s_t weak_mat_tmp20 = weak_mat_tmp17*weak_mat_tmp19;
    const s_t weak_mat_tmp21 = ((s_t(5) / s_t(3)))*weak_mat_tmp0 - (s_t(5) / s_t(3))*weak_mat_tmp18;
    const s_t weak_mat_tmp22 = pow_2(gu[3]);
    const s_t weak_mat_tmp23 = pow_2(gu[6]);
    const s_t weak_mat_tmp24 = pow_2(weak_mat_tmp7);
    const s_t weak_mat_tmp25 = weak_mat_tmp22 + weak_mat_tmp23 + weak_mat_tmp24;
    const s_t weak_mat_tmp26 = pow_2(gu[1]);
    const s_t weak_mat_tmp27 = pow_2(gu[7]);
    const s_t weak_mat_tmp28 = pow_2(weak_mat_tmp1);
    const s_t weak_mat_tmp29 = weak_mat_tmp26 + weak_mat_tmp27 + weak_mat_tmp28;
    const s_t weak_mat_tmp30 = pow_2(gu[2]);
    const s_t weak_mat_tmp31 = pow_2(gu[5]);
    const s_t weak_mat_tmp32 = pow_2(weak_mat_tmp2);
    const s_t weak_mat_tmp33 = weak_mat_tmp30 + weak_mat_tmp31 + weak_mat_tmp32;
    const s_t weak_mat_tmp34 = weak_mat_tmp25 + weak_mat_tmp29 + weak_mat_tmp33;
    const s_t weak_mat_tmp35 = weak_mat_tmp34/pow(weak_mat_tmp8, (s_t(8) / s_t(3)));
    const s_t weak_mat_tmp36 = weak_mat_tmp19*weak_mat_tmp35;
    const s_t weak_mat_tmp37 = s_t(2)*weak_mat_tmp31;
    const s_t weak_mat_tmp38 = s_t(2)*weak_mat_tmp32;
    const s_t weak_mat_tmp39 = weak_mat_tmp37 + weak_mat_tmp38;
    const s_t weak_mat_tmp40 = s_t(2)*weak_mat_tmp27;
    const s_t weak_mat_tmp41 = s_t(2)*weak_mat_tmp28;
    const s_t weak_mat_tmp42 = weak_mat_tmp40 + weak_mat_tmp41;
    const s_t weak_mat_tmp43 = pow(weak_mat_tmp8, (s_t(-4) / s_t(3)));
    const s_t weak_mat_tmp44 = ((s_t(4) / s_t(3)))*weak_mat_tmp0 - (s_t(4) / s_t(3))*weak_mat_tmp18;
    const s_t weak_mat_tmp45 = pow(weak_mat_tmp8, (s_t(-7) / s_t(3)));
    const s_t weak_mat_tmp46 = gu[6]*gu[7];
    const s_t weak_mat_tmp47 = gu[1]*weak_mat_tmp7;
    const s_t weak_mat_tmp48 = gu[3]*weak_mat_tmp1;
    const s_t weak_mat_tmp49 = weak_mat_tmp46 + weak_mat_tmp47 + weak_mat_tmp48;
    const s_t weak_mat_tmp50 = s_t(2)*gu[1];
    const s_t weak_mat_tmp51 = gu[3]*gu[5];
    const s_t weak_mat_tmp52 = gu[2]*weak_mat_tmp7;
    const s_t weak_mat_tmp53 = gu[6]*weak_mat_tmp2;
    const s_t weak_mat_tmp54 = weak_mat_tmp51 + weak_mat_tmp52 + weak_mat_tmp53;
    const s_t weak_mat_tmp55 = s_t(2)*gu[2];
    const s_t weak_mat_tmp56 = s_t(2)*weak_mat_tmp7;
    const s_t weak_mat_tmp57 = weak_mat_tmp45*(-weak_mat_tmp25*weak_mat_tmp56 + s_t(2)*weak_mat_tmp34*weak_mat_tmp7 - weak_mat_tmp49*weak_mat_tmp50 - weak_mat_tmp54*weak_mat_tmp55);
    const s_t weak_mat_tmp58 = ((s_t(7) / s_t(3)))*weak_mat_tmp0 - (s_t(7) / s_t(3))*weak_mat_tmp18;
    const s_t weak_mat_tmp59 = gu[1]*gu[2];
    const s_t weak_mat_tmp60 = gu[5]*weak_mat_tmp1;
    const s_t weak_mat_tmp61 = gu[7]*weak_mat_tmp2;
    const s_t weak_mat_tmp62 = weak_mat_tmp59 + weak_mat_tmp60 + weak_mat_tmp61;
    const s_t weak_mat_tmp63 = -(s_t(1) / s_t(2))*pow_2(weak_mat_tmp25) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp29) - (s_t(1) / s_t(2))*pow_2(weak_mat_tmp33) + ((s_t(1) / s_t(2)))*pow_2(weak_mat_tmp34) - pow_2(weak_mat_tmp49) - pow_2(weak_mat_tmp54) - pow_2(weak_mat_tmp62);
    const s_t weak_mat_tmp64 = weak_mat_tmp63/pow(weak_mat_tmp8, (s_t(10) / s_t(3)));
    const s_t weak_mat_tmp65 = weak_mat_tmp44*weak_mat_tmp64;
    const s_t weak_mat_tmp66 = gu[3]*weak_mat_tmp2;
    const s_t weak_mat_tmp67 = -gu[5]*gu[6] + weak_mat_tmp66;
    const s_t weak_mat_tmp68 = -weak_mat_tmp67;
    const s_t weak_mat_tmp69 = weak_mat_tmp14*weak_mat_tmp68;
    const s_t weak_mat_tmp70 = -(s_t(5) / s_t(3))*weak_mat_tmp11 + ((s_t(5) / s_t(3)))*weak_mat_tmp66;
    const s_t weak_mat_tmp71 = -(s_t(2) / s_t(3))*weak_mat_tmp11 + ((s_t(2) / s_t(3)))*weak_mat_tmp66;
    const s_t weak_mat_tmp72 = weak_mat_tmp17*weak_mat_tmp56;
    const s_t weak_mat_tmp73 = weak_mat_tmp20*weak_mat_tmp50 + weak_mat_tmp71*weak_mat_tmp72;
    const s_t weak_mat_tmp74 = -(s_t(7) / s_t(3))*weak_mat_tmp11 + ((s_t(7) / s_t(3)))*weak_mat_tmp66;
    const s_t weak_mat_tmp75 = s_t(2)*weak_mat_tmp46;
    const s_t weak_mat_tmp76 = s_t(2)*weak_mat_tmp48;
    const s_t weak_mat_tmp77 = -(s_t(4) / s_t(3))*weak_mat_tmp11 + ((s_t(4) / s_t(3)))*weak_mat_tmp66;
    const s_t weak_mat_tmp78 = s_t(2)*gu[1]*weak_mat_tmp34 - weak_mat_tmp29*weak_mat_tmp50 - weak_mat_tmp49*weak_mat_tmp56 - weak_mat_tmp55*weak_mat_tmp62;
    const s_t weak_mat_tmp79 = weak_mat_tmp44*weak_mat_tmp45;
    const s_t weak_mat_tmp80 = weak_mat_tmp43*(-weak_mat_tmp75 - weak_mat_tmp76) + weak_mat_tmp57*weak_mat_tmp77 + weak_mat_tmp78*weak_mat_tmp79;
    const s_t weak_mat_tmp81 = gu[6]*weak_mat_tmp1;
    const s_t weak_mat_tmp82 = weak_mat_tmp12 - weak_mat_tmp81;
    const s_t weak_mat_tmp83 = weak_mat_tmp14*weak_mat_tmp82;
    const s_t weak_mat_tmp84 = -weak_mat_tmp82;
    const s_t weak_mat_tmp85 = ((s_t(5) / s_t(3)))*gu[6]*weak_mat_tmp1 - (s_t(5) / s_t(3))*weak_mat_tmp12;
    const s_t weak_mat_tmp86 = ((s_t(2) / s_t(3)))*gu[6]*weak_mat_tmp1 - (s_t(2) / s_t(3))*weak_mat_tmp12;
    const s_t weak_mat_tmp87 = weak_mat_tmp20*weak_mat_tmp55 + weak_mat_tmp72*weak_mat_tmp86;
    const s_t weak_mat_tmp88 = ((s_t(7) / s_t(3)))*gu[6]*weak_mat_tmp1 - (s_t(7) / s_t(3))*weak_mat_tmp12;
    const s_t weak_mat_tmp89 = s_t(2)*weak_mat_tmp51;
    const s_t weak_mat_tmp90 = s_t(2)*weak_mat_tmp53;
    const s_t weak_mat_tmp91 = ((s_t(4) / s_t(3)))*gu[6]*weak_mat_tmp1 - (s_t(4) / s_t(3))*weak_mat_tmp12;
    const s_t weak_mat_tmp92 = s_t(2)*gu[2]*weak_mat_tmp34 - weak_mat_tmp33*weak_mat_tmp55 - weak_mat_tmp50*weak_mat_tmp62 - weak_mat_tmp54*weak_mat_tmp56;
    const s_t weak_mat_tmp93 = weak_mat_tmp43*(-weak_mat_tmp89 - weak_mat_tmp90) + weak_mat_tmp57*weak_mat_tmp91 + weak_mat_tmp79*weak_mat_tmp92;
    const s_t weak_mat_tmp94 = gu[1]*weak_mat_tmp2;
    const s_t weak_mat_tmp95 = -gu[2]*gu[7] + weak_mat_tmp94;
    const s_t weak_mat_tmp96 = -weak_mat_tmp95;
    const s_t weak_mat_tmp97 = weak_mat_tmp14*weak_mat_tmp96;
    const s_t weak_mat_tmp98 = gu[2]*gu[7];
    const s_t weak_mat_tmp99 = ((s_t(5) / s_t(3)))*weak_mat_tmp94 - (s_t(5) / s_t(3))*weak_mat_tmp98;
    const s_t weak_mat_tmp100 = s_t(2)*gu[3];
    const s_t weak_mat_tmp101 = ((s_t(2) / s_t(3)))*weak_mat_tmp94 - (s_t(2) / s_t(3))*weak_mat_tmp98;
    const s_t weak_mat_tmp102 = weak_mat_tmp100*weak_mat_tmp20 + weak_mat_tmp101*weak_mat_tmp72;
    const s_t weak_mat_tmp103 = ((s_t(7) / s_t(3)))*weak_mat_tmp94 - (s_t(7) / s_t(3))*weak_mat_tmp98;
    const s_t weak_mat_tmp104 = gu[5]*weak_mat_tmp55;
    const s_t weak_mat_tmp105 = weak_mat_tmp1*weak_mat_tmp50;
    const s_t weak_mat_tmp106 = ((s_t(4) / s_t(3)))*weak_mat_tmp94 - (s_t(4) / s_t(3))*weak_mat_tmp98;
    const s_t weak_mat_tmp107 = s_t(2)*gu[5];
    const s_t weak_mat_tmp108 = s_t(2)*weak_mat_tmp1;
    const s_t weak_mat_tmp109 = s_t(2)*gu[3]*weak_mat_tmp34 - weak_mat_tmp100*weak_mat_tmp25 - weak_mat_tmp107*weak_mat_tmp54 - weak_mat_tmp108*weak_mat_tmp49;
    const s_t weak_mat_tmp110 = weak_mat_tmp106*weak_mat_tmp57 + weak_mat_tmp109*weak_mat_tmp79 + weak_mat_tmp43*(-weak_mat_tmp104 - weak_mat_tmp105);
    const s_t weak_mat_tmp111 = gu[1]*gu[5];
    const s_t weak_mat_tmp112 = gu[2]*weak_mat_tmp1;
    const s_t weak_mat_tmp113 = weak_mat_tmp111 - weak_mat_tmp112;
    const s_t weak_mat_tmp114 = weak_mat_tmp113*weak_mat_tmp14;
    const s_t weak_mat_tmp115 = -weak_mat_tmp113;
    const s_t weak_mat_tmp116 = ((s_t(5) / s_t(3)))*gu[2]*weak_mat_tmp1 - (s_t(5) / s_t(3))*weak_mat_tmp111;
    const s_t weak_mat_tmp117 = s_t(2)*gu[6];
    const s_t weak_mat_tmp118 = ((s_t(2) / s_t(3)))*gu[2]*weak_mat_tmp1 - (s_t(2) / s_t(3))*weak_mat_tmp111;
    const s_t weak_mat_tmp119 = weak_mat_tmp117*weak_mat_tmp20 + weak_mat_tmp118*weak_mat_tmp72;
    const s_t weak_mat_tmp120 = ((s_t(7) / s_t(3)))*gu[2]*weak_mat_tmp1 - (s_t(7) / s_t(3))*weak_mat_tmp111;
    const s_t weak_mat_tmp121 = gu[7]*weak_mat_tmp50;
    const s_t weak_mat_tmp122 = weak_mat_tmp2*weak_mat_tmp55;
    const s_t weak_mat_tmp123 = ((s_t(4) / s_t(3)))*gu[2]*weak_mat_tmp1 - (s_t(4) / s_t(3))*weak_mat_tmp111;
    const s_t weak_mat_tmp124 = s_t(2)*gu[7];
    const s_t weak_mat_tmp125 = s_t(2)*weak_mat_tmp2;
    const s_t weak_mat_tmp126 = s_t(2)*gu[6]*weak_mat_tmp34 - weak_mat_tmp117*weak_mat_tmp25 - weak_mat_tmp124*weak_mat_tmp49 - weak_mat_tmp125*weak_mat_tmp54;
    const s_t weak_mat_tmp127 = weak_mat_tmp123*weak_mat_tmp57 + weak_mat_tmp126*weak_mat_tmp79 + weak_mat_tmp43*(-weak_mat_tmp121 - weak_mat_tmp122);
    const s_t weak_mat_tmp128 = gu[1]*gu[6];
    const s_t weak_mat_tmp129 = ((s_t(5) / s_t(3)))*gu[7]*weak_mat_tmp7 - (s_t(5) / s_t(3))*weak_mat_tmp128;
    const s_t weak_mat_tmp130 = ((s_t(2) / s_t(3)))*gu[7]*weak_mat_tmp7 - (s_t(2) / s_t(3))*weak_mat_tmp128;
    const s_t weak_mat_tmp131 = ((s_t(2) / s_t(3)))*weak_mat_tmp34;
    const s_t weak_mat_tmp132 = weak_mat_tmp131*weak_mat_tmp17;
    const s_t weak_mat_tmp133 = gu[7]*weak_mat_tmp132;
    const s_t weak_mat_tmp134 = weak_mat_tmp107*weak_mat_tmp20 + weak_mat_tmp130*weak_mat_tmp72 + weak_mat_tmp133;
    const s_t weak_mat_tmp135 = ((s_t(7) / s_t(3)))*gu[7]*weak_mat_tmp7 - (s_t(7) / s_t(3))*weak_mat_tmp128;
    const s_t weak_mat_tmp136 = gu[2]*gu[3];
    const s_t weak_mat_tmp137 = ((s_t(4) / s_t(3)))*gu[7]*weak_mat_tmp7 - (s_t(4) / s_t(3))*weak_mat_tmp128;
    const s_t weak_mat_tmp138 = s_t(2)*gu[5]*weak_mat_tmp34 - weak_mat_tmp100*weak_mat_tmp54 - weak_mat_tmp107*weak_mat_tmp33 - weak_mat_tmp108*weak_mat_tmp62;
    const s_t weak_mat_tmp139 = ((s_t(4) / s_t(3)))*weak_mat_tmp45*weak_mat_tmp63;
    const s_t weak_mat_tmp140 = gu[7]*weak_mat_tmp139;
    const s_t weak_mat_tmp141 = weak_mat_tmp137*weak_mat_tmp57 + weak_mat_tmp138*weak_mat_tmp79 + weak_mat_tmp140 + weak_mat_tmp43*(s_t(4)*gu[5]*weak_mat_tmp7 - s_t(2)*weak_mat_tmp136);
    const s_t weak_mat_tmp142 = gu[7]*weak_mat_tmp7;
    const s_t weak_mat_tmp143 = weak_mat_tmp128 - weak_mat_tmp142;
    const s_t weak_mat_tmp144 = -weak_mat_tmp143;
    const s_t weak_mat_tmp145 = kappa*weak_mat_tmp13/weak_mat_tmp8;
    const s_t weak_mat_tmp146 = gu[7]*weak_mat_tmp145;
    const s_t weak_mat_tmp147 = weak_mat_tmp14*weak_mat_tmp143 - weak_mat_tmp146;
    const s_t weak_mat_tmp148 = ((s_t(5) / s_t(3)))*gu[5]*weak_mat_tmp7 - (s_t(5) / s_t(3))*weak_mat_tmp136;
    const s_t weak_mat_tmp149 = ((s_t(2) / s_t(3)))*gu[5]*weak_mat_tmp7 - (s_t(2) / s_t(3))*weak_mat_tmp136;
    const s_t weak_mat_tmp150 = gu[5]*weak_mat_tmp132;
    const s_t weak_mat_tmp151 = weak_mat_tmp124*weak_mat_tmp20 + weak_mat_tmp149*weak_mat_tmp72 + weak_mat_tmp150;
    const s_t weak_mat_tmp152 = ((s_t(7) / s_t(3)))*gu[5]*weak_mat_tmp7 - (s_t(7) / s_t(3))*weak_mat_tmp136;
    const s_t weak_mat_tmp153 = ((s_t(4) / s_t(3)))*gu[5]*weak_mat_tmp7 - (s_t(4) / s_t(3))*weak_mat_tmp136;
    const s_t weak_mat_tmp154 = s_t(2)*gu[7]*weak_mat_tmp34 - weak_mat_tmp117*weak_mat_tmp49 - weak_mat_tmp124*weak_mat_tmp29 - weak_mat_tmp125*weak_mat_tmp62;
    const s_t weak_mat_tmp155 = gu[5]*weak_mat_tmp139;
    const s_t weak_mat_tmp156 = weak_mat_tmp153*weak_mat_tmp57 + weak_mat_tmp154*weak_mat_tmp79 + weak_mat_tmp155 + weak_mat_tmp43*(s_t(4)*gu[7]*weak_mat_tmp7 - s_t(2)*weak_mat_tmp128);
    const s_t weak_mat_tmp157 = gu[5]*weak_mat_tmp7;
    const s_t weak_mat_tmp158 = weak_mat_tmp136 - weak_mat_tmp157;
    const s_t weak_mat_tmp159 = -weak_mat_tmp158;
    const s_t weak_mat_tmp160 = gu[5]*weak_mat_tmp145;
    const s_t weak_mat_tmp161 = weak_mat_tmp14*weak_mat_tmp158 - weak_mat_tmp160;
    const s_t weak_mat_tmp162 = weak_mat_tmp2*weak_mat_tmp7;
    const s_t weak_mat_tmp163 = -(s_t(5) / s_t(3))*weak_mat_tmp162 + ((s_t(5) / s_t(3)))*weak_mat_tmp6;
    const s_t weak_mat_tmp164 = -(s_t(2) / s_t(3))*weak_mat_tmp162 + ((s_t(2) / s_t(3)))*weak_mat_tmp6;
    const s_t weak_mat_tmp165 = weak_mat_tmp17*weak_mat_tmp2;
    const s_t weak_mat_tmp166 = weak_mat_tmp131*weak_mat_tmp165;
    const s_t weak_mat_tmp167 = weak_mat_tmp108*weak_mat_tmp20 + weak_mat_tmp164*weak_mat_tmp72 - weak_mat_tmp166;
    const s_t weak_mat_tmp168 = -(s_t(7) / s_t(3))*weak_mat_tmp162 + ((s_t(7) / s_t(3)))*weak_mat_tmp6;
    const s_t weak_mat_tmp169 = -(s_t(4) / s_t(3))*weak_mat_tmp162 + ((s_t(4) / s_t(3)))*weak_mat_tmp6;
    const s_t weak_mat_tmp170 = s_t(2)*weak_mat_tmp1*weak_mat_tmp34 - weak_mat_tmp100*weak_mat_tmp49 - weak_mat_tmp107*weak_mat_tmp62 - weak_mat_tmp108*weak_mat_tmp29;
    const s_t weak_mat_tmp171 = weak_mat_tmp139*weak_mat_tmp2;
    const s_t weak_mat_tmp172 = weak_mat_tmp169*weak_mat_tmp57 + weak_mat_tmp170*weak_mat_tmp79 - weak_mat_tmp171 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp1*weak_mat_tmp7 - s_t(2)*weak_mat_tmp5);
    const s_t weak_mat_tmp173 = -weak_mat_tmp2*weak_mat_tmp7 + weak_mat_tmp6;
    const s_t weak_mat_tmp174 = weak_mat_tmp145*weak_mat_tmp2;
    const s_t weak_mat_tmp175 = -weak_mat_tmp173;
    const s_t weak_mat_tmp176 = weak_mat_tmp14*weak_mat_tmp175 + weak_mat_tmp174;
    const s_t weak_mat_tmp177 = weak_mat_tmp1*weak_mat_tmp7;
    const s_t weak_mat_tmp178 = -(s_t(5) / s_t(3))*weak_mat_tmp177 + ((s_t(5) / s_t(3)))*weak_mat_tmp5;
    const s_t weak_mat_tmp179 = -(s_t(2) / s_t(3))*weak_mat_tmp177 + ((s_t(2) / s_t(3)))*weak_mat_tmp5;
    const s_t weak_mat_tmp180 = weak_mat_tmp1*weak_mat_tmp132;
    const s_t weak_mat_tmp181 = weak_mat_tmp125*weak_mat_tmp20 + weak_mat_tmp179*weak_mat_tmp72 - weak_mat_tmp180;
    const s_t weak_mat_tmp182 = -(s_t(7) / s_t(3))*weak_mat_tmp177 + ((s_t(7) / s_t(3)))*weak_mat_tmp5;
    const s_t weak_mat_tmp183 = -(s_t(4) / s_t(3))*weak_mat_tmp177 + ((s_t(4) / s_t(3)))*weak_mat_tmp5;
    const s_t weak_mat_tmp184 = -weak_mat_tmp117*weak_mat_tmp54 - weak_mat_tmp124*weak_mat_tmp62 - weak_mat_tmp125*weak_mat_tmp33 + s_t(2)*weak_mat_tmp2*weak_mat_tmp34;
    const s_t weak_mat_tmp185 = weak_mat_tmp1*weak_mat_tmp139;
    const s_t weak_mat_tmp186 = weak_mat_tmp183*weak_mat_tmp57 + weak_mat_tmp184*weak_mat_tmp79 - weak_mat_tmp185 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp2*weak_mat_tmp7 - s_t(2)*weak_mat_tmp6);
    const s_t weak_mat_tmp187 = -weak_mat_tmp1*weak_mat_tmp7 + weak_mat_tmp5;
    const s_t weak_mat_tmp188 = weak_mat_tmp1*weak_mat_tmp145;
    const s_t weak_mat_tmp189 = -weak_mat_tmp187;
    const s_t weak_mat_tmp190 = weak_mat_tmp14*weak_mat_tmp189 + weak_mat_tmp188;
    const s_t weak_mat_tmp191 = weak_mat_tmp68*weak_mat_tmp9;
    const s_t weak_mat_tmp192 = weak_mat_tmp13*weak_mat_tmp191;
    const s_t weak_mat_tmp193 = weak_mat_tmp17*weak_mat_tmp71;
    const s_t weak_mat_tmp194 = weak_mat_tmp35*weak_mat_tmp71;
    const s_t weak_mat_tmp195 = s_t(2)*weak_mat_tmp22;
    const s_t weak_mat_tmp196 = s_t(2)*weak_mat_tmp23;
    const s_t weak_mat_tmp197 = weak_mat_tmp195 + weak_mat_tmp196;
    const s_t weak_mat_tmp198 = weak_mat_tmp45*weak_mat_tmp78;
    const s_t weak_mat_tmp199 = weak_mat_tmp64*weak_mat_tmp77;
    const s_t weak_mat_tmp200 = weak_mat_tmp191*weak_mat_tmp82;
    const s_t weak_mat_tmp201 = weak_mat_tmp17*weak_mat_tmp50;
    const s_t weak_mat_tmp202 = weak_mat_tmp193*weak_mat_tmp55 + weak_mat_tmp201*weak_mat_tmp86;
    const s_t weak_mat_tmp203 = s_t(2)*weak_mat_tmp60;
    const s_t weak_mat_tmp204 = s_t(2)*weak_mat_tmp61;
    const s_t weak_mat_tmp205 = weak_mat_tmp45*weak_mat_tmp77;
    const s_t weak_mat_tmp206 = weak_mat_tmp198*weak_mat_tmp91 + weak_mat_tmp205*weak_mat_tmp92 + weak_mat_tmp43*(-weak_mat_tmp203 - weak_mat_tmp204);
    const s_t weak_mat_tmp207 = weak_mat_tmp158*weak_mat_tmp191;
    const s_t weak_mat_tmp208 = weak_mat_tmp124*weak_mat_tmp193 + weak_mat_tmp149*weak_mat_tmp201;
    const s_t weak_mat_tmp209 = gu[6]*weak_mat_tmp56;
    const s_t weak_mat_tmp210 = weak_mat_tmp153*weak_mat_tmp198 + weak_mat_tmp154*weak_mat_tmp205 + weak_mat_tmp43*(-weak_mat_tmp122 - weak_mat_tmp209);
    const s_t weak_mat_tmp211 = weak_mat_tmp175*weak_mat_tmp191;
    const s_t weak_mat_tmp212 = weak_mat_tmp108*weak_mat_tmp193 + weak_mat_tmp164*weak_mat_tmp201;
    const s_t weak_mat_tmp213 = gu[3]*weak_mat_tmp56;
    const s_t weak_mat_tmp214 = weak_mat_tmp169*weak_mat_tmp198 + weak_mat_tmp170*weak_mat_tmp205 + weak_mat_tmp43*(-weak_mat_tmp104 - weak_mat_tmp213);
    const s_t weak_mat_tmp215 = gu[6]*weak_mat_tmp132;
    const s_t weak_mat_tmp216 = weak_mat_tmp107*weak_mat_tmp193 + weak_mat_tmp130*weak_mat_tmp201 - weak_mat_tmp215;
    const s_t weak_mat_tmp217 = gu[6]*weak_mat_tmp139;
    const s_t weak_mat_tmp218 = weak_mat_tmp137*weak_mat_tmp198 + weak_mat_tmp138*weak_mat_tmp205 - weak_mat_tmp217 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp111 - s_t(2)*weak_mat_tmp112);
    const s_t weak_mat_tmp219 = gu[6]*weak_mat_tmp145;
    const s_t weak_mat_tmp220 = weak_mat_tmp143*weak_mat_tmp191 + weak_mat_tmp219;
    const s_t weak_mat_tmp221 = weak_mat_tmp117*weak_mat_tmp193 + weak_mat_tmp118*weak_mat_tmp201 - weak_mat_tmp150;
    const s_t weak_mat_tmp222 = weak_mat_tmp123*weak_mat_tmp198 + weak_mat_tmp126*weak_mat_tmp205 - weak_mat_tmp155 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp128 - s_t(2)*weak_mat_tmp142);
    const s_t weak_mat_tmp223 = weak_mat_tmp113*weak_mat_tmp191 + weak_mat_tmp160;
    const s_t weak_mat_tmp224 = weak_mat_tmp100*weak_mat_tmp193 + weak_mat_tmp101*weak_mat_tmp201 + weak_mat_tmp166;
    const s_t weak_mat_tmp225 = weak_mat_tmp106*weak_mat_tmp198 + weak_mat_tmp109*weak_mat_tmp205 + weak_mat_tmp171 + weak_mat_tmp43*(-s_t(2)*weak_mat_tmp177 + s_t(4)*weak_mat_tmp5);
    const s_t weak_mat_tmp226 = -weak_mat_tmp174 + weak_mat_tmp191*weak_mat_tmp96;
    const s_t weak_mat_tmp227 = gu[3]*weak_mat_tmp132;
    const s_t weak_mat_tmp228 = weak_mat_tmp125*weak_mat_tmp193 + weak_mat_tmp179*weak_mat_tmp201 + weak_mat_tmp227;
    const s_t weak_mat_tmp229 = gu[3]*weak_mat_tmp139;
    const s_t weak_mat_tmp230 = weak_mat_tmp183*weak_mat_tmp198 + weak_mat_tmp184*weak_mat_tmp205 + weak_mat_tmp229 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp94 - s_t(2)*weak_mat_tmp98);
    const s_t weak_mat_tmp231 = gu[3]*weak_mat_tmp145;
    const s_t weak_mat_tmp232 = weak_mat_tmp189*weak_mat_tmp191 - weak_mat_tmp231;
    const s_t weak_mat_tmp233 = weak_mat_tmp82*weak_mat_tmp9;
    const s_t weak_mat_tmp234 = weak_mat_tmp13*weak_mat_tmp233;
    const s_t weak_mat_tmp235 = weak_mat_tmp17*weak_mat_tmp86;
    const s_t weak_mat_tmp236 = weak_mat_tmp35*weak_mat_tmp86;
    const s_t weak_mat_tmp237 = weak_mat_tmp45*weak_mat_tmp92;
    const s_t weak_mat_tmp238 = weak_mat_tmp64*weak_mat_tmp91;
    const s_t weak_mat_tmp239 = weak_mat_tmp143*weak_mat_tmp233;
    const s_t weak_mat_tmp240 = weak_mat_tmp17*weak_mat_tmp55;
    const s_t weak_mat_tmp241 = weak_mat_tmp107*weak_mat_tmp235 + weak_mat_tmp130*weak_mat_tmp240;
    const s_t weak_mat_tmp242 = weak_mat_tmp45*weak_mat_tmp91;
    const s_t weak_mat_tmp243 = weak_mat_tmp137*weak_mat_tmp237 + weak_mat_tmp138*weak_mat_tmp242 + weak_mat_tmp43*(-weak_mat_tmp105 - weak_mat_tmp213);
    const s_t weak_mat_tmp244 = weak_mat_tmp189*weak_mat_tmp233;
    const s_t weak_mat_tmp245 = weak_mat_tmp125*weak_mat_tmp235 + weak_mat_tmp179*weak_mat_tmp240;
    const s_t weak_mat_tmp246 = weak_mat_tmp183*weak_mat_tmp237 + weak_mat_tmp184*weak_mat_tmp242 + weak_mat_tmp43*(-weak_mat_tmp121 - weak_mat_tmp209);
    const s_t weak_mat_tmp247 = weak_mat_tmp100*weak_mat_tmp235 + weak_mat_tmp101*weak_mat_tmp240 - weak_mat_tmp133;
    const s_t weak_mat_tmp248 = weak_mat_tmp106*weak_mat_tmp237 + weak_mat_tmp109*weak_mat_tmp242 - weak_mat_tmp140 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp136 - s_t(2)*weak_mat_tmp157);
    const s_t weak_mat_tmp249 = weak_mat_tmp146 + weak_mat_tmp233*weak_mat_tmp96;
    const s_t weak_mat_tmp250 = weak_mat_tmp124*weak_mat_tmp235 + weak_mat_tmp149*weak_mat_tmp240 - weak_mat_tmp227;
    const s_t weak_mat_tmp251 = weak_mat_tmp153*weak_mat_tmp237 + weak_mat_tmp154*weak_mat_tmp242 - weak_mat_tmp229 + weak_mat_tmp43*(s_t(4)*gu[2]*gu[7] - s_t(2)*weak_mat_tmp94);
    const s_t weak_mat_tmp252 = weak_mat_tmp158*weak_mat_tmp233 + weak_mat_tmp231;
    const s_t weak_mat_tmp253 = weak_mat_tmp117*weak_mat_tmp235 + weak_mat_tmp118*weak_mat_tmp240 + weak_mat_tmp180;
    const s_t weak_mat_tmp254 = weak_mat_tmp123*weak_mat_tmp237 + weak_mat_tmp126*weak_mat_tmp242 + weak_mat_tmp185 + weak_mat_tmp43*(-s_t(2)*weak_mat_tmp162 + s_t(4)*weak_mat_tmp6);
    const s_t weak_mat_tmp255 = weak_mat_tmp113*weak_mat_tmp233 - weak_mat_tmp188;
    const s_t weak_mat_tmp256 = weak_mat_tmp108*weak_mat_tmp235 + weak_mat_tmp164*weak_mat_tmp240 + weak_mat_tmp215;
    const s_t weak_mat_tmp257 = weak_mat_tmp169*weak_mat_tmp237 + weak_mat_tmp170*weak_mat_tmp242 + weak_mat_tmp217 + weak_mat_tmp43*(s_t(4)*gu[2]*weak_mat_tmp1 - s_t(2)*weak_mat_tmp111);
    const s_t weak_mat_tmp258 = weak_mat_tmp175*weak_mat_tmp233 - weak_mat_tmp219;
    const s_t weak_mat_tmp259 = weak_mat_tmp9*weak_mat_tmp96;
    const s_t weak_mat_tmp260 = weak_mat_tmp13*weak_mat_tmp259;
    const s_t weak_mat_tmp261 = weak_mat_tmp101*weak_mat_tmp17;
    const s_t weak_mat_tmp262 = weak_mat_tmp101*weak_mat_tmp35;
    const s_t weak_mat_tmp263 = s_t(2)*weak_mat_tmp30;
    const s_t weak_mat_tmp264 = weak_mat_tmp263 + weak_mat_tmp38;
    const s_t weak_mat_tmp265 = s_t(2)*weak_mat_tmp26;
    const s_t weak_mat_tmp266 = weak_mat_tmp265 + weak_mat_tmp40;
    const s_t weak_mat_tmp267 = weak_mat_tmp106*weak_mat_tmp45;
    const s_t weak_mat_tmp268 = weak_mat_tmp106*weak_mat_tmp64;
    const s_t weak_mat_tmp269 = weak_mat_tmp143*weak_mat_tmp259;
    const s_t weak_mat_tmp270 = weak_mat_tmp100*weak_mat_tmp17;
    const s_t weak_mat_tmp271 = weak_mat_tmp107*weak_mat_tmp261 + weak_mat_tmp130*weak_mat_tmp270;
    const s_t weak_mat_tmp272 = s_t(2)*weak_mat_tmp52;
    const s_t weak_mat_tmp273 = weak_mat_tmp109*weak_mat_tmp45;
    const s_t weak_mat_tmp274 = weak_mat_tmp137*weak_mat_tmp273 + weak_mat_tmp138*weak_mat_tmp267 + weak_mat_tmp43*(-weak_mat_tmp272 - weak_mat_tmp90);
    const s_t weak_mat_tmp275 = weak_mat_tmp113*weak_mat_tmp259;
    const s_t weak_mat_tmp276 = weak_mat_tmp117*weak_mat_tmp261 + weak_mat_tmp118*weak_mat_tmp270;
    const s_t weak_mat_tmp277 = weak_mat_tmp107*weak_mat_tmp2;
    const s_t weak_mat_tmp278 = gu[7]*weak_mat_tmp108;
    const s_t weak_mat_tmp279 = weak_mat_tmp123*weak_mat_tmp273 + weak_mat_tmp126*weak_mat_tmp267 + weak_mat_tmp43*(-weak_mat_tmp277 - weak_mat_tmp278);
    const s_t weak_mat_tmp280 = weak_mat_tmp175*weak_mat_tmp259;
    const s_t weak_mat_tmp281 = weak_mat_tmp108*weak_mat_tmp261 + weak_mat_tmp164*weak_mat_tmp270;
    const s_t weak_mat_tmp282 = s_t(2)*weak_mat_tmp47;
    const s_t weak_mat_tmp283 = weak_mat_tmp169*weak_mat_tmp273 + weak_mat_tmp170*weak_mat_tmp267 + weak_mat_tmp43*(-weak_mat_tmp282 - weak_mat_tmp75);
    const s_t weak_mat_tmp284 = gu[2]*weak_mat_tmp132;
    const s_t weak_mat_tmp285 = weak_mat_tmp124*weak_mat_tmp261 + weak_mat_tmp149*weak_mat_tmp270 - weak_mat_tmp284;
    const s_t weak_mat_tmp286 = gu[2]*weak_mat_tmp139;
    const s_t weak_mat_tmp287 = weak_mat_tmp153*weak_mat_tmp273 + weak_mat_tmp154*weak_mat_tmp267 - weak_mat_tmp286 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp12 - s_t(2)*weak_mat_tmp81);
    const s_t weak_mat_tmp288 = gu[2]*weak_mat_tmp145;
    const s_t weak_mat_tmp289 = weak_mat_tmp158*weak_mat_tmp259 + weak_mat_tmp288;
    const s_t weak_mat_tmp290 = gu[1]*weak_mat_tmp132;
    const s_t weak_mat_tmp291 = weak_mat_tmp125*weak_mat_tmp261 + weak_mat_tmp179*weak_mat_tmp270 + weak_mat_tmp290;
    const s_t weak_mat_tmp292 = gu[1]*weak_mat_tmp139;
    const s_t weak_mat_tmp293 = weak_mat_tmp183*weak_mat_tmp273 + weak_mat_tmp184*weak_mat_tmp267 + weak_mat_tmp292 + weak_mat_tmp43*(-s_t(2)*weak_mat_tmp11 + s_t(4)*weak_mat_tmp66);
    const s_t weak_mat_tmp294 = gu[1]*weak_mat_tmp145;
    const s_t weak_mat_tmp295 = weak_mat_tmp189*weak_mat_tmp259 - weak_mat_tmp294;
    const s_t weak_mat_tmp296 = weak_mat_tmp175*weak_mat_tmp9;
    const s_t weak_mat_tmp297 = weak_mat_tmp13*weak_mat_tmp296;
    const s_t weak_mat_tmp298 = weak_mat_tmp164*weak_mat_tmp17;
    const s_t weak_mat_tmp299 = weak_mat_tmp164*weak_mat_tmp35;
    const s_t weak_mat_tmp300 = s_t(2)*weak_mat_tmp24;
    const s_t weak_mat_tmp301 = weak_mat_tmp196 + weak_mat_tmp300;
    const s_t weak_mat_tmp302 = weak_mat_tmp170*weak_mat_tmp45;
    const s_t weak_mat_tmp303 = weak_mat_tmp169*weak_mat_tmp64;
    const s_t weak_mat_tmp304 = weak_mat_tmp143*weak_mat_tmp296;
    const s_t weak_mat_tmp305 = weak_mat_tmp108*weak_mat_tmp17;
    const s_t weak_mat_tmp306 = weak_mat_tmp107*weak_mat_tmp298 + weak_mat_tmp130*weak_mat_tmp305;
    const s_t weak_mat_tmp307 = s_t(2)*weak_mat_tmp59;
    const s_t weak_mat_tmp308 = weak_mat_tmp169*weak_mat_tmp45;
    const s_t weak_mat_tmp309 = weak_mat_tmp137*weak_mat_tmp302 + weak_mat_tmp138*weak_mat_tmp308 + weak_mat_tmp43*(-weak_mat_tmp204 - weak_mat_tmp307);
    const s_t weak_mat_tmp310 = weak_mat_tmp158*weak_mat_tmp296;
    const s_t weak_mat_tmp311 = weak_mat_tmp124*weak_mat_tmp298 + weak_mat_tmp149*weak_mat_tmp305;
    const s_t weak_mat_tmp312 = gu[6]*weak_mat_tmp100;
    const s_t weak_mat_tmp313 = weak_mat_tmp153*weak_mat_tmp302 + weak_mat_tmp154*weak_mat_tmp308 + weak_mat_tmp43*(-weak_mat_tmp277 - weak_mat_tmp312);
    const s_t weak_mat_tmp314 = weak_mat_tmp117*weak_mat_tmp298 + weak_mat_tmp118*weak_mat_tmp305 + weak_mat_tmp284;
    const s_t weak_mat_tmp315 = weak_mat_tmp123*weak_mat_tmp302 + weak_mat_tmp126*weak_mat_tmp308 + weak_mat_tmp286 + weak_mat_tmp43*(s_t(4)*gu[6]*weak_mat_tmp1 - s_t(2)*weak_mat_tmp12);
    const s_t weak_mat_tmp316 = weak_mat_tmp113*weak_mat_tmp296 - weak_mat_tmp288;
    const s_t weak_mat_tmp317 = weak_mat_tmp132*weak_mat_tmp7;
    const s_t weak_mat_tmp318 = weak_mat_tmp125*weak_mat_tmp298 + weak_mat_tmp179*weak_mat_tmp305 - weak_mat_tmp317;
    const s_t weak_mat_tmp319 = weak_mat_tmp139*weak_mat_tmp7;
    const s_t weak_mat_tmp320 = weak_mat_tmp183*weak_mat_tmp302 + weak_mat_tmp184*weak_mat_tmp308 - weak_mat_tmp319 + weak_mat_tmp43*(-s_t(2)*weak_mat_tmp0 + s_t(4)*weak_mat_tmp1*weak_mat_tmp2);
    const s_t weak_mat_tmp321 = weak_mat_tmp145*weak_mat_tmp7;
    const s_t weak_mat_tmp322 = weak_mat_tmp189*weak_mat_tmp296 + weak_mat_tmp321;
    const s_t weak_mat_tmp323 = weak_mat_tmp143*weak_mat_tmp9;
    const s_t weak_mat_tmp324 = weak_mat_tmp13*weak_mat_tmp323;
    const s_t weak_mat_tmp325 = weak_mat_tmp130*weak_mat_tmp17;
    const s_t weak_mat_tmp326 = weak_mat_tmp130*weak_mat_tmp35;
    const s_t weak_mat_tmp327 = weak_mat_tmp138*weak_mat_tmp45;
    const s_t weak_mat_tmp328 = weak_mat_tmp137*weak_mat_tmp64;
    const s_t weak_mat_tmp329 = weak_mat_tmp189*weak_mat_tmp323;
    const s_t weak_mat_tmp330 = weak_mat_tmp107*weak_mat_tmp17;
    const s_t weak_mat_tmp331 = weak_mat_tmp125*weak_mat_tmp325 + weak_mat_tmp179*weak_mat_tmp330;
    const s_t weak_mat_tmp332 = weak_mat_tmp137*weak_mat_tmp45;
    const s_t weak_mat_tmp333 = weak_mat_tmp183*weak_mat_tmp327 + weak_mat_tmp184*weak_mat_tmp332 + weak_mat_tmp43*(-weak_mat_tmp278 - weak_mat_tmp312);
    const s_t weak_mat_tmp334 = weak_mat_tmp117*weak_mat_tmp325 + weak_mat_tmp118*weak_mat_tmp330 - weak_mat_tmp290;
    const s_t weak_mat_tmp335 = weak_mat_tmp123*weak_mat_tmp327 + weak_mat_tmp126*weak_mat_tmp332 - weak_mat_tmp292 + weak_mat_tmp43*(s_t(4)*gu[5]*gu[6] - s_t(2)*weak_mat_tmp66);
    const s_t weak_mat_tmp336 = weak_mat_tmp113*weak_mat_tmp323 + weak_mat_tmp294;
    const s_t weak_mat_tmp337 = weak_mat_tmp124*weak_mat_tmp325 + weak_mat_tmp149*weak_mat_tmp330 + weak_mat_tmp317;
    const s_t weak_mat_tmp338 = weak_mat_tmp153*weak_mat_tmp327 + weak_mat_tmp154*weak_mat_tmp332 + weak_mat_tmp319 + weak_mat_tmp43*(s_t(4)*weak_mat_tmp0 - s_t(2)*weak_mat_tmp18);
    const s_t weak_mat_tmp339 = weak_mat_tmp158*weak_mat_tmp323 - weak_mat_tmp321;
    const s_t weak_mat_tmp340 = weak_mat_tmp113*weak_mat_tmp9;
    const s_t weak_mat_tmp341 = weak_mat_tmp13*weak_mat_tmp340;
    const s_t weak_mat_tmp342 = weak_mat_tmp118*weak_mat_tmp17;
    const s_t weak_mat_tmp343 = weak_mat_tmp118*weak_mat_tmp35;
    const s_t weak_mat_tmp344 = weak_mat_tmp263 + weak_mat_tmp37;
    const s_t weak_mat_tmp345 = weak_mat_tmp265 + weak_mat_tmp41;
    const s_t weak_mat_tmp346 = weak_mat_tmp123*weak_mat_tmp45;
    const s_t weak_mat_tmp347 = weak_mat_tmp123*weak_mat_tmp64;
    const s_t weak_mat_tmp348 = weak_mat_tmp158*weak_mat_tmp340;
    const s_t weak_mat_tmp349 = weak_mat_tmp117*weak_mat_tmp17;
    const s_t weak_mat_tmp350 = weak_mat_tmp124*weak_mat_tmp342 + weak_mat_tmp149*weak_mat_tmp349;
    const s_t weak_mat_tmp351 = weak_mat_tmp126*weak_mat_tmp45;
    const s_t weak_mat_tmp352 = weak_mat_tmp153*weak_mat_tmp351 + weak_mat_tmp154*weak_mat_tmp346 + weak_mat_tmp43*(-weak_mat_tmp282 - weak_mat_tmp76);
    const s_t weak_mat_tmp353 = weak_mat_tmp189*weak_mat_tmp340;
    const s_t weak_mat_tmp354 = weak_mat_tmp125*weak_mat_tmp342 + weak_mat_tmp179*weak_mat_tmp349;
    const s_t weak_mat_tmp355 = weak_mat_tmp183*weak_mat_tmp351 + weak_mat_tmp184*weak_mat_tmp346 + weak_mat_tmp43*(-weak_mat_tmp272 - weak_mat_tmp89);
    const s_t weak_mat_tmp356 = weak_mat_tmp158*weak_mat_tmp9;
    const s_t weak_mat_tmp357 = weak_mat_tmp13*weak_mat_tmp356;
    const s_t weak_mat_tmp358 = weak_mat_tmp149*weak_mat_tmp17;
    const s_t weak_mat_tmp359 = weak_mat_tmp149*weak_mat_tmp35;
    const s_t weak_mat_tmp360 = weak_mat_tmp195 + weak_mat_tmp300;
    const s_t weak_mat_tmp361 = weak_mat_tmp153*weak_mat_tmp45;
    const s_t weak_mat_tmp362 = weak_mat_tmp153*weak_mat_tmp64;
    const s_t weak_mat_tmp363 = weak_mat_tmp189*weak_mat_tmp356;
    const s_t weak_mat_tmp364 = weak_mat_tmp124*weak_mat_tmp17*weak_mat_tmp179 + weak_mat_tmp125*weak_mat_tmp358;
    const s_t weak_mat_tmp365 = weak_mat_tmp183*weak_mat_tmp45;
    const s_t weak_mat_tmp366 = weak_mat_tmp154*weak_mat_tmp365 + weak_mat_tmp184*weak_mat_tmp361 + weak_mat_tmp43*(-weak_mat_tmp203 - weak_mat_tmp307);
    const s_t weak_mat_tmp367 = weak_mat_tmp13*weak_mat_tmp189*weak_mat_tmp9;
    const s_t weak_mat_tmp368 = weak_mat_tmp179*weak_mat_tmp35;
    const s_t weak_mat_tmp369 = weak_mat_tmp183*weak_mat_tmp64;
    material[0] = trial_grad[0]*(c1*(weak_mat_tmp16 + s_t(4)*weak_mat_tmp20*weak_mat_tmp7 + weak_mat_tmp21*weak_mat_tmp36) + c2*(weak_mat_tmp43*(weak_mat_tmp39 + weak_mat_tmp42) + s_t(2)*weak_mat_tmp44*weak_mat_tmp57 + weak_mat_tmp58*weak_mat_tmp65) + weak_mat_tmp15*weak_mat_tmp3 + pow_2(weak_mat_tmp4)*weak_mat_tmp9) + trial_grad[1]*(c1*(weak_mat_tmp36*weak_mat_tmp70 + weak_mat_tmp73) + c2*(weak_mat_tmp65*weak_mat_tmp74 + weak_mat_tmp80) + weak_mat_tmp15*weak_mat_tmp67 + weak_mat_tmp69) + trial_grad[2]*(c1*(weak_mat_tmp36*weak_mat_tmp85 + weak_mat_tmp87) + c2*(weak_mat_tmp65*weak_mat_tmp88 + weak_mat_tmp93) + weak_mat_tmp15*weak_mat_tmp84 + weak_mat_tmp83) + trial_grad[3]*(c1*(weak_mat_tmp102 + weak_mat_tmp36*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp65 + weak_mat_tmp110) + weak_mat_tmp15*weak_mat_tmp95 + weak_mat_tmp97) + trial_grad[4]*(c1*(weak_mat_tmp163*weak_mat_tmp36 + weak_mat_tmp167) + c2*(weak_mat_tmp168*weak_mat_tmp65 + weak_mat_tmp172) + weak_mat_tmp15*weak_mat_tmp173 + weak_mat_tmp176) + trial_grad[5]*(c1*(weak_mat_tmp129*weak_mat_tmp36 + weak_mat_tmp134) + c2*(weak_mat_tmp135*weak_mat_tmp65 + weak_mat_tmp141) + weak_mat_tmp144*weak_mat_tmp15 + weak_mat_tmp147) + trial_grad[6]*(c1*(weak_mat_tmp116*weak_mat_tmp36 + weak_mat_tmp119) + c2*(weak_mat_tmp120*weak_mat_tmp65 + weak_mat_tmp127) + weak_mat_tmp114 + weak_mat_tmp115*weak_mat_tmp15) + trial_grad[7]*(c1*(weak_mat_tmp148*weak_mat_tmp36 + weak_mat_tmp151) + c2*(weak_mat_tmp152*weak_mat_tmp65 + weak_mat_tmp156) + weak_mat_tmp15*weak_mat_tmp159 + weak_mat_tmp161) + trial_grad[8]*(c1*(weak_mat_tmp178*weak_mat_tmp36 + weak_mat_tmp181) + c2*(weak_mat_tmp182*weak_mat_tmp65 + weak_mat_tmp186) + weak_mat_tmp15*weak_mat_tmp187 + weak_mat_tmp190);
    material[1] = trial_grad[0]*(c1*(weak_mat_tmp194*weak_mat_tmp21 + weak_mat_tmp73) + c2*(weak_mat_tmp199*weak_mat_tmp58 + weak_mat_tmp80) + weak_mat_tmp192*weak_mat_tmp3 + weak_mat_tmp69) + trial_grad[1]*(c1*(s_t(4)*gu[1]*weak_mat_tmp193 + weak_mat_tmp16 + weak_mat_tmp194*weak_mat_tmp70) + c2*(s_t(2)*weak_mat_tmp198*weak_mat_tmp77 + weak_mat_tmp199*weak_mat_tmp74 + weak_mat_tmp43*(weak_mat_tmp197 + weak_mat_tmp39)) + weak_mat_tmp192*weak_mat_tmp67 + pow_2(weak_mat_tmp68)*weak_mat_tmp9) + trial_grad[2]*(c1*(weak_mat_tmp194*weak_mat_tmp85 + weak_mat_tmp202) + c2*(weak_mat_tmp199*weak_mat_tmp88 + weak_mat_tmp206) + weak_mat_tmp192*weak_mat_tmp84 + weak_mat_tmp200) + trial_grad[3]*(c1*(weak_mat_tmp194*weak_mat_tmp99 + weak_mat_tmp224) + c2*(weak_mat_tmp103*weak_mat_tmp199 + weak_mat_tmp225) + weak_mat_tmp192*weak_mat_tmp95 + weak_mat_tmp226) + trial_grad[4]*(c1*(weak_mat_tmp163*weak_mat_tmp194 + weak_mat_tmp212) + c2*(weak_mat_tmp168*weak_mat_tmp199 + weak_mat_tmp214) + weak_mat_tmp173*weak_mat_tmp192 + weak_mat_tmp211) + trial_grad[5]*(c1*(weak_mat_tmp129*weak_mat_tmp194 + weak_mat_tmp216) + c2*(weak_mat_tmp135*weak_mat_tmp199 + weak_mat_tmp218) + weak_mat_tmp144*weak_mat_tmp192 + weak_mat_tmp220) + trial_grad[6]*(c1*(weak_mat_tmp116*weak_mat_tmp194 + weak_mat_tmp221) + c2*(weak_mat_tmp120*weak_mat_tmp199 + weak_mat_tmp222) + weak_mat_tmp115*weak_mat_tmp192 + weak_mat_tmp223) + trial_grad[7]*(c1*(weak_mat_tmp148*weak_mat_tmp194 + weak_mat_tmp208) + c2*(weak_mat_tmp152*weak_mat_tmp199 + weak_mat_tmp210) + weak_mat_tmp159*weak_mat_tmp192 + weak_mat_tmp207) + trial_grad[8]*(c1*(weak_mat_tmp178*weak_mat_tmp194 + weak_mat_tmp228) + c2*(weak_mat_tmp182*weak_mat_tmp199 + weak_mat_tmp230) + weak_mat_tmp187*weak_mat_tmp192 + weak_mat_tmp232);
    material[2] = trial_grad[0]*(c1*(weak_mat_tmp21*weak_mat_tmp236 + weak_mat_tmp87) + c2*(weak_mat_tmp238*weak_mat_tmp58 + weak_mat_tmp93) + weak_mat_tmp234*weak_mat_tmp3 + weak_mat_tmp83) + trial_grad[1]*(c1*(weak_mat_tmp202 + weak_mat_tmp236*weak_mat_tmp70) + c2*(weak_mat_tmp206 + weak_mat_tmp238*weak_mat_tmp74) + weak_mat_tmp200 + weak_mat_tmp234*weak_mat_tmp67) + trial_grad[2]*(c1*(s_t(4)*gu[2]*weak_mat_tmp235 + weak_mat_tmp16 + weak_mat_tmp236*weak_mat_tmp85) + c2*(s_t(2)*weak_mat_tmp237*weak_mat_tmp91 + weak_mat_tmp238*weak_mat_tmp88 + weak_mat_tmp43*(weak_mat_tmp197 + weak_mat_tmp42)) + weak_mat_tmp234*weak_mat_tmp84 + pow_2(weak_mat_tmp82)*weak_mat_tmp9) + trial_grad[3]*(c1*(weak_mat_tmp236*weak_mat_tmp99 + weak_mat_tmp247) + c2*(weak_mat_tmp103*weak_mat_tmp238 + weak_mat_tmp248) + weak_mat_tmp234*weak_mat_tmp95 + weak_mat_tmp249) + trial_grad[4]*(c1*(weak_mat_tmp163*weak_mat_tmp236 + weak_mat_tmp256) + c2*(weak_mat_tmp168*weak_mat_tmp238 + weak_mat_tmp257) + weak_mat_tmp173*weak_mat_tmp234 + weak_mat_tmp258) + trial_grad[5]*(c1*(weak_mat_tmp129*weak_mat_tmp236 + weak_mat_tmp241) + c2*(weak_mat_tmp135*weak_mat_tmp238 + weak_mat_tmp243) + weak_mat_tmp144*weak_mat_tmp234 + weak_mat_tmp239) + trial_grad[6]*(c1*(weak_mat_tmp116*weak_mat_tmp236 + weak_mat_tmp253) + c2*(weak_mat_tmp120*weak_mat_tmp238 + weak_mat_tmp254) + weak_mat_tmp115*weak_mat_tmp234 + weak_mat_tmp255) + trial_grad[7]*(c1*(weak_mat_tmp148*weak_mat_tmp236 + weak_mat_tmp250) + c2*(weak_mat_tmp152*weak_mat_tmp238 + weak_mat_tmp251) + weak_mat_tmp159*weak_mat_tmp234 + weak_mat_tmp252) + trial_grad[8]*(c1*(weak_mat_tmp178*weak_mat_tmp236 + weak_mat_tmp245) + c2*(weak_mat_tmp182*weak_mat_tmp238 + weak_mat_tmp246) + weak_mat_tmp187*weak_mat_tmp234 + weak_mat_tmp244);
    material[3] = trial_grad[0]*(c1*(weak_mat_tmp102 + weak_mat_tmp21*weak_mat_tmp262) + c2*(weak_mat_tmp110 + weak_mat_tmp268*weak_mat_tmp58) + weak_mat_tmp260*weak_mat_tmp3 + weak_mat_tmp97) + trial_grad[1]*(c1*(weak_mat_tmp224 + weak_mat_tmp262*weak_mat_tmp70) + c2*(weak_mat_tmp225 + weak_mat_tmp268*weak_mat_tmp74) + weak_mat_tmp226 + weak_mat_tmp260*weak_mat_tmp67) + trial_grad[2]*(c1*(weak_mat_tmp247 + weak_mat_tmp262*weak_mat_tmp85) + c2*(weak_mat_tmp248 + weak_mat_tmp268*weak_mat_tmp88) + weak_mat_tmp249 + weak_mat_tmp260*weak_mat_tmp84) + trial_grad[3]*(c1*(s_t(4)*gu[3]*weak_mat_tmp261 + weak_mat_tmp16 + weak_mat_tmp262*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp268 + s_t(2)*weak_mat_tmp109*weak_mat_tmp267 + weak_mat_tmp43*(weak_mat_tmp264 + weak_mat_tmp266)) + weak_mat_tmp260*weak_mat_tmp95 + weak_mat_tmp9*pow_2(weak_mat_tmp96)) + trial_grad[4]*(c1*(weak_mat_tmp163*weak_mat_tmp262 + weak_mat_tmp281) + c2*(weak_mat_tmp168*weak_mat_tmp268 + weak_mat_tmp283) + weak_mat_tmp173*weak_mat_tmp260 + weak_mat_tmp280) + trial_grad[5]*(c1*(weak_mat_tmp129*weak_mat_tmp262 + weak_mat_tmp271) + c2*(weak_mat_tmp135*weak_mat_tmp268 + weak_mat_tmp274) + weak_mat_tmp144*weak_mat_tmp260 + weak_mat_tmp269) + trial_grad[6]*(c1*(weak_mat_tmp116*weak_mat_tmp262 + weak_mat_tmp276) + c2*(weak_mat_tmp120*weak_mat_tmp268 + weak_mat_tmp279) + weak_mat_tmp115*weak_mat_tmp260 + weak_mat_tmp275) + trial_grad[7]*(c1*(weak_mat_tmp148*weak_mat_tmp262 + weak_mat_tmp285) + c2*(weak_mat_tmp152*weak_mat_tmp268 + weak_mat_tmp287) + weak_mat_tmp159*weak_mat_tmp260 + weak_mat_tmp289) + trial_grad[8]*(c1*(weak_mat_tmp178*weak_mat_tmp262 + weak_mat_tmp291) + c2*(weak_mat_tmp182*weak_mat_tmp268 + weak_mat_tmp293) + weak_mat_tmp187*weak_mat_tmp260 + weak_mat_tmp295);
    material[4] = trial_grad[0]*(c1*(weak_mat_tmp167 + weak_mat_tmp21*weak_mat_tmp299) + c2*(weak_mat_tmp172 + weak_mat_tmp303*weak_mat_tmp58) + weak_mat_tmp176 + weak_mat_tmp297*weak_mat_tmp3) + trial_grad[1]*(c1*(weak_mat_tmp212 + weak_mat_tmp299*weak_mat_tmp70) + c2*(weak_mat_tmp214 + weak_mat_tmp303*weak_mat_tmp74) + weak_mat_tmp211 + weak_mat_tmp297*weak_mat_tmp67) + trial_grad[2]*(c1*(weak_mat_tmp256 + weak_mat_tmp299*weak_mat_tmp85) + c2*(weak_mat_tmp257 + weak_mat_tmp303*weak_mat_tmp88) + weak_mat_tmp258 + weak_mat_tmp297*weak_mat_tmp84) + trial_grad[3]*(c1*(weak_mat_tmp281 + weak_mat_tmp299*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp303 + weak_mat_tmp283) + weak_mat_tmp280 + weak_mat_tmp297*weak_mat_tmp95) + trial_grad[4]*(c1*(s_t(4)*weak_mat_tmp1*weak_mat_tmp298 + weak_mat_tmp16 + weak_mat_tmp163*weak_mat_tmp299) + c2*(weak_mat_tmp168*weak_mat_tmp303 + s_t(2)*weak_mat_tmp169*weak_mat_tmp302 + weak_mat_tmp43*(weak_mat_tmp264 + weak_mat_tmp301)) + weak_mat_tmp173*weak_mat_tmp297 + pow_2(weak_mat_tmp175)*weak_mat_tmp9) + trial_grad[5]*(c1*(weak_mat_tmp129*weak_mat_tmp299 + weak_mat_tmp306) + c2*(weak_mat_tmp135*weak_mat_tmp303 + weak_mat_tmp309) + weak_mat_tmp144*weak_mat_tmp297 + weak_mat_tmp304) + trial_grad[6]*(c1*(weak_mat_tmp116*weak_mat_tmp299 + weak_mat_tmp314) + c2*(weak_mat_tmp120*weak_mat_tmp303 + weak_mat_tmp315) + weak_mat_tmp115*weak_mat_tmp297 + weak_mat_tmp316) + trial_grad[7]*(c1*(weak_mat_tmp148*weak_mat_tmp299 + weak_mat_tmp311) + c2*(weak_mat_tmp152*weak_mat_tmp303 + weak_mat_tmp313) + weak_mat_tmp159*weak_mat_tmp297 + weak_mat_tmp310) + trial_grad[8]*(c1*(weak_mat_tmp178*weak_mat_tmp299 + weak_mat_tmp318) + c2*(weak_mat_tmp182*weak_mat_tmp303 + weak_mat_tmp320) + weak_mat_tmp187*weak_mat_tmp297 + weak_mat_tmp322);
    material[5] = trial_grad[0]*(c1*(weak_mat_tmp134 + weak_mat_tmp21*weak_mat_tmp326) + c2*(weak_mat_tmp141 + weak_mat_tmp328*weak_mat_tmp58) + weak_mat_tmp147 + weak_mat_tmp3*weak_mat_tmp324) + trial_grad[1]*(c1*(weak_mat_tmp216 + weak_mat_tmp326*weak_mat_tmp70) + c2*(weak_mat_tmp218 + weak_mat_tmp328*weak_mat_tmp74) + weak_mat_tmp220 + weak_mat_tmp324*weak_mat_tmp67) + trial_grad[2]*(c1*(weak_mat_tmp241 + weak_mat_tmp326*weak_mat_tmp85) + c2*(weak_mat_tmp243 + weak_mat_tmp328*weak_mat_tmp88) + weak_mat_tmp239 + weak_mat_tmp324*weak_mat_tmp84) + trial_grad[3]*(c1*(weak_mat_tmp271 + weak_mat_tmp326*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp328 + weak_mat_tmp274) + weak_mat_tmp269 + weak_mat_tmp324*weak_mat_tmp95) + trial_grad[4]*(c1*(weak_mat_tmp163*weak_mat_tmp326 + weak_mat_tmp306) + c2*(weak_mat_tmp168*weak_mat_tmp328 + weak_mat_tmp309) + weak_mat_tmp173*weak_mat_tmp324 + weak_mat_tmp304) + trial_grad[5]*(c1*(s_t(4)*gu[5]*weak_mat_tmp325 + weak_mat_tmp129*weak_mat_tmp326 + weak_mat_tmp16) + c2*(weak_mat_tmp135*weak_mat_tmp328 + s_t(2)*weak_mat_tmp137*weak_mat_tmp327 + weak_mat_tmp43*(weak_mat_tmp266 + weak_mat_tmp301)) + pow_2(weak_mat_tmp143)*weak_mat_tmp9 + weak_mat_tmp144*weak_mat_tmp324) + trial_grad[6]*(c1*(weak_mat_tmp116*weak_mat_tmp326 + weak_mat_tmp334) + c2*(weak_mat_tmp120*weak_mat_tmp328 + weak_mat_tmp335) + weak_mat_tmp115*weak_mat_tmp324 + weak_mat_tmp336) + trial_grad[7]*(c1*(weak_mat_tmp148*weak_mat_tmp326 + weak_mat_tmp337) + c2*(weak_mat_tmp152*weak_mat_tmp328 + weak_mat_tmp338) + weak_mat_tmp159*weak_mat_tmp324 + weak_mat_tmp339) + trial_grad[8]*(c1*(weak_mat_tmp178*weak_mat_tmp326 + weak_mat_tmp331) + c2*(weak_mat_tmp182*weak_mat_tmp328 + weak_mat_tmp333) + weak_mat_tmp187*weak_mat_tmp324 + weak_mat_tmp329);
    material[6] = trial_grad[0]*(c1*(weak_mat_tmp119 + weak_mat_tmp21*weak_mat_tmp343) + c2*(weak_mat_tmp127 + weak_mat_tmp347*weak_mat_tmp58) + weak_mat_tmp114 + weak_mat_tmp3*weak_mat_tmp341) + trial_grad[1]*(c1*(weak_mat_tmp221 + weak_mat_tmp343*weak_mat_tmp70) + c2*(weak_mat_tmp222 + weak_mat_tmp347*weak_mat_tmp74) + weak_mat_tmp223 + weak_mat_tmp341*weak_mat_tmp67) + trial_grad[2]*(c1*(weak_mat_tmp253 + weak_mat_tmp343*weak_mat_tmp85) + c2*(weak_mat_tmp254 + weak_mat_tmp347*weak_mat_tmp88) + weak_mat_tmp255 + weak_mat_tmp341*weak_mat_tmp84) + trial_grad[3]*(c1*(weak_mat_tmp276 + weak_mat_tmp343*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp347 + weak_mat_tmp279) + weak_mat_tmp275 + weak_mat_tmp341*weak_mat_tmp95) + trial_grad[4]*(c1*(weak_mat_tmp163*weak_mat_tmp343 + weak_mat_tmp314) + c2*(weak_mat_tmp168*weak_mat_tmp347 + weak_mat_tmp315) + weak_mat_tmp173*weak_mat_tmp341 + weak_mat_tmp316) + trial_grad[5]*(c1*(weak_mat_tmp129*weak_mat_tmp343 + weak_mat_tmp334) + c2*(weak_mat_tmp135*weak_mat_tmp347 + weak_mat_tmp335) + weak_mat_tmp144*weak_mat_tmp341 + weak_mat_tmp336) + trial_grad[6]*(c1*(s_t(4)*gu[6]*weak_mat_tmp342 + weak_mat_tmp116*weak_mat_tmp343 + weak_mat_tmp16) + c2*(weak_mat_tmp120*weak_mat_tmp347 + s_t(2)*weak_mat_tmp126*weak_mat_tmp346 + weak_mat_tmp43*(weak_mat_tmp344 + weak_mat_tmp345)) + pow_2(weak_mat_tmp113)*weak_mat_tmp9 + weak_mat_tmp115*weak_mat_tmp341) + trial_grad[7]*(c1*(weak_mat_tmp148*weak_mat_tmp343 + weak_mat_tmp350) + c2*(weak_mat_tmp152*weak_mat_tmp347 + weak_mat_tmp352) + weak_mat_tmp159*weak_mat_tmp341 + weak_mat_tmp348) + trial_grad[8]*(c1*(weak_mat_tmp178*weak_mat_tmp343 + weak_mat_tmp354) + c2*(weak_mat_tmp182*weak_mat_tmp347 + weak_mat_tmp355) + weak_mat_tmp187*weak_mat_tmp341 + weak_mat_tmp353);
    material[7] = trial_grad[0]*(c1*(weak_mat_tmp151 + weak_mat_tmp21*weak_mat_tmp359) + c2*(weak_mat_tmp156 + weak_mat_tmp362*weak_mat_tmp58) + weak_mat_tmp161 + weak_mat_tmp3*weak_mat_tmp357) + trial_grad[1]*(c1*(weak_mat_tmp208 + weak_mat_tmp359*weak_mat_tmp70) + c2*(weak_mat_tmp210 + weak_mat_tmp362*weak_mat_tmp74) + weak_mat_tmp207 + weak_mat_tmp357*weak_mat_tmp67) + trial_grad[2]*(c1*(weak_mat_tmp250 + weak_mat_tmp359*weak_mat_tmp85) + c2*(weak_mat_tmp251 + weak_mat_tmp362*weak_mat_tmp88) + weak_mat_tmp252 + weak_mat_tmp357*weak_mat_tmp84) + trial_grad[3]*(c1*(weak_mat_tmp285 + weak_mat_tmp359*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp362 + weak_mat_tmp287) + weak_mat_tmp289 + weak_mat_tmp357*weak_mat_tmp95) + trial_grad[4]*(c1*(weak_mat_tmp163*weak_mat_tmp359 + weak_mat_tmp311) + c2*(weak_mat_tmp168*weak_mat_tmp362 + weak_mat_tmp313) + weak_mat_tmp173*weak_mat_tmp357 + weak_mat_tmp310) + trial_grad[5]*(c1*(weak_mat_tmp129*weak_mat_tmp359 + weak_mat_tmp337) + c2*(weak_mat_tmp135*weak_mat_tmp362 + weak_mat_tmp338) + weak_mat_tmp144*weak_mat_tmp357 + weak_mat_tmp339) + trial_grad[6]*(c1*(weak_mat_tmp116*weak_mat_tmp359 + weak_mat_tmp350) + c2*(weak_mat_tmp120*weak_mat_tmp362 + weak_mat_tmp352) + weak_mat_tmp115*weak_mat_tmp357 + weak_mat_tmp348) + trial_grad[7]*(c1*(s_t(4)*gu[7]*weak_mat_tmp358 + weak_mat_tmp148*weak_mat_tmp359 + weak_mat_tmp16) + c2*(weak_mat_tmp152*weak_mat_tmp362 + s_t(2)*weak_mat_tmp154*weak_mat_tmp361 + weak_mat_tmp43*(weak_mat_tmp344 + weak_mat_tmp360)) + pow_2(weak_mat_tmp158)*weak_mat_tmp9 + weak_mat_tmp159*weak_mat_tmp357) + trial_grad[8]*(c1*(weak_mat_tmp178*weak_mat_tmp359 + weak_mat_tmp364) + c2*(weak_mat_tmp182*weak_mat_tmp362 + weak_mat_tmp366) + weak_mat_tmp187*weak_mat_tmp357 + weak_mat_tmp363);
    material[8] = trial_grad[0]*(c1*(weak_mat_tmp181 + weak_mat_tmp21*weak_mat_tmp368) + c2*(weak_mat_tmp186 + weak_mat_tmp369*weak_mat_tmp58) + weak_mat_tmp190 + weak_mat_tmp3*weak_mat_tmp367) + trial_grad[1]*(c1*(weak_mat_tmp228 + weak_mat_tmp368*weak_mat_tmp70) + c2*(weak_mat_tmp230 + weak_mat_tmp369*weak_mat_tmp74) + weak_mat_tmp232 + weak_mat_tmp367*weak_mat_tmp67) + trial_grad[2]*(c1*(weak_mat_tmp245 + weak_mat_tmp368*weak_mat_tmp85) + c2*(weak_mat_tmp246 + weak_mat_tmp369*weak_mat_tmp88) + weak_mat_tmp244 + weak_mat_tmp367*weak_mat_tmp84) + trial_grad[3]*(c1*(weak_mat_tmp291 + weak_mat_tmp368*weak_mat_tmp99) + c2*(weak_mat_tmp103*weak_mat_tmp369 + weak_mat_tmp293) + weak_mat_tmp295 + weak_mat_tmp367*weak_mat_tmp95) + trial_grad[4]*(c1*(weak_mat_tmp163*weak_mat_tmp368 + weak_mat_tmp318) + c2*(weak_mat_tmp168*weak_mat_tmp369 + weak_mat_tmp320) + weak_mat_tmp173*weak_mat_tmp367 + weak_mat_tmp322) + trial_grad[5]*(c1*(weak_mat_tmp129*weak_mat_tmp368 + weak_mat_tmp331) + c2*(weak_mat_tmp135*weak_mat_tmp369 + weak_mat_tmp333) + weak_mat_tmp144*weak_mat_tmp367 + weak_mat_tmp329) + trial_grad[6]*(c1*(weak_mat_tmp116*weak_mat_tmp368 + weak_mat_tmp354) + c2*(weak_mat_tmp120*weak_mat_tmp369 + weak_mat_tmp355) + weak_mat_tmp115*weak_mat_tmp367 + weak_mat_tmp353) + trial_grad[7]*(c1*(weak_mat_tmp148*weak_mat_tmp368 + weak_mat_tmp364) + c2*(weak_mat_tmp152*weak_mat_tmp369 + weak_mat_tmp366) + weak_mat_tmp159*weak_mat_tmp367 + weak_mat_tmp363) + trial_grad[8]*(c1*(weak_mat_tmp16 + s_t(4)*weak_mat_tmp165*weak_mat_tmp179 + weak_mat_tmp178*weak_mat_tmp368) + c2*(weak_mat_tmp182*weak_mat_tmp369 + s_t(2)*weak_mat_tmp184*weak_mat_tmp365 + weak_mat_tmp43*(weak_mat_tmp345 + weak_mat_tmp360)) + weak_mat_tmp187*weak_mat_tmp367 + pow_2(weak_mat_tmp189)*weak_mat_tmp9);
    loperand[0] = qw * (material[0] * adj_lane0 + material[1] * adj_lane1 + material[2] * adj_lane2);
    loperand[1] = qw * (material[0] * adj_lane3 + material[1] * adj_lane4 + material[2] * adj_lane5);
    loperand[2] = qw * (material[0] * adj_lane6 + material[1] * adj_lane7 + material[2] * adj_lane8);
    loperand[3] = qw * (material[3] * adj_lane0 + material[4] * adj_lane1 + material[5] * adj_lane2);
    loperand[4] = qw * (material[3] * adj_lane3 + material[4] * adj_lane4 + material[5] * adj_lane5);
    loperand[5] = qw * (material[3] * adj_lane6 + material[4] * adj_lane7 + material[5] * adj_lane8);
    loperand[6] = qw * (material[6] * adj_lane0 + material[7] * adj_lane1 + material[8] * adj_lane2);
    loperand[7] = qw * (material[6] * adj_lane3 + material[7] * adj_lane4 + material[8] * adj_lane5);
    loperand[8] = qw * (material[6] * adj_lane6 + material[7] * adj_lane7 + material[8] * adj_lane8);
      loperand0[lane] = loperand[0];
      loperand1[lane] = loperand[1];
      loperand2[lane] = loperand[2];
      loperand3[lane] = loperand[3];
      loperand4[lane] = loperand[4];
      loperand5[lane] = loperand[5];
      loperand6[lane] = loperand[6];
      loperand7[lane] = loperand[7];
      loperand8[lane] = loperand[8];
    }
  }
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[0], out_streams, 0);
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[3 * NQ * VS], out_streams, 1);
  tensor_test<s_t, NQ, NS, VS, 3, 3>(ne, shape_1d, grad_1d, &loperand_q[6 * NQ * VS], out_streams, 2);
}

} // namespace codegen
} // namespace sfem

#endif
