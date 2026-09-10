#ifndef NAVIER_STOKES_FORM_2_U_P_D3_TENSOR_PRODUCT_MIXED_LOCAL_HPP
#define NAVIER_STOKES_FORM_2_U_P_D3_TENSOR_PRODUCT_MIXED_LOCAL_HPP

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

template <typename s_t, int NQ, int CELL_NS, int VS>
static SFEM_INLINE void navier_stokes_form_2_u_p_d3_tensor_product_mixed_jacobian_action_block(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR field_shape_1d[2],
    const s_t *const RSTR field_grad_1d[2],
    const s_t *const RSTR q_weight_1d,
    const s_t *const RSTR direction[89],
    s_t *const RSTR output[89]
) {
  static constexpr int ND = 3;
  static constexpr int U_NS = 27;
  static constexpr int P_NS = 8;
  static constexpr int NQ1 = integer_root(NQ, ND);
  static_assert(ipow(NQ1, ND) == NQ, "NQ must be tensor-product compatible");
  static constexpr int U_NS1 = integer_root(U_NS, ND);
  static_assert(ipow(U_NS1, ND) == U_NS, "U_NS must be tensor-product compatible");
  static constexpr int P_NS1 = integer_root(P_NS, ND);
  static_assert(ipow(P_NS1, ND) == P_NS, "P_NS must be tensor-product compatible");
  s_t direction_p_value[NQ * VS];
  const s_t *const direction_p_streams[P_NS] = {direction[81], direction[82], direction[83], direction[84], direction[85], direction[86], direction[87], direction[88]};
  tensor_evaluate_value<s_t, NQ, P_NS, VS, ND, 1>(
      ne, field_shape_1d[1], direction_p_streams, direction_p_value);
  s_t u0_value_coeff[NQ * VS];
  s_t u0_grad_coeff_ref[NQ * ND * VS];
  s_t u1_value_coeff[NQ * VS];
  s_t u1_grad_coeff_ref[NQ * ND * VS];
  s_t u2_value_coeff[NQ * VS];
  s_t u2_grad_coeff_ref[NQ * ND * VS];
  s_t p_value_coeff[NQ * VS];
  s_t p_grad_coeff_ref[NQ * ND * VS];
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
    const s_t *const RSTR det_q = determinant + q * geometry_stride;
    const s_t *const RSTR adj_q0 = adjugate[0] + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adjugate[1] + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adjugate[2] + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adjugate[3] + q * geometry_stride;
    const s_t *const RSTR adj_q4 = adjugate[4] + q * geometry_stride;
    const s_t *const RSTR adj_q5 = adjugate[5] + q * geometry_stride;
    const s_t *const RSTR adj_q6 = adjugate[6] + q * geometry_stride;
    const s_t *const RSTR adj_q7 = adjugate[7] + q * geometry_stride;
    const s_t *const RSTR adj_q8 = adjugate[8] + q * geometry_stride;
    const s_t *const RSTR direction_p_value_q = &direction_p_value[q * VS];
    s_t *const RSTR u0_value_coeff_q = &u0_value_coeff[q * VS];
    s_t *const RSTR u0_grad_coeff_ref_q0 = &u0_grad_coeff_ref[(q * ND + 0) * VS];
    s_t *const RSTR u0_grad_coeff_ref_q1 = &u0_grad_coeff_ref[(q * ND + 1) * VS];
    s_t *const RSTR u0_grad_coeff_ref_q2 = &u0_grad_coeff_ref[(q * ND + 2) * VS];
    s_t *const RSTR u1_value_coeff_q = &u1_value_coeff[q * VS];
    s_t *const RSTR u1_grad_coeff_ref_q0 = &u1_grad_coeff_ref[(q * ND + 0) * VS];
    s_t *const RSTR u1_grad_coeff_ref_q1 = &u1_grad_coeff_ref[(q * ND + 1) * VS];
    s_t *const RSTR u1_grad_coeff_ref_q2 = &u1_grad_coeff_ref[(q * ND + 2) * VS];
    s_t *const RSTR u2_value_coeff_q = &u2_value_coeff[q * VS];
    s_t *const RSTR u2_grad_coeff_ref_q0 = &u2_grad_coeff_ref[(q * ND + 0) * VS];
    s_t *const RSTR u2_grad_coeff_ref_q1 = &u2_grad_coeff_ref[(q * ND + 1) * VS];
    s_t *const RSTR u2_grad_coeff_ref_q2 = &u2_grad_coeff_ref[(q * ND + 2) * VS];
    s_t *const RSTR p_value_coeff_q = &p_value_coeff[q * VS];
    s_t *const RSTR p_grad_coeff_ref_q0 = &p_grad_coeff_ref[(q * ND + 0) * VS];
    s_t *const RSTR p_grad_coeff_ref_q1 = &p_grad_coeff_ref[(q * ND + 1) * VS];
    s_t *const RSTR p_grad_coeff_ref_q2 = &p_grad_coeff_ref[(q * ND + 2) * VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t det = det_q[lane];
      const s_t adj0 = adj_q0[lane];
      const s_t adj1 = adj_q1[lane];
      const s_t adj2 = adj_q2[lane];
      const s_t adj3 = adj_q3[lane];
      const s_t adj4 = adj_q4[lane];
      const s_t adj5 = adj_q5[lane];
      const s_t adj6 = adj_q6[lane];
      const s_t adj7 = adj_q7[lane];
      const s_t adj8 = adj_q8[lane];
      const s_t p_direction = direction_p_value_q[lane];
      const s_t residual_tmp0 = -p_direction;
      const s_t grad_coeff0_0 = residual_tmp0;
      const s_t grad_coeff1_1 = residual_tmp0;
      const s_t grad_coeff2_2 = residual_tmp0;
      u0_value_coeff_q[lane] = s_t(0);
      u0_grad_coeff_ref_q0[lane] = qw * (adj0 * grad_coeff0_0);
      u0_grad_coeff_ref_q1[lane] = qw * (adj3 * grad_coeff0_0);
      u0_grad_coeff_ref_q2[lane] = qw * (adj6 * grad_coeff0_0);
      u1_value_coeff_q[lane] = s_t(0);
      u1_grad_coeff_ref_q0[lane] = qw * (adj1 * grad_coeff1_1);
      u1_grad_coeff_ref_q1[lane] = qw * (adj4 * grad_coeff1_1);
      u1_grad_coeff_ref_q2[lane] = qw * (adj7 * grad_coeff1_1);
      u2_value_coeff_q[lane] = s_t(0);
      u2_grad_coeff_ref_q0[lane] = qw * (adj2 * grad_coeff2_2);
      u2_grad_coeff_ref_q1[lane] = qw * (adj5 * grad_coeff2_2);
      u2_grad_coeff_ref_q2[lane] = qw * (adj8 * grad_coeff2_2);
      p_value_coeff_q[lane] = s_t(0);
      p_grad_coeff_ref_q0[lane] = s_t(0);
      p_grad_coeff_ref_q1[lane] = s_t(0);
      p_grad_coeff_ref_q2[lane] = s_t(0);
    }
  }
  s_t *const u0_output_streams[U_NS] = {output[0], output[1], output[2], output[3], output[4], output[5], output[6], output[7], output[8], output[9], output[10], output[11], output[12], output[13], output[14], output[15], output[16], output[17], output[18], output[19], output[20], output[21], output[22], output[23], output[24], output[25], output[26]};
  tensor_integrate<s_t, NQ, U_NS, VS, ND, 1>(
      ne, field_shape_1d[0], field_grad_1d[0], u0_value_coeff, u0_grad_coeff_ref, u0_output_streams);
  s_t *const u1_output_streams[U_NS] = {output[27], output[28], output[29], output[30], output[31], output[32], output[33], output[34], output[35], output[36], output[37], output[38], output[39], output[40], output[41], output[42], output[43], output[44], output[45], output[46], output[47], output[48], output[49], output[50], output[51], output[52], output[53]};
  tensor_integrate<s_t, NQ, U_NS, VS, ND, 1>(
      ne, field_shape_1d[0], field_grad_1d[0], u1_value_coeff, u1_grad_coeff_ref, u1_output_streams);
  s_t *const u2_output_streams[U_NS] = {output[54], output[55], output[56], output[57], output[58], output[59], output[60], output[61], output[62], output[63], output[64], output[65], output[66], output[67], output[68], output[69], output[70], output[71], output[72], output[73], output[74], output[75], output[76], output[77], output[78], output[79], output[80]};
  tensor_integrate<s_t, NQ, U_NS, VS, ND, 1>(
      ne, field_shape_1d[0], field_grad_1d[0], u2_value_coeff, u2_grad_coeff_ref, u2_output_streams);
  s_t *const p_output_streams[P_NS] = {output[81], output[82], output[83], output[84], output[85], output[86], output[87], output[88]};
  tensor_integrate<s_t, NQ, P_NS, VS, ND, 1>(
      ne, field_shape_1d[1], field_grad_1d[1], p_value_coeff, p_grad_coeff_ref, p_output_streams);
}

template <typename s_t, int NQ, int CELL_NS, int VS>
static SFEM_INLINE void navier_stokes_form_2_u_p_d3_tensor_product_mixed_jacobian_action_block_contiguous(
    const int ne,
    const ptrdiff_t geometry_stride,
    const s_t *const RSTR determinant,
    const s_t *const RSTR adjugate[9],
    const s_t *const RSTR field_shape_1d[2],
    const s_t *const RSTR field_grad_1d[2],
    const s_t *const RSTR q_weight_1d,
    const s_t direction[89][VS],
    s_t output[89][VS]
) {
  static constexpr int ND = 3;
  static constexpr int U_NS = 27;
  static constexpr int P_NS = 8;
  static constexpr int NQ1 = integer_root(NQ, ND);
  static_assert(ipow(NQ1, ND) == NQ, "NQ must be tensor-product compatible");
  static constexpr int U_NS1 = integer_root(U_NS, ND);
  static_assert(ipow(U_NS1, ND) == U_NS, "U_NS must be tensor-product compatible");
  static constexpr int P_NS1 = integer_root(P_NS, ND);
  static_assert(ipow(P_NS1, ND) == P_NS, "P_NS must be tensor-product compatible");
  s_t direction_p_value[NQ * VS];
  tensor_evaluate_value_contiguous<s_t, NQ, P_NS, VS, ND, 1>(
      ne, field_shape_1d[1], direction + 81, direction_p_value);
  s_t u0_value_coeff[NQ * VS];
  s_t u0_grad_coeff_ref[NQ * ND * VS];
  s_t u1_value_coeff[NQ * VS];
  s_t u1_grad_coeff_ref[NQ * ND * VS];
  s_t u2_value_coeff[NQ * VS];
  s_t u2_grad_coeff_ref[NQ * ND * VS];
  s_t p_value_coeff[NQ * VS];
  s_t p_grad_coeff_ref[NQ * ND * VS];
  for (int q = 0; q < NQ; ++q) {
    const int qx = q % NQ1;
    const int qy = (q / NQ1) % NQ1;
    const int qz = q / (NQ1 * NQ1);
    const s_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
    const s_t *const RSTR det_q = determinant + q * geometry_stride;
    const s_t *const RSTR adj_q0 = adjugate[0] + q * geometry_stride;
    const s_t *const RSTR adj_q1 = adjugate[1] + q * geometry_stride;
    const s_t *const RSTR adj_q2 = adjugate[2] + q * geometry_stride;
    const s_t *const RSTR adj_q3 = adjugate[3] + q * geometry_stride;
    const s_t *const RSTR adj_q4 = adjugate[4] + q * geometry_stride;
    const s_t *const RSTR adj_q5 = adjugate[5] + q * geometry_stride;
    const s_t *const RSTR adj_q6 = adjugate[6] + q * geometry_stride;
    const s_t *const RSTR adj_q7 = adjugate[7] + q * geometry_stride;
    const s_t *const RSTR adj_q8 = adjugate[8] + q * geometry_stride;
    const s_t *const RSTR direction_p_value_q = &direction_p_value[q * VS];
    s_t *const RSTR u0_value_coeff_q = &u0_value_coeff[q * VS];
    s_t *const RSTR u0_grad_coeff_ref_q0 = &u0_grad_coeff_ref[(q * ND + 0) * VS];
    s_t *const RSTR u0_grad_coeff_ref_q1 = &u0_grad_coeff_ref[(q * ND + 1) * VS];
    s_t *const RSTR u0_grad_coeff_ref_q2 = &u0_grad_coeff_ref[(q * ND + 2) * VS];
    s_t *const RSTR u1_value_coeff_q = &u1_value_coeff[q * VS];
    s_t *const RSTR u1_grad_coeff_ref_q0 = &u1_grad_coeff_ref[(q * ND + 0) * VS];
    s_t *const RSTR u1_grad_coeff_ref_q1 = &u1_grad_coeff_ref[(q * ND + 1) * VS];
    s_t *const RSTR u1_grad_coeff_ref_q2 = &u1_grad_coeff_ref[(q * ND + 2) * VS];
    s_t *const RSTR u2_value_coeff_q = &u2_value_coeff[q * VS];
    s_t *const RSTR u2_grad_coeff_ref_q0 = &u2_grad_coeff_ref[(q * ND + 0) * VS];
    s_t *const RSTR u2_grad_coeff_ref_q1 = &u2_grad_coeff_ref[(q * ND + 1) * VS];
    s_t *const RSTR u2_grad_coeff_ref_q2 = &u2_grad_coeff_ref[(q * ND + 2) * VS];
    s_t *const RSTR p_value_coeff_q = &p_value_coeff[q * VS];
    s_t *const RSTR p_grad_coeff_ref_q0 = &p_grad_coeff_ref[(q * ND + 0) * VS];
    s_t *const RSTR p_grad_coeff_ref_q1 = &p_grad_coeff_ref[(q * ND + 1) * VS];
    s_t *const RSTR p_grad_coeff_ref_q2 = &p_grad_coeff_ref[(q * ND + 2) * VS];
    #pragma omp simd
    for (int lane = 0; lane < ne; ++lane) {
      const s_t det = det_q[lane];
      const s_t adj0 = adj_q0[lane];
      const s_t adj1 = adj_q1[lane];
      const s_t adj2 = adj_q2[lane];
      const s_t adj3 = adj_q3[lane];
      const s_t adj4 = adj_q4[lane];
      const s_t adj5 = adj_q5[lane];
      const s_t adj6 = adj_q6[lane];
      const s_t adj7 = adj_q7[lane];
      const s_t adj8 = adj_q8[lane];
      const s_t p_direction = direction_p_value_q[lane];
      const s_t residual_tmp0 = -p_direction;
      const s_t grad_coeff0_0 = residual_tmp0;
      const s_t grad_coeff1_1 = residual_tmp0;
      const s_t grad_coeff2_2 = residual_tmp0;
      u0_value_coeff_q[lane] = s_t(0);
      u0_grad_coeff_ref_q0[lane] = qw * (adj0 * grad_coeff0_0);
      u0_grad_coeff_ref_q1[lane] = qw * (adj3 * grad_coeff0_0);
      u0_grad_coeff_ref_q2[lane] = qw * (adj6 * grad_coeff0_0);
      u1_value_coeff_q[lane] = s_t(0);
      u1_grad_coeff_ref_q0[lane] = qw * (adj1 * grad_coeff1_1);
      u1_grad_coeff_ref_q1[lane] = qw * (adj4 * grad_coeff1_1);
      u1_grad_coeff_ref_q2[lane] = qw * (adj7 * grad_coeff1_1);
      u2_value_coeff_q[lane] = s_t(0);
      u2_grad_coeff_ref_q0[lane] = qw * (adj2 * grad_coeff2_2);
      u2_grad_coeff_ref_q1[lane] = qw * (adj5 * grad_coeff2_2);
      u2_grad_coeff_ref_q2[lane] = qw * (adj8 * grad_coeff2_2);
      p_value_coeff_q[lane] = s_t(0);
      p_grad_coeff_ref_q0[lane] = s_t(0);
      p_grad_coeff_ref_q1[lane] = s_t(0);
      p_grad_coeff_ref_q2[lane] = s_t(0);
    }
  }
  tensor_integrate_contiguous<s_t, NQ, U_NS, VS, ND, 1>(
      ne, field_shape_1d[0], field_grad_1d[0], u0_value_coeff, u0_grad_coeff_ref, output + 0);
  tensor_integrate_contiguous<s_t, NQ, U_NS, VS, ND, 1>(
      ne, field_shape_1d[0], field_grad_1d[0], u1_value_coeff, u1_grad_coeff_ref, output + 27);
  tensor_integrate_contiguous<s_t, NQ, U_NS, VS, ND, 1>(
      ne, field_shape_1d[0], field_grad_1d[0], u2_value_coeff, u2_grad_coeff_ref, output + 54);
  tensor_integrate_contiguous<s_t, NQ, P_NS, VS, ND, 1>(
      ne, field_shape_1d[1], field_grad_1d[1], p_value_coeff, p_grad_coeff_ref, output + 81);
}

} // namespace codegen
} // namespace sfem

#endif
