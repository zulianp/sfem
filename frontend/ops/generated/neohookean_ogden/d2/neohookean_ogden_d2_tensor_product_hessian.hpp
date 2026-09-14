#ifndef NEOHOOKEAN_OGDEN_D2_TENSOR_PRODUCT_HESSIAN_HPP
#define NEOHOOKEAN_OGDEN_D2_TENSOR_PRODUCT_HESSIAN_HPP
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
static SFEM_INLINE void neohookean_ogden_d2_tensor_product_direct_hessian_tensor_product_element_matrix(
    const s_t *const RSTR badj0,
    const s_t *const RSTR badj1,
    const s_t *const RSTR badj2,
    const s_t *const RSTR badj3,
    const s_t *const RSTR bdet0,
    const s_t *const RSTR shape_1d,
    const s_t *const RSTR grad_1d,
    const s_t *const RSTR q_weight_1d,
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
    for (int trial_component = 0; trial_component < NC; ++trial_component) {
      for (int trial_shape = 0; trial_shape < NS; ++trial_shape) {
        const int trial_sx = trial_shape % NS1;
        const int trial_sy = trial_shape / NS1;
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
            const int test_sx = test_shape % NS1;
            const int test_sy = test_shape / NS1;
            const s_t test_grad_ref0 = grad_1d[qx * NS1 + test_sx] * shape_1d[qy * NS1 + test_sy];
            const s_t test_grad_ref1 = shape_1d[qx * NS1 + test_sx] * grad_1d[qy * NS1 + test_sy];
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

} // namespace codegen
} // namespace sfem

#endif
