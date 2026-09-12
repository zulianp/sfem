#ifndef MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_ELASTIC_D2_TENSOR_PRODUCT_HESSIAN_HPP
#define MOONEY_RIVLIN_KELVIN_VOIGT_NEWMARK_ELASTIC_D2_TENSOR_PRODUCT_HESSIAN_HPP
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
static SFEM_INLINE void mooney_rivlin_kelvin_voigt_newmark_elastic_d2_tensor_product_direct_hessian_tensor_product_element_matrix(
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
    s_t gu_ref0 = s_t(0);
    s_t gu_ref1 = s_t(0);
    s_t gu_ref2 = s_t(0);
    s_t gu_ref3 = s_t(0);
    for (int shape = 0; shape < NS; ++shape) {
      const int state_sx = shape % NS1;
      const int state_sy = shape / NS1;
      const s_t state_grad_ref0 = grad_1d[qx * NS1 + state_sx] * shape_1d[qy * NS1 + state_sy];
      const s_t state_grad_ref1 = shape_1d[qx * NS1 + state_sx] * grad_1d[qy * NS1 + state_sy];
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
