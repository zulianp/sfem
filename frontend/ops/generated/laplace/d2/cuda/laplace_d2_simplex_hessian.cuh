#ifndef LAPLACE_D2_SIMPLEX_HESSIAN_CUH
#define LAPLACE_D2_SIMPLEX_HESSIAN_CUH
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
static __host__ __device__ __forceinline__ void laplace_d2_simplex_direct_hessian_reference_element_matrix(
    const s_t *const RSTR badj0,
    const s_t *const RSTR badj1,
    const s_t *const RSTR badj2,
    const s_t *const RSTR badj3,
    const s_t *const RSTR bdet0,
    const s_t *const RSTR grad_ref_x,
    const s_t *const RSTR grad_ref_y,
    const s_t *const RSTR q_weight,
    const s_t kappa,
    s_t *const RSTR element_matrix
) {
  static_assert(NQ > 0, "NQ must be positive");
  static_assert(NS > 0, "NS must be positive");
  static constexpr int NC = 1;
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
    s_t D[4];
    D[0] = kappa;
    D[1] = 0;
    D[3] = kappa;
    D[2] = D[1];
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
static __host__ __device__ __forceinline__ void laplace_d2_simplex_tri3_direct_hessian_element_matrix(
    const s_t *const RSTR badj0,
    const s_t *const RSTR badj1,
    const s_t *const RSTR badj2,
    const s_t *const RSTR badj3,
    const s_t *const RSTR bdet0,
    const s_t kappa,
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
  const s_t hessian_tmp0 = ((s_t(1) / s_t(2)))*idet*kappa;
  const s_t hessian_tmp1 = hessian_tmp0*(-adj_value0 - adj_value2);
  const s_t hessian_tmp2 = hessian_tmp0*(-adj_value1 - adj_value3);
  const s_t hessian_tmp3 = adj_value0*hessian_tmp1 + adj_value1*hessian_tmp2;
  const s_t hessian_tmp4 = adj_value2*hessian_tmp1 + adj_value3*hessian_tmp2;
  const s_t hessian_tmp5 = pow_2(adj_value0)*hessian_tmp0 + pow_2(adj_value1)*hessian_tmp0;
  const s_t hessian_tmp6 = adj_value0*adj_value2*hessian_tmp0 + adj_value1*adj_value3*hessian_tmp0;
  const s_t hessian_tmp7 = pow_2(adj_value2)*hessian_tmp0 + pow_2(adj_value3)*hessian_tmp0;
  element_matrix[0] = -hessian_tmp3 - hessian_tmp4;
  element_matrix[3] = hessian_tmp3;
  element_matrix[6] = hessian_tmp4;
  element_matrix[1] = -hessian_tmp5 - hessian_tmp6;
  element_matrix[4] = hessian_tmp5;
  element_matrix[7] = hessian_tmp6;
  element_matrix[2] = -hessian_tmp6 - hessian_tmp7;
  element_matrix[5] = hessian_tmp6;
  element_matrix[8] = hessian_tmp7;
}

} // namespace codegen
} // namespace sfem

#endif
