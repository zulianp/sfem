#ifndef LAPLACE_D2_TENSOR_PRODUCT_HESSIAN_HPP
#define LAPLACE_D2_TENSOR_PRODUCT_HESSIAN_HPP
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
static SFEM_INLINE void laplace_d2_tensor_product_direct_hessian_tensor_product_element_matrix(
        const s_t *const SFEM_RESTRICT block_jacobian_adjugate0,
        const s_t *const SFEM_RESTRICT block_jacobian_adjugate1,
        const s_t *const SFEM_RESTRICT block_jacobian_adjugate2,
        const s_t *const SFEM_RESTRICT block_jacobian_adjugate3,
        const s_t *const SFEM_RESTRICT block_jacobian_determinant0,
        const s_t *const SFEM_RESTRICT shape_1d,
        const s_t *const SFEM_RESTRICT grad_1d,
        const s_t *const SFEM_RESTRICT q_weight_1d,
        const s_t kappa,
        s_t *const SFEM_RESTRICT element_matrix
) {
    static_assert(NQ > 0, "NQ must be positive");
    static_assert(NS > 0, "NS must be positive");
    static_assert(VS > 0, "VS must be positive");
    static constexpr int NC = 1;
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
        const s_t jacobian_adjugate_lane0 = block_jacobian_adjugate0[goff];
        const s_t jacobian_adjugate_lane1 = block_jacobian_adjugate1[goff];
        const s_t jacobian_adjugate_lane2 = block_jacobian_adjugate2[goff];
        const s_t jacobian_adjugate_lane3 = block_jacobian_adjugate3[goff];
        const s_t jacobian_determinant_lane0 = block_jacobian_determinant0[goff];
        const s_t idet = s_t(1) / jacobian_determinant_lane0;
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
                trial_grad[trial_component * ND + 0] = (trial_grad_ref0 * jacobian_adjugate_lane0 + trial_grad_ref1 * jacobian_adjugate_lane2) * idet;
                trial_grad[trial_component * ND + 1] = (trial_grad_ref0 * jacobian_adjugate_lane1 + trial_grad_ref1 * jacobian_adjugate_lane3) * idet;
                s_t material[NC * ND];
                material[0] = kappa*trial_grad[0];
                material[1] = kappa*trial_grad[1];
                for (int test_component = 0; test_component < NC; ++test_component) {
                    for (int test_shape = 0; test_shape < NS; ++test_shape) {
                        const int test_sx = test_shape % NS1;
                        const int test_sy = test_shape / NS1;
                        const s_t test_grad_ref0 = grad_1d[qx * NS1 + test_sx] * shape_1d[qy * NS1 + test_sy];
                        const s_t test_grad_ref1 = shape_1d[qx * NS1 + test_sx] * grad_1d[qy * NS1 + test_sy];
                        s_t entry = s_t(0);
                        entry += test_grad_ref0 * qw * (material[test_component * ND + 0] * jacobian_adjugate_lane0 + material[test_component * ND + 1] * jacobian_adjugate_lane1);
                        entry += test_grad_ref1 * qw * (material[test_component * ND + 0] * jacobian_adjugate_lane2 + material[test_component * ND + 1] * jacobian_adjugate_lane3);
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
