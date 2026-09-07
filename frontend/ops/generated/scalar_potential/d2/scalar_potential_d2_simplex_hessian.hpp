#ifndef SCALAR_POTENTIAL_D2_SIMPLEX_HESSIAN_HPP
#define SCALAR_POTENTIAL_D2_SIMPLEX_HESSIAN_HPP
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

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void scalar_potential_d2_simplex_direct_hessian_reference_element_matrix(
        const scalar_t *const SFEM_RESTRICT block_jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT block_jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT block_jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT block_jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT block_jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT grad_ref_x,
        const scalar_t *const SFEM_RESTRICT grad_ref_y,
        const scalar_t *const SFEM_RESTRICT q_weight,
        const scalar_t kappa,
        scalar_t *const SFEM_RESTRICT element_matrix
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(N_SHAPE > 0, "N_SHAPE must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
    static constexpr int N_FIELD_COMPONENTS = 1;
    static constexpr int SPATIAL_DIM = 2;
    static constexpr int NDOFS = N_FIELD_COMPONENTS * N_SHAPE;
    for (int entry = 0; entry < NDOFS * NDOFS; ++entry) {
        element_matrix[entry] = scalar_t(0);
    }
    for (int q = 0; q < N_QP; ++q) {
        const scalar_t qw = q_weight[q];
        const int lane = 0;
        const ptrdiff_t geometry_offset = q * VECTOR_SIZE + lane;
        const scalar_t jacobian_adjugate_lane0 = block_jacobian_adjugate0[geometry_offset];
        const scalar_t jacobian_adjugate_lane1 = block_jacobian_adjugate1[geometry_offset];
        const scalar_t jacobian_adjugate_lane2 = block_jacobian_adjugate2[geometry_offset];
        const scalar_t jacobian_adjugate_lane3 = block_jacobian_adjugate3[geometry_offset];
        const scalar_t jacobian_determinant_lane0 = block_jacobian_determinant0[geometry_offset];
        const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
        for (int trial_component = 0; trial_component < N_FIELD_COMPONENTS; ++trial_component) {
            for (int trial_shape = 0; trial_shape < N_SHAPE; ++trial_shape) {
                const scalar_t trial_grad_ref0 = grad_ref_x[q * N_SHAPE + trial_shape];
                const scalar_t trial_grad_ref1 = grad_ref_y[q * N_SHAPE + trial_shape];
                scalar_t trial_grad[N_FIELD_COMPONENTS * SPATIAL_DIM];
                for (int i = 0; i < N_FIELD_COMPONENTS * SPATIAL_DIM; ++i) {
                    trial_grad[i] = scalar_t(0);
                }
                trial_grad[trial_component * SPATIAL_DIM + 0] = (trial_grad_ref0 * jacobian_adjugate_lane0 + trial_grad_ref1 * jacobian_adjugate_lane2) * inv_jacobian_determinant;
                trial_grad[trial_component * SPATIAL_DIM + 1] = (trial_grad_ref0 * jacobian_adjugate_lane1 + trial_grad_ref1 * jacobian_adjugate_lane3) * inv_jacobian_determinant;
                scalar_t material[N_FIELD_COMPONENTS * SPATIAL_DIM];
                material[0] = kappa*trial_grad[0];
                material[1] = kappa*trial_grad[1];
                for (int test_component = 0; test_component < N_FIELD_COMPONENTS; ++test_component) {
                    for (int test_shape = 0; test_shape < N_SHAPE; ++test_shape) {
                        const scalar_t test_grad_ref0 = grad_ref_x[q * N_SHAPE + test_shape];
                        const scalar_t test_grad_ref1 = grad_ref_y[q * N_SHAPE + test_shape];
                        scalar_t entry = scalar_t(0);
                        entry += test_grad_ref0 * qw * (material[test_component * SPATIAL_DIM + 0] * jacobian_adjugate_lane0 + material[test_component * SPATIAL_DIM + 1] * jacobian_adjugate_lane1);
                        entry += test_grad_ref1 * qw * (material[test_component * SPATIAL_DIM + 0] * jacobian_adjugate_lane2 + material[test_component * SPATIAL_DIM + 1] * jacobian_adjugate_lane3);
                        const int row = test_component * N_SHAPE + test_shape;
                        const int col = trial_component * N_SHAPE + trial_shape;
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
