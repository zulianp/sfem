#ifndef LINEAR_ELASTICITY_D3_TENSOR_PRODUCT_HESSIAN_HPP
#define LINEAR_ELASTICITY_D3_TENSOR_PRODUCT_HESSIAN_HPP
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
typedef double    real_t;
typedef ptrdiff_t idx_t;
typedef double    geom_t;
#endif
namespace sfem {
    namespace codegen {

        template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
        static SFEM_INLINE void linear_elasticity_d3_tensor_product_direct_hessian_tensor_product_element_matrix(
                const scalar_t *const SFEM_RESTRICT block_jacobian_adjugate0,
                const scalar_t *const SFEM_RESTRICT block_jacobian_adjugate1,
                const scalar_t *const SFEM_RESTRICT block_jacobian_adjugate2,
                const scalar_t *const SFEM_RESTRICT block_jacobian_adjugate3,
                const scalar_t *const SFEM_RESTRICT block_jacobian_adjugate4,
                const scalar_t *const SFEM_RESTRICT block_jacobian_adjugate5,
                const scalar_t *const SFEM_RESTRICT block_jacobian_adjugate6,
                const scalar_t *const SFEM_RESTRICT block_jacobian_adjugate7,
                const scalar_t *const SFEM_RESTRICT block_jacobian_adjugate8,
                const scalar_t *const SFEM_RESTRICT block_jacobian_determinant0,
                const scalar_t *const SFEM_RESTRICT shape_1d,
                const scalar_t *const SFEM_RESTRICT grad_1d,
                const scalar_t *const SFEM_RESTRICT q_weight_1d,
                const scalar_t                      lmbda,
                const scalar_t                      mu,
                scalar_t *const SFEM_RESTRICT       element_matrix) {
            static_assert(N_QP > 0, "N_QP must be positive");
            static_assert(N_SHAPE > 0, "N_SHAPE must be positive");
            static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
            static constexpr int DIM        = 3;
            static constexpr int NDOFS      = DIM * N_SHAPE;
            static constexpr int N_QP_1D    = integer_root(N_QP, 3);
            static constexpr int N_SHAPE_1D = integer_root(N_SHAPE, 3);
            static_assert(ipow(N_QP_1D, 3) == N_QP, "N_QP must be tensor-product compatible");
            static_assert(ipow(N_SHAPE_1D, 3) == N_SHAPE, "N_SHAPE must be tensor-product compatible");
            for (int entry = 0; entry < NDOFS * NDOFS; ++entry) {
                element_matrix[entry] = scalar_t(0);
            }
            for (int q = 0; q < N_QP; ++q) {
                const int       qx                         = q % N_QP_1D;
                const int       qy                         = (q / N_QP_1D) % N_QP_1D;
                const int       qz                         = q / (N_QP_1D * N_QP_1D);
                const scalar_t  qw                         = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
                const int       lane                       = 0;
                const ptrdiff_t geometry_offset            = q * VECTOR_SIZE + lane;
                const scalar_t  jacobian_adjugate_lane0    = block_jacobian_adjugate0[geometry_offset];
                const scalar_t  jacobian_adjugate_lane1    = block_jacobian_adjugate1[geometry_offset];
                const scalar_t  jacobian_adjugate_lane2    = block_jacobian_adjugate2[geometry_offset];
                const scalar_t  jacobian_adjugate_lane3    = block_jacobian_adjugate3[geometry_offset];
                const scalar_t  jacobian_adjugate_lane4    = block_jacobian_adjugate4[geometry_offset];
                const scalar_t  jacobian_adjugate_lane5    = block_jacobian_adjugate5[geometry_offset];
                const scalar_t  jacobian_adjugate_lane6    = block_jacobian_adjugate6[geometry_offset];
                const scalar_t  jacobian_adjugate_lane7    = block_jacobian_adjugate7[geometry_offset];
                const scalar_t  jacobian_adjugate_lane8    = block_jacobian_adjugate8[geometry_offset];
                const scalar_t  jacobian_determinant_lane0 = block_jacobian_determinant0[geometry_offset];
                const scalar_t  inv_jacobian_determinant   = scalar_t(1) / jacobian_determinant_lane0;
                for (int trial_component = 0; trial_component < DIM; ++trial_component) {
                    for (int trial_shape = 0; trial_shape < N_SHAPE; ++trial_shape) {
                        const int      trial_sx        = trial_shape % N_SHAPE_1D;
                        const int      trial_sy        = (trial_shape / N_SHAPE_1D) % N_SHAPE_1D;
                        const int      trial_sz        = trial_shape / (N_SHAPE_1D * N_SHAPE_1D);
                        const scalar_t trial_grad_ref0 = grad_1d[qx * N_SHAPE_1D + trial_sx] *
                                                         shape_1d[qy * N_SHAPE_1D + trial_sy] *
                                                         shape_1d[qz * N_SHAPE_1D + trial_sz];
                        const scalar_t trial_grad_ref1 = shape_1d[qx * N_SHAPE_1D + trial_sx] *
                                                         grad_1d[qy * N_SHAPE_1D + trial_sy] *
                                                         shape_1d[qz * N_SHAPE_1D + trial_sz];
                        const scalar_t trial_grad_ref2 = shape_1d[qx * N_SHAPE_1D + trial_sx] *
                                                         shape_1d[qy * N_SHAPE_1D + trial_sy] *
                                                         grad_1d[qz * N_SHAPE_1D + trial_sz];
                        scalar_t trial_grad[DIM * DIM];
                        for (int i = 0; i < DIM * DIM; ++i) {
                            trial_grad[i] = scalar_t(0);
                        }
                        trial_grad[trial_component * DIM + 0] =
                                (trial_grad_ref0 * jacobian_adjugate_lane0 + trial_grad_ref1 * jacobian_adjugate_lane3 +
                                 trial_grad_ref2 * jacobian_adjugate_lane6) *
                                inv_jacobian_determinant;
                        trial_grad[trial_component * DIM + 1] =
                                (trial_grad_ref0 * jacobian_adjugate_lane1 + trial_grad_ref1 * jacobian_adjugate_lane4 +
                                 trial_grad_ref2 * jacobian_adjugate_lane7) *
                                inv_jacobian_determinant;
                        trial_grad[trial_component * DIM + 2] =
                                (trial_grad_ref0 * jacobian_adjugate_lane2 + trial_grad_ref1 * jacobian_adjugate_lane5 +
                                 trial_grad_ref2 * jacobian_adjugate_lane8) *
                                inv_jacobian_determinant;
                        scalar_t       material[DIM * DIM];
                        const scalar_t weak_hess_tmp0 = scalar_t(2) * trial_grad[0];
                        const scalar_t weak_hess_tmp1 = scalar_t(2) * trial_grad[4];
                        const scalar_t weak_hess_tmp2 = scalar_t(2) * trial_grad[8];
                        const scalar_t weak_hess_tmp3 =
                                ((scalar_t(1) / scalar_t(2))) * lmbda * (weak_hess_tmp0 + weak_hess_tmp1 + weak_hess_tmp2);
                        const scalar_t weak_hess_tmp4 = mu * (trial_grad[1] + trial_grad[3]);
                        const scalar_t weak_hess_tmp5 = mu * (trial_grad[2] + trial_grad[6]);
                        const scalar_t weak_hess_tmp6 = mu * (trial_grad[5] + trial_grad[7]);
                        material[0]                   = mu * weak_hess_tmp0 + weak_hess_tmp3;
                        material[1]                   = weak_hess_tmp4;
                        material[2]                   = weak_hess_tmp5;
                        material[3]                   = weak_hess_tmp4;
                        material[4]                   = mu * weak_hess_tmp1 + weak_hess_tmp3;
                        material[5]                   = weak_hess_tmp6;
                        material[6]                   = weak_hess_tmp5;
                        material[7]                   = weak_hess_tmp6;
                        material[8]                   = mu * weak_hess_tmp2 + weak_hess_tmp3;
                        for (int test_component = 0; test_component < DIM; ++test_component) {
                            for (int test_shape = 0; test_shape < N_SHAPE; ++test_shape) {
                                const int      test_sx        = test_shape % N_SHAPE_1D;
                                const int      test_sy        = (test_shape / N_SHAPE_1D) % N_SHAPE_1D;
                                const int      test_sz        = test_shape / (N_SHAPE_1D * N_SHAPE_1D);
                                const scalar_t test_grad_ref0 = grad_1d[qx * N_SHAPE_1D + test_sx] *
                                                                shape_1d[qy * N_SHAPE_1D + test_sy] *
                                                                shape_1d[qz * N_SHAPE_1D + test_sz];
                                const scalar_t test_grad_ref1 = shape_1d[qx * N_SHAPE_1D + test_sx] *
                                                                grad_1d[qy * N_SHAPE_1D + test_sy] *
                                                                shape_1d[qz * N_SHAPE_1D + test_sz];
                                const scalar_t test_grad_ref2 = shape_1d[qx * N_SHAPE_1D + test_sx] *
                                                                shape_1d[qy * N_SHAPE_1D + test_sy] *
                                                                grad_1d[qz * N_SHAPE_1D + test_sz];
                                scalar_t entry = scalar_t(0);
                                entry += test_grad_ref0 * qw *
                                         (material[test_component * DIM + 0] * jacobian_adjugate_lane0 +
                                          material[test_component * DIM + 1] * jacobian_adjugate_lane1 +
                                          material[test_component * DIM + 2] * jacobian_adjugate_lane2);
                                entry += test_grad_ref1 * qw *
                                         (material[test_component * DIM + 0] * jacobian_adjugate_lane3 +
                                          material[test_component * DIM + 1] * jacobian_adjugate_lane4 +
                                          material[test_component * DIM + 2] * jacobian_adjugate_lane5);
                                entry += test_grad_ref2 * qw *
                                         (material[test_component * DIM + 0] * jacobian_adjugate_lane6 +
                                          material[test_component * DIM + 1] * jacobian_adjugate_lane7 +
                                          material[test_component * DIM + 2] * jacobian_adjugate_lane8);
                                const int row = test_component * N_SHAPE + test_shape;
                                const int col = trial_component * N_SHAPE + trial_shape;
                                element_matrix[row * NDOFS + col] += entry;
                            }
                        }
                    }
                }
            }
        }

    }  // namespace codegen
}  // namespace sfem

#endif
