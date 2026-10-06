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
#ifndef SFEM_GENERATED_SCALAR_T
#define SFEM_GENERATED_SCALAR_T
typedef double real_t;
typedef ptrdiff_t idx_t;
typedef double geom_t;
#endif
namespace sfem {
namespace codegen {

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void modified_mooney_rivlin_d3_tensor_product_objective_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate4,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate5,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate6,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate7,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate8,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t c1,
        const scalar_t c2,
        const scalar_t kappa,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 3],
        scalar_t *const SFEM_RESTRICT value
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
    static constexpr int N_QP_1D = integer_root(N_QP, 3);
    static constexpr int N_SHAPE_1D = integer_root(N_SHAPE, 3);
    static_assert(ipow(N_QP_1D, 3) == N_QP, "N_QP must be tensor-product compatible");
    static_assert(ipow(N_SHAPE_1D, 3) == N_SHAPE, "N_SHAPE must be tensor-product compatible");
    scalar_t grad_u_ref_q[N_QP * 9 * VECTOR_SIZE];
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, shape_1d, grad_1d, u_streams, 0, &grad_u_ref_q[0 * N_QP * 3 * VECTOR_SIZE]);
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, shape_1d, grad_1d, u_streams, 1, &grad_u_ref_q[1 * N_QP * 3 * VECTOR_SIZE]);
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, shape_1d, grad_1d, u_streams, 2, &grad_u_ref_q[2 * N_QP * 3 * VECTOR_SIZE]);
    for (int q = 0; q < N_QP; ++q) {
        const int qx = q % N_QP_1D;
        const int qy = (q / N_QP_1D) % N_QP_1D;
        const int qz = q / (N_QP_1D * N_QP_1D);
        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const scalar_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const scalar_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const scalar_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const scalar_t jacobian_adjugate_lane4 = jacobian_adjugate4[geometry_offset];
            const scalar_t jacobian_adjugate_lane5 = jacobian_adjugate5[geometry_offset];
            const scalar_t jacobian_adjugate_lane6 = jacobian_adjugate6[geometry_offset];
            const scalar_t jacobian_adjugate_lane7 = jacobian_adjugate7[geometry_offset];
            const scalar_t jacobian_adjugate_lane8 = jacobian_adjugate8[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            scalar_t grad_u_ref[9];
            grad_u_ref[0] = grad_u_ref_q[((0 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[1] = grad_u_ref_q[((0 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[2] = grad_u_ref_q[((0 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            grad_u_ref[3] = grad_u_ref_q[((1 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[4] = grad_u_ref_q[((1 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[5] = grad_u_ref_q[((1 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            grad_u_ref[6] = grad_u_ref_q[((2 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[7] = grad_u_ref_q[((2 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[8] = grad_u_ref_q[((2 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            scalar_t grad_u[9];
            const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
            grad_u[0] = (grad_u_ref[0] * jacobian_adjugate_lane0 + grad_u_ref[1] * jacobian_adjugate_lane3 + grad_u_ref[2] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            grad_u[1] = (grad_u_ref[0] * jacobian_adjugate_lane1 + grad_u_ref[1] * jacobian_adjugate_lane4 + grad_u_ref[2] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            grad_u[2] = (grad_u_ref[0] * jacobian_adjugate_lane2 + grad_u_ref[1] * jacobian_adjugate_lane5 + grad_u_ref[2] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            grad_u[3] = (grad_u_ref[3] * jacobian_adjugate_lane0 + grad_u_ref[4] * jacobian_adjugate_lane3 + grad_u_ref[5] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            grad_u[4] = (grad_u_ref[3] * jacobian_adjugate_lane1 + grad_u_ref[4] * jacobian_adjugate_lane4 + grad_u_ref[5] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            grad_u[5] = (grad_u_ref[3] * jacobian_adjugate_lane2 + grad_u_ref[4] * jacobian_adjugate_lane5 + grad_u_ref[5] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            grad_u[6] = (grad_u_ref[6] * jacobian_adjugate_lane0 + grad_u_ref[7] * jacobian_adjugate_lane3 + grad_u_ref[8] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            grad_u[7] = (grad_u_ref[6] * jacobian_adjugate_lane1 + grad_u_ref[7] * jacobian_adjugate_lane4 + grad_u_ref[8] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            grad_u[8] = (grad_u_ref[6] * jacobian_adjugate_lane2 + grad_u_ref[7] * jacobian_adjugate_lane5 + grad_u_ref[8] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
        const scalar_t weak_obj_tmp0 = grad_u[8] + scalar_t(1);
        const scalar_t weak_obj_tmp1 = grad_u[4] + scalar_t(1);
        const scalar_t weak_obj_tmp2 = grad_u[0] + scalar_t(1);
        const scalar_t weak_obj_tmp3 = -grad_u[1]*grad_u[3]*weak_obj_tmp0 + grad_u[1]*grad_u[5]*grad_u[6] + grad_u[2]*grad_u[3]*grad_u[7] - grad_u[2]*grad_u[6]*weak_obj_tmp1 - grad_u[5]*grad_u[7]*weak_obj_tmp2 + weak_obj_tmp0*weak_obj_tmp1*weak_obj_tmp2;
        const scalar_t weak_obj_tmp4 = pow_2(grad_u[1]) + pow_2(grad_u[7]) + pow_2(weak_obj_tmp1);
        const scalar_t weak_obj_tmp5 = pow_2(grad_u[2]) + pow_2(grad_u[5]) + pow_2(weak_obj_tmp0);
        const scalar_t weak_obj_tmp6 = pow_2(grad_u[3]) + pow_2(grad_u[6]) + pow_2(weak_obj_tmp2);
        const scalar_t weak_obj_tmp7 = weak_obj_tmp4 + weak_obj_tmp5 + weak_obj_tmp6;
        value[lane] += qw * jacobian_determinant_lane0 * (c1*(scalar_t(-3) + weak_obj_tmp7/pow(weak_obj_tmp3, (scalar_t(2) / scalar_t(3)))) + c2*(scalar_t(-3) + (-(scalar_t(1) / scalar_t(2))*pow_2(weak_obj_tmp4) - (scalar_t(1) / scalar_t(2))*pow_2(weak_obj_tmp5) - (scalar_t(1) / scalar_t(2))*pow_2(weak_obj_tmp6) + ((scalar_t(1) / scalar_t(2)))*pow_2(weak_obj_tmp7) - pow_2(grad_u[1]*grad_u[2] + grad_u[5]*weak_obj_tmp1 + grad_u[7]*weak_obj_tmp0) - pow_2(grad_u[1]*weak_obj_tmp2 + grad_u[3]*weak_obj_tmp1 + grad_u[6]*grad_u[7]) - pow_2(grad_u[2]*weak_obj_tmp2 + grad_u[3]*grad_u[5] + grad_u[6]*weak_obj_tmp0))/pow(weak_obj_tmp3, (scalar_t(4) / scalar_t(3)))) + ((scalar_t(1) / scalar_t(2)))*kappa*pow_2(log(weak_obj_tmp3)));
        }
    }
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void modified_mooney_rivlin_d3_tensor_product_gradient_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate4,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate5,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate6,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate7,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate8,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t c1,
        const scalar_t c2,
        const scalar_t kappa,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 3],
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 3]
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
    static constexpr int N_QP_1D = integer_root(N_QP, 3);
    static constexpr int N_SHAPE_1D = integer_root(N_SHAPE, 3);
    static_assert(ipow(N_QP_1D, 3) == N_QP, "N_QP must be tensor-product compatible");
    static_assert(ipow(N_SHAPE_1D, 3) == N_SHAPE, "N_SHAPE must be tensor-product compatible");
    scalar_t grad_u_ref_q[N_QP * 9 * VECTOR_SIZE];
    scalar_t loperand_q[N_QP * 9 * VECTOR_SIZE];
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, shape_1d, grad_1d, u_streams, 0, &grad_u_ref_q[0 * N_QP * 3 * VECTOR_SIZE]);
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, shape_1d, grad_1d, u_streams, 1, &grad_u_ref_q[1 * N_QP * 3 * VECTOR_SIZE]);
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, shape_1d, grad_1d, u_streams, 2, &grad_u_ref_q[2 * N_QP * 3 * VECTOR_SIZE]);
    for (int q = 0; q < N_QP; ++q) {
        const int qx = q % N_QP_1D;
        const int qy = (q / N_QP_1D) % N_QP_1D;
        const int qz = q / (N_QP_1D * N_QP_1D);
        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const scalar_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const scalar_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const scalar_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const scalar_t jacobian_adjugate_lane4 = jacobian_adjugate4[geometry_offset];
            const scalar_t jacobian_adjugate_lane5 = jacobian_adjugate5[geometry_offset];
            const scalar_t jacobian_adjugate_lane6 = jacobian_adjugate6[geometry_offset];
            const scalar_t jacobian_adjugate_lane7 = jacobian_adjugate7[geometry_offset];
            const scalar_t jacobian_adjugate_lane8 = jacobian_adjugate8[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            scalar_t grad_u_ref[9];
            grad_u_ref[0] = grad_u_ref_q[((0 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[1] = grad_u_ref_q[((0 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[2] = grad_u_ref_q[((0 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            grad_u_ref[3] = grad_u_ref_q[((1 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[4] = grad_u_ref_q[((1 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[5] = grad_u_ref_q[((1 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            grad_u_ref[6] = grad_u_ref_q[((2 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[7] = grad_u_ref_q[((2 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[8] = grad_u_ref_q[((2 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            scalar_t grad_u[9];
            const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
            grad_u[0] = (grad_u_ref[0] * jacobian_adjugate_lane0 + grad_u_ref[1] * jacobian_adjugate_lane3 + grad_u_ref[2] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            grad_u[1] = (grad_u_ref[0] * jacobian_adjugate_lane1 + grad_u_ref[1] * jacobian_adjugate_lane4 + grad_u_ref[2] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            grad_u[2] = (grad_u_ref[0] * jacobian_adjugate_lane2 + grad_u_ref[1] * jacobian_adjugate_lane5 + grad_u_ref[2] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            grad_u[3] = (grad_u_ref[3] * jacobian_adjugate_lane0 + grad_u_ref[4] * jacobian_adjugate_lane3 + grad_u_ref[5] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            grad_u[4] = (grad_u_ref[3] * jacobian_adjugate_lane1 + grad_u_ref[4] * jacobian_adjugate_lane4 + grad_u_ref[5] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            grad_u[5] = (grad_u_ref[3] * jacobian_adjugate_lane2 + grad_u_ref[4] * jacobian_adjugate_lane5 + grad_u_ref[5] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            grad_u[6] = (grad_u_ref[6] * jacobian_adjugate_lane0 + grad_u_ref[7] * jacobian_adjugate_lane3 + grad_u_ref[8] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            grad_u[7] = (grad_u_ref[6] * jacobian_adjugate_lane1 + grad_u_ref[7] * jacobian_adjugate_lane4 + grad_u_ref[8] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            grad_u[8] = (grad_u_ref[6] * jacobian_adjugate_lane2 + grad_u_ref[7] * jacobian_adjugate_lane5 + grad_u_ref[8] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            scalar_t loperand[9];
        scalar_t material[9];
        const scalar_t weak_mat_tmp0 = grad_u[5]*grad_u[7];
        const scalar_t weak_mat_tmp1 = grad_u[4] + scalar_t(1);
        const scalar_t weak_mat_tmp2 = grad_u[8] + scalar_t(1);
        const scalar_t weak_mat_tmp3 = grad_u[3]*weak_mat_tmp2;
        const scalar_t weak_mat_tmp4 = grad_u[6]*weak_mat_tmp1;
        const scalar_t weak_mat_tmp5 = grad_u[0] + scalar_t(1);
        const scalar_t weak_mat_tmp6 = grad_u[1]*grad_u[5]*grad_u[6] - grad_u[1]*weak_mat_tmp3 + grad_u[2]*grad_u[3]*grad_u[7] - grad_u[2]*weak_mat_tmp4 - weak_mat_tmp0*weak_mat_tmp5 + weak_mat_tmp1*weak_mat_tmp2*weak_mat_tmp5;
        const scalar_t weak_mat_tmp7 = kappa*log(weak_mat_tmp6)/weak_mat_tmp6;
        const scalar_t weak_mat_tmp8 = pow(weak_mat_tmp6, (scalar_t(-2) / scalar_t(3)));
        const scalar_t weak_mat_tmp9 = scalar_t(2)*weak_mat_tmp5;
        const scalar_t weak_mat_tmp10 = weak_mat_tmp1*weak_mat_tmp2;
        const scalar_t weak_mat_tmp11 = pow_2(grad_u[3]) + pow_2(grad_u[6]) + pow_2(weak_mat_tmp5);
        const scalar_t weak_mat_tmp12 = pow_2(grad_u[1]) + pow_2(grad_u[7]) + pow_2(weak_mat_tmp1);
        const scalar_t weak_mat_tmp13 = pow_2(grad_u[2]) + pow_2(grad_u[5]) + pow_2(weak_mat_tmp2);
        const scalar_t weak_mat_tmp14 = weak_mat_tmp11 + weak_mat_tmp12 + weak_mat_tmp13;
        const scalar_t weak_mat_tmp15 = weak_mat_tmp14/pow(weak_mat_tmp6, (scalar_t(5) / scalar_t(3)));
        const scalar_t weak_mat_tmp16 = pow(weak_mat_tmp6, (scalar_t(-4) / scalar_t(3)));
        const scalar_t weak_mat_tmp17 = grad_u[1]*weak_mat_tmp5 + grad_u[3]*weak_mat_tmp1 + grad_u[6]*grad_u[7];
        const scalar_t weak_mat_tmp18 = scalar_t(2)*grad_u[1];
        const scalar_t weak_mat_tmp19 = grad_u[2]*weak_mat_tmp5 + grad_u[3]*grad_u[5] + grad_u[6]*weak_mat_tmp2;
        const scalar_t weak_mat_tmp20 = scalar_t(2)*grad_u[2];
        const scalar_t weak_mat_tmp21 = grad_u[1]*grad_u[2] + grad_u[5]*weak_mat_tmp1 + grad_u[7]*weak_mat_tmp2;
        const scalar_t weak_mat_tmp22 = (-(scalar_t(1) / scalar_t(2))*pow_2(weak_mat_tmp11) - (scalar_t(1) / scalar_t(2))*pow_2(weak_mat_tmp12) - (scalar_t(1) / scalar_t(2))*pow_2(weak_mat_tmp13) + ((scalar_t(1) / scalar_t(2)))*pow_2(weak_mat_tmp14) - pow_2(weak_mat_tmp17) - pow_2(weak_mat_tmp19) - pow_2(weak_mat_tmp21))/pow(weak_mat_tmp6, (scalar_t(7) / scalar_t(3)));
        const scalar_t weak_mat_tmp23 = grad_u[5]*grad_u[6];
        const scalar_t weak_mat_tmp24 = grad_u[3]*grad_u[7];
        const scalar_t weak_mat_tmp25 = grad_u[1]*weak_mat_tmp2;
        const scalar_t weak_mat_tmp26 = scalar_t(2)*grad_u[3];
        const scalar_t weak_mat_tmp27 = grad_u[2]*grad_u[7];
        const scalar_t weak_mat_tmp28 = scalar_t(2)*grad_u[5];
        const scalar_t weak_mat_tmp29 = scalar_t(2)*weak_mat_tmp1;
        const scalar_t weak_mat_tmp30 = grad_u[2]*grad_u[6];
        const scalar_t weak_mat_tmp31 = weak_mat_tmp2*weak_mat_tmp5;
        const scalar_t weak_mat_tmp32 = grad_u[1]*grad_u[6];
        const scalar_t weak_mat_tmp33 = grad_u[1]*grad_u[5];
        const scalar_t weak_mat_tmp34 = scalar_t(2)*grad_u[6];
        const scalar_t weak_mat_tmp35 = scalar_t(2)*grad_u[7];
        const scalar_t weak_mat_tmp36 = scalar_t(2)*weak_mat_tmp2;
        const scalar_t weak_mat_tmp37 = grad_u[2]*grad_u[3];
        const scalar_t weak_mat_tmp38 = grad_u[1]*grad_u[3];
        const scalar_t weak_mat_tmp39 = weak_mat_tmp1*weak_mat_tmp5;
        material[0] = c1*(weak_mat_tmp15*(((scalar_t(2) / scalar_t(3)))*weak_mat_tmp0 - (scalar_t(2) / scalar_t(3))*weak_mat_tmp10) + weak_mat_tmp8*weak_mat_tmp9) + c2*(weak_mat_tmp16*(-weak_mat_tmp11*weak_mat_tmp9 + scalar_t(2)*weak_mat_tmp14*weak_mat_tmp5 - weak_mat_tmp17*weak_mat_tmp18 - weak_mat_tmp19*weak_mat_tmp20) + weak_mat_tmp22*(((scalar_t(4) / scalar_t(3)))*weak_mat_tmp0 - (scalar_t(4) / scalar_t(3))*weak_mat_tmp10)) + weak_mat_tmp7*(-weak_mat_tmp0 + weak_mat_tmp1*weak_mat_tmp2);
        material[1] = c1*(weak_mat_tmp15*(-(scalar_t(2) / scalar_t(3))*weak_mat_tmp23 + ((scalar_t(2) / scalar_t(3)))*weak_mat_tmp3) + weak_mat_tmp18*weak_mat_tmp8) + c2*(weak_mat_tmp16*(scalar_t(2)*grad_u[1]*weak_mat_tmp14 - weak_mat_tmp12*weak_mat_tmp18 - weak_mat_tmp17*weak_mat_tmp9 - weak_mat_tmp20*weak_mat_tmp21) + weak_mat_tmp22*(-(scalar_t(4) / scalar_t(3))*weak_mat_tmp23 + ((scalar_t(4) / scalar_t(3)))*weak_mat_tmp3)) + weak_mat_tmp7*(grad_u[5]*grad_u[6] - weak_mat_tmp3);
        material[2] = c1*(weak_mat_tmp15*(((scalar_t(2) / scalar_t(3)))*grad_u[6]*weak_mat_tmp1 - (scalar_t(2) / scalar_t(3))*weak_mat_tmp24) + weak_mat_tmp20*weak_mat_tmp8) + c2*(weak_mat_tmp16*(scalar_t(2)*grad_u[2]*weak_mat_tmp14 - weak_mat_tmp13*weak_mat_tmp20 - weak_mat_tmp18*weak_mat_tmp21 - weak_mat_tmp19*weak_mat_tmp9) + weak_mat_tmp22*(((scalar_t(4) / scalar_t(3)))*grad_u[6]*weak_mat_tmp1 - (scalar_t(4) / scalar_t(3))*weak_mat_tmp24)) + weak_mat_tmp7*(weak_mat_tmp24 - weak_mat_tmp4);
        material[3] = c1*(weak_mat_tmp15*(((scalar_t(2) / scalar_t(3)))*weak_mat_tmp25 - (scalar_t(2) / scalar_t(3))*weak_mat_tmp27) + weak_mat_tmp26*weak_mat_tmp8) + c2*(weak_mat_tmp16*(scalar_t(2)*grad_u[3]*weak_mat_tmp14 - weak_mat_tmp11*weak_mat_tmp26 - weak_mat_tmp17*weak_mat_tmp29 - weak_mat_tmp19*weak_mat_tmp28) + weak_mat_tmp22*(((scalar_t(4) / scalar_t(3)))*weak_mat_tmp25 - (scalar_t(4) / scalar_t(3))*weak_mat_tmp27)) + weak_mat_tmp7*(grad_u[2]*grad_u[7] - weak_mat_tmp25);
        material[4] = c1*(weak_mat_tmp15*(((scalar_t(2) / scalar_t(3)))*weak_mat_tmp30 - (scalar_t(2) / scalar_t(3))*weak_mat_tmp31) + weak_mat_tmp29*weak_mat_tmp8) + c2*(weak_mat_tmp16*(scalar_t(2)*weak_mat_tmp1*weak_mat_tmp14 - weak_mat_tmp12*weak_mat_tmp29 - weak_mat_tmp17*weak_mat_tmp26 - weak_mat_tmp21*weak_mat_tmp28) + weak_mat_tmp22*(((scalar_t(4) / scalar_t(3)))*weak_mat_tmp30 - (scalar_t(4) / scalar_t(3))*weak_mat_tmp31)) + weak_mat_tmp7*(weak_mat_tmp2*weak_mat_tmp5 - weak_mat_tmp30);
        material[5] = c1*(weak_mat_tmp15*(((scalar_t(2) / scalar_t(3)))*grad_u[7]*weak_mat_tmp5 - (scalar_t(2) / scalar_t(3))*weak_mat_tmp32) + weak_mat_tmp28*weak_mat_tmp8) + c2*(weak_mat_tmp16*(scalar_t(2)*grad_u[5]*weak_mat_tmp14 - weak_mat_tmp13*weak_mat_tmp28 - weak_mat_tmp19*weak_mat_tmp26 - weak_mat_tmp21*weak_mat_tmp29) + weak_mat_tmp22*(((scalar_t(4) / scalar_t(3)))*grad_u[7]*weak_mat_tmp5 - (scalar_t(4) / scalar_t(3))*weak_mat_tmp32)) + weak_mat_tmp7*(-grad_u[7]*weak_mat_tmp5 + weak_mat_tmp32);
        material[6] = c1*(weak_mat_tmp15*(((scalar_t(2) / scalar_t(3)))*grad_u[2]*weak_mat_tmp1 - (scalar_t(2) / scalar_t(3))*weak_mat_tmp33) + weak_mat_tmp34*weak_mat_tmp8) + c2*(weak_mat_tmp16*(scalar_t(2)*grad_u[6]*weak_mat_tmp14 - weak_mat_tmp11*weak_mat_tmp34 - weak_mat_tmp17*weak_mat_tmp35 - weak_mat_tmp19*weak_mat_tmp36) + weak_mat_tmp22*(((scalar_t(4) / scalar_t(3)))*grad_u[2]*weak_mat_tmp1 - (scalar_t(4) / scalar_t(3))*weak_mat_tmp33)) + weak_mat_tmp7*(-grad_u[2]*weak_mat_tmp1 + weak_mat_tmp33);
        material[7] = c1*(weak_mat_tmp15*(((scalar_t(2) / scalar_t(3)))*grad_u[5]*weak_mat_tmp5 - (scalar_t(2) / scalar_t(3))*weak_mat_tmp37) + weak_mat_tmp35*weak_mat_tmp8) + c2*(weak_mat_tmp16*(scalar_t(2)*grad_u[7]*weak_mat_tmp14 - weak_mat_tmp12*weak_mat_tmp35 - weak_mat_tmp17*weak_mat_tmp34 - weak_mat_tmp21*weak_mat_tmp36) + weak_mat_tmp22*(((scalar_t(4) / scalar_t(3)))*grad_u[5]*weak_mat_tmp5 - (scalar_t(4) / scalar_t(3))*weak_mat_tmp37)) + weak_mat_tmp7*(-grad_u[5]*weak_mat_tmp5 + weak_mat_tmp37);
        material[8] = c1*(weak_mat_tmp15*(((scalar_t(2) / scalar_t(3)))*weak_mat_tmp38 - (scalar_t(2) / scalar_t(3))*weak_mat_tmp39) + weak_mat_tmp36*weak_mat_tmp8) + c2*(weak_mat_tmp16*(-weak_mat_tmp13*weak_mat_tmp36 + scalar_t(2)*weak_mat_tmp14*weak_mat_tmp2 - weak_mat_tmp19*weak_mat_tmp34 - weak_mat_tmp21*weak_mat_tmp35) + weak_mat_tmp22*(((scalar_t(4) / scalar_t(3)))*weak_mat_tmp38 - (scalar_t(4) / scalar_t(3))*weak_mat_tmp39)) + weak_mat_tmp7*(weak_mat_tmp1*weak_mat_tmp5 - weak_mat_tmp38);
        loperand[0] = qw * (material[0] * jacobian_adjugate_lane0 + material[1] * jacobian_adjugate_lane1 + material[2] * jacobian_adjugate_lane2);
        loperand[1] = qw * (material[0] * jacobian_adjugate_lane3 + material[1] * jacobian_adjugate_lane4 + material[2] * jacobian_adjugate_lane5);
        loperand[2] = qw * (material[0] * jacobian_adjugate_lane6 + material[1] * jacobian_adjugate_lane7 + material[2] * jacobian_adjugate_lane8);
        loperand[3] = qw * (material[3] * jacobian_adjugate_lane0 + material[4] * jacobian_adjugate_lane1 + material[5] * jacobian_adjugate_lane2);
        loperand[4] = qw * (material[3] * jacobian_adjugate_lane3 + material[4] * jacobian_adjugate_lane4 + material[5] * jacobian_adjugate_lane5);
        loperand[5] = qw * (material[3] * jacobian_adjugate_lane6 + material[4] * jacobian_adjugate_lane7 + material[5] * jacobian_adjugate_lane8);
        loperand[6] = qw * (material[6] * jacobian_adjugate_lane0 + material[7] * jacobian_adjugate_lane1 + material[8] * jacobian_adjugate_lane2);
        loperand[7] = qw * (material[6] * jacobian_adjugate_lane3 + material[7] * jacobian_adjugate_lane4 + material[8] * jacobian_adjugate_lane5);
        loperand[8] = qw * (material[6] * jacobian_adjugate_lane6 + material[7] * jacobian_adjugate_lane7 + material[8] * jacobian_adjugate_lane8);
            loperand_q[((0 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane] = loperand[0];
            loperand_q[((0 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane] = loperand[1];
            loperand_q[((0 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane] = loperand[2];
            loperand_q[((1 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane] = loperand[3];
            loperand_q[((1 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane] = loperand[4];
            loperand_q[((1 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane] = loperand[5];
            loperand_q[((2 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane] = loperand[6];
            loperand_q[((2 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane] = loperand[7];
            loperand_q[((2 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane] = loperand[8];
        }
    }
    tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, shape_1d, grad_1d, &loperand_q[0 * N_QP * 3 * VECTOR_SIZE], out_streams, 0);
    tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, shape_1d, grad_1d, &loperand_q[1 * N_QP * 3 * VECTOR_SIZE], out_streams, 1);
    tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, shape_1d, grad_1d, &loperand_q[2 * N_QP * 3 * VECTOR_SIZE], out_streams, 2);
}

template <typename scalar_t, int N_QP, int N_SHAPE, int VECTOR_SIZE>
static SFEM_INLINE void modified_mooney_rivlin_d3_tensor_product_apply_block(
        const int nelems,
        const ptrdiff_t geometry_stride,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate0,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate1,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate2,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate3,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate4,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate5,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate6,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate7,
        const scalar_t *const SFEM_RESTRICT jacobian_adjugate8,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant0,
        const scalar_t *const SFEM_RESTRICT shape_1d,
        const scalar_t *const SFEM_RESTRICT grad_1d,
        const scalar_t *const SFEM_RESTRICT q_weight_1d,
        const scalar_t c1,
        const scalar_t c2,
        const scalar_t kappa,
        const scalar_t *const SFEM_RESTRICT u_streams[N_SHAPE * 3],
        const scalar_t *const SFEM_RESTRICT h_streams[N_SHAPE * 3],
        scalar_t *const SFEM_RESTRICT out_streams[N_SHAPE * 3]
) {
    static_assert(N_QP > 0, "N_QP must be positive");
    static_assert(VECTOR_SIZE > 0, "VECTOR_SIZE must be positive");
    static constexpr int N_QP_1D = integer_root(N_QP, 3);
    static constexpr int N_SHAPE_1D = integer_root(N_SHAPE, 3);
    static_assert(ipow(N_QP_1D, 3) == N_QP, "N_QP must be tensor-product compatible");
    static_assert(ipow(N_SHAPE_1D, 3) == N_SHAPE, "N_SHAPE must be tensor-product compatible");
    scalar_t grad_u_ref_q[N_QP * 9 * VECTOR_SIZE];
    scalar_t grad_h_ref_q[N_QP * 9 * VECTOR_SIZE];
    scalar_t loperand_q[N_QP * 9 * VECTOR_SIZE];
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, shape_1d, grad_1d, u_streams, 0, &grad_u_ref_q[0 * N_QP * 3 * VECTOR_SIZE]);
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, shape_1d, grad_1d, h_streams, 0, &grad_h_ref_q[0 * N_QP * 3 * VECTOR_SIZE]);
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, shape_1d, grad_1d, u_streams, 1, &grad_u_ref_q[1 * N_QP * 3 * VECTOR_SIZE]);
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, shape_1d, grad_1d, h_streams, 1, &grad_h_ref_q[1 * N_QP * 3 * VECTOR_SIZE]);
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, shape_1d, grad_1d, u_streams, 2, &grad_u_ref_q[2 * N_QP * 3 * VECTOR_SIZE]);
    tensor_gradient<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, shape_1d, grad_1d, h_streams, 2, &grad_h_ref_q[2 * N_QP * 3 * VECTOR_SIZE]);
    for (int q = 0; q < N_QP; ++q) {
        const int qx = q % N_QP_1D;
        const int qy = (q / N_QP_1D) % N_QP_1D;
        const int qz = q / (N_QP_1D * N_QP_1D);
        const scalar_t qw = q_weight_1d[qx] * q_weight_1d[qy] * q_weight_1d[qz];
        #pragma omp simd
        for (int lane = 0; lane < nelems; ++lane) {
            const ptrdiff_t geometry_offset = q * geometry_stride + lane;
            const scalar_t jacobian_adjugate_lane0 = jacobian_adjugate0[geometry_offset];
            const scalar_t jacobian_adjugate_lane1 = jacobian_adjugate1[geometry_offset];
            const scalar_t jacobian_adjugate_lane2 = jacobian_adjugate2[geometry_offset];
            const scalar_t jacobian_adjugate_lane3 = jacobian_adjugate3[geometry_offset];
            const scalar_t jacobian_adjugate_lane4 = jacobian_adjugate4[geometry_offset];
            const scalar_t jacobian_adjugate_lane5 = jacobian_adjugate5[geometry_offset];
            const scalar_t jacobian_adjugate_lane6 = jacobian_adjugate6[geometry_offset];
            const scalar_t jacobian_adjugate_lane7 = jacobian_adjugate7[geometry_offset];
            const scalar_t jacobian_adjugate_lane8 = jacobian_adjugate8[geometry_offset];
            const scalar_t jacobian_determinant_lane0 = jacobian_determinant0[geometry_offset];
            scalar_t grad_u_ref[9];
            grad_u_ref[0] = grad_u_ref_q[((0 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[1] = grad_u_ref_q[((0 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[2] = grad_u_ref_q[((0 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            grad_u_ref[3] = grad_u_ref_q[((1 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[4] = grad_u_ref_q[((1 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[5] = grad_u_ref_q[((1 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            grad_u_ref[6] = grad_u_ref_q[((2 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_u_ref[7] = grad_u_ref_q[((2 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_u_ref[8] = grad_u_ref_q[((2 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            scalar_t grad_h_ref[9];
            grad_h_ref[0] = grad_h_ref_q[((0 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_h_ref[1] = grad_h_ref_q[((0 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_h_ref[2] = grad_h_ref_q[((0 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            grad_h_ref[3] = grad_h_ref_q[((1 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_h_ref[4] = grad_h_ref_q[((1 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_h_ref[5] = grad_h_ref_q[((1 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            grad_h_ref[6] = grad_h_ref_q[((2 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane];
            grad_h_ref[7] = grad_h_ref_q[((2 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane];
            grad_h_ref[8] = grad_h_ref_q[((2 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane];
            scalar_t grad_u[9];
            scalar_t trial_grad[9];
            const scalar_t inv_jacobian_determinant = scalar_t(1) / jacobian_determinant_lane0;
            grad_u[0] = (grad_u_ref[0] * jacobian_adjugate_lane0 + grad_u_ref[1] * jacobian_adjugate_lane3 + grad_u_ref[2] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            trial_grad[0] = (grad_h_ref[0] * jacobian_adjugate_lane0 + grad_h_ref[1] * jacobian_adjugate_lane3 + grad_h_ref[2] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            grad_u[1] = (grad_u_ref[0] * jacobian_adjugate_lane1 + grad_u_ref[1] * jacobian_adjugate_lane4 + grad_u_ref[2] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            trial_grad[1] = (grad_h_ref[0] * jacobian_adjugate_lane1 + grad_h_ref[1] * jacobian_adjugate_lane4 + grad_h_ref[2] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            grad_u[2] = (grad_u_ref[0] * jacobian_adjugate_lane2 + grad_u_ref[1] * jacobian_adjugate_lane5 + grad_u_ref[2] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            trial_grad[2] = (grad_h_ref[0] * jacobian_adjugate_lane2 + grad_h_ref[1] * jacobian_adjugate_lane5 + grad_h_ref[2] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            grad_u[3] = (grad_u_ref[3] * jacobian_adjugate_lane0 + grad_u_ref[4] * jacobian_adjugate_lane3 + grad_u_ref[5] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            trial_grad[3] = (grad_h_ref[3] * jacobian_adjugate_lane0 + grad_h_ref[4] * jacobian_adjugate_lane3 + grad_h_ref[5] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            grad_u[4] = (grad_u_ref[3] * jacobian_adjugate_lane1 + grad_u_ref[4] * jacobian_adjugate_lane4 + grad_u_ref[5] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            trial_grad[4] = (grad_h_ref[3] * jacobian_adjugate_lane1 + grad_h_ref[4] * jacobian_adjugate_lane4 + grad_h_ref[5] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            grad_u[5] = (grad_u_ref[3] * jacobian_adjugate_lane2 + grad_u_ref[4] * jacobian_adjugate_lane5 + grad_u_ref[5] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            trial_grad[5] = (grad_h_ref[3] * jacobian_adjugate_lane2 + grad_h_ref[4] * jacobian_adjugate_lane5 + grad_h_ref[5] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            grad_u[6] = (grad_u_ref[6] * jacobian_adjugate_lane0 + grad_u_ref[7] * jacobian_adjugate_lane3 + grad_u_ref[8] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            trial_grad[6] = (grad_h_ref[6] * jacobian_adjugate_lane0 + grad_h_ref[7] * jacobian_adjugate_lane3 + grad_h_ref[8] * jacobian_adjugate_lane6) * inv_jacobian_determinant;
            grad_u[7] = (grad_u_ref[6] * jacobian_adjugate_lane1 + grad_u_ref[7] * jacobian_adjugate_lane4 + grad_u_ref[8] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            trial_grad[7] = (grad_h_ref[6] * jacobian_adjugate_lane1 + grad_h_ref[7] * jacobian_adjugate_lane4 + grad_h_ref[8] * jacobian_adjugate_lane7) * inv_jacobian_determinant;
            grad_u[8] = (grad_u_ref[6] * jacobian_adjugate_lane2 + grad_u_ref[7] * jacobian_adjugate_lane5 + grad_u_ref[8] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            trial_grad[8] = (grad_h_ref[6] * jacobian_adjugate_lane2 + grad_h_ref[7] * jacobian_adjugate_lane5 + grad_h_ref[8] * jacobian_adjugate_lane8) * inv_jacobian_determinant;
            scalar_t loperand[9];
        scalar_t material[9];
        const scalar_t weak_mat_tmp0 = grad_u[5]*grad_u[7];
        const scalar_t weak_mat_tmp1 = grad_u[4] + scalar_t(1);
        const scalar_t weak_mat_tmp2 = grad_u[8] + scalar_t(1);
        const scalar_t weak_mat_tmp3 = weak_mat_tmp0 - weak_mat_tmp1*weak_mat_tmp2;
        const scalar_t weak_mat_tmp4 = -weak_mat_tmp3;
        const scalar_t weak_mat_tmp5 = grad_u[3]*weak_mat_tmp2;
        const scalar_t weak_mat_tmp6 = grad_u[6]*weak_mat_tmp1;
        const scalar_t weak_mat_tmp7 = grad_u[0] + scalar_t(1);
        const scalar_t weak_mat_tmp8 = grad_u[1]*grad_u[5]*grad_u[6] - grad_u[1]*weak_mat_tmp5 + grad_u[2]*grad_u[3]*grad_u[7] - grad_u[2]*weak_mat_tmp6 - weak_mat_tmp0*weak_mat_tmp7 + weak_mat_tmp1*weak_mat_tmp2*weak_mat_tmp7;
        const scalar_t weak_mat_tmp9 = kappa/pow_2(weak_mat_tmp8);
        const scalar_t weak_mat_tmp10 = log(weak_mat_tmp8);
        const scalar_t weak_mat_tmp11 = weak_mat_tmp4*weak_mat_tmp9;
        const scalar_t weak_mat_tmp12 = weak_mat_tmp10*weak_mat_tmp11;
        const scalar_t weak_mat_tmp13 = scalar_t(2)/pow(weak_mat_tmp8, (scalar_t(2) / scalar_t(3)));
        const scalar_t weak_mat_tmp14 = pow(weak_mat_tmp8, (scalar_t(-5) / scalar_t(3)));
        const scalar_t weak_mat_tmp15 = weak_mat_tmp1*weak_mat_tmp2;
        const scalar_t weak_mat_tmp16 = ((scalar_t(2) / scalar_t(3)))*weak_mat_tmp0 - (scalar_t(2) / scalar_t(3))*weak_mat_tmp15;
        const scalar_t weak_mat_tmp17 = weak_mat_tmp14*weak_mat_tmp16;
        const scalar_t weak_mat_tmp18 = ((scalar_t(5) / scalar_t(3)))*weak_mat_tmp0 - (scalar_t(5) / scalar_t(3))*weak_mat_tmp15;
        const scalar_t weak_mat_tmp19 = pow_2(grad_u[3]);
        const scalar_t weak_mat_tmp20 = pow_2(grad_u[6]);
        const scalar_t weak_mat_tmp21 = pow_2(weak_mat_tmp7);
        const scalar_t weak_mat_tmp22 = weak_mat_tmp19 + weak_mat_tmp20 + weak_mat_tmp21;
        const scalar_t weak_mat_tmp23 = pow_2(grad_u[1]);
        const scalar_t weak_mat_tmp24 = pow_2(grad_u[7]);
        const scalar_t weak_mat_tmp25 = pow_2(weak_mat_tmp1);
        const scalar_t weak_mat_tmp26 = weak_mat_tmp23 + weak_mat_tmp24 + weak_mat_tmp25;
        const scalar_t weak_mat_tmp27 = pow_2(grad_u[2]);
        const scalar_t weak_mat_tmp28 = pow_2(grad_u[5]);
        const scalar_t weak_mat_tmp29 = pow_2(weak_mat_tmp2);
        const scalar_t weak_mat_tmp30 = weak_mat_tmp27 + weak_mat_tmp28 + weak_mat_tmp29;
        const scalar_t weak_mat_tmp31 = weak_mat_tmp22 + weak_mat_tmp26 + weak_mat_tmp30;
        const scalar_t weak_mat_tmp32 = weak_mat_tmp31/pow(weak_mat_tmp8, (scalar_t(8) / scalar_t(3)));
        const scalar_t weak_mat_tmp33 = weak_mat_tmp16*weak_mat_tmp32;
        const scalar_t weak_mat_tmp34 = scalar_t(2)*weak_mat_tmp28;
        const scalar_t weak_mat_tmp35 = scalar_t(2)*weak_mat_tmp29;
        const scalar_t weak_mat_tmp36 = weak_mat_tmp34 + weak_mat_tmp35;
        const scalar_t weak_mat_tmp37 = scalar_t(2)*weak_mat_tmp24;
        const scalar_t weak_mat_tmp38 = scalar_t(2)*weak_mat_tmp25;
        const scalar_t weak_mat_tmp39 = weak_mat_tmp37 + weak_mat_tmp38;
        const scalar_t weak_mat_tmp40 = pow(weak_mat_tmp8, (scalar_t(-4) / scalar_t(3)));
        const scalar_t weak_mat_tmp41 = ((scalar_t(4) / scalar_t(3)))*weak_mat_tmp0 - (scalar_t(4) / scalar_t(3))*weak_mat_tmp15;
        const scalar_t weak_mat_tmp42 = pow(weak_mat_tmp8, (scalar_t(-7) / scalar_t(3)));
        const scalar_t weak_mat_tmp43 = grad_u[6]*grad_u[7];
        const scalar_t weak_mat_tmp44 = grad_u[1]*weak_mat_tmp7;
        const scalar_t weak_mat_tmp45 = grad_u[3]*weak_mat_tmp1;
        const scalar_t weak_mat_tmp46 = weak_mat_tmp43 + weak_mat_tmp44 + weak_mat_tmp45;
        const scalar_t weak_mat_tmp47 = scalar_t(2)*grad_u[1];
        const scalar_t weak_mat_tmp48 = grad_u[3]*grad_u[5];
        const scalar_t weak_mat_tmp49 = grad_u[2]*weak_mat_tmp7;
        const scalar_t weak_mat_tmp50 = grad_u[6]*weak_mat_tmp2;
        const scalar_t weak_mat_tmp51 = weak_mat_tmp48 + weak_mat_tmp49 + weak_mat_tmp50;
        const scalar_t weak_mat_tmp52 = scalar_t(2)*grad_u[2];
        const scalar_t weak_mat_tmp53 = scalar_t(2)*weak_mat_tmp7;
        const scalar_t weak_mat_tmp54 = weak_mat_tmp42*(-weak_mat_tmp22*weak_mat_tmp53 + scalar_t(2)*weak_mat_tmp31*weak_mat_tmp7 - weak_mat_tmp46*weak_mat_tmp47 - weak_mat_tmp51*weak_mat_tmp52);
        const scalar_t weak_mat_tmp55 = ((scalar_t(7) / scalar_t(3)))*weak_mat_tmp0 - (scalar_t(7) / scalar_t(3))*weak_mat_tmp15;
        const scalar_t weak_mat_tmp56 = grad_u[1]*grad_u[2];
        const scalar_t weak_mat_tmp57 = grad_u[5]*weak_mat_tmp1;
        const scalar_t weak_mat_tmp58 = grad_u[7]*weak_mat_tmp2;
        const scalar_t weak_mat_tmp59 = weak_mat_tmp56 + weak_mat_tmp57 + weak_mat_tmp58;
        const scalar_t weak_mat_tmp60 = -(scalar_t(1) / scalar_t(2))*pow_2(weak_mat_tmp22) - (scalar_t(1) / scalar_t(2))*pow_2(weak_mat_tmp26) - (scalar_t(1) / scalar_t(2))*pow_2(weak_mat_tmp30) + ((scalar_t(1) / scalar_t(2)))*pow_2(weak_mat_tmp31) - pow_2(weak_mat_tmp46) - pow_2(weak_mat_tmp51) - pow_2(weak_mat_tmp59);
        const scalar_t weak_mat_tmp61 = weak_mat_tmp60/pow(weak_mat_tmp8, (scalar_t(10) / scalar_t(3)));
        const scalar_t weak_mat_tmp62 = weak_mat_tmp41*weak_mat_tmp61;
        const scalar_t weak_mat_tmp63 = -grad_u[5]*grad_u[6] + weak_mat_tmp5;
        const scalar_t weak_mat_tmp64 = -weak_mat_tmp63;
        const scalar_t weak_mat_tmp65 = weak_mat_tmp11*weak_mat_tmp64;
        const scalar_t weak_mat_tmp66 = grad_u[5]*grad_u[6];
        const scalar_t weak_mat_tmp67 = ((scalar_t(5) / scalar_t(3)))*weak_mat_tmp5 - (scalar_t(5) / scalar_t(3))*weak_mat_tmp66;
        const scalar_t weak_mat_tmp68 = ((scalar_t(2) / scalar_t(3)))*weak_mat_tmp5 - (scalar_t(2) / scalar_t(3))*weak_mat_tmp66;
        const scalar_t weak_mat_tmp69 = weak_mat_tmp14*weak_mat_tmp53;
        const scalar_t weak_mat_tmp70 = weak_mat_tmp17*weak_mat_tmp47 + weak_mat_tmp68*weak_mat_tmp69;
        const scalar_t weak_mat_tmp71 = ((scalar_t(7) / scalar_t(3)))*weak_mat_tmp5 - (scalar_t(7) / scalar_t(3))*weak_mat_tmp66;
        const scalar_t weak_mat_tmp72 = scalar_t(2)*weak_mat_tmp43;
        const scalar_t weak_mat_tmp73 = scalar_t(2)*weak_mat_tmp45;
        const scalar_t weak_mat_tmp74 = ((scalar_t(4) / scalar_t(3)))*weak_mat_tmp5 - (scalar_t(4) / scalar_t(3))*weak_mat_tmp66;
        const scalar_t weak_mat_tmp75 = scalar_t(2)*grad_u[1]*weak_mat_tmp31 - weak_mat_tmp26*weak_mat_tmp47 - weak_mat_tmp46*weak_mat_tmp53 - weak_mat_tmp52*weak_mat_tmp59;
        const scalar_t weak_mat_tmp76 = weak_mat_tmp41*weak_mat_tmp42;
        const scalar_t weak_mat_tmp77 = weak_mat_tmp40*(-weak_mat_tmp72 - weak_mat_tmp73) + weak_mat_tmp54*weak_mat_tmp74 + weak_mat_tmp75*weak_mat_tmp76;
        const scalar_t weak_mat_tmp78 = grad_u[3]*grad_u[7];
        const scalar_t weak_mat_tmp79 = -weak_mat_tmp6 + weak_mat_tmp78;
        const scalar_t weak_mat_tmp80 = weak_mat_tmp11*weak_mat_tmp79;
        const scalar_t weak_mat_tmp81 = -weak_mat_tmp79;
        const scalar_t weak_mat_tmp82 = ((scalar_t(5) / scalar_t(3)))*grad_u[6]*weak_mat_tmp1 - (scalar_t(5) / scalar_t(3))*weak_mat_tmp78;
        const scalar_t weak_mat_tmp83 = ((scalar_t(2) / scalar_t(3)))*grad_u[6]*weak_mat_tmp1 - (scalar_t(2) / scalar_t(3))*weak_mat_tmp78;
        const scalar_t weak_mat_tmp84 = weak_mat_tmp17*weak_mat_tmp52 + weak_mat_tmp69*weak_mat_tmp83;
        const scalar_t weak_mat_tmp85 = ((scalar_t(7) / scalar_t(3)))*grad_u[6]*weak_mat_tmp1 - (scalar_t(7) / scalar_t(3))*weak_mat_tmp78;
        const scalar_t weak_mat_tmp86 = scalar_t(2)*weak_mat_tmp48;
        const scalar_t weak_mat_tmp87 = scalar_t(2)*weak_mat_tmp50;
        const scalar_t weak_mat_tmp88 = ((scalar_t(4) / scalar_t(3)))*grad_u[6]*weak_mat_tmp1 - (scalar_t(4) / scalar_t(3))*weak_mat_tmp78;
        const scalar_t weak_mat_tmp89 = scalar_t(2)*grad_u[2]*weak_mat_tmp31 - weak_mat_tmp30*weak_mat_tmp52 - weak_mat_tmp47*weak_mat_tmp59 - weak_mat_tmp51*weak_mat_tmp53;
        const scalar_t weak_mat_tmp90 = weak_mat_tmp40*(-weak_mat_tmp86 - weak_mat_tmp87) + weak_mat_tmp54*weak_mat_tmp88 + weak_mat_tmp76*weak_mat_tmp89;
        const scalar_t weak_mat_tmp91 = grad_u[1]*weak_mat_tmp2;
        const scalar_t weak_mat_tmp92 = -grad_u[2]*grad_u[7] + weak_mat_tmp91;
        const scalar_t weak_mat_tmp93 = -weak_mat_tmp92;
        const scalar_t weak_mat_tmp94 = weak_mat_tmp11*weak_mat_tmp93;
        const scalar_t weak_mat_tmp95 = grad_u[2]*grad_u[7];
        const scalar_t weak_mat_tmp96 = ((scalar_t(5) / scalar_t(3)))*weak_mat_tmp91 - (scalar_t(5) / scalar_t(3))*weak_mat_tmp95;
        const scalar_t weak_mat_tmp97 = scalar_t(2)*grad_u[3];
        const scalar_t weak_mat_tmp98 = ((scalar_t(2) / scalar_t(3)))*weak_mat_tmp91 - (scalar_t(2) / scalar_t(3))*weak_mat_tmp95;
        const scalar_t weak_mat_tmp99 = weak_mat_tmp17*weak_mat_tmp97 + weak_mat_tmp69*weak_mat_tmp98;
        const scalar_t weak_mat_tmp100 = ((scalar_t(7) / scalar_t(3)))*weak_mat_tmp91 - (scalar_t(7) / scalar_t(3))*weak_mat_tmp95;
        const scalar_t weak_mat_tmp101 = grad_u[5]*weak_mat_tmp52;
        const scalar_t weak_mat_tmp102 = weak_mat_tmp1*weak_mat_tmp47;
        const scalar_t weak_mat_tmp103 = ((scalar_t(4) / scalar_t(3)))*weak_mat_tmp91 - (scalar_t(4) / scalar_t(3))*weak_mat_tmp95;
        const scalar_t weak_mat_tmp104 = scalar_t(2)*grad_u[5];
        const scalar_t weak_mat_tmp105 = scalar_t(2)*weak_mat_tmp1;
        const scalar_t weak_mat_tmp106 = scalar_t(2)*grad_u[3]*weak_mat_tmp31 - weak_mat_tmp104*weak_mat_tmp51 - weak_mat_tmp105*weak_mat_tmp46 - weak_mat_tmp22*weak_mat_tmp97;
        const scalar_t weak_mat_tmp107 = weak_mat_tmp103*weak_mat_tmp54 + weak_mat_tmp106*weak_mat_tmp76 + weak_mat_tmp40*(-weak_mat_tmp101 - weak_mat_tmp102);
        const scalar_t weak_mat_tmp108 = grad_u[1]*grad_u[5];
        const scalar_t weak_mat_tmp109 = grad_u[2]*weak_mat_tmp1;
        const scalar_t weak_mat_tmp110 = weak_mat_tmp108 - weak_mat_tmp109;
        const scalar_t weak_mat_tmp111 = weak_mat_tmp11*weak_mat_tmp110;
        const scalar_t weak_mat_tmp112 = -weak_mat_tmp110;
        const scalar_t weak_mat_tmp113 = ((scalar_t(5) / scalar_t(3)))*grad_u[2]*weak_mat_tmp1 - (scalar_t(5) / scalar_t(3))*weak_mat_tmp108;
        const scalar_t weak_mat_tmp114 = scalar_t(2)*grad_u[6];
        const scalar_t weak_mat_tmp115 = ((scalar_t(2) / scalar_t(3)))*grad_u[2]*weak_mat_tmp1 - (scalar_t(2) / scalar_t(3))*weak_mat_tmp108;
        const scalar_t weak_mat_tmp116 = weak_mat_tmp114*weak_mat_tmp17 + weak_mat_tmp115*weak_mat_tmp69;
        const scalar_t weak_mat_tmp117 = ((scalar_t(7) / scalar_t(3)))*grad_u[2]*weak_mat_tmp1 - (scalar_t(7) / scalar_t(3))*weak_mat_tmp108;
        const scalar_t weak_mat_tmp118 = grad_u[7]*weak_mat_tmp47;
        const scalar_t weak_mat_tmp119 = weak_mat_tmp2*weak_mat_tmp52;
        const scalar_t weak_mat_tmp120 = ((scalar_t(4) / scalar_t(3)))*grad_u[2]*weak_mat_tmp1 - (scalar_t(4) / scalar_t(3))*weak_mat_tmp108;
        const scalar_t weak_mat_tmp121 = scalar_t(2)*grad_u[7];
        const scalar_t weak_mat_tmp122 = scalar_t(2)*weak_mat_tmp2;
        const scalar_t weak_mat_tmp123 = scalar_t(2)*grad_u[6]*weak_mat_tmp31 - weak_mat_tmp114*weak_mat_tmp22 - weak_mat_tmp121*weak_mat_tmp46 - weak_mat_tmp122*weak_mat_tmp51;
        const scalar_t weak_mat_tmp124 = weak_mat_tmp120*weak_mat_tmp54 + weak_mat_tmp123*weak_mat_tmp76 + weak_mat_tmp40*(-weak_mat_tmp118 - weak_mat_tmp119);
        const scalar_t weak_mat_tmp125 = grad_u[1]*grad_u[6];
        const scalar_t weak_mat_tmp126 = ((scalar_t(5) / scalar_t(3)))*grad_u[7]*weak_mat_tmp7 - (scalar_t(5) / scalar_t(3))*weak_mat_tmp125;
        const scalar_t weak_mat_tmp127 = ((scalar_t(2) / scalar_t(3)))*grad_u[7]*weak_mat_tmp7 - (scalar_t(2) / scalar_t(3))*weak_mat_tmp125;
        const scalar_t weak_mat_tmp128 = ((scalar_t(2) / scalar_t(3)))*weak_mat_tmp31;
        const scalar_t weak_mat_tmp129 = weak_mat_tmp128*weak_mat_tmp14;
        const scalar_t weak_mat_tmp130 = grad_u[7]*weak_mat_tmp129;
        const scalar_t weak_mat_tmp131 = weak_mat_tmp104*weak_mat_tmp17 + weak_mat_tmp127*weak_mat_tmp69 + weak_mat_tmp130;
        const scalar_t weak_mat_tmp132 = ((scalar_t(7) / scalar_t(3)))*grad_u[7]*weak_mat_tmp7 - (scalar_t(7) / scalar_t(3))*weak_mat_tmp125;
        const scalar_t weak_mat_tmp133 = grad_u[2]*grad_u[3];
        const scalar_t weak_mat_tmp134 = ((scalar_t(4) / scalar_t(3)))*grad_u[7]*weak_mat_tmp7 - (scalar_t(4) / scalar_t(3))*weak_mat_tmp125;
        const scalar_t weak_mat_tmp135 = scalar_t(2)*grad_u[5]*weak_mat_tmp31 - weak_mat_tmp104*weak_mat_tmp30 - weak_mat_tmp105*weak_mat_tmp59 - weak_mat_tmp51*weak_mat_tmp97;
        const scalar_t weak_mat_tmp136 = ((scalar_t(4) / scalar_t(3)))*weak_mat_tmp42*weak_mat_tmp60;
        const scalar_t weak_mat_tmp137 = grad_u[7]*weak_mat_tmp136;
        const scalar_t weak_mat_tmp138 = weak_mat_tmp134*weak_mat_tmp54 + weak_mat_tmp135*weak_mat_tmp76 + weak_mat_tmp137 + weak_mat_tmp40*(scalar_t(4)*grad_u[5]*weak_mat_tmp7 - scalar_t(2)*weak_mat_tmp133);
        const scalar_t weak_mat_tmp139 = grad_u[7]*weak_mat_tmp7;
        const scalar_t weak_mat_tmp140 = weak_mat_tmp125 - weak_mat_tmp139;
        const scalar_t weak_mat_tmp141 = -weak_mat_tmp140;
        const scalar_t weak_mat_tmp142 = kappa*weak_mat_tmp10/weak_mat_tmp8;
        const scalar_t weak_mat_tmp143 = grad_u[7]*weak_mat_tmp142;
        const scalar_t weak_mat_tmp144 = weak_mat_tmp11*weak_mat_tmp140 - weak_mat_tmp143;
        const scalar_t weak_mat_tmp145 = ((scalar_t(5) / scalar_t(3)))*grad_u[5]*weak_mat_tmp7 - (scalar_t(5) / scalar_t(3))*weak_mat_tmp133;
        const scalar_t weak_mat_tmp146 = ((scalar_t(2) / scalar_t(3)))*grad_u[5]*weak_mat_tmp7 - (scalar_t(2) / scalar_t(3))*weak_mat_tmp133;
        const scalar_t weak_mat_tmp147 = grad_u[5]*weak_mat_tmp129;
        const scalar_t weak_mat_tmp148 = weak_mat_tmp121*weak_mat_tmp17 + weak_mat_tmp146*weak_mat_tmp69 + weak_mat_tmp147;
        const scalar_t weak_mat_tmp149 = ((scalar_t(7) / scalar_t(3)))*grad_u[5]*weak_mat_tmp7 - (scalar_t(7) / scalar_t(3))*weak_mat_tmp133;
        const scalar_t weak_mat_tmp150 = ((scalar_t(4) / scalar_t(3)))*grad_u[5]*weak_mat_tmp7 - (scalar_t(4) / scalar_t(3))*weak_mat_tmp133;
        const scalar_t weak_mat_tmp151 = scalar_t(2)*grad_u[7]*weak_mat_tmp31 - weak_mat_tmp114*weak_mat_tmp46 - weak_mat_tmp121*weak_mat_tmp26 - weak_mat_tmp122*weak_mat_tmp59;
        const scalar_t weak_mat_tmp152 = grad_u[5]*weak_mat_tmp136;
        const scalar_t weak_mat_tmp153 = weak_mat_tmp150*weak_mat_tmp54 + weak_mat_tmp151*weak_mat_tmp76 + weak_mat_tmp152 + weak_mat_tmp40*(scalar_t(4)*grad_u[7]*weak_mat_tmp7 - scalar_t(2)*weak_mat_tmp125);
        const scalar_t weak_mat_tmp154 = grad_u[5]*weak_mat_tmp7;
        const scalar_t weak_mat_tmp155 = weak_mat_tmp133 - weak_mat_tmp154;
        const scalar_t weak_mat_tmp156 = -weak_mat_tmp155;
        const scalar_t weak_mat_tmp157 = grad_u[5]*weak_mat_tmp142;
        const scalar_t weak_mat_tmp158 = weak_mat_tmp11*weak_mat_tmp155 - weak_mat_tmp157;
        const scalar_t weak_mat_tmp159 = grad_u[2]*grad_u[6];
        const scalar_t weak_mat_tmp160 = weak_mat_tmp2*weak_mat_tmp7;
        const scalar_t weak_mat_tmp161 = ((scalar_t(5) / scalar_t(3)))*weak_mat_tmp159 - (scalar_t(5) / scalar_t(3))*weak_mat_tmp160;
        const scalar_t weak_mat_tmp162 = ((scalar_t(2) / scalar_t(3)))*weak_mat_tmp159 - (scalar_t(2) / scalar_t(3))*weak_mat_tmp160;
        const scalar_t weak_mat_tmp163 = weak_mat_tmp14*weak_mat_tmp2;
        const scalar_t weak_mat_tmp164 = weak_mat_tmp128*weak_mat_tmp163;
        const scalar_t weak_mat_tmp165 = weak_mat_tmp105*weak_mat_tmp17 + weak_mat_tmp162*weak_mat_tmp69 - weak_mat_tmp164;
        const scalar_t weak_mat_tmp166 = ((scalar_t(7) / scalar_t(3)))*weak_mat_tmp159 - (scalar_t(7) / scalar_t(3))*weak_mat_tmp160;
        const scalar_t weak_mat_tmp167 = grad_u[1]*grad_u[3];
        const scalar_t weak_mat_tmp168 = ((scalar_t(4) / scalar_t(3)))*weak_mat_tmp159 - (scalar_t(4) / scalar_t(3))*weak_mat_tmp160;
        const scalar_t weak_mat_tmp169 = scalar_t(2)*weak_mat_tmp1*weak_mat_tmp31 - weak_mat_tmp104*weak_mat_tmp59 - weak_mat_tmp105*weak_mat_tmp26 - weak_mat_tmp46*weak_mat_tmp97;
        const scalar_t weak_mat_tmp170 = weak_mat_tmp136*weak_mat_tmp2;
        const scalar_t weak_mat_tmp171 = weak_mat_tmp168*weak_mat_tmp54 + weak_mat_tmp169*weak_mat_tmp76 - weak_mat_tmp170 + weak_mat_tmp40*(scalar_t(4)*weak_mat_tmp1*weak_mat_tmp7 - scalar_t(2)*weak_mat_tmp167);
        const scalar_t weak_mat_tmp172 = weak_mat_tmp159 - weak_mat_tmp2*weak_mat_tmp7;
        const scalar_t weak_mat_tmp173 = weak_mat_tmp142*weak_mat_tmp2;
        const scalar_t weak_mat_tmp174 = -weak_mat_tmp172;
        const scalar_t weak_mat_tmp175 = weak_mat_tmp11*weak_mat_tmp174 + weak_mat_tmp173;
        const scalar_t weak_mat_tmp176 = weak_mat_tmp1*weak_mat_tmp7;
        const scalar_t weak_mat_tmp177 = ((scalar_t(5) / scalar_t(3)))*weak_mat_tmp167 - (scalar_t(5) / scalar_t(3))*weak_mat_tmp176;
        const scalar_t weak_mat_tmp178 = ((scalar_t(2) / scalar_t(3)))*weak_mat_tmp167 - (scalar_t(2) / scalar_t(3))*weak_mat_tmp176;
        const scalar_t weak_mat_tmp179 = weak_mat_tmp1*weak_mat_tmp129;
        const scalar_t weak_mat_tmp180 = weak_mat_tmp122*weak_mat_tmp17 + weak_mat_tmp178*weak_mat_tmp69 - weak_mat_tmp179;
        const scalar_t weak_mat_tmp181 = ((scalar_t(7) / scalar_t(3)))*weak_mat_tmp167 - (scalar_t(7) / scalar_t(3))*weak_mat_tmp176;
        const scalar_t weak_mat_tmp182 = ((scalar_t(4) / scalar_t(3)))*weak_mat_tmp167 - (scalar_t(4) / scalar_t(3))*weak_mat_tmp176;
        const scalar_t weak_mat_tmp183 = -weak_mat_tmp114*weak_mat_tmp51 - weak_mat_tmp121*weak_mat_tmp59 - weak_mat_tmp122*weak_mat_tmp30 + scalar_t(2)*weak_mat_tmp2*weak_mat_tmp31;
        const scalar_t weak_mat_tmp184 = weak_mat_tmp1*weak_mat_tmp136;
        const scalar_t weak_mat_tmp185 = weak_mat_tmp182*weak_mat_tmp54 + weak_mat_tmp183*weak_mat_tmp76 - weak_mat_tmp184 + weak_mat_tmp40*(-scalar_t(2)*weak_mat_tmp159 + scalar_t(4)*weak_mat_tmp2*weak_mat_tmp7);
        const scalar_t weak_mat_tmp186 = -weak_mat_tmp1*weak_mat_tmp7 + weak_mat_tmp167;
        const scalar_t weak_mat_tmp187 = weak_mat_tmp1*weak_mat_tmp142;
        const scalar_t weak_mat_tmp188 = -weak_mat_tmp186;
        const scalar_t weak_mat_tmp189 = weak_mat_tmp11*weak_mat_tmp188 + weak_mat_tmp187;
        const scalar_t weak_mat_tmp190 = weak_mat_tmp64*weak_mat_tmp9;
        const scalar_t weak_mat_tmp191 = weak_mat_tmp10*weak_mat_tmp190;
        const scalar_t weak_mat_tmp192 = weak_mat_tmp14*weak_mat_tmp68;
        const scalar_t weak_mat_tmp193 = weak_mat_tmp32*weak_mat_tmp68;
        const scalar_t weak_mat_tmp194 = scalar_t(2)*weak_mat_tmp19;
        const scalar_t weak_mat_tmp195 = scalar_t(2)*weak_mat_tmp20;
        const scalar_t weak_mat_tmp196 = weak_mat_tmp194 + weak_mat_tmp195;
        const scalar_t weak_mat_tmp197 = weak_mat_tmp42*weak_mat_tmp75;
        const scalar_t weak_mat_tmp198 = weak_mat_tmp61*weak_mat_tmp74;
        const scalar_t weak_mat_tmp199 = weak_mat_tmp190*weak_mat_tmp79;
        const scalar_t weak_mat_tmp200 = weak_mat_tmp14*weak_mat_tmp47;
        const scalar_t weak_mat_tmp201 = weak_mat_tmp192*weak_mat_tmp52 + weak_mat_tmp200*weak_mat_tmp83;
        const scalar_t weak_mat_tmp202 = scalar_t(2)*weak_mat_tmp57;
        const scalar_t weak_mat_tmp203 = scalar_t(2)*weak_mat_tmp58;
        const scalar_t weak_mat_tmp204 = weak_mat_tmp42*weak_mat_tmp74;
        const scalar_t weak_mat_tmp205 = weak_mat_tmp197*weak_mat_tmp88 + weak_mat_tmp204*weak_mat_tmp89 + weak_mat_tmp40*(-weak_mat_tmp202 - weak_mat_tmp203);
        const scalar_t weak_mat_tmp206 = weak_mat_tmp155*weak_mat_tmp190;
        const scalar_t weak_mat_tmp207 = weak_mat_tmp121*weak_mat_tmp192 + weak_mat_tmp146*weak_mat_tmp200;
        const scalar_t weak_mat_tmp208 = grad_u[6]*weak_mat_tmp53;
        const scalar_t weak_mat_tmp209 = weak_mat_tmp150*weak_mat_tmp197 + weak_mat_tmp151*weak_mat_tmp204 + weak_mat_tmp40*(-weak_mat_tmp119 - weak_mat_tmp208);
        const scalar_t weak_mat_tmp210 = weak_mat_tmp174*weak_mat_tmp190;
        const scalar_t weak_mat_tmp211 = weak_mat_tmp105*weak_mat_tmp192 + weak_mat_tmp162*weak_mat_tmp200;
        const scalar_t weak_mat_tmp212 = grad_u[3]*weak_mat_tmp53;
        const scalar_t weak_mat_tmp213 = weak_mat_tmp168*weak_mat_tmp197 + weak_mat_tmp169*weak_mat_tmp204 + weak_mat_tmp40*(-weak_mat_tmp101 - weak_mat_tmp212);
        const scalar_t weak_mat_tmp214 = grad_u[6]*weak_mat_tmp129;
        const scalar_t weak_mat_tmp215 = weak_mat_tmp104*weak_mat_tmp192 + weak_mat_tmp127*weak_mat_tmp200 - weak_mat_tmp214;
        const scalar_t weak_mat_tmp216 = grad_u[6]*weak_mat_tmp136;
        const scalar_t weak_mat_tmp217 = weak_mat_tmp134*weak_mat_tmp197 + weak_mat_tmp135*weak_mat_tmp204 - weak_mat_tmp216 + weak_mat_tmp40*(scalar_t(4)*weak_mat_tmp108 - scalar_t(2)*weak_mat_tmp109);
        const scalar_t weak_mat_tmp218 = grad_u[6]*weak_mat_tmp142;
        const scalar_t weak_mat_tmp219 = weak_mat_tmp140*weak_mat_tmp190 + weak_mat_tmp218;
        const scalar_t weak_mat_tmp220 = weak_mat_tmp114*weak_mat_tmp192 + weak_mat_tmp115*weak_mat_tmp200 - weak_mat_tmp147;
        const scalar_t weak_mat_tmp221 = weak_mat_tmp120*weak_mat_tmp197 + weak_mat_tmp123*weak_mat_tmp204 - weak_mat_tmp152 + weak_mat_tmp40*(scalar_t(4)*weak_mat_tmp125 - scalar_t(2)*weak_mat_tmp139);
        const scalar_t weak_mat_tmp222 = weak_mat_tmp110*weak_mat_tmp190 + weak_mat_tmp157;
        const scalar_t weak_mat_tmp223 = weak_mat_tmp164 + weak_mat_tmp192*weak_mat_tmp97 + weak_mat_tmp200*weak_mat_tmp98;
        const scalar_t weak_mat_tmp224 = weak_mat_tmp103*weak_mat_tmp197 + weak_mat_tmp106*weak_mat_tmp204 + weak_mat_tmp170 + weak_mat_tmp40*(scalar_t(4)*weak_mat_tmp167 - scalar_t(2)*weak_mat_tmp176);
        const scalar_t weak_mat_tmp225 = -weak_mat_tmp173 + weak_mat_tmp190*weak_mat_tmp93;
        const scalar_t weak_mat_tmp226 = grad_u[3]*weak_mat_tmp129;
        const scalar_t weak_mat_tmp227 = weak_mat_tmp122*weak_mat_tmp192 + weak_mat_tmp178*weak_mat_tmp200 + weak_mat_tmp226;
        const scalar_t weak_mat_tmp228 = grad_u[3]*weak_mat_tmp136;
        const scalar_t weak_mat_tmp229 = weak_mat_tmp182*weak_mat_tmp197 + weak_mat_tmp183*weak_mat_tmp204 + weak_mat_tmp228 + weak_mat_tmp40*(scalar_t(4)*weak_mat_tmp91 - scalar_t(2)*weak_mat_tmp95);
        const scalar_t weak_mat_tmp230 = grad_u[3]*weak_mat_tmp142;
        const scalar_t weak_mat_tmp231 = weak_mat_tmp188*weak_mat_tmp190 - weak_mat_tmp230;
        const scalar_t weak_mat_tmp232 = weak_mat_tmp79*weak_mat_tmp9;
        const scalar_t weak_mat_tmp233 = weak_mat_tmp10*weak_mat_tmp232;
        const scalar_t weak_mat_tmp234 = weak_mat_tmp14*weak_mat_tmp83;
        const scalar_t weak_mat_tmp235 = weak_mat_tmp32*weak_mat_tmp83;
        const scalar_t weak_mat_tmp236 = weak_mat_tmp42*weak_mat_tmp89;
        const scalar_t weak_mat_tmp237 = weak_mat_tmp61*weak_mat_tmp88;
        const scalar_t weak_mat_tmp238 = weak_mat_tmp140*weak_mat_tmp232;
        const scalar_t weak_mat_tmp239 = weak_mat_tmp14*weak_mat_tmp52;
        const scalar_t weak_mat_tmp240 = weak_mat_tmp104*weak_mat_tmp234 + weak_mat_tmp127*weak_mat_tmp239;
        const scalar_t weak_mat_tmp241 = weak_mat_tmp42*weak_mat_tmp88;
        const scalar_t weak_mat_tmp242 = weak_mat_tmp134*weak_mat_tmp236 + weak_mat_tmp135*weak_mat_tmp241 + weak_mat_tmp40*(-weak_mat_tmp102 - weak_mat_tmp212);
        const scalar_t weak_mat_tmp243 = weak_mat_tmp188*weak_mat_tmp232;
        const scalar_t weak_mat_tmp244 = weak_mat_tmp122*weak_mat_tmp234 + weak_mat_tmp178*weak_mat_tmp239;
        const scalar_t weak_mat_tmp245 = weak_mat_tmp182*weak_mat_tmp236 + weak_mat_tmp183*weak_mat_tmp241 + weak_mat_tmp40*(-weak_mat_tmp118 - weak_mat_tmp208);
        const scalar_t weak_mat_tmp246 = -weak_mat_tmp130 + weak_mat_tmp234*weak_mat_tmp97 + weak_mat_tmp239*weak_mat_tmp98;
        const scalar_t weak_mat_tmp247 = weak_mat_tmp103*weak_mat_tmp236 + weak_mat_tmp106*weak_mat_tmp241 - weak_mat_tmp137 + weak_mat_tmp40*(scalar_t(4)*weak_mat_tmp133 - scalar_t(2)*weak_mat_tmp154);
        const scalar_t weak_mat_tmp248 = weak_mat_tmp143 + weak_mat_tmp232*weak_mat_tmp93;
        const scalar_t weak_mat_tmp249 = weak_mat_tmp121*weak_mat_tmp234 + weak_mat_tmp146*weak_mat_tmp239 - weak_mat_tmp226;
        const scalar_t weak_mat_tmp250 = weak_mat_tmp150*weak_mat_tmp236 + weak_mat_tmp151*weak_mat_tmp241 - weak_mat_tmp228 + weak_mat_tmp40*(scalar_t(4)*grad_u[2]*grad_u[7] - scalar_t(2)*weak_mat_tmp91);
        const scalar_t weak_mat_tmp251 = weak_mat_tmp155*weak_mat_tmp232 + weak_mat_tmp230;
        const scalar_t weak_mat_tmp252 = weak_mat_tmp114*weak_mat_tmp234 + weak_mat_tmp115*weak_mat_tmp239 + weak_mat_tmp179;
        const scalar_t weak_mat_tmp253 = weak_mat_tmp120*weak_mat_tmp236 + weak_mat_tmp123*weak_mat_tmp241 + weak_mat_tmp184 + weak_mat_tmp40*(scalar_t(4)*weak_mat_tmp159 - scalar_t(2)*weak_mat_tmp160);
        const scalar_t weak_mat_tmp254 = weak_mat_tmp110*weak_mat_tmp232 - weak_mat_tmp187;
        const scalar_t weak_mat_tmp255 = weak_mat_tmp105*weak_mat_tmp234 + weak_mat_tmp162*weak_mat_tmp239 + weak_mat_tmp214;
        const scalar_t weak_mat_tmp256 = weak_mat_tmp168*weak_mat_tmp236 + weak_mat_tmp169*weak_mat_tmp241 + weak_mat_tmp216 + weak_mat_tmp40*(scalar_t(4)*grad_u[2]*weak_mat_tmp1 - scalar_t(2)*weak_mat_tmp108);
        const scalar_t weak_mat_tmp257 = weak_mat_tmp174*weak_mat_tmp232 - weak_mat_tmp218;
        const scalar_t weak_mat_tmp258 = weak_mat_tmp9*weak_mat_tmp93;
        const scalar_t weak_mat_tmp259 = weak_mat_tmp10*weak_mat_tmp258;
        const scalar_t weak_mat_tmp260 = weak_mat_tmp14*weak_mat_tmp98;
        const scalar_t weak_mat_tmp261 = weak_mat_tmp32*weak_mat_tmp98;
        const scalar_t weak_mat_tmp262 = scalar_t(2)*weak_mat_tmp27;
        const scalar_t weak_mat_tmp263 = weak_mat_tmp262 + weak_mat_tmp35;
        const scalar_t weak_mat_tmp264 = scalar_t(2)*weak_mat_tmp23;
        const scalar_t weak_mat_tmp265 = weak_mat_tmp264 + weak_mat_tmp37;
        const scalar_t weak_mat_tmp266 = weak_mat_tmp103*weak_mat_tmp42;
        const scalar_t weak_mat_tmp267 = weak_mat_tmp103*weak_mat_tmp61;
        const scalar_t weak_mat_tmp268 = weak_mat_tmp140*weak_mat_tmp258;
        const scalar_t weak_mat_tmp269 = weak_mat_tmp14*weak_mat_tmp97;
        const scalar_t weak_mat_tmp270 = weak_mat_tmp104*weak_mat_tmp260 + weak_mat_tmp127*weak_mat_tmp269;
        const scalar_t weak_mat_tmp271 = scalar_t(2)*weak_mat_tmp49;
        const scalar_t weak_mat_tmp272 = weak_mat_tmp106*weak_mat_tmp42;
        const scalar_t weak_mat_tmp273 = weak_mat_tmp134*weak_mat_tmp272 + weak_mat_tmp135*weak_mat_tmp266 + weak_mat_tmp40*(-weak_mat_tmp271 - weak_mat_tmp87);
        const scalar_t weak_mat_tmp274 = weak_mat_tmp110*weak_mat_tmp258;
        const scalar_t weak_mat_tmp275 = weak_mat_tmp114*weak_mat_tmp260 + weak_mat_tmp115*weak_mat_tmp269;
        const scalar_t weak_mat_tmp276 = weak_mat_tmp104*weak_mat_tmp2;
        const scalar_t weak_mat_tmp277 = grad_u[7]*weak_mat_tmp105;
        const scalar_t weak_mat_tmp278 = weak_mat_tmp120*weak_mat_tmp272 + weak_mat_tmp123*weak_mat_tmp266 + weak_mat_tmp40*(-weak_mat_tmp276 - weak_mat_tmp277);
        const scalar_t weak_mat_tmp279 = weak_mat_tmp174*weak_mat_tmp258;
        const scalar_t weak_mat_tmp280 = weak_mat_tmp105*weak_mat_tmp260 + weak_mat_tmp162*weak_mat_tmp269;
        const scalar_t weak_mat_tmp281 = scalar_t(2)*weak_mat_tmp44;
        const scalar_t weak_mat_tmp282 = weak_mat_tmp168*weak_mat_tmp272 + weak_mat_tmp169*weak_mat_tmp266 + weak_mat_tmp40*(-weak_mat_tmp281 - weak_mat_tmp72);
        const scalar_t weak_mat_tmp283 = grad_u[2]*weak_mat_tmp129;
        const scalar_t weak_mat_tmp284 = weak_mat_tmp121*weak_mat_tmp260 + weak_mat_tmp146*weak_mat_tmp269 - weak_mat_tmp283;
        const scalar_t weak_mat_tmp285 = grad_u[2]*weak_mat_tmp136;
        const scalar_t weak_mat_tmp286 = weak_mat_tmp150*weak_mat_tmp272 + weak_mat_tmp151*weak_mat_tmp266 - weak_mat_tmp285 + weak_mat_tmp40*(-scalar_t(2)*weak_mat_tmp6 + scalar_t(4)*weak_mat_tmp78);
        const scalar_t weak_mat_tmp287 = grad_u[2]*weak_mat_tmp142;
        const scalar_t weak_mat_tmp288 = weak_mat_tmp155*weak_mat_tmp258 + weak_mat_tmp287;
        const scalar_t weak_mat_tmp289 = grad_u[1]*weak_mat_tmp129;
        const scalar_t weak_mat_tmp290 = weak_mat_tmp122*weak_mat_tmp260 + weak_mat_tmp178*weak_mat_tmp269 + weak_mat_tmp289;
        const scalar_t weak_mat_tmp291 = grad_u[1]*weak_mat_tmp136;
        const scalar_t weak_mat_tmp292 = weak_mat_tmp182*weak_mat_tmp272 + weak_mat_tmp183*weak_mat_tmp266 + weak_mat_tmp291 + weak_mat_tmp40*(scalar_t(4)*weak_mat_tmp5 - scalar_t(2)*weak_mat_tmp66);
        const scalar_t weak_mat_tmp293 = grad_u[1]*weak_mat_tmp142;
        const scalar_t weak_mat_tmp294 = weak_mat_tmp188*weak_mat_tmp258 - weak_mat_tmp293;
        const scalar_t weak_mat_tmp295 = weak_mat_tmp174*weak_mat_tmp9;
        const scalar_t weak_mat_tmp296 = weak_mat_tmp10*weak_mat_tmp295;
        const scalar_t weak_mat_tmp297 = weak_mat_tmp14*weak_mat_tmp162;
        const scalar_t weak_mat_tmp298 = weak_mat_tmp162*weak_mat_tmp32;
        const scalar_t weak_mat_tmp299 = scalar_t(2)*weak_mat_tmp21;
        const scalar_t weak_mat_tmp300 = weak_mat_tmp195 + weak_mat_tmp299;
        const scalar_t weak_mat_tmp301 = weak_mat_tmp169*weak_mat_tmp42;
        const scalar_t weak_mat_tmp302 = weak_mat_tmp168*weak_mat_tmp61;
        const scalar_t weak_mat_tmp303 = weak_mat_tmp140*weak_mat_tmp295;
        const scalar_t weak_mat_tmp304 = weak_mat_tmp105*weak_mat_tmp14;
        const scalar_t weak_mat_tmp305 = weak_mat_tmp104*weak_mat_tmp297 + weak_mat_tmp127*weak_mat_tmp304;
        const scalar_t weak_mat_tmp306 = scalar_t(2)*weak_mat_tmp56;
        const scalar_t weak_mat_tmp307 = weak_mat_tmp168*weak_mat_tmp42;
        const scalar_t weak_mat_tmp308 = weak_mat_tmp134*weak_mat_tmp301 + weak_mat_tmp135*weak_mat_tmp307 + weak_mat_tmp40*(-weak_mat_tmp203 - weak_mat_tmp306);
        const scalar_t weak_mat_tmp309 = weak_mat_tmp155*weak_mat_tmp295;
        const scalar_t weak_mat_tmp310 = weak_mat_tmp121*weak_mat_tmp297 + weak_mat_tmp146*weak_mat_tmp304;
        const scalar_t weak_mat_tmp311 = grad_u[6]*weak_mat_tmp97;
        const scalar_t weak_mat_tmp312 = weak_mat_tmp150*weak_mat_tmp301 + weak_mat_tmp151*weak_mat_tmp307 + weak_mat_tmp40*(-weak_mat_tmp276 - weak_mat_tmp311);
        const scalar_t weak_mat_tmp313 = weak_mat_tmp114*weak_mat_tmp297 + weak_mat_tmp115*weak_mat_tmp304 + weak_mat_tmp283;
        const scalar_t weak_mat_tmp314 = weak_mat_tmp120*weak_mat_tmp301 + weak_mat_tmp123*weak_mat_tmp307 + weak_mat_tmp285 + weak_mat_tmp40*(scalar_t(4)*grad_u[6]*weak_mat_tmp1 - scalar_t(2)*weak_mat_tmp78);
        const scalar_t weak_mat_tmp315 = weak_mat_tmp110*weak_mat_tmp295 - weak_mat_tmp287;
        const scalar_t weak_mat_tmp316 = weak_mat_tmp129*weak_mat_tmp7;
        const scalar_t weak_mat_tmp317 = weak_mat_tmp122*weak_mat_tmp297 + weak_mat_tmp178*weak_mat_tmp304 - weak_mat_tmp316;
        const scalar_t weak_mat_tmp318 = weak_mat_tmp136*weak_mat_tmp7;
        const scalar_t weak_mat_tmp319 = weak_mat_tmp182*weak_mat_tmp301 + weak_mat_tmp183*weak_mat_tmp307 - weak_mat_tmp318 + weak_mat_tmp40*(-scalar_t(2)*weak_mat_tmp0 + scalar_t(4)*weak_mat_tmp1*weak_mat_tmp2);
        const scalar_t weak_mat_tmp320 = weak_mat_tmp142*weak_mat_tmp7;
        const scalar_t weak_mat_tmp321 = weak_mat_tmp188*weak_mat_tmp295 + weak_mat_tmp320;
        const scalar_t weak_mat_tmp322 = weak_mat_tmp140*weak_mat_tmp9;
        const scalar_t weak_mat_tmp323 = weak_mat_tmp10*weak_mat_tmp322;
        const scalar_t weak_mat_tmp324 = weak_mat_tmp127*weak_mat_tmp14;
        const scalar_t weak_mat_tmp325 = weak_mat_tmp127*weak_mat_tmp32;
        const scalar_t weak_mat_tmp326 = weak_mat_tmp135*weak_mat_tmp42;
        const scalar_t weak_mat_tmp327 = weak_mat_tmp134*weak_mat_tmp61;
        const scalar_t weak_mat_tmp328 = weak_mat_tmp188*weak_mat_tmp322;
        const scalar_t weak_mat_tmp329 = weak_mat_tmp104*weak_mat_tmp14;
        const scalar_t weak_mat_tmp330 = weak_mat_tmp122*weak_mat_tmp324 + weak_mat_tmp178*weak_mat_tmp329;
        const scalar_t weak_mat_tmp331 = weak_mat_tmp134*weak_mat_tmp42;
        const scalar_t weak_mat_tmp332 = weak_mat_tmp182*weak_mat_tmp326 + weak_mat_tmp183*weak_mat_tmp331 + weak_mat_tmp40*(-weak_mat_tmp277 - weak_mat_tmp311);
        const scalar_t weak_mat_tmp333 = weak_mat_tmp114*weak_mat_tmp324 + weak_mat_tmp115*weak_mat_tmp329 - weak_mat_tmp289;
        const scalar_t weak_mat_tmp334 = weak_mat_tmp120*weak_mat_tmp326 + weak_mat_tmp123*weak_mat_tmp331 - weak_mat_tmp291 + weak_mat_tmp40*(scalar_t(4)*grad_u[5]*grad_u[6] - scalar_t(2)*weak_mat_tmp5);
        const scalar_t weak_mat_tmp335 = weak_mat_tmp110*weak_mat_tmp322 + weak_mat_tmp293;
        const scalar_t weak_mat_tmp336 = weak_mat_tmp121*weak_mat_tmp324 + weak_mat_tmp146*weak_mat_tmp329 + weak_mat_tmp316;
        const scalar_t weak_mat_tmp337 = weak_mat_tmp150*weak_mat_tmp326 + weak_mat_tmp151*weak_mat_tmp331 + weak_mat_tmp318 + weak_mat_tmp40*(scalar_t(4)*weak_mat_tmp0 - scalar_t(2)*weak_mat_tmp15);
        const scalar_t weak_mat_tmp338 = weak_mat_tmp155*weak_mat_tmp322 - weak_mat_tmp320;
        const scalar_t weak_mat_tmp339 = weak_mat_tmp110*weak_mat_tmp9;
        const scalar_t weak_mat_tmp340 = weak_mat_tmp10*weak_mat_tmp339;
        const scalar_t weak_mat_tmp341 = weak_mat_tmp115*weak_mat_tmp14;
        const scalar_t weak_mat_tmp342 = weak_mat_tmp115*weak_mat_tmp32;
        const scalar_t weak_mat_tmp343 = weak_mat_tmp262 + weak_mat_tmp34;
        const scalar_t weak_mat_tmp344 = weak_mat_tmp264 + weak_mat_tmp38;
        const scalar_t weak_mat_tmp345 = weak_mat_tmp120*weak_mat_tmp42;
        const scalar_t weak_mat_tmp346 = weak_mat_tmp120*weak_mat_tmp61;
        const scalar_t weak_mat_tmp347 = weak_mat_tmp155*weak_mat_tmp339;
        const scalar_t weak_mat_tmp348 = weak_mat_tmp114*weak_mat_tmp14;
        const scalar_t weak_mat_tmp349 = weak_mat_tmp121*weak_mat_tmp341 + weak_mat_tmp146*weak_mat_tmp348;
        const scalar_t weak_mat_tmp350 = weak_mat_tmp123*weak_mat_tmp42;
        const scalar_t weak_mat_tmp351 = weak_mat_tmp150*weak_mat_tmp350 + weak_mat_tmp151*weak_mat_tmp345 + weak_mat_tmp40*(-weak_mat_tmp281 - weak_mat_tmp73);
        const scalar_t weak_mat_tmp352 = weak_mat_tmp188*weak_mat_tmp339;
        const scalar_t weak_mat_tmp353 = weak_mat_tmp122*weak_mat_tmp341 + weak_mat_tmp178*weak_mat_tmp348;
        const scalar_t weak_mat_tmp354 = weak_mat_tmp182*weak_mat_tmp350 + weak_mat_tmp183*weak_mat_tmp345 + weak_mat_tmp40*(-weak_mat_tmp271 - weak_mat_tmp86);
        const scalar_t weak_mat_tmp355 = weak_mat_tmp155*weak_mat_tmp9;
        const scalar_t weak_mat_tmp356 = weak_mat_tmp10*weak_mat_tmp355;
        const scalar_t weak_mat_tmp357 = weak_mat_tmp14*weak_mat_tmp146;
        const scalar_t weak_mat_tmp358 = weak_mat_tmp146*weak_mat_tmp32;
        const scalar_t weak_mat_tmp359 = weak_mat_tmp194 + weak_mat_tmp299;
        const scalar_t weak_mat_tmp360 = weak_mat_tmp150*weak_mat_tmp42;
        const scalar_t weak_mat_tmp361 = weak_mat_tmp150*weak_mat_tmp61;
        const scalar_t weak_mat_tmp362 = weak_mat_tmp188*weak_mat_tmp355;
        const scalar_t weak_mat_tmp363 = weak_mat_tmp121*weak_mat_tmp14*weak_mat_tmp178 + weak_mat_tmp122*weak_mat_tmp357;
        const scalar_t weak_mat_tmp364 = weak_mat_tmp182*weak_mat_tmp42;
        const scalar_t weak_mat_tmp365 = weak_mat_tmp151*weak_mat_tmp364 + weak_mat_tmp183*weak_mat_tmp360 + weak_mat_tmp40*(-weak_mat_tmp202 - weak_mat_tmp306);
        const scalar_t weak_mat_tmp366 = weak_mat_tmp10*weak_mat_tmp188*weak_mat_tmp9;
        const scalar_t weak_mat_tmp367 = weak_mat_tmp178*weak_mat_tmp32;
        const scalar_t weak_mat_tmp368 = weak_mat_tmp182*weak_mat_tmp61;
        material[0] = trial_grad[0]*(c1*(weak_mat_tmp13 + scalar_t(4)*weak_mat_tmp17*weak_mat_tmp7 + weak_mat_tmp18*weak_mat_tmp33) + c2*(weak_mat_tmp40*(weak_mat_tmp36 + weak_mat_tmp39) + scalar_t(2)*weak_mat_tmp41*weak_mat_tmp54 + weak_mat_tmp55*weak_mat_tmp62) + weak_mat_tmp12*weak_mat_tmp3 + pow_2(weak_mat_tmp4)*weak_mat_tmp9) + trial_grad[1]*(c1*(weak_mat_tmp33*weak_mat_tmp67 + weak_mat_tmp70) + c2*(weak_mat_tmp62*weak_mat_tmp71 + weak_mat_tmp77) + weak_mat_tmp12*weak_mat_tmp63 + weak_mat_tmp65) + trial_grad[2]*(c1*(weak_mat_tmp33*weak_mat_tmp82 + weak_mat_tmp84) + c2*(weak_mat_tmp62*weak_mat_tmp85 + weak_mat_tmp90) + weak_mat_tmp12*weak_mat_tmp81 + weak_mat_tmp80) + trial_grad[3]*(c1*(weak_mat_tmp33*weak_mat_tmp96 + weak_mat_tmp99) + c2*(weak_mat_tmp100*weak_mat_tmp62 + weak_mat_tmp107) + weak_mat_tmp12*weak_mat_tmp92 + weak_mat_tmp94) + trial_grad[4]*(c1*(weak_mat_tmp161*weak_mat_tmp33 + weak_mat_tmp165) + c2*(weak_mat_tmp166*weak_mat_tmp62 + weak_mat_tmp171) + weak_mat_tmp12*weak_mat_tmp172 + weak_mat_tmp175) + trial_grad[5]*(c1*(weak_mat_tmp126*weak_mat_tmp33 + weak_mat_tmp131) + c2*(weak_mat_tmp132*weak_mat_tmp62 + weak_mat_tmp138) + weak_mat_tmp12*weak_mat_tmp141 + weak_mat_tmp144) + trial_grad[6]*(c1*(weak_mat_tmp113*weak_mat_tmp33 + weak_mat_tmp116) + c2*(weak_mat_tmp117*weak_mat_tmp62 + weak_mat_tmp124) + weak_mat_tmp111 + weak_mat_tmp112*weak_mat_tmp12) + trial_grad[7]*(c1*(weak_mat_tmp145*weak_mat_tmp33 + weak_mat_tmp148) + c2*(weak_mat_tmp149*weak_mat_tmp62 + weak_mat_tmp153) + weak_mat_tmp12*weak_mat_tmp156 + weak_mat_tmp158) + trial_grad[8]*(c1*(weak_mat_tmp177*weak_mat_tmp33 + weak_mat_tmp180) + c2*(weak_mat_tmp181*weak_mat_tmp62 + weak_mat_tmp185) + weak_mat_tmp12*weak_mat_tmp186 + weak_mat_tmp189);
        material[1] = trial_grad[0]*(c1*(weak_mat_tmp18*weak_mat_tmp193 + weak_mat_tmp70) + c2*(weak_mat_tmp198*weak_mat_tmp55 + weak_mat_tmp77) + weak_mat_tmp191*weak_mat_tmp3 + weak_mat_tmp65) + trial_grad[1]*(c1*(scalar_t(4)*grad_u[1]*weak_mat_tmp192 + weak_mat_tmp13 + weak_mat_tmp193*weak_mat_tmp67) + c2*(scalar_t(2)*weak_mat_tmp197*weak_mat_tmp74 + weak_mat_tmp198*weak_mat_tmp71 + weak_mat_tmp40*(weak_mat_tmp196 + weak_mat_tmp36)) + weak_mat_tmp191*weak_mat_tmp63 + pow_2(weak_mat_tmp64)*weak_mat_tmp9) + trial_grad[2]*(c1*(weak_mat_tmp193*weak_mat_tmp82 + weak_mat_tmp201) + c2*(weak_mat_tmp198*weak_mat_tmp85 + weak_mat_tmp205) + weak_mat_tmp191*weak_mat_tmp81 + weak_mat_tmp199) + trial_grad[3]*(c1*(weak_mat_tmp193*weak_mat_tmp96 + weak_mat_tmp223) + c2*(weak_mat_tmp100*weak_mat_tmp198 + weak_mat_tmp224) + weak_mat_tmp191*weak_mat_tmp92 + weak_mat_tmp225) + trial_grad[4]*(c1*(weak_mat_tmp161*weak_mat_tmp193 + weak_mat_tmp211) + c2*(weak_mat_tmp166*weak_mat_tmp198 + weak_mat_tmp213) + weak_mat_tmp172*weak_mat_tmp191 + weak_mat_tmp210) + trial_grad[5]*(c1*(weak_mat_tmp126*weak_mat_tmp193 + weak_mat_tmp215) + c2*(weak_mat_tmp132*weak_mat_tmp198 + weak_mat_tmp217) + weak_mat_tmp141*weak_mat_tmp191 + weak_mat_tmp219) + trial_grad[6]*(c1*(weak_mat_tmp113*weak_mat_tmp193 + weak_mat_tmp220) + c2*(weak_mat_tmp117*weak_mat_tmp198 + weak_mat_tmp221) + weak_mat_tmp112*weak_mat_tmp191 + weak_mat_tmp222) + trial_grad[7]*(c1*(weak_mat_tmp145*weak_mat_tmp193 + weak_mat_tmp207) + c2*(weak_mat_tmp149*weak_mat_tmp198 + weak_mat_tmp209) + weak_mat_tmp156*weak_mat_tmp191 + weak_mat_tmp206) + trial_grad[8]*(c1*(weak_mat_tmp177*weak_mat_tmp193 + weak_mat_tmp227) + c2*(weak_mat_tmp181*weak_mat_tmp198 + weak_mat_tmp229) + weak_mat_tmp186*weak_mat_tmp191 + weak_mat_tmp231);
        material[2] = trial_grad[0]*(c1*(weak_mat_tmp18*weak_mat_tmp235 + weak_mat_tmp84) + c2*(weak_mat_tmp237*weak_mat_tmp55 + weak_mat_tmp90) + weak_mat_tmp233*weak_mat_tmp3 + weak_mat_tmp80) + trial_grad[1]*(c1*(weak_mat_tmp201 + weak_mat_tmp235*weak_mat_tmp67) + c2*(weak_mat_tmp205 + weak_mat_tmp237*weak_mat_tmp71) + weak_mat_tmp199 + weak_mat_tmp233*weak_mat_tmp63) + trial_grad[2]*(c1*(scalar_t(4)*grad_u[2]*weak_mat_tmp234 + weak_mat_tmp13 + weak_mat_tmp235*weak_mat_tmp82) + c2*(scalar_t(2)*weak_mat_tmp236*weak_mat_tmp88 + weak_mat_tmp237*weak_mat_tmp85 + weak_mat_tmp40*(weak_mat_tmp196 + weak_mat_tmp39)) + weak_mat_tmp233*weak_mat_tmp81 + pow_2(weak_mat_tmp79)*weak_mat_tmp9) + trial_grad[3]*(c1*(weak_mat_tmp235*weak_mat_tmp96 + weak_mat_tmp246) + c2*(weak_mat_tmp100*weak_mat_tmp237 + weak_mat_tmp247) + weak_mat_tmp233*weak_mat_tmp92 + weak_mat_tmp248) + trial_grad[4]*(c1*(weak_mat_tmp161*weak_mat_tmp235 + weak_mat_tmp255) + c2*(weak_mat_tmp166*weak_mat_tmp237 + weak_mat_tmp256) + weak_mat_tmp172*weak_mat_tmp233 + weak_mat_tmp257) + trial_grad[5]*(c1*(weak_mat_tmp126*weak_mat_tmp235 + weak_mat_tmp240) + c2*(weak_mat_tmp132*weak_mat_tmp237 + weak_mat_tmp242) + weak_mat_tmp141*weak_mat_tmp233 + weak_mat_tmp238) + trial_grad[6]*(c1*(weak_mat_tmp113*weak_mat_tmp235 + weak_mat_tmp252) + c2*(weak_mat_tmp117*weak_mat_tmp237 + weak_mat_tmp253) + weak_mat_tmp112*weak_mat_tmp233 + weak_mat_tmp254) + trial_grad[7]*(c1*(weak_mat_tmp145*weak_mat_tmp235 + weak_mat_tmp249) + c2*(weak_mat_tmp149*weak_mat_tmp237 + weak_mat_tmp250) + weak_mat_tmp156*weak_mat_tmp233 + weak_mat_tmp251) + trial_grad[8]*(c1*(weak_mat_tmp177*weak_mat_tmp235 + weak_mat_tmp244) + c2*(weak_mat_tmp181*weak_mat_tmp237 + weak_mat_tmp245) + weak_mat_tmp186*weak_mat_tmp233 + weak_mat_tmp243);
        material[3] = trial_grad[0]*(c1*(weak_mat_tmp18*weak_mat_tmp261 + weak_mat_tmp99) + c2*(weak_mat_tmp107 + weak_mat_tmp267*weak_mat_tmp55) + weak_mat_tmp259*weak_mat_tmp3 + weak_mat_tmp94) + trial_grad[1]*(c1*(weak_mat_tmp223 + weak_mat_tmp261*weak_mat_tmp67) + c2*(weak_mat_tmp224 + weak_mat_tmp267*weak_mat_tmp71) + weak_mat_tmp225 + weak_mat_tmp259*weak_mat_tmp63) + trial_grad[2]*(c1*(weak_mat_tmp246 + weak_mat_tmp261*weak_mat_tmp82) + c2*(weak_mat_tmp247 + weak_mat_tmp267*weak_mat_tmp85) + weak_mat_tmp248 + weak_mat_tmp259*weak_mat_tmp81) + trial_grad[3]*(c1*(scalar_t(4)*grad_u[3]*weak_mat_tmp260 + weak_mat_tmp13 + weak_mat_tmp261*weak_mat_tmp96) + c2*(weak_mat_tmp100*weak_mat_tmp267 + scalar_t(2)*weak_mat_tmp106*weak_mat_tmp266 + weak_mat_tmp40*(weak_mat_tmp263 + weak_mat_tmp265)) + weak_mat_tmp259*weak_mat_tmp92 + weak_mat_tmp9*pow_2(weak_mat_tmp93)) + trial_grad[4]*(c1*(weak_mat_tmp161*weak_mat_tmp261 + weak_mat_tmp280) + c2*(weak_mat_tmp166*weak_mat_tmp267 + weak_mat_tmp282) + weak_mat_tmp172*weak_mat_tmp259 + weak_mat_tmp279) + trial_grad[5]*(c1*(weak_mat_tmp126*weak_mat_tmp261 + weak_mat_tmp270) + c2*(weak_mat_tmp132*weak_mat_tmp267 + weak_mat_tmp273) + weak_mat_tmp141*weak_mat_tmp259 + weak_mat_tmp268) + trial_grad[6]*(c1*(weak_mat_tmp113*weak_mat_tmp261 + weak_mat_tmp275) + c2*(weak_mat_tmp117*weak_mat_tmp267 + weak_mat_tmp278) + weak_mat_tmp112*weak_mat_tmp259 + weak_mat_tmp274) + trial_grad[7]*(c1*(weak_mat_tmp145*weak_mat_tmp261 + weak_mat_tmp284) + c2*(weak_mat_tmp149*weak_mat_tmp267 + weak_mat_tmp286) + weak_mat_tmp156*weak_mat_tmp259 + weak_mat_tmp288) + trial_grad[8]*(c1*(weak_mat_tmp177*weak_mat_tmp261 + weak_mat_tmp290) + c2*(weak_mat_tmp181*weak_mat_tmp267 + weak_mat_tmp292) + weak_mat_tmp186*weak_mat_tmp259 + weak_mat_tmp294);
        material[4] = trial_grad[0]*(c1*(weak_mat_tmp165 + weak_mat_tmp18*weak_mat_tmp298) + c2*(weak_mat_tmp171 + weak_mat_tmp302*weak_mat_tmp55) + weak_mat_tmp175 + weak_mat_tmp296*weak_mat_tmp3) + trial_grad[1]*(c1*(weak_mat_tmp211 + weak_mat_tmp298*weak_mat_tmp67) + c2*(weak_mat_tmp213 + weak_mat_tmp302*weak_mat_tmp71) + weak_mat_tmp210 + weak_mat_tmp296*weak_mat_tmp63) + trial_grad[2]*(c1*(weak_mat_tmp255 + weak_mat_tmp298*weak_mat_tmp82) + c2*(weak_mat_tmp256 + weak_mat_tmp302*weak_mat_tmp85) + weak_mat_tmp257 + weak_mat_tmp296*weak_mat_tmp81) + trial_grad[3]*(c1*(weak_mat_tmp280 + weak_mat_tmp298*weak_mat_tmp96) + c2*(weak_mat_tmp100*weak_mat_tmp302 + weak_mat_tmp282) + weak_mat_tmp279 + weak_mat_tmp296*weak_mat_tmp92) + trial_grad[4]*(c1*(scalar_t(4)*weak_mat_tmp1*weak_mat_tmp297 + weak_mat_tmp13 + weak_mat_tmp161*weak_mat_tmp298) + c2*(weak_mat_tmp166*weak_mat_tmp302 + scalar_t(2)*weak_mat_tmp168*weak_mat_tmp301 + weak_mat_tmp40*(weak_mat_tmp263 + weak_mat_tmp300)) + weak_mat_tmp172*weak_mat_tmp296 + pow_2(weak_mat_tmp174)*weak_mat_tmp9) + trial_grad[5]*(c1*(weak_mat_tmp126*weak_mat_tmp298 + weak_mat_tmp305) + c2*(weak_mat_tmp132*weak_mat_tmp302 + weak_mat_tmp308) + weak_mat_tmp141*weak_mat_tmp296 + weak_mat_tmp303) + trial_grad[6]*(c1*(weak_mat_tmp113*weak_mat_tmp298 + weak_mat_tmp313) + c2*(weak_mat_tmp117*weak_mat_tmp302 + weak_mat_tmp314) + weak_mat_tmp112*weak_mat_tmp296 + weak_mat_tmp315) + trial_grad[7]*(c1*(weak_mat_tmp145*weak_mat_tmp298 + weak_mat_tmp310) + c2*(weak_mat_tmp149*weak_mat_tmp302 + weak_mat_tmp312) + weak_mat_tmp156*weak_mat_tmp296 + weak_mat_tmp309) + trial_grad[8]*(c1*(weak_mat_tmp177*weak_mat_tmp298 + weak_mat_tmp317) + c2*(weak_mat_tmp181*weak_mat_tmp302 + weak_mat_tmp319) + weak_mat_tmp186*weak_mat_tmp296 + weak_mat_tmp321);
        material[5] = trial_grad[0]*(c1*(weak_mat_tmp131 + weak_mat_tmp18*weak_mat_tmp325) + c2*(weak_mat_tmp138 + weak_mat_tmp327*weak_mat_tmp55) + weak_mat_tmp144 + weak_mat_tmp3*weak_mat_tmp323) + trial_grad[1]*(c1*(weak_mat_tmp215 + weak_mat_tmp325*weak_mat_tmp67) + c2*(weak_mat_tmp217 + weak_mat_tmp327*weak_mat_tmp71) + weak_mat_tmp219 + weak_mat_tmp323*weak_mat_tmp63) + trial_grad[2]*(c1*(weak_mat_tmp240 + weak_mat_tmp325*weak_mat_tmp82) + c2*(weak_mat_tmp242 + weak_mat_tmp327*weak_mat_tmp85) + weak_mat_tmp238 + weak_mat_tmp323*weak_mat_tmp81) + trial_grad[3]*(c1*(weak_mat_tmp270 + weak_mat_tmp325*weak_mat_tmp96) + c2*(weak_mat_tmp100*weak_mat_tmp327 + weak_mat_tmp273) + weak_mat_tmp268 + weak_mat_tmp323*weak_mat_tmp92) + trial_grad[4]*(c1*(weak_mat_tmp161*weak_mat_tmp325 + weak_mat_tmp305) + c2*(weak_mat_tmp166*weak_mat_tmp327 + weak_mat_tmp308) + weak_mat_tmp172*weak_mat_tmp323 + weak_mat_tmp303) + trial_grad[5]*(c1*(scalar_t(4)*grad_u[5]*weak_mat_tmp324 + weak_mat_tmp126*weak_mat_tmp325 + weak_mat_tmp13) + c2*(weak_mat_tmp132*weak_mat_tmp327 + scalar_t(2)*weak_mat_tmp134*weak_mat_tmp326 + weak_mat_tmp40*(weak_mat_tmp265 + weak_mat_tmp300)) + pow_2(weak_mat_tmp140)*weak_mat_tmp9 + weak_mat_tmp141*weak_mat_tmp323) + trial_grad[6]*(c1*(weak_mat_tmp113*weak_mat_tmp325 + weak_mat_tmp333) + c2*(weak_mat_tmp117*weak_mat_tmp327 + weak_mat_tmp334) + weak_mat_tmp112*weak_mat_tmp323 + weak_mat_tmp335) + trial_grad[7]*(c1*(weak_mat_tmp145*weak_mat_tmp325 + weak_mat_tmp336) + c2*(weak_mat_tmp149*weak_mat_tmp327 + weak_mat_tmp337) + weak_mat_tmp156*weak_mat_tmp323 + weak_mat_tmp338) + trial_grad[8]*(c1*(weak_mat_tmp177*weak_mat_tmp325 + weak_mat_tmp330) + c2*(weak_mat_tmp181*weak_mat_tmp327 + weak_mat_tmp332) + weak_mat_tmp186*weak_mat_tmp323 + weak_mat_tmp328);
        material[6] = trial_grad[0]*(c1*(weak_mat_tmp116 + weak_mat_tmp18*weak_mat_tmp342) + c2*(weak_mat_tmp124 + weak_mat_tmp346*weak_mat_tmp55) + weak_mat_tmp111 + weak_mat_tmp3*weak_mat_tmp340) + trial_grad[1]*(c1*(weak_mat_tmp220 + weak_mat_tmp342*weak_mat_tmp67) + c2*(weak_mat_tmp221 + weak_mat_tmp346*weak_mat_tmp71) + weak_mat_tmp222 + weak_mat_tmp340*weak_mat_tmp63) + trial_grad[2]*(c1*(weak_mat_tmp252 + weak_mat_tmp342*weak_mat_tmp82) + c2*(weak_mat_tmp253 + weak_mat_tmp346*weak_mat_tmp85) + weak_mat_tmp254 + weak_mat_tmp340*weak_mat_tmp81) + trial_grad[3]*(c1*(weak_mat_tmp275 + weak_mat_tmp342*weak_mat_tmp96) + c2*(weak_mat_tmp100*weak_mat_tmp346 + weak_mat_tmp278) + weak_mat_tmp274 + weak_mat_tmp340*weak_mat_tmp92) + trial_grad[4]*(c1*(weak_mat_tmp161*weak_mat_tmp342 + weak_mat_tmp313) + c2*(weak_mat_tmp166*weak_mat_tmp346 + weak_mat_tmp314) + weak_mat_tmp172*weak_mat_tmp340 + weak_mat_tmp315) + trial_grad[5]*(c1*(weak_mat_tmp126*weak_mat_tmp342 + weak_mat_tmp333) + c2*(weak_mat_tmp132*weak_mat_tmp346 + weak_mat_tmp334) + weak_mat_tmp141*weak_mat_tmp340 + weak_mat_tmp335) + trial_grad[6]*(c1*(scalar_t(4)*grad_u[6]*weak_mat_tmp341 + weak_mat_tmp113*weak_mat_tmp342 + weak_mat_tmp13) + c2*(weak_mat_tmp117*weak_mat_tmp346 + scalar_t(2)*weak_mat_tmp123*weak_mat_tmp345 + weak_mat_tmp40*(weak_mat_tmp343 + weak_mat_tmp344)) + pow_2(weak_mat_tmp110)*weak_mat_tmp9 + weak_mat_tmp112*weak_mat_tmp340) + trial_grad[7]*(c1*(weak_mat_tmp145*weak_mat_tmp342 + weak_mat_tmp349) + c2*(weak_mat_tmp149*weak_mat_tmp346 + weak_mat_tmp351) + weak_mat_tmp156*weak_mat_tmp340 + weak_mat_tmp347) + trial_grad[8]*(c1*(weak_mat_tmp177*weak_mat_tmp342 + weak_mat_tmp353) + c2*(weak_mat_tmp181*weak_mat_tmp346 + weak_mat_tmp354) + weak_mat_tmp186*weak_mat_tmp340 + weak_mat_tmp352);
        material[7] = trial_grad[0]*(c1*(weak_mat_tmp148 + weak_mat_tmp18*weak_mat_tmp358) + c2*(weak_mat_tmp153 + weak_mat_tmp361*weak_mat_tmp55) + weak_mat_tmp158 + weak_mat_tmp3*weak_mat_tmp356) + trial_grad[1]*(c1*(weak_mat_tmp207 + weak_mat_tmp358*weak_mat_tmp67) + c2*(weak_mat_tmp209 + weak_mat_tmp361*weak_mat_tmp71) + weak_mat_tmp206 + weak_mat_tmp356*weak_mat_tmp63) + trial_grad[2]*(c1*(weak_mat_tmp249 + weak_mat_tmp358*weak_mat_tmp82) + c2*(weak_mat_tmp250 + weak_mat_tmp361*weak_mat_tmp85) + weak_mat_tmp251 + weak_mat_tmp356*weak_mat_tmp81) + trial_grad[3]*(c1*(weak_mat_tmp284 + weak_mat_tmp358*weak_mat_tmp96) + c2*(weak_mat_tmp100*weak_mat_tmp361 + weak_mat_tmp286) + weak_mat_tmp288 + weak_mat_tmp356*weak_mat_tmp92) + trial_grad[4]*(c1*(weak_mat_tmp161*weak_mat_tmp358 + weak_mat_tmp310) + c2*(weak_mat_tmp166*weak_mat_tmp361 + weak_mat_tmp312) + weak_mat_tmp172*weak_mat_tmp356 + weak_mat_tmp309) + trial_grad[5]*(c1*(weak_mat_tmp126*weak_mat_tmp358 + weak_mat_tmp336) + c2*(weak_mat_tmp132*weak_mat_tmp361 + weak_mat_tmp337) + weak_mat_tmp141*weak_mat_tmp356 + weak_mat_tmp338) + trial_grad[6]*(c1*(weak_mat_tmp113*weak_mat_tmp358 + weak_mat_tmp349) + c2*(weak_mat_tmp117*weak_mat_tmp361 + weak_mat_tmp351) + weak_mat_tmp112*weak_mat_tmp356 + weak_mat_tmp347) + trial_grad[7]*(c1*(scalar_t(4)*grad_u[7]*weak_mat_tmp357 + weak_mat_tmp13 + weak_mat_tmp145*weak_mat_tmp358) + c2*(weak_mat_tmp149*weak_mat_tmp361 + scalar_t(2)*weak_mat_tmp151*weak_mat_tmp360 + weak_mat_tmp40*(weak_mat_tmp343 + weak_mat_tmp359)) + pow_2(weak_mat_tmp155)*weak_mat_tmp9 + weak_mat_tmp156*weak_mat_tmp356) + trial_grad[8]*(c1*(weak_mat_tmp177*weak_mat_tmp358 + weak_mat_tmp363) + c2*(weak_mat_tmp181*weak_mat_tmp361 + weak_mat_tmp365) + weak_mat_tmp186*weak_mat_tmp356 + weak_mat_tmp362);
        material[8] = trial_grad[0]*(c1*(weak_mat_tmp18*weak_mat_tmp367 + weak_mat_tmp180) + c2*(weak_mat_tmp185 + weak_mat_tmp368*weak_mat_tmp55) + weak_mat_tmp189 + weak_mat_tmp3*weak_mat_tmp366) + trial_grad[1]*(c1*(weak_mat_tmp227 + weak_mat_tmp367*weak_mat_tmp67) + c2*(weak_mat_tmp229 + weak_mat_tmp368*weak_mat_tmp71) + weak_mat_tmp231 + weak_mat_tmp366*weak_mat_tmp63) + trial_grad[2]*(c1*(weak_mat_tmp244 + weak_mat_tmp367*weak_mat_tmp82) + c2*(weak_mat_tmp245 + weak_mat_tmp368*weak_mat_tmp85) + weak_mat_tmp243 + weak_mat_tmp366*weak_mat_tmp81) + trial_grad[3]*(c1*(weak_mat_tmp290 + weak_mat_tmp367*weak_mat_tmp96) + c2*(weak_mat_tmp100*weak_mat_tmp368 + weak_mat_tmp292) + weak_mat_tmp294 + weak_mat_tmp366*weak_mat_tmp92) + trial_grad[4]*(c1*(weak_mat_tmp161*weak_mat_tmp367 + weak_mat_tmp317) + c2*(weak_mat_tmp166*weak_mat_tmp368 + weak_mat_tmp319) + weak_mat_tmp172*weak_mat_tmp366 + weak_mat_tmp321) + trial_grad[5]*(c1*(weak_mat_tmp126*weak_mat_tmp367 + weak_mat_tmp330) + c2*(weak_mat_tmp132*weak_mat_tmp368 + weak_mat_tmp332) + weak_mat_tmp141*weak_mat_tmp366 + weak_mat_tmp328) + trial_grad[6]*(c1*(weak_mat_tmp113*weak_mat_tmp367 + weak_mat_tmp353) + c2*(weak_mat_tmp117*weak_mat_tmp368 + weak_mat_tmp354) + weak_mat_tmp112*weak_mat_tmp366 + weak_mat_tmp352) + trial_grad[7]*(c1*(weak_mat_tmp145*weak_mat_tmp367 + weak_mat_tmp363) + c2*(weak_mat_tmp149*weak_mat_tmp368 + weak_mat_tmp365) + weak_mat_tmp156*weak_mat_tmp366 + weak_mat_tmp362) + trial_grad[8]*(c1*(weak_mat_tmp13 + scalar_t(4)*weak_mat_tmp163*weak_mat_tmp178 + weak_mat_tmp177*weak_mat_tmp367) + c2*(weak_mat_tmp181*weak_mat_tmp368 + scalar_t(2)*weak_mat_tmp183*weak_mat_tmp364 + weak_mat_tmp40*(weak_mat_tmp344 + weak_mat_tmp359)) + weak_mat_tmp186*weak_mat_tmp366 + pow_2(weak_mat_tmp188)*weak_mat_tmp9);
        loperand[0] = qw * (material[0] * jacobian_adjugate_lane0 + material[1] * jacobian_adjugate_lane1 + material[2] * jacobian_adjugate_lane2);
        loperand[1] = qw * (material[0] * jacobian_adjugate_lane3 + material[1] * jacobian_adjugate_lane4 + material[2] * jacobian_adjugate_lane5);
        loperand[2] = qw * (material[0] * jacobian_adjugate_lane6 + material[1] * jacobian_adjugate_lane7 + material[2] * jacobian_adjugate_lane8);
        loperand[3] = qw * (material[3] * jacobian_adjugate_lane0 + material[4] * jacobian_adjugate_lane1 + material[5] * jacobian_adjugate_lane2);
        loperand[4] = qw * (material[3] * jacobian_adjugate_lane3 + material[4] * jacobian_adjugate_lane4 + material[5] * jacobian_adjugate_lane5);
        loperand[5] = qw * (material[3] * jacobian_adjugate_lane6 + material[4] * jacobian_adjugate_lane7 + material[5] * jacobian_adjugate_lane8);
        loperand[6] = qw * (material[6] * jacobian_adjugate_lane0 + material[7] * jacobian_adjugate_lane1 + material[8] * jacobian_adjugate_lane2);
        loperand[7] = qw * (material[6] * jacobian_adjugate_lane3 + material[7] * jacobian_adjugate_lane4 + material[8] * jacobian_adjugate_lane5);
        loperand[8] = qw * (material[6] * jacobian_adjugate_lane6 + material[7] * jacobian_adjugate_lane7 + material[8] * jacobian_adjugate_lane8);
            loperand_q[((0 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane] = loperand[0];
            loperand_q[((0 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane] = loperand[1];
            loperand_q[((0 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane] = loperand[2];
            loperand_q[((1 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane] = loperand[3];
            loperand_q[((1 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane] = loperand[4];
            loperand_q[((1 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane] = loperand[5];
            loperand_q[((2 * N_QP + q) * 3 + 0) * VECTOR_SIZE + lane] = loperand[6];
            loperand_q[((2 * N_QP + q) * 3 + 1) * VECTOR_SIZE + lane] = loperand[7];
            loperand_q[((2 * N_QP + q) * 3 + 2) * VECTOR_SIZE + lane] = loperand[8];
        }
    }
    tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, shape_1d, grad_1d, &loperand_q[0 * N_QP * 3 * VECTOR_SIZE], out_streams, 0);
    tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, shape_1d, grad_1d, &loperand_q[1 * N_QP * 3 * VECTOR_SIZE], out_streams, 1);
    tensor_test<scalar_t, N_QP, N_SHAPE, VECTOR_SIZE, 3>(nelems, shape_1d, grad_1d, &loperand_q[2 * N_QP * 3 * VECTOR_SIZE], out_streams, 2);
}

} // namespace codegen
} // namespace sfem

#endif
