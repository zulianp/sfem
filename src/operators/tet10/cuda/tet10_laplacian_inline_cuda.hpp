#ifndef CU_TET10_LAPLACIAN_INLINE_HPP
#define CU_TET10_LAPLACIAN_INLINE_HPP

#include "sfem_base.hpp"

template <typename fff_t, typename scalar_t>
static inline __device__ __host__ void cu_tet10_laplacian_trial_operand(const scalar_t                      qx,
                                                                        const scalar_t                      qy,
                                                                        const scalar_t                      qz,
                                                                        const scalar_t                      qw,
                                                                        const ptrdiff_t                     stride,
                                                                        const fff_t *const SFEM_RESTRICT    fff,
                                                                        const scalar_t *const SFEM_RESTRICT u,
                                                                        scalar_t *const SFEM_RESTRICT       out) {
    const scalar_t x0  = 4 * qx;
    const scalar_t x1  = 4 * qy;
    const scalar_t x2  = 4 * qz;
    const scalar_t x3  = x1 + x2;
    const scalar_t x4  = -u[6] * x1;
    const scalar_t x5  = u[0] * (x0 + x3 - 3);
    const scalar_t x6  = -u[7] * x2 + x5;
    const scalar_t x7  = u[1] * (x0 - 1) + u[4] * (-8 * qx - x3 + 4) + u[5] * x1 + u[8] * x2 + x4 + x6;
    const scalar_t x8  = x0 - 4;
    const scalar_t x9  = -u[4] * x0;
    const scalar_t x10 = u[2] * (x1 - 1) + u[5] * x0 + u[6] * (-8 * qy - x2 - x8) + u[9] * x2 + x6 + x9;
    const scalar_t x11 = u[3] * (x2 - 1) + u[7] * (-8 * qz - x1 - x8) + u[8] * x0 + u[9] * x1 + x4 + x5 + x9;
    out[0]             = qw * (fff[0 * stride] * x7 + fff[1 * stride] * x10 + fff[2 * stride] * x11);
    out[1]             = qw * (fff[1 * stride] * x7 + fff[3 * stride] * x10 + fff[4 * stride] * x11);
    out[2]             = qw * (fff[2 * stride] * x7 + fff[4 * stride] * x10 + fff[5 * stride] * x11);
}

template <typename scalar_t>
static inline __device__ __host__ void cu_tet10_laplacian_ref_shape_grad_x(const scalar_t qx,
                                                                           const scalar_t qy,
                                                                           const scalar_t qz,
                                                                           scalar_t *const out) {
    const scalar_t x0 = 4 * qx;
    const scalar_t x1 = 4 * qy;
    const scalar_t x2 = 4 * qz;
    const scalar_t x3 = x1 + x2;
    out[0]            = x0 + x3 - 3;
    out[1]            = x0 - 1;
    out[2]            = 0;
    out[3]            = 0;
    out[4]            = -8 * qx - x3 + 4;
    out[5]            = x1;
    out[6]            = -x1;
    out[7]            = -x2;
    out[8]            = x2;
    out[9]            = 0;
}

template <typename scalar_t>
static inline __device__ __host__ void cu_tet10_laplacian_ref_shape_grad_y(const scalar_t qx,
                                                                           const scalar_t qy,
                                                                           const scalar_t qz,
                                                                           scalar_t *const out) {
    const scalar_t x0 = 4 * qy;
    const scalar_t x1 = 4 * qx;
    const scalar_t x2 = 4 * qz;
    const scalar_t x3 = x1 + x2;
    out[0]            = x0 + x3 - 3;
    out[1]            = 0;
    out[2]            = x0 - 1;
    out[3]            = 0;
    out[4]            = -x1;
    out[5]            = x1;
    out[6]            = -8 * qy - x3 + 4;
    out[7]            = -x2;
    out[8]            = 0;
    out[9]            = x2;
}

template <typename scalar_t>
static inline __device__ __host__ void cu_tet10_laplacian_ref_shape_grad_z(const scalar_t qx,
                                                                           const scalar_t qy,
                                                                           const scalar_t qz,
                                                                           scalar_t *const out) {
    const scalar_t x0 = 4 * qz;
    const scalar_t x1 = 4 * qx;
    const scalar_t x2 = 4 * qy;
    const scalar_t x3 = x1 + x2;
    out[0]            = x0 + x3 - 3;
    out[1]            = 0;
    out[2]            = 0;
    out[3]            = x0 - 1;
    out[4]            = -x1;
    out[5]            = 0;
    out[6]            = -x2;
    out[7]            = -8 * qz - x3 + 4;
    out[8]            = x1;
    out[9]            = x2;
}

template <typename fff_t, typename scalar_t>
static inline __device__ __host__ void cu_tet10_laplacian_apply_micro_kernel(const scalar_t                      qx,
                                                                             const scalar_t                      qy,
                                                                             const scalar_t                      qz,
                                                                             const scalar_t                      qw,
                                                                             const ptrdiff_t                     fff_stride,
                                                                             const fff_t *const SFEM_RESTRICT    fff,
                                                                             const scalar_t *const SFEM_RESTRICT u,
                                                                             scalar_t *const SFEM_RESTRICT element_vector) {
    scalar_t ref_grad[10];
    scalar_t grad_u[3];

    cu_tet10_laplacian_trial_operand(qx, qy, qz, qw, fff_stride, fff, u, grad_u);

    cu_tet10_laplacian_ref_shape_grad_x(qx, qy, qz, ref_grad);
#pragma unroll(10)
    for (int i = 0; i < 10; i++) {
        element_vector[i] += ref_grad[i] * grad_u[0];
    }

    cu_tet10_laplacian_ref_shape_grad_y(qx, qy, qz, ref_grad);
#pragma unroll(10)
    for (int i = 0; i < 10; i++) {
        element_vector[i] += ref_grad[i] * grad_u[1];
    }

    cu_tet10_laplacian_ref_shape_grad_z(qx, qy, qz, ref_grad);
#pragma unroll(10)
    for (int i = 0; i < 10; i++) {
        element_vector[i] += ref_grad[i] * grad_u[2];
    }
}

template <typename fff_t, typename scalar_t>
static inline __device__ __host__ void cu_tet10_laplacian_apply_fff(const fff_t *const SFEM_RESTRICT    fff,
                                                                    const ptrdiff_t                     fff_stride,
                                                                    const scalar_t *const SFEM_RESTRICT u,
                                                                    scalar_t *const SFEM_RESTRICT       element_vector) {
    const scalar_t zero     = static_cast<scalar_t>(0);
    const scalar_t one      = static_cast<scalar_t>(1);
    const scalar_t athird   = static_cast<scalar_t>(1. / 3);
    const scalar_t w_corner = static_cast<scalar_t>(0.025);
    const scalar_t w_mid    = static_cast<scalar_t>(0.225);

    cu_tet10_laplacian_apply_micro_kernel<fff_t, scalar_t>(zero, zero, zero, w_corner, fff_stride, fff, u, element_vector);
    cu_tet10_laplacian_apply_micro_kernel<fff_t, scalar_t>(one, zero, zero, w_corner, fff_stride, fff, u, element_vector);
    cu_tet10_laplacian_apply_micro_kernel<fff_t, scalar_t>(zero, one, zero, w_corner, fff_stride, fff, u, element_vector);
    cu_tet10_laplacian_apply_micro_kernel<fff_t, scalar_t>(zero, zero, one, w_corner, fff_stride, fff, u, element_vector);

    cu_tet10_laplacian_apply_micro_kernel<fff_t, scalar_t>(athird, athird, zero, w_mid, fff_stride, fff, u, element_vector);
    cu_tet10_laplacian_apply_micro_kernel<fff_t, scalar_t>(athird, zero, athird, w_mid, fff_stride, fff, u, element_vector);
    cu_tet10_laplacian_apply_micro_kernel<fff_t, scalar_t>(zero, athird, athird, w_mid, fff_stride, fff, u, element_vector);
    cu_tet10_laplacian_apply_micro_kernel<fff_t, scalar_t>(athird, athird, athird, w_mid, fff_stride, fff, u, element_vector);
}

#endif  // CU_TET10_LAPLACIAN_INLINE_HPP
