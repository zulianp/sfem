#ifndef TET10_LAPLACIAN_INLINE_CPU_HPP
#define TET10_LAPLACIAN_INLINE_CPU_HPP

#include "tet4_inline_cpu.h"

static SFEM_INLINE void tet10_laplacian_trial_operand(const real_t qx,
                                                      const real_t qy,
                                                      const real_t qz,
                                                      const real_t qw,
                                                      const jacobian_t *const SFEM_RESTRICT fff,
                                                      const real_t *const SFEM_RESTRICT u,
                                                      real_t *const SFEM_RESTRICT out) {
    const real_t x0 = 4 * qx;
    const real_t x1 = 4 * qy;
    const real_t x2 = 4 * qz;
    const real_t x3 = x1 + x2;
    const real_t x4 = -u[6] * x1;
    const real_t x5 = u[0] * (x0 + x3 - 3);
    const real_t x6 = -u[7] * x2 + x5;
    const real_t x7 = u[1] * (x0 - 1) + u[4] * (-8 * qx - x3 + 4) + u[5] * x1 + u[8] * x2 + x4 + x6;
    const real_t x8 = x0 - 4;
    const real_t x9 = -u[4] * x0;
    const real_t x10 =
            u[2] * (x1 - 1) + u[5] * x0 + u[6] * (-8 * qy - x2 - x8) + u[9] * x2 + x6 + x9;
    const real_t x11 =
            u[3] * (x2 - 1) + u[7] * (-8 * qz - x1 - x8) + u[8] * x0 + u[9] * x1 + x4 + x5 + x9;
    out[0] = qw * (fff[0] * x7 + fff[1] * x10 + fff[2] * x11);
    out[1] = qw * (fff[1] * x7 + fff[3] * x10 + fff[4] * x11);
    out[2] = qw * (fff[2] * x7 + fff[4] * x10 + fff[5] * x11);
}

static SFEM_INLINE void tet10_ref_shape_grad_x(const real_t qx,
                                               const real_t qy,
                                               const real_t qz,
                                               real_t *const out) {
    const real_t x0 = 4 * qx;
    const real_t x1 = 4 * qy;
    const real_t x2 = 4 * qz;
    const real_t x3 = x1 + x2;
    out[0] = x0 + x3 - 3;
    out[1] = x0 - 1;
    out[2] = 0;
    out[3] = 0;
    out[4] = -8 * qx - x3 + 4;
    out[5] = x1;
    out[6] = -x1;
    out[7] = -x2;
    out[8] = x2;
    out[9] = 0;
}

static SFEM_INLINE void tet10_ref_shape_grad_y(const real_t qx,
                                               const real_t qy,
                                               const real_t qz,
                                               real_t *const out) {
    const real_t x0 = 4 * qy;
    const real_t x1 = 4 * qx;
    const real_t x2 = 4 * qz;
    const real_t x3 = x1 + x2;
    out[0] = x0 + x3 - 3;
    out[1] = 0;
    out[2] = x0 - 1;
    out[3] = 0;
    out[4] = -x1;
    out[5] = x1;
    out[6] = -8 * qy - x3 + 4;
    out[7] = -x2;
    out[8] = 0;
    out[9] = x2;
}

static SFEM_INLINE void tet10_ref_shape_grad_z(const real_t qx,
                                               const real_t qy,
                                               const real_t qz,
                                               real_t *const out) {
    const real_t x0 = 4 * qz;
    const real_t x1 = 4 * qx;
    const real_t x2 = 4 * qy;
    const real_t x3 = x1 + x2;
    out[0] = x0 + x3 - 3;
    out[1] = 0;
    out[2] = 0;
    out[3] = x0 - 1;
    out[4] = -x1;
    out[5] = 0;
    out[6] = -x2;
    out[7] = -8 * qz - x3 + 4;
    out[8] = x1;
    out[9] = x2;
}

static SFEM_INLINE void tet10_laplacian_apply_qp_fff(const real_t qx,
                                                     const real_t qy,
                                                     const real_t qz,
                                                     const real_t qw,
                                                     const jacobian_t *const SFEM_RESTRICT fff,
                                                     const real_t *const SFEM_RESTRICT u,
                                                     real_t *const SFEM_RESTRICT element_vector) {
    // Registers
    real_t ref_grad[10];
    real_t grad_u[3];

    // Evaluate gradient fe function transformed with fff and scaling factors
    tet10_laplacian_trial_operand(qx, qy, qz, qw, fff, u, grad_u);

    {  // X-components
        tet10_ref_shape_grad_x(qx, qy, qz, ref_grad);
#pragma unroll(10)
        for (int i = 0; i < 10; i++) {
            element_vector[i] += ref_grad[i] * grad_u[0];
        }
    }
    {  // Y-components
        tet10_ref_shape_grad_y(qx, qy, qz, ref_grad);
#pragma unroll(10)
        for (int i = 0; i < 10; i++) {
            element_vector[i] += ref_grad[i] * grad_u[1];
        }
    }

    {  // Z-components
        tet10_ref_shape_grad_z(qx, qy, qz, ref_grad);
#pragma unroll(10)
        for (int i = 0; i < 10; i++) {
            element_vector[i] += ref_grad[i] * grad_u[2];
        }
    }
}

static SFEM_INLINE void tet10_laplacian_apply_fff(const jacobian_t *const SFEM_RESTRICT fff,
                                                  const real_t *const SFEM_RESTRICT ex,
                                                  real_t *const SFEM_RESTRICT ey) {
    // Numerical quadrature
    tet10_laplacian_apply_qp_fff(0, 0, 0, 0.025, 1, fffe, ex, ey);
    tet10_laplacian_apply_qp_fff(1, 0, 0, 0.025, 1, fffe, ex, ey);
    tet10_laplacian_apply_qp_fff(0, 1, 0, 0.025, 1, fffe, ex, ey);
    tet10_laplacian_apply_qp_fff(0, 0, 1, 0.025, 1, fffe, ex, ey);

    static const real_t athird = 1. / 3;
    tet10_laplacian_apply_qp_fff(athird, athird, 0., 0.225, 1, fffe, ex, ey);
    tet10_laplacian_apply_qp_fff(athird, 0., athird, 0.225, 1, fffe, ex, ey);
    tet10_laplacian_apply_qp_fff(0., athird, athird, 0.225, 1, fffe, ex, ey);
    tet10_laplacian_apply_qp_fff(athird, athird, athird, 0.225, 1, fffe, ex, ey);
}

static SFEM_INLINE void tet10_laplacian_hessian_fff(const jacobian_t *fff, real_t *element_matrix)

{
    // TODO
}

#endif  // TET10_LAPLACIAN_INLINE_CPU_HPP
