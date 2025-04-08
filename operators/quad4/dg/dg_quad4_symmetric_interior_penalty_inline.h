#ifndef DG_HEX8_SYMMETRIC_INTERIOR_PENALTY_INLINE_H
#define DG_HEX8_SYMMETRIC_INTERIOR_PENALTY_INLINE_H

#include "sfem_base.h"
#include "sfem_macros.h"

#include <math.h>

static void SFEM_INLINE dg_quad4_sip_0(const scalar_t* const SFEM_RESTRICT x,
                                       const scalar_t* const SFEM_RESTRICT y,
                                       const scalar_t* const SFEM_RESTRICT jacobian_inverse,
                                       const scalar_t                      qx,
                                       const scalar_t                      qw,
                                       const scalar_t                      tau,
                                       const scalar_t* const SFEM_RESTRICT u,
                                       scalar_t* const SFEM_RESTRICT       element_vector) {
    scalar_t uh;
    scalar_t guh[2];
    {
        const scalar_t x0 = qx - 1;
        const scalar_t x1 = jacobian_inverse[2] * qx;
        const scalar_t x2 = jacobian_inverse[2] * x0;
        const scalar_t x3 = jacobian_inverse[3] * qx;
        const scalar_t x4 = jacobian_inverse[3] * x0;
        uh                = qx * u[1] - u[0] * x0;
        guh[0]            = -u[0] * (jacobian_inverse[0] - x2) + u[1] * (jacobian_inverse[0] - x1) + u[2] * x1 - u[3] * x2;
        guh[1]            = -u[0] * (jacobian_inverse[1] - x4) + u[1] * (jacobian_inverse[1] - x3) + u[2] * x3 - u[3] * x4;
    }

    const scalar_t x0 = qx - 1;
    const scalar_t x1 = x[0] - x[1];
    const scalar_t x2 = y[0] - y[1];
    const scalar_t x3 = (1.0 / 2.0) * guh[0] * x2 - 1.0 / 2.0 * guh[1] * x1;
    const scalar_t x4 = tau * uh * sqrt(POW2(x1) + POW2(x2));
    element_vector[0] += qw * ((1.0 / 2.0) * uh * (-x0 * x1 - x2) - x0 * x3 - x0 * x4);
    element_vector[1] += qw * (qx * x3 + qx * x4 - 1.0 / 2.0 * uh * (-qx * x1 - x2));
}

static void SFEM_INLINE dg_quad4_sip_1(const scalar_t* const SFEM_RESTRICT x,
                                       const scalar_t* const SFEM_RESTRICT y,
                                       const scalar_t* const SFEM_RESTRICT jacobian_inverse,
                                       const scalar_t                      qx,
                                       const scalar_t                      qw,
                                       const scalar_t                      tau,
                                       const scalar_t* const SFEM_RESTRICT u,
                                       scalar_t* const SFEM_RESTRICT       element_vector) {
    scalar_t uh;
    scalar_t guh[2];
    {
        const scalar_t x0 = qx - 1;
        const scalar_t x1 = jacobian_inverse[0] * qx;
        const scalar_t x2 = jacobian_inverse[1] * qx;
        uh                = qx * u[2] - u[1] * x0;
        guh[0]            = jacobian_inverse[0] * u[0] * x0 - u[1] * (jacobian_inverse[0] * x0 + jacobian_inverse[2]) +
                 u[2] * (jacobian_inverse[2] + x1) - u[3] * x1;
        guh[1] = jacobian_inverse[1] * u[0] * x0 - u[1] * (jacobian_inverse[1] * x0 + jacobian_inverse[3]) +
                 u[2] * (jacobian_inverse[3] + x2) - u[3] * x2;
    }

    const scalar_t x0 = qx - 1;
    const scalar_t x1 = y[1] - y[2];
    const scalar_t x2 = guh[0] * x1;
    const scalar_t x3 = x[1] - x[2];
    const scalar_t x4 = tau * uh * sqrt(POW2(x1) + POW2(x3));
    element_vector[0] += qw * x0 * ((1.0 / 2.0) * guh[1] * x3 + (1.0 / 2.0) * uh * x1 - 1.0 / 2.0 * x2 - x4);
    element_vector[1] += qw * (qx * x4 + (1.0 / 2.0) * qx * (-guh[1] * x3 + x2) - 1.0 / 2.0 * uh * (x0 * x1 - x[1] + x[2]));
}

static void SFEM_INLINE dg_quad4_sip_2(const scalar_t* const SFEM_RESTRICT x,
                                       const scalar_t* const SFEM_RESTRICT y,
                                       const scalar_t* const SFEM_RESTRICT jacobian_inverse,
                                       const scalar_t                      qx,
                                       const scalar_t                      qw,
                                       const scalar_t                      tau,
                                       const scalar_t* const SFEM_RESTRICT u,
                                       scalar_t* const SFEM_RESTRICT       element_vector) {
    scalar_t uh;
    scalar_t guh[2];
    {
        const scalar_t x0 = qx + 1;
        const scalar_t x1 = jacobian_inverse[2] * qx;
        const scalar_t x2 = jacobian_inverse[2] * x0;
        const scalar_t x3 = jacobian_inverse[3] * qx;
        const scalar_t x4 = jacobian_inverse[3] * x0;
        uh                = -qx * u[2] + u[3] * x0;
        guh[0]            = -u[0] * x2 + u[1] * x1 + u[2] * (jacobian_inverse[0] - x1) - u[3] * (jacobian_inverse[0] - x2);
        guh[1]            = -u[0] * x4 + u[1] * x3 + u[2] * (jacobian_inverse[1] - x3) - u[3] * (jacobian_inverse[1] - x4);
    }

    const scalar_t x0 = x[2] - x[3];
    const scalar_t x1 = qx - 1;
    const scalar_t x2 = y[2] - y[3];
    const scalar_t x3 = guh[0] * x2;
    const scalar_t x4 = guh[1] * x0;
    const scalar_t x5 = tau * uh * sqrt(POW2(x0) + POW2(x2));
    element_vector[0] += qw * ((1.0 / 2.0) * uh * x0 * (qx + 1) - x1 * x5 - 1.0 / 2.0 * x1 * (x3 - x4));
    element_vector[1] += qw * qx * (-1.0 / 2.0 * uh * x0 + (1.0 / 2.0) * x3 - 1.0 / 2.0 * x4 + x5);
}

static void SFEM_INLINE dg_quad4_sip_3(const scalar_t* const SFEM_RESTRICT x,
                                       const scalar_t* const SFEM_RESTRICT y,
                                       const scalar_t* const SFEM_RESTRICT jacobian_inverse,
                                       const scalar_t                      qx,
                                       const scalar_t                      qw,
                                       const scalar_t                      tau,
                                       const scalar_t* const SFEM_RESTRICT u,
                                       scalar_t* const SFEM_RESTRICT       element_vector) {
    scalar_t uh;
    scalar_t guh[2];
    {
        const scalar_t x0 = qx + 1;
        const scalar_t x1 = jacobian_inverse[0] * qx;
        const scalar_t x2 = jacobian_inverse[1] * qx;
        uh                = -qx * u[3] + u[0] * x0;
        guh[0] = jacobian_inverse[0] * u[1] * x0 - u[0] * (jacobian_inverse[0] * x0 + jacobian_inverse[2]) - u[2] * x1 +
                 u[3] * (jacobian_inverse[2] + x1);
        guh[1] = jacobian_inverse[1] * u[1] * x0 - u[0] * (jacobian_inverse[1] * x0 + jacobian_inverse[3]) - u[2] * x2 +
                 u[3] * (jacobian_inverse[3] + x2);
    }

    const scalar_t x0 = y[0] - y[3];
    const scalar_t x1 = x0 * (qx + 1);
    const scalar_t x2 = qx - 1;
    const scalar_t x3 = x[0] - x[3];
    const scalar_t x4 = guh[0] * x0 - guh[1] * x3;
    const scalar_t x5 = sqrt(POW2(x0) + POW2(x3));
    element_vector[0] += qw * (-tau * uh * x2 * x5 + (1.0 / 2.0) * uh * (x1 - x[0] + x[3]) + (1.0 / 2.0) * x2 * x4);
    element_vector[1] += qw * (qx * tau * uh * x5 - 1.0 / 2.0 * qx * x4 - 1.0 / 2.0 * uh * x1);
}

#endif  // DG_HEX8_SYMMETRIC_INTERIOR_PENALTY_INLINE_H
