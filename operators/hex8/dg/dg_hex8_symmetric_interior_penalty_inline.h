#ifndef DG_HEX8_SYMMETRIC_INTERIOR_PENALTY_INLINE_H
#define DG_HEX8_SYMMETRIC_INTERIOR_PENALTY_INLINE_H

#include "sfem_base.h"
#include "sfem_macros.h"

#include <math.h>

static void SFEM_INLINE dg_hex8_sip_0(const scalar_t* const SFEM_RESTRICT x,
                                      const scalar_t* const SFEM_RESTRICT y,
                                      const scalar_t* const SFEM_RESTRICT z,
                                      const scalar_t* const SFEM_RESTRICT jacobian_inverse,
                                      const scalar_t                      qx,
                                      const scalar_t                      qy,
                                      const scalar_t                      qw,
                                      const scalar_t                      tau,
                                      const scalar_t* const SFEM_RESTRICT u,
                                      scalar_t* const SFEM_RESTRICT       element_vector) {
    scalar_t uh;
    scalar_t guh[3];
    {
        const scalar_t x0  = qx * qy;
        const scalar_t x1  = qy - 1;
        const scalar_t x2  = qx * x1;
        const scalar_t x3  = qx - 1;
        const scalar_t x4  = qy * x3;
        const scalar_t x5  = u[2] * x2;
        const scalar_t x6  = u[7] * x4;
        const scalar_t x7  = jacobian_inverse[0] * qy;
        const scalar_t x8  = jacobian_inverse[6] * qx;
        const scalar_t x9  = -x1;
        const scalar_t x10 = jacobian_inverse[0] * x9;
        const scalar_t x11 = jacobian_inverse[3] * x9;
        const scalar_t x12 = -x3;
        const scalar_t x13 = jacobian_inverse[6] * x12;
        const scalar_t x14 = qy * x12;
        const scalar_t x15 = jacobian_inverse[1] * qy;
        const scalar_t x16 = jacobian_inverse[7] * qx;
        const scalar_t x17 = jacobian_inverse[1] * x9;
        const scalar_t x18 = jacobian_inverse[4] * x9;
        const scalar_t x19 = jacobian_inverse[7] * x12;
        const scalar_t x20 = jacobian_inverse[2] * qy;
        const scalar_t x21 = jacobian_inverse[8] * qx;
        const scalar_t x22 = jacobian_inverse[2] * x9;
        const scalar_t x23 = jacobian_inverse[5] * x9;
        const scalar_t x24 = jacobian_inverse[8] * x12;
        uh                 = u[0] * x1 * x3 - u[1] * x2 - u[4] * x4 + u[5] * x0;
        guh[0] = jacobian_inverse[3] * qx * qy * u[6] + jacobian_inverse[3] * u[3] * x1 * x3 - jacobian_inverse[3] * x5 -
                 jacobian_inverse[3] * x6 - u[0] * (x10 + x11 * x12 + x13) - u[1] * (qx * x11 - x10 + x8) -
                 u[4] * (jacobian_inverse[3] * x14 - x13 + x7) + u[5] * (-jacobian_inverse[3] * x0 + x7 + x8);
        guh[1] = jacobian_inverse[4] * qx * qy * u[6] + jacobian_inverse[4] * u[3] * x1 * x3 - jacobian_inverse[4] * x5 -
                 jacobian_inverse[4] * x6 - u[0] * (x12 * x18 + x17 + x19) - u[1] * (qx * x18 + x16 - x17) -
                 u[4] * (jacobian_inverse[4] * x14 + x15 - x19) + u[5] * (-jacobian_inverse[4] * x0 + x15 + x16);
        guh[2] = jacobian_inverse[5] * qx * qy * u[6] + jacobian_inverse[5] * u[3] * x1 * x3 - jacobian_inverse[5] * x5 -
                 jacobian_inverse[5] * x6 - u[0] * (x12 * x23 + x22 + x24) - u[1] * (qx * x23 + x21 - x22) -
                 u[4] * (jacobian_inverse[5] * x14 + x20 - x24) + u[5] * (-jacobian_inverse[5] * x0 + x20 + x21);
    }

    const scalar_t x0  = qx - 1;
    const scalar_t x1  = qx * y[1] - qx * y[5] - x0 * y[0] + x0 * y[4];
    const scalar_t x2  = -x1;
    const scalar_t x3  = qy - 1;
    const scalar_t x4  = qy * z[4] - qy * z[5] - x3 * z[0] + x3 * z[1];
    const scalar_t x5  = -x4;
    const scalar_t x6  = qx * z[1] - qx * z[5] - x0 * z[0] + x0 * z[4];
    const scalar_t x7  = -x6;
    const scalar_t x8  = qy * y[4] - qy * y[5] - x3 * y[0] + x3 * y[1];
    const scalar_t x9  = -x8;
    const scalar_t x10 = x2 * x5 - x7 * x9;
    const scalar_t x11 = qx * x[1] - qx * x[5] - x0 * x[0] + x0 * x[4];
    const scalar_t x12 = -x11;
    const scalar_t x13 = qy * x[4] - qy * x[5] - x3 * x[0] + x3 * x[1];
    const scalar_t x14 = -x13;
    const scalar_t x15 = x12 * x9 - x14 * x2;
    const scalar_t x16 = x11 * x4 - x13 * x6;
    const scalar_t x17 = guh[0] * x10 - guh[1] * x16 + guh[2] * x15;
    const scalar_t x18 = x0 * x3;
    const scalar_t x19 = (1.0 / 2.0) * uh;
    const scalar_t x20 = POW2(x16);
    const scalar_t x21 = sqrt(x20 + POW2(-x1 * x13 + x11 * x8) + POW2(x1 * x4 - x6 * x8));
    const scalar_t x22 = -x17;
    const scalar_t x23 = qx * x3;
    const scalar_t x24 = sqrt(POW2(x10) + POW2(x15) + x20);
    const scalar_t x25 = tau * uh;
    const scalar_t x26 = -x3;
    const scalar_t x27 = x16 * x19 * x3;
    const scalar_t x28 = (1.0 / 2.0) * qy;
    element_vector[0] += qw * (tau * uh * x0 * x21 * x3 - 1.0 / 2.0 * x17 * x18 - x19 * (x0 * x15 + x10 * x3 + x16 * x18));
    element_vector[1] +=
            -qw * (x19 * (-qx * x15 + qx * x26 * (x12 * x5 - x14 * x7) + x10 * x26) + (1.0 / 2.0) * x22 * x23 + x23 * x24 * x25);
    element_vector[2] += qw * qx * (qy * tau * uh * x21 - x17 * x28 - x27);
    element_vector[3] += qw * x0 * (-qy * x24 * x25 - x22 * x28 + x27);
}

static void SFEM_INLINE dg_hex8_sip_1(const scalar_t* const SFEM_RESTRICT x,
                                      const scalar_t* const SFEM_RESTRICT y,
                                      const scalar_t* const SFEM_RESTRICT z,
                                      const scalar_t* const SFEM_RESTRICT jacobian_inverse,
                                      const scalar_t                      qx,
                                      const scalar_t                      qy,
                                      const scalar_t                      qw,
                                      const scalar_t                      tau,
                                      const scalar_t* const SFEM_RESTRICT u,
                                      scalar_t* const SFEM_RESTRICT       element_vector) {
    scalar_t uh;
    scalar_t guh[3];
    {
        const scalar_t x0  = qx * qy;
        const scalar_t x1  = qy - 1;
        const scalar_t x2  = qx * x1;
        const scalar_t x3  = qx - 1;
        const scalar_t x4  = qy * x3;
        const scalar_t x5  = x1 * x3;
        const scalar_t x6  = jacobian_inverse[0] * x0;
        const scalar_t x7  = u[0] * x5;
        const scalar_t x8  = jacobian_inverse[3] * qy;
        const scalar_t x9  = jacobian_inverse[6] * qx;
        const scalar_t x10 = -x1;
        const scalar_t x11 = -x3;
        const scalar_t x12 = x10 * x11;
        const scalar_t x13 = jacobian_inverse[1] * x0;
        const scalar_t x14 = jacobian_inverse[4] * qy;
        const scalar_t x15 = jacobian_inverse[7] * qx;
        const scalar_t x16 = jacobian_inverse[2] * x0;
        const scalar_t x17 = jacobian_inverse[5] * qy;
        const scalar_t x18 = jacobian_inverse[8] * qx;
        uh                 = u[1] * x5 - u[2] * x2 - u[5] * x4 + u[6] * x0;
        guh[0] = jacobian_inverse[0] * qx * u[3] * x1 + jacobian_inverse[0] * qy * u[4] * x3 - jacobian_inverse[0] * x7 -
                 u[1] * (-jacobian_inverse[0] * x12 + jacobian_inverse[3] * x10 + jacobian_inverse[6] * x11) -
                 u[2] * (jacobian_inverse[0] * x2 + jacobian_inverse[3] * x1 + x9) -
                 u[5] * (jacobian_inverse[0] * x4 + jacobian_inverse[6] * x3 + x8) + u[6] * (x6 + x8 + x9) - u[7] * x6;
        guh[1] = jacobian_inverse[1] * qx * u[3] * x1 + jacobian_inverse[1] * qy * u[4] * x3 - jacobian_inverse[1] * x7 -
                 u[1] * (-jacobian_inverse[1] * x12 + jacobian_inverse[4] * x10 + jacobian_inverse[7] * x11) -
                 u[2] * (jacobian_inverse[1] * x2 + jacobian_inverse[4] * x1 + x15) -
                 u[5] * (jacobian_inverse[1] * x4 + jacobian_inverse[7] * x3 + x14) + u[6] * (x13 + x14 + x15) - u[7] * x13;
        guh[2] = jacobian_inverse[2] * qx * u[3] * x1 + jacobian_inverse[2] * qy * u[4] * x3 - jacobian_inverse[2] * x7 -
                 u[1] * (-jacobian_inverse[2] * x12 + jacobian_inverse[5] * x10 + jacobian_inverse[8] * x11) -
                 u[2] * (jacobian_inverse[2] * x2 + jacobian_inverse[5] * x1 + x18) -
                 u[5] * (jacobian_inverse[2] * x4 + jacobian_inverse[8] * x3 + x17) + u[6] * (x16 + x17 + x18) - u[7] * x16;
    }

    const scalar_t x0  = qx - 1;
    const scalar_t x1  = qx * y[2] - qx * y[6] - x0 * y[1] + x0 * y[5];
    const scalar_t x2  = -x1;
    const scalar_t x3  = qy - 1;
    const scalar_t x4  = qy * z[5] - qy * z[6] - x3 * z[1] + x3 * z[2];
    const scalar_t x5  = qx * z[2] - qx * z[6] - x0 * z[1] + x0 * z[5];
    const scalar_t x6  = qy * y[5] - qy * y[6] - x3 * y[1] + x3 * y[2];
    const scalar_t x7  = -x6;
    const scalar_t x8  = -x2 * x4 + x5 * x7;
    const scalar_t x9  = guh[0] * x8;
    const scalar_t x10 = qx * x[2] - qx * x[6] - x0 * x[1] + x0 * x[5];
    const scalar_t x11 = qy * x[5] - qy * x[6] - x3 * x[1] + x3 * x[2];
    const scalar_t x12 = x10 * x4 - x11 * x5;
    const scalar_t x13 = -x10 * x7 + x11 * x2;
    const scalar_t x14 = guh[2] * x13;
    const scalar_t x15 = x1 * x4 - x5 * x6;
    const scalar_t x16 = (1.0 / 2.0) * uh;
    const scalar_t x17 = -x1 * x11 + x10 * x6;
    const scalar_t x18 = POW2(x12);
    const scalar_t x19 = tau * uh;
    const scalar_t x20 = x19 * sqrt(POW2(x15) + POW2(x17) + x18);
    const scalar_t x21 = x0 * x3;
    const scalar_t x22 = qx * x3;
    const scalar_t x23 = -guh[1] * x12 + x14 + x9;
    const scalar_t x24 = -1.0 / 2.0 * x23;
    const scalar_t x25 = x19 * sqrt(POW2(x13) + x18 + POW2(x8));
    const scalar_t x26 = -x12 * x3;
    const scalar_t x27 = qx * qy;
    const scalar_t x28 = qy * x0;
    element_vector[0] += qw * x21 * ((1.0 / 2.0) * guh[1] * x12 - 1.0 / 2.0 * x14 + x15 * x16 + x20 - 1.0 / 2.0 * x9);
    element_vector[1] += qw * ((1.0 / 2.0) * uh * (-x0 * x13 - x21 * x8 - x26) - x22 * x24 - x22 * x25);
    element_vector[2] += qw * (x16 * (qx * x17 + x15 * x22 + x26) + x20 * x27 - 1.0 / 2.0 * x23 * x27);
    element_vector[3] += -qw * (x16 * x22 * x8 + x24 * x28 + x25 * x28);
}

static void SFEM_INLINE dg_hex8_sip_2(const scalar_t* const SFEM_RESTRICT x,
                                      const scalar_t* const SFEM_RESTRICT y,
                                      const scalar_t* const SFEM_RESTRICT z,
                                      const scalar_t* const SFEM_RESTRICT jacobian_inverse,
                                      const scalar_t                      qx,
                                      const scalar_t                      qy,
                                      const scalar_t                      qw,
                                      const scalar_t                      tau,
                                      const scalar_t* const SFEM_RESTRICT u,
                                      scalar_t* const SFEM_RESTRICT       element_vector) {
    scalar_t uh;
    scalar_t guh[3];
    {
        const scalar_t x0  = qx * qy;
        const scalar_t x1  = qy - 1;
        const scalar_t x2  = qx * x1;
        const scalar_t x3  = qx - 1;
        const scalar_t x4  = qy * x3;
        const scalar_t x5  = x1 * x3;
        const scalar_t x6  = jacobian_inverse[3] * x0;
        const scalar_t x7  = u[0] * x2;
        const scalar_t x8  = jacobian_inverse[3] * x4;
        const scalar_t x9  = jacobian_inverse[3] * x5;
        const scalar_t x10 = jacobian_inverse[6] * qx;
        const scalar_t x11 = -jacobian_inverse[0] * qy;
        const scalar_t x12 = jacobian_inverse[6] * x3;
        const scalar_t x13 = -x1;
        const scalar_t x14 = qx * x13;
        const scalar_t x15 = jacobian_inverse[4] * x0;
        const scalar_t x16 = jacobian_inverse[4] * x4;
        const scalar_t x17 = jacobian_inverse[4] * x5;
        const scalar_t x18 = jacobian_inverse[7] * qx;
        const scalar_t x19 = -jacobian_inverse[1] * qy;
        const scalar_t x20 = jacobian_inverse[7] * x3;
        const scalar_t x21 = jacobian_inverse[5] * x0;
        const scalar_t x22 = jacobian_inverse[5] * x4;
        const scalar_t x23 = jacobian_inverse[5] * x5;
        const scalar_t x24 = jacobian_inverse[8] * qx;
        const scalar_t x25 = -jacobian_inverse[2] * qy;
        const scalar_t x26 = jacobian_inverse[8] * x3;
        uh                 = u[2] * x5 - u[3] * x2 - u[6] * x4 + u[7] * x0;
        guh[0]             = jacobian_inverse[3] * x7 - u[1] * x9 + u[2] * (-jacobian_inverse[0] * x1 + x12 + x9) -
                 u[3] * (jacobian_inverse[0] * x13 - jacobian_inverse[3] * x14 + x10) - u[4] * x6 + u[5] * x8 +
                 u[6] * (-x11 - x12 - x8) + u[7] * (x10 + x11 + x6);
        guh[1] = jacobian_inverse[4] * x7 - u[1] * x17 + u[2] * (-jacobian_inverse[1] * x1 + x17 + x20) -
                 u[3] * (jacobian_inverse[1] * x13 - jacobian_inverse[4] * x14 + x18) - u[4] * x15 + u[5] * x16 +
                 u[6] * (-x16 - x19 - x20) + u[7] * (x15 + x18 + x19);
        guh[2] = jacobian_inverse[5] * x7 - u[1] * x23 + u[2] * (-jacobian_inverse[2] * x1 + x23 + x26) -
                 u[3] * (jacobian_inverse[2] * x13 - jacobian_inverse[5] * x14 + x24) - u[4] * x21 + u[5] * x22 +
                 u[6] * (-x22 - x25 - x26) + u[7] * (x21 + x24 + x25);
    }

    const scalar_t x0  = (1.0 / 2.0) * qx;
    const scalar_t x1  = qx - 1;
    const scalar_t x2  = qx * x[3] - qx * x[7] - x1 * x[2] + x1 * x[6];
    const scalar_t x3  = qy - 1;
    const scalar_t x4  = qy * z[6] - qy * z[7] - x3 * z[2] + x3 * z[3];
    const scalar_t x5  = qx * z[3] - qx * z[7] - x1 * z[2] + x1 * z[6];
    const scalar_t x6  = qy * x[6] - qy * x[7] - x3 * x[2] + x3 * x[3];
    const scalar_t x7  = x2 * x4 - x5 * x6;
    const scalar_t x8  = uh * x7;
    const scalar_t x9  = qx * y[3] - qx * y[7] - x1 * y[2] + x1 * y[6];
    const scalar_t x10 = -x9;
    const scalar_t x11 = qy * y[6] - qy * y[7] - x3 * y[2] + x3 * y[3];
    const scalar_t x12 = -x11;
    const scalar_t x13 = -x10 * x4 + x12 * x5;
    const scalar_t x14 = x10 * x6 - x12 * x2;
    const scalar_t x15 = guh[0] * x13 - guh[1] * x7 + guh[2] * x14;
    const scalar_t x16 = (1.0 / 2.0) * x1;
    const scalar_t x17 = x11 * x2 - x6 * x9;
    const scalar_t x18 = POW2(x7);
    const scalar_t x19 = -x11 * x5 + x4 * x9;
    const scalar_t x20 = tau * uh;
    const scalar_t x21 = x20 * sqrt(POW2(x17) + x18 + POW2(x19));
    const scalar_t x22 = qw * x3;
    const scalar_t x23 = -x15;
    const scalar_t x24 = x20 * sqrt(POW2(x13) + POW2(x14) + x18);
    const scalar_t x25 = x3 * x7;
    element_vector[0] += x22 * (x0 * x8 + x1 * x21 - x15 * x16);
    element_vector[1] += -x22 * (qx * x24 + x0 * x23 + x16 * x8);
    element_vector[2] += qw * (qx * qy * x21 - qy * x0 * x15 + (1.0 / 2.0) * uh * (-x1 * x17 + x1 * x25 + x19 * x3));
    element_vector[3] += qw * (-qy * x1 * x24 - qy * x16 * x23 + (1.0 / 2.0) * uh * (qx * x14 - qx * x25 - x13 * x3));
}

static void SFEM_INLINE dg_hex8_sip_3(const scalar_t* const SFEM_RESTRICT x,
                                      const scalar_t* const SFEM_RESTRICT y,
                                      const scalar_t* const SFEM_RESTRICT z,
                                      const scalar_t* const SFEM_RESTRICT jacobian_inverse,
                                      const scalar_t                      qx,
                                      const scalar_t                      qy,
                                      const scalar_t                      qw,
                                      const scalar_t                      tau,
                                      const scalar_t* const SFEM_RESTRICT u,
                                      scalar_t* const SFEM_RESTRICT       element_vector) {
    scalar_t uh;
    scalar_t guh[3];
    {
        const scalar_t x0  = qx * qy;
        const scalar_t x1  = qy - 1;
        const scalar_t x2  = qx * x1;
        const scalar_t x3  = qx - 1;
        const scalar_t x4  = qy * x3;
        const scalar_t x5  = u[1] * x2;
        const scalar_t x6  = jacobian_inverse[0] * x4;
        const scalar_t x7  = jacobian_inverse[3] * qy;
        const scalar_t x8  = jacobian_inverse[6] * qx;
        const scalar_t x9  = -x1;
        const scalar_t x10 = jacobian_inverse[3] * x9;
        const scalar_t x11 = jacobian_inverse[0] * x9;
        const scalar_t x12 = -x3;
        const scalar_t x13 = jacobian_inverse[1] * x4;
        const scalar_t x14 = jacobian_inverse[4] * qy;
        const scalar_t x15 = jacobian_inverse[7] * qx;
        const scalar_t x16 = jacobian_inverse[4] * x9;
        const scalar_t x17 = jacobian_inverse[1] * x9;
        const scalar_t x18 = jacobian_inverse[2] * x4;
        const scalar_t x19 = jacobian_inverse[5] * qy;
        const scalar_t x20 = jacobian_inverse[8] * qx;
        const scalar_t x21 = jacobian_inverse[5] * x9;
        const scalar_t x22 = jacobian_inverse[2] * x9;
        uh                 = -u[0] * x2 + u[3] * x1 * x3 + u[4] * x0 - u[7] * x4;
        guh[0] = jacobian_inverse[0] * qx * qy * u[5] + jacobian_inverse[0] * u[2] * x1 * x3 - jacobian_inverse[0] * x5 -
                 u[0] * (qx * x11 + x10 + x8) - u[3] * (jacobian_inverse[6] * x12 - x10 + x11 * x12) -
                 u[4] * (jacobian_inverse[0] * x0 + x7 - x8) - u[6] * x6 + u[7] * (-jacobian_inverse[6] * x3 + x6 + x7);
        guh[1] = jacobian_inverse[1] * qx * qy * u[5] + jacobian_inverse[1] * u[2] * x1 * x3 - jacobian_inverse[1] * x5 -
                 u[0] * (qx * x17 + x15 + x16) - u[3] * (jacobian_inverse[7] * x12 + x12 * x17 - x16) -
                 u[4] * (jacobian_inverse[1] * x0 + x14 - x15) - u[6] * x13 + u[7] * (-jacobian_inverse[7] * x3 + x13 + x14);
        guh[2] = jacobian_inverse[2] * qx * qy * u[5] + jacobian_inverse[2] * u[2] * x1 * x3 - jacobian_inverse[2] * x5 -
                 u[0] * (qx * x22 + x20 + x21) - u[3] * (jacobian_inverse[8] * x12 + x12 * x22 - x21) -
                 u[4] * (jacobian_inverse[2] * x0 + x19 - x20) - u[6] * x18 + u[7] * (-jacobian_inverse[8] * x3 + x18 + x19);
    }

    const scalar_t x0  = qx - 1;
    const scalar_t x1  = qx * y[0] - qx * y[4] - x0 * y[3] + x0 * y[7];
    const scalar_t x2  = -x1;
    const scalar_t x3  = qy - 1;
    const scalar_t x4  = qy * z[4] - qy * z[7] - x3 * z[0] + x3 * z[3];
    const scalar_t x5  = qx * z[0] - qx * z[4] - x0 * z[3] + x0 * z[7];
    const scalar_t x6  = qy * y[4] - qy * y[7] - x3 * y[0] + x3 * y[3];
    const scalar_t x7  = x2 * x4 + x5 * x6;
    const scalar_t x8  = guh[0] * x7;
    const scalar_t x9  = qx * x[0] - qx * x[4] - x0 * x[3] + x0 * x[7];
    const scalar_t x10 = qy * x[4] - qy * x[7] - x3 * x[0] + x3 * x[3];
    const scalar_t x11 = -x10 * x2 - x6 * x9;
    const scalar_t x12 = guh[2] * x11;
    const scalar_t x13 = -x10 * x5 + x4 * x9;
    const scalar_t x14 = -x13;
    const scalar_t x15 = guh[1] * x14;
    const scalar_t x16 = x12 - x15 + x8;
    const scalar_t x17 = (1.0 / 2.0) * x16;
    const scalar_t x18 = x0 * x3;
    const scalar_t x19 = x14 * x3;
    const scalar_t x20 = (1.0 / 2.0) * uh;
    const scalar_t x21 = -x1 * x4 + x5 * x6;
    const scalar_t x22 = sqrt(POW2(x13) + POW2(x21) + POW2(x1 * x10 - x6 * x9));
    const scalar_t x23 = tau * uh * sqrt(POW2(x11) + POW2(x14) + POW2(x7));
    const scalar_t x24 = qy * x0;
    element_vector[0] += qw * (tau * uh * x0 * x22 * x3 - x17 * x18 - x20 * (-qx * x11 + qx * x3 * x7 - x19));
    element_vector[1] += qw * qx * x3 * ((1.0 / 2.0) * x12 - 1.0 / 2.0 * x15 + x20 * x7 - x23 + (1.0 / 2.0) * x8);
    element_vector[2] += qw * (qx * qy * tau * uh * x22 - qx * qy * x17 - x18 * x20 * x21);
    element_vector[3] += qw * ((1.0 / 2.0) * uh * (-x0 * x11 + x0 * x3 * x7 - x19) + (1.0 / 2.0) * x16 * x24 - x23 * x24);
}

static void SFEM_INLINE dg_hex8_sip_4(const scalar_t* const SFEM_RESTRICT x,
                                      const scalar_t* const SFEM_RESTRICT y,
                                      const scalar_t* const SFEM_RESTRICT z,
                                      const scalar_t* const SFEM_RESTRICT jacobian_inverse,
                                      const scalar_t                      qx,
                                      const scalar_t                      qy,
                                      const scalar_t                      qw,
                                      const scalar_t                      tau,
                                      const scalar_t* const SFEM_RESTRICT u,
                                      scalar_t* const SFEM_RESTRICT       element_vector) {
    scalar_t uh;
    scalar_t guh[3];
    {
        const scalar_t x0  = qx * qy;
        const scalar_t x1  = qy - 1;
        const scalar_t x2  = qx * x1;
        const scalar_t x3  = qx - 1;
        const scalar_t x4  = qy * x3;
        const scalar_t x5  = jacobian_inverse[6] * x2;
        const scalar_t x6  = u[4] * x4;
        const scalar_t x7  = jacobian_inverse[3] * qx;
        const scalar_t x8  = jacobian_inverse[0] * qy;
        const scalar_t x9  = -x3;
        const scalar_t x10 = jacobian_inverse[3] * x9;
        const scalar_t x11 = jacobian_inverse[6] * x9;
        const scalar_t x12 = -x1;
        const scalar_t x13 = jacobian_inverse[7] * x2;
        const scalar_t x14 = jacobian_inverse[4] * qx;
        const scalar_t x15 = jacobian_inverse[1] * qy;
        const scalar_t x16 = jacobian_inverse[4] * x9;
        const scalar_t x17 = jacobian_inverse[7] * x9;
        const scalar_t x18 = jacobian_inverse[8] * x2;
        const scalar_t x19 = jacobian_inverse[5] * qx;
        const scalar_t x20 = jacobian_inverse[2] * qy;
        const scalar_t x21 = jacobian_inverse[5] * x9;
        const scalar_t x22 = jacobian_inverse[8] * x9;
        uh                 = -u[0] * x4 + u[1] * x0 - u[2] * x2 + u[3] * x1 * x3;
        guh[0] = jacobian_inverse[6] * qx * qy * u[5] + jacobian_inverse[6] * u[7] * x1 * x3 - jacobian_inverse[6] * x6 -
                 u[0] * (qy * x11 + x10 + x8) - u[1] * (jacobian_inverse[6] * x0 + x7 - x8) +
                 u[2] * (-jacobian_inverse[0] * x1 + x5 + x7) - u[3] * (jacobian_inverse[0] * x12 - x10 + x11 * x12) - u[6] * x5;
        guh[1] = jacobian_inverse[7] * qx * qy * u[5] + jacobian_inverse[7] * u[7] * x1 * x3 - jacobian_inverse[7] * x6 -
                 u[0] * (qy * x17 + x15 + x16) - u[1] * (jacobian_inverse[7] * x0 + x14 - x15) +
                 u[2] * (-jacobian_inverse[1] * x1 + x13 + x14) - u[3] * (jacobian_inverse[1] * x12 + x12 * x17 - x16) -
                 u[6] * x13;
        guh[2] = jacobian_inverse[8] * qx * qy * u[5] + jacobian_inverse[8] * u[7] * x1 * x3 - jacobian_inverse[8] * x6 -
                 u[0] * (qy * x22 + x20 + x21) - u[1] * (jacobian_inverse[8] * x0 + x19 - x20) +
                 u[2] * (-jacobian_inverse[2] * x1 + x18 + x19) - u[3] * (jacobian_inverse[2] * x12 + x12 * x22 - x21) -
                 u[6] * x18;
    }

    const scalar_t x0  = qx - 1;
    const scalar_t x1  = qx * y[1] - qx * y[2] - x0 * y[0] + x0 * y[3];
    const scalar_t x2  = qy - 1;
    const scalar_t x3  = qy * z[0] - qy * z[1] + x2 * z[2] - x2 * z[3];
    const scalar_t x4  = -x3;
    const scalar_t x5  = qx * z[1] - qx * z[2] - x0 * z[0] + x0 * z[3];
    const scalar_t x6  = qy * y[0] - qy * y[1] + x2 * y[2] - x2 * y[3];
    const scalar_t x7  = -x6;
    const scalar_t x8  = x1 * x4 - x5 * x7;
    const scalar_t x9  = qx * x[1] - qx * x[2] - x0 * x[0] + x0 * x[3];
    const scalar_t x10 = qy * x[0] - qy * x[1] + x2 * x[2] - x2 * x[3];
    const scalar_t x11 = -x10;
    const scalar_t x12 = -x1 * x11 + x7 * x9;
    const scalar_t x13 = -x10 * x5 + x3 * x9;
    const scalar_t x14 = -x13;
    const scalar_t x15 = guh[0] * x8 - guh[1] * x14 + guh[2] * x12;
    const scalar_t x16 = (1.0 / 2.0) * x15;
    const scalar_t x17 = x0 * x14;
    const scalar_t x18 = (1.0 / 2.0) * uh;
    const scalar_t x19 = x1 * x10 - x6 * x9;
    const scalar_t x20 = -x1 * x3 + x5 * x6;
    const scalar_t x21 = sqrt(POW2(x13) + POW2(x19) + POW2(x20));
    const scalar_t x22 = qx * qy;
    const scalar_t x23 = qx * x2;
    const scalar_t x24 = -1.0 / 2.0 * x15;
    const scalar_t x25 = tau * uh;
    const scalar_t x26 = x25 * sqrt(POW2(x12) + POW2(x14) + POW2(x8));
    const scalar_t x27 = qy * x0;
    element_vector[0] += qw * (tau * uh * x0 * x2 * x21 - x0 * x16 * x2 - x18 * (qy * x0 * x12 - qy * x8 - x17));
    element_vector[1] += -qw * (x18 * (qx * (-x11 * x5 + x4 * x9) + qy * x8 - x12 * x22) + x23 * x24 + x23 * x26);
    element_vector[2] += qw * (-x16 * x22 + x18 * (-qx * x13 - x19 * x23 + x2 * x20) + x21 * x22 * x25);
    element_vector[3] += qw * ((1.0 / 2.0) * uh * (x0 * x12 * x2 - x17 - x2 * x8) - x24 * x27 - x26 * x27);
}

static void SFEM_INLINE dg_hex8_sip_5(const scalar_t* const SFEM_RESTRICT x,
                                      const scalar_t* const SFEM_RESTRICT y,
                                      const scalar_t* const SFEM_RESTRICT z,
                                      const scalar_t* const SFEM_RESTRICT jacobian_inverse,
                                      const scalar_t                      qx,
                                      const scalar_t                      qy,
                                      const scalar_t                      qw,
                                      const scalar_t                      tau,
                                      const scalar_t* const SFEM_RESTRICT u,
                                      scalar_t* const SFEM_RESTRICT       element_vector) {
    scalar_t uh;
    scalar_t guh[3];
    {
        const scalar_t x0  = qx * qy;
        const scalar_t x1  = qy - 1;
        const scalar_t x2  = qx * x1;
        const scalar_t x3  = qx - 1;
        const scalar_t x4  = qy * x3;
        const scalar_t x5  = x1 * x3;
        const scalar_t x6  = jacobian_inverse[6] * x0;
        const scalar_t x7  = u[0] * x5;
        const scalar_t x8  = jacobian_inverse[0] * qy;
        const scalar_t x9  = jacobian_inverse[3] * qx;
        const scalar_t x10 = -x1;
        const scalar_t x11 = -x3;
        const scalar_t x12 = x10 * x11;
        const scalar_t x13 = jacobian_inverse[7] * x0;
        const scalar_t x14 = jacobian_inverse[1] * qy;
        const scalar_t x15 = jacobian_inverse[4] * qx;
        const scalar_t x16 = jacobian_inverse[8] * x0;
        const scalar_t x17 = jacobian_inverse[2] * qy;
        const scalar_t x18 = jacobian_inverse[5] * qx;
        uh                 = u[4] * x5 - u[5] * x2 + u[6] * x0 - u[7] * x4;
        guh[0] = jacobian_inverse[6] * qx * u[1] * x1 + jacobian_inverse[6] * qy * u[3] * x3 - jacobian_inverse[6] * x7 -
                 u[2] * x6 - u[4] * (jacobian_inverse[0] * x10 + jacobian_inverse[3] * x11 - jacobian_inverse[6] * x12) -
                 u[5] * (jacobian_inverse[0] * x1 + jacobian_inverse[6] * x2 + x9) + u[6] * (x6 + x8 + x9) -
                 u[7] * (jacobian_inverse[3] * x3 + jacobian_inverse[6] * x4 + x8);
        guh[1] = jacobian_inverse[7] * qx * u[1] * x1 + jacobian_inverse[7] * qy * u[3] * x3 - jacobian_inverse[7] * x7 -
                 u[2] * x13 - u[4] * (jacobian_inverse[1] * x10 + jacobian_inverse[4] * x11 - jacobian_inverse[7] * x12) -
                 u[5] * (jacobian_inverse[1] * x1 + jacobian_inverse[7] * x2 + x15) + u[6] * (x13 + x14 + x15) -
                 u[7] * (jacobian_inverse[4] * x3 + jacobian_inverse[7] * x4 + x14);
        guh[2] = jacobian_inverse[8] * qx * u[1] * x1 + jacobian_inverse[8] * qy * u[3] * x3 - jacobian_inverse[8] * x7 -
                 u[2] * x16 - u[4] * (jacobian_inverse[2] * x10 + jacobian_inverse[5] * x11 - jacobian_inverse[8] * x12) -
                 u[5] * (jacobian_inverse[2] * x1 + jacobian_inverse[8] * x2 + x18) + u[6] * (x16 + x17 + x18) -
                 u[7] * (jacobian_inverse[5] * x3 + jacobian_inverse[8] * x4 + x17);
    }

    const scalar_t x0  = qx - 1;
    const scalar_t x1  = qx * x[5] - qx * x[6] - x0 * x[4] + x0 * x[7];
    const scalar_t x2  = qy - 1;
    const scalar_t x3  = qy * y[6] - qy * y[7] + x2 * y[4] - x2 * y[5];
    const scalar_t x4  = qx * y[5] - qx * y[6] - x0 * y[4] + x0 * y[7];
    const scalar_t x5  = qy * x[6] - qy * x[7] + x2 * x[4] - x2 * x[5];
    const scalar_t x6  = -x1 * x3 + x4 * x5;
    const scalar_t x7  = (1.0 / 2.0) * uh;
    const scalar_t x8  = qy * z[6] - qy * z[7] + x2 * z[4] - x2 * z[5];
    const scalar_t x9  = qx * z[5] - qx * z[6] - x0 * z[4] + x0 * z[7];
    const scalar_t x10 = x1 * x8 - x5 * x9;
    const scalar_t x11 = tau * uh;
    const scalar_t x12 = -x10;
    const scalar_t x13 = -x4;
    const scalar_t x14 = x13 * x8 + x3 * x9;
    const scalar_t x15 = -x1 * x3 - x13 * x5;
    const scalar_t x16 = -1.0 / 2.0 * guh[0] * x14 + (1.0 / 2.0) * guh[1] * x12 - 1.0 / 2.0 * guh[2] * x15;
    const scalar_t x17 = x11 * sqrt(POW2(x10) + POW2(x6) + POW2(x3 * x9 - x4 * x8)) + x16 + x6 * x7;
    const scalar_t x18 = qw * x2;
    const scalar_t x19 = x11 * sqrt(POW2(x12) + POW2(x14) + POW2(x15)) + x15 * x7 + x16;
    const scalar_t x20 = qw * qy;
    element_vector[0] += x0 * x17 * x18;
    element_vector[1] += -qx * x18 * x19;
    element_vector[2] += qx * x17 * x20;
    element_vector[3] += -x0 * x19 * x20;
}

#endif  // DG_HEX8_SYMMETRIC_INTERIOR_PENALTY_INLINE_H
