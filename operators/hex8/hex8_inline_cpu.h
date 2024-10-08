#ifndef HEX8_INLINE_CPU_H
#define HEX8_INLINE_CPU_H

#include "sfem_defs.h"

#ifndef POW2
#define POW2(a) ((a) * (a))
#endif

static SFEM_INLINE void hex8_fff(const scalar_t *const SFEM_RESTRICT x,
                                 const scalar_t *const SFEM_RESTRICT y,
                                 const scalar_t *const SFEM_RESTRICT z,
                                 const scalar_t qx,
                                 const scalar_t qy,
                                 const scalar_t qz,
                                 scalar_t *const SFEM_RESTRICT fff) {
    const scalar_t x0 = qy * qz;
    const scalar_t x1 = 1 - qz;
    const scalar_t x2 = qy * x1;
    const scalar_t x3 = 1 - qy;
    const scalar_t x4 = qz * x3;
    const scalar_t x5 = x1 * x3;
    const scalar_t x6 = x0 * z[6] - x0 * z[7] + x2 * z[2] - x2 * z[3] - x4 * z[4] + x4 * z[5] -
                        x5 * z[0] + x5 * z[1];
    const scalar_t x7 = qx * qy;
    const scalar_t x8 = qx * x3;
    const scalar_t x9 = 1 - qx;
    const scalar_t x10 = qy * x9;
    const scalar_t x11 = x3 * x9;
    const scalar_t x12 = qx * qy * x[6] + qx * x3 * x[5] + qy * x9 * x[7] - x10 * x[3] -
                         x11 * x[0] + x3 * x9 * x[4] - x7 * x[2] - x8 * x[1];
    const scalar_t x13 = qx * qz;
    const scalar_t x14 = qx * x1;
    const scalar_t x15 = qz * x9;
    const scalar_t x16 = x1 * x9;
    const scalar_t x17 = qx * qz * y[6] + qx * x1 * y[2] + qz * x9 * y[7] + x1 * x9 * y[3] -
                         x13 * y[5] - x14 * y[1] - x15 * y[4] - x16 * y[0];
    const scalar_t x18 = x12 * x17;
    const scalar_t x19 = x0 * x[6] - x0 * x[7] + x2 * x[2] - x2 * x[3] - x4 * x[4] + x4 * x[5] -
                         x5 * x[0] + x5 * x[1];
    const scalar_t x20 = qx * qy * y[6] + qx * x3 * y[5] + qy * x9 * y[7] - x10 * y[3] -
                         x11 * y[0] + x3 * x9 * y[4] - x7 * y[2] - x8 * y[1];
    const scalar_t x21 = qx * qz * z[6] + qx * x1 * z[2] + qz * x9 * z[7] + x1 * x9 * z[3] -
                         x13 * z[5] - x14 * z[1] - x15 * z[4] - x16 * z[0];
    const scalar_t x22 = x20 * x21;
    const scalar_t x23 = x0 * y[6] - x0 * y[7] + x2 * y[2] - x2 * y[3] - x4 * y[4] + x4 * y[5] -
                         x5 * y[0] + x5 * y[1];
    const scalar_t x24 = qx * qy * z[6] + qx * x3 * z[5] + qy * x9 * z[7] - x10 * z[3] -
                         x11 * z[0] + x3 * x9 * z[4] - x7 * z[2] - x8 * z[1];
    const scalar_t x25 = qx * qz * x[6] + qx * x1 * x[2] + qz * x9 * x[7] + x1 * x9 * x[3] -
                         x13 * x[5] - x14 * x[1] - x15 * x[4] - x16 * x[0];
    const scalar_t x26 = x24 * x25;
    const scalar_t x27 =
            x12 * x21 * x23 + x17 * x19 * x24 - x18 * x6 - x19 * x22 + x20 * x25 * x6 - x23 * x26;
    const scalar_t x28 = -x18 + x20 * x25;
    const scalar_t x29 = (1 / POW2(x27));
    const scalar_t x30 = x12 * x21 - x26;
    const scalar_t x31 = x17 * x24 - x22;
    const scalar_t x32 = x12 * x23 - x19 * x20;
    const scalar_t x33 = x28 * x29;
    const scalar_t x34 = -x12 * x6 + x19 * x24;
    const scalar_t x35 = x29 * x30;
    const scalar_t x36 = x20 * x6 - x23 * x24;
    const scalar_t x37 = x29 * x31;
    const scalar_t x38 = x17 * x19 - x23 * x25;
    const scalar_t x39 = -x19 * x21 + x25 * x6;
    const scalar_t x40 = -x17 * x6 + x21 * x23;
    fff[0] = x27 * (POW2(x28) * x29 + x29 * POW2(x30) + x29 * POW2(x31));
    fff[1] = x27 * (x32 * x33 + x34 * x35 + x36 * x37);
    fff[2] = x27 * (x33 * x38 + x35 * x39 + x37 * x40);
    fff[3] = x27 * (x29 * POW2(x32) + x29 * POW2(x34) + x29 * POW2(x36));
    fff[4] = x27 * (x29 * x32 * x38 + x29 * x34 * x39 + x29 * x36 * x40);
    fff[5] = x27 * (x29 * POW2(x38) + x29 * POW2(x39) + x29 * POW2(x40));
}

static SFEM_INLINE void aahex8_jac_diag(const scalar_t px0,
                                        const scalar_t px6,
                                        const scalar_t py0,
                                        const scalar_t py6,
                                        const scalar_t pz0,
                                        const scalar_t pz6,
                                        scalar_t *const SFEM_RESTRICT jac_diag) {
    const scalar_t x0 = -py0 + py6;
    const scalar_t x1 = -pz0 + pz6;
    const scalar_t x2 = -px0 + px6;
    jac_diag[0] = x0 * x1 / x2;
    jac_diag[1] = x1 * x2 / x0;
    jac_diag[2] = x0 * x2 / x1;
}

static SFEM_INLINE void hex8_adjugate_and_det(const scalar_t *const SFEM_RESTRICT x,
                                              const scalar_t *const SFEM_RESTRICT y,
                                              const scalar_t *const SFEM_RESTRICT z,
                                              const scalar_t qx,
                                              const scalar_t qy,
                                              const scalar_t qz,
                                              scalar_t *const SFEM_RESTRICT adjugate,
                                              scalar_t *const SFEM_RESTRICT jacobian_determinant) {
    scalar_t jacobian[9];
    {
        const scalar_t x0 = qy * qz;
        const scalar_t x1 = 1 - qz;
        const scalar_t x2 = qy * x1;
        const scalar_t x3 = 1 - qy;
        const scalar_t x4 = qz * x3;
        const scalar_t x5 = x1 * x3;
        const scalar_t x6 = qx * qz;
        const scalar_t x7 = qx * x1;
        const scalar_t x8 = 1 - qx;
        const scalar_t x9 = qz * x8;
        const scalar_t x10 = x1 * x8;
        const scalar_t x11 = qx * qy;
        const scalar_t x12 = qx * x3;
        const scalar_t x13 = qy * x8;
        const scalar_t x14 = x3 * x8;

        jacobian[0] = x0 * x[6] - x0 * x[7] + x2 * x[2] - x2 * x[3] - x4 * x[4] + x4 * x[5] -
                      x5 * x[0] + x5 * x[1];
        jacobian[1] = qx * qz * x[6] + qx * x1 * x[2] + qz * x8 * x[7] + x1 * x8 * x[3] -
                      x10 * x[0] - x6 * x[5] - x7 * x[1] - x9 * x[4];
        jacobian[2] = qx * qy * x[6] + qx * x3 * x[5] + qy * x8 * x[7] - x11 * x[2] - x12 * x[1] -
                      x13 * x[3] - x14 * x[0] + x3 * x8 * x[4];
        jacobian[3] = x0 * y[6] - x0 * y[7] + x2 * y[2] - x2 * y[3] - x4 * y[4] + x4 * y[5] -
                      x5 * y[0] + x5 * y[1];
        jacobian[4] = qx * qz * y[6] + qx * x1 * y[2] + qz * x8 * y[7] + x1 * x8 * y[3] -
                      x10 * y[0] - x6 * y[5] - x7 * y[1] - x9 * y[4];
        jacobian[5] = qx * qy * y[6] + qx * x3 * y[5] + qy * x8 * y[7] - x11 * y[2] - x12 * y[1] -
                      x13 * y[3] - x14 * y[0] + x3 * x8 * y[4];
        jacobian[6] = x0 * z[6] - x0 * z[7] + x2 * z[2] - x2 * z[3] - x4 * z[4] + x4 * z[5] -
                      x5 * z[0] + x5 * z[1];
        jacobian[7] = qx * qz * z[6] + qx * x1 * z[2] + qz * x8 * z[7] + x1 * x8 * z[3] -
                      x10 * z[0] - x6 * z[5] - x7 * z[1] - x9 * z[4];
        jacobian[8] = qx * qy * z[6] + qx * x3 * z[5] + qy * x8 * z[7] - x11 * z[2] - x12 * z[1] -
                      x13 * z[3] - x14 * z[0] + x3 * x8 * z[4];
    }

    const scalar_t x0 = jacobian[4] * jacobian[8];
    const scalar_t x1 = jacobian[5] * jacobian[7];
    const scalar_t x2 = jacobian[1] * jacobian[8];
    const scalar_t x3 = jacobian[1] * jacobian[5];
    const scalar_t x4 = jacobian[2] * jacobian[4];

    adjugate[0] = x0 - x1;
    adjugate[1] = jacobian[2] * jacobian[7] - x2;
    adjugate[2] = x3 - x4;
    adjugate[3] = -jacobian[3] * jacobian[8] + jacobian[5] * jacobian[6];
    adjugate[4] = jacobian[0] * jacobian[8] - jacobian[2] * jacobian[6];
    adjugate[5] = -jacobian[0] * jacobian[5] + jacobian[2] * jacobian[3];
    adjugate[6] = jacobian[3] * jacobian[7] - jacobian[4] * jacobian[6];
    adjugate[7] = -jacobian[0] * jacobian[7] + jacobian[1] * jacobian[6];
    adjugate[8] = jacobian[0] * jacobian[4] - jacobian[1] * jacobian[3];
    *jacobian_determinant = jacobian[0] * x0 - jacobian[0] * x1 +
                            jacobian[2] * jacobian[3] * jacobian[7] - jacobian[3] * x2 +
                            jacobian[6] * x3 - jacobian[6] * x4;
}

#endif  // HEX8_INLINE_CPU_H
