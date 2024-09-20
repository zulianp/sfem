#ifndef HEX8_LINEAR_ELASTICITY_INLINE_CPU_H
#define HEX8_LINEAR_ELASTICITY_INLINE_CPU_H

#include "hex8_inline_cpu.h"

static SFEM_INLINE void hex8_linear_elasticity_apply_adj(const scalar_t mu,
                                                         const scalar_t lambda,
                                                         const scalar_t *const SFEM_RESTRICT
                                                                 adjugate,
                                                         const scalar_t jacobian_determinant,
                                                         const scalar_t qx,
                                                         const scalar_t qy,
                                                         const scalar_t qz,
                                                         const scalar_t qw,
                                                         const scalar_t *const SFEM_RESTRICT ux,
                                                         const scalar_t *const SFEM_RESTRICT uy,
                                                         const scalar_t *const SFEM_RESTRICT uz,
                                                         accumulator_t *const SFEM_RESTRICT outx,
                                                         accumulator_t *const SFEM_RESTRICT outy,
                                                         accumulator_t *const SFEM_RESTRICT outz) {
    scalar_t disp_grad[9];
    {
        const scalar_t x0 = 1.0 / jacobian_determinant;
        const scalar_t x1 = qx * qz;
        const scalar_t x2 = qz - 1;
        const scalar_t x3 = qx * x2;
        const scalar_t x4 = qx - 1;
        const scalar_t x5 = qz * x4;
        const scalar_t x6 = x2 * x4;
        const scalar_t x7 = ux[0] * x6 - ux[1] * x3 + ux[2] * x3 - ux[3] * x6 - ux[4] * x5 +
                            ux[5] * x1 - ux[6] * x1 + ux[7] * x5;
        const scalar_t x8 = qx * qy;
        const scalar_t x9 = qy - 1;
        const scalar_t x10 = qx * x9;
        const scalar_t x11 = qy * x4;
        const scalar_t x12 = x4 * x9;
        const scalar_t x13 = ux[0] * x12 - ux[1] * x10 + ux[2] * x8 - ux[3] * x11 - ux[4] * x12 +
                             ux[5] * x10 - ux[6] * x8 + ux[7] * x11;
        const scalar_t x14 = qy * qz;
        const scalar_t x15 = qy * x2;
        const scalar_t x16 = qz * x9;
        const scalar_t x17 = x2 * x9;
        const scalar_t x18 = -ux[0] * x17 + ux[1] * x17 - ux[2] * x15 + ux[3] * x15 + ux[4] * x16 -
                             ux[5] * x16 + ux[6] * x14 - ux[7] * x14;
        const scalar_t x19 = uy[0] * x6 - uy[1] * x3 + uy[2] * x3 - uy[3] * x6 - uy[4] * x5 +
                             uy[5] * x1 - uy[6] * x1 + uy[7] * x5;
        const scalar_t x20 = uy[0] * x12 - uy[1] * x10 + uy[2] * x8 - uy[3] * x11 - uy[4] * x12 +
                             uy[5] * x10 - uy[6] * x8 + uy[7] * x11;
        const scalar_t x21 = -uy[0] * x17 + uy[1] * x17 - uy[2] * x15 + uy[3] * x15 + uy[4] * x16 -
                             uy[5] * x16 + uy[6] * x14 - uy[7] * x14;
        const scalar_t x22 = uz[0] * x6 - uz[1] * x3 + uz[2] * x3 - uz[3] * x6 - uz[4] * x5 +
                             uz[5] * x1 - uz[6] * x1 + uz[7] * x5;
        const scalar_t x23 = uz[0] * x12 - uz[1] * x10 + uz[2] * x8 - uz[3] * x11 - uz[4] * x12 +
                             uz[5] * x10 - uz[6] * x8 + uz[7] * x11;
        const scalar_t x24 = -uz[0] * x17 + uz[1] * x17 - uz[2] * x15 + uz[3] * x15 + uz[4] * x16 -
                             uz[5] * x16 + uz[6] * x14 - uz[7] * x14;
        disp_grad[0] = x0 * (adjugate[0] * x18 - adjugate[3] * x7 - adjugate[6] * x13);
        disp_grad[1] = x0 * (adjugate[1] * x18 - adjugate[4] * x7 - adjugate[7] * x13);
        disp_grad[2] = x0 * (adjugate[2] * x18 - adjugate[5] * x7 - adjugate[8] * x13);
        disp_grad[3] = x0 * (adjugate[0] * x21 - adjugate[3] * x19 - adjugate[6] * x20);
        disp_grad[4] = x0 * (adjugate[1] * x21 - adjugate[4] * x19 - adjugate[7] * x20);
        disp_grad[5] = x0 * (adjugate[2] * x21 - adjugate[5] * x19 - adjugate[8] * x20);
        disp_grad[6] = x0 * (adjugate[0] * x24 - adjugate[3] * x22 - adjugate[6] * x23);
        disp_grad[7] = x0 * (adjugate[1] * x24 - adjugate[4] * x22 - adjugate[7] * x23);
        disp_grad[8] = x0 * (adjugate[2] * x24 - adjugate[5] * x22 - adjugate[8] * x23);
    }

    scalar_t *P_tXJinv_t = disp_grad;
    {
        const scalar_t x0 = mu * (disp_grad[1] + disp_grad[3]);
        const scalar_t x1 = mu * (disp_grad[2] + disp_grad[6]);
        const scalar_t x2 = 2 * mu;
        const scalar_t x3 = lambda * (disp_grad[0] + disp_grad[4] + disp_grad[8]);
        const scalar_t x4 = disp_grad[0] * x2 + x3;
        const scalar_t x5 = mu * (disp_grad[5] + disp_grad[7]);
        const scalar_t x6 = disp_grad[4] * x2 + x3;
        const scalar_t x7 = disp_grad[8] * x2 + x3;
        P_tXJinv_t[0] = adjugate[0] * x4 + adjugate[1] * x0 + adjugate[2] * x1;
        P_tXJinv_t[1] = adjugate[3] * x4 + adjugate[4] * x0 + adjugate[5] * x1;
        P_tXJinv_t[2] = adjugate[6] * x4 + adjugate[7] * x0 + adjugate[8] * x1;
        P_tXJinv_t[3] = adjugate[0] * x0 + adjugate[1] * x6 + adjugate[2] * x5;
        P_tXJinv_t[4] = adjugate[3] * x0 + adjugate[4] * x6 + adjugate[5] * x5;
        P_tXJinv_t[5] = adjugate[6] * x0 + adjugate[7] * x6 + adjugate[8] * x5;
        P_tXJinv_t[6] = adjugate[0] * x1 + adjugate[1] * x5 + adjugate[2] * x7;
        P_tXJinv_t[7] = adjugate[3] * x1 + adjugate[4] * x5 + adjugate[5] * x7;
        P_tXJinv_t[8] = adjugate[6] * x1 + adjugate[7] * x5 + adjugate[8] * x7;
    }

    {
        const scalar_t x0 = qy - 1;
        const scalar_t x1 = qz - 1;
        const scalar_t x2 = P_tXJinv_t[0] * x1;
        const scalar_t x3 = x0 * x2;
        const scalar_t x4 = qx - 1;
        const scalar_t x5 = P_tXJinv_t[1] * x1;
        const scalar_t x6 = x4 * x5;
        const scalar_t x7 = P_tXJinv_t[2] * x0;
        const scalar_t x8 = x4 * x7;
        const scalar_t x9 = qx * x5;
        const scalar_t x10 = qx * x7;
        const scalar_t x11 = P_tXJinv_t[2] * qy;
        const scalar_t x12 = qx * x11;
        const scalar_t x13 = qy * x2;
        const scalar_t x14 = x11 * x4;
        const scalar_t x15 = P_tXJinv_t[0] * qz;
        const scalar_t x16 = x0 * x15;
        const scalar_t x17 = P_tXJinv_t[1] * qz;
        const scalar_t x18 = x17 * x4;
        const scalar_t x19 = qx * x17;
        const scalar_t x20 = qy * x15;
        const scalar_t x21 = P_tXJinv_t[3] * x1;
        const scalar_t x22 = x0 * x21;
        const scalar_t x23 = P_tXJinv_t[4] * x1;
        const scalar_t x24 = x23 * x4;
        const scalar_t x25 = P_tXJinv_t[5] * x0;
        const scalar_t x26 = x25 * x4;
        const scalar_t x27 = qx * x23;
        const scalar_t x28 = qx * x25;
        const scalar_t x29 = P_tXJinv_t[5] * qy;
        const scalar_t x30 = qx * x29;
        const scalar_t x31 = qy * x21;
        const scalar_t x32 = x29 * x4;
        const scalar_t x33 = P_tXJinv_t[3] * qz;
        const scalar_t x34 = x0 * x33;
        const scalar_t x35 = P_tXJinv_t[4] * qz;
        const scalar_t x36 = x35 * x4;
        const scalar_t x37 = qx * x35;
        const scalar_t x38 = qy * x33;
        const scalar_t x39 = P_tXJinv_t[6] * x1;
        const scalar_t x40 = x0 * x39;
        const scalar_t x41 = P_tXJinv_t[7] * x1;
        const scalar_t x42 = x4 * x41;
        const scalar_t x43 = P_tXJinv_t[8] * x0;
        const scalar_t x44 = x4 * x43;
        const scalar_t x45 = qx * x41;
        const scalar_t x46 = qx * x43;
        const scalar_t x47 = P_tXJinv_t[8] * qy;
        const scalar_t x48 = qx * x47;
        const scalar_t x49 = qy * x39;
        const scalar_t x50 = x4 * x47;
        const scalar_t x51 = P_tXJinv_t[6] * qz;
        const scalar_t x52 = x0 * x51;
        const scalar_t x53 = P_tXJinv_t[7] * qz;
        const scalar_t x54 = x4 * x53;
        const scalar_t x55 = qx * x53;
        const scalar_t x56 = qy * x51;
        outx[0] += -x3 - x6 - x8;
        outx[1] += x10 + x3 + x9;
        outx[2] += -x12 - x13 - x9;
        outx[3] += x13 + x14 + x6;
        outx[4] += x16 + x18 + x8;
        outx[5] += -x10 - x16 - x19;
        outx[6] += x12 + x19 + x20;
        outx[7] += -x14 - x18 - x20;
        outy[0] += -x22 - x24 - x26;
        outy[1] += x22 + x27 + x28;
        outy[2] += -x27 - x30 - x31;
        outy[3] += x24 + x31 + x32;
        outy[4] += x26 + x34 + x36;
        outy[5] += -x28 - x34 - x37;
        outy[6] += x30 + x37 + x38;
        outy[7] += -x32 - x36 - x38;
        outz[0] += -x40 - x42 - x44;
        outz[1] += x40 + x45 + x46;
        outz[2] += -x45 - x48 - x49;
        outz[3] += x42 + x49 + x50;
        outz[4] += x44 + x52 + x54;
        outz[5] += -x46 - x52 - x55;
        outz[6] += x48 + x55 + x56;
        outz[7] += -x50 - x54 - x56;
    }
}

#endif  // HEX8_LINEAR_ELASTICITY_INLINE_CPU_H
