#ifndef TET10_LINEAR_ELASTICITY_INLINE_CPU_H
#define TET10_LINEAR_ELASTICITY_INLINE_CPU_H

#include "tet10_inline_cpu.h"

static SFEM_INLINE void tet10_linear_elasticity_diag_adj(const jacobian_t *const SFEM_RESTRICT
                                                                 adjugate,
                                                         const jacobian_t jacobian_determinant,
                                                         const scalar_t mu,
                                                         const scalar_t lambda,
                                                         const scalar_t qx,
                                                         const scalar_t qy,
                                                         const scalar_t qz,
                                                         const scalar_t qw,
                                                         accumulator_t *const SFEM_RESTRICT outx,
                                                         accumulator_t *const SFEM_RESTRICT outy,
                                                         accumulator_t *const SFEM_RESTRICT outz) {
    const scalar_t x0 = POW2(adjugate[1] + adjugate[4] + adjugate[7]);
    const scalar_t x1 = mu * x0;
    const scalar_t x2 = POW2(adjugate[2] + adjugate[5] + adjugate[8]);
    const scalar_t x3 = mu * x2;
    const scalar_t x4 = lambda + 2 * mu;
    const scalar_t x5 = POW2(adjugate[0] + adjugate[3] + adjugate[6]);
    const scalar_t x6 = 4 * qx;
    const scalar_t x7 = 4 * qy;
    const scalar_t x8 = 4 * qz;
    const scalar_t x9 = 1.0 / jacobian_determinant;
    const scalar_t x10 = (1.0 / 6.0) * x9;
    const scalar_t x11 = x10 * POW2(x6 + x7 + x8 - 3);
    const scalar_t x12 = POW2(adjugate[1]);
    const scalar_t x13 = mu * x12;
    const scalar_t x14 = POW2(adjugate[2]);
    const scalar_t x15 = mu * x14;
    const scalar_t x16 = POW2(adjugate[0]);
    const scalar_t x17 = x10 * POW2(x6 - 1);
    const scalar_t x18 = POW2(adjugate[4]);
    const scalar_t x19 = mu * x18;
    const scalar_t x20 = POW2(adjugate[5]);
    const scalar_t x21 = mu * x20;
    const scalar_t x22 = POW2(adjugate[3]);
    const scalar_t x23 = x10 * POW2(x7 - 1);
    const scalar_t x24 = POW2(adjugate[7]);
    const scalar_t x25 = mu * x24;
    const scalar_t x26 = POW2(adjugate[8]);
    const scalar_t x27 = mu * x26;
    const scalar_t x28 = POW2(adjugate[6]);
    const scalar_t x29 = x10 * POW2(x8 - 1);
    const scalar_t x30 = adjugate[4] * qx;
    const scalar_t x31 = adjugate[7] * qx;
    const scalar_t x32 = qz - 1;
    const scalar_t x33 = 2 * qx + qy + x32;
    const scalar_t x34 = POW2(adjugate[1] * x33 + x30 + x31);
    const scalar_t x35 = mu * x34;
    const scalar_t x36 = adjugate[5] * qx;
    const scalar_t x37 = adjugate[8] * qx;
    const scalar_t x38 = POW2(adjugate[2] * x33 + x36 + x37);
    const scalar_t x39 = mu * x38;
    const scalar_t x40 = adjugate[3] * qx;
    const scalar_t x41 = adjugate[6] * qx;
    const scalar_t x42 = POW2(adjugate[0] * x33 + x40 + x41);
    const scalar_t x43 = (8.0 / 3.0) * x9;
    const scalar_t x44 = adjugate[1] * qy;
    const scalar_t x45 = POW2(x30 + x44);
    const scalar_t x46 = mu * x45;
    const scalar_t x47 = adjugate[2] * qy;
    const scalar_t x48 = POW2(x36 + x47);
    const scalar_t x49 = mu * x48;
    const scalar_t x50 = adjugate[0] * qy;
    const scalar_t x51 = POW2(x40 + x50);
    const scalar_t x52 = adjugate[7] * qy;
    const scalar_t x53 = qx + 2 * qy + x32;
    const scalar_t x54 = POW2(adjugate[4] * x53 + x44 + x52);
    const scalar_t x55 = mu * x54;
    const scalar_t x56 = adjugate[8] * qy;
    const scalar_t x57 = POW2(adjugate[5] * x53 + x47 + x56);
    const scalar_t x58 = mu * x57;
    const scalar_t x59 = adjugate[6] * qy;
    const scalar_t x60 = POW2(adjugate[3] * x53 + x50 + x59);
    const scalar_t x61 = adjugate[1] * qz;
    const scalar_t x62 = adjugate[4] * qz;
    const scalar_t x63 = qx + qy + 2 * qz - 1;
    const scalar_t x64 = POW2(adjugate[7] * x63 + x61 + x62);
    const scalar_t x65 = mu * x64;
    const scalar_t x66 = adjugate[2] * qz;
    const scalar_t x67 = adjugate[5] * qz;
    const scalar_t x68 = POW2(adjugate[8] * x63 + x66 + x67);
    const scalar_t x69 = mu * x68;
    const scalar_t x70 = adjugate[0] * qz;
    const scalar_t x71 = adjugate[3] * qz;
    const scalar_t x72 = POW2(adjugate[6] * x63 + x70 + x71);
    const scalar_t x73 = POW2(x31 + x61);
    const scalar_t x74 = mu * x73;
    const scalar_t x75 = POW2(x37 + x66);
    const scalar_t x76 = mu * x75;
    const scalar_t x77 = POW2(x41 + x70);
    const scalar_t x78 = POW2(x52 + x62);
    const scalar_t x79 = mu * x78;
    const scalar_t x80 = POW2(x56 + x67);
    const scalar_t x81 = mu * x80;
    const scalar_t x82 = POW2(x59 + x71);
    const scalar_t x83 = mu * x5;
    const scalar_t x84 = mu * x16;
    const scalar_t x85 = mu * x22;
    const scalar_t x86 = mu * x28;
    const scalar_t x87 = mu * x42;
    const scalar_t x88 = mu * x51;
    const scalar_t x89 = mu * x60;
    const scalar_t x90 = mu * x72;
    const scalar_t x91 = mu * x77;
    const scalar_t x92 = mu * x82;

    outx[0] += qw * (x11 * (x1 + x3 + x4 * x5));
    outx[1] += qw * (x17 * (x13 + x15 + x16 * x4));
    outx[2] += qw * (x23 * (x19 + x21 + x22 * x4));
    outx[3] += qw * (x29 * (x25 + x27 + x28 * x4));
    outx[4] += qw * (x43 * (x35 + x39 + x4 * x42));
    outx[5] += qw * (x43 * (x4 * x51 + x46 + x49));
    outx[6] += qw * (x43 * (x4 * x60 + x55 + x58));
    outx[7] += qw * (x43 * (x4 * x72 + x65 + x69));
    outx[8] += qw * (x43 * (x4 * x77 + x74 + x76));
    outx[9] += qw * (x43 * (x4 * x82 + x79 + x81));

    outy[0] += qw * (x11 * (x0 * x4 + x3 + x83));
    outy[1] += qw * (x17 * (x12 * x4 + x15 + x84));
    outy[2] += qw * (x23 * (x18 * x4 + x21 + x85));
    outy[3] += qw * (x29 * (x24 * x4 + x27 + x86));
    outy[4] += qw * (x43 * (x34 * x4 + x39 + x87));
    outy[5] += qw * (x43 * (x4 * x45 + x49 + x88));
    outy[6] += qw * (x43 * (x4 * x54 + x58 + x89));
    outy[7] += qw * (x43 * (x4 * x64 + x69 + x90));
    outy[8] += qw * (x43 * (x4 * x73 + x76 + x91));
    outy[9] += qw * (x43 * (x4 * x78 + x81 + x92));

    outz[0] += qw * (x11 * (x1 + x2 * x4 + x83));
    outz[1] += qw * (x17 * (x13 + x14 * x4 + x84));
    outz[2] += qw * (x23 * (x19 + x20 * x4 + x85));
    outz[3] += qw * (x29 * (x25 + x26 * x4 + x86));
    outz[4] += qw * (x43 * (x35 + x38 * x4 + x87));
    outz[5] += qw * (x43 * (x4 * x48 + x46 + x88));
    outz[6] += qw * (x43 * (x4 * x57 + x55 + x89));
    outz[7] += qw * (x43 * (x4 * x68 + x65 + x90));
    outz[8] += qw * (x43 * (x4 * x75 + x74 + x91));
    outz[9] += qw * (x43 * (x4 * x80 + x79 + x92));
}

#endif  // TET10_LINEAR_ELASTICITY_INLINE_CPU_H
