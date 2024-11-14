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
        outx[0] += qw * (-x3 - x6 - x8);
        outx[1] += qw * (x10 + x3 + x9);
        outx[2] += qw * (-x12 - x13 - x9);
        outx[3] += qw * (x13 + x14 + x6);
        outx[4] += qw * (x16 + x18 + x8);
        outx[5] += qw * (-x10 - x16 - x19);
        outx[6] += qw * (x12 + x19 + x20);
        outx[7] += qw * (-x14 - x18 - x20);
        outy[0] += qw * (-x22 - x24 - x26);
        outy[1] += qw * (x22 + x27 + x28);
        outy[2] += qw * (-x27 - x30 - x31);
        outy[3] += qw * (x24 + x31 + x32);
        outy[4] += qw * (x26 + x34 + x36);
        outy[5] += qw * (-x28 - x34 - x37);
        outy[6] += qw * (x30 + x37 + x38);
        outy[7] += qw * (-x32 - x36 - x38);
        outz[0] += qw * (-x40 - x42 - x44);
        outz[1] += qw * (x40 + x45 + x46);
        outz[2] += qw * (-x45 - x48 - x49);
        outz[3] += qw * (x42 + x49 + x50);
        outz[4] += qw * (x44 + x52 + x54);
        outz[5] += qw * (-x46 - x52 - x55);
        outz[6] += qw * (x48 + x55 + x56);
        outz[7] += qw * (-x50 - x54 - x56);
    }
}

static SFEM_INLINE void hex8_linear_elasticity_matrix(const scalar_t mu,
                                                      const scalar_t lambda,
                                                      const scalar_t *const SFEM_RESTRICT adjugate,
                                                      const scalar_t jacobian_determinant,
                                                      scalar_t *const SFEM_RESTRICT
                                                              element_matrix) {
    const scalar_t x0 = POW2(adjugate[7]);
    const scalar_t x1 = mu * x0;
    const scalar_t x2 = 2 * x1;
    const scalar_t x3 = 3 * mu;
    const scalar_t x4 = adjugate[4] * adjugate[7];
    const scalar_t x5 = x3 * x4;
    const scalar_t x6 = x2 + x5;
    const scalar_t x7 = adjugate[5] * adjugate[8];
    const scalar_t x8 = x3 * x7;
    const scalar_t x9 = POW2(adjugate[5]);
    const scalar_t x10 = mu * x9;
    const scalar_t x11 = 2 * x10;
    const scalar_t x12 = POW2(adjugate[8]);
    const scalar_t x13 = mu * x12;
    const scalar_t x14 = 2 * x13;
    const scalar_t x15 = x11 + x14;
    const scalar_t x16 = x15 + x8;
    const scalar_t x17 = adjugate[3] * adjugate[6];
    const scalar_t x18 = lambda * x17;
    const scalar_t x19 = 3 * x18;
    const scalar_t x20 = 6 * mu;
    const scalar_t x21 = x17 * x20;
    const scalar_t x22 = x19 + x21;
    const scalar_t x23 = POW2(adjugate[4]);
    const scalar_t x24 = mu * x23;
    const scalar_t x25 = 2 * x24;
    const scalar_t x26 = POW2(adjugate[3]);
    const scalar_t x27 = lambda * x26;
    const scalar_t x28 = 2 * x27;
    const scalar_t x29 = mu * x26;
    const scalar_t x30 = 4 * x29;
    const scalar_t x31 = x28 + x30;
    const scalar_t x32 = x25 + x31;
    const scalar_t x33 = POW2(adjugate[6]);
    const scalar_t x34 = lambda * x33;
    const scalar_t x35 = 2 * x34;
    const scalar_t x36 = mu * x33;
    const scalar_t x37 = 4 * x36;
    const scalar_t x38 = x35 + x37;
    const scalar_t x39 = x32 + x38;
    const scalar_t x40 = x16 + x22 + x39 + x6;
    const scalar_t x41 = (1 / POW2(jacobian_determinant));
    const scalar_t x42 = (1.0 / 18.0) * x41;
    const scalar_t x43 = x40 * x42;
    const scalar_t x44 = adjugate[0] * lambda;
    const scalar_t x45 = adjugate[6] * x44;
    const scalar_t x46 = adjugate[1] * mu;
    const scalar_t x47 = adjugate[7] * x46;
    const scalar_t x48 = adjugate[2] * mu;
    const scalar_t x49 = adjugate[8] * x48;
    const scalar_t x50 = adjugate[0] * mu;
    const scalar_t x51 = adjugate[6] * x50;
    const scalar_t x52 = 2 * x51;
    const scalar_t x53 = x45 + x47 + x49 + x52;
    const scalar_t x54 = adjugate[3] * x44;
    const scalar_t x55 = adjugate[4] * x46;
    const scalar_t x56 = adjugate[5] * x48;
    const scalar_t x57 = adjugate[3] * x50;
    const scalar_t x58 = 2 * x57;
    const scalar_t x59 = x54 + x55 + x56 + x58;
    const scalar_t x60 = x53 + x59;
    const scalar_t x61 = (1.0 / 6.0) * x41;
    const scalar_t x62 = 6 * x54;
    const scalar_t x63 = 6 * x55;
    const scalar_t x64 = 12 * x57;
    const scalar_t x65 = POW2(adjugate[1]);
    const scalar_t x66 = mu * x65;
    const scalar_t x67 = 2 * x66;
    const scalar_t x68 = 6 * x56;
    const scalar_t x69 = x67 + x68;
    const scalar_t x70 = x62 + x63 + x64 + x69;
    const scalar_t x71 = POW2(adjugate[2]);
    const scalar_t x72 = mu * x71;
    const scalar_t x73 = 2 * x72;
    const scalar_t x74 = POW2(adjugate[0]);
    const scalar_t x75 = lambda * x74;
    const scalar_t x76 = 2 * x75;
    const scalar_t x77 = mu * x74;
    const scalar_t x78 = 4 * x77;
    const scalar_t x79 = x76 + x78;
    const scalar_t x80 = x73 + x79;
    const scalar_t x81 = 6 * x45;
    const scalar_t x82 = 6 * x47;
    const scalar_t x83 = 6 * x49;
    const scalar_t x84 = 12 * x51;
    const scalar_t x85 = x81 + x82 + x83 + x84;
    const scalar_t x86 = x80 + x85;
    const scalar_t x87 = mu * x17;
    const scalar_t x88 = 18 * x87;
    const scalar_t x89 = 9 * x18;
    const scalar_t x90 = 6 * x27;
    const scalar_t x91 = 6 * x34;
    const scalar_t x92 = 12 * x29;
    const scalar_t x93 = 12 * x36;
    const scalar_t x94 = x90 + x91 + x92 + x93;
    const scalar_t x95 = 9 * mu;
    const scalar_t x96 = x7 * x95;
    const scalar_t x97 = 6 * x10;
    const scalar_t x98 = 6 * x13;
    const scalar_t x99 = x97 + x98;
    const scalar_t x100 = x96 + x99;
    const scalar_t x101 = x4 * x95;
    const scalar_t x102 = 6 * x24;
    const scalar_t x103 = 6 * x1;
    const scalar_t x104 = x102 + x103;
    const scalar_t x105 = x101 + x104;
    const scalar_t x106 = x100 + x105 + x88 + x89 + x94;
    const scalar_t x107 = 3 * x45;
    const scalar_t x108 = 6 * x51;
    const scalar_t x109 = x108 + x73;
    const scalar_t x110 = x107 + x109 + x79;
    const scalar_t x111 = 3 * x54;
    const scalar_t x112 = 6 * x57;
    const scalar_t x113 = x112 + x67;
    const scalar_t x114 = x111 + x113;
    const scalar_t x115 = 3 * x56;
    const scalar_t x116 = 3 * x49;
    const scalar_t x117 = x115 + x116;
    const scalar_t x118 = 3 * x55;
    const scalar_t x119 = 3 * x47;
    const scalar_t x120 = x118 + x119;
    const scalar_t x121 = x117 + x120;
    const scalar_t x122 = -x40 * x42;
    const scalar_t x123 = 2 * x49;
    const scalar_t x124 = 2 * x56;
    const scalar_t x125 = x123 + x124;
    const scalar_t x126 = 2 * x47;
    const scalar_t x127 = 2 * x55;
    const scalar_t x128 = x126 + x127;
    const scalar_t x129 = 2 * x45;
    const scalar_t x130 = 4 * x51;
    const scalar_t x131 = x129 + x130;
    const scalar_t x132 = 2 * x54;
    const scalar_t x133 = 4 * x57;
    const scalar_t x134 = x132 + x133;
    const scalar_t x135 = (1.0 / 12.0) * x41;
    const scalar_t x136 = jacobian_determinant * (x122 + x135 * (x125 + x128 + x131 + x134 + x40) +
                                                  x42 * (-x110 - x114 - x121));
    const scalar_t x137 = x116 + x119;
    const scalar_t x138 = x110 + x137;
    const scalar_t x139 = (1.0 / 36.0) * x41;
    const scalar_t x140 = -x28;
    const scalar_t x141 = -x11;
    const scalar_t x142 = x13 + x141;
    const scalar_t x143 = -x30;
    const scalar_t x144 = 2 * x36;
    const scalar_t x145 = x143 + x144;
    const scalar_t x146 = -x25;
    const scalar_t x147 = x1 + x146;
    const scalar_t x148 = -x1;
    const scalar_t x149 = x11 - x13;
    const scalar_t x150 = -x144;
    const scalar_t x151 = x150 + x25;
    const scalar_t x152 = x148 + x149 + x151 + x31 - x34;
    const scalar_t x153 = x152 * x42;
    const scalar_t x154 = x135 * (x140 + x142 + x145 + x147 + x34 + x53) + x153;
    const scalar_t x155 = jacobian_determinant * (x139 * (-x138 - x70) + x154);
    const scalar_t x156 = -x152 * x42;
    const scalar_t x157 = 4 * x24;
    const scalar_t x158 = -x157;
    const scalar_t x159 = x158 + x2;
    const scalar_t x160 = 4 * x10;
    const scalar_t x161 = -x160;
    const scalar_t x162 = x14 + x161;
    const scalar_t x163 = x107 + x108;
    const scalar_t x164 = x137 + x163;
    const scalar_t x165 = 6 * x36;
    const scalar_t x166 = -x97;
    const scalar_t x167 = 3 * x13 + x166;
    const scalar_t x168 = -x102;
    const scalar_t x169 = 3 * x1 + x168;
    const scalar_t x170 = 2 * x77;
    const scalar_t x171 = x170 + x66 + x72 + x75;
    const scalar_t x172 = x171 - x90 - x92;
    const scalar_t x173 = x165 + x167 + x169 + x172 + 3 * x34;
    const scalar_t x174 = jacobian_determinant *
                          (x135 * (4 * lambda * x26 + 8 * mu * x26 - x159 - x162 - x38 - x53) +
                           x156 + x42 * (x164 + x173));
    const scalar_t x175 = -x2;
    const scalar_t x176 = 2 * x29;
    const scalar_t x177 = x175 + x176;
    const scalar_t x178 = -x14;
    const scalar_t x179 = x10 + x178;
    const scalar_t x180 = x24 + x27;
    const scalar_t x181 = -x37;
    const scalar_t x182 = x181 - x35;
    const scalar_t x183 = x177 + x179 + x180 + x182;
    const scalar_t x184 = x183 * x42;
    const scalar_t x185 = 4 * x1;
    const scalar_t x186 = -x185;
    const scalar_t x187 = 4 * x13;
    const scalar_t x188 = -x187;
    const scalar_t x189 = x11 + x188;
    const scalar_t x190 = x115 + x118;
    const scalar_t x191 = x111 + x112;
    const scalar_t x192 = x171 + x191;
    const scalar_t x193 = x190 + x192;
    const scalar_t x194 = 6 * x29;
    const scalar_t x195 = -x98;
    const scalar_t x196 = 3 * x10 + x195;
    const scalar_t x197 = -x103;
    const scalar_t x198 = x197 + 3 * x24;
    const scalar_t x199 = -x91 - x93;
    const scalar_t x200 = x194 + x196 + x198 + x199 + 3 * x27;
    const scalar_t x201 = jacobian_determinant *
                          (x135 * (4 * lambda * x33 + 8 * mu * x33 - x186 - x189 - x32 - x59) +
                           x184 + x42 * (x193 + x200));
    const scalar_t x202 = x114 + x190 + x80;
    const scalar_t x203 = -x183 * x42;
    const scalar_t x204 = x135 * (x183 + x59) + x203;
    const scalar_t x205 = jacobian_determinant * (x139 * (-x202 - x85) + x204);
    const scalar_t x206 = x121 + x163;
    const scalar_t x207 = x1 + x5;
    const scalar_t x208 = x10 + x13;
    const scalar_t x209 = x208 + x8;
    const scalar_t x210 = x144 + x176;
    const scalar_t x211 = x180 + x210 + x34;
    const scalar_t x212 = x207 + x209 + x211 + x22;
    const scalar_t x213 = -x212;
    const scalar_t x214 = x135 * x213 + x212 * x42;
    const scalar_t x215 = jacobian_determinant * (x139 * (-x192 - x206) + x214);
    const scalar_t x216 = x213 * x42;
    const scalar_t x217 = -x66;
    const scalar_t x218 = -x72;
    const scalar_t x219 = -x170;
    const scalar_t x220 = x217 + x218 + x219 - x75;
    const scalar_t x221 = 18 * mu;
    const scalar_t x222 = x221 * x7;
    const scalar_t x223 = x222 + x99;
    const scalar_t x224 = x221 * x4;
    const scalar_t x225 = x104 + x224;
    const scalar_t x226 = 18 * x18 + 36 * x87;
    const scalar_t x227 =
            jacobian_determinant * (x139 * (-x220 - x223 - x225 - x226 - x94) + x212 * x61 + x216);
    const scalar_t x228 = adjugate[3] * adjugate[7];
    const scalar_t x229 = x228 * x3;
    const scalar_t x230 = 3 * lambda;
    const scalar_t x231 = adjugate[4] * adjugate[6];
    const scalar_t x232 = x230 * x231;
    const scalar_t x233 = x229 + x232;
    const scalar_t x234 = 4 * lambda;
    const scalar_t x235 = adjugate[3] * adjugate[4];
    const scalar_t x236 = x234 * x235;
    const scalar_t x237 = 4 * mu;
    const scalar_t x238 = x235 * x237;
    const scalar_t x239 = x236 + x238;
    const scalar_t x240 = x233 + x239;
    const scalar_t x241 = x228 * x230;
    const scalar_t x242 = x231 * x3;
    const scalar_t x243 = x241 + x242;
    const scalar_t x244 = adjugate[6] * adjugate[7];
    const scalar_t x245 = x234 * x244;
    const scalar_t x246 = x237 * x244;
    const scalar_t x247 = x245 + x246;
    const scalar_t x248 = x243 + x247;
    const scalar_t x249 = x240 + x248;
    const scalar_t x250 = x139 * x249;
    const scalar_t x251 = adjugate[7] * x44;
    const scalar_t x252 = adjugate[7] * x50;
    const scalar_t x253 = adjugate[1] * lambda;
    const scalar_t x254 = adjugate[6] * x253;
    const scalar_t x255 = adjugate[6] * x46;
    const scalar_t x256 = x251 + x252 + x254 + x255;
    const scalar_t x257 = adjugate[4] * x44;
    const scalar_t x258 = adjugate[4] * x50;
    const scalar_t x259 = adjugate[3] * x253;
    const scalar_t x260 = adjugate[3] * x46;
    const scalar_t x261 = x257 + x258 + x259 + x260;
    const scalar_t x262 = x256 + x261;
    const scalar_t x263 = 9 * lambda;
    const scalar_t x264 = x228 * x263;
    const scalar_t x265 = x231 * x95;
    const scalar_t x266 = x264 + x265;
    const scalar_t x267 = x228 * x95;
    const scalar_t x268 = x231 * x263;
    const scalar_t x269 = x267 + x268;
    const scalar_t x270 = x266 + x269;
    const scalar_t x271 = adjugate[1] * x44;
    const scalar_t x272 = 4 * x271;
    const scalar_t x273 = adjugate[0] * x46;
    const scalar_t x274 = 4 * x273;
    const scalar_t x275 = 6 * x252;
    const scalar_t x276 = 6 * x254;
    const scalar_t x277 = x275 + x276;
    const scalar_t x278 = 6 * x251;
    const scalar_t x279 = 6 * x255;
    const scalar_t x280 = x278 + x279;
    const scalar_t x281 = 12 * x244;
    const scalar_t x282 = lambda * x281;
    const scalar_t x283 = mu * x281;
    const scalar_t x284 = x282 + x283;
    const scalar_t x285 = 12 * x235;
    const scalar_t x286 = lambda * x285;
    const scalar_t x287 = mu * x285;
    const scalar_t x288 = x286 + x287;
    const scalar_t x289 = x272 + x274 + x277 + x280 + x284 + x288;
    const scalar_t x290 = 6 * x258;
    const scalar_t x291 = 6 * x259;
    const scalar_t x292 = x290 + x291;
    const scalar_t x293 = 6 * x257;
    const scalar_t x294 = 6 * x260;
    const scalar_t x295 = x293 + x294;
    const scalar_t x296 = x292 + x295;
    const scalar_t x297 =
            jacobian_determinant * (x135 * (-x249 - x262) + x139 * (x270 + x289 + x296) + x250);
    const scalar_t x298 = 3 * x258;
    const scalar_t x299 = 3 * x259;
    const scalar_t x300 = x298 + x299;
    const scalar_t x301 = 2 * x271;
    const scalar_t x302 = 2 * x273;
    const scalar_t x303 = x301 + x302;
    const scalar_t x304 = 3 * x252;
    const scalar_t x305 = 3 * x254;
    const scalar_t x306 = x304 + x305;
    const scalar_t x307 = x303 + x306;
    const scalar_t x308 = 2 * x252;
    const scalar_t x309 = 2 * x254;
    const scalar_t x310 = x308 + x309;
    const scalar_t x311 = 2 * x251;
    const scalar_t x312 = 2 * x255;
    const scalar_t x313 = x311 + x312;
    const scalar_t x314 = 2 * x258;
    const scalar_t x315 = 2 * x259;
    const scalar_t x316 = x314 + x315;
    const scalar_t x317 = 2 * x257;
    const scalar_t x318 = 2 * x260;
    const scalar_t x319 = x317 + x318;
    const scalar_t x320 = x316 + x319;
    const scalar_t x321 = x310 + x313 + x320;
    const scalar_t x322 = (1.0 / 24.0) * x41;
    const scalar_t x323 = -x139 * x249;
    const scalar_t x324 = x322 * (x249 + x321) + x323;
    const scalar_t x325 = jacobian_determinant * (x324 + x42 * (-x300 - x307));
    const scalar_t x326 = x292 + x307;
    const scalar_t x327 = x256 + x316 - x317 - x318;
    const scalar_t x328 = -x229 - x232;
    const scalar_t x329 = x243 + x328;
    const scalar_t x330 = 2 * x244;
    const scalar_t x331 = lambda * x330;
    const scalar_t x332 = mu * x330;
    const scalar_t x333 = x331 + x332;
    const scalar_t x334 = -x236 - x238;
    const scalar_t x335 = x333 + x334;
    const scalar_t x336 = -x241 - x242;
    const scalar_t x337 = -x331 - x332;
    const scalar_t x338 = x240 + x336 + x337;
    const scalar_t x339 = x139 * x338;
    const scalar_t x340 = x322 * (x327 + x329 + x335) + x339;
    const scalar_t x341 = jacobian_determinant * (-x139 * x326 + x340);
    const scalar_t x342 = -x139 * x338;
    const scalar_t x343 = 6 * lambda;
    const scalar_t x344 = x228 * x343;
    const scalar_t x345 = x20 * x231;
    const scalar_t x346 = x344 + x345;
    const scalar_t x347 = x247 + x346;
    const scalar_t x348 =
            -8 * adjugate[3] * adjugate[4] * lambda - 8 * adjugate[3] * adjugate[4] * mu;
    const scalar_t x349 = x20 * x228;
    const scalar_t x350 = x231 * x343;
    const scalar_t x351 = -x349 - x350;
    const scalar_t x352 = 3 * x251;
    const scalar_t x353 = 3 * x255;
    const scalar_t x354 = x352 + x353;
    const scalar_t x355 = -x293 - x294;
    const scalar_t x356 = x354 + x355;
    const scalar_t x357 = -x267 - x268;
    const scalar_t x358 = x266 + x357;
    const scalar_t x359 = x244 * x343;
    const scalar_t x360 = x20 * x244;
    const scalar_t x361 = x359 + x360;
    const scalar_t x362 = -x286 - x287 + x361;
    const scalar_t x363 = jacobian_determinant * (x139 * (x326 + x356 + x358 + x362) +
                                                  x322 * (-x327 - x347 - x348 - x351) + x342);
    const scalar_t x364 = x233 + x336;
    const scalar_t x365 = -x245 - x246;
    const scalar_t x366 = 2 * x235;
    const scalar_t x367 = lambda * x366;
    const scalar_t x368 = mu * x366;
    const scalar_t x369 = x367 + x368;
    const scalar_t x370 = x365 + x369;
    const scalar_t x371 = x364 + x370;
    const scalar_t x372 = x139 * x371;
    const scalar_t x373 = -x311 - x312;
    const scalar_t x374 = x261 + x310 + x373;
    const scalar_t x375 = x349 + x350;
    const scalar_t x376 = x239 + x375;
    const scalar_t x377 =
            -8 * adjugate[6] * adjugate[7] * lambda - 8 * adjugate[6] * adjugate[7] * mu;
    const scalar_t x378 = -x344 - x345;
    const scalar_t x379 = x300 + x303;
    const scalar_t x380 = x277 + x379;
    const scalar_t x381 = 3 * x257;
    const scalar_t x382 = 3 * x260;
    const scalar_t x383 = x381 + x382;
    const scalar_t x384 = -x278 - x279;
    const scalar_t x385 = x383 + x384;
    const scalar_t x386 = -x264 - x265;
    const scalar_t x387 = x269 + x386;
    const scalar_t x388 = x235 * x343;
    const scalar_t x389 = x20 * x235;
    const scalar_t x390 = x388 + x389;
    const scalar_t x391 = -x282 - x283 + x390;
    const scalar_t x392 = jacobian_determinant * (x139 * (x380 + x385 + x387 + x391) +
                                                  x322 * (-x374 - x376 - x377 - x378) + x372);
    const scalar_t x393 = -x139 * x371;
    const scalar_t x394 = x322 * (x371 + x374) + x393;
    const scalar_t x395 = jacobian_determinant * (-x139 * x380 + x394);
    const scalar_t x396 = x271 + x273;
    const scalar_t x397 = x306 + x396;
    const scalar_t x398 = x233 + x243;
    const scalar_t x399 = x333 + x369;
    const scalar_t x400 = x398 + x399;
    const scalar_t x401 = -x252 - x254;
    const scalar_t x402 = x257 - x258 - x259 + x260;
    const scalar_t x403 = x251 + x255 + x401 + x402;
    const scalar_t x404 = x139 * x400;
    const scalar_t x405 = x322 * (-x400 - x403) + x404;
    const scalar_t x406 = jacobian_determinant * (x139 * (-x300 - x397) + x405);
    const scalar_t x407 = -x139 * x400;
    const scalar_t x408 = x347 + x376;
    const scalar_t x409 = x354 + x383;
    const scalar_t x410 = -x271 - x273;
    const scalar_t x411 = x409 + x410;
    const scalar_t x412 = -x298 - x299;
    const scalar_t x413 = -x304 - x305;
    const scalar_t x414 = x270 + x361 + x390;
    const scalar_t x415 = jacobian_determinant *
                          (x139 * (-x411 - x412 - x413 - x414) + x322 * (x403 + x408) + x407);
    const scalar_t x416 = adjugate[3] * adjugate[8];
    const scalar_t x417 = x3 * x416;
    const scalar_t x418 = adjugate[5] * adjugate[6];
    const scalar_t x419 = x230 * x418;
    const scalar_t x420 = x417 + x419;
    const scalar_t x421 = adjugate[3] * adjugate[5];
    const scalar_t x422 = x234 * x421;
    const scalar_t x423 = x237 * x421;
    const scalar_t x424 = x422 + x423;
    const scalar_t x425 = x420 + x424;
    const scalar_t x426 = x230 * x416;
    const scalar_t x427 = x3 * x418;
    const scalar_t x428 = x426 + x427;
    const scalar_t x429 = adjugate[6] * adjugate[8];
    const scalar_t x430 = x234 * x429;
    const scalar_t x431 = x237 * x429;
    const scalar_t x432 = x430 + x431;
    const scalar_t x433 = x428 + x432;
    const scalar_t x434 = x425 + x433;
    const scalar_t x435 = x139 * x434;
    const scalar_t x436 = adjugate[8] * x44;
    const scalar_t x437 = adjugate[8] * x50;
    const scalar_t x438 = adjugate[2] * lambda;
    const scalar_t x439 = adjugate[6] * x438;
    const scalar_t x440 = adjugate[6] * x48;
    const scalar_t x441 = x436 + x437 + x439 + x440;
    const scalar_t x442 = adjugate[5] * x44;
    const scalar_t x443 = adjugate[5] * x50;
    const scalar_t x444 = adjugate[3] * x438;
    const scalar_t x445 = adjugate[3] * x48;
    const scalar_t x446 = x442 + x443 + x444 + x445;
    const scalar_t x447 = x441 + x446;
    const scalar_t x448 = x263 * x416;
    const scalar_t x449 = x418 * x95;
    const scalar_t x450 = x448 + x449;
    const scalar_t x451 = x416 * x95;
    const scalar_t x452 = x263 * x418;
    const scalar_t x453 = x451 + x452;
    const scalar_t x454 = x450 + x453;
    const scalar_t x455 = adjugate[2] * x44;
    const scalar_t x456 = 4 * x455;
    const scalar_t x457 = adjugate[0] * x48;
    const scalar_t x458 = 4 * x457;
    const scalar_t x459 = 6 * x437;
    const scalar_t x460 = 6 * x439;
    const scalar_t x461 = x459 + x460;
    const scalar_t x462 = 6 * x436;
    const scalar_t x463 = 6 * x440;
    const scalar_t x464 = x462 + x463;
    const scalar_t x465 = 12 * x429;
    const scalar_t x466 = lambda * x465;
    const scalar_t x467 = mu * x465;
    const scalar_t x468 = x466 + x467;
    const scalar_t x469 = 12 * x421;
    const scalar_t x470 = lambda * x469;
    const scalar_t x471 = mu * x469;
    const scalar_t x472 = x470 + x471;
    const scalar_t x473 = x456 + x458 + x461 + x464 + x468 + x472;
    const scalar_t x474 = 6 * x443;
    const scalar_t x475 = 6 * x444;
    const scalar_t x476 = x474 + x475;
    const scalar_t x477 = 6 * x442;
    const scalar_t x478 = 6 * x445;
    const scalar_t x479 = x477 + x478;
    const scalar_t x480 = x476 + x479;
    const scalar_t x481 =
            jacobian_determinant * (x135 * (-x434 - x447) + x139 * (x454 + x473 + x480) + x435);
    const scalar_t x482 = 3 * x443;
    const scalar_t x483 = 3 * x444;
    const scalar_t x484 = x482 + x483;
    const scalar_t x485 = 2 * x455;
    const scalar_t x486 = 2 * x457;
    const scalar_t x487 = x485 + x486;
    const scalar_t x488 = 3 * x437;
    const scalar_t x489 = 3 * x439;
    const scalar_t x490 = x488 + x489;
    const scalar_t x491 = x487 + x490;
    const scalar_t x492 = 2 * x437;
    const scalar_t x493 = 2 * x439;
    const scalar_t x494 = x492 + x493;
    const scalar_t x495 = 2 * x436;
    const scalar_t x496 = 2 * x440;
    const scalar_t x497 = x495 + x496;
    const scalar_t x498 = 2 * x443;
    const scalar_t x499 = 2 * x444;
    const scalar_t x500 = x498 + x499;
    const scalar_t x501 = 2 * x442;
    const scalar_t x502 = 2 * x445;
    const scalar_t x503 = x501 + x502;
    const scalar_t x504 = x500 + x503;
    const scalar_t x505 = x494 + x497 + x504;
    const scalar_t x506 = -x139 * x434;
    const scalar_t x507 = x322 * (x434 + x505) + x506;
    const scalar_t x508 = jacobian_determinant * (x42 * (-x484 - x491) + x507);
    const scalar_t x509 = x476 + x491;
    const scalar_t x510 = x441 + x500 - x501 - x502;
    const scalar_t x511 = -x417 - x419;
    const scalar_t x512 = x428 + x511;
    const scalar_t x513 = 2 * x429;
    const scalar_t x514 = lambda * x513;
    const scalar_t x515 = mu * x513;
    const scalar_t x516 = x514 + x515;
    const scalar_t x517 = -x422 - x423;
    const scalar_t x518 = x516 + x517;
    const scalar_t x519 = -x426 - x427;
    const scalar_t x520 = -x514 - x515;
    const scalar_t x521 = x425 + x519 + x520;
    const scalar_t x522 = x139 * x521;
    const scalar_t x523 = x322 * (x510 + x512 + x518) + x522;
    const scalar_t x524 = jacobian_determinant * (-x139 * x509 + x523);
    const scalar_t x525 = -x139 * x521;
    const scalar_t x526 = x343 * x416;
    const scalar_t x527 = x20 * x418;
    const scalar_t x528 = x526 + x527;
    const scalar_t x529 = x432 + x528;
    const scalar_t x530 =
            -8 * adjugate[3] * adjugate[5] * lambda - 8 * adjugate[3] * adjugate[5] * mu;
    const scalar_t x531 = x20 * x416;
    const scalar_t x532 = x343 * x418;
    const scalar_t x533 = -x531 - x532;
    const scalar_t x534 = 3 * x436;
    const scalar_t x535 = 3 * x440;
    const scalar_t x536 = x534 + x535;
    const scalar_t x537 = -x477 - x478;
    const scalar_t x538 = x536 + x537;
    const scalar_t x539 = -x451 - x452;
    const scalar_t x540 = x450 + x539;
    const scalar_t x541 = x343 * x429;
    const scalar_t x542 = x20 * x429;
    const scalar_t x543 = x541 + x542;
    const scalar_t x544 = -x470 - x471 + x543;
    const scalar_t x545 = jacobian_determinant * (x139 * (x509 + x538 + x540 + x544) +
                                                  x322 * (-x510 - x529 - x530 - x533) + x525);
    const scalar_t x546 = x420 + x519;
    const scalar_t x547 = -x430 - x431;
    const scalar_t x548 = 2 * x421;
    const scalar_t x549 = lambda * x548;
    const scalar_t x550 = mu * x548;
    const scalar_t x551 = x549 + x550;
    const scalar_t x552 = x547 + x551;
    const scalar_t x553 = x546 + x552;
    const scalar_t x554 = x139 * x553;
    const scalar_t x555 = -x495 - x496;
    const scalar_t x556 = x446 + x494 + x555;
    const scalar_t x557 = x531 + x532;
    const scalar_t x558 = x424 + x557;
    const scalar_t x559 =
            -8 * adjugate[6] * adjugate[8] * lambda - 8 * adjugate[6] * adjugate[8] * mu;
    const scalar_t x560 = -x526 - x527;
    const scalar_t x561 = x484 + x487;
    const scalar_t x562 = x461 + x561;
    const scalar_t x563 = 3 * x442;
    const scalar_t x564 = 3 * x445;
    const scalar_t x565 = x563 + x564;
    const scalar_t x566 = -x462 - x463;
    const scalar_t x567 = x565 + x566;
    const scalar_t x568 = -x448 - x449;
    const scalar_t x569 = x453 + x568;
    const scalar_t x570 = x343 * x421;
    const scalar_t x571 = x20 * x421;
    const scalar_t x572 = x570 + x571;
    const scalar_t x573 = -x466 - x467 + x572;
    const scalar_t x574 = jacobian_determinant * (x139 * (x562 + x567 + x569 + x573) +
                                                  x322 * (-x556 - x558 - x559 - x560) + x554);
    const scalar_t x575 = -x139 * x553;
    const scalar_t x576 = x322 * (x553 + x556) + x575;
    const scalar_t x577 = jacobian_determinant * (-x139 * x562 + x576);
    const scalar_t x578 = x455 + x457;
    const scalar_t x579 = x490 + x578;
    const scalar_t x580 = x420 + x428;
    const scalar_t x581 = x516 + x551;
    const scalar_t x582 = x580 + x581;
    const scalar_t x583 = -x437 - x439;
    const scalar_t x584 = x442 - x443 - x444 + x445;
    const scalar_t x585 = x436 + x440 + x583 + x584;
    const scalar_t x586 = x139 * x582;
    const scalar_t x587 = x322 * (-x582 - x585) + x586;
    const scalar_t x588 = jacobian_determinant * (x139 * (-x484 - x579) + x587);
    const scalar_t x589 = -x139 * x582;
    const scalar_t x590 = x529 + x558;
    const scalar_t x591 = x536 + x565;
    const scalar_t x592 = -x455 - x457;
    const scalar_t x593 = x591 + x592;
    const scalar_t x594 = -x482 - x483;
    const scalar_t x595 = -x488 - x489;
    const scalar_t x596 = x454 + x543 + x572;
    const scalar_t x597 = jacobian_determinant *
                          (x139 * (-x593 - x594 - x595 - x596) + x322 * (x585 + x590) + x589);
    const scalar_t x598 = (1.0 / 9.0) * x41;
    const scalar_t x599 = x171 * x598;
    const scalar_t x600 = x43 + x599;
    const scalar_t x601 = x171 * x42;
    const scalar_t x602 = x156 + x601;
    const scalar_t x603 = jacobian_determinant * (-x135 * x53 + x602);
    const scalar_t x604 = x138 + x67;
    const scalar_t x605 = -x63;
    const scalar_t x606 = -x68;
    const scalar_t x607 = x605 + x606 - x62 - x64;
    const scalar_t x608 = jacobian_determinant * (x139 * (-x604 - x607) + x154);
    const scalar_t x609 = -x82;
    const scalar_t x610 = -x83;
    const scalar_t x611 = x609 + x610 - x81 - x84;
    const scalar_t x612 = jacobian_determinant * (x139 * (-x202 - x611) + x204);
    const scalar_t x613 = x184 + x601;
    const scalar_t x614 = jacobian_determinant * (-x135 * x59 + x613);
    const scalar_t x615 = x139 * x171;
    const scalar_t x616 = jacobian_determinant * (x216 + x615);
    const scalar_t x617 = x191 + x206;
    const scalar_t x618 = jacobian_determinant * (x139 * (x220 + x617) + x214);
    const scalar_t x619 = x303 + x354;
    const scalar_t x620 = jacobian_determinant * (x324 + x42 * (-x383 - x619));
    const scalar_t x621 = x396 * x598;
    const scalar_t x622 = x250 + x621;
    const scalar_t x623 = jacobian_determinant * (-x135 * x262 + x622);
    const scalar_t x624 = x396 * x42;
    const scalar_t x625 = x342 + x624;
    const scalar_t x626 = jacobian_determinant * (-x322 * x327 + x625);
    const scalar_t x627 = jacobian_determinant * (x139 * (-x355 - x619) + x340);
    const scalar_t x628 = x303 + x383;
    const scalar_t x629 = jacobian_determinant * (x139 * (-x384 - x628) + x394);
    const scalar_t x630 = x372 + x624;
    const scalar_t x631 = jacobian_determinant * (-x322 * x374 + x630);
    const scalar_t x632 = x139 * x396;
    const scalar_t x633 = x407 + x632;
    const scalar_t x634 = jacobian_determinant * (x322 * x403 + x633);
    const scalar_t x635 = jacobian_determinant * (x139 * x411 + x405);
    const scalar_t x636 = x487 + x536;
    const scalar_t x637 = jacobian_determinant * (x42 * (-x565 - x636) + x507);
    const scalar_t x638 = x578 * x598;
    const scalar_t x639 = x435 + x638;
    const scalar_t x640 = jacobian_determinant * (-x135 * x447 + x639);
    const scalar_t x641 = x42 * x578;
    const scalar_t x642 = x525 + x641;
    const scalar_t x643 = jacobian_determinant * (-x322 * x510 + x642);
    const scalar_t x644 = jacobian_determinant * (x139 * (-x537 - x636) + x523);
    const scalar_t x645 = x487 + x565;
    const scalar_t x646 = jacobian_determinant * (x139 * (-x566 - x645) + x576);
    const scalar_t x647 = x554 + x641;
    const scalar_t x648 = jacobian_determinant * (-x322 * x556 + x647);
    const scalar_t x649 = x139 * x578;
    const scalar_t x650 = x589 + x649;
    const scalar_t x651 = jacobian_determinant * (x322 * x585 + x650);
    const scalar_t x652 = jacobian_determinant * (x139 * x593 + x587);
    const scalar_t x653 = -x47;
    const scalar_t x654 = -x49;
    const scalar_t x655 = -x52;
    const scalar_t x656 = -x45 + x653 + x654 + x655;
    const scalar_t x657 = x59 + x656;
    const scalar_t x658 = -x8;
    const scalar_t x659 = x15 + x658;
    const scalar_t x660 = -x5;
    const scalar_t x661 = x2 + x660;
    const scalar_t x662 = -x19 - x21;
    const scalar_t x663 = x39 + x659 + x661 + x662;
    const scalar_t x664 = x42 * x663;
    const scalar_t x665 = x599 + x664;
    const scalar_t x666 = -x112;
    const scalar_t x667 = -x118;
    const scalar_t x668 = -x115;
    const scalar_t x669 = -x111 + x666 + x667 + x668;
    const scalar_t x670 = -x42 * x663;
    const scalar_t x671 = -x124;
    const scalar_t x672 = x123 + x671;
    const scalar_t x673 = -x127;
    const scalar_t x674 = x126 + x673;
    const scalar_t x675 = -x132 - x133;
    const scalar_t x676 = jacobian_determinant *
                          (x135 * (x131 + x663 + x672 + x674 + x675) + x42 * (-x604 - x669) + x670);
    const scalar_t x677 = -x108;
    const scalar_t x678 = -x119;
    const scalar_t x679 = -x116;
    const scalar_t x680 = -x107 + x677 + x678 + x679;
    const scalar_t x681 = x208 + x658;
    const scalar_t x682 = x1 + x660;
    const scalar_t x683 = x211 + x662 + x681 + x682;
    const scalar_t x684 = -x683;
    const scalar_t x685 = x135 * x684 + x42 * x683;
    const scalar_t x686 = jacobian_determinant * (x139 * (-x193 - x680) + x685);
    const scalar_t x687 = x42 * x684;
    const scalar_t x688 = jacobian_determinant * (x615 + x687);
    const scalar_t x689 = jacobian_determinant * (x135 * x59 + x613);
    const scalar_t x690 = -x67;
    const scalar_t x691 = -x73;
    const scalar_t x692 = x690 + x691 - x76 - x78;
    const scalar_t x693 = -x176;
    const scalar_t x694 = -x10 + x14;
    const scalar_t x695 = x2 - x24;
    const scalar_t x696 = x135 * (x27 - x38 - x59 - x693 - x694 - x695) + x203;
    const scalar_t x697 = jacobian_determinant * (x139 * (x190 + x191 + x692 + x85) + x696);
    const scalar_t x698 = x256 - x314 - x315 + x319;
    const scalar_t x699 = x239 + x328;
    const scalar_t x700 = x243 + x337 + x699;
    const scalar_t x701 = x139 * x700;
    const scalar_t x702 = x322 * (x335 + x364 + x698) + x701;
    const scalar_t x703 = jacobian_determinant * (x139 * (-x295 - x619) + x702);
    const scalar_t x704 = -x139 * x700;
    const scalar_t x705 = x624 + x704;
    const scalar_t x706 = jacobian_determinant * (-x322 * x698 + x705);
    const scalar_t x707 = -x251 - x255;
    const scalar_t x708 = x261 + x401 + x707;
    const scalar_t x709 = x247 + x336 + x699;
    const scalar_t x710 = x139 * x709;
    const scalar_t x711 = x621 + x710;
    const scalar_t x712 = jacobian_determinant * (x135 * x708 + x711);
    const scalar_t x713 = -x381 - x382;
    const scalar_t x714 = x334 + x365;
    const scalar_t x715 = x398 + x714;
    const scalar_t x716 = -x308 - x309;
    const scalar_t x717 = x320 + x373 + x716;
    const scalar_t x718 = -x139 * x709;
    const scalar_t x719 = x322 * (-x715 - x717) + x718;
    const scalar_t x720 = jacobian_determinant * (x42 * (-x619 - x713) + x719);
    const scalar_t x721 = -x352 - x353;
    const scalar_t x722 = x383 + x721;
    const scalar_t x723 = x252 + x254 + x707;
    const scalar_t x724 = x402 + x723;
    const scalar_t x725 = -x367 - x368;
    const scalar_t x726 = x328 + x336;
    const scalar_t x727 = x399 + x726;
    const scalar_t x728 = x139 * x727;
    const scalar_t x729 = x322 * (x337 + x398 + x724 + x725) + x728;
    const scalar_t x730 = jacobian_determinant * (x139 * (-x396 - x722) + x729);
    const scalar_t x731 = -x139 * x727;
    const scalar_t x732 = x632 + x731;
    const scalar_t x733 = jacobian_determinant * (-x322 * x724 + x732);
    const scalar_t x734 = x261 + x313 + x716;
    const scalar_t x735 = x329 + x370;
    const scalar_t x736 = x139 * x735;
    const scalar_t x737 = x624 + x736;
    const scalar_t x738 = jacobian_determinant * (x322 * x734 + x737);
    const scalar_t x739 = -x301 - x302;
    const scalar_t x740 = x280 + x383;
    const scalar_t x741 = x739 + x740;
    const scalar_t x742 = -x139 * x735;
    const scalar_t x743 = x322 * (-x247 - x364 - x725 - x734) + x742;
    const scalar_t x744 = jacobian_determinant * (x139 * x741 + x743);
    const scalar_t x745 = x441 - x498 - x499 + x503;
    const scalar_t x746 = x424 + x511;
    const scalar_t x747 = x428 + x520 + x746;
    const scalar_t x748 = x139 * x747;
    const scalar_t x749 = x322 * (x518 + x546 + x745) + x748;
    const scalar_t x750 = jacobian_determinant * (x139 * (-x479 - x636) + x749);
    const scalar_t x751 = -x139 * x747;
    const scalar_t x752 = x641 + x751;
    const scalar_t x753 = jacobian_determinant * (-x322 * x745 + x752);
    const scalar_t x754 = -x436 - x440;
    const scalar_t x755 = x446 + x583 + x754;
    const scalar_t x756 = x432 + x519 + x746;
    const scalar_t x757 = x139 * x756;
    const scalar_t x758 = x638 + x757;
    const scalar_t x759 = jacobian_determinant * (x135 * x755 + x758);
    const scalar_t x760 = -x563 - x564;
    const scalar_t x761 = x517 + x547;
    const scalar_t x762 = x580 + x761;
    const scalar_t x763 = -x492 - x493;
    const scalar_t x764 = x504 + x555 + x763;
    const scalar_t x765 = -x139 * x756;
    const scalar_t x766 = x322 * (-x762 - x764) + x765;
    const scalar_t x767 = jacobian_determinant * (x42 * (-x636 - x760) + x766);
    const scalar_t x768 = -x534 - x535;
    const scalar_t x769 = x565 + x768;
    const scalar_t x770 = x437 + x439 + x754;
    const scalar_t x771 = x584 + x770;
    const scalar_t x772 = -x549 - x550;
    const scalar_t x773 = x511 + x519;
    const scalar_t x774 = x581 + x773;
    const scalar_t x775 = x139 * x774;
    const scalar_t x776 = x322 * (x520 + x580 + x771 + x772) + x775;
    const scalar_t x777 = jacobian_determinant * (x139 * (-x578 - x769) + x776);
    const scalar_t x778 = -x139 * x774;
    const scalar_t x779 = x649 + x778;
    const scalar_t x780 = jacobian_determinant * (-x322 * x771 + x779);
    const scalar_t x781 = x446 + x497 + x763;
    const scalar_t x782 = x512 + x552;
    const scalar_t x783 = x139 * x782;
    const scalar_t x784 = x641 + x783;
    const scalar_t x785 = jacobian_determinant * (x322 * x781 + x784);
    const scalar_t x786 = -x485 - x486;
    const scalar_t x787 = x464 + x565;
    const scalar_t x788 = x786 + x787;
    const scalar_t x789 = -x139 * x782;
    const scalar_t x790 = x322 * (-x432 - x546 - x772 - x781) + x789;
    const scalar_t x791 = jacobian_determinant * (x139 * x788 + x790);
    const scalar_t x792 = -x55;
    const scalar_t x793 = -x56;
    const scalar_t x794 = -x58;
    const scalar_t x795 = -x54 + x792 + x793 + x794;
    const scalar_t x796 = x67 + x86;
    const scalar_t x797 = -x96 + x99;
    const scalar_t x798 = -x101 + x104;
    const scalar_t x799 = x797 + x798 - x88 - x89 + x94;
    const scalar_t x800 = x166 + x195 + x222;
    const scalar_t x801 = x168 + x197 + x224;
    const scalar_t x802 =
            jacobian_determinant * (x139 * (x172 + x199 + x226 + x800 + x801) + x61 * x683 + x687);
    const scalar_t x803 = x171 + x669;
    const scalar_t x804 = jacobian_determinant * (x139 * (-x164 - x803) + x685);
    const scalar_t x805 = jacobian_determinant * (x139 * (-x669 - x796) + x696);
    const scalar_t x806 = x146 + x185;
    const scalar_t x807 = x141 + x187;
    const scalar_t x808 =
            jacobian_determinant * (x135 * (x140 + x143 + 4 * x34 + 8 * x36 + x59 + x806 + x807) +
                                    x184 + x42 * (x200 + x803));
    const scalar_t x809 = x375 + x378;
    const scalar_t x810 = -x290 - x291;
    const scalar_t x811 = x307 + x810;
    const scalar_t x812 = x295 + x354;
    const scalar_t x813 = jacobian_determinant * (x139 * (x362 + x387 + x811 + x812) +
                                                  x322 * (-x247 - x348 - x698 - x809) + x704);
    const scalar_t x814 = jacobian_determinant * (-x139 * x811 + x702);
    const scalar_t x815 = jacobian_determinant * (x42 * (-x307 - x412) + x719);
    const scalar_t x816 = x357 + x386;
    const scalar_t x817 = x355 + x810;
    const scalar_t x818 =
            jacobian_determinant * (x135 * (x708 + x715) + x139 * (x289 + x816 + x817) + x710);
    const scalar_t x819 = x397 + x412;
    const scalar_t x820 = -x388 - x389;
    const scalar_t x821 = -x359 - x360;
    const scalar_t x822 = x270 + x820 + x821;
    const scalar_t x823 = jacobian_determinant * (x139 * (x722 + x819 + x822) +
                                                  x322 * (-x346 - x375 - x714 - x724) + x731);
    const scalar_t x824 = jacobian_determinant * (-x139 * x819 + x729);
    const scalar_t x825 = jacobian_determinant * (x139 * (-x277 - x303 - x412) + x743);
    const scalar_t x826 = 8 * x244;
    const scalar_t x827 = lambda * x826 + mu * x826 + x334;
    const scalar_t x828 = -x275 - x276;
    const scalar_t x829 = x284 + x828;
    const scalar_t x830 = jacobian_determinant * (x139 * (-x300 - x387 - x741 - x820 - x829) +
                                                  x322 * (x734 + x809 + x827) + x736);
    const scalar_t x831 = x557 + x560;
    const scalar_t x832 = -x474 - x475;
    const scalar_t x833 = x491 + x832;
    const scalar_t x834 = x479 + x536;
    const scalar_t x835 = jacobian_determinant * (x139 * (x544 + x569 + x833 + x834) +
                                                  x322 * (-x432 - x530 - x745 - x831) + x751);
    const scalar_t x836 = jacobian_determinant * (-x139 * x833 + x749);
    const scalar_t x837 = jacobian_determinant * (x42 * (-x491 - x594) + x766);
    const scalar_t x838 = x539 + x568;
    const scalar_t x839 = x537 + x832;
    const scalar_t x840 =
            jacobian_determinant * (x135 * (x755 + x762) + x139 * (x473 + x838 + x839) + x757);
    const scalar_t x841 = x579 + x594;
    const scalar_t x842 = -x570 - x571;
    const scalar_t x843 = -x541 - x542;
    const scalar_t x844 = x454 + x842 + x843;
    const scalar_t x845 = jacobian_determinant * (x139 * (x769 + x841 + x844) +
                                                  x322 * (-x528 - x557 - x761 - x771) + x778);
    const scalar_t x846 = jacobian_determinant * (-x139 * x841 + x776);
    const scalar_t x847 = jacobian_determinant * (x139 * (-x461 - x487 - x594) + x790);
    const scalar_t x848 = 8 * x429;
    const scalar_t x849 = lambda * x848 + mu * x848 + x517;
    const scalar_t x850 = -x459 - x460;
    const scalar_t x851 = x468 + x850;
    const scalar_t x852 = jacobian_determinant * (x139 * (-x484 - x569 - x788 - x842 - x851) +
                                                  x322 * (x781 + x831 + x849) + x783);
    const scalar_t x853 = x70 + x80;
    const scalar_t x854 = -x123;
    const scalar_t x855 = x124 + x854;
    const scalar_t x856 = -x126;
    const scalar_t x857 = x127 + x856;
    const scalar_t x858 = -x129 - x130;
    const scalar_t x859 = jacobian_determinant *
                          (x135 * (x134 + x663 + x855 + x857 + x858) + x42 * (-x202 - x680) + x670);
    const scalar_t x860 = x135 * (-x152 - x53) + x153;
    const scalar_t x861 = jacobian_determinant * (x139 * (-x680 - x853) + x860);
    const scalar_t x862 = x160 + x178;
    const scalar_t x863 =
            jacobian_determinant * (x135 * (x157 + x175 + x182 + 4 * x27 + 8 * x29 + x53 + x862) +
                                    x156 + x42 * (x173 + x680));
    const scalar_t x864 = x239 + x351;
    const scalar_t x865 = x379 + x828;
    const scalar_t x866 = jacobian_determinant * (x139 * (x358 + x391 + x740 + x865) +
                                                  x322 * (-x346 - x377 - x734 - x864) + x736);
    const scalar_t x867 = x322 * (x734 + x735) + x742;
    const scalar_t x868 = jacobian_determinant * (-x139 * x865 + x867);
    const scalar_t x869 = x300 + x396 + x413;
    const scalar_t x870 = x322 * (-x724 - x727) + x728;
    const scalar_t x871 = jacobian_determinant * (-x139 * x869 + x870);
    const scalar_t x872 = x354 + x713;
    const scalar_t x873 = jacobian_determinant *
                          (x139 * (x822 + x869 + x872) + x322 * (x247 + x378 + x724 + x864) + x731);
    const scalar_t x874 = x272 + x274 + x288 + x384 + x829;
    const scalar_t x875 =
            jacobian_determinant * (x135 * (-x708 - x709) + x139 * (x296 + x816 + x874) + x710);
    const scalar_t x876 = x322 * (x709 + x717) + x718;
    const scalar_t x877 = jacobian_determinant * (x42 * (-x379 - x413) + x876);
    const scalar_t x878 = x322 * (-x698 - x700) + x701;
    const scalar_t x879 = jacobian_determinant * (x139 * (-x292 - x303 - x413) + x878);
    const scalar_t x880 = x346 + x351;
    const scalar_t x881 = 8 * x235;
    const scalar_t x882 = lambda * x881 + mu * x881 + x365;
    const scalar_t x883 = x739 + x812;
    const scalar_t x884 = x288 + x821;
    const scalar_t x885 = jacobian_determinant * (x139 * (-x306 - x358 - x810 - x883 - x884) +
                                                  x322 * (x698 + x880 + x882) + x704);
    const scalar_t x886 = x424 + x533;
    const scalar_t x887 = x561 + x850;
    const scalar_t x888 = jacobian_determinant * (x139 * (x540 + x573 + x787 + x887) +
                                                  x322 * (-x528 - x559 - x781 - x886) + x783);
    const scalar_t x889 = x322 * (x781 + x782) + x789;
    const scalar_t x890 = jacobian_determinant * (-x139 * x887 + x889);
    const scalar_t x891 = x484 + x578 + x595;
    const scalar_t x892 = x322 * (-x771 - x774) + x775;
    const scalar_t x893 = jacobian_determinant * (-x139 * x891 + x892);
    const scalar_t x894 = x536 + x760;
    const scalar_t x895 = jacobian_determinant *
                          (x139 * (x844 + x891 + x894) + x322 * (x432 + x560 + x771 + x886) + x778);
    const scalar_t x896 = x456 + x458 + x472 + x566 + x851;
    const scalar_t x897 =
            jacobian_determinant * (x135 * (-x755 - x756) + x139 * (x480 + x838 + x896) + x757);
    const scalar_t x898 = x322 * (x756 + x764) + x765;
    const scalar_t x899 = jacobian_determinant * (x42 * (-x561 - x595) + x898);
    const scalar_t x900 = x322 * (-x745 - x747) + x748;
    const scalar_t x901 = jacobian_determinant * (x139 * (-x476 - x487 - x595) + x900);
    const scalar_t x902 = x528 + x533;
    const scalar_t x903 = 8 * x421;
    const scalar_t x904 = lambda * x903 + mu * x903 + x547;
    const scalar_t x905 = x786 + x834;
    const scalar_t x906 = x472 + x843;
    const scalar_t x907 = jacobian_determinant * (x139 * (-x490 - x540 - x832 - x905 - x906) +
                                                  x322 * (x745 + x902 + x904) + x751);
    const scalar_t x908 = jacobian_determinant * (x135 * x53 + x602);
    const scalar_t x909 =
            jacobian_determinant * (x139 * (x164 + x62 + x63 + x64 + x68 + x692) + x860);
    const scalar_t x910 = jacobian_determinant * (x139 * (-x280 - x628) + x867);
    const scalar_t x911 = jacobian_determinant * (-x322 * x734 + x737);
    const scalar_t x912 = jacobian_determinant * (x322 * x724 + x732);
    const scalar_t x913 = jacobian_determinant * (x139 * (-x396 - x872) + x870);
    const scalar_t x914 = jacobian_determinant * (x42 * (-x628 - x721) + x876);
    const scalar_t x915 = jacobian_determinant * (-x135 * x708 + x711);
    const scalar_t x916 = jacobian_determinant * (x322 * x698 + x705);
    const scalar_t x917 = jacobian_determinant * (x139 * x883 + x878);
    const scalar_t x918 = jacobian_determinant * (x139 * (-x464 - x645) + x889);
    const scalar_t x919 = jacobian_determinant * (-x322 * x781 + x784);
    const scalar_t x920 = jacobian_determinant * (x322 * x771 + x779);
    const scalar_t x921 = jacobian_determinant * (x139 * (-x578 - x894) + x892);
    const scalar_t x922 = jacobian_determinant * (x42 * (-x645 - x768) + x898);
    const scalar_t x923 = jacobian_determinant * (-x135 * x755 + x758);
    const scalar_t x924 = jacobian_determinant * (x322 * x745 + x752);
    const scalar_t x925 = jacobian_determinant * (x139 * x905 + x900);
    const scalar_t x926 = x671 + x854;
    const scalar_t x927 = x673 + x856;
    const scalar_t x928 = jacobian_determinant *
                          (x122 + x135 * (x40 + x675 + x858 + x926 + x927) + x42 * (x617 + x692));
    const scalar_t x929 = -x257 + x258 + x259 - x260 + x723;
    const scalar_t x930 = x322 * (-x400 - x929) + x404;
    const scalar_t x931 = jacobian_determinant * (x139 * (-x396 - x409) + x930);
    const scalar_t x932 = jacobian_determinant * (-x322 * x403 + x633);
    const scalar_t x933 = jacobian_determinant * (x322 * x374 + x630);
    const scalar_t x934 = x322 * (-x248 - x328 - x374 - x725) + x393;
    const scalar_t x935 = jacobian_determinant * (x139 * (-x280 - x303 - x713) + x934);
    const scalar_t x936 = x322 * (-x327 - x338) + x339;
    const scalar_t x937 = jacobian_determinant * (x139 * (-x295 - x303 - x721) + x936);
    const scalar_t x938 = jacobian_determinant * (x322 * x327 + x625);
    const scalar_t x939 = jacobian_determinant * (x135 * x262 + x622);
    const scalar_t x940 = x714 + x726;
    const scalar_t x941 = x322 * (-x321 - x940) + x323;
    const scalar_t x942 = jacobian_determinant * (x42 * (x409 + x739) + x941);
    const scalar_t x943 = -x442 + x443 + x444 - x445 + x770;
    const scalar_t x944 = x322 * (-x582 - x943) + x586;
    const scalar_t x945 = jacobian_determinant * (x139 * (-x578 - x591) + x944);
    const scalar_t x946 = jacobian_determinant * (-x322 * x585 + x650);
    const scalar_t x947 = jacobian_determinant * (x322 * x556 + x647);
    const scalar_t x948 = x322 * (-x433 - x511 - x556 - x772) + x575;
    const scalar_t x949 = jacobian_determinant * (x139 * (-x464 - x487 - x760) + x948);
    const scalar_t x950 = x322 * (-x510 - x521) + x522;
    const scalar_t x951 = jacobian_determinant * (x139 * (-x479 - x487 - x768) + x950);
    const scalar_t x952 = jacobian_determinant * (x322 * x510 + x642);
    const scalar_t x953 = jacobian_determinant * (x135 * x447 + x639);
    const scalar_t x954 = x761 + x773;
    const scalar_t x955 = x322 * (-x505 - x954) + x506;
    const scalar_t x956 = jacobian_determinant * (x42 * (x591 + x786) + x955);
    const scalar_t x957 = x300 + x306;
    const scalar_t x958 = x410 + x957;
    const scalar_t x959 = jacobian_determinant *
                          (x139 * (-x414 - x713 - x721 - x958) + x322 * (x408 + x929) + x407);
    const scalar_t x960 = jacobian_determinant * (x139 * x958 + x930);
    const scalar_t x961 = x277 + x300 + x739;
    const scalar_t x962 = jacobian_determinant * (x139 * x961 + x934);
    const scalar_t x963 = jacobian_determinant * (x139 * (-x284 - x358 - x385 - x820 - x961) +
                                                  x322 * (x374 + x827 + x880) + x372);
    const scalar_t x964 = x292 + x306 + x739;
    const scalar_t x965 = jacobian_determinant * (x139 * (-x356 - x387 - x884 - x964) +
                                                  x322 * (x327 + x809 + x882) + x342);
    const scalar_t x966 = jacobian_determinant * (x139 * x964 + x936);
    const scalar_t x967 = jacobian_determinant * (x42 * (x739 + x957) + x941);
    const scalar_t x968 =
            jacobian_determinant * (x135 * (x262 + x940) + x139 * (x270 + x817 + x874) + x250);
    const scalar_t x969 = x484 + x490;
    const scalar_t x970 = x592 + x969;
    const scalar_t x971 = jacobian_determinant *
                          (x139 * (-x596 - x760 - x768 - x970) + x322 * (x590 + x943) + x589);
    const scalar_t x972 = jacobian_determinant * (x139 * x970 + x944);
    const scalar_t x973 = x461 + x484 + x786;
    const scalar_t x974 = jacobian_determinant * (x139 * x973 + x948);
    const scalar_t x975 = jacobian_determinant * (x139 * (-x468 - x540 - x567 - x842 - x973) +
                                                  x322 * (x556 + x849 + x902) + x554);
    const scalar_t x976 = x476 + x490 + x786;
    const scalar_t x977 = jacobian_determinant * (x139 * (-x538 - x569 - x906 - x976) +
                                                  x322 * (x510 + x831 + x904) + x525);
    const scalar_t x978 = jacobian_determinant * (x139 * x976 + x950);
    const scalar_t x979 = jacobian_determinant * (x42 * (x786 + x969) + x955);
    const scalar_t x980 =
            jacobian_determinant * (x135 * (x447 + x954) + x139 * (x454 + x839 + x896) + x435);
    const scalar_t x981 = x17 * x3;
    const scalar_t x982 = x230 * x4;
    const scalar_t x983 = x20 * x4;
    const scalar_t x984 = x981 + x982 + x983;
    const scalar_t x985 = lambda * x23;
    const scalar_t x986 = 2 * x985;
    const scalar_t x987 = x157 + x986;
    const scalar_t x988 = lambda * x0;
    const scalar_t x989 = 2 * x988;
    const scalar_t x990 = x185 + x989;
    const scalar_t x991 = x210 + x987 + x990;
    const scalar_t x992 = x16 + x984 + x991;
    const scalar_t x993 = x42 * x992;
    const scalar_t x994 = adjugate[7] * x253;
    const scalar_t x995 = x126 + x49 + x51 + x994;
    const scalar_t x996 = adjugate[4] * x253;
    const scalar_t x997 = x127 + x56 + x57 + x996;
    const scalar_t x998 = x995 + x997;
    const scalar_t x999 = 6 * x996;
    const scalar_t x1000 = 12 * x55;
    const scalar_t x1001 = x1000 + x112 + x68 + x999;
    const scalar_t x1002 = 6 * x994;
    const scalar_t x1003 = 12 * x47;
    const scalar_t x1004 = lambda * x65;
    const scalar_t x1005 = 2 * x1004;
    const scalar_t x1006 = 4 * x66;
    const scalar_t x1007 = x1005 + x1006;
    const scalar_t x1008 = x170 + x83;
    const scalar_t x1009 = x1007 + x1008 + x109;
    const scalar_t x1010 = x1002 + x1003 + x1009;
    const scalar_t x1011 = 6 * x985;
    const scalar_t x1012 = 6 * x988;
    const scalar_t x1013 = 12 * x24;
    const scalar_t x1014 = 12 * x1;
    const scalar_t x1015 = x165 + x194;
    const scalar_t x1016 = x1011 + x1012 + x1013 + x1014 + x1015;
    const scalar_t x1017 = x1010 + x1016;
    const scalar_t x1018 = x17 * x95;
    const scalar_t x1019 = x263 * x4;
    const scalar_t x1020 = x100 + x1018 + x1019 + x224;
    const scalar_t x1021 = 3 * x994;
    const scalar_t x1022 = x170 + x82;
    const scalar_t x1023 = x1007 + x73;
    const scalar_t x1024 = x1021 + x1022 + x1023;
    const scalar_t x1025 = 3 * x57;
    const scalar_t x1026 = 3 * x51;
    const scalar_t x1027 = x1025 + x1026;
    const scalar_t x1028 = 3 * x996;
    const scalar_t x1029 = x1028 + x63;
    const scalar_t x1030 = x1027 + x1029 + x117;
    const scalar_t x1031 = -x42 * x992;
    const scalar_t x1032 = x52 + x58;
    const scalar_t x1033 = 2 * x994;
    const scalar_t x1034 = 4 * x47;
    const scalar_t x1035 = x1033 + x1034;
    const scalar_t x1036 = 2 * x996;
    const scalar_t x1037 = 4 * x55;
    const scalar_t x1038 = x1036 + x1037;
    const scalar_t x1039 =
            jacobian_determinant *
            (x1031 + x135 * (x1032 + x1035 + x1038 + x125 + x992) + x42 * (-x1024 - x1030));
    const scalar_t x1040 = x1026 + x116;
    const scalar_t x1041 = x1024 + x1040;
    const scalar_t x1042 = x36 + x988;
    const scalar_t x1043 = x693 - x986;
    const scalar_t x1044 = -x36;
    const scalar_t x1045 = x1044 + x149 + x177 + x987 - x988;
    const scalar_t x1046 = x1045 * x42;
    const scalar_t x1047 = x1046 + x135 * (x1042 + x1043 + x142 + x159 + x995);
    const scalar_t x1048 = jacobian_determinant * (x1047 + x139 * (-x1001 - x1041));
    const scalar_t x1049 = -x1045 * x42;
    const scalar_t x1050 = x1021 + x82;
    const scalar_t x1051 = x1040 + x1050;
    const scalar_t x1052 = -x194;
    const scalar_t x1053 = x1052 + 3 * x36;
    const scalar_t x1054 = x1004 + x67 + x72 + x77;
    const scalar_t x1055 = -x1011 - x1013 + x1054;
    const scalar_t x1056 = x103 + x1053 + x1055 + x167 + 3 * x988;
    const scalar_t x1057 =
            jacobian_determinant *
            (x1049 + x135 * (4 * lambda * x23 + 8 * mu * x23 - x145 - x162 - x990 - x995) +
             x42 * (x1051 + x1056));
    const scalar_t x1058 = x29 + x985;
    const scalar_t x1059 = x186 - x989;
    const scalar_t x1060 = x1058 + x1059 + x151 + x179;
    const scalar_t x1061 = x1060 * x42;
    const scalar_t x1062 = x1025 + x1029 + x115;
    const scalar_t x1063 = x1054 + x1062;
    const scalar_t x1064 = -x165;
    const scalar_t x1065 = x1064 + 3 * x29;
    const scalar_t x1066 = -x1012 - x1014;
    const scalar_t x1067 = x102 + x1065 + x1066 + x196 + 3 * x985;
    const scalar_t x1068 =
            jacobian_determinant *
            (x1061 + x135 * (4 * lambda * x0 + 8 * mu * x0 - x176 - x181 - x189 - x987 - x997) +
             x42 * (x1063 + x1067));
    const scalar_t x1069 = x1002 + x1003 + x1062;
    const scalar_t x1070 = -x1060 * x42;
    const scalar_t x1071 = x1070 + x135 * (x1060 + x997);
    const scalar_t x1072 = jacobian_determinant * (x1071 + x139 * (-x1009 - x1069));
    const scalar_t x1073 = x1030 + x1050;
    const scalar_t x1074 = x1042 + x1058 + x2 + x25;
    const scalar_t x1075 = x1074 + x209 + x984;
    const scalar_t x1076 = -x1075;
    const scalar_t x1077 = x1075 * x42 + x1076 * x135;
    const scalar_t x1078 = jacobian_determinant * (x1077 + x139 * (-x1054 - x1073));
    const scalar_t x1079 = x1076 * x42;
    const scalar_t x1080 = -x77;
    const scalar_t x1081 = -x1004 + x1080 + x218 + x690;
    const scalar_t x1082 = 18 * lambda;
    const scalar_t x1083 = 36 * mu;
    const scalar_t x1084 = x1082 * x4 + x1083 * x4 + x88;
    const scalar_t x1085 =
            jacobian_determinant * (x1075 * x61 + x1079 + x139 * (-x1016 - x1081 - x1084 - x223));
    const scalar_t x1086 = adjugate[4] * adjugate[8];
    const scalar_t x1087 = x1086 * x3;
    const scalar_t x1088 = adjugate[5] * adjugate[7];
    const scalar_t x1089 = x1088 * x230;
    const scalar_t x1090 = x1087 + x1089;
    const scalar_t x1091 = adjugate[4] * adjugate[5];
    const scalar_t x1092 = x1091 * x234;
    const scalar_t x1093 = x1091 * x237;
    const scalar_t x1094 = x1092 + x1093;
    const scalar_t x1095 = x1090 + x1094;
    const scalar_t x1096 = x1086 * x230;
    const scalar_t x1097 = x1088 * x3;
    const scalar_t x1098 = x1096 + x1097;
    const scalar_t x1099 = adjugate[7] * adjugate[8];
    const scalar_t x1100 = x1099 * x234;
    const scalar_t x1101 = x1099 * x237;
    const scalar_t x1102 = x1100 + x1101;
    const scalar_t x1103 = x1098 + x1102;
    const scalar_t x1104 = x1095 + x1103;
    const scalar_t x1105 = x1104 * x139;
    const scalar_t x1106 = adjugate[8] * x253;
    const scalar_t x1107 = adjugate[8] * x46;
    const scalar_t x1108 = adjugate[7] * x438;
    const scalar_t x1109 = adjugate[7] * x48;
    const scalar_t x1110 = x1106 + x1107 + x1108 + x1109;
    const scalar_t x1111 = adjugate[5] * x253;
    const scalar_t x1112 = adjugate[5] * x46;
    const scalar_t x1113 = adjugate[4] * x438;
    const scalar_t x1114 = adjugate[4] * x48;
    const scalar_t x1115 = x1111 + x1112 + x1113 + x1114;
    const scalar_t x1116 = x1110 + x1115;
    const scalar_t x1117 = x1086 * x263;
    const scalar_t x1118 = x1088 * x95;
    const scalar_t x1119 = x1117 + x1118;
    const scalar_t x1120 = x1086 * x95;
    const scalar_t x1121 = x1088 * x263;
    const scalar_t x1122 = x1120 + x1121;
    const scalar_t x1123 = x1119 + x1122;
    const scalar_t x1124 = adjugate[2] * x253;
    const scalar_t x1125 = 4 * x1124;
    const scalar_t x1126 = adjugate[2] * x46;
    const scalar_t x1127 = 4 * x1126;
    const scalar_t x1128 = 6 * x1107;
    const scalar_t x1129 = 6 * x1108;
    const scalar_t x1130 = x1128 + x1129;
    const scalar_t x1131 = 6 * x1106;
    const scalar_t x1132 = 6 * x1109;
    const scalar_t x1133 = x1131 + x1132;
    const scalar_t x1134 = 12 * x1099;
    const scalar_t x1135 = lambda * x1134;
    const scalar_t x1136 = mu * x1134;
    const scalar_t x1137 = x1135 + x1136;
    const scalar_t x1138 = 12 * x1091;
    const scalar_t x1139 = lambda * x1138;
    const scalar_t x1140 = mu * x1138;
    const scalar_t x1141 = x1139 + x1140;
    const scalar_t x1142 = x1125 + x1127 + x1130 + x1133 + x1137 + x1141;
    const scalar_t x1143 = 6 * x1112;
    const scalar_t x1144 = 6 * x1113;
    const scalar_t x1145 = x1143 + x1144;
    const scalar_t x1146 = 6 * x1111;
    const scalar_t x1147 = 6 * x1114;
    const scalar_t x1148 = x1146 + x1147;
    const scalar_t x1149 = x1145 + x1148;
    const scalar_t x1150 = jacobian_determinant *
                           (x1105 + x135 * (-x1104 - x1116) + x139 * (x1123 + x1142 + x1149));
    const scalar_t x1151 = 3 * x1112;
    const scalar_t x1152 = 3 * x1113;
    const scalar_t x1153 = x1151 + x1152;
    const scalar_t x1154 = 2 * x1124;
    const scalar_t x1155 = 2 * x1126;
    const scalar_t x1156 = x1154 + x1155;
    const scalar_t x1157 = 3 * x1107;
    const scalar_t x1158 = 3 * x1108;
    const scalar_t x1159 = x1157 + x1158;
    const scalar_t x1160 = x1156 + x1159;
    const scalar_t x1161 = 2 * x1107;
    const scalar_t x1162 = 2 * x1108;
    const scalar_t x1163 = x1161 + x1162;
    const scalar_t x1164 = 2 * x1106;
    const scalar_t x1165 = 2 * x1109;
    const scalar_t x1166 = x1164 + x1165;
    const scalar_t x1167 = 2 * x1112;
    const scalar_t x1168 = 2 * x1113;
    const scalar_t x1169 = x1167 + x1168;
    const scalar_t x1170 = 2 * x1111;
    const scalar_t x1171 = 2 * x1114;
    const scalar_t x1172 = x1170 + x1171;
    const scalar_t x1173 = x1169 + x1172;
    const scalar_t x1174 = x1163 + x1166 + x1173;
    const scalar_t x1175 = -x1104 * x139;
    const scalar_t x1176 = x1175 + x322 * (x1104 + x1174);
    const scalar_t x1177 = jacobian_determinant * (x1176 + x42 * (-x1153 - x1160));
    const scalar_t x1178 = x1145 + x1160;
    const scalar_t x1179 = x1110 + x1169 - x1170 - x1171;
    const scalar_t x1180 = -x1087 - x1089;
    const scalar_t x1181 = x1098 + x1180;
    const scalar_t x1182 = 2 * x1099;
    const scalar_t x1183 = lambda * x1182;
    const scalar_t x1184 = mu * x1182;
    const scalar_t x1185 = x1183 + x1184;
    const scalar_t x1186 = -x1092 - x1093;
    const scalar_t x1187 = x1185 + x1186;
    const scalar_t x1188 = -x1096 - x1097;
    const scalar_t x1189 = -x1183 - x1184;
    const scalar_t x1190 = x1095 + x1188 + x1189;
    const scalar_t x1191 = x1190 * x139;
    const scalar_t x1192 = x1191 + x322 * (x1179 + x1181 + x1187);
    const scalar_t x1193 = jacobian_determinant * (-x1178 * x139 + x1192);
    const scalar_t x1194 = -x1190 * x139;
    const scalar_t x1195 = x1086 * x343;
    const scalar_t x1196 = x1088 * x20;
    const scalar_t x1197 = x1195 + x1196;
    const scalar_t x1198 = x1102 + x1197;
    const scalar_t x1199 =
            -8 * adjugate[4] * adjugate[5] * lambda - 8 * adjugate[4] * adjugate[5] * mu;
    const scalar_t x1200 = x1086 * x20;
    const scalar_t x1201 = x1088 * x343;
    const scalar_t x1202 = -x1200 - x1201;
    const scalar_t x1203 = 3 * x1106;
    const scalar_t x1204 = 3 * x1109;
    const scalar_t x1205 = x1203 + x1204;
    const scalar_t x1206 = -x1146 - x1147;
    const scalar_t x1207 = x1205 + x1206;
    const scalar_t x1208 = -x1120 - x1121;
    const scalar_t x1209 = x1119 + x1208;
    const scalar_t x1210 = x1099 * x343;
    const scalar_t x1211 = x1099 * x20;
    const scalar_t x1212 = x1210 + x1211;
    const scalar_t x1213 = -x1139 - x1140 + x1212;
    const scalar_t x1214 = jacobian_determinant * (x1194 + x139 * (x1178 + x1207 + x1209 + x1213) +
                                                   x322 * (-x1179 - x1198 - x1199 - x1202));
    const scalar_t x1215 = x1090 + x1188;
    const scalar_t x1216 = -x1100 - x1101;
    const scalar_t x1217 = 2 * x1091;
    const scalar_t x1218 = lambda * x1217;
    const scalar_t x1219 = mu * x1217;
    const scalar_t x1220 = x1218 + x1219;
    const scalar_t x1221 = x1216 + x1220;
    const scalar_t x1222 = x1215 + x1221;
    const scalar_t x1223 = x1222 * x139;
    const scalar_t x1224 = -x1164 - x1165;
    const scalar_t x1225 = x1115 + x1163 + x1224;
    const scalar_t x1226 = x1200 + x1201;
    const scalar_t x1227 = x1094 + x1226;
    const scalar_t x1228 =
            -8 * adjugate[7] * adjugate[8] * lambda - 8 * adjugate[7] * adjugate[8] * mu;
    const scalar_t x1229 = -x1195 - x1196;
    const scalar_t x1230 = x1153 + x1156;
    const scalar_t x1231 = x1130 + x1230;
    const scalar_t x1232 = 3 * x1111;
    const scalar_t x1233 = 3 * x1114;
    const scalar_t x1234 = x1232 + x1233;
    const scalar_t x1235 = -x1131 - x1132;
    const scalar_t x1236 = x1234 + x1235;
    const scalar_t x1237 = -x1117 - x1118;
    const scalar_t x1238 = x1122 + x1237;
    const scalar_t x1239 = x1091 * x343;
    const scalar_t x1240 = x1091 * x20;
    const scalar_t x1241 = x1239 + x1240;
    const scalar_t x1242 = -x1135 - x1136 + x1241;
    const scalar_t x1243 = jacobian_determinant * (x1223 + x139 * (x1231 + x1236 + x1238 + x1242) +
                                                   x322 * (-x1225 - x1227 - x1228 - x1229));
    const scalar_t x1244 = -x1222 * x139;
    const scalar_t x1245 = x1244 + x322 * (x1222 + x1225);
    const scalar_t x1246 = jacobian_determinant * (-x1231 * x139 + x1245);
    const scalar_t x1247 = x1124 + x1126;
    const scalar_t x1248 = x1159 + x1247;
    const scalar_t x1249 = x1090 + x1098;
    const scalar_t x1250 = x1185 + x1220;
    const scalar_t x1251 = x1249 + x1250;
    const scalar_t x1252 = -x1107 - x1108;
    const scalar_t x1253 = x1111 - x1112 - x1113 + x1114;
    const scalar_t x1254 = x1106 + x1109 + x1252 + x1253;
    const scalar_t x1255 = x1251 * x139;
    const scalar_t x1256 = x1255 + x322 * (-x1251 - x1254);
    const scalar_t x1257 = jacobian_determinant * (x1256 + x139 * (-x1153 - x1248));
    const scalar_t x1258 = -x1251 * x139;
    const scalar_t x1259 = x1198 + x1227;
    const scalar_t x1260 = x1205 + x1234;
    const scalar_t x1261 = -x1124 - x1126;
    const scalar_t x1262 = x1260 + x1261;
    const scalar_t x1263 = -x1151 - x1152;
    const scalar_t x1264 = -x1157 - x1158;
    const scalar_t x1265 = x1123 + x1212 + x1241;
    const scalar_t x1266 = jacobian_determinant * (x1258 + x139 * (-x1262 - x1263 - x1264 - x1265) +
                                                   x322 * (x1254 + x1259));
    const scalar_t x1267 = x1054 * x598;
    const scalar_t x1268 = x1267 + x993;
    const scalar_t x1269 = x1054 * x42;
    const scalar_t x1270 = x1049 + x1269;
    const scalar_t x1271 = jacobian_determinant * (x1270 - x135 * x995);
    const scalar_t x1272 = -x1000 + x606 + x666 - x999;
    const scalar_t x1273 = jacobian_determinant * (x1047 + x139 * (-x1041 - x1272));
    const scalar_t x1274 = x1023 + x170;
    const scalar_t x1275 = x1062 + x1274;
    const scalar_t x1276 = -x1002 - x1003 + x610 + x677;
    const scalar_t x1277 = jacobian_determinant * (x1071 + x139 * (-x1275 - x1276));
    const scalar_t x1278 = x1061 + x1269;
    const scalar_t x1279 = jacobian_determinant * (x1278 - x135 * x997);
    const scalar_t x1280 = x1054 * x139;
    const scalar_t x1281 = jacobian_determinant * (x1079 + x1280);
    const scalar_t x1282 = jacobian_determinant * (x1077 + x139 * (x1073 + x1081));
    const scalar_t x1283 = x1156 + x1205;
    const scalar_t x1284 = jacobian_determinant * (x1176 + x42 * (-x1234 - x1283));
    const scalar_t x1285 = x1247 * x598;
    const scalar_t x1286 = x1105 + x1285;
    const scalar_t x1287 = jacobian_determinant * (-x1116 * x135 + x1286);
    const scalar_t x1288 = x1247 * x42;
    const scalar_t x1289 = x1194 + x1288;
    const scalar_t x1290 = jacobian_determinant * (-x1179 * x322 + x1289);
    const scalar_t x1291 = jacobian_determinant * (x1192 + x139 * (-x1206 - x1283));
    const scalar_t x1292 = x1156 + x1234;
    const scalar_t x1293 = jacobian_determinant * (x1245 + x139 * (-x1235 - x1292));
    const scalar_t x1294 = x1223 + x1288;
    const scalar_t x1295 = jacobian_determinant * (-x1225 * x322 + x1294);
    const scalar_t x1296 = x1247 * x139;
    const scalar_t x1297 = x1258 + x1296;
    const scalar_t x1298 = jacobian_determinant * (x1254 * x322 + x1297);
    const scalar_t x1299 = jacobian_determinant * (x1256 + x1262 * x139);
    const scalar_t x1300 = -x51;
    const scalar_t x1301 = x1300 + x654 + x856 - x994;
    const scalar_t x1302 = x1301 + x997;
    const scalar_t x1303 = -x981;
    const scalar_t x1304 = x1303 - x982 - x983;
    const scalar_t x1305 = x1304 + x659 + x991;
    const scalar_t x1306 = x1305 * x42;
    const scalar_t x1307 = x1267 + x1306;
    const scalar_t x1308 = -x1025;
    const scalar_t x1309 = -x1028 + x1308 + x605 + x668;
    const scalar_t x1310 = -x1305 * x42;
    const scalar_t x1311 = x52 + x794;
    const scalar_t x1312 = -x1036 - x1037;
    const scalar_t x1313 =
            jacobian_determinant *
            (x1310 + x135 * (x1035 + x1305 + x1311 + x1312 + x672) + x42 * (-x1041 - x1309));
    const scalar_t x1314 = -x1026;
    const scalar_t x1315 = -x1021 + x1314 + x609 + x679;
    const scalar_t x1316 = x1074 + x1304 + x681;
    const scalar_t x1317 = -x1316;
    const scalar_t x1318 = x1316 * x42 + x1317 * x135;
    const scalar_t x1319 = jacobian_determinant * (x1318 + x139 * (-x1063 - x1315));
    const scalar_t x1320 = x1317 * x42;
    const scalar_t x1321 = jacobian_determinant * (x1280 + x1320);
    const scalar_t x1322 = jacobian_determinant * (x1278 + x135 * x997);
    const scalar_t x1323 = x691 + x83;
    const scalar_t x1324 = x108 + x219;
    const scalar_t x1325 = -x1005 - x1006;
    const scalar_t x1326 = x144 - x29;
    const scalar_t x1327 = x1070 + x135 * (-x1326 - x694 - x806 + x985 - x989 - x997);
    const scalar_t x1328 = jacobian_determinant * (x1327 + x139 * (x1069 + x1323 + x1324 + x1325));
    const scalar_t x1329 = x1110 - x1167 - x1168 + x1172;
    const scalar_t x1330 = x1094 + x1180;
    const scalar_t x1331 = x1098 + x1189 + x1330;
    const scalar_t x1332 = x1331 * x139;
    const scalar_t x1333 = x1332 + x322 * (x1187 + x1215 + x1329);
    const scalar_t x1334 = jacobian_determinant * (x1333 + x139 * (-x1148 - x1283));
    const scalar_t x1335 = -x1331 * x139;
    const scalar_t x1336 = x1288 + x1335;
    const scalar_t x1337 = jacobian_determinant * (-x1329 * x322 + x1336);
    const scalar_t x1338 = -x1106 - x1109;
    const scalar_t x1339 = x1115 + x1252 + x1338;
    const scalar_t x1340 = x1102 + x1188 + x1330;
    const scalar_t x1341 = x1340 * x139;
    const scalar_t x1342 = x1285 + x1341;
    const scalar_t x1343 = jacobian_determinant * (x1339 * x135 + x1342);
    const scalar_t x1344 = -x1232 - x1233;
    const scalar_t x1345 = x1186 + x1216;
    const scalar_t x1346 = x1249 + x1345;
    const scalar_t x1347 = -x1161 - x1162;
    const scalar_t x1348 = x1173 + x1224 + x1347;
    const scalar_t x1349 = -x1340 * x139;
    const scalar_t x1350 = x1349 + x322 * (-x1346 - x1348);
    const scalar_t x1351 = jacobian_determinant * (x1350 + x42 * (-x1283 - x1344));
    const scalar_t x1352 = -x1203 - x1204;
    const scalar_t x1353 = x1234 + x1352;
    const scalar_t x1354 = x1107 + x1108 + x1338;
    const scalar_t x1355 = x1253 + x1354;
    const scalar_t x1356 = -x1218 - x1219;
    const scalar_t x1357 = x1180 + x1188;
    const scalar_t x1358 = x1250 + x1357;
    const scalar_t x1359 = x1358 * x139;
    const scalar_t x1360 = x1359 + x322 * (x1189 + x1249 + x1355 + x1356);
    const scalar_t x1361 = jacobian_determinant * (x1360 + x139 * (-x1247 - x1353));
    const scalar_t x1362 = -x1358 * x139;
    const scalar_t x1363 = x1296 + x1362;
    const scalar_t x1364 = jacobian_determinant * (-x1355 * x322 + x1363);
    const scalar_t x1365 = x1115 + x1166 + x1347;
    const scalar_t x1366 = x1181 + x1221;
    const scalar_t x1367 = x1366 * x139;
    const scalar_t x1368 = x1288 + x1367;
    const scalar_t x1369 = jacobian_determinant * (x1365 * x322 + x1368);
    const scalar_t x1370 = -x1154 - x1155;
    const scalar_t x1371 = x1133 + x1234;
    const scalar_t x1372 = x1370 + x1371;
    const scalar_t x1373 = -x1366 * x139;
    const scalar_t x1374 = x1373 + x322 * (-x1102 - x1215 - x1356 - x1365);
    const scalar_t x1375 = jacobian_determinant * (x1372 * x139 + x1374);
    const scalar_t x1376 = -x57;
    const scalar_t x1377 = x1376 + x673 + x793 - x996;
    const scalar_t x1378 = -x1018;
    const scalar_t x1379 = -x1019 + x1378 - x224 + x797;
    const scalar_t x1380 = x1052 + x1064;
    const scalar_t x1381 = jacobian_determinant *
                           (x1316 * x61 + x1320 + x139 * (x1055 + x1066 + x1084 + x1380 + x800));
    const scalar_t x1382 = x1054 + x1309;
    const scalar_t x1383 = jacobian_determinant * (x1318 + x139 * (-x1051 - x1382));
    const scalar_t x1384 = jacobian_determinant * (x1327 + x139 * (-x1010 - x1309));
    const scalar_t x1385 = jacobian_determinant *
                           (x1061 + x135 * (8 * x1 + x1043 + x158 + x37 + x807 + 4 * x988 + x997) +
                            x42 * (x1067 + x1382));
    const scalar_t x1386 = x1226 + x1229;
    const scalar_t x1387 = -x1143 - x1144;
    const scalar_t x1388 = x1160 + x1387;
    const scalar_t x1389 = x1148 + x1205;
    const scalar_t x1390 = jacobian_determinant * (x1335 + x139 * (x1213 + x1238 + x1388 + x1389) +
                                                   x322 * (-x1102 - x1199 - x1329 - x1386));
    const scalar_t x1391 = jacobian_determinant * (x1333 - x1388 * x139);
    const scalar_t x1392 = jacobian_determinant * (x1350 + x42 * (-x1160 - x1263));
    const scalar_t x1393 = x1208 + x1237;
    const scalar_t x1394 = x1206 + x1387;
    const scalar_t x1395 = jacobian_determinant *
                           (x1341 + x135 * (x1339 + x1346) + x139 * (x1142 + x1393 + x1394));
    const scalar_t x1396 = x1248 + x1263;
    const scalar_t x1397 = -x1239 - x1240;
    const scalar_t x1398 = -x1210 - x1211;
    const scalar_t x1399 = x1123 + x1397 + x1398;
    const scalar_t x1400 = jacobian_determinant * (x1362 + x139 * (x1353 + x1396 + x1399) +
                                                   x322 * (-x1197 - x1226 - x1345 - x1355));
    const scalar_t x1401 = jacobian_determinant * (x1360 - x139 * x1396);
    const scalar_t x1402 = jacobian_determinant * (x1374 + x139 * (-x1130 - x1156 - x1263));
    const scalar_t x1403 = 8 * x1099;
    const scalar_t x1404 = lambda * x1403 + mu * x1403 + x1186;
    const scalar_t x1405 = -x1128 - x1129;
    const scalar_t x1406 = x1137 + x1405;
    const scalar_t x1407 =
            jacobian_determinant * (x1367 + x139 * (-x1153 - x1238 - x1372 - x1397 - x1406) +
                                    x322 * (x1365 + x1386 + x1404));
    const scalar_t x1408 = x1001 + x1274;
    const scalar_t x1409 = x1016 + x1276;
    const scalar_t x1410 = x58 + x655;
    const scalar_t x1411 = -x1033 - x1034;
    const scalar_t x1412 =
            jacobian_determinant *
            (x1310 + x135 * (x1038 + x1305 + x1410 + x1411 + x855) + x42 * (-x1275 - x1315));
    const scalar_t x1413 = x1046 + x135 * (-x1045 - x995);
    const scalar_t x1414 = jacobian_determinant * (x139 * (-x1315 - x1408) + x1413);
    const scalar_t x1415 = jacobian_determinant *
                           (x1049 + x135 * (x1059 + x150 + 8 * x24 + x30 + x862 + 4 * x985 + x995) +
                            x42 * (x1056 + x1315));
    const scalar_t x1416 = x1094 + x1202;
    const scalar_t x1417 = x1230 + x1405;
    const scalar_t x1418 = jacobian_determinant * (x1367 + x139 * (x1209 + x1242 + x1371 + x1417) +
                                                   x322 * (-x1197 - x1228 - x1365 - x1416));
    const scalar_t x1419 = x1373 + x322 * (x1365 + x1366);
    const scalar_t x1420 = jacobian_determinant * (-x139 * x1417 + x1419);
    const scalar_t x1421 = x1153 + x1247 + x1264;
    const scalar_t x1422 = x1359 + x322 * (-x1355 - x1358);
    const scalar_t x1423 = jacobian_determinant * (-x139 * x1421 + x1422);
    const scalar_t x1424 = x1205 + x1344;
    const scalar_t x1425 = jacobian_determinant * (x1362 + x139 * (x1399 + x1421 + x1424) +
                                                   x322 * (x1102 + x1229 + x1355 + x1416));
    const scalar_t x1426 = x1125 + x1127 + x1141 + x1235 + x1406;
    const scalar_t x1427 = jacobian_determinant *
                           (x1341 + x135 * (-x1339 - x1340) + x139 * (x1149 + x1393 + x1426));
    const scalar_t x1428 = x1349 + x322 * (x1340 + x1348);
    const scalar_t x1429 = jacobian_determinant * (x1428 + x42 * (-x1230 - x1264));
    const scalar_t x1430 = x1332 + x322 * (-x1329 - x1331);
    const scalar_t x1431 = jacobian_determinant * (x139 * (-x1145 - x1156 - x1264) + x1430);
    const scalar_t x1432 = x1197 + x1202;
    const scalar_t x1433 = 8 * x1091;
    const scalar_t x1434 = lambda * x1433 + mu * x1433 + x1216;
    const scalar_t x1435 = x1370 + x1389;
    const scalar_t x1436 = x1141 + x1398;
    const scalar_t x1437 =
            jacobian_determinant * (x1335 + x139 * (-x1159 - x1209 - x1387 - x1435 - x1436) +
                                    x322 * (x1329 + x1432 + x1434));
    const scalar_t x1438 = jacobian_determinant * (x1270 + x135 * x995);
    const scalar_t x1439 = x1325 + x219 + x691;
    const scalar_t x1440 = jacobian_determinant * (x139 * (x1001 + x1051 + x1439) + x1413);
    const scalar_t x1441 = jacobian_determinant * (x139 * (-x1133 - x1292) + x1419);
    const scalar_t x1442 = jacobian_determinant * (-x1365 * x322 + x1368);
    const scalar_t x1443 = jacobian_determinant * (x1355 * x322 + x1363);
    const scalar_t x1444 = jacobian_determinant * (x139 * (-x1247 - x1424) + x1422);
    const scalar_t x1445 = jacobian_determinant * (x1428 + x42 * (-x1292 - x1352));
    const scalar_t x1446 = jacobian_determinant * (-x1339 * x135 + x1342);
    const scalar_t x1447 = jacobian_determinant * (x1329 * x322 + x1336);
    const scalar_t x1448 = jacobian_determinant * (x139 * x1435 + x1430);
    const scalar_t x1449 = x655 + x794;
    const scalar_t x1450 =
            jacobian_determinant *
            (x1031 + x135 * (x1312 + x1411 + x1449 + x926 + x992) + x42 * (x1073 + x1439));
    const scalar_t x1451 = -x1111 + x1112 + x1113 - x1114 + x1354;
    const scalar_t x1452 = x1255 + x322 * (-x1251 - x1451);
    const scalar_t x1453 = jacobian_determinant * (x139 * (-x1247 - x1260) + x1452);
    const scalar_t x1454 = jacobian_determinant * (-x1254 * x322 + x1297);
    const scalar_t x1455 = jacobian_determinant * (x1225 * x322 + x1294);
    const scalar_t x1456 = x1244 + x322 * (-x1103 - x1180 - x1225 - x1356);
    const scalar_t x1457 = jacobian_determinant * (x139 * (-x1133 - x1156 - x1344) + x1456);
    const scalar_t x1458 = x1191 + x322 * (-x1179 - x1190);
    const scalar_t x1459 = jacobian_determinant * (x139 * (-x1148 - x1156 - x1352) + x1458);
    const scalar_t x1460 = jacobian_determinant * (x1179 * x322 + x1289);
    const scalar_t x1461 = jacobian_determinant * (x1116 * x135 + x1286);
    const scalar_t x1462 = x1345 + x1357;
    const scalar_t x1463 = x1175 + x322 * (-x1174 - x1462);
    const scalar_t x1464 = jacobian_determinant * (x1463 + x42 * (x1260 + x1370));
    const scalar_t x1465 = x1153 + x1159;
    const scalar_t x1466 = x1261 + x1465;
    const scalar_t x1467 = jacobian_determinant * (x1258 + x139 * (-x1265 - x1344 - x1352 - x1466) +
                                                   x322 * (x1259 + x1451));
    const scalar_t x1468 = jacobian_determinant * (x139 * x1466 + x1452);
    const scalar_t x1469 = x1130 + x1153 + x1370;
    const scalar_t x1470 = jacobian_determinant * (x139 * x1469 + x1456);
    const scalar_t x1471 =
            jacobian_determinant * (x1223 + x139 * (-x1137 - x1209 - x1236 - x1397 - x1469) +
                                    x322 * (x1225 + x1404 + x1432));
    const scalar_t x1472 = x1145 + x1159 + x1370;
    const scalar_t x1473 = jacobian_determinant * (x1194 + x139 * (-x1207 - x1238 - x1436 - x1472) +
                                                   x322 * (x1179 + x1386 + x1434));
    const scalar_t x1474 = jacobian_determinant * (x139 * x1472 + x1458);
    const scalar_t x1475 = jacobian_determinant * (x1463 + x42 * (x1370 + x1465));
    const scalar_t x1476 = jacobian_determinant *
                           (x1105 + x135 * (x1116 + x1462) + x139 * (x1123 + x1394 + x1426));
    const scalar_t x1477 = x230 * x7;
    const scalar_t x1478 = x20 * x7;
    const scalar_t x1479 = x1477 + x1478 + x981;
    const scalar_t x1480 = lambda * x12;
    const scalar_t x1481 = 2 * x1480;
    const scalar_t x1482 = x1481 + x187;
    const scalar_t x1483 = lambda * x9;
    const scalar_t x1484 = 2 * x1483;
    const scalar_t x1485 = x1484 + x176 + x25;
    const scalar_t x1486 = x1485 + x160;
    const scalar_t x1487 = x144 + x1482 + x1486;
    const scalar_t x1488 = x1479 + x1487 + x6;
    const scalar_t x1489 = x1488 * x42;
    const scalar_t x1490 = adjugate[8] * x438;
    const scalar_t x1491 = x123 + x1490 + x47 + x51;
    const scalar_t x1492 = adjugate[5] * x438;
    const scalar_t x1493 = x124 + x1492 + x55 + x57;
    const scalar_t x1494 = x1491 + x1493;
    const scalar_t x1495 = 6 * x1492;
    const scalar_t x1496 = 12 * x56;
    const scalar_t x1497 = x113 + x1495 + x1496 + x63;
    const scalar_t x1498 = lambda * x71;
    const scalar_t x1499 = 2 * x1498;
    const scalar_t x1500 = 4 * x72;
    const scalar_t x1501 = x1499 + x1500;
    const scalar_t x1502 = 6 * x1490;
    const scalar_t x1503 = 12 * x49;
    const scalar_t x1504 = x1502 + x1503;
    const scalar_t x1505 = x1022 + x108 + x1504;
    const scalar_t x1506 = x1501 + x1505;
    const scalar_t x1507 = x263 * x7;
    const scalar_t x1508 = 6 * x1483;
    const scalar_t x1509 = 6 * x1480;
    const scalar_t x1510 = 12 * x10;
    const scalar_t x1511 = 12 * x13;
    const scalar_t x1512 = x1015 + x1508 + x1509 + x1510 + x1511;
    const scalar_t x1513 = x1018 + x105 + x1507 + x1512 + x222;
    const scalar_t x1514 = 3 * x1490;
    const scalar_t x1515 = x1008 + x1501 + x1514;
    const scalar_t x1516 = 3 * x1492;
    const scalar_t x1517 = x1516 + x69;
    const scalar_t x1518 = x1027 + x120;
    const scalar_t x1519 = -x1488 * x42;
    const scalar_t x1520 = 2 * x1490;
    const scalar_t x1521 = 4 * x49;
    const scalar_t x1522 = x1520 + x1521;
    const scalar_t x1523 = 2 * x1492;
    const scalar_t x1524 = 4 * x56;
    const scalar_t x1525 = x1523 + x1524;
    const scalar_t x1526 = jacobian_determinant * (x135 * (x1032 + x128 + x1488 + x1522 + x1525) +
                                                   x1519 + x42 * (-x1515 - x1517 - x1518));
    const scalar_t x1527 = x1026 + x119;
    const scalar_t x1528 = x1515 + x1527;
    const scalar_t x1529 = x1480 + x36;
    const scalar_t x1530 = -x1484 + x693;
    const scalar_t x1531 = x1044 + x148 - x1480 + x1485 + x862;
    const scalar_t x1532 = x1531 * x42;
    const scalar_t x1533 = x135 * (x147 + x1491 + x1529 + x1530 + x162) + x1532;
    const scalar_t x1534 = jacobian_determinant * (x139 * (-x1497 - x1528) + x1533);
    const scalar_t x1535 = -x1531 * x42;
    const scalar_t x1536 = x1514 + x83;
    const scalar_t x1537 = x1527 + x1536;
    const scalar_t x1538 = x1498 + x66 + x73 + x77;
    const scalar_t x1539 = -x1508 - x1510 + x1538;
    const scalar_t x1540 = x1053 + 3 * x1480 + x1539 + x169 + x98;
    const scalar_t x1541 = jacobian_determinant *
                           (x135 * (4 * lambda * x9 + 8 * mu * x9 - x145 - x1482 - x1491 - x159) +
                            x1535 + x42 * (x1537 + x1540));
    const scalar_t x1542 = x1483 + x24 + x29;
    const scalar_t x1543 = -x1481 + x150 + x175;
    const scalar_t x1544 = x1542 + x1543 + x189;
    const scalar_t x1545 = x1544 * x42;
    const scalar_t x1546 = x1025 + x118;
    const scalar_t x1547 = x1516 + x68;
    const scalar_t x1548 = x1538 + x1547;
    const scalar_t x1549 = x1546 + x1548;
    const scalar_t x1550 = -x1509 - x1511;
    const scalar_t x1551 = x1065 + 3 * x1483 + x1550 + x198 + x97;
    const scalar_t x1552 = jacobian_determinant *
                           (x135 * (4 * lambda * x12 + 8 * mu * x12 - x1486 - x1493 - x181 - x186) +
                            x1545 + x42 * (x1549 + x1551));
    const scalar_t x1553 = x1501 + x1517 + x1546;
    const scalar_t x1554 = -x1544 * x42;
    const scalar_t x1555 = x135 * (x1493 + x1544) + x1554;
    const scalar_t x1556 = jacobian_determinant * (x139 * (-x1505 - x1553) + x1555);
    const scalar_t x1557 = x1518 + x1536;
    const scalar_t x1558 = x15 + x1529 + x1542;
    const scalar_t x1559 = x1479 + x1558 + x207;
    const scalar_t x1560 = -x1559;
    const scalar_t x1561 = x135 * x1560 + x1559 * x42;
    const scalar_t x1562 = jacobian_determinant * (x139 * (-x1548 - x1557) + x1561);
    const scalar_t x1563 = x1560 * x42;
    const scalar_t x1564 = x1080 - x1498 + x217;
    const scalar_t x1565 = x1082 * x7 + x1083 * x7 + x88;
    const scalar_t x1566 = jacobian_determinant *
                           (x139 * (-x1512 - x1564 - x1565 - x225 - x691) + x1559 * x61 + x1563);
    const scalar_t x1567 = x1538 * x598;
    const scalar_t x1568 = x1489 + x1567;
    const scalar_t x1569 = x1538 * x42;
    const scalar_t x1570 = x1535 + x1569;
    const scalar_t x1571 = jacobian_determinant * (-x135 * x1491 + x1570);
    const scalar_t x1572 = x1528 + x67;
    const scalar_t x1573 = -x1495 - x1496 + x605 + x666;
    const scalar_t x1574 = jacobian_determinant * (x139 * (-x1572 - x1573) + x1533);
    const scalar_t x1575 = x1553 + x170;
    const scalar_t x1576 = -x1502 - x1503 + x609 + x677;
    const scalar_t x1577 = jacobian_determinant * (x139 * (-x1575 - x1576) + x1555);
    const scalar_t x1578 = x1545 + x1569;
    const scalar_t x1579 = jacobian_determinant * (-x135 * x1493 + x1578);
    const scalar_t x1580 = x139 * x1538;
    const scalar_t x1581 = jacobian_determinant * (x1563 + x1580);
    const scalar_t x1582 =
            jacobian_determinant * (x139 * (x1323 + x1514 + x1518 + x1547 + x1564) + x1561);
    const scalar_t x1583 = x1300 - x1490 + x653 + x854;
    const scalar_t x1584 = x1493 + x1583;
    const scalar_t x1585 = x1303 - x1477 - x1478;
    const scalar_t x1586 = x1487 + x1585 + x661;
    const scalar_t x1587 = x1586 * x42;
    const scalar_t x1588 = x1567 + x1587;
    const scalar_t x1589 = x1308 - x1516 + x606 + x667;
    const scalar_t x1590 = -x1586 * x42;
    const scalar_t x1591 = -x1523 - x1524;
    const scalar_t x1592 = jacobian_determinant * (x135 * (x1311 + x1522 + x1586 + x1591 + x674) +
                                                   x1590 + x42 * (-x1572 - x1589));
    const scalar_t x1593 = x1314 - x1514 + x610 + x678;
    const scalar_t x1594 = x1558 + x1585 + x682;
    const scalar_t x1595 = -x1594;
    const scalar_t x1596 = x135 * x1595 + x1594 * x42;
    const scalar_t x1597 = jacobian_determinant * (x139 * (-x1549 - x1593) + x1596);
    const scalar_t x1598 = x1595 * x42;
    const scalar_t x1599 = jacobian_determinant * (x1580 + x1598);
    const scalar_t x1600 = jacobian_determinant * (x135 * x1493 + x1578);
    const scalar_t x1601 = -x1499 - x1500 + x690;
    const scalar_t x1602 = x1547 + x1601;
    const scalar_t x1603 = x135 * (-x1326 - x1481 + x1483 - x1493 - x695 - x807) + x1554;
    const scalar_t x1604 =
            jacobian_determinant * (x139 * (x1324 + x1504 + x1546 + x1602 + x82) + x1603);
    const scalar_t x1605 = x1376 - x1492 + x671 + x792;
    const scalar_t x1606 = x1506 + x67;
    const scalar_t x1607 = x1378 - x1507 + x1512 - x222 + x798;
    const scalar_t x1608 = jacobian_determinant *
                           (x139 * (x1380 + x1539 + x1550 + x1565 + x801) + x1594 * x61 + x1598);
    const scalar_t x1609 = x1538 + x1589;
    const scalar_t x1610 = jacobian_determinant * (x139 * (-x1537 - x1609) + x1596);
    const scalar_t x1611 = jacobian_determinant * (x139 * (-x1589 - x1606) + x1603);
    const scalar_t x1612 = jacobian_determinant *
                           (x135 * (8 * x13 + 4 * x1480 + x1493 + x1530 + x161 + x37 + x806) +
                            x1545 + x42 * (x1551 + x1609));
    const scalar_t x1613 = x1501 + x170;
    const scalar_t x1614 = x1497 + x1613;
    const scalar_t x1615 = -x1520 - x1521;
    const scalar_t x1616 = jacobian_determinant * (x135 * (x1410 + x1525 + x1586 + x1615 + x857) +
                                                   x1590 + x42 * (-x1575 - x1593));
    const scalar_t x1617 = x135 * (-x1491 - x1531) + x1532;
    const scalar_t x1618 = jacobian_determinant * (x139 * (-x1593 - x1614) + x1617);
    const scalar_t x1619 = jacobian_determinant *
                           (x135 * (8 * x10 + 4 * x1483 + x1491 + x1543 + x157 + x188 + x30) +
                            x1535 + x42 * (x1540 + x1593));
    const scalar_t x1620 = jacobian_determinant * (x135 * x1491 + x1570);
    const scalar_t x1621 = jacobian_determinant *
                           (x139 * (x112 + x1495 + x1496 + x1537 + x1601 + x219 + x63) + x1617);
    const scalar_t x1622 = jacobian_determinant * (x135 * (x1449 + x1488 + x1591 + x1615 + x927) +
                                                   x1519 + x42 * (x1557 + x1602 + x219));
    element_matrix[0] =
            jacobian_determinant * (x42 * (x106 + x70 + x86) + x43 + x61 * (-x40 - x60));
    element_matrix[1] = x136;
    element_matrix[2] = x155;
    element_matrix[3] = x174;
    element_matrix[4] = x201;
    element_matrix[5] = x205;
    element_matrix[6] = x215;
    element_matrix[7] = x227;
    element_matrix[8] = x297;
    element_matrix[9] = x325;
    element_matrix[10] = x341;
    element_matrix[11] = x363;
    element_matrix[12] = x392;
    element_matrix[13] = x395;
    element_matrix[14] = x406;
    element_matrix[15] = x415;
    element_matrix[16] = x481;
    element_matrix[17] = x508;
    element_matrix[18] = x524;
    element_matrix[19] = x545;
    element_matrix[20] = x574;
    element_matrix[21] = x577;
    element_matrix[22] = x588;
    element_matrix[23] = x597;
    element_matrix[24] = x136;
    element_matrix[25] = jacobian_determinant * (-x60 * x61 + x600);
    element_matrix[26] = x603;
    element_matrix[27] = x608;
    element_matrix[28] = x612;
    element_matrix[29] = x614;
    element_matrix[30] = x616;
    element_matrix[31] = x618;
    element_matrix[32] = x620;
    element_matrix[33] = x623;
    element_matrix[34] = x626;
    element_matrix[35] = x627;
    element_matrix[36] = x629;
    element_matrix[37] = x631;
    element_matrix[38] = x634;
    element_matrix[39] = x635;
    element_matrix[40] = x637;
    element_matrix[41] = x640;
    element_matrix[42] = x643;
    element_matrix[43] = x644;
    element_matrix[44] = x646;
    element_matrix[45] = x648;
    element_matrix[46] = x651;
    element_matrix[47] = x652;
    element_matrix[48] = x155;
    element_matrix[49] = x603;
    element_matrix[50] = jacobian_determinant * (x61 * x657 + x665);
    element_matrix[51] = x676;
    element_matrix[52] = x686;
    element_matrix[53] = x688;
    element_matrix[54] = x689;
    element_matrix[55] = x697;
    element_matrix[56] = x703;
    element_matrix[57] = x706;
    element_matrix[58] = x712;
    element_matrix[59] = x720;
    element_matrix[60] = x730;
    element_matrix[61] = x733;
    element_matrix[62] = x738;
    element_matrix[63] = x744;
    element_matrix[64] = x750;
    element_matrix[65] = x753;
    element_matrix[66] = x759;
    element_matrix[67] = x767;
    element_matrix[68] = x777;
    element_matrix[69] = x780;
    element_matrix[70] = x785;
    element_matrix[71] = x791;
    element_matrix[72] = x174;
    element_matrix[73] = x608;
    element_matrix[74] = x676;
    element_matrix[75] =
            jacobian_determinant * (x42 * (x607 + x796 + x799) + x61 * (-x53 - x663 - x795) + x664);
    element_matrix[76] = x802;
    element_matrix[77] = x804;
    element_matrix[78] = x805;
    element_matrix[79] = x808;
    element_matrix[80] = x813;
    element_matrix[81] = x814;
    element_matrix[82] = x815;
    element_matrix[83] = x818;
    element_matrix[84] = x823;
    element_matrix[85] = x824;
    element_matrix[86] = x825;
    element_matrix[87] = x830;
    element_matrix[88] = x835;
    element_matrix[89] = x836;
    element_matrix[90] = x837;
    element_matrix[91] = x840;
    element_matrix[92] = x845;
    element_matrix[93] = x846;
    element_matrix[94] = x847;
    element_matrix[95] = x852;
    element_matrix[96] = x201;
    element_matrix[97] = x612;
    element_matrix[98] = x686;
    element_matrix[99] = x802;
    element_matrix[100] =
            jacobian_determinant * (x42 * (x611 + x799 + x853) + x61 * (-x657 - x663) + x664);
    element_matrix[101] = x859;
    element_matrix[102] = x861;
    element_matrix[103] = x863;
    element_matrix[104] = x866;
    element_matrix[105] = x868;
    element_matrix[106] = x871;
    element_matrix[107] = x873;
    element_matrix[108] = x875;
    element_matrix[109] = x877;
    element_matrix[110] = x879;
    element_matrix[111] = x885;
    element_matrix[112] = x888;
    element_matrix[113] = x890;
    element_matrix[114] = x893;
    element_matrix[115] = x895;
    element_matrix[116] = x897;
    element_matrix[117] = x899;
    element_matrix[118] = x901;
    element_matrix[119] = x907;
    element_matrix[120] = x205;
    element_matrix[121] = x614;
    element_matrix[122] = x688;
    element_matrix[123] = x804;
    element_matrix[124] = x859;
    element_matrix[125] = jacobian_determinant * (-x61 * x657 + x665);
    element_matrix[126] = x908;
    element_matrix[127] = x909;
    element_matrix[128] = x910;
    element_matrix[129] = x911;
    element_matrix[130] = x912;
    element_matrix[131] = x913;
    element_matrix[132] = x914;
    element_matrix[133] = x915;
    element_matrix[134] = x916;
    element_matrix[135] = x917;
    element_matrix[136] = x918;
    element_matrix[137] = x919;
    element_matrix[138] = x920;
    element_matrix[139] = x921;
    element_matrix[140] = x922;
    element_matrix[141] = x923;
    element_matrix[142] = x924;
    element_matrix[143] = x925;
    element_matrix[144] = x215;
    element_matrix[145] = x616;
    element_matrix[146] = x689;
    element_matrix[147] = x805;
    element_matrix[148] = x861;
    element_matrix[149] = x908;
    element_matrix[150] = jacobian_determinant * (x60 * x61 + x600);
    element_matrix[151] = x928;
    element_matrix[152] = x931;
    element_matrix[153] = x932;
    element_matrix[154] = x933;
    element_matrix[155] = x935;
    element_matrix[156] = x937;
    element_matrix[157] = x938;
    element_matrix[158] = x939;
    element_matrix[159] = x942;
    element_matrix[160] = x945;
    element_matrix[161] = x946;
    element_matrix[162] = x947;
    element_matrix[163] = x949;
    element_matrix[164] = x951;
    element_matrix[165] = x952;
    element_matrix[166] = x953;
    element_matrix[167] = x956;
    element_matrix[168] = x227;
    element_matrix[169] = x618;
    element_matrix[170] = x697;
    element_matrix[171] = x808;
    element_matrix[172] = x863;
    element_matrix[173] = x909;
    element_matrix[174] = x928;
    element_matrix[175] = jacobian_determinant * (x42 * (x106 + x607 + x611 + x67 + x80) + x43 +
                                                  x61 * (-x40 - x656 - x795));
    element_matrix[176] = x959;
    element_matrix[177] = x960;
    element_matrix[178] = x962;
    element_matrix[179] = x963;
    element_matrix[180] = x965;
    element_matrix[181] = x966;
    element_matrix[182] = x967;
    element_matrix[183] = x968;
    element_matrix[184] = x971;
    element_matrix[185] = x972;
    element_matrix[186] = x974;
    element_matrix[187] = x975;
    element_matrix[188] = x977;
    element_matrix[189] = x978;
    element_matrix[190] = x979;
    element_matrix[191] = x980;
    element_matrix[192] = x297;
    element_matrix[193] = x620;
    element_matrix[194] = x703;
    element_matrix[195] = x813;
    element_matrix[196] = x866;
    element_matrix[197] = x910;
    element_matrix[198] = x931;
    element_matrix[199] = x959;
    element_matrix[200] =
            jacobian_determinant * (x42 * (x1001 + x1017 + x1020) + x61 * (-x992 - x998) + x993);
    element_matrix[201] = x1039;
    element_matrix[202] = x1048;
    element_matrix[203] = x1057;
    element_matrix[204] = x1068;
    element_matrix[205] = x1072;
    element_matrix[206] = x1078;
    element_matrix[207] = x1085;
    element_matrix[208] = x1150;
    element_matrix[209] = x1177;
    element_matrix[210] = x1193;
    element_matrix[211] = x1214;
    element_matrix[212] = x1243;
    element_matrix[213] = x1246;
    element_matrix[214] = x1257;
    element_matrix[215] = x1266;
    element_matrix[216] = x325;
    element_matrix[217] = x623;
    element_matrix[218] = x706;
    element_matrix[219] = x814;
    element_matrix[220] = x868;
    element_matrix[221] = x911;
    element_matrix[222] = x932;
    element_matrix[223] = x960;
    element_matrix[224] = x1039;
    element_matrix[225] = jacobian_determinant * (x1268 - x61 * x998);
    element_matrix[226] = x1271;
    element_matrix[227] = x1273;
    element_matrix[228] = x1277;
    element_matrix[229] = x1279;
    element_matrix[230] = x1281;
    element_matrix[231] = x1282;
    element_matrix[232] = x1284;
    element_matrix[233] = x1287;
    element_matrix[234] = x1290;
    element_matrix[235] = x1291;
    element_matrix[236] = x1293;
    element_matrix[237] = x1295;
    element_matrix[238] = x1298;
    element_matrix[239] = x1299;
    element_matrix[240] = x341;
    element_matrix[241] = x626;
    element_matrix[242] = x712;
    element_matrix[243] = x815;
    element_matrix[244] = x871;
    element_matrix[245] = x912;
    element_matrix[246] = x933;
    element_matrix[247] = x962;
    element_matrix[248] = x1048;
    element_matrix[249] = x1271;
    element_matrix[250] = jacobian_determinant * (x1302 * x61 + x1307);
    element_matrix[251] = x1313;
    element_matrix[252] = x1319;
    element_matrix[253] = x1321;
    element_matrix[254] = x1322;
    element_matrix[255] = x1328;
    element_matrix[256] = x1334;
    element_matrix[257] = x1337;
    element_matrix[258] = x1343;
    element_matrix[259] = x1351;
    element_matrix[260] = x1361;
    element_matrix[261] = x1364;
    element_matrix[262] = x1369;
    element_matrix[263] = x1375;
    element_matrix[264] = x363;
    element_matrix[265] = x627;
    element_matrix[266] = x720;
    element_matrix[267] = x818;
    element_matrix[268] = x873;
    element_matrix[269] = x913;
    element_matrix[270] = x935;
    element_matrix[271] = x963;
    element_matrix[272] = x1057;
    element_matrix[273] = x1273;
    element_matrix[274] = x1313;
    element_matrix[275] = jacobian_determinant *
                          (x1306 + x42 * (x1017 + x1272 + x1379) + x61 * (-x1305 - x1377 - x995));
    element_matrix[276] = x1381;
    element_matrix[277] = x1383;
    element_matrix[278] = x1384;
    element_matrix[279] = x1385;
    element_matrix[280] = x1390;
    element_matrix[281] = x1391;
    element_matrix[282] = x1392;
    element_matrix[283] = x1395;
    element_matrix[284] = x1400;
    element_matrix[285] = x1401;
    element_matrix[286] = x1402;
    element_matrix[287] = x1407;
    element_matrix[288] = x392;
    element_matrix[289] = x629;
    element_matrix[290] = x730;
    element_matrix[291] = x823;
    element_matrix[292] = x875;
    element_matrix[293] = x914;
    element_matrix[294] = x937;
    element_matrix[295] = x965;
    element_matrix[296] = x1068;
    element_matrix[297] = x1277;
    element_matrix[298] = x1319;
    element_matrix[299] = x1381;
    element_matrix[300] =
            jacobian_determinant * (x1306 + x42 * (x1379 + x1408 + x1409) + x61 * (-x1302 - x1305));
    element_matrix[301] = x1412;
    element_matrix[302] = x1414;
    element_matrix[303] = x1415;
    element_matrix[304] = x1418;
    element_matrix[305] = x1420;
    element_matrix[306] = x1423;
    element_matrix[307] = x1425;
    element_matrix[308] = x1427;
    element_matrix[309] = x1429;
    element_matrix[310] = x1431;
    element_matrix[311] = x1437;
    element_matrix[312] = x395;
    element_matrix[313] = x631;
    element_matrix[314] = x733;
    element_matrix[315] = x824;
    element_matrix[316] = x877;
    element_matrix[317] = x915;
    element_matrix[318] = x938;
    element_matrix[319] = x966;
    element_matrix[320] = x1072;
    element_matrix[321] = x1279;
    element_matrix[322] = x1321;
    element_matrix[323] = x1383;
    element_matrix[324] = x1412;
    element_matrix[325] = jacobian_determinant * (-x1302 * x61 + x1307);
    element_matrix[326] = x1438;
    element_matrix[327] = x1440;
    element_matrix[328] = x1441;
    element_matrix[329] = x1442;
    element_matrix[330] = x1443;
    element_matrix[331] = x1444;
    element_matrix[332] = x1445;
    element_matrix[333] = x1446;
    element_matrix[334] = x1447;
    element_matrix[335] = x1448;
    element_matrix[336] = x406;
    element_matrix[337] = x634;
    element_matrix[338] = x738;
    element_matrix[339] = x825;
    element_matrix[340] = x879;
    element_matrix[341] = x916;
    element_matrix[342] = x939;
    element_matrix[343] = x967;
    element_matrix[344] = x1078;
    element_matrix[345] = x1281;
    element_matrix[346] = x1322;
    element_matrix[347] = x1384;
    element_matrix[348] = x1414;
    element_matrix[349] = x1438;
    element_matrix[350] = jacobian_determinant * (x1268 + x61 * x998);
    element_matrix[351] = x1450;
    element_matrix[352] = x1453;
    element_matrix[353] = x1454;
    element_matrix[354] = x1455;
    element_matrix[355] = x1457;
    element_matrix[356] = x1459;
    element_matrix[357] = x1460;
    element_matrix[358] = x1461;
    element_matrix[359] = x1464;
    element_matrix[360] = x415;
    element_matrix[361] = x635;
    element_matrix[362] = x744;
    element_matrix[363] = x830;
    element_matrix[364] = x885;
    element_matrix[365] = x917;
    element_matrix[366] = x942;
    element_matrix[367] = x968;
    element_matrix[368] = x1085;
    element_matrix[369] = x1282;
    element_matrix[370] = x1328;
    element_matrix[371] = x1385;
    element_matrix[372] = x1415;
    element_matrix[373] = x1440;
    element_matrix[374] = x1450;
    element_matrix[375] = jacobian_determinant * (x42 * (x1020 + x1272 + x1274 + x1409) +
                                                  x61 * (-x1301 - x1377 - x992) + x993);
    element_matrix[376] = x1467;
    element_matrix[377] = x1468;
    element_matrix[378] = x1470;
    element_matrix[379] = x1471;
    element_matrix[380] = x1473;
    element_matrix[381] = x1474;
    element_matrix[382] = x1475;
    element_matrix[383] = x1476;
    element_matrix[384] = x481;
    element_matrix[385] = x637;
    element_matrix[386] = x750;
    element_matrix[387] = x835;
    element_matrix[388] = x888;
    element_matrix[389] = x918;
    element_matrix[390] = x945;
    element_matrix[391] = x971;
    element_matrix[392] = x1150;
    element_matrix[393] = x1284;
    element_matrix[394] = x1334;
    element_matrix[395] = x1390;
    element_matrix[396] = x1418;
    element_matrix[397] = x1441;
    element_matrix[398] = x1453;
    element_matrix[399] = x1467;
    element_matrix[400] =
            jacobian_determinant * (x1489 + x42 * (x1497 + x1506 + x1513) + x61 * (-x1488 - x1494));
    element_matrix[401] = x1526;
    element_matrix[402] = x1534;
    element_matrix[403] = x1541;
    element_matrix[404] = x1552;
    element_matrix[405] = x1556;
    element_matrix[406] = x1562;
    element_matrix[407] = x1566;
    element_matrix[408] = x508;
    element_matrix[409] = x640;
    element_matrix[410] = x753;
    element_matrix[411] = x836;
    element_matrix[412] = x890;
    element_matrix[413] = x919;
    element_matrix[414] = x946;
    element_matrix[415] = x972;
    element_matrix[416] = x1177;
    element_matrix[417] = x1287;
    element_matrix[418] = x1337;
    element_matrix[419] = x1391;
    element_matrix[420] = x1420;
    element_matrix[421] = x1442;
    element_matrix[422] = x1454;
    element_matrix[423] = x1468;
    element_matrix[424] = x1526;
    element_matrix[425] = jacobian_determinant * (-x1494 * x61 + x1568);
    element_matrix[426] = x1571;
    element_matrix[427] = x1574;
    element_matrix[428] = x1577;
    element_matrix[429] = x1579;
    element_matrix[430] = x1581;
    element_matrix[431] = x1582;
    element_matrix[432] = x524;
    element_matrix[433] = x643;
    element_matrix[434] = x759;
    element_matrix[435] = x837;
    element_matrix[436] = x893;
    element_matrix[437] = x920;
    element_matrix[438] = x947;
    element_matrix[439] = x974;
    element_matrix[440] = x1193;
    element_matrix[441] = x1290;
    element_matrix[442] = x1343;
    element_matrix[443] = x1392;
    element_matrix[444] = x1423;
    element_matrix[445] = x1443;
    element_matrix[446] = x1455;
    element_matrix[447] = x1470;
    element_matrix[448] = x1534;
    element_matrix[449] = x1571;
    element_matrix[450] = jacobian_determinant * (x1584 * x61 + x1588);
    element_matrix[451] = x1592;
    element_matrix[452] = x1597;
    element_matrix[453] = x1599;
    element_matrix[454] = x1600;
    element_matrix[455] = x1604;
    element_matrix[456] = x545;
    element_matrix[457] = x644;
    element_matrix[458] = x767;
    element_matrix[459] = x840;
    element_matrix[460] = x895;
    element_matrix[461] = x921;
    element_matrix[462] = x949;
    element_matrix[463] = x975;
    element_matrix[464] = x1214;
    element_matrix[465] = x1291;
    element_matrix[466] = x1351;
    element_matrix[467] = x1395;
    element_matrix[468] = x1425;
    element_matrix[469] = x1444;
    element_matrix[470] = x1457;
    element_matrix[471] = x1471;
    element_matrix[472] = x1541;
    element_matrix[473] = x1574;
    element_matrix[474] = x1592;
    element_matrix[475] = jacobian_determinant *
                          (x1587 + x42 * (x1573 + x1606 + x1607) + x61 * (-x1491 - x1586 - x1605));
    element_matrix[476] = x1608;
    element_matrix[477] = x1610;
    element_matrix[478] = x1611;
    element_matrix[479] = x1612;
    element_matrix[480] = x574;
    element_matrix[481] = x646;
    element_matrix[482] = x777;
    element_matrix[483] = x845;
    element_matrix[484] = x897;
    element_matrix[485] = x922;
    element_matrix[486] = x951;
    element_matrix[487] = x977;
    element_matrix[488] = x1243;
    element_matrix[489] = x1293;
    element_matrix[490] = x1361;
    element_matrix[491] = x1400;
    element_matrix[492] = x1427;
    element_matrix[493] = x1445;
    element_matrix[494] = x1459;
    element_matrix[495] = x1473;
    element_matrix[496] = x1552;
    element_matrix[497] = x1577;
    element_matrix[498] = x1597;
    element_matrix[499] = x1608;
    element_matrix[500] =
            jacobian_determinant * (x1587 + x42 * (x1576 + x1607 + x1614) + x61 * (-x1584 - x1586));
    element_matrix[501] = x1616;
    element_matrix[502] = x1618;
    element_matrix[503] = x1619;
    element_matrix[504] = x577;
    element_matrix[505] = x648;
    element_matrix[506] = x780;
    element_matrix[507] = x846;
    element_matrix[508] = x899;
    element_matrix[509] = x923;
    element_matrix[510] = x952;
    element_matrix[511] = x978;
    element_matrix[512] = x1246;
    element_matrix[513] = x1295;
    element_matrix[514] = x1364;
    element_matrix[515] = x1401;
    element_matrix[516] = x1429;
    element_matrix[517] = x1446;
    element_matrix[518] = x1460;
    element_matrix[519] = x1474;
    element_matrix[520] = x1556;
    element_matrix[521] = x1579;
    element_matrix[522] = x1599;
    element_matrix[523] = x1610;
    element_matrix[524] = x1616;
    element_matrix[525] = jacobian_determinant * (-x1584 * x61 + x1588);
    element_matrix[526] = x1620;
    element_matrix[527] = x1621;
    element_matrix[528] = x588;
    element_matrix[529] = x651;
    element_matrix[530] = x785;
    element_matrix[531] = x847;
    element_matrix[532] = x901;
    element_matrix[533] = x924;
    element_matrix[534] = x953;
    element_matrix[535] = x979;
    element_matrix[536] = x1257;
    element_matrix[537] = x1298;
    element_matrix[538] = x1369;
    element_matrix[539] = x1402;
    element_matrix[540] = x1431;
    element_matrix[541] = x1447;
    element_matrix[542] = x1461;
    element_matrix[543] = x1475;
    element_matrix[544] = x1562;
    element_matrix[545] = x1581;
    element_matrix[546] = x1600;
    element_matrix[547] = x1611;
    element_matrix[548] = x1618;
    element_matrix[549] = x1620;
    element_matrix[550] = jacobian_determinant * (x1494 * x61 + x1568);
    element_matrix[551] = x1622;
    element_matrix[552] = x597;
    element_matrix[553] = x652;
    element_matrix[554] = x791;
    element_matrix[555] = x852;
    element_matrix[556] = x907;
    element_matrix[557] = x925;
    element_matrix[558] = x956;
    element_matrix[559] = x980;
    element_matrix[560] = x1266;
    element_matrix[561] = x1299;
    element_matrix[562] = x1375;
    element_matrix[563] = x1407;
    element_matrix[564] = x1437;
    element_matrix[565] = x1448;
    element_matrix[566] = x1464;
    element_matrix[567] = x1476;
    element_matrix[568] = x1566;
    element_matrix[569] = x1582;
    element_matrix[570] = x1604;
    element_matrix[571] = x1612;
    element_matrix[572] = x1619;
    element_matrix[573] = x1621;
    element_matrix[574] = x1622;
    element_matrix[575] =
            jacobian_determinant *
            (x1489 + x42 * (x1513 + x1573 + x1576 + x1613 + x67) + x61 * (-x1488 - x1583 - x1605));
}

static SFEM_INLINE void hex8_linear_elasticity_matrix_coord_taylor_sym(
        const scalar_t mu,
        const scalar_t lambda,
        const scalar_t *const SFEM_RESTRICT adjugate,
        const scalar_t jacobian_determinant,
        const scalar_t *const SFEM_RESTRICT trial_g,
        const scalar_t *const SFEM_RESTRICT test_g,
        const scalar_t *const SFEM_RESTRICT trial_H,
        const scalar_t *const SFEM_RESTRICT test_H,
        const scalar_t *const SFEM_RESTRICT trial_diff3,
        const scalar_t *const SFEM_RESTRICT test_diff3,
        accumulator_t *const SFEM_RESTRICT element_matrix) {
    const scalar_t x0 = POW2(trial_diff3[0]);
    const scalar_t x1 = POW2(adjugate[3]);
    const scalar_t x2 = lambda * x1;
    const scalar_t x3 = x0 * x2;
    const scalar_t x4 = POW2(trial_diff3[1]);
    const scalar_t x5 = POW2(adjugate[6]);
    const scalar_t x6 = lambda * x5;
    const scalar_t x7 = x4 * x6;
    const scalar_t x8 = POW2(trial_diff3[3]);
    const scalar_t x9 = x6 * x8;
    const scalar_t x10 = POW2(trial_diff3[5]);
    const scalar_t x11 = x10 * x2;
    const scalar_t x12 = mu * x0;
    const scalar_t x13 = x1 * x12;
    const scalar_t x14 = 2 * x13;
    const scalar_t x15 = mu * x4;
    const scalar_t x16 = x15 * x5;
    const scalar_t x17 = 2 * x16;
    const scalar_t x18 = mu * x8;
    const scalar_t x19 = x18 * x5;
    const scalar_t x20 = 2 * x19;
    const scalar_t x21 = mu * x10;
    const scalar_t x22 = x1 * x21;
    const scalar_t x23 = 2 * x22;
    const scalar_t x24 = POW2(trial_H[0]);
    const scalar_t x25 = 48 * x24;
    const scalar_t x26 = x2 * x25;
    const scalar_t x27 = POW2(trial_H[1]);
    const scalar_t x28 = 48 * x27;
    const scalar_t x29 = x28 * x6;
    const scalar_t x30 = 96 * mu;
    const scalar_t x31 = x24 * x30;
    const scalar_t x32 = x1 * x31;
    const scalar_t x33 = x27 * x30;
    const scalar_t x34 = x33 * x5;
    const scalar_t x35 = trial_diff3[0] * trial_diff3[5];
    const scalar_t x36 = 2 * x35;
    const scalar_t x37 = x2 * x36;
    const scalar_t x38 = trial_diff3[1] * trial_diff3[3];
    const scalar_t x39 = 2 * x38;
    const scalar_t x40 = x39 * x6;
    const scalar_t x41 = 4 * mu;
    const scalar_t x42 = x1 * x35;
    const scalar_t x43 = x41 * x42;
    const scalar_t x44 = x38 * x5;
    const scalar_t x45 = x41 * x44;
    const scalar_t x46 = 96 * lambda;
    const scalar_t x47 = trial_H[0] * trial_H[1];
    const scalar_t x48 = adjugate[3] * adjugate[6];
    const scalar_t x49 = x47 * x48;
    const scalar_t x50 = x46 * x49;
    const scalar_t x51 = 192 * mu;
    const scalar_t x52 = x47 * x51;
    const scalar_t x53 = x48 * x52;
    const scalar_t x54 = POW2(adjugate[5]);
    const scalar_t x55 = x12 * x54;
    const scalar_t x56 = POW2(adjugate[8]);
    const scalar_t x57 = x15 * x56;
    const scalar_t x58 = x18 * x56;
    const scalar_t x59 = x21 * x54;
    const scalar_t x60 = mu * x25;
    const scalar_t x61 = x54 * x60;
    const scalar_t x62 = mu * x28;
    const scalar_t x63 = x56 * x62;
    const scalar_t x64 = mu * x36;
    const scalar_t x65 = x54 * x64;
    const scalar_t x66 = mu * x39;
    const scalar_t x67 = x56 * x66;
    const scalar_t x68 = x30 * x47;
    const scalar_t x69 = adjugate[5] * adjugate[8];
    const scalar_t x70 = x68 * x69;
    const scalar_t x71 = x55 + x57 + x58 + x59 + x61 + x63 + x65 + x67 + x70;
    const scalar_t x72 = POW2(adjugate[4]);
    const scalar_t x73 = x12 * x72;
    const scalar_t x74 = POW2(adjugate[7]);
    const scalar_t x75 = x15 * x74;
    const scalar_t x76 = x18 * x74;
    const scalar_t x77 = x21 * x72;
    const scalar_t x78 = x60 * x72;
    const scalar_t x79 = x62 * x74;
    const scalar_t x80 = x64 * x72;
    const scalar_t x81 = x66 * x74;
    const scalar_t x82 = adjugate[4] * adjugate[7];
    const scalar_t x83 = x68 * x82;
    const scalar_t x84 = x73 + x75 + x76 + x77 + x78 + x79 + x80 + x81 + x83;
    const scalar_t x85 = 1.0 / jacobian_determinant;
    const scalar_t x86 = (1.0 / 144.0) * x85;
    const scalar_t x87 = trial_H[2] * trial_diff3[0];
    const scalar_t x88 = 4 * x2;
    const scalar_t x89 = trial_H[2] * trial_diff3[1];
    const scalar_t x90 = 4 * x6;
    const scalar_t x91 = trial_H[2] * trial_diff3[3];
    const scalar_t x92 = trial_H[2] * trial_diff3[5];
    const scalar_t x93 = 8 * mu;
    const scalar_t x94 = x1 * x93;
    const scalar_t x95 = x5 * x93;
    const scalar_t x96 = trial_H[0] * trial_g[1];
    const scalar_t x97 = x2 * x96;
    const scalar_t x98 = trial_H[1] * trial_g[2];
    const scalar_t x99 = x6 * x98;
    const scalar_t x100 = x1 * x96;
    const scalar_t x101 = x5 * x98;
    const scalar_t x102 = 4 * lambda;
    const scalar_t x103 = trial_H[0] * trial_diff3[1];
    const scalar_t x104 = adjugate[0] * adjugate[6];
    const scalar_t x105 = x103 * x104;
    const scalar_t x106 = trial_H[0] * trial_diff3[3];
    const scalar_t x107 = x104 * x106;
    const scalar_t x108 = adjugate[0] * adjugate[3];
    const scalar_t x109 = trial_H[1] * trial_diff3[0];
    const scalar_t x110 = x102 * x109;
    const scalar_t x111 = trial_H[1] * trial_diff3[5];
    const scalar_t x112 = x108 * x111;
    const scalar_t x113 = x108 * x109;
    const scalar_t x114 = trial_H[0] * trial_g[0];
    const scalar_t x115 = x108 * x46;
    const scalar_t x116 = trial_H[0] * trial_g[2];
    const scalar_t x117 = x46 * x48;
    const scalar_t x118 = trial_H[1] * trial_g[0];
    const scalar_t x119 = x104 * x46;
    const scalar_t x120 = trial_H[1] * trial_g[1];
    const scalar_t x121 = x108 * x114;
    const scalar_t x122 = x48 * x51;
    const scalar_t x123 = x104 * x118;
    const scalar_t x124 = x41 * x87;
    const scalar_t x125 = x41 * x89;
    const scalar_t x126 = x41 * x91;
    const scalar_t x127 = x41 * x92;
    const scalar_t x128 = x30 * x96;
    const scalar_t x129 = x30 * x98;
    const scalar_t x130 = x103 * x41;
    const scalar_t x131 = adjugate[2] * adjugate[8];
    const scalar_t x132 = x106 * x41;
    const scalar_t x133 = x109 * x41;
    const scalar_t x134 = adjugate[2] * adjugate[5];
    const scalar_t x135 = x111 * x41;
    const scalar_t x136 = x114 * x30;
    const scalar_t x137 = x116 * x30;
    const scalar_t x138 = x118 * x30;
    const scalar_t x139 = x120 * x30;
    const scalar_t x140 = x124 * x54 + x125 * x56 + x126 * x56 + x127 * x54 + x128 * x54 +
                          x129 * x56 + x130 * x131 + x131 * x132 + x131 * x138 + x133 * x134 +
                          x134 * x135 + x134 * x136 + x137 * x69 + x139 * x69 - x55 - x57 - x58 -
                          x59 - x61 - x63 - x65 - x67 - x70;
    const scalar_t x141 = adjugate[1] * adjugate[7];
    const scalar_t x142 = adjugate[1] * adjugate[4];
    const scalar_t x143 = x124 * x72 + x125 * x74 + x126 * x74 + x127 * x72 + x128 * x72 +
                          x129 * x74 + x130 * x141 + x132 * x141 + x133 * x142 + x135 * x142 +
                          x136 * x142 + x137 * x82 + x138 * x141 + x139 * x82 - x73 - x75 - x76 -
                          x77 - x78 - x79 - x80 - x81 - x83;
    const scalar_t x144 = (1.0 / 96.0) * x85;
    const scalar_t x145 = POW2(trial_diff3[2]);
    const scalar_t x146 = POW2(adjugate[0]);
    const scalar_t x147 = lambda * x146;
    const scalar_t x148 = POW2(trial_diff3[4]);
    const scalar_t x149 = mu * x145;
    const scalar_t x150 = x146 * x149;
    const scalar_t x151 = mu * x148;
    const scalar_t x152 = x146 * x151;
    const scalar_t x153 = POW2(trial_H[2]);
    const scalar_t x154 = 48 * x153;
    const scalar_t x155 = x153 * x30;
    const scalar_t x156 = 144 * x24;
    const scalar_t x157 = 144 * x27;
    const scalar_t x158 = 288 * mu;
    const scalar_t x159 = x158 * x24;
    const scalar_t x160 = x158 * x27;
    const scalar_t x161 = POW2(trial_g[0]);
    const scalar_t x162 = 576 * x161;
    const scalar_t x163 = POW2(trial_g[1]);
    const scalar_t x164 = 576 * x163;
    const scalar_t x165 = POW2(trial_g[2]);
    const scalar_t x166 = 576 * x165;
    const scalar_t x167 = 1152 * mu;
    const scalar_t x168 = x161 * x167;
    const scalar_t x169 = x163 * x167;
    const scalar_t x170 = x165 * x167;
    const scalar_t x171 = 48 * mu;
    const scalar_t x172 = x1 * x171;
    const scalar_t x173 = x171 * x5;
    const scalar_t x174 = 24 * x87;
    const scalar_t x175 = 24 * x89;
    const scalar_t x176 = 24 * x91;
    const scalar_t x177 = 24 * x92;
    const scalar_t x178 = trial_diff3[2] * trial_diff3[4];
    const scalar_t x179 = 2 * x178;
    const scalar_t x180 = x178 * x41;
    const scalar_t x181 = 6 * x35;
    const scalar_t x182 = 6 * x38;
    const scalar_t x183 = 12 * mu;
    const scalar_t x184 = x167 * x48;
    const scalar_t x185 = 576 * lambda;
    const scalar_t x186 = x185 * x48;
    const scalar_t x187 = 24 * lambda;
    const scalar_t x188 = trial_H[0] * trial_H[2];
    const scalar_t x189 = trial_H[1] * trial_H[2];
    const scalar_t x190 = x188 * x51;
    const scalar_t x191 = x189 * x51;
    const scalar_t x192 = 288 * lambda;
    const scalar_t x193 = 576 * mu;
    const scalar_t x194 = 1152 * lambda;
    const scalar_t x195 = trial_g[0] * trial_g[1];
    const scalar_t x196 = x108 * x195;
    const scalar_t x197 = trial_g[0] * trial_g[2];
    const scalar_t x198 = x104 * x197;
    const scalar_t x199 = trial_g[1] * trial_g[2];
    const scalar_t x200 = x199 * x48;
    const scalar_t x201 = 2304 * mu;
    const scalar_t x202 = POW2(adjugate[2]);
    const scalar_t x203 = x149 * x202;
    const scalar_t x204 = x151 * x202;
    const scalar_t x205 = mu * x154;
    const scalar_t x206 = mu * x156;
    const scalar_t x207 = mu * x157;
    const scalar_t x208 = mu * x162;
    const scalar_t x209 = mu * x164;
    const scalar_t x210 = mu * x166;
    const scalar_t x211 = x54 * x96;
    const scalar_t x212 = x193 * x98;
    const scalar_t x213 = mu * x174;
    const scalar_t x214 = mu * x175;
    const scalar_t x215 = mu * x176;
    const scalar_t x216 = mu * x177;
    const scalar_t x217 = mu * x179;
    const scalar_t x218 = mu * x181;
    const scalar_t x219 = mu * x182;
    const scalar_t x220 = x114 * x193;
    const scalar_t x221 = x116 * x193;
    const scalar_t x222 = x118 * x193;
    const scalar_t x223 = x120 * x193;
    const scalar_t x224 = 24 * mu;
    const scalar_t x225 = x131 * x224;
    const scalar_t x226 = x109 * x224;
    const scalar_t x227 = x111 * x224;
    const scalar_t x228 = x188 * x30;
    const scalar_t x229 = x189 * x30;
    const scalar_t x230 = x158 * x47;
    const scalar_t x231 = x167 * x195;
    const scalar_t x232 = x167 * x197;
    const scalar_t x233 = x167 * x199;
    const scalar_t x234 = -x103 * x225 - x106 * x225 - x131 * x222 + x131 * x228 + x131 * x232 -
                          x134 * x220 - x134 * x226 - x134 * x227 + x134 * x229 + x134 * x231 -
                          x193 * x211 + x202 * x208 + x202 * x217 + x202 * x60 + x202 * x62 + x203 +
                          x204 + x205 * x54 + x205 * x56 + x206 * x54 + x207 * x56 + x209 * x54 +
                          x210 * x56 - x212 * x56 - x213 * x54 - x214 * x56 - x215 * x56 -
                          x216 * x54 + x218 * x54 + x219 * x56 - x221 * x69 - x223 * x69 +
                          x230 * x69 + x233 * x69 + 3 * x55 + 3 * x57 + 3 * x58 + 3 * x59;
    const scalar_t x235 = POW2(adjugate[1]);
    const scalar_t x236 = x149 * x235;
    const scalar_t x237 = x151 * x235;
    const scalar_t x238 = x72 * x96;
    const scalar_t x239 = x103 * x141;
    const scalar_t x240 = x106 * x141;
    const scalar_t x241 =
            -x141 * x222 + x141 * x228 + x141 * x232 - x142 * x220 - x142 * x226 - x142 * x227 +
            x142 * x229 + x142 * x231 - x193 * x238 + x205 * x72 + x205 * x74 + x206 * x72 +
            x207 * x74 + x208 * x235 + x209 * x72 + x210 * x74 - x212 * x74 - x213 * x72 -
            x214 * x74 - x215 * x74 - x216 * x72 + x217 * x235 + x218 * x72 + x219 * x74 -
            x221 * x82 - x223 * x82 - x224 * x239 - x224 * x240 + x230 * x82 + x233 * x82 +
            x235 * x60 + x235 * x62 + x236 + x237 + 3 * x73 + 3 * x75 + 3 * x76 + 3 * x77;
    const scalar_t x242 = (1.0 / 576.0) * x85;
    const scalar_t x243 = adjugate[3] * adjugate[4];
    const scalar_t x244 = lambda * x243;
    const scalar_t x245 = x0 * x244;
    const scalar_t x246 = x10 * x244;
    const scalar_t x247 = x12 * x243;
    const scalar_t x248 = x21 * x243;
    const scalar_t x249 = adjugate[6] * adjugate[7];
    const scalar_t x250 = lambda * x249;
    const scalar_t x251 = x250 * x4;
    const scalar_t x252 = x250 * x8;
    const scalar_t x253 = x15 * x249;
    const scalar_t x254 = x18 * x249;
    const scalar_t x255 = x244 * x36;
    const scalar_t x256 = x243 * x64;
    const scalar_t x257 = adjugate[3] * x47;
    const scalar_t x258 = adjugate[7] * lambda;
    const scalar_t x259 = 48 * x258;
    const scalar_t x260 = x257 * x259;
    const scalar_t x261 = adjugate[7] * x171;
    const scalar_t x262 = x257 * x261;
    const scalar_t x263 = adjugate[4] * x47;
    const scalar_t x264 = adjugate[6] * lambda;
    const scalar_t x265 = 48 * x264;
    const scalar_t x266 = x263 * x265;
    const scalar_t x267 = adjugate[6] * x171;
    const scalar_t x268 = x263 * x267;
    const scalar_t x269 = x250 * x39;
    const scalar_t x270 = x249 * x66;
    const scalar_t x271 = x244 * x25;
    const scalar_t x272 = x243 * x60;
    const scalar_t x273 = x250 * x28;
    const scalar_t x274 = x249 * x62;
    const scalar_t x275 = adjugate[0] * adjugate[4];
    const scalar_t x276 = x114 * x275;
    const scalar_t x277 = 48 * lambda;
    const scalar_t x278 = 2 * lambda;
    const scalar_t x279 = x109 * x275;
    const scalar_t x280 = x111 * x275;
    const scalar_t x281 = 2 * mu;
    const scalar_t x282 = adjugate[0] * x103;
    const scalar_t x283 = 2 * x258;
    const scalar_t x284 = adjugate[0] * x106;
    const scalar_t x285 = adjugate[0] * x118;
    const scalar_t x286 = adjugate[7] * x281;
    const scalar_t x287 = adjugate[1] * adjugate[3];
    const scalar_t x288 = x114 * x287;
    const scalar_t x289 = x109 * x278;
    const scalar_t x290 = x111 * x287;
    const scalar_t x291 = x281 * x287;
    const scalar_t x292 = adjugate[1] * x103;
    const scalar_t x293 = 2 * x264;
    const scalar_t x294 = adjugate[1] * x106;
    const scalar_t x295 = adjugate[1] * x118;
    const scalar_t x296 = adjugate[6] * x281;
    const scalar_t x297 = x244 * x96;
    const scalar_t x298 = 4 * x244;
    const scalar_t x299 = adjugate[3] * x116;
    const scalar_t x300 = adjugate[3] * x120;
    const scalar_t x301 = adjugate[4] * x116;
    const scalar_t x302 = adjugate[4] * x120;
    const scalar_t x303 = x250 * x98;
    const scalar_t x304 = 4 * x250;
    const scalar_t x305 = adjugate[0] * adjugate[1];
    const scalar_t x306 = lambda * x305;
    const scalar_t x307 = x189 * x275;
    const scalar_t x308 = 12 * lambda;
    const scalar_t x309 = x195 * x275;
    const scalar_t x310 = adjugate[0] * x188;
    const scalar_t x311 = 12 * x258;
    const scalar_t x312 = adjugate[7] * x285;
    const scalar_t x313 = x185 * x197;
    const scalar_t x314 = adjugate[0] * adjugate[7];
    const scalar_t x315 = adjugate[7] * x183;
    const scalar_t x316 = x193 * x197;
    const scalar_t x317 = x189 * x287;
    const scalar_t x318 = x109 * x287;
    const scalar_t x319 = x195 * x287;
    const scalar_t x320 = adjugate[1] * x188;
    const scalar_t x321 = 12 * x264;
    const scalar_t x322 = adjugate[6] * x295;
    const scalar_t x323 = adjugate[1] * adjugate[6];
    const scalar_t x324 = adjugate[6] * x183;
    const scalar_t x325 = x193 * x96;
    const scalar_t x326 = 144 * x257;
    const scalar_t x327 = adjugate[7] * x299;
    const scalar_t x328 = adjugate[7] * x300;
    const scalar_t x329 = x185 * x199;
    const scalar_t x330 = adjugate[3] * adjugate[7];
    const scalar_t x331 = mu * x326;
    const scalar_t x332 = x193 * x199;
    const scalar_t x333 = 144 * x263;
    const scalar_t x334 = adjugate[6] * x301;
    const scalar_t x335 = adjugate[6] * x302;
    const scalar_t x336 = adjugate[4] * adjugate[6];
    const scalar_t x337 = adjugate[6] * mu;
    const scalar_t x338 = adjugate[3] * adjugate[5];
    const scalar_t x339 = lambda * x338;
    const scalar_t x340 = x0 * x339;
    const scalar_t x341 = x10 * x339;
    const scalar_t x342 = x12 * x338;
    const scalar_t x343 = x21 * x338;
    const scalar_t x344 = adjugate[6] * adjugate[8];
    const scalar_t x345 = lambda * x344;
    const scalar_t x346 = x345 * x4;
    const scalar_t x347 = x345 * x8;
    const scalar_t x348 = x15 * x344;
    const scalar_t x349 = x18 * x344;
    const scalar_t x350 = x339 * x36;
    const scalar_t x351 = x338 * x64;
    const scalar_t x352 = adjugate[8] * x257;
    const scalar_t x353 = x277 * x352;
    const scalar_t x354 = x171 * x352;
    const scalar_t x355 = adjugate[5] * x47;
    const scalar_t x356 = x265 * x355;
    const scalar_t x357 = x267 * x355;
    const scalar_t x358 = x345 * x39;
    const scalar_t x359 = x344 * x66;
    const scalar_t x360 = x25 * x339;
    const scalar_t x361 = x338 * x60;
    const scalar_t x362 = x28 * x345;
    const scalar_t x363 = x344 * x62;
    const scalar_t x364 = adjugate[0] * adjugate[5];
    const scalar_t x365 = x114 * x364;
    const scalar_t x366 = x111 * x364;
    const scalar_t x367 = x109 * x281;
    const scalar_t x368 = adjugate[8] * x282;
    const scalar_t x369 = adjugate[8] * x284;
    const scalar_t x370 = adjugate[8] * x285;
    const scalar_t x371 = adjugate[2] * adjugate[3];
    const scalar_t x372 = x114 * x371;
    const scalar_t x373 = x111 * x371;
    const scalar_t x374 = adjugate[2] * x103;
    const scalar_t x375 = adjugate[2] * x106;
    const scalar_t x376 = adjugate[2] * x118;
    const scalar_t x377 = x339 * x96;
    const scalar_t x378 = 4 * x339;
    const scalar_t x379 = adjugate[8] * x299;
    const scalar_t x380 = adjugate[8] * x300;
    const scalar_t x381 = adjugate[5] * x116;
    const scalar_t x382 = adjugate[5] * x120;
    const scalar_t x383 = x345 * x98;
    const scalar_t x384 = 4 * x345;
    const scalar_t x385 = adjugate[0] * adjugate[2];
    const scalar_t x386 = lambda * x385;
    const scalar_t x387 = x189 * x364;
    const scalar_t x388 = x109 * x364;
    const scalar_t x389 = x195 * x364;
    const scalar_t x390 = adjugate[8] * x310;
    const scalar_t x391 = adjugate[0] * adjugate[8];
    const scalar_t x392 = x189 * x371;
    const scalar_t x393 = x109 * x371;
    const scalar_t x394 = x195 * x371;
    const scalar_t x395 = adjugate[2] * x188;
    const scalar_t x396 = adjugate[6] * x376;
    const scalar_t x397 = adjugate[2] * adjugate[6];
    const scalar_t x398 = adjugate[8] * lambda;
    const scalar_t x399 = adjugate[3] * adjugate[8];
    const scalar_t x400 = 144 * x355;
    const scalar_t x401 = adjugate[6] * x381;
    const scalar_t x402 = adjugate[6] * x382;
    const scalar_t x403 = adjugate[5] * adjugate[6];
    const scalar_t x404 = lambda * x72;
    const scalar_t x405 = x0 * x404;
    const scalar_t x406 = lambda * x74;
    const scalar_t x407 = x4 * x406;
    const scalar_t x408 = x406 * x8;
    const scalar_t x409 = x10 * x404;
    const scalar_t x410 = 2 * x73;
    const scalar_t x411 = 2 * x75;
    const scalar_t x412 = 2 * x76;
    const scalar_t x413 = 2 * x77;
    const scalar_t x414 = x25 * x404;
    const scalar_t x415 = x28 * x406;
    const scalar_t x416 = x31 * x72;
    const scalar_t x417 = x33 * x74;
    const scalar_t x418 = x36 * x404;
    const scalar_t x419 = x39 * x406;
    const scalar_t x420 = x35 * x72;
    const scalar_t x421 = x41 * x420;
    const scalar_t x422 = x38 * x74;
    const scalar_t x423 = x41 * x422;
    const scalar_t x424 = x46 * x82;
    const scalar_t x425 = x424 * x47;
    const scalar_t x426 = x52 * x82;
    const scalar_t x427 = x1 * x60;
    const scalar_t x428 = x5 * x62;
    const scalar_t x429 = x281 * x42;
    const scalar_t x430 = x281 * x44;
    const scalar_t x431 = x30 * x49;
    const scalar_t x432 = x13 + x16 + x19 + x22 + x427 + x428 + x429 + x430 + x431;
    const scalar_t x433 = 4 * x404;
    const scalar_t x434 = 4 * x406;
    const scalar_t x435 = x72 * x93;
    const scalar_t x436 = x74 * x93;
    const scalar_t x437 = x404 * x96;
    const scalar_t x438 = x406 * x98;
    const scalar_t x439 = x74 * x98;
    const scalar_t x440 = x111 * x142;
    const scalar_t x441 = x109 * x142;
    const scalar_t x442 = x114 * x142;
    const scalar_t x443 = x118 * x141;
    const scalar_t x444 = x51 * x82;
    const scalar_t x445 = x1 * x124 + x1 * x127 + x1 * x128 + x104 * x138 + x105 * x41 +
                          x107 * x41 + x108 * x133 + x108 * x136 + x112 * x41 + x125 * x5 +
                          x126 * x5 + x129 * x5 - x13 + x137 * x48 + x139 * x48 - x16 - x19 - x22 -
                          x427 - x428 - x429 - x430 - x431;
    const scalar_t x446 = lambda * x235;
    const scalar_t x447 = x171 * x72;
    const scalar_t x448 = x171 * x74;
    const scalar_t x449 = x167 * x82;
    const scalar_t x450 = x185 * x82;
    const scalar_t x451 = x188 * x46;
    const scalar_t x452 = x189 * x46;
    const scalar_t x453 = x47 * x82;
    const scalar_t x454 = x142 * x195;
    const scalar_t x455 = x141 * x197;
    const scalar_t x456 = x199 * x82;
    const scalar_t x457 = 6 * mu;
    const scalar_t x458 = x1 * x205 + x1 * x206 + x1 * x209 - x1 * x213 - x1 * x216 - x100 * x193 -
                          x101 * x193 + x104 * x228 - x105 * x224 - x107 * x224 + x108 * x229 -
                          x112 * x224 - x113 * x224 - x121 * x193 - x123 * x193 + 3 * x13 +
                          x146 * x208 + x146 * x217 + x146 * x60 + x146 * x62 + x150 + x152 +
                          x158 * x49 + 3 * x16 + x167 * x196 + x167 * x198 + x184 * x199 + 3 * x19 +
                          x205 * x5 + x207 * x5 + x210 * x5 - x214 * x5 - x215 * x5 + 3 * x22 -
                          x221 * x48 - x223 * x48 + x42 * x457 + x44 * x457;
    const scalar_t x459 = adjugate[4] * adjugate[5];
    const scalar_t x460 = lambda * x459;
    const scalar_t x461 = x0 * x460;
    const scalar_t x462 = x10 * x460;
    const scalar_t x463 = x12 * x459;
    const scalar_t x464 = x21 * x459;
    const scalar_t x465 = adjugate[7] * adjugate[8];
    const scalar_t x466 = lambda * x465;
    const scalar_t x467 = x4 * x466;
    const scalar_t x468 = x466 * x8;
    const scalar_t x469 = x15 * x465;
    const scalar_t x470 = x18 * x465;
    const scalar_t x471 = x36 * x460;
    const scalar_t x472 = x459 * x64;
    const scalar_t x473 = adjugate[8] * x263;
    const scalar_t x474 = x277 * x473;
    const scalar_t x475 = x171 * x473;
    const scalar_t x476 = x259 * x355;
    const scalar_t x477 = x261 * x355;
    const scalar_t x478 = x39 * x466;
    const scalar_t x479 = x465 * x66;
    const scalar_t x480 = x25 * x460;
    const scalar_t x481 = x459 * x60;
    const scalar_t x482 = x28 * x466;
    const scalar_t x483 = x465 * x62;
    const scalar_t x484 = adjugate[1] * adjugate[5];
    const scalar_t x485 = x114 * x484;
    const scalar_t x486 = x111 * x484;
    const scalar_t x487 = adjugate[8] * x292;
    const scalar_t x488 = adjugate[8] * x294;
    const scalar_t x489 = adjugate[8] * x295;
    const scalar_t x490 = adjugate[2] * adjugate[4];
    const scalar_t x491 = x114 * x490;
    const scalar_t x492 = x111 * x490;
    const scalar_t x493 = x460 * x96;
    const scalar_t x494 = 4 * x460;
    const scalar_t x495 = adjugate[8] * x301;
    const scalar_t x496 = adjugate[8] * x302;
    const scalar_t x497 = x466 * x98;
    const scalar_t x498 = 4 * x466;
    const scalar_t x499 = adjugate[1] * adjugate[2];
    const scalar_t x500 = lambda * x499;
    const scalar_t x501 = x189 * x484;
    const scalar_t x502 = x109 * x484;
    const scalar_t x503 = x195 * x484;
    const scalar_t x504 = adjugate[8] * x320;
    const scalar_t x505 = adjugate[1] * adjugate[8];
    const scalar_t x506 = x189 * x490;
    const scalar_t x507 = x109 * x490;
    const scalar_t x508 = x195 * x490;
    const scalar_t x509 = adjugate[7] * x376;
    const scalar_t x510 = adjugate[2] * adjugate[7];
    const scalar_t x511 = adjugate[4] * adjugate[8];
    const scalar_t x512 = adjugate[7] * x381;
    const scalar_t x513 = adjugate[7] * x382;
    const scalar_t x514 = adjugate[5] * adjugate[7];
    const scalar_t x515 = lambda * x54;
    const scalar_t x516 = x0 * x515;
    const scalar_t x517 = lambda * x56;
    const scalar_t x518 = x4 * x517;
    const scalar_t x519 = x517 * x8;
    const scalar_t x520 = x10 * x515;
    const scalar_t x521 = 2 * x55;
    const scalar_t x522 = 2 * x57;
    const scalar_t x523 = 2 * x58;
    const scalar_t x524 = 2 * x59;
    const scalar_t x525 = x25 * x515;
    const scalar_t x526 = x28 * x517;
    const scalar_t x527 = x31 * x54;
    const scalar_t x528 = x33 * x56;
    const scalar_t x529 = x36 * x515;
    const scalar_t x530 = x39 * x517;
    const scalar_t x531 = x35 * x54;
    const scalar_t x532 = x41 * x531;
    const scalar_t x533 = x38 * x56;
    const scalar_t x534 = x41 * x533;
    const scalar_t x535 = x46 * x69;
    const scalar_t x536 = x47 * x535;
    const scalar_t x537 = x52 * x69;
    const scalar_t x538 = 4 * x515;
    const scalar_t x539 = 4 * x517;
    const scalar_t x540 = x54 * x93;
    const scalar_t x541 = x56 * x93;
    const scalar_t x542 = x515 * x96;
    const scalar_t x543 = x517 * x98;
    const scalar_t x544 = x56 * x98;
    const scalar_t x545 = x102 * x131;
    const scalar_t x546 = x111 * x134;
    const scalar_t x547 = x131 * x93;
    const scalar_t x548 = x109 * x134;
    const scalar_t x549 = x114 * x134;
    const scalar_t x550 = x118 * x131;
    const scalar_t x551 = x51 * x69;
    const scalar_t x552 = lambda * x202;
    const scalar_t x553 = x171 * x54;
    const scalar_t x554 = x171 * x56;
    const scalar_t x555 = x167 * x69;
    const scalar_t x556 = x185 * x69;
    const scalar_t x557 = x131 * x171;
    const scalar_t x558 = x131 * x187;
    const scalar_t x559 = x47 * x69;
    const scalar_t x560 = x134 * x195;
    const scalar_t x561 = x131 * x197;
    const scalar_t x562 = x199 * x69;
    element_matrix[0] =
            x144 * (x100 * x51 + x101 * x51 + x102 * x105 + x102 * x107 + x102 * x112 + x105 * x93 +
                    x107 * x93 + x108 * x110 - x11 + x112 * x93 + x113 * x93 + x114 * x115 +
                    x116 * x117 + x116 * x122 + x117 * x120 + x118 * x119 + x120 * x122 +
                    x121 * x51 + x123 * x51 - x14 + x140 + x143 - x17 - x20 - x23 - x26 - x29 - x3 -
                    x32 - x34 - x37 - x40 - x43 - x45 - x50 - x53 - x7 + x87 * x88 + x87 * x94 +
                    x88 * x92 + x89 * x90 + x89 * x95 - x9 + x90 * x91 + x91 * x95 + x92 * x94 +
                    96 * x97 + 96 * x99) +
            x242 * (x1 * x155 + x1 * x159 + x1 * x169 - x100 * x167 - x101 * x167 + x104 * x190 -
                    x105 * x171 - x105 * x187 - x107 * x171 - x107 * x187 + x108 * x191 + 3 * x11 -
                    x112 * x171 - x112 * x187 - x113 * x171 - x113 * x187 + x115 * x189 -
                    x116 * x184 - x116 * x186 + x119 * x188 - x120 * x184 - x120 * x186 -
                    x121 * x167 - x121 * x185 - x123 * x167 - x123 * x185 + 6 * x13 + x145 * x147 +
                    x146 * x168 + x146 * x180 + x146 * x31 + x146 * x33 + x147 * x148 +
                    x147 * x162 + x147 * x179 + x147 * x25 + x147 * x28 + 2 * x150 + 2 * x152 +
                    x154 * x2 + x154 * x6 + x155 * x5 + x156 * x2 + x157 * x6 + 6 * x16 +
                    x160 * x5 + x164 * x2 + x166 * x6 + x170 * x5 - x172 * x87 - x172 * x92 -
                    x173 * x89 - x173 * x91 - x174 * x2 - x175 * x6 - x176 * x6 - x177 * x2 +
                    x181 * x2 + x182 * x6 + x183 * x42 + x183 * x44 + 6 * x19 + x192 * x49 +
                    x193 * x49 + x194 * x196 + x194 * x198 + x194 * x200 + x196 * x201 +
                    x198 * x201 + x200 * x201 + 6 * x22 + x234 + x241 + 3 * x3 + 3 * x7 + 3 * x9 -
                    576 * x97 - 576 * x99) +
            x86 * (x11 + x14 + x17 + x20 + x23 + x26 + x29 + x3 + x32 + x34 + x37 + x40 + x43 +
                   x45 + x50 + x53 + x7 + x71 + x84 + x9);
    element_matrix[1] =
            x144 * (x109 * x291 + x111 * x291 + x124 * x243 + x125 * x249 + x126 * x249 +
                    x127 * x243 + x128 * x243 + x129 * x249 + x171 * x276 + x171 * x288 - x245 -
                    x246 - x247 - x248 - x251 - x252 - x253 - x254 - x255 - x256 + x259 * x285 +
                    x259 * x299 + x259 * x300 - x260 + x261 * x285 + x261 * x299 + x261 * x300 -
                    x262 + x265 * x295 + x265 * x301 + x265 * x302 - x266 + x267 * x295 +
                    x267 * x301 + x267 * x302 - x268 - x269 - x270 - x271 - x272 - x273 - x274 +
                    x276 * x277 + x277 * x288 + x278 * x279 + x278 * x280 + x278 * x290 +
                    x279 * x281 + x280 * x281 + x282 * x283 + x282 * x286 + x283 * x284 +
                    x284 * x286 + x287 * x289 + x292 * x293 + x292 * x296 + x293 * x294 +
                    x294 * x296 + 96 * x297 + x298 * x87 + x298 * x92 + 96 * x303 + x304 * x89 +
                    x304 * x91) +
            x242 * (adjugate[7] * x331 + x145 * x306 + x148 * x306 + x149 * x305 + x151 * x305 +
                    x154 * x244 + x154 * x250 + x156 * x244 + x157 * x250 - x158 * x276 -
                    x158 * x288 - x158 * x312 - x158 * x322 - x158 * x327 - x158 * x328 -
                    x158 * x334 - x158 * x335 + x162 * x306 + x164 * x244 + x166 * x250 +
                    x171 * x307 + x171 * x317 - x174 * x244 - x175 * x250 - x176 * x250 -
                    x177 * x244 + x179 * x306 + x181 * x244 + x182 * x250 - x183 * x279 -
                    x183 * x280 - x183 * x290 - x183 * x318 + x185 * x309 + x185 * x319 -
                    x192 * x276 - x192 * x288 - x192 * x312 - x192 * x322 - x192 * x327 -
                    x192 * x328 - x192 * x334 - x192 * x335 + x193 * x309 + x193 * x319 +
                    x205 * x243 + x205 * x249 + x206 * x243 + x207 * x249 + x208 * x305 +
                    x209 * x243 + x210 * x249 - x212 * x249 - x213 * x243 - x214 * x249 -
                    x215 * x249 - x216 * x243 + x217 * x305 + x218 * x243 + x219 * x249 -
                    x243 * x325 + 3 * x245 + 3 * x246 + 3 * x247 + 3 * x248 + x25 * x306 +
                    3 * x251 + 3 * x252 + 3 * x253 + 3 * x254 + x258 * x326 + x259 * x310 +
                    x261 * x310 + x264 * x333 + x265 * x320 + x267 * x320 + x277 * x307 +
                    x277 * x317 - x279 * x308 + x28 * x306 - x280 * x308 - x282 * x311 -
                    x282 * x315 - x284 * x311 - x284 * x315 - x290 * x308 - x292 * x321 -
                    x292 * x324 - x294 * x321 - x294 * x324 - 576 * x297 - 576 * x303 + x305 * x60 +
                    x305 * x62 - x308 * x318 + x313 * x314 + x313 * x323 + x314 * x316 +
                    x316 * x323 + x329 * x330 + x329 * x336 + x330 * x332 + x332 * x336 +
                    x333 * x337) +
            x86 * (x245 + x246 + x247 + x248 + x251 + x252 + x253 + x254 + x255 + x256 + x260 +
                   x262 + x266 + x268 + x269 + x270 + x271 + x272 + x273 + x274);
    element_matrix[2] =
            x144 * (x124 * x338 + x125 * x344 + x126 * x344 + x127 * x338 + x128 * x338 +
                    x129 * x344 + x171 * x365 + x171 * x370 + x171 * x372 + x171 * x379 +
                    x171 * x380 + x265 * x376 + x265 * x381 + x265 * x382 + x267 * x376 +
                    x267 * x381 + x267 * x382 + x277 * x365 + x277 * x370 + x277 * x372 +
                    x277 * x379 + x277 * x380 + x278 * x366 + x278 * x368 + x278 * x369 +
                    x278 * x373 + x281 * x366 + x281 * x368 + x281 * x369 + x281 * x373 +
                    x289 * x364 + x289 * x371 + x293 * x374 + x293 * x375 + x296 * x374 +
                    x296 * x375 - x340 - x341 - x342 - x343 - x346 - x347 - x348 - x349 - x350 -
                    x351 - x353 - x354 - x356 - x357 - x358 - x359 - x360 - x361 - x362 - x363 +
                    x364 * x367 + x367 * x371 + 96 * x377 + x378 * x87 + x378 * x92 + 96 * x383 +
                    x384 * x89 + x384 * x91) +
            x242 * (adjugate[8] * x331 + x145 * x386 + x148 * x386 + x149 * x385 + x151 * x385 +
                    x154 * x339 + x154 * x345 + x156 * x339 + x157 * x345 - x158 * x365 -
                    x158 * x370 - x158 * x372 - x158 * x379 - x158 * x380 - x158 * x396 -
                    x158 * x401 - x158 * x402 + x162 * x386 + x164 * x339 + x166 * x345 +
                    x171 * x387 + x171 * x390 + x171 * x392 - x174 * x339 - x175 * x345 -
                    x176 * x345 - x177 * x339 + x179 * x386 + x181 * x339 + x182 * x345 -
                    x183 * x366 - x183 * x368 - x183 * x369 - x183 * x373 - x183 * x388 -
                    x183 * x393 + x185 * x389 + x185 * x394 - x192 * x365 - x192 * x370 -
                    x192 * x372 - x192 * x379 - x192 * x380 - x192 * x396 - x192 * x401 -
                    x192 * x402 + x193 * x389 + x193 * x394 + x205 * x338 + x205 * x344 +
                    x206 * x338 + x207 * x344 + x208 * x385 + x209 * x338 + x210 * x344 -
                    x212 * x344 - x213 * x338 - x214 * x344 - x215 * x344 - x216 * x338 +
                    x217 * x385 + x218 * x338 + x219 * x344 + x25 * x386 + x264 * x400 +
                    x265 * x395 + x267 * x395 + x277 * x387 + x277 * x390 + x277 * x392 +
                    x28 * x386 - x308 * x366 - x308 * x368 - x308 * x369 - x308 * x373 -
                    x308 * x388 - x308 * x393 + x313 * x391 + x313 * x397 + x316 * x391 +
                    x316 * x397 - x321 * x374 - x321 * x375 - x324 * x374 - x324 * x375 -
                    x325 * x338 + x326 * x398 + x329 * x399 + x329 * x403 + x332 * x399 +
                    x332 * x403 + x337 * x400 + 3 * x340 + 3 * x341 + 3 * x342 + 3 * x343 +
                    3 * x346 + 3 * x347 + 3 * x348 + 3 * x349 - 576 * x377 - 576 * x383 +
                    x385 * x60 + x385 * x62) +
            x86 * (x340 + x341 + x342 + x343 + x346 + x347 + x348 + x349 + x350 + x351 + x353 +
                   x354 + x356 + x357 + x358 + x359 + x360 + x361 + x362 + x363);
    element_matrix[3] =
            x144 * (x102 * x239 + x102 * x240 + x102 * x440 + x110 * x142 + x116 * x424 +
                    x116 * x444 + x120 * x424 + x120 * x444 + x140 + x238 * x51 + x239 * x93 +
                    x240 * x93 - x405 - x407 - x408 - x409 - x410 - x411 - x412 - x413 - x414 -
                    x415 - x416 - x417 - x418 - x419 - x421 - x423 - x425 - x426 + x433 * x87 +
                    x433 * x92 + x434 * x89 + x434 * x91 + x435 * x87 + x435 * x92 + x436 * x89 +
                    x436 * x91 + 96 * x437 + 96 * x438 + x439 * x51 + x440 * x93 + x441 * x93 +
                    x442 * x46 + x442 * x51 + x443 * x46 + x443 * x51 + x445) +
            x242 * (-x116 * x449 - x116 * x450 - x120 * x449 - x120 * x450 + x141 * x190 +
                    x141 * x451 + x142 * x191 + x142 * x452 + x145 * x446 + x148 * x446 +
                    x154 * x404 + x154 * x406 + x155 * x72 + x155 * x74 + x156 * x404 +
                    x157 * x406 + x159 * x72 + x160 * x74 + x162 * x446 + x164 * x404 +
                    x166 * x406 - x167 * x238 - x167 * x439 - x167 * x442 - x167 * x443 +
                    x168 * x235 + x169 * x72 + x170 * x74 - x171 * x239 - x171 * x240 -
                    x171 * x440 - x171 * x441 - x174 * x404 - x175 * x406 - x176 * x406 -
                    x177 * x404 + x179 * x446 + x180 * x235 + x181 * x404 + x182 * x406 +
                    x183 * x420 + x183 * x422 - x185 * x442 - x185 * x443 - x187 * x239 -
                    x187 * x240 - x187 * x440 - x187 * x441 + x192 * x453 + x193 * x453 +
                    x194 * x454 + x194 * x455 + x194 * x456 + x201 * x454 + x201 * x455 +
                    x201 * x456 + x234 + x235 * x31 + x235 * x33 + 2 * x236 + 2 * x237 +
                    x25 * x446 + x28 * x446 + 3 * x405 + 3 * x407 + 3 * x408 + 3 * x409 -
                    576 * x437 - 576 * x438 - x447 * x87 - x447 * x92 - x448 * x89 - x448 * x91 +
                    x458 + 6 * x73 + 6 * x75 + 6 * x76 + 6 * x77) +
            x86 * (x405 + x407 + x408 + x409 + x410 + x411 + x412 + x413 + x414 + x415 + x416 +
                   x417 + x418 + x419 + x421 + x423 + x425 + x426 + x432 + x71);
    element_matrix[4] =
            x144 * (x124 * x459 + x125 * x465 + x126 * x465 + x127 * x459 + x128 * x459 +
                    x129 * x465 + x171 * x485 + x171 * x489 + x171 * x491 + x171 * x495 +
                    x171 * x496 + x259 * x376 + x259 * x381 + x259 * x382 + x261 * x376 +
                    x261 * x381 + x261 * x382 + x277 * x485 + x277 * x489 + x277 * x491 +
                    x277 * x495 + x277 * x496 + x278 * x486 + x278 * x487 + x278 * x488 +
                    x278 * x492 + x281 * x486 + x281 * x487 + x281 * x488 + x281 * x492 +
                    x283 * x374 + x283 * x375 + x286 * x374 + x286 * x375 + x289 * x484 +
                    x289 * x490 + x367 * x484 + x367 * x490 - x461 - x462 - x463 - x464 - x467 -
                    x468 - x469 - x470 - x471 - x472 - x474 - x475 - x476 - x477 - x478 - x479 -
                    x480 - x481 - x482 - x483 + 96 * x493 + x494 * x87 + x494 * x92 + 96 * x497 +
                    x498 * x89 + x498 * x91) +
            x242 * (adjugate[7] * mu * x400 + adjugate[8] * mu * x333 + x145 * x500 + x148 * x500 +
                    x149 * x499 + x151 * x499 + x154 * x460 + x154 * x466 + x156 * x460 +
                    x157 * x466 - x158 * x485 - x158 * x489 - x158 * x491 - x158 * x495 -
                    x158 * x496 - x158 * x509 - x158 * x512 - x158 * x513 + x162 * x500 +
                    x164 * x460 + x166 * x466 + x171 * x501 + x171 * x504 + x171 * x506 -
                    x174 * x460 - x175 * x466 - x176 * x466 - x177 * x460 + x179 * x500 +
                    x181 * x460 + x182 * x466 - x183 * x486 - x183 * x487 - x183 * x488 -
                    x183 * x492 - x183 * x502 - x183 * x507 + x185 * x503 + x185 * x508 -
                    x192 * x485 - x192 * x489 - x192 * x491 - x192 * x495 - x192 * x496 -
                    x192 * x509 - x192 * x512 - x192 * x513 + x193 * x503 + x193 * x508 +
                    x205 * x459 + x205 * x465 + x206 * x459 + x207 * x465 + x208 * x499 +
                    x209 * x459 + x210 * x465 - x212 * x465 - x213 * x459 - x214 * x465 -
                    x215 * x465 - x216 * x459 + x217 * x499 + x218 * x459 + x219 * x465 +
                    x25 * x500 + x258 * x400 + x259 * x395 + x261 * x395 + x277 * x501 +
                    x277 * x504 + x277 * x506 + x28 * x500 - x308 * x486 - x308 * x487 -
                    x308 * x488 - x308 * x492 - x308 * x502 - x308 * x507 - x311 * x374 -
                    x311 * x375 + x313 * x505 + x313 * x510 - x315 * x374 - x315 * x375 +
                    x316 * x505 + x316 * x510 - x325 * x459 + x329 * x511 + x329 * x514 +
                    x332 * x511 + x332 * x514 + x333 * x398 + 3 * x461 + 3 * x462 + 3 * x463 +
                    3 * x464 + 3 * x467 + 3 * x468 + 3 * x469 + 3 * x470 - 576 * x493 - 576 * x497 +
                    x499 * x60 + x499 * x62) +
            x86 * (x461 + x462 + x463 + x464 + x467 + x468 + x469 + x470 + x471 + x472 + x474 +
                   x475 + x476 + x477 + x478 + x479 + x480 + x481 + x482 + x483);
    element_matrix[5] =
            x144 * (x102 * x546 + x103 * x545 + x103 * x547 + x106 * x545 + x106 * x547 +
                    x110 * x134 + x116 * x535 + x116 * x551 + x120 * x535 + x120 * x551 + x143 +
                    x211 * x51 + x445 + x46 * x549 + x46 * x550 + x51 * x544 + x51 * x549 +
                    x51 * x550 - x516 - x518 - x519 - x520 - x521 - x522 - x523 - x524 - x525 -
                    x526 - x527 - x528 - x529 - x530 - x532 - x534 - x536 - x537 + x538 * x87 +
                    x538 * x92 + x539 * x89 + x539 * x91 + x540 * x87 + x540 * x92 + x541 * x89 +
                    x541 * x91 + 96 * x542 + 96 * x543 + x546 * x93 + x548 * x93) +
            x242 * (-x103 * x557 - x103 * x558 - x106 * x557 - x106 * x558 - x116 * x555 -
                    x116 * x556 - x120 * x555 - x120 * x556 + x131 * x190 + x131 * x451 +
                    x134 * x191 + x134 * x452 + x145 * x552 + x148 * x552 + x154 * x515 +
                    x154 * x517 + x155 * x54 + x155 * x56 + x156 * x515 + x157 * x517 + x159 * x54 +
                    x160 * x56 + x162 * x552 + x164 * x515 + x166 * x517 - x167 * x211 -
                    x167 * x544 - x167 * x549 - x167 * x550 + x168 * x202 + x169 * x54 +
                    x170 * x56 - x171 * x546 - x171 * x548 - x174 * x515 - x175 * x517 -
                    x176 * x517 - x177 * x515 + x179 * x552 + x180 * x202 + x181 * x515 +
                    x182 * x517 + x183 * x531 + x183 * x533 - x185 * x549 - x185 * x550 -
                    x187 * x546 - x187 * x548 + x192 * x559 + x193 * x559 + x194 * x560 +
                    x194 * x561 + x194 * x562 + x201 * x560 + x201 * x561 + x201 * x562 +
                    x202 * x31 + x202 * x33 + 2 * x203 + 2 * x204 + x241 + x25 * x552 + x28 * x552 +
                    x458 + 3 * x516 + 3 * x518 + 3 * x519 + 3 * x520 - 576 * x542 - 576 * x543 +
                    6 * x55 - x553 * x87 - x553 * x92 - x554 * x89 - x554 * x91 + 6 * x57 +
                    6 * x58 + 6 * x59) +
            x86 * (x432 + x516 + x518 + x519 + x520 + x521 + x522 + x523 + x524 + x525 + x526 +
                   x527 + x528 + x529 + x530 + x532 + x534 + x536 + x537 + x84);
}

static SFEM_INLINE void linear_elasticity_matrix_sym(const scalar_t mu,
                                                     const scalar_t lambda,
                                                     const scalar_t *SFEM_RESTRICT adjugate,
                                                     const scalar_t jacobian_determinant,
                                                     const scalar_t *SFEM_RESTRICT trial_grad,
                                                     const scalar_t *SFEM_RESTRICT test_grad,
                                                     const scalar_t qw,
                                                     scalar_t *const SFEM_RESTRICT element_matrix) {
    const scalar_t x0 = 1.0 / jacobian_determinant;
    const scalar_t x1 = mu * x0;
    const scalar_t x2 = adjugate[1] * test_grad[0];
    const scalar_t x3 = test_grad[1] * x1;
    const scalar_t x4 = test_grad[2] * x1;
    const scalar_t x5 = x0 * (adjugate[4] * x3 + adjugate[7] * x4 + x1 * x2);
    const scalar_t x6 = adjugate[1] * x5;
    const scalar_t x7 = adjugate[2] * test_grad[0];
    const scalar_t x8 = x0 * (adjugate[5] * x3 + adjugate[8] * x4 + x1 * x7);
    const scalar_t x9 = adjugate[2] * x8;
    const scalar_t x10 = x0 * (lambda + 2 * mu);
    const scalar_t x11 = adjugate[0] * test_grad[0];
    const scalar_t x12 = adjugate[3] * test_grad[1];
    const scalar_t x13 = adjugate[6] * test_grad[2];
    const scalar_t x14 = x0 * (x10 * x11 + x10 * x12 + x10 * x13);
    const scalar_t x15 = adjugate[4] * x5;
    const scalar_t x16 = adjugate[5] * x8;
    const scalar_t x17 = adjugate[7] * x5;
    const scalar_t x18 = adjugate[8] * x8;
    const scalar_t x19 = jacobian_determinant * qw;
    const scalar_t x20 = lambda * x0;
    const scalar_t x21 = x0 * (x11 * x20 + x12 * x20 + x13 * x20);
    const scalar_t x22 = x0 * (x1 * x11 + x1 * x12 + x1 * x13);
    const scalar_t x23 = adjugate[0] * x22;
    const scalar_t x24 = adjugate[4] * test_grad[1];
    const scalar_t x25 = adjugate[7] * test_grad[2];
    const scalar_t x26 = x0 * (x10 * x2 + x10 * x24 + x10 * x25);
    const scalar_t x27 = adjugate[3] * x22;
    const scalar_t x28 = adjugate[6] * x22;
    const scalar_t x29 = x0 * (x2 * x20 + x20 * x24 + x20 * x25);
    const scalar_t x30 =
            x0 * (adjugate[5] * test_grad[1] * x10 + adjugate[8] * test_grad[2] * x10 + x10 * x7);
    element_matrix[0] += x19 * (trial_grad[0] * (adjugate[0] * x14 + x6 + x9) +
                                trial_grad[1] * (adjugate[3] * x14 + x15 + x16) +
                                trial_grad[2] * (adjugate[6] * x14 + x17 + x18));
    element_matrix[1] += x19 * (trial_grad[0] * (adjugate[0] * x5 + adjugate[1] * x21) +
                                trial_grad[1] * (adjugate[3] * x5 + adjugate[4] * x21) +
                                trial_grad[2] * (adjugate[6] * x5 + adjugate[7] * x21));
    element_matrix[2] += x19 * (trial_grad[0] * (adjugate[0] * x8 + adjugate[2] * x21) +
                                trial_grad[1] * (adjugate[3] * x8 + adjugate[5] * x21) +
                                trial_grad[2] * (adjugate[6] * x8 + adjugate[8] * x21));
    element_matrix[3] += x19 * (trial_grad[0] * (adjugate[1] * x26 + x23 + x9) +
                                trial_grad[1] * (adjugate[4] * x26 + x16 + x27) +
                                trial_grad[2] * (adjugate[7] * x26 + x18 + x28));
    element_matrix[4] += x19 * (trial_grad[0] * (adjugate[1] * x8 + adjugate[2] * x29) +
                                trial_grad[1] * (adjugate[4] * x8 + adjugate[5] * x29) +
                                trial_grad[2] * (adjugate[7] * x8 + adjugate[8] * x29));
    element_matrix[5] += x19 * (trial_grad[0] * (adjugate[2] * x30 + x23 + x6) +
                                trial_grad[1] * (adjugate[5] * x30 + x15 + x27) +
                                trial_grad[2] * (adjugate[8] * x30 + x17 + x28));
}

#endif  // HEX8_LINEAR_ELASTICITY_INLINE_CPU_H
