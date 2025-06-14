#ifndef HEX8_KELVIN_VOIGT_NEWMARK_INLINE_CPU_H
#define HEX8_KELVIN_VOIGT_NEWMARK_INLINE_CPU_H

#include "hex8_inline_cpu.h"



static SFEM_INLINE void hex8_displacement_gradient(const scalar_t *const SFEM_RESTRICT adjugate,
                                                   const scalar_t                      jacobian_determinant,
                                                   const scalar_t                      qx,
                                                   const scalar_t                      qy,
                                                   const scalar_t                      qz,
                                                   const scalar_t *const SFEM_RESTRICT ux,
                                                   const scalar_t *const SFEM_RESTRICT uy,
                                                   const scalar_t *const SFEM_RESTRICT uz,
                                                   scalar_t *const SFEM_RESTRICT       disp_grad) {
    const scalar_t x0  = 1.0 / jacobian_determinant;
    const scalar_t x1  = qx * qz;
    const scalar_t x2  = qz - 1;
    const scalar_t x3  = qx * x2;
    const scalar_t x4  = qx - 1;
    const scalar_t x5  = qz * x4;
    const scalar_t x6  = x2 * x4;
    const scalar_t x7  = ux[0] * x6 - ux[1] * x3 + ux[2] * x3 - ux[3] * x6 - ux[4] * x5 + ux[5] * x1 - ux[6] * x1 + ux[7] * x5;
    const scalar_t x8  = qx * qy;
    const scalar_t x9  = qy - 1;
    const scalar_t x10 = qx * x9;
    const scalar_t x11 = qy * x4;
    const scalar_t x12 = x4 * x9;
    const scalar_t x13 =
            ux[0] * x12 - ux[1] * x10 + ux[2] * x8 - ux[3] * x11 - ux[4] * x12 + ux[5] * x10 - ux[6] * x8 + ux[7] * x11;
    const scalar_t x14 = qy * qz;
    const scalar_t x15 = qy * x2;
    const scalar_t x16 = qz * x9;
    const scalar_t x17 = x2 * x9;
    const scalar_t x18 =
            -ux[0] * x17 + ux[1] * x17 - ux[2] * x15 + ux[3] * x15 + ux[4] * x16 - ux[5] * x16 + ux[6] * x14 - ux[7] * x14;
    const scalar_t x19 = uy[0] * x6 - uy[1] * x3 + uy[2] * x3 - uy[3] * x6 - uy[4] * x5 + uy[5] * x1 - uy[6] * x1 + uy[7] * x5;
    const scalar_t x20 =
            uy[0] * x12 - uy[1] * x10 + uy[2] * x8 - uy[3] * x11 - uy[4] * x12 + uy[5] * x10 - uy[6] * x8 + uy[7] * x11;
    const scalar_t x21 =
            -uy[0] * x17 + uy[1] * x17 - uy[2] * x15 + uy[3] * x15 + uy[4] * x16 - uy[5] * x16 + uy[6] * x14 - uy[7] * x14;
    const scalar_t x22 = uz[0] * x6 - uz[1] * x3 + uz[2] * x3 - uz[3] * x6 - uz[4] * x5 + uz[5] * x1 - uz[6] * x1 + uz[7] * x5;
    const scalar_t x23 =
            uz[0] * x12 - uz[1] * x10 + uz[2] * x8 - uz[3] * x11 - uz[4] * x12 + uz[5] * x10 - uz[6] * x8 + uz[7] * x11;
    const scalar_t x24 =
            -uz[0] * x17 + uz[1] * x17 - uz[2] * x15 + uz[3] * x15 + uz[4] * x16 - uz[5] * x16 + uz[6] * x14 - uz[7] * x14;
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


static SFEM_INLINE void hex8_velocity_gradient(const scalar_t *const SFEM_RESTRICT adjugate,
                                                   const scalar_t                      jacobian_determinant,
                                                   const scalar_t                      qx,
                                                   const scalar_t                      qy,
                                                   const scalar_t                      qz,
                                                   const scalar_t *const SFEM_RESTRICT vx,
                                                   const scalar_t *const SFEM_RESTRICT vy,
                                                   const scalar_t *const SFEM_RESTRICT vz,
                                                   scalar_t *const SFEM_RESTRICT       velo_grad) {
        const scalar_t x0 = 1.0/jacobian_determinant;
        const scalar_t x1 = qx*qz;
        const scalar_t x2 = qz - 1;
        const scalar_t x3 = qx*x2;
        const scalar_t x4 = qx - 1;
        const scalar_t x5 = qz*x4;
        const scalar_t x6 = x2*x4;
        const scalar_t x7 = vx[0]*x6 - vx[1]*x3 + vx[2]*x3 - vx[3]*x6 - vx[4]*x5 + vx[5]*x1 - 
        vx[6]*x1 + vx[7]*x5;
        const scalar_t x8 = qx*qy;
        const scalar_t x9 = qy - 1;
        const scalar_t x10 = qx*x9;
        const scalar_t x11 = qy*x4;
        const scalar_t x12 = x4*x9;
        const scalar_t x13 = vx[0]*x12 - vx[1]*x10 + vx[2]*x8 - vx[3]*x11 - vx[4]*x12 + vx[5]*x10 -
        vx[6]*x8 + vx[7]*x11;
        const scalar_t x14 = qy*qz;
        const scalar_t x15 = qy*x2;
        const scalar_t x16 = qz*x9;
        const scalar_t x17 = x2*x9;
        const scalar_t x18 = -vx[0]*x17 + vx[1]*x17 - vx[2]*x15 + vx[3]*x15 + vx[4]*x16 - vx[5]*x16
        + vx[6]*x14 - vx[7]*x14;
        const scalar_t x19 = vy[0]*x6 - vy[1]*x3 + vy[2]*x3 - vy[3]*x6 - vy[4]*x5 + vy[5]*x1 - 
        vy[6]*x1 + vy[7]*x5;
        const scalar_t x20 = vy[0]*x12 - vy[1]*x10 + vy[2]*x8 - vy[3]*x11 - vy[4]*x12 + vy[5]*x10 -
        vy[6]*x8 + vy[7]*x11;
        const scalar_t x21 = -vy[0]*x17 + vy[1]*x17 - vy[2]*x15 + vy[3]*x15 + vy[4]*x16 - vy[5]*x16
        + vy[6]*x14 - vy[7]*x14;
        const scalar_t x22 = vz[0]*x6 - vz[1]*x3 + vz[2]*x3 - vz[3]*x6 - vz[4]*x5 + vz[5]*x1 - 
        vz[6]*x1 + vz[7]*x5;
        const scalar_t x23 = vz[0]*x12 - vz[1]*x10 + vz[2]*x8 - vz[3]*x11 - vz[4]*x12 + vz[5]*x10 -
        vz[6]*x8 + vz[7]*x11;
        const scalar_t x24 = -vz[0]*x17 + vz[1]*x17 - vz[2]*x15 + vz[3]*x15 + vz[4]*x16 - vz[5]*x16
        + vz[6]*x14 - vz[7]*x14;
        velo_grad[0] = -x0*(-adjugate[0]*x18 + adjugate[3]*x7 + adjugate[6]*x13);
        velo_grad[1] = -x0*(-adjugate[1]*x18 + adjugate[4]*x7 + adjugate[7]*x13);
        velo_grad[2] = -x0*(-adjugate[2]*x18 + adjugate[5]*x7 + adjugate[8]*x13);
        velo_grad[3] = -x0*(-adjugate[0]*x21 + adjugate[3]*x19 + adjugate[6]*x20);
        velo_grad[4] = -x0*(-adjugate[1]*x21 + adjugate[4]*x19 + adjugate[7]*x20);
        velo_grad[5] = -x0*(-adjugate[2]*x21 + adjugate[5]*x19 + adjugate[8]*x20);
        velo_grad[6] = -x0*(-adjugate[0]*x24 + adjugate[3]*x22 + adjugate[6]*x23);
        velo_grad[7] = -x0*(-adjugate[1]*x24 + adjugate[4]*x22 + adjugate[7]*x23);
        velo_grad[8] = -x0*(-adjugate[2]*x24 + adjugate[5]*x22 + adjugate[8]*x23);
}



static SFEM_INLINE void hex8_kelvin_voigt_newmark_lhs_apply_adj(const scalar_t                      k,
                                                         const scalar_t                      K,
                                                         const scalar_t                      eta,
                                                         const scalar_t                      dt,
                                                         const scalar_t                      gamma,
                                                         const scalar_t                      beta,
                                                         const scalar_t *const SFEM_RESTRICT adjugate,
                                                         const scalar_t                      jacobian_determinant,
                                                         const scalar_t                      qx,
                                                         const scalar_t                      qy,
                                                         const scalar_t                      qz,
                                                         const scalar_t                      qw,
                                                         const scalar_t *const SFEM_RESTRICT incrementx,
                                                         const scalar_t *const SFEM_RESTRICT incrementy,
                                                         const scalar_t *const SFEM_RESTRICT incrementz,
                                                         accumulator_t *const SFEM_RESTRICT  outx,
                                                         accumulator_t *const SFEM_RESTRICT  outy,
                                                         accumulator_t *const SFEM_RESTRICT  outz){
        const scalar_t x0 = qy - 1;
        const scalar_t x1 = qz - 1;
        const scalar_t x2 = x0*x1;
        const scalar_t x3 = adjugate[1]*x2;
        const scalar_t x4 = qx - 1;
        const scalar_t x5 = x1*x4;
        const scalar_t x6 = adjugate[4]*x5;
        const scalar_t x7 = x0*x4;
        const scalar_t x8 = adjugate[7]*x7;
        const scalar_t x9 = x3 + x6 + x8;
        const scalar_t x10 = adjugate[0]*x2;
        const scalar_t x11 = adjugate[3]*x5;
        const scalar_t x12 = adjugate[6]*x7;
        const scalar_t x13 = x10 + x11 + x12;
        const scalar_t x14 = K + 0.16666666666666669*k;
        const scalar_t x15 = x13*x14;
        const scalar_t x16 = -x0;
        const scalar_t x17 = -x1;
        const scalar_t x18 = x16*x17;
        const scalar_t x19 = adjugate[1]*x18;
        const scalar_t x20 = -x4;
        const scalar_t x21 = x17*x20;
        const scalar_t x22 = adjugate[4]*x21;
        const scalar_t x23 = x16*x20;
        const scalar_t x24 = adjugate[7]*x23;
        const scalar_t x25 = x19 + x22 + x24;
        const scalar_t x26 = adjugate[0]*x18;
        const scalar_t x27 = adjugate[3]*x21;
        const scalar_t x28 = adjugate[6]*x23;
        const scalar_t x29 = x26 + x27 + x28;
        const scalar_t x30 = 1.0/beta;
        const scalar_t x31 = 1.0/dt;
        const scalar_t x32 = eta*gamma*x30*x31;
        const scalar_t x33 = 0.16666666666666669*x32;
        const scalar_t x34 = x29*x33;
        const scalar_t x35 = x15*x9 + x25*x34;
        const scalar_t x36 = adjugate[2]*x2;
        const scalar_t x37 = adjugate[5]*x5;
        const scalar_t x38 = adjugate[8]*x7;
        const scalar_t x39 = x36 + x37 + x38;
        const scalar_t x40 = adjugate[2]*x18;
        const scalar_t x41 = adjugate[5]*x21;
        const scalar_t x42 = adjugate[8]*x23;
        const scalar_t x43 = x40 + x41 + x42;
        const scalar_t x44 = x15*x39 + x34*x43;
        const scalar_t x45 = qy*x1;
        const scalar_t x46 = adjugate[0]*x45;
        const scalar_t x47 = qy*x4;
        const scalar_t x48 = adjugate[6]*x47;
        const scalar_t x49 = x11 + x46 + x48;
        const scalar_t x50 = x49*x9;
        const scalar_t x51 = adjugate[1]*x45;
        const scalar_t x52 = adjugate[7]*x47;
        const scalar_t x53 = x51 + x52 + x6;
        const scalar_t x54 = K - 0.33333333333333331*k;
        const scalar_t x55 = 2*x54;
        const scalar_t x56 = x13*x55;
        const scalar_t x57 = eta*gamma*x30*x31*(0.66666666666666663*x13*x53 - x50) - k*x50 - x53*x56;
        const scalar_t x58 = (1.0/2.0)*incrementy[3];
        const scalar_t x59 = qz*x0;
        const scalar_t x60 = adjugate[0]*x59;
        const scalar_t x61 = qz*x4;
        const scalar_t x62 = adjugate[3]*x61;
        const scalar_t x63 = x12 + x60 + x62;
        const scalar_t x64 = x63*x9;
        const scalar_t x65 = adjugate[1]*x59;
        const scalar_t x66 = adjugate[4]*x61;
        const scalar_t x67 = x65 + x66 + x8;
        const scalar_t x68 = eta*gamma*x30*x31*(0.66666666666666663*x13*x67 - x64) - k*x64 - x56*x67;
        const scalar_t x69 = (1.0/2.0)*incrementy[4];
        const scalar_t x70 = x39*x49;
        const scalar_t x71 = adjugate[2]*x45;
        const scalar_t x72 = adjugate[8]*x47;
        const scalar_t x73 = x37 + x71 + x72;
        const scalar_t x74 = eta*gamma*x30*x31*(0.66666666666666663*x13*x73 - x70) - k*x70 - x56*x73;
        const scalar_t x75 = (1.0/2.0)*incrementz[3];
        const scalar_t x76 = x39*x63;
        const scalar_t x77 = adjugate[2]*x59;
        const scalar_t x78 = adjugate[5]*x61;
        const scalar_t x79 = x38 + x77 + x78;
        const scalar_t x80 = eta*gamma*x30*x31*(0.66666666666666663*x13*x79 - x76) - k*x76 - x56*x79;
        const scalar_t x81 = (1.0/2.0)*incrementz[4];
        const scalar_t x82 = POW2(x13);
        const scalar_t x83 = 2*x82;
        const scalar_t x84 = POW2(x9);
        const scalar_t x85 = POW2(x39);
        const scalar_t x86 = x84 + x85;
        const scalar_t x87 = (1.0/2.0)*incrementx[0];
        const scalar_t x88 = qy*qz;
        const scalar_t x89 = adjugate[0]*x88;
        const scalar_t x90 = qx*qz;
        const scalar_t x91 = adjugate[3]*x90;
        const scalar_t x92 = qx*qy;
        const scalar_t x93 = adjugate[6]*x92;
        const scalar_t x94 = x89 + x91 + x93;
        const scalar_t x95 = x25*x94;
        const scalar_t x96 = adjugate[1]*x88;
        const scalar_t x97 = adjugate[4]*x90;
        const scalar_t x98 = adjugate[7]*x92;
        const scalar_t x99 = x96 + x97 + x98;
        const scalar_t x100 = x29*x99;
        const scalar_t x101 = k*x95 + x100*x55 + x32*(-0.66666666666666663*x100 + x95);
        const scalar_t x102 = (1.0/2.0)*incrementy[6];
        const scalar_t x103 = x43*x94;
        const scalar_t x104 = adjugate[2]*x88;
        const scalar_t x105 = adjugate[5]*x90;
        const scalar_t x106 = adjugate[8]*x92;
        const scalar_t x107 = x104 + x105 + x106;
        const scalar_t x108 = x107*x29;
        const scalar_t x109 = k*x103 + x108*x55 + x32*(x103 - 0.66666666666666663*x108);
        const scalar_t x110 = (1.0/2.0)*incrementz[6];
        const scalar_t x111 = qx*x1;
        const scalar_t x112 = adjugate[3]*x111;
        const scalar_t x113 = x112 + x46 + x93;
        const scalar_t x114 = x113*x25;
        const scalar_t x115 = adjugate[4]*x111;
        const scalar_t x116 = x115 + x51 + x98;
        const scalar_t x117 = x116*x29;
        const scalar_t x118 = k*x114 + x117*x55 - x32*(-x114 + 0.66666666666666663*x116*x29);
        const scalar_t x119 = (1.0/2.0)*incrementy[2];
        const scalar_t x120 = qx*x0;
        const scalar_t x121 = adjugate[6]*x120;
        const scalar_t x122 = x121 + x60 + x91;
        const scalar_t x123 = x122*x25;
        const scalar_t x124 = adjugate[7]*x120;
        const scalar_t x125 = x124 + x65 + x97;
        const scalar_t x126 = x125*x29;
        const scalar_t x127 = k*x123 + x126*x55 - x32*(-x123 + 0.66666666666666663*x125*x29);
        const scalar_t x128 = (1.0/2.0)*incrementy[5];
        const scalar_t x129 = x48 + x62 + x89;
        const scalar_t x130 = x129*x25;
        const scalar_t x131 = x52 + x66 + x96;
        const scalar_t x132 = x131*x29;
        const scalar_t x133 = k*x130 + x132*x55 - x32*(-x130 + 0.66666666666666663*x131*x29);
        const scalar_t x134 = (1.0/2.0)*incrementy[7];
        const scalar_t x135 = x113*x43;
        const scalar_t x136 = adjugate[5]*x111;
        const scalar_t x137 = x106 + x136 + x71;
        const scalar_t x138 = x137*x29;
        const scalar_t x139 = k*x135 + x138*x55 - x32*(-x135 + 0.66666666666666663*x137*x29);
        const scalar_t x140 = (1.0/2.0)*incrementz[2];
        const scalar_t x141 = x122*x43;
        const scalar_t x142 = adjugate[8]*x120;
        const scalar_t x143 = x105 + x142 + x77;
        const scalar_t x144 = x143*x29;
        const scalar_t x145 = k*x141 + x144*x55 - x32*(-x141 + 0.66666666666666663*x143*x29);
        const scalar_t x146 = (1.0/2.0)*incrementz[5];
        const scalar_t x147 = x129*x43;
        const scalar_t x148 = x104 + x72 + x78;
        const scalar_t x149 = x148*x29;
        const scalar_t x150 = k*x147 + x149*x55 - x32*(-x147 + 0.66666666666666663*x148*x29);
        const scalar_t x151 = (1.0/2.0)*incrementz[7];
        const scalar_t x152 = x10 + x112 + x121;
        const scalar_t x153 = k*x152;
        const scalar_t x154 = x115 + x124 + x3;
        const scalar_t x155 = qx*x17;
        const scalar_t x156 = qx*x16;
        const scalar_t x157 = adjugate[3]*x155 + adjugate[6]*x156 - x26;
        const scalar_t x158 = x157*x25;
        const scalar_t x159 = adjugate[4]*x155 + adjugate[7]*x156 - x19;
        const scalar_t x160 = x153*x9 + x154*x56 + x32*(-x158 + 0.66666666666666663*x159*x29);
        const scalar_t x161 = (1.0/2.0)*incrementy[1];
        const scalar_t x162 = x136 + x142 + x36;
        const scalar_t x163 = x157*x43;
        const scalar_t x164 = adjugate[5]*x155 + adjugate[8]*x156 - x40;
        const scalar_t x165 = x153*x39 + x162*x56 + x32*(-x163 + 0.66666666666666663*x164*x29);
        const scalar_t x166 = (1.0/2.0)*incrementz[1];
        const scalar_t x167 = x13*x152;
        const scalar_t x168 = 2*x167;
        const scalar_t x169 = x154*x9;
        const scalar_t x170 = x162*x39;
        const scalar_t x171 = x169 + x170;
        const scalar_t x172 = eta*gamma*x30*x31*(-1.3333333333333335*x167 - x171) - k*(x168 + x171) - x168*x54;
        const scalar_t x173 = (1.0/2.0)*incrementx[1];
        const scalar_t x174 = x13*x49;
        const scalar_t x175 = 2*x174;
        const scalar_t x176 = x53*x9;
        const scalar_t x177 = x39*x73;
        const scalar_t x178 = x176 + x177;
        const scalar_t x179 = eta*gamma*x30*x31*(-1.3333333333333335*x174 - x178) - k*(x175 + x178) - x175*x54;
        const scalar_t x180 = (1.0/2.0)*incrementx[3];
        const scalar_t x181 = x13*x63;
        const scalar_t x182 = 2*x181;
        const scalar_t x183 = x67*x9;
        const scalar_t x184 = x39*x79;
        const scalar_t x185 = x183 + x184;
        const scalar_t x186 = eta*gamma*x30*x31*(-1.3333333333333335*x181 - x185) - k*(x182 + x185) - x182*x54;
        const scalar_t x187 = (1.0/2.0)*incrementx[4];
        const scalar_t x188 = x29*x94;
        const scalar_t x189 = x25*x99;
        const scalar_t x190 = x107*x43;
        const scalar_t x191 = x189 + x190;
        const scalar_t x192 = k*(2*x188 + x191) + x32*(1.3333333333333335*x188 + x191) + x56*x94;
        const scalar_t x193 = (1.0/2.0)*incrementx[6];
        const scalar_t x194 = x113*x29;
        const scalar_t x195 = x116*x25;
        const scalar_t x196 = x137*x43;
        const scalar_t x197 = x195 + x196;
        const scalar_t x198 = k*(2*x194 + x197) + x113*x56 - x32*(-1.3333333333333335*x194 - x197);
        const scalar_t x199 = (1.0/2.0)*incrementx[2];
        const scalar_t x200 = x122*x29;
        const scalar_t x201 = x125*x25;
        const scalar_t x202 = x143*x43;
        const scalar_t x203 = x201 + x202;
        const scalar_t x204 = k*(2*x200 + x203) + x122*x56 - x32*(-1.3333333333333335*x200 - x203);
        const scalar_t x205 = (1.0/2.0)*incrementx[5];
        const scalar_t x206 = x129*x29;
        const scalar_t x207 = x131*x25;
        const scalar_t x208 = x148*x43;
        const scalar_t x209 = x207 + x208;
        const scalar_t x210 = k*(2*x206 + x209) + x129*x56 - x32*(-1.3333333333333335*x206 - x209);
        const scalar_t x211 = (1.0/2.0)*incrementx[7];
        const scalar_t x212 = qw/jacobian_determinant;
        const scalar_t x213 = x14*x152;
        const scalar_t x214 = x157*x33;
        const scalar_t x215 = x154*x213 + x159*x214;
        const scalar_t x216 = x162*x213 + x164*x214;
        const scalar_t x217 = POW2(x152);
        const scalar_t x218 = 2*x217;
        const scalar_t x219 = POW2(x154);
        const scalar_t x220 = POW2(x162);
        const scalar_t x221 = x219 + x220;
        const scalar_t x222 = x154*x49;
        const scalar_t x223 = x152*x55;
        const scalar_t x224 = 0.66666666666666663*x152;
        const scalar_t x225 = k*x222 + x223*x53 + x32*(x222 - x224*x53);
        const scalar_t x226 = x154*x63;
        const scalar_t x227 = k*x226 + x223*x67 + x32*(-x224*x67 + x226);
        const scalar_t x228 = x162*x49;
        const scalar_t x229 = k*x228 + x223*x73 + x32*(-x224*x73 + x228);
        const scalar_t x230 = x162*x63;
        const scalar_t x231 = k*x230 + x223*x79 + x32*(-x224*x79 + x230);
        const scalar_t x232 = x159*x94;
        const scalar_t x233 = x157*x99;
        const scalar_t x234 = k*x232 + x233*x55 + x32*(x232 - 0.66666666666666663*x233);
        const scalar_t x235 = x164*x94;
        const scalar_t x236 = x107*x157;
        const scalar_t x237 = k*x235 + x236*x55 + x32*(x235 - 0.66666666666666663*x236);
        const scalar_t x238 = x113*x159;
        const scalar_t x239 = x116*x157;
        const scalar_t x240 = k*x238 + x239*x55 - x32*(0.66666666666666663*x116*x157 - x238);
        const scalar_t x241 = x122*x159;
        const scalar_t x242 = x125*x157;
        const scalar_t x243 = k*x241 + x242*x55 - x32*(0.66666666666666663*x125*x157 - x241);
        const scalar_t x244 = x129*x159;
        const scalar_t x245 = x131*x157;
        const scalar_t x246 = k*x244 + x245*x55 - x32*(0.66666666666666663*x131*x157 - x244);
        const scalar_t x247 = x113*x164;
        const scalar_t x248 = x137*x157;
        const scalar_t x249 = k*x247 + x248*x55 - x32*(0.66666666666666663*x137*x157 - x247);
        const scalar_t x250 = x122*x164;
        const scalar_t x251 = x143*x157;
        const scalar_t x252 = k*x250 + x251*x55 - x32*(0.66666666666666663*x143*x157 - x250);
        const scalar_t x253 = x129*x164;
        const scalar_t x254 = x148*x157;
        const scalar_t x255 = k*x253 + x254*x55 - x32*(0.66666666666666663*x148*x157 - x253);
        const scalar_t x256 = k*x13;
        const scalar_t x257 = x154*x256 + x223*x9 + x32*(0.66666666666666663*x158 - x159*x29);
        const scalar_t x258 = (1.0/2.0)*incrementy[0];
        const scalar_t x259 = x162*x256 + x223*x39 + x32*(0.66666666666666663*x163 - x164*x29);
        const scalar_t x260 = (1.0/2.0)*incrementz[0];
        const scalar_t x261 = x152*x49;
        const scalar_t x262 = 2*x261;
        const scalar_t x263 = x154*x53;
        const scalar_t x264 = x162*x73;
        const scalar_t x265 = x263 + x264;
        const scalar_t x266 = k*(x262 + x265) + x262*x54 + x32*(1.3333333333333335*x261 + x265);
        const scalar_t x267 = x152*x63;
        const scalar_t x268 = 2*x267;
        const scalar_t x269 = x154*x67;
        const scalar_t x270 = x162*x79;
        const scalar_t x271 = x269 + x270;
        const scalar_t x272 = k*(x268 + x271) + x268*x54 + x32*(1.3333333333333335*x267 + x271);
        const scalar_t x273 = x157*x94;
        const scalar_t x274 = x159*x99;
        const scalar_t x275 = x107*x164;
        const scalar_t x276 = x274 + x275;
        const scalar_t x277 = -k*(2*x273 + x276) + 2*x152*x54*x94 - x32*(1.3333333333333335*x273 + x276);
        const scalar_t x278 = x113*x157;
        const scalar_t x279 = x116*x159;
        const scalar_t x280 = x137*x164;
        const scalar_t x281 = x279 + x280;
        const scalar_t x282 = k*(2*x278 + x281) - x113*x223 - x32*(-1.3333333333333335*x278 - x281);
        const scalar_t x283 = x122*x157;
        const scalar_t x284 = x125*x159;
        const scalar_t x285 = x143*x164;
        const scalar_t x286 = x284 + x285;
        const scalar_t x287 = k*(2*x283 + x286) - x122*x223 - x32*(-1.3333333333333335*x283 - x286);
        const scalar_t x288 = x129*x157;
        const scalar_t x289 = x131*x159;
        const scalar_t x290 = x148*x164;
        const scalar_t x291 = x289 + x290;
        const scalar_t x292 = k*(2*x288 + x291) - x129*x223 - x32*(-1.3333333333333335*x288 - x291);
        const scalar_t x293 = x14 + x33;
        const scalar_t x294 = x113*x293;
        const scalar_t x295 = x116*x294;
        const scalar_t x296 = incrementz[2]*x137;
        const scalar_t x297 = x116*x94;
        const scalar_t x298 = x113*x99;
        const scalar_t x299 = eta*gamma*x30*x31*(0.66666666666666663*x113*x99 - x297) - k*x297 - x298*x55;
        const scalar_t x300 = x137*x94;
        const scalar_t x301 = x107*x113;
        const scalar_t x302 = eta*gamma*x30*x31*(0.66666666666666663*x107*x113 - x300) - k*x300 - x301*x55;
        const scalar_t x303 = POW2(x113);
        const scalar_t x304 = 2*x303;
        const scalar_t x305 = POW2(x116);
        const scalar_t x306 = POW2(x137);
        const scalar_t x307 = x305 + x306;
        const scalar_t x308 = x116*x129;
        const scalar_t x309 = x113*x131;
        const scalar_t x310 = k*x308 + x309*x55 + x32*(x308 - 0.66666666666666663*x309);
        const scalar_t x311 = x129*x137;
        const scalar_t x312 = x113*x148;
        const scalar_t x313 = k*x311 + x312*x55 + x32*(x311 - 0.66666666666666663*x312);
        const scalar_t x314 = x116*x122;
        const scalar_t x315 = x113*x125;
        const scalar_t x316 = k*x314 + x315*x55 - x32*(-x314 + 0.66666666666666663*x315);
        const scalar_t x317 = x122*x137;
        const scalar_t x318 = x113*x143;
        const scalar_t x319 = k*x317 + x318*x55 - x32*(-x317 + 0.66666666666666663*x318);
        const scalar_t x320 = k*x239 + x238*x55 - x32*(0.66666666666666663*x238 - x239);
        const scalar_t x321 = qy*x17;
        const scalar_t x322 = qy*x20;
        const scalar_t x323 = adjugate[0]*x321 + adjugate[6]*x322 - x27;
        const scalar_t x324 = x116*x323;
        const scalar_t x325 = adjugate[1]*x321 + adjugate[7]*x322 - x22;
        const scalar_t x326 = x113*x325;
        const scalar_t x327 = k*x324 - x32*(0.66666666666666663*x113*x325 - x324) + x326*x55;
        const scalar_t x328 = qz*x16;
        const scalar_t x329 = qz*x20;
        const scalar_t x330 = adjugate[0]*x328 + adjugate[3]*x329 - x28;
        const scalar_t x331 = x116*x330;
        const scalar_t x332 = adjugate[1]*x328 + adjugate[4]*x329 - x24;
        const scalar_t x333 = x113*x332;
        const scalar_t x334 = k*x331 - x32*(-x331 + 0.66666666666666663*x333) + x333*x55;
        const scalar_t x335 = k*x248 + x247*x55 - x32*(0.66666666666666663*x247 - x248);
        const scalar_t x336 = x137*x323;
        const scalar_t x337 = adjugate[2]*x321 + adjugate[8]*x322 - x41;
        const scalar_t x338 = x113*x337;
        const scalar_t x339 = k*x336 - x32*(0.66666666666666663*x113*x337 - x336) + x338*x55;
        const scalar_t x340 = x137*x330;
        const scalar_t x341 = adjugate[2]*x328 + adjugate[5]*x329 - x42;
        const scalar_t x342 = x113*x341;
        const scalar_t x343 = k*x340 - x32*(-x340 + 0.66666666666666663*x342) + x342*x55;
        const scalar_t x344 = k*x117 + x114*x55 - x32*(0.66666666666666663*x114 - x117);
        const scalar_t x345 = k*x138 + x135*x55 - x32*(0.66666666666666663*x135 - x138);
        const scalar_t x346 = x113*x94;
        const scalar_t x347 = 2*x346;
        const scalar_t x348 = x116*x99;
        const scalar_t x349 = x107*x137;
        const scalar_t x350 = x348 + x349;
        const scalar_t x351 = eta*gamma*x30*x31*(-1.3333333333333335*x346 - x350) - k*(x347 + x350) - x347*x54;
        const scalar_t x352 = x113*x122;
        const scalar_t x353 = 2*x352;
        const scalar_t x354 = x116*x125;
        const scalar_t x355 = x137*x143;
        const scalar_t x356 = x354 + x355;
        const scalar_t x357 = k*(x353 + x356) + x32*(1.3333333333333335*x352 + x356) + x353*x54;
        const scalar_t x358 = x113*x129;
        const scalar_t x359 = 2*x358;
        const scalar_t x360 = x116*x131;
        const scalar_t x361 = x137*x148;
        const scalar_t x362 = x360 + x361;
        const scalar_t x363 = k*(x359 + x362) + x32*(1.3333333333333335*x358 + x362) + x359*x54;
        const scalar_t x364 = x113*x55;
        const scalar_t x365 = x113*x323;
        const scalar_t x366 = x116*x325;
        const scalar_t x367 = x137*x337;
        const scalar_t x368 = x366 + x367;
        const scalar_t x369 = k*(2*x365 + x368) - x32*(-1.3333333333333335*x365 - x368) - x364*x49;
        const scalar_t x370 = x113*x330;
        const scalar_t x371 = x116*x332;
        const scalar_t x372 = x137*x341;
        const scalar_t x373 = x371 + x372;
        const scalar_t x374 = k*(2*x370 + x373) - x32*(-1.3333333333333335*x370 - x373) - x364*x63;
        const scalar_t x375 = x14*x49;
        const scalar_t x376 = x323*x33;
        const scalar_t x377 = x325*x376 + x375*x53;
        const scalar_t x378 = x337*x376 + x375*x73;
        const scalar_t x379 = POW2(x49);
        const scalar_t x380 = 2*x379;
        const scalar_t x381 = POW2(x53);
        const scalar_t x382 = POW2(x73);
        const scalar_t x383 = x381 + x382;
        const scalar_t x384 = x325*x94;
        const scalar_t x385 = x323*x99;
        const scalar_t x386 = k*x384 + x32*(x384 - 0.66666666666666663*x385) + x385*x55;
        const scalar_t x387 = x337*x94;
        const scalar_t x388 = x107*x323;
        const scalar_t x389 = k*x387 + x32*(x387 - 0.66666666666666663*x388) + x388*x55;
        const scalar_t x390 = k*x326 - x32*(0.66666666666666663*x324 - x326) + x324*x55;
        const scalar_t x391 = x122*x325;
        const scalar_t x392 = x125*x323;
        const scalar_t x393 = k*x391 - x32*(-x391 + 0.66666666666666663*x392) + x392*x55;
        const scalar_t x394 = x129*x325;
        const scalar_t x395 = x131*x323;
        const scalar_t x396 = k*x394 - x32*(0.66666666666666663*x131*x323 - x394) + x395*x55;
        const scalar_t x397 = k*x338 - x32*(0.66666666666666663*x336 - x338) + x336*x55;
        const scalar_t x398 = x122*x337;
        const scalar_t x399 = x143*x323;
        const scalar_t x400 = k*x398 - x32*(-x398 + 0.66666666666666663*x399) + x399*x55;
        const scalar_t x401 = x129*x337;
        const scalar_t x402 = x148*x323;
        const scalar_t x403 = k*x401 - x32*(0.66666666666666663*x148*x323 - x401) + x402*x55;
        const scalar_t x404 = 0.66666666666666663*x323;
        const scalar_t x405 = x153*x53 + x222*x55 - x32*(-x157*x325 + x159*x404);
        const scalar_t x406 = k*x63;
        const scalar_t x407 = x49*x67;
        const scalar_t x408 = -x32*(-x325*x330 + x332*x404) + x406*x53 + x407*x55;
        const scalar_t x409 = x153*x73 + x228*x55 - x32*(-x157*x337 + x164*x404);
        const scalar_t x410 = x49*x79;
        const scalar_t x411 = -x32*(-x330*x337 + x341*x404) + x406*x73 + x410*x55;
        const scalar_t x412 = x256*x53 + x32*(x25*x404 - x29*x325) + x50*x55;
        const scalar_t x413 = x256*x73 + x32*(-x29*x337 + x404*x43) + x55*x70;
        const scalar_t x414 = x49*x63;
        const scalar_t x415 = 2*x414;
        const scalar_t x416 = x53*x67;
        const scalar_t x417 = x73*x79;
        const scalar_t x418 = x416 + x417;
        const scalar_t x419 = k*(x415 + x418) + x32*(1.3333333333333335*x414 + x418) + x415*x54;
        const scalar_t x420 = x323*x94;
        const scalar_t x421 = x325*x99;
        const scalar_t x422 = x107*x337;
        const scalar_t x423 = x421 + x422;
        const scalar_t x424 = -k*(2*x420 + x423) - x32*(1.3333333333333335*x420 + x423) + 2*x49*x54*x94;
        const scalar_t x425 = x49*x55;
        const scalar_t x426 = x122*x323;
        const scalar_t x427 = x125*x325;
        const scalar_t x428 = x143*x337;
        const scalar_t x429 = x427 + x428;
        const scalar_t x430 = k*(2*x426 + x429) - x122*x425 - x32*(-1.3333333333333335*x426 - x429);
        const scalar_t x431 = x129*x323;
        const scalar_t x432 = x131*x325;
        const scalar_t x433 = x148*x337;
        const scalar_t x434 = x432 + x433;
        const scalar_t x435 = k*(2*x431 + x434) - x129*x425 - x32*(-1.3333333333333335*x431 - x434);
        const scalar_t x436 = x14*x63;
        const scalar_t x437 = x33*x330;
        const scalar_t x438 = x332*x437 + x436*x67;
        const scalar_t x439 = x341*x437 + x436*x79;
        const scalar_t x440 = POW2(x63);
        const scalar_t x441 = 2*x440;
        const scalar_t x442 = POW2(x67);
        const scalar_t x443 = POW2(x79);
        const scalar_t x444 = x442 + x443;
        const scalar_t x445 = x55*x63;
        const scalar_t x446 = 0.66666666666666663*x63;
        const scalar_t x447 = k*x407 + x32*(x407 - x446*x53) + x445*x53;
        const scalar_t x448 = k*x410 + x32*(x410 - x446*x73) + x445*x73;
        const scalar_t x449 = x332*x94;
        const scalar_t x450 = x330*x99;
        const scalar_t x451 = k*x449 + x32*(x449 - 0.66666666666666663*x450) + x450*x55;
        const scalar_t x452 = x341*x94;
        const scalar_t x453 = x107*x330;
        const scalar_t x454 = k*x452 + x32*(x452 - 0.66666666666666663*x453) + x453*x55;
        const scalar_t x455 = k*x333 - x32*(0.66666666666666663*x116*x330 - x333) + x331*x55;
        const scalar_t x456 = x122*x332;
        const scalar_t x457 = x125*x330;
        const scalar_t x458 = k*x456 - x32*(-x456 + 0.66666666666666663*x457) + x457*x55;
        const scalar_t x459 = x129*x332;
        const scalar_t x460 = x131*x330;
        const scalar_t x461 = k*x459 - x32*(0.66666666666666663*x131*x330 - x459) + x460*x55;
        const scalar_t x462 = k*x342 - x32*(0.66666666666666663*x137*x330 - x342) + x340*x55;
        const scalar_t x463 = x122*x341;
        const scalar_t x464 = x143*x330;
        const scalar_t x465 = k*x463 - x32*(-x463 + 0.66666666666666663*x464) + x464*x55;
        const scalar_t x466 = x129*x341;
        const scalar_t x467 = x148*x330;
        const scalar_t x468 = k*x466 - x32*(0.66666666666666663*x148*x330 - x466) + x467*x55;
        const scalar_t x469 = 0.66666666666666663*x330;
        const scalar_t x470 = x153*x67 + x226*x55 - x32*(-x157*x332 + x159*x469);
        const scalar_t x471 = x153*x79 + x230*x55 - x32*(-x157*x341 + x164*x469);
        const scalar_t x472 = x256*x67 + x32*(x25*x469 - x29*x332) + x55*x64;
        const scalar_t x473 = x256*x79 + x32*(-x29*x341 + x43*x469) + x55*x76;
        const scalar_t x474 = x330*x94;
        const scalar_t x475 = x332*x99;
        const scalar_t x476 = x107*x341;
        const scalar_t x477 = x475 + x476;
        const scalar_t x478 = -k*(2*x474 + x477) - x32*(1.3333333333333335*x474 + x477) + 2*x54*x63*x94;
        const scalar_t x479 = x122*x330;
        const scalar_t x480 = x125*x332;
        const scalar_t x481 = x143*x341;
        const scalar_t x482 = x480 + x481;
        const scalar_t x483 = k*(2*x479 + x482) - x122*x445 - x32*(-1.3333333333333335*x479 - x482);
        const scalar_t x484 = x129*x330;
        const scalar_t x485 = x131*x332;
        const scalar_t x486 = x148*x341;
        const scalar_t x487 = x485 + x486;
        const scalar_t x488 = k*(2*x484 + x487) - x129*x445 - x32*(-1.3333333333333335*x484 - x487);
        const scalar_t x489 = x122*x293;
        const scalar_t x490 = x125*x489;
        const scalar_t x491 = incrementz[5]*x143;
        const scalar_t x492 = x125*x94;
        const scalar_t x493 = x122*x99;
        const scalar_t x494 = eta*gamma*x30*x31*(0.66666666666666663*x122*x99 - x492) - k*x492 - x493*x55;
        const scalar_t x495 = x143*x94;
        const scalar_t x496 = x107*x122;
        const scalar_t x497 = eta*gamma*x30*x31*(0.66666666666666663*x107*x122 - x495) - k*x495 - x496*x55;
        const scalar_t x498 = POW2(x122);
        const scalar_t x499 = 2*x498;
        const scalar_t x500 = POW2(x125);
        const scalar_t x501 = POW2(x143);
        const scalar_t x502 = x500 + x501;
        const scalar_t x503 = k*x315 + x314*x55 + x32*(-0.66666666666666663*x314 + x315);
        const scalar_t x504 = x125*x129;
        const scalar_t x505 = x122*x131;
        const scalar_t x506 = k*x504 + x32*(x504 - 0.66666666666666663*x505) + x505*x55;
        const scalar_t x507 = k*x318 + x317*x55 + x32*(-0.66666666666666663*x317 + x318);
        const scalar_t x508 = x129*x143;
        const scalar_t x509 = x122*x148;
        const scalar_t x510 = k*x508 + x32*(x508 - 0.66666666666666663*x509) + x509*x55;
        const scalar_t x511 = k*x242 + x241*x55 - x32*(0.66666666666666663*x241 - x242);
        const scalar_t x512 = k*x392 - x32*(0.66666666666666663*x122*x325 - x392) + x391*x55;
        const scalar_t x513 = k*x457 - x32*(0.66666666666666663*x122*x332 - x457) + x456*x55;
        const scalar_t x514 = k*x251 + x250*x55 - x32*(0.66666666666666663*x250 - x251);
        const scalar_t x515 = k*x399 - x32*(0.66666666666666663*x122*x337 - x399) + x398*x55;
        const scalar_t x516 = k*x464 - x32*(0.66666666666666663*x122*x341 - x464) + x463*x55;
        const scalar_t x517 = k*x126 + x123*x55 - x32*(0.66666666666666663*x123 - x126);
        const scalar_t x518 = k*x144 + x141*x55 - x32*(0.66666666666666663*x141 - x144);
        const scalar_t x519 = x122*x94;
        const scalar_t x520 = 2*x519;
        const scalar_t x521 = x125*x99;
        const scalar_t x522 = x107*x143;
        const scalar_t x523 = x521 + x522;
        const scalar_t x524 = eta*gamma*x30*x31*(-1.3333333333333335*x519 - x523) - k*(x520 + x523) - x520*x54;
        const scalar_t x525 = x122*x129;
        const scalar_t x526 = 2*x525;
        const scalar_t x527 = x125*x131;
        const scalar_t x528 = x143*x148;
        const scalar_t x529 = x527 + x528;
        const scalar_t x530 = k*(x526 + x529) + x32*(1.3333333333333335*x525 + x529) + x526*x54;
        const scalar_t x531 = POW2(x94);
        const scalar_t x532 = 2*x531;
        const scalar_t x533 = POW2(x99);
        const scalar_t x534 = POW2(x107);
        const scalar_t x535 = x533 + x534;
        const scalar_t x536 = k*x298 + x297*x55 + x32*(x113*x99 - 0.66666666666666663*x297);
        const scalar_t x537 = k*x493 + x32*(x122*x99 - 0.66666666666666663*x492) + x492*x55;
        const scalar_t x538 = k*x301 + x300*x55 + x32*(x107*x113 - 0.66666666666666663*x300);
        const scalar_t x539 = k*x496 + x32*(x107*x122 - 0.66666666666666663*x495) + x495*x55;
        const scalar_t x540 = x129*x99;
        const scalar_t x541 = x131*x94;
        const scalar_t x542 = eta*gamma*x30*x31*(-x540 + 0.66666666666666663*x541) - k*x540 - x541*x55;
        const scalar_t x543 = x107*x129;
        const scalar_t x544 = x148*x94;
        const scalar_t x545 = eta*gamma*x30*x31*(-x543 + 0.66666666666666663*x544) - k*x543 - x544*x55;
        const scalar_t x546 = k*x233 + x232*x55 + x32*(x157*x99 - 0.66666666666666663*x232);
        const scalar_t x547 = k*x385 + x32*(x323*x99 - 0.66666666666666663*x384) + x384*x55;
        const scalar_t x548 = k*x450 + x32*(x330*x99 - 0.66666666666666663*x449) + x449*x55;
        const scalar_t x549 = k*x236 + x235*x55 + x32*(x107*x157 - 0.66666666666666663*x235);
        const scalar_t x550 = k*x388 + x32*(x107*x323 - 0.66666666666666663*x387) + x387*x55;
        const scalar_t x551 = k*x453 + x32*(x107*x330 - 0.66666666666666663*x452) + x452*x55;
        const scalar_t x552 = k*x100 + x32*(x29*x99 - 0.66666666666666663*x95) + x55*x95;
        const scalar_t x553 = k*x108 + x103*x55 + x32*(-0.66666666666666663*x103 + x107*x29);
        const scalar_t x554 = x129*x94;
        const scalar_t x555 = 2*x554;
        const scalar_t x556 = x131*x99;
        const scalar_t x557 = x107*x148;
        const scalar_t x558 = x556 + x557;
        const scalar_t x559 = eta*gamma*x30*x31*(-1.3333333333333335*x554 - x558) - k*(x555 + x558) - x54*x555;
        const scalar_t x560 = x129*x293;
        const scalar_t x561 = x131*x560;
        const scalar_t x562 = incrementz[7]*x148;
        const scalar_t x563 = k*x541 + x32*(-0.66666666666666663*x540 + x541) + x540*x55;
        const scalar_t x564 = k*x544 + x32*(-0.66666666666666663*x543 + x544) + x543*x55;
        const scalar_t x565 = POW2(x129);
        const scalar_t x566 = 2*x565;
        const scalar_t x567 = POW2(x131);
        const scalar_t x568 = POW2(x148);
        const scalar_t x569 = x567 + x568;
        const scalar_t x570 = k*x309 + x308*x55 - x32*(0.66666666666666663*x308 - x309);
        const scalar_t x571 = k*x505 - x32*(0.66666666666666663*x504 - x505) + x504*x55;
        const scalar_t x572 = k*x312 + x311*x55 - x32*(0.66666666666666663*x311 - x312);
        const scalar_t x573 = k*x509 - x32*(0.66666666666666663*x508 - x509) + x508*x55;
        const scalar_t x574 = k*x245 + x244*x55 - x32*(0.66666666666666663*x244 - x245);
        const scalar_t x575 = k*x395 - x32*(0.66666666666666663*x394 - x395) + x394*x55;
        const scalar_t x576 = k*x460 - x32*(0.66666666666666663*x459 - x460) + x459*x55;
        const scalar_t x577 = k*x254 + x253*x55 - x32*(0.66666666666666663*x253 - x254);
        const scalar_t x578 = k*x402 - x32*(0.66666666666666663*x401 - x402) + x401*x55;
        const scalar_t x579 = k*x467 - x32*(0.66666666666666663*x466 - x467) + x466*x55;
        const scalar_t x580 = k*x132 + x130*x55 - x32*(0.66666666666666663*x130 - x132);
        const scalar_t x581 = k*x149 + x147*x55 - x32*(0.66666666666666663*x147 - x149);
        const scalar_t x582 = x14*x39*x9 + x25*x33*x43;
        const scalar_t x583 = x39*x53;
        const scalar_t x584 = x55*x9;
        const scalar_t x585 = eta*gamma*x30*x31*(-x583 + 0.66666666666666663*x73*x9) - k*x583 - x584*x73;
        const scalar_t x586 = x39*x67;
        const scalar_t x587 = eta*gamma*x30*x31*(-x586 + 0.66666666666666663*x79*x9) - k*x586 - x584*x79;
        const scalar_t x588 = 2*x84;
        const scalar_t x589 = x82 + x85;
        const scalar_t x590 = x43*x99;
        const scalar_t x591 = x107*x25;
        const scalar_t x592 = k*x590 + x32*(x590 - 0.66666666666666663*x591) + x55*x591;
        const scalar_t x593 = x116*x43;
        const scalar_t x594 = x137*x25;
        const scalar_t x595 = k*x593 - x32*(0.66666666666666663*x137*x25 - x593) + x55*x594;
        const scalar_t x596 = x125*x43;
        const scalar_t x597 = x143*x25;
        const scalar_t x598 = k*x596 - x32*(0.66666666666666663*x143*x25 - x596) + x55*x597;
        const scalar_t x599 = x131*x43;
        const scalar_t x600 = x148*x25;
        const scalar_t x601 = k*x599 - x32*(0.66666666666666663*x148*x25 - x599) + x55*x600;
        const scalar_t x602 = k*x154;
        const scalar_t x603 = x159*x43;
        const scalar_t x604 = x162*x584 + x32*(0.66666666666666663*x164*x25 - x603) + x39*x602;
        const scalar_t x605 = 2*x169;
        const scalar_t x606 = x167 + x170;
        const scalar_t x607 = eta*gamma*x30*x31*(-1.3333333333333335*x169 - x606) - k*(x605 + x606) - x54*x605;
        const scalar_t x608 = 2*x176;
        const scalar_t x609 = x174 + x177;
        const scalar_t x610 = eta*gamma*x30*x31*(-1.3333333333333335*x176 - x609) - k*(x608 + x609) - x54*x608;
        const scalar_t x611 = 2*x183;
        const scalar_t x612 = x181 + x184;
        const scalar_t x613 = eta*gamma*x30*x31*(-1.3333333333333335*x183 - x612) - k*(x611 + x612) - x54*x611;
        const scalar_t x614 = x188 + x190;
        const scalar_t x615 = k*(2*x189 + x614) + x32*(1.3333333333333335*x189 + x614) + x584*x99;
        const scalar_t x616 = x194 + x196;
        const scalar_t x617 = k*(2*x195 + x616) + x116*x584 - x32*(-1.3333333333333335*x195 - x616);
        const scalar_t x618 = x200 + x202;
        const scalar_t x619 = k*(2*x201 + x618) + x125*x584 - x32*(-1.3333333333333335*x201 - x618);
        const scalar_t x620 = x206 + x208;
        const scalar_t x621 = k*(2*x207 + x620) + x131*x584 - x32*(-1.3333333333333335*x207 - x620);
        const scalar_t x622 = x14*x154*x162 + x159*x164*x33;
        const scalar_t x623 = 2*x219;
        const scalar_t x624 = x217 + x220;
        const scalar_t x625 = x162*x53;
        const scalar_t x626 = x154*x55;
        const scalar_t x627 = 0.66666666666666663*x154;
        const scalar_t x628 = k*x625 + x32*(x625 - x627*x73) + x626*x73;
        const scalar_t x629 = x162*x67;
        const scalar_t x630 = k*x629 + x32*(-x627*x79 + x629) + x626*x79;
        const scalar_t x631 = x164*x99;
        const scalar_t x632 = x107*x159;
        const scalar_t x633 = k*x631 + x32*(x631 - 0.66666666666666663*x632) + x55*x632;
        const scalar_t x634 = x116*x164;
        const scalar_t x635 = x137*x159;
        const scalar_t x636 = k*x634 - x32*(0.66666666666666663*x137*x159 - x634) + x55*x635;
        const scalar_t x637 = x125*x164;
        const scalar_t x638 = x143*x159;
        const scalar_t x639 = k*x637 - x32*(0.66666666666666663*x143*x159 - x637) + x55*x638;
        const scalar_t x640 = x131*x164;
        const scalar_t x641 = x148*x159;
        const scalar_t x642 = k*x640 - x32*(0.66666666666666663*x148*x159 - x640) + x55*x641;
        const scalar_t x643 = k*x9;
        const scalar_t x644 = x162*x643 + x32*(-x164*x25 + 0.66666666666666663*x603) + x39*x626;
        const scalar_t x645 = 2*x263;
        const scalar_t x646 = x261 + x264;
        const scalar_t x647 = k*(x645 + x646) + x32*(1.3333333333333335*x263 + x646) + x54*x645;
        const scalar_t x648 = 2*x269;
        const scalar_t x649 = x267 + x270;
        const scalar_t x650 = k*(x648 + x649) + x32*(1.3333333333333335*x269 + x649) + x54*x648;
        const scalar_t x651 = x273 + x275;
        const scalar_t x652 = -k*(2*x274 + x651) + 2*x154*x54*x99 - x32*(1.3333333333333335*x274 + x651);
        const scalar_t x653 = x278 + x280;
        const scalar_t x654 = k*(2*x279 + x653) - x116*x626 - x32*(-1.3333333333333335*x279 - x653);
        const scalar_t x655 = x283 + x285;
        const scalar_t x656 = k*(2*x284 + x655) - x125*x626 - x32*(-1.3333333333333335*x284 - x655);
        const scalar_t x657 = x288 + x290;
        const scalar_t x658 = k*(2*x289 + x657) - x131*x626 - x32*(-1.3333333333333335*x289 - x657);
        const scalar_t x659 = x116*x293;
        const scalar_t x660 = x137*x99;
        const scalar_t x661 = x107*x116;
        const scalar_t x662 = eta*gamma*x30*x31*(0.66666666666666663*x107*x116 - x660) - k*x660 - x55*x661;
        const scalar_t x663 = 2*x305;
        const scalar_t x664 = x303 + x306;
        const scalar_t x665 = x131*x137;
        const scalar_t x666 = x116*x148;
        const scalar_t x667 = k*x665 + x32*(x665 - 0.66666666666666663*x666) + x55*x666;
        const scalar_t x668 = x125*x137;
        const scalar_t x669 = x116*x143;
        const scalar_t x670 = k*x668 - x32*(-x668 + 0.66666666666666663*x669) + x55*x669;
        const scalar_t x671 = k*x635 - x32*(0.66666666666666663*x634 - x635) + x55*x634;
        const scalar_t x672 = x137*x325;
        const scalar_t x673 = x116*x337;
        const scalar_t x674 = k*x672 - x32*(0.66666666666666663*x116*x337 - x672) + x55*x673;
        const scalar_t x675 = x137*x332;
        const scalar_t x676 = x116*x341;
        const scalar_t x677 = k*x675 - x32*(-x675 + 0.66666666666666663*x676) + x55*x676;
        const scalar_t x678 = k*x594 - x32*(0.66666666666666663*x593 - x594) + x55*x593;
        const scalar_t x679 = 2*x348;
        const scalar_t x680 = x346 + x349;
        const scalar_t x681 = eta*gamma*x30*x31*(-1.3333333333333335*x348 - x680) - k*(x679 + x680) - x54*x679;
        const scalar_t x682 = 2*x354;
        const scalar_t x683 = x352 + x355;
        const scalar_t x684 = k*(x682 + x683) + x32*(1.3333333333333335*x354 + x683) + x54*x682;
        const scalar_t x685 = 2*x360;
        const scalar_t x686 = x358 + x361;
        const scalar_t x687 = k*(x685 + x686) + x32*(1.3333333333333335*x360 + x686) + x54*x685;
        const scalar_t x688 = x116*x55;
        const scalar_t x689 = x365 + x367;
        const scalar_t x690 = k*(2*x366 + x689) - x32*(-1.3333333333333335*x366 - x689) - x53*x688;
        const scalar_t x691 = x370 + x372;
        const scalar_t x692 = k*(2*x371 + x691) - x32*(-1.3333333333333335*x371 - x691) - x67*x688;
        const scalar_t x693 = x14*x53*x73 + x325*x33*x337;
        const scalar_t x694 = 2*x381;
        const scalar_t x695 = x379 + x382;
        const scalar_t x696 = x337*x99;
        const scalar_t x697 = x107*x325;
        const scalar_t x698 = k*x696 + x32*(x696 - 0.66666666666666663*x697) + x55*x697;
        const scalar_t x699 = k*x673 - x32*(0.66666666666666663*x672 - x673) + x55*x672;
        const scalar_t x700 = x125*x337;
        const scalar_t x701 = x143*x325;
        const scalar_t x702 = k*x700 - x32*(-x700 + 0.66666666666666663*x701) + x55*x701;
        const scalar_t x703 = x131*x337;
        const scalar_t x704 = x148*x325;
        const scalar_t x705 = k*x703 - x32*(0.66666666666666663*x148*x325 - x703) + x55*x704;
        const scalar_t x706 = 0.66666666666666663*x325;
        const scalar_t x707 = -x32*(-x159*x337 + x164*x706) + x55*x625 + x602*x73;
        const scalar_t x708 = x67*x73;
        const scalar_t x709 = x53*x79;
        const scalar_t x710 = k*x708 - x32*(-x332*x337 + x341*x706) + x55*x709;
        const scalar_t x711 = x32*(-x25*x337 + x43*x706) + x55*x583 + x643*x73;
        const scalar_t x712 = 2*x416;
        const scalar_t x713 = x414 + x417;
        const scalar_t x714 = k*(x712 + x713) + x32*(1.3333333333333335*x416 + x713) + x54*x712;
        const scalar_t x715 = x420 + x422;
        const scalar_t x716 = -k*(2*x421 + x715) - x32*(1.3333333333333335*x421 + x715) + 2*x53*x54*x99;
        const scalar_t x717 = x53*x55;
        const scalar_t x718 = x426 + x428;
        const scalar_t x719 = k*(2*x427 + x718) - x125*x717 - x32*(-1.3333333333333335*x427 - x718);
        const scalar_t x720 = x431 + x433;
        const scalar_t x721 = k*(2*x432 + x720) - x131*x717 - x32*(-1.3333333333333335*x432 - x720);
        const scalar_t x722 = x14*x67*x79 + x33*x332*x341;
        const scalar_t x723 = 2*x442;
        const scalar_t x724 = x440 + x443;
        const scalar_t x725 = k*x709 + x32*(-0.66666666666666663*x708 + x709) + x55*x708;
        const scalar_t x726 = x341*x99;
        const scalar_t x727 = x107*x332;
        const scalar_t x728 = k*x726 + x32*(x726 - 0.66666666666666663*x727) + x55*x727;
        const scalar_t x729 = k*x676 - x32*(0.66666666666666663*x137*x332 - x676) + x55*x675;
        const scalar_t x730 = x125*x341;
        const scalar_t x731 = x143*x332;
        const scalar_t x732 = k*x730 - x32*(-x730 + 0.66666666666666663*x731) + x55*x731;
        const scalar_t x733 = x131*x341;
        const scalar_t x734 = x148*x332;
        const scalar_t x735 = k*x733 - x32*(0.66666666666666663*x148*x332 - x733) + x55*x734;
        const scalar_t x736 = 0.66666666666666663*x332;
        const scalar_t x737 = -x32*(-x159*x341 + x164*x736) + x55*x629 + x602*x79;
        const scalar_t x738 = x32*(-x25*x341 + x43*x736) + x55*x586 + x643*x79;
        const scalar_t x739 = x474 + x476;
        const scalar_t x740 = -k*(2*x475 + x739) - x32*(1.3333333333333335*x475 + x739) + 2*x54*x67*x99;
        const scalar_t x741 = x55*x67;
        const scalar_t x742 = x479 + x481;
        const scalar_t x743 = k*(2*x480 + x742) - x125*x741 - x32*(-1.3333333333333335*x480 - x742);
        const scalar_t x744 = x484 + x486;
        const scalar_t x745 = k*(2*x485 + x744) - x131*x741 - x32*(-1.3333333333333335*x485 - x744);
        const scalar_t x746 = x125*x293;
        const scalar_t x747 = x143*x99;
        const scalar_t x748 = x107*x125;
        const scalar_t x749 = eta*gamma*x30*x31*(0.66666666666666663*x107*x125 - x747) - k*x747 - x55*x748;
        const scalar_t x750 = 2*x500;
        const scalar_t x751 = x498 + x501;
        const scalar_t x752 = k*x669 + x32*(-0.66666666666666663*x668 + x669) + x55*x668;
        const scalar_t x753 = x131*x143;
        const scalar_t x754 = x125*x148;
        const scalar_t x755 = k*x753 + x32*(x753 - 0.66666666666666663*x754) + x55*x754;
        const scalar_t x756 = k*x638 - x32*(0.66666666666666663*x637 - x638) + x55*x637;
        const scalar_t x757 = k*x701 - x32*(0.66666666666666663*x125*x337 - x701) + x55*x700;
        const scalar_t x758 = k*x731 - x32*(0.66666666666666663*x125*x341 - x731) + x55*x730;
        const scalar_t x759 = k*x597 - x32*(0.66666666666666663*x596 - x597) + x55*x596;
        const scalar_t x760 = 2*x521;
        const scalar_t x761 = x519 + x522;
        const scalar_t x762 = eta*gamma*x30*x31*(-1.3333333333333335*x521 - x761) - k*(x760 + x761) - x54*x760;
        const scalar_t x763 = 2*x527;
        const scalar_t x764 = x525 + x528;
        const scalar_t x765 = k*(x763 + x764) + x32*(1.3333333333333335*x527 + x764) + x54*x763;
        const scalar_t x766 = 2*x533;
        const scalar_t x767 = x531 + x534;
        const scalar_t x768 = k*x661 + x32*(x107*x116 - 0.66666666666666663*x660) + x55*x660;
        const scalar_t x769 = k*x748 + x32*(x107*x125 - 0.66666666666666663*x747) + x55*x747;
        const scalar_t x770 = x107*x131;
        const scalar_t x771 = x148*x99;
        const scalar_t x772 = eta*gamma*x30*x31*(-x770 + 0.66666666666666663*x771) - k*x770 - x55*x771;
        const scalar_t x773 = k*x632 + x32*(x107*x159 - 0.66666666666666663*x631) + x55*x631;
        const scalar_t x774 = k*x697 + x32*(x107*x325 - 0.66666666666666663*x696) + x55*x696;
        const scalar_t x775 = k*x727 + x32*(x107*x332 - 0.66666666666666663*x726) + x55*x726;
        const scalar_t x776 = k*x591 + x32*(x107*x25 - 0.66666666666666663*x590) + x55*x590;
        const scalar_t x777 = 2*x556;
        const scalar_t x778 = x554 + x557;
        const scalar_t x779 = eta*gamma*x30*x31*(-1.3333333333333335*x556 - x778) - k*(x777 + x778) - x54*x777;
        const scalar_t x780 = x131*x293;
        const scalar_t x781 = k*x771 + x32*(-0.66666666666666663*x770 + x771) + x55*x770;
        const scalar_t x782 = 2*x567;
        const scalar_t x783 = x565 + x568;
        const scalar_t x784 = k*x666 - x32*(0.66666666666666663*x665 - x666) + x55*x665;
        const scalar_t x785 = k*x754 - x32*(0.66666666666666663*x753 - x754) + x55*x753;
        const scalar_t x786 = k*x641 - x32*(0.66666666666666663*x640 - x641) + x55*x640;
        const scalar_t x787 = k*x704 - x32*(0.66666666666666663*x703 - x704) + x55*x703;
        const scalar_t x788 = k*x734 - x32*(0.66666666666666663*x733 - x734) + x55*x733;
        const scalar_t x789 = k*x600 - x32*(0.66666666666666663*x599 - x600) + x55*x599;
        const scalar_t x790 = 2*x85;
        const scalar_t x791 = x82 + x84;
        const scalar_t x792 = 2*x170;
        const scalar_t x793 = x167 + x169;
        const scalar_t x794 = eta*gamma*x30*x31*(-1.3333333333333335*x170 - x793) - k*(x792 + x793) - x54*x792;
        const scalar_t x795 = 2*x177;
        const scalar_t x796 = x174 + x176;
        const scalar_t x797 = eta*gamma*x30*x31*(-1.3333333333333335*x177 - x796) - k*(x795 + x796) - x54*x795;
        const scalar_t x798 = 2*x184;
        const scalar_t x799 = x181 + x183;
        const scalar_t x800 = eta*gamma*x30*x31*(-1.3333333333333335*x184 - x799) - k*(x798 + x799) - x54*x798;
        const scalar_t x801 = x39*x55;
        const scalar_t x802 = x188 + x189;
        const scalar_t x803 = k*(2*x190 + x802) + x107*x801 + x32*(1.3333333333333335*x190 + x802);
        const scalar_t x804 = x194 + x195;
        const scalar_t x805 = k*(2*x196 + x804) + x137*x801 - x32*(-1.3333333333333335*x196 - x804);
        const scalar_t x806 = x200 + x201;
        const scalar_t x807 = k*(2*x202 + x806) + x143*x801 - x32*(-1.3333333333333335*x202 - x806);
        const scalar_t x808 = x206 + x207;
        const scalar_t x809 = k*(2*x208 + x808) + x148*x801 - x32*(-1.3333333333333335*x208 - x808);
        const scalar_t x810 = 2*x220;
        const scalar_t x811 = x217 + x219;
        const scalar_t x812 = 2*x264;
        const scalar_t x813 = x261 + x263;
        const scalar_t x814 = k*(x812 + x813) + x32*(1.3333333333333335*x264 + x813) + x54*x812;
        const scalar_t x815 = 2*x270;
        const scalar_t x816 = x267 + x269;
        const scalar_t x817 = k*(x815 + x816) + x32*(1.3333333333333335*x270 + x816) + x54*x815;
        const scalar_t x818 = x273 + x274;
        const scalar_t x819 = -k*(2*x275 + x818) + 2*x107*x162*x54 - x32*(1.3333333333333335*x275 + x818);
        const scalar_t x820 = x162*x55;
        const scalar_t x821 = x278 + x279;
        const scalar_t x822 = k*(2*x280 + x821) - x137*x820 - x32*(-1.3333333333333335*x280 - x821);
        const scalar_t x823 = x283 + x284;
        const scalar_t x824 = k*(2*x285 + x823) - x143*x820 - x32*(-1.3333333333333335*x285 - x823);
        const scalar_t x825 = x288 + x289;
        const scalar_t x826 = k*(2*x290 + x825) - x148*x820 - x32*(-1.3333333333333335*x290 - x825);
        const scalar_t x827 = 2*x306;
        const scalar_t x828 = x303 + x305;
        const scalar_t x829 = 2*x349;
        const scalar_t x830 = x346 + x348;
        const scalar_t x831 = eta*gamma*x30*x31*(-1.3333333333333335*x349 - x830) - k*(x829 + x830) - x54*x829;
        const scalar_t x832 = 2*x355;
        const scalar_t x833 = x352 + x354;
        const scalar_t x834 = k*(x832 + x833) + x32*(1.3333333333333335*x355 + x833) + x54*x832;
        const scalar_t x835 = 2*x361;
        const scalar_t x836 = x358 + x360;
        const scalar_t x837 = k*(x835 + x836) + x32*(1.3333333333333335*x361 + x836) + x54*x835;
        const scalar_t x838 = x137*x55;
        const scalar_t x839 = x365 + x366;
        const scalar_t x840 = k*(2*x367 + x839) - x32*(-1.3333333333333335*x367 - x839) - x73*x838;
        const scalar_t x841 = x370 + x371;
        const scalar_t x842 = k*(2*x372 + x841) - x32*(-1.3333333333333335*x372 - x841) - x79*x838;
        const scalar_t x843 = 2*x382;
        const scalar_t x844 = x379 + x381;
        const scalar_t x845 = 2*x417;
        const scalar_t x846 = x414 + x416;
        const scalar_t x847 = k*(x845 + x846) + x32*(1.3333333333333335*x417 + x846) + x54*x845;
        const scalar_t x848 = x420 + x421;
        const scalar_t x849 = -k*(2*x422 + x848) + 2*x107*x54*x73 - x32*(1.3333333333333335*x422 + x848);
        const scalar_t x850 = x55*x73;
        const scalar_t x851 = x426 + x427;
        const scalar_t x852 = k*(2*x428 + x851) - x143*x850 - x32*(-1.3333333333333335*x428 - x851);
        const scalar_t x853 = x431 + x432;
        const scalar_t x854 = k*(2*x433 + x853) - x148*x850 - x32*(-1.3333333333333335*x433 - x853);
        const scalar_t x855 = 2*x443;
        const scalar_t x856 = x440 + x442;
        const scalar_t x857 = x474 + x475;
        const scalar_t x858 = -k*(2*x476 + x857) + 2*x107*x54*x79 - x32*(1.3333333333333335*x476 + x857);
        const scalar_t x859 = x55*x79;
        const scalar_t x860 = x479 + x480;
        const scalar_t x861 = k*(2*x481 + x860) - x143*x859 - x32*(-1.3333333333333335*x481 - x860);
        const scalar_t x862 = x484 + x485;
        const scalar_t x863 = k*(2*x486 + x862) - x148*x859 - x32*(-1.3333333333333335*x486 - x862);
        const scalar_t x864 = 2*x501;
        const scalar_t x865 = x498 + x500;
        const scalar_t x866 = 2*x522;
        const scalar_t x867 = x519 + x521;
        const scalar_t x868 = eta*gamma*x30*x31*(-1.3333333333333335*x522 - x867) - k*(x866 + x867) - x54*x866;
        const scalar_t x869 = 2*x528;
        const scalar_t x870 = x525 + x527;
        const scalar_t x871 = k*(x869 + x870) + x32*(1.3333333333333335*x528 + x870) + x54*x869;
        const scalar_t x872 = 2*x534;
        const scalar_t x873 = x531 + x533;
        const scalar_t x874 = 2*x557;
        const scalar_t x875 = x554 + x556;
        const scalar_t x876 = eta*gamma*x30*x31*(-1.3333333333333335*x557 - x875) - k*(x874 + x875) - x54*x874;
        const scalar_t x877 = 2*x568;
        const scalar_t x878 = x565 + x567;
        outx[0] += x212*(incrementy[0]*x35 + incrementz[0]*x44 - x101*x102 - x109*x110 + x118*x119 + x127*x128 + 
        x133*x134 + x139*x140 + x145*x146 + x150*x151 - x160*x161 - x165*x166 + x172*x173 + x179*x180 + 
        x186*x187 - x192*x193 + x198*x199 + x204*x205 + x210*x211 + x57*x58 + x68*x69 + x74*x75 + x80*x81 + 
        x87*(k*(x83 + x86) + x32*(1.3333333333333335*x82 + x86) + x54*x83));
        outx[1] += x212*(incrementy[1]*x215 + incrementz[1]*x216 - x102*x234 - x110*x237 + x119*x240 + x128*x243 
        + x134*x246 + x140*x249 + x146*x252 + x151*x255 + x172*x87 + x173*(k*(x218 + x221) + x218*x54 + 
        x32*(1.3333333333333335*x217 + x221)) + x180*x266 + x187*x272 + x193*x277 + x199*x282 + x205*x287 + 
        x211*x292 + x225*x58 + x227*x69 + x229*x75 + x231*x81 - x257*x258 - x259*x260);
        outx[2] += x212*(incrementy[2]*x295 + x102*x299 + x110*x302 + x128*x316 + x134*x310 + x146*x319 + 
        x151*x313 + x161*x320 + x166*x335 + x173*x282 + x180*x369 + x187*x374 + x193*x351 + x198*x87 + 
        x199*(k*(x304 + x307) + x304*x54 + x32*(1.3333333333333335*x303 + x307)) + x205*x357 + x211*x363 + 
        x258*x344 + x260*x345 + x294*x296 + x327*x58 + x334*x69 + x339*x75 + x343*x81);
        outx[3] += x212*(incrementy[3]*x377 + incrementz[3]*x378 - x102*x386 - x110*x389 + x119*x390 + x128*x393 
        + x134*x396 + x140*x397 + x146*x400 + x151*x403 + x161*x405 + x166*x409 + x173*x266 + x179*x87 + 
        x180*(k*(x380 + x383) + x32*(1.3333333333333335*x379 + x383) + x380*x54) + x187*x419 + x193*x424 + 
        x199*x369 + x205*x430 + x211*x435 - x258*x412 - x260*x413 + x408*x69 + x411*x81);
        outx[4] += x212*(incrementy[4]*x438 + incrementz[4]*x439 - x102*x451 - x110*x454 + x119*x455 + x128*x458 
        + x134*x461 + x140*x462 + x146*x465 + x151*x468 + x161*x470 + x166*x471 + x173*x272 + x180*x419 + 
        x186*x87 + x187*(k*(x441 + x444) + x32*(1.3333333333333335*x440 + x444) + x441*x54) + x193*x478 + 
        x199*x374 + x205*x483 + x211*x488 - x258*x472 - x260*x473 + x447*x58 + x448*x75);
        outx[5] += x212*(incrementy[5]*x490 + x102*x494 + x110*x497 + x119*x503 + x134*x506 + x140*x507 + 
        x151*x510 + x161*x511 + x166*x514 + x173*x287 + x180*x430 + x187*x483 + x193*x524 + x199*x357 + x204*x87
        + x205*(k*(x499 + x502) + x32*(1.3333333333333335*x498 + x502) + x499*x54) + x211*x530 + x258*x517 + 
        x260*x518 + x489*x491 + x512*x58 + x513*x69 + x515*x75 + x516*x81);
        outx[6] += x212*((1.0/2.0)*incrementx[1]*x277 + (1.0/2.0)*incrementx[2]*x351 + 
        (1.0/2.0)*incrementx[3]*x424 + (1.0/2.0)*incrementx[4]*x478 + (1.0/2.0)*incrementx[5]*x524 + 
        (1.0/2.0)*incrementx[6]*(k*(x532 + x535) + x32*(1.3333333333333335*x531 + x535) + x532*x54) + 
        (1.0/2.0)*incrementx[7]*x559 + incrementy[6]*x293*x94*x99 + (1.0/2.0)*incrementy[7]*x542 + 
        incrementz[6]*x107*x293*x94 + (1.0/2.0)*incrementz[7]*x545 - x119*x536 - x128*x537 - x140*x538 - 
        x146*x539 - x161*x546 - x166*x549 - x192*x87 - x258*x552 - x260*x553 - x547*x58 - x548*x69 - x550*x75 - 
        x551*x81);
        outx[7] += x212*(incrementy[7]*x561 - x102*x563 - x110*x564 + x119*x570 + x128*x571 + x140*x572 + 
        x146*x573 + x161*x574 + x166*x577 + x173*x292 + x180*x435 + x187*x488 + x193*x559 + x199*x363 + 
        x205*x530 + x210*x87 + x211*(k*(x566 + x569) + x32*(1.3333333333333335*x565 + x569) + x54*x566) + 
        x258*x580 + x260*x581 + x560*x562 + x575*x58 + x576*x69 + x578*x75 + x579*x81);
        outy[0] += x212*(incrementx[0]*x35 + incrementz[0]*x582 - x102*x615 - x110*x592 + x119*x617 + x128*x619 +
        x134*x621 + x140*x595 + x146*x598 + x151*x601 + x161*x607 - x166*x604 - x173*x257 - x180*x412 - 
        x187*x472 - x193*x552 + x199*x344 + x205*x517 + x211*x580 + x258*(k*(x588 + x589) + x32*(x589 + 
        1.3333333333333335*x84) + x54*x588) + x58*x610 + x585*x75 + x587*x81 + x613*x69);
        outy[1] += x212*(incrementx[1]*x215 + incrementz[1]*x622 + x102*x652 - x110*x633 + x119*x654 + x128*x656 
        + x134*x658 + x140*x636 + x146*x639 + x151*x642 - x160*x87 + x161*(k*(x623 + x624) + 
        x32*(1.3333333333333335*x219 + x624) + x54*x623) + x180*x405 + x187*x470 - x193*x546 + x199*x320 + 
        x205*x511 + x211*x574 + x258*x607 - x260*x644 + x58*x647 + x628*x75 + x630*x81 + x650*x69);
        outy[2] += x212*(incrementx[2]*x295 + x102*x681 + x110*x662 + x118*x87 + x119*(k*(x663 + x664) + 
        x32*(1.3333333333333335*x305 + x664) + x54*x663) + x128*x684 + x134*x687 + x146*x670 + x151*x667 + 
        x161*x654 + x166*x671 + x173*x240 + x180*x390 + x187*x455 - x193*x536 + x205*x503 + x211*x570 + 
        x258*x617 + x260*x678 + x296*x659 + x58*x690 + x674*x75 + x677*x81 + x69*x692);
        outy[3] += x212*(incrementx[3]*x377 + incrementz[3]*x693 + x102*x716 - x110*x698 + x119*x690 + x128*x719 
        + x134*x721 + x140*x699 + x146*x702 + x151*x705 + x161*x647 + x166*x707 + x173*x225 + x187*x447 - 
        x193*x547 + x199*x327 + x205*x512 + x211*x575 + x258*x610 - x260*x711 + x57*x87 + x58*(k*(x694 + x695) +
        x32*(1.3333333333333335*x381 + x695) + x54*x694) + x69*x714 + x710*x81);
        outy[4] += x212*(incrementx[4]*x438 + incrementz[4]*x722 + x102*x740 - x110*x728 + x119*x692 + x128*x743 
        + x134*x745 + x140*x729 + x146*x732 + x151*x735 + x161*x650 + x166*x737 + x173*x227 + x180*x408 - 
        x193*x548 + x199*x334 + x205*x513 + x211*x576 + x258*x613 - x260*x738 + x58*x714 + x68*x87 + 
        x69*(k*(x723 + x724) + x32*(1.3333333333333335*x442 + x724) + x54*x723) + x725*x75);
        outy[5] += x212*(incrementx[5]*x490 + x102*x762 + x110*x749 + x119*x684 + x127*x87 + x128*(k*(x750 + 
        x751) + x32*(1.3333333333333335*x500 + x751) + x54*x750) + x134*x765 + x140*x752 + x151*x755 + x161*x656
        + x166*x756 + x173*x243 + x180*x393 + x187*x458 - x193*x537 + x199*x316 + x211*x571 + x258*x619 + 
        x260*x759 + x491*x746 + x58*x719 + x69*x743 + x75*x757 + x758*x81);
        outy[6] += x212*((1.0/2.0)*incrementx[2]*x299 + (1.0/2.0)*incrementx[5]*x494 + incrementx[6]*x293*x94*x99
        + (1.0/2.0)*incrementy[1]*x652 + (1.0/2.0)*incrementy[2]*x681 + (1.0/2.0)*incrementy[3]*x716 + 
        (1.0/2.0)*incrementy[4]*x740 + (1.0/2.0)*incrementy[5]*x762 + (1.0/2.0)*incrementy[6]*(k*(x766 + x767) +
        x32*(1.3333333333333335*x533 + x767) + x54*x766) + (1.0/2.0)*incrementy[7]*x779 + 
        incrementz[6]*x107*x293*x99 + (1.0/2.0)*incrementz[7]*x772 - x101*x87 - x140*x768 - x146*x769 - 
        x166*x773 - x173*x234 - x180*x386 - x187*x451 - x211*x563 - x258*x615 - x260*x776 - x75*x774 - 
        x775*x81);
        outy[7] += x212*(incrementx[7]*x561 + x102*x779 - x110*x781 + x119*x687 + x128*x765 + x133*x87 + 
        x134*(k*(x782 + x783) + x32*(1.3333333333333335*x567 + x783) + x54*x782) + x140*x784 + x146*x785 + 
        x161*x658 + x166*x786 + x173*x246 + x180*x396 + x187*x461 + x193*x542 + x199*x310 + x205*x506 + 
        x258*x621 + x260*x789 + x562*x780 + x58*x721 + x69*x745 + x75*x787 + x788*x81);
        outz[0] += x212*(incrementx[0]*x44 + incrementy[0]*x582 - x102*x776 - x110*x803 + x119*x678 + x128*x759 +
        x134*x789 + x140*x805 + x146*x807 + x151*x809 - x161*x644 + x166*x794 - x173*x259 - x180*x413 - 
        x187*x473 - x193*x553 + x199*x345 + x205*x518 + x211*x581 + x260*(k*(x790 + x791) + x32*(x791 + 
        1.3333333333333335*x85) + x54*x790) - x58*x711 - x69*x738 + x75*x797 + x800*x81);
        outz[1] += x212*(incrementx[1]*x216 + incrementy[1]*x622 - x102*x773 + x110*x819 + x119*x671 + x128*x756 
        + x134*x786 + x140*x822 + x146*x824 + x151*x826 - x165*x87 + x166*(k*(x810 + x811) + 
        x32*(1.3333333333333335*x220 + x811) + x54*x810) + x180*x409 + x187*x471 - x193*x549 + x199*x335 + 
        x205*x514 + x211*x577 - x258*x604 + x260*x794 + x58*x707 + x69*x737 + x75*x814 + x81*x817);
        outz[2] += x212*(incrementx[2]*x137*x294 + incrementy[2]*x137*x659 - x102*x768 + x110*x831 + x128*x752 + 
        x134*x784 + x139*x87 + x140*(k*(x827 + x828) + x32*(1.3333333333333335*x306 + x828) + x54*x827) + 
        x146*x834 + x151*x837 + x161*x636 + x166*x822 + x173*x249 + x180*x397 + x187*x462 - x193*x538 + 
        x205*x507 + x211*x572 + x258*x595 + x260*x805 + x58*x699 + x69*x729 + x75*x840 + x81*x842);
        outz[3] += x212*(incrementx[3]*x378 + incrementy[3]*x693 - x102*x774 + x110*x849 + x119*x674 + x128*x757 
        + x134*x787 + x140*x840 + x146*x852 + x151*x854 + x161*x628 + x166*x814 + x173*x229 + x187*x448 - 
        x193*x550 + x199*x339 + x205*x515 + x211*x578 + x258*x585 + x260*x797 + x69*x725 + x74*x87 + 
        x75*(k*(x843 + x844) + x32*(1.3333333333333335*x382 + x844) + x54*x843) + x81*x847);
        outz[4] += x212*(incrementx[4]*x439 + incrementy[4]*x722 - x102*x775 + x110*x858 + x119*x677 + x128*x758 
        + x134*x788 + x140*x842 + x146*x861 + x151*x863 + x161*x630 + x166*x817 + x173*x231 + x180*x411 - 
        x193*x551 + x199*x343 + x205*x516 + x211*x579 + x258*x587 + x260*x800 + x58*x710 + x75*x847 + x80*x87 + 
        x81*(k*(x855 + x856) + x32*(1.3333333333333335*x443 + x856) + x54*x855));
        outz[5] += x212*(incrementx[5]*x143*x489 + incrementy[5]*x143*x746 - x102*x769 + x110*x868 + x119*x670 + 
        x134*x785 + x140*x834 + x145*x87 + x146*(k*(x864 + x865) + x32*(1.3333333333333335*x501 + x865) + 
        x54*x864) + x151*x871 + x161*x639 + x166*x824 + x173*x252 + x180*x400 + x187*x465 - x193*x539 + 
        x199*x319 + x211*x573 + x258*x598 + x260*x807 + x58*x702 + x69*x732 + x75*x852 + x81*x861);
        outz[6] += x212*(incrementx[6]*x107*x293*x94 + incrementy[6]*x107*x293*x99 - x109*x87 + x110*(k*(x872 + 
        x873) + x32*(1.3333333333333335*x534 + x873) + x54*x872) + x119*x662 + x128*x749 - x134*x781 + x140*x831
        + x146*x868 + x151*x876 - x161*x633 + x166*x819 - x173*x237 - x180*x389 - x187*x454 + x199*x302 + 
        x205*x497 - x211*x564 - x258*x592 - x260*x803 - x58*x698 - x69*x728 + x75*x849 + x81*x858);
        outz[7] += x212*(incrementx[7]*x148*x560 + incrementy[7]*x148*x780 + x102*x772 + x110*x876 + x119*x667 + 
        x128*x755 + x140*x837 + x146*x871 + x150*x87 + x151*(k*(x877 + x878) + x32*(1.3333333333333335*x568 + 
        x878) + x54*x877) + x161*x642 + x166*x826 + x173*x255 + x180*x403 + x187*x468 + x193*x545 + x199*x313 + 
        x205*x510 + x258*x601 + x260*x809 + x58*x705 + x69*x735 + x75*x854 + x81*x863);

}


static SFEM_INLINE void hex8_kelvin_voigt_newmark_gradient_adj(const scalar_t                      k,
                                                         const scalar_t                      K,
                                                         const scalar_t                      eta,

                                                         const scalar_t *const SFEM_RESTRICT adjugate,
                                                         const scalar_t                      jacobian_determinant,

                                                         const scalar_t                      qx,
                                                         const scalar_t                      qy,
                                                         const scalar_t                      qz,
                                                         const scalar_t                      qw,

                                                         const scalar_t *const SFEM_RESTRICT ux,
                                                         const scalar_t *const SFEM_RESTRICT uy,
                                                         const scalar_t *const SFEM_RESTRICT uz,

                                                         const scalar_t *const SFEM_RESTRICT vx,
                                                         const scalar_t *const SFEM_RESTRICT vy,
                                                         const scalar_t *const SFEM_RESTRICT vz,

                                                         accumulator_t *const SFEM_RESTRICT  outx,
                                                         accumulator_t *const SFEM_RESTRICT  outy,
                                                         accumulator_t *const SFEM_RESTRICT  outz){

        scalar_t disp_grad[9];
        hex8_displacement_gradient(adjugate, jacobian_determinant, qx, qy, qz, ux, uy, uz, disp_grad); 

        scalar_t velo_grad[9];
        hex8_velocity_gradient(adjugate, jacobian_determinant, qx, qy, qz, vx, vy, vz, velo_grad);

        // scalar_t acce_vec[3];
        // hex8_acceleration_vector(qx, qy, qz, ax, ay, az, acce_vec);                                                      
        
        const scalar_t x0 = qy - 1;
        const scalar_t x1 = -x0;
        const scalar_t x2 = qz - 1;
        const scalar_t x3 = -x2;
        const scalar_t x4 = x1*x3;
        const scalar_t x5 = adjugate[1]*x4;
        const scalar_t x6 = qx - 1;
        const scalar_t x7 = -x6;
        const scalar_t x8 = x3*x7;
        const scalar_t x9 = adjugate[4]*x8;
        const scalar_t x10 = x1*x7;
        const scalar_t x11 = adjugate[7]*x10;
        const scalar_t x12 = x11 + x5 + x9;
        const scalar_t x13 = 3*k;
        const scalar_t x14 = x13*(disp_grad[1] + disp_grad[3]);
        const scalar_t x15 = adjugate[2]*x4;
        const scalar_t x16 = adjugate[5]*x8;
        const scalar_t x17 = adjugate[8]*x10;
        const scalar_t x18 = x15 + x16 + x17;
        const scalar_t x19 = x13*(disp_grad[2] + disp_grad[6]);
        const scalar_t x20 = (3*K - k)*(disp_grad[0] + disp_grad[4] + disp_grad[8]);
        const scalar_t x21 = disp_grad[0]*x13 + x20;
        const scalar_t x22 = adjugate[0]*x4;
        const scalar_t x23 = adjugate[3]*x8;
        const scalar_t x24 = adjugate[6]*x10;
        const scalar_t x25 = x22 + x23 + x24;
        const scalar_t x26 = 2*x25;
        const scalar_t x27 = velo_grad[1] + velo_grad[3];
        const scalar_t x28 = velo_grad[2] + velo_grad[6];
        const scalar_t x29 = 0.33333333333333331*velo_grad[4];
        const scalar_t x30 = 0.33333333333333331*velo_grad[8];
        const scalar_t x31 = -0.66666666666666674*velo_grad[0] + x29 + x30;
        const scalar_t x32 = 3*eta;
        const scalar_t x33 = (1.0/6.0)*qw;
        const scalar_t x34 = qx*x3;
        const scalar_t x35 = qx*x1;
        const scalar_t x36 = adjugate[4]*x34 + adjugate[7]*x35 - x5;
        const scalar_t x37 = adjugate[5]*x34 + adjugate[8]*x35 - x15;
        const scalar_t x38 = adjugate[3]*x34 + adjugate[6]*x35 - x22;
        const scalar_t x39 = 2*x38;
        const scalar_t x40 = qx*qy;
        const scalar_t x41 = adjugate[7]*x40;
        const scalar_t x42 = qy*x2;
        const scalar_t x43 = qx*x2;
        const scalar_t x44 = adjugate[1]*x42 + adjugate[4]*x43 + x41;
        const scalar_t x45 = adjugate[8]*x40;
        const scalar_t x46 = adjugate[2]*x42 + adjugate[5]*x43 + x45;
        const scalar_t x47 = adjugate[6]*x40;
        const scalar_t x48 = adjugate[0]*x42 + adjugate[3]*x43 + x47;
        const scalar_t x49 = qy*x3;
        const scalar_t x50 = qy*x7;
        const scalar_t x51 = adjugate[1]*x49 + adjugate[7]*x50 - x9;
        const scalar_t x52 = adjugate[2]*x49 + adjugate[8]*x50 - x16;
        const scalar_t x53 = adjugate[0]*x49 + adjugate[6]*x50 - x23;
        const scalar_t x54 = 2*x53;
        const scalar_t x55 = qz*x1;
        const scalar_t x56 = qz*x7;
        const scalar_t x57 = adjugate[1]*x55 + adjugate[4]*x56 - x11;
        const scalar_t x58 = adjugate[2]*x55 + adjugate[5]*x56 - x17;
        const scalar_t x59 = adjugate[0]*x55 + adjugate[3]*x56 - x24;
        const scalar_t x60 = 2*x59;
        const scalar_t x61 = qx*qz;
        const scalar_t x62 = adjugate[4]*x61;
        const scalar_t x63 = qz*x0;
        const scalar_t x64 = qx*x0;
        const scalar_t x65 = adjugate[1]*x63 + adjugate[7]*x64 + x62;
        const scalar_t x66 = adjugate[5]*x61;
        const scalar_t x67 = adjugate[2]*x63 + adjugate[8]*x64 + x66;
        const scalar_t x68 = adjugate[3]*x61;
        const scalar_t x69 = adjugate[0]*x63 + adjugate[6]*x64 + x68;
        const scalar_t x70 = qy*qz;
        const scalar_t x71 = adjugate[1]*x70;
        const scalar_t x72 = x41 + x62 + x71;
        const scalar_t x73 = adjugate[2]*x70;
        const scalar_t x74 = x45 + x66 + x73;
        const scalar_t x75 = adjugate[0]*x70;
        const scalar_t x76 = x47 + x68 + x75;
        const scalar_t x77 = 2*x76;
        const scalar_t x78 = qz*x6;
        const scalar_t x79 = qy*x6;
        const scalar_t x80 = adjugate[4]*x78 + adjugate[7]*x79 + x71;
        const scalar_t x81 = adjugate[5]*x78 + adjugate[8]*x79 + x73;
        const scalar_t x82 = adjugate[3]*x78 + adjugate[6]*x79 + x75;
        const scalar_t x83 = x13*(disp_grad[5] + disp_grad[7]);
        const scalar_t x84 = disp_grad[4]*x13 + x20;
        const scalar_t x85 = 2*x12;
        const scalar_t x86 = velo_grad[5] + velo_grad[7];
        const scalar_t x87 = 0.33333333333333331*velo_grad[0];
        const scalar_t x88 = -0.66666666666666674*velo_grad[4] + x30 + x87;
        const scalar_t x89 = 2*x36;
        const scalar_t x90 = 2*x51;
        const scalar_t x91 = 2*x57;
        const scalar_t x92 = 2*x72;
        const scalar_t x93 = disp_grad[8]*x13 + x20;
        const scalar_t x94 = 2*x18;
        const scalar_t x95 = -0.66666666666666674*velo_grad[8] + x29 + x87;
        const scalar_t x96 = 2*x37;
        const scalar_t x97 = 2*x52;
        const scalar_t x98 = 2*x58;
        const scalar_t x99 = 2*x74;
        outx[0] += -x33*(x12*x14 + x18*x19 + x21*x26 + x32*(x12*x27 + x18*x28 - x26*x31));
        outx[1] += -x33*(x14*x36 + x19*x37 + x21*x39 + x32*(x27*x36 + x28*x37 - x31*x39));
        outx[2] += x33*(3*eta*(-x27*x44 - x28*x46 + 2*x31*x48) - x14*x44 - x19*x46 - 2*x21*x48);
        outx[3] += -x33*(x14*x51 + x19*x52 + x21*x54 + x32*(x27*x51 + x28*x52 - x31*x54));
        outx[4] += -x33*(x14*x57 + x19*x58 + x21*x60 + x32*(x27*x57 + x28*x58 - x31*x60));
        outx[5] += x33*(3*eta*(-x27*x65 - x28*x67 + 2*x31*x69) - x14*x65 - x19*x67 - 2*x21*x69);
        outx[6] += x33*(x14*x72 + x19*x74 + x21*x77 + x32*(x27*x72 + x28*x74 - x31*x77));
        outx[7] += x33*(3*eta*(-x27*x80 - x28*x81 + 2*x31*x82) - x14*x80 - x19*x81 - 2*x21*x82);
        outy[0] += -x33*(x14*x25 + x18*x83 + x32*(x18*x86 + x25*x27 - x85*x88) + x84*x85);
        outy[1] += -x33*(x14*x38 + x32*(x27*x38 + x37*x86 - x88*x89) + x37*x83 + x84*x89);
        outy[2] += x33*(3*eta*(-x27*x48 + 2*x44*x88 - x46*x86) - x14*x48 - 2*x44*x84 - x46*x83);
        outy[3] += -x33*(x14*x53 + x32*(x27*x53 + x52*x86 - x88*x90) + x52*x83 + x84*x90);
        outy[4] += -x33*(x14*x59 + x32*(x27*x59 + x58*x86 - x88*x91) + x58*x83 + x84*x91);
        outy[5] += x33*(3*eta*(-x27*x69 + 2*x65*x88 - x67*x86) - x14*x69 - 2*x65*x84 - x67*x83);
        outy[6] += x33*(x14*x76 + x32*(x27*x76 + x74*x86 - x88*x92) + x74*x83 + x84*x92);
        outy[7] += x33*(3*eta*(-x27*x82 + 2*x80*x88 - x81*x86) - x14*x82 - 2*x80*x84 - x81*x83);
        outz[0] += -x33*(x12*x83 + x19*x25 + x32*(x12*x86 + x25*x28 - x94*x95) + x93*x94);
        outz[1] += -x33*(x19*x38 + x32*(x28*x38 + x36*x86 - x95*x96) + x36*x83 + x93*x96);
        outz[2] += x33*(3*eta*(-x28*x48 - x44*x86 + 2*x46*x95) - x19*x48 - x44*x83 - 2*x46*x93);
        outz[3] += -x33*(x19*x53 + x32*(x28*x53 + x51*x86 - x95*x97) + x51*x83 + x93*x97);
        outz[4] += -x33*(x19*x59 + x32*(x28*x59 + x57*x86 - x95*x98) + x57*x83 + x93*x98);
        outz[5] += x33*(3*eta*(-x28*x69 - x65*x86 + 2*x67*x95) - x19*x69 - x65*x83 - 2*x67*x93);
        outz[6] += x33*(x19*x76 + x32*(x28*x76 + x72*x86 - x95*x99) + x72*x83 + x93*x99);
        outz[7] += x33*(3*eta*(-x28*x82 - x80*x86 + 2*x81*x95) - x19*x82 - x80*x83 - 2*x81*x93);
}





#endif  // HEX8_KELVIN_VOIGT_NEWMARK_INLINE_CPU_H

