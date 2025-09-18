#ifndef SFEM_HEX8_PARTIAL_ASSEMBLY_NEOHOOKEAN_INLINE_H
#define SFEM_HEX8_PARTIAL_ASSEMBLY_NEOHOOKEAN_INLINE_H

#include "sfem_macros.h"

static SFEM_INLINE void hex8_F(const scalar_t *const SFEM_RESTRICT adjugate,
                               const scalar_t                      jacobian_determinant,
                               const scalar_t                      qx,
                               const scalar_t                      qy,
                               const scalar_t                      qz,
                               const scalar_t *const SFEM_RESTRICT dispx,
                               const scalar_t *const SFEM_RESTRICT dispy,
                               const scalar_t *const SFEM_RESTRICT dispz,
                               scalar_t *const SFEM_RESTRICT       F) {
    // mundane ops: 267 divs: 1 sqrts: 0
    // total ops: 275
    const scalar_t x0 = 1.0 / jacobian_determinant;
    const scalar_t x1 = qy * qz;
    const scalar_t x2 = 1 - qz;
    const scalar_t x3 = qy * x2;
    const scalar_t x4 = 1 - qy;
    const scalar_t x5 = qz * x4;
    const scalar_t x6 = x2 * x4;
    const scalar_t x7 = dispx[0] * x6 - dispx[1] * x6 - dispx[2] * x3 + dispx[3] * x3 + dispx[4] * x5 - dispx[5] * x5 -
                        dispx[6] * x1 + dispx[7] * x1;
    const scalar_t x8  = qx * qz;
    const scalar_t x9  = qx * x2;
    const scalar_t x10 = 1 - qx;
    const scalar_t x11 = qz * x10;
    const scalar_t x12 = x10 * x2;
    const scalar_t x13 = dispx[0] * x12 + dispx[1] * x9 - dispx[2] * x9 - dispx[3] * x12 + dispx[4] * x11 + dispx[5] * x8 -
                         dispx[6] * x8 - dispx[7] * x11;
    const scalar_t x14 = qx * qy;
    const scalar_t x15 = qx * x4;
    const scalar_t x16 = qy * x10;
    const scalar_t x17 = x10 * x4;
    const scalar_t x18 = dispx[0] * x17 + dispx[1] * x15 + dispx[2] * x14 + dispx[3] * x16 - dispx[4] * x17 - dispx[5] * x15 -
                         dispx[6] * x14 - dispx[7] * x16;
    const scalar_t x19 = dispy[0] * x6 - dispy[1] * x6 - dispy[2] * x3 + dispy[3] * x3 + dispy[4] * x5 - dispy[5] * x5 -
                         dispy[6] * x1 + dispy[7] * x1;
    const scalar_t x20 = dispy[0] * x12 + dispy[1] * x9 - dispy[2] * x9 - dispy[3] * x12 + dispy[4] * x11 + dispy[5] * x8 -
                         dispy[6] * x8 - dispy[7] * x11;
    const scalar_t x21 = dispy[0] * x17 + dispy[1] * x15 + dispy[2] * x14 + dispy[3] * x16 - dispy[4] * x17 - dispy[5] * x15 -
                         dispy[6] * x14 - dispy[7] * x16;
    const scalar_t x22 = dispz[0] * x6 - dispz[1] * x6 - dispz[2] * x3 + dispz[3] * x3 + dispz[4] * x5 - dispz[5] * x5 -
                         dispz[6] * x1 + dispz[7] * x1;
    const scalar_t x23 = dispz[0] * x12 + dispz[1] * x9 - dispz[2] * x9 - dispz[3] * x12 + dispz[4] * x11 + dispz[5] * x8 -
                         dispz[6] * x8 - dispz[7] * x11;
    const scalar_t x24 = dispz[0] * x17 + dispz[1] * x15 + dispz[2] * x14 + dispz[3] * x16 - dispz[4] * x17 - dispz[5] * x15 -
                         dispz[6] * x14 - dispz[7] * x16;
    F[0] = -adjugate[0] * x0 * x7 - adjugate[3] * x0 * x13 - adjugate[6] * x0 * x18 + 1;
    F[1] = -x0 * (adjugate[1] * x7 + adjugate[4] * x13 + adjugate[7] * x18);
    F[2] = -x0 * (adjugate[2] * x7 + adjugate[5] * x13 + adjugate[8] * x18);
    F[3] = -x0 * (adjugate[0] * x19 + adjugate[3] * x20 + adjugate[6] * x21);
    F[4] = -adjugate[1] * x0 * x19 - adjugate[4] * x0 * x20 - adjugate[7] * x0 * x21 + 1;
    F[5] = -x0 * (adjugate[2] * x19 + adjugate[5] * x20 + adjugate[8] * x21);
    F[6] = -x0 * (adjugate[0] * x22 + adjugate[3] * x23 + adjugate[6] * x24);
    F[7] = -x0 * (adjugate[1] * x22 + adjugate[4] * x23 + adjugate[7] * x24);
    F[8] = -adjugate[2] * x0 * x22 - adjugate[5] * x0 * x23 - adjugate[8] * x0 * x24 + 1;
}

static SFEM_INLINE void hex8_Wimpn_compressed(scalar_t *const SFEM_RESTRICT Wimpn_compressed) {
    // mundane ops: 0 divs: 0 sqrts: 0
    // total ops: 0
    Wimpn_compressed[0] = 1.0 / 9.0;
    Wimpn_compressed[1] = 1.0 / 12.0;
    Wimpn_compressed[2] = -1.0 / 9.0;
    Wimpn_compressed[3] = -1.0 / 18.0;
    Wimpn_compressed[4] = -1.0 / 12.0;
    Wimpn_compressed[5] = 1.0 / 24.0;
    Wimpn_compressed[6] = 1.0 / 18.0;
    Wimpn_compressed[7] = -1.0 / 36.0;
    Wimpn_compressed[8] = -1.0 / 24.0;
    Wimpn_compressed[9] = 1.0 / 36.0;
}

static SFEM_INLINE void hex8_SdotHdotG(const scalar_t *const SFEM_RESTRICT S_ikmn_canonical,
                                       const scalar_t *const SFEM_RESTRICT Wimpn_compressed,
                                       const scalar_t *const SFEM_RESTRICT incx,
                                       const scalar_t *const SFEM_RESTRICT incy,
                                       const scalar_t *const SFEM_RESTRICT incz,
                                       scalar_t *const SFEM_RESTRICT       outx,
                                       scalar_t *const SFEM_RESTRICT       outy,
                                       scalar_t *const SFEM_RESTRICT       outz) {
    // mundane ops: 1558 divs: 0 sqrts: 0
    // total ops: 1558
    const scalar_t x0  = Wimpn_compressed[0] * incx[0];
    const scalar_t x1  = Wimpn_compressed[7] * incx[6];
    const scalar_t x2  = Wimpn_compressed[3] * incx[2] + Wimpn_compressed[6] * incx[4];
    const scalar_t x3  = x0 + x1 + x2;
    const scalar_t x4  = Wimpn_compressed[2] * incx[1];
    const scalar_t x5  = Wimpn_compressed[9] * incx[7];
    const scalar_t x6  = Wimpn_compressed[3] * incx[5] + Wimpn_compressed[6] * incx[3];
    const scalar_t x7  = x4 + x5 + x6;
    const scalar_t x8  = x3 + x7;
    const scalar_t x9  = Wimpn_compressed[0] * incy[0];
    const scalar_t x10 = Wimpn_compressed[7] * incy[6];
    const scalar_t x11 = Wimpn_compressed[3] * incy[7] + Wimpn_compressed[6] * incy[1];
    const scalar_t x12 = x10 + x11 + x9;
    const scalar_t x13 = Wimpn_compressed[3] * incy[2] + Wimpn_compressed[6] * incy[4];
    const scalar_t x14 = Wimpn_compressed[2] * incy[3] + Wimpn_compressed[9] * incy[5];
    const scalar_t x15 = x13 + x14;
    const scalar_t x16 = x12 + x15;
    const scalar_t x17 = Wimpn_compressed[3] * incy[5] + Wimpn_compressed[6] * incy[3];
    const scalar_t x18 = Wimpn_compressed[2] * incy[4] + Wimpn_compressed[9] * incy[2];
    const scalar_t x19 = x17 + x18;
    const scalar_t x20 = x12 + x19;
    const scalar_t x21 = Wimpn_compressed[0] * incz[0];
    const scalar_t x22 = Wimpn_compressed[7] * incz[6];
    const scalar_t x23 = Wimpn_compressed[3] * incz[2] + Wimpn_compressed[6] * incz[4];
    const scalar_t x24 = x21 + x22 + x23;
    const scalar_t x25 = Wimpn_compressed[2] * incz[1];
    const scalar_t x26 = Wimpn_compressed[9] * incz[7];
    const scalar_t x27 = Wimpn_compressed[3] * incz[5] + Wimpn_compressed[6] * incz[3];
    const scalar_t x28 = x25 + x26 + x27;
    const scalar_t x29 = x24 + x28;
    const scalar_t x30 = Wimpn_compressed[3] * incz[7] + Wimpn_compressed[6] * incz[1];
    const scalar_t x31 = Wimpn_compressed[2] * incz[3] + Wimpn_compressed[9] * incz[5];
    const scalar_t x32 = x30 + x31;
    const scalar_t x33 = x24 + x32;
    const scalar_t x34 = x27 + x30;
    const scalar_t x35 = Wimpn_compressed[2] * incz[4] + Wimpn_compressed[9] * incz[2];
    const scalar_t x36 = x21 + x22 + x34 + x35;
    const scalar_t x37 = Wimpn_compressed[3] * incx[7] + Wimpn_compressed[6] * incx[1];
    const scalar_t x38 = Wimpn_compressed[2] * incx[3] + Wimpn_compressed[9] * incx[5];
    const scalar_t x39 = x37 + x38;
    const scalar_t x40 = x3 + x39;
    const scalar_t x41 = x37 + x6;
    const scalar_t x42 = Wimpn_compressed[2] * incx[4] + Wimpn_compressed[9] * incx[2];
    const scalar_t x43 = x0 + x1 + x41 + x42;
    const scalar_t x44 = x13 + x17;
    const scalar_t x45 = Wimpn_compressed[2] * incy[1] + Wimpn_compressed[9] * incy[7];
    const scalar_t x46 = x10 + x44 + x45 + x9;
    const scalar_t x47 = Wimpn_compressed[1] * incz[0] + Wimpn_compressed[8] * incz[6];
    const scalar_t x48 = Wimpn_compressed[4] * incz[1] + Wimpn_compressed[5] * incz[7];
    const scalar_t x49 = x47 + x48;
    const scalar_t x50 = Wimpn_compressed[4] * incz[2] + Wimpn_compressed[5] * incz[4];
    const scalar_t x51 = Wimpn_compressed[1] * incz[3] + Wimpn_compressed[8] * incz[5];
    const scalar_t x52 = x50 + x51;
    const scalar_t x53 = x49 + x52;
    const scalar_t x54 = Wimpn_compressed[1] * incz[4] + Wimpn_compressed[8] * incz[2];
    const scalar_t x55 = Wimpn_compressed[4] * incz[5] + Wimpn_compressed[5] * incz[3];
    const scalar_t x56 = x54 + x55;
    const scalar_t x57 = x49 + x56;
    const scalar_t x58 = Wimpn_compressed[1] * incx[0] + Wimpn_compressed[8] * incx[6];
    const scalar_t x59 = Wimpn_compressed[4] * incx[2] + Wimpn_compressed[5] * incx[4];
    const scalar_t x60 = x58 + x59;
    const scalar_t x61 = Wimpn_compressed[4] * incx[1] + Wimpn_compressed[5] * incx[7];
    const scalar_t x62 = Wimpn_compressed[1] * incx[3] + Wimpn_compressed[8] * incx[5];
    const scalar_t x63 = x61 + x62;
    const scalar_t x64 = x60 + x63;
    const scalar_t x65 = Wimpn_compressed[4] * incx[5] + Wimpn_compressed[5] * incx[3];
    const scalar_t x66 = x58 + x65;
    const scalar_t x67 = Wimpn_compressed[1] * incx[4] + Wimpn_compressed[8] * incx[2];
    const scalar_t x68 = x61 + x67;
    const scalar_t x69 = x66 + x68;
    const scalar_t x70 = Wimpn_compressed[1] * incy[0] + Wimpn_compressed[8] * incy[6];
    const scalar_t x71 = Wimpn_compressed[4] * incy[2] + Wimpn_compressed[5] * incy[4];
    const scalar_t x72 = x70 + x71;
    const scalar_t x73 = Wimpn_compressed[1] * incy[3] + Wimpn_compressed[8] * incy[5];
    const scalar_t x74 = Wimpn_compressed[4] * incy[1] + Wimpn_compressed[5] * incy[7];
    const scalar_t x75 = x73 + x74;
    const scalar_t x76 = x72 + x75;
    const scalar_t x77 = Wimpn_compressed[4] * incy[5] + Wimpn_compressed[5] * incy[3];
    const scalar_t x78 = x70 + x77;
    const scalar_t x79 = Wimpn_compressed[1] * incy[4] + Wimpn_compressed[8] * incy[2];
    const scalar_t x80 = x74 + x79;
    const scalar_t x81 = x78 + x80;
    const scalar_t x82 = S_ikmn_canonical[16] * x53 + S_ikmn_canonical[17] * x57 + S_ikmn_canonical[1] * x64 +
                         S_ikmn_canonical[2] * x69 + S_ikmn_canonical[7] * x76 + S_ikmn_canonical[8] * x81;
    const scalar_t x83  = Wimpn_compressed[4] * incy[7] + Wimpn_compressed[5] * incy[1];
    const scalar_t x84  = x70 + x83;
    const scalar_t x85  = Wimpn_compressed[4] * incy[3] + Wimpn_compressed[5] * incy[5];
    const scalar_t x86  = x79 + x85;
    const scalar_t x87  = x84 + x86;
    const scalar_t x88  = Wimpn_compressed[4] * incz[3] + Wimpn_compressed[5] * incz[5];
    const scalar_t x89  = x47 + x88;
    const scalar_t x90  = Wimpn_compressed[1] * incz[1] + Wimpn_compressed[8] * incz[7];
    const scalar_t x91  = x50 + x90;
    const scalar_t x92  = x89 + x91;
    const scalar_t x93  = Wimpn_compressed[1] * incx[1] + Wimpn_compressed[8] * incx[7];
    const scalar_t x94  = Wimpn_compressed[4] * incx[3] + Wimpn_compressed[5] * incx[5];
    const scalar_t x95  = x93 + x94;
    const scalar_t x96  = x60 + x95;
    const scalar_t x97  = Wimpn_compressed[4] * incz[7] + Wimpn_compressed[5] * incz[1];
    const scalar_t x98  = x54 + x97;
    const scalar_t x99  = x89 + x98;
    const scalar_t x100 = Wimpn_compressed[4] * incx[7] + Wimpn_compressed[5] * incx[1];
    const scalar_t x101 = x100 + x58;
    const scalar_t x102 = x67 + x94;
    const scalar_t x103 = x101 + x102;
    const scalar_t x104 = Wimpn_compressed[1] * incy[1] + Wimpn_compressed[8] * incy[7];
    const scalar_t x105 = x104 + x85;
    const scalar_t x106 = x105 + x72;
    const scalar_t x107 = S_ikmn_canonical[11] * x87 + S_ikmn_canonical[18] * x92 + S_ikmn_canonical[1] * x96 +
                          S_ikmn_canonical[20] * x99 + S_ikmn_canonical[4] * x103 + S_ikmn_canonical[9] * x106;
    const scalar_t x108 = Wimpn_compressed[4] * incy[4] + Wimpn_compressed[5] * incy[2];
    const scalar_t x109 = x104 + x108;
    const scalar_t x110 = x109 + x78;
    const scalar_t x111 = x108 + x73;
    const scalar_t x112 = x111 + x84;
    const scalar_t x113 = x55 + x90;
    const scalar_t x114 = Wimpn_compressed[4] * incz[4] + Wimpn_compressed[5] * incz[2];
    const scalar_t x115 = x114 + x47;
    const scalar_t x116 = x113 + x115;
    const scalar_t x117 = x51 + x97;
    const scalar_t x118 = x115 + x117;
    const scalar_t x119 = Wimpn_compressed[4] * incx[4] + Wimpn_compressed[5] * incx[2];
    const scalar_t x120 = x119 + x93;
    const scalar_t x121 = x120 + x66;
    const scalar_t x122 = x119 + x62;
    const scalar_t x123 = x101 + x122;
    const scalar_t x124 = S_ikmn_canonical[12] * x110 + S_ikmn_canonical[13] * x112 + S_ikmn_canonical[21] * x116 +
                          S_ikmn_canonical[22] * x118 + S_ikmn_canonical[2] * x121 + S_ikmn_canonical[4] * x123;
    const scalar_t x125 = Wimpn_compressed[0] * incx[1];
    const scalar_t x126 = Wimpn_compressed[7] * incx[7];
    const scalar_t x127 = Wimpn_compressed[3] * incx[3] + Wimpn_compressed[6] * incx[5];
    const scalar_t x128 = x125 + x126 + x127;
    const scalar_t x129 = Wimpn_compressed[2] * incx[0];
    const scalar_t x130 = Wimpn_compressed[9] * incx[6];
    const scalar_t x131 = Wimpn_compressed[3] * incx[4] + Wimpn_compressed[6] * incx[2];
    const scalar_t x132 = x129 + x130 + x131;
    const scalar_t x133 = x128 + x132;
    const scalar_t x134 = Wimpn_compressed[0] * incy[1];
    const scalar_t x135 = Wimpn_compressed[7] * incy[7];
    const scalar_t x136 = Wimpn_compressed[3] * incy[6] + Wimpn_compressed[6] * incy[0];
    const scalar_t x137 = x134 + x135 + x136;
    const scalar_t x138 = Wimpn_compressed[3] * incy[3] + Wimpn_compressed[6] * incy[5];
    const scalar_t x139 = Wimpn_compressed[2] * incy[2] + Wimpn_compressed[9] * incy[4];
    const scalar_t x140 = x138 + x139;
    const scalar_t x141 = x137 + x140;
    const scalar_t x142 = Wimpn_compressed[3] * incy[4] + Wimpn_compressed[6] * incy[2];
    const scalar_t x143 = Wimpn_compressed[2] * incy[5] + Wimpn_compressed[9] * incy[3];
    const scalar_t x144 = x142 + x143;
    const scalar_t x145 = x137 + x144;
    const scalar_t x146 = Wimpn_compressed[0] * incz[1];
    const scalar_t x147 = Wimpn_compressed[7] * incz[7];
    const scalar_t x148 = Wimpn_compressed[3] * incz[3] + Wimpn_compressed[6] * incz[5];
    const scalar_t x149 = x146 + x147 + x148;
    const scalar_t x150 = Wimpn_compressed[2] * incz[0];
    const scalar_t x151 = Wimpn_compressed[9] * incz[6];
    const scalar_t x152 = Wimpn_compressed[3] * incz[4] + Wimpn_compressed[6] * incz[2];
    const scalar_t x153 = x150 + x151 + x152;
    const scalar_t x154 = x149 + x153;
    const scalar_t x155 = Wimpn_compressed[3] * incz[6] + Wimpn_compressed[6] * incz[0];
    const scalar_t x156 = Wimpn_compressed[2] * incz[2] + Wimpn_compressed[9] * incz[4];
    const scalar_t x157 = x155 + x156;
    const scalar_t x158 = x149 + x157;
    const scalar_t x159 = x152 + x155;
    const scalar_t x160 = Wimpn_compressed[2] * incz[5] + Wimpn_compressed[9] * incz[3];
    const scalar_t x161 = x146 + x147 + x159 + x160;
    const scalar_t x162 = Wimpn_compressed[3] * incx[6] + Wimpn_compressed[6] * incx[0];
    const scalar_t x163 = Wimpn_compressed[2] * incx[2] + Wimpn_compressed[9] * incx[4];
    const scalar_t x164 = x162 + x163;
    const scalar_t x165 = x128 + x164;
    const scalar_t x166 = x131 + x162;
    const scalar_t x167 = Wimpn_compressed[2] * incx[5] + Wimpn_compressed[9] * incx[3];
    const scalar_t x168 = x125 + x126 + x166 + x167;
    const scalar_t x169 = x138 + x142;
    const scalar_t x170 = Wimpn_compressed[2] * incy[0] + Wimpn_compressed[9] * incy[6];
    const scalar_t x171 = x134 + x135 + x169 + x170;
    const scalar_t x172 = Wimpn_compressed[4] * incy[6] + Wimpn_compressed[5] * incy[0];
    const scalar_t x173 = x104 + x172;
    const scalar_t x174 = Wimpn_compressed[1] * incy[5] + Wimpn_compressed[8] * incy[3];
    const scalar_t x175 = x174 + x71;
    const scalar_t x176 = x173 + x175;
    const scalar_t x177 = Wimpn_compressed[1] * incz[2] + Wimpn_compressed[8] * incz[4];
    const scalar_t x178 = Wimpn_compressed[4] * incz[0] + Wimpn_compressed[5] * incz[6];
    const scalar_t x179 = x178 + x48;
    const scalar_t x180 = x177 + x179 + x51;
    const scalar_t x181 = Wimpn_compressed[4] * incx[0] + Wimpn_compressed[5] * incx[6];
    const scalar_t x182 = Wimpn_compressed[1] * incx[2] + Wimpn_compressed[8] * incx[4];
    const scalar_t x183 = x181 + x182;
    const scalar_t x184 = x183 + x63;
    const scalar_t x185 = Wimpn_compressed[1] * incz[5] + Wimpn_compressed[8] * incz[3];
    const scalar_t x186 = Wimpn_compressed[4] * incz[6] + Wimpn_compressed[5] * incz[0];
    const scalar_t x187 = x185 + x186;
    const scalar_t x188 = x187 + x91;
    const scalar_t x189 = Wimpn_compressed[4] * incx[6] + Wimpn_compressed[5] * incx[0];
    const scalar_t x190 = x189 + x93;
    const scalar_t x191 = Wimpn_compressed[1] * incx[5] + Wimpn_compressed[8] * incx[3];
    const scalar_t x192 = x191 + x59;
    const scalar_t x193 = x190 + x192;
    const scalar_t x194 = Wimpn_compressed[4] * incy[0] + Wimpn_compressed[5] * incy[6];
    const scalar_t x195 = Wimpn_compressed[1] * incy[2] + Wimpn_compressed[8] * incy[4];
    const scalar_t x196 = x194 + x195;
    const scalar_t x197 = x196 + x75;
    const scalar_t x198 = S_ikmn_canonical[11] * x176 + S_ikmn_canonical[18] * x180 + S_ikmn_canonical[1] * x184 +
                          S_ikmn_canonical[20] * x188 + S_ikmn_canonical[4] * x193 + S_ikmn_canonical[9] * x197;
    const scalar_t x199 = x174 + x194;
    const scalar_t x200 = x199 + x80;
    const scalar_t x201 = x195 + x77;
    const scalar_t x202 = x173 + x201;
    const scalar_t x203 = x179 + x185 + x54;
    const scalar_t x204 = x177 + x186;
    const scalar_t x205 = x113 + x204;
    const scalar_t x206 = x181 + x191;
    const scalar_t x207 = x206 + x68;
    const scalar_t x208 = x182 + x65;
    const scalar_t x209 = x190 + x208;
    const scalar_t x210 = S_ikmn_canonical[12] * x200 + S_ikmn_canonical[13] * x202 + S_ikmn_canonical[21] * x203 +
                          S_ikmn_canonical[22] * x205 + S_ikmn_canonical[2] * x207 + S_ikmn_canonical[4] * x209;
    const scalar_t x211 = Wimpn_compressed[0] * incx[2] + Wimpn_compressed[7] * incx[4];
    const scalar_t x212 = Wimpn_compressed[3] * incx[0] + Wimpn_compressed[6] * incx[6];
    const scalar_t x213 = x211 + x212;
    const scalar_t x214 = x213 + x39;
    const scalar_t x215 = Wimpn_compressed[0] * incy[2];
    const scalar_t x216 = Wimpn_compressed[7] * incy[4];
    const scalar_t x217 = x17 + x215 + x216;
    const scalar_t x218 = Wimpn_compressed[3] * incy[0] + Wimpn_compressed[6] * incy[6];
    const scalar_t x219 = x218 + x45;
    const scalar_t x220 = x217 + x219;
    const scalar_t x221 = Wimpn_compressed[2] * incy[6] + Wimpn_compressed[9] * incy[0];
    const scalar_t x222 = x11 + x221;
    const scalar_t x223 = x217 + x222;
    const scalar_t x224 = Wimpn_compressed[0] * incz[2] + Wimpn_compressed[7] * incz[4];
    const scalar_t x225 = Wimpn_compressed[3] * incz[0] + Wimpn_compressed[6] * incz[6];
    const scalar_t x226 = x224 + x225;
    const scalar_t x227 = x226 + x32;
    const scalar_t x228 = x226 + x28;
    const scalar_t x229 = Wimpn_compressed[2] * incz[6] + Wimpn_compressed[9] * incz[0];
    const scalar_t x230 = x224 + x229 + x34;
    const scalar_t x231 = x213 + x7;
    const scalar_t x232 = Wimpn_compressed[2] * incx[6] + Wimpn_compressed[9] * incx[0];
    const scalar_t x233 = x211 + x232 + x41;
    const scalar_t x234 = x11 + x218;
    const scalar_t x235 = x14 + x215 + x216 + x234;
    const scalar_t x236 = x178 + x88;
    const scalar_t x237 = x177 + x236 + x90;
    const scalar_t x238 = Wimpn_compressed[1] * incz[7] + Wimpn_compressed[8] * incz[1];
    const scalar_t x239 = x186 + x238;
    const scalar_t x240 = x239 + x52;
    const scalar_t x241 = x183 + x95;
    const scalar_t x242 = x189 + x62;
    const scalar_t x243 = Wimpn_compressed[1] * incx[7] + Wimpn_compressed[8] * incx[1];
    const scalar_t x244 = x243 + x59;
    const scalar_t x245 = x242 + x244;
    const scalar_t x246 = x105 + x196;
    const scalar_t x247 = Wimpn_compressed[1] * incy[7] + Wimpn_compressed[8] * incy[1];
    const scalar_t x248 = x247 + x71;
    const scalar_t x249 = x172 + x73;
    const scalar_t x250 = x248 + x249;
    const scalar_t x251 = S_ikmn_canonical[16] * x237 + S_ikmn_canonical[17] * x240 + S_ikmn_canonical[1] * x241 +
                          S_ikmn_canonical[2] * x245 + S_ikmn_canonical[7] * x246 + S_ikmn_canonical[8] * x250;
    const scalar_t x252 = Wimpn_compressed[1] * incy[6] + Wimpn_compressed[8] * incy[0];
    const scalar_t x253 = x252 + x85;
    const scalar_t x254 = x248 + x253;
    const scalar_t x255 = x252 + x74;
    const scalar_t x256 = x175 + x255;
    const scalar_t x257 = Wimpn_compressed[1] * incz[6] + Wimpn_compressed[8] * incz[0];
    const scalar_t x258 = x257 + x50;
    const scalar_t x259 = x238 + x258 + x88;
    const scalar_t x260 = x185 + x258 + x48;
    const scalar_t x261 = Wimpn_compressed[1] * incx[6] + Wimpn_compressed[8] * incx[0];
    const scalar_t x262 = x261 + x94;
    const scalar_t x263 = x244 + x262;
    const scalar_t x264 = x261 + x61;
    const scalar_t x265 = x192 + x264;
    const scalar_t x266 = S_ikmn_canonical[12] * x254 + S_ikmn_canonical[13] * x256 + S_ikmn_canonical[21] * x259 +
                          S_ikmn_canonical[22] * x260 + S_ikmn_canonical[2] * x263 + S_ikmn_canonical[4] * x265;
    const scalar_t x267 = Wimpn_compressed[0] * incx[3] + Wimpn_compressed[7] * incx[5];
    const scalar_t x268 = Wimpn_compressed[3] * incx[1] + Wimpn_compressed[6] * incx[7];
    const scalar_t x269 = x267 + x268;
    const scalar_t x270 = x164 + x269;
    const scalar_t x271 = Wimpn_compressed[0] * incy[3];
    const scalar_t x272 = Wimpn_compressed[7] * incy[5];
    const scalar_t x273 = x142 + x271 + x272;
    const scalar_t x274 = Wimpn_compressed[3] * incy[1] + Wimpn_compressed[6] * incy[7];
    const scalar_t x275 = x170 + x274;
    const scalar_t x276 = x273 + x275;
    const scalar_t x277 = Wimpn_compressed[2] * incy[7] + Wimpn_compressed[9] * incy[1];
    const scalar_t x278 = x136 + x277;
    const scalar_t x279 = x273 + x278;
    const scalar_t x280 = Wimpn_compressed[0] * incz[3] + Wimpn_compressed[7] * incz[5];
    const scalar_t x281 = Wimpn_compressed[3] * incz[1] + Wimpn_compressed[6] * incz[7];
    const scalar_t x282 = x280 + x281;
    const scalar_t x283 = x157 + x282;
    const scalar_t x284 = x153 + x282;
    const scalar_t x285 = Wimpn_compressed[2] * incz[7] + Wimpn_compressed[9] * incz[1];
    const scalar_t x286 = x159 + x280 + x285;
    const scalar_t x287 = x132 + x269;
    const scalar_t x288 = Wimpn_compressed[2] * incx[7] + Wimpn_compressed[9] * incx[1];
    const scalar_t x289 = x166 + x267 + x288;
    const scalar_t x290 = x136 + x274;
    const scalar_t x291 = x139 + x271 + x272 + x290;
    const scalar_t x292 = x195 + x83;
    const scalar_t x293 = x249 + x292;
    const scalar_t x294 = x194 + x247;
    const scalar_t x295 = x294 + x86;
    const scalar_t x296 = x117 + x204;
    const scalar_t x297 = x236 + x238 + x54;
    const scalar_t x298 = x100 + x182;
    const scalar_t x299 = x242 + x298;
    const scalar_t x300 = x181 + x243;
    const scalar_t x301 = x102 + x300;
    const scalar_t x302 = S_ikmn_canonical[12] * x293 + S_ikmn_canonical[13] * x295 + S_ikmn_canonical[21] * x296 +
                          S_ikmn_canonical[22] * x297 + S_ikmn_canonical[2] * x299 + S_ikmn_canonical[4] * x301;
    const scalar_t x303 = x167 + x268;
    const scalar_t x304 = Wimpn_compressed[0] * incx[4];
    const scalar_t x305 = Wimpn_compressed[7] * incx[2];
    const scalar_t x306 = x162 + x304 + x305;
    const scalar_t x307 = x303 + x306;
    const scalar_t x308 = Wimpn_compressed[0] * incy[4] + Wimpn_compressed[7] * incy[2];
    const scalar_t x309 = x138 + x308;
    const scalar_t x310 = x278 + x309;
    const scalar_t x311 = x275 + x309;
    const scalar_t x312 = x160 + x281;
    const scalar_t x313 = Wimpn_compressed[0] * incz[4];
    const scalar_t x314 = Wimpn_compressed[7] * incz[2];
    const scalar_t x315 = x155 + x313 + x314;
    const scalar_t x316 = x312 + x315;
    const scalar_t x317 = x148 + x285;
    const scalar_t x318 = x315 + x317;
    const scalar_t x319 = x148 + x281;
    const scalar_t x320 = x150 + x151 + x313 + x314 + x319;
    const scalar_t x321 = x127 + x288;
    const scalar_t x322 = x306 + x321;
    const scalar_t x323 = x127 + x268;
    const scalar_t x324 = x129 + x130 + x304 + x305 + x323;
    const scalar_t x325 = x143 + x290 + x308;
    const scalar_t x326 = x239 + x56;
    const scalar_t x327 = x114 + x178;
    const scalar_t x328 = x185 + x327 + x90;
    const scalar_t x329 = x189 + x67;
    const scalar_t x330 = x243 + x65;
    const scalar_t x331 = x329 + x330;
    const scalar_t x332 = x120 + x206;
    const scalar_t x333 = x172 + x79;
    const scalar_t x334 = x247 + x77;
    const scalar_t x335 = x333 + x334;
    const scalar_t x336 = x109 + x199;
    const scalar_t x337 = S_ikmn_canonical[16] * x326 + S_ikmn_canonical[17] * x328 + S_ikmn_canonical[1] * x331 +
                          S_ikmn_canonical[2] * x332 + S_ikmn_canonical[7] * x335 + S_ikmn_canonical[8] * x336;
    const scalar_t x338 = x111 + x294;
    const scalar_t x339 = x187 + x98;
    const scalar_t x340 = x100 + x191;
    const scalar_t x341 = x329 + x340;
    const scalar_t x342 = x238 + x327 + x51;
    const scalar_t x343 = x122 + x300;
    const scalar_t x344 = x174 + x83;
    const scalar_t x345 = x333 + x344;
    const scalar_t x346 = S_ikmn_canonical[11] * x338 + S_ikmn_canonical[18] * x339 + S_ikmn_canonical[1] * x341 +
                          S_ikmn_canonical[20] * x342 + S_ikmn_canonical[4] * x343 + S_ikmn_canonical[9] * x345;
    const scalar_t x347 = x212 + x42;
    const scalar_t x348 = Wimpn_compressed[0] * incx[5];
    const scalar_t x349 = Wimpn_compressed[7] * incx[3];
    const scalar_t x350 = x348 + x349 + x37;
    const scalar_t x351 = x347 + x350;
    const scalar_t x352 = Wimpn_compressed[0] * incy[5] + Wimpn_compressed[7] * incy[3];
    const scalar_t x353 = x13 + x352;
    const scalar_t x354 = x222 + x353;
    const scalar_t x355 = x219 + x353;
    const scalar_t x356 = x225 + x35;
    const scalar_t x357 = Wimpn_compressed[0] * incz[5];
    const scalar_t x358 = Wimpn_compressed[7] * incz[3];
    const scalar_t x359 = x30 + x357 + x358;
    const scalar_t x360 = x356 + x359;
    const scalar_t x361 = x229 + x23;
    const scalar_t x362 = x359 + x361;
    const scalar_t x363 = x225 + x23;
    const scalar_t x364 = x25 + x26 + x357 + x358 + x363;
    const scalar_t x365 = x2 + x232;
    const scalar_t x366 = x350 + x365;
    const scalar_t x367 = x2 + x212;
    const scalar_t x368 = x348 + x349 + x367 + x4 + x5;
    const scalar_t x369 = x18 + x234 + x352;
    const scalar_t x370 = x201 + x255;
    const scalar_t x371 = x257 + x55;
    const scalar_t x372 = x114 + x238 + x371;
    const scalar_t x373 = x119 + x261;
    const scalar_t x374 = x330 + x373;
    const scalar_t x375 = x177 + x371 + x48;
    const scalar_t x376 = x208 + x264;
    const scalar_t x377 = x108 + x252;
    const scalar_t x378 = x334 + x377;
    const scalar_t x379 = S_ikmn_canonical[11] * x370 + S_ikmn_canonical[18] * x372 + S_ikmn_canonical[1] * x374 +
                          S_ikmn_canonical[20] * x375 + S_ikmn_canonical[4] * x376 + S_ikmn_canonical[9] * x378;
    const scalar_t x380 = Wimpn_compressed[0] * incx[6] + Wimpn_compressed[7] * incx[0];
    const scalar_t x381 = x131 + x380;
    const scalar_t x382 = x321 + x381;
    const scalar_t x383 = Wimpn_compressed[0] * incy[6] + Wimpn_compressed[7] * incy[0];
    const scalar_t x384 = x274 + x383;
    const scalar_t x385 = x144 + x384;
    const scalar_t x386 = x140 + x384;
    const scalar_t x387 = Wimpn_compressed[0] * incz[6] + Wimpn_compressed[7] * incz[0];
    const scalar_t x388 = x152 + x387;
    const scalar_t x389 = x317 + x388;
    const scalar_t x390 = x312 + x388;
    const scalar_t x391 = x156 + x319 + x387;
    const scalar_t x392 = x303 + x381;
    const scalar_t x393 = x163 + x323 + x380;
    const scalar_t x394 = x169 + x277 + x383;
    const scalar_t x395 = x257 + x97;
    const scalar_t x396 = x114 + x185 + x395;
    const scalar_t x397 = x177 + x395 + x88;
    const scalar_t x398 = x340 + x373;
    const scalar_t x399 = x262 + x298;
    const scalar_t x400 = x344 + x377;
    const scalar_t x401 = x253 + x292;
    const scalar_t x402 = S_ikmn_canonical[16] * x396 + S_ikmn_canonical[17] * x397 + S_ikmn_canonical[1] * x398 +
                          S_ikmn_canonical[2] * x399 + S_ikmn_canonical[7] * x400 + S_ikmn_canonical[8] * x401;
    const scalar_t x403 = Wimpn_compressed[0] * incx[7] + Wimpn_compressed[7] * incx[1];
    const scalar_t x404 = x403 + x6;
    const scalar_t x405 = x365 + x404;
    const scalar_t x406 = Wimpn_compressed[0] * incy[7] + Wimpn_compressed[7] * incy[1];
    const scalar_t x407 = x218 + x406;
    const scalar_t x408 = x19 + x407;
    const scalar_t x409 = x15 + x407;
    const scalar_t x410 = Wimpn_compressed[0] * incz[7] + Wimpn_compressed[7] * incz[1];
    const scalar_t x411 = x27 + x410;
    const scalar_t x412 = x361 + x411;
    const scalar_t x413 = x356 + x411;
    const scalar_t x414 = x31 + x363 + x410;
    const scalar_t x415 = x347 + x404;
    const scalar_t x416 = x367 + x38 + x403;
    const scalar_t x417 = x221 + x406 + x44;
    const scalar_t x418 = S_ikmn_canonical[12] * x69 + S_ikmn_canonical[25] * x76 + S_ikmn_canonical[26] * x81 +
                          S_ikmn_canonical[31] * x53 + S_ikmn_canonical[32] * x57 + S_ikmn_canonical[9] * x64;
    const scalar_t x419 = S_ikmn_canonical[13] * x103 + S_ikmn_canonical[25] * x106 + S_ikmn_canonical[28] * x87 +
                          S_ikmn_canonical[33] * x92 + S_ikmn_canonical[35] * x99 + S_ikmn_canonical[7] * x96;
    const scalar_t x420 = S_ikmn_canonical[11] * x123 + S_ikmn_canonical[26] * x110 + S_ikmn_canonical[28] * x112 +
                          S_ikmn_canonical[36] * x116 + S_ikmn_canonical[37] * x118 + S_ikmn_canonical[8] * x121;
    const scalar_t x421 = S_ikmn_canonical[13] * x193 + S_ikmn_canonical[25] * x197 + S_ikmn_canonical[28] * x176 +
                          S_ikmn_canonical[33] * x180 + S_ikmn_canonical[35] * x188 + S_ikmn_canonical[7] * x184;
    const scalar_t x422 = S_ikmn_canonical[11] * x209 + S_ikmn_canonical[26] * x200 + S_ikmn_canonical[28] * x202 +
                          S_ikmn_canonical[36] * x203 + S_ikmn_canonical[37] * x205 + S_ikmn_canonical[8] * x207;
    const scalar_t x423 = S_ikmn_canonical[12] * x245 + S_ikmn_canonical[25] * x246 + S_ikmn_canonical[26] * x250 +
                          S_ikmn_canonical[31] * x237 + S_ikmn_canonical[32] * x240 + S_ikmn_canonical[9] * x241;
    const scalar_t x424 = S_ikmn_canonical[11] * x265 + S_ikmn_canonical[26] * x254 + S_ikmn_canonical[28] * x256 +
                          S_ikmn_canonical[36] * x259 + S_ikmn_canonical[37] * x260 + S_ikmn_canonical[8] * x263;
    const scalar_t x425 = S_ikmn_canonical[11] * x301 + S_ikmn_canonical[26] * x293 + S_ikmn_canonical[28] * x295 +
                          S_ikmn_canonical[36] * x296 + S_ikmn_canonical[37] * x297 + S_ikmn_canonical[8] * x299;
    const scalar_t x426 = S_ikmn_canonical[12] * x332 + S_ikmn_canonical[25] * x335 + S_ikmn_canonical[26] * x336 +
                          S_ikmn_canonical[31] * x326 + S_ikmn_canonical[32] * x328 + S_ikmn_canonical[9] * x331;
    const scalar_t x427 = S_ikmn_canonical[13] * x343 + S_ikmn_canonical[25] * x345 + S_ikmn_canonical[28] * x338 +
                          S_ikmn_canonical[33] * x339 + S_ikmn_canonical[35] * x342 + S_ikmn_canonical[7] * x341;
    const scalar_t x428 = S_ikmn_canonical[13] * x376 + S_ikmn_canonical[25] * x378 + S_ikmn_canonical[28] * x370 +
                          S_ikmn_canonical[33] * x372 + S_ikmn_canonical[35] * x375 + S_ikmn_canonical[7] * x374;
    const scalar_t x429 = S_ikmn_canonical[12] * x399 + S_ikmn_canonical[25] * x400 + S_ikmn_canonical[26] * x401 +
                          S_ikmn_canonical[31] * x396 + S_ikmn_canonical[32] * x397 + S_ikmn_canonical[9] * x398;
    const scalar_t x430 = S_ikmn_canonical[18] * x64 + S_ikmn_canonical[21] * x69 + S_ikmn_canonical[33] * x76 +
                          S_ikmn_canonical[36] * x81 + S_ikmn_canonical[40] * x53 + S_ikmn_canonical[41] * x57;
    const scalar_t x431 = S_ikmn_canonical[16] * x96 + S_ikmn_canonical[22] * x103 + S_ikmn_canonical[31] * x106 +
                          S_ikmn_canonical[37] * x87 + S_ikmn_canonical[40] * x92 + S_ikmn_canonical[43] * x99;
    const scalar_t x432 = S_ikmn_canonical[17] * x121 + S_ikmn_canonical[20] * x123 + S_ikmn_canonical[32] * x110 +
                          S_ikmn_canonical[35] * x112 + S_ikmn_canonical[41] * x116 + S_ikmn_canonical[43] * x118;
    const scalar_t x433 = S_ikmn_canonical[16] * x184 + S_ikmn_canonical[22] * x193 + S_ikmn_canonical[31] * x197 +
                          S_ikmn_canonical[37] * x176 + S_ikmn_canonical[40] * x180 + S_ikmn_canonical[43] * x188;
    const scalar_t x434 = S_ikmn_canonical[17] * x207 + S_ikmn_canonical[20] * x209 + S_ikmn_canonical[32] * x200 +
                          S_ikmn_canonical[35] * x202 + S_ikmn_canonical[41] * x203 + S_ikmn_canonical[43] * x205;
    const scalar_t x435 = S_ikmn_canonical[18] * x241 + S_ikmn_canonical[21] * x245 + S_ikmn_canonical[33] * x246 +
                          S_ikmn_canonical[36] * x250 + S_ikmn_canonical[40] * x237 + S_ikmn_canonical[41] * x240;
    const scalar_t x436 = S_ikmn_canonical[17] * x263 + S_ikmn_canonical[20] * x265 + S_ikmn_canonical[32] * x254 +
                          S_ikmn_canonical[35] * x256 + S_ikmn_canonical[41] * x259 + S_ikmn_canonical[43] * x260;
    const scalar_t x437 = S_ikmn_canonical[17] * x299 + S_ikmn_canonical[20] * x301 + S_ikmn_canonical[32] * x293 +
                          S_ikmn_canonical[35] * x295 + S_ikmn_canonical[41] * x296 + S_ikmn_canonical[43] * x297;
    const scalar_t x438 = S_ikmn_canonical[18] * x331 + S_ikmn_canonical[21] * x332 + S_ikmn_canonical[33] * x335 +
                          S_ikmn_canonical[36] * x336 + S_ikmn_canonical[40] * x326 + S_ikmn_canonical[41] * x328;
    const scalar_t x439 = S_ikmn_canonical[16] * x341 + S_ikmn_canonical[22] * x343 + S_ikmn_canonical[31] * x345 +
                          S_ikmn_canonical[37] * x338 + S_ikmn_canonical[40] * x339 + S_ikmn_canonical[43] * x342;
    const scalar_t x440 = S_ikmn_canonical[16] * x374 + S_ikmn_canonical[22] * x376 + S_ikmn_canonical[31] * x378 +
                          S_ikmn_canonical[37] * x370 + S_ikmn_canonical[40] * x372 + S_ikmn_canonical[43] * x375;
    const scalar_t x441 = S_ikmn_canonical[18] * x398 + S_ikmn_canonical[21] * x399 + S_ikmn_canonical[33] * x400 +
                          S_ikmn_canonical[36] * x401 + S_ikmn_canonical[40] * x396 + S_ikmn_canonical[41] * x397;
    outx[0] = S_ikmn_canonical[0] * x8 + S_ikmn_canonical[10] * x16 + S_ikmn_canonical[14] * x20 + S_ikmn_canonical[15] * x29 +
              S_ikmn_canonical[19] * x33 + S_ikmn_canonical[23] * x36 + S_ikmn_canonical[3] * x40 + S_ikmn_canonical[5] * x43 +
              S_ikmn_canonical[6] * x46 + x107 + x124 + x82;
    outx[1] = S_ikmn_canonical[0] * x133 + S_ikmn_canonical[10] * x141 + S_ikmn_canonical[14] * x145 +
              S_ikmn_canonical[15] * x154 + S_ikmn_canonical[19] * x158 + S_ikmn_canonical[23] * x161 +
              S_ikmn_canonical[3] * x165 + S_ikmn_canonical[5] * x168 + S_ikmn_canonical[6] * x171 + x198 + x210 + x82;
    outx[2] = S_ikmn_canonical[0] * x214 + S_ikmn_canonical[10] * x220 + S_ikmn_canonical[14] * x223 +
              S_ikmn_canonical[15] * x227 + S_ikmn_canonical[19] * x228 + S_ikmn_canonical[23] * x230 +
              S_ikmn_canonical[3] * x231 + S_ikmn_canonical[5] * x233 + S_ikmn_canonical[6] * x235 + x198 + x251 + x266;
    outx[3] = S_ikmn_canonical[0] * x270 + S_ikmn_canonical[10] * x276 + S_ikmn_canonical[14] * x279 +
              S_ikmn_canonical[15] * x283 + S_ikmn_canonical[19] * x284 + S_ikmn_canonical[23] * x286 +
              S_ikmn_canonical[3] * x287 + S_ikmn_canonical[5] * x289 + S_ikmn_canonical[6] * x291 + x107 + x251 + x302;
    outx[4] = S_ikmn_canonical[0] * x307 + S_ikmn_canonical[10] * x310 + S_ikmn_canonical[14] * x311 +
              S_ikmn_canonical[15] * x316 + S_ikmn_canonical[19] * x318 + S_ikmn_canonical[23] * x320 +
              S_ikmn_canonical[3] * x322 + S_ikmn_canonical[5] * x324 + S_ikmn_canonical[6] * x325 + x124 + x337 + x346;
    outx[5] = S_ikmn_canonical[0] * x351 + S_ikmn_canonical[10] * x354 + S_ikmn_canonical[14] * x355 +
              S_ikmn_canonical[15] * x360 + S_ikmn_canonical[19] * x362 + S_ikmn_canonical[23] * x364 +
              S_ikmn_canonical[3] * x366 + S_ikmn_canonical[5] * x368 + S_ikmn_canonical[6] * x369 + x210 + x337 + x379;
    outx[6] = S_ikmn_canonical[0] * x382 + S_ikmn_canonical[10] * x385 + S_ikmn_canonical[14] * x386 +
              S_ikmn_canonical[15] * x389 + S_ikmn_canonical[19] * x390 + S_ikmn_canonical[23] * x391 +
              S_ikmn_canonical[3] * x392 + S_ikmn_canonical[5] * x393 + S_ikmn_canonical[6] * x394 + x266 + x379 + x402;
    outx[7] = S_ikmn_canonical[0] * x405 + S_ikmn_canonical[10] * x408 + S_ikmn_canonical[14] * x409 +
              S_ikmn_canonical[15] * x412 + S_ikmn_canonical[19] * x413 + S_ikmn_canonical[23] * x414 +
              S_ikmn_canonical[3] * x415 + S_ikmn_canonical[5] * x416 + S_ikmn_canonical[6] * x417 + x302 + x346 + x402;
    outy[0] = S_ikmn_canonical[10] * x40 + S_ikmn_canonical[14] * x43 + S_ikmn_canonical[24] * x46 + S_ikmn_canonical[27] * x16 +
              S_ikmn_canonical[29] * x20 + S_ikmn_canonical[30] * x29 + S_ikmn_canonical[34] * x33 + S_ikmn_canonical[38] * x36 +
              S_ikmn_canonical[6] * x8 + x418 + x419 + x420;
    outy[1] = S_ikmn_canonical[10] * x165 + S_ikmn_canonical[14] * x168 + S_ikmn_canonical[24] * x171 +
              S_ikmn_canonical[27] * x141 + S_ikmn_canonical[29] * x145 + S_ikmn_canonical[30] * x154 +
              S_ikmn_canonical[34] * x158 + S_ikmn_canonical[38] * x161 + S_ikmn_canonical[6] * x133 + x418 + x421 + x422;
    outy[2] = S_ikmn_canonical[10] * x231 + S_ikmn_canonical[14] * x233 + S_ikmn_canonical[24] * x235 +
              S_ikmn_canonical[27] * x220 + S_ikmn_canonical[29] * x223 + S_ikmn_canonical[30] * x227 +
              S_ikmn_canonical[34] * x228 + S_ikmn_canonical[38] * x230 + S_ikmn_canonical[6] * x214 + x421 + x423 + x424;
    outy[3] = S_ikmn_canonical[10] * x287 + S_ikmn_canonical[14] * x289 + S_ikmn_canonical[24] * x291 +
              S_ikmn_canonical[27] * x276 + S_ikmn_canonical[29] * x279 + S_ikmn_canonical[30] * x283 +
              S_ikmn_canonical[34] * x284 + S_ikmn_canonical[38] * x286 + S_ikmn_canonical[6] * x270 + x419 + x423 + x425;
    outy[4] = S_ikmn_canonical[10] * x322 + S_ikmn_canonical[14] * x324 + S_ikmn_canonical[24] * x325 +
              S_ikmn_canonical[27] * x310 + S_ikmn_canonical[29] * x311 + S_ikmn_canonical[30] * x316 +
              S_ikmn_canonical[34] * x318 + S_ikmn_canonical[38] * x320 + S_ikmn_canonical[6] * x307 + x420 + x426 + x427;
    outy[5] = S_ikmn_canonical[10] * x366 + S_ikmn_canonical[14] * x368 + S_ikmn_canonical[24] * x369 +
              S_ikmn_canonical[27] * x354 + S_ikmn_canonical[29] * x355 + S_ikmn_canonical[30] * x360 +
              S_ikmn_canonical[34] * x362 + S_ikmn_canonical[38] * x364 + S_ikmn_canonical[6] * x351 + x422 + x426 + x428;
    outy[6] = S_ikmn_canonical[10] * x392 + S_ikmn_canonical[14] * x393 + S_ikmn_canonical[24] * x394 +
              S_ikmn_canonical[27] * x385 + S_ikmn_canonical[29] * x386 + S_ikmn_canonical[30] * x389 +
              S_ikmn_canonical[34] * x390 + S_ikmn_canonical[38] * x391 + S_ikmn_canonical[6] * x382 + x424 + x428 + x429;
    outy[7] = S_ikmn_canonical[10] * x415 + S_ikmn_canonical[14] * x416 + S_ikmn_canonical[24] * x417 +
              S_ikmn_canonical[27] * x408 + S_ikmn_canonical[29] * x409 + S_ikmn_canonical[30] * x412 +
              S_ikmn_canonical[34] * x413 + S_ikmn_canonical[38] * x414 + S_ikmn_canonical[6] * x405 + x425 + x427 + x429;
    outz[0] = S_ikmn_canonical[15] * x8 + S_ikmn_canonical[19] * x40 + S_ikmn_canonical[23] * x43 + S_ikmn_canonical[30] * x46 +
              S_ikmn_canonical[34] * x16 + S_ikmn_canonical[38] * x20 + S_ikmn_canonical[39] * x29 + S_ikmn_canonical[42] * x33 +
              S_ikmn_canonical[44] * x36 + x430 + x431 + x432;
    outz[1] = S_ikmn_canonical[15] * x133 + S_ikmn_canonical[19] * x165 + S_ikmn_canonical[23] * x168 +
              S_ikmn_canonical[30] * x171 + S_ikmn_canonical[34] * x141 + S_ikmn_canonical[38] * x145 +
              S_ikmn_canonical[39] * x154 + S_ikmn_canonical[42] * x158 + S_ikmn_canonical[44] * x161 + x430 + x433 + x434;
    outz[2] = S_ikmn_canonical[15] * x214 + S_ikmn_canonical[19] * x231 + S_ikmn_canonical[23] * x233 +
              S_ikmn_canonical[30] * x235 + S_ikmn_canonical[34] * x220 + S_ikmn_canonical[38] * x223 +
              S_ikmn_canonical[39] * x227 + S_ikmn_canonical[42] * x228 + S_ikmn_canonical[44] * x230 + x433 + x435 + x436;
    outz[3] = S_ikmn_canonical[15] * x270 + S_ikmn_canonical[19] * x287 + S_ikmn_canonical[23] * x289 +
              S_ikmn_canonical[30] * x291 + S_ikmn_canonical[34] * x276 + S_ikmn_canonical[38] * x279 +
              S_ikmn_canonical[39] * x283 + S_ikmn_canonical[42] * x284 + S_ikmn_canonical[44] * x286 + x431 + x435 + x437;
    outz[4] = S_ikmn_canonical[15] * x307 + S_ikmn_canonical[19] * x322 + S_ikmn_canonical[23] * x324 +
              S_ikmn_canonical[30] * x325 + S_ikmn_canonical[34] * x310 + S_ikmn_canonical[38] * x311 +
              S_ikmn_canonical[39] * x316 + S_ikmn_canonical[42] * x318 + S_ikmn_canonical[44] * x320 + x432 + x438 + x439;
    outz[5] = S_ikmn_canonical[15] * x351 + S_ikmn_canonical[19] * x366 + S_ikmn_canonical[23] * x368 +
              S_ikmn_canonical[30] * x369 + S_ikmn_canonical[34] * x354 + S_ikmn_canonical[38] * x355 +
              S_ikmn_canonical[39] * x360 + S_ikmn_canonical[42] * x362 + S_ikmn_canonical[44] * x364 + x434 + x438 + x440;
    outz[6] = S_ikmn_canonical[15] * x382 + S_ikmn_canonical[19] * x392 + S_ikmn_canonical[23] * x393 +
              S_ikmn_canonical[30] * x394 + S_ikmn_canonical[34] * x385 + S_ikmn_canonical[38] * x386 +
              S_ikmn_canonical[39] * x389 + S_ikmn_canonical[42] * x390 + S_ikmn_canonical[44] * x391 + x436 + x440 + x441;
    outz[7] = S_ikmn_canonical[15] * x405 + S_ikmn_canonical[19] * x415 + S_ikmn_canonical[23] * x416 +
              S_ikmn_canonical[30] * x417 + S_ikmn_canonical[34] * x408 + S_ikmn_canonical[38] * x409 +
              S_ikmn_canonical[39] * x412 + S_ikmn_canonical[42] * x413 + S_ikmn_canonical[44] * x414 + x437 + x439 + x441;
}

static SFEM_INLINE void hex8_ref_inc_grad(const scalar_t                      qx,
                                          const scalar_t                      qy,
                                          const scalar_t                      qz,
                                          const scalar_t *const SFEM_RESTRICT incx,
                                          const scalar_t *const SFEM_RESTRICT incy,
                                          const scalar_t *const SFEM_RESTRICT incz,
                                          scalar_t *const SFEM_RESTRICT       inc_grad) {
    // mundane ops: 222 divs: 0 sqrts: 0
    // total ops: 222
    const scalar_t x0  = qy * qz;
    const scalar_t x1  = qz - 1;
    const scalar_t x2  = qy * x1;
    const scalar_t x3  = qy - 1;
    const scalar_t x4  = qz * x3;
    const scalar_t x5  = x1 * x3;
    const scalar_t x6  = qx * qz;
    const scalar_t x7  = qx * x1;
    const scalar_t x8  = qx - 1;
    const scalar_t x9  = qz * x8;
    const scalar_t x10 = x1 * x8;
    const scalar_t x11 = qx * qy;
    const scalar_t x12 = qx * x3;
    const scalar_t x13 = qy * x8;
    const scalar_t x14 = x3 * x8;
    inc_grad[0]        = -incx[0] * x5 + incx[1] * x1 * x3 - incx[2] * x2 + incx[3] * qy * x1 + incx[4] * qz * x3 - incx[5] * x4 +
                  incx[6] * qy * qz - incx[7] * x0;
    inc_grad[1] = -incx[0] * x10 + incx[1] * qx * x1 - incx[2] * x7 + incx[3] * x1 * x8 + incx[4] * qz * x8 - incx[5] * x6 +
                  incx[6] * qx * qz - incx[7] * x9;
    inc_grad[2] = -incx[0] * x14 + incx[1] * qx * x3 - incx[2] * x11 + incx[3] * qy * x8 + incx[4] * x3 * x8 - incx[5] * x12 +
                  incx[6] * qx * qy - incx[7] * x13;
    inc_grad[3] = -incy[0] * x5 + incy[1] * x1 * x3 - incy[2] * x2 + incy[3] * qy * x1 + incy[4] * qz * x3 - incy[5] * x4 +
                  incy[6] * qy * qz - incy[7] * x0;
    inc_grad[4] = -incy[0] * x10 + incy[1] * qx * x1 - incy[2] * x7 + incy[3] * x1 * x8 + incy[4] * qz * x8 - incy[5] * x6 +
                  incy[6] * qx * qz - incy[7] * x9;
    inc_grad[5] = -incy[0] * x14 + incy[1] * qx * x3 - incy[2] * x11 + incy[3] * qy * x8 + incy[4] * x3 * x8 - incy[5] * x12 +
                  incy[6] * qx * qy - incy[7] * x13;
    inc_grad[6] = -incz[0] * x5 + incz[1] * x1 * x3 - incz[2] * x2 + incz[3] * qy * x1 + incz[4] * qz * x3 - incz[5] * x4 +
                  incz[6] * qy * qz - incz[7] * x0;
    inc_grad[7] = -incz[0] * x10 + incz[1] * qx * x1 - incz[2] * x7 + incz[3] * x1 * x8 + incz[4] * qz * x8 - incz[5] * x6 +
                  incz[6] * qx * qz - incz[7] * x9;
    inc_grad[8] = -incz[0] * x14 + incz[1] * qx * x3 - incz[2] * x11 + incz[3] * qy * x8 + incz[4] * x3 * x8 - incz[5] * x12 +
                  incz[6] * qx * qy - incz[7] * x13;
}

static SFEM_INLINE void hex8_Zpkmn(const scalar_t *const SFEM_RESTRICT Wimpn_compressed,
                                   const scalar_t *const SFEM_RESTRICT incx,
                                   const scalar_t *const SFEM_RESTRICT incy,
                                   const scalar_t *const SFEM_RESTRICT incz,
                                   scalar_t *const SFEM_RESTRICT       Zpkmn) {
    // mundane ops: 696 divs: 0 sqrts: 0
    // total ops: 696
    const scalar_t x0   = Wimpn_compressed[0] * incx[0];
    const scalar_t x1   = Wimpn_compressed[7] * incx[6];
    const scalar_t x2   = Wimpn_compressed[3] * incx[2] + Wimpn_compressed[6] * incx[4];
    const scalar_t x3   = x0 + x1 + x2;
    const scalar_t x4   = Wimpn_compressed[2] * incx[1];
    const scalar_t x5   = Wimpn_compressed[9] * incx[7];
    const scalar_t x6   = Wimpn_compressed[3] * incx[5] + Wimpn_compressed[6] * incx[3];
    const scalar_t x7   = x4 + x5 + x6;
    const scalar_t x8   = Wimpn_compressed[1] * incx[0] + Wimpn_compressed[8] * incx[6];
    const scalar_t x9   = Wimpn_compressed[4] * incx[1] + Wimpn_compressed[5] * incx[7];
    const scalar_t x10  = x8 + x9;
    const scalar_t x11  = Wimpn_compressed[4] * incx[2] + Wimpn_compressed[5] * incx[4];
    const scalar_t x12  = Wimpn_compressed[1] * incx[3] + Wimpn_compressed[8] * incx[5];
    const scalar_t x13  = x11 + x12;
    const scalar_t x14  = x10 + x13;
    const scalar_t x15  = Wimpn_compressed[1] * incx[4] + Wimpn_compressed[8] * incx[2];
    const scalar_t x16  = Wimpn_compressed[4] * incx[5] + Wimpn_compressed[5] * incx[3];
    const scalar_t x17  = x15 + x16;
    const scalar_t x18  = x10 + x17;
    const scalar_t x19  = Wimpn_compressed[4] * incx[3] + Wimpn_compressed[5] * incx[5];
    const scalar_t x20  = x19 + x8;
    const scalar_t x21  = Wimpn_compressed[1] * incx[1] + Wimpn_compressed[8] * incx[7];
    const scalar_t x22  = x11 + x21;
    const scalar_t x23  = x20 + x22;
    const scalar_t x24  = Wimpn_compressed[3] * incx[7] + Wimpn_compressed[6] * incx[1];
    const scalar_t x25  = Wimpn_compressed[2] * incx[3] + Wimpn_compressed[9] * incx[5];
    const scalar_t x26  = x24 + x25;
    const scalar_t x27  = Wimpn_compressed[4] * incx[7] + Wimpn_compressed[5] * incx[1];
    const scalar_t x28  = x15 + x27;
    const scalar_t x29  = x20 + x28;
    const scalar_t x30  = x16 + x21;
    const scalar_t x31  = Wimpn_compressed[4] * incx[4] + Wimpn_compressed[5] * incx[2];
    const scalar_t x32  = x31 + x8;
    const scalar_t x33  = x30 + x32;
    const scalar_t x34  = x12 + x27;
    const scalar_t x35  = x32 + x34;
    const scalar_t x36  = x24 + x6;
    const scalar_t x37  = Wimpn_compressed[2] * incx[4] + Wimpn_compressed[9] * incx[2];
    const scalar_t x38  = Wimpn_compressed[0] * incy[0];
    const scalar_t x39  = Wimpn_compressed[7] * incy[6];
    const scalar_t x40  = Wimpn_compressed[3] * incy[2] + Wimpn_compressed[6] * incy[4];
    const scalar_t x41  = x38 + x39 + x40;
    const scalar_t x42  = Wimpn_compressed[2] * incy[1];
    const scalar_t x43  = Wimpn_compressed[9] * incy[7];
    const scalar_t x44  = Wimpn_compressed[3] * incy[5] + Wimpn_compressed[6] * incy[3];
    const scalar_t x45  = x42 + x43 + x44;
    const scalar_t x46  = Wimpn_compressed[1] * incy[0] + Wimpn_compressed[8] * incy[6];
    const scalar_t x47  = Wimpn_compressed[4] * incy[1] + Wimpn_compressed[5] * incy[7];
    const scalar_t x48  = x46 + x47;
    const scalar_t x49  = Wimpn_compressed[4] * incy[2] + Wimpn_compressed[5] * incy[4];
    const scalar_t x50  = Wimpn_compressed[1] * incy[3] + Wimpn_compressed[8] * incy[5];
    const scalar_t x51  = x49 + x50;
    const scalar_t x52  = x48 + x51;
    const scalar_t x53  = Wimpn_compressed[1] * incy[4] + Wimpn_compressed[8] * incy[2];
    const scalar_t x54  = Wimpn_compressed[4] * incy[5] + Wimpn_compressed[5] * incy[3];
    const scalar_t x55  = x53 + x54;
    const scalar_t x56  = x48 + x55;
    const scalar_t x57  = Wimpn_compressed[4] * incy[3] + Wimpn_compressed[5] * incy[5];
    const scalar_t x58  = x46 + x57;
    const scalar_t x59  = Wimpn_compressed[1] * incy[1] + Wimpn_compressed[8] * incy[7];
    const scalar_t x60  = x49 + x59;
    const scalar_t x61  = x58 + x60;
    const scalar_t x62  = Wimpn_compressed[3] * incy[7] + Wimpn_compressed[6] * incy[1];
    const scalar_t x63  = Wimpn_compressed[2] * incy[3] + Wimpn_compressed[9] * incy[5];
    const scalar_t x64  = x62 + x63;
    const scalar_t x65  = Wimpn_compressed[4] * incy[7] + Wimpn_compressed[5] * incy[1];
    const scalar_t x66  = x53 + x65;
    const scalar_t x67  = x58 + x66;
    const scalar_t x68  = x54 + x59;
    const scalar_t x69  = Wimpn_compressed[4] * incy[4] + Wimpn_compressed[5] * incy[2];
    const scalar_t x70  = x46 + x69;
    const scalar_t x71  = x68 + x70;
    const scalar_t x72  = x50 + x65;
    const scalar_t x73  = x70 + x72;
    const scalar_t x74  = x44 + x62;
    const scalar_t x75  = Wimpn_compressed[2] * incy[4] + Wimpn_compressed[9] * incy[2];
    const scalar_t x76  = Wimpn_compressed[0] * incz[0];
    const scalar_t x77  = Wimpn_compressed[7] * incz[6];
    const scalar_t x78  = Wimpn_compressed[3] * incz[2] + Wimpn_compressed[6] * incz[4];
    const scalar_t x79  = x76 + x77 + x78;
    const scalar_t x80  = Wimpn_compressed[2] * incz[1];
    const scalar_t x81  = Wimpn_compressed[9] * incz[7];
    const scalar_t x82  = Wimpn_compressed[3] * incz[5] + Wimpn_compressed[6] * incz[3];
    const scalar_t x83  = x80 + x81 + x82;
    const scalar_t x84  = Wimpn_compressed[1] * incz[0] + Wimpn_compressed[8] * incz[6];
    const scalar_t x85  = Wimpn_compressed[4] * incz[1] + Wimpn_compressed[5] * incz[7];
    const scalar_t x86  = x84 + x85;
    const scalar_t x87  = Wimpn_compressed[4] * incz[2] + Wimpn_compressed[5] * incz[4];
    const scalar_t x88  = Wimpn_compressed[1] * incz[3] + Wimpn_compressed[8] * incz[5];
    const scalar_t x89  = x87 + x88;
    const scalar_t x90  = x86 + x89;
    const scalar_t x91  = Wimpn_compressed[1] * incz[4] + Wimpn_compressed[8] * incz[2];
    const scalar_t x92  = Wimpn_compressed[4] * incz[5] + Wimpn_compressed[5] * incz[3];
    const scalar_t x93  = x91 + x92;
    const scalar_t x94  = x86 + x93;
    const scalar_t x95  = Wimpn_compressed[4] * incz[3] + Wimpn_compressed[5] * incz[5];
    const scalar_t x96  = x84 + x95;
    const scalar_t x97  = Wimpn_compressed[1] * incz[1] + Wimpn_compressed[8] * incz[7];
    const scalar_t x98  = x87 + x97;
    const scalar_t x99  = x96 + x98;
    const scalar_t x100 = Wimpn_compressed[3] * incz[7] + Wimpn_compressed[6] * incz[1];
    const scalar_t x101 = Wimpn_compressed[2] * incz[3] + Wimpn_compressed[9] * incz[5];
    const scalar_t x102 = x100 + x101;
    const scalar_t x103 = Wimpn_compressed[4] * incz[7] + Wimpn_compressed[5] * incz[1];
    const scalar_t x104 = x103 + x91;
    const scalar_t x105 = x104 + x96;
    const scalar_t x106 = x92 + x97;
    const scalar_t x107 = Wimpn_compressed[4] * incz[4] + Wimpn_compressed[5] * incz[2];
    const scalar_t x108 = x107 + x84;
    const scalar_t x109 = x106 + x108;
    const scalar_t x110 = x103 + x88;
    const scalar_t x111 = x108 + x110;
    const scalar_t x112 = x100 + x82;
    const scalar_t x113 = Wimpn_compressed[2] * incz[4] + Wimpn_compressed[9] * incz[2];
    const scalar_t x114 = Wimpn_compressed[0] * incx[1];
    const scalar_t x115 = Wimpn_compressed[7] * incx[7];
    const scalar_t x116 = Wimpn_compressed[3] * incx[3] + Wimpn_compressed[6] * incx[5];
    const scalar_t x117 = x114 + x115 + x116;
    const scalar_t x118 = Wimpn_compressed[2] * incx[0];
    const scalar_t x119 = Wimpn_compressed[9] * incx[6];
    const scalar_t x120 = Wimpn_compressed[3] * incx[4] + Wimpn_compressed[6] * incx[2];
    const scalar_t x121 = x118 + x119 + x120;
    const scalar_t x122 = Wimpn_compressed[1] * incx[2] + Wimpn_compressed[8] * incx[4];
    const scalar_t x123 = Wimpn_compressed[4] * incx[0] + Wimpn_compressed[5] * incx[6];
    const scalar_t x124 = x123 + x9;
    const scalar_t x125 = x12 + x122 + x124;
    const scalar_t x126 = Wimpn_compressed[3] * incx[6] + Wimpn_compressed[6] * incx[0];
    const scalar_t x127 = Wimpn_compressed[2] * incx[2] + Wimpn_compressed[9] * incx[4];
    const scalar_t x128 = x126 + x127;
    const scalar_t x129 = Wimpn_compressed[1] * incx[5] + Wimpn_compressed[8] * incx[3];
    const scalar_t x130 = Wimpn_compressed[4] * incx[6] + Wimpn_compressed[5] * incx[0];
    const scalar_t x131 = x129 + x130;
    const scalar_t x132 = x131 + x22;
    const scalar_t x133 = x124 + x129 + x15;
    const scalar_t x134 = x122 + x130;
    const scalar_t x135 = x134 + x30;
    const scalar_t x136 = x120 + x126;
    const scalar_t x137 = Wimpn_compressed[2] * incx[5] + Wimpn_compressed[9] * incx[3];
    const scalar_t x138 = Wimpn_compressed[0] * incy[1];
    const scalar_t x139 = Wimpn_compressed[7] * incy[7];
    const scalar_t x140 = Wimpn_compressed[3] * incy[3] + Wimpn_compressed[6] * incy[5];
    const scalar_t x141 = x138 + x139 + x140;
    const scalar_t x142 = Wimpn_compressed[2] * incy[0];
    const scalar_t x143 = Wimpn_compressed[9] * incy[6];
    const scalar_t x144 = Wimpn_compressed[3] * incy[4] + Wimpn_compressed[6] * incy[2];
    const scalar_t x145 = x142 + x143 + x144;
    const scalar_t x146 = Wimpn_compressed[1] * incy[2] + Wimpn_compressed[8] * incy[4];
    const scalar_t x147 = Wimpn_compressed[4] * incy[0] + Wimpn_compressed[5] * incy[6];
    const scalar_t x148 = x147 + x47;
    const scalar_t x149 = x146 + x148 + x50;
    const scalar_t x150 = Wimpn_compressed[3] * incy[6] + Wimpn_compressed[6] * incy[0];
    const scalar_t x151 = Wimpn_compressed[2] * incy[2] + Wimpn_compressed[9] * incy[4];
    const scalar_t x152 = x150 + x151;
    const scalar_t x153 = Wimpn_compressed[1] * incy[5] + Wimpn_compressed[8] * incy[3];
    const scalar_t x154 = Wimpn_compressed[4] * incy[6] + Wimpn_compressed[5] * incy[0];
    const scalar_t x155 = x153 + x154;
    const scalar_t x156 = x155 + x60;
    const scalar_t x157 = x148 + x153 + x53;
    const scalar_t x158 = x146 + x154;
    const scalar_t x159 = x158 + x68;
    const scalar_t x160 = x144 + x150;
    const scalar_t x161 = Wimpn_compressed[2] * incy[5] + Wimpn_compressed[9] * incy[3];
    const scalar_t x162 = Wimpn_compressed[0] * incz[1];
    const scalar_t x163 = Wimpn_compressed[7] * incz[7];
    const scalar_t x164 = Wimpn_compressed[3] * incz[3] + Wimpn_compressed[6] * incz[5];
    const scalar_t x165 = x162 + x163 + x164;
    const scalar_t x166 = Wimpn_compressed[2] * incz[0];
    const scalar_t x167 = Wimpn_compressed[9] * incz[6];
    const scalar_t x168 = Wimpn_compressed[3] * incz[4] + Wimpn_compressed[6] * incz[2];
    const scalar_t x169 = x166 + x167 + x168;
    const scalar_t x170 = Wimpn_compressed[1] * incz[2] + Wimpn_compressed[8] * incz[4];
    const scalar_t x171 = Wimpn_compressed[4] * incz[0] + Wimpn_compressed[5] * incz[6];
    const scalar_t x172 = x171 + x85;
    const scalar_t x173 = x170 + x172 + x88;
    const scalar_t x174 = Wimpn_compressed[3] * incz[6] + Wimpn_compressed[6] * incz[0];
    const scalar_t x175 = Wimpn_compressed[2] * incz[2] + Wimpn_compressed[9] * incz[4];
    const scalar_t x176 = x174 + x175;
    const scalar_t x177 = Wimpn_compressed[1] * incz[5] + Wimpn_compressed[8] * incz[3];
    const scalar_t x178 = Wimpn_compressed[4] * incz[6] + Wimpn_compressed[5] * incz[0];
    const scalar_t x179 = x177 + x178;
    const scalar_t x180 = x179 + x98;
    const scalar_t x181 = x172 + x177 + x91;
    const scalar_t x182 = x170 + x178;
    const scalar_t x183 = x106 + x182;
    const scalar_t x184 = x168 + x174;
    const scalar_t x185 = Wimpn_compressed[2] * incz[5] + Wimpn_compressed[9] * incz[3];
    const scalar_t x186 = Wimpn_compressed[0] * incx[2] + Wimpn_compressed[7] * incx[4];
    const scalar_t x187 = Wimpn_compressed[3] * incx[0] + Wimpn_compressed[6] * incx[6];
    const scalar_t x188 = x186 + x187;
    const scalar_t x189 = x123 + x19;
    const scalar_t x190 = x122 + x189 + x21;
    const scalar_t x191 = Wimpn_compressed[1] * incx[7] + Wimpn_compressed[8] * incx[1];
    const scalar_t x192 = x130 + x191;
    const scalar_t x193 = x13 + x192;
    const scalar_t x194 = Wimpn_compressed[1] * incx[6] + Wimpn_compressed[8] * incx[0];
    const scalar_t x195 = x11 + x194;
    const scalar_t x196 = x19 + x191 + x195;
    const scalar_t x197 = x129 + x195 + x9;
    const scalar_t x198 = Wimpn_compressed[2] * incx[6] + Wimpn_compressed[9] * incx[0];
    const scalar_t x199 = Wimpn_compressed[0] * incy[2] + Wimpn_compressed[7] * incy[4];
    const scalar_t x200 = Wimpn_compressed[3] * incy[0] + Wimpn_compressed[6] * incy[6];
    const scalar_t x201 = x199 + x200;
    const scalar_t x202 = x147 + x57;
    const scalar_t x203 = x146 + x202 + x59;
    const scalar_t x204 = Wimpn_compressed[1] * incy[7] + Wimpn_compressed[8] * incy[1];
    const scalar_t x205 = x154 + x204;
    const scalar_t x206 = x205 + x51;
    const scalar_t x207 = Wimpn_compressed[1] * incy[6] + Wimpn_compressed[8] * incy[0];
    const scalar_t x208 = x207 + x49;
    const scalar_t x209 = x204 + x208 + x57;
    const scalar_t x210 = x153 + x208 + x47;
    const scalar_t x211 = Wimpn_compressed[2] * incy[6] + Wimpn_compressed[9] * incy[0];
    const scalar_t x212 = Wimpn_compressed[0] * incz[2] + Wimpn_compressed[7] * incz[4];
    const scalar_t x213 = Wimpn_compressed[3] * incz[0] + Wimpn_compressed[6] * incz[6];
    const scalar_t x214 = x212 + x213;
    const scalar_t x215 = x171 + x95;
    const scalar_t x216 = x170 + x215 + x97;
    const scalar_t x217 = Wimpn_compressed[1] * incz[7] + Wimpn_compressed[8] * incz[1];
    const scalar_t x218 = x178 + x217;
    const scalar_t x219 = x218 + x89;
    const scalar_t x220 = Wimpn_compressed[1] * incz[6] + Wimpn_compressed[8] * incz[0];
    const scalar_t x221 = x220 + x87;
    const scalar_t x222 = x217 + x221 + x95;
    const scalar_t x223 = x177 + x221 + x85;
    const scalar_t x224 = Wimpn_compressed[2] * incz[6] + Wimpn_compressed[9] * incz[0];
    const scalar_t x225 = Wimpn_compressed[0] * incx[3] + Wimpn_compressed[7] * incx[5];
    const scalar_t x226 = Wimpn_compressed[3] * incx[1] + Wimpn_compressed[6] * incx[7];
    const scalar_t x227 = x225 + x226;
    const scalar_t x228 = x134 + x34;
    const scalar_t x229 = x15 + x189 + x191;
    const scalar_t x230 = Wimpn_compressed[2] * incx[7] + Wimpn_compressed[9] * incx[1];
    const scalar_t x231 = Wimpn_compressed[0] * incy[3] + Wimpn_compressed[7] * incy[5];
    const scalar_t x232 = Wimpn_compressed[3] * incy[1] + Wimpn_compressed[6] * incy[7];
    const scalar_t x233 = x231 + x232;
    const scalar_t x234 = x158 + x72;
    const scalar_t x235 = x202 + x204 + x53;
    const scalar_t x236 = Wimpn_compressed[2] * incy[7] + Wimpn_compressed[9] * incy[1];
    const scalar_t x237 = Wimpn_compressed[0] * incz[3] + Wimpn_compressed[7] * incz[5];
    const scalar_t x238 = Wimpn_compressed[3] * incz[1] + Wimpn_compressed[6] * incz[7];
    const scalar_t x239 = x237 + x238;
    const scalar_t x240 = x110 + x182;
    const scalar_t x241 = x215 + x217 + x91;
    const scalar_t x242 = Wimpn_compressed[2] * incz[7] + Wimpn_compressed[9] * incz[1];
    const scalar_t x243 = x137 + x226;
    const scalar_t x244 = Wimpn_compressed[0] * incx[4];
    const scalar_t x245 = Wimpn_compressed[7] * incx[2];
    const scalar_t x246 = x126 + x244 + x245;
    const scalar_t x247 = x17 + x192;
    const scalar_t x248 = x123 + x31;
    const scalar_t x249 = x129 + x21 + x248;
    const scalar_t x250 = x131 + x28;
    const scalar_t x251 = x116 + x230;
    const scalar_t x252 = x12 + x191 + x248;
    const scalar_t x253 = x116 + x226;
    const scalar_t x254 = x161 + x232;
    const scalar_t x255 = Wimpn_compressed[0] * incy[4];
    const scalar_t x256 = Wimpn_compressed[7] * incy[2];
    const scalar_t x257 = x150 + x255 + x256;
    const scalar_t x258 = x205 + x55;
    const scalar_t x259 = x147 + x69;
    const scalar_t x260 = x153 + x259 + x59;
    const scalar_t x261 = x155 + x66;
    const scalar_t x262 = x140 + x236;
    const scalar_t x263 = x204 + x259 + x50;
    const scalar_t x264 = x140 + x232;
    const scalar_t x265 = x185 + x238;
    const scalar_t x266 = Wimpn_compressed[0] * incz[4];
    const scalar_t x267 = Wimpn_compressed[7] * incz[2];
    const scalar_t x268 = x174 + x266 + x267;
    const scalar_t x269 = x218 + x93;
    const scalar_t x270 = x107 + x171;
    const scalar_t x271 = x177 + x270 + x97;
    const scalar_t x272 = x104 + x179;
    const scalar_t x273 = x164 + x242;
    const scalar_t x274 = x217 + x270 + x88;
    const scalar_t x275 = x164 + x238;
    const scalar_t x276 = x187 + x37;
    const scalar_t x277 = Wimpn_compressed[0] * incx[5];
    const scalar_t x278 = Wimpn_compressed[7] * incx[3];
    const scalar_t x279 = x24 + x277 + x278;
    const scalar_t x280 = x16 + x194;
    const scalar_t x281 = x191 + x280 + x31;
    const scalar_t x282 = x198 + x2;
    const scalar_t x283 = x122 + x280 + x9;
    const scalar_t x284 = x187 + x2;
    const scalar_t x285 = x200 + x75;
    const scalar_t x286 = Wimpn_compressed[0] * incy[5];
    const scalar_t x287 = Wimpn_compressed[7] * incy[3];
    const scalar_t x288 = x286 + x287 + x62;
    const scalar_t x289 = x207 + x54;
    const scalar_t x290 = x204 + x289 + x69;
    const scalar_t x291 = x211 + x40;
    const scalar_t x292 = x146 + x289 + x47;
    const scalar_t x293 = x200 + x40;
    const scalar_t x294 = x113 + x213;
    const scalar_t x295 = Wimpn_compressed[0] * incz[5];
    const scalar_t x296 = Wimpn_compressed[7] * incz[3];
    const scalar_t x297 = x100 + x295 + x296;
    const scalar_t x298 = x220 + x92;
    const scalar_t x299 = x107 + x217 + x298;
    const scalar_t x300 = x224 + x78;
    const scalar_t x301 = x170 + x298 + x85;
    const scalar_t x302 = x213 + x78;
    const scalar_t x303 = Wimpn_compressed[0] * incx[6] + Wimpn_compressed[7] * incx[0];
    const scalar_t x304 = x120 + x303;
    const scalar_t x305 = x194 + x27;
    const scalar_t x306 = x129 + x305 + x31;
    const scalar_t x307 = x122 + x19 + x305;
    const scalar_t x308 = Wimpn_compressed[0] * incy[6] + Wimpn_compressed[7] * incy[0];
    const scalar_t x309 = x144 + x308;
    const scalar_t x310 = x207 + x65;
    const scalar_t x311 = x153 + x310 + x69;
    const scalar_t x312 = x146 + x310 + x57;
    const scalar_t x313 = Wimpn_compressed[0] * incz[6] + Wimpn_compressed[7] * incz[0];
    const scalar_t x314 = x168 + x313;
    const scalar_t x315 = x103 + x220;
    const scalar_t x316 = x107 + x177 + x315;
    const scalar_t x317 = x170 + x315 + x95;
    const scalar_t x318 = Wimpn_compressed[0] * incx[7] + Wimpn_compressed[7] * incx[1];
    const scalar_t x319 = x318 + x6;
    const scalar_t x320 = Wimpn_compressed[0] * incy[7] + Wimpn_compressed[7] * incy[1];
    const scalar_t x321 = x320 + x44;
    const scalar_t x322 = Wimpn_compressed[0] * incz[7] + Wimpn_compressed[7] * incz[1];
    const scalar_t x323 = x322 + x82;
    Zpkmn[0]            = x3 + x7;
    Zpkmn[1]            = x14;
    Zpkmn[2]            = x18;
    Zpkmn[3]            = x23;
    Zpkmn[4]            = x26 + x3;
    Zpkmn[5]            = x29;
    Zpkmn[6]            = x33;
    Zpkmn[7]            = x35;
    Zpkmn[8]            = x0 + x1 + x36 + x37;
    Zpkmn[9]            = x41 + x45;
    Zpkmn[10]           = x52;
    Zpkmn[11]           = x56;
    Zpkmn[12]           = x61;
    Zpkmn[13]           = x41 + x64;
    Zpkmn[14]           = x67;
    Zpkmn[15]           = x71;
    Zpkmn[16]           = x73;
    Zpkmn[17]           = x38 + x39 + x74 + x75;
    Zpkmn[18]           = x79 + x83;
    Zpkmn[19]           = x90;
    Zpkmn[20]           = x94;
    Zpkmn[21]           = x99;
    Zpkmn[22]           = x102 + x79;
    Zpkmn[23]           = x105;
    Zpkmn[24]           = x109;
    Zpkmn[25]           = x111;
    Zpkmn[26]           = x112 + x113 + x76 + x77;
    Zpkmn[27]           = x117 + x121;
    Zpkmn[28]           = x14;
    Zpkmn[29]           = x18;
    Zpkmn[30]           = x125;
    Zpkmn[31]           = x117 + x128;
    Zpkmn[32]           = x132;
    Zpkmn[33]           = x133;
    Zpkmn[34]           = x135;
    Zpkmn[35]           = x114 + x115 + x136 + x137;
    Zpkmn[36]           = x141 + x145;
    Zpkmn[37]           = x52;
    Zpkmn[38]           = x56;
    Zpkmn[39]           = x149;
    Zpkmn[40]           = x141 + x152;
    Zpkmn[41]           = x156;
    Zpkmn[42]           = x157;
    Zpkmn[43]           = x159;
    Zpkmn[44]           = x138 + x139 + x160 + x161;
    Zpkmn[45]           = x165 + x169;
    Zpkmn[46]           = x90;
    Zpkmn[47]           = x94;
    Zpkmn[48]           = x173;
    Zpkmn[49]           = x165 + x176;
    Zpkmn[50]           = x180;
    Zpkmn[51]           = x181;
    Zpkmn[52]           = x183;
    Zpkmn[53]           = x162 + x163 + x184 + x185;
    Zpkmn[54]           = x188 + x26;
    Zpkmn[55]           = x190;
    Zpkmn[56]           = x193;
    Zpkmn[57]           = x125;
    Zpkmn[58]           = x188 + x7;
    Zpkmn[59]           = x132;
    Zpkmn[60]           = x196;
    Zpkmn[61]           = x197;
    Zpkmn[62]           = x186 + x198 + x36;
    Zpkmn[63]           = x201 + x64;
    Zpkmn[64]           = x203;
    Zpkmn[65]           = x206;
    Zpkmn[66]           = x149;
    Zpkmn[67]           = x201 + x45;
    Zpkmn[68]           = x156;
    Zpkmn[69]           = x209;
    Zpkmn[70]           = x210;
    Zpkmn[71]           = x199 + x211 + x74;
    Zpkmn[72]           = x102 + x214;
    Zpkmn[73]           = x216;
    Zpkmn[74]           = x219;
    Zpkmn[75]           = x173;
    Zpkmn[76]           = x214 + x83;
    Zpkmn[77]           = x180;
    Zpkmn[78]           = x222;
    Zpkmn[79]           = x223;
    Zpkmn[80]           = x112 + x212 + x224;
    Zpkmn[81]           = x128 + x227;
    Zpkmn[82]           = x190;
    Zpkmn[83]           = x193;
    Zpkmn[84]           = x23;
    Zpkmn[85]           = x121 + x227;
    Zpkmn[86]           = x29;
    Zpkmn[87]           = x228;
    Zpkmn[88]           = x229;
    Zpkmn[89]           = x136 + x225 + x230;
    Zpkmn[90]           = x152 + x233;
    Zpkmn[91]           = x203;
    Zpkmn[92]           = x206;
    Zpkmn[93]           = x61;
    Zpkmn[94]           = x145 + x233;
    Zpkmn[95]           = x67;
    Zpkmn[96]           = x234;
    Zpkmn[97]           = x235;
    Zpkmn[98]           = x160 + x231 + x236;
    Zpkmn[99]           = x176 + x239;
    Zpkmn[100]          = x216;
    Zpkmn[101]          = x219;
    Zpkmn[102]          = x99;
    Zpkmn[103]          = x169 + x239;
    Zpkmn[104]          = x105;
    Zpkmn[105]          = x240;
    Zpkmn[106]          = x241;
    Zpkmn[107]          = x184 + x237 + x242;
    Zpkmn[108]          = x243 + x246;
    Zpkmn[109]          = x247;
    Zpkmn[110]          = x249;
    Zpkmn[111]          = x250;
    Zpkmn[112]          = x246 + x251;
    Zpkmn[113]          = x252;
    Zpkmn[114]          = x33;
    Zpkmn[115]          = x35;
    Zpkmn[116]          = x118 + x119 + x244 + x245 + x253;
    Zpkmn[117]          = x254 + x257;
    Zpkmn[118]          = x258;
    Zpkmn[119]          = x260;
    Zpkmn[120]          = x261;
    Zpkmn[121]          = x257 + x262;
    Zpkmn[122]          = x263;
    Zpkmn[123]          = x71;
    Zpkmn[124]          = x73;
    Zpkmn[125]          = x142 + x143 + x255 + x256 + x264;
    Zpkmn[126]          = x265 + x268;
    Zpkmn[127]          = x269;
    Zpkmn[128]          = x271;
    Zpkmn[129]          = x272;
    Zpkmn[130]          = x268 + x273;
    Zpkmn[131]          = x274;
    Zpkmn[132]          = x109;
    Zpkmn[133]          = x111;
    Zpkmn[134]          = x166 + x167 + x266 + x267 + x275;
    Zpkmn[135]          = x276 + x279;
    Zpkmn[136]          = x247;
    Zpkmn[137]          = x249;
    Zpkmn[138]          = x281;
    Zpkmn[139]          = x279 + x282;
    Zpkmn[140]          = x283;
    Zpkmn[141]          = x133;
    Zpkmn[142]          = x135;
    Zpkmn[143]          = x277 + x278 + x284 + x4 + x5;
    Zpkmn[144]          = x285 + x288;
    Zpkmn[145]          = x258;
    Zpkmn[146]          = x260;
    Zpkmn[147]          = x290;
    Zpkmn[148]          = x288 + x291;
    Zpkmn[149]          = x292;
    Zpkmn[150]          = x157;
    Zpkmn[151]          = x159;
    Zpkmn[152]          = x286 + x287 + x293 + x42 + x43;
    Zpkmn[153]          = x294 + x297;
    Zpkmn[154]          = x269;
    Zpkmn[155]          = x271;
    Zpkmn[156]          = x299;
    Zpkmn[157]          = x297 + x300;
    Zpkmn[158]          = x301;
    Zpkmn[159]          = x181;
    Zpkmn[160]          = x183;
    Zpkmn[161]          = x295 + x296 + x302 + x80 + x81;
    Zpkmn[162]          = x251 + x304;
    Zpkmn[163]          = x306;
    Zpkmn[164]          = x307;
    Zpkmn[165]          = x281;
    Zpkmn[166]          = x243 + x304;
    Zpkmn[167]          = x283;
    Zpkmn[168]          = x196;
    Zpkmn[169]          = x197;
    Zpkmn[170]          = x127 + x253 + x303;
    Zpkmn[171]          = x262 + x309;
    Zpkmn[172]          = x311;
    Zpkmn[173]          = x312;
    Zpkmn[174]          = x290;
    Zpkmn[175]          = x254 + x309;
    Zpkmn[176]          = x292;
    Zpkmn[177]          = x209;
    Zpkmn[178]          = x210;
    Zpkmn[179]          = x151 + x264 + x308;
    Zpkmn[180]          = x273 + x314;
    Zpkmn[181]          = x316;
    Zpkmn[182]          = x317;
    Zpkmn[183]          = x299;
    Zpkmn[184]          = x265 + x314;
    Zpkmn[185]          = x301;
    Zpkmn[186]          = x222;
    Zpkmn[187]          = x223;
    Zpkmn[188]          = x175 + x275 + x313;
    Zpkmn[189]          = x282 + x319;
    Zpkmn[190]          = x306;
    Zpkmn[191]          = x307;
    Zpkmn[192]          = x250;
    Zpkmn[193]          = x276 + x319;
    Zpkmn[194]          = x252;
    Zpkmn[195]          = x228;
    Zpkmn[196]          = x229;
    Zpkmn[197]          = x25 + x284 + x318;
    Zpkmn[198]          = x291 + x321;
    Zpkmn[199]          = x311;
    Zpkmn[200]          = x312;
    Zpkmn[201]          = x261;
    Zpkmn[202]          = x285 + x321;
    Zpkmn[203]          = x263;
    Zpkmn[204]          = x234;
    Zpkmn[205]          = x235;
    Zpkmn[206]          = x293 + x320 + x63;
    Zpkmn[207]          = x300 + x323;
    Zpkmn[208]          = x316;
    Zpkmn[209]          = x317;
    Zpkmn[210]          = x272;
    Zpkmn[211]          = x294 + x323;
    Zpkmn[212]          = x274;
    Zpkmn[213]          = x240;
    Zpkmn[214]          = x241;
    Zpkmn[215]          = x101 + x302 + x322;
}

static SFEM_INLINE void hex8_SdotZ(const scalar_t *const SFEM_RESTRICT S_ikmn_canonical,
                                   const scalar_t *const SFEM_RESTRICT Zpkmn,
                                   scalar_t *const SFEM_RESTRICT       outx,
                                   scalar_t *const SFEM_RESTRICT       outy,
                                   scalar_t *const SFEM_RESTRICT       outz) {
    // mundane ops: 1272 divs: 0 sqrts: 0
    // total ops: 1272
    outx[0] = S_ikmn_canonical[0] * Zpkmn[0] + S_ikmn_canonical[10] * Zpkmn[13] + S_ikmn_canonical[11] * Zpkmn[14] +
              S_ikmn_canonical[12] * Zpkmn[15] + S_ikmn_canonical[13] * Zpkmn[16] + S_ikmn_canonical[14] * Zpkmn[17] +
              S_ikmn_canonical[15] * Zpkmn[18] + S_ikmn_canonical[16] * Zpkmn[19] + S_ikmn_canonical[17] * Zpkmn[20] +
              S_ikmn_canonical[18] * Zpkmn[21] + S_ikmn_canonical[19] * Zpkmn[22] + S_ikmn_canonical[1] * Zpkmn[1] +
              S_ikmn_canonical[1] * Zpkmn[3] + S_ikmn_canonical[20] * Zpkmn[23] + S_ikmn_canonical[21] * Zpkmn[24] +
              S_ikmn_canonical[22] * Zpkmn[25] + S_ikmn_canonical[23] * Zpkmn[26] + S_ikmn_canonical[2] * Zpkmn[2] +
              S_ikmn_canonical[2] * Zpkmn[6] + S_ikmn_canonical[3] * Zpkmn[4] + S_ikmn_canonical[4] * Zpkmn[5] +
              S_ikmn_canonical[4] * Zpkmn[7] + S_ikmn_canonical[5] * Zpkmn[8] + S_ikmn_canonical[6] * Zpkmn[9] +
              S_ikmn_canonical[7] * Zpkmn[10] + S_ikmn_canonical[8] * Zpkmn[11] + S_ikmn_canonical[9] * Zpkmn[12];
    outx[1] = S_ikmn_canonical[0] * Zpkmn[27] + S_ikmn_canonical[10] * Zpkmn[40] + S_ikmn_canonical[11] * Zpkmn[41] +
              S_ikmn_canonical[12] * Zpkmn[42] + S_ikmn_canonical[13] * Zpkmn[43] + S_ikmn_canonical[14] * Zpkmn[44] +
              S_ikmn_canonical[15] * Zpkmn[45] + S_ikmn_canonical[16] * Zpkmn[46] + S_ikmn_canonical[17] * Zpkmn[47] +
              S_ikmn_canonical[18] * Zpkmn[48] + S_ikmn_canonical[19] * Zpkmn[49] + S_ikmn_canonical[1] * Zpkmn[28] +
              S_ikmn_canonical[1] * Zpkmn[30] + S_ikmn_canonical[20] * Zpkmn[50] + S_ikmn_canonical[21] * Zpkmn[51] +
              S_ikmn_canonical[22] * Zpkmn[52] + S_ikmn_canonical[23] * Zpkmn[53] + S_ikmn_canonical[2] * Zpkmn[29] +
              S_ikmn_canonical[2] * Zpkmn[33] + S_ikmn_canonical[3] * Zpkmn[31] + S_ikmn_canonical[4] * Zpkmn[32] +
              S_ikmn_canonical[4] * Zpkmn[34] + S_ikmn_canonical[5] * Zpkmn[35] + S_ikmn_canonical[6] * Zpkmn[36] +
              S_ikmn_canonical[7] * Zpkmn[37] + S_ikmn_canonical[8] * Zpkmn[38] + S_ikmn_canonical[9] * Zpkmn[39];
    outx[2] = S_ikmn_canonical[0] * Zpkmn[54] + S_ikmn_canonical[10] * Zpkmn[67] + S_ikmn_canonical[11] * Zpkmn[68] +
              S_ikmn_canonical[12] * Zpkmn[69] + S_ikmn_canonical[13] * Zpkmn[70] + S_ikmn_canonical[14] * Zpkmn[71] +
              S_ikmn_canonical[15] * Zpkmn[72] + S_ikmn_canonical[16] * Zpkmn[73] + S_ikmn_canonical[17] * Zpkmn[74] +
              S_ikmn_canonical[18] * Zpkmn[75] + S_ikmn_canonical[19] * Zpkmn[76] + S_ikmn_canonical[1] * Zpkmn[55] +
              S_ikmn_canonical[1] * Zpkmn[57] + S_ikmn_canonical[20] * Zpkmn[77] + S_ikmn_canonical[21] * Zpkmn[78] +
              S_ikmn_canonical[22] * Zpkmn[79] + S_ikmn_canonical[23] * Zpkmn[80] + S_ikmn_canonical[2] * Zpkmn[56] +
              S_ikmn_canonical[2] * Zpkmn[60] + S_ikmn_canonical[3] * Zpkmn[58] + S_ikmn_canonical[4] * Zpkmn[59] +
              S_ikmn_canonical[4] * Zpkmn[61] + S_ikmn_canonical[5] * Zpkmn[62] + S_ikmn_canonical[6] * Zpkmn[63] +
              S_ikmn_canonical[7] * Zpkmn[64] + S_ikmn_canonical[8] * Zpkmn[65] + S_ikmn_canonical[9] * Zpkmn[66];
    outx[3] = S_ikmn_canonical[0] * Zpkmn[81] + S_ikmn_canonical[10] * Zpkmn[94] + S_ikmn_canonical[11] * Zpkmn[95] +
              S_ikmn_canonical[12] * Zpkmn[96] + S_ikmn_canonical[13] * Zpkmn[97] + S_ikmn_canonical[14] * Zpkmn[98] +
              S_ikmn_canonical[15] * Zpkmn[99] + S_ikmn_canonical[16] * Zpkmn[100] + S_ikmn_canonical[17] * Zpkmn[101] +
              S_ikmn_canonical[18] * Zpkmn[102] + S_ikmn_canonical[19] * Zpkmn[103] + S_ikmn_canonical[1] * Zpkmn[82] +
              S_ikmn_canonical[1] * Zpkmn[84] + S_ikmn_canonical[20] * Zpkmn[104] + S_ikmn_canonical[21] * Zpkmn[105] +
              S_ikmn_canonical[22] * Zpkmn[106] + S_ikmn_canonical[23] * Zpkmn[107] + S_ikmn_canonical[2] * Zpkmn[83] +
              S_ikmn_canonical[2] * Zpkmn[87] + S_ikmn_canonical[3] * Zpkmn[85] + S_ikmn_canonical[4] * Zpkmn[86] +
              S_ikmn_canonical[4] * Zpkmn[88] + S_ikmn_canonical[5] * Zpkmn[89] + S_ikmn_canonical[6] * Zpkmn[90] +
              S_ikmn_canonical[7] * Zpkmn[91] + S_ikmn_canonical[8] * Zpkmn[92] + S_ikmn_canonical[9] * Zpkmn[93];
    outx[4] = S_ikmn_canonical[0] * Zpkmn[108] + S_ikmn_canonical[10] * Zpkmn[121] + S_ikmn_canonical[11] * Zpkmn[122] +
              S_ikmn_canonical[12] * Zpkmn[123] + S_ikmn_canonical[13] * Zpkmn[124] + S_ikmn_canonical[14] * Zpkmn[125] +
              S_ikmn_canonical[15] * Zpkmn[126] + S_ikmn_canonical[16] * Zpkmn[127] + S_ikmn_canonical[17] * Zpkmn[128] +
              S_ikmn_canonical[18] * Zpkmn[129] + S_ikmn_canonical[19] * Zpkmn[130] + S_ikmn_canonical[1] * Zpkmn[109] +
              S_ikmn_canonical[1] * Zpkmn[111] + S_ikmn_canonical[20] * Zpkmn[131] + S_ikmn_canonical[21] * Zpkmn[132] +
              S_ikmn_canonical[22] * Zpkmn[133] + S_ikmn_canonical[23] * Zpkmn[134] + S_ikmn_canonical[2] * Zpkmn[110] +
              S_ikmn_canonical[2] * Zpkmn[114] + S_ikmn_canonical[3] * Zpkmn[112] + S_ikmn_canonical[4] * Zpkmn[113] +
              S_ikmn_canonical[4] * Zpkmn[115] + S_ikmn_canonical[5] * Zpkmn[116] + S_ikmn_canonical[6] * Zpkmn[117] +
              S_ikmn_canonical[7] * Zpkmn[118] + S_ikmn_canonical[8] * Zpkmn[119] + S_ikmn_canonical[9] * Zpkmn[120];
    outx[5] = S_ikmn_canonical[0] * Zpkmn[135] + S_ikmn_canonical[10] * Zpkmn[148] + S_ikmn_canonical[11] * Zpkmn[149] +
              S_ikmn_canonical[12] * Zpkmn[150] + S_ikmn_canonical[13] * Zpkmn[151] + S_ikmn_canonical[14] * Zpkmn[152] +
              S_ikmn_canonical[15] * Zpkmn[153] + S_ikmn_canonical[16] * Zpkmn[154] + S_ikmn_canonical[17] * Zpkmn[155] +
              S_ikmn_canonical[18] * Zpkmn[156] + S_ikmn_canonical[19] * Zpkmn[157] + S_ikmn_canonical[1] * Zpkmn[136] +
              S_ikmn_canonical[1] * Zpkmn[138] + S_ikmn_canonical[20] * Zpkmn[158] + S_ikmn_canonical[21] * Zpkmn[159] +
              S_ikmn_canonical[22] * Zpkmn[160] + S_ikmn_canonical[23] * Zpkmn[161] + S_ikmn_canonical[2] * Zpkmn[137] +
              S_ikmn_canonical[2] * Zpkmn[141] + S_ikmn_canonical[3] * Zpkmn[139] + S_ikmn_canonical[4] * Zpkmn[140] +
              S_ikmn_canonical[4] * Zpkmn[142] + S_ikmn_canonical[5] * Zpkmn[143] + S_ikmn_canonical[6] * Zpkmn[144] +
              S_ikmn_canonical[7] * Zpkmn[145] + S_ikmn_canonical[8] * Zpkmn[146] + S_ikmn_canonical[9] * Zpkmn[147];
    outx[6] = S_ikmn_canonical[0] * Zpkmn[162] + S_ikmn_canonical[10] * Zpkmn[175] + S_ikmn_canonical[11] * Zpkmn[176] +
              S_ikmn_canonical[12] * Zpkmn[177] + S_ikmn_canonical[13] * Zpkmn[178] + S_ikmn_canonical[14] * Zpkmn[179] +
              S_ikmn_canonical[15] * Zpkmn[180] + S_ikmn_canonical[16] * Zpkmn[181] + S_ikmn_canonical[17] * Zpkmn[182] +
              S_ikmn_canonical[18] * Zpkmn[183] + S_ikmn_canonical[19] * Zpkmn[184] + S_ikmn_canonical[1] * Zpkmn[163] +
              S_ikmn_canonical[1] * Zpkmn[165] + S_ikmn_canonical[20] * Zpkmn[185] + S_ikmn_canonical[21] * Zpkmn[186] +
              S_ikmn_canonical[22] * Zpkmn[187] + S_ikmn_canonical[23] * Zpkmn[188] + S_ikmn_canonical[2] * Zpkmn[164] +
              S_ikmn_canonical[2] * Zpkmn[168] + S_ikmn_canonical[3] * Zpkmn[166] + S_ikmn_canonical[4] * Zpkmn[167] +
              S_ikmn_canonical[4] * Zpkmn[169] + S_ikmn_canonical[5] * Zpkmn[170] + S_ikmn_canonical[6] * Zpkmn[171] +
              S_ikmn_canonical[7] * Zpkmn[172] + S_ikmn_canonical[8] * Zpkmn[173] + S_ikmn_canonical[9] * Zpkmn[174];
    outx[7] = S_ikmn_canonical[0] * Zpkmn[189] + S_ikmn_canonical[10] * Zpkmn[202] + S_ikmn_canonical[11] * Zpkmn[203] +
              S_ikmn_canonical[12] * Zpkmn[204] + S_ikmn_canonical[13] * Zpkmn[205] + S_ikmn_canonical[14] * Zpkmn[206] +
              S_ikmn_canonical[15] * Zpkmn[207] + S_ikmn_canonical[16] * Zpkmn[208] + S_ikmn_canonical[17] * Zpkmn[209] +
              S_ikmn_canonical[18] * Zpkmn[210] + S_ikmn_canonical[19] * Zpkmn[211] + S_ikmn_canonical[1] * Zpkmn[190] +
              S_ikmn_canonical[1] * Zpkmn[192] + S_ikmn_canonical[20] * Zpkmn[212] + S_ikmn_canonical[21] * Zpkmn[213] +
              S_ikmn_canonical[22] * Zpkmn[214] + S_ikmn_canonical[23] * Zpkmn[215] + S_ikmn_canonical[2] * Zpkmn[191] +
              S_ikmn_canonical[2] * Zpkmn[195] + S_ikmn_canonical[3] * Zpkmn[193] + S_ikmn_canonical[4] * Zpkmn[194] +
              S_ikmn_canonical[4] * Zpkmn[196] + S_ikmn_canonical[5] * Zpkmn[197] + S_ikmn_canonical[6] * Zpkmn[198] +
              S_ikmn_canonical[7] * Zpkmn[199] + S_ikmn_canonical[8] * Zpkmn[200] + S_ikmn_canonical[9] * Zpkmn[201];
    outy[0] = S_ikmn_canonical[10] * Zpkmn[4] + S_ikmn_canonical[11] * Zpkmn[7] + S_ikmn_canonical[12] * Zpkmn[2] +
              S_ikmn_canonical[13] * Zpkmn[5] + S_ikmn_canonical[14] * Zpkmn[8] + S_ikmn_canonical[24] * Zpkmn[9] +
              S_ikmn_canonical[25] * Zpkmn[10] + S_ikmn_canonical[25] * Zpkmn[12] + S_ikmn_canonical[26] * Zpkmn[11] +
              S_ikmn_canonical[26] * Zpkmn[15] + S_ikmn_canonical[27] * Zpkmn[13] + S_ikmn_canonical[28] * Zpkmn[14] +
              S_ikmn_canonical[28] * Zpkmn[16] + S_ikmn_canonical[29] * Zpkmn[17] + S_ikmn_canonical[30] * Zpkmn[18] +
              S_ikmn_canonical[31] * Zpkmn[19] + S_ikmn_canonical[32] * Zpkmn[20] + S_ikmn_canonical[33] * Zpkmn[21] +
              S_ikmn_canonical[34] * Zpkmn[22] + S_ikmn_canonical[35] * Zpkmn[23] + S_ikmn_canonical[36] * Zpkmn[24] +
              S_ikmn_canonical[37] * Zpkmn[25] + S_ikmn_canonical[38] * Zpkmn[26] + S_ikmn_canonical[6] * Zpkmn[0] +
              S_ikmn_canonical[7] * Zpkmn[3] + S_ikmn_canonical[8] * Zpkmn[6] + S_ikmn_canonical[9] * Zpkmn[1];
    outy[1] = S_ikmn_canonical[10] * Zpkmn[31] + S_ikmn_canonical[11] * Zpkmn[34] + S_ikmn_canonical[12] * Zpkmn[29] +
              S_ikmn_canonical[13] * Zpkmn[32] + S_ikmn_canonical[14] * Zpkmn[35] + S_ikmn_canonical[24] * Zpkmn[36] +
              S_ikmn_canonical[25] * Zpkmn[37] + S_ikmn_canonical[25] * Zpkmn[39] + S_ikmn_canonical[26] * Zpkmn[38] +
              S_ikmn_canonical[26] * Zpkmn[42] + S_ikmn_canonical[27] * Zpkmn[40] + S_ikmn_canonical[28] * Zpkmn[41] +
              S_ikmn_canonical[28] * Zpkmn[43] + S_ikmn_canonical[29] * Zpkmn[44] + S_ikmn_canonical[30] * Zpkmn[45] +
              S_ikmn_canonical[31] * Zpkmn[46] + S_ikmn_canonical[32] * Zpkmn[47] + S_ikmn_canonical[33] * Zpkmn[48] +
              S_ikmn_canonical[34] * Zpkmn[49] + S_ikmn_canonical[35] * Zpkmn[50] + S_ikmn_canonical[36] * Zpkmn[51] +
              S_ikmn_canonical[37] * Zpkmn[52] + S_ikmn_canonical[38] * Zpkmn[53] + S_ikmn_canonical[6] * Zpkmn[27] +
              S_ikmn_canonical[7] * Zpkmn[30] + S_ikmn_canonical[8] * Zpkmn[33] + S_ikmn_canonical[9] * Zpkmn[28];
    outy[2] = S_ikmn_canonical[10] * Zpkmn[58] + S_ikmn_canonical[11] * Zpkmn[61] + S_ikmn_canonical[12] * Zpkmn[56] +
              S_ikmn_canonical[13] * Zpkmn[59] + S_ikmn_canonical[14] * Zpkmn[62] + S_ikmn_canonical[24] * Zpkmn[63] +
              S_ikmn_canonical[25] * Zpkmn[64] + S_ikmn_canonical[25] * Zpkmn[66] + S_ikmn_canonical[26] * Zpkmn[65] +
              S_ikmn_canonical[26] * Zpkmn[69] + S_ikmn_canonical[27] * Zpkmn[67] + S_ikmn_canonical[28] * Zpkmn[68] +
              S_ikmn_canonical[28] * Zpkmn[70] + S_ikmn_canonical[29] * Zpkmn[71] + S_ikmn_canonical[30] * Zpkmn[72] +
              S_ikmn_canonical[31] * Zpkmn[73] + S_ikmn_canonical[32] * Zpkmn[74] + S_ikmn_canonical[33] * Zpkmn[75] +
              S_ikmn_canonical[34] * Zpkmn[76] + S_ikmn_canonical[35] * Zpkmn[77] + S_ikmn_canonical[36] * Zpkmn[78] +
              S_ikmn_canonical[37] * Zpkmn[79] + S_ikmn_canonical[38] * Zpkmn[80] + S_ikmn_canonical[6] * Zpkmn[54] +
              S_ikmn_canonical[7] * Zpkmn[57] + S_ikmn_canonical[8] * Zpkmn[60] + S_ikmn_canonical[9] * Zpkmn[55];
    outy[3] = S_ikmn_canonical[10] * Zpkmn[85] + S_ikmn_canonical[11] * Zpkmn[88] + S_ikmn_canonical[12] * Zpkmn[83] +
              S_ikmn_canonical[13] * Zpkmn[86] + S_ikmn_canonical[14] * Zpkmn[89] + S_ikmn_canonical[24] * Zpkmn[90] +
              S_ikmn_canonical[25] * Zpkmn[91] + S_ikmn_canonical[25] * Zpkmn[93] + S_ikmn_canonical[26] * Zpkmn[92] +
              S_ikmn_canonical[26] * Zpkmn[96] + S_ikmn_canonical[27] * Zpkmn[94] + S_ikmn_canonical[28] * Zpkmn[95] +
              S_ikmn_canonical[28] * Zpkmn[97] + S_ikmn_canonical[29] * Zpkmn[98] + S_ikmn_canonical[30] * Zpkmn[99] +
              S_ikmn_canonical[31] * Zpkmn[100] + S_ikmn_canonical[32] * Zpkmn[101] + S_ikmn_canonical[33] * Zpkmn[102] +
              S_ikmn_canonical[34] * Zpkmn[103] + S_ikmn_canonical[35] * Zpkmn[104] + S_ikmn_canonical[36] * Zpkmn[105] +
              S_ikmn_canonical[37] * Zpkmn[106] + S_ikmn_canonical[38] * Zpkmn[107] + S_ikmn_canonical[6] * Zpkmn[81] +
              S_ikmn_canonical[7] * Zpkmn[84] + S_ikmn_canonical[8] * Zpkmn[87] + S_ikmn_canonical[9] * Zpkmn[82];
    outy[4] = S_ikmn_canonical[10] * Zpkmn[112] + S_ikmn_canonical[11] * Zpkmn[115] + S_ikmn_canonical[12] * Zpkmn[110] +
              S_ikmn_canonical[13] * Zpkmn[113] + S_ikmn_canonical[14] * Zpkmn[116] + S_ikmn_canonical[24] * Zpkmn[117] +
              S_ikmn_canonical[25] * Zpkmn[118] + S_ikmn_canonical[25] * Zpkmn[120] + S_ikmn_canonical[26] * Zpkmn[119] +
              S_ikmn_canonical[26] * Zpkmn[123] + S_ikmn_canonical[27] * Zpkmn[121] + S_ikmn_canonical[28] * Zpkmn[122] +
              S_ikmn_canonical[28] * Zpkmn[124] + S_ikmn_canonical[29] * Zpkmn[125] + S_ikmn_canonical[30] * Zpkmn[126] +
              S_ikmn_canonical[31] * Zpkmn[127] + S_ikmn_canonical[32] * Zpkmn[128] + S_ikmn_canonical[33] * Zpkmn[129] +
              S_ikmn_canonical[34] * Zpkmn[130] + S_ikmn_canonical[35] * Zpkmn[131] + S_ikmn_canonical[36] * Zpkmn[132] +
              S_ikmn_canonical[37] * Zpkmn[133] + S_ikmn_canonical[38] * Zpkmn[134] + S_ikmn_canonical[6] * Zpkmn[108] +
              S_ikmn_canonical[7] * Zpkmn[111] + S_ikmn_canonical[8] * Zpkmn[114] + S_ikmn_canonical[9] * Zpkmn[109];
    outy[5] = S_ikmn_canonical[10] * Zpkmn[139] + S_ikmn_canonical[11] * Zpkmn[142] + S_ikmn_canonical[12] * Zpkmn[137] +
              S_ikmn_canonical[13] * Zpkmn[140] + S_ikmn_canonical[14] * Zpkmn[143] + S_ikmn_canonical[24] * Zpkmn[144] +
              S_ikmn_canonical[25] * Zpkmn[145] + S_ikmn_canonical[25] * Zpkmn[147] + S_ikmn_canonical[26] * Zpkmn[146] +
              S_ikmn_canonical[26] * Zpkmn[150] + S_ikmn_canonical[27] * Zpkmn[148] + S_ikmn_canonical[28] * Zpkmn[149] +
              S_ikmn_canonical[28] * Zpkmn[151] + S_ikmn_canonical[29] * Zpkmn[152] + S_ikmn_canonical[30] * Zpkmn[153] +
              S_ikmn_canonical[31] * Zpkmn[154] + S_ikmn_canonical[32] * Zpkmn[155] + S_ikmn_canonical[33] * Zpkmn[156] +
              S_ikmn_canonical[34] * Zpkmn[157] + S_ikmn_canonical[35] * Zpkmn[158] + S_ikmn_canonical[36] * Zpkmn[159] +
              S_ikmn_canonical[37] * Zpkmn[160] + S_ikmn_canonical[38] * Zpkmn[161] + S_ikmn_canonical[6] * Zpkmn[135] +
              S_ikmn_canonical[7] * Zpkmn[138] + S_ikmn_canonical[8] * Zpkmn[141] + S_ikmn_canonical[9] * Zpkmn[136];
    outy[6] = S_ikmn_canonical[10] * Zpkmn[166] + S_ikmn_canonical[11] * Zpkmn[169] + S_ikmn_canonical[12] * Zpkmn[164] +
              S_ikmn_canonical[13] * Zpkmn[167] + S_ikmn_canonical[14] * Zpkmn[170] + S_ikmn_canonical[24] * Zpkmn[171] +
              S_ikmn_canonical[25] * Zpkmn[172] + S_ikmn_canonical[25] * Zpkmn[174] + S_ikmn_canonical[26] * Zpkmn[173] +
              S_ikmn_canonical[26] * Zpkmn[177] + S_ikmn_canonical[27] * Zpkmn[175] + S_ikmn_canonical[28] * Zpkmn[176] +
              S_ikmn_canonical[28] * Zpkmn[178] + S_ikmn_canonical[29] * Zpkmn[179] + S_ikmn_canonical[30] * Zpkmn[180] +
              S_ikmn_canonical[31] * Zpkmn[181] + S_ikmn_canonical[32] * Zpkmn[182] + S_ikmn_canonical[33] * Zpkmn[183] +
              S_ikmn_canonical[34] * Zpkmn[184] + S_ikmn_canonical[35] * Zpkmn[185] + S_ikmn_canonical[36] * Zpkmn[186] +
              S_ikmn_canonical[37] * Zpkmn[187] + S_ikmn_canonical[38] * Zpkmn[188] + S_ikmn_canonical[6] * Zpkmn[162] +
              S_ikmn_canonical[7] * Zpkmn[165] + S_ikmn_canonical[8] * Zpkmn[168] + S_ikmn_canonical[9] * Zpkmn[163];
    outy[7] = S_ikmn_canonical[10] * Zpkmn[193] + S_ikmn_canonical[11] * Zpkmn[196] + S_ikmn_canonical[12] * Zpkmn[191] +
              S_ikmn_canonical[13] * Zpkmn[194] + S_ikmn_canonical[14] * Zpkmn[197] + S_ikmn_canonical[24] * Zpkmn[198] +
              S_ikmn_canonical[25] * Zpkmn[199] + S_ikmn_canonical[25] * Zpkmn[201] + S_ikmn_canonical[26] * Zpkmn[200] +
              S_ikmn_canonical[26] * Zpkmn[204] + S_ikmn_canonical[27] * Zpkmn[202] + S_ikmn_canonical[28] * Zpkmn[203] +
              S_ikmn_canonical[28] * Zpkmn[205] + S_ikmn_canonical[29] * Zpkmn[206] + S_ikmn_canonical[30] * Zpkmn[207] +
              S_ikmn_canonical[31] * Zpkmn[208] + S_ikmn_canonical[32] * Zpkmn[209] + S_ikmn_canonical[33] * Zpkmn[210] +
              S_ikmn_canonical[34] * Zpkmn[211] + S_ikmn_canonical[35] * Zpkmn[212] + S_ikmn_canonical[36] * Zpkmn[213] +
              S_ikmn_canonical[37] * Zpkmn[214] + S_ikmn_canonical[38] * Zpkmn[215] + S_ikmn_canonical[6] * Zpkmn[189] +
              S_ikmn_canonical[7] * Zpkmn[192] + S_ikmn_canonical[8] * Zpkmn[195] + S_ikmn_canonical[9] * Zpkmn[190];
    outz[0] = S_ikmn_canonical[15] * Zpkmn[0] + S_ikmn_canonical[16] * Zpkmn[3] + S_ikmn_canonical[17] * Zpkmn[6] +
              S_ikmn_canonical[18] * Zpkmn[1] + S_ikmn_canonical[19] * Zpkmn[4] + S_ikmn_canonical[20] * Zpkmn[7] +
              S_ikmn_canonical[21] * Zpkmn[2] + S_ikmn_canonical[22] * Zpkmn[5] + S_ikmn_canonical[23] * Zpkmn[8] +
              S_ikmn_canonical[30] * Zpkmn[9] + S_ikmn_canonical[31] * Zpkmn[12] + S_ikmn_canonical[32] * Zpkmn[15] +
              S_ikmn_canonical[33] * Zpkmn[10] + S_ikmn_canonical[34] * Zpkmn[13] + S_ikmn_canonical[35] * Zpkmn[16] +
              S_ikmn_canonical[36] * Zpkmn[11] + S_ikmn_canonical[37] * Zpkmn[14] + S_ikmn_canonical[38] * Zpkmn[17] +
              S_ikmn_canonical[39] * Zpkmn[18] + S_ikmn_canonical[40] * Zpkmn[19] + S_ikmn_canonical[40] * Zpkmn[21] +
              S_ikmn_canonical[41] * Zpkmn[20] + S_ikmn_canonical[41] * Zpkmn[24] + S_ikmn_canonical[42] * Zpkmn[22] +
              S_ikmn_canonical[43] * Zpkmn[23] + S_ikmn_canonical[43] * Zpkmn[25] + S_ikmn_canonical[44] * Zpkmn[26];
    outz[1] = S_ikmn_canonical[15] * Zpkmn[27] + S_ikmn_canonical[16] * Zpkmn[30] + S_ikmn_canonical[17] * Zpkmn[33] +
              S_ikmn_canonical[18] * Zpkmn[28] + S_ikmn_canonical[19] * Zpkmn[31] + S_ikmn_canonical[20] * Zpkmn[34] +
              S_ikmn_canonical[21] * Zpkmn[29] + S_ikmn_canonical[22] * Zpkmn[32] + S_ikmn_canonical[23] * Zpkmn[35] +
              S_ikmn_canonical[30] * Zpkmn[36] + S_ikmn_canonical[31] * Zpkmn[39] + S_ikmn_canonical[32] * Zpkmn[42] +
              S_ikmn_canonical[33] * Zpkmn[37] + S_ikmn_canonical[34] * Zpkmn[40] + S_ikmn_canonical[35] * Zpkmn[43] +
              S_ikmn_canonical[36] * Zpkmn[38] + S_ikmn_canonical[37] * Zpkmn[41] + S_ikmn_canonical[38] * Zpkmn[44] +
              S_ikmn_canonical[39] * Zpkmn[45] + S_ikmn_canonical[40] * Zpkmn[46] + S_ikmn_canonical[40] * Zpkmn[48] +
              S_ikmn_canonical[41] * Zpkmn[47] + S_ikmn_canonical[41] * Zpkmn[51] + S_ikmn_canonical[42] * Zpkmn[49] +
              S_ikmn_canonical[43] * Zpkmn[50] + S_ikmn_canonical[43] * Zpkmn[52] + S_ikmn_canonical[44] * Zpkmn[53];
    outz[2] = S_ikmn_canonical[15] * Zpkmn[54] + S_ikmn_canonical[16] * Zpkmn[57] + S_ikmn_canonical[17] * Zpkmn[60] +
              S_ikmn_canonical[18] * Zpkmn[55] + S_ikmn_canonical[19] * Zpkmn[58] + S_ikmn_canonical[20] * Zpkmn[61] +
              S_ikmn_canonical[21] * Zpkmn[56] + S_ikmn_canonical[22] * Zpkmn[59] + S_ikmn_canonical[23] * Zpkmn[62] +
              S_ikmn_canonical[30] * Zpkmn[63] + S_ikmn_canonical[31] * Zpkmn[66] + S_ikmn_canonical[32] * Zpkmn[69] +
              S_ikmn_canonical[33] * Zpkmn[64] + S_ikmn_canonical[34] * Zpkmn[67] + S_ikmn_canonical[35] * Zpkmn[70] +
              S_ikmn_canonical[36] * Zpkmn[65] + S_ikmn_canonical[37] * Zpkmn[68] + S_ikmn_canonical[38] * Zpkmn[71] +
              S_ikmn_canonical[39] * Zpkmn[72] + S_ikmn_canonical[40] * Zpkmn[73] + S_ikmn_canonical[40] * Zpkmn[75] +
              S_ikmn_canonical[41] * Zpkmn[74] + S_ikmn_canonical[41] * Zpkmn[78] + S_ikmn_canonical[42] * Zpkmn[76] +
              S_ikmn_canonical[43] * Zpkmn[77] + S_ikmn_canonical[43] * Zpkmn[79] + S_ikmn_canonical[44] * Zpkmn[80];
    outz[3] = S_ikmn_canonical[15] * Zpkmn[81] + S_ikmn_canonical[16] * Zpkmn[84] + S_ikmn_canonical[17] * Zpkmn[87] +
              S_ikmn_canonical[18] * Zpkmn[82] + S_ikmn_canonical[19] * Zpkmn[85] + S_ikmn_canonical[20] * Zpkmn[88] +
              S_ikmn_canonical[21] * Zpkmn[83] + S_ikmn_canonical[22] * Zpkmn[86] + S_ikmn_canonical[23] * Zpkmn[89] +
              S_ikmn_canonical[30] * Zpkmn[90] + S_ikmn_canonical[31] * Zpkmn[93] + S_ikmn_canonical[32] * Zpkmn[96] +
              S_ikmn_canonical[33] * Zpkmn[91] + S_ikmn_canonical[34] * Zpkmn[94] + S_ikmn_canonical[35] * Zpkmn[97] +
              S_ikmn_canonical[36] * Zpkmn[92] + S_ikmn_canonical[37] * Zpkmn[95] + S_ikmn_canonical[38] * Zpkmn[98] +
              S_ikmn_canonical[39] * Zpkmn[99] + S_ikmn_canonical[40] * Zpkmn[100] + S_ikmn_canonical[40] * Zpkmn[102] +
              S_ikmn_canonical[41] * Zpkmn[101] + S_ikmn_canonical[41] * Zpkmn[105] + S_ikmn_canonical[42] * Zpkmn[103] +
              S_ikmn_canonical[43] * Zpkmn[104] + S_ikmn_canonical[43] * Zpkmn[106] + S_ikmn_canonical[44] * Zpkmn[107];
    outz[4] = S_ikmn_canonical[15] * Zpkmn[108] + S_ikmn_canonical[16] * Zpkmn[111] + S_ikmn_canonical[17] * Zpkmn[114] +
              S_ikmn_canonical[18] * Zpkmn[109] + S_ikmn_canonical[19] * Zpkmn[112] + S_ikmn_canonical[20] * Zpkmn[115] +
              S_ikmn_canonical[21] * Zpkmn[110] + S_ikmn_canonical[22] * Zpkmn[113] + S_ikmn_canonical[23] * Zpkmn[116] +
              S_ikmn_canonical[30] * Zpkmn[117] + S_ikmn_canonical[31] * Zpkmn[120] + S_ikmn_canonical[32] * Zpkmn[123] +
              S_ikmn_canonical[33] * Zpkmn[118] + S_ikmn_canonical[34] * Zpkmn[121] + S_ikmn_canonical[35] * Zpkmn[124] +
              S_ikmn_canonical[36] * Zpkmn[119] + S_ikmn_canonical[37] * Zpkmn[122] + S_ikmn_canonical[38] * Zpkmn[125] +
              S_ikmn_canonical[39] * Zpkmn[126] + S_ikmn_canonical[40] * Zpkmn[127] + S_ikmn_canonical[40] * Zpkmn[129] +
              S_ikmn_canonical[41] * Zpkmn[128] + S_ikmn_canonical[41] * Zpkmn[132] + S_ikmn_canonical[42] * Zpkmn[130] +
              S_ikmn_canonical[43] * Zpkmn[131] + S_ikmn_canonical[43] * Zpkmn[133] + S_ikmn_canonical[44] * Zpkmn[134];
    outz[5] = S_ikmn_canonical[15] * Zpkmn[135] + S_ikmn_canonical[16] * Zpkmn[138] + S_ikmn_canonical[17] * Zpkmn[141] +
              S_ikmn_canonical[18] * Zpkmn[136] + S_ikmn_canonical[19] * Zpkmn[139] + S_ikmn_canonical[20] * Zpkmn[142] +
              S_ikmn_canonical[21] * Zpkmn[137] + S_ikmn_canonical[22] * Zpkmn[140] + S_ikmn_canonical[23] * Zpkmn[143] +
              S_ikmn_canonical[30] * Zpkmn[144] + S_ikmn_canonical[31] * Zpkmn[147] + S_ikmn_canonical[32] * Zpkmn[150] +
              S_ikmn_canonical[33] * Zpkmn[145] + S_ikmn_canonical[34] * Zpkmn[148] + S_ikmn_canonical[35] * Zpkmn[151] +
              S_ikmn_canonical[36] * Zpkmn[146] + S_ikmn_canonical[37] * Zpkmn[149] + S_ikmn_canonical[38] * Zpkmn[152] +
              S_ikmn_canonical[39] * Zpkmn[153] + S_ikmn_canonical[40] * Zpkmn[154] + S_ikmn_canonical[40] * Zpkmn[156] +
              S_ikmn_canonical[41] * Zpkmn[155] + S_ikmn_canonical[41] * Zpkmn[159] + S_ikmn_canonical[42] * Zpkmn[157] +
              S_ikmn_canonical[43] * Zpkmn[158] + S_ikmn_canonical[43] * Zpkmn[160] + S_ikmn_canonical[44] * Zpkmn[161];
    outz[6] = S_ikmn_canonical[15] * Zpkmn[162] + S_ikmn_canonical[16] * Zpkmn[165] + S_ikmn_canonical[17] * Zpkmn[168] +
              S_ikmn_canonical[18] * Zpkmn[163] + S_ikmn_canonical[19] * Zpkmn[166] + S_ikmn_canonical[20] * Zpkmn[169] +
              S_ikmn_canonical[21] * Zpkmn[164] + S_ikmn_canonical[22] * Zpkmn[167] + S_ikmn_canonical[23] * Zpkmn[170] +
              S_ikmn_canonical[30] * Zpkmn[171] + S_ikmn_canonical[31] * Zpkmn[174] + S_ikmn_canonical[32] * Zpkmn[177] +
              S_ikmn_canonical[33] * Zpkmn[172] + S_ikmn_canonical[34] * Zpkmn[175] + S_ikmn_canonical[35] * Zpkmn[178] +
              S_ikmn_canonical[36] * Zpkmn[173] + S_ikmn_canonical[37] * Zpkmn[176] + S_ikmn_canonical[38] * Zpkmn[179] +
              S_ikmn_canonical[39] * Zpkmn[180] + S_ikmn_canonical[40] * Zpkmn[181] + S_ikmn_canonical[40] * Zpkmn[183] +
              S_ikmn_canonical[41] * Zpkmn[182] + S_ikmn_canonical[41] * Zpkmn[186] + S_ikmn_canonical[42] * Zpkmn[184] +
              S_ikmn_canonical[43] * Zpkmn[185] + S_ikmn_canonical[43] * Zpkmn[187] + S_ikmn_canonical[44] * Zpkmn[188];
    outz[7] = S_ikmn_canonical[15] * Zpkmn[189] + S_ikmn_canonical[16] * Zpkmn[192] + S_ikmn_canonical[17] * Zpkmn[195] +
              S_ikmn_canonical[18] * Zpkmn[190] + S_ikmn_canonical[19] * Zpkmn[193] + S_ikmn_canonical[20] * Zpkmn[196] +
              S_ikmn_canonical[21] * Zpkmn[191] + S_ikmn_canonical[22] * Zpkmn[194] + S_ikmn_canonical[23] * Zpkmn[197] +
              S_ikmn_canonical[30] * Zpkmn[198] + S_ikmn_canonical[31] * Zpkmn[201] + S_ikmn_canonical[32] * Zpkmn[204] +
              S_ikmn_canonical[33] * Zpkmn[199] + S_ikmn_canonical[34] * Zpkmn[202] + S_ikmn_canonical[35] * Zpkmn[205] +
              S_ikmn_canonical[36] * Zpkmn[200] + S_ikmn_canonical[37] * Zpkmn[203] + S_ikmn_canonical[38] * Zpkmn[206] +
              S_ikmn_canonical[39] * Zpkmn[207] + S_ikmn_canonical[40] * Zpkmn[208] + S_ikmn_canonical[40] * Zpkmn[210] +
              S_ikmn_canonical[41] * Zpkmn[209] + S_ikmn_canonical[41] * Zpkmn[213] + S_ikmn_canonical[42] * Zpkmn[211] +
              S_ikmn_canonical[43] * Zpkmn[212] + S_ikmn_canonical[43] * Zpkmn[214] + S_ikmn_canonical[44] * Zpkmn[215];
}

static SFEM_INLINE void hex8_expand_S(const metric_tensor_t *const SFEM_RESTRICT S_ikmn_canonical,
                                      scalar_t *const SFEM_RESTRICT              S_ikmn) {
    // mundane ops: 0 divs: 0 sqrts: 0
    // total ops: 0
    S_ikmn[0]  = S_ikmn_canonical[0];
    S_ikmn[1]  = S_ikmn_canonical[1];
    S_ikmn[2]  = S_ikmn_canonical[2];
    S_ikmn[3]  = S_ikmn_canonical[1];
    S_ikmn[4]  = S_ikmn_canonical[3];
    S_ikmn[5]  = S_ikmn_canonical[4];
    S_ikmn[6]  = S_ikmn_canonical[2];
    S_ikmn[7]  = S_ikmn_canonical[4];
    S_ikmn[8]  = S_ikmn_canonical[5];
    S_ikmn[9]  = S_ikmn_canonical[6];
    S_ikmn[10] = S_ikmn_canonical[7];
    S_ikmn[11] = S_ikmn_canonical[8];
    S_ikmn[12] = S_ikmn_canonical[9];
    S_ikmn[13] = S_ikmn_canonical[10];
    S_ikmn[14] = S_ikmn_canonical[11];
    S_ikmn[15] = S_ikmn_canonical[12];
    S_ikmn[16] = S_ikmn_canonical[13];
    S_ikmn[17] = S_ikmn_canonical[14];
    S_ikmn[18] = S_ikmn_canonical[15];
    S_ikmn[19] = S_ikmn_canonical[16];
    S_ikmn[20] = S_ikmn_canonical[17];
    S_ikmn[21] = S_ikmn_canonical[18];
    S_ikmn[22] = S_ikmn_canonical[19];
    S_ikmn[23] = S_ikmn_canonical[20];
    S_ikmn[24] = S_ikmn_canonical[21];
    S_ikmn[25] = S_ikmn_canonical[22];
    S_ikmn[26] = S_ikmn_canonical[23];
    S_ikmn[27] = S_ikmn_canonical[6];
    S_ikmn[28] = S_ikmn_canonical[9];
    S_ikmn[29] = S_ikmn_canonical[12];
    S_ikmn[30] = S_ikmn_canonical[7];
    S_ikmn[31] = S_ikmn_canonical[10];
    S_ikmn[32] = S_ikmn_canonical[13];
    S_ikmn[33] = S_ikmn_canonical[8];
    S_ikmn[34] = S_ikmn_canonical[11];
    S_ikmn[35] = S_ikmn_canonical[14];
    S_ikmn[36] = S_ikmn_canonical[24];
    S_ikmn[37] = S_ikmn_canonical[25];
    S_ikmn[38] = S_ikmn_canonical[26];
    S_ikmn[39] = S_ikmn_canonical[25];
    S_ikmn[40] = S_ikmn_canonical[27];
    S_ikmn[41] = S_ikmn_canonical[28];
    S_ikmn[42] = S_ikmn_canonical[26];
    S_ikmn[43] = S_ikmn_canonical[28];
    S_ikmn[44] = S_ikmn_canonical[29];
    S_ikmn[45] = S_ikmn_canonical[30];
    S_ikmn[46] = S_ikmn_canonical[31];
    S_ikmn[47] = S_ikmn_canonical[32];
    S_ikmn[48] = S_ikmn_canonical[33];
    S_ikmn[49] = S_ikmn_canonical[34];
    S_ikmn[50] = S_ikmn_canonical[35];
    S_ikmn[51] = S_ikmn_canonical[36];
    S_ikmn[52] = S_ikmn_canonical[37];
    S_ikmn[53] = S_ikmn_canonical[38];
    S_ikmn[54] = S_ikmn_canonical[15];
    S_ikmn[55] = S_ikmn_canonical[18];
    S_ikmn[56] = S_ikmn_canonical[21];
    S_ikmn[57] = S_ikmn_canonical[16];
    S_ikmn[58] = S_ikmn_canonical[19];
    S_ikmn[59] = S_ikmn_canonical[22];
    S_ikmn[60] = S_ikmn_canonical[17];
    S_ikmn[61] = S_ikmn_canonical[20];
    S_ikmn[62] = S_ikmn_canonical[23];
    S_ikmn[63] = S_ikmn_canonical[30];
    S_ikmn[64] = S_ikmn_canonical[33];
    S_ikmn[65] = S_ikmn_canonical[36];
    S_ikmn[66] = S_ikmn_canonical[31];
    S_ikmn[67] = S_ikmn_canonical[34];
    S_ikmn[68] = S_ikmn_canonical[37];
    S_ikmn[69] = S_ikmn_canonical[32];
    S_ikmn[70] = S_ikmn_canonical[35];
    S_ikmn[71] = S_ikmn_canonical[38];
    S_ikmn[72] = S_ikmn_canonical[39];
    S_ikmn[73] = S_ikmn_canonical[40];
    S_ikmn[74] = S_ikmn_canonical[41];
    S_ikmn[75] = S_ikmn_canonical[40];
    S_ikmn[76] = S_ikmn_canonical[42];
    S_ikmn[77] = S_ikmn_canonical[43];
    S_ikmn[78] = S_ikmn_canonical[41];
    S_ikmn[79] = S_ikmn_canonical[43];
    S_ikmn[80] = S_ikmn_canonical[44];
}

static SFEM_INLINE void hex8_SdotZ_expanded(const scalar_t *const SFEM_RESTRICT S_ikmn,
                                            const scalar_t *const SFEM_RESTRICT Zpkmn,
                                            scalar_t *const SFEM_RESTRICT       outx,
                                            scalar_t *const SFEM_RESTRICT       outy,
                                            scalar_t *const SFEM_RESTRICT       outz)

{
    static const int pstride = 3 * 3 * 3;
    static const int ksize   = 3 * 3 * 3;

    for (int p = 0; p < 8; p++) {
        scalar_t                            acc[3] = {0};
        const scalar_t *const SFEM_RESTRICT Zkmn   = &Zpkmn[p * pstride];
        for (int i = 0; i < 3; i++) {
            const scalar_t *const SFEM_RESTRICT Skmn = &S_ikmn[i * ksize];
            for (int k = 0; k < ksize; k++) {
                acc[i] += Skmn[k] * Zkmn[k];
            }
        }

        outx[p] = acc[0];
        outy[p] = acc[1];
        outz[p] = acc[2];
    }
}

static SFEM_INLINE void hex8_neohookean_hessian_from_S_ikmn(const metric_tensor_t *const SFEM_RESTRICT S_ikmn_canonical,
                                                            const scalar_t *const SFEM_RESTRICT        Wimpn_compressed,
                                                            scalar_t *const SFEM_RESTRICT              H) {
    // mundane ops: 924 divs: 0 sqrts: 0
    // total ops: 924
    const scalar_t x0  = S_ikmn_canonical[1] * Wimpn_compressed[1];
    const scalar_t x1  = 2 * x0;
    const scalar_t x2  = S_ikmn_canonical[2] * Wimpn_compressed[1];
    const scalar_t x3  = 2 * x2;
    const scalar_t x4  = S_ikmn_canonical[0] * Wimpn_compressed[0];
    const scalar_t x5  = S_ikmn_canonical[3] * Wimpn_compressed[0];
    const scalar_t x6  = S_ikmn_canonical[5] * Wimpn_compressed[0];
    const scalar_t x7  = S_ikmn_canonical[4] * Wimpn_compressed[1];
    const scalar_t x8  = 2 * x7;
    const scalar_t x9  = x4 + x5 + x6 + x8;
    const scalar_t x10 = x1 + x3 + x9;
    const scalar_t x11 = S_ikmn_canonical[4] * Wimpn_compressed[5];
    const scalar_t x12 = 2 * x11;
    const scalar_t x13 = S_ikmn_canonical[1] * Wimpn_compressed[4];
    const scalar_t x14 = S_ikmn_canonical[5] * Wimpn_compressed[6] + x0 + x13;
    const scalar_t x15 = S_ikmn_canonical[2] * Wimpn_compressed[4];
    const scalar_t x16 = S_ikmn_canonical[3] * Wimpn_compressed[6] + x15 + x2;
    const scalar_t x17 = S_ikmn_canonical[0] * Wimpn_compressed[2] + x14 + x16;
    const scalar_t x18 = x12 + x17;
    const scalar_t x19 = 2 * x13;
    const scalar_t x20 = S_ikmn_canonical[4] * Wimpn_compressed[8];
    const scalar_t x21 = S_ikmn_canonical[0] * Wimpn_compressed[3] + x11 + x20;
    const scalar_t x22 = S_ikmn_canonical[2] * Wimpn_compressed[5];
    const scalar_t x23 = S_ikmn_canonical[2] * Wimpn_compressed[8];
    const scalar_t x24 = S_ikmn_canonical[3] * Wimpn_compressed[3] + x22 + x23;
    const scalar_t x25 = S_ikmn_canonical[5] * Wimpn_compressed[9] + x21 + x24;
    const scalar_t x26 = x19 + x25;
    const scalar_t x27 = 2 * x22;
    const scalar_t x28 = S_ikmn_canonical[4] * Wimpn_compressed[4];
    const scalar_t x29 = S_ikmn_canonical[0] * Wimpn_compressed[6] + x28 + x7;
    const scalar_t x30 = S_ikmn_canonical[3] * Wimpn_compressed[2] + x14 + x29;
    const scalar_t x31 = x27 + x30;
    const scalar_t x32 = S_ikmn_canonical[1] * Wimpn_compressed[5];
    const scalar_t x33 = 2 * x32;
    const scalar_t x34 = S_ikmn_canonical[5] * Wimpn_compressed[2] + x16 + x29;
    const scalar_t x35 = x33 + x34;
    const scalar_t x36 = 2 * x15;
    const scalar_t x37 = S_ikmn_canonical[1] * Wimpn_compressed[8];
    const scalar_t x38 = S_ikmn_canonical[5] * Wimpn_compressed[3] + x32 + x37;
    const scalar_t x39 = S_ikmn_canonical[3] * Wimpn_compressed[9] + x21 + x38;
    const scalar_t x40 = x36 + x39;
    const scalar_t x41 = 2 * x37;
    const scalar_t x42 = 2 * x23;
    const scalar_t x43 = S_ikmn_canonical[0] * Wimpn_compressed[7];
    const scalar_t x44 = S_ikmn_canonical[3] * Wimpn_compressed[7];
    const scalar_t x45 = S_ikmn_canonical[5] * Wimpn_compressed[7];
    const scalar_t x46 = 2 * x20;
    const scalar_t x47 = x43 + x44 + x45 + x46;
    const scalar_t x48 = x41 + x42 + x47;
    const scalar_t x49 = 2 * x28;
    const scalar_t x50 = S_ikmn_canonical[0] * Wimpn_compressed[9] + x24 + x38;
    const scalar_t x51 = x49 + x50;
    const scalar_t x52 = S_ikmn_canonical[12] * Wimpn_compressed[1];
    const scalar_t x53 = S_ikmn_canonical[9] * Wimpn_compressed[1];
    const scalar_t x54 = x52 + x53;
    const scalar_t x55 = S_ikmn_canonical[13] * Wimpn_compressed[1];
    const scalar_t x56 = S_ikmn_canonical[7] * Wimpn_compressed[1];
    const scalar_t x57 = x55 + x56;
    const scalar_t x58 = S_ikmn_canonical[11] * Wimpn_compressed[1];
    const scalar_t x59 = S_ikmn_canonical[8] * Wimpn_compressed[1];
    const scalar_t x60 = x58 + x59;
    const scalar_t x61 = S_ikmn_canonical[10] * Wimpn_compressed[0] + S_ikmn_canonical[14] * Wimpn_compressed[0] +
                         S_ikmn_canonical[6] * Wimpn_compressed[0];
    const scalar_t x62  = x54 + x57 + x60 + x61;
    const scalar_t x63  = S_ikmn_canonical[13] * Wimpn_compressed[5];
    const scalar_t x64  = S_ikmn_canonical[7] * Wimpn_compressed[4];
    const scalar_t x65  = x63 + x64;
    const scalar_t x66  = S_ikmn_canonical[11] * Wimpn_compressed[5];
    const scalar_t x67  = S_ikmn_canonical[8] * Wimpn_compressed[4];
    const scalar_t x68  = x66 + x67;
    const scalar_t x69  = S_ikmn_canonical[10] * Wimpn_compressed[6];
    const scalar_t x70  = S_ikmn_canonical[14] * Wimpn_compressed[6];
    const scalar_t x71  = S_ikmn_canonical[6] * Wimpn_compressed[2] + x69 + x70;
    const scalar_t x72  = x54 + x65 + x68 + x71;
    const scalar_t x73  = S_ikmn_canonical[12] * Wimpn_compressed[5];
    const scalar_t x74  = S_ikmn_canonical[9] * Wimpn_compressed[4];
    const scalar_t x75  = x73 + x74;
    const scalar_t x76  = S_ikmn_canonical[11] * Wimpn_compressed[8];
    const scalar_t x77  = S_ikmn_canonical[8] * Wimpn_compressed[8];
    const scalar_t x78  = x76 + x77;
    const scalar_t x79  = S_ikmn_canonical[10] * Wimpn_compressed[3];
    const scalar_t x80  = S_ikmn_canonical[6] * Wimpn_compressed[3];
    const scalar_t x81  = S_ikmn_canonical[14] * Wimpn_compressed[9] + x79 + x80;
    const scalar_t x82  = x65 + x75 + x78 + x81;
    const scalar_t x83  = S_ikmn_canonical[11] * Wimpn_compressed[4];
    const scalar_t x84  = S_ikmn_canonical[8] * Wimpn_compressed[5];
    const scalar_t x85  = x83 + x84;
    const scalar_t x86  = S_ikmn_canonical[6] * Wimpn_compressed[6];
    const scalar_t x87  = S_ikmn_canonical[10] * Wimpn_compressed[2] + x70 + x86;
    const scalar_t x88  = x57 + x75 + x85 + x87;
    const scalar_t x89  = S_ikmn_canonical[12] * Wimpn_compressed[4];
    const scalar_t x90  = S_ikmn_canonical[9] * Wimpn_compressed[5];
    const scalar_t x91  = x89 + x90;
    const scalar_t x92  = S_ikmn_canonical[13] * Wimpn_compressed[4];
    const scalar_t x93  = S_ikmn_canonical[7] * Wimpn_compressed[5];
    const scalar_t x94  = x92 + x93;
    const scalar_t x95  = S_ikmn_canonical[14] * Wimpn_compressed[2] + x69 + x86;
    const scalar_t x96  = x60 + x91 + x94 + x95;
    const scalar_t x97  = S_ikmn_canonical[13] * Wimpn_compressed[8];
    const scalar_t x98  = S_ikmn_canonical[7] * Wimpn_compressed[8];
    const scalar_t x99  = x97 + x98;
    const scalar_t x100 = S_ikmn_canonical[14] * Wimpn_compressed[3];
    const scalar_t x101 = S_ikmn_canonical[10] * Wimpn_compressed[9] + x100 + x80;
    const scalar_t x102 = x101 + x68 + x91 + x99;
    const scalar_t x103 = S_ikmn_canonical[12] * Wimpn_compressed[8];
    const scalar_t x104 = S_ikmn_canonical[9] * Wimpn_compressed[8];
    const scalar_t x105 = x103 + x104;
    const scalar_t x106 = S_ikmn_canonical[10] * Wimpn_compressed[7] + S_ikmn_canonical[14] * Wimpn_compressed[7] +
                          S_ikmn_canonical[6] * Wimpn_compressed[7];
    const scalar_t x107 = x105 + x106 + x78 + x99;
    const scalar_t x108 = S_ikmn_canonical[6] * Wimpn_compressed[9] + x100 + x79;
    const scalar_t x109 = x105 + x108 + x85 + x94;
    const scalar_t x110 = S_ikmn_canonical[18] * Wimpn_compressed[1];
    const scalar_t x111 = S_ikmn_canonical[21] * Wimpn_compressed[1];
    const scalar_t x112 = x110 + x111;
    const scalar_t x113 = S_ikmn_canonical[16] * Wimpn_compressed[1];
    const scalar_t x114 = S_ikmn_canonical[22] * Wimpn_compressed[1];
    const scalar_t x115 = x113 + x114;
    const scalar_t x116 = S_ikmn_canonical[17] * Wimpn_compressed[1];
    const scalar_t x117 = S_ikmn_canonical[20] * Wimpn_compressed[1];
    const scalar_t x118 = x116 + x117;
    const scalar_t x119 = S_ikmn_canonical[15] * Wimpn_compressed[0] + S_ikmn_canonical[19] * Wimpn_compressed[0] +
                          S_ikmn_canonical[23] * Wimpn_compressed[0];
    const scalar_t x120 = x112 + x115 + x118 + x119;
    const scalar_t x121 = S_ikmn_canonical[16] * Wimpn_compressed[4];
    const scalar_t x122 = S_ikmn_canonical[22] * Wimpn_compressed[5];
    const scalar_t x123 = x121 + x122;
    const scalar_t x124 = S_ikmn_canonical[17] * Wimpn_compressed[4];
    const scalar_t x125 = S_ikmn_canonical[20] * Wimpn_compressed[5];
    const scalar_t x126 = x124 + x125;
    const scalar_t x127 = S_ikmn_canonical[19] * Wimpn_compressed[6];
    const scalar_t x128 = S_ikmn_canonical[23] * Wimpn_compressed[6];
    const scalar_t x129 = S_ikmn_canonical[15] * Wimpn_compressed[2] + x127 + x128;
    const scalar_t x130 = x112 + x123 + x126 + x129;
    const scalar_t x131 = S_ikmn_canonical[18] * Wimpn_compressed[4];
    const scalar_t x132 = S_ikmn_canonical[21] * Wimpn_compressed[5];
    const scalar_t x133 = x131 + x132;
    const scalar_t x134 = S_ikmn_canonical[17] * Wimpn_compressed[8];
    const scalar_t x135 = S_ikmn_canonical[20] * Wimpn_compressed[8];
    const scalar_t x136 = x134 + x135;
    const scalar_t x137 = S_ikmn_canonical[15] * Wimpn_compressed[3];
    const scalar_t x138 = S_ikmn_canonical[19] * Wimpn_compressed[3];
    const scalar_t x139 = S_ikmn_canonical[23] * Wimpn_compressed[9] + x137 + x138;
    const scalar_t x140 = x123 + x133 + x136 + x139;
    const scalar_t x141 = S_ikmn_canonical[17] * Wimpn_compressed[5];
    const scalar_t x142 = S_ikmn_canonical[20] * Wimpn_compressed[4];
    const scalar_t x143 = x141 + x142;
    const scalar_t x144 = S_ikmn_canonical[15] * Wimpn_compressed[6];
    const scalar_t x145 = S_ikmn_canonical[19] * Wimpn_compressed[2] + x128 + x144;
    const scalar_t x146 = x115 + x133 + x143 + x145;
    const scalar_t x147 = S_ikmn_canonical[18] * Wimpn_compressed[5];
    const scalar_t x148 = S_ikmn_canonical[21] * Wimpn_compressed[4];
    const scalar_t x149 = x147 + x148;
    const scalar_t x150 = S_ikmn_canonical[16] * Wimpn_compressed[5];
    const scalar_t x151 = S_ikmn_canonical[22] * Wimpn_compressed[4];
    const scalar_t x152 = x150 + x151;
    const scalar_t x153 = S_ikmn_canonical[23] * Wimpn_compressed[2] + x127 + x144;
    const scalar_t x154 = x118 + x149 + x152 + x153;
    const scalar_t x155 = S_ikmn_canonical[16] * Wimpn_compressed[8];
    const scalar_t x156 = S_ikmn_canonical[22] * Wimpn_compressed[8];
    const scalar_t x157 = x155 + x156;
    const scalar_t x158 = S_ikmn_canonical[23] * Wimpn_compressed[3];
    const scalar_t x159 = S_ikmn_canonical[19] * Wimpn_compressed[9] + x137 + x158;
    const scalar_t x160 = x126 + x149 + x157 + x159;
    const scalar_t x161 = S_ikmn_canonical[18] * Wimpn_compressed[8];
    const scalar_t x162 = S_ikmn_canonical[21] * Wimpn_compressed[8];
    const scalar_t x163 = x161 + x162;
    const scalar_t x164 = S_ikmn_canonical[15] * Wimpn_compressed[7] + S_ikmn_canonical[19] * Wimpn_compressed[7] +
                          S_ikmn_canonical[23] * Wimpn_compressed[7];
    const scalar_t x165 = x136 + x157 + x163 + x164;
    const scalar_t x166 = S_ikmn_canonical[15] * Wimpn_compressed[9] + x138 + x158;
    const scalar_t x167 = x143 + x152 + x163 + x166;
    const scalar_t x168 = x19 + x36 + x9;
    const scalar_t x169 = x30 + x42;
    const scalar_t x170 = x1 + x25;
    const scalar_t x171 = x3 + x39;
    const scalar_t x172 = x34 + x41;
    const scalar_t x173 = x27 + x33 + x47;
    const scalar_t x174 = x74 + x89;
    const scalar_t x175 = x56 + x63;
    const scalar_t x176 = x59 + x66;
    const scalar_t x177 = x174 + x175 + x176 + x71;
    const scalar_t x178 = x55 + x64;
    const scalar_t x179 = x58 + x67;
    const scalar_t x180 = x174 + x178 + x179 + x61;
    const scalar_t x181 = x103 + x53;
    const scalar_t x182 = x77 + x83;
    const scalar_t x183 = x178 + x181 + x182 + x87;
    const scalar_t x184 = x76 + x84;
    const scalar_t x185 = x175 + x181 + x184 + x81;
    const scalar_t x186 = x104 + x52;
    const scalar_t x187 = x93 + x97;
    const scalar_t x188 = x101 + x176 + x186 + x187;
    const scalar_t x189 = x92 + x98;
    const scalar_t x190 = x179 + x186 + x189 + x95;
    const scalar_t x191 = x73 + x90;
    const scalar_t x192 = x108 + x182 + x189 + x191;
    const scalar_t x193 = x106 + x184 + x187 + x191;
    const scalar_t x194 = x131 + x148;
    const scalar_t x195 = x113 + x122;
    const scalar_t x196 = x116 + x125;
    const scalar_t x197 = x129 + x194 + x195 + x196;
    const scalar_t x198 = x114 + x121;
    const scalar_t x199 = x117 + x124;
    const scalar_t x200 = x119 + x194 + x198 + x199;
    const scalar_t x201 = x110 + x162;
    const scalar_t x202 = x134 + x142;
    const scalar_t x203 = x145 + x198 + x201 + x202;
    const scalar_t x204 = x135 + x141;
    const scalar_t x205 = x139 + x195 + x201 + x204;
    const scalar_t x206 = x111 + x161;
    const scalar_t x207 = x150 + x156;
    const scalar_t x208 = x159 + x196 + x206 + x207;
    const scalar_t x209 = x151 + x155;
    const scalar_t x210 = x153 + x199 + x206 + x209;
    const scalar_t x211 = x132 + x147;
    const scalar_t x212 = x166 + x202 + x209 + x211;
    const scalar_t x213 = x164 + x204 + x207 + x211;
    const scalar_t x214 = x4 + x49 + x5 + x6;
    const scalar_t x215 = x1 + x214 + x36;
    const scalar_t x216 = x17 + x46;
    const scalar_t x217 = x12 + x43 + x44 + x45;
    const scalar_t x218 = x217 + x27 + x41;
    const scalar_t x219 = x50 + x8;
    const scalar_t x220 = x103 + x74;
    const scalar_t x221 = x64 + x97;
    const scalar_t x222 = x66 + x84;
    const scalar_t x223 = x220 + x221 + x222 + x81;
    const scalar_t x224 = x56 + x92;
    const scalar_t x225 = x58 + x77;
    const scalar_t x226 = x220 + x224 + x225 + x87;
    const scalar_t x227 = x53 + x89;
    const scalar_t x228 = x67 + x83;
    const scalar_t x229 = x224 + x227 + x228 + x61;
    const scalar_t x230 = x59 + x76;
    const scalar_t x231 = x221 + x227 + x230 + x71;
    const scalar_t x232 = x104 + x73;
    const scalar_t x233 = x63 + x98;
    const scalar_t x234 = x106 + x222 + x232 + x233;
    const scalar_t x235 = x55 + x93;
    const scalar_t x236 = x108 + x225 + x232 + x235;
    const scalar_t x237 = x52 + x90;
    const scalar_t x238 = x228 + x235 + x237 + x95;
    const scalar_t x239 = x101 + x230 + x233 + x237;
    const scalar_t x240 = x131 + x162;
    const scalar_t x241 = x121 + x156;
    const scalar_t x242 = x125 + x141;
    const scalar_t x243 = x139 + x240 + x241 + x242;
    const scalar_t x244 = x113 + x151;
    const scalar_t x245 = x117 + x134;
    const scalar_t x246 = x145 + x240 + x244 + x245;
    const scalar_t x247 = x110 + x148;
    const scalar_t x248 = x124 + x142;
    const scalar_t x249 = x119 + x244 + x247 + x248;
    const scalar_t x250 = x116 + x135;
    const scalar_t x251 = x129 + x241 + x247 + x250;
    const scalar_t x252 = x132 + x161;
    const scalar_t x253 = x122 + x155;
    const scalar_t x254 = x164 + x242 + x252 + x253;
    const scalar_t x255 = x114 + x150;
    const scalar_t x256 = x166 + x245 + x252 + x255;
    const scalar_t x257 = x111 + x147;
    const scalar_t x258 = x153 + x248 + x255 + x257;
    const scalar_t x259 = x159 + x250 + x253 + x257;
    const scalar_t x260 = x19 + x214 + x3;
    const scalar_t x261 = x217 + x33 + x42;
    const scalar_t x262 = x53 + x73;
    const scalar_t x263 = x64 + x92;
    const scalar_t x264 = x58 + x84;
    const scalar_t x265 = x262 + x263 + x264 + x87;
    const scalar_t x266 = x56 + x97;
    const scalar_t x267 = x66 + x77;
    const scalar_t x268 = x262 + x266 + x267 + x81;
    const scalar_t x269 = x52 + x74;
    const scalar_t x270 = x67 + x76;
    const scalar_t x271 = x266 + x269 + x270 + x71;
    const scalar_t x272 = x59 + x83;
    const scalar_t x273 = x263 + x269 + x272 + x61;
    const scalar_t x274 = x103 + x90;
    const scalar_t x275 = x55 + x98;
    const scalar_t x276 = x108 + x264 + x274 + x275;
    const scalar_t x277 = x63 + x93;
    const scalar_t x278 = x106 + x267 + x274 + x277;
    const scalar_t x279 = x104 + x89;
    const scalar_t x280 = x101 + x270 + x277 + x279;
    const scalar_t x281 = x272 + x275 + x279 + x95;
    const scalar_t x282 = x110 + x132;
    const scalar_t x283 = x121 + x151;
    const scalar_t x284 = x117 + x141;
    const scalar_t x285 = x145 + x282 + x283 + x284;
    const scalar_t x286 = x113 + x156;
    const scalar_t x287 = x125 + x134;
    const scalar_t x288 = x139 + x282 + x286 + x287;
    const scalar_t x289 = x111 + x131;
    const scalar_t x290 = x124 + x135;
    const scalar_t x291 = x129 + x286 + x289 + x290;
    const scalar_t x292 = x116 + x142;
    const scalar_t x293 = x119 + x283 + x289 + x292;
    const scalar_t x294 = x147 + x162;
    const scalar_t x295 = x114 + x155;
    const scalar_t x296 = x166 + x284 + x294 + x295;
    const scalar_t x297 = x122 + x150;
    const scalar_t x298 = x164 + x287 + x294 + x297;
    const scalar_t x299 = x148 + x161;
    const scalar_t x300 = x159 + x290 + x297 + x299;
    const scalar_t x301 = x153 + x292 + x295 + x299;
    const scalar_t x302 = S_ikmn_canonical[25] * Wimpn_compressed[1];
    const scalar_t x303 = 2 * x302;
    const scalar_t x304 = S_ikmn_canonical[26] * Wimpn_compressed[1];
    const scalar_t x305 = 2 * x304;
    const scalar_t x306 = S_ikmn_canonical[24] * Wimpn_compressed[0];
    const scalar_t x307 = S_ikmn_canonical[27] * Wimpn_compressed[0];
    const scalar_t x308 = S_ikmn_canonical[29] * Wimpn_compressed[0];
    const scalar_t x309 = S_ikmn_canonical[28] * Wimpn_compressed[1];
    const scalar_t x310 = 2 * x309;
    const scalar_t x311 = x306 + x307 + x308 + x310;
    const scalar_t x312 = x303 + x305 + x311;
    const scalar_t x313 = S_ikmn_canonical[28] * Wimpn_compressed[5];
    const scalar_t x314 = 2 * x313;
    const scalar_t x315 = S_ikmn_canonical[25] * Wimpn_compressed[4];
    const scalar_t x316 = S_ikmn_canonical[29] * Wimpn_compressed[6] + x302 + x315;
    const scalar_t x317 = S_ikmn_canonical[26] * Wimpn_compressed[4];
    const scalar_t x318 = S_ikmn_canonical[27] * Wimpn_compressed[6] + x304 + x317;
    const scalar_t x319 = S_ikmn_canonical[24] * Wimpn_compressed[2] + x316 + x318;
    const scalar_t x320 = x314 + x319;
    const scalar_t x321 = 2 * x315;
    const scalar_t x322 = S_ikmn_canonical[28] * Wimpn_compressed[8];
    const scalar_t x323 = S_ikmn_canonical[24] * Wimpn_compressed[3] + x313 + x322;
    const scalar_t x324 = S_ikmn_canonical[26] * Wimpn_compressed[5];
    const scalar_t x325 = S_ikmn_canonical[26] * Wimpn_compressed[8];
    const scalar_t x326 = S_ikmn_canonical[27] * Wimpn_compressed[3] + x324 + x325;
    const scalar_t x327 = S_ikmn_canonical[29] * Wimpn_compressed[9] + x323 + x326;
    const scalar_t x328 = x321 + x327;
    const scalar_t x329 = 2 * x324;
    const scalar_t x330 = S_ikmn_canonical[28] * Wimpn_compressed[4];
    const scalar_t x331 = S_ikmn_canonical[24] * Wimpn_compressed[6] + x309 + x330;
    const scalar_t x332 = S_ikmn_canonical[27] * Wimpn_compressed[2] + x316 + x331;
    const scalar_t x333 = x329 + x332;
    const scalar_t x334 = S_ikmn_canonical[25] * Wimpn_compressed[5];
    const scalar_t x335 = 2 * x334;
    const scalar_t x336 = S_ikmn_canonical[29] * Wimpn_compressed[2] + x318 + x331;
    const scalar_t x337 = x335 + x336;
    const scalar_t x338 = 2 * x317;
    const scalar_t x339 = S_ikmn_canonical[25] * Wimpn_compressed[8];
    const scalar_t x340 = S_ikmn_canonical[29] * Wimpn_compressed[3] + x334 + x339;
    const scalar_t x341 = S_ikmn_canonical[27] * Wimpn_compressed[9] + x323 + x340;
    const scalar_t x342 = x338 + x341;
    const scalar_t x343 = 2 * x339;
    const scalar_t x344 = 2 * x325;
    const scalar_t x345 = S_ikmn_canonical[24] * Wimpn_compressed[7];
    const scalar_t x346 = S_ikmn_canonical[27] * Wimpn_compressed[7];
    const scalar_t x347 = S_ikmn_canonical[29] * Wimpn_compressed[7];
    const scalar_t x348 = 2 * x322;
    const scalar_t x349 = x345 + x346 + x347 + x348;
    const scalar_t x350 = x343 + x344 + x349;
    const scalar_t x351 = 2 * x330;
    const scalar_t x352 = S_ikmn_canonical[24] * Wimpn_compressed[9] + x326 + x340;
    const scalar_t x353 = x351 + x352;
    const scalar_t x354 = S_ikmn_canonical[33] * Wimpn_compressed[1];
    const scalar_t x355 = S_ikmn_canonical[36] * Wimpn_compressed[1];
    const scalar_t x356 = x354 + x355;
    const scalar_t x357 = S_ikmn_canonical[31] * Wimpn_compressed[1];
    const scalar_t x358 = S_ikmn_canonical[37] * Wimpn_compressed[1];
    const scalar_t x359 = x357 + x358;
    const scalar_t x360 = S_ikmn_canonical[32] * Wimpn_compressed[1];
    const scalar_t x361 = S_ikmn_canonical[35] * Wimpn_compressed[1];
    const scalar_t x362 = x360 + x361;
    const scalar_t x363 = S_ikmn_canonical[30] * Wimpn_compressed[0] + S_ikmn_canonical[34] * Wimpn_compressed[0] +
                          S_ikmn_canonical[38] * Wimpn_compressed[0];
    const scalar_t x364 = x356 + x359 + x362 + x363;
    const scalar_t x365 = S_ikmn_canonical[31] * Wimpn_compressed[4];
    const scalar_t x366 = S_ikmn_canonical[37] * Wimpn_compressed[5];
    const scalar_t x367 = x365 + x366;
    const scalar_t x368 = S_ikmn_canonical[32] * Wimpn_compressed[4];
    const scalar_t x369 = S_ikmn_canonical[35] * Wimpn_compressed[5];
    const scalar_t x370 = x368 + x369;
    const scalar_t x371 = S_ikmn_canonical[34] * Wimpn_compressed[6];
    const scalar_t x372 = S_ikmn_canonical[38] * Wimpn_compressed[6];
    const scalar_t x373 = S_ikmn_canonical[30] * Wimpn_compressed[2] + x371 + x372;
    const scalar_t x374 = x356 + x367 + x370 + x373;
    const scalar_t x375 = S_ikmn_canonical[33] * Wimpn_compressed[4];
    const scalar_t x376 = S_ikmn_canonical[36] * Wimpn_compressed[5];
    const scalar_t x377 = x375 + x376;
    const scalar_t x378 = S_ikmn_canonical[32] * Wimpn_compressed[8];
    const scalar_t x379 = S_ikmn_canonical[35] * Wimpn_compressed[8];
    const scalar_t x380 = x378 + x379;
    const scalar_t x381 = S_ikmn_canonical[30] * Wimpn_compressed[3];
    const scalar_t x382 = S_ikmn_canonical[34] * Wimpn_compressed[3];
    const scalar_t x383 = S_ikmn_canonical[38] * Wimpn_compressed[9] + x381 + x382;
    const scalar_t x384 = x367 + x377 + x380 + x383;
    const scalar_t x385 = S_ikmn_canonical[32] * Wimpn_compressed[5];
    const scalar_t x386 = S_ikmn_canonical[35] * Wimpn_compressed[4];
    const scalar_t x387 = x385 + x386;
    const scalar_t x388 = S_ikmn_canonical[30] * Wimpn_compressed[6];
    const scalar_t x389 = S_ikmn_canonical[34] * Wimpn_compressed[2] + x372 + x388;
    const scalar_t x390 = x359 + x377 + x387 + x389;
    const scalar_t x391 = S_ikmn_canonical[33] * Wimpn_compressed[5];
    const scalar_t x392 = S_ikmn_canonical[36] * Wimpn_compressed[4];
    const scalar_t x393 = x391 + x392;
    const scalar_t x394 = S_ikmn_canonical[31] * Wimpn_compressed[5];
    const scalar_t x395 = S_ikmn_canonical[37] * Wimpn_compressed[4];
    const scalar_t x396 = x394 + x395;
    const scalar_t x397 = S_ikmn_canonical[38] * Wimpn_compressed[2] + x371 + x388;
    const scalar_t x398 = x362 + x393 + x396 + x397;
    const scalar_t x399 = S_ikmn_canonical[31] * Wimpn_compressed[8];
    const scalar_t x400 = S_ikmn_canonical[37] * Wimpn_compressed[8];
    const scalar_t x401 = x399 + x400;
    const scalar_t x402 = S_ikmn_canonical[38] * Wimpn_compressed[3];
    const scalar_t x403 = S_ikmn_canonical[34] * Wimpn_compressed[9] + x381 + x402;
    const scalar_t x404 = x370 + x393 + x401 + x403;
    const scalar_t x405 = S_ikmn_canonical[33] * Wimpn_compressed[8];
    const scalar_t x406 = S_ikmn_canonical[36] * Wimpn_compressed[8];
    const scalar_t x407 = x405 + x406;
    const scalar_t x408 = S_ikmn_canonical[30] * Wimpn_compressed[7] + S_ikmn_canonical[34] * Wimpn_compressed[7] +
                          S_ikmn_canonical[38] * Wimpn_compressed[7];
    const scalar_t x409 = x380 + x401 + x407 + x408;
    const scalar_t x410 = S_ikmn_canonical[30] * Wimpn_compressed[9] + x382 + x402;
    const scalar_t x411 = x387 + x396 + x407 + x410;
    const scalar_t x412 = x311 + x321 + x338;
    const scalar_t x413 = x332 + x344;
    const scalar_t x414 = x303 + x327;
    const scalar_t x415 = x305 + x341;
    const scalar_t x416 = x336 + x343;
    const scalar_t x417 = x329 + x335 + x349;
    const scalar_t x418 = x375 + x392;
    const scalar_t x419 = x357 + x366;
    const scalar_t x420 = x360 + x369;
    const scalar_t x421 = x373 + x418 + x419 + x420;
    const scalar_t x422 = x358 + x365;
    const scalar_t x423 = x361 + x368;
    const scalar_t x424 = x363 + x418 + x422 + x423;
    const scalar_t x425 = x354 + x406;
    const scalar_t x426 = x378 + x386;
    const scalar_t x427 = x389 + x422 + x425 + x426;
    const scalar_t x428 = x379 + x385;
    const scalar_t x429 = x383 + x419 + x425 + x428;
    const scalar_t x430 = x355 + x405;
    const scalar_t x431 = x394 + x400;
    const scalar_t x432 = x403 + x420 + x430 + x431;
    const scalar_t x433 = x395 + x399;
    const scalar_t x434 = x397 + x423 + x430 + x433;
    const scalar_t x435 = x376 + x391;
    const scalar_t x436 = x410 + x426 + x433 + x435;
    const scalar_t x437 = x408 + x428 + x431 + x435;
    const scalar_t x438 = x306 + x307 + x308 + x351;
    const scalar_t x439 = x303 + x338 + x438;
    const scalar_t x440 = x319 + x348;
    const scalar_t x441 = x314 + x345 + x346 + x347;
    const scalar_t x442 = x329 + x343 + x441;
    const scalar_t x443 = x310 + x352;
    const scalar_t x444 = x375 + x406;
    const scalar_t x445 = x365 + x400;
    const scalar_t x446 = x369 + x385;
    const scalar_t x447 = x383 + x444 + x445 + x446;
    const scalar_t x448 = x357 + x395;
    const scalar_t x449 = x361 + x378;
    const scalar_t x450 = x389 + x444 + x448 + x449;
    const scalar_t x451 = x354 + x392;
    const scalar_t x452 = x368 + x386;
    const scalar_t x453 = x363 + x448 + x451 + x452;
    const scalar_t x454 = x360 + x379;
    const scalar_t x455 = x373 + x445 + x451 + x454;
    const scalar_t x456 = x376 + x405;
    const scalar_t x457 = x366 + x399;
    const scalar_t x458 = x408 + x446 + x456 + x457;
    const scalar_t x459 = x358 + x394;
    const scalar_t x460 = x410 + x449 + x456 + x459;
    const scalar_t x461 = x355 + x391;
    const scalar_t x462 = x397 + x452 + x459 + x461;
    const scalar_t x463 = x403 + x454 + x457 + x461;
    const scalar_t x464 = x305 + x321 + x438;
    const scalar_t x465 = x335 + x344 + x441;
    const scalar_t x466 = x354 + x376;
    const scalar_t x467 = x365 + x395;
    const scalar_t x468 = x361 + x385;
    const scalar_t x469 = x389 + x466 + x467 + x468;
    const scalar_t x470 = x357 + x400;
    const scalar_t x471 = x369 + x378;
    const scalar_t x472 = x383 + x466 + x470 + x471;
    const scalar_t x473 = x355 + x375;
    const scalar_t x474 = x368 + x379;
    const scalar_t x475 = x373 + x470 + x473 + x474;
    const scalar_t x476 = x360 + x386;
    const scalar_t x477 = x363 + x467 + x473 + x476;
    const scalar_t x478 = x391 + x406;
    const scalar_t x479 = x358 + x399;
    const scalar_t x480 = x410 + x468 + x478 + x479;
    const scalar_t x481 = x366 + x394;
    const scalar_t x482 = x408 + x471 + x478 + x481;
    const scalar_t x483 = x392 + x405;
    const scalar_t x484 = x403 + x474 + x481 + x483;
    const scalar_t x485 = x397 + x476 + x479 + x483;
    const scalar_t x486 = S_ikmn_canonical[40] * Wimpn_compressed[1];
    const scalar_t x487 = 2 * x486;
    const scalar_t x488 = S_ikmn_canonical[41] * Wimpn_compressed[1];
    const scalar_t x489 = 2 * x488;
    const scalar_t x490 = S_ikmn_canonical[39] * Wimpn_compressed[0];
    const scalar_t x491 = S_ikmn_canonical[42] * Wimpn_compressed[0];
    const scalar_t x492 = S_ikmn_canonical[44] * Wimpn_compressed[0];
    const scalar_t x493 = S_ikmn_canonical[43] * Wimpn_compressed[1];
    const scalar_t x494 = 2 * x493;
    const scalar_t x495 = x490 + x491 + x492 + x494;
    const scalar_t x496 = x487 + x489 + x495;
    const scalar_t x497 = S_ikmn_canonical[43] * Wimpn_compressed[5];
    const scalar_t x498 = 2 * x497;
    const scalar_t x499 = S_ikmn_canonical[40] * Wimpn_compressed[4];
    const scalar_t x500 = S_ikmn_canonical[44] * Wimpn_compressed[6] + x486 + x499;
    const scalar_t x501 = S_ikmn_canonical[41] * Wimpn_compressed[4];
    const scalar_t x502 = S_ikmn_canonical[42] * Wimpn_compressed[6] + x488 + x501;
    const scalar_t x503 = S_ikmn_canonical[39] * Wimpn_compressed[2] + x500 + x502;
    const scalar_t x504 = x498 + x503;
    const scalar_t x505 = 2 * x499;
    const scalar_t x506 = S_ikmn_canonical[43] * Wimpn_compressed[8];
    const scalar_t x507 = S_ikmn_canonical[39] * Wimpn_compressed[3] + x497 + x506;
    const scalar_t x508 = S_ikmn_canonical[41] * Wimpn_compressed[5];
    const scalar_t x509 = S_ikmn_canonical[41] * Wimpn_compressed[8];
    const scalar_t x510 = S_ikmn_canonical[42] * Wimpn_compressed[3] + x508 + x509;
    const scalar_t x511 = S_ikmn_canonical[44] * Wimpn_compressed[9] + x507 + x510;
    const scalar_t x512 = x505 + x511;
    const scalar_t x513 = 2 * x508;
    const scalar_t x514 = S_ikmn_canonical[43] * Wimpn_compressed[4];
    const scalar_t x515 = S_ikmn_canonical[39] * Wimpn_compressed[6] + x493 + x514;
    const scalar_t x516 = S_ikmn_canonical[42] * Wimpn_compressed[2] + x500 + x515;
    const scalar_t x517 = x513 + x516;
    const scalar_t x518 = S_ikmn_canonical[40] * Wimpn_compressed[5];
    const scalar_t x519 = 2 * x518;
    const scalar_t x520 = S_ikmn_canonical[44] * Wimpn_compressed[2] + x502 + x515;
    const scalar_t x521 = x519 + x520;
    const scalar_t x522 = 2 * x501;
    const scalar_t x523 = S_ikmn_canonical[40] * Wimpn_compressed[8];
    const scalar_t x524 = S_ikmn_canonical[44] * Wimpn_compressed[3] + x518 + x523;
    const scalar_t x525 = S_ikmn_canonical[42] * Wimpn_compressed[9] + x507 + x524;
    const scalar_t x526 = x522 + x525;
    const scalar_t x527 = 2 * x523;
    const scalar_t x528 = 2 * x509;
    const scalar_t x529 = S_ikmn_canonical[39] * Wimpn_compressed[7];
    const scalar_t x530 = S_ikmn_canonical[42] * Wimpn_compressed[7];
    const scalar_t x531 = S_ikmn_canonical[44] * Wimpn_compressed[7];
    const scalar_t x532 = 2 * x506;
    const scalar_t x533 = x529 + x530 + x531 + x532;
    const scalar_t x534 = x527 + x528 + x533;
    const scalar_t x535 = 2 * x514;
    const scalar_t x536 = S_ikmn_canonical[39] * Wimpn_compressed[9] + x510 + x524;
    const scalar_t x537 = x535 + x536;
    const scalar_t x538 = x495 + x505 + x522;
    const scalar_t x539 = x516 + x528;
    const scalar_t x540 = x487 + x511;
    const scalar_t x541 = x489 + x525;
    const scalar_t x542 = x520 + x527;
    const scalar_t x543 = x513 + x519 + x533;
    const scalar_t x544 = x490 + x491 + x492 + x535;
    const scalar_t x545 = x487 + x522 + x544;
    const scalar_t x546 = x503 + x532;
    const scalar_t x547 = x498 + x529 + x530 + x531;
    const scalar_t x548 = x513 + x527 + x547;
    const scalar_t x549 = x494 + x536;
    const scalar_t x550 = x489 + x505 + x544;
    const scalar_t x551 = x519 + x528 + x547;
    H[0]                = x10;
    H[1]                = x18;
    H[2]                = x26;
    H[3]                = x31;
    H[4]                = x35;
    H[5]                = x40;
    H[6]                = x48;
    H[7]                = x51;
    H[8]                = x62;
    H[9]                = x72;
    H[10]               = x82;
    H[11]               = x88;
    H[12]               = x96;
    H[13]               = x102;
    H[14]               = x107;
    H[15]               = x109;
    H[16]               = x120;
    H[17]               = x130;
    H[18]               = x140;
    H[19]               = x146;
    H[20]               = x154;
    H[21]               = x160;
    H[22]               = x165;
    H[23]               = x167;
    H[24]               = x18;
    H[25]               = x168;
    H[26]               = x169;
    H[27]               = x170;
    H[28]               = x171;
    H[29]               = x172;
    H[30]               = x51;
    H[31]               = x173;
    H[32]               = x177;
    H[33]               = x180;
    H[34]               = x183;
    H[35]               = x185;
    H[36]               = x188;
    H[37]               = x190;
    H[38]               = x192;
    H[39]               = x193;
    H[40]               = x197;
    H[41]               = x200;
    H[42]               = x203;
    H[43]               = x205;
    H[44]               = x208;
    H[45]               = x210;
    H[46]               = x212;
    H[47]               = x213;
    H[48]               = x26;
    H[49]               = x169;
    H[50]               = x215;
    H[51]               = x216;
    H[52]               = x218;
    H[53]               = x219;
    H[54]               = x35;
    H[55]               = x171;
    H[56]               = x223;
    H[57]               = x226;
    H[58]               = x229;
    H[59]               = x231;
    H[60]               = x234;
    H[61]               = x236;
    H[62]               = x238;
    H[63]               = x239;
    H[64]               = x243;
    H[65]               = x246;
    H[66]               = x249;
    H[67]               = x251;
    H[68]               = x254;
    H[69]               = x256;
    H[70]               = x258;
    H[71]               = x259;
    H[72]               = x31;
    H[73]               = x170;
    H[74]               = x216;
    H[75]               = x260;
    H[76]               = x219;
    H[77]               = x261;
    H[78]               = x40;
    H[79]               = x172;
    H[80]               = x265;
    H[81]               = x268;
    H[82]               = x271;
    H[83]               = x273;
    H[84]               = x276;
    H[85]               = x278;
    H[86]               = x280;
    H[87]               = x281;
    H[88]               = x285;
    H[89]               = x288;
    H[90]               = x291;
    H[91]               = x293;
    H[92]               = x296;
    H[93]               = x298;
    H[94]               = x300;
    H[95]               = x301;
    H[96]               = x35;
    H[97]               = x171;
    H[98]               = x218;
    H[99]               = x219;
    H[100]              = x215;
    H[101]              = x216;
    H[102]              = x26;
    H[103]              = x169;
    H[104]              = x238;
    H[105]              = x239;
    H[106]              = x234;
    H[107]              = x236;
    H[108]              = x229;
    H[109]              = x231;
    H[110]              = x223;
    H[111]              = x226;
    H[112]              = x258;
    H[113]              = x259;
    H[114]              = x254;
    H[115]              = x256;
    H[116]              = x249;
    H[117]              = x251;
    H[118]              = x243;
    H[119]              = x246;
    H[120]              = x40;
    H[121]              = x172;
    H[122]              = x219;
    H[123]              = x261;
    H[124]              = x216;
    H[125]              = x260;
    H[126]              = x31;
    H[127]              = x170;
    H[128]              = x280;
    H[129]              = x281;
    H[130]              = x276;
    H[131]              = x278;
    H[132]              = x271;
    H[133]              = x273;
    H[134]              = x265;
    H[135]              = x268;
    H[136]              = x300;
    H[137]              = x301;
    H[138]              = x296;
    H[139]              = x298;
    H[140]              = x291;
    H[141]              = x293;
    H[142]              = x285;
    H[143]              = x288;
    H[144]              = x48;
    H[145]              = x51;
    H[146]              = x35;
    H[147]              = x40;
    H[148]              = x26;
    H[149]              = x31;
    H[150]              = x10;
    H[151]              = x18;
    H[152]              = x107;
    H[153]              = x109;
    H[154]              = x96;
    H[155]              = x102;
    H[156]              = x82;
    H[157]              = x88;
    H[158]              = x62;
    H[159]              = x72;
    H[160]              = x165;
    H[161]              = x167;
    H[162]              = x154;
    H[163]              = x160;
    H[164]              = x140;
    H[165]              = x146;
    H[166]              = x120;
    H[167]              = x130;
    H[168]              = x51;
    H[169]              = x173;
    H[170]              = x171;
    H[171]              = x172;
    H[172]              = x169;
    H[173]              = x170;
    H[174]              = x18;
    H[175]              = x168;
    H[176]              = x192;
    H[177]              = x193;
    H[178]              = x188;
    H[179]              = x190;
    H[180]              = x183;
    H[181]              = x185;
    H[182]              = x177;
    H[183]              = x180;
    H[184]              = x212;
    H[185]              = x213;
    H[186]              = x208;
    H[187]              = x210;
    H[188]              = x203;
    H[189]              = x205;
    H[190]              = x197;
    H[191]              = x200;
    H[192]              = x62;
    H[193]              = x177;
    H[194]              = x223;
    H[195]              = x265;
    H[196]              = x238;
    H[197]              = x280;
    H[198]              = x107;
    H[199]              = x192;
    H[200]              = x312;
    H[201]              = x320;
    H[202]              = x328;
    H[203]              = x333;
    H[204]              = x337;
    H[205]              = x342;
    H[206]              = x350;
    H[207]              = x353;
    H[208]              = x364;
    H[209]              = x374;
    H[210]              = x384;
    H[211]              = x390;
    H[212]              = x398;
    H[213]              = x404;
    H[214]              = x409;
    H[215]              = x411;
    H[216]              = x72;
    H[217]              = x180;
    H[218]              = x226;
    H[219]              = x268;
    H[220]              = x239;
    H[221]              = x281;
    H[222]              = x109;
    H[223]              = x193;
    H[224]              = x320;
    H[225]              = x412;
    H[226]              = x413;
    H[227]              = x414;
    H[228]              = x415;
    H[229]              = x416;
    H[230]              = x353;
    H[231]              = x417;
    H[232]              = x421;
    H[233]              = x424;
    H[234]              = x427;
    H[235]              = x429;
    H[236]              = x432;
    H[237]              = x434;
    H[238]              = x436;
    H[239]              = x437;
    H[240]              = x82;
    H[241]              = x183;
    H[242]              = x229;
    H[243]              = x271;
    H[244]              = x234;
    H[245]              = x276;
    H[246]              = x96;
    H[247]              = x188;
    H[248]              = x328;
    H[249]              = x413;
    H[250]              = x439;
    H[251]              = x440;
    H[252]              = x442;
    H[253]              = x443;
    H[254]              = x337;
    H[255]              = x415;
    H[256]              = x447;
    H[257]              = x450;
    H[258]              = x453;
    H[259]              = x455;
    H[260]              = x458;
    H[261]              = x460;
    H[262]              = x462;
    H[263]              = x463;
    H[264]              = x88;
    H[265]              = x185;
    H[266]              = x231;
    H[267]              = x273;
    H[268]              = x236;
    H[269]              = x278;
    H[270]              = x102;
    H[271]              = x190;
    H[272]              = x333;
    H[273]              = x414;
    H[274]              = x440;
    H[275]              = x464;
    H[276]              = x443;
    H[277]              = x465;
    H[278]              = x342;
    H[279]              = x416;
    H[280]              = x469;
    H[281]              = x472;
    H[282]              = x475;
    H[283]              = x477;
    H[284]              = x480;
    H[285]              = x482;
    H[286]              = x484;
    H[287]              = x485;
    H[288]              = x96;
    H[289]              = x188;
    H[290]              = x234;
    H[291]              = x276;
    H[292]              = x229;
    H[293]              = x271;
    H[294]              = x82;
    H[295]              = x183;
    H[296]              = x337;
    H[297]              = x415;
    H[298]              = x442;
    H[299]              = x443;
    H[300]              = x439;
    H[301]              = x440;
    H[302]              = x328;
    H[303]              = x413;
    H[304]              = x462;
    H[305]              = x463;
    H[306]              = x458;
    H[307]              = x460;
    H[308]              = x453;
    H[309]              = x455;
    H[310]              = x447;
    H[311]              = x450;
    H[312]              = x102;
    H[313]              = x190;
    H[314]              = x236;
    H[315]              = x278;
    H[316]              = x231;
    H[317]              = x273;
    H[318]              = x88;
    H[319]              = x185;
    H[320]              = x342;
    H[321]              = x416;
    H[322]              = x443;
    H[323]              = x465;
    H[324]              = x440;
    H[325]              = x464;
    H[326]              = x333;
    H[327]              = x414;
    H[328]              = x484;
    H[329]              = x485;
    H[330]              = x480;
    H[331]              = x482;
    H[332]              = x475;
    H[333]              = x477;
    H[334]              = x469;
    H[335]              = x472;
    H[336]              = x107;
    H[337]              = x192;
    H[338]              = x238;
    H[339]              = x280;
    H[340]              = x223;
    H[341]              = x265;
    H[342]              = x62;
    H[343]              = x177;
    H[344]              = x350;
    H[345]              = x353;
    H[346]              = x337;
    H[347]              = x342;
    H[348]              = x328;
    H[349]              = x333;
    H[350]              = x312;
    H[351]              = x320;
    H[352]              = x409;
    H[353]              = x411;
    H[354]              = x398;
    H[355]              = x404;
    H[356]              = x384;
    H[357]              = x390;
    H[358]              = x364;
    H[359]              = x374;
    H[360]              = x109;
    H[361]              = x193;
    H[362]              = x239;
    H[363]              = x281;
    H[364]              = x226;
    H[365]              = x268;
    H[366]              = x72;
    H[367]              = x180;
    H[368]              = x353;
    H[369]              = x417;
    H[370]              = x415;
    H[371]              = x416;
    H[372]              = x413;
    H[373]              = x414;
    H[374]              = x320;
    H[375]              = x412;
    H[376]              = x436;
    H[377]              = x437;
    H[378]              = x432;
    H[379]              = x434;
    H[380]              = x427;
    H[381]              = x429;
    H[382]              = x421;
    H[383]              = x424;
    H[384]              = x120;
    H[385]              = x197;
    H[386]              = x243;
    H[387]              = x285;
    H[388]              = x258;
    H[389]              = x300;
    H[390]              = x165;
    H[391]              = x212;
    H[392]              = x364;
    H[393]              = x421;
    H[394]              = x447;
    H[395]              = x469;
    H[396]              = x462;
    H[397]              = x484;
    H[398]              = x409;
    H[399]              = x436;
    H[400]              = x496;
    H[401]              = x504;
    H[402]              = x512;
    H[403]              = x517;
    H[404]              = x521;
    H[405]              = x526;
    H[406]              = x534;
    H[407]              = x537;
    H[408]              = x130;
    H[409]              = x200;
    H[410]              = x246;
    H[411]              = x288;
    H[412]              = x259;
    H[413]              = x301;
    H[414]              = x167;
    H[415]              = x213;
    H[416]              = x374;
    H[417]              = x424;
    H[418]              = x450;
    H[419]              = x472;
    H[420]              = x463;
    H[421]              = x485;
    H[422]              = x411;
    H[423]              = x437;
    H[424]              = x504;
    H[425]              = x538;
    H[426]              = x539;
    H[427]              = x540;
    H[428]              = x541;
    H[429]              = x542;
    H[430]              = x537;
    H[431]              = x543;
    H[432]              = x140;
    H[433]              = x203;
    H[434]              = x249;
    H[435]              = x291;
    H[436]              = x254;
    H[437]              = x296;
    H[438]              = x154;
    H[439]              = x208;
    H[440]              = x384;
    H[441]              = x427;
    H[442]              = x453;
    H[443]              = x475;
    H[444]              = x458;
    H[445]              = x480;
    H[446]              = x398;
    H[447]              = x432;
    H[448]              = x512;
    H[449]              = x539;
    H[450]              = x545;
    H[451]              = x546;
    H[452]              = x548;
    H[453]              = x549;
    H[454]              = x521;
    H[455]              = x541;
    H[456]              = x146;
    H[457]              = x205;
    H[458]              = x251;
    H[459]              = x293;
    H[460]              = x256;
    H[461]              = x298;
    H[462]              = x160;
    H[463]              = x210;
    H[464]              = x390;
    H[465]              = x429;
    H[466]              = x455;
    H[467]              = x477;
    H[468]              = x460;
    H[469]              = x482;
    H[470]              = x404;
    H[471]              = x434;
    H[472]              = x517;
    H[473]              = x540;
    H[474]              = x546;
    H[475]              = x550;
    H[476]              = x549;
    H[477]              = x551;
    H[478]              = x526;
    H[479]              = x542;
    H[480]              = x154;
    H[481]              = x208;
    H[482]              = x254;
    H[483]              = x296;
    H[484]              = x249;
    H[485]              = x291;
    H[486]              = x140;
    H[487]              = x203;
    H[488]              = x398;
    H[489]              = x432;
    H[490]              = x458;
    H[491]              = x480;
    H[492]              = x453;
    H[493]              = x475;
    H[494]              = x384;
    H[495]              = x427;
    H[496]              = x521;
    H[497]              = x541;
    H[498]              = x548;
    H[499]              = x549;
    H[500]              = x545;
    H[501]              = x546;
    H[502]              = x512;
    H[503]              = x539;
    H[504]              = x160;
    H[505]              = x210;
    H[506]              = x256;
    H[507]              = x298;
    H[508]              = x251;
    H[509]              = x293;
    H[510]              = x146;
    H[511]              = x205;
    H[512]              = x404;
    H[513]              = x434;
    H[514]              = x460;
    H[515]              = x482;
    H[516]              = x455;
    H[517]              = x477;
    H[518]              = x390;
    H[519]              = x429;
    H[520]              = x526;
    H[521]              = x542;
    H[522]              = x549;
    H[523]              = x551;
    H[524]              = x546;
    H[525]              = x550;
    H[526]              = x517;
    H[527]              = x540;
    H[528]              = x165;
    H[529]              = x212;
    H[530]              = x258;
    H[531]              = x300;
    H[532]              = x243;
    H[533]              = x285;
    H[534]              = x120;
    H[535]              = x197;
    H[536]              = x409;
    H[537]              = x436;
    H[538]              = x462;
    H[539]              = x484;
    H[540]              = x447;
    H[541]              = x469;
    H[542]              = x364;
    H[543]              = x421;
    H[544]              = x534;
    H[545]              = x537;
    H[546]              = x521;
    H[547]              = x526;
    H[548]              = x512;
    H[549]              = x517;
    H[550]              = x496;
    H[551]              = x504;
    H[552]              = x167;
    H[553]              = x213;
    H[554]              = x259;
    H[555]              = x301;
    H[556]              = x246;
    H[557]              = x288;
    H[558]              = x130;
    H[559]              = x200;
    H[560]              = x411;
    H[561]              = x437;
    H[562]              = x463;
    H[563]              = x485;
    H[564]              = x450;
    H[565]              = x472;
    H[566]              = x374;
    H[567]              = x424;
    H[568]              = x537;
    H[569]              = x543;
    H[570]              = x541;
    H[571]              = x542;
    H[572]              = x539;
    H[573]              = x540;
    H[574]              = x504;
    H[575]              = x538;
}

#define HEX8_S_IKMN_SIZE 45
static SFEM_INLINE void hex8_S_ikmn_neohookean(const scalar_t *const SFEM_RESTRICT adjugate,
                                               const scalar_t                      jacobian_determinant,
                                               const scalar_t                      qx,
                                               const scalar_t                      qy,
                                               const scalar_t                      qz,
                                               const scalar_t *const SFEM_RESTRICT F,
                                               const scalar_t                      mu,
                                               const scalar_t                      lmbda,
                                               const scalar_t                      qw,
                                               scalar_t *const SFEM_RESTRICT       S_ikmn_canonical) {
    // mundane ops: 1014 divs: 2 sqrts: 0
    // total ops: 1030
    const scalar_t x0    = F[4] * F[8];
    const scalar_t x1    = F[5] * F[6];
    const scalar_t x2    = F[3] * F[7];
    const scalar_t x3    = F[5] * F[7];
    const scalar_t x4    = F[3] * F[8];
    const scalar_t x5    = F[4] * F[6];
    const scalar_t x6    = F[0] * x0 - F[0] * x3 + F[1] * x1 - F[1] * x4 + F[2] * x2 - F[2] * x5;
    const scalar_t x7    = (1 / POW2(x6));
    const scalar_t x8    = x0 - x3;
    const scalar_t x9    = x7 * x8;
    const scalar_t x10   = x2 - x5;
    const scalar_t x11   = lmbda * log(x6);
    const scalar_t x12   = lmbda + mu - x11;
    const scalar_t x13   = adjugate[2] * x12;
    const scalar_t x14   = x10 * x13;
    const scalar_t x15   = -x1 + x4;
    const scalar_t x16   = x12 * x9;
    const scalar_t x17   = x15 * x16;
    const scalar_t x18   = x7 * POW2(x8);
    const scalar_t x19   = lmbda * x18 + mu * x18 + mu - x11 * x18;
    const scalar_t x20   = adjugate[0] * x19 - adjugate[1] * x17 + x14 * x9;
    const scalar_t x21   = x10 * x16;
    const scalar_t x22   = adjugate[1] * x12;
    const scalar_t x23   = x15 * x7;
    const scalar_t x24   = x10 * x23;
    const scalar_t x25   = POW2(x10) * x7;
    const scalar_t x26   = lmbda * x25 + mu * x25 + mu - x11 * x25;
    const scalar_t x27   = adjugate[0] * x21 + adjugate[2] * x26 - x22 * x24;
    const scalar_t x28   = POW2(x15) * x7;
    const scalar_t x29   = lmbda * x28 + mu * x28 + mu - x11 * x28;
    const scalar_t x30   = -adjugate[0] * x17 + adjugate[1] * x29 - x14 * x23;
    const scalar_t x31   = qw / jacobian_determinant;
    const scalar_t x32   = adjugate[3] * x19 - adjugate[4] * x17 + adjugate[5] * x21;
    const scalar_t x33   = x12 * x24;
    const scalar_t x34   = adjugate[3] * x21 - adjugate[4] * x33 + adjugate[5] * x26;
    const scalar_t x35   = -adjugate[3] * x17 + adjugate[4] * x29 - adjugate[5] * x33;
    const scalar_t x36   = F[0] * F[8] - F[2] * F[6];
    const scalar_t x37   = 1.0 / x6;
    const scalar_t x38   = x15 * x37;
    const scalar_t x39   = x36 * x38;
    const scalar_t x40   = F[8] * mu;
    const scalar_t x41   = F[8] * x11;
    const scalar_t x42   = F[1] * F[8] - F[2] * F[7];
    const scalar_t x43   = x38 * x42;
    const scalar_t x44   = lmbda * x43 + mu * x43 - x11 * x43 + x40 - x41;
    const scalar_t x45   = F[6] * mu;
    const scalar_t x46   = F[6] * x11;
    const scalar_t x47   = F[0] * F[7] - F[1] * F[6];
    const scalar_t x48   = x38 * x47;
    const scalar_t x49   = lmbda * x48 + mu * x48 - x11 * x48 - x45 + x46;
    const scalar_t x50   = adjugate[0] * x44 + adjugate[2] * x49 - x22 * x39;
    const scalar_t x51   = x37 * x8;
    const scalar_t x52   = x12 * x51;
    const scalar_t x53   = x42 * x52;
    const scalar_t x54   = F[7] * mu;
    const scalar_t x55   = F[7] * x11;
    const scalar_t x56   = x47 * x51;
    const scalar_t x57   = lmbda * x56 + mu * x56 - x11 * x56 - x54 + x55;
    const scalar_t x58   = x36 * x51;
    const scalar_t x59   = lmbda * x58 + mu * x58 - x11 * x58 - x40 + x41;
    const scalar_t x60   = -adjugate[0] * x53 + adjugate[1] * x59 - adjugate[2] * x57;
    const scalar_t x61   = x10 * x37;
    const scalar_t x62   = x47 * x61;
    const scalar_t x63   = x42 * x61;
    const scalar_t x64   = lmbda * x63 + mu * x63 - x11 * x63 + x54 - x55;
    const scalar_t x65   = x36 * x61;
    const scalar_t x66   = lmbda * x65 + mu * x65 - x11 * x65 + x45 - x46;
    const scalar_t x67   = -adjugate[0] * x64 + adjugate[1] * x66 - x13 * x62;
    const scalar_t x68   = x31 * x37;
    const scalar_t x69   = x12 * x39;
    const scalar_t x70   = adjugate[3] * x44 - adjugate[4] * x69 + adjugate[5] * x49;
    const scalar_t x71   = -adjugate[3] * x53 + adjugate[4] * x59 - adjugate[5] * x57;
    const scalar_t x72   = x12 * x62;
    const scalar_t x73   = -adjugate[3] * x64 + adjugate[4] * x66 - adjugate[5] * x72;
    const scalar_t x74   = adjugate[6] * x44 - adjugate[7] * x69 + adjugate[8] * x49;
    const scalar_t x75   = -adjugate[6] * x53 + adjugate[7] * x59 - adjugate[8] * x57;
    const scalar_t x76   = -adjugate[6] * x64 + adjugate[7] * x66 - adjugate[8] * x72;
    const scalar_t x77   = F[1] * F[5] - F[2] * F[4];
    const scalar_t x78   = x52 * x77;
    const scalar_t x79   = F[4] * mu;
    const scalar_t x80   = F[4] * x11;
    const scalar_t x81   = F[0] * F[4] - F[1] * F[3];
    const scalar_t x82   = x51 * x81;
    const scalar_t x83   = lmbda * x82 + mu * x82 - x11 * x82 - x79 + x80;
    const scalar_t x84   = F[5] * mu;
    const scalar_t x85   = F[5] * x11;
    const scalar_t x86   = F[0] * F[5] - F[2] * F[3];
    const scalar_t x87   = x51 * x86;
    const scalar_t x88   = lmbda * x87 + mu * x87 - x11 * x87 - x84 + x85;
    const scalar_t x89   = adjugate[0] * x78 - adjugate[1] * x88 + adjugate[2] * x83;
    const scalar_t x90   = x61 * x81;
    const scalar_t x91   = x61 * x77;
    const scalar_t x92   = lmbda * x91 + mu * x91 - x11 * x91 + x79 - x80;
    const scalar_t x93   = F[3] * mu;
    const scalar_t x94   = F[3] * x11;
    const scalar_t x95   = x61 * x86;
    const scalar_t x96   = lmbda * x95 + mu * x95 - x11 * x95 + x93 - x94;
    const scalar_t x97   = adjugate[0] * x92 - adjugate[1] * x96 + x13 * x90;
    const scalar_t x98   = x38 * x77;
    const scalar_t x99   = lmbda * x98 + mu * x98 - x11 * x98 + x84 - x85;
    const scalar_t x100  = x38 * x81;
    const scalar_t x101  = lmbda * x100 + mu * x100 - x100 * x11 - x93 + x94;
    const scalar_t x102  = -adjugate[0] * x99 + adjugate[1] * x12 * x15 * x37 * x86 - adjugate[2] * x101;
    const scalar_t x103  = adjugate[3] * x78 - adjugate[4] * x88 + adjugate[5] * x83;
    const scalar_t x104  = x12 * x90;
    const scalar_t x105  = adjugate[3] * x92 - adjugate[4] * x96 + adjugate[5] * x104;
    const scalar_t x106  = -adjugate[3] * x99 + adjugate[4] * x12 * x15 * x37 * x86 - adjugate[5] * x101;
    const scalar_t x107  = adjugate[6] * x78 - adjugate[7] * x88 + adjugate[8] * x83;
    const scalar_t x108  = adjugate[6] * x92 - adjugate[7] * x96 + adjugate[8] * x104;
    const scalar_t x109  = -adjugate[6] * x99 + adjugate[7] * x12 * x15 * x37 * x86 - adjugate[8] * x101;
    const scalar_t x110  = x42 * x7;
    const scalar_t x111  = x13 * x47;
    const scalar_t x112  = x110 * x36;
    const scalar_t x113  = POW2(x42) * x7;
    const scalar_t x114  = lmbda * x113 + mu * x113 + mu - x11 * x113;
    const scalar_t x115  = adjugate[0] * x114 + x110 * x111 - x112 * x22;
    const scalar_t x116  = adjugate[0] * x12;
    const scalar_t x117  = x110 * x47;
    const scalar_t x118  = x36 * x7;
    const scalar_t x119  = x118 * x47;
    const scalar_t x120  = POW2(x47) * x7;
    const scalar_t x121  = lmbda * x120 + mu * x120 + mu - x11 * x120;
    const scalar_t x122  = adjugate[2] * x121 + x116 * x117 - x119 * x22;
    const scalar_t x123  = POW2(x36) * x7;
    const scalar_t x124  = lmbda * x123 + mu * x123 + mu - x11 * x123;
    const scalar_t x125  = adjugate[1] * x124 - x111 * x118 - x112 * x116;
    const scalar_t x126  = adjugate[5] * x12;
    const scalar_t x127  = x112 * x12;
    const scalar_t x128  = adjugate[3] * x114 - adjugate[4] * x127 + x117 * x126;
    const scalar_t x129  = x117 * x12;
    const scalar_t x130  = x119 * x12;
    const scalar_t x131  = adjugate[3] * x129 - adjugate[4] * x130 + adjugate[5] * x121;
    const scalar_t x132  = -adjugate[3] * x127 + adjugate[4] * x124 - x119 * x126;
    const scalar_t x133  = x36 * x37;
    const scalar_t x134  = x133 * x86;
    const scalar_t x135  = F[2] * mu;
    const scalar_t x136  = F[2] * x11;
    const scalar_t x137  = x133 * x77;
    const scalar_t x138  = lmbda * x137 + mu * x137 - x11 * x137 + x135 - x136;
    const scalar_t x139  = F[0] * mu;
    const scalar_t x140  = F[0] * x11;
    const scalar_t x141  = x133 * x81;
    const scalar_t x142  = lmbda * x141 + mu * x141 - x11 * x141 - x139 + x140;
    const scalar_t x143  = adjugate[0] * x138 + adjugate[2] * x142 - x134 * x22;
    const scalar_t x144  = x37 * x42;
    const scalar_t x145  = x144 * x77;
    const scalar_t x146  = F[1] * mu;
    const scalar_t x147  = F[1] * x11;
    const scalar_t x148  = x144 * x81;
    const scalar_t x149  = lmbda * x148 + mu * x148 - x11 * x148 - x146 + x147;
    const scalar_t x150  = x144 * x86;
    const scalar_t x151  = lmbda * x150 + mu * x150 - x11 * x150 - x135 + x136;
    const scalar_t x152  = adjugate[1] * x151 - adjugate[2] * x149 - x116 * x145;
    const scalar_t x153  = x37 * x47;
    const scalar_t x154  = x153 * x81;
    const scalar_t x155  = x153 * x77;
    const scalar_t x156  = lmbda * x155 + mu * x155 - x11 * x155 + x146 - x147;
    const scalar_t x157  = x153 * x86;
    const scalar_t x158  = lmbda * x157 + mu * x157 - x11 * x157 + x139 - x140;
    const scalar_t x159  = -adjugate[0] * x156 + adjugate[1] * x158 - x13 * x154;
    const scalar_t x160  = x12 * x134;
    const scalar_t x161  = adjugate[3] * x138 - adjugate[4] * x160 + adjugate[5] * x142;
    const scalar_t x162  = x12 * x145;
    const scalar_t x163  = -adjugate[3] * x162 + adjugate[4] * x151 - adjugate[5] * x149;
    const scalar_t x164  = -adjugate[3] * x156 + adjugate[4] * x158 - x126 * x154;
    const scalar_t x165  = adjugate[6] * x138 - adjugate[7] * x160 + adjugate[8] * x142;
    const scalar_t x166  = -adjugate[6] * x162 + adjugate[7] * x151 - adjugate[8] * x149;
    const scalar_t x167  = adjugate[8] * x12;
    const scalar_t x168  = -adjugate[6] * x156 + adjugate[7] * x158 - x154 * x167;
    const scalar_t x169  = x7 * x77;
    const scalar_t x170  = x13 * x81;
    const scalar_t x171  = x169 * x86;
    const scalar_t x172  = x7 * POW2(x77);
    const scalar_t x173  = lmbda * x172 + mu * x172 + mu - x11 * x172;
    const scalar_t x174  = adjugate[0] * x173 + x169 * x170 - x171 * x22;
    const scalar_t x175  = x169 * x81;
    const scalar_t x176  = x7 * x86;
    const scalar_t x177  = x176 * x81;
    const scalar_t x178  = x7 * POW2(x81);
    const scalar_t x179  = lmbda * x178 + mu * x178 + mu - x11 * x178;
    const scalar_t x180  = adjugate[2] * x179 + x116 * x175 - x177 * x22;
    const scalar_t x181  = x7 * POW2(x86);
    const scalar_t x182  = lmbda * x181 + mu * x181 + mu - x11 * x181;
    const scalar_t x183  = adjugate[1] * x182 - x116 * x171 - x170 * x176;
    const scalar_t x184  = x12 * x171;
    const scalar_t x185  = adjugate[3] * x173 - adjugate[4] * x184 + x126 * x175;
    const scalar_t x186  = x12 * x175;
    const scalar_t x187  = x12 * x177;
    const scalar_t x188  = adjugate[3] * x186 - adjugate[4] * x187 + adjugate[5] * x179;
    const scalar_t x189  = -adjugate[3] * x184 + adjugate[4] * x182 - x126 * x177;
    S_ikmn_canonical[0]  = x31 * (adjugate[0] * x20 + adjugate[1] * x30 + adjugate[2] * x27);
    S_ikmn_canonical[1]  = x31 * (adjugate[3] * x20 + adjugate[4] * x30 + adjugate[5] * x27);
    S_ikmn_canonical[2]  = x31 * (adjugate[6] * x20 + adjugate[7] * x30 + adjugate[8] * x27);
    S_ikmn_canonical[3]  = x31 * (adjugate[3] * x32 + adjugate[4] * x35 + adjugate[5] * x34);
    S_ikmn_canonical[4]  = x31 * (adjugate[6] * x32 + adjugate[7] * x35 + adjugate[8] * x34);
    S_ikmn_canonical[5]  = x31 * (adjugate[6] * (adjugate[6] * x19 - adjugate[7] * x17 + adjugate[8] * x21) +
                                 adjugate[7] * (-adjugate[6] * x17 + adjugate[7] * x29 - adjugate[8] * x33) +
                                 adjugate[8] * (adjugate[6] * x21 - adjugate[7] * x33 + adjugate[8] * x26));
    S_ikmn_canonical[6]  = x68 * (adjugate[0] * x60 + adjugate[1] * x50 + adjugate[2] * x67);
    S_ikmn_canonical[7]  = x68 * (adjugate[3] * x60 + adjugate[4] * x50 + adjugate[5] * x67);
    S_ikmn_canonical[8]  = x68 * (adjugate[6] * x60 + adjugate[7] * x50 + adjugate[8] * x67);
    S_ikmn_canonical[9]  = x68 * (adjugate[0] * x71 + adjugate[1] * x70 + adjugate[2] * x73);
    S_ikmn_canonical[10] = x68 * (adjugate[3] * x71 + adjugate[4] * x70 + adjugate[5] * x73);
    S_ikmn_canonical[11] = x68 * (adjugate[6] * x71 + adjugate[7] * x70 + adjugate[8] * x73);
    S_ikmn_canonical[12] = x68 * (adjugate[0] * x75 + adjugate[1] * x74 + adjugate[2] * x76);
    S_ikmn_canonical[13] = x68 * (adjugate[3] * x75 + adjugate[4] * x74 + adjugate[5] * x76);
    S_ikmn_canonical[14] = x68 * (adjugate[6] * x75 + adjugate[7] * x74 + adjugate[8] * x76);
    S_ikmn_canonical[15] = x68 * (adjugate[0] * x89 + adjugate[1] * x102 + adjugate[2] * x97);
    S_ikmn_canonical[16] = x68 * (adjugate[3] * x89 + adjugate[4] * x102 + adjugate[5] * x97);
    S_ikmn_canonical[17] = x68 * (adjugate[6] * x89 + adjugate[7] * x102 + adjugate[8] * x97);
    S_ikmn_canonical[18] = x68 * (adjugate[0] * x103 + adjugate[1] * x106 + adjugate[2] * x105);
    S_ikmn_canonical[19] = x68 * (adjugate[3] * x103 + adjugate[4] * x106 + adjugate[5] * x105);
    S_ikmn_canonical[20] = x68 * (adjugate[6] * x103 + adjugate[7] * x106 + adjugate[8] * x105);
    S_ikmn_canonical[21] = x68 * (adjugate[0] * x107 + adjugate[1] * x109 + adjugate[2] * x108);
    S_ikmn_canonical[22] = x68 * (adjugate[3] * x107 + adjugate[4] * x109 + adjugate[5] * x108);
    S_ikmn_canonical[23] = x68 * (adjugate[6] * x107 + adjugate[7] * x109 + adjugate[8] * x108);
    S_ikmn_canonical[24] = x31 * (adjugate[0] * x115 + adjugate[1] * x125 + adjugate[2] * x122);
    S_ikmn_canonical[25] = x31 * (adjugate[3] * x115 + adjugate[4] * x125 + adjugate[5] * x122);
    S_ikmn_canonical[26] = x31 * (adjugate[6] * x115 + adjugate[7] * x125 + adjugate[8] * x122);
    S_ikmn_canonical[27] = x31 * (adjugate[3] * x128 + adjugate[4] * x132 + adjugate[5] * x131);
    S_ikmn_canonical[28] = x31 * (adjugate[6] * x128 + adjugate[7] * x132 + adjugate[8] * x131);
    S_ikmn_canonical[29] = x31 * (adjugate[6] * (adjugate[6] * x114 - adjugate[7] * x127 + adjugate[8] * x129) +
                                  adjugate[7] * (-adjugate[6] * x127 + adjugate[7] * x124 - adjugate[8] * x130) +
                                  adjugate[8] * (adjugate[6] * x129 - adjugate[7] * x130 + adjugate[8] * x121));
    S_ikmn_canonical[30] = x68 * (adjugate[0] * x152 + adjugate[1] * x143 + adjugate[2] * x159);
    S_ikmn_canonical[31] = x68 * (adjugate[3] * x152 + adjugate[4] * x143 + adjugate[5] * x159);
    S_ikmn_canonical[32] = x68 * (adjugate[6] * x152 + adjugate[7] * x143 + adjugate[8] * x159);
    S_ikmn_canonical[33] = x68 * (adjugate[0] * x163 + adjugate[1] * x161 + adjugate[2] * x164);
    S_ikmn_canonical[34] = x68 * (adjugate[3] * x163 + adjugate[4] * x161 + adjugate[5] * x164);
    S_ikmn_canonical[35] = x68 * (adjugate[6] * x163 + adjugate[7] * x161 + adjugate[8] * x164);
    S_ikmn_canonical[36] = x68 * (adjugate[0] * x166 + adjugate[1] * x165 + adjugate[2] * x168);
    S_ikmn_canonical[37] = x68 * (adjugate[3] * x166 + adjugate[4] * x165 + adjugate[5] * x168);
    S_ikmn_canonical[38] = x68 * (adjugate[6] * x166 + adjugate[7] * x165 + adjugate[8] * x168);
    S_ikmn_canonical[39] = x31 * (adjugate[0] * x174 + adjugate[1] * x183 + adjugate[2] * x180);
    S_ikmn_canonical[40] = x31 * (adjugate[3] * x174 + adjugate[4] * x183 + adjugate[5] * x180);
    S_ikmn_canonical[41] = x31 * (adjugate[6] * x174 + adjugate[7] * x183 + adjugate[8] * x180);
    S_ikmn_canonical[42] = x31 * (adjugate[3] * x185 + adjugate[4] * x189 + adjugate[5] * x188);
    S_ikmn_canonical[43] = x31 * (adjugate[6] * x185 + adjugate[7] * x189 + adjugate[8] * x188);
    S_ikmn_canonical[44] = x31 * (adjugate[6] * (adjugate[6] * x173 - adjugate[7] * x184 + x167 * x175) +
                                  adjugate[7] * (-adjugate[6] * x184 + adjugate[7] * x182 - x167 * x177) +
                                  adjugate[8] * (adjugate[6] * x186 - adjugate[7] * x187 + adjugate[8] * x179));
}

static SFEM_INLINE void hex8_apply_S_ikmn(const scalar_t                      qx,
                                          const scalar_t                      qy,
                                          const scalar_t                      qz,
                                          const scalar_t *const SFEM_RESTRICT S_ikmn_canonical,  // 3x3x3x3, includes dV
                                          const scalar_t *const SFEM_RESTRICT inc_grad,          // 3x3 reference trial gradient R
                                          scalar_t *const SFEM_RESTRICT       eoutx,
                                          scalar_t *const SFEM_RESTRICT       eouty,
                                          scalar_t *const SFEM_RESTRICT       eoutz) {
    // mundane ops: 294 divs: 0 sqrts: 0
    // total ops: 294
    const scalar_t x0 = qx - 1;
    const scalar_t x1 = qy - 1;
    const scalar_t x2 = S_ikmn_canonical[11] * inc_grad[4] + S_ikmn_canonical[14] * inc_grad[5] +
                        S_ikmn_canonical[17] * inc_grad[6] + S_ikmn_canonical[20] * inc_grad[7] +
                        S_ikmn_canonical[23] * inc_grad[8] + S_ikmn_canonical[2] * inc_grad[0] +
                        S_ikmn_canonical[4] * inc_grad[1] + S_ikmn_canonical[5] * inc_grad[2] + S_ikmn_canonical[8] * inc_grad[3];
    const scalar_t x3 = x1 * x2;
    const scalar_t x4 = x0 * x3;
    const scalar_t x5 = qz - 1;
    const scalar_t x6 = S_ikmn_canonical[10] * inc_grad[4] + S_ikmn_canonical[13] * inc_grad[5] +
                        S_ikmn_canonical[16] * inc_grad[6] + S_ikmn_canonical[19] * inc_grad[7] +
                        S_ikmn_canonical[1] * inc_grad[0] + S_ikmn_canonical[22] * inc_grad[8] +
                        S_ikmn_canonical[3] * inc_grad[1] + S_ikmn_canonical[4] * inc_grad[2] + S_ikmn_canonical[7] * inc_grad[3];
    const scalar_t x7 = x5 * x6;
    const scalar_t x8 = x0 * x7;
    const scalar_t x9 = S_ikmn_canonical[0] * inc_grad[0] + S_ikmn_canonical[12] * inc_grad[5] +
                        S_ikmn_canonical[15] * inc_grad[6] + S_ikmn_canonical[18] * inc_grad[7] +
                        S_ikmn_canonical[1] * inc_grad[1] + S_ikmn_canonical[21] * inc_grad[8] +
                        S_ikmn_canonical[2] * inc_grad[2] + S_ikmn_canonical[6] * inc_grad[3] + S_ikmn_canonical[9] * inc_grad[4];
    const scalar_t x10 = x5 * x9;
    const scalar_t x11 = x1 * x10;
    const scalar_t x12 = qx * x3;
    const scalar_t x13 = qx * x7;
    const scalar_t x14 = qy * x2;
    const scalar_t x15 = qx * x14;
    const scalar_t x16 = qy * x10;
    const scalar_t x17 = x0 * x14;
    const scalar_t x18 = qz * x6;
    const scalar_t x19 = x0 * x18;
    const scalar_t x20 = qz * x9;
    const scalar_t x21 = x1 * x20;
    const scalar_t x22 = qx * x18;
    const scalar_t x23 = qy * x20;
    const scalar_t x24 =
            S_ikmn_canonical[12] * inc_grad[0] + S_ikmn_canonical[13] * inc_grad[1] + S_ikmn_canonical[14] * inc_grad[2] +
            S_ikmn_canonical[26] * inc_grad[3] + S_ikmn_canonical[28] * inc_grad[4] + S_ikmn_canonical[29] * inc_grad[5] +
            S_ikmn_canonical[32] * inc_grad[6] + S_ikmn_canonical[35] * inc_grad[7] + S_ikmn_canonical[38] * inc_grad[8];
    const scalar_t x25 = x1 * x24;
    const scalar_t x26 = x0 * x25;
    const scalar_t x27 =
            S_ikmn_canonical[10] * inc_grad[1] + S_ikmn_canonical[11] * inc_grad[2] + S_ikmn_canonical[25] * inc_grad[3] +
            S_ikmn_canonical[27] * inc_grad[4] + S_ikmn_canonical[28] * inc_grad[5] + S_ikmn_canonical[31] * inc_grad[6] +
            S_ikmn_canonical[34] * inc_grad[7] + S_ikmn_canonical[37] * inc_grad[8] + S_ikmn_canonical[9] * inc_grad[0];
    const scalar_t x28 = x27 * x5;
    const scalar_t x29 = x0 * x28;
    const scalar_t x30 =
            S_ikmn_canonical[24] * inc_grad[3] + S_ikmn_canonical[25] * inc_grad[4] + S_ikmn_canonical[26] * inc_grad[5] +
            S_ikmn_canonical[30] * inc_grad[6] + S_ikmn_canonical[33] * inc_grad[7] + S_ikmn_canonical[36] * inc_grad[8] +
            S_ikmn_canonical[6] * inc_grad[0] + S_ikmn_canonical[7] * inc_grad[1] + S_ikmn_canonical[8] * inc_grad[2];
    const scalar_t x31 = x30 * x5;
    const scalar_t x32 = x1 * x31;
    const scalar_t x33 = qx * x25;
    const scalar_t x34 = qx * x28;
    const scalar_t x35 = qy * x24;
    const scalar_t x36 = qx * x35;
    const scalar_t x37 = qy * x31;
    const scalar_t x38 = x0 * x35;
    const scalar_t x39 = qz * x27;
    const scalar_t x40 = x0 * x39;
    const scalar_t x41 = qz * x30;
    const scalar_t x42 = x1 * x41;
    const scalar_t x43 = qx * x39;
    const scalar_t x44 = qy * x41;
    const scalar_t x45 =
            S_ikmn_canonical[21] * inc_grad[0] + S_ikmn_canonical[22] * inc_grad[1] + S_ikmn_canonical[23] * inc_grad[2] +
            S_ikmn_canonical[36] * inc_grad[3] + S_ikmn_canonical[37] * inc_grad[4] + S_ikmn_canonical[38] * inc_grad[5] +
            S_ikmn_canonical[41] * inc_grad[6] + S_ikmn_canonical[43] * inc_grad[7] + S_ikmn_canonical[44] * inc_grad[8];
    const scalar_t x46 = x1 * x45;
    const scalar_t x47 = x0 * x46;
    const scalar_t x48 =
            S_ikmn_canonical[18] * inc_grad[0] + S_ikmn_canonical[19] * inc_grad[1] + S_ikmn_canonical[20] * inc_grad[2] +
            S_ikmn_canonical[33] * inc_grad[3] + S_ikmn_canonical[34] * inc_grad[4] + S_ikmn_canonical[35] * inc_grad[5] +
            S_ikmn_canonical[40] * inc_grad[6] + S_ikmn_canonical[42] * inc_grad[7] + S_ikmn_canonical[43] * inc_grad[8];
    const scalar_t x49 = x48 * x5;
    const scalar_t x50 = x0 * x49;
    const scalar_t x51 =
            S_ikmn_canonical[15] * inc_grad[0] + S_ikmn_canonical[16] * inc_grad[1] + S_ikmn_canonical[17] * inc_grad[2] +
            S_ikmn_canonical[30] * inc_grad[3] + S_ikmn_canonical[31] * inc_grad[4] + S_ikmn_canonical[32] * inc_grad[5] +
            S_ikmn_canonical[39] * inc_grad[6] + S_ikmn_canonical[40] * inc_grad[7] + S_ikmn_canonical[41] * inc_grad[8];
    const scalar_t x52 = x5 * x51;
    const scalar_t x53 = x1 * x52;
    const scalar_t x54 = qx * x46;
    const scalar_t x55 = qx * x49;
    const scalar_t x56 = qy * x45;
    const scalar_t x57 = qx * x56;
    const scalar_t x58 = qy * x52;
    const scalar_t x59 = x0 * x56;
    const scalar_t x60 = qz * x48;
    const scalar_t x61 = x0 * x60;
    const scalar_t x62 = qz * x51;
    const scalar_t x63 = x1 * x62;
    const scalar_t x64 = qx * x60;
    const scalar_t x65 = qy * x62;
    eoutx[0]           = -x11 - x4 - x8;
    eoutx[1]           = x11 + x12 + x13;
    eoutx[2]           = -x13 - x15 - x16;
    eoutx[3]           = x16 + x17 + x8;
    eoutx[4]           = x19 + x21 + x4;
    eoutx[5]           = -x12 - x21 - x22;
    eoutx[6]           = x15 + x22 + x23;
    eoutx[7]           = -x17 - x19 - x23;
    eouty[0]           = -x26 - x29 - x32;
    eouty[1]           = x32 + x33 + x34;
    eouty[2]           = -x34 - x36 - x37;
    eouty[3]           = x29 + x37 + x38;
    eouty[4]           = x26 + x40 + x42;
    eouty[5]           = -x33 - x42 - x43;
    eouty[6]           = x36 + x43 + x44;
    eouty[7]           = -x38 - x40 - x44;
    eoutz[0]           = -x47 - x50 - x53;
    eoutz[1]           = x53 + x54 + x55;
    eoutz[2]           = -x55 - x57 - x58;
    eoutz[3]           = x50 + x58 + x59;
    eoutz[4]           = x47 + x61 + x63;
    eoutz[5]           = -x54 - x63 - x64;
    eoutz[6]           = x57 + x64 + x65;
    eoutz[7]           = -x59 - x61 - x65;
}

#endif /* SFEM_HEX8_PARTIAL_ASSEMBLY_NEOHOOKEAN_INLINE_H */
