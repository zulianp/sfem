#ifndef SFEM_TET10_PARTIAL_ASSEMBLY_NEOHOOKEAN_OGDEN_INLINE_H
#define SFEM_TET10_PARTIAL_ASSEMBLY_NEOHOOKEAN_OGDEN_INLINE_H

#include "sfem_macros.h"

static SFEM_INLINE void tet10_F(const scalar_t *const SFEM_RESTRICT adjugate,
                                const scalar_t                      jacobian_determinant,
                                const scalar_t                      qx,
                                const scalar_t                      qy,
                                const scalar_t                      qz,
                                const scalar_t *const SFEM_RESTRICT dispx,
                                const scalar_t *const SFEM_RESTRICT dispy,
                                const scalar_t *const SFEM_RESTRICT dispz,
                                scalar_t *const SFEM_RESTRICT       F) {
    // mundane ops: 206 divs: 1 sqrts: 0
    // total ops: 214
    const scalar_t x0  = 1.0 / jacobian_determinant;
    const scalar_t x1  = 4 * qx;
    const scalar_t x2  = x1 - 1;
    const scalar_t x3  = 4 * qy;
    const scalar_t x4  = -dispx[6] * x3;
    const scalar_t x5  = qz - 1;
    const scalar_t x6  = 8 * qx + 4 * qy + 4 * x5;
    const scalar_t x7  = 4 * qz;
    const scalar_t x8  = x1 + x3 + x7 - 3;
    const scalar_t x9  = dispx[0] * x8;
    const scalar_t x10 = -dispx[7] * x7 + x9;
    const scalar_t x11 = dispx[1] * x2 - dispx[4] * x6 + dispx[5] * x3 + dispx[8] * x7 + x10 + x4;
    const scalar_t x12 = x3 - 1;
    const scalar_t x13 = -dispx[4] * x1;
    const scalar_t x14 = 4 * qx + 8 * qy + 4 * x5;
    const scalar_t x15 = dispx[2] * x12 + dispx[5] * x1 - dispx[6] * x14 + dispx[9] * x7 + x10 + x13;
    const scalar_t x16 = x7 - 1;
    const scalar_t x17 = 4 * qx + 4 * qy + 8 * qz - 4;
    const scalar_t x18 = dispx[3] * x16 - dispx[7] * x17 + dispx[8] * x1 + dispx[9] * x3 + x13 + x4 + x9;
    const scalar_t x19 = -dispy[6] * x3;
    const scalar_t x20 = dispy[0] * x8;
    const scalar_t x21 = -dispy[7] * x7 + x20;
    const scalar_t x22 = dispy[1] * x2 - dispy[4] * x6 + dispy[5] * x3 + dispy[8] * x7 + x19 + x21;
    const scalar_t x23 = -dispy[4] * x1;
    const scalar_t x24 = dispy[2] * x12 + dispy[5] * x1 - dispy[6] * x14 + dispy[9] * x7 + x21 + x23;
    const scalar_t x25 = dispy[3] * x16 - dispy[7] * x17 + dispy[8] * x1 + dispy[9] * x3 + x19 + x20 + x23;
    const scalar_t x26 = -dispz[6] * x3;
    const scalar_t x27 = dispz[0] * x8;
    const scalar_t x28 = -dispz[7] * x7 + x27;
    const scalar_t x29 = dispz[1] * x2 - dispz[4] * x6 + dispz[5] * x3 + dispz[8] * x7 + x26 + x28;
    const scalar_t x30 = -dispz[4] * x1;
    const scalar_t x31 = dispz[2] * x12 + dispz[5] * x1 - dispz[6] * x14 + dispz[9] * x7 + x28 + x30;
    const scalar_t x32 = dispz[3] * x16 - dispz[7] * x17 + dispz[8] * x1 + dispz[9] * x3 + x26 + x27 + x30;
    F[0]               = adjugate[0] * x0 * x11 + adjugate[3] * x0 * x15 + adjugate[6] * x0 * x18 + 1;
    F[1]               = x0 * (adjugate[1] * x11 + adjugate[4] * x15 + adjugate[7] * x18);
    F[2]               = x0 * (adjugate[2] * x11 + adjugate[5] * x15 + adjugate[8] * x18);
    F[3]               = x0 * (adjugate[0] * x22 + adjugate[3] * x24 + adjugate[6] * x25);
    F[4]               = adjugate[1] * x0 * x22 + adjugate[4] * x0 * x24 + adjugate[7] * x0 * x25 + 1;
    F[5]               = x0 * (adjugate[2] * x22 + adjugate[5] * x24 + adjugate[8] * x25);
    F[6]               = x0 * (adjugate[0] * x29 + adjugate[3] * x31 + adjugate[6] * x32);
    F[7]               = x0 * (adjugate[1] * x29 + adjugate[4] * x31 + adjugate[7] * x32);
    F[8]               = adjugate[2] * x0 * x29 + adjugate[5] * x0 * x31 + adjugate[8] * x0 * x32 + 1;
}

static SFEM_INLINE void tet10_Wimpn_compressed(scalar_t *const SFEM_RESTRICT Wimpn_compressed) {
    // mundane ops: 0 divs: 0 sqrts: 0
    // total ops: 0
    Wimpn_compressed[0] = 3.0 / 5.0;
    Wimpn_compressed[1] = 1.0 / 5.0;
    Wimpn_compressed[2] = 0;
    Wimpn_compressed[3] = -4.0 / 5.0;
    Wimpn_compressed[4] = -1.0 / 5.0;
    Wimpn_compressed[5] = -3.0 / 5.0;
    Wimpn_compressed[6] = 8.0 / 5.0;
    Wimpn_compressed[7] = 4.0 / 5.0;
    Wimpn_compressed[8] = -8.0 / 5.0;
}

static SFEM_INLINE void tet10_SdotHdotG(const scalar_t *const SFEM_RESTRICT S_ikmn_canonical,
                                        const scalar_t *const SFEM_RESTRICT Wimpn_compressed,
                                        const scalar_t *const SFEM_RESTRICT incx,
                                        const scalar_t *const SFEM_RESTRICT incy,
                                        const scalar_t *const SFEM_RESTRICT incz,
                                        scalar_t *const SFEM_RESTRICT       outx,
                                        scalar_t *const SFEM_RESTRICT       outy,
                                        scalar_t *const SFEM_RESTRICT       outz) {
    // mundane ops: 2060 divs: 0 sqrts: 0
    // total ops: 2060
    const scalar_t x0   = Wimpn_compressed[1] * incx[1];
    const scalar_t x1   = Wimpn_compressed[0] * incx[0];
    const scalar_t x2   = Wimpn_compressed[1] * incx[5] + Wimpn_compressed[4] * incx[7] + x1;
    const scalar_t x3   = Wimpn_compressed[1] * incx[8] + Wimpn_compressed[4] * incx[6];
    const scalar_t x4   = Wimpn_compressed[2] * incx[3];
    const scalar_t x5   = Wimpn_compressed[3] * incx[4];
    const scalar_t x6   = x4 + x5;
    const scalar_t x7   = Wimpn_compressed[2] * incx[2];
    const scalar_t x8   = Wimpn_compressed[2] * incx[9];
    const scalar_t x9   = x7 + x8;
    const scalar_t x10  = x6 + x9;
    const scalar_t x11  = x0 + x10 + x2 + x3;
    const scalar_t x12  = Wimpn_compressed[1] * incy[2];
    const scalar_t x13  = Wimpn_compressed[0] * incy[0];
    const scalar_t x14  = Wimpn_compressed[1] * incy[9] + Wimpn_compressed[4] * incy[4] + x13;
    const scalar_t x15  = Wimpn_compressed[1] * incy[5] + Wimpn_compressed[4] * incy[7];
    const scalar_t x16  = Wimpn_compressed[3] * incy[6];
    const scalar_t x17  = Wimpn_compressed[2] * incy[8];
    const scalar_t x18  = Wimpn_compressed[2] * incy[1];
    const scalar_t x19  = Wimpn_compressed[2] * incy[3];
    const scalar_t x20  = x18 + x19;
    const scalar_t x21  = x17 + x20;
    const scalar_t x22  = x16 + x21;
    const scalar_t x23  = x12 + x14 + x15 + x22;
    const scalar_t x24  = Wimpn_compressed[1] * incy[3];
    const scalar_t x25  = Wimpn_compressed[1] * incy[8] + Wimpn_compressed[4] * incy[6];
    const scalar_t x26  = Wimpn_compressed[3] * incy[7];
    const scalar_t x27  = Wimpn_compressed[2] * incy[2];
    const scalar_t x28  = Wimpn_compressed[2] * incy[5];
    const scalar_t x29  = x27 + x28;
    const scalar_t x30  = x18 + x29;
    const scalar_t x31  = x26 + x30;
    const scalar_t x32  = x14 + x24 + x25 + x31;
    const scalar_t x33  = Wimpn_compressed[1] * incz[1];
    const scalar_t x34  = Wimpn_compressed[0] * incz[0];
    const scalar_t x35  = Wimpn_compressed[1] * incz[5] + Wimpn_compressed[4] * incz[7] + x34;
    const scalar_t x36  = Wimpn_compressed[1] * incz[8] + Wimpn_compressed[4] * incz[6];
    const scalar_t x37  = Wimpn_compressed[2] * incz[3];
    const scalar_t x38  = Wimpn_compressed[3] * incz[4];
    const scalar_t x39  = x37 + x38;
    const scalar_t x40  = Wimpn_compressed[2] * incz[2];
    const scalar_t x41  = Wimpn_compressed[2] * incz[9];
    const scalar_t x42  = x40 + x41;
    const scalar_t x43  = x39 + x42;
    const scalar_t x44  = x33 + x35 + x36 + x43;
    const scalar_t x45  = Wimpn_compressed[1] * incz[2];
    const scalar_t x46  = Wimpn_compressed[1] * incz[9] + Wimpn_compressed[4] * incz[4];
    const scalar_t x47  = Wimpn_compressed[2] * incz[8];
    const scalar_t x48  = x37 + x47;
    const scalar_t x49  = Wimpn_compressed[2] * incz[1];
    const scalar_t x50  = Wimpn_compressed[3] * incz[6];
    const scalar_t x51  = x49 + x50;
    const scalar_t x52  = x48 + x51;
    const scalar_t x53  = x35 + x45 + x46 + x52;
    const scalar_t x54  = Wimpn_compressed[1] * incx[2];
    const scalar_t x55  = Wimpn_compressed[1] * incx[9] + Wimpn_compressed[4] * incx[4];
    const scalar_t x56  = Wimpn_compressed[2] * incx[8];
    const scalar_t x57  = x4 + x56;
    const scalar_t x58  = Wimpn_compressed[2] * incx[1];
    const scalar_t x59  = Wimpn_compressed[3] * incx[6];
    const scalar_t x60  = x58 + x59;
    const scalar_t x61  = x57 + x60;
    const scalar_t x62  = x2 + x54 + x55 + x61;
    const scalar_t x63  = Wimpn_compressed[1] * incz[3];
    const scalar_t x64  = Wimpn_compressed[3] * incz[7];
    const scalar_t x65  = Wimpn_compressed[2] * incz[5];
    const scalar_t x66  = x49 + x65;
    const scalar_t x67  = x40 + x66;
    const scalar_t x68  = x64 + x67;
    const scalar_t x69  = x34 + x36 + x46 + x63 + x68;
    const scalar_t x70  = Wimpn_compressed[1] * incx[3];
    const scalar_t x71  = Wimpn_compressed[3] * incx[7];
    const scalar_t x72  = Wimpn_compressed[2] * incx[5];
    const scalar_t x73  = x58 + x72;
    const scalar_t x74  = x7 + x73;
    const scalar_t x75  = x71 + x74;
    const scalar_t x76  = x1 + x3 + x55 + x70 + x75;
    const scalar_t x77  = Wimpn_compressed[1] * incy[1];
    const scalar_t x78  = Wimpn_compressed[3] * incy[4];
    const scalar_t x79  = Wimpn_compressed[2] * incy[9];
    const scalar_t x80  = x19 + x79;
    const scalar_t x81  = x27 + x80;
    const scalar_t x82  = x78 + x81;
    const scalar_t x83  = x13 + x15 + x25 + x77 + x82;
    const scalar_t x84  = Wimpn_compressed[4] * incx[8];
    const scalar_t x85  = Wimpn_compressed[1] * incx[0];
    const scalar_t x86  = Wimpn_compressed[1] * incx[7] + x85;
    const scalar_t x87  = x84 + x86;
    const scalar_t x88  = Wimpn_compressed[1] * incx[6];
    const scalar_t x89  = Wimpn_compressed[4] * incx[5];
    const scalar_t x90  = x88 + x89;
    const scalar_t x91  = Wimpn_compressed[0] * incx[1] + x10;
    const scalar_t x92  = x87 + x90 + x91;
    const scalar_t x93  = Wimpn_compressed[4] * incy[9];
    const scalar_t x94  = Wimpn_compressed[5] * incy[4] + x93;
    const scalar_t x95  = Wimpn_compressed[1] * incy[0];
    const scalar_t x96  = Wimpn_compressed[1] * incy[6] + x95;
    const scalar_t x97  = Wimpn_compressed[0] * incy[8] + x96;
    const scalar_t x98  = Wimpn_compressed[2] * incy[7];
    const scalar_t x99  = x30 + x98;
    const scalar_t x100 = Wimpn_compressed[4] * incy[3] + x99;
    const scalar_t x101 = x100 + x94 + x97;
    const scalar_t x102 = Wimpn_compressed[4] * incz[8];
    const scalar_t x103 = Wimpn_compressed[1] * incz[0];
    const scalar_t x104 = Wimpn_compressed[1] * incz[7] + x103;
    const scalar_t x105 = x102 + x104;
    const scalar_t x106 = Wimpn_compressed[1] * incz[6];
    const scalar_t x107 = Wimpn_compressed[4] * incz[5];
    const scalar_t x108 = x106 + x107;
    const scalar_t x109 = Wimpn_compressed[0] * incz[1] + x43;
    const scalar_t x110 = x105 + x108 + x109;
    const scalar_t x111 = Wimpn_compressed[0] * incz[5];
    const scalar_t x112 = Wimpn_compressed[4] * incz[9];
    const scalar_t x113 = Wimpn_compressed[5] * incz[4] + x112;
    const scalar_t x114 = Wimpn_compressed[4] * incz[2];
    const scalar_t x115 = Wimpn_compressed[2] * incz[6];
    const scalar_t x116 = x115 + x48;
    const scalar_t x117 = x116 + x49;
    const scalar_t x118 = x114 + x117;
    const scalar_t x119 = x104 + x111 + x113 + x118;
    const scalar_t x120 = Wimpn_compressed[0] * incx[5];
    const scalar_t x121 = Wimpn_compressed[4] * incx[9];
    const scalar_t x122 = Wimpn_compressed[5] * incx[4] + x121;
    const scalar_t x123 = Wimpn_compressed[4] * incx[2];
    const scalar_t x124 = Wimpn_compressed[2] * incx[6];
    const scalar_t x125 = x124 + x57;
    const scalar_t x126 = x125 + x58;
    const scalar_t x127 = x123 + x126;
    const scalar_t x128 = x120 + x122 + x127 + x86;
    const scalar_t x129 = Wimpn_compressed[0] * incz[8] + x103;
    const scalar_t x130 = Wimpn_compressed[4] * incz[3];
    const scalar_t x131 = Wimpn_compressed[2] * incz[7];
    const scalar_t x132 = x131 + x67;
    const scalar_t x133 = x130 + x132;
    const scalar_t x134 = x106 + x113 + x129 + x133;
    const scalar_t x135 = Wimpn_compressed[0] * incx[8] + x85;
    const scalar_t x136 = Wimpn_compressed[4] * incx[3];
    const scalar_t x137 = Wimpn_compressed[2] * incx[7];
    const scalar_t x138 = x137 + x74;
    const scalar_t x139 = x136 + x138;
    const scalar_t x140 = x122 + x135 + x139 + x88;
    const scalar_t x141 = Wimpn_compressed[1] * incy[7];
    const scalar_t x142 = Wimpn_compressed[4] * incy[5];
    const scalar_t x143 = x141 + x142;
    const scalar_t x144 = Wimpn_compressed[4] * incy[8];
    const scalar_t x145 = x144 + x96;
    const scalar_t x146 = Wimpn_compressed[0] * incy[1] + x82;
    const scalar_t x147 = x143 + x145 + x146;
    const scalar_t x148 = Wimpn_compressed[0] * incy[5] + x141 + x95;
    const scalar_t x149 = Wimpn_compressed[2] * incy[6];
    const scalar_t x150 = x149 + x21;
    const scalar_t x151 = Wimpn_compressed[4] * incy[2] + x150;
    const scalar_t x152 = x148 + x151 + x94;
    const scalar_t x153 = Wimpn_compressed[2] *
                          (incx[0] + incx[1] + incx[2] + incx[3] + incx[4] + incx[5] + incx[6] + incx[7] + incx[8] + incx[9]);
    const scalar_t x154 = S_ikmn_canonical[4] * x153;
    const scalar_t x155 = S_ikmn_canonical[1] * x153;
    const scalar_t x156 = S_ikmn_canonical[2] * x153;
    const scalar_t x157 = x155 + x156;
    const scalar_t x158 = Wimpn_compressed[2] *
                          (incy[0] + incy[1] + incy[2] + incy[3] + incy[4] + incy[5] + incy[6] + incy[7] + incy[8] + incy[9]);
    const scalar_t x159 = Wimpn_compressed[2] *
                          (incz[0] + incz[1] + incz[2] + incz[3] + incz[4] + incz[5] + incz[6] + incz[7] + incz[8] + incz[9]);
    const scalar_t x160 = S_ikmn_canonical[11] * x158 + S_ikmn_canonical[14] * x158 + S_ikmn_canonical[17] * x159 +
                          S_ikmn_canonical[20] * x159 + S_ikmn_canonical[23] * x159 + S_ikmn_canonical[5] * x153 +
                          S_ikmn_canonical[8] * x158;
    const scalar_t x161 = S_ikmn_canonical[10] * x158 + S_ikmn_canonical[13] * x158 + S_ikmn_canonical[16] * x159 +
                          S_ikmn_canonical[19] * x159 + S_ikmn_canonical[22] * x159 + S_ikmn_canonical[3] * x153 +
                          S_ikmn_canonical[7] * x158;
    const scalar_t x162 = Wimpn_compressed[1] * incy[4];
    const scalar_t x163 = x162 + x95;
    const scalar_t x164 = Wimpn_compressed[0] * incy[2] + x22;
    const scalar_t x165 = x143 + x163 + x164 + x93;
    const scalar_t x166 = Wimpn_compressed[5] * incy[6] + x144;
    const scalar_t x167 = Wimpn_compressed[0] * incy[9] + x163;
    const scalar_t x168 = x100 + x166 + x167;
    const scalar_t x169 = Wimpn_compressed[5] * incz[6];
    const scalar_t x170 = Wimpn_compressed[2] * incz[4];
    const scalar_t x171 = x37 + x42;
    const scalar_t x172 = x170 + x171;
    const scalar_t x173 = Wimpn_compressed[4] * incz[1] + x172;
    const scalar_t x174 = x105 + x111 + x169 + x173;
    const scalar_t x175 = Wimpn_compressed[0] * incz[2];
    const scalar_t x176 = Wimpn_compressed[1] * incz[4];
    const scalar_t x177 = x107 + x176;
    const scalar_t x178 = x104 + x112 + x175 + x177 + x52;
    const scalar_t x179 = Wimpn_compressed[5] * incx[6];
    const scalar_t x180 = Wimpn_compressed[2] * incx[4];
    const scalar_t x181 = x4 + x9;
    const scalar_t x182 = x180 + x181;
    const scalar_t x183 = Wimpn_compressed[4] * incx[1] + x182;
    const scalar_t x184 = x120 + x179 + x183 + x87;
    const scalar_t x185 = Wimpn_compressed[0] * incz[9] + x103;
    const scalar_t x186 = x102 + x176;
    const scalar_t x187 = x133 + x169 + x185 + x186;
    const scalar_t x188 = Wimpn_compressed[0] * incx[2];
    const scalar_t x189 = Wimpn_compressed[1] * incx[4];
    const scalar_t x190 = x189 + x89;
    const scalar_t x191 = x121 + x188 + x190 + x61 + x86;
    const scalar_t x192 = Wimpn_compressed[0] * incx[9] + x85;
    const scalar_t x193 = x189 + x84;
    const scalar_t x194 = x139 + x179 + x192 + x193;
    const scalar_t x195 = Wimpn_compressed[2] * incy[4];
    const scalar_t x196 = x195 + x81;
    const scalar_t x197 = Wimpn_compressed[4] * incy[1] + x196;
    const scalar_t x198 = x148 + x166 + x197;
    const scalar_t x199 = x154 + x160;
    const scalar_t x200 = S_ikmn_canonical[0] * x153 + S_ikmn_canonical[12] * x158 + S_ikmn_canonical[15] * x159 +
                          S_ikmn_canonical[18] * x159 + S_ikmn_canonical[21] * x159 + S_ikmn_canonical[6] * x158 +
                          S_ikmn_canonical[9] * x158;
    const scalar_t x201 = Wimpn_compressed[5] * incy[7] + x142;
    const scalar_t x202 = x151 + x167 + x201;
    const scalar_t x203 = Wimpn_compressed[0] * incy[3] + x31;
    const scalar_t x204 = x145 + x162 + x203 + x93;
    const scalar_t x205 = Wimpn_compressed[5] * incz[7];
    const scalar_t x206 = x108 + x129 + x173 + x205;
    const scalar_t x207 = x118 + x177 + x185 + x205;
    const scalar_t x208 = Wimpn_compressed[0] * incz[3] + x103;
    const scalar_t x209 = x106 + x112 + x186 + x208 + x68;
    const scalar_t x210 = Wimpn_compressed[5] * incx[7];
    const scalar_t x211 = x135 + x183 + x210 + x90;
    const scalar_t x212 = x127 + x190 + x192 + x210;
    const scalar_t x213 = Wimpn_compressed[0] * incx[3] + x85;
    const scalar_t x214 = x121 + x193 + x213 + x75 + x88;
    const scalar_t x215 = x197 + x201 + x97;
    const scalar_t x216 = x154 + x161;
    const scalar_t x217 = Wimpn_compressed[6] * incx[4];
    const scalar_t x218 = x137 + x72;
    const scalar_t x219 = Wimpn_compressed[3] * incx[0];
    const scalar_t x220 = x219 + x9;
    const scalar_t x221 = Wimpn_compressed[3] * incx[1] + x125 + x217 + x218 + x220;
    const scalar_t x222 = Wimpn_compressed[3] * incy[9];
    const scalar_t x223 = Wimpn_compressed[6] * incy[4];
    const scalar_t x224 = x222 + x223;
    const scalar_t x225 = Wimpn_compressed[4] * incy[0];
    const scalar_t x226 = Wimpn_compressed[7] * incy[7];
    const scalar_t x227 = x225 + x226;
    const scalar_t x228 = Wimpn_compressed[8] * incy[5] + x227;
    const scalar_t x229 = x12 + x150;
    const scalar_t x230 = x224 + x228 + x229;
    const scalar_t x231 = Wimpn_compressed[3] * incy[8];
    const scalar_t x232 = Wimpn_compressed[7] * incy[4];
    const scalar_t x233 = x231 + x232;
    const scalar_t x234 = Wimpn_compressed[3] * incy[0];
    const scalar_t x235 = x234 + x79;
    const scalar_t x236 = x149 + x29;
    const scalar_t x237 = x20 + x226 + x233 + x235 + x236;
    const scalar_t x238 = x225 + x24 + x99;
    const scalar_t x239 = Wimpn_compressed[7] * incy[6];
    const scalar_t x240 = Wimpn_compressed[8] * incy[8] + x239;
    const scalar_t x241 = x224 + x238 + x240;
    const scalar_t x242 = Wimpn_compressed[6] * incz[4];
    const scalar_t x243 = x131 + x65;
    const scalar_t x244 = Wimpn_compressed[3] * incz[0];
    const scalar_t x245 = x244 + x42;
    const scalar_t x246 = Wimpn_compressed[3] * incz[1] + x116 + x242 + x243 + x245;
    const scalar_t x247 = Wimpn_compressed[3] * incz[5];
    const scalar_t x248 = Wimpn_compressed[7] * incz[4];
    const scalar_t x249 = Wimpn_compressed[7] * incz[6];
    const scalar_t x250 = x247 + x248 + x249;
    const scalar_t x251 = Wimpn_compressed[3] * incz[8];
    const scalar_t x252 = Wimpn_compressed[4] * incz[0];
    const scalar_t x253 = Wimpn_compressed[7] * incz[7];
    const scalar_t x254 = x252 + x253;
    const scalar_t x255 = x251 + x254;
    const scalar_t x256 = Wimpn_compressed[5] * incz[1] + x171 + x250 + x255;
    const scalar_t x257 = x131 + x48;
    const scalar_t x258 = x250 + x49;
    const scalar_t x259 = x245 + x257 + x258;
    const scalar_t x260 = Wimpn_compressed[8] * incz[5];
    const scalar_t x261 = Wimpn_compressed[3] * incz[9];
    const scalar_t x262 = x242 + x261;
    const scalar_t x263 = x117 + x45;
    const scalar_t x264 = x254 + x260 + x262 + x263;
    const scalar_t x265 = x137 + x57;
    const scalar_t x266 = Wimpn_compressed[7] * incx[4];
    const scalar_t x267 = Wimpn_compressed[3] * incx[5];
    const scalar_t x268 = Wimpn_compressed[7] * incx[6];
    const scalar_t x269 = x267 + x268;
    const scalar_t x270 = x266 + x269;
    const scalar_t x271 = x270 + x58;
    const scalar_t x272 = x220 + x265 + x271;
    const scalar_t x273 = S_ikmn_canonical[1] * x272;
    const scalar_t x274 = Wimpn_compressed[4] * incx[0];
    const scalar_t x275 = Wimpn_compressed[7] * incx[7];
    const scalar_t x276 = x274 + x275;
    const scalar_t x277 = Wimpn_compressed[3] * incx[8];
    const scalar_t x278 = x181 + x277;
    const scalar_t x279 = Wimpn_compressed[5] * incx[1] + x270 + x276 + x278;
    const scalar_t x280 = x244 + x66;
    const scalar_t x281 = x248 + x251;
    const scalar_t x282 = x115 + x171 + x253 + x280 + x281;
    const scalar_t x283 = x132 + x252 + x63;
    const scalar_t x284 = Wimpn_compressed[8] * incz[8] + x249;
    const scalar_t x285 = x262 + x283 + x284;
    const scalar_t x286 = x219 + x73;
    const scalar_t x287 = x124 + x266 + x275 + x278 + x286;
    const scalar_t x288 = S_ikmn_canonical[2] * x287;
    const scalar_t x289 = Wimpn_compressed[3] * incx[9];
    const scalar_t x290 = x217 + x289;
    const scalar_t x291 = Wimpn_compressed[8] * incx[5] + x276;
    const scalar_t x292 = x126 + x54;
    const scalar_t x293 = x290 + x291 + x292;
    const scalar_t x294 = Wimpn_compressed[8] * incx[8] + x274;
    const scalar_t x295 = x138 + x70;
    const scalar_t x296 = x268 + x290 + x294 + x295;
    const scalar_t x297 = Wimpn_compressed[3] * incy[1] + x17 + x223 + x234 + x236 + x80 + x98;
    const scalar_t x298 = Wimpn_compressed[3] * incy[5];
    const scalar_t x299 = x239 + x298;
    const scalar_t x300 = x227 + x299;
    const scalar_t x301 = Wimpn_compressed[5] * incy[1] + x233 + x300 + x81;
    const scalar_t x302 = x21 + x232;
    const scalar_t x303 = x235 + x98;
    const scalar_t x304 = x27 + x299 + x302 + x303;
    const scalar_t x305 = Wimpn_compressed[6] * incx[5];
    const scalar_t x306 = Wimpn_compressed[8] * incx[6];
    const scalar_t x307 = Wimpn_compressed[7] * incx[8];
    const scalar_t x308 = x71 + x85;
    const scalar_t x309 = x307 + x308;
    const scalar_t x310 = x183 + x305 + x306 + x309;
    const scalar_t x311 = Wimpn_compressed[7] * incy[9];
    const scalar_t x312 = Wimpn_compressed[8] * incy[4] + x311;
    const scalar_t x313 = x26 + x95;
    const scalar_t x314 = Wimpn_compressed[6] * incy[5] + x313;
    const scalar_t x315 = x151 + x312 + x314;
    const scalar_t x316 = x100 + x95;
    const scalar_t x317 = Wimpn_compressed[7] * incy[8];
    const scalar_t x318 = Wimpn_compressed[8] * incy[6] + x317;
    const scalar_t x319 = Wimpn_compressed[6] * incy[9] + x78;
    const scalar_t x320 = x316 + x318 + x319;
    const scalar_t x321 = Wimpn_compressed[6] * incy[8] + x16;
    const scalar_t x322 = x312 + x316 + x321;
    const scalar_t x323 = Wimpn_compressed[6] * incz[5];
    const scalar_t x324 = Wimpn_compressed[8] * incz[6];
    const scalar_t x325 = Wimpn_compressed[7] * incz[8];
    const scalar_t x326 = x103 + x64;
    const scalar_t x327 = x325 + x326;
    const scalar_t x328 = x173 + x323 + x324 + x327;
    const scalar_t x329 = Wimpn_compressed[7] * incz[5];
    const scalar_t x330 = x329 + x50;
    const scalar_t x331 = x109 + x327 + x330;
    const scalar_t x332 = Wimpn_compressed[7] * incz[9];
    const scalar_t x333 = x326 + x332;
    const scalar_t x334 = x329 + x39 + x47;
    const scalar_t x335 = x175 + x333 + x334 + x51;
    const scalar_t x336 = Wimpn_compressed[8] * incz[4];
    const scalar_t x337 = x118 + x323 + x333 + x336;
    const scalar_t x338 = Wimpn_compressed[7] * incx[5];
    const scalar_t x339 = x338 + x59;
    const scalar_t x340 = x309 + x339 + x91;
    const scalar_t x341 = Wimpn_compressed[7] * incx[9];
    const scalar_t x342 = x308 + x341;
    const scalar_t x343 = x338 + x56 + x6;
    const scalar_t x344 = x188 + x342 + x343 + x60;
    const scalar_t x345 = Wimpn_compressed[6] * incz[9] + x103;
    const scalar_t x346 = x325 + x38;
    const scalar_t x347 = x133 + x324 + x345 + x346;
    const scalar_t x348 = Wimpn_compressed[6] * incz[8] + x103;
    const scalar_t x349 = x332 + x40 + x51;
    const scalar_t x350 = x130 + x243 + x336 + x348 + x349;
    const scalar_t x351 = Wimpn_compressed[6] * incx[9] + x85;
    const scalar_t x352 = x307 + x5;
    const scalar_t x353 = x139 + x306 + x351 + x352;
    const scalar_t x354 = Wimpn_compressed[8] * incx[4];
    const scalar_t x355 = x127 + x305 + x342 + x354;
    const scalar_t x356 = Wimpn_compressed[6] * incx[8] + x85;
    const scalar_t x357 = x341 + x60 + x7;
    const scalar_t x358 = x136 + x218 + x354 + x356 + x357;
    const scalar_t x359 = x197 + x314 + x318;
    const scalar_t x360 = Wimpn_compressed[7] * incy[5];
    const scalar_t x361 = x313 + x360;
    const scalar_t x362 = x16 + x317;
    const scalar_t x363 = x146 + x361 + x362;
    const scalar_t x364 = x311 + x78;
    const scalar_t x365 = x164 + x361 + x364;
    const scalar_t x366 = Wimpn_compressed[6] * incx[6];
    const scalar_t x367 = x277 + x366;
    const scalar_t x368 = x0 + x182;
    const scalar_t x369 = x291 + x367 + x368;
    const scalar_t x370 = Wimpn_compressed[6] * incy[6];
    const scalar_t x371 = x195 + x21;
    const scalar_t x372 = Wimpn_compressed[3] * incy[2] + x28 + x303 + x370 + x371;
    const scalar_t x373 = Wimpn_compressed[5] * incy[2] + x222 + x300 + x302;
    const scalar_t x374 = Wimpn_compressed[8] * incy[9];
    const scalar_t x375 = x233 + x238 + x370 + x374;
    const scalar_t x376 = x222 + x239;
    const scalar_t x377 = x226 + x234 + x29 + x371 + x376;
    const scalar_t x378 = Wimpn_compressed[6] * incz[6];
    const scalar_t x379 = x172 + x33;
    const scalar_t x380 = x255 + x260 + x378 + x379;
    const scalar_t x381 = x261 + x48;
    const scalar_t x382 = Wimpn_compressed[5] * incz[2] + x254 + x258 + x381;
    const scalar_t x383 = Wimpn_compressed[3] * incz[2] + x170 + x257 + x280 + x378 + x41;
    const scalar_t x384 = x289 + x57;
    const scalar_t x385 = Wimpn_compressed[5] * incx[2] + x271 + x276 + x384;
    const scalar_t x386 = Wimpn_compressed[8] * incz[9];
    const scalar_t x387 = x281 + x283 + x378 + x386;
    const scalar_t x388 = x249 + x67;
    const scalar_t x389 = x170 + x244 + x253 + x381 + x388;
    const scalar_t x390 = Wimpn_compressed[8] * incx[9] + x266 + x274;
    const scalar_t x391 = x295 + x367 + x390;
    const scalar_t x392 = Wimpn_compressed[3] * incx[2] + x180 + x265 + x286 + x366 + x8;
    const scalar_t x393 = x268 + x74;
    const scalar_t x394 = x180 + x219 + x275 + x384 + x393;
    const scalar_t x395 = S_ikmn_canonical[4] * x394;
    const scalar_t x396 = x196 + x77;
    const scalar_t x397 = x228 + x231 + x370 + x396;
    const scalar_t x398 = Wimpn_compressed[6] * incx[7];
    const scalar_t x399 = x269 + x294 + x368 + x398;
    const scalar_t x400 = Wimpn_compressed[6] * incy[7];
    const scalar_t x401 = x225 + x298 + x400;
    const scalar_t x402 = x229 + x232 + x374 + x401;
    const scalar_t x403 = Wimpn_compressed[5] * incy[3] + x227 + x233 + x30 + x376;
    const scalar_t x404 = Wimpn_compressed[3] * incy[3] + x149 + x17 + x195 + x235 + x30 + x400;
    const scalar_t x405 = Wimpn_compressed[6] * incz[7];
    const scalar_t x406 = x247 + x252 + x405;
    const scalar_t x407 = x284 + x379 + x406;
    const scalar_t x408 = x248 + x263 + x386 + x406;
    const scalar_t x409 = x267 + x292 + x390 + x398;
    const scalar_t x410 = Wimpn_compressed[5] * incz[3] + x248 + x255 + x261 + x388;
    const scalar_t x411 = Wimpn_compressed[3] * incz[3] + x115 + x170 + x245 + x405 + x47 + x66;
    const scalar_t x412 = Wimpn_compressed[5] * incx[3] + x266 + x276 + x277 + x289 + x393;
    const scalar_t x413 = Wimpn_compressed[3] * incx[3] + x124 + x180 + x220 + x398 + x56 + x73;
    const scalar_t x414 = x240 + x396 + x401;
    const scalar_t x415 = Wimpn_compressed[8] * incx[7];
    const scalar_t x416 = x183 + x339 + x356 + x415;
    const scalar_t x417 = x203 + x362 + x364 + x95;
    const scalar_t x418 = Wimpn_compressed[8] * incz[7];
    const scalar_t x419 = x173 + x330 + x348 + x418;
    const scalar_t x420 = x114 + x115 + x334 + x345 + x418 + x49;
    const scalar_t x421 = x123 + x124 + x343 + x351 + x415 + x58;
    const scalar_t x422 = x208 + x346 + x349 + x64 + x65;
    const scalar_t x423 = x213 + x352 + x357 + x71 + x72;
    const scalar_t x424 = Wimpn_compressed[8] * incy[7] + x360 + x95;
    const scalar_t x425 = x197 + x321 + x424;
    const scalar_t x426 = x151 + x319 + x424;
    const scalar_t x427 = S_ikmn_canonical[28] * x158;
    const scalar_t x428 = S_ikmn_canonical[25] * x158;
    const scalar_t x429 = S_ikmn_canonical[26] * x158;
    const scalar_t x430 = x428 + x429;
    const scalar_t x431 = S_ikmn_canonical[12] * x153 + S_ikmn_canonical[13] * x153 + S_ikmn_canonical[14] * x153 +
                          S_ikmn_canonical[29] * x158 + S_ikmn_canonical[32] * x159 + S_ikmn_canonical[35] * x159 +
                          S_ikmn_canonical[38] * x159;
    const scalar_t x432 = S_ikmn_canonical[10] * x153 + S_ikmn_canonical[11] * x153 + S_ikmn_canonical[27] * x158 +
                          S_ikmn_canonical[31] * x159 + S_ikmn_canonical[34] * x159 + S_ikmn_canonical[37] * x159 +
                          S_ikmn_canonical[9] * x153;
    const scalar_t x433 = x427 + x431;
    const scalar_t x434 = S_ikmn_canonical[24] * x158 + S_ikmn_canonical[30] * x159 + S_ikmn_canonical[33] * x159 +
                          S_ikmn_canonical[36] * x159 + S_ikmn_canonical[6] * x153 + S_ikmn_canonical[7] * x153 +
                          S_ikmn_canonical[8] * x153;
    const scalar_t x435 = x427 + x432;
    const scalar_t x436 = S_ikmn_canonical[25] * x304;
    const scalar_t x437 = S_ikmn_canonical[26] * x237;
    const scalar_t x438 = S_ikmn_canonical[28] * x377;
    const scalar_t x439 = S_ikmn_canonical[43] * x159;
    const scalar_t x440 = S_ikmn_canonical[40] * x159;
    const scalar_t x441 = S_ikmn_canonical[41] * x159;
    const scalar_t x442 = x440 + x441;
    const scalar_t x443 = S_ikmn_canonical[21] * x153 + S_ikmn_canonical[22] * x153 + S_ikmn_canonical[23] * x153 +
                          S_ikmn_canonical[36] * x158 + S_ikmn_canonical[37] * x158 + S_ikmn_canonical[38] * x158 +
                          S_ikmn_canonical[44] * x159;
    const scalar_t x444 = S_ikmn_canonical[18] * x153 + S_ikmn_canonical[19] * x153 + S_ikmn_canonical[20] * x153 +
                          S_ikmn_canonical[33] * x158 + S_ikmn_canonical[34] * x158 + S_ikmn_canonical[35] * x158 +
                          S_ikmn_canonical[42] * x159;
    const scalar_t x445 = x439 + x443;
    const scalar_t x446 = S_ikmn_canonical[15] * x153 + S_ikmn_canonical[16] * x153 + S_ikmn_canonical[17] * x153 +
                          S_ikmn_canonical[30] * x158 + S_ikmn_canonical[31] * x158 + S_ikmn_canonical[32] * x158 +
                          S_ikmn_canonical[39] * x159;
    const scalar_t x447 = x439 + x444;
    const scalar_t x448 = S_ikmn_canonical[40] * x259;
    const scalar_t x449 = S_ikmn_canonical[41] * x282;
    const scalar_t x450 = S_ikmn_canonical[43] * x389;
    outx[0] = S_ikmn_canonical[0] * x11 + S_ikmn_canonical[10] * x23 + S_ikmn_canonical[11] * x23 + S_ikmn_canonical[12] * x32 +
              S_ikmn_canonical[13] * x32 + S_ikmn_canonical[14] * x32 + S_ikmn_canonical[15] * x44 + S_ikmn_canonical[16] * x44 +
              S_ikmn_canonical[17] * x44 + S_ikmn_canonical[18] * x53 + S_ikmn_canonical[19] * x53 + S_ikmn_canonical[1] * x11 +
              S_ikmn_canonical[1] * x62 + S_ikmn_canonical[20] * x53 + S_ikmn_canonical[21] * x69 + S_ikmn_canonical[22] * x69 +
              S_ikmn_canonical[23] * x69 + S_ikmn_canonical[2] * x11 + S_ikmn_canonical[2] * x76 + S_ikmn_canonical[3] * x62 +
              S_ikmn_canonical[4] * x62 + S_ikmn_canonical[4] * x76 + S_ikmn_canonical[5] * x76 + S_ikmn_canonical[6] * x83 +
              S_ikmn_canonical[7] * x83 + S_ikmn_canonical[8] * x83 + S_ikmn_canonical[9] * x23;
    outx[1] = S_ikmn_canonical[0] * x92 + S_ikmn_canonical[12] * x101 + S_ikmn_canonical[15] * x110 +
              S_ikmn_canonical[18] * x119 + S_ikmn_canonical[1] * x128 + S_ikmn_canonical[21] * x134 +
              S_ikmn_canonical[2] * x140 + S_ikmn_canonical[6] * x147 + S_ikmn_canonical[9] * x152 + 2 * x154 + x157 + x160 +
              x161;
    outx[2] = S_ikmn_canonical[10] * x165 + S_ikmn_canonical[13] * x168 + S_ikmn_canonical[16] * x174 +
              S_ikmn_canonical[19] * x178 + S_ikmn_canonical[1] * x184 + S_ikmn_canonical[22] * x187 +
              S_ikmn_canonical[3] * x191 + S_ikmn_canonical[4] * x194 + S_ikmn_canonical[7] * x198 + x155 + 2 * x156 + x199 +
              x200;
    outx[3] = S_ikmn_canonical[11] * x202 + S_ikmn_canonical[14] * x204 + S_ikmn_canonical[17] * x206 +
              S_ikmn_canonical[20] * x207 + S_ikmn_canonical[23] * x209 + S_ikmn_canonical[2] * x211 +
              S_ikmn_canonical[4] * x212 + S_ikmn_canonical[5] * x214 + S_ikmn_canonical[8] * x215 + 2 * x155 + x156 + x200 +
              x216;
    outx[4] = S_ikmn_canonical[0] * x221 + S_ikmn_canonical[10] * x230 + S_ikmn_canonical[11] * x230 +
              S_ikmn_canonical[12] * x237 + S_ikmn_canonical[13] * x241 + S_ikmn_canonical[14] * x241 +
              S_ikmn_canonical[15] * x246 + S_ikmn_canonical[16] * x256 + S_ikmn_canonical[17] * x256 +
              S_ikmn_canonical[18] * x259 + S_ikmn_canonical[19] * x264 + S_ikmn_canonical[1] * x279 +
              S_ikmn_canonical[20] * x264 + S_ikmn_canonical[21] * x282 + S_ikmn_canonical[22] * x285 +
              S_ikmn_canonical[23] * x285 + S_ikmn_canonical[2] * x279 + S_ikmn_canonical[3] * x293 + S_ikmn_canonical[4] * x293 +
              S_ikmn_canonical[4] * x296 + S_ikmn_canonical[5] * x296 + S_ikmn_canonical[6] * x297 + S_ikmn_canonical[7] * x301 +
              S_ikmn_canonical[8] * x301 + S_ikmn_canonical[9] * x304 + x273 + x288;
    outx[5] = S_ikmn_canonical[0] * x310 + S_ikmn_canonical[10] * x315 + S_ikmn_canonical[12] * x320 +
              S_ikmn_canonical[13] * x322 + S_ikmn_canonical[15] * x328 + S_ikmn_canonical[16] * x331 +
              S_ikmn_canonical[18] * x335 + S_ikmn_canonical[19] * x337 + S_ikmn_canonical[1] * x340 +
              S_ikmn_canonical[1] * x344 + S_ikmn_canonical[21] * x347 + S_ikmn_canonical[22] * x350 +
              S_ikmn_canonical[2] * x353 + S_ikmn_canonical[3] * x355 + S_ikmn_canonical[4] * x358 + S_ikmn_canonical[6] * x359 +
              S_ikmn_canonical[7] * x363 + S_ikmn_canonical[9] * x365 + x156 + x199;
    outx[6] = S_ikmn_canonical[0] * x369 + S_ikmn_canonical[10] * x372 + S_ikmn_canonical[11] * x373 +
              S_ikmn_canonical[12] * x375 + S_ikmn_canonical[13] * x377 + S_ikmn_canonical[14] * x375 +
              S_ikmn_canonical[15] * x380 + S_ikmn_canonical[16] * x259 + S_ikmn_canonical[17] * x380 +
              S_ikmn_canonical[18] * x382 + S_ikmn_canonical[19] * x383 + S_ikmn_canonical[1] * x385 +
              S_ikmn_canonical[20] * x382 + S_ikmn_canonical[21] * x387 + S_ikmn_canonical[22] * x389 +
              S_ikmn_canonical[23] * x387 + S_ikmn_canonical[2] * x369 + S_ikmn_canonical[2] * x391 + S_ikmn_canonical[3] * x392 +
              S_ikmn_canonical[4] * x385 + S_ikmn_canonical[5] * x391 + S_ikmn_canonical[6] * x397 + S_ikmn_canonical[7] * x304 +
              S_ikmn_canonical[8] * x397 + S_ikmn_canonical[9] * x373 + x273 + x395;
    outx[7] = S_ikmn_canonical[0] * x399 + S_ikmn_canonical[10] * x402 + S_ikmn_canonical[11] * x377 +
              S_ikmn_canonical[12] * x403 + S_ikmn_canonical[13] * x403 + S_ikmn_canonical[14] * x404 +
              S_ikmn_canonical[15] * x407 + S_ikmn_canonical[16] * x407 + S_ikmn_canonical[17] * x282 +
              S_ikmn_canonical[18] * x408 + S_ikmn_canonical[19] * x408 + S_ikmn_canonical[1] * x399 +
              S_ikmn_canonical[1] * x409 + S_ikmn_canonical[20] * x389 + S_ikmn_canonical[21] * x410 +
              S_ikmn_canonical[22] * x410 + S_ikmn_canonical[23] * x411 + S_ikmn_canonical[2] * x412 +
              S_ikmn_canonical[3] * x409 + S_ikmn_canonical[4] * x412 + S_ikmn_canonical[5] * x413 + S_ikmn_canonical[6] * x414 +
              S_ikmn_canonical[7] * x414 + S_ikmn_canonical[8] * x237 + S_ikmn_canonical[9] * x402 + x288 + x395;
    outx[8] = S_ikmn_canonical[0] * x416 + S_ikmn_canonical[11] * x315 + S_ikmn_canonical[12] * x417 +
              S_ikmn_canonical[14] * x322 + S_ikmn_canonical[15] * x419 + S_ikmn_canonical[17] * x331 +
              S_ikmn_canonical[18] * x420 + S_ikmn_canonical[1] * x421 + S_ikmn_canonical[20] * x337 +
              S_ikmn_canonical[21] * x422 + S_ikmn_canonical[23] * x350 + S_ikmn_canonical[2] * x340 +
              S_ikmn_canonical[2] * x423 + S_ikmn_canonical[4] * x355 + S_ikmn_canonical[5] * x358 + S_ikmn_canonical[6] * x425 +
              S_ikmn_canonical[8] * x363 + S_ikmn_canonical[9] * x426 + x155 + x216;
    outx[9] = S_ikmn_canonical[10] * x426 + S_ikmn_canonical[11] * x365 + S_ikmn_canonical[13] * x417 +
              S_ikmn_canonical[14] * x320 + S_ikmn_canonical[16] * x419 + S_ikmn_canonical[17] * x328 +
              S_ikmn_canonical[19] * x420 + S_ikmn_canonical[1] * x416 + S_ikmn_canonical[20] * x335 +
              S_ikmn_canonical[22] * x422 + S_ikmn_canonical[23] * x347 + S_ikmn_canonical[2] * x310 +
              S_ikmn_canonical[3] * x421 + S_ikmn_canonical[4] * x344 + S_ikmn_canonical[4] * x423 + S_ikmn_canonical[5] * x353 +
              S_ikmn_canonical[7] * x425 + S_ikmn_canonical[8] * x359 + x157 + x200;
    outy[0] = S_ikmn_canonical[10] * x62 + S_ikmn_canonical[11] * x76 + S_ikmn_canonical[12] * x11 + S_ikmn_canonical[13] * x62 +
              S_ikmn_canonical[14] * x76 + S_ikmn_canonical[24] * x83 + S_ikmn_canonical[25] * x23 + S_ikmn_canonical[25] * x83 +
              S_ikmn_canonical[26] * x32 + S_ikmn_canonical[26] * x83 + S_ikmn_canonical[27] * x23 + S_ikmn_canonical[28] * x23 +
              S_ikmn_canonical[28] * x32 + S_ikmn_canonical[29] * x32 + S_ikmn_canonical[30] * x44 + S_ikmn_canonical[31] * x44 +
              S_ikmn_canonical[32] * x44 + S_ikmn_canonical[33] * x53 + S_ikmn_canonical[34] * x53 + S_ikmn_canonical[35] * x53 +
              S_ikmn_canonical[36] * x69 + S_ikmn_canonical[37] * x69 + S_ikmn_canonical[38] * x69 + S_ikmn_canonical[6] * x11 +
              S_ikmn_canonical[7] * x62 + S_ikmn_canonical[8] * x76 + S_ikmn_canonical[9] * x11;
    outy[1] = S_ikmn_canonical[24] * x147 + S_ikmn_canonical[25] * x152 + S_ikmn_canonical[26] * x101 +
              S_ikmn_canonical[30] * x110 + S_ikmn_canonical[33] * x119 + S_ikmn_canonical[36] * x134 +
              S_ikmn_canonical[6] * x92 + S_ikmn_canonical[7] * x128 + S_ikmn_canonical[8] * x140 + 2 * x427 + x430 + x431 + x432;
    outy[2] = S_ikmn_canonical[10] * x191 + S_ikmn_canonical[11] * x194 + S_ikmn_canonical[25] * x198 +
              S_ikmn_canonical[27] * x165 + S_ikmn_canonical[28] * x168 + S_ikmn_canonical[31] * x174 +
              S_ikmn_canonical[34] * x178 + S_ikmn_canonical[37] * x187 + S_ikmn_canonical[9] * x184 + x428 + 2 * x429 + x433 +
              x434;
    outy[3] = S_ikmn_canonical[12] * x211 + S_ikmn_canonical[13] * x212 + S_ikmn_canonical[14] * x214 +
              S_ikmn_canonical[26] * x215 + S_ikmn_canonical[28] * x202 + S_ikmn_canonical[29] * x204 +
              S_ikmn_canonical[32] * x206 + S_ikmn_canonical[35] * x207 + S_ikmn_canonical[38] * x209 + 2 * x428 + x429 + x434 +
              x435;
    outy[4] = S_ikmn_canonical[10] * x293 + S_ikmn_canonical[11] * x296 + S_ikmn_canonical[12] * x279 +
              S_ikmn_canonical[13] * x293 + S_ikmn_canonical[14] * x296 + S_ikmn_canonical[24] * x297 +
              S_ikmn_canonical[25] * x301 + S_ikmn_canonical[26] * x301 + S_ikmn_canonical[27] * x230 +
              S_ikmn_canonical[28] * x230 + S_ikmn_canonical[28] * x241 + S_ikmn_canonical[29] * x241 +
              S_ikmn_canonical[30] * x246 + S_ikmn_canonical[31] * x256 + S_ikmn_canonical[32] * x256 +
              S_ikmn_canonical[33] * x259 + S_ikmn_canonical[34] * x264 + S_ikmn_canonical[35] * x264 +
              S_ikmn_canonical[36] * x282 + S_ikmn_canonical[37] * x285 + S_ikmn_canonical[38] * x285 +
              S_ikmn_canonical[6] * x221 + S_ikmn_canonical[7] * x272 + S_ikmn_canonical[8] * x287 + S_ikmn_canonical[9] * x279 +
              x436 + x437;
    outy[5] = S_ikmn_canonical[10] * x355 + S_ikmn_canonical[11] * x358 + S_ikmn_canonical[24] * x359 +
              S_ikmn_canonical[25] * x363 + S_ikmn_canonical[25] * x365 + S_ikmn_canonical[26] * x320 +
              S_ikmn_canonical[27] * x315 + S_ikmn_canonical[28] * x322 + S_ikmn_canonical[30] * x328 +
              S_ikmn_canonical[31] * x331 + S_ikmn_canonical[33] * x335 + S_ikmn_canonical[34] * x337 +
              S_ikmn_canonical[36] * x347 + S_ikmn_canonical[37] * x350 + S_ikmn_canonical[6] * x310 +
              S_ikmn_canonical[7] * x344 + S_ikmn_canonical[8] * x353 + S_ikmn_canonical[9] * x340 + x429 + x433;
    outy[6] = S_ikmn_canonical[10] * x392 + S_ikmn_canonical[11] * x394 + S_ikmn_canonical[12] * x369 +
              S_ikmn_canonical[13] * x385 + S_ikmn_canonical[14] * x391 + S_ikmn_canonical[24] * x397 +
              S_ikmn_canonical[25] * x373 + S_ikmn_canonical[26] * x375 + S_ikmn_canonical[26] * x397 +
              S_ikmn_canonical[27] * x372 + S_ikmn_canonical[28] * x373 + S_ikmn_canonical[29] * x375 +
              S_ikmn_canonical[30] * x380 + S_ikmn_canonical[31] * x259 + S_ikmn_canonical[32] * x380 +
              S_ikmn_canonical[33] * x382 + S_ikmn_canonical[34] * x383 + S_ikmn_canonical[35] * x382 +
              S_ikmn_canonical[36] * x387 + S_ikmn_canonical[37] * x389 + S_ikmn_canonical[38] * x387 +
              S_ikmn_canonical[6] * x369 + S_ikmn_canonical[7] * x385 + S_ikmn_canonical[8] * x391 + S_ikmn_canonical[9] * x272 +
              x436 + x438;
    outy[7] = S_ikmn_canonical[10] * x409 + S_ikmn_canonical[11] * x412 + S_ikmn_canonical[12] * x287 +
              S_ikmn_canonical[13] * x394 + S_ikmn_canonical[14] * x413 + S_ikmn_canonical[24] * x414 +
              S_ikmn_canonical[25] * x402 + S_ikmn_canonical[25] * x414 + S_ikmn_canonical[26] * x403 +
              S_ikmn_canonical[27] * x402 + S_ikmn_canonical[28] * x403 + S_ikmn_canonical[29] * x404 +
              S_ikmn_canonical[30] * x407 + S_ikmn_canonical[31] * x407 + S_ikmn_canonical[32] * x282 +
              S_ikmn_canonical[33] * x408 + S_ikmn_canonical[34] * x408 + S_ikmn_canonical[35] * x389 +
              S_ikmn_canonical[36] * x410 + S_ikmn_canonical[37] * x410 + S_ikmn_canonical[38] * x411 +
              S_ikmn_canonical[6] * x399 + S_ikmn_canonical[7] * x409 + S_ikmn_canonical[8] * x412 + S_ikmn_canonical[9] * x399 +
              x437 + x438;
    outy[8] = S_ikmn_canonical[12] * x340 + S_ikmn_canonical[13] * x355 + S_ikmn_canonical[14] * x358 +
              S_ikmn_canonical[24] * x425 + S_ikmn_canonical[25] * x426 + S_ikmn_canonical[26] * x363 +
              S_ikmn_canonical[26] * x417 + S_ikmn_canonical[28] * x315 + S_ikmn_canonical[29] * x322 +
              S_ikmn_canonical[30] * x419 + S_ikmn_canonical[32] * x331 + S_ikmn_canonical[33] * x420 +
              S_ikmn_canonical[35] * x337 + S_ikmn_canonical[36] * x422 + S_ikmn_canonical[38] * x350 +
              S_ikmn_canonical[6] * x416 + S_ikmn_canonical[7] * x421 + S_ikmn_canonical[8] * x423 + x428 + x435;
    outy[9] = S_ikmn_canonical[10] * x421 + S_ikmn_canonical[11] * x423 + S_ikmn_canonical[12] * x310 +
              S_ikmn_canonical[13] * x344 + S_ikmn_canonical[14] * x353 + S_ikmn_canonical[25] * x425 +
              S_ikmn_canonical[26] * x359 + S_ikmn_canonical[27] * x426 + S_ikmn_canonical[28] * x365 +
              S_ikmn_canonical[28] * x417 + S_ikmn_canonical[29] * x320 + S_ikmn_canonical[31] * x419 +
              S_ikmn_canonical[32] * x328 + S_ikmn_canonical[34] * x420 + S_ikmn_canonical[35] * x335 +
              S_ikmn_canonical[37] * x422 + S_ikmn_canonical[38] * x347 + S_ikmn_canonical[9] * x416 + x430 + x434;
    outz[0] = S_ikmn_canonical[15] * x11 + S_ikmn_canonical[16] * x62 + S_ikmn_canonical[17] * x76 + S_ikmn_canonical[18] * x11 +
              S_ikmn_canonical[19] * x62 + S_ikmn_canonical[20] * x76 + S_ikmn_canonical[21] * x11 + S_ikmn_canonical[22] * x62 +
              S_ikmn_canonical[23] * x76 + S_ikmn_canonical[30] * x83 + S_ikmn_canonical[31] * x23 + S_ikmn_canonical[32] * x32 +
              S_ikmn_canonical[33] * x83 + S_ikmn_canonical[34] * x23 + S_ikmn_canonical[35] * x32 + S_ikmn_canonical[36] * x83 +
              S_ikmn_canonical[37] * x23 + S_ikmn_canonical[38] * x32 + S_ikmn_canonical[39] * x44 + S_ikmn_canonical[40] * x44 +
              S_ikmn_canonical[40] * x53 + S_ikmn_canonical[41] * x44 + S_ikmn_canonical[41] * x69 + S_ikmn_canonical[42] * x53 +
              S_ikmn_canonical[43] * x53 + S_ikmn_canonical[43] * x69 + S_ikmn_canonical[44] * x69;
    outz[1] = S_ikmn_canonical[15] * x92 + S_ikmn_canonical[16] * x128 + S_ikmn_canonical[17] * x140 +
              S_ikmn_canonical[30] * x147 + S_ikmn_canonical[31] * x152 + S_ikmn_canonical[32] * x101 +
              S_ikmn_canonical[39] * x110 + S_ikmn_canonical[40] * x119 + S_ikmn_canonical[41] * x134 + 2 * x439 + x442 + x443 +
              x444;
    outz[2] = S_ikmn_canonical[18] * x184 + S_ikmn_canonical[19] * x191 + S_ikmn_canonical[20] * x194 +
              S_ikmn_canonical[33] * x198 + S_ikmn_canonical[34] * x165 + S_ikmn_canonical[35] * x168 +
              S_ikmn_canonical[40] * x174 + S_ikmn_canonical[42] * x178 + S_ikmn_canonical[43] * x187 + x440 + 2 * x441 + x445 +
              x446;
    outz[3] = S_ikmn_canonical[21] * x211 + S_ikmn_canonical[22] * x212 + S_ikmn_canonical[23] * x214 +
              S_ikmn_canonical[36] * x215 + S_ikmn_canonical[37] * x202 + S_ikmn_canonical[38] * x204 +
              S_ikmn_canonical[41] * x206 + S_ikmn_canonical[43] * x207 + S_ikmn_canonical[44] * x209 + 2 * x440 + x441 + x446 +
              x447;
    outz[4] = S_ikmn_canonical[15] * x221 + S_ikmn_canonical[16] * x272 + S_ikmn_canonical[17] * x287 +
              S_ikmn_canonical[18] * x279 + S_ikmn_canonical[19] * x293 + S_ikmn_canonical[20] * x296 +
              S_ikmn_canonical[21] * x279 + S_ikmn_canonical[22] * x293 + S_ikmn_canonical[23] * x296 +
              S_ikmn_canonical[30] * x297 + S_ikmn_canonical[31] * x304 + S_ikmn_canonical[32] * x237 +
              S_ikmn_canonical[33] * x301 + S_ikmn_canonical[34] * x230 + S_ikmn_canonical[35] * x241 +
              S_ikmn_canonical[36] * x301 + S_ikmn_canonical[37] * x230 + S_ikmn_canonical[38] * x241 +
              S_ikmn_canonical[39] * x246 + S_ikmn_canonical[40] * x256 + S_ikmn_canonical[41] * x256 +
              S_ikmn_canonical[42] * x264 + S_ikmn_canonical[43] * x264 + S_ikmn_canonical[43] * x285 +
              S_ikmn_canonical[44] * x285 + x448 + x449;
    outz[5] = S_ikmn_canonical[15] * x310 + S_ikmn_canonical[16] * x344 + S_ikmn_canonical[17] * x353 +
              S_ikmn_canonical[18] * x340 + S_ikmn_canonical[19] * x355 + S_ikmn_canonical[20] * x358 +
              S_ikmn_canonical[30] * x359 + S_ikmn_canonical[31] * x365 + S_ikmn_canonical[32] * x320 +
              S_ikmn_canonical[33] * x363 + S_ikmn_canonical[34] * x315 + S_ikmn_canonical[35] * x322 +
              S_ikmn_canonical[39] * x328 + S_ikmn_canonical[40] * x331 + S_ikmn_canonical[40] * x335 +
              S_ikmn_canonical[41] * x347 + S_ikmn_canonical[42] * x337 + S_ikmn_canonical[43] * x350 + x441 + x445;
    outz[6] = S_ikmn_canonical[15] * x369 + S_ikmn_canonical[16] * x385 + S_ikmn_canonical[17] * x391 +
              S_ikmn_canonical[18] * x272 + S_ikmn_canonical[19] * x392 + S_ikmn_canonical[20] * x394 +
              S_ikmn_canonical[21] * x369 + S_ikmn_canonical[22] * x385 + S_ikmn_canonical[23] * x391 +
              S_ikmn_canonical[30] * x397 + S_ikmn_canonical[31] * x373 + S_ikmn_canonical[32] * x375 +
              S_ikmn_canonical[33] * x304 + S_ikmn_canonical[34] * x372 + S_ikmn_canonical[35] * x377 +
              S_ikmn_canonical[36] * x397 + S_ikmn_canonical[37] * x373 + S_ikmn_canonical[38] * x375 +
              S_ikmn_canonical[39] * x380 + S_ikmn_canonical[40] * x382 + S_ikmn_canonical[41] * x380 +
              S_ikmn_canonical[41] * x387 + S_ikmn_canonical[42] * x383 + S_ikmn_canonical[43] * x382 +
              S_ikmn_canonical[44] * x387 + x448 + x450;
    outz[7] = S_ikmn_canonical[15] * x399 + S_ikmn_canonical[16] * x409 + S_ikmn_canonical[17] * x412 +
              S_ikmn_canonical[18] * x399 + S_ikmn_canonical[19] * x409 + S_ikmn_canonical[20] * x412 +
              S_ikmn_canonical[21] * x287 + S_ikmn_canonical[22] * x394 + S_ikmn_canonical[23] * x413 +
              S_ikmn_canonical[30] * x414 + S_ikmn_canonical[31] * x402 + S_ikmn_canonical[32] * x403 +
              S_ikmn_canonical[33] * x414 + S_ikmn_canonical[34] * x402 + S_ikmn_canonical[35] * x403 +
              S_ikmn_canonical[36] * x237 + S_ikmn_canonical[37] * x377 + S_ikmn_canonical[38] * x404 +
              S_ikmn_canonical[39] * x407 + S_ikmn_canonical[40] * x407 + S_ikmn_canonical[40] * x408 +
              S_ikmn_canonical[41] * x410 + S_ikmn_canonical[42] * x408 + S_ikmn_canonical[43] * x410 +
              S_ikmn_canonical[44] * x411 + x449 + x450;
    outz[8] = S_ikmn_canonical[15] * x416 + S_ikmn_canonical[16] * x421 + S_ikmn_canonical[17] * x423 +
              S_ikmn_canonical[21] * x340 + S_ikmn_canonical[22] * x355 + S_ikmn_canonical[23] * x358 +
              S_ikmn_canonical[30] * x425 + S_ikmn_canonical[31] * x426 + S_ikmn_canonical[32] * x417 +
              S_ikmn_canonical[36] * x363 + S_ikmn_canonical[37] * x315 + S_ikmn_canonical[38] * x322 +
              S_ikmn_canonical[39] * x419 + S_ikmn_canonical[40] * x420 + S_ikmn_canonical[41] * x331 +
              S_ikmn_canonical[41] * x422 + S_ikmn_canonical[43] * x337 + S_ikmn_canonical[44] * x350 + x440 + x447;
    outz[9] = S_ikmn_canonical[18] * x416 + S_ikmn_canonical[19] * x421 + S_ikmn_canonical[20] * x423 +
              S_ikmn_canonical[21] * x310 + S_ikmn_canonical[22] * x344 + S_ikmn_canonical[23] * x353 +
              S_ikmn_canonical[33] * x425 + S_ikmn_canonical[34] * x426 + S_ikmn_canonical[35] * x417 +
              S_ikmn_canonical[36] * x359 + S_ikmn_canonical[37] * x365 + S_ikmn_canonical[38] * x320 +
              S_ikmn_canonical[40] * x419 + S_ikmn_canonical[41] * x328 + S_ikmn_canonical[42] * x420 +
              S_ikmn_canonical[43] * x335 + S_ikmn_canonical[43] * x422 + S_ikmn_canonical[44] * x347 + x442 + x446;
}

static SFEM_INLINE void tet10_ref_inc_grad(const scalar_t                      qx,
                                           const scalar_t                      qy,
                                           const scalar_t                      qz,
                                           const scalar_t *const SFEM_RESTRICT incx,
                                           const scalar_t *const SFEM_RESTRICT incy,
                                           const scalar_t *const SFEM_RESTRICT incz,
                                           scalar_t *const SFEM_RESTRICT       inc_grad) {
    // mundane ops: 143 divs: 0 sqrts: 0
    // total ops: 143
    const scalar_t x0  = 4 * qx;
    const scalar_t x1  = x0 - 1;
    const scalar_t x2  = 4 * qy;
    const scalar_t x3  = -incx[6] * x2;
    const scalar_t x4  = qz - 1;
    const scalar_t x5  = 8 * qx + 4 * qy + 4 * x4;
    const scalar_t x6  = 4 * qz;
    const scalar_t x7  = x0 + x2 + x6 - 3;
    const scalar_t x8  = incx[0] * x7;
    const scalar_t x9  = -incx[7] * x6 + x8;
    const scalar_t x10 = x2 - 1;
    const scalar_t x11 = -incx[4] * x0;
    const scalar_t x12 = 4 * qx + 8 * qy + 4 * x4;
    const scalar_t x13 = x6 - 1;
    const scalar_t x14 = 4 * qx + 4 * qy + 8 * qz - 4;
    const scalar_t x15 = -incy[6] * x2;
    const scalar_t x16 = incy[0] * x7;
    const scalar_t x17 = -incy[7] * x6 + x16;
    const scalar_t x18 = -incy[4] * x0;
    const scalar_t x19 = -incz[6] * x2;
    const scalar_t x20 = incz[0] * x7;
    const scalar_t x21 = -incz[7] * x6 + x20;
    const scalar_t x22 = -incz[4] * x0;
    inc_grad[0]        = incx[1] * x1 - incx[4] * x5 + incx[5] * x2 + incx[8] * x6 + x3 + x9;
    inc_grad[1]        = incx[2] * x10 + incx[5] * x0 - incx[6] * x12 + incx[9] * x6 + x11 + x9;
    inc_grad[2]        = incx[3] * x13 - incx[7] * x14 + incx[8] * x0 + incx[9] * x2 + x11 + x3 + x8;
    inc_grad[3]        = incy[1] * x1 - incy[4] * x5 + incy[5] * x2 + incy[8] * x6 + x15 + x17;
    inc_grad[4]        = incy[2] * x10 + incy[5] * x0 - incy[6] * x12 + incy[9] * x6 + x17 + x18;
    inc_grad[5]        = incy[3] * x13 - incy[7] * x14 + incy[8] * x0 + incy[9] * x2 + x15 + x16 + x18;
    inc_grad[6]        = incz[1] * x1 - incz[4] * x5 + incz[5] * x2 + incz[8] * x6 + x19 + x21;
    inc_grad[7]        = incz[2] * x10 + incz[5] * x0 - incz[6] * x12 + incz[9] * x6 + x21 + x22;
    inc_grad[8]        = incz[3] * x13 - incz[7] * x14 + incz[8] * x0 + incz[9] * x2 + x19 + x20 + x22;
}

static SFEM_INLINE void tet10_Zpkmn(const scalar_t *const SFEM_RESTRICT Wimpn_compressed,
                                    const scalar_t *const SFEM_RESTRICT incx,
                                    const scalar_t *const SFEM_RESTRICT incy,
                                    const scalar_t *const SFEM_RESTRICT incz,
                                    scalar_t *const SFEM_RESTRICT       Zpkmn) {
    // mundane ops: 783 divs: 0 sqrts: 0
    // total ops: 783
    const scalar_t x0  = Wimpn_compressed[1] * incx[1];
    const scalar_t x1  = Wimpn_compressed[0] * incx[0];
    const scalar_t x2  = Wimpn_compressed[1] * incx[5] + Wimpn_compressed[4] * incx[7] + x1;
    const scalar_t x3  = Wimpn_compressed[1] * incx[8] + Wimpn_compressed[4] * incx[6];
    const scalar_t x4  = Wimpn_compressed[2] * incx[3];
    const scalar_t x5  = Wimpn_compressed[3] * incx[4];
    const scalar_t x6  = x4 + x5;
    const scalar_t x7  = Wimpn_compressed[2] * incx[2];
    const scalar_t x8  = Wimpn_compressed[2] * incx[9];
    const scalar_t x9  = x7 + x8;
    const scalar_t x10 = x6 + x9;
    const scalar_t x11 = x0 + x10 + x2 + x3;
    const scalar_t x12 = Wimpn_compressed[1] * incx[2];
    const scalar_t x13 = Wimpn_compressed[1] * incx[9] + Wimpn_compressed[4] * incx[4];
    const scalar_t x14 = Wimpn_compressed[2] * incx[8];
    const scalar_t x15 = x14 + x4;
    const scalar_t x16 = Wimpn_compressed[2] * incx[1];
    const scalar_t x17 = Wimpn_compressed[3] * incx[6];
    const scalar_t x18 = x16 + x17;
    const scalar_t x19 = x15 + x18;
    const scalar_t x20 = x12 + x13 + x19 + x2;
    const scalar_t x21 = Wimpn_compressed[1] * incx[3];
    const scalar_t x22 = Wimpn_compressed[3] * incx[7];
    const scalar_t x23 = Wimpn_compressed[2] * incx[5];
    const scalar_t x24 = x16 + x23;
    const scalar_t x25 = x24 + x7;
    const scalar_t x26 = x22 + x25;
    const scalar_t x27 = x1 + x13 + x21 + x26 + x3;
    const scalar_t x28 = Wimpn_compressed[1] * incy[1];
    const scalar_t x29 = Wimpn_compressed[0] * incy[0];
    const scalar_t x30 = Wimpn_compressed[1] * incy[5] + Wimpn_compressed[4] * incy[7] + x29;
    const scalar_t x31 = Wimpn_compressed[1] * incy[8] + Wimpn_compressed[4] * incy[6];
    const scalar_t x32 = Wimpn_compressed[2] * incy[3];
    const scalar_t x33 = Wimpn_compressed[3] * incy[4];
    const scalar_t x34 = x32 + x33;
    const scalar_t x35 = Wimpn_compressed[2] * incy[2];
    const scalar_t x36 = Wimpn_compressed[2] * incy[9];
    const scalar_t x37 = x35 + x36;
    const scalar_t x38 = x34 + x37;
    const scalar_t x39 = x28 + x30 + x31 + x38;
    const scalar_t x40 = Wimpn_compressed[1] * incy[2];
    const scalar_t x41 = Wimpn_compressed[1] * incy[9] + Wimpn_compressed[4] * incy[4];
    const scalar_t x42 = Wimpn_compressed[2] * incy[8];
    const scalar_t x43 = x32 + x42;
    const scalar_t x44 = Wimpn_compressed[2] * incy[1];
    const scalar_t x45 = Wimpn_compressed[3] * incy[6];
    const scalar_t x46 = x44 + x45;
    const scalar_t x47 = x43 + x46;
    const scalar_t x48 = x30 + x40 + x41 + x47;
    const scalar_t x49 = Wimpn_compressed[1] * incy[3];
    const scalar_t x50 = Wimpn_compressed[3] * incy[7];
    const scalar_t x51 = Wimpn_compressed[2] * incy[5];
    const scalar_t x52 = x44 + x51;
    const scalar_t x53 = x35 + x52;
    const scalar_t x54 = x50 + x53;
    const scalar_t x55 = x29 + x31 + x41 + x49 + x54;
    const scalar_t x56 = Wimpn_compressed[1] * incz[1];
    const scalar_t x57 = Wimpn_compressed[0] * incz[0];
    const scalar_t x58 = Wimpn_compressed[1] * incz[5] + Wimpn_compressed[4] * incz[7] + x57;
    const scalar_t x59 = Wimpn_compressed[1] * incz[8] + Wimpn_compressed[4] * incz[6];
    const scalar_t x60 = Wimpn_compressed[2] * incz[3];
    const scalar_t x61 = Wimpn_compressed[3] * incz[4];
    const scalar_t x62 = x60 + x61;
    const scalar_t x63 = Wimpn_compressed[2] * incz[2];
    const scalar_t x64 = Wimpn_compressed[2] * incz[9];
    const scalar_t x65 = x63 + x64;
    const scalar_t x66 = x62 + x65;
    const scalar_t x67 = x56 + x58 + x59 + x66;
    const scalar_t x68 = Wimpn_compressed[1] * incz[2];
    const scalar_t x69 = Wimpn_compressed[1] * incz[9] + Wimpn_compressed[4] * incz[4];
    const scalar_t x70 = Wimpn_compressed[2] * incz[8];
    const scalar_t x71 = x60 + x70;
    const scalar_t x72 = Wimpn_compressed[2] * incz[1];
    const scalar_t x73 = Wimpn_compressed[3] * incz[6];
    const scalar_t x74 = x72 + x73;
    const scalar_t x75 = x71 + x74;
    const scalar_t x76 = x58 + x68 + x69 + x75;
    const scalar_t x77 = Wimpn_compressed[1] * incz[3];
    const scalar_t x78 = Wimpn_compressed[3] * incz[7];
    const scalar_t x79 = Wimpn_compressed[2] * incz[5];
    const scalar_t x80 = x72 + x79;
    const scalar_t x81 = x63 + x80;
    const scalar_t x82 = x78 + x81;
    const scalar_t x83 = x57 + x59 + x69 + x77 + x82;
    const scalar_t x84 = Wimpn_compressed[4] * incx[8];
    const scalar_t x85 = Wimpn_compressed[1] * incx[0];
    const scalar_t x86 = Wimpn_compressed[1] * incx[7] + x85;
    const scalar_t x87 = x84 + x86;
    const scalar_t x88 = Wimpn_compressed[1] * incx[6];
    const scalar_t x89 = Wimpn_compressed[4] * incx[5];
    const scalar_t x90 = x88 + x89;
    const scalar_t x91 = Wimpn_compressed[0] * incx[1] + x10;
    const scalar_t x92 = Wimpn_compressed[2] *
                         (incx[0] + incx[1] + incx[2] + incx[3] + incx[4] + incx[5] + incx[6] + incx[7] + incx[8] + incx[9]);
    const scalar_t x93  = Wimpn_compressed[0] * incx[5];
    const scalar_t x94  = Wimpn_compressed[4] * incx[9];
    const scalar_t x95  = Wimpn_compressed[5] * incx[4] + x94;
    const scalar_t x96  = Wimpn_compressed[4] * incx[2];
    const scalar_t x97  = Wimpn_compressed[2] * incx[6];
    const scalar_t x98  = x15 + x97;
    const scalar_t x99  = x16 + x98;
    const scalar_t x100 = x96 + x99;
    const scalar_t x101 = Wimpn_compressed[0] * incx[8] + x85;
    const scalar_t x102 = Wimpn_compressed[4] * incx[3];
    const scalar_t x103 = Wimpn_compressed[2] * incx[7];
    const scalar_t x104 = x103 + x25;
    const scalar_t x105 = x102 + x104;
    const scalar_t x106 = Wimpn_compressed[4] * incy[8];
    const scalar_t x107 = Wimpn_compressed[1] * incy[0];
    const scalar_t x108 = Wimpn_compressed[1] * incy[7] + x107;
    const scalar_t x109 = x106 + x108;
    const scalar_t x110 = Wimpn_compressed[1] * incy[6];
    const scalar_t x111 = Wimpn_compressed[4] * incy[5];
    const scalar_t x112 = x110 + x111;
    const scalar_t x113 = Wimpn_compressed[0] * incy[1] + x38;
    const scalar_t x114 = Wimpn_compressed[2] *
                          (incy[0] + incy[1] + incy[2] + incy[3] + incy[4] + incy[5] + incy[6] + incy[7] + incy[8] + incy[9]);
    const scalar_t x115 = Wimpn_compressed[0] * incy[5];
    const scalar_t x116 = Wimpn_compressed[4] * incy[9];
    const scalar_t x117 = Wimpn_compressed[5] * incy[4] + x116;
    const scalar_t x118 = Wimpn_compressed[4] * incy[2];
    const scalar_t x119 = Wimpn_compressed[2] * incy[6];
    const scalar_t x120 = x119 + x43;
    const scalar_t x121 = x120 + x44;
    const scalar_t x122 = x118 + x121;
    const scalar_t x123 = Wimpn_compressed[0] * incy[8] + x107;
    const scalar_t x124 = Wimpn_compressed[4] * incy[3];
    const scalar_t x125 = Wimpn_compressed[2] * incy[7];
    const scalar_t x126 = x125 + x53;
    const scalar_t x127 = x124 + x126;
    const scalar_t x128 = Wimpn_compressed[4] * incz[8];
    const scalar_t x129 = Wimpn_compressed[1] * incz[0];
    const scalar_t x130 = Wimpn_compressed[1] * incz[7] + x129;
    const scalar_t x131 = x128 + x130;
    const scalar_t x132 = Wimpn_compressed[1] * incz[6];
    const scalar_t x133 = Wimpn_compressed[4] * incz[5];
    const scalar_t x134 = x132 + x133;
    const scalar_t x135 = Wimpn_compressed[0] * incz[1] + x66;
    const scalar_t x136 = Wimpn_compressed[2] *
                          (incz[0] + incz[1] + incz[2] + incz[3] + incz[4] + incz[5] + incz[6] + incz[7] + incz[8] + incz[9]);
    const scalar_t x137 = Wimpn_compressed[0] * incz[5];
    const scalar_t x138 = Wimpn_compressed[4] * incz[9];
    const scalar_t x139 = Wimpn_compressed[5] * incz[4] + x138;
    const scalar_t x140 = Wimpn_compressed[4] * incz[2];
    const scalar_t x141 = Wimpn_compressed[2] * incz[6];
    const scalar_t x142 = x141 + x71;
    const scalar_t x143 = x142 + x72;
    const scalar_t x144 = x140 + x143;
    const scalar_t x145 = Wimpn_compressed[0] * incz[8] + x129;
    const scalar_t x146 = Wimpn_compressed[4] * incz[3];
    const scalar_t x147 = Wimpn_compressed[2] * incz[7];
    const scalar_t x148 = x147 + x81;
    const scalar_t x149 = x146 + x148;
    const scalar_t x150 = Wimpn_compressed[5] * incx[6];
    const scalar_t x151 = Wimpn_compressed[2] * incx[4];
    const scalar_t x152 = x4 + x9;
    const scalar_t x153 = x151 + x152;
    const scalar_t x154 = Wimpn_compressed[4] * incx[1] + x153;
    const scalar_t x155 = Wimpn_compressed[0] * incx[2];
    const scalar_t x156 = Wimpn_compressed[1] * incx[4];
    const scalar_t x157 = x156 + x89;
    const scalar_t x158 = Wimpn_compressed[0] * incx[9] + x85;
    const scalar_t x159 = x156 + x84;
    const scalar_t x160 = Wimpn_compressed[5] * incy[6];
    const scalar_t x161 = Wimpn_compressed[2] * incy[4];
    const scalar_t x162 = x32 + x37;
    const scalar_t x163 = x161 + x162;
    const scalar_t x164 = Wimpn_compressed[4] * incy[1] + x163;
    const scalar_t x165 = Wimpn_compressed[0] * incy[2];
    const scalar_t x166 = Wimpn_compressed[1] * incy[4];
    const scalar_t x167 = x111 + x166;
    const scalar_t x168 = Wimpn_compressed[0] * incy[9] + x107;
    const scalar_t x169 = x106 + x166;
    const scalar_t x170 = Wimpn_compressed[5] * incz[6];
    const scalar_t x171 = Wimpn_compressed[2] * incz[4];
    const scalar_t x172 = x60 + x65;
    const scalar_t x173 = x171 + x172;
    const scalar_t x174 = Wimpn_compressed[4] * incz[1] + x173;
    const scalar_t x175 = Wimpn_compressed[0] * incz[2];
    const scalar_t x176 = Wimpn_compressed[1] * incz[4];
    const scalar_t x177 = x133 + x176;
    const scalar_t x178 = Wimpn_compressed[0] * incz[9] + x129;
    const scalar_t x179 = x128 + x176;
    const scalar_t x180 = Wimpn_compressed[5] * incx[7];
    const scalar_t x181 = Wimpn_compressed[0] * incx[3] + x85;
    const scalar_t x182 = Wimpn_compressed[5] * incy[7];
    const scalar_t x183 = Wimpn_compressed[0] * incy[3] + x107;
    const scalar_t x184 = Wimpn_compressed[5] * incz[7];
    const scalar_t x185 = Wimpn_compressed[0] * incz[3] + x129;
    const scalar_t x186 = Wimpn_compressed[6] * incx[4];
    const scalar_t x187 = x103 + x23;
    const scalar_t x188 = Wimpn_compressed[3] * incx[0];
    const scalar_t x189 = x188 + x9;
    const scalar_t x190 = Wimpn_compressed[3] * incx[5];
    const scalar_t x191 = Wimpn_compressed[7] * incx[4];
    const scalar_t x192 = Wimpn_compressed[7] * incx[6];
    const scalar_t x193 = x190 + x191 + x192;
    const scalar_t x194 = Wimpn_compressed[3] * incx[8];
    const scalar_t x195 = Wimpn_compressed[4] * incx[0];
    const scalar_t x196 = Wimpn_compressed[7] * incx[7];
    const scalar_t x197 = x195 + x196;
    const scalar_t x198 = x194 + x197;
    const scalar_t x199 = Wimpn_compressed[5] * incx[1] + x152 + x193 + x198;
    const scalar_t x200 = x103 + x15;
    const scalar_t x201 = x16 + x193;
    const scalar_t x202 = x189 + x200 + x201;
    const scalar_t x203 = Wimpn_compressed[8] * incx[5];
    const scalar_t x204 = Wimpn_compressed[3] * incx[9];
    const scalar_t x205 = x186 + x204;
    const scalar_t x206 = x12 + x99;
    const scalar_t x207 = x197 + x203 + x205 + x206;
    const scalar_t x208 = x188 + x24;
    const scalar_t x209 = x191 + x194;
    const scalar_t x210 = x152 + x196 + x208 + x209 + x97;
    const scalar_t x211 = x104 + x195 + x21;
    const scalar_t x212 = Wimpn_compressed[8] * incx[8] + x192;
    const scalar_t x213 = x205 + x211 + x212;
    const scalar_t x214 = Wimpn_compressed[6] * incy[4];
    const scalar_t x215 = x125 + x51;
    const scalar_t x216 = Wimpn_compressed[3] * incy[0];
    const scalar_t x217 = x216 + x37;
    const scalar_t x218 = Wimpn_compressed[3] * incy[5];
    const scalar_t x219 = Wimpn_compressed[7] * incy[4];
    const scalar_t x220 = Wimpn_compressed[7] * incy[6];
    const scalar_t x221 = x218 + x219 + x220;
    const scalar_t x222 = Wimpn_compressed[3] * incy[8];
    const scalar_t x223 = Wimpn_compressed[4] * incy[0];
    const scalar_t x224 = Wimpn_compressed[7] * incy[7];
    const scalar_t x225 = x223 + x224;
    const scalar_t x226 = x222 + x225;
    const scalar_t x227 = Wimpn_compressed[5] * incy[1] + x162 + x221 + x226;
    const scalar_t x228 = x125 + x43;
    const scalar_t x229 = x221 + x44;
    const scalar_t x230 = x217 + x228 + x229;
    const scalar_t x231 = Wimpn_compressed[8] * incy[5];
    const scalar_t x232 = Wimpn_compressed[3] * incy[9];
    const scalar_t x233 = x214 + x232;
    const scalar_t x234 = x121 + x40;
    const scalar_t x235 = x225 + x231 + x233 + x234;
    const scalar_t x236 = x216 + x52;
    const scalar_t x237 = x219 + x222;
    const scalar_t x238 = x119 + x162 + x224 + x236 + x237;
    const scalar_t x239 = x126 + x223 + x49;
    const scalar_t x240 = Wimpn_compressed[8] * incy[8] + x220;
    const scalar_t x241 = x233 + x239 + x240;
    const scalar_t x242 = Wimpn_compressed[6] * incz[4];
    const scalar_t x243 = x147 + x79;
    const scalar_t x244 = Wimpn_compressed[3] * incz[0];
    const scalar_t x245 = x244 + x65;
    const scalar_t x246 = Wimpn_compressed[3] * incz[5];
    const scalar_t x247 = Wimpn_compressed[7] * incz[4];
    const scalar_t x248 = Wimpn_compressed[7] * incz[6];
    const scalar_t x249 = x246 + x247 + x248;
    const scalar_t x250 = Wimpn_compressed[3] * incz[8];
    const scalar_t x251 = Wimpn_compressed[4] * incz[0];
    const scalar_t x252 = Wimpn_compressed[7] * incz[7];
    const scalar_t x253 = x251 + x252;
    const scalar_t x254 = x250 + x253;
    const scalar_t x255 = Wimpn_compressed[5] * incz[1] + x172 + x249 + x254;
    const scalar_t x256 = x147 + x71;
    const scalar_t x257 = x249 + x72;
    const scalar_t x258 = x245 + x256 + x257;
    const scalar_t x259 = Wimpn_compressed[8] * incz[5];
    const scalar_t x260 = Wimpn_compressed[3] * incz[9];
    const scalar_t x261 = x242 + x260;
    const scalar_t x262 = x143 + x68;
    const scalar_t x263 = x253 + x259 + x261 + x262;
    const scalar_t x264 = x244 + x80;
    const scalar_t x265 = x247 + x250;
    const scalar_t x266 = x141 + x172 + x252 + x264 + x265;
    const scalar_t x267 = x148 + x251 + x77;
    const scalar_t x268 = Wimpn_compressed[8] * incz[8] + x248;
    const scalar_t x269 = x261 + x267 + x268;
    const scalar_t x270 = Wimpn_compressed[6] * incx[5];
    const scalar_t x271 = Wimpn_compressed[8] * incx[6];
    const scalar_t x272 = Wimpn_compressed[7] * incx[8];
    const scalar_t x273 = x22 + x85;
    const scalar_t x274 = x272 + x273;
    const scalar_t x275 = x154 + x270 + x271 + x274;
    const scalar_t x276 = Wimpn_compressed[7] * incx[5];
    const scalar_t x277 = x17 + x276;
    const scalar_t x278 = x274 + x277 + x91;
    const scalar_t x279 = Wimpn_compressed[7] * incx[9];
    const scalar_t x280 = x273 + x279;
    const scalar_t x281 = x14 + x276 + x6;
    const scalar_t x282 = x155 + x18 + x280 + x281;
    const scalar_t x283 = Wimpn_compressed[8] * incx[4];
    const scalar_t x284 = x100 + x270 + x280 + x283;
    const scalar_t x285 = Wimpn_compressed[6] * incx[9] + x85;
    const scalar_t x286 = x272 + x5;
    const scalar_t x287 = x105 + x271 + x285 + x286;
    const scalar_t x288 = Wimpn_compressed[6] * incx[8] + x85;
    const scalar_t x289 = x18 + x279 + x7;
    const scalar_t x290 = x102 + x187 + x283 + x288 + x289;
    const scalar_t x291 = Wimpn_compressed[6] * incy[5];
    const scalar_t x292 = Wimpn_compressed[8] * incy[6];
    const scalar_t x293 = Wimpn_compressed[7] * incy[8];
    const scalar_t x294 = x107 + x50;
    const scalar_t x295 = x293 + x294;
    const scalar_t x296 = x164 + x291 + x292 + x295;
    const scalar_t x297 = Wimpn_compressed[7] * incy[5];
    const scalar_t x298 = x297 + x45;
    const scalar_t x299 = x113 + x295 + x298;
    const scalar_t x300 = Wimpn_compressed[7] * incy[9];
    const scalar_t x301 = x294 + x300;
    const scalar_t x302 = x297 + x34 + x42;
    const scalar_t x303 = x165 + x301 + x302 + x46;
    const scalar_t x304 = Wimpn_compressed[8] * incy[4];
    const scalar_t x305 = x122 + x291 + x301 + x304;
    const scalar_t x306 = Wimpn_compressed[6] * incy[9] + x107;
    const scalar_t x307 = x293 + x33;
    const scalar_t x308 = x127 + x292 + x306 + x307;
    const scalar_t x309 = Wimpn_compressed[6] * incy[8] + x107;
    const scalar_t x310 = x300 + x35 + x46;
    const scalar_t x311 = x124 + x215 + x304 + x309 + x310;
    const scalar_t x312 = Wimpn_compressed[6] * incz[5];
    const scalar_t x313 = Wimpn_compressed[8] * incz[6];
    const scalar_t x314 = Wimpn_compressed[7] * incz[8];
    const scalar_t x315 = x129 + x78;
    const scalar_t x316 = x314 + x315;
    const scalar_t x317 = x174 + x312 + x313 + x316;
    const scalar_t x318 = Wimpn_compressed[7] * incz[5];
    const scalar_t x319 = x318 + x73;
    const scalar_t x320 = x135 + x316 + x319;
    const scalar_t x321 = Wimpn_compressed[7] * incz[9];
    const scalar_t x322 = x315 + x321;
    const scalar_t x323 = x318 + x62 + x70;
    const scalar_t x324 = x175 + x322 + x323 + x74;
    const scalar_t x325 = Wimpn_compressed[8] * incz[4];
    const scalar_t x326 = x144 + x312 + x322 + x325;
    const scalar_t x327 = Wimpn_compressed[6] * incz[9] + x129;
    const scalar_t x328 = x314 + x61;
    const scalar_t x329 = x149 + x313 + x327 + x328;
    const scalar_t x330 = Wimpn_compressed[6] * incz[8] + x129;
    const scalar_t x331 = x321 + x63 + x74;
    const scalar_t x332 = x146 + x243 + x325 + x330 + x331;
    const scalar_t x333 = Wimpn_compressed[6] * incx[6];
    const scalar_t x334 = x0 + x153;
    const scalar_t x335 = x198 + x203 + x333 + x334;
    const scalar_t x336 = x15 + x204;
    const scalar_t x337 = Wimpn_compressed[5] * incx[2] + x197 + x201 + x336;
    const scalar_t x338 = Wimpn_compressed[8] * incx[9];
    const scalar_t x339 = x209 + x211 + x333 + x338;
    const scalar_t x340 = x192 + x25;
    const scalar_t x341 = x151 + x188 + x196 + x336 + x340;
    const scalar_t x342 = Wimpn_compressed[6] * incy[6];
    const scalar_t x343 = x163 + x28;
    const scalar_t x344 = x226 + x231 + x342 + x343;
    const scalar_t x345 = x232 + x43;
    const scalar_t x346 = Wimpn_compressed[5] * incy[2] + x225 + x229 + x345;
    const scalar_t x347 = Wimpn_compressed[8] * incy[9];
    const scalar_t x348 = x237 + x239 + x342 + x347;
    const scalar_t x349 = x220 + x53;
    const scalar_t x350 = x161 + x216 + x224 + x345 + x349;
    const scalar_t x351 = Wimpn_compressed[6] * incz[6];
    const scalar_t x352 = x173 + x56;
    const scalar_t x353 = x254 + x259 + x351 + x352;
    const scalar_t x354 = x260 + x71;
    const scalar_t x355 = Wimpn_compressed[5] * incz[2] + x253 + x257 + x354;
    const scalar_t x356 = Wimpn_compressed[8] * incz[9];
    const scalar_t x357 = x265 + x267 + x351 + x356;
    const scalar_t x358 = x248 + x81;
    const scalar_t x359 = x171 + x244 + x252 + x354 + x358;
    const scalar_t x360 = Wimpn_compressed[6] * incx[7];
    const scalar_t x361 = x190 + x195 + x360;
    const scalar_t x362 = x212 + x334 + x361;
    const scalar_t x363 = x191 + x206 + x338 + x361;
    const scalar_t x364 = Wimpn_compressed[5] * incx[3] + x191 + x198 + x204 + x340;
    const scalar_t x365 = Wimpn_compressed[6] * incy[7];
    const scalar_t x366 = x218 + x223 + x365;
    const scalar_t x367 = x240 + x343 + x366;
    const scalar_t x368 = x219 + x234 + x347 + x366;
    const scalar_t x369 = Wimpn_compressed[5] * incy[3] + x219 + x226 + x232 + x349;
    const scalar_t x370 = Wimpn_compressed[6] * incz[7];
    const scalar_t x371 = x246 + x251 + x370;
    const scalar_t x372 = x268 + x352 + x371;
    const scalar_t x373 = x247 + x262 + x356 + x371;
    const scalar_t x374 = Wimpn_compressed[5] * incz[3] + x247 + x254 + x260 + x358;
    const scalar_t x375 = Wimpn_compressed[8] * incx[7];
    const scalar_t x376 = x154 + x277 + x288 + x375;
    const scalar_t x377 = x16 + x281 + x285 + x375 + x96 + x97;
    const scalar_t x378 = x181 + x22 + x23 + x286 + x289;
    const scalar_t x379 = Wimpn_compressed[8] * incy[7];
    const scalar_t x380 = x164 + x298 + x309 + x379;
    const scalar_t x381 = x118 + x119 + x302 + x306 + x379 + x44;
    const scalar_t x382 = x183 + x307 + x310 + x50 + x51;
    const scalar_t x383 = Wimpn_compressed[8] * incz[7];
    const scalar_t x384 = x174 + x319 + x330 + x383;
    const scalar_t x385 = x140 + x141 + x323 + x327 + x383 + x72;
    const scalar_t x386 = x185 + x328 + x331 + x78 + x79;
    Zpkmn[0]            = x11;
    Zpkmn[1]            = x11;
    Zpkmn[2]            = x11;
    Zpkmn[3]            = x20;
    Zpkmn[4]            = x20;
    Zpkmn[5]            = x20;
    Zpkmn[6]            = x27;
    Zpkmn[7]            = x27;
    Zpkmn[8]            = x27;
    Zpkmn[9]            = x39;
    Zpkmn[10]           = x39;
    Zpkmn[11]           = x39;
    Zpkmn[12]           = x48;
    Zpkmn[13]           = x48;
    Zpkmn[14]           = x48;
    Zpkmn[15]           = x55;
    Zpkmn[16]           = x55;
    Zpkmn[17]           = x55;
    Zpkmn[18]           = x67;
    Zpkmn[19]           = x67;
    Zpkmn[20]           = x67;
    Zpkmn[21]           = x76;
    Zpkmn[22]           = x76;
    Zpkmn[23]           = x76;
    Zpkmn[24]           = x83;
    Zpkmn[25]           = x83;
    Zpkmn[26]           = x83;
    Zpkmn[27]           = x87 + x90 + x91;
    Zpkmn[28]           = x92;
    Zpkmn[29]           = x92;
    Zpkmn[30]           = x100 + x86 + x93 + x95;
    Zpkmn[31]           = x92;
    Zpkmn[32]           = x92;
    Zpkmn[33]           = x101 + x105 + x88 + x95;
    Zpkmn[34]           = x92;
    Zpkmn[35]           = x92;
    Zpkmn[36]           = x109 + x112 + x113;
    Zpkmn[37]           = x114;
    Zpkmn[38]           = x114;
    Zpkmn[39]           = x108 + x115 + x117 + x122;
    Zpkmn[40]           = x114;
    Zpkmn[41]           = x114;
    Zpkmn[42]           = x110 + x117 + x123 + x127;
    Zpkmn[43]           = x114;
    Zpkmn[44]           = x114;
    Zpkmn[45]           = x131 + x134 + x135;
    Zpkmn[46]           = x136;
    Zpkmn[47]           = x136;
    Zpkmn[48]           = x130 + x137 + x139 + x144;
    Zpkmn[49]           = x136;
    Zpkmn[50]           = x136;
    Zpkmn[51]           = x132 + x139 + x145 + x149;
    Zpkmn[52]           = x136;
    Zpkmn[53]           = x136;
    Zpkmn[54]           = x92;
    Zpkmn[55]           = x150 + x154 + x87 + x93;
    Zpkmn[56]           = x92;
    Zpkmn[57]           = x92;
    Zpkmn[58]           = x155 + x157 + x19 + x86 + x94;
    Zpkmn[59]           = x92;
    Zpkmn[60]           = x92;
    Zpkmn[61]           = x105 + x150 + x158 + x159;
    Zpkmn[62]           = x92;
    Zpkmn[63]           = x114;
    Zpkmn[64]           = x109 + x115 + x160 + x164;
    Zpkmn[65]           = x114;
    Zpkmn[66]           = x114;
    Zpkmn[67]           = x108 + x116 + x165 + x167 + x47;
    Zpkmn[68]           = x114;
    Zpkmn[69]           = x114;
    Zpkmn[70]           = x127 + x160 + x168 + x169;
    Zpkmn[71]           = x114;
    Zpkmn[72]           = x136;
    Zpkmn[73]           = x131 + x137 + x170 + x174;
    Zpkmn[74]           = x136;
    Zpkmn[75]           = x136;
    Zpkmn[76]           = x130 + x138 + x175 + x177 + x75;
    Zpkmn[77]           = x136;
    Zpkmn[78]           = x136;
    Zpkmn[79]           = x149 + x170 + x178 + x179;
    Zpkmn[80]           = x136;
    Zpkmn[81]           = x92;
    Zpkmn[82]           = x92;
    Zpkmn[83]           = x101 + x154 + x180 + x90;
    Zpkmn[84]           = x92;
    Zpkmn[85]           = x92;
    Zpkmn[86]           = x100 + x157 + x158 + x180;
    Zpkmn[87]           = x92;
    Zpkmn[88]           = x92;
    Zpkmn[89]           = x159 + x181 + x26 + x88 + x94;
    Zpkmn[90]           = x114;
    Zpkmn[91]           = x114;
    Zpkmn[92]           = x112 + x123 + x164 + x182;
    Zpkmn[93]           = x114;
    Zpkmn[94]           = x114;
    Zpkmn[95]           = x122 + x167 + x168 + x182;
    Zpkmn[96]           = x114;
    Zpkmn[97]           = x114;
    Zpkmn[98]           = x110 + x116 + x169 + x183 + x54;
    Zpkmn[99]           = x136;
    Zpkmn[100]          = x136;
    Zpkmn[101]          = x134 + x145 + x174 + x184;
    Zpkmn[102]          = x136;
    Zpkmn[103]          = x136;
    Zpkmn[104]          = x144 + x177 + x178 + x184;
    Zpkmn[105]          = x136;
    Zpkmn[106]          = x136;
    Zpkmn[107]          = x132 + x138 + x179 + x185 + x82;
    Zpkmn[108]          = Wimpn_compressed[3] * incx[1] + x186 + x187 + x189 + x98;
    Zpkmn[109]          = x199;
    Zpkmn[110]          = x199;
    Zpkmn[111]          = x202;
    Zpkmn[112]          = x207;
    Zpkmn[113]          = x207;
    Zpkmn[114]          = x210;
    Zpkmn[115]          = x213;
    Zpkmn[116]          = x213;
    Zpkmn[117]          = Wimpn_compressed[3] * incy[1] + x120 + x214 + x215 + x217;
    Zpkmn[118]          = x227;
    Zpkmn[119]          = x227;
    Zpkmn[120]          = x230;
    Zpkmn[121]          = x235;
    Zpkmn[122]          = x235;
    Zpkmn[123]          = x238;
    Zpkmn[124]          = x241;
    Zpkmn[125]          = x241;
    Zpkmn[126]          = Wimpn_compressed[3] * incz[1] + x142 + x242 + x243 + x245;
    Zpkmn[127]          = x255;
    Zpkmn[128]          = x255;
    Zpkmn[129]          = x258;
    Zpkmn[130]          = x263;
    Zpkmn[131]          = x263;
    Zpkmn[132]          = x266;
    Zpkmn[133]          = x269;
    Zpkmn[134]          = x269;
    Zpkmn[135]          = x275;
    Zpkmn[136]          = x278;
    Zpkmn[137]          = x92;
    Zpkmn[138]          = x282;
    Zpkmn[139]          = x284;
    Zpkmn[140]          = x92;
    Zpkmn[141]          = x287;
    Zpkmn[142]          = x290;
    Zpkmn[143]          = x92;
    Zpkmn[144]          = x296;
    Zpkmn[145]          = x299;
    Zpkmn[146]          = x114;
    Zpkmn[147]          = x303;
    Zpkmn[148]          = x305;
    Zpkmn[149]          = x114;
    Zpkmn[150]          = x308;
    Zpkmn[151]          = x311;
    Zpkmn[152]          = x114;
    Zpkmn[153]          = x317;
    Zpkmn[154]          = x320;
    Zpkmn[155]          = x136;
    Zpkmn[156]          = x324;
    Zpkmn[157]          = x326;
    Zpkmn[158]          = x136;
    Zpkmn[159]          = x329;
    Zpkmn[160]          = x332;
    Zpkmn[161]          = x136;
    Zpkmn[162]          = x335;
    Zpkmn[163]          = x202;
    Zpkmn[164]          = x335;
    Zpkmn[165]          = x337;
    Zpkmn[166]          = Wimpn_compressed[3] * incx[2] + x151 + x200 + x208 + x333 + x8;
    Zpkmn[167]          = x337;
    Zpkmn[168]          = x339;
    Zpkmn[169]          = x341;
    Zpkmn[170]          = x339;
    Zpkmn[171]          = x344;
    Zpkmn[172]          = x230;
    Zpkmn[173]          = x344;
    Zpkmn[174]          = x346;
    Zpkmn[175]          = Wimpn_compressed[3] * incy[2] + x161 + x228 + x236 + x342 + x36;
    Zpkmn[176]          = x346;
    Zpkmn[177]          = x348;
    Zpkmn[178]          = x350;
    Zpkmn[179]          = x348;
    Zpkmn[180]          = x353;
    Zpkmn[181]          = x258;
    Zpkmn[182]          = x353;
    Zpkmn[183]          = x355;
    Zpkmn[184]          = Wimpn_compressed[3] * incz[2] + x171 + x256 + x264 + x351 + x64;
    Zpkmn[185]          = x355;
    Zpkmn[186]          = x357;
    Zpkmn[187]          = x359;
    Zpkmn[188]          = x357;
    Zpkmn[189]          = x362;
    Zpkmn[190]          = x362;
    Zpkmn[191]          = x210;
    Zpkmn[192]          = x363;
    Zpkmn[193]          = x363;
    Zpkmn[194]          = x341;
    Zpkmn[195]          = x364;
    Zpkmn[196]          = x364;
    Zpkmn[197]          = Wimpn_compressed[3] * incx[3] + x14 + x151 + x189 + x24 + x360 + x97;
    Zpkmn[198]          = x367;
    Zpkmn[199]          = x367;
    Zpkmn[200]          = x238;
    Zpkmn[201]          = x368;
    Zpkmn[202]          = x368;
    Zpkmn[203]          = x350;
    Zpkmn[204]          = x369;
    Zpkmn[205]          = x369;
    Zpkmn[206]          = Wimpn_compressed[3] * incy[3] + x119 + x161 + x217 + x365 + x42 + x52;
    Zpkmn[207]          = x372;
    Zpkmn[208]          = x372;
    Zpkmn[209]          = x266;
    Zpkmn[210]          = x373;
    Zpkmn[211]          = x373;
    Zpkmn[212]          = x359;
    Zpkmn[213]          = x374;
    Zpkmn[214]          = x374;
    Zpkmn[215]          = Wimpn_compressed[3] * incz[3] + x141 + x171 + x245 + x370 + x70 + x80;
    Zpkmn[216]          = x376;
    Zpkmn[217]          = x92;
    Zpkmn[218]          = x278;
    Zpkmn[219]          = x377;
    Zpkmn[220]          = x92;
    Zpkmn[221]          = x284;
    Zpkmn[222]          = x378;
    Zpkmn[223]          = x92;
    Zpkmn[224]          = x290;
    Zpkmn[225]          = x380;
    Zpkmn[226]          = x114;
    Zpkmn[227]          = x299;
    Zpkmn[228]          = x381;
    Zpkmn[229]          = x114;
    Zpkmn[230]          = x305;
    Zpkmn[231]          = x382;
    Zpkmn[232]          = x114;
    Zpkmn[233]          = x311;
    Zpkmn[234]          = x384;
    Zpkmn[235]          = x136;
    Zpkmn[236]          = x320;
    Zpkmn[237]          = x385;
    Zpkmn[238]          = x136;
    Zpkmn[239]          = x326;
    Zpkmn[240]          = x386;
    Zpkmn[241]          = x136;
    Zpkmn[242]          = x332;
    Zpkmn[243]          = x92;
    Zpkmn[244]          = x376;
    Zpkmn[245]          = x275;
    Zpkmn[246]          = x92;
    Zpkmn[247]          = x377;
    Zpkmn[248]          = x282;
    Zpkmn[249]          = x92;
    Zpkmn[250]          = x378;
    Zpkmn[251]          = x287;
    Zpkmn[252]          = x114;
    Zpkmn[253]          = x380;
    Zpkmn[254]          = x296;
    Zpkmn[255]          = x114;
    Zpkmn[256]          = x381;
    Zpkmn[257]          = x303;
    Zpkmn[258]          = x114;
    Zpkmn[259]          = x382;
    Zpkmn[260]          = x308;
    Zpkmn[261]          = x136;
    Zpkmn[262]          = x384;
    Zpkmn[263]          = x317;
    Zpkmn[264]          = x136;
    Zpkmn[265]          = x385;
    Zpkmn[266]          = x324;
    Zpkmn[267]          = x136;
    Zpkmn[268]          = x386;
    Zpkmn[269]          = x329;
}

static SFEM_INLINE void tet10_SdotZ(const scalar_t *const SFEM_RESTRICT S_ikmn_canonical,
                                    const scalar_t *const SFEM_RESTRICT Zpkmn,
                                    scalar_t *const SFEM_RESTRICT       outx,
                                    scalar_t *const SFEM_RESTRICT       outy,
                                    scalar_t *const SFEM_RESTRICT       outz) {
    // mundane ops: 1590 divs: 0 sqrts: 0
    // total ops: 1590
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
    outx[8] = S_ikmn_canonical[0] * Zpkmn[216] + S_ikmn_canonical[10] * Zpkmn[229] + S_ikmn_canonical[11] * Zpkmn[230] +
              S_ikmn_canonical[12] * Zpkmn[231] + S_ikmn_canonical[13] * Zpkmn[232] + S_ikmn_canonical[14] * Zpkmn[233] +
              S_ikmn_canonical[15] * Zpkmn[234] + S_ikmn_canonical[16] * Zpkmn[235] + S_ikmn_canonical[17] * Zpkmn[236] +
              S_ikmn_canonical[18] * Zpkmn[237] + S_ikmn_canonical[19] * Zpkmn[238] + S_ikmn_canonical[1] * Zpkmn[217] +
              S_ikmn_canonical[1] * Zpkmn[219] + S_ikmn_canonical[20] * Zpkmn[239] + S_ikmn_canonical[21] * Zpkmn[240] +
              S_ikmn_canonical[22] * Zpkmn[241] + S_ikmn_canonical[23] * Zpkmn[242] + S_ikmn_canonical[2] * Zpkmn[218] +
              S_ikmn_canonical[2] * Zpkmn[222] + S_ikmn_canonical[3] * Zpkmn[220] + S_ikmn_canonical[4] * Zpkmn[221] +
              S_ikmn_canonical[4] * Zpkmn[223] + S_ikmn_canonical[5] * Zpkmn[224] + S_ikmn_canonical[6] * Zpkmn[225] +
              S_ikmn_canonical[7] * Zpkmn[226] + S_ikmn_canonical[8] * Zpkmn[227] + S_ikmn_canonical[9] * Zpkmn[228];
    outx[9] = S_ikmn_canonical[0] * Zpkmn[243] + S_ikmn_canonical[10] * Zpkmn[256] + S_ikmn_canonical[11] * Zpkmn[257] +
              S_ikmn_canonical[12] * Zpkmn[258] + S_ikmn_canonical[13] * Zpkmn[259] + S_ikmn_canonical[14] * Zpkmn[260] +
              S_ikmn_canonical[15] * Zpkmn[261] + S_ikmn_canonical[16] * Zpkmn[262] + S_ikmn_canonical[17] * Zpkmn[263] +
              S_ikmn_canonical[18] * Zpkmn[264] + S_ikmn_canonical[19] * Zpkmn[265] + S_ikmn_canonical[1] * Zpkmn[244] +
              S_ikmn_canonical[1] * Zpkmn[246] + S_ikmn_canonical[20] * Zpkmn[266] + S_ikmn_canonical[21] * Zpkmn[267] +
              S_ikmn_canonical[22] * Zpkmn[268] + S_ikmn_canonical[23] * Zpkmn[269] + S_ikmn_canonical[2] * Zpkmn[245] +
              S_ikmn_canonical[2] * Zpkmn[249] + S_ikmn_canonical[3] * Zpkmn[247] + S_ikmn_canonical[4] * Zpkmn[248] +
              S_ikmn_canonical[4] * Zpkmn[250] + S_ikmn_canonical[5] * Zpkmn[251] + S_ikmn_canonical[6] * Zpkmn[252] +
              S_ikmn_canonical[7] * Zpkmn[253] + S_ikmn_canonical[8] * Zpkmn[254] + S_ikmn_canonical[9] * Zpkmn[255];
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
    outy[8] = S_ikmn_canonical[10] * Zpkmn[220] + S_ikmn_canonical[11] * Zpkmn[223] + S_ikmn_canonical[12] * Zpkmn[218] +
              S_ikmn_canonical[13] * Zpkmn[221] + S_ikmn_canonical[14] * Zpkmn[224] + S_ikmn_canonical[24] * Zpkmn[225] +
              S_ikmn_canonical[25] * Zpkmn[226] + S_ikmn_canonical[25] * Zpkmn[228] + S_ikmn_canonical[26] * Zpkmn[227] +
              S_ikmn_canonical[26] * Zpkmn[231] + S_ikmn_canonical[27] * Zpkmn[229] + S_ikmn_canonical[28] * Zpkmn[230] +
              S_ikmn_canonical[28] * Zpkmn[232] + S_ikmn_canonical[29] * Zpkmn[233] + S_ikmn_canonical[30] * Zpkmn[234] +
              S_ikmn_canonical[31] * Zpkmn[235] + S_ikmn_canonical[32] * Zpkmn[236] + S_ikmn_canonical[33] * Zpkmn[237] +
              S_ikmn_canonical[34] * Zpkmn[238] + S_ikmn_canonical[35] * Zpkmn[239] + S_ikmn_canonical[36] * Zpkmn[240] +
              S_ikmn_canonical[37] * Zpkmn[241] + S_ikmn_canonical[38] * Zpkmn[242] + S_ikmn_canonical[6] * Zpkmn[216] +
              S_ikmn_canonical[7] * Zpkmn[219] + S_ikmn_canonical[8] * Zpkmn[222] + S_ikmn_canonical[9] * Zpkmn[217];
    outy[9] = S_ikmn_canonical[10] * Zpkmn[247] + S_ikmn_canonical[11] * Zpkmn[250] + S_ikmn_canonical[12] * Zpkmn[245] +
              S_ikmn_canonical[13] * Zpkmn[248] + S_ikmn_canonical[14] * Zpkmn[251] + S_ikmn_canonical[24] * Zpkmn[252] +
              S_ikmn_canonical[25] * Zpkmn[253] + S_ikmn_canonical[25] * Zpkmn[255] + S_ikmn_canonical[26] * Zpkmn[254] +
              S_ikmn_canonical[26] * Zpkmn[258] + S_ikmn_canonical[27] * Zpkmn[256] + S_ikmn_canonical[28] * Zpkmn[257] +
              S_ikmn_canonical[28] * Zpkmn[259] + S_ikmn_canonical[29] * Zpkmn[260] + S_ikmn_canonical[30] * Zpkmn[261] +
              S_ikmn_canonical[31] * Zpkmn[262] + S_ikmn_canonical[32] * Zpkmn[263] + S_ikmn_canonical[33] * Zpkmn[264] +
              S_ikmn_canonical[34] * Zpkmn[265] + S_ikmn_canonical[35] * Zpkmn[266] + S_ikmn_canonical[36] * Zpkmn[267] +
              S_ikmn_canonical[37] * Zpkmn[268] + S_ikmn_canonical[38] * Zpkmn[269] + S_ikmn_canonical[6] * Zpkmn[243] +
              S_ikmn_canonical[7] * Zpkmn[246] + S_ikmn_canonical[8] * Zpkmn[249] + S_ikmn_canonical[9] * Zpkmn[244];
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
    outz[8] = S_ikmn_canonical[15] * Zpkmn[216] + S_ikmn_canonical[16] * Zpkmn[219] + S_ikmn_canonical[17] * Zpkmn[222] +
              S_ikmn_canonical[18] * Zpkmn[217] + S_ikmn_canonical[19] * Zpkmn[220] + S_ikmn_canonical[20] * Zpkmn[223] +
              S_ikmn_canonical[21] * Zpkmn[218] + S_ikmn_canonical[22] * Zpkmn[221] + S_ikmn_canonical[23] * Zpkmn[224] +
              S_ikmn_canonical[30] * Zpkmn[225] + S_ikmn_canonical[31] * Zpkmn[228] + S_ikmn_canonical[32] * Zpkmn[231] +
              S_ikmn_canonical[33] * Zpkmn[226] + S_ikmn_canonical[34] * Zpkmn[229] + S_ikmn_canonical[35] * Zpkmn[232] +
              S_ikmn_canonical[36] * Zpkmn[227] + S_ikmn_canonical[37] * Zpkmn[230] + S_ikmn_canonical[38] * Zpkmn[233] +
              S_ikmn_canonical[39] * Zpkmn[234] + S_ikmn_canonical[40] * Zpkmn[235] + S_ikmn_canonical[40] * Zpkmn[237] +
              S_ikmn_canonical[41] * Zpkmn[236] + S_ikmn_canonical[41] * Zpkmn[240] + S_ikmn_canonical[42] * Zpkmn[238] +
              S_ikmn_canonical[43] * Zpkmn[239] + S_ikmn_canonical[43] * Zpkmn[241] + S_ikmn_canonical[44] * Zpkmn[242];
    outz[9] = S_ikmn_canonical[15] * Zpkmn[243] + S_ikmn_canonical[16] * Zpkmn[246] + S_ikmn_canonical[17] * Zpkmn[249] +
              S_ikmn_canonical[18] * Zpkmn[244] + S_ikmn_canonical[19] * Zpkmn[247] + S_ikmn_canonical[20] * Zpkmn[250] +
              S_ikmn_canonical[21] * Zpkmn[245] + S_ikmn_canonical[22] * Zpkmn[248] + S_ikmn_canonical[23] * Zpkmn[251] +
              S_ikmn_canonical[30] * Zpkmn[252] + S_ikmn_canonical[31] * Zpkmn[255] + S_ikmn_canonical[32] * Zpkmn[258] +
              S_ikmn_canonical[33] * Zpkmn[253] + S_ikmn_canonical[34] * Zpkmn[256] + S_ikmn_canonical[35] * Zpkmn[259] +
              S_ikmn_canonical[36] * Zpkmn[254] + S_ikmn_canonical[37] * Zpkmn[257] + S_ikmn_canonical[38] * Zpkmn[260] +
              S_ikmn_canonical[39] * Zpkmn[261] + S_ikmn_canonical[40] * Zpkmn[262] + S_ikmn_canonical[40] * Zpkmn[264] +
              S_ikmn_canonical[41] * Zpkmn[263] + S_ikmn_canonical[41] * Zpkmn[267] + S_ikmn_canonical[42] * Zpkmn[265] +
              S_ikmn_canonical[43] * Zpkmn[266] + S_ikmn_canonical[43] * Zpkmn[268] + S_ikmn_canonical[44] * Zpkmn[269];
}

static SFEM_INLINE void tet10_expand_S(const metric_tensor_t *const SFEM_RESTRICT S_ikmn_canonical,
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

static SFEM_INLINE void tet10_SdotZ_expanded(const scalar_t *const SFEM_RESTRICT S_ikmn,
                                             const scalar_t *const SFEM_RESTRICT Zpkmn,
                                             scalar_t *const SFEM_RESTRICT       outx,
                                             scalar_t *const SFEM_RESTRICT       outy,
                                             scalar_t *const SFEM_RESTRICT       outz)

{
    static const int pstride = 3 * 3 * 3;
    static const int ksize   = 3 * 3 * 3;

    for (int p = 0; p < 10; p++) {
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

static SFEM_INLINE void tet10_neohookean_ogden_hessian_from_S_ikmn(const metric_tensor_t *const SFEM_RESTRICT S_ikmn_canonical,
                                                                   const scalar_t *const SFEM_RESTRICT        Wimpn_compressed,
                                                                   scalar_t *const SFEM_RESTRICT              H) {
    // mundane ops: 2145 divs: 0 sqrts: 0
    // total ops: 2145
    const scalar_t x0    = 2 * S_ikmn_canonical[1];
    const scalar_t x1    = 2 * S_ikmn_canonical[2];
    const scalar_t x2    = 2 * S_ikmn_canonical[4];
    const scalar_t x3    = S_ikmn_canonical[1] * Wimpn_compressed[1];
    const scalar_t x4    = S_ikmn_canonical[2] * Wimpn_compressed[2];
    const scalar_t x5    = S_ikmn_canonical[0] * Wimpn_compressed[1];
    const scalar_t x6    = S_ikmn_canonical[2] * Wimpn_compressed[1];
    const scalar_t x7    = x5 + x6;
    const scalar_t x8    = x4 + x7;
    const scalar_t x9    = S_ikmn_canonical[1] * Wimpn_compressed[2];
    const scalar_t x10   = Wimpn_compressed[2] * x2;
    const scalar_t x11   = S_ikmn_canonical[3] * Wimpn_compressed[2];
    const scalar_t x12   = S_ikmn_canonical[5] * Wimpn_compressed[2];
    const scalar_t x13   = x11 + x12;
    const scalar_t x14   = x10 + x13;
    const scalar_t x15   = x14 + x9;
    const scalar_t x16   = x15 + x3 + x8;
    const scalar_t x17   = S_ikmn_canonical[3] * Wimpn_compressed[1];
    const scalar_t x18   = S_ikmn_canonical[4] * Wimpn_compressed[1];
    const scalar_t x19   = x17 + x18;
    const scalar_t x20   = x12 + x19;
    const scalar_t x21   = Wimpn_compressed[2] * x1;
    const scalar_t x22   = x21 + x9;
    const scalar_t x23   = S_ikmn_canonical[0] * Wimpn_compressed[2];
    const scalar_t x24   = S_ikmn_canonical[4] * Wimpn_compressed[2];
    const scalar_t x25   = x23 + x24;
    const scalar_t x26   = x22 + x25;
    const scalar_t x27   = x20 + x26 + x3;
    const scalar_t x28   = x23 + x6;
    const scalar_t x29   = Wimpn_compressed[2] * x0;
    const scalar_t x30   = x29 + x4;
    const scalar_t x31   = x11 + x30;
    const scalar_t x32   = S_ikmn_canonical[5] * Wimpn_compressed[1];
    const scalar_t x33   = x18 + x32;
    const scalar_t x34   = x24 + x33;
    const scalar_t x35   = x28 + x31 + x34;
    const scalar_t x36   = S_ikmn_canonical[0] * Wimpn_compressed[3];
    const scalar_t x37   = S_ikmn_canonical[1] * Wimpn_compressed[4];
    const scalar_t x38   = S_ikmn_canonical[5] * Wimpn_compressed[4];
    const scalar_t x39   = S_ikmn_canonical[1] * Wimpn_compressed[3];
    const scalar_t x40   = x37 + x38 + x39;
    const scalar_t x41   = S_ikmn_canonical[2] * Wimpn_compressed[4];
    const scalar_t x42   = S_ikmn_canonical[3] * Wimpn_compressed[4];
    const scalar_t x43   = S_ikmn_canonical[2] * Wimpn_compressed[3];
    const scalar_t x44   = x41 + x42 + x43;
    const scalar_t x45   = Wimpn_compressed[4] * x2 + x36 + x40 + x44;
    const scalar_t x46   = Wimpn_compressed[1] * x0 + x20 + x24 + x8;
    const scalar_t x47   = S_ikmn_canonical[3] * Wimpn_compressed[3];
    const scalar_t x48   = S_ikmn_canonical[0] * Wimpn_compressed[4];
    const scalar_t x49   = S_ikmn_canonical[4] * Wimpn_compressed[4];
    const scalar_t x50   = S_ikmn_canonical[4] * Wimpn_compressed[3];
    const scalar_t x51   = x48 + x49 + x50;
    const scalar_t x52   = Wimpn_compressed[4] * x1 + x40 + x47 + x51;
    const scalar_t x53   = S_ikmn_canonical[5] * Wimpn_compressed[3];
    const scalar_t x54   = Wimpn_compressed[4] * x0 + x44 + x51 + x53;
    const scalar_t x55   = x3 + x5;
    const scalar_t x56   = Wimpn_compressed[1] * x1 + x11 + x34 + x55 + x9;
    const scalar_t x57   = x17 + x3;
    const scalar_t x58   = x23 + x9;
    const scalar_t x59   = x4 + x58;
    const scalar_t x60   = Wimpn_compressed[1] * x2 + x32 + x57 + x59 + x6;
    const scalar_t x61   = Wimpn_compressed[0] * (S_ikmn_canonical[10] + S_ikmn_canonical[11] + S_ikmn_canonical[12] +
                                                S_ikmn_canonical[13] + S_ikmn_canonical[14] + S_ikmn_canonical[6] +
                                                S_ikmn_canonical[7] + S_ikmn_canonical[8] + S_ikmn_canonical[9]);
    const scalar_t x62   = S_ikmn_canonical[6] * Wimpn_compressed[1];
    const scalar_t x63   = S_ikmn_canonical[7] * Wimpn_compressed[1];
    const scalar_t x64   = x62 + x63;
    const scalar_t x65   = S_ikmn_canonical[11] * Wimpn_compressed[2];
    const scalar_t x66   = S_ikmn_canonical[14] * Wimpn_compressed[2];
    const scalar_t x67   = x65 + x66;
    const scalar_t x68   = S_ikmn_canonical[10] * Wimpn_compressed[2];
    const scalar_t x69   = S_ikmn_canonical[13] * Wimpn_compressed[2];
    const scalar_t x70   = x68 + x69;
    const scalar_t x71   = x67 + x70;
    const scalar_t x72   = S_ikmn_canonical[8] * Wimpn_compressed[1];
    const scalar_t x73   = S_ikmn_canonical[12] * Wimpn_compressed[2];
    const scalar_t x74   = S_ikmn_canonical[9] * Wimpn_compressed[2];
    const scalar_t x75   = x73 + x74;
    const scalar_t x76   = x72 + x75;
    const scalar_t x77   = x71 + x76;
    const scalar_t x78   = x64 + x77;
    const scalar_t x79   = S_ikmn_canonical[6] * Wimpn_compressed[2];
    const scalar_t x80   = x73 + x79;
    const scalar_t x81   = x66 + x69;
    const scalar_t x82   = x80 + x81;
    const scalar_t x83   = S_ikmn_canonical[9] * Wimpn_compressed[1];
    const scalar_t x84   = S_ikmn_canonical[7] * Wimpn_compressed[2];
    const scalar_t x85   = S_ikmn_canonical[8] * Wimpn_compressed[2];
    const scalar_t x86   = x84 + x85;
    const scalar_t x87   = x83 + x86;
    const scalar_t x88   = S_ikmn_canonical[10] * Wimpn_compressed[1];
    const scalar_t x89   = S_ikmn_canonical[11] * Wimpn_compressed[1];
    const scalar_t x90   = x88 + x89;
    const scalar_t x91   = x87 + x90;
    const scalar_t x92   = x82 + x91;
    const scalar_t x93   = S_ikmn_canonical[13] * Wimpn_compressed[1];
    const scalar_t x94   = x79 + x93;
    const scalar_t x95   = S_ikmn_canonical[12] * Wimpn_compressed[1];
    const scalar_t x96   = S_ikmn_canonical[14] * Wimpn_compressed[1];
    const scalar_t x97   = x95 + x96;
    const scalar_t x98   = x94 + x97;
    const scalar_t x99   = x74 + x86;
    const scalar_t x100  = x65 + x68;
    const scalar_t x101  = x100 + x99;
    const scalar_t x102  = x101 + x98;
    const scalar_t x103  = S_ikmn_canonical[12] * Wimpn_compressed[4];
    const scalar_t x104  = S_ikmn_canonical[9] * Wimpn_compressed[4];
    const scalar_t x105  = x103 + x104;
    const scalar_t x106  = S_ikmn_canonical[7] * Wimpn_compressed[3];
    const scalar_t x107  = S_ikmn_canonical[8] * Wimpn_compressed[3];
    const scalar_t x108  = x106 + x107;
    const scalar_t x109  = S_ikmn_canonical[6] * Wimpn_compressed[3];
    const scalar_t x110  = S_ikmn_canonical[13] * Wimpn_compressed[4];
    const scalar_t x111  = S_ikmn_canonical[14] * Wimpn_compressed[4];
    const scalar_t x112  = x110 + x111;
    const scalar_t x113  = S_ikmn_canonical[10] * Wimpn_compressed[4];
    const scalar_t x114  = S_ikmn_canonical[11] * Wimpn_compressed[4];
    const scalar_t x115  = x113 + x114;
    const scalar_t x116  = x109 + x112 + x115;
    const scalar_t x117  = x105 + x108 + x116;
    const scalar_t x118  = x64 + x72;
    const scalar_t x119  = x81 + x90;
    const scalar_t x120  = x118 + x119 + x73 + x83;
    const scalar_t x121  = S_ikmn_canonical[9] * Wimpn_compressed[3];
    const scalar_t x122  = S_ikmn_canonical[10] * Wimpn_compressed[3];
    const scalar_t x123  = S_ikmn_canonical[11] * Wimpn_compressed[3];
    const scalar_t x124  = x122 + x123;
    const scalar_t x125  = S_ikmn_canonical[6] * Wimpn_compressed[4];
    const scalar_t x126  = S_ikmn_canonical[7] * Wimpn_compressed[4];
    const scalar_t x127  = S_ikmn_canonical[8] * Wimpn_compressed[4];
    const scalar_t x128  = x126 + x127;
    const scalar_t x129  = x125 + x128;
    const scalar_t x130  = x103 + x112 + x121 + x124 + x129;
    const scalar_t x131  = S_ikmn_canonical[12] * Wimpn_compressed[3];
    const scalar_t x132  = S_ikmn_canonical[13] * Wimpn_compressed[3];
    const scalar_t x133  = S_ikmn_canonical[14] * Wimpn_compressed[3];
    const scalar_t x134  = x132 + x133;
    const scalar_t x135  = x104 + x115 + x129 + x131 + x134;
    const scalar_t x136  = x100 + x74;
    const scalar_t x137  = x118 + x136 + x93 + x97;
    const scalar_t x138  = x91 + x98;
    const scalar_t x139  = Wimpn_compressed[0] * (S_ikmn_canonical[15] + S_ikmn_canonical[16] + S_ikmn_canonical[17] +
                                                 S_ikmn_canonical[18] + S_ikmn_canonical[19] + S_ikmn_canonical[20] +
                                                 S_ikmn_canonical[21] + S_ikmn_canonical[22] + S_ikmn_canonical[23]);
    const scalar_t x140  = S_ikmn_canonical[15] * Wimpn_compressed[1];
    const scalar_t x141  = S_ikmn_canonical[16] * Wimpn_compressed[1];
    const scalar_t x142  = x140 + x141;
    const scalar_t x143  = S_ikmn_canonical[20] * Wimpn_compressed[2];
    const scalar_t x144  = S_ikmn_canonical[23] * Wimpn_compressed[2];
    const scalar_t x145  = x143 + x144;
    const scalar_t x146  = S_ikmn_canonical[19] * Wimpn_compressed[2];
    const scalar_t x147  = S_ikmn_canonical[22] * Wimpn_compressed[2];
    const scalar_t x148  = x146 + x147;
    const scalar_t x149  = x145 + x148;
    const scalar_t x150  = S_ikmn_canonical[17] * Wimpn_compressed[1];
    const scalar_t x151  = S_ikmn_canonical[18] * Wimpn_compressed[2];
    const scalar_t x152  = S_ikmn_canonical[21] * Wimpn_compressed[2];
    const scalar_t x153  = x151 + x152;
    const scalar_t x154  = x150 + x153;
    const scalar_t x155  = x149 + x154;
    const scalar_t x156  = x142 + x155;
    const scalar_t x157  = S_ikmn_canonical[15] * Wimpn_compressed[2];
    const scalar_t x158  = x152 + x157;
    const scalar_t x159  = x144 + x147;
    const scalar_t x160  = x158 + x159;
    const scalar_t x161  = S_ikmn_canonical[18] * Wimpn_compressed[1];
    const scalar_t x162  = S_ikmn_canonical[16] * Wimpn_compressed[2];
    const scalar_t x163  = S_ikmn_canonical[17] * Wimpn_compressed[2];
    const scalar_t x164  = x162 + x163;
    const scalar_t x165  = x161 + x164;
    const scalar_t x166  = S_ikmn_canonical[19] * Wimpn_compressed[1];
    const scalar_t x167  = S_ikmn_canonical[20] * Wimpn_compressed[1];
    const scalar_t x168  = x166 + x167;
    const scalar_t x169  = x165 + x168;
    const scalar_t x170  = x160 + x169;
    const scalar_t x171  = S_ikmn_canonical[22] * Wimpn_compressed[1];
    const scalar_t x172  = x157 + x171;
    const scalar_t x173  = S_ikmn_canonical[21] * Wimpn_compressed[1];
    const scalar_t x174  = S_ikmn_canonical[23] * Wimpn_compressed[1];
    const scalar_t x175  = x173 + x174;
    const scalar_t x176  = x172 + x175;
    const scalar_t x177  = x151 + x164;
    const scalar_t x178  = x143 + x146;
    const scalar_t x179  = x177 + x178;
    const scalar_t x180  = x176 + x179;
    const scalar_t x181  = S_ikmn_canonical[18] * Wimpn_compressed[4];
    const scalar_t x182  = S_ikmn_canonical[21] * Wimpn_compressed[4];
    const scalar_t x183  = x181 + x182;
    const scalar_t x184  = S_ikmn_canonical[16] * Wimpn_compressed[3];
    const scalar_t x185  = S_ikmn_canonical[17] * Wimpn_compressed[3];
    const scalar_t x186  = x184 + x185;
    const scalar_t x187  = S_ikmn_canonical[15] * Wimpn_compressed[3];
    const scalar_t x188  = S_ikmn_canonical[22] * Wimpn_compressed[4];
    const scalar_t x189  = S_ikmn_canonical[23] * Wimpn_compressed[4];
    const scalar_t x190  = x188 + x189;
    const scalar_t x191  = S_ikmn_canonical[19] * Wimpn_compressed[4];
    const scalar_t x192  = S_ikmn_canonical[20] * Wimpn_compressed[4];
    const scalar_t x193  = x191 + x192;
    const scalar_t x194  = x187 + x190 + x193;
    const scalar_t x195  = x183 + x186 + x194;
    const scalar_t x196  = x142 + x150;
    const scalar_t x197  = x159 + x168;
    const scalar_t x198  = x152 + x161 + x196 + x197;
    const scalar_t x199  = S_ikmn_canonical[18] * Wimpn_compressed[3];
    const scalar_t x200  = S_ikmn_canonical[19] * Wimpn_compressed[3];
    const scalar_t x201  = S_ikmn_canonical[20] * Wimpn_compressed[3];
    const scalar_t x202  = x200 + x201;
    const scalar_t x203  = S_ikmn_canonical[15] * Wimpn_compressed[4];
    const scalar_t x204  = S_ikmn_canonical[16] * Wimpn_compressed[4];
    const scalar_t x205  = S_ikmn_canonical[17] * Wimpn_compressed[4];
    const scalar_t x206  = x204 + x205;
    const scalar_t x207  = x203 + x206;
    const scalar_t x208  = x182 + x190 + x199 + x202 + x207;
    const scalar_t x209  = S_ikmn_canonical[21] * Wimpn_compressed[3];
    const scalar_t x210  = S_ikmn_canonical[22] * Wimpn_compressed[3];
    const scalar_t x211  = S_ikmn_canonical[23] * Wimpn_compressed[3];
    const scalar_t x212  = x210 + x211;
    const scalar_t x213  = x181 + x193 + x207 + x209 + x212;
    const scalar_t x214  = x151 + x178;
    const scalar_t x215  = x171 + x175 + x196 + x214;
    const scalar_t x216  = x169 + x176;
    const scalar_t x217  = x21 + x29;
    const scalar_t x218  = x14 + x23;
    const scalar_t x219  = x218 + x37;
    const scalar_t x220  = x219 + x22;
    const scalar_t x221  = x30 + x41;
    const scalar_t x222  = x218 + x221;
    const scalar_t x223  = S_ikmn_canonical[1] * Wimpn_compressed[5];
    const scalar_t x224  = S_ikmn_canonical[2] * Wimpn_compressed[5];
    const scalar_t x225  = x15 + x223 + x224 + x36 + x4;
    const scalar_t x226  = S_ikmn_canonical[1] * Wimpn_compressed[0];
    const scalar_t x227  = x14 + x22;
    const scalar_t x228  = x226 + x227 + x48;
    const scalar_t x229  = x14 + x30;
    const scalar_t x230  = x229 + x7;
    const scalar_t x231  = x227 + x55;
    const scalar_t x232  = S_ikmn_canonical[2] * Wimpn_compressed[0];
    const scalar_t x233  = x229 + x232 + x48;
    const scalar_t x234  = x4 + x9;
    const scalar_t x235  = x219 + x234 + x41;
    const scalar_t x236  = x62 + x71 + x95;
    const scalar_t x237  = x236 + x87;
    const scalar_t x238  = x71 + x86;
    const scalar_t x239  = S_ikmn_canonical[6] * Wimpn_compressed[0] + x238 + x75;
    const scalar_t x240  = x104 + x80;
    const scalar_t x241  = x238 + x240;
    const scalar_t x242  = x71 + x99;
    const scalar_t x243  = x103 + x79;
    const scalar_t x244  = x242 + x243;
    const scalar_t x245  = S_ikmn_canonical[12] * Wimpn_compressed[5];
    const scalar_t x246  = S_ikmn_canonical[9] * Wimpn_compressed[5];
    const scalar_t x247  = x109 + x238 + x245 + x246;
    const scalar_t x248  = S_ikmn_canonical[9] * Wimpn_compressed[0];
    const scalar_t x249  = x125 + x238 + x248 + x73;
    const scalar_t x250  = x236 + x99;
    const scalar_t x251  = x62 + x71 + x73 + x87;
    const scalar_t x252  = S_ikmn_canonical[12] * Wimpn_compressed[0];
    const scalar_t x253  = x125 + x242 + x252;
    const scalar_t x254  = x105 + x238 + x79;
    const scalar_t x255  = x140 + x149 + x173;
    const scalar_t x256  = x165 + x255;
    const scalar_t x257  = x149 + x164;
    const scalar_t x258  = S_ikmn_canonical[15] * Wimpn_compressed[0] + x153 + x257;
    const scalar_t x259  = x158 + x181;
    const scalar_t x260  = x257 + x259;
    const scalar_t x261  = x149 + x177;
    const scalar_t x262  = x157 + x182;
    const scalar_t x263  = x261 + x262;
    const scalar_t x264  = S_ikmn_canonical[18] * Wimpn_compressed[5];
    const scalar_t x265  = S_ikmn_canonical[21] * Wimpn_compressed[5];
    const scalar_t x266  = x187 + x257 + x264 + x265;
    const scalar_t x267  = S_ikmn_canonical[18] * Wimpn_compressed[0];
    const scalar_t x268  = x152 + x203 + x257 + x267;
    const scalar_t x269  = x177 + x255;
    const scalar_t x270  = x140 + x149 + x152 + x165;
    const scalar_t x271  = S_ikmn_canonical[21] * Wimpn_compressed[0];
    const scalar_t x272  = x203 + x261 + x271;
    const scalar_t x273  = x157 + x183 + x257;
    const scalar_t x274  = x217 + x23;
    const scalar_t x275  = x10 + x12;
    const scalar_t x276  = x24 + x274;
    const scalar_t x277  = x13 + x49;
    const scalar_t x278  = x276 + x277;
    const scalar_t x279  = x12 + x276;
    const scalar_t x280  = x19 + x279;
    const scalar_t x281  = x21 + x275;
    const scalar_t x282  = x281 + x58;
    const scalar_t x283  = x226 + x282 + x42;
    const scalar_t x284  = S_ikmn_canonical[4] * Wimpn_compressed[5];
    const scalar_t x285  = x12 + x223 + x26 + x284 + x47;
    const scalar_t x286  = x282 + x57;
    const scalar_t x287  = x26 + x277 + x37;
    const scalar_t x288  = S_ikmn_canonical[4] * Wimpn_compressed[0];
    const scalar_t x289  = x279 + x288 + x42;
    const scalar_t x290  = x63 + x94;
    const scalar_t x291  = x75 + x85;
    const scalar_t x292  = x67 + x88;
    const scalar_t x293  = x291 + x292;
    const scalar_t x294  = x290 + x293;
    const scalar_t x295  = x291 + x79;
    const scalar_t x296  = x126 + x295;
    const scalar_t x297  = x296 + x71;
    const scalar_t x298  = x67 + x69;
    const scalar_t x299  = x75 + x86;
    const scalar_t x300  = x299 + x79;
    const scalar_t x301  = S_ikmn_canonical[10] * Wimpn_compressed[0] + x298 + x300;
    const scalar_t x302  = x300 + x67;
    const scalar_t x303  = x110 + x68;
    const scalar_t x304  = x302 + x303;
    const scalar_t x305  = x299 + x94;
    const scalar_t x306  = x292 + x305;
    const scalar_t x307  = S_ikmn_canonical[7] * Wimpn_compressed[0];
    const scalar_t x308  = x113 + x298;
    const scalar_t x309  = x295 + x307 + x308;
    const scalar_t x310  = S_ikmn_canonical[13] * Wimpn_compressed[5];
    const scalar_t x311  = S_ikmn_canonical[7] * Wimpn_compressed[5];
    const scalar_t x312  = x122 + x295 + x310 + x311 + x67;
    const scalar_t x313  = x293 + x63 + x69 + x79;
    const scalar_t x314  = x303 + x67;
    const scalar_t x315  = x296 + x314;
    const scalar_t x316  = S_ikmn_canonical[13] * Wimpn_compressed[0];
    const scalar_t x317  = x113 + x302 + x316;
    const scalar_t x318  = x141 + x172;
    const scalar_t x319  = x153 + x163;
    const scalar_t x320  = x145 + x166;
    const scalar_t x321  = x319 + x320;
    const scalar_t x322  = x318 + x321;
    const scalar_t x323  = x157 + x319;
    const scalar_t x324  = x204 + x323;
    const scalar_t x325  = x149 + x324;
    const scalar_t x326  = x145 + x147;
    const scalar_t x327  = x153 + x164;
    const scalar_t x328  = x157 + x327;
    const scalar_t x329  = S_ikmn_canonical[19] * Wimpn_compressed[0] + x326 + x328;
    const scalar_t x330  = x145 + x328;
    const scalar_t x331  = x146 + x188;
    const scalar_t x332  = x330 + x331;
    const scalar_t x333  = x172 + x327;
    const scalar_t x334  = x320 + x333;
    const scalar_t x335  = S_ikmn_canonical[16] * Wimpn_compressed[0];
    const scalar_t x336  = x191 + x326;
    const scalar_t x337  = x323 + x335 + x336;
    const scalar_t x338  = S_ikmn_canonical[16] * Wimpn_compressed[5];
    const scalar_t x339  = S_ikmn_canonical[22] * Wimpn_compressed[5];
    const scalar_t x340  = x145 + x200 + x323 + x338 + x339;
    const scalar_t x341  = x141 + x147 + x157 + x321;
    const scalar_t x342  = x145 + x331;
    const scalar_t x343  = x324 + x342;
    const scalar_t x344  = S_ikmn_canonical[22] * Wimpn_compressed[0];
    const scalar_t x345  = x191 + x330 + x344;
    const scalar_t x346  = x10 + x11;
    const scalar_t x347  = x11 + x276;
    const scalar_t x348  = x33 + x347;
    const scalar_t x349  = x221 + x25 + x277;
    const scalar_t x350  = x29 + x346;
    const scalar_t x351  = x350 + x4;
    const scalar_t x352  = x28 + x32 + x351;
    const scalar_t x353  = x224 + x25 + x284 + x31 + x53;
    const scalar_t x354  = x23 + x232 + x351 + x38;
    const scalar_t x355  = x288 + x347 + x38;
    const scalar_t x356  = x70 + x89;
    const scalar_t x357  = x79 + x84;
    const scalar_t x358  = x76 + x96;
    const scalar_t x359  = x357 + x358;
    const scalar_t x360  = x356 + x359;
    const scalar_t x361  = x357 + x75;
    const scalar_t x362  = x127 + x361;
    const scalar_t x363  = x362 + x71;
    const scalar_t x364  = x300 + x70;
    const scalar_t x365  = x114 + x66;
    const scalar_t x366  = x364 + x365;
    const scalar_t x367  = S_ikmn_canonical[14] * Wimpn_compressed[0] + x364 + x65;
    const scalar_t x368  = x300 + x356 + x96;
    const scalar_t x369  = x365 + x70;
    const scalar_t x370  = x362 + x369;
    const scalar_t x371  = x65 + x70;
    const scalar_t x372  = x359 + x371;
    const scalar_t x373  = S_ikmn_canonical[11] * Wimpn_compressed[5];
    const scalar_t x374  = S_ikmn_canonical[8] * Wimpn_compressed[5];
    const scalar_t x375  = x133 + x361 + x373 + x374 + x70;
    const scalar_t x376  = S_ikmn_canonical[8] * Wimpn_compressed[0];
    const scalar_t x377  = x111 + x371;
    const scalar_t x378  = x361 + x376 + x377;
    const scalar_t x379  = S_ikmn_canonical[11] * Wimpn_compressed[0];
    const scalar_t x380  = x111 + x364 + x379;
    const scalar_t x381  = x148 + x167;
    const scalar_t x382  = x157 + x162;
    const scalar_t x383  = x154 + x174;
    const scalar_t x384  = x382 + x383;
    const scalar_t x385  = x381 + x384;
    const scalar_t x386  = x153 + x382;
    const scalar_t x387  = x205 + x386;
    const scalar_t x388  = x149 + x387;
    const scalar_t x389  = x148 + x328;
    const scalar_t x390  = x144 + x192;
    const scalar_t x391  = x389 + x390;
    const scalar_t x392  = S_ikmn_canonical[23] * Wimpn_compressed[0] + x143 + x389;
    const scalar_t x393  = x174 + x328 + x381;
    const scalar_t x394  = x148 + x390;
    const scalar_t x395  = x387 + x394;
    const scalar_t x396  = x143 + x148;
    const scalar_t x397  = x384 + x396;
    const scalar_t x398  = S_ikmn_canonical[17] * Wimpn_compressed[5];
    const scalar_t x399  = S_ikmn_canonical[20] * Wimpn_compressed[5];
    const scalar_t x400  = x148 + x211 + x386 + x398 + x399;
    const scalar_t x401  = S_ikmn_canonical[17] * Wimpn_compressed[0];
    const scalar_t x402  = x189 + x396;
    const scalar_t x403  = x386 + x401 + x402;
    const scalar_t x404  = S_ikmn_canonical[20] * Wimpn_compressed[0];
    const scalar_t x405  = x189 + x389 + x404;
    const scalar_t x406  = Wimpn_compressed[7] * x1;
    const scalar_t x407  = S_ikmn_canonical[3] * Wimpn_compressed[6];
    const scalar_t x408  = S_ikmn_canonical[5] * Wimpn_compressed[6];
    const scalar_t x409  = x407 + x408;
    const scalar_t x410  = S_ikmn_canonical[0] * Wimpn_compressed[6];
    const scalar_t x411  = Wimpn_compressed[7] * x0;
    const scalar_t x412  = x410 + x411;
    const scalar_t x413  = x409 + x412;
    const scalar_t x414  = Wimpn_compressed[3] * x0;
    const scalar_t x415  = S_ikmn_canonical[4] * Wimpn_compressed[8];
    const scalar_t x416  = x25 + x4;
    const scalar_t x417  = S_ikmn_canonical[3] * Wimpn_compressed[8] + x12 + x43;
    const scalar_t x418  = x414 + x415 + x416 + x417;
    const scalar_t x419  = S_ikmn_canonical[2] * Wimpn_compressed[7];
    const scalar_t x420  = S_ikmn_canonical[4] * Wimpn_compressed[7];
    const scalar_t x421  = S_ikmn_canonical[5] * Wimpn_compressed[7] + x11 + x419 + x420;
    const scalar_t x422  = x411 + x416 + x421;
    const scalar_t x423  = x24 + x58;
    const scalar_t x424  = S_ikmn_canonical[1] * Wimpn_compressed[7];
    const scalar_t x425  = S_ikmn_canonical[3] * Wimpn_compressed[7] + x12 + x420 + x424;
    const scalar_t x426  = x406 + x423 + x425;
    const scalar_t x427  = Wimpn_compressed[3] * x1;
    const scalar_t x428  = S_ikmn_canonical[5] * Wimpn_compressed[8] + x11 + x39;
    const scalar_t x429  = x415 + x423 + x427 + x428;
    const scalar_t x430  = Wimpn_compressed[3] * x2;
    const scalar_t x431  = x274 + x430 + x47 + x53;
    const scalar_t x432  = x121 + x131;
    const scalar_t x433  = x116 + x128 + x432;
    const scalar_t x434  = x71 + x75;
    const scalar_t x435  = x109 + x311 + x374 + x434;
    const scalar_t x436  = x119 + x300;
    const scalar_t x437  = x100 + x305 + x96;
    const scalar_t x438  = S_ikmn_canonical[12] * Wimpn_compressed[7];
    const scalar_t x439  = S_ikmn_canonical[7] * Wimpn_compressed[7];
    const scalar_t x440  = x438 + x439;
    const scalar_t x441  = S_ikmn_canonical[13] * Wimpn_compressed[6] + x440;
    const scalar_t x442  = S_ikmn_canonical[8] * Wimpn_compressed[7];
    const scalar_t x443  = S_ikmn_canonical[9] * Wimpn_compressed[7];
    const scalar_t x444  = x442 + x443;
    const scalar_t x445  = S_ikmn_canonical[11] * Wimpn_compressed[6] + x444;
    const scalar_t x446  = S_ikmn_canonical[6] * Wimpn_compressed[6];
    const scalar_t x447  = S_ikmn_canonical[10] * Wimpn_compressed[6];
    const scalar_t x448  = S_ikmn_canonical[14] * Wimpn_compressed[6];
    const scalar_t x449  = x447 + x448;
    const scalar_t x450  = x446 + x449;
    const scalar_t x451  = x441 + x445 + x450;
    const scalar_t x452  = S_ikmn_canonical[10] * Wimpn_compressed[8];
    const scalar_t x453  = S_ikmn_canonical[11] * Wimpn_compressed[8];
    const scalar_t x454  = x108 + x121 + x452 + x453 + x82;
    const scalar_t x455  = S_ikmn_canonical[13] * Wimpn_compressed[7];
    const scalar_t x456  = x439 + x455;
    const scalar_t x457  = S_ikmn_canonical[14] * Wimpn_compressed[7];
    const scalar_t x458  = x100 + x457;
    const scalar_t x459  = x444 + x456 + x458 + x80;
    const scalar_t x460  = S_ikmn_canonical[11] * Wimpn_compressed[7];
    const scalar_t x461  = x442 + x460;
    const scalar_t x462  = x440 + x74;
    const scalar_t x463  = S_ikmn_canonical[10] * Wimpn_compressed[7];
    const scalar_t x464  = x463 + x79;
    const scalar_t x465  = x464 + x81;
    const scalar_t x466  = x461 + x462 + x465;
    const scalar_t x467  = S_ikmn_canonical[14] * Wimpn_compressed[8];
    const scalar_t x468  = S_ikmn_canonical[13] * Wimpn_compressed[8] + x79;
    const scalar_t x469  = x108 + x131 + x136 + x467 + x468;
    const scalar_t x470  = x124 + x134 + x300;
    const scalar_t x471  = x199 + x209;
    const scalar_t x472  = x194 + x206 + x471;
    const scalar_t x473  = x149 + x153;
    const scalar_t x474  = x187 + x338 + x398 + x473;
    const scalar_t x475  = x197 + x328;
    const scalar_t x476  = x174 + x178 + x333;
    const scalar_t x477  = S_ikmn_canonical[16] * Wimpn_compressed[7];
    const scalar_t x478  = S_ikmn_canonical[21] * Wimpn_compressed[7];
    const scalar_t x479  = x477 + x478;
    const scalar_t x480  = S_ikmn_canonical[22] * Wimpn_compressed[6] + x479;
    const scalar_t x481  = S_ikmn_canonical[17] * Wimpn_compressed[7];
    const scalar_t x482  = S_ikmn_canonical[18] * Wimpn_compressed[7];
    const scalar_t x483  = x481 + x482;
    const scalar_t x484  = S_ikmn_canonical[20] * Wimpn_compressed[6] + x483;
    const scalar_t x485  = S_ikmn_canonical[15] * Wimpn_compressed[6];
    const scalar_t x486  = S_ikmn_canonical[19] * Wimpn_compressed[6];
    const scalar_t x487  = S_ikmn_canonical[23] * Wimpn_compressed[6];
    const scalar_t x488  = x486 + x487;
    const scalar_t x489  = x485 + x488;
    const scalar_t x490  = x480 + x484 + x489;
    const scalar_t x491  = S_ikmn_canonical[19] * Wimpn_compressed[8];
    const scalar_t x492  = S_ikmn_canonical[20] * Wimpn_compressed[8];
    const scalar_t x493  = x160 + x186 + x199 + x491 + x492;
    const scalar_t x494  = S_ikmn_canonical[22] * Wimpn_compressed[7];
    const scalar_t x495  = x477 + x494;
    const scalar_t x496  = S_ikmn_canonical[23] * Wimpn_compressed[7];
    const scalar_t x497  = x178 + x496;
    const scalar_t x498  = x158 + x483 + x495 + x497;
    const scalar_t x499  = S_ikmn_canonical[20] * Wimpn_compressed[7];
    const scalar_t x500  = x481 + x499;
    const scalar_t x501  = x151 + x479;
    const scalar_t x502  = S_ikmn_canonical[19] * Wimpn_compressed[7];
    const scalar_t x503  = x157 + x502;
    const scalar_t x504  = x159 + x503;
    const scalar_t x505  = x500 + x501 + x504;
    const scalar_t x506  = S_ikmn_canonical[23] * Wimpn_compressed[8];
    const scalar_t x507  = S_ikmn_canonical[22] * Wimpn_compressed[8] + x157;
    const scalar_t x508  = x186 + x209 + x214 + x506 + x507;
    const scalar_t x509  = x202 + x212 + x328;
    const scalar_t x510  = S_ikmn_canonical[2] * Wimpn_compressed[8];
    const scalar_t x511  = x13 + x24;
    const scalar_t x512  = S_ikmn_canonical[0] * Wimpn_compressed[8] + x50 + x511;
    const scalar_t x513  = x4 + x414 + x510 + x512;
    const scalar_t x514  = x281 + x36 + x414 + x47;
    const scalar_t x515  = S_ikmn_canonical[0] * Wimpn_compressed[7] + x234 + x419 + x424;
    const scalar_t x516  = S_ikmn_canonical[4] * Wimpn_compressed[6] + x511 + x515;
    const scalar_t x517  = x24 + x59;
    const scalar_t x518  = S_ikmn_canonical[2] * Wimpn_compressed[6] + x425 + x517;
    const scalar_t x519  = x292 + x64 + x83 + x85 + x93 + x95;
    const scalar_t x520  = x291 + x71;
    const scalar_t x521  = x125 + x307 + x520;
    const scalar_t x522  = x248 + x308 + x80 + x86;
    const scalar_t x523  = x243 + x314 + x99;
    const scalar_t x524  = x106 + x85;
    const scalar_t x525  = x524 + x67;
    const scalar_t x526  = x432 + x452 + x468 + x525;
    const scalar_t x527  = x298 + x73;
    const scalar_t x528  = x443 + x85;
    const scalar_t x529  = x439 + x446 + x447 + x527 + x528;
    const scalar_t x530  = S_ikmn_canonical[6] * Wimpn_compressed[8];
    const scalar_t x531  = x121 + x530;
    const scalar_t x532  = x132 + x68;
    const scalar_t x533  = S_ikmn_canonical[12] * Wimpn_compressed[8] + x532;
    const scalar_t x534  = x525 + x531 + x533;
    const scalar_t x535  = x109 + x121 + x122 + x524 + x527;
    const scalar_t x536  = x67 + x74;
    const scalar_t x537  = S_ikmn_canonical[6] * Wimpn_compressed[7];
    const scalar_t x538  = x537 + x68;
    const scalar_t x539  = x538 + x85;
    const scalar_t x540  = x441 + x536 + x539;
    const scalar_t x541  = S_ikmn_canonical[12] * Wimpn_compressed[6] + x443;
    const scalar_t x542  = x455 + x67;
    const scalar_t x543  = x464 + x541 + x542 + x86;
    const scalar_t x544  = x142 + x161 + x163 + x171 + x173 + x320;
    const scalar_t x545  = x149 + x319;
    const scalar_t x546  = x203 + x335 + x545;
    const scalar_t x547  = x158 + x164 + x267 + x336;
    const scalar_t x548  = x177 + x262 + x342;
    const scalar_t x549  = x163 + x184;
    const scalar_t x550  = x145 + x549;
    const scalar_t x551  = x471 + x491 + x507 + x550;
    const scalar_t x552  = x152 + x326;
    const scalar_t x553  = x163 + x482;
    const scalar_t x554  = x477 + x485 + x486 + x552 + x553;
    const scalar_t x555  = S_ikmn_canonical[15] * Wimpn_compressed[8];
    const scalar_t x556  = x199 + x555;
    const scalar_t x557  = x146 + x210;
    const scalar_t x558  = S_ikmn_canonical[21] * Wimpn_compressed[8] + x557;
    const scalar_t x559  = x550 + x556 + x558;
    const scalar_t x560  = x187 + x199 + x200 + x549 + x552;
    const scalar_t x561  = x145 + x151;
    const scalar_t x562  = S_ikmn_canonical[15] * Wimpn_compressed[7];
    const scalar_t x563  = x146 + x562;
    const scalar_t x564  = x163 + x563;
    const scalar_t x565  = x480 + x561 + x564;
    const scalar_t x566  = S_ikmn_canonical[21] * Wimpn_compressed[6] + x482;
    const scalar_t x567  = x145 + x494;
    const scalar_t x568  = x164 + x503 + x566 + x567;
    const scalar_t x569  = Wimpn_compressed[7] * x2;
    const scalar_t x570  = x13 + x515 + x569;
    const scalar_t x571  = x350 + x36 + x427 + x53;
    const scalar_t x572  = x430 + x59;
    const scalar_t x573  = x428 + x510 + x572;
    const scalar_t x574  = x105 + x125;
    const scalar_t x575  = x106 + x111 + x114 + x122 + x127 + x132 + x574;
    const scalar_t x576  = x62 + x84;
    const scalar_t x577  = x576 + x77;
    const scalar_t x578  = x122 + x246 + x373 + x82 + x86;
    const scalar_t x579  = x79 + x99;
    const scalar_t x580  = x371 + x579 + x97;
    const scalar_t x581  = x457 + x460 + x70 + x79;
    const scalar_t x582  = x440 + x528 + x581;
    const scalar_t x583  = x66 + x73;
    const scalar_t x584  = x583 + x70;
    const scalar_t x585  = S_ikmn_canonical[8] * Wimpn_compressed[8] + x106 + x123;
    const scalar_t x586  = x531 + x584 + x585;
    const scalar_t x587  = S_ikmn_canonical[8] * Wimpn_compressed[6] + x460;
    const scalar_t x588  = x450 + x456 + x541 + x587;
    const scalar_t x589  = x455 + x460;
    const scalar_t x590  = x444 + x538 + x583 + x589 + x84;
    const scalar_t x591  = x107 + x133;
    const scalar_t x592  = x131 + x84;
    const scalar_t x593  = x371 + x74;
    const scalar_t x594  = x109 + x591 + x592 + x593;
    const scalar_t x595  = x467 + x79;
    const scalar_t x596  = x123 + x86;
    const scalar_t x597  = x121 + x533 + x595 + x596;
    const scalar_t x598  = x183 + x203;
    const scalar_t x599  = x184 + x189 + x192 + x200 + x205 + x210 + x598;
    const scalar_t x600  = x140 + x162;
    const scalar_t x601  = x155 + x600;
    const scalar_t x602  = x160 + x164 + x200 + x264 + x399;
    const scalar_t x603  = x157 + x177;
    const scalar_t x604  = x175 + x396 + x603;
    const scalar_t x605  = x148 + x157 + x496 + x499;
    const scalar_t x606  = x479 + x553 + x605;
    const scalar_t x607  = x144 + x152;
    const scalar_t x608  = x148 + x607;
    const scalar_t x609  = S_ikmn_canonical[17] * Wimpn_compressed[8] + x184 + x201;
    const scalar_t x610  = x556 + x608 + x609;
    const scalar_t x611  = S_ikmn_canonical[17] * Wimpn_compressed[6] + x499;
    const scalar_t x612  = x489 + x495 + x566 + x611;
    const scalar_t x613  = x494 + x499;
    const scalar_t x614  = x162 + x483 + x563 + x607 + x613;
    const scalar_t x615  = x185 + x211;
    const scalar_t x616  = x162 + x209;
    const scalar_t x617  = x151 + x396;
    const scalar_t x618  = x187 + x615 + x616 + x617;
    const scalar_t x619  = x157 + x506;
    const scalar_t x620  = x164 + x201;
    const scalar_t x621  = x199 + x558 + x619 + x620;
    const scalar_t x622  = x406 + x410;
    const scalar_t x623  = x409 + x569;
    const scalar_t x624  = S_ikmn_canonical[1] * Wimpn_compressed[8];
    const scalar_t x625  = x427 + x512 + x624 + x9;
    const scalar_t x626  = x417 + x572 + x624;
    const scalar_t x627  = x110 + x113 + x123 + x126 + x574 + x591;
    const scalar_t x628  = x520 + x64;
    const scalar_t x629  = x292 + x69 + x80 + x87;
    const scalar_t x630  = x101 + x133 + x245 + x310 + x79;
    const scalar_t x631  = x357 + x438 + x444 + x463 + x542;
    const scalar_t x632  = x462 + x539 + x589 + x66;
    const scalar_t x633  = S_ikmn_canonical[9] * Wimpn_compressed[6] + x438;
    const scalar_t x634  = S_ikmn_canonical[7] * Wimpn_compressed[6] + x455;
    const scalar_t x635  = x450 + x461 + x633 + x634;
    const scalar_t x636  = S_ikmn_canonical[7] * Wimpn_compressed[8];
    const scalar_t x637  = x107 + x530;
    const scalar_t x638  = x131 + x532 + x536 + x636 + x637;
    const scalar_t x639  = S_ikmn_canonical[9] * Wimpn_compressed[8] + x66;
    const scalar_t x640  = x132 + x452 + x79;
    const scalar_t x641  = x131 + x596 + x639 + x640;
    const scalar_t x642  = x188 + x191 + x201 + x204 + x598 + x615;
    const scalar_t x643  = x142 + x545;
    const scalar_t x644  = x147 + x158 + x165 + x320;
    const scalar_t x645  = x157 + x179 + x211 + x265 + x339;
    const scalar_t x646  = x382 + x478 + x483 + x502 + x567;
    const scalar_t x647  = x144 + x501 + x564 + x613;
    const scalar_t x648  = S_ikmn_canonical[18] * Wimpn_compressed[6] + x478;
    const scalar_t x649  = S_ikmn_canonical[16] * Wimpn_compressed[6] + x494;
    const scalar_t x650  = x489 + x500 + x648 + x649;
    const scalar_t x651  = S_ikmn_canonical[16] * Wimpn_compressed[8];
    const scalar_t x652  = x185 + x555;
    const scalar_t x653  = x209 + x557 + x561 + x651 + x652;
    const scalar_t x654  = S_ikmn_canonical[18] * Wimpn_compressed[8] + x144;
    const scalar_t x655  = x157 + x210 + x491;
    const scalar_t x656  = x209 + x620 + x654 + x655;
    const scalar_t x657  = S_ikmn_canonical[1] * Wimpn_compressed[6] + x421 + x517;
    const scalar_t x658  = x356 + x576 + x72 + x83 + x97;
    const scalar_t x659  = x125 + x376 + x434 + x84;
    const scalar_t x660  = x240 + x369 + x86;
    const scalar_t x661  = x252 + x377 + x579;
    const scalar_t x662  = x107 + x357 + x432 + x453 + x467 + x70;
    const scalar_t x663  = x445 + x537 + x584 + x84;
    const scalar_t x664  = x123 + x592 + x637 + x639 + x70;
    const scalar_t x665  = x438 + x442 + x446 + x448 + x593 + x84;
    const scalar_t x666  = x581 + x633 + x86;
    const scalar_t x667  = x150 + x161 + x175 + x381 + x600;
    const scalar_t x668  = x162 + x203 + x401 + x473;
    const scalar_t x669  = x164 + x259 + x394;
    const scalar_t x670  = x271 + x402 + x603;
    const scalar_t x671  = x148 + x185 + x382 + x471 + x492 + x506;
    const scalar_t x672  = x162 + x484 + x562 + x608;
    const scalar_t x673  = x148 + x201 + x616 + x652 + x654;
    const scalar_t x674  = x162 + x478 + x481 + x485 + x487 + x617;
    const scalar_t x675  = x164 + x605 + x648;
    const scalar_t x676  = x290 + x358 + x90;
    const scalar_t x677  = x128 + x434 + x79;
    const scalar_t x678  = x113 + x300 + x379 + x81;
    const scalar_t x679  = x100 + x111 + x300 + x316;
    const scalar_t x680  = x439 + x465 + x587 + x75;
    const scalar_t x681  = x532 + x585 + x595 + x75;
    const scalar_t x682  = x107 + x123 + x636 + x640 + x66 + x75;
    const scalar_t x683  = x442 + x458 + x634 + x75 + x79;
    const scalar_t x684  = x300 + x449 + x589;
    const scalar_t x685  = x168 + x318 + x383;
    const scalar_t x686  = x157 + x206 + x473;
    const scalar_t x687  = x159 + x191 + x328 + x404;
    const scalar_t x688  = x178 + x189 + x328 + x344;
    const scalar_t x689  = x153 + x477 + x504 + x611;
    const scalar_t x690  = x153 + x557 + x609 + x619;
    const scalar_t x691  = x144 + x153 + x185 + x201 + x651 + x655;
    const scalar_t x692  = x153 + x157 + x481 + x497 + x649;
    const scalar_t x693  = x328 + x488 + x613;
    const scalar_t x694  = 2 * S_ikmn_canonical[25];
    const scalar_t x695  = 2 * S_ikmn_canonical[26];
    const scalar_t x696  = 2 * S_ikmn_canonical[28];
    const scalar_t x697  = S_ikmn_canonical[25] * Wimpn_compressed[1];
    const scalar_t x698  = S_ikmn_canonical[26] * Wimpn_compressed[2];
    const scalar_t x699  = S_ikmn_canonical[24] * Wimpn_compressed[1];
    const scalar_t x700  = S_ikmn_canonical[26] * Wimpn_compressed[1];
    const scalar_t x701  = x699 + x700;
    const scalar_t x702  = x698 + x701;
    const scalar_t x703  = S_ikmn_canonical[25] * Wimpn_compressed[2];
    const scalar_t x704  = Wimpn_compressed[2] * x696;
    const scalar_t x705  = S_ikmn_canonical[27] * Wimpn_compressed[2];
    const scalar_t x706  = S_ikmn_canonical[29] * Wimpn_compressed[2];
    const scalar_t x707  = x705 + x706;
    const scalar_t x708  = x704 + x707;
    const scalar_t x709  = x703 + x708;
    const scalar_t x710  = x697 + x702 + x709;
    const scalar_t x711  = S_ikmn_canonical[27] * Wimpn_compressed[1];
    const scalar_t x712  = S_ikmn_canonical[28] * Wimpn_compressed[1];
    const scalar_t x713  = x711 + x712;
    const scalar_t x714  = x706 + x713;
    const scalar_t x715  = Wimpn_compressed[2] * x695;
    const scalar_t x716  = x703 + x715;
    const scalar_t x717  = S_ikmn_canonical[24] * Wimpn_compressed[2];
    const scalar_t x718  = S_ikmn_canonical[28] * Wimpn_compressed[2];
    const scalar_t x719  = x717 + x718;
    const scalar_t x720  = x716 + x719;
    const scalar_t x721  = x697 + x714 + x720;
    const scalar_t x722  = x700 + x717;
    const scalar_t x723  = Wimpn_compressed[2] * x694;
    const scalar_t x724  = x698 + x723;
    const scalar_t x725  = x705 + x724;
    const scalar_t x726  = S_ikmn_canonical[29] * Wimpn_compressed[1];
    const scalar_t x727  = x712 + x726;
    const scalar_t x728  = x718 + x727;
    const scalar_t x729  = x722 + x725 + x728;
    const scalar_t x730  = S_ikmn_canonical[24] * Wimpn_compressed[3];
    const scalar_t x731  = S_ikmn_canonical[25] * Wimpn_compressed[4];
    const scalar_t x732  = S_ikmn_canonical[29] * Wimpn_compressed[4];
    const scalar_t x733  = S_ikmn_canonical[25] * Wimpn_compressed[3];
    const scalar_t x734  = x731 + x732 + x733;
    const scalar_t x735  = S_ikmn_canonical[26] * Wimpn_compressed[4];
    const scalar_t x736  = S_ikmn_canonical[27] * Wimpn_compressed[4];
    const scalar_t x737  = S_ikmn_canonical[26] * Wimpn_compressed[3];
    const scalar_t x738  = x735 + x736 + x737;
    const scalar_t x739  = Wimpn_compressed[4] * x696 + x730 + x734 + x738;
    const scalar_t x740  = Wimpn_compressed[1] * x694 + x702 + x714 + x718;
    const scalar_t x741  = S_ikmn_canonical[27] * Wimpn_compressed[3];
    const scalar_t x742  = S_ikmn_canonical[24] * Wimpn_compressed[4];
    const scalar_t x743  = S_ikmn_canonical[28] * Wimpn_compressed[4];
    const scalar_t x744  = S_ikmn_canonical[28] * Wimpn_compressed[3];
    const scalar_t x745  = x742 + x743 + x744;
    const scalar_t x746  = Wimpn_compressed[4] * x695 + x734 + x741 + x745;
    const scalar_t x747  = S_ikmn_canonical[29] * Wimpn_compressed[3];
    const scalar_t x748  = Wimpn_compressed[4] * x694 + x738 + x745 + x747;
    const scalar_t x749  = x697 + x699;
    const scalar_t x750  = Wimpn_compressed[1] * x695 + x703 + x705 + x728 + x749;
    const scalar_t x751  = x697 + x711;
    const scalar_t x752  = x703 + x717;
    const scalar_t x753  = x698 + x752;
    const scalar_t x754  = Wimpn_compressed[1] * x696 + x700 + x726 + x751 + x753;
    const scalar_t x755  = Wimpn_compressed[0] * (S_ikmn_canonical[30] + S_ikmn_canonical[31] + S_ikmn_canonical[32] +
                                                 S_ikmn_canonical[33] + S_ikmn_canonical[34] + S_ikmn_canonical[35] +
                                                 S_ikmn_canonical[36] + S_ikmn_canonical[37] + S_ikmn_canonical[38]);
    const scalar_t x756  = S_ikmn_canonical[30] * Wimpn_compressed[1];
    const scalar_t x757  = S_ikmn_canonical[31] * Wimpn_compressed[1];
    const scalar_t x758  = x756 + x757;
    const scalar_t x759  = S_ikmn_canonical[35] * Wimpn_compressed[2];
    const scalar_t x760  = S_ikmn_canonical[38] * Wimpn_compressed[2];
    const scalar_t x761  = x759 + x760;
    const scalar_t x762  = S_ikmn_canonical[34] * Wimpn_compressed[2];
    const scalar_t x763  = S_ikmn_canonical[37] * Wimpn_compressed[2];
    const scalar_t x764  = x762 + x763;
    const scalar_t x765  = x761 + x764;
    const scalar_t x766  = S_ikmn_canonical[32] * Wimpn_compressed[1];
    const scalar_t x767  = S_ikmn_canonical[33] * Wimpn_compressed[2];
    const scalar_t x768  = S_ikmn_canonical[36] * Wimpn_compressed[2];
    const scalar_t x769  = x767 + x768;
    const scalar_t x770  = x766 + x769;
    const scalar_t x771  = x765 + x770;
    const scalar_t x772  = x758 + x771;
    const scalar_t x773  = S_ikmn_canonical[30] * Wimpn_compressed[2];
    const scalar_t x774  = x768 + x773;
    const scalar_t x775  = x760 + x763;
    const scalar_t x776  = x774 + x775;
    const scalar_t x777  = S_ikmn_canonical[33] * Wimpn_compressed[1];
    const scalar_t x778  = S_ikmn_canonical[31] * Wimpn_compressed[2];
    const scalar_t x779  = S_ikmn_canonical[32] * Wimpn_compressed[2];
    const scalar_t x780  = x778 + x779;
    const scalar_t x781  = x777 + x780;
    const scalar_t x782  = S_ikmn_canonical[34] * Wimpn_compressed[1];
    const scalar_t x783  = S_ikmn_canonical[35] * Wimpn_compressed[1];
    const scalar_t x784  = x782 + x783;
    const scalar_t x785  = x781 + x784;
    const scalar_t x786  = x776 + x785;
    const scalar_t x787  = S_ikmn_canonical[37] * Wimpn_compressed[1];
    const scalar_t x788  = x773 + x787;
    const scalar_t x789  = S_ikmn_canonical[36] * Wimpn_compressed[1];
    const scalar_t x790  = S_ikmn_canonical[38] * Wimpn_compressed[1];
    const scalar_t x791  = x789 + x790;
    const scalar_t x792  = x788 + x791;
    const scalar_t x793  = x767 + x780;
    const scalar_t x794  = x759 + x762;
    const scalar_t x795  = x793 + x794;
    const scalar_t x796  = x792 + x795;
    const scalar_t x797  = S_ikmn_canonical[33] * Wimpn_compressed[4];
    const scalar_t x798  = S_ikmn_canonical[36] * Wimpn_compressed[4];
    const scalar_t x799  = x797 + x798;
    const scalar_t x800  = S_ikmn_canonical[31] * Wimpn_compressed[3];
    const scalar_t x801  = S_ikmn_canonical[32] * Wimpn_compressed[3];
    const scalar_t x802  = x800 + x801;
    const scalar_t x803  = S_ikmn_canonical[30] * Wimpn_compressed[3];
    const scalar_t x804  = S_ikmn_canonical[37] * Wimpn_compressed[4];
    const scalar_t x805  = S_ikmn_canonical[38] * Wimpn_compressed[4];
    const scalar_t x806  = x804 + x805;
    const scalar_t x807  = S_ikmn_canonical[34] * Wimpn_compressed[4];
    const scalar_t x808  = S_ikmn_canonical[35] * Wimpn_compressed[4];
    const scalar_t x809  = x807 + x808;
    const scalar_t x810  = x803 + x806 + x809;
    const scalar_t x811  = x799 + x802 + x810;
    const scalar_t x812  = x758 + x766;
    const scalar_t x813  = x775 + x784;
    const scalar_t x814  = x768 + x777 + x812 + x813;
    const scalar_t x815  = S_ikmn_canonical[33] * Wimpn_compressed[3];
    const scalar_t x816  = S_ikmn_canonical[34] * Wimpn_compressed[3];
    const scalar_t x817  = S_ikmn_canonical[35] * Wimpn_compressed[3];
    const scalar_t x818  = x816 + x817;
    const scalar_t x819  = S_ikmn_canonical[30] * Wimpn_compressed[4];
    const scalar_t x820  = S_ikmn_canonical[31] * Wimpn_compressed[4];
    const scalar_t x821  = S_ikmn_canonical[32] * Wimpn_compressed[4];
    const scalar_t x822  = x820 + x821;
    const scalar_t x823  = x819 + x822;
    const scalar_t x824  = x798 + x806 + x815 + x818 + x823;
    const scalar_t x825  = S_ikmn_canonical[36] * Wimpn_compressed[3];
    const scalar_t x826  = S_ikmn_canonical[37] * Wimpn_compressed[3];
    const scalar_t x827  = S_ikmn_canonical[38] * Wimpn_compressed[3];
    const scalar_t x828  = x826 + x827;
    const scalar_t x829  = x797 + x809 + x823 + x825 + x828;
    const scalar_t x830  = x767 + x794;
    const scalar_t x831  = x787 + x791 + x812 + x830;
    const scalar_t x832  = x785 + x792;
    const scalar_t x833  = x715 + x723;
    const scalar_t x834  = x708 + x717;
    const scalar_t x835  = x731 + x834;
    const scalar_t x836  = x716 + x835;
    const scalar_t x837  = x724 + x735;
    const scalar_t x838  = x834 + x837;
    const scalar_t x839  = S_ikmn_canonical[25] * Wimpn_compressed[5];
    const scalar_t x840  = S_ikmn_canonical[26] * Wimpn_compressed[5];
    const scalar_t x841  = x698 + x709 + x730 + x839 + x840;
    const scalar_t x842  = S_ikmn_canonical[25] * Wimpn_compressed[0];
    const scalar_t x843  = x708 + x716;
    const scalar_t x844  = x742 + x842 + x843;
    const scalar_t x845  = x708 + x724;
    const scalar_t x846  = x701 + x845;
    const scalar_t x847  = x749 + x843;
    const scalar_t x848  = S_ikmn_canonical[26] * Wimpn_compressed[0];
    const scalar_t x849  = x742 + x845 + x848;
    const scalar_t x850  = x698 + x703;
    const scalar_t x851  = x735 + x835 + x850;
    const scalar_t x852  = x756 + x765 + x789;
    const scalar_t x853  = x781 + x852;
    const scalar_t x854  = x765 + x780;
    const scalar_t x855  = S_ikmn_canonical[30] * Wimpn_compressed[0] + x769 + x854;
    const scalar_t x856  = x774 + x797;
    const scalar_t x857  = x854 + x856;
    const scalar_t x858  = x765 + x793;
    const scalar_t x859  = x773 + x798;
    const scalar_t x860  = x858 + x859;
    const scalar_t x861  = S_ikmn_canonical[33] * Wimpn_compressed[5];
    const scalar_t x862  = S_ikmn_canonical[36] * Wimpn_compressed[5];
    const scalar_t x863  = x803 + x854 + x861 + x862;
    const scalar_t x864  = S_ikmn_canonical[33] * Wimpn_compressed[0];
    const scalar_t x865  = x768 + x819 + x854 + x864;
    const scalar_t x866  = x793 + x852;
    const scalar_t x867  = x756 + x765 + x768 + x781;
    const scalar_t x868  = S_ikmn_canonical[36] * Wimpn_compressed[0];
    const scalar_t x869  = x819 + x858 + x868;
    const scalar_t x870  = x773 + x799 + x854;
    const scalar_t x871  = x717 + x833;
    const scalar_t x872  = x704 + x706;
    const scalar_t x873  = x718 + x871;
    const scalar_t x874  = x707 + x743;
    const scalar_t x875  = x873 + x874;
    const scalar_t x876  = x706 + x873;
    const scalar_t x877  = x713 + x876;
    const scalar_t x878  = x715 + x872;
    const scalar_t x879  = x752 + x878;
    const scalar_t x880  = x736 + x842 + x879;
    const scalar_t x881  = S_ikmn_canonical[28] * Wimpn_compressed[5];
    const scalar_t x882  = x706 + x720 + x741 + x839 + x881;
    const scalar_t x883  = x751 + x879;
    const scalar_t x884  = x720 + x731 + x874;
    const scalar_t x885  = S_ikmn_canonical[28] * Wimpn_compressed[0];
    const scalar_t x886  = x736 + x876 + x885;
    const scalar_t x887  = x757 + x788;
    const scalar_t x888  = x769 + x779;
    const scalar_t x889  = x761 + x782;
    const scalar_t x890  = x888 + x889;
    const scalar_t x891  = x887 + x890;
    const scalar_t x892  = x773 + x888;
    const scalar_t x893  = x820 + x892;
    const scalar_t x894  = x765 + x893;
    const scalar_t x895  = x761 + x763;
    const scalar_t x896  = x769 + x780;
    const scalar_t x897  = x773 + x896;
    const scalar_t x898  = S_ikmn_canonical[34] * Wimpn_compressed[0] + x895 + x897;
    const scalar_t x899  = x761 + x897;
    const scalar_t x900  = x762 + x804;
    const scalar_t x901  = x899 + x900;
    const scalar_t x902  = x788 + x896;
    const scalar_t x903  = x889 + x902;
    const scalar_t x904  = S_ikmn_canonical[31] * Wimpn_compressed[0];
    const scalar_t x905  = x807 + x895;
    const scalar_t x906  = x892 + x904 + x905;
    const scalar_t x907  = S_ikmn_canonical[31] * Wimpn_compressed[5];
    const scalar_t x908  = S_ikmn_canonical[37] * Wimpn_compressed[5];
    const scalar_t x909  = x761 + x816 + x892 + x907 + x908;
    const scalar_t x910  = x757 + x763 + x773 + x890;
    const scalar_t x911  = x761 + x900;
    const scalar_t x912  = x893 + x911;
    const scalar_t x913  = S_ikmn_canonical[37] * Wimpn_compressed[0];
    const scalar_t x914  = x807 + x899 + x913;
    const scalar_t x915  = x704 + x705;
    const scalar_t x916  = x705 + x873;
    const scalar_t x917  = x727 + x916;
    const scalar_t x918  = x719 + x837 + x874;
    const scalar_t x919  = x723 + x915;
    const scalar_t x920  = x698 + x919;
    const scalar_t x921  = x722 + x726 + x920;
    const scalar_t x922  = x719 + x725 + x747 + x840 + x881;
    const scalar_t x923  = x717 + x732 + x848 + x920;
    const scalar_t x924  = x732 + x885 + x916;
    const scalar_t x925  = x764 + x783;
    const scalar_t x926  = x773 + x778;
    const scalar_t x927  = x770 + x790;
    const scalar_t x928  = x926 + x927;
    const scalar_t x929  = x925 + x928;
    const scalar_t x930  = x769 + x926;
    const scalar_t x931  = x821 + x930;
    const scalar_t x932  = x765 + x931;
    const scalar_t x933  = x764 + x897;
    const scalar_t x934  = x760 + x808;
    const scalar_t x935  = x933 + x934;
    const scalar_t x936  = S_ikmn_canonical[38] * Wimpn_compressed[0] + x759 + x933;
    const scalar_t x937  = x790 + x897 + x925;
    const scalar_t x938  = x764 + x934;
    const scalar_t x939  = x931 + x938;
    const scalar_t x940  = x759 + x764;
    const scalar_t x941  = x928 + x940;
    const scalar_t x942  = S_ikmn_canonical[32] * Wimpn_compressed[5];
    const scalar_t x943  = S_ikmn_canonical[35] * Wimpn_compressed[5];
    const scalar_t x944  = x764 + x827 + x930 + x942 + x943;
    const scalar_t x945  = S_ikmn_canonical[32] * Wimpn_compressed[0];
    const scalar_t x946  = x805 + x940;
    const scalar_t x947  = x930 + x945 + x946;
    const scalar_t x948  = S_ikmn_canonical[35] * Wimpn_compressed[0];
    const scalar_t x949  = x805 + x933 + x948;
    const scalar_t x950  = Wimpn_compressed[7] * x695;
    const scalar_t x951  = S_ikmn_canonical[27] * Wimpn_compressed[6];
    const scalar_t x952  = S_ikmn_canonical[29] * Wimpn_compressed[6];
    const scalar_t x953  = x951 + x952;
    const scalar_t x954  = S_ikmn_canonical[24] * Wimpn_compressed[6];
    const scalar_t x955  = Wimpn_compressed[7] * x694;
    const scalar_t x956  = x954 + x955;
    const scalar_t x957  = x953 + x956;
    const scalar_t x958  = Wimpn_compressed[3] * x694;
    const scalar_t x959  = S_ikmn_canonical[28] * Wimpn_compressed[8];
    const scalar_t x960  = x698 + x719;
    const scalar_t x961  = S_ikmn_canonical[27] * Wimpn_compressed[8] + x706 + x737;
    const scalar_t x962  = x958 + x959 + x960 + x961;
    const scalar_t x963  = S_ikmn_canonical[26] * Wimpn_compressed[7];
    const scalar_t x964  = S_ikmn_canonical[28] * Wimpn_compressed[7];
    const scalar_t x965  = S_ikmn_canonical[29] * Wimpn_compressed[7] + x705 + x963 + x964;
    const scalar_t x966  = x955 + x960 + x965;
    const scalar_t x967  = x718 + x752;
    const scalar_t x968  = S_ikmn_canonical[25] * Wimpn_compressed[7];
    const scalar_t x969  = S_ikmn_canonical[27] * Wimpn_compressed[7] + x706 + x964 + x968;
    const scalar_t x970  = x950 + x967 + x969;
    const scalar_t x971  = Wimpn_compressed[3] * x695;
    const scalar_t x972  = S_ikmn_canonical[29] * Wimpn_compressed[8] + x705 + x733;
    const scalar_t x973  = x959 + x967 + x971 + x972;
    const scalar_t x974  = Wimpn_compressed[3] * x696;
    const scalar_t x975  = x741 + x747 + x871 + x974;
    const scalar_t x976  = x815 + x825;
    const scalar_t x977  = x810 + x822 + x976;
    const scalar_t x978  = x765 + x769;
    const scalar_t x979  = x803 + x907 + x942 + x978;
    const scalar_t x980  = x813 + x897;
    const scalar_t x981  = x790 + x794 + x902;
    const scalar_t x982  = S_ikmn_canonical[31] * Wimpn_compressed[7];
    const scalar_t x983  = S_ikmn_canonical[36] * Wimpn_compressed[7];
    const scalar_t x984  = x982 + x983;
    const scalar_t x985  = S_ikmn_canonical[37] * Wimpn_compressed[6] + x984;
    const scalar_t x986  = S_ikmn_canonical[32] * Wimpn_compressed[7];
    const scalar_t x987  = S_ikmn_canonical[33] * Wimpn_compressed[7];
    const scalar_t x988  = x986 + x987;
    const scalar_t x989  = S_ikmn_canonical[35] * Wimpn_compressed[6] + x988;
    const scalar_t x990  = S_ikmn_canonical[30] * Wimpn_compressed[6];
    const scalar_t x991  = S_ikmn_canonical[34] * Wimpn_compressed[6];
    const scalar_t x992  = S_ikmn_canonical[38] * Wimpn_compressed[6];
    const scalar_t x993  = x991 + x992;
    const scalar_t x994  = x990 + x993;
    const scalar_t x995  = x985 + x989 + x994;
    const scalar_t x996  = S_ikmn_canonical[34] * Wimpn_compressed[8];
    const scalar_t x997  = S_ikmn_canonical[35] * Wimpn_compressed[8];
    const scalar_t x998  = x776 + x802 + x815 + x996 + x997;
    const scalar_t x999  = S_ikmn_canonical[37] * Wimpn_compressed[7];
    const scalar_t x1000 = x982 + x999;
    const scalar_t x1001 = S_ikmn_canonical[38] * Wimpn_compressed[7];
    const scalar_t x1002 = x1001 + x794;
    const scalar_t x1003 = x1000 + x1002 + x774 + x988;
    const scalar_t x1004 = S_ikmn_canonical[35] * Wimpn_compressed[7];
    const scalar_t x1005 = x1004 + x986;
    const scalar_t x1006 = x767 + x984;
    const scalar_t x1007 = S_ikmn_canonical[34] * Wimpn_compressed[7];
    const scalar_t x1008 = x1007 + x773;
    const scalar_t x1009 = x1008 + x775;
    const scalar_t x1010 = x1005 + x1006 + x1009;
    const scalar_t x1011 = S_ikmn_canonical[38] * Wimpn_compressed[8];
    const scalar_t x1012 = S_ikmn_canonical[37] * Wimpn_compressed[8] + x773;
    const scalar_t x1013 = x1011 + x1012 + x802 + x825 + x830;
    const scalar_t x1014 = x818 + x828 + x897;
    const scalar_t x1015 = S_ikmn_canonical[26] * Wimpn_compressed[8];
    const scalar_t x1016 = x707 + x718;
    const scalar_t x1017 = S_ikmn_canonical[24] * Wimpn_compressed[8] + x1016 + x744;
    const scalar_t x1018 = x1015 + x1017 + x698 + x958;
    const scalar_t x1019 = x730 + x741 + x878 + x958;
    const scalar_t x1020 = S_ikmn_canonical[24] * Wimpn_compressed[7] + x850 + x963 + x968;
    const scalar_t x1021 = S_ikmn_canonical[28] * Wimpn_compressed[6] + x1016 + x1020;
    const scalar_t x1022 = x718 + x753;
    const scalar_t x1023 = S_ikmn_canonical[26] * Wimpn_compressed[6] + x1022 + x969;
    const scalar_t x1024 = x758 + x777 + x779 + x787 + x789 + x889;
    const scalar_t x1025 = x765 + x888;
    const scalar_t x1026 = x1025 + x819 + x904;
    const scalar_t x1027 = x774 + x780 + x864 + x905;
    const scalar_t x1028 = x793 + x859 + x911;
    const scalar_t x1029 = x779 + x800;
    const scalar_t x1030 = x1029 + x761;
    const scalar_t x1031 = x1012 + x1030 + x976 + x996;
    const scalar_t x1032 = x768 + x895;
    const scalar_t x1033 = x779 + x987;
    const scalar_t x1034 = x1032 + x1033 + x982 + x990 + x991;
    const scalar_t x1035 = S_ikmn_canonical[30] * Wimpn_compressed[8];
    const scalar_t x1036 = x1035 + x815;
    const scalar_t x1037 = x762 + x826;
    const scalar_t x1038 = S_ikmn_canonical[36] * Wimpn_compressed[8] + x1037;
    const scalar_t x1039 = x1030 + x1036 + x1038;
    const scalar_t x1040 = x1029 + x1032 + x803 + x815 + x816;
    const scalar_t x1041 = x761 + x767;
    const scalar_t x1042 = S_ikmn_canonical[30] * Wimpn_compressed[7];
    const scalar_t x1043 = x1042 + x762;
    const scalar_t x1044 = x1043 + x779;
    const scalar_t x1045 = x1041 + x1044 + x985;
    const scalar_t x1046 = S_ikmn_canonical[36] * Wimpn_compressed[6] + x987;
    const scalar_t x1047 = x761 + x999;
    const scalar_t x1048 = x1008 + x1046 + x1047 + x780;
    const scalar_t x1049 = Wimpn_compressed[7] * x696;
    const scalar_t x1050 = x1020 + x1049 + x707;
    const scalar_t x1051 = x730 + x747 + x919 + x971;
    const scalar_t x1052 = x753 + x974;
    const scalar_t x1053 = x1015 + x1052 + x972;
    const scalar_t x1054 = x799 + x819;
    const scalar_t x1055 = x1054 + x800 + x805 + x808 + x816 + x821 + x826;
    const scalar_t x1056 = x756 + x778;
    const scalar_t x1057 = x1056 + x771;
    const scalar_t x1058 = x776 + x780 + x816 + x861 + x943;
    const scalar_t x1059 = x773 + x793;
    const scalar_t x1060 = x1059 + x791 + x940;
    const scalar_t x1061 = x1001 + x1004 + x764 + x773;
    const scalar_t x1062 = x1033 + x1061 + x984;
    const scalar_t x1063 = x760 + x768;
    const scalar_t x1064 = x1063 + x764;
    const scalar_t x1065 = S_ikmn_canonical[32] * Wimpn_compressed[8] + x800 + x817;
    const scalar_t x1066 = x1036 + x1064 + x1065;
    const scalar_t x1067 = S_ikmn_canonical[32] * Wimpn_compressed[6] + x1004;
    const scalar_t x1068 = x1000 + x1046 + x1067 + x994;
    const scalar_t x1069 = x1004 + x999;
    const scalar_t x1070 = x1043 + x1063 + x1069 + x778 + x988;
    const scalar_t x1071 = x801 + x827;
    const scalar_t x1072 = x778 + x825;
    const scalar_t x1073 = x767 + x940;
    const scalar_t x1074 = x1071 + x1072 + x1073 + x803;
    const scalar_t x1075 = x1011 + x773;
    const scalar_t x1076 = x780 + x817;
    const scalar_t x1077 = x1038 + x1075 + x1076 + x815;
    const scalar_t x1078 = x950 + x954;
    const scalar_t x1079 = x1049 + x953;
    const scalar_t x1080 = S_ikmn_canonical[25] * Wimpn_compressed[8];
    const scalar_t x1081 = x1017 + x1080 + x703 + x971;
    const scalar_t x1082 = x1052 + x1080 + x961;
    const scalar_t x1083 = x1054 + x1071 + x804 + x807 + x817 + x820;
    const scalar_t x1084 = x1025 + x758;
    const scalar_t x1085 = x763 + x774 + x781 + x889;
    const scalar_t x1086 = x773 + x795 + x827 + x862 + x908;
    const scalar_t x1087 = x1007 + x1047 + x926 + x983 + x988;
    const scalar_t x1088 = x1006 + x1044 + x1069 + x760;
    const scalar_t x1089 = S_ikmn_canonical[33] * Wimpn_compressed[6] + x983;
    const scalar_t x1090 = S_ikmn_canonical[31] * Wimpn_compressed[6] + x999;
    const scalar_t x1091 = x1005 + x1089 + x1090 + x994;
    const scalar_t x1092 = S_ikmn_canonical[31] * Wimpn_compressed[8];
    const scalar_t x1093 = x1035 + x801;
    const scalar_t x1094 = x1037 + x1041 + x1092 + x1093 + x825;
    const scalar_t x1095 = S_ikmn_canonical[33] * Wimpn_compressed[8] + x760;
    const scalar_t x1096 = x773 + x826 + x996;
    const scalar_t x1097 = x1076 + x1095 + x1096 + x825;
    const scalar_t x1098 = S_ikmn_canonical[25] * Wimpn_compressed[6] + x1022 + x965;
    const scalar_t x1099 = x1056 + x766 + x777 + x791 + x925;
    const scalar_t x1100 = x778 + x819 + x945 + x978;
    const scalar_t x1101 = x780 + x856 + x938;
    const scalar_t x1102 = x1059 + x868 + x946;
    const scalar_t x1103 = x1011 + x764 + x801 + x926 + x976 + x997;
    const scalar_t x1104 = x1042 + x1064 + x778 + x989;
    const scalar_t x1105 = x1072 + x1093 + x1095 + x764 + x817;
    const scalar_t x1106 = x1073 + x778 + x983 + x986 + x990 + x992;
    const scalar_t x1107 = x1061 + x1089 + x780;
    const scalar_t x1108 = x784 + x887 + x927;
    const scalar_t x1109 = x773 + x822 + x978;
    const scalar_t x1110 = x775 + x807 + x897 + x948;
    const scalar_t x1111 = x794 + x805 + x897 + x913;
    const scalar_t x1112 = x1009 + x1067 + x769 + x982;
    const scalar_t x1113 = x1037 + x1065 + x1075 + x769;
    const scalar_t x1114 = x1092 + x1096 + x760 + x769 + x801 + x817;
    const scalar_t x1115 = x1002 + x1090 + x769 + x773 + x986;
    const scalar_t x1116 = x1069 + x897 + x993;
    const scalar_t x1117 = 2 * S_ikmn_canonical[40];
    const scalar_t x1118 = 2 * S_ikmn_canonical[41];
    const scalar_t x1119 = 2 * S_ikmn_canonical[43];
    const scalar_t x1120 = S_ikmn_canonical[40] * Wimpn_compressed[1];
    const scalar_t x1121 = S_ikmn_canonical[41] * Wimpn_compressed[2];
    const scalar_t x1122 = S_ikmn_canonical[39] * Wimpn_compressed[1];
    const scalar_t x1123 = S_ikmn_canonical[41] * Wimpn_compressed[1];
    const scalar_t x1124 = x1122 + x1123;
    const scalar_t x1125 = x1121 + x1124;
    const scalar_t x1126 = S_ikmn_canonical[40] * Wimpn_compressed[2];
    const scalar_t x1127 = Wimpn_compressed[2] * x1119;
    const scalar_t x1128 = S_ikmn_canonical[42] * Wimpn_compressed[2];
    const scalar_t x1129 = S_ikmn_canonical[44] * Wimpn_compressed[2];
    const scalar_t x1130 = x1128 + x1129;
    const scalar_t x1131 = x1127 + x1130;
    const scalar_t x1132 = x1126 + x1131;
    const scalar_t x1133 = x1120 + x1125 + x1132;
    const scalar_t x1134 = S_ikmn_canonical[42] * Wimpn_compressed[1];
    const scalar_t x1135 = S_ikmn_canonical[43] * Wimpn_compressed[1];
    const scalar_t x1136 = x1134 + x1135;
    const scalar_t x1137 = x1129 + x1136;
    const scalar_t x1138 = Wimpn_compressed[2] * x1118;
    const scalar_t x1139 = x1126 + x1138;
    const scalar_t x1140 = S_ikmn_canonical[39] * Wimpn_compressed[2];
    const scalar_t x1141 = S_ikmn_canonical[43] * Wimpn_compressed[2];
    const scalar_t x1142 = x1140 + x1141;
    const scalar_t x1143 = x1139 + x1142;
    const scalar_t x1144 = x1120 + x1137 + x1143;
    const scalar_t x1145 = x1123 + x1140;
    const scalar_t x1146 = Wimpn_compressed[2] * x1117;
    const scalar_t x1147 = x1121 + x1146;
    const scalar_t x1148 = x1128 + x1147;
    const scalar_t x1149 = S_ikmn_canonical[44] * Wimpn_compressed[1];
    const scalar_t x1150 = x1135 + x1149;
    const scalar_t x1151 = x1141 + x1150;
    const scalar_t x1152 = x1145 + x1148 + x1151;
    const scalar_t x1153 = S_ikmn_canonical[39] * Wimpn_compressed[3];
    const scalar_t x1154 = S_ikmn_canonical[40] * Wimpn_compressed[4];
    const scalar_t x1155 = S_ikmn_canonical[44] * Wimpn_compressed[4];
    const scalar_t x1156 = S_ikmn_canonical[40] * Wimpn_compressed[3];
    const scalar_t x1157 = x1154 + x1155 + x1156;
    const scalar_t x1158 = S_ikmn_canonical[41] * Wimpn_compressed[4];
    const scalar_t x1159 = S_ikmn_canonical[42] * Wimpn_compressed[4];
    const scalar_t x1160 = S_ikmn_canonical[41] * Wimpn_compressed[3];
    const scalar_t x1161 = x1158 + x1159 + x1160;
    const scalar_t x1162 = Wimpn_compressed[4] * x1119 + x1153 + x1157 + x1161;
    const scalar_t x1163 = Wimpn_compressed[1] * x1117 + x1125 + x1137 + x1141;
    const scalar_t x1164 = S_ikmn_canonical[42] * Wimpn_compressed[3];
    const scalar_t x1165 = S_ikmn_canonical[39] * Wimpn_compressed[4];
    const scalar_t x1166 = S_ikmn_canonical[43] * Wimpn_compressed[4];
    const scalar_t x1167 = S_ikmn_canonical[43] * Wimpn_compressed[3];
    const scalar_t x1168 = x1165 + x1166 + x1167;
    const scalar_t x1169 = Wimpn_compressed[4] * x1118 + x1157 + x1164 + x1168;
    const scalar_t x1170 = S_ikmn_canonical[44] * Wimpn_compressed[3];
    const scalar_t x1171 = Wimpn_compressed[4] * x1117 + x1161 + x1168 + x1170;
    const scalar_t x1172 = x1120 + x1122;
    const scalar_t x1173 = Wimpn_compressed[1] * x1118 + x1126 + x1128 + x1151 + x1172;
    const scalar_t x1174 = x1120 + x1134;
    const scalar_t x1175 = x1126 + x1140;
    const scalar_t x1176 = x1121 + x1175;
    const scalar_t x1177 = Wimpn_compressed[1] * x1119 + x1123 + x1149 + x1174 + x1176;
    const scalar_t x1178 = x1138 + x1146;
    const scalar_t x1179 = x1131 + x1140;
    const scalar_t x1180 = x1154 + x1179;
    const scalar_t x1181 = x1139 + x1180;
    const scalar_t x1182 = x1147 + x1158;
    const scalar_t x1183 = x1179 + x1182;
    const scalar_t x1184 = S_ikmn_canonical[40] * Wimpn_compressed[5];
    const scalar_t x1185 = S_ikmn_canonical[41] * Wimpn_compressed[5];
    const scalar_t x1186 = x1121 + x1132 + x1153 + x1184 + x1185;
    const scalar_t x1187 = S_ikmn_canonical[40] * Wimpn_compressed[0];
    const scalar_t x1188 = x1131 + x1139;
    const scalar_t x1189 = x1165 + x1187 + x1188;
    const scalar_t x1190 = x1131 + x1147;
    const scalar_t x1191 = x1124 + x1190;
    const scalar_t x1192 = x1172 + x1188;
    const scalar_t x1193 = S_ikmn_canonical[41] * Wimpn_compressed[0];
    const scalar_t x1194 = x1165 + x1190 + x1193;
    const scalar_t x1195 = x1121 + x1126;
    const scalar_t x1196 = x1158 + x1180 + x1195;
    const scalar_t x1197 = x1140 + x1178;
    const scalar_t x1198 = x1127 + x1129;
    const scalar_t x1199 = x1141 + x1197;
    const scalar_t x1200 = x1130 + x1166;
    const scalar_t x1201 = x1199 + x1200;
    const scalar_t x1202 = x1129 + x1199;
    const scalar_t x1203 = x1136 + x1202;
    const scalar_t x1204 = x1138 + x1198;
    const scalar_t x1205 = x1175 + x1204;
    const scalar_t x1206 = x1159 + x1187 + x1205;
    const scalar_t x1207 = S_ikmn_canonical[43] * Wimpn_compressed[5];
    const scalar_t x1208 = x1129 + x1143 + x1164 + x1184 + x1207;
    const scalar_t x1209 = x1174 + x1205;
    const scalar_t x1210 = x1143 + x1154 + x1200;
    const scalar_t x1211 = S_ikmn_canonical[43] * Wimpn_compressed[0];
    const scalar_t x1212 = x1159 + x1202 + x1211;
    const scalar_t x1213 = x1127 + x1128;
    const scalar_t x1214 = x1128 + x1199;
    const scalar_t x1215 = x1150 + x1214;
    const scalar_t x1216 = x1142 + x1182 + x1200;
    const scalar_t x1217 = x1146 + x1213;
    const scalar_t x1218 = x1121 + x1217;
    const scalar_t x1219 = x1145 + x1149 + x1218;
    const scalar_t x1220 = x1142 + x1148 + x1170 + x1185 + x1207;
    const scalar_t x1221 = x1140 + x1155 + x1193 + x1218;
    const scalar_t x1222 = x1155 + x1211 + x1214;
    const scalar_t x1223 = Wimpn_compressed[7] * x1118;
    const scalar_t x1224 = S_ikmn_canonical[42] * Wimpn_compressed[6];
    const scalar_t x1225 = S_ikmn_canonical[44] * Wimpn_compressed[6];
    const scalar_t x1226 = x1224 + x1225;
    const scalar_t x1227 = S_ikmn_canonical[39] * Wimpn_compressed[6];
    const scalar_t x1228 = Wimpn_compressed[7] * x1117;
    const scalar_t x1229 = x1227 + x1228;
    const scalar_t x1230 = x1226 + x1229;
    const scalar_t x1231 = Wimpn_compressed[3] * x1117;
    const scalar_t x1232 = S_ikmn_canonical[43] * Wimpn_compressed[8];
    const scalar_t x1233 = x1121 + x1142;
    const scalar_t x1234 = S_ikmn_canonical[42] * Wimpn_compressed[8] + x1129 + x1160;
    const scalar_t x1235 = x1231 + x1232 + x1233 + x1234;
    const scalar_t x1236 = S_ikmn_canonical[41] * Wimpn_compressed[7];
    const scalar_t x1237 = S_ikmn_canonical[43] * Wimpn_compressed[7];
    const scalar_t x1238 = S_ikmn_canonical[44] * Wimpn_compressed[7] + x1128 + x1236 + x1237;
    const scalar_t x1239 = x1228 + x1233 + x1238;
    const scalar_t x1240 = x1141 + x1175;
    const scalar_t x1241 = S_ikmn_canonical[40] * Wimpn_compressed[7];
    const scalar_t x1242 = S_ikmn_canonical[42] * Wimpn_compressed[7] + x1129 + x1237 + x1241;
    const scalar_t x1243 = x1223 + x1240 + x1242;
    const scalar_t x1244 = Wimpn_compressed[3] * x1118;
    const scalar_t x1245 = S_ikmn_canonical[44] * Wimpn_compressed[8] + x1128 + x1156;
    const scalar_t x1246 = x1232 + x1240 + x1244 + x1245;
    const scalar_t x1247 = Wimpn_compressed[3] * x1119;
    const scalar_t x1248 = x1164 + x1170 + x1197 + x1247;
    const scalar_t x1249 = S_ikmn_canonical[41] * Wimpn_compressed[8];
    const scalar_t x1250 = x1130 + x1141;
    const scalar_t x1251 = S_ikmn_canonical[39] * Wimpn_compressed[8] + x1167 + x1250;
    const scalar_t x1252 = x1121 + x1231 + x1249 + x1251;
    const scalar_t x1253 = x1153 + x1164 + x1204 + x1231;
    const scalar_t x1254 = S_ikmn_canonical[39] * Wimpn_compressed[7] + x1195 + x1236 + x1241;
    const scalar_t x1255 = S_ikmn_canonical[43] * Wimpn_compressed[6] + x1250 + x1254;
    const scalar_t x1256 = x1141 + x1176;
    const scalar_t x1257 = S_ikmn_canonical[41] * Wimpn_compressed[6] + x1242 + x1256;
    const scalar_t x1258 = Wimpn_compressed[7] * x1119;
    const scalar_t x1259 = x1130 + x1254 + x1258;
    const scalar_t x1260 = x1153 + x1170 + x1217 + x1244;
    const scalar_t x1261 = x1176 + x1247;
    const scalar_t x1262 = x1245 + x1249 + x1261;
    const scalar_t x1263 = x1223 + x1227;
    const scalar_t x1264 = x1226 + x1258;
    const scalar_t x1265 = S_ikmn_canonical[40] * Wimpn_compressed[8];
    const scalar_t x1266 = x1126 + x1244 + x1251 + x1265;
    const scalar_t x1267 = x1234 + x1261 + x1265;
    const scalar_t x1268 = S_ikmn_canonical[40] * Wimpn_compressed[6] + x1238 + x1256;
    H[0]                 = Wimpn_compressed[0] * (S_ikmn_canonical[0] + S_ikmn_canonical[3] + S_ikmn_canonical[5] + x0 + x1 + x2);
    H[1]                 = x16;
    H[2]                 = x27;
    H[3]                 = x35;
    H[4]                 = x45;
    H[5]                 = x46;
    H[6]                 = x52;
    H[7]                 = x54;
    H[8]                 = x56;
    H[9]                 = x60;
    H[10]                = x61;
    H[11]                = x78;
    H[12]                = x92;
    H[13]                = x102;
    H[14]                = x117;
    H[15]                = x120;
    H[16]                = x130;
    H[17]                = x135;
    H[18]                = x137;
    H[19]                = x138;
    H[20]                = x139;
    H[21]                = x156;
    H[22]                = x170;
    H[23]                = x180;
    H[24]                = x195;
    H[25]                = x198;
    H[26]                = x208;
    H[27]                = x213;
    H[28]                = x215;
    H[29]                = x216;
    H[30]                = x16;
    H[31]                = S_ikmn_canonical[0] * Wimpn_compressed[0] + x14 + x217;
    H[32]                = x220;
    H[33]                = x222;
    H[34]                = x225;
    H[35]                = x228;
    H[36]                = x230;
    H[37]                = x231;
    H[38]                = x233;
    H[39]                = x235;
    H[40]                = x237;
    H[41]                = x239;
    H[42]                = x241;
    H[43]                = x244;
    H[44]                = x247;
    H[45]                = x249;
    H[46]                = x250;
    H[47]                = x251;
    H[48]                = x253;
    H[49]                = x254;
    H[50]                = x256;
    H[51]                = x258;
    H[52]                = x260;
    H[53]                = x263;
    H[54]                = x266;
    H[55]                = x268;
    H[56]                = x269;
    H[57]                = x270;
    H[58]                = x272;
    H[59]                = x273;
    H[60]                = x27;
    H[61]                = x220;
    H[62]                = S_ikmn_canonical[3] * Wimpn_compressed[0] + x274 + x275;
    H[63]                = x278;
    H[64]                = x280;
    H[65]                = x283;
    H[66]                = x285;
    H[67]                = x286;
    H[68]                = x287;
    H[69]                = x289;
    H[70]                = x294;
    H[71]                = x297;
    H[72]                = x301;
    H[73]                = x304;
    H[74]                = x306;
    H[75]                = x309;
    H[76]                = x312;
    H[77]                = x313;
    H[78]                = x315;
    H[79]                = x317;
    H[80]                = x322;
    H[81]                = x325;
    H[82]                = x329;
    H[83]                = x332;
    H[84]                = x334;
    H[85]                = x337;
    H[86]                = x340;
    H[87]                = x341;
    H[88]                = x343;
    H[89]                = x345;
    H[90]                = x35;
    H[91]                = x222;
    H[92]                = x278;
    H[93]                = S_ikmn_canonical[5] * Wimpn_compressed[0] + x274 + x346;
    H[94]                = x348;
    H[95]                = x349;
    H[96]                = x352;
    H[97]                = x353;
    H[98]                = x354;
    H[99]                = x355;
    H[100]               = x360;
    H[101]               = x363;
    H[102]               = x366;
    H[103]               = x367;
    H[104]               = x368;
    H[105]               = x370;
    H[106]               = x372;
    H[107]               = x375;
    H[108]               = x378;
    H[109]               = x380;
    H[110]               = x385;
    H[111]               = x388;
    H[112]               = x391;
    H[113]               = x392;
    H[114]               = x393;
    H[115]               = x395;
    H[116]               = x397;
    H[117]               = x400;
    H[118]               = x403;
    H[119]               = x405;
    H[120]               = x45;
    H[121]               = x225;
    H[122]               = x280;
    H[123]               = x348;
    H[124]               = Wimpn_compressed[6] * x2 + x406 + x413;
    H[125]               = x418;
    H[126]               = x422;
    H[127]               = x426;
    H[128]               = x429;
    H[129]               = x431;
    H[130]               = x433;
    H[131]               = x435;
    H[132]               = x436;
    H[133]               = x437;
    H[134]               = x451;
    H[135]               = x454;
    H[136]               = x459;
    H[137]               = x466;
    H[138]               = x469;
    H[139]               = x470;
    H[140]               = x472;
    H[141]               = x474;
    H[142]               = x475;
    H[143]               = x476;
    H[144]               = x490;
    H[145]               = x493;
    H[146]               = x498;
    H[147]               = x505;
    H[148]               = x508;
    H[149]               = x509;
    H[150]               = x46;
    H[151]               = x228;
    H[152]               = x283;
    H[153]               = x349;
    H[154]               = x418;
    H[155]               = x281 + x407 + x412;
    H[156]               = x513;
    H[157]               = x514;
    H[158]               = x516;
    H[159]               = x518;
    H[160]               = x519;
    H[161]               = x521;
    H[162]               = x522;
    H[163]               = x523;
    H[164]               = x526;
    H[165]               = x529;
    H[166]               = x534;
    H[167]               = x535;
    H[168]               = x540;
    H[169]               = x543;
    H[170]               = x544;
    H[171]               = x546;
    H[172]               = x547;
    H[173]               = x548;
    H[174]               = x551;
    H[175]               = x554;
    H[176]               = x559;
    H[177]               = x560;
    H[178]               = x565;
    H[179]               = x568;
    H[180]               = x52;
    H[181]               = x230;
    H[182]               = x285;
    H[183]               = x352;
    H[184]               = x422;
    H[185]               = x513;
    H[186]               = Wimpn_compressed[6] * x1 + x413 + x569;
    H[187]               = x570;
    H[188]               = x571;
    H[189]               = x573;
    H[190]               = x575;
    H[191]               = x577;
    H[192]               = x578;
    H[193]               = x580;
    H[194]               = x582;
    H[195]               = x586;
    H[196]               = x588;
    H[197]               = x590;
    H[198]               = x594;
    H[199]               = x597;
    H[200]               = x599;
    H[201]               = x601;
    H[202]               = x602;
    H[203]               = x604;
    H[204]               = x606;
    H[205]               = x610;
    H[206]               = x612;
    H[207]               = x614;
    H[208]               = x618;
    H[209]               = x621;
    H[210]               = x54;
    H[211]               = x231;
    H[212]               = x286;
    H[213]               = x353;
    H[214]               = x426;
    H[215]               = x514;
    H[216]               = x570;
    H[217]               = Wimpn_compressed[6] * x0 + x622 + x623;
    H[218]               = x625;
    H[219]               = x626;
    H[220]               = x627;
    H[221]               = x628;
    H[222]               = x629;
    H[223]               = x630;
    H[224]               = x631;
    H[225]               = x535;
    H[226]               = x632;
    H[227]               = x635;
    H[228]               = x638;
    H[229]               = x641;
    H[230]               = x642;
    H[231]               = x643;
    H[232]               = x644;
    H[233]               = x645;
    H[234]               = x646;
    H[235]               = x560;
    H[236]               = x647;
    H[237]               = x650;
    H[238]               = x653;
    H[239]               = x656;
    H[240]               = x56;
    H[241]               = x233;
    H[242]               = x287;
    H[243]               = x354;
    H[244]               = x429;
    H[245]               = x516;
    H[246]               = x571;
    H[247]               = x625;
    H[248]               = x350 + x408 + x622;
    H[249]               = x657;
    H[250]               = x658;
    H[251]               = x659;
    H[252]               = x660;
    H[253]               = x661;
    H[254]               = x662;
    H[255]               = x663;
    H[256]               = x594;
    H[257]               = x664;
    H[258]               = x665;
    H[259]               = x666;
    H[260]               = x667;
    H[261]               = x668;
    H[262]               = x669;
    H[263]               = x670;
    H[264]               = x671;
    H[265]               = x672;
    H[266]               = x618;
    H[267]               = x673;
    H[268]               = x674;
    H[269]               = x675;
    H[270]               = x60;
    H[271]               = x235;
    H[272]               = x289;
    H[273]               = x355;
    H[274]               = x431;
    H[275]               = x518;
    H[276]               = x573;
    H[277]               = x626;
    H[278]               = x657;
    H[279]               = x274 + x623;
    H[280]               = x676;
    H[281]               = x677;
    H[282]               = x678;
    H[283]               = x679;
    H[284]               = x470;
    H[285]               = x680;
    H[286]               = x681;
    H[287]               = x682;
    H[288]               = x683;
    H[289]               = x684;
    H[290]               = x685;
    H[291]               = x686;
    H[292]               = x687;
    H[293]               = x688;
    H[294]               = x509;
    H[295]               = x689;
    H[296]               = x690;
    H[297]               = x691;
    H[298]               = x692;
    H[299]               = x693;
    H[300]               = x61;
    H[301]               = x237;
    H[302]               = x294;
    H[303]               = x360;
    H[304]               = x433;
    H[305]               = x519;
    H[306]               = x575;
    H[307]               = x627;
    H[308]               = x658;
    H[309]               = x676;
    H[310] = Wimpn_compressed[0] * (S_ikmn_canonical[24] + S_ikmn_canonical[27] + S_ikmn_canonical[29] + x694 + x695 + x696);
    H[311] = x710;
    H[312] = x721;
    H[313] = x729;
    H[314] = x739;
    H[315] = x740;
    H[316] = x746;
    H[317] = x748;
    H[318] = x750;
    H[319] = x754;
    H[320] = x755;
    H[321] = x772;
    H[322] = x786;
    H[323] = x796;
    H[324] = x811;
    H[325] = x814;
    H[326] = x824;
    H[327] = x829;
    H[328] = x831;
    H[329] = x832;
    H[330] = x78;
    H[331] = x239;
    H[332] = x297;
    H[333] = x363;
    H[334] = x435;
    H[335] = x521;
    H[336] = x577;
    H[337] = x628;
    H[338] = x659;
    H[339] = x677;
    H[340] = x710;
    H[341] = S_ikmn_canonical[24] * Wimpn_compressed[0] + x708 + x833;
    H[342] = x836;
    H[343] = x838;
    H[344] = x841;
    H[345] = x844;
    H[346] = x846;
    H[347] = x847;
    H[348] = x849;
    H[349] = x851;
    H[350] = x853;
    H[351] = x855;
    H[352] = x857;
    H[353] = x860;
    H[354] = x863;
    H[355] = x865;
    H[356] = x866;
    H[357] = x867;
    H[358] = x869;
    H[359] = x870;
    H[360] = x92;
    H[361] = x241;
    H[362] = x301;
    H[363] = x366;
    H[364] = x436;
    H[365] = x522;
    H[366] = x578;
    H[367] = x629;
    H[368] = x660;
    H[369] = x678;
    H[370] = x721;
    H[371] = x836;
    H[372] = S_ikmn_canonical[27] * Wimpn_compressed[0] + x871 + x872;
    H[373] = x875;
    H[374] = x877;
    H[375] = x880;
    H[376] = x882;
    H[377] = x883;
    H[378] = x884;
    H[379] = x886;
    H[380] = x891;
    H[381] = x894;
    H[382] = x898;
    H[383] = x901;
    H[384] = x903;
    H[385] = x906;
    H[386] = x909;
    H[387] = x910;
    H[388] = x912;
    H[389] = x914;
    H[390] = x102;
    H[391] = x244;
    H[392] = x304;
    H[393] = x367;
    H[394] = x437;
    H[395] = x523;
    H[396] = x580;
    H[397] = x630;
    H[398] = x661;
    H[399] = x679;
    H[400] = x729;
    H[401] = x838;
    H[402] = x875;
    H[403] = S_ikmn_canonical[29] * Wimpn_compressed[0] + x871 + x915;
    H[404] = x917;
    H[405] = x918;
    H[406] = x921;
    H[407] = x922;
    H[408] = x923;
    H[409] = x924;
    H[410] = x929;
    H[411] = x932;
    H[412] = x935;
    H[413] = x936;
    H[414] = x937;
    H[415] = x939;
    H[416] = x941;
    H[417] = x944;
    H[418] = x947;
    H[419] = x949;
    H[420] = x117;
    H[421] = x247;
    H[422] = x306;
    H[423] = x368;
    H[424] = x451;
    H[425] = x526;
    H[426] = x582;
    H[427] = x631;
    H[428] = x662;
    H[429] = x470;
    H[430] = x739;
    H[431] = x841;
    H[432] = x877;
    H[433] = x917;
    H[434] = Wimpn_compressed[6] * x696 + x950 + x957;
    H[435] = x962;
    H[436] = x966;
    H[437] = x970;
    H[438] = x973;
    H[439] = x975;
    H[440] = x977;
    H[441] = x979;
    H[442] = x980;
    H[443] = x981;
    H[444] = x995;
    H[445] = x998;
    H[446] = x1003;
    H[447] = x1010;
    H[448] = x1013;
    H[449] = x1014;
    H[450] = x120;
    H[451] = x249;
    H[452] = x309;
    H[453] = x370;
    H[454] = x454;
    H[455] = x529;
    H[456] = x586;
    H[457] = x535;
    H[458] = x663;
    H[459] = x680;
    H[460] = x740;
    H[461] = x844;
    H[462] = x880;
    H[463] = x918;
    H[464] = x962;
    H[465] = x878 + x951 + x956;
    H[466] = x1018;
    H[467] = x1019;
    H[468] = x1021;
    H[469] = x1023;
    H[470] = x1024;
    H[471] = x1026;
    H[472] = x1027;
    H[473] = x1028;
    H[474] = x1031;
    H[475] = x1034;
    H[476] = x1039;
    H[477] = x1040;
    H[478] = x1045;
    H[479] = x1048;
    H[480] = x130;
    H[481] = x250;
    H[482] = x312;
    H[483] = x372;
    H[484] = x459;
    H[485] = x534;
    H[486] = x588;
    H[487] = x632;
    H[488] = x594;
    H[489] = x681;
    H[490] = x746;
    H[491] = x846;
    H[492] = x882;
    H[493] = x921;
    H[494] = x966;
    H[495] = x1018;
    H[496] = Wimpn_compressed[6] * x695 + x1049 + x957;
    H[497] = x1050;
    H[498] = x1051;
    H[499] = x1053;
    H[500] = x1055;
    H[501] = x1057;
    H[502] = x1058;
    H[503] = x1060;
    H[504] = x1062;
    H[505] = x1066;
    H[506] = x1068;
    H[507] = x1070;
    H[508] = x1074;
    H[509] = x1077;
    H[510] = x135;
    H[511] = x251;
    H[512] = x313;
    H[513] = x375;
    H[514] = x466;
    H[515] = x535;
    H[516] = x590;
    H[517] = x635;
    H[518] = x664;
    H[519] = x682;
    H[520] = x748;
    H[521] = x847;
    H[522] = x883;
    H[523] = x922;
    H[524] = x970;
    H[525] = x1019;
    H[526] = x1050;
    H[527] = Wimpn_compressed[6] * x694 + x1078 + x1079;
    H[528] = x1081;
    H[529] = x1082;
    H[530] = x1083;
    H[531] = x1084;
    H[532] = x1085;
    H[533] = x1086;
    H[534] = x1087;
    H[535] = x1040;
    H[536] = x1088;
    H[537] = x1091;
    H[538] = x1094;
    H[539] = x1097;
    H[540] = x137;
    H[541] = x253;
    H[542] = x315;
    H[543] = x378;
    H[544] = x469;
    H[545] = x540;
    H[546] = x594;
    H[547] = x638;
    H[548] = x665;
    H[549] = x683;
    H[550] = x750;
    H[551] = x849;
    H[552] = x884;
    H[553] = x923;
    H[554] = x973;
    H[555] = x1021;
    H[556] = x1051;
    H[557] = x1081;
    H[558] = x1078 + x919 + x952;
    H[559] = x1098;
    H[560] = x1099;
    H[561] = x1100;
    H[562] = x1101;
    H[563] = x1102;
    H[564] = x1103;
    H[565] = x1104;
    H[566] = x1074;
    H[567] = x1105;
    H[568] = x1106;
    H[569] = x1107;
    H[570] = x138;
    H[571] = x254;
    H[572] = x317;
    H[573] = x380;
    H[574] = x470;
    H[575] = x543;
    H[576] = x597;
    H[577] = x641;
    H[578] = x666;
    H[579] = x684;
    H[580] = x754;
    H[581] = x851;
    H[582] = x886;
    H[583] = x924;
    H[584] = x975;
    H[585] = x1023;
    H[586] = x1053;
    H[587] = x1082;
    H[588] = x1098;
    H[589] = x1079 + x871;
    H[590] = x1108;
    H[591] = x1109;
    H[592] = x1110;
    H[593] = x1111;
    H[594] = x1014;
    H[595] = x1112;
    H[596] = x1113;
    H[597] = x1114;
    H[598] = x1115;
    H[599] = x1116;
    H[600] = x139;
    H[601] = x256;
    H[602] = x322;
    H[603] = x385;
    H[604] = x472;
    H[605] = x544;
    H[606] = x599;
    H[607] = x642;
    H[608] = x667;
    H[609] = x685;
    H[610] = x755;
    H[611] = x853;
    H[612] = x891;
    H[613] = x929;
    H[614] = x977;
    H[615] = x1024;
    H[616] = x1055;
    H[617] = x1083;
    H[618] = x1099;
    H[619] = x1108;
    H[620] = Wimpn_compressed[0] * (S_ikmn_canonical[39] + S_ikmn_canonical[42] + S_ikmn_canonical[44] + x1117 + x1118 + x1119);
    H[621] = x1133;
    H[622] = x1144;
    H[623] = x1152;
    H[624] = x1162;
    H[625] = x1163;
    H[626] = x1169;
    H[627] = x1171;
    H[628] = x1173;
    H[629] = x1177;
    H[630] = x156;
    H[631] = x258;
    H[632] = x325;
    H[633] = x388;
    H[634] = x474;
    H[635] = x546;
    H[636] = x601;
    H[637] = x643;
    H[638] = x668;
    H[639] = x686;
    H[640] = x772;
    H[641] = x855;
    H[642] = x894;
    H[643] = x932;
    H[644] = x979;
    H[645] = x1026;
    H[646] = x1057;
    H[647] = x1084;
    H[648] = x1100;
    H[649] = x1109;
    H[650] = x1133;
    H[651] = S_ikmn_canonical[39] * Wimpn_compressed[0] + x1131 + x1178;
    H[652] = x1181;
    H[653] = x1183;
    H[654] = x1186;
    H[655] = x1189;
    H[656] = x1191;
    H[657] = x1192;
    H[658] = x1194;
    H[659] = x1196;
    H[660] = x170;
    H[661] = x260;
    H[662] = x329;
    H[663] = x391;
    H[664] = x475;
    H[665] = x547;
    H[666] = x602;
    H[667] = x644;
    H[668] = x669;
    H[669] = x687;
    H[670] = x786;
    H[671] = x857;
    H[672] = x898;
    H[673] = x935;
    H[674] = x980;
    H[675] = x1027;
    H[676] = x1058;
    H[677] = x1085;
    H[678] = x1101;
    H[679] = x1110;
    H[680] = x1144;
    H[681] = x1181;
    H[682] = S_ikmn_canonical[42] * Wimpn_compressed[0] + x1197 + x1198;
    H[683] = x1201;
    H[684] = x1203;
    H[685] = x1206;
    H[686] = x1208;
    H[687] = x1209;
    H[688] = x1210;
    H[689] = x1212;
    H[690] = x180;
    H[691] = x263;
    H[692] = x332;
    H[693] = x392;
    H[694] = x476;
    H[695] = x548;
    H[696] = x604;
    H[697] = x645;
    H[698] = x670;
    H[699] = x688;
    H[700] = x796;
    H[701] = x860;
    H[702] = x901;
    H[703] = x936;
    H[704] = x981;
    H[705] = x1028;
    H[706] = x1060;
    H[707] = x1086;
    H[708] = x1102;
    H[709] = x1111;
    H[710] = x1152;
    H[711] = x1183;
    H[712] = x1201;
    H[713] = S_ikmn_canonical[44] * Wimpn_compressed[0] + x1197 + x1213;
    H[714] = x1215;
    H[715] = x1216;
    H[716] = x1219;
    H[717] = x1220;
    H[718] = x1221;
    H[719] = x1222;
    H[720] = x195;
    H[721] = x266;
    H[722] = x334;
    H[723] = x393;
    H[724] = x490;
    H[725] = x551;
    H[726] = x606;
    H[727] = x646;
    H[728] = x671;
    H[729] = x509;
    H[730] = x811;
    H[731] = x863;
    H[732] = x903;
    H[733] = x937;
    H[734] = x995;
    H[735] = x1031;
    H[736] = x1062;
    H[737] = x1087;
    H[738] = x1103;
    H[739] = x1014;
    H[740] = x1162;
    H[741] = x1186;
    H[742] = x1203;
    H[743] = x1215;
    H[744] = Wimpn_compressed[6] * x1119 + x1223 + x1230;
    H[745] = x1235;
    H[746] = x1239;
    H[747] = x1243;
    H[748] = x1246;
    H[749] = x1248;
    H[750] = x198;
    H[751] = x268;
    H[752] = x337;
    H[753] = x395;
    H[754] = x493;
    H[755] = x554;
    H[756] = x610;
    H[757] = x560;
    H[758] = x672;
    H[759] = x689;
    H[760] = x814;
    H[761] = x865;
    H[762] = x906;
    H[763] = x939;
    H[764] = x998;
    H[765] = x1034;
    H[766] = x1066;
    H[767] = x1040;
    H[768] = x1104;
    H[769] = x1112;
    H[770] = x1163;
    H[771] = x1189;
    H[772] = x1206;
    H[773] = x1216;
    H[774] = x1235;
    H[775] = x1204 + x1224 + x1229;
    H[776] = x1252;
    H[777] = x1253;
    H[778] = x1255;
    H[779] = x1257;
    H[780] = x208;
    H[781] = x269;
    H[782] = x340;
    H[783] = x397;
    H[784] = x498;
    H[785] = x559;
    H[786] = x612;
    H[787] = x647;
    H[788] = x618;
    H[789] = x690;
    H[790] = x824;
    H[791] = x866;
    H[792] = x909;
    H[793] = x941;
    H[794] = x1003;
    H[795] = x1039;
    H[796] = x1068;
    H[797] = x1088;
    H[798] = x1074;
    H[799] = x1113;
    H[800] = x1169;
    H[801] = x1191;
    H[802] = x1208;
    H[803] = x1219;
    H[804] = x1239;
    H[805] = x1252;
    H[806] = Wimpn_compressed[6] * x1118 + x1230 + x1258;
    H[807] = x1259;
    H[808] = x1260;
    H[809] = x1262;
    H[810] = x213;
    H[811] = x270;
    H[812] = x341;
    H[813] = x400;
    H[814] = x505;
    H[815] = x560;
    H[816] = x614;
    H[817] = x650;
    H[818] = x673;
    H[819] = x691;
    H[820] = x829;
    H[821] = x867;
    H[822] = x910;
    H[823] = x944;
    H[824] = x1010;
    H[825] = x1040;
    H[826] = x1070;
    H[827] = x1091;
    H[828] = x1105;
    H[829] = x1114;
    H[830] = x1171;
    H[831] = x1192;
    H[832] = x1209;
    H[833] = x1220;
    H[834] = x1243;
    H[835] = x1253;
    H[836] = x1259;
    H[837] = Wimpn_compressed[6] * x1117 + x1263 + x1264;
    H[838] = x1266;
    H[839] = x1267;
    H[840] = x215;
    H[841] = x272;
    H[842] = x343;
    H[843] = x403;
    H[844] = x508;
    H[845] = x565;
    H[846] = x618;
    H[847] = x653;
    H[848] = x674;
    H[849] = x692;
    H[850] = x831;
    H[851] = x869;
    H[852] = x912;
    H[853] = x947;
    H[854] = x1013;
    H[855] = x1045;
    H[856] = x1074;
    H[857] = x1094;
    H[858] = x1106;
    H[859] = x1115;
    H[860] = x1173;
    H[861] = x1194;
    H[862] = x1210;
    H[863] = x1221;
    H[864] = x1246;
    H[865] = x1255;
    H[866] = x1260;
    H[867] = x1266;
    H[868] = x1217 + x1225 + x1263;
    H[869] = x1268;
    H[870] = x216;
    H[871] = x273;
    H[872] = x345;
    H[873] = x405;
    H[874] = x509;
    H[875] = x568;
    H[876] = x621;
    H[877] = x656;
    H[878] = x675;
    H[879] = x693;
    H[880] = x832;
    H[881] = x870;
    H[882] = x914;
    H[883] = x949;
    H[884] = x1014;
    H[885] = x1048;
    H[886] = x1077;
    H[887] = x1097;
    H[888] = x1107;
    H[889] = x1116;
    H[890] = x1177;
    H[891] = x1196;
    H[892] = x1212;
    H[893] = x1222;
    H[894] = x1248;
    H[895] = x1257;
    H[896] = x1262;
    H[897] = x1267;
    H[898] = x1268;
    H[899] = x1197 + x1264;
}

#define TET10_S_IKMN_SIZE 45
static SFEM_INLINE void tet10_S_ikmn_neohookean_ogden(const scalar_t *const SFEM_RESTRICT adjugate,
                                                      const scalar_t                      jacobian_determinant,
                                                      const scalar_t                      qx,
                                                      const scalar_t                      qy,
                                                      const scalar_t                      qz,
                                                      const scalar_t                      qw,
                                                      const scalar_t *const SFEM_RESTRICT F,
                                                      const scalar_t                      mu,
                                                      const scalar_t                      lmbda,
                                                      scalar_t *const SFEM_RESTRICT       S_ikmn_canonical) {
    // mundane ops: 1015 divs: 2 sqrts: 0
    // total ops: 1031
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
    const scalar_t x31   = (1.0 / 6.0) * qw / jacobian_determinant;
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


static SFEM_INLINE void tet10_S_ikmn_neohookean_ogden_add(const scalar_t *const SFEM_RESTRICT adjugate,
    const scalar_t                      jacobian_determinant,
    const scalar_t                      qx,
    const scalar_t                      qy,
    const scalar_t                      qz,
    const scalar_t                      qw,
    const scalar_t *const SFEM_RESTRICT F,
    const scalar_t                      mu,
    const scalar_t                      lmbda,
    scalar_t *const SFEM_RESTRICT       S_ikmn_canonical) {
// mundane ops: 1015 divs: 2 sqrts: 0
// total ops: 1031
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
const scalar_t x31   = (1.0 / 6.0) * qw / jacobian_determinant;
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
S_ikmn_canonical[0] += x31 * (adjugate[0] * x20 + adjugate[1] * x30 + adjugate[2] * x27);
S_ikmn_canonical[1] += x31 * (adjugate[3] * x20 + adjugate[4] * x30 + adjugate[5] * x27);
S_ikmn_canonical[2] += x31 * (adjugate[6] * x20 + adjugate[7] * x30 + adjugate[8] * x27);
S_ikmn_canonical[3] += x31 * (adjugate[3] * x32 + adjugate[4] * x35 + adjugate[5] * x34);
S_ikmn_canonical[4] += x31 * (adjugate[6] * x32 + adjugate[7] * x35 + adjugate[8] * x34);
S_ikmn_canonical[5] += x31 * (adjugate[6] * (adjugate[6] * x19 - adjugate[7] * x17 + adjugate[8] * x21) +
adjugate[7] * (-adjugate[6] * x17 + adjugate[7] * x29 - adjugate[8] * x33) +
adjugate[8] * (adjugate[6] * x21 - adjugate[7] * x33 + adjugate[8] * x26));
S_ikmn_canonical[6] += x68 * (adjugate[0] * x60 + adjugate[1] * x50 + adjugate[2] * x67);
S_ikmn_canonical[7] += x68 * (adjugate[3] * x60 + adjugate[4] * x50 + adjugate[5] * x67);
S_ikmn_canonical[8] += x68 * (adjugate[6] * x60 + adjugate[7] * x50 + adjugate[8] * x67);
S_ikmn_canonical[9] += x68 * (adjugate[0] * x71 + adjugate[1] * x70 + adjugate[2] * x73);
S_ikmn_canonical[10] += x68 * (adjugate[3] * x71 + adjugate[4] * x70 + adjugate[5] * x73);
S_ikmn_canonical[11] += x68 * (adjugate[6] * x71 + adjugate[7] * x70 + adjugate[8] * x73);
S_ikmn_canonical[12] += x68 * (adjugate[0] * x75 + adjugate[1] * x74 + adjugate[2] * x76);
S_ikmn_canonical[13] += x68 * (adjugate[3] * x75 + adjugate[4] * x74 + adjugate[5] * x76);
S_ikmn_canonical[14] += x68 * (adjugate[6] * x75 + adjugate[7] * x74 + adjugate[8] * x76);
S_ikmn_canonical[15] += x68 * (adjugate[0] * x89 + adjugate[1] * x102 + adjugate[2] * x97);
S_ikmn_canonical[16] += x68 * (adjugate[3] * x89 + adjugate[4] * x102 + adjugate[5] * x97);
S_ikmn_canonical[17] += x68 * (adjugate[6] * x89 + adjugate[7] * x102 + adjugate[8] * x97);
S_ikmn_canonical[18] += x68 * (adjugate[0] * x103 + adjugate[1] * x106 + adjugate[2] * x105);
S_ikmn_canonical[19] += x68 * (adjugate[3] * x103 + adjugate[4] * x106 + adjugate[5] * x105);
S_ikmn_canonical[20] += x68 * (adjugate[6] * x103 + adjugate[7] * x106 + adjugate[8] * x105);
S_ikmn_canonical[21] += x68 * (adjugate[0] * x107 + adjugate[1] * x109 + adjugate[2] * x108);
S_ikmn_canonical[22] += x68 * (adjugate[3] * x107 + adjugate[4] * x109 + adjugate[5] * x108);
S_ikmn_canonical[23] += x68 * (adjugate[6] * x107 + adjugate[7] * x109 + adjugate[8] * x108);
S_ikmn_canonical[24] += x31 * (adjugate[0] * x115 + adjugate[1] * x125 + adjugate[2] * x122);
S_ikmn_canonical[25] += x31 * (adjugate[3] * x115 + adjugate[4] * x125 + adjugate[5] * x122);
S_ikmn_canonical[26] += x31 * (adjugate[6] * x115 + adjugate[7] * x125 + adjugate[8] * x122);
S_ikmn_canonical[27] += x31 * (adjugate[3] * x128 + adjugate[4] * x132 + adjugate[5] * x131);
S_ikmn_canonical[28] += x31 * (adjugate[6] * x128 + adjugate[7] * x132 + adjugate[8] * x131);
S_ikmn_canonical[29] += x31 * (adjugate[6] * (adjugate[6] * x114 - adjugate[7] * x127 + adjugate[8] * x129) +
adjugate[7] * (-adjugate[6] * x127 + adjugate[7] * x124 - adjugate[8] * x130) +
adjugate[8] * (adjugate[6] * x129 - adjugate[7] * x130 + adjugate[8] * x121));
S_ikmn_canonical[30] += x68 * (adjugate[0] * x152 + adjugate[1] * x143 + adjugate[2] * x159);
S_ikmn_canonical[31] += x68 * (adjugate[3] * x152 + adjugate[4] * x143 + adjugate[5] * x159);
S_ikmn_canonical[32] += x68 * (adjugate[6] * x152 + adjugate[7] * x143 + adjugate[8] * x159);
S_ikmn_canonical[33] += x68 * (adjugate[0] * x163 + adjugate[1] * x161 + adjugate[2] * x164);
S_ikmn_canonical[34] += x68 * (adjugate[3] * x163 + adjugate[4] * x161 + adjugate[5] * x164);
S_ikmn_canonical[35] += x68 * (adjugate[6] * x163 + adjugate[7] * x161 + adjugate[8] * x164);
S_ikmn_canonical[36] += x68 * (adjugate[0] * x166 + adjugate[1] * x165 + adjugate[2] * x168);
S_ikmn_canonical[37] += x68 * (adjugate[3] * x166 + adjugate[4] * x165 + adjugate[5] * x168);
S_ikmn_canonical[38] += x68 * (adjugate[6] * x166 + adjugate[7] * x165 + adjugate[8] * x168);
S_ikmn_canonical[39] += x31 * (adjugate[0] * x174 + adjugate[1] * x183 + adjugate[2] * x180);
S_ikmn_canonical[40] += x31 * (adjugate[3] * x174 + adjugate[4] * x183 + adjugate[5] * x180);
S_ikmn_canonical[41] += x31 * (adjugate[6] * x174 + adjugate[7] * x183 + adjugate[8] * x180);
S_ikmn_canonical[42] += x31 * (adjugate[3] * x185 + adjugate[4] * x189 + adjugate[5] * x188);
S_ikmn_canonical[43] += x31 * (adjugate[6] * x185 + adjugate[7] * x189 + adjugate[8] * x188);
S_ikmn_canonical[44] += x31 * (adjugate[6] * (adjugate[6] * x173 - adjugate[7] * x184 + x167 * x175) +
adjugate[7] * (-adjugate[6] * x184 + adjugate[7] * x182 - x167 * x177) +
adjugate[8] * (adjugate[6] * x186 - adjugate[7] * x187 + adjugate[8] * x179));
}

static SFEM_INLINE void tet10_apply_S_ikmn(const scalar_t                      qx,
                                           const scalar_t                      qy,
                                           const scalar_t                      qz,
                                           const scalar_t *const SFEM_RESTRICT S_ikmn_canonical,  // 3x3x3x3, includes dV
                                           const scalar_t *const SFEM_RESTRICT inc_grad,  // 3x3 reference trial gradient R
                                           scalar_t *const SFEM_RESTRICT       eoutx,
                                           scalar_t *const SFEM_RESTRICT       eouty,
                                           scalar_t *const SFEM_RESTRICT       eoutz) {
    // mundane ops: 290 divs: 0 sqrts: 0
    // total ops: 290
    const scalar_t x0 = 4 * qx;
    const scalar_t x1 = 4 * qy;
    const scalar_t x2 = 4 * qz;
    const scalar_t x3 = x0 + x1 + x2 - 3;
    const scalar_t x4 = S_ikmn_canonical[0] * inc_grad[0] + S_ikmn_canonical[12] * inc_grad[5] +
                        S_ikmn_canonical[15] * inc_grad[6] + S_ikmn_canonical[18] * inc_grad[7] +
                        S_ikmn_canonical[1] * inc_grad[1] + S_ikmn_canonical[21] * inc_grad[8] +
                        S_ikmn_canonical[2] * inc_grad[2] + S_ikmn_canonical[6] * inc_grad[3] + S_ikmn_canonical[9] * inc_grad[4];
    const scalar_t x5 = S_ikmn_canonical[10] * inc_grad[4] + S_ikmn_canonical[13] * inc_grad[5] +
                        S_ikmn_canonical[16] * inc_grad[6] + S_ikmn_canonical[19] * inc_grad[7] +
                        S_ikmn_canonical[1] * inc_grad[0] + S_ikmn_canonical[22] * inc_grad[8] +
                        S_ikmn_canonical[3] * inc_grad[1] + S_ikmn_canonical[4] * inc_grad[2] + S_ikmn_canonical[7] * inc_grad[3];
    const scalar_t x6 = S_ikmn_canonical[11] * inc_grad[4] + S_ikmn_canonical[14] * inc_grad[5] +
                        S_ikmn_canonical[17] * inc_grad[6] + S_ikmn_canonical[20] * inc_grad[7] +
                        S_ikmn_canonical[23] * inc_grad[8] + S_ikmn_canonical[2] * inc_grad[0] +
                        S_ikmn_canonical[4] * inc_grad[1] + S_ikmn_canonical[5] * inc_grad[2] + S_ikmn_canonical[8] * inc_grad[3];
    const scalar_t x7  = x0 - 1;
    const scalar_t x8  = x1 - 1;
    const scalar_t x9  = x2 - 1;
    const scalar_t x10 = qx * x5;
    const scalar_t x11 = qx * x6;
    const scalar_t x12 = qz - 1;
    const scalar_t x13 = 2 * qx + qy + x12;
    const scalar_t x14 = qy * x4;
    const scalar_t x15 = qy * x6;
    const scalar_t x16 = qx + 2 * qy + x12;
    const scalar_t x17 = qz * x4;
    const scalar_t x18 = qz * x5;
    const scalar_t x19 = qx + qy + 2 * qz - 1;
    const scalar_t x20 =
            S_ikmn_canonical[24] * inc_grad[3] + S_ikmn_canonical[25] * inc_grad[4] + S_ikmn_canonical[26] * inc_grad[5] +
            S_ikmn_canonical[30] * inc_grad[6] + S_ikmn_canonical[33] * inc_grad[7] + S_ikmn_canonical[36] * inc_grad[8] +
            S_ikmn_canonical[6] * inc_grad[0] + S_ikmn_canonical[7] * inc_grad[1] + S_ikmn_canonical[8] * inc_grad[2];
    const scalar_t x21 =
            S_ikmn_canonical[10] * inc_grad[1] + S_ikmn_canonical[11] * inc_grad[2] + S_ikmn_canonical[25] * inc_grad[3] +
            S_ikmn_canonical[27] * inc_grad[4] + S_ikmn_canonical[28] * inc_grad[5] + S_ikmn_canonical[31] * inc_grad[6] +
            S_ikmn_canonical[34] * inc_grad[7] + S_ikmn_canonical[37] * inc_grad[8] + S_ikmn_canonical[9] * inc_grad[0];
    const scalar_t x22 =
            S_ikmn_canonical[12] * inc_grad[0] + S_ikmn_canonical[13] * inc_grad[1] + S_ikmn_canonical[14] * inc_grad[2] +
            S_ikmn_canonical[26] * inc_grad[3] + S_ikmn_canonical[28] * inc_grad[4] + S_ikmn_canonical[29] * inc_grad[5] +
            S_ikmn_canonical[32] * inc_grad[6] + S_ikmn_canonical[35] * inc_grad[7] + S_ikmn_canonical[38] * inc_grad[8];
    const scalar_t x23 = qx * x21;
    const scalar_t x24 = qx * x22;
    const scalar_t x25 = qy * x20;
    const scalar_t x26 = qy * x22;
    const scalar_t x27 = qz * x21;
    const scalar_t x28 = qz * x20;
    const scalar_t x29 =
            S_ikmn_canonical[15] * inc_grad[0] + S_ikmn_canonical[16] * inc_grad[1] + S_ikmn_canonical[17] * inc_grad[2] +
            S_ikmn_canonical[30] * inc_grad[3] + S_ikmn_canonical[31] * inc_grad[4] + S_ikmn_canonical[32] * inc_grad[5] +
            S_ikmn_canonical[39] * inc_grad[6] + S_ikmn_canonical[40] * inc_grad[7] + S_ikmn_canonical[41] * inc_grad[8];
    const scalar_t x30 =
            S_ikmn_canonical[18] * inc_grad[0] + S_ikmn_canonical[19] * inc_grad[1] + S_ikmn_canonical[20] * inc_grad[2] +
            S_ikmn_canonical[33] * inc_grad[3] + S_ikmn_canonical[34] * inc_grad[4] + S_ikmn_canonical[35] * inc_grad[5] +
            S_ikmn_canonical[40] * inc_grad[6] + S_ikmn_canonical[42] * inc_grad[7] + S_ikmn_canonical[43] * inc_grad[8];
    const scalar_t x31 =
            S_ikmn_canonical[21] * inc_grad[0] + S_ikmn_canonical[22] * inc_grad[1] + S_ikmn_canonical[23] * inc_grad[2] +
            S_ikmn_canonical[36] * inc_grad[3] + S_ikmn_canonical[37] * inc_grad[4] + S_ikmn_canonical[38] * inc_grad[5] +
            S_ikmn_canonical[41] * inc_grad[6] + S_ikmn_canonical[43] * inc_grad[7] + S_ikmn_canonical[44] * inc_grad[8];
    const scalar_t x32 = qx * x30;
    const scalar_t x33 = qx * x31;
    const scalar_t x34 = qy * x29;
    const scalar_t x35 = qy * x31;
    const scalar_t x36 = qz * x29;
    const scalar_t x37 = qz * x30;
    eoutx[0]           = x3 * (x4 + x5 + x6);
    eoutx[1]           = x4 * x7;
    eoutx[2]           = x5 * x8;
    eoutx[3]           = x6 * x9;
    eoutx[4]           = -4 * x10 - 4 * x11 - 4 * x13 * x4;
    eoutx[5]           = 4 * x10 + 4 * x14;
    eoutx[6]           = -4 * x14 - 4 * x15 - 4 * x16 * x5;
    eoutx[7]           = -4 * x17 - 4 * x18 - 4 * x19 * x6;
    eoutx[8]           = 4 * x11 + 4 * x17;
    eoutx[9]           = 4 * x15 + 4 * x18;
    eouty[0]           = x3 * (x20 + x21 + x22);
    eouty[1]           = x20 * x7;
    eouty[2]           = x21 * x8;
    eouty[3]           = x22 * x9;
    eouty[4]           = -4 * x13 * x20 - 4 * x23 - 4 * x24;
    eouty[5]           = 4 * x23 + 4 * x25;
    eouty[6]           = -4 * x16 * x21 - 4 * x25 - 4 * x26;
    eouty[7]           = -4 * x19 * x22 - 4 * x27 - 4 * x28;
    eouty[8]           = 4 * x24 + 4 * x28;
    eouty[9]           = 4 * x26 + 4 * x27;
    eoutz[0]           = x3 * (x29 + x30 + x31);
    eoutz[1]           = x29 * x7;
    eoutz[2]           = x30 * x8;
    eoutz[3]           = x31 * x9;
    eoutz[4]           = -4 * x13 * x29 - 4 * x32 - 4 * x33;
    eoutz[5]           = 4 * x32 + 4 * x34;
    eoutz[6]           = -4 * x16 * x30 - 4 * x34 - 4 * x35;
    eoutz[7]           = -4 * x19 * x31 - 4 * x36 - 4 * x37;
    eoutz[8]           = 4 * x33 + 4 * x36;
    eoutz[9]           = 4 * x35 + 4 * x37;
}

#endif /* SFEM_TET10_PARTIAL_ASSEMBLY_NEOHOOKEAN_OGDEN_INLINE_H */
