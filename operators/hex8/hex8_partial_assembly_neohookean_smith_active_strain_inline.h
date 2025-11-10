#ifndef SFEM_HEX8_PARTIAL_ASSEMBLY_NEOHOOKEAN_SMITH_ACTIVE_STRAIN_INLINE_H
#define SFEM_HEX8_PARTIAL_ASSEMBLY_NEOHOOKEAN_SMITH_ACTIVE_STRAIN_INLINE_H

#include "sfem_macros.h"


#define HEX8_S_IKMN_SIZE 45
static SFEM_INLINE void hex8_S_ikmn_neohookean_smith_active_strain(const scalar_t *const SFEM_RESTRICT adjugate,
                                                                   const scalar_t                      jacobian_determinant,
                                                                   const scalar_t                      qx,
                                                                   const scalar_t                      qy,
                                                                   const scalar_t                      qz,
                                                                   const scalar_t                      qw,
                                                                   const scalar_t *const SFEM_RESTRICT F,
                                                                   const scalar_t                      lmda,
                                                                   const scalar_t                      mu,
                                                                   const scalar_t                      lmbda,
                                                                   const scalar_t *const SFEM_RESTRICT Fa_inv,
                                                                   const scalar_t                      Ja,
                                                                   scalar_t *const SFEM_RESTRICT       S_ikmn_canonical) {
    // mundane ops: 1494 divs: 3 sqrts: 0
    // total ops: 1518
    const scalar_t x0  = F[4] * Fa_inv[4];
    const scalar_t x1  = F[8] * Fa_inv[8];
    const scalar_t x2  = Fa_inv[0] * x1;
    const scalar_t x3  = F[4] * Fa_inv[5];
    const scalar_t x4  = F[8] * Fa_inv[6];
    const scalar_t x5  = Fa_inv[1] * x4;
    const scalar_t x6  = F[4] * Fa_inv[3];
    const scalar_t x7  = F[8] * Fa_inv[7];
    const scalar_t x8  = Fa_inv[2] * x7;
    const scalar_t x9  = F[5] * Fa_inv[7];
    const scalar_t x10 = F[7] * Fa_inv[5];
    const scalar_t x11 = Fa_inv[0] * x10;
    const scalar_t x12 = F[5] * Fa_inv[8];
    const scalar_t x13 = F[7] * Fa_inv[3];
    const scalar_t x14 = Fa_inv[1] * x13;
    const scalar_t x15 = F[5] * Fa_inv[6];
    const scalar_t x16 = F[7] * Fa_inv[4];
    const scalar_t x17 = Fa_inv[2] * x16;
    const scalar_t x18 = Fa_inv[0] * x7;
    const scalar_t x19 = Fa_inv[1] * x1;
    const scalar_t x20 = Fa_inv[2] * x4;
    const scalar_t x21 = Fa_inv[0] * x16;
    const scalar_t x22 = Fa_inv[1] * x10;
    const scalar_t x23 = Fa_inv[2] * x13;
    const scalar_t x24 = x0 * x2 - x0 * x20 + x11 * x9 + x12 * x14 - x12 * x21 + x15 * x17 - x15 * x22 - x18 * x3 - x19 * x6 -
                         x23 * x9 + x3 * x5 + x6 * x8;
    const scalar_t x25 = F[0] * Fa_inv[0];
    const scalar_t x26 = F[1] * Fa_inv[3];
    const scalar_t x27 = F[2] * Fa_inv[6];
    const scalar_t x28 = x25 + x26 + x27;
    const scalar_t x29 = F[0] * Fa_inv[1];
    const scalar_t x30 = F[1] * Fa_inv[4];
    const scalar_t x31 = F[2] * Fa_inv[7];
    const scalar_t x32 = x29 + x30 + x31;
    const scalar_t x33 = F[0] * Fa_inv[2];
    const scalar_t x34 = F[1] * Fa_inv[5];
    const scalar_t x35 = F[2] * Fa_inv[8];
    const scalar_t x36 = x33 + x34 + x35;
    const scalar_t x37 = Fa_inv[0] * x28 + Fa_inv[1] * x32 + Fa_inv[2] * x36;
    const scalar_t x38 = F[3] * Fa_inv[0];
    const scalar_t x39 = x15 + x38 + x6;
    const scalar_t x40 = F[3] * Fa_inv[1];
    const scalar_t x41 = x0 + x40 + x9;
    const scalar_t x42 = F[3] * Fa_inv[2];
    const scalar_t x43 = x12 + x3 + x42;
    const scalar_t x44 = F[6] * Fa_inv[0];
    const scalar_t x45 = x13 + x4 + x44;
    const scalar_t x46 = F[6] * Fa_inv[1];
    const scalar_t x47 = x16 + x46 + x7;
    const scalar_t x48 = F[6] * Fa_inv[2];
    const scalar_t x49 = x1 + x10 + x48;
    const scalar_t x50 =
            POW2(x28) + POW2(x32) + POW2(x36) + POW2(x39) + POW2(x41) + POW2(x43) + POW2(x45) + POW2(x47) + POW2(x49) + 1;
    const scalar_t x51 = mu / POW2(x50);
    const scalar_t x52 = 2 * x51;
    const scalar_t x53 = mu * (POW2(Fa_inv[0]) + POW2(Fa_inv[1]) + POW2(Fa_inv[2]));
    const scalar_t x54 = 1.0 / x50;
    const scalar_t x55 = -x53 * x54 + x53;
    const scalar_t x56 = lmbda * POW2(x24) + POW2(x37) * x52 + x55;
    const scalar_t x57 = Fa_inv[8] * x16;
    const scalar_t x58 = Fa_inv[6] * x10;
    const scalar_t x59 = Fa_inv[7] * x13;
    const scalar_t x60 = Fa_inv[7] * x44;
    const scalar_t x61 = Fa_inv[8] * x46;
    const scalar_t x62 = Fa_inv[6] * x48;
    const scalar_t x63 = Fa_inv[7] * x10;
    const scalar_t x64 = Fa_inv[8] * x13;
    const scalar_t x65 = Fa_inv[6] * x16;
    const scalar_t x66 = Fa_inv[8] * x44;
    const scalar_t x67 = Fa_inv[6] * x46;
    const scalar_t x68 = Fa_inv[7] * x48;
    const scalar_t x69 = x0 * x62 - x0 * x66 + x3 * x60 - x3 * x67 + x38 * x57 - x38 * x63 + x40 * x58 - x40 * x64 + x42 * x59 -
                         x42 * x65 + x6 * x61 - x6 * x68;
    const scalar_t x70 = lmbda * x24;
    const scalar_t x71 = Fa_inv[6] * x28 + Fa_inv[7] * x32 + Fa_inv[8] * x36;
    const scalar_t x72 = x37 * x52;
    const scalar_t x73 = mu * (Fa_inv[0] * Fa_inv[6] + Fa_inv[1] * Fa_inv[7] + Fa_inv[2] * Fa_inv[8]);
    const scalar_t x74 = -x54 * x73 + x73;
    const scalar_t x75 = x69 * x70 + x71 * x72 + x74;
    const scalar_t x76 = Fa_inv[4] * x1;
    const scalar_t x77 = Fa_inv[5] * x4;
    const scalar_t x78 = Fa_inv[3] * x7;
    const scalar_t x79 = Fa_inv[5] * x44;
    const scalar_t x80 = Fa_inv[3] * x46;
    const scalar_t x81 = Fa_inv[4] * x48;
    const scalar_t x82 = Fa_inv[5] * x7;
    const scalar_t x83 = Fa_inv[3] * x1;
    const scalar_t x84 = Fa_inv[4] * x4;
    const scalar_t x85 = Fa_inv[4] * x44;
    const scalar_t x86 = Fa_inv[5] * x46;
    const scalar_t x87 = Fa_inv[3] * x48;
    const scalar_t x88 = x12 * x80 - x12 * x85 + x15 * x81 - x15 * x86 + x38 * x76 - x38 * x82 + x40 * x77 - x40 * x83 +
                         x42 * x78 - x42 * x84 + x79 * x9 - x87 * x9;
    const scalar_t x89  = Fa_inv[3] * x28 + Fa_inv[4] * x32 + Fa_inv[5] * x36;
    const scalar_t x90  = mu * (Fa_inv[0] * Fa_inv[3] + Fa_inv[1] * Fa_inv[4] + Fa_inv[2] * Fa_inv[5]);
    const scalar_t x91  = x54 * x90 - x90;
    const scalar_t x92  = x70 * x88 - x72 * x89 + x91;
    const scalar_t x93  = adjugate[0] * x56 - adjugate[1] * x92 + adjugate[2] * x75;
    const scalar_t x94  = mu * (POW2(Fa_inv[6]) + POW2(Fa_inv[7]) + POW2(Fa_inv[8]));
    const scalar_t x95  = -x54 * x94 + x94;
    const scalar_t x96  = lmbda * POW2(x69) + x52 * POW2(x71) + x95;
    const scalar_t x97  = lmbda * x88;
    const scalar_t x98  = x52 * x89;
    const scalar_t x99  = mu * (Fa_inv[3] * Fa_inv[6] + Fa_inv[4] * Fa_inv[7] + Fa_inv[5] * Fa_inv[8]);
    const scalar_t x100 = x54 * x99 - x99;
    const scalar_t x101 = x100 + x69 * x97 - x71 * x98;
    const scalar_t x102 = adjugate[0] * x75 - adjugate[1] * x101 + adjugate[2] * x96;
    const scalar_t x103 = mu * (POW2(Fa_inv[3]) + POW2(Fa_inv[4]) + POW2(Fa_inv[5]));
    const scalar_t x104 = -x103 * x54 + x103;
    const scalar_t x105 = lmbda * POW2(x88) + x104 + x52 * POW2(x89);
    const scalar_t x106 = -adjugate[0] * x92 + adjugate[1] * x105 - adjugate[2] * x101;
    const scalar_t x107 = Ja * qw / jacobian_determinant;
    const scalar_t x108 = adjugate[3] * x56 - adjugate[4] * x92 + adjugate[5] * x75;
    const scalar_t x109 = adjugate[3] * x75 - adjugate[4] * x101 + adjugate[5] * x96;
    const scalar_t x110 = -adjugate[3] * x92 + adjugate[4] * x105 - adjugate[5] * x101;
    const scalar_t x111 = x11 * x31 + x14 * x35 + x17 * x27 - x18 * x34 - x19 * x26 + x2 * x30 - x20 * x30 - x21 * x35 -
                          x22 * x27 - x23 * x31 + x26 * x8 + x34 * x5;
    const scalar_t x112 = Fa_inv[0] * x39 + Fa_inv[1] * x41 + Fa_inv[2] * x43;
    const scalar_t x113 = 4 * x111 * x70 - 4 * x112 * x72;
    const scalar_t x114 = x25 * x57 - x25 * x63 + x26 * x61 - x26 * x68 + x29 * x58 - x29 * x64 + x30 * x62 - x30 * x66 +
                          x33 * x59 - x33 * x65 + x34 * x60 - x34 * x67;
    const scalar_t x115 = 4 * x70;
    const scalar_t x116 = Fa_inv[6] * x39 + Fa_inv[7] * x41 + Fa_inv[8] * x43;
    const scalar_t x117 = 8 * x51;
    const scalar_t x118 = x117 * x37;
    const scalar_t x119 = x0 * x25;
    const scalar_t x120 = 4 * x1;
    const scalar_t x121 = x25 * x3;
    const scalar_t x122 = 4 * x7;
    const scalar_t x123 = x29 * x6;
    const scalar_t x124 = x29 * x3;
    const scalar_t x125 = 4 * x4;
    const scalar_t x126 = x33 * x6;
    const scalar_t x127 = x0 * x33;
    const scalar_t x128 = x12 * x25;
    const scalar_t x129 = 4 * x16;
    const scalar_t x130 = x25 * x9;
    const scalar_t x131 = 4 * x10;
    const scalar_t x132 = x12 * x29;
    const scalar_t x133 = 4 * x13;
    const scalar_t x134 = x15 * x29;
    const scalar_t x135 = x33 * x9;
    const scalar_t x136 = x15 * x33;
    const scalar_t x137 = x30 * x38;
    const scalar_t x138 = x34 * x38;
    const scalar_t x139 = x26 * x40;
    const scalar_t x140 = x34 * x40;
    const scalar_t x141 = x26 * x42;
    const scalar_t x142 = x30 * x42;
    const scalar_t x143 = x12 * x30;
    const scalar_t x144 = 4 * x44;
    const scalar_t x145 = x34 * x9;
    const scalar_t x146 = x12 * x26;
    const scalar_t x147 = 4 * x46;
    const scalar_t x148 = x15 * x34;
    const scalar_t x149 = x26 * x9;
    const scalar_t x150 = 4 * x48;
    const scalar_t x151 = x15 * x30;
    const scalar_t x152 = x35 * x38;
    const scalar_t x153 = x31 * x38;
    const scalar_t x154 = x35 * x40;
    const scalar_t x155 = x27 * x40;
    const scalar_t x156 = x31 * x42;
    const scalar_t x157 = x27 * x42;
    const scalar_t x158 = x0 * x35;
    const scalar_t x159 = x3 * x31;
    const scalar_t x160 = x35 * x6;
    const scalar_t x161 = x27 * x3;
    const scalar_t x162 = x31 * x6;
    const scalar_t x163 = x0 * x27;
    const scalar_t x164 =
            lmbda *
            (Fa_inv[0] * Fa_inv[4] * Fa_inv[8] - Fa_inv[0] * Fa_inv[5] * Fa_inv[7] - Fa_inv[1] * Fa_inv[3] * Fa_inv[8] +
             Fa_inv[1] * Fa_inv[5] * Fa_inv[6] + Fa_inv[2] * Fa_inv[3] * Fa_inv[7] - Fa_inv[2] * Fa_inv[4] * Fa_inv[6]) *
            (-x119 * x120 + x120 * x123 + x120 * x137 - x120 * x139 + x121 * x122 - x122 * x126 - x122 * x138 + x122 * x141 -
             x124 * x125 + x125 * x127 + x125 * x140 - x125 * x142 + x128 * x129 - x129 * x136 - x129 * x152 + x129 * x157 -
             x130 * x131 + x131 * x134 + x131 * x153 - x131 * x155 - x132 * x133 + x133 * x135 + x133 * x154 - x133 * x156 -
             x143 * x144 + x144 * x145 + x144 * x158 - x144 * x159 + x146 * x147 - x147 * x148 - x147 * x160 + x147 * x161 -
             x149 * x150 + x150 * x151 + x150 * x162 - x150 * x163 + 4 + 3 * mu / lmda);
    const scalar_t x165 = F[7] * x164;
    const scalar_t x166 = -x114 * x115 + x116 * x118 + x165;
    const scalar_t x167 = x25 * x76 - x25 * x82 + x27 * x81 - x27 * x86 + x29 * x77 - x29 * x83 + x31 * x79 - x31 * x87 +
                          x33 * x78 - x33 * x84 + x35 * x80 - x35 * x85;
    const scalar_t x168 = Fa_inv[3] * x39 + Fa_inv[4] * x41 + Fa_inv[5] * x43;
    const scalar_t x169 = F[8] * x164;
    const scalar_t x170 = x115 * x167 + x118 * x168 - x169;
    const scalar_t x171 = -adjugate[0] * x113 + adjugate[1] * x170 + adjugate[2] * x166;
    const scalar_t x172 = 4 * x167 * x97 - 4 * x168 * x98;
    const scalar_t x173 = 4 * x97;
    const scalar_t x174 = x117 * x89;
    const scalar_t x175 = x111 * x173 + x112 * x174 + x169;
    const scalar_t x176 = F[6] * x164;
    const scalar_t x177 = x114 * x173 + x116 * x174 - x176;
    const scalar_t x178 = adjugate[0] * x175 - adjugate[1] * x172 + adjugate[2] * x177;
    const scalar_t x179 = lmbda * x69;
    const scalar_t x180 = x52 * x71;
    const scalar_t x181 = 4 * x114 * x179 - 4 * x116 * x180;
    const scalar_t x182 = 4 * x179;
    const scalar_t x183 = x117 * x71;
    const scalar_t x184 = x111 * x182 - x112 * x183 + x165;
    const scalar_t x185 = x167 * x182 + x168 * x183 + x176;
    const scalar_t x186 = -adjugate[0] * x184 + adjugate[1] * x185 - adjugate[2] * x181;
    const scalar_t x187 = (1.0 / 4.0) * x107;
    const scalar_t x188 = -adjugate[3] * x113 + adjugate[4] * x170 + adjugate[5] * x166;
    const scalar_t x189 = adjugate[3] * x175 - adjugate[4] * x172 + adjugate[5] * x177;
    const scalar_t x190 = -adjugate[3] * x184 + adjugate[4] * x185 - adjugate[5] * x181;
    const scalar_t x191 = -adjugate[6] * x113 + adjugate[7] * x170 + adjugate[8] * x166;
    const scalar_t x192 = adjugate[6] * x175 - adjugate[7] * x172 + adjugate[8] * x177;
    const scalar_t x193 = -adjugate[6] * x184 + adjugate[7] * x185 - adjugate[8] * x181;
    const scalar_t x194 = Fa_inv[0] * x143 - Fa_inv[0] * x145 - Fa_inv[0] * x158 + Fa_inv[0] * x159 - Fa_inv[1] * x146 +
                          Fa_inv[1] * x148 + Fa_inv[1] * x160 - Fa_inv[1] * x161 + Fa_inv[2] * x149 - Fa_inv[2] * x151 -
                          Fa_inv[2] * x162 + Fa_inv[2] * x163;
    const scalar_t x195 = Fa_inv[0] * x45 + Fa_inv[1] * x47 + Fa_inv[2] * x49;
    const scalar_t x196 = 4 * x194 * x70 + 4 * x195 * x72;
    const scalar_t x197 = -Fa_inv[3] * x132 + Fa_inv[3] * x135 + Fa_inv[3] * x154 - Fa_inv[3] * x156 + Fa_inv[4] * x128 -
                          Fa_inv[4] * x136 - Fa_inv[4] * x152 + Fa_inv[4] * x157 - Fa_inv[5] * x130 + Fa_inv[5] * x134 +
                          Fa_inv[5] * x153 - Fa_inv[5] * x155;
    const scalar_t x198 = Fa_inv[3] * x45 + Fa_inv[4] * x47 + Fa_inv[5] * x49;
    const scalar_t x199 = F[5] * x164;
    const scalar_t x200 = -x115 * x197 + x118 * x198 + x199;
    const scalar_t x201 = Fa_inv[6] * x124 - Fa_inv[6] * x127 - Fa_inv[6] * x140 + Fa_inv[6] * x142 - Fa_inv[7] * x121 +
                          Fa_inv[7] * x126 + Fa_inv[7] * x138 - Fa_inv[7] * x141 + Fa_inv[8] * x119 - Fa_inv[8] * x123 -
                          Fa_inv[8] * x137 + Fa_inv[8] * x139;
    const scalar_t x202  = Fa_inv[6] * x45 + Fa_inv[7] * x47 + Fa_inv[8] * x49;
    const scalar_t x203  = F[4] * x164;
    const scalar_t x204  = x115 * x201 + x118 * x202 - x203;
    const scalar_t x205  = adjugate[0] * x196 + adjugate[1] * x200 + adjugate[2] * x204;
    const scalar_t x206  = 4 * x197 * x97 + 4 * x198 * x98;
    const scalar_t x207  = F[3] * x164;
    const scalar_t x208  = -x173 * x201 + x174 * x202 + x207;
    const scalar_t x209  = x173 * x194 - x174 * x195 + x199;
    const scalar_t x210  = -adjugate[0] * x209 + adjugate[1] * x206 + adjugate[2] * x208;
    const scalar_t x211  = 4 * x179 * x201 + 4 * x180 * x202;
    const scalar_t x212  = x182 * x194 + x183 * x195 + x203;
    const scalar_t x213  = x182 * x197 - x183 * x198 + x207;
    const scalar_t x214  = adjugate[0] * x212 - adjugate[1] * x213 + adjugate[2] * x211;
    const scalar_t x215  = adjugate[3] * x196 + adjugate[4] * x200 + adjugate[5] * x204;
    const scalar_t x216  = -adjugate[3] * x209 + adjugate[4] * x206 + adjugate[5] * x208;
    const scalar_t x217  = adjugate[3] * x212 - adjugate[4] * x213 + adjugate[5] * x211;
    const scalar_t x218  = adjugate[6] * x196 + adjugate[7] * x200 + adjugate[8] * x204;
    const scalar_t x219  = -adjugate[6] * x209 + adjugate[7] * x206 + adjugate[8] * x208;
    const scalar_t x220  = adjugate[6] * x212 - adjugate[7] * x213 + adjugate[8] * x211;
    const scalar_t x221  = lmbda * POW2(x111) + POW2(x112) * x52 + x55;
    const scalar_t x222  = lmbda * x111;
    const scalar_t x223  = x112 * x52;
    const scalar_t x224  = x114 * x222 + x116 * x223 + x74;
    const scalar_t x225  = x167 * x222 - x168 * x223 + x91;
    const scalar_t x226  = adjugate[0] * x221 - adjugate[1] * x225 + adjugate[2] * x224;
    const scalar_t x227  = lmbda * POW2(x114) + POW2(x116) * x52 + x95;
    const scalar_t x228  = lmbda * x167;
    const scalar_t x229  = x168 * x52;
    const scalar_t x230  = x100 + x114 * x228 - x116 * x229;
    const scalar_t x231  = adjugate[0] * x224 - adjugate[1] * x230 + adjugate[2] * x227;
    const scalar_t x232  = lmbda * POW2(x167) + x104 + POW2(x168) * x52;
    const scalar_t x233  = -adjugate[0] * x225 + adjugate[1] * x232 - adjugate[2] * x230;
    const scalar_t x234  = adjugate[3] * x221 - adjugate[4] * x225 + adjugate[5] * x224;
    const scalar_t x235  = adjugate[3] * x224 - adjugate[4] * x230 + adjugate[5] * x227;
    const scalar_t x236  = -adjugate[3] * x225 + adjugate[4] * x232 - adjugate[5] * x230;
    const scalar_t x237  = 4 * x194 * x222 - 4 * x195 * x223;
    const scalar_t x238  = 4 * x222;
    const scalar_t x239  = x112 * x117;
    const scalar_t x240  = F[1] * x164;
    const scalar_t x241  = -x201 * x238 + x202 * x239 + x240;
    const scalar_t x242  = F[2] * x164;
    const scalar_t x243  = x197 * x238 + x198 * x239 - x242;
    const scalar_t x244  = -adjugate[0] * x237 + adjugate[1] * x243 + adjugate[2] * x241;
    const scalar_t x245  = 4 * x197 * x228 - 4 * x198 * x229;
    const scalar_t x246  = 4 * x228;
    const scalar_t x247  = x117 * x168;
    const scalar_t x248  = x194 * x246 + x195 * x247 + x242;
    const scalar_t x249  = F[0] * x164;
    const scalar_t x250  = x201 * x246 + x202 * x247 - x249;
    const scalar_t x251  = adjugate[0] * x248 - adjugate[1] * x245 + adjugate[2] * x250;
    const scalar_t x252  = lmbda * x201;
    const scalar_t x253  = x202 * x52;
    const scalar_t x254  = 4 * x114 * x252 - 4 * x116 * x253;
    const scalar_t x255  = 4 * x114;
    const scalar_t x256  = x116 * x117;
    const scalar_t x257  = lmbda * x194 * x255 - x195 * x256 + x240;
    const scalar_t x258  = lmbda * x197;
    const scalar_t x259  = x198 * x256 + x249 + x255 * x258;
    const scalar_t x260  = -adjugate[0] * x257 + adjugate[1] * x259 - adjugate[2] * x254;
    const scalar_t x261  = -adjugate[3] * x237 + adjugate[4] * x243 + adjugate[5] * x241;
    const scalar_t x262  = adjugate[3] * x248 - adjugate[4] * x245 + adjugate[5] * x250;
    const scalar_t x263  = -adjugate[3] * x257 + adjugate[4] * x259 - adjugate[5] * x254;
    const scalar_t x264  = -adjugate[6] * x237 + adjugate[7] * x243 + adjugate[8] * x241;
    const scalar_t x265  = adjugate[6] * x248 - adjugate[7] * x245 + adjugate[8] * x250;
    const scalar_t x266  = -adjugate[6] * x257 + adjugate[7] * x259 - adjugate[8] * x254;
    const scalar_t x267  = lmbda * POW2(x194) + POW2(x195) * x52 + x55;
    const scalar_t x268  = x194 * x252 + x195 * x253 + x74;
    const scalar_t x269  = x194 * x258 - x195 * x198 * x52 + x91;
    const scalar_t x270  = adjugate[0] * x267 - adjugate[1] * x269 + adjugate[2] * x268;
    const scalar_t x271  = lmbda * POW2(x201) + POW2(x202) * x52 + x95;
    const scalar_t x272  = x100 + x197 * x252 - x198 * x253;
    const scalar_t x273  = adjugate[0] * x268 - adjugate[1] * x272 + adjugate[2] * x271;
    const scalar_t x274  = lmbda * POW2(x197) + x104 + POW2(x198) * x52;
    const scalar_t x275  = -adjugate[0] * x269 + adjugate[1] * x274 - adjugate[2] * x272;
    const scalar_t x276  = adjugate[3] * x267 - adjugate[4] * x269 + adjugate[5] * x268;
    const scalar_t x277  = adjugate[3] * x268 - adjugate[4] * x272 + adjugate[5] * x271;
    const scalar_t x278  = -adjugate[3] * x269 + adjugate[4] * x274 - adjugate[5] * x272;
    S_ikmn_canonical[0]  = x107 * (adjugate[0] * x93 + adjugate[1] * x106 + adjugate[2] * x102);
    S_ikmn_canonical[1]  = x107 * (adjugate[3] * x93 + adjugate[4] * x106 + adjugate[5] * x102);
    S_ikmn_canonical[2]  = x107 * (adjugate[6] * x93 + adjugate[7] * x106 + adjugate[8] * x102);
    S_ikmn_canonical[3]  = x107 * (adjugate[3] * x108 + adjugate[4] * x110 + adjugate[5] * x109);
    S_ikmn_canonical[4]  = x107 * (adjugate[6] * x108 + adjugate[7] * x110 + adjugate[8] * x109);
    S_ikmn_canonical[5]  = x107 * (adjugate[6] * (adjugate[6] * x56 - adjugate[7] * x92 + adjugate[8] * x75) +
                                  adjugate[7] * (-adjugate[6] * x92 + adjugate[7] * x105 - adjugate[8] * x101) +
                                  adjugate[8] * (adjugate[6] * x75 - adjugate[7] * x101 + adjugate[8] * x96));
    S_ikmn_canonical[6]  = x187 * (adjugate[0] * x171 + adjugate[1] * x178 + adjugate[2] * x186);
    S_ikmn_canonical[7]  = x187 * (adjugate[3] * x171 + adjugate[4] * x178 + adjugate[5] * x186);
    S_ikmn_canonical[8]  = x187 * (adjugate[6] * x171 + adjugate[7] * x178 + adjugate[8] * x186);
    S_ikmn_canonical[9]  = x187 * (adjugate[0] * x188 + adjugate[1] * x189 + adjugate[2] * x190);
    S_ikmn_canonical[10] = x187 * (adjugate[3] * x188 + adjugate[4] * x189 + adjugate[5] * x190);
    S_ikmn_canonical[11] = x187 * (adjugate[6] * x188 + adjugate[7] * x189 + adjugate[8] * x190);
    S_ikmn_canonical[12] = x187 * (adjugate[0] * x191 + adjugate[1] * x192 + adjugate[2] * x193);
    S_ikmn_canonical[13] = x187 * (adjugate[3] * x191 + adjugate[4] * x192 + adjugate[5] * x193);
    S_ikmn_canonical[14] = x187 * (adjugate[6] * x191 + adjugate[7] * x192 + adjugate[8] * x193);
    S_ikmn_canonical[15] = x187 * (adjugate[0] * x205 + adjugate[1] * x210 + adjugate[2] * x214);
    S_ikmn_canonical[16] = x187 * (adjugate[3] * x205 + adjugate[4] * x210 + adjugate[5] * x214);
    S_ikmn_canonical[17] = x187 * (adjugate[6] * x205 + adjugate[7] * x210 + adjugate[8] * x214);
    S_ikmn_canonical[18] = x187 * (adjugate[0] * x215 + adjugate[1] * x216 + adjugate[2] * x217);
    S_ikmn_canonical[19] = x187 * (adjugate[3] * x215 + adjugate[4] * x216 + adjugate[5] * x217);
    S_ikmn_canonical[20] = x187 * (adjugate[6] * x215 + adjugate[7] * x216 + adjugate[8] * x217);
    S_ikmn_canonical[21] = x187 * (adjugate[0] * x218 + adjugate[1] * x219 + adjugate[2] * x220);
    S_ikmn_canonical[22] = x187 * (adjugate[3] * x218 + adjugate[4] * x219 + adjugate[5] * x220);
    S_ikmn_canonical[23] = x187 * (adjugate[6] * x218 + adjugate[7] * x219 + adjugate[8] * x220);
    S_ikmn_canonical[24] = x107 * (adjugate[0] * x226 + adjugate[1] * x233 + adjugate[2] * x231);
    S_ikmn_canonical[25] = x107 * (adjugate[3] * x226 + adjugate[4] * x233 + adjugate[5] * x231);
    S_ikmn_canonical[26] = x107 * (adjugate[6] * x226 + adjugate[7] * x233 + adjugate[8] * x231);
    S_ikmn_canonical[27] = x107 * (adjugate[3] * x234 + adjugate[4] * x236 + adjugate[5] * x235);
    S_ikmn_canonical[28] = x107 * (adjugate[6] * x234 + adjugate[7] * x236 + adjugate[8] * x235);
    S_ikmn_canonical[29] = x107 * (adjugate[6] * (adjugate[6] * x221 - adjugate[7] * x225 + adjugate[8] * x224) +
                                   adjugate[7] * (-adjugate[6] * x225 + adjugate[7] * x232 - adjugate[8] * x230) +
                                   adjugate[8] * (adjugate[6] * x224 - adjugate[7] * x230 + adjugate[8] * x227));
    S_ikmn_canonical[30] = x187 * (adjugate[0] * x244 + adjugate[1] * x251 + adjugate[2] * x260);
    S_ikmn_canonical[31] = x187 * (adjugate[3] * x244 + adjugate[4] * x251 + adjugate[5] * x260);
    S_ikmn_canonical[32] = x187 * (adjugate[6] * x244 + adjugate[7] * x251 + adjugate[8] * x260);
    S_ikmn_canonical[33] = x187 * (adjugate[0] * x261 + adjugate[1] * x262 + adjugate[2] * x263);
    S_ikmn_canonical[34] = x187 * (adjugate[3] * x261 + adjugate[4] * x262 + adjugate[5] * x263);
    S_ikmn_canonical[35] = x187 * (adjugate[6] * x261 + adjugate[7] * x262 + adjugate[8] * x263);
    S_ikmn_canonical[36] = x187 * (adjugate[0] * x264 + adjugate[1] * x265 + adjugate[2] * x266);
    S_ikmn_canonical[37] = x187 * (adjugate[3] * x264 + adjugate[4] * x265 + adjugate[5] * x266);
    S_ikmn_canonical[38] = x187 * (adjugate[6] * x264 + adjugate[7] * x265 + adjugate[8] * x266);
    S_ikmn_canonical[39] = x107 * (adjugate[0] * x270 + adjugate[1] * x275 + adjugate[2] * x273);
    S_ikmn_canonical[40] = x107 * (adjugate[3] * x270 + adjugate[4] * x275 + adjugate[5] * x273);
    S_ikmn_canonical[41] = x107 * (adjugate[6] * x270 + adjugate[7] * x275 + adjugate[8] * x273);
    S_ikmn_canonical[42] = x107 * (adjugate[3] * x276 + adjugate[4] * x278 + adjugate[5] * x277);
    S_ikmn_canonical[43] = x107 * (adjugate[6] * x276 + adjugate[7] * x278 + adjugate[8] * x277);
    S_ikmn_canonical[44] = x107 * (adjugate[6] * (adjugate[6] * x267 - adjugate[7] * x269 + adjugate[8] * x268) +
                                   adjugate[7] * (-adjugate[6] * x269 + adjugate[7] * x274 - adjugate[8] * x272) +
                                   adjugate[8] * (adjugate[6] * x268 - adjugate[7] * x272 + adjugate[8] * x271));
}

static SFEM_INLINE void hex8_S_ikmn_neohookean_smith_active_strain_add(const scalar_t *const SFEM_RESTRICT adjugate,
                                                                   const scalar_t                      jacobian_determinant,
                                                                   const scalar_t                      qx,
                                                                   const scalar_t                      qy,
                                                                   const scalar_t                      qz,
                                                                   const scalar_t                      qw,
                                                                   const scalar_t *const SFEM_RESTRICT F,
                                                                   const scalar_t                      lmda,
                                                                   const scalar_t                      mu,
                                                                   const scalar_t                      lmbda,
                                                                   const scalar_t *const SFEM_RESTRICT Fa_inv,
                                                                   const scalar_t                      Ja,
                                                                   scalar_t *const SFEM_RESTRICT       S_ikmn_canonical) {
    // mundane ops: 1494 divs: 3 sqrts: 0
    // total ops: 1518
    const scalar_t x0  = F[4] * Fa_inv[4];
    const scalar_t x1  = F[8] * Fa_inv[8];
    const scalar_t x2  = Fa_inv[0] * x1;
    const scalar_t x3  = F[4] * Fa_inv[5];
    const scalar_t x4  = F[8] * Fa_inv[6];
    const scalar_t x5  = Fa_inv[1] * x4;
    const scalar_t x6  = F[4] * Fa_inv[3];
    const scalar_t x7  = F[8] * Fa_inv[7];
    const scalar_t x8  = Fa_inv[2] * x7;
    const scalar_t x9  = F[5] * Fa_inv[7];
    const scalar_t x10 = F[7] * Fa_inv[5];
    const scalar_t x11 = Fa_inv[0] * x10;
    const scalar_t x12 = F[5] * Fa_inv[8];
    const scalar_t x13 = F[7] * Fa_inv[3];
    const scalar_t x14 = Fa_inv[1] * x13;
    const scalar_t x15 = F[5] * Fa_inv[6];
    const scalar_t x16 = F[7] * Fa_inv[4];
    const scalar_t x17 = Fa_inv[2] * x16;
    const scalar_t x18 = Fa_inv[0] * x7;
    const scalar_t x19 = Fa_inv[1] * x1;
    const scalar_t x20 = Fa_inv[2] * x4;
    const scalar_t x21 = Fa_inv[0] * x16;
    const scalar_t x22 = Fa_inv[1] * x10;
    const scalar_t x23 = Fa_inv[2] * x13;
    const scalar_t x24 = x0 * x2 - x0 * x20 + x11 * x9 + x12 * x14 - x12 * x21 + x15 * x17 - x15 * x22 - x18 * x3 - x19 * x6 -
                         x23 * x9 + x3 * x5 + x6 * x8;
    const scalar_t x25 = F[0] * Fa_inv[0];
    const scalar_t x26 = F[1] * Fa_inv[3];
    const scalar_t x27 = F[2] * Fa_inv[6];
    const scalar_t x28 = x25 + x26 + x27;
    const scalar_t x29 = F[0] * Fa_inv[1];
    const scalar_t x30 = F[1] * Fa_inv[4];
    const scalar_t x31 = F[2] * Fa_inv[7];
    const scalar_t x32 = x29 + x30 + x31;
    const scalar_t x33 = F[0] * Fa_inv[2];
    const scalar_t x34 = F[1] * Fa_inv[5];
    const scalar_t x35 = F[2] * Fa_inv[8];
    const scalar_t x36 = x33 + x34 + x35;
    const scalar_t x37 = Fa_inv[0] * x28 + Fa_inv[1] * x32 + Fa_inv[2] * x36;
    const scalar_t x38 = F[3] * Fa_inv[0];
    const scalar_t x39 = x15 + x38 + x6;
    const scalar_t x40 = F[3] * Fa_inv[1];
    const scalar_t x41 = x0 + x40 + x9;
    const scalar_t x42 = F[3] * Fa_inv[2];
    const scalar_t x43 = x12 + x3 + x42;
    const scalar_t x44 = F[6] * Fa_inv[0];
    const scalar_t x45 = x13 + x4 + x44;
    const scalar_t x46 = F[6] * Fa_inv[1];
    const scalar_t x47 = x16 + x46 + x7;
    const scalar_t x48 = F[6] * Fa_inv[2];
    const scalar_t x49 = x1 + x10 + x48;
    const scalar_t x50 =
            POW2(x28) + POW2(x32) + POW2(x36) + POW2(x39) + POW2(x41) + POW2(x43) + POW2(x45) + POW2(x47) + POW2(x49) + 1;
    const scalar_t x51 = mu / POW2(x50);
    const scalar_t x52 = 2 * x51;
    const scalar_t x53 = mu * (POW2(Fa_inv[0]) + POW2(Fa_inv[1]) + POW2(Fa_inv[2]));
    const scalar_t x54 = 1.0 / x50;
    const scalar_t x55 = -x53 * x54 + x53;
    const scalar_t x56 = lmbda * POW2(x24) + POW2(x37) * x52 + x55;
    const scalar_t x57 = Fa_inv[8] * x16;
    const scalar_t x58 = Fa_inv[6] * x10;
    const scalar_t x59 = Fa_inv[7] * x13;
    const scalar_t x60 = Fa_inv[7] * x44;
    const scalar_t x61 = Fa_inv[8] * x46;
    const scalar_t x62 = Fa_inv[6] * x48;
    const scalar_t x63 = Fa_inv[7] * x10;
    const scalar_t x64 = Fa_inv[8] * x13;
    const scalar_t x65 = Fa_inv[6] * x16;
    const scalar_t x66 = Fa_inv[8] * x44;
    const scalar_t x67 = Fa_inv[6] * x46;
    const scalar_t x68 = Fa_inv[7] * x48;
    const scalar_t x69 = x0 * x62 - x0 * x66 + x3 * x60 - x3 * x67 + x38 * x57 - x38 * x63 + x40 * x58 - x40 * x64 + x42 * x59 -
                         x42 * x65 + x6 * x61 - x6 * x68;
    const scalar_t x70 = lmbda * x24;
    const scalar_t x71 = Fa_inv[6] * x28 + Fa_inv[7] * x32 + Fa_inv[8] * x36;
    const scalar_t x72 = x37 * x52;
    const scalar_t x73 = mu * (Fa_inv[0] * Fa_inv[6] + Fa_inv[1] * Fa_inv[7] + Fa_inv[2] * Fa_inv[8]);
    const scalar_t x74 = -x54 * x73 + x73;
    const scalar_t x75 = x69 * x70 + x71 * x72 + x74;
    const scalar_t x76 = Fa_inv[4] * x1;
    const scalar_t x77 = Fa_inv[5] * x4;
    const scalar_t x78 = Fa_inv[3] * x7;
    const scalar_t x79 = Fa_inv[5] * x44;
    const scalar_t x80 = Fa_inv[3] * x46;
    const scalar_t x81 = Fa_inv[4] * x48;
    const scalar_t x82 = Fa_inv[5] * x7;
    const scalar_t x83 = Fa_inv[3] * x1;
    const scalar_t x84 = Fa_inv[4] * x4;
    const scalar_t x85 = Fa_inv[4] * x44;
    const scalar_t x86 = Fa_inv[5] * x46;
    const scalar_t x87 = Fa_inv[3] * x48;
    const scalar_t x88 = x12 * x80 - x12 * x85 + x15 * x81 - x15 * x86 + x38 * x76 - x38 * x82 + x40 * x77 - x40 * x83 +
                         x42 * x78 - x42 * x84 + x79 * x9 - x87 * x9;
    const scalar_t x89  = Fa_inv[3] * x28 + Fa_inv[4] * x32 + Fa_inv[5] * x36;
    const scalar_t x90  = mu * (Fa_inv[0] * Fa_inv[3] + Fa_inv[1] * Fa_inv[4] + Fa_inv[2] * Fa_inv[5]);
    const scalar_t x91  = x54 * x90 - x90;
    const scalar_t x92  = x70 * x88 - x72 * x89 + x91;
    const scalar_t x93  = adjugate[0] * x56 - adjugate[1] * x92 + adjugate[2] * x75;
    const scalar_t x94  = mu * (POW2(Fa_inv[6]) + POW2(Fa_inv[7]) + POW2(Fa_inv[8]));
    const scalar_t x95  = -x54 * x94 + x94;
    const scalar_t x96  = lmbda * POW2(x69) + x52 * POW2(x71) + x95;
    const scalar_t x97  = lmbda * x88;
    const scalar_t x98  = x52 * x89;
    const scalar_t x99  = mu * (Fa_inv[3] * Fa_inv[6] + Fa_inv[4] * Fa_inv[7] + Fa_inv[5] * Fa_inv[8]);
    const scalar_t x100 = x54 * x99 - x99;
    const scalar_t x101 = x100 + x69 * x97 - x71 * x98;
    const scalar_t x102 = adjugate[0] * x75 - adjugate[1] * x101 + adjugate[2] * x96;
    const scalar_t x103 = mu * (POW2(Fa_inv[3]) + POW2(Fa_inv[4]) + POW2(Fa_inv[5]));
    const scalar_t x104 = -x103 * x54 + x103;
    const scalar_t x105 = lmbda * POW2(x88) + x104 + x52 * POW2(x89);
    const scalar_t x106 = -adjugate[0] * x92 + adjugate[1] * x105 - adjugate[2] * x101;
    const scalar_t x107 = Ja * qw / jacobian_determinant;
    const scalar_t x108 = adjugate[3] * x56 - adjugate[4] * x92 + adjugate[5] * x75;
    const scalar_t x109 = adjugate[3] * x75 - adjugate[4] * x101 + adjugate[5] * x96;
    const scalar_t x110 = -adjugate[3] * x92 + adjugate[4] * x105 - adjugate[5] * x101;
    const scalar_t x111 = x11 * x31 + x14 * x35 + x17 * x27 - x18 * x34 - x19 * x26 + x2 * x30 - x20 * x30 - x21 * x35 -
                          x22 * x27 - x23 * x31 + x26 * x8 + x34 * x5;
    const scalar_t x112 = Fa_inv[0] * x39 + Fa_inv[1] * x41 + Fa_inv[2] * x43;
    const scalar_t x113 = 4 * x111 * x70 - 4 * x112 * x72;
    const scalar_t x114 = x25 * x57 - x25 * x63 + x26 * x61 - x26 * x68 + x29 * x58 - x29 * x64 + x30 * x62 - x30 * x66 +
                          x33 * x59 - x33 * x65 + x34 * x60 - x34 * x67;
    const scalar_t x115 = 4 * x70;
    const scalar_t x116 = Fa_inv[6] * x39 + Fa_inv[7] * x41 + Fa_inv[8] * x43;
    const scalar_t x117 = 8 * x51;
    const scalar_t x118 = x117 * x37;
    const scalar_t x119 = x0 * x25;
    const scalar_t x120 = 4 * x1;
    const scalar_t x121 = x25 * x3;
    const scalar_t x122 = 4 * x7;
    const scalar_t x123 = x29 * x6;
    const scalar_t x124 = x29 * x3;
    const scalar_t x125 = 4 * x4;
    const scalar_t x126 = x33 * x6;
    const scalar_t x127 = x0 * x33;
    const scalar_t x128 = x12 * x25;
    const scalar_t x129 = 4 * x16;
    const scalar_t x130 = x25 * x9;
    const scalar_t x131 = 4 * x10;
    const scalar_t x132 = x12 * x29;
    const scalar_t x133 = 4 * x13;
    const scalar_t x134 = x15 * x29;
    const scalar_t x135 = x33 * x9;
    const scalar_t x136 = x15 * x33;
    const scalar_t x137 = x30 * x38;
    const scalar_t x138 = x34 * x38;
    const scalar_t x139 = x26 * x40;
    const scalar_t x140 = x34 * x40;
    const scalar_t x141 = x26 * x42;
    const scalar_t x142 = x30 * x42;
    const scalar_t x143 = x12 * x30;
    const scalar_t x144 = 4 * x44;
    const scalar_t x145 = x34 * x9;
    const scalar_t x146 = x12 * x26;
    const scalar_t x147 = 4 * x46;
    const scalar_t x148 = x15 * x34;
    const scalar_t x149 = x26 * x9;
    const scalar_t x150 = 4 * x48;
    const scalar_t x151 = x15 * x30;
    const scalar_t x152 = x35 * x38;
    const scalar_t x153 = x31 * x38;
    const scalar_t x154 = x35 * x40;
    const scalar_t x155 = x27 * x40;
    const scalar_t x156 = x31 * x42;
    const scalar_t x157 = x27 * x42;
    const scalar_t x158 = x0 * x35;
    const scalar_t x159 = x3 * x31;
    const scalar_t x160 = x35 * x6;
    const scalar_t x161 = x27 * x3;
    const scalar_t x162 = x31 * x6;
    const scalar_t x163 = x0 * x27;
    const scalar_t x164 =
            lmbda *
            (Fa_inv[0] * Fa_inv[4] * Fa_inv[8] - Fa_inv[0] * Fa_inv[5] * Fa_inv[7] - Fa_inv[1] * Fa_inv[3] * Fa_inv[8] +
             Fa_inv[1] * Fa_inv[5] * Fa_inv[6] + Fa_inv[2] * Fa_inv[3] * Fa_inv[7] - Fa_inv[2] * Fa_inv[4] * Fa_inv[6]) *
            (-x119 * x120 + x120 * x123 + x120 * x137 - x120 * x139 + x121 * x122 - x122 * x126 - x122 * x138 + x122 * x141 -
             x124 * x125 + x125 * x127 + x125 * x140 - x125 * x142 + x128 * x129 - x129 * x136 - x129 * x152 + x129 * x157 -
             x130 * x131 + x131 * x134 + x131 * x153 - x131 * x155 - x132 * x133 + x133 * x135 + x133 * x154 - x133 * x156 -
             x143 * x144 + x144 * x145 + x144 * x158 - x144 * x159 + x146 * x147 - x147 * x148 - x147 * x160 + x147 * x161 -
             x149 * x150 + x150 * x151 + x150 * x162 - x150 * x163 + 4 + 3 * mu / lmda);
    const scalar_t x165 = F[7] * x164;
    const scalar_t x166 = -x114 * x115 + x116 * x118 + x165;
    const scalar_t x167 = x25 * x76 - x25 * x82 + x27 * x81 - x27 * x86 + x29 * x77 - x29 * x83 + x31 * x79 - x31 * x87 +
                          x33 * x78 - x33 * x84 + x35 * x80 - x35 * x85;
    const scalar_t x168 = Fa_inv[3] * x39 + Fa_inv[4] * x41 + Fa_inv[5] * x43;
    const scalar_t x169 = F[8] * x164;
    const scalar_t x170 = x115 * x167 + x118 * x168 - x169;
    const scalar_t x171 = -adjugate[0] * x113 + adjugate[1] * x170 + adjugate[2] * x166;
    const scalar_t x172 = 4 * x167 * x97 - 4 * x168 * x98;
    const scalar_t x173 = 4 * x97;
    const scalar_t x174 = x117 * x89;
    const scalar_t x175 = x111 * x173 + x112 * x174 + x169;
    const scalar_t x176 = F[6] * x164;
    const scalar_t x177 = x114 * x173 + x116 * x174 - x176;
    const scalar_t x178 = adjugate[0] * x175 - adjugate[1] * x172 + adjugate[2] * x177;
    const scalar_t x179 = lmbda * x69;
    const scalar_t x180 = x52 * x71;
    const scalar_t x181 = 4 * x114 * x179 - 4 * x116 * x180;
    const scalar_t x182 = 4 * x179;
    const scalar_t x183 = x117 * x71;
    const scalar_t x184 = x111 * x182 - x112 * x183 + x165;
    const scalar_t x185 = x167 * x182 + x168 * x183 + x176;
    const scalar_t x186 = -adjugate[0] * x184 + adjugate[1] * x185 - adjugate[2] * x181;
    const scalar_t x187 = (1.0 / 4.0) * x107;
    const scalar_t x188 = -adjugate[3] * x113 + adjugate[4] * x170 + adjugate[5] * x166;
    const scalar_t x189 = adjugate[3] * x175 - adjugate[4] * x172 + adjugate[5] * x177;
    const scalar_t x190 = -adjugate[3] * x184 + adjugate[4] * x185 - adjugate[5] * x181;
    const scalar_t x191 = -adjugate[6] * x113 + adjugate[7] * x170 + adjugate[8] * x166;
    const scalar_t x192 = adjugate[6] * x175 - adjugate[7] * x172 + adjugate[8] * x177;
    const scalar_t x193 = -adjugate[6] * x184 + adjugate[7] * x185 - adjugate[8] * x181;
    const scalar_t x194 = Fa_inv[0] * x143 - Fa_inv[0] * x145 - Fa_inv[0] * x158 + Fa_inv[0] * x159 - Fa_inv[1] * x146 +
                          Fa_inv[1] * x148 + Fa_inv[1] * x160 - Fa_inv[1] * x161 + Fa_inv[2] * x149 - Fa_inv[2] * x151 -
                          Fa_inv[2] * x162 + Fa_inv[2] * x163;
    const scalar_t x195 = Fa_inv[0] * x45 + Fa_inv[1] * x47 + Fa_inv[2] * x49;
    const scalar_t x196 = 4 * x194 * x70 + 4 * x195 * x72;
    const scalar_t x197 = -Fa_inv[3] * x132 + Fa_inv[3] * x135 + Fa_inv[3] * x154 - Fa_inv[3] * x156 + Fa_inv[4] * x128 -
                          Fa_inv[4] * x136 - Fa_inv[4] * x152 + Fa_inv[4] * x157 - Fa_inv[5] * x130 + Fa_inv[5] * x134 +
                          Fa_inv[5] * x153 - Fa_inv[5] * x155;
    const scalar_t x198 = Fa_inv[3] * x45 + Fa_inv[4] * x47 + Fa_inv[5] * x49;
    const scalar_t x199 = F[5] * x164;
    const scalar_t x200 = -x115 * x197 + x118 * x198 + x199;
    const scalar_t x201 = Fa_inv[6] * x124 - Fa_inv[6] * x127 - Fa_inv[6] * x140 + Fa_inv[6] * x142 - Fa_inv[7] * x121 +
                          Fa_inv[7] * x126 + Fa_inv[7] * x138 - Fa_inv[7] * x141 + Fa_inv[8] * x119 - Fa_inv[8] * x123 -
                          Fa_inv[8] * x137 + Fa_inv[8] * x139;
    const scalar_t x202  = Fa_inv[6] * x45 + Fa_inv[7] * x47 + Fa_inv[8] * x49;
    const scalar_t x203  = F[4] * x164;
    const scalar_t x204  = x115 * x201 + x118 * x202 - x203;
    const scalar_t x205  = adjugate[0] * x196 + adjugate[1] * x200 + adjugate[2] * x204;
    const scalar_t x206  = 4 * x197 * x97 + 4 * x198 * x98;
    const scalar_t x207  = F[3] * x164;
    const scalar_t x208  = -x173 * x201 + x174 * x202 + x207;
    const scalar_t x209  = x173 * x194 - x174 * x195 + x199;
    const scalar_t x210  = -adjugate[0] * x209 + adjugate[1] * x206 + adjugate[2] * x208;
    const scalar_t x211  = 4 * x179 * x201 + 4 * x180 * x202;
    const scalar_t x212  = x182 * x194 + x183 * x195 + x203;
    const scalar_t x213  = x182 * x197 - x183 * x198 + x207;
    const scalar_t x214  = adjugate[0] * x212 - adjugate[1] * x213 + adjugate[2] * x211;
    const scalar_t x215  = adjugate[3] * x196 + adjugate[4] * x200 + adjugate[5] * x204;
    const scalar_t x216  = -adjugate[3] * x209 + adjugate[4] * x206 + adjugate[5] * x208;
    const scalar_t x217  = adjugate[3] * x212 - adjugate[4] * x213 + adjugate[5] * x211;
    const scalar_t x218  = adjugate[6] * x196 + adjugate[7] * x200 + adjugate[8] * x204;
    const scalar_t x219  = -adjugate[6] * x209 + adjugate[7] * x206 + adjugate[8] * x208;
    const scalar_t x220  = adjugate[6] * x212 - adjugate[7] * x213 + adjugate[8] * x211;
    const scalar_t x221  = lmbda * POW2(x111) + POW2(x112) * x52 + x55;
    const scalar_t x222  = lmbda * x111;
    const scalar_t x223  = x112 * x52;
    const scalar_t x224  = x114 * x222 + x116 * x223 + x74;
    const scalar_t x225  = x167 * x222 - x168 * x223 + x91;
    const scalar_t x226  = adjugate[0] * x221 - adjugate[1] * x225 + adjugate[2] * x224;
    const scalar_t x227  = lmbda * POW2(x114) + POW2(x116) * x52 + x95;
    const scalar_t x228  = lmbda * x167;
    const scalar_t x229  = x168 * x52;
    const scalar_t x230  = x100 + x114 * x228 - x116 * x229;
    const scalar_t x231  = adjugate[0] * x224 - adjugate[1] * x230 + adjugate[2] * x227;
    const scalar_t x232  = lmbda * POW2(x167) + x104 + POW2(x168) * x52;
    const scalar_t x233  = -adjugate[0] * x225 + adjugate[1] * x232 - adjugate[2] * x230;
    const scalar_t x234  = adjugate[3] * x221 - adjugate[4] * x225 + adjugate[5] * x224;
    const scalar_t x235  = adjugate[3] * x224 - adjugate[4] * x230 + adjugate[5] * x227;
    const scalar_t x236  = -adjugate[3] * x225 + adjugate[4] * x232 - adjugate[5] * x230;
    const scalar_t x237  = 4 * x194 * x222 - 4 * x195 * x223;
    const scalar_t x238  = 4 * x222;
    const scalar_t x239  = x112 * x117;
    const scalar_t x240  = F[1] * x164;
    const scalar_t x241  = -x201 * x238 + x202 * x239 + x240;
    const scalar_t x242  = F[2] * x164;
    const scalar_t x243  = x197 * x238 + x198 * x239 - x242;
    const scalar_t x244  = -adjugate[0] * x237 + adjugate[1] * x243 + adjugate[2] * x241;
    const scalar_t x245  = 4 * x197 * x228 - 4 * x198 * x229;
    const scalar_t x246  = 4 * x228;
    const scalar_t x247  = x117 * x168;
    const scalar_t x248  = x194 * x246 + x195 * x247 + x242;
    const scalar_t x249  = F[0] * x164;
    const scalar_t x250  = x201 * x246 + x202 * x247 - x249;
    const scalar_t x251  = adjugate[0] * x248 - adjugate[1] * x245 + adjugate[2] * x250;
    const scalar_t x252  = lmbda * x201;
    const scalar_t x253  = x202 * x52;
    const scalar_t x254  = 4 * x114 * x252 - 4 * x116 * x253;
    const scalar_t x255  = 4 * x114;
    const scalar_t x256  = x116 * x117;
    const scalar_t x257  = lmbda * x194 * x255 - x195 * x256 + x240;
    const scalar_t x258  = lmbda * x197;
    const scalar_t x259  = x198 * x256 + x249 + x255 * x258;
    const scalar_t x260  = -adjugate[0] * x257 + adjugate[1] * x259 - adjugate[2] * x254;
    const scalar_t x261  = -adjugate[3] * x237 + adjugate[4] * x243 + adjugate[5] * x241;
    const scalar_t x262  = adjugate[3] * x248 - adjugate[4] * x245 + adjugate[5] * x250;
    const scalar_t x263  = -adjugate[3] * x257 + adjugate[4] * x259 - adjugate[5] * x254;
    const scalar_t x264  = -adjugate[6] * x237 + adjugate[7] * x243 + adjugate[8] * x241;
    const scalar_t x265  = adjugate[6] * x248 - adjugate[7] * x245 + adjugate[8] * x250;
    const scalar_t x266  = -adjugate[6] * x257 + adjugate[7] * x259 - adjugate[8] * x254;
    const scalar_t x267  = lmbda * POW2(x194) + POW2(x195) * x52 + x55;
    const scalar_t x268  = x194 * x252 + x195 * x253 + x74;
    const scalar_t x269  = x194 * x258 - x195 * x198 * x52 + x91;
    const scalar_t x270  = adjugate[0] * x267 - adjugate[1] * x269 + adjugate[2] * x268;
    const scalar_t x271  = lmbda * POW2(x201) + POW2(x202) * x52 + x95;
    const scalar_t x272  = x100 + x197 * x252 - x198 * x253;
    const scalar_t x273  = adjugate[0] * x268 - adjugate[1] * x272 + adjugate[2] * x271;
    const scalar_t x274  = lmbda * POW2(x197) + x104 + POW2(x198) * x52;
    const scalar_t x275  = -adjugate[0] * x269 + adjugate[1] * x274 - adjugate[2] * x272;
    const scalar_t x276  = adjugate[3] * x267 - adjugate[4] * x269 + adjugate[5] * x268;
    const scalar_t x277  = adjugate[3] * x268 - adjugate[4] * x272 + adjugate[5] * x271;
    const scalar_t x278  = -adjugate[3] * x269 + adjugate[4] * x274 - adjugate[5] * x272;
    S_ikmn_canonical[0]  += x107 * (adjugate[0] * x93 + adjugate[1] * x106 + adjugate[2] * x102);
    S_ikmn_canonical[1]  += x107 * (adjugate[3] * x93 + adjugate[4] * x106 + adjugate[5] * x102);
    S_ikmn_canonical[2]  += x107 * (adjugate[6] * x93 + adjugate[7] * x106 + adjugate[8] * x102);
    S_ikmn_canonical[3]  += x107 * (adjugate[3] * x108 + adjugate[4] * x110 + adjugate[5] * x109);
    S_ikmn_canonical[4]  += x107 * (adjugate[6] * x108 + adjugate[7] * x110 + adjugate[8] * x109);
    S_ikmn_canonical[5]  += x107 * (adjugate[6] * (adjugate[6] * x56 - adjugate[7] * x92 + adjugate[8] * x75) +
                                  adjugate[7] * (-adjugate[6] * x92 + adjugate[7] * x105 - adjugate[8] * x101) +
                                  adjugate[8] * (adjugate[6] * x75 - adjugate[7] * x101 + adjugate[8] * x96));
    S_ikmn_canonical[6]  += x187 * (adjugate[0] * x171 + adjugate[1] * x178 + adjugate[2] * x186);
    S_ikmn_canonical[7]  += x187 * (adjugate[3] * x171 + adjugate[4] * x178 + adjugate[5] * x186);
    S_ikmn_canonical[8]  += x187 * (adjugate[6] * x171 + adjugate[7] * x178 + adjugate[8] * x186);
    S_ikmn_canonical[9]  += x187 * (adjugate[0] * x188 + adjugate[1] * x189 + adjugate[2] * x190);
    S_ikmn_canonical[10] += x187 * (adjugate[3] * x188 + adjugate[4] * x189 + adjugate[5] * x190);
    S_ikmn_canonical[11] += x187 * (adjugate[6] * x188 + adjugate[7] * x189 + adjugate[8] * x190);
    S_ikmn_canonical[12] += x187 * (adjugate[0] * x191 + adjugate[1] * x192 + adjugate[2] * x193);
    S_ikmn_canonical[13] += x187 * (adjugate[3] * x191 + adjugate[4] * x192 + adjugate[5] * x193);
    S_ikmn_canonical[14] += x187 * (adjugate[6] * x191 + adjugate[7] * x192 + adjugate[8] * x193);
    S_ikmn_canonical[15] += x187 * (adjugate[0] * x205 + adjugate[1] * x210 + adjugate[2] * x214);
    S_ikmn_canonical[16] += x187 * (adjugate[3] * x205 + adjugate[4] * x210 + adjugate[5] * x214);
    S_ikmn_canonical[17] += x187 * (adjugate[6] * x205 + adjugate[7] * x210 + adjugate[8] * x214);
    S_ikmn_canonical[18] += x187 * (adjugate[0] * x215 + adjugate[1] * x216 + adjugate[2] * x217);
    S_ikmn_canonical[19] += x187 * (adjugate[3] * x215 + adjugate[4] * x216 + adjugate[5] * x217);
    S_ikmn_canonical[20] += x187 * (adjugate[6] * x215 + adjugate[7] * x216 + adjugate[8] * x217);
    S_ikmn_canonical[21] += x187 * (adjugate[0] * x218 + adjugate[1] * x219 + adjugate[2] * x220);
    S_ikmn_canonical[22] += x187 * (adjugate[3] * x218 + adjugate[4] * x219 + adjugate[5] * x220);
    S_ikmn_canonical[23] += x187 * (adjugate[6] * x218 + adjugate[7] * x219 + adjugate[8] * x220);
    S_ikmn_canonical[24] += x107 * (adjugate[0] * x226 + adjugate[1] * x233 + adjugate[2] * x231);
    S_ikmn_canonical[25] += x107 * (adjugate[3] * x226 + adjugate[4] * x233 + adjugate[5] * x231);
    S_ikmn_canonical[26] += x107 * (adjugate[6] * x226 + adjugate[7] * x233 + adjugate[8] * x231);
    S_ikmn_canonical[27] += x107 * (adjugate[3] * x234 + adjugate[4] * x236 + adjugate[5] * x235);
    S_ikmn_canonical[28] += x107 * (adjugate[6] * x234 + adjugate[7] * x236 + adjugate[8] * x235);
    S_ikmn_canonical[29] += x107 * (adjugate[6] * (adjugate[6] * x221 - adjugate[7] * x225 + adjugate[8] * x224) +
                                   adjugate[7] * (-adjugate[6] * x225 + adjugate[7] * x232 - adjugate[8] * x230) +
                                   adjugate[8] * (adjugate[6] * x224 - adjugate[7] * x230 + adjugate[8] * x227));
    S_ikmn_canonical[30] += x187 * (adjugate[0] * x244 + adjugate[1] * x251 + adjugate[2] * x260);
    S_ikmn_canonical[31] += x187 * (adjugate[3] * x244 + adjugate[4] * x251 + adjugate[5] * x260);
    S_ikmn_canonical[32] += x187 * (adjugate[6] * x244 + adjugate[7] * x251 + adjugate[8] * x260);
    S_ikmn_canonical[33] += x187 * (adjugate[0] * x261 + adjugate[1] * x262 + adjugate[2] * x263);
    S_ikmn_canonical[34] += x187 * (adjugate[3] * x261 + adjugate[4] * x262 + adjugate[5] * x263);
    S_ikmn_canonical[35] += x187 * (adjugate[6] * x261 + adjugate[7] * x262 + adjugate[8] * x263);
    S_ikmn_canonical[36] += x187 * (adjugate[0] * x264 + adjugate[1] * x265 + adjugate[2] * x266);
    S_ikmn_canonical[37] += x187 * (adjugate[3] * x264 + adjugate[4] * x265 + adjugate[5] * x266);
    S_ikmn_canonical[38] += x187 * (adjugate[6] * x264 + adjugate[7] * x265 + adjugate[8] * x266);
    S_ikmn_canonical[39] += x107 * (adjugate[0] * x270 + adjugate[1] * x275 + adjugate[2] * x273);
    S_ikmn_canonical[40] += x107 * (adjugate[3] * x270 + adjugate[4] * x275 + adjugate[5] * x273);
    S_ikmn_canonical[41] += x107 * (adjugate[6] * x270 + adjugate[7] * x275 + adjugate[8] * x273);
    S_ikmn_canonical[42] += x107 * (adjugate[3] * x276 + adjugate[4] * x278 + adjugate[5] * x277);
    S_ikmn_canonical[43] += x107 * (adjugate[6] * x276 + adjugate[7] * x278 + adjugate[8] * x277);
    S_ikmn_canonical[44] += x107 * (adjugate[6] * (adjugate[6] * x267 - adjugate[7] * x269 + adjugate[8] * x268) +
                                   adjugate[7] * (-adjugate[6] * x269 + adjugate[7] * x274 - adjugate[8] * x272) +
                                   adjugate[8] * (adjugate[6] * x268 - adjugate[7] * x272 + adjugate[8] * x271));
}
#endif /* SFEM_HEX8_PARTIAL_ASSEMBLY_NEOHOOKEAN_SMITH_ACTIVE_STRAIN_INLINE_H */
