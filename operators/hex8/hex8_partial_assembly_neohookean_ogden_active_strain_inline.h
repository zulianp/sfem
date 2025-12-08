#ifndef SFEM_HEX8_PARTIAL_ASSEMBLY_NEOHOOKEAN_OGDEN_ACTIVE_STRAIN_INLINE_H
#define SFEM_HEX8_PARTIAL_ASSEMBLY_NEOHOOKEAN_OGDEN_ACTIVE_STRAIN_INLINE_H

#include "hex8_partial_assembly_neohookean_inline.h"

#define HEX8_S_IKMN_SIZE 45
static SFEM_INLINE void hex8_S_ikmn_neohookean_ogden_active_strain(const scalar_t *const SFEM_RESTRICT adjugate,
                                                                   const scalar_t                      jacobian_determinant,
                                                                   const scalar_t                      qx,
                                                                   const scalar_t                      qy,
                                                                   const scalar_t                      qz,
                                                                   const scalar_t                      qw,
                                                                   const scalar_t *const SFEM_RESTRICT F,
                                                                   const scalar_t                      mu,
                                                                   const scalar_t                      lmbda,
                                                                   const scalar_t *const SFEM_RESTRICT Fa_inv,
                                                                   const scalar_t                      Ja,
                                                                   scalar_t *const SFEM_RESTRICT       S_ikmn_canonical) {
    // mundane ops: 1410 divs: 2 sqrts: 0
    // total ops: 1426
    const scalar_t x0  = POW2(Ja) * mu;
    const scalar_t x1  = x0 * (POW2(Fa_inv[0]) + POW2(Fa_inv[1]) + POW2(Fa_inv[2]));
    const scalar_t x2  = Fa_inv[0] * Fa_inv[4] * Fa_inv[8];
    const scalar_t x3  = F[4] * F[8];
    const scalar_t x4  = x2 * x3;
    const scalar_t x5  = Fa_inv[1] * Fa_inv[5] * Fa_inv[6];
    const scalar_t x6  = x3 * x5;
    const scalar_t x7  = Fa_inv[2] * Fa_inv[3] * Fa_inv[7];
    const scalar_t x8  = x3 * x7;
    const scalar_t x9  = Fa_inv[0] * Fa_inv[5] * Fa_inv[7];
    const scalar_t x10 = F[5] * F[7];
    const scalar_t x11 = x10 * x9;
    const scalar_t x12 = Fa_inv[1] * Fa_inv[3] * Fa_inv[8];
    const scalar_t x13 = x10 * x12;
    const scalar_t x14 = Fa_inv[2] * Fa_inv[4] * Fa_inv[6];
    const scalar_t x15 = x10 * x14;
    const scalar_t x16 = x3 * x9;
    const scalar_t x17 = x12 * x3;
    const scalar_t x18 = x14 * x3;
    const scalar_t x19 = x10 * x2;
    const scalar_t x20 = x10 * x5;
    const scalar_t x21 = x10 * x7;
    const scalar_t x22 = x11 + x13 + x15 - x16 - x17 - x18 - x19 - x20 - x21 + x4 + x6 + x8;
    const scalar_t x23 = F[3] * F[8];
    const scalar_t x24 = x23 * x9;
    const scalar_t x25 = x12 * x23;
    const scalar_t x26 = x14 * x23;
    const scalar_t x27 = F[5] * F[6];
    const scalar_t x28 = x2 * x27;
    const scalar_t x29 = x27 * x5;
    const scalar_t x30 = x27 * x7;
    const scalar_t x31 = F[3] * F[7];
    const scalar_t x32 = x2 * x31;
    const scalar_t x33 = x31 * x5;
    const scalar_t x34 = x31 * x7;
    const scalar_t x35 = F[4] * F[6];
    const scalar_t x36 = x35 * x9;
    const scalar_t x37 = x12 * x35;
    const scalar_t x38 = x14 * x35;
    const scalar_t x39 = x2 * x23;
    const scalar_t x40 = x23 * x5;
    const scalar_t x41 = x23 * x7;
    const scalar_t x42 = x27 * x9;
    const scalar_t x43 = x12 * x27;
    const scalar_t x44 = x14 * x27;
    const scalar_t x45 = x31 * x9;
    const scalar_t x46 = x12 * x31;
    const scalar_t x47 = x14 * x31;
    const scalar_t x48 = x2 * x35;
    const scalar_t x49 = x35 * x5;
    const scalar_t x50 = x35 * x7;
    const scalar_t x51 = F[0] * x11 + F[0] * x13 + F[0] * x15 - F[0] * x16 - F[0] * x17 - F[0] * x18 - F[0] * x19 - F[0] * x20 -
                         F[0] * x21 + F[0] * x4 + F[0] * x6 + F[0] * x8 + F[1] * x24 + F[1] * x25 + F[1] * x26 + F[1] * x28 +
                         F[1] * x29 + F[1] * x30 - F[1] * x39 - F[1] * x40 - F[1] * x41 - F[1] * x42 - F[1] * x43 - F[1] * x44 +
                         F[2] * x32 + F[2] * x33 + F[2] * x34 + F[2] * x36 + F[2] * x37 + F[2] * x38 - F[2] * x45 - F[2] * x46 -
                         F[2] * x47 - F[2] * x48 - F[2] * x49 - F[2] * x50;
    const scalar_t x52 = (1 / POW2(x51));
    const scalar_t x53 = POW2(x22) * x52;
    const scalar_t x54 = lmbda * log((POW3(Ja)) * x51);
    const scalar_t x55 = lmbda * x53 + mu * x53 + x1 - x53 * x54;
    const scalar_t x56 = x0 * (Fa_inv[0] * Fa_inv[6] + Fa_inv[1] * Fa_inv[7] + Fa_inv[2] * Fa_inv[8]);
    const scalar_t x57 = x22 * x52;
    const scalar_t x58 = x32 + x33 + x34 + x36 + x37 + x38 - x45 - x46 - x47 - x48 - x49 - x50;
    const scalar_t x59 = lmbda * x58;
    const scalar_t x60 = mu * x57;
    const scalar_t x61 = x54 * x57;
    const scalar_t x62 = x56 + x57 * x59 + x58 * x60 - x58 * x61;
    const scalar_t x63 = x0 * (Fa_inv[0] * Fa_inv[3] + Fa_inv[1] * Fa_inv[4] + Fa_inv[2] * Fa_inv[5]);
    const scalar_t x64 = -x24 - x25 - x26 - x28 - x29 - x30 + x39 + x40 + x41 + x42 + x43 + x44;
    const scalar_t x65 = lmbda * x64;
    const scalar_t x66 = -x57 * x65 - x60 * x64 + x61 * x64 + x63;
    const scalar_t x67 = adjugate[0] * x55 + adjugate[1] * x66 + adjugate[2] * x62;
    const scalar_t x68 = x0 * (POW2(Fa_inv[6]) + POW2(Fa_inv[7]) + POW2(Fa_inv[8]));
    const scalar_t x69 = x52 * POW2(x58);
    const scalar_t x70 = lmbda * x69 + mu * x69 - x54 * x69 + x68;
    const scalar_t x71 = x0 * (Fa_inv[3] * Fa_inv[6] + Fa_inv[4] * Fa_inv[7] + Fa_inv[5] * Fa_inv[8]);
    const scalar_t x72 = x52 * x58;
    const scalar_t x73 = mu * x64;
    const scalar_t x74 = x54 * x64;
    const scalar_t x75 = -x65 * x72 + x71 - x72 * x73 + x72 * x74;
    const scalar_t x76 = adjugate[0] * x62 + adjugate[1] * x75 + adjugate[2] * x70;
    const scalar_t x77 = x0 * (POW2(Fa_inv[3]) + POW2(Fa_inv[4]) + POW2(Fa_inv[5]));
    const scalar_t x78 = x52 * POW2(x64);
    const scalar_t x79 = lmbda * x78 + mu * x78 - x54 * x78 + x77;
    const scalar_t x80 = adjugate[0] * x66 + adjugate[1] * x79 + adjugate[2] * x75;
    const scalar_t x81 = qw / jacobian_determinant;
    const scalar_t x82 = adjugate[3] * x55 + adjugate[4] * x66 + adjugate[5] * x62;
    const scalar_t x83 = adjugate[3] * x62 + adjugate[4] * x75 + adjugate[5] * x70;
    const scalar_t x84 = adjugate[3] * x66 + adjugate[4] * x79 + adjugate[5] * x75;
    const scalar_t x85 = 1.0 / x51;
    const scalar_t x86 = F[0] * F[8];
    const scalar_t x87 = F[2] * F[6];
    const scalar_t x88 = -x12 * x86 + x12 * x87 - x14 * x86 + x14 * x87 + x2 * x86 - x2 * x87 + x5 * x86 - x5 * x87 + x7 * x86 -
                         x7 * x87 - x86 * x9 + x87 * x9;
    const scalar_t x89 = x85 * x88;
    const scalar_t x90 = lmbda + mu - x54;
    const scalar_t x91 = x64 * x89 * x90;
    const scalar_t x92 = -x12 - x14 + x2 + x5 + x7 - x9;
    const scalar_t x93 = mu * x92;
    const scalar_t x94 = F[8] * x93;
    const scalar_t x95 = x54 * x92;
    const scalar_t x96 = F[8] * x95;
    const scalar_t x97 = F[1] * F[8];
    const scalar_t x98 = F[2] * F[7];
    const scalar_t x99 = -x12 * x97 + x12 * x98 - x14 * x97 + x14 * x98 + x2 * x97 - x2 * x98 + x5 * x97 - x5 * x98 + x7 * x97 -
                         x7 * x98 - x9 * x97 + x9 * x98;
    const scalar_t x100 = x85 * x99;
    const scalar_t x101 = x100 * x65 + x100 * x73 - x100 * x74 + x94 - x96;
    const scalar_t x102 = F[6] * x93;
    const scalar_t x103 = F[6] * x95;
    const scalar_t x104 = F[0] * F[7];
    const scalar_t x105 = F[1] * F[6];
    const scalar_t x106 = -x104 * x12 - x104 * x14 + x104 * x2 + x104 * x5 + x104 * x7 - x104 * x9 + x105 * x12 + x105 * x14 -
                          x105 * x2 - x105 * x5 - x105 * x7 + x105 * x9;
    const scalar_t x107 = x106 * x85;
    const scalar_t x108 = -x102 + x103 + x107 * x65 + x107 * x73 - x107 * x74;
    const scalar_t x109 = adjugate[0] * x101 - adjugate[1] * x91 + adjugate[2] * x108;
    const scalar_t x110 = x22 * x85;
    const scalar_t x111 = x110 * x90;
    const scalar_t x112 = x111 * x99;
    const scalar_t x113 = F[7] * x93;
    const scalar_t x114 = F[7] * x95;
    const scalar_t x115 = lmbda * x110;
    const scalar_t x116 = mu * x110;
    const scalar_t x117 = x110 * x54;
    const scalar_t x118 = x106 * x115 + x106 * x116 - x106 * x117 - x113 + x114;
    const scalar_t x119 = lmbda * x88;
    const scalar_t x120 = x110 * x119 + x116 * x88 - x117 * x88 - x94 + x96;
    const scalar_t x121 = -adjugate[0] * x112 + adjugate[1] * x120 - adjugate[2] * x118;
    const scalar_t x122 = x58 * x90;
    const scalar_t x123 = x107 * x122;
    const scalar_t x124 = mu * x58;
    const scalar_t x125 = x54 * x58;
    const scalar_t x126 = x100 * x124 - x100 * x125 + x100 * x59 + x113 - x114;
    const scalar_t x127 = x102 - x103 + x124 * x89 - x125 * x89 + x59 * x89;
    const scalar_t x128 = -adjugate[0] * x126 + adjugate[1] * x127 - adjugate[2] * x123;
    const scalar_t x129 = x81 * x85;
    const scalar_t x130 = adjugate[3] * x101 - adjugate[4] * x91 + adjugate[5] * x108;
    const scalar_t x131 = -adjugate[3] * x112 + adjugate[4] * x120 - adjugate[5] * x118;
    const scalar_t x132 = -adjugate[3] * x126 + adjugate[4] * x127 - adjugate[5] * x123;
    const scalar_t x133 = adjugate[6] * x101 - adjugate[7] * x91 + adjugate[8] * x108;
    const scalar_t x134 = -adjugate[6] * x112 + adjugate[7] * x120 - adjugate[8] * x118;
    const scalar_t x135 = -adjugate[6] * x126 + adjugate[7] * x127 - adjugate[8] * x123;
    const scalar_t x136 = F[1] * F[5];
    const scalar_t x137 = F[2] * F[4];
    const scalar_t x138 = -x12 * x136 + x12 * x137 - x136 * x14 + x136 * x2 + x136 * x5 + x136 * x7 - x136 * x9 + x137 * x14 -
                          x137 * x2 - x137 * x5 - x137 * x7 + x137 * x9;
    const scalar_t x139 = x111 * x138;
    const scalar_t x140 = F[4] * x93;
    const scalar_t x141 = F[4] * x95;
    const scalar_t x142 = F[0] * F[4];
    const scalar_t x143 = F[1] * F[3];
    const scalar_t x144 = -x12 * x142 + x12 * x143 - x14 * x142 + x14 * x143 + x142 * x2 + x142 * x5 + x142 * x7 - x142 * x9 -
                          x143 * x2 - x143 * x5 - x143 * x7 + x143 * x9;
    const scalar_t x145 = x115 * x144 + x116 * x144 - x117 * x144 - x140 + x141;
    const scalar_t x146 = F[5] * x93;
    const scalar_t x147 = F[5] * x95;
    const scalar_t x148 = F[0] * F[5];
    const scalar_t x149 = F[2] * F[3];
    const scalar_t x150 = -x12 * x148 + x12 * x149 - x14 * x148 + x14 * x149 + x148 * x2 + x148 * x5 + x148 * x7 - x148 * x9 -
                          x149 * x2 - x149 * x5 - x149 * x7 + x149 * x9;
    const scalar_t x151  = x115 * x150 + x116 * x150 - x117 * x150 - x146 + x147;
    const scalar_t x152  = adjugate[0] * x139 - adjugate[1] * x151 + adjugate[2] * x145;
    const scalar_t x153  = x144 * x85;
    const scalar_t x154  = x122 * x153;
    const scalar_t x155  = x138 * x85;
    const scalar_t x156  = x124 * x155 - x125 * x155 + x140 - x141 + x155 * x59;
    const scalar_t x157  = F[3] * x93;
    const scalar_t x158  = F[3] * x95;
    const scalar_t x159  = x150 * x85;
    const scalar_t x160  = x124 * x159 - x125 * x159 + x157 - x158 + x159 * x59;
    const scalar_t x161  = adjugate[0] * x156 - adjugate[1] * x160 + adjugate[2] * x154;
    const scalar_t x162  = x146 - x147 + x155 * x65 + x155 * x73 - x155 * x74;
    const scalar_t x163  = x153 * x65 + x153 * x73 - x153 * x74 - x157 + x158;
    const scalar_t x164  = -adjugate[0] * x162 + adjugate[1] * x150 * x64 * x85 * x90 - adjugate[2] * x163;
    const scalar_t x165  = adjugate[3] * x139 - adjugate[4] * x151 + adjugate[5] * x145;
    const scalar_t x166  = adjugate[3] * x156 - adjugate[4] * x160 + adjugate[5] * x154;
    const scalar_t x167  = -adjugate[3] * x162 + adjugate[4] * x150 * x64 * x85 * x90 - adjugate[5] * x163;
    const scalar_t x168  = adjugate[6] * x139 - adjugate[7] * x151 + adjugate[8] * x145;
    const scalar_t x169  = adjugate[6] * x156 - adjugate[7] * x160 + adjugate[8] * x154;
    const scalar_t x170  = -adjugate[6] * x162 + adjugate[7] * x150 * x64 * x85 * x90 - adjugate[8] * x163;
    const scalar_t x171  = x52 * POW2(x99);
    const scalar_t x172  = lmbda * x171 + mu * x171 + x1 - x171 * x54;
    const scalar_t x173  = x52 * x99;
    const scalar_t x174  = x106 * x173;
    const scalar_t x175  = mu * x173;
    const scalar_t x176  = lmbda * x174 + x106 * x175 - x174 * x54 + x56;
    const scalar_t x177  = -x119 * x173 + x173 * x54 * x88 - x175 * x88 + x63;
    const scalar_t x178  = adjugate[0] * x172 + adjugate[1] * x177 + adjugate[2] * x176;
    const scalar_t x179  = POW2(x106) * x52;
    const scalar_t x180  = lmbda * x179 + mu * x179 - x179 * x54 + x68;
    const scalar_t x181  = x106 * x52;
    const scalar_t x182  = x181 * x88;
    const scalar_t x183  = -mu * x182 - x119 * x181 + x182 * x54 + x71;
    const scalar_t x184  = adjugate[0] * x176 + adjugate[1] * x183 + adjugate[2] * x180;
    const scalar_t x185  = x52 * POW2(x88);
    const scalar_t x186  = lmbda * x185 + mu * x185 - x185 * x54 + x77;
    const scalar_t x187  = adjugate[0] * x177 + adjugate[1] * x186 + adjugate[2] * x183;
    const scalar_t x188  = adjugate[3] * x172 + adjugate[4] * x177 + adjugate[5] * x176;
    const scalar_t x189  = adjugate[3] * x176 + adjugate[4] * x183 + adjugate[5] * x180;
    const scalar_t x190  = adjugate[3] * x177 + adjugate[4] * x186 + adjugate[5] * x183;
    const scalar_t x191  = x150 * x89 * x90;
    const scalar_t x192  = F[2] * x93;
    const scalar_t x193  = F[2] * x95;
    const scalar_t x194  = mu * x89;
    const scalar_t x195  = x54 * x89;
    const scalar_t x196  = x119 * x155 + x138 * x194 - x138 * x195 + x192 - x193;
    const scalar_t x197  = F[0] * x93;
    const scalar_t x198  = F[0] * x95;
    const scalar_t x199  = x119 * x153 + x144 * x194 - x144 * x195 - x197 + x198;
    const scalar_t x200  = adjugate[0] * x196 - adjugate[1] * x191 + adjugate[2] * x199;
    const scalar_t x201  = x100 * x138 * x90;
    const scalar_t x202  = F[1] * x93;
    const scalar_t x203  = F[1] * x95;
    const scalar_t x204  = x100 * x144;
    const scalar_t x205  = lmbda * x204 + mu * x204 - x202 + x203 - x204 * x54;
    const scalar_t x206  = x100 * x150;
    const scalar_t x207  = lmbda * x206 + mu * x206 - x192 + x193 - x206 * x54;
    const scalar_t x208  = -adjugate[0] * x201 + adjugate[1] * x207 - adjugate[2] * x205;
    const scalar_t x209  = x107 * x144 * x90;
    const scalar_t x210  = x107 * x138;
    const scalar_t x211  = lmbda * x210 + mu * x210 + x202 - x203 - x210 * x54;
    const scalar_t x212  = x107 * x150;
    const scalar_t x213  = lmbda * x212 + mu * x212 + x197 - x198 - x212 * x54;
    const scalar_t x214  = -adjugate[0] * x211 + adjugate[1] * x213 - adjugate[2] * x209;
    const scalar_t x215  = adjugate[3] * x196 - adjugate[4] * x191 + adjugate[5] * x199;
    const scalar_t x216  = -adjugate[3] * x201 + adjugate[4] * x207 - adjugate[5] * x205;
    const scalar_t x217  = -adjugate[3] * x211 + adjugate[4] * x213 - adjugate[5] * x209;
    const scalar_t x218  = adjugate[6] * x196 - adjugate[7] * x191 + adjugate[8] * x199;
    const scalar_t x219  = -adjugate[6] * x201 + adjugate[7] * x207 - adjugate[8] * x205;
    const scalar_t x220  = -adjugate[6] * x211 + adjugate[7] * x213 - adjugate[8] * x209;
    const scalar_t x221  = POW2(x138) * x52;
    const scalar_t x222  = lmbda * x221 + mu * x221 + x1 - x221 * x54;
    const scalar_t x223  = x138 * x52;
    const scalar_t x224  = x144 * x223;
    const scalar_t x225  = lmbda * x224 + mu * x224 - x224 * x54 + x56;
    const scalar_t x226  = x150 * x223;
    const scalar_t x227  = -lmbda * x226 - mu * x226 + x226 * x54 + x63;
    const scalar_t x228  = adjugate[0] * x222 + adjugate[1] * x227 + adjugate[2] * x225;
    const scalar_t x229  = POW2(x144) * x52;
    const scalar_t x230  = lmbda * x229 + mu * x229 - x229 * x54 + x68;
    const scalar_t x231  = x144 * x150 * x52;
    const scalar_t x232  = -lmbda * x231 - mu * x231 + x231 * x54 + x71;
    const scalar_t x233  = adjugate[0] * x225 + adjugate[1] * x232 + adjugate[2] * x230;
    const scalar_t x234  = POW2(x150) * x52;
    const scalar_t x235  = lmbda * x234 + mu * x234 - x234 * x54 + x77;
    const scalar_t x236  = adjugate[0] * x227 + adjugate[1] * x235 + adjugate[2] * x232;
    const scalar_t x237  = adjugate[3] * x222 + adjugate[4] * x227 + adjugate[5] * x225;
    const scalar_t x238  = adjugate[3] * x225 + adjugate[4] * x232 + adjugate[5] * x230;
    const scalar_t x239  = adjugate[3] * x227 + adjugate[4] * x235 + adjugate[5] * x232;
    S_ikmn_canonical[0]  = x81 * (adjugate[0] * x67 + adjugate[1] * x80 + adjugate[2] * x76);
    S_ikmn_canonical[1]  = x81 * (adjugate[3] * x67 + adjugate[4] * x80 + adjugate[5] * x76);
    S_ikmn_canonical[2]  = x81 * (adjugate[6] * x67 + adjugate[7] * x80 + adjugate[8] * x76);
    S_ikmn_canonical[3]  = x81 * (adjugate[3] * x82 + adjugate[4] * x84 + adjugate[5] * x83);
    S_ikmn_canonical[4]  = x81 * (adjugate[6] * x82 + adjugate[7] * x84 + adjugate[8] * x83);
    S_ikmn_canonical[5]  = x81 * (adjugate[6] * (adjugate[6] * x55 + adjugate[7] * x66 + adjugate[8] * x62) +
                                 adjugate[7] * (adjugate[6] * x66 + adjugate[7] * x79 + adjugate[8] * x75) +
                                 adjugate[8] * (adjugate[6] * x62 + adjugate[7] * x75 + adjugate[8] * x70));
    S_ikmn_canonical[6]  = x129 * (adjugate[0] * x121 + adjugate[1] * x109 + adjugate[2] * x128);
    S_ikmn_canonical[7]  = x129 * (adjugate[3] * x121 + adjugate[4] * x109 + adjugate[5] * x128);
    S_ikmn_canonical[8]  = x129 * (adjugate[6] * x121 + adjugate[7] * x109 + adjugate[8] * x128);
    S_ikmn_canonical[9]  = x129 * (adjugate[0] * x131 + adjugate[1] * x130 + adjugate[2] * x132);
    S_ikmn_canonical[10] = x129 * (adjugate[3] * x131 + adjugate[4] * x130 + adjugate[5] * x132);
    S_ikmn_canonical[11] = x129 * (adjugate[6] * x131 + adjugate[7] * x130 + adjugate[8] * x132);
    S_ikmn_canonical[12] = x129 * (adjugate[0] * x134 + adjugate[1] * x133 + adjugate[2] * x135);
    S_ikmn_canonical[13] = x129 * (adjugate[3] * x134 + adjugate[4] * x133 + adjugate[5] * x135);
    S_ikmn_canonical[14] = x129 * (adjugate[6] * x134 + adjugate[7] * x133 + adjugate[8] * x135);
    S_ikmn_canonical[15] = x129 * (adjugate[0] * x152 + adjugate[1] * x164 + adjugate[2] * x161);
    S_ikmn_canonical[16] = x129 * (adjugate[3] * x152 + adjugate[4] * x164 + adjugate[5] * x161);
    S_ikmn_canonical[17] = x129 * (adjugate[6] * x152 + adjugate[7] * x164 + adjugate[8] * x161);
    S_ikmn_canonical[18] = x129 * (adjugate[0] * x165 + adjugate[1] * x167 + adjugate[2] * x166);
    S_ikmn_canonical[19] = x129 * (adjugate[3] * x165 + adjugate[4] * x167 + adjugate[5] * x166);
    S_ikmn_canonical[20] = x129 * (adjugate[6] * x165 + adjugate[7] * x167 + adjugate[8] * x166);
    S_ikmn_canonical[21] = x129 * (adjugate[0] * x168 + adjugate[1] * x170 + adjugate[2] * x169);
    S_ikmn_canonical[22] = x129 * (adjugate[3] * x168 + adjugate[4] * x170 + adjugate[5] * x169);
    S_ikmn_canonical[23] = x129 * (adjugate[6] * x168 + adjugate[7] * x170 + adjugate[8] * x169);
    S_ikmn_canonical[24] = x81 * (adjugate[0] * x178 + adjugate[1] * x187 + adjugate[2] * x184);
    S_ikmn_canonical[25] = x81 * (adjugate[3] * x178 + adjugate[4] * x187 + adjugate[5] * x184);
    S_ikmn_canonical[26] = x81 * (adjugate[6] * x178 + adjugate[7] * x187 + adjugate[8] * x184);
    S_ikmn_canonical[27] = x81 * (adjugate[3] * x188 + adjugate[4] * x190 + adjugate[5] * x189);
    S_ikmn_canonical[28] = x81 * (adjugate[6] * x188 + adjugate[7] * x190 + adjugate[8] * x189);
    S_ikmn_canonical[29] = x81 * (adjugate[6] * (adjugate[6] * x172 + adjugate[7] * x177 + adjugate[8] * x176) +
                                  adjugate[7] * (adjugate[6] * x177 + adjugate[7] * x186 + adjugate[8] * x183) +
                                  adjugate[8] * (adjugate[6] * x176 + adjugate[7] * x183 + adjugate[8] * x180));
    S_ikmn_canonical[30] = x129 * (adjugate[0] * x208 + adjugate[1] * x200 + adjugate[2] * x214);
    S_ikmn_canonical[31] = x129 * (adjugate[3] * x208 + adjugate[4] * x200 + adjugate[5] * x214);
    S_ikmn_canonical[32] = x129 * (adjugate[6] * x208 + adjugate[7] * x200 + adjugate[8] * x214);
    S_ikmn_canonical[33] = x129 * (adjugate[0] * x216 + adjugate[1] * x215 + adjugate[2] * x217);
    S_ikmn_canonical[34] = x129 * (adjugate[3] * x216 + adjugate[4] * x215 + adjugate[5] * x217);
    S_ikmn_canonical[35] = x129 * (adjugate[6] * x216 + adjugate[7] * x215 + adjugate[8] * x217);
    S_ikmn_canonical[36] = x129 * (adjugate[0] * x219 + adjugate[1] * x218 + adjugate[2] * x220);
    S_ikmn_canonical[37] = x129 * (adjugate[3] * x219 + adjugate[4] * x218 + adjugate[5] * x220);
    S_ikmn_canonical[38] = x129 * (adjugate[6] * x219 + adjugate[7] * x218 + adjugate[8] * x220);
    S_ikmn_canonical[39] = x81 * (adjugate[0] * x228 + adjugate[1] * x236 + adjugate[2] * x233);
    S_ikmn_canonical[40] = x81 * (adjugate[3] * x228 + adjugate[4] * x236 + adjugate[5] * x233);
    S_ikmn_canonical[41] = x81 * (adjugate[6] * x228 + adjugate[7] * x236 + adjugate[8] * x233);
    S_ikmn_canonical[42] = x81 * (adjugate[3] * x237 + adjugate[4] * x239 + adjugate[5] * x238);
    S_ikmn_canonical[43] = x81 * (adjugate[6] * x237 + adjugate[7] * x239 + adjugate[8] * x238);
    S_ikmn_canonical[44] = x81 * (adjugate[6] * (adjugate[6] * x222 + adjugate[7] * x227 + adjugate[8] * x225) +
                                  adjugate[7] * (adjugate[6] * x227 + adjugate[7] * x235 + adjugate[8] * x232) +
                                  adjugate[8] * (adjugate[6] * x225 + adjugate[7] * x232 + adjugate[8] * x230));
}

static SFEM_INLINE void hex8_S_ikmn_neohookean_ogden_active_strain_add(const scalar_t *const SFEM_RESTRICT adjugate,
                                                                       const scalar_t                      jacobian_determinant,
                                                                       const scalar_t                      qx,
                                                                       const scalar_t                      qy,
                                                                       const scalar_t                      qz,
                                                                       const scalar_t                      qw,
                                                                       const scalar_t *const SFEM_RESTRICT F,
                                                                       const scalar_t                      mu,
                                                                       const scalar_t                      lmbda,
                                                                       const scalar_t *const SFEM_RESTRICT Fa_inv,
                                                                       const scalar_t                      Ja,
                                                                       scalar_t *const SFEM_RESTRICT       S_ikmn_canonical) {
    // mundane ops: 1410 divs: 2 sqrts: 0
    // total ops: 1426
    const scalar_t x0  = POW2(Ja) * mu;
    const scalar_t x1  = x0 * (POW2(Fa_inv[0]) + POW2(Fa_inv[1]) + POW2(Fa_inv[2]));
    const scalar_t x2  = Fa_inv[0] * Fa_inv[4] * Fa_inv[8];
    const scalar_t x3  = F[4] * F[8];
    const scalar_t x4  = x2 * x3;
    const scalar_t x5  = Fa_inv[1] * Fa_inv[5] * Fa_inv[6];
    const scalar_t x6  = x3 * x5;
    const scalar_t x7  = Fa_inv[2] * Fa_inv[3] * Fa_inv[7];
    const scalar_t x8  = x3 * x7;
    const scalar_t x9  = Fa_inv[0] * Fa_inv[5] * Fa_inv[7];
    const scalar_t x10 = F[5] * F[7];
    const scalar_t x11 = x10 * x9;
    const scalar_t x12 = Fa_inv[1] * Fa_inv[3] * Fa_inv[8];
    const scalar_t x13 = x10 * x12;
    const scalar_t x14 = Fa_inv[2] * Fa_inv[4] * Fa_inv[6];
    const scalar_t x15 = x10 * x14;
    const scalar_t x16 = x3 * x9;
    const scalar_t x17 = x12 * x3;
    const scalar_t x18 = x14 * x3;
    const scalar_t x19 = x10 * x2;
    const scalar_t x20 = x10 * x5;
    const scalar_t x21 = x10 * x7;
    const scalar_t x22 = x11 + x13 + x15 - x16 - x17 - x18 - x19 - x20 - x21 + x4 + x6 + x8;
    const scalar_t x23 = F[3] * F[8];
    const scalar_t x24 = x23 * x9;
    const scalar_t x25 = x12 * x23;
    const scalar_t x26 = x14 * x23;
    const scalar_t x27 = F[5] * F[6];
    const scalar_t x28 = x2 * x27;
    const scalar_t x29 = x27 * x5;
    const scalar_t x30 = x27 * x7;
    const scalar_t x31 = F[3] * F[7];
    const scalar_t x32 = x2 * x31;
    const scalar_t x33 = x31 * x5;
    const scalar_t x34 = x31 * x7;
    const scalar_t x35 = F[4] * F[6];
    const scalar_t x36 = x35 * x9;
    const scalar_t x37 = x12 * x35;
    const scalar_t x38 = x14 * x35;
    const scalar_t x39 = x2 * x23;
    const scalar_t x40 = x23 * x5;
    const scalar_t x41 = x23 * x7;
    const scalar_t x42 = x27 * x9;
    const scalar_t x43 = x12 * x27;
    const scalar_t x44 = x14 * x27;
    const scalar_t x45 = x31 * x9;
    const scalar_t x46 = x12 * x31;
    const scalar_t x47 = x14 * x31;
    const scalar_t x48 = x2 * x35;
    const scalar_t x49 = x35 * x5;
    const scalar_t x50 = x35 * x7;
    const scalar_t x51 = F[0] * x11 + F[0] * x13 + F[0] * x15 - F[0] * x16 - F[0] * x17 - F[0] * x18 - F[0] * x19 - F[0] * x20 -
                         F[0] * x21 + F[0] * x4 + F[0] * x6 + F[0] * x8 + F[1] * x24 + F[1] * x25 + F[1] * x26 + F[1] * x28 +
                         F[1] * x29 + F[1] * x30 - F[1] * x39 - F[1] * x40 - F[1] * x41 - F[1] * x42 - F[1] * x43 - F[1] * x44 +
                         F[2] * x32 + F[2] * x33 + F[2] * x34 + F[2] * x36 + F[2] * x37 + F[2] * x38 - F[2] * x45 - F[2] * x46 -
                         F[2] * x47 - F[2] * x48 - F[2] * x49 - F[2] * x50;
    const scalar_t x52 = (1 / POW2(x51));
    const scalar_t x53 = POW2(x22) * x52;
    const scalar_t x54 = lmbda * log((POW3(Ja)) * x51);
    const scalar_t x55 = lmbda * x53 + mu * x53 + x1 - x53 * x54;
    const scalar_t x56 = x0 * (Fa_inv[0] * Fa_inv[6] + Fa_inv[1] * Fa_inv[7] + Fa_inv[2] * Fa_inv[8]);
    const scalar_t x57 = x22 * x52;
    const scalar_t x58 = x32 + x33 + x34 + x36 + x37 + x38 - x45 - x46 - x47 - x48 - x49 - x50;
    const scalar_t x59 = lmbda * x58;
    const scalar_t x60 = mu * x57;
    const scalar_t x61 = x54 * x57;
    const scalar_t x62 = x56 + x57 * x59 + x58 * x60 - x58 * x61;
    const scalar_t x63 = x0 * (Fa_inv[0] * Fa_inv[3] + Fa_inv[1] * Fa_inv[4] + Fa_inv[2] * Fa_inv[5]);
    const scalar_t x64 = -x24 - x25 - x26 - x28 - x29 - x30 + x39 + x40 + x41 + x42 + x43 + x44;
    const scalar_t x65 = lmbda * x64;
    const scalar_t x66 = -x57 * x65 - x60 * x64 + x61 * x64 + x63;
    const scalar_t x67 = adjugate[0] * x55 + adjugate[1] * x66 + adjugate[2] * x62;
    const scalar_t x68 = x0 * (POW2(Fa_inv[6]) + POW2(Fa_inv[7]) + POW2(Fa_inv[8]));
    const scalar_t x69 = x52 * POW2(x58);
    const scalar_t x70 = lmbda * x69 + mu * x69 - x54 * x69 + x68;
    const scalar_t x71 = x0 * (Fa_inv[3] * Fa_inv[6] + Fa_inv[4] * Fa_inv[7] + Fa_inv[5] * Fa_inv[8]);
    const scalar_t x72 = x52 * x58;
    const scalar_t x73 = mu * x64;
    const scalar_t x74 = x54 * x64;
    const scalar_t x75 = -x65 * x72 + x71 - x72 * x73 + x72 * x74;
    const scalar_t x76 = adjugate[0] * x62 + adjugate[1] * x75 + adjugate[2] * x70;
    const scalar_t x77 = x0 * (POW2(Fa_inv[3]) + POW2(Fa_inv[4]) + POW2(Fa_inv[5]));
    const scalar_t x78 = x52 * POW2(x64);
    const scalar_t x79 = lmbda * x78 + mu * x78 - x54 * x78 + x77;
    const scalar_t x80 = adjugate[0] * x66 + adjugate[1] * x79 + adjugate[2] * x75;
    const scalar_t x81 = qw / jacobian_determinant;
    const scalar_t x82 = adjugate[3] * x55 + adjugate[4] * x66 + adjugate[5] * x62;
    const scalar_t x83 = adjugate[3] * x62 + adjugate[4] * x75 + adjugate[5] * x70;
    const scalar_t x84 = adjugate[3] * x66 + adjugate[4] * x79 + adjugate[5] * x75;
    const scalar_t x85 = 1.0 / x51;
    const scalar_t x86 = F[0] * F[8];
    const scalar_t x87 = F[2] * F[6];
    const scalar_t x88 = -x12 * x86 + x12 * x87 - x14 * x86 + x14 * x87 + x2 * x86 - x2 * x87 + x5 * x86 - x5 * x87 + x7 * x86 -
                         x7 * x87 - x86 * x9 + x87 * x9;
    const scalar_t x89 = x85 * x88;
    const scalar_t x90 = lmbda + mu - x54;
    const scalar_t x91 = x64 * x89 * x90;
    const scalar_t x92 = -x12 - x14 + x2 + x5 + x7 - x9;
    const scalar_t x93 = mu * x92;
    const scalar_t x94 = F[8] * x93;
    const scalar_t x95 = x54 * x92;
    const scalar_t x96 = F[8] * x95;
    const scalar_t x97 = F[1] * F[8];
    const scalar_t x98 = F[2] * F[7];
    const scalar_t x99 = -x12 * x97 + x12 * x98 - x14 * x97 + x14 * x98 + x2 * x97 - x2 * x98 + x5 * x97 - x5 * x98 + x7 * x97 -
                         x7 * x98 - x9 * x97 + x9 * x98;
    const scalar_t x100 = x85 * x99;
    const scalar_t x101 = x100 * x65 + x100 * x73 - x100 * x74 + x94 - x96;
    const scalar_t x102 = F[6] * x93;
    const scalar_t x103 = F[6] * x95;
    const scalar_t x104 = F[0] * F[7];
    const scalar_t x105 = F[1] * F[6];
    const scalar_t x106 = -x104 * x12 - x104 * x14 + x104 * x2 + x104 * x5 + x104 * x7 - x104 * x9 + x105 * x12 + x105 * x14 -
                          x105 * x2 - x105 * x5 - x105 * x7 + x105 * x9;
    const scalar_t x107 = x106 * x85;
    const scalar_t x108 = -x102 + x103 + x107 * x65 + x107 * x73 - x107 * x74;
    const scalar_t x109 = adjugate[0] * x101 - adjugate[1] * x91 + adjugate[2] * x108;
    const scalar_t x110 = x22 * x85;
    const scalar_t x111 = x110 * x90;
    const scalar_t x112 = x111 * x99;
    const scalar_t x113 = F[7] * x93;
    const scalar_t x114 = F[7] * x95;
    const scalar_t x115 = lmbda * x110;
    const scalar_t x116 = mu * x110;
    const scalar_t x117 = x110 * x54;
    const scalar_t x118 = x106 * x115 + x106 * x116 - x106 * x117 - x113 + x114;
    const scalar_t x119 = lmbda * x88;
    const scalar_t x120 = x110 * x119 + x116 * x88 - x117 * x88 - x94 + x96;
    const scalar_t x121 = -adjugate[0] * x112 + adjugate[1] * x120 - adjugate[2] * x118;
    const scalar_t x122 = x58 * x90;
    const scalar_t x123 = x107 * x122;
    const scalar_t x124 = mu * x58;
    const scalar_t x125 = x54 * x58;
    const scalar_t x126 = x100 * x124 - x100 * x125 + x100 * x59 + x113 - x114;
    const scalar_t x127 = x102 - x103 + x124 * x89 - x125 * x89 + x59 * x89;
    const scalar_t x128 = -adjugate[0] * x126 + adjugate[1] * x127 - adjugate[2] * x123;
    const scalar_t x129 = x81 * x85;
    const scalar_t x130 = adjugate[3] * x101 - adjugate[4] * x91 + adjugate[5] * x108;
    const scalar_t x131 = -adjugate[3] * x112 + adjugate[4] * x120 - adjugate[5] * x118;
    const scalar_t x132 = -adjugate[3] * x126 + adjugate[4] * x127 - adjugate[5] * x123;
    const scalar_t x133 = adjugate[6] * x101 - adjugate[7] * x91 + adjugate[8] * x108;
    const scalar_t x134 = -adjugate[6] * x112 + adjugate[7] * x120 - adjugate[8] * x118;
    const scalar_t x135 = -adjugate[6] * x126 + adjugate[7] * x127 - adjugate[8] * x123;
    const scalar_t x136 = F[1] * F[5];
    const scalar_t x137 = F[2] * F[4];
    const scalar_t x138 = -x12 * x136 + x12 * x137 - x136 * x14 + x136 * x2 + x136 * x5 + x136 * x7 - x136 * x9 + x137 * x14 -
                          x137 * x2 - x137 * x5 - x137 * x7 + x137 * x9;
    const scalar_t x139 = x111 * x138;
    const scalar_t x140 = F[4] * x93;
    const scalar_t x141 = F[4] * x95;
    const scalar_t x142 = F[0] * F[4];
    const scalar_t x143 = F[1] * F[3];
    const scalar_t x144 = -x12 * x142 + x12 * x143 - x14 * x142 + x14 * x143 + x142 * x2 + x142 * x5 + x142 * x7 - x142 * x9 -
                          x143 * x2 - x143 * x5 - x143 * x7 + x143 * x9;
    const scalar_t x145 = x115 * x144 + x116 * x144 - x117 * x144 - x140 + x141;
    const scalar_t x146 = F[5] * x93;
    const scalar_t x147 = F[5] * x95;
    const scalar_t x148 = F[0] * F[5];
    const scalar_t x149 = F[2] * F[3];
    const scalar_t x150 = -x12 * x148 + x12 * x149 - x14 * x148 + x14 * x149 + x148 * x2 + x148 * x5 + x148 * x7 - x148 * x9 -
                          x149 * x2 - x149 * x5 - x149 * x7 + x149 * x9;
    const scalar_t x151 = x115 * x150 + x116 * x150 - x117 * x150 - x146 + x147;
    const scalar_t x152 = adjugate[0] * x139 - adjugate[1] * x151 + adjugate[2] * x145;
    const scalar_t x153 = x144 * x85;
    const scalar_t x154 = x122 * x153;
    const scalar_t x155 = x138 * x85;
    const scalar_t x156 = x124 * x155 - x125 * x155 + x140 - x141 + x155 * x59;
    const scalar_t x157 = F[3] * x93;
    const scalar_t x158 = F[3] * x95;
    const scalar_t x159 = x150 * x85;
    const scalar_t x160 = x124 * x159 - x125 * x159 + x157 - x158 + x159 * x59;
    const scalar_t x161 = adjugate[0] * x156 - adjugate[1] * x160 + adjugate[2] * x154;
    const scalar_t x162 = x146 - x147 + x155 * x65 + x155 * x73 - x155 * x74;
    const scalar_t x163 = x153 * x65 + x153 * x73 - x153 * x74 - x157 + x158;
    const scalar_t x164 = -adjugate[0] * x162 + adjugate[1] * x150 * x64 * x85 * x90 - adjugate[2] * x163;
    const scalar_t x165 = adjugate[3] * x139 - adjugate[4] * x151 + adjugate[5] * x145;
    const scalar_t x166 = adjugate[3] * x156 - adjugate[4] * x160 + adjugate[5] * x154;
    const scalar_t x167 = -adjugate[3] * x162 + adjugate[4] * x150 * x64 * x85 * x90 - adjugate[5] * x163;
    const scalar_t x168 = adjugate[6] * x139 - adjugate[7] * x151 + adjugate[8] * x145;
    const scalar_t x169 = adjugate[6] * x156 - adjugate[7] * x160 + adjugate[8] * x154;
    const scalar_t x170 = -adjugate[6] * x162 + adjugate[7] * x150 * x64 * x85 * x90 - adjugate[8] * x163;
    const scalar_t x171 = x52 * POW2(x99);
    const scalar_t x172 = lmbda * x171 + mu * x171 + x1 - x171 * x54;
    const scalar_t x173 = x52 * x99;
    const scalar_t x174 = x106 * x173;
    const scalar_t x175 = mu * x173;
    const scalar_t x176 = lmbda * x174 + x106 * x175 - x174 * x54 + x56;
    const scalar_t x177 = -x119 * x173 + x173 * x54 * x88 - x175 * x88 + x63;
    const scalar_t x178 = adjugate[0] * x172 + adjugate[1] * x177 + adjugate[2] * x176;
    const scalar_t x179 = POW2(x106) * x52;
    const scalar_t x180 = lmbda * x179 + mu * x179 - x179 * x54 + x68;
    const scalar_t x181 = x106 * x52;
    const scalar_t x182 = x181 * x88;
    const scalar_t x183 = -mu * x182 - x119 * x181 + x182 * x54 + x71;
    const scalar_t x184 = adjugate[0] * x176 + adjugate[1] * x183 + adjugate[2] * x180;
    const scalar_t x185 = x52 * POW2(x88);
    const scalar_t x186 = lmbda * x185 + mu * x185 - x185 * x54 + x77;
    const scalar_t x187 = adjugate[0] * x177 + adjugate[1] * x186 + adjugate[2] * x183;
    const scalar_t x188 = adjugate[3] * x172 + adjugate[4] * x177 + adjugate[5] * x176;
    const scalar_t x189 = adjugate[3] * x176 + adjugate[4] * x183 + adjugate[5] * x180;
    const scalar_t x190 = adjugate[3] * x177 + adjugate[4] * x186 + adjugate[5] * x183;
    const scalar_t x191 = x150 * x89 * x90;
    const scalar_t x192 = F[2] * x93;
    const scalar_t x193 = F[2] * x95;
    const scalar_t x194 = mu * x89;
    const scalar_t x195 = x54 * x89;
    const scalar_t x196 = x119 * x155 + x138 * x194 - x138 * x195 + x192 - x193;
    const scalar_t x197 = F[0] * x93;
    const scalar_t x198 = F[0] * x95;
    const scalar_t x199 = x119 * x153 + x144 * x194 - x144 * x195 - x197 + x198;
    const scalar_t x200 = adjugate[0] * x196 - adjugate[1] * x191 + adjugate[2] * x199;
    const scalar_t x201 = x100 * x138 * x90;
    const scalar_t x202 = F[1] * x93;
    const scalar_t x203 = F[1] * x95;
    const scalar_t x204 = x100 * x144;
    const scalar_t x205 = lmbda * x204 + mu * x204 - x202 + x203 - x204 * x54;
    const scalar_t x206 = x100 * x150;
    const scalar_t x207 = lmbda * x206 + mu * x206 - x192 + x193 - x206 * x54;
    const scalar_t x208 = -adjugate[0] * x201 + adjugate[1] * x207 - adjugate[2] * x205;
    const scalar_t x209 = x107 * x144 * x90;
    const scalar_t x210 = x107 * x138;
    const scalar_t x211 = lmbda * x210 + mu * x210 + x202 - x203 - x210 * x54;
    const scalar_t x212 = x107 * x150;
    const scalar_t x213 = lmbda * x212 + mu * x212 + x197 - x198 - x212 * x54;
    const scalar_t x214 = -adjugate[0] * x211 + adjugate[1] * x213 - adjugate[2] * x209;
    const scalar_t x215 = adjugate[3] * x196 - adjugate[4] * x191 + adjugate[5] * x199;
    const scalar_t x216 = -adjugate[3] * x201 + adjugate[4] * x207 - adjugate[5] * x205;
    const scalar_t x217 = -adjugate[3] * x211 + adjugate[4] * x213 - adjugate[5] * x209;
    const scalar_t x218 = adjugate[6] * x196 - adjugate[7] * x191 + adjugate[8] * x199;
    const scalar_t x219 = -adjugate[6] * x201 + adjugate[7] * x207 - adjugate[8] * x205;
    const scalar_t x220 = -adjugate[6] * x211 + adjugate[7] * x213 - adjugate[8] * x209;
    const scalar_t x221 = POW2(x138) * x52;
    const scalar_t x222 = lmbda * x221 + mu * x221 + x1 - x221 * x54;
    const scalar_t x223 = x138 * x52;
    const scalar_t x224 = x144 * x223;
    const scalar_t x225 = lmbda * x224 + mu * x224 - x224 * x54 + x56;
    const scalar_t x226 = x150 * x223;
    const scalar_t x227 = -lmbda * x226 - mu * x226 + x226 * x54 + x63;
    const scalar_t x228 = adjugate[0] * x222 + adjugate[1] * x227 + adjugate[2] * x225;
    const scalar_t x229 = POW2(x144) * x52;
    const scalar_t x230 = lmbda * x229 + mu * x229 - x229 * x54 + x68;
    const scalar_t x231 = x144 * x150 * x52;
    const scalar_t x232 = -lmbda * x231 - mu * x231 + x231 * x54 + x71;
    const scalar_t x233 = adjugate[0] * x225 + adjugate[1] * x232 + adjugate[2] * x230;
    const scalar_t x234 = POW2(x150) * x52;
    const scalar_t x235 = lmbda * x234 + mu * x234 - x234 * x54 + x77;
    const scalar_t x236 = adjugate[0] * x227 + adjugate[1] * x235 + adjugate[2] * x232;
    const scalar_t x237 = adjugate[3] * x222 + adjugate[4] * x227 + adjugate[5] * x225;
    const scalar_t x238 = adjugate[3] * x225 + adjugate[4] * x232 + adjugate[5] * x230;
    const scalar_t x239 = adjugate[3] * x227 + adjugate[4] * x235 + adjugate[5] * x232;
    S_ikmn_canonical[0] += x81 * (adjugate[0] * x67 + adjugate[1] * x80 + adjugate[2] * x76);
    S_ikmn_canonical[1] += x81 * (adjugate[3] * x67 + adjugate[4] * x80 + adjugate[5] * x76);
    S_ikmn_canonical[2] += x81 * (adjugate[6] * x67 + adjugate[7] * x80 + adjugate[8] * x76);
    S_ikmn_canonical[3] += x81 * (adjugate[3] * x82 + adjugate[4] * x84 + adjugate[5] * x83);
    S_ikmn_canonical[4] += x81 * (adjugate[6] * x82 + adjugate[7] * x84 + adjugate[8] * x83);
    S_ikmn_canonical[5] += x81 * (adjugate[6] * (adjugate[6] * x55 + adjugate[7] * x66 + adjugate[8] * x62) +
                                  adjugate[7] * (adjugate[6] * x66 + adjugate[7] * x79 + adjugate[8] * x75) +
                                  adjugate[8] * (adjugate[6] * x62 + adjugate[7] * x75 + adjugate[8] * x70));
    S_ikmn_canonical[6] += x129 * (adjugate[0] * x121 + adjugate[1] * x109 + adjugate[2] * x128);
    S_ikmn_canonical[7] += x129 * (adjugate[3] * x121 + adjugate[4] * x109 + adjugate[5] * x128);
    S_ikmn_canonical[8] += x129 * (adjugate[6] * x121 + adjugate[7] * x109 + adjugate[8] * x128);
    S_ikmn_canonical[9] += x129 * (adjugate[0] * x131 + adjugate[1] * x130 + adjugate[2] * x132);
    S_ikmn_canonical[10] += x129 * (adjugate[3] * x131 + adjugate[4] * x130 + adjugate[5] * x132);
    S_ikmn_canonical[11] += x129 * (adjugate[6] * x131 + adjugate[7] * x130 + adjugate[8] * x132);
    S_ikmn_canonical[12] += x129 * (adjugate[0] * x134 + adjugate[1] * x133 + adjugate[2] * x135);
    S_ikmn_canonical[13] += x129 * (adjugate[3] * x134 + adjugate[4] * x133 + adjugate[5] * x135);
    S_ikmn_canonical[14] += x129 * (adjugate[6] * x134 + adjugate[7] * x133 + adjugate[8] * x135);
    S_ikmn_canonical[15] += x129 * (adjugate[0] * x152 + adjugate[1] * x164 + adjugate[2] * x161);
    S_ikmn_canonical[16] += x129 * (adjugate[3] * x152 + adjugate[4] * x164 + adjugate[5] * x161);
    S_ikmn_canonical[17] += x129 * (adjugate[6] * x152 + adjugate[7] * x164 + adjugate[8] * x161);
    S_ikmn_canonical[18] += x129 * (adjugate[0] * x165 + adjugate[1] * x167 + adjugate[2] * x166);
    S_ikmn_canonical[19] += x129 * (adjugate[3] * x165 + adjugate[4] * x167 + adjugate[5] * x166);
    S_ikmn_canonical[20] += x129 * (adjugate[6] * x165 + adjugate[7] * x167 + adjugate[8] * x166);
    S_ikmn_canonical[21] += x129 * (adjugate[0] * x168 + adjugate[1] * x170 + adjugate[2] * x169);
    S_ikmn_canonical[22] += x129 * (adjugate[3] * x168 + adjugate[4] * x170 + adjugate[5] * x169);
    S_ikmn_canonical[23] += x129 * (adjugate[6] * x168 + adjugate[7] * x170 + adjugate[8] * x169);
    S_ikmn_canonical[24] += x81 * (adjugate[0] * x178 + adjugate[1] * x187 + adjugate[2] * x184);
    S_ikmn_canonical[25] += x81 * (adjugate[3] * x178 + adjugate[4] * x187 + adjugate[5] * x184);
    S_ikmn_canonical[26] += x81 * (adjugate[6] * x178 + adjugate[7] * x187 + adjugate[8] * x184);
    S_ikmn_canonical[27] += x81 * (adjugate[3] * x188 + adjugate[4] * x190 + adjugate[5] * x189);
    S_ikmn_canonical[28] += x81 * (adjugate[6] * x188 + adjugate[7] * x190 + adjugate[8] * x189);
    S_ikmn_canonical[29] += x81 * (adjugate[6] * (adjugate[6] * x172 + adjugate[7] * x177 + adjugate[8] * x176) +
                                   adjugate[7] * (adjugate[6] * x177 + adjugate[7] * x186 + adjugate[8] * x183) +
                                   adjugate[8] * (adjugate[6] * x176 + adjugate[7] * x183 + adjugate[8] * x180));
    S_ikmn_canonical[30] += x129 * (adjugate[0] * x208 + adjugate[1] * x200 + adjugate[2] * x214);
    S_ikmn_canonical[31] += x129 * (adjugate[3] * x208 + adjugate[4] * x200 + adjugate[5] * x214);
    S_ikmn_canonical[32] += x129 * (adjugate[6] * x208 + adjugate[7] * x200 + adjugate[8] * x214);
    S_ikmn_canonical[33] += x129 * (adjugate[0] * x216 + adjugate[1] * x215 + adjugate[2] * x217);
    S_ikmn_canonical[34] += x129 * (adjugate[3] * x216 + adjugate[4] * x215 + adjugate[5] * x217);
    S_ikmn_canonical[35] += x129 * (adjugate[6] * x216 + adjugate[7] * x215 + adjugate[8] * x217);
    S_ikmn_canonical[36] += x129 * (adjugate[0] * x219 + adjugate[1] * x218 + adjugate[2] * x220);
    S_ikmn_canonical[37] += x129 * (adjugate[3] * x219 + adjugate[4] * x218 + adjugate[5] * x220);
    S_ikmn_canonical[38] += x129 * (adjugate[6] * x219 + adjugate[7] * x218 + adjugate[8] * x220);
    S_ikmn_canonical[39] += x81 * (adjugate[0] * x228 + adjugate[1] * x236 + adjugate[2] * x233);
    S_ikmn_canonical[40] += x81 * (adjugate[3] * x228 + adjugate[4] * x236 + adjugate[5] * x233);
    S_ikmn_canonical[41] += x81 * (adjugate[6] * x228 + adjugate[7] * x236 + adjugate[8] * x233);
    S_ikmn_canonical[42] += x81 * (adjugate[3] * x237 + adjugate[4] * x239 + adjugate[5] * x238);
    S_ikmn_canonical[43] += x81 * (adjugate[6] * x237 + adjugate[7] * x239 + adjugate[8] * x238);
    S_ikmn_canonical[44] += x81 * (adjugate[6] * (adjugate[6] * x222 + adjugate[7] * x227 + adjugate[8] * x225) +
                                   adjugate[7] * (adjugate[6] * x227 + adjugate[7] * x235 + adjugate[8] * x232) +
                                   adjugate[8] * (adjugate[6] * x225 + adjugate[7] * x232 + adjugate[8] * x230));
}

#endif /* SFEM_HEX8_PARTIAL_ASSEMBLY_NEOHOOKEAN_OGDEN_ACTIVE_STRAIN_INLINE_H */
