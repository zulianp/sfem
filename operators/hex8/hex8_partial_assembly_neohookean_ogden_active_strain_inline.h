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
                                                                   const scalar_t                      lmbda,
                                                                   const scalar_t                      mu,
                                                                   const scalar_t *const SFEM_RESTRICT Fa_inv,
                                                                   const scalar_t                      Ja,
                                                                   scalar_t *const SFEM_RESTRICT       S_ikmn_canonical) {
    // mundane ops: 1407 divs: 2 sqrts: 0
    // total ops: 1423
    const scalar_t x0  = mu * (POW2(Fa_inv[0]) + POW2(Fa_inv[1]) + POW2(Fa_inv[2]));
    const scalar_t x1  = Fa_inv[0] * Fa_inv[4] * Fa_inv[8];
    const scalar_t x2  = F[4] * F[8];
    const scalar_t x3  = x1 * x2;
    const scalar_t x4  = Fa_inv[1] * Fa_inv[5] * Fa_inv[6];
    const scalar_t x5  = x2 * x4;
    const scalar_t x6  = Fa_inv[2] * Fa_inv[3] * Fa_inv[7];
    const scalar_t x7  = x2 * x6;
    const scalar_t x8  = Fa_inv[0] * Fa_inv[5] * Fa_inv[7];
    const scalar_t x9  = F[5] * F[7];
    const scalar_t x10 = x8 * x9;
    const scalar_t x11 = Fa_inv[1] * Fa_inv[3] * Fa_inv[8];
    const scalar_t x12 = x11 * x9;
    const scalar_t x13 = Fa_inv[2] * Fa_inv[4] * Fa_inv[6];
    const scalar_t x14 = x13 * x9;
    const scalar_t x15 = x2 * x8;
    const scalar_t x16 = x11 * x2;
    const scalar_t x17 = x13 * x2;
    const scalar_t x18 = x1 * x9;
    const scalar_t x19 = x4 * x9;
    const scalar_t x20 = x6 * x9;
    const scalar_t x21 = x10 + x12 + x14 - x15 - x16 - x17 - x18 - x19 - x20 + x3 + x5 + x7;
    const scalar_t x22 = F[3] * F[8];
    const scalar_t x23 = x22 * x8;
    const scalar_t x24 = x11 * x22;
    const scalar_t x25 = x13 * x22;
    const scalar_t x26 = F[5] * F[6];
    const scalar_t x27 = x1 * x26;
    const scalar_t x28 = x26 * x4;
    const scalar_t x29 = x26 * x6;
    const scalar_t x30 = F[3] * F[7];
    const scalar_t x31 = x1 * x30;
    const scalar_t x32 = x30 * x4;
    const scalar_t x33 = x30 * x6;
    const scalar_t x34 = F[4] * F[6];
    const scalar_t x35 = x34 * x8;
    const scalar_t x36 = x11 * x34;
    const scalar_t x37 = x13 * x34;
    const scalar_t x38 = x1 * x22;
    const scalar_t x39 = x22 * x4;
    const scalar_t x40 = x22 * x6;
    const scalar_t x41 = x26 * x8;
    const scalar_t x42 = x11 * x26;
    const scalar_t x43 = x13 * x26;
    const scalar_t x44 = x30 * x8;
    const scalar_t x45 = x11 * x30;
    const scalar_t x46 = x13 * x30;
    const scalar_t x47 = x1 * x34;
    const scalar_t x48 = x34 * x4;
    const scalar_t x49 = x34 * x6;
    const scalar_t x50 = F[0] * x10 + F[0] * x12 + F[0] * x14 - F[0] * x15 - F[0] * x16 - F[0] * x17 - F[0] * x18 - F[0] * x19 -
                         F[0] * x20 + F[0] * x3 + F[0] * x5 + F[0] * x7 + F[1] * x23 + F[1] * x24 + F[1] * x25 + F[1] * x27 +
                         F[1] * x28 + F[1] * x29 - F[1] * x38 - F[1] * x39 - F[1] * x40 - F[1] * x41 - F[1] * x42 - F[1] * x43 +
                         F[2] * x31 + F[2] * x32 + F[2] * x33 + F[2] * x35 + F[2] * x36 + F[2] * x37 - F[2] * x44 - F[2] * x45 -
                         F[2] * x46 - F[2] * x47 - F[2] * x48 - F[2] * x49;
    const scalar_t x51 = (1 / POW2(x50));
    const scalar_t x52 = POW2(x21) * x51;
    const scalar_t x53 = lmbda * log(x50);
    const scalar_t x54 = lmbda * x52 + mu * x52 + x0 - x52 * x53;
    const scalar_t x55 = mu * (Fa_inv[0] * Fa_inv[6] + Fa_inv[1] * Fa_inv[7] + Fa_inv[2] * Fa_inv[8]);
    const scalar_t x56 = x21 * x51;
    const scalar_t x57 = x31 + x32 + x33 + x35 + x36 + x37 - x44 - x45 - x46 - x47 - x48 - x49;
    const scalar_t x58 = lmbda * x57;
    const scalar_t x59 = mu * x56;
    const scalar_t x60 = x53 * x56;
    const scalar_t x61 = x55 + x56 * x58 + x57 * x59 - x57 * x60;
    const scalar_t x62 = mu * (Fa_inv[0] * Fa_inv[3] + Fa_inv[1] * Fa_inv[4] + Fa_inv[2] * Fa_inv[5]);
    const scalar_t x63 = -x23 - x24 - x25 - x27 - x28 - x29 + x38 + x39 + x40 + x41 + x42 + x43;
    const scalar_t x64 = lmbda * x63;
    const scalar_t x65 = -x56 * x64 - x59 * x63 + x60 * x63 + x62;
    const scalar_t x66 = adjugate[0] * x54 + adjugate[1] * x65 + adjugate[2] * x61;
    const scalar_t x67 = mu * (POW2(Fa_inv[6]) + POW2(Fa_inv[7]) + POW2(Fa_inv[8]));
    const scalar_t x68 = x51 * POW2(x57);
    const scalar_t x69 = lmbda * x68 + mu * x68 - x53 * x68 + x67;
    const scalar_t x70 = mu * (Fa_inv[3] * Fa_inv[6] + Fa_inv[4] * Fa_inv[7] + Fa_inv[5] * Fa_inv[8]);
    const scalar_t x71 = x51 * x57;
    const scalar_t x72 = mu * x63;
    const scalar_t x73 = x53 * x63;
    const scalar_t x74 = -x64 * x71 + x70 - x71 * x72 + x71 * x73;
    const scalar_t x75 = adjugate[0] * x61 + adjugate[1] * x74 + adjugate[2] * x69;
    const scalar_t x76 = mu * (POW2(Fa_inv[3]) + POW2(Fa_inv[4]) + POW2(Fa_inv[5]));
    const scalar_t x77 = x51 * POW2(x63);
    const scalar_t x78 = lmbda * x77 + mu * x77 - x53 * x77 + x76;
    const scalar_t x79 = adjugate[0] * x65 + adjugate[1] * x78 + adjugate[2] * x74;
    const scalar_t x80 = Ja * qw / jacobian_determinant;
    const scalar_t x81 = adjugate[3] * x54 + adjugate[4] * x65 + adjugate[5] * x61;
    const scalar_t x82 = adjugate[3] * x61 + adjugate[4] * x74 + adjugate[5] * x69;
    const scalar_t x83 = adjugate[3] * x65 + adjugate[4] * x78 + adjugate[5] * x74;
    const scalar_t x84 = 1.0 / x50;
    const scalar_t x85 = F[0] * F[8];
    const scalar_t x86 = F[2] * F[6];
    const scalar_t x87 = x1 * x85 - x1 * x86 - x11 * x85 + x11 * x86 - x13 * x85 + x13 * x86 + x4 * x85 - x4 * x86 + x6 * x85 -
                         x6 * x86 - x8 * x85 + x8 * x86;
    const scalar_t x88 = x84 * x87;
    const scalar_t x89 = lmbda + mu - x53;
    const scalar_t x90 = x63 * x88 * x89;
    const scalar_t x91 = x1 - x11 - x13 + x4 + x6 - x8;
    const scalar_t x92 = mu * x91;
    const scalar_t x93 = F[8] * x92;
    const scalar_t x94 = x53 * x91;
    const scalar_t x95 = F[8] * x94;
    const scalar_t x96 = F[1] * F[8];
    const scalar_t x97 = F[2] * F[7];
    const scalar_t x98 = x1 * x96 - x1 * x97 - x11 * x96 + x11 * x97 - x13 * x96 + x13 * x97 + x4 * x96 - x4 * x97 + x6 * x96 -
                         x6 * x97 - x8 * x96 + x8 * x97;
    const scalar_t x99  = x84 * x98;
    const scalar_t x100 = x64 * x99 + x72 * x99 - x73 * x99 + x93 - x95;
    const scalar_t x101 = F[6] * x92;
    const scalar_t x102 = F[6] * x94;
    const scalar_t x103 = F[0] * F[7];
    const scalar_t x104 = F[1] * F[6];
    const scalar_t x105 = x1 * x103 - x1 * x104 - x103 * x11 - x103 * x13 + x103 * x4 + x103 * x6 - x103 * x8 + x104 * x11 +
                          x104 * x13 - x104 * x4 - x104 * x6 + x104 * x8;
    const scalar_t x106 = x105 * x84;
    const scalar_t x107 = -x101 + x102 + x106 * x64 + x106 * x72 - x106 * x73;
    const scalar_t x108 = adjugate[0] * x100 - adjugate[1] * x90 + adjugate[2] * x107;
    const scalar_t x109 = x21 * x84;
    const scalar_t x110 = x109 * x89;
    const scalar_t x111 = x110 * x98;
    const scalar_t x112 = F[7] * x92;
    const scalar_t x113 = F[7] * x94;
    const scalar_t x114 = lmbda * x109;
    const scalar_t x115 = mu * x109;
    const scalar_t x116 = x109 * x53;
    const scalar_t x117 = x105 * x114 + x105 * x115 - x105 * x116 - x112 + x113;
    const scalar_t x118 = lmbda * x87;
    const scalar_t x119 = x109 * x118 + x115 * x87 - x116 * x87 - x93 + x95;
    const scalar_t x120 = -adjugate[0] * x111 + adjugate[1] * x119 - adjugate[2] * x117;
    const scalar_t x121 = x57 * x89;
    const scalar_t x122 = x106 * x121;
    const scalar_t x123 = mu * x57;
    const scalar_t x124 = x53 * x57;
    const scalar_t x125 = x112 - x113 + x123 * x99 - x124 * x99 + x58 * x99;
    const scalar_t x126 = x101 - x102 + x123 * x88 - x124 * x88 + x58 * x88;
    const scalar_t x127 = -adjugate[0] * x125 + adjugate[1] * x126 - adjugate[2] * x122;
    const scalar_t x128 = x80 * x84;
    const scalar_t x129 = adjugate[3] * x100 - adjugate[4] * x90 + adjugate[5] * x107;
    const scalar_t x130 = -adjugate[3] * x111 + adjugate[4] * x119 - adjugate[5] * x117;
    const scalar_t x131 = -adjugate[3] * x125 + adjugate[4] * x126 - adjugate[5] * x122;
    const scalar_t x132 = adjugate[6] * x100 - adjugate[7] * x90 + adjugate[8] * x107;
    const scalar_t x133 = -adjugate[6] * x111 + adjugate[7] * x119 - adjugate[8] * x117;
    const scalar_t x134 = -adjugate[6] * x125 + adjugate[7] * x126 - adjugate[8] * x122;
    const scalar_t x135 = F[1] * F[5];
    const scalar_t x136 = F[2] * F[4];
    const scalar_t x137 = x1 * x135 - x1 * x136 - x11 * x135 + x11 * x136 - x13 * x135 + x13 * x136 + x135 * x4 + x135 * x6 -
                          x135 * x8 - x136 * x4 - x136 * x6 + x136 * x8;
    const scalar_t x138 = x110 * x137;
    const scalar_t x139 = F[4] * x92;
    const scalar_t x140 = F[4] * x94;
    const scalar_t x141 = F[0] * F[4];
    const scalar_t x142 = F[1] * F[3];
    const scalar_t x143 = x1 * x141 - x1 * x142 - x11 * x141 + x11 * x142 - x13 * x141 + x13 * x142 + x141 * x4 + x141 * x6 -
                          x141 * x8 - x142 * x4 - x142 * x6 + x142 * x8;
    const scalar_t x144 = x114 * x143 + x115 * x143 - x116 * x143 - x139 + x140;
    const scalar_t x145 = F[5] * x92;
    const scalar_t x146 = F[5] * x94;
    const scalar_t x147 = F[0] * F[5];
    const scalar_t x148 = F[2] * F[3];
    const scalar_t x149 = x1 * x147 - x1 * x148 - x11 * x147 + x11 * x148 - x13 * x147 + x13 * x148 + x147 * x4 + x147 * x6 -
                          x147 * x8 - x148 * x4 - x148 * x6 + x148 * x8;
    const scalar_t x150  = x114 * x149 + x115 * x149 - x116 * x149 - x145 + x146;
    const scalar_t x151  = adjugate[0] * x138 - adjugate[1] * x150 + adjugate[2] * x144;
    const scalar_t x152  = x143 * x84;
    const scalar_t x153  = x121 * x152;
    const scalar_t x154  = x137 * x84;
    const scalar_t x155  = x123 * x154 - x124 * x154 + x139 - x140 + x154 * x58;
    const scalar_t x156  = F[3] * x92;
    const scalar_t x157  = F[3] * x94;
    const scalar_t x158  = x149 * x84;
    const scalar_t x159  = x123 * x158 - x124 * x158 + x156 - x157 + x158 * x58;
    const scalar_t x160  = adjugate[0] * x155 - adjugate[1] * x159 + adjugate[2] * x153;
    const scalar_t x161  = x145 - x146 + x154 * x64 + x154 * x72 - x154 * x73;
    const scalar_t x162  = x152 * x64 + x152 * x72 - x152 * x73 - x156 + x157;
    const scalar_t x163  = -adjugate[0] * x161 + adjugate[1] * x149 * x63 * x84 * x89 - adjugate[2] * x162;
    const scalar_t x164  = adjugate[3] * x138 - adjugate[4] * x150 + adjugate[5] * x144;
    const scalar_t x165  = adjugate[3] * x155 - adjugate[4] * x159 + adjugate[5] * x153;
    const scalar_t x166  = -adjugate[3] * x161 + adjugate[4] * x149 * x63 * x84 * x89 - adjugate[5] * x162;
    const scalar_t x167  = adjugate[6] * x138 - adjugate[7] * x150 + adjugate[8] * x144;
    const scalar_t x168  = adjugate[6] * x155 - adjugate[7] * x159 + adjugate[8] * x153;
    const scalar_t x169  = -adjugate[6] * x161 + adjugate[7] * x149 * x63 * x84 * x89 - adjugate[8] * x162;
    const scalar_t x170  = x51 * POW2(x98);
    const scalar_t x171  = lmbda * x170 + mu * x170 + x0 - x170 * x53;
    const scalar_t x172  = x51 * x98;
    const scalar_t x173  = x105 * x172;
    const scalar_t x174  = mu * x172;
    const scalar_t x175  = lmbda * x173 + x105 * x174 - x173 * x53 + x55;
    const scalar_t x176  = -x118 * x172 + x172 * x53 * x87 - x174 * x87 + x62;
    const scalar_t x177  = adjugate[0] * x171 + adjugate[1] * x176 + adjugate[2] * x175;
    const scalar_t x178  = POW2(x105) * x51;
    const scalar_t x179  = lmbda * x178 + mu * x178 - x178 * x53 + x67;
    const scalar_t x180  = x105 * x51;
    const scalar_t x181  = x180 * x87;
    const scalar_t x182  = -mu * x181 - x118 * x180 + x181 * x53 + x70;
    const scalar_t x183  = adjugate[0] * x175 + adjugate[1] * x182 + adjugate[2] * x179;
    const scalar_t x184  = x51 * POW2(x87);
    const scalar_t x185  = lmbda * x184 + mu * x184 - x184 * x53 + x76;
    const scalar_t x186  = adjugate[0] * x176 + adjugate[1] * x185 + adjugate[2] * x182;
    const scalar_t x187  = adjugate[3] * x171 + adjugate[4] * x176 + adjugate[5] * x175;
    const scalar_t x188  = adjugate[3] * x175 + adjugate[4] * x182 + adjugate[5] * x179;
    const scalar_t x189  = adjugate[3] * x176 + adjugate[4] * x185 + adjugate[5] * x182;
    const scalar_t x190  = x149 * x88 * x89;
    const scalar_t x191  = F[2] * x92;
    const scalar_t x192  = F[2] * x94;
    const scalar_t x193  = mu * x88;
    const scalar_t x194  = x53 * x88;
    const scalar_t x195  = x118 * x154 + x137 * x193 - x137 * x194 + x191 - x192;
    const scalar_t x196  = F[0] * x92;
    const scalar_t x197  = F[0] * x94;
    const scalar_t x198  = x118 * x152 + x143 * x193 - x143 * x194 - x196 + x197;
    const scalar_t x199  = adjugate[0] * x195 - adjugate[1] * x190 + adjugate[2] * x198;
    const scalar_t x200  = x137 * x89 * x99;
    const scalar_t x201  = F[1] * x92;
    const scalar_t x202  = F[1] * x94;
    const scalar_t x203  = x143 * x99;
    const scalar_t x204  = lmbda * x203 + mu * x203 - x201 + x202 - x203 * x53;
    const scalar_t x205  = x149 * x99;
    const scalar_t x206  = lmbda * x205 + mu * x205 - x191 + x192 - x205 * x53;
    const scalar_t x207  = -adjugate[0] * x200 + adjugate[1] * x206 - adjugate[2] * x204;
    const scalar_t x208  = x106 * x143 * x89;
    const scalar_t x209  = x106 * x137;
    const scalar_t x210  = lmbda * x209 + mu * x209 + x201 - x202 - x209 * x53;
    const scalar_t x211  = x106 * x149;
    const scalar_t x212  = lmbda * x211 + mu * x211 + x196 - x197 - x211 * x53;
    const scalar_t x213  = -adjugate[0] * x210 + adjugate[1] * x212 - adjugate[2] * x208;
    const scalar_t x214  = adjugate[3] * x195 - adjugate[4] * x190 + adjugate[5] * x198;
    const scalar_t x215  = -adjugate[3] * x200 + adjugate[4] * x206 - adjugate[5] * x204;
    const scalar_t x216  = -adjugate[3] * x210 + adjugate[4] * x212 - adjugate[5] * x208;
    const scalar_t x217  = adjugate[6] * x195 - adjugate[7] * x190 + adjugate[8] * x198;
    const scalar_t x218  = -adjugate[6] * x200 + adjugate[7] * x206 - adjugate[8] * x204;
    const scalar_t x219  = -adjugate[6] * x210 + adjugate[7] * x212 - adjugate[8] * x208;
    const scalar_t x220  = POW2(x137) * x51;
    const scalar_t x221  = lmbda * x220 + mu * x220 + x0 - x220 * x53;
    const scalar_t x222  = x137 * x51;
    const scalar_t x223  = x143 * x222;
    const scalar_t x224  = lmbda * x223 + mu * x223 - x223 * x53 + x55;
    const scalar_t x225  = x149 * x222;
    const scalar_t x226  = -lmbda * x225 - mu * x225 + x225 * x53 + x62;
    const scalar_t x227  = adjugate[0] * x221 + adjugate[1] * x226 + adjugate[2] * x224;
    const scalar_t x228  = POW2(x143) * x51;
    const scalar_t x229  = lmbda * x228 + mu * x228 - x228 * x53 + x67;
    const scalar_t x230  = x143 * x149 * x51;
    const scalar_t x231  = -lmbda * x230 - mu * x230 + x230 * x53 + x70;
    const scalar_t x232  = adjugate[0] * x224 + adjugate[1] * x231 + adjugate[2] * x229;
    const scalar_t x233  = POW2(x149) * x51;
    const scalar_t x234  = lmbda * x233 + mu * x233 - x233 * x53 + x76;
    const scalar_t x235  = adjugate[0] * x226 + adjugate[1] * x234 + adjugate[2] * x231;
    const scalar_t x236  = adjugate[3] * x221 + adjugate[4] * x226 + adjugate[5] * x224;
    const scalar_t x237  = adjugate[3] * x224 + adjugate[4] * x231 + adjugate[5] * x229;
    const scalar_t x238  = adjugate[3] * x226 + adjugate[4] * x234 + adjugate[5] * x231;
    S_ikmn_canonical[0]  = x80 * (adjugate[0] * x66 + adjugate[1] * x79 + adjugate[2] * x75);
    S_ikmn_canonical[1]  = x80 * (adjugate[3] * x66 + adjugate[4] * x79 + adjugate[5] * x75);
    S_ikmn_canonical[2]  = x80 * (adjugate[6] * x66 + adjugate[7] * x79 + adjugate[8] * x75);
    S_ikmn_canonical[3]  = x80 * (adjugate[3] * x81 + adjugate[4] * x83 + adjugate[5] * x82);
    S_ikmn_canonical[4]  = x80 * (adjugate[6] * x81 + adjugate[7] * x83 + adjugate[8] * x82);
    S_ikmn_canonical[5]  = x80 * (adjugate[6] * (adjugate[6] * x54 + adjugate[7] * x65 + adjugate[8] * x61) +
                                 adjugate[7] * (adjugate[6] * x65 + adjugate[7] * x78 + adjugate[8] * x74) +
                                 adjugate[8] * (adjugate[6] * x61 + adjugate[7] * x74 + adjugate[8] * x69));
    S_ikmn_canonical[6]  = x128 * (adjugate[0] * x120 + adjugate[1] * x108 + adjugate[2] * x127);
    S_ikmn_canonical[7]  = x128 * (adjugate[3] * x120 + adjugate[4] * x108 + adjugate[5] * x127);
    S_ikmn_canonical[8]  = x128 * (adjugate[6] * x120 + adjugate[7] * x108 + adjugate[8] * x127);
    S_ikmn_canonical[9]  = x128 * (adjugate[0] * x130 + adjugate[1] * x129 + adjugate[2] * x131);
    S_ikmn_canonical[10] = x128 * (adjugate[3] * x130 + adjugate[4] * x129 + adjugate[5] * x131);
    S_ikmn_canonical[11] = x128 * (adjugate[6] * x130 + adjugate[7] * x129 + adjugate[8] * x131);
    S_ikmn_canonical[12] = x128 * (adjugate[0] * x133 + adjugate[1] * x132 + adjugate[2] * x134);
    S_ikmn_canonical[13] = x128 * (adjugate[3] * x133 + adjugate[4] * x132 + adjugate[5] * x134);
    S_ikmn_canonical[14] = x128 * (adjugate[6] * x133 + adjugate[7] * x132 + adjugate[8] * x134);
    S_ikmn_canonical[15] = x128 * (adjugate[0] * x151 + adjugate[1] * x163 + adjugate[2] * x160);
    S_ikmn_canonical[16] = x128 * (adjugate[3] * x151 + adjugate[4] * x163 + adjugate[5] * x160);
    S_ikmn_canonical[17] = x128 * (adjugate[6] * x151 + adjugate[7] * x163 + adjugate[8] * x160);
    S_ikmn_canonical[18] = x128 * (adjugate[0] * x164 + adjugate[1] * x166 + adjugate[2] * x165);
    S_ikmn_canonical[19] = x128 * (adjugate[3] * x164 + adjugate[4] * x166 + adjugate[5] * x165);
    S_ikmn_canonical[20] = x128 * (adjugate[6] * x164 + adjugate[7] * x166 + adjugate[8] * x165);
    S_ikmn_canonical[21] = x128 * (adjugate[0] * x167 + adjugate[1] * x169 + adjugate[2] * x168);
    S_ikmn_canonical[22] = x128 * (adjugate[3] * x167 + adjugate[4] * x169 + adjugate[5] * x168);
    S_ikmn_canonical[23] = x128 * (adjugate[6] * x167 + adjugate[7] * x169 + adjugate[8] * x168);
    S_ikmn_canonical[24] = x80 * (adjugate[0] * x177 + adjugate[1] * x186 + adjugate[2] * x183);
    S_ikmn_canonical[25] = x80 * (adjugate[3] * x177 + adjugate[4] * x186 + adjugate[5] * x183);
    S_ikmn_canonical[26] = x80 * (adjugate[6] * x177 + adjugate[7] * x186 + adjugate[8] * x183);
    S_ikmn_canonical[27] = x80 * (adjugate[3] * x187 + adjugate[4] * x189 + adjugate[5] * x188);
    S_ikmn_canonical[28] = x80 * (adjugate[6] * x187 + adjugate[7] * x189 + adjugate[8] * x188);
    S_ikmn_canonical[29] = x80 * (adjugate[6] * (adjugate[6] * x171 + adjugate[7] * x176 + adjugate[8] * x175) +
                                  adjugate[7] * (adjugate[6] * x176 + adjugate[7] * x185 + adjugate[8] * x182) +
                                  adjugate[8] * (adjugate[6] * x175 + adjugate[7] * x182 + adjugate[8] * x179));
    S_ikmn_canonical[30] = x128 * (adjugate[0] * x207 + adjugate[1] * x199 + adjugate[2] * x213);
    S_ikmn_canonical[31] = x128 * (adjugate[3] * x207 + adjugate[4] * x199 + adjugate[5] * x213);
    S_ikmn_canonical[32] = x128 * (adjugate[6] * x207 + adjugate[7] * x199 + adjugate[8] * x213);
    S_ikmn_canonical[33] = x128 * (adjugate[0] * x215 + adjugate[1] * x214 + adjugate[2] * x216);
    S_ikmn_canonical[34] = x128 * (adjugate[3] * x215 + adjugate[4] * x214 + adjugate[5] * x216);
    S_ikmn_canonical[35] = x128 * (adjugate[6] * x215 + adjugate[7] * x214 + adjugate[8] * x216);
    S_ikmn_canonical[36] = x128 * (adjugate[0] * x218 + adjugate[1] * x217 + adjugate[2] * x219);
    S_ikmn_canonical[37] = x128 * (adjugate[3] * x218 + adjugate[4] * x217 + adjugate[5] * x219);
    S_ikmn_canonical[38] = x128 * (adjugate[6] * x218 + adjugate[7] * x217 + adjugate[8] * x219);
    S_ikmn_canonical[39] = x80 * (adjugate[0] * x227 + adjugate[1] * x235 + adjugate[2] * x232);
    S_ikmn_canonical[40] = x80 * (adjugate[3] * x227 + adjugate[4] * x235 + adjugate[5] * x232);
    S_ikmn_canonical[41] = x80 * (adjugate[6] * x227 + adjugate[7] * x235 + adjugate[8] * x232);
    S_ikmn_canonical[42] = x80 * (adjugate[3] * x236 + adjugate[4] * x238 + adjugate[5] * x237);
    S_ikmn_canonical[43] = x80 * (adjugate[6] * x236 + adjugate[7] * x238 + adjugate[8] * x237);
    S_ikmn_canonical[44] = x80 * (adjugate[6] * (adjugate[6] * x221 + adjugate[7] * x226 + adjugate[8] * x224) +
                                  adjugate[7] * (adjugate[6] * x226 + adjugate[7] * x234 + adjugate[8] * x231) +
                                  adjugate[8] * (adjugate[6] * x224 + adjugate[7] * x231 + adjugate[8] * x229));
}

static SFEM_INLINE void hex8_S_ikmn_neohookean_ogden_active_strain_add(const scalar_t *const SFEM_RESTRICT adjugate,
                                                                       const scalar_t                      jacobian_determinant,
                                                                       const scalar_t                      qx,
                                                                       const scalar_t                      qy,
                                                                       const scalar_t                      qz,
                                                                       const scalar_t                      qw,
                                                                       const scalar_t *const SFEM_RESTRICT F,
                                                                       const scalar_t                      lmbda,
                                                                       const scalar_t                      mu,
                                                                       const scalar_t *const SFEM_RESTRICT Fa_inv,
                                                                       const scalar_t                      Ja,
                                                                       scalar_t *const SFEM_RESTRICT       S_ikmn_canonical) {
    // mundane ops: 1407 divs: 2 sqrts: 0
    // total ops: 1423
    const scalar_t x0  = mu * (POW2(Fa_inv[0]) + POW2(Fa_inv[1]) + POW2(Fa_inv[2]));
    const scalar_t x1  = Fa_inv[0] * Fa_inv[4] * Fa_inv[8];
    const scalar_t x2  = F[4] * F[8];
    const scalar_t x3  = x1 * x2;
    const scalar_t x4  = Fa_inv[1] * Fa_inv[5] * Fa_inv[6];
    const scalar_t x5  = x2 * x4;
    const scalar_t x6  = Fa_inv[2] * Fa_inv[3] * Fa_inv[7];
    const scalar_t x7  = x2 * x6;
    const scalar_t x8  = Fa_inv[0] * Fa_inv[5] * Fa_inv[7];
    const scalar_t x9  = F[5] * F[7];
    const scalar_t x10 = x8 * x9;
    const scalar_t x11 = Fa_inv[1] * Fa_inv[3] * Fa_inv[8];
    const scalar_t x12 = x11 * x9;
    const scalar_t x13 = Fa_inv[2] * Fa_inv[4] * Fa_inv[6];
    const scalar_t x14 = x13 * x9;
    const scalar_t x15 = x2 * x8;
    const scalar_t x16 = x11 * x2;
    const scalar_t x17 = x13 * x2;
    const scalar_t x18 = x1 * x9;
    const scalar_t x19 = x4 * x9;
    const scalar_t x20 = x6 * x9;
    const scalar_t x21 = x10 + x12 + x14 - x15 - x16 - x17 - x18 - x19 - x20 + x3 + x5 + x7;
    const scalar_t x22 = F[3] * F[8];
    const scalar_t x23 = x22 * x8;
    const scalar_t x24 = x11 * x22;
    const scalar_t x25 = x13 * x22;
    const scalar_t x26 = F[5] * F[6];
    const scalar_t x27 = x1 * x26;
    const scalar_t x28 = x26 * x4;
    const scalar_t x29 = x26 * x6;
    const scalar_t x30 = F[3] * F[7];
    const scalar_t x31 = x1 * x30;
    const scalar_t x32 = x30 * x4;
    const scalar_t x33 = x30 * x6;
    const scalar_t x34 = F[4] * F[6];
    const scalar_t x35 = x34 * x8;
    const scalar_t x36 = x11 * x34;
    const scalar_t x37 = x13 * x34;
    const scalar_t x38 = x1 * x22;
    const scalar_t x39 = x22 * x4;
    const scalar_t x40 = x22 * x6;
    const scalar_t x41 = x26 * x8;
    const scalar_t x42 = x11 * x26;
    const scalar_t x43 = x13 * x26;
    const scalar_t x44 = x30 * x8;
    const scalar_t x45 = x11 * x30;
    const scalar_t x46 = x13 * x30;
    const scalar_t x47 = x1 * x34;
    const scalar_t x48 = x34 * x4;
    const scalar_t x49 = x34 * x6;
    const scalar_t x50 = F[0] * x10 + F[0] * x12 + F[0] * x14 - F[0] * x15 - F[0] * x16 - F[0] * x17 - F[0] * x18 - F[0] * x19 -
                         F[0] * x20 + F[0] * x3 + F[0] * x5 + F[0] * x7 + F[1] * x23 + F[1] * x24 + F[1] * x25 + F[1] * x27 +
                         F[1] * x28 + F[1] * x29 - F[1] * x38 - F[1] * x39 - F[1] * x40 - F[1] * x41 - F[1] * x42 - F[1] * x43 +
                         F[2] * x31 + F[2] * x32 + F[2] * x33 + F[2] * x35 + F[2] * x36 + F[2] * x37 - F[2] * x44 - F[2] * x45 -
                         F[2] * x46 - F[2] * x47 - F[2] * x48 - F[2] * x49;
    const scalar_t x51 = (1 / POW2(x50));
    const scalar_t x52 = POW2(x21) * x51;
    const scalar_t x53 = lmbda * log(x50);
    const scalar_t x54 = lmbda * x52 + mu * x52 + x0 - x52 * x53;
    const scalar_t x55 = mu * (Fa_inv[0] * Fa_inv[6] + Fa_inv[1] * Fa_inv[7] + Fa_inv[2] * Fa_inv[8]);
    const scalar_t x56 = x21 * x51;
    const scalar_t x57 = x31 + x32 + x33 + x35 + x36 + x37 - x44 - x45 - x46 - x47 - x48 - x49;
    const scalar_t x58 = lmbda * x57;
    const scalar_t x59 = mu * x56;
    const scalar_t x60 = x53 * x56;
    const scalar_t x61 = x55 + x56 * x58 + x57 * x59 - x57 * x60;
    const scalar_t x62 = mu * (Fa_inv[0] * Fa_inv[3] + Fa_inv[1] * Fa_inv[4] + Fa_inv[2] * Fa_inv[5]);
    const scalar_t x63 = -x23 - x24 - x25 - x27 - x28 - x29 + x38 + x39 + x40 + x41 + x42 + x43;
    const scalar_t x64 = lmbda * x63;
    const scalar_t x65 = -x56 * x64 - x59 * x63 + x60 * x63 + x62;
    const scalar_t x66 = adjugate[0] * x54 + adjugate[1] * x65 + adjugate[2] * x61;
    const scalar_t x67 = mu * (POW2(Fa_inv[6]) + POW2(Fa_inv[7]) + POW2(Fa_inv[8]));
    const scalar_t x68 = x51 * POW2(x57);
    const scalar_t x69 = lmbda * x68 + mu * x68 - x53 * x68 + x67;
    const scalar_t x70 = mu * (Fa_inv[3] * Fa_inv[6] + Fa_inv[4] * Fa_inv[7] + Fa_inv[5] * Fa_inv[8]);
    const scalar_t x71 = x51 * x57;
    const scalar_t x72 = mu * x63;
    const scalar_t x73 = x53 * x63;
    const scalar_t x74 = -x64 * x71 + x70 - x71 * x72 + x71 * x73;
    const scalar_t x75 = adjugate[0] * x61 + adjugate[1] * x74 + adjugate[2] * x69;
    const scalar_t x76 = mu * (POW2(Fa_inv[3]) + POW2(Fa_inv[4]) + POW2(Fa_inv[5]));
    const scalar_t x77 = x51 * POW2(x63);
    const scalar_t x78 = lmbda * x77 + mu * x77 - x53 * x77 + x76;
    const scalar_t x79 = adjugate[0] * x65 + adjugate[1] * x78 + adjugate[2] * x74;
    const scalar_t x80 = Ja * qw / jacobian_determinant;
    const scalar_t x81 = adjugate[3] * x54 + adjugate[4] * x65 + adjugate[5] * x61;
    const scalar_t x82 = adjugate[3] * x61 + adjugate[4] * x74 + adjugate[5] * x69;
    const scalar_t x83 = adjugate[3] * x65 + adjugate[4] * x78 + adjugate[5] * x74;
    const scalar_t x84 = 1.0 / x50;
    const scalar_t x85 = F[0] * F[8];
    const scalar_t x86 = F[2] * F[6];
    const scalar_t x87 = x1 * x85 - x1 * x86 - x11 * x85 + x11 * x86 - x13 * x85 + x13 * x86 + x4 * x85 - x4 * x86 + x6 * x85 -
                         x6 * x86 - x8 * x85 + x8 * x86;
    const scalar_t x88 = x84 * x87;
    const scalar_t x89 = lmbda + mu - x53;
    const scalar_t x90 = x63 * x88 * x89;
    const scalar_t x91 = x1 - x11 - x13 + x4 + x6 - x8;
    const scalar_t x92 = mu * x91;
    const scalar_t x93 = F[8] * x92;
    const scalar_t x94 = x53 * x91;
    const scalar_t x95 = F[8] * x94;
    const scalar_t x96 = F[1] * F[8];
    const scalar_t x97 = F[2] * F[7];
    const scalar_t x98 = x1 * x96 - x1 * x97 - x11 * x96 + x11 * x97 - x13 * x96 + x13 * x97 + x4 * x96 - x4 * x97 + x6 * x96 -
                         x6 * x97 - x8 * x96 + x8 * x97;
    const scalar_t x99  = x84 * x98;
    const scalar_t x100 = x64 * x99 + x72 * x99 - x73 * x99 + x93 - x95;
    const scalar_t x101 = F[6] * x92;
    const scalar_t x102 = F[6] * x94;
    const scalar_t x103 = F[0] * F[7];
    const scalar_t x104 = F[1] * F[6];
    const scalar_t x105 = x1 * x103 - x1 * x104 - x103 * x11 - x103 * x13 + x103 * x4 + x103 * x6 - x103 * x8 + x104 * x11 +
                          x104 * x13 - x104 * x4 - x104 * x6 + x104 * x8;
    const scalar_t x106 = x105 * x84;
    const scalar_t x107 = -x101 + x102 + x106 * x64 + x106 * x72 - x106 * x73;
    const scalar_t x108 = adjugate[0] * x100 - adjugate[1] * x90 + adjugate[2] * x107;
    const scalar_t x109 = x21 * x84;
    const scalar_t x110 = x109 * x89;
    const scalar_t x111 = x110 * x98;
    const scalar_t x112 = F[7] * x92;
    const scalar_t x113 = F[7] * x94;
    const scalar_t x114 = lmbda * x109;
    const scalar_t x115 = mu * x109;
    const scalar_t x116 = x109 * x53;
    const scalar_t x117 = x105 * x114 + x105 * x115 - x105 * x116 - x112 + x113;
    const scalar_t x118 = lmbda * x87;
    const scalar_t x119 = x109 * x118 + x115 * x87 - x116 * x87 - x93 + x95;
    const scalar_t x120 = -adjugate[0] * x111 + adjugate[1] * x119 - adjugate[2] * x117;
    const scalar_t x121 = x57 * x89;
    const scalar_t x122 = x106 * x121;
    const scalar_t x123 = mu * x57;
    const scalar_t x124 = x53 * x57;
    const scalar_t x125 = x112 - x113 + x123 * x99 - x124 * x99 + x58 * x99;
    const scalar_t x126 = x101 - x102 + x123 * x88 - x124 * x88 + x58 * x88;
    const scalar_t x127 = -adjugate[0] * x125 + adjugate[1] * x126 - adjugate[2] * x122;
    const scalar_t x128 = x80 * x84;
    const scalar_t x129 = adjugate[3] * x100 - adjugate[4] * x90 + adjugate[5] * x107;
    const scalar_t x130 = -adjugate[3] * x111 + adjugate[4] * x119 - adjugate[5] * x117;
    const scalar_t x131 = -adjugate[3] * x125 + adjugate[4] * x126 - adjugate[5] * x122;
    const scalar_t x132 = adjugate[6] * x100 - adjugate[7] * x90 + adjugate[8] * x107;
    const scalar_t x133 = -adjugate[6] * x111 + adjugate[7] * x119 - adjugate[8] * x117;
    const scalar_t x134 = -adjugate[6] * x125 + adjugate[7] * x126 - adjugate[8] * x122;
    const scalar_t x135 = F[1] * F[5];
    const scalar_t x136 = F[2] * F[4];
    const scalar_t x137 = x1 * x135 - x1 * x136 - x11 * x135 + x11 * x136 - x13 * x135 + x13 * x136 + x135 * x4 + x135 * x6 -
                          x135 * x8 - x136 * x4 - x136 * x6 + x136 * x8;
    const scalar_t x138 = x110 * x137;
    const scalar_t x139 = F[4] * x92;
    const scalar_t x140 = F[4] * x94;
    const scalar_t x141 = F[0] * F[4];
    const scalar_t x142 = F[1] * F[3];
    const scalar_t x143 = x1 * x141 - x1 * x142 - x11 * x141 + x11 * x142 - x13 * x141 + x13 * x142 + x141 * x4 + x141 * x6 -
                          x141 * x8 - x142 * x4 - x142 * x6 + x142 * x8;
    const scalar_t x144 = x114 * x143 + x115 * x143 - x116 * x143 - x139 + x140;
    const scalar_t x145 = F[5] * x92;
    const scalar_t x146 = F[5] * x94;
    const scalar_t x147 = F[0] * F[5];
    const scalar_t x148 = F[2] * F[3];
    const scalar_t x149 = x1 * x147 - x1 * x148 - x11 * x147 + x11 * x148 - x13 * x147 + x13 * x148 + x147 * x4 + x147 * x6 -
                          x147 * x8 - x148 * x4 - x148 * x6 + x148 * x8;
    const scalar_t x150 = x114 * x149 + x115 * x149 - x116 * x149 - x145 + x146;
    const scalar_t x151 = adjugate[0] * x138 - adjugate[1] * x150 + adjugate[2] * x144;
    const scalar_t x152 = x143 * x84;
    const scalar_t x153 = x121 * x152;
    const scalar_t x154 = x137 * x84;
    const scalar_t x155 = x123 * x154 - x124 * x154 + x139 - x140 + x154 * x58;
    const scalar_t x156 = F[3] * x92;
    const scalar_t x157 = F[3] * x94;
    const scalar_t x158 = x149 * x84;
    const scalar_t x159 = x123 * x158 - x124 * x158 + x156 - x157 + x158 * x58;
    const scalar_t x160 = adjugate[0] * x155 - adjugate[1] * x159 + adjugate[2] * x153;
    const scalar_t x161 = x145 - x146 + x154 * x64 + x154 * x72 - x154 * x73;
    const scalar_t x162 = x152 * x64 + x152 * x72 - x152 * x73 - x156 + x157;
    const scalar_t x163 = -adjugate[0] * x161 + adjugate[1] * x149 * x63 * x84 * x89 - adjugate[2] * x162;
    const scalar_t x164 = adjugate[3] * x138 - adjugate[4] * x150 + adjugate[5] * x144;
    const scalar_t x165 = adjugate[3] * x155 - adjugate[4] * x159 + adjugate[5] * x153;
    const scalar_t x166 = -adjugate[3] * x161 + adjugate[4] * x149 * x63 * x84 * x89 - adjugate[5] * x162;
    const scalar_t x167 = adjugate[6] * x138 - adjugate[7] * x150 + adjugate[8] * x144;
    const scalar_t x168 = adjugate[6] * x155 - adjugate[7] * x159 + adjugate[8] * x153;
    const scalar_t x169 = -adjugate[6] * x161 + adjugate[7] * x149 * x63 * x84 * x89 - adjugate[8] * x162;
    const scalar_t x170 = x51 * POW2(x98);
    const scalar_t x171 = lmbda * x170 + mu * x170 + x0 - x170 * x53;
    const scalar_t x172 = x51 * x98;
    const scalar_t x173 = x105 * x172;
    const scalar_t x174 = mu * x172;
    const scalar_t x175 = lmbda * x173 + x105 * x174 - x173 * x53 + x55;
    const scalar_t x176 = -x118 * x172 + x172 * x53 * x87 - x174 * x87 + x62;
    const scalar_t x177 = adjugate[0] * x171 + adjugate[1] * x176 + adjugate[2] * x175;
    const scalar_t x178 = POW2(x105) * x51;
    const scalar_t x179 = lmbda * x178 + mu * x178 - x178 * x53 + x67;
    const scalar_t x180 = x105 * x51;
    const scalar_t x181 = x180 * x87;
    const scalar_t x182 = -mu * x181 - x118 * x180 + x181 * x53 + x70;
    const scalar_t x183 = adjugate[0] * x175 + adjugate[1] * x182 + adjugate[2] * x179;
    const scalar_t x184 = x51 * POW2(x87);
    const scalar_t x185 = lmbda * x184 + mu * x184 - x184 * x53 + x76;
    const scalar_t x186 = adjugate[0] * x176 + adjugate[1] * x185 + adjugate[2] * x182;
    const scalar_t x187 = adjugate[3] * x171 + adjugate[4] * x176 + adjugate[5] * x175;
    const scalar_t x188 = adjugate[3] * x175 + adjugate[4] * x182 + adjugate[5] * x179;
    const scalar_t x189 = adjugate[3] * x176 + adjugate[4] * x185 + adjugate[5] * x182;
    const scalar_t x190 = x149 * x88 * x89;
    const scalar_t x191 = F[2] * x92;
    const scalar_t x192 = F[2] * x94;
    const scalar_t x193 = mu * x88;
    const scalar_t x194 = x53 * x88;
    const scalar_t x195 = x118 * x154 + x137 * x193 - x137 * x194 + x191 - x192;
    const scalar_t x196 = F[0] * x92;
    const scalar_t x197 = F[0] * x94;
    const scalar_t x198 = x118 * x152 + x143 * x193 - x143 * x194 - x196 + x197;
    const scalar_t x199 = adjugate[0] * x195 - adjugate[1] * x190 + adjugate[2] * x198;
    const scalar_t x200 = x137 * x89 * x99;
    const scalar_t x201 = F[1] * x92;
    const scalar_t x202 = F[1] * x94;
    const scalar_t x203 = x143 * x99;
    const scalar_t x204 = lmbda * x203 + mu * x203 - x201 + x202 - x203 * x53;
    const scalar_t x205 = x149 * x99;
    const scalar_t x206 = lmbda * x205 + mu * x205 - x191 + x192 - x205 * x53;
    const scalar_t x207 = -adjugate[0] * x200 + adjugate[1] * x206 - adjugate[2] * x204;
    const scalar_t x208 = x106 * x143 * x89;
    const scalar_t x209 = x106 * x137;
    const scalar_t x210 = lmbda * x209 + mu * x209 + x201 - x202 - x209 * x53;
    const scalar_t x211 = x106 * x149;
    const scalar_t x212 = lmbda * x211 + mu * x211 + x196 - x197 - x211 * x53;
    const scalar_t x213 = -adjugate[0] * x210 + adjugate[1] * x212 - adjugate[2] * x208;
    const scalar_t x214 = adjugate[3] * x195 - adjugate[4] * x190 + adjugate[5] * x198;
    const scalar_t x215 = -adjugate[3] * x200 + adjugate[4] * x206 - adjugate[5] * x204;
    const scalar_t x216 = -adjugate[3] * x210 + adjugate[4] * x212 - adjugate[5] * x208;
    const scalar_t x217 = adjugate[6] * x195 - adjugate[7] * x190 + adjugate[8] * x198;
    const scalar_t x218 = -adjugate[6] * x200 + adjugate[7] * x206 - adjugate[8] * x204;
    const scalar_t x219 = -adjugate[6] * x210 + adjugate[7] * x212 - adjugate[8] * x208;
    const scalar_t x220 = POW2(x137) * x51;
    const scalar_t x221 = lmbda * x220 + mu * x220 + x0 - x220 * x53;
    const scalar_t x222 = x137 * x51;
    const scalar_t x223 = x143 * x222;
    const scalar_t x224 = lmbda * x223 + mu * x223 - x223 * x53 + x55;
    const scalar_t x225 = x149 * x222;
    const scalar_t x226 = -lmbda * x225 - mu * x225 + x225 * x53 + x62;
    const scalar_t x227 = adjugate[0] * x221 + adjugate[1] * x226 + adjugate[2] * x224;
    const scalar_t x228 = POW2(x143) * x51;
    const scalar_t x229 = lmbda * x228 + mu * x228 - x228 * x53 + x67;
    const scalar_t x230 = x143 * x149 * x51;
    const scalar_t x231 = -lmbda * x230 - mu * x230 + x230 * x53 + x70;
    const scalar_t x232 = adjugate[0] * x224 + adjugate[1] * x231 + adjugate[2] * x229;
    const scalar_t x233 = POW2(x149) * x51;
    const scalar_t x234 = lmbda * x233 + mu * x233 - x233 * x53 + x76;
    const scalar_t x235 = adjugate[0] * x226 + adjugate[1] * x234 + adjugate[2] * x231;
    const scalar_t x236 = adjugate[3] * x221 + adjugate[4] * x226 + adjugate[5] * x224;
    const scalar_t x237 = adjugate[3] * x224 + adjugate[4] * x231 + adjugate[5] * x229;
    const scalar_t x238 = adjugate[3] * x226 + adjugate[4] * x234 + adjugate[5] * x231;
    S_ikmn_canonical[0] += x80 * (adjugate[0] * x66 + adjugate[1] * x79 + adjugate[2] * x75);
    S_ikmn_canonical[1] += x80 * (adjugate[3] * x66 + adjugate[4] * x79 + adjugate[5] * x75);
    S_ikmn_canonical[2] += x80 * (adjugate[6] * x66 + adjugate[7] * x79 + adjugate[8] * x75);
    S_ikmn_canonical[3] += x80 * (adjugate[3] * x81 + adjugate[4] * x83 + adjugate[5] * x82);
    S_ikmn_canonical[4] += x80 * (adjugate[6] * x81 + adjugate[7] * x83 + adjugate[8] * x82);
    S_ikmn_canonical[5] += x80 * (adjugate[6] * (adjugate[6] * x54 + adjugate[7] * x65 + adjugate[8] * x61) +
                                  adjugate[7] * (adjugate[6] * x65 + adjugate[7] * x78 + adjugate[8] * x74) +
                                  adjugate[8] * (adjugate[6] * x61 + adjugate[7] * x74 + adjugate[8] * x69));
    S_ikmn_canonical[6] += x128 * (adjugate[0] * x120 + adjugate[1] * x108 + adjugate[2] * x127);
    S_ikmn_canonical[7] += x128 * (adjugate[3] * x120 + adjugate[4] * x108 + adjugate[5] * x127);
    S_ikmn_canonical[8] += x128 * (adjugate[6] * x120 + adjugate[7] * x108 + adjugate[8] * x127);
    S_ikmn_canonical[9] += x128 * (adjugate[0] * x130 + adjugate[1] * x129 + adjugate[2] * x131);
    S_ikmn_canonical[10] += x128 * (adjugate[3] * x130 + adjugate[4] * x129 + adjugate[5] * x131);
    S_ikmn_canonical[11] += x128 * (adjugate[6] * x130 + adjugate[7] * x129 + adjugate[8] * x131);
    S_ikmn_canonical[12] += x128 * (adjugate[0] * x133 + adjugate[1] * x132 + adjugate[2] * x134);
    S_ikmn_canonical[13] += x128 * (adjugate[3] * x133 + adjugate[4] * x132 + adjugate[5] * x134);
    S_ikmn_canonical[14] += x128 * (adjugate[6] * x133 + adjugate[7] * x132 + adjugate[8] * x134);
    S_ikmn_canonical[15] += x128 * (adjugate[0] * x151 + adjugate[1] * x163 + adjugate[2] * x160);
    S_ikmn_canonical[16] += x128 * (adjugate[3] * x151 + adjugate[4] * x163 + adjugate[5] * x160);
    S_ikmn_canonical[17] += x128 * (adjugate[6] * x151 + adjugate[7] * x163 + adjugate[8] * x160);
    S_ikmn_canonical[18] += x128 * (adjugate[0] * x164 + adjugate[1] * x166 + adjugate[2] * x165);
    S_ikmn_canonical[19] += x128 * (adjugate[3] * x164 + adjugate[4] * x166 + adjugate[5] * x165);
    S_ikmn_canonical[20] += x128 * (adjugate[6] * x164 + adjugate[7] * x166 + adjugate[8] * x165);
    S_ikmn_canonical[21] += x128 * (adjugate[0] * x167 + adjugate[1] * x169 + adjugate[2] * x168);
    S_ikmn_canonical[22] += x128 * (adjugate[3] * x167 + adjugate[4] * x169 + adjugate[5] * x168);
    S_ikmn_canonical[23] += x128 * (adjugate[6] * x167 + adjugate[7] * x169 + adjugate[8] * x168);
    S_ikmn_canonical[24] += x80 * (adjugate[0] * x177 + adjugate[1] * x186 + adjugate[2] * x183);
    S_ikmn_canonical[25] += x80 * (adjugate[3] * x177 + adjugate[4] * x186 + adjugate[5] * x183);
    S_ikmn_canonical[26] += x80 * (adjugate[6] * x177 + adjugate[7] * x186 + adjugate[8] * x183);
    S_ikmn_canonical[27] += x80 * (adjugate[3] * x187 + adjugate[4] * x189 + adjugate[5] * x188);
    S_ikmn_canonical[28] += x80 * (adjugate[6] * x187 + adjugate[7] * x189 + adjugate[8] * x188);
    S_ikmn_canonical[29] += x80 * (adjugate[6] * (adjugate[6] * x171 + adjugate[7] * x176 + adjugate[8] * x175) +
                                   adjugate[7] * (adjugate[6] * x176 + adjugate[7] * x185 + adjugate[8] * x182) +
                                   adjugate[8] * (adjugate[6] * x175 + adjugate[7] * x182 + adjugate[8] * x179));
    S_ikmn_canonical[30] += x128 * (adjugate[0] * x207 + adjugate[1] * x199 + adjugate[2] * x213);
    S_ikmn_canonical[31] += x128 * (adjugate[3] * x207 + adjugate[4] * x199 + adjugate[5] * x213);
    S_ikmn_canonical[32] += x128 * (adjugate[6] * x207 + adjugate[7] * x199 + adjugate[8] * x213);
    S_ikmn_canonical[33] += x128 * (adjugate[0] * x215 + adjugate[1] * x214 + adjugate[2] * x216);
    S_ikmn_canonical[34] += x128 * (adjugate[3] * x215 + adjugate[4] * x214 + adjugate[5] * x216);
    S_ikmn_canonical[35] += x128 * (adjugate[6] * x215 + adjugate[7] * x214 + adjugate[8] * x216);
    S_ikmn_canonical[36] += x128 * (adjugate[0] * x218 + adjugate[1] * x217 + adjugate[2] * x219);
    S_ikmn_canonical[37] += x128 * (adjugate[3] * x218 + adjugate[4] * x217 + adjugate[5] * x219);
    S_ikmn_canonical[38] += x128 * (adjugate[6] * x218 + adjugate[7] * x217 + adjugate[8] * x219);
    S_ikmn_canonical[39] += x80 * (adjugate[0] * x227 + adjugate[1] * x235 + adjugate[2] * x232);
    S_ikmn_canonical[40] += x80 * (adjugate[3] * x227 + adjugate[4] * x235 + adjugate[5] * x232);
    S_ikmn_canonical[41] += x80 * (adjugate[6] * x227 + adjugate[7] * x235 + adjugate[8] * x232);
    S_ikmn_canonical[42] += x80 * (adjugate[3] * x236 + adjugate[4] * x238 + adjugate[5] * x237);
    S_ikmn_canonical[43] += x80 * (adjugate[6] * x236 + adjugate[7] * x238 + adjugate[8] * x237);
    S_ikmn_canonical[44] += x80 * (adjugate[6] * (adjugate[6] * x221 + adjugate[7] * x226 + adjugate[8] * x224) +
                                   adjugate[7] * (adjugate[6] * x226 + adjugate[7] * x234 + adjugate[8] * x231) +
                                   adjugate[8] * (adjugate[6] * x224 + adjugate[7] * x231 + adjugate[8] * x229));
}

#endif /* SFEM_HEX8_PARTIAL_ASSEMBLY_NEOHOOKEAN_OGDEN_ACTIVE_STRAIN_INLINE_H */
