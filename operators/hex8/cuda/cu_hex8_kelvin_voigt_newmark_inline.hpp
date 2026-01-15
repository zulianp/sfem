#ifndef CU_HEX8_KELVIN_VOIGT_NEWMARK_INLINE_HPP
#define CU_HEX8_KELVIN_VOIGT_NEWMARK_INLINE_HPP

#include "cu_hex8_inline.hpp"

//--------------------------
// hessian block_0_0
//--------------------------

template <typename scalar_t, typename accumulator_t>
static inline __host__ __device__ void cu_hex8_kelvin_voigt_newmark_matrix_block_0_0(const scalar_t                      k,
                                                                                     const scalar_t                      K,
                                                                                     const scalar_t                      eta,
                                                                                     const scalar_t                      rho,
                                                                                     const scalar_t                      dt,
                                                                                     const scalar_t                      gamma,
                                                                                     const scalar_t                      beta,
                                                                                     const scalar_t *const SFEM_RESTRICT adjugate,
                                                                                     const scalar_t jacobian_determinant,
                                                                                     const scalar_t qx,
                                                                                     const scalar_t qy,
                                                                                     const scalar_t qz,
                                                                                     const scalar_t qw,
                                                                                     accumulator_t *const SFEM_RESTRICT
                                                                                             element_matrix) {
    // mundane ops: 931 divs: 3 sqrts: 0
    // total ops: 955
    const scalar_t x0   = qx - 1;
    const scalar_t x1   = POW2(x0);
    const scalar_t x2   = qy - 1;
    const scalar_t x3   = POW2(x2);
    const scalar_t x4   = qz - 1;
    const scalar_t x5   = 1.0 / beta;
    const scalar_t x6   = jacobian_determinant * rho * x5 / POW2(dt);
    const scalar_t x7   = POW2(x4) * x6;
    const scalar_t x8   = x3 * x7;
    const scalar_t x9   = x2 * x4;
    const scalar_t x10  = adjugate[0] * x9;
    const scalar_t x11  = x0 * x4;
    const scalar_t x12  = adjugate[3] * x11;
    const scalar_t x13  = x0 * x2;
    const scalar_t x14  = adjugate[6] * x13;
    const scalar_t x15  = x10 + x12 + x14;
    const scalar_t x16  = POW2(x15);
    const scalar_t x17  = adjugate[1] * x9;
    const scalar_t x18  = adjugate[4] * x11;
    const scalar_t x19  = adjugate[7] * x13;
    const scalar_t x20  = x17 + x18 + x19;
    const scalar_t x21  = adjugate[2] * x9;
    const scalar_t x22  = adjugate[5] * x11;
    const scalar_t x23  = adjugate[8] * x13;
    const scalar_t x24  = x21 + x22 + x23;
    const scalar_t x25  = POW2(x20) + POW2(x24);
    const scalar_t x26  = 1.0 / dt;
    const scalar_t x27  = 1.0 / jacobian_determinant;
    const scalar_t x28  = (1.0 / 2.0) * x27;
    const scalar_t x29  = eta * gamma * x26 * x28 * x5;
    const scalar_t x30  = K - 0.33333333333333331 * k;
    const scalar_t x31  = 2 * x16;
    const scalar_t x32  = qx * x0;
    const scalar_t x33  = qx * x4;
    const scalar_t x34  = adjugate[3] * x33;
    const scalar_t x35  = qx * x2;
    const scalar_t x36  = adjugate[6] * x35;
    const scalar_t x37  = x10 + x34 + x36;
    const scalar_t x38  = x15 * x37;
    const scalar_t x39  = adjugate[4] * x33;
    const scalar_t x40  = adjugate[7] * x35;
    const scalar_t x41  = x17 + x39 + x40;
    const scalar_t x42  = adjugate[5] * x33;
    const scalar_t x43  = adjugate[8] * x35;
    const scalar_t x44  = x21 + x42 + x43;
    const scalar_t x45  = x20 * x41 + x24 * x44;
    const scalar_t x46  = 2 * x38;
    const scalar_t x47  = qw * ((1.0 / 2.0) * eta * gamma * x26 * x27 * x5 * (-1.3333333333333335 * x38 - x45) -
                               x28 * (k * (x45 + x46) + x30 * x46) - x32 * x8);
    const scalar_t x48  = qx * qy;
    const scalar_t x49  = x13 * x48;
    const scalar_t x50  = x49 * x7;
    const scalar_t x51  = adjugate[6] * x48;
    const scalar_t x52  = qy * x4;
    const scalar_t x53  = adjugate[0] * x52;
    const scalar_t x54  = x34 + x51 + x53;
    const scalar_t x55  = -x2;
    const scalar_t x56  = -x4;
    const scalar_t x57  = x55 * x56;
    const scalar_t x58  = adjugate[0] * x57;
    const scalar_t x59  = -x0;
    const scalar_t x60  = x56 * x59;
    const scalar_t x61  = adjugate[3] * x60;
    const scalar_t x62  = x55 * x59;
    const scalar_t x63  = adjugate[6] * x62;
    const scalar_t x64  = x58 + x61 + x63;
    const scalar_t x65  = x54 * x64;
    const scalar_t x66  = adjugate[7] * x48;
    const scalar_t x67  = adjugate[1] * x52;
    const scalar_t x68  = x39 + x66 + x67;
    const scalar_t x69  = adjugate[1] * x57;
    const scalar_t x70  = adjugate[4] * x60;
    const scalar_t x71  = adjugate[7] * x62;
    const scalar_t x72  = x69 + x70 + x71;
    const scalar_t x73  = adjugate[8] * x48;
    const scalar_t x74  = adjugate[2] * x52;
    const scalar_t x75  = x42 + x73 + x74;
    const scalar_t x76  = adjugate[2] * x57;
    const scalar_t x77  = adjugate[5] * x60;
    const scalar_t x78  = adjugate[8] * x62;
    const scalar_t x79  = x76 + x77 + x78;
    const scalar_t x80  = x68 * x72 + x75 * x79;
    const scalar_t x81  = 2 * x30;
    const scalar_t x82  = x15 * x81;
    const scalar_t x83  = qw * (x28 * (k * (2 * x65 + x80) + x54 * x82) - x29 * (-1.3333333333333335 * x65 - x80) + x50);
    const scalar_t x84  = qy * x2;
    const scalar_t x85  = x7 * x84;
    const scalar_t x86  = qy * x0;
    const scalar_t x87  = adjugate[6] * x86;
    const scalar_t x88  = x12 + x53 + x87;
    const scalar_t x89  = x15 * x88;
    const scalar_t x90  = adjugate[7] * x86;
    const scalar_t x91  = x18 + x67 + x90;
    const scalar_t x92  = adjugate[8] * x86;
    const scalar_t x93  = x22 + x74 + x92;
    const scalar_t x94  = x20 * x91 + x24 * x93;
    const scalar_t x95  = 2 * x89;
    const scalar_t x96  = qw * ((1.0 / 2.0) * eta * gamma * x26 * x27 * x5 * (-1.3333333333333335 * x89 - x94) - x1 * x85 -
                               x28 * (k * (x94 + x95) + x30 * x95));
    const scalar_t x97  = x3 * x6;
    const scalar_t x98  = qz * x4;
    const scalar_t x99  = x97 * x98;
    const scalar_t x100 = qz * x2;
    const scalar_t x101 = adjugate[0] * x100;
    const scalar_t x102 = qz * x0;
    const scalar_t x103 = adjugate[3] * x102;
    const scalar_t x104 = x101 + x103 + x14;
    const scalar_t x105 = x104 * x15;
    const scalar_t x106 = adjugate[1] * x100;
    const scalar_t x107 = adjugate[4] * x102;
    const scalar_t x108 = x106 + x107 + x19;
    const scalar_t x109 = adjugate[2] * x100;
    const scalar_t x110 = adjugate[5] * x102;
    const scalar_t x111 = x109 + x110 + x23;
    const scalar_t x112 = x108 * x20 + x111 * x24;
    const scalar_t x113 = 2 * x105;
    const scalar_t x114 = qw * ((1.0 / 2.0) * eta * gamma * x26 * x27 * x5 * (-1.3333333333333335 * x105 - x112) - x1 * x99 -
                                x28 * (k * (x112 + x113) + x113 * x30));
    const scalar_t x115 = qx * qz;
    const scalar_t x116 = x11 * x115;
    const scalar_t x117 = x116 * x97;
    const scalar_t x118 = adjugate[3] * x115;
    const scalar_t x119 = x101 + x118 + x36;
    const scalar_t x120 = x119 * x64;
    const scalar_t x121 = adjugate[4] * x115;
    const scalar_t x122 = x106 + x121 + x40;
    const scalar_t x123 = adjugate[5] * x115;
    const scalar_t x124 = x109 + x123 + x43;
    const scalar_t x125 = x122 * x72 + x124 * x79;
    const scalar_t x126 = qw * (x117 + x28 * (k * (2 * x120 + x125) + x119 * x82) - x29 * (-1.3333333333333335 * x120 - x125));
    const scalar_t x127 = x102 * x48 * x6 * x9;
    const scalar_t x128 = qy * qz;
    const scalar_t x129 = adjugate[0] * x128;
    const scalar_t x130 = x118 + x129 + x51;
    const scalar_t x131 = x130 * x64;
    const scalar_t x132 = adjugate[1] * x128;
    const scalar_t x133 = x121 + x132 + x66;
    const scalar_t x134 = adjugate[2] * x128;
    const scalar_t x135 = x123 + x134 + x73;
    const scalar_t x136 = x133 * x72 + x135 * x79;
    const scalar_t x137 = -qw * (x127 + x28 * (k * (2 * x131 + x136) + x130 * x82) + x29 * (1.3333333333333335 * x131 + x136));
    const scalar_t x138 = x1 * x6;
    const scalar_t x139 = x128 * x9;
    const scalar_t x140 = x138 * x139;
    const scalar_t x141 = x103 + x129 + x87;
    const scalar_t x142 = x141 * x64;
    const scalar_t x143 = x107 + x132 + x90;
    const scalar_t x144 = x110 + x134 + x92;
    const scalar_t x145 = x143 * x72 + x144 * x79;
    const scalar_t x146 = qw * (x140 + x28 * (k * (2 * x142 + x145) + x141 * x82) - x29 * (-1.3333333333333335 * x142 - x145));
    const scalar_t x147 = POW2(qx);
    const scalar_t x148 = POW2(x37);
    const scalar_t x149 = POW2(x41) + POW2(x44);
    const scalar_t x150 = 2 * x148;
    const scalar_t x151 = qx * x56;
    const scalar_t x152 = qx * x55;
    const scalar_t x153 = adjugate[3] * x151 + adjugate[6] * x152 - x58;
    const scalar_t x154 = x153 * x54;
    const scalar_t x155 = adjugate[4] * x151 + adjugate[7] * x152 - x69;
    const scalar_t x156 = adjugate[5] * x151 + adjugate[8] * x152 - x76;
    const scalar_t x157 = x155 * x68 + x156 * x75;
    const scalar_t x158 = x37 * x81;
    const scalar_t x159 = qw * (-x147 * x85 + (1.0 / 2.0) * x27 * (k * (2 * x154 + x157) - x158 * x54) -
                                x29 * (-1.3333333333333335 * x154 - x157));
    const scalar_t x160 = x37 * x88;
    const scalar_t x161 = x41 * x91 + x44 * x93;
    const scalar_t x162 = 2 * x160;
    const scalar_t x163 = qw * (x28 * (k * (x161 + x162) + x162 * x30) + x29 * (1.3333333333333335 * x160 + x161) + x50);
    const scalar_t x164 = x104 * x37;
    const scalar_t x165 = x108 * x41 + x111 * x44;
    const scalar_t x166 = 2 * x164;
    const scalar_t x167 = qw * (x117 + x28 * (k * (x165 + x166) + x166 * x30) + x29 * (1.3333333333333335 * x164 + x165));
    const scalar_t x168 = x119 * x153;
    const scalar_t x169 = x122 * x155 + x124 * x156;
    const scalar_t x170 = qw * (-x147 * x99 + (1.0 / 2.0) * x27 * (k * (2 * x168 + x169) - x119 * x158) -
                                x29 * (-1.3333333333333335 * x168 - x169));
    const scalar_t x171 = x147 * x6;
    const scalar_t x172 = x139 * x171;
    const scalar_t x173 = x130 * x153;
    const scalar_t x174 = x133 * x155 + x135 * x156;
    const scalar_t x175 =
            qw * (x172 + x28 * (-k * (2 * x173 + x174) + 2 * x130 * x30 * x37) - x29 * (1.3333333333333335 * x173 + x174));
    const scalar_t x176 = x141 * x153;
    const scalar_t x177 = x143 * x155 + x144 * x156;
    const scalar_t x178 =
            qw * (-x127 + (1.0 / 2.0) * x27 * (k * (2 * x176 + x177) - x141 * x158) - x29 * (-1.3333333333333335 * x176 - x177));
    const scalar_t x179 = POW2(qy);
    const scalar_t x180 = x179 * x7;
    const scalar_t x181 = POW2(x54);
    const scalar_t x182 = POW2(x68) + POW2(x75);
    const scalar_t x183 = 2 * x181;
    const scalar_t x184 = qy * x56;
    const scalar_t x185 = qy * x59;
    const scalar_t x186 = adjugate[0] * x184 + adjugate[6] * x185 - x61;
    const scalar_t x187 = x186 * x54;
    const scalar_t x188 = adjugate[1] * x184 + adjugate[7] * x185 - x70;
    const scalar_t x189 = adjugate[2] * x184 + adjugate[8] * x185 - x77;
    const scalar_t x190 = x188 * x68 + x189 * x75;
    const scalar_t x191 = x54 * x81;
    const scalar_t x192 = qw * (-x180 * x32 + (1.0 / 2.0) * x27 * (k * (2 * x187 + x190) - x191 * x88) -
                                x29 * (-1.3333333333333335 * x187 - x190));
    const scalar_t x193 = qz * x55;
    const scalar_t x194 = qz * x59;
    const scalar_t x195 = adjugate[0] * x193 + adjugate[3] * x194 - x63;
    const scalar_t x196 = x195 * x54;
    const scalar_t x197 = adjugate[1] * x193 + adjugate[4] * x194 - x71;
    const scalar_t x198 = adjugate[2] * x193 + adjugate[5] * x194 - x78;
    const scalar_t x199 = x197 * x68 + x198 * x75;
    const scalar_t x200 =
            qw * (-x127 + (1.0 / 2.0) * x27 * (k * (2 * x196 + x199) - x104 * x191) - x29 * (-1.3333333333333335 * x196 - x199));
    const scalar_t x201 = x119 * x54;
    const scalar_t x202 = x122 * x68 + x124 * x75;
    const scalar_t x203 = 2 * x201;
    const scalar_t x204 = qw * (x172 + x28 * (k * (x202 + x203) + x203 * x30) + x29 * (1.3333333333333335 * x201 + x202));
    const scalar_t x205 = x179 * x98;
    const scalar_t x206 = x130 * x54;
    const scalar_t x207 = x133 * x68 + x135 * x75;
    const scalar_t x208 = 2 * x206;
    const scalar_t x209 = qw * ((1.0 / 2.0) * eta * gamma * x26 * x27 * x5 * (-1.3333333333333335 * x206 - x207) - x171 * x205 -
                                x28 * (k * (x207 + x208) + x208 * x30));
    const scalar_t x210 = x116 * x179 * x6;
    const scalar_t x211 = x141 * x54;
    const scalar_t x212 = x143 * x68 + x144 * x75;
    const scalar_t x213 = 2 * x211;
    const scalar_t x214 = qw * (x210 + x28 * (k * (x212 + x213) + x213 * x30) + x29 * (1.3333333333333335 * x211 + x212));
    const scalar_t x215 = POW2(x88);
    const scalar_t x216 = POW2(x91) + POW2(x93);
    const scalar_t x217 = 2 * x215;
    const scalar_t x218 = x104 * x88;
    const scalar_t x219 = x108 * x91 + x111 * x93;
    const scalar_t x220 = 2 * x218;
    const scalar_t x221 = qw * (x140 + x28 * (k * (x219 + x220) + x220 * x30) + x29 * (1.3333333333333335 * x218 + x219));
    const scalar_t x222 = x119 * x186;
    const scalar_t x223 = x122 * x188 + x124 * x189;
    const scalar_t x224 = x81 * x88;
    const scalar_t x225 =
            qw * (-x127 + (1.0 / 2.0) * x27 * (k * (2 * x222 + x223) - x119 * x224) - x29 * (-1.3333333333333335 * x222 - x223));
    const scalar_t x226 = x130 * x186;
    const scalar_t x227 = x133 * x188 + x135 * x189;
    const scalar_t x228 =
            qw * (x210 + x28 * (-k * (2 * x226 + x227) + 2 * x130 * x30 * x88) - x29 * (1.3333333333333335 * x226 + x227));
    const scalar_t x229 = x141 * x186;
    const scalar_t x230 = x143 * x188 + x144 * x189;
    const scalar_t x231 = qw * (-x138 * x205 + (1.0 / 2.0) * x27 * (k * (2 * x229 + x230) - x141 * x224) -
                                x29 * (-1.3333333333333335 * x229 - x230));
    const scalar_t x232 = POW2(qz) * x6;
    const scalar_t x233 = x232 * x3;
    const scalar_t x234 = POW2(x104);
    const scalar_t x235 = POW2(x108) + POW2(x111);
    const scalar_t x236 = 2 * x234;
    const scalar_t x237 = x119 * x195;
    const scalar_t x238 = x122 * x197 + x124 * x198;
    const scalar_t x239 = x104 * x81;
    const scalar_t x240 = qw * (-x233 * x32 + (1.0 / 2.0) * x27 * (k * (2 * x237 + x238) - x119 * x239) -
                                x29 * (-1.3333333333333335 * x237 - x238));
    const scalar_t x241 = x232 * x49;
    const scalar_t x242 = x130 * x195;
    const scalar_t x243 = x133 * x197 + x135 * x198;
    const scalar_t x244 =
            qw * (x241 + x28 * (-k * (2 * x242 + x243) + 2 * x104 * x130 * x30) - x29 * (1.3333333333333335 * x242 + x243));
    const scalar_t x245 = x232 * x84;
    const scalar_t x246 = x141 * x195;
    const scalar_t x247 = x143 * x197 + x144 * x198;
    const scalar_t x248 = qw * (-x1 * x245 + (1.0 / 2.0) * x27 * (k * (2 * x246 + x247) - x141 * x239) -
                                x29 * (-1.3333333333333335 * x246 - x247));
    const scalar_t x249 = POW2(x119);
    const scalar_t x250 = POW2(x122) + POW2(x124);
    const scalar_t x251 = 2 * x249;
    const scalar_t x252 = x119 * x130;
    const scalar_t x253 = x122 * x133 + x124 * x135;
    const scalar_t x254 = 2 * x252;
    const scalar_t x255 = qw * ((1.0 / 2.0) * eta * gamma * x26 * x27 * x5 * (-1.3333333333333335 * x252 - x253) - x147 * x245 -
                                x28 * (k * (x253 + x254) + x254 * x30));
    const scalar_t x256 = x119 * x141;
    const scalar_t x257 = x122 * x143 + x124 * x144;
    const scalar_t x258 = 2 * x256;
    const scalar_t x259 = qw * (x241 + x28 * (k * (x257 + x258) + x258 * x30) + x29 * (1.3333333333333335 * x256 + x257));
    const scalar_t x260 = x179 * x232;
    const scalar_t x261 = POW2(x130);
    const scalar_t x262 = POW2(x133) + POW2(x135);
    const scalar_t x263 = 2 * x261;
    const scalar_t x264 = x130 * x141;
    const scalar_t x265 = x133 * x143 + x135 * x144;
    const scalar_t x266 = 2 * x264;
    const scalar_t x267 = qw * ((1.0 / 2.0) * eta * gamma * x26 * x27 * x5 * (-1.3333333333333335 * x264 - x265) - x260 * x32 -
                                x28 * (k * (x265 + x266) + x266 * x30));
    const scalar_t x268 = POW2(x141);
    const scalar_t x269 = POW2(x143) + POW2(x144);
    const scalar_t x270 = 2 * x268;
    element_matrix[0] += qw * (x1 * x8 + x28 * (k * (x25 + x31) + x30 * x31) + x29 * (1.3333333333333335 * x16 + x25));
    element_matrix[1] += x47;
    element_matrix[2] += x83;
    element_matrix[3] += x96;
    element_matrix[4] += x114;
    element_matrix[5] += x126;
    element_matrix[6] += x137;
    element_matrix[7] += x146;
    element_matrix[8] += x47;
    element_matrix[9] += qw * (x147 * x8 + x28 * (k * (x149 + x150) + x150 * x30) + x29 * (1.3333333333333335 * x148 + x149));
    element_matrix[10] += x159;
    element_matrix[11] += x163;
    element_matrix[12] += x167;
    element_matrix[13] += x170;
    element_matrix[14] += x175;
    element_matrix[15] += x178;
    element_matrix[16] += x83;
    element_matrix[17] += x159;
    element_matrix[18] += qw * (x147 * x180 + x28 * (k * (x182 + x183) + x183 * x30) + x29 * (1.3333333333333335 * x181 + x182));
    element_matrix[19] += x192;
    element_matrix[20] += x200;
    element_matrix[21] += x204;
    element_matrix[22] += x209;
    element_matrix[23] += x214;
    element_matrix[24] += x96;
    element_matrix[25] += x163;
    element_matrix[26] += x192;
    element_matrix[27] += qw * (x1 * x180 + x28 * (k * (x216 + x217) + x217 * x30) + x29 * (1.3333333333333335 * x215 + x216));
    element_matrix[28] += x221;
    element_matrix[29] += x225;
    element_matrix[30] += x228;
    element_matrix[31] += x231;
    element_matrix[32] += x114;
    element_matrix[33] += x167;
    element_matrix[34] += x200;
    element_matrix[35] += x221;
    element_matrix[36] += qw * (x1 * x233 + x28 * (k * (x235 + x236) + x236 * x30) + x29 * (1.3333333333333335 * x234 + x235));
    element_matrix[37] += x240;
    element_matrix[38] += x244;
    element_matrix[39] += x248;
    element_matrix[40] += x126;
    element_matrix[41] += x170;
    element_matrix[42] += x204;
    element_matrix[43] += x225;
    element_matrix[44] += x240;
    element_matrix[45] += qw * (x147 * x233 + x28 * (k * (x250 + x251) + x251 * x30) + x29 * (1.3333333333333335 * x249 + x250));
    element_matrix[46] += x255;
    element_matrix[47] += x259;
    element_matrix[48] += x137;
    element_matrix[49] += x175;
    element_matrix[50] += x209;
    element_matrix[51] += x228;
    element_matrix[52] += x244;
    element_matrix[53] += x255;
    element_matrix[54] += qw * (x147 * x260 + x28 * (k * (x262 + x263) + x263 * x30) + x29 * (1.3333333333333335 * x261 + x262));
    element_matrix[55] += x267;
    element_matrix[56] += x146;
    element_matrix[57] += x178;
    element_matrix[58] += x214;
    element_matrix[59] += x231;
    element_matrix[60] += x248;
    element_matrix[61] += x259;
    element_matrix[62] += x267;
    element_matrix[63] += qw * (x1 * x260 + x28 * (k * (x269 + x270) + x270 * x30) + x29 * (1.3333333333333335 * x268 + x269));
}

//--------------------------
// hessian block_0_1
//--------------------------

template <typename scalar_t, typename accumulator_t>
static inline __host__ __device__ void cu_hex8_kelvin_voigt_newmark_matrix_block_0_1(const scalar_t                      k,
                                                                                     const scalar_t                      K,
                                                                                     const scalar_t                      eta,
                                                                                     const scalar_t                      rho,
                                                                                     const scalar_t                      dt,
                                                                                     const scalar_t                      gamma,
                                                                                     const scalar_t                      beta,
                                                                                     const scalar_t *const SFEM_RESTRICT adjugate,
                                                                                     const scalar_t jacobian_determinant,
                                                                                     const scalar_t qx,
                                                                                     const scalar_t qy,
                                                                                     const scalar_t qz,
                                                                                     const scalar_t qw,
                                                                                     accumulator_t *const SFEM_RESTRICT
                                                                                             element_matrix) {
    // mundane ops: 819 divs: 3 sqrts: 0
    // total ops: 843
    const scalar_t x0   = K + 0.16666666666666669 * k;
    const scalar_t x1   = qy - 1;
    const scalar_t x2   = qz - 1;
    const scalar_t x3   = x1 * x2;
    const scalar_t x4   = adjugate[0] * x3;
    const scalar_t x5   = qx - 1;
    const scalar_t x6   = x2 * x5;
    const scalar_t x7   = adjugate[3] * x6;
    const scalar_t x8   = x1 * x5;
    const scalar_t x9   = adjugate[6] * x8;
    const scalar_t x10  = x4 + x7 + x9;
    const scalar_t x11  = adjugate[1] * x3;
    const scalar_t x12  = adjugate[4] * x6;
    const scalar_t x13  = adjugate[7] * x8;
    const scalar_t x14  = x11 + x12 + x13;
    const scalar_t x15  = -x1;
    const scalar_t x16  = -x2;
    const scalar_t x17  = x15 * x16;
    const scalar_t x18  = adjugate[1] * x17;
    const scalar_t x19  = -x5;
    const scalar_t x20  = x16 * x19;
    const scalar_t x21  = adjugate[4] * x20;
    const scalar_t x22  = x15 * x19;
    const scalar_t x23  = adjugate[7] * x22;
    const scalar_t x24  = x18 + x21 + x23;
    const scalar_t x25  = adjugate[0] * x17;
    const scalar_t x26  = adjugate[3] * x20;
    const scalar_t x27  = adjugate[6] * x22;
    const scalar_t x28  = x25 + x26 + x27;
    const scalar_t x29  = 1.0 / beta;
    const scalar_t x30  = 1.0 / dt;
    const scalar_t x31  = eta * gamma * x29 * x30;
    const scalar_t x32  = 0.16666666666666669 * x31;
    const scalar_t x33  = qw / jacobian_determinant;
    const scalar_t x34  = qx * x2;
    const scalar_t x35  = adjugate[3] * x34;
    const scalar_t x36  = qx * x1;
    const scalar_t x37  = adjugate[6] * x36;
    const scalar_t x38  = x35 + x37 + x4;
    const scalar_t x39  = (1.0 / 2.0) * k;
    const scalar_t x40  = x38 * x39;
    const scalar_t x41  = adjugate[4] * x34;
    const scalar_t x42  = adjugate[7] * x36;
    const scalar_t x43  = x11 + x41 + x42;
    const scalar_t x44  = K - 0.33333333333333331 * k;
    const scalar_t x45  = x10 * x44;
    const scalar_t x46  = qx * x16;
    const scalar_t x47  = qx * x15;
    const scalar_t x48  = adjugate[3] * x46 + adjugate[6] * x47 - x25;
    const scalar_t x49  = x24 * x48;
    const scalar_t x50  = adjugate[4] * x46 + adjugate[7] * x47 - x18;
    const scalar_t x51  = (1.0 / 2.0) * x31;
    const scalar_t x52  = qx * qy;
    const scalar_t x53  = adjugate[6] * x52;
    const scalar_t x54  = qy * x2;
    const scalar_t x55  = adjugate[0] * x54;
    const scalar_t x56  = x35 + x53 + x55;
    const scalar_t x57  = x24 * x56;
    const scalar_t x58  = adjugate[7] * x52;
    const scalar_t x59  = adjugate[1] * x54;
    const scalar_t x60  = x41 + x58 + x59;
    const scalar_t x61  = x28 * x60;
    const scalar_t x62  = qy * x5;
    const scalar_t x63  = adjugate[6] * x62;
    const scalar_t x64  = x55 + x63 + x7;
    const scalar_t x65  = x14 * x64;
    const scalar_t x66  = adjugate[7] * x62;
    const scalar_t x67  = x12 + x59 + x66;
    const scalar_t x68  = qz * x1;
    const scalar_t x69  = adjugate[0] * x68;
    const scalar_t x70  = qz * x5;
    const scalar_t x71  = adjugate[3] * x70;
    const scalar_t x72  = x69 + x71 + x9;
    const scalar_t x73  = x14 * x72;
    const scalar_t x74  = adjugate[1] * x68;
    const scalar_t x75  = adjugate[4] * x70;
    const scalar_t x76  = x13 + x74 + x75;
    const scalar_t x77  = qx * qz;
    const scalar_t x78  = adjugate[3] * x77;
    const scalar_t x79  = x37 + x69 + x78;
    const scalar_t x80  = x24 * x79;
    const scalar_t x81  = adjugate[4] * x77;
    const scalar_t x82  = x42 + x74 + x81;
    const scalar_t x83  = x28 * x82;
    const scalar_t x84  = qy * qz;
    const scalar_t x85  = adjugate[0] * x84;
    const scalar_t x86  = x53 + x78 + x85;
    const scalar_t x87  = x24 * x86;
    const scalar_t x88  = adjugate[1] * x84;
    const scalar_t x89  = x58 + x81 + x88;
    const scalar_t x90  = x28 * x89;
    const scalar_t x91  = x63 + x71 + x85;
    const scalar_t x92  = x24 * x91;
    const scalar_t x93  = x66 + x75 + x88;
    const scalar_t x94  = x28 * x93;
    const scalar_t x95  = x10 * x39;
    const scalar_t x96  = x38 * x44;
    const scalar_t x97  = x50 * x56;
    const scalar_t x98  = x48 * x60;
    const scalar_t x99  = x43 * x64;
    const scalar_t x100 = 0.66666666666666663 * x38;
    const scalar_t x101 = x43 * x72;
    const scalar_t x102 = x50 * x79;
    const scalar_t x103 = x48 * x82;
    const scalar_t x104 = x50 * x86;
    const scalar_t x105 = x48 * x89;
    const scalar_t x106 = x50 * x91;
    const scalar_t x107 = x48 * x93;
    const scalar_t x108 = x33 * (x0 + x32);
    const scalar_t x109 = qy * x16;
    const scalar_t x110 = qy * x19;
    const scalar_t x111 = adjugate[0] * x109 + adjugate[6] * x110 - x26;
    const scalar_t x112 = x111 * x60;
    const scalar_t x113 = adjugate[1] * x109 + adjugate[7] * x110 - x21;
    const scalar_t x114 = x113 * x56;
    const scalar_t x115 = qz * x15;
    const scalar_t x116 = qz * x19;
    const scalar_t x117 = adjugate[0] * x115 + adjugate[3] * x116 - x27;
    const scalar_t x118 = x117 * x60;
    const scalar_t x119 = adjugate[1] * x115 + adjugate[4] * x116 - x23;
    const scalar_t x120 = x119 * x56;
    const scalar_t x121 = x60 * x79;
    const scalar_t x122 = x56 * x82;
    const scalar_t x123 = x60 * x86;
    const scalar_t x124 = x56 * x89;
    const scalar_t x125 = x60 * x91;
    const scalar_t x126 = x56 * x93;
    const scalar_t x127 = 0.66666666666666663 * x111;
    const scalar_t x128 = x67 * x72;
    const scalar_t x129 = x64 * x76;
    const scalar_t x130 = x113 * x79;
    const scalar_t x131 = x111 * x82;
    const scalar_t x132 = x113 * x86;
    const scalar_t x133 = x111 * x89;
    const scalar_t x134 = x113 * x91;
    const scalar_t x135 = x111 * x93;
    const scalar_t x136 = 0.66666666666666663 * x117;
    const scalar_t x137 = x119 * x79;
    const scalar_t x138 = x117 * x82;
    const scalar_t x139 = x119 * x86;
    const scalar_t x140 = x117 * x89;
    const scalar_t x141 = x119 * x91;
    const scalar_t x142 = x117 * x93;
    const scalar_t x143 = x82 * x86;
    const scalar_t x144 = x79 * x89;
    const scalar_t x145 = x82 * x91;
    const scalar_t x146 = x79 * x93;
    const scalar_t x147 = x89 * x91;
    const scalar_t x148 = x86 * x93;
    element_matrix[0] += x33 * (x0 * x10 * x14 + x24 * x28 * x32);
    element_matrix[1] += -x33 * (x14 * x40 + x43 * x45 + x51 * (0.66666666666666663 * x28 * x50 - x49));
    element_matrix[2] += x33 * (x39 * x57 + x44 * x61 - x51 * (0.66666666666666663 * x28 * x60 - x57));
    element_matrix[3] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (0.66666666666666663 * x10 * x67 - x65) - x39 * x65 - x45 * x67);
    element_matrix[4] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (0.66666666666666663 * x10 * x76 - x73) - x39 * x73 - x45 * x76);
    element_matrix[5] += x33 * (x39 * x80 + x44 * x83 - x51 * (0.66666666666666663 * x28 * x82 - x80));
    element_matrix[6] += -x33 * (x39 * x87 + x44 * x90 + x51 * (x87 - 0.66666666666666663 * x90));
    element_matrix[7] += x33 * (x39 * x92 + x44 * x94 - x51 * (0.66666666666666663 * x28 * x93 - x92));
    element_matrix[8] += -x33 * (x14 * x96 + x43 * x95 + x51 * (-x28 * x50 + 0.66666666666666663 * x49));
    element_matrix[9] += x33 * (x0 * x38 * x43 + x32 * x48 * x50);
    element_matrix[10] += x33 * (x39 * x97 + x44 * x98 - x51 * (0.66666666666666663 * x48 * x60 - x97));
    element_matrix[11] += x33 * (x39 * x99 + x51 * (-x100 * x67 + x99) + x67 * x96);
    element_matrix[12] += x33 * (x101 * x39 + x51 * (-x100 * x76 + x101) + x76 * x96);
    element_matrix[13] += x33 * (x102 * x39 + x103 * x44 - x51 * (-x102 + 0.66666666666666663 * x48 * x82));
    element_matrix[14] += -x33 * (x104 * x39 + x105 * x44 + x51 * (x104 - 0.66666666666666663 * x105));
    element_matrix[15] += x33 * (x106 * x39 + x107 * x44 - x51 * (-x106 + 0.66666666666666663 * x48 * x93));
    element_matrix[16] += x33 * (x39 * x61 + x44 * x57 - x51 * (0.66666666666666663 * x57 - x61));
    element_matrix[17] += x33 * (x39 * x98 + x44 * x97 - x51 * (0.66666666666666663 * x97 - x98));
    element_matrix[18] += x108 * x56 * x60;
    element_matrix[19] += x33 * (x112 * x39 + x114 * x44 - x51 * (-x112 + 0.66666666666666663 * x113 * x56));
    element_matrix[20] += x33 * (x118 * x39 + x120 * x44 - x51 * (-x118 + 0.66666666666666663 * x120));
    element_matrix[21] += x33 * (x121 * x39 + x122 * x44 - x51 * (-x121 + 0.66666666666666663 * x122));
    element_matrix[22] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (-x123 + 0.66666666666666663 * x56 * x89) - x123 * x39 - x124 * x44);
    element_matrix[23] += x33 * (x125 * x39 + x126 * x44 + x51 * (x125 - 0.66666666666666663 * x126));
    element_matrix[24] += -x33 * (x44 * x65 + x51 * (-x113 * x28 + x127 * x24) + x67 * x95);
    element_matrix[25] += x33 * (x40 * x67 + x44 * x99 - x51 * (-x113 * x48 + x127 * x50));
    element_matrix[26] += x33 * (x112 * x44 + x114 * x39 - x51 * (0.66666666666666663 * x112 - x114));
    element_matrix[27] += x33 * (x0 * x64 * x67 + x111 * x113 * x32);
    element_matrix[28] += x33 * (x128 * x39 + x129 * x44 - x51 * (-x113 * x117 + x119 * x127));
    element_matrix[29] += x33 * (x130 * x39 + x131 * x44 - x51 * (-x130 + 0.66666666666666663 * x131));
    element_matrix[30] += -x33 * (x132 * x39 + x133 * x44 + x51 * (x132 - 0.66666666666666663 * x133));
    element_matrix[31] += x33 * (x134 * x39 + x135 * x44 - x51 * (0.66666666666666663 * x111 * x93 - x134));
    element_matrix[32] += -x33 * (x44 * x73 + x51 * (-x119 * x28 + x136 * x24) + x76 * x95);
    element_matrix[33] += x33 * (x101 * x44 + x40 * x76 - x51 * (-x119 * x48 + x136 * x50));
    element_matrix[34] += x33 * (x118 * x44 + x120 * x39 - x51 * (0.66666666666666663 * x117 * x60 - x120));
    element_matrix[35] += x33 * (x128 * x44 + x129 * x39 + x51 * (-0.66666666666666663 * x128 + x129));
    element_matrix[36] += x33 * (x0 * x72 * x76 + x117 * x119 * x32);
    element_matrix[37] += x33 * (x137 * x39 + x138 * x44 - x51 * (-x137 + 0.66666666666666663 * x138));
    element_matrix[38] += -x33 * (x139 * x39 + x140 * x44 + x51 * (x139 - 0.66666666666666663 * x140));
    element_matrix[39] += x33 * (x141 * x39 + x142 * x44 - x51 * (0.66666666666666663 * x117 * x93 - x141));
    element_matrix[40] += x33 * (x39 * x83 + x44 * x80 - x51 * (0.66666666666666663 * x80 - x83));
    element_matrix[41] += x33 * (x102 * x44 + x103 * x39 - x51 * (0.66666666666666663 * x102 - x103));
    element_matrix[42] += x33 * (x121 * x44 + x122 * x39 + x51 * (-0.66666666666666663 * x121 + x122));
    element_matrix[43] += x33 * (x130 * x44 + x131 * x39 - x51 * (0.66666666666666663 * x113 * x79 - x131));
    element_matrix[44] += x33 * (x137 * x44 + x138 * x39 - x51 * (0.66666666666666663 * x119 * x79 - x138));
    element_matrix[45] += x108 * x79 * x82;
    element_matrix[46] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (-x143 + 0.66666666666666663 * x79 * x89) - x143 * x39 - x144 * x44);
    element_matrix[47] += x33 * (x145 * x39 + x146 * x44 + x51 * (x145 - 0.66666666666666663 * x146));
    element_matrix[48] += -x33 * (x39 * x90 + x44 * x87 + x51 * (x28 * x89 - 0.66666666666666663 * x87));
    element_matrix[49] += -x33 * (x104 * x44 + x105 * x39 + x51 * (-0.66666666666666663 * x104 + x48 * x89));
    element_matrix[50] += -x33 * (x123 * x44 + x124 * x39 + x51 * (-0.66666666666666663 * x123 + x56 * x89));
    element_matrix[51] += -x33 * (x132 * x44 + x133 * x39 + x51 * (x111 * x89 - 0.66666666666666663 * x132));
    element_matrix[52] += -x33 * (x139 * x44 + x140 * x39 + x51 * (x117 * x89 - 0.66666666666666663 * x139));
    element_matrix[53] += -x33 * (x143 * x44 + x144 * x39 + x51 * (-0.66666666666666663 * x143 + x79 * x89));
    element_matrix[54] += x108 * x86 * x89;
    element_matrix[55] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (-x147 + 0.66666666666666663 * x148) - x147 * x39 - x148 * x44);
    element_matrix[56] += x33 * (x39 * x94 + x44 * x92 - x51 * (0.66666666666666663 * x92 - x94));
    element_matrix[57] += x33 * (x106 * x44 + x107 * x39 - x51 * (0.66666666666666663 * x106 - x107));
    element_matrix[58] += x33 * (x125 * x44 + x126 * x39 - x51 * (0.66666666666666663 * x125 - x126));
    element_matrix[59] += x33 * (x134 * x44 + x135 * x39 - x51 * (0.66666666666666663 * x134 - x135));
    element_matrix[60] += x33 * (x141 * x44 + x142 * x39 - x51 * (0.66666666666666663 * x141 - x142));
    element_matrix[61] += x33 * (x145 * x44 + x146 * x39 - x51 * (0.66666666666666663 * x145 - x146));
    element_matrix[62] += -x33 * (x147 * x44 + x148 * x39 + x51 * (-0.66666666666666663 * x147 + x148));
    element_matrix[63] += x108 * x91 * x93;
}

//--------------------------
// hessian block_0_2
//--------------------------

template <typename scalar_t, typename accumulator_t>
static inline __host__ __device__ void cu_hex8_kelvin_voigt_newmark_matrix_block_0_2(const scalar_t                      k,
                                                                                     const scalar_t                      K,
                                                                                     const scalar_t                      eta,
                                                                                     const scalar_t                      rho,
                                                                                     const scalar_t                      dt,
                                                                                     const scalar_t                      gamma,
                                                                                     const scalar_t                      beta,
                                                                                     const scalar_t *const SFEM_RESTRICT adjugate,
                                                                                     const scalar_t jacobian_determinant,
                                                                                     const scalar_t qx,
                                                                                     const scalar_t qy,
                                                                                     const scalar_t qz,
                                                                                     const scalar_t qw,
                                                                                     accumulator_t *const SFEM_RESTRICT
                                                                                             element_matrix) {
    // mundane ops: 819 divs: 3 sqrts: 0
    // total ops: 843
    const scalar_t x0   = K + 0.16666666666666669 * k;
    const scalar_t x1   = qy - 1;
    const scalar_t x2   = qz - 1;
    const scalar_t x3   = x1 * x2;
    const scalar_t x4   = adjugate[0] * x3;
    const scalar_t x5   = qx - 1;
    const scalar_t x6   = x2 * x5;
    const scalar_t x7   = adjugate[3] * x6;
    const scalar_t x8   = x1 * x5;
    const scalar_t x9   = adjugate[6] * x8;
    const scalar_t x10  = x4 + x7 + x9;
    const scalar_t x11  = adjugate[2] * x3;
    const scalar_t x12  = adjugate[5] * x6;
    const scalar_t x13  = adjugate[8] * x8;
    const scalar_t x14  = x11 + x12 + x13;
    const scalar_t x15  = -x1;
    const scalar_t x16  = -x2;
    const scalar_t x17  = x15 * x16;
    const scalar_t x18  = adjugate[2] * x17;
    const scalar_t x19  = -x5;
    const scalar_t x20  = x16 * x19;
    const scalar_t x21  = adjugate[5] * x20;
    const scalar_t x22  = x15 * x19;
    const scalar_t x23  = adjugate[8] * x22;
    const scalar_t x24  = x18 + x21 + x23;
    const scalar_t x25  = adjugate[0] * x17;
    const scalar_t x26  = adjugate[3] * x20;
    const scalar_t x27  = adjugate[6] * x22;
    const scalar_t x28  = x25 + x26 + x27;
    const scalar_t x29  = 1.0 / beta;
    const scalar_t x30  = 1.0 / dt;
    const scalar_t x31  = eta * gamma * x29 * x30;
    const scalar_t x32  = 0.16666666666666669 * x31;
    const scalar_t x33  = qw / jacobian_determinant;
    const scalar_t x34  = qx * x2;
    const scalar_t x35  = adjugate[3] * x34;
    const scalar_t x36  = qx * x1;
    const scalar_t x37  = adjugate[6] * x36;
    const scalar_t x38  = x35 + x37 + x4;
    const scalar_t x39  = (1.0 / 2.0) * k;
    const scalar_t x40  = x38 * x39;
    const scalar_t x41  = adjugate[5] * x34;
    const scalar_t x42  = adjugate[8] * x36;
    const scalar_t x43  = x11 + x41 + x42;
    const scalar_t x44  = K - 0.33333333333333331 * k;
    const scalar_t x45  = x10 * x44;
    const scalar_t x46  = qx * x16;
    const scalar_t x47  = qx * x15;
    const scalar_t x48  = adjugate[3] * x46 + adjugate[6] * x47 - x25;
    const scalar_t x49  = x24 * x48;
    const scalar_t x50  = adjugate[5] * x46 + adjugate[8] * x47 - x18;
    const scalar_t x51  = (1.0 / 2.0) * x31;
    const scalar_t x52  = qx * qy;
    const scalar_t x53  = adjugate[6] * x52;
    const scalar_t x54  = qy * x2;
    const scalar_t x55  = adjugate[0] * x54;
    const scalar_t x56  = x35 + x53 + x55;
    const scalar_t x57  = x24 * x56;
    const scalar_t x58  = adjugate[8] * x52;
    const scalar_t x59  = adjugate[2] * x54;
    const scalar_t x60  = x41 + x58 + x59;
    const scalar_t x61  = x28 * x60;
    const scalar_t x62  = qy * x5;
    const scalar_t x63  = adjugate[6] * x62;
    const scalar_t x64  = x55 + x63 + x7;
    const scalar_t x65  = x14 * x64;
    const scalar_t x66  = adjugate[8] * x62;
    const scalar_t x67  = x12 + x59 + x66;
    const scalar_t x68  = qz * x1;
    const scalar_t x69  = adjugate[0] * x68;
    const scalar_t x70  = qz * x5;
    const scalar_t x71  = adjugate[3] * x70;
    const scalar_t x72  = x69 + x71 + x9;
    const scalar_t x73  = x14 * x72;
    const scalar_t x74  = adjugate[2] * x68;
    const scalar_t x75  = adjugate[5] * x70;
    const scalar_t x76  = x13 + x74 + x75;
    const scalar_t x77  = qx * qz;
    const scalar_t x78  = adjugate[3] * x77;
    const scalar_t x79  = x37 + x69 + x78;
    const scalar_t x80  = x24 * x79;
    const scalar_t x81  = adjugate[5] * x77;
    const scalar_t x82  = x42 + x74 + x81;
    const scalar_t x83  = x28 * x82;
    const scalar_t x84  = qy * qz;
    const scalar_t x85  = adjugate[0] * x84;
    const scalar_t x86  = x53 + x78 + x85;
    const scalar_t x87  = x24 * x86;
    const scalar_t x88  = adjugate[2] * x84;
    const scalar_t x89  = x58 + x81 + x88;
    const scalar_t x90  = x28 * x89;
    const scalar_t x91  = x63 + x71 + x85;
    const scalar_t x92  = x24 * x91;
    const scalar_t x93  = x66 + x75 + x88;
    const scalar_t x94  = x28 * x93;
    const scalar_t x95  = x10 * x39;
    const scalar_t x96  = x38 * x44;
    const scalar_t x97  = x50 * x56;
    const scalar_t x98  = x48 * x60;
    const scalar_t x99  = x43 * x64;
    const scalar_t x100 = 0.66666666666666663 * x38;
    const scalar_t x101 = x43 * x72;
    const scalar_t x102 = x50 * x79;
    const scalar_t x103 = x48 * x82;
    const scalar_t x104 = x50 * x86;
    const scalar_t x105 = x48 * x89;
    const scalar_t x106 = x50 * x91;
    const scalar_t x107 = x48 * x93;
    const scalar_t x108 = x33 * (x0 + x32);
    const scalar_t x109 = qy * x16;
    const scalar_t x110 = qy * x19;
    const scalar_t x111 = adjugate[0] * x109 + adjugate[6] * x110 - x26;
    const scalar_t x112 = x111 * x60;
    const scalar_t x113 = adjugate[2] * x109 + adjugate[8] * x110 - x21;
    const scalar_t x114 = x113 * x56;
    const scalar_t x115 = qz * x15;
    const scalar_t x116 = qz * x19;
    const scalar_t x117 = adjugate[0] * x115 + adjugate[3] * x116 - x27;
    const scalar_t x118 = x117 * x60;
    const scalar_t x119 = adjugate[2] * x115 + adjugate[5] * x116 - x23;
    const scalar_t x120 = x119 * x56;
    const scalar_t x121 = x60 * x79;
    const scalar_t x122 = x56 * x82;
    const scalar_t x123 = x60 * x86;
    const scalar_t x124 = x56 * x89;
    const scalar_t x125 = x60 * x91;
    const scalar_t x126 = x56 * x93;
    const scalar_t x127 = 0.66666666666666663 * x111;
    const scalar_t x128 = x67 * x72;
    const scalar_t x129 = x64 * x76;
    const scalar_t x130 = x113 * x79;
    const scalar_t x131 = x111 * x82;
    const scalar_t x132 = x113 * x86;
    const scalar_t x133 = x111 * x89;
    const scalar_t x134 = x113 * x91;
    const scalar_t x135 = x111 * x93;
    const scalar_t x136 = 0.66666666666666663 * x117;
    const scalar_t x137 = x119 * x79;
    const scalar_t x138 = x117 * x82;
    const scalar_t x139 = x119 * x86;
    const scalar_t x140 = x117 * x89;
    const scalar_t x141 = x119 * x91;
    const scalar_t x142 = x117 * x93;
    const scalar_t x143 = x82 * x86;
    const scalar_t x144 = x79 * x89;
    const scalar_t x145 = x82 * x91;
    const scalar_t x146 = x79 * x93;
    const scalar_t x147 = x89 * x91;
    const scalar_t x148 = x86 * x93;
    element_matrix[0] += x33 * (x0 * x10 * x14 + x24 * x28 * x32);
    element_matrix[1] += -x33 * (x14 * x40 + x43 * x45 + x51 * (0.66666666666666663 * x28 * x50 - x49));
    element_matrix[2] += x33 * (x39 * x57 + x44 * x61 - x51 * (0.66666666666666663 * x28 * x60 - x57));
    element_matrix[3] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (0.66666666666666663 * x10 * x67 - x65) - x39 * x65 - x45 * x67);
    element_matrix[4] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (0.66666666666666663 * x10 * x76 - x73) - x39 * x73 - x45 * x76);
    element_matrix[5] += x33 * (x39 * x80 + x44 * x83 - x51 * (0.66666666666666663 * x28 * x82 - x80));
    element_matrix[6] += -x33 * (x39 * x87 + x44 * x90 + x51 * (x87 - 0.66666666666666663 * x90));
    element_matrix[7] += x33 * (x39 * x92 + x44 * x94 - x51 * (0.66666666666666663 * x28 * x93 - x92));
    element_matrix[8] += -x33 * (x14 * x96 + x43 * x95 + x51 * (-x28 * x50 + 0.66666666666666663 * x49));
    element_matrix[9] += x33 * (x0 * x38 * x43 + x32 * x48 * x50);
    element_matrix[10] += x33 * (x39 * x97 + x44 * x98 - x51 * (0.66666666666666663 * x48 * x60 - x97));
    element_matrix[11] += x33 * (x39 * x99 + x51 * (-x100 * x67 + x99) + x67 * x96);
    element_matrix[12] += x33 * (x101 * x39 + x51 * (-x100 * x76 + x101) + x76 * x96);
    element_matrix[13] += x33 * (x102 * x39 + x103 * x44 - x51 * (-x102 + 0.66666666666666663 * x48 * x82));
    element_matrix[14] += -x33 * (x104 * x39 + x105 * x44 + x51 * (x104 - 0.66666666666666663 * x105));
    element_matrix[15] += x33 * (x106 * x39 + x107 * x44 - x51 * (-x106 + 0.66666666666666663 * x48 * x93));
    element_matrix[16] += x33 * (x39 * x61 + x44 * x57 - x51 * (0.66666666666666663 * x57 - x61));
    element_matrix[17] += x33 * (x39 * x98 + x44 * x97 - x51 * (0.66666666666666663 * x97 - x98));
    element_matrix[18] += x108 * x56 * x60;
    element_matrix[19] += x33 * (x112 * x39 + x114 * x44 - x51 * (-x112 + 0.66666666666666663 * x113 * x56));
    element_matrix[20] += x33 * (x118 * x39 + x120 * x44 - x51 * (-x118 + 0.66666666666666663 * x120));
    element_matrix[21] += x33 * (x121 * x39 + x122 * x44 - x51 * (-x121 + 0.66666666666666663 * x122));
    element_matrix[22] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (-x123 + 0.66666666666666663 * x56 * x89) - x123 * x39 - x124 * x44);
    element_matrix[23] += x33 * (x125 * x39 + x126 * x44 + x51 * (x125 - 0.66666666666666663 * x126));
    element_matrix[24] += -x33 * (x44 * x65 + x51 * (-x113 * x28 + x127 * x24) + x67 * x95);
    element_matrix[25] += x33 * (x40 * x67 + x44 * x99 - x51 * (-x113 * x48 + x127 * x50));
    element_matrix[26] += x33 * (x112 * x44 + x114 * x39 - x51 * (0.66666666666666663 * x112 - x114));
    element_matrix[27] += x33 * (x0 * x64 * x67 + x111 * x113 * x32);
    element_matrix[28] += x33 * (x128 * x39 + x129 * x44 - x51 * (-x113 * x117 + x119 * x127));
    element_matrix[29] += x33 * (x130 * x39 + x131 * x44 - x51 * (-x130 + 0.66666666666666663 * x131));
    element_matrix[30] += -x33 * (x132 * x39 + x133 * x44 + x51 * (x132 - 0.66666666666666663 * x133));
    element_matrix[31] += x33 * (x134 * x39 + x135 * x44 - x51 * (0.66666666666666663 * x111 * x93 - x134));
    element_matrix[32] += -x33 * (x44 * x73 + x51 * (-x119 * x28 + x136 * x24) + x76 * x95);
    element_matrix[33] += x33 * (x101 * x44 + x40 * x76 - x51 * (-x119 * x48 + x136 * x50));
    element_matrix[34] += x33 * (x118 * x44 + x120 * x39 - x51 * (0.66666666666666663 * x117 * x60 - x120));
    element_matrix[35] += x33 * (x128 * x44 + x129 * x39 + x51 * (-0.66666666666666663 * x128 + x129));
    element_matrix[36] += x33 * (x0 * x72 * x76 + x117 * x119 * x32);
    element_matrix[37] += x33 * (x137 * x39 + x138 * x44 - x51 * (-x137 + 0.66666666666666663 * x138));
    element_matrix[38] += -x33 * (x139 * x39 + x140 * x44 + x51 * (x139 - 0.66666666666666663 * x140));
    element_matrix[39] += x33 * (x141 * x39 + x142 * x44 - x51 * (0.66666666666666663 * x117 * x93 - x141));
    element_matrix[40] += x33 * (x39 * x83 + x44 * x80 - x51 * (0.66666666666666663 * x80 - x83));
    element_matrix[41] += x33 * (x102 * x44 + x103 * x39 - x51 * (0.66666666666666663 * x102 - x103));
    element_matrix[42] += x33 * (x121 * x44 + x122 * x39 + x51 * (-0.66666666666666663 * x121 + x122));
    element_matrix[43] += x33 * (x130 * x44 + x131 * x39 - x51 * (0.66666666666666663 * x113 * x79 - x131));
    element_matrix[44] += x33 * (x137 * x44 + x138 * x39 - x51 * (0.66666666666666663 * x119 * x79 - x138));
    element_matrix[45] += x108 * x79 * x82;
    element_matrix[46] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (-x143 + 0.66666666666666663 * x79 * x89) - x143 * x39 - x144 * x44);
    element_matrix[47] += x33 * (x145 * x39 + x146 * x44 + x51 * (x145 - 0.66666666666666663 * x146));
    element_matrix[48] += -x33 * (x39 * x90 + x44 * x87 + x51 * (x28 * x89 - 0.66666666666666663 * x87));
    element_matrix[49] += -x33 * (x104 * x44 + x105 * x39 + x51 * (-0.66666666666666663 * x104 + x48 * x89));
    element_matrix[50] += -x33 * (x123 * x44 + x124 * x39 + x51 * (-0.66666666666666663 * x123 + x56 * x89));
    element_matrix[51] += -x33 * (x132 * x44 + x133 * x39 + x51 * (x111 * x89 - 0.66666666666666663 * x132));
    element_matrix[52] += -x33 * (x139 * x44 + x140 * x39 + x51 * (x117 * x89 - 0.66666666666666663 * x139));
    element_matrix[53] += -x33 * (x143 * x44 + x144 * x39 + x51 * (-0.66666666666666663 * x143 + x79 * x89));
    element_matrix[54] += x108 * x86 * x89;
    element_matrix[55] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (-x147 + 0.66666666666666663 * x148) - x147 * x39 - x148 * x44);
    element_matrix[56] += x33 * (x39 * x94 + x44 * x92 - x51 * (0.66666666666666663 * x92 - x94));
    element_matrix[57] += x33 * (x106 * x44 + x107 * x39 - x51 * (0.66666666666666663 * x106 - x107));
    element_matrix[58] += x33 * (x125 * x44 + x126 * x39 - x51 * (0.66666666666666663 * x125 - x126));
    element_matrix[59] += x33 * (x134 * x44 + x135 * x39 - x51 * (0.66666666666666663 * x134 - x135));
    element_matrix[60] += x33 * (x141 * x44 + x142 * x39 - x51 * (0.66666666666666663 * x141 - x142));
    element_matrix[61] += x33 * (x145 * x44 + x146 * x39 - x51 * (0.66666666666666663 * x145 - x146));
    element_matrix[62] += -x33 * (x147 * x44 + x148 * x39 + x51 * (-0.66666666666666663 * x147 + x148));
    element_matrix[63] += x108 * x91 * x93;
}

//--------------------------
// hessian block_1_0
//--------------------------

template <typename scalar_t, typename accumulator_t>
static inline __host__ __device__ void cu_hex8_kelvin_voigt_newmark_matrix_block_1_0(const scalar_t                      k,
                                                                                     const scalar_t                      K,
                                                                                     const scalar_t                      eta,
                                                                                     const scalar_t                      rho,
                                                                                     const scalar_t                      dt,
                                                                                     const scalar_t                      gamma,
                                                                                     const scalar_t                      beta,
                                                                                     const scalar_t *const SFEM_RESTRICT adjugate,
                                                                                     const scalar_t jacobian_determinant,
                                                                                     const scalar_t qx,
                                                                                     const scalar_t qy,
                                                                                     const scalar_t qz,
                                                                                     const scalar_t qw,
                                                                                     accumulator_t *const SFEM_RESTRICT
                                                                                             element_matrix) {
    // mundane ops: 821 divs: 3 sqrts: 0
    // total ops: 845
    const scalar_t x0   = K + 0.16666666666666669 * k;
    const scalar_t x1   = qy - 1;
    const scalar_t x2   = qz - 1;
    const scalar_t x3   = x1 * x2;
    const scalar_t x4   = adjugate[0] * x3;
    const scalar_t x5   = qx - 1;
    const scalar_t x6   = x2 * x5;
    const scalar_t x7   = adjugate[3] * x6;
    const scalar_t x8   = x1 * x5;
    const scalar_t x9   = adjugate[6] * x8;
    const scalar_t x10  = x4 + x7 + x9;
    const scalar_t x11  = adjugate[1] * x3;
    const scalar_t x12  = adjugate[4] * x6;
    const scalar_t x13  = adjugate[7] * x8;
    const scalar_t x14  = x11 + x12 + x13;
    const scalar_t x15  = -x1;
    const scalar_t x16  = -x2;
    const scalar_t x17  = x15 * x16;
    const scalar_t x18  = adjugate[0] * x17;
    const scalar_t x19  = -x5;
    const scalar_t x20  = x16 * x19;
    const scalar_t x21  = adjugate[3] * x20;
    const scalar_t x22  = x15 * x19;
    const scalar_t x23  = adjugate[6] * x22;
    const scalar_t x24  = x18 + x21 + x23;
    const scalar_t x25  = adjugate[1] * x17;
    const scalar_t x26  = adjugate[4] * x20;
    const scalar_t x27  = adjugate[7] * x22;
    const scalar_t x28  = x25 + x26 + x27;
    const scalar_t x29  = 1.0 / beta;
    const scalar_t x30  = 1.0 / dt;
    const scalar_t x31  = eta * gamma * x29 * x30;
    const scalar_t x32  = 0.16666666666666669 * x31;
    const scalar_t x33  = qw / jacobian_determinant;
    const scalar_t x34  = qx * x2;
    const scalar_t x35  = adjugate[4] * x34;
    const scalar_t x36  = qx * x1;
    const scalar_t x37  = adjugate[7] * x36;
    const scalar_t x38  = x11 + x35 + x37;
    const scalar_t x39  = (1.0 / 2.0) * k;
    const scalar_t x40  = x10 * x39;
    const scalar_t x41  = K - 0.33333333333333331 * k;
    const scalar_t x42  = adjugate[3] * x34;
    const scalar_t x43  = adjugate[6] * x36;
    const scalar_t x44  = x4 + x42 + x43;
    const scalar_t x45  = x41 * x44;
    const scalar_t x46  = qx * x16;
    const scalar_t x47  = qx * x15;
    const scalar_t x48  = adjugate[3] * x46 + adjugate[6] * x47 - x18;
    const scalar_t x49  = x28 * x48;
    const scalar_t x50  = adjugate[4] * x46 + adjugate[7] * x47 - x25;
    const scalar_t x51  = (1.0 / 2.0) * x31;
    const scalar_t x52  = qx * qy;
    const scalar_t x53  = adjugate[7] * x52;
    const scalar_t x54  = qy * x2;
    const scalar_t x55  = adjugate[1] * x54;
    const scalar_t x56  = x35 + x53 + x55;
    const scalar_t x57  = x24 * x56;
    const scalar_t x58  = adjugate[6] * x52;
    const scalar_t x59  = adjugate[0] * x54;
    const scalar_t x60  = x42 + x58 + x59;
    const scalar_t x61  = x28 * x60;
    const scalar_t x62  = qy * x5;
    const scalar_t x63  = adjugate[7] * x62;
    const scalar_t x64  = x12 + x55 + x63;
    const scalar_t x65  = adjugate[6] * x62;
    const scalar_t x66  = x59 + x65 + x7;
    const scalar_t x67  = x14 * x66;
    const scalar_t x68  = qy * x16;
    const scalar_t x69  = qy * x19;
    const scalar_t x70  = adjugate[0] * x68 + adjugate[6] * x69 - x21;
    const scalar_t x71  = 0.66666666666666663 * x28;
    const scalar_t x72  = adjugate[1] * x68 + adjugate[7] * x69 - x26;
    const scalar_t x73  = qz * x1;
    const scalar_t x74  = adjugate[1] * x73;
    const scalar_t x75  = qz * x5;
    const scalar_t x76  = adjugate[4] * x75;
    const scalar_t x77  = x13 + x74 + x76;
    const scalar_t x78  = adjugate[0] * x73;
    const scalar_t x79  = adjugate[3] * x75;
    const scalar_t x80  = x78 + x79 + x9;
    const scalar_t x81  = x14 * x80;
    const scalar_t x82  = qz * x15;
    const scalar_t x83  = qz * x19;
    const scalar_t x84  = adjugate[0] * x82 + adjugate[3] * x83 - x23;
    const scalar_t x85  = adjugate[1] * x82 + adjugate[4] * x83 - x27;
    const scalar_t x86  = qx * qz;
    const scalar_t x87  = adjugate[4] * x86;
    const scalar_t x88  = x37 + x74 + x87;
    const scalar_t x89  = x24 * x88;
    const scalar_t x90  = adjugate[3] * x86;
    const scalar_t x91  = x43 + x78 + x90;
    const scalar_t x92  = x28 * x91;
    const scalar_t x93  = qy * qz;
    const scalar_t x94  = adjugate[1] * x93;
    const scalar_t x95  = x53 + x87 + x94;
    const scalar_t x96  = x24 * x95;
    const scalar_t x97  = adjugate[0] * x93;
    const scalar_t x98  = x58 + x90 + x97;
    const scalar_t x99  = x28 * x98;
    const scalar_t x100 = x63 + x76 + x94;
    const scalar_t x101 = x100 * x24;
    const scalar_t x102 = x65 + x79 + x97;
    const scalar_t x103 = x102 * x28;
    const scalar_t x104 = x39 * x44;
    const scalar_t x105 = x10 * x41;
    const scalar_t x106 = x48 * x56;
    const scalar_t x107 = x50 * x60;
    const scalar_t x108 = x38 * x66;
    const scalar_t x109 = 0.66666666666666663 * x50;
    const scalar_t x110 = x38 * x80;
    const scalar_t x111 = x48 * x88;
    const scalar_t x112 = x50 * x91;
    const scalar_t x113 = x48 * x95;
    const scalar_t x114 = x50 * x98;
    const scalar_t x115 = x100 * x48;
    const scalar_t x116 = x102 * x50;
    const scalar_t x117 = x33 * (x0 + x32);
    const scalar_t x118 = x60 * x72;
    const scalar_t x119 = x56 * x70;
    const scalar_t x120 = x60 * x85;
    const scalar_t x121 = x56 * x84;
    const scalar_t x122 = x60 * x88;
    const scalar_t x123 = x56 * x91;
    const scalar_t x124 = x60 * x95;
    const scalar_t x125 = x56 * x98;
    const scalar_t x126 = x100 * x60;
    const scalar_t x127 = x102 * x56;
    const scalar_t x128 = 0.66666666666666663 * x64;
    const scalar_t x129 = x66 * x77;
    const scalar_t x130 = x64 * x80;
    const scalar_t x131 = x70 * x88;
    const scalar_t x132 = x72 * x91;
    const scalar_t x133 = x70 * x95;
    const scalar_t x134 = x72 * x98;
    const scalar_t x135 = x100 * x70;
    const scalar_t x136 = x102 * x72;
    const scalar_t x137 = x84 * x88;
    const scalar_t x138 = x85 * x91;
    const scalar_t x139 = x84 * x95;
    const scalar_t x140 = x85 * x98;
    const scalar_t x141 = x100 * x84;
    const scalar_t x142 = x102 * x85;
    const scalar_t x143 = x91 * x95;
    const scalar_t x144 = x88 * x98;
    const scalar_t x145 = x100 * x91;
    const scalar_t x146 = x102 * x88;
    const scalar_t x147 = x100 * x98;
    const scalar_t x148 = x102 * x95;
    element_matrix[0] += x33 * (x0 * x10 * x14 + x24 * x28 * x32);
    element_matrix[1] += -x33 * (x14 * x45 + x38 * x40 + x51 * (-x24 * x50 + 0.66666666666666663 * x49));
    element_matrix[2] += x33 * (x39 * x57 + x41 * x61 - x51 * (-x57 + 0.66666666666666663 * x61));
    element_matrix[3] += -x33 * (x40 * x64 + x41 * x67 + x51 * (-x24 * x72 + x70 * x71));
    element_matrix[4] += -x33 * (x40 * x77 + x41 * x81 + x51 * (-x24 * x85 + x71 * x84));
    element_matrix[5] += x33 * (x39 * x89 + x41 * x92 - x51 * (-x89 + 0.66666666666666663 * x92));
    element_matrix[6] += -x33 * (x39 * x96 + x41 * x99 + x51 * (x24 * x95 - 0.66666666666666663 * x99));
    element_matrix[7] += x33 * (x101 * x39 + x103 * x41 - x51 * (-x101 + 0.66666666666666663 * x103));
    element_matrix[8] += -x33 * (x104 * x14 + x105 * x38 + x51 * (0.66666666666666663 * x24 * x50 - x49));
    element_matrix[9] += x33 * (x0 * x38 * x44 + x32 * x48 * x50);
    element_matrix[10] += x33 * (x106 * x39 + x107 * x41 - x51 * (-x106 + 0.66666666666666663 * x107));
    element_matrix[11] += x33 * (x104 * x64 + x108 * x41 - x51 * (x109 * x70 - x48 * x72));
    element_matrix[12] += x33 * (x104 * x77 + x110 * x41 - x51 * (x109 * x84 - x48 * x85));
    element_matrix[13] += x33 * (x111 * x39 + x112 * x41 - x51 * (-x111 + 0.66666666666666663 * x112));
    element_matrix[14] += -x33 * (x113 * x39 + x114 * x41 + x51 * (-0.66666666666666663 * x114 + x48 * x95));
    element_matrix[15] += x33 * (x115 * x39 + x116 * x41 - x51 * (-x115 + 0.66666666666666663 * x116));
    element_matrix[16] += x33 * (x39 * x61 + x41 * x57 - x51 * (0.66666666666666663 * x24 * x56 - x61));
    element_matrix[17] += x33 * (x106 * x41 + x107 * x39 - x51 * (-x107 + 0.66666666666666663 * x48 * x56));
    element_matrix[18] += x117 * x56 * x60;
    element_matrix[19] += x33 * (x118 * x39 + x119 * x41 - x51 * (-x118 + 0.66666666666666663 * x119));
    element_matrix[20] += x33 * (x120 * x39 + x121 * x41 - x51 * (-x120 + 0.66666666666666663 * x56 * x84));
    element_matrix[21] += x33 * (x122 * x39 + x123 * x41 + x51 * (x122 - 0.66666666666666663 * x123));
    element_matrix[22] += -x33 * (x124 * x39 + x125 * x41 + x51 * (-0.66666666666666663 * x125 + x60 * x95));
    element_matrix[23] += x33 * (x126 * x39 + x127 * x41 - x51 * (-x126 + 0.66666666666666663 * x127));
    element_matrix[24] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (0.66666666666666663 * x10 * x64 - x67) - x105 * x64 - x39 * x67);
    element_matrix[25] += x33 * (x108 * x39 + x45 * x64 + x51 * (x108 - x128 * x44));
    element_matrix[26] += x33 * (x118 * x41 + x119 * x39 - x51 * (-x119 + 0.66666666666666663 * x60 * x72));
    element_matrix[27] += x33 * (x0 * x64 * x66 + x32 * x70 * x72);
    element_matrix[28] += x33 * (x129 * x39 + x130 * x41 + x51 * (-x128 * x80 + x129));
    element_matrix[29] += x33 * (x131 * x39 + x132 * x41 - x51 * (-x131 + 0.66666666666666663 * x72 * x91));
    element_matrix[30] += -x33 * (x133 * x39 + x134 * x41 + x51 * (-0.66666666666666663 * x134 + x70 * x95));
    element_matrix[31] += x33 * (x135 * x39 + x136 * x41 - x51 * (-x135 + 0.66666666666666663 * x136));
    element_matrix[32] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (0.66666666666666663 * x10 * x77 - x81) - x105 * x77 - x39 * x81);
    element_matrix[33] += x33 * (x110 * x39 + x45 * x77 + x51 * (x110 - 0.66666666666666663 * x44 * x77));
    element_matrix[34] += x33 * (x120 * x41 + x121 * x39 - x51 * (0.66666666666666663 * x120 - x121));
    element_matrix[35] += x33 * (x129 * x41 + x130 * x39 - x51 * (0.66666666666666663 * x70 * x85 - x72 * x84));
    element_matrix[36] += x33 * (x0 * x77 * x80 + x32 * x84 * x85);
    element_matrix[37] += x33 * (x137 * x39 + x138 * x41 - x51 * (-x137 + 0.66666666666666663 * x85 * x91));
    element_matrix[38] += -x33 * (x139 * x39 + x140 * x41 + x51 * (-0.66666666666666663 * x140 + x84 * x95));
    element_matrix[39] += x33 * (x141 * x39 + x142 * x41 - x51 * (-x141 + 0.66666666666666663 * x142));
    element_matrix[40] += x33 * (x39 * x92 + x41 * x89 - x51 * (0.66666666666666663 * x24 * x88 - x92));
    element_matrix[41] += x33 * (x111 * x41 + x112 * x39 - x51 * (-x112 + 0.66666666666666663 * x48 * x88));
    element_matrix[42] += x33 * (x122 * x41 + x123 * x39 - x51 * (0.66666666666666663 * x122 - x123));
    element_matrix[43] += x33 * (x131 * x41 + x132 * x39 - x51 * (0.66666666666666663 * x131 - x132));
    element_matrix[44] += x33 * (x137 * x41 + x138 * x39 - x51 * (0.66666666666666663 * x137 - x138));
    element_matrix[45] += x117 * x88 * x91;
    element_matrix[46] += -x33 * (x143 * x39 + x144 * x41 + x51 * (-0.66666666666666663 * x144 + x91 * x95));
    element_matrix[47] += x33 * (x145 * x39 + x146 * x41 - x51 * (-x145 + 0.66666666666666663 * x146));
    element_matrix[48] += -x33 * (x39 * x99 + x41 * x96 + x51 * (-0.66666666666666663 * x96 + x99));
    element_matrix[49] += -x33 * (x113 * x41 + x114 * x39 + x51 * (-0.66666666666666663 * x113 + x114));
    element_matrix[50] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (-x125 + 0.66666666666666663 * x60 * x95) - x124 * x41 - x125 * x39);
    element_matrix[51] += -x33 * (x133 * x41 + x134 * x39 + x51 * (-0.66666666666666663 * x133 + x134));
    element_matrix[52] += -x33 * (x139 * x41 + x140 * x39 + x51 * (-0.66666666666666663 * x139 + x140));
    element_matrix[53] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (-x144 + 0.66666666666666663 * x91 * x95) - x143 * x41 - x144 * x39);
    element_matrix[54] += x117 * x95 * x98;
    element_matrix[55] += -x33 * (x147 * x39 + x148 * x41 + x51 * (x147 - 0.66666666666666663 * x148));
    element_matrix[56] += x33 * (x101 * x41 + x103 * x39 - x51 * (0.66666666666666663 * x100 * x24 - x103));
    element_matrix[57] += x33 * (x115 * x41 + x116 * x39 - x51 * (0.66666666666666663 * x100 * x48 - x116));
    element_matrix[58] += x33 * (x126 * x41 + x127 * x39 + x51 * (-0.66666666666666663 * x126 + x127));
    element_matrix[59] += x33 * (x135 * x41 + x136 * x39 - x51 * (0.66666666666666663 * x100 * x70 - x136));
    element_matrix[60] += x33 * (x141 * x41 + x142 * x39 - x51 * (0.66666666666666663 * x100 * x84 - x142));
    element_matrix[61] += x33 * (x145 * x41 + x146 * x39 + x51 * (-0.66666666666666663 * x145 + x146));
    element_matrix[62] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (0.66666666666666663 * x147 - x148) - x147 * x41 - x148 * x39);
    element_matrix[63] += x100 * x102 * x117;
}

//--------------------------
// hessian block_1_1
//--------------------------

template <typename scalar_t, typename accumulator_t>
static inline __host__ __device__ void cu_hex8_kelvin_voigt_newmark_matrix_block_1_1(const scalar_t                      k,
                                                                                     const scalar_t                      K,
                                                                                     const scalar_t                      eta,
                                                                                     const scalar_t                      rho,
                                                                                     const scalar_t                      dt,
                                                                                     const scalar_t                      gamma,
                                                                                     const scalar_t                      beta,
                                                                                     const scalar_t *const SFEM_RESTRICT adjugate,
                                                                                     const scalar_t jacobian_determinant,
                                                                                     const scalar_t qx,
                                                                                     const scalar_t qy,
                                                                                     const scalar_t qz,
                                                                                     const scalar_t qw,
                                                                                     accumulator_t *const SFEM_RESTRICT
                                                                                             element_matrix) {
    // mundane ops: 931 divs: 3 sqrts: 0
    // total ops: 955
    const scalar_t x0   = qx - 1;
    const scalar_t x1   = POW2(x0);
    const scalar_t x2   = qy - 1;
    const scalar_t x3   = POW2(x2);
    const scalar_t x4   = qz - 1;
    const scalar_t x5   = 1.0 / beta;
    const scalar_t x6   = jacobian_determinant * rho * x5 / POW2(dt);
    const scalar_t x7   = POW2(x4) * x6;
    const scalar_t x8   = x3 * x7;
    const scalar_t x9   = x2 * x4;
    const scalar_t x10  = adjugate[1] * x9;
    const scalar_t x11  = x0 * x4;
    const scalar_t x12  = adjugate[4] * x11;
    const scalar_t x13  = x0 * x2;
    const scalar_t x14  = adjugate[7] * x13;
    const scalar_t x15  = x10 + x12 + x14;
    const scalar_t x16  = POW2(x15);
    const scalar_t x17  = adjugate[0] * x9;
    const scalar_t x18  = adjugate[3] * x11;
    const scalar_t x19  = adjugate[6] * x13;
    const scalar_t x20  = x17 + x18 + x19;
    const scalar_t x21  = adjugate[2] * x9;
    const scalar_t x22  = adjugate[5] * x11;
    const scalar_t x23  = adjugate[8] * x13;
    const scalar_t x24  = x21 + x22 + x23;
    const scalar_t x25  = POW2(x20) + POW2(x24);
    const scalar_t x26  = 1.0 / dt;
    const scalar_t x27  = 1.0 / jacobian_determinant;
    const scalar_t x28  = (1.0 / 2.0) * x27;
    const scalar_t x29  = eta * gamma * x26 * x28 * x5;
    const scalar_t x30  = K - 0.33333333333333331 * k;
    const scalar_t x31  = 2 * x16;
    const scalar_t x32  = qx * x0;
    const scalar_t x33  = qx * x4;
    const scalar_t x34  = adjugate[4] * x33;
    const scalar_t x35  = qx * x2;
    const scalar_t x36  = adjugate[7] * x35;
    const scalar_t x37  = x10 + x34 + x36;
    const scalar_t x38  = x15 * x37;
    const scalar_t x39  = adjugate[3] * x33;
    const scalar_t x40  = adjugate[6] * x35;
    const scalar_t x41  = x17 + x39 + x40;
    const scalar_t x42  = adjugate[5] * x33;
    const scalar_t x43  = adjugate[8] * x35;
    const scalar_t x44  = x21 + x42 + x43;
    const scalar_t x45  = x20 * x41 + x24 * x44;
    const scalar_t x46  = 2 * x38;
    const scalar_t x47  = qw * ((1.0 / 2.0) * eta * gamma * x26 * x27 * x5 * (-1.3333333333333335 * x38 - x45) -
                               x28 * (k * (x45 + x46) + x30 * x46) - x32 * x8);
    const scalar_t x48  = qx * qy;
    const scalar_t x49  = x13 * x48;
    const scalar_t x50  = x49 * x7;
    const scalar_t x51  = adjugate[7] * x48;
    const scalar_t x52  = qy * x4;
    const scalar_t x53  = adjugate[1] * x52;
    const scalar_t x54  = x34 + x51 + x53;
    const scalar_t x55  = -x2;
    const scalar_t x56  = -x4;
    const scalar_t x57  = x55 * x56;
    const scalar_t x58  = adjugate[1] * x57;
    const scalar_t x59  = -x0;
    const scalar_t x60  = x56 * x59;
    const scalar_t x61  = adjugate[4] * x60;
    const scalar_t x62  = x55 * x59;
    const scalar_t x63  = adjugate[7] * x62;
    const scalar_t x64  = x58 + x61 + x63;
    const scalar_t x65  = x54 * x64;
    const scalar_t x66  = adjugate[6] * x48;
    const scalar_t x67  = adjugate[0] * x52;
    const scalar_t x68  = x39 + x66 + x67;
    const scalar_t x69  = adjugate[0] * x57;
    const scalar_t x70  = adjugate[3] * x60;
    const scalar_t x71  = adjugate[6] * x62;
    const scalar_t x72  = x69 + x70 + x71;
    const scalar_t x73  = adjugate[8] * x48;
    const scalar_t x74  = adjugate[2] * x52;
    const scalar_t x75  = x42 + x73 + x74;
    const scalar_t x76  = adjugate[2] * x57;
    const scalar_t x77  = adjugate[5] * x60;
    const scalar_t x78  = adjugate[8] * x62;
    const scalar_t x79  = x76 + x77 + x78;
    const scalar_t x80  = x68 * x72 + x75 * x79;
    const scalar_t x81  = 2 * x30;
    const scalar_t x82  = x15 * x81;
    const scalar_t x83  = qw * (x28 * (k * (2 * x65 + x80) + x54 * x82) - x29 * (-1.3333333333333335 * x65 - x80) + x50);
    const scalar_t x84  = qy * x2;
    const scalar_t x85  = x7 * x84;
    const scalar_t x86  = qy * x0;
    const scalar_t x87  = adjugate[7] * x86;
    const scalar_t x88  = x12 + x53 + x87;
    const scalar_t x89  = x15 * x88;
    const scalar_t x90  = adjugate[6] * x86;
    const scalar_t x91  = x18 + x67 + x90;
    const scalar_t x92  = adjugate[8] * x86;
    const scalar_t x93  = x22 + x74 + x92;
    const scalar_t x94  = x20 * x91 + x24 * x93;
    const scalar_t x95  = 2 * x89;
    const scalar_t x96  = qw * ((1.0 / 2.0) * eta * gamma * x26 * x27 * x5 * (-1.3333333333333335 * x89 - x94) - x1 * x85 -
                               x28 * (k * (x94 + x95) + x30 * x95));
    const scalar_t x97  = x3 * x6;
    const scalar_t x98  = qz * x4;
    const scalar_t x99  = x97 * x98;
    const scalar_t x100 = qz * x2;
    const scalar_t x101 = adjugate[1] * x100;
    const scalar_t x102 = qz * x0;
    const scalar_t x103 = adjugate[4] * x102;
    const scalar_t x104 = x101 + x103 + x14;
    const scalar_t x105 = x104 * x15;
    const scalar_t x106 = adjugate[0] * x100;
    const scalar_t x107 = adjugate[3] * x102;
    const scalar_t x108 = x106 + x107 + x19;
    const scalar_t x109 = adjugate[2] * x100;
    const scalar_t x110 = adjugate[5] * x102;
    const scalar_t x111 = x109 + x110 + x23;
    const scalar_t x112 = x108 * x20 + x111 * x24;
    const scalar_t x113 = 2 * x105;
    const scalar_t x114 = qw * ((1.0 / 2.0) * eta * gamma * x26 * x27 * x5 * (-1.3333333333333335 * x105 - x112) - x1 * x99 -
                                x28 * (k * (x112 + x113) + x113 * x30));
    const scalar_t x115 = qx * qz;
    const scalar_t x116 = x11 * x115;
    const scalar_t x117 = x116 * x97;
    const scalar_t x118 = adjugate[4] * x115;
    const scalar_t x119 = x101 + x118 + x36;
    const scalar_t x120 = x119 * x64;
    const scalar_t x121 = adjugate[3] * x115;
    const scalar_t x122 = x106 + x121 + x40;
    const scalar_t x123 = adjugate[5] * x115;
    const scalar_t x124 = x109 + x123 + x43;
    const scalar_t x125 = x122 * x72 + x124 * x79;
    const scalar_t x126 = qw * (x117 + x28 * (k * (2 * x120 + x125) + x119 * x82) - x29 * (-1.3333333333333335 * x120 - x125));
    const scalar_t x127 = x102 * x48 * x6 * x9;
    const scalar_t x128 = qy * qz;
    const scalar_t x129 = adjugate[1] * x128;
    const scalar_t x130 = x118 + x129 + x51;
    const scalar_t x131 = x130 * x64;
    const scalar_t x132 = adjugate[0] * x128;
    const scalar_t x133 = x121 + x132 + x66;
    const scalar_t x134 = adjugate[2] * x128;
    const scalar_t x135 = x123 + x134 + x73;
    const scalar_t x136 = x133 * x72 + x135 * x79;
    const scalar_t x137 = -qw * (x127 + x28 * (k * (2 * x131 + x136) + x130 * x82) + x29 * (1.3333333333333335 * x131 + x136));
    const scalar_t x138 = x1 * x6;
    const scalar_t x139 = x128 * x9;
    const scalar_t x140 = x138 * x139;
    const scalar_t x141 = x103 + x129 + x87;
    const scalar_t x142 = x141 * x64;
    const scalar_t x143 = x107 + x132 + x90;
    const scalar_t x144 = x110 + x134 + x92;
    const scalar_t x145 = x143 * x72 + x144 * x79;
    const scalar_t x146 = qw * (x140 + x28 * (k * (2 * x142 + x145) + x141 * x82) - x29 * (-1.3333333333333335 * x142 - x145));
    const scalar_t x147 = POW2(qx);
    const scalar_t x148 = POW2(x37);
    const scalar_t x149 = POW2(x41) + POW2(x44);
    const scalar_t x150 = 2 * x148;
    const scalar_t x151 = qx * x56;
    const scalar_t x152 = qx * x55;
    const scalar_t x153 = adjugate[4] * x151 + adjugate[7] * x152 - x58;
    const scalar_t x154 = x153 * x54;
    const scalar_t x155 = adjugate[3] * x151 + adjugate[6] * x152 - x69;
    const scalar_t x156 = adjugate[5] * x151 + adjugate[8] * x152 - x76;
    const scalar_t x157 = x155 * x68 + x156 * x75;
    const scalar_t x158 = x37 * x81;
    const scalar_t x159 = qw * (-x147 * x85 + (1.0 / 2.0) * x27 * (k * (2 * x154 + x157) - x158 * x54) -
                                x29 * (-1.3333333333333335 * x154 - x157));
    const scalar_t x160 = x37 * x88;
    const scalar_t x161 = x41 * x91 + x44 * x93;
    const scalar_t x162 = 2 * x160;
    const scalar_t x163 = qw * (x28 * (k * (x161 + x162) + x162 * x30) + x29 * (1.3333333333333335 * x160 + x161) + x50);
    const scalar_t x164 = x104 * x37;
    const scalar_t x165 = x108 * x41 + x111 * x44;
    const scalar_t x166 = 2 * x164;
    const scalar_t x167 = qw * (x117 + x28 * (k * (x165 + x166) + x166 * x30) + x29 * (1.3333333333333335 * x164 + x165));
    const scalar_t x168 = x119 * x153;
    const scalar_t x169 = x122 * x155 + x124 * x156;
    const scalar_t x170 = qw * (-x147 * x99 + (1.0 / 2.0) * x27 * (k * (2 * x168 + x169) - x119 * x158) -
                                x29 * (-1.3333333333333335 * x168 - x169));
    const scalar_t x171 = x147 * x6;
    const scalar_t x172 = x139 * x171;
    const scalar_t x173 = x130 * x153;
    const scalar_t x174 = x133 * x155 + x135 * x156;
    const scalar_t x175 =
            qw * (x172 + x28 * (-k * (2 * x173 + x174) + 2 * x130 * x30 * x37) - x29 * (1.3333333333333335 * x173 + x174));
    const scalar_t x176 = x141 * x153;
    const scalar_t x177 = x143 * x155 + x144 * x156;
    const scalar_t x178 =
            qw * (-x127 + (1.0 / 2.0) * x27 * (k * (2 * x176 + x177) - x141 * x158) - x29 * (-1.3333333333333335 * x176 - x177));
    const scalar_t x179 = POW2(qy);
    const scalar_t x180 = x179 * x7;
    const scalar_t x181 = POW2(x54);
    const scalar_t x182 = POW2(x68) + POW2(x75);
    const scalar_t x183 = 2 * x181;
    const scalar_t x184 = qy * x56;
    const scalar_t x185 = qy * x59;
    const scalar_t x186 = adjugate[1] * x184 + adjugate[7] * x185 - x61;
    const scalar_t x187 = x186 * x54;
    const scalar_t x188 = adjugate[0] * x184 + adjugate[6] * x185 - x70;
    const scalar_t x189 = adjugate[2] * x184 + adjugate[8] * x185 - x77;
    const scalar_t x190 = x188 * x68 + x189 * x75;
    const scalar_t x191 = x54 * x81;
    const scalar_t x192 = qw * (-x180 * x32 + (1.0 / 2.0) * x27 * (k * (2 * x187 + x190) - x191 * x88) -
                                x29 * (-1.3333333333333335 * x187 - x190));
    const scalar_t x193 = qz * x55;
    const scalar_t x194 = qz * x59;
    const scalar_t x195 = adjugate[1] * x193 + adjugate[4] * x194 - x63;
    const scalar_t x196 = x195 * x54;
    const scalar_t x197 = adjugate[0] * x193 + adjugate[3] * x194 - x71;
    const scalar_t x198 = adjugate[2] * x193 + adjugate[5] * x194 - x78;
    const scalar_t x199 = x197 * x68 + x198 * x75;
    const scalar_t x200 =
            qw * (-x127 + (1.0 / 2.0) * x27 * (k * (2 * x196 + x199) - x104 * x191) - x29 * (-1.3333333333333335 * x196 - x199));
    const scalar_t x201 = x119 * x54;
    const scalar_t x202 = x122 * x68 + x124 * x75;
    const scalar_t x203 = 2 * x201;
    const scalar_t x204 = qw * (x172 + x28 * (k * (x202 + x203) + x203 * x30) + x29 * (1.3333333333333335 * x201 + x202));
    const scalar_t x205 = x179 * x98;
    const scalar_t x206 = x130 * x54;
    const scalar_t x207 = x133 * x68 + x135 * x75;
    const scalar_t x208 = 2 * x206;
    const scalar_t x209 = qw * ((1.0 / 2.0) * eta * gamma * x26 * x27 * x5 * (-1.3333333333333335 * x206 - x207) - x171 * x205 -
                                x28 * (k * (x207 + x208) + x208 * x30));
    const scalar_t x210 = x116 * x179 * x6;
    const scalar_t x211 = x141 * x54;
    const scalar_t x212 = x143 * x68 + x144 * x75;
    const scalar_t x213 = 2 * x211;
    const scalar_t x214 = qw * (x210 + x28 * (k * (x212 + x213) + x213 * x30) + x29 * (1.3333333333333335 * x211 + x212));
    const scalar_t x215 = POW2(x88);
    const scalar_t x216 = POW2(x91) + POW2(x93);
    const scalar_t x217 = 2 * x215;
    const scalar_t x218 = x104 * x88;
    const scalar_t x219 = x108 * x91 + x111 * x93;
    const scalar_t x220 = 2 * x218;
    const scalar_t x221 = qw * (x140 + x28 * (k * (x219 + x220) + x220 * x30) + x29 * (1.3333333333333335 * x218 + x219));
    const scalar_t x222 = x119 * x186;
    const scalar_t x223 = x122 * x188 + x124 * x189;
    const scalar_t x224 = x81 * x88;
    const scalar_t x225 =
            qw * (-x127 + (1.0 / 2.0) * x27 * (k * (2 * x222 + x223) - x119 * x224) - x29 * (-1.3333333333333335 * x222 - x223));
    const scalar_t x226 = x130 * x186;
    const scalar_t x227 = x133 * x188 + x135 * x189;
    const scalar_t x228 =
            qw * (x210 + x28 * (-k * (2 * x226 + x227) + 2 * x130 * x30 * x88) - x29 * (1.3333333333333335 * x226 + x227));
    const scalar_t x229 = x141 * x186;
    const scalar_t x230 = x143 * x188 + x144 * x189;
    const scalar_t x231 = qw * (-x138 * x205 + (1.0 / 2.0) * x27 * (k * (2 * x229 + x230) - x141 * x224) -
                                x29 * (-1.3333333333333335 * x229 - x230));
    const scalar_t x232 = POW2(qz) * x6;
    const scalar_t x233 = x232 * x3;
    const scalar_t x234 = POW2(x104);
    const scalar_t x235 = POW2(x108) + POW2(x111);
    const scalar_t x236 = 2 * x234;
    const scalar_t x237 = x119 * x195;
    const scalar_t x238 = x122 * x197 + x124 * x198;
    const scalar_t x239 = x104 * x81;
    const scalar_t x240 = qw * (-x233 * x32 + (1.0 / 2.0) * x27 * (k * (2 * x237 + x238) - x119 * x239) -
                                x29 * (-1.3333333333333335 * x237 - x238));
    const scalar_t x241 = x232 * x49;
    const scalar_t x242 = x130 * x195;
    const scalar_t x243 = x133 * x197 + x135 * x198;
    const scalar_t x244 =
            qw * (x241 + x28 * (-k * (2 * x242 + x243) + 2 * x104 * x130 * x30) - x29 * (1.3333333333333335 * x242 + x243));
    const scalar_t x245 = x232 * x84;
    const scalar_t x246 = x141 * x195;
    const scalar_t x247 = x143 * x197 + x144 * x198;
    const scalar_t x248 = qw * (-x1 * x245 + (1.0 / 2.0) * x27 * (k * (2 * x246 + x247) - x141 * x239) -
                                x29 * (-1.3333333333333335 * x246 - x247));
    const scalar_t x249 = POW2(x119);
    const scalar_t x250 = POW2(x122) + POW2(x124);
    const scalar_t x251 = 2 * x249;
    const scalar_t x252 = x119 * x130;
    const scalar_t x253 = x122 * x133 + x124 * x135;
    const scalar_t x254 = 2 * x252;
    const scalar_t x255 = qw * ((1.0 / 2.0) * eta * gamma * x26 * x27 * x5 * (-1.3333333333333335 * x252 - x253) - x147 * x245 -
                                x28 * (k * (x253 + x254) + x254 * x30));
    const scalar_t x256 = x119 * x141;
    const scalar_t x257 = x122 * x143 + x124 * x144;
    const scalar_t x258 = 2 * x256;
    const scalar_t x259 = qw * (x241 + x28 * (k * (x257 + x258) + x258 * x30) + x29 * (1.3333333333333335 * x256 + x257));
    const scalar_t x260 = x179 * x232;
    const scalar_t x261 = POW2(x130);
    const scalar_t x262 = POW2(x133) + POW2(x135);
    const scalar_t x263 = 2 * x261;
    const scalar_t x264 = x130 * x141;
    const scalar_t x265 = x133 * x143 + x135 * x144;
    const scalar_t x266 = 2 * x264;
    const scalar_t x267 = qw * ((1.0 / 2.0) * eta * gamma * x26 * x27 * x5 * (-1.3333333333333335 * x264 - x265) - x260 * x32 -
                                x28 * (k * (x265 + x266) + x266 * x30));
    const scalar_t x268 = POW2(x141);
    const scalar_t x269 = POW2(x143) + POW2(x144);
    const scalar_t x270 = 2 * x268;
    element_matrix[0] += qw * (x1 * x8 + x28 * (k * (x25 + x31) + x30 * x31) + x29 * (1.3333333333333335 * x16 + x25));
    element_matrix[1] += x47;
    element_matrix[2] += x83;
    element_matrix[3] += x96;
    element_matrix[4] += x114;
    element_matrix[5] += x126;
    element_matrix[6] += x137;
    element_matrix[7] += x146;
    element_matrix[8] += x47;
    element_matrix[9] += qw * (x147 * x8 + x28 * (k * (x149 + x150) + x150 * x30) + x29 * (1.3333333333333335 * x148 + x149));
    element_matrix[10] += x159;
    element_matrix[11] += x163;
    element_matrix[12] += x167;
    element_matrix[13] += x170;
    element_matrix[14] += x175;
    element_matrix[15] += x178;
    element_matrix[16] += x83;
    element_matrix[17] += x159;
    element_matrix[18] += qw * (x147 * x180 + x28 * (k * (x182 + x183) + x183 * x30) + x29 * (1.3333333333333335 * x181 + x182));
    element_matrix[19] += x192;
    element_matrix[20] += x200;
    element_matrix[21] += x204;
    element_matrix[22] += x209;
    element_matrix[23] += x214;
    element_matrix[24] += x96;
    element_matrix[25] += x163;
    element_matrix[26] += x192;
    element_matrix[27] += qw * (x1 * x180 + x28 * (k * (x216 + x217) + x217 * x30) + x29 * (1.3333333333333335 * x215 + x216));
    element_matrix[28] += x221;
    element_matrix[29] += x225;
    element_matrix[30] += x228;
    element_matrix[31] += x231;
    element_matrix[32] += x114;
    element_matrix[33] += x167;
    element_matrix[34] += x200;
    element_matrix[35] += x221;
    element_matrix[36] += qw * (x1 * x233 + x28 * (k * (x235 + x236) + x236 * x30) + x29 * (1.3333333333333335 * x234 + x235));
    element_matrix[37] += x240;
    element_matrix[38] += x244;
    element_matrix[39] += x248;
    element_matrix[40] += x126;
    element_matrix[41] += x170;
    element_matrix[42] += x204;
    element_matrix[43] += x225;
    element_matrix[44] += x240;
    element_matrix[45] += qw * (x147 * x233 + x28 * (k * (x250 + x251) + x251 * x30) + x29 * (1.3333333333333335 * x249 + x250));
    element_matrix[46] += x255;
    element_matrix[47] += x259;
    element_matrix[48] += x137;
    element_matrix[49] += x175;
    element_matrix[50] += x209;
    element_matrix[51] += x228;
    element_matrix[52] += x244;
    element_matrix[53] += x255;
    element_matrix[54] += qw * (x147 * x260 + x28 * (k * (x262 + x263) + x263 * x30) + x29 * (1.3333333333333335 * x261 + x262));
    element_matrix[55] += x267;
    element_matrix[56] += x146;
    element_matrix[57] += x178;
    element_matrix[58] += x214;
    element_matrix[59] += x231;
    element_matrix[60] += x248;
    element_matrix[61] += x259;
    element_matrix[62] += x267;
    element_matrix[63] += qw * (x1 * x260 + x28 * (k * (x269 + x270) + x270 * x30) + x29 * (1.3333333333333335 * x268 + x269));
}

//--------------------------
// hessian block_1_2
//--------------------------

template <typename scalar_t, typename accumulator_t>
static inline __host__ __device__ void cu_hex8_kelvin_voigt_newmark_matrix_block_1_2(const scalar_t                      k,
                                                                                     const scalar_t                      K,
                                                                                     const scalar_t                      eta,
                                                                                     const scalar_t                      rho,
                                                                                     const scalar_t                      dt,
                                                                                     const scalar_t                      gamma,
                                                                                     const scalar_t                      beta,
                                                                                     const scalar_t *const SFEM_RESTRICT adjugate,
                                                                                     const scalar_t jacobian_determinant,
                                                                                     const scalar_t qx,
                                                                                     const scalar_t qy,
                                                                                     const scalar_t qz,
                                                                                     const scalar_t qw,
                                                                                     accumulator_t *const SFEM_RESTRICT
                                                                                             element_matrix) {
    // mundane ops: 819 divs: 3 sqrts: 0
    // total ops: 843
    const scalar_t x0   = K + 0.16666666666666669 * k;
    const scalar_t x1   = qy - 1;
    const scalar_t x2   = qz - 1;
    const scalar_t x3   = x1 * x2;
    const scalar_t x4   = adjugate[1] * x3;
    const scalar_t x5   = qx - 1;
    const scalar_t x6   = x2 * x5;
    const scalar_t x7   = adjugate[4] * x6;
    const scalar_t x8   = x1 * x5;
    const scalar_t x9   = adjugate[7] * x8;
    const scalar_t x10  = x4 + x7 + x9;
    const scalar_t x11  = adjugate[2] * x3;
    const scalar_t x12  = adjugate[5] * x6;
    const scalar_t x13  = adjugate[8] * x8;
    const scalar_t x14  = x11 + x12 + x13;
    const scalar_t x15  = -x1;
    const scalar_t x16  = -x2;
    const scalar_t x17  = x15 * x16;
    const scalar_t x18  = adjugate[2] * x17;
    const scalar_t x19  = -x5;
    const scalar_t x20  = x16 * x19;
    const scalar_t x21  = adjugate[5] * x20;
    const scalar_t x22  = x15 * x19;
    const scalar_t x23  = adjugate[8] * x22;
    const scalar_t x24  = x18 + x21 + x23;
    const scalar_t x25  = adjugate[1] * x17;
    const scalar_t x26  = adjugate[4] * x20;
    const scalar_t x27  = adjugate[7] * x22;
    const scalar_t x28  = x25 + x26 + x27;
    const scalar_t x29  = 1.0 / beta;
    const scalar_t x30  = 1.0 / dt;
    const scalar_t x31  = eta * gamma * x29 * x30;
    const scalar_t x32  = 0.16666666666666669 * x31;
    const scalar_t x33  = qw / jacobian_determinant;
    const scalar_t x34  = qx * x2;
    const scalar_t x35  = adjugate[4] * x34;
    const scalar_t x36  = qx * x1;
    const scalar_t x37  = adjugate[7] * x36;
    const scalar_t x38  = x35 + x37 + x4;
    const scalar_t x39  = (1.0 / 2.0) * k;
    const scalar_t x40  = x38 * x39;
    const scalar_t x41  = adjugate[5] * x34;
    const scalar_t x42  = adjugate[8] * x36;
    const scalar_t x43  = x11 + x41 + x42;
    const scalar_t x44  = K - 0.33333333333333331 * k;
    const scalar_t x45  = x10 * x44;
    const scalar_t x46  = qx * x16;
    const scalar_t x47  = qx * x15;
    const scalar_t x48  = adjugate[4] * x46 + adjugate[7] * x47 - x25;
    const scalar_t x49  = x24 * x48;
    const scalar_t x50  = adjugate[5] * x46 + adjugate[8] * x47 - x18;
    const scalar_t x51  = (1.0 / 2.0) * x31;
    const scalar_t x52  = qx * qy;
    const scalar_t x53  = adjugate[7] * x52;
    const scalar_t x54  = qy * x2;
    const scalar_t x55  = adjugate[1] * x54;
    const scalar_t x56  = x35 + x53 + x55;
    const scalar_t x57  = x24 * x56;
    const scalar_t x58  = adjugate[8] * x52;
    const scalar_t x59  = adjugate[2] * x54;
    const scalar_t x60  = x41 + x58 + x59;
    const scalar_t x61  = x28 * x60;
    const scalar_t x62  = qy * x5;
    const scalar_t x63  = adjugate[7] * x62;
    const scalar_t x64  = x55 + x63 + x7;
    const scalar_t x65  = x14 * x64;
    const scalar_t x66  = adjugate[8] * x62;
    const scalar_t x67  = x12 + x59 + x66;
    const scalar_t x68  = qz * x1;
    const scalar_t x69  = adjugate[1] * x68;
    const scalar_t x70  = qz * x5;
    const scalar_t x71  = adjugate[4] * x70;
    const scalar_t x72  = x69 + x71 + x9;
    const scalar_t x73  = x14 * x72;
    const scalar_t x74  = adjugate[2] * x68;
    const scalar_t x75  = adjugate[5] * x70;
    const scalar_t x76  = x13 + x74 + x75;
    const scalar_t x77  = qx * qz;
    const scalar_t x78  = adjugate[4] * x77;
    const scalar_t x79  = x37 + x69 + x78;
    const scalar_t x80  = x24 * x79;
    const scalar_t x81  = adjugate[5] * x77;
    const scalar_t x82  = x42 + x74 + x81;
    const scalar_t x83  = x28 * x82;
    const scalar_t x84  = qy * qz;
    const scalar_t x85  = adjugate[1] * x84;
    const scalar_t x86  = x53 + x78 + x85;
    const scalar_t x87  = x24 * x86;
    const scalar_t x88  = adjugate[2] * x84;
    const scalar_t x89  = x58 + x81 + x88;
    const scalar_t x90  = x28 * x89;
    const scalar_t x91  = x63 + x71 + x85;
    const scalar_t x92  = x24 * x91;
    const scalar_t x93  = x66 + x75 + x88;
    const scalar_t x94  = x28 * x93;
    const scalar_t x95  = x10 * x39;
    const scalar_t x96  = x38 * x44;
    const scalar_t x97  = x50 * x56;
    const scalar_t x98  = x48 * x60;
    const scalar_t x99  = x43 * x64;
    const scalar_t x100 = 0.66666666666666663 * x38;
    const scalar_t x101 = x43 * x72;
    const scalar_t x102 = x50 * x79;
    const scalar_t x103 = x48 * x82;
    const scalar_t x104 = x50 * x86;
    const scalar_t x105 = x48 * x89;
    const scalar_t x106 = x50 * x91;
    const scalar_t x107 = x48 * x93;
    const scalar_t x108 = x33 * (x0 + x32);
    const scalar_t x109 = qy * x16;
    const scalar_t x110 = qy * x19;
    const scalar_t x111 = adjugate[1] * x109 + adjugate[7] * x110 - x26;
    const scalar_t x112 = x111 * x60;
    const scalar_t x113 = adjugate[2] * x109 + adjugate[8] * x110 - x21;
    const scalar_t x114 = x113 * x56;
    const scalar_t x115 = qz * x15;
    const scalar_t x116 = qz * x19;
    const scalar_t x117 = adjugate[1] * x115 + adjugate[4] * x116 - x27;
    const scalar_t x118 = x117 * x60;
    const scalar_t x119 = adjugate[2] * x115 + adjugate[5] * x116 - x23;
    const scalar_t x120 = x119 * x56;
    const scalar_t x121 = x60 * x79;
    const scalar_t x122 = x56 * x82;
    const scalar_t x123 = x60 * x86;
    const scalar_t x124 = x56 * x89;
    const scalar_t x125 = x60 * x91;
    const scalar_t x126 = x56 * x93;
    const scalar_t x127 = 0.66666666666666663 * x111;
    const scalar_t x128 = x67 * x72;
    const scalar_t x129 = x64 * x76;
    const scalar_t x130 = x113 * x79;
    const scalar_t x131 = x111 * x82;
    const scalar_t x132 = x113 * x86;
    const scalar_t x133 = x111 * x89;
    const scalar_t x134 = x113 * x91;
    const scalar_t x135 = x111 * x93;
    const scalar_t x136 = 0.66666666666666663 * x117;
    const scalar_t x137 = x119 * x79;
    const scalar_t x138 = x117 * x82;
    const scalar_t x139 = x119 * x86;
    const scalar_t x140 = x117 * x89;
    const scalar_t x141 = x119 * x91;
    const scalar_t x142 = x117 * x93;
    const scalar_t x143 = x82 * x86;
    const scalar_t x144 = x79 * x89;
    const scalar_t x145 = x82 * x91;
    const scalar_t x146 = x79 * x93;
    const scalar_t x147 = x89 * x91;
    const scalar_t x148 = x86 * x93;
    element_matrix[0] += x33 * (x0 * x10 * x14 + x24 * x28 * x32);
    element_matrix[1] += -x33 * (x14 * x40 + x43 * x45 + x51 * (0.66666666666666663 * x28 * x50 - x49));
    element_matrix[2] += x33 * (x39 * x57 + x44 * x61 - x51 * (0.66666666666666663 * x28 * x60 - x57));
    element_matrix[3] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (0.66666666666666663 * x10 * x67 - x65) - x39 * x65 - x45 * x67);
    element_matrix[4] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (0.66666666666666663 * x10 * x76 - x73) - x39 * x73 - x45 * x76);
    element_matrix[5] += x33 * (x39 * x80 + x44 * x83 - x51 * (0.66666666666666663 * x28 * x82 - x80));
    element_matrix[6] += -x33 * (x39 * x87 + x44 * x90 + x51 * (x87 - 0.66666666666666663 * x90));
    element_matrix[7] += x33 * (x39 * x92 + x44 * x94 - x51 * (0.66666666666666663 * x28 * x93 - x92));
    element_matrix[8] += -x33 * (x14 * x96 + x43 * x95 + x51 * (-x28 * x50 + 0.66666666666666663 * x49));
    element_matrix[9] += x33 * (x0 * x38 * x43 + x32 * x48 * x50);
    element_matrix[10] += x33 * (x39 * x97 + x44 * x98 - x51 * (0.66666666666666663 * x48 * x60 - x97));
    element_matrix[11] += x33 * (x39 * x99 + x51 * (-x100 * x67 + x99) + x67 * x96);
    element_matrix[12] += x33 * (x101 * x39 + x51 * (-x100 * x76 + x101) + x76 * x96);
    element_matrix[13] += x33 * (x102 * x39 + x103 * x44 - x51 * (-x102 + 0.66666666666666663 * x48 * x82));
    element_matrix[14] += -x33 * (x104 * x39 + x105 * x44 + x51 * (x104 - 0.66666666666666663 * x105));
    element_matrix[15] += x33 * (x106 * x39 + x107 * x44 - x51 * (-x106 + 0.66666666666666663 * x48 * x93));
    element_matrix[16] += x33 * (x39 * x61 + x44 * x57 - x51 * (0.66666666666666663 * x57 - x61));
    element_matrix[17] += x33 * (x39 * x98 + x44 * x97 - x51 * (0.66666666666666663 * x97 - x98));
    element_matrix[18] += x108 * x56 * x60;
    element_matrix[19] += x33 * (x112 * x39 + x114 * x44 - x51 * (-x112 + 0.66666666666666663 * x113 * x56));
    element_matrix[20] += x33 * (x118 * x39 + x120 * x44 - x51 * (-x118 + 0.66666666666666663 * x120));
    element_matrix[21] += x33 * (x121 * x39 + x122 * x44 - x51 * (-x121 + 0.66666666666666663 * x122));
    element_matrix[22] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (-x123 + 0.66666666666666663 * x56 * x89) - x123 * x39 - x124 * x44);
    element_matrix[23] += x33 * (x125 * x39 + x126 * x44 + x51 * (x125 - 0.66666666666666663 * x126));
    element_matrix[24] += -x33 * (x44 * x65 + x51 * (-x113 * x28 + x127 * x24) + x67 * x95);
    element_matrix[25] += x33 * (x40 * x67 + x44 * x99 - x51 * (-x113 * x48 + x127 * x50));
    element_matrix[26] += x33 * (x112 * x44 + x114 * x39 - x51 * (0.66666666666666663 * x112 - x114));
    element_matrix[27] += x33 * (x0 * x64 * x67 + x111 * x113 * x32);
    element_matrix[28] += x33 * (x128 * x39 + x129 * x44 - x51 * (-x113 * x117 + x119 * x127));
    element_matrix[29] += x33 * (x130 * x39 + x131 * x44 - x51 * (-x130 + 0.66666666666666663 * x131));
    element_matrix[30] += -x33 * (x132 * x39 + x133 * x44 + x51 * (x132 - 0.66666666666666663 * x133));
    element_matrix[31] += x33 * (x134 * x39 + x135 * x44 - x51 * (0.66666666666666663 * x111 * x93 - x134));
    element_matrix[32] += -x33 * (x44 * x73 + x51 * (-x119 * x28 + x136 * x24) + x76 * x95);
    element_matrix[33] += x33 * (x101 * x44 + x40 * x76 - x51 * (-x119 * x48 + x136 * x50));
    element_matrix[34] += x33 * (x118 * x44 + x120 * x39 - x51 * (0.66666666666666663 * x117 * x60 - x120));
    element_matrix[35] += x33 * (x128 * x44 + x129 * x39 + x51 * (-0.66666666666666663 * x128 + x129));
    element_matrix[36] += x33 * (x0 * x72 * x76 + x117 * x119 * x32);
    element_matrix[37] += x33 * (x137 * x39 + x138 * x44 - x51 * (-x137 + 0.66666666666666663 * x138));
    element_matrix[38] += -x33 * (x139 * x39 + x140 * x44 + x51 * (x139 - 0.66666666666666663 * x140));
    element_matrix[39] += x33 * (x141 * x39 + x142 * x44 - x51 * (0.66666666666666663 * x117 * x93 - x141));
    element_matrix[40] += x33 * (x39 * x83 + x44 * x80 - x51 * (0.66666666666666663 * x80 - x83));
    element_matrix[41] += x33 * (x102 * x44 + x103 * x39 - x51 * (0.66666666666666663 * x102 - x103));
    element_matrix[42] += x33 * (x121 * x44 + x122 * x39 + x51 * (-0.66666666666666663 * x121 + x122));
    element_matrix[43] += x33 * (x130 * x44 + x131 * x39 - x51 * (0.66666666666666663 * x113 * x79 - x131));
    element_matrix[44] += x33 * (x137 * x44 + x138 * x39 - x51 * (0.66666666666666663 * x119 * x79 - x138));
    element_matrix[45] += x108 * x79 * x82;
    element_matrix[46] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (-x143 + 0.66666666666666663 * x79 * x89) - x143 * x39 - x144 * x44);
    element_matrix[47] += x33 * (x145 * x39 + x146 * x44 + x51 * (x145 - 0.66666666666666663 * x146));
    element_matrix[48] += -x33 * (x39 * x90 + x44 * x87 + x51 * (x28 * x89 - 0.66666666666666663 * x87));
    element_matrix[49] += -x33 * (x104 * x44 + x105 * x39 + x51 * (-0.66666666666666663 * x104 + x48 * x89));
    element_matrix[50] += -x33 * (x123 * x44 + x124 * x39 + x51 * (-0.66666666666666663 * x123 + x56 * x89));
    element_matrix[51] += -x33 * (x132 * x44 + x133 * x39 + x51 * (x111 * x89 - 0.66666666666666663 * x132));
    element_matrix[52] += -x33 * (x139 * x44 + x140 * x39 + x51 * (x117 * x89 - 0.66666666666666663 * x139));
    element_matrix[53] += -x33 * (x143 * x44 + x144 * x39 + x51 * (-0.66666666666666663 * x143 + x79 * x89));
    element_matrix[54] += x108 * x86 * x89;
    element_matrix[55] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (-x147 + 0.66666666666666663 * x148) - x147 * x39 - x148 * x44);
    element_matrix[56] += x33 * (x39 * x94 + x44 * x92 - x51 * (0.66666666666666663 * x92 - x94));
    element_matrix[57] += x33 * (x106 * x44 + x107 * x39 - x51 * (0.66666666666666663 * x106 - x107));
    element_matrix[58] += x33 * (x125 * x44 + x126 * x39 - x51 * (0.66666666666666663 * x125 - x126));
    element_matrix[59] += x33 * (x134 * x44 + x135 * x39 - x51 * (0.66666666666666663 * x134 - x135));
    element_matrix[60] += x33 * (x141 * x44 + x142 * x39 - x51 * (0.66666666666666663 * x141 - x142));
    element_matrix[61] += x33 * (x145 * x44 + x146 * x39 - x51 * (0.66666666666666663 * x145 - x146));
    element_matrix[62] += -x33 * (x147 * x44 + x148 * x39 + x51 * (-0.66666666666666663 * x147 + x148));
    element_matrix[63] += x108 * x91 * x93;
}

//--------------------------
// hessian block_2_0
//--------------------------

template <typename scalar_t, typename accumulator_t>
static inline __host__ __device__ void cu_hex8_kelvin_voigt_newmark_matrix_block_2_0(const scalar_t                      k,
                                                                                     const scalar_t                      K,
                                                                                     const scalar_t                      eta,
                                                                                     const scalar_t                      rho,
                                                                                     const scalar_t                      dt,
                                                                                     const scalar_t                      gamma,
                                                                                     const scalar_t                      beta,
                                                                                     const scalar_t *const SFEM_RESTRICT adjugate,
                                                                                     const scalar_t jacobian_determinant,
                                                                                     const scalar_t qx,
                                                                                     const scalar_t qy,
                                                                                     const scalar_t qz,
                                                                                     const scalar_t qw,
                                                                                     accumulator_t *const SFEM_RESTRICT
                                                                                             element_matrix) {
    // mundane ops: 821 divs: 3 sqrts: 0
    // total ops: 845
    const scalar_t x0   = K + 0.16666666666666669 * k;
    const scalar_t x1   = qy - 1;
    const scalar_t x2   = qz - 1;
    const scalar_t x3   = x1 * x2;
    const scalar_t x4   = adjugate[0] * x3;
    const scalar_t x5   = qx - 1;
    const scalar_t x6   = x2 * x5;
    const scalar_t x7   = adjugate[3] * x6;
    const scalar_t x8   = x1 * x5;
    const scalar_t x9   = adjugate[6] * x8;
    const scalar_t x10  = x4 + x7 + x9;
    const scalar_t x11  = adjugate[2] * x3;
    const scalar_t x12  = adjugate[5] * x6;
    const scalar_t x13  = adjugate[8] * x8;
    const scalar_t x14  = x11 + x12 + x13;
    const scalar_t x15  = -x1;
    const scalar_t x16  = -x2;
    const scalar_t x17  = x15 * x16;
    const scalar_t x18  = adjugate[0] * x17;
    const scalar_t x19  = -x5;
    const scalar_t x20  = x16 * x19;
    const scalar_t x21  = adjugate[3] * x20;
    const scalar_t x22  = x15 * x19;
    const scalar_t x23  = adjugate[6] * x22;
    const scalar_t x24  = x18 + x21 + x23;
    const scalar_t x25  = adjugate[2] * x17;
    const scalar_t x26  = adjugate[5] * x20;
    const scalar_t x27  = adjugate[8] * x22;
    const scalar_t x28  = x25 + x26 + x27;
    const scalar_t x29  = 1.0 / beta;
    const scalar_t x30  = 1.0 / dt;
    const scalar_t x31  = eta * gamma * x29 * x30;
    const scalar_t x32  = 0.16666666666666669 * x31;
    const scalar_t x33  = qw / jacobian_determinant;
    const scalar_t x34  = qx * x2;
    const scalar_t x35  = adjugate[5] * x34;
    const scalar_t x36  = qx * x1;
    const scalar_t x37  = adjugate[8] * x36;
    const scalar_t x38  = x11 + x35 + x37;
    const scalar_t x39  = (1.0 / 2.0) * k;
    const scalar_t x40  = x10 * x39;
    const scalar_t x41  = K - 0.33333333333333331 * k;
    const scalar_t x42  = adjugate[3] * x34;
    const scalar_t x43  = adjugate[6] * x36;
    const scalar_t x44  = x4 + x42 + x43;
    const scalar_t x45  = x41 * x44;
    const scalar_t x46  = qx * x16;
    const scalar_t x47  = qx * x15;
    const scalar_t x48  = adjugate[3] * x46 + adjugate[6] * x47 - x18;
    const scalar_t x49  = x28 * x48;
    const scalar_t x50  = adjugate[5] * x46 + adjugate[8] * x47 - x25;
    const scalar_t x51  = (1.0 / 2.0) * x31;
    const scalar_t x52  = qx * qy;
    const scalar_t x53  = adjugate[8] * x52;
    const scalar_t x54  = qy * x2;
    const scalar_t x55  = adjugate[2] * x54;
    const scalar_t x56  = x35 + x53 + x55;
    const scalar_t x57  = x24 * x56;
    const scalar_t x58  = adjugate[6] * x52;
    const scalar_t x59  = adjugate[0] * x54;
    const scalar_t x60  = x42 + x58 + x59;
    const scalar_t x61  = x28 * x60;
    const scalar_t x62  = qy * x5;
    const scalar_t x63  = adjugate[8] * x62;
    const scalar_t x64  = x12 + x55 + x63;
    const scalar_t x65  = adjugate[6] * x62;
    const scalar_t x66  = x59 + x65 + x7;
    const scalar_t x67  = x14 * x66;
    const scalar_t x68  = qy * x16;
    const scalar_t x69  = qy * x19;
    const scalar_t x70  = adjugate[0] * x68 + adjugate[6] * x69 - x21;
    const scalar_t x71  = 0.66666666666666663 * x28;
    const scalar_t x72  = adjugate[2] * x68 + adjugate[8] * x69 - x26;
    const scalar_t x73  = qz * x1;
    const scalar_t x74  = adjugate[2] * x73;
    const scalar_t x75  = qz * x5;
    const scalar_t x76  = adjugate[5] * x75;
    const scalar_t x77  = x13 + x74 + x76;
    const scalar_t x78  = adjugate[0] * x73;
    const scalar_t x79  = adjugate[3] * x75;
    const scalar_t x80  = x78 + x79 + x9;
    const scalar_t x81  = x14 * x80;
    const scalar_t x82  = qz * x15;
    const scalar_t x83  = qz * x19;
    const scalar_t x84  = adjugate[0] * x82 + adjugate[3] * x83 - x23;
    const scalar_t x85  = adjugate[2] * x82 + adjugate[5] * x83 - x27;
    const scalar_t x86  = qx * qz;
    const scalar_t x87  = adjugate[5] * x86;
    const scalar_t x88  = x37 + x74 + x87;
    const scalar_t x89  = x24 * x88;
    const scalar_t x90  = adjugate[3] * x86;
    const scalar_t x91  = x43 + x78 + x90;
    const scalar_t x92  = x28 * x91;
    const scalar_t x93  = qy * qz;
    const scalar_t x94  = adjugate[2] * x93;
    const scalar_t x95  = x53 + x87 + x94;
    const scalar_t x96  = x24 * x95;
    const scalar_t x97  = adjugate[0] * x93;
    const scalar_t x98  = x58 + x90 + x97;
    const scalar_t x99  = x28 * x98;
    const scalar_t x100 = x63 + x76 + x94;
    const scalar_t x101 = x100 * x24;
    const scalar_t x102 = x65 + x79 + x97;
    const scalar_t x103 = x102 * x28;
    const scalar_t x104 = x39 * x44;
    const scalar_t x105 = x10 * x41;
    const scalar_t x106 = x48 * x56;
    const scalar_t x107 = x50 * x60;
    const scalar_t x108 = x38 * x66;
    const scalar_t x109 = 0.66666666666666663 * x50;
    const scalar_t x110 = x38 * x80;
    const scalar_t x111 = x48 * x88;
    const scalar_t x112 = x50 * x91;
    const scalar_t x113 = x48 * x95;
    const scalar_t x114 = x50 * x98;
    const scalar_t x115 = x100 * x48;
    const scalar_t x116 = x102 * x50;
    const scalar_t x117 = x33 * (x0 + x32);
    const scalar_t x118 = x60 * x72;
    const scalar_t x119 = x56 * x70;
    const scalar_t x120 = x60 * x85;
    const scalar_t x121 = x56 * x84;
    const scalar_t x122 = x60 * x88;
    const scalar_t x123 = x56 * x91;
    const scalar_t x124 = x60 * x95;
    const scalar_t x125 = x56 * x98;
    const scalar_t x126 = x100 * x60;
    const scalar_t x127 = x102 * x56;
    const scalar_t x128 = 0.66666666666666663 * x64;
    const scalar_t x129 = x66 * x77;
    const scalar_t x130 = x64 * x80;
    const scalar_t x131 = x70 * x88;
    const scalar_t x132 = x72 * x91;
    const scalar_t x133 = x70 * x95;
    const scalar_t x134 = x72 * x98;
    const scalar_t x135 = x100 * x70;
    const scalar_t x136 = x102 * x72;
    const scalar_t x137 = x84 * x88;
    const scalar_t x138 = x85 * x91;
    const scalar_t x139 = x84 * x95;
    const scalar_t x140 = x85 * x98;
    const scalar_t x141 = x100 * x84;
    const scalar_t x142 = x102 * x85;
    const scalar_t x143 = x91 * x95;
    const scalar_t x144 = x88 * x98;
    const scalar_t x145 = x100 * x91;
    const scalar_t x146 = x102 * x88;
    const scalar_t x147 = x100 * x98;
    const scalar_t x148 = x102 * x95;
    element_matrix[0] += x33 * (x0 * x10 * x14 + x24 * x28 * x32);
    element_matrix[1] += -x33 * (x14 * x45 + x38 * x40 + x51 * (-x24 * x50 + 0.66666666666666663 * x49));
    element_matrix[2] += x33 * (x39 * x57 + x41 * x61 - x51 * (-x57 + 0.66666666666666663 * x61));
    element_matrix[3] += -x33 * (x40 * x64 + x41 * x67 + x51 * (-x24 * x72 + x70 * x71));
    element_matrix[4] += -x33 * (x40 * x77 + x41 * x81 + x51 * (-x24 * x85 + x71 * x84));
    element_matrix[5] += x33 * (x39 * x89 + x41 * x92 - x51 * (-x89 + 0.66666666666666663 * x92));
    element_matrix[6] += -x33 * (x39 * x96 + x41 * x99 + x51 * (x24 * x95 - 0.66666666666666663 * x99));
    element_matrix[7] += x33 * (x101 * x39 + x103 * x41 - x51 * (-x101 + 0.66666666666666663 * x103));
    element_matrix[8] += -x33 * (x104 * x14 + x105 * x38 + x51 * (0.66666666666666663 * x24 * x50 - x49));
    element_matrix[9] += x33 * (x0 * x38 * x44 + x32 * x48 * x50);
    element_matrix[10] += x33 * (x106 * x39 + x107 * x41 - x51 * (-x106 + 0.66666666666666663 * x107));
    element_matrix[11] += x33 * (x104 * x64 + x108 * x41 - x51 * (x109 * x70 - x48 * x72));
    element_matrix[12] += x33 * (x104 * x77 + x110 * x41 - x51 * (x109 * x84 - x48 * x85));
    element_matrix[13] += x33 * (x111 * x39 + x112 * x41 - x51 * (-x111 + 0.66666666666666663 * x112));
    element_matrix[14] += -x33 * (x113 * x39 + x114 * x41 + x51 * (-0.66666666666666663 * x114 + x48 * x95));
    element_matrix[15] += x33 * (x115 * x39 + x116 * x41 - x51 * (-x115 + 0.66666666666666663 * x116));
    element_matrix[16] += x33 * (x39 * x61 + x41 * x57 - x51 * (0.66666666666666663 * x24 * x56 - x61));
    element_matrix[17] += x33 * (x106 * x41 + x107 * x39 - x51 * (-x107 + 0.66666666666666663 * x48 * x56));
    element_matrix[18] += x117 * x56 * x60;
    element_matrix[19] += x33 * (x118 * x39 + x119 * x41 - x51 * (-x118 + 0.66666666666666663 * x119));
    element_matrix[20] += x33 * (x120 * x39 + x121 * x41 - x51 * (-x120 + 0.66666666666666663 * x56 * x84));
    element_matrix[21] += x33 * (x122 * x39 + x123 * x41 + x51 * (x122 - 0.66666666666666663 * x123));
    element_matrix[22] += -x33 * (x124 * x39 + x125 * x41 + x51 * (-0.66666666666666663 * x125 + x60 * x95));
    element_matrix[23] += x33 * (x126 * x39 + x127 * x41 - x51 * (-x126 + 0.66666666666666663 * x127));
    element_matrix[24] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (0.66666666666666663 * x10 * x64 - x67) - x105 * x64 - x39 * x67);
    element_matrix[25] += x33 * (x108 * x39 + x45 * x64 + x51 * (x108 - x128 * x44));
    element_matrix[26] += x33 * (x118 * x41 + x119 * x39 - x51 * (-x119 + 0.66666666666666663 * x60 * x72));
    element_matrix[27] += x33 * (x0 * x64 * x66 + x32 * x70 * x72);
    element_matrix[28] += x33 * (x129 * x39 + x130 * x41 + x51 * (-x128 * x80 + x129));
    element_matrix[29] += x33 * (x131 * x39 + x132 * x41 - x51 * (-x131 + 0.66666666666666663 * x72 * x91));
    element_matrix[30] += -x33 * (x133 * x39 + x134 * x41 + x51 * (-0.66666666666666663 * x134 + x70 * x95));
    element_matrix[31] += x33 * (x135 * x39 + x136 * x41 - x51 * (-x135 + 0.66666666666666663 * x136));
    element_matrix[32] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (0.66666666666666663 * x10 * x77 - x81) - x105 * x77 - x39 * x81);
    element_matrix[33] += x33 * (x110 * x39 + x45 * x77 + x51 * (x110 - 0.66666666666666663 * x44 * x77));
    element_matrix[34] += x33 * (x120 * x41 + x121 * x39 - x51 * (0.66666666666666663 * x120 - x121));
    element_matrix[35] += x33 * (x129 * x41 + x130 * x39 - x51 * (0.66666666666666663 * x70 * x85 - x72 * x84));
    element_matrix[36] += x33 * (x0 * x77 * x80 + x32 * x84 * x85);
    element_matrix[37] += x33 * (x137 * x39 + x138 * x41 - x51 * (-x137 + 0.66666666666666663 * x85 * x91));
    element_matrix[38] += -x33 * (x139 * x39 + x140 * x41 + x51 * (-0.66666666666666663 * x140 + x84 * x95));
    element_matrix[39] += x33 * (x141 * x39 + x142 * x41 - x51 * (-x141 + 0.66666666666666663 * x142));
    element_matrix[40] += x33 * (x39 * x92 + x41 * x89 - x51 * (0.66666666666666663 * x24 * x88 - x92));
    element_matrix[41] += x33 * (x111 * x41 + x112 * x39 - x51 * (-x112 + 0.66666666666666663 * x48 * x88));
    element_matrix[42] += x33 * (x122 * x41 + x123 * x39 - x51 * (0.66666666666666663 * x122 - x123));
    element_matrix[43] += x33 * (x131 * x41 + x132 * x39 - x51 * (0.66666666666666663 * x131 - x132));
    element_matrix[44] += x33 * (x137 * x41 + x138 * x39 - x51 * (0.66666666666666663 * x137 - x138));
    element_matrix[45] += x117 * x88 * x91;
    element_matrix[46] += -x33 * (x143 * x39 + x144 * x41 + x51 * (-0.66666666666666663 * x144 + x91 * x95));
    element_matrix[47] += x33 * (x145 * x39 + x146 * x41 - x51 * (-x145 + 0.66666666666666663 * x146));
    element_matrix[48] += -x33 * (x39 * x99 + x41 * x96 + x51 * (-0.66666666666666663 * x96 + x99));
    element_matrix[49] += -x33 * (x113 * x41 + x114 * x39 + x51 * (-0.66666666666666663 * x113 + x114));
    element_matrix[50] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (-x125 + 0.66666666666666663 * x60 * x95) - x124 * x41 - x125 * x39);
    element_matrix[51] += -x33 * (x133 * x41 + x134 * x39 + x51 * (-0.66666666666666663 * x133 + x134));
    element_matrix[52] += -x33 * (x139 * x41 + x140 * x39 + x51 * (-0.66666666666666663 * x139 + x140));
    element_matrix[53] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (-x144 + 0.66666666666666663 * x91 * x95) - x143 * x41 - x144 * x39);
    element_matrix[54] += x117 * x95 * x98;
    element_matrix[55] += -x33 * (x147 * x39 + x148 * x41 + x51 * (x147 - 0.66666666666666663 * x148));
    element_matrix[56] += x33 * (x101 * x41 + x103 * x39 - x51 * (0.66666666666666663 * x100 * x24 - x103));
    element_matrix[57] += x33 * (x115 * x41 + x116 * x39 - x51 * (0.66666666666666663 * x100 * x48 - x116));
    element_matrix[58] += x33 * (x126 * x41 + x127 * x39 + x51 * (-0.66666666666666663 * x126 + x127));
    element_matrix[59] += x33 * (x135 * x41 + x136 * x39 - x51 * (0.66666666666666663 * x100 * x70 - x136));
    element_matrix[60] += x33 * (x141 * x41 + x142 * x39 - x51 * (0.66666666666666663 * x100 * x84 - x142));
    element_matrix[61] += x33 * (x145 * x41 + x146 * x39 + x51 * (-0.66666666666666663 * x145 + x146));
    element_matrix[62] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (0.66666666666666663 * x147 - x148) - x147 * x41 - x148 * x39);
    element_matrix[63] += x100 * x102 * x117;
}

//--------------------------
// hessian block_2_1
//--------------------------

template <typename scalar_t, typename accumulator_t>
static inline __host__ __device__ void cu_hex8_kelvin_voigt_newmark_matrix_block_2_1(const scalar_t                      k,
                                                                                     const scalar_t                      K,
                                                                                     const scalar_t                      eta,
                                                                                     const scalar_t                      rho,
                                                                                     const scalar_t                      dt,
                                                                                     const scalar_t                      gamma,
                                                                                     const scalar_t                      beta,
                                                                                     const scalar_t *const SFEM_RESTRICT adjugate,
                                                                                     const scalar_t jacobian_determinant,
                                                                                     const scalar_t qx,
                                                                                     const scalar_t qy,
                                                                                     const scalar_t qz,
                                                                                     const scalar_t qw,
                                                                                     accumulator_t *const SFEM_RESTRICT
                                                                                             element_matrix) {
    // mundane ops: 821 divs: 3 sqrts: 0
    // total ops: 845
    const scalar_t x0   = K + 0.16666666666666669 * k;
    const scalar_t x1   = qy - 1;
    const scalar_t x2   = qz - 1;
    const scalar_t x3   = x1 * x2;
    const scalar_t x4   = adjugate[1] * x3;
    const scalar_t x5   = qx - 1;
    const scalar_t x6   = x2 * x5;
    const scalar_t x7   = adjugate[4] * x6;
    const scalar_t x8   = x1 * x5;
    const scalar_t x9   = adjugate[7] * x8;
    const scalar_t x10  = x4 + x7 + x9;
    const scalar_t x11  = adjugate[2] * x3;
    const scalar_t x12  = adjugate[5] * x6;
    const scalar_t x13  = adjugate[8] * x8;
    const scalar_t x14  = x11 + x12 + x13;
    const scalar_t x15  = -x1;
    const scalar_t x16  = -x2;
    const scalar_t x17  = x15 * x16;
    const scalar_t x18  = adjugate[1] * x17;
    const scalar_t x19  = -x5;
    const scalar_t x20  = x16 * x19;
    const scalar_t x21  = adjugate[4] * x20;
    const scalar_t x22  = x15 * x19;
    const scalar_t x23  = adjugate[7] * x22;
    const scalar_t x24  = x18 + x21 + x23;
    const scalar_t x25  = adjugate[2] * x17;
    const scalar_t x26  = adjugate[5] * x20;
    const scalar_t x27  = adjugate[8] * x22;
    const scalar_t x28  = x25 + x26 + x27;
    const scalar_t x29  = 1.0 / beta;
    const scalar_t x30  = 1.0 / dt;
    const scalar_t x31  = eta * gamma * x29 * x30;
    const scalar_t x32  = 0.16666666666666669 * x31;
    const scalar_t x33  = qw / jacobian_determinant;
    const scalar_t x34  = qx * x2;
    const scalar_t x35  = adjugate[5] * x34;
    const scalar_t x36  = qx * x1;
    const scalar_t x37  = adjugate[8] * x36;
    const scalar_t x38  = x11 + x35 + x37;
    const scalar_t x39  = (1.0 / 2.0) * k;
    const scalar_t x40  = x10 * x39;
    const scalar_t x41  = K - 0.33333333333333331 * k;
    const scalar_t x42  = adjugate[4] * x34;
    const scalar_t x43  = adjugate[7] * x36;
    const scalar_t x44  = x4 + x42 + x43;
    const scalar_t x45  = x41 * x44;
    const scalar_t x46  = qx * x16;
    const scalar_t x47  = qx * x15;
    const scalar_t x48  = adjugate[4] * x46 + adjugate[7] * x47 - x18;
    const scalar_t x49  = x28 * x48;
    const scalar_t x50  = adjugate[5] * x46 + adjugate[8] * x47 - x25;
    const scalar_t x51  = (1.0 / 2.0) * x31;
    const scalar_t x52  = qx * qy;
    const scalar_t x53  = adjugate[8] * x52;
    const scalar_t x54  = qy * x2;
    const scalar_t x55  = adjugate[2] * x54;
    const scalar_t x56  = x35 + x53 + x55;
    const scalar_t x57  = x24 * x56;
    const scalar_t x58  = adjugate[7] * x52;
    const scalar_t x59  = adjugate[1] * x54;
    const scalar_t x60  = x42 + x58 + x59;
    const scalar_t x61  = x28 * x60;
    const scalar_t x62  = qy * x5;
    const scalar_t x63  = adjugate[8] * x62;
    const scalar_t x64  = x12 + x55 + x63;
    const scalar_t x65  = adjugate[7] * x62;
    const scalar_t x66  = x59 + x65 + x7;
    const scalar_t x67  = x14 * x66;
    const scalar_t x68  = qy * x16;
    const scalar_t x69  = qy * x19;
    const scalar_t x70  = adjugate[1] * x68 + adjugate[7] * x69 - x21;
    const scalar_t x71  = 0.66666666666666663 * x28;
    const scalar_t x72  = adjugate[2] * x68 + adjugate[8] * x69 - x26;
    const scalar_t x73  = qz * x1;
    const scalar_t x74  = adjugate[2] * x73;
    const scalar_t x75  = qz * x5;
    const scalar_t x76  = adjugate[5] * x75;
    const scalar_t x77  = x13 + x74 + x76;
    const scalar_t x78  = adjugate[1] * x73;
    const scalar_t x79  = adjugate[4] * x75;
    const scalar_t x80  = x78 + x79 + x9;
    const scalar_t x81  = x14 * x80;
    const scalar_t x82  = qz * x15;
    const scalar_t x83  = qz * x19;
    const scalar_t x84  = adjugate[1] * x82 + adjugate[4] * x83 - x23;
    const scalar_t x85  = adjugate[2] * x82 + adjugate[5] * x83 - x27;
    const scalar_t x86  = qx * qz;
    const scalar_t x87  = adjugate[5] * x86;
    const scalar_t x88  = x37 + x74 + x87;
    const scalar_t x89  = x24 * x88;
    const scalar_t x90  = adjugate[4] * x86;
    const scalar_t x91  = x43 + x78 + x90;
    const scalar_t x92  = x28 * x91;
    const scalar_t x93  = qy * qz;
    const scalar_t x94  = adjugate[2] * x93;
    const scalar_t x95  = x53 + x87 + x94;
    const scalar_t x96  = x24 * x95;
    const scalar_t x97  = adjugate[1] * x93;
    const scalar_t x98  = x58 + x90 + x97;
    const scalar_t x99  = x28 * x98;
    const scalar_t x100 = x63 + x76 + x94;
    const scalar_t x101 = x100 * x24;
    const scalar_t x102 = x65 + x79 + x97;
    const scalar_t x103 = x102 * x28;
    const scalar_t x104 = x39 * x44;
    const scalar_t x105 = x10 * x41;
    const scalar_t x106 = x48 * x56;
    const scalar_t x107 = x50 * x60;
    const scalar_t x108 = x38 * x66;
    const scalar_t x109 = 0.66666666666666663 * x50;
    const scalar_t x110 = x38 * x80;
    const scalar_t x111 = x48 * x88;
    const scalar_t x112 = x50 * x91;
    const scalar_t x113 = x48 * x95;
    const scalar_t x114 = x50 * x98;
    const scalar_t x115 = x100 * x48;
    const scalar_t x116 = x102 * x50;
    const scalar_t x117 = x33 * (x0 + x32);
    const scalar_t x118 = x60 * x72;
    const scalar_t x119 = x56 * x70;
    const scalar_t x120 = x60 * x85;
    const scalar_t x121 = x56 * x84;
    const scalar_t x122 = x60 * x88;
    const scalar_t x123 = x56 * x91;
    const scalar_t x124 = x60 * x95;
    const scalar_t x125 = x56 * x98;
    const scalar_t x126 = x100 * x60;
    const scalar_t x127 = x102 * x56;
    const scalar_t x128 = 0.66666666666666663 * x64;
    const scalar_t x129 = x66 * x77;
    const scalar_t x130 = x64 * x80;
    const scalar_t x131 = x70 * x88;
    const scalar_t x132 = x72 * x91;
    const scalar_t x133 = x70 * x95;
    const scalar_t x134 = x72 * x98;
    const scalar_t x135 = x100 * x70;
    const scalar_t x136 = x102 * x72;
    const scalar_t x137 = x84 * x88;
    const scalar_t x138 = x85 * x91;
    const scalar_t x139 = x84 * x95;
    const scalar_t x140 = x85 * x98;
    const scalar_t x141 = x100 * x84;
    const scalar_t x142 = x102 * x85;
    const scalar_t x143 = x91 * x95;
    const scalar_t x144 = x88 * x98;
    const scalar_t x145 = x100 * x91;
    const scalar_t x146 = x102 * x88;
    const scalar_t x147 = x100 * x98;
    const scalar_t x148 = x102 * x95;
    element_matrix[0] += x33 * (x0 * x10 * x14 + x24 * x28 * x32);
    element_matrix[1] += -x33 * (x14 * x45 + x38 * x40 + x51 * (-x24 * x50 + 0.66666666666666663 * x49));
    element_matrix[2] += x33 * (x39 * x57 + x41 * x61 - x51 * (-x57 + 0.66666666666666663 * x61));
    element_matrix[3] += -x33 * (x40 * x64 + x41 * x67 + x51 * (-x24 * x72 + x70 * x71));
    element_matrix[4] += -x33 * (x40 * x77 + x41 * x81 + x51 * (-x24 * x85 + x71 * x84));
    element_matrix[5] += x33 * (x39 * x89 + x41 * x92 - x51 * (-x89 + 0.66666666666666663 * x92));
    element_matrix[6] += -x33 * (x39 * x96 + x41 * x99 + x51 * (x24 * x95 - 0.66666666666666663 * x99));
    element_matrix[7] += x33 * (x101 * x39 + x103 * x41 - x51 * (-x101 + 0.66666666666666663 * x103));
    element_matrix[8] += -x33 * (x104 * x14 + x105 * x38 + x51 * (0.66666666666666663 * x24 * x50 - x49));
    element_matrix[9] += x33 * (x0 * x38 * x44 + x32 * x48 * x50);
    element_matrix[10] += x33 * (x106 * x39 + x107 * x41 - x51 * (-x106 + 0.66666666666666663 * x107));
    element_matrix[11] += x33 * (x104 * x64 + x108 * x41 - x51 * (x109 * x70 - x48 * x72));
    element_matrix[12] += x33 * (x104 * x77 + x110 * x41 - x51 * (x109 * x84 - x48 * x85));
    element_matrix[13] += x33 * (x111 * x39 + x112 * x41 - x51 * (-x111 + 0.66666666666666663 * x112));
    element_matrix[14] += -x33 * (x113 * x39 + x114 * x41 + x51 * (-0.66666666666666663 * x114 + x48 * x95));
    element_matrix[15] += x33 * (x115 * x39 + x116 * x41 - x51 * (-x115 + 0.66666666666666663 * x116));
    element_matrix[16] += x33 * (x39 * x61 + x41 * x57 - x51 * (0.66666666666666663 * x24 * x56 - x61));
    element_matrix[17] += x33 * (x106 * x41 + x107 * x39 - x51 * (-x107 + 0.66666666666666663 * x48 * x56));
    element_matrix[18] += x117 * x56 * x60;
    element_matrix[19] += x33 * (x118 * x39 + x119 * x41 - x51 * (-x118 + 0.66666666666666663 * x119));
    element_matrix[20] += x33 * (x120 * x39 + x121 * x41 - x51 * (-x120 + 0.66666666666666663 * x56 * x84));
    element_matrix[21] += x33 * (x122 * x39 + x123 * x41 + x51 * (x122 - 0.66666666666666663 * x123));
    element_matrix[22] += -x33 * (x124 * x39 + x125 * x41 + x51 * (-0.66666666666666663 * x125 + x60 * x95));
    element_matrix[23] += x33 * (x126 * x39 + x127 * x41 - x51 * (-x126 + 0.66666666666666663 * x127));
    element_matrix[24] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (0.66666666666666663 * x10 * x64 - x67) - x105 * x64 - x39 * x67);
    element_matrix[25] += x33 * (x108 * x39 + x45 * x64 + x51 * (x108 - x128 * x44));
    element_matrix[26] += x33 * (x118 * x41 + x119 * x39 - x51 * (-x119 + 0.66666666666666663 * x60 * x72));
    element_matrix[27] += x33 * (x0 * x64 * x66 + x32 * x70 * x72);
    element_matrix[28] += x33 * (x129 * x39 + x130 * x41 + x51 * (-x128 * x80 + x129));
    element_matrix[29] += x33 * (x131 * x39 + x132 * x41 - x51 * (-x131 + 0.66666666666666663 * x72 * x91));
    element_matrix[30] += -x33 * (x133 * x39 + x134 * x41 + x51 * (-0.66666666666666663 * x134 + x70 * x95));
    element_matrix[31] += x33 * (x135 * x39 + x136 * x41 - x51 * (-x135 + 0.66666666666666663 * x136));
    element_matrix[32] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (0.66666666666666663 * x10 * x77 - x81) - x105 * x77 - x39 * x81);
    element_matrix[33] += x33 * (x110 * x39 + x45 * x77 + x51 * (x110 - 0.66666666666666663 * x44 * x77));
    element_matrix[34] += x33 * (x120 * x41 + x121 * x39 - x51 * (0.66666666666666663 * x120 - x121));
    element_matrix[35] += x33 * (x129 * x41 + x130 * x39 - x51 * (0.66666666666666663 * x70 * x85 - x72 * x84));
    element_matrix[36] += x33 * (x0 * x77 * x80 + x32 * x84 * x85);
    element_matrix[37] += x33 * (x137 * x39 + x138 * x41 - x51 * (-x137 + 0.66666666666666663 * x85 * x91));
    element_matrix[38] += -x33 * (x139 * x39 + x140 * x41 + x51 * (-0.66666666666666663 * x140 + x84 * x95));
    element_matrix[39] += x33 * (x141 * x39 + x142 * x41 - x51 * (-x141 + 0.66666666666666663 * x142));
    element_matrix[40] += x33 * (x39 * x92 + x41 * x89 - x51 * (0.66666666666666663 * x24 * x88 - x92));
    element_matrix[41] += x33 * (x111 * x41 + x112 * x39 - x51 * (-x112 + 0.66666666666666663 * x48 * x88));
    element_matrix[42] += x33 * (x122 * x41 + x123 * x39 - x51 * (0.66666666666666663 * x122 - x123));
    element_matrix[43] += x33 * (x131 * x41 + x132 * x39 - x51 * (0.66666666666666663 * x131 - x132));
    element_matrix[44] += x33 * (x137 * x41 + x138 * x39 - x51 * (0.66666666666666663 * x137 - x138));
    element_matrix[45] += x117 * x88 * x91;
    element_matrix[46] += -x33 * (x143 * x39 + x144 * x41 + x51 * (-0.66666666666666663 * x144 + x91 * x95));
    element_matrix[47] += x33 * (x145 * x39 + x146 * x41 - x51 * (-x145 + 0.66666666666666663 * x146));
    element_matrix[48] += -x33 * (x39 * x99 + x41 * x96 + x51 * (-0.66666666666666663 * x96 + x99));
    element_matrix[49] += -x33 * (x113 * x41 + x114 * x39 + x51 * (-0.66666666666666663 * x113 + x114));
    element_matrix[50] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (-x125 + 0.66666666666666663 * x60 * x95) - x124 * x41 - x125 * x39);
    element_matrix[51] += -x33 * (x133 * x41 + x134 * x39 + x51 * (-0.66666666666666663 * x133 + x134));
    element_matrix[52] += -x33 * (x139 * x41 + x140 * x39 + x51 * (-0.66666666666666663 * x139 + x140));
    element_matrix[53] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (-x144 + 0.66666666666666663 * x91 * x95) - x143 * x41 - x144 * x39);
    element_matrix[54] += x117 * x95 * x98;
    element_matrix[55] += -x33 * (x147 * x39 + x148 * x41 + x51 * (x147 - 0.66666666666666663 * x148));
    element_matrix[56] += x33 * (x101 * x41 + x103 * x39 - x51 * (0.66666666666666663 * x100 * x24 - x103));
    element_matrix[57] += x33 * (x115 * x41 + x116 * x39 - x51 * (0.66666666666666663 * x100 * x48 - x116));
    element_matrix[58] += x33 * (x126 * x41 + x127 * x39 + x51 * (-0.66666666666666663 * x126 + x127));
    element_matrix[59] += x33 * (x135 * x41 + x136 * x39 - x51 * (0.66666666666666663 * x100 * x70 - x136));
    element_matrix[60] += x33 * (x141 * x41 + x142 * x39 - x51 * (0.66666666666666663 * x100 * x84 - x142));
    element_matrix[61] += x33 * (x145 * x41 + x146 * x39 + x51 * (-0.66666666666666663 * x145 + x146));
    element_matrix[62] +=
            x33 * ((1.0 / 2.0) * eta * gamma * x29 * x30 * (0.66666666666666663 * x147 - x148) - x147 * x41 - x148 * x39);
    element_matrix[63] += x100 * x102 * x117;
}

//--------------------------
// hessian block_2_2
//--------------------------

template <typename scalar_t, typename accumulator_t>
static inline __host__ __device__ void cu_hex8_kelvin_voigt_newmark_matrix_block_2_2(const scalar_t                      k,
                                                                                     const scalar_t                      K,
                                                                                     const scalar_t                      eta,
                                                                                     const scalar_t                      rho,
                                                                                     const scalar_t                      dt,
                                                                                     const scalar_t                      gamma,
                                                                                     const scalar_t                      beta,
                                                                                     const scalar_t *const SFEM_RESTRICT adjugate,
                                                                                     const scalar_t jacobian_determinant,
                                                                                     const scalar_t qx,
                                                                                     const scalar_t qy,
                                                                                     const scalar_t qz,
                                                                                     const scalar_t qw,
                                                                                     accumulator_t *const SFEM_RESTRICT
                                                                                             element_matrix) {
    // mundane ops: 931 divs: 3 sqrts: 0
    // total ops: 955
    const scalar_t x0   = qx - 1;
    const scalar_t x1   = POW2(x0);
    const scalar_t x2   = qy - 1;
    const scalar_t x3   = POW2(x2);
    const scalar_t x4   = qz - 1;
    const scalar_t x5   = 1.0 / beta;
    const scalar_t x6   = jacobian_determinant * rho * x5 / POW2(dt);
    const scalar_t x7   = POW2(x4) * x6;
    const scalar_t x8   = x3 * x7;
    const scalar_t x9   = x2 * x4;
    const scalar_t x10  = adjugate[2] * x9;
    const scalar_t x11  = x0 * x4;
    const scalar_t x12  = adjugate[5] * x11;
    const scalar_t x13  = x0 * x2;
    const scalar_t x14  = adjugate[8] * x13;
    const scalar_t x15  = x10 + x12 + x14;
    const scalar_t x16  = POW2(x15);
    const scalar_t x17  = adjugate[0] * x9;
    const scalar_t x18  = adjugate[3] * x11;
    const scalar_t x19  = adjugate[6] * x13;
    const scalar_t x20  = x17 + x18 + x19;
    const scalar_t x21  = adjugate[1] * x9;
    const scalar_t x22  = adjugate[4] * x11;
    const scalar_t x23  = adjugate[7] * x13;
    const scalar_t x24  = x21 + x22 + x23;
    const scalar_t x25  = POW2(x20) + POW2(x24);
    const scalar_t x26  = 1.0 / dt;
    const scalar_t x27  = 1.0 / jacobian_determinant;
    const scalar_t x28  = (1.0 / 2.0) * x27;
    const scalar_t x29  = eta * gamma * x26 * x28 * x5;
    const scalar_t x30  = K - 0.33333333333333331 * k;
    const scalar_t x31  = 2 * x16;
    const scalar_t x32  = qx * x0;
    const scalar_t x33  = qx * x4;
    const scalar_t x34  = adjugate[5] * x33;
    const scalar_t x35  = qx * x2;
    const scalar_t x36  = adjugate[8] * x35;
    const scalar_t x37  = x10 + x34 + x36;
    const scalar_t x38  = x15 * x37;
    const scalar_t x39  = adjugate[3] * x33;
    const scalar_t x40  = adjugate[6] * x35;
    const scalar_t x41  = x17 + x39 + x40;
    const scalar_t x42  = adjugate[4] * x33;
    const scalar_t x43  = adjugate[7] * x35;
    const scalar_t x44  = x21 + x42 + x43;
    const scalar_t x45  = x20 * x41 + x24 * x44;
    const scalar_t x46  = 2 * x38;
    const scalar_t x47  = qw * ((1.0 / 2.0) * eta * gamma * x26 * x27 * x5 * (-1.3333333333333335 * x38 - x45) -
                               x28 * (k * (x45 + x46) + x30 * x46) - x32 * x8);
    const scalar_t x48  = qx * qy;
    const scalar_t x49  = x13 * x48;
    const scalar_t x50  = x49 * x7;
    const scalar_t x51  = adjugate[8] * x48;
    const scalar_t x52  = qy * x4;
    const scalar_t x53  = adjugate[2] * x52;
    const scalar_t x54  = x34 + x51 + x53;
    const scalar_t x55  = -x2;
    const scalar_t x56  = -x4;
    const scalar_t x57  = x55 * x56;
    const scalar_t x58  = adjugate[2] * x57;
    const scalar_t x59  = -x0;
    const scalar_t x60  = x56 * x59;
    const scalar_t x61  = adjugate[5] * x60;
    const scalar_t x62  = x55 * x59;
    const scalar_t x63  = adjugate[8] * x62;
    const scalar_t x64  = x58 + x61 + x63;
    const scalar_t x65  = x54 * x64;
    const scalar_t x66  = adjugate[6] * x48;
    const scalar_t x67  = adjugate[0] * x52;
    const scalar_t x68  = x39 + x66 + x67;
    const scalar_t x69  = adjugate[0] * x57;
    const scalar_t x70  = adjugate[3] * x60;
    const scalar_t x71  = adjugate[6] * x62;
    const scalar_t x72  = x69 + x70 + x71;
    const scalar_t x73  = adjugate[7] * x48;
    const scalar_t x74  = adjugate[1] * x52;
    const scalar_t x75  = x42 + x73 + x74;
    const scalar_t x76  = adjugate[1] * x57;
    const scalar_t x77  = adjugate[4] * x60;
    const scalar_t x78  = adjugate[7] * x62;
    const scalar_t x79  = x76 + x77 + x78;
    const scalar_t x80  = x68 * x72 + x75 * x79;
    const scalar_t x81  = 2 * x30;
    const scalar_t x82  = x15 * x81;
    const scalar_t x83  = qw * (x28 * (k * (2 * x65 + x80) + x54 * x82) - x29 * (-1.3333333333333335 * x65 - x80) + x50);
    const scalar_t x84  = qy * x2;
    const scalar_t x85  = x7 * x84;
    const scalar_t x86  = qy * x0;
    const scalar_t x87  = adjugate[8] * x86;
    const scalar_t x88  = x12 + x53 + x87;
    const scalar_t x89  = x15 * x88;
    const scalar_t x90  = adjugate[6] * x86;
    const scalar_t x91  = x18 + x67 + x90;
    const scalar_t x92  = adjugate[7] * x86;
    const scalar_t x93  = x22 + x74 + x92;
    const scalar_t x94  = x20 * x91 + x24 * x93;
    const scalar_t x95  = 2 * x89;
    const scalar_t x96  = qw * ((1.0 / 2.0) * eta * gamma * x26 * x27 * x5 * (-1.3333333333333335 * x89 - x94) - x1 * x85 -
                               x28 * (k * (x94 + x95) + x30 * x95));
    const scalar_t x97  = x3 * x6;
    const scalar_t x98  = qz * x4;
    const scalar_t x99  = x97 * x98;
    const scalar_t x100 = qz * x2;
    const scalar_t x101 = adjugate[2] * x100;
    const scalar_t x102 = qz * x0;
    const scalar_t x103 = adjugate[5] * x102;
    const scalar_t x104 = x101 + x103 + x14;
    const scalar_t x105 = x104 * x15;
    const scalar_t x106 = adjugate[0] * x100;
    const scalar_t x107 = adjugate[3] * x102;
    const scalar_t x108 = x106 + x107 + x19;
    const scalar_t x109 = adjugate[1] * x100;
    const scalar_t x110 = adjugate[4] * x102;
    const scalar_t x111 = x109 + x110 + x23;
    const scalar_t x112 = x108 * x20 + x111 * x24;
    const scalar_t x113 = 2 * x105;
    const scalar_t x114 = qw * ((1.0 / 2.0) * eta * gamma * x26 * x27 * x5 * (-1.3333333333333335 * x105 - x112) - x1 * x99 -
                                x28 * (k * (x112 + x113) + x113 * x30));
    const scalar_t x115 = qx * qz;
    const scalar_t x116 = x11 * x115;
    const scalar_t x117 = x116 * x97;
    const scalar_t x118 = adjugate[5] * x115;
    const scalar_t x119 = x101 + x118 + x36;
    const scalar_t x120 = x119 * x64;
    const scalar_t x121 = adjugate[3] * x115;
    const scalar_t x122 = x106 + x121 + x40;
    const scalar_t x123 = adjugate[4] * x115;
    const scalar_t x124 = x109 + x123 + x43;
    const scalar_t x125 = x122 * x72 + x124 * x79;
    const scalar_t x126 = qw * (x117 + x28 * (k * (2 * x120 + x125) + x119 * x82) - x29 * (-1.3333333333333335 * x120 - x125));
    const scalar_t x127 = x102 * x48 * x6 * x9;
    const scalar_t x128 = qy * qz;
    const scalar_t x129 = adjugate[2] * x128;
    const scalar_t x130 = x118 + x129 + x51;
    const scalar_t x131 = x130 * x64;
    const scalar_t x132 = adjugate[0] * x128;
    const scalar_t x133 = x121 + x132 + x66;
    const scalar_t x134 = adjugate[1] * x128;
    const scalar_t x135 = x123 + x134 + x73;
    const scalar_t x136 = x133 * x72 + x135 * x79;
    const scalar_t x137 = -qw * (x127 + x28 * (k * (2 * x131 + x136) + x130 * x82) + x29 * (1.3333333333333335 * x131 + x136));
    const scalar_t x138 = x1 * x6;
    const scalar_t x139 = x128 * x9;
    const scalar_t x140 = x138 * x139;
    const scalar_t x141 = x103 + x129 + x87;
    const scalar_t x142 = x141 * x64;
    const scalar_t x143 = x107 + x132 + x90;
    const scalar_t x144 = x110 + x134 + x92;
    const scalar_t x145 = x143 * x72 + x144 * x79;
    const scalar_t x146 = qw * (x140 + x28 * (k * (2 * x142 + x145) + x141 * x82) - x29 * (-1.3333333333333335 * x142 - x145));
    const scalar_t x147 = POW2(qx);
    const scalar_t x148 = POW2(x37);
    const scalar_t x149 = POW2(x41) + POW2(x44);
    const scalar_t x150 = 2 * x148;
    const scalar_t x151 = qx * x56;
    const scalar_t x152 = qx * x55;
    const scalar_t x153 = adjugate[5] * x151 + adjugate[8] * x152 - x58;
    const scalar_t x154 = x153 * x54;
    const scalar_t x155 = adjugate[3] * x151 + adjugate[6] * x152 - x69;
    const scalar_t x156 = adjugate[4] * x151 + adjugate[7] * x152 - x76;
    const scalar_t x157 = x155 * x68 + x156 * x75;
    const scalar_t x158 = x37 * x81;
    const scalar_t x159 = qw * (-x147 * x85 + (1.0 / 2.0) * x27 * (k * (2 * x154 + x157) - x158 * x54) -
                                x29 * (-1.3333333333333335 * x154 - x157));
    const scalar_t x160 = x37 * x88;
    const scalar_t x161 = x41 * x91 + x44 * x93;
    const scalar_t x162 = 2 * x160;
    const scalar_t x163 = qw * (x28 * (k * (x161 + x162) + x162 * x30) + x29 * (1.3333333333333335 * x160 + x161) + x50);
    const scalar_t x164 = x104 * x37;
    const scalar_t x165 = x108 * x41 + x111 * x44;
    const scalar_t x166 = 2 * x164;
    const scalar_t x167 = qw * (x117 + x28 * (k * (x165 + x166) + x166 * x30) + x29 * (1.3333333333333335 * x164 + x165));
    const scalar_t x168 = x119 * x153;
    const scalar_t x169 = x122 * x155 + x124 * x156;
    const scalar_t x170 = qw * (-x147 * x99 + (1.0 / 2.0) * x27 * (k * (2 * x168 + x169) - x119 * x158) -
                                x29 * (-1.3333333333333335 * x168 - x169));
    const scalar_t x171 = x147 * x6;
    const scalar_t x172 = x139 * x171;
    const scalar_t x173 = x130 * x153;
    const scalar_t x174 = x133 * x155 + x135 * x156;
    const scalar_t x175 =
            qw * (x172 + x28 * (-k * (2 * x173 + x174) + 2 * x130 * x30 * x37) - x29 * (1.3333333333333335 * x173 + x174));
    const scalar_t x176 = x141 * x153;
    const scalar_t x177 = x143 * x155 + x144 * x156;
    const scalar_t x178 =
            qw * (-x127 + (1.0 / 2.0) * x27 * (k * (2 * x176 + x177) - x141 * x158) - x29 * (-1.3333333333333335 * x176 - x177));
    const scalar_t x179 = POW2(qy);
    const scalar_t x180 = x179 * x7;
    const scalar_t x181 = POW2(x54);
    const scalar_t x182 = POW2(x68) + POW2(x75);
    const scalar_t x183 = 2 * x181;
    const scalar_t x184 = qy * x56;
    const scalar_t x185 = qy * x59;
    const scalar_t x186 = adjugate[2] * x184 + adjugate[8] * x185 - x61;
    const scalar_t x187 = x186 * x54;
    const scalar_t x188 = adjugate[0] * x184 + adjugate[6] * x185 - x70;
    const scalar_t x189 = adjugate[1] * x184 + adjugate[7] * x185 - x77;
    const scalar_t x190 = x188 * x68 + x189 * x75;
    const scalar_t x191 = x54 * x81;
    const scalar_t x192 = qw * (-x180 * x32 + (1.0 / 2.0) * x27 * (k * (2 * x187 + x190) - x191 * x88) -
                                x29 * (-1.3333333333333335 * x187 - x190));
    const scalar_t x193 = qz * x55;
    const scalar_t x194 = qz * x59;
    const scalar_t x195 = adjugate[2] * x193 + adjugate[5] * x194 - x63;
    const scalar_t x196 = x195 * x54;
    const scalar_t x197 = adjugate[0] * x193 + adjugate[3] * x194 - x71;
    const scalar_t x198 = adjugate[1] * x193 + adjugate[4] * x194 - x78;
    const scalar_t x199 = x197 * x68 + x198 * x75;
    const scalar_t x200 =
            qw * (-x127 + (1.0 / 2.0) * x27 * (k * (2 * x196 + x199) - x104 * x191) - x29 * (-1.3333333333333335 * x196 - x199));
    const scalar_t x201 = x119 * x54;
    const scalar_t x202 = x122 * x68 + x124 * x75;
    const scalar_t x203 = 2 * x201;
    const scalar_t x204 = qw * (x172 + x28 * (k * (x202 + x203) + x203 * x30) + x29 * (1.3333333333333335 * x201 + x202));
    const scalar_t x205 = x179 * x98;
    const scalar_t x206 = x130 * x54;
    const scalar_t x207 = x133 * x68 + x135 * x75;
    const scalar_t x208 = 2 * x206;
    const scalar_t x209 = qw * ((1.0 / 2.0) * eta * gamma * x26 * x27 * x5 * (-1.3333333333333335 * x206 - x207) - x171 * x205 -
                                x28 * (k * (x207 + x208) + x208 * x30));
    const scalar_t x210 = x116 * x179 * x6;
    const scalar_t x211 = x141 * x54;
    const scalar_t x212 = x143 * x68 + x144 * x75;
    const scalar_t x213 = 2 * x211;
    const scalar_t x214 = qw * (x210 + x28 * (k * (x212 + x213) + x213 * x30) + x29 * (1.3333333333333335 * x211 + x212));
    const scalar_t x215 = POW2(x88);
    const scalar_t x216 = POW2(x91) + POW2(x93);
    const scalar_t x217 = 2 * x215;
    const scalar_t x218 = x104 * x88;
    const scalar_t x219 = x108 * x91 + x111 * x93;
    const scalar_t x220 = 2 * x218;
    const scalar_t x221 = qw * (x140 + x28 * (k * (x219 + x220) + x220 * x30) + x29 * (1.3333333333333335 * x218 + x219));
    const scalar_t x222 = x119 * x186;
    const scalar_t x223 = x122 * x188 + x124 * x189;
    const scalar_t x224 = x81 * x88;
    const scalar_t x225 =
            qw * (-x127 + (1.0 / 2.0) * x27 * (k * (2 * x222 + x223) - x119 * x224) - x29 * (-1.3333333333333335 * x222 - x223));
    const scalar_t x226 = x130 * x186;
    const scalar_t x227 = x133 * x188 + x135 * x189;
    const scalar_t x228 =
            qw * (x210 + x28 * (-k * (2 * x226 + x227) + 2 * x130 * x30 * x88) - x29 * (1.3333333333333335 * x226 + x227));
    const scalar_t x229 = x141 * x186;
    const scalar_t x230 = x143 * x188 + x144 * x189;
    const scalar_t x231 = qw * (-x138 * x205 + (1.0 / 2.0) * x27 * (k * (2 * x229 + x230) - x141 * x224) -
                                x29 * (-1.3333333333333335 * x229 - x230));
    const scalar_t x232 = POW2(qz) * x6;
    const scalar_t x233 = x232 * x3;
    const scalar_t x234 = POW2(x104);
    const scalar_t x235 = POW2(x108) + POW2(x111);
    const scalar_t x236 = 2 * x234;
    const scalar_t x237 = x119 * x195;
    const scalar_t x238 = x122 * x197 + x124 * x198;
    const scalar_t x239 = x104 * x81;
    const scalar_t x240 = qw * (-x233 * x32 + (1.0 / 2.0) * x27 * (k * (2 * x237 + x238) - x119 * x239) -
                                x29 * (-1.3333333333333335 * x237 - x238));
    const scalar_t x241 = x232 * x49;
    const scalar_t x242 = x130 * x195;
    const scalar_t x243 = x133 * x197 + x135 * x198;
    const scalar_t x244 =
            qw * (x241 + x28 * (-k * (2 * x242 + x243) + 2 * x104 * x130 * x30) - x29 * (1.3333333333333335 * x242 + x243));
    const scalar_t x245 = x232 * x84;
    const scalar_t x246 = x141 * x195;
    const scalar_t x247 = x143 * x197 + x144 * x198;
    const scalar_t x248 = qw * (-x1 * x245 + (1.0 / 2.0) * x27 * (k * (2 * x246 + x247) - x141 * x239) -
                                x29 * (-1.3333333333333335 * x246 - x247));
    const scalar_t x249 = POW2(x119);
    const scalar_t x250 = POW2(x122) + POW2(x124);
    const scalar_t x251 = 2 * x249;
    const scalar_t x252 = x119 * x130;
    const scalar_t x253 = x122 * x133 + x124 * x135;
    const scalar_t x254 = 2 * x252;
    const scalar_t x255 = qw * ((1.0 / 2.0) * eta * gamma * x26 * x27 * x5 * (-1.3333333333333335 * x252 - x253) - x147 * x245 -
                                x28 * (k * (x253 + x254) + x254 * x30));
    const scalar_t x256 = x119 * x141;
    const scalar_t x257 = x122 * x143 + x124 * x144;
    const scalar_t x258 = 2 * x256;
    const scalar_t x259 = qw * (x241 + x28 * (k * (x257 + x258) + x258 * x30) + x29 * (1.3333333333333335 * x256 + x257));
    const scalar_t x260 = x179 * x232;
    const scalar_t x261 = POW2(x130);
    const scalar_t x262 = POW2(x133) + POW2(x135);
    const scalar_t x263 = 2 * x261;
    const scalar_t x264 = x130 * x141;
    const scalar_t x265 = x133 * x143 + x135 * x144;
    const scalar_t x266 = 2 * x264;
    const scalar_t x267 = qw * ((1.0 / 2.0) * eta * gamma * x26 * x27 * x5 * (-1.3333333333333335 * x264 - x265) - x260 * x32 -
                                x28 * (k * (x265 + x266) + x266 * x30));
    const scalar_t x268 = POW2(x141);
    const scalar_t x269 = POW2(x143) + POW2(x144);
    const scalar_t x270 = 2 * x268;
    element_matrix[0] += qw * (x1 * x8 + x28 * (k * (x25 + x31) + x30 * x31) + x29 * (1.3333333333333335 * x16 + x25));
    element_matrix[1] += x47;
    element_matrix[2] += x83;
    element_matrix[3] += x96;
    element_matrix[4] += x114;
    element_matrix[5] += x126;
    element_matrix[6] += x137;
    element_matrix[7] += x146;
    element_matrix[8] += x47;
    element_matrix[9] += qw * (x147 * x8 + x28 * (k * (x149 + x150) + x150 * x30) + x29 * (1.3333333333333335 * x148 + x149));
    element_matrix[10] += x159;
    element_matrix[11] += x163;
    element_matrix[12] += x167;
    element_matrix[13] += x170;
    element_matrix[14] += x175;
    element_matrix[15] += x178;
    element_matrix[16] += x83;
    element_matrix[17] += x159;
    element_matrix[18] += qw * (x147 * x180 + x28 * (k * (x182 + x183) + x183 * x30) + x29 * (1.3333333333333335 * x181 + x182));
    element_matrix[19] += x192;
    element_matrix[20] += x200;
    element_matrix[21] += x204;
    element_matrix[22] += x209;
    element_matrix[23] += x214;
    element_matrix[24] += x96;
    element_matrix[25] += x163;
    element_matrix[26] += x192;
    element_matrix[27] += qw * (x1 * x180 + x28 * (k * (x216 + x217) + x217 * x30) + x29 * (1.3333333333333335 * x215 + x216));
    element_matrix[28] += x221;
    element_matrix[29] += x225;
    element_matrix[30] += x228;
    element_matrix[31] += x231;
    element_matrix[32] += x114;
    element_matrix[33] += x167;
    element_matrix[34] += x200;
    element_matrix[35] += x221;
    element_matrix[36] += qw * (x1 * x233 + x28 * (k * (x235 + x236) + x236 * x30) + x29 * (1.3333333333333335 * x234 + x235));
    element_matrix[37] += x240;
    element_matrix[38] += x244;
    element_matrix[39] += x248;
    element_matrix[40] += x126;
    element_matrix[41] += x170;
    element_matrix[42] += x204;
    element_matrix[43] += x225;
    element_matrix[44] += x240;
    element_matrix[45] += qw * (x147 * x233 + x28 * (k * (x250 + x251) + x251 * x30) + x29 * (1.3333333333333335 * x249 + x250));
    element_matrix[46] += x255;
    element_matrix[47] += x259;
    element_matrix[48] += x137;
    element_matrix[49] += x175;
    element_matrix[50] += x209;
    element_matrix[51] += x228;
    element_matrix[52] += x244;
    element_matrix[53] += x255;
    element_matrix[54] += qw * (x147 * x260 + x28 * (k * (x262 + x263) + x263 * x30) + x29 * (1.3333333333333335 * x261 + x262));
    element_matrix[55] += x267;
    element_matrix[56] += x146;
    element_matrix[57] += x178;
    element_matrix[58] += x214;
    element_matrix[59] += x231;
    element_matrix[60] += x248;
    element_matrix[61] += x259;
    element_matrix[62] += x267;
    element_matrix[63] += qw * (x1 * x260 + x28 * (k * (x269 + x270) + x270 * x30) + x29 * (1.3333333333333335 * x268 + x269));
}

template <typename scalar_t>
static inline __device__ __host__ void cu_hex8_kv_ref_shape_grad_x(const scalar_t  qx,
                                                                   const scalar_t  qy,
                                                                   const scalar_t  qz,
                                                                   scalar_t *const out) {
    const scalar_t x0 = 1 - qy;
    const scalar_t x1 = 1 - qz;
    const scalar_t x2 = x0 * x1;
    const scalar_t x3 = qy * x1;
    const scalar_t x4 = qz * x0;
    const scalar_t x5 = qy * qz;
    out[0]            = -x2;
    out[1]            = x2;
    out[2]            = x3;
    out[3]            = -x3;
    out[4]            = -x4;
    out[5]            = x4;
    out[6]            = x5;
    out[7]            = -x5;
}

template <typename scalar_t>
static inline __device__ __host__ void cu_hex8_kv_ref_shape_grad_y(const scalar_t  qx,
                                                                   const scalar_t  qy,
                                                                   const scalar_t  qz,
                                                                   scalar_t *const out) {
    const scalar_t x0 = 1 - qx;
    const scalar_t x1 = 1 - qz;
    const scalar_t x2 = x0 * x1;
    const scalar_t x3 = qx * x1;
    const scalar_t x4 = qz * x0;
    const scalar_t x5 = qx * qz;
    out[0]            = -x2;
    out[1]            = -x3;
    out[2]            = x3;
    out[3]            = x2;
    out[4]            = -x4;
    out[5]            = -x5;
    out[6]            = x5;
    out[7]            = x4;
}

template <typename scalar_t>
static inline __device__ __host__ void cu_hex8_kv_ref_shape_grad_z(const scalar_t  qx,
                                                                   const scalar_t  qy,
                                                                   const scalar_t  qz,
                                                                   scalar_t *const out) {
    const scalar_t x0 = 1 - qx;
    const scalar_t x1 = 1 - qy;
    const scalar_t x2 = x0 * x1;
    const scalar_t x3 = qx * x1;
    const scalar_t x4 = qx * qy;
    const scalar_t x5 = qy * x0;
    out[0]            = -x2;
    out[1]            = -x3;
    out[2]            = -x4;
    out[3]            = -x5;
    out[4]            = x2;
    out[5]            = x3;
    out[6]            = x4;
    out[7]            = x5;
}

template <typename scalar_t>
static inline __device__ __host__ void cu_hex8_kv_ref_shape_fun(const scalar_t  qx,
                                                                const scalar_t  qy,
                                                                const scalar_t  qz,
                                                                scalar_t *const out) {
    const scalar_t x0 = 1 - qx;
    const scalar_t x1 = 1 - qy;
    const scalar_t x2 = 1 - qz;
    const scalar_t x3 = x0 * x1;
    const scalar_t x4 = qx * x1;
    const scalar_t x5 = qx * qy;
    const scalar_t x6 = qy * x0;

    out[0] = x3 * x2;  // (1-qx)(1-qy)(1-qz)
    out[1] = x4 * x2;  // qx(1-qy)(1-qz)
    out[2] = x5 * x2;  // qx*qy*(1-qz)
    out[3] = x6 * x2;  // (1-qx)*qy*(1-qz)
    out[4] = x3 * qz;  // (1-qx)(1-qy)*qz
    out[5] = x4 * qz;  // qx(1-qy)*qz
    out[6] = x5 * qz;  // qx*qy*qz
    out[7] = x6 * qz;  // (1-qx)*qy*qz
}

template <typename scalar_t, typename accumulator_t>
static __host__ __device__ void cu_hex8_kelvin_voigt_newmark_apply_adj(const scalar_t                      k,
                                                                       const scalar_t                      K,
                                                                       const scalar_t                      eta,
                                                                       const scalar_t                      rho,
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
                                                                       const scalar_t *const SFEM_RESTRICT ax,
                                                                       const scalar_t *const SFEM_RESTRICT ay,
                                                                       const scalar_t *const SFEM_RESTRICT az,
                                                                       accumulator_t *const SFEM_RESTRICT  outx,
                                                                       accumulator_t *const SFEM_RESTRICT  outy,
                                                                       accumulator_t *const SFEM_RESTRICT  outz) {
    const scalar_t denom        = jacobian_determinant;
    scalar_t       disp_grad[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    scalar_t       velo_grad[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    scalar_t       acce_vec[3]  = {0, 0, 0};
    assert(denom > 0);
    {
        scalar_t temp_u[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        scalar_t grad[8];

        cu_hex8_kv_ref_shape_grad_x(qx, qy, qz, grad);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const scalar_t g = grad[i];
            temp_u[0] += ux[i] * g;
            temp_u[3] += uy[i] * g;
            temp_u[6] += uz[i] * g;
        }

        cu_hex8_kv_ref_shape_grad_y(qx, qy, qz, grad);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const scalar_t g = grad[i];
            temp_u[1] += ux[i] * g;
            temp_u[4] += uy[i] * g;
            temp_u[7] += uz[i] * g;
        }

        cu_hex8_kv_ref_shape_grad_z(qx, qy, qz, grad);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const scalar_t g = grad[i];
            temp_u[2] += ux[i] * g;
            temp_u[5] += uy[i] * g;
            temp_u[8] += uz[i] * g;
        }
        for (int i = 0; i < 3; i++) {
#pragma unroll
            for (int j = 0; j < 3; j++) {
#pragma unroll
                for (int k = 0; k < 3; k++) {
                    disp_grad[i * 3 + j] += temp_u[i * 3 + k] * adjugate[k * 3 + j];
                    assert(disp_grad[i * 3 + j] == disp_grad[i * 3 + j]);
                }
            }
        }
    }

    {
        scalar_t temp_v[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        scalar_t grad[8];

        cu_hex8_kv_ref_shape_grad_x(qx, qy, qz, grad);

#pragma unroll
        for (int i = 0; i < 8; i++) {
            const scalar_t g = grad[i];
            temp_v[0] += vx[i] * g;
            temp_v[3] += vy[i] * g;
            temp_v[6] += vz[i] * g;
        }

        cu_hex8_kv_ref_shape_grad_y(qx, qy, qz, grad);

#pragma unroll
        for (int i = 0; i < 8; i++) {
            const scalar_t g = grad[i];
            temp_v[1] += vx[i] * g;
            temp_v[4] += vy[i] * g;
            temp_v[7] += vz[i] * g;
        }

        cu_hex8_kv_ref_shape_grad_z(qx, qy, qz, grad);

#pragma unroll
        for (int i = 0; i < 8; i++) {
            const scalar_t g = grad[i];
            temp_v[2] += vx[i] * g;
            temp_v[5] += vy[i] * g;
            temp_v[8] += vz[i] * g;
        }
        for (int i = 0; i < 3; i++) {
#pragma unroll
            for (int j = 0; j < 3; j++) {
#pragma unroll
                for (int k = 0; k < 3; k++) {
                    velo_grad[i * 3 + j] += temp_v[i * 3 + k] * adjugate[k * 3 + j];
                    assert(velo_grad[i * 3 + j] == velo_grad[i * 3 + j]);
                }
            }
        }
    }

    scalar_t *K_tXJinv_t = disp_grad;
    {
        const scalar_t x0 = (1.0 / 2.0) * k;
        const scalar_t x1 = x0 * (disp_grad[1] + disp_grad[3]);
        const scalar_t x2 = x0 * (disp_grad[2] + disp_grad[6]);
        const scalar_t x3 = 3 * k;
        const scalar_t x4 = (3 * K - k) * (disp_grad[0] + disp_grad[4] + disp_grad[8]);
        const scalar_t x5 = (1.0 / 3.0) * disp_grad[0] * x3 + (1.0 / 3.0) * x4;
        const scalar_t x6 = x0 * (disp_grad[5] + disp_grad[7]);
        const scalar_t x7 = (1.0 / 3.0) * disp_grad[4] * x3 + (1.0 / 3.0) * x4;
        const scalar_t x8 = (1.0 / 3.0) * disp_grad[8] * x3 + (1.0 / 3.0) * x4;
        K_tXJinv_t[0]     = adjugate[0] * x5 + adjugate[1] * x1 + adjugate[2] * x2;
        K_tXJinv_t[1]     = adjugate[3] * x5 + adjugate[4] * x1 + adjugate[5] * x2;
        K_tXJinv_t[2]     = adjugate[6] * x5 + adjugate[7] * x1 + adjugate[8] * x2;
        K_tXJinv_t[3]     = adjugate[0] * x1 + adjugate[1] * x7 + adjugate[2] * x6;
        K_tXJinv_t[4]     = adjugate[3] * x1 + adjugate[4] * x7 + adjugate[5] * x6;
        K_tXJinv_t[5]     = adjugate[6] * x1 + adjugate[7] * x7 + adjugate[8] * x6;
        K_tXJinv_t[6]     = adjugate[0] * x2 + adjugate[1] * x6 + adjugate[2] * x8;
        K_tXJinv_t[7]     = adjugate[3] * x2 + adjugate[4] * x6 + adjugate[5] * x8;
        K_tXJinv_t[8]     = adjugate[6] * x2 + adjugate[7] * x6 + adjugate[8] * x8;
    }

    scalar_t *C_tXJinv_t = velo_grad;
    {
        const scalar_t x0 = (1.0 / 2.0) * velo_grad[1] + (1.0 / 2.0) * velo_grad[3];
        const scalar_t x1 = (1.0 / 2.0) * velo_grad[2] + (1.0 / 2.0) * velo_grad[6];
        const scalar_t x2 = 0.33333333333333331 * velo_grad[4];
        const scalar_t x3 = 0.33333333333333331 * velo_grad[8];
        const scalar_t x4 = -0.66666666666666674 * velo_grad[0] + x2 + x3;
        const scalar_t x5 = (1.0 / 2.0) * velo_grad[5] + (1.0 / 2.0) * velo_grad[7];
        const scalar_t x6 = 0.33333333333333331 * velo_grad[0];
        const scalar_t x7 = -0.66666666666666674 * velo_grad[4] + x3 + x6;
        const scalar_t x8 = -0.66666666666666674 * velo_grad[8] + x2 + x6;
        C_tXJinv_t[0]     = eta * (-adjugate[0] * x4 + adjugate[1] * x0 + adjugate[2] * x1);
        C_tXJinv_t[1]     = eta * (-adjugate[3] * x4 + adjugate[4] * x0 + adjugate[5] * x1);
        C_tXJinv_t[2]     = eta * (-adjugate[6] * x4 + adjugate[7] * x0 + adjugate[8] * x1);
        C_tXJinv_t[3]     = eta * (adjugate[0] * x0 - adjugate[1] * x7 + adjugate[2] * x5);
        C_tXJinv_t[4]     = eta * (adjugate[3] * x0 - adjugate[4] * x7 + adjugate[5] * x5);
        C_tXJinv_t[5]     = eta * (adjugate[6] * x0 - adjugate[7] * x7 + adjugate[8] * x5);
        C_tXJinv_t[6]     = eta * (adjugate[0] * x1 + adjugate[1] * x5 - adjugate[2] * x8);
        C_tXJinv_t[7]     = eta * (adjugate[3] * x1 + adjugate[4] * x5 - adjugate[5] * x8);
        C_tXJinv_t[8]     = eta * (adjugate[6] * x1 + adjugate[7] * x5 - adjugate[8] * x8);
    }

    // Scale by quadrature weight and combine K and C
    scalar_t P_tXJinv_t[9];
    for (int i = 0; i < 9; i++) {
        P_tXJinv_t[i] = (K_tXJinv_t[i] + C_tXJinv_t[i]) * (qw / denom);
        assert(P_tXJinv_t[i] == P_tXJinv_t[i]);
    }

    {
        scalar_t grad[8];
        cu_hex8_kv_ref_shape_grad_x(qx, qy, qz, grad);

#pragma unroll
        for (int i = 0; i < 8; i++) {
            scalar_t g = grad[i];
            outx[i] += P_tXJinv_t[0] * g;
            outy[i] += P_tXJinv_t[3] * g;
            outz[i] += P_tXJinv_t[6] * g;
        }

        cu_hex8_kv_ref_shape_grad_y(qx, qy, qz, grad);

#pragma unroll
        for (int i = 0; i < 8; i++) {
            scalar_t g = grad[i];
            outx[i] += P_tXJinv_t[1] * g;
            outy[i] += P_tXJinv_t[4] * g;
            outz[i] += P_tXJinv_t[7] * g;
        }

        cu_hex8_kv_ref_shape_grad_z(qx, qy, qz, grad);

#pragma unroll
        for (int i = 0; i < 8; i++) {
            scalar_t g = grad[i];
            outx[i] += P_tXJinv_t[2] * g;
            outy[i] += P_tXJinv_t[5] * g;
            outz[i] += P_tXJinv_t[8] * g;
        }
    }

#pragma unroll
    {
        // Inertia contribution: interpolate acceleration, then distribute with shape values
        scalar_t shape[8];
        cu_hex8_kv_ref_shape_fun(qx, qy, qz, shape);

        for (int i = 0; i < 8; i++) {
            const scalar_t Ni = shape[i];
            acce_vec[0] += ax[i] * Ni;
            acce_vec[1] += ay[i] * Ni;
            acce_vec[2] += az[i] * Ni;
        }

        const scalar_t mscale = rho * denom * qw;  // rho * detJ * qw

        for (int i = 0; i < 8; i++) {
            const scalar_t Ni = shape[i];
            outx[i] += mscale * acce_vec[0] * Ni;
            outy[i] += mscale * acce_vec[1] * Ni;
            outz[i] += mscale * acce_vec[2] * Ni;
        }
    }

#ifndef NDEBUG
    for (int i = 0; i < 8; i++) {
        assert(outx[i] == outx[i]);
        assert(outy[i] == outy[i]);
        assert(outz[i] == outz[i]);
    }
#endif
}

template <typename scalar_t>
static __host__ __device__ void cu_sshex8_kelvin_voigt_newmark_diag(const scalar_t                      k,
                                                                    const scalar_t                      K,
                                                                    const scalar_t                      eta,
                                                                    const scalar_t                      rho,
                                                                    const scalar_t                      dt,
                                                                    const scalar_t                      gamma,
                                                                    const scalar_t                      beta,
                                                                    const scalar_t *const SFEM_RESTRICT adjugate,
                                                                    const scalar_t                      jacobian_determinant,
                                                                    scalar_t *const SFEM_RESTRICT       element_diag) {
    const scalar_t x0   = POW2(adjugate[2]);
    const scalar_t x1   = POW2(dt);
    const scalar_t x2   = beta * x1;
    const scalar_t x3   = k * x2;
    const scalar_t x4   = 0.055555555555555552 * x3;
    const scalar_t x5   = x0 * x4;
    const scalar_t x6   = eta * gamma;
    const scalar_t x7   = 0.055555555555555552 * x6;
    const scalar_t x8   = x0 * x7;
    const scalar_t x9   = dt * x8;
    const scalar_t x10  = 0.083333333333333329 * x3;
    const scalar_t x11  = adjugate[2] * x10;
    const scalar_t x12  = adjugate[8] * x11;
    const scalar_t x13  = 0.083333333333333329 * x6;
    const scalar_t x14  = adjugate[2] * x13;
    const scalar_t x15  = adjugate[8] * x14;
    const scalar_t x16  = dt * x15;
    const scalar_t x17  = x12 + x16 + x5 + x9;
    const scalar_t x18  = adjugate[5] * x11;
    const scalar_t x19  = adjugate[5] * x14;
    const scalar_t x20  = dt * x19;
    const scalar_t x21  = x18 + x20;
    const scalar_t x22  = 0.083333333333333315 * x3;
    const scalar_t x23  = adjugate[5] * adjugate[8];
    const scalar_t x24  = x22 * x23;
    const scalar_t x25  = dt * x6;
    const scalar_t x26  = 0.083333333333333315 * x25;
    const scalar_t x27  = x23 * x26;
    const scalar_t x28  = x24 + x27;
    const scalar_t x29  = x17 + x21 + x28;
    const scalar_t x30  = POW2(adjugate[1]);
    const scalar_t x31  = x30 * x4;
    const scalar_t x32  = x30 * x7;
    const scalar_t x33  = dt * x32;
    const scalar_t x34  = adjugate[1] * adjugate[7];
    const scalar_t x35  = x10 * x34;
    const scalar_t x36  = x13 * x34;
    const scalar_t x37  = dt * x36;
    const scalar_t x38  = x31 + x33 + x35 + x37;
    const scalar_t x39  = adjugate[1] * adjugate[4];
    const scalar_t x40  = x10 * x39;
    const scalar_t x41  = x13 * x39;
    const scalar_t x42  = dt * x41;
    const scalar_t x43  = x40 + x42;
    const scalar_t x44  = adjugate[4] * adjugate[7];
    const scalar_t x45  = x22 * x44;
    const scalar_t x46  = x26 * x44;
    const scalar_t x47  = x45 + x46;
    const scalar_t x48  = x38 + x43 + x47;
    const scalar_t x49  = POW2(adjugate[0]);
    const scalar_t x50  = 0.07407407407407407 * x49;
    const scalar_t x51  = x3 * x50;
    const scalar_t x52  = x50 * x6;
    const scalar_t x53  = dt * x52;
    const scalar_t x54  = 0.1111111111111111 * K;
    const scalar_t x55  = x2 * x54;
    const scalar_t x56  = x49 * x55;
    const scalar_t x57  = POW2(adjugate[3]);
    const scalar_t x58  = 0.074074074074074042 * x3;
    const scalar_t x59  = x57 * x58;
    const scalar_t x60  = POW2(adjugate[6]);
    const scalar_t x61  = x58 * x60;
    const scalar_t x62  = dt * x57;
    const scalar_t x63  = 0.074074074074074042 * x6;
    const scalar_t x64  = x62 * x63;
    const scalar_t x65  = dt * x60;
    const scalar_t x66  = x63 * x65;
    const scalar_t x67  = adjugate[0] * adjugate[6];
    const scalar_t x68  = K * x2;
    const scalar_t x69  = 0.16666666666666666 * x68;
    const scalar_t x70  = x67 * x69;
    const scalar_t x71  = 0.11111111111111112 * x3;
    const scalar_t x72  = x67 * x71;
    const scalar_t x73  = 0.11111111111111112 * x6;
    const scalar_t x74  = x67 * x73;
    const scalar_t x75  = dt * x74;
    const scalar_t x76  = x55 * x57 + x55 * x60;
    const scalar_t x77  = 0.037037037037037035 * POW2(jacobian_determinant) * rho;
    const scalar_t x78  = POW2(adjugate[5]);
    const scalar_t x79  = POW2(adjugate[8]);
    const scalar_t x80  = dt * x7;
    const scalar_t x81  = x4 * x78 + x4 * x79 + x77 + x78 * x80 + x79 * x80;
    const scalar_t x82  = POW2(adjugate[4]);
    const scalar_t x83  = POW2(adjugate[7]);
    const scalar_t x84  = x4 * x82 + x4 * x83 + x80 * x82 + x80 * x83;
    const scalar_t x85  = x81 + x84;
    const scalar_t x86  = x51 + x53 + x56 + x59 + x61 + x64 + x66 + x70 + x72 + x75 + x76 + x85;
    const scalar_t x87  = adjugate[0] * adjugate[3];
    const scalar_t x88  = x69 * x87;
    const scalar_t x89  = x71 * x87;
    const scalar_t x90  = x73 * x87;
    const scalar_t x91  = dt * x90;
    const scalar_t x92  = x88 + x89 + x91;
    const scalar_t x93  = adjugate[3] * adjugate[6];
    const scalar_t x94  = 0.16666666666666663 * x68;
    const scalar_t x95  = x93 * x94;
    const scalar_t x96  = 0.1111111111111111 * x93;
    const scalar_t x97  = x3 * x96;
    const scalar_t x98  = x25 * x96;
    const scalar_t x99  = x95 + x97 + x98;
    const scalar_t x100 = 1 / (beta * jacobian_determinant);
    const scalar_t x101 = x100 / x1;
    const scalar_t x102 = 1.0 / dt;
    const scalar_t x103 = x69 * x93;
    const scalar_t x104 = x71 * x93;
    const scalar_t x105 = dt * x73;
    const scalar_t x106 = x105 * x93;
    const scalar_t x107 = x10 * x23;
    const scalar_t x108 = dt * x13;
    const scalar_t x109 = x108 * x23;
    const scalar_t x110 = x107 + x109 + x81;
    const scalar_t x111 = x10 * x44;
    const scalar_t x112 = x108 * x44;
    const scalar_t x113 = x111 + x112 + x84;
    const scalar_t x114 = 0.07407407407407407 * x3;
    const scalar_t x115 = 0.07407407407407407 * x6;
    const scalar_t x116 = x114 * x57 + x114 * x60 + x115 * x62 + x115 * x65 + x76;
    const scalar_t x117 = x103 + x104 + x106 + x110 + x113 + x116;
    const scalar_t x118 = beta * dt;
    const scalar_t x119 = k * x118;
    const scalar_t x120 = 0.083333333333333329 * x119;
    const scalar_t x121 = adjugate[2] * x120;
    const scalar_t x122 = adjugate[5] * x121;
    const scalar_t x123 = x122 + x19;
    const scalar_t x124 = adjugate[8] * x121;
    const scalar_t x125 = x124 + x15;
    const scalar_t x126 = x123 + x125;
    const scalar_t x127 = -0.055555555555555552 * beta * dt * k * x0 - 0.055555555555555552 * eta * gamma * x0 + x126;
    const scalar_t x128 = x120 * x39;
    const scalar_t x129 = x128 + x41;
    const scalar_t x130 = x120 * x34;
    const scalar_t x131 = x130 + x36;
    const scalar_t x132 = x129 + x131;
    const scalar_t x133 = -0.055555555555555552 * beta * dt * k * x30 - 0.055555555555555552 * eta * gamma * x30 + x132;
    const scalar_t x134 = 0.16666666666666666 * K * x118;
    const scalar_t x135 = x134 * x87;
    const scalar_t x136 = 0.11111111111111112 * x119;
    const scalar_t x137 = x136 * x87;
    const scalar_t x138 = x135 + x137 + x90;
    const scalar_t x139 = x134 * x67;
    const scalar_t x140 = x136 * x67;
    const scalar_t x141 = x139 + x140 + x74;
    const scalar_t x142 = x138 + x141;
    const scalar_t x143 = x100 * x102;
    const scalar_t x144 = 0.055555555555555552 * x119;
    const scalar_t x145 = x0 * x144 + x8;
    const scalar_t x146 = x123 - x124 + x145 - x15;
    const scalar_t x147 = x144 * x30 + x32;
    const scalar_t x148 = x129 - x130 + x147 - x36;
    const scalar_t x149 = -x107 - x109;
    const scalar_t x150 = -x111 - x112;
    const scalar_t x151 = x118 * x54;
    const scalar_t x152 = x119 * x50 + x151 * x49 + x52;
    const scalar_t x153 = x102 * (-x103 - x104 - x106 + x116 + x149 + x150 + x85) + x152;
    const scalar_t x154 = -x24 - x27;
    const scalar_t x155 = -x18 - x20;
    const scalar_t x156 = x154 + x155 + x17;
    const scalar_t x157 = -x45 - x46;
    const scalar_t x158 = -x40 - x42;
    const scalar_t x159 = x157 + x158 + x38;
    const scalar_t x160 = -x95 - x97 - x98;
    const scalar_t x161 = -x88 - x89 - x91;
    const scalar_t x162 = -x12 - x16 + x5 + x9;
    const scalar_t x163 = x154 + x162 + x21;
    const scalar_t x164 = x31 + x33 - x35 - x37;
    const scalar_t x165 = x157 + x164 + x43;
    const scalar_t x166 = x51 + x53 + x56 + x59 + x61 + x64 + x66 - x70 - x72 - x75 + x76 + x85;
    const scalar_t x167 = x145 + x147;
    const scalar_t x168 = -x122 + x125 - x19;
    const scalar_t x169 = -x128 + x131 - x41;
    const scalar_t x170 = x155 + x162 + x28;
    const scalar_t x171 = x158 + x164 + x47;
    const scalar_t x172 = x4 * x49;
    const scalar_t x173 = x49 * x7;
    const scalar_t x174 = dt * x173;
    const scalar_t x175 = x10 * x67;
    const scalar_t x176 = x13 * x67;
    const scalar_t x177 = dt * x176;
    const scalar_t x178 = x172 + x174 + x175 + x177;
    const scalar_t x179 = x10 * x87;
    const scalar_t x180 = x13 * x87;
    const scalar_t x181 = dt * x180;
    const scalar_t x182 = x179 + x181;
    const scalar_t x183 = x22 * x93;
    const scalar_t x184 = x26 * x93;
    const scalar_t x185 = x183 + x184;
    const scalar_t x186 = x178 + x182 + x185;
    const scalar_t x187 = x114 * x30;
    const scalar_t x188 = x115 * x30;
    const scalar_t x189 = dt * x188;
    const scalar_t x190 = x30 * x55;
    const scalar_t x191 = x58 * x82;
    const scalar_t x192 = x58 * x83;
    const scalar_t x193 = dt * x63;
    const scalar_t x194 = x193 * x82;
    const scalar_t x195 = x193 * x83;
    const scalar_t x196 = x34 * x69;
    const scalar_t x197 = x34 * x71;
    const scalar_t x198 = x34 * x73;
    const scalar_t x199 = dt * x198;
    const scalar_t x200 = x55 * x82 + x55 * x83;
    const scalar_t x201 = x4 * x57 + x4 * x60 + x57 * x80 + x60 * x80;
    const scalar_t x202 = x201 + x81;
    const scalar_t x203 = x187 + x189 + x190 + x191 + x192 + x194 + x195 + x196 + x197 + x199 + x200 + x202;
    const scalar_t x204 = x39 * x69;
    const scalar_t x205 = x39 * x71;
    const scalar_t x206 = x39 * x73;
    const scalar_t x207 = dt * x206;
    const scalar_t x208 = x204 + x205 + x207;
    const scalar_t x209 = x44 * x94;
    const scalar_t x210 = 0.1111111111111111 * x44;
    const scalar_t x211 = x210 * x3;
    const scalar_t x212 = x210 * x25;
    const scalar_t x213 = x209 + x211 + x212;
    const scalar_t x214 = x44 * x69;
    const scalar_t x215 = x44 * x71;
    const scalar_t x216 = x105 * x44;
    const scalar_t x217 = x10 * x93;
    const scalar_t x218 = x108 * x93;
    const scalar_t x219 = x201 + x217 + x218;
    const scalar_t x220 = dt * x115;
    const scalar_t x221 = x114 * x82 + x114 * x83 + x200 + x220 * x82 + x220 * x83;
    const scalar_t x222 = x110 + x214 + x215 + x216 + x219 + x221;
    const scalar_t x223 = x120 * x87;
    const scalar_t x224 = x180 + x223;
    const scalar_t x225 = x120 * x67;
    const scalar_t x226 = x176 + x225;
    const scalar_t x227 = x224 + x226;
    const scalar_t x228 = -0.055555555555555552 * beta * dt * k * x49 - 0.055555555555555552 * eta * gamma * x49 + x227;
    const scalar_t x229 = x134 * x39;
    const scalar_t x230 = x136 * x39;
    const scalar_t x231 = x206 + x229 + x230;
    const scalar_t x232 = x134 * x34;
    const scalar_t x233 = x136 * x34;
    const scalar_t x234 = x198 + x232 + x233;
    const scalar_t x235 = x231 + x234;
    const scalar_t x236 = -x176 + x224 - x225;
    const scalar_t x237 = -x217 - x218;
    const scalar_t x238 = 0.07407407407407407 * x119;
    const scalar_t x239 = x144 * x49 + x173;
    const scalar_t x240 = x151 * x30 + x188 + x238 * x30 + x239;
    const scalar_t x241 = x102 * (x149 + x202 - x214 - x215 - x216 + x221 + x237) + x240;
    const scalar_t x242 = -x183 - x184;
    const scalar_t x243 = -x209 - x211 - x212 + x242;
    const scalar_t x244 = -x179 - x181;
    const scalar_t x245 = -x204 - x205 - x207 + x244;
    const scalar_t x246 = x172 + x174 - x175 - x177;
    const scalar_t x247 = x182 + x246;
    const scalar_t x248 = x187 + x189 + x190 + x191 + x192 + x194 + x195 - x196 - x197 - x199 + x200 + x202;
    const scalar_t x249 = -x180 - x223 + x226;
    const scalar_t x250 = x185 + x246;
    const scalar_t x251 = x0 * x114;
    const scalar_t x252 = x0 * x115;
    const scalar_t x253 = dt * x252;
    const scalar_t x254 = x0 * x55;
    const scalar_t x255 = x58 * x78;
    const scalar_t x256 = x58 * x79;
    const scalar_t x257 = x193 * x78;
    const scalar_t x258 = x193 * x79;
    const scalar_t x259 = adjugate[2] * x69;
    const scalar_t x260 = adjugate[8] * x259;
    const scalar_t x261 = adjugate[2] * x71;
    const scalar_t x262 = adjugate[8] * x261;
    const scalar_t x263 = adjugate[2] * x73;
    const scalar_t x264 = adjugate[8] * x263;
    const scalar_t x265 = dt * x264;
    const scalar_t x266 = x55 * x78 + x55 * x79 + x77;
    const scalar_t x267 = x201 + x84;
    const scalar_t x268 = x251 + x253 + x254 + x255 + x256 + x257 + x258 + x260 + x262 + x265 + x266 + x267;
    const scalar_t x269 = adjugate[5] * x259;
    const scalar_t x270 = adjugate[5] * x261;
    const scalar_t x271 = adjugate[5] * x263;
    const scalar_t x272 = dt * x271;
    const scalar_t x273 = x269 + x270 + x272;
    const scalar_t x274 = x23 * x94;
    const scalar_t x275 = 0.1111111111111111 * x23;
    const scalar_t x276 = x275 * x3;
    const scalar_t x277 = x25 * x275;
    const scalar_t x278 = x274 + x276 + x277;
    const scalar_t x279 = x23 * x69;
    const scalar_t x280 = x23 * x71;
    const scalar_t x281 = x105 * x23;
    const scalar_t x282 = x114 * x78 + x114 * x79 + x220 * x78 + x220 * x79 + x266;
    const scalar_t x283 = x113 + x219 + x279 + x280 + x281 + x282;
    const scalar_t x284 = adjugate[2] * x134;
    const scalar_t x285 = adjugate[5] * x284;
    const scalar_t x286 = adjugate[2] * x136;
    const scalar_t x287 = adjugate[5] * x286;
    const scalar_t x288 = x271 + x285 + x287;
    const scalar_t x289 = adjugate[8] * x284;
    const scalar_t x290 = adjugate[8] * x286;
    const scalar_t x291 = x264 + x289 + x290;
    const scalar_t x292 = x288 + x291;
    const scalar_t x293 = x0 * x151 + x0 * x238 + x239 + x252;
    const scalar_t x294 = x102 * (x150 + x237 + x267 - x279 - x280 - x281 + x282) + x293;
    const scalar_t x295 = x242 - x274 - x276 - x277;
    const scalar_t x296 = x244 - x269 - x270 - x272;
    const scalar_t x297 = x251 + x253 + x254 + x255 + x256 + x257 + x258 - x260 - x262 - x265 + x266 + x267;
    element_diag[0]     = x101 * (x29 + x48 + x86 + x92 + x99);
    element_diag[1]     = x143 * (0.1111111111111111 * K * beta * dt * x49 + 0.07407407407407407 * beta * dt * k * x49 +
                              0.07407407407407407 * eta * gamma * x49 + x102 * x117 - x127 - x133 - x142);
    element_diag[2]     = x143 * (x138 - x139 - x140 + x146 + x148 + x153 - x74);
    element_diag[3]     = x101 * (x156 + x159 + x160 + x161 + x86);
    element_diag[4]     = x101 * (x160 + x163 + x165 + x166 + x92);
    element_diag[5]     = x143 * (-x135 - x137 + x141 + x153 + x167 + x168 + x169 - x90);
    element_diag[6]     = x143 * (x102 * x117 + x126 + x132 + x142 + x152 + x167);
    element_diag[7]     = x101 * (x161 + x166 + x170 + x171 + x99);
    element_diag[8]     = x101 * (x186 + x203 + x208 + x213 + x29);
    element_diag[9]     = x143 * (0.1111111111111111 * K * beta * dt * x30 + 0.07407407407407407 * beta * dt * k * x30 +
                              0.07407407407407407 * eta * gamma * x30 + x102 * x222 - x127 - x228 - x235);
    element_diag[10]    = x143 * (x146 - x198 + x231 - x232 - x233 + x236 + x241);
    element_diag[11]    = x101 * (x156 + x178 + x203 + x243 + x245);
    element_diag[12]    = x101 * (x163 + x208 + x243 + x247 + x248);
    element_diag[13]    = x143 * (x145 + x168 - x206 - x229 - x230 + x234 + x241 + x249);
    element_diag[14]    = x143 * (x102 * x222 + x126 + x145 + x227 + x235 + x240);
    element_diag[15]    = x101 * (x170 + x213 + x245 + x248 + x250);
    element_diag[16]    = x101 * (x186 + x268 + x273 + x278 + x48);
    element_diag[17]    = x143 * (0.1111111111111111 * K * beta * dt * x0 + 0.07407407407407407 * beta * dt * k * x0 +
                               0.07407407407407407 * eta * gamma * x0 + x102 * x283 - x133 - x228 - x292);
    element_diag[18]    = x143 * (x148 + x236 - x264 + x288 - x289 - x290 + x294);
    element_diag[19]    = x101 * (x159 + x178 + x268 + x295 + x296);
    element_diag[20]    = x101 * (x165 + x247 + x273 + x295 + x297);
    element_diag[21]    = x143 * (x147 + x169 + x249 - x271 - x285 - x287 + x291 + x294);
    element_diag[22]    = x143 * (x102 * x283 + x132 + x147 + x227 + x292 + x293);
    element_diag[23]    = x101 * (x171 + x250 + x278 + x296 + x297);
}

#endif  // CU_HEX8_KELVIN_VOIGT_NEWMARK_INLINE_HPP