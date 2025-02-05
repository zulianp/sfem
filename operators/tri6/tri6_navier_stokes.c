#include "tri6_navier_stokes.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"

static SFEM_INLINE void tri6_momentum_lhs_scalar_kernel(const real_t px0,
                                                        const real_t px1,
                                                        const real_t px2,
                                                        const real_t py0,
                                                        const real_t py1,
                                                        const real_t py2,
                                                        const real_t dt,
                                                        const real_t nu,
                                                        real_t *const SFEM_RESTRICT
                                                            element_matrix) {
    const real_t x0 = px0 - px1;
    const real_t x1 = py0 - py2;
    const real_t x2 = x0 * x1;
    const real_t x3 = px0 - px2;
    const real_t x4 = py0 - py1;
    const real_t x5 = x3 * x4;
    const real_t x6 = x2 - x5;
    const real_t x7 = 1.0 / x6;
    const real_t x8 = pow(x6, 2);
    const real_t x9 = (1.0 / 60.0) * x8;
    const real_t x10 = x0 * x3;
    const real_t x11 = x1 * x4;
    const real_t x12 = pow(x3, 2);
    const real_t x13 = pow(x1, 2);
    const real_t x14 = x12 + x13;
    const real_t x15 = pow(x0, 2) + pow(x4, 2);
    const real_t x16 = dt * nu;
    const real_t x17 = (1.0 / 2.0) * x16;
    const real_t x18 = (1.0 / 360.0) * pow(x6, 2);
    const real_t x19 = x10 + x11;
    const real_t x20 = -x12 - x13 + x19;
    const real_t x21 = (1.0 / 6.0) * x16;
    const real_t x22 = x7 * (-x18 - x20 * x21);
    const real_t x23 = -x18;
    const real_t x24 = -x10 - x11 + x15;
    const real_t x25 = x7 * (x21 * x24 + x23);
    const real_t x26 = 2 * x16 / (3 * x2 - 3 * x5);
    const real_t x27 = x20 * x26;
    const real_t x28 = -1.0 / 90.0 * x2 + (1.0 / 90.0) * x3 * x4;
    const real_t x29 = -x24 * x26;
    const real_t x30 = x7 * (x19 * x21 + x23);
    const real_t x31 = -x19 * x26;
    const real_t x32 = (4.0 / 45.0) * x7 * (15 * x16 * (x14 + x24) + x8);
    const real_t x33 = -x8;
    const real_t x34 = 30 * x16;
    const real_t x35 = (2.0 / 45.0) * x7;
    const real_t x36 = x35 * (-x24 * x34 - x33);
    const real_t x37 = x35 * (-x19 * x34 - x33);
    const real_t x38 = x35 * (x20 * x34 + x8);
    element_matrix[0] = x7 * (x17 * (-2 * x10 - 2 * x11 + x14 + x15) + x9);
    element_matrix[1] = x22;
    element_matrix[2] = x25;
    element_matrix[3] = x27;
    element_matrix[4] = x28;
    element_matrix[5] = x29;
    element_matrix[6] = x22;
    element_matrix[7] = x7 * (x14 * x17 + x9);
    element_matrix[8] = x30;
    element_matrix[9] = x27;
    element_matrix[10] = x31;
    element_matrix[11] = x28;
    element_matrix[12] = x25;
    element_matrix[13] = x30;
    element_matrix[14] = x7 * (x15 * x17 + x9);
    element_matrix[15] = x28;
    element_matrix[16] = x31;
    element_matrix[17] = x29;
    element_matrix[18] = x27;
    element_matrix[19] = x27;
    element_matrix[20] = x28;
    element_matrix[21] = x32;
    element_matrix[22] = x36;
    element_matrix[23] = x37;
    element_matrix[24] = x28;
    element_matrix[25] = x31;
    element_matrix[26] = x31;
    element_matrix[27] = x36;
    element_matrix[28] = x32;
    element_matrix[29] = x38;
    element_matrix[30] = x29;
    element_matrix[31] = x28;
    element_matrix[32] = x29;
    element_matrix[33] = x37;
    element_matrix[34] = x38;
    element_matrix[35] = x32;
}

static SFEM_INLINE void tri6_momentum_rhs_kernel(const real_t px0,
                                                 const real_t px1,
                                                 const real_t px2,
                                                 const real_t py0,
                                                 const real_t py1,
                                                 const real_t py2,
                                                 const real_t dt,
                                                 const real_t nu,
                                                 real_t *const SFEM_RESTRICT u,
                                                 real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = 6 * u[0];
    const real_t x1 = 4 * u[4];
    const real_t x2 = px0 - px1;
    const real_t x3 = py0 - py2;
    const real_t x4 = x2 * x3;
    const real_t x5 = px0 - px2;
    const real_t x6 = py0 - py1;
    const real_t x7 = x5 * x6;
    const real_t x8 = x4 - x7;
    const real_t x9 = 7 * pow(x8, 3);
    const real_t x10 = pow(x5, 2);
    const real_t x11 = u[1] * x10;
    const real_t x12 = pow(x6, 2);
    const real_t x13 = u[2] * x12;
    const real_t x14 = pow(x3, 2);
    const real_t x15 = u[1] * x14;
    const real_t x16 = pow(x2, 2);
    const real_t x17 = u[2] * x16;
    const real_t x18 = u[0] * x16;
    const real_t x19 = u[0] * x10;
    const real_t x20 = u[0] * x12;
    const real_t x21 = u[0] * x14;
    const real_t x22 = u[1] * x5;
    const real_t x23 = x2 * x22;
    const real_t x24 = u[1] * x3;
    const real_t x25 = x24 * x6;
    const real_t x26 = u[2] * x2;
    const real_t x27 = x26 * x5;
    const real_t x28 = u[2] * x6;
    const real_t x29 = x28 * x3;
    const real_t x30 = x2 * x5;
    const real_t x31 = x3 * x6;
    const real_t x32 = 4 * u[3];
    const real_t x33 = u[3] * x6;
    const real_t x34 = x3 * x33;
    const real_t x35 = -x10 * x32 - x14 * x32 + x30 * x32 + 4 * x34;
    const real_t x36 = 4 * u[5];
    const real_t x37 = u[5] * x3;
    const real_t x38 = x37 * x6;
    const real_t x39 = -x12 * x36 - x16 * x36 + x30 * x36 + 4 * x38;
    const real_t x40 = dt * x8;
    const real_t x41 = 420 * nu;
    const real_t x42 = x40 * x41;
    const real_t x43 = pow(u[5], 2);
    const real_t x44 = x43 * x6;
    const real_t x45 = pow(u[3], 2);
    const real_t x46 = x45 * x6;
    const real_t x47 = 32 * x46;
    const real_t x48 = u[10] * x2;
    const real_t x49 = u[4] * x6;
    const real_t x50 = u[10] * x5;
    const real_t x51 = u[4] * x3;
    const real_t x52 = x48 + x49 - x50 - x51;
    const real_t x53 = 12 * u[0];
    const real_t x54 = pow(u[1], 2);
    const real_t x55 = 9 * x3;
    const real_t x56 = x54 * x55;
    const real_t x57 = pow(u[2], 2);
    const real_t x58 = 9 * x6;
    const real_t x59 = x57 * x58;
    const real_t x60 = u[6] * (x22 - x26);
    const real_t x61 = x3 * x43;
    const real_t x62 = u[0] * x6;
    const real_t x63 = u[6] * x5;
    const real_t x64 = u[0] * x3;
    const real_t x65 = u[6] * x2;
    const real_t x66 = x62 + x63 - x64 - x65;
    const real_t x67 = x3 * x45;
    const real_t x68 = 96 * u[3];
    const real_t x69 = 80 * u[5];
    const real_t x70 = u[9] * x5;
    const real_t x71 = 80 * u[3];
    const real_t x72 = u[11] * x2;
    const real_t x73 = 48 * u[0];
    const real_t x74 = 48 * u[5];
    const real_t x75 = u[9] * x2;
    const real_t x76 = 32 * u[3];
    const real_t x77 = u[11] * x26;
    const real_t x78 = u[11] * x5;
    const real_t x79 = 32 * u[5];
    const real_t x80 = x78 * x79;
    const real_t x81 = 32 * u[4];
    const real_t x82 = 24 * x65;
    const real_t x83 = 24 * x63;
    const real_t x84 = 20 * u[2];
    const real_t x85 = 20 * u[1];
    const real_t x86 = 16 * u[4];
    const real_t x87 = x72 * x86;
    const real_t x88 = 16 * u[1];
    const real_t x89 = x33 * x88;
    const real_t x90 = 16 * u[2];
    const real_t x91 = x51 * x90;
    const real_t x92 = 16 * u[9];
    const real_t x93 = x26 * x92;
    const real_t x94 = u[7] * x2;
    const real_t x95 = x86 * x94;
    const real_t x96 = 16 * u[5];
    const real_t x97 = u[8] * x5;
    const real_t x98 = x96 * x97;
    const real_t x99 = x70 * x96;
    const real_t x100 = u[1] * x28;
    const real_t x101 = 11 * u[8];
    const real_t x102 = 9 * u[1];
    const real_t x103 = u[7] * x5;
    const real_t x104 = 9 * u[0];
    const real_t x105 = u[0] * x5;
    const real_t x106 = 9 * u[8];
    const real_t x107 = x106 * x26;
    const real_t x108 = u[8] * x2;
    const real_t x109 = x1 * x103;
    const real_t x110 = x103 * x36;
    const real_t x111 = 9 * u[2];
    const real_t x112 = 9 * x2;
    const real_t x113 = u[0] * u[7];
    const real_t x114 = 9 * x22;
    const real_t x115 = u[7] * x114;
    const real_t x116 = 11 * u[2];
    const real_t x117 = 11 * x26;
    const real_t x118 = u[11] * x22;
    const real_t x119 = 16 * x118;
    const real_t x120 = 16 * u[3];
    const real_t x121 = x120 * x72;
    const real_t x122 = x49 * x88;
    const real_t x123 = x49 * x90;
    const real_t x124 = x37 * x90;
    const real_t x125 = x120 * x94;
    const real_t x126 = x86 * x97;
    const real_t x127 = x70 * x86;
    const real_t x128 = x50 * x85;
    const real_t x129 = x78 * x81;
    const real_t x130 = u[9] * x22;
    const real_t x131 = u[5] * x6;
    const real_t x132 = 32 * u[2];
    const real_t x133 = 48 * u[3];
    const real_t x134 = 64 * u[5];
    const real_t x135 = 96 * u[5];
    const real_t x136 = pow(u[4], 2);
    const real_t x137 = 48 * x136;
    const real_t x138 = 48 * x48;
    const real_t x139 = u[3] * x3;
    const real_t x140 = 20 * u[3];
    const real_t x141 =
        u[3] * x138 - u[4] * x138 - x102 * x64 + x137 * x6 - x139 * x84 + x140 * x97 + x51 * x96;
    const real_t x142 = x137 * x3;
    const real_t x143 = x50 * x74;
    const real_t x144 = 20 * u[5];
    const real_t x145 = 16 * x49;
    const real_t x146 = u[2] * x62;
    const real_t x147 = 48 * u[4];
    const real_t x148 = x147 * x50;
    const real_t x149 = -u[3] * x145 + x131 * x85 - x142 - x143 - x144 * x94 + 9 * x146 + x148;
    const real_t x150 = 64 * u[3];
    const real_t x151 = -u[5] * x145 - x150 * x78;
    const real_t x152 = pow(x8, 2);
    const real_t x153 = dt * x152;
    const real_t x154 = (1.0 / 2520.0) * dt / x152;
    const real_t x155 = 7 * x152;
    const real_t x156 = x27 + x29;
    const real_t x157 = -x1 * x30 - x1 * x31 - x105 * x2 - x3 * x62;
    const real_t x158 = dt * x41;
    const real_t x159 = -x103 + x24;
    const real_t x160 = u[1] * x159;
    const real_t x161 = 48 * x61;
    const real_t x162 = 12 * x37 + 12 * x78;
    const real_t x163 = 120 * u[1];
    const real_t x164 = 120 * x94;
    const real_t x165 = u[1] * x51;
    const real_t x166 = x147 * x78;
    const real_t x167 = 48 * u[2];
    const real_t x168 = 32 * u[0];
    const real_t x169 = 24 * x103;
    const real_t x170 = 20 * u[0];
    const real_t x171 = 18 * u[0];
    const real_t x172 = 16 * u[0];
    const real_t x173 = 4 * x63;
    const real_t x174 = 11 * u[0];
    const real_t x175 = u[1] * x62;
    const real_t x176 = u[7] * x26;
    const real_t x177 = 20 * x77;
    const real_t x178 = u[1] * x50;
    const real_t x179 = x74 * x78;
    const real_t x180 = 32 * x136;
    const real_t x181 = x170 * x78 + x180 * x3 + x50 * x79 - x50 * x81;
    const real_t x182 = 48 * x46;
    const real_t x183 = x147 * x75;
    const real_t x184 = x133 * x75;
    const real_t x185 = -x150 * x50 - x182 - x183 + x184;
    const real_t x186 = pow(u[0], 2);
    const real_t x187 = x104 * x63 - x104 * x65 - x121 - x127 - x172 * x48 + x172 * x50 -
                        x186 * x55 + x186 * x58 + x87 + x99;
    const real_t x188 = dt / (2520 * x4 - 2520 * x7);
    const real_t x189 = x23 + x25;
    const real_t x190 = 12 * x33 + 12 * x75;
    const real_t x191 = -x108;
    const real_t x192 = x191 + x28;
    const real_t x193 = 78 * x192;
    const real_t x194 = 120 * u[2];
    const real_t x195 = 120 * x97;
    const real_t x196 = 24 * x108;
    const real_t x197 = 20 * x130;
    const real_t x198 = 18 * u[2];
    const real_t x199 = 18 * u[8];
    const real_t x200 = x33 * x96;
    const real_t x201 = 4 * x65;
    const real_t x202 = 9 * u[6];
    const real_t x203 = u[6] * x22;
    const real_t x204 = u[2] * x49;
    const real_t x205 = x170 * x70 + x172 * x33;
    const real_t x206 = x103 * x140 + x120 * x50 - x120 * x78 - x140 * x63;
    const real_t x207 = x134 * x48 + x161 + x166 - x179;
    const real_t x208 = 8 * u[3];
    const real_t x209 = -x2;
    const real_t x210 = -x3;
    const real_t x211 = -x5;
    const real_t x212 = -x6;
    const real_t x213 = x209 * x210 - x211 * x212;
    const real_t x214 = pow(x213, 2);
    const real_t x215 = 7 * x214;
    const real_t x216 = pow(x210, 2);
    const real_t x217 = 2 * u[3];
    const real_t x218 = pow(x209, 2);
    const real_t x219 = x12 * x217;
    const real_t x220 = 2 * u[4];
    const real_t x221 = x12 * x220;
    const real_t x222 = u[1] * x210;
    const real_t x223 = x209 * x5;
    const real_t x224 = 2 * x210;
    const real_t x225 = 2 * u[5];
    const real_t x226 = x105 * x209 - x131 * x224 + x210 * x62 - x217 * x223 + x220 * x223 -
                        x223 * x225 - x224 * x33 + x224 * x49;
    const real_t x227 = u[8] * x209;
    const real_t x228 = 12 * x3;
    const real_t x229 = x228 * x54;
    const real_t x230 = x210 * x43;
    const real_t x231 = u[9] * x209;
    const real_t x232 = 96 * u[4];
    const real_t x233 = u[10] * x209;
    const real_t x234 = u[11] * x209;
    const real_t x235 = u[3] * x210;
    const real_t x236 = u[6] * x209;
    const real_t x237 = 12 * u[1];
    const real_t x238 = x237 * x49;
    const real_t x239 = 12 * u[7];
    const real_t x240 = x22 * x239;
    const real_t x241 = 12 * x227;
    const real_t x242 = x209 * x239;
    const real_t x243 = 8 * u[0];
    const real_t x244 = x113 * x209;
    const real_t x245 = 8 * u[5];
    const real_t x246 = 5 * u[0];
    const real_t x247 = u[2] * x210;
    const real_t x248 = 5 * u[8];
    const real_t x249 = 4 * u[10];
    const real_t x250 = u[2] * x209;
    const real_t x251 = 4 * u[11];
    const real_t x252 = x22 * x251;
    const real_t x253 = 4 * x250;
    const real_t x254 = u[7] * x209;
    const real_t x255 = u[2] * x222;
    const real_t x256 = 8 * u[4];
    const real_t x257 = u[0] * x210;
    const real_t x258 = x237 * x33;
    const real_t x259 = 24 * u[9];
    const real_t x260 = 40 * u[0];
    const real_t x261 = u[1] * x6;
    const real_t x262 = 12 * u[2];
    const real_t x263 = -x133 * x49 + x261 * x36 + x262 * x33 + x28 * x36 - x33 * x79 + x36 * x62 +
                        x49 * x53 - x49 * x79;
    const real_t x264 = 12 * x186;
    const real_t x265 = x70 * x81;
    const real_t x266 = 4 * u[0];
    const real_t x267 = u[0] * x249;
    const real_t x268 = x70 * x79;
    const real_t x269 = x209 * x267 + x210 * x264 + x234 * x76 - x234 * x81 + x236 * x53 +
                        x264 * x6 - x265 + x266 * x50 + x268 + x53 * x63;
    const real_t x270 = dt * x213;
    const real_t x271 = (1.0 / 630.0) * x40 / x214;
    const real_t x272 = x10 * x220;
    const real_t x273 = x10 * x225;
    const real_t x274 = 96 * x136;
    const real_t x275 = -3 * x66;
    const real_t x276 = 12 * x57;
    const real_t x277 = 24 * u[0];
    const real_t x278 = 12 * u[8];
    const real_t x279 = 12 * x65;
    const real_t x280 = 12 * u[4];
    const real_t x281 = 12 * x2;
    const real_t x282 = u[7] * x281;
    const real_t x283 = 12 * u[5];
    const real_t x284 = x283 * x97;
    const real_t x285 = 8 * u[8];
    const real_t x286 = 4 * u[2];
    const real_t x287 = 4 * u[9];
    const real_t x288 = 4 * u[8];
    const real_t x289 = 8 * u[2];
    const real_t x290 = x280 * x97;
    const real_t x291 = -x120 * x70 + x32 * x97 + 16 * x67;
    const real_t x292 = dt / (630 * x4 - 630 * x7);
    const real_t x293 = u[4] * x210;
    const real_t x294 = x210 * x76;
    const real_t x295 = u[5] * x92;
    const real_t x296 = 4 * u[1];
    const real_t x297 = 5 * u[7];
    const real_t x298 = 8 * u[6];
    const real_t x299 = 6 * u[6];
    const real_t x300 = u[7] * x10;
    const real_t x301 = u[8] * x12;
    const real_t x302 = u[7] * x14;
    const real_t x303 = u[8] * x16;
    const real_t x304 = u[6] * x16;
    const real_t x305 = u[6] * x10;
    const real_t x306 = u[6] * x12;
    const real_t x307 = u[6] * x14;
    const real_t x308 = x103 * x2;
    const real_t x309 = u[7] * x3;
    const real_t x310 = x309 * x6;
    const real_t x311 = x108 * x5;
    const real_t x312 = u[8] * x6;
    const real_t x313 = x3 * x312;
    const real_t x314 = x2 * x63;
    const real_t x315 = 4 * x2;
    const real_t x316 = -x10 * x287 - x14 * x287 + x287 * x31 + x315 * x70;
    const real_t x317 = -x12 * x251 - x16 * x251 + x251 * x31 + x315 * x78;
    const real_t x318 = pow(u[10], 2);
    const real_t x319 = 48 * x318;
    const real_t x320 = x2 * x319;
    const real_t x321 = 48 * u[10];
    const real_t x322 = -x321 * x49;
    const real_t x323 = 48 * u[9];
    const real_t x324 = x323 * x49;
    const real_t x325 = pow(u[11], 2);
    const real_t x326 = pow(u[9], 2);
    const real_t x327 = pow(u[8], 2);
    const real_t x328 = x112 * x327;
    const real_t x329 = 12 * u[6];
    const real_t x330 = u[0] * (x309 - x312);
    const real_t x331 = x326 * x5;
    const real_t x332 = x106 * x65;
    const real_t x333 = u[1] * x312;
    const real_t x334 = x3 * x90;
    const real_t x335 = u[10] * x334;
    const real_t x336 = 16 * x48;
    const real_t x337 = x50 * x92;
    const real_t x338 = 16 * u[8];
    const real_t x339 = x338 * x78;
    const real_t x340 = 20 * u[7];
    const real_t x341 = x340 * x72;
    const real_t x342 = 20 * u[8];
    const real_t x343 = 24 * u[10];
    const real_t x344 = 24 * u[11];
    const real_t x345 = 32 * u[10];
    const real_t x346 = 32 * u[11];
    const real_t x347 = 32 * u[8];
    const real_t x348 = 32 * x139;
    const real_t x349 = x321 * x51;
    const real_t x350 = 48 * u[6];
    const real_t x351 = u[11] * x33;
    const real_t x352 = 64 * x351;
    const real_t x353 = 80 * u[11];
    const real_t x354 = 80 * u[9];
    const real_t x355 = u[11] * x62;
    const real_t x356 = u[7] * x63;
    const real_t x357 = 16 * u[11];
    const real_t x358 = u[9] * x3 * x84 - x342 * x70 - 9 * x356 + x357 * x50;
    const real_t x359 = pow(u[7], 2);
    const real_t x360 = 9 * x5;
    const real_t x361 = x6 * x88;
    const real_t x362 = 16 * u[10];
    const real_t x363 = x131 * x362;
    const real_t x364 = x139 * x357;
    const real_t x365 = 16 * u[7];
    const real_t x366 = 9 * u[7];
    const real_t x367 = x139 * x362;
    const real_t x368 = x131 * x92;
    const real_t x369 = -u[10] * x361 + u[7] * x336 + u[9] * x361 + x24 * x366 - x359 * x360 -
                        x363 - x364 + x365 * x37 - x365 * x75 + x367 + x368;
    const real_t x370 = x325 * x5;
    const real_t x371 = -x365 * x50 + 32 * x370;
    const real_t x372 = 32 * u[9];
    const real_t x373 = x249 * x28 - x28 * x287 - x33 * x345 + x33 * x372 - x342 * x49;
    const real_t x374 = x311 + x313;
    const real_t x375 = u[6] * x6;
    const real_t x376 = x48 * x5;
    const real_t x377 = -x249 * x31 - x3 * x375 - x314 - 4 * x376;
    const real_t x378 = pow(u[6], 2);
    const real_t x379 = x360 * x378;
    const real_t x380 = x112 * x378;
    const real_t x381 = 32 * x318;
    const real_t x382 = x163 * x6;
    const real_t x383 = 120 * u[7];
    const real_t x384 = 96 * u[9];
    const real_t x385 = u[7] * x50;
    const real_t x386 = 64 * u[9] * x51;
    const real_t x387 = 32 * u[6];
    const real_t x388 = 20 * u[6];
    const real_t x389 = 20 * u[11];
    const real_t x390 = x28 * x389;
    const real_t x391 = x131 * x388;
    const real_t x392 = u[1] * x375;
    const real_t x393 = u[7] * x108;
    const real_t x394 = u[11] * x145;
    const real_t x395 = x357 * x63;
    const real_t x396 = u[6] * x145;
    const real_t x397 = u[6] * x3;
    const real_t x398 = x202 * x62;
    const real_t x399 = x202 * x64;
    const real_t x400 = 11 * u[6];
    const real_t x401 = 16 * x351;
    const real_t x402 = u[6] * x51;
    const real_t x403 = 16 * x402;
    const real_t x404 = 18 * u[6];
    const real_t x405 = u[7] * x65;
    const real_t x406 = 20 * x355;
    const real_t x407 = 48 * u[7];
    const real_t x408 = x320 + x322 + x324 + x78 * x92;
    const real_t x409 = -48 * x370;
    const real_t x410 = -x321 * x37;
    const real_t x411 = 48 * u[11];
    const real_t x412 = x37 * x411;
    const real_t x413 = x131 * x342 + x409 + x410 + x412;
    const real_t x414 = 48 * x2 * x326;
    const real_t x415 = x321 * x33;
    const real_t x416 = x323 * x33;
    const real_t x417 = -x414 - x415 + x416;
    const real_t x418 = x308 + x310;
    const real_t x419 = x194 * x3;
    const real_t x420 = 120 * u[8];
    const real_t x421 = 96 * u[11];
    const real_t x422 = -64 * u[11] * x49;
    const real_t x423 = 20 * u[9];
    const real_t x424 = x139 * x340;
    const real_t x425 = u[7] * x51;
    const real_t x426 = 48 * u[8];
    const real_t x427 = x2 * x381 + x33 * x388 - x345 * x49 + x372 * x49;
    const real_t x428 = 8 * u[9];
    const real_t x429 = 2 * u[9];
    const real_t x430 = x12 * x429;
    const real_t x431 = 2 * u[10];
    const real_t x432 = x12 * x431;
    const real_t x433 = u[7] * x210;
    const real_t x434 = 2 * x209;
    const real_t x435 = 2 * u[11];
    const real_t x436 = x210 * x6;
    const real_t x437 = x209 * x63 + x210 * x375 - x429 * x436 + x431 * x436 + x434 * x50 -
                        x434 * x70 - x434 * x78 - x435 * x436;
    const real_t x438 = 12 * x359;
    const real_t x439 = x209 * x326;
    const real_t x440 = 96 * u[10];
    const real_t x441 = u[10] * x210;
    const real_t x442 = u[9] * x210;
    const real_t x443 = x237 * x6;
    const real_t x444 = u[10] * x443;
    const real_t x445 = 12 * x28;
    const real_t x446 = 8 * u[11];
    const real_t x447 = 5 * u[6];
    const real_t x448 = 4 * u[7];
    const real_t x449 = 8 * u[10];
    const real_t x450 = u[9] * x443;
    const real_t x451 = u[6] * x210;
    const real_t x452 = u[11] * x210;
    const real_t x453 = 40 * u[6];
    const real_t x454 = -x131 * x357 + 16 * x2 * x325 + x251 * x261;
    const real_t x455 = 12 * x378;
    const real_t x456 = x131 * x345;
    const real_t x457 = x131 * x372;
    const real_t x458 = -u[10] * x294 + u[11] * x294 + x1 * x375 + x1 * x451 + x209 * x455 +
                        x329 * x62 + x451 * x53 + x455 * x5 - x456 + x457;
    const real_t x459 = x10 * x431;
    const real_t x460 = x10 * x435;
    const real_t x461 = 2 * x2;
    const real_t x462 = 96 * x318;
    const real_t x463 = x281 * x327;
    const real_t x464 = 12 * u[10];
    const real_t x465 = 12 * u[11];
    const real_t x466 = u[2] * x228;
    const real_t x467 = x278 * x78;
    const real_t x468 = 4 * u[6];
    const real_t x469 = x288 * x33;
    const real_t x470 = x278 * x50;
    const real_t x471 = x278 * x28;
    const real_t x472 = 40 * u[8];
    const real_t x473 = x103 * x287 + x103 * x465 + x287 * x63 + x288 * x70 + x329 * x50 -
                        x372 * x50 - x372 * x78 - x411 * x50;
    element_vector[0] =
        -x154 *
        (-x153 * (78 * u[0] * x66 + 72 * u[0] * (x33 - x37) - u[3] * x82 + u[4] * x82 - u[4] * x83 +
                  u[5] * x83 + u[7] * x117 + x1 * x108 - 11 * x100 - x101 * x22 - x102 * x62 -
                  x103 * x104 + x104 * x108 - x105 * x106 - x107 - x108 * x32 - x109 + x110 +
                  x111 * x64 + x112 * x113 + x115 + x116 * x24 + x119 + x120 * x51 + x121 + x122 +
                  x123 + x124 + x125 + x126 + x127 + x128 + x129 + 32 * x130 + x131 * x132 +
                  x133 * x64 + x134 * x75 + x135 * x65 + x141 + x149 + x151 - x24 * x76 -
                  x33 * x69 + x33 * x84 + x37 * x71 - x37 * x85 - 80 * x44 - x47 + x48 * x79 -
                  x48 * x84 - x50 * x76 - x51 * x88 - x52 * x53 - x56 + x59 + 18 * x60 + 32 * x61 -
                  x62 * x74 - x63 * x68 + 80 * x67 + x69 * x72 - x70 * x71 + x70 * x73 - x72 * x73 -
                  x73 * x75 + x73 * x78 + x75 * x76 - x75 * x81 - 32 * x77 - x80 - x87 - x89 - x91 -
                  x93 - x95 - x98 - x99) +
         x42 * (-x0 * x30 - x0 * x31 + x11 + x13 + x15 + x17 + 3 * x18 + 3 * x19 + 3 * x20 +
                3 * x21 - x23 - x25 - x27 - x29 + x35 + x39) +
         x9 * (u[1] + u[2] - x0 + x1));
    element_vector[1] =
        -x188 *
        (x155 * (u[0] - 6 * u[1] + u[2] + x36) +
         x158 * (3 * x11 + 3 * x15 + x156 + x157 + x19 + x21 + x35) +
         x40 * (u[1] * x162 + u[3] * x164 - u[4] * x164 + u[4] * x169 - u[5] * x169 - u[6] * x114 -
                u[6] * x117 - 18 * x100 - x101 * x105 + x103 * x171 - x103 * x68 - x106 * x22 +
                x107 - x108 * x120 - x108 * x144 + x108 * x174 + x108 * x86 + x111 * x24 +
                x116 * x64 + x120 * x37 - x120 * x65 - x124 - x126 + 48 * x130 + x132 * x48 +
                x133 * x24 + x141 + x144 * x65 - 78 * x160 - x161 - x163 * x33 + x163 * x49 -
                72 * x165 - x166 - x167 * x49 + x168 * x70 - x168 * x75 - x170 * x51 - x170 * x72 -
                x171 * x94 - x172 * x37 + x173 * (u[4] - u[5]) + 18 * x175 + 18 * x176 + x177 +
                48 * x178 + x179 + x181 + x185 + x187 + x33 * x73 - x48 * x96 - x59 - x64 * x76 +
                x65 * x86 + x71 * (x139 + x51 - x70) + x75 * x96 - x76 * x78 + x91 + x93 + x98));
    element_vector[2] =
        -x188 *
        (x155 * (u[0] + u[1] - 6 * u[2] + x32) +
         x158 * (3 * x13 + x157 + 3 * x17 + x18 + x189 + x20 + x39) +
         x40 * (-u[2] * x138 - u[2] * x190 + u[2] * x193 + u[3] * x196 + u[4] * x195 - u[4] * x196 -
                u[5] * x195 - 9 * x100 - x103 * x174 - x103 * x86 + x103 * x96 + x105 * x199 +
                x108 * x135 - x108 * x171 - x115 - x119 - x122 - x125 - x131 * x167 + x149 +
                48 * x165 - x168 * x72 + x168 * x78 + x170 * x49 - x170 * x75 + x174 * x94 -
                11 * x175 + 9 * x176 - 32 * x178 - x180 * x6 + x182 + x183 - x184 + x187 +
                x194 * x37 - x194 * x51 - x197 + x198 * x24 - x198 * x64 - x199 * x22 - x200 +
                x201 * (u[3] - u[4]) + x202 * x26 + 11 * x203 + 72 * x204 + x205 + x206 + x207 -
                x37 * x73 - x48 * x76 + x48 * x81 + x56 + x62 * x79 - x63 * x86 + x63 * x96 -
                x69 * (x131 + x49 - x72) + x75 * x79 - 48 * x77 + x89 + x95));
    element_vector[3] =
        x271 *
        (x158 * (u[0] * x216 + u[1] * x216 - x10 * x217 + x11 + x19 + x209 * x22 - x216 * x217 -
                 x217 * x218 + x218 * x220 - x219 + x221 + x222 * x6 + x226) +
         x215 * (-u[2] + x1 + x208 + x36) +
         x270 * (3 * u[2] * (u[2] * x212 - x227) - u[3] * x241 + u[3] * x242 + u[4] * x241 -
                 u[4] * x242 + u[6] * x253 + u[7] * x253 + x1 * x247 + x1 * x63 + x1 * x97 +
                 4 * x100 - x103 * x243 - x105 * x248 + x109 - x110 - x128 - x129 - 40 * x130 +
                 x133 * x233 - x134 * x231 + x137 * x212 - x140 * x222 + x140 * x257 - x146 -
                 x147 * x233 + x168 * x33 + x170 * x234 - 8 * x175 + x181 + 8 * x203 + 8 * x204 +
                 x206 - x208 * x236 + 16 * x212 * x43 + x22 * x248 - x222 * x245 - x222 * x86 +
                 x227 * x245 - x227 * x246 + x229 + 32 * x230 - x231 * x232 + x231 * x260 +
                 x231 * x68 - x234 * x96 - x235 * x86 + x235 * x96 + x236 * x256 - x236 * x96 -
                 x238 - x240 - 8 * x244 - x246 * x247 - x247 * x36 - x249 * x250 - x250 * x251 +
                 x250 * x259 - x252 + x254 * x36 + 5 * x255 + x256 * x257 + x257 * x96 + x258 +
                 x260 * x70 + x263 + x269 - x36 * x63 - x36 * x97 + 96 * x46 + x80));
    element_vector[4] =
        x292 *
        (x155 * (-u[0] + x256 + x32 + x36) +
         x158 *
             (-x14 * x220 + x14 * x225 + x156 + x16 * x217 - x16 * x220 + x189 - x217 * x30 + x219 +
              x220 * x30 - x221 - x225 * x30 - x272 + x273 + 2 * x3 * x49 - 2 * x34 - 2 * x38) -
         x40 * (u[0] * x275 + u[0] * (x24 - x28) - 40 * u[2] * x48 - u[3] * x279 + u[3] * x282 +
                u[4] * x279 - u[4] * x282 - 8 * x100 - x103 * x120 + x103 * x245 - x103 * x256 +
                x103 * x266 + x105 * x288 - x108 * x208 + x108 * x256 - x108 * x266 + x108 * x96 +
                x132 * x49 - x135 * x50 - x139 * x286 - 32 * x165 + 4 * x175 + 8 * x176 - x177 +
                40 * x178 + x185 + x197 + x207 + x208 * x63 - x22 * x285 - x229 - x232 * x48 +
                x232 * x50 - x237 * x37 + x238 + x24 * x289 - x24 * x32 + x240 + x243 * x33 -
                x243 * x37 - x245 * x65 + x252 - x258 - x26 * x278 - x26 * x287 + x262 * x37 -
                x262 * x51 + x263 + x265 - x266 * x70 + x266 * x72 + x266 * x75 - x266 * x78 -
                x266 * x94 - x268 - x274 * x3 + x274 * x6 + x276 * x6 - x277 * x48 + x277 * x50 -
                x280 * x63 + x283 * x63 - x284 - x286 * x64 + x290 + x291 - x32 * x64 - x36 * x94 +
                x37 * x76 - 16 * x44 + x48 * x68 - x51 * x53 + x51 * x74 + x51 * x76 - 5 * x60 +
                x72 * x76 - x72 * x81 + x72 * x96));
    element_vector[5] =
        x271 *
        (x158 * (u[0] * x218 + u[2] * x218 - x12 * x225 + x13 + x20 + x210 * x28 + x216 * x220 -
                 x216 * x225 - x218 * x225 + x226 + x250 * x5 + x272 - x273) +
         x215 * (-u[1] + x1 + x245 + x32) +
         x270 *
             (-u[0] * x222 - 20 * u[10] * x250 - 40 * u[11] * x250 - u[2] * x241 - u[5] * x294 +
              x1 * x227 + x1 * x236 + x1 * x254 + x1 * x261 + 5 * x100 + x103 * x208 - x103 * x246 +
              x103 * x280 - x103 * x283 - x105 * x285 + 24 * x118 - x120 * x63 - x123 - x131 * x84 +
              x135 * x78 + x142 + x143 + x144 * x227 - x144 * x236 + x144 * x62 - x148 + x151 +
              3 * x160 + x170 * x231 - 5 * x175 - 4 * x178 + x180 * x212 + x200 + 4 * x203 + x205 -
              x209 * x295 + x212 * x276 - x22 * x287 + x22 * x288 + x222 * x256 + x222 * x283 +
              x222 * x32 - x227 * x243 - x227 * x32 + 96 * x230 + x231 * x76 - x231 * x81 -
              x232 * x78 + x233 * x76 - x233 * x81 + x233 * x96 + x234 * x260 - x236 * x32 -
              x243 * x247 + x243 * x49 - 5 * x244 - x245 * x63 - x247 * x280 + x247 * x283 +
              x247 * x32 - x250 * x287 + x250 * x297 + x250 * x298 - x254 * x32 + 4 * x255 +
              x256 * x63 + x257 * x32 + x257 * x79 + x260 * x78 + x269 + x284 - x289 * x33 - x290 +
              x291 + x293 * x53 - x293 * x74 - x293 * x76 - x296 * x33 + x47));
    element_vector[6] =
        -x154 * (-x153 * (24 * u[0] * u[10] * x3 + 96 * u[0] * u[9] * x3 + 24 * u[0] * u[9] * x6 +
                          16 * u[10] * u[11] * x2 + 4 * u[10] * u[1] * x3 + 16 * u[10] * u[8] * x5 +
                          16 * u[10] * u[9] * x2 + 20 * u[11] * u[1] * x6 + 16 * u[11] * u[2] * x3 +
                          48 * u[11] * u[4] * x3 + 32 * u[11] * u[5] * x3 + 48 * u[11] * u[6] * x2 +
                          20 * u[11] * u[7] * x5 + 80 * u[11] * u[9] * x2 + 9 * u[1] * u[6] * x3 -
                          u[1] * u[6] * x58 + 9 * u[2] * u[6] * x3 + 11 * u[2] * u[7] * x3 +
                          9 * u[2] * u[8] * x6 + 48 * u[3] * u[6] * x6 + 16 * u[3] * u[8] * x6 +
                          80 * u[3] * u[9] * x3 + 32 * u[4] * u[9] * x3 + 48 * u[5] * u[6] * x6 +
                          32 * u[5] * u[8] * x6 + 64 * u[5] * u[9] * x3 + 9 * u[6] * u[7] * x2 +
                          78 * u[6] * x66 + 72 * u[6] * (-x75 + x78) + 11 * u[7] * u[8] * x2 +
                          32 * u[7] * u[9] * x5 - u[7] * x348 - u[8] * x336 - x101 * x103 -
                          x106 * x63 - x131 * x353 - x139 * x350 + 80 * x2 * x325 + 32 * x2 * x326 -
                          x202 * x28 - x24 * x251 + 48 * x318 * x5 - x320 - x322 - x323 * x63 -
                          x324 - x328 + x329 * x52 - 18 * x330 - 80 * x331 - x332 - 11 * x333 -
                          x335 - x337 - x339 - x340 * x51 - x341 - x342 * x75 - x343 * x62 -
                          x344 * x64 - x345 * x37 - x346 * x49 - x347 * x72 - x349 - x350 * x37 -
                          x352 - x354 * x78 - 96 * x355 - x358 - x369 - x371 - x373) +
                 x42 * (-x299 * x31 + x300 + x301 + x302 + x303 + 3 * x304 + 3 * x305 + 3 * x306 +
                        3 * x307 - x308 - x310 - x311 - x313 - 6 * x314 + x316 + x317) +
                 x9 * (u[7] + u[8] + x249 - x299));
    element_vector[7] =
        -x188 *
        (x155 * (u[6] - 6 * u[7] + u[8] + x251) +
         x158 * (3 * x300 + 3 * x302 + x305 + x307 + x316 + x374 + x377) -
         x40 *
             (-u[10] * x382 + u[11] * x334 + u[6] * x348 + 78 * u[7] * x159 + u[7] * x162 -
              u[8] * x138 + u[9] * x382 - x101 * x62 + x101 * x63 + x103 * x106 + x103 * x323 +
              x106 * x28 - x111 * x309 - x116 * x397 + x139 * x407 + x24 * x343 - x24 * x344 -
              x24 * x384 + x24 * x404 + x28 * x362 + x28 * x400 - x28 * x92 + x323 * x65 - x328 +
              x33 * x338 - x33 * x387 + 18 * x333 - x335 + x338 * x50 - x339 - x345 * x51 +
              x346 * x51 + x347 * x49 - x354 * (u[3] * x3 - x50 - x70) + x358 + x362 * x62 + x363 +
              x364 - x366 * x64 - x367 - x368 - x37 * x372 + x37 * x388 - x372 * x63 - x379 + x380 +
              x381 * x5 + x383 * x48 - x383 * x75 - 72 * x385 - x386 - x388 * x50 - x390 - x391 -
              18 * x392 - 18 * x393 - x394 - x395 - x396 - x398 + x399 + x401 + x403 + 18 * x405 +
              x406 + x407 * x51 + x408 + x413 + x417 - x62 * x92 + 4 * x64 * (u[10] - u[11])));
    element_vector[8] =
        -x188 *
        (x155 * (u[6] + u[7] - 6 * u[8] + x287) +
         x158 * (3 * x301 + 3 * x303 + x304 + x306 + x317 + x377 + x418) +
         x40 *
             (-u[10] * x419 + u[11] * x419 + u[11] * x6 * x85 - 11 * u[7] * x64 + u[8] * x190 +
              u[8] * x193 - 72 * u[8] * x48 - x102 * x312 - x103 * x199 - x106 * x62 + x108 * x366 +
              x131 * x387 + x131 * x426 - x139 * x388 + x198 * x309 - x198 * x397 + x199 * x63 -
              x24 * x357 + x24 * x362 + x24 * x400 - x24 * x423 - x259 * x28 + x28 * x343 +
              x28 * x404 - x28 * x421 + x319 * x5 - x332 - x341 - x346 * x65 - x349 - 32 * x351 +
              x353 * (-x131 + x48 + x72) - x357 * x64 + x362 * x64 + x369 - x37 * x387 + x37 * x92 +
              x379 - x380 - 48 * x385 - x388 * x48 - 11 * x392 + x396 + x398 - x399 - x403 +
              11 * x405 + x409 + x410 + x411 * x51 + x411 * x63 + x412 + x417 + x420 * x50 -
              x420 * x78 + x422 + x423 * x64 + x424 + 32 * x425 + x426 * x49 + x426 * x72 + x427 +
              x48 * x92 - x51 * x92 + 4 * x62 * (u[10] - u[9]) - x65 * x92 + x72 * x92));
    element_vector[9] =
        x271 *
        (x158 * (u[6] * x216 + u[7] * x216 - x10 * x429 + x103 * x209 - x216 * x429 - x218 * x429 +
                 x218 * x431 + x300 + x305 - x430 + x432 + x433 * x6 + x437) +
         x215 * (-u[8] + x249 + x251 + x428) +
         x270 * (u[10] * x445 + 5 * u[2] * x433 - 40 * u[3] * x433 - 20 * u[4] * x433 -
                 u[6] * x227 - 8 * u[7] * x236 - 3 * u[8] * x192 + 24 * u[8] * x33 + u[9] * x241 -
                 u[9] * x445 - x1 * x312 + x103 * x248 - x103 * x423 - x103 * x446 + x144 * x451 -
                 x170 * x442 + x210 * x267 - x210 * x295 + x211 * x381 + x211 * x438 - x222 * x239 +
                 x222 * x249 - x222 * x251 - x222 * x298 + x222 * x423 + x227 * x251 + x227 * x448 +
                 x227 * x449 + x231 * x239 - x231 * x321 - x231 * x346 + x231 * x387 - x233 * x239 +
                 x233 * x329 - x233 * x346 + x235 * x453 + x236 * x251 + x243 * x433 + x247 * x249 -
                 x247 * x251 - x247 * x447 - x248 * x63 + x251 * x254 - x251 * x257 + x28 * x446 -
                 x28 * x447 + x288 * x50 + x288 * x62 - x288 * x78 + x296 * x312 + x298 * x50 -
                 x312 * x36 + x33 * x384 - x33 * x440 + x33 * x453 - x337 - x352 - 16 * x355 -
                 x36 * x433 + x371 + x391 - 8 * x392 + x395 + x408 + x423 * x63 - x428 * x62 +
                 96 * x439 - x441 * x79 - x441 * x81 + x442 * x86 - x444 + x449 * x62 + x450 +
                 x452 * x79 + x452 * x81 + x454 + x458));
    element_vector[10] =
        x292 *
        (x155 * (-u[6] + x251 + x287 + x449) +
         x158 * (-x14 * x431 + x14 * x435 + x16 * x429 - x16 * x431 - x31 * x429 + x31 * x431 -
                 x31 * x435 + x374 + 2 * x376 + x418 + x430 - x432 - x459 + x460 - x461 * x70 -
                 x461 * x78) -
         x40 *
             (-u[10] * x466 + u[11] * x466 - u[2] * x287 * x3 + u[6] * x275 + 24 * u[6] * x49 -
              u[6] * (x103 + x191) - u[7] * x201 + u[8] * x173 + u[9] * x138 + 12 * u[9] * x62 -
              x103 * x285 - x139 * x345 + x139 * x346 + x139 * x468 + x139 * x92 - x2 * x462 -
              x239 * x24 - x239 * x48 + x239 * x75 - x24 * x446 + x24 * x449 - x24 * x468 +
              x24 * x92 - x251 * x65 - x278 * x75 - x28 * x357 + x28 * x428 - x28 * x449 +
              x28 * x468 - x286 * x397 - x288 * x72 + x289 * x309 + x296 * x375 - x329 * x48 -
              x33 * x468 + 5 * x330 - 16 * x331 - 8 * x333 + x346 * x48 - x347 * x48 - x36 * x375 -
              x37 * x448 + x37 * x468 + x372 * x72 - x384 * x49 + 32 * x385 + x386 + 8 * x393 -
              24 * x402 + x413 + x414 + x415 - x416 + x421 * x51 + x422 - x424 - 40 * x425 -
              x428 * x64 - x428 * x65 + x438 * x5 + x440 * x49 - x440 * x51 + x444 + x446 * x62 +
              x446 * x63 - x448 * x72 - x450 + x454 + x456 - x457 + x462 * x5 - x463 - x464 * x62 +
              x464 * x64 - x465 * x64 - x467 + x469 + x470 + x471 + x472 * x49 + x473));
    element_vector[11] =
        x271 *
        (x158 * (u[6] * x218 + u[8] * x218 - x12 * x435 + x210 * x312 + x216 * x431 - x216 * x435 -
                 x218 * x435 + x227 * x5 + x301 + x306 + x437 + x459 - x460) +
         x215 * (-u[7] + x249 + x287 + x446) +
         x270 * (24 * u[5] * x433 + 40 * u[5] * x451 + 3 * u[7] * (u[7] * x211 - x222) - x1 * x433 +
                 x103 * x288 + x131 * x453 - x131 * x472 - x134 * x442 - x135 * x441 + x135 * x452 +
                 x140 * x451 - x147 * x441 + x147 * x452 + x211 * x319 + 16 * x211 * x326 +
                 x222 * x428 - x222 * x447 + x222 * x464 - x222 * x465 + x227 * x297 - x227 * x362 -
                 x227 * x389 - x227 * x428 - x233 * x357 + x234 * x388 + x234 * x92 - x235 * x92 -
                 x236 * x297 + x236 * x449 + x236 * x92 + x247 * x287 - x247 * x298 - x247 * x464 +
                 x247 * x465 + x249 * x254 + x249 * x261 + x249 * x62 - x254 * x287 - x257 * x446 +
                 x257 * x449 - x257 * x92 - x261 * x287 + x266 * x433 - x28 * x298 + x285 * x62 -
                 x285 * x63 + x286 * x433 - x287 * x62 - x32 * x433 + 5 * x333 + x346 * x63 - x356 +
                 96 * x370 + x373 + 8 * x385 + x390 - 5 * x392 + x394 - x401 - x406 + x427 +
                 32 * x439 + x458 + x463 + x467 - x469 - x470 - x471 + x473));
}

// 2) Potential equation
// static SFEM_INLINE void tri3_laplacian_lhs_kernel(const real_t px0,
//                                                   const real_t px1,
//                                                   const real_t px2,
//                                                   const real_t py0,
//                                                   const real_t py1,
//                                                   const real_t py2,
//                                                   real_t *const SFEM_RESTRICT element_matrix) {
// }

static SFEM_INLINE void tri3_tri6_divergence_rhs_kernel(const real_t px0,
                                                        const real_t px1,
                                                        const real_t px2,
                                                        const real_t py0,
                                                        const real_t py1,
                                                        const real_t py2,
                                                        const real_t dt,
                                                        const real_t rho,
                                                        const real_t *const SFEM_RESTRICT u,
                                                        real_t *const SFEM_RESTRICT
                                                            element_vector) {
    const real_t x0 = 2 * px0;
    const real_t x1 = 2 * py0;
    const real_t x2 = (1.0 / 6.0) * rho / dt;
    const real_t x3 = px0 - px2;
    const real_t x4 = u[11] * x3;
    const real_t x5 = py0 - py2;
    const real_t x6 = u[4] * x5;
    const real_t x7 = px0 - px1;
    const real_t x8 = u[10] * x7;
    const real_t x9 = u[10] * x3;
    const real_t x10 = py0 - py1;
    const real_t x11 = u[3] * x10;
    const real_t x12 = u[4] * x10;
    const real_t x13 = u[5] * x5;
    const real_t x14 = u[9] * x7;
    element_vector[0] = x2 * (-px1 * u[10] - px1 * u[11] + px1 * u[6] + px1 * u[9] + px2 * u[10] -
                              px2 * u[11] - px2 * u[6] + px2 * u[9] - py1 * u[0] - py1 * u[3] +
                              py1 * u[4] + py1 * u[5] + py2 * u[0] - py2 * u[3] - py2 * u[4] +
                              py2 * u[5] + u[11] * x0 + u[3] * x1 - u[5] * x1 - u[9] * x0);
    element_vector[1] = x2 * (u[1] * x5 - u[3] * x5 - u[7] * x3 + u[9] * x3 + 2 * x11 - 2 * x12 -
                              x13 - 2 * x14 + x4 + x6 + 2 * x8 - x9);
    element_vector[2] = x2 * (-u[11] * x7 - u[2] * x10 + u[5] * x10 + u[8] * x7 + x11 - x12 -
                              2 * x13 - x14 + 2 * x4 + 2 * x6 + x8 - 2 * x9);
}

// 2) Correction/Projection
static SFEM_INLINE void tri6_tri3_rhs_correction_kernel(const real_t px0,
                                                        const real_t px1,
                                                        const real_t px2,
                                                        const real_t py0,
                                                        const real_t py1,
                                                        const real_t py2,
                                                        const real_t dt,
                                                        const real_t rho,
                                                        const real_t *const SFEM_RESTRICT p,
                                                        real_t *const SFEM_RESTRICT
                                                            element_vector) {
    const real_t x0 = py0 - py1;
    const real_t x1 = py0 - py2;
    const real_t x2 = (1.0 / 6.0) * dt / rho;
    const real_t x3 = x2 * (p[0] * x0 - p[0] * x1 + p[1] * x1 - p[2] * x0);
    const real_t x4 = px0 - px1;
    const real_t x5 = px0 - px2;
    const real_t x6 = x2 * (-p[0] * x4 + p[0] * x5 - p[1] * x5 + p[2] * x4);
    element_vector[0] = 0;
    element_vector[1] = 0;
    element_vector[2] = 0;
    element_vector[3] = x3;
    element_vector[4] = x3;
    element_vector[5] = x3;
    element_vector[6] = 0;
    element_vector[7] = 0;
    element_vector[8] = 0;
    element_vector[9] = x6;
    element_vector[10] = x6;
    element_vector[11] = x6;

    // Weak derivative on the pressure
    // const real_t x0 = -py2;
    // const real_t x1 = (1.0/6.0)*dt/rho;
    // const real_t x2 = p[0]*x1;
    // const real_t x3 = py0 + x0;
    // const real_t x4 = py0 - py1;
    // const real_t x5 = p[2]*x4;
    // const real_t x6 = p[1]*x3;
    // const real_t x7 = 2*p[1];
    // const real_t x8 = p[0]*x3 + p[0]*x4;
    // const real_t x9 = 2*p[2];
    // const real_t x10 = -px2;
    // const real_t x11 = px0 + x10;
    // const real_t x12 = p[1]*x11;
    // const real_t x13 = px0 - px1;
    // const real_t x14 = p[2]*x13;
    // const real_t x15 = p[0]*x11 + p[0]*x13;
    // element_vector[0] = x2*(py1 + x0);
    // element_vector[1] = -p[1]*x1*x3;
    // element_vector[2] = x1*x5;
    // element_vector[3] = -x1*(x4*x7 + x5 - x6 + x8);
    // element_vector[4] = x1*(-p[0]*py1 + p[0]*py2 + p[1]*py0 + p[1]*py2 - p[2]*py0 - p[2]*py1 + 2*p[2]*py2 - py1*x7);
    // element_vector[5] = x1*(x3*x9 - x5 + x6 + x8);
    // element_vector[6] = x2*(-px1 - x10);
    // element_vector[7] = x1*x12;
    // element_vector[8] = -p[2]*x1*x13;
    // element_vector[9] = x1*(-x12 + x13*x7 + x14 + x15);
    // element_vector[10] = x1*(p[0]*px1 - p[0]*px2 - p[1]*px0 - p[1]*px2 + p[2]*px0 + p[2]*px1 + px1*x7 - px2*x9);
    // element_vector[11] = -x1*(x11*x9 + x12 - x14 + x15);
}

static SFEM_INLINE void tri6_explict_momentum_rhs_kernel(const real_t px0,
                                                         const real_t px1,
                                                         const real_t px2,
                                                         const real_t py0,
                                                         const real_t py1,
                                                         const real_t py2,
                                                         const real_t dt,
                                                         const real_t nu,
                                                         const real_t convonoff,
                                                         real_t *const SFEM_RESTRICT u,
                                                         real_t *const SFEM_RESTRICT
                                                             element_vector) {
    const real_t x0 = pow(u[2], 2);
    const real_t x1 = py0 - py1;
    const real_t x2 = 9 * x1;
    const real_t x3 = x0 * x2;
    const real_t x4 = px0 - px1;
    const real_t x5 = u[10] * x4;
    const real_t x6 = u[4] * x1;
    const real_t x7 = px0 - px2;
    const real_t x8 = py0 - py2;
    const real_t x9 = -u[10] * x7 - u[4] * x8 + x5 + x6;
    const real_t x10 = 12 * u[0];
    const real_t x11 = u[1] * x7;
    const real_t x12 = u[2] * x4;
    const real_t x13 = u[6] * (x11 - x12);
    const real_t x14 = -x1;
    const real_t x15 = pow(u[3], 2);
    const real_t x16 = x14 * x15;
    const real_t x17 = pow(u[5], 2);
    const real_t x18 = x17 * x8;
    const real_t x19 = -x8;
    const real_t x20 = pow(u[4], 2);
    const real_t x21 = 48 * x20;
    const real_t x22 = u[3] * x1;
    const real_t x23 = u[5] * x8;
    const real_t x24 = u[6] * x4;
    const real_t x25 = u[0] * x1;
    const real_t x26 = u[6] * x7;
    const real_t x27 = x25 + x26;
    const real_t x28 = u[0] * (u[0] * x19 - x24 + x27);
    const real_t x29 = x15 * x8;
    const real_t x30 = x14 * x17;
    const real_t x31 = 96 * u[3];
    const real_t x32 = 80 * u[5];
    const real_t x33 = u[9] * x7;
    const real_t x34 = 80 * u[3];
    const real_t x35 = u[11] * x4;
    const real_t x36 = 48 * u[0];
    const real_t x37 = 48 * u[5];
    const real_t x38 = u[9] * x4;
    const real_t x39 = u[10] * x7;
    const real_t x40 = x37 * x39;
    const real_t x41 = 32 * u[3];
    const real_t x42 = u[11] * x12;
    const real_t x43 = u[11] * x7;
    const real_t x44 = 32 * u[5];
    const real_t x45 = x43 * x44;
    const real_t x46 = u[1] * x8;
    const real_t x47 = 32 * u[4];
    const real_t x48 = x38 * x47;
    const real_t x49 = 24 * x24;
    const real_t x50 = 24 * x26;
    const real_t x51 = 20 * u[2];
    const real_t x52 = x5 * x51;
    const real_t x53 = 20 * u[1];
    const real_t x54 = u[7] * x4;
    const real_t x55 = 20 * u[5];
    const real_t x56 = x54 * x55;
    const real_t x57 = u[4] * x8;
    const real_t x58 = 16 * u[1];
    const real_t x59 = x57 * x58;
    const real_t x60 = 16 * u[2];
    const real_t x61 = x57 * x60;
    const real_t x62 = u[9] * x12;
    const real_t x63 = 16 * x62;
    const real_t x64 = 16 * x6;
    const real_t x65 = u[3] * x64;
    const real_t x66 = 16 * x7;
    const real_t x67 = u[8] * x66;
    const real_t x68 = u[5] * x67;
    const real_t x69 = u[2] * x1;
    const real_t x70 = u[1] * x69;
    const real_t x71 = 11 * u[8];
    const real_t x72 = u[1] * x25;
    const real_t x73 = u[7] * x7;
    const real_t x74 = 9 * u[0];
    const real_t x75 = 9 * x7;
    const real_t x76 = u[0] * u[8];
    const real_t x77 = 9 * u[8];
    const real_t x78 = x12 * x77;
    const real_t x79 = u[8] * x4;
    const real_t x80 = 4 * x79;
    const real_t x81 = u[3] * x80;
    const real_t x82 = 4 * x73;
    const real_t x83 = u[4] * x82;
    const real_t x84 = u[4] * x80;
    const real_t x85 = u[5] * x82;
    const real_t x86 = u[2] * x25;
    const real_t x87 = 9 * x86;
    const real_t x88 = u[0] * x8;
    const real_t x89 = 9 * u[2];
    const real_t x90 = 11 * u[2];
    const real_t x91 = 11 * x12;
    const real_t x92 = x6 * x60;
    const real_t x93 = x23 * x60;
    const real_t x94 = u[4] * x67;
    const real_t x95 = x39 * x53;
    const real_t x96 = u[5] * x1;
    const real_t x97 = x53 * x96;
    const real_t x98 = x43 * x47;
    const real_t x99 = u[9] * x11;
    const real_t x100 = 32 * u[2];
    const real_t x101 = x38 * x41;
    const real_t x102 = u[3] * x8;
    const real_t x103 = 48 * u[4];
    const real_t x104 = x103 * x39;
    const real_t x105 = 96 * u[5];
    const real_t x106 = 48 * x5;
    const real_t x107 = u[4] * x106;
    const real_t x108 = u[0] * x46;
    const real_t x109 = 16 * u[5];
    const real_t x110 = u[8] * x7;
    const real_t x111 = 20 * u[3];
    const real_t x112 = u[3] * x106;
    const real_t x113 = x1 * x21 - x102 * x51 - x107 - 9 * x108 + x109 * x57 + x110 * x111 + x112;
    const real_t x114 = pow(u[1], 2);
    const real_t x115 = 16 * u[4];
    const real_t x116 = x115 * x35;
    const real_t x117 = x109 * x33;
    const real_t x118 = 9 * x11;
    const real_t x119 = u[11] * x11;
    const real_t x120 = 16 * u[3];
    const real_t x121 = x120 * x35;
    const real_t x122 = x115 * x33;
    const real_t x123 = u[7] * x118 + 9 * x114 * x19 - x115 * x54 - x116 - x117 + 16 * x119 +
                        x120 * x54 + x121 + x122 - x22 * x58 + x58 * x6;
    const real_t x124 = 64 * u[5];
    const real_t x125 = x120 * x57 + x124 * x38;
    const real_t x126 = 64 * u[3];
    const real_t x127 = -u[5] * x64 - x126 * x43;
    const real_t x128 = (1.0 / 2520.0) * dt;
    const real_t x129 = u[1] * (x46 - x73);
    const real_t x130 = x1 * x15;
    const real_t x131 = 48 * x130;
    const real_t x132 = pow(u[0], 2);
    const real_t x133 = 9 * x132;
    const real_t x134 = x133 * x8;
    const real_t x135 = 12 * x23 + 12 * x43;
    const real_t x136 = 120 * u[1];
    const real_t x137 = 120 * x54;
    const real_t x138 = u[1] * x57;
    const real_t x139 = x126 * x39;
    const real_t x140 = u[2] * x6;
    const real_t x141 = 32 * u[0];
    const real_t x142 = 24 * x73;
    const real_t x143 = 20 * u[0];
    const real_t x144 = 18 * u[0];
    const real_t x145 = 16 * u[0];
    const real_t x146 = x145 * x5;
    const real_t x147 = x24 * x74;
    const real_t x148 = 4 * x26;
    const real_t x149 = x26 * x74;
    const real_t x150 = 11 * u[0];
    const real_t x151 = x145 * x39;
    const real_t x152 = x120 * x23;
    const real_t x153 = u[7] * x12;
    const real_t x154 = u[1] * x39;
    const real_t x155 = 48 * u[3];
    const real_t x156 = -x103 * x43;
    const real_t x157 = x37 * x43;
    const real_t x158 = x156 + x157 + 20 * x42;
    const real_t x159 = x103 * x38;
    const real_t x160 = x155 * x38;
    const real_t x161 = -x159 + x160;
    const real_t x162 = x109 * x38 - x109 * x5 + x24 * x55 - x55 * x79;
    const real_t x163 = 32 * x20;
    const real_t x164 = -x143 * x35 + x143 * x43 - x145 * x23 + x163 * x8 + x39 * x44 - x39 * x47;
    const real_t x165 = 12 * x22 + 12 * x38;
    const real_t x166 = u[2] * (u[2] * x14 + x79);
    const real_t x167 = 120 * u[2];
    const real_t x168 = 120 * x110;
    const real_t x169 = x47 * x5;
    const real_t x170 = 24 * x79;
    const real_t x171 = x143 * x33;
    const real_t x172 = x111 * x73;
    const real_t x173 = x7 * x76;
    const real_t x174 = 18 * u[2];
    const real_t x175 = x145 * x22;
    const real_t x176 = x120 * x39;
    const real_t x177 = u[6] * x11;
    const real_t x178 = u[6] * x12;
    const real_t x179 = 4 * x24;
    const real_t x180 = x120 * x43;
    const real_t x181 = u[8] * x11;
    const real_t x182 = x143 * x38;
    const real_t x183 = 20 * x99;
    const real_t x184 = x111 * x26;
    const real_t x185 = x41 * x5;
    const real_t x186 = u[2] * x96;
    const real_t x187 = x17 * x19;
    const real_t x188 = -x124 * x5 + 48 * x187;
    const real_t x189 = -x104 + x109 * x22 + x21 * x8 + x40;
    const real_t x190 = x155 * x6;
    const real_t x191 = 40 * u[0];
    const real_t x192 = x22 * x44;
    const real_t x193 = x44 * x6;
    const real_t x194 = 12 * x79;
    const real_t x195 = 8 * u[0];
    const real_t x196 = 8 * x24;
    const real_t x197 = 8 * x79;
    const real_t x198 = 5 * u[2];
    const real_t x199 = 4 * u[2];
    const real_t x200 = 4 * u[5];
    const real_t x201 = x200 * x54;
    const real_t x202 = x200 * x25;
    const real_t x203 = 4 * u[1];
    const real_t x204 = x203 * x96;
    const real_t x205 = 4 * x186;
    const real_t x206 = u[4] * x110;
    const real_t x207 = 5 * u[0];
    const real_t x208 = 8 * u[1];
    const real_t x209 = x10 * x6;
    const real_t x210 = 12 * u[2];
    const real_t x211 = x210 * x22;
    const real_t x212 = x109 * x35;
    const real_t x213 = 96 * u[4];
    const real_t x214 = 12 * x132;
    const real_t x215 = -x35 * x41;
    const real_t x216 = -x33 * x47;
    const real_t x217 = 4 * u[0];
    const real_t x218 = x35 * x47;
    const real_t x219 = x33 * x44;
    const real_t x220 = x1 * x214 - x10 * x24 + x10 * x26 + x19 * x214 + x215 + x216 + x217 * x39 -
                        x217 * x5 + x218 + x219;
    const real_t x221 = 12 * x8;
    const real_t x222 = 12 * u[1];
    const real_t x223 = 12 * u[7];
    const real_t x224 = 12 * x4;
    const real_t x225 = u[7] * x224;
    const real_t x226 =
        -u[3] * x225 + u[4] * x225 - x11 * x223 + x114 * x221 - 4 * x119 + x22 * x222 - x222 * x6;
    const real_t x227 = (1.0 / 630.0) * dt;
    const real_t x228 = 16 * x1;
    const real_t x229 = 96 * x20;
    const real_t x230 = x37 * x57;
    const real_t x231 = x41 * x57;
    const real_t x232 = x23 * x41;
    const real_t x233 = 24 * u[0];
    const real_t x234 = 12 * x24;
    const real_t x235 = 12 * x26;
    const real_t x236 = 8 * u[2];
    const real_t x237 = 8 * x26;
    const real_t x238 = 8 * x73;
    const real_t x239 = 4 * u[3];
    const real_t x240 = x110 * x239;
    const real_t x241 = x102 * x217;
    const real_t x242 = x239 * x46;
    const real_t x243 = x102 * x199;
    const real_t x244 = x10 * x57;
    const real_t x245 = x222 * x23;
    const real_t x246 = x120 * x33;
    const real_t x247 = 12 * u[8];
    const real_t x248 = 12 * u[5];
    const real_t x249 =
        12 * x0 * x14 + x110 * x248 + x12 * x247 - 12 * x206 - x210 * x23 + x210 * x57 + 4 * x62;
    const real_t x250 = 12 * x73;
    const real_t x251 = pow(u[7], 2);
    const real_t x252 = x251 * x75;
    const real_t x253 = 12 * u[6];
    const real_t x254 = u[7] * x8;
    const real_t x255 = u[0] * (u[8] * x1 - x254);
    const real_t x256 = -x7;
    const real_t x257 = pow(u[11], 2);
    const real_t x258 = x256 * x257;
    const real_t x259 = pow(u[9], 2);
    const real_t x260 = x259 * x4;
    const real_t x261 = -x4;
    const real_t x262 = pow(u[10], 2);
    const real_t x263 = 48 * x262;
    const real_t x264 = u[6] * (u[6] * x261 + x27 - x88);
    const real_t x265 = x257 * x4;
    const real_t x266 = x256 * x259;
    const real_t x267 = u[11] * x25;
    const real_t x268 = 80 * u[11];
    const real_t x269 = 80 * u[9];
    const real_t x270 = 48 * u[6];
    const real_t x271 = 48 * u[9];
    const real_t x272 = x271 * x6;
    const real_t x273 = 32 * u[10];
    const real_t x274 = x23 * x273;
    const real_t x275 = 32 * u[11];
    const real_t x276 = 32 * x102;
    const real_t x277 = 32 * u[9];
    const real_t x278 = x22 * x277;
    const real_t x279 = 24 * u[10];
    const real_t x280 = 24 * u[11];
    const real_t x281 = u[9] * x8;
    const real_t x282 = x281 * x51;
    const real_t x283 = 20 * u[7];
    const real_t x284 = x283 * x57;
    const real_t x285 = 20 * u[8];
    const real_t x286 = 16 * x39;
    const real_t x287 = u[11] * x286;
    const real_t x288 = 16 * x5;
    const real_t x289 = u[7] * x288;
    const real_t x290 = u[8] * x288;
    const real_t x291 = u[1] * x228;
    const real_t x292 = u[9] * x291;
    const real_t x293 = 16 * u[7];
    const real_t x294 = x23 * x293;
    const real_t x295 = u[8] * x1;
    const real_t x296 = u[1] * x295;
    const real_t x297 = u[1] * u[6];
    const real_t x298 = 9 * x46;
    const real_t x299 = u[7] * x298;
    const real_t x300 = 9 * u[6];
    const real_t x301 = 4 * u[10];
    const real_t x302 = x301 * x69;
    const real_t x303 = u[11] * x46;
    const real_t x304 = 4 * x303;
    const real_t x305 = x301 * x46;
    const real_t x306 = u[9] * x69;
    const real_t x307 = 4 * x306;
    const real_t x308 = u[6] * x8;
    const real_t x309 = u[7] * x24;
    const real_t x310 = u[7] * x26;
    const real_t x311 = 9 * x310;
    const real_t x312 = u[7] * x79;
    const real_t x313 = u[10] * x291;
    const real_t x314 = u[7] * x286;
    const real_t x315 = x293 * x38;
    const real_t x316 = 20 * u[11];
    const real_t x317 = x285 * x6;
    const real_t x318 = x285 * x33;
    const real_t x319 = u[9] * x25;
    const real_t x320 = x22 * x273;
    const real_t x321 = x23 * x275;
    const real_t x322 = 32 * u[8];
    const real_t x323 = 48 * u[10];
    const real_t x324 = x323 * x6;
    const real_t x325 = 48 * u[11];
    const real_t x326 = 96 * u[9];
    const real_t x327 = pow(u[8], 2);
    const real_t x328 = x60 * x8;
    const real_t x329 = 16 * u[10];
    const real_t x330 = x102 * x329;
    const real_t x331 = 16 * u[8];
    const real_t x332 = 16 * u[9];
    const real_t x333 = x332 * x96;
    const real_t x334 = x329 * x96;
    const real_t x335 = 16 * u[11];
    const real_t x336 = x102 * x335;
    const real_t x337 = -u[10] * x328 + u[11] * x328 + u[8] * x286 + x22 * x331 + 9 * x261 * x327 -
                        x330 - x331 * x43 - x333 + x334 + x336 + x69 * x77;
    const real_t x338 = x323 * x57;
    const real_t x339 = u[8] * x24;
    const real_t x340 = u[11] * x1;
    const real_t x341 = x325 * x57;
    const real_t x342 = u[9] * x288 + x263 * x7 - x283 * x35 - x338 - 9 * x339 + x340 * x53 + x341;
    const real_t x343 = u[11] * x22;
    const real_t x344 = -u[9] * x286 - 64 * x343;
    const real_t x345 = 64 * u[9];
    const real_t x346 = u[11] * x288 + x23 * x345;
    const real_t x347 = pow(u[6], 2);
    const real_t x348 = 9 * x347;
    const real_t x349 = x348 * x4;
    const real_t x350 = 32 * x262;
    const real_t x351 = u[7] * (u[7] * x256 + x46);
    const real_t x352 = x1 * x136;
    const real_t x353 = 120 * u[7];
    const real_t x354 = u[7] * x39;
    const real_t x355 = x273 * x57;
    const real_t x356 = 32 * u[6];
    const real_t x357 = 20 * u[6];
    const real_t x358 = x316 * x69;
    const real_t x359 = x357 * x96;
    const real_t x360 = x1 * x297;
    const real_t x361 = u[11] * x64;
    const real_t x362 = x26 * x335;
    const real_t x363 = u[6] * x64;
    const real_t x364 = x25 * x300;
    const real_t x365 = 4 * x88;
    const real_t x366 = x300 * x88;
    const real_t x367 = 11 * u[6];
    const real_t x368 = 16 * x343;
    const real_t x369 = u[6] * x57;
    const real_t x370 = 16 * x369;
    const real_t x371 = 18 * u[6];
    const real_t x372 = 20 * x267;
    const real_t x373 = x23 * x357;
    const real_t x374 = x285 * x96;
    const real_t x375 = x275 * x57;
    const real_t x376 = 48 * u[7];
    const real_t x377 = -x22 * x323;
    const real_t x378 = x23 * x323;
    const real_t x379 = x23 * x325;
    const real_t x380 = x22 * x271;
    const real_t x381 = x377 - x378 + x379 + x380;
    const real_t x382 = x263 * x4 + x272 - x324 + x332 * x43;
    const real_t x383 = x259 * x261;
    const real_t x384 = -x345 * x57 + 48 * x383;
    const real_t x385 = x257 * x7;
    const real_t x386 = 48 * x385;
    const real_t x387 = -x79;
    const real_t x388 = x387 + x69;
    const real_t x389 = x167 * x8;
    const real_t x390 = 120 * u[8];
    const real_t x391 = u[11] * x69;
    const real_t x392 = u[8] * x5;
    const real_t x393 = 64 * u[11] * x6;
    const real_t x394 = 18 * u[8];
    const real_t x395 = 4 * x25;
    const real_t x396 = x335 * x38;
    const real_t x397 = x102 * x283;
    const real_t x398 = u[7] * x57;
    const real_t x399 = 48 * u[8];
    const real_t x400 = 20 * u[9];
    const real_t x401 = x23 * x332 - x332 * x57 - x400 * x46 + x400 * x88;
    const real_t x402 = -x102 * x357 + x22 * x357 - x24 * x332 - x273 * x6 + x277 * x6 + x350 * x4;
    const real_t x403 = 96 * u[10];
    const real_t x404 = 40 * u[6];
    const real_t x405 = x335 * x96;
    const real_t x406 = x253 * x5;
    const real_t x407 = x247 * x38;
    const real_t x408 = u[6] * x1;
    const real_t x409 = u[6] * x69;
    const real_t x410 = 5 * u[8];
    const real_t x411 = x199 * x8;
    const real_t x412 = u[11] * x179;
    const real_t x413 = 4 * u[7];
    const real_t x414 = x35 * x413;
    const real_t x415 = u[11] * x80;
    const real_t x416 = 4 * u[8];
    const real_t x417 = x203 * x340;
    const real_t x418 = 8 * u[10];
    const real_t x419 = 8 * u[6];
    const real_t x420 = 12 * u[10];
    const real_t x421 = x275 * x5;
    const real_t x422 = x275 * x38;
    const real_t x423 = u[9] * x106;
    const real_t x424 = 12 * x347;
    const real_t x425 = -x273 * x96;
    const real_t x426 = -x102 * x275;
    const real_t x427 = u[6] * x6;
    const real_t x428 = x102 * x273;
    const real_t x429 = x277 * x96;
    const real_t x430 = x25 * x253 - x253 * x88 + x261 * x424 - 4 * x369 + x424 * x7 + x425 + x426 +
                        4 * x427 + x428 + x429;
    const real_t x431 = x1 * x222;
    const real_t x432 = -u[10] * x431 + u[9] * x431 - x223 * x38 + x223 * x46 + x223 * x5 +
                        x23 * x413 + 12 * x251 * x256;
    const real_t x433 = 96 * x262;
    const real_t x434 = 96 * u[11];
    const real_t x435 = 40 * u[8];
    const real_t x436 = x102 * x332;
    const real_t x437 = x253 * x39;
    const real_t x438 = u[11] * x250;
    const real_t x439 = 4 * u[6];
    const real_t x440 = u[9] * x148;
    const real_t x441 = u[9] * x82;
    const real_t x442 = x33 * x416;
    const real_t x443 = x199 * x281;
    const real_t x444 = 8 * u[9];
    const real_t x445 = u[11] * x88;
    const real_t x446 = x277 * x39;
    const real_t x447 = x277 * x43;
    const real_t x448 = x325 * x39;
    const real_t x449 = u[2] * x221;
    const real_t x450 = u[10] * x449 - u[11] * x449 - x22 * x416 + x224 * x327 - x247 * x39 +
                        x247 * x43 - x247 * x69;
    const real_t x451 = x1 * x203;
    const real_t x452 = 8 * u[8];
    element_vector[0] +=
        x128 *
        (72 * u[0] * (x22 - x23) - u[3] * x49 + u[4] * x49 - u[4] * x50 + u[5] * x50 + u[7] * x91 -
         x10 * x9 + x100 * x96 + x101 + x102 * x36 + x104 + x105 * x24 - x11 * x71 + x113 + x123 +
         x125 + x127 + 18 * x13 + 32 * x16 + 32 * x18 + x19 * x21 - x22 * x32 + x22 * x51 +
         x23 * x34 - x23 * x53 - x25 * x37 - x26 * x31 + 78 * x28 + 80 * x29 + x3 + 80 * x30 +
         x32 * x35 - x33 * x34 + x33 * x36 - x35 * x36 - x36 * x38 + x36 * x43 - x39 * x41 - x40 -
         x41 * x46 - 32 * x42 + x44 * x5 - x45 + x46 * x90 - x48 - x52 + x54 * x74 - x56 - x59 -
         x61 - x63 - x65 - x68 - 11 * x70 - 9 * x72 - x73 * x74 + x74 * x79 - x75 * x76 - x78 -
         x81 - x83 + x84 + x85 + x87 + x88 * x89 + x92 + x93 + x94 + x95 + x97 + x98 + 32 * x99);
    element_vector[1] +=
        -x128 * (-u[0] * x7 * x71 + u[1] * x135 + u[3] * x137 - u[4] * x137 + u[4] * x142 -
                 u[5] * x142 - u[6] * x118 - u[6] * x91 + x100 * x5 - x102 * x141 - x11 * x77 +
                 x113 + x115 * x24 + x115 * x79 + x116 + x117 - x120 * x24 - x120 * x79 - x121 -
                 x122 - 78 * x129 - x131 + x132 * x2 - x134 - x136 * x22 + x136 * x6 - 72 * x138 -
                 x139 - 48 * x140 + x141 * x33 - x141 * x38 - x143 * x57 - x144 * x54 + x144 * x73 -
                 x146 - x147 + x148 * (u[4] - u[5]) + x149 + x150 * x79 + x151 + x152 + 18 * x153 +
                 48 * x154 + x155 * x46 + x158 + x161 + x162 + x164 - 48 * x18 + x22 * x36 - x3 -
                 x31 * x73 + x34 * (x102 - x33 + x57) - x41 * x43 + x46 * x89 + x61 + x63 + x68 -
                 18 * x70 + 18 * x72 + x78 + x88 * x90 - x93 - x94 + 48 * x99);
    element_vector[2] +=
        x128 * (u[2] * x106 + u[2] * x165 - u[3] * x170 - u[4] * x168 + u[4] * x170 + u[5] * x168 +
                x1 * x163 - x105 * x79 - x109 * x26 - x109 * x73 + x115 * x26 + x115 * x73 + x123 +
                x133 * x14 + x134 - 48 * x138 - 72 * x140 + x141 * x35 - x141 * x43 - x143 * x6 +
                x144 * x79 + x146 + x147 - x149 - x150 * x54 + x150 * x73 - x151 - 9 * x153 +
                32 * x154 + x156 + x157 + 48 * x16 + x161 + 78 * x166 - x167 * x23 + x167 * x57 -
                x169 - x171 - x172 - 18 * x173 - x174 * x46 + x174 * x88 - x175 - x176 - 11 * x177 -
                9 * x178 + x179 * (-u[3] + u[4]) + x180 + 18 * x181 + x182 + x183 + x184 + x185 +
                48 * x186 + x188 + x189 + x23 * x36 - x25 * x44 + x32 * (-x35 + x6 + x96) -
                x38 * x44 + 48 * x42 + x56 + x65 + 9 * x70 + 11 * x72 - x87 - x97);
    element_vector[3] +=
        x227 * (u[3] * x194 + u[3] * x196 + u[4] * x148 - u[4] * x194 - u[4] * x196 - u[5] * x148 -
                u[5] * x197 - x102 * x143 + x107 + x109 * x24 - x110 * x200 + x111 * x46 - x112 +
                x125 + 96 * x130 + x14 * x21 + 8 * x140 + x141 * x22 - x152 - 4 * x153 + x164 +
                3 * x166 + x172 - 5 * x173 + x176 + 8 * x177 - 4 * x178 - x180 + 5 * x181 - x184 +
                32 * x187 - x190 + x191 * x33 - x191 * x38 - x192 - x193 + x195 * x54 - x195 * x57 -
                x195 * x73 - x198 * x46 + x198 * x88 + x199 * x23 + x199 * x5 - x199 * x57 - x201 +
                x202 + x204 + x205 + 4 * x206 + x207 * x79 + x208 * x23 + x209 + x211 + x212 +
                x213 * x38 + x220 + x226 + 16 * x30 - x31 * x38 + 4 * x42 + x45 + x59 - 24 * x62 +
                4 * x70 - 8 * x72 + x83 - x85 - x86 - x95 - x98 - 40 * x99);
    element_vector[4] +=
        x227 *
        (u[0] * x80 - u[0] * x82 + u[0] * (-x46 + x69) + 40 * u[2] * x5 + u[3] * x197 +
         u[3] * x234 - u[3] * x237 - u[4] * x197 - u[4] * x234 + u[4] * x235 + u[4] * x238 +
         u[5] * x196 - u[5] * x235 - u[5] * x238 - x100 * x6 + x105 * x39 - x109 * x79 +
         x120 * x73 + 5 * x13 + x131 + 32 * x138 + x139 + x14 * x229 + 16 * x15 * x19 - 8 * x153 -
         40 * x154 + x158 + x159 - x160 + x17 * x228 - 4 * x173 + 8 * x181 - x183 + x188 + x190 +
         x192 + x193 - x195 * x22 + x195 * x23 + x199 * x88 + x201 - x202 - x204 - x205 - x209 -
         x211 - x212 - x213 * x39 + x213 * x5 + x215 + x216 + x217 * x33 - x217 * x35 - x217 * x38 +
         x217 * x43 + x217 * x54 + x218 + x219 + x226 + x229 * x8 - x230 - x231 - x232 -
         x233 * x39 + x233 * x5 - x236 * x46 - x240 + x241 + x242 + x243 + x244 + x245 + x246 +
         x249 + 3 * x28 - x31 * x5 + 8 * x70 - 4 * x72);
    element_vector[5] +=
        x227 * (u[3] * x179 + u[3] * x238 - u[4] * x179 + u[4] * x237 + u[4] * x250 -
                4 * u[4] * x54 - u[5] * x237 - x101 + x105 * x43 + x108 + 24 * x119 - x120 * x26 +
                x127 + 3 * x129 + 32 * x130 - 8 * x138 + x14 * x163 - x141 * x23 - 5 * x153 -
                4 * x154 + x162 + x169 + x171 - 8 * x173 + x175 + 4 * x177 - 8 * x178 + 4 * x181 -
                x182 - x185 + 96 * x187 + x189 - x191 * x35 + x191 * x43 + x195 * x6 + x195 * x79 -
                x199 * x46 - x203 * x22 + x203 * x6 + x207 * x54 - x207 * x73 - x213 * x43 -
                x22 * x236 + x220 + x230 + x231 + x232 + x236 * x88 + x239 * x54 + x240 - x241 -
                x242 - x243 - x244 - x245 - x246 - x248 * x73 + x249 + x25 * x55 + 16 * x29 +
                40 * x42 + x48 - x51 * x96 + x52 + 5 * x70 - 5 * x72 + x81 - x84 - x92 - 4 * x99);
    element_vector[6] +=
        x128 *
        (u[6] * x298 + 72 * u[6] * (-x38 + x43) - u[7] * x276 + x102 * x269 - x102 * x270 -
         x2 * x297 + x22 * x270 - x23 * x270 + x24 * x325 - x25 * x279 + x252 + x253 * x9 +
         x254 * x90 + 18 * x255 + 32 * x258 - x26 * x271 - x26 * x77 + 32 * x260 + x261 * x263 +
         78 * x264 + 80 * x265 + 80 * x266 - 96 * x267 + x268 * x38 - x268 * x96 - x269 * x43 +
         x270 * x96 - x272 - x274 - x275 * x6 - x275 * x79 + x277 * x57 + x277 * x73 - x278 +
         x279 * x88 - x280 * x88 - x282 - x284 - x285 * x38 - x287 - x289 - x290 - x292 - x294 -
         11 * x296 - x299 - x300 * x69 - x302 - x304 + x305 + x307 + x308 * x89 + 9 * x309 + x311 +
         11 * x312 + x313 + x314 + x315 + x316 * x73 + x317 + x318 + 24 * x319 + x320 + x321 +
         x322 * x96 + x324 + x326 * x88 + x337 + x342 + x344 + x346 - x71 * x73);
    element_vector[7] +=
        x128 *
        (-u[10] * x352 + u[6] * x276 + u[7] * x135 - u[8] * x106 + u[9] * x352 + x102 * x376 -
         x22 * x356 - x23 * x277 + x24 * x271 + x25 * x329 - x25 * x332 - x25 * x71 - x254 * x74 -
         x254 * x89 + x256 * x348 + 48 * x258 - x26 * x277 + x26 * x71 +
         x269 * (-x102 + x33 + x39) + x271 * x73 + x279 * x46 - x280 * x46 + x282 + x287 +
         18 * x296 - x308 * x90 + 18 * x309 - x311 - 18 * x312 - x318 + x322 * x6 - x326 * x46 +
         x329 * x69 - x332 * x69 + x337 + x349 + x350 * x7 + 78 * x351 - x353 * x38 + x353 * x5 -
         72 * x354 - x355 - x357 * x39 - x358 - x359 - 18 * x360 - x361 - x362 - x363 - x364 +
         x365 * (u[10] - u[11]) + x366 + x367 * x69 + x368 + x370 + x371 * x46 + x372 + x373 +
         x374 + x375 + x376 * x57 + x381 + x382 + x384 + x73 * x77);
    element_vector[8] +=
        -x128 *
        (-u[10] * x389 + u[11] * x389 + u[8] * x165 + 78 * u[8] * x388 - x150 * x254 + x174 * x254 -
         x174 * x308 - x23 * x356 - x24 * x275 - x25 * x77 - x252 + x26 * x325 + x26 * x394 -
         48 * x260 + x268 * (x35 + x5 - x96) + x279 * x69 + x289 + x292 + x294 - 9 * x296 + x299 -
         16 * x303 - 24 * x306 + 11 * x309 + 9 * x312 - x313 - x315 + x325 * x79 + x329 * x46 +
         x329 * x88 + x330 + x333 - x334 - x335 * x88 - x336 + x342 - 32 * x343 + x347 * x75 -
         x349 - 48 * x354 + x356 * x96 - x357 * x5 - 11 * x360 + x363 + x364 - x366 + x367 * x46 -
         x370 + x371 * x69 + x381 - x386 + x39 * x390 - x390 * x43 - 96 * x391 - 72 * x392 - x393 -
         x394 * x73 + x395 * (u[10] - u[9]) + x396 + x397 + 32 * x398 + x399 * x6 + x399 * x96 +
         x401 + x402);
    element_vector[9] +=
        x227 *
        (-u[10] * x411 - u[11] * x238 + u[11] * x365 + u[11] * x411 + 40 * u[7] * x102 +
         u[7] * x196 - u[7] * x80 + 24 * u[8] * x22 - 3 * u[8] * x388 + u[8] * x395 - x102 * x404 -
         x195 * x254 - x198 * x254 + x198 * x308 + x203 * x295 - x208 * x408 + x22 * x326 -
         x22 * x403 + x22 * x404 - x24 * x277 + x25 * x418 + x256 * x350 + x26 * x400 - x26 * x410 +
         16 * x265 - 16 * x267 + x274 + x284 - x301 * x88 + x304 - x305 - 12 * x306 - x314 -
         8 * x319 - x321 + x339 + x344 + x355 + x359 + x362 - x373 - x375 + x382 + 96 * x383 +
         32 * x385 + x39 * x416 + x39 * x419 + 8 * x391 - 8 * x392 - x400 * x73 + x401 - x405 -
         x406 - x407 - 5 * x409 + x410 * x73 - x412 - x414 - x415 - x416 * x43 - x416 * x6 -
         x416 * x96 + x417 + x419 * x46 + x420 * x69 + x421 + x422 + x423 + x430 + x432);
    element_vector[10] +=
        x227 *
        (-u[11] * x237 + u[6] * (x387 + x73) + u[7] * x179 - u[7] * x197 - u[8] * x148 +
         u[8] * x238 + u[9] * x196 - x102 * x439 + x199 * x308 - x203 * x408 + x208 * x295 +
         x22 * x439 - x23 * x439 - x236 * x254 + x25 * x420 + 5 * x255 + x256 * x433 +
         16 * x257 * x261 + x259 * x66 + 3 * x264 - 8 * x267 + 8 * x303 - 8 * x306 - 12 * x319 +
         x322 * x5 + x326 * x6 - x332 * x46 + x335 * x69 - 32 * x354 + 24 * x369 - x374 + x377 +
         x378 - x379 + x380 + x384 + x386 + x393 + x397 + 40 * x398 + x4 * x433 + x403 * x57 -
         x403 * x6 + x405 + x406 + x407 - 4 * x409 + x412 + x414 + x415 - x417 - x418 * x46 +
         x418 * x69 - x420 * x88 - x421 - x422 - x423 + x425 + x426 - 24 * x427 + x428 + x429 +
         x432 - x434 * x57 - x435 * x6 - x436 - x437 - x438 + x439 * x46 + x439 * x96 - x440 -
         x441 - x442 + x443 + x444 * x88 + 12 * x445 + x446 + x447 + x448 + x450);
    element_vector[11] +=
        x227 *
        (u[10] * x451 + 5 * u[6] * x46 - 24 * u[7] * x23 - u[8] * x237 + u[8] * x82 - u[9] * x451 +
         x102 * x413 - x199 * x254 - x217 * x254 + x23 * x403 - x23 * x404 - x23 * x434 +
         x236 * x308 - x24 * x316 + x25 * x301 + x25 * x452 + x256 * x263 + x26 * x275 + 16 * x266 +
         x278 + x290 + 5 * x296 + x302 + 12 * x303 - x307 + 5 * x309 - x310 - 5 * x312 +
         x316 * x79 - x317 - 4 * x319 - x320 + x332 * x88 + x338 - x341 + x346 + 3 * x351 +
         8 * x354 + x358 - 5 * x360 + x361 - x368 - x372 + x38 * x413 + x38 * x452 + 32 * x383 +
         96 * x385 - x396 + 4 * x398 + x402 + x404 * x96 - 8 * x409 - x413 * x5 - x418 * x88 -
         x419 * x5 - x420 * x46 + x430 - x435 * x96 + x436 + x437 + x438 + x440 + x441 + x442 -
         x443 - x444 * x46 + 8 * x445 - x446 - x447 - x448 + x450);
}

static SFEM_INLINE void tri6_add_momentum_rhs_kernel(const real_t px0,
                                                     const real_t px1,
                                                     const real_t px2,
                                                     const real_t py0,
                                                     const real_t py1,
                                                     const real_t py2,
                                                     real_t *const SFEM_RESTRICT u,
                                                     real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = 4 * u[4];
    const real_t x1 = (px0 - px1) * (py0 - py2) - (px0 - px2) * (py0 - py1);
    const real_t x2 = (1.0 / 360.0) * x1;
    const real_t x3 = 4 * u[5];
    const real_t x4 = 4 * u[3];
    const real_t x5 = (1.0 / 90.0) * x1;
    const real_t x6 = 4 * u[10];
    const real_t x7 = 4 * u[11];
    const real_t x8 = 4 * u[9];
    element_vector[0] += -x2 * (-6 * u[0] + u[1] + u[2] + x0);
    element_vector[1] += -x2 * (u[0] - 6 * u[1] + u[2] + x3);
    element_vector[2] += -x2 * (u[0] + u[1] - 6 * u[2] + x4);
    element_vector[3] += x5 * (-u[2] + 8 * u[3] + x0 + x3);
    element_vector[4] += x5 * (-u[0] + 8 * u[4] + x3 + x4);
    element_vector[5] += x5 * (-u[1] + 8 * u[5] + x0 + x4);
    element_vector[6] += -x2 * (-6 * u[6] + u[7] + u[8] + x6);
    element_vector[7] += -x2 * (u[6] - 6 * u[7] + u[8] + x7);
    element_vector[8] += -x2 * (u[6] + u[7] - 6 * u[8] + x8);
    element_vector[9] += x5 * (-u[8] + 8 * u[9] + x6 + x7);
    element_vector[10] += x5 * (8 * u[10] - u[6] + x7 + x8);
    element_vector[11] += x5 * (8 * u[11] - u[7] + x6 + x8);
}

static SFEM_INLINE void tri6_add_diffusion_rhs_kernel(const real_t px0,
                                                      const real_t px1,
                                                      const real_t px2,
                                                      const real_t py0,
                                                      const real_t py1,
                                                      const real_t py2,
                                                      const real_t dt,
                                                      const real_t nu,
                                                      real_t *const SFEM_RESTRICT u,
                                                      real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = px0 - px2;
    const real_t x1 = pow(x0, 2);
    const real_t x2 = u[1] * x1;
    const real_t x3 = py0 - py2;
    const real_t x4 = pow(x3, 2);
    const real_t x5 = u[1] * x4;
    const real_t x6 = px0 - px1;
    const real_t x7 = pow(x6, 2);
    const real_t x8 = u[0] * x7;
    const real_t x9 = u[0] * x1;
    const real_t x10 = py0 - py1;
    const real_t x11 = pow(x10, 2);
    const real_t x12 = u[0] * x11;
    const real_t x13 = u[0] * x4;
    const real_t x14 = x0 * x6;
    const real_t x15 = u[1] * x14;
    const real_t x16 = x10 * x3;
    const real_t x17 = u[1] * x16;
    const real_t x18 = u[0] * x14;
    const real_t x19 = u[0] * x16;
    const real_t x20 = 4 * u[3];
    const real_t x21 = -x1 * x20 + x14 * x20 + x16 * x20 - x20 * x4;
    const real_t x22 = 4 * u[5];
    const real_t x23 = -x11 * x22 + x14 * x22 + x16 * x22 - x22 * x7;
    const real_t x24 = u[2] * x7;
    const real_t x25 = u[2] * x11;
    const real_t x26 = u[2] * x14;
    const real_t x27 = u[2] * x16;
    const real_t x28 = x24 + x25 - x26 - x27;
    const real_t x29 = x3 * x6;
    const real_t x30 = x0 * x10;
    const real_t x31 = dt * nu;
    const real_t x32 = x31 / (6 * x29 - 6 * x30);
    const real_t x33 = x26 + x27;
    const real_t x34 = 4 * u[4];
    const real_t x35 = -x18 - x19;
    const real_t x36 = -x14 * x34 - x16 * x34 + x35;
    const real_t x37 = x15 + x17;
    const real_t x38 = x12 + x8;
    const real_t x39 = 2 * u[3];
    const real_t x40 = 2 * u[4];
    const real_t x41 = x14 * x39;
    const real_t x42 = x16 * x39;
    const real_t x43 = 2 * u[5];
    const real_t x44 = x14 * x43;
    const real_t x45 = x16 * x43;
    const real_t x46 = x14 * x40;
    const real_t x47 = x16 * x40;
    const real_t x48 =
        x11 * x39 - x11 * x40 + x37 + x39 * x7 - x40 * x7 - x41 - x42 - x44 - x45 + x46 + x47;
    const real_t x49 = (2.0 / 3.0) * x31 / (x29 - x30);
    const real_t x50 = x1 * x40;
    const real_t x51 = x4 * x40;
    const real_t x52 = x1 * x43;
    const real_t x53 = x4 * x43;
    const real_t x54 = u[8] * x7;
    const real_t x55 = u[8] * x11;
    const real_t x56 = u[6] * x7;
    const real_t x57 = u[6] * x1;
    const real_t x58 = u[6] * x11;
    const real_t x59 = u[6] * x4;
    const real_t x60 = u[8] * x14;
    const real_t x61 = u[8] * x16;
    const real_t x62 = u[6] * x14;
    const real_t x63 = u[6] * x16;
    const real_t x64 = 4 * u[9];
    const real_t x65 = -x1 * x64 + x14 * x64 + x16 * x64 - x4 * x64;
    const real_t x66 = 4 * u[11];
    const real_t x67 = -x11 * x66 + x14 * x66 + x16 * x66 - x66 * x7;
    const real_t x68 = u[7] * x1;
    const real_t x69 = u[7] * x4;
    const real_t x70 = u[7] * x14;
    const real_t x71 = u[7] * x16;
    const real_t x72 = x68 + x69 - x70 - x71;
    const real_t x73 = x60 + x61;
    const real_t x74 = -x62;
    const real_t x75 = -x63;
    const real_t x76 = 4 * u[10];
    const real_t x77 = -x14 * x76 - x16 * x76 + x74 + x75;
    const real_t x78 = x57 + x59;
    const real_t x79 = x70 + x71;
    const real_t x80 = 2 * u[9];
    const real_t x81 = x7 * x80;
    const real_t x82 = x11 * x80;
    const real_t x83 = 2 * u[10];
    const real_t x84 = x7 * x83;
    const real_t x85 = x11 * x83;
    const real_t x86 = x14 * x83;
    const real_t x87 = x16 * x83;
    const real_t x88 = 2 * u[11];
    const real_t x89 = x14 * x88;
    const real_t x90 = x16 * x88;
    const real_t x91 = x14 * x80;
    const real_t x92 = x16 * x80;
    const real_t x93 =
        -x1 * x83 + x1 * x88 - x4 * x83 + x4 * x88 + x73 + x86 + x87 - x89 - x90 - x91 - x92;
    element_vector[0] += -x32 * (3 * x12 + 3 * x13 - x15 - x17 - 6 * x18 - 6 * x19 + x2 + x21 +
                                 x23 + x28 + x5 + 3 * x8 + 3 * x9);
    element_vector[1] += -x32 * (x13 + 3 * x2 + x21 + x33 + x36 + 3 * x5 + x9);
    element_vector[2] += -x32 * (x23 + 3 * x24 + 3 * x25 + x36 + x37 + x38);
    element_vector[3] += x49 * (u[0] * x1 + u[0] * x4 + u[1] * x1 + u[1] * x4 - x1 * x39 - x18 -
                                x19 - x39 * x4 - x48);
    element_vector[4] += x49 * (x33 + x48 - x50 - x51 + x52 + x53);
    element_vector[5] += x49 * (-x11 * x43 + x28 + x35 + x38 + x41 + x42 - x43 * x7 + x44 + x45 -
                                x46 - x47 + x50 + x51 - x52 - x53);
    element_vector[6] += -x32 * (x54 + x55 + 3 * x56 + 3 * x57 + 3 * x58 + 3 * x59 - x60 - x61 -
                                 6 * x62 - 6 * x63 + x65 + x67 + x72);
    element_vector[7] += -x32 * (x65 + 3 * x68 + 3 * x69 + x73 + x77 + x78);
    element_vector[8] += -x32 * (3 * x54 + 3 * x55 + x56 + x58 + x67 + x77 + x79);
    element_vector[9] += x49 * (-x1 * x80 - x4 * x80 + x72 + x74 + x75 + x78 - x81 - x82 + x84 +
                                x85 - x86 - x87 + x89 + x90 + x91 + x92);
    element_vector[10] += x49 * (x79 + x81 + x82 - x84 - x85 + x93);
    element_vector[11] += x49 * (u[6] * x11 + u[6] * x7 + u[8] * x11 + u[8] * x7 - x11 * x88 - x62 -
                                 x63 - x7 * x88 - x93);
}

static SFEM_INLINE void tri6_add_convection_rhs_kernel(const real_t px0,
                                                       const real_t px1,
                                                       const real_t px2,
                                                       const real_t py0,
                                                       const real_t py1,
                                                       const real_t py2,
                                                       const real_t dt,
                                                       const real_t nu,
                                                       real_t *const SFEM_RESTRICT u,
                                                       real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = pow(u[2], 2);
    const real_t x1 = py0 - py1;
    const real_t x2 = 9 * x1;
    const real_t x3 = x0 * x2;
    const real_t x4 = px0 - px1;
    const real_t x5 = u[10] * x4;
    const real_t x6 = u[4] * x1;
    const real_t x7 = px0 - px2;
    const real_t x8 = py0 - py2;
    const real_t x9 = -u[10] * x7 - u[4] * x8 + x5 + x6;
    const real_t x10 = 12 * u[0];
    const real_t x11 = u[1] * x7;
    const real_t x12 = u[2] * x4;
    const real_t x13 = u[6] * (x11 - x12);
    const real_t x14 = -x1;
    const real_t x15 = pow(u[3], 2);
    const real_t x16 = x14 * x15;
    const real_t x17 = pow(u[5], 2);
    const real_t x18 = x17 * x8;
    const real_t x19 = -x8;
    const real_t x20 = pow(u[4], 2);
    const real_t x21 = 48 * x20;
    const real_t x22 = u[3] * x1;
    const real_t x23 = u[5] * x8;
    const real_t x24 = u[6] * x4;
    const real_t x25 = u[0] * x1;
    const real_t x26 = u[6] * x7;
    const real_t x27 = x25 + x26;
    const real_t x28 = u[0] * (u[0] * x19 - x24 + x27);
    const real_t x29 = x15 * x8;
    const real_t x30 = x14 * x17;
    const real_t x31 = 96 * u[3];
    const real_t x32 = 80 * u[5];
    const real_t x33 = u[9] * x7;
    const real_t x34 = 80 * u[3];
    const real_t x35 = u[11] * x4;
    const real_t x36 = 48 * u[0];
    const real_t x37 = 48 * u[5];
    const real_t x38 = u[9] * x4;
    const real_t x39 = u[10] * x7;
    const real_t x40 = x37 * x39;
    const real_t x41 = 32 * u[3];
    const real_t x42 = u[11] * x12;
    const real_t x43 = u[11] * x7;
    const real_t x44 = 32 * u[5];
    const real_t x45 = x43 * x44;
    const real_t x46 = u[1] * x8;
    const real_t x47 = 32 * u[4];
    const real_t x48 = x38 * x47;
    const real_t x49 = 24 * x24;
    const real_t x50 = 24 * x26;
    const real_t x51 = 20 * u[2];
    const real_t x52 = x5 * x51;
    const real_t x53 = 20 * u[1];
    const real_t x54 = u[7] * x4;
    const real_t x55 = 20 * u[5];
    const real_t x56 = x54 * x55;
    const real_t x57 = u[4] * x8;
    const real_t x58 = 16 * u[1];
    const real_t x59 = x57 * x58;
    const real_t x60 = 16 * u[2];
    const real_t x61 = x57 * x60;
    const real_t x62 = u[9] * x12;
    const real_t x63 = 16 * x62;
    const real_t x64 = 16 * x6;
    const real_t x65 = u[3] * x64;
    const real_t x66 = 16 * x7;
    const real_t x67 = u[8] * x66;
    const real_t x68 = u[5] * x67;
    const real_t x69 = u[2] * x1;
    const real_t x70 = u[1] * x69;
    const real_t x71 = 11 * u[8];
    const real_t x72 = u[1] * x25;
    const real_t x73 = u[7] * x7;
    const real_t x74 = 9 * u[0];
    const real_t x75 = 9 * x7;
    const real_t x76 = u[0] * u[8];
    const real_t x77 = 9 * u[8];
    const real_t x78 = x12 * x77;
    const real_t x79 = u[8] * x4;
    const real_t x80 = 4 * x79;
    const real_t x81 = u[3] * x80;
    const real_t x82 = 4 * x73;
    const real_t x83 = u[4] * x82;
    const real_t x84 = u[4] * x80;
    const real_t x85 = u[5] * x82;
    const real_t x86 = u[2] * x25;
    const real_t x87 = 9 * x86;
    const real_t x88 = u[0] * x8;
    const real_t x89 = 9 * u[2];
    const real_t x90 = 11 * u[2];
    const real_t x91 = 11 * x12;
    const real_t x92 = x6 * x60;
    const real_t x93 = x23 * x60;
    const real_t x94 = u[4] * x67;
    const real_t x95 = x39 * x53;
    const real_t x96 = u[5] * x1;
    const real_t x97 = x53 * x96;
    const real_t x98 = x43 * x47;
    const real_t x99 = u[9] * x11;
    const real_t x100 = 32 * u[2];
    const real_t x101 = x38 * x41;
    const real_t x102 = u[3] * x8;
    const real_t x103 = 48 * u[4];
    const real_t x104 = x103 * x39;
    const real_t x105 = 96 * u[5];
    const real_t x106 = 48 * x5;
    const real_t x107 = u[4] * x106;
    const real_t x108 = u[0] * x46;
    const real_t x109 = 16 * u[5];
    const real_t x110 = u[8] * x7;
    const real_t x111 = 20 * u[3];
    const real_t x112 = u[3] * x106;
    const real_t x113 = x1 * x21 - x102 * x51 - x107 - 9 * x108 + x109 * x57 + x110 * x111 + x112;
    const real_t x114 = pow(u[1], 2);
    const real_t x115 = 16 * u[4];
    const real_t x116 = x115 * x35;
    const real_t x117 = x109 * x33;
    const real_t x118 = 9 * x11;
    const real_t x119 = u[11] * x11;
    const real_t x120 = 16 * u[3];
    const real_t x121 = x120 * x35;
    const real_t x122 = x115 * x33;
    const real_t x123 = u[7] * x118 + 9 * x114 * x19 - x115 * x54 - x116 - x117 + 16 * x119 +
                        x120 * x54 + x121 + x122 - x22 * x58 + x58 * x6;
    const real_t x124 = 64 * u[5];
    const real_t x125 = x120 * x57 + x124 * x38;
    const real_t x126 = 64 * u[3];
    const real_t x127 = -u[5] * x64 - x126 * x43;
    const real_t x128 = (1.0 / 2520.0) * dt;
    const real_t x129 = u[1] * (x46 - x73);
    const real_t x130 = x1 * x15;
    const real_t x131 = 48 * x130;
    const real_t x132 = pow(u[0], 2);
    const real_t x133 = 9 * x132;
    const real_t x134 = x133 * x8;
    const real_t x135 = 12 * x23 + 12 * x43;
    const real_t x136 = 120 * u[1];
    const real_t x137 = 120 * x54;
    const real_t x138 = u[1] * x57;
    const real_t x139 = x126 * x39;
    const real_t x140 = u[2] * x6;
    const real_t x141 = 32 * u[0];
    const real_t x142 = 24 * x73;
    const real_t x143 = 20 * u[0];
    const real_t x144 = 18 * u[0];
    const real_t x145 = 16 * u[0];
    const real_t x146 = x145 * x5;
    const real_t x147 = x24 * x74;
    const real_t x148 = 4 * x26;
    const real_t x149 = x26 * x74;
    const real_t x150 = 11 * u[0];
    const real_t x151 = x145 * x39;
    const real_t x152 = x120 * x23;
    const real_t x153 = u[7] * x12;
    const real_t x154 = u[1] * x39;
    const real_t x155 = 48 * u[3];
    const real_t x156 = -x103 * x43;
    const real_t x157 = x37 * x43;
    const real_t x158 = x156 + x157 + 20 * x42;
    const real_t x159 = x103 * x38;
    const real_t x160 = x155 * x38;
    const real_t x161 = -x159 + x160;
    const real_t x162 = x109 * x38 - x109 * x5 + x24 * x55 - x55 * x79;
    const real_t x163 = 32 * x20;
    const real_t x164 = -x143 * x35 + x143 * x43 - x145 * x23 + x163 * x8 + x39 * x44 - x39 * x47;
    const real_t x165 = 12 * x22 + 12 * x38;
    const real_t x166 = u[2] * (u[2] * x14 + x79);
    const real_t x167 = 120 * u[2];
    const real_t x168 = 120 * x110;
    const real_t x169 = x47 * x5;
    const real_t x170 = 24 * x79;
    const real_t x171 = x143 * x33;
    const real_t x172 = x111 * x73;
    const real_t x173 = x7 * x76;
    const real_t x174 = 18 * u[2];
    const real_t x175 = x145 * x22;
    const real_t x176 = x120 * x39;
    const real_t x177 = u[6] * x11;
    const real_t x178 = u[6] * x12;
    const real_t x179 = 4 * x24;
    const real_t x180 = x120 * x43;
    const real_t x181 = u[8] * x11;
    const real_t x182 = x143 * x38;
    const real_t x183 = 20 * x99;
    const real_t x184 = x111 * x26;
    const real_t x185 = x41 * x5;
    const real_t x186 = u[2] * x96;
    const real_t x187 = x17 * x19;
    const real_t x188 = -x124 * x5 + 48 * x187;
    const real_t x189 = -x104 + x109 * x22 + x21 * x8 + x40;
    const real_t x190 = x155 * x6;
    const real_t x191 = 40 * u[0];
    const real_t x192 = x22 * x44;
    const real_t x193 = x44 * x6;
    const real_t x194 = 12 * x79;
    const real_t x195 = 8 * u[0];
    const real_t x196 = 8 * x24;
    const real_t x197 = 8 * x79;
    const real_t x198 = 5 * u[2];
    const real_t x199 = 4 * u[2];
    const real_t x200 = 4 * u[5];
    const real_t x201 = x200 * x54;
    const real_t x202 = x200 * x25;
    const real_t x203 = 4 * u[1];
    const real_t x204 = x203 * x96;
    const real_t x205 = 4 * x186;
    const real_t x206 = u[4] * x110;
    const real_t x207 = 5 * u[0];
    const real_t x208 = 8 * u[1];
    const real_t x209 = x10 * x6;
    const real_t x210 = 12 * u[2];
    const real_t x211 = x210 * x22;
    const real_t x212 = x109 * x35;
    const real_t x213 = 96 * u[4];
    const real_t x214 = 12 * x132;
    const real_t x215 = -x35 * x41;
    const real_t x216 = -x33 * x47;
    const real_t x217 = 4 * u[0];
    const real_t x218 = x35 * x47;
    const real_t x219 = x33 * x44;
    const real_t x220 = x1 * x214 - x10 * x24 + x10 * x26 + x19 * x214 + x215 + x216 + x217 * x39 -
                        x217 * x5 + x218 + x219;
    const real_t x221 = 12 * x8;
    const real_t x222 = 12 * u[1];
    const real_t x223 = 12 * u[7];
    const real_t x224 = 12 * x4;
    const real_t x225 = u[7] * x224;
    const real_t x226 =
        -u[3] * x225 + u[4] * x225 - x11 * x223 + x114 * x221 - 4 * x119 + x22 * x222 - x222 * x6;
    const real_t x227 = (1.0 / 630.0) * dt;
    const real_t x228 = 16 * x1;
    const real_t x229 = 96 * x20;
    const real_t x230 = x37 * x57;
    const real_t x231 = x41 * x57;
    const real_t x232 = x23 * x41;
    const real_t x233 = 24 * u[0];
    const real_t x234 = 12 * x24;
    const real_t x235 = 12 * x26;
    const real_t x236 = 8 * u[2];
    const real_t x237 = 8 * x26;
    const real_t x238 = 8 * x73;
    const real_t x239 = 4 * u[3];
    const real_t x240 = x110 * x239;
    const real_t x241 = x102 * x217;
    const real_t x242 = x239 * x46;
    const real_t x243 = x102 * x199;
    const real_t x244 = x10 * x57;
    const real_t x245 = x222 * x23;
    const real_t x246 = x120 * x33;
    const real_t x247 = 12 * u[8];
    const real_t x248 = 12 * u[5];
    const real_t x249 =
        12 * x0 * x14 + x110 * x248 + x12 * x247 - 12 * x206 - x210 * x23 + x210 * x57 + 4 * x62;
    const real_t x250 = 12 * x73;
    const real_t x251 = pow(u[7], 2);
    const real_t x252 = x251 * x75;
    const real_t x253 = 12 * u[6];
    const real_t x254 = u[7] * x8;
    const real_t x255 = u[0] * (u[8] * x1 - x254);
    const real_t x256 = -x7;
    const real_t x257 = pow(u[11], 2);
    const real_t x258 = x256 * x257;
    const real_t x259 = pow(u[9], 2);
    const real_t x260 = x259 * x4;
    const real_t x261 = -x4;
    const real_t x262 = pow(u[10], 2);
    const real_t x263 = 48 * x262;
    const real_t x264 = u[6] * (u[6] * x261 + x27 - x88);
    const real_t x265 = x257 * x4;
    const real_t x266 = x256 * x259;
    const real_t x267 = u[11] * x25;
    const real_t x268 = 80 * u[11];
    const real_t x269 = 80 * u[9];
    const real_t x270 = 48 * u[6];
    const real_t x271 = 48 * u[9];
    const real_t x272 = x271 * x6;
    const real_t x273 = 32 * u[10];
    const real_t x274 = x23 * x273;
    const real_t x275 = 32 * u[11];
    const real_t x276 = 32 * x102;
    const real_t x277 = 32 * u[9];
    const real_t x278 = x22 * x277;
    const real_t x279 = 24 * u[10];
    const real_t x280 = 24 * u[11];
    const real_t x281 = u[9] * x8;
    const real_t x282 = x281 * x51;
    const real_t x283 = 20 * u[7];
    const real_t x284 = x283 * x57;
    const real_t x285 = 20 * u[8];
    const real_t x286 = 16 * x39;
    const real_t x287 = u[11] * x286;
    const real_t x288 = 16 * x5;
    const real_t x289 = u[7] * x288;
    const real_t x290 = u[8] * x288;
    const real_t x291 = u[1] * x228;
    const real_t x292 = u[9] * x291;
    const real_t x293 = 16 * u[7];
    const real_t x294 = x23 * x293;
    const real_t x295 = u[8] * x1;
    const real_t x296 = u[1] * x295;
    const real_t x297 = u[1] * u[6];
    const real_t x298 = 9 * x46;
    const real_t x299 = u[7] * x298;
    const real_t x300 = 9 * u[6];
    const real_t x301 = 4 * u[10];
    const real_t x302 = x301 * x69;
    const real_t x303 = u[11] * x46;
    const real_t x304 = 4 * x303;
    const real_t x305 = x301 * x46;
    const real_t x306 = u[9] * x69;
    const real_t x307 = 4 * x306;
    const real_t x308 = u[6] * x8;
    const real_t x309 = u[7] * x24;
    const real_t x310 = u[7] * x26;
    const real_t x311 = 9 * x310;
    const real_t x312 = u[7] * x79;
    const real_t x313 = u[10] * x291;
    const real_t x314 = u[7] * x286;
    const real_t x315 = x293 * x38;
    const real_t x316 = 20 * u[11];
    const real_t x317 = x285 * x6;
    const real_t x318 = x285 * x33;
    const real_t x319 = u[9] * x25;
    const real_t x320 = x22 * x273;
    const real_t x321 = x23 * x275;
    const real_t x322 = 32 * u[8];
    const real_t x323 = 48 * u[10];
    const real_t x324 = x323 * x6;
    const real_t x325 = 48 * u[11];
    const real_t x326 = 96 * u[9];
    const real_t x327 = pow(u[8], 2);
    const real_t x328 = x60 * x8;
    const real_t x329 = 16 * u[10];
    const real_t x330 = x102 * x329;
    const real_t x331 = 16 * u[8];
    const real_t x332 = 16 * u[9];
    const real_t x333 = x332 * x96;
    const real_t x334 = x329 * x96;
    const real_t x335 = 16 * u[11];
    const real_t x336 = x102 * x335;
    const real_t x337 = -u[10] * x328 + u[11] * x328 + u[8] * x286 + x22 * x331 + 9 * x261 * x327 -
                        x330 - x331 * x43 - x333 + x334 + x336 + x69 * x77;
    const real_t x338 = x323 * x57;
    const real_t x339 = u[8] * x24;
    const real_t x340 = u[11] * x1;
    const real_t x341 = x325 * x57;
    const real_t x342 = u[9] * x288 + x263 * x7 - x283 * x35 - x338 - 9 * x339 + x340 * x53 + x341;
    const real_t x343 = u[11] * x22;
    const real_t x344 = -u[9] * x286 - 64 * x343;
    const real_t x345 = 64 * u[9];
    const real_t x346 = u[11] * x288 + x23 * x345;
    const real_t x347 = pow(u[6], 2);
    const real_t x348 = 9 * x347;
    const real_t x349 = x348 * x4;
    const real_t x350 = 32 * x262;
    const real_t x351 = u[7] * (u[7] * x256 + x46);
    const real_t x352 = x1 * x136;
    const real_t x353 = 120 * u[7];
    const real_t x354 = u[7] * x39;
    const real_t x355 = x273 * x57;
    const real_t x356 = 32 * u[6];
    const real_t x357 = 20 * u[6];
    const real_t x358 = x316 * x69;
    const real_t x359 = x357 * x96;
    const real_t x360 = x1 * x297;
    const real_t x361 = u[11] * x64;
    const real_t x362 = x26 * x335;
    const real_t x363 = u[6] * x64;
    const real_t x364 = x25 * x300;
    const real_t x365 = 4 * x88;
    const real_t x366 = x300 * x88;
    const real_t x367 = 11 * u[6];
    const real_t x368 = 16 * x343;
    const real_t x369 = u[6] * x57;
    const real_t x370 = 16 * x369;
    const real_t x371 = 18 * u[6];
    const real_t x372 = 20 * x267;
    const real_t x373 = x23 * x357;
    const real_t x374 = x285 * x96;
    const real_t x375 = x275 * x57;
    const real_t x376 = 48 * u[7];
    const real_t x377 = -x22 * x323;
    const real_t x378 = x23 * x323;
    const real_t x379 = x23 * x325;
    const real_t x380 = x22 * x271;
    const real_t x381 = x377 - x378 + x379 + x380;
    const real_t x382 = x263 * x4 + x272 - x324 + x332 * x43;
    const real_t x383 = x259 * x261;
    const real_t x384 = -x345 * x57 + 48 * x383;
    const real_t x385 = x257 * x7;
    const real_t x386 = 48 * x385;
    const real_t x387 = -x79;
    const real_t x388 = x387 + x69;
    const real_t x389 = x167 * x8;
    const real_t x390 = 120 * u[8];
    const real_t x391 = u[11] * x69;
    const real_t x392 = u[8] * x5;
    const real_t x393 = 64 * u[11] * x6;
    const real_t x394 = 18 * u[8];
    const real_t x395 = 4 * x25;
    const real_t x396 = x335 * x38;
    const real_t x397 = x102 * x283;
    const real_t x398 = u[7] * x57;
    const real_t x399 = 48 * u[8];
    const real_t x400 = 20 * u[9];
    const real_t x401 = x23 * x332 - x332 * x57 - x400 * x46 + x400 * x88;
    const real_t x402 = -x102 * x357 + x22 * x357 - x24 * x332 - x273 * x6 + x277 * x6 + x350 * x4;
    const real_t x403 = 96 * u[10];
    const real_t x404 = 40 * u[6];
    const real_t x405 = x335 * x96;
    const real_t x406 = x253 * x5;
    const real_t x407 = x247 * x38;
    const real_t x408 = u[6] * x1;
    const real_t x409 = u[6] * x69;
    const real_t x410 = 5 * u[8];
    const real_t x411 = x199 * x8;
    const real_t x412 = u[11] * x179;
    const real_t x413 = 4 * u[7];
    const real_t x414 = x35 * x413;
    const real_t x415 = u[11] * x80;
    const real_t x416 = 4 * u[8];
    const real_t x417 = x203 * x340;
    const real_t x418 = 8 * u[10];
    const real_t x419 = 8 * u[6];
    const real_t x420 = 12 * u[10];
    const real_t x421 = x275 * x5;
    const real_t x422 = x275 * x38;
    const real_t x423 = u[9] * x106;
    const real_t x424 = 12 * x347;
    const real_t x425 = -x273 * x96;
    const real_t x426 = -x102 * x275;
    const real_t x427 = u[6] * x6;
    const real_t x428 = x102 * x273;
    const real_t x429 = x277 * x96;
    const real_t x430 = x25 * x253 - x253 * x88 + x261 * x424 - 4 * x369 + x424 * x7 + x425 + x426 +
                        4 * x427 + x428 + x429;
    const real_t x431 = x1 * x222;
    const real_t x432 = -u[10] * x431 + u[9] * x431 - x223 * x38 + x223 * x46 + x223 * x5 +
                        x23 * x413 + 12 * x251 * x256;
    const real_t x433 = 96 * x262;
    const real_t x434 = 96 * u[11];
    const real_t x435 = 40 * u[8];
    const real_t x436 = x102 * x332;
    const real_t x437 = x253 * x39;
    const real_t x438 = u[11] * x250;
    const real_t x439 = 4 * u[6];
    const real_t x440 = u[9] * x148;
    const real_t x441 = u[9] * x82;
    const real_t x442 = x33 * x416;
    const real_t x443 = x199 * x281;
    const real_t x444 = 8 * u[9];
    const real_t x445 = u[11] * x88;
    const real_t x446 = x277 * x39;
    const real_t x447 = x277 * x43;
    const real_t x448 = x325 * x39;
    const real_t x449 = u[2] * x221;
    const real_t x450 = u[10] * x449 - u[11] * x449 - x22 * x416 + x224 * x327 - x247 * x39 +
                        x247 * x43 - x247 * x69;
    const real_t x451 = x1 * x203;
    const real_t x452 = 8 * u[8];
    element_vector[0] +=
        x128 *
        (72 * u[0] * (x22 - x23) - u[3] * x49 + u[4] * x49 - u[4] * x50 + u[5] * x50 + u[7] * x91 -
         x10 * x9 + x100 * x96 + x101 + x102 * x36 + x104 + x105 * x24 - x11 * x71 + x113 + x123 +
         x125 + x127 + 18 * x13 + 32 * x16 + 32 * x18 + x19 * x21 - x22 * x32 + x22 * x51 +
         x23 * x34 - x23 * x53 - x25 * x37 - x26 * x31 + 78 * x28 + 80 * x29 + x3 + 80 * x30 +
         x32 * x35 - x33 * x34 + x33 * x36 - x35 * x36 - x36 * x38 + x36 * x43 - x39 * x41 - x40 -
         x41 * x46 - 32 * x42 + x44 * x5 - x45 + x46 * x90 - x48 - x52 + x54 * x74 - x56 - x59 -
         x61 - x63 - x65 - x68 - 11 * x70 - 9 * x72 - x73 * x74 + x74 * x79 - x75 * x76 - x78 -
         x81 - x83 + x84 + x85 + x87 + x88 * x89 + x92 + x93 + x94 + x95 + x97 + x98 + 32 * x99);
    element_vector[1] +=
        -x128 * (-u[0] * x7 * x71 + u[1] * x135 + u[3] * x137 - u[4] * x137 + u[4] * x142 -
                 u[5] * x142 - u[6] * x118 - u[6] * x91 + x100 * x5 - x102 * x141 - x11 * x77 +
                 x113 + x115 * x24 + x115 * x79 + x116 + x117 - x120 * x24 - x120 * x79 - x121 -
                 x122 - 78 * x129 - x131 + x132 * x2 - x134 - x136 * x22 + x136 * x6 - 72 * x138 -
                 x139 - 48 * x140 + x141 * x33 - x141 * x38 - x143 * x57 - x144 * x54 + x144 * x73 -
                 x146 - x147 + x148 * (u[4] - u[5]) + x149 + x150 * x79 + x151 + x152 + 18 * x153 +
                 48 * x154 + x155 * x46 + x158 + x161 + x162 + x164 - 48 * x18 + x22 * x36 - x3 -
                 x31 * x73 + x34 * (x102 - x33 + x57) - x41 * x43 + x46 * x89 + x61 + x63 + x68 -
                 18 * x70 + 18 * x72 + x78 + x88 * x90 - x93 - x94 + 48 * x99);
    element_vector[2] +=
        x128 * (u[2] * x106 + u[2] * x165 - u[3] * x170 - u[4] * x168 + u[4] * x170 + u[5] * x168 +
                x1 * x163 - x105 * x79 - x109 * x26 - x109 * x73 + x115 * x26 + x115 * x73 + x123 +
                x133 * x14 + x134 - 48 * x138 - 72 * x140 + x141 * x35 - x141 * x43 - x143 * x6 +
                x144 * x79 + x146 + x147 - x149 - x150 * x54 + x150 * x73 - x151 - 9 * x153 +
                32 * x154 + x156 + x157 + 48 * x16 + x161 + 78 * x166 - x167 * x23 + x167 * x57 -
                x169 - x171 - x172 - 18 * x173 - x174 * x46 + x174 * x88 - x175 - x176 - 11 * x177 -
                9 * x178 + x179 * (-u[3] + u[4]) + x180 + 18 * x181 + x182 + x183 + x184 + x185 +
                48 * x186 + x188 + x189 + x23 * x36 - x25 * x44 + x32 * (-x35 + x6 + x96) -
                x38 * x44 + 48 * x42 + x56 + x65 + 9 * x70 + 11 * x72 - x87 - x97);
    element_vector[3] +=
        x227 * (u[3] * x194 + u[3] * x196 + u[4] * x148 - u[4] * x194 - u[4] * x196 - u[5] * x148 -
                u[5] * x197 - x102 * x143 + x107 + x109 * x24 - x110 * x200 + x111 * x46 - x112 +
                x125 + 96 * x130 + x14 * x21 + 8 * x140 + x141 * x22 - x152 - 4 * x153 + x164 +
                3 * x166 + x172 - 5 * x173 + x176 + 8 * x177 - 4 * x178 - x180 + 5 * x181 - x184 +
                32 * x187 - x190 + x191 * x33 - x191 * x38 - x192 - x193 + x195 * x54 - x195 * x57 -
                x195 * x73 - x198 * x46 + x198 * x88 + x199 * x23 + x199 * x5 - x199 * x57 - x201 +
                x202 + x204 + x205 + 4 * x206 + x207 * x79 + x208 * x23 + x209 + x211 + x212 +
                x213 * x38 + x220 + x226 + 16 * x30 - x31 * x38 + 4 * x42 + x45 + x59 - 24 * x62 +
                4 * x70 - 8 * x72 + x83 - x85 - x86 - x95 - x98 - 40 * x99);
    element_vector[4] +=
        x227 *
        (u[0] * x80 - u[0] * x82 + u[0] * (-x46 + x69) + 40 * u[2] * x5 + u[3] * x197 +
         u[3] * x234 - u[3] * x237 - u[4] * x197 - u[4] * x234 + u[4] * x235 + u[4] * x238 +
         u[5] * x196 - u[5] * x235 - u[5] * x238 - x100 * x6 + x105 * x39 - x109 * x79 +
         x120 * x73 + 5 * x13 + x131 + 32 * x138 + x139 + x14 * x229 + 16 * x15 * x19 - 8 * x153 -
         40 * x154 + x158 + x159 - x160 + x17 * x228 - 4 * x173 + 8 * x181 - x183 + x188 + x190 +
         x192 + x193 - x195 * x22 + x195 * x23 + x199 * x88 + x201 - x202 - x204 - x205 - x209 -
         x211 - x212 - x213 * x39 + x213 * x5 + x215 + x216 + x217 * x33 - x217 * x35 - x217 * x38 +
         x217 * x43 + x217 * x54 + x218 + x219 + x226 + x229 * x8 - x230 - x231 - x232 -
         x233 * x39 + x233 * x5 - x236 * x46 - x240 + x241 + x242 + x243 + x244 + x245 + x246 +
         x249 + 3 * x28 - x31 * x5 + 8 * x70 - 4 * x72);
    element_vector[5] +=
        x227 * (u[3] * x179 + u[3] * x238 - u[4] * x179 + u[4] * x237 + u[4] * x250 -
                4 * u[4] * x54 - u[5] * x237 - x101 + x105 * x43 + x108 + 24 * x119 - x120 * x26 +
                x127 + 3 * x129 + 32 * x130 - 8 * x138 + x14 * x163 - x141 * x23 - 5 * x153 -
                4 * x154 + x162 + x169 + x171 - 8 * x173 + x175 + 4 * x177 - 8 * x178 + 4 * x181 -
                x182 - x185 + 96 * x187 + x189 - x191 * x35 + x191 * x43 + x195 * x6 + x195 * x79 -
                x199 * x46 - x203 * x22 + x203 * x6 + x207 * x54 - x207 * x73 - x213 * x43 -
                x22 * x236 + x220 + x230 + x231 + x232 + x236 * x88 + x239 * x54 + x240 - x241 -
                x242 - x243 - x244 - x245 - x246 - x248 * x73 + x249 + x25 * x55 + 16 * x29 +
                40 * x42 + x48 - x51 * x96 + x52 + 5 * x70 - 5 * x72 + x81 - x84 - x92 - 4 * x99);
    element_vector[6] +=
        x128 *
        (u[6] * x298 + 72 * u[6] * (-x38 + x43) - u[7] * x276 + x102 * x269 - x102 * x270 -
         x2 * x297 + x22 * x270 - x23 * x270 + x24 * x325 - x25 * x279 + x252 + x253 * x9 +
         x254 * x90 + 18 * x255 + 32 * x258 - x26 * x271 - x26 * x77 + 32 * x260 + x261 * x263 +
         78 * x264 + 80 * x265 + 80 * x266 - 96 * x267 + x268 * x38 - x268 * x96 - x269 * x43 +
         x270 * x96 - x272 - x274 - x275 * x6 - x275 * x79 + x277 * x57 + x277 * x73 - x278 +
         x279 * x88 - x280 * x88 - x282 - x284 - x285 * x38 - x287 - x289 - x290 - x292 - x294 -
         11 * x296 - x299 - x300 * x69 - x302 - x304 + x305 + x307 + x308 * x89 + 9 * x309 + x311 +
         11 * x312 + x313 + x314 + x315 + x316 * x73 + x317 + x318 + 24 * x319 + x320 + x321 +
         x322 * x96 + x324 + x326 * x88 + x337 + x342 + x344 + x346 - x71 * x73);
    element_vector[7] +=
        x128 *
        (-u[10] * x352 + u[6] * x276 + u[7] * x135 - u[8] * x106 + u[9] * x352 + x102 * x376 -
         x22 * x356 - x23 * x277 + x24 * x271 + x25 * x329 - x25 * x332 - x25 * x71 - x254 * x74 -
         x254 * x89 + x256 * x348 + 48 * x258 - x26 * x277 + x26 * x71 +
         x269 * (-x102 + x33 + x39) + x271 * x73 + x279 * x46 - x280 * x46 + x282 + x287 +
         18 * x296 - x308 * x90 + 18 * x309 - x311 - 18 * x312 - x318 + x322 * x6 - x326 * x46 +
         x329 * x69 - x332 * x69 + x337 + x349 + x350 * x7 + 78 * x351 - x353 * x38 + x353 * x5 -
         72 * x354 - x355 - x357 * x39 - x358 - x359 - 18 * x360 - x361 - x362 - x363 - x364 +
         x365 * (u[10] - u[11]) + x366 + x367 * x69 + x368 + x370 + x371 * x46 + x372 + x373 +
         x374 + x375 + x376 * x57 + x381 + x382 + x384 + x73 * x77);
    element_vector[8] +=
        -x128 *
        (-u[10] * x389 + u[11] * x389 + u[8] * x165 + 78 * u[8] * x388 - x150 * x254 + x174 * x254 -
         x174 * x308 - x23 * x356 - x24 * x275 - x25 * x77 - x252 + x26 * x325 + x26 * x394 -
         48 * x260 + x268 * (x35 + x5 - x96) + x279 * x69 + x289 + x292 + x294 - 9 * x296 + x299 -
         16 * x303 - 24 * x306 + 11 * x309 + 9 * x312 - x313 - x315 + x325 * x79 + x329 * x46 +
         x329 * x88 + x330 + x333 - x334 - x335 * x88 - x336 + x342 - 32 * x343 + x347 * x75 -
         x349 - 48 * x354 + x356 * x96 - x357 * x5 - 11 * x360 + x363 + x364 - x366 + x367 * x46 -
         x370 + x371 * x69 + x381 - x386 + x39 * x390 - x390 * x43 - 96 * x391 - 72 * x392 - x393 -
         x394 * x73 + x395 * (u[10] - u[9]) + x396 + x397 + 32 * x398 + x399 * x6 + x399 * x96 +
         x401 + x402);
    element_vector[9] +=
        x227 *
        (-u[10] * x411 - u[11] * x238 + u[11] * x365 + u[11] * x411 + 40 * u[7] * x102 +
         u[7] * x196 - u[7] * x80 + 24 * u[8] * x22 - 3 * u[8] * x388 + u[8] * x395 - x102 * x404 -
         x195 * x254 - x198 * x254 + x198 * x308 + x203 * x295 - x208 * x408 + x22 * x326 -
         x22 * x403 + x22 * x404 - x24 * x277 + x25 * x418 + x256 * x350 + x26 * x400 - x26 * x410 +
         16 * x265 - 16 * x267 + x274 + x284 - x301 * x88 + x304 - x305 - 12 * x306 - x314 -
         8 * x319 - x321 + x339 + x344 + x355 + x359 + x362 - x373 - x375 + x382 + 96 * x383 +
         32 * x385 + x39 * x416 + x39 * x419 + 8 * x391 - 8 * x392 - x400 * x73 + x401 - x405 -
         x406 - x407 - 5 * x409 + x410 * x73 - x412 - x414 - x415 - x416 * x43 - x416 * x6 -
         x416 * x96 + x417 + x419 * x46 + x420 * x69 + x421 + x422 + x423 + x430 + x432);
    element_vector[10] +=
        x227 *
        (-u[11] * x237 + u[6] * (x387 + x73) + u[7] * x179 - u[7] * x197 - u[8] * x148 +
         u[8] * x238 + u[9] * x196 - x102 * x439 + x199 * x308 - x203 * x408 + x208 * x295 +
         x22 * x439 - x23 * x439 - x236 * x254 + x25 * x420 + 5 * x255 + x256 * x433 +
         16 * x257 * x261 + x259 * x66 + 3 * x264 - 8 * x267 + 8 * x303 - 8 * x306 - 12 * x319 +
         x322 * x5 + x326 * x6 - x332 * x46 + x335 * x69 - 32 * x354 + 24 * x369 - x374 + x377 +
         x378 - x379 + x380 + x384 + x386 + x393 + x397 + 40 * x398 + x4 * x433 + x403 * x57 -
         x403 * x6 + x405 + x406 + x407 - 4 * x409 + x412 + x414 + x415 - x417 - x418 * x46 +
         x418 * x69 - x420 * x88 - x421 - x422 - x423 + x425 + x426 - 24 * x427 + x428 + x429 +
         x432 - x434 * x57 - x435 * x6 - x436 - x437 - x438 + x439 * x46 + x439 * x96 - x440 -
         x441 - x442 + x443 + x444 * x88 + 12 * x445 + x446 + x447 + x448 + x450);
    element_vector[11] +=
        x227 *
        (u[10] * x451 + 5 * u[6] * x46 - 24 * u[7] * x23 - u[8] * x237 + u[8] * x82 - u[9] * x451 +
         x102 * x413 - x199 * x254 - x217 * x254 + x23 * x403 - x23 * x404 - x23 * x434 +
         x236 * x308 - x24 * x316 + x25 * x301 + x25 * x452 + x256 * x263 + x26 * x275 + 16 * x266 +
         x278 + x290 + 5 * x296 + x302 + 12 * x303 - x307 + 5 * x309 - x310 - 5 * x312 +
         x316 * x79 - x317 - 4 * x319 - x320 + x332 * x88 + x338 - x341 + x346 + 3 * x351 +
         8 * x354 + x358 - 5 * x360 + x361 - x368 - x372 + x38 * x413 + x38 * x452 + 32 * x383 +
         96 * x385 - x396 + 4 * x398 + x402 + x404 * x96 - 8 * x409 - x413 * x5 - x418 * x88 -
         x419 * x5 - x420 * x46 + x430 - x435 * x96 + x436 + x437 + x438 + x440 + x441 + x442 -
         x443 - x444 * x46 + 8 * x445 - x446 - x447 - x448 + x450);
}

void tri3_tri6_divergence(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const elems,
                          geom_t **const points,
                          const real_t dt,
                          const real_t rho,
                          const real_t nu,
                          real_t **const SFEM_RESTRICT vel,
                          real_t *const SFEM_RESTRICT f) {
    SFEM_UNUSED(nnodes);
    double tick = MPI_Wtime();

    static const int n_vars = 2;
    static const int element_nnodes = 6;

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[6];
            real_t element_vector[3];
            real_t element_vel[6 * 2];

#pragma unroll(6)
            for (int v = 0; v < element_nnodes; ++v) {
                ev[v] = elems[v][i];
            }

            for (int enode = 0; enode < element_nnodes; ++enode) {
                idx_t dof = ev[enode];

                for (int b = 0; b < n_vars; ++b) {
                    element_vel[b * element_nnodes + enode] = vel[b][dof];
                }
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];

            const real_t x0 = points[0][i0];
            const real_t x1 = points[0][i1];
            const real_t x2 = points[0][i2];

            const real_t y0 = points[1][i0];
            const real_t y1 = points[1][i1];
            const real_t y2 = points[1][i2];

            tri3_tri6_divergence_rhs_kernel(
                // X coords
                points[0][i0],
                points[0][i1],
                points[0][i2],
                // Y coords
                points[1][i0],
                points[1][i1],
                points[1][i2],
                dt,
                rho,
                //  buffers
                element_vel,
                element_vector);

            for (int edof_i = 0; edof_i < 3; ++edof_i) {
                const idx_t dof_i = ev[edof_i];
#pragma omp atomic update
                f[dof_i] += element_vector[edof_i];
            }
        }
    }

    double tock = MPI_Wtime();
    // printf("tri6_naviers_stokes.c: tri6_explict_momentum_tentative\t%g seconds\n", tock - tick);
}

void tri6_tri3_correction(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const elems,
                          geom_t **const points,
                          const real_t dt,
                          const real_t rho,
                          real_t *const SFEM_RESTRICT p,
                          real_t **const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);
    double tick = MPI_Wtime();

    static const int n_vars = 2;
    static const int element_nnodes = 6;

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[6];
            real_t element_vector[6 * 2];
            real_t element_pressure[3];

#pragma unroll(6)
            for (int v = 0; v < element_nnodes; ++v) {
                ev[v] = elems[v][i];
            }

#pragma unroll(3)
            for (int enode = 0; enode < 3; ++enode) {
                idx_t dof = ev[enode];
                element_pressure[enode] = p[dof];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];

            const real_t x0 = points[0][i0];
            const real_t x1 = points[0][i1];
            const real_t x2 = points[0][i2];

            const real_t y0 = points[1][i0];
            const real_t y1 = points[1][i1];
            const real_t y2 = points[1][i2];

            tri6_tri3_rhs_correction_kernel(
                // X coords
                points[0][i0],
                points[0][i1],
                points[0][i2],
                // Y coords
                points[1][i0],
                points[1][i1],
                points[1][i2],
                dt,
                rho,
                //  buffers
                element_pressure,
                element_vector);

            for (int b = 0; b < n_vars; ++b) {
#pragma unroll(6)
                for (int edof_i = 0; edof_i < element_nnodes; ++edof_i) {
                    const idx_t dof_i = ev[edof_i];
#pragma omp atomic update
                    values[b][dof_i] += element_vector[b * element_nnodes + edof_i];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    // printf("tri6_naviers_stokes.c: tri6_explict_momentum_tentative\t%g seconds\n", tock - tick);
}

static SFEM_INLINE int linear_search(const idx_t target, const idx_t *const arr, const int size) {
    int i;
    for (i = 0; i < size - 4; i += 4) {
        if (arr[i] == target) return i;
        if (arr[i + 1] == target) return i + 1;
        if (arr[i + 2] == target) return i + 2;
        if (arr[i + 3] == target) return i + 3;
    }
    for (; i < size; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

static SFEM_INLINE int find_col(const idx_t key, const idx_t *const row, const int lenrow) {
    if (lenrow <= 32) {
        return linear_search(key, row, lenrow);

        // Using sentinel (potentially dangerous if matrix is buggy and column does not exist)
        // while (key > row[++k]) {
        //     // Hi
        // }
        // assert(k < lenrow);
        // assert(key == row[k]);
    } else {
        // Use this for larger number of dofs per row
        return find_idx_binary_search(key, row, lenrow);
    }
}

static SFEM_INLINE void find_cols6(const idx_t *targets,
                                   const idx_t *const row,
                                   const int lenrow,
                                   idx_t *ks) {
    if (lenrow > 32) {
        for (int d = 0; d < 6; ++d) {
            ks[d] = find_col(targets[d], row, lenrow);
        }
    } else {
#pragma unroll(6)
        for (int d = 0; d < 6; ++d) {
            ks[d] = 0;
        }

        for (int i = 0; i < lenrow; ++i) {
#pragma unroll(6)
            for (int d = 0; d < 6; ++d) {
                ks[d] += row[i] < targets[d];
            }
        }
    }
}

void tri6_momentum_lhs_scalar_crs(const ptrdiff_t nelements,
                                  const ptrdiff_t nnodes,
                                  idx_t **const elems,
                                  geom_t **const points,
                                  const real_t dt,
                                  const real_t nu,
                                  const count_t *const SFEM_RESTRICT rowptr,
                                  const idx_t *const SFEM_RESTRICT colidx,
                                  real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[6];
            idx_t ks[6];
            real_t element_matrix[6 * 6];

#pragma unroll(6)
            for (int v = 0; v < 6; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];

            tri6_momentum_lhs_scalar_kernel(
                // X-coordinates
                points[0][i0],
                points[0][i1],
                points[0][i2],
                // Y-coordinates
                points[1][i0],
                points[1][i1],
                points[1][i2],
                dt,
                nu,
                element_matrix);

            for (int edof_i = 0; edof_i < 6; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];
                const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

                const idx_t *row = &colidx[rowptr[dof_i]];

                find_cols6(ev, row, lenrow, ks);

                real_t *rowvalues = &values[rowptr[dof_i]];
                const real_t *element_row = &element_matrix[edof_i * 6];

#pragma unroll(6)
                for (int edof_j = 0; edof_j < 6; ++edof_j) {
#pragma omp atomic update
                    rowvalues[ks[edof_j]] += element_row[edof_j];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tri3_laplacian.c: tri3_laplacian_crs\t%g seconds\n", tock - tick);
}

void tri6_explict_momentum_tentative(const ptrdiff_t nelements,
                                     const ptrdiff_t nnodes,
                                     idx_t **const elems,
                                     geom_t **const points,
                                     const real_t dt,
                                     const real_t nu,
                                     const real_t convonoff,
                                     real_t **const SFEM_RESTRICT vel,
                                     real_t **const SFEM_RESTRICT f) {
    SFEM_UNUSED(nnodes);
    double tick = MPI_Wtime();

    static const int n_vars = 2;
    static const int element_nnodes = 6;

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[6];
            real_t element_vector[6 * 2];
            real_t element_vel[6 * 2];

#pragma unroll(6)
            for (int v = 0; v < element_nnodes; ++v) {
                ev[v] = elems[v][i];
            }

            for (int enode = 0; enode < element_nnodes; ++enode) {
                idx_t dof = ev[enode];

                for (int b = 0; b < n_vars; ++b) {
                    element_vel[b * element_nnodes + enode] = vel[b][dof];
                }
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];

            const real_t x0 = points[0][i0];
            const real_t x1 = points[0][i1];
            const real_t x2 = points[0][i2];

            const real_t y0 = points[1][i0];
            const real_t y1 = points[1][i1];
            const real_t y2 = points[1][i2];

            // tri6_explict_momentum_rhs_kernel(
            //     // X coords
            //     points[0][i0],
            //     points[0][i1],
            //     points[0][i2],
            //     // Y coords
            //     points[1][i0],
            //     points[1][i1],
            //     points[1][i2],
            //     dt,
            //     nu,
            //     convonoff,
            //     //  buffers
            //     element_vel,
            //     element_vector);

            memset(element_vector, 0, 6 * 2 * sizeof(real_t));

            // tri6_add_momentum_rhs_kernel(points[0][i0],
            //                              points[0][i1],
            //                              points[0][i2],
            //                              // Y coords
            //                              points[1][i0],
            //                              points[1][i1],
            //                              points[1][i2],
            //                              //  buffers
            //                              element_vel,
            //                              element_vector);

            tri6_add_diffusion_rhs_kernel(points[0][i0],
                                          points[0][i1],
                                          points[0][i2],
                                          // Y coords
                                          points[1][i0],
                                          points[1][i1],
                                          points[1][i2],
                                          dt,
                                          nu,
                                          //  buffers
                                          element_vel,
                                          element_vector);

            if (convonoff != 0) {
                tri6_add_convection_rhs_kernel(points[0][i0],
                                               points[0][i1],
                                               points[0][i2],
                                               // Y coords
                                               points[1][i0],
                                               points[1][i1],
                                               points[1][i2],
                                               dt,
                                               nu,
                                               //  buffers
                                               element_vel,
                                               element_vector);
            }

            for (int d1 = 0; d1 < n_vars; d1++) {
                for (int edof_i = 0; edof_i < element_nnodes; ++edof_i) {
                    const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
                    f[d1][dof_i] += element_vector[d1 * element_nnodes + edof_i];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    // printf("tri6_naviers_stokes.c: tri6_explict_momentum_tentative\t%g seconds\n", tock - tick);
}