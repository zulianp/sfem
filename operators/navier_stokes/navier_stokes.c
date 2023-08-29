#include "navier_stokes.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"

void navier_stokes_assemble_value_aos(const enum ElemType element_type,
                                      const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const SFEM_RESTRICT elems,
                                      geom_t **const SFEM_RESTRICT xyz,
                                      const real_t nu,
                                      const real_t rho,
                                      const real_t *const SFEM_RESTRICT u,
                                      real_t *const SFEM_RESTRICT value) {
    // TODO
}

void navier_stokes_assemble_gradient_aos(const enum ElemType element_type,
                                         const ptrdiff_t nelements,
                                         const ptrdiff_t nnodes,
                                         idx_t **const SFEM_RESTRICT elems,
                                         geom_t **const SFEM_RESTRICT xyz,
                                         const real_t nu,
                                         const real_t rho,
                                         const real_t *const SFEM_RESTRICT u,
                                         real_t *const SFEM_RESTRICT values) {
    // TODO
}

void navier_stokes_assemble_hessian_aos(const enum ElemType element_type,
                                        const ptrdiff_t nelements,
                                        const ptrdiff_t nnodes,
                                        idx_t **const SFEM_RESTRICT elems,
                                        geom_t **const SFEM_RESTRICT xyz,
                                        const real_t nu,
                                        const real_t rho,
                                        const count_t *const SFEM_RESTRICT rowptr,
                                        const idx_t *const SFEM_RESTRICT colidx,
                                        real_t *const SFEM_RESTRICT values) {
    // TODO
}

void navier_stokes_apply_aos(const enum ElemType element_type,
                             const ptrdiff_t nelements,
                             const ptrdiff_t nnodes,
                             idx_t **const SFEM_RESTRICT elems,
                             geom_t **const SFEM_RESTRICT xyz,
                             const real_t nu,
                             const real_t rho,
                             const real_t *const SFEM_RESTRICT u,
                             real_t *const SFEM_RESTRICT values) {
    // TODO
}

// Chorin's projection method
// 1) temptative momentum step
// - Implicit Euler
//   `<u, v> + dt * nu * <grad(u), grad(v)> =  <u_old, v> - dt * <(u_old . div) * u_old, v>`
// - Explicit Euler
//    `<u, v> = <u_old, v> - dt * ( <(u_old . div) * u_old, v> + nu * <grad(u), grad(v)> )`
// 2) Potential eqaution
//   `<grad(p), grad(q)> = - 1/dt * <div(u), q>`
// 3) Projection/Correction
//   `<u_new, v> = <u, v> - dt * <grad(p), v>`

// Taylor hood Triangle P2/P1
// 1) Temptative momentum step

// static SFEM_INLINE void tri6_momentum_lhs_kernel(const real_t px0,
//                                                  const real_t px1,
//                                                  const real_t px2,
//                                                  const real_t py0,
//                                                  const real_t py1,
//                                                  const real_t py2,
//                                                  const real_t dt,
//                                                  const real_t nu,
//                                                  real_t *const SFEM_RESTRICT element_matrix) {}

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
    const real_t x0 = (1.0 / 6.0) * rho / dt;
    const real_t x1 = px0 - px2;
    const real_t x2 = u[11] * x1;
    const real_t x3 = py0 - py2;
    const real_t x4 = u[4] * x3;
    const real_t x5 = px0 - px1;
    const real_t x6 = u[10] * x5;
    const real_t x7 = u[10] * x1;
    const real_t x8 = py0 - py1;
    const real_t x9 = u[3] * x8;
    const real_t x10 = u[4] * x8;
    const real_t x11 = u[5] * x3;
    const real_t x12 = u[9] * x5;
    element_vector[0] =
        x0 * (-2 * px0 * u[11] + 2 * px0 * u[9] + px1 * u[10] + px1 * u[11] - px1 * u[6] -
              px1 * u[9] - px2 * u[10] + px2 * u[11] + px2 * u[6] - px2 * u[9] - 2 * py0 * u[3] +
              2 * py0 * u[5] + py1 * u[0] + py1 * u[3] - py1 * u[4] - py1 * u[5] - py2 * u[0] +
              py2 * u[3] + py2 * u[4] - py2 * u[5]);
    element_vector[1] = -x0 * (u[1] * x3 - u[3] * x3 - u[7] * x1 + u[9] * x1 - 2 * x10 - x11 -
                               2 * x12 + x2 + x4 + 2 * x6 - x7 + 2 * x9);
    element_vector[2] = -x0 * (-u[11] * x5 - u[2] * x8 + u[5] * x8 + u[8] * x5 - x10 - 2 * x11 -
                               x12 + 2 * x2 + 2 * x4 + x6 - 2 * x7 + x9);
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
                                                        const real_t *const SFEM_RESTRICT u,
                                                        const real_t *const SFEM_RESTRICT p,
                                                        real_t *const SFEM_RESTRICT
                                                            element_vector) {
    const real_t x0 = py0 - py1;
    const real_t x1 = py0 - py2;
    const real_t x2 = (1.0/6.0)*dt/rho;
    const real_t x3 = x2*(p[0]*x0 - p[0]*x1 + p[1]*x1 - p[2]*x0);
    const real_t x4 = px0 - px1;
    const real_t x5 = px0 - px2;
    const real_t x6 = x2*(-p[0]*x4 + p[0]*x5 - p[1]*x5 + p[2]*x4);
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
}

static SFEM_INLINE void tri6_explict_momentum_rhs_kernel(const real_t px0,
                                                         const real_t px1,
                                                         const real_t px2,
                                                         const real_t py0,
                                                         const real_t py1,
                                                         const real_t py2,
                                                         const real_t dt,
                                                         const real_t nu,
                                                         real_t *const SFEM_RESTRICT u,
                                                         real_t *const SFEM_RESTRICT
                                                             element_vector) {
    const real_t x0 = 6 * u[0];
    const real_t x1 = 4 * u[4];
    const real_t x2 = px0 - px1;
    const real_t x3 = py0 - py2;
    const real_t x4 = px0 - px2;
    const real_t x5 = py0 - py1;
    const real_t x6 = x2 * x3 - x4 * x5;
    const real_t x7 = pow(x6, 2);
    const real_t x8 = 7 * x7;
    const real_t x9 = pow(x4, 2);
    const real_t x10 = u[1] * x9;
    const real_t x11 = pow(x3, 2);
    const real_t x12 = u[1] * x11;
    const real_t x13 = pow(x2, 2);
    const real_t x14 = u[0] * x13;
    const real_t x15 = u[0] * x9;
    const real_t x16 = pow(x5, 2);
    const real_t x17 = u[0] * x16;
    const real_t x18 = u[0] * x11;
    const real_t x19 = u[1] * x4;
    const real_t x20 = x19 * x2;
    const real_t x21 = u[1] * x3;
    const real_t x22 = x21 * x5;
    const real_t x23 = x2 * x4;
    const real_t x24 = x3 * x5;
    const real_t x25 = 4 * u[3];
    const real_t x26 = u[3] * x5;
    const real_t x27 = x26 * x3;
    const real_t x28 = -x11 * x25 + x23 * x25 - x25 * x9 + 4 * x27;
    const real_t x29 = 4 * u[5];
    const real_t x30 = u[5] * x3;
    const real_t x31 = 4 * x30;
    const real_t x32 = -x13 * x29 - x16 * x29 + x23 * x29 + x31 * x5;
    const real_t x33 = u[2] * x13;
    const real_t x34 = u[2] * x16;
    const real_t x35 = u[2] * x2;
    const real_t x36 = x35 * x4;
    const real_t x37 = u[2] * x5;
    const real_t x38 = x3 * x37;
    const real_t x39 = x33 + x34 - x36 - x38;
    const real_t x40 = 420 * dt * nu;
    const real_t x41 = -x3;
    const real_t x42 = pow(u[4], 2);
    const real_t x43 = 48 * x42;
    const real_t x44 = x41 * x43;
    const real_t x45 = u[10] * x4;
    const real_t x46 = 48 * u[5];
    const real_t x47 = -x45 * x46;
    const real_t x48 = 48 * u[4];
    const real_t x49 = x45 * x48;
    const real_t x50 = pow(u[1], 2);
    const real_t x51 = 9 * x41;
    const real_t x52 = pow(u[2], 2);
    const real_t x53 = 9 * x5;
    const real_t x54 = u[10] * x2;
    const real_t x55 = u[4] * x5;
    const real_t x56 = -u[4] * x3 - x45 + x54 + x55;
    const real_t x57 = 12 * u[0];
    const real_t x58 = 18 * u[6];
    const real_t x59 = pow(u[5], 2);
    const real_t x60 = 32 * x3;
    const real_t x61 = u[0] * x5;
    const real_t x62 = u[6] * x4;
    const real_t x63 = x61 + x62;
    const real_t x64 = u[0] * x41;
    const real_t x65 = u[6] * x2;
    const real_t x66 = x64 - x65;
    const real_t x67 = pow(u[3], 2);
    const real_t x68 = x3 * x67;
    const real_t x69 = -x5;
    const real_t x70 = x59 * x69;
    const real_t x71 = 96 * u[3];
    const real_t x72 = 80 * u[5];
    const real_t x73 = u[9] * x4;
    const real_t x74 = 80 * u[3];
    const real_t x75 = u[11] * x4;
    const real_t x76 = 64 * u[3];
    const real_t x77 = x75 * x76;
    const real_t x78 = u[11] * x2;
    const real_t x79 = 48 * u[0];
    const real_t x80 = u[9] * x2;
    const real_t x81 = 48 * x54;
    const real_t x82 = u[4] * x81;
    const real_t x83 = 32 * u[3];
    const real_t x84 = u[11] * x35;
    const real_t x85 = 32 * u[5];
    const real_t x86 = x75 * x85;
    const real_t x87 = u[3] * x3;
    const real_t x88 = 32 * u[1];
    const real_t x89 = 24 * x65;
    const real_t x90 = 24 * x62;
    const real_t x91 = 20 * u[1];
    const real_t x92 = 20 * u[2];
    const real_t x93 = u[7] * x2;
    const real_t x94 = 20 * u[5];
    const real_t x95 = 16 * u[4];
    const real_t x96 = 16 * u[1];
    const real_t x97 = x26 * x96;
    const real_t x98 = u[4] * x3;
    const real_t x99 = x96 * x98;
    const real_t x100 = 16 * u[2];
    const real_t x101 = u[9] * x35;
    const real_t x102 = 16 * x55;
    const real_t x103 = u[5] * x102;
    const real_t x104 = 16 * u[5];
    const real_t x105 = u[8] * x4;
    const real_t x106 = x104 * x105;
    const real_t x107 = x104 * x73;
    const real_t x108 = u[1] * x37;
    const real_t x109 = 11 * u[8];
    const real_t x110 = 9 * u[1];
    const real_t x111 = u[0] * x3;
    const real_t x112 = u[1] * x111;
    const real_t x113 = u[7] * x4;
    const real_t x114 = 9 * u[0];
    const real_t x115 = u[0] * x4;
    const real_t x116 = 9 * u[8];
    const real_t x117 = x1 * x113;
    const real_t x118 = x113 * x29;
    const real_t x119 = 9 * u[2];
    const real_t x120 = u[8] * x2;
    const real_t x121 = 9 * x19;
    const real_t x122 = u[7] * x121;
    const real_t x123 = 11 * u[2];
    const real_t x124 = 11 * u[7];
    const real_t x125 = u[11] * x19;
    const real_t x126 = 16 * x125;
    const real_t x127 = 16 * u[3];
    const real_t x128 = x55 * x96;
    const real_t x129 = x105 * x95;
    const real_t x130 = x73 * x95;
    const real_t x131 = x45 * x91;
    const real_t x132 = 32 * u[4];
    const real_t x133 = x132 * x75;
    const real_t x134 = u[9] * x19;
    const real_t x135 = u[3] * x81;
    const real_t x136 = 96 * u[5];
    const real_t x137 = 20 * u[3];
    const real_t x138 = x105 * x137 + x43 * x5;
    const real_t x139 = u[2] * x61;
    const real_t x140 = u[5] * x5;
    const real_t x141 = -u[3] * x102 + 9 * x139 + x140 * x91;
    const real_t x142 = 64 * u[5];
    const real_t x143 = x127 * x98 + x142 * x80;
    const real_t x144 = x67 * x69;
    const real_t x145 =
        x1 * x120 + x100 * x55 - x120 * x25 - x132 * x80 + 32 * x144 - x54 * x92 + x80 * x83;
    const real_t x146 = 1.0 / x6;
    const real_t x147 = (1.0 / 2520.0) * x146;
    const real_t x148 = -x2;
    const real_t x149 = -x4;
    const real_t x150 = x148 * x41 - x149 * x69;
    const real_t x151 = x150 * x8;
    const real_t x152 = x36 + x38;
    const real_t x153 = x115 * x2;
    const real_t x154 = x3 * x61;
    const real_t x155 = -x153 - x154;
    const real_t x156 = -x1 * x23 - x1 * x24 + x155;
    const real_t x157 = x150 * x40;
    const real_t x158 = u[5] * x41;
    const real_t x159 = -x158 + x75;
    const real_t x160 = 12 * u[1];
    const real_t x161 = 48 * x59;
    const real_t x162 = u[1] * x41;
    const real_t x163 = x113 + x162;
    const real_t x164 = u[1] * x163;
    const real_t x165 = u[4] * x41;
    const real_t x166 = 120 * u[1];
    const real_t x167 = u[7] * x148;
    const real_t x168 = 120 * x167;
    const real_t x169 = u[10] * x148;
    const real_t x170 = 48 * u[3];
    const real_t x171 = x48 * x75;
    const real_t x172 = 48 * u[2];
    const real_t x173 = u[2] * x148;
    const real_t x174 = u[10] * x173;
    const real_t x175 = 24 * x113;
    const real_t x176 = u[11] * x148;
    const real_t x177 = x176 * x92;
    const real_t x178 = u[6] * x148;
    const real_t x179 = u[7] * x173;
    const real_t x180 = u[9] * x148;
    const real_t x181 = u[8] * x148;
    const real_t x182 = 11 * u[0];
    const real_t x183 = 4 * x62;
    const real_t x184 = u[6] * x173;
    const real_t x185 = u[1] * x61;
    const real_t x186 = 18 * u[0];
    const real_t x187 = 20 * u[0];
    const real_t x188 = u[4] * x64;
    const real_t x189 = u[3] * x41;
    const real_t x190 = 32 * u[0];
    const real_t x191 = u[1] * x45;
    const real_t x192 = x46 * x75;
    const real_t x193 = -x132 * x45 + x187 * x75 + x42 * x60 + x45 * x85;
    const real_t x194 = x170 * x180;
    const real_t x195 = x180 * x48;
    const real_t x196 = 48 * x144 - x194 + x195 - x45 * x76;
    const real_t x197 = pow(u[0], 2);
    const real_t x198 = 16 * u[0];
    const real_t x199 = x107 + x114 * x178 + x114 * x62 + x127 * x176 - x130 + x169 * x198 -
                        x176 * x95 + x197 * x51 + x197 * x53 + x198 * x45;
    const real_t x200 = dt * x7;
    const real_t x201 = 1.0 / x150;
    const real_t x202 = x147 * x201;
    const real_t x203 = x20 + x22;
    const real_t x204 = x14 + x17;
    const real_t x205 = x3 * x50;
    const real_t x206 = -u[9] * x148 + x26;
    const real_t x207 = 12 * u[2];
    const real_t x208 = 32 * x42;
    const real_t x209 = x5 * x67;
    const real_t x210 = x181 + x37;
    const real_t x211 = 78 * x210;
    const real_t x212 = 120 * u[2];
    const real_t x213 = 120 * x105;
    const real_t x214 = 24 * x181;
    const real_t x215 = 20 * x134;
    const real_t x216 = 18 * u[2];
    const real_t x217 = 18 * u[8];
    const real_t x218 = 16 * x148;
    const real_t x219 = u[7] * x218;
    const real_t x220 = 4 * x178;
    const real_t x221 = u[6] * x19;
    const real_t x222 = x198 * x26;
    const real_t x223 = x187 * x73;
    const real_t x224 = u[2] * x55;
    const real_t x225 = x113 * x137 + x127 * x45 - x127 * x75 - x137 * x62;
    const real_t x226 = -x104 * x26 + x44 + x47 + x49;
    const real_t x227 = -x142 * x169 + x161 * x3 + x171 - x192;
    const real_t x228 = 8 * u[3];
    const real_t x229 = 2 * u[3];
    const real_t x230 = 2 * u[4];
    const real_t x231 = x23 * x230;
    const real_t x232 = 2 * x5;
    const real_t x233 = x232 * x98;
    const real_t x234 = -2 * u[3] * x2 * x4 - 2 * u[3] * x3 * x5 - 2 * u[4] * x13 - 2 * u[4] * x16 -
                        2 * u[5] * x2 * x4 - 2 * u[5] * x3 * x5 + x13 * x229 + x16 * x229 + x203 +
                        x231 + x233;
    const real_t x235 = 12 * x197;
    const real_t x236 = 40 * u[0];
    const real_t x237 = x78 * x83;
    const real_t x238 = x132 * x73;
    const real_t x239 = x57 * x65;
    const real_t x240 = x160 * x55;
    const real_t x241 = 12 * u[7];
    const real_t x242 = x19 * x241;
    const real_t x243 = x2 * x241;
    const real_t x244 = 12 * x120;
    const real_t x245 = 8 * u[0];
    const real_t x246 = 8 * u[4];
    const real_t x247 = 8 * u[5];
    const real_t x248 = 5 * u[8];
    const real_t x249 = 5 * u[2];
    const real_t x250 = 4 * u[0];
    const real_t x251 = x250 * x54;
    const real_t x252 = 4 * u[11];
    const real_t x253 = x19 * x252;
    const real_t x254 = u[2] * x3;
    const real_t x255 = 4 * x35;
    const real_t x256 = x250 * x45;
    const real_t x257 = 4 * u[2];
    const real_t x258 = 5 * u[0];
    const real_t x259 = 8 * u[1];
    const real_t x260 = x57 * x62;
    const real_t x261 = x160 * x26;
    const real_t x262 = x132 * x78;
    const real_t x263 = x73 * x85;
    const real_t x264 = 96 * u[4];
    const real_t x265 = u[1] * x5;
    const real_t x266 = -x170 * x55 + x207 * x26 - x26 * x85 + x265 * x29 + x29 * x37 + x29 * x61 +
                        x55 * x57 - x55 * x85 + 16 * x70;
    const real_t x267 = dt * x6;
    const real_t x268 = (1.0 / 630.0) * x146;
    const real_t x269 = 2 * u[5];
    const real_t x270 = x269 * x9;
    const real_t x271 = x11 * x269;
    const real_t x272 = 5 * u[6];
    const real_t x273 = 96 * x42;
    const real_t x274 = x127 * x73;
    const real_t x275 = x148 * x241;
    const real_t x276 = 12 * u[4];
    const real_t x277 = 8 * u[2];
    const real_t x278 = u[8] * x19;
    const real_t x279 = u[8] * x115;
    const real_t x280 = u[2] * x41;
    const real_t x281 = x105 * x25;
    const real_t x282 = 12 * x178;
    const real_t x283 = 12 * u[5];
    const real_t x284 = 24 * u[0];
    const real_t x285 = 12 * u[8];
    const real_t x286 = x285 * x4;
    const real_t x287 = u[4] * x286 - u[5] * x286 + x238 - x263 + 12 * x5 * x52;
    const real_t x288 = x201 * x268;
    const real_t x289 = 16 * x41;
    const real_t x290 = 4 * u[9];
    const real_t x291 = 4 * u[1];
    const real_t x292 = 5 * u[7];
    const real_t x293 = 8 * u[6];
    const real_t x294 = 4 * u[10];
    const real_t x295 = 6 * u[6];
    const real_t x296 = u[8] * x13;
    const real_t x297 = u[8] * x16;
    const real_t x298 = u[6] * x13;
    const real_t x299 = u[6] * x9;
    const real_t x300 = u[6] * x16;
    const real_t x301 = u[6] * x11;
    const real_t x302 = x120 * x4;
    const real_t x303 = u[8] * x5;
    const real_t x304 = x3 * x303;
    const real_t x305 = x2 * x62;
    const real_t x306 = 4 * x2;
    const real_t x307 = -x11 * x290 + x24 * x290 - x290 * x9 + x306 * x73;
    const real_t x308 = -x13 * x252 - x16 * x252 + x24 * x252 + x306 * x75;
    const real_t x309 = u[7] * x9;
    const real_t x310 = u[7] * x11;
    const real_t x311 = x113 * x2;
    const real_t x312 = u[7] * x3;
    const real_t x313 = x312 * x5;
    const real_t x314 = x309 + x310 - x311 - x313;
    const real_t x315 = pow(u[10], 2);
    const real_t x316 = 48 * x315;
    const real_t x317 = x148 * x316;
    const real_t x318 = 48 * u[9];
    const real_t x319 = -x318 * x55;
    const real_t x320 = 48 * u[10];
    const real_t x321 = x320 * x55;
    const real_t x322 = pow(u[7], 2);
    const real_t x323 = 9 * x4;
    const real_t x324 = pow(u[8], 2);
    const real_t x325 = 9 * x148;
    const real_t x326 = 12 * u[6];
    const real_t x327 = pow(u[9], 2);
    const real_t x328 = x2 * x327;
    const real_t x329 = pow(u[11], 2);
    const real_t x330 = x2 * x329;
    const real_t x331 = x149 * x327;
    const real_t x332 = u[11] * x61;
    const real_t x333 = 80 * u[11];
    const real_t x334 = 80 * u[9];
    const real_t x335 = u[11] * x26;
    const real_t x336 = 64 * x335;
    const real_t x337 = x320 * x98;
    const real_t x338 = 48 * u[6];
    const real_t x339 = 32 * u[11];
    const real_t x340 = 32 * u[7];
    const real_t x341 = 32 * u[9];
    const real_t x342 = x26 * x341;
    const real_t x343 = 24 * u[10];
    const real_t x344 = 24 * u[11];
    const real_t x345 = 20 * u[7];
    const real_t x346 = u[9] * x92;
    const real_t x347 = 20 * u[9];
    const real_t x348 = x100 * x3;
    const real_t x349 = 16 * u[10];
    const real_t x350 = 16 * x54;
    const real_t x351 = u[8] * x350;
    const real_t x352 = 16 * x45;
    const real_t x353 = u[9] * x352;
    const real_t x354 = 16 * u[8];
    const real_t x355 = x354 * x75;
    const real_t x356 = x5 * x96;
    const real_t x357 = u[9] * x356;
    const real_t x358 = 16 * u[7];
    const real_t x359 = 16 * u[9];
    const real_t x360 = x140 * x359;
    const real_t x361 = u[1] * x303;
    const real_t x362 = 9 * u[6];
    const real_t x363 = u[8] * x65;
    const real_t x364 = x294 * x37;
    const real_t x365 = x290 * x37;
    const real_t x366 = u[6] * x3;
    const real_t x367 = x116 * x37;
    const real_t x368 = 9 * u[7];
    const real_t x369 = u[10] * x356;
    const real_t x370 = x140 * x349;
    const real_t x371 = u[8] * x352;
    const real_t x372 = 16 * u[11];
    const real_t x373 = x26 * x354;
    const real_t x374 = 20 * u[8];
    const real_t x375 = x374 * x55;
    const real_t x376 = 24 * u[9];
    const real_t x377 = 32 * u[10];
    const real_t x378 = x26 * x377;
    const real_t x379 = 48 * u[11];
    const real_t x380 = x379 * x98;
    const real_t x381 = 96 * u[9];
    const real_t x382 = u[7] * x62;
    const real_t x383 = -u[11] * x352 + x374 * x73 + 9 * x382;
    const real_t x384 = u[11] * x5 * x91 + x316 * x4;
    const real_t x385 = 64 * u[9];
    const real_t x386 = u[11] * x350 + x30 * x385;
    const real_t x387 = 32 * x149;
    const real_t x388 =
        u[7] * x352 - x21 * x252 + x21 * x294 + x30 * x339 - x30 * x377 + x329 * x387 - x345 * x98;
    const real_t x389 = x302 + x304;
    const real_t x390 = -x305;
    const real_t x391 = u[6] * x24;
    const real_t x392 = -x391;
    const real_t x393 = -x24 * x294 - x306 * x45 + x390 + x392;
    const real_t x394 = x299 + x301;
    const real_t x395 = x2 * x324;
    const real_t x396 = 48 * x329;
    const real_t x397 = x166 * x5;
    const real_t x398 = 120 * u[7];
    const real_t x399 = x158 * x320;
    const real_t x400 = u[8] * x55;
    const real_t x401 = x303 * x94;
    const real_t x402 = 18 * u[7];
    const real_t x403 = u[2] * x289;
    const real_t x404 = 11 * u[6];
    const real_t x405 = u[6] * x41;
    const real_t x406 = u[7] * x41;
    const real_t x407 = x372 * x62;
    const real_t x408 = 20 * u[6];
    const real_t x409 = x140 * x408;
    const real_t x410 = 32 * u[6];
    const real_t x411 = x158 * x379;
    const real_t x412 = 48 * u[7];
    const real_t x413 = u[7] * x45;
    const real_t x414 = x317 + x319 + x321 - x359 * x75;
    const real_t x415 = u[11] * x37;
    const real_t x416 = u[11] * x102 - 20 * x332 - 16 * x335 + 20 * x415;
    const real_t x417 = x26 * x318;
    const real_t x418 = x26 * x320;
    const real_t x419 = -x165 * x385 + 48 * x328 - x417 + x418;
    const real_t x420 = pow(u[6], 2);
    const real_t x421 = 16 * u[6];
    const real_t x422 = u[6] * x102 + x165 * x421 - x189 * x349 + x189 * x372 + x323 * x420 +
                        x325 * x420 + x360 + x362 * x61 + x362 * x64 - x370;
    const real_t x423 = x311 + x313;
    const real_t x424 = x148 * x327;
    const real_t x425 = x212 * x41;
    const real_t x426 = 120 * u[8];
    const real_t x427 = x189 * x345;
    const real_t x428 = 4 * x61;
    const real_t x429 = 32 * x315;
    const real_t x430 = x2 * x429 + x26 * x408 + x341 * x55 - x377 * x55;
    const real_t x431 = -64 * u[11] * x55 + x149 * x396 + x399 - x411;
    const real_t x432 = 8 * u[9];
    const real_t x433 = 2 * u[9];
    const real_t x434 = x13 * x433;
    const real_t x435 = x16 * x433;
    const real_t x436 = 2 * u[10];
    const real_t x437 = 2 * x2;
    const real_t x438 = x437 * x45;
    const real_t x439 = x24 * x436;
    const real_t x440 = 2 * u[11];
    const real_t x441 = 12 * x420;
    const real_t x442 = 40 * u[6];
    const real_t x443 = 40 * u[7];
    const real_t x444 = x377 * x87;
    const real_t x445 = u[8] * x26;
    const real_t x446 = x326 * x61;
    const real_t x447 = 12 * x37;
    const real_t x448 = 8 * u[10];
    const real_t x449 = 8 * u[11];
    const real_t x450 = 8 * u[7];
    const real_t x451 = 4 * u[8];
    const real_t x452 = x252 * x265;
    const real_t x453 = u[6] * x5;
    const real_t x454 = x1 * x453;
    const real_t x455 = x1 * x366;
    const real_t x456 = 4 * u[7];
    const real_t x457 = 8 * u[8];
    const real_t x458 = x111 * x326;
    const real_t x459 = x140 * x372;
    const real_t x460 = x339 * x87;
    const real_t x461 = 96 * u[10];
    const real_t x462 = x140 * x341;
    const real_t x463 = x160 * x5;
    const real_t x464 = x140 * x377;
    const real_t x465 = u[10] * x463 - u[9] * x463 + 12 * x322 * x4 - x462 + x464;
    const real_t x466 = -2 * u[10] * x11 - 2 * u[10] * x9 - 2 * u[11] * x2 * x4 -
                        2 * u[11] * x3 * x5 - 2 * u[9] * x2 * x4 - 2 * u[9] * x3 * x5 + x11 * x440 +
                        x389 + x438 + x439 + x440 * x9;
    const real_t x467 = 96 * x315;
    const real_t x468 = 96 * u[11];
    const real_t x469 = 12 * u[10];
    const real_t x470 = x207 * x41;
    const real_t x471 = x285 * x75;
    const real_t x472 = 4 * u[6];
    const real_t x473 = 4 * x445;
    const real_t x474 = 12 * u[11];
    const real_t x475 = x285 * x45;
    const real_t x476 = x285 * x37;
    const real_t x477 = 24 * u[6];
    const real_t x478 = x113 * x290 + x241 * x75 + x290 * x62 + x326 * x45 + 16 * x331 -
                        x341 * x45 - x341 * x75 - x379 * x45 + x451 * x73;
    const real_t x479 = 20 * u[11];
    const real_t x480 = x207 * x3;
    element_vector[0] =
        x147 *
        (dt * x6 *
             (72 * u[0] * (x26 - x30) + 78 * u[0] * (x63 + x66) - u[3] * x89 + u[4] * x89 -
              u[4] * x90 + u[5] * x90 + x100 * x30 - x100 * x98 - 16 * x101 - x103 + x104 * x98 -
              x106 - x107 - 11 * x108 - x109 * x19 - x110 * x61 + x111 * x119 - 9 * x112 -
              x113 * x114 + x114 * x120 + x114 * x93 - x115 * x116 - x116 * x35 - x117 + x118 +
              x122 + x123 * x21 + x124 * x35 + x126 + x127 * x78 + x127 * x93 + x128 + x129 + x130 +
              x131 + x133 + 32 * x134 + x135 + x136 * x65 + x138 + x141 + x143 + x145 - x26 * x72 +
              x26 * x92 + x30 * x74 - x30 * x91 + x37 * x85 + x44 - x45 * x83 - x46 * x61 + x47 +
              x49 + x50 * x51 + x52 * x53 + x54 * x85 - x56 * x57 + x58 * (x19 - x35) + x59 * x60 -
              x62 * x71 + 80 * x68 + 80 * x70 + x72 * x78 - x73 * x74 + x73 * x79 + x75 * x79 -
              x77 - x78 * x79 - x78 * x95 - x79 * x80 + x79 * x87 - x82 - 32 * x84 - x86 -
              x87 * x88 - x87 * x92 - x93 * x94 - x93 * x95 - x97 - x99) -
         x40 * (-x0 * x23 - x0 * x24 + x10 + x12 + 3 * x14 + 3 * x15 + 3 * x17 + 3 * x18 - x20 -
                x22 + x28 + x32 + x39) -
         x8 * (u[1] + u[2] - x0 + x1));
    element_vector[1] =
        x202 *
        (-x151 * (u[0] - 6 * u[1] + u[2] + x29) -
         x157 * (3 * x10 + 3 * x12 + x15 + x152 + x156 + x18 + x28) -
         x200 * (-u[3] * x168 + 72 * u[4] * x162 + u[4] * x168 + u[4] * x175 - u[5] * x175 -
                 u[6] * x121 + x100 * x158 - x100 * x165 - x100 * x180 + x104 * x169 - x104 * x180 +
                 x104 * x64 + x106 - 18 * x108 - x109 * x115 + x110 * x64 + x113 * x186 -
                 x113 * x71 - x116 * x19 - x119 * x162 - x119 * x181 - x123 * x64 - x127 * x158 +
                 x127 * x178 + x127 * x181 - x129 + 48 * x134 + x138 - x158 * x95 + x159 * x160 +
                 x161 * x41 - x162 * x170 + 78 * x164 - x166 * x26 + x166 * x55 + x167 * x186 -
                 x169 * x170 + x169 * x48 - x171 - x172 * x55 - 32 * x174 + x176 * x187 - x177 -
                 x178 * x94 - x178 * x95 - 18 * x179 + x180 * x190 - x181 * x182 + x181 * x94 -
                 x181 * x95 + x183 * (u[4] - u[5]) + 11 * x184 + 18 * x185 + 20 * x188 +
                 x189 * x92 + x190 * x73 + 48 * x191 + x192 + x193 + x196 + x199 + x26 * x79 +
                 9 * x52 * x69 + x64 * x83 + x74 * (u[3] * x3 - x165 - x73) - x75 * x83));
    element_vector[2] =
        x202 *
        (-x151 * (u[0] + u[1] - 6 * u[2] + x25) -
         x157 * (x156 + x203 + x204 + x32 + 3 * x33 + 3 * x34) -
         x200 * (-u[0] * x124 * x148 + u[2] * x211 - u[3] * x214 + u[3] * x219 + u[4] * x213 +
                 u[4] * x214 - u[4] * x219 - u[5] * x213 + x104 * x113 + x104 * x62 - 9 * x108 -
                 x113 * x182 - x113 * x95 + x115 * x217 - x122 - x126 - x128 - x132 * x169 -
                 x136 * x181 + x141 - x158 * x212 - x162 * x216 - x162 * x48 + x165 * x212 +
                 x167 * x94 + x169 * x83 + x172 * x176 + 48 * x174 + x176 * x190 - 9 * x179 +
                 x180 * x187 - x180 * x85 + x181 * x186 - 9 * x184 - 11 * x185 + x187 * x55 -
                 x19 * x217 + x190 * x75 + x194 - x195 + x199 + 9 * x205 - x206 * x207 +
                 x208 * x69 + 48 * x209 - x215 + x216 * x64 + x220 * (-u[3] + u[4]) + 11 * x221 +
                 x222 + x223 + 72 * x224 + x225 + x226 + x227 - x37 * x46 - x45 * x88 + x46 * x64 +
                 x61 * x85 - x62 * x95 + x72 * (u[5] * x69 - x176 - x55) + x97));
    element_vector[3] =
        x268 *
        (x267 * (u[2] * x31 + 3 * u[2] * (u[2] * x69 + x120) - u[3] * x243 + u[3] * x244 +
                 u[4] * x243 - u[4] * x244 - u[6] * x255 - u[7] * x255 + x1 * x105 - x1 * x254 +
                 x1 * x62 - 24 * x101 + x104 * x65 + x104 * x78 - x105 * x29 + 4 * x108 +
                 x111 * x249 - x113 * x245 - x115 * x248 + x117 - x118 - x120 * x247 + x120 * x258 -
                 x127 * x30 - x131 - x133 - 40 * x134 - x135 - x139 + x143 - 8 * x185 - x187 * x78 -
                 x187 * x87 + x19 * x248 + x190 * x26 + x193 - x198 * x30 + 12 * x205 + 96 * x209 -
                 x21 * x249 + 8 * x221 + 8 * x224 + x225 + x228 * x65 + x235 * x41 + x235 * x5 +
                 x236 * x73 - x236 * x80 - x237 - x238 - x239 - x240 - x242 + x245 * x93 -
                 x245 * x98 - x246 * x65 - x251 + x252 * x35 - x253 + x256 + x257 * x54 +
                 x259 * x30 + x260 + x261 + x262 + x263 + x264 * x80 + x266 - x29 * x62 -
                 x29 * x93 + 32 * x41 * x59 + x43 * x69 - x71 * x80 + x82 + x86 + x87 * x91 + x99) +
         x40 * (u[0] * x11 + u[0] * x9 + u[1] * x11 + u[1] * x9 - x11 * x229 - x153 - x154 -
                x229 * x9 - x234) +
         x8 * (-u[2] + x1 + x228 + x29));
    element_vector[4] =
        x288 *
        (7 * x150 * x7 * (-u[0] + x246 + x25 + x29) -
         x157 * (2 * u[4] * x11 + 2 * u[4] * x9 - x152 - x234 - x270 - x271) -
         x200 *
             (-u[0] * (x162 + x37) + 3 * u[0] * (u[0] * x69 + x111 - x178 - x62) - u[3] * x275 +
              u[3] * x282 + u[4] * x275 - x104 * x176 - x104 * x181 - 8 * x108 - x113 * x127 -
              x113 * x246 + x113 * x247 + x113 * x250 + x132 * x162 + x132 * x176 - x136 * x45 +
              x158 * x160 - x158 * x207 - x158 * x48 - x158 * x83 + x162 * x25 - x162 * x277 +
              x165 * x207 - x165 * x83 + x167 * x250 + x167 * x29 + x169 * x264 + x169 * x284 -
              x169 * x71 + 40 * x174 - x176 * x250 - x176 * x83 + x177 + x178 * x247 - x178 * x276 -
              8 * x179 - x180 * x250 + x180 * x257 + x181 * x207 + x181 * x228 - x181 * x246 +
              x181 * x250 + 4 * x185 + 12 * x188 + 40 * x191 + x196 + x215 + 32 * x224 + x227 +
              x228 * x62 + x240 + x242 + x245 * x26 + x247 * x64 + x25 * x280 + x25 * x64 -
              x250 * x73 - x250 * x75 + x253 + x257 * x64 - x261 + x264 * x45 + x266 -
              x272 * (x173 + x19) + x273 * x41 + x273 * x5 - x274 - x276 * x62 - 8 * x278 +
              4 * x279 + x281 + x283 * x62 + x284 * x45 + x287 + 12 * x41 * x50 + 16 * x68));
    element_vector[5] =
        x268 *
        (-x267 * (-x1 * x265 + x1 * x65 + x1 * x93 + x103 + x104 * x54 - x104 * x80 - 5 * x108 -
                  x111 * x277 - x112 - x113 * x228 + x113 * x258 - x113 * x276 + x113 * x283 -
                  x120 * x245 + x120 * x94 - 24 * x125 + x127 * x62 - x132 * x54 - x136 * x75 +
                  x145 + x160 * x30 + 3 * x164 + 5 * x185 + x187 * x80 + x19 * x290 + x190 * x30 +
                  4 * x191 + x207 * x30 - x207 * x98 + x208 * x5 + x21 * x257 - 4 * x221 - x222 -
                  x223 + x226 + x235 * x3 + x235 * x69 - x236 * x75 + x236 * x78 + x237 + x239 -
                  x245 * x55 - x246 * x62 + x247 * x62 - x25 * x65 - x25 * x93 + x250 * x87 + x251 -
                  x256 + x257 * x87 - x258 * x93 + x259 * x98 + x26 * x277 + x26 * x291 - x260 -
                  x262 + x264 * x75 + x274 - 4 * x278 + 8 * x279 - x281 - x285 * x35 + x287 +
                  x289 * x67 - x290 * x35 + x291 * x87 + x292 * x35 + x293 * x35 + 96 * x3 * x59 -
                  x30 * x83 + x37 * x94 - x46 * x98 + x54 * x83 + x57 * x98 - x61 * x94 -
                  x65 * x94 + x77 - x83 * x98 - 40 * x84) +
         x40 * (x11 * x230 - x13 * x269 + x155 - x16 * x269 + x204 + x229 * x23 + x23 * x269 +
                x230 * x9 - x231 + x232 * x30 - x233 + 2 * x27 - x270 - x271 + x39) +
         x8 * (-u[1] + x1 + x247 + x25));
    element_vector[6] =
        x147 *
        (dt * x6 *
             (-u[10] * x348 + u[11] * x348 - u[1] * u[6] * x53 + 72 * u[6] * (x75 - x80) +
              78 * u[6] * (-x111 + x178 + x63) - u[7] * x350 + u[9] * x350 - x109 * x113 -
              x110 * x312 + x111 * x343 - x111 * x344 + x111 * x381 + x113 * x341 - x116 * x62 +
              x119 * x366 + x120 * x124 - x120 * x339 - x120 * x347 + x123 * x312 - x140 * x333 +
              x140 * x338 + x186 * (u[8] * x5 - x312) + x21 * x362 + x26 * x338 - x3 * x346 -
              x30 * x338 - x30 * x358 + x303 * x85 + x317 - x318 * x62 + x319 + x321 + x322 * x323 +
              x324 * x325 + x326 * x56 + 32 * x328 + 80 * x330 + 80 * x331 - 96 * x332 +
              x333 * x80 - x334 * x75 + x334 * x87 - x336 - x337 - x338 * x87 - x339 * x55 -
              x340 * x87 + x341 * x98 - x342 - x343 * x61 + x345 * x75 - x345 * x78 - x349 * x87 -
              x351 - x353 - x355 - x357 + x358 * x80 - x360 - 11 * x361 - x362 * x37 - 9 * x363 -
              x364 + x365 + x367 + x368 * x65 + x369 + x370 + x371 + x372 * x87 + x373 + x375 +
              x376 * x61 + x378 + x379 * x65 + x380 + x383 + x384 + x386 + x388) -
         x40 * (-x24 * x295 + x296 + x297 + 3 * x298 + 3 * x299 + 3 * x300 + 3 * x301 - x302 -
                x304 - 6 * x305 + x307 + x308 + x314) -
         x8 * (u[7] + u[8] + x294 - x295));
    element_vector[7] =
        x202 *
        (-x151 * (u[6] - 6 * u[7] + u[8] + x252) -
         x157 * (x307 + 3 * x309 + 3 * x310 + x389 + x393 + x394) -
         x200 * (u[10] * x397 - u[10] * x403 + u[11] * x403 + 78 * u[7] * x163 - u[9] * x397 +
                 x109 * x61 - x109 * x62 - x113 * x116 - x113 * x318 - x119 * x406 - x123 * x405 -
                 x158 * x341 + x158 * x408 - x159 * x241 + x162 * x343 - x162 * x344 - x162 * x381 +
                 x162 * x58 + x165 * x339 - x165 * x377 + x165 * x412 + x169 * x398 + x178 * x402 +
                 x180 * x338 - x180 * x398 - x181 * x320 - x181 * x402 + x189 * x410 + x189 * x412 +
                 x26 * x410 + x265 * x58 + x315 * x387 + x334 * (u[9] * x149 - x189 - x45) +
                 x341 * x62 + x346 * x41 - x349 * x37 - x349 * x61 + x355 + x359 * x37 +
                 x359 * x61 - 18 * x361 - x367 - x368 * x64 - x37 * x404 - x371 - x373 + x383 +
                 9 * x395 + x396 * x4 - x399 - 32 * x400 - x401 + x407 + x408 * x45 + x409 + x411 +
                 72 * x413 + x414 + x416 + x419 + x422 + 4 * x64 * (u[10] - u[11])));
    element_vector[8] =
        x202 *
        (-x151 * (u[6] + u[7] - 6 * u[8] + x290) -
         x157 * (3 * x296 + 3 * x297 + x298 + x300 + x308 + x393 + x423) -
         x200 * (20 * u[10] * x178 + 72 * u[10] * x181 + u[10] * x425 - u[11] * x425 + u[8] * x211 -
                 x110 * x303 - x113 * x217 - x116 * x61 - x124 * x178 + x124 * x64 + x140 * x410 +
                 9 * x149 * x322 - x158 * x358 - x158 * x359 + x158 * x410 + x162 * x347 -
                 x162 * x349 - x162 * x368 + x162 * x372 - x162 * x404 + x165 * x320 - x165 * x340 +
                 x165 * x359 - x165 * x379 - x169 * x358 + x176 * x345 + x176 * x410 - x180 * x349 +
                 x180 * x358 - x180 * x372 + x180 * x421 + x181 * x362 - x181 * x368 - x181 * x379 +
                 x189 * x408 + x206 * x285 - x216 * x406 + x217 * x62 - x265 * x404 + x280 * x58 +
                 x303 * x46 + x333 * (u[11] * x2 - x140 - x169) - 32 * x335 + x343 * x37 -
                 x347 * x64 - x349 * x64 + x357 - x369 - x37 * x376 + x37 * x58 + x372 * x64 +
                 x379 * x62 + x384 + 48 * x400 - x412 * x45 - 96 * x415 + x417 - x418 + x422 +
                 48 * x424 + x426 * x45 - x426 * x75 - x427 + x428 * (u[10] - u[9]) + x430 + x431));
    element_vector[9] =
        x268 *
        (-x267 *
             (-u[10] * x447 - u[7] * x31 + 3 * u[8] * x210 - u[8] * x428 + u[9] * x244 +
              u[9] * x447 - u[9] * x81 + x1 * x303 - x111 * x252 + x111 * x294 - x111 * x347 +
              x111 * x450 - x113 * x248 + x113 * x347 + x120 * x252 + x120 * x456 + x149 * x441 -
              x160 * x312 + x2 * x441 - x21 * x293 + x21 * x347 + x218 * x329 - x241 * x54 +
              x241 * x80 + x248 * x62 + x249 * x312 - x252 * x254 + x252 * x65 + x252 * x93 -
              x254 * x272 + x254 * x294 - x26 * x381 - x26 * x442 + x26 * x461 + x265 * x293 +
              x272 * x37 + x29 * x303 - x291 * x303 - x293 * x45 - x30 * x359 + x30 * x408 +
              x326 * x54 + 96 * x328 + 16 * x332 + x336 - x339 * x54 - x339 * x80 + x339 * x98 +
              x341 * x65 - x347 * x62 + x353 + x359 * x98 - x363 - x37 * x449 - x377 * x98 + x388 +
              x4 * x429 - x407 - x409 + x414 + x432 * x61 + x442 * x87 - x443 * x87 - x444 -
              24 * x445 - x446 - x448 * x61 - x45 * x451 - x450 * x65 + x450 * x75 + x451 * x75 -
              x452 - x454 + x455 + x457 * x54 + x458 + x459 + x460 + x465) +
         x40 * (-x11 * x433 + x13 * x436 + x16 * x436 + x24 * x433 + x24 * x440 + x314 + x390 +
                x392 + x394 - x433 * x9 - x434 - x435 + x437 * x73 + x437 * x75 - x438 - x439) +
         x8 * (-u[8] + x252 + x294 + x432));
    element_vector[10] =
        x288 *
        (7 * x150 * x7 * (-u[6] + x252 + x290 + x448) -
         x157 * (2 * u[10] * x13 + 2 * u[10] * x16 - x423 - x434 - x435 - x466) -
         x200 * (u[10] * x282 + u[10] * x470 - u[11] * x470 - u[6] * (x113 + x181) +
                 3 * u[6] * (u[6] * x149 - x61 - x66) + u[7] * x220 + u[8] * x183 +
                 12 * u[9] * x61 - x113 * x457 + 12 * x148 * x324 + x148 * x467 + x158 * x456 -
                 x158 * x472 + x162 * x241 - x162 * x359 - x162 * x448 + x162 * x449 + x162 * x472 +
                 x165 * x443 + x165 * x461 - x165 * x468 + x165 * x477 + x169 * x241 - x176 * x377 +
                 x176 * x456 + x176 * x472 - x180 * x241 + x180 * x285 + x180 * x293 - x180 * x320 -
                 x180 * x339 + x181 * x252 + x181 * x377 - x181 * x450 - x189 * x339 - x189 * x359 +
                 x189 * x377 - x25 * x405 + x257 * x405 - x258 * (x303 + x406) - x259 * x303 -
                 x26 * x472 + x265 * x472 - x277 * x406 + x280 * x290 - x29 * x453 + 16 * x330 +
                 x340 * x45 - x37 * x372 + x37 * x432 - x37 * x448 + x37 * x472 - x381 * x55 +
                 x4 * x467 + 40 * x400 + x401 + x419 + x427 + x431 + x432 * x64 + x449 * x61 +
                 x449 * x62 + x452 - x459 + x461 * x55 + x465 - x469 * x61 - x469 * x64 - x471 +
                 x473 + x474 * x64 + x475 + x476 + x477 * x55 + x478));
    element_vector[11] =
        x268 *
        (x267 * (u[10] * x480 - u[11] * x480 - 40 * u[5] * x303 - 24 * u[7] * x30 +
                 3 * u[7] * (u[7] * x149 + x21) + x1 * x312 + x111 * x359 - x111 * x448 +
                 x111 * x449 - x111 * x456 + x113 * x451 - x120 * x292 + x120 * x432 + x120 * x479 +
                 x140 * x442 + x148 * x441 + x149 * x316 + x21 * x272 - x21 * x432 - x21 * x469 +
                 x21 * x474 - x254 * x290 + x254 * x293 - x257 * x312 - x265 * x272 - x265 * x290 +
                 x265 * x294 - x290 * x61 + x292 * x65 - x293 * x37 - x293 * x54 + x294 * x61 -
                 x30 * x442 + x30 * x461 - x30 * x468 + 96 * x329 * x4 + x337 + x339 * x62 + x342 +
                 x351 - x359 * x65 + x359 * x87 + 5 * x361 + x364 - x365 - x372 * x80 - x375 -
                 x378 - x380 - x382 + x386 + 12 * x395 + x4 * x441 - x408 * x87 + 8 * x413 + x416 +
                 32 * x424 + x430 + x444 + x446 + x454 - x455 - x456 * x54 + x456 * x80 +
                 x456 * x87 + x457 * x61 - x457 * x62 - x458 - x460 + x462 - x464 + x471 - x473 -
                 x475 - x476 + x478 - x479 * x65) +
         x40 * (u[6] * x13 + u[6] * x16 + u[8] * x13 + u[8] * x16 - x13 * x440 - x16 * x440 - x305 -
                x391 - x466) +
         x8 * (-u[7] + x290 + x294 + x449));
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
#pragma omp for nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[6];
            real_t element_vector[3];
            real_t element_x[6 * 2];

#pragma unroll(3)
            for (int v = 0; v < element_nnodes; ++v) {
                ev[v] = elems[v][i];
            }

            for (int enode = 0; enode < element_nnodes; ++enode) {
                idx_t dof = ev[enode];

                for (int b = 0; b < n_vars; ++b) {
                    element_x[b * element_nnodes + enode] = vel[b][dof];
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
                element_x,
                element_vector);

            for (int edof_i = 0; edof_i < 3; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];
#pragma omp atomic update
                f[dof_i] += element_vector[edof_i];
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tri6_naviers_stokes.c: tri6_explict_momentum_tentative\t%g seconds\n", tock - tick);
}

void tri6_tri3_correction(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const elems,
                          geom_t **const points,
                          const real_t dt,
                          const real_t rho,
                          real_t **const SFEM_RESTRICT vel,
                          real_t *const SFEM_RESTRICT p,
                          real_t **const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);
    double tick = MPI_Wtime();

    static const int n_vars = 2;
    static const int element_nnodes = 6;

#pragma omp parallel
    {
#pragma omp for nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[6];
            real_t element_vector[6 * 2];
            real_t element_vel[6 * 2];
            real_t element_p[3];

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

#pragma unroll(3)
            for (int enode = 0; enode < 3; ++enode) {
                idx_t dof = ev[enode];
                element_p[enode] = p[dof];
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
                element_vel,
                element_p,
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
    printf("tri6_naviers_stokes.c: tri6_explict_momentum_tentative\t%g seconds\n", tock - tick);
}

void tri6_explict_momentum_tentative(const ptrdiff_t nelements,
                                     const ptrdiff_t nnodes,
                                     idx_t **const elems,
                                     geom_t **const points,
                                     const real_t dt,
                                     const real_t nu,
                                     real_t **const SFEM_RESTRICT vel,
                                     real_t **const SFEM_RESTRICT f) {
    SFEM_UNUSED(nnodes);
    double tick = MPI_Wtime();

    static const int n_vars = 2;
    static const int element_nnodes = 6;

#pragma omp parallel
    {
#pragma omp for nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[6];
            real_t element_vector[6 * 2];
            real_t element_x[6 * 2];

#pragma unroll(3)
            for (int v = 0; v < element_nnodes; ++v) {
                ev[v] = elems[v][i];
            }

            for (int enode = 0; enode < element_nnodes; ++enode) {
                idx_t dof = ev[enode];

                for (int b = 0; b < n_vars; ++b) {
                    element_x[b * element_nnodes + enode] = vel[b][dof];
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

            tri6_explict_momentum_rhs_kernel(
                // X coords
                points[0][i0],
                points[0][i1],
                points[0][i2],
                // Y coords
                points[1][i0],
                points[1][i1],
                points[1][i2],
                dt,
                nu,
                //  buffers
                element_x,
                element_vector);

            for (int d1 = 0; d1 < n_vars; d1++) {
                for (int edof_i = 0; edof_i < element_nnodes; ++edof_i) {
                    const idx_t dof_i = elems[edof_i][i];

#pragma omp atomic update
                    f[d1][dof_i] += element_vector[d1 * element_nnodes + edof_i];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tri6_naviers_stokes.c: tri6_explict_momentum_tentative\t%g seconds\n", tock - tick);
}