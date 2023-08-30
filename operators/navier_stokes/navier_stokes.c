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
    const real_t x0 = 6 * u[0];
    const real_t x1 = 4 * u[4];
    const real_t x2 = px0 - px1;
    const real_t x3 = py0 - py2;
    const real_t x4 = px0 - px2;
    const real_t x5 = py0 - py1;
    const real_t x6 = x2 * x3 - x4 * x5;
    const real_t x7 = 7 * pow(x6, 2);
    const real_t x8 = pow(x4, 2);
    const real_t x9 = u[1] * x8;
    const real_t x10 = pow(x3, 2);
    const real_t x11 = u[1] * x10;
    const real_t x12 = pow(x2, 2);
    const real_t x13 = u[0] * x12;
    const real_t x14 = u[0] * x8;
    const real_t x15 = pow(x5, 2);
    const real_t x16 = u[0] * x15;
    const real_t x17 = u[0] * x10;
    const real_t x18 = u[1] * x4;
    const real_t x19 = x18 * x2;
    const real_t x20 = u[1] * x3;
    const real_t x21 = x20 * x5;
    const real_t x22 = x2 * x4;
    const real_t x23 = x3 * x5;
    const real_t x24 = 4 * u[3];
    const real_t x25 = u[3] * x5;
    const real_t x26 = x25 * x3;
    const real_t x27 = -x10 * x24 + x22 * x24 - x24 * x8 + 4 * x26;
    const real_t x28 = 4 * u[5];
    const real_t x29 = u[5] * x3;
    const real_t x30 = x29 * x5;
    const real_t x31 = -x12 * x28 - x15 * x28 + x22 * x28 + 4 * x30;
    const real_t x32 = u[2] * x12;
    const real_t x33 = u[2] * x15;
    const real_t x34 = u[2] * x2;
    const real_t x35 = x34 * x4;
    const real_t x36 = u[2] * x5;
    const real_t x37 = x3 * x36;
    const real_t x38 = x32 + x33 - x35 - x37;
    const real_t x39 = 420 * dt * nu;
    const real_t x40 = pow(u[2], 2);
    const real_t x41 = 9 * x5;
    const real_t x42 = u[10] * x2;
    const real_t x43 = u[4] * x5;
    const real_t x44 = 12 * u[10] * x4 + 12 * u[4] * x3 - 12 * x42 - 12 * x43;
    const real_t x45 = u[6] * (x18 - x34);
    const real_t x46 = pow(u[5], 2);
    const real_t x47 = 32 * x3;
    const real_t x48 = -x3;
    const real_t x49 = u[6] * x2;
    const real_t x50 = u[0] * x5;
    const real_t x51 = u[6] * x4;
    const real_t x52 = x50 + x51;
    const real_t x53 = u[0] * (u[0] * x48 - x49 + x52);
    const real_t x54 = pow(u[3], 2);
    const real_t x55 = -x5;
    const real_t x56 = x46 * x55;
    const real_t x57 = 96 * u[3];
    const real_t x58 = 80 * u[5];
    const real_t x59 = u[9] * x4;
    const real_t x60 = 80 * u[3];
    const real_t x61 = u[11] * x4;
    const real_t x62 = 64 * u[3];
    const real_t x63 = x61 * x62;
    const real_t x64 = u[11] * x2;
    const real_t x65 = 48 * u[0];
    const real_t x66 = 48 * u[5];
    const real_t x67 = u[9] * x2;
    const real_t x68 = u[10] * x4;
    const real_t x69 = 32 * u[3];
    const real_t x70 = u[11] * x34;
    const real_t x71 = 32 * u[5];
    const real_t x72 = x61 * x71;
    const real_t x73 = u[3] * x3;
    const real_t x74 = 32 * x73;
    const real_t x75 = 24 * x49;
    const real_t x76 = 24 * x51;
    const real_t x77 = 20 * u[1];
    const real_t x78 = u[7] * x2;
    const real_t x79 = 20 * u[5];
    const real_t x80 = x78 * x79;
    const real_t x81 = u[4] * x3;
    const real_t x82 = 16 * u[1];
    const real_t x83 = x81 * x82;
    const real_t x84 = 16 * u[2];
    const real_t x85 = x81 * x84;
    const real_t x86 = u[9] * x34;
    const real_t x87 = 16 * x86;
    const real_t x88 = 16 * x43;
    const real_t x89 = u[3] * x88;
    const real_t x90 = u[5] * x88;
    const real_t x91 = 16 * u[5];
    const real_t x92 = u[8] * x4;
    const real_t x93 = x91 * x92;
    const real_t x94 = u[1] * x36;
    const real_t x95 = u[8] * x18;
    const real_t x96 = u[1] * x50;
    const real_t x97 = u[7] * x4;
    const real_t x98 = 9 * u[0];
    const real_t x99 = 9 * x34;
    const real_t x100 = u[8] * x99;
    const real_t x101 = x1 * x97;
    const real_t x102 = x28 * x97;
    const real_t x103 = u[2] * x50;
    const real_t x104 = 9 * x103;
    const real_t x105 = u[0] * x3;
    const real_t x106 = 9 * u[2];
    const real_t x107 = 9 * x2;
    const real_t x108 = u[8] * x2;
    const real_t x109 = 11 * u[2];
    const real_t x110 = 11 * x34;
    const real_t x111 = x29 * x84;
    const real_t x112 = 16 * u[4];
    const real_t x113 = x112 * x92;
    const real_t x114 = x68 * x77;
    const real_t x115 = u[5] * x5;
    const real_t x116 = x115 * x77;
    const real_t x117 = 20 * u[2];
    const real_t x118 = 32 * u[4];
    const real_t x119 = x118 * x61;
    const real_t x120 = u[9] * x18;
    const real_t x121 = 32 * u[2];
    const real_t x122 = 96 * u[5];
    const real_t x123 = pow(u[4], 2);
    const real_t x124 = 48 * x123;
    const real_t x125 = 48 * x42;
    const real_t x126 = u[4] * x125;
    const real_t x127 = u[0] * x20;
    const real_t x128 = 20 * u[3];
    const real_t x129 = u[3] * x125;
    const real_t x130 = -x117 * x73 + x124 * x5 - x126 - 9 * x127 + x128 * x92 + x129 + x81 * x91;
    const real_t x131 = pow(u[1], 2);
    const real_t x132 = 9 * x48;
    const real_t x133 = x112 * x64;
    const real_t x134 = x59 * x91;
    const real_t x135 = 9 * x18;
    const real_t x136 = u[11] * x18;
    const real_t x137 = 16 * u[3];
    const real_t x138 = x137 * x64;
    const real_t x139 = x112 * x59;
    const real_t x140 = u[7] * x135 - x112 * x78 + x131 * x132 - x133 - x134 + 16 * x136 +
                        x137 * x78 + x138 + x139 - x25 * x82 + x43 * x82;
    const real_t x141 = 64 * u[5];
    const real_t x142 = x137 * x81 + x141 * x67;
    const real_t x143 = x54 * x55;
    const real_t x144 = x66 * x68;
    const real_t x145 = 48 * u[4];
    const real_t x146 = x145 * x68;
    const real_t x147 = x1 * x108 - x108 * x24 - x117 * x42 - x118 * x67 + x124 * x48 + 32 * x143 -
                        x144 + x146 + x43 * x84 + x67 * x69;
    const real_t x148 = 1.0 / x6;
    const real_t x149 = (1.0 / 2520.0) * x148;
    const real_t x150 = x35 + x37;
    const real_t x151 = u[0] * x22;
    const real_t x152 = x3 * x50;
    const real_t x153 = -x151 - x152;
    const real_t x154 = -x1 * x22 - x1 * x23 + x153;
    const real_t x155 = pow(u[0], 2);
    const real_t x156 = 9 * x55;
    const real_t x157 = 12 * x29 + 12 * x61;
    const real_t x158 = u[1] * (u[1] * x48 + x97);
    const real_t x159 = 120 * u[1];
    const real_t x160 = 120 * x78;
    const real_t x161 = u[1] * x81;
    const real_t x162 = x62 * x68;
    const real_t x163 = u[2] * x43;
    const real_t x164 = 32 * u[0];
    const real_t x165 = 24 * x97;
    const real_t x166 = 20 * u[0];
    const real_t x167 = x108 * x79;
    const real_t x168 = 18 * u[0];
    const real_t x169 = 16 * u[0];
    const real_t x170 = x169 * x42;
    const real_t x171 = x42 * x91;
    const real_t x172 = 11 * u[0];
    const real_t x173 = x49 * x98;
    const real_t x174 = 4 * x51;
    const real_t x175 = x51 * x98;
    const real_t x176 = x169 * x68;
    const real_t x177 = x137 * x29;
    const real_t x178 = x67 * x91;
    const real_t x179 = u[7] * x34;
    const real_t x180 = x49 * x79;
    const real_t x181 = 48 * u[1];
    const real_t x182 = x46 * x48;
    const real_t x183 = 48 * x182;
    const real_t x184 = -x145 * x61;
    const real_t x185 = x61 * x66;
    const real_t x186 = x183 + x184 + x185 + 20 * x70;
    const real_t x187 = x145 * x67;
    const real_t x188 = 48 * u[3];
    const real_t x189 = x188 * x67;
    const real_t x190 = 48 * x143 - x187 + x189;
    const real_t x191 = -x118 * x68 + x123 * x47 + x166 * x61 - x166 * x64 - x169 * x29 + x68 * x71;
    const real_t x192 = convonoff * dt * x6;
    const real_t x193 = x19 + x21;
    const real_t x194 = x13 + x16;
    const real_t x195 = x155 * x3;
    const real_t x196 = 12 * x25 + 12 * x67;
    const real_t x197 = u[2] * (u[2] * x55 + x108);
    const real_t x198 = 120 * u[2];
    const real_t x199 = 120 * x92;
    const real_t x200 = -x141 * x42;
    const real_t x201 = 24 * x108;
    const real_t x202 = x128 * x97;
    const real_t x203 = 18 * u[2];
    const real_t x204 = x137 * x68;
    const real_t x205 = u[6] * x18;
    const real_t x206 = 4 * x49;
    const real_t x207 = x137 * x61;
    const real_t x208 = x25 * x91;
    const real_t x209 = 20 * x120;
    const real_t x210 = x128 * x51;
    const real_t x211 = u[1] * x68;
    const real_t x212 = 48 * x115;
    const real_t x213 =
        -x118 * x42 + 32 * x123 * x5 - x166 * x59 + x166 * x67 - x169 * x25 + x42 * x69;
    const real_t x214 = 8 * u[3];
    const real_t x215 = 2 * u[3];
    const real_t x216 = 2 * u[4];
    const real_t x217 = x215 * x22;
    const real_t x218 = 2 * x26;
    const real_t x219 = 2 * u[5];
    const real_t x220 = x219 * x22;
    const real_t x221 = 2 * x30;
    const real_t x222 = x216 * x22;
    const real_t x223 = 2 * x5 * x81;
    const real_t x224 = x12 * x215 - x12 * x216 + x15 * x215 - x15 * x216 + x193 - x217 - x218 -
                        x220 - x221 + x222 + x223;
    const real_t x225 = 12 * x155;
    const real_t x226 = x5 * x54;
    const real_t x227 = x188 * x43;
    const real_t x228 = 40 * u[0];
    const real_t x229 = x25 * x71;
    const real_t x230 = x43 * x71;
    const real_t x231 = 12 * u[0];
    const real_t x232 = x231 * x49;
    const real_t x233 = 12 * x108;
    const real_t x234 = 8 * u[0];
    const real_t x235 = 8 * u[4];
    const real_t x236 = 8 * u[5];
    const real_t x237 = 5 * u[0];
    const real_t x238 = 5 * u[2];
    const real_t x239 = 4 * u[0];
    const real_t x240 = x239 * x42;
    const real_t x241 = u[2] * x3;
    const real_t x242 = u[6] * x34;
    const real_t x243 = x28 * x78;
    const real_t x244 = x239 * x68;
    const real_t x245 = x28 * x50;
    const real_t x246 = 4 * u[2];
    const real_t x247 = 4 * u[11];
    const real_t x248 = u[1] * x5;
    const real_t x249 = x248 * x28;
    const real_t x250 = x28 * x36;
    const real_t x251 = 8 * u[1];
    const real_t x252 = x231 * x43;
    const real_t x253 = x231 * x51;
    const real_t x254 = 12 * u[2];
    const real_t x255 = x25 * x254;
    const real_t x256 = x64 * x91;
    const real_t x257 = 96 * u[4];
    const real_t x258 = 12 * x3;
    const real_t x259 = x64 * x69;
    const real_t x260 = x118 * x59;
    const real_t x261 = 12 * u[1];
    const real_t x262 = 12 * u[7];
    const real_t x263 = 12 * x2;
    const real_t x264 = u[7] * x263;
    const real_t x265 = x118 * x64;
    const real_t x266 = x59 * x71;
    const real_t x267 = -u[3] * x264 + u[4] * x264 + x131 * x258 - x18 * x247 - x18 * x262 +
                        x25 * x261 - x259 - x260 - x261 * x43 + x265 + x266;
    const real_t x268 = (1.0 / 630.0) * x148;
    const real_t x269 = x216 * x8;
    const real_t x270 = x10 * x216;
    const real_t x271 = x219 * x8;
    const real_t x272 = x10 * x219;
    const real_t x273 = 12 * x40;
    const real_t x274 = 16 * x5;
    const real_t x275 = 96 * x123;
    const real_t x276 = 24 * u[0];
    const real_t x277 = x254 * x29;
    const real_t x278 = 12 * u[4];
    const real_t x279 = x278 * x92;
    const real_t x280 = 12 * u[5];
    const real_t x281 = 8 * u[2];
    const real_t x282 = 4 * u[9];
    const real_t x283 = x282 * x34;
    const real_t x284 = x254 * x81;
    const real_t x285 = 12 * u[8];
    const real_t x286 = x285 * x34;
    const real_t x287 = x280 * x92;
    const real_t x288 = 4 * u[1];
    const real_t x289 = x137 * x59 + x231 * x81 + x239 * x73 - x24 * x92 + x246 * x73 + x261 * x29 +
                        x288 * x73 - x29 * x69 + 16 * x48 * x54 - x66 * x81 - x69 * x81;
    const real_t x290 = 4 * u[10];
    const real_t x291 = 6 * u[6];
    const real_t x292 = u[8] * x12;
    const real_t x293 = u[8] * x15;
    const real_t x294 = u[6] * x12;
    const real_t x295 = u[6] * x8;
    const real_t x296 = u[6] * x15;
    const real_t x297 = u[6] * x10;
    const real_t x298 = x108 * x4;
    const real_t x299 = u[8] * x5;
    const real_t x300 = x299 * x3;
    const real_t x301 = x2 * x51;
    const real_t x302 = 4 * x2;
    const real_t x303 = -x10 * x282 + x23 * x282 - x282 * x8 + x302 * x59;
    const real_t x304 = -x12 * x247 - x15 * x247 + x23 * x247 + x302 * x61;
    const real_t x305 = u[7] * x8;
    const real_t x306 = u[7] * x10;
    const real_t x307 = x2 * x97;
    const real_t x308 = u[7] * x3;
    const real_t x309 = x308 * x5;
    const real_t x310 = x305 + x306 - x307 - x309;
    const real_t x311 = pow(u[8], 2);
    const real_t x312 = -x299 + x308;
    const real_t x313 = pow(u[11], 2);
    const real_t x314 = 32 * x4;
    const real_t x315 = -x4;
    const real_t x316 = -x2;
    const real_t x317 = x313 * x316;
    const real_t x318 = pow(u[9], 2);
    const real_t x319 = x318 * x4;
    const real_t x320 = 96 * u[9];
    const real_t x321 = 80 * u[11];
    const real_t x322 = 80 * u[9];
    const real_t x323 = 64 * u[9];
    const real_t x324 = x29 * x323;
    const real_t x325 = 48 * u[11];
    const real_t x326 = 48 * u[6];
    const real_t x327 = 32 * u[11];
    const real_t x328 = x29 * x327;
    const real_t x329 = 32 * u[9];
    const real_t x330 = u[8] * x115;
    const real_t x331 = 24 * u[10];
    const real_t x332 = u[9] * x50;
    const real_t x333 = u[11] * x5 * x77;
    const real_t x334 = 20 * u[7];
    const real_t x335 = 16 * x42;
    const real_t x336 = u[11] * x335;
    const real_t x337 = 16 * x68;
    const real_t x338 = u[7] * x337;
    const real_t x339 = u[8] * x337;
    const real_t x340 = u[9] * x335;
    const real_t x341 = x3 * x84;
    const real_t x342 = u[11] * x341;
    const real_t x343 = 16 * u[8];
    const real_t x344 = x25 * x343;
    const real_t x345 = u[7] * x108;
    const real_t x346 = 9 * x20;
    const real_t x347 = u[6] * x3;
    const real_t x348 = 9 * x36;
    const real_t x349 = u[8] * x348;
    const real_t x350 = u[7] * x49;
    const real_t x351 = x20 * x290;
    const real_t x352 = x20 * x247;
    const real_t x353 = u[8] * x49;
    const real_t x354 = 9 * x353;
    const real_t x355 = 9 * u[8];
    const real_t x356 = u[1] * x299;
    const real_t x357 = 11 * u[8];
    const real_t x358 = u[10] * x341;
    const real_t x359 = x343 * x61;
    const real_t x360 = x334 * x64;
    const real_t x361 = x334 * x81;
    const real_t x362 = 20 * u[8];
    const real_t x363 = 24 * u[11];
    const real_t x364 = 32 * u[10];
    const real_t x365 = x29 * x364;
    const real_t x366 = 48 * u[9];
    const real_t x367 = u[11] * x50;
    const real_t x368 = pow(u[10], 2);
    const real_t x369 = 48 * x368;
    const real_t x370 = 48 * u[10];
    const real_t x371 = x370 * x43;
    const real_t x372 = u[7] * x51;
    const real_t x373 = x366 * x43;
    const real_t x374 =
        u[11] * x337 + u[9] * x117 * x3 + x2 * x369 - x362 * x59 - x371 - 9 * x372 + x373;
    const real_t x375 = pow(u[7], 2);
    const real_t x376 = 9 * x315;
    const real_t x377 = u[1] * x274;
    const real_t x378 = 16 * u[10];
    const real_t x379 = x115 * x378;
    const real_t x380 = 16 * u[11];
    const real_t x381 = x380 * x73;
    const real_t x382 = 16 * u[7];
    const real_t x383 = x378 * x73;
    const real_t x384 = 16 * u[9];
    const real_t x385 = x115 * x384;
    const real_t x386 = -u[10] * x377 + u[7] * x335 + u[7] * x346 + u[9] * x377 + x29 * x382 +
                        x375 * x376 - x379 - x381 - x382 * x67 + x383 + x385;
    const real_t x387 = 64 * u[11];
    const real_t x388 = u[9] * x337 + x25 * x387;
    const real_t x389 = x316 * x318;
    const real_t x390 = x325 * x81;
    const real_t x391 = x370 * x81;
    const real_t x392 = u[8] * x335 + x25 * x329 - x25 * x364 - x282 * x36 + x290 * x36 +
                        x315 * x369 - x362 * x43 + 32 * x389 - x390 + x391;
    const real_t x393 = x298 + x300;
    const real_t x394 = -x301;
    const real_t x395 = u[6] * x23;
    const real_t x396 = -x395;
    const real_t x397 = -x23 * x290 - x302 * x68 + x394 + x396;
    const real_t x398 = x295 + x297;
    const real_t x399 = pow(u[6], 2);
    const real_t x400 = 9 * x316;
    const real_t x401 = u[7] * (u[7] * x315 + x20);
    const real_t x402 = x159 * x5;
    const real_t x403 = 120 * u[7];
    const real_t x404 = u[7] * x68;
    const real_t x405 = 32 * u[6];
    const real_t x406 = 20 * u[6];
    const real_t x407 = u[11] * x36;
    const real_t x408 = 20 * x407;
    const real_t x409 = 18 * u[6];
    const real_t x410 = u[11] * x88;
    const real_t x411 = u[6] * x88;
    const real_t x412 = 9 * u[6];
    const real_t x413 = x412 * x50;
    const real_t x414 = x105 * x412;
    const real_t x415 = 11 * u[6];
    const real_t x416 = x25 * x380;
    const real_t x417 = x384 * x61;
    const real_t x418 = u[6] * x81;
    const real_t x419 = 16 * x418;
    const real_t x420 = 20 * x367;
    const real_t x421 = 20 * x330;
    const real_t x422 = u[8] * x43;
    const real_t x423 = 48 * u[7];
    const real_t x424 = 48 * x389;
    const real_t x425 = -x25 * x370;
    const real_t x426 = x25 * x366;
    const real_t x427 = -x323 * x81 + x424 + x425 + x426;
    const real_t x428 = x313 * x315;
    const real_t x429 = x29 * x370;
    const real_t x430 = x29 * x325;
    const real_t x431 = 48 * x428 - x429 + x430;
    const real_t x432 =
        -x115 * x406 + x29 * x406 + x314 * x368 + x327 * x81 - x364 * x81 - x380 * x51;
    const real_t x433 = x307 + x309;
    const real_t x434 = x399 * x4;
    const real_t x435 = u[8] * (u[8] * x316 + x36);
    const real_t x436 = x198 * x3;
    const real_t x437 = 120 * u[8];
    const real_t x438 = u[8] * x42;
    const real_t x439 = x387 * x43;
    const real_t x440 = u[9] * x36;
    const real_t x441 = 20 * u[9];
    const real_t x442 = x20 * x441;
    const real_t x443 = 18 * u[8];
    const real_t x444 = x384 * x81;
    const real_t x445 = 4 * x50;
    const real_t x446 = x380 * x67;
    const real_t x447 = x29 * x384;
    const real_t x448 = x105 * x441;
    const real_t x449 = x334 * x73;
    const real_t x450 = u[7] * x81;
    const real_t x451 = x2 * x368;
    const real_t x452 = x25 * x406 + x329 * x43 - x364 * x43 - x384 * x49 - x406 * x73 + 32 * x451;
    const real_t x453 = 8 * u[9];
    const real_t x454 = 2 * u[9];
    const real_t x455 = x12 * x454;
    const real_t x456 = x15 * x454;
    const real_t x457 = 2 * u[10];
    const real_t x458 = x12 * x457;
    const real_t x459 = x15 * x457;
    const real_t x460 = 2 * x2;
    const real_t x461 = x460 * x68;
    const real_t x462 = x23 * x457;
    const real_t x463 = x460 * x61;
    const real_t x464 = 2 * u[11];
    const real_t x465 = x23 * x464;
    const real_t x466 = x460 * x59;
    const real_t x467 = x23 * x454;
    const real_t x468 = 12 * x399;
    const real_t x469 = 12 * x375;
    const real_t x470 = 40 * u[6];
    const real_t x471 = u[7] * x73;
    const real_t x472 = u[10] * x74;
    const real_t x473 = x115 * x329;
    const real_t x474 = u[8] * x25;
    const real_t x475 = 12 * u[6];
    const real_t x476 = x475 * x50;
    const real_t x477 = 12 * u[10];
    const real_t x478 = x262 * x42;
    const real_t x479 = x20 * x262;
    const real_t x480 = x261 * x5;
    const real_t x481 = u[9] * x480;
    const real_t x482 = 8 * u[10];
    const real_t x483 = 8 * u[6];
    const real_t x484 = 8 * u[11];
    const real_t x485 = 5 * u[8];
    const real_t x486 = 4 * u[8];
    const real_t x487 = u[6] * x5;
    const real_t x488 = x1 * x487;
    const real_t x489 = 4 * u[7];
    const real_t x490 = x29 * x489;
    const real_t x491 = x1 * x347;
    const real_t x492 = u[6] * x36;
    const real_t x493 = x105 * x475;
    const real_t x494 = u[10] * x480;
    const real_t x495 = x262 * x67;
    const real_t x496 = x115 * x364;
    const real_t x497 = u[11] * x74;
    const real_t x498 = 96 * u[10];
    const real_t x499 = -u[9] * x125 + x108 * x247 + x115 * x380 - x247 * x248 + x247 * x49 +
                        x285 * x67 + 16 * x317 - x327 * x42 - x327 * x67 + x42 * x475 + x489 * x64;
    const real_t x500 = -x10 * x457 + x10 * x464 + x393 - x457 * x8 + x461 + x462 - x463 +
                        x464 * x8 - x465 - x466 - x467;
    const real_t x501 = x313 * x4;
    const real_t x502 = 96 * u[11];
    const real_t x503 = x384 * x73;
    const real_t x504 = x475 * x68;
    const real_t x505 = x262 * x61;
    const real_t x506 = 4 * u[6];
    const real_t x507 = x282 * x51;
    const real_t x508 = x282 * x97;
    const real_t x509 = x486 * x59;
    const real_t x510 = x241 * x282;
    const real_t x511 = 8 * u[8];
    const real_t x512 = 12 * u[11];
    const real_t x513 = x329 * x68;
    const real_t x514 = x329 * x61;
    const real_t x515 = x325 * x68;
    const real_t x516 = u[2] * x258;
    const real_t x517 = u[10] * x516 - u[11] * x516 + x263 * x311 - x285 * x36 + x285 * x61 -
                        x285 * x68 + x472 + x473 - 4 * x474 - x496 - x497;
    const real_t x518 = 20 * u[11];
    const real_t x519 = 5 * u[6];
    element_vector[0] =
        x149 * (convonoff * dt * x6 *
                    (u[0] * u[7] * x107 + u[0] * x44 + 72 * u[0] * (x25 - x29) - u[1] * x74 -
                     u[3] * x75 + u[4] * x75 - u[4] * x76 + u[5] * x76 + u[7] * x110 - x100 - x101 +
                     x102 + x104 + x105 * x106 + x108 * x98 + x109 * x20 + x111 + x113 + x114 +
                     x115 * x121 + x116 + x117 * x25 + x119 + 32 * x120 + x122 * x49 + x130 + x140 +
                     x142 + x147 - x25 * x58 + x29 * x60 - x29 * x77 + 80 * x3 * x54 + x40 * x41 +
                     x42 * x71 + 18 * x45 + x46 * x47 - x50 * x66 - x51 * x57 + 78 * x53 +
                     80 * x56 + x58 * x64 - x59 * x60 + x59 * x65 + x61 * x65 - x63 - x64 * x65 -
                     x65 * x67 + x65 * x73 - x68 * x69 - 32 * x70 - x72 - x80 - x83 - x85 - x87 -
                     x89 - x90 - x92 * x98 - x93 - 11 * x94 - 11 * x95 - 9 * x96 - x97 * x98) -
                x39 * (-x0 * x22 - x0 * x23 + x11 + 3 * x13 + 3 * x14 + 3 * x16 + 3 * x17 - x19 -
                       x21 + x27 + x31 + x38 + x9) -
                x7 * (u[1] + u[2] - x0 + x1));
    element_vector[1] =
        x149 *
        (-x192 * (-u[0] * x74 + u[1] * x157 + u[3] * x160 - u[4] * x160 + u[4] * x165 -
                  u[5] * x165 - u[6] * x110 - u[6] * x135 + x100 + x105 * x109 + x106 * x20 +
                  x108 * x112 - x108 * x137 + x108 * x172 - x111 + x112 * x49 - x113 + 48 * x120 +
                  x121 * x42 + x130 + x132 * x155 + x133 + x134 - x137 * x49 - x138 - x139 +
                  x155 * x41 + x156 * x40 + 78 * x158 - x159 * x25 + x159 * x43 - 72 * x161 - x162 -
                  48 * x163 + x164 * x59 - x164 * x67 - x166 * x81 - x167 - x168 * x78 +
                  x168 * x97 - x170 - x171 - x172 * x92 - x173 + x174 * (u[4] - u[5]) + x175 +
                  x176 + x177 + x178 + 18 * x179 + x180 + x181 * x68 + x181 * x73 + x186 + x190 +
                  x191 + x25 * x65 - x57 * x97 + x60 * (-x59 + x73 + x81) - x61 * x69 + x85 + x87 +
                  x93 - 18 * x94 - 9 * x95 + 18 * x96) -
         x39 * (3 * x11 + x14 + x150 + x154 + x17 + x27 + 3 * x9) -
         x7 * (u[0] - 6 * u[1] + u[2] + x28));
    element_vector[2] =
        x149 *
        (convonoff * dt * x6 *
             (u[2] * x125 + u[2] * x196 + u[2] * x212 - u[3] * x201 - u[4] * x199 + u[4] * x201 +
              u[5] * x199 - u[6] * x99 - u[7] * x99 - x104 + x105 * x203 - x108 * x122 +
              x108 * x168 + x112 * x51 + x112 * x97 - x116 + x124 * x3 + x140 + x144 - x146 +
              x155 * x156 - 48 * x161 - 72 * x163 - x164 * x61 + x164 * x64 - x166 * x43 -
              x168 * x92 + x170 - x172 * x78 + x172 * x97 + x173 - x175 - x176 + x183 + x184 +
              x185 + x190 + 9 * x195 + 78 * x197 - x198 * x29 + x198 * x81 - x20 * x203 + x200 -
              x202 - x204 - 11 * x205 + x206 * (-u[3] + u[4]) + x207 + x208 + x209 + x210 +
              32 * x211 + x213 + x29 * x65 - x50 * x71 - x51 * x91 + x58 * (x115 + x43 - x64) -
              x67 * x71 + 48 * x70 + x80 + x89 - x91 * x97 + 9 * x94 + 18 * x95 + 11 * x96) -
         x39 * (x154 + x193 + x194 + x31 + 3 * x32 + 3 * x33) -
         x7 * (u[0] + u[1] - 6 * u[2] + x24));
    element_vector[3] =
        x268 *
        (x192 * (u[3] * x233 - u[4] * x233 - x1 * x241 + x1 * x51 + x1 * x92 + x101 - x102 - x103 +
                 x105 * x238 - x108 * x236 + x108 * x237 - x114 - x119 - 40 * x120 + x124 * x55 +
                 x126 - x129 + x142 + 8 * x163 + x164 * x25 - x166 * x73 - x177 - 4 * x179 +
                 32 * x182 + x191 + 3 * x197 - x20 * x238 + x202 + x204 + 8 * x205 - x207 - x210 +
                 x214 * x49 + x225 * x48 + x225 * x5 + 96 * x226 - x227 + x228 * x59 - x228 * x67 -
                 x229 - x230 - x232 + x234 * x78 - x234 * x81 - x234 * x97 - x235 * x49 -
                 x237 * x92 - x240 - 4 * x242 - x243 + x244 + x245 + x246 * x29 + x246 * x42 +
                 x247 * x34 + x249 + x250 + x251 * x29 + x252 + x253 + x255 + x256 + x257 * x67 +
                 x267 - x28 * x51 - x28 * x92 + x49 * x91 + 16 * x56 - x57 * x67 + x72 + x73 * x77 +
                 x83 - 24 * x86 + 4 * x94 + 5 * x95 - 8 * x96) -
         x39 * (x10 * x215 - x11 - x14 + x151 + x152 - x17 + x215 * x8 + x224 - x9) +
         x7 * (-u[2] + x1 + x214 + x28));
    element_vector[4] =
        x268 *
        (x192 * (u[0] * (u[2] * x5 - x20) + 40 * u[2] * x42 + 12 * u[3] * x49 + x105 * x246 +
                 x108 * x214 - x108 * x235 + x108 * x239 - x108 * x91 - x121 * x43 + x122 * x68 +
                 x137 * x97 + 32 * x161 + x162 - 8 * x179 + x186 + x187 - x189 - x20 * x281 + x200 -
                 x209 - 40 * x211 - x214 * x51 + 48 * x226 + x227 + x229 + x230 - x234 * x25 +
                 x234 * x29 + x235 * x97 + x236 * x49 - x236 * x97 + x239 * x59 + x239 * x61 -
                 x239 * x64 - x239 * x67 + x239 * x78 - x239 * x92 - x239 * x97 + x243 - x245 -
                 x249 - x250 - x252 - x255 - x256 + x257 * x42 - x257 * x68 + x267 + x273 * x55 +
                 x274 * x46 + x275 * x3 + x275 * x55 + x276 * x42 - x276 * x68 - x277 - x278 * x49 +
                 x278 * x51 - x279 - x280 * x51 + x283 + x284 + x286 + x287 + x289 - x42 * x57 +
                 5 * x45 + 3 * x53 + 8 * x94 + 8 * x95 - 4 * x96) +
         x39 * (x150 + x224 - x269 - x270 + x271 + x272) + x7 * (-u[0] + x235 + x24 + x28));
    element_vector[5] =
        x268 *
        (-x192 *
             (-x1 * x248 + x1 * x49 + x1 * x78 - x105 * x281 - x108 * x234 + x115 * x117 -
              x122 * x61 - x127 - 24 * x136 + x137 * x51 + x147 + 3 * x158 + 8 * x161 + x164 * x29 +
              x167 + x171 - x178 + 5 * x179 + x18 * x282 - x180 + 12 * x195 + x20 * x246 -
              4 * x205 - x208 + 4 * x211 + x213 - x214 * x97 + x225 * x55 - x228 * x61 +
              x228 * x64 + x232 - x234 * x43 + x234 * x92 - x235 * x51 + x236 * x51 - x237 * x78 +
              x237 * x97 - x24 * x49 - x24 * x78 + x240 + 8 * x242 - x244 + x25 * x281 +
              x25 * x288 - x253 + x257 * x61 + x259 + x260 - x265 - x266 + x273 * x5 + x277 -
              x278 * x97 + x279 + x280 * x97 - x283 - x284 - x286 - x287 + x289 + 96 * x3 * x46 -
              x50 * x79 + x63 - 40 * x70 + x90 - 5 * x94 - 4 * x95 + 5 * x96) +
         x39 * (-x12 * x219 - x15 * x219 + x153 + x194 + x217 + x218 + x220 + x221 - x222 - x223 +
                x269 + x270 - x271 - x272 + x38) +
         x7 * (-u[1] + x1 + x236 + x24));
    element_vector[6] =
        x149 *
        (-x192 * (u[1] * u[6] * x41 - u[6] * x212 - u[6] * x346 + u[6] * x348 + u[6] * x44 +
                  72 * u[6] * (u[9] * x2 - x61) + 78 * u[6] * (u[6] * x315 + x105 + x49 - x50) +
                  u[7] * x74 - x105 * x320 - x105 * x331 + x105 * x363 - x106 * x347 + x107 * x311 +
                  x108 * x327 - x109 * x308 + x115 * x321 + x168 * x312 - x25 * x326 + x29 * x326 +
                  x313 * x314 + 80 * x317 + 80 * x319 - x321 * x67 + x322 * x61 - x322 * x73 -
                  x324 - x325 * x49 + x326 * x73 + x327 * x43 - x328 - x329 * x81 - x329 * x97 -
                  32 * x330 + x331 * x50 - 24 * x332 - x333 - x334 * x61 - x336 - x338 - x339 -
                  x340 - x342 - x344 - 11 * x345 - x349 - 9 * x350 - x351 + x352 + x354 +
                  x355 * x51 + 11 * x356 + x357 * x97 + x358 + x359 + x360 + x361 + x362 * x67 +
                  x365 + x366 * x51 + 96 * x367 + x374 + x386 + x388 + x392) -
         x39 * (-x23 * x291 + x292 + x293 + 3 * x294 + 3 * x295 + 3 * x296 + 3 * x297 - x298 -
                x300 - 6 * x301 + x303 + x304 + x310) -
         x7 * (u[7] + u[8] + x290 - x291));
    element_vector[7] =
        x149 * (convonoff * dt * x6 *
                    (-u[10] * x402 + u[6] * x74 + u[7] * x157 - u[8] * x125 + u[9] * x402 +
                     4 * x105 * (u[10] - u[11]) - x106 * x308 + x107 * x399 - x109 * x347 -
                     x20 * x320 + x20 * x331 - x20 * x363 + x20 * x409 - x248 * x409 - x25 * x405 -
                     x29 * x329 - x308 * x98 + x311 * x400 + x322 * (x59 + x68 - x73) - x329 * x51 -
                     16 * x332 + x339 + x342 + x344 - 18 * x345 + x349 + 18 * x350 + x355 * x97 +
                     18 * x356 - x357 * x50 + x357 * x51 - x358 - x359 + x36 * x378 - x36 * x384 +
                     x36 * x415 + x366 * x49 + x366 * x97 + x374 + x376 * x399 + x378 * x50 + x379 +
                     x381 - x383 - x385 + 78 * x401 + x403 * x42 - x403 * x67 - 72 * x404 -
                     x406 * x68 - x408 - x410 - x411 - x413 + x414 + x416 + x417 + x419 + x420 +
                     x421 + 32 * x422 + x423 * x73 + x423 * x81 + x427 + x431 + x432) -
                x39 * (x303 + 3 * x305 + 3 * x306 + x393 + x397 + x398) -
                x7 * (u[6] - 6 * u[7] + u[8] + x247));
    element_vector[8] =
        x149 *
        (-x192 * (-u[10] * x436 + u[11] * x436 + u[8] * x196 + u[8] * x212 + x105 * x378 -
                  x105 * x380 + x108 * x325 + x115 * x405 - x172 * x308 + x20 * x378 - x20 * x380 +
                  x20 * x415 + x203 * x308 - x203 * x347 - x248 * x415 - x25 * x327 - x29 * x405 +
                  x321 * (-x115 + x42 + x64) + x325 * x51 - x327 * x49 + x331 * x36 + x333 + x340 +
                  9 * x345 + 11 * x350 - x354 - x355 * x50 - 9 * x356 + x36 * x409 - x360 +
                  x369 * x4 + x386 + x390 - x391 + x399 * x400 - 48 * x404 - x406 * x42 -
                  96 * x407 + x411 + x413 - x414 - x419 + 48 * x422 + x424 + x425 + x426 + x431 +
                  9 * x434 + 78 * x435 - x437 * x61 + x437 * x68 - 72 * x438 - x439 - 24 * x440 -
                  x442 + x443 * x51 - x443 * x97 - x444 + x445 * (u[10] - u[9]) + x446 + x447 +
                  x448 + x449 + 32 * x450 + x452) -
         x39 * (3 * x292 + 3 * x293 + x294 + x296 + x304 + x397 + x433) -
         x7 * (u[6] + u[7] - 6 * u[8] + x282));
    element_vector[9] =
        x268 *
        (-x192 *
             (8 * u[7] * x61 - u[8] * x445 + x1 * x299 - x105 * x247 + x105 * x290 +
              96 * x2 * x318 + x2 * x468 - x20 * x483 + x234 * x308 + x238 * x308 - x238 * x347 -
              x241 * x247 + x241 * x290 + x248 * x483 - x25 * x320 - x25 * x470 + x25 * x498 +
              x28 * x299 - x288 * x299 + x315 * x468 + x316 * x369 + x328 + x329 * x49 + x338 +
              4 * x345 - 8 * x350 + x351 - x352 - x353 - x36 * x477 - x36 * x484 - x361 - x365 +
              x371 - x373 + x380 * x50 + x388 + x4 * x469 - x417 + 32 * x428 + x432 + 3 * x435 +
              8 * x438 + 12 * x440 - x441 * x51 + x441 * x97 + x442 + x444 - x447 - x448 +
              x453 * x50 + x470 * x73 - 40 * x471 - x472 - x473 - 24 * x474 - x476 - x478 - x479 -
              x481 - x482 * x50 - x483 * x68 + x485 * x51 - x485 * x97 + x486 * x61 - x486 * x68 -
              x488 - x490 + x491 + 5 * x492 + x493 + x494 + x495 + x496 + x497 + x499) +
         x39 * (-x10 * x454 + x310 + x394 + x396 + x398 - x454 * x8 - x455 - x456 + x458 + x459 -
                x461 - x462 + x463 + x465 + x466 + x467) +
         x7 * (-u[8] + x247 + x290 + x453));
    element_vector[10] =
        x268 *
        (x192 *
             (-24 * u[6] * x43 + u[6] * (-x108 + x97) + 3 * u[6] * (u[6] * x316 - x105 + x52) +
              u[7] * x206 - u[8] * x174 + x105 * x453 - x105 * x477 + x105 * x512 - x20 * x384 -
              x20 * x482 + x20 * x484 + x20 * x506 - x237 * x312 + x241 * x506 - x248 * x506 +
              x25 * x506 + x251 * x299 + x28 * x487 - x281 * x308 - x29 * x506 + 96 * x315 * x368 +
              x315 * x469 + 16 * x319 + x320 * x43 - 12 * x332 - 8 * x345 + x36 * x380 -
              x36 * x453 + x36 * x482 - 32 * x404 + 24 * x418 - x421 - 40 * x422 + x427 + x429 -
              x43 * x498 - x430 + 32 * x438 + x439 + x449 + 40 * x450 + 96 * x451 + x453 * x49 +
              x477 * x50 + x478 + x479 + x481 - x484 * x50 - x484 * x51 + x490 - 4 * x492 - x494 -
              x495 + x498 * x81 + x499 + 48 * x501 - x502 * x81 - x503 - x504 - x505 - x506 * x73 -
              x507 - x508 - x509 + x510 + x511 * x97 + x513 + x514 + x515 + x517) +
         x39 * (x433 + x455 + x456 - x458 - x459 + x500) + x7 * (-u[6] + x247 + x282 + x482));
    element_vector[11] =
        x268 *
        (x192 *
             (-24 * u[7] * x29 + x1 * x308 + x105 * x384 - x105 * x482 + x105 * x484 + x108 * x518 +
              x115 * x470 - x20 * x453 - x20 * x477 + x20 * x512 + x20 * x519 - x239 * x308 +
              x241 * x483 - x246 * x308 - x248 * x282 + x248 * x290 - x248 * x519 - x282 * x50 -
              x29 * x470 + x29 * x498 - x29 * x502 + x290 * x50 + 16 * x315 * x318 + x316 * x468 +
              x324 + x327 * x51 - 40 * x330 + x336 - 5 * x345 + 5 * x350 + 5 * x356 - x36 * x483 -
              x372 + x392 + 3 * x401 + 8 * x404 + x408 + x410 - x416 - x42 * x483 - x42 * x489 -
              x420 + 12 * x434 - x446 + x452 + 4 * x471 + x476 + x486 * x97 + x488 + x489 * x67 -
              x49 * x518 - x491 - x493 + x50 * x511 + 96 * x501 + x503 + x504 + x505 + x507 + x508 +
              x509 - x51 * x511 - x510 + x511 * x67 - x513 - x514 - x515 + x517) -
         x39 * (x12 * x464 + x15 * x464 - x292 - x293 - x294 - x296 + x301 + x395 + x500) +
         x7 * (-u[7] + x282 + x290 + x484));
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
    printf("tri6_naviers_stokes.c: tri6_explict_momentum_tentative\t%g seconds\n", tock - tick);
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
#pragma omp for nowait
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
    printf("tri6_naviers_stokes.c: tri6_explict_momentum_tentative\t%g seconds\n", tock - tick);
}

static SFEM_INLINE int linear_search(const idx_t target, const idx_t *const arr, const int size) {
    int i;
    for (i = 0; i < size - SFEM_VECTOR_SIZE; i += SFEM_VECTOR_SIZE) {
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
                                   int *ks) {
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
#pragma omp for nowait
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
    printf("tri3_laplacian.c: tri3_laplacian_assemble_hessian\t%g seconds\n", tock - tick);
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
#pragma omp for nowait
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
                convonoff,
                //  buffers
                element_vel,
                element_vector);

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
    printf("tri6_naviers_stokes.c: tri6_explict_momentum_tentative\t%g seconds\n", tock - tick);
}