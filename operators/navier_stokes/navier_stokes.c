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

// Taylor hood Triangle P2/P1

static SFEM_INLINE void tri6_nse_taylor_hood_correction_rhs_kernel(
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t dt,
    const real_t *const SFEM_RESTRICT u,
    const real_t *const SFEM_RESTRICT p,
    real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = (1.0 / 360.0) * u[1];
    const real_t x1 = (1.0 / 360.0) * u[2];
    const real_t x2 = (-px0 + px1) * (-py0 + py2) - (-px0 + px2) * (-py0 + py1);
    const real_t x3 = (1.0 / 360.0) * u[0];
    const real_t x4 = (2.0 / 45.0) * u[4];
    const real_t x5 = (1.0 / 6.0) * p[0];
    const real_t x6 = -dt * ((1.0 / 6.0) * p[1] - x5);
    const real_t x7 = (2.0 / 45.0) * u[5] + x6;
    const real_t x8 = (2.0 / 45.0) * u[3];
    const real_t x9 = (1.0 / 360.0) * u[7];
    const real_t x10 = (1.0 / 360.0) * u[8];
    const real_t x11 = (1.0 / 360.0) * u[6];
    const real_t x12 = (2.0 / 45.0) * u[10];
    const real_t x13 = -dt * ((1.0 / 6.0) * p[2] - x5);
    const real_t x14 = (2.0 / 45.0) * u[11] + x13;
    const real_t x15 = (2.0 / 45.0) * u[9];
    element_vector[0] = x2 * ((1.0 / 60.0) * u[0] - 1.0 / 90.0 * u[4] - x0 - x1);
    element_vector[1] = x2 * ((1.0 / 60.0) * u[1] - 1.0 / 90.0 * u[5] - x1 - x3);
    element_vector[2] = x2 * ((1.0 / 60.0) * u[2] - 1.0 / 90.0 * u[3] - x0 - x3);
    element_vector[3] = x2 * (-1.0 / 90.0 * u[2] + (4.0 / 45.0) * u[3] + x4 + x7);
    element_vector[4] = x2 * (-1.0 / 90.0 * u[0] + (4.0 / 45.0) * u[4] + x7 + x8);
    element_vector[5] = x2 * (-1.0 / 90.0 * u[1] + (4.0 / 45.0) * u[5] + x4 + x6 + x8);
    element_vector[6] = x2 * (-1.0 / 90.0 * u[10] + (1.0 / 60.0) * u[6] - x10 - x9);
    element_vector[7] = x2 * (-1.0 / 90.0 * u[11] + (1.0 / 60.0) * u[7] - x10 - x11);
    element_vector[8] = x2 * ((1.0 / 60.0) * u[8] - 1.0 / 90.0 * u[9] - x11 - x9);
    element_vector[9] = x2 * (-1.0 / 90.0 * u[8] + (4.0 / 45.0) * u[9] + x12 + x14);
    element_vector[10] = x2 * ((4.0 / 45.0) * u[10] - 1.0 / 90.0 * u[6] + x14 + x15);
    element_vector[11] = x2 * ((4.0 / 45.0) * u[11] - 1.0 / 90.0 * u[7] + x12 + x13 + x15);
}

static SFEM_INLINE void tri3_laplacian_kernel(const real_t px0,
                                                 const real_t px1,
                                                 const real_t px2,
                                                 const real_t py0,
                                                 const real_t py1,
                                                 const real_t py2,
                                                 real_t *const SFEM_RESTRICT element_matrix) {
    const real_t x0 = (-px0 + px1) * (-py0 + py2);
    const real_t x1 = -px0 + px2;
    const real_t x2 = -py0 + py1;
    const real_t x3 = (1.0 / 2.0) * x0 - 1.0 / 2.0 * x1 * x2;
    const real_t x4 = -x3;
    element_matrix[0] = x0 - x1 * x2;
    element_matrix[1] = x4;
    element_matrix[2] = x4;
    element_matrix[3] = x4;
    element_matrix[4] = x3;
    element_matrix[5] = 0;
    element_matrix[6] = x4;
    element_matrix[7] = 0;
    element_matrix[8] = x3;
}

static SFEM_INLINE void tri6_momentum_rhs_kernel(const real_t px0,
                                                const real_t px1,
                                                const real_t px2,
                                                const real_t py0,
                                                const real_t py1,
                                                const real_t py2,
                                                real_t *const SFEM_RESTRICT u,
                                                real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = (1.0 / 360.0) * u[1];
    const real_t x1 = (1.0 / 360.0) * u[2];
    const real_t x2 = -px0 + px1;
    const real_t x3 = -py0 + py2;
    const real_t x4 = px0 - px2;
    const real_t x5 = py0 - py1;
    const real_t x6 = x2 * x3 - x4 * x5;
    const real_t x7 = pow(u[3], 2);
    const real_t x8 = 1.0 / x6;
    const real_t x9 = pow(u[5], 2);
    const real_t x10 = pow(u[1], 2);
    const real_t x11 = x3 * x8;
    const real_t x12 = (1.0 / 280.0) * x11;
    const real_t x13 = x10 * x12;
    const real_t x14 = pow(u[2], 2);
    const real_t x15 = x5 * x8;
    const real_t x16 = (1.0 / 280.0) * x15;
    const real_t x17 = x14 * x16;
    const real_t x18 = pow(u[0], 2);
    const real_t x19 = (13.0 / 420.0) * x18;
    const real_t x20 = (1.0 / 35.0) * u[0];
    const real_t x21 = u[3] * x15;
    const real_t x22 = u[5] * x11;
    const real_t x23 = (1.0 / 105.0) * u[6];
    const real_t x24 = x2 * x8;
    const real_t x25 = u[3] * x24;
    const real_t x26 = x4 * x8;
    const real_t x27 = u[5] * x26;
    const real_t x28 = (1.0 / 126.0) * u[10];
    const real_t x29 = u[1] * x26;
    const real_t x30 = u[2] * x24;
    const real_t x31 = (1.0 / 126.0) * u[1];
    const real_t x32 = (1.0 / 126.0) * u[2];
    const real_t x33 = (1.0 / 140.0) * u[6];
    const real_t x34 = u[0] * x26;
    const real_t x35 = (1.0 / 210.0) * u[10];
    const real_t x36 = u[0] * x24;
    const real_t x37 = (1.0 / 280.0) * x26;
    const real_t x38 = u[7] * x37;
    const real_t x39 = u[1] * x38;
    const real_t x40 = (1.0 / 280.0) * x24;
    const real_t x41 = u[8] * x40;
    const real_t x42 = u[2] * x41;
    const real_t x43 = (1.0 / 630.0) * u[8];
    const real_t x44 = (1.0 / 630.0) * u[7];
    const real_t x45 = (2.0 / 105.0) * u[11];
    const real_t x46 = (2.0 / 105.0) * u[9];
    const real_t x47 = u[11] * x24;
    const real_t x48 = (2.0 / 315.0) * u[4];
    const real_t x49 = x47 * x48;
    const real_t x50 = (2.0 / 315.0) * u[1];
    const real_t x51 = u[4] * x50;
    const real_t x52 = u[2] * x48;
    const real_t x53 = u[4] * x26;
    const real_t x54 = (2.0 / 315.0) * u[9];
    const real_t x55 = x53 * x54;
    const real_t x56 = (4.0 / 315.0) * u[2];
    const real_t x57 = (4.0 / 315.0) * x26;
    const real_t x58 = u[11] * x57;
    const real_t x59 = (4.0 / 315.0) * x11;
    const real_t x60 = u[1] * u[3];
    const real_t x61 = u[1] * u[9];
    const real_t x62 = u[5] * x15;
    const real_t x63 = u[4] * x24;
    const real_t x64 = (4.0 / 315.0) * u[9];
    const real_t x65 = (13.0 / 420.0) * u[6];
    const real_t x66 = x11 * x7;
    const real_t x67 = pow(u[4], 2);
    const real_t x68 = (2.0 / 105.0) * x67;
    const real_t x69 = u[3] * x26;
    const real_t x70 = (2.0 / 63.0) * x69;
    const real_t x71 = (2.0 / 105.0) * u[10];
    const real_t x72 = u[3] * x11;
    const real_t x73 = (1.0 / 126.0) * u[8];
    const real_t x74 = u[0] * u[1];
    const real_t x75 = -u[9] * x70 + x12 * x74 + x15 * x68 - x22 * x48 - x25 * x71 + x32 * x72 +
                       x63 * x71 - 2.0 / 63.0 * x66 + x69 * x73;
    const real_t x76 = x15 * x9;
    const real_t x77 = u[5] * x47;
    const real_t x78 = u[5] * x24;
    const real_t x79 = (1.0 / 126.0) * u[7];
    const real_t x80 = u[0] * u[2];
    const real_t x81 = x11 * x68 + x16 * x80 - x21 * x48 - x27 * x71 + x31 * x62 + x53 * x71 -
                       2.0 / 63.0 * x76 - 2.0 / 63.0 * x77 + x78 * x79;
    const real_t x82 = x11 * x52;
    const real_t x83 = (2.0 / 315.0) * u[8];
    const real_t x84 = x53 * x83;
    const real_t x85 = -8.0 / 315.0 * u[11] * u[3] * x4 * x8 - 2.0 / 315.0 * u[2] * u[5] * x3 * x8 -
                       2.0 / 315.0 * u[5] * u[8] * x4 * x8 + x82 + x84;
    const real_t x86 = x15 * x51;
    const real_t x87 = (2.0 / 315.0) * u[7];
    const real_t x88 = x63 * x87;
    const real_t x89 = -2.0 / 315.0 * u[1] * u[3] * x5 * x8 - 2.0 / 315.0 * u[3] * u[7] * x2 * x8 -
                       8.0 / 315.0 * u[5] * u[9] * x2 * x8 + x86 + x88;
    const real_t x90 = u[11] * x26;
    const real_t x91 = x50 * x90;
    const real_t x92 = x30 * x54;
    const real_t x93 = x91 + x92;
    const real_t x94 = (1.0 / 360.0) * u[0];
    const real_t x95 = x15 * x7;
    const real_t x96 = (2.0 / 105.0) * x95;
    const real_t x97 = x11 * x9;
    const real_t x98 = (2.0 / 105.0) * x97;
    const real_t x99 = x10 * x11;
    const real_t x100 = (11.0 / 2520.0) * x11;
    const real_t x101 = (11.0 / 2520.0) * u[8];
    const real_t x102 = u[10] * x69;
    const real_t x103 = (8.0 / 315.0) * x102;
    const real_t x104 = u[7] * x69;
    const real_t x105 = x24 * x56;
    const real_t x106 = u[10] * x57;
    const real_t x107 = (2.0 / 63.0) * u[4];
    const real_t x108 = x45 * x53;
    const real_t x109 = x11 * x60;
    const real_t x110 = u[4] * x15;
    const real_t x111 = (2.0 / 105.0) * u[2];
    const real_t x112 = x25 * x46;
    const real_t x113 = (2.0 / 315.0) * u[3];
    const real_t x114 = (2.0 / 315.0) * u[6];
    const real_t x115 = x114 * x63;
    const real_t x116 = x63 * x83;
    const real_t x117 = x15 * x60;
    const real_t x118 = (1.0 / 21.0) * u[7];
    const real_t x119 = (1.0 / 105.0) * u[7];
    const real_t x120 = (1.0 / 126.0) * u[6];
    const real_t x121 = u[1] * u[2];
    const real_t x122 = (1.0 / 140.0) * x15;
    const real_t x123 = (1.0 / 210.0) * u[1];
    const real_t x124 = u[6] * x37;
    const real_t x125 = (1.0 / 630.0) * u[6];
    const real_t x126 = u[1] * x110;
    const real_t x127 = u[4] * x11;
    const real_t x128 = u[1] * x127;
    const real_t x129 = (1.0 / 126.0) * u[11];
    const real_t x130 = (1.0 / 126.0) * u[0];
    const real_t x131 = (1.0 / 140.0) * u[7];
    const real_t x132 = (2.0 / 105.0) * u[0];
    const real_t x133 = x27 * x45;
    const real_t x134 = x26 * x61;
    const real_t x135 = x46 * x63;
    const real_t x136 = (2.0 / 315.0) * x22;
    const real_t x137 = u[10] * x78;
    const real_t x138 = u[0] * x59;
    const real_t x139 = (11.0 / 2520.0) * u[6];
    const real_t x140 = u[7] * x29;
    const real_t x141 = (2.0 / 315.0) * u[10];
    const real_t x142 = x141 * x34 + x141 * x36;
    const real_t x143 = x142 - x92;
    const real_t x144 = u[0] * u[6] * x40 + u[0] * x124 + x113 * x47 + x12 * x18 + x16 * x18 +
                        x27 * x54 - x49 - x55;
    const real_t x145 = x15 * x67;
    const real_t x146 = x14 * x15;
    const real_t x147 = (11.0 / 2520.0) * x15;
    const real_t x148 = (11.0 / 2520.0) * u[7];
    const real_t x149 = (8.0 / 315.0) * x137;
    const real_t x150 = u[8] * x78;
    const real_t x151 = (4.0 / 315.0) * u[10];
    const real_t x152 = x114 * x53;
    const real_t x153 = x53 * x87;
    const real_t x154 = (1.0 / 21.0) * u[2];
    const real_t x155 = (1.0 / 21.0) * u[8];
    const real_t x156 = (1.0 / 105.0) * u[8];
    const real_t x157 = (1.0 / 140.0) * x11;
    const real_t x158 = (1.0 / 140.0) * u[8];
    const real_t x159 = (1.0 / 210.0) * x21;
    const real_t x160 = u[2] * x40;
    const real_t x161 = u[2] * x110;
    const real_t x162 = (1.0 / 126.0) * u[9];
    const real_t x163 = u[9] * x30;
    const real_t x164 = u[0] * x21;
    const real_t x165 = u[11] * x36;
    const real_t x166 = u[0] * x62;
    const real_t x167 = u[8] * x30;
    const real_t x168 = x142 - x91;
    const real_t x169 = (2.0 / 45.0) * u[4];
    const real_t x170 = (2.0 / 45.0) * u[5];
    const real_t x171 = u[2] * x11;
    const real_t x172 = u[11] * x34;
    const real_t x173 = u[0] * x72;
    const real_t x174 = (2.0 / 105.0) * x117;
    const real_t x175 = (2.0 / 105.0) * u[7] * x25;
    const real_t x176 = (2.0 / 105.0) * x63;
    const real_t x177 = x15 * x50;
    const real_t x178 = (4.0 / 63.0) * u[9];
    const real_t x179 = u[1] * x57;
    const real_t x180 = (4.0 / 315.0) * u[6];
    const real_t x181 = (8.0 / 105.0) * u[10];
    const real_t x182 = u[0] * x22;
    const real_t x183 = (8.0 / 315.0) * u[3];
    const real_t x184 = (16.0 / 105.0) * u[9];
    const real_t x185 = (16.0 / 315.0) * x27;
    const real_t x186 = u[5] * x90;
    const real_t x187 = (2.0 / 105.0) * x18;
    const real_t x188 = (2.0 / 105.0) * u[6];
    const real_t x189 = (16.0 / 315.0) * x47;
    const real_t x190 = u[3] * x189;
    const real_t x191 = u[9] * x185;
    const real_t x192 = -16.0 / 315.0 * u[11] * u[4] * x2 * x8 -
                        16.0 / 315.0 * u[4] * u[9] * x4 * x8 + x11 * x187 + x15 * x187 +
                        x188 * x34 + x188 * x36 + x190 + x191;
    const real_t x193 = (16.0 / 315.0) * x62;
    const real_t x194 = (8.0 / 105.0) * u[4];
    const real_t x195 = (1.0 / 630.0) * x15;
    const real_t x196 = (2.0 / 315.0) * u[2];
    const real_t x197 = -u[3] * x193 - u[4] * x193 + x110 * x132 + x111 * x21 +
                        (2.0 / 315.0) * x166 - x194 * x21 - x195 * x80 + x196 * x62 + x50 * x62 -
                        8.0 / 315.0 * x76 - 8.0 / 315.0 * x77 + x78 * x87;
    const real_t x198 = (2.0 / 45.0) * u[3];
    const real_t x199 = (1.0 / 210.0) * x18;
    const real_t x200 = (16.0 / 105.0) * u[10];
    const real_t x201 = (8.0 / 105.0) * u[9];
    const real_t x202 = u[1] * x15;
    const real_t x203 = u[7] * x57;
    const real_t x204 = (4.0 / 315.0) * u[8];
    const real_t x205 = x111 * x22;
    const real_t x206 = (2.0 / 105.0) * u[8];
    const real_t x207 = x206 * x27;
    const real_t x208 = (1.0 / 210.0) * u[6];
    const real_t x209 = (2.0 / 63.0) * u[2];
    const real_t x210 = (4.0 / 105.0) * u[10];
    const real_t x211 = u[6] * x57;
    const real_t x212 = (16.0 / 315.0) * x22;
    const real_t x213 = (1.0 / 630.0) * x11;
    const real_t x214 = (2.0 / 105.0) * u[1];
    const real_t x215 = -u[3] * x212 - 16.0 / 315.0 * u[4] * x72 - 8.0 / 315.0 * u[9] * x69 +
                        (2.0 / 315.0) * x109 + x127 * x132 + (2.0 / 315.0) * x173 - x194 * x22 +
                        x196 * x72 - x213 * x74 + x214 * x22 - 8.0 / 315.0 * x66 + x69 * x83;
    const real_t x216 = x15 * x31;
    const real_t x217 = (2.0 / 63.0) * u[9];
    const real_t x218 = (2.0 / 105.0) * u[7];
    const real_t x219 = x26 * x50;
    const real_t x220 = u[1] * x59;
    const real_t x221 = (16.0 / 315.0) * x25;
    const real_t x222 = (1.0 / 360.0) * u[7];
    const real_t x223 = (1.0 / 360.0) * u[8];
    const real_t x224 = pow(u[11], 2);
    const real_t x225 = pow(u[9], 2);
    const real_t x226 = pow(u[7], 2);
    const real_t x227 = x226 * x37;
    const real_t x228 = pow(u[8], 2);
    const real_t x229 = x228 * x40;
    const real_t x230 = pow(u[6], 2);
    const real_t x231 = (13.0 / 420.0) * x230;
    const real_t x232 = (1.0 / 35.0) * u[6];
    const real_t x233 = u[9] * x24;
    const real_t x234 = u[0] * x11;
    const real_t x235 = u[0] * x15;
    const real_t x236 = u[8] * x122;
    const real_t x237 = u[1] * u[7] * x12;
    const real_t x238 = u[2] * u[8] * x16;
    const real_t x239 = x141 * x72;
    const real_t x240 = x141 * x62;
    const real_t x241 = u[10] * x87;
    const real_t x242 = u[10] * x83;
    const real_t x243 = u[5] * x59;
    const real_t x244 = u[3] * x59;
    const real_t x245 = x225 * x26;
    const real_t x246 = pow(u[10], 2);
    const real_t x247 = (2.0 / 105.0) * x246;
    const real_t x248 = u[9] * x11;
    const real_t x249 = x26 * x73;
    const real_t x250 = u[6] * x38 + u[9] * x249 - x110 * x46 + x110 * x71 - x141 * x90 -
                        x217 * x72 + x24 * x247 - 2.0 / 63.0 * x245 + x248 * x32;
    const real_t x251 = x224 * x24;
    const real_t x252 = (2.0 / 63.0) * x62;
    const real_t x253 = -u[10] * x24 * x54 + u[11] * x216 - u[11] * x252 + u[6] * x41 - x127 * x45 +
                        x127 * x71 + x247 * x26 - 2.0 / 63.0 * x251 + x47 * x79;
    const real_t x254 = x141 * x171;
    const real_t x255 = x242 * x26;
    const real_t x256 = -2.0 / 315.0 * u[11] * u[2] * x3 * x8 -
                        2.0 / 315.0 * u[11] * u[8] * x4 * x8 - 8.0 / 315.0 * u[5] * u[9] * x3 * x8 +
                        x254 + x255;
    const real_t x257 = u[10] * x177;
    const real_t x258 = x24 * x241;
    const real_t x259 = -8.0 / 315.0 * u[11] * u[3] * x5 * x8 -
                        2.0 / 315.0 * u[1] * u[9] * x5 * x8 - 2.0 / 315.0 * u[7] * u[9] * x2 * x8 +
                        x257 + x258;
    const real_t x260 = x21 * x83;
    const real_t x261 = x22 * x87;
    const real_t x262 = x260 + x261;
    const real_t x263 = (1.0 / 360.0) * u[6];
    const real_t x264 = x225 * x24;
    const real_t x265 = (2.0 / 105.0) * x264;
    const real_t x266 = x224 * x26;
    const real_t x267 = (2.0 / 105.0) * x266;
    const real_t x268 = x226 * x26;
    const real_t x269 = u[2] * u[6];
    const real_t x270 = u[6] * x26;
    const real_t x271 = (8.0 / 315.0) * u[9];
    const real_t x272 = x127 * x271;
    const real_t x273 = x11 * x61;
    const real_t x274 = u[4] * x59;
    const real_t x275 = u[10] * x26;
    const real_t x276 = x22 * x71;
    const real_t x277 = u[8] * x24;
    const real_t x278 = x21 * x46;
    const real_t x279 = u[7] * x26;
    const real_t x280 = x141 * x235;
    const real_t x281 = u[2] * x15;
    const real_t x282 = x141 * x281;
    const real_t x283 = (2.0 / 315.0) * u[11];
    const real_t x284 = x15 * x61;
    const real_t x285 = u[11] * x11;
    const real_t x286 = (1.0 / 105.0) * u[1];
    const real_t x287 = (1.0 / 210.0) * u[7];
    const real_t x288 = u[0] * x12;
    const real_t x289 = u[2] * u[7];
    const real_t x290 = (1.0 / 630.0) * x234;
    const real_t x291 = (1.0 / 35.0) * u[10];
    const real_t x292 = u[10] * x11;
    const real_t x293 = u[11] * x15;
    const real_t x294 = u[1] * x11;
    const real_t x295 = u[7] * x24;
    const real_t x296 = x21 * x71;
    const real_t x297 = x22 * x45;
    const real_t x298 = u[6] * x24;
    const real_t x299 = u[6] * x48;
    const real_t x300 = x11 * x299 + x15 * x299;
    const real_t x301 = -x260 + x300;
    const real_t x302 = u[0] * u[6] * x16 + u[6] * x288 + x230 * x37 + x230 * x40 - x239 - x240 +
                        x283 * x72 + x54 * x62;
    const real_t x303 = x24 * x246;
    const real_t x304 = x228 * x24;
    const real_t x305 = u[1] * u[6];
    const real_t x306 = (8.0 / 315.0) * u[11];
    const real_t x307 = x110 * x306;
    const real_t x308 = x141 * x234;
    const real_t x309 = x292 * x50;
    const real_t x310 = (1.0 / 105.0) * x281;
    const real_t x311 = u[8] * x26;
    const real_t x312 = u[8] * x16;
    const real_t x313 = u[0] * x195;
    const real_t x314 = -x261 + x300;
    const real_t x315 = (2.0 / 45.0) * u[10];
    const real_t x316 = (2.0 / 45.0) * u[11];
    const real_t x317 = (2.0 / 63.0) * u[6];
    const real_t x318 = (2.0 / 105.0) * x284;
    const real_t x319 = x295 * x46;
    const real_t x320 = (4.0 / 63.0) * u[6];
    const real_t x321 = (8.0 / 315.0) * u[6];
    const real_t x322 = (16.0 / 315.0) * u[11];
    const real_t x323 = (16.0 / 315.0) * u[6];
    const real_t x324 = (2.0 / 105.0) * x230;
    const real_t x325 = u[6] * x132;
    const real_t x326 = x322 * x72;
    const real_t x327 = u[9] * x193;
    const real_t x328 = -16.0 / 315.0 * u[10] * u[3] * x3 * x8 -
                        16.0 / 315.0 * u[10] * u[5] * x5 * x8 + x11 * x325 + x15 * x325 +
                        x24 * x324 + x26 * x324 + x326 + x327;
    const real_t x329 = -u[10] * x189 + u[11] * x177 - 8.0 / 315.0 * u[11] * x62 - u[9] * x189 +
                        x114 * x47 - x181 * x233 - 8.0 / 315.0 * x251 + x277 * x46 - x298 * x43 +
                        x298 * x71 + x47 * x83 + x47 * x87;
    const real_t x330 = (2.0 / 45.0) * u[9];
    const real_t x331 = (1.0 / 210.0) * x230;
    const real_t x332 = (16.0 / 105.0) * u[11];
    const real_t x333 = (8.0 / 105.0) * u[11];
    const real_t x334 = x15 * x56;
    const real_t x335 = (4.0 / 315.0) * x15;
    const real_t x336 = u[8] * x335;
    const real_t x337 = x171 * x45;
    const real_t x338 = x311 * x45;
    const real_t x339 = (1.0 / 210.0) * u[0];
    const real_t x340 = u[6] * x11;
    const real_t x341 = (4.0 / 105.0) * u[6];
    const real_t x342 = (16.0 / 315.0) * u[10];
    const real_t x343 = (16.0 / 315.0) * u[9];
    const real_t x344 = x171 * x54 - x181 * x90 - 8.0 / 315.0 * x245 - x270 * x44 + x270 * x54 +
                        x270 * x71 - x271 * x72 - x275 * x343 + x279 * x45 + x279 * x54 +
                        x311 * x54 - x343 * x90;
    element_vector[0] =
        x6 * ((1.0 / 60.0) * u[0] - 1.0 / 90.0 * u[4] - x0 - x1) -
        x6 * ((1.0 / 280.0) * u[0] * u[1] * x5 * x8 + (1.0 / 280.0) * u[0] * u[2] * x3 * x8 +
              (2.0 / 105.0) * u[0] * u[3] * x3 * x8 + (1.0 / 210.0) * u[0] * u[4] * x3 * x8 +
              (1.0 / 210.0) * u[0] * u[4] * x5 * x8 + (2.0 / 105.0) * u[0] * u[5] * x5 * x8 +
              (1.0 / 280.0) * u[0] * u[7] * x2 * x8 + (1.0 / 280.0) * u[0] * u[7] * x4 * x8 +
              (1.0 / 280.0) * u[0] * u[8] * x2 * x8 + (1.0 / 280.0) * u[0] * u[8] * x4 * x8 +
              (4.0 / 315.0) * u[10] * u[3] * x4 * x8 + (4.0 / 315.0) * u[10] * u[5] * x2 * x8 +
              (2.0 / 315.0) * u[11] * u[3] * x2 * x8 + (4.0 / 315.0) * u[11] * u[5] * x4 * x8 +
              (11.0 / 2520.0) * u[1] * u[2] * x3 * x8 + (11.0 / 2520.0) * u[1] * u[2] * x5 * x8 +
              (11.0 / 2520.0) * u[1] * u[8] * x4 * x8 + (11.0 / 2520.0) * u[2] * u[7] * x2 * x8 +
              (2.0 / 315.0) * u[3] * u[4] * x3 * x8 + (2.0 / 63.0) * u[3] * u[5] * x3 * x8 +
              (2.0 / 63.0) * u[3] * u[5] * x5 * x8 + (4.0 / 105.0) * u[3] * u[6] * x4 * x8 +
              (4.0 / 315.0) * u[3] * u[9] * x2 * x8 + (2.0 / 315.0) * u[4] * u[5] * x5 * x8 +
              (1.0 / 105.0) * u[4] * u[6] * x2 * x8 + (1.0 / 105.0) * u[4] * u[6] * x4 * x8 +
              (1.0 / 630.0) * u[4] * u[7] * x4 * x8 + (1.0 / 630.0) * u[4] * u[8] * x2 * x8 -
              u[4] * x58 + (4.0 / 105.0) * u[5] * u[6] * x2 * x8 +
              (2.0 / 315.0) * u[5] * u[9] * x4 * x8 - x11 * x19 - x11 * x51 - x13 - x15 * x19 -
              x15 * x52 - x17 - x20 * x21 - x20 * x22 - x21 * x32 - x22 * x31 - x23 * x25 -
              x23 * x27 - x25 * x43 - x27 * x44 - x28 * x29 - x28 * x30 - x29 * x33 +
              (4.0 / 315.0) * x3 * x8 * x9 - x30 * x33 - x34 * x35 - x34 * x45 - x34 * x46 -
              x34 * x65 - x35 * x36 - x36 * x45 - x36 * x46 - x36 * x65 - x39 - x42 - x47 * x56 -
              x49 + (4.0 / 315.0) * x5 * x7 * x8 - x55 - x56 * x62 - x57 * x61 - x59 * x60 -
              x63 * x64 - x75 - x81 - x85 - x89 - x93);
    element_vector[1] =
        x6 * ((1.0 / 60.0) * u[1] - 1.0 / 90.0 * u[5] - x1 - x94) -
        x6 * (u[0] * u[9] * x57 + u[0] * x136 - u[10] * x105 - u[1] * u[8] * x37 - u[1] * x124 +
              u[2] * x136 + u[3] * x138 - u[3] * x58 - u[4] * x106 + u[5] * x106 -
              1.0 / 140.0 * u[7] * x30 - x100 * x80 - x101 * x34 - x101 * x36 - x103 -
              4.0 / 105.0 * x104 - x107 * x72 - x108 - 2.0 / 105.0 * x109 - x110 * x111 - x112 -
              x113 * x22 + x114 * x25 - x115 - x116 - 1.0 / 21.0 * x117 - x118 * x25 + x118 * x63 -
              x119 * x27 + x119 * x53 - x12 * x121 - x120 * x78 - x121 * x122 + x122 * x74 -
              x123 * x22 + x123 * x90 - x125 * x27 + x125 * x53 + (1.0 / 21.0) * x126 +
              x127 * x130 + (1.0 / 35.0) * x128 + x129 * x34 + x129 * x36 + x131 * x34 +
              x131 * x36 + x132 * x21 + x133 + (2.0 / 105.0) * x134 + x135 + (2.0 / 315.0) * x137 +
              x139 * x30 + (13.0 / 420.0) * x140 + x143 + x144 - x17 + x25 * x83 + x27 * x83 +
              x29 * x71 - x32 * x47 + x36 * x64 - x42 - x54 * x78 - x59 * x67 + x73 * x78 + x75 -
              x82 - x84 - x96 + x98 + (13.0 / 420.0) * x99);
    element_vector[2] =
        x6 * ((1.0 / 60.0) * u[2] - 1.0 / 90.0 * u[3] - x0 - x94) -
        x6 * (u[0] * x58 - u[1] * x106 - u[2] * x159 - u[6] * x160 - u[7] * x160 +
              (2.0 / 315.0) * x102 - x107 * x62 + x108 + x110 * x130 + x111 * x47 - x111 * x62 +
              x112 - x113 * x62 - x113 * x90 + x114 * x27 + (2.0 / 315.0) * x117 - x120 * x69 -
              x121 * x157 - x121 * x16 - x125 * x25 + x125 * x63 + x127 * x154 -
              2.0 / 105.0 * x128 - x13 + x132 * x22 - x133 - 1.0 / 126.0 * x134 - x135 +
              x139 * x29 + x144 - 4.0 / 315.0 * x145 + (13.0 / 420.0) * x146 - x147 * x74 -
              x148 * x34 - x148 * x36 - x149 - 4.0 / 105.0 * x150 + x151 * x25 - x151 * x63 - x152 -
              x153 - x154 * x22 - x155 * x27 + x155 * x53 - x156 * x25 + x156 * x63 + x157 * x80 -
              x158 * x29 + x158 * x34 + x158 * x36 + (1.0 / 35.0) * x161 + x162 * x34 + x162 * x36 +
              (1.0 / 210.0) * x163 + (2.0 / 315.0) * x164 + (4.0 / 315.0) * x165 +
              (4.0 / 315.0) * x166 + (13.0 / 420.0) * x167 + x168 + x25 * x87 + x27 * x87 +
              x30 * x71 - x39 - x64 * x78 + x69 * x79 + x81 - x86 - x88 + x96 - x98);
    element_vector[3] =
        x6 * (-1.0 / 90.0 * u[2] + (4.0 / 45.0) * u[3] + x169 + x170) -
        x6 * ((4.0 / 315.0) * u[0] * u[1] * x5 * x8 + (1.0 / 126.0) * u[0] * u[2] * x3 * x8 +
              (4.0 / 315.0) * u[0] * u[7] * x2 * x8 + (4.0 / 315.0) * u[0] * u[7] * x4 * x8 +
              (1.0 / 126.0) * u[0] * u[8] * x2 * x8 + (1.0 / 126.0) * u[0] * u[8] * x4 * x8 +
              (2.0 / 63.0) * u[10] * u[1] * x4 * x8 + (2.0 / 315.0) * u[10] * u[2] * x2 * x8 +
              (8.0 / 105.0) * u[10] * u[4] * x2 * x8 + (16.0 / 315.0) * u[10] * u[4] * x4 * x8 -
              u[10] * x185 + (2.0 / 315.0) * u[11] * u[2] * x2 * x8 +
              (16.0 / 315.0) * u[11] * u[4] * x4 * x8 + (2.0 / 63.0) * u[1] * u[3] * x3 * x8 +
              (8.0 / 315.0) * u[1] * u[4] * x3 * x8 + (2.0 / 105.0) * u[1] * u[4] * x5 * x8 +
              (4.0 / 315.0) * u[1] * u[5] * x3 * x8 + (2.0 / 105.0) * u[1] * u[7] * x4 * x8 +
              (4.0 / 63.0) * u[1] * u[9] * x4 * x8 + (1.0 / 210.0) * u[2] * u[8] * x2 * x8 -
              u[2] * x177 + (8.0 / 315.0) * u[3] * u[4] * x3 * x8 +
              (4.0 / 315.0) * u[3] * u[6] * x2 * x8 + (2.0 / 63.0) * u[3] * u[6] * x4 * x8 +
              (2.0 / 105.0) * u[3] * u[8] * x2 * x8 + (2.0 / 105.0) * u[4] * u[7] * x2 * x8 +
              (16.0 / 105.0) * u[4] * u[9] * x2 * x8 - u[4] * x138 +
              (8.0 / 315.0) * u[5] * u[6] * x2 * x8 + (2.0 / 315.0) * u[5] * u[6] * x4 * x8 +
              (2.0 / 315.0) * u[5] * u[7] * x4 * x8 + (32.0 / 315.0) * u[5] * u[9] * x2 * x8 -
              u[6] * x179 - u[7] * x70 - u[8] * x176 + (2.0 / 105.0) * x10 * x3 * x8 - x103 -
              x110 * x56 - x114 * x30 + (1.0 / 210.0) * x14 * x5 * x8 - 4.0 / 315.0 * x150 - x152 -
              x153 - 4.0 / 105.0 * x163 - 16.0 / 315.0 * x164 - 2.0 / 63.0 * x165 - x168 -
              x171 * x31 - 2.0 / 63.0 * x172 - 2.0 / 63.0 * x173 - x174 - x175 - x178 * x34 -
              x178 * x36 - x180 * x63 - x181 * x25 - 8.0 / 315.0 * x182 - x183 * x22 - x184 * x25 -
              16.0 / 315.0 * x186 - x192 - x197 - x29 * x73 + (16.0 / 315.0) * x3 * x67 * x8 -
              x30 * x87 + (8.0 / 105.0) * x5 * x67 * x8 - x85 - 16.0 / 105.0 * x95 -
              16.0 / 315.0 * x97);
    element_vector[4] =
        x6 * (-1.0 / 90.0 * u[0] + (4.0 / 45.0) * u[4] + x170 + x198) -
        x6 *
            (u[0] * x177 + (4.0 / 63.0) * u[10] * x29 + (4.0 / 63.0) * u[10] * x30 +
             (8.0 / 105.0) * u[11] * x53 + u[3] * x211 + u[4] * x189 - u[4] * x203 + u[5] * x138 +
             u[5] * x203 - u[7] * x105 + u[7] * x176 - u[8] * x179 - 8.0 / 315.0 * u[8] * x78 +
             (16.0 / 315.0) * u[9] * x53 - 32.0 / 315.0 * x102 - 8.0 / 315.0 * x104 - x11 * x199 +
             (16.0 / 105.0) * x11 * x67 + (2.0 / 315.0) * x11 * x80 + x111 * x127 - x120 * x29 -
             x120 * x30 - x121 * x59 + (2.0 / 105.0) * x126 + (16.0 / 315.0) * x128 +
             (2.0 / 63.0) * x134 - 32.0 / 315.0 * x137 + (2.0 / 105.0) * x140 +
             (16.0 / 105.0) * x145 + (2.0 / 105.0) * x146 - x15 * x199 + (16.0 / 315.0) * x161 +
             (4.0 / 315.0) * x164 - 2.0 / 315.0 * x165 + (2.0 / 105.0) * x167 - 2.0 / 315.0 * x172 -
             x174 - x175 + x180 * x78 - 8.0 / 105.0 * x186 + x188 * x25 + x188 * x27 - x188 * x53 -
             x188 * x63 - x190 - x191 + x197 - x200 * x25 - x200 * x27 + x200 * x53 + x200 * x63 -
             x201 * x25 + x201 * x63 - x202 * x56 + x204 * x25 - x204 * x63 - x205 + x206 * x53 -
             x207 - x208 * x34 - x208 * x36 + x209 * x47 + x210 * x34 + x210 * x36 + x215 -
             x34 * x54 + x34 * x83 + x34 * x87 - x36 * x54 + x36 * x83 + x36 * x87 + x93 -
             8.0 / 105.0 * x95 - 8.0 / 105.0 * x97 + (2.0 / 105.0) * x99);
    element_vector[5] =
        x6 * (-1.0 / 90.0 * u[1] + (4.0 / 45.0) * u[5] + x169 + x198) -
        x6 * ((1.0 / 126.0) * u[0] * u[1] * x5 * x8 + (4.0 / 315.0) * u[0] * u[2] * x3 * x8 +
              (1.0 / 126.0) * u[0] * u[7] * x2 * x8 + (1.0 / 126.0) * u[0] * u[7] * x4 * x8 +
              (4.0 / 315.0) * u[0] * u[8] * x2 * x8 + (4.0 / 315.0) * u[0] * u[8] * x4 * x8 -
              4.0 / 315.0 * u[0] * x110 + (2.0 / 315.0) * u[10] * u[1] * x4 * x8 +
              (2.0 / 63.0) * u[10] * u[2] * x2 * x8 + (16.0 / 315.0) * u[10] * u[4] * x2 * x8 +
              (8.0 / 105.0) * u[10] * u[4] * x4 * x8 - u[10] * x221 +
              (4.0 / 63.0) * u[11] * u[2] * x2 * x8 + (32.0 / 315.0) * u[11] * u[3] * x4 * x8 +
              (16.0 / 105.0) * u[11] * u[4] * x4 * x8 + (1.0 / 210.0) * u[1] * u[7] * x4 * x8 +
              (2.0 / 315.0) * u[1] * u[9] * x4 * x8 - 4.0 / 105.0 * u[1] * x90 +
              (4.0 / 315.0) * u[2] * u[3] * x5 * x8 + (2.0 / 105.0) * u[2] * u[4] * x3 * x8 +
              (8.0 / 315.0) * u[2] * u[4] * x5 * x8 + (2.0 / 63.0) * u[2] * u[5] * x5 * x8 +
              (2.0 / 105.0) * u[2] * u[8] * x2 * x8 - u[2] * x216 +
              (2.0 / 315.0) * u[3] * u[6] * x2 * x8 + (8.0 / 315.0) * u[3] * u[6] * x4 * x8 +
              (2.0 / 315.0) * u[3] * u[8] * x2 * x8 - u[3] * x203 +
              (8.0 / 315.0) * u[4] * u[5] * x5 * x8 + (2.0 / 105.0) * u[4] * u[8] * x4 * x8 +
              (16.0 / 315.0) * u[4] * u[9] * x2 * x8 - u[4] * x211 - u[4] * x220 +
              (2.0 / 63.0) * u[5] * u[6] * x2 * x8 + (4.0 / 315.0) * u[5] * u[6] * x4 * x8 +
              (2.0 / 105.0) * u[5] * u[7] * x4 * x8 - u[6] * x105 - u[6] * x219 - u[8] * x219 -
              u[9] * x221 + (1.0 / 210.0) * x10 * x3 * x8 - x115 - x116 +
              (2.0 / 105.0) * x14 * x5 * x8 - x143 - x149 - 2.0 / 63.0 * x150 - 8.0 / 315.0 * x164 -
              4.0 / 63.0 * x165 - 2.0 / 63.0 * x166 - x171 * x50 - 4.0 / 63.0 * x172 - x181 * x27 -
              16.0 / 315.0 * x182 - x183 * x62 - 16.0 / 105.0 * x186 - x192 - x205 - x207 - x215 -
              x217 * x34 - x217 * x36 - x218 * x53 + (8.0 / 105.0) * x3 * x67 * x8 - x30 * x79 +
              (16.0 / 315.0) * x5 * x67 * x8 - x89 - 16.0 / 315.0 * x95 - 16.0 / 105.0 * x97);
    element_vector[6] =
        x6 * (-1.0 / 90.0 * u[10] + (1.0 / 60.0) * u[6] - x222 - x223) -
        x6 * ((1.0 / 105.0) * u[0] * u[10] * x3 * x8 + (1.0 / 105.0) * u[0] * u[10] * x5 * x8 +
              (4.0 / 105.0) * u[0] * u[11] * x5 * x8 + (4.0 / 105.0) * u[0] * u[9] * x3 * x8 -
              u[0] * x236 + (2.0 / 315.0) * u[10] * u[11] * x2 * x8 +
              (1.0 / 630.0) * u[10] * u[1] * x3 * x8 + (1.0 / 630.0) * u[10] * u[2] * x5 * x8 +
              (1.0 / 210.0) * u[10] * u[6] * x2 * x8 + (1.0 / 210.0) * u[10] * u[6] * x4 * x8 +
              (2.0 / 315.0) * u[10] * u[9] * x4 * x8 - u[10] * x243 - u[11] * u[1] * x213 +
              (2.0 / 315.0) * u[11] * u[3] * x3 * x8 + (4.0 / 315.0) * u[11] * u[4] * x5 * x8 +
              (4.0 / 315.0) * u[11] * u[5] * x3 * x8 + (2.0 / 105.0) * u[11] * u[6] * x2 * x8 +
              (2.0 / 63.0) * u[11] * u[9] * x2 * x8 + (2.0 / 63.0) * u[11] * u[9] * x4 * x8 -
              1.0 / 105.0 * u[11] * x234 + (1.0 / 280.0) * u[1] * u[6] * x3 * x8 +
              (1.0 / 280.0) * u[1] * u[6] * x5 * x8 + (11.0 / 2520.0) * u[1] * u[8] * x5 * x8 +
              (1.0 / 280.0) * u[2] * u[6] * x3 * x8 + (1.0 / 280.0) * u[2] * u[6] * x5 * x8 +
              (11.0 / 2520.0) * u[2] * u[7] * x3 * x8 - u[2] * u[9] * x195 +
              (4.0 / 315.0) * u[3] * u[9] * x5 * x8 + (4.0 / 315.0) * u[4] * u[9] * x3 * x8 +
              (2.0 / 315.0) * u[5] * u[9] * x5 * x8 + (1.0 / 280.0) * u[6] * u[7] * x2 * x8 +
              (1.0 / 280.0) * u[6] * u[8] * x4 * x8 + (2.0 / 105.0) * u[6] * u[9] * x4 * x8 +
              (11.0 / 2520.0) * u[7] * u[8] * x2 * x8 + (11.0 / 2520.0) * u[7] * u[8] * x4 * x8 -
              u[7] * x244 - u[9] * x203 - 1.0 / 105.0 * u[9] * x235 - x110 * x208 - x110 * x73 -
              x127 * x208 - x127 * x79 - x131 * x234 - x151 * x21 - x188 * x21 - x188 * x22 -
              x188 * x62 - x188 * x72 + (4.0 / 315.0) * x2 * x225 * x8 - x204 * x47 - x204 * x62 +
              (4.0 / 315.0) * x224 * x4 * x8 - x227 - x229 - x231 * x24 - x231 * x26 - x232 * x233 -
              x232 * x90 - x233 * x73 - x234 * x65 - x235 * x65 - x237 - x238 - x239 - x24 * x242 -
              x240 - x241 * x26 - x250 - x253 - x256 - x259 - x262 - x79 * x90);
    element_vector[7] =
        x6 * (-1.0 / 90.0 * u[11] + (1.0 / 60.0) * u[7] - x223 - x263) -
        x6 * (u[0] * u[8] * x147 + u[10] * x118 * x24 + (1.0 / 21.0) * u[10] * x202 - u[10] * x274 +
              u[10] * x290 + u[11] * x274 - u[11] * x290 - u[1] * x236 + u[6] * x244 - u[7] * x288 +
              (13.0 / 420.0) * u[7] * x294 - u[8] * x38 + u[9] * x211 - u[9] * x243 - x100 * x269 -
              x101 * x270 - x110 * x204 + x114 * x90 - x118 * x233 - x12 * x289 + x120 * x22 +
              x120 * x62 + x127 * x218 - x129 * x235 - x131 * x277 - x147 * x269 + x180 * x21 +
              x196 * x285 + x202 * x33 - x21 * x283 - x217 * x275 + x218 * x72 + x22 * x287 - x229 +
              x235 * x54 - x238 - x246 * x57 + x250 - x254 - x255 - x265 + x267 +
              (13.0 / 420.0) * x268 + x270 * x28 - x272 - 4.0 / 105.0 * x273 - x276 - x277 * x71 -
              x278 + x279 * x291 - x279 * x46 - x280 + x281 * x54 - x282 - 1.0 / 21.0 * x284 -
              x285 * x286 + x286 * x292 - x287 * x90 + x293 * x32 + x293 * x48 + x294 * x33 +
              x295 * x33 + x296 + x297 + x298 * x46 + x301 + x302 - x54 * x90 - x62 * x73 +
              x83 * x90);
    element_vector[8] =
        x6 * ((1.0 / 60.0) * u[8] - 1.0 / 90.0 * u[9] - x222 - x263) -
        x6 * (u[0] * u[7] * x100 - u[0] * x312 + u[10] * x310 + u[10] * x313 -
              2.0 / 63.0 * u[10] * x47 - 4.0 / 315.0 * u[11] * x21 - 4.0 / 105.0 * u[11] * x281 -
              u[1] * x312 + u[6] * x243 - u[7] * x274 - u[7] * x41 + u[8] * x159 -
              1.0 / 210.0 * u[8] * x233 + (13.0 / 420.0) * u[8] * x281 - u[9] * x310 - u[9] * x313 -
              x100 * x305 - x110 * x151 + x110 * x206 + x110 * x64 + x120 * x21 + x120 * x72 -
              x130 * x248 - x131 * x171 - x131 * x311 - x139 * x295 - x147 * x305 - x154 * x285 +
              x154 * x292 + x155 * x275 - x155 * x90 + x171 * x33 + x180 * x47 + x180 * x62 -
              x206 * x47 + x206 * x62 - x22 * x54 - x227 + x234 * x283 - x237 + x248 * x48 + x253 -
              x257 - x258 + x265 - x267 + x270 * x45 + (1.0 / 126.0) * x273 + x276 + x277 * x291 +
              x278 - x279 * x71 + x28 * x298 + x281 * x33 + (2.0 / 315.0) * x284 + x285 * x50 +
              x295 * x54 - x296 - x297 + x298 * x54 + x302 - 4.0 / 315.0 * x303 +
              (13.0 / 420.0) * x304 - x307 - x308 - x309 + x311 * x33 + x314 - x47 * x54 -
              x72 * x79);
    element_vector[9] =
        x6 * (-1.0 / 90.0 * u[8] + (4.0 / 45.0) * u[9] + x315 + x316) -
        x6 * ((2.0 / 315.0) * u[0] * u[11] * x3 * x8 + (8.0 / 315.0) * u[0] * u[11] * x5 * x8 +
              (2.0 / 63.0) * u[0] * u[9] * x3 * x8 + (4.0 / 315.0) * u[0] * u[9] * x5 * x8 +
              (2.0 / 105.0) * u[10] * u[1] * x5 * x8 + (16.0 / 105.0) * u[10] * u[3] * x5 * x8 +
              (16.0 / 315.0) * u[10] * u[4] * x3 * x8 + (8.0 / 105.0) * u[10] * u[4] * x5 * x8 +
              (16.0 / 315.0) * u[10] * u[5] * x3 * x8 + (2.0 / 105.0) * u[10] * u[7] * x2 * x8 +
              (8.0 / 315.0) * u[10] * u[7] * x4 * x8 + (8.0 / 315.0) * u[10] * u[9] * x4 * x8 +
              (2.0 / 315.0) * u[11] * u[1] * x3 * x8 + (32.0 / 315.0) * u[11] * u[3] * x5 * x8 +
              (4.0 / 315.0) * u[11] * u[7] * x4 * x8 - u[11] * x212 +
              (4.0 / 315.0) * u[1] * u[6] * x3 * x8 + (4.0 / 315.0) * u[1] * u[6] * x5 * x8 +
              (2.0 / 105.0) * u[1] * u[7] * x3 * x8 + (1.0 / 126.0) * u[2] * u[6] * x3 * x8 +
              (1.0 / 126.0) * u[2] * u[6] * x5 * x8 + (1.0 / 210.0) * u[2] * u[8] * x5 * x8 +
              (2.0 / 105.0) * u[2] * u[9] * x5 * x8 + (4.0 / 63.0) * u[3] * u[7] * x3 * x8 +
              (2.0 / 63.0) * u[4] * u[7] * x3 * x8 + (2.0 / 315.0) * u[4] * u[8] * x5 * x8 +
              (2.0 / 315.0) * u[5] * u[8] * x5 * x8 + (4.0 / 315.0) * u[6] * u[7] * x2 * x8 +
              (1.0 / 126.0) * u[6] * u[8] * x4 * x8 - u[6] * x106 +
              (2.0 / 63.0) * u[7] * u[9] * x4 * x8 - u[7] * x11 * x32 - u[7] * x138 - u[7] * x249 -
              u[8] * x177 - 4.0 / 105.0 * u[8] * x21 - x110 * x201 - x127 * x322 - x151 * x235 -
              x151 * x277 - x184 * x21 + (1.0 / 210.0) * x2 * x228 * x8 +
              (8.0 / 105.0) * x2 * x246 * x8 - x21 * x320 - x217 * x270 - x22 * x317 +
              (2.0 / 105.0) * x226 * x4 * x8 - x233 * x323 - x235 * x83 +
              (16.0 / 315.0) * x246 * x4 * x8 - x256 - 16.0 / 105.0 * x264 - 16.0 / 315.0 * x266 -
              x271 * x90 - x272 - 2.0 / 63.0 * x273 - x277 * x87 - x281 * x71 - x293 * x56 - x308 -
              x309 - x314 - x317 * x62 - x318 - x319 - x320 * x72 - x321 * x90 - x328 - x329);
    element_vector[10] =
        x6 * ((4.0 / 45.0) * u[10] - 1.0 / 90.0 * u[6] + x316 + x330) -
        x6 *
            (u[0] * u[11] * x335 + u[10] * x193 - u[10] * x220 + (16.0 / 315.0) * u[10] * x279 -
             u[10] * x334 - 32.0 / 315.0 * u[11] * x110 + u[11] * x220 - u[1] * x336 - u[6] * x136 -
             u[6] * x15 * x339 + u[6] * x177 + u[6] * x58 + u[7] * x11 * x214 +
             (4.0 / 63.0) * u[7] * x127 + (2.0 / 63.0) * u[7] * x72 + (4.0 / 63.0) * u[8] * x110 +
             u[8] * x111 * x15 - u[8] * x203 + u[8] * x252 - 32.0 / 315.0 * u[9] * x127 +
             u[9] * x138 + u[9] * x334 - x110 * x184 + x110 * x200 + x110 * x341 + x114 * x171 -
             x114 * x21 + x114 * x281 - x114 * x62 - x114 * x72 + x127 * x200 - x127 * x332 +
             x127 * x341 + x171 * x71 + x181 * x21 + x181 * x22 - x201 * x21 + x202 * x71 -
             x204 * x295 - x22 * x333 + x234 * x45 - x234 * x71 - x234 * x79 + x235 * x46 -
             x235 * x71 - x235 * x73 - x24 * x331 + (16.0 / 105.0) * x246 * x26 - x26 * x331 +
             x262 - 8.0 / 105.0 * x264 - 8.0 / 105.0 * x266 + (2.0 / 105.0) * x268 + x270 * x83 -
             8.0 / 315.0 * x273 + x277 * x342 - x281 * x306 - x289 * x59 + x295 * x71 + x298 * x64 +
             x298 * x87 + (16.0 / 105.0) * x303 + (2.0 / 105.0) * x304 + x311 * x71 - x318 - x319 -
             x326 - x327 + x329 - x337 - x338 - x339 * x340 + x340 * x50 + x342 * x72 + x344);
    element_vector[11] =
        x6 * ((4.0 / 45.0) * u[11] - 1.0 / 90.0 * u[7] + x315 + x330) -
        x6 * ((4.0 / 315.0) * u[0] * u[11] * x3 * x8 + (2.0 / 63.0) * u[0] * u[11] * x5 * x8 +
              (8.0 / 315.0) * u[0] * u[9] * x3 * x8 + (2.0 / 315.0) * u[0] * u[9] * x5 * x8 -
              u[0] * x336 + (8.0 / 315.0) * u[10] * u[11] * x2 * x8 +
              (2.0 / 105.0) * u[10] * u[2] * x3 * x8 + (16.0 / 315.0) * u[10] * u[3] * x5 * x8 +
              (8.0 / 105.0) * u[10] * u[4] * x3 * x8 + (16.0 / 315.0) * u[10] * u[4] * x5 * x8 +
              (16.0 / 105.0) * u[10] * u[5] * x3 * x8 + (8.0 / 315.0) * u[10] * u[8] * x2 * x8 +
              (2.0 / 105.0) * u[10] * u[8] * x4 * x8 - u[10] * x138 +
              (2.0 / 105.0) * u[11] * u[1] * x3 * x8 + (2.0 / 63.0) * u[11] * u[8] * x2 * x8 +
              (1.0 / 126.0) * u[1] * u[6] * x3 * x8 + (1.0 / 126.0) * u[1] * u[6] * x5 * x8 +
              (1.0 / 210.0) * u[1] * u[7] * x3 * x8 + (4.0 / 315.0) * u[2] * u[6] * x3 * x8 +
              (4.0 / 315.0) * u[2] * u[6] * x5 * x8 + (2.0 / 105.0) * u[2] * u[8] * x5 * x8 +
              (2.0 / 315.0) * u[2] * u[9] * x5 * x8 + (2.0 / 315.0) * u[3] * u[7] * x3 * x8 +
              (2.0 / 315.0) * u[4] * u[7] * x3 * x8 + (2.0 / 63.0) * u[4] * u[8] * x5 * x8 +
              (4.0 / 63.0) * u[5] * u[8] * x5 * x8 + (32.0 / 315.0) * u[5] * u[9] * x3 * x8 +
              (1.0 / 126.0) * u[6] * u[7] * x2 * x8 + (4.0 / 315.0) * u[6] * u[8] * x4 * x8 -
              u[7] * x106 - 4.0 / 105.0 * u[7] * x22 + (4.0 / 315.0) * u[8] * u[9] * x2 * x8 -
              u[8] * x216 - x110 * x343 - x127 * x333 - x151 * x298 - x171 * x87 +
              (2.0 / 105.0) * x2 * x228 * x8 + (16.0 / 315.0) * x2 * x246 * x8 - x209 * x293 -
              x21 * x317 - x21 * x343 - x22 * x320 - x22 * x332 + (1.0 / 210.0) * x226 * x4 * x8 -
              x233 * x321 - x234 * x87 + (8.0 / 105.0) * x246 * x4 * x8 - x259 -
              16.0 / 315.0 * x264 - 16.0 / 105.0 * x266 - x271 * x47 - x280 - x282 - x294 * x71 -
              x295 * x73 - x301 - x307 - x311 * x87 - x317 * x47 - x317 * x72 - x320 * x62 -
              x323 * x90 - x328 - x337 - x338 - x344 - x59 * x61);
}

static SFEM_INLINE void tr6_momentum_kernel(const real_t px0,
                                            const real_t px1,
                                            const real_t px2,
                                            const real_t py0,
                                            const real_t py1,
                                            const real_t py2,
                                            const real_t dt,
                                            const real_t nu,
                                            real_t *const SFEM_RESTRICT element_matrix) {
    const real_t x0 = -px0 + px1;
    const real_t x1 = -py0 + py2;
    const real_t x2 = px0 - px2;
    const real_t x3 = py0 - py1;
    const real_t x4 = x0 * x1 - x2 * x3;
    const real_t x5 = x4 / dt;
    const real_t x6 = (1.0 / 60.0) * x5;
    const real_t x7 = pow(x4, -2);
    const real_t x8 = x0 * x2 * x7;
    const real_t x9 = x1 * x3 * x7;
    const real_t x10 = pow(x2, 2);
    const real_t x11 = (1.0 / 2.0) * x7;
    const real_t x12 = pow(x1, 2);
    const real_t x13 = x10 * x11 + x11 * x12;
    const real_t x14 = pow(x0, 2);
    const real_t x15 = pow(x3, 2);
    const real_t x16 = x11 * x14 + x11 * x15;
    const real_t x17 = nu * x4;
    const real_t x18 = x17 * (x13 + x16 + x8 + x9) + x6;
    const real_t x19 = -1.0 / 360.0 * x5;
    const real_t x20 = (1.0 / 6.0) * x7;
    const real_t x21 = (1.0 / 6.0) * x8 + (1.0 / 6.0) * x9;
    const real_t x22 = x17 * (x10 * x20 + x12 * x20 + x21) + x19;
    const real_t x23 = x17 * (x14 * x20 + x15 * x20 + x21) + x19;
    const real_t x24 = (2.0 / 3.0) * x7;
    const real_t x25 = (2.0 / 3.0) * x8 + (2.0 / 3.0) * x9;
    const real_t x26 = x17 * (-x10 * x24 - x12 * x24 - x25);
    const real_t x27 = -1.0 / 90.0 * x5;
    const real_t x28 = x17 * (-x14 * x24 - x15 * x24 - x25);
    const real_t x29 = x13 * x17 + x6;
    const real_t x30 = -x17 * x21 + x19;
    const real_t x31 = x17 * x25;
    const real_t x32 = x16 * x17 + x6;
    const real_t x33 = (4.0 / 3.0) * x7;
    const real_t x34 = (4.0 / 3.0) * x8 + (4.0 / 3.0) * x9;
    const real_t x35 = x14 * x33 + x15 * x33 + x34;
    const real_t x36 = x10 * x33 + x12 * x33;
    const real_t x37 = x17 * (x35 + x36) + (4.0 / 45.0) * x5;
    const real_t x38 = (2.0 / 45.0) * x5;
    const real_t x39 = -x17 * x35 + x38;
    const real_t x40 = x17 * x34 + x38;
    const real_t x41 = x17 * (-x34 - x36) + x38;
    element_matrix[0] = x18;
    element_matrix[1] = x22;
    element_matrix[2] = x23;
    element_matrix[3] = x26;
    element_matrix[4] = x27;
    element_matrix[5] = x28;
    element_matrix[6] = 0;
    element_matrix[7] = 0;
    element_matrix[8] = 0;
    element_matrix[9] = 0;
    element_matrix[10] = 0;
    element_matrix[11] = 0;
    element_matrix[12] = x22;
    element_matrix[13] = x29;
    element_matrix[14] = x30;
    element_matrix[15] = x26;
    element_matrix[16] = x31;
    element_matrix[17] = x27;
    element_matrix[18] = 0;
    element_matrix[19] = 0;
    element_matrix[20] = 0;
    element_matrix[21] = 0;
    element_matrix[22] = 0;
    element_matrix[23] = 0;
    element_matrix[24] = x23;
    element_matrix[25] = x30;
    element_matrix[26] = x32;
    element_matrix[27] = x27;
    element_matrix[28] = x31;
    element_matrix[29] = x28;
    element_matrix[30] = 0;
    element_matrix[31] = 0;
    element_matrix[32] = 0;
    element_matrix[33] = 0;
    element_matrix[34] = 0;
    element_matrix[35] = 0;
    element_matrix[36] = x26;
    element_matrix[37] = x26;
    element_matrix[38] = x27;
    element_matrix[39] = x37;
    element_matrix[40] = x39;
    element_matrix[41] = x40;
    element_matrix[42] = 0;
    element_matrix[43] = 0;
    element_matrix[44] = 0;
    element_matrix[45] = 0;
    element_matrix[46] = 0;
    element_matrix[47] = 0;
    element_matrix[48] = x27;
    element_matrix[49] = x31;
    element_matrix[50] = x31;
    element_matrix[51] = x39;
    element_matrix[52] = x37;
    element_matrix[53] = x41;
    element_matrix[54] = 0;
    element_matrix[55] = 0;
    element_matrix[56] = 0;
    element_matrix[57] = 0;
    element_matrix[58] = 0;
    element_matrix[59] = 0;
    element_matrix[60] = x28;
    element_matrix[61] = x27;
    element_matrix[62] = x28;
    element_matrix[63] = x40;
    element_matrix[64] = x41;
    element_matrix[65] = x37;
    element_matrix[66] = 0;
    element_matrix[67] = 0;
    element_matrix[68] = 0;
    element_matrix[69] = 0;
    element_matrix[70] = 0;
    element_matrix[71] = 0;
    element_matrix[72] = 0;
    element_matrix[73] = 0;
    element_matrix[74] = 0;
    element_matrix[75] = 0;
    element_matrix[76] = 0;
    element_matrix[77] = 0;
    element_matrix[78] = x18;
    element_matrix[79] = x22;
    element_matrix[80] = x23;
    element_matrix[81] = x26;
    element_matrix[82] = x27;
    element_matrix[83] = x28;
    element_matrix[84] = 0;
    element_matrix[85] = 0;
    element_matrix[86] = 0;
    element_matrix[87] = 0;
    element_matrix[88] = 0;
    element_matrix[89] = 0;
    element_matrix[90] = x22;
    element_matrix[91] = x29;
    element_matrix[92] = x30;
    element_matrix[93] = x26;
    element_matrix[94] = x31;
    element_matrix[95] = x27;
    element_matrix[96] = 0;
    element_matrix[97] = 0;
    element_matrix[98] = 0;
    element_matrix[99] = 0;
    element_matrix[100] = 0;
    element_matrix[101] = 0;
    element_matrix[102] = x23;
    element_matrix[103] = x30;
    element_matrix[104] = x32;
    element_matrix[105] = x27;
    element_matrix[106] = x31;
    element_matrix[107] = x28;
    element_matrix[108] = 0;
    element_matrix[109] = 0;
    element_matrix[110] = 0;
    element_matrix[111] = 0;
    element_matrix[112] = 0;
    element_matrix[113] = 0;
    element_matrix[114] = x26;
    element_matrix[115] = x26;
    element_matrix[116] = x27;
    element_matrix[117] = x37;
    element_matrix[118] = x39;
    element_matrix[119] = x40;
    element_matrix[120] = 0;
    element_matrix[121] = 0;
    element_matrix[122] = 0;
    element_matrix[123] = 0;
    element_matrix[124] = 0;
    element_matrix[125] = 0;
    element_matrix[126] = x27;
    element_matrix[127] = x31;
    element_matrix[128] = x31;
    element_matrix[129] = x39;
    element_matrix[130] = x37;
    element_matrix[131] = x41;
    element_matrix[132] = 0;
    element_matrix[133] = 0;
    element_matrix[134] = 0;
    element_matrix[135] = 0;
    element_matrix[136] = 0;
    element_matrix[137] = 0;
    element_matrix[138] = x28;
    element_matrix[139] = x27;
    element_matrix[140] = x28;
    element_matrix[141] = x40;
    element_matrix[142] = x41;
    element_matrix[143] = x37;
}
