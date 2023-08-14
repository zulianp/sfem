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

// Implicit Euler
// Chorin's projection method
// 1) temptative momentum step
//   `1/dt * <u, v> + nu * <grad(u), grad(v)> = <u_old, v> - <(u_old . div) * u_old, v>`
// 2) Potential eqaution
//   `<grad(p), grad(q)> = - 1/dt * <div(u), q>`
// 3) Projection/Correction
//   `<u_new, v> = <u, v> - dt * <grad(p), v>`

// Taylor hood Triangle P2/P1
// 1) Temptative momentum step

static SFEM_INLINE void tri6_momentum_lhs_kernel(const real_t px0,
                                                 const real_t px1,
                                                 const real_t px2,
                                                 const real_t py0,
                                                 const real_t py1,
                                                 const real_t py2,
                                                 const real_t dt,
                                                 const real_t nu,
                                                 real_t *const SFEM_RESTRICT element_matrix) {
    const real_t x0 = px0 - px1;
    const real_t x1 = py0 - py2;
    const real_t x2 = x0 * x1;
    const real_t x3 = px0 - px2;
    const real_t x4 = py0 - py1;
    const real_t x5 = x3 * x4;
    const real_t x6 = x2 - x5;
    const real_t x7 = pow(x6, 2);
    const real_t x8 = x0 * x3;
    const real_t x9 = x1 * x4;
    const real_t x10 = pow(x3, 2);
    const real_t x11 = pow(x1, 2);
    const real_t x12 = x10 + x11;
    const real_t x13 = pow(x0, 2) + pow(x4, 2);
    const real_t x14 = dt * nu;
    const real_t x15 = 30 * x14;
    const real_t x16 = 1.0 / dt;
    const real_t x17 = x16 / x6;
    const real_t x18 = (1.0 / 60.0) * x17;
    const real_t x19 = x18 * (x15 * (x12 + x13 - 2 * x8 - 2 * x9) + x7);
    const real_t x20 = -x6;
    const real_t x21 = pow(x20, 2);
    const real_t x22 = x8 + x9;
    const real_t x23 = -x10 - x11 + x22;
    const real_t x24 = 60 * x14;
    const real_t x25 = (1.0 / 360.0) * x17;
    const real_t x26 = x25 * (-x21 - x23 * x24);
    const real_t x27 = -x21;
    const real_t x28 = x13 - x8 - x9;
    const real_t x29 = x25 * (x24 * x28 + x27);
    const real_t x30 = 2 * nu / (3 * x2 - 3 * x5);
    const real_t x31 = x23 * x30;
    const real_t x32 = (1.0 / 90.0) * x16 * x20;
    const real_t x33 = -x28 * x30;
    const real_t x34 = x18 * (x12 * x15 + x7);
    const real_t x35 = x25 * (x22 * x24 + x27);
    const real_t x36 = -x22 * x30;
    const real_t x37 = x18 * (x13 * x15 + x7);
    const real_t x38 = (4.0 / 45.0) * x17 * (15 * x14 * (x12 + x28) + x7);
    const real_t x39 = -x7;
    const real_t x40 = (2.0 / 45.0) * x17;
    const real_t x41 = x40 * (-x15 * x28 - x39);
    const real_t x42 = x40 * (-x15 * x22 - x39);
    const real_t x43 = x40 * (x15 * x23 + x7);
    element_matrix[0] = x19;
    element_matrix[1] = x26;
    element_matrix[2] = x29;
    element_matrix[3] = x31;
    element_matrix[4] = x32;
    element_matrix[5] = x33;
    element_matrix[6] = 0;
    element_matrix[7] = 0;
    element_matrix[8] = 0;
    element_matrix[9] = 0;
    element_matrix[10] = 0;
    element_matrix[11] = 0;
    element_matrix[12] = x26;
    element_matrix[13] = x34;
    element_matrix[14] = x35;
    element_matrix[15] = x31;
    element_matrix[16] = x36;
    element_matrix[17] = x32;
    element_matrix[18] = 0;
    element_matrix[19] = 0;
    element_matrix[20] = 0;
    element_matrix[21] = 0;
    element_matrix[22] = 0;
    element_matrix[23] = 0;
    element_matrix[24] = x29;
    element_matrix[25] = x35;
    element_matrix[26] = x37;
    element_matrix[27] = x32;
    element_matrix[28] = x36;
    element_matrix[29] = x33;
    element_matrix[30] = 0;
    element_matrix[31] = 0;
    element_matrix[32] = 0;
    element_matrix[33] = 0;
    element_matrix[34] = 0;
    element_matrix[35] = 0;
    element_matrix[36] = x31;
    element_matrix[37] = x31;
    element_matrix[38] = x32;
    element_matrix[39] = x38;
    element_matrix[40] = x41;
    element_matrix[41] = x42;
    element_matrix[42] = 0;
    element_matrix[43] = 0;
    element_matrix[44] = 0;
    element_matrix[45] = 0;
    element_matrix[46] = 0;
    element_matrix[47] = 0;
    element_matrix[48] = x32;
    element_matrix[49] = x36;
    element_matrix[50] = x36;
    element_matrix[51] = x41;
    element_matrix[52] = x38;
    element_matrix[53] = x43;
    element_matrix[54] = 0;
    element_matrix[55] = 0;
    element_matrix[56] = 0;
    element_matrix[57] = 0;
    element_matrix[58] = 0;
    element_matrix[59] = 0;
    element_matrix[60] = x33;
    element_matrix[61] = x32;
    element_matrix[62] = x33;
    element_matrix[63] = x42;
    element_matrix[64] = x43;
    element_matrix[65] = x38;
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
    element_matrix[78] = x19;
    element_matrix[79] = x26;
    element_matrix[80] = x29;
    element_matrix[81] = x31;
    element_matrix[82] = x32;
    element_matrix[83] = x33;
    element_matrix[84] = 0;
    element_matrix[85] = 0;
    element_matrix[86] = 0;
    element_matrix[87] = 0;
    element_matrix[88] = 0;
    element_matrix[89] = 0;
    element_matrix[90] = x26;
    element_matrix[91] = x34;
    element_matrix[92] = x35;
    element_matrix[93] = x31;
    element_matrix[94] = x36;
    element_matrix[95] = x32;
    element_matrix[96] = 0;
    element_matrix[97] = 0;
    element_matrix[98] = 0;
    element_matrix[99] = 0;
    element_matrix[100] = 0;
    element_matrix[101] = 0;
    element_matrix[102] = x29;
    element_matrix[103] = x35;
    element_matrix[104] = x37;
    element_matrix[105] = x32;
    element_matrix[106] = x36;
    element_matrix[107] = x33;
    element_matrix[108] = 0;
    element_matrix[109] = 0;
    element_matrix[110] = 0;
    element_matrix[111] = 0;
    element_matrix[112] = 0;
    element_matrix[113] = 0;
    element_matrix[114] = x31;
    element_matrix[115] = x31;
    element_matrix[116] = x32;
    element_matrix[117] = x38;
    element_matrix[118] = x41;
    element_matrix[119] = x42;
    element_matrix[120] = 0;
    element_matrix[121] = 0;
    element_matrix[122] = 0;
    element_matrix[123] = 0;
    element_matrix[124] = 0;
    element_matrix[125] = 0;
    element_matrix[126] = x32;
    element_matrix[127] = x36;
    element_matrix[128] = x36;
    element_matrix[129] = x41;
    element_matrix[130] = x38;
    element_matrix[131] = x43;
    element_matrix[132] = 0;
    element_matrix[133] = 0;
    element_matrix[134] = 0;
    element_matrix[135] = 0;
    element_matrix[136] = 0;
    element_matrix[137] = 0;
    element_matrix[138] = x33;
    element_matrix[139] = x32;
    element_matrix[140] = x33;
    element_matrix[141] = x42;
    element_matrix[142] = x43;
    element_matrix[143] = x38;
}

static SFEM_INLINE void tri6_momentum_rhs_kernel(const real_t px0,
                                                 const real_t px1,
                                                 const real_t px2,
                                                 const real_t py0,
                                                 const real_t py1,
                                                 const real_t py2,
                                                 real_t *const SFEM_RESTRICT u,
                                                 real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = py0 - py1;
    const real_t x1 = u[3] * x0;
    const real_t x2 = (2.0 / 315.0) * u[1];
    const real_t x3 = -x1 * x2;
    const real_t x4 = (2.0 / 315.0) * u[4];
    const real_t x5 = px0 - px1;
    const real_t x6 = u[7] * x5;
    const real_t x7 = -x4 * x6;
    const real_t x8 = u[4] * x0;
    const real_t x9 = x2 * x8;
    const real_t x10 = (2.0 / 315.0) * u[3];
    const real_t x11 = x10 * x6;
    const real_t x12 = 4 * u[4];
    const real_t x13 = py0 - py2;
    const real_t x14 = px0 - px2;
    const real_t x15 = -x0 * x14 + x13 * x5;
    const real_t x16 = (1.0 / 360.0) * x15;
    const real_t x17 = u[5] * x13;
    const real_t x18 = u[1] * x14;
    const real_t x19 = u[2] * x5;
    const real_t x20 = u[6] * (x18 - x19);
    const real_t x21 = u[10] * x5;
    const real_t x22 = -u[10] * x14 - u[4] * x13 + x21 + x8;
    const real_t x23 = (1.0 / 210.0) * u[0];
    const real_t x24 = pow(u[2], 2);
    const real_t x25 = (1.0 / 280.0) * x0;
    const real_t x26 = pow(u[3], 2);
    const real_t x27 = x13 * x26;
    const real_t x28 = -x0;
    const real_t x29 = pow(u[5], 2);
    const real_t x30 = x28 * x29;
    const real_t x31 = -x13;
    const real_t x32 = pow(u[4], 2);
    const real_t x33 = (2.0 / 105.0) * x32;
    const real_t x34 = x26 * x28;
    const real_t x35 = (4.0 / 315.0) * x13;
    const real_t x36 = u[6] * x5;
    const real_t x37 = u[0] * x0;
    const real_t x38 = u[6] * x14;
    const real_t x39 = x37 + x38;
    const real_t x40 = -x36 + x39;
    const real_t x41 = u[0] * x31 + x40;
    const real_t x42 = u[2] * x0;
    const real_t x43 = u[1] * x42;
    const real_t x44 = u[8] * x18;
    const real_t x45 = u[10] * x14;
    const real_t x46 = (4.0 / 315.0) * u[3];
    const real_t x47 = (4.0 / 315.0) * x19;
    const real_t x48 = u[11] * x14;
    const real_t x49 = (4.0 / 315.0) * u[5];
    const real_t x50 = u[3] * x13;
    const real_t x51 = u[1] * x50;
    const real_t x52 = u[9] * x5;
    const real_t x53 = (4.0 / 315.0) * u[4];
    const real_t x54 = (2.0 / 63.0) * u[5];
    const real_t x55 = u[9] * x14;
    const real_t x56 = (2.0 / 63.0) * u[3];
    const real_t x57 = u[11] * x5;
    const real_t x58 = (2.0 / 105.0) * u[0];
    const real_t x59 = (2.0 / 105.0) * u[5];
    const real_t x60 = x45 * x59;
    const real_t x61 = u[4] * x13;
    const real_t x62 = u[9] * x19;
    const real_t x63 = (2.0 / 315.0) * x62;
    const real_t x64 = x1 * x4;
    const real_t x65 = (2.0 / 315.0) * u[5];
    const real_t x66 = (1.0 / 105.0) * x36;
    const real_t x67 = (1.0 / 105.0) * x38;
    const real_t x68 = (1.0 / 126.0) * u[10];
    const real_t x69 = (1.0 / 126.0) * u[1];
    const real_t x70 = (1.0 / 126.0) * u[5];
    const real_t x71 = x6 * x70;
    const real_t x72 = (1.0 / 280.0) * x37;
    const real_t x73 = u[7] * x14;
    const real_t x74 = (1.0 / 280.0) * u[0];
    const real_t x75 = (1.0 / 280.0) * x14;
    const real_t x76 = u[0] * u[8];
    const real_t x77 = (1.0 / 280.0) * x19;
    const real_t x78 = u[8] * x77;
    const real_t x79 = u[8] * x5;
    const real_t x80 = (1.0 / 630.0) * x79;
    const real_t x81 = (1.0 / 630.0) * x73;
    const real_t x82 = u[5] * x0;
    const real_t x83 = x69 * x82;
    const real_t x84 = (1.0 / 126.0) * u[2];
    const real_t x85 = u[2] * x72;
    const real_t x86 = u[0] * x13;
    const real_t x87 = u[2] * x86;
    const real_t x88 = (1.0 / 280.0) * x5;
    const real_t x89 = u[0] * u[7];
    const real_t x90 = (2.0 / 105.0) * u[4];
    const real_t x91 = x45 * x90;
    const real_t x92 = (2.0 / 315.0) * u[2];
    const real_t x93 = (4.0 / 105.0) * u[5];
    const real_t x94 = u[9] * x18;
    const real_t x95 = u[2] * x82;
    const real_t x96 = (8.0 / 315.0) * u[5];
    const real_t x97 = x52 * x96;
    const real_t x98 = u[1] * x13;
    const real_t x99 = u[2] * x98;
    const real_t x100 = u[7] * x19;
    const real_t x101 = (2.0 / 105.0) * x21;
    const real_t x102 = u[8] * x14;
    const real_t x103 = (1.0 / 126.0) * u[3];
    const real_t x104 =
        u[3] * x101 - u[4] * x101 + x0 * x33 + x102 * x103 + x17 * x4 - x50 * x84 - x74 * x98;
    const real_t x105 = pow(u[1], 2);
    const real_t x106 = x4 * x57;
    const real_t x107 = x55 * x65;
    const real_t x108 = (2.0 / 315.0) * u[11];
    const real_t x109 = x108 * x18;
    const real_t x110 = (1.0 / 280.0) * u[7] * x18 + x10 * x57 + (1.0 / 280.0) * x105 * x31 - x106 -
                        x107 + x109 + x4 * x55;
    const real_t x111 = u[3] * x48;
    const real_t x112 = x61 * x92;
    const real_t x113 = x102 * x65;
    const real_t x114 = x102 * x4 - 8.0 / 315.0 * x111 - x112 - x113 + x17 * x92;
    const real_t x115 = -x73 + x98;
    const real_t x116 = pow(u[0], 2);
    const real_t x117 = x17 + x48;
    const real_t x118 = (1.0 / 210.0) * u[1];
    const real_t x119 = 4 * u[5];
    const real_t x120 = (1.0 / 21.0) * u[1];
    const real_t x121 = (1.0 / 126.0) * u[0];
    const real_t x122 = u[11] * x19;
    const real_t x123 = u[1] * x37;
    const real_t x124 = (1.0 / 140.0) * u[0];
    const real_t x125 = x38 * x74;
    const real_t x126 = (1.0 / 630.0) * x38;
    const real_t x127 = (2.0 / 105.0) * u[10];
    const real_t x128 = x36 * x4;
    const real_t x129 = x4 * x79;
    const real_t x130 = (4.0 / 315.0) * u[0];
    const real_t x131 = (11.0 / 2520.0) * u[0];
    const real_t x132 = (2.0 / 315.0) * u[0];
    const real_t x133 = x132 * x21;
    const real_t x134 = x132 * x45;
    const real_t x135 = -x133 + x134;
    const real_t x136 = x135 + x63;
    const real_t x137 = u[3] * x52;
    const real_t x138 = (2.0 / 105.0) * x137 + x48 * x59 - x48 * x90 - x52 * x90;
    const real_t x139 = 4 * u[3];
    const real_t x140 = x1 + x52;
    const real_t x141 = (1.0 / 210.0) * u[2];
    const real_t x142 = (1.0 / 280.0) * x116;
    const real_t x143 = x0 * x32;
    const real_t x144 = u[2] * x28 + x79;
    const real_t x145 = (11.0 / 2520.0) * u[6];
    const real_t x146 = u[1] * x61;
    const real_t x147 = (1.0 / 21.0) * u[2];
    const real_t x148 = (1.0 / 21.0) * x102;
    const real_t x149 = u[2] * x8;
    const real_t x150 = (1.0 / 105.0) * x79;
    const real_t x151 = u[10] * x18;
    const real_t x152 = x38 * x4 - x38 * x65 + x4 * x73 - x65 * x73;
    const real_t x153 = x11 - x21 * x96 + x3 + x7 + x9;
    const real_t x154 = (1.0 / 90.0) * x15;
    const real_t x155 = (2.0 / 105.0) * x116;
    const real_t x156 = x28 * x32;
    const real_t x157 = x0 * x26;
    const real_t x158 = x13 * x32;
    const real_t x159 = x29 * x31;
    const real_t x160 = (16.0 / 315.0) * u[4];
    const real_t x161 = (16.0 / 315.0) * u[5];
    const real_t x162 = x1 * x161;
    const real_t x163 = x161 * x8;
    const real_t x164 = (8.0 / 105.0) * x21;
    const real_t x165 = (8.0 / 105.0) * u[4];
    const real_t x166 = x1 * x165;
    const real_t x167 = u[0] * x17;
    const real_t x168 = (8.0 / 315.0) * u[3];
    const real_t x169 = (4.0 / 63.0) * u[0];
    const real_t x170 = (2.0 / 63.0) * u[0];
    const real_t x171 = (2.0 / 315.0) * x19;
    const real_t x172 = x6 * x65;
    const real_t x173 = (1.0 / 630.0) * x37;
    const real_t x174 = x58 * x8;
    const real_t x175 = (2.0 / 105.0) * u[2];
    const real_t x176 = x1 * x175;
    const real_t x177 = (2.0 / 105.0) * u[3];
    const real_t x178 = x37 * x65;
    const real_t x179 = x2 * x82;
    const real_t x180 = x82 * x92;
    const real_t x181 = (4.0 / 315.0) * x5;
    const real_t x182 = (4.0 / 315.0) * x17;
    const real_t x183 = u[6] * x18;
    const real_t x184 = x57 * x96;
    const real_t x185 = (16.0 / 105.0) * u[4];
    const real_t x186 = u[0] * x1;
    const real_t x187 = (32.0 / 315.0) * u[5];
    const real_t x188 = (2.0 / 105.0) * x0;
    const real_t x189 = (16.0 / 315.0) * u[3];
    const real_t x190 = -x189 * x57;
    const real_t x191 = -x160 * x55;
    const real_t x192 = x160 * x57;
    const real_t x193 = x161 * x55;
    const real_t x194 = x116 * x188 + x190 + x191 + x192 + x193 - x36 * x58 + x38 * x58;
    const real_t x195 = (2.0 / 105.0) * x13;
    const real_t x196 = (2.0 / 105.0) * u[1];
    const real_t x197 = (2.0 / 105.0) * u[7];
    const real_t x198 = (2.0 / 105.0) * x5;
    const real_t x199 = u[7] * x198;
    const real_t x200 =
        -u[3] * x199 + u[4] * x199 + x1 * x196 + x105 * x195 - x109 - x18 * x197 - x196 * x8;
    const real_t x201 = (1.0 / 630.0) * u[0];
    const real_t x202 = x189 * x61;
    const real_t x203 = x17 * x189;
    const real_t x204 = x165 * x17;
    const real_t x205 = (4.0 / 105.0) * u[0];
    const real_t x206 = (2.0 / 63.0) * u[9];
    const real_t x207 = x10 * x102;
    const real_t x208 = (2.0 / 63.0) * u[11];
    const real_t x209 = x58 * x61;
    const real_t x210 = x17 * x196;
    const real_t x211 = x132 * x50;
    const real_t x212 = (2.0 / 315.0) * x51;
    const real_t x213 = x50 * x92;
    const real_t x214 = u[10] * x19;
    const real_t x215 = (8.0 / 105.0) * u[5];
    const real_t x216 = x168 * x55;
    const real_t x217 = (16.0 / 105.0) * u[5];
    const real_t x218 = (2.0 / 105.0) * u[8];
    const real_t x219 = x102 * x59 - x102 * x90 - x17 * x175 + x175 * x61 + x19 * x218;
    const real_t x220 = (4.0 / 315.0) * x14;
    const real_t x221 = (4.0 / 315.0) * x1;
    const real_t x222 = (4.0 / 105.0) * u[11];
    const real_t x223 = pow(u[11], 2);
    const real_t x224 = pow(u[9], 2);
    const real_t x225 = x14 * x224;
    const real_t x226 = pow(u[10], 2);
    const real_t x227 = (2.0 / 105.0) * x226;
    const real_t x228 = x227 * x5;
    const real_t x229 = u[7] * x13;
    const real_t x230 = u[8] * x0;
    const real_t x231 = x229 - x230;
    const real_t x232 = pow(u[8], 2);
    const real_t x233 = 4 * u[10];
    const real_t x234 = (1.0 / 210.0) * u[6];
    const real_t x235 = pow(u[7], 2);
    const real_t x236 = -x86;
    const real_t x237 = u[1] * x230;
    const real_t x238 = (11.0 / 2520.0) * u[8];
    const real_t x239 = (4.0 / 315.0) * u[11];
    const real_t x240 = (4.0 / 315.0) * u[8];
    const real_t x241 = (4.0 / 315.0) * u[7];
    const real_t x242 = (2.0 / 105.0) * u[6];
    const real_t x243 = (2.0 / 105.0) * u[9];
    const real_t x244 = x243 * x8;
    const real_t x245 = x108 * x45;
    const real_t x246 = (2.0 / 315.0) * x21;
    const real_t x247 = (2.0 / 315.0) * u[9];
    const real_t x248 = (2.0 / 315.0) * u[7];
    const real_t x249 = x17 * x248;
    const real_t x250 = -x249;
    const real_t x251 = (1.0 / 105.0) * u[10];
    const real_t x252 = (1.0 / 105.0) * u[11];
    const real_t x253 = u[9] * x13 * x84;
    const real_t x254 = (1.0 / 126.0) * u[7];
    const real_t x255 = (1.0 / 126.0) * u[8];
    const real_t x256 = u[1] * u[6];
    const real_t x257 = (1.0 / 280.0) * x98;
    const real_t x258 = u[7] * x257;
    const real_t x259 = (1.0 / 280.0) * u[6];
    const real_t x260 = (1.0 / 280.0) * u[8];
    const real_t x261 = (1.0 / 630.0) * u[10];
    const real_t x262 = u[11] * x98;
    const real_t x263 = x255 * x55;
    const real_t x264 = u[2] * x13;
    const real_t x265 = (1.0 / 280.0) * u[7];
    const real_t x266 = x265 * x38;
    const real_t x267 = u[9] * x42;
    const real_t x268 = x127 * x8;
    const real_t x269 = (2.0 / 105.0) * u[11];
    const real_t x270 = (4.0 / 105.0) * u[9];
    const real_t x271 = u[9] * x61;
    const real_t x272 = (8.0 / 315.0) * u[9];
    const real_t x273 = u[2] * x229;
    const real_t x274 = u[7] * x79;
    const real_t x275 = x13 * x92;
    const real_t x276 = u[10] * x275;
    const real_t x277 = (2.0 / 315.0) * u[10];
    const real_t x278 = x277 * x50;
    const real_t x279 = (2.0 / 315.0) * u[8];
    const real_t x280 = x279 * x48;
    const real_t x281 = x247 * x82;
    const real_t x282 = x1 * x279;
    const real_t x283 = u[11] * x275 + x108 * x50 + x260 * x42 - x276 + x277 * x82 - x278 +
                        x279 * x45 - x280 - x281 + x282;
    const real_t x284 = x0 * x69;
    const real_t x285 =
        u[11] * x284 + u[9] * x246 - x127 * x61 + x14 * x227 - x254 * x57 - x260 * x36 + x269 * x61;
    const real_t x286 = (8.0 / 315.0) * u[11];
    const real_t x287 = u[7] * x246;
    const real_t x288 = x0 * x2;
    const real_t x289 = u[9] * x288;
    const real_t x290 = u[10] * x288 - x1 * x286 + x248 * x52 - x287 - x289;
    const real_t x291 = 4 * u[11];
    const real_t x292 = (1.0 / 210.0) * u[7];
    const real_t x293 = pow(u[6], 2);
    const real_t x294 = -x14;
    const real_t x295 = -x5;
    const real_t x296 = u[7] * x294 + x98;
    const real_t x297 = (8.0 / 315.0) * x271;
    const real_t x298 = (4.0 / 315.0) * u[10];
    const real_t x299 = (4.0 / 315.0) * u[9];
    const real_t x300 = x0 * x120;
    const real_t x301 = (1.0 / 21.0) * u[7];
    const real_t x302 = u[7] * x45;
    const real_t x303 = (1.0 / 126.0) * u[6];
    const real_t x304 = (1.0 / 126.0) * u[11];
    const real_t x305 = x0 * x256;
    const real_t x306 = u[6] * x72;
    const real_t x307 = (1.0 / 140.0) * u[6];
    const real_t x308 = u[7] * x36;
    const real_t x309 = (4.0 / 315.0) * u[6];
    const real_t x310 = (2.0 / 315.0) * u[6];
    const real_t x311 = x310 * x8;
    const real_t x312 = x310 * x61;
    const real_t x313 = -x311 + x312;
    const real_t x314 = -x1 * x127 + x1 * x243 - x127 * x17 + x17 * x269;
    const real_t x315 = -x247 * x37 - x247 * x42 + x277 * x37 + x277 * x42;
    const real_t x316 = 4 * u[9];
    const real_t x317 = -x79;
    const real_t x318 = u[8] * (x317 + x42);
    const real_t x319 = x311 - x312;
    const real_t x320 = -2.0 / 315.0 * u[0] * u[11] * x13 - 2.0 / 315.0 * u[11] * u[1] * x13 +
                        x277 * x86 + x277 * x98;
    const real_t x321 = (2.0 / 105.0) * x293;
    const real_t x322 = (2.0 / 105.0) * x235;
    const real_t x323 = (16.0 / 105.0) * x5;
    const real_t x324 = (2.0 / 63.0) * u[6];
    const real_t x325 = (2.0 / 63.0) * u[7];
    const real_t x326 = x242 * x86;
    const real_t x327 = u[1] * x188;
    const real_t x328 = u[10] * x327;
    const real_t x329 = x197 * x52;
    const real_t x330 = (4.0 / 63.0) * u[6];
    const real_t x331 = (16.0 / 315.0) * u[10];
    const real_t x332 = x331 * x82;
    const real_t x333 = (16.0 / 315.0) * u[11];
    const real_t x334 = x333 * x50;
    const real_t x335 = (16.0 / 315.0) * u[9];
    const real_t x336 = (32.0 / 315.0) * u[11];
    const real_t x337 = -u[11] * x288 + u[6] * x101 - u[9] * x164 + x108 * x36 - x21 * x333 +
                        x218 * x52 + x248 * x57 + x279 * x57 + x286 * x82 - x333 * x52;
    const real_t x338 = x14 * x223;
    const real_t x339 = x224 * x295;
    const real_t x340 = (16.0 / 105.0) * u[10];
    const real_t x341 = (16.0 / 105.0) * u[11];
    const real_t x342 = (8.0 / 105.0) * u[10];
    const real_t x343 = (8.0 / 105.0) * u[11];
    const real_t x344 = x272 * x50;
    const real_t x345 = (4.0 / 63.0) * u[8];
    const real_t x346 = (4.0 / 105.0) * u[6];
    const real_t x347 = (2.0 / 63.0) * u[8];
    const real_t x348 = x242 * x45;
    const real_t x349 = x197 * x48;
    const real_t x350 = x247 * x38;
    const real_t x351 = x248 * x55;
    const real_t x352 = x279 * x55;
    const real_t x353 = u[9] * x275;
    const real_t x354 = x343 * x45;
    const real_t x355 = u[8] * x21;
    const real_t x356 = x335 * x45;
    const real_t x357 = x333 * x55;
    const real_t x358 = u[2] * x195;
    const real_t x359 = u[10] * x358 - u[11] * x358 + x198 * x232 - x218 * x42 - x218 * x45 +
                        x218 * x48 - x282 + x331 * x50 - x332 - x334 + x335 * x82;
    element_vector[0] =
        (13.0 / 420.0) * u[0] * x41 + (1.0 / 35.0) * u[0] * (x1 - x17) - u[11] * x47 - u[1] * x72 -
        4.0 / 105.0 * u[3] * x38 - u[3] * x66 - u[3] * x80 + u[4] * x66 - u[4] * x67 + u[4] * x80 -
        u[4] * x81 + u[5] * x67 + u[5] * x81 - x1 * x54 + x1 * x84 + x10 * x61 +
        (11.0 / 2520.0) * x100 + x104 + x11 + x110 + x114 - x16 * (-6 * u[0] + u[1] + u[2] + x12) +
        x17 * x56 - x17 * x69 + x18 * x68 - x19 * x68 - x2 * x61 + (1.0 / 140.0) * x20 + x21 * x49 -
        x22 * x23 + x24 * x25 + (2.0 / 63.0) * x27 + x29 * x35 + x3 + (2.0 / 63.0) * x30 +
        x31 * x33 + (4.0 / 315.0) * x34 + x36 * x93 - x37 * x59 - 11.0 / 2520.0 * x43 -
        11.0 / 2520.0 * x44 - x45 * x46 + x46 * x52 - x48 * x49 + x48 * x53 + x48 * x58 +
        x50 * x58 - 4.0 / 315.0 * x51 - x52 * x53 - x52 * x58 + x54 * x57 - x55 * x56 + x55 * x58 -
        x57 * x58 - x60 - x63 - x64 - x65 * x8 + x7 - x71 - x73 * x74 + x74 * x79 - x75 * x76 -
        x78 + x8 * x92 + x83 + x85 + (1.0 / 280.0) * x87 + x88 * x89 + x9 + x91 +
        (4.0 / 315.0) * x94 + (4.0 / 315.0) * x95 + x97 + (11.0 / 2520.0) * x99;
    element_vector[1] = (1.0 / 126.0) * u[0] * u[11] * x5 + (4.0 / 315.0) * u[0] * u[3] * x13 +
                        (1.0 / 126.0) * u[0] * u[4] * x13 + (2.0 / 315.0) * u[0] * u[5] * x13 +
                        (1.0 / 280.0) * u[0] * u[6] * x5 + (1.0 / 140.0) * u[0] * u[7] * x5 +
                        (11.0 / 2520.0) * u[0] * u[8] * x14 + (4.0 / 315.0) * u[0] * u[9] * x5 +
                        (8.0 / 315.0) * u[10] * u[3] * x14 + (4.0 / 315.0) * u[10] * u[4] * x14 +
                        (2.0 / 315.0) * u[10] * u[5] * x5 - u[10] * x47 +
                        (4.0 / 315.0) * u[11] * u[3] * x14 + (2.0 / 315.0) * u[11] * u[3] * x5 +
                        (1.0 / 140.0) * u[1] * u[2] * x0 + (1.0 / 21.0) * u[1] * u[3] * x0 +
                        (1.0 / 35.0) * u[1] * u[4] * x13 + (1.0 / 280.0) * u[1] * u[6] * x14 +
                        (1.0 / 280.0) * u[1] * u[8] * x14 + (13.0 / 420.0) * u[1] * x115 +
                        (2.0 / 105.0) * u[2] * u[4] * x0 + (2.0 / 315.0) * u[2] * u[5] * x13 +
                        (11.0 / 2520.0) * u[2] * u[6] * x5 + (2.0 / 315.0) * u[3] * u[6] * x5 +
                        (4.0 / 105.0) * u[3] * u[7] * x14 + (2.0 / 315.0) * u[3] * u[8] * x5 -
                        1.0 / 21.0 * u[3] * x6 + (1.0 / 21.0) * u[4] * u[7] * x5 +
                        (2.0 / 315.0) * u[4] * u[8] * x14 + (2.0 / 315.0) * u[4] * u[9] * x14 -
                        1.0 / 105.0 * u[4] * x73 + (1.0 / 105.0) * u[5] * u[7] * x14 +
                        (1.0 / 126.0) * u[5] * u[8] * x5 + (1.0 / 280.0) * x0 * x24 +
                        (2.0 / 105.0) * x0 * x26 - x1 * x58 - x10 * x17 - 1.0 / 140.0 * x100 -
                        x104 - x106 - x107 - x112 - x113 + (1.0 / 280.0) * x116 * x13 - x116 * x25 -
                        x117 * x118 - x120 * x8 - x121 * x48 - 1.0 / 126.0 * x122 -
                        1.0 / 140.0 * x123 - x124 * x73 - x125 - x126 * (u[4] - u[5]) - x127 * x18 -
                        x128 - x129 + (2.0 / 105.0) * x13 * x29 - x130 * x55 - x131 * x79 - x136 -
                        x138 - x16 * (u[0] - 6 * u[1] + u[2] + x119) - x32 * x35 - x36 * x70 -
                        x45 * x49 - 2.0 / 105.0 * x51 - x52 * x65 - x56 * (x50 - x55 + x61) - x78 -
                        11.0 / 2520.0 * x87 - 2.0 / 105.0 * x94 - 1.0 / 280.0 * x99;
    element_vector[2] =
        (13.0 / 420.0) * u[2] * x144 - u[3] * x150 - u[4] * x148 + u[4] * x150 + u[5] * x148 -
        u[6] * x77 - u[7] * x77 - x1 * x132 + x1 * x65 - x10 * x45 - x102 * x124 + x103 * x38 -
        x103 * x73 + x110 + (2.0 / 315.0) * x111 + x121 * x52 - x121 * x55 - x121 * x8 +
        (2.0 / 105.0) * x122 + (11.0 / 2520.0) * x123 + x124 * x79 - x125 + x127 * x19 +
        x13 * x142 + x13 * x33 - x130 * x48 + x130 * x57 - x131 * x6 + x131 * x73 + x133 - x134 +
        x138 + x140 * x141 + x142 * x28 + (4.0 / 315.0) * x143 - x145 * x18 - 2.0 / 105.0 * x146 -
        x147 * x17 + x147 * x61 - 1.0 / 35.0 * x149 + (4.0 / 315.0) * x151 + x152 + x153 -
        x16 * (u[0] + u[1] - 6 * u[2] + x139) + x17 * x58 + x21 * x46 - x21 * x53 +
        (2.0 / 105.0) * x29 * x31 + (2.0 / 105.0) * x34 + x36 * x74 +
        (1.0 / 630.0) * x36 * (-u[3] + u[4]) - x37 * x49 + (1.0 / 280.0) * x43 +
        (1.0 / 140.0) * x44 - x49 * x52 + x54 * (-x57 + x8 + x82) + x60 + x64 + x71 - x79 * x93 -
        x83 - x85 + (1.0 / 140.0) * x87 - x91 + (1.0 / 126.0) * x94 + (2.0 / 105.0) * x95 -
        1.0 / 140.0 * x99;
    element_vector[3] =
        u[10] * x171 + u[1] * x182 - u[2] * x173 - u[3] * x164 + u[4] * x164 - u[6] * x171 -
        2.0 / 315.0 * x100 - x102 * x121 + x108 * x19 + x114 + x121 * x79 - 4.0 / 315.0 * x123 -
        x130 * x61 - x130 * x73 + x135 - 16.0 / 105.0 * x137 + x141 * x144 + (8.0 / 315.0) * x146 +
        (4.0 / 315.0) * x149 - 2.0 / 63.0 * x151 + x152 + x154 * (-u[2] + 8 * u[3] + x119 + x12) +
        x155 * x31 + (8.0 / 105.0) * x156 + (16.0 / 105.0) * x157 + (16.0 / 315.0) * x158 +
        (16.0 / 315.0) * x159 - x160 * x45 - x160 * x48 + x161 * x45 + x161 * x48 - x162 - x163 -
        x166 - 8.0 / 315.0 * x167 - x168 * x17 + x168 * x45 + x168 * x61 - x169 * x52 + x169 * x55 +
        x170 * x48 - x170 * x50 - x170 * x57 - x172 + x174 + x176 + x177 * x79 + x178 + x179 +
        x180 + x181 * x89 + (4.0 / 315.0) * x183 + x184 + x185 * x52 + (16.0 / 315.0) * x186 +
        x187 * x52 + x194 + x200 + (8.0 / 315.0) * x30 + x36 * x46 - x36 * x53 + x36 * x96 -
        x38 * x56 + (2.0 / 315.0) * x43 + (1.0 / 126.0) * x44 - x49 * x79 + (2.0 / 63.0) * x51 +
        x56 * x73 - 4.0 / 105.0 * x62 - x79 * x90 + x84 * x86 - x84 * x98 - 4.0 / 63.0 * x94;
    element_vector[4] =
        -16.0 / 105.0 * u[3] * x21 + (32.0 / 315.0) * u[3] * x45 - u[7] * x47 +
        (8.0 / 315.0) * x0 * x29 - x1 * x130 - x102 * x132 + x130 * x17 + x132 * x48 - x132 * x52 +
        x132 * x55 - x132 * x57 + x132 * x6 - x132 * x73 + x132 * x79 - 8.0 / 105.0 * x137 +
        (16.0 / 315.0) * x146 - 16.0 / 315.0 * x149 - 4.0 / 63.0 * x151 +
        x154 * (-u[0] + 8 * u[4] + x119 + x139) + (16.0 / 105.0) * x156 + (8.0 / 105.0) * x157 +
        (16.0 / 105.0) * x158 + (8.0 / 105.0) * x159 + x162 + x163 - x165 * x48 + x165 * x52 +
        x166 + x168 * x73 + x172 - x174 - x176 + x177 * x36 - x178 - x179 - x18 * x206 - x180 -
        x184 + x185 * x21 - x185 * x45 - x187 * x21 + x19 * x208 + x190 + x191 + x192 + x193 -
        x2 * x37 + (1.0 / 126.0) * x20 + x200 + x201 * (u[2] * x0 - x98) - x202 - x203 - x204 +
        x205 * x21 - x205 * x45 - x207 + x209 + x210 + x211 + x212 + x213 + (4.0 / 63.0) * x214 +
        x215 * x48 + x216 + x217 * x45 + x219 + x23 * x41 + (2.0 / 105.0) * x24 * x28 +
        (8.0 / 315.0) * x26 * x31 + x36 * x49 - x36 * x90 - x38 * x46 - x38 * x59 + x38 * x90 +
        (4.0 / 315.0) * x43 + (4.0 / 315.0) * x44 + x46 * x79 - x49 * x73 + x53 * x73 - x53 * x79 +
        x63 - x79 * x96 + x86 * x92 - 4.0 / 315.0 * x99;
    element_vector[5] =
        -u[2] * x221 - u[6] * x47 + x1 * x96 + x10 * x36 + x10 * x79 - 1.0 / 126.0 * x100 -
        32.0 / 315.0 * x111 + x115 * x118 + x121 * x6 - x121 * x73 + (4.0 / 63.0) * x122 - x128 -
        x129 - x13 * x155 - 16.0 / 105.0 * x13 * x29 + x130 * x79 + x130 * x8 + x136 -
        16.0 / 315.0 * x137 - 16.0 / 315.0 * x143 - 4.0 / 315.0 * x146 - 8.0 / 315.0 * x149 -
        2.0 / 315.0 * x151 + x153 + x154 * (-u[1] + 8 * u[5] + x12 + x139) + (16.0 / 315.0) * x157 +
        (8.0 / 105.0) * x158 + x160 * x21 + x160 * x52 - x165 * x45 - 16.0 / 315.0 * x167 -
        x168 * x38 + x169 * x48 - x169 * x57 - x170 * x52 + x170 * x55 + x18 * x222 +
        (2.0 / 315.0) * x183 - x185 * x48 + (8.0 / 315.0) * x186 - x188 * x24 - x189 * x21 + x194 +
        x201 * x98 + x202 + x203 + x204 + x207 - x209 - x210 - x211 - x212 - x213 +
        (2.0 / 63.0) * x214 + x215 * x45 - x216 + x217 * x48 + x219 - x220 * x76 +
        (8.0 / 315.0) * x27 + x36 * x54 + x37 * x54 - x37 * x69 - x38 * x49 + x38 * x53 +
        (1.0 / 126.0) * x43 + (2.0 / 315.0) * x44 + x46 * x73 - x54 * x79 - x59 * x73 + x73 * x90 -
        x8 * x96 + (4.0 / 315.0) * x87 - x92 * x98 - 2.0 / 315.0 * x94 - 2.0 / 63.0 * x95 + x97;
    element_vector[6] =
        -u[10] * x182 + u[10] * x221 + u[11] * x182 + u[6] * x257 +
        (13.0 / 420.0) * u[6] * (x236 + x40) + (1.0 / 35.0) * u[6] * (x48 - x52) - u[8] * x246 -
        u[9] * x221 + (1.0 / 105.0) * u[9] * x37 + x1 * x242 + x108 * x21 - x124 * x231 -
        x16 * (-6 * u[6] + u[7] + u[8] + x233) - x17 * x242 + x17 * x272 + x181 * x224 +
        x206 * x50 + x208 * x52 - x208 * x55 - x208 * x82 + x22 * x234 - x220 * x223 - x222 * x37 +
        (2.0 / 63.0) * x223 * x5 - 2.0 / 63.0 * x225 - x228 - x232 * x88 + x235 * x75 -
        11.0 / 2520.0 * x237 - x238 * x73 - x239 * x8 - x240 * x57 + x240 * x82 - x241 * x50 +
        x241 * x55 - x242 * x50 + x242 * x82 - x243 * x38 - x244 - x245 - x247 * x45 + x248 * x45 -
        x25 * x256 + x250 - x251 * x37 + x251 * x86 - x252 * x86 - x253 + x254 * x48 - x254 * x61 -
        x255 * x52 + x255 * x8 - x258 + x259 * x264 - x259 * x42 - x260 * x38 - x261 * x42 +
        x261 * x98 - 1.0 / 630.0 * x262 + x263 + x265 * x36 + x266 + (1.0 / 630.0) * x267 + x268 +
        x269 * x36 + x270 * x86 + (4.0 / 315.0) * x271 + (11.0 / 2520.0) * x273 +
        (11.0 / 2520.0) * x274 + x283 + x285 + x290;
    element_vector[7] =
        -u[10] * x300 - u[6] * x221 + (13.0 / 420.0) * u[7] * x296 - u[8] * x101 - u[9] * x182 +
        u[9] * x300 + x1 * x108 - x108 * x38 + x108 * x55 - x108 * x8 + x117 * x292 - x145 * x264 +
        x145 * x42 - x16 * (u[6] - 6 * u[7] + u[8] + x291) + x17 * x303 + x197 * x50 + x197 * x55 +
        x197 * x61 + x206 * (x45 - x50 + x55) + x21 * x301 + x220 * x226 +
        (2.0 / 105.0) * x223 * x294 + (2.0 / 105.0) * x224 * x295 + x228 - x229 * x74 +
        (1.0 / 280.0) * x232 * x295 + (1.0 / 140.0) * x237 - x238 * x37 + x238 * x38 + x239 * x61 +
        x240 * x8 + x243 * x36 + x244 + x245 + x251 * x98 - x252 * x98 + x253 + x255 * x82 +
        x259 * x86 + x260 * x73 - x263 - x266 - x268 - x270 * x98 - 1.0 / 280.0 * x273 -
        1.0 / 140.0 * x274 + x283 + (1.0 / 280.0) * x293 * x294 + x293 * x88 - x297 - x298 * x61 -
        x299 * x38 - x301 * x52 - 1.0 / 35.0 * x302 - x303 * x45 - x303 * x82 + x304 * x37 -
        x304 * x42 - 1.0 / 140.0 * x305 - x306 + x307 * x98 + (1.0 / 140.0) * x308 + x309 * x50 +
        x313 + x314 + x315 + (1.0 / 630.0) * x86 * (u[10] - u[11]);
    element_vector[8] =
        (1.0 / 280.0) * u[0] * u[6] * x13 + (11.0 / 2520.0) * u[0] * u[7] * x13 +
        (1.0 / 280.0) * u[0] * u[8] * x0 + (2.0 / 315.0) * u[10] * u[1] * x0 +
        (1.0 / 21.0) * u[10] * u[2] * x13 + (4.0 / 315.0) * u[10] * u[4] * x0 +
        (2.0 / 315.0) * u[10] * u[5] * x0 + (1.0 / 126.0) * u[10] * u[6] * x5 +
        (2.0 / 105.0) * u[10] * u[7] * x14 + (1.0 / 35.0) * u[10] * u[8] * x5 +
        (4.0 / 105.0) * u[11] * u[2] * x0 + (4.0 / 315.0) * u[11] * u[3] * x0 +
        (2.0 / 315.0) * u[11] * u[3] * x13 + (8.0 / 315.0) * u[11] * u[4] * x0 +
        (4.0 / 315.0) * u[11] * u[6] * x5 + (1.0 / 21.0) * u[11] * u[8] * x14 - u[11] * x13 * x147 +
        (11.0 / 2520.0) * u[1] * u[6] * x0 + (1.0 / 280.0) * u[1] * u[8] * x0 +
        (1.0 / 126.0) * u[1] * u[9] * x13 + (1.0 / 140.0) * u[2] * u[6] * x13 +
        (1.0 / 105.0) * u[2] * u[9] * x0 + (1.0 / 126.0) * u[3] * u[6] * x13 +
        (2.0 / 315.0) * u[4] * u[9] * x13 + (4.0 / 315.0) * u[5] * u[6] * x13 +
        (2.0 / 315.0) * u[6] * u[9] * x5 + (1.0 / 140.0) * u[7] * u[8] * x14 +
        (2.0 / 315.0) * u[7] * u[9] * x5 - 1.0 / 210.0 * u[8] * x140 - 1.0 / 140.0 * u[8] * x38 -
        1.0 / 21.0 * u[8] * x45 - 1.0 / 126.0 * u[9] * x86 - x1 * x303 - x108 * x52 +
        (2.0 / 105.0) * x14 * x223 + (1.0 / 280.0) * x14 * x235 - x145 * x98 -
        x16 * (u[6] + u[7] - 6 * u[8] + x316) - x17 * x247 - x173 * (u[10] - u[9]) - x181 * x226 -
        x208 * (x21 + x57 - x82) - x218 * x57 - x218 * x8 - x218 * x82 + (2.0 / 105.0) * x224 * x5 -
        x241 * x61 - x249 - x251 * x42 - x254 * x50 - x258 - x265 * x79 - x269 * x38 -
        1.0 / 140.0 * x273 - x278 - x281 - x285 - x287 - x289 + (1.0 / 280.0) * x293 * x5 -
        x293 * x75 - x299 * x8 - x306 - x307 * x42 - 11.0 / 2520.0 * x308 - x309 * x82 - x314 -
        13.0 / 420.0 * x318 - x319 - x320;
    element_vector[9] =
        (4.0 / 315.0) * u[0] * u[10] * x0 + (2.0 / 105.0) * u[0] * u[6] * x0 +
        (2.0 / 315.0) * u[0] * u[8] * x0 + (2.0 / 63.0) * u[0] * u[9] * x13 +
        (2.0 / 105.0) * u[10] * u[2] * x0 + (16.0 / 315.0) * u[10] * u[3] * x13 +
        (16.0 / 315.0) * u[10] * u[4] * x13 + (16.0 / 315.0) * u[10] * u[5] * x13 +
        (4.0 / 315.0) * u[10] * u[6] * x14 + (2.0 / 105.0) * u[10] * u[7] * x5 +
        (2.0 / 315.0) * u[10] * u[8] * x14 - 16.0 / 105.0 * u[10] * x1 - 8.0 / 105.0 * u[10] * x8 +
        (4.0 / 315.0) * u[11] * u[2] * x0 + (2.0 / 315.0) * u[11] * u[2] * x13 +
        (8.0 / 315.0) * u[11] * u[6] * x14 + (8.0 / 315.0) * u[11] * u[9] * x14 +
        (4.0 / 315.0) * u[1] * u[6] * x13 + (2.0 / 105.0) * u[1] * u[7] * x13 +
        (2.0 / 315.0) * u[1] * u[8] * x0 + (2.0 / 105.0) * u[1] * u[9] * x0 +
        (1.0 / 126.0) * u[2] * u[6] * x13 + (4.0 / 63.0) * u[3] * u[6] * x0 +
        (4.0 / 63.0) * u[3] * u[7] * x13 + (4.0 / 105.0) * u[3] * u[8] * x0 +
        (16.0 / 105.0) * u[3] * u[9] * x0 + (2.0 / 63.0) * u[4] * u[7] * x13 +
        (8.0 / 105.0) * u[4] * u[9] * x0 + (2.0 / 63.0) * u[5] * u[6] * x0 +
        (16.0 / 315.0) * u[5] * u[9] * x0 + (8.0 / 315.0) * u[5] * u[9] * x13 +
        (4.0 / 315.0) * u[6] * u[7] * x5 + (1.0 / 630.0) * u[6] * u[8] * x5 +
        (2.0 / 63.0) * u[6] * u[9] * x14 + (1.0 / 126.0) * u[7] * u[8] * x14 - x1 * x336 -
        x130 * x229 + (16.0 / 315.0) * x14 * x223 - 16.0 / 315.0 * x14 * x226 +
        (2.0 / 105.0) * x14 * x293 - x14 * x322 +
        (1.0 / 90.0) * x15 * (-u[8] + 8 * u[9] + x233 + x291) - x17 * x324 - x17 * x333 -
        x206 * x98 - x21 * x240 + (8.0 / 315.0) * x223 * x5 - x224 * x323 +
        (8.0 / 105.0) * x226 * x5 - x229 * x84 - x241 * x48 - x243 * x42 - x248 * x79 - x250 -
        x255 * x38 - x272 * x45 - x276 - x279 * x8 - x279 * x82 - x280 - x286 * x37 - x297 -
        x299 * x37 - 8.0 / 315.0 * x302 - x303 * x42 - 4.0 / 315.0 * x305 - x313 -
        1.0 / 210.0 * x318 - x320 - x321 * x5 - x325 * x55 - x326 - x328 - x329 - x330 * x50 -
        x332 - x333 * x61 - x334 - x335 * x36 - x337;
    element_vector[10] =
        u[6] * x275 - u[6] * x288 + (1.0 / 630.0) * u[6] * (x317 + x73) + u[7] * x101 +
        (4.0 / 63.0) * u[7] * x61 + (8.0 / 105.0) * u[9] * x1 + u[9] * x327 +
        (16.0 / 105.0) * u[9] * x8 + x1 * x310 - x1 * x342 - x121 * x231 + x127 * x37 - x127 * x86 +
        x154 * (8 * u[10] - u[6] + x291 + x316) - x17 * x310 + x17 * x342 - x17 * x343 +
        x197 * x98 + (8.0 / 315.0) * x223 * x295 + (8.0 / 315.0) * x225 +
        (16.0 / 105.0) * x226 * x294 + x226 * x323 + x234 * (u[6] * x295 + x236 + x39) +
        (4.0 / 315.0) * x237 - x239 * x37 - x239 * x38 + x239 * x98 + x240 * x73 - x241 * x79 -
        x243 * x37 + x248 * x36 + x249 - 4.0 / 315.0 * x267 + x269 * x86 - 32.0 / 315.0 * x271 -
        x272 * x98 - 4.0 / 315.0 * x273 - x279 * x38 + x286 * x42 + x294 * x322 + x298 * x42 -
        x298 * x98 + x299 * x36 + x299 * x86 - 16.0 / 315.0 * x302 - x310 * x42 - x310 * x50 +
        x310 * x82 + x310 * x98 + x325 * x50 - x328 - x329 + x336 * x8 + x337 +
        (8.0 / 105.0) * x338 + (8.0 / 105.0) * x339 + x340 * x61 - x340 * x8 - x341 * x61 - x344 -
        x345 * x8 + x346 * x61 - x346 * x8 - x347 * x82 - x348 - x349 - x350 - x351 - x352 + x353 +
        x354 + (16.0 / 315.0) * x355 + x356 + x357 + x359;
    element_vector[11] =
        u[2] * u[6] * x35 - u[6] * x284 - u[7] * x126 - 4.0 / 105.0 * u[7] * x17 +
        (32.0 / 315.0) * u[9] * x17 + x1 * x324 - x1 * x331 + x1 * x335 - x127 * x98 - x132 * x229 +
        x14 * x321 + x154 * (8 * u[11] - u[7] + x233 + x316) - x17 * x330 + x17 * x340 -
        x17 * x341 - x208 * x36 - x208 * x37 + x208 * x42 + x21 * x286 - x21 * x309 +
        (8.0 / 315.0) * x224 * x294 + (8.0 / 105.0) * x226 * x294 + (16.0 / 315.0) * x226 * x5 -
        x229 * x92 + x230 * x69 + x239 * x86 + x240 * x37 - x240 * x38 + x240 * x52 + x241 * x45 +
        x242 * x37 + x248 * x50 + x248 * x61 + x254 * x36 - x254 * x79 + (2.0 / 105.0) * x262 -
        x272 * x36 + x272 * x86 + x279 * x73 - x286 * x52 + x286 * x8 + x290 + x292 * x296 +
        x295 * x321 - x298 * x86 - x299 * x98 + x303 * x98 - x309 * x42 + x315 + x319 - x324 * x50 -
        x326 + x330 * x82 - x331 * x8 + x333 * x38 + x335 * x8 + (16.0 / 105.0) * x338 +
        (16.0 / 315.0) * x339 + x342 * x61 - x343 * x61 + x344 - x345 * x82 + x347 * x57 -
        x347 * x8 + x348 + x349 + x350 + x351 + x352 - x353 - x354 + (8.0 / 315.0) * x355 - x356 -
        x357 + x359;
}

// 2) Potential equation
static SFEM_INLINE void tri3_laplacian_lhs_kernel(const real_t px0,
                                                  const real_t px1,
                                                  const real_t px2,
                                                  const real_t py0,
                                                  const real_t py1,
                                                  const real_t py2,
                                                  real_t *const SFEM_RESTRICT element_matrix) {
    const real_t x0 = px0 - px1;
    const real_t x1 = px0 - px2;
    const real_t x2 = x0 * x1;
    const real_t x3 = py0 - py1;
    const real_t x4 = py0 - py2;
    const real_t x5 = x3 * x4;
    const real_t x6 = pow(x1, 2);
    const real_t x7 = pow(x4, 2);
    const real_t x8 = x6 + x7;
    const real_t x9 = pow(x0, 2) + pow(x3, 2);
    const real_t x10 = (1.0 / 2.0) / (x0 * x4 - x1 * x3);
    const real_t x11 = x2 + x5;
    const real_t x12 = x10 * (x11 - x6 - x7);
    const real_t x13 = x10 * (x2 + x5 - x9);
    const real_t x14 = -x10 * x11;
    element_matrix[0] = x10 * (-2 * x2 - 2 * x5 + x8 + x9);
    element_matrix[1] = x12;
    element_matrix[2] = x13;
    element_matrix[3] = x12;
    element_matrix[4] = x10 * x8;
    element_matrix[5] = x14;
    element_matrix[6] = x13;
    element_matrix[7] = x14;
    element_matrix[8] = x10 * x9;
}

static SFEM_INLINE void tri3_tri6_divergence_rhs_kernel(const real_t px0,
                                                        const real_t px1,
                                                        const real_t px2,
                                                        const real_t py0,
                                                        const real_t py1,
                                                        const real_t py2,
                                                        const real_t dt,
                                                        const real_t *const SFEM_RESTRICT u,
                                                        real_t *const SFEM_RESTRICT
                                                            element_vector) {
    const real_t x0 = 2 * px0;
    const real_t x1 = 2 * py0;
    const real_t x2 = (1.0 / 6.0) / dt;
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
                                                        const real_t *const SFEM_RESTRICT u,
                                                        const real_t *const SFEM_RESTRICT p,
                                                        real_t *const SFEM_RESTRICT
                                                            element_vector) {
    const real_t x0 = 4 * u[4];
    const real_t x1 = px0 - px1;
    const real_t x2 = py0 - py2;
    const real_t x3 = px0 - px2;
    const real_t x4 = py0 - py1;
    const real_t x5 = x1 * x2 - x3 * x4;
    const real_t x6 = (1.0 / 360.0) * x5;
    const real_t x7 = 4 * u[5];
    const real_t x8 = 4 * u[3];
    const real_t x9 = (1.0 / 6.0) * dt;
    const real_t x10 = x9 * (-p[0] * x2 + p[0] * x4 + p[1] * x2 - p[2] * x4);
    const real_t x11 = (1.0 / 90.0) * x5;
    const real_t x12 = 4 * u[10];
    const real_t x13 = 4 * u[11];
    const real_t x14 = 4 * u[9];
    const real_t x15 = x9 * (-p[0] * x1 + p[0] * x3 - p[1] * x3 + p[2] * x1);
    element_vector[0] = -x6 * (-6 * u[0] + u[1] + u[2] + x0);
    element_vector[1] = -x6 * (u[0] - 6 * u[1] + u[2] + x7);
    element_vector[2] = -x6 * (u[0] + u[1] - 6 * u[2] + x8);
    element_vector[3] = x10 + x11 * (-u[2] + 8 * u[3] + x0 + x7);
    element_vector[4] = x10 + x11 * (-u[0] + 8 * u[4] + x7 + x8);
    element_vector[5] = x10 + x11 * (-u[1] + 8 * u[5] + x0 + x8);
    element_vector[6] = -x6 * (-6 * u[6] + u[7] + u[8] + x12);
    element_vector[7] = -x6 * (u[6] - 6 * u[7] + u[8] + x13);
    element_vector[8] = -x6 * (u[6] + u[7] - 6 * u[8] + x14);
    element_vector[9] = x11 * (-u[8] + 8 * u[9] + x12 + x13) + x15;
    element_vector[10] = x11 * (8 * u[10] - u[6] + x13 + x14) + x15;
    element_vector[11] = x11 * (8 * u[11] - u[7] + x12 + x14) + x15;
}
