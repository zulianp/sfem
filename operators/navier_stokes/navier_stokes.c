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

void tri3_tri6_divergence(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const elems,
                          geom_t **const points,
                          const real_t dt,
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

void tri6_tri3_correction(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const elems,
                          geom_t **const points,
                          const real_t dt,
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
            real_t element_p[6];

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
                //  buffers
                element_vel,
                element_p,
                element_vector);

            for (int b = 0; b < n_vars; ++b) {
#pragma unroll(6)
                for (int edof_i = 0; edof_i < element_nnodes; ++edof_i) {
                    const idx_t dof_i = elems[edof_i][ev[edof_i]];
#pragma omp atomic update
                    values[b][dof_i] += element_vector[b * element_nnodes + edof_i];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tri6_naviers_stokes.c: tri6_explict_momentum_tentative\t%g seconds\n", tock - tick);
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
                idx_t dof = ev[enode] * n_vars;

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