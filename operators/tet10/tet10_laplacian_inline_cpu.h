#ifndef TET10_LAPLACIAN_INLINE_CPU_HPP
#define TET10_LAPLACIAN_INLINE_CPU_HPP

#include "tet4_inline_cpu.h"

static SFEM_INLINE void tet10_laplacian_trial_operand(const real_t qx,
                                                      const real_t qy,
                                                      const real_t qz,
                                                      const real_t qw,
                                                      const jacobian_t *const SFEM_RESTRICT fff,
                                                      const real_t *const SFEM_RESTRICT u,
                                                      real_t *const SFEM_RESTRICT out) {
    const real_t x0 = 4 * qx;
    const real_t x1 = 4 * qy;
    const real_t x2 = 4 * qz;
    const real_t x3 = x1 + x2;
    const real_t x4 = -u[6] * x1;
    const real_t x5 = u[0] * (x0 + x3 - 3);
    const real_t x6 = -u[7] * x2 + x5;
    const real_t x7 = u[1] * (x0 - 1) + u[4] * (-8 * qx - x3 + 4) + u[5] * x1 + u[8] * x2 + x4 + x6;
    const real_t x8 = x0 - 4;
    const real_t x9 = -u[4] * x0;
    const real_t x10 =
            u[2] * (x1 - 1) + u[5] * x0 + u[6] * (-8 * qy - x2 - x8) + u[9] * x2 + x6 + x9;
    const real_t x11 =
            u[3] * (x2 - 1) + u[7] * (-8 * qz - x1 - x8) + u[8] * x0 + u[9] * x1 + x4 + x5 + x9;
    out[0] = qw * (fff[0] * x7 + fff[1] * x10 + fff[2] * x11);
    out[1] = qw * (fff[1] * x7 + fff[3] * x10 + fff[4] * x11);
    out[2] = qw * (fff[2] * x7 + fff[4] * x10 + fff[5] * x11);
}

static SFEM_INLINE void tet10_ref_shape_grad_x(const real_t qx,
                                               const real_t qy,
                                               const real_t qz,
                                               real_t *const out) {
    const real_t x0 = 4 * qx;
    const real_t x1 = 4 * qy;
    const real_t x2 = 4 * qz;
    const real_t x3 = x1 + x2;
    out[0] = x0 + x3 - 3;
    out[1] = x0 - 1;
    out[2] = 0;
    out[3] = 0;
    out[4] = -8 * qx - x3 + 4;
    out[5] = x1;
    out[6] = -x1;
    out[7] = -x2;
    out[8] = x2;
    out[9] = 0;
}

static SFEM_INLINE void tet10_ref_shape_grad_y(const real_t qx,
                                               const real_t qy,
                                               const real_t qz,
                                               real_t *const out) {
    const real_t x0 = 4 * qy;
    const real_t x1 = 4 * qx;
    const real_t x2 = 4 * qz;
    const real_t x3 = x1 + x2;
    out[0] = x0 + x3 - 3;
    out[1] = 0;
    out[2] = x0 - 1;
    out[3] = 0;
    out[4] = -x1;
    out[5] = x1;
    out[6] = -8 * qy - x3 + 4;
    out[7] = -x2;
    out[8] = 0;
    out[9] = x2;
}

static SFEM_INLINE void tet10_ref_shape_grad_z(const real_t qx,
                                               const real_t qy,
                                               const real_t qz,
                                               real_t *const out) {
    const real_t x0 = 4 * qz;
    const real_t x1 = 4 * qx;
    const real_t x2 = 4 * qy;
    const real_t x3 = x1 + x2;
    out[0] = x0 + x3 - 3;
    out[1] = 0;
    out[2] = 0;
    out[3] = x0 - 1;
    out[4] = -x1;
    out[5] = 0;
    out[6] = -x2;
    out[7] = -8 * qz - x3 + 4;
    out[8] = x1;
    out[9] = x2;
}

static SFEM_INLINE void tet10_laplacian_apply_qp_fff(const real_t qx,
                                                     const real_t qy,
                                                     const real_t qz,
                                                     const real_t qw,
                                                     const jacobian_t *const SFEM_RESTRICT fff,
                                                     const real_t *const SFEM_RESTRICT u,
                                                     real_t *const SFEM_RESTRICT element_vector) {
    // Evaluate gradient fe function transformed with fff and scaling factors

    real_t trial_operand[3];
    tet10_laplacian_trial_operand(qx, qy, qz, qw, fff, u, trial_operand);

// TODO check perf difference of two techniques
#if 0
    real_t ref_grad[10];

    {  // X-components
        tet10_ref_shape_grad_x(qx, qy, qz, ref_grad);
#pragma unroll(10)
        for (int i = 0; i < 10; i++) {
            element_vector[i] += ref_grad[i] * trial_operand[0];
        }
    }
    {  // Y-components
        tet10_ref_shape_grad_y(qx, qy, qz, ref_grad);
#pragma unroll(10)
        for (int i = 0; i < 10; i++) {
            element_vector[i] += ref_grad[i] * trial_operand[1];
        }
    }

    {  // Z-components
        tet10_ref_shape_grad_z(qx, qy, qz, ref_grad);
#pragma unroll(10)
        for (int i = 0; i < 10; i++) {
            element_vector[i] += ref_grad[i] * trial_operand[2];
        }
    }
#else
    const real_t x0 = 4*qx;
    const real_t x1 = 4*qy;
    const real_t x2 = 4*qz;
    const real_t x3 = x1 + x2;
    const real_t x4 = x0 + x3 - 3;
    const real_t x5 = trial_operand[1]*x0;
    const real_t x6 = trial_operand[2]*x0;
    const real_t x7 = trial_operand[0]*x1;
    const real_t x8 = trial_operand[2]*x1;
    const real_t x9 = x0 - 4;
    const real_t x10 = trial_operand[0]*x2;
    const real_t x11 = trial_operand[1]*x2;
    element_vector[0] += trial_operand[0]*x4 + trial_operand[1]*x4 + trial_operand[2]*x4;
    element_vector[1] += trial_operand[0]*(x0 - 1);
    element_vector[2] += trial_operand[1]*(x1 - 1);
    element_vector[3] += trial_operand[2]*(x2 - 1);
    element_vector[4] += trial_operand[0]*(-8*qx - x3 + 4) - x5 - x6;
    element_vector[5] += x5 + x7;
    element_vector[6] += trial_operand[1]*(-8*qy - x2 - x9) - x7 - x8;
    element_vector[7] += trial_operand[2]*(-8*qz - x1 - x9) - x10 - x11;
    element_vector[8] += x10 + x6;
    element_vector[9] += x11 + x8;
#endif
}

static SFEM_INLINE void tet10_laplacian_apply_fff(const jacobian_t *const SFEM_RESTRICT fff,
                                                  const real_t *const SFEM_RESTRICT ex,
                                                  real_t *const SFEM_RESTRICT ey) {
    // Numerical quadrature
    tet10_laplacian_apply_qp_fff(0, 0, 0, 0.025, 1, fffe, ex, ey);
    tet10_laplacian_apply_qp_fff(1, 0, 0, 0.025, 1, fffe, ex, ey);
    tet10_laplacian_apply_qp_fff(0, 1, 0, 0.025, 1, fffe, ex, ey);
    tet10_laplacian_apply_qp_fff(0, 0, 1, 0.025, 1, fffe, ex, ey);

    static const real_t athird = 1. / 3;
    tet10_laplacian_apply_qp_fff(athird, athird, 0., 0.225, 1, fffe, ex, ey);
    tet10_laplacian_apply_qp_fff(athird, 0., athird, 0.225, 1, fffe, ex, ey);
    tet10_laplacian_apply_qp_fff(0., athird, athird, 0.225, 1, fffe, ex, ey);
    tet10_laplacian_apply_qp_fff(athird, athird, athird, 0.225, 1, fffe, ex, ey);
}

static SFEM_INLINE void tet10_laplacian_hessian_fff(const jacobian_t *fff, real_t *element_matrix) {
    const real_t x0 = 4 * qx;
    const real_t x1 = 4 * qy;
    const real_t x2 = 4 * qz;
    const real_t x3 = x1 + x2;
    const real_t x4 = x0 + x3 - 3;
    const real_t x5 = fff[0] * x4;
    const real_t x6 = fff[1] * x4;
    const real_t x7 = fff[2] * x4;
    const real_t x8 = x5 + x6 + x7;
    const real_t x9 = qw * x4;
    const real_t x10 = fff[3] * x4;
    const real_t x11 = fff[4] * x4;
    const real_t x12 = x10 + x11 + x6;
    const real_t x13 = fff[5] * x4;
    const real_t x14 = x11 + x13 + x7;
    const real_t x15 = x0 - 1;
    const real_t x16 = qw * x8;
    const real_t x17 = x1 - 1;
    const real_t x18 = qw * x12;
    const real_t x19 = x2 - 1;
    const real_t x20 = qw * x14;
    const real_t x21 = x0 * x18;
    const real_t x22 = x0 * x20;
    const real_t x23 = -8 * qx - x3 + 4;
    const real_t x24 = x1 * x16;
    const real_t x25 = x1 * x20;
    const real_t x26 = x0 - 4;
    const real_t x27 = -8 * qy - x2 - x26;
    const real_t x28 = x16 * x2;
    const real_t x29 = x18 * x2;
    const real_t x30 = -8 * qz - x1 - x26;
    const real_t x31 = qw * x15;
    const real_t x32 = fff[0] * qw;
    const real_t x33 = fff[1] * x17 * x31;
    const real_t x34 = fff[2] * x19 * x31;
    const real_t x35 = fff[1] * x0;
    const real_t x36 = x31 * x35;
    const real_t x37 = fff[2] * x0;
    const real_t x38 = x31 * x37;
    const real_t x39 = x15 * x32;
    const real_t x40 = x1 * x39;
    const real_t x41 = fff[2] * x1;
    const real_t x42 = x31 * x41;
    const real_t x43 = x2 * x39;
    const real_t x44 = fff[1] * x2;
    const real_t x45 = x31 * x44;
    const real_t x46 = qw * x17;
    const real_t x47 = fff[3] * qw;
    const real_t x48 = fff[4] * x19 * x46;
    const real_t x49 = x17 * x47;
    const real_t x50 = x0 * x49;
    const real_t x51 = fff[4] * x0;
    const real_t x52 = x46 * x51;
    const real_t x53 = fff[1] * x1;
    const real_t x54 = x46 * x53;
    const real_t x55 = fff[4] * x1;
    const real_t x56 = x46 * x55;
    const real_t x57 = x44 * x46;
    const real_t x58 = x2 * x49;
    const real_t x59 = qw * x19;
    const real_t x60 = fff[5] * qw;
    const real_t x61 = x51 * x59;
    const real_t x62 = x19 * x60;
    const real_t x63 = x0 * x62;
    const real_t x64 = x41 * x59;
    const real_t x65 = x1 * x62;
    const real_t x66 = fff[2] * x2;
    const real_t x67 = x59 * x66;
    const real_t x68 = fff[4] * x2;
    const real_t x69 = x59 * x68;
    const real_t x70 = fff[0] * x23 - x35 - x37;
    const real_t x71 = fff[3] * x0;
    const real_t x72 = fff[1] * x23 - x51 - x71;
    const real_t x73 = fff[5] * x0;
    const real_t x74 = fff[2] * x23 - x51 - x73;
    const real_t x75 = qw * x70;
    const real_t x76 = qw * x72;
    const real_t x77 = qw * x74;
    const real_t x78 = x0 * x76;
    const real_t x79 = x0 * x77;
    const real_t x80 = x1 * x75;
    const real_t x81 = x1 * x77;
    const real_t x82 = x2 * x75;
    const real_t x83 = x2 * x76;
    const real_t x84 = fff[0] * x1;
    const real_t x85 = x35 + x84;
    const real_t x86 = x53 + x71;
    const real_t x87 = x41 + x51;
    const real_t x88 = qw * x85;
    const real_t x89 = qw * x86;
    const real_t x90 = qw * x87;
    const real_t x91 = x0 * x89;
    const real_t x92 = x0 * x90;
    const real_t x93 = x1 * x88;
    const real_t x94 = x1 * x90;
    const real_t x95 = x2 * x88;
    const real_t x96 = x2 * x89;
    const real_t x97 = fff[1] * x27 - x41 - x84;
    const real_t x98 = fff[3] * x27 - x53 - x55;
    const real_t x99 = fff[5] * x1;
    const real_t x100 = fff[4] * x27 - x41 - x99;
    const real_t x101 = qw * x97;
    const real_t x102 = qw * x98;
    const real_t x103 = qw * x100;
    const real_t x104 = x0 * x102;
    const real_t x105 = x0 * x103;
    const real_t x106 = x1 * x101;
    const real_t x107 = x1 * x103;
    const real_t x108 = x101 * x2;
    const real_t x109 = x102 * x2;
    const real_t x110 = fff[0] * x2;
    const real_t x111 = fff[2] * x30 - x110 - x44;
    const real_t x112 = fff[3] * x2;
    const real_t x113 = fff[4] * x30 - x112 - x44;
    const real_t x114 = fff[5] * x30 - x66 - x68;
    const real_t x115 = qw * x111;
    const real_t x116 = qw * x113;
    const real_t x117 = qw * x114;
    const real_t x118 = x0 * x116;
    const real_t x119 = x0 * x117;
    const real_t x120 = x1 * x115;
    const real_t x121 = x1 * x117;
    const real_t x122 = x115 * x2;
    const real_t x123 = x116 * x2;
    const real_t x124 = x110 + x37;
    const real_t x125 = x44 + x51;
    const real_t x126 = x66 + x73;
    const real_t x127 = qw * x124;
    const real_t x128 = qw * x125;
    const real_t x129 = qw * x126;
    const real_t x130 = x0 * x128;
    const real_t x131 = x0 * x129;
    const real_t x132 = x1 * x127;
    const real_t x133 = x1 * x129;
    const real_t x134 = x127 * x2;
    const real_t x135 = x128 * x2;
    const real_t x136 = x41 + x44;
    const real_t x137 = x112 + x55;
    const real_t x138 = x68 + x99;
    const real_t x139 = qw * x136;
    const real_t x140 = qw * x137;
    const real_t x141 = qw * x138;
    const real_t x142 = x0 * x140;
    const real_t x143 = x0 * x141;
    const real_t x144 = x1 * x139;
    const real_t x145 = x1 * x141;
    const real_t x146 = x139 * x2;
    const real_t x147 = x140 * x2;
    element_matrix[0] = +x12 * x9 + x14 * x9 + x8 * x9;
    element_matrix[1] = +x15 * x16;
    element_matrix[2] = +x17 * x18;
    element_matrix[3] = +x19 * x20;
    element_matrix[4] = +qw * x23 * x8 - x21 - x22;
    element_matrix[5] = +x21 + x24;
    element_matrix[6] = +qw * x12 * x27 - x24 - x25;
    element_matrix[7] = +qw * x14 * x30 - x28 - x29;
    element_matrix[8] = +x22 + x28;
    element_matrix[9] = +x25 + x29;
    element_matrix[10] += x31 * x5 + x31 * x6 + x31 * x7;
    element_matrix[11] += POW2(x15) * x32;
    element_matrix[12] += x33;
    element_matrix[13] += x34;
    element_matrix[14] += fff[0] * qw * x15 * x23 - x36 - x38;
    element_matrix[15] += x36 + x40;
    element_matrix[16] += fff[1] * qw * x15 * x27 - x40 - x42;
    element_matrix[17] += fff[2] * qw * x15 * x30 - x43 - x45;
    element_matrix[18] += x38 + x43;
    element_matrix[19] += x42 + x45;
    element_matrix[20] += x10 * x46 + x11 * x46 + x46 * x6;
    element_matrix[21] += x33;
    element_matrix[22] += POW2(x17) * x47;
    element_matrix[23] += x48;
    element_matrix[24] += fff[1] * qw * x17 * x23 - x50 - x52;
    element_matrix[25] += x50 + x54;
    element_matrix[26] += fff[3] * qw * x17 * x27 - x54 - x56;
    element_matrix[27] += fff[4] * qw * x17 * x30 - x57 - x58;
    element_matrix[28] += x52 + x57;
    element_matrix[29] += x56 + x58;
    element_matrix[30] += x11 * x59 + x13 * x59 + x59 * x7;
    element_matrix[31] += x34;
    element_matrix[32] += x48;
    element_matrix[33] += POW2(x19) * x60;
    element_matrix[34] += fff[2] * qw * x19 * x23 - x61 - x63;
    element_matrix[35] += x61 + x64;
    element_matrix[36] += fff[4] * qw * x19 * x27 - x64 - x65;
    element_matrix[37] += fff[5] * qw * x19 * x30 - x67 - x69;
    element_matrix[38] += x63 + x67;
    element_matrix[39] += x65 + x69;
    element_matrix[40] += x70 * x9 + x72 * x9 + x74 * x9;
    element_matrix[41] += x15 * x75;
    element_matrix[42] += x17 * x76;
    element_matrix[43] += x19 * x77;
    element_matrix[44] += qw * x23 * x70 - x78 - x79;
    element_matrix[45] += x78 + x80;
    element_matrix[46] += qw * x27 * x72 - x80 - x81;
    element_matrix[47] += qw * x30 * x74 - x82 - x83;
    element_matrix[48] += x79 + x82;
    element_matrix[49] += x81 + x83;
    element_matrix[50] += x85 * x9 + x86 * x9 + x87 * x9;
    element_matrix[51] += x15 * x88;
    element_matrix[52] += x17 * x89;
    element_matrix[53] += x19 * x90;
    element_matrix[54] += qw * x23 * x85 - x91 - x92;
    element_matrix[55] += x91 + x93;
    element_matrix[56] += qw * x27 * x86 - x93 - x94;
    element_matrix[57] += qw * x30 * x87 - x95 - x96;
    element_matrix[58] += x92 + x95;
    element_matrix[59] += x94 + x96;
    element_matrix[60] += x100 * x9 + x9 * x97 + x9 * x98;
    element_matrix[61] += x101 * x15;
    element_matrix[62] += x102 * x17;
    element_matrix[63] += x103 * x19;
    element_matrix[64] += qw * x23 * x97 - x104 - x105;
    element_matrix[65] += x104 + x106;
    element_matrix[66] += qw * x27 * x98 - x106 - x107;
    element_matrix[67] += qw * x100 * x30 - x108 - x109;
    element_matrix[68] += x105 + x108;
    element_matrix[69] += x107 + x109;
    element_matrix[70] += x111 * x9 + x113 * x9 + x114 * x9;
    element_matrix[71] += x115 * x15;
    element_matrix[72] += x116 * x17;
    element_matrix[73] += x117 * x19;
    element_matrix[74] += qw * x111 * x23 - x118 - x119;
    element_matrix[75] += x118 + x120;
    element_matrix[76] += qw * x113 * x27 - x120 - x121;
    element_matrix[77] += qw * x114 * x30 - x122 - x123;
    element_matrix[78] += x119 + x122;
    element_matrix[79] += x121 + x123;
    element_matrix[80] += x124 * x9 + x125 * x9 + x126 * x9;
    element_matrix[81] += x127 * x15;
    element_matrix[82] += x128 * x17;
    element_matrix[83] += x129 * x19;
    element_matrix[84] += qw * x124 * x23 - x130 - x131;
    element_matrix[85] += x130 + x132;
    element_matrix[86] += qw * x125 * x27 - x132 - x133;
    element_matrix[87] += qw * x126 * x30 - x134 - x135;
    element_matrix[88] += x131 + x134;
    element_matrix[89] += x133 + x135;
    element_matrix[90] += x136 * x9 + x137 * x9 + x138 * x9;
    element_matrix[91] += x139 * x15;
    element_matrix[92] += x140 * x17;
    element_matrix[93] += x141 * x19;
    element_matrix[94] += qw * x136 * x23 - x142 - x143;
    element_matrix[95] += x142 + x144;
    element_matrix[96] += qw * x137 * x27 - x144 - x145;
    element_matrix[97] += qw * x138 * x30 - x146 - x147;
    element_matrix[98] += x143 + x146;
    element_matrix[99] += x145 + x147;
}

static SFEM_INLINE void tet10_laplacian_diag_add_fff(const jacobian_t *const SFEM_RESTRICT fff,
                                                     real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = 4 * qx;
    const real_t x1 = 4 * qy;
    const real_t x2 = 4 * qz;
    const real_t x3 = x1 + x2;
    const real_t x4 = x0 + x3 - 3;
    const real_t x5 = fff[1] * x4;
    const real_t x6 = fff[2] * x4;
    const real_t x7 = qw * x4;
    const real_t x8 = fff[4] * x4;
    const real_t x9 = fff[3] * x0;
    const real_t x10 = fff[4] * x0;
    const real_t x11 = -8 * qx - x3 + 4;
    const real_t x12 = qw * x0;
    const real_t x13 = fff[5] * x0;
    const real_t x14 = fff[1] * x0;
    const real_t x15 = fff[2] * x0;
    const real_t x16 = fff[1] * x1;
    const real_t x17 = fff[0] * x1;
    const real_t x18 = qw * x1;
    const real_t x19 = fff[2] * x1;
    const real_t x20 = x0 - 4;
    const real_t x21 = -8 * qy - x2 - x20;
    const real_t x22 = fff[5] * x1;
    const real_t x23 = fff[4] * x1;
    const real_t x24 = fff[0] * x2;
    const real_t x25 = fff[1] * x2;
    const real_t x26 = -8 * qz - x1 - x20;
    const real_t x27 = qw * x2;
    const real_t x28 = fff[3] * x2;
    const real_t x29 = fff[2] * x2;
    const real_t x30 = fff[4] * x2;
    element_vector[0] += x7 * (fff[0] * x4 + x5 + x6) + x7 * (fff[3] * x4 + x5 + x8) +
                         x7 * (fff[5] * x4 + x6 + x8);
    element_vector[1] += fff[0] * qw * POW2(x0 - 1);
    element_vector[2] += fff[3] * qw * POW2(x1 - 1);
    element_vector[3] += fff[5] * qw * POW2(x2 - 1);
    element_vector[4] += qw * x11 * (fff[0] * x11 - x14 - x15) - x12 * (fff[1] * x11 - x10 - x9) -
                         x12 * (fff[2] * x11 - x10 - x13);
    element_vector[5] += x12 * (x16 + x9) + x18 * (x14 + x17);
    element_vector[6] += qw * x21 * (fff[3] * x21 - x16 - x23) - x18 * (fff[1] * x21 - x17 - x19) -
                         x18 * (fff[4] * x21 - x19 - x22);
    element_vector[7] += qw * x26 * (fff[5] * x26 - x29 - x30) - x27 * (fff[2] * x26 - x24 - x25) -
                         x27 * (fff[4] * x26 - x25 - x28);
    element_vector[8] += x12 * (x13 + x29) + x27 * (x15 + x24);
    element_vector[9] += x18 * (x22 + x30) + x27 * (x23 + x28);
}

static SFEM_INLINE void tet10_laplacian_energy_add_fff(const jacobian_t *const SFEM_RESTRICT fff,
                                                       real_t *const SFEM_RESTRICT element_scalar) {
    real_t trial_operand[3];
    tet10_laplacian_trial_operand(qx, qy, qz, qw, fff, u, trial_operand);

    const real_t x0 = 4*qx;
    const real_t x1 = 4*qy;
    const real_t x2 = 4*qz;
    const real_t x3 = x1 + x2;
    const real_t x4 = -u[6]*x1;
    const real_t x5 = u[0]*(x0 + x3 - 3);
    const real_t x6 = -u[7]*x2 + x5;
    const real_t x7 = x0 - 4;
    const real_t x8 = -u[4]*x0;
    element_scalar[0] += (1.0/2.0)*trial_operand[0]*(u[1]*(x0 - 1) + u[4]*(-8*qx - x3 + 4) + u[5]*x1 + u[8]*x2 + x4 + x6) + (1.0/2.0)*trial_operand[1]*(u[2]*(x1 - 1) + u[5]*x0 + u[6]*(-8*qy - x2 - x7) + 
    u[9]*x2 + x6 + x8) + (1.0/2.0)*trial_operand[2]*(u[3]*(x2 - 1) + u[7]*(-8*qz - x1 - x7) + u[8]*x0 + u[9]*x1 + x4 + x5 + x8);
}

#endif  // TET10_LAPLACIAN_INLINE_CPU_HPP
