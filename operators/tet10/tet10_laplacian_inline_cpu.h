#ifndef TET10_LAPLACIAN_INLINE_CPU_HPP
#define TET10_LAPLACIAN_INLINE_CPU_HPP

#include "tet10_inline_cpu.h"

static SFEM_INLINE void tet10_laplacian_trial_operand(const scalar_t qx,
                                                      const scalar_t qy,
                                                      const scalar_t qz,
                                                      const scalar_t qw,
                                                      const jacobian_t *const SFEM_RESTRICT fff,
                                                      const scalar_t *const SFEM_RESTRICT u,
                                                      scalar_t *const SFEM_RESTRICT out) {
    const scalar_t x0 = 4 * qx;
    const scalar_t x1 = 4 * qy;
    const scalar_t x2 = 4 * qz;
    const scalar_t x3 = x1 + x2;
    const scalar_t x4 = -u[6] * x1;
    const scalar_t x5 = u[0] * (x0 + x3 - 3);
    const scalar_t x6 = -u[7] * x2 + x5;
    const scalar_t x7 =
            u[1] * (x0 - 1) + u[4] * (-8 * qx - x3 + 4) + u[5] * x1 + u[8] * x2 + x4 + x6;
    const scalar_t x8 = x0 - 4;
    const scalar_t x9 = -u[4] * x0;
    const scalar_t x10 =
            u[2] * (x1 - 1) + u[5] * x0 + u[6] * (-8 * qy - x2 - x8) + u[9] * x2 + x6 + x9;
    const scalar_t x11 =
            u[3] * (x2 - 1) + u[7] * (-8 * qz - x1 - x8) + u[8] * x0 + u[9] * x1 + x4 + x5 + x9;
    out[0] = qw * (fff[0] * x7 + fff[1] * x10 + fff[2] * x11);
    out[1] = qw * (fff[1] * x7 + fff[3] * x10 + fff[4] * x11);
    out[2] = qw * (fff[2] * x7 + fff[4] * x10 + fff[5] * x11);
}

static SFEM_INLINE void tet10_laplacian_apply_qp_fff(const scalar_t qx,
                                                     const scalar_t qy,
                                                     const scalar_t qz,
                                                     const scalar_t qw,
                                                     const jacobian_t *const SFEM_RESTRICT fff,
                                                     const scalar_t *const SFEM_RESTRICT u,
                                                     accumulator_t *const SFEM_RESTRICT
                                                             element_vector) {
    // Evaluate gradient fe function transformed with fff and scaling factors

    scalar_t trial_operand[3];
    tet10_laplacian_trial_operand(qx, qy, qz, qw, fff, u, trial_operand);

// TODO check perf difference of two techniques
#if 0
    scalar_t ref_grad[10];

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
    const scalar_t x0 = 4 * qx;
    const scalar_t x1 = 4 * qy;
    const scalar_t x2 = 4 * qz;
    const scalar_t x3 = x1 + x2;
    const scalar_t x4 = x0 + x3 - 3;
    const scalar_t x5 = trial_operand[1] * x0;
    const scalar_t x6 = trial_operand[2] * x0;
    const scalar_t x7 = trial_operand[0] * x1;
    const scalar_t x8 = trial_operand[2] * x1;
    const scalar_t x9 = x0 - 4;
    const scalar_t x10 = trial_operand[0] * x2;
    const scalar_t x11 = trial_operand[1] * x2;
    element_vector[0] += trial_operand[0] * x4 + trial_operand[1] * x4 + trial_operand[2] * x4;
    element_vector[1] += trial_operand[0] * (x0 - 1);
    element_vector[2] += trial_operand[1] * (x1 - 1);
    element_vector[3] += trial_operand[2] * (x2 - 1);
    element_vector[4] += trial_operand[0] * (-8 * qx - x3 + 4) - x5 - x6;
    element_vector[5] += x5 + x7;
    element_vector[6] += trial_operand[1] * (-8 * qy - x2 - x9) - x7 - x8;
    element_vector[7] += trial_operand[2] * (-8 * qz - x1 - x9) - x10 - x11;
    element_vector[8] += x10 + x6;
    element_vector[9] += x11 + x8;
#endif
}

static SFEM_INLINE void tet10_laplacian_apply_add_fff(const jacobian_t *const SFEM_RESTRICT fff,
                                                      const scalar_t *const SFEM_RESTRICT ex,
                                                      accumulator_t *const SFEM_RESTRICT ey) {
    // Numerical quadrature
    tet10_laplacian_apply_qp_fff(0, 0, 0, 0.025, fff, ex, ey);
    tet10_laplacian_apply_qp_fff(1, 0, 0, 0.025, fff, ex, ey);
    tet10_laplacian_apply_qp_fff(0, 1, 0, 0.025, fff, ex, ey);
    tet10_laplacian_apply_qp_fff(0, 0, 1, 0.025, fff, ex, ey);

    static const scalar_t athird = 1. / 3;
    tet10_laplacian_apply_qp_fff(athird, athird, 0., 0.225, fff, ex, ey);
    tet10_laplacian_apply_qp_fff(athird, 0., athird, 0.225, fff, ex, ey);
    tet10_laplacian_apply_qp_fff(0., athird, athird, 0.225, fff, ex, ey);
    tet10_laplacian_apply_qp_fff(athird, athird, athird, 0.225, fff, ex, ey);
}

static SFEM_INLINE void tet10_laplacian_hessian_fff(const jacobian_t *fff,
                                                    accumulator_t *element_matrix) {
    const scalar_t x0 = (1.0 / 10.0) * fff[0];
    const scalar_t x1 = (1.0 / 10.0) * fff[3];
    const scalar_t x2 = (1.0 / 10.0) * fff[5];
    const scalar_t x3 = (1.0 / 30.0) * fff[1];
    const scalar_t x4 = (1.0 / 30.0) * fff[0];
    const scalar_t x5 = (1.0 / 30.0) * fff[2];
    const scalar_t x6 = x4 + x5;
    const scalar_t x7 = x3 + x6;
    const scalar_t x8 = (1.0 / 30.0) * fff[3];
    const scalar_t x9 = (1.0 / 30.0) * fff[4];
    const scalar_t x10 = x8 + x9;
    const scalar_t x11 = x10 + x3;
    const scalar_t x12 = (1.0 / 30.0) * fff[5];
    const scalar_t x13 = x12 + x9;
    const scalar_t x14 = x13 + x5;
    const scalar_t x15 = (2.0 / 15.0) * fff[0];
    const scalar_t x16 = (1.0 / 6.0) * fff[1];
    const scalar_t x17 = (1.0 / 6.0) * fff[2];
    const scalar_t x18 = (1.0 / 15.0) * fff[4] + x12 + x8;
    const scalar_t x19 = -x15 - x16 - x17 - x18;
    const scalar_t x20 = (1.0 / 15.0) * fff[1];
    const scalar_t x21 = x10 + x20 + x6;
    const scalar_t x22 = (2.0 / 15.0) * fff[3];
    const scalar_t x23 = (1.0 / 15.0) * fff[2];
    const scalar_t x24 = (1.0 / 6.0) * fff[4] + x4;
    const scalar_t x25 = -x12 - x16 - x22 - x23 - x24;
    const scalar_t x26 = (2.0 / 15.0) * fff[5];
    const scalar_t x27 = -x17 - x20 - x24 - x26 - x8;
    const scalar_t x28 = x3 + x4;
    const scalar_t x29 = x13 + x23 + x28;
    const scalar_t x30 = x3 + x5;
    const scalar_t x31 = x18 + x30;
    const scalar_t x32 = -x3;
    const scalar_t x33 = -x5;
    const scalar_t x34 = (1.0 / 10.0) * fff[1];
    const scalar_t x35 = (1.0 / 10.0) * fff[2];
    const scalar_t x36 = -x15 - x34 - x35;
    const scalar_t x37 = x34 - x4;
    const scalar_t x38 = x35 - x4;
    const scalar_t x39 = -x30;
    const scalar_t x40 = -x9;
    const scalar_t x41 = x34 - x8;
    const scalar_t x42 = (1.0 / 10.0) * fff[4];
    const scalar_t x43 = -x22 - x34 - x42;
    const scalar_t x44 = x3 + x8;
    const scalar_t x45 = -x3 - x9;
    const scalar_t x46 = x42 - x8;
    const scalar_t x47 = -x5 - x9;
    const scalar_t x48 = x12 + x5;
    const scalar_t x49 = -x26 - x35 - x42;
    const scalar_t x50 = -x12;
    const scalar_t x51 = x35 + x50;
    const scalar_t x52 = x42 + x50;
    const scalar_t x53 = (4.0 / 15.0) * fff[0];
    const scalar_t x54 = (4.0 / 15.0) * fff[1];
    const scalar_t x55 = (4.0 / 15.0) * fff[3];
    const scalar_t x56 = x54 + x55;
    const scalar_t x57 = x53 + x56;
    const scalar_t x58 = (4.0 / 15.0) * fff[2];
    const scalar_t x59 = (4.0 / 15.0) * fff[5];
    const scalar_t x60 = x58 + x59;
    const scalar_t x61 = (4.0 / 15.0) * fff[4];
    const scalar_t x62 = (2.0 / 15.0) * fff[2];
    const scalar_t x63 = x61 + x62;
    const scalar_t x64 = -x56 - x63;
    const scalar_t x65 = (2.0 / 15.0) * fff[4];
    const scalar_t x66 = x54 + x65;
    const scalar_t x67 = x26 + x62 + x66;
    const scalar_t x68 = (2.0 / 15.0) * fff[1];
    const scalar_t x69 = x22 + x58 + x65 + x68;
    const scalar_t x70 = -x60 - x61 - x68;
    const scalar_t x71 = -x22 - x26 - x61;
    const scalar_t x72 = -x53 - x58 - x66;
    const scalar_t x73 = -x15 - x22 - x54;
    const scalar_t x74 = x15 + x63 + x68;
    const scalar_t x75 = x59 + x61;
    const scalar_t x76 = -x15 - x26 - x58;
    const scalar_t x77 = x53 + x60;
    element_matrix[0] =
            (1.0 / 5.0) * fff[1] + (1.0 / 5.0) * fff[2] + (1.0 / 5.0) * fff[4] + x0 + x1 + x2;
    element_matrix[1] = x7;
    element_matrix[2] = x11;
    element_matrix[3] = x14;
    element_matrix[4] = x19;
    element_matrix[5] = x21;
    element_matrix[6] = x25;
    element_matrix[7] = x27;
    element_matrix[8] = x29;
    element_matrix[9] = x31;
    element_matrix[10] = x7;
    element_matrix[11] = x0;
    element_matrix[12] = x32;
    element_matrix[13] = x33;
    element_matrix[14] = x36;
    element_matrix[15] = x37;
    element_matrix[16] = x6;
    element_matrix[17] = x28;
    element_matrix[18] = x38;
    element_matrix[19] = x39;
    element_matrix[20] = x11;
    element_matrix[21] = x32;
    element_matrix[22] = x1;
    element_matrix[23] = x40;
    element_matrix[24] = x10;
    element_matrix[25] = x41;
    element_matrix[26] = x43;
    element_matrix[27] = x44;
    element_matrix[28] = x45;
    element_matrix[29] = x46;
    element_matrix[30] = x14;
    element_matrix[31] = x33;
    element_matrix[32] = x40;
    element_matrix[33] = x2;
    element_matrix[34] = x13;
    element_matrix[35] = x47;
    element_matrix[36] = x48;
    element_matrix[37] = x49;
    element_matrix[38] = x51;
    element_matrix[39] = x52;
    element_matrix[40] = x19;
    element_matrix[41] = x36;
    element_matrix[42] = x10;
    element_matrix[43] = x13;
    element_matrix[44] = (8.0 / 15.0) * fff[4] + x57 + x60;
    element_matrix[45] = x64;
    element_matrix[46] = x67;
    element_matrix[47] = x69;
    element_matrix[48] = x70;
    element_matrix[49] = x71;
    element_matrix[50] = x21;
    element_matrix[51] = x37;
    element_matrix[52] = x41;
    element_matrix[53] = x47;
    element_matrix[54] = x64;
    element_matrix[55] = x57;
    element_matrix[56] = x72;
    element_matrix[57] = x73;
    element_matrix[58] = x74;
    element_matrix[59] = x69;
    element_matrix[60] = x25;
    element_matrix[61] = x6;
    element_matrix[62] = x43;
    element_matrix[63] = x48;
    element_matrix[64] = x67;
    element_matrix[65] = x72;
    element_matrix[66] = (8.0 / 15.0) * fff[2] + x57 + x75;
    element_matrix[67] = x74;
    element_matrix[68] = x76;
    element_matrix[69] = x70;
    element_matrix[70] = x27;
    element_matrix[71] = x28;
    element_matrix[72] = x44;
    element_matrix[73] = x49;
    element_matrix[74] = x69;
    element_matrix[75] = x73;
    element_matrix[76] = x74;
    element_matrix[77] = (8.0 / 15.0) * fff[1] + x55 + x61 + x77;
    element_matrix[78] = x72;
    element_matrix[79] = x64;
    element_matrix[80] = x29;
    element_matrix[81] = x38;
    element_matrix[82] = x45;
    element_matrix[83] = x51;
    element_matrix[84] = x70;
    element_matrix[85] = x74;
    element_matrix[86] = x76;
    element_matrix[87] = x72;
    element_matrix[88] = x77;
    element_matrix[89] = x67;
    element_matrix[90] = x31;
    element_matrix[91] = x39;
    element_matrix[92] = x46;
    element_matrix[93] = x52;
    element_matrix[94] = x71;
    element_matrix[95] = x69;
    element_matrix[96] = x70;
    element_matrix[97] = x64;
    element_matrix[98] = x67;
    element_matrix[99] = x55 + x75;
}

static SFEM_INLINE void tet10_laplacian_diag_fff(const jacobian_t *const SFEM_RESTRICT fff,
                                                 accumulator_t *const SFEM_RESTRICT
                                                         element_vector) {
    const real_t x0 = (1.0 / 10.0) * fff[0];
    const real_t x1 = (1.0 / 10.0) * fff[3];
    const real_t x2 = (1.0 / 10.0) * fff[5];
    const real_t x3 = (4.0 / 15.0) * fff[0];
    const real_t x4 = (4.0 / 15.0) * fff[3];
    const real_t x5 = x3 + x4;
    const real_t x6 = (4.0 / 15.0) * fff[1] + x5;
    const real_t x7 = (4.0 / 15.0) * fff[5];
    const real_t x8 = (4.0 / 15.0) * fff[2] + x7;
    const real_t x9 = (4.0 / 15.0) * fff[4];
    const real_t x10 = x7 + x9;
    element_vector[0] =
            (1.0 / 5.0) * fff[1] + (1.0 / 5.0) * fff[2] + (1.0 / 5.0) * fff[4] + x0 + x1 + x2;
    element_vector[1] = x0;
    element_vector[2] = x1;
    element_vector[3] = x2;
    element_vector[4] = (8.0 / 15.0) * fff[4] + x6 + x8;
    element_vector[5] = x6;
    element_vector[6] = (8.0 / 15.0) * fff[2] + x10 + x6;
    element_vector[7] = (8.0 / 15.0) * fff[1] + x5 + x8 + x9;
    element_vector[8] = x3 + x8;
    element_vector[9] = x10 + x4;
}

static SFEM_INLINE void tet10_laplacian_energy_fff(const jacobian_t *const SFEM_RESTRICT fff,
                                                   const scalar_t *const SFEM_RESTRICT u,
                                                   accumulator_t *const SFEM_RESTRICT
                                                           element_scalar) {
    const real_t x0 = (1.0 / 30.0) * fff[0];
    const real_t x1 = u[0] * x0;
    const real_t x2 = (2.0 / 15.0) * fff[0];
    const real_t x3 = u[4] * x2;
    const real_t x4 = u[1] * x0;
    const real_t x5 = (4.0 / 15.0) * fff[0];
    const real_t x6 = u[5] * u[6];
    const real_t x7 = u[5] * u[7];
    const real_t x8 = u[5] * u[8];
    const real_t x9 = u[6] * u[7];
    const real_t x10 = u[6] * u[8];
    const real_t x11 = u[7] * u[8];
    const real_t x12 = (1.0 / 30.0) * u[1];
    const real_t x13 = fff[1] * u[0];
    const real_t x14 = fff[1] * u[2];
    const real_t x15 = (1.0 / 30.0) * u[0];
    const real_t x16 = (1.0 / 6.0) * u[0];
    const real_t x17 = fff[1] * u[4];
    const real_t x18 = (1.0 / 15.0) * u[0];
    const real_t x19 = fff[1] * u[5];
    const real_t x20 = u[6] * x16;
    const real_t x21 = fff[1] * u[7];
    const real_t x22 = (1.0 / 30.0) * u[8];
    const real_t x23 = u[9] * x15;
    const real_t x24 = (1.0 / 10.0) * u[1];
    const real_t x25 = u[9] * x12;
    const real_t x26 = (1.0 / 10.0) * u[6];
    const real_t x27 = (4.0 / 15.0) * fff[1];
    const real_t x28 = u[5] * x27;
    const real_t x29 = u[4] * u[6];
    const real_t x30 = (2.0 / 15.0) * fff[1];
    const real_t x31 = u[4] * u[7];
    const real_t x32 = u[4] * u[8];
    const real_t x33 = u[5] * u[9];
    const real_t x34 = u[6] * u[9];
    const real_t x35 = u[7] * u[9];
    const real_t x36 = u[8] * u[9];
    const real_t x37 = fff[2] * u[0];
    const real_t x38 = fff[2] * u[3];
    const real_t x39 = fff[2] * x16;
    const real_t x40 = (1.0 / 30.0) * u[5];
    const real_t x41 = fff[2] * x18;
    const real_t x42 = fff[2] * x24;
    const real_t x43 = (1.0 / 10.0) * u[3];
    const real_t x44 = fff[2] * x43;
    const real_t x45 = (2.0 / 15.0) * fff[2];
    const real_t x46 = u[4] * u[5];
    const real_t x47 = (4.0 / 15.0) * fff[2];
    const real_t x48 = u[8] * x47;
    const real_t x49 = (1.0 / 30.0) * fff[3];
    const real_t x50 = u[0] * x49;
    const real_t x51 = fff[3] * x40;
    const real_t x52 = (2.0 / 15.0) * fff[3];
    const real_t x53 = u[6] * x52;
    const real_t x54 = u[2] * x49;
    const real_t x55 = (4.0 / 15.0) * fff[3];
    const real_t x56 = u[4] * u[9];
    const real_t x57 = fff[4] * u[2];
    const real_t x58 = (1.0 / 30.0) * u[3];
    const real_t x59 = fff[4] * u[0];
    const real_t x60 = fff[4] * x18;
    const real_t x61 = fff[4] * x40;
    const real_t x62 = fff[4] * u[7];
    const real_t x63 = (4.0 / 15.0) * fff[4];
    const real_t x64 = (2.0 / 15.0) * fff[4];
    const real_t x65 = u[4] * x63;
    const real_t x66 = fff[5] * x58;
    const real_t x67 = fff[5] * u[4];
    const real_t x68 = fff[5] * x15;
    const real_t x69 = (2.0 / 15.0) * fff[5];
    const real_t x70 = u[7] * x69;
    const real_t x71 = fff[5] * x22;
    const real_t x72 = (4.0 / 15.0) * fff[5];
    const real_t x73 = POW2(u[0]);
    const real_t x74 = (1.0 / 20.0) * fff[0];
    const real_t x75 = POW2(u[4]);
    const real_t x76 = POW2(u[5]);
    const real_t x77 = POW2(u[6]);
    const real_t x78 = POW2(u[7]);
    const real_t x79 = POW2(u[8]);
    const real_t x80 = (1.0 / 10.0) * x73;
    const real_t x81 = (1.0 / 20.0) * fff[3];
    const real_t x82 = POW2(u[9]);
    const real_t x83 = (1.0 / 20.0) * fff[5];
    element_scalar[0] =
            -fff[1] * x20 + fff[1] * x23 - fff[1] * x25 + fff[1] * x80 + fff[2] * u[6] * x12 +
            fff[2] * x23 - fff[2] * x25 + fff[2] * x80 + fff[4] * u[4] * x58 + fff[4] * u[9] * x43 -
            fff[4] * x20 + fff[4] * x80 - u[0] * x3 + u[0] * x51 - u[0] * x53 + u[0] * x61 +
            u[0] * x66 - u[0] * x70 + u[0] * x71 + POW2(u[1]) * x74 + u[1] * x1 - u[1] * x3 +
            POW2(u[2]) * x81 + u[2] * x50 - u[2] * x51 - u[2] * x53 + POW2(u[3]) * x83 -
            u[3] * x61 - u[3] * x70 - u[3] * x71 - u[4] * x28 - u[4] * x39 - u[4] * x42 -
            u[4] * x48 - u[4] * x50 + u[4] * x54 + (1.0 / 30.0) * u[4] * x57 - u[4] * x60 +
            u[5] * x1 + (1.0 / 10.0) * u[5] * x14 - u[5] * x4 - u[6] * x1 +
            (1.0 / 30.0) * u[6] * x38 + u[6] * x4 - u[6] * x41 - u[6] * x48 + u[6] * x66 -
            u[6] * x68 - u[7] * x1 + (1.0 / 30.0) * u[7] * x14 - u[7] * x28 - u[7] * x39 +
            u[7] * x4 - u[7] * x44 - u[7] * x50 + u[7] * x54 + u[8] * x1 - u[8] * x4 + u[8] * x41 +
            u[8] * x42 + u[8] * x44 - u[8] * x65 + u[9] * x50 - u[9] * x54 +
            (1.0 / 10.0) * u[9] * x57 + u[9] * x60 - u[9] * x65 - u[9] * x66 + u[9] * x68 -
            x10 * x2 - x10 * x69 - x11 * x27 - x11 * x47 - x11 * x5 - x11 * x64 + x12 * x13 -
            x12 * x14 + x12 * x21 + x12 * x37 - x12 * x38 + x13 * x22 + x14 * x15 - x14 * x22 -
            x14 * x26 + x15 * x38 + x15 * x57 - x15 * x67 - x16 * x17 - x16 * x62 - x17 * x24 +
            x18 * x19 - x18 * x21 + x19 * x24 - x2 * x7 + x2 * x75 + x2 * x76 + x2 * x77 +
            x2 * x78 + x2 * x79 + x2 * x8 + x2 * x9 - x22 * x57 + x22 * x59 - x26 * x57 +
            x27 * x29 - x27 * x35 + x27 * x36 - x27 * x6 + x27 * x78 + x29 * x45 + x29 * x64 +
            x29 * x69 + x30 * x31 - x30 * x32 + x30 * x33 - x30 * x34 + x30 * x75 + x30 * x76 +
            x30 * x77 + x30 * x8 + x30 * x9 + x31 * x47 + x31 * x52 + x31 * x64 - x32 * x72 +
            x33 * x47 + x33 * x52 + x33 * x64 - x34 * x47 - x34 * x63 - x34 * x72 - x35 * x45 -
            x35 * x55 - x35 * x63 + x36 * x45 + x36 * x64 + x36 * x69 + x37 * x40 - x38 * x40 -
            x43 * x62 - x45 * x46 + x45 * x75 + x45 * x78 + x45 * x79 + x45 * x8 + x45 * x9 -
            x46 * x55 - x46 * x63 - x47 * x6 + x47 * x77 - x5 * x6 - x52 * x56 - x52 * x7 +
            x52 * x75 + x52 * x76 + x52 * x77 + x52 * x78 + x52 * x82 - x56 * x69 - x57 * x58 +
            x58 * x59 + x58 * x67 - x6 * x64 + x63 * x75 + x63 * x8 + x63 * x9 + x64 * x77 +
            x64 * x78 + x64 * x82 + x69 * x75 + x69 * x77 + x69 * x78 + x69 * x79 + x69 * x82 +
            x73 * x74 + x73 * x81 + x73 * x83;
}

#endif  // TET10_LAPLACIAN_INLINE_CPU_HPP
