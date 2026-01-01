#ifndef HEX8_LAPLACIAN_INLINE_CPU_H
#define HEX8_LAPLACIAN_INLINE_CPU_H

#include "hex8_inline_cpu.h"

static SFEM_INLINE void hex8_laplacian_apply_fff(const scalar_t *const SFEM_RESTRICT fff,
                                                 const scalar_t                      qx,
                                                 const scalar_t                      qy,
                                                 const scalar_t                      qz,
                                                 const scalar_t                      qw,
                                                 const scalar_t *SFEM_RESTRICT       u,
                                                 accumulator_t *SFEM_RESTRICT        element_vector) {
    scalar_t trial_operand[3];
    {
        const scalar_t x0 = qy * qz;
        const scalar_t x1 = 1 - qz;
        const scalar_t x2 = qy * x1;
        const scalar_t x3 = 1 - qy;
        const scalar_t x4 = qz * x3;
        const scalar_t x5 = x1 * x3;
        const scalar_t x6 = -u[0] * x5 + u[1] * x5 + u[2] * x2 - u[3] * x2 - u[4] * x4 + u[5] * x4 + u[6] * x0 - u[7] * x0;
        const scalar_t x7 = 1 - qx;
        const scalar_t x8 = -qx * qz * u[5] + qx * qz * u[6] - qx * u[1] * x1 + qx * u[2] * x1 - qz * u[4] * x7 + qz * u[7] * x7 -
                            u[0] * x1 * x7 + u[3] * x1 * x7;
        const scalar_t x9 = -qx * qy * u[2] + qx * qy * u[6] - qx * u[1] * x3 + qx * u[5] * x3 - qy * u[3] * x7 + qy * u[7] * x7 -
                            u[0] * x3 * x7 + u[4] * x3 * x7;
        trial_operand[0] = qw * (fff[0] * x6 + fff[1] * x8 + fff[2] * x9);
        trial_operand[1] = qw * (fff[1] * x6 + fff[3] * x8 + fff[4] * x9);
        trial_operand[2] = qw * (fff[2] * x6 + fff[4] * x8 + fff[5] * x9);
    }

    // Dot product
    {
        const scalar_t x0  = 1 - qy;
        const scalar_t x1  = 1 - qz;
        const scalar_t x2  = trial_operand[0] * x1;
        const scalar_t x3  = x0 * x2;
        const scalar_t x4  = 1 - qx;
        const scalar_t x5  = trial_operand[1] * x1;
        const scalar_t x6  = x4 * x5;
        const scalar_t x7  = trial_operand[2] * x0;
        const scalar_t x8  = x4 * x7;
        const scalar_t x9  = qx * x5;
        const scalar_t x10 = qx * x7;
        const scalar_t x11 = qy * trial_operand[2];
        const scalar_t x12 = qx * x11;
        const scalar_t x13 = qy * x2;
        const scalar_t x14 = x11 * x4;
        const scalar_t x15 = qz * trial_operand[0];
        const scalar_t x16 = x0 * x15;
        const scalar_t x17 = qz * trial_operand[1];
        const scalar_t x18 = x17 * x4;
        const scalar_t x19 = qx * x17;
        const scalar_t x20 = qy * x15;
        element_vector[0] += -x3 - x6 - x8;
        element_vector[1] += -x10 + x3 - x9;
        element_vector[2] += -x12 + x13 + x9;
        element_vector[3] += -x13 - x14 + x6;
        element_vector[4] += -x16 - x18 + x8;
        element_vector[5] += x10 + x16 - x19;
        element_vector[6] += x12 + x19 + x20;
        element_vector[7] += x14 + x18 - x20;
    }
}

static SFEM_INLINE void hex8_laplacian_apply_points(const scalar_t *const SFEM_RESTRICT x,
                                                    const scalar_t *const SFEM_RESTRICT y,
                                                    const scalar_t *const SFEM_RESTRICT z,
                                                    const scalar_t                      qx,
                                                    const scalar_t                      qy,
                                                    const scalar_t                      qz,
                                                    const scalar_t                      qw,
                                                    const scalar_t *SFEM_RESTRICT       u,
                                                    accumulator_t *SFEM_RESTRICT        element_vector) {
    scalar_t trial_operand[3];
    {
        scalar_t fff[6];
        hex8_fff(x, y, z, qx, qy, qz, fff);

        const scalar_t x0 = qy * qz;
        const scalar_t x1 = 1 - qz;
        const scalar_t x2 = qy * x1;
        const scalar_t x3 = 1 - qy;
        const scalar_t x4 = qz * x3;
        const scalar_t x5 = x1 * x3;
        const scalar_t x6 = -u[0] * x5 + u[1] * x5 + u[2] * x2 - u[3] * x2 - u[4] * x4 + u[5] * x4 + u[6] * x0 - u[7] * x0;
        const scalar_t x7 = 1 - qx;
        const scalar_t x8 = -qx * qz * u[5] + qx * qz * u[6] - qx * u[1] * x1 + qx * u[2] * x1 - qz * u[4] * x7 + qz * u[7] * x7 -
                            u[0] * x1 * x7 + u[3] * x1 * x7;
        const scalar_t x9 = -qx * qy * u[2] + qx * qy * u[6] - qx * u[1] * x3 + qx * u[5] * x3 - qy * u[3] * x7 + qy * u[7] * x7 -
                            u[0] * x3 * x7 + u[4] * x3 * x7;
        trial_operand[0] = qw * (fff[0] * x6 + fff[1] * x8 + fff[2] * x9);
        trial_operand[1] = qw * (fff[1] * x6 + fff[3] * x8 + fff[4] * x9);
        trial_operand[2] = qw * (fff[2] * x6 + fff[4] * x8 + fff[5] * x9);
    }

    // Dot product
    {
        const scalar_t x0  = 1 - qy;
        const scalar_t x1  = 1 - qz;
        const scalar_t x2  = trial_operand[0] * x1;
        const scalar_t x3  = x0 * x2;
        const scalar_t x4  = 1 - qx;
        const scalar_t x5  = trial_operand[1] * x1;
        const scalar_t x6  = x4 * x5;
        const scalar_t x7  = trial_operand[2] * x0;
        const scalar_t x8  = x4 * x7;
        const scalar_t x9  = qx * x5;
        const scalar_t x10 = qx * x7;
        const scalar_t x11 = qy * trial_operand[2];
        const scalar_t x12 = qx * x11;
        const scalar_t x13 = qy * x2;
        const scalar_t x14 = x11 * x4;
        const scalar_t x15 = qz * trial_operand[0];
        const scalar_t x16 = x0 * x15;
        const scalar_t x17 = qz * trial_operand[1];
        const scalar_t x18 = x17 * x4;
        const scalar_t x19 = qx * x17;
        const scalar_t x20 = qy * x15;
        element_vector[0] += -x3 - x6 - x8;
        element_vector[1] += -x10 + x3 - x9;
        element_vector[2] += -x12 + x13 + x9;
        element_vector[3] += -x13 - x14 + x6;
        element_vector[4] += -x16 - x18 + x8;
        element_vector[5] += x10 + x16 - x19;
        element_vector[6] += x12 + x19 + x20;
        element_vector[7] += x14 + x18 - x20;
    }
}

static SFEM_INLINE void aahex8_laplacian_apply_integral(const scalar_t *const SFEM_RESTRICT jac_diag,
                                                        const scalar_t *SFEM_RESTRICT       u,
                                                        accumulator_t *SFEM_RESTRICT        element_vector) {
    const scalar_t x0  = (scalar_t)(1.0 / 9.0) * jac_diag[1];
    const scalar_t x1  = u[3] * x0;
    const scalar_t x2  = (scalar_t)(1.0 / 9.0) * jac_diag[2];
    const scalar_t x3  = u[4] * x2;
    const scalar_t x4  = (scalar_t)(1.0 / 36.0) * u[6];
    const scalar_t x5  = jac_diag[1] * x4;
    const scalar_t x6  = jac_diag[2] * x4;
    const scalar_t x7  = u[0] * x0;
    const scalar_t x8  = u[0] * x2;
    const scalar_t x9  = (scalar_t)(1.0 / 36.0) * jac_diag[1];
    const scalar_t x10 = u[5] * x9;
    const scalar_t x11 = (scalar_t)(1.0 / 36.0) * jac_diag[2];
    const scalar_t x12 = u[2] * x11;
    const scalar_t x13 = (scalar_t)(1.0 / 9.0) * jac_diag[0];
    const scalar_t x14 = (scalar_t)(1.0 / 36.0) * u[7];
    const scalar_t x15 = (scalar_t)(1.0 / 18.0) * jac_diag[0];
    const scalar_t x16 = -u[2] * x15 + u[3] * x15 + u[4] * x15 - u[5] * x15;
    const scalar_t x17 = jac_diag[0] * x14 - jac_diag[0] * x4 + u[0] * x13 - u[1] * x13 + x16;
    const scalar_t x18 = (scalar_t)(1.0 / 18.0) * jac_diag[1];
    const scalar_t x19 = u[2] * x18;
    const scalar_t x20 = u[7] * x18;
    const scalar_t x21 = (scalar_t)(1.0 / 18.0) * jac_diag[2];
    const scalar_t x22 = u[5] * x21;
    const scalar_t x23 = u[7] * x21;
    const scalar_t x24 = u[1] * x18;
    const scalar_t x25 = u[4] * x18;
    const scalar_t x26 = u[1] * x21;
    const scalar_t x27 = u[3] * x21;
    const scalar_t x28 = -x19 - x20 - x22 - x23 + x24 + x25 + x26 + x27;
    const scalar_t x29 = u[1] * x0;
    const scalar_t x30 = u[1] * x2;
    const scalar_t x31 = u[4] * x9;
    const scalar_t x32 = u[3] * x11;
    const scalar_t x33 = u[2] * x0;
    const scalar_t x34 = u[5] * x2;
    const scalar_t x35 = jac_diag[1] * x14;
    const scalar_t x36 = jac_diag[2] * x14;
    const scalar_t x37 = u[0] * x18;
    const scalar_t x38 = u[5] * x18;
    const scalar_t x39 = u[0] * x21;
    const scalar_t x40 = u[2] * x21;
    const scalar_t x41 = u[3] * x18;
    const scalar_t x42 = u[6] * x18;
    const scalar_t x43 = u[4] * x21;
    const scalar_t x44 = u[6] * x21;
    const scalar_t x45 = -x37 - x38 - x39 - x40 + x41 + x42 + x43 + x44;
    const scalar_t x46 = u[2] * x2;
    const scalar_t x47 = u[0] * x11;
    const scalar_t x48 = u[6] * x2;
    const scalar_t x49 = u[4] * x11;
    const scalar_t x50 = (scalar_t)(1.0 / 36.0) * jac_diag[0];
    const scalar_t x51 = u[0] * x15 - u[1] * x15 - u[6] * x15 + u[7] * x15;
    const scalar_t x52 = -u[2] * x13 + u[3] * x13 + u[4] * x50 - u[5] * x50 + x51;
    const scalar_t x53 = x22 + x23 - x26 - x27 + x37 + x38 - x41 - x42;
    const scalar_t x54 = u[7] * x2;
    const scalar_t x55 = u[5] * x11;
    const scalar_t x56 = u[3] * x2;
    const scalar_t x57 = u[1] * x11;
    const scalar_t x58 = x19 + x20 - x24 - x25 + x39 + x40 - x43 - x44;
    const scalar_t x59 = u[7] * x0;
    const scalar_t x60 = u[2] * x9;
    const scalar_t x61 = u[4] * x0;
    const scalar_t x62 = u[1] * x9;
    const scalar_t x63 = -u[2] * x50 + u[3] * x50 + u[4] * x13 - u[5] * x13 + x51;
    const scalar_t x64 = u[5] * x0;
    const scalar_t x65 = u[0] * x9;
    const scalar_t x66 = u[6] * x0;
    const scalar_t x67 = u[3] * x9;
    const scalar_t x68 = u[0] * x50 - u[1] * x50 - u[6] * x13 + u[7] * x13 + x16;
    element_vector[0]  = -x1 + x10 + x12 + x17 + x28 - x3 - x5 - x6 + x7 + x8;
    element_vector[1]  = -x17 + x29 + x30 + x31 + x32 - x33 - x34 - x35 - x36 - x45;
    element_vector[2]  = -x29 - x31 + x33 + x35 + x46 + x47 - x48 - x49 - x52 - x53;
    element_vector[3]  = x1 - x10 + x5 + x52 - x54 - x55 + x56 + x57 + x58 - x7;
    element_vector[4]  = -x12 + x3 + x53 - x59 + x6 - x60 + x61 + x62 + x63 - x8;
    element_vector[5]  = -x30 - x32 + x34 + x36 - x58 - x63 + x64 + x65 - x66 - x67;
    element_vector[6]  = -x28 - x46 - x47 + x48 + x49 - x64 - x65 + x66 + x67 - x68;
    element_vector[7]  = x45 + x54 + x55 - x56 - x57 + x59 + x60 - x61 - x62 + x68;
}

static SFEM_INLINE void hex8_laplacian_matrix_fff_integral(const scalar_t *const SFEM_RESTRICT fff,
                                                           scalar_t *const SFEM_RESTRICT       element_matrix) {
    const scalar_t x0  = (scalar_t)(1.0 / 6.0) * fff[1];
    const scalar_t x1  = (scalar_t)(1.0 / 6.0) * fff[2];
    const scalar_t x2  = (scalar_t)(1.0 / 6.0) * fff[4];
    const scalar_t x3  = (scalar_t)(1.0 / 9.0) * fff[0];
    const scalar_t x4  = (scalar_t)(1.0 / 9.0) * fff[3];
    const scalar_t x5  = (scalar_t)(1.0 / 9.0) * fff[5];
    const scalar_t x6  = x2 + x3 + x4 + x5;
    const scalar_t x7  = x0 + x1 + x6;
    const scalar_t x8  = (scalar_t)(1.0 / 12.0) * fff[4];
    const scalar_t x9  = (scalar_t)(1.0 / 18.0) * fff[3];
    const scalar_t x10 = (scalar_t)(1.0 / 18.0) * fff[5];
    const scalar_t x11 = x10 + x9;
    const scalar_t x12 = x11 - x3 + x8;
    const scalar_t x13 = (scalar_t)(1.0 / 36.0) * fff[5];
    const scalar_t x14 = (scalar_t)(1.0 / 18.0) * fff[0];
    const scalar_t x15 = x14 + x9;
    const scalar_t x16 = -x13 + x15;
    const scalar_t x17 = -x0 - x16;
    const scalar_t x18 = (scalar_t)(1.0 / 12.0) * fff[2];
    const scalar_t x19 = x10 + x14;
    const scalar_t x20 = x19 - x4;
    const scalar_t x21 = x18 + x20;
    const scalar_t x22 = (scalar_t)(1.0 / 12.0) * fff[1];
    const scalar_t x23 = x15 - x5;
    const scalar_t x24 = x22 + x23;
    const scalar_t x25 = (scalar_t)(1.0 / 36.0) * fff[3];
    const scalar_t x26 = x19 - x25;
    const scalar_t x27 = -x1 - x26;
    const scalar_t x28 = (scalar_t)(1.0 / 36.0) * fff[0];
    const scalar_t x29 = x13 + x25 + x28 + x8;
    const scalar_t x30 = -x18 - x22 - x29;
    const scalar_t x31 = -x11 - x2 + x28;
    const scalar_t x32 = -x0;
    const scalar_t x33 = -x1;
    const scalar_t x34 = x32 + x33 + x6;
    const scalar_t x35 = -x18;
    const scalar_t x36 = x20 + x35;
    const scalar_t x37 = -x16 - x32;
    const scalar_t x38 = -x26 - x33;
    const scalar_t x39 = -x22;
    const scalar_t x40 = x23 + x39;
    const scalar_t x41 = -x29 - x35 - x39;
    const scalar_t x42 = -x2 + x3 + x4 + x5;
    const scalar_t x43 = x0 + x33 + x42;
    const scalar_t x44 = -x10 - x9;
    const scalar_t x45 = -x3 - x44 - x8;
    const scalar_t x46 = x13 + x25 + x28 - x8;
    const scalar_t x47 = -x22 - x35 - x46;
    const scalar_t x48 = x2 + x28 + x44;
    const scalar_t x49 = x1 + x32 + x42;
    const scalar_t x50 = -x18 - x39 - x46;
    element_matrix[0]  = x7;
    element_matrix[1]  = x12;
    element_matrix[2]  = x17;
    element_matrix[3]  = x21;
    element_matrix[4]  = x24;
    element_matrix[5]  = x27;
    element_matrix[6]  = x30;
    element_matrix[7]  = x31;
    element_matrix[8]  = x12;
    element_matrix[9]  = x34;
    element_matrix[10] = x36;
    element_matrix[11] = x37;
    element_matrix[12] = x38;
    element_matrix[13] = x40;
    element_matrix[14] = x31;
    element_matrix[15] = x41;
    element_matrix[16] = x17;
    element_matrix[17] = x36;
    element_matrix[18] = x43;
    element_matrix[19] = x45;
    element_matrix[20] = x47;
    element_matrix[21] = x48;
    element_matrix[22] = x24;
    element_matrix[23] = x38;
    element_matrix[24] = x21;
    element_matrix[25] = x37;
    element_matrix[26] = x45;
    element_matrix[27] = x49;
    element_matrix[28] = x48;
    element_matrix[29] = x50;
    element_matrix[30] = x27;
    element_matrix[31] = x40;
    element_matrix[32] = x24;
    element_matrix[33] = x38;
    element_matrix[34] = x47;
    element_matrix[35] = x48;
    element_matrix[36] = x43;
    element_matrix[37] = x45;
    element_matrix[38] = x17;
    element_matrix[39] = x36;
    element_matrix[40] = x27;
    element_matrix[41] = x40;
    element_matrix[42] = x48;
    element_matrix[43] = x50;
    element_matrix[44] = x45;
    element_matrix[45] = x49;
    element_matrix[46] = x21;
    element_matrix[47] = x37;
    element_matrix[48] = x30;
    element_matrix[49] = x31;
    element_matrix[50] = x24;
    element_matrix[51] = x27;
    element_matrix[52] = x17;
    element_matrix[53] = x21;
    element_matrix[54] = x7;
    element_matrix[55] = x12;
    element_matrix[56] = x31;
    element_matrix[57] = x41;
    element_matrix[58] = x38;
    element_matrix[59] = x40;
    element_matrix[60] = x36;
    element_matrix[61] = x37;
    element_matrix[62] = x12;
    element_matrix[63] = x34;
}

static SFEM_INLINE void hex8_laplacian_apply_fff_integral(const scalar_t *const SFEM_RESTRICT fff,
                                                          const scalar_t *SFEM_RESTRICT       u,
                                                          accumulator_t *SFEM_RESTRICT        element_vector) {
    const scalar_t x0  = (scalar_t)(1.0 / 6.0) * fff[4];
    const scalar_t x1  = u[7] * x0;
    const scalar_t x2  = (scalar_t)(1.0 / 9.0) * fff[3];
    const scalar_t x3  = u[3] * x2;
    const scalar_t x4  = (scalar_t)(1.0 / 9.0) * fff[5];
    const scalar_t x5  = u[4] * x4;
    const scalar_t x6  = (scalar_t)(1.0 / 12.0) * u[6];
    const scalar_t x7  = fff[4] * x6;
    const scalar_t x8  = (scalar_t)(1.0 / 36.0) * u[6];
    const scalar_t x9  = fff[3] * x8;
    const scalar_t x10 = fff[5] * x8;
    const scalar_t x11 = u[0] * x0;
    const scalar_t x12 = u[0] * x2;
    const scalar_t x13 = u[0] * x4;
    const scalar_t x14 = (scalar_t)(1.0 / 12.0) * fff[4];
    const scalar_t x15 = u[1] * x14;
    const scalar_t x16 = (scalar_t)(1.0 / 36.0) * fff[3];
    const scalar_t x17 = u[5] * x16;
    const scalar_t x18 = (scalar_t)(1.0 / 36.0) * fff[5];
    const scalar_t x19 = u[2] * x18;
    const scalar_t x20 = (scalar_t)(1.0 / 6.0) * fff[1];
    const scalar_t x21 = (scalar_t)(1.0 / 12.0) * fff[1];
    const scalar_t x22 = -fff[1] * x6 + u[0] * x20 - u[2] * x20 + u[4] * x21;
    const scalar_t x23 = (scalar_t)(1.0 / 6.0) * fff[2];
    const scalar_t x24 = (scalar_t)(1.0 / 12.0) * fff[2];
    const scalar_t x25 = -fff[2] * x6 + u[0] * x23 + u[3] * x24 - u[5] * x23;
    const scalar_t x26 = (scalar_t)(1.0 / 9.0) * fff[0];
    const scalar_t x27 = (scalar_t)(1.0 / 36.0) * u[7];
    const scalar_t x28 = (scalar_t)(1.0 / 18.0) * fff[0];
    const scalar_t x29 = -u[2] * x28 + u[3] * x28 + u[4] * x28 - u[5] * x28;
    const scalar_t x30 = fff[0] * x27 - fff[0] * x8 + u[0] * x26 - u[1] * x26 + x29;
    const scalar_t x31 = (scalar_t)(1.0 / 18.0) * fff[3];
    const scalar_t x32 = u[2] * x31;
    const scalar_t x33 = u[7] * x31;
    const scalar_t x34 = (scalar_t)(1.0 / 18.0) * fff[5];
    const scalar_t x35 = u[5] * x34;
    const scalar_t x36 = u[7] * x34;
    const scalar_t x37 = u[1] * x31;
    const scalar_t x38 = u[4] * x31;
    const scalar_t x39 = u[1] * x34;
    const scalar_t x40 = u[3] * x34;
    const scalar_t x41 = -x32 - x33 - x35 - x36 + x37 + x38 + x39 + x40;
    const scalar_t x42 = u[1] * x0;
    const scalar_t x43 = u[1] * x2;
    const scalar_t x44 = u[1] * x4;
    const scalar_t x45 = u[0] * x14;
    const scalar_t x46 = u[4] * x16;
    const scalar_t x47 = u[3] * x18;
    const scalar_t x48 = u[6] * x0;
    const scalar_t x49 = u[2] * x2;
    const scalar_t x50 = u[5] * x4;
    const scalar_t x51 = u[7] * x14;
    const scalar_t x52 = fff[3] * x27;
    const scalar_t x53 = fff[5] * x27;
    const scalar_t x54 = u[1] * x20 - u[3] * x20 + u[5] * x21 - u[7] * x21;
    const scalar_t x55 = u[1] * x23 + u[2] * x24 - u[4] * x23 - u[7] * x24;
    const scalar_t x56 = u[0] * x31;
    const scalar_t x57 = u[5] * x31;
    const scalar_t x58 = u[0] * x34;
    const scalar_t x59 = u[2] * x34;
    const scalar_t x60 = u[3] * x31;
    const scalar_t x61 = u[6] * x31;
    const scalar_t x62 = u[4] * x34;
    const scalar_t x63 = u[6] * x34;
    const scalar_t x64 = -x56 - x57 - x58 - x59 + x60 + x61 + x62 + x63;
    const scalar_t x65 = u[5] * x0;
    const scalar_t x66 = u[2] * x4;
    const scalar_t x67 = u[4] * x14;
    const scalar_t x68 = u[0] * x18;
    const scalar_t x69 = u[2] * x0;
    const scalar_t x70 = u[6] * x4;
    const scalar_t x71 = u[3] * x14;
    const scalar_t x72 = u[4] * x18;
    const scalar_t x73 = u[1] * x24 + u[2] * x23 - u[4] * x24 - u[7] * x23;
    const scalar_t x74 = (scalar_t)(1.0 / 36.0) * fff[0];
    const scalar_t x75 = u[0] * x28 - u[1] * x28 - u[6] * x28 + u[7] * x28;
    const scalar_t x76 = -u[2] * x26 + u[3] * x26 + u[4] * x74 - u[5] * x74 + x75;
    const scalar_t x77 = x35 + x36 - x39 - x40 + x56 + x57 - x60 - x61;
    const scalar_t x78 = u[3] * x0;
    const scalar_t x79 = u[7] * x4;
    const scalar_t x80 = u[2] * x14;
    const scalar_t x81 = u[5] * x18;
    const scalar_t x82 = u[4] * x0;
    const scalar_t x83 = u[3] * x4;
    const scalar_t x84 = u[5] * x14;
    const scalar_t x85 = u[1] * x18;
    const scalar_t x86 = u[0] * x24 + u[3] * x23 - u[5] * x24 - u[6] * x23;
    const scalar_t x87 = x32 + x33 - x37 - x38 + x58 + x59 - x62 - x63;
    const scalar_t x88 = u[7] * x2;
    const scalar_t x89 = u[2] * x16;
    const scalar_t x90 = u[4] * x2;
    const scalar_t x91 = u[1] * x16;
    const scalar_t x92 = u[0] * x21 - u[2] * x21 + u[4] * x20 - u[6] * x20;
    const scalar_t x93 = -u[2] * x74 + u[3] * x74 + u[4] * x26 - u[5] * x26 + x75;
    const scalar_t x94 = u[5] * x2;
    const scalar_t x95 = u[0] * x16;
    const scalar_t x96 = u[6] * x2;
    const scalar_t x97 = u[3] * x16;
    const scalar_t x98 = u[1] * x21 - u[3] * x21 + u[5] * x20 - u[7] * x20;
    const scalar_t x99 = u[0] * x74 - u[1] * x74 - u[6] * x26 + u[7] * x26 + x29;
    element_vector[0]  = -x1 - x10 + x11 + x12 + x13 + x15 + x17 + x19 + x22 + x25 - x3 + x30 + x41 - x5 - x7 - x9;
    element_vector[1]  = -x30 + x42 + x43 + x44 + x45 + x46 + x47 - x48 - x49 - x50 - x51 - x52 - x53 - x54 - x55 - x64;
    element_vector[2]  = -x22 - x43 - x46 + x49 + x52 + x65 + x66 + x67 + x68 - x69 - x70 - x71 - x72 - x73 - x76 - x77;
    element_vector[3]  = -x12 - x17 + x3 + x54 + x76 - x78 - x79 - x80 - x81 + x82 + x83 + x84 + x85 + x86 + x87 + x9;
    element_vector[4]  = x10 - x13 - x19 + x5 + x55 + x77 + x78 + x80 - x82 - x84 - x88 - x89 + x90 + x91 + x92 + x93;
    element_vector[5]  = -x25 - x44 - x47 + x50 + x53 - x65 - x67 + x69 + x71 - x87 - x93 + x94 + x95 - x96 - x97 - x98;
    element_vector[6]  = -x41 - x42 - x45 + x48 + x51 - x66 - x68 + x70 + x72 - x86 - x92 - x94 - x95 + x96 + x97 - x99;
    element_vector[7]  = x1 - x11 - x15 + x64 + x7 + x73 + x79 + x81 - x83 - x85 + x88 + x89 - x90 - x91 + x98 + x99;
}

static SFEM_INLINE void hex8_laplacian_diag_fff_integral(const scalar_t *const SFEM_RESTRICT fff,
                                                         accumulator_t *SFEM_RESTRICT        element_vector) {
    const scalar_t x0  = (scalar_t)(1.0 / 6.0) * fff[1];
    const scalar_t x1  = (scalar_t)(1.0 / 6.0) * fff[2];
    const scalar_t x2  = (scalar_t)(1.0 / 6.0) * fff[4];
    const scalar_t x3  = (scalar_t)(1.0 / 9.0) * fff[0];
    const scalar_t x4  = (scalar_t)(1.0 / 9.0) * fff[3];
    const scalar_t x5  = (scalar_t)(1.0 / 9.0) * fff[5];
    const scalar_t x6  = x2 + x3 + x4 + x5;
    const scalar_t x7  = x0 + x1 + x6;
    const scalar_t x8  = -x0;
    const scalar_t x9  = -x1;
    const scalar_t x10 = x6 + x8 + x9;
    const scalar_t x11 = -x2 + x3 + x4 + x5;
    const scalar_t x12 = x0 + x11 + x9;
    const scalar_t x13 = x1 + x11 + x8;
    element_vector[0]  = x7;
    element_vector[1]  = x10;
    element_vector[2]  = x12;
    element_vector[3]  = x13;
    element_vector[4]  = x12;
    element_vector[5]  = x13;
    element_vector[6]  = x7;
    element_vector[7]  = x10;
}

static SFEM_INLINE void hex8_laplacian_matrix_ij_taylor(const scalar_t *const SFEM_RESTRICT fff,
                                                        const scalar_t *const SFEM_RESTRICT trial_g,
                                                        const scalar_t *const SFEM_RESTRICT trial_H,
                                                        const scalar_t                      trial_diff3,
                                                        const scalar_t *const SFEM_RESTRICT test_g,
                                                        const scalar_t *const SFEM_RESTRICT test_H,
                                                        const scalar_t                      test_diff3,
                                                        accumulator_t *const SFEM_RESTRICT  val) {
    const scalar_t x0  = (scalar_t)(1.0 / 12.0) * fff[0];
    const scalar_t x1  = test_H[0] * trial_H[0];
    const scalar_t x2  = test_H[1] * trial_H[1];
    const scalar_t x3  = (scalar_t)(1.0 / 576.0) * fff[0];
    const scalar_t x4  = test_diff3 * x3;
    const scalar_t x5  = test_diff3 * x3;
    const scalar_t x6  = (scalar_t)(1.0 / 12.0) * fff[1];
    const scalar_t x7  = (scalar_t)(1.0 / 12.0) * fff[2];
    const scalar_t x8  = (scalar_t)(1.0 / 12.0) * fff[3];
    const scalar_t x9  = test_H[2] * trial_H[2];
    const scalar_t x10 = (scalar_t)(1.0 / 576.0) * fff[3];
    const scalar_t x11 = test_diff3 * x10;
    const scalar_t x12 = test_diff3 * x10;
    const scalar_t x13 = (scalar_t)(1.0 / 12.0) * fff[4];
    const scalar_t x14 = (scalar_t)(1.0 / 12.0) * fff[5];
    const scalar_t x15 = (scalar_t)(1.0 / 576.0) * fff[5];
    const scalar_t x16 = test_diff3 * x15;
    const scalar_t x17 = test_diff3 * x15;
    val[0]             = fff[0] * test_g[0] * trial_g[0] + fff[1] * test_g[0] * trial_g[1] + fff[1] * test_g[1] * trial_g[0] +
             fff[2] * test_g[0] * trial_g[2] + fff[2] * test_g[2] * trial_g[0] + fff[3] * test_g[1] * trial_g[1] +
             fff[4] * test_g[1] * trial_g[2] + fff[4] * test_g[2] * trial_g[1] + fff[5] * test_g[2] * trial_g[2] +
             test_H[0] * trial_H[1] * x13 + test_H[0] * trial_H[2] * x7 + test_H[1] * trial_H[0] * x13 +
             test_H[1] * trial_H[2] * x6 + test_H[2] * trial_H[0] * x7 + test_H[2] * trial_H[1] * x6 + trial_diff3 * x11 +
             trial_diff3 * x12 + trial_diff3 * x16 + trial_diff3 * x17 + trial_diff3 * x4 + trial_diff3 * x5 + trial_diff3 * x16 +
             trial_diff3 * x17 + trial_diff3 * x4 + trial_diff3 * x5 + trial_diff3 * x11 + trial_diff3 * x12 + x0 * x1 + x0 * x2 +
             x1 * x8 + x14 * x2 + x14 * x9 + x8 * x9;
}

static SFEM_INLINE void hex8_laplacian_matrix_fff_taylor(const scalar_t *const SFEM_RESTRICT fff,
                                                         accumulator_t *const SFEM_RESTRICT  element_matrix) {
    for (int i = 0; i < 8; i++) {
        accumulator_t val;
        hex8_laplacian_matrix_ij_taylor(
                fff, hex8_g_0[i], hex8_H_0[i], hex8_diff3_0[i], hex8_g_0[i], hex8_H_0[i], hex8_diff3_0[i], &val);

        element_matrix[i * 8 + i] = val;

        for (int j = i + 1; j < 8; j++) {
            hex8_laplacian_matrix_ij_taylor(
                    fff, hex8_g_0[j], hex8_H_0[j], hex8_diff3_0[j], hex8_g_0[i], hex8_H_0[i], hex8_diff3_0[i], &val);
            // Exploit symmetry
            element_matrix[i * 8 + j] = val;
            element_matrix[j * 8 + i] = val;
        }
    }
}

static SFEM_INLINE void hex8_laplacian_apply_fff_taylor(const scalar_t *const SFEM_RESTRICT fff,
                                                        const scalar_t *SFEM_RESTRICT       u,
                                                        accumulator_t *SFEM_RESTRICT        element_vector) {
    scalar_t gu[3], Hu[3], diff3u;
    {
        const scalar_t x0  = (scalar_t)(1.0 / 4.0) * u[1];
        const scalar_t x1  = (scalar_t)(1.0 / 4.0) * u[7];
        const scalar_t x2  = (scalar_t)(1.0 / 4.0) * u[2];
        const scalar_t x3  = (scalar_t)(-1.0 / 4.0) * u[6];
        const scalar_t x4  = (scalar_t)(1.0 / 4.0) * u[0];
        const scalar_t x5  = (scalar_t)(1.0 / 4.0) * u[4];
        const scalar_t x6  = -x2 + x3 + x4 + x5;
        const scalar_t x7  = (scalar_t)(1.0 / 4.0) * u[5];
        const scalar_t x8  = (scalar_t)(1.0 / 4.0) * u[3];
        const scalar_t x9  = -x7 + x8;
        const scalar_t x10 = x0 - x1;
        const scalar_t x11 = (scalar_t)(1.0 / 2.0) * u[2];
        const scalar_t x12 = (scalar_t)(1.0 / 2.0) * u[4];
        const scalar_t x13 = (scalar_t)(1.0 / 2.0) * u[0];
        const scalar_t x14 = (scalar_t)(1.0 / 2.0) * u[6];
        const scalar_t x15 = (scalar_t)(1.0 / 2.0) * u[1];
        const scalar_t x16 = (scalar_t)(1.0 / 2.0) * u[7];
        const scalar_t x17 = x13 + x14 - x15 - x16;
        const scalar_t x18 = (scalar_t)(1.0 / 2.0) * u[3];
        const scalar_t x19 = (scalar_t)(1.0 / 2.0) * u[5];
        const scalar_t x20 = -x18 - x19;
        const scalar_t x21 = -x11 - x12;
        const scalar_t x22 = -u[0] + u[1] - u[2] + u[3] + u[4] - u[5] + u[6] - u[7];
        gu[0]              = x0 - x1 - x6 - x9;
        gu[1]              = -x10 - x6 - x7 + x8;
        gu[2]              = -x10 - x2 - x3 - x4 + x5 - x9;
        Hu[0]              = x11 + x12 + x17 + x20;
        Hu[1]              = x17 + x18 + x19 + x21;
        Hu[2]              = x13 + x14 + x15 + x16 + x20 + x21;
        diff3u             = x22;
    }

    for (int i = 0; i < 8; i++) {
        hex8_laplacian_matrix_ij_taylor(fff, gu, Hu, diff3u, hex8_g_0[i], hex8_H_0[i], hex8_diff3_0[i], &element_vector[i]);
    }
}

static SFEM_INLINE void hex8_laplacian_apply_fff_integral_soa(const scalar_t                fff0,
                                                              const scalar_t                fff1,
                                                              const scalar_t                fff2,
                                                              const scalar_t                fff3,
                                                              const scalar_t                fff4,
                                                              const scalar_t                fff5,
                                                              const scalar_t                u0,
                                                              const scalar_t                u1,
                                                              const scalar_t                u2,
                                                              const scalar_t                u3,
                                                              const scalar_t                u4,
                                                              const scalar_t                u5,
                                                              const scalar_t                u6,
                                                              const scalar_t                u7,
                                                              scalar_t *const SFEM_RESTRICT out0,
                                                              scalar_t *const SFEM_RESTRICT out1,
                                                              scalar_t *const SFEM_RESTRICT out2,
                                                              scalar_t *const SFEM_RESTRICT out3,
                                                              scalar_t *const SFEM_RESTRICT out4,
                                                              scalar_t *const SFEM_RESTRICT out5,
                                                              scalar_t *const SFEM_RESTRICT out6,
                                                              scalar_t *const SFEM_RESTRICT out7) {
    const scalar_t x0  = (scalar_t)(1.0 / 6.0) * fff0;
    const scalar_t x1  = u7 * x0;
    const scalar_t x2  = (scalar_t)(1.0 / 9.0) * fff3;
    const scalar_t x3  = u3 * x2;
    const scalar_t x4  = (scalar_t)(1.0 / 9.0) * fff5;
    const scalar_t x5  = u4 * x4;
    const scalar_t x6  = (scalar_t)(1.0 / 12.0) * u6;
    const scalar_t x7  = fff4 * x6;
    const scalar_t x8  = (scalar_t)(1.0 / 36.0) * u6;
    const scalar_t x9  = fff3 * x8;
    const scalar_t x10 = fff5 * x8;
    const scalar_t x11 = u0 * x0;
    const scalar_t x12 = u0 * x2;
    const scalar_t x13 = u0 * x4;
    const scalar_t x14 = (scalar_t)(1.0 / 12.0) * fff4;
    const scalar_t x15 = u1 * x14;
    const scalar_t x16 = (scalar_t)(1.0 / 36.0) * fff3;
    const scalar_t x17 = u5 * x16;
    const scalar_t x18 = (scalar_t)(1.0 / 36.0) * fff5;
    const scalar_t x19 = u2 * x18;
    const scalar_t x20 = (scalar_t)(1.0 / 6.0) * fff1;
    const scalar_t x21 = (scalar_t)(1.0 / 12.0) * fff1;
    const scalar_t x22 = -fff1 * x6 + u0 * x20 - u2 * x20 + u4 * x21;
    const scalar_t x23 = (scalar_t)(1.0 / 6.0) * fff2;
    const scalar_t x24 = (scalar_t)(1.0 / 12.0) * fff2;
    const scalar_t x25 = -fff2 * x6 + u0 * x23 + u3 * x24 - u5 * x23;
    const scalar_t x26 = (scalar_t)(1.0 / 9.0) * fff0;
    const scalar_t x27 = (scalar_t)(1.0 / 36.0) * u7;
    const scalar_t x28 = (scalar_t)(1.0 / 18.0) * fff0;
    const scalar_t x29 = -u2 * x28 + u3 * x28 + u4 * x28 - u5 * x28;
    const scalar_t x30 = fff0 * x27 - fff0 * x8 + u0 * x26 - u1 * x26 + x29;
    const scalar_t x31 = (scalar_t)(1.0 / 18.0) * fff3;
    const scalar_t x32 = u2 * x31;
    const scalar_t x33 = u7 * x31;
    const scalar_t x34 = (scalar_t)(1.0 / 18.0) * fff5;
    const scalar_t x35 = u5 * x34;
    const scalar_t x36 = u7 * x34;
    const scalar_t x37 = u1 * x31;
    const scalar_t x38 = u4 * x31;
    const scalar_t x39 = u1 * x34;
    const scalar_t x40 = u3 * x34;
    const scalar_t x41 = -x32 - x33 - x35 - x36 + x37 + x38 + x39 + x40;
    const scalar_t x42 = u1 * x0;
    const scalar_t x43 = u1 * x2;
    const scalar_t x44 = u1 * x4;
    const scalar_t x45 = u0 * x14;
    const scalar_t x46 = u4 * x16;
    const scalar_t x47 = u3 * x18;
    const scalar_t x48 = u6 * x0;
    const scalar_t x49 = u2 * x2;
    const scalar_t x50 = u5 * x4;
    const scalar_t x51 = u7 * x14;
    const scalar_t x52 = fff3 * x27;
    const scalar_t x53 = fff5 * x27;
    const scalar_t x54 = u1 * x20 - u3 * x20 + u5 * x21 - u7 * x21;
    const scalar_t x55 = u1 * x23 + u2 * x24 - u4 * x23 - u7 * x24;
    const scalar_t x56 = u0 * x31;
    const scalar_t x57 = u5 * x31;
    const scalar_t x58 = u0 * x34;
    const scalar_t x59 = u2 * x34;
    const scalar_t x60 = u3 * x31;
    const scalar_t x61 = u6 * x31;
    const scalar_t x62 = u4 * x34;
    const scalar_t x63 = u6 * x34;
    const scalar_t x64 = -x56 - x57 - x58 - x59 + x60 + x61 + x62 + x63;
    const scalar_t x65 = u5 * x0;
    const scalar_t x66 = u2 * x4;
    const scalar_t x67 = u4 * x14;
    const scalar_t x68 = u0 * x18;
    const scalar_t x69 = u2 * x0;
    const scalar_t x70 = u6 * x4;
    const scalar_t x71 = u3 * x14;
    const scalar_t x72 = u4 * x18;
    const scalar_t x73 = u1 * x24 + u2 * x23 - u4 * x24 - u7 * x23;
    const scalar_t x74 = (scalar_t)(1.0 / 36.0) * fff0;
    const scalar_t x75 = u0 * x28 - u1 * x28 - u6 * x28 + u7 * x28;
    const scalar_t x76 = -u2 * x26 + u3 * x26 + u4 * x74 - u5 * x74 + x75;
    const scalar_t x77 = x35 + x36 - x39 - x40 + x56 + x57 - x60 - x61;
    const scalar_t x78 = u3 * x0;
    const scalar_t x79 = u7 * x4;
    const scalar_t x80 = u2 * x14;
    const scalar_t x81 = u5 * x18;
    const scalar_t x82 = u4 * x0;
    const scalar_t x83 = u3 * x4;
    const scalar_t x84 = u5 * x14;
    const scalar_t x85 = u1 * x18;
    const scalar_t x86 = u0 * x24 + u3 * x23 - u5 * x24 - u6 * x23;
    const scalar_t x87 = x32 + x33 - x37 - x38 + x58 + x59 - x62 - x63;
    const scalar_t x88 = u7 * x2;
    const scalar_t x89 = u2 * x16;
    const scalar_t x90 = u4 * x2;
    const scalar_t x91 = u1 * x16;
    const scalar_t x92 = u0 * x21 - u2 * x21 + u4 * x20 - u6 * x20;
    const scalar_t x93 = -u2 * x74 + u3 * x74 + u4 * x26 - u5 * x26 + x75;
    const scalar_t x94 = u5 * x2;
    const scalar_t x95 = u0 * x16;
    const scalar_t x96 = u6 * x2;
    const scalar_t x97 = u3 * x16;
    const scalar_t x98 = u1 * x21 - u3 * x21 + u5 * x20 - u7 * x20;
    const scalar_t x99 = u0 * x74 - u1 * x74 - u6 * x26 + u7 * x26 + x29;
    out0[0] = -x1 - x10 + x11 + x12 + x13 + x15 + x17 + x19 + x22 + x25 - x3 + x30 + x41 - x5 - x7 - x9;
    out1[0] = -x30 + x42 + x43 + x44 + x45 + x46 + x47 - x48 - x49 - x50 - x51 - x52 - x53 - x54 - x55 - x64;
    out2[0] = -x22 - x43 - x46 + x49 + x52 + x65 + x66 + x67 + x68 - x69 - x70 - x71 - x72 - x73 - x76 - x77;
    out3[0] = -x12 - x17 + x3 + x54 + x76 - x78 - x79 - x80 - x81 + x82 + x83 + x84 + x85 + x86 + x87 + x9;
    out4[0] = x10 - x13 - x19 + x5 + x55 + x77 + x78 + x80 - x82 - x84 - x88 - x89 + x90 + x91 + x92 + x93;
    out5[0] = -x25 - x44 - x47 + x50 + x53 - x65 - x67 + x69 + x71 - x87 - x93 + x94 + x95 - x96 - x97 - x98;
    out6[0] = -x41 - x42 - x45 + x48 + x51 - x66 - x68 + x70 + x72 - x86 - x92 - x94 - x95 + x96 + x97 - x99;
    out7[0] = x1 - x11 - x15 + x64 + x7 + x73 + x79 + x81 - x83 - x85 + x88 + x89 - x90 - x91 + x98 + x99;
}

#endif