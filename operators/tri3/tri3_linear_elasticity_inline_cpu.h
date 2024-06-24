#ifndef TRI3_LINEAR_ELASTICITY_INLINE_CPU_H
#define TRI3_LINEAR_ELASTICITY_INLINE_CPU_H

#include "tri3_inline_cpu.h"

static SFEM_INLINE void tri3_linear_elasticity_value_points(const scalar_t mu,
                                                            const scalar_t lambda,
                                                            const scalar_t px0,
                                                            const scalar_t px1,
                                                            const scalar_t px2,
                                                            const scalar_t py0,
                                                            const scalar_t py1,
                                                            const scalar_t py2,
                                                            const scalar_t *const SFEM_RESTRICT ux,
                                                            const scalar_t *const SFEM_RESTRICT uy,
                                                            accumulator_t *const SFEM_RESTRICT
                                                                    element_scalar) {
    const scalar_t x0 = -px0 + px1;
    const scalar_t x1 = -py0 + py2;
    const scalar_t x2 = px0 - px2;
    const scalar_t x3 = py0 - py1;
    const scalar_t x4 = x0 * x1 - x2 * x3;
    const scalar_t x5 = 1.0 / x4;
    const scalar_t x6 = ux[0] * x2 * x5 + uy[3] * x0 * x5;
    const scalar_t x7 = ux[2] * x2 * x5 + uy[5] * x0 * x5 - x6;
    const scalar_t x8 = pow(x7, 2);
    const scalar_t x9 = (1.0 / 4.0) * lambda;
    const scalar_t x10 = ux[0] * x1 * x5 + uy[3] * x3 * x5;
    const scalar_t x11 = ux[1] * x1 * x5 + uy[4] * x3 * x5 - x10;
    const scalar_t x12 = pow(x11, 2);
    const scalar_t x13 = ux[1] * x2 * x5 + uy[4] * x0 * x5 - x6;
    const scalar_t x14 = (1.0 / 4.0) * mu;
    const scalar_t x15 = (1.0 / 2.0) * mu;
    const scalar_t x16 = ux[2] * x1 * x5 + uy[5] * x3 * x5 - x10;
    element_scalar[0] =
            x4 * ((1.0 / 2.0) * lambda * x11 * x7 + x12 * x15 + x12 * x9 + pow(x13, 2) * x14 +
                  x13 * x15 * x16 + x14 * pow(x16, 2) + x15 * x8 + x8 * x9);
}

static SFEM_INLINE void tri3_linear_elasticity_apply_points(const scalar_t mu,
                                                            const scalar_t lambda,
                                                            const scalar_t px0,
                                                            const scalar_t px1,
                                                            const scalar_t px2,
                                                            const scalar_t py0,
                                                            const scalar_t py1,
                                                            const scalar_t py2,
                                                            const scalar_t *const SFEM_RESTRICT ux,
                                                            const scalar_t *const SFEM_RESTRICT uy,
                                                            accumulator_t *const SFEM_RESTRICT outx,
                                                            accumulator_t *const SFEM_RESTRICT
                                                                    outy) {
    const scalar_t x0 = -px0 + px1;
    const scalar_t x1 = -py0 + py2;
    const scalar_t x2 = px0 - px2;
    const scalar_t x3 = py0 - py1;
    const scalar_t x4 = x0 * x1 - x2 * x3;
    const scalar_t x5 = 1.0 / x4;
    const scalar_t x6 = x1 * x5;
    const scalar_t x7 = x3 * x5;
    const scalar_t x8 = -x6 - x7;
    const scalar_t x9 = ux[0] * x8 + ux[1] * x6 + ux[2] * x7;
    const scalar_t x10 = mu * x9;
    const scalar_t x11 = (1.0 / 2.0) * lambda;
    const scalar_t x12 = x11 * x6;
    const scalar_t x13 = x2 * x5;
    const scalar_t x14 = x0 * x5;
    const scalar_t x15 = -x13 - x14;
    const scalar_t x16 = uy[3] * x15 + uy[4] * x13 + uy[5] * x14;
    const scalar_t x17 = ux[0] * x15 + ux[1] * x13 + ux[2] * x14;
    const scalar_t x18 = (1.0 / 2.0) * mu;
    const scalar_t x19 = x13 * x18;
    const scalar_t x20 = uy[3] * x8 + uy[4] * x6 + uy[5] * x7;
    const scalar_t x21 = x10 * x6 + x12 * x16 + x12 * x9 + x17 * x19 + x19 * x20;
    const scalar_t x22 = x11 * x7;
    const scalar_t x23 = x14 * x18;
    const scalar_t x24 = x10 * x7 + x16 * x22 + x17 * x23 + x20 * x23 + x22 * x9;
    const scalar_t x25 = mu * x16;
    const scalar_t x26 = x11 * x13;
    const scalar_t x27 = x18 * x6;
    const scalar_t x28 = x13 * x25 + x16 * x26 + x17 * x27 + x20 * x27 + x26 * x9;
    const scalar_t x29 = x11 * x14;
    const scalar_t x30 = x18 * x7;
    const scalar_t x31 = x14 * x25 + x16 * x29 + x17 * x30 + x20 * x30 + x29 * x9;
    outx[0] = x4 * (-x21 - x24);
    outx[1] = x21 * x4;
    outx[2] = x24 * x4;
    outy[0] = x4 * (-x28 - x31);
    outy[1] = x28 * x4;
    outy[2] = x31 * x4;
}



static SFEM_INLINE void tri3_linear_elasticity_hessian_points(const real_t mu,
                                                              const real_t lambda,
                                                              const real_t px0,
                                                              const real_t px1,
                                                              const real_t px2,
                                                              const real_t py0,
                                                              const real_t py1,
                                                              const real_t py2,
                                                              real_t *const SFEM_RESTRICT
                                                                      element_matrix) {
    const real_t x0 = -px0 + px1;
    const real_t x1 = -py0 + py2;
    const real_t x2 = x0 * x1;
    const real_t x3 = px0 - px2;
    const real_t x4 = py0 - py1;
    const real_t x5 = x2 - x3 * x4;
    const real_t x6 = pow(x5, -2);
    const real_t x7 = mu * x6;
    const real_t x8 = x0 * x3;
    const real_t x9 = x7 * x8;
    const real_t x10 = lambda * x6;
    const real_t x11 = x1 * x4;
    const real_t x12 = x11 * x7;
    const real_t x13 = pow(x1, 2);
    const real_t x14 = x13 * x7;
    const real_t x15 = (1.0 / 2.0) * x10;
    const real_t x16 = pow(x3, 2);
    const real_t x17 = x16 * x7;
    const real_t x18 = x13 * x15 + x14 + (1.0 / 2.0) * x17;
    const real_t x19 = pow(x4, 2);
    const real_t x20 = x19 * x7;
    const real_t x21 = pow(x0, 2);
    const real_t x22 = x21 * x7;
    const real_t x23 = x15 * x19 + x20 + (1.0 / 2.0) * x22;
    const real_t x24 = x11 * x15 + x12 + (1.0 / 2.0) * x9;
    const real_t x25 = x5 * (-x18 - x24);
    const real_t x26 = x5 * (-x23 - x24);
    const real_t x27 = x15 * x3;
    const real_t x28 = (1.0 / 2.0) * x7;
    const real_t x29 = x28 * x3;
    const real_t x30 = x1 * x27 + x1 * x29;
    const real_t x31 = x2 * x28 + x27 * x4;
    const real_t x32 = x30 + x31;
    const real_t x33 = x15 * x2 + x29 * x4;
    const real_t x34 = x0 * x4;
    const real_t x35 = x15 * x34 + x28 * x34;
    const real_t x36 = x33 + x35;
    const real_t x37 = x5 * (x32 + x36);
    const real_t x38 = -x32 * x5;
    const real_t x39 = -x36 * x5;
    const real_t x40 = x24 * x5;
    const real_t x41 = x5 * (-x30 - x33);
    const real_t x42 = x30 * x5;
    const real_t x43 = x33 * x5;
    const real_t x44 = x5 * (-x31 - x35);
    const real_t x45 = x31 * x5;
    const real_t x46 = x35 * x5;
    const real_t x47 = (1.0 / 2.0) * x14 + x15 * x16 + x17;
    const real_t x48 = x15 * x21 + (1.0 / 2.0) * x20 + x22;
    const real_t x49 = (1.0 / 2.0) * x12 + x15 * x8 + x9;
    const real_t x50 = x5 * (-x47 - x49);
    const real_t x51 = x5 * (-x48 - x49);
    const real_t x52 = x49 * x5;
    element_matrix[0] = x5 * (x10 * x11 + 2 * x12 + x18 + x23 + x9);
    element_matrix[1] = x25;
    element_matrix[2] = x26;
    element_matrix[3] = x37;
    element_matrix[4] = x38;
    element_matrix[5] = x39;
    element_matrix[6] = x25;
    element_matrix[7] = x18 * x5;
    element_matrix[8] = x40;
    element_matrix[9] = x41;
    element_matrix[10] = x42;
    element_matrix[11] = x43;
    element_matrix[12] = x26;
    element_matrix[13] = x40;
    element_matrix[14] = x23 * x5;
    element_matrix[15] = x44;
    element_matrix[16] = x45;
    element_matrix[17] = x46;
    element_matrix[18] = x37;
    element_matrix[19] = x41;
    element_matrix[20] = x44;
    element_matrix[21] = x5 * (x10 * x8 + x12 + x47 + x48 + 2 * x9);
    element_matrix[22] = x50;
    element_matrix[23] = x51;
    element_matrix[24] = x38;
    element_matrix[25] = x42;
    element_matrix[26] = x45;
    element_matrix[27] = x50;
    element_matrix[28] = x47 * x5;
    element_matrix[29] = x52;
    element_matrix[30] = x39;
    element_matrix[31] = x43;
    element_matrix[32] = x46;
    element_matrix[33] = x51;
    element_matrix[34] = x52;
    element_matrix[35] = x48 * x5;
}



// static SFEM_INLINE void tri3_linear_elasticity_apply_kernel(
//         const real_t mu,
//         const real_t lambda,
//         const real_t px0,
//         const real_t px1,
//         const real_t px2,
//         const real_t py0,
//         const real_t py1,
//         const real_t py2,
//         const real_t *const SFEM_RESTRICT increment,
//         real_t *const SFEM_RESTRICT element_vector) {
//     const real_t x0 = -py0 + py2;
//     const real_t x1 = px0 - px2;
//     const real_t x2 = -px0 + px1;
//     const real_t x3 = x0 * x2;
//     const real_t x4 = py0 - py1;
//     const real_t x5 = -x1 * x4 + x3;
//     const real_t x6 = pow(x5, -2);
//     const real_t x7 = lambda * x6;
//     const real_t x8 = (1.0 / 2.0) * x7;
//     const real_t x9 = x1 * x8;
//     const real_t x10 = mu * x6;
//     const real_t x11 = (1.0 / 2.0) * x10;
//     const real_t x12 = x1 * x11;
//     const real_t x13 = x0 * x12 + x0 * x9;
//     const real_t x14 = x11 * x3 + x4 * x9;
//     const real_t x15 = x13 + x14;
//     const real_t x16 = -x15;
//     const real_t x17 = increment[4] * x5;
//     const real_t x18 = x12 * x4 + x3 * x8;
//     const real_t x19 = x2 * x4;
//     const real_t x20 = x11 * x19 + x19 * x8;
//     const real_t x21 = x18 + x20;
//     const real_t x22 = -x21;
//     const real_t x23 = increment[5] * x5;
//     const real_t x24 = pow(x0, 2);
//     const real_t x25 = x10 * x24;
//     const real_t x26 = pow(x1, 2);
//     const real_t x27 = x10 * x26;
//     const real_t x28 = x24 * x8 + x25 + (1.0 / 2.0) * x27;
//     const real_t x29 = x0 * x4;
//     const real_t x30 = x10 * x29;
//     const real_t x31 = x1 * x2;
//     const real_t x32 = x10 * x31;
//     const real_t x33 = x29 * x8 + x30 + (1.0 / 2.0) * x32;
//     const real_t x34 = -x28 - x33;
//     const real_t x35 = increment[1] * x5;
//     const real_t x36 = pow(x4, 2);
//     const real_t x37 = x10 * x36;
//     const real_t x38 = pow(x2, 2);
//     const real_t x39 = x10 * x38;
//     const real_t x40 = x36 * x8 + x37 + (1.0 / 2.0) * x39;
//     const real_t x41 = -x33 - x40;
//     const real_t x42 = increment[2] * x5;
//     const real_t x43 = x15 + x21;
//     const real_t x44 = increment[3] * x5;
//     const real_t x45 = increment[0] * x5;
//     const real_t x46 = -x13 - x18;
//     const real_t x47 = -x14 - x20;
//     const real_t x48 = (1.0 / 2.0) * x25 + x26 * x8 + x27;
//     const real_t x49 = (1.0 / 2.0) * x30 + x31 * x8 + x32;
//     const real_t x50 = -x48 - x49;
//     const real_t x51 = (1.0 / 2.0) * x37 + x38 * x8 + x39;
//     const real_t x52 = -x49 - x51;
//     element_vector[0] = x16 * x17 + x22 * x23 + x34 * x35 + x41 * x42 + x43 * x44 +
//                                  x45 * (x28 + x29 * x7 + 2 * x30 + x32 + x40);
//     element_vector[1] =
//             x13 * x17 + x18 * x23 + x28 * x35 + x33 * x42 + x34 * x45 + x44 * x46;
//     element_vector[2] =
//             x14 * x17 + x20 * x23 + x33 * x35 + x40 * x42 + x41 * x45 + x44 * x47;
//     element_vector[3] = x17 * x50 + x23 * x52 + x35 * x46 + x42 * x47 + x43 * x45 +
//                                  x44 * (x30 + x31 * x7 + 2 * x32 + x48 + x51);
//     element_vector[4] =
//             x13 * x35 + x14 * x42 + x16 * x45 + x17 * x48 + x23 * x49 + x44 * x50;
//     element_vector[5] =
//             x17 * x49 + x18 * x35 + x20 * x42 + x22 * x45 + x23 * x51 + x44 * x52;
// }

#endif  // TRI3_LINEAR_ELASTICITY_INLINE_CPU_H
