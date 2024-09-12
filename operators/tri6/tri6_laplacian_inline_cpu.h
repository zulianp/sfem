#ifndef TRI6_LAPLACIAN_INLINE_CPU_HPP
#define TRI6_LAPLACIAN_INLINE_CPU_HPP

#include "tri6_inline_cpu.h"

static SFEM_INLINE void tri6_laplacian_apply_fff(const jacobian_t *const SFEM_RESTRICT fff,
                                                 const scalar_t *const SFEM_RESTRICT u,
                                                 accumulator_t *const SFEM_RESTRICT
                                                         element_vector) {
    const real_t x0 = fff[1] * u[0];
    const real_t x1 = (1.0 / 2.0) * u[0];
    const real_t x2 = (1.0 / 6.0) * u[1];
    const real_t x3 = fff[1] * x2;
    const real_t x4 = (1.0 / 6.0) * u[2];
    const real_t x5 = fff[1] * x4;
    const real_t x6 = (2.0 / 3.0) * u[3];
    const real_t x7 = -fff[0] * x6 - fff[1] * x6;
    const real_t x8 = (2.0 / 3.0) * u[5];
    const real_t x9 = -fff[1] * x8 - fff[2] * x8;
    const real_t x10 = fff[0] * u[1];
    const real_t x11 = (1.0 / 6.0) * u[0];
    const real_t x12 = fff[1] * u[4];
    const real_t x13 = (1.0 / 6.0) * x0 + (2.0 / 3.0) * x12;
    const real_t x14 = fff[2] * u[2];
    const real_t x15 = (2.0 / 3.0) * u[0];
    const real_t x16 = (2.0 / 3.0) * x0;
    const real_t x17 = -4.0 / 3.0 * fff[1] * u[3];
    const real_t x18 = (4.0 / 3.0) * u[5];
    const real_t x19 = -fff[1] * x18;
    const real_t x20 = (4.0 / 3.0) * fff[2];
    const real_t x21 = (2.0 / 3.0) * fff[1];
    const real_t x22 = (4.0 / 3.0) * x12;
    const real_t x23 = u[1] * x21 - u[3] * x20 + u[4] * x20 + x17 + x19 + x22;
    const real_t x24 = (4.0 / 3.0) * fff[0] * u[4] - fff[0] * x18 + u[2] * x21;
    element_vector[0] = fff[0] * x1 + fff[0] * x2 +
                                 fff[2] * x1 + fff[2] * x4 + x0 + x3 + x5 + x7 +
                                 x9;
    element_vector[1] = fff[0] * x11 + (1.0 / 2.0) * x10 + x13 - x5 + x7;
    element_vector[2] = fff[2] * x11 + x13 + (1.0 / 2.0) * x14 - x3 + x9;
    element_vector[3] = (4.0 / 3.0) * fff[0] * u[3] - fff[0] * x15 -
                                 2.0 / 3.0 * x10 - x16 - x23;
    element_vector[4] = x23 + x24;
    element_vector[5] = (4.0 / 3.0) * fff[2] * u[5] - fff[2] * x15 -
                                 2.0 / 3.0 * x14 - x16 - x17 - x19 - x22 - x24;
}

static SFEM_INLINE void tri6_laplacian_hessian_fff(const jacobian_t *fff,
                                                   accumulator_t *element_matrix) {
    const real_t x0 = (1.0 / 2.0) * fff[0];
    const real_t x1 = (1.0 / 2.0) * fff[2];
    const real_t x2 = (1.0 / 6.0) * fff[1];
    const real_t x3 = (1.0 / 6.0) * fff[0] + x2;
    const real_t x4 = (1.0 / 6.0) * fff[2] + x2;
    const real_t x5 = (2.0 / 3.0) * fff[1];
    const real_t x6 = -2.0 / 3.0 * fff[0] - x5;
    const real_t x7 = -2.0 / 3.0 * fff[2] - x5;
    const real_t x8 = -x2;
    const real_t x9 = (4.0 / 3.0) * fff[0];
    const real_t x10 = (4.0 / 3.0) * fff[1];
    const real_t x11 = (4.0 / 3.0) * fff[2] + x10;
    const real_t x12 = x11 + x9;
    const real_t x13 = -x11;
    const real_t x14 = -x10 - x9;
    element_matrix[0] = fff[1] + x0 + x1;
    element_matrix[1] = x3;
    element_matrix[2] = x4;
    element_matrix[3] = x6;
    element_matrix[4] = 0;
    element_matrix[5] = x7;
    element_matrix[6] = x3;
    element_matrix[7] = x0;
    element_matrix[8] = x8;
    element_matrix[9] = x6;
    element_matrix[10] = x5;
    element_matrix[11] = 0;
    element_matrix[12] = x4;
    element_matrix[13] = x8;
    element_matrix[14] = x1;
    element_matrix[15] = 0;
    element_matrix[16] = x5;
    element_matrix[17] = x7;
    element_matrix[18] = x6;
    element_matrix[19] = x6;
    element_matrix[20] = 0;
    element_matrix[21] = x12;
    element_matrix[22] = x13;
    element_matrix[23] = x10;
    element_matrix[24] = 0;
    element_matrix[25] = x5;
    element_matrix[26] = x5;
    element_matrix[27] = x13;
    element_matrix[28] = x12;
    element_matrix[29] = x14;
    element_matrix[30] = x7;
    element_matrix[31] = 0;
    element_matrix[32] = x7;
    element_matrix[33] = x10;
    element_matrix[34] = x14;
    element_matrix[35] = x12;
}

static SFEM_INLINE void tri6_laplacian_diag_fff(const jacobian_t *const SFEM_RESTRICT fff,
                                                accumulator_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = (1.0 / 2.0) * fff[0];
    const real_t x1 = (1.0 / 2.0) * fff[2];
    const real_t x2 = (4.0 / 3.0) * fff[0] + (4.0 / 3.0) * fff[1] +
                      (4.0 / 3.0) * fff[2];
    element_vector[0] = fff[1] + x0 + x1;
    element_vector[1] = x0;
    element_vector[2] = x1;
    element_vector[3] = x2;
    element_vector[4] = x2;
    element_vector[5] = x2;
}

static SFEM_INLINE void tri6_laplacian_energy_fff(const jacobian_t *const SFEM_RESTRICT fff,
                                                  const scalar_t *const SFEM_RESTRICT u,
                                                  accumulator_t *const SFEM_RESTRICT
                                                          element_scalar) {
    const real_t x0 = (1.0 / 6.0) * u[1];
    const real_t x1 = (2.0 / 3.0) * fff[0];
    const real_t x2 = u[3] * x1;
    const real_t x3 = (4.0 / 3.0) * u[4];
    const real_t x4 = u[5] * x3;
    const real_t x5 = fff[1] * x0;
    const real_t x6 = (1.0 / 6.0) * u[0] * u[2];
    const real_t x7 = (2.0 / 3.0) * fff[1];
    const real_t x8 = u[0] * x7;
    const real_t x9 = u[1] * x7;
    const real_t x10 = u[2] * u[5];
    const real_t x11 = u[3] * x3;
    const real_t x12 = (2.0 / 3.0) * fff[2];
    const real_t x13 = POW2(u[0]);
    const real_t x14 = (1.0 / 4.0) * fff[0];
    const real_t x15 = POW2(u[3]);
    const real_t x16 = POW2(u[4]);
    const real_t x17 = POW2(u[5]);
    const real_t x18 = (1.0 / 4.0) * fff[2];
    element_scalar[0] = fff[0] * u[0] * x0 - fff[0] * x4 +
                        (4.0 / 3.0) * fff[1] * u[3] * u[5] - fff[1] * x11 +
                        (1.0 / 2.0) * fff[1] * x13 - fff[1] * x4 +
                        fff[1] * x6 - fff[2] * x11 + fff[2] * x6 -
                        u[0] * u[5] * x12 - u[0] * x2 + u[0] * x5 + POW2(u[1]) * x14 - u[1] * x2 +
                        POW2(u[2]) * x18 + u[2] * u[4] * x7 - u[2] * x5 - u[3] * x8 - u[3] * x9 +
                        u[4] * x9 - u[5] * x8 + x1 * x15 + x1 * x16 + x1 * x17 - x10 * x12 -
                        x10 * x7 + x12 * x15 + x12 * x16 + x12 * x17 + x13 * x14 + x13 * x18 +
                        x15 * x7 + x16 * x7 + x17 * x7;
}

#endif  // TRI6_LAPLACIAN_INLINE_CPU_HPP
