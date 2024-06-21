#ifndef TRI3_LAPLACIAN_INLINE_CPU_H
#define TRI3_LAPLACIAN_INLINE_CPU_H

#include "operator_inline_cpu.h"

// FFF
static SFEM_INLINE void tri3_laplacian_hessian_fff(const jacobian_t *const SFEM_RESTRICT fff,
                                                   accumulator_t *const SFEM_RESTRICT
                                                           element_matrix) {
    const real_t x0 = (1.0 / 2.0) * fff[0];
    const real_t x1 = (1.0 / 2.0) * fff[2];
    const real_t x2 = (1.0 / 2.0) * fff[1];
    const real_t x3 = -x0 - x2;
    const real_t x4 = -x1 - x2;
    element_matrix[0] = fff[1] + x0 + x1;
    element_matrix[1] = x3;
    element_matrix[2] = x4;
    element_matrix[3] = x3;
    element_matrix[4] = x0;
    element_matrix[5] = x2;
    element_matrix[6] = x4;
    element_matrix[7] = x2;
    element_matrix[8] = x1;
}

static SFEM_INLINE void tri3_laplacian_diag_fff(const jacobian_t *const SFEM_RESTRICT fff,
                                                accumulator_t *const SFEM_RESTRICT e0,
                                                accumulator_t *const SFEM_RESTRICT e1,
                                                accumulator_t *const SFEM_RESTRICT e2) {
    const real_t x0 = (1.0 / 2.0) * fff[0];
    const real_t x1 = (1.0 / 2.0) * fff[2];
    *e0 = fff[1] + x0 + x1;
    *e1 = x0;
    *e2 = x1;
}

static SFEM_INLINE void tri3_laplacian_diag_add_fff(const jacobian_t *const SFEM_RESTRICT fff,
                                                    accumulator_t *const SFEM_RESTRICT e0,
                                                    accumulator_t *const SFEM_RESTRICT e1,
                                                    accumulator_t *const SFEM_RESTRICT e2) {
    const real_t x0 = (1.0 / 2.0) * fff[0];
    const real_t x1 = (1.0 / 2.0) * fff[2];
    *e0 += fff[1] + x0 + x1;
    *e1 += x0;
    *e2 += x1;
}

static SFEM_INLINE void tri3_laplacian_apply_fff(const jacobian_t *const SFEM_RESTRICT fff,
                                                 const scalar_t u0,
                                                 const scalar_t u1,
                                                 const scalar_t u2,
                                                 accumulator_t *const SFEM_RESTRICT e0,
                                                 accumulator_t *const SFEM_RESTRICT e1,
                                                 accumulator_t *const SFEM_RESTRICT e2) {
    const real_t x0 = (1.0 / 2.0) * u0;
    const real_t x1 = fff[0] * x0;
    const real_t x2 = (1.0 / 2.0) * u1;
    const real_t x3 = fff[0] * x2;
    const real_t x4 = fff[1] * x2;
    const real_t x5 = (1.0 / 2.0) * u2;
    const real_t x6 = fff[1] * x5;
    const real_t x7 = fff[2] * x0;
    const real_t x8 = fff[2] * x5;
    const real_t x9 = (1.0 / 2.0) * fff[1] * u0;
    *e0 = fff[1] * u0 + x1 - x3 - x4 - x6 + x7 - x8;
    *e1 = -x1 + x3 + x6 - x9;
    *e2 = x4 - x7 + x8 - x9;
}

static SFEM_INLINE void tri3_laplacian_apply_add_fff(const jacobian_t *const SFEM_RESTRICT fff,
                                                     const scalar_t u0,
                                                     const scalar_t u1,
                                                     const scalar_t u2,
                                                     accumulator_t *const SFEM_RESTRICT e0,
                                                     accumulator_t *const SFEM_RESTRICT e1,
                                                     accumulator_t *const SFEM_RESTRICT e2) {
    const real_t x0 = (1.0 / 2.0) * u0;
    const real_t x1 = fff[0] * x0;
    const real_t x2 = (1.0 / 2.0) * u1;
    const real_t x3 = fff[0] * x2;
    const real_t x4 = fff[1] * x2;
    const real_t x5 = (1.0 / 2.0) * u2;
    const real_t x6 = fff[1] * x5;
    const real_t x7 = fff[2] * x0;
    const real_t x8 = fff[2] * x5;
    const real_t x9 = (1.0 / 2.0) * fff[1] * u0;
    *e0 += fff[1] * u0 + x1 - x3 - x4 - x6 + x7 - x8;
    *e1 += -x1 + x3 + x6 - x9;
    *e2 += x4 - x7 + x8 - x9;
}

static SFEM_INLINE void tri3_laplacian_energy_fff(const jacobian_t *const SFEM_RESTRICT fff,
                                                  const scalar_t u0,
                                                  const scalar_t u1,
                                                  const scalar_t u2,
                                                  accumulator_t *const SFEM_RESTRICT
                                                          element_scalar) {
    const real_t x0 = (1.0 / 2.0) * u0;
    const real_t x1 = (1.0 / 2.0) * fff[1];
    const real_t x2 = u1 * x1;
    const real_t x3 = POW2(u0);
    const real_t x4 = (1.0 / 4.0) * fff[0];
    const real_t x5 = (1.0 / 4.0) * fff[2];
    element_scalar[0] = -fff[0] * u1 * x0 - fff[2] * u2 * x0 - u0 * u2 * x1 - u0 * x2 +
                        POW2(u1) * x4 + POW2(u2) * x5 + u2 * x2 + x1 * x3 + x3 * x4 + x3 * x5;
}

static SFEM_INLINE void tri3_laplacian_energy_add_fff(const jacobian_t *const SFEM_RESTRICT fff,
                                                      const scalar_t u0,
                                                      const scalar_t u1,
                                                      const scalar_t u2,
                                                      accumulator_t *const SFEM_RESTRICT
                                                              element_scalar) {
    const real_t x0 = (1.0 / 2.0) * u0;
    const real_t x1 = (1.0 / 2.0) * fff[1];
    const real_t x2 = u1 * x1;
    const real_t x3 = POW2(u0);
    const real_t x4 = (1.0 / 4.0) * fff[0];
    const real_t x5 = (1.0 / 4.0) * fff[2];
    element_scalar[0] += -fff[0] * u1 * x0 - fff[2] * u2 * x0 - u0 * u2 * x1 - u0 * x2 +
                         POW2(u1) * x4 + POW2(u2) * x5 + u2 * x2 + x1 * x3 + x3 * x4 + x3 * x5;
}

#endif  // TRI3_LAPLACIAN_INLINE_CPU_H
