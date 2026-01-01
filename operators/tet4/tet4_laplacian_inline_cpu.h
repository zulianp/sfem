#ifndef TET4_LAPLACIAN_INLINE_CPU_H
#define TET4_LAPLACIAN_INLINE_CPU_H

#include "operator_inline_cpu.h"

// FFF
static SFEM_INLINE void tet4_laplacian_hessian_fff(const scalar_t *const SFEM_RESTRICT fff,
                                                   accumulator_t *const SFEM_RESTRICT
                                                           element_matrix) {
    const scalar_t x0 = -fff[0] - fff[1] - fff[2];
    const scalar_t x1 = -fff[1] - fff[3] - fff[4];
    const scalar_t x2 = -fff[2] - fff[4] - fff[5];
    element_matrix[0] = fff[0] + 2 * fff[1] + 2 * fff[2] + fff[3] + 2 * fff[4] + fff[5];
    element_matrix[1] = x0;
    element_matrix[2] = x1;
    element_matrix[3] = x2;
    element_matrix[4] = x0;
    element_matrix[5] = fff[0];
    element_matrix[6] = fff[1];
    element_matrix[7] = fff[2];
    element_matrix[8] = x1;
    element_matrix[9] = fff[1];
    element_matrix[10] = fff[3];
    element_matrix[11] = fff[4];
    element_matrix[12] = x2;
    element_matrix[13] = fff[2];
    element_matrix[14] = fff[4];
    element_matrix[15] = fff[5];
}

static SFEM_INLINE void tet4_laplacian_diag_fff(const scalar_t *const SFEM_RESTRICT fff,
                                                accumulator_t *const SFEM_RESTRICT e0,
                                                accumulator_t *const SFEM_RESTRICT e1,
                                                accumulator_t *const SFEM_RESTRICT e2,
                                                accumulator_t *const SFEM_RESTRICT e3) {
    *e0 = fff[0] + 2 * fff[1] + 2 * fff[2] + fff[3] + 2 * fff[4] + fff[5];
    *e1 = fff[0];
    *e2 = fff[3];
    *e3 = fff[5];
}

static SFEM_INLINE void tet4_laplacian_diag_add_fff(const scalar_t *const SFEM_RESTRICT fff,
                                                    accumulator_t *const SFEM_RESTRICT e0,
                                                    accumulator_t *const SFEM_RESTRICT e1,
                                                    accumulator_t *const SFEM_RESTRICT e2,
                                                    accumulator_t *const SFEM_RESTRICT e3) {
    *e0 += fff[0] + 2 * fff[1] + 2 * fff[2] + fff[3] + 2 * fff[4] + fff[5];
    *e1 += fff[0];
    *e2 += fff[3];
    *e3 += fff[5];
}

static SFEM_INLINE void tet4_laplacian_apply_fff(const scalar_t *const SFEM_RESTRICT fff,
                                                 const scalar_t u0,
                                                 const scalar_t u1,
                                                 const scalar_t u2,
                                                 const scalar_t u3,
                                                 accumulator_t *const SFEM_RESTRICT e0,
                                                 accumulator_t *const SFEM_RESTRICT e1,
                                                 accumulator_t *const SFEM_RESTRICT e2,
                                                 accumulator_t *const SFEM_RESTRICT e3) {
    const scalar_t x0 = fff[0] + fff[1] + fff[2];
    const scalar_t x1 = fff[1] + fff[3] + fff[4];
    const scalar_t x2 = fff[2] + fff[4] + fff[5];
    const scalar_t x3 = fff[1] * u0;
    const scalar_t x4 = fff[2] * u0;
    const scalar_t x5 = fff[4] * u0;
    *e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
    *e1 = -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;
    *e2 = fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;
    *e3 = fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;
}

// SoA-style wrapper (useful for vectorization over element index)
static SFEM_INLINE void tet4_laplacian_apply_fff_soa(const scalar_t                fff0,
                                                     const scalar_t                fff1,
                                                     const scalar_t                fff2,
                                                     const scalar_t                fff3,
                                                     const scalar_t                fff4,
                                                     const scalar_t                fff5,
                                                     const scalar_t                u0,
                                                     const scalar_t                u1,
                                                     const scalar_t                u2,
                                                     const scalar_t                u3,
                                                     accumulator_t *const SFEM_RESTRICT e0,
                                                     accumulator_t *const SFEM_RESTRICT e1,
                                                     accumulator_t *const SFEM_RESTRICT e2,
                                                     accumulator_t *const SFEM_RESTRICT e3) {
    const scalar_t x0 = fff0 + fff1 + fff2;
    const scalar_t x1 = fff1 + fff3 + fff4;
    const scalar_t x2 = fff2 + fff4 + fff5;
    const scalar_t x3 = fff1 * u0;
    const scalar_t x4 = fff2 * u0;
    const scalar_t x5 = fff4 * u0;
    *e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
    *e1 = -fff0 * u0 + fff0 * u1 + fff1 * u2 + fff2 * u3 - x3 - x4;
    *e2 = fff1 * u1 - fff3 * u0 + fff3 * u2 + fff4 * u3 - x3 - x5;
    *e3 = fff2 * u1 + fff4 * u2 - fff5 * u0 + fff5 * u3 - x4 - x5;
}

static SFEM_INLINE void tet4_laplacian_apply_add_fff(const scalar_t *const SFEM_RESTRICT fff,
                                                     const scalar_t u0,
                                                     const scalar_t u1,
                                                     const scalar_t u2,
                                                     const scalar_t u3,
                                                     accumulator_t *const SFEM_RESTRICT e0,
                                                     accumulator_t *const SFEM_RESTRICT e1,
                                                     accumulator_t *const SFEM_RESTRICT e2,
                                                     accumulator_t *const SFEM_RESTRICT e3) {
    const scalar_t x0 = fff[0] + fff[1] + fff[2];
    const scalar_t x1 = fff[1] + fff[3] + fff[4];
    const scalar_t x2 = fff[2] + fff[4] + fff[5];
    const scalar_t x3 = fff[1] * u0;
    const scalar_t x4 = fff[2] * u0;
    const scalar_t x5 = fff[4] * u0;
    *e0 += u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
    *e1 += -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;
    *e2 += fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;
    *e3 += fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;
}

// Points

static SFEM_INLINE void tet4_laplacian_apply_points(const scalar_t px0,
                                                    const scalar_t px1,
                                                    const scalar_t px2,
                                                    const scalar_t px3,
                                                    const scalar_t py0,
                                                    const scalar_t py1,
                                                    const scalar_t py2,
                                                    const scalar_t py3,
                                                    const scalar_t pz0,
                                                    const scalar_t pz1,
                                                    const scalar_t pz2,
                                                    const scalar_t pz3,
                                                    const scalar_t *SFEM_RESTRICT u,
                                                    accumulator_t *SFEM_RESTRICT element_vector) {
    scalar_t fff[6];
    tet4_fff_s(px0, px1, px2, px3, py0, py1, py2, py3, pz0, pz1, pz2, pz3, fff);

    const scalar_t x0 = fff[0] + fff[1] + fff[2];
    const scalar_t x1 = fff[1] + fff[3] + fff[4];
    const scalar_t x2 = fff[2] + fff[4] + fff[5];
    const scalar_t x3 = fff[1] * u[0];
    const scalar_t x4 = fff[2] * u[0];
    const scalar_t x5 = fff[4] * u[0];
    element_vector[0] = u[0] * x0 + u[0] * x1 + u[0] * x2 - u[1] * x0 - u[2] * x1 - u[3] * x2;
    element_vector[1] = -fff[0] * u[0] + fff[0] * u[1] + fff[1] * u[2] + fff[2] * u[3] - x3 - x4;
    element_vector[2] = fff[1] * u[1] - fff[3] * u[0] + fff[3] * u[2] + fff[4] * u[3] - x3 - x5;
    element_vector[3] = fff[2] * u[1] + fff[4] * u[2] - fff[5] * u[0] + fff[5] * u[3] - x4 - x5;
}

// UNTESTED
static SFEM_INLINE void tet4_laplacian_value_points(const scalar_t px0,
                                                    const scalar_t px1,
                                                    const scalar_t px2,
                                                    const scalar_t px3,
                                                    const scalar_t py0,
                                                    const scalar_t py1,
                                                    const scalar_t py2,
                                                    const scalar_t py3,
                                                    const scalar_t pz0,
                                                    const scalar_t pz1,
                                                    const scalar_t pz2,
                                                    const scalar_t pz3,
                                                    const scalar_t *SFEM_RESTRICT u,
                                                    accumulator_t *SFEM_RESTRICT element_scalar) {
    scalar_t fff[6];
    tet4_fff_s(px0, px1, px2, px3, py0, py1, py2, py3, pz0, pz1, pz2, pz3, fff);

    const scalar_t x0 = (scalar_t)(3.0 / 16.0) * u[1];
    const scalar_t x1 = fff[1] * u[0];
    const scalar_t x2 = (scalar_t)(5.0 / 16.0) * u[2];
    const scalar_t x3 = fff[2] * u[0];
    const scalar_t x4 = (scalar_t)(9.0 / 16.0) * u[3];
    const scalar_t x5 = u[0] * x2;
    const scalar_t x6 = u[0] * x4;
    const scalar_t x7 = POW2(u[0]);
    const scalar_t x8 = (scalar_t)(1.0 / 16.0) * x7;
    const scalar_t x9 = (scalar_t)(1.0 / 8.0) * x7;
    element_scalar[0] = -fff[0] * u[0] * x0 + (scalar_t)(1.0 / 8.0) * fff[0] * POW2(u[1]) +
                        fff[0] * x8 + (scalar_t)(3.0 / 8.0) * fff[1] * u[1] * u[2] + fff[1] * x9 +
                        (scalar_t)(5.0 / 8.0) * fff[2] * u[1] * u[3] + fff[2] * x9 +
                        (scalar_t)(1.0 / 4.0) * fff[3] * POW2(u[2]) - fff[3] * x5 + fff[3] * x8 +
                        (scalar_t)(3.0 / 4.0) * fff[4] * u[2] * u[3] - fff[4] * x5 - fff[4] * x6 +
                        fff[4] * x9 + (scalar_t)(1.0 / 2.0) * fff[5] * POW2(u[3]) - fff[5] * x6 +
                        fff[5] * x8 - x0 * x1 - x0 * x3 - x1 * x2 - x3 * x4;
}

static SFEM_INLINE void tet4_laplacian_hessian_points(const scalar_t px0,
                                                      const scalar_t px1,
                                                      const scalar_t px2,
                                                      const scalar_t px3,
                                                      const scalar_t py0,
                                                      const scalar_t py1,
                                                      const scalar_t py2,
                                                      const scalar_t py3,
                                                      const scalar_t pz0,
                                                      const scalar_t pz1,
                                                      const scalar_t pz2,
                                                      const scalar_t pz3,
                                                      accumulator_t *element_matrix) {
    scalar_t fff[6];
    tet4_fff_s(px0, px1, px2, px3, py0, py1, py2, py3, pz0, pz1, pz2, pz3, fff);
    tet4_laplacian_hessian_fff(fff, element_matrix);
}

static SFEM_INLINE void tet4_laplacian_diag_points(const scalar_t px0,
                                                   const scalar_t px1,
                                                   const scalar_t px2,
                                                   const scalar_t px3,
                                                   const scalar_t py0,
                                                   const scalar_t py1,
                                                   const scalar_t py2,
                                                   const scalar_t py3,
                                                   const scalar_t pz0,
                                                   const scalar_t pz1,
                                                   const scalar_t pz2,
                                                   const scalar_t pz3,
                                                   accumulator_t *SFEM_RESTRICT element_vector) {
    scalar_t fff[6];
    tet4_fff_s(px0, px1, px2, px3, py0, py1, py2, py3, pz0, pz1, pz2, pz3, fff);

    element_vector[0] = fff[0] + 2 * fff[1] + 2 * fff[2] + fff[3] + 2 * fff[4] + fff[5];
    element_vector[1] = fff[0];
    element_vector[2] = fff[3];
    element_vector[3] = fff[5];
}

#endif  // TET4_LAPLACIAN_INLINE_CPU_H
