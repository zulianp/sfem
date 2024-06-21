#ifndef TET4_LAPLACIAN_INLINE_CPU_H
#define TET4_LAPLACIAN_INLINE_CPU_H

#include "operator_inline_cpu.h"

#include "vtet4_inline_cpu.h"

// FFF
static SFEM_INLINE void vtet4_laplacian_hessian_fff(const vscalar_t *const SFEM_RESTRICT fff,
                                                    vscalar_t *const SFEM_RESTRICT element_matrix) {
    const vscalar_t x0 = -fff[0] - fff[1] - fff[2];
    const vscalar_t x1 = -fff[1] - fff[3] - fff[4];
    const vscalar_t x2 = -fff[2] - fff[4] - fff[5];
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

static SFEM_INLINE void vtet4_laplacian_diag_fff(const vscalar_t *const SFEM_RESTRICT fff,
                                                 vscalar_t *const SFEM_RESTRICT e0,
                                                 vscalar_t *const SFEM_RESTRICT e1,
                                                 vscalar_t *const SFEM_RESTRICT e2,
                                                 vscalar_t *const SFEM_RESTRICT e3) {
    *e0 = fff[0] + 2 * fff[1] + 2 * fff[2] + fff[3] + 2 * fff[4] + fff[5];
    *e1 = fff[0];
    *e2 = fff[3];
    *e3 = fff[5];
}

static SFEM_INLINE void vtet4_laplacian_diag_add_fff(const vscalar_t *const SFEM_RESTRICT fff,
                                                     vscalar_t *const SFEM_RESTRICT e0,
                                                     vscalar_t *const SFEM_RESTRICT e1,
                                                     vscalar_t *const SFEM_RESTRICT e2,
                                                     vscalar_t *const SFEM_RESTRICT e3) {
    *e0 += fff[0] + 2 * fff[1] + 2 * fff[2] + fff[3] + 2 * fff[4] + fff[5];
    *e1 += fff[0];
    *e2 += fff[3];
    *e3 += fff[5];
}

static SFEM_INLINE void vtet4_laplacian_apply_fff(const vscalar_t *const SFEM_RESTRICT fff,
                                                  const vscalar_t u0,
                                                  const vscalar_t u1,
                                                  const vscalar_t u2,
                                                  const vscalar_t u3,
                                                  vscalar_t *const SFEM_RESTRICT e0,
                                                  vscalar_t *const SFEM_RESTRICT e1,
                                                  vscalar_t *const SFEM_RESTRICT e2,
                                                  vscalar_t *const SFEM_RESTRICT e3) {
    const vscalar_t x0 = fff[0] + fff[1] + fff[2];
    const vscalar_t x1 = fff[1] + fff[3] + fff[4];
    const vscalar_t x2 = fff[2] + fff[4] + fff[5];
    const vscalar_t x3 = fff[1] * u0;
    const vscalar_t x4 = fff[2] * u0;
    const vscalar_t x5 = fff[4] * u0;
    *e0 = u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
    *e1 = -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;
    *e2 = fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;
    *e3 = fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;
}

static SFEM_INLINE void vtet4_laplacian_apply_add_fff(const vscalar_t *const SFEM_RESTRICT fff,
                                                      const vscalar_t u0,
                                                      const vscalar_t u1,
                                                      const vscalar_t u2,
                                                      const vscalar_t u3,
                                                      vscalar_t *const SFEM_RESTRICT e0,
                                                      vscalar_t *const SFEM_RESTRICT e1,
                                                      vscalar_t *const SFEM_RESTRICT e2,
                                                      vscalar_t *const SFEM_RESTRICT e3) {
    const vscalar_t x0 = fff[0] + fff[1] + fff[2];
    const vscalar_t x1 = fff[1] + fff[3] + fff[4];
    const vscalar_t x2 = fff[2] + fff[4] + fff[5];
    const vscalar_t x3 = fff[1] * u0;
    const vscalar_t x4 = fff[2] * u0;
    const vscalar_t x5 = fff[4] * u0;
    *e0 += u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
    *e1 += -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;
    *e2 += fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;
    *e3 += fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;
}

// Points

static SFEM_INLINE void vtet4_laplacian_apply_points(const vscalar_t px0,
                                                     const vscalar_t px1,
                                                     const vscalar_t px2,
                                                     const vscalar_t px3,
                                                     const vscalar_t py0,
                                                     const vscalar_t py1,
                                                     const vscalar_t py2,
                                                     const vscalar_t py3,
                                                     const vscalar_t pz0,
                                                     const vscalar_t pz1,
                                                     const vscalar_t pz2,
                                                     const vscalar_t pz3,
                                                     const vscalar_t *SFEM_RESTRICT u,
                                                     vscalar_t *SFEM_RESTRICT element_vector) {
    vscalar_t fff[6];
    vtet4_fff(px0, px1, px2, px3, py0, py1, py2, py3, pz0, pz1, pz2, pz3, fff);

    const vscalar_t x0 = fff[0] + fff[1] + fff[2];
    const vscalar_t x1 = fff[1] + fff[3] + fff[4];
    const vscalar_t x2 = fff[2] + fff[4] + fff[5];
    const vscalar_t x3 = fff[1] * u[0];
    const vscalar_t x4 = fff[2] * u[0];
    const vscalar_t x5 = fff[4] * u[0];
    element_vector[0] = u[0] * x0 + u[0] * x1 + u[0] * x2 - u[1] * x0 - u[2] * x1 - u[3] * x2;
    element_vector[1] = -fff[0] * u[0] + fff[0] * u[1] + fff[1] * u[2] + fff[2] * u[3] - x3 - x4;
    element_vector[2] = fff[1] * u[1] - fff[3] * u[0] + fff[3] * u[2] + fff[4] * u[3] - x3 - x5;
    element_vector[3] = fff[2] * u[1] + fff[4] * u[2] - fff[5] * u[0] + fff[5] * u[3] - x4 - x5;
}

// UNTESTED
static SFEM_INLINE void vtet4_laplacian_value_points(const vscalar_t px0,
                                                     const vscalar_t px1,
                                                     const vscalar_t px2,
                                                     const vscalar_t px3,
                                                     const vscalar_t py0,
                                                     const vscalar_t py1,
                                                     const vscalar_t py2,
                                                     const vscalar_t py3,
                                                     const vscalar_t pz0,
                                                     const vscalar_t pz1,
                                                     const vscalar_t pz2,
                                                     const vscalar_t pz3,
                                                     const vscalar_t *SFEM_RESTRICT u,
                                                     vscalar_t *SFEM_RESTRICT element_scalar) {
    vscalar_t fff[6];
    vtet4_fff(px0, px1, px2, px3, py0, py1, py2, py3, pz0, pz1, pz2, pz3, fff);

    const vscalar_t x0 = (3.0 / 16.0) * u[1];
    const vscalar_t x1 = fff[1] * u[0];
    const vscalar_t x2 = (5.0 / 16.0) * u[2];
    const vscalar_t x3 = fff[2] * u[0];
    const vscalar_t x4 = (9.0 / 16.0) * u[3];
    const vscalar_t x5 = u[0] * x2;
    const vscalar_t x6 = u[0] * x4;
    const vscalar_t x7 = POW2(u[0]);
    const vscalar_t x8 = (1.0 / 16.0) * x7;
    const vscalar_t x9 = (1.0 / 8.0) * x7;
    element_scalar[0] = -fff[0] * u[0] * x0 + (1.0 / 8.0) * fff[0] * POW2(u[1]) + fff[0] * x8 +
                        (3.0 / 8.0) * fff[1] * u[1] * u[2] + fff[1] * x9 +
                        (5.0 / 8.0) * fff[2] * u[1] * u[3] + fff[2] * x9 +
                        (1.0 / 4.0) * fff[3] * POW2(u[2]) - fff[3] * x5 + fff[3] * x8 +
                        (3.0 / 4.0) * fff[4] * u[2] * u[3] - fff[4] * x5 - fff[4] * x6 +
                        fff[4] * x9 + (1.0 / 2.0) * fff[5] * POW2(u[3]) - fff[5] * x6 +
                        fff[5] * x8 - x0 * x1 - x0 * x3 - x1 * x2 - x3 * x4;
}

static SFEM_INLINE void vtet4_laplacian_hessian_points(const vscalar_t px0,
                                                       const vscalar_t px1,
                                                       const vscalar_t px2,
                                                       const vscalar_t px3,
                                                       const vscalar_t py0,
                                                       const vscalar_t py1,
                                                       const vscalar_t py2,
                                                       const vscalar_t py3,
                                                       const vscalar_t pz0,
                                                       const vscalar_t pz1,
                                                       const vscalar_t pz2,
                                                       const vscalar_t pz3,
                                                       vscalar_t *element_matrix) {
    vscalar_t fff[6];
    vtet4_fff(px0, px1, px2, px3, py0, py1, py2, py3, pz0, pz1, pz2, pz3, fff);
    vtet4_laplacian_hessian_fff(fff, element_matrix);
}

static SFEM_INLINE void vtet4_laplacian_diag_points(const vscalar_t px0,
                                                    const vscalar_t px1,
                                                    const vscalar_t px2,
                                                    const vscalar_t px3,
                                                    const vscalar_t py0,
                                                    const vscalar_t py1,
                                                    const vscalar_t py2,
                                                    const vscalar_t py3,
                                                    const vscalar_t pz0,
                                                    const vscalar_t pz1,
                                                    const vscalar_t pz2,
                                                    const vscalar_t pz3,
                                                    vscalar_t *SFEM_RESTRICT element_vector) {
    vscalar_t fff[6];
    vtet4_fff(px0, px1, px2, px3, py0, py1, py2, py3, pz0, pz1, pz2, pz3, fff);

    element_vector[0] = fff[0] + 2 * fff[1] + 2 * fff[2] + fff[3] + 2 * fff[4] + fff[5];
    element_vector[1] = fff[0];
    element_vector[2] = fff[3];
    element_vector[3] = fff[5];
}

#endif  // TET4_LAPLACIAN_INLINE_CPU_H
