#include "macro_tet4_laplacian.h"

#include <mpi.h>
#include <stdio.h>
#include "sfem_base.h"

#define POW2(a) ((a) * (a))

static void fff_micro_kernel(const geom_t px0,
                             const geom_t px1,
                             const geom_t px2,
                             const geom_t px3,
                             const geom_t py0,
                             const geom_t py1,
                             const geom_t py2,
                             const geom_t py3,
                             const geom_t pz0,
                             const geom_t pz1,
                             const geom_t pz2,
                             const geom_t pz3,
                             geom_t *fff) {
    const geom_t x0 = -px0 + px1;
    const geom_t x1 = -py0 + py2;
    const geom_t x2 = -pz0 + pz3;
    const geom_t x3 = x1 * x2;
    const geom_t x4 = x0 * x3;
    const geom_t x5 = -py0 + py3;
    const geom_t x6 = -pz0 + pz2;
    const geom_t x7 = x5 * x6;
    const geom_t x8 = x0 * x7;
    const geom_t x9 = -py0 + py1;
    const geom_t x10 = -px0 + px2;
    const geom_t x11 = x10 * x2;
    const geom_t x12 = x11 * x9;
    const geom_t x13 = -pz0 + pz1;
    const geom_t x14 = x10 * x5;
    const geom_t x15 = x13 * x14;
    const geom_t x16 = -px0 + px3;
    const geom_t x17 = x16 * x6 * x9;
    const geom_t x18 = x1 * x16;
    const geom_t x19 = x13 * x18;
    const geom_t x20 = -1.0 / 6.0 * x12 + (1.0 / 6.0) * x15 + (1.0 / 6.0) * x17 - 1.0 / 6.0 * x19 +
                       (1.0 / 6.0) * x4 - 1.0 / 6.0 * x8;
    const geom_t x21 = x14 - x18;
    const geom_t x22 = 1. / POW2(-x12 + x15 + x17 - x19 + x4 - x8);
    const geom_t x23 = -x11 + x16 * x6;
    const geom_t x24 = x3 - x7;
    const geom_t x25 = -x0 * x5 + x16 * x9;
    const geom_t x26 = x21 * x22;
    const geom_t x27 = x0 * x2 - x13 * x16;
    const geom_t x28 = x22 * x23;
    const geom_t x29 = x13 * x5 - x2 * x9;
    const geom_t x30 = x22 * x24;
    const geom_t x31 = x0 * x1 - x10 * x9;
    const geom_t x32 = -x0 * x6 + x10 * x13;
    const geom_t x33 = -x1 * x13 + x6 * x9;
    fff[0] = x20 * (POW2(x21) * x22 + x22 * POW2(x23) + x22 * POW2(x24));
    fff[1] = x20 * (x25 * x26 + x27 * x28 + x29 * x30);
    fff[2] = x20 * (x26 * x31 + x28 * x32 + x30 * x33);
    fff[3] = x20 * (x22 * POW2(x25) + x22 * POW2(x27) + x22 * POW2(x29));
    fff[4] = x20 * (x22 * x25 * x31 + x22 * x27 * x32 + x22 * x29 * x33);
    fff[5] = x20 * (x22 * POW2(x31) + x22 * POW2(x32) + x22 * POW2(x33));
}

static SFEM_INLINE void sub_fff_0(const geom_t *const SFEM_RESTRICT fff,
                                  geom_t *const SFEM_RESTRICT sub_fff) {
    sub_fff[0] = (1.0 / 2.0) * fff[0];
    sub_fff[1] = (1.0 / 2.0) * fff[1];
    sub_fff[2] = (1.0 / 2.0) * fff[2];
    sub_fff[3] = (1.0 / 2.0) * fff[3];
    sub_fff[4] = (1.0 / 2.0) * fff[4];
    sub_fff[5] = (1.0 / 2.0) * fff[5];
}

static SFEM_INLINE void sub_fff_4(const geom_t *const SFEM_RESTRICT fff,
                                  geom_t *const SFEM_RESTRICT sub_fff) {
    const geom_t x0 = (1.0 / 2.0) * fff[0];
    const geom_t x1 = (1.0 / 2.0) * fff[2];
    sub_fff[0] = fff[1] + (1.0 / 2.0) * fff[3] + x0;
    sub_fff[1] = -1.0 / 2.0 * fff[1] - x0;
    sub_fff[2] = (1.0 / 2.0) * fff[4] + x1;
    sub_fff[3] = x0;
    sub_fff[4] = -x1;
    sub_fff[5] = (1.0 / 2.0) * fff[5];
}

static SFEM_INLINE void sub_fff_5(const geom_t *const SFEM_RESTRICT fff,
                                  geom_t *const SFEM_RESTRICT sub_fff) {
    const geom_t x0 = (1.0 / 2.0) * fff[3];
    const geom_t x1 = fff[4] + (1.0 / 2.0) * fff[5] + x0;
    const geom_t x2 = (1.0 / 2.0) * fff[4] + x0;
    const geom_t x3 = (1.0 / 2.0) * fff[1];
    sub_fff[0] = x1;
    sub_fff[1] = -x2;
    sub_fff[2] = -1.0 / 2.0 * fff[2] - x1 - x3;
    sub_fff[3] = x0;
    sub_fff[4] = x2 + x3;
    sub_fff[5] = (1.0 / 2.0) * fff[0] + fff[1] + fff[2] + x1;
}

static SFEM_INLINE void sub_fff_6(const geom_t *const SFEM_RESTRICT fff,
                                  geom_t *const SFEM_RESTRICT sub_fff) {
    const geom_t x0 = (1.0 / 2.0) * fff[3];
    const geom_t x1 = (1.0 / 2.0) * fff[4];
    const geom_t x2 = (1.0 / 2.0) * fff[1] + x0;
    sub_fff[0] = (1.0 / 2.0) * fff[0] + fff[1] + x0;
    sub_fff[1] = (1.0 / 2.0) * fff[2] + x1 + x2;
    sub_fff[2] = -x2;
    sub_fff[3] = fff[4] + (1.0 / 2.0) * fff[5] + x0;
    sub_fff[4] = -x0 - x1;
    sub_fff[5] = x0;
}

static SFEM_INLINE void sub_fff_7(const geom_t *const SFEM_RESTRICT fff,
                                  geom_t *const SFEM_RESTRICT sub_fff) {
    const geom_t x0 = (1.0 / 2.0) * fff[5];
    const geom_t x1 = (1.0 / 2.0) * fff[2];
    sub_fff[0] = x0;
    sub_fff[1] = -1.0 / 2.0 * fff[4] - x0;
    sub_fff[2] = -x1;
    sub_fff[3] = (1.0 / 2.0) * fff[3] + fff[4] + x0;
    sub_fff[4] = (1.0 / 2.0) * fff[1] + x1;
    sub_fff[5] = (1.0 / 2.0) * fff[0];
}

static void lapl_apply_micro_kernel(const geom_t *const SFEM_RESTRICT fff,
                                    const real_t u0,
                                    const real_t u1,
                                    const real_t u2,
                                    const real_t u3,
                                    real_t *const SFEM_RESTRICT e0,
                                    real_t *const SFEM_RESTRICT e1,
                                    real_t *const SFEM_RESTRICT e2,
                                    real_t *const SFEM_RESTRICT e3) {
    const real_t x0 = fff[0] + fff[1] + fff[2];
    const real_t x1 = fff[1] + fff[3] + fff[4];
    const real_t x2 = fff[2] + fff[4] + fff[5];
    const real_t x3 = fff[1] * u0;
    const real_t x4 = fff[2] * u0;
    const real_t x5 = fff[4] * u0;
    *e0 += u0 * x0 + u0 * x1 + u0 * x2 - u1 * x0 - u2 * x1 - u3 * x2;
    *e1 += -fff[0] * u0 + fff[0] * u1 + fff[1] * u2 + fff[2] * u3 - x3 - x4;
    *e2 += fff[1] * u1 - fff[3] * u0 + fff[3] * u2 + fff[4] * u3 - x3 - x5;
    *e3 += fff[2] * u1 + fff[4] * u2 - fff[5] * u0 + fff[5] * u3 - x4 - x5;
}

static void lapl_diag_micro_kernel(const geom_t *const SFEM_RESTRICT fff,
                                   real_t *const SFEM_RESTRICT e0,
                                   real_t *const SFEM_RESTRICT e1,
                                   real_t *const SFEM_RESTRICT e2,
                                   real_t *const SFEM_RESTRICT e3) {
    *e0 += fff[0] + 2 * fff[1] + 2 * fff[2] + fff[3] +
           2 * fff[4] + fff[5];
    *e1 += fff[0];
    *e2 += fff[3];
    *e3 += fff[5];
}

void macro_tet4_laplacian_apply(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                const real_t *const SFEM_RESTRICT u,
                                real_t *const SFEM_RESTRICT values) {
#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[10];
            geom_t fff[6];
            geom_t sub_fff[6];
            real_t element_u[10];
            real_t element_vector[10] = {0};

#pragma unroll(10)
            for (int v = 0; v < 10; ++v) {
                ev[v] = elems[v][i];
            }

            for (int v = 0; v < 10; ++v) {
                element_u[v] = u[ev[v]];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            fff_micro_kernel(
                // X
                xyz[0][i0],
                xyz[0][i1],
                xyz[0][i2],
                xyz[0][i3],
                // Y
                xyz[1][i0],
                xyz[1][i1],
                xyz[1][i2],
                xyz[1][i3],
                // Z
                xyz[2][i0],
                xyz[2][i1],
                xyz[2][i2],
                xyz[2][i3],
                //
                fff);

            {  // Corner tests
                sub_fff_0(fff, sub_fff);

                // [0, 4, 6, 7],
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[0],
                                        element_u[4],
                                        element_u[6],
                                        element_u[7],
                                        &element_vector[0],
                                        &element_vector[4],
                                        &element_vector[6],
                                        &element_vector[7]);

                // [4, 1, 5, 8],
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[4],
                                        element_u[1],
                                        element_u[5],
                                        element_u[8],
                                        &element_vector[4],
                                        &element_vector[1],
                                        &element_vector[5],
                                        &element_vector[8]);

                // [6, 5, 2, 9],
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[6],
                                        element_u[5],
                                        element_u[2],
                                        element_u[9],
                                        &element_vector[6],
                                        &element_vector[5],
                                        &element_vector[2],
                                        &element_vector[9]);

                // [7, 8, 9, 3],
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[7],
                                        element_u[8],
                                        element_u[9],
                                        element_u[3],
                                        &element_vector[7],
                                        &element_vector[8],
                                        &element_vector[9],
                                        &element_vector[3]);
            }

            {  // Octahedron tets

                // [4, 5, 6, 8],
                sub_fff_4(fff, sub_fff);
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[4],
                                        element_u[5],
                                        element_u[6],
                                        element_u[8],
                                        &element_vector[4],
                                        &element_vector[5],
                                        &element_vector[6],
                                        &element_vector[8]);

                // [7, 4, 6, 8],
                sub_fff_5(fff, sub_fff);
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[7],
                                        element_u[4],
                                        element_u[6],
                                        element_u[8],
                                        &element_vector[7],
                                        &element_vector[4],
                                        &element_vector[6],
                                        &element_vector[8]);

                // [6, 5, 9, 8],
                sub_fff_6(fff, sub_fff);
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[6],
                                        element_u[5],
                                        element_u[9],
                                        element_u[8],
                                        &element_vector[6],
                                        &element_vector[5],
                                        &element_vector[9],
                                        &element_vector[8]);

                // [7, 6, 9, 8]]
                sub_fff_7(fff, sub_fff);
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[7],
                                        element_u[6],
                                        element_u[9],
                                        element_u[8],
                                        &element_vector[7],
                                        &element_vector[6],
                                        &element_vector[9],
                                        &element_vector[8]);
            }

#pragma unroll(10)
            for (int v = 0; v < 10; v++) {
#pragma omp atomic update
                values[ev[v]] += element_vector[v];
            }
        }
    }
}

void macro_tet4_laplacian_init(macro_tet4_laplacian_t *const ctx,
                               const ptrdiff_t nelements,
                               idx_t **const SFEM_RESTRICT elements,
                               geom_t **const SFEM_RESTRICT
                                   points) {  // Create FFF and store it on device
    geom_t *h_fff = (geom_t *)calloc(6 * nelements, sizeof(geom_t));

#pragma omp parallel
    {
#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; e++) {
            fff_micro_kernel(points[0][elements[0][e]],
                             points[0][elements[1][e]],
                             points[0][elements[2][e]],
                             points[0][elements[3][e]],
                             points[1][elements[0][e]],
                             points[1][elements[1][e]],
                             points[1][elements[2][e]],
                             points[1][elements[3][e]],
                             points[2][elements[0][e]],
                             points[2][elements[1][e]],
                             points[2][elements[2][e]],
                             points[2][elements[3][e]],
                             &h_fff[e * 6]);
        }
    }

    ctx->fff = h_fff;
    ctx->elements = elements;
    ctx->nelements = nelements;
}

void macro_tet4_laplacian_destroy(macro_tet4_laplacian_t *const ctx) {
    free(ctx->fff);
    ctx->fff = 0;
    ctx->nelements = 0;
    ctx->elements = 0;
}

void macro_tet4_laplacian_apply_opt(const macro_tet4_laplacian_t *const ctx,
                                    const real_t *const SFEM_RESTRICT u,
                                    real_t *const SFEM_RESTRICT values) {
#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < ctx->nelements; ++i) {
            idx_t ev[10];
            geom_t sub_fff[6];
            real_t element_u[10];
            real_t element_vector[10] = {0};
            const geom_t *const fff = &ctx->fff[i * 6];

#pragma unroll(10)
            for (int v = 0; v < 10; ++v) {
                ev[v] = ctx->elements[v][i];
            }

            for (int v = 0; v < 10; ++v) {
                element_u[v] = u[ev[v]];
            }

            {  // Corner tests
                sub_fff_0(fff, sub_fff);

                // [0, 4, 6, 7],
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[0],
                                        element_u[4],
                                        element_u[6],
                                        element_u[7],
                                        &element_vector[0],
                                        &element_vector[4],
                                        &element_vector[6],
                                        &element_vector[7]);

                // [4, 1, 5, 8],
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[4],
                                        element_u[1],
                                        element_u[5],
                                        element_u[8],
                                        &element_vector[4],
                                        &element_vector[1],
                                        &element_vector[5],
                                        &element_vector[8]);

                // [6, 5, 2, 9],
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[6],
                                        element_u[5],
                                        element_u[2],
                                        element_u[9],
                                        &element_vector[6],
                                        &element_vector[5],
                                        &element_vector[2],
                                        &element_vector[9]);

                // [7, 8, 9, 3],
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[7],
                                        element_u[8],
                                        element_u[9],
                                        element_u[3],
                                        &element_vector[7],
                                        &element_vector[8],
                                        &element_vector[9],
                                        &element_vector[3]);
            }

            {  // Octahedron tets

                // [4, 5, 6, 8],
                sub_fff_4(fff, sub_fff);
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[4],
                                        element_u[5],
                                        element_u[6],
                                        element_u[8],
                                        &element_vector[4],
                                        &element_vector[5],
                                        &element_vector[6],
                                        &element_vector[8]);

                // [7, 4, 6, 8],
                sub_fff_5(fff, sub_fff);
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[7],
                                        element_u[4],
                                        element_u[6],
                                        element_u[8],
                                        &element_vector[7],
                                        &element_vector[4],
                                        &element_vector[6],
                                        &element_vector[8]);

                // [6, 5, 9, 8],
                sub_fff_6(fff, sub_fff);
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[6],
                                        element_u[5],
                                        element_u[9],
                                        element_u[8],
                                        &element_vector[6],
                                        &element_vector[5],
                                        &element_vector[9],
                                        &element_vector[8]);

                // [7, 6, 9, 8]]
                sub_fff_7(fff, sub_fff);
                lapl_apply_micro_kernel(sub_fff,
                                        element_u[7],
                                        element_u[6],
                                        element_u[9],
                                        element_u[8],
                                        &element_vector[7],
                                        &element_vector[6],
                                        &element_vector[9],
                                        &element_vector[8]);
            }

#pragma unroll(10)
            for (int v = 0; v < 10; v++) {
#pragma omp atomic update
                values[ev[v]] += element_vector[v];
            }
        }
    }
}

void macro_tet4_laplacian_diag(const macro_tet4_laplacian_t *const ctx,
                               real_t *const SFEM_RESTRICT diag) {
#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < ctx->nelements; ++i) {
            idx_t ev[10];
            geom_t sub_fff[6];
            real_t element_u[10];
            real_t element_vector[10] = {0};
            const geom_t *const fff = &ctx->fff[i * 6];

#pragma unroll(10)
            for (int v = 0; v < 10; ++v) {
                ev[v] = ctx->elements[v][i];
            }

            {  // Corner tests
                sub_fff_0(fff, sub_fff);

                // [0, 4, 6, 7],
                lapl_diag_micro_kernel(sub_fff,
                                       &element_vector[0],
                                       &element_vector[4],
                                       &element_vector[6],
                                       &element_vector[7]);

                // [4, 1, 5, 8],
                lapl_diag_micro_kernel(sub_fff,
                                       &element_vector[4],
                                       &element_vector[1],
                                       &element_vector[5],
                                       &element_vector[8]);

                // [6, 5, 2, 9],
                lapl_diag_micro_kernel(sub_fff,
                                       &element_vector[6],
                                       &element_vector[5],
                                       &element_vector[2],
                                       &element_vector[9]);

                // [7, 8, 9, 3],
                lapl_diag_micro_kernel(sub_fff,
                                       &element_vector[7],
                                       &element_vector[8],
                                       &element_vector[9],
                                       &element_vector[3]);
            }

            {  // Octahedron tets

                // [4, 5, 6, 8],
                sub_fff_4(fff, sub_fff);
                lapl_diag_micro_kernel(sub_fff,
                                       &element_vector[4],
                                       &element_vector[5],
                                       &element_vector[6],
                                       &element_vector[8]);

                // [7, 4, 6, 8],
                sub_fff_5(fff, sub_fff);
                lapl_diag_micro_kernel(sub_fff,
                                       &element_vector[7],
                                       &element_vector[4],
                                       &element_vector[6],
                                       &element_vector[8]);

                // [6, 5, 9, 8],
                sub_fff_6(fff, sub_fff);
                lapl_diag_micro_kernel(sub_fff,
                                       &element_vector[6],
                                       &element_vector[5],
                                       &element_vector[9],
                                       &element_vector[8]);

                // [7, 6, 9, 8]]
                sub_fff_7(fff, sub_fff);
                lapl_diag_micro_kernel(sub_fff,
                                       &element_vector[7],
                                       &element_vector[6],
                                       &element_vector[9],
                                       &element_vector[8]);
            }

#pragma unroll(10)
            for (int v = 0; v < 10; v++) {
#pragma omp atomic update
                diag[ev[v]] += element_vector[v];
            }
        }
    }
}
