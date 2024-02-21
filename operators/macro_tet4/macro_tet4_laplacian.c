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
    const geom_t x4 = -pz0 + pz1;
    const geom_t x5 = -px0 + px2;
    const geom_t x6 = -py0 + py3;
    const geom_t x7 = x5 * x6;
    const geom_t x8 = -py0 + py1;
    const geom_t x9 = -px0 + px3;
    const geom_t x10 = -pz0 + pz2;
    const geom_t x11 = x10 * x6;
    const geom_t x12 = x2 * x5;
    const geom_t x13 = x1 * x9;
    const geom_t x14 = -x0 * x11 + x0 * x3 + x10 * x8 * x9 - x12 * x8 - x13 * x4 + x4 * x7;
    const geom_t x15 = -x13 + x7;
    const geom_t x16 = 1. / POW2(x14);
    const geom_t x17 = x10 * x9 - x12;
    const geom_t x18 = -x11 + x3;
    const geom_t x19 = -x0 * x6 + x8 * x9;
    const geom_t x20 = x15 * x16;
    const geom_t x21 = x0 * x2 - x4 * x9;
    const geom_t x22 = x16 * x17;
    const geom_t x23 = -x2 * x8 + x4 * x6;
    const geom_t x24 = x16 * x18;
    const geom_t x25 = x0 * x1 - x5 * x8;
    const geom_t x26 = -x0 * x10 + x4 * x5;
    const geom_t x27 = -x1 * x4 + x10 * x8;
    fff[0] = x14 * (POW2(x15) * x16 + x16 * POW2(x17) + x16 * POW2(x18));
    fff[1] = x14 * (x19 * x20 + x21 * x22 + x23 * x24);
    fff[2] = x14 * (x20 * x25 + x22 * x26 + x24 * x27);
    fff[3] = x14 * (x16 * POW2(x19) + x16 * POW2(x21) + x16 * POW2(x23));
    fff[4] = x14 * (x16 * x19 * x25 + x16 * x21 * x26 + x16 * x23 * x27);
    fff[5] = x14 * (x16 * POW2(x25) + x16 * POW2(x26) + x16 * POW2(x27));
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
    const real_t x0 = (1.0 / 6.0) * u0;
    const real_t x1 = fff[0] * x0;
    const real_t x2 = (1.0 / 6.0) * u1;
    const real_t x3 = fff[0] * x2;
    const real_t x4 = fff[1] * x2;
    const real_t x5 = (1.0 / 6.0) * u2;
    const real_t x6 = fff[1] * x5;
    const real_t x7 = fff[2] * x2;
    const real_t x8 = (1.0 / 6.0) * u3;
    const real_t x9 = fff[2] * x8;
    const real_t x10 = fff[3] * x0;
    const real_t x11 = fff[3] * x5;
    const real_t x12 = fff[4] * x5;
    const real_t x13 = fff[4] * x8;
    const real_t x14 = fff[5] * x0;
    const real_t x15 = fff[5] * x8;
    const real_t x16 = fff[1] * x0;
    const real_t x17 = fff[2] * x0;
    const real_t x18 = fff[4] * x0;
    *e0 += (1.0 / 3.0) * fff[1] * u0 + (1.0 / 3.0) * fff[2] * u0 + (1.0 / 3.0) * fff[4] * u0 + x1 +
           x10 - x11 - x12 - x13 + x14 - x15 - x3 - x4 - x6 - x7 - x9;
    *e1 += -x1 - x16 - x17 + x3 + x6 + x9;
    *e2 += -x10 + x11 + x13 - x16 - x18 + x4;
    *e3 += x12 - x14 + x15 - x17 - x18 + x7;
}

void macro_tet4_laplacian_apply(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                const real_t *const SFEM_RESTRICT u,
                                real_t *const SFEM_RESTRICT values) {
    // double tick = MPI_Wtime();

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

    // double tock = MPI_Wtime();
    // printf("macro_tet4_laplacian_apply %g (seconds)\n", tock - tick);
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
            const geom_t *const fff = &ctx->fff[i*6];

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
