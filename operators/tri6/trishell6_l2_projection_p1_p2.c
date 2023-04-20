// #include "trishell6_l2_projection_p1_p2.h"

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#define POW2(a) ((a) * (a))

static SFEM_INLINE void trishell6_p1_p2_l2_projection_apply_kernel(const real_t px0,
                                                                   const real_t px1,
                                                                   const real_t px2,
                                                                   const real_t py0,
                                                                   const real_t py1,
                                                                   const real_t py2,
                                                                   const real_t pz0,
                                                                   const real_t pz1,
                                                                   const real_t pz2,
                                                                   // Data
                                                                   const real_t *const SFEM_RESTRICT u,
                                                                   // Output
                                                                   real_t *const SFEM_RESTRICT element_vector) {
    // FLOATING POINT OPS!
    //       - Result: 6*ADD + 6*ASSIGNMENT + 9*MUL
    //       - Subexpressions: 6*ADD + 3*DIV + 10*MUL + 8*POW + 7*SUB
    const real_t x0 = 7 * u[1];
    const real_t x1 = 7 * u[2];
    const real_t x2 = -px0 + px1;
    const real_t x3 = -px0 + px2;
    const real_t x4 = -py0 + py1;
    const real_t x5 = -py0 + py2;
    const real_t x6 = -pz0 + pz1;
    const real_t x7 = -pz0 + pz2;
    const real_t x8 =
        sqrt((POW2(x2) + POW2(x4) + POW2(x6)) * (POW2(x3) + POW2(x5) + POW2(x7)) - POW2(x2 * x3 + x4 * x5 + x6 * x7));
    const real_t x9 = (1.0 / 600.0) * x8;
    const real_t x10 = 7 * u[0];
    const real_t x11 = 2 * u[0];
    const real_t x12 = 2 * u[1];
    const real_t x13 = (1.0 / 50.0) * x8;
    const real_t x14 = 2 * u[2];
    element_vector[0] = x9 * (26 * u[0] + x0 + x1);
    element_vector[1] = x9 * (26 * u[1] + x1 + x10);
    element_vector[2] = x9 * (26 * u[2] + x0 + x10);
    element_vector[3] = x13 * (u[2] + x11 + x12);
    element_vector[4] = x13 * (u[0] + x12 + x14);
    element_vector[5] = x13 * (u[1] + x11 + x14);
}

static SFEM_INLINE void lumped_mass_kernel(const real_t px0,
                                           const real_t px1,
                                           const real_t px2,
                                           const real_t py0,
                                           const real_t py1,
                                           const real_t py2,
                                           const real_t pz0,
                                           const real_t pz1,
                                           const real_t pz2,
                                           real_t *const SFEM_RESTRICT element_matrix_diag) {
    // FLOATING POINT OPS!
    //       - Result: 6*ASSIGNMENT
    //       - Subexpressions: 6*ADD + 3*DIV + 4*MUL + 8*POW + 7*SUB
    const real_t x0 = -px0 + px1;
    const real_t x1 = -px0 + px2;
    const real_t x2 = -py0 + py1;
    const real_t x3 = -py0 + py2;
    const real_t x4 = -pz0 + pz1;
    const real_t x5 = -pz0 + pz2;
    const real_t x6 =
        sqrt((POW2(x0) + POW2(x2) + POW2(x4)) * (POW2(x1) + POW2(x3) + POW2(x5)) - POW2(x0 * x1 + x2 * x3 + x4 * x5));
    const real_t x7 = (1.0 / 15.0) * x6;
    const real_t x8 = (1.0 / 10.0) * x6;
    element_matrix_diag[0] = x7;
    element_matrix_diag[1] = x7;
    element_matrix_diag[2] = x7;
    element_matrix_diag[3] = x8;
    element_matrix_diag[4] = x8;
    element_matrix_diag[5] = x8;
}

static SFEM_INLINE void trishell6_transform_kernel(const real_t *const SFEM_RESTRICT x,
                                                   real_t *const SFEM_RESTRICT values) {
    // FLOATING POINT OPS!
    //       - Result: 3*ADD + 6*ASSIGNMENT + 3*MUL
    //       - Subexpressions: 3*DIV
    const real_t x0 = (1.0 / 5.0) * x[3];
    const real_t x1 = (1.0 / 5.0) * x[5];
    const real_t x2 = (1.0 / 5.0) * x[4];
    values[0] = x0 + x1 + x[0];
    values[1] = x0 + x2 + x[1];
    values[2] = x1 + x2 + x[2];
    values[3] = (3.0 / 5.0) * x[3];
    values[4] = (3.0 / 5.0) * x[4];
    values[5] = (3.0 / 5.0) * x[5];
}

void trishell6_ep1_p2_l2_projection_apply(const ptrdiff_t nelements,
                                          const ptrdiff_t nnodes,
                                          idx_t **const SFEM_RESTRICT elems,
                                          geom_t **const SFEM_RESTRICT xyz,
                                          const real_t *const SFEM_RESTRICT element_wise_p1,
                                          real_t *const SFEM_RESTRICT p2) {
    double tick = MPI_Wtime();

    idx_t ev[6];
    real_t element_p2[6];

    memset(p2, 0, nnodes * sizeof(real_t));

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(6)
        for (int v = 0; v < 6; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        trishell6_p1_p2_l2_projection_apply_kernel(
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
            // Z-coordinates
            xyz[2][i0],
            xyz[2][i1],
            xyz[2][i2],
            // Data
            &element_wise_p1[i * 3],
            // Output
            element_p2);

        for (int v = 0; v < 6; ++v) {
            const idx_t idx = ev[v];
            p2[idx] += element_p2[v];
        }
    }

    double tock = MPI_Wtime();
    printf("trishell6_l2_projection_p1_p2.c: trishell6_p1_p2_l2_projection_apply\t%g seconds\n", tock - tick);
}

void trishell6_ep1_p2_projection_coeffs(const ptrdiff_t nelements,
                                        const ptrdiff_t nnodes,
                                        idx_t **const SFEM_RESTRICT elems,
                                        geom_t **const SFEM_RESTRICT xyz,
                                        const real_t *const SFEM_RESTRICT element_wise_p1,
                                        real_t *const SFEM_RESTRICT p2) {
    double tick = MPI_Wtime();

    idx_t ev[6];

    real_t element_p2[6];
    real_t element_p2_pre_trafo[6];
    real_t element_weights[6];

    real_t *weights = (real_t *)malloc(nnodes * sizeof(real_t));
    memset(weights, 0, nnodes * sizeof(real_t));
    memset(p2, 0, nnodes * sizeof(real_t));

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(6)
        for (int v = 0; v < 6; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        trishell6_p1_p2_l2_projection_apply_kernel(
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
            // Z-coordinates
            xyz[2][i0],
            xyz[2][i1],
            xyz[2][i2],
            // Data
            &element_wise_p1[i * 3],
            // Output
            element_p2);

        lumped_mass_kernel(
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
            // Z-coordinates
            xyz[2][i0],
            xyz[2][i1],
            xyz[2][i2],
            element_weights);

        for (int v = 0; v < 6; ++v) {
            const idx_t idx = ev[v];
            p2[idx] += element_p2[v];
            weights[idx] += element_weights[v];
        }
    }

    for (ptrdiff_t i = 0; i < nnodes; i++) {
        p2[i] /= weights[i];
    }

    free(weights);

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(6)
        for (int v = 0; v < 6; ++v) {
            element_p2_pre_trafo[v] = p2[elems[v][i]];
        }

        trishell6_transform_kernel(element_p2_pre_trafo, element_p2);

        for (int v = 0; v < 6; ++v) {
            const idx_t idx = ev[v];
            p2[idx] = element_p2[v];
        }
    }

    double tock = MPI_Wtime();
    printf("trishell6_l2_projection_p0_p1.c: trishell6_p0_p1_projection_coeffs\t%g seconds\n", tock - tick);
}
