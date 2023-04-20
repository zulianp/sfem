#include "tet10_l2_projection_p1_p2.h"

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>

static SFEM_INLINE void tet10_p1_p2_l2_projection_apply_kernel(const real_t px0,
                                                               const real_t px1,
                                                               const real_t px2,
                                                               const real_t px3,
                                                               const real_t py0,
                                                               const real_t py1,
                                                               const real_t py2,
                                                               const real_t py3,
                                                               const real_t pz0,
                                                               const real_t pz1,
                                                               const real_t pz2,
                                                               const real_t pz3,
                                                               // Data
                                                               const real_t *const SFEM_RESTRICT u,
                                                               // Output
                                                               real_t *const SFEM_RESTRICT element_vector) {
    // FLOATING POINT OPS!
    //       - Result: 10*ADD + 10*ASSIGNMENT + 14*MUL
    //       - Subexpressions: 8*ADD + 2*DIV + 16*MUL + 12*SUB
    const real_t x0 = u[2] + u[3];
    const real_t x1 = px0 - px1;
    const real_t x2 = py0 - py2;
    const real_t x3 = pz0 - pz3;
    const real_t x4 = px0 - px2;
    const real_t x5 = py0 - py3;
    const real_t x6 = pz0 - pz1;
    const real_t x7 = px0 - px3;
    const real_t x8 = py0 - py1;
    const real_t x9 = pz0 - pz2;
    const real_t x10 = x1 * x2 * x3 - x1 * x5 * x9 - x2 * x6 * x7 - x3 * x4 * x8 + x4 * x5 * x6 + x7 * x8 * x9;
    const real_t x11 = (1.0 / 600.0) * x10;
    const real_t x12 = u[0] + u[1];
    const real_t x13 = 2 * u[0];
    const real_t x14 = 2 * u[1];
    const real_t x15 = (1.0 / 300.0) * x10;
    const real_t x16 = 2 * u[2];
    const real_t x17 = u[3] + x16;
    const real_t x18 = u[0] + x14;
    const real_t x19 = u[1] + x13;
    const real_t x20 = 2 * u[3];
    const real_t x21 = u[2] + x20;
    element_vector[0] = -x11 * (4 * u[0] + u[1] + x0);
    element_vector[1] = -x11 * (u[0] + 4 * u[1] + x0);
    element_vector[2] = -x11 * (4 * u[2] + u[3] + x12);
    element_vector[3] = -x11 * (u[2] + 4 * u[3] + x12);
    element_vector[4] = -x15 * (x0 + x13 + x14);
    element_vector[5] = -x15 * (x17 + x18);
    element_vector[6] = -x15 * (x17 + x19);
    element_vector[7] = -x15 * (x19 + x21);
    element_vector[8] = -x15 * (x18 + x21);
    element_vector[9] = -x15 * (x12 + x16 + x20);
}

static SFEM_INLINE void lumped_mass_kernel(const real_t px0,
                                           const real_t px1,
                                           const real_t px2,
                                           const real_t px3,
                                           const real_t py0,
                                           const real_t py1,
                                           const real_t py2,
                                           const real_t py3,
                                           const real_t pz0,
                                           const real_t pz1,
                                           const real_t pz2,
                                           const real_t pz3,
                                           real_t *const SFEM_RESTRICT element_matrix_diag) {
    // generated code
    // FLOATING POINT OPS!
    //       - Result: 10*ASSIGNMENT
    //       - Subexpressions: 4*ADD + 12*DIV + 27*MUL + 15*SUB
    const real_t x0 = px0 - px1;
    const real_t x1 = py0 - py2;
    const real_t x2 = pz0 - pz3;
    const real_t x3 = x1 * x2;
    const real_t x4 = py0 - py3;
    const real_t x5 = pz0 - pz2;
    const real_t x6 = px0 - px2;
    const real_t x7 = py0 - py1;
    const real_t x8 = pz0 - pz1;
    const real_t x9 = x4 * x8;
    const real_t x10 = px0 - px3;
    const real_t x11 = x5 * x7;
    const real_t x12 = -7.0 / 600.0 * x0 * x3 + (7.0 / 600.0) * x0 * x4 * x5 + (7.0 / 600.0) * x1 * x10 * x8 -
                       7.0 / 600.0 * x10 * x11 + (7.0 / 600.0) * x2 * x6 * x7 - 7.0 / 600.0 * x6 * x9;
    const real_t x13 = -1.0 / 50.0 * x0 * x3 + (1.0 / 50.0) * x0 * x4 * x5 + (1.0 / 50.0) * x1 * x10 * x8 -
                       1.0 / 50.0 * x10 * x11 + (1.0 / 50.0) * x2 * x6 * x7 - 1.0 / 50.0 * x6 * x9;
    element_matrix_diag[0] = x12;
    element_matrix_diag[1] = x12;
    element_matrix_diag[2] = x12;
    element_matrix_diag[3] = x12;
    element_matrix_diag[4] = x13;
    element_matrix_diag[5] = x13;
    element_matrix_diag[6] = x13;
    element_matrix_diag[7] = x13;
    element_matrix_diag[8] = x13;
    element_matrix_diag[9] = x13;
}

static SFEM_INLINE void tet10_transform_kernel(const real_t *const SFEM_RESTRICT x,
                                               real_t *const SFEM_RESTRICT values) {
    // FLOATING POINT OPS!
    //       - Result: 4*ADD + 10*ASSIGNMENT + 6*MUL
    //       - Subexpressions: 6*DIV
    const real_t x0 = (1.0 / 5.0) * x[4];
    const real_t x1 = (1.0 / 5.0) * x[6];
    const real_t x2 = (1.0 / 5.0) * x[7];
    const real_t x3 = (1.0 / 5.0) * x[5];
    const real_t x4 = (1.0 / 5.0) * x[8];
    const real_t x5 = (1.0 / 5.0) * x[9];
    values[0] = x0 + x1 + x2 + x[0];
    values[1] = x0 + x3 + x4 + x[1];
    values[2] = x1 + x3 + x5 + x[2];
    values[3] = x2 + x4 + x5 + x[3];
    values[4] = (3.0 / 5.0) * x[4];
    values[5] = (3.0 / 5.0) * x[5];
    values[6] = (3.0 / 5.0) * x[6];
    values[7] = (3.0 / 5.0) * x[7];
    values[8] = (3.0 / 5.0) * x[8];
    values[9] = (3.0 / 5.0) * x[9];
}

void tet10_ep1_p2_l2_projection_apply(const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const SFEM_RESTRICT elems,
                                      geom_t **const SFEM_RESTRICT xyz,
                                      const real_t *const SFEM_RESTRICT element_wise_p1,
                                      real_t *const SFEM_RESTRICT p2) {
    double tick = MPI_Wtime();

    idx_t ev[10];
    real_t element_p2[10];

    memset(p2, 0, nnodes * sizeof(real_t));

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        tet10_p1_p2_l2_projection_apply_kernel(
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            xyz[0][i3],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
            xyz[1][i3],
            // Z-coordinates
            xyz[2][i0],
            xyz[2][i1],
            xyz[2][i2],
            xyz[2][i3],
            // Data
            &element_wise_p1[i * 4],
            // Output
            element_p2);

        for (int v = 0; v < 10; ++v) {
            const idx_t idx = ev[v];
            p2[idx] += element_p2[v];
        }
    }

    double tock = MPI_Wtime();
    printf("tet10_l2_projection_p1_p2.c: tet10_p1_p2_l2_projection_apply\t%g seconds\n", tock - tick);
}

void tet10_ep1_p2_projection_coeffs(const ptrdiff_t nelements,
                                    const ptrdiff_t nnodes,
                                    idx_t **const SFEM_RESTRICT elems,
                                    geom_t **const SFEM_RESTRICT xyz,
                                    const real_t *const SFEM_RESTRICT element_wise_p1,
                                    real_t *const SFEM_RESTRICT p2) {
    double tick = MPI_Wtime();

    idx_t ev[10];

    real_t element_p2[10];
    real_t element_p2_pre_trafo[10];
    real_t element_weights[10];

    real_t *weights = (real_t *)malloc(nnodes * sizeof(real_t));
    memset(weights, 0, nnodes * sizeof(real_t));
    memset(p2, 0, nnodes * sizeof(real_t));

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        tet10_p1_p2_l2_projection_apply_kernel(
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            xyz[0][i3],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
            xyz[1][i3],
            // Z-coordinates
            xyz[2][i0],
            xyz[2][i1],
            xyz[2][i2],
            xyz[2][i3],
            // Data
            &element_wise_p1[i * 4],
            // Output
            element_p2);

        lumped_mass_kernel(
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            xyz[0][i3],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
            xyz[1][i3],
            // Z-coordinates
            xyz[2][i0],
            xyz[2][i1],
            xyz[2][i2],
            xyz[2][i3],
            element_weights);

        for (int v = 0; v < 10; ++v) {
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
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            element_p2_pre_trafo[v] = p2[elems[v][i]];
        }

        tet10_transform_kernel(element_p2_pre_trafo, element_p2);

        for (int v = 0; v < 10; ++v) {
            const idx_t idx = ev[v];
            p2[idx] = element_p2[v];
        }
    }

    double tock = MPI_Wtime();
    printf("tet10_l2_projection_p0_p1.c: tet10_p0_p1_projection_coeffs\t%g seconds\n", tock - tick);
}
