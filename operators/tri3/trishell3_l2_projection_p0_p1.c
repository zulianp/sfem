#include "trishell3_l2_projection_p0_p1.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <mpi.h>

#define POW2(a) ((a) * (a))

static SFEM_INLINE void surface_projection_p0_to_p1_kernel(const real_t px0,
                                                           const real_t px1,
                                                           const real_t px2,
                                                           const real_t py0,
                                                           const real_t py1,
                                                           const real_t py2,
                                                           const real_t pz0,
                                                           const real_t pz1,
                                                           const real_t pz2,
                                                           // Data
                                                           const real_t *const SFEM_RESTRICT u_p0,
                                                           // Output
                                                           real_t *const SFEM_RESTRICT u_p1,
                                                           real_t *const SFEM_RESTRICT weight) {
    // FLOATING POINT OPS!
    //       - Result: 6*ASSIGNMENT
    //       - Subexpressions: 6*ADD + 2*DIV + 5*MUL + 8*POW + 7*SUB
    const real_t x0 = -px0 + px1;
    const real_t x1 = -px0 + px2;
    const real_t x2 = -py0 + py1;
    const real_t x3 = -py0 + py2;
    const real_t x4 = -pz0 + pz1;
    const real_t x5 = -pz0 + pz2;
    const real_t x6 =
        (1.0 / 6.0) * sqrt((pow(x0, 2) + pow(x2, 2) + pow(x4, 2)) * (pow(x1, 2) + pow(x3, 2) + pow(x5, 2)) -
                           pow(x0 * x1 + x2 * x3 + x4 * x5, 2));
    const real_t x7 = u_p0[0] * x6;
    weight[0] = x6;
    u_p1[0] = x7;
    weight[1] = x6;
    u_p1[1] = x7;
    weight[2] = x6;
    u_p1[2] = x7;
}

static SFEM_INLINE void integrate_p0_to_p1_kernel(const real_t px0,
                                                  const real_t px1,
                                                  const real_t px2,
                                                  const real_t py0,
                                                  const real_t py1,
                                                  const real_t py2,
                                                  const real_t pz0,
                                                  const real_t pz1,
                                                  const real_t pz2,
                                                  // Data
                                                  const real_t *const SFEM_RESTRICT u_p0,
                                                  // Output
                                                  real_t *const SFEM_RESTRICT u_p1) {
    // FLOATING POINT OPS!
    //       - Result: 3*ASSIGNMENT
    //       - Subexpressions: 6*ADD + 2*DIV + 5*MUL + 8*POW + 7*SUB
    const real_t x0 = -px0 + px1;
    const real_t x1 = -px0 + px2;
    const real_t x2 = -py0 + py1;
    const real_t x3 = -py0 + py2;
    const real_t x4 = -pz0 + pz1;
    const real_t x5 = -pz0 + pz2;
    const real_t x6 = (1.0 / 6.0) * u_p0[0] *
                      sqrt((pow(x0, 2) + pow(x2, 2) + pow(x4, 2)) * (pow(x1, 2) + pow(x3, 2) + pow(x5, 2)) -
                           pow(x0 * x1 + x2 * x3 + x4 * x5, 2));
    u_p1[0] = x6;
    u_p1[1] = x6;
    u_p1[2] = x6;
}

void trishell3_p0_p1_l2_projection_apply(const ptrdiff_t nelements,
                                         const ptrdiff_t nnodes,
                                         idx_t **const SFEM_RESTRICT elems,
                                         geom_t **const SFEM_RESTRICT xyz,
                                         const real_t *const SFEM_RESTRICT p0,
                                         real_t *const SFEM_RESTRICT p1) {
    double tick = MPI_Wtime();

    idx_t ev[3];

    real_t element_p0;
    real_t element_p1[3];

    memset(p1, 0, nnodes * sizeof(real_t));

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        element_p0 = p0[i];

        integrate_p0_to_p1_kernel(
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
            &element_p0,
            // Output
            element_p1);

        for (int v = 0; v < 3; ++v) {
            const idx_t idx = ev[v];
            p1[idx] += element_p1[v];
        }
    }

    double tock = MPI_Wtime();
    printf("surface_projection_p0_to_p1.c: surface_assemble_p0_to_p1\t%g seconds\n", tock - tick);
}

void trishell3_p0_p1_projection_coeffs(const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const SFEM_RESTRICT elems,
                                       geom_t **const SFEM_RESTRICT xyz,
                                       const real_t *const SFEM_RESTRICT p0,
                                       real_t *const SFEM_RESTRICT p1) {
    double tick = MPI_Wtime();

    idx_t ev[3];

    real_t element_p0;
    real_t element_p1[3];
    real_t element_weights[3];

    real_t *weights = (real_t *)malloc(nnodes * sizeof(real_t));
    memset(weights, 0, nnodes * sizeof(real_t));
    memset(p1, 0, nnodes * sizeof(real_t));

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        element_p0 = p0[i];

        surface_projection_p0_to_p1_kernel(
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
            &element_p0,
            // Output
            element_p1,
            element_weights);

        for (int v = 0; v < 3; ++v) {
            const idx_t idx = ev[v];
            p1[idx] += element_p1[v];
            weights[idx] += element_weights[v];
        }
    }

    for (ptrdiff_t i = 0; i < nnodes; i++) {
        p1[i] /= weights[i];
    }

    free(weights);

    double tock = MPI_Wtime();
    printf("surface_projection_p0_to_p1.c: surface_projection_p0_to_p1\t%g seconds\n", tock - tick);
}
