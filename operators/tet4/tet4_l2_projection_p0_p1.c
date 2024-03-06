
#include "tet4_l2_projection_p0_p1.h"

#include <mpi.h>
#include <string.h>
#include <stdio.h>

static SFEM_INLINE void tet4_p0_p1_projection_coeffs_kernel(const real_t px0,
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
                                          const real_t *const SFEM_RESTRICT u_p0,
                                          // Output
                                          real_t *const SFEM_RESTRICT u_p1,
                                          real_t *const SFEM_RESTRICT weight) {
    // FLOATING POINT OPS!
    //       - Result: 8*ASSIGNMENT
    //       - Subexpressions: 2*ADD + 6*DIV + 13*MUL + 12*SUB
    const real_t x0 = py0 - py2;
    const real_t x1 = pz0 - pz3;
    const real_t x2 = px0 - px1;
    const real_t x3 = py0 - py3;
    const real_t x4 = pz0 - pz2;
    const real_t x5 = px0 - px2;
    const real_t x6 = py0 - py1;
    const real_t x7 = pz0 - pz1;
    const real_t x8 = px0 - px3;
    const real_t x9 = -1.0 / 24.0 * x0 * x1 * x2 + (1.0 / 24.0) * x0 * x7 * x8 + (1.0 / 24.0) * x1 * x5 * x6 +
                      (1.0 / 24.0) * x2 * x3 * x4 - 1.0 / 24.0 * x3 * x5 * x7 - 1.0 / 24.0 * x4 * x6 * x8;
    const real_t x10 = u_p0[0] * x9;
    weight[0] = x9;
    u_p1[0] = x10;
    weight[1] = x9;
    u_p1[1] = x10;
    weight[2] = x9;
    u_p1[2] = x10;
    weight[3] = x9;
    u_p1[3] = x10;
}

static SFEM_INLINE void tet4_p0_p1_l2_projection_apply_kernel(const real_t px0,
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
                                                  const real_t *const SFEM_RESTRICT u_p0,
                                                  // Output
                                                  real_t *const SFEM_RESTRICT u_p1) {
    // FLOATING POINT OPS!
    //       - Result: 4*ASSIGNMENT
    //       - Subexpressions: 2*ADD + 6*DIV + 13*MUL + 12*SUB
    const real_t x0 = py0 - py2;
    const real_t x1 = pz0 - pz3;
    const real_t x2 = px0 - px1;
    const real_t x3 = py0 - py3;
    const real_t x4 = pz0 - pz2;
    const real_t x5 = px0 - px2;
    const real_t x6 = py0 - py1;
    const real_t x7 = pz0 - pz1;
    const real_t x8 = px0 - px3;
    const real_t x9 =
        u_p0[0] * (-1.0 / 24.0 * x0 * x1 * x2 + (1.0 / 24.0) * x0 * x7 * x8 + (1.0 / 24.0) * x1 * x5 * x6 +
                   (1.0 / 24.0) * x2 * x3 * x4 - 1.0 / 24.0 * x3 * x5 * x7 - 1.0 / 24.0 * x4 * x6 * x8);
    u_p1[0] = x9;
    u_p1[1] = x9;
    u_p1[2] = x9;
    u_p1[3] = x9;
}

void tet4_p0_p1_projection_coeffs(const ptrdiff_t nelements,
                         const ptrdiff_t nnodes,
                         idx_t **const SFEM_RESTRICT elems,
                         geom_t **const SFEM_RESTRICT xyz,
                         const real_t *const SFEM_RESTRICT p0,
                         real_t *const SFEM_RESTRICT p1) {
    double tick = MPI_Wtime();

    idx_t ev[4];

    real_t element_p0;
    real_t element_p1[4];
    real_t element_weights[4];

    real_t *weights = (real_t *)malloc(nnodes * sizeof(real_t));
    memset(weights, 0, nnodes * sizeof(real_t));
    memset(p1, 0, nnodes * sizeof(real_t));

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        element_p0 = p0[i];

        tet4_p0_p1_projection_coeffs_kernel(
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
            &element_p0,
            // Output
            element_p1,
            element_weights);

        for (int v = 0; v < 4; ++v) {
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
    printf("tet4_l2_projection_p0_p1.c: tet4_p0_p1_projection_coeffs\t%g seconds\n", tock - tick);
}

void tet4_p0_p1_l2_projection_apply(const ptrdiff_t nelements,
                       const ptrdiff_t nnodes,
                       idx_t **const SFEM_RESTRICT elems,
                       geom_t **const SFEM_RESTRICT xyz,
                       const real_t *const SFEM_RESTRICT p0,
                       real_t *const SFEM_RESTRICT p1) {
    double tick = MPI_Wtime();

    idx_t ev[4];

    real_t element_p0;
    real_t element_p1[4];

    memset(p1, 0, nnodes * sizeof(real_t));

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        element_p0 = p0[i];

        tet4_p0_p1_l2_projection_apply_kernel(
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
            &element_p0,
            // Output
            element_p1);

        for (int v = 0; v < 4; ++v) {
            const idx_t idx = ev[v];
            p1[idx] += element_p1[v];
        }
    }

    double tock = MPI_Wtime();
    printf("tet4_l2_projection_p0_p1.c: tet4_p0_p1_l2_projection_apply\t%g seconds\n", tock - tick);
}
