
#include <stddef.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"

static SFEM_INLINE void div_gradient(const real_t px0,
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
                                     const real_t *ux,
                                     const real_t *uy,
                                     const real_t *uz,
                                     real_t *element_vector) {
    // FLOATING POINT OPS!
    //       - Result: 4*ASSIGNMENT
    //       - Subexpressions: 18*ADD + DIV + 97*MUL + 4*NEG + POW + 50*SUB
    const real_t x0 = px0 - px1;
    const real_t x1 = py0 - py2;
    const real_t x2 = pz0 - pz3;
    const real_t x3 = x1 * x2;
    const real_t x4 = px0 - px2;
    const real_t x5 = py0 - py3;
    const real_t x6 = pz0 - pz1;
    const real_t x7 = x5 * x6;
    const real_t x8 = py0 - py1;
    const real_t x9 = px0 - px3;
    const real_t x10 = pz0 - pz2;
    const real_t x11 = x10 * x9;
    const real_t x12 = -x0 * x10 * x5 + x0 * x3 - x1 * x6 * x9 + x11 * x8 - x2 * x4 * x8 + x4 * x7;
    const real_t x13 = -x12;
    const real_t x14 = -x10;
    const real_t x15 = -x4;
    const real_t x16 = x6 * x9;
    const real_t x17 = x0 * x2;
    const real_t x18 = -x10 * x5 + x3;
    const real_t x19 = -x1 * x6 + x10 * x8;
    const real_t x20 = -x1 * x9 + x4 * x5;
    const real_t x21 = x0 * x1 - x4 * x8;
    const real_t x22 = x8 * x9;
    const real_t x23 =
        (1.0 / 24.0) *
        (-uy[0] * x12 * (-x0 * x14 + x14 * x9 - x15 * x2 + x15 * x6 + x16 - x17) -
         x13 * (-ux[0] * (-x18 - x19 + x2 * x8 - x7) - ux[1] * x18 + ux[2] * (x2 * x8 - x7) - ux[3] * x19 +
                uy[1] * (-x11 + x2 * x4) - uy[2] * (-x16 + x17) + uy[3] * (x0 * x10 - x4 * x6) -
                uz[0] * (x0 * x5 - x20 - x21 - x22) - uz[1] * x20 + uz[2] * (x0 * x5 - x22) - uz[3] * x21)) *
        (-px0 * py1 * pz2 + px0 * py1 * pz3 + px0 * py2 * pz1 - px0 * py2 * pz3 - px0 * py3 * pz1 + px0 * py3 * pz2 +
         px1 * py0 * pz2 - px1 * py0 * pz3 - px1 * py2 * pz0 + px1 * py2 * pz3 + px1 * py3 * pz0 - px1 * py3 * pz2 -
         px2 * py0 * pz1 + px2 * py0 * pz3 + px2 * py1 * pz0 - px2 * py1 * pz3 - px2 * py3 * pz0 + px2 * py3 * pz1 +
         px3 * py0 * pz1 - px3 * py0 * pz2 - px3 * py1 * pz0 + px3 * py1 * pz2 + px3 * py2 * pz0 - px3 * py2 * pz1) /
        pow(x13, 2);
    element_vector[0] = x23;
    element_vector[1] = x23;
    element_vector[2] = x23;
    element_vector[3] = x23;
}

void div_apply(const ptrdiff_t nelements,
               const ptrdiff_t nnodes,
               idx_t **const elems,
               geom_t **const xyz,
               const real_t *const ux,
               const real_t *const uy,
               const real_t *const uz,
               real_t *const values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[4];
    real_t element_vector[4];
    real_t element_ux[4];
    real_t element_uy[4];
    real_t element_uz[4];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            element_ux[v] = ux[ev[v]];
        }

        for (int v = 0; v < 4; ++v) {
            element_uy[v] = uy[ev[v]];
        }

        for (int v = 0; v < 4; ++v) {
            element_uz[v] = uz[ev[v]];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        div_gradient(
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
            element_ux,
            element_uy,
            element_uz,
            // Output
            element_vector);

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t dof_i = ev[edof_i];
            values[dof_i] += element_vector[edof_i];
        }
    }

    double tock = MPI_Wtime();
    printf("div.c: div_apply\t%g seconds\n", tock - tick);
}
