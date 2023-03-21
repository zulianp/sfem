
#include <assert.h>
#include <math.h>
#include <stddef.h>
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
    //      - Result: 4*ADD + 4*ASSIGNMENT + 108*MUL
    //      - Subexpressions: 24*DIV + 37*MUL + 2*NEG + 18*SUB
    const real_t x0 = (1.0 / 24.0) * px1;
    const real_t x1 = py2 * uz[0];
    const real_t x2 = py2 * x0;
    const real_t x3 = py3 * x0;
    const real_t x4 = pz2 * x0;
    const real_t x5 = pz3 * x0;
    const real_t x6 = (1.0 / 24.0) * px2;
    const real_t x7 = py1 * x6;
    const real_t x8 = py3 * x6;
    const real_t x9 = pz1 * x6;
    const real_t x10 = pz3 * x6;
    const real_t x11 = (1.0 / 24.0) * px3;
    const real_t x12 = py1 * x11;
    const real_t x13 = py2 * x11;
    const real_t x14 = pz1 * x11;
    const real_t x15 = pz2 * x11;
    const real_t x16 = (1.0 / 24.0) * ux[0];
    const real_t x17 = py1 * pz2;
    const real_t x18 = (1.0 / 24.0) * x17;
    const real_t x19 = py1 * pz3;
    const real_t x20 = (1.0 / 24.0) * x19;
    const real_t x21 = py2 * pz1;
    const real_t x22 = (1.0 / 24.0) * x21;
    const real_t x23 = py2 * pz3;
    const real_t x24 = (1.0 / 24.0) * x23;
    const real_t x25 = py3 * pz1;
    const real_t x26 = (1.0 / 24.0) * x25;
    const real_t x27 = py3 * pz2;
    const real_t x28 = (1.0 / 24.0) * x27;
    const real_t x29 = py0 - py2;
    const real_t x30 = pz0 - pz3;
    const real_t x31 = py0 - py3;
    const real_t x32 = pz0 - pz2;
    const real_t x33 = (1.0 / 24.0) * x29 * x30 - 1.0 / 24.0 * x31 * x32;
    const real_t x34 = px0 - px2;
    const real_t x35 = px0 - px3;
    const real_t x36 = x30 * x34 - x32 * x35;
    const real_t x37 = -1.0 / 24.0 * x29 * x35 + (1.0 / 24.0) * x31 * x34;
    const real_t x38 = py0 - py1;
    const real_t x39 = -x38;
    const real_t x40 = -x31;
    const real_t x41 = pz0 - pz1;
    const real_t x42 = -1.0 / 24.0 * x30 * x39 + (1.0 / 24.0) * x40 * x41;
    const real_t x43 = px0 - px1;
    const real_t x44 = (1.0 / 24.0) * x30 * x43 - 1.0 / 24.0 * x35 * x41;
    const real_t x45 = (1.0 / 24.0) * x35 * x39 - 1.0 / 24.0 * x40 * x43;
    const real_t x46 = -1.0 / 24.0 * x29 * x41 + (1.0 / 24.0) * x32 * x38;
    const real_t x47 = x32 * x43 - x34 * x41;
    const real_t x48 = (1.0 / 24.0) * x29 * x43 - 1.0 / 24.0 * x34 * x38;
    element_vector[0] =
        ux[1] * x18 - ux[1] * x20 - ux[1] * x22 + ux[1] * x24 + ux[1] * x26 - ux[1] * x28 + ux[2] * x18 - ux[2] * x20 -
        ux[2] * x22 + ux[2] * x24 + ux[2] * x26 - ux[2] * x28 + ux[3] * x18 - ux[3] * x20 - ux[3] * x22 + ux[3] * x24 +
        ux[3] * x26 - ux[3] * x28 - uy[0] * x10 - uy[0] * x14 + uy[0] * x15 - uy[0] * x4 + uy[0] * x5 + uy[0] * x9 -
        uy[1] * x10 - uy[1] * x14 + uy[1] * x15 - uy[1] * x4 + uy[1] * x5 + uy[1] * x9 - uy[2] * x10 - uy[2] * x14 +
        uy[2] * x15 - uy[2] * x4 + uy[2] * x5 + uy[2] * x9 - uy[3] * x10 - uy[3] * x14 + uy[3] * x15 - uy[3] * x4 +
        uy[3] * x5 + uy[3] * x9 + uz[0] * x12 - uz[0] * x3 - uz[0] * x7 + uz[0] * x8 + uz[1] * x12 - uz[1] * x13 +
        uz[1] * x2 - uz[1] * x3 - uz[1] * x7 + uz[1] * x8 + uz[2] * x12 - uz[2] * x13 + uz[2] * x2 - uz[2] * x3 -
        uz[2] * x7 + uz[2] * x8 + uz[3] * x12 - uz[3] * x13 + uz[3] * x2 - uz[3] * x3 - uz[3] * x7 + uz[3] * x8 +
        x0 * x1 - x1 * x11 + x16 * x17 - x16 * x19 - x16 * x21 + x16 * x23 + x16 * x25 - x16 * x27;
    element_vector[1] = -ux[0] * x33 - ux[1] * x33 - ux[2] * x33 - ux[3] * x33 + (1.0 / 24.0) * uy[0] * x36 +
                        (1.0 / 24.0) * uy[1] * x36 + (1.0 / 24.0) * uy[2] * x36 + (1.0 / 24.0) * uy[3] * x36 -
                        uz[0] * x37 - uz[1] * x37 - uz[2] * x37 - uz[3] * x37;
    element_vector[2] = ux[0] * x42 + ux[1] * x42 + ux[2] * x42 + ux[3] * x42 - uy[0] * x44 - uy[1] * x44 -
                        uy[2] * x44 - uy[3] * x44 + uz[0] * x45 + uz[1] * x45 + uz[2] * x45 + uz[3] * x45;
    element_vector[3] = -ux[0] * x46 - ux[1] * x46 - ux[2] * x46 - ux[3] * x46 + (1.0 / 24.0) * uy[0] * x47 +
                        (1.0 / 24.0) * uy[1] * x47 + (1.0 / 24.0) * uy[2] * x47 + (1.0 / 24.0) * uy[3] * x47 -
                        uz[0] * x48 - uz[1] * x48 - uz[2] * x48 - uz[3] * x48;
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

static SFEM_INLINE void ediv(const real_t px0,
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
                             real_t *element_value) {
    // FLOATING POINT OPS!
    //       - Result: 10*ADD + ASSIGNMENT + 40*MUL
    //       - Subexpressions: 16*MUL + 8*NEG + 13*SUB
    const real_t x0 = px0 - px1;
    const real_t x1 = py0 - py2;
    const real_t x2 = pz0 - pz3;
    const real_t x3 = x1 * x2;
    const real_t x4 = pz0 - pz1;
    const real_t x5 = px0 - px2;
    const real_t x6 = py0 - py3;
    const real_t x7 = x5 * x6;
    const real_t x8 = px0 - px3;
    const real_t x9 = py0 - py1;
    const real_t x10 = pz0 - pz2;
    const real_t x11 = x10 * x9;
    const real_t x12 = x10 * x6;
    const real_t x13 = x5 * x9;
    const real_t x14 = x1 * x4;
    const real_t x15 = -x9;
    const real_t x16 = -x2;
    const real_t x17 = -x6;
    const real_t x18 = -x4;
    const real_t x19 = x15 * x16 - x17 * x18;
    const real_t x20 = -x5;
    const real_t x21 = -x8;
    const real_t x22 = -x10;
    const real_t x23 = x16 * x20 - x21 * x22;
    const real_t x24 = -x0;
    const real_t x25 = -x18 * x20 + x22 * x24;
    const real_t x26 = -x15 * x21 + x17 * x24;
    const real_t x27 = x4 * x8;
    const real_t x28 = x1 * x8;
    element_value[0] =
        (1.0 / 36.0) * (-x0 * x12 + x0 * x3 + x11 * x8 - x13 * x2 - x14 * x8 + x4 * x7) *
        (-ux[0] * (x1 * x2 + x10 * x9 - x12 - x14 - x19) + ux[1] * (-x12 + x3) - ux[2] * x19 + ux[3] * (x11 - x14) -
         uy[0] * (x0 * x2 - x23 - x25 - x27) - uy[1] * x23 + uy[2] * (x0 * x2 - x27) - uy[3] * x25 -
         uz[0] * (x0 * x1 - x13 - x26 - x28 + x5 * x6) + uz[1] * (-x28 + x7) - uz[2] * x26 + uz[3] * (x0 * x1 - x13));
}

void integrate_div(const ptrdiff_t nelements,
                   const ptrdiff_t nnodes,
                   idx_t **const elems,
                   geom_t **const xyz,
                   const real_t *const ux,
                   const real_t *const uy,
                   const real_t *const uz,
                   real_t *const value) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[4];
    real_t element_ux[4];
    real_t element_uy[4];
    real_t element_uz[4];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];

            assert(ev[v] >= 0);
            assert(ev[v] < nnodes);
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


        real_t element_scalar = 0;

        ediv(
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
            &element_scalar);

        *value += element_scalar;
    }

    double tock = MPI_Wtime();
    printf("div.c: integrate_div\t%g seconds\n", tock - tick);
}
