
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
    //       - Result: 4*ADD + 4*ASSIGNMENT + 52*MUL
    //       - Subexpressions: 44*ADD + 17*DIV + 181*MUL + 48*SUB
    const real_t x0 = (1.0 / 24.0) * px1;
    const real_t x1 = py3 * x0;
    const real_t x2 = uz[0] * x1;
    const real_t x3 = uz[1] * x1;
    const real_t x4 = uz[2] * x1;
    const real_t x5 = uz[3] * x1;
    const real_t x6 = (1.0 / 24.0) * px3;
    const real_t x7 = pz1 * x6;
    const real_t x8 = uy[0] * x7;
    const real_t x9 = uy[1] * x7;
    const real_t x10 = uy[2] * x7;
    const real_t x11 = uy[3] * x7;
    const real_t x12 = (1.0 / 24.0) * ux[0];
    const real_t x13 = py1 * pz3;
    const real_t x14 = x12 * x13;
    const real_t x15 = (1.0 / 24.0) * x13;
    const real_t x16 = ux[1] * x15;
    const real_t x17 = ux[2] * x15;
    const real_t x18 = ux[3] * x15;
    const real_t x19 = pz3 * x0;
    const real_t x20 = uy[0] * x19;
    const real_t x21 = uy[1] * x19;
    const real_t x22 = uy[2] * x19;
    const real_t x23 = uy[3] * x19;
    const real_t x24 = py1 * x6;
    const real_t x25 = uz[0] * x24;
    const real_t x26 = uz[1] * x24;
    const real_t x27 = uz[2] * x24;
    const real_t x28 = uz[3] * x24;
    const real_t x29 = py3 * pz1;
    const real_t x30 = x12 * x29;
    const real_t x31 = (1.0 / 24.0) * x29;
    const real_t x32 = ux[1] * x31;
    const real_t x33 = ux[2] * x31;
    const real_t x34 = ux[3] * x31;
    const real_t x35 = (1.0 / 24.0) * px2;
    const real_t x36 = pz3 * x35;
    const real_t x37 = py2 * uz[0];
    const real_t x38 = py2 * x6;
    const real_t x39 = py3 * pz2;
    const real_t x40 = (1.0 / 24.0) * x39;
    const real_t x41 = py3 * x35;
    const real_t x42 = pz2 * x6;
    const real_t x43 = py2 * pz3;
    const real_t x44 = (1.0 / 24.0) * x43;
    const real_t x45 = -ux[1] * x40 + ux[1] * x44 - ux[2] * x40 + ux[2] * x44 - ux[3] * x40 + ux[3] * x44 -
                       uy[0] * x36 + uy[0] * x42 - uy[1] * x36 + uy[1] * x42 - uy[2] * x36 + uy[2] * x42 - uy[3] * x36 +
                       uy[3] * x42 + uz[0] * x41 - uz[1] * x38 + uz[1] * x41 - uz[2] * x38 + uz[2] * x41 - uz[3] * x38 +
                       uz[3] * x41 - x12 * x39 + x12 * x43 - x37 * x6;
    const real_t x46 = pz2 * x0;
    const real_t x47 = py1 * x35;
    const real_t x48 = py2 * pz1;
    const real_t x49 = (1.0 / 24.0) * x48;
    const real_t x50 = py2 * x0;
    const real_t x51 = pz1 * x35;
    const real_t x52 = py1 * pz2;
    const real_t x53 = (1.0 / 24.0) * x52;
    const real_t x54 = -ux[1] * x49 + ux[1] * x53 - ux[2] * x49 + ux[2] * x53 - ux[3] * x49 + ux[3] * x53 -
                       uy[0] * x46 + uy[0] * x51 - uy[1] * x46 + uy[1] * x51 - uy[2] * x46 + uy[2] * x51 - uy[3] * x46 +
                       uy[3] * x51 - uz[0] * x47 - uz[1] * x47 + uz[1] * x50 - uz[2] * x47 + uz[2] * x50 - uz[3] * x47 +
                       uz[3] * x50 + x0 * x37 - x12 * x48 + x12 * x52;
    const real_t x55 = (1.0 / 24.0) * px0;
    const real_t x56 = pz2 * x55;
    const real_t x57 = uy[0] * x56;
    const real_t x58 = uy[1] * x56;
    const real_t x59 = uy[2] * x56;
    const real_t x60 = uy[3] * x56;
    const real_t x61 = py0 * x35;
    const real_t x62 = uz[0] * x61;
    const real_t x63 = uz[1] * x61;
    const real_t x64 = uz[2] * x61;
    const real_t x65 = uz[3] * x61;
    const real_t x66 = py2 * pz0;
    const real_t x67 = x12 * x66;
    const real_t x68 = (1.0 / 24.0) * x66;
    const real_t x69 = ux[1] * x68;
    const real_t x70 = ux[2] * x68;
    const real_t x71 = ux[3] * x68;
    const real_t x72 = x37 * x55;
    const real_t x73 = py2 * x55;
    const real_t x74 = uz[1] * x73;
    const real_t x75 = uz[2] * x73;
    const real_t x76 = uz[3] * x73;
    const real_t x77 = pz0 * x35;
    const real_t x78 = uy[0] * x77;
    const real_t x79 = uy[1] * x77;
    const real_t x80 = uy[2] * x77;
    const real_t x81 = uy[3] * x77;
    const real_t x82 = py0 * pz2;
    const real_t x83 = x12 * x82;
    const real_t x84 = (1.0 / 24.0) * x82;
    const real_t x85 = ux[1] * x84;
    const real_t x86 = ux[2] * x84;
    const real_t x87 = ux[3] * x84;
    const real_t x88 = py3 * x55;
    const real_t x89 = pz0 * x6;
    const real_t x90 = py0 * pz3;
    const real_t x91 = (1.0 / 24.0) * x90;
    const real_t x92 = pz3 * x55;
    const real_t x93 = py0 * x6;
    const real_t x94 = py3 * pz0;
    const real_t x95 = (1.0 / 24.0) * x94;
    const real_t x96 = -ux[1] * x91 + ux[1] * x95 - ux[2] * x91 + ux[2] * x95 - ux[3] * x91 + ux[3] * x95 -
                       uy[0] * x89 + uy[0] * x92 - uy[1] * x89 + uy[1] * x92 - uy[2] * x89 + uy[2] * x92 - uy[3] * x89 +
                       uy[3] * x92 - uz[0] * x88 + uz[0] * x93 - uz[1] * x88 + uz[1] * x93 - uz[2] * x88 + uz[2] * x93 -
                       uz[3] * x88 + uz[3] * x93 - x12 * x90 + x12 * x94;
    const real_t x97 = pz1 * x55;
    const real_t x98 = py0 * x0;
    const real_t x99 = py1 * pz0;
    const real_t x100 = (1.0 / 24.0) * x99;
    const real_t x101 = py1 * x55;
    const real_t x102 = pz0 * x0;
    const real_t x103 = py0 * pz1;
    const real_t x104 = (1.0 / 24.0) * x103;
    const real_t x105 = -ux[1] * x100 + ux[1] * x104 - ux[2] * x100 + ux[2] * x104 - ux[3] * x100 + ux[3] * x104 +
                        uy[0] * x102 - uy[0] * x97 + uy[1] * x102 - uy[1] * x97 + uy[2] * x102 - uy[2] * x97 +
                        uy[3] * x102 - uy[3] * x97 + uz[0] * x101 - uz[0] * x98 + uz[1] * x101 - uz[1] * x98 +
                        uz[2] * x101 - uz[2] * x98 + uz[3] * x101 - uz[3] * x98 + x103 * x12 - x12 * x99;
    element_vector[0] = -x10 - x11 - x14 - x16 - x17 - x18 - x2 + x20 + x21 + x22 + x23 + x25 + x26 + x27 + x28 - x3 +
                        x30 + x32 + x33 + x34 - x4 + x45 - x5 + x54 - x8 - x9;
    element_vector[1] = -x45 + x57 + x58 + x59 + x60 + x62 + x63 + x64 + x65 + x67 + x69 + x70 + x71 - x72 - x74 - x75 -
                        x76 - x78 - x79 - x80 - x81 - x83 - x85 - x86 - x87 - x96;
    element_vector[2] = x10 + x105 + x11 + x14 + x16 + x17 + x18 + x2 - x20 - x21 - x22 - x23 - x25 - x26 - x27 - x28 +
                        x3 - x30 - x32 - x33 - x34 + x4 + x5 + x8 + x9 + x96;
    element_vector[3] = -x105 - x54 - x57 - x58 - x59 - x60 - x62 - x63 - x64 - x65 - x67 - x69 - x70 - x71 + x72 +
                        x74 + x75 + x76 + x78 + x79 + x80 + x81 + x83 + x85 + x86 + x87;
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
