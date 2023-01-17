#include "mass.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

static SFEM_INLINE void mass(const real_t x0,
                             const real_t x1,
                             const real_t x2,
                             const real_t x3,
                             const real_t y0,
                             const real_t y1,
                             const real_t y2,
                             const real_t y3,
                             const real_t z0,
                             const real_t z1,
                             const real_t z2,
                             const real_t z3,
                             real_t *element_matrix) {
    // FLOATING POINT OPS!
    //  - Result: 16*ASSIGNMENT
    //  - Subexpressions: 22*ADD + 32*DIV + 89*MUL + 24*SUB
    const real_t x4 = (1.0 / 60.0) * x0;
    const real_t x5 = y1 * z2;
    const real_t x6 = (1.0 / 60.0) * x1;
    const real_t x7 = y0 * z3;
    const real_t x8 = y3 * z2;
    const real_t x9 = (1.0 / 60.0) * x2;
    const real_t x10 = y0 * z1;
    const real_t x11 = y3 * z0;
    const real_t x12 = (1.0 / 60.0) * x3;
    const real_t x13 = y0 * z2;
    const real_t x14 = y2 * z1;
    const real_t x15 = (1.0 / 60.0) * x0 * y1 * z3 + (1.0 / 60.0) * x0 * y2 * z1 + (1.0 / 60.0) * x0 * y3 * z2 +
            (1.0 / 60.0) * x1 * y0 * z2 + (1.0 / 60.0) * x1 * y2 * z3 + (1.0 / 60.0) * x1 * y3 * z0 - x10 * x9 -
            x11 * x9 - x12 * x13 - x12 * x14 - x12 * y1 * z0 + (1.0 / 60.0) * x2 * y0 * z3 +
            (1.0 / 60.0) * x2 * y1 * z0 + (1.0 / 60.0) * x2 * y3 * z1 + (1.0 / 60.0) * x3 * y0 * z1 +
            (1.0 / 60.0) * x3 * y1 * z2 + (1.0 / 60.0) * x3 * y2 * z0 - x4 * x5 - x4 * y2 * z3 - x4 * y3 * z1 -
            x6 * x7 - x6 * x8 - x6 * y2 * z0 - x9 * y1 * z3;
    const real_t x16 = (1.0 / 120.0) * x0;
    const real_t x17 = (1.0 / 120.0) * x1;
    const real_t x18 = (1.0 / 120.0) * x2;
    const real_t x19 = (1.0 / 120.0) * x3;
    const real_t x20 = (1.0 / 120.0) * x0 * y1 * z3 + (1.0 / 120.0) * x0 * y2 * z1 + (1.0 / 120.0) * x0 * y3 * z2 +
            (1.0 / 120.0) * x1 * y0 * z2 + (1.0 / 120.0) * x1 * y2 * z3 + (1.0 / 120.0) * x1 * y3 * z0 - x10 * x18 -
            x11 * x18 - x13 * x19 - x14 * x19 - x16 * x5 - x16 * y2 * z3 - x16 * y3 * z1 - x17 * x7 - x17 * x8 -
            x17 * y2 * z0 - x18 * y1 * z3 - x19 * y1 * z0 + (1.0 / 120.0) * x2 * y0 * z3 +
            (1.0 / 120.0) * x2 * y1 * z0 + (1.0 / 120.0) * x2 * y3 * z1 + (1.0 / 120.0) * x3 * y0 * z1 +
            (1.0 / 120.0) * x3 * y1 * z2 + (1.0 / 120.0) * x3 * y2 * z0;
    element_matrix[0] = x15;
    element_matrix[1] = x20;
    element_matrix[2] = x20;
    element_matrix[3] = x20;
    element_matrix[4] = x20;
    element_matrix[5] = x15;
    element_matrix[6] = x20;
    element_matrix[7] = x20;
    element_matrix[8] = x20;
    element_matrix[9] = x20;
    element_matrix[10] = x15;
    element_matrix[11] = x20;
    element_matrix[12] = x20;
    element_matrix[13] = x20;
    element_matrix[14] = x20;
    element_matrix[15] = x15;
}

void assemble_mass(const ptrdiff_t nelements,
                   const ptrdiff_t nnodes,
                   idx_t *const elems[4],
                   geom_t *const xyz[3],
                   idx_t *const rowptr,
                   idx_t *const colidx,
                   real_t *const values) {
    double tick = MPI_Wtime();

    real_t element_matrix[4 * 4];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
        // Use code generated kernel

        // Element indices
        const idx_t i0 = elems[0][i];
        const idx_t i1 = elems[1][i];
        const idx_t i2 = elems[2][i];
        const idx_t i3 = elems[3][i];

        mass(
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
            element_matrix);

        // Local to global
        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            idx_t dof_i = elems[edof_i][i];
            idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

            idx_t *row = &colidx[rowptr[dof_i]];
            real_t *rowvalues = &values[rowptr[dof_i]];

            for (int edof_j = 0; edof_j < 4; ++edof_j) {
                idx_t dof_j = elems[edof_j][i];
                int k = -1;

                if (lenrow <= 8) {
                    k = find_idx(dof_j, row, lenrow);
                } else {
                    // Use this for larger number of dofs per row
                    k = find_idx_binary_search(dof_j, row, lenrow);
                }

                rowvalues[k] += element_matrix[edof_i * 4 + edof_j];
            }
        }
    }

    double tock = MPI_Wtime();
    printf("mass.c: assemble_mass\t%g seconds\n", tock - tick);
}
