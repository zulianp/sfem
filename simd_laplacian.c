#include "laplacian.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

static SFEM_INLINE void laplacian(const vreal_t x0,
                                  const vreal_t x1,
                                  const vreal_t x2,
                                  const vreal_t x3,
                                  const vreal_t y0,
                                  const vreal_t y1,
                                  const vreal_t y2,
                                  const vreal_t y3,
                                  const vreal_t z0,
                                  const vreal_t z1,
                                  const vreal_t z2,
                                  const vreal_t z3,
                                  vreal_t *element_matrix) {
    const vreal_t x4 = z0 - z3;
    const vreal_t x5 = x0 - x1;
    const vreal_t x6 = y0 - y2;
    const vreal_t x7 = x5 * x6;
    const vreal_t x8 = z0 - z1;
    const vreal_t x9 = x0 - x2;
    const vreal_t x10 = y0 - y3;
    const vreal_t x11 = x10 * x9;
    const vreal_t x12 = z0 - z2;
    const vreal_t x13 = x0 - x3;
    const vreal_t x14 = y0 - y1;
    const vreal_t x15 = x13 * x14;
    const vreal_t x16 = x10 * x5;
    const vreal_t x17 = x14 * x9;
    const vreal_t x18 = x13 * x6;
    const vreal_t x19 = x11 * x8 + x12 * x15 - x12 * x16 - x17 * x4 - x18 * x8 + x4 * x7;
    const vreal_t x20 = 1.0 / x19;
    const vreal_t x21 = x11 - x18;
    const vreal_t x22 = -x17 + x7;
    const vreal_t x23 = x15 - x16 + x21 + x22;
    const vreal_t x24 = -x12 * x13 + x4 * x9;
    const vreal_t x25 = x12 * x5 - x8 * x9;
    const vreal_t x26 = x13 * x8;
    const vreal_t x27 = x4 * x5;
    const vreal_t x28 = x26 - x27;
    const vreal_t x29 = -x24 - x25 - x28;
    const vreal_t x30 = x10 * x8;
    const vreal_t x31 = x14 * x4;
    const vreal_t x32 = -x10 * x12 + x4 * x6;
    const vreal_t x33 = x12 * x14 - x6 * x8;
    const vreal_t x34 = x30 - x31 + x32 + x33;
    const vreal_t x35 = -x12;
    const vreal_t x36 = -x9;
    const vreal_t x37 = x19 * (x13 * x35 + x28 - x35 * x5 - x36 * x4 + x36 * x8);
    const vreal_t x38 = -x19;
    const vreal_t x39 = -x23;
    const vreal_t x40 = -x34;
    const vreal_t x41 = (1.0 / 6.0) / (x19 * x19);
    const vreal_t x42 = x41 * (x24 * x37 + x38 * (x21 * x39 + x32 * x40));
    const vreal_t x43 = -x15 + x16;
    const vreal_t x44 = (1.0 / 6.0) * x43;
    const vreal_t x45 = -x26 + x27;
    const vreal_t x46 = -x30 + x31;
    const vreal_t x47 = (1.0 / 6.0) * x46;
    const vreal_t x48 = x20 * (-x23 * x44 + (1.0 / 6.0) * x29 * x45 - x34 * x47);
    const vreal_t x49 = x41 * (x25 * x37 + x38 * (x22 * x39 + x33 * x40));
    const vreal_t x50 = (1.0 / 6.0) * x45;
    const vreal_t x51 = x20 * (x21 * x44 + x24 * x50 + x32 * x47);
    const vreal_t x52 = x20 * (-1.0 / 6.0 * x21 * x22 - 1.0 / 6.0 * x24 * x25 - 1.0 / 6.0 * x32 * x33);
    const vreal_t x53 = x20 * (x22 * x44 + x25 * x50 + x33 * x47);

    element_matrix[0] = x20 * (-1.0 / 6.0 * (x23 * x23) - 1.0 / 6.0 * (x29 * x29) - 1.0 / 6.0 * (x34 * x34));
    element_matrix[1] = x42;
    element_matrix[2] = x48;
    element_matrix[3] = x49;
    element_matrix[4] = x42;
    element_matrix[5] = x20 * (-1.0 / 6.0 * (x21 * x21) - 1.0 / 6.0 * (x24 * x24) - 1.0 / 6.0 * (x32 * x32));
    element_matrix[6] = x51;
    element_matrix[7] = x52;
    element_matrix[8] = x48;
    element_matrix[9] = x51;
    element_matrix[10] = x20 * (-1.0 / 6.0 * (x43 * x43) - 1.0 / 6.0 * (x45 * x45) - 1.0 / 6.0 * (x46 * x46));
    element_matrix[11] = x53;
    element_matrix[12] = x49;
    element_matrix[13] = x52;
    element_matrix[14] = x53;
    element_matrix[15] = x20 * (-1.0 / 6.0 * (x22 * x22) - 1.0 / 6.0 * (x25 * x25) - 1.0 / 6.0 * (x33 * x33));
}

static SFEM_INLINE int find_col(const idx_t key, const idx_t *const row, const int lenrow)
{

    int k = -1;
    if (lenrow <= 32) 
    // 
    {
        // Using sentinel (potentially dangerous if matrix is buggy and column does not exist)
        while (key > row[++k]) {
            // Hi
        }
        assert(k < lenrow);
        assert(key == row[k]);
    } else {
        // Use this for larger number of dofs per row
        k = find_idx_binary_search(key, row, lenrow);
    }
    return k;
}

void assemble_laplacian(const ptrdiff_t nelements,
                        const ptrdiff_t nnodes,
                        idx_t *const elems[4],
                        geom_t *const xyz[3],
                        idx_t *const rowptr,
                        idx_t *const colidx,
                        real_t *const values) {
    double tick = MPI_Wtime();

    vreal_t element_matrix[4 * 4];

    vreal_t x[4];
    vreal_t y[4];
    vreal_t z[4];

    for (ptrdiff_t i = 0; i < nelements; i += SFEM_VECTOR_SIZE) {
        const int nvec = MIN(nelements - (i + SFEM_VECTOR_SIZE), SFEM_VECTOR_SIZE);

        for (int vi = 0; vi < nvec; ++vi) {
            const ptrdiff_t offset = i + vi;
            for (int d = 0; d < 4; ++d) {
                const idx_t vidx = elems[d][offset];
                x[d][vi] = xyz[0][vidx];
                y[d][vi] = xyz[1][vidx];
                z[d][vi] = xyz[2][vidx];
            }
        }

        laplacian(
            // X-coordinates
            x[0],
            x[1],
            x[2],
            x[3],
            // Y-coordinates
            y[0],
            y[1],
            y[2],
            y[3],
            // Z-coordinates
            z[0],
            z[1],
            z[2],
            z[3],
            element_matrix);

        // Local to global
        for (int vi = 0; vi < nvec; ++vi) {
            idx_t offset = i + vi;

            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t dof_i = elems[edof_i][offset];
                const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

                const idx_t *row = &colidx[rowptr[dof_i]];
                real_t *rowvalues = &values[rowptr[dof_i]];

                for (int edof_j = 0; edof_j < 4; ++edof_j) {
                    const idx_t dof_j = elems[edof_j][offset];
                    int k = find_col(dof_j, row, lenrow);
                    rowvalues[k] += element_matrix[edof_i * 4 + edof_j][vi];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("simd_laplacian.c: assemble_laplacian\t%g seconds\n", tock - tick);
}
