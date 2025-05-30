#include "boundary_mass.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"

static SFEM_INLINE void boundary_mass(const real_t px0,
                                      const real_t px1,
                                      const real_t px2,
                                      const real_t py0,
                                      const real_t py1,
                                      const real_t py2,
                                      const real_t pz0,
                                      const real_t pz1,
                                      const real_t pz2,
                                      real_t *element_matrix) {
    // FLOATING POINT OPS!
    //       - Result: 9*ASSIGNMENT
    //       - Subexpressions: 6*ADD + 3*DIV + 4*MUL + 8*POW + 7*SUB
    const real_t x0 = -px0 + px1;
    const real_t x1 = -px0 + px2;
    const real_t x2 = -py0 + py1;
    const real_t x3 = -py0 + py2;
    const real_t x4 = -pz0 + pz1;
    const real_t x5 = -pz0 + pz2;
    const real_t x6 = sqrt((pow(x0, 2) + pow(x2, 2) + pow(x4, 2)) * (pow(x1, 2) + pow(x3, 2) + pow(x5, 2)) -
                           pow(x0 * x1 + x2 * x3 + x4 * x5, 2));
    const real_t x7 = (1.0 / 12.0) * x6;
    const real_t x8 = (1.0 / 24.0) * x6;
    element_matrix[0] = x7;
    element_matrix[1] = x8;
    element_matrix[2] = x8;
    element_matrix[3] = x8;
    element_matrix[4] = x7;
    element_matrix[5] = x8;
    element_matrix[6] = x8;
    element_matrix[7] = x8;
    element_matrix[8] = x7;
}

static SFEM_INLINE void lumped_boundary_mass(const real_t px0,
                                             const real_t px1,
                                             const real_t px2,
                                             const real_t py0,
                                             const real_t py1,
                                             const real_t py2,
                                             const real_t pz0,
                                             const real_t pz1,
                                             const real_t pz2,
                                             real_t *element_vector) {
    // FLOATING POINT OPS!
    //       - Result: 3*ASSIGNMENT
    //       - Subexpressions: 6*ADD + 2*DIV + 4*MUL + 8*POW + 7*SUB
    const real_t x0 = -px0 + px1;
    const real_t x1 = -px0 + px2;
    const real_t x2 = -py0 + py1;
    const real_t x3 = -py0 + py2;
    const real_t x4 = -pz0 + pz1;
    const real_t x5 = -pz0 + pz2;
    const real_t x6 =
        (1.0 / 6.0) * sqrt((pow(x0, 2) + pow(x2, 2) + pow(x4, 2)) * (pow(x1, 2) + pow(x3, 2) + pow(x5, 2)) -
                           pow(x0 * x1 + x2 * x3 + x4 * x5, 2));
    element_vector[0] = x6;
    element_vector[1] = x6;
    element_vector[2] = x6;
}


static SFEM_INLINE idx_t linear_search(const idx_t target, const idx_t *const arr, const int size) {
    idx_t i;
    for (i = 0; i < size - 4; i += 4) {
        if (arr[i] == target) return i;
        if (arr[i + 1] == target) return i + 1;
        if (arr[i + 2] == target) return i + 2;
        if (arr[i + 3] == target) return i + 3;
    }
    for (; i < size; i++) {
        if (arr[i] == target) return i;
    }
    return SFEM_IDX_INVALID;
}

static SFEM_INLINE idx_t find_col(const idx_t key, const idx_t *const row, const int lenrow) {
    if (lenrow <= 32) {
        return linear_search(key, row, lenrow);
    } else {
        return find_idx_binary_search(key, row, lenrow);
    }
}

static SFEM_INLINE void find_cols3(const idx_t *targets, const idx_t *const row, const int lenrow, idx_t *ks) {
    if (lenrow > 32) {
        for (int d = 0; d < 3; ++d) {
            ks[d] = find_col(targets[d], row, lenrow);
        }
    } else {
#pragma unroll(3)
        for (int d = 0; d < 3; ++d) {
            ks[d] = 0;
        }

        for (int i = 0; i < lenrow; ++i) {
#pragma unroll(3)
            for (int d = 0; d < 3; ++d) {
                ks[d] += row[i] < targets[d];
            }
        }
    }
}

void assemble_boundary_mass(const ptrdiff_t nelements,
                            const ptrdiff_t nnodes,
                            idx_t *const elems[3],
                            geom_t *const xyz[3],
                            count_t *const rowptr,
                            idx_t *const colidx,
                            real_t *const values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[3];
    idx_t ks[3];

    real_t element_matrix[3 * 3];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        boundary_mass(
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
            element_matrix);

        for (int edof_i = 0; edof_i < 3; ++edof_i) {
            const idx_t dof_i = elems[edof_i][i];
            const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

            const idx_t *row = &colidx[rowptr[dof_i]];

            find_cols3(ev, row, lenrow, ks);

            real_t *rowvalues = &values[rowptr[dof_i]];
            const real_t *element_row = &element_matrix[edof_i * 3];

#pragma unroll(3)
            for (int edof_j = 0; edof_j < 3; ++edof_j) {
                rowvalues[ks[edof_j]] += element_row[edof_j];
            }
        }
    }

    double tock = MPI_Wtime();
    printf("boundary_mass.c: assemble_boundary_mass\t%g seconds\n", tock - tick);
}

void assemble_lumped_boundary_mass(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t *const elems[3],
                          geom_t *const xyz[3],
                          real_t *const values)
{
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[3];
    idx_t ks[3];

    real_t element_vector[3];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        lumped_boundary_mass(
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
            element_vector);

        for (int edof_i = 0; edof_i < 3; ++edof_i) {
            values[ev[edof_i]] += element_vector[edof_i];
        }
    }

    double tock = MPI_Wtime();
    printf("boundary_mass.c: assemble_lumped_boundary_mass\t%g seconds\n", tock - tick);
}
