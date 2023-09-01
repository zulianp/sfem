#include "tri6_mass.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_base.h"
#include "sfem_vec.h"

static SFEM_INLINE int linear_search(const idx_t target, const idx_t *const arr, const int size) {
    int i;
    for (i = 0; i < size - SFEM_VECTOR_SIZE; i += SFEM_VECTOR_SIZE) {
        if (arr[i] == target) return i;
        if (arr[i + 1] == target) return i + 1;
        if (arr[i + 2] == target) return i + 2;
        if (arr[i + 3] == target) return i + 3;
    }
    for (; i < size; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

static SFEM_INLINE int find_col(const idx_t key, const idx_t *const row, const int lenrow) {
    if (lenrow <= 32) {
        return linear_search(key, row, lenrow);

        // Using sentinel (potentially dangerous if matrix is buggy and column does not exist)
        // while (key > row[++k]) {
        //     // Hi
        // }
        // assert(k < lenrow);
        // assert(key == row[k]);
    } else {
        // Use this for larger number of dofs per row
        return find_idx_binary_search(key, row, lenrow);
    }
}

static SFEM_INLINE void find_cols6(const idx_t *targets,
                                   const idx_t *const row,
                                   const int lenrow,
                                   int *ks) {
    if (lenrow > 32) {
        for (int d = 0; d < 6; ++d) {
            ks[d] = find_col(targets[d], row, lenrow);
        }
    } else {
#pragma unroll(6)
        for (int d = 0; d < 6; ++d) {
            ks[d] = 0;
        }

        for (int i = 0; i < lenrow; ++i) {
#pragma unroll(6)
            for (int d = 0; d < 6; ++d) {
                ks[d] += row[i] < targets[d];
            }
        }
    }
}

static SFEM_INLINE void tri6_mass_kernel(const real_t px0,
                                         const real_t px1,
                                         const real_t px2,
                                         const real_t py0,
                                         const real_t py1,
                                         const real_t py2,
                                         real_t *const SFEM_RESTRICT element_matrix) {
    const real_t x0 = (px0 - px1) * (py0 - py2);
    const real_t x1 = px0 - px2;
    const real_t x2 = py0 - py1;
    const real_t x3 = x1 * x2;
    const real_t x4 = (1.0 / 60.0) * x0 - 1.0 / 60.0 * x3;
    const real_t x5 = -1.0 / 360.0 * x0 + (1.0 / 360.0) * x1 * x2;
    const real_t x6 = -1.0 / 90.0 * x0 + (1.0 / 90.0) * x1 * x2;
    const real_t x7 = (4.0 / 45.0) * x0 - 4.0 / 45.0 * x3;
    const real_t x8 = (2.0 / 45.0) * x0 - 2.0 / 45.0 * x3;
    element_matrix[0] = x4;
    element_matrix[1] = x5;
    element_matrix[2] = x5;
    element_matrix[3] = 0;
    element_matrix[4] = x6;
    element_matrix[5] = 0;
    element_matrix[6] = x5;
    element_matrix[7] = x4;
    element_matrix[8] = x5;
    element_matrix[9] = 0;
    element_matrix[10] = 0;
    element_matrix[11] = x6;
    element_matrix[12] = x5;
    element_matrix[13] = x5;
    element_matrix[14] = x4;
    element_matrix[15] = x6;
    element_matrix[16] = 0;
    element_matrix[17] = 0;
    element_matrix[18] = 0;
    element_matrix[19] = 0;
    element_matrix[20] = x6;
    element_matrix[21] = x7;
    element_matrix[22] = x8;
    element_matrix[23] = x8;
    element_matrix[24] = x6;
    element_matrix[25] = 0;
    element_matrix[26] = 0;
    element_matrix[27] = x8;
    element_matrix[28] = x7;
    element_matrix[29] = x8;
    element_matrix[30] = 0;
    element_matrix[31] = x6;
    element_matrix[32] = 0;
    element_matrix[33] = x8;
    element_matrix[34] = x8;
    element_matrix[35] = x7;
}

static SFEM_INLINE void lumped_mass_kernel(const real_t px0,
                                           const real_t px1,
                                           const real_t px2,
                                           const real_t py0,
                                           const real_t py1,
                                           const real_t py2,
                                           real_t *const SFEM_RESTRICT element_matrix_diag) {
    const real_t x0 = (px0 - px1) * (py0 - py2);
    const real_t x1 = (px0 - px2) * (py0 - py1);
    const real_t x2 = (1.0 / 15.0) * x0 - 1.0 / 15.0 * x1;
    const real_t x3 = (1.0 / 10.0) * x0 - 1.0 / 10.0 * x1;
    element_matrix_diag[0] = x2;
    element_matrix_diag[1] = x2;
    element_matrix_diag[2] = x2;
    element_matrix_diag[3] = x3;
    element_matrix_diag[4] = x3;
    element_matrix_diag[5] = x3;
}

static SFEM_INLINE void tri6_transform_kernel(const real_t *const SFEM_RESTRICT x,
                                              real_t *const SFEM_RESTRICT values) {
    const real_t x0 = (1.0 / 5.0) * x[0];
    const real_t x1 = (1.0 / 5.0) * x[1];
    const real_t x2 = (1.0 / 5.0) * x[2];
    values[0] = x[0];
    values[1] = x[1];
    values[2] = x[2];
    values[3] = x0 + x1 + (3.0 / 5.0) * x[3];
    values[4] = x1 + x2 + (3.0 / 5.0) * x[4];
    values[5] = x0 + x2 + (3.0 / 5.0) * x[5];
}

void tri6_apply_inv_lumped_mass(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                const real_t *const x,
                                real_t *const values) {
    double tick = MPI_Wtime();

    idx_t ev[6];
    real_t element_x[6];
    real_t element_x_pre_trafo[6];
    real_t element_weights[6];

    // Apply diagonal
    {
        real_t *weights = (real_t *)malloc(nnodes * sizeof(real_t));
        memset(weights, 0, nnodes * sizeof(real_t));

        for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(6)
            for (int v = 0; v < 6; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];

            lumped_mass_kernel(
                // X-coordinates
                xyz[0][i0],
                xyz[0][i1],
                xyz[0][i2],
                // Y-coordinates
                xyz[1][i0],
                xyz[1][i1],
                xyz[1][i2],
                element_weights);

            for (int v = 0; v < 6; ++v) {
                const idx_t idx = ev[v];
                weights[idx] += element_weights[v];
            }
        }

        for (ptrdiff_t i = 0; i < nnodes; i++) {
            values[i] = x[i] / weights[i];
        }

        free(weights);
    }

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(6)
        for (int v = 0; v < 6; ++v) {
            element_x_pre_trafo[v] = values[elems[v][i]];
        }

        tri6_transform_kernel(element_x_pre_trafo, element_x);

        for (int v = 0; v < 6; ++v) {
            const idx_t idx = ev[v];
            values[idx] = element_x[v];
        }
    }

    double tock = MPI_Wtime();
    printf("tri6_mass.c: tri6_apply_inv_lumped_mass\t%g seconds\n", tock - tick);
}

static SFEM_INLINE void tri6_hrz_lumped_mass_kernel(const real_t px0,
                                                    const real_t px1,
                                                    const real_t px2,
                                                    const real_t py0,
                                                    const real_t py1,
                                                    const real_t py2,
                                                    real_t *const SFEM_RESTRICT
                                                        element_matrix_diag) {
    const real_t x0 = px0 - px1;
    const real_t x1 = py0 - py2;
    const real_t x2 = x0 * x1;
    const real_t x3 = px0 - px2;
    const real_t x4 = py0 - py1;
    const real_t x5 = x3 * x4;
    const real_t x6 =
        ((1.0 / 2.0) * x0 * x1 - 1.0 / 2.0 * x3 * x4) / ((19.0 / 60.0) * x2 - 19.0 / 60.0 * x5);
    const real_t x7 = x6 * ((1.0 / 60.0) * x2 - 1.0 / 60.0 * x5);
    const real_t x8 = x6 * ((4.0 / 45.0) * x2 - 4.0 / 45.0 * x5);
    element_matrix_diag[0] = x7;
    element_matrix_diag[1] = x7;
    element_matrix_diag[2] = x7;
    element_matrix_diag[3] = x8;
    element_matrix_diag[4] = x8;
    element_matrix_diag[5] = x8;
}

void tri6_assemble_lumped_mass(const ptrdiff_t nelements,
                               const ptrdiff_t nnodes,
                               idx_t **const SFEM_RESTRICT elems,
                               geom_t **const SFEM_RESTRICT xyz,
                               real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[6];
            idx_t ks[6];

            real_t element_vector[6];
#pragma unroll(6)
            for (int v = 0; v < 6; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];

            tri6_hrz_lumped_mass_kernel(
                // X-coordinates
                xyz[0][i0],
                xyz[0][i1],
                xyz[0][i2],
                // Y-coordinates
                xyz[1][i0],
                xyz[1][i1],
                xyz[1][i2],
                element_vector);

#pragma unroll(6)
            for (int edof_i = 0; edof_i < 6; ++edof_i) {
#pragma omp atomic update
                values[ev[edof_i]] += element_vector[edof_i];
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tri6_mass.c: tri6_assemble_lumped_mass\t%g seconds\n", tock - tick);
}

void tri6_assemble_mass(const ptrdiff_t nelements,
                        const ptrdiff_t nnodes,
                        idx_t **const SFEM_RESTRICT elems,
                        geom_t **const SFEM_RESTRICT xyz,
                        count_t *const SFEM_RESTRICT rowptr,
                        idx_t *const SFEM_RESTRICT colidx,
                        real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[6];
            idx_t ks[6];

            real_t element_matrix[6 * 6];
#pragma unroll(6)
            for (int v = 0; v < 6; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];

            tri6_mass_kernel(
                // X-coordinates
                xyz[0][i0],
                xyz[0][i1],
                xyz[0][i2],
                // Y-coordinates
                xyz[1][i0],
                xyz[1][i1],
                xyz[1][i2],
                element_matrix);

            for (int edof_i = 0; edof_i < 6; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];
                const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

                const idx_t *row = &colidx[rowptr[dof_i]];

                find_cols6(ev, row, lenrow, ks);

                real_t *rowvalues = &values[rowptr[dof_i]];
                const real_t *element_row = &element_matrix[edof_i * 6];

#pragma unroll(6)
                for (int edof_j = 0; edof_j < 6; ++edof_j) {
#pragma omp atomic update
                    rowvalues[ks[edof_j]] += element_row[edof_j];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tri6_mass.c: assemble_mass\t%g seconds\n", tock - tick);
}
