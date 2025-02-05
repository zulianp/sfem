#include "beam2_mass.h"

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"

#define POW2(x) ((x) * (x))

static SFEM_INLINE int linear_search(const idx_t target, const idx_t *const arr, const int size) {
    int i;
    for (i = 0; i < size - 4; i += 4) {
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

static SFEM_INLINE void find_cols2(const idx_t *targets,
                                   const idx_t *const row,
                                   const int lenrow,
                                   idx_t *ks) {
    if (lenrow > 32) {
        for (int d = 0; d < 2; ++d) {
            ks[d] = find_col(targets[d], row, lenrow);
        }
    } else {
#pragma unroll(2)
        for (int d = 0; d < 2; ++d) {
            ks[d] = 0;
        }

        for (int i = 0; i < lenrow; ++i) {
#pragma unroll(2)
            for (int d = 0; d < 2; ++d) {
                ks[d] += row[i] < targets[d];
            }
        }
    }
}

static SFEM_INLINE void beam2_mass_kernel(const real_t px0,
                                          const real_t px1,
                                          const real_t py0,
                                          const real_t py1,
                                          const real_t pz0,
                                          const real_t pz1,
                                          real_t *const SFEM_RESTRICT element_matrix) {
    const real_t x0 = sqrt(pow(px0 - px1, 2) + pow(py0 - py1, 2) + pow(pz0 - pz1, 2));
    const real_t x1 = (1.0 / 3.0) * x0;
    const real_t x2 = (1.0 / 6.0) * x0;
    element_matrix[0] = x1;
    element_matrix[1] = x2;
    element_matrix[2] = x2;
    element_matrix[3] = x1;
}

static SFEM_INLINE void beam2_apply_mass_kernel(const real_t px0,
                                                const real_t px1,
                                                const real_t py0,
                                                const real_t py1,
                                                const real_t pz0,
                                                const real_t pz1,
                                                const real_t *const SFEM_RESTRICT u,
                                                real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = (1.0 / 6.0) * sqrt(pow(px0 - px1, 2) + pow(py0 - py1, 2) + pow(pz0 - pz1, 2));
    element_vector[0] = x0 * (2 * u[0] + u[1]);
    element_vector[1] = x0 * (u[0] + 2 * u[1]);
}

static SFEM_INLINE void lumped_mass(const real_t px0,
                                    const real_t px1,
                                    const real_t py0,
                                    const real_t py1,
                                    const real_t pz0,
                                    const real_t pz1,
                                    real_t *element_matrix_diag) {
    const real_t x0 = (1.0 / 2.0) * sqrt(pow(px0 - px1, 2) + pow(py0 - py1, 2) + pow(pz0 - pz1, 2));
    element_matrix_diag[0] = x0;
    element_matrix_diag[1] = x0;
}

void beam2_apply_mass(const ptrdiff_t nelements,
                      const ptrdiff_t nnodes,
                      idx_t **const SFEM_RESTRICT elems,
                      geom_t **const SFEM_RESTRICT xyz,
                      const ptrdiff_t stride_x,
                      const real_t *const x,
                      const ptrdiff_t stride_values,
                      real_t *const values) {
    SFEM_UNUSED(nnodes);
    double tick = MPI_Wtime();

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[2];
            idx_t ks[2];
            real_t element_x[2];
            real_t element_vector[2];

#pragma unroll(2)
            for (int v = 0; v < 2; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];

            for (int enode = 0; enode < 2; ++enode) {
                element_x[enode] = x[ev[enode] * stride_x];
            }

            beam2_apply_mass_kernel(
                    // X-coordinates
                    xyz[0][i0],
                    xyz[0][i1],
                    // Y-coordinates
                    xyz[1][i0],
                    xyz[1][i1],
                    // Z-coordinates
                    xyz[2][i0],
                    xyz[2][i1],
                    element_x,
                    // output vector
                    element_vector);

#pragma unroll(2)
            for (int edof_i = 0; edof_i < 2; edof_i++) {
#pragma omp atomic update
                values[ev[edof_i] * stride_values] += element_vector[edof_i];
            }
        }
    }

    double tock = MPI_Wtime();
    printf("beam2_mass.c: beam2_apply_mass\t%g seconds\n", tock - tick);
}

void beam2_assemble_mass(const ptrdiff_t nelements,
                         const ptrdiff_t nnodes,
                         idx_t **const SFEM_RESTRICT elems,
                         geom_t **const SFEM_RESTRICT xyz,
                         const count_t *const SFEM_RESTRICT rowptr,
                         const idx_t *const SFEM_RESTRICT colidx,
                         real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[2];
            idx_t ks[2];

            real_t element_matrix[2 * 2];
#pragma unroll(2)
            for (int v = 0; v < 2; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];

            beam2_mass_kernel(
                    // X-coordinates
                    xyz[0][i0],
                    xyz[0][i1],
                    // Y-coordinates
                    xyz[1][i0],
                    xyz[1][i1],
                    // Z-coordinates
                    xyz[2][i0],
                    xyz[2][i1],
                    element_matrix);

            for (int edof_i = 0; edof_i < 2; ++edof_i) {
                const idx_t dof_i = ev[edof_i];
                const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

                const idx_t *row = &colidx[rowptr[dof_i]];

                find_cols2(ev, row, lenrow, ks);

                real_t *rowvalues = &values[rowptr[dof_i]];
                const real_t *element_row = &element_matrix[edof_i * 2];

#pragma unroll(2)
                for (int edof_j = 0; edof_j < 2; ++edof_j) {
#pragma omp atomic update
                    rowvalues[ks[edof_j]] += element_row[edof_j];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("beam2_mass.c: assemble_mass\t%g seconds\n", tock - tick);
}

int beam2_assemble_lumped_mass(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    // double tick = MPI_Wtime();

    idx_t ev[2];
    idx_t ks[2];

    real_t element_vector[2];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(2)
        for (int v = 0; v < 2; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];

        lumped_mass(
                // X-coordinates
                xyz[0][i0],
                xyz[0][i1],
                // Y-coordinates
                xyz[1][i0],
                xyz[1][i1],
                // Z-coordinates
                xyz[2][i0],
                xyz[2][i1],
                element_vector);

        for (int edof_i = 0; edof_i < 2; ++edof_i) {
            values[ev[edof_i]] += element_vector[edof_i];
        }
    }

    return SFEM_SUCCESS;
}

void beam2_apply_inv_lumped_mass(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 const real_t *const x,
                                 real_t *const values) {
    real_t *buff = 0;
    if (x == values) {
        buff = (real_t *)calloc(nnodes, sizeof(real_t));
    } else {
        buff = values;
        memset(buff, 0, nnodes * sizeof(real_t));
    }

    beam2_assemble_lumped_mass(nelements, nnodes, elems, xyz, values);

    for (ptrdiff_t i = 0; i < nnodes; i++) {
        values[i] = x[i] / values[i];
    }

    if (x == values) {
        free(buff);
    }
}
