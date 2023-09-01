#include "tri6_laplacian.h"

#include "sfem_vec.h"
#include "sortreduce.h"

#include <mpi.h>
#include <stdio.h>


#define POW2(a) ((a) * (a))

static SFEM_INLINE void tri6_laplacian_assemble_hessian_kernel(const real_t px0,
                                                               const real_t px1,
                                                               const real_t px2,
                                                               const real_t py0,
                                                               const real_t py1,
                                                               const real_t py2,
                                                               real_t *element_matrix) {
    static const int stride = 1;
    real_t fff[3];

    {
        const real_t x0 = -px0 + px1;
        const real_t x1 = -py0 + py2;
        const real_t x2 = px0 - px2;
        const real_t x3 = py0 - py1;
        const real_t x4 = x0 * x1 - x2 * x3;
        const real_t x5 = 1./POW2(x4);
        fff[0 * stride] = x4 * (POW2(x1) * x5 + POW2(x2) * x5);
        fff[1 * stride] = x4 * (x0 * x2 * x5 + x1 * x3 * x5);
        fff[2 * stride] = x4 * (POW2(x0) * x5 + POW2(x3) * x5);
    }

    const real_t x0 = (1.0 / 2.0) * fff[0 * stride];
    const real_t x1 = (1.0 / 2.0) * fff[2 * stride];
    const real_t x2 = (1.0 / 6.0) * fff[1 * stride];
    const real_t x3 = (1.0 / 6.0) * fff[0 * stride] + x2;
    const real_t x4 = (1.0 / 6.0) * fff[2 * stride] + x2;
    const real_t x5 = (2.0 / 3.0) * fff[1 * stride];
    const real_t x6 = -2.0 / 3.0 * fff[0 * stride] - x5;
    const real_t x7 = -2.0 / 3.0 * fff[2 * stride] - x5;
    const real_t x8 = -x2;
    const real_t x9 = (4.0 / 3.0) * fff[0 * stride];
    const real_t x10 = (4.0 / 3.0) * fff[1 * stride];
    const real_t x11 = (4.0 / 3.0) * fff[2 * stride] + x10;
    const real_t x12 = x11 + x9;
    const real_t x13 = -x11;
    const real_t x14 = -x10 - x9;
    element_matrix[0 * stride] = fff[1 * stride] + x0 + x1;
    element_matrix[1 * stride] = x3;
    element_matrix[2 * stride] = x4;
    element_matrix[3 * stride] = x6;
    element_matrix[4 * stride] = 0;
    element_matrix[5 * stride] = x7;
    element_matrix[6 * stride] = x3;
    element_matrix[7 * stride] = x0;
    element_matrix[8 * stride] = x8;
    element_matrix[9 * stride] = x6;
    element_matrix[10 * stride] = x5;
    element_matrix[11 * stride] = 0;
    element_matrix[12 * stride] = x4;
    element_matrix[13 * stride] = x8;
    element_matrix[14 * stride] = x1;
    element_matrix[15 * stride] = 0;
    element_matrix[16 * stride] = x5;
    element_matrix[17 * stride] = x7;
    element_matrix[18 * stride] = x6;
    element_matrix[19 * stride] = x6;
    element_matrix[20 * stride] = 0;
    element_matrix[21 * stride] = x12;
    element_matrix[22 * stride] = x13;
    element_matrix[23 * stride] = x10;
    element_matrix[24 * stride] = 0;
    element_matrix[25 * stride] = x5;
    element_matrix[26 * stride] = x5;
    element_matrix[27 * stride] = x13;
    element_matrix[28 * stride] = x12;
    element_matrix[29 * stride] = x14;
    element_matrix[30 * stride] = x7;
    element_matrix[31 * stride] = 0;
    element_matrix[32 * stride] = x7;
    element_matrix[33 * stride] = x10;
    element_matrix[34 * stride] = x14;
    element_matrix[35 * stride] = x12;
}

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

void tri6_laplacian_assemble_hessian(const ptrdiff_t nelements,
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

            tri6_laplacian_assemble_hessian_kernel(
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
                const idx_t dof_i = ev[edof_i];
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
    printf("tri6_laplacian.c: tri6_laplacian_assemble_hessian\t%g seconds\n", tock - tick);
}
