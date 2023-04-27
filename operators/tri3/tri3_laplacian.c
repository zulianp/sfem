#include "tri3_laplacian.h"

#include "sfem_vec.h"
#include "sortreduce.h"

#include <stdio.h>
#include <mpi.h>

#define POW2(a) ((a) * (a))

static SFEM_INLINE void tri3_laplacian_assemble_hessian_kernel(const real_t px0,
                                                               const real_t px1,
                                                               const real_t px2,
                                                               const real_t py0,
                                                               const real_t py1,
                                                               const real_t py2,
                                                               real_t *element_matrix) {
    static const int stride = 1;
    real_t fff[3];
    {
        // FLOATING POINT OPS!
        //       - Result: 3*ADD + 3*ASSIGNMENT + 9*MUL + 4*POW
        //       - Subexpressions: 2*MUL + NEG + POW + 5*SUB
        const real_t x0 = -px0 + px1;
        const real_t x1 = -py0 + py2;
        const real_t x2 = px0 - px2;
        const real_t x3 = py0 - py1;
        const real_t x4 = x0 * x1 - x2 * x3;
        const real_t x5 = 1. / POW2(x4);
        fff[0 * stride] = x4 * (POW2(x1) * x5 + POW2(x2) * x5);
        fff[1 * stride] = x4 * (x0 * x2 * x5 + x1 * x3 * x5);
        fff[2 * stride] = x4 * (POW2(x0) * x5 + POW2(x3) * x5);
    }

    // FLOATING POINT OPS!
    //       - Result: ADD + 9*ASSIGNMENT
    //       - Subexpressions: 3*DIV + 2*NEG + 2*SUB
    const real_t x0 = (1.0 / 2.0) * fff[0 * stride];
    const real_t x1 = (1.0 / 2.0) * fff[2 * stride];
    const real_t x2 = (1.0 / 2.0) * fff[1 * stride];
    const real_t x3 = -x0 - x2;
    const real_t x4 = -x1 - x2;
    element_matrix[0 * stride] = fff[1 * stride] + x0 + x1;
    element_matrix[1 * stride] = x3;
    element_matrix[2 * stride] = x4;
    element_matrix[3 * stride] = x3;
    element_matrix[4 * stride] = x0;
    element_matrix[5 * stride] = x2;
    element_matrix[6 * stride] = x4;
    element_matrix[7 * stride] = x2;
    element_matrix[8 * stride] = x1;
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

static SFEM_INLINE void find_cols3(const idx_t *targets, const idx_t *const row, const int lenrow, int *ks) {
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

void tri3_laplacian_assemble_hessian(const ptrdiff_t nelements,
                                     const ptrdiff_t nnodes,
                                     idx_t **const SFEM_RESTRICT elems,
                                     geom_t **const SFEM_RESTRICT xyz,
                                     const count_t *const SFEM_RESTRICT rowptr,
                                     const idx_t *const SFEM_RESTRICT colidx,
                                     real_t *const SFEM_RESTRICT values) {
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

        tri3_laplacian_assemble_hessian_kernel(
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
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
    printf("tet4_laplacian.c: tet4_laplacian_assemble_hessian\t%g seconds\n", tock - tick);
}
