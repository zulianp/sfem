#include "cvfem_quad4_laplacian.h"

#include "sfem_base.h"
#include "sfem_vec.h"
#include "sortreduce.h"

#include <assert.h>
#include <mpi.h>
#include <stdio.h>

#define POW2(a) ((a) * (a))

static SFEM_INLINE void cvfem_quad4_laplacian_assemble_hessian_kernel(
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t px3,
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t py3,
    real_t *const SFEM_RESTRICT element_matrix) {
    const real_t x0 = 0.0625 * px0 - 0.0625 * px2 + 0.0625 * py0 - 0.0625 * py2;
    const real_t x1 = 0.3125 * py1 - 0.3125 * py3 + x0;
    const real_t x2 = -0.3125 * px1 + 0.3125 * px3;
    const real_t x3 = -x1 - x2;
    const real_t x4 = 0.1875 * px1 - 0.1875 * px3;
    const real_t x5 = x1 + x4;
    const real_t x6 = -0.1875 * py1 + 0.1875 * py3 + x0;
    const real_t x7 = -x4 - x6;
    const real_t x8 = x2 + x6;
    const real_t x9 = 0.3125 * py2;
    const real_t x10 = 0.3125 * py0;
    const real_t x11 = 0.0625 * px1;
    const real_t x12 = 0.0625 * py3;
    const real_t x13 = 0.0625 * px3;
    const real_t x14 = 0.0625 * py1;
    const real_t x15 = 0.1875 * px0 - 0.1875 * px2 + x11 + x12 - x13 - x14;
    const real_t x16 = x10 - x15 - x9;
    const real_t x17 = 0.3125 * px0 - 0.3125 * px2 - x11 - x12 + x13 + x14;
    const real_t x18 = -x10 - x17 + x9;
    const real_t x19 = 0.1875 * py2;
    const real_t x20 = 0.1875 * py0;
    const real_t x21 = x17 + x19 - x20;
    const real_t x22 = x15 - x19 + x20;
    element_matrix[0] = x3;
    element_matrix[1] = x5;
    element_matrix[2] = x7;
    element_matrix[3] = x8;
    element_matrix[4] = x16;
    element_matrix[5] = x18;
    element_matrix[6] = x21;
    element_matrix[7] = x22;
    element_matrix[8] = x7;
    element_matrix[9] = x8;
    element_matrix[10] = x3;
    element_matrix[11] = x5;
    element_matrix[12] = x21;
    element_matrix[13] = x22;
    element_matrix[14] = x16;
    element_matrix[15] = x18;
}

static SFEM_INLINE void cvfem_quad4_laplacian_assemble_apply_kernel(
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t px3,
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t py3,
    const real_t *const SFEM_RESTRICT x,
    real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = 0.0625 * px0 - 0.0625 * px2 + 0.0625 * py0 - 0.0625 * py2;
    const real_t x1 = 0.3125 * py1 - 0.3125 * py3 + x0;
    const real_t x2 = -0.3125 * px1 + 0.3125 * px3;
    const real_t x3 = -x1 - x2;
    const real_t x4 = 0.1875 * px1 - 0.1875 * px3;
    const real_t x5 = x1 + x4;
    const real_t x6 = -0.1875 * py1 + 0.1875 * py3 + x0;
    const real_t x7 = -x4 - x6;
    const real_t x8 = x2 + x6;
    const real_t x9 = 0.3125 * py2;
    const real_t x10 = 0.3125 * py0;
    const real_t x11 = 0.0625 * px1;
    const real_t x12 = 0.0625 * py3;
    const real_t x13 = 0.0625 * px3;
    const real_t x14 = 0.0625 * py1;
    const real_t x15 = 0.1875 * px0 - 0.1875 * px2 + x11 + x12 - x13 - x14;
    const real_t x16 = x10 - x15 - x9;
    const real_t x17 = 0.3125 * px0 - 0.3125 * px2 - x11 - x12 + x13 + x14;
    const real_t x18 = -x10 - x17 + x9;
    const real_t x19 = 0.1875 * py2;
    const real_t x20 = 0.1875 * py0;
    const real_t x21 = x17 + x19 - x20;
    const real_t x22 = x15 - x19 + x20;
    element_vector[0] = x3 * x[0] + x5 * x[1] + x7 * x[2] + x8 * x[3];
    element_vector[1] = x16 * x[0] + x18 * x[1] + x21 * x[2] + x22 * x[3];
    element_vector[2] = x3 * x[2] + x5 * x[3] + x7 * x[0] + x8 * x[1];
    element_vector[3] = x16 * x[2] + x18 * x[3] + x21 * x[0] + x22 * x[1];
}

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

static SFEM_INLINE void find_cols4(const idx_t *targets,
                                   const idx_t *const row,
                                   const int lenrow,
                                   int *ks) {
    if (lenrow > 32) {
        for (int d = 0; d < 4; ++d) {
            ks[d] = find_col(targets[d], row, lenrow);
        }
    } else {
#pragma unroll(4)
        for (int d = 0; d < 4; ++d) {
            ks[d] = 0;
        }

        for (int i = 0; i < lenrow; ++i) {
#pragma unroll(4)
            for (int d = 0; d < 4; ++d) {
                ks[d] += row[i] < targets[d];
            }
        }
    }
}

void cvfem_quad4_laplacian_assemble_hessian(const ptrdiff_t nelements,
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
#pragma omp for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[4];
            idx_t ks[4];
            real_t element_matrix[4 * 4];
            

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            cvfem_quad4_laplacian_assemble_hessian_kernel(
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
                element_matrix);

            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];
                const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

                const idx_t *row = &colidx[rowptr[dof_i]];

                find_cols4(ev, row, lenrow, ks);

                real_t *rowvalues = &values[rowptr[dof_i]];
                const real_t *element_row = &element_matrix[edof_i * 4];

#pragma unroll(4)
                for (int edof_j = 0; edof_j < 4; ++edof_j) {
#pragma omp atomic update
                    rowvalues[ks[edof_j]] += element_row[edof_j];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("cvfem_quad4_laplacian.c: cvfem_quad4_laplacian_assemble_hessian\t%g seconds\n",
           tock - tick);
}

void cvfem_quad4_laplacian_apply(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[4];

            real_t element_u[4];
            real_t element_vector[4];

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                element_u[v] = u[ev[v]];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            cvfem_quad4_laplacian_assemble_apply_kernel(
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
                element_u,
                element_vector);

            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
                values[dof_i] += element_vector[edof_i];
            }
        }
    }
}
