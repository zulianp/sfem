#include "sfem_base.h"
#include "sfem_vec.h"

#include "sortreduce.h"

#include <mpi.h>
#include <stdio.h>

static SFEM_INLINE void cvfem_tri3_diffusion_hessian_kernel(const real_t px0,
                                                            const real_t px1,
                                                            const real_t px2,
                                                            const real_t py0,
                                                            const real_t py1,
                                                            const real_t py2,
                                                            real_t *element_matrix) {
    static const int stride = 1;
    // FLOATING POINT OPS!
    //       - Result: 9*ADD + 9*ASSIGNMENT + 36*MUL
    //       - Subexpressions: 9*ADD + 13*DIV + 8*MUL + 11*NEG + 10*SUB
    const real_t x0 = (1.0 / 6.0) * px0;
    const real_t x1 = (1.0 / 6.0) * px1;
    const real_t x2 = -1.0 / 3.0 * px2 + x0 + x1;
    const real_t x3 = -x2;
    const real_t x4 = px0 - px1;
    const real_t x5 = -py2;
    const real_t x6 = py0 + x5;
    const real_t x7 = -px2;
    const real_t x8 = px0 + x7;
    const real_t x9 = py0 - py1;
    const real_t x10 = 1.0 / (x4 * x6 - x8 * x9);
    const real_t x11 = x10 * (-px1 - x7);
    const real_t x12 = (1.0 / 6.0) * px2;
    const real_t x13 = -1.0 / 3.0 * px1 + x0 + x12;
    const real_t x14 = (1.0 / 6.0) * py0;
    const real_t x15 = (1.0 / 6.0) * py2;
    const real_t x16 = -1.0 / 3.0 * py1 + x14 + x15;
    const real_t x17 = -x16;
    const real_t x18 = x10 * (py1 + x5);
    const real_t x19 = (1.0 / 6.0) * py1;
    const real_t x20 = -1.0 / 3.0 * py2 + x14 + x19;
    const real_t x21 = x10 * x8;
    const real_t x22 = -x10 * x6;
    const real_t x23 = -x10 * x4;
    const real_t x24 = x10 * x9;
    const real_t x25 = -1.0 / 3.0 * px0 + x1 + x12;
    const real_t x26 = -x25;
    const real_t x27 = -1.0 / 3.0 * py0 + x15 + x19;
    const real_t x28 = -x20;
    const real_t x29 = -x13;
    const real_t x30 = -x27;
    element_matrix[0 * stride] = x11 * x13 + x11 * x3 + x17 * x18 + x18 * x20;
    element_matrix[1 * stride] = x13 * x21 + x17 * x22 + x20 * x22 + x21 * x3;
    element_matrix[2 * stride] = x13 * x23 + x17 * x24 + x20 * x24 + x23 * x3;
    element_matrix[3 * stride] = x11 * x2 + x11 * x26 + x18 * x27 + x18 * x28;
    element_matrix[4 * stride] = x2 * x21 + x21 * x26 + x22 * x27 + x22 * x28;
    element_matrix[5 * stride] = x2 * x23 + x23 * x26 + x24 * x27 + x24 * x28;
    element_matrix[6 * stride] = x11 * x25 + x11 * x29 + x16 * x18 + x18 * x30;
    element_matrix[7 * stride] = x16 * x22 + x21 * x25 + x21 * x29 + x22 * x30;
    element_matrix[8 * stride] = x16 * x24 + x23 * x25 + x23 * x29 + x24 * x30;
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

static SFEM_INLINE void find_cols3(const idx_t *targets,
                                   const idx_t *const row,
                                   const int lenrow,
                                   idx_t *ks) {
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

void cvfem_tri3_diffusion_assemble_hessian(const ptrdiff_t nelements,
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
#pragma omp for //nowait

        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[3];
            idx_t ks[3];

            real_t element_matrix[3 * 3];

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];

            cvfem_tri3_diffusion_hessian_kernel(
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
#pragma omp atomic update
                    rowvalues[ks[edof_j]] += element_row[edof_j];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("cvfem_tri3_diffusion.c: cvfem_tri3_diffusion_assemble_hessian\t%g seconds\n",
           tock - tick);
}
