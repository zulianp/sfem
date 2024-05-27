#include "tet10_mass.h"

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

static SFEM_INLINE void tet10_mass_kernel(const real_t px0,
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
                                          real_t *const SFEM_RESTRICT element_matrix) {
    // FLOATING POINT OPS!
    //       - Result: 100*ASSIGNMENT
    //       - Subexpressions: 12*ADD + 33*DIV + 69*MUL + NEG + 27*SUB
    const real_t x0 = px0 - px1;
    const real_t x1 = py0 - py2;
    const real_t x2 = pz0 - pz3;
    const real_t x3 = x1 * x2;
    const real_t x4 = py0 - py3;
    const real_t x5 = pz0 - pz2;
    const real_t x6 = px0 - px2;
    const real_t x7 = py0 - py1;
    const real_t x8 = pz0 - pz1;
    const real_t x9 = x4 * x8;
    const real_t x10 = px0 - px3;
    const real_t x11 = x5 * x7;
    const real_t x12 = (1.0 / 420.0) * x0 * x3 - 1.0 / 420.0 * x0 * x4 * x5 -
                       1.0 / 420.0 * x1 * x10 * x8 + (1.0 / 420.0) * x10 * x11 -
                       1.0 / 420.0 * x2 * x6 * x7 + (1.0 / 420.0) * x6 * x9;
    const real_t x13 = -x12;
    const real_t x14 = -1.0 / 2520.0 * x0 * x3 + (1.0 / 2520.0) * x0 * x4 * x5 +
                       (1.0 / 2520.0) * x1 * x10 * x8 - 1.0 / 2520.0 * x10 * x11 +
                       (1.0 / 2520.0) * x2 * x6 * x7 - 1.0 / 2520.0 * x6 * x9;
    const real_t x15 = (1.0 / 630.0) * x0;
    const real_t x16 = (1.0 / 630.0) * x6;
    const real_t x17 = (1.0 / 630.0) * x10;
    const real_t x18 =
        -x1 * x17 * x8 + x11 * x17 + x15 * x3 - x15 * x4 * x5 - x16 * x2 * x7 + x16 * x9;
    const real_t x19 = -4.0 / 315.0 * x0 * x3 + (4.0 / 315.0) * x0 * x4 * x5 +
                       (4.0 / 315.0) * x1 * x10 * x8 - 4.0 / 315.0 * x10 * x11 +
                       (4.0 / 315.0) * x2 * x6 * x7 - 4.0 / 315.0 * x6 * x9;
    const real_t x20 = -2.0 / 315.0 * x0 * x3 + (2.0 / 315.0) * x0 * x4 * x5 +
                       (2.0 / 315.0) * x1 * x10 * x8 - 2.0 / 315.0 * x10 * x11 +
                       (2.0 / 315.0) * x2 * x6 * x7 - 2.0 / 315.0 * x6 * x9;
    const real_t x21 = -1.0 / 315.0 * x0 * x3 + (1.0 / 315.0) * x0 * x4 * x5 +
                       (1.0 / 315.0) * x1 * x10 * x8 - 1.0 / 315.0 * x10 * x11 +
                       (1.0 / 315.0) * x2 * x6 * x7 - 1.0 / 315.0 * x6 * x9;
    element_matrix[0] = x13;
    element_matrix[1] = x14;
    element_matrix[2] = x14;
    element_matrix[3] = x14;
    element_matrix[4] = x18;
    element_matrix[5] = x12;
    element_matrix[6] = x18;
    element_matrix[7] = x18;
    element_matrix[8] = x12;
    element_matrix[9] = x12;
    element_matrix[10] = x14;
    element_matrix[11] = x13;
    element_matrix[12] = x14;
    element_matrix[13] = x14;
    element_matrix[14] = x18;
    element_matrix[15] = x18;
    element_matrix[16] = x12;
    element_matrix[17] = x12;
    element_matrix[18] = x18;
    element_matrix[19] = x12;
    element_matrix[20] = x14;
    element_matrix[21] = x14;
    element_matrix[22] = x13;
    element_matrix[23] = x14;
    element_matrix[24] = x12;
    element_matrix[25] = x18;
    element_matrix[26] = x18;
    element_matrix[27] = x12;
    element_matrix[28] = x12;
    element_matrix[29] = x18;
    element_matrix[30] = x14;
    element_matrix[31] = x14;
    element_matrix[32] = x14;
    element_matrix[33] = x13;
    element_matrix[34] = x12;
    element_matrix[35] = x12;
    element_matrix[36] = x12;
    element_matrix[37] = x18;
    element_matrix[38] = x18;
    element_matrix[39] = x18;
    element_matrix[40] = x18;
    element_matrix[41] = x18;
    element_matrix[42] = x12;
    element_matrix[43] = x12;
    element_matrix[44] = x19;
    element_matrix[45] = x20;
    element_matrix[46] = x20;
    element_matrix[47] = x20;
    element_matrix[48] = x20;
    element_matrix[49] = x21;
    element_matrix[50] = x12;
    element_matrix[51] = x18;
    element_matrix[52] = x18;
    element_matrix[53] = x12;
    element_matrix[54] = x20;
    element_matrix[55] = x19;
    element_matrix[56] = x20;
    element_matrix[57] = x21;
    element_matrix[58] = x20;
    element_matrix[59] = x20;
    element_matrix[60] = x18;
    element_matrix[61] = x12;
    element_matrix[62] = x18;
    element_matrix[63] = x12;
    element_matrix[64] = x20;
    element_matrix[65] = x20;
    element_matrix[66] = x19;
    element_matrix[67] = x20;
    element_matrix[68] = x21;
    element_matrix[69] = x20;
    element_matrix[70] = x18;
    element_matrix[71] = x12;
    element_matrix[72] = x12;
    element_matrix[73] = x18;
    element_matrix[74] = x20;
    element_matrix[75] = x21;
    element_matrix[76] = x20;
    element_matrix[77] = x19;
    element_matrix[78] = x20;
    element_matrix[79] = x20;
    element_matrix[80] = x12;
    element_matrix[81] = x18;
    element_matrix[82] = x12;
    element_matrix[83] = x18;
    element_matrix[84] = x20;
    element_matrix[85] = x20;
    element_matrix[86] = x21;
    element_matrix[87] = x20;
    element_matrix[88] = x19;
    element_matrix[89] = x20;
    element_matrix[90] = x12;
    element_matrix[91] = x12;
    element_matrix[92] = x18;
    element_matrix[93] = x18;
    element_matrix[94] = x21;
    element_matrix[95] = x20;
    element_matrix[96] = x20;
    element_matrix[97] = x20;
    element_matrix[98] = x20;
    element_matrix[99] = x19;
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

static SFEM_INLINE void find_cols10(const idx_t *targets,
                                    const idx_t *const row,
                                    const int lenrow,
                                    int *ks) {
    if (lenrow > 32) {
        for (int d = 0; d < 10; ++d) {
            ks[d] = find_col(targets[d], row, lenrow);
        }
    } else {
#pragma unroll(10)
        for (int d = 0; d < 10; ++d) {
            ks[d] = 0;
        }

        for (int i = 0; i < lenrow; ++i) {
#pragma unroll(10)
            for (int d = 0; d < 10; ++d) {
                ks[d] += row[i] < targets[d];
            }
        }
    }
}

void tet10_assemble_mass(const ptrdiff_t nelements,
                         const ptrdiff_t nnodes,
                         idx_t **const SFEM_RESTRICT elems,
                         geom_t **const SFEM_RESTRICT xyz,
                         const count_t *const SFEM_RESTRICT rowptr,
                         const idx_t *const SFEM_RESTRICT colidx,
                         real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[10];
    idx_t ks[10];

    real_t element_matrix[10 * 10];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices for affine coordinates
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        tet10_mass_kernel(
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

        for (int edof_i = 0; edof_i < 10; ++edof_i) {
            const idx_t dof_i = elems[edof_i][i];
            const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

            const idx_t *row = &colidx[rowptr[dof_i]];

            find_cols10(ev, row, lenrow, ks);

            real_t *rowvalues = &values[rowptr[dof_i]];
            const real_t *element_row = &element_matrix[edof_i * 10];

#pragma unroll(10)
            for (int edof_j = 0; edof_j < 10; ++edof_j) {
                rowvalues[ks[edof_j]] += element_row[edof_j];
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tet10_mass.c: tet10_assemble_mass\t%g seconds\n", tock - tick);
}

static SFEM_INLINE void lumped_mass_kernel(const real_t px0,
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
                                           real_t *const SFEM_RESTRICT element_matrix_diag) {
    // generated code
    // FLOATING POINT OPS!
    //       - Result: 10*ASSIGNMENT
    //       - Subexpressions: 4*ADD + 12*DIV + 27*MUL + 15*SUB
    const real_t x0 = px0 - px1;
    const real_t x1 = py0 - py2;
    const real_t x2 = pz0 - pz3;
    const real_t x3 = x1 * x2;
    const real_t x4 = py0 - py3;
    const real_t x5 = pz0 - pz2;
    const real_t x6 = px0 - px2;
    const real_t x7 = py0 - py1;
    const real_t x8 = pz0 - pz1;
    const real_t x9 = x4 * x8;
    const real_t x10 = px0 - px3;
    const real_t x11 = x5 * x7;
    const real_t x12 = -7.0 / 600.0 * x0 * x3 + (7.0 / 600.0) * x0 * x4 * x5 +
                       (7.0 / 600.0) * x1 * x10 * x8 - 7.0 / 600.0 * x10 * x11 +
                       (7.0 / 600.0) * x2 * x6 * x7 - 7.0 / 600.0 * x6 * x9;
    const real_t x13 = -1.0 / 50.0 * x0 * x3 + (1.0 / 50.0) * x0 * x4 * x5 +
                       (1.0 / 50.0) * x1 * x10 * x8 - 1.0 / 50.0 * x10 * x11 +
                       (1.0 / 50.0) * x2 * x6 * x7 - 1.0 / 50.0 * x6 * x9;
    element_matrix_diag[0] = x12;
    element_matrix_diag[1] = x12;
    element_matrix_diag[2] = x12;
    element_matrix_diag[3] = x12;
    element_matrix_diag[4] = x13;
    element_matrix_diag[5] = x13;
    element_matrix_diag[6] = x13;
    element_matrix_diag[7] = x13;
    element_matrix_diag[8] = x13;
    element_matrix_diag[9] = x13;
}

static SFEM_INLINE void tet10_transform_kernel(const real_t *const SFEM_RESTRICT x,
                                               real_t *const SFEM_RESTRICT values) {
    // FLOATING POINT OPS!
    //       - Result: 4*ADD + 10*ASSIGNMENT + 6*MUL
    //       - Subexpressions: 6*DIV
    const real_t x0 = (1.0 / 5.0) * x[4];
    const real_t x1 = (1.0 / 5.0) * x[6];
    const real_t x2 = (1.0 / 5.0) * x[7];
    const real_t x3 = (1.0 / 5.0) * x[5];
    const real_t x4 = (1.0 / 5.0) * x[8];
    const real_t x5 = (1.0 / 5.0) * x[9];
    values[0] = x0 + x1 + x2 + x[0];
    values[1] = x0 + x3 + x4 + x[1];
    values[2] = x1 + x3 + x5 + x[2];
    values[3] = x2 + x4 + x5 + x[3];
    values[4] = (3.0 / 5.0) * x[4];
    values[5] = (3.0 / 5.0) * x[5];
    values[6] = (3.0 / 5.0) * x[6];
    values[7] = (3.0 / 5.0) * x[7];
    values[8] = (3.0 / 5.0) * x[8];
    values[9] = (3.0 / 5.0) * x[9];
}

void tet10_apply_inv_lumped_mass(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 const real_t *const x,
                                 real_t *const values) {
    double tick = MPI_Wtime();

    idx_t ev[10];
    real_t element_x[10];
    real_t element_x_pre_trafo[10];
    real_t element_weights[10];

    // Apply diagonal
    {
        real_t *weights = (real_t *)malloc(nnodes * sizeof(real_t));
        memset(weights, 0, nnodes * sizeof(real_t));

        for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(10)
            for (int v = 0; v < 10; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            lumped_mass_kernel(
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
                element_weights);

            for (int v = 0; v < 10; ++v) {
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
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            element_x_pre_trafo[v] = values[elems[v][i]];
        }

        tet10_transform_kernel(element_x_pre_trafo, element_x);

        for (int v = 0; v < 10; ++v) {
            const idx_t idx = ev[v];
            values[idx] = element_x[v];
        }
    }

    double tock = MPI_Wtime();
    printf("tet10_mass.c: tet10_apply_inv_lumped_mass\t%g seconds\n", tock - tick);
}

static SFEM_INLINE void tet10_hrz_lumped_mass_kernel(const real_t px0,
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
                                                     real_t *const SFEM_RESTRICT
                                                         element_matrix_diag) {
    // const real_t x0 = py0 - py2;
    // const real_t x1 = pz0 - pz3;
    // const real_t x2 = x0 * x1;
    // const real_t x3 = px0 - px1;
    // const real_t x4 = py0 - py3;
    // const real_t x5 = pz0 - pz2;
    // const real_t x6 = px0 - px2;
    // const real_t x7 = py0 - py1;
    // const real_t x8 = pz0 - pz1;
    // const real_t x9 = x4 * x8;
    // const real_t x10 = x5 * x7;
    // const real_t x11 = px0 - px3;
    // const real_t x12 = -x1;
    // const real_t x13 = -x0;
    // const real_t x14 = -1.0 / 6.0 * x3;
    // const real_t x15 = -x4;
    // const real_t x16 = -x5;
    // const real_t x17 = -x7;
    // const real_t x18 = -1.0 / 6.0 * x6;
    // const real_t x19 = -x8;
    // const real_t x20 = -1.0 / 6.0 * x11;
    // const real_t x21 =
    //     (x12 * x13 * x14 - x12 * x17 * x18 - x13 * x19 * x20 - x14 * x15 * x16 + x15 * x18 * x19 +
    //      x16 * x17 * x20) /
    //     ((3.0 / 35.0) * x0 * x11 * x8 + (3.0 / 35.0) * x1 * x6 * x7 - 3.0 / 35.0 * x10 * x11 -
    //      3.0 / 35.0 * x2 * x3 + (3.0 / 35.0) * x3 * x4 * x5 - 3.0 / 35.0 * x6 * x9);
    // const real_t x22 = x21 * ((1.0 / 420.0) * x0 * x11 * x8 + (1.0 / 420.0) * x1 * x6 * x7 -
    //                           1.0 / 420.0 * x10 * x11 - 1.0 / 420.0 * x2 * x3 +
    //                           (1.0 / 420.0) * x3 * x4 * x5 - 1.0 / 420.0 * x6 * x9);
    // const real_t x23 = x21 * ((4.0 / 315.0) * x0 * x11 * x8 + (4.0 / 315.0) * x1 * x6 * x7 -
    //                           4.0 / 315.0 * x10 * x11 - 4.0 / 315.0 * x2 * x3 +
    //                           (4.0 / 315.0) * x3 * x4 * x5 - 4.0 / 315.0 * x6 * x9);
    // element_matrix_diag[0] = x22;
    // element_matrix_diag[1] = x22;
    // element_matrix_diag[2] = x22;
    // element_matrix_diag[3] = x22;
    // element_matrix_diag[4] = x23;
    // element_matrix_diag[5] = x23;
    // element_matrix_diag[6] = x23;
    // element_matrix_diag[7] = x23;
    // element_matrix_diag[8] = x23;
    // element_matrix_diag[9] = x23;
    const real_t x0 = px0 - px1;
    const real_t x1 = (1.0/120.0)*x0;
    const real_t x2 = pz0 - pz3;
    const real_t x3 = py0 - py2;
    const real_t x4 = x2*x3;
    const real_t x5 = py0 - py3;
    const real_t x6 = pz0 - pz2;
    const real_t x7 = px0 - px2;
    const real_t x8 = (1.0/120.0)*x7;
    const real_t x9 = py0 - py1;
    const real_t x10 = pz0 - pz1;
    const real_t x11 = x10*x5;
    const real_t x12 = px0 - px3;
    const real_t x13 = (1.0/120.0)*x12;
    const real_t x14 = x6*x9;
    const real_t x15 = x1*x4 - x1*x5*x6 - x10*x13*x3 + x11*x8 + x13*x14 - x2*x8*x9;
    const real_t x16 = -1.0/30.0*x0*x4 + (1.0/30.0)*x0*x5*x6 + (1.0/30.0)*x10*x12*x3 - 1.0/30.0*x11*x7 - 1.0/30.0*x12*x14 + (1.0/30.0)*x2*x7*x9;
    element_matrix_diag[0] = x15;
    element_matrix_diag[1] = x15;
    element_matrix_diag[2] = x15;
    element_matrix_diag[3] = x15;
    element_matrix_diag[4] = x16;
    element_matrix_diag[5] = x16;
    element_matrix_diag[6] = x16;
    element_matrix_diag[7] = x16;
    element_matrix_diag[8] = x16;
    element_matrix_diag[9] = x16;
}

void tet10_assemble_lumped_mass(const ptrdiff_t nelements,
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
            idx_t ev[10];
            real_t element_vector[10];

#pragma unroll(10)
            for (int v = 0; v < 10; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            tet10_hrz_lumped_mass_kernel(
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
                element_vector);

#pragma unroll(10)
            for (int edof_i = 0; edof_i < 10; ++edof_i) {
#pragma omp atomic update
                values[ev[edof_i]] += element_vector[edof_i];
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tet10_mass.c: tet10_assemble_lumped_mass\t%g seconds\n", tock - tick);
}
