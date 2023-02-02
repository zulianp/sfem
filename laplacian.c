#include "laplacian.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"

static SFEM_INLINE void laplacian_hessian(const real_t x0,
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
    //    - Result: 4*ADD + 16*ASSIGNMENT + 16*MUL + 12*POW
    //    - Subexpressions: 16*ADD + 9*DIV + 56*MUL + 7*NEG + POW + 32*SUB
    const real_t x4 = z0 - z3;
    const real_t x5 = x0 - x1;
    const real_t x6 = y0 - y2;
    const real_t x7 = x5 * x6;
    const real_t x8 = z0 - z1;
    const real_t x9 = x0 - x2;
    const real_t x10 = y0 - y3;
    const real_t x11 = x10 * x9;
    const real_t x12 = z0 - z2;
    const real_t x13 = x0 - x3;
    const real_t x14 = y0 - y1;
    const real_t x15 = x13 * x14;
    const real_t x16 = x10 * x5;
    const real_t x17 = x14 * x9;
    const real_t x18 = x13 * x6;
    const real_t x19 = x11 * x8 + x12 * x15 - x12 * x16 - x17 * x4 - x18 * x8 + x4 * x7;
    const real_t x20 = 1.0 / x19;
    const real_t x21 = x11 - x18;
    const real_t x22 = -x17 + x7;
    const real_t x23 = x15 - x16 + x21 + x22;
    const real_t x24 = -x12 * x13 + x4 * x9;
    const real_t x25 = x12 * x5 - x8 * x9;
    const real_t x26 = x13 * x8;
    const real_t x27 = x4 * x5;
    const real_t x28 = x26 - x27;
    const real_t x29 = -x24 - x25 - x28;
    const real_t x30 = x10 * x8;
    const real_t x31 = x14 * x4;
    const real_t x32 = -x10 * x12 + x4 * x6;
    const real_t x33 = x12 * x14 - x6 * x8;
    const real_t x34 = x30 - x31 + x32 + x33;
    const real_t x35 = -x12;
    const real_t x36 = -x9;
    const real_t x37 = x19 * (x13 * x35 + x28 - x35 * x5 - x36 * x4 + x36 * x8);
    const real_t x38 = -x19;
    const real_t x39 = -x23;
    const real_t x40 = -x34;
    const real_t x41 = (1.0 / 6.0) / pow(x19, 2);
    const real_t x42 = x41 * (x24 * x37 + x38 * (x21 * x39 + x32 * x40));
    const real_t x43 = -x15 + x16;
    const real_t x44 = (1.0 / 6.0) * x43;
    const real_t x45 = -x26 + x27;
    const real_t x46 = -x30 + x31;
    const real_t x47 = (1.0 / 6.0) * x46;
    const real_t x48 = x20 * (-x23 * x44 + (1.0 / 6.0) * x29 * x45 - x34 * x47);
    const real_t x49 = x41 * (x25 * x37 + x38 * (x22 * x39 + x33 * x40));
    const real_t x50 = (1.0 / 6.0) * x45;
    const real_t x51 = x20 * (x21 * x44 + x24 * x50 + x32 * x47);
    const real_t x52 = x20 * (-1.0 / 6.0 * x21 * x22 - 1.0 / 6.0 * x24 * x25 - 1.0 / 6.0 * x32 * x33);
    const real_t x53 = x20 * (x22 * x44 + x25 * x50 + x33 * x47);

    element_matrix[0] = x20 * (-1.0 / 6.0 * pow(x23, 2) - 1.0 / 6.0 * pow(x29, 2) - 1.0 / 6.0 * pow(x34, 2));
    element_matrix[1] = x42;
    element_matrix[2] = x48;
    element_matrix[3] = x49;
    element_matrix[4] = x42;
    element_matrix[5] = x20 * (-1.0 / 6.0 * pow(x21, 2) - 1.0 / 6.0 * pow(x24, 2) - 1.0 / 6.0 * pow(x32, 2));
    element_matrix[6] = x51;
    element_matrix[7] = x52;
    element_matrix[8] = x48;
    element_matrix[9] = x51;
    element_matrix[10] = x20 * (-1.0 / 6.0 * pow(x43, 2) - 1.0 / 6.0 * pow(x45, 2) - 1.0 / 6.0 * pow(x46, 2));
    element_matrix[11] = x53;
    element_matrix[12] = x49;
    element_matrix[13] = x52;
    element_matrix[14] = x53;
    element_matrix[15] = x20 * (-1.0 / 6.0 * pow(x22, 2) - 1.0 / 6.0 * pow(x25, 2) - 1.0 / 6.0 * pow(x33, 2));
}

static SFEM_INLINE void laplacian_gradient(const real_t px0,
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
                                           const real_t *u,
                                           real_t *element_vector) {
    // FLOATING POINT OPS!
    //      - Result: 4*ADD + 4*ASSIGNMENT + 16*MUL
    //      - Subexpressions: 13*ADD + 7*DIV + 46*MUL + 3*NEG + 30*SUB
    const real_t x0 = -pz0 + pz3;
    const real_t x1 = -px0 + px1;
    const real_t x2 = -py0 + py2;
    const real_t x3 = x1 * x2;
    const real_t x4 = x0 * x3;
    const real_t x5 = -pz0 + pz2;
    const real_t x6 = -py0 + py3;
    const real_t x7 = x1 * x6;
    const real_t x8 = x5 * x7;
    const real_t x9 = -px0 + px2;
    const real_t x10 = -py0 + py1;
    const real_t x11 = x10 * x9;
    const real_t x12 = x0 * x11;
    const real_t x13 = -pz0 + pz1;
    const real_t x14 = x6 * x9;
    const real_t x15 = x13 * x14;
    const real_t x16 = -px0 + px3;
    const real_t x17 = x10 * x16 * x5;
    const real_t x18 = x16 * x2;
    const real_t x19 = x13 * x18;
    const real_t x20 =
        -1.0 / 6.0 * x12 + (1.0 / 6.0) * x15 + (1.0 / 6.0) * x17 - 1.0 / 6.0 * x19 + (1.0 / 6.0) * x4 - 1.0 / 6.0 * x8;
    const real_t x21 = 1.0 / (-x12 + x15 + x17 - x19 + x4 - x8);
    const real_t x22 = x21 * (-x11 + x3);
    const real_t x23 = x21 * (x10 * x16 - x7);
    const real_t x24 = x21 * (x14 - x18);
    const real_t x25 = -x22 - x23 - x24;
    const real_t x26 = u[0] * x25 + u[1] * x24 + u[2] * x23 + u[3] * x22;
    const real_t x27 = x21 * (-x1 * x5 + x13 * x9);
    const real_t x28 = x21 * (x0 * x1 - x13 * x16);
    const real_t x29 = x21 * (-x0 * x9 + x16 * x5);
    const real_t x30 = -x27 - x28 - x29;
    const real_t x31 = u[0] * x30 + u[1] * x29 + u[2] * x28 + u[3] * x27;
    const real_t x32 = x21 * (x10 * x5 - x13 * x2);
    const real_t x33 = x21 * (-x0 * x10 + x13 * x6);
    const real_t x34 = x21 * (x0 * x2 - x5 * x6);
    const real_t x35 = -x32 - x33 - x34;
    const real_t x36 = u[0] * x35 + u[1] * x34 + u[2] * x33 + u[3] * x32;
    element_vector[0] = x20 * (x25 * x26 + x30 * x31 + x35 * x36);
    element_vector[1] = x20 * (x24 * x26 + x29 * x31 + x34 * x36);
    element_vector[2] = x20 * (x23 * x26 + x28 * x31 + x33 * x36);
    element_vector[3] = x20 * (x22 * x26 + x27 * x31 + x32 * x36);
}

static SFEM_INLINE void laplacian_value(const real_t px0,
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
                                        const real_t *u,
                                        real_t *element_scalar) {
    // FLOATING POINT OPS!
    //       - Result: 8*ADD + ASSIGNMENT + 31*MUL + 3*POW
    //       - Subexpressions: 2*ADD + DIV + 34*MUL + 21*SUB
    const real_t x0 = -pz0 + pz3;
    const real_t x1 = -px0 + px1;
    const real_t x2 = -py0 + py2;
    const real_t x3 = x1 * x2;
    const real_t x4 = x0 * x3;
    const real_t x5 = -pz0 + pz2;
    const real_t x6 = -py0 + py3;
    const real_t x7 = x1 * x6;
    const real_t x8 = x5 * x7;
    const real_t x9 = -px0 + px2;
    const real_t x10 = -py0 + py1;
    const real_t x11 = x10 * x9;
    const real_t x12 = x0 * x11;
    const real_t x13 = -pz0 + pz1;
    const real_t x14 = x6 * x9;
    const real_t x15 = x13 * x14;
    const real_t x16 = -px0 + px3;
    const real_t x17 = x10 * x16 * x5;
    const real_t x18 = x16 * x2;
    const real_t x19 = x13 * x18;
    const real_t x20 = 1.0 / (-x12 + x15 + x17 - x19 + x4 - x8);
    const real_t x21 = x20 * (x14 - x18);
    const real_t x22 = x20 * (x10 * x16 - x7);
    const real_t x23 = x20 * (-x11 + x3);
    const real_t x24 = x20 * (-x0 * x9 + x16 * x5);
    const real_t x25 = x20 * (x0 * x1 - x13 * x16);
    const real_t x26 = x20 * (-x1 * x5 + x13 * x9);
    const real_t x27 = x20 * (x0 * x2 - x5 * x6);
    const real_t x28 = x20 * (-x0 * x10 + x13 * x6);
    const real_t x29 = x20 * (x10 * x5 - x13 * x2);
    element_scalar[0] = ((1.0 / 2.0) * pow(u[0] * (-x21 - x22 - x23) + u[1] * x21 + u[2] * x22 + u[3] * x23, 2) +
                         (1.0 / 2.0) * pow(u[0] * (-x24 - x25 - x26) + u[1] * x24 + u[2] * x25 + u[3] * x26, 2) +
                         (1.0 / 2.0) * pow(u[0] * (-x27 - x28 - x29) + u[1] * x27 + u[2] * x28 + u[3] * x29, 2)) *
                        (-1.0 / 6.0 * x12 + (1.0 / 6.0) * x15 + (1.0 / 6.0) * x17 - 1.0 / 6.0 * x19 + (1.0 / 6.0) * x4 -
                         1.0 / 6.0 * x8);
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

static SFEM_INLINE void find_cols4(const idx_t *targets, const idx_t *const row, const int lenrow, int *ks) {
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

void laplacian_assemble_hessian(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const elems,
                                geom_t **const xyz,
                                const idx_t *const rowptr,
                                const idx_t *const colidx,
                                real_t *const values) {
    double tick = MPI_Wtime();

    idx_t ev[4];
    idx_t ks[4];

    real_t element_matrix[4 * 4];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        laplacian_hessian(
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

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t dof_i = elems[edof_i][i];
            const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

            const idx_t *row = &colidx[rowptr[dof_i]];

            find_cols4(ev, row, lenrow, ks);

            real_t *rowvalues = &values[rowptr[dof_i]];
            const real_t *element_row = &element_matrix[edof_i * 4];

#pragma unroll(4)
            for (int edof_j = 0; edof_j < 4; ++edof_j) {
                rowvalues[ks[edof_j]] += element_row[edof_j];
            }
        }
    }

    double tock = MPI_Wtime();
    printf("laplacian.c: laplacian_assemble_hessian\t%g seconds\n", tock - tick);
}

void laplacian_assemble_gradient(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const elems,
                                 geom_t **const xyz,
                                 const real_t *const u,
                                 real_t *const values) {
    double tick = MPI_Wtime();

    idx_t ev[4];
    real_t element_vector[4 * 4];
    real_t element_u[4];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            element_u[v] = u[ev[v]];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        laplacian_gradient(
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
            element_u,
            element_vector);

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t dof_i = ev[edof_i];
            values[dof_i] += element_vector[edof_i];
        }
    }

    double tock = MPI_Wtime();
    printf("laplacian.c: laplacian_assemble_gradient\t%g seconds\n", tock - tick);
}

void laplacian_assemble_value(const ptrdiff_t nelements,
                              const ptrdiff_t nnodes,
                              idx_t **const elems,
                              geom_t **const xyz,
                              const real_t *const u,
                              real_t *const value) {
    double tick = MPI_Wtime();

    idx_t ev[4];
    real_t element_u[4];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            element_u[v] = u[ev[v]];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        real_t element_scalar = 0;

        laplacian_value(
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
            element_u,
            &element_scalar);

        *value += element_scalar;
    }

    double tock = MPI_Wtime();
    printf("laplacian.c: laplacian_assemble_value\t%g seconds\n", tock - tick);
}
