#include "tri3_stokes_mini.h"

#include "sfem_defs.h"
#include "sfem_vec.h"
#include "sortreduce.h"

#include <mpi.h>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static int check_symmetric(int n, const real_t *const element_matrix) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const real_t diff = element_matrix[i * n + j] - element_matrix[i + j * n];
            assert(diff < 1e-16);
            if (diff > 1e-16) {
                return 1;
            }

            // printf("%g ", element_matrix[i * n + j]);
        }

        // printf("\n");
    }

    // printf("\n");

    return 0;
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

static SFEM_INLINE void find_cols3(const idx_t *targets,
                                   const idx_t *const row,
                                   const int lenrow,
                                   int *ks) {
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

static SFEM_INLINE void tri3_stokes_assemble_hessian_kernel(const real_t mu,
                                                            const real_t px0,
                                                            const real_t px1,
                                                            const real_t px2,
                                                            const real_t py0,
                                                            const real_t py1,
                                                            const real_t py2,
                                                            real_t *const SFEM_RESTRICT
                                                                element_matrix) {
    const real_t x0 = px0 - px1;
    const real_t x1 = -px2;
    const real_t x2 = px0 + x1;
    const real_t x3 = x0 * x2;
    const real_t x4 = py0 - py1;
    const real_t x5 = -py2;
    const real_t x6 = py0 + x5;
    const real_t x7 = x4 * x6;
    const real_t x8 = pow(x2, 2);
    const real_t x9 = pow(x6, 2);
    const real_t x10 = x8 + x9;
    const real_t x11 = pow(x0, 2) + pow(x4, 2);
    const real_t x12 = x0 * x6;
    const real_t x13 = x2 * x4;
    const real_t x14 = x12 - x13;
    const real_t x15 = (1.0 / 2.0) * mu / x14;
    const real_t x16 = x15 * (x10 + x11 - 2 * x3 - 2 * x7);
    const real_t x17 = x3 + x7;
    const real_t x18 = mu / (2 * x12 - 2 * x13);
    const real_t x19 = x18 * (x17 - x8 - x9);
    const real_t x20 = x11 - x3 - x7;
    const real_t x21 = -x18 * x20;
    const real_t x22 = (1.0 / 6.0) * py1;
    const real_t x23 = -1.0 / 6.0 * py2;
    const real_t x24 = -x22 - x23;
    const real_t x25 = x10 * x15;
    const real_t x26 = -x17 * x18;
    const real_t x27 = (1.0 / 6.0) * py0;
    const real_t x28 = x23 + x27;
    const real_t x29 = x11 * x15;
    const real_t x30 = x22 - x27;
    const real_t x31 = (1.0 / 6.0) * px1;
    const real_t x32 = -1.0 / 6.0 * px2;
    const real_t x33 = x31 + x32;
    const real_t x34 = (1.0 / 6.0) * px0;
    const real_t x35 = -x32 - x34;
    const real_t x36 = -x31 + x34;
    const real_t x37 = -px1 - x1;
    const real_t x38 = -py1 - x5;
    const real_t x39 = (1.0 / 80.0) / (mu * (x10 + x20));
    const real_t x40 = x14 * x39;
    const real_t x41 = x40 * (-x2 * x37 - x38 * x6);
    const real_t x42 = x40 * (x0 * x37 + x38 * x4);
    const real_t x43 = -x14 * x39;
    const real_t x44 = x17 * x40;
    element_matrix[0] = x16;
    element_matrix[1] = x19;
    element_matrix[2] = x21;
    element_matrix[3] = 0;
    element_matrix[4] = 0;
    element_matrix[5] = 0;
    element_matrix[6] = x24;
    element_matrix[7] = x24;
    element_matrix[8] = x24;
    element_matrix[9] = x19;
    element_matrix[10] = x25;
    element_matrix[11] = x26;
    element_matrix[12] = 0;
    element_matrix[13] = 0;
    element_matrix[14] = 0;
    element_matrix[15] = x28;
    element_matrix[16] = x28;
    element_matrix[17] = x28;
    element_matrix[18] = x21;
    element_matrix[19] = x26;
    element_matrix[20] = x29;
    element_matrix[21] = 0;
    element_matrix[22] = 0;
    element_matrix[23] = 0;
    element_matrix[24] = x30;
    element_matrix[25] = x30;
    element_matrix[26] = x30;
    element_matrix[27] = 0;
    element_matrix[28] = 0;
    element_matrix[29] = 0;
    element_matrix[30] = x16;
    element_matrix[31] = x19;
    element_matrix[32] = x21;
    element_matrix[33] = x33;
    element_matrix[34] = x33;
    element_matrix[35] = x33;
    element_matrix[36] = 0;
    element_matrix[37] = 0;
    element_matrix[38] = 0;
    element_matrix[39] = x19;
    element_matrix[40] = x25;
    element_matrix[41] = x26;
    element_matrix[42] = x35;
    element_matrix[43] = x35;
    element_matrix[44] = x35;
    element_matrix[45] = 0;
    element_matrix[46] = 0;
    element_matrix[47] = 0;
    element_matrix[48] = x21;
    element_matrix[49] = x26;
    element_matrix[50] = x29;
    element_matrix[51] = x36;
    element_matrix[52] = x36;
    element_matrix[53] = x36;
    element_matrix[54] = x24;
    element_matrix[55] = x28;
    element_matrix[56] = x30;
    element_matrix[57] = x33;
    element_matrix[58] = x35;
    element_matrix[59] = x36;
    element_matrix[60] = x40 * (-pow(x37, 2) - pow(x38, 2));
    element_matrix[61] = x41;
    element_matrix[62] = x42;
    element_matrix[63] = x24;
    element_matrix[64] = x28;
    element_matrix[65] = x30;
    element_matrix[66] = x33;
    element_matrix[67] = x35;
    element_matrix[68] = x36;
    element_matrix[69] = x41;
    element_matrix[70] = x10 * x43;
    element_matrix[71] = x44;
    element_matrix[72] = x24;
    element_matrix[73] = x28;
    element_matrix[74] = x30;
    element_matrix[75] = x33;
    element_matrix[76] = x35;
    element_matrix[77] = x36;
    element_matrix[78] = x42;
    element_matrix[79] = x44;
    element_matrix[80] = x11 * x43;
}

static SFEM_INLINE void tri3_stokes_mini_assemble_rhs_kernel(
    const real_t mu,
    const real_t rho,
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t *const SFEM_RESTRICT u_rhs,
    const real_t *const SFEM_RESTRICT p_rhs,
    real_t *const SFEM_RESTRICT element_vector) {
    // const real_t x0 = u_rhs[2] + u_rhs[3];
    // const real_t x1 = px0 - px1;
    // const real_t x2 = -py2;
    // const real_t x3 = py0 + x2;
    // const real_t x4 = -px2;
    // const real_t x5 = px0 + x4;
    // const real_t x6 = py0 - py1;
    // const real_t x7 = x1 * x3 - x5 * x6;
    // const real_t x8 = rho * x7;
    // const real_t x9 = (1.0 / 24.0) * x8;
    // const real_t x10 = u_rhs[6] + u_rhs[7];
    // const real_t x11 = x7 * (u_rhs[1] + x0);
    // const real_t x12 = x7 * (u_rhs[5] + x10);
    // const real_t x13 = pow(x1, 2) - x1 * x5 + pow(x3, 2) - x3 * x6 + pow(x5, 2) + pow(x6, 2);
    // const real_t x14 = 10 * mu * x13;
    // const real_t x15 = (1.0 / 240.0) * x8 / (mu * x13);
    // element_vector[0] = x9 * (2 * u_rhs[1] + x0);
    // element_vector[1] = x9 * (u_rhs[1] + 2 * u_rhs[2] + u_rhs[3]);
    // element_vector[2] = x9 * (u_rhs[1] + u_rhs[2] + 2 * u_rhs[3]);
    // element_vector[3] = x9 * (2 * u_rhs[5] + x10);
    // element_vector[4] = x9 * (u_rhs[5] + 2 * u_rhs[6] + u_rhs[7]);
    // element_vector[5] = x9 * (u_rhs[5] + u_rhs[6] + 2 * u_rhs[7]);
    // element_vector[6] =
    //     x15 * (x11 * (-py1 - x2) - x12 * (-px1 - x4) + x14 * (2 * p_rhs[0] + p_rhs[1] +
    //     p_rhs[2]));
    // element_vector[7] = x15 * (x11 * x3 - x12 * x5 + x14 * (p_rhs[0] + 2 * p_rhs[1] + p_rhs[2]));
    // element_vector[8] = x15 * (x1 * x12 - x11 * x6 + x14 * (p_rhs[0] + p_rhs[1] + 2 * p_rhs[2]));
    //
    const real_t x0 = 5 * u_rhs[2];
    const real_t x1 = 9 * u_rhs[0];
    const real_t x2 = 5 * u_rhs[3] + x1;
    const real_t x3 = px0 - px1;
    const real_t x4 = -py2;
    const real_t x5 = py0 + x4;
    const real_t x6 = -px2;
    const real_t x7 = px0 + x6;
    const real_t x8 = py0 - py1;
    const real_t x9 = x3 * x5 - x7 * x8;
    const real_t x10 = rho * x9;
    const real_t x11 = (1.0 / 120.0) * x10;
    const real_t x12 = 5 * u_rhs[1];
    const real_t x13 = 5 * u_rhs[6];
    const real_t x14 = 9 * u_rhs[4];
    const real_t x15 = 5 * u_rhs[7] + x14;
    const real_t x16 = 5 * u_rhs[5];
    const real_t x17 = x9 * (27 * u_rhs[0] + 14 * u_rhs[1] + 14 * u_rhs[2] + 14 * u_rhs[3]);
    const real_t x18 = x9 * (27 * u_rhs[4] + 14 * u_rhs[5] + 14 * u_rhs[6] + 14 * u_rhs[7]);
    const real_t x19 = pow(x3, 2) - x3 * x7 + pow(x5, 2) - x5 * x8 + pow(x7, 2) + pow(x8, 2);
    const real_t x20 = 140 * mu * x19;
    const real_t x21 = (1.0 / 3360.0) * x10 / (mu * x19);
    element_vector[0] = x11 * (10 * u_rhs[1] + x0 + x2);
    element_vector[1] = x11 * (10 * u_rhs[2] + x12 + x2);
    element_vector[2] = x11 * (10 * u_rhs[3] + x0 + x1 + x12);
    element_vector[3] = x11 * (10 * u_rhs[5] + x13 + x15);
    element_vector[4] = x11 * (10 * u_rhs[6] + x15 + x16);
    element_vector[5] = x11 * (10 * u_rhs[7] + x13 + x14 + x16);
    element_vector[6] =
        x21 * (x17 * (-py1 - x4) - x18 * (-px1 - x6) + x20 * (2 * p_rhs[0] + p_rhs[1] + p_rhs[2]));
    element_vector[7] = x21 * (x17 * x5 - x18 * x7 + x20 * (p_rhs[0] + 2 * p_rhs[1] + p_rhs[2]));
    element_vector[8] = x21 * (-x17 * x8 + x18 * x3 + x20 * (p_rhs[0] + p_rhs[1] + 2 * p_rhs[2]));

    //
    // const real_t x0 = (1.0/24.0)*rho*((px0 - px1)*(py0 - py2) - (px0 - px2)*(py0 -
    // py1));
    // element_vector[0] = x0*(2*u_rhs[1] + u_rhs[2] + u_rhs[3]);
    // element_vector[1] = x0*(u_rhs[1] + 2*u_rhs[2] + u_rhs[3]);
    // element_vector[2] = x0*(u_rhs[1] + u_rhs[2] + 2*u_rhs[3]);
    // element_vector[3] = x0*(2*u_rhs[5] + u_rhs[6] + u_rhs[7]);
    // element_vector[4] = x0*(u_rhs[5] + 2*u_rhs[6] + u_rhs[7]);
    // element_vector[5] = x0*(u_rhs[5] + u_rhs[6] + 2*u_rhs[7]);
    // element_vector[6] = x0*(2*p_rhs[0] + p_rhs[1] + p_rhs[2]);
    // element_vector[7] = x0*(p_rhs[0] + 2*p_rhs[1] + p_rhs[2]);
    // element_vector[8] = x0*(p_rhs[0] + p_rhs[1] + 2*p_rhs[2]);
}

static SFEM_INLINE void tri3_stokes_mini_apply_kernel(const real_t mu,
                                                      // const real_t rho,
                                                      const real_t px0,
                                                      const real_t px1,
                                                      const real_t px2,
                                                      const real_t py0,
                                                      const real_t py1,
                                                      const real_t py2,
                                                      const real_t *const SFEM_RESTRICT increment,
                                                      real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = px0 - px1;
    const real_t x1 = -py2;
    const real_t x2 = py0 + x1;
    const real_t x3 = -px2;
    const real_t x4 = px0 + x3;
    const real_t x5 = py0 - py1;
    const real_t x6 = x0 * x2 - x4 * x5;
    const real_t x7 = 1.0 / x6;
    const real_t x8 = -py1 - x1;
    const real_t x9 = increment[6] + increment[7] + increment[8];
    const real_t x10 = (1.0 / 6.0) * x6;
    const real_t x11 = x10 * x9;
    const real_t x12 = x0 * x4;
    const real_t x13 = x2 * x5;
    const real_t x14 = pow(x0, 2) + pow(x5, 2);
    const real_t x15 = -x12 - x13 + x14;
    const real_t x16 = pow(x4, 2);
    const real_t x17 = pow(x2, 2);
    const real_t x18 = x12 + x13;
    const real_t x19 = -x16 - x17 + x18;
    const real_t x20 = x16 + x17;
    const real_t x21 = -2 * x12 - 2 * x13 + x14 + x20;
    const real_t x22 = (1.0 / 2.0) * mu;
    const real_t x23 = increment[1] * x22;
    const real_t x24 = increment[0] * x22;
    const real_t x25 = -x9;
    const real_t x26 = px1 + x3;
    const real_t x27 = -x26;
    const real_t x28 = x10 * x25;
    const real_t x29 = increment[4] * x22;
    const real_t x30 = increment[3] * x22;
    const real_t x31 = (1.0 / 80.0) * pow(x6, 2) * (-x2 * x8 + x26 * x4);
    const real_t x32 = x0 * x2 - x4 * x5;
    const real_t x33 = (1.0 / 80.0) * x6;
    const real_t x34 = x32 * x33;
    const real_t x35 = x0 * x26 - x5 * x8;
    const real_t x36 = x15 + x20;
    const real_t x37 = (1.0 / 6.0) * mu * x36;
    const real_t x38 = x37 * (increment[0] * x8 - increment[3] * x27);
    const real_t x39 =
        x37 * (increment[1] * x2 - increment[2] * x5 - increment[4] * x4 + increment[5] * x0);
    const real_t x40 = x32 * x38 + x32 * x39;
    const real_t x41 = 1 / (mu * x36);
    const real_t x42 = x41 / x32;
    element_vector[0] =
        x7 * (x11 * x8 + x22 * (increment[0] * x21 + increment[1] * x19 - increment[2] * x15));
    element_vector[1] = x7 * (-increment[2] * x18 * x22 + x11 * x2 + x19 * x24 + x20 * x23);
    element_vector[2] = x7 * ((1.0 / 2.0) * increment[2] * mu * x14 - x15 * x24 - x18 * x23 +
                              (1.0 / 6.0) * x25 * x5 * x6);
    element_vector[3] =
        x7 * (x22 * (increment[3] * x21 + increment[4] * x19 - increment[5] * x15) + x27 * x28);
    element_vector[4] = x7 * (-increment[5] * x18 * x22 + x19 * x30 + x20 * x29 + x28 * x4);
    element_vector[5] = x7 * ((1.0 / 2.0) * increment[5] * mu * x14 + (1.0 / 6.0) * x0 * x6 * x9 -
                              x15 * x30 - x18 * x29);
    element_vector[6] = x42 * (-increment[6] * x34 * (pow(x26, 2) + pow(x8, 2)) +
                               increment[7] * x31 - increment[8] * x34 * x35 + x40);
    element_vector[7] =
        x42 * (increment[6] * x31 + x34 * (-increment[7] * x20 + increment[8] * x18) + x40);
    element_vector[8] = x41 * (-increment[6] * x33 * x35 +
                               x33 * (increment[7] * x18 - increment[8] * x14) + x38 + x39);
}

void tri3_stokes_mini_assemble_hessian_soa(const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const elems,
                                           geom_t **const points,
                                           const real_t mu,
                                           const count_t *const rowptr,
                                           const idx_t *const colidx,
                                           real_t **const values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    static const int n_vars = 3;
    static const int nxe = 3;
    // static const int rows = 9;
    static const int cols = 9;

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[3];
            idx_t ks[3][3];
            real_t element_matrix[3 * 3 * 3 * 3];
#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];

            const real_t x0 = points[0][i0];
            const real_t x1 = points[0][i1];
            const real_t x2 = points[0][i2];

            const real_t y0 = points[1][i0];
            const real_t y1 = points[1][i1];
            const real_t y2 = points[1][i2];

            tri3_stokes_assemble_hessian_kernel(mu,
                                                // X-coordinates
                                                x0,
                                                x1,
                                                x2,
                                                // Y-coordinates
                                                y0,
                                                y1,
                                                y2,
                                                element_matrix);

            // find all indices
            for (int edof_i = 0; edof_i < nxe; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];
                const idx_t r_begin = rowptr[dof_i];
                const idx_t lenrow = rowptr[dof_i + 1] - r_begin;
                const idx_t *row = &colidx[rowptr[dof_i]];
                find_cols3(ev, row, lenrow, ks[edof_i]);
            }

            for (int bi = 0; bi < n_vars; ++bi) {
                for (int bj = 0; bj < n_vars; ++bj) {
                    for (int edof_i = 0; edof_i < nxe; ++edof_i) {
                        const int ii = bi * nxe + edof_i;

                        const idx_t dof_i = elems[edof_i][i];
                        const idx_t r_begin = rowptr[dof_i];
                        const int bb = bi * n_vars + bj;

                        real_t *const row_values = &values[bb][r_begin];

                        for (int edof_j = 0; edof_j < nxe; ++edof_j) {
                            const int jj = bj * nxe + edof_j;
                            const real_t val = element_matrix[ii * cols + jj];

                            assert(val == val);
#pragma omp atomic update
                            row_values[ks[edof_i][edof_j]] += val;
                        }
                    }
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tri3_stokes.c: tri3_stokes_assemble_hessian\t%g seconds\n", tock - tick);
}

void tri3_stokes_mini_assemble_hessian_aos(const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const elems,
                                           geom_t **const points,
                                           const real_t mu,
                                           const count_t *const rowptr,
                                           const idx_t *const colidx,
                                           real_t *const values) {
    SFEM_UNUSED(nnodes);

    const double tick = MPI_Wtime();

    static const int block_size = 3;
    static const int mat_block_size = block_size * block_size;

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[3];
            idx_t ks[3];

            real_t element_matrix[(3 * 3) * (3 * 3)];

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];

            tri3_stokes_assemble_hessian_kernel(
                // Model parameters
                mu,
                // X-coordinates
                points[0][i0],
                points[0][i1],
                points[0][i2],
                // Y-coordinates
                points[1][i0],
                points[1][i1],
                points[1][i2],
                // output matrix
                element_matrix);

            assert(!check_symmetric(9, element_matrix));

            for (int edof_i = 0; edof_i < 3; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];
                const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

                {
                    const idx_t *row = &colidx[rowptr[dof_i]];
                    find_cols3(ev, row, lenrow, ks);
                }

                // Blocks for row
                real_t *block_start = &values[rowptr[dof_i] * mat_block_size];

                for (int edof_j = 0; edof_j < 3; ++edof_j) {
                    const idx_t offset_j = ks[edof_j] * block_size;

                    for (int bi = 0; bi < block_size; ++bi) {
                        const int ii = bi * 3 + edof_i;

                        // Jump rows (including the block-size for the columns)
                        real_t *row = &block_start[bi * lenrow * block_size];

                        for (int bj = 0; bj < block_size; ++bj) {
                            const int jj = bj * 3 + edof_j;
                            const real_t val = element_matrix[ii * 9 + jj];
#pragma omp atomic update
                            row[offset_j + bj] += val;
                        }
                    }
                }
            }
        }
    }
    const double tock = MPI_Wtime();
    printf("stokes.c: stokes_assemble_hessian_aos\t%g seconds\n", tock - tick);
}

void tri3_stokes_mini_assemble_rhs_soa(const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const elems,
                                       geom_t **const points,
                                       const real_t mu,
                                       const real_t rho,
                                       real_t **SFEM_RESTRICT forcing,
                                       real_t **const SFEM_RESTRICT rhs) {
    SFEM_UNUSED(nnodes);
    double tick = MPI_Wtime();

    static const int n_vars = 3;
    static const int ndofs = 3;
    static const int rows = 9;
    static const int cols = 9;

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[3];
            idx_t ks[3];
            real_t element_vector[3 * 3];
            real_t u_rhs[4 * 2];
            real_t p_rhs[3] = {0., 0., 0.};

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];

            const real_t x0 = points[0][i0];
            const real_t x1 = points[0][i1];
            const real_t x2 = points[0][i2];

            const real_t y0 = points[1][i0];
            const real_t y1 = points[1][i1];
            const real_t y2 = points[1][i2];

            memset(u_rhs, 0, 4 * 2 * sizeof(real_t));

            for (int v = 0; v < 2; v++) {
                for (int ii = 0; ii < 3; ii++) {
                    if (forcing[v]) {
                        // Skip bubble dof
                        u_rhs[v * 4 + ii + 1] = forcing[v][ev[ii]];
                    }
                }

                // Bubble function is not considered for this
                // if(forcing[v]) {
                //     u_rhs[v * 4 + 0] = (forcing[v][i0] + forcing[v][i1] + forcing[v][i2])/3;
                // }
            }

            if (forcing[2]) {
                for (int ii = 0; ii < 3; ii++) {
                    p_rhs[ii] = forcing[2][ev[ii]];
                }
            }

            tri3_stokes_mini_assemble_rhs_kernel(mu,
                                                 rho,
                                                 // X coords
                                                 points[0][i0],
                                                 points[0][i1],
                                                 points[0][i2],
                                                 // Y coords
                                                 points[1][i0],
                                                 points[1][i1],
                                                 points[1][i2],
                                                 //  buffers
                                                 u_rhs,
                                                 p_rhs,
                                                 element_vector);

            for (int edof_i = 0; edof_i < 3; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];

                // Add block
                for (int d1 = 0; d1 < n_vars; d1++) {
#pragma omp atomic update
                    rhs[d1][dof_i] += element_vector[d1 * 3 + edof_i];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tri3_stokes.c: tri3_stokes_mini_assemble_rhs\t%g seconds\n", tock - tick);
}

void tri3_stokes_mini_assemble_rhs_aos(const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const elems,
                                       geom_t **const points,
                                       const real_t mu,
                                       const real_t rho,
                                       real_t **SFEM_RESTRICT forcing,
                                       real_t *const SFEM_RESTRICT rhs) {
    SFEM_UNUSED(nnodes);
    double tick = MPI_Wtime();

    static const int n_vars = 3;
    static const int ndofs = 3;
    static const int rows = 9;
    static const int cols = 9;

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[3];
            idx_t ks[3];
            real_t element_vector[3 * 3];
            real_t u_rhs[4 * 2];
            real_t p_rhs[3] = {0., 0., 0.};

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];

            const real_t x0 = points[0][i0];
            const real_t x1 = points[0][i1];
            const real_t x2 = points[0][i2];

            const real_t y0 = points[1][i0];
            const real_t y1 = points[1][i1];
            const real_t y2 = points[1][i2];

            memset(u_rhs, 0, 4 * 2 * sizeof(real_t));

            for (int v = 0; v < 2; v++) {
                for (int ii = 0; ii < 3; ii++) {
                    if (forcing[v]) {
                        // Skip bubble dof
                        u_rhs[v * 4 + ii + 1] = forcing[v][ev[ii]];
                    }
                }
            }

            if (forcing[2]) {
                for (int ii = 0; ii < 3; ii++) {
                    p_rhs[ii] = forcing[2][ev[ii]];
                }
            }

            tri3_stokes_mini_assemble_rhs_kernel(mu,
                                                 rho,
                                                 // X coords
                                                 points[0][i0],
                                                 points[0][i1],
                                                 points[0][i2],
                                                 // Y coords
                                                 points[1][i0],
                                                 points[1][i1],
                                                 points[1][i2],
                                                 //  buffers
                                                 u_rhs,
                                                 p_rhs,
                                                 element_vector);

            for (int edof_i = 0; edof_i < 3; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];

                // Add block
                for (int d1 = 0; d1 < n_vars; d1++) {
#pragma omp atomic update
                    rhs[dof_i * n_vars + d1] += element_vector[d1 * 3 + edof_i];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tri3_stokes.c: tri3_stokes_mini_assemble_rhs\t%g seconds\n", tock - tick);
}

void tri3_stokes_mini_apply_aos(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const elems,
                                geom_t **const points,
                                const real_t mu,
                                // const real_t rho,
                                const real_t *const SFEM_RESTRICT x,
                                real_t *const SFEM_RESTRICT rhs) {
    SFEM_UNUSED(nnodes);
    double tick = MPI_Wtime();

    static const int n_vars = 3;
    static const int ndofs = 3;
    static const int rows = 9;
    static const int cols = 9;

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[3];
            idx_t ks[3];
            real_t element_vector[3 * 3];
            real_t element_x[3 * 3];

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                ev[v] = elems[v][i];
            }

            for (int enode = 0; enode < 3; ++enode) {
                idx_t dof = ev[enode] * n_vars;

                for (int b = 0; b < n_vars; ++b) {
                    element_x[b * 3 + enode] = x[dof + b];
                }
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];

            const real_t x0 = points[0][i0];
            const real_t x1 = points[0][i1];
            const real_t x2 = points[0][i2];

            const real_t y0 = points[1][i0];
            const real_t y1 = points[1][i1];
            const real_t y2 = points[1][i2];

            tri3_stokes_mini_apply_kernel(mu,
                                          // rho,
                                          // X coords
                                          points[0][i0],
                                          points[0][i1],
                                          points[0][i2],
                                          // Y coords
                                          points[1][i0],
                                          points[1][i1],
                                          points[1][i2],
                                          //  buffers
                                          element_x,
                                          element_vector);

            for (int edof_i = 0; edof_i < 3; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];

                // Add block
                for (int d1 = 0; d1 < n_vars; d1++) {
#pragma omp atomic update
                    rhs[dof_i * n_vars + d1] += element_vector[d1 * 3 + edof_i];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tri3_stokes.c: tri3_stokes_mini_apply\t%g seconds\n", tock - tick);
}

void tri3_stokes_mini_assemble_gradient_aos(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const elems,
                                geom_t **const points,
                                const real_t mu,
                                const real_t *const SFEM_RESTRICT x,
                                real_t *const SFEM_RESTRICT g)
{
    tri3_stokes_mini_apply_aos(nelements, nnodes, elems, points, mu, x, g);
}
