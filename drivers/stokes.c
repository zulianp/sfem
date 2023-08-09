#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "../matrix.io/array_dtof.h"
#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/matrixio_crs.h"
#include "../matrix.io/utils.h"

#include "crs_graph.h"
#include "sfem_base.h"
#include "sfem_defs.h"
#include "sfem_vec.h"
#include "sortreduce.h"

#include "mass.h"

#include "dirichlet.h"
#include "neumann.h"

#include "read_mesh.h"

// static SFEM_INLINE real_t ux1(const real_t x, const real_t y) {
//     return x * x * (1 - x) * (1 - x) * 2 * y * (1 - y) * (2 * y - 1);
// }

// static SFEM_INLINE real_t uy1(const real_t x, const real_t y) {
//     return y * y * (1 - y) * (1 - y) * 2 * x * (1 - x) * (1 - 2 * x);
// }

// static SFEM_INLINE real_t p1(const real_t x, const real_t y) {
//     return x * (1 - x) * (1 - y) - 1. / 12;
// }

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

static SFEM_INLINE real_t ref_p1(const real_t x, const real_t y) {
    return x * (1 - x) * (1 - y) - 1. / 12;
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

SFEM_INLINE void tri3_stokes_assemble_hessian_kernel(const real_t mu,
                                                     const real_t px0,
                                                     const real_t px1,
                                                     const real_t px2,
                                                     const real_t py0,
                                                     const real_t py1,
                                                     const real_t py2,
                                                     real_t *const SFEM_RESTRICT element_matrix) {
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

SFEM_INLINE void tri3_stokes_mini_assemble_rhs_kernel(const real_t mu,
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
}

void tri3_stokes_mini_assemble_hessian_soa(const real_t mu,
                                           enum ElemType element_type,
                                           const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const elems,
                                           geom_t **const points,
                                           const count_t *const rowptr,
                                           const idx_t *const colidx,
                                           real_t **const values) {
    assert(element_type == TRI3);
    SFEM_UNUSED(nnodes);
    SFEM_UNUSED(element_type);

    double tick = MPI_Wtime();

    static const int n_vars = 3;
    static const int nxe = 3;
    // static const int rows = 9;
    static const int cols = 9;

    idx_t ev[3];
    idx_t ks[3][3];
    real_t element_matrix[3 * 3 * 3 * 3];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
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
                        row_values[ks[edof_i][edof_j]] += val;
                    }
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tri3_stokes.c: tri3_stokes_assemble_hessian\t%g seconds\n", tock - tick);
}

// The MINI mixed finite element for the Stokes problem: An experimental
// investigation
static SFEM_INLINE void rhs1(const real_t mu, const real_t x, const real_t y, real_t *const f) {
    f[0] = -mu * (4 * y * (1 - y) * (2 * y - 1) * ((1 - 2 * x) * (1 - 2 * x) - 2 * x * (1 - x)) +
                  12 * x * x * (1 - x) * (1 - x) * (1 - 2 * y)) +
           (1 - 2 * x) * (1 - y);

    f[1] = -mu * (4 * x * (1 - x) * (1 - 2 * x) * ((1 - 2 * y) * (1 - 2 * y) - 2 * y * (1 - y)) +
                  12 * y * y * (1 - y) * (1 - y) * (2 * x - 1)) -
           x * (1 - x);
}

static SFEM_INLINE void rhs2(const real_t mu, const real_t x, const real_t y, real_t *const f) {
    f[0] = -mu * ((2 - 12 * x + 12 * x * x) * (2 * y - 6 * y * y + 4 * y * y * y) +
                  (x * x - 2 * x * x * x + x * x * x * x) * (-12 + 24 * y)) +
           1. / 24;

    f[1] = mu * ((2 - 12 * y + 12 * y * y) * (2 * x - 6 * x * x + 4 * x * x * x) +
                 (y * y - 2 * y * y * y + y * y * y * y) * (-12 + 24 * x)) +
           1. / 24;
}

static SFEM_INLINE void rhs3(const real_t mu, const real_t x, const real_t y, real_t *const f) {
    const real_t pis4 = 4 * M_PI * M_PI;
    const real_t pi2 = 2 * M_PI;

    f[0] = -pis4 * mu * sin(pi2 * y) * (2 * cos(pi2 * x) - 1) + pis4 * sin(pi2 * x);
    f[1] = pis4 * mu * sin(pi2 * x) * (2 * cos(pi2 * y) - 1) - pis4 * sin(pi2 * y);
}

void tri3_stokes_mini_assemble_rhs_soa(const int tp_num,
                                       const real_t mu,
                                       const real_t rho,
                                       enum ElemType element_type,
                                       const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const elems,
                                       geom_t **const points,
                                       real_t **const rhs) {
    assert(element_type == TRI3);
    SFEM_UNUSED(nnodes);
    SFEM_UNUSED(element_type);

    double tick = MPI_Wtime();

    static const int n_vars = 3;
    static const int ndofs = 3;
    static const int rows = 9;
    static const int cols = 9;

    idx_t ev[3];
    idx_t ks[3];
    real_t element_vector[3 * 3];
    real_t fb[2] = {0, 0};
    real_t xx[4];
    real_t yy[4];

    real_t u_rhs[4 * 2];
    real_t p_rhs[3] = {0., 0., 0.};

    for (ptrdiff_t i = 0; i < nelements; ++i) {
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

        const real_t bx = (x0 + x1 + x2) / 3;
        const real_t by = (y0 + y1 + y2) / 3;

        xx[0] = bx;
        yy[0] = by;

        xx[1] = x0;
        yy[1] = y0;

        xx[2] = x1;
        yy[2] = y1;

        xx[3] = x2;
        yy[3] = y2;

        memset(u_rhs, 0, 4*2*sizeof(real_t));

        // Not in the bubble dof??
        for (int ii = 1; ii < 4; ii++) {
        // for (int ii = 0; ii < 4; ii++) {
            switch (tp_num) {
                case 1: {
                    rhs1(mu, xx[ii], yy[ii], fb);
                    break;
                }
                case 2: {
                    rhs2(mu, xx[ii], yy[ii], fb);
                    break;
                }
                case 3: {
                    rhs3(mu, xx[ii], yy[ii], fb);
                    break;
                }
                default: {
                    assert(0);
                    break;
                }
            }

            u_rhs[0 * 4 + ii] = fb[0];
            u_rhs[1 * 4 + ii] = fb[1];
        }

        tri3_stokes_mini_assemble_rhs_kernel(mu,
                                             rho,
                                             // X coords
                                             x0,
                                             x1,
                                             x2,
                                             // Y coords
                                             y0,
                                             y1,
                                             y2,
                                             //  buffers
                                             u_rhs,
                                             p_rhs,
                                             element_vector);

        for (int edof_i = 0; edof_i < 3; ++edof_i) {
            const idx_t dof_i = elems[edof_i][i];

            // Add block
            for (int d1 = 0; d1 < n_vars; d1++) {
                rhs[d1][dof_i] += element_vector[d1 * 3 + edof_i];
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tri3_stokes.c: tri3_stokes_mini_assemble_rhs\t%g seconds\n", tock - tick);
}

//////////////////////////////////////////////

void tri3_stokes_mini_assemble_hessian_aos(const real_t mu,
                                           enum ElemType element_type,
                                           const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const elems,
                                           geom_t **const points,
                                           const count_t *const rowptr,
                                           const idx_t *const colidx,
                                           real_t *const values) {
    SFEM_UNUSED(nnodes);

    const double tick = MPI_Wtime();

    static const int block_size = 3;
    static const int mat_block_size = block_size * block_size;

    // #pragma omp parallel
    {
        // #pragma omp for nowait
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
                            // #pragma omp atomic update
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

void tri3_stokes_mini_assemble_rhs_aos(const int tp_num,
                                       const real_t mu,
                                       const real_t rho,
                                       enum ElemType element_type,
                                       const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const elems,
                                       geom_t **const points,
                                       real_t *const rhs) {
    assert(element_type == TRI3);
    SFEM_UNUSED(nnodes);
    SFEM_UNUSED(element_type);

    double tick = MPI_Wtime();

    static const int n_vars = 3;
    static const int ndofs = 3;
    static const int rows = 9;

    idx_t ev[3];
    idx_t ks[3];
    real_t element_vector[rows];
    real_t fb[2] = {0, 0};
    real_t xx[4];
    real_t yy[4];

    real_t u_rhs[4 * 2];
    real_t p_rhs[3] = {0., 0., 0.};

    for (ptrdiff_t i = 0; i < nelements; ++i) {
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

        const real_t bx = (x0 + x1 + x2) / 3;
        const real_t by = (y0 + y1 + y2) / 3;

        xx[0] = bx;
        yy[0] = by;

        xx[1] = x0;
        yy[1] = y0;

        xx[2] = x1;
        yy[2] = y1;

        xx[3] = x2;
        yy[3] = y2;

        memset(u_rhs, 0, 4*2*sizeof(real_t));

        // Not in the bubble dof??
        for (int ii = 1; ii < 4; ii++) {
        // for (int ii = 0; ii < (ndofs + 1); ii++) {
            switch (tp_num) {
                case 1: {
                    rhs1(mu, xx[ii], yy[ii], fb);
                    break;
                }
                case 2: {
                    rhs2(mu, xx[ii], yy[ii], fb);
                    break;
                }
                case 3: {
                    rhs3(mu, xx[ii], yy[ii], fb);
                    break;
                }
                default: {
                    assert(0);
                    break;
                }
            }

            u_rhs[0 * (ndofs + 1) + ii] = fb[0];
            u_rhs[1 * (ndofs + 1) + ii] = fb[1];
        }

        tri3_stokes_mini_assemble_rhs_kernel(mu,
                                             rho,
                                             // X coords
                                             x0,
                                             x1,
                                             x2,
                                             // Y coords
                                             y0,
                                             y1,
                                             y2,
                                             //  buffers
                                             u_rhs,
                                             p_rhs,
                                             element_vector);

        for (int edof_i = 0; edof_i < 3; ++edof_i) {
            const idx_t dof_i = elems[edof_i][i];

            // Add block
            for (int d1 = 0; d1 < n_vars; d1++) {
                rhs[dof_i * n_vars + d1] += element_vector[d1 * 3 + edof_i];
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tri3_stokes.c: tri3_stokes_mini_assemble_rhs\t%g seconds\n", tock - tick);
}

//////////////////////////////////////////////

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 3) {
        fprintf(stderr, "usage: %s <folder> <output>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_folder = argv[2];

    struct stat st = {0};
    if (stat(output_folder, &st) == -1) {
        mkdir(output_folder, 0700);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    if (mesh.element_type != TRI3) {
        fprintf(stderr, "element_type must be TRI3\n");
        return EXIT_FAILURE;
    }

    // Optional params
    real_t SFEM_MU = 1;
    real_t SFEM_RHO = 1;
    int SFEM_PROBLEM_TYPE = 1;
    int SFEM_AOS = 0;
    const char *SFEM_DIRICHLET_NODES = 0;

    SFEM_READ_ENV(SFEM_PROBLEM_TYPE, atoi);
    SFEM_READ_ENV(SFEM_MU, atof);
    SFEM_READ_ENV(SFEM_RHO, atof);
    SFEM_READ_ENV(SFEM_AOS, atoi);
    SFEM_READ_ENV(SFEM_DIRICHLET_NODES, );

    if (rank == 0) {
        printf(
            "----------------------------------------\n"
            "Options:\n"
            "----------------------------------------\n"
            "- SFEM_PROBLEM_TYPE=%d\n"
            "- SFEM_MU=%g\n"
            "- SFEM_RHO=%g\n"
            "- SFEM_DIRICHLET_NODES=%s\n"
            "----------------------------------------\n",
            SFEM_PROBLEM_TYPE,
            SFEM_MU,
            SFEM_RHO,
            SFEM_DIRICHLET_NODES);
    }

    double tack = MPI_Wtime();
    printf("stokes.c: read\t\t%g seconds\n", tack - tick);

    ///////////////////////////////////////////////////////////////////////////////
    // Build CRS graph
    ///////////////////////////////////////////////////////////////////////////////

    ptrdiff_t nnz = 0;
    count_t *rowptr = 0;
    idx_t *colidx = 0;

    build_crs_graph_for_elem_type(
        mesh.element_type, mesh.nelements, mesh.nnodes, mesh.elements, &rowptr, &colidx);
    nnz = rowptr[mesh.nnodes];

    double tock = MPI_Wtime();
    printf("stokes.c: build crs graph\t\t%g seconds\n", tock - tack);
    tack = tock;

    const int sdim = elem_manifold_dim(mesh.element_type);
    const int n_vars = sdim + 1;

    if (SFEM_AOS) {
        real_t *values = calloc(n_vars * n_vars * nnz, sizeof(real_t));
        real_t *rhs = calloc(n_vars * mesh.nnodes, sizeof(real_t));

        ///////////////////////////////////////////////////////////////////////////////
        // Operator assembly
        ///////////////////////////////////////////////////////////////////////////////

        tri3_stokes_mini_assemble_hessian_aos(SFEM_MU,
                                              mesh.element_type,
                                              mesh.nelements,
                                              mesh.nnodes,
                                              mesh.elements,
                                              mesh.points,
                                              rowptr,
                                              colidx,
                                              values);

        tri3_stokes_mini_assemble_rhs_aos(SFEM_PROBLEM_TYPE,
                                          SFEM_MU,
                                          SFEM_RHO,
                                          mesh.element_type,
                                          mesh.nelements,
                                          mesh.nnodes,
                                          mesh.elements,
                                          mesh.points,
                                          rhs);

        count_t *b_rowptr = (count_t *)malloc((mesh.nnodes + 1) * n_vars * sizeof(count_t));
        idx_t *b_colidx = (idx_t *)malloc(rowptr[mesh.nnodes] * n_vars * n_vars * sizeof(idx_t));
        crs_graph_block_to_scalar(mesh.nnodes, n_vars, rowptr, colidx, b_rowptr, b_colidx);

        if (SFEM_DIRICHLET_NODES) {
            idx_t *dirichlet_nodes = 0;
            ptrdiff_t _nope_, nn;
            array_create_from_file(comm,
                                   SFEM_DIRICHLET_NODES,
                                   SFEM_MPI_IDX_T,
                                   (void **)&dirichlet_nodes,
                                   &_nope_,
                                   &nn);

            for (int d = 0; d < sdim; d++) {
                constraint_nodes_to_value_vec(nn, dirichlet_nodes, n_vars, d, 0, rhs);
            }

            for (int d1 = 0; d1 < sdim; d1++) {
                crs_constraint_nodes_to_identity_vec(
                    nn, dirichlet_nodes, n_vars, d1, 1, b_rowptr, b_colidx, values);
            }

            if (1)
            // if (0)
            {
                // One point to 0 to fix pressure degree of freedom
                // ptrdiff_t node = nn - 1;
                ptrdiff_t node = 0;
                crs_constraint_nodes_to_identity_vec(
                    1, &dirichlet_nodes[node], n_vars, (n_vars - 1), 1, b_rowptr, b_colidx, values);

                constraint_nodes_to_value_vec(
                    1, &dirichlet_nodes[node], n_vars, n_vars - 1, 0, rhs);
            }

        } else {
            assert(0);
        }

        {
            crs_t crs_out;
            crs_out.rowptr = (char *)b_rowptr;
            crs_out.colidx = (char *)b_colidx;
            crs_out.values = (char *)values;
            crs_out.grows = mesh.nnodes * n_vars;
            crs_out.lrows = mesh.nnodes * n_vars;
            crs_out.lnnz = b_rowptr[mesh.nnodes * n_vars];
            crs_out.gnnz = b_rowptr[mesh.nnodes * n_vars];
            crs_out.start = 0;
            crs_out.rowoffset = 0;
            crs_out.rowptr_type = SFEM_MPI_COUNT_T;
            crs_out.colidx_type = SFEM_MPI_IDX_T;
            crs_out.values_type = SFEM_MPI_REAL_T;

            crs_write_folder(comm, output_folder, &crs_out);
        }

        {
            char path[1024 * 10];
            // Write rhs vectors
            sprintf(path, "%s/rhs.raw", output_folder);
            array_write(
                comm, path, SFEM_MPI_REAL_T, rhs, mesh.nnodes * n_vars, mesh.nnodes * n_vars);
        }

        free(b_rowptr);
        free(b_colidx);
        free(values);
        free(rhs);
    } else {
        real_t **values = 0;
        values = (real_t **)malloc((n_vars * n_vars) * sizeof(real_t *));
        for (int d = 0; d < (n_vars * n_vars); d++) {
            values[d] = calloc(nnz, sizeof(real_t));
        }

        real_t **rhs = 0;
        rhs = (real_t **)malloc((n_vars) * sizeof(real_t *));
        for (int d = 0; d < n_vars; d++) {
            rhs[d] = calloc(mesh.nnodes, sizeof(real_t));
        }

        ///////////////////////////////////////////////////////////////////////////////
        // Operator assembly
        ///////////////////////////////////////////////////////////////////////////////

        tri3_stokes_mini_assemble_hessian_soa(SFEM_MU,
                                              mesh.element_type,
                                              mesh.nelements,
                                              mesh.nnodes,
                                              mesh.elements,
                                              mesh.points,
                                              rowptr,
                                              colidx,
                                              values);

        tri3_stokes_mini_assemble_rhs_soa(SFEM_PROBLEM_TYPE,
                                          SFEM_MU,
                                          SFEM_RHO,
                                          mesh.element_type,
                                          mesh.nelements,
                                          mesh.nnodes,
                                          mesh.elements,
                                          mesh.points,
                                          rhs);

        tock = MPI_Wtime();
        printf("stokes.c: assembly\t\t%g seconds\n", tock - tack);
        tack = tock;

        ///////////////////////////////////////////////////////////////////////////////
        // Boundary conditions
        ///////////////////////////////////////////////////////////////////////////////

        if (SFEM_DIRICHLET_NODES) {
            idx_t *dirichlet_nodes = 0;
            ptrdiff_t _nope_, nn;
            array_create_from_file(comm,
                                   SFEM_DIRICHLET_NODES,
                                   SFEM_MPI_IDX_T,
                                   (void **)&dirichlet_nodes,
                                   &_nope_,
                                   &nn);

            for (int d = 0; d < sdim; d++) {
                constraint_nodes_to_value(nn, dirichlet_nodes, 0, rhs[d]);
            }

            for (int d1 = 0; d1 < sdim; d1++) {
                for (int d2 = 0; d2 < n_vars; d2++) {
                    crs_constraint_nodes_to_identity(
                        nn, dirichlet_nodes, d1 == d2, rowptr, colidx, values[d1 * n_vars + d2]);
                }
            }

            if (1)
            // if (0)
            {
                // One point to 0 to fix pressure degree of freedom
                // ptrdiff_t node = nn - 1;
                ptrdiff_t node = 0;
                for (int d2 = 0; d2 < n_vars; d2++) {
                    crs_constraint_nodes_to_identity(1,
                                                     &dirichlet_nodes[node],
                                                     (n_vars - 1) == d2,
                                                     rowptr,
                                                     colidx,
                                                     values[(n_vars - 1) * n_vars + d2]);
                }

                constraint_nodes_to_value(1, &dirichlet_nodes[node], 0, rhs[n_vars - 1]);
            }

        } else {
            assert(0);
        }

        tock = MPI_Wtime();
        printf("stokes.c: boundary\t\t%g seconds\n", tock - tack);
        tack = tock;

        ///////////////////////////////////////////////////////////////////////////////
        // Write to disk
        ///////////////////////////////////////////////////////////////////////////////

        {
            // Write block CRS matrix
            block_crs_t crs_out;
            crs_out.rowptr = (char *)rowptr;
            crs_out.colidx = (char *)colidx;

            crs_out.block_size = n_vars * n_vars;
            crs_out.values = (char **)values;
            crs_out.grows = mesh.nnodes;
            crs_out.lrows = mesh.nnodes;
            crs_out.lnnz = nnz;
            crs_out.gnnz = nnz;
            crs_out.start = 0;
            crs_out.rowoffset = 0;
            crs_out.rowptr_type = SFEM_MPI_COUNT_T;
            crs_out.colidx_type = SFEM_MPI_IDX_T;
            crs_out.values_type = SFEM_MPI_REAL_T;

            char path_rowptr[1024 * 10];
            sprintf(path_rowptr, "%s/rowptr.raw", output_folder);

            char path_colidx[1024 * 10];
            sprintf(path_colidx, "%s/colidx.raw", output_folder);

            char format_values[1024 * 10];
            sprintf(format_values, "%s/values.%%d.raw", output_folder);
            block_crs_write(comm, path_rowptr, path_colidx, format_values, &crs_out);
        }

        {
            char path[1024 * 10];
            // Write rhs vectors
            for (int d = 0; d < n_vars; d++) {
                sprintf(path, "%s/rhs.%d.raw", output_folder, d);
                array_write(comm, path, SFEM_MPI_REAL_T, rhs[d], mesh.nnodes, mesh.nnodes);
            }
        }

        tock = MPI_Wtime();
        printf("stokes.c: write\t\t%g seconds\n", tock - tack);
        tack = tock;

        ///////////////////////////////////////////////////////////////////////////////
        // Free resources
        ///////////////////////////////////////////////////////////////////////////////

        for (int d = 0; d < (n_vars * n_vars); d++) {
            free(values[d]);
        }

        free(values);

        for (int d = 0; d < n_vars; d++) {
            free(rhs[d]);
        }

        free(rhs);
    }

    // Mesh n2n graph
    free(rowptr);
    free(colidx);

    ptrdiff_t nelements = mesh.nelements;
    ptrdiff_t nnodes = mesh.nnodes;

    mesh_destroy(&mesh);

    tock = MPI_Wtime();
    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #nz %ld\n", (long)nelements, (long)nnodes, (long)nnz);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
