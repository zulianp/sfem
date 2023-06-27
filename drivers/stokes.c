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

ptrdiff_t read_file(MPI_Comm comm, const char *path, void **data) {
    MPI_Status status;
    MPI_Offset nbytes;
    MPI_File file;
    CATCH_MPI_ERROR(MPI_File_open(comm, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &file));
    CATCH_MPI_ERROR(MPI_File_get_size(file, &nbytes));
    *data = malloc(nbytes);

    CATCH_MPI_ERROR(MPI_File_read_at_all(file, 0, *data, nbytes, MPI_CHAR, &status));
    return nbytes;
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
    const real_t x1 = -x0;
    const real_t x2 = px0 - px2;
    const real_t x3 = py0 - py2;
    const real_t x4 = -x3;
    const real_t x5 = py0 - py1;
    const real_t x6 = x1 * x4 - x2 * x5;
    const real_t x7 = pow(x6, -2);
    const real_t x8 = x1 * x2 * x7;
    const real_t x9 = x4 * x5 * x7;
    const real_t x10 = pow(x2, 2) * x7;
    const real_t x11 = pow(x4, 2) * x7;
    const real_t x12 = x10 + x11;
    const real_t x13 = pow(x1, 2) * x7;
    const real_t x14 = pow(x5, 2) * x7;
    const real_t x15 = x13 + x14;
    const real_t x16 = x0 * x3 - x2 * x5;
    const real_t x17 = (1.0 / 2.0) * mu * x16;
    const real_t x18 = x17 * (x12 + x15 + 2 * x8 + 2 * x9);
    const real_t x19 = x8 + x9;
    const real_t x20 = x12 + x19;
    const real_t x21 = -x17 * x20;
    const real_t x22 = -x17 * (x15 + x19);
    const real_t x23 = 1.0 / x6;
    const real_t x24 = x23 * x4;
    const real_t x25 = x23 * x5;
    const real_t x26 = x24 + x25;
    const real_t x27 = (1.0 / 6.0) * x16;
    const real_t x28 = x26 * x27;
    const real_t x29 = x12 * x17;
    const real_t x30 = x17 * x19;
    const real_t x31 = -x24 * x27;
    const real_t x32 = x15 * x17;
    const real_t x33 = -x25 * x27;
    const real_t x34 = x1 * x23;
    const real_t x35 = x2 * x23;
    const real_t x36 = x34 + x35;
    const real_t x37 = x27 * x36;
    const real_t x38 = -x27 * x35;
    const real_t x39 = -x27 * x34;
    const real_t x40 = (1.0 / 80.0) * x16 / (mu * (x15 + x20));
    const real_t x41 = x36 * x40;
    const real_t x42 = x26 * x40;
    const real_t x43 = x24 * x42 + x35 * x41;
    const real_t x44 = x25 * x42 + x34 * x41;
    const real_t x45 = -x40 * x8 - x40 * x9;
    element_matrix[0] = x18;
    element_matrix[1] = x21;
    element_matrix[2] = x22;
    element_matrix[3] = 0;
    element_matrix[4] = 0;
    element_matrix[5] = 0;
    element_matrix[6] = x28;
    element_matrix[7] = x28;
    element_matrix[8] = x28;
    element_matrix[9] = x21;
    element_matrix[10] = x29;
    element_matrix[11] = x30;
    element_matrix[12] = 0;
    element_matrix[13] = 0;
    element_matrix[14] = 0;
    element_matrix[15] = x31;
    element_matrix[16] = x31;
    element_matrix[17] = x31;
    element_matrix[18] = x22;
    element_matrix[19] = x30;
    element_matrix[20] = x32;
    element_matrix[21] = 0;
    element_matrix[22] = 0;
    element_matrix[23] = 0;
    element_matrix[24] = x33;
    element_matrix[25] = x33;
    element_matrix[26] = x33;
    element_matrix[27] = 0;
    element_matrix[28] = 0;
    element_matrix[29] = 0;
    element_matrix[30] = x18;
    element_matrix[31] = x21;
    element_matrix[32] = x22;
    element_matrix[33] = x37;
    element_matrix[34] = x37;
    element_matrix[35] = x37;
    element_matrix[36] = 0;
    element_matrix[37] = 0;
    element_matrix[38] = 0;
    element_matrix[39] = x21;
    element_matrix[40] = x29;
    element_matrix[41] = x30;
    element_matrix[42] = x38;
    element_matrix[43] = x38;
    element_matrix[44] = x38;
    element_matrix[45] = 0;
    element_matrix[46] = 0;
    element_matrix[47] = 0;
    element_matrix[48] = x22;
    element_matrix[49] = x30;
    element_matrix[50] = x32;
    element_matrix[51] = x39;
    element_matrix[52] = x39;
    element_matrix[53] = x39;
    element_matrix[54] = x28;
    element_matrix[55] = x31;
    element_matrix[56] = x33;
    element_matrix[57] = x37;
    element_matrix[58] = x38;
    element_matrix[59] = x39;
    element_matrix[60] = -pow(x26, 2) * x40 - pow(x36, 2) * x40;
    element_matrix[61] = x43;
    element_matrix[62] = x44;
    element_matrix[63] = x28;
    element_matrix[64] = x31;
    element_matrix[65] = x33;
    element_matrix[66] = x37;
    element_matrix[67] = x38;
    element_matrix[68] = x39;
    element_matrix[69] = x43;
    element_matrix[70] = -x10 * x40 - x11 * x40;
    element_matrix[71] = x45;
    element_matrix[72] = x28;
    element_matrix[73] = x31;
    element_matrix[74] = x33;
    element_matrix[75] = x37;
    element_matrix[76] = x38;
    element_matrix[77] = x39;
    element_matrix[78] = x44;
    element_matrix[79] = x45;
    element_matrix[80] = -x13 * x40 - x14 * x40;
}

SFEM_INLINE void tri3_stokes_mini_condense_rhs_kernel(const real_t mu,
                                                      const real_t px0,
                                                      const real_t px1,
                                                      const real_t px2,
                                                      const real_t py0,
                                                      const real_t py1,
                                                      const real_t py2,
                                                      const real_t *const SFEM_RESTRICT rhs_bubble,
                                                      real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = px0 - px1;
    const real_t x1 = -x0;
    const real_t x2 = py0 - py2;
    const real_t x3 = -x2;
    const real_t x4 = px0 - px2;
    const real_t x5 = py0 - py1;
    const real_t x6 = x1 * x3 - x4 * x5;
    const real_t x7 = pow(x6, -2);
    const real_t x8 = pow(x1, 2) * x7 + pow(x5, 2) * x7;
    const real_t x9 = 1.0 / x6;
    const real_t x10 = x4 * x9;
    const real_t x11 = x3 * x9;
    const real_t x12 = x10 + x11;
    const real_t x13 = pow(x0 * x2 - x4 * x5, 2);
    const real_t x14 = x1 * x4 * x7 + x3 * x5 * x7;
    const real_t x15 = (6561.0 / 100.0) * pow(mu, 2) * x13;
    const real_t x16 = pow(x3, 2) * x7 + pow(x4, 2) * x7;
    const real_t x17 = 1.0 / (-pow(x14, 2) * x15 + x15 * x16 * x8);
    const real_t x18 = (729.0 / 400.0) * mu * x13 * x17;
    const real_t x19 = x1 * x9;
    const real_t x20 = x5 * x9;
    const real_t x21 = x18 * (x19 + x20);
    const real_t x22 = x14 * x18;

    element_vector[6] += rhs_bubble[0] * (-x12 * x18 * x8 + x14 * x21) +
                         rhs_bubble[1] * ((729.0 / 400.0) * mu * x12 * x13 * x14 * x17 - x16 * x21);
    element_vector[7] +=
        rhs_bubble[0] * (x11 * x18 * x8 - x20 * x22) +
        rhs_bubble[1] * ((729.0 / 400.0) * mu * x13 * x16 * x17 * x5 * x9 - x11 * x22);
    element_vector[8] +=
        rhs_bubble[0] * ((729.0 / 400.0) * mu * x13 * x17 * x4 * x8 * x9 - x19 * x22) +
        rhs_bubble[1] * (-x10 * x22 + x16 * x18 * x19);
}

SFEM_INLINE void tri3_stokes_mini_assemble_rhs_kernel(const real_t mu,
                                                      const real_t px0,
                                                      const real_t px1,
                                                      const real_t px2,
                                                      const real_t py0,
                                                      const real_t py1,
                                                      const real_t py2,
                                                      const real_t *const SFEM_RESTRICT u_rhs,
                                                      const real_t *const SFEM_RESTRICT p_rhs,
                                                      real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = px0 - px1;
    const real_t x1 = py0 - py2;
    const real_t x2 = x0 * x1;
    const real_t x3 = px0 - px2;
    const real_t x4 = py0 - py1;
    const real_t x5 = x3 * x4;
    const real_t x6 = (1.0 / 12.0) * x2 - 1.0 / 12.0 * x5;
    const real_t x7 = (1.0 / 24.0) * x2 - 1.0 / 24.0 * x5;
    const real_t x8 = u_rhs[2] * x7;
    const real_t x9 = (3.0 / 40.0) * x2 - 3.0 / 40.0 * x5;
    const real_t x10 = u_rhs[0] * x9;
    const real_t x11 = u_rhs[3] * x7 + x10;
    const real_t x12 = u_rhs[1] * x7;
    const real_t x13 = u_rhs[6] * x7;
    const real_t x14 = u_rhs[4] * x9;
    const real_t x15 = u_rhs[7] * x7 + x14;
    const real_t x16 = u_rhs[5] * x7;
    const real_t x17 = p_rhs[1] * x7;
    const real_t x18 = p_rhs[2] * x7;
    const real_t x19 = -x0;
    const real_t x20 = -x1;
    const real_t x21 = x19 * x20 - x3 * x4;
    const real_t x22 = 1.0 / x21;
    const real_t x23 = x19 * x22;
    const real_t x24 = x22 * x3;
    const real_t x25 = (81.0 / 560.0) * x2 - 81.0 / 560.0 * x5;
    const real_t x26 = pow(x21, -2);
    const real_t x27 =
        (1.0 / 18.0) / (mu * (pow(x19, 2) * x26 + x19 * x26 * x3 + pow(x20, 2) * x26 +
                              x20 * x26 * x4 + x26 * pow(x3, 2) + x26 * pow(x4, 2)));
    const real_t x28 = x27 * (u_rhs[4] * x25 + u_rhs[5] * x9 + u_rhs[6] * x9 + u_rhs[7] * x9);
    const real_t x29 = x20 * x22;
    const real_t x30 = x22 * x4;
    const real_t x31 = x27 * (u_rhs[0] * x25 + u_rhs[1] * x9 + u_rhs[2] * x9 + u_rhs[3] * x9);
    const real_t x32 = p_rhs[0] * x7;
    element_vector[0] = u_rhs[1] * x6 + x11 + x8;
    element_vector[1] = u_rhs[2] * x6 + x11 + x12;
    element_vector[2] = u_rhs[3] * x6 + x10 + x12 + x8;
    element_vector[3] = u_rhs[5] * x6 + x13 + x15;
    element_vector[4] = u_rhs[6] * x6 + x15 + x16;
    element_vector[5] = u_rhs[7] * x6 + x13 + x14 + x16;
    element_vector[6] = p_rhs[0] * x6 + x17 + x18 - x28 * (x23 + x24) - x31 * (x29 + x30);
    element_vector[7] = p_rhs[1] * x6 + x18 + x24 * x28 + x29 * x31 + x32;
    element_vector[8] = p_rhs[2] * x6 + x17 + x23 * x28 + x30 * x31 + x32;
}

void tri3_stokes_mini_assemble_hessian(const real_t mu,
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
    static const int ndofs = 3;
    static const int rows = 9;
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
        for (int edof_i = 0; edof_i < ndofs; ++edof_i) {
            const idx_t dof_i = elems[edof_i][i];
            const idx_t r_begin = rowptr[dof_i];
            const idx_t lenrow = rowptr[dof_i + 1] - r_begin;
            const idx_t *row = &colidx[rowptr[dof_i]];
            find_cols3(ev, row, lenrow, ks[edof_i]);
        }

        for (int bi = 0; bi < n_vars; ++bi) {
            for (int bj = 0; bj < n_vars; ++bj) {
                for (int edof_i = 0; edof_i < ndofs; ++edof_i) {
                    const int ii = bi * ndofs + edof_i;

                    const idx_t dof_i = elems[edof_i][i];
                    const idx_t r_begin = rowptr[dof_i];
                    const int bb = bi * n_vars + bj;

                    real_t *const row_values = &values[bb][r_begin];

                    for (int edof_j = 0; edof_j < ndofs; ++edof_j) {
                        const int jj = bj * ndofs + edof_j;
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

    f[1] = -mu * (4 * x * (1 - x) * ((1 - 2 * x) * ((1 - 2 * y) * (1 - 2 * y) - 2 * y * 1 - y)) +
                  12 * y * y * (1 - y) * (1 - y) * (2 * x - 1)) -
           x * (1 - x);
}

void tri3_stokes_mini_assemble_rhs(const int tp_num,
                                   const real_t mu,
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

        for (int ii = 0; ii < 4; ii++) {
            switch (tp_num) {
                case 1: {
                    rhs1(mu, xx[ii], yy[ii], fb);
                    break;
                }
                default: {
                    assert(0);
                    break;
                }
            }

            u_rhs[0 * 3 + ii] = fb[0];
            u_rhs[1 * 3 + ii] = fb[1];
        }

        tri3_stokes_mini_assemble_rhs_kernel(
            mu, x0, x1, x2, y0, y1, y2, u_rhs, p_rhs, element_vector);

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
    int SFEM_PROBLEM_TYPE = 1;
    const char *SFEM_DIRICHLET_NODES = 0;

    SFEM_READ_ENV(SFEM_PROBLEM_TYPE, atoi);
    SFEM_READ_ENV(SFEM_MU, atof);
    SFEM_READ_ENV(SFEM_DIRICHLET_NODES, );

    if (rank == 0) {
        printf(
            "----------------------------------------\n"
            "Options:\n"
            "----------------------------------------\n"
            "- SFEM_PROBLEM_TYPE=%d\n"
            "- SFEM_MU=%g\n"
            "- SFEM_DIRICHLET_NODES=%s\n"
            "----------------------------------------\n",
            SFEM_PROBLEM_TYPE,
            SFEM_MU,
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
    real_t **values = 0;

    build_crs_graph_for_elem_type(
        mesh.element_type, mesh.nelements, mesh.nnodes, mesh.elements, &rowptr, &colidx);

    nnz = rowptr[mesh.nnodes];

    static const int n_vars = 3;
    values = (real_t **)malloc((n_vars * n_vars) * sizeof(real_t *));
    for (int d = 0; d < (n_vars * n_vars); d++) {
        values[d] = calloc(nnz, sizeof(real_t));
    }

    real_t **rhs = 0;
    rhs = (real_t **)malloc((n_vars) * sizeof(real_t *));
    for (int d = 0; d < n_vars; d++) {
        rhs[d] = calloc(mesh.nnodes, sizeof(real_t));
    }

    double tock = MPI_Wtime();
    printf("stokes.c: build crs\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Operator assembly
    ///////////////////////////////////////////////////////////////////////////////

    tri3_stokes_mini_assemble_hessian(SFEM_MU,
                                      mesh.element_type,
                                      mesh.nelements,
                                      mesh.nnodes,
                                      mesh.elements,
                                      mesh.points,
                                      rowptr,
                                      colidx,
                                      values);

    tri3_stokes_mini_assemble_rhs(SFEM_PROBLEM_TYPE,
                                  SFEM_MU,
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
        ptrdiff_t nn = read_file(comm, SFEM_DIRICHLET_NODES, (void **)&dirichlet_nodes);
        assert((nn / sizeof(idx_t)) * sizeof(idx_t) == nn);
        nn /= sizeof(idx_t);

        for (int d = 0; d < mesh.spatial_dim; d++) {
            constraint_nodes_to_value(nn, dirichlet_nodes, 0, rhs[d]);
        }

        for (int d1 = 0; d1 < mesh.spatial_dim; d1++) {
            for (int d2 = 0; d2 < n_vars; d2++) {
                crs_constraint_nodes_to_identity(
                    nn, dirichlet_nodes, d1 == d2, rowptr, colidx, values[d1 * n_vars + d2]);
            }
        }

        if (1) 
        // if (0) 
        {
            // One point to 0
            for (int d2 = 0; d2 < n_vars; d2++) {
                crs_constraint_nodes_to_identity(nn,
                                                 dirichlet_nodes,
                                                 (n_vars - 1) == d2,
                                                 rowptr,
                                                 colidx,
                                                 values[(n_vars - 1) * n_vars + d2]);
            }

            constraint_nodes_to_value(nn, dirichlet_nodes, 0, rhs[n_vars - 1]);
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

    free(rowptr);
    free(colidx);

    for (int d = 0; d < (n_vars * n_vars); d++) {
        free(values[d]);
    }

    free(values);

    for (int d = 0; d < n_vars; d++) {
        free(rhs[d]);
    }

    free(rhs);

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
