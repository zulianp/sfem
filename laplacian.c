#include "laplacian.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

static SFEM_INLINE real_t det3(const real_t *mat) {
    return mat[0] * mat[4] * mat[8] + mat[1] * mat[5] * mat[6] + mat[2] * mat[3] * mat[7] - mat[0] * mat[5] * mat[7] -
           mat[1] * mat[3] * mat[8] - mat[2] * mat[4] * mat[6];
}

SFEM_INLINE void adjugate3(const real_t *mat, real_t *mat_adj) {
    mat_adj[0] = (mat[4] * mat[8] - mat[5] * mat[7]);
    mat_adj[1] = (mat[2] * mat[7] - mat[1] * mat[8]);
    mat_adj[2] = (mat[1] * mat[5] - mat[2] * mat[4]);
    mat_adj[3] = (mat[5] * mat[6] - mat[3] * mat[8]);
    mat_adj[4] = (mat[0] * mat[8] - mat[2] * mat[6]);
    mat_adj[5] = (mat[2] * mat[3] - mat[0] * mat[5]);
    mat_adj[6] = (mat[3] * mat[7] - mat[4] * mat[6]);
    mat_adj[7] = (mat[1] * mat[6] - mat[0] * mat[7]);
    mat_adj[8] = (mat[0] * mat[4] - mat[1] * mat[3]);
}

SFEM_INLINE void inverse3(const real_t *mat, real_t *mat_inv, const real_t det) {
    assert(det != 0.);
    adjugate3(mat, mat_inv);

    for (int i = 0; i < 9; ++i) {
        mat_inv[i] /= det;
    }
}

SFEM_INLINE void mtv3(const real_t A[3 * 3], const real_t v[3], real_t *out) {
    for (int i = 0; i < 3; ++i) {
        out[i] = 0;
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i] += A[i + j * 3] * v[j];
        }
    }
}

SFEM_INLINE real_t dot3(const real_t v1[3], const real_t v2[3]) {
    real_t ret = 0;
    for (int i = 0; i < 3; ++i) {
        ret += v1[i] * v2[i];
    }

    return ret;
}

void integrate(const real_t *inverse_jacobian, const real_t dV, real_t *element_matrix) {
    const real_t grad_ref[4][3] = {{-1, -1, -1}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    real_t grad_test[3];
    real_t grad_trial[3];

    for (int edof_i = 0; edof_i < 4; ++edof_i) {
        mtv3(inverse_jacobian, grad_ref[edof_i], grad_test);

        const real_t aii = dot3(grad_test, grad_test) * dV;

        element_matrix[edof_i * 4 + edof_i] = aii;

        for (int edof_j = edof_i + 1; edof_j < 4; ++edof_j) {
            mtv3(inverse_jacobian, grad_ref[edof_j], grad_trial);

            const real_t aij = dot3(grad_test, grad_trial) * dV;

            element_matrix[edof_i * 4 + edof_j] = aij;
            element_matrix[edof_i + edof_j * 4] = aij;
        }
    }
}

void print_element_matrix(const real_t *element_matrix) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%g ", element_matrix[i * 4 + j]);
        }
        printf("\n");
    }
    printf("\n");
}

static SFEM_INLINE void laplacian(const real_t x0,
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
    real_t x4 = z0 - z3;
    real_t x5 = x0 - x1;
    real_t x6 = y0 - y2;
    real_t x7 = x5 * x6;
    real_t x8 = z0 - z1;
    real_t x9 = x0 - x2;
    real_t x10 = y0 - y3;
    real_t x11 = x10 * x9;
    real_t x12 = z0 - z2;
    real_t x13 = x0 - x3;
    real_t x14 = y0 - y1;
    real_t x15 = x13 * x14;
    real_t x16 = x10 * x5;
    real_t x17 = x14 * x9;
    real_t x18 = x13 * x6;
    real_t x19 = x11 * x8 + x12 * x15 - x12 * x16 - x17 * x4 - x18 * x8 + x4 * x7;
    real_t x20 = 1.0 / x19;
    real_t x21 = x11 - x18;
    real_t x22 = -x17 + x7;
    real_t x23 = x15 - x16 + x21 + x22;
    real_t x24 = -x12 * x13 + x4 * x9;
    real_t x25 = x12 * x5 - x8 * x9;
    real_t x26 = x13 * x8;
    real_t x27 = x4 * x5;
    real_t x28 = x26 - x27;
    real_t x29 = -x24 - x25 - x28;
    real_t x30 = x10 * x8;
    real_t x31 = x14 * x4;
    real_t x32 = -x10 * x12 + x4 * x6;
    real_t x33 = x12 * x14 - x6 * x8;
    real_t x34 = x30 - x31 + x32 + x33;
    real_t x35 = -x12;
    real_t x36 = -x9;
    real_t x37 = x19 * (x13 * x35 + x28 - x35 * x5 - x36 * x4 + x36 * x8);
    real_t x38 = -x19;
    real_t x39 = -x23;
    real_t x40 = -x34;
    real_t x41 = (1.0 / 6.0) / pow(x19, 2);
    real_t x42 = x41 * (x24 * x37 + x38 * (x21 * x39 + x32 * x40));
    real_t x43 = -x15 + x16;
    real_t x44 = (1.0 / 6.0) * x43;
    real_t x45 = -x26 + x27;
    real_t x46 = -x30 + x31;
    real_t x47 = (1.0 / 6.0) * x46;
    real_t x48 = x20 * (-x23 * x44 + (1.0 / 6.0) * x29 * x45 - x34 * x47);
    real_t x49 = x41 * (x25 * x37 + x38 * (x22 * x39 + x33 * x40));
    real_t x50 = (1.0 / 6.0) * x45;
    real_t x51 = x20 * (x21 * x44 + x24 * x50 + x32 * x47);
    real_t x52 = x20 * (-1.0 / 6.0 * x21 * x22 - 1.0 / 6.0 * x24 * x25 - 1.0 / 6.0 * x32 * x33);
    real_t x53 = x20 * (x22 * x44 + x25 * x50 + x33 * x47);

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

void assemble_laplacian(const ptrdiff_t nelements,
                        const ptrdiff_t nnodes,
                        idx_t *const elems[4],
                        geom_t *const xyz[3],
                        idx_t *const rowptr,
                        idx_t *const colidx,
                        real_t *const values) {
    double tick = MPI_Wtime();

    real_t jacobian[3 * 3];
    real_t inverse_jacobian[3 * 3];
    real_t element_matrix[4 * 4];

    real_t grad_ref[4][3] = {{-1, -1, -1}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    real_t grad_test[3];
    real_t grad_trial[3];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
        if (1) {
            // Use code generated kernel

            // Element indices
            const idx_t i0 = elems[0][i];
            const idx_t i1 = elems[1][i];
            const idx_t i2 = elems[2][i];
            const idx_t i3 = elems[3][i];

            laplacian(
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

        } else {
            // Use handwritten kernel
            // Collect element coordinates
            for (int d1 = 0; d1 < 3; ++d1) {
                real_t x0 = (real_t)xyz[d1][elems[0][i]];

                for (int d2 = 0; d2 < 3; ++d2) {
                    real_t x1 = (real_t)xyz[d1][elems[d2 + 1][i]];
                    jacobian[d1 * 3 + d2] = x1 - x0;
                }
            }

            if (1) {
                real_t jacobian_determinant = det3(jacobian);
                inverse3(jacobian, inverse_jacobian, jacobian_determinant);

                assert(jacobian_determinant > 0.);

                const real_t dx = jacobian_determinant / 6.;
                integrate(inverse_jacobian, dx, element_matrix);
            } else {
                // 9 less operations
                real_t jacobian_determinant = det3(jacobian);
                adjugate3(jacobian, inverse_jacobian);

                assert(jacobian_determinant > 0.);

                const real_t dx = 1. / (jacobian_determinant * 6.);
                integrate(inverse_jacobian, dx, element_matrix);
            }
        }

        // Local to global
        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            idx_t dof_i = elems[edof_i][i];
            idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

            idx_t *row = &colidx[rowptr[dof_i]];
            real_t *rowvalues = &values[rowptr[dof_i]];

            for (int edof_j = 0; edof_j < 4; ++edof_j) {
                idx_t dof_j = elems[edof_j][i];
                int k = -1;

                if (lenrow <= 32) {
                    k = find_idx(dof_j, row, lenrow);
                } else {
                    // Use this for larger number of dofs per row
                    k = find_idx_binary_search(dof_j, row, lenrow);
                }

                rowvalues[k] += element_matrix[edof_i * 4 + edof_j];
            }
        }
    }

    double tock = MPI_Wtime();
    printf("laplacian.c: assemble_laplacian\t%g seconds\n", tock - tick);
}
