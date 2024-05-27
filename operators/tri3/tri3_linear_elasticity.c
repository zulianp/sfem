#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"

static const int stride = 1;

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

static int check_symmetric(int n, const real_t *const element_matrix) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const real_t diff = element_matrix[i * n + j] - element_matrix[i + j * n];
            assert(diff < 1e-16);
            if (diff > 1e-16) {
                return 1;
            }

            // printf("%g ",  element_matrix[i*n + j] );
        }

        // printf("\n");
    }

    // printf("\n");

    return 0;
}

static SFEM_INLINE void tri3_linear_elasticity_assemble_value_kernel(
    const real_t mu,
    const real_t lambda,
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t *const SFEM_RESTRICT u,
    real_t *const SFEM_RESTRICT element_scalar) {
    const real_t x0 = -px0 + px1;
    const real_t x1 = -py0 + py2;
    const real_t x2 = px0 - px2;
    const real_t x3 = py0 - py1;
    const real_t x4 = x0 * x1 - x2 * x3;
    const real_t x5 = 1.0 / x4;
    const real_t x6 = u[0] * x2 * x5 + u[3] * x0 * x5;
    const real_t x7 = u[2] * x2 * x5 + u[5] * x0 * x5 - x6;
    const real_t x8 = pow(x7, 2);
    const real_t x9 = (1.0 / 4.0) * lambda;
    const real_t x10 = u[0] * x1 * x5 + u[3] * x3 * x5;
    const real_t x11 = u[1] * x1 * x5 + u[4] * x3 * x5 - x10;
    const real_t x12 = pow(x11, 2);
    const real_t x13 = u[1] * x2 * x5 + u[4] * x0 * x5 - x6;
    const real_t x14 = (1.0 / 4.0) * mu;
    const real_t x15 = (1.0 / 2.0) * mu;
    const real_t x16 = u[2] * x1 * x5 + u[5] * x3 * x5 - x10;
    element_scalar[0] =
        x4 * ((1.0 / 2.0) * lambda * x11 * x7 + x12 * x15 + x12 * x9 + pow(x13, 2) * x14 +
              x13 * x15 * x16 + x14 * pow(x16, 2) + x15 * x8 + x8 * x9);
}

void tri3_linear_elasticity_assemble_value_soa(const ptrdiff_t nelements,
                                               const ptrdiff_t nnodes,
                                               idx_t **const SFEM_RESTRICT elems,
                                               geom_t **const SFEM_RESTRICT xyz,
                                               const real_t mu,
                                               const real_t lambda,
                                               const real_t **const SFEM_RESTRICT u,
                                               real_t *const SFEM_RESTRICT value) {
    SFEM_UNUSED(nnodes);

    const double tick = MPI_Wtime();

    idx_t ev[3];
    idx_t ks[3][3];

    real_t element_displacement[(3 * 2)];

    static const int block_size = 2;

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elems[v][i];
        }

        for (int b = 0; b < block_size; b++) {
            for (int v = 0; v < 3; ++v) {
                element_displacement[b * 3 + v] = u[b][ev[v]];
            }
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        real_t element_scalar = 0;
        tri3_linear_elasticity_assemble_value_kernel(
            // Model parameters
            mu,
            lambda,
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
            element_displacement,
            // output matrix
            &element_scalar);

        *value += element_scalar;
    }

    const double tock = MPI_Wtime();
    printf("tri3_linear_elasticity.c: tri3_linear_elasticity_assemble_gradient_soa\t%g seconds\n",
           tock - tick);
}

static SFEM_INLINE void tri3_linear_elasticity_assemble_gradient_kernel(
    const real_t mu,
    const real_t lambda,
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t *const SFEM_RESTRICT u,
    real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = -px0 + px1;
    const real_t x1 = -py0 + py2;
    const real_t x2 = px0 - px2;
    const real_t x3 = py0 - py1;
    const real_t x4 = x0 * x1 - x2 * x3;
    const real_t x5 = 1.0 / x4;
    const real_t x6 = x1 * x5;
    const real_t x7 = x3 * x5;
    const real_t x8 = -x6 - x7;
    const real_t x9 = u[0] * x8 + u[1] * x6 + u[2] * x7;
    const real_t x10 = mu * x9;
    const real_t x11 = (1.0 / 2.0) * lambda;
    const real_t x12 = x11 * x6;
    const real_t x13 = x2 * x5;
    const real_t x14 = x0 * x5;
    const real_t x15 = -x13 - x14;
    const real_t x16 = u[3] * x15 + u[4] * x13 + u[5] * x14;
    const real_t x17 = u[0] * x15 + u[1] * x13 + u[2] * x14;
    const real_t x18 = (1.0 / 2.0) * mu;
    const real_t x19 = x13 * x18;
    const real_t x20 = u[3] * x8 + u[4] * x6 + u[5] * x7;
    const real_t x21 = x10 * x6 + x12 * x16 + x12 * x9 + x17 * x19 + x19 * x20;
    const real_t x22 = x11 * x7;
    const real_t x23 = x14 * x18;
    const real_t x24 = x10 * x7 + x16 * x22 + x17 * x23 + x20 * x23 + x22 * x9;
    const real_t x25 = mu * x16;
    const real_t x26 = x11 * x13;
    const real_t x27 = x18 * x6;
    const real_t x28 = x13 * x25 + x16 * x26 + x17 * x27 + x20 * x27 + x26 * x9;
    const real_t x29 = x11 * x14;
    const real_t x30 = x18 * x7;
    const real_t x31 = x14 * x25 + x16 * x29 + x17 * x30 + x20 * x30 + x29 * x9;
    element_vector[0 * stride] = x4 * (-x21 - x24);
    element_vector[1 * stride] = x21 * x4;
    element_vector[2 * stride] = x24 * x4;
    element_vector[3 * stride] = x4 * (-x28 - x31);
    element_vector[4 * stride] = x28 * x4;
    element_vector[5 * stride] = x31 * x4;
}

void tri3_linear_elasticity_assemble_gradient_soa(const ptrdiff_t nelements,
                                                  const ptrdiff_t nnodes,
                                                  idx_t **const SFEM_RESTRICT elems,
                                                  geom_t **const SFEM_RESTRICT xyz,
                                                  const real_t mu,
                                                  const real_t lambda,
                                                  const real_t **const SFEM_RESTRICT u,
                                                  real_t **const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const double tick = MPI_Wtime();

    idx_t ev[3];
    idx_t ks[3][3];

    real_t element_vector[(3 * 2)];
    real_t element_displacement[(3 * 2)];

    static const int block_size = 2;

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elems[v][i];
        }

        for (int b = 0; b < block_size; b++) {
            for (int v = 0; v < 3; ++v) {
                element_displacement[b * 3 + v] = u[b][ev[v]];
            }
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        tri3_linear_elasticity_assemble_gradient_kernel(
            // Model parameters
            mu,
            lambda,
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
            element_displacement,
            // output matrix
            element_vector);

        for (int bi = 0; bi < block_size; ++bi) {
            for (int edof_i = 0; edof_i < 3; edof_i++) {
                values[bi][ev[edof_i]] += element_vector[bi * 3 + edof_i];
            }
        }
    }

    const double tock = MPI_Wtime();
    printf("tri3_linear_elasticity.c: tri3_linear_elasticity_assemble_gradient_soa\t%g seconds\n",
           tock - tick);
}

static SFEM_INLINE void tri3_linear_elasticity_assemble_hessian_kernel(const real_t mu,
                                                                       const real_t lambda,
                                                                       const real_t px0,
                                                                       const real_t px1,
                                                                       const real_t px2,
                                                                       const real_t py0,
                                                                       const real_t py1,
                                                                       const real_t py2,
                                                                       real_t *const SFEM_RESTRICT
                                                                           element_matrix) {
    const real_t x0 = -px0 + px1;
    const real_t x1 = -py0 + py2;
    const real_t x2 = x0 * x1;
    const real_t x3 = px0 - px2;
    const real_t x4 = py0 - py1;
    const real_t x5 = x2 - x3 * x4;
    const real_t x6 = pow(x5, -2);
    const real_t x7 = mu * x6;
    const real_t x8 = x0 * x3;
    const real_t x9 = x7 * x8;
    const real_t x10 = lambda * x6;
    const real_t x11 = x1 * x4;
    const real_t x12 = x11 * x7;
    const real_t x13 = pow(x1, 2);
    const real_t x14 = x13 * x7;
    const real_t x15 = (1.0 / 2.0) * x10;
    const real_t x16 = pow(x3, 2);
    const real_t x17 = x16 * x7;
    const real_t x18 = x13 * x15 + x14 + (1.0 / 2.0) * x17;
    const real_t x19 = pow(x4, 2);
    const real_t x20 = x19 * x7;
    const real_t x21 = pow(x0, 2);
    const real_t x22 = x21 * x7;
    const real_t x23 = x15 * x19 + x20 + (1.0 / 2.0) * x22;
    const real_t x24 = x11 * x15 + x12 + (1.0 / 2.0) * x9;
    const real_t x25 = x5 * (-x18 - x24);
    const real_t x26 = x5 * (-x23 - x24);
    const real_t x27 = x15 * x3;
    const real_t x28 = (1.0 / 2.0) * x7;
    const real_t x29 = x28 * x3;
    const real_t x30 = x1 * x27 + x1 * x29;
    const real_t x31 = x2 * x28 + x27 * x4;
    const real_t x32 = x30 + x31;
    const real_t x33 = x15 * x2 + x29 * x4;
    const real_t x34 = x0 * x4;
    const real_t x35 = x15 * x34 + x28 * x34;
    const real_t x36 = x33 + x35;
    const real_t x37 = x5 * (x32 + x36);
    const real_t x38 = -x32 * x5;
    const real_t x39 = -x36 * x5;
    const real_t x40 = x24 * x5;
    const real_t x41 = x5 * (-x30 - x33);
    const real_t x42 = x30 * x5;
    const real_t x43 = x33 * x5;
    const real_t x44 = x5 * (-x31 - x35);
    const real_t x45 = x31 * x5;
    const real_t x46 = x35 * x5;
    const real_t x47 = (1.0 / 2.0) * x14 + x15 * x16 + x17;
    const real_t x48 = x15 * x21 + (1.0 / 2.0) * x20 + x22;
    const real_t x49 = (1.0 / 2.0) * x12 + x15 * x8 + x9;
    const real_t x50 = x5 * (-x47 - x49);
    const real_t x51 = x5 * (-x48 - x49);
    const real_t x52 = x49 * x5;
    element_matrix[0 * stride] = x5 * (x10 * x11 + 2 * x12 + x18 + x23 + x9);
    element_matrix[1 * stride] = x25;
    element_matrix[2 * stride] = x26;
    element_matrix[3 * stride] = x37;
    element_matrix[4 * stride] = x38;
    element_matrix[5 * stride] = x39;
    element_matrix[6 * stride] = x25;
    element_matrix[7 * stride] = x18 * x5;
    element_matrix[8 * stride] = x40;
    element_matrix[9 * stride] = x41;
    element_matrix[10 * stride] = x42;
    element_matrix[11 * stride] = x43;
    element_matrix[12 * stride] = x26;
    element_matrix[13 * stride] = x40;
    element_matrix[14 * stride] = x23 * x5;
    element_matrix[15 * stride] = x44;
    element_matrix[16 * stride] = x45;
    element_matrix[17 * stride] = x46;
    element_matrix[18 * stride] = x37;
    element_matrix[19 * stride] = x41;
    element_matrix[20 * stride] = x44;
    element_matrix[21 * stride] = x5 * (x10 * x8 + x12 + x47 + x48 + 2 * x9);
    element_matrix[22 * stride] = x50;
    element_matrix[23 * stride] = x51;
    element_matrix[24 * stride] = x38;
    element_matrix[25 * stride] = x42;
    element_matrix[26 * stride] = x45;
    element_matrix[27 * stride] = x50;
    element_matrix[28 * stride] = x47 * x5;
    element_matrix[29 * stride] = x52;
    element_matrix[30 * stride] = x39;
    element_matrix[31 * stride] = x43;
    element_matrix[32 * stride] = x46;
    element_matrix[33 * stride] = x51;
    element_matrix[34 * stride] = x52;
    element_matrix[35 * stride] = x48 * x5;
}

void tri3_linear_elasticity_assemble_hessian_soa(const ptrdiff_t nelements,
                                                 const ptrdiff_t nnodes,
                                                 idx_t **const SFEM_RESTRICT elems,
                                                 geom_t **const SFEM_RESTRICT xyz,
                                                 const real_t mu,
                                                 const real_t lambda,
                                                 const count_t *const SFEM_RESTRICT rowptr,
                                                 const idx_t *const SFEM_RESTRICT colidx,
                                                 real_t **const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const double tick = MPI_Wtime();

    idx_t ev[3];
    idx_t ks[3][3];

    real_t element_matrix[(3 * 2) * (3 * 2)];

    static const int block_size = 2;
    static const int mat_block_size = block_size * block_size;

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        tri3_linear_elasticity_assemble_hessian_kernel(
            // Model parameters
            mu,
            lambda,
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
            // output matrix
            element_matrix);

        assert(!check_symmetric(3 * block_size, element_matrix));

        // find all indices
        for (int edof_i = 0; edof_i < 3; ++edof_i) {
            const idx_t dof_i = elems[edof_i][i];
            const idx_t r_begin = rowptr[dof_i];
            const idx_t lenrow = rowptr[dof_i + 1] - r_begin;
            const idx_t *row = &colidx[rowptr[dof_i]];
            find_cols3(ev, row, lenrow, ks[edof_i]);
        }

        for (int bi = 0; bi < block_size; ++bi) {
            for (int bj = 0; bj < block_size; ++bj) {
                for (int edof_i = 0; edof_i < 3; ++edof_i) {
                    const int ii = bi * 3 + edof_i;

                    const idx_t dof_i = elems[edof_i][i];
                    const idx_t r_begin = rowptr[dof_i];
                    const int bb = bi * block_size + bj;

                    real_t *const row_values = &values[bb][r_begin];

                    for (int edof_j = 0; edof_j < 3; ++edof_j) {
                        const int jj = bj * 3 + edof_j;
                        const real_t val = element_matrix[ii * 6 + jj];

                        assert(val == val);
                        row_values[ks[edof_i][edof_j]] += val;
                    }
                }
            }
        }
    }

    const double tock = MPI_Wtime();
    printf("tri3_linear_elasticity.c: tri3_linear_elasticity_assemble_hessian_soa\t%g seconds\n",
           tock - tick);
}

static SFEM_INLINE void tri3_linear_elasticity_apply_kernel(
    const real_t mu,
    const real_t lambda,
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t *const SFEM_RESTRICT increment,
    real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = -py0 + py2;
    const real_t x1 = px0 - px2;
    const real_t x2 = -px0 + px1;
    const real_t x3 = x0 * x2;
    const real_t x4 = py0 - py1;
    const real_t x5 = -x1 * x4 + x3;
    const real_t x6 = pow(x5, -2);
    const real_t x7 = lambda * x6;
    const real_t x8 = (1.0 / 2.0) * x7;
    const real_t x9 = x1 * x8;
    const real_t x10 = mu * x6;
    const real_t x11 = (1.0 / 2.0) * x10;
    const real_t x12 = x1 * x11;
    const real_t x13 = x0 * x12 + x0 * x9;
    const real_t x14 = x11 * x3 + x4 * x9;
    const real_t x15 = x13 + x14;
    const real_t x16 = -x15;
    const real_t x17 = increment[4] * x5;
    const real_t x18 = x12 * x4 + x3 * x8;
    const real_t x19 = x2 * x4;
    const real_t x20 = x11 * x19 + x19 * x8;
    const real_t x21 = x18 + x20;
    const real_t x22 = -x21;
    const real_t x23 = increment[5] * x5;
    const real_t x24 = pow(x0, 2);
    const real_t x25 = x10 * x24;
    const real_t x26 = pow(x1, 2);
    const real_t x27 = x10 * x26;
    const real_t x28 = x24 * x8 + x25 + (1.0 / 2.0) * x27;
    const real_t x29 = x0 * x4;
    const real_t x30 = x10 * x29;
    const real_t x31 = x1 * x2;
    const real_t x32 = x10 * x31;
    const real_t x33 = x29 * x8 + x30 + (1.0 / 2.0) * x32;
    const real_t x34 = -x28 - x33;
    const real_t x35 = increment[1] * x5;
    const real_t x36 = pow(x4, 2);
    const real_t x37 = x10 * x36;
    const real_t x38 = pow(x2, 2);
    const real_t x39 = x10 * x38;
    const real_t x40 = x36 * x8 + x37 + (1.0 / 2.0) * x39;
    const real_t x41 = -x33 - x40;
    const real_t x42 = increment[2] * x5;
    const real_t x43 = x15 + x21;
    const real_t x44 = increment[3] * x5;
    const real_t x45 = increment[0] * x5;
    const real_t x46 = -x13 - x18;
    const real_t x47 = -x14 - x20;
    const real_t x48 = (1.0 / 2.0) * x25 + x26 * x8 + x27;
    const real_t x49 = (1.0 / 2.0) * x30 + x31 * x8 + x32;
    const real_t x50 = -x48 - x49;
    const real_t x51 = (1.0 / 2.0) * x37 + x38 * x8 + x39;
    const real_t x52 = -x49 - x51;
    element_vector[0 * stride] = x16 * x17 + x22 * x23 + x34 * x35 + x41 * x42 + x43 * x44 +
                                 x45 * (x28 + x29 * x7 + 2 * x30 + x32 + x40);
    element_vector[1 * stride] =
        x13 * x17 + x18 * x23 + x28 * x35 + x33 * x42 + x34 * x45 + x44 * x46;
    element_vector[2 * stride] =
        x14 * x17 + x20 * x23 + x33 * x35 + x40 * x42 + x41 * x45 + x44 * x47;
    element_vector[3 * stride] = x17 * x50 + x23 * x52 + x35 * x46 + x42 * x47 + x43 * x45 +
                                 x44 * (x30 + x31 * x7 + 2 * x32 + x48 + x51);
    element_vector[4 * stride] =
        x13 * x35 + x14 * x42 + x16 * x45 + x17 * x48 + x23 * x49 + x44 * x50;
    element_vector[5 * stride] =
        x17 * x49 + x18 * x35 + x20 * x42 + x22 * x45 + x23 * x51 + x44 * x52;
}

void tri3_linear_elasticity_apply_soa(const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const SFEM_RESTRICT elems,
                                      geom_t **const SFEM_RESTRICT xyz,
                                      const real_t mu,
                                      const real_t lambda,
                                      const real_t **const SFEM_RESTRICT u,
                                      real_t **const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const double tick = MPI_Wtime();

    idx_t ev[3];
    real_t element_vector[(3 * 2)];
    real_t element_displacement[(3 * 2)];

    static const int block_size = 2;

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elems[v][i];
        }

        for (int b = 0; b < block_size; b++) {
            for (int v = 0; v < 3; ++v) {
                element_displacement[b * 3 + v] = u[b][ev[v]];
            }
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        tri3_linear_elasticity_apply_kernel(
            // Model parameters
            mu,
            lambda,
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
            element_displacement,
            // output matrix
            element_vector);

        for (int bi = 0; bi < block_size; ++bi) {
            for (int edof_i = 0; edof_i < 3; edof_i++) {
                values[bi][ev[edof_i]] += element_vector[bi * 3 + edof_i];
            }
        }
    }

    const double tock = MPI_Wtime();
    printf("tri3_linear_elasticity.c: tri3_linear_elasticity_assemble_apply_soa\t%g seconds\n",
           tock - tick);
}

void tri3_linear_elasticity_assemble_value_aos(const ptrdiff_t nelements,
                                               const ptrdiff_t nnodes,
                                               idx_t **const SFEM_RESTRICT elems,
                                               geom_t **const SFEM_RESTRICT xyz,
                                               const real_t mu,
                                               const real_t lambda,
                                               const real_t *const SFEM_RESTRICT displacement,
                                               real_t *const SFEM_RESTRICT value) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    static const int block_size = 2;

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[3];
            idx_t ks[3];

            real_t element_displacement[(3 * 2)];

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];

            for (int enode = 0; enode < 3; ++enode) {
                idx_t dof = ev[enode] * block_size;

                for (int b = 0; b < block_size; ++b) {
                    element_displacement[b * 3 + enode] = displacement[dof + b];
                }
            }

            real_t element_scalar = 0;
            tri3_linear_elasticity_assemble_value_kernel(  // Model parameters
                mu,
                lambda,
                // X-coordinates
                xyz[0][i0],
                xyz[0][i1],
                xyz[0][i2],
                // Y-coordinates
                xyz[1][i0],
                xyz[1][i1],
                xyz[1][i2],
                element_displacement,
                // output vector
                &element_scalar);

#pragma omp atomic update
            *value += element_scalar;
        }
    }

    double tock = MPI_Wtime();
    printf("tri3_linear_elasticity.c: tri3_linear_elasticity_assemble_value\t%g seconds\n",
           tock - tick);
}

void tri3_linear_elasticity_assemble_gradient_aos(const ptrdiff_t nelements,
                                                  const ptrdiff_t nnodes,
                                                  idx_t **const SFEM_RESTRICT elems,
                                                  geom_t **const SFEM_RESTRICT xyz,
                                                  const real_t mu,
                                                  const real_t lambda,
                                                  const real_t *const SFEM_RESTRICT displacement,
                                                  real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[3];
    idx_t ks[3];

    real_t element_vector[(3 * 2)];
    real_t element_displacement[(3 * 2)];

    static const int block_size = 2;

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[3];
            idx_t ks[3];

            real_t element_vector[(3 * 2)];
            real_t element_displacement[(3 * 2)];

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];

            for (int enode = 0; enode < 3; ++enode) {
                idx_t dof = ev[enode] * block_size;

                for (int b = 0; b < block_size; ++b) {
                    element_displacement[b * 3 + enode] = displacement[dof + b];
                }
            }

            tri3_linear_elasticity_assemble_gradient_kernel(  // Model parameters
                mu,
                lambda,
                // X-coordinates
                xyz[0][i0],
                xyz[0][i1],
                xyz[0][i2],
                // Y-coordinates
                xyz[1][i0],
                xyz[1][i1],
                xyz[1][i2],
                element_displacement,
                // output vector
                element_vector);

            for (int edof_i = 0; edof_i < 3; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];

                for (int b = 0; b < block_size; b++) {
#pragma omp atomic update
                    values[dof_i * block_size + b] += element_vector[b * 3 + edof_i];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tri3_linear_elasticity.c: tri3_linear_elasticity_assemble_gradient\t%g seconds\n",
           tock - tick);
}

void tri3_linear_elasticity_assemble_hessian_aos(const ptrdiff_t nelements,
                                                 const ptrdiff_t nnodes,
                                                 idx_t **const SFEM_RESTRICT elems,
                                                 geom_t **const SFEM_RESTRICT xyz,
                                                 const real_t mu,
                                                 const real_t lambda,
                                                 const count_t *const SFEM_RESTRICT rowptr,
                                                 const idx_t *const SFEM_RESTRICT colidx,
                                                 real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const double tick = MPI_Wtime();

    static const int block_size = 2;
    static const int mat_block_size = block_size * block_size;

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[3];
            idx_t ks[3];

            real_t element_matrix[(3 * 2) * (3 * 2)];

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];

            tri3_linear_elasticity_assemble_hessian_kernel(
                // Model parameters
                mu,
                lambda,
                // X-coordinates
                xyz[0][i0],
                xyz[0][i1],
                xyz[0][i2],
                // Y-coordinates
                xyz[1][i0],
                xyz[1][i1],
                xyz[1][i2],
                // output matrix
                element_matrix);

            assert(!check_symmetric(3 * block_size, element_matrix));

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
                            const real_t val = element_matrix[ii * 6 + jj];
#pragma omp atomic update
                            row[offset_j + bj] += val;
                        }
                    }
                }
            }
        }
    }
    const double tock = MPI_Wtime();
    printf("tri3_linear_elasticity.c: tri3_linear_elasticity_assemble_hessian_aos\t%g seconds\n",
           tock - tick);
}

void tri3_linear_elasticity_apply_aos(const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const SFEM_RESTRICT elems,
                                      geom_t **const SFEM_RESTRICT xyz,
                                      const real_t mu,
                                      const real_t lambda,
                                      const real_t *const SFEM_RESTRICT displacement,
                                      real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    // double tick = MPI_Wtime();
    static const int block_size = 2;

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[3];
            idx_t ks[3];

            real_t element_vector[(3 * 2)];
            real_t element_displacement[(3 * 2)];

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];

            for (int enode = 0; enode < 3; ++enode) {
                idx_t dof = ev[enode] * block_size;

                for (int b = 0; b < block_size; ++b) {
                    element_displacement[b * 3 + enode] = displacement[dof + b];
                }
            }

            tri3_linear_elasticity_apply_kernel(  // Model parameters
                mu,
                lambda,
                // X-coordinates
                xyz[0][i0],
                xyz[0][i1],
                xyz[0][i2],
                // Y-coordinates
                xyz[1][i0],
                xyz[1][i1],
                xyz[1][i2],
                element_displacement,
                // output vector
                element_vector);

            for (int edof_i = 0; edof_i < 3; ++edof_i) {
                const idx_t dof = ev[edof_i] * block_size;

                for (int b = 0; b < block_size; b++) {
#pragma omp atomic update
                    values[dof + b] += element_vector[b * 3 + edof_i];
                }
            }
        }
    }

    // double tock = MPI_Wtime();
    // printf("tri3_linear_elasticity.c: tri3_linear_elasticity_assemble_apply\t%g seconds\n", tock
    // - tick);
}
