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
    // TODO
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
    const real_t x8 = u[0] * x6 + u[3] * x7;
    const real_t x9 = u[1] * x1 * x5 + u[4] * x3 * x5 - x8;
    const real_t x10 = mu * x9;
    const real_t x11 = x2 * x5;
    const real_t x12 = x0 * x5;
    const real_t x13 = u[0] * x11 + u[3] * x12;
    const real_t x14 = u[2] * x2 * x5 + u[5] * x0 * x5 - x13;
    const real_t x15 = (1.0 / 2.0) * lambda;
    const real_t x16 = x15 * x6;
    const real_t x17 = u[1] * x2 * x5 + u[4] * x0 * x5 - x13;
    const real_t x18 = (1.0 / 2.0) * mu;
    const real_t x19 = x11 * x18;
    const real_t x20 = u[2] * x1 * x5 + u[5] * x3 * x5 - x8;
    const real_t x21 = x10 * x6 + x14 * x16 + x16 * x9 + x17 * x19 + x19 * x20;
    const real_t x22 = mu * x14;
    const real_t x23 = x11 * x15;
    const real_t x24 = x18 * x6;
    const real_t x25 = x11 * x22 + x14 * x23 + x17 * x24 + x20 * x24 + x23 * x9;
    const real_t x26 = x15 * x7;
    const real_t x27 = x12 * x18;
    const real_t x28 = x10 * x7 + x14 * x26 + x17 * x27 + x20 * x27 + x26 * x9;
    const real_t x29 = x12 * x15;
    const real_t x30 = x18 * x7;
    const real_t x31 = x12 * x22 + x14 * x29 + x17 * x30 + x20 * x30 + x29 * x9;
    element_vector[0 * stride] = x4 * (-x21 - x25);
    element_vector[1 * stride] = x21 * x4;
    element_vector[2 * stride] = x25 * x4;
    element_vector[3 * stride] = x4 * (-x28 - x31);
    element_vector[4 * stride] = x28 * x4;
    element_vector[5 * stride] = x31 * x4;
}

void tri3_linear_elasticity_assemble_gradient_soa(const ptrdiff_t nelements,
                                                  const ptrdiff_t nnodes,
                                                  idx_t **const SFEM_RESTRICT elems,
                                                  geom_t **const SFEM_RESTRICT xyz,
                                                  const real_t **const SFEM_RESTRICT u,
                                                  real_t **const SFEM_RESTRICT values) {
    // TODO
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
    const real_t x6 = pow(x3, 2);
    const real_t x7 = pow(x5, -2);
    const real_t x8 = lambda * x7;
    const real_t x9 = (1.0 / 2.0) * x8;
    const real_t x10 = x6 * x9;
    const real_t x11 = pow(x1, 2);
    const real_t x12 = x11 * x9;
    const real_t x13 = mu * x7;
    const real_t x14 = x13 * x6;
    const real_t x15 = x11 * x13;
    const real_t x16 = x1 * x3;
    const real_t x17 = x13 * x16;
    const real_t x18 = x16 * x9 + (1.0 / 2.0) * x17;
    const real_t x19 = x12 + (1.0 / 2.0) * x14 + x15;
    const real_t x20 = x5 * (-x18 - x19);
    const real_t x21 = x10 + x14 + (1.0 / 2.0) * x15;
    const real_t x22 = x5 * (-x18 - x21);
    const real_t x23 = x4 * x9;
    const real_t x24 = x1 * x23;
    const real_t x25 = x0 * x3;
    const real_t x26 = x25 * x9;
    const real_t x27 = x13 * x25;
    const real_t x28 = x13 * x4;
    const real_t x29 = x1 * x28;
    const real_t x30 = x2 * x9 + (1.0 / 2.0) * x28 * x3;
    const real_t x31 = (1.0 / 2.0) * x13 * x2 + x23 * x3;
    const real_t x32 = x5 * (x24 + x26 + (3.0 / 2.0) * x27 + (3.0 / 2.0) * x29 + x30 + x31);
    const real_t x33 = x24 + (1.0 / 2.0) * x27 + x29;
    const real_t x34 = x5 * (-x31 - x33);
    const real_t x35 = x26 + x27 + (1.0 / 2.0) * x29;
    const real_t x36 = x5 * (-x30 - x35);
    const real_t x37 = x18 * x5;
    const real_t x38 = x5 * (-x30 - x33);
    const real_t x39 = x33 * x5;
    const real_t x40 = x30 * x5;
    const real_t x41 = x5 * (-x31 - x35);
    const real_t x42 = x31 * x5;
    const real_t x43 = x35 * x5;
    const real_t x44 = pow(x0, 2);
    const real_t x45 = x44 * x9;
    const real_t x46 = pow(x4, 2);
    const real_t x47 = x46 * x9;
    const real_t x48 = x13 * x44;
    const real_t x49 = x13 * x46;
    const real_t x50 = x0 * x4;
    const real_t x51 = x0 * x28;
    const real_t x52 = x50 * x9 + (1.0 / 2.0) * x51;
    const real_t x53 = x47 + (1.0 / 2.0) * x48 + x49;
    const real_t x54 = x5 * (-x52 - x53);
    const real_t x55 = x45 + x48 + (1.0 / 2.0) * x49;
    const real_t x56 = x5 * (-x52 - x55);
    const real_t x57 = x5 * x52;
    element_matrix[0 * stride] =
        x5 * (x10 + x12 + (3.0 / 2.0) * x14 + (3.0 / 2.0) * x15 + x16 * x8 + x17);
    element_matrix[1 * stride] = x20;
    element_matrix[2 * stride] = x22;
    element_matrix[3 * stride] = x32;
    element_matrix[4 * stride] = x34;
    element_matrix[5 * stride] = x36;
    element_matrix[6 * stride] = x20;
    element_matrix[7 * stride] = x19 * x5;
    element_matrix[8 * stride] = x37;
    element_matrix[9 * stride] = x38;
    element_matrix[10 * stride] = x39;
    element_matrix[11 * stride] = x40;
    element_matrix[12 * stride] = x22;
    element_matrix[13 * stride] = x37;
    element_matrix[14 * stride] = x21 * x5;
    element_matrix[15 * stride] = x41;
    element_matrix[16 * stride] = x42;
    element_matrix[17 * stride] = x43;
    element_matrix[18 * stride] = x32;
    element_matrix[19 * stride] = x38;
    element_matrix[20 * stride] = x41;
    element_matrix[21 * stride] =
        x5 * (x45 + x47 + (3.0 / 2.0) * x48 + (3.0 / 2.0) * x49 + x50 * x8 + x51);
    element_matrix[22 * stride] = x54;
    element_matrix[23 * stride] = x56;
    element_matrix[24 * stride] = x34;
    element_matrix[25 * stride] = x39;
    element_matrix[26 * stride] = x42;
    element_matrix[27 * stride] = x54;
    element_matrix[28 * stride] = x5 * x53;
    element_matrix[29 * stride] = x57;
    element_matrix[30 * stride] = x36;
    element_matrix[31 * stride] = x40;
    element_matrix[32 * stride] = x43;
    element_matrix[33 * stride] = x56;
    element_matrix[34 * stride] = x57;
    element_matrix[35 * stride] = x5 * x55;
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
    real_t element_displacement[(3 * 2)];

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

static SFEM_INLINE void tri3_linear_elasticity_apply_soa_kernel(
    const real_t mu,
    const real_t lambda,
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t *const SFEM_RESTRICT u,
    const real_t *const SFEM_RESTRICT increment,
    real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = -px0 + px1;
    const real_t x1 = -py0 + py2;
    const real_t x2 = x0 * x1;
    const real_t x3 = px0 - px2;
    const real_t x4 = py0 - py1;
    const real_t x5 = x2 - x3 * x4;
    const real_t x6 = pow(x5, -2);
    const real_t x7 = lambda * x6;
    const real_t x8 = (1.0 / 2.0) * x7;
    const real_t x9 = x1 * x3;
    const real_t x10 = mu * x6;
    const real_t x11 = x10 * x9;
    const real_t x12 = (1.0 / 2.0) * x11 + x8 * x9;
    const real_t x13 = pow(x1, 2);
    const real_t x14 = x10 * x13;
    const real_t x15 = x13 * x8;
    const real_t x16 = pow(x3, 2);
    const real_t x17 = x10 * x16;
    const real_t x18 = x14 + x15 + (1.0 / 2.0) * x17;
    const real_t x19 = -x12 - x18;
    const real_t x20 = increment[1] * x5;
    const real_t x21 = x16 * x8;
    const real_t x22 = (1.0 / 2.0) * x14 + x17 + x21;
    const real_t x23 = -x12 - x22;
    const real_t x24 = increment[2] * x5;
    const real_t x25 = x4 * x8;
    const real_t x26 = (1.0 / 2.0) * x10 * x2 + x25 * x3;
    const real_t x27 = x10 * x4;
    const real_t x28 = x1 * x27;
    const real_t x29 = x1 * x25;
    const real_t x30 = x0 * x3;
    const real_t x31 = x10 * x30;
    const real_t x32 = x28 + x29 + (1.0 / 2.0) * x31;
    const real_t x33 = -x26 - x32;
    const real_t x34 = increment[4] * x5;
    const real_t x35 = x2 * x8 + (1.0 / 2.0) * x27 * x3;
    const real_t x36 = x30 * x8;
    const real_t x37 = (1.0 / 2.0) * x28 + x31 + x36;
    const real_t x38 = -x35 - x37;
    const real_t x39 = increment[5] * x5;
    const real_t x40 = increment[0] * x5;
    const real_t x41 = x26 + (3.0 / 2.0) * x28 + x29 + (3.0 / 2.0) * x31 + x35 + x36;
    const real_t x42 = increment[3] * x5;
    const real_t x43 = -x32 - x35;
    const real_t x44 = -x26 - x37;
    const real_t x45 = x0 * x4;
    const real_t x46 = x0 * x27;
    const real_t x47 = x45 * x8 + (1.0 / 2.0) * x46;
    const real_t x48 = pow(x4, 2);
    const real_t x49 = x10 * x48;
    const real_t x50 = x48 * x8;
    const real_t x51 = pow(x0, 2);
    const real_t x52 = x10 * x51;
    const real_t x53 = x49 + x50 + (1.0 / 2.0) * x52;
    const real_t x54 = -x47 - x53;
    const real_t x55 = x51 * x8;
    const real_t x56 = (1.0 / 2.0) * x49 + x52 + x55;
    const real_t x57 = -x47 - x56;
    element_vector[0 * stride] =
        x19 * x20 + x23 * x24 + x33 * x34 + x38 * x39 +
        x40 * (x11 + (3.0 / 2.0) * x14 + x15 + (3.0 / 2.0) * x17 + x21 + x7 * x9) + x41 * x42;
    element_vector[1 * stride] =
        x12 * x24 + x18 * x20 + x19 * x40 + x32 * x34 + x35 * x39 + x42 * x43;
    element_vector[2 * stride] =
        x12 * x20 + x22 * x24 + x23 * x40 + x26 * x34 + x37 * x39 + x42 * x44;
    element_vector[3 * stride] =
        x20 * x43 + x24 * x44 + x34 * x54 + x39 * x57 + x40 * x41 +
        x42 * (x45 * x7 + x46 + (3.0 / 2.0) * x49 + x50 + (3.0 / 2.0) * x52 + x55);
    element_vector[4 * stride] =
        x20 * x32 + x24 * x26 + x33 * x40 + x34 * x53 + x39 * x47 + x42 * x54;
    element_vector[5 * stride] =
        x20 * x35 + x24 * x37 + x34 * x47 + x38 * x40 + x39 * x56 + x42 * x57;
}

void tri3_linear_elasticity_apply_soa(const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const SFEM_RESTRICT elems,
                                      geom_t **const SFEM_RESTRICT xyz,
                                      const real_t mu,
                                      const real_t lambda,
                                      const real_t **const SFEM_RESTRICT u,
                                      real_t **const SFEM_RESTRICT values) {
    // TODO
}