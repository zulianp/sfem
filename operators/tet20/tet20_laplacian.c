#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"

#define POW2(a) ((a) * (a))

static SFEM_INLINE void tet20_laplacian_hessian(const real_t px0,
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
                                                real_t *element_matrix) {
    static const int stride = 1;

    real_t fff[6];

    {
        // FLOATING POINT OPS!
        //       - Result: 6*ADD + 6*ASSIGNMENT + 24*MUL + 9*POW
        //       - Subexpressions: 2*ADD + 28*MUL + NEG + POW + 21*SUB
        const real_t x0 = -px0 + px1;
        const real_t x1 = -py0 + py2;
        const real_t x2 = -pz0 + pz3;
        const real_t x3 = x1 * x2;
        const real_t x4 = -pz0 + pz1;
        const real_t x5 = -px0 + px2;
        const real_t x6 = -py0 + py3;
        const real_t x7 = x5 * x6;
        const real_t x8 = -py0 + py1;
        const real_t x9 = -px0 + px3;
        const real_t x10 = -pz0 + pz2;
        const real_t x11 = x10 * x6;
        const real_t x12 = x2 * x5;
        const real_t x13 = x1 * x9;
        const real_t x14 = -x0 * x11 + x0 * x3 + x10 * x8 * x9 - x12 * x8 - x13 * x4 + x4 * x7;
        const real_t x15 = -x13 + x7;
        // const real_t x16 = pow(x14, -2);
        const real_t x16 = 1. / POW2(x14);
        const real_t x17 = x10 * x9 - x12;
        const real_t x18 = -x11 + x3;
        const real_t x19 = -x0 * x6 + x8 * x9;
        const real_t x20 = x15 * x16;
        const real_t x21 = x0 * x2 - x4 * x9;
        const real_t x22 = x16 * x17;
        const real_t x23 = -x2 * x8 + x4 * x6;
        const real_t x24 = x16 * x18;
        const real_t x25 = x0 * x1 - x5 * x8;
        const real_t x26 = -x0 * x10 + x4 * x5;
        const real_t x27 = -x1 * x4 + x10 * x8;
        fff[0 * stride] = x14 * (POW2(x15) * x16 + x16 * POW2(x17) + x16 * POW2(x18));
        fff[1 * stride] = x14 * (x19 * x20 + x21 * x22 + x23 * x24);
        fff[2 * stride] = x14 * (x20 * x25 + x22 * x26 + x24 * x27);
        fff[3 * stride] = x14 * (x16 * POW2(x19) + x16 * POW2(x21) + x16 * POW2(x23));
        fff[4 * stride] = x14 * (x16 * x19 * x25 + x16 * x21 * x26 + x16 * x23 * x27);
        fff[5 * stride] = x14 * (x16 * POW2(x25) + x16 * POW2(x26) + x16 * POW2(x27));
    }

    // TODO
}

static SFEM_INLINE void tet20_laplacian_gradient(const real_t px0,
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
                                                 const real_t *SFEM_RESTRICT u,
                                                 real_t *SFEM_RESTRICT element_vector) {
    static const int stride = 1;
    real_t fff[6];

    {
        // FLOATING POINT OPS!
        //       - Result: 6*ADD + 6*ASSIGNMENT + 24*MUL + 9*POW
        //       - Subexpressions: 2*ADD + 28*MUL + NEG + POW + 21*SUB
        const real_t x0 = -px0 + px1;
        const real_t x1 = -py0 + py2;
        const real_t x2 = -pz0 + pz3;
        const real_t x3 = x1 * x2;
        const real_t x4 = -pz0 + pz1;
        const real_t x5 = -px0 + px2;
        const real_t x6 = -py0 + py3;
        const real_t x7 = x5 * x6;
        const real_t x8 = -py0 + py1;
        const real_t x9 = -px0 + px3;
        const real_t x10 = -pz0 + pz2;
        const real_t x11 = x10 * x6;
        const real_t x12 = x2 * x5;
        const real_t x13 = x1 * x9;
        const real_t x14 = -x0 * x11 + x0 * x3 + x10 * x8 * x9 - x12 * x8 - x13 * x4 + x4 * x7;
        const real_t x15 = -x13 + x7;
        // const real_t x16 = pow(x14, -2);
        const real_t x16 = 1 / POW2(x14);
        const real_t x17 = x10 * x9 - x12;
        const real_t x18 = -x11 + x3;
        const real_t x19 = -x0 * x6 + x8 * x9;
        const real_t x20 = x15 * x16;
        const real_t x21 = x0 * x2 - x4 * x9;
        const real_t x22 = x16 * x17;
        const real_t x23 = -x2 * x8 + x4 * x6;
        const real_t x24 = x16 * x18;
        const real_t x25 = x0 * x1 - x5 * x8;
        const real_t x26 = -x0 * x10 + x4 * x5;
        const real_t x27 = -x1 * x4 + x10 * x8;
        fff[0 * stride] = x14 * (POW2(x15) * x16 + x16 * POW2(x17) + x16 * POW2(x18));
        fff[1 * stride] = x14 * (x19 * x20 + x21 * x22 + x23 * x24);
        fff[2 * stride] = x14 * (x20 * x25 + x22 * x26 + x24 * x27);
        fff[3 * stride] = x14 * (x16 * POW2(x19) + x16 * POW2(x21) + x16 * POW2(x23));
        fff[4 * stride] = x14 * (x16 * x19 * x25 + x16 * x21 * x26 + x16 * x23 * x27);
        fff[5 * stride] = x14 * (x16 * POW2(x25) + x16 * POW2(x26) + x16 * POW2(x27));
    }

   // TODO
}

static SFEM_INLINE void tet20_laplacian_value(const real_t px0,
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
                                              const real_t *SFEM_RESTRICT u,
                                              real_t *SFEM_RESTRICT element_scalar) {
    // TODO
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
        for (int d = 0; d < 20; ++d) {
            ks[d] = find_col(targets[d], row, lenrow);
        }
    } else {
#pragma unroll
        for (int d = 0; d < 20; ++d) {
            ks[d] = 0;
        }

        for (int i = 0; i < lenrow; ++i) {
#pragma unroll
            for (int d = 0; d < 20; ++d) {
                ks[d] += row[i] < targets[d];
            }
        }
    }
}

void tet20_laplacian_assemble_hessian(const ptrdiff_t nelements,
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
            idx_t ev[20];
            idx_t ks[20];

            real_t element_matrix[20 * 20];

#pragma unroll(20)
            for (int v = 0; v < 20; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices for affine coordinates
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            tet20_laplacian_hessian(
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

            for (int edof_i = 0; edof_i < 20; ++edof_i) {
                const idx_t dof_i = elems[edof_i][i];
                const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

                const idx_t *row = &colidx[rowptr[dof_i]];

                find_cols20(ev, row, lenrow, ks);

                real_t *rowvalues = &values[rowptr[dof_i]];
                const real_t *element_row = &element_matrix[edof_i * 20];

#pragma unroll(20)
                for (int edof_j = 0; edof_j < 20; ++edof_j) {
#pragma omp atomic update
                    rowvalues[ks[edof_j]] += element_row[edof_j];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tet20_laplacian.c: tet20_laplacian_assemble_hessian\t%g seconds\n", tock - tick);
}

void tet20_laplacian_assemble_gradient(const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const SFEM_RESTRICT elems,
                                       geom_t **const SFEM_RESTRICT xyz,
                                       const real_t *const SFEM_RESTRICT u,
                                       real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    // double tick = MPI_Wtime();

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[20];
            real_t element_vector[20 * 20];
            real_t element_u[20];

#pragma unroll(20)
            for (int v = 0; v < 20; ++v) {
                ev[v] = elems[v][i];
            }

            for (int v = 0; v < 20; ++v) {
                element_u[v] = u[ev[v]];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            tet20_laplacian_gradient(
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

            for (int edof_i = 0; edof_i < 20; ++edof_i) {
                const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
                values[dof_i] += element_vector[edof_i];
            }
        }
    }

    // double tock = MPI_Wtime();
    // printf("tet20_laplacian.c: tet20_laplacian_assemble_gradient\t%g seconds\n", tock - tick);
}

void tet20_laplacian_assemble_value(const ptrdiff_t nelements,
                                    const ptrdiff_t nnodes,
                                    idx_t **const SFEM_RESTRICT elems,
                                    geom_t **const SFEM_RESTRICT xyz,
                                    const real_t *const SFEM_RESTRICT u,
                                    real_t *const SFEM_RESTRICT value) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[20];
            real_t element_u[20];

#pragma unroll(20)
            for (int v = 0; v < 20; ++v) {
                ev[v] = elems[v][i];
            }

            for (int v = 0; v < 20; ++v) {
                element_u[v] = u[ev[v]];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            real_t element_scalar = 0;

            tet20_laplacian_value(
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

#pragma omp atomic update
            *value += element_scalar;
        }
    }

    double tock = MPI_Wtime();
    printf("tet20_laplacian.c: tet20_laplacian_assemble_value\t%g seconds\n", tock - tick);
}

void tet20_laplacian_apply(const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const SFEM_RESTRICT elems,
                           geom_t **const SFEM_RESTRICT xyz,
                           const real_t *const SFEM_RESTRICT u,
                           real_t *const SFEM_RESTRICT values) {
    tet20_laplacian_assemble_gradient(nelements, nnodes, elems, xyz, u, values);
}
