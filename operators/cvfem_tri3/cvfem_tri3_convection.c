#include "cvfem_tri3_convection.h"

#include "sfem_base.h"
#include "sfem_vec.h"
#include "sortreduce.h"

#include <assert.h>
#include <mpi.h>
#include <stdio.h>

#define POW2(a) ((a) * (a))

static SFEM_INLINE void cvfem_tri3_convection_assemble_hessian_kernel(
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t *const SFEM_RESTRICT vx,
    const real_t *const SFEM_RESTRICT vy,
    real_t *const SFEM_RESTRICT element_matrix) {
    const real_t x0 = (1.0 / 6.0) * px0;
    const real_t x1 = (1.0 / 6.0) * px1;
    const real_t x2 = (5.0 / 12.0) * vy[0];
    const real_t x3 = (5.0 / 12.0) * vy[1];
    const real_t x4 = (1.0 / 6.0) * py0;
    const real_t x5 = (1.0 / 6.0) * py1;
    const real_t x6 = (5.0 / 12.0) * vx[0];
    const real_t x7 = (5.0 / 12.0) * vx[1];
    const real_t x8 = (-1.0 / 3.0 * px2 + x0 + x1) * ((1.0 / 6.0) * vy[2] + x2 + x3) +
                      ((1.0 / 3.0) * py2 - x4 - x5) * ((1.0 / 6.0) * vx[2] + x6 + x7);
    const real_t x9 = ((x8 < 0) ? (0) : (x8));
    const real_t x10 = (1.0 / 6.0) * px2;
    const real_t x11 = (5.0 / 12.0) * vy[2];
    const real_t x12 = (1.0 / 6.0) * py2;
    const real_t x13 = (5.0 / 12.0) * vx[2];
    const real_t x14 = (-1.0 / 3.0 * px1 + x0 + x10) * ((1.0 / 6.0) * vy[1] + x11 + x2) +
                       ((1.0 / 3.0) * py1 - x12 - x4) * ((1.0 / 6.0) * vx[1] + x13 + x6);
    const real_t x15 = ((x14 > 0) ? (0) : (-x14));
    const real_t x16 = ((x8 > 0) ? (0) : (-x8));
    const real_t x17 = ((x14 < 0) ? (0) : (x14));
    const real_t x18 = (-1.0 / 3.0 * px0 + x1 + x10) * ((1.0 / 6.0) * vy[0] + x11 + x3) +
                       ((1.0 / 3.0) * py0 - x12 - x5) * ((1.0 / 6.0) * vx[0] + x13 + x7);
    const real_t x19 = ((x18 < 0) ? (0) : (x18));
    const real_t x20 = ((x18 > 0) ? (0) : (-x18));
    element_matrix[0] = -x15 - x9;
    element_matrix[1] = x16;
    element_matrix[2] = x17;
    element_matrix[3] = x9;
    element_matrix[4] = -x16 - x19;
    element_matrix[5] = x20;
    element_matrix[6] = x15;
    element_matrix[7] = x19;
    element_matrix[8] = -x17 - x20;
}

static SFEM_INLINE void cvfem_tri3_convection_assemble_apply_kernel(
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t *const SFEM_RESTRICT vx,
    const real_t *const SFEM_RESTRICT vy,
    const real_t *const SFEM_RESTRICT x,
    real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = (1.0 / 6.0) * px0;
    const real_t x1 = (1.0 / 6.0) * px2;
    const real_t x2 = (5.0 / 12.0) * vy[0];
    const real_t x3 = (5.0 / 12.0) * vy[2];
    const real_t x4 = (1.0 / 6.0) * py0;
    const real_t x5 = (1.0 / 6.0) * py2;
    const real_t x6 = (5.0 / 12.0) * vx[0];
    const real_t x7 = (5.0 / 12.0) * vx[2];
    const real_t x8 = (-1.0 / 3.0 * px1 + x0 + x1) * ((1.0 / 6.0) * vy[1] + x2 + x3) +
                      ((1.0 / 3.0) * py1 - x4 - x5) * ((1.0 / 6.0) * vx[1] + x6 + x7);
    const real_t x9 = ((x8 < 0) ? (0) : (x8));
    const real_t x10 = (1.0 / 6.0) * px1;
    const real_t x11 = (5.0 / 12.0) * vy[1];
    const real_t x12 = (1.0 / 6.0) * py1;
    const real_t x13 = (5.0 / 12.0) * vx[1];
    const real_t x14 = (-1.0 / 3.0 * px2 + x0 + x10) * ((1.0 / 6.0) * vy[2] + x11 + x2) +
                       ((1.0 / 3.0) * py2 - x12 - x4) * ((1.0 / 6.0) * vx[2] + x13 + x6);
    const real_t x15 = ((x14 > 0) ? (0) : (-x14));
    const real_t x16 = ((x14 < 0) ? (0) : (x14));
    const real_t x17 = ((x8 > 0) ? (0) : (-x8));
    const real_t x18 = (-1.0 / 3.0 * px0 + x1 + x10) * ((1.0 / 6.0) * vy[0] + x11 + x3) +
                       ((1.0 / 3.0) * py0 - x12 - x5) * ((1.0 / 6.0) * vx[0] + x13 + x7);
    const real_t x19 = ((x18 > 0) ? (0) : (-x18));
    const real_t x20 = ((x18 < 0) ? (0) : (x18));
    element_vector[0] = x15 * x[1] + x9 * x[2] + x[0] * (-x16 - x17);
    element_vector[1] = x16 * x[0] + x19 * x[2] + x[1] * (-x15 - x20);
    element_vector[2] = x17 * x[0] + x20 * x[1] + x[2] * (-x19 - x9);
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

void cvfem_tri3_convection_assemble_hessian(const ptrdiff_t nelements,
                                            const ptrdiff_t nnodes,
                                            idx_t **const SFEM_RESTRICT elems,
                                            geom_t **const SFEM_RESTRICT xyz,
                                            real_t **const SFEM_RESTRICT velocity,
                                            const count_t *const SFEM_RESTRICT rowptr,
                                            const idx_t *const SFEM_RESTRICT colidx,
                                            real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

#pragma omp parallel
    {
#pragma omp for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[3];
            idx_t ks[3];
            real_t element_matrix[3 * 3];
            real_t vx[3];
            real_t vy[3];

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                ev[v] = elems[v][i];
            }

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                vx[v] = velocity[0][ev[v]];
            }

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                vy[v] = velocity[1][ev[v]];
            }

            cvfem_tri3_convection_assemble_hessian_kernel(
                // X-coordinates
                xyz[0][ev[0]],
                xyz[0][ev[1]],
                xyz[0][ev[2]],
                // Y-coordinates
                xyz[1][ev[0]],
                xyz[1][ev[1]],
                xyz[1][ev[2]],
                vx,
                vy,
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
    printf("cvfem_tri3_convection.c: cvfem_tri3_convection_assemble_hessian\t%g seconds\n",
           tock - tick);
}

void cvfem_tri3_convection_apply(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 real_t **const SFEM_RESTRICT velocity,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[3];

            real_t vx[3];
            real_t vy[3];
            real_t element_u[3];
            real_t element_vector[3];

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                ev[v] = elems[v][i];
            }

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                vx[v] = velocity[0][ev[v]];
            }

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                vy[v] = velocity[1][ev[v]];
            }

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                element_u[v] = u[ev[v]];
            }

            cvfem_tri3_convection_assemble_apply_kernel(
                // X-coordinates
                xyz[0][ev[0]],
                xyz[0][ev[1]],
                xyz[0][ev[2]],
                // Y-coordinates
                xyz[1][ev[0]],
                xyz[1][ev[1]],
                xyz[1][ev[2]],
                vx,
                vy,
                element_u,
                element_vector);

            for (int edof_i = 0; edof_i < 3; ++edof_i) {
                const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
                values[dof_i] += element_vector[edof_i];
            }
        }
    }
}

void cvfem_tri3_cv_volumes(const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const SFEM_RESTRICT elems,
                           geom_t **const SFEM_RESTRICT xyz,
                           real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = xyz[0];
    const geom_t *const y = xyz[1];

#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[3];

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                ev[v] = elems[v][i];
            }

            real_t J[4];
            J[0] = -x[ev[0]] + x[ev[1]];
            J[1] = -x[ev[0]] + x[ev[2]];
            J[2] = -y[ev[0]] + y[ev[1]];
            J[3] = -y[ev[0]] + y[ev[2]];

            const real_t measure = ((J[0] * J[3] - J[1] * J[2]) / 2) / 3;

            assert(measure > 0);
            for (int edof_i = 0; edof_i < 3; ++edof_i) {
                const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
                values[dof_i] += measure;
            }
        }
    }
}
