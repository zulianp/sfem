#include "cvfem_quad4_convection.h"

#include "sfem_base.h"
#include "sfem_vec.h"
#include "sortreduce.h"

#include <assert.h>
#include <mpi.h>
#include <stdio.h>

#define POW2(a) ((a) * (a))

static SFEM_INLINE void cvfem_quad4_convection_assemble_hessian_kernel(
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t px3,
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t py3,
    const real_t *const SFEM_RESTRICT vx,
    const real_t *const SFEM_RESTRICT vy,
    real_t *const SFEM_RESTRICT element_matrix) {
    const real_t x0 = 0.25 * px1;
    const real_t x1 = 0.25 * px3;
    const real_t x2 = 0.25 * px0 - 0.25 * px2;
    const real_t x3 = x0 - x1 + x2;
    const real_t x4 = 0.375 * vy[0] + 0.125 * vy[2];
    const real_t x5 = 0.375 * vy[1] + 0.125 * vy[3];
    const real_t x6 = 0.25 * py1;
    const real_t x7 = 0.25 * py3;
    const real_t x8 = 0.25 * py0 - 0.25 * py2;
    const real_t x9 = x6 - x7 + x8;
    const real_t x10 = 0.375 * vx[0] + 0.125 * vx[2];
    const real_t x11 = 0.375 * vx[1] + 0.125 * vx[3];
    const real_t x12 = x3 * (x4 + x5) - x9 * (x10 + x11);
    const real_t x13 = ((x12 < 0) ? (0) : (x12));
    const real_t x14 = -x0 + x1 + x2;
    const real_t x15 = 0.125 * vy[1] + 0.375 * vy[3];
    const real_t x16 = -x6 + x7 + x8;
    const real_t x17 = 0.125 * vx[1] + 0.375 * vx[3];
    const real_t x18 = x14 * (x15 + x4) - x16 * (x10 + x17);
    const real_t x19 = ((x18 > 0) ? (0) : (-x18));
    const real_t x20 = ((x12 > 0) ? (0) : (-x12));
    const real_t x21 = ((x18 < 0) ? (0) : (x18));
    const real_t x22 = 0.125 * vy[0] + 0.375 * vy[2];
    const real_t x23 = 0.125 * vx[0] + 0.375 * vx[2];
    const real_t x24 = -x14 * (x22 + x5) + x16 * (x11 + x23);
    const real_t x25 = ((x24 < 0) ? (0) : (x24));
    const real_t x26 = ((x24 > 0) ? (0) : (-x24));
    const real_t x27 = -x3 * (x15 + x22) + x9 * (x17 + x23);
    const real_t x28 = ((x27 < 0) ? (0) : (x27));
    const real_t x29 = ((x27 > 0) ? (0) : (-x27));
    element_matrix[0] = -x13 - x19;
    element_matrix[1] = x20;
    element_matrix[2] = 0;
    element_matrix[3] = x21;
    element_matrix[4] = x13;
    element_matrix[5] = -x20 - x25;
    element_matrix[6] = x26;
    element_matrix[7] = 0;
    element_matrix[8] = 0;
    element_matrix[9] = x25;
    element_matrix[10] = -x26 - x28;
    element_matrix[11] = x29;
    element_matrix[12] = x19;
    element_matrix[13] = 0;
    element_matrix[14] = x28;
    element_matrix[15] = -x21 - x29;
}

static SFEM_INLINE void cvfem_quad4_convection_assemble_apply_kernel(
    const real_t px0,
    const real_t px1,
    const real_t px2,
    const real_t px3,
    const real_t py0,
    const real_t py1,
    const real_t py2,
    const real_t py3,
    const real_t *const SFEM_RESTRICT vx,
    const real_t *const SFEM_RESTRICT vy,
    const real_t *const SFEM_RESTRICT x,
    real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = 0.25 * px3;
    const real_t x1 = 0.25 * px1;
    const real_t x2 = 0.25 * px0 - 0.25 * px2;
    const real_t x3 = x0 - x1 + x2;
    const real_t x4 = 0.375 * vy[0] + 0.125 * vy[2];
    const real_t x5 = 0.125 * vy[1] + 0.375 * vy[3];
    const real_t x6 = 0.25 * py3;
    const real_t x7 = 0.25 * py1;
    const real_t x8 = 0.25 * py0 - 0.25 * py2;
    const real_t x9 = x6 - x7 + x8;
    const real_t x10 = 0.375 * vx[0] + 0.125 * vx[2];
    const real_t x11 = 0.125 * vx[1] + 0.375 * vx[3];
    const real_t x12 = x3 * (x4 + x5) - x9 * (x10 + x11);
    const real_t x13 = ((x12 < 0) ? (0) : (x12));
    const real_t x14 = -x0 + x1 + x2;
    const real_t x15 = 0.375 * vy[1] + 0.125 * vy[3];
    const real_t x16 = -x6 + x7 + x8;
    const real_t x17 = 0.375 * vx[1] + 0.125 * vx[3];
    const real_t x18 = x14 * (x15 + x4) - x16 * (x10 + x17);
    const real_t x19 = ((x18 > 0) ? (0) : (-x18));
    const real_t x20 = ((x18 < 0) ? (0) : (x18));
    const real_t x21 = ((x12 > 0) ? (0) : (-x12));
    const real_t x22 = 0.125 * vy[0] + 0.375 * vy[2];
    const real_t x23 = 0.125 * vx[0] + 0.375 * vx[2];
    const real_t x24 = -x3 * (x15 + x22) + x9 * (x17 + x23);
    const real_t x25 = ((x24 > 0) ? (0) : (-x24));
    const real_t x26 = ((x24 < 0) ? (0) : (x24));
    const real_t x27 = -x14 * (x22 + x5) + x16 * (x11 + x23);
    const real_t x28 = ((x27 > 0) ? (0) : (-x27));
    const real_t x29 = ((x27 < 0) ? (0) : (x27));
    element_vector[0] = x13 * x[3] + x19 * x[1] + x[0] * (-x20 - x21);
    element_vector[1] = x20 * x[0] + x25 * x[2] + x[1] * (-x19 - x26);
    element_vector[2] = x26 * x[1] + x28 * x[3] + x[2] * (-x25 - x29);
    element_vector[3] = x21 * x[0] + x29 * x[2] + x[3] * (-x13 - x28);
}

static SFEM_INLINE idx_t linear_search(const idx_t target, const idx_t *const arr, const int size) {
    idx_t i;
    for (i = 0; i < size - 4; i += 4) {
        if (arr[i] == target) return i;
        if (arr[i + 1] == target) return i + 1;
        if (arr[i + 2] == target) return i + 2;
        if (arr[i + 3] == target) return i + 3;
    }
    for (; i < size; i++) {
        if (arr[i] == target) return i;
    }
    return SFEM_IDX_INVALID;
}

static SFEM_INLINE idx_t find_col(const idx_t key, const idx_t *const row, const int lenrow) {
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

static SFEM_INLINE void find_cols4(const idx_t *targets,
                                   const idx_t *const row,
                                   const int lenrow,
                                   idx_t *ks) {
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

void cvfem_quad4_convection_assemble_hessian(const ptrdiff_t nelements,
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
            idx_t ev[4];
            idx_t ks[4];
            real_t element_matrix[4 * 4];
            real_t vx[4];
            real_t vy[4];

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                vx[v] = velocity[0][ev[v]];
            }

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                vy[v] = velocity[1][ev[v]];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            cvfem_quad4_convection_assemble_hessian_kernel(
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
                vx,
                vy,
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
#pragma omp atomic update
                    rowvalues[ks[edof_j]] += element_row[edof_j];
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("cvfem_quad4_convection.c: cvfem_quad4_convection_assemble_hessian\t%g seconds\n",
           tock - tick);
}

void cvfem_quad4_convection_apply(const ptrdiff_t nelements,
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
            idx_t ev[4];

            real_t vx[4];
            real_t vy[4];
            real_t element_u[4];
            real_t element_vector[4];

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                vx[v] = velocity[0][ev[v]];
            }

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                vy[v] = velocity[1][ev[v]];
            }

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                element_u[v] = u[ev[v]];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];
            const idx_t i3 = ev[3];

            cvfem_quad4_convection_assemble_apply_kernel(
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
                vx,
                vy,
                element_u,
                element_vector);

            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
                values[dof_i] += element_vector[edof_i];
            }
        }
    }
}

void cvfem_quad4_cv_volumes(const ptrdiff_t nelements,
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
            idx_t ev[4];

#pragma unroll(4)
            for (int v = 0; v < 4; ++v) {
                ev[v] = elems[v][i];
            }

            // FIXME (this assumes element is affine)
            real_t J[4];
            J[0] = -x[ev[0]] + x[ev[1]];
            J[1] = -x[ev[0]] + x[ev[3]];
            J[2] = -y[ev[0]] + y[ev[1]];
            J[3] = -y[ev[0]] + y[ev[3]];

            const real_t measure = (J[0] * J[3] - J[1] * J[2]) / 4;

            assert(measure > 0);
            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
                values[dof_i] += measure;
            }
        }
    }
}
