#ifndef TET10_INLINE_CPU_H
#define TET10_INLINE_CPU_H

#include "tet4_inline_cpu.h"

<<<<<<< HEAD
static SFEM_INLINE void tet10_ref_shape_grad_x(const scalar_t qx,
                                               const scalar_t qy,
                                               const scalar_t qz,
                                               scalar_t *const out) {
=======
static SFEM_INLINE void tet10_ref_shape_grad_x(const scalar_t qx, const scalar_t qy, const scalar_t qz, scalar_t *const out) {
>>>>>>> origin/main
    const scalar_t x0 = 4 * qx;
    const scalar_t x1 = 4 * qy;
    const scalar_t x2 = 4 * qz;
    const scalar_t x3 = x1 + x2;
<<<<<<< HEAD
    out[0] = x0 + x3 - 3;
    out[1] = x0 - 1;
    out[2] = 0;
    out[3] = 0;
    out[4] = -8 * qx - x3 + 4;
    out[5] = x1;
    out[6] = -x1;
    out[7] = -x2;
    out[8] = x2;
    out[9] = 0;
}

static SFEM_INLINE void tet10_ref_shape_grad_y(const scalar_t qx,
                                               const scalar_t qy,
                                               const scalar_t qz,
                                               scalar_t *const out) {
=======
    out[0]            = x0 + x3 - 3;
    out[1]            = x0 - 1;
    out[2]            = 0;
    out[3]            = 0;
    out[4]            = -8 * qx - x3 + 4;
    out[5]            = x1;
    out[6]            = -x1;
    out[7]            = -x2;
    out[8]            = x2;
    out[9]            = 0;
}

static SFEM_INLINE void tet10_ref_shape_grad_y(const scalar_t qx, const scalar_t qy, const scalar_t qz, scalar_t *const out) {
>>>>>>> origin/main
    const scalar_t x0 = 4 * qy;
    const scalar_t x1 = 4 * qx;
    const scalar_t x2 = 4 * qz;
    const scalar_t x3 = x1 + x2;
<<<<<<< HEAD
    out[0] = x0 + x3 - 3;
    out[1] = 0;
    out[2] = x0 - 1;
    out[3] = 0;
    out[4] = -x1;
    out[5] = x1;
    out[6] = -8 * qy - x3 + 4;
    out[7] = -x2;
    out[8] = 0;
    out[9] = x2;
}

static SFEM_INLINE void tet10_ref_shape_grad_z(const scalar_t qx,
                                               const scalar_t qy,
                                               const scalar_t qz,
                                               scalar_t *const out) {
=======
    out[0]            = x0 + x3 - 3;
    out[1]            = 0;
    out[2]            = x0 - 1;
    out[3]            = 0;
    out[4]            = -x1;
    out[5]            = x1;
    out[6]            = -8 * qy - x3 + 4;
    out[7]            = -x2;
    out[8]            = 0;
    out[9]            = x2;
}

static SFEM_INLINE void tet10_ref_shape_grad_z(const scalar_t qx, const scalar_t qy, const scalar_t qz, scalar_t *const out) {
>>>>>>> origin/main
    const scalar_t x0 = 4 * qz;
    const scalar_t x1 = 4 * qx;
    const scalar_t x2 = 4 * qy;
    const scalar_t x3 = x1 + x2;
<<<<<<< HEAD
    out[0] = x0 + x3 - 3;
    out[1] = 0;
    out[2] = 0;
    out[3] = x0 - 1;
    out[4] = -x1;
    out[5] = 0;
    out[6] = -x2;
    out[7] = -8 * qz - x3 + 4;
    out[8] = x1;
    out[9] = x2;
=======
    out[0]            = x0 + x3 - 3;
    out[1]            = 0;
    out[2]            = 0;
    out[3]            = x0 - 1;
    out[4]            = -x1;
    out[5]            = 0;
    out[6]            = -x2;
    out[7]            = -8 * qz - x3 + 4;
    out[8]            = x1;
    out[9]            = x2;
>>>>>>> origin/main
}

static SFEM_INLINE idx_t tet10_linear_search(const idx_t target, const idx_t *const arr, const int size) {
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

static SFEM_INLINE idx_t tet10_find_col(const idx_t key, const idx_t *const row, const int lenrow) {
    if (lenrow <= 32) {
        return tet10_linear_search(key, row, lenrow);
    } else {
        // Use this for larger number of dofs per row
        return find_idx_binary_search(key, row, lenrow);
    }
}

<<<<<<< HEAD
static SFEM_INLINE void tet10_find_cols(const idx_t *targets,
                                    const idx_t *const row,
                                    const int lenrow,
                                    idx_t *ks) {
=======
static SFEM_INLINE void tet10_find_cols(const idx_t *targets, const idx_t *const row, const int lenrow, idx_t *ks) {
>>>>>>> origin/main
    if (lenrow > 32) {
        for (int d = 0; d < 10; ++d) {
            ks[d] = tet10_find_col(targets[d], row, lenrow);
        }
    } else {
#pragma unroll(10)
        for (int d = 0; d < 10; ++d) {
            ks[d] = 0;
        }

        for (int i = 0; i < lenrow; ++i) {
#pragma unroll(10)
            for (int d = 0; d < 10; ++d) {
                ks[d] += row[i] < targets[d];
            }
        }
    }
}

<<<<<<< HEAD
static SFEM_INLINE void tet10_local_to_global(const idx_t *const SFEM_RESTRICT ev,
                                             const accumulator_t *const SFEM_RESTRICT element_matrix,
                                             const count_t *const SFEM_RESTRICT rowptr,
                                             const idx_t *const SFEM_RESTRICT colidx,
                                             real_t *const SFEM_RESTRICT values) {
    idx_t ks[10];
    for (int edof_i = 0; edof_i < 10; ++edof_i) {
        const idx_t dof_i = ev[edof_i];
        const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];
        const idx_t *row = &colidx[rowptr[dof_i]];

        tet10_find_cols(ev, row, lenrow, ks);

        real_t *rowvalues = &values[rowptr[dof_i]];
=======
static SFEM_INLINE void tet10_local_to_global(const idx_t *const SFEM_RESTRICT         ev,
                                              const accumulator_t *const SFEM_RESTRICT element_matrix,
                                              const count_t *const SFEM_RESTRICT       rowptr,
                                              const idx_t *const SFEM_RESTRICT         colidx,
                                              real_t *const SFEM_RESTRICT              values) {
    idx_t ks[10];
    for (int edof_i = 0; edof_i < 10; ++edof_i) {
        const idx_t  dof_i  = ev[edof_i];
        const idx_t  lenrow = rowptr[dof_i + 1] - rowptr[dof_i];
        const idx_t *row    = &colidx[rowptr[dof_i]];

        tet10_find_cols(ev, row, lenrow, ks);

        real_t              *rowvalues   = &values[rowptr[dof_i]];
>>>>>>> origin/main
        const accumulator_t *element_row = &element_matrix[edof_i * 10];

#pragma unroll(10)
        for (int edof_j = 0; edof_j < 10; ++edof_j) {
            assert(ks[edof_j] >= 0);
#pragma omp atomic update
            rowvalues[ks[edof_j]] += element_row[edof_j];
        }
    }
}

<<<<<<< HEAD
=======
static SFEM_INLINE void tet10_local_to_global_bsr3(const idx_t *const SFEM_RESTRICT         ev,
                                                  const accumulator_t *const SFEM_RESTRICT element_matrix,
                                                  const count_t *const SFEM_RESTRICT       rowptr,
                                                  const idx_t *const SFEM_RESTRICT         colidx,
                                                  real_t *const SFEM_RESTRICT              values) {
    idx_t ks[10];
    for (int edof_i = 0; edof_i < 10; ++edof_i) {
        const idx_t dof_i = ev[edof_i];

        {
            const idx_t  lenrow = rowptr[dof_i + 1] - rowptr[dof_i];
            const idx_t *cols   = &colidx[rowptr[dof_i]];
            tet10_find_cols(ev, cols, lenrow, ks);
        }

        real_t *const row = &values[rowptr[dof_i] * 9];

        for (int edof_j = 0; edof_j < 10; ++edof_j) {
            // extract the i, j block
            real_t *const block = &row[ks[edof_j] * 9];

            for (int bi = 0; bi < 3; ++bi) {
                const int ii = bi * 10 + edof_i;

                for (int bj = 0; bj < 3; ++bj) {
                    const int    jj  = bj * 10 + edof_j;
                    const real_t val = element_matrix[ii * 30 + jj];

#pragma omp atomic update
                    block[bi * 3 + bj] += val;
                }
            }
        }
    }
}

static SFEM_INLINE void tet10_adjugate_and_det(const scalar_t *const SFEM_RESTRICT x,
                                               const scalar_t *const SFEM_RESTRICT y,
                                               const scalar_t *const SFEM_RESTRICT z,
                                               const scalar_t                      qx,
                                               const scalar_t                      qy,
                                               const scalar_t                      qz,
                                               scalar_t *const SFEM_RESTRICT       adjugate,
                                               scalar_t *const SFEM_RESTRICT       determinant) {
    // mundane ops: 194 divs: 0 sqrts: 0
    // total ops: 194
    const scalar_t x0  = 4 * qy;
    const scalar_t x1  = x0 - 1;
    const scalar_t x2  = 4 * qz;
    const scalar_t x3  = -x2 * y[7];
    const scalar_t x4  = qx - 1;
    const scalar_t x5  = 8 * qy + 4 * qz + 4 * x4;
    const scalar_t x6  = 4 * qx;
    const scalar_t x7  = x0 + x2 + x6 - 3;
    const scalar_t x8  = x7 * y[0];
    const scalar_t x9  = -x6 * y[4] + x8;
    const scalar_t x10 = x1 * y[2] + x2 * y[9] + x3 - x5 * y[6] + x6 * y[5] + x9;
    const scalar_t x11 = x2 - 1;
    const scalar_t x12 = -x0 * z[6];
    const scalar_t x13 = 4 * qy + 8 * qz + 4 * x4;
    const scalar_t x14 = x7 * z[0];
    const scalar_t x15 = x14 - x6 * z[4];
    const scalar_t x16 = x0 * z[9] + x11 * z[3] + x12 - x13 * z[7] + x15 + x6 * z[8];
    const scalar_t x17 = x10 * x16;
    const scalar_t x18 = -x0 * y[6];
    const scalar_t x19 = x0 * y[9] + x11 * y[3] - x13 * y[7] + x18 + x6 * y[8] + x9;
    const scalar_t x20 = -x2 * z[7];
    const scalar_t x21 = x1 * z[2] + x15 + x2 * z[9] + x20 - x5 * z[6] + x6 * z[5];
    const scalar_t x22 = x19 * x21;
    const scalar_t x23 = -x2 * x[7];
    const scalar_t x24 = x7 * x[0];
    const scalar_t x25 = x24 - x6 * x[4];
    const scalar_t x26 = x1 * x[2] + x2 * x[9] + x23 + x25 - x5 * x[6] + x6 * x[5];
    const scalar_t x27 = x16 * x26;
    const scalar_t x28 = -x0 * x[6];
    const scalar_t x29 = x0 * x[9] + x11 * x[3] - x13 * x[7] + x25 + x28 + x6 * x[8];
    const scalar_t x30 = x19 * x26;
    const scalar_t x31 = x10 * x29;
    const scalar_t x32 = x6 - 1;
    const scalar_t x33 = 8 * qx + 4 * qy + 4 * qz - 4;
    const scalar_t x34 = x0 * z[5] + x12 + x14 + x2 * z[8] + x20 + x32 * z[1] - x33 * z[4];
    const scalar_t x35 = x0 * y[5] + x18 + x2 * y[8] + x3 + x32 * y[1] - x33 * y[4] + x8;
    const scalar_t x36 = x0 * x[5] + x2 * x[8] + x23 + x24 + x28 + x32 * x[1] - x33 * x[4];
    adjugate[0]        = x17 - x22;
    adjugate[1]        = x21 * x29 - x27;
    adjugate[2]        = x30 - x31;
    adjugate[3]        = -x16 * x35 + x19 * x34;
    adjugate[4]        = x16 * x36 - x29 * x34;
    adjugate[5]        = -x19 * x36 + x29 * x35;
    adjugate[6]        = -x10 * x34 + x21 * x35;
    adjugate[7]        = -x21 * x36 + x26 * x34;
    adjugate[8]        = x10 * x36 - x26 * x35;
    determinant[0]     = x17 * x36 + x21 * x29 * x35 - x22 * x36 - x27 * x35 + x30 * x34 - x31 * x34;
}

>>>>>>> origin/main
#endif  // TET10_INLINE_CPU_H
