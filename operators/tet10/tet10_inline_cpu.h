#ifndef TET10_INLINE_CPU_H
#define TET10_INLINE_CPU_H

#include "tet4_inline_cpu.h"

static SFEM_INLINE void tet10_ref_shape_grad_x(const scalar_t qx,
                                               const scalar_t qy,
                                               const scalar_t qz,
                                               scalar_t *const out) {
    const scalar_t x0 = 4 * qx;
    const scalar_t x1 = 4 * qy;
    const scalar_t x2 = 4 * qz;
    const scalar_t x3 = x1 + x2;
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
    const scalar_t x0 = 4 * qy;
    const scalar_t x1 = 4 * qx;
    const scalar_t x2 = 4 * qz;
    const scalar_t x3 = x1 + x2;
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
    const scalar_t x0 = 4 * qz;
    const scalar_t x1 = 4 * qx;
    const scalar_t x2 = 4 * qy;
    const scalar_t x3 = x1 + x2;
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

static SFEM_INLINE void tet10_find_cols(const idx_t *targets,
                                    const idx_t *const row,
                                    const int lenrow,
                                    idx_t *ks) {
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
        const accumulator_t *element_row = &element_matrix[edof_i * 10];

#pragma unroll(10)
        for (int edof_j = 0; edof_j < 10; ++edof_j) {
            assert(ks[edof_j] >= 0);
#pragma omp atomic update
            rowvalues[ks[edof_j]] += element_row[edof_j];
        }
    }
}

#endif  // TET10_INLINE_CPU_H
