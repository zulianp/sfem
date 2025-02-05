#ifndef TRI6_INLINE_CPU_H
#define TRI6_INLINE_CPU_H

#include "tri3_inline_cpu.h"

static SFEM_INLINE int tri6_linear_search(const idx_t target, const idx_t *const arr, const int size) {
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

static SFEM_INLINE int tri6_find_col(const idx_t key, const idx_t *const row, const int lenrow) {
    if (lenrow <= 32) {
        return tri6_linear_search(key, row, lenrow);
    } else {
        // Use this for larger number of dofs per row
        return find_idx_binary_search(key, row, lenrow);
    }
}

static SFEM_INLINE void tri6_find_cols(const idx_t *targets,
                                    const idx_t *const row,
                                    const int lenrow,
                                    idx_t *ks) {
    if (lenrow > 32) {
        for (int d = 0; d < 6; ++d) {
            ks[d] = tri6_find_col(targets[d], row, lenrow);
        }
    } else {
#pragma unroll(6)
        for (int d = 0; d < 6; ++d) {
            ks[d] = 0;
        }

        for (int i = 0; i < lenrow; ++i) {
#pragma unroll(6)
            for (int d = 0; d < 6; ++d) {
                ks[d] += row[i] < targets[d];
            }
        }
    }
}

static SFEM_INLINE void tri6_local_to_global(const idx_t *const SFEM_RESTRICT ev,
                                             const accumulator_t *const SFEM_RESTRICT element_matrix,
                                             const count_t *const SFEM_RESTRICT rowptr,
                                             const idx_t *const SFEM_RESTRICT colidx,
                                             real_t *const SFEM_RESTRICT values) {
    idx_t ks[6];
    for (int edof_i = 0; edof_i < 6; ++edof_i) {
        const idx_t dof_i = ev[edof_i];
        const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];
        const idx_t *row = &colidx[rowptr[dof_i]];

        tri6_find_cols(ev, row, lenrow, ks);

        real_t *rowvalues = &values[rowptr[dof_i]];
        const accumulator_t *element_row = &element_matrix[edof_i * 6];

#pragma unroll(6)
        for (int edof_j = 0; edof_j < 6; ++edof_j) {
            assert(ks[edof_j] >= 0);
#pragma omp atomic update
            rowvalues[ks[edof_j]] += element_row[edof_j];
        }
    }
}

#endif //TRI6_INLINE_CPU_H
