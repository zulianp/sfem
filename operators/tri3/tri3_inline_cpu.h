#ifndef TRI3_INLINE_CPU_H
#define TRI3_INLINE_CPU_H

#include "operator_inline_cpu.h"

#include <assert.h>
#include "sortreduce.h"

#ifndef POW2
#define POW2(a) ((a) * (a))
#endif

static SFEM_INLINE idx_t tri3_linear_search(const idx_t target, const idx_t *const arr, const int size) {
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

static SFEM_INLINE idx_t tri3_find_col(const idx_t key, const idx_t *const SFEM_RESTRICT row, const int lenrow) {
    if (lenrow <= 32) {
        return tri3_linear_search(key, row, lenrow);
    } else {
        // Use this for larger number of dofs per row
        return find_idx_binary_search(key, row, lenrow);
    }
}

static SFEM_INLINE void tri3_find_cols(const idx_t *const SFEM_RESTRICT targets,
                                       const idx_t *const SFEM_RESTRICT row,
                                       const int                        lenrow,
                                       idx_t *const SFEM_RESTRICT       ks) {
    if (lenrow > 32) {
        for (int d = 0; d < 3; ++d) {
            ks[d] = tri3_find_col(targets[d], row, lenrow);
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

static SFEM_INLINE void tri3_local_to_global(const idx_t *const SFEM_RESTRICT         ev,
                                             const accumulator_t *const SFEM_RESTRICT element_matrix,
                                             const count_t *const SFEM_RESTRICT       rowptr,
                                             const idx_t *const SFEM_RESTRICT         colidx,
                                             real_t *const SFEM_RESTRICT              values) {
    idx_t ks[3];
    for (int edof_i = 0; edof_i < 3; ++edof_i) {
        const idx_t  dof_i  = ev[edof_i];
        const idx_t  lenrow = rowptr[dof_i + 1] - rowptr[dof_i];
        const idx_t *row    = &colidx[rowptr[dof_i]];

        tri3_find_cols(ev, row, lenrow, ks);

        real_t              *rowvalues   = &values[rowptr[dof_i]];
        const accumulator_t *element_row = &element_matrix[edof_i * 3];

#pragma unroll(3)
        for (int edof_j = 0; edof_j < 3; ++edof_j) {
            assert(ks[edof_j] >= 0);
#pragma omp atomic update
            rowvalues[ks[edof_j]] += element_row[edof_j];
        }
    }
}

static SFEM_INLINE void tri3_fff(const geom_t      px0,
                                 const geom_t      px1,
                                 const geom_t      px2,
                                 const geom_t      py0,
                                 const geom_t      py1,
                                 const geom_t      py2,
                                 jacobian_t *const fff) {
    const real_t x0 = -px0 + px1;
    const real_t x1 = -py0 + py2;
    const real_t x2 = px0 - px2;
    const real_t x3 = py0 - py1;
    const real_t x4 = x0 * x1 - x2 * x3;
    const real_t x5 = (1 / POW2(x4));
    fff[0]          = x4 * (POW2(x1) * x5 + POW2(x2) * x5);
    fff[1]          = x4 * (x0 * x2 * x5 + x1 * x3 * x5);
    fff[2]          = x4 * (POW2(x0) * x5 + POW2(x3) * x5);
}
static SFEM_INLINE void tri3_fff_s(const scalar_t  px0,
                                   const scalar_t  px1,
                                   const scalar_t  px2,
                                   const scalar_t  py0,
                                   const scalar_t  py1,
                                   const scalar_t  py2,
                                   scalar_t *const fff) {
    const scalar_t x0 = -px0 + px1;
    const scalar_t x1 = -py0 + py2;
    const scalar_t x2 = px0 - px2;
    const scalar_t x3 = py0 - py1;
    const scalar_t x4 = x0 * x1 - x2 * x3;
    const scalar_t x5 = (1 / POW2(x4));
    fff[0]            = x4 * (POW2(x1) * x5 + POW2(x2) * x5);
    fff[1]            = x4 * (x0 * x2 * x5 + x1 * x3 * x5);
    fff[2]            = x4 * (POW2(x0) * x5 + POW2(x3) * x5);
}

static SFEM_INLINE scalar_t tri3_det_fff(const scalar_t *const fff) { return fff[0] * fff[2] - POW2(fff[1]); }

// static SFEM_INLINE void tri3_adjugate_and_det(const geom_t px0,
//                                               const geom_t px1,
//                                               const geom_t px2,
//                                               const geom_t px3,
//                                               const geom_t py0,
//                                               const geom_t py1,
//                                               const geom_t py2,
//                                               const geom_t py3,
//                                               const geom_t pz0,
//                                               const geom_t pz1,
//                                               const geom_t pz2,
//                                               const geom_t pz3,
//                                               jacobian_t *const SFEM_RESTRICT adjugate,
//                                               jacobian_t *const SFEM_RESTRICT
//                                                       jacobian_determinant) {

// }

#endif  // TRI3_INLINE_CPU_H
