#ifndef CVFEM_SCATTER_HPP
#define CVFEM_SCATTER_HPP

// WHAT THE SWEEPS DO WITH A RESULT ONCE THEY HAVE IT.
//
// Three operations the standard-layout sweeps use on every element and one they use once per
// apply, all of which were in the bench's staging header and none of which touches a mesh, a
// pack or a matrix object any more: the atomic scatter, the BSR slot search, the residual reset,
// and MIN.
//
// They are here because the kernels are their only callers, and because with them here
// src/kernels/standard/ includes nothing from outside this directory -- which was the point of
// converting those 27 sweeps.
#include <cstddef>   // ptrdiff_t

#include "kernels/cvfem_portability.hpp"

// MIN was defined twice -- once in this directory's packed header and once in the bench's
// staging header -- with the same body. One definition.
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

static void reset_residual(const ptrdiff_t nnodes,
                           scalar_t *const SFEM_RESTRICT rx,
                           scalar_t *const SFEM_RESTRICT ry,
                           scalar_t *const SFEM_RESTRICT rz,
                           scalar_t *const SFEM_RESTRICT rc) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        rx[i] = scalar_t(0);
        ry[i] = scalar_t(0);
        rz[i] = scalar_t(0);
        rc[i] = scalar_t(0);
    }
}

static SFEM_INLINE void atomic_add(scalar_t *const SFEM_RESTRICT f, const idx_t id, const scalar_t value) {
    CVFEM_ATOMIC_ADD(f[id], value);
}

static SFEM_INLINE count_t find_bsr_slot(const count_t *const SFEM_RESTRICT rowptr,
                                                const idx_t *const SFEM_RESTRICT   colidx,
                                                const idx_t                        row,
                                                const idx_t                        col) {
    const count_t begin = rowptr[row];
    const count_t end   = rowptr[row + 1];
    for (count_t k = begin; k < end; ++k) {
        if (colidx[k] == col) return k;
    }
    return begin;
}

#endif  // CVFEM_SCATTER_HPP
