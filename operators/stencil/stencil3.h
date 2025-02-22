#ifndef STENCIL_3_H
#define STENCIL_3_H

#include "sfem_base.h"
#include "stencil2.h"

static void slice_stencil_3x3x3(const ptrdiff_t                   xc,
                                const ptrdiff_t                   yc,
                                const ptrdiff_t                   zc,
                                const real_t *const SFEM_RESTRICT s,
                                const ptrdiff_t                   in_ystride,
                                const ptrdiff_t                   in_zstride,
                                const real_t *const SFEM_RESTRICT in,
                                const ptrdiff_t                   out_ystride,
                                const ptrdiff_t                   out_zstride,
                                real_t *const SFEM_RESTRICT       out) {
    // TODO tiling
    for (ptrdiff_t zi = 0; zi < zc; zi++) {
        for (int d = 0; d < 3; d++) {
            slice_stencil_3x3(xc,
                              yc,
                              s[d * 9 + 0 * 3 + 0],
                              s[d * 9 + 0 * 3 + 1],
                              s[d * 9 + 0 * 3 + 2],
                              s[d * 9 + 1 * 3 + 0],
                              s[d * 9 + 1 * 3 + 1],
                              s[d * 9 + 1 * 3 + 2],
                              s[d * 9 + 2 * 3 + 0],
                              s[d * 9 + 2 * 3 + 1],
                              s[d * 9 + 2 * 3 + 2],
                              in_ystride,
                              &in[(zi + d) * in_zstride],
                              out_ystride,
                              &out[zi * out_zstride]);
        }
    }
}

static void par_slice_stencil_3x3x3(const ptrdiff_t                   xc,
                                    const ptrdiff_t                   yc,
                                    const ptrdiff_t                   zc,
                                    const real_t *const SFEM_RESTRICT s,
                                    const ptrdiff_t                   in_ystride,
                                    const ptrdiff_t                   in_zstride,
                                    const real_t *const SFEM_RESTRICT in,
                                    const ptrdiff_t                   out_ystride,
                                    const ptrdiff_t                   out_zstride,
                                    real_t *const SFEM_RESTRICT       out) {
    // TODO tiling

#pragma omp parallel for proc_bind(spread)  // collapse(2)
    for (ptrdiff_t zi = 0; zi < zc; zi++) {
        for (int d = 0; d < 3; d++) {
            slice_stencil_3x3(xc,
                              yc,
                              s[d * 9 + 0 * 3 + 0],
                              s[d * 9 + 0 * 3 + 1],
                              s[d * 9 + 0 * 3 + 2],
                              s[d * 9 + 1 * 3 + 0],
                              s[d * 9 + 1 * 3 + 1],
                              s[d * 9 + 1 * 3 + 2],
                              s[d * 9 + 2 * 3 + 0],
                              s[d * 9 + 2 * 3 + 1],
                              s[d * 9 + 2 * 3 + 2],
                              in_ystride,
                              &in[(zi + d) * in_zstride],
                              out_ystride,
                              &out[zi * out_zstride]);
        }
    }
}

#endif  // STENCIL_3_H
