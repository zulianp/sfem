#ifndef STENCIL_3_H
#define STENCIL_3_H

#include "sfem_base.h"
#include "stencil2.h"

// #define SFEM_ENABLE_STENCIL_TILING

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

#ifdef SFEM_ENABLE_STENCIL_TILING

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
    
    ptrdiff_t SFEM_STENCIL3_Z_FRAGMENT = 1;
    ptrdiff_t SFEM_STENCIL3_Y_FRAGMENT = 16;
    SFEM_READ_ENV(SFEM_STENCIL3_Z_FRAGMENT, atol);
    SFEM_READ_ENV(SFEM_STENCIL3_Y_FRAGMENT, atol);

#pragma omp parallel for proc_bind(spread) collapse(2)
    for (ptrdiff_t zoffset = 0; zoffset < zc; zoffset += SFEM_STENCIL3_Z_FRAGMENT) {
        const ptrdiff_t zc_frag = MIN(SFEM_STENCIL3_Z_FRAGMENT, zc - zoffset);

        for (ptrdiff_t yoffset = 0; yoffset < yc; yoffset += SFEM_STENCIL3_Y_FRAGMENT) {
            const ptrdiff_t yc_frag = MIN(SFEM_STENCIL3_Y_FRAGMENT, yc - yoffset);

            for (ptrdiff_t zi = 0; zi < zc_frag; zi++) {
                for (int d = 0; d < 3; d++) {
                    slice_stencil_3x3(xc,
                                      yc_frag,
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
                                      &in[(zoffset + zi + d) * in_zstride + yoffset * in_ystride],
                                      out_ystride,
                                      &out[(zoffset + zi) * out_zstride + yoffset * out_ystride]);
                }
            }
        }
    }
}

#else

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

#endif  //SFEM_ENABLE_STENCIL_TILING
#endif  // STENCIL_3_H
