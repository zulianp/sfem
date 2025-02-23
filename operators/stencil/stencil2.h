#ifndef STENCIL_2_H
#define STENCIL_2_H

#include "sfem_base.h"
#include "sfem_macros.h"

static void slice_stencil_9(const ptrdiff_t                   n,
                            const real_t                      s00,
                            const real_t                      s01,
                            const real_t                      s02,
                            const real_t                      s10,
                            const real_t                      s11,
                            const real_t                      s12,
                            const real_t                      s20,
                            const real_t                      s21,
                            const real_t                      s22,
                            const real_t *const SFEM_RESTRICT in00,
                            const real_t *const SFEM_RESTRICT in01,
                            const real_t *const SFEM_RESTRICT in02,
                            const real_t *const SFEM_RESTRICT in10,
                            const real_t *const SFEM_RESTRICT in11,
                            const real_t *const SFEM_RESTRICT in12,
                            const real_t *const SFEM_RESTRICT in20,
                            const real_t *const SFEM_RESTRICT in21,
                            const real_t *const SFEM_RESTRICT in22,
                            real_t *const SFEM_RESTRICT       out) {
    for (ptrdiff_t i = 0; i < n; i++) {
        out[i] += s00 * in00[i] +  //
                  s01 * in01[i] +  //
                  s02 * in02[i] +  //
                  s10 * in10[i] +  //
                  s11 * in11[i] +  //
                  s12 * in12[i] +  //
                  s20 * in20[i] +  //
                  s21 * in21[i] +  //
                  s22 * in22[i];
    }
}

#ifdef SFEM_ENABLE_STENCIL_TILING

static void slice_stencil_3x3(const ptrdiff_t                   xc,
                              const ptrdiff_t                   yc,
                              const real_t                      s00,
                              const real_t                      s01,
                              const real_t                      s02,
                              const real_t                      s10,
                              const real_t                      s11,
                              const real_t                      s12,
                              const real_t                      s20,
                              const real_t                      s21,
                              const real_t                      s22,
                              const ptrdiff_t                   in_stride,
                              const real_t *const SFEM_RESTRICT in,
                              const ptrdiff_t                   out_stride,
                              real_t *const SFEM_RESTRICT       out) {
    ptrdiff_t SFEM_STENCIL2_FRAGMENT = 16;
    // SFEM_READ_ENV(SFEM_STENCIL2_FRAGMENT, atol);

    for (ptrdiff_t yoffset = 0; yoffset < yc; yoffset += SFEM_STENCIL2_FRAGMENT) {
        const ptrdiff_t yc_frag = MIN(SFEM_STENCIL2_FRAGMENT, yc - yoffset);

        for (ptrdiff_t xoffset = 0; xoffset < xc; xoffset += SFEM_STENCIL2_FRAGMENT) {
            const ptrdiff_t xc_frag = MIN(SFEM_STENCIL2_FRAGMENT, xc - xoffset);

            for (ptrdiff_t yi = 0; yi < yc_frag; yi++) {
                slice_stencil_9(xc_frag,
                                s00,
                                s01,
                                s02,
                                s10,
                                s11,
                                s12,
                                s20,
                                s21,
                                s22,
                                // Input
                                &in[xoffset + 0 + (yoffset + yi + 0) * in_stride],
                                &in[xoffset + 1 + (yoffset + yi + 0) * in_stride],
                                &in[xoffset + 2 + (yoffset + yi + 0) * in_stride],

                                &in[xoffset + 0 + (yoffset + yi + 1) * in_stride],
                                &in[xoffset + 1 + (yoffset + yi + 1) * in_stride],
                                &in[xoffset + 2 + (yoffset + yi + 1) * in_stride],

                                &in[xoffset + 0 + (yoffset + yi + 2) * in_stride],
                                &in[xoffset + 1 + (yoffset + yi + 2) * in_stride],
                                &in[xoffset + 2 + (yoffset + yi + 2) * in_stride],
                                // Output
                                &out[xoffset + 0 + (yoffset + yi + 0) * out_stride]);
            }
        }
    }
}
#else

static void slice_stencil_3x3(const ptrdiff_t                   xc,
                              const ptrdiff_t                   yc,
                              const real_t                      s00,
                              const real_t                      s01,
                              const real_t                      s02,
                              const real_t                      s10,
                              const real_t                      s11,
                              const real_t                      s12,
                              const real_t                      s20,
                              const real_t                      s21,
                              const real_t                      s22,
                              const ptrdiff_t                   in_stride,
                              const real_t *const SFEM_RESTRICT in,
                              const ptrdiff_t                   out_stride,
                              real_t *const SFEM_RESTRICT       out) {
    // TODO experiment with tiling
    for (ptrdiff_t yi = 0; yi < yc; yi++) {
        slice_stencil_9(xc,
                        s00,
                        s01,
                        s02,
                        s10,
                        s11,
                        s12,
                        s20,
                        s21,
                        s22,
                        // Input
                        &in[0 + (yi + 0) * in_stride],
                        &in[1 + (yi + 0) * in_stride],
                        &in[2 + (yi + 0) * in_stride],

                        &in[0 + (yi + 1) * in_stride],
                        &in[1 + (yi + 1) * in_stride],
                        &in[2 + (yi + 1) * in_stride],

                        &in[0 + (yi + 2) * in_stride],
                        &in[1 + (yi + 2) * in_stride],
                        &in[2 + (yi + 2) * in_stride],
                        // Output
                        &out[0 + (yi + 0) * out_stride]);
    }
}
#endif

#endif  // STENCIL_2_H
