#ifndef STENCIL_3_H
#define STENCIL_3_H

#include "sfem_base.h"
#include "stencil2.h"

// #define SFEM_ENABLE_STENCIL_TILING

static void slice_stencil_3x3x3(const ptrdiff_t                     xc,
                                const ptrdiff_t                     yc,
                                const ptrdiff_t                     zc,
                                const scalar_t *const SFEM_RESTRICT s,
                                const ptrdiff_t                     in_ystride,
                                const ptrdiff_t                     in_zstride,
                                const scalar_t *const SFEM_RESTRICT in,
                                const ptrdiff_t                     out_ystride,
                                const ptrdiff_t                     out_zstride,
                                scalar_t *const SFEM_RESTRICT       out) {
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

static void par_slice_stencil_3x3x3(const ptrdiff_t                     xc,
                                    const ptrdiff_t                     yc,
                                    const ptrdiff_t                     zc,
                                    const scalar_t *const SFEM_RESTRICT s,
                                    const ptrdiff_t                     in_ystride,
                                    const ptrdiff_t                     in_zstride,
                                    const scalar_t *const SFEM_RESTRICT in,
                                    const ptrdiff_t                     out_ystride,
                                    const ptrdiff_t                     out_zstride,
                                    scalar_t *const SFEM_RESTRICT       out) {
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

static void par_slice_stencil_3x3x3(const ptrdiff_t                     xc,
                                    const ptrdiff_t                     yc,
                                    const ptrdiff_t                     zc,
                                    const scalar_t *const SFEM_RESTRICT s,
                                    const ptrdiff_t                     in_ystride,
                                    const ptrdiff_t                     in_zstride,
                                    const scalar_t *const SFEM_RESTRICT in,
                                    const ptrdiff_t                     out_ystride,
                                    const ptrdiff_t                     out_zstride,
                                    scalar_t *const SFEM_RESTRICT       out) {
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

static void hex8_side_stencil(const ptrdiff_t                     xc,
                              const ptrdiff_t                     yc,
                              const ptrdiff_t                     zc,
                              const ptrdiff_t                     xstride,
                              const ptrdiff_t                     ystride,
                              const ptrdiff_t                     zstride,
                              const int                           v0,
                              const int                           v1,
                              const int                           v2,
                              const int                           v3,
                              const scalar_t *const SFEM_RESTRICT A,
                              const scalar_t *const SFEM_RESTRICT input,
                              scalar_t *const SFEM_RESTRICT       output) {
    for (ptrdiff_t zi = 0; zi < zc; zi++) {
        for (ptrdiff_t yi = 0; yi < yc; yi++) {
            for (ptrdiff_t xi = 0; xi < xc; xi++) {
                scalar_t acc[4] = {0, 0, 0, 0};

                idx_t v[8];
                v[0] = xi * xstride + yi * ystride + zi * zstride;
                v[1] = (xi + 1) * xstride + yi * ystride + zi * zstride;
                v[2] = (xi + 1) * xstride + (yi + 1) * ystride + zi * zstride;
                v[3] = xi * xstride + (yi + 1) * ystride + zi * zstride;

                v[4] = xi * xstride + yi * ystride + (zi + 1) * zstride;
                v[5] = (xi + 1) * xstride + yi * ystride + (zi + 1) * zstride;
                v[6] = (xi + 1) * xstride + (yi + 1) * ystride + (zi + 1) * zstride;
                v[7] = xi * xstride + (yi + 1) * ystride + (zi + 1) * zstride;

                for (int i = 0; i < 8; i++) {
                    const scalar_t val = input[v[i]];
                    acc[0] += A[v0 * 8 + i] * val;
                    acc[1] += A[v1 * 8 + i] * val;
                    acc[2] += A[v2 * 8 + i] * val;
                    acc[3] += A[v3 * 8 + i] * val;
                }

                output[v[v0]] += acc[0];
                output[v[v1]] += acc[1];
                output[v[v2]] += acc[2];
                output[v[v3]] += acc[3];
            }
        }
    }
}

static void sshex8_stencil(const ptrdiff_t                     xc,
                           const ptrdiff_t                     yc,
                           const ptrdiff_t                     zc,
                           const scalar_t *const SFEM_RESTRICT stencil,
                           const scalar_t *const SFEM_RESTRICT input,
                           scalar_t *const                     output) {
    slice_stencil_3x3x3(xc - 2,
                        yc - 2,
                        zc - 2,
                        stencil,  //
                        xc,
                        xc * yc,
                        input,
                        //
                        xc,
                        xc * yc,
                        &output[1 + 1 * xc + 1 * xc * yc]);
}

static void par_sshex8_stencil(const ptrdiff_t                     xc,
                               const ptrdiff_t                     yc,
                               const ptrdiff_t                     zc,
                               const scalar_t *const SFEM_RESTRICT stencil,
                               const scalar_t *const SFEM_RESTRICT input,
                               scalar_t *const                     output) {
    par_slice_stencil_3x3x3(xc - 2,
                            yc - 2,
                            zc - 2,
                            stencil,  //
                            xc,
                            xc * yc,
                            input,
                            //
                            xc,
                            xc * yc,
                            &output[1 + 1 * xc + 1 * xc * yc]);
}

static void hex8_matrix_to_stencil(const scalar_t *const SFEM_RESTRICT A, scalar_t *const SFEM_RESTRICT s) {
    s[0]  = A[48];
    s[1]  = A[49] + A[56];
    s[2]  = A[57];
    s[3]  = A[40] + A[51];
    s[4]  = A[32] + A[41] + A[50] + A[59];
    s[5]  = A[33] + A[58];
    s[6]  = A[43];
    s[7]  = A[35] + A[42];
    s[8]  = A[34];
    s[9]  = A[16] + A[52];
    s[10] = A[17] + A[24] + A[53] + A[60];
    s[11] = A[25] + A[61];
    s[12] = A[19] + A[44] + A[55] + A[8];
    s[13] = A[0] + A[18] + A[27] + A[36] + A[45] + A[54] + A[63] + A[9];
    s[14] = A[1] + A[26] + A[37] + A[62];
    s[15] = A[11] + A[47];
    s[16] = A[10] + A[39] + A[3] + A[46];
    s[17] = A[2] + A[38];
    s[18] = A[20];
    s[19] = A[21] + A[28];
    s[20] = A[29];
    s[21] = A[12] + A[23];
    s[22] = A[13] + A[22] + A[31] + A[4];
    s[23] = A[30] + A[5];
    s[24] = A[15];
    s[25] = A[14] + A[7];
    s[26] = A[6];
}

#endif  // SFEM_ENABLE_STENCIL_TILING
#endif  // STENCIL_3_H
