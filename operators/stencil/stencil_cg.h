#ifndef STENCIL_CG_H
#define STENCIL_CG_H

#include <math.h>
#include "sfem_base.h"
#include "stencil3.h"

int sshex8_copy_boundary(const ptrdiff_t                     xc,
                         const ptrdiff_t                     yc,
                         const ptrdiff_t                     zc,
                         const scalar_t *const SFEM_RESTRICT in,
                         scalar_t *const SFEM_RESTRICT       out) {
    const ptrdiff_t zstride = yc * xc;
    const ptrdiff_t ystride = xc;
    // Bottom
    for (int yi = 0; yi < yc; yi++) {
        for (int xi = 0; xi < xc; xi++) {
            const ptrdiff_t idx = yi * ystride + xi;
            out[idx]            = in[idx];
        }
    }

    // Top
    for (int yi = 0; yi < yc; yi++) {
        for (int xi = 0; xi < xc; xi++) {
            const ptrdiff_t idx = (zc - 1) * zstride + yi * ystride + xi;
            out[idx]            = in[idx];
        }
    }

    // Front
    for (int zi = 1; zi < zc - 1; zi++) {
        for (int xi = 0; xi < xc; xi++) {
            const ptrdiff_t idx = zi * zstride + 0 * ystride + xi;
            out[idx]            = in[idx];
        }
    }

    // Back
    for (int zi = 1; zi < zc - 1; zi++) {
        for (int xi = 0; xi < xc; xi++) {
            const ptrdiff_t idx = zi * zstride + (yc - 1) * ystride + xi;
            out[idx]            = in[idx];
        }
    }

    // Left
    for (int zi = 1; zi < zc - 1; zi++) {
        for (int yi = 1; yi < (yc - 1); yi++) {
            const ptrdiff_t idx = zi * zstride + yi * ystride + 0;
            out[idx]            = in[idx];
        }
    }

    // Right
    for (int zi = 1; zi < zc - 1; zi++) {
        for (int yi = 1; yi < (yc - 1); yi++) {
            const ptrdiff_t idx = zi * zstride + yi * ystride + xc - 1;
            out[idx]            = in[idx];
        }
    }

    return SFEM_SUCCESS;
}

int sshex8_zero_boundary(const ptrdiff_t xc, const ptrdiff_t yc, const ptrdiff_t zc, scalar_t *const SFEM_RESTRICT out) {
    const ptrdiff_t zstride = yc * xc;
    const ptrdiff_t ystride = xc;
    // Bottom
    for (int yi = 0; yi < yc; yi++) {
        for (int xi = 0; xi < xc; xi++) {
            const ptrdiff_t idx = yi * ystride + xi;
            out[idx]            = 0;
        }
    }

    // Top
    for (int yi = 0; yi < yc; yi++) {
        for (int xi = 0; xi < xc; xi++) {
            const ptrdiff_t idx = (zc - 1) * zstride + yi * ystride + xi;
            out[idx]            = 0;
        }
    }

    // Front
    for (int zi = 1; zi < zc - 1; zi++) {
        for (int xi = 0; xi < xc; xi++) {
            const ptrdiff_t idx = zi * zstride + 0 * ystride + xi;
            out[idx]            = 0;
        }
    }

    // Back
    for (int zi = 1; zi < zc - 1; zi++) {
        for (int xi = 0; xi < xc; xi++) {
            const ptrdiff_t idx = zi * zstride + (yc - 1) * ystride + xi;
            out[idx]            = 0;
        }
    }

    // Left
    for (int zi = 1; zi < zc - 1; zi++) {
        for (int yi = 1; yi < (yc - 1); yi++) {
            const ptrdiff_t idx = zi * zstride + yi * ystride + 0;
            out[idx]            = 0;
        }
    }

    // Right
    for (int zi = 1; zi < zc - 1; zi++) {
        for (int yi = 1; yi < (yc - 1); yi++) {
            const ptrdiff_t idx = zi * zstride + yi * ystride + xc - 1;
            out[idx]            = 0;
        }
    }

    return SFEM_SUCCESS;
}

int sshex8_stencil_cg(const int      max_it,
                      const scalar_t rtol,
                      const scalar_t atol,
                      // Grid info
                      const ptrdiff_t                     xc,
                      const ptrdiff_t                     yc,
                      const ptrdiff_t                     zc,
                      const scalar_t *const SFEM_RESTRICT stencil,
                      const scalar_t *const SFEM_RESTRICT b,
                      // Temps
                      scalar_t *const SFEM_RESTRICT r,
                      scalar_t *const SFEM_RESTRICT p,
                      scalar_t *const SFEM_RESTRICT Ap,
                      // Out
                      scalar_t *const x) {
    const ptrdiff_t size = xc * yc * zc;
    memset(r, 0, size * sizeof(scalar_t));
    sshex8_stencil(xc, yc, zc, stencil, x, r);

    for (ptrdiff_t i = 0; i < size; i++) {
        r[i] = b[i] - r[i];
    }

    scalar_t rtr0 = 0;
    for (ptrdiff_t i = 0; i < size; i++) {
        rtr0 += r[i] * r[i];
    }

    const scalar_t r_norm0 = sqrt(rtr0);

    scalar_t rtr = rtr0;
    if (rtr0 == 0) {
        return SFEM_SUCCESS;
    }

    for (ptrdiff_t i = 0; i < size; i++) {
        p[i] = r[i];
    }

    int info = SFEM_FAILURE;
    for (int iterations = 0; iterations < max_it; iterations++) {
        memset(Ap, 0, size * sizeof(scalar_t));

        sshex8_stencil(xc, yc, zc, stencil, p, Ap);

        scalar_t ptAp = 0;
        for (ptrdiff_t i = 0; i < size; i++) {
            ptAp += p[i] * Ap[i];
        }

        if (ptAp == 0) {
            // printf("%d %g\n", iterations, r_norm0);
            // printf("----------------------------\n");
            // printf("B\n");
            // printf("----------------------------\n");
            // for (ptrdiff_t zi = 0; zi < zc; zi++) {
            //     for (ptrdiff_t yi = 0; yi < yc; yi++) {
            //         for (ptrdiff_t xi = 0; xi < xc; xi++) {
            //             ptrdiff_t idx = zi * yc * xc + yi * xc + xi;
            //             printf("%g ", b[idx]);
            //         }
            //         printf("\t");
            //     }
            //     printf("\n");
            // }
            // printf("----------------------------\n");

            // printf("----------------------------\n");
            // printf("X\n");
            // printf("----------------------------\n");
            // for (ptrdiff_t zi = 0; zi < zc; zi++) {
            //     for (ptrdiff_t yi = 0; yi < yc; yi++) {
            //         for (ptrdiff_t xi = 0; xi < xc; xi++) {
            //             ptrdiff_t idx = zi * yc * xc + yi * xc + xi;
            //             printf("%g ", r[idx]);
            //         }
            //         printf("\t");
            //     }
            //     printf("\n");
            // }
            // printf("----------------------------\n");
            info = SFEM_FAILURE;
            break;
        }

        const scalar_t alpha = rtr / ptAp;

        for (ptrdiff_t i = 0; i < size; i++) {
            x[i] += alpha * p[i];
        }

        for (ptrdiff_t i = 0; i < size; i++) {
            r[i] -= alpha * Ap[i];
        }

        assert(rtr != 0);

        scalar_t rtr_new = 0;
        for (ptrdiff_t i = 0; i < size; i++) {
            rtr_new += r[i] * r[i];
        }
        const scalar_t beta = rtr_new / rtr;
        rtr                 = rtr_new;

        for (ptrdiff_t i = 0; i < size; i++) {
            p[i] = beta * p[i] + r[i];
        }

        scalar_t r_norm = sqrt(rtr_new);

        // printf("%d %g\n", iterations, r_norm);
        if (r_norm < atol || rtr_new == 0 || r_norm / r_norm0 < rtol) {
            info = SFEM_SUCCESS;
            break;
        }
    }

    return info;
}

int sshex8_stencil_cg_constrained(const int      max_it,
                                  const scalar_t rtol,
                                  const scalar_t atol,
                                  // Grid info
                                  const ptrdiff_t                     xc,
                                  const ptrdiff_t                     yc,
                                  const ptrdiff_t                     zc,
                                  const scalar_t *const SFEM_RESTRICT stencil,
                                  const int *const                    constraints,
                                  const scalar_t *const SFEM_RESTRICT b,
                                  // Temps
                                  scalar_t *const SFEM_RESTRICT r,
                                  scalar_t *const SFEM_RESTRICT p,
                                  scalar_t *const SFEM_RESTRICT Ap,
                                  // Out
                                  scalar_t *const x) {
    const ptrdiff_t size = xc * yc * zc;
    memset(r, 0, size * sizeof(scalar_t));
    sshex8_stencil(xc, yc, zc, stencil, x, r);

    for (ptrdiff_t i = 0; i < size; i++) {
        if(constraints[i]) {
            r[i] = x[i];
        }
    }

    for (ptrdiff_t i = 0; i < size; i++) {
        r[i] = b[i] - r[i];
    }

    scalar_t rtr0 = 0;
    for (ptrdiff_t i = 0; i < size; i++) {
        rtr0 += r[i] * r[i];
    }

    assert(rtr0 == rtr0);

    const scalar_t r_norm0 = sqrt(rtr0);

    scalar_t rtr = rtr0;
    if (rtr0 == 0) {
        return SFEM_SUCCESS;
    }

    assert(rtr == rtr);

    for (ptrdiff_t i = 0; i < size; i++) {
        p[i] = r[i];
    }

    int info = SFEM_FAILURE;
    for (int iterations = 0; iterations < max_it; iterations++) {
        memset(Ap, 0, size * sizeof(scalar_t));

        sshex8_stencil(xc, yc, zc, stencil, p, Ap);

        for (ptrdiff_t i = 0; i < size; i++) {
            if(constraints[i]) {
                Ap[i] = p[i];
            }
        }


        scalar_t ptAp = 0;
        for (ptrdiff_t i = 0; i < size; i++) {
            ptAp += p[i] * Ap[i];
        }

        if (ptAp == 0) {
            info = SFEM_FAILURE;
            break;
        }

        const scalar_t alpha = rtr / ptAp;

        for (ptrdiff_t i = 0; i < size; i++) {
            x[i] += alpha * p[i];
        }

        for (ptrdiff_t i = 0; i < size; i++) {
            r[i] -= alpha * Ap[i];
        }

        assert(rtr != 0);

        scalar_t rtr_new = 0;
        for (ptrdiff_t i = 0; i < size; i++) {
            rtr_new += r[i] * r[i];
        }
        const scalar_t beta = rtr_new / rtr;
        rtr                 = rtr_new;

        for (ptrdiff_t i = 0; i < size; i++) {
            p[i] = beta * p[i] + r[i];
        }

        scalar_t r_norm = sqrt(rtr_new);

        // printf("%d %g\n", iterations, r_norm);
        if (r_norm < atol || rtr_new == 0 || r_norm / r_norm0 < rtol) {
            info = SFEM_SUCCESS;
            break;
        }
    }

    return info;
}

#endif  // STENCIL_CG_H
