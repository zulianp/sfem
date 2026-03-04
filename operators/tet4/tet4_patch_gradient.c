#include "tet4_patch_gradient.h"

#include "sfem_base.h"

#include <math.h>
#include <stddef.h>

inline static void cross(const real_t                a1,
                         const real_t                a2,
                         const real_t                a3,
                         const real_t                b1,
                         const real_t                b2,
                         const real_t                b3,
                         real_t* const SFEM_RESTRICT s1,
                         real_t* const SFEM_RESTRICT s2,
                         real_t* const SFEM_RESTRICT s3) {
    *s1 = a2 * b3 - a3 * b2;
    *s2 = a3 * b1 - a1 * b3;
    *s3 = a1 * b2 - a2 * b1;
}

inline static void grad(const real_t                xp,
                        const real_t                x1,
                        const real_t                x2,
                        const real_t                x3,
                        const real_t                yp,
                        const real_t                y1,
                        const real_t                y2,
                        const real_t                y3,
                        const real_t                zp,
                        const real_t                z1,
                        const real_t                z2,
                        const real_t                z3,
                        const real_t                up,
                        const real_t                u1,
                        const real_t                u2,
                        const real_t                u3,
                        real_t* const SFEM_RESTRICT gx,
                        real_t* const SFEM_RESTRICT gy,
                        real_t* const SFEM_RESTRICT gz) {
    // mundane ops: 121 divs: 1 sqrts: 0
    // total ops: 129
    const scalar_t x0  = -xp;
    const scalar_t x4  = x0 + x1;
    const scalar_t x5  = -yp;
    const scalar_t x6  = x5 + y2;
    const scalar_t x7  = -zp;
    const scalar_t x8  = x7 + z3;
    const scalar_t x9  = x5 + y3;
    const scalar_t x10 = x7 + z2;
    const scalar_t x11 = x7 + z1;
    const scalar_t x12 = x0 + x2;
    const scalar_t x13 = x0 + x3;
    const scalar_t x14 = x5 + y1;
    const scalar_t x15 = 1.0 / fabs(x11 * (x12 * x9 - x13 * x6) - x14 * (-x10 * x13 + x12 * x8) + x4 * (-x10 * x9 + x6 * x8));
    const scalar_t x16 = -y3;
    const scalar_t x17 = x16 + y2;
    const scalar_t x18 = -z3;
    const scalar_t x19 = x18 + z2;
    const scalar_t x20 = y1 - y2;
    const scalar_t x21 = x18 + z1;
    const scalar_t x22 = x16 + y1;
    const scalar_t x23 = z1 - z2;
    const scalar_t x24 = -x3;
    const scalar_t x25 = x2 + x24;
    const scalar_t x26 = x1 - x2;
    const scalar_t x27 = x1 + x24;
    *gx                = x15 *
          (u1 * (x10 * x17 - x19 * x6) - u2 * (x21 * x9 - x22 * x8) + u3 * (x10 * x14 - x11 * x6) + up * (x20 * x21 - x22 * x23));
    *gy = -x15 * (u1 * (x10 * x25 - x12 * x19) - u2 * (x13 * x21 - x27 * x8) + u3 * (x10 * x4 - x11 * x12) +
                  up * (x21 * x26 - x23 * x27));
    *gz = x15 * (u1 * (-x12 * x17 + x25 * x6) - u2 * (x13 * x22 - x27 * x9) + u3 * (-x12 * x14 + x4 * x6) +
                 up * (-x20 * x27 + x22 * x26));
}

int         tet4_patch_gradient(const ptrdiff_t                                        nelements,
                                const idx_t* const SFEM_RESTRICT* const SFEM_RESTRICT  elements,
                                const ptrdiff_t                                        nnodes,
                                const ptrdiff_t                                        max_indicence,
                                const count_t* const SFEM_RESTRICT                     n2e_ptr,
                                const element_idx_t* const SFEM_RESTRICT               n2e_idx,
                                const geom_t* const SFEM_RESTRICT* const SFEM_RESTRICT points,
                                const real_t* const SFEM_RESTRICT                      in,
                                const ptrdiff_t                                        out_stride,
                                real_t* const SFEM_RESTRICT                            outx,
                                real_t* const SFEM_RESTRICT                            outy,
                                real_t* const SFEM_RESTRICT                            outz) {
#pragma omp parallel
    {
        real_t* arena = malloc(sizeof(real_t) * max_indicence * 15);

        real_t* x1 = arena;
        real_t* y1 = &arena[max_indicence];
        real_t* z1 = &arena[2 * max_indicence];

        real_t* x2 = &arena[3 * max_indicence];
        real_t* y2 = &arena[4 * max_indicence];
        real_t* z2 = &arena[5 * max_indicence];

        real_t* x3 = &arena[6 * max_indicence];
        real_t* y3 = &arena[7 * max_indicence];
        real_t* z3 = &arena[8 * max_indicence];

        real_t* u1 = &arena[9 * max_indicence];
        real_t* u2 = &arena[10 * max_indicence];
        real_t* u3 = &arena[11 * max_indicence];

        real_t* gx = &arena[12 * max_indicence];
        real_t* gy = &arena[13 * max_indicence];
        real_t* gz = &arena[14 * max_indicence];

        real_t* xx[3] = {x1, x2, x3};
        real_t* yy[3] = {y1, y2, y3};
        real_t* zz[3] = {z1, z2, z3};
        real_t* uu[3] = {u1, u2, u3};
        real_t* gg[3] = {gx, gy, gz};

#pragma omp for
        for (ptrdiff_t node = 0; node < nnodes; node++) {
            const count_t              patch_begin = n2e_ptr[node];
            const count_t              patch_end   = n2e_ptr[node + 1];
            const count_t              extent      = patch_end - patch_begin;
            const element_idx_t* const ln2e_idx    = &n2e_idx[patch_begin];

            const real_t xp = points[0][node];
            const real_t yp = points[1][node];
            const real_t zp = points[2][node];
            const real_t up = in[node];

            for (count_t k = 0; k < extent; k++) {
                idx_t face[3];

                int f = 0;
                for (int v = 0; v < 4; v++) {
                    const idx_t node_k = elements[v][ln2e_idx[k]];
                    if (node_k == node) continue;
                    face[f++] = node_k;
                }

                for (int f = 0; f < 3; f++) {
                    const ptrdiff_t ff = face[f];

                    xx[f][k] = points[0][ff];
                    yy[f][k] = points[1][ff];
                    xx[f][k] = points[2][ff];
                    uu[f][k] = in[ff];
                }
            }

#pragma omp simd
            for (count_t k = 0; k < extent; k++) {
                grad(xp,
                     x1[k],
                     x2[k],
                     x3[k],
                     yp,
                     y1[k],
                     y2[k],
                     y3[k],
                     zp,
                     z1[k],
                     z2[k],
                     z3[k],
                     up,
                     u1[k],
                     u2[k],
                     u3[k],
                     &gx[k],
                     &gy[k],
                     &gz[k]);
            }

            real_t rxp = 0;
            real_t ryp = 0;
            real_t rzp = 0;
            for (count_t k = 0; k < extent; k++) {
                rxp += gx[k];
                ryp += gy[k];
                rzp += gz[k];
            }

            const ptrdiff_t idx = node * out_stride;
            outx[idx] += rxp;
            outy[idx] += ryp;
            outz[idx] += rzp;
        }

        free(arena);
    }

    return SFEM_SUCCESS;
}
