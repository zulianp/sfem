#include "proteus_hex8_laplacian.h"

#include "hex8_inline_cpu.h"
#include "hex8_laplacian_inline_cpu.h"

#include "hex8_quadrature.h"

#include <math.h>

int proteus_hex8_nxe(int level) {
    const int corners = 8;
    const int edge_nodes = 12 * (level - 1);
    const int face_nodes = 6 * (level - 1) * (level - 1);
    const int vol_nodes = (level - 1) * (level - 1) * (level - 1);
    return corners + edge_nodes + face_nodes + vol_nodes;
}

int proteus_hex8_txe(int level) { return level * level * level; }

static inline int proteus_hex8_lidx(const int L, const int x, const int y, const int z) {
    int Lp1 = L + 1;
    int ret = z * (Lp1 * Lp1) + y * Lp1 + x;
    assert(ret < proteus_hex8_nxe(L));
    assert(ret >= 0);
    return ret;
}

int proteus_hex8_laplacian_apply(const int level,
                                 const ptrdiff_t nelements,
                                 idx_t **const SFEM_RESTRICT elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT values) {
    const int nxe = proteus_hex8_nxe(level);
    const int txe = proteus_hex8_txe(level);

    const int proteus_corners[8] = {// Bottom
                                    proteus_hex8_lidx(level, 0, 0, 0),
                                    proteus_hex8_lidx(level, level, 0, 0),
                                    proteus_hex8_lidx(level, 0, level, 0),
                                    proteus_hex8_lidx(level, level, level, 0),

                                    // Top
                                    proteus_hex8_lidx(level, 0, 0, level),
                                    proteus_hex8_lidx(level, level, 0, level),
                                    proteus_hex8_lidx(level, 0, level, level),
                                    proteus_hex8_lidx(level, level, level, level)};

    int n_qp = q27_n;
    const scalar_t *qx = q27_x;
    const scalar_t *qy = q27_y;
    const scalar_t *qz = q27_z;
    const scalar_t *qw = q27_w;

#pragma omp parallel
    {
        scalar_t *eu = malloc(nxe * sizeof(scalar_t));
        idx_t *ev = = malloc(nxe * sizeof(idx_t));
        accumulator_t *v = malloc(nxe * sizeof(accumulator_t));

        scalar_t x[8];
        scalar_t y[8];
        scalar_t z[8];

        scalar_t refx[2];
        scalar_t refy[2];
        scalar_t refz[2];

        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];

        scalar_t fff[6];

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            {
                // Gather elemental data
                for (int d = 0; d < nxe; d++) {
                    ev[d] = elements[d][e];
                }

                for (int d = 0; d < 8; d++) {
                    x[d] = points[0][ev[proteus_corners[d]]];
                    y[d] = points[1][ev[proteus_corners[d]]];
                    z[d] = points[2][ev[proteus_corners[d]]];
                }

                for (int d = 0; d < nxe; d++) {
                    eu[d] = u[ev[d]];
                }

                memset(v, 0, nxe * sizeof(accumulator_t));
            }

            // Iterate over sub-elements
            for (int zi = 0; zi < level - 1; zi++) {
                refz[0] = (scalar_t)zi / level;
                refz[1] = (scalar_t)(zi + 1) / level;

                for (int yi = 0; yi < level - 1; yi++) {
                    refy[0] = (scalar_t)yi / level;
                    refy[1] = (scalar_t)(yi + 1) / level;

                    for (int xi = 0; xi < level - 1; xi++) {
                        refx[0] = (scalar_t)xi / level;
                        refx[1] = (scalar_t)(xi + 1) / level;

                        accumulator_t element_vector[8] = {0};

                        // Use standard local ordering (see 3-4 and 6-7)
                        scalar_t element_u[8] = {// Bottom
                                                 eu[proteus_hex8_lidx(level, xi, yi, zi)],
                                                 eu[proteus_hex8_lidx(level, xi + 1, yi, zi)],
                                                 eu[proteus_hex8_lidx(level, xi + 1, yi + 1, zi)],
                                                 eu[proteus_hex8_lidx(level, xi, yi + 1, zi)],
                                                 // Top
                                                 eu[proteus_hex8_lidx(level, xi, yi, zi + 1)],
                                                 eu[proteus_hex8_lidx(level, xi + 1, yi, zi + 1)],
                                                 eu[proteus_hex8_lidx(level, xi + 1, yi + 1, zi + 1)],
                                                 eu[proteus_hex8_lidx(level, xi, yi + 1, zi + 1)]
                        };

                        // TODO compute lx, ly, lz

                        // Quadrature
                        for (int k = 0; k < n_qp; k++) {
                            hex8_laplacian_apply_points(lx,
                                                        ly,
                                                        lz,
                                                        qx[k],
                                                        qy[k],
                                                        qz[k],
                                                        qw[k],
                                                        element_u,
                                                        element_vector);
                        }
                    }
                }
            }

            {
                // Scatter elemental data
            }
        }

        free(ev);
        free(eu);
        free(v);
    }

    return SFEM_SUCCESS;
}
