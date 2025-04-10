#include "sshex8_mass.h"

#include "sfem_defs.h"

#include "hex8_inline_cpu.h"
#include "hex8_mass_inline_cpu.h"
#include "hex8_quadrature.h"
#include "line_quadrature.h"
#include "sshex8.h"

extern int affine_sshex8_mass_lumped(const int                    level,
                                     const ptrdiff_t              nelements,
                                     ptrdiff_t                    interior_start,
                                     idx_t **const SFEM_RESTRICT  elements,
                                     geom_t **const SFEM_RESTRICT std_hex8_points,
                                     real_t *const SFEM_RESTRICT  diag) {
    const int nxe = sshex8_nxe(level);
    const int txe = sshex8_txe(level);

    const int proteus_to_std_hex8_corners[8] = {// Bottom
                                                sshex8_lidx(level, 0, 0, 0),
                                                sshex8_lidx(level, level, 0, 0),
                                                sshex8_lidx(level, level, level, 0),
                                                sshex8_lidx(level, 0, level, 0),

                                                // Top
                                                sshex8_lidx(level, 0, 0, level),
                                                sshex8_lidx(level, level, 0, level),
                                                sshex8_lidx(level, level, level, level),
                                                sshex8_lidx(level, 0, level, level)};

    int             n_qp = q27_n;
    const scalar_t *qx   = q27_x;
    const scalar_t *qy   = q27_y;
    const scalar_t *qz   = q27_z;
    const scalar_t *qw   = q27_w;

#pragma omp parallel
    {
        // Allocation per thread
        idx_t         *ev = malloc(nxe * sizeof(idx_t));
        accumulator_t *v  = malloc(nxe * sizeof(accumulator_t));

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            scalar_t x[8];
            scalar_t y[8];
            scalar_t z[8];

            {
                // Gather elemental data
                for (int d = 0; d < nxe; d++) {
                    ev[d] = elements[d][e];
                }

                for (int d = 0; d < 8; d++) {
                    x[d] = std_hex8_points[0][ev[proteus_to_std_hex8_corners[d]]];
                    y[d] = std_hex8_points[1][ev[proteus_to_std_hex8_corners[d]]];
                    z[d] = std_hex8_points[2][ev[proteus_to_std_hex8_corners[d]]];
                }

                memset(v, 0, nxe * sizeof(accumulator_t));
            }

            const scalar_t h = 1. / level;

            accumulator_t mass_lumped[8] = {0};
            for (int k = 0; k < n_qp; k++) {
                hex8_lumped_mass_points(x, y, z, qx[k], qy[k], qz[k], qw[k], mass_lumped);
            }

            for(int v = 0; v < 8; v++) {
                mass_lumped[v] *= (h * h * h);
            }

            // Iterate over sub-elements
            for (int zi = 0; zi < level; zi++) {
                for (int yi = 0; yi < level; yi++) {
                    for (int xi = 0; xi < level; xi++) {
                        // Convert to standard HEX8 local ordering (see 3-4 and 6-7)
                        int lev[8] = {// Bottom
                                      sshex8_lidx(level, xi, yi, zi),
                                      sshex8_lidx(level, xi + 1, yi, zi),
                                      sshex8_lidx(level, xi + 1, yi + 1, zi),
                                      sshex8_lidx(level, xi, yi + 1, zi),
                                      // Top
                                      sshex8_lidx(level, xi, yi, zi + 1),
                                      sshex8_lidx(level, xi + 1, yi, zi + 1),
                                      sshex8_lidx(level, xi + 1, yi + 1, zi + 1),
                                      sshex8_lidx(level, xi, yi + 1, zi + 1)};

                        // Accumulate to macro-element buffer
                        for (int d = 0; d < 8; d++) {
                            v[lev[d]] += mass_lumped[d];
                        }
                    }
                }
            }

            {
                // Scatter elemental data
                for (int d = 0; d < nxe; d++) {
                    assert(v[d] == v[d]);
#pragma omp atomic update
                    diag[ev[d]] += v[d];
                }
            }
        }

        // Clean-up
        free(ev);
        free(v);
    }

    return SFEM_SUCCESS;
}