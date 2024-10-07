#include "proteus_hex8_interpolate.h"

#include "proteus_hex8.h"

int proteus_hex8_hierarchical_restriction(int level,
                                          const ptrdiff_t nelements,
                                          idx_t **const SFEM_RESTRICT elements,
                                          const uint16_t *const SFEM_RESTRICT
                                                  element_to_node_incidence_count,
                                          const int vec_size,
                                          const real_t *const SFEM_RESTRICT from,
                                          real_t *const SFEM_RESTRICT to) {
#pragma omp parallel
    {
        const int nxe = proteus_hex8_nxe(level);
        scalar_t *e_from = malloc(nxe * sizeof(scalar_t));
        idx_t *ev = malloc(nxe * sizeof(idx_t));
        scalar_t element_vector[8];

        const int corners[8] = {// Bottom
                                proteus_hex8_lidx(level, 0, 0, 0),
                                proteus_hex8_lidx(level, level, 0, 0),
                                proteus_hex8_lidx(level, level, level, 0),
                                proteus_hex8_lidx(level, 0, level, 0),
                                // Top
                                proteus_hex8_lidx(level, 0, 0, level),
                                proteus_hex8_lidx(level, level, 0, level),
                                proteus_hex8_lidx(level, level, level, level),
                                proteus_hex8_lidx(level, 0, level, level)};

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; e++) {
            {
                // Gather elemental data
                for (int d = 0; d < nxe; d++) {
                    ev[d] = elements[d][e];
                }

                for (int d = 0; d < nxe; d++) {
                    e_from[d] = from[ev[d]];
                    assert(e_from[d] == e_from[d]);
                }

                for (int d = 0; d < 8; d++) {
                    element_vector[d] = 0;
                }

                const scalar_t h = 1. / level;
                // Iterate over structrued grid (nodes)
                for (int zi = 0; zi < level + 1; zi++) {
                    for (int yi = 0; yi < level + 1; yi++) {
                        for (int xi = 0; xi < level + 1; xi++) {
                            int lidx = proteus_hex8_lidx(level, xi, yi, zi);

                            const scalar_t x = xi * h;
                            const scalar_t y = yi * h;
                            const scalar_t z = zi * h;

                            // Evaluate Hex8 basis functions at x, y, z
                            const scalar_t xm = (1 - x);
                            const scalar_t ym = (1 - y);
                            const scalar_t zm = (1 - z);

                            scalar_t f[8];
                            f[0] = xm * ym * zm;  // (0, 0, 0)
                            f[1] = x * ym * zm;   // (1, 0, 0)
                            f[2] = x * y * zm;    // (1, 1, 0)
                            f[3] = xm * y * zm;   // (0, 1, 0)
                            f[4] = xm * ym * z;   // (0, 0, 1)
                            f[5] = x * ym * z;    // (1, 0, 1)
                            f[6] = x * y * z;     // (1, 1, 1)
                            f[7] = xm * y * z;    // (0, 1, 1)

                            const scalar_t val = e_from[lidx];
                            for (int i = 0; i < 8; i++) {
                                element_vector[i] += f[i] * val;
                            }
                        }
                    }
                }

                for (int i = 0; i < 8; i++) {
                    const int c = corners[i];
#pragma omp atomic update
                    to[ev[c]] += element_vector[i] / element_to_node_incidence_count[ev[c]];
                }
            }
        }

        free(e_from);
        free(ev);
    }

    return SFEM_SUCCESS;
}

int proteus_hex8_hierarchical_prolongation(int level,
                                           const ptrdiff_t nelements,
                                           idx_t **const SFEM_RESTRICT elements,
                                           const int vec_size,
                                           const real_t *const SFEM_RESTRICT from,
                                           real_t *const SFEM_RESTRICT to) {
#pragma omp parallel
    {
        const int nxe = proteus_hex8_nxe(level);
        scalar_t e_from[8];
        scalar_t *e_to = malloc(nxe * sizeof(scalar_t));
        idx_t *ev = malloc(nxe * sizeof(idx_t));

        const int corners[8] = {// Bottom
                                proteus_hex8_lidx(level, 0, 0, 0),
                                proteus_hex8_lidx(level, level, 0, 0),
                                proteus_hex8_lidx(level, level, level, 0),
                                proteus_hex8_lidx(level, 0, level, 0),
                                // Top
                                proteus_hex8_lidx(level, 0, 0, level),
                                proteus_hex8_lidx(level, level, 0, level),
                                proteus_hex8_lidx(level, level, level, level),
                                proteus_hex8_lidx(level, 0, level, level)};

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; e++) {
            {
                // Gather elemental data
                for (int d = 0; d < nxe; d++) {
                    ev[d] = elements[d][e];
                }

                for (int d = 0; d < nxe; d++) {
                    e_to[d] = 0;
                }

                for (int d = 0; d < 8; d++) {
                    e_from[d] = from[ev[corners[d]]];
                    assert(e_from[d] == e_from[d]);
                }

                const scalar_t h = 1. / level;
                // Iterate over structrued grid (nodes)
                for (int zi = 0; zi < level + 1; zi++) {
                    for (int yi = 0; yi < level + 1; yi++) {
                        for (int xi = 0; xi < level + 1; xi++) {
                            int lidx = proteus_hex8_lidx(level, xi, yi, zi);

                            const scalar_t x = xi * h;
                            const scalar_t y = yi * h;
                            const scalar_t z = zi * h;

                            // Evaluate Hex8 basis functions at x, y, z
                            const scalar_t xm = (1 - x);
                            const scalar_t ym = (1 - y);
                            const scalar_t zm = (1 - z);

                            scalar_t f[8];
                            f[0] = xm * ym * zm;  // (0, 0, 0)
                            f[1] = x * ym * zm;   // (1, 0, 0)
                            f[2] = x * y * zm;    // (1, 1, 0)
                            f[3] = xm * y * zm;   // (0, 1, 0)
                            f[4] = xm * ym * z;   // (0, 0, 1)
                            f[5] = x * ym * z;    // (1, 0, 1)
                            f[6] = x * y * z;     // (1, 1, 1)
                            f[7] = xm * y * z;    // (0, 1, 1)

                            for (int i = 0; i < 8; i++) {
                                e_to[lidx] += f[i] * e_from[i];
                            }
                        }
                    }
                }

                for (int i = 0; i < nxe; i++) {
                    to[ev[i]] = e_to[i];
                }
            }
        }

        free(e_to);
        free(ev);
    }

    return SFEM_SUCCESS;
}
