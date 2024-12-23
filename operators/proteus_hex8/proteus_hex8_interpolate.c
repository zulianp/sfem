#include "proteus_hex8_interpolate.h"

#include "proteus_hex8.h"

#include <stdio.h>

int         proteus_hex8_hierarchical_restriction(int                                 level,
                                                  const ptrdiff_t                     nelements,
                                                  idx_t **const SFEM_RESTRICT         elements,
                                                  const uint16_t *const SFEM_RESTRICT element_to_node_incidence_count,
                                                  const int                           vec_size,
                                                  const real_t *const SFEM_RESTRICT   from,
                                                  real_t *const SFEM_RESTRICT         to) {
#pragma omp parallel
    {
        const int  nxe    = proteus_hex8_nxe(level);
        scalar_t **e_from = malloc(vec_size * sizeof(scalar_t *));
        scalar_t **e_to   = malloc(vec_size * sizeof(scalar_t *));
        uint16_t  *weight = malloc(nxe * sizeof(uint16_t));

        for (int d = 0; d < vec_size; d++) {
            e_from[d] = malloc(nxe * sizeof(scalar_t));
            e_to[d]   = malloc(8 * sizeof(scalar_t));
        }

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

                for (int d = 0; d < vec_size; d++) {
                    for (int v = 0; v < nxe; v++) {
                        e_from[d][v] = from[ev[v] * vec_size + d];
                        assert(e_from[d][v] == e_from[d][v]);
                    }
                }

                for (int v = 0; v < nxe; v++) {
                    weight[v] = element_to_node_incidence_count[ev[v]];
                }

                for (int d = 0; d < vec_size; d++) {
                    for (int v = 0; v < 8; v++) {
                        e_to[d][v] = 0;
                    }
                }

                const scalar_t h = 1. / level;
                // Iterate over structrued grid (nodes)
                for (int zi = 0; zi < level + 1; zi++) {
                    for (int yi = 0; yi < level + 1; yi++) {
                        for (int xi = 0; xi < level + 1; xi++) {
                            const int lidx = proteus_hex8_lidx(level, xi, yi, zi);

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

                            for (int d = 0; d < vec_size; d++) {
                                const scalar_t val = e_from[d][lidx] / weight[lidx];
                                for (int i = 0; i < 8; i++) {
                                    e_to[d][i] += f[i] * val;
                                }
                            }
                        }
                    }
                }

                for (int d = 0; d < vec_size; d++) {
                    for (int i = 0; i < 8; i++) {
                        const int idx = ev[corners[i]] * vec_size + d;
#pragma omp atomic update
                        to[idx] += e_to[d][i];
                    }
                }
            }
        }

        for (int d = 0; d < vec_size; d++) {
            free(e_to[d]);
            free(e_from[d]);
        }

        free(e_from);
        free(e_to);
        free(ev);
        free(weight);
    }

    return SFEM_SUCCESS;
}

int         proteus_hex8_hierarchical_prolongation(int                               level,
                                                   const ptrdiff_t                   nelements,
                                                   idx_t **const SFEM_RESTRICT       elements,
                                                   const int                         vec_size,
                                                   const real_t *const SFEM_RESTRICT from,
                                                   real_t *const SFEM_RESTRICT       to) {
#pragma omp parallel
    {
        const int  nxe    = proteus_hex8_nxe(level);
        scalar_t **e_from = malloc(vec_size * sizeof(scalar_t *));
        scalar_t **e_to   = malloc(vec_size * sizeof(scalar_t *));

        for (int d = 0; d < vec_size; d++) {
            e_from[d] = malloc(8 * sizeof(scalar_t));
            e_to[d]   = malloc(nxe * sizeof(scalar_t));
        }

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

                for (int d = 0; d < vec_size; d++) {
                    for (int v = 0; v < nxe; v++) {
                        e_to[d][v] = 0;
                    }
                }

                for (int d = 0; d < vec_size; d++) {
                    for (int v = 0; v < 8; v++) {
                        e_from[d][v] = from[ev[corners[v]] * vec_size + d];
                        assert(e_from[d][v] == e_from[d][v]);
                    }
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

                            for (int d = 0; d < vec_size; d++) {
                                for (int v = 0; v < 8; v++) {
                                    e_to[d][lidx] += f[v] * e_from[d][v];
                                }
                            }
                        }
                    }
                }

                for (int i = 0; i < nxe; i++) {
                    for (int d = 0; d < vec_size; d++) {
                        to[ev[i] * vec_size + d] = e_to[d][i];
                    }
                }
            }
        }

        for (int d = 0; d < vec_size; d++) {
            free(e_to[d]);
            free(e_from[d]);
        }

        free(e_from);
        free(e_to);
        free(ev);
    }

    return SFEM_SUCCESS;
}

// OpenMP nested parallelization? https://docs.oracle.com/cd/E19205-01/819-5270/aewbc/index.html
// https://ppc.cs.aalto.fi/ch3/nested/

int sshex8_element_node_incidence_count(const int                     level,
                                        const int                     stride,
                                        const ptrdiff_t               nelements,
                                        idx_t **const SFEM_RESTRICT   elements,
                                        uint16_t *const SFEM_RESTRICT count) {
    for (int zi = 0; zi <= level; zi++) {
        for (int yi = 0; yi <= level; yi++) {
            for (int xi = 0; xi <= level; xi++) {
                const int v = proteus_hex8_lidx(level * stride, xi * stride, yi * stride, zi * stride);
#pragma omp parallel for
                for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma omp atomic update
                    count[elements[v][i]]++;
                }
            }
        }
    }
    return SFEM_SUCCESS;
}

int sshex8_restrict(const ptrdiff_t                     nelements,
                    const int                           from_level,
                    const int                           from_level_stride,
                    idx_t **const SFEM_RESTRICT         from_elements,
                    const uint16_t *const SFEM_RESTRICT from_element_to_node_incidence_count,
                    const int                           to_level,
                    const int                           to_level_stride,
                    idx_t **const SFEM_RESTRICT         to_elements,
                    const int                           vec_size,
                    const real_t *const SFEM_RESTRICT   from,
                    real_t *const SFEM_RESTRICT         to) {
    return SFEM_FAILURE;
}

int sshex8_prolongate(const ptrdiff_t                   nelements,
                      const int                         from_level,
                      const int                         from_level_stride,
                      idx_t **const SFEM_RESTRICT       from_elements,
                      const int                         to_level,
                      const int                         to_level_stride,
                      idx_t **const SFEM_RESTRICT       to_elements,
                      const int                         vec_size,
                      const real_t *const SFEM_RESTRICT from,
                      real_t *const SFEM_RESTRICT       to) {
    //     assert(to_level % from_level == 0);

    //     if (to_level % from_level != 0) {
    //         SFEM_ERROR("Nested meshes requirement: to_level must be divisible by from_level!");
    //         return SFEM_FAILURE;
    //     }

    // #pragma omp parallel
    //     {
    //         const int from_nxe     = proteus_hex8_nxe(from_level);
    //         const int to_nxe       = proteus_hex8_nxe(to_level);
    //         const int from_to_step = to_level / from_level;

    //         scalar_t **to_coeffs = malloc(vec_size * sizeof(scalar_t *));
    //         for (int d = 0; d < vec_size; d++) {
    //             to_coeffs[d] = malloc(to_nxe * sizeof(scalar_t));
    //         }

    // #pragma omp for
    //         for (ptrdiff_t e = 0; e < nelements; e++) {
    //             {
    // #ifndef NDEBUG
    //                 // Only for debugging
    //                 for (int d = 0; d < vec_size; d++) {
    //                     memset(to_coeffs[d], 0, to_nxe * sizeof(scalar_t));
    //                 }
    // #endif
    //                 // Gather elemental data
    //                 // Fill matching nodes with from data while gathering
    //                 for (int d = 0; d < vec_size; d++) {
    //                     for (int zi = 0; yi <= from_level; yi++) {
    //                         for (int yi = 0; yi <= from_level; yi++) {
    //                             for (int xi = 0; xi <= from_level; xi++) {
    //                                 // Use top level stride
    //                                 const int from_lidx = proteus_hex8_lidx(
    //                                         from_level * from_level_stride, xi * from_level_stride, yi *
    //                                         from_level_stride);

    //                                 // Use stride to convert from "from" to "to" local indexing
    //                                 const int to_lidx = proteus_hex8_lidx(to_level, xi * from_to_step, yi * from_to_step);

    //                                 const idx_t    idx = from_elements[from_lidx][e];
    //                                 const scalar_t val = from[idx * vec_size + d];

    //                                 to_coeffs[d][to_lidx] = val;
    //                                 assert(to_coeffs[d][to_lidx] == to_coeffs[d][to_lidx]);
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    // }

    return SFEM_FAILURE;
}