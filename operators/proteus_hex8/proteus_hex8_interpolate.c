#include "proteus_hex8_interpolate.h"

#include "proteus_hex8.h"

#include <stdio.h>
#include <string.h>

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
    assert(to_level % from_level == 0);

    if (to_level % from_level != 0) {
        SFEM_ERROR("Nested meshes requirement: to_level must be divisible by from_level!");
        return SFEM_FAILURE;
    }

#pragma omp parallel
    {
        const int from_nxe     = proteus_hex8_nxe(from_level);
        const int to_nxe       = proteus_hex8_nxe(to_level);
        const int from_to_step = to_level / from_level;

        scalar_t **to_coeffs = malloc(vec_size * sizeof(scalar_t *));
        for (int d = 0; d < vec_size; d++) {
            to_coeffs[d] = malloc(to_nxe * sizeof(scalar_t));
        }

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; e++) {
            {
#ifndef NDEBUG
                // Only for debugging
                for (int d = 0; d < vec_size; d++) {
                    memset(to_coeffs[d], 0, to_nxe * sizeof(scalar_t));
                }
#endif
                // Gather elemental data
                // Fill matching nodes with from data while gathering
                for (int d = 0; d < vec_size; d++) {
                    for (int zi = 0; zi <= from_level; zi++) {
                        for (int yi = 0; yi <= from_level; yi++) {
                            for (int xi = 0; xi <= from_level; xi++) {
                                // Use top level stride
                                const int from_lidx = proteus_hex8_lidx(from_level * from_level_stride,
                                                                        xi * from_level_stride,
                                                                        yi * from_level_stride,
                                                                        zi * from_level_stride);

                                // Use stride to convert from "from" to "to" local indexing
                                const int to_lidx =
                                        proteus_hex8_lidx(to_level, xi * from_to_step, yi * from_to_step, zi * from_to_step);

                                const idx_t    idx = from_elements[from_lidx][e];
                                const scalar_t val = from[idx * vec_size + d];

                                to_coeffs[d][to_lidx] = val;
                                assert(to_coeffs[d][to_lidx] == to_coeffs[d][to_lidx]);
                            }
                        }
                    }

#if 0
printf("nodal interpolation\n");
for (int zi = 0; zi <= to_level; zi++) {
for (int yi = 0; yi <= to_level; yi++) {
for (int xi = 0; xi <= to_level; xi++) {
printf("%g ", to_coeffs[d][proteus_hex8_lidx(to_level, xi, yi, zi)]);
}
printf("\n");
}
printf("\n");
}
#endif
                }
            }

            const scalar_t to_h = from_level * 1. / to_level;
            for (int d = 0; d < vec_size; d++) {
                scalar_t *c = to_coeffs[d];

                // Interpolate the coefficients along the x-axis (edges)
                for (int zi = 0; zi <= from_level; zi++) {
                    for (int yi = 0; yi <= from_level; yi++) {
                        for (int xi = 0; xi < from_level; xi++) {
                            const scalar_t c0 =
                                    c[proteus_hex8_lidx(to_level, xi * from_to_step, yi * from_to_step, zi * from_to_step)];
                            const scalar_t c1 =
                                    c[proteus_hex8_lidx(to_level, (xi + 1) * from_to_step, yi * from_to_step, zi * from_to_step)];

                            for (int between_xi = 1; between_xi < from_to_step; between_xi++) {
                                const scalar_t fl      = (1 - between_xi * to_h);
                                const scalar_t fr      = (between_xi * to_h);
                                const int      to_lidx = proteus_hex8_lidx(
                                        to_level, xi * from_to_step + between_xi, yi * from_to_step, zi * from_to_step);
                                c[to_lidx] = fl * c0 + fr * c1;
                            }
                        }
                    }
                }

#if 0
printf("x-axis interpolation (edge)\n");
for (int zi = 0; zi <= to_level; zi++) {
for (int yi = 0; yi <= to_level; yi++) {
for (int xi = 0; xi <= to_level; xi++) {
printf("%g ", c[proteus_hex8_lidx(to_level, xi, yi, zi)]);
}
printf("\n");
}
printf("\n");
}
#endif

                // Interpolate the coefficients along the y-axis (edges)
                for (int zi = 0; zi <= from_level; zi++) {
                    for (int yi = 0; yi < from_level; yi++) {
                        for (int xi = 0; xi <= from_level; xi++) {
                            const scalar_t c0 =
                                    c[proteus_hex8_lidx(to_level, xi * from_to_step, yi * from_to_step, zi * from_to_step)];
                            const scalar_t c1 =
                                    c[proteus_hex8_lidx(to_level, xi * from_to_step, (yi + 1) * from_to_step, zi * from_to_step)];

                            for (int between_yi = 1; between_yi < from_to_step; between_yi++) {
                                const scalar_t fb      = (1 - between_yi * to_h);
                                const scalar_t ft      = (between_yi * to_h);
                                const int      to_lidx = proteus_hex8_lidx(
                                        to_level, xi * from_to_step, yi * from_to_step + between_yi, zi * from_to_step);
                                c[to_lidx] = fb * c0 + ft * c1;
                            }
                        }
                    }
                }

#if 0
printf("y-axis interpolation (edge)\n");
for (int zi = 0; zi <= to_level; zi++) {
for (int yi = 0; yi <= to_level; yi++) {
for (int xi = 0; xi <= to_level; xi++) {
printf("%g ", c[proteus_hex8_lidx(to_level, xi, yi, zi)]);
}
printf("\n");
}
printf("\n");
}
#endif

                // Interpolate the coefficients along the y-axis (edges)
                for (int zi = 0; zi < from_level; zi++) {
                    for (int yi = 0; yi <= from_level; yi++) {
                        for (int xi = 0; xi <= from_level; xi++) {
                            const scalar_t c0 =
                                    c[proteus_hex8_lidx(to_level, xi * from_to_step, yi * from_to_step, zi * from_to_step)];
                            const scalar_t c1 =
                                    c[proteus_hex8_lidx(to_level, xi * from_to_step, yi * from_to_step, (zi + 1) * from_to_step)];

                            for (int between_zi = 1; between_zi < from_to_step; between_zi++) {
                                const scalar_t fb      = (1 - between_zi * to_h);
                                const scalar_t ft      = (between_zi * to_h);
                                const int      to_lidx = proteus_hex8_lidx(
                                        to_level, xi * from_to_step, yi * from_to_step, zi * from_to_step + between_zi);
                                c[to_lidx] = fb * c0 + ft * c1;
                            }
                        }
                    }
                }

#if 0
printf("z-axis interpolation (edge)\n");
for (int zi = 0; zi <= to_level; zi++) {
for (int yi = 0; yi <= to_level; yi++) {
for (int xi = 0; xi <= to_level; xi++) {
printf("%g ", c[proteus_hex8_lidx(to_level, xi, yi, zi)]);
}
printf("\n");
}
printf("\n");
}
#endif

                // Interpolate the coefficients along the x-axis (center) in the x-y-planes
                for (int zi = 0; zi <= from_level; zi++) {
                    for (int yi = 0; yi < from_level; yi++) {
                        for (int between_yi = 1; between_yi < from_to_step; between_yi++) {
                            for (int xi = 0; xi < from_level; xi++) {
                                const int      zz    = zi * from_to_step;
                                const int      yy    = yi * from_to_step + between_yi;
                                const int      left  = proteus_hex8_lidx(to_level, xi * from_to_step, yy, zz);
                                const int      right = proteus_hex8_lidx(to_level, (xi + 1) * from_to_step, yy, zz);
                                const scalar_t cl    = c[left];
                                const scalar_t cr    = c[right];

                                for (int between_xi = 1; between_xi < from_to_step; between_xi++) {
                                    const scalar_t fl = (1 - between_xi * to_h);
                                    const scalar_t fr = (between_xi * to_h);

                                    const int xx     = xi * from_to_step + between_xi;
                                    const int center = proteus_hex8_lidx(to_level, xx, yy, zz);
                                    c[center]        = fl * cl + fr * cr;
                                }
                            }
                        }
                    }
                }

#if 0
printf("xy-plane interpolation (face)\n");
for (int zi = 0; zi <= to_level; zi++) {
for (int yi = 0; yi <= to_level; yi++) {
for (int xi = 0; xi <= to_level; xi++) {
printf("%g ", c[proteus_hex8_lidx(to_level, xi, yi, zi)]);
}
printf("\n");
}
printf("\n");
}
#endif

                // Interpolate the coefficients along the x-axis (center) in the x-z-planes
                for (int zi = 0; zi < from_level; zi++) {
                    for (int between_zi = 1; between_zi < from_to_step; between_zi++) {
                        for (int yi = 0; yi <= from_level; yi++) {
                            for (int xi = 0; xi < from_level; xi++) {
                                const int      yy    = yi * from_to_step;
                                const int      zz    = zi * from_to_step + between_zi;
                                const int      left  = proteus_hex8_lidx(to_level, xi * from_to_step, yy, zz);
                                const int      right = proteus_hex8_lidx(to_level, (xi + 1) * from_to_step, yy, zz);
                                const scalar_t cl    = c[left];
                                const scalar_t cr    = c[right];

                                for (int between_xi = 1; between_xi < from_to_step; between_xi++) {
                                    const scalar_t fl = (1 - between_xi * to_h);
                                    const scalar_t fr = (between_xi * to_h);

                                    const int xx     = xi * from_to_step + between_xi;
                                    const int center = proteus_hex8_lidx(to_level, xx, yy, zz);
                                    c[center]        = fl * cl + fr * cr;
                                }
                            }
                        }
                    }
                }

#if 0
printf("xy-plane interpolation (face)\n");
for (int zi = 0; zi <= to_level; zi++) {
for (int yi = 0; yi <= to_level; yi++) {
for (int xi = 0; xi <= to_level; xi++) {
printf("%g ", c[proteus_hex8_lidx(to_level, xi, yi, zi)]);
}
printf("\n");
}
printf("\n");
}
#endif

                // Interpolate the coefficients along the y-axis (center) in the y-z-planes
                for (int zi = 0; zi < from_level; zi++) {
                    for (int between_zi = 1; between_zi < from_to_step; between_zi++) {
                        for (int yi = 0; yi < from_level; yi++) {
                            for (int xi = 0; xi <= from_level; xi++) {
                                const int      xx    = xi * from_to_step;
                                const int      zz    = zi * from_to_step + between_zi;
                                const int      left  = proteus_hex8_lidx(to_level, xx, yi * from_to_step, zz);
                                const int      right = proteus_hex8_lidx(to_level, xx, (yi + 1) * from_to_step, zz);
                                const scalar_t cl    = c[left];
                                const scalar_t cr    = c[right];

                                for (int between_yi = 1; between_yi < from_to_step; between_yi++) {
                                    const scalar_t fl = (1 - between_yi * to_h);
                                    const scalar_t fr = (between_yi * to_h);

                                    const int yy     = yi * from_to_step + between_yi;
                                    const int center = proteus_hex8_lidx(to_level, xx, yy, zz);
                                    c[center]        = fl * cl + fr * cr;
                                }
                            }
                        }
                    }
                }

#if 0
printf("yz-plane interpolation (face)\n");
for (int zi = 0; zi <= to_level; zi++) {
for (int yi = 0; yi <= to_level; yi++) {
for (int xi = 0; xi <= to_level; xi++) {
printf("%g ", c[proteus_hex8_lidx(to_level, xi, yi, zi)]);
}
printf("\n");
}
printf("\n");
}
#endif

                // Interpolate the coefficients along the x-axis (center) in the x-y-planes
                for (int zi = 0; zi < from_level; zi++) {
                    for (int between_zi = 1; between_zi < from_to_step; between_zi++) {
                        for (int yi = 0; yi < from_level; yi++) {
                            for (int between_yi = 1; between_yi < from_to_step; between_yi++) {
                                for (int xi = 0; xi < from_level; xi++) {
                                    const int      zz    = zi * from_to_step + between_zi;
                                    const int      yy    = yi * from_to_step + between_yi;
                                    const int      left  = proteus_hex8_lidx(to_level, xi * from_to_step, yy, zz);
                                    const int      right = proteus_hex8_lidx(to_level, (xi + 1) * from_to_step, yy, zz);
                                    const scalar_t cl    = c[left];
                                    const scalar_t cr    = c[right];

                                    for (int between_xi = 1; between_xi < from_to_step; between_xi++) {
                                        const scalar_t fl = (1 - between_xi * to_h);
                                        const scalar_t fr = (between_xi * to_h);

                                        const int xx     = xi * from_to_step + between_xi;
                                        const int center = proteus_hex8_lidx(to_level, xx, yy, zz);
                                        c[center]        = fl * cl + fr * cr;
                                    }
                                }
                            }
                        }
                    }
                }

#if 0
printf("interpolation (volume)\n");
for (int zi = 0; zi <= to_level; zi++) {
for (int yi = 0; yi <= to_level; yi++) {
for (int xi = 0; xi <= to_level; xi++) {
printf("%g ", c[proteus_hex8_lidx(to_level, xi, yi, zi)]);
}
printf("\n");
}
printf("\n");
}
#endif
            }

            // Scatter elemental data
            for (int zi = 0; zi <= to_level; zi++) {
                for (int yi = 0; yi <= to_level; yi++) {
                    for (int xi = 0; xi <= to_level; xi++) {
                        const int v         = proteus_hex8_lidx(to_level, xi, yi, zi);
                        const int strided_v = proteus_hex8_lidx(
                                to_level * to_level_stride, xi * to_level_stride, yi * to_level_stride, zi * to_level_stride);
                        const ptrdiff_t gid = to_elements[strided_v][e];

                        for (int d = 0; d < vec_size; d++) {
                            to[gid * vec_size + d] = to_coeffs[d][v];
                        }
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}
