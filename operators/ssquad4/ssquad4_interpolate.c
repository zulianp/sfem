#include "ssquad4_interpolate.h"
#include "sfem_base.h"
#include "sfem_defs.h"

#include "argsort.h"
#include "ssquad4.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

int ssquad4_element_node_incidence_count(const int                     level,
                                         const int                     stride,
                                         const ptrdiff_t               nelements,
                                         idx_t **const SFEM_RESTRICT   elements,
                                         uint16_t *const SFEM_RESTRICT count) {
    for (int yi = 0; yi <= level; yi++) {
        for (int xi = 0; xi <= level; xi++) {
            const int v = ssquad4_lidx(level * stride, xi * stride, yi * stride);
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma omp atomic update
                count[elements[v][i]]++;
            }
        }
    }

    return SFEM_SUCCESS;
}

int         ssquad4_hierarchical_restriction(int                                 level,
                                             const ptrdiff_t                     nelements,
                                             idx_t **const SFEM_RESTRICT         elements,
                                             const uint16_t *const SFEM_RESTRICT element_to_node_incidence_count,
                                             const int                           vec_size,
                                             const real_t *const SFEM_RESTRICT   from,
                                             real_t *const SFEM_RESTRICT         to) {
#pragma omp parallel
    {
        const int  nxe    = ssquad4_nxe(level);
        scalar_t **e_from = malloc(vec_size * sizeof(scalar_t *));
        scalar_t **e_to   = malloc(vec_size * sizeof(scalar_t *));
        uint16_t  *weight = malloc(nxe * sizeof(uint16_t));

        for (int d = 0; d < vec_size; d++) {
            e_from[d] = malloc(nxe * sizeof(scalar_t));
            e_to[d]   = malloc(4 * sizeof(scalar_t));
        }

        idx_t *ev = malloc(nxe * sizeof(idx_t));

        const int corners[4] = {ssquad4_lidx(level, 0, 0),
                                ssquad4_lidx(level, level, 0),
                                ssquad4_lidx(level, level, level),
                                ssquad4_lidx(level, 0, level)};

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
                    for (int v = 0; v < 4; v++) {
                        e_to[d][v] = 0;
                    }
                }

                const scalar_t h = 1. / level;
                // Iterate over structrued grid (nodes)

                for (int yi = 0; yi < level + 1; yi++) {
                    for (int xi = 0; xi < level + 1; xi++) {
                        const int lidx = ssquad4_lidx(level, xi, yi);

                        const scalar_t x = xi * h;
                        const scalar_t y = yi * h;

                        // Evaluate Quad4 basis functions at x, y
                        const scalar_t xm = (1 - x);
                        const scalar_t ym = (1 - y);

                        scalar_t f[4];
                        f[0] = xm * ym;  // (0, 0, 0)
                        f[1] = x * ym;   // (1, 0, 0)
                        f[2] = x * y;    // (1, 1, 0)
                        f[3] = xm * y;   // (0, 1, 0)

                        for (int d = 0; d < vec_size; d++) {
                            const scalar_t val = e_from[d][lidx] / weight[lidx];
                            for (int i = 0; i < 4; i++) {
                                e_to[d][i] += f[i] * val;
                            }
                        }
                    }
                }

                for (int d = 0; d < vec_size; d++) {
                    for (int i = 0; i < 4; i++) {
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

int         ssquad4_hierarchical_prolongation(int                               level,
                                              const ptrdiff_t                   nelements,
                                              idx_t **const SFEM_RESTRICT       elements,
                                              const int                         vec_size,
                                              const real_t *const SFEM_RESTRICT from,
                                              real_t *const SFEM_RESTRICT       to) {
#pragma omp parallel
    {
        const int  nxe    = ssquad4_nxe(level);
        scalar_t **e_from = malloc(vec_size * sizeof(scalar_t *));
        scalar_t **e_to   = malloc(vec_size * sizeof(scalar_t *));

        for (int d = 0; d < vec_size; d++) {
            e_from[d] = malloc(4 * sizeof(scalar_t));
            e_to[d]   = malloc(nxe * sizeof(scalar_t));
        }

        idx_t *ev = malloc(nxe * sizeof(idx_t));

        const int corners[4] = {ssquad4_lidx(level, 0, 0),
                                ssquad4_lidx(level, level, 0),
                                ssquad4_lidx(level, level, level),
                                ssquad4_lidx(level, 0, level)};

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
                    for (int v = 0; v < 4; v++) {
                        e_from[d][v] = from[ev[corners[v]] * vec_size + d];
                        assert(e_from[d][v] == e_from[d][v]);
                    }
                }

                const scalar_t h = 1. / level;
                // Iterate over structrued grid (nodes)

                for (int yi = 0; yi < level + 1; yi++) {
                    for (int xi = 0; xi < level + 1; xi++) {
                        int lidx = ssquad4_lidx(level, xi, yi);

                        const scalar_t x = xi * h;
                        const scalar_t y = yi * h;

                        // Evaluate Quad4 basis functions at x, y
                        const scalar_t xm = (1 - x);
                        const scalar_t ym = (1 - y);

                        scalar_t f[4];
                        f[0] = xm * ym;  // (0, 0, 0)
                        f[1] = x * ym;   // (1, 0, 0)
                        f[2] = x * y;    // (1, 1, 0)
                        f[3] = xm * y;   // (0, 1, 0)

                        for (int d = 0; d < vec_size; d++) {
                            for (int v = 0; v < 4; v++) {
                                e_to[d][lidx] += f[v] * e_from[d][v];
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

int ssquad4_restrict(const ptrdiff_t                     nelements,
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
    // FIXME this should be handled outside!!!
    // if (to_level == 1) {
    //     return ssquad4_hierarchical_restriction(
    //             from_level, nelements, from_elements, from_element_to_node_incidence_count, vec_size, from, to);
    // }

    assert(from_level % to_level == 0);

    if (from_level % to_level != 0) {
        SFEM_ERROR("Nested meshes requirement: from_level must be divisible by to_level!");
        return SFEM_FAILURE;
    }

#pragma omp parallel
    {
        const int from_nxe    = ssquad4_nxe(from_level);
        const int to_nxe      = ssquad4_nxe(to_level);
        const int step_factor = from_level / to_level;

        scalar_t **from_coeffs = malloc(vec_size * sizeof(scalar_t *));
        for (int d = 0; d < vec_size; d++) {
            from_coeffs[d] = malloc(from_nxe * sizeof(scalar_t));
        }

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; e++) {
            {  // Gather elemental coefficients
                for (int yi = 0; yi <= from_level; yi++) {
                    for (int xi = 0; xi <= from_level; xi++) {
                        const int v = ssquad4_lidx(from_level, xi, yi);
                        const int strided_v =
                                ssquad4_lidx(from_level * from_level_stride, xi * from_level_stride, yi * from_level_stride);
                        const ptrdiff_t gid = from_elements[strided_v][e];

                        for (int d = 0; d < vec_size; d++) {
                            assert(from_element_to_node_incidence_count[gid] != 0);
                            from_coeffs[d][v] = from[gid * vec_size + d] / from_element_to_node_incidence_count[gid];
                        }
                    }
                }
            }

            const scalar_t h = to_level * 1. / from_level;
            for (int d = 0; d < vec_size; d++) {
                scalar_t *c = from_coeffs[d];

                // Restict the coefficients along the x-axis (edges)
                for (int yi = 0; yi <= to_level; yi++) {
                    for (int xi = 0; xi < to_level; xi++) {
                        for (int between_xi = 1; between_xi < step_factor; between_xi++) {
                            const scalar_t fl     = (1 - between_xi * h);
                            const scalar_t fr     = (between_xi * h);
                            const int between_idx = ssquad4_lidx(from_level, xi * step_factor + between_xi, yi * step_factor);
                            c[ssquad4_lidx(from_level, xi * step_factor, yi * step_factor)] += fl * c[between_idx];
                            c[ssquad4_lidx(from_level, (xi + 1) * step_factor, yi * step_factor)] += fr * c[between_idx];
                        }
                    }
                }

                // Restrict the coefficients along the x-axis (center)
                for (int yi = 0; yi < to_level; yi++) {
                    for (int xi = 0; xi < to_level; xi++) {
                        for (int between_yi = 1; between_yi < step_factor; between_yi++) {
                            const int yy    = yi * step_factor + between_yi;
                            const int left  = ssquad4_lidx(from_level, xi * step_factor, yy);
                            const int right = ssquad4_lidx(from_level, (xi + 1) * step_factor, yy);

                            for (int between_xi = 1; between_xi < step_factor; between_xi++) {
                                const scalar_t fl = (1 - between_xi * h);
                                const scalar_t fr = (between_xi * h);

                                const int xx     = xi * step_factor + between_xi;
                                const int center = ssquad4_lidx(from_level, xx, yy);

                                c[left] += fl * c[center];
                                c[right] += fr * c[center];
                            }
                        }
                    }
                }

                // Restrict the coefficients along the y-axis (edges)
                for (int yi = 0; yi < to_level; yi++) {
                    for (int xi = 0; xi <= to_level; xi++) {
                        for (int between_yi = 1; between_yi < step_factor; between_yi++) {
                            const scalar_t fb      = (1 - between_yi * h);
                            const scalar_t ft      = (between_yi * h);
                            const int      to_lidx = ssquad4_lidx(from_level, xi * step_factor, yi * step_factor + between_yi);
                            c[ssquad4_lidx(from_level, xi * step_factor, yi * step_factor)] += fb * c[to_lidx];
                            c[ssquad4_lidx(from_level, xi * step_factor, (yi + 1) * step_factor)] += ft * c[to_lidx];
                        }
                    }
                }
            }

            // Scatter elemental data
            // Extract coarse coeffs and discard rest
            for (int d = 0; d < vec_size; d++) {
                for (int yi = 0; yi <= to_level; yi++) {
                    for (int xi = 0; xi <= to_level; xi++) {
                        // Use top level stride
                        const int to_lidx = ssquad4_lidx(to_level * to_level_stride, xi * to_level_stride, yi * to_level_stride);

                        // Use stride to convert from "from" to "to" local indexing
                        const int from_lidx = ssquad4_lidx(from_level, xi * step_factor, yi * step_factor);

                        const idx_t idx = to_elements[to_lidx][e];
                        assert(from_coeffs[d][from_lidx] == from_coeffs[d][from_lidx]);
#pragma omp atomic update
                        to[idx * vec_size + d] += from_coeffs[d][from_lidx];
                    }
                }
            }
        }

        for (int d = 0; d < vec_size; d++) {
            free(from_coeffs[d]);
        }

        free(from_coeffs);
    }

    return SFEM_SUCCESS;
}

int ssquad4_prolongate(const ptrdiff_t                   nelements,
                       const int                         from_level,
                       const int                         from_level_stride,
                       idx_t **const SFEM_RESTRICT       from_elements,
                       const int                         to_level,
                       const int                         to_level_stride,
                       idx_t **const SFEM_RESTRICT       to_elements,
                       const int                         vec_size,
                       const real_t *const SFEM_RESTRICT from,
                       real_t *const SFEM_RESTRICT       to) {
    // FIXME handle otuside!!!
    // if (from_level == 1) {
    //     return ssquad4_hierarchical_prolongation(to_level, nelements, to_elements, vec_size, from, to);
    // }

    assert(to_level % from_level == 0);

    if (to_level % from_level != 0) {
        SFEM_ERROR("Nested meshes requirement: to_level must be divisible by from_level!");
        return SFEM_FAILURE;
    }

#pragma omp parallel
    {
        const int from_nxe     = ssquad4_nxe(from_level);
        const int to_nxe       = ssquad4_nxe(to_level);
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
                    for (int yi = 0; yi <= from_level; yi++) {
                        for (int xi = 0; xi <= from_level; xi++) {
                            // Use top level stride
                            const int from_lidx =
                                    ssquad4_lidx(from_level * from_level_stride, xi * from_level_stride, yi * from_level_stride);

                            // Use stride to convert from "from" to "to" local indexing
                            const int to_lidx = ssquad4_lidx(to_level, xi * from_to_step, yi * from_to_step);

                            const idx_t    idx = from_elements[from_lidx][e];
                            const scalar_t val = from[idx * vec_size + d];

                            to_coeffs[d][to_lidx] = val;
                            assert(to_coeffs[d][to_lidx] == to_coeffs[d][to_lidx]);
                        }
                    }
                }
            }

            const scalar_t to_h = from_level * 1. / to_level;
            for (int d = 0; d < vec_size; d++) {
                scalar_t *c = to_coeffs[d];

                // Interpolate the coefficients along the x-axis (edges)
                for (int yi = 0; yi <= from_level; yi++) {
                    for (int xi = 0; xi < from_level; xi++) {
                        const scalar_t c0 = c[ssquad4_lidx(to_level, xi * from_to_step, yi * from_to_step)];
                        const scalar_t c1 = c[ssquad4_lidx(to_level, (xi + 1) * from_to_step, yi * from_to_step)];

                        for (int between_xi = 1; between_xi < from_to_step; between_xi++) {
                            const scalar_t fl      = (1 - between_xi * to_h);
                            const scalar_t fr      = (between_xi * to_h);
                            const int      to_lidx = ssquad4_lidx(to_level, xi * from_to_step + between_xi, yi * from_to_step);
                            c[to_lidx]             = fl * c0 + fr * c1;
                        }
                    }
                }

#if 0
                printf("x-axis interpolation\n");
                for (int yi = 0; yi <= to_level; yi++) {
                    for (int xi = 0; xi <= to_level; xi++) {
                        printf("%g ", c[ssquad4_lidx(to_level, xi, yi)]);
                    }
                    printf("\n");
                }
#endif

                // Interpolate the coefficients along the y-axis (edges)
                for (int yi = 0; yi < from_level; yi++) {
                    for (int xi = 0; xi <= from_level; xi++) {
                        const scalar_t c0 = c[ssquad4_lidx(to_level, xi * from_to_step, yi * from_to_step)];
                        const scalar_t c1 = c[ssquad4_lidx(to_level, xi * from_to_step, (yi + 1) * from_to_step)];

                        for (int between_yi = 1; between_yi < from_to_step; between_yi++) {
                            const scalar_t fb      = (1 - between_yi * to_h);
                            const scalar_t ft      = (between_yi * to_h);
                            const int      to_lidx = ssquad4_lidx(to_level, xi * from_to_step, yi * from_to_step + between_yi);
                            c[to_lidx]             = fb * c0 + ft * c1;
                        }
                    }
                }
#if 0
                printf("y-axis interpolation\n");
                for (int yi = 0; yi <= to_level; yi++) {
                    for (int xi = 0; xi <= to_level; xi++) {
                        printf("%g ", c[ssquad4_lidx(to_level, xi, yi)]);
                    }
                    printf("\n");
                }
#endif

                // Interpolate the coefficients along the x-axis (center)
                for (int yi = 0; yi < from_level; yi++) {
                    for (int between_yi = 1; between_yi < from_to_step; between_yi++) {
                        for (int xi = 0; xi < from_level; xi++) {
                            const int      yy    = yi * from_to_step + between_yi;
                            const int      left  = ssquad4_lidx(to_level, xi * from_to_step, yy);
                            const int      right = ssquad4_lidx(to_level, (xi + 1) * from_to_step, yy);
                            const scalar_t cl    = c[left];
                            const scalar_t cr    = c[right];

                            for (int between_xi = 1; between_xi < from_to_step; between_xi++) {
                                const scalar_t fl = (1 - between_xi * to_h);
                                const scalar_t fr = (between_xi * to_h);

                                const int xx     = xi * from_to_step + between_xi;
                                const int center = ssquad4_lidx(to_level, xx, yy);
                                c[center]        = fl * cl + fr * cr;
                            }
                        }
                    }
                }
#if 0
                printf("Interior interpolation\n");
                for (int yi = 0; yi <= to_level; yi++) {
                    for (int xi = 0; xi <= to_level; xi++) {
                        printf("%g ", c[ssquad4_lidx(to_level, xi, yi)]);
                    }
                    printf("\n");
                }
#endif
            }

            // Scatter elemental data
            for (int yi = 0; yi <= to_level; yi++) {
                for (int xi = 0; xi <= to_level; xi++) {
                    const int v         = ssquad4_lidx(to_level, xi, yi);
                    const int strided_v = ssquad4_lidx(to_level * to_level_stride, xi * to_level_stride, yi * to_level_stride);
                    const ptrdiff_t gid = to_elements[strided_v][e];

                    for (int d = 0; d < vec_size; d++) {
                        to[gid * vec_size + d] = to_coeffs[d][v];
                    }
                }
            }
        }

        for (int d = 0; d < vec_size; d++) {
            free(to_coeffs[d]);
        }

        free(to_coeffs);
    }

    return SFEM_SUCCESS;
}

int ssquad4_prolongation_crs_nnz(const int                    level,
                                 const ptrdiff_t              nelements,
                                 idx_t **const SFEM_RESTRICT  elements,
                                 const ptrdiff_t              to_nnodes,
                                 count_t *const SFEM_RESTRICT rowptr) {
    const int corners[4] = {ssquad4_lidx(level, 0, 0),
                            ssquad4_lidx(level, level, 0),
                            ssquad4_lidx(level, level, level),
                            ssquad4_lidx(level, 0, level)};

    rowptr[0] = 0;

    // Corners
#pragma omp parallel for
    for (int d = 0; d < 4; d++) {
        const int c = corners[d];
        for (ptrdiff_t i = 0; i < nelements; i++) {
            const idx_t node = elements[c][i];
            rowptr[node + 1] = 1;
        }
    }

    // Edges
#pragma omp parallel for
    for (int vi = 1; vi < level; vi++) {
        const int edges[4] = {
                ssquad4_lidx(level, 0, vi),      // Left
                ssquad4_lidx(level, level, vi),  // Right
                ssquad4_lidx(level, vi, 0),      // Bottom
                ssquad4_lidx(level, vi, level)   // Top
        };

        for (int d = 0; d < 4; d++) {
            int e = edges[d];
            for (ptrdiff_t i = 0; i < nelements; i++) {
                const idx_t node = elements[e][i];
                rowptr[node + 1] = 2;
            }
        }
    }

    // Interior
#pragma omp parallel for collapse(2)
    for (int yi = 1; yi < level; yi++) {
        for (int xi = 1; xi < level; xi++) {
            const int ii = ssquad4_lidx(level, xi, yi);
            for (ptrdiff_t i = 0; i < nelements; i++) {
                const idx_t node = elements[ii][i];
                rowptr[node + 1] = 4;
            }
        }
    }

    for (ptrdiff_t i = 0; i < to_nnodes; i++) {
        assert(rowptr[i + 1] != 0);
        rowptr[i + 1] += rowptr[i];
    }

    return SFEM_SUCCESS;
}

int ssquad4_prolongation_crs_fill(const int                    level,
                                  const ptrdiff_t              nelements,
                                  idx_t **const SFEM_RESTRICT  elements,
                                  const ptrdiff_t              to_nnodes,
                                  count_t *const SFEM_RESTRICT rowptr,
                                  idx_t *const SFEM_RESTRICT   colidx,
                                  real_t *const SFEM_RESTRICT  values) {
    const int corners[4] = {ssquad4_lidx(level, 0, 0),
                            ssquad4_lidx(level, level, 0),
                            ssquad4_lidx(level, level, level),
                            ssquad4_lidx(level, 0, level)};

    // Corners
#pragma omp parallel for
    for (int d = 0; d < 4; d++) {
        const int c = corners[d];
        for (ptrdiff_t i = 0; i < nelements; i++) {
            const idx_t node = elements[c][i];
            // Identity
            colidx[rowptr[node]] = node;
            values[rowptr[node]] = 1;
        }
    }

    const real_t h = 1. / level;

    const int ev0[4] = {corners[0], corners[1], corners[0], corners[3]};
    const int ev1[4] = {corners[3], corners[2], corners[1], corners[2]};

    // Edges
#pragma omp parallel for
    for (int vi = 1; vi < level; vi++) {
        const int edges[4] = {
                ssquad4_lidx(level, 0, vi),      // Left
                ssquad4_lidx(level, level, vi),  // Right
                ssquad4_lidx(level, vi, 0),      // Bottom
                ssquad4_lidx(level, vi, level)   // Top
        };

        for (int d = 0; d < 4; d++) {
            const int    e  = edges[d];
            const real_t w0 = vi * h;
            const real_t w1 = vi * (1 - h);
            const int    v0 = ev0[d];
            const int    v1 = ev1[d];

            for (ptrdiff_t i = 0; i < nelements; i++) {
                const idx_t node  = elements[e][i];
                idx_t       node0 = elements[v0][i];
                idx_t       node1 = elements[v1][i];
                real_t      w0i   = w0;
                real_t      w1i   = w1;

                // Ensure sorted ordering
                if (node1 < node0) {
                    const idx_t temp = node0;
                    node0            = node1;
                    node1            = temp;

                    w0i = w1;
                    w1i = w0;
                }

                colidx[rowptr[node]] = node0;
                values[rowptr[node]] = w0i;

                colidx[rowptr[node] + 1] = node1;
                values[rowptr[node] + 1] = w1i;
            }
        }
    }

    // Interior
#pragma omp parallel for collapse(2)
    for (int yi = 1; yi < level; yi++) {
        for (int xi = 1; xi < level; xi++) {
            const int    ii   = ssquad4_lidx(level, xi, yi);
            const real_t w[4] = {
                    (1 - xi * h) * (1 - yi * h),  // v0
                    (xi * h) * (1 - yi * h),      // v1
                    (xi * h) * (yi * h),          // v2
                    (1 - xi * h) * (yi * h)       // v3
            };

            for (ptrdiff_t i = 0; i < nelements; i++) {
                const idx_t node = elements[ii][i];

                idx_t c[4] = {
                        elements[corners[0]][i],
                        elements[corners[1]][i],
                        elements[corners[2]][i],
                        elements[corners[3]][i],
                };

                idx_t order[4] = {0, 1, 2, 3};
                argsort_i(4, c, order);

                assert(c[order[0]] < c[order[1]]);
                assert(c[order[1]] < c[order[2]]);
                assert(c[order[2]] < c[order[3]]);

                for (int d = 0; d < 4; d++) {
                    colidx[rowptr[node] + d] = c[order[d]];
                    values[rowptr[node] + d] = w[order[d]];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}
