#include "ssquad4_interpolate.h"
#include "sfem_base.h"
#include "sfem_defs.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

static SFEM_INLINE int ssquad4_lidx(const int L, const int x, const int y) {
    int Lp1 = L + 1;
    int ret = y * Lp1 + x;

    assert(ret < Lp1 * Lp1);
    assert(ret >= 0);
    return ret;
}

static SFEM_INLINE int ssquad4_txe(int level) { return level * level; }

static SFEM_INLINE int ssquad4_nxe(int level) {
    const int corners    = 4;
    const int edge_nodes = 4 * (level - 1);
    const int area_nodes = (level - 1) * (level - 1);
    return corners + edge_nodes + area_nodes;
}

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
//     assert(from_level % to_level == 0);

//     if (from_level % to_level != 0) {
//         SFEM_ERROR("Nested meshes requirement: to_level must be divisible by from_level!");
//         return SFEM_FAILURE;
//     }

// #pragma omp parallel
//     {
//         const int from_nxe    = ssquad4_nxe(from_level);
//         const int to_nxe      = ssquad4_nxe(to_level);
//         const int step_factor = from_level / to_level;

//         scalar_t **from_coeffs = malloc(vec_size * sizeof(scalar_t *));
//         for (int d = 0; d < vec_size; d++) {
//             from_coeffs[d] = malloc(from_nxe * sizeof(scalar_t));
//         }

//         scalar_t **to_coeffs = malloc(vec_size * sizeof(scalar_t *));
//         for (int d = 0; d < vec_size; d++) {
//             to_coeffs[d] = malloc(to_nxe * sizeof(scalar_t));
//         }

// #pragma omp for
//         for (ptrdiff_t e = 0; e < nelements; e++) {
//             {  // Gather elemental data

//                 for (int yi = 0; yi <= from_level; yi++) {
//                     for (int xi = 0; xi <= from_level; xi++) {
//                         const int v = ssquad4_lidx(from_level, xi, yi);
//                         const int strided_v =
//                                 ssquad4_lidx(from_level * from_level_stride, xi * from_level_stride, yi * from_level_stride);

//                         const ptrdiff_t gid = from_elements[strided_v][e];
//                         for (int d = 0; d < vec_size; d++) {
//                             from_coeffs[d][v] = from[gid * vec_size + d] / from_element_to_node_incidence_count[gid];
//                             // printf("%d %d %d) %g\n", xi, yi, from_coeffs[d][v]);
//                         }
//                     }
//                 }

//                 for (int d = 0; d < vec_size; d++) {
//                     memset(to_coeffs[d], 0, to_nxe * sizeof(scalar_t));
//                 }
//             }

//             for (int d = 0; d < vec_size; d++) {
//                 scalar_t *in  = from_coeffs[d];
//                 scalar_t *out = to_coeffs[d];

//                 for (int yi = 0; yi <= from_level; yi++) {
//                     for (int xi = 0; xi <= from_level; xi++) {
//                         const int v = ssquad4_lidx(from_level, xi, yi);
//                         // Floor
//                         const int cxi = xi / step_factor;
//                         const int cyi = yi / step_factor;

//                         const int xinc = MIN(cxi + 1, to_level) - cxi;
//                         const int yinc = MIN(cyi + 1, to_level) - cyi;

//                         const scalar_t lx = (xi - cxi * step_factor) / ((scalar_t)step_factor);
//                         const scalar_t ly = (yi - cyi * step_factor) / ((scalar_t)step_factor);

//                         assert(lx <= 1 + 1e-8);
//                         assert(lx >= -1e-8);

//                         const scalar_t phi0x[2] = {1 - lx, lx};
//                         const scalar_t phi0y[2] = {1 - ly, ly};

//                         for (int jj = 0; jj <= yinc; jj++) {
//                             for (int ii = 0; ii <= xinc; ii++) {
//                                 const scalar_t val = phi0x[ii] * phi0y[jj] * in[v];
//                                 out[ssquad4_lidx(to_level, cxi + ii, cyi + jj)] += val;
//                                 // printf("(%d %d %d) -> %g (%d %d %d)\n", xi, yi, val, cxi + ii, cyi + jj, czi + kk);
//                             }
//                         }
//                     }
//                 }
//             }

//             for (int d = 0; d < vec_size; d++) {
//                 for (int yi = 0; yi <= to_level; yi++) {
//                     for (int xi = 0; xi <= to_level; xi++) {
//                         // Use top level stride
//                         const int to_lidx =  // ssquad4_lidx(to_level, xi, yi);
//                                 ssquad4_lidx(to_level * to_level_stride, xi * to_level_stride, yi * to_level_stride);

//                         // Use stride to convert from "from" to "to" local indexing

//                         const idx_t idx = to_elements[to_lidx][e];
// #pragma omp atomic update
//                         to[idx * vec_size + d] += to_coeffs[d][ssquad4_lidx(to_level, xi, yi)];
//                     }
//                 }
//             }
//         }

//         for (int d = 0; d < vec_size; d++) {
//             free(from_coeffs[d]);
//         }

//         free(from_coeffs);

//         for (int d = 0; d < vec_size; d++) {
//             free(to_coeffs[d]);
//         }

//         free(to_coeffs);
//     }

//     return SFEM_SUCCESS;
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
                                    ssquad4_lidx(from_level * from_level_stride, xi * from_level_stride, yi *
                                    from_level_stride);
                            const ptrdiff_t gid = from_elements[strided_v][e];

                            for (int d = 0; d < vec_size; d++) {
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
                                const int between_idx = ssquad4_lidx(from_level, xi * step_factor + between_xi, yi *
                                step_factor); c[ssquad4_lidx(from_level, xi * step_factor, yi * step_factor)] += fl *
                                c[between_idx]; c[ssquad4_lidx(from_level, (xi + 1) * step_factor, yi * step_factor)] += fr *
                                c[between_idx];
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
                                const int      to_lidx = ssquad4_lidx(from_level, xi * step_factor, yi * step_factor +
                                between_yi); c[ssquad4_lidx(from_level, xi * step_factor, yi * step_factor)] += fb *
                                c[to_lidx]; c[ssquad4_lidx(from_level, xi * step_factor, (yi + 1) * step_factor)] += ft *
                                c[to_lidx];
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
                            const int to_lidx = ssquad4_lidx(to_level * to_level_stride, xi * to_level_stride, yi *
                            to_level_stride);

                            // Use stride to convert from "from" to "to" local indexing
                            const int from_lidx = ssquad4_lidx(from_level, xi * step_factor, yi * step_factor);

                            const idx_t idx = to_elements[to_lidx][e];
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
