#include "sshex8_laplacian.h"

#include "hex8_inline_cpu.h"
#include "hex8_laplacian_inline_cpu.h"

#include "hex8_quadrature.h"
#include "sfem_mask.h"
#include "sshex8_skeleton_stencil.h"
#include "stencil3.h"
#include "stencil_cg.h"
#include "tet4_inline_cpu.h"

#include <math.h>
#include <stdio.h>

#include "sshex8.h"

static SFEM_INLINE void hex8_sub_fff_0(const scalar_t *const SFEM_RESTRICT fff,
                                       const scalar_t                      h,
                                       scalar_t *const SFEM_RESTRICT       sub_fff) {
    sub_fff[0] = fff[0] * h;
    sub_fff[1] = fff[1] * h;
    sub_fff[2] = fff[2] * h;
    sub_fff[3] = fff[3] * h;
    sub_fff[4] = fff[4] * h;
    sub_fff[5] = fff[5] * h;
}

void print_matrix(int r, int c, const accumulator_t *const m) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%g\t", m[i * c + j]);
        }
        printf("\n");
    }
}

int sshex8_laplacian_apply(const int                         level,
                           const ptrdiff_t                   nelements,
                           ptrdiff_t                         interior_start,
                           idx_t **const SFEM_RESTRICT       elements,
                           geom_t **const SFEM_RESTRICT      points,
                           const real_t *const SFEM_RESTRICT u,
                           real_t *const SFEM_RESTRICT       values) {
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

    const int       n_qp = q6_n;
    const scalar_t *qx   = q6_x;
    const scalar_t *qy   = q6_y;
    const scalar_t *qz   = q6_z;
    const scalar_t *qw   = q6_w;

    int Lm1  = level - 1;
    int Lm13 = Lm1 * Lm1 * Lm1;

#pragma omp parallel
    {
        // Allocation per thread
        scalar_t      *eu = malloc(nxe * sizeof(scalar_t));
        idx_t         *ev = malloc(nxe * sizeof(idx_t));
        accumulator_t *v  = malloc(nxe * sizeof(accumulator_t));

        scalar_t x[8];
        scalar_t y[8];
        scalar_t z[8];

        scalar_t      element_u[8];
        accumulator_t element_vector[8];
        scalar_t      m_fff[6], fff[6];

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            {
                // Gather elemental data
                for (int d = 0; d < nxe; d++) {
                    ev[d] = elements[d][e];
                }

                for (int d = 0; d < 8; d++) {
                    x[d] = points[0][ev[proteus_to_std_hex8_corners[d]]];
                    y[d] = points[1][ev[proteus_to_std_hex8_corners[d]]];
                    z[d] = points[2][ev[proteus_to_std_hex8_corners[d]]];
                }

                for (int d = 0; d < nxe; d++) {
                    eu[d] = u[ev[d]];

                    assert(eu[d] == eu[d]);
                }

                memset(v, 0, nxe * sizeof(accumulator_t));
            }

            const scalar_t h = 1. / level;

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

                        for (int d = 0; d < 8; d++) {
                            element_u[d] = eu[lev[d]];

                            assert(element_u[d] == element_u[d]);
                        }

                        for (int d = 0; d < 8; d++) {
                            element_vector[d] = 0;
                        }

                        // Translation
                        const scalar_t tx = (scalar_t)xi / level;
                        const scalar_t ty = (scalar_t)yi / level;
                        const scalar_t tz = (scalar_t)zi / level;

                        // Quadrature
                        for (int k = 0; k < n_qp; k++) {
                            // 1) Compute qp in macro-element coordinates
                            const scalar_t m_qx = h * qx[k] + tx;
                            const scalar_t m_qy = h * qy[k] + ty;
                            const scalar_t m_qz = h * qz[k] + tz;

                            // 2) Evaluate FFF
                            hex8_fff(x, y, z, m_qx, m_qy, m_qz, m_fff);

                            // 3) Transform to sub-FFF
                            hex8_sub_fff_0(m_fff, h, fff);

                            // Evaluate y = op * x
                            hex8_laplacian_apply_fff(fff, qx[k], qy[k], qz[k], qw[k], element_u, element_vector);
                        }

                        // Accumulate to macro-element buffer
                        for (int d = 0; d < 8; d++) {
                            assert(element_vector[d] == element_vector[d]);
                            v[lev[d]] += element_vector[d];
                        }
                    }
                }
            }

            {
                // Scatter elemental data
                for (int d = 0; d < nxe; d++) {
#pragma omp atomic update
                    values[ev[d]] += v[d];
                }
            }
        }

        // Clean-up
        free(ev);
        free(eu);
        free(v);
    }

    return SFEM_SUCCESS;
}

int affine_sshex8_laplacian_apply(const int                         level,
                                  const ptrdiff_t                   nelements,
                                  ptrdiff_t                         interior_start,
                                  idx_t **const SFEM_RESTRICT       elements,
                                  geom_t **const SFEM_RESTRICT      std_hex8_points,
                                  const real_t *const SFEM_RESTRICT u,
                                  real_t *const SFEM_RESTRICT       values) {
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

    const int Lm1  = level - 1;
    const int Lm13 = Lm1 * Lm1 * Lm1;

#pragma omp parallel
    {
        // Allocation per thread
        scalar_t      *eu = malloc(nxe * sizeof(scalar_t));
        idx_t         *ev = malloc(nxe * sizeof(idx_t));
        accumulator_t *v  = malloc(nxe * sizeof(accumulator_t));

        scalar_t x[8];
        scalar_t y[8];
        scalar_t z[8];

        scalar_t      element_u[8];
        accumulator_t element_vector[8];
        scalar_t      m_fff[6], fff[6];

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
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

                for (int d = 0; d < nxe; d++) {
                    eu[d] = u[ev[d]];
                    assert(eu[d] == eu[d]);
                }

                memset(v, 0, nxe * sizeof(accumulator_t));
            }

            const scalar_t h = 1. / level;

            // We evaluate the jacobian at the center of the element
            // in case that the mapping is not linear
            hex8_fff(x, y, z, 0.5, 0.5, 0.5, m_fff);

            // Assume affine here!
            hex8_sub_fff_0(m_fff, h, fff);

#define SSHEX8_USE_MV  // assemblying the elemental matrix is much faster
#ifdef SSHEX8_USE_MV
            accumulator_t laplacian_matrix[8 * 8];
            hex8_laplacian_matrix_fff_integral(fff, laplacian_matrix);
            // hex8_laplacian_matrix_fff_taylor(fff, laplacian_matrix);

#endif

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

                        for (int d = 0; d < 8; d++) {
                            element_u[d] = eu[lev[d]];
                        }

#ifdef SSHEX8_USE_MV

                        for (int i = 0; i < 8; i++) {
                            element_vector[i] = 0;
                        }

                        for (int i = 0; i < 8; i++) {
                            const scalar_t *const row = &laplacian_matrix[i * 8];
                            const scalar_t        ui  = element_u[i];
                            assert(ui == ui);
                            for (int j = 0; j < 8; j++) {
                                assert(row[j] == row[j]);
                                element_vector[j] += ui * row[j];
                            }
                        }
#else
                        hex8_laplacian_apply_fff_integral(
                                fff,
                                element_u,
                                element_vector);  // 2x faster than taylor version below
                                                  // hex8_laplacian_apply_fff_taylor(fff, element_u, element_vector);
#endif

                        // Accumulate to macro-element buffer
                        for (int d = 0; d < 8; d++) {
                            assert(element_vector[d] == element_vector[d]);
                            v[lev[d]] += element_vector[d];
                        }
                    }
                }
            }

            {
                // Scatter elemental data
                for (int d = 0; d < nxe; d++) {
                    assert(v[d] == v[d]);
#pragma omp atomic update
                    values[ev[d]] += v[d];
                }
            }
        }

        // Clean-up
        free(ev);
        free(eu);
        free(v);
    }

    return SFEM_SUCCESS;
}

int affine_sshex8_laplacian_diag(const int                    level,
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

    const int Lm1  = level - 1;
    const int Lm13 = Lm1 * Lm1 * Lm1;

#pragma omp parallel
    {
        // Allocation per thread
        idx_t         *ev = malloc(nxe * sizeof(idx_t));
        accumulator_t *v  = malloc(nxe * sizeof(accumulator_t));

        scalar_t x[8];
        scalar_t y[8];
        scalar_t z[8];

        scalar_t m_fff[6], fff[6];

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
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

            // We evaluate the jacobian at the center of the element
            // in case that the mapping is affine
            hex8_fff(x, y, z, 0.5, 0.5, 0.5, m_fff);
            hex8_sub_fff_0(m_fff, h, fff);

            accumulator_t laplacian_diag[8];
            hex8_laplacian_diag_fff_integral(fff, laplacian_diag);

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
                            v[lev[d]] += laplacian_diag[d];
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

int affine_sshex8_laplacian_stencil_apply(const int                         level,
                                          const ptrdiff_t                   nelements,
                                          ptrdiff_t                         interior_start,
                                          idx_t **const SFEM_RESTRICT       elements,
                                          geom_t **const SFEM_RESTRICT      std_hex8_points,
                                          const real_t *const SFEM_RESTRICT u,
                                          real_t *const SFEM_RESTRICT       values) {
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

    const int Lm1  = level - 1;
    const int Lm13 = Lm1 * Lm1 * Lm1;

#pragma omp parallel
    {
        // Allocation per thread
        scalar_t      *eu = malloc(nxe * sizeof(scalar_t));
        idx_t         *ev = malloc(nxe * sizeof(idx_t));
        accumulator_t *v  = malloc(nxe * sizeof(accumulator_t));

        scalar_t x[8];
        scalar_t y[8];
        scalar_t z[8];

        scalar_t      element_u[8];
        accumulator_t element_vector[8];
        scalar_t      m_fff[6], fff[6];

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
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

                for (int d = 0; d < nxe; d++) {
                    eu[d] = u[ev[d]];
                    assert(eu[d] == eu[d]);
                }

                memset(v, 0, nxe * sizeof(accumulator_t));
            }

            const scalar_t h = 1. / level;

            // We evaluate the jacobian at the center of the element
            // in case that the mapping is not linear
            hex8_fff(x, y, z, 0.5, 0.5, 0.5, m_fff);

            // Assume affine here!
            hex8_sub_fff_0(m_fff, h, fff);

            accumulator_t laplacian_matrix[8 * 8];
            hex8_laplacian_matrix_fff_integral(fff, laplacian_matrix);
            scalar_t laplacian_stencil[3 * 3 * 3];
            hex8_matrix_to_stencil(laplacian_matrix, laplacian_stencil);
            sshex8_stencil(level + 1, level + 1, level + 1, laplacian_stencil, eu, v);
            sshex8_surface_stencil(
                    level + 1, level + 1, level + 1, 1, level + 1, (level + 1) * (level + 1), laplacian_matrix, eu, v);

#ifndef NDEBUG
            {
                accumulator_t *test = calloc(nxe, sizeof(accumulator_t));
                sshex8_apply_element_matrix(level, laplacian_matrix, eu, test);

                int isdiff = 0;
                for (int zi = 0; zi <= level; zi++) {
                    for (int yi = 0; yi <= level; yi++) {
                        for (int xi = 0; xi <= level; xi++) {
                            int                 t      = sshex8_lidx(level, xi, yi, zi);
                            const accumulator_t diff   = test[t] - v[t];
                            const int           isbdry = (!xi || xi == level || !yi || yi == level || !zi || zi == level);
                            const int           wrong  = fabs(diff) > 1e-10;

                            if (wrong) {
                                printf("%d|(%d, %d, %d) %g (expected) - %g = %g ", t, xi, yi, zi, test[t], v[t], diff);
                                printf("%s\n", isbdry ? "(boundary)" : "");
                                isdiff++;
                            }
                        }
                    }
                }

                if (isdiff) {
                    printf("------------------------------\n");
                    printf("Input\n");
                    for (int zi = 0; zi <= level; zi++) {
                        for (int yi = 0; yi <= level; yi++) {
                            for (int xi = 0; xi <= level; xi++) {
                                int t = sshex8_lidx(level, xi, yi, zi);
                                printf("%d|(%d, %d, %d) %g\n", t, xi, yi, zi, eu[t]);
                            }
                        }
                    }
                    printf("------------------------------\n");
                    printf("Expected\n");
                    for (int zi = 0; zi <= level; zi++) {
                        for (int yi = 0; yi <= level; yi++) {
                            for (int xi = 0; xi <= level; xi++) {
                                int t = sshex8_lidx(level, xi, yi, zi);
                                printf("%d|(%d, %d, %d) %g\n", t, xi, yi, zi, test[t]);
                            }
                        }
                    }
                    printf("------------------------------\n");
                    printf("Actual\n");
                    for (int zi = 0; zi <= level; zi++) {
                        for (int yi = 0; yi <= level; yi++) {
                            for (int xi = 0; xi <= level; xi++) {
                                int t = sshex8_lidx(level, xi, yi, zi);
                                printf("%d|(%d, %d, %d) %g\n", t, xi, yi, zi, v[t]);
                            }
                        }
                    }
                }

                assert(!isdiff);
                free(test);
            }
#endif  // NDEBUG

            {
                // Scatter elemental data
                for (int d = 0; d < nxe; d++) {
                    assert(v[d] == v[d]);
#pragma omp atomic update
                    values[ev[d]] += v[d];
                }
            }
        }

        // Clean-up
        free(ev);
        free(eu);
        free(v);
    }

    return SFEM_SUCCESS;
}

int sshex8_laplacian_element_matrix(int                           level,
                                    const ptrdiff_t               nelements,
                                    const ptrdiff_t               nnodes,
                                    idx_t **const SFEM_RESTRICT   elements,
                                    geom_t **const SFEM_RESTRICT  points,
                                    scalar_t *const SFEM_RESTRICT values) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];
    const scalar_t      h = 1. / level;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];

        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];
        scalar_t m_fff[6];
        scalar_t fff[6];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 8; v++) {
            lx[v] = x[ev[v]];
            ly[v] = y[ev[v]];
            lz[v] = z[ev[v]];
        }

        hex8_fff(lx, ly, lz, 0.5, 0.5, 0.5, m_fff);
        hex8_sub_fff_0(m_fff, h, fff);

        accumulator_t element_matrix[8 * 8];
        hex8_laplacian_matrix_fff_integral(fff, &values[i * 64]);
    }

    return SFEM_SUCCESS;
}

int affine_sshex8_laplacian_stencil_apply_fff(const int                             level,
                                              const ptrdiff_t                       nelements,
                                              idx_t **const SFEM_RESTRICT           elements,
                                              const jacobian_t *const SFEM_RESTRICT g_fff,
                                              const real_t *const SFEM_RESTRICT     u,
                                              real_t *const SFEM_RESTRICT           values) {
    const int nxe = sshex8_nxe(level);
    const int txe = sshex8_txe(level);

    const int Lm1  = level - 1;
    const int Lm13 = Lm1 * Lm1 * Lm1;

#pragma omp parallel
    {
        // Allocation per thread
        scalar_t      *eu = malloc(nxe * sizeof(scalar_t));
        idx_t         *ev = malloc(nxe * sizeof(idx_t));
        accumulator_t *v  = malloc(nxe * sizeof(accumulator_t));

        scalar_t      element_u[8];
        accumulator_t element_vector[8];
        scalar_t      m_fff[6], fff[6];

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            {
                // Gather elemental data
                for (int d = 0; d < nxe; d++) {
                    ev[d] = elements[d][e];
                }

                for (int d = 0; d < nxe; d++) {
                    eu[d] = u[ev[d]];
                    assert(eu[d] == eu[d]);
                }

                memset(v, 0, nxe * sizeof(accumulator_t));

                for (int d = 0; d < 6; d++) {
                    m_fff[d] = g_fff[e * 6 + d];
                }
            }

            const scalar_t h = 1. / level;
            hex8_sub_fff_0(m_fff, h, fff);

            accumulator_t laplacian_matrix[8 * 8];
            hex8_laplacian_matrix_fff_integral(fff, laplacian_matrix);
            scalar_t laplacian_stencil[3 * 3 * 3];
            hex8_matrix_to_stencil(laplacian_matrix, laplacian_stencil);
            sshex8_stencil(level + 1, level + 1, level + 1, laplacian_stencil, eu, v);
            sshex8_surface_stencil(
                    level + 1, level + 1, level + 1, 1, level + 1, (level + 1) * (level + 1), laplacian_matrix, eu, v);

            {
                // Scatter elemental data
                for (int d = 0; d < nxe; d++) {
                    assert(v[d] == v[d]);
#pragma omp atomic update
                    values[ev[d]] += v[d];
                }
            }
        }

        // Clean-up
        free(ev);
        free(eu);
        free(v);
    }

    return SFEM_SUCCESS;
}

int affine_sshex8_laplacian_substructuring_inner_fff(const int                             level,
                                                     const ptrdiff_t                       nelements,
                                                     idx_t **const SFEM_RESTRICT           elements,
                                                     const jacobian_t *const SFEM_RESTRICT g_fff,
                                                     const real_t *const SFEM_RESTRICT     rhs,
                                                     real_t *const SFEM_RESTRICT           u) {
    const int nxe = sshex8_nxe(level);
    const int txe = sshex8_txe(level);

#pragma omp parallel
    {
        // Allocation per thread
        idx_t    *ev   = malloc(nxe * sizeof(idx_t));
        scalar_t *eu   = malloc(nxe * sizeof(scalar_t));
        scalar_t *erhs = malloc(nxe * sizeof(scalar_t));
        scalar_t *r    = malloc(nxe * sizeof(scalar_t));
        scalar_t *p    = malloc(nxe * sizeof(scalar_t));
        scalar_t *Ap   = malloc(nxe * sizeof(scalar_t));

        scalar_t      element_u[8];
        accumulator_t element_vector[8];
        scalar_t      laplacian_stencil[3 * 3 * 3];

        const scalar_t h = 1. / level;

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            {
                // Gather elemental data
                for (int d = 0; d < nxe; d++) {
                    ev[d] = elements[d][e];
                }

                for (int d = 0; d < nxe; d++) {
                    eu[d] = u[ev[d]];
                    assert(eu[d] == eu[d]);
                }

                for (int d = 0; d < nxe; d++) {
                    erhs[d] = rhs[ev[d]];
                    assert(erhs[d] == erhs[d]);
                }

                scalar_t fff[6];
                for (int d = 0; d < 6; d++) {
                    fff[d] = g_fff[e * 6 + d] * h;
                }

                accumulator_t laplacian_matrix[8 * 8];
                hex8_laplacian_matrix_fff_integral(fff, laplacian_matrix);
                hex8_matrix_to_stencil(laplacian_matrix, laplacian_stencil);
            }

            memset(r, 0, nxe * sizeof(scalar_t));
            sshex8_stencil(level + 1, level + 1, level + 1, laplacian_stencil, eu, r);

            for (int v = 0; v < nxe; v++) {
                erhs[v] -= r[v];
            }

            sshex8_zero_boundary(level + 1, level + 1, level + 1, eu);
            sshex8_zero_boundary(level + 1, level + 1, level + 1, erhs);

            // printf("----------------------------\n");
            // printf("R\n");
            // printf("----------------------------\n");
            // for (ptrdiff_t zi = 0; zi <= level; zi++) {
            //     for (ptrdiff_t yi = 0; yi <= level; yi++) {
            //         for (ptrdiff_t xi = 0; xi <= level; xi++) {
            //             const int   lidx = sshex8_lidx(level, xi, yi, zi);
            //             printf("%g ", r[lidx]);
            //         }
            //         printf("\t");
            //     }
            //     printf("\n");
            // }
            // printf("----------------------------\n");

            // printf("----------------------------\n");
            // printf("RHS\n");
            // printf("----------------------------\n");
            // for (ptrdiff_t zi = 0; zi <= level; zi++) {
            //     for (ptrdiff_t yi = 0; yi <= level; yi++) {
            //         for (ptrdiff_t xi = 0; xi <= level; xi++) {
            //             const int   lidx = sshex8_lidx(level, xi, yi, zi);
            //             printf("%g ", erhs[lidx]);
            //         }
            //         printf("\t");
            //     }
            //     printf("\n");
            // }
            // printf("----------------------------\n");

            int err = sshex8_stencil_cg(nxe,
                                        1e-8,
                                        1e-16,
                                        // Grid info
                                        level + 1,
                                        level + 1,
                                        level + 1,
                                        laplacian_stencil,
                                        //
                                        erhs,
                                        //
                                        r,
                                        p,
                                        Ap,
                                        //
                                        eu);

            if (SFEM_SUCCESS != err) {
                // fprintf(stderr, "FAILED to solved laplacian subsystem\n");
                SFEM_ERROR("FAILED to solved laplacian subsystem\n");

                continue;
            }

            // printf("----------------------------\n");
            // printf("OUT\n");
            // printf("----------------------------\n");
            // for (ptrdiff_t zi = 0; zi <= level; zi++) {
            //     for (ptrdiff_t yi = 0; yi <= level; yi++) {
            //         for (ptrdiff_t xi = 0; xi <= level; xi++) {
            //             const int lidx = sshex8_lidx(level, xi, yi, zi);
            //             printf("%g ", eu[lidx]);
            //         }
            //         printf("\t");
            //     }
            //     printf("\n");
            // }
            // printf("----------------------------\n");

            {
                // Scatter elemental data
                for (int zi = 1; zi < level; zi++) {
                    for (int yi = 1; yi < level; yi++) {
                        for (int xi = 1; xi < level; xi++) {
                            const int   lidx = sshex8_lidx(level, xi, yi, zi);
                            const idx_t gidx = ev[lidx];
                            u[gidx] += eu[lidx];
                        }
                    }
                }
            }
        }

        // Clean-up
        free(ev);
        free(eu);
        free(erhs);
        free(r);
        free(p);
        free(Ap);
    }

    return SFEM_SUCCESS;
}

int affine_sshex8_laplacian_bjacobi_fff(const int                             level,
                                        const ptrdiff_t                       nelements,
                                        idx_t **const SFEM_RESTRICT           elements,
                                        const jacobian_t *const SFEM_RESTRICT g_fff,
                                        const uint16_t *const                 count,
                                        const mask_t *const                   mask,
                                        const element_idx_t *const            adjaciency_table,
                                        const real_t *const SFEM_RESTRICT     rhs,
                                        real_t *const SFEM_RESTRICT           u) {
    const int nxe      = sshex8_nxe(level);
    const int txe      = sshex8_txe(level);
    const int nn       = level + 1;
    const int lystride = nn;
    const int lzstride = lystride * lystride;

#pragma omp parallel
    {
        // Allocation per thread
        idx_t    *ev    = malloc(nxe * sizeof(idx_t));
        scalar_t *eu    = calloc(nxe, sizeof(scalar_t));
        scalar_t *erhs  = calloc(nxe, sizeof(scalar_t));
        scalar_t *r     = calloc(nxe, sizeof(scalar_t));
        scalar_t *p     = calloc(nxe, sizeof(scalar_t));
        scalar_t *Ap    = calloc(nxe, sizeof(scalar_t));
        int      *emask = calloc(nxe, sizeof(int));

        scalar_t      element_u[8];
        accumulator_t element_vector[8];
        accumulator_t laplacian_matrix[8 * 8];
        scalar_t      laplacian_stencil[3 * 3 * 3];

        const scalar_t h = 1. / level;

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            {
                // Gather elemental data
                for (int d = 0; d < nxe; d++) {
                    ev[d] = elements[d][e];
                }

                for (int d = 0; d < nxe; d++) {
                    emask[d] = mask_get(ev[d], mask);
                }

                for (int d = 0; d < nxe; d++) {
                    eu[d] = u[ev[d]];
                    assert(eu[d] == eu[d]);
                }

                for (int d = 0; d < nxe; d++) {
                    erhs[d] = rhs[ev[d]];
                    assert(erhs[d] == erhs[d]);
                }

                scalar_t fff[6];
                for (int d = 0; d < 6; d++) {
                    fff[d] = g_fff[e * 6 + d] * h;
                }

                hex8_laplacian_matrix_fff_integral(fff, laplacian_matrix);
                hex8_matrix_to_stencil(laplacian_matrix, laplacian_stencil);
            }

            memset(r, 0, nxe * sizeof(scalar_t));

            const element_idx_t * const neighs = &adjaciency_table[e * 6];

            sshex8_stencil(nn, nn, nn, laplacian_stencil, eu, r);
            sshex8_surface_stencil(nn, nn, nn, 1, nn, nn * nn, laplacian_matrix, eu, r);

            // FIXME Handle 0-neumann boundary conditions and inner boundaries

            for (int v = 0; v < nxe; v++) {
                if (emask[v]) {
                    r[v] = eu[v];
                }
            }

            for (int v = 0; v < nxe; v++) {
                erhs[v] -= r[v];
            }

            // sshe8_print("RHS", nn, nn, nn, erhs);
            memset(eu, 0, nxe * sizeof(scalar_t));
            int err = sshex8_stencil_cg_constrained(nxe,
                                                    1e-8,
                                                    1e-16,
                                                    // Grid info
                                                    nn,
                                                    nn,
                                                    nn,
                                                    laplacian_stencil,
                                                    laplacian_matrix,
                                                    emask,
                                                    //
                                                    erhs,
                                                    //
                                                    r,
                                                    p,
                                                    Ap,
                                                    //
                                                    eu);

            if (SFEM_SUCCESS != err) {
                SFEM_ERROR("FAILED to solve laplacian subsystem\n");
                continue;
            }

            // Scatter elemental data
            for (int d = 0; d < nxe; d++) {
                if (emask[d]) continue;
                const idx_t gidx = ev[d];
#pragma omp atomic update
                u[gidx] += (eu[d] / count[gidx]);
            }
        }

        // Clean-up
        free(ev);
        free(emask);
        free(eu);
        free(erhs);
        free(r);
        free(p);
        free(Ap);
    }

    return SFEM_SUCCESS;
}
