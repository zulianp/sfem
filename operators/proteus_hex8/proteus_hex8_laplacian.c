#include "proteus_hex8_laplacian.h"

#include "hex8_inline_cpu.h"
#include "hex8_laplacian_inline_cpu.h"

#include "hex8_quadrature.h"

#include <math.h>
#include <stdio.h>

int proteus_hex8_nxe(int level) {
    const int corners = 8;
    const int edge_nodes = 12 * (level - 1);
    const int face_nodes = 6 * (level - 1) * (level - 1);
    const int vol_nodes = (level - 1) * (level - 1) * (level - 1);
    return corners + edge_nodes + face_nodes + vol_nodes;
}

int proteus_hex8_txe(int level) { return level * level * level; }

static SFEM_INLINE int proteus_hex8_lidx(const int L, const int x, const int y, const int z) {
    int Lp1 = L + 1;
    int ret = z * (Lp1 * Lp1) + y * Lp1 + x;
    assert(ret < proteus_hex8_nxe(L));
    assert(ret >= 0);
    return ret;
}

static SFEM_INLINE void hex8_sub_fff_0(const scalar_t *const SFEM_RESTRICT fff,
                                       const scalar_t h,
                                       scalar_t *const SFEM_RESTRICT sub_fff) {
    sub_fff[0] = fff[0] * h;
    sub_fff[1] = fff[1] * h;
    sub_fff[2] = fff[2] * h;
    sub_fff[3] = fff[3] * h;
    sub_fff[4] = fff[4] * h;
    sub_fff[5] = fff[5] * h;
}

int proteus_hex8_laplacian_apply(const int level,
                                 const ptrdiff_t nelements,
                                 ptrdiff_t interior_start,
                                 idx_t **const SFEM_RESTRICT elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT values) {
    const int nxe = proteus_hex8_nxe(level);
    const int txe = proteus_hex8_txe(level);

    const int proteus_to_std_hex8_corners[8] = {// Bottom
                                                proteus_hex8_lidx(level, 0, 0, 0),
                                                proteus_hex8_lidx(level, level, 0, 0),
                                                proteus_hex8_lidx(level, level, level, 0),
                                                proteus_hex8_lidx(level, 0, level, 0),

                                                // Top
                                                proteus_hex8_lidx(level, 0, 0, level),
                                                proteus_hex8_lidx(level, level, 0, level),
                                                proteus_hex8_lidx(level, level, level, level),
                                                proteus_hex8_lidx(level, 0, level, level)};

    const int n_qp = q27_n;
    const scalar_t *qx = q27_x;
    const scalar_t *qy = q27_y;
    const scalar_t *qz = q27_z;
    const scalar_t *qw = q27_w;

    // const int n_qp = q6_n;
    // const scalar_t *qx = q6_x;
    // const scalar_t *qy = q6_y;
    // const scalar_t *qz = q6_z;
    // const scalar_t *qw = q6_w;

    int Lm1 = level - 1;
    int Lm13 = Lm1 * Lm1 * Lm1;

#pragma omp parallel
    {
        // Allocation per thread
        scalar_t *eu = malloc(nxe * sizeof(scalar_t));
        idx_t *ev = malloc(nxe * sizeof(idx_t));
        accumulator_t *v = malloc(nxe * sizeof(accumulator_t));

        scalar_t x[8];
        scalar_t y[8];
        scalar_t z[8];

        scalar_t element_u[8];
        accumulator_t element_vector[8];
        scalar_t m_fff[6], fff[6];

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
                }

                memset(v, 0, nxe * sizeof(accumulator_t));
            }

            const scalar_t h = 1. / level;

            // Iterate over sub-elements
            for (int zi = 0; zi < level - 1; zi++) {
                for (int yi = 0; yi < level - 1; yi++) {
                    for (int xi = 0; xi < level - 1; xi++) {
                        // Convert to standard HEX8 local ordering (see 3-4 and 6-7)
                        int lev[8] = {// Bottom
                                      proteus_hex8_lidx(level, xi, yi, zi),
                                      proteus_hex8_lidx(level, xi + 1, yi, zi),
                                      proteus_hex8_lidx(level, xi + 1, yi + 1, zi),
                                      proteus_hex8_lidx(level, xi, yi + 1, zi),
                                      // Top
                                      proteus_hex8_lidx(level, xi, yi, zi + 1),
                                      proteus_hex8_lidx(level, xi + 1, yi, zi + 1),
                                      proteus_hex8_lidx(level, xi + 1, yi + 1, zi + 1),
                                      proteus_hex8_lidx(level, xi, yi + 1, zi + 1)};

                        for (int d = 0; d < 8; d++) {
                            element_u[d] = eu[lev[d]];
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
                            hex8_laplacian_apply_fff(
                                    fff, qx[k], qy[k], qz[k], qw[k], element_u, element_vector);
                        }

                        // Accumulate to macro-element buffer
                        for (int d = 0; d < 8; d++) {
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

int proteus_affine_hex8_laplacian_apply(const int level,
                                        const ptrdiff_t nelements,
                                        ptrdiff_t interior_start,
                                        idx_t **const SFEM_RESTRICT elements,
                                        geom_t **const SFEM_RESTRICT std_hex8_points,
                                        const real_t *const SFEM_RESTRICT u,
                                        real_t *const SFEM_RESTRICT values) {
    const int nxe = proteus_hex8_nxe(level);
    const int txe = proteus_hex8_txe(level);

    const int proteus_to_std_hex8_corners[8] = {// Bottom
                                                proteus_hex8_lidx(level, 0, 0, 0),
                                                proteus_hex8_lidx(level, level, 0, 0),
                                                proteus_hex8_lidx(level, level, level, 0),
                                                proteus_hex8_lidx(level, 0, level, 0),

                                                // Top
                                                proteus_hex8_lidx(level, 0, 0, level),
                                                proteus_hex8_lidx(level, level, 0, level),
                                                proteus_hex8_lidx(level, level, level, level),
                                                proteus_hex8_lidx(level, 0, level, level)};

    const int Lm1 = level - 1;
    const int Lm13 = Lm1 * Lm1 * Lm1;

#pragma omp parallel
    {
        // Allocation per thread
        scalar_t *eu = malloc(nxe * sizeof(scalar_t));
        idx_t *ev = malloc(nxe * sizeof(idx_t));
        accumulator_t *v = malloc(nxe * sizeof(accumulator_t));

        scalar_t x[8];
        scalar_t y[8];
        scalar_t z[8];

        scalar_t element_u[8];
        accumulator_t element_vector[8];
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

                for (int d = 0; d < nxe; d++) {
                    eu[d] = u[ev[d]];
                }

                memset(v, 0, nxe * sizeof(accumulator_t));
            }

            const scalar_t h = 1. / level;

            // We evaluate the jacobian at the center of the element
            // in case that it that the mapping is not linear
            hex8_fff(x, y, z, 0.5, 0.5, 0.5, m_fff);
            hex8_sub_fff_0(m_fff, h, fff);

#define PROTEUS_HEX8_USE_MV  // assemblying the elemental matrix is faster
#ifdef PROTEUS_HEX8_USE_MV
            scalar_t laplacian_matrix[8 * 8];
            hex8_laplacian_matrix_fff_integral(fff, laplacian_matrix);
#endif

            // Iterate over sub-elements
            for (int zi = 0; zi < level - 1; zi++) {
                for (int yi = 0; yi < level - 1; yi++) {
                    for (int xi = 0; xi < level - 1; xi++) {
                        // Convert to standard HEX8 local ordering (see 3-4 and 6-7)
                        int lev[8] = {// Bottom
                                      proteus_hex8_lidx(level, xi, yi, zi),
                                      proteus_hex8_lidx(level, xi + 1, yi, zi),
                                      proteus_hex8_lidx(level, xi + 1, yi + 1, zi),
                                      proteus_hex8_lidx(level, xi, yi + 1, zi),
                                      // Top
                                      proteus_hex8_lidx(level, xi, yi, zi + 1),
                                      proteus_hex8_lidx(level, xi + 1, yi, zi + 1),
                                      proteus_hex8_lidx(level, xi + 1, yi + 1, zi + 1),
                                      proteus_hex8_lidx(level, xi, yi + 1, zi + 1)};

                        for (int d = 0; d < 8; d++) {
                            element_u[d] = eu[lev[d]];
                        }

#ifdef PROTEUS_HEX8_USE_MV

                        for (int i = 0; i < 8; i++) {
                            element_vector[i] = 0;
                        }

                        for (int i = 0; i < 8; i++) {
                            const scalar_t *const row = &laplacian_matrix[i * 8];
                            const scalar_t ui = element_u[i];
                            for (int j = 0; j < 8; j++) {
                                element_vector[j] += ui * row[j];
                            }
                        }
#else
                        hex8_laplacian_apply_fff_integral(fff, element_u, element_vector);
#endif

                        // Accumulate to macro-element buffer
                        for (int d = 0; d < 8; d++) {
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
