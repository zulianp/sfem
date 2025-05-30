#include "sshex8_linear_elasticity.h"

#include "sfem_defs.h"

#include "hex8_inline_cpu.h"
#include "hex8_linear_elasticity_inline_cpu.h"
// #include "hex8_quadrature.h"
#include "line_quadrature.h"
#include "sshex8.h"

#ifndef POW3
#define POW3(x) ((x) * (x) * (x))
#endif

#include <stdio.h>
#include <stdlib.h>
static void print_matrix(int r, int c, const scalar_t *const m) {
    printf("A = [");
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%g", m[i * c + j]);

            if (j < c - 1) {
                printf(",");
            }
        }
        if (i < r - 1) {
            printf(";\n");
        }
    }
    printf("]\n");
}

// FIXME there is probably a bug in here!
int sshex8_linear_elasticity_apply(const int                    level,
                                   const ptrdiff_t              nelements,
                                   const ptrdiff_t              nnodes,
                                   idx_t **const SFEM_RESTRICT  elements,
                                   geom_t **const SFEM_RESTRICT points,
                                   const real_t                 mu,
                                   const real_t                 lambda,
                                   const ptrdiff_t              u_stride,
                                   const real_t *const          ux,
                                   const real_t *const          uy,
                                   const real_t *const          uz,
                                   const ptrdiff_t              out_stride,
                                   real_t *const                outx,
                                   real_t *const                outy,
                                   real_t *const                outz) {
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

    int SFEM_HEX8_QUADRATURE_ORDER = 2;
    SFEM_READ_ENV(SFEM_HEX8_QUADRATURE_ORDER, atoi);
    // printf("SFEM_HEX8_QUADRATURE_ORDER = %d\n", SFEM_HEX8_QUADRATURE_ORDER);

    int             n_qp = line_q3_n;
    const scalar_t *qx   = line_q3_x;
    const scalar_t *qw   = line_q3_w;
    if (SFEM_HEX8_QUADRATURE_ORDER == 1) {
        n_qp = line_q2_n;
        qx   = line_q2_x;
        qw   = line_q2_w;
    } else if (SFEM_HEX8_QUADRATURE_ORDER == 5) {
        n_qp = line_q6_n;
        qx   = line_q6_x;
        qw   = line_q6_w;
    }

    int Lm1  = level - 1;
    int Lm13 = Lm1 * Lm1 * Lm1;

#pragma omp parallel
    {
        // Allocation per thread
        scalar_t      *eu[3];
        accumulator_t *v[3];

        for (int d = 0; d < 3; d++) {
            eu[d] = malloc(nxe * sizeof(scalar_t));
            v[d]  = malloc(nxe * sizeof(accumulator_t));
        }

        idx_t *ev = malloc(nxe * sizeof(idx_t));

        scalar_t x[8];
        scalar_t y[8];
        scalar_t z[8];

        scalar_t element_ux[8];
        scalar_t element_uy[8];
        scalar_t element_uz[8];

        accumulator_t element_outx[8];
        accumulator_t element_outy[8];
        accumulator_t element_outz[8];

        scalar_t m_adjugate[9], adjugate[9];

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
                    eu[0][d] = ux[ev[d] * u_stride];
                    eu[1][d] = uy[ev[d] * u_stride];
                    eu[2][d] = uz[ev[d] * u_stride];

                    assert(eu[0][d] == eu[0][d]);
                    assert(eu[1][d] == eu[1][d]);
                    assert(eu[2][d] == eu[2][d]);
                }

                for (int d = 0; d < 3; d++) {
                    memset(v[d], 0, nxe * sizeof(accumulator_t));
                }
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
                            element_ux[d] = eu[0][lev[d]];
                            element_uy[d] = eu[1][lev[d]];
                            element_uz[d] = eu[2][lev[d]];
                        }

                        for (int d = 0; d < 8; d++) {
                            element_outx[d] = 0;
                            element_outy[d] = 0;
                            element_outz[d] = 0;
                        }

                        // Translation
                        const scalar_t tx = (scalar_t)xi / level;
                        const scalar_t ty = (scalar_t)yi / level;
                        const scalar_t tz = (scalar_t)zi / level;

                        // Quadrature
                        for (int kz = 0; kz < n_qp; kz++) {
                            for (int ky = 0; ky < n_qp; ky++) {
                                for (int kx = 0; kx < n_qp; kx++) {
                                    // 1) Compute qp in macro-element coordinates
                                    const scalar_t m_qx = h * qx[kx] + tx;
                                    const scalar_t m_qy = h * qx[ky] + ty;
                                    const scalar_t m_qz = h * qx[kz] + tz;

                                    // 2) Evaluate Adjugate
                                    scalar_t adjugate[9];
                                    scalar_t jacobian_determinant;
                                    hex8_adjugate_and_det(x, y, z, qx[kx], qx[ky], qx[kz], adjugate, &jacobian_determinant);

                                    // 3) Transform to sub-FFF
                                    scalar_t sub_adjugate[9];
                                    scalar_t sub_determinant;
                                    hex8_sub_adj_0(adjugate, jacobian_determinant, h, sub_adjugate, &sub_determinant);

                                    assert(sub_determinant == sub_determinant);
                                    assert(sub_determinant != 0);

                                    // // Evaluate y = op * x
                                    hex8_linear_elasticity_apply_adj(mu,
                                                                     lambda,
                                                                     sub_adjugate,
                                                                     sub_determinant,
                                                                     qx[kx],
                                                                     qx[ky],
                                                                     qx[kz],
                                                                     qw[kx] * qw[ky] * qw[kz],
                                                                     element_ux,
                                                                     element_uy,
                                                                     element_uz,
                                                                     element_outx,
                                                                     element_outy,
                                                                     element_outz);
                                }

                                // Accumulate to macro-element buffer
                                for (int d = 0; d < 8; d++) {
                                    assert(element_outx[d] == element_outx[d]);
                                    assert(element_outy[d] == element_outy[d]);
                                    assert(element_outz[d] == element_outz[d]);

                                    v[0][lev[d]] += element_outx[d];
                                    v[1][lev[d]] += element_outy[d];
                                    v[2][lev[d]] += element_outz[d];
                                }
                            }
                        }
                    }
                }
            }

            // for (int d = 0; d < nxe; d++) {
            //     printf("%g\t", v[0][d]);
            // }

            // printf("\n");

            {
                // Scatter elemental data
                for (int d = 0; d < nxe; d++) {
                    assert(v[0][d] == v[0][d]);
                    assert(v[1][d] == v[1][d]);
                    assert(v[2][d] == v[2][d]);

#pragma omp atomic update
                    outx[ev[d] * out_stride] += v[0][d];

#pragma omp atomic update
                    outy[ev[d] * out_stride] += v[1][d];

#pragma omp atomic update
                    outz[ev[d] * out_stride] += v[2][d];
                }
            }
        }

        // Clean-up
        free(ev);

        for (int d = 0; d < 3; d++) {
            free(eu[d]);
            free(v[d]);
        }
    }

    return SFEM_SUCCESS;
}

// ---------------

int affine_sshex8_linear_elasticity_apply(const int                    level,
                                          const ptrdiff_t              nelements,
                                          const ptrdiff_t              nnodes,
                                          idx_t **const SFEM_RESTRICT  elements,
                                          geom_t **const SFEM_RESTRICT points,
                                          const real_t                 mu,
                                          const real_t                 lambda,
                                          const ptrdiff_t              u_stride,
                                          const real_t *const          ux,
                                          const real_t *const          uy,
                                          const real_t *const          uz,
                                          const ptrdiff_t              out_stride,
                                          real_t *const                outx,
                                          real_t *const                outy,
                                          real_t *const                outz) {
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

#pragma omp parallel
    {
        // Allocation per thread
        scalar_t      *eu[3];
        accumulator_t *v[3];

        for (int d = 0; d < 3; d++) {
            eu[d] = malloc(nxe * sizeof(scalar_t));
            v[d]  = malloc(nxe * sizeof(accumulator_t));
        }

        idx_t *ev = malloc(nxe * sizeof(idx_t));

        scalar_t x[8];
        scalar_t y[8];
        scalar_t z[8];

        scalar_t m_adjugate[9], adjugate[9];

        scalar_t      element_u[3 * 8];
        accumulator_t element_out[3 * 8];
        scalar_t      element_matrix[(3 * 8) * (3 * 8)];

        // Aliases for reduced complexity inside
        scalar_t *element_ux = &element_u[0 * 8];
        scalar_t *element_uy = &element_u[1 * 8];
        scalar_t *element_uz = &element_u[2 * 8];

        scalar_t *element_outx = &element_out[0 * 8];
        scalar_t *element_outy = &element_out[1 * 8];
        scalar_t *element_outz = &element_out[2 * 8];

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
                    eu[0][d] = ux[ev[d] * u_stride];
                    eu[1][d] = uy[ev[d] * u_stride];
                    eu[2][d] = uz[ev[d] * u_stride];

                    assert(eu[0][d] == eu[0][d]);
                    assert(eu[1][d] == eu[1][d]);
                    assert(eu[2][d] == eu[2][d]);
                }

                for (int d = 0; d < 3; d++) {
                    memset(v[d], 0, nxe * sizeof(accumulator_t));
                }
            }

            const scalar_t h = 1. / level;

            // 2) Evaluate Adjugate
            scalar_t adjugate[9];
            scalar_t jacobian_determinant;
            hex8_adjugate_and_det(x, y, z, 0.5, 0.5, 0.5, adjugate, &jacobian_determinant);

            // 3) Transform to sub-FFF
            scalar_t sub_adjugate[9];
            scalar_t sub_determinant;
            hex8_sub_adj_0(adjugate, jacobian_determinant, h, sub_adjugate, &sub_determinant);
            hex8_linear_elasticity_matrix(mu, lambda, sub_adjugate, sub_determinant, element_matrix);

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
                            const int lidx = lev[d];
                            element_ux[d]  = eu[0][lidx];
                            element_uy[d]  = eu[1][lidx];
                            element_uz[d]  = eu[2][lidx];
                        }

                        for (int d = 0; d < 3 * 8; d++) {
                            element_out[d] = 0;
                        }

                        for (int i = 0; i < 3 * 8; i++) {
                            const scalar_t *const col = &element_matrix[i * 3 * 8];
                            const scalar_t        ui  = element_u[i];
                            for (int j = 0; j < 3 * 8; j++) {
                                element_out[j] += ui * col[j];
                            }
                        }

                        // Accumulate to macro-element buffer
                        for (int d = 0; d < 8; d++) {
                            const int lidx = lev[d];
                            v[0][lidx] += element_outx[d];
                            v[1][lidx] += element_outy[d];
                            v[2][lidx] += element_outz[d];
                        }
                    }
                }
            }

            {
                // Scatter elemental data
                for (int d = 0; d < nxe; d++) {
                    const ptrdiff_t idx = ev[d] * out_stride;
#pragma omp atomic update
                    outx[idx] += v[0][d];

#pragma omp atomic update
                    outy[idx] += v[1][d];

#pragma omp atomic update
                    outz[idx] += v[2][d];
                }
            }
        }

        // Clean-up
        free(ev);

        for (int d = 0; d < 3; d++) {
            free(eu[d]);
            free(v[d]);
        }
    }

    return SFEM_SUCCESS;
}

int affine_sshex8_elasticity_bsr(const int                          level,
                                 const ptrdiff_t                    nelements,
                                 const ptrdiff_t                    nnodes,
                                 idx_t **const SFEM_RESTRICT        elements,
                                 geom_t **const SFEM_RESTRICT       points,
                                 const real_t                       mu,
                                 const real_t                       lambda,
                                 const count_t *const SFEM_RESTRICT rowptr,
                                 const idx_t *const SFEM_RESTRICT   colidx,
                                 real_t *const SFEM_RESTRICT        values) {
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
    int       Lm1                            = level - 1;
    int       Lm13                           = Lm1 * Lm1 * Lm1;

#pragma omp parallel
    {
        idx_t *ev = malloc(nxe * sizeof(idx_t));

        scalar_t x[8];
        scalar_t y[8];
        scalar_t z[8];

        scalar_t m_adjugate[9], adjugate[9];
        scalar_t element_matrix[(3 * 8) * (3 * 8)];

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
            }

            const scalar_t h = 1. / level;

            // 2) Evaluate Adjugate
            scalar_t adjugate[9];
            scalar_t jacobian_determinant;
            hex8_adjugate_and_det(x, y, z, 0.5, 0.5, 0.5, adjugate, &jacobian_determinant);

            // 3) Transform to sub-FFF
            scalar_t sub_adjugate[9];
            scalar_t sub_determinant;
            hex8_sub_adj_0(adjugate, jacobian_determinant, h, sub_adjugate, &sub_determinant);

            hex8_linear_elasticity_matrix(mu, lambda, sub_adjugate, sub_determinant, element_matrix);

            // Iterate over sub-elements
            for (int zi = 0; zi < level; zi++) {
                for (int yi = 0; yi < level; yi++) {
                    for (int xi = 0; xi < level; xi++) {
                        // Convert to standard HEX8 local ordering (see 3-4 and 6-7)
                        idx_t lev[8] = {// Bottom
                                        ev[sshex8_lidx(level, xi, yi, zi)],
                                        ev[sshex8_lidx(level, xi + 1, yi, zi)],
                                        ev[sshex8_lidx(level, xi + 1, yi + 1, zi)],
                                        ev[sshex8_lidx(level, xi, yi + 1, zi)],
                                        // Top
                                        ev[sshex8_lidx(level, xi, yi, zi + 1)],
                                        ev[sshex8_lidx(level, xi + 1, yi, zi + 1)],
                                        ev[sshex8_lidx(level, xi + 1, yi + 1, zi + 1)],
                                        ev[sshex8_lidx(level, xi, yi + 1, zi + 1)]};

                        hex8_local_to_global_bsr3(lev, element_matrix, rowptr, colidx, values);
                    }
                }
            }
        }

        // Clean-up
        free(ev);
    }

    return SFEM_SUCCESS;
}

int affine_sshex8_elasticity_crs_sym(const int                          level,
                                     const ptrdiff_t                    nelements,
                                     const ptrdiff_t                    nnodes,
                                     idx_t **const SFEM_RESTRICT        elements,
                                     geom_t **const SFEM_RESTRICT       points,
                                     const real_t                       mu,
                                     const real_t                       lambda,
                                     const count_t *const SFEM_RESTRICT rowptr,
                                     const idx_t *const SFEM_RESTRICT   colidx,
                                     // Output in SoA format (6)
                                     real_t **const SFEM_RESTRICT block_diag,
                                     real_t **const SFEM_RESTRICT block_offdiag) {
    // TODO Implement
    assert(0);
    return SFEM_FAILURE;
}

int affine_sshex8_linear_elasticity_diag(const int                    level,
                                         const ptrdiff_t              nelements,
                                         const ptrdiff_t              nnodes,
                                         idx_t **const SFEM_RESTRICT  elements,
                                         geom_t **const SFEM_RESTRICT points,
                                         const real_t                 mu,
                                         const real_t                 lambda,
                                         const ptrdiff_t              out_stride,
                                         real_t *const                outx,
                                         real_t *const                outy,
                                         real_t *const                outz) {
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
    int       Lm1                            = level - 1;
    int       Lm13                           = Lm1 * Lm1 * Lm1;

#pragma omp parallel
    {
        // Allocation per thread
        accumulator_t *v[3];

        for (int d = 0; d < 3; d++) {
            v[d] = malloc(nxe * sizeof(accumulator_t));
        }

        idx_t *ev = malloc(nxe * sizeof(idx_t));

        scalar_t x[8];
        scalar_t y[8];
        scalar_t z[8];

        scalar_t m_adjugate[9], adjugate[9];
        scalar_t element_diag[(3 * 8)];

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

                for (int d = 0; d < 3; d++) {
                    memset(v[d], 0, nxe * sizeof(accumulator_t));
                }
            }

            const scalar_t h = 1. / level;

            // 2) Evaluate Adjugate
            scalar_t adjugate[9];
            scalar_t jacobian_determinant;
            hex8_adjugate_and_det(x, y, z, 0.5, 0.5, 0.5, adjugate, &jacobian_determinant);

            // 3) Transform to sub-FFF
            scalar_t sub_adjugate[9];
            scalar_t sub_determinant;
            hex8_sub_adj_0(adjugate, jacobian_determinant, h, sub_adjugate, &sub_determinant);

            hex8_linear_elasticity_diag(mu, lambda, sub_adjugate, sub_determinant, element_diag);

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
                            v[0][lev[d]] += element_diag[0 * 8 + d];
                            v[1][lev[d]] += element_diag[1 * 8 + d];
                            v[2][lev[d]] += element_diag[2 * 8 + d];
                        }
                    }
                }
            }

            {
                // Scatter elemental data
                for (int d = 0; d < nxe; d++) {
                    const ptrdiff_t idx = ev[d] * out_stride;
#pragma omp atomic update
                    outx[idx] += v[0][d];

#pragma omp atomic update
                    outy[idx] += v[1][d];

#pragma omp atomic update
                    outz[idx] += v[2][d];
                }
            }
        }

        // Clean-up
        free(ev);
        for (int d = 0; d < 3; d++) {
            free(v[d]);
        }
    }

    return SFEM_SUCCESS;
}

int affine_sshex8_linear_elasticity_block_diag_sym(const int                    level,
                                                   const ptrdiff_t              nelements,
                                                   const ptrdiff_t              nnodes,
                                                   idx_t **const SFEM_RESTRICT  elements,
                                                   geom_t **const SFEM_RESTRICT points,
                                                   const real_t                 mu,
                                                   const real_t                 lambda,
                                                   const ptrdiff_t              out_stride,
                                                   real_t *const                out0,
                                                   real_t *const                out1,
                                                   real_t *const                out2,
                                                   real_t *const                out3,
                                                   real_t *const                out4,
                                                   real_t *const                out5) {
    const int nxe = sshex8_nxe(level);
    const int txe = sshex8_txe(level);

    int SFEM_HEX8_QUADRATURE_ORDER = 2;
    SFEM_READ_ENV(SFEM_HEX8_QUADRATURE_ORDER, atoi);
    // printf("SFEM_HEX8_QUADRATURE_ORDER = %d\n", SFEM_HEX8_QUADRATURE_ORDER);

    int             n_qp = line_q3_n;
    const scalar_t *qx   = line_q3_x;
    const scalar_t *qw   = line_q3_w;

    if (SFEM_HEX8_QUADRATURE_ORDER == 1) {
        n_qp = line_q2_n;
        qx   = line_q2_x;
        qw   = line_q2_w;
    } else if (SFEM_HEX8_QUADRATURE_ORDER == 5) {
        n_qp = line_q6_n;
        qx   = line_q6_x;
        qw   = line_q6_w;
    }

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
    int       Lm1                            = level - 1;
    int       Lm13                           = Lm1 * Lm1 * Lm1;

#pragma omp parallel
    {
        idx_t *ev = malloc(nxe * sizeof(idx_t));

        scalar_t x[8];
        scalar_t y[8];
        scalar_t z[8];

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
            }

            const scalar_t h = 1. / level;

            scalar_t sub_adjugate[9];
            scalar_t sub_determinant;
            {
                // 2) Evaluate Adjugate
                scalar_t adjugate[9];
                scalar_t jacobian_determinant;
                hex8_adjugate_and_det(x, y, z, 0.5, 0.5, 0.5, adjugate, &jacobian_determinant);

                // 3) Transform to sub-FFF
                hex8_sub_adj_0(adjugate, jacobian_determinant, h, sub_adjugate, &sub_determinant);
            }

            accumulator_t blocks[8][6];

            // Assemble the diagonal part of the matrix
            for (int edof_i = 0; edof_i < 8; edof_i++) {
                for (int k = 0; k < 6; k++) {
                    blocks[edof_i][k] = 0;
                }

                for (int zi = 0; zi < n_qp; zi++) {
                    for (int yi = 0; yi < n_qp; yi++) {
                        for (int xi = 0; xi < n_qp; xi++) {
                            scalar_t test_grad[3];
                            hex8_ref_shape_grad(edof_i, qx[xi], qx[yi], qx[zi], test_grad);
                            linear_elasticity_matrix_sym(mu,
                                                         lambda,
                                                         sub_adjugate,
                                                         sub_determinant,
                                                         test_grad,
                                                         test_grad,
                                                         qw[xi] * qw[yi] * qw[zi],
                                                         blocks[edof_i]);
                        }
                    }
                }
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

                        for (int edof_i = 0; edof_i < 8; edof_i++) {
                            const ptrdiff_t v = ev[lev[edof_i]];
                            // local to global
#pragma omp atomic update
                            out0[v * out_stride] += blocks[edof_i][0];
#pragma omp atomic update
                            out1[v * out_stride] += blocks[edof_i][1];
#pragma omp atomic update
                            out2[v * out_stride] += blocks[edof_i][2];
#pragma omp atomic update
                            out3[v * out_stride] += blocks[edof_i][3];
#pragma omp atomic update
                            out4[v * out_stride] += blocks[edof_i][4];
#pragma omp atomic update
                            out5[v * out_stride] += blocks[edof_i][5];
                        }
                    }
                }
            }
        }

        // Clean-up
        free(ev);
    }

    return SFEM_SUCCESS;
}
