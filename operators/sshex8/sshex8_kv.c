#include "sshex8_linear_elasticity.h"

#include "sfem_defs.h"

#include "hex8_inline_cpu.h"
#include "hex8_kelvin_voigt_newmark_inline_cpu.h"
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

int affine_sshex8_kv_apply(const int                    level,
                           const ptrdiff_t              nelements,
                           const ptrdiff_t              nnodes,
                           idx_t **const SFEM_RESTRICT  elements,
                           geom_t **const SFEM_RESTRICT points,
                           const real_t                 k,
                           const real_t                 K,
                           const real_t                 eta,
                           const real_t                 rho,
                           const real_t                 dt,
                           const real_t                 gamma,
                           const real_t                 beta,
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
            sshex8_kelvin_voigt_newmark_matrix(k, K, eta, rho, dt, gamma, beta, sub_adjugate, sub_determinant, element_matrix);

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

int affine_sshex8_kv_gradient(const int                    level,
                              const ptrdiff_t              nelements,
                              const ptrdiff_t              nnodes,
                              idx_t **const SFEM_RESTRICT  elements,
                              geom_t **const SFEM_RESTRICT points,
                              const real_t                 k,
                              const real_t                 K,
                              const real_t                 eta,
                              const real_t                 rho,
                              const ptrdiff_t              u_stride,
                              const real_t *const          ux,
                              const real_t *const          uy,
                              const real_t *const          uz,
                              const real_t *const          vx,
                              const real_t *const          vy,
                              const real_t *const          vz,
                              const real_t *const          ax,
                              const real_t *const          ay,
                              const real_t *const          az,
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
        scalar_t      *evel[3];
        scalar_t      *eax[3];
        accumulator_t *v[3];

        for (int d = 0; d < 3; d++) {
            eu[d]   = malloc(nxe * sizeof(scalar_t));
            evel[d] = malloc(nxe * sizeof(scalar_t));
            eax[d]  = malloc(nxe * sizeof(scalar_t));
            v[d]    = malloc(nxe * sizeof(accumulator_t));
        }

        idx_t *ev = malloc(nxe * sizeof(idx_t));

        scalar_t x[8];
        scalar_t y[8];
        scalar_t z[8];

        scalar_t m_adjugate[9], adjugate[9];

        scalar_t      element_u[3 * 8];
        scalar_t      element_vel[3 * 8];
        scalar_t      element_a[3 * 8];
        accumulator_t element_out[3 * 8];

        // Aliases for reduced complexity inside
        scalar_t *element_ux = &element_u[0 * 8];
        scalar_t *element_uy = &element_u[1 * 8];
        scalar_t *element_uz = &element_u[2 * 8];

        scalar_t *element_velx = &element_vel[0 * 8];
        scalar_t *element_vely = &element_vel[1 * 8];
        scalar_t *element_velz = &element_vel[2 * 8];

        scalar_t *element_ax = &element_a[0 * 8];
        scalar_t *element_ay = &element_a[1 * 8];
        scalar_t *element_az = &element_a[2 * 8];

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

                    evel[0][d] = vx[ev[d] * u_stride];
                    evel[1][d] = vy[ev[d] * u_stride];
                    evel[2][d] = vz[ev[d] * u_stride];

                    eax[0][d] = ax[ev[d] * u_stride];
                    eax[1][d] = ay[ev[d] * u_stride];
                    eax[2][d] = az[ev[d] * u_stride];

                    assert(eu[0][d] == eu[0][d]);
                    assert(eu[1][d] == eu[1][d]);
                    assert(eu[2][d] == eu[2][d]);
                    assert(evel[0][d] == evel[0][d]);
                    assert(evel[1][d] == evel[1][d]);
                    assert(evel[2][d] == evel[2][d]);
                    assert(eax[0][d] == eax[0][d]);
                    assert(eax[1][d] == eax[1][d]);
                    assert(eax[2][d] == eax[2][d]);
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
                            const int lidx  = lev[d];
                            element_ux[d]   = eu[0][lidx];
                            element_uy[d]   = eu[1][lidx];
                            element_uz[d]   = eu[2][lidx];
                            element_velx[d] = evel[0][lidx];
                            element_vely[d] = evel[1][lidx];
                            element_velz[d] = evel[2][lidx];
                            element_ax[d]   = eax[0][lidx];
                            element_ay[d]   = eax[1][lidx];
                            element_az[d]   = eax[2][lidx];
                        }

                        sshex8_kelvin_voigt_newmark_gradient_adj(k,
                                                                 K,
                                                                 eta,
                                                                 rho,
                                                                 sub_adjugate,
                                                                 sub_determinant,
                                                                 element_ux,
                                                                 element_uy,
                                                                 element_uz,
                                                                 element_velx,
                                                                 element_vely,
                                                                 element_velz,
                                                                 element_ax,
                                                                 element_ay,
                                                                 element_az,
                                                                 element_outx,
                                                                 element_outy,
                                                                 element_outz);
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
            free(evel[d]);
            free(v[d]);
        }
    }

    return SFEM_SUCCESS;
}

int affine_sshex8_kelvin_voigt_newmark_diag(const int                    level,
                                            const ptrdiff_t              nelements,
                                            const ptrdiff_t              nnodes,
                                            idx_t **const SFEM_RESTRICT  elements,
                                            geom_t **const SFEM_RESTRICT points,
                                            const real_t                 K,
                                            const real_t                 eta,
                                            const real_t                 rho,
                                            const real_t                 k,
                                            const real_t                 beta,
                                            const real_t                 gamma,
                                            const real_t                 dt,
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

            hex8_kelvin_voigt_newmark_diag(K, eta, rho, k, beta, gamma, dt, sub_adjugate, sub_determinant, element_diag);

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