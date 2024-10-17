#include "proteus_hex8_linear_elasticity.h"

#include "sfem_defs.h"

#include "hex8_inline_cpu.h"
#include "hex8_linear_elasticity_inline_cpu.h"
#include "hex8_quadrature.h"
#include "proteus_hex8.h"

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

static SFEM_INLINE void hex8_sub_adj_0(const scalar_t *const SFEM_RESTRICT adjugate,
                                       const scalar_t determinant,
                                       const scalar_t h,
                                       scalar_t *const SFEM_RESTRICT sub_adjugate,
                                       scalar_t *const SFEM_RESTRICT sub_determinant) {
    const scalar_t x0 = POW2(h);
    sub_adjugate[0] = adjugate[0] * x0;
    sub_adjugate[1] = adjugate[1] * x0;
    sub_adjugate[2] = adjugate[2] * x0;
    sub_adjugate[3] = adjugate[3] * x0;
    sub_adjugate[4] = adjugate[4] * x0;
    sub_adjugate[5] = adjugate[5] * x0;
    sub_adjugate[6] = adjugate[6] * x0;
    sub_adjugate[7] = adjugate[7] * x0;
    sub_adjugate[8] = adjugate[8] * x0;
    sub_determinant[0] = determinant * (POW3(h));
}

int proteus_hex8_linear_elasticity_apply(const int level,
                                         const ptrdiff_t nelements,
                                         const ptrdiff_t nnodes,
                                         idx_t **const SFEM_RESTRICT elements,
                                         geom_t **const SFEM_RESTRICT points,
                                         const real_t mu,
                                         const real_t lambda,
                                         const ptrdiff_t u_stride,
                                         const real_t *const ux,
                                         const real_t *const uy,
                                         const real_t *const uz,
                                         const ptrdiff_t out_stride,
                                         real_t *const outx,
                                         real_t *const outy,
                                         real_t *const outz) {
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

    const int n_qp = q6_n;
    const scalar_t *qx = q6_x;
    const scalar_t *qy = q6_y;
    const scalar_t *qz = q6_z;
    const scalar_t *qw = q6_w;

    // const int n_qp = q27_n;
    // const scalar_t *qx = q27_x;
    // const scalar_t *qy = q27_y;
    // const scalar_t *qz = q27_z;
    // const scalar_t *qw = q27_w;

    // const int n_qp = q58_n;
    // const scalar_t *qx = q58_x;
    // const scalar_t *qy = q58_y;
    // const scalar_t *qz = q58_z;
    // const scalar_t *qw = q58_w;

    int Lm1 = level - 1;
    int Lm13 = Lm1 * Lm1 * Lm1;

#pragma omp parallel
    {
        // Allocation per thread
        scalar_t *eu[3];
        accumulator_t *v[3];

        for (int d = 0; d < 3; d++) {
            eu[d] = malloc(nxe * sizeof(scalar_t));
            v[d] = malloc(nxe * sizeof(accumulator_t));
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
                        for (int k = 0; k < n_qp; k++) {
                            // 1) Compute qp in macro-element coordinates
                            const scalar_t m_qx = h * qx[k] + tx;
                            const scalar_t m_qy = h * qy[k] + ty;
                            const scalar_t m_qz = h * qz[k] + tz;

                            // 2) Evaluate Adjugate
                            scalar_t adjugate[9];
                            scalar_t jacobian_determinant;
                            hex8_adjugate_and_det(
                                    x, y, z, qx[k], qy[k], qz[k], adjugate, &jacobian_determinant);

                            // 3) Transform to sub-FFF
                            scalar_t sub_adjugate[9];
                            scalar_t sub_determinant;
                            hex8_sub_adj_0(adjugate,
                                           jacobian_determinant,
                                           h,
                                           sub_adjugate,
                                           &sub_determinant);

                            assert(sub_determinant == sub_determinant);
                            assert(sub_determinant != 0);

                            // // Evaluate y = op * x
                            hex8_linear_elasticity_apply_adj(mu,
                                                             lambda,
                                                             sub_adjugate,
                                                             sub_determinant,
                                                             qx[k],
                                                             qy[k],
                                                             qz[k],
                                                             qw[k],
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

int proteus_affine_hex8_linear_elasticity_apply(const int level,
                                                const ptrdiff_t nelements,
                                                const ptrdiff_t nnodes,
                                                idx_t **const SFEM_RESTRICT elements,
                                                geom_t **const SFEM_RESTRICT points,
                                                const real_t mu,
                                                const real_t lambda,
                                                const ptrdiff_t u_stride,
                                                const real_t *const ux,
                                                const real_t *const uy,
                                                const real_t *const uz,
                                                const ptrdiff_t out_stride,
                                                real_t *const outx,
                                                real_t *const outy,
                                                real_t *const outz) {
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
    int Lm1 = level - 1;
    int Lm13 = Lm1 * Lm1 * Lm1;

#pragma omp parallel
    {
        // Allocation per thread
        scalar_t *eu[3];
        accumulator_t *v[3];

        for (int d = 0; d < 3; d++) {
            eu[d] = malloc(nxe * sizeof(scalar_t));
            v[d] = malloc(nxe * sizeof(accumulator_t));
        }

        idx_t *ev = malloc(nxe * sizeof(idx_t));

        scalar_t x[8];
        scalar_t y[8];
        scalar_t z[8];

        scalar_t m_adjugate[9], adjugate[9];

        scalar_t element_u[3][8];
        accumulator_t element_out[3][8];
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

            // hex8_linear_elasticity_matrix(
            //         mu, lambda, adjugate, jacobian_determinant, element_matrix);
            // print_matrix(8*3, 8*3, element_matrix);

            hex8_linear_elasticity_matrix(
                    mu, lambda, sub_adjugate, sub_determinant, element_matrix);

            // Iterate over sub-elements
            for (int zi = 0; zi < level; zi++) {
                for (int yi = 0; yi < level; yi++) {
                    for (int xi = 0; xi < level; xi++) {
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
                            element_u[0][d] = eu[0][lev[d]];
                            element_u[1][d] = eu[1][lev[d]];
                            element_u[2][d] = eu[2][lev[d]];
                        }

                        for (int d = 0; d < 8; d++) {
                            element_out[0][d] = 0;
                            element_out[1][d] = 0;
                            element_out[2][d] = 0;
                        }

                        for (int d = 0; d < 3; d++) {
                            const scalar_t *const block = &element_matrix[d * 8 * (3 * 8)];

                            for (int i = 0; i < 8; i++) {
                                const int offset = i * (3 * 8);

                                const scalar_t *const xcol = &block[offset];
                                for (int j = 0; j < 8; j++) {
                                    element_out[d][i] += xcol[j] * element_u[0][j];
                                }

                                const scalar_t *const ycol = &block[offset + 8];
                                for (int j = 0; j < 8; j++) {
                                    element_out[d][i] += ycol[j] * element_u[1][j];
                                }

                                const scalar_t *const zcol = &block[offset + 2 * 8];
                                for (int j = 0; j < 8; j++) {
                                    element_out[d][i] += zcol[j] * element_u[2][j];
                                }
                            }
                        }

                        // printf("%d, %d, %d)\n", xi, yi, zi);
                        // for (int d = 0; d < 8; d++) {
                        //     printf("%d) %g => %g\n", lev[d], element_u[0][d], element_out[0][d]);
                        // }
                        // printf("\n");

                        // Accumulate to macro-element buffer
                        for (int d = 0; d < 8; d++) {
                            v[0][lev[d]] += element_out[0][d];
                            v[1][lev[d]] += element_out[1][d];
                            v[2][lev[d]] += element_out[2][d];
                        }
                    }
                }
            }

            for (int d = 0; d < nxe; d++) {
                printf("%d)\t%g -> %g\n", ev[d], eu[0][d], v[0][d]);
            }

            printf("\n");

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
