#include "sshex8_neohookean_ogden.h"

#include "sfem_defs.h"
#include "sfem_unroll.h"

#include "hex8_inline_cpu.h"
#include "hex8_linear_elasticity_inline_cpu.h"
// #include "hex8_quadrature.h"
#include "line_quadrature.h"
#include "sshex8.h"

#include "hex8_neohookean_ogden_local.h"
#include "packed_elements.h"

int sshex8_neohookean_ogden_objective(int                               level,
                                      const ptrdiff_t                   nelements,
                                      const ptrdiff_t                   stride,
                                      const ptrdiff_t                   nnodes,
                                      idx_t **const SFEM_RESTRICT       elements,
                                      geom_t **const SFEM_RESTRICT      points,
                                      const real_t                      mu,
                                      const real_t                      lambda,
                                      const ptrdiff_t                   u_stride,
                                      const real_t *const SFEM_RESTRICT ux,
                                      const real_t *const SFEM_RESTRICT uy,
                                      const real_t *const SFEM_RESTRICT uz,
                                      const int                         is_element_wise,
                                      real_t *const SFEM_RESTRICT       out) {
    // FIXME
    out[0] = 1;
    return SFEM_SUCCESS;
}

int sshex8_neohookean_ogden_objective_steps(int                               level,
                                            const ptrdiff_t                   nelements,
                                            const ptrdiff_t                   stride,
                                            const ptrdiff_t                   nnodes,
                                            idx_t **const SFEM_RESTRICT       elements,
                                            geom_t **const SFEM_RESTRICT      points,
                                            const real_t                      mu,
                                            const real_t                      lambda,
                                            const ptrdiff_t                   u_stride,
                                            const real_t *const SFEM_RESTRICT ux,
                                            const real_t *const SFEM_RESTRICT uy,
                                            const real_t *const SFEM_RESTRICT uz,
                                            const ptrdiff_t                   inc_stride,
                                            const real_t *const SFEM_RESTRICT incx,
                                            const real_t *const SFEM_RESTRICT incy,
                                            const real_t *const SFEM_RESTRICT incz,
                                            const int                         nsteps,
                                            const real_t *const SFEM_RESTRICT steps,
                                            real_t *const SFEM_RESTRICT       out) {
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

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;
    const scalar_t h = 1. / level;

#pragma omp parallel
    {
        idx_t *ev = malloc(nxe * sizeof(idx_t));
        scalar_t *x = malloc(8 * sizeof(scalar_t));
        scalar_t *y = malloc(8 * sizeof(scalar_t));
        scalar_t *z = malloc(8 * sizeof(scalar_t));
        scalar_t *out_local = (scalar_t *)calloc(nsteps, sizeof(scalar_t));
        scalar_t *eu[3];
        scalar_t *einc[3];
        for (int d = 0; d < 3; d++) {
            eu[d]   = malloc(nxe * sizeof(scalar_t));
            einc[d] = malloc(nxe * sizeof(scalar_t));
        }

#pragma omp for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            {
                // Gather elemental data
                for (int d = 0; d < nxe; d++) {
                    ev[d] = elements[d][i * stride];
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
                }

                for (int d = 0; d < nxe; d++) {
                    einc[0][d] = incx[ev[d] * inc_stride];
                    einc[1][d] = incy[ev[d] * inc_stride];
                    einc[2][d] = incz[ev[d] * inc_stride];
                }
            }

            for (int zi = 0; zi < level; zi++) {
                for (int yi = 0; yi < level; yi++) {
                    for (int xi = 0; xi < level; xi++) {
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

                        scalar_t mux[8];
                        scalar_t muy[8];
                        scalar_t muz[8];

                        for (int s = 0; s < nsteps; s++) {
                            for (int d = 0; d < 8; d++) {
                                mux[d] = eu[0][lev[d]] + einc[0][lev[d]] * steps[s];
                                muy[d] = eu[1][lev[d]] + einc[1][lev[d]] * steps[s];
                                muz[d] = eu[2][lev[d]] + einc[2][lev[d]] * steps[s];
                            }

                            scalar_t jacobian_adjugate[9];
                            scalar_t jacobian_determinant = 0;
                            scalar_t sub_adjugate[9];
                            scalar_t sub_determinant;

                            for (int kz = 0; kz < n_qp; kz++) {
                                for (int ky = 0; ky < n_qp; ky++) {
                                    for (int kx = 0; kx < n_qp; kx++) {
                                        hex8_adjugate_and_det(x,
                                                              y,
                                                              z,
                                                              (xi + qx[kx]) * h,
                                                              (yi + qx[ky]) * h,
                                                              (zi + qx[kz]) * h,
                                                              jacobian_adjugate,
                                                              &jacobian_determinant);
                                        hex8_sub_adj_0(
                                                jacobian_adjugate, jacobian_determinant, h, sub_adjugate, &sub_determinant);

                                        assert(jacobian_determinant == jacobian_determinant);
                                        assert(jacobian_determinant != 0);

                                        hex8_neohookean_ogden_objective_at_qp(sub_adjugate,
                                                                              sub_determinant,
                                                                              qx[kx],
                                                                              qx[ky],
                                                                              qx[kz],
                                                                              qw[kx] * qw[ky] * qw[kz],
                                                                              mu,
                                                                              lambda,
                                                                              mux,
                                                                              muy,
                                                                              muz,
                                                                              &out_local[s]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for (int s = 0; s < nsteps; s++) {
#pragma omp atomic update
            out[s] += out_local[s];
        }

        free(out_local);

        for (int d = 0; d < 3; d++) {
            free(eu[d]);
            free(einc[d]);
        }

        free(ev);
        free(x);
        free(y);
        free(z);
    }

    for (int s = 0; s < nsteps; s++) {
        if(isnan(out[s])) {
            out[s] = 1e10;
        }
    }

    return SFEM_SUCCESS;
}

int sshex8_neohookean_ogden_gradient(int                               level,
                                     const ptrdiff_t                   nelements,
                                     const ptrdiff_t                   stride,
                                     const ptrdiff_t                   nnodes,
                                     idx_t **const SFEM_RESTRICT       elements,
                                     geom_t **const SFEM_RESTRICT      points,
                                     const real_t                      mu,
                                     const real_t                      lambda,
                                     const ptrdiff_t                   u_stride,
                                     const real_t *const SFEM_RESTRICT ux,
                                     const real_t *const SFEM_RESTRICT uy,
                                     const real_t *const SFEM_RESTRICT uz,
                                     const ptrdiff_t                   out_stride,
                                     real_t *const SFEM_RESTRICT       outx,
                                     real_t *const SFEM_RESTRICT       outy,
                                     real_t *const SFEM_RESTRICT       outz) {
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

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

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

        scalar_t      element_u[3 * 8];
        accumulator_t eout[3 * 8];
        scalar_t      element_matrix[(3 * 8) * (3 * 8)];

        // Aliases for reduced complexity inside
        scalar_t *element_ux = &element_u[0 * 8];
        scalar_t *element_uy = &element_u[1 * 8];
        scalar_t *element_uz = &element_u[2 * 8];

        scalar_t *eoutx = &eout[0 * 8];
        scalar_t *eouty = &eout[1 * 8];
        scalar_t *eoutz = &eout[2 * 8];

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            {
                // Gather elemental data
                for (int d = 0; d < nxe; d++) {
                    ev[d] = elements[d][e * stride];
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
                            const int lidx = lev[d];
                            element_ux[d]  = eu[0][lidx];
                            element_uy[d]  = eu[1][lidx];
                            element_uz[d]  = eu[2][lidx];
                        }

                        for (int d = 0; d < 3 * 8; d++) {
                            eout[d] = 0;
                        }

                        scalar_t jacobian_adjugate[9];
                        scalar_t jacobian_determinant;
                        scalar_t sub_adjugate[9];
                        scalar_t sub_determinant;

                        for (int kz = 0; kz < n_qp; kz++) {
                            for (int ky = 0; ky < n_qp; ky++) {
                                for (int kx = 0; kx < n_qp; kx++) {
                                    hex8_adjugate_and_det(x,
                                                          y,
                                                          z,
                                                          (xi + qx[kx]) * h,
                                                          (yi + qx[ky]) * h,
                                                          (zi + qx[kz]) * h,
                                                          jacobian_adjugate,
                                                          &jacobian_determinant);
                                    hex8_sub_adj_0(jacobian_adjugate, jacobian_determinant, h, sub_adjugate, &sub_determinant);
                                    assert(jacobian_determinant == jacobian_determinant);
                                    assert(jacobian_determinant != 0);

                                    hex8_neohookean_ogden_grad(sub_adjugate,
                                                               sub_determinant,
                                                               qx[kx],
                                                               qx[ky],
                                                         qx[kz],
                                                         qw[kx] * qw[ky] * qw[kz],
                                                         mu,
                                                         lambda,
                                                         element_ux,
                                                         element_uy,
                                                         element_uz,
                                                         eoutx,
                                                         eouty,
                                                         eoutz);
                                }
                            }
                        }
                        // Accumulate to macro-element buffer
                        for (int d = 0; d < 8; d++) {
                            const int lidx = lev[d];
                            v[0][lidx] += eoutx[d];
                            v[1][lidx] += eouty[d];
                            v[2][lidx] += eoutz[d];
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

int sshex8_neohookean_ogden_hessian_partial_assembly(int                                  level,
                                                     const ptrdiff_t                      nelements,
                                                     const ptrdiff_t                      stride,
                                                     idx_t **const SFEM_RESTRICT          elements,
                                                     geom_t **const SFEM_RESTRICT         points,
                                                     const real_t                         mu,
                                                     const real_t                         lambda,
                                                     const ptrdiff_t                      u_stride,
                                                     const real_t *const SFEM_RESTRICT    ux,
                                                     const real_t *const SFEM_RESTRICT    uy,
                                                     const real_t *const SFEM_RESTRICT    uz,
                                                     metric_tensor_t *const SFEM_RESTRICT partial_assembly) {
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

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

#pragma omp parallel
    {
        // Allocation per thread
        scalar_t *eu[3];
        for (int d = 0; d < 3; d++) {
            eu[d] = malloc(nxe * sizeof(scalar_t));
        }

        idx_t *ev = malloc(nxe * sizeof(idx_t));

        scalar_t x[8];
        scalar_t y[8];
        scalar_t z[8];

        scalar_t element_u[3 * 8];

        // Aliases for reduced complexity inside
        scalar_t *element_ux = &element_u[0 * 8];
        scalar_t *element_uy = &element_u[1 * 8];
        scalar_t *element_uz = &element_u[2 * 8];

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            {
                // Gather elemental data
                for (int d = 0; d < nxe; d++) {
                    ev[d] = elements[d][e * stride];
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
            }

            const scalar_t h  = 1. / level;
            const scalar_t h3 = h * h * h;

            metric_tensor_t *const pai = &partial_assembly[e * HEX8_S_IKMN_SIZE];
            for (int d = 0; d < HEX8_S_IKMN_SIZE; d++) {
                pai[d] = 0;
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

                        for (int d = 0; d < 8; d++) {
                            const int lidx = lev[d];
                            element_ux[d]  = eu[0][lidx];
                            element_uy[d]  = eu[1][lidx];
                            element_uz[d]  = eu[2][lidx];
                        }

                        scalar_t jacobian_adjugate[9];
                        scalar_t jacobian_determinant;
                        scalar_t sub_adjugate[9];
                        scalar_t sub_determinant;

                        for (int kz = 0; kz < n_qp; kz++) {
                            for (int ky = 0; ky < n_qp; ky++) {
                                for (int kx = 0; kx < n_qp; kx++) {
                                    hex8_adjugate_and_det(x,
                                                          y,
                                                          z,
                                                          (xi + qx[kx]) * h,
                                                          (yi + qx[ky]) * h,
                                                          (zi + qx[kz]) * h,
                                                          jacobian_adjugate,
                                                          &jacobian_determinant);
                                    hex8_sub_adj_0(jacobian_adjugate, jacobian_determinant, h, sub_adjugate, &sub_determinant);
                                    assert(jacobian_determinant == jacobian_determinant);
                                    assert(jacobian_determinant != 0);

                                    scalar_t F[9] = {0};
                                    hex8_F(sub_adjugate,
                                           sub_determinant,
                                           qx[kx],
                                           qx[ky],
                                           qx[kz],
                                           element_ux,
                                           element_uy,
                                           element_uz,
                                           F);

                                    scalar_t partial_assembly_local[HEX8_S_IKMN_SIZE] = {0};
                                    hex8_S_ikmn_neohookean(sub_adjugate,
                                                           sub_determinant,
                                                           qx[kx],
                                                           qx[ky],
                                                           qx[kz],
                                                           F,
                                                           mu,
                                                           lambda,
                                                           qw[kx] * qw[ky] * qw[kz],
                                                           partial_assembly_local);

                                    for (int d = 0; d < HEX8_S_IKMN_SIZE; d++) {
                                        pai[d] += partial_assembly_local[d] * h3;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Clean-up
        free(ev);

        for (int d = 0; d < 3; d++) {
            free(eu[d]);
        }
    }

    return SFEM_SUCCESS;
}

int sshex8_neohookean_ogden_partial_assembly_apply(int                                        level,
                                                   const ptrdiff_t                            nelements,
                                                   const ptrdiff_t                            stride,
                                                   idx_t **const SFEM_RESTRICT                elements,
                                                   const metric_tensor_t *const SFEM_RESTRICT partial_assembly,
                                                   const ptrdiff_t                            h_stride,
                                                   const real_t *const SFEM_RESTRICT          hx,
                                                   const real_t *const SFEM_RESTRICT          hy,
                                                   const real_t *const SFEM_RESTRICT          hz,
                                                   const ptrdiff_t                            out_stride,
                                                   real_t *const SFEM_RESTRICT                outx,
                                                   real_t *const SFEM_RESTRICT                outy,
                                                   real_t *const SFEM_RESTRICT                outz) {
    const int nxe = sshex8_nxe(level);
    const int txe = sshex8_txe(level);

    scalar_t Wimpn_compressed[10];
    hex8_Wimpn_compressed(Wimpn_compressed);

#pragma omp parallel
    {
        // Allocation per thread
        scalar_t *eh[3];
        scalar_t *v[3];
        for (int d = 0; d < 3; d++) {
            eh[d] = malloc(nxe * sizeof(scalar_t));
            v[d]  = malloc(nxe * sizeof(scalar_t));
        }

        idx_t *ev = malloc(nxe * sizeof(idx_t));

        scalar_t element_h[3 * 8];

        // Aliases for reduced complexity inside
        scalar_t *element_hx = &element_h[0 * 8];
        scalar_t *element_hy = &element_h[1 * 8];
        scalar_t *element_hz = &element_h[2 * 8];

        scalar_t  eout[3 * 8];
        scalar_t *eoutx = &eout[0 * 8];
        scalar_t *eouty = &eout[1 * 8];
        scalar_t *eoutz = &eout[2 * 8];

#ifdef SFEM_ENABLE_BLAS
        scalar_t *X = malloc(txe * 3 * 8 * sizeof(scalar_t));
        scalar_t *Y = malloc(txe * 3 * 8 * sizeof(scalar_t));
#endif

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            {
                // Gather elemental data
                for (int d = 0; d < nxe; d++) {
                    ev[d] = elements[d][e * stride];
                }

                for (int d = 0; d < nxe; d++) {
                    const ptrdiff_t idx = ev[d] * h_stride;
                    eh[0][d]            = hx[idx];
                    eh[1][d]            = hy[idx];
                    eh[2][d]            = hz[idx];
                }

                for (int d = 0; d < 3; d++) {
                    memset(v[d], 0, nxe * sizeof(scalar_t));
                }
            }

            scalar_t element_matrix[24 * 24];
            hex8_neohookean_hessian_from_S_ikmn(&partial_assembly[e * HEX8_S_IKMN_SIZE], Wimpn_compressed, element_matrix);

#ifdef SFEM_ENABLE_BLAS
            sshex8_SoA_pack_elements(level, eh, X);
            packed_elements_matmul(24, txe, 24, element_matrix, X, Y);
            sshex8_SoA_unpack_add_elements(level, Y, v);
#else
            // Iterate over sub-elements
            for (int zi = 0; zi < level; zi++) {
                for (int yi = 0; yi < level; yi++) {
                    for (int xi = 0; xi < level; xi++) {
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
                            element_hx[d]  = eh[0][lidx];
                            element_hy[d]  = eh[1][lidx];
                            element_hz[d]  = eh[2][lidx];
                        }

                        for (int d = 0; d < 3 * 8; d++) {
                            eout[d] = 0;
                        }

                        for (int i = 0; i < 3 * 8; i++) {
                            const scalar_t *const SFEM_RESTRICT col = &element_matrix[i * 3 * 8];
                            const scalar_t        ui  = element_h[i];
#pragma omp simd
                            for (int j = 0; j < 3 * 8; j++) {
                                eout[j] += ui * col[j];
                            }
                        }

                        for (int d = 0; d < 8; d++) {
                            const int lidx = lev[d];
                            v[0][lidx] += eoutx[d];
                            v[1][lidx] += eouty[d];
                            v[2][lidx] += eoutz[d];
                        }
                    }
                }
            }
#endif

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
            free(eh[d]);
            free(v[d]);
        }

#ifdef SFEM_ENABLE_BLAS
        free(X);
        free(Y);
#endif
    }

    return SFEM_SUCCESS;
}

int sshex8_neohookean_ogden_compressed_partial_assembly_apply(int                                     level,
                                                              const ptrdiff_t                         nelements,
                                                              const ptrdiff_t                         stride,
                                                              idx_t **const SFEM_RESTRICT             elements,
                                                              const compressed_t *const SFEM_RESTRICT partial_assembly,
                                                              const scaling_t *const SFEM_RESTRICT    scaling,
                                                              const ptrdiff_t                         h_stride,
                                                              const real_t *const SFEM_RESTRICT       hx,
                                                              const real_t *const SFEM_RESTRICT       hy,
                                                              const real_t *const SFEM_RESTRICT       hz,
                                                              const ptrdiff_t                         out_stride,
                                                              real_t *const SFEM_RESTRICT             outx,
                                                              real_t *const SFEM_RESTRICT             outy,
                                                              real_t *const SFEM_RESTRICT             outz) {
    SFEM_IMPLEMENT_ME();
    return SFEM_FAILURE;
}