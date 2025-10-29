#include "hex8_neohookean_ogden.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sfem_macros.h"
#include "sfem_vec.h"
#include "sortreduce.h"

#include "hex8_inline_cpu.h"
#include "line_quadrature.h"

<<<<<<< HEAD
#include "hex8_partial_assembly_neohookean_inline.h"
#include "hex8_neohookean_ogden_local.h"
=======
#include "hex8_neohookean_ogden_local.h"
#include "hex8_partial_assembly_neohookean_inline.h"
>>>>>>> origin/main

int hex8_neohookean_ogden_objective(const ptrdiff_t                   nelements,
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
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];

        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];

        scalar_t edispx[8];
        scalar_t edispy[8];
        scalar_t edispz[8];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int d = 0; d < 8; d++) {
            lx[d] = x[ev[d]];
            ly[d] = y[ev[d]];
            lz[d] = z[ev[d]];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            edispx[v]           = ux[idx];
            edispy[v]           = uy[idx];
            edispz[v]           = uz[idx];
        }

        scalar_t v = 0;
        hex8_neohookean_ogden_objective_integral(lx, ly, lz, n_qp, qx, qw, mu, lambda, edispx, edispy, edispz, &v);
        assert(v == v);

        if (is_element_wise) {
            out[i] = v;
        } else {
#pragma omp atomic update
            *out += v;
        }
    }

<<<<<<< HEAD
    if(*out != *out) {
=======
    if (*out != *out) {
>>>>>>> origin/main
        *out = 1e10;
    }

    return SFEM_SUCCESS;
}

int hex8_neohookean_ogden_objective_steps(const ptrdiff_t                   nelements,
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
                                          const real_t *const               steps,
                                          real_t *const SFEM_RESTRICT       out) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

#pragma omp parallel
    {
        scalar_t *out_local = (scalar_t *)calloc(nsteps, sizeof(scalar_t));

#pragma omp for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[8];

            scalar_t lx[8];
            scalar_t ly[8];
            scalar_t lz[8];

            scalar_t edispx[8];
            scalar_t edispy[8];
            scalar_t edispz[8];

            scalar_t eincx[8];
            scalar_t eincy[8];
            scalar_t eincz[8];

            for (int v = 0; v < 8; ++v) {
                ev[v] = elements[v][i * stride];
            }

            for (int d = 0; d < 8; d++) {
                lx[d] = x[ev[d]];
                ly[d] = y[ev[d]];
                lz[d] = z[ev[d]];
            }

            for (int v = 0; v < 8; ++v) {
                const ptrdiff_t idx = ev[v] * u_stride;
                edispx[v]           = ux[idx];
                edispy[v]           = uy[idx];
                edispz[v]           = uz[idx];
            }

            for (int v = 0; v < 8; ++v) {
                const ptrdiff_t idx = ev[v] * inc_stride;
                eincx[v]            = incx[idx];
                eincy[v]            = incy[idx];
                eincz[v]            = incz[idx];
            }

            hex8_neohookean_ogden_objective_steps_integral(
                    lx, ly, lz, n_qp, qx, qw, mu, lambda, edispx, edispy, edispz, eincx, eincy, eincz, nsteps, steps, out_local);
        }

        for (int s = 0; s < nsteps; s++) {
#pragma omp atomic update
            out[s] += out_local[s];
        }

        free(out_local);
    }

    for (int s = 0; s < nsteps; s++) {
<<<<<<< HEAD
        if(out[s] != out[s]) {
=======
        if (out[s] != out[s]) {
>>>>>>> origin/main
            out[s] = 1e10;
        }
    }

    return SFEM_SUCCESS;
}

int hex8_neohookean_ogden_gradient(const ptrdiff_t                   nelements,
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
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];

        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];

        scalar_t edispx[8];
        scalar_t edispy[8];
        scalar_t edispz[8];

        accumulator_t eoutx[8] = {0};
        accumulator_t eouty[8] = {0};
        accumulator_t eoutz[8] = {0};

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int d = 0; d < 8; d++) {
            lx[d] = x[ev[d]];
            ly[d] = y[ev[d]];
            lz[d] = z[ev[d]];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            edispx[v]           = ux[idx];
            edispy[v]           = uy[idx];
            edispz[v]           = uz[idx];
        }

        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], jacobian_adjugate, &jacobian_determinant);
                    assert(jacobian_determinant == jacobian_determinant);
                    assert(jacobian_determinant != 0);

<<<<<<< HEAD
                    hex8_neohookean_grad(jacobian_adjugate,
                                         jacobian_determinant,
                                         qx[kx],
                                         qx[ky],
                                         qx[kz],
                                         qw[kx] * qw[ky] * qw[kz],
                                         mu,
                                         lambda,
                                         edispx,
                                         edispy,
                                         edispz,
                                         eoutx,
                                         eouty,
                                         eoutz);
=======
                    hex8_neohookean_ogden_grad(jacobian_adjugate,
                                               jacobian_determinant,
                                               qx[kx],
                                               qx[ky],
                                               qx[kz],
                                               qw[kx] * qw[ky] * qw[kz],
                                               mu,
                                               lambda,
                                               edispx,
                                               edispy,
                                               edispz,
                                               eoutx,
                                               eouty,
                                               eoutz);
>>>>>>> origin/main
                }
            }
        }

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

            assert(eoutx[edof_i] == eoutx[edof_i]);
            assert(eouty[edof_i] == eouty[edof_i]);
            assert(eoutz[edof_i] == eoutz[edof_i]);

#pragma omp atomic update
            outx[idx] += eoutx[edof_i];

#pragma omp atomic update
            outy[idx] += eouty[edof_i];

#pragma omp atomic update
            outz[idx] += eoutz[edof_i];
        }
    }

    return SFEM_SUCCESS;
}

int hex8_neohookean_ogden_hessian_partial_assembly(const ptrdiff_t                      nelements,
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
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

<<<<<<< HEAD
=======
    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

>>>>>>> origin/main
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t    ev[8];
        scalar_t element_ux[8];
        scalar_t element_uy[8];
        scalar_t element_uz[8];
        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int v = 0; v < 8; ++v) {
            lx[v] = x[ev[v]];
            ly[v] = y[ev[v]];
            lz[v] = z[ev[v]];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v]       = ux[idx];
            element_uy[v]       = uy[idx];
            element_uz[v]       = uz[idx];
        }

<<<<<<< HEAD
=======
        scalar_t S_ikmn[HEX8_S_IKMN_SIZE] = {0};
#if 0

>>>>>>> origin/main
        static const scalar_t samplex = 0.5, sampley = 0.5, samplez = 0.5;
        hex8_adjugate_and_det(lx, ly, lz, samplex, sampley, samplez, jacobian_adjugate, &jacobian_determinant);

        // Sample at the centroid
        scalar_t F[9] = {0};
        hex8_F(jacobian_adjugate, jacobian_determinant, samplex, sampley, samplez, element_ux, element_uy, element_uz, F);
<<<<<<< HEAD
        scalar_t S_ikmn[HEX8_S_IKMN_SIZE] = {0};
        hex8_S_ikmn_neohookean(jacobian_adjugate, jacobian_determinant, samplex, sampley, samplez, F, mu, lambda, 1, S_ikmn);

=======
        
        hex8_S_ikmn_neohookean(jacobian_adjugate, jacobian_determinant, samplex, sampley, samplez,1,F, mu, lambda,  S_ikmn);

#else
        for(int qz = 0; qz < n_qp; qz++) {
            for(int ky = 0; ky < n_qp; ky++) {
                for(int kx = 0; kx < n_qp; kx++) {
                    hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[qz], jacobian_adjugate, &jacobian_determinant);
                    assert(jacobian_determinant == jacobian_determinant);
                    assert(jacobian_determinant != 0);

                    scalar_t F[9] = {0};
                    hex8_F(jacobian_adjugate, jacobian_determinant, qx[kx], qx[ky], qx[qz], element_ux, element_uy, element_uz, F);

                    hex8_S_ikmn_neohookean_add(
                        jacobian_adjugate, jacobian_determinant, 
                        qx[kx], qx[ky], qx[qz], qw[kx] * qw[ky] * qw[qz], F, mu, lambda, S_ikmn);
                }
            }
        }
#endif
>>>>>>> origin/main
        metric_tensor_t *const pai = &partial_assembly[i * HEX8_S_IKMN_SIZE];
        for (int k = 0; k < HEX8_S_IKMN_SIZE; k++) {
            assert(S_ikmn[k] == S_ikmn[k]);
            pai[k] = S_ikmn[k];
        }
    }

    return SFEM_SUCCESS;
}

int hex8_neohookean_ogden_partial_assembly_apply(const ptrdiff_t                            nelements,
                                                 const ptrdiff_t                            stride,
                                                 idx_t **const SFEM_RESTRICT                elements,
                                                 const metric_tensor_t *const SFEM_RESTRICT partial_assembly,
                                                 const ptrdiff_t                            h_stride,
                                                 const real_t *const                        hx,
                                                 const real_t *const                        hy,
                                                 const real_t *const                        hz,
                                                 const ptrdiff_t                            out_stride,
                                                 real_t *const                              outx,
                                                 real_t *const                              outy,
                                                 real_t *const                              outz) {
    scalar_t Wimpn_compressed[10];
    hex8_Wimpn_compressed(Wimpn_compressed);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];

        scalar_t element_hx[8];
        scalar_t element_hy[8];
        scalar_t element_hz[8];

        accumulator_t eoutx[8];
        accumulator_t eouty[8];
        accumulator_t eoutz[8];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * h_stride;
            element_hx[v]       = hx[idx];
            element_hy[v]       = hy[idx];
            element_hz[v]       = hz[idx];
        }

#if 0 
    // Slower than other variant
    scalar_t                     S_ikmn[3*3*3*3];
    hex8_expand_S(&partial_assembly[i * HEX8_S_IKMN_SIZE], S_ikmn);

    scalar_t                     Zpkmn[8*3*3*3];
    hex8_Zpkmn(Wimpn_compressed, element_hx, element_hy, element_hz, Zpkmn);
    hex8_SdotZ_expanded(S_ikmn, Zpkmn, eoutx, eouty, eoutz);
#else
        const metric_tensor_t *const pai = &partial_assembly[i * HEX8_S_IKMN_SIZE];
        scalar_t                     S_ikmn[HEX8_S_IKMN_SIZE];
        for (int k = 0; k < HEX8_S_IKMN_SIZE; k++) {
            S_ikmn[k] = pai[k];
            assert(S_ikmn[k] == S_ikmn[k]);
        }

        hex8_SdotHdotG(S_ikmn, Wimpn_compressed, element_hx, element_hy, element_hz, eoutx, eouty, eoutz);
#endif

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

            assert(eoutx[edof_i] == eoutx[edof_i]);
            assert(eouty[edof_i] == eouty[edof_i]);
            assert(eoutz[edof_i] == eoutz[edof_i]);

#pragma omp atomic update
            outx[idx] += eoutx[edof_i];

#pragma omp atomic update
            outy[idx] += eouty[edof_i];

#pragma omp atomic update
            outz[idx] += eoutz[edof_i];
        }
    }

    return SFEM_SUCCESS;
}

// Apply partially assembled operator
int hex8_neohookean_ogden_compressed_partial_assembly_apply(const ptrdiff_t                         nelements,
                                                            const ptrdiff_t                         stride,
                                                            idx_t **const SFEM_RESTRICT             elements,
                                                            const compressed_t *const SFEM_RESTRICT partial_assembly,
                                                            const scaling_t *const SFEM_RESTRICT    scaling,
                                                            const ptrdiff_t                         h_stride,
                                                            const real_t *const                     hx,
                                                            const real_t *const                     hy,
                                                            const real_t *const                     hz,
                                                            const ptrdiff_t                         out_stride,
                                                            real_t *const                           outx,
                                                            real_t *const                           outy,
                                                            real_t *const                           outz) {
    scalar_t Wimpn_compressed[10];
    hex8_Wimpn_compressed(Wimpn_compressed);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];

        scalar_t element_hx[8];
        scalar_t element_hy[8];
        scalar_t element_hz[8];

        accumulator_t eoutx[8];
        accumulator_t eouty[8];
        accumulator_t eoutz[8];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * h_stride;
            element_hx[v]       = hx[idx];
            element_hy[v]       = hy[idx];
            element_hz[v]       = hz[idx];
        }

        // Load and decompress low precision tensor
        const scalar_t            s   = scaling[i];
        const compressed_t *const pai = &partial_assembly[i * HEX8_S_IKMN_SIZE];
        scalar_t                  S_ikmn[HEX8_S_IKMN_SIZE];
        for (int k = 0; k < HEX8_S_IKMN_SIZE; k++) {
            S_ikmn[k] = s * (scalar_t)(pai[k]);
            assert(S_ikmn[k] == S_ikmn[k]);
        }

        hex8_SdotHdotG(S_ikmn, Wimpn_compressed, element_hx, element_hy, element_hz, eoutx, eouty, eoutz);

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

            assert(eoutx[edof_i] == eoutx[edof_i]);
            assert(eouty[edof_i] == eouty[edof_i]);
            assert(eoutz[edof_i] == eoutz[edof_i]);

#pragma omp atomic update
            outx[idx] += eoutx[edof_i];

#pragma omp atomic update
            outy[idx] += eouty[edof_i];

#pragma omp atomic update
            outz[idx] += eoutz[edof_i];
        }
    }

    return SFEM_SUCCESS;
}
<<<<<<< HEAD
=======

int hex8_neohookean_ogden_partial_assembly_diag(const ptrdiff_t                   nelements,
                                                const ptrdiff_t                   stride,
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
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t    ev[8];
        scalar_t element_ux[8];
        scalar_t element_uy[8];
        scalar_t element_uz[8];

        scalar_t eoutx[8] = {0};
        scalar_t eouty[8] = {0};
        scalar_t eoutz[8] = {0};

        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int v = 0; v < 8; ++v) {
            lx[v] = x[ev[v]];
            ly[v] = y[ev[v]];
            lz[v] = z[ev[v]];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v]       = ux[idx];
            element_uy[v]       = uy[idx];
            element_uz[v]       = uz[idx];
        }

        static const scalar_t samplex = 0.5, sampley = 0.5, samplez = 0.5;
        hex8_adjugate_and_det(lx, ly, lz, samplex, sampley, samplez, jacobian_adjugate, &jacobian_determinant);

        // Sample at the centroid
        scalar_t F[9] = {0};
        hex8_F(jacobian_adjugate, jacobian_determinant, samplex, sampley, samplez, element_ux, element_uy, element_uz, F);

        hex8_neohookean_ogden_hessian_diag(jacobian_adjugate,
                                           jacobian_determinant,
                                           samplex,
                                           sampley,
                                           samplez,
                                           1,
                                           mu,
                                           lambda,
                                           element_ux,
                                           element_uy,
                                           element_uz,
                                           eoutx,
                                           eouty,
                                           eoutz);

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

            assert(eoutx[edof_i] == eoutx[edof_i]);
            assert(eouty[edof_i] == eouty[edof_i]);
            assert(eoutz[edof_i] == eoutz[edof_i]);

#pragma omp atomic update
            outx[idx] += eoutx[edof_i];

#pragma omp atomic update
            outy[idx] += eouty[edof_i];

#pragma omp atomic update
            outz[idx] += eoutz[edof_i];
        }
    }

    return SFEM_SUCCESS;
}

int hex8_neohookean_ogden_elasticity_diag(const ptrdiff_t                   nelements,
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
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];

        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];

        scalar_t edispx[8];
        scalar_t edispy[8];
        scalar_t edispz[8];

        accumulator_t eoutx[8] = {0};
        accumulator_t eouty[8] = {0};
        accumulator_t eoutz[8] = {0};

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int d = 0; d < 8; d++) {
            lx[d] = x[ev[d]];
            ly[d] = y[ev[d]];
            lz[d] = z[ev[d]];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            edispx[v]           = ux[idx];
            edispy[v]           = uy[idx];
            edispz[v]           = uz[idx];
        }

        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], jacobian_adjugate, &jacobian_determinant);
                    assert(jacobian_determinant == jacobian_determinant);
                    assert(jacobian_determinant != 0);

                    hex8_neohookean_ogden_hessian_diag(jacobian_adjugate,
                                                       jacobian_determinant,
                                                       qx[kx],
                                                       qx[ky],
                                                       qx[kz],
                                                       qw[kx] * qw[ky] * qw[kz],
                                                       mu,
                                                       lambda,
                                                       edispx,
                                                       edispy,
                                                       edispz,
                                                       eoutx,
                                                       eouty,
                                                       eoutz);
                }
            }
        }

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

            assert(eoutx[edof_i] == eoutx[edof_i]);
            assert(eouty[edof_i] == eouty[edof_i]);
            assert(eoutz[edof_i] == eoutz[edof_i]);

#pragma omp atomic update
            outx[idx] += eoutx[edof_i];

#pragma omp atomic update
            outy[idx] += eouty[edof_i];

#pragma omp atomic update
            outz[idx] += eoutz[edof_i];
        }
    }

    return SFEM_SUCCESS;
}

int hex8_neohookean_ogden_bsr(const ptrdiff_t                    nelements,
                              const ptrdiff_t                    stride,
                              idx_t **const SFEM_RESTRICT        elements,
                              geom_t **const SFEM_RESTRICT       points,
                              const real_t                       mu,
                              const real_t                       lambda,
                              const ptrdiff_t                    u_stride,
                              const real_t *const SFEM_RESTRICT  ux,
                              const real_t *const SFEM_RESTRICT  uy,
                              const real_t *const SFEM_RESTRICT  uz,
                              const count_t *const SFEM_RESTRICT rowptr,
                              const idx_t *const SFEM_RESTRICT   colidx,
                              real_t *const SFEM_RESTRICT        values) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

#pragma omp parallel
    {
#pragma omp for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            scalar_t element_matrix[(3 * 8) * (3 * 8)] = {0};
            idx_t    ev[8];

            scalar_t lx[8];
            scalar_t ly[8];
            scalar_t lz[8];

            scalar_t edispx[8];
            scalar_t edispy[8];
            scalar_t edispz[8];

            for (int v = 0; v < 8; ++v) {
                ev[v] = elements[v][i * stride];
            }

            for (int d = 0; d < 8; d++) {
                lx[d] = x[ev[d]];
                ly[d] = y[ev[d]];
                lz[d] = z[ev[d]];
            }
            for (int v = 0; v < 8; ++v) {
                const ptrdiff_t idx = ev[v] * u_stride;
                edispx[v]           = ux[idx];
                edispy[v]           = uy[idx];
                edispz[v]           = uz[idx];
            }

            scalar_t jacobian_adjugate[9];
            scalar_t jacobian_determinant;

            for (int kz = 0; kz < n_qp; kz++) {
                for (int ky = 0; ky < n_qp; ky++) {
                    for (int kx = 0; kx < n_qp; kx++) {
                        hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], jacobian_adjugate, &jacobian_determinant);

                        hex8_neohookean_ogden_hessian(jacobian_adjugate,
                                                      jacobian_determinant,
                                                      qx[kx],
                                                      qx[ky],
                                                      qx[kz],
                                                      qw[kx] * qw[ky] * qw[kz],
                                                      mu,
                                                      lambda,
                                                      edispx,
                                                      edispy,
                                                      edispz,
                                                      element_matrix);
                    }
                }
            }

            hex8_local_to_global_bsr3(ev, element_matrix, rowptr, colidx, values);
        }
    }

    return SFEM_SUCCESS;
}

int hex8_neohookean_ogden_bcrs_sym(const ptrdiff_t                    nelements,
                                   const ptrdiff_t                    stride,
                                   idx_t **const SFEM_RESTRICT        elements,
                                   geom_t **const SFEM_RESTRICT       points,
                                   const real_t                       mu,
                                   const real_t                       lambda,
                                   const ptrdiff_t                    u_stride,
                                   const real_t *const SFEM_RESTRICT  ux,
                                   const real_t *const SFEM_RESTRICT  uy,
                                   const real_t *const SFEM_RESTRICT  uz,
                                   const count_t *const SFEM_RESTRICT rowptr,
                                   const idx_t *const SFEM_RESTRICT   colidx,
                                   const ptrdiff_t                    block_stride,
                                   real_t **const SFEM_RESTRICT       block_diag,
                                   real_t **const SFEM_RESTRICT       block_offdiag) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    int SFEM_HEX8_QUADRATURE_ORDER = 2;
    SFEM_READ_ENV(SFEM_HEX8_QUADRATURE_ORDER, atoi);

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

#pragma omp parallel
    {
#pragma omp for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[8];

            scalar_t lx[8];
            scalar_t ly[8];
            scalar_t lz[8];

            for (int v = 0; v < 8; ++v) {
                ev[v] = elements[v][i];
            }

            for (int v = 0; v < 8; v++) {
                lx[v] = x[ev[v]];
                ly[v] = y[ev[v]];
                lz[v] = z[ev[v]];
            }

            scalar_t jacobian_adjugate[9];
            scalar_t jacobian_determinant;
            hex8_adjugate_and_det(lx, ly, lz, 0.5, 0.5, 0.5, jacobian_adjugate, &jacobian_determinant);

            // Assemble the diagonal part of the matrix
            for (int edof_i = 0; edof_i < 8; edof_i++) {
                accumulator_t element_matrix[6] = {0, 0, 0, 0, 0, 0};
                for (int zi = 0; zi < n_qp; zi++) {
                    for (int yi = 0; yi < n_qp; yi++) {
                        for (int xi = 0; xi < n_qp; xi++) {
                            scalar_t test_grad[3];
                            hex8_ref_shape_grad(edof_i, qx[xi], qx[yi], qx[zi], test_grad);
                            // linear_elasticity_matrix_sym(mu,
                            //                              lambda,
                            //                              jacobian_adjugate,
                            //                              jacobian_determinant,
                            //                              test_grad,
                            //                              test_grad,
                            //                              qw[xi] * qw[yi] * qw[zi],
                            //                              element_matrix);
                            SFEM_IMPLEMENT_ME();
                            // TODO: Implement neohookean_ogden_matrix
                        }
                    }
                }
                // local to global
                int d_idx = 0;
                for (int d1 = 0; d1 < 3; d1++) {
                    for (int d2 = d1; d2 < 3; d2++, d_idx++) {
                        real_t *values = &block_diag[d_idx][ev[edof_i] * block_stride];
                        assert(element_matrix[d_idx] == element_matrix[d_idx]);
#pragma omp atomic update
                        *values += element_matrix[d_idx];
                    }
                }
            }

            // Assemble the upper-triangular part of the matrix
            for (int edof_i = 0; edof_i < 8; edof_i++) {
                // For each row we find the corresponding entries in the off-diag
                // We select the entries associated with ev[row] < ev[col]
                const int    lenrow = rowptr[ev[edof_i] + 1] - rowptr[ev[edof_i]];
                const idx_t *cols   = &colidx[rowptr[ev[edof_i]]];
                // Find the columns associated with the current row and mask what is not found with
                // -1
                int ks[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
                for (int i = 0; i < lenrow; i++) {
                    for (int k = 0; k < 8; k++) {
                        if (cols[i] == ev[k]) {
                            ks[k] = i;
                            break;
                        }
                    }
                }

                for (int edof_j = 0; edof_j < 8; edof_j++) {
                    if (ev[edof_j] > ev[edof_i]) {
                        assert(ks[edof_j] != -1);

                        accumulator_t element_matrix[6] = {0, 0, 0, 0, 0, 0};
                        for (int zi = 0; zi < n_qp; zi++) {
                            for (int yi = 0; yi < n_qp; yi++) {
                                for (int xi = 0; xi < n_qp; xi++) {
                                    scalar_t trial_grad[3];
                                    scalar_t test_grad[3];
                                    hex8_ref_shape_grad(edof_i, qx[xi], qx[yi], qx[zi], trial_grad);
                                    hex8_ref_shape_grad(edof_j, qx[xi], qx[yi], qx[zi], test_grad);
                                    // linear_elasticity_matrix_sym(mu,
                                    //                              lambda,
                                    //                              jacobian_adjugate,
                                    //                              jacobian_determinant,
                                    //                              trial_grad,
                                    //                              test_grad,
                                    //                              qw[xi] * qw[yi] * qw[zi],
                                    //                              element_matrix);
                                    SFEM_IMPLEMENT_ME();
                                    // TODO: Implement neohookean_ogden_matrix
                                }
                            }
                        }

                        // local to global
                        int d_idx = 0;
                        for (int d1 = 0; d1 < 3; d1++) {
                            for (int d2 = d1; d2 < 3; d2++, d_idx++) {
                                real_t *values = &block_offdiag[d_idx][(rowptr[ev[edof_i]] + ks[edof_j]) * block_stride];
#pragma omp atomic update
                                *values += element_matrix[d_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    return SFEM_SUCCESS;
}
>>>>>>> origin/main
