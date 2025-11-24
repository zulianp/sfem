// Mooney-Rivlin (active strain) HEX8 implementations
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include "crs_graph.h"
#include "hex8_inline_cpu.h"
#include "hex8_mooney_rivlin_active_strain.h"
#include "hex8_partial_assembly_mooney_rivlin_active_strain_inline.h"
#include "hex8_partial_assembly_neohookean_inline.h"
#include "hex8_mooney_rivlin_local.h"
#include "line_quadrature.h"
#include "sfem_macros.h"
#include "sfem_vec.h"
#include "sortreduce.h"

// Local inverse of 3x3 matrix and determinant (copied from smith implementation)
static SFEM_INLINE void inverse3(
        // Input
        const real_t mat_0,
        const real_t mat_1,
        const real_t mat_2,
        const real_t mat_3,
        const real_t mat_4,
        const real_t mat_5,
        const real_t mat_6,
        const real_t mat_7,
        const real_t mat_8,
        // Output
        scalar_t *const SFEM_RESTRICT mat_inv_0,
        scalar_t *const SFEM_RESTRICT mat_inv_1,
        scalar_t *const SFEM_RESTRICT mat_inv_2,
        scalar_t *const SFEM_RESTRICT mat_inv_3,
        scalar_t *const SFEM_RESTRICT mat_inv_4,
        scalar_t *const SFEM_RESTRICT mat_inv_5,
        scalar_t *const SFEM_RESTRICT mat_inv_6,
        scalar_t *const SFEM_RESTRICT mat_inv_7,
        scalar_t *const SFEM_RESTRICT mat_inv_8,
        scalar_t *const SFEM_RESTRICT Ja) {
    const real_t x0 = mat_4 * mat_8;
    const real_t x1 = mat_5 * mat_7;
    const real_t x2 = mat_1 * mat_5;
    const real_t x3 = mat_1 * mat_8;
    const real_t x4 = mat_2 * mat_4;
    Ja[0]           = (mat_0 * x0 - mat_0 * x1 + mat_2 * mat_3 * mat_7 - mat_3 * x3 + mat_6 * x2 - mat_6 * x4);
    const real_t x5 = 1.0 / Ja[0];
    *mat_inv_0      = x5 * (x0 - x1);
    *mat_inv_1      = x5 * (mat_2 * mat_7 - x3);
    *mat_inv_2      = x5 * (x2 - x4);
    *mat_inv_3      = x5 * (-mat_3 * mat_8 + mat_5 * mat_6);
    *mat_inv_4      = x5 * (mat_0 * mat_8 - mat_2 * mat_6);
    *mat_inv_5      = x5 * (-mat_0 * mat_5 + mat_2 * mat_3);
    *mat_inv_6      = x5 * (mat_3 * mat_7 - mat_4 * mat_6);
    *mat_inv_7      = x5 * (-mat_0 * mat_7 + mat_1 * mat_6);
    *mat_inv_8      = x5 * (mat_0 * mat_4 - mat_1 * mat_3);
}

int hex8_mooney_rivlin_active_strain_objective(const ptrdiff_t      nelements,
                                               const ptrdiff_t      stride,
                                               const ptrdiff_t      nnodes,
                                               idx_t **const        elements,
                                               geom_t **const       points,
                                               const real_t         mu,
                                               const real_t         lambda,
                                               const real_t         lmda,
                                               const ptrdiff_t      Fa_stride,
                                               const real_t **const Fa,
                                               const ptrdiff_t      u_stride,
                                               const real_t *const  ux,
                                               const real_t *const  uy,
                                               const real_t *const  uz,
                                               const int            is_element_wise,
                                               real_t *const        out) {
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

        scalar_t Fa_inv[9];
        scalar_t Ja;
        inverse3(Fa[0][i * Fa_stride],
                 Fa[1][i * Fa_stride],
                 Fa[2][i * Fa_stride],
                 Fa[3][i * Fa_stride],
                 Fa[4][i * Fa_stride],
                 Fa[5][i * Fa_stride],
                 Fa[6][i * Fa_stride],
                 Fa[7][i * Fa_stride],
                 Fa[8][i * Fa_stride],
                 &Fa_inv[0],
                 &Fa_inv[1],
                 &Fa_inv[2],
                 &Fa_inv[3],
                 &Fa_inv[4],
                 &Fa_inv[5],
                 &Fa_inv[6],
                 &Fa_inv[7],
                 &Fa_inv[8],
                 &Ja);

        scalar_t v = 0;
        // Integrate element objective over quadrature points
        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;
        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], jacobian_adjugate, &jacobian_determinant);
                    hex8_mooney_rivlin_active_strain_objective_at_qp(jacobian_adjugate,
                                                                     jacobian_determinant,
                                                                     qx[kx],
                                                                     qx[ky],
                                                                     qx[kz],
                                                                     qw[kx] * qw[ky] * qw[kz],
                                                                     lmda,
                                                                     lambda,
                                                                     mu,
                                                                     Fa_inv,
                                                                     Ja,
                                                                     edispx,
                                                                     edispy,
                                                                     edispz,
                                                                     &v);
                }
            }
        }
        assert(v == v);

        if (is_element_wise) {
            out[i] = v;
        } else {
#pragma omp atomic update
            *out += v;
        }
    }

    if (*out != *out) {
        *out = 1e10;
    }

    return SFEM_SUCCESS;
}

int hex8_mooney_rivlin_active_strain_objective_steps(const ptrdiff_t      nelements,
                                                     const ptrdiff_t      stride,
                                                     const ptrdiff_t      nnodes,
                                                     idx_t **const        elements,
                                                     geom_t **const       points,
                                                     const real_t         mu,
                                                     const real_t         lambda,
                                                     const real_t         lmda,
                                                     const ptrdiff_t      Fa_stride,
                                                     const real_t **const Fa,
                                                     const ptrdiff_t      u_stride,
                                                     const real_t *const  ux,
                                                     const real_t *const  uy,
                                                     const real_t *const  uz,
                                                     const ptrdiff_t      inc_stride,
                                                     const real_t *const  incx,
                                                     const real_t *const  incy,
                                                     const real_t *const  incz,
                                                     const int            nsteps,
                                                     const real_t *const  steps,
                                                     real_t *const        out) {
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

            scalar_t Fa_inv[9];
            scalar_t Ja;
            inverse3(Fa[0][i * Fa_stride],
                     Fa[1][i * Fa_stride],
                     Fa[2][i * Fa_stride],
                     Fa[3][i * Fa_stride],
                     Fa[4][i * Fa_stride],
                     Fa[5][i * Fa_stride],
                     Fa[6][i * Fa_stride],
                     Fa[7][i * Fa_stride],
                     Fa[8][i * Fa_stride],
                     &Fa_inv[0],
                     &Fa_inv[1],
                     &Fa_inv[2],
                     &Fa_inv[3],
                     &Fa_inv[4],
                     &Fa_inv[5],
                     &Fa_inv[6],
                     &Fa_inv[7],
                     &Fa_inv[8],
                     &Ja);

            scalar_t jacobian_adjugate[9];
            scalar_t jacobian_determinant = 0;

            for (int s = 0; s < nsteps; s++) {
                scalar_t ux_s[8];
                scalar_t uy_s[8];
                scalar_t uz_s[8];
                for (int j = 0; j < 8; j++) {
                    ux_s[j] = edispx[j] + eincx[j] * steps[s];
                    uy_s[j] = edispy[j] + eincy[j] * steps[s];
                    uz_s[j] = edispz[j] + eincz[j] * steps[s];
                }

                scalar_t v = 0;
                for (int kz = 0; kz < n_qp; kz++) {
                    for (int ky = 0; ky < n_qp; ky++) {
                        for (int kx = 0; kx < n_qp; kx++) {
                            hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], jacobian_adjugate, &jacobian_determinant);
                            hex8_mooney_rivlin_active_strain_objective_at_qp(jacobian_adjugate,
                                                                             jacobian_determinant,
                                                                             qx[kx],
                                                                             qx[ky],
                                                                             qx[kz],
                                                                             qw[kx] * qw[ky] * qw[kz],
                                                                             lmda,
                                                                             lambda,
                                                                             mu,
                                                                             Fa_inv,
                                                                             Ja,
                                                                             ux_s,
                                                                             uy_s,
                                                                             uz_s,
                                                                             &v);
                        }
                    }
                }
                out_local[s] += v;
            }
        }

        for (int s = 0; s < nsteps; s++) {
#pragma omp atomic update
            out[s] += out_local[s];
        }

        free(out_local);
    }

    for (int s = 0; s < nsteps; s++) {
        if (out[s] != out[s]) {
            out[s] = 1e10;
        }
    }

    return SFEM_SUCCESS;
}

int hex8_mooney_rivlin_active_strain_gradient(const ptrdiff_t      nelements,
                                              const ptrdiff_t      stride,
                                              const ptrdiff_t      nnodes,
                                              idx_t **const        elements,
                                              geom_t **const       points,
                                              const real_t         mu,
                                              const real_t         lambda,
                                              const real_t         lmda,
                                              const ptrdiff_t      Fa_stride,
                                              const real_t **const Fa,
                                              const ptrdiff_t      u_stride,
                                              const real_t *const  ux,
                                              const real_t *const  uy,
                                              const real_t *const  uz,
                                              const ptrdiff_t      out_stride,
                                              real_t *const        outx,
                                              real_t *const        outy,
                                              real_t *const        outz) {
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

        scalar_t Fa_inv[9];
        scalar_t Ja;
        inverse3(Fa[0][i * Fa_stride],
                 Fa[1][i * Fa_stride],
                 Fa[2][i * Fa_stride],
                 Fa[3][i * Fa_stride],
                 Fa[4][i * Fa_stride],
                 Fa[5][i * Fa_stride],
                 Fa[6][i * Fa_stride],
                 Fa[7][i * Fa_stride],
                 Fa[8][i * Fa_stride],
                 &Fa_inv[0],
                 &Fa_inv[1],
                 &Fa_inv[2],
                 &Fa_inv[3],
                 &Fa_inv[4],
                 &Fa_inv[5],
                 &Fa_inv[6],
                 &Fa_inv[7],
                 &Fa_inv[8],
                 &Ja);

        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], jacobian_adjugate, &jacobian_determinant);
                    assert(jacobian_determinant == jacobian_determinant);
                    assert(jacobian_determinant != 0);

                    hex8_mooney_rivlin_active_strain_grad(jacobian_adjugate,
                                                          jacobian_determinant,
                                                          qx[kx],
                                                          qx[ky],
                                                          qx[kz],
                                                          qw[kx] * qw[ky] * qw[kz],
                                                          lmda,
                                                          lambda,
                                                          mu,
                                                          Fa_inv,
                                                          Ja,
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

int hex8_mooney_rivlin_active_strain_hessian_partial_assembly(const ptrdiff_t        nelements,
                                                              const ptrdiff_t        stride,
                                                              idx_t **const          elements,
                                                              geom_t **const         points,
                                                              const real_t           mu,
                                                              const real_t           lambda,
                                                              const real_t           lmda,
                                                              const ptrdiff_t        Fa_stride,
                                                              const real_t **const   Fa,
                                                              const ptrdiff_t        u_stride,
                                                              const real_t *const    ux,
                                                              const real_t *const    uy,
                                                              const real_t *const    uz,
                                                              metric_tensor_t *const partial_assembly) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

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

        scalar_t Fa_inv[9];
        scalar_t Ja;
        inverse3(Fa[0][i * Fa_stride],
                 Fa[1][i * Fa_stride],
                 Fa[2][i * Fa_stride],
                 Fa[3][i * Fa_stride],
                 Fa[4][i * Fa_stride],
                 Fa[5][i * Fa_stride],
                 Fa[6][i * Fa_stride],
                 Fa[7][i * Fa_stride],
                 Fa[8][i * Fa_stride],
                 &Fa_inv[0],
                 &Fa_inv[1],
                 &Fa_inv[2],
                 &Fa_inv[3],
                 &Fa_inv[4],
                 &Fa_inv[5],
                 &Fa_inv[6],
                 &Fa_inv[7],
                 &Fa_inv[8],
                 &Ja);

        scalar_t S_ikmn[HEX8_S_IKMN_SIZE] = {0};
        for (int qz = 0; qz < n_qp; qz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[qz], jacobian_adjugate, &jacobian_determinant);
                    assert(jacobian_determinant == jacobian_determinant);
                    assert(jacobian_determinant != 0);

                    scalar_t F[9] = {0};
                    hex8_F(jacobian_adjugate,
                           jacobian_determinant,
                           qx[kx],
                           qx[ky],
                           qx[qz],
                           element_ux,
                           element_uy,
                           element_uz,
                           F);

                    hex8_S_ikmn_mooney_rivlin_active_strain(jacobian_adjugate,
                                                            jacobian_determinant,
                                                            qx[kx],
                                                            qx[ky],
                                                            qx[qz],
                                                            qw[kx] * qw[ky] * qw[qz],
                                                            F,
                                                            lmda,
                                                            lambda,
                                                            mu,
                                                            Fa_inv,
                                                            Ja,
                                                            S_ikmn);
                }
            }
        }
        metric_tensor_t *const pai = &partial_assembly[i * HEX8_S_IKMN_SIZE];
        for (int k = 0; k < HEX8_S_IKMN_SIZE; k++) {
            assert(S_ikmn[k] == S_ikmn[k]);
            pai[k] = S_ikmn[k];
        }
    }

    return SFEM_SUCCESS;
}

int hex8_mooney_rivlin_active_strain_partial_assembly_apply(const ptrdiff_t              nelements,
                                                            const ptrdiff_t              stride,
                                                            idx_t **const                elements,
                                                            const metric_tensor_t *const partial_assembly,
                                                            const ptrdiff_t              h_stride,
                                                            const real_t *const          hx,
                                                            const real_t *const          hy,
                                                            const real_t *const          hz,
                                                            const ptrdiff_t              out_stride,
                                                            real_t *const                outx,
                                                            real_t *const                outy,
                                                            real_t *const                outz) {
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

        const metric_tensor_t *const pai = &partial_assembly[i * HEX8_S_IKMN_SIZE];
        scalar_t                     S_ikmn[HEX8_S_IKMN_SIZE];
        for (int k = 0; k < HEX8_S_IKMN_SIZE; k++) {
            S_ikmn[k] = pai[k];
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

int hex8_mooney_rivlin_active_strain_partial_assembly_diag(const ptrdiff_t      nelements,
                                                           const ptrdiff_t      stride,
                                                           idx_t **const        elements,
                                                           geom_t **const       points,
                                                           const real_t         mu,
                                                           const real_t         lambda,
                                                           const real_t         lmda,
                                                           const ptrdiff_t      Fa_stride,
                                                           const real_t **const Fa,
                                                           const ptrdiff_t      u_stride,
                                                           const real_t *const  ux,
                                                           const real_t *const  uy,
                                                           const real_t *const  uz,
                                                           const ptrdiff_t      out_stride,
                                                           real_t *const        outx,
                                                           real_t *const        outy,
                                                           real_t *const        outz) {
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

        scalar_t Fa_inv[9];
        scalar_t Ja;
        inverse3(Fa[0][i * Fa_stride],
                 Fa[1][i * Fa_stride],
                 Fa[2][i * Fa_stride],
                 Fa[3][i * Fa_stride],
                 Fa[4][i * Fa_stride],
                 Fa[5][i * Fa_stride],
                 Fa[6][i * Fa_stride],
                 Fa[7][i * Fa_stride],
                 Fa[8][i * Fa_stride],
                 &Fa_inv[0],
                 &Fa_inv[1],
                 &Fa_inv[2],
                 &Fa_inv[3],
                 &Fa_inv[4],
                 &Fa_inv[5],
                 &Fa_inv[6],
                 &Fa_inv[7],
                 &Fa_inv[8],
                 &Ja);

        hex8_mooney_rivlin_active_strain_hessian_diag(jacobian_adjugate,
                                                      jacobian_determinant,
                                                      samplex,
                                                      sampley,
                                                      samplez,
                                                      1,
                                                      lmda,
                                                      lambda,
                                                      mu,
                                                      Fa_inv,
                                                      Ja,
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

int hex8_mooney_rivlin_active_strain_compressed_partial_assembly_apply(const ptrdiff_t           nelements,
                                                                       const ptrdiff_t           stride,
                                                                       idx_t **const             elements,
                                                                       const compressed_t *const partial_assembly,
                                                                       const scaling_t *const    scaling,
                                                                       const ptrdiff_t           h_stride,
                                                                       const real_t *const       hx,
                                                                       const real_t *const       hy,
                                                                       const real_t *const       hz,
                                                                       const ptrdiff_t           out_stride,
                                                                       real_t *const             outx,
                                                                       real_t *const             outy,
                                                                       real_t *const             outz) {
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

int hex8_mooney_rivlin_active_strain_elasticity_diag(const ptrdiff_t      nelements,
                                                     const ptrdiff_t      stride,
                                                     const ptrdiff_t      nnodes,
                                                     idx_t **const        elements,
                                                     geom_t **const       points,
                                                     const real_t         mu,
                                                     const real_t         lambda,
                                                     const real_t         lmda,
                                                     const ptrdiff_t      Fa_stride,
                                                     const real_t **const Fa,
                                                     const ptrdiff_t      u_stride,
                                                     const real_t *const  ux,
                                                     const real_t *const  uy,
                                                     const real_t *const  uz,
                                                     const ptrdiff_t      out_stride,
                                                     real_t *const        outx,
                                                     real_t *const        outy,
                                                     real_t *const        outz) {
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

        scalar_t Fa_inv[9];
        scalar_t Ja;
        inverse3(Fa[0][i * Fa_stride],
                 Fa[1][i * Fa_stride],
                 Fa[2][i * Fa_stride],
                 Fa[3][i * Fa_stride],
                 Fa[4][i * Fa_stride],
                 Fa[5][i * Fa_stride],
                 Fa[6][i * Fa_stride],
                 Fa[7][i * Fa_stride],
                 Fa[8][i * Fa_stride],
                 &Fa_inv[0],
                 &Fa_inv[1],
                 &Fa_inv[2],
                 &Fa_inv[3],
                 &Fa_inv[4],
                 &Fa_inv[5],
                 &Fa_inv[6],
                 &Fa_inv[7],
                 &Fa_inv[8],
                 &Ja);

        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], jacobian_adjugate, &jacobian_determinant);
                    assert(jacobian_determinant == jacobian_determinant);
                    assert(jacobian_determinant != 0);

                    hex8_mooney_rivlin_active_strain_hessian_diag(jacobian_adjugate,
                                                                  jacobian_determinant,
                                                                  qx[kx],
                                                                  qx[ky],
                                                                  qx[kz],
                                                                  qw[kx] * qw[ky] * qw[kz],
                                                                  lmda,
                                                                  lambda,
                                                                  mu,
                                                                  Fa_inv,
                                                                  Ja,
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

int hex8_mooney_rivlin_active_strain_bsr(const ptrdiff_t      nelements,
                                         const ptrdiff_t      stride,
                                         idx_t **const        elements,
                                         geom_t **const       points,
                                         const real_t         mu,
                                         const real_t         lambda,
                                         const real_t         lmda,
                                         const ptrdiff_t      Fa_stride,
                                         const real_t **const Fa,
                                         const ptrdiff_t      u_stride,
                                         const real_t *const  ux,
                                         const real_t *const  uy,
                                         const real_t *const  uz,
                                         const count_t *const rowptr,
                                         const idx_t *const   colidx,
                                         real_t *const        values) {
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

            scalar_t Fa_inv[9];
            scalar_t Ja;
            inverse3(Fa[0][i * Fa_stride],
                     Fa[1][i * Fa_stride],
                     Fa[2][i * Fa_stride],
                     Fa[3][i * Fa_stride],
                     Fa[4][i * Fa_stride],
                     Fa[5][i * Fa_stride],
                     Fa[6][i * Fa_stride],
                     Fa[7][i * Fa_stride],
                     Fa[8][i * Fa_stride],
                     &Fa_inv[0],
                     &Fa_inv[1],
                     &Fa_inv[2],
                     &Fa_inv[3],
                     &Fa_inv[4],
                     &Fa_inv[5],
                     &Fa_inv[6],
                     &Fa_inv[7],
                     &Fa_inv[8],
                     &Ja);

            for (int kz = 0; kz < n_qp; kz++) {
                for (int ky = 0; ky < n_qp; ky++) {
                    for (int kx = 0; kx < n_qp; kx++) {
                        hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], jacobian_adjugate, &jacobian_determinant);

                        hex8_mooney_rivlin_active_strain_hessian(jacobian_adjugate,
                                                                    jacobian_determinant,
                                                                    qx[kx],
                                                                    qx[ky],
                                                                    qx[kz],
                                                                    qw[kx] * qw[ky] * qw[kz],
                                                                    lmda,
                                                                    lambda,
                                                                    mu,
                                                                    Fa_inv,
                                                                    Ja,
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

int hex8_mooney_rivlin_active_strain_bcrs_sym(const ptrdiff_t      nelements,
                                              const ptrdiff_t      stride,
                                              idx_t **const        elements,
                                              geom_t **const       points,
                                              const real_t         mu,
                                              const real_t         lambda,
                                              const real_t         lmda,
                                              const ptrdiff_t      Fa_stride,
                                              const real_t **const Fa,
                                              const ptrdiff_t      u_stride,
                                              const real_t *const  ux,
                                              const real_t *const  uy,
                                              const real_t *const  uz,
                                              const count_t *const rowptr,
                                              const idx_t *const   colidx,
                                              const ptrdiff_t      block_stride,
                                              real_t **const       block_diag,
                                              real_t **const       block_offdiag) {
    SFEM_IMPLEMENT_ME();
    (void)nelements;
    (void)stride;
    (void)elements;
    (void)points;
    (void)mu;
    (void)lambda;
    (void)lmda;
    (void)Fa_stride;
    (void)Fa;
    (void)u_stride;
    (void)ux;
    (void)uy;
    (void)uz;
    (void)rowptr;
    (void)colidx;
    (void)block_stride;
    (void)block_diag;
    (void)block_offdiag;
    return SFEM_FAILURE;
}
