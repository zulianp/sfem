#include "tet10_neohookean_ogden.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sfem_macros.h"
#include "sfem_vec.h"
#include "sortreduce.h"

#include "line_quadrature.h"
#include "tet10_inline_cpu.h"

#include "tet10_neohookean_ogden_local.h"
#include "tet10_partial_assembly_neohookean_ogden_inline.h"

#define TET_QUAD_NQP 35
static real_t tet_qw[TET_QUAD_NQP] = {
        0.0021900463965388, 0.0021900463965388, 0.0021900463965388, 0.0021900463965388, 0.0143395670177665, 0.0143395670177665,
        0.0143395670177665, 0.0143395670177665, 0.0143395670177665, 0.0143395670177665, 0.0143395670177665, 0.0143395670177665,
        0.0143395670177665, 0.0143395670177665, 0.0143395670177665, 0.0143395670177665, 0.0250305395686746, 0.0250305395686746,
        0.0250305395686746, 0.0250305395686746, 0.0250305395686746, 0.0250305395686746, 0.0479839333057554, 0.0479839333057554,
        0.0479839333057554, 0.0479839333057554, 0.0479839333057554, 0.0479839333057554, 0.0479839333057554, 0.0479839333057554,
        0.0479839333057554, 0.0479839333057554, 0.0479839333057554, 0.0479839333057554, 0.0931745731195340};

static real_t tet_qx[TET_QUAD_NQP] = {
        0.0267367755543735, 0.9197896733368801, 0.0267367755543735, 0.0267367755543735, 0.7477598884818091, 0.1740356302468940,
        0.0391022406356488, 0.0391022406356488, 0.0391022406356488, 0.0391022406356488, 0.1740356302468940, 0.7477598884818091,
        0.1740356302468940, 0.7477598884818091, 0.0391022406356488, 0.0391022406356488, 0.4547545999844830, 0.0452454000155172,
        0.0452454000155172, 0.4547545999844830, 0.4547545999844830, 0.0452454000155172, 0.2232010379623150, 0.5031186450145980,
        0.2232010379623150, 0.2232010379623150, 0.5031186450145980, 0.2232010379623150, 0.0504792790607720, 0.0504792790607720,
        0.0504792790607720, 0.5031186450145980, 0.2232010379623150, 0.2232010379623150, 0.2500000000000000};

static real_t tet_qy[TET_QUAD_NQP] = {
        0.0267367755543735, 0.0267367755543735, 0.9197896733368801, 0.0267367755543735, 0.0391022406356488, 0.0391022406356488,
        0.7477598884818091, 0.1740356302468940, 0.0391022406356488, 0.0391022406356488, 0.7477598884818091, 0.1740356302468940,
        0.0391022406356488, 0.0391022406356488, 0.1740356302468940, 0.7477598884818091, 0.0452454000155172, 0.4547545999844830,
        0.0452454000155172, 0.4547545999844830, 0.0452454000155172, 0.4547545999844830, 0.2232010379623150, 0.2232010379623150,
        0.5031186450145980, 0.0504792790607720, 0.0504792790607720, 0.0504792790607720, 0.2232010379623150, 0.5031186450145980,
        0.2232010379623150, 0.2232010379623150, 0.5031186450145980, 0.2232010379623150, 0.2500000000000000};

static real_t tet_qz[TET_QUAD_NQP] = {
        0.0267367755543735, 0.0267367755543735, 0.0267367755543735, 0.9197896733368801, 0.0391022406356488, 0.0391022406356488,
        0.0391022406356488, 0.0391022406356488, 0.7477598884818091, 0.1740356302468940, 0.0391022406356488, 0.0391022406356488,
        0.7477598884818091, 0.1740356302468940, 0.7477598884818091, 0.1740356302468940, 0.0452454000155172, 0.0452454000155172,
        0.4547545999844830, 0.0452454000155172, 0.4547545999844830, 0.4547545999844830, 0.0504792790607720, 0.0504792790607720,
        0.0504792790607720, 0.2232010379623150, 0.2232010379623150, 0.5031186450145980, 0.2232010379623150, 0.2232010379623150,
        0.5031186450145980, 0.2232010379623150, 0.2232010379623150, 0.5031186450145980, 0.2500000000000000};

int tet10_neohookean_ogden_objective(const ptrdiff_t                   nelements,
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

    static const int       n_qp = TET_QUAD_NQP;
    static const scalar_t *qx   = tet_qx;
    static const scalar_t *qy   = tet_qy;
    static const scalar_t *qz   = tet_qz;
    static const scalar_t *qw   = tet_qw;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];

        scalar_t lx[10];
        scalar_t ly[10];
        scalar_t lz[10];

        scalar_t edispx[10];
        scalar_t edispy[10];
        scalar_t edispz[10];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int d = 0; d < 10; d++) {
            lx[d] = x[ev[d]];
            ly[d] = y[ev[d]];
            lz[d] = z[ev[d]];
        }

        for (int v = 0; v < 10; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            edispx[v]           = ux[idx];
            edispy[v]           = uy[idx];
            edispz[v]           = uz[idx];
        }

        scalar_t v = 0;
        tet10_neohookean_ogden_objective_integral(lx, ly, lz, n_qp, qx, qy, qz, qw, mu, lambda, edispx, edispy, edispz, &v);
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

int tet10_neohookean_ogden_objective_steps(const ptrdiff_t                   nelements,
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

    static const int       n_qp = TET_QUAD_NQP;
    static const scalar_t *qx   = tet_qx;
    static const scalar_t *qy   = tet_qy;
    static const scalar_t *qz   = tet_qz;
    static const scalar_t *qw   = tet_qw;

#pragma omp parallel
    {
        scalar_t *out_local = (scalar_t *)calloc(nsteps, sizeof(scalar_t));

#pragma omp for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[10];

            scalar_t lx[10];
            scalar_t ly[10];
            scalar_t lz[10];

            scalar_t edispx[10];
            scalar_t edispy[10];
            scalar_t edispz[10];

            scalar_t eincx[10];
            scalar_t eincy[10];
            scalar_t eincz[10];

            for (int v = 0; v < 10; ++v) {
                ev[v] = elements[v][i * stride];
            }

            for (int d = 0; d < 10; d++) {
                lx[d] = x[ev[d]];
                ly[d] = y[ev[d]];
                lz[d] = z[ev[d]];
            }

            for (int v = 0; v < 10; ++v) {
                const ptrdiff_t idx = ev[v] * u_stride;
                edispx[v]           = ux[idx];
                edispy[v]           = uy[idx];
                edispz[v]           = uz[idx];
            }

            for (int v = 0; v < 10; ++v) {
                const ptrdiff_t idx = ev[v] * inc_stride;
                eincx[v]            = incx[idx];
                eincy[v]            = incy[idx];
                eincz[v]            = incz[idx];
            }

            tet10_neohookean_ogden_objective_steps_integral(lx,
                                                            ly,
                                                            lz,
                                                            n_qp,
                                                            qx,
                                                            qy,
                                                            qz,
                                                            qw,
                                                            mu,
                                                            lambda,
                                                            edispx,
                                                            edispy,
                                                            edispz,
                                                            eincx,
                                                            eincy,
                                                            eincz,
                                                            nsteps,
                                                            steps,
                                                            out_local);
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

int tet10_neohookean_ogden_gradient(const ptrdiff_t                   nelements,
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

    static const int       n_qp = TET_QUAD_NQP;
    static const scalar_t *qx   = tet_qx;
    static const scalar_t *qy   = tet_qy;
    static const scalar_t *qz   = tet_qz;
    static const scalar_t *qw   = tet_qw;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];

        scalar_t lx[10];
        scalar_t ly[10];
        scalar_t lz[10];

        scalar_t edispx[10];
        scalar_t edispy[10];
        scalar_t edispz[10];

        accumulator_t eoutx[10] = {0};
        accumulator_t eouty[10] = {0};
        accumulator_t eoutz[10] = {0};

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int d = 0; d < 10; d++) {
            lx[d] = x[ev[d]];
            ly[d] = y[ev[d]];
            lz[d] = z[ev[d]];
        }

        for (int v = 0; v < 10; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            edispx[v]           = ux[idx];
            edispy[v]           = uy[idx];
            edispz[v]           = uz[idx];
        }

        for (int k = 0; k < n_qp; k++) {
            tet10_adjugate_and_det(lx, ly, lz, qx[k], qy[k], qz[k], jacobian_adjugate, &jacobian_determinant);
            assert(jacobian_determinant == jacobian_determinant);
            assert(jacobian_determinant != 0);

            tet10_neohookean_ogden_grad(jacobian_adjugate,
                                        jacobian_determinant,
                                        qx[k],
                                        qy[k],
                                        qz[k],
                                        qw[k],
                                        mu,
                                        lambda,
                                        edispx,
                                        edispy,
                                        edispz,
                                        eoutx,
                                        eouty,
                                        eoutz);
        }

        for (int edof_i = 0; edof_i < 10; edof_i++) {
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

int tet10_neohookean_ogden_hessian_partial_assembly(const ptrdiff_t                      nelements,
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

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t    ev[10];
        scalar_t element_ux[10];
        scalar_t element_uy[10];
        scalar_t element_uz[10];
        scalar_t lx[10];
        scalar_t ly[10];
        scalar_t lz[10];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int v = 0; v < 10; ++v) {
            lx[v] = x[ev[v]];
            ly[v] = y[ev[v]];
            lz[v] = z[ev[v]];
        }

        for (int v = 0; v < 10; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v]       = ux[idx];
            element_uy[v]       = uy[idx];
            element_uz[v]       = uz[idx];
        }

        static const scalar_t samplex = 1. / 4, sampley = 1. / 4, samplez = 1. / 4;
        tet10_adjugate_and_det(lx, ly, lz, samplex, sampley, samplez, jacobian_adjugate, &jacobian_determinant);

        // Sample at the centroid
        scalar_t F[9] = {0};
        tet10_F(jacobian_adjugate, jacobian_determinant, samplex, sampley, samplez, element_ux, element_uy, element_uz, F);
        scalar_t S_ikmn[TET10_S_IKMN_SIZE] = {0};
        tet10_S_ikmn_neohookean_ogden(
                jacobian_adjugate, jacobian_determinant, samplex, sampley, samplez, F, mu, lambda, 1, S_ikmn);

        metric_tensor_t *const pai = &partial_assembly[i * TET10_S_IKMN_SIZE];
        for (int k = 0; k < TET10_S_IKMN_SIZE; k++) {
            assert(S_ikmn[k] == S_ikmn[k]);
            pai[k] = S_ikmn[k];
        }
    }

    return SFEM_SUCCESS;
}

int tet10_neohookean_ogden_partial_assembly_apply(const ptrdiff_t                            nelements,
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
    tet10_Wimpn_compressed(Wimpn_compressed);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];

        scalar_t element_hx[10];
        scalar_t element_hy[10];
        scalar_t element_hz[10];

        accumulator_t eoutx[10];
        accumulator_t eouty[10];
        accumulator_t eoutz[10];

        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int v = 0; v < 10; ++v) {
            const ptrdiff_t idx = ev[v] * h_stride;
            element_hx[v]       = hx[idx];
            element_hy[v]       = hy[idx];
            element_hz[v]       = hz[idx];
        }

#if 0 
    // Slower than other variant
    scalar_t                     S_ikmn[3*3*3*3];
    tet10_expand_S(&partial_assembly[i * TET10_S_IKMN_SIZE], S_ikmn);

    scalar_t                     Zpkmn[10*3*3*3];
    tet10_Zpkmn(Wimpn_compressed, element_hx, element_hy, element_hz, Zpkmn);
    tet10_SdotZ_expanded(S_ikmn, Zpkmn, eoutx, eouty, eoutz);
#else
        const metric_tensor_t *const pai = &partial_assembly[i * TET10_S_IKMN_SIZE];
        scalar_t                     S_ikmn[TET10_S_IKMN_SIZE];
        for (int k = 0; k < TET10_S_IKMN_SIZE; k++) {
            S_ikmn[k] = pai[k];
            assert(S_ikmn[k] == S_ikmn[k]);
        }

        tet10_SdotHdotG(S_ikmn, Wimpn_compressed, element_hx, element_hy, element_hz, eoutx, eouty, eoutz);
#endif

        for (int edof_i = 0; edof_i < 10; edof_i++) {
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
int tet10_neohookean_ogden_compressed_partial_assembly_apply(const ptrdiff_t                         nelements,
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
    tet10_Wimpn_compressed(Wimpn_compressed);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];

        scalar_t element_hx[10];
        scalar_t element_hy[10];
        scalar_t element_hz[10];

        accumulator_t eoutx[10];
        accumulator_t eouty[10];
        accumulator_t eoutz[10];

        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int v = 0; v < 10; ++v) {
            const ptrdiff_t idx = ev[v] * h_stride;
            element_hx[v]       = hx[idx];
            element_hy[v]       = hy[idx];
            element_hz[v]       = hz[idx];
        }

        // Load and decompress low precision tensor
        const scalar_t            s   = scaling[i];
        const compressed_t *const pai = &partial_assembly[i * TET10_S_IKMN_SIZE];
        scalar_t                  S_ikmn[TET10_S_IKMN_SIZE];
        for (int k = 0; k < TET10_S_IKMN_SIZE; k++) {
            S_ikmn[k] = s * (scalar_t)(pai[k]);
            assert(S_ikmn[k] == S_ikmn[k]);
        }

        tet10_SdotHdotG(S_ikmn, Wimpn_compressed, element_hx, element_hy, element_hz, eoutx, eouty, eoutz);

        for (int edof_i = 0; edof_i < 10; edof_i++) {
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

int tet10_neohookean_ogden_bsr(const ptrdiff_t                    nelements,
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

    static const int       n_qp = TET_QUAD_NQP;
    static const scalar_t *qx   = tet_qx;
    static const scalar_t *qy   = tet_qy;
    static const scalar_t *qz   = tet_qz;
    static const scalar_t *qw   = tet_qw;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t    ev[10];
        scalar_t element_ux[10];
        scalar_t element_uy[10];
        scalar_t element_uz[10];
        scalar_t lx[10];
        scalar_t ly[10];
        scalar_t lz[10];
        scalar_t element_matrix[10 * 10 * 3 * 3] = {0};

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i * stride];
        }

        for (int v = 0; v < 10; ++v) {
            lx[v] = x[ev[v]];
            ly[v] = y[ev[v]];
            lz[v] = z[ev[v]];
        }

        for (int v = 0; v < 10; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v]       = ux[idx];
            element_uy[v]       = uy[idx];
            element_uz[v]       = uz[idx];
        }

        for (int k = 0; k < n_qp; k++) {
            tet10_adjugate_and_det(lx, ly, lz, qx[k], qy[k], qz[k], jacobian_adjugate, &jacobian_determinant);
            assert(jacobian_determinant == jacobian_determinant);
            assert(jacobian_determinant != 0);

            tet10_neohookean_ogden_hessian(jacobian_adjugate,
                                           jacobian_determinant,
                                           qx[k],
                                           qy[k],
                                           qz[k],
                                           qw[k],
                                           mu,
                                           lambda,
                                           element_ux,
                                           element_uy,
                                           element_uz,
                                           element_matrix);
        }

        tet10_local_to_global_bsr3(ev, element_matrix, rowptr, colidx, values);
    }

    return SFEM_SUCCESS;
}