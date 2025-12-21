#include "hex8_mooney_rivlin_visco.h"

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

#include "hex8_mooney_rivlin_visco_local.h"

int hex8_mooney_rivlin_visco_gradient(const ptrdiff_t                   nelements,
                                    const ptrdiff_t                   stride,
                                    const ptrdiff_t                   nnodes,
                                    idx_t **const SFEM_RESTRICT       elements,
                                    geom_t **const SFEM_RESTRICT      points,
                                      const real_t                      C10,
                                      const real_t                      C01,
                                      const real_t                      K,
                                      const real_t                      dt,
                                      const int                         num_prony_terms,
                                      const real_t *const SFEM_RESTRICT g,
                                      const real_t *const SFEM_RESTRICT tau,
                                      const ptrdiff_t                   history_stride,
                                      const real_t *const SFEM_RESTRICT history,
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

    // History size per quadrature point: 6 (S_dev_n) + num_prony_terms * 6 (H_i)
    const ptrdiff_t history_per_qp = 6 + num_prony_terms * 6;

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

                    // Calculate history offset for this quadrature point
                    const ptrdiff_t qp_idx = (kz * n_qp * n_qp + ky * n_qp + kx);
                    const ptrdiff_t hist_offset = (i * history_stride) + (qp_idx * history_per_qp);
                    const real_t *const qp_history = history + hist_offset;

                    hex8_mooney_rivlin_grad(jacobian_adjugate,
                                               jacobian_determinant,
                                               qx[kx],
                                               qx[ky],
                                               qx[kz],
                                               qw[kx] * qw[ky] * qw[kz],
                                            C10,
                                            C01,
                                            K,
                                            dt,
                                            num_prony_terms,
                                            g,
                                            tau,
                                            qp_history,
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

int hex8_mooney_rivlin_visco_update_history(const ptrdiff_t                   nelements,
                                            const ptrdiff_t                   stride,
                                            const ptrdiff_t                   nnodes,
                                            idx_t **const SFEM_RESTRICT       elements,
                                            geom_t **const SFEM_RESTRICT      points,
                                            const real_t                      C10,
                                            const real_t                      C01,
                                            const real_t                      K,
                                            const real_t                      dt,
                                            const int                         num_prony_terms,
                                            const real_t *const SFEM_RESTRICT g,
                                            const real_t *const SFEM_RESTRICT tau,
                                            const ptrdiff_t                   history_stride,
                                            const real_t *const SFEM_RESTRICT history,
                                            real_t *const SFEM_RESTRICT       new_history,
                                            const ptrdiff_t                   u_stride,
                                            const real_t *const SFEM_RESTRICT ux,
                                            const real_t *const SFEM_RESTRICT uy,
                                            const real_t *const SFEM_RESTRICT uz) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

    // History size per quadrature point: 6 (S_dev_n) + num_prony_terms * 6 (H_i)
    const ptrdiff_t history_per_qp = 6 + num_prony_terms * 6;

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

        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], jacobian_adjugate, &jacobian_determinant);
                    assert(jacobian_determinant == jacobian_determinant);
                    assert(jacobian_determinant != 0);

                    // Calculate history offset for this quadrature point
                    const ptrdiff_t qp_idx = (kz * n_qp * n_qp + ky * n_qp + kx);
                    const ptrdiff_t hist_offset = (i * history_stride) + (qp_idx * history_per_qp);
                    const real_t *const qp_history = history + hist_offset;
                    real_t *const qp_new_history = new_history + hist_offset;

                    hex8_mooney_rivlin_update_history(jacobian_adjugate,
                                                      jacobian_determinant,
                                                      qx[kx],
                                                      qx[ky],
                                                      qx[kz],
                                                      qw[kx] * qw[ky] * qw[kz],
                                                      C10,
                                                      C01,
                                                      K,
                                                      dt,
                                                      num_prony_terms,
                                                      g,
                                                      tau,
                                                      qp_history,
                                                      qp_new_history,
                                                      edispx,
                                                      edispy,
                                                      edispz);
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

int hex8_mooney_rivlin_visco_bsr(const ptrdiff_t                   nelements,
                                                const ptrdiff_t                   stride,
                                 const ptrdiff_t                   nnodes,
                                                idx_t **const SFEM_RESTRICT       elements,
                                                geom_t **const SFEM_RESTRICT      points,
                                 const real_t                      C10,
                                 const real_t                      C01,
                                 const real_t                      K,
                                 const real_t                      dt,
                                 const int                         num_prony_terms,
                                 const real_t *const SFEM_RESTRICT g,
                                 const real_t *const SFEM_RESTRICT tau,
                                 const ptrdiff_t                   history_stride,
                                 const real_t *const SFEM_RESTRICT history,
                                                const ptrdiff_t                   u_stride,
                                                const real_t *const SFEM_RESTRICT ux,
                                                const real_t *const SFEM_RESTRICT uy,
                                                const real_t *const SFEM_RESTRICT uz,
                                                const ptrdiff_t                   out_stride,
                                 real_t *const SFEM_RESTRICT       values,
                                 const idx_t *const SFEM_RESTRICT  rowptr,
                                 const idx_t *const SFEM_RESTRICT  colidx) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

    // History size per quadrature point
    const ptrdiff_t history_per_qp = 6 + num_prony_terms * 6;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];
        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];
        scalar_t edispx[8];
        scalar_t edispy[8];
        scalar_t edispz[8];

        scalar_t element_matrix[24 * 24] = {0}; // Initialize to 0

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

                    const ptrdiff_t qp_idx = (kz * n_qp * n_qp + ky * n_qp + kx);
                    const ptrdiff_t hist_offset = (i * history_stride) + (qp_idx * history_per_qp);
                    const real_t *const qp_history = history + hist_offset;

                    hex8_mooney_rivlin_hessian(jacobian_adjugate,
                                           jacobian_determinant,
                                               qx[kx],
                                               qx[ky],
                                               qx[kz],
                                               qw[kx] * qw[ky] * qw[kz],
                                               C10,
                                               C01,
                                               K,
                                               dt,
                                               num_prony_terms,
                                               g,
                                               tau,
                                               qp_history,
                                               edispx,
                                               edispy,
                                               edispz,
                                               element_matrix);
                }
            }
        }

        hex8_local_to_global_bsr3(ev, element_matrix, rowptr, colidx, values);
        }
    return SFEM_SUCCESS;
}

int hex8_mooney_rivlin_visco_hessian_diag(const ptrdiff_t                   nelements,
                                          const ptrdiff_t                   stride,
                                          const ptrdiff_t                   nnodes,
                                          idx_t **const SFEM_RESTRICT       elements,
                                          geom_t **const SFEM_RESTRICT      points,
                                          const real_t                      C10,
                                          const real_t                      C01,
                                          const real_t                      K,
                                          const real_t                      dt,
                                          const int                         num_prony_terms,
                                          const real_t *const SFEM_RESTRICT g,
                                          const real_t *const SFEM_RESTRICT tau,
                                          const ptrdiff_t                   history_stride,
                                          const real_t *const SFEM_RESTRICT history,
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

    const ptrdiff_t history_per_qp = 6 + num_prony_terms * 6;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];
        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];
        scalar_t edispx[8];
        scalar_t edispy[8];
        scalar_t edispz[8];

        scalar_t eoutx[8] = {0};
        scalar_t eouty[8] = {0};
        scalar_t eoutz[8] = {0};

        scalar_t element_diag[24] = {0};

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

                    const ptrdiff_t qp_idx = (kz * n_qp * n_qp + ky * n_qp + kx);
                    const ptrdiff_t hist_offset = (i * history_stride) + (qp_idx * history_per_qp);
                    const real_t *const qp_history = history + hist_offset;

                    hex8_mooney_rivlin_hessian_diag(jacobian_adjugate,
                                                       jacobian_determinant,
                                                       qx[kx],
                                                       qx[ky],
                                                       qx[kz],
                                                       qw[kx] * qw[ky] * qw[kz],
                                                    C10,
                                                    C01,
                                                    K,
                                                    dt,
                                                    num_prony_terms,
                                                    g,
                                                    tau,
                                                    qp_history,
                                                       edispx,
                                                       edispy,
                                                       edispz,
                                                    element_diag);
                }
            }
        }

        // Unpack diagonal to eoutx, eouty, eoutz
        for (int v = 0; v < 8; ++v) {
            eoutx[v] = element_diag[v * 3 + 0];
            eouty[v] = element_diag[v * 3 + 1];
            eoutz[v] = element_diag[v * 3 + 2];
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

