#include "hex8_mooney_rivlin_visco_flexible.hpp"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <mpi.h>


#include "sfem_macros.hpp"
#include "sfem_vec.hpp"
#include "sortreduce.hpp"

#include "hex8_inline_cpu.hpp"
#include "line_quadrature.hpp"

#include "hex8_mooney_rivlin_visco_unique_Hi_local.hpp"  // Unimodular form
// #include "hex8_mooney_rivlin_visco_unique_Hi_standard.hpp"   // Standard form (has non-zero initial stress)

template <class Run>
static int dispatch_history_read(const smesh::PrimitiveType storage, const void *const history, const Run &run) {
    if (storage == smesh::SMESH_FLOAT16) {
        return run(static_cast<const smesh::f16 *>(history));
    }

    if (storage == smesh::SMESH_FLOAT32) {
        return run(static_cast<const float *>(history));
    }

    if (storage == smesh::TypeToEnum<real_t>::value()) {
        return run(static_cast<const real_t *>(history));
    }

    SFEM_ERROR("Unsupported Mooney-Rivlin history storage: %s\n", smesh::to_string(storage).data());
    return SFEM_FAILURE;
}

template <class Run>
static int dispatch_history_update(const smesh::PrimitiveType storage,
                                   const void *const          history,
                                   void *const                new_history,
                                   const Run                 &run) {
    if (storage == smesh::SMESH_FLOAT16) {
        return run(static_cast<const smesh::f16 *>(history), static_cast<smesh::f16 *>(new_history));
    }

    if (storage == smesh::SMESH_FLOAT32) {
        return run(static_cast<const float *>(history), static_cast<float *>(new_history));
    }

    if (storage == smesh::TypeToEnum<real_t>::value()) {
        return run(static_cast<const real_t *>(history), static_cast<real_t *>(new_history));
    }

    SFEM_ERROR("Unsupported Mooney-Rivlin history storage: %s\n", smesh::to_string(storage).data());
    return SFEM_FAILURE;
}

// ============================================================================
// HISTORY UPDATE
// ============================================================================

int hex8_mooney_rivlin_visco_update_history_unique_hi(
    // 1. Mesh geometry
    const ptrdiff_t                   nelements,
    const ptrdiff_t                   stride,
    const ptrdiff_t                   nnodes,
    idx_t **const SFEM_RESTRICT       elements,
    geom_t **const SFEM_RESTRICT      points,
    // 2. Material parameters
    const real_t                      C10,
    const real_t                      C01,
    const real_t                      K,
    // 3. Viscoelastic parameters (precomputed)
    const int                         num_prony_terms,
    const real_t *const SFEM_RESTRICT alpha,
    const real_t *const SFEM_RESTRICT beta,
    // 4. History variables
    const ptrdiff_t                   history_stride,
    const int                         history_n_qp,
    const smesh::PrimitiveType        history_storage,
    const void *const SFEM_RESTRICT   history_data,
    void *const SFEM_RESTRICT         new_history_data,
    // 5. Displacement input
    const ptrdiff_t                   u_stride,
    const real_t *const SFEM_RESTRICT prev_ux,
    const real_t *const SFEM_RESTRICT prev_uy,
    const real_t *const SFEM_RESTRICT prev_uz,
    const real_t *const SFEM_RESTRICT ux,
    const real_t *const SFEM_RESTRICT uy,
    const real_t *const SFEM_RESTRICT uz) {
    auto run = [&](const auto *const history, auto *const new_history) -> int {
    SFEM_UNUSED(nnodes);
    
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

    const ptrdiff_t history_per_qp = num_prony_terms * 6;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];
        scalar_t lx[8], ly[8], lz[8];
        scalar_t prev_edispx[8], prev_edispy[8], prev_edispz[8];
        scalar_t edispx[8], edispy[8], edispz[8];
        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 8; ++v) ev[v] = elements[v][i * stride];
        for (int d = 0; d < 8; d++) { lx[d] = x[ev[d]]; ly[d] = y[ev[d]]; lz[d] = z[ev[d]]; }
        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            prev_edispx[v] = prev_ux[idx]; prev_edispy[v] = prev_uy[idx]; prev_edispz[v] = prev_uz[idx];
            edispx[v] = ux[idx]; edispy[v] = uy[idx]; edispz[v] = uz[idx];
        }

        if (history_n_qp == 1) {
            scalar_t S_dev_prev_avg[6] = {0};
            scalar_t S_dev_curr_avg[6] = {0};
            scalar_t weight_sum = 0;

            for (int kz = 0; kz < n_qp; kz++) {
                for (int ky = 0; ky < n_qp; ky++) {
                    for (int kx = 0; kx < n_qp; kx++) {
                        hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz],
                                              jacobian_adjugate, &jacobian_determinant);
                        assert(jacobian_determinant > 0);

                        scalar_t S_dev_prev[6];
                        hex8_mooney_rivlin_S_dev_from_disp(
                            jacobian_adjugate, jacobian_determinant,
                            qx[kx], qx[ky], qx[kz], qw[kx] * qw[ky] * qw[kz],
                            C10, C01, K,
                            prev_edispx, prev_edispy, prev_edispz,
                            S_dev_prev);

                        scalar_t S_dev_curr[6];
                        hex8_mooney_rivlin_S_dev_from_disp(
                            jacobian_adjugate, jacobian_determinant,
                            qx[kx], qx[ky], qx[kz], qw[kx] * qw[ky] * qw[kz],
                            C10, C01, K,
                            edispx, edispy, edispz,
                            S_dev_curr);

                        const scalar_t weight = qw[kx] * qw[ky] * qw[kz] * jacobian_determinant;
                        weight_sum += weight;
                        for (int c = 0; c < 6; ++c) {
                            S_dev_prev_avg[c] += weight * S_dev_prev[c];
                            S_dev_curr_avg[c] += weight * S_dev_curr[c];
                        }
                    }
                }
            }

            assert(weight_sum > 0);
            for (int c = 0; c < 6; ++c) {
                S_dev_prev_avg[c] /= weight_sum;
                S_dev_curr_avg[c] /= weight_sum;
            }

            const ptrdiff_t hist_offset = i * history_stride;
            for (int p = 0; p < num_prony_terms; ++p) {
                const auto *H_old = history + hist_offset + p * 6;
                auto *H_new = new_history + hist_offset + p * 6;
                for (int c = 0; c < 6; ++c) {
                    H_new[c] = alpha[p] * H_old[c] + beta[p] * (S_dev_curr_avg[c] - S_dev_prev_avg[c]);
                }
            }
        } else {
            for (int kz = 0; kz < n_qp; kz++) {
                for (int ky = 0; ky < n_qp; ky++) {
                    for (int kx = 0; kx < n_qp; kx++) {
                        hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz],
                                              jacobian_adjugate, &jacobian_determinant);
                        assert(jacobian_determinant > 0);

                        const ptrdiff_t qp_idx = (kz * n_qp * n_qp + ky * n_qp + kx);
                        const ptrdiff_t hist_offset = (i * history_stride) + (qp_idx * history_per_qp);

                        scalar_t S_dev_prev[6];
                        hex8_mooney_rivlin_S_dev_from_disp(
                            jacobian_adjugate, jacobian_determinant,
                            qx[kx], qx[ky], qx[kz], qw[kx] * qw[ky] * qw[kz],
                            C10, C01, K,
                            prev_edispx, prev_edispy, prev_edispz,
                            S_dev_prev);

                        scalar_t S_dev_curr[6];
                        hex8_mooney_rivlin_S_dev_from_disp(
                            jacobian_adjugate, jacobian_determinant,
                            qx[kx], qx[ky], qx[kz], qw[kx] * qw[ky] * qw[kz],
                            C10, C01, K,
                            edispx, edispy, edispz,
                            S_dev_curr);

                        for (int p = 0; p < num_prony_terms; ++p) {
                            const auto *H_old = history + hist_offset + p * 6;
                            auto *H_new = new_history + hist_offset + p * 6;

                            for (int c = 0; c < 6; ++c) {
                                H_new[c] = alpha[p] * H_old[c] + beta[p] * (S_dev_curr[c] - S_dev_prev[c]);
                            }
                        }
                    }
                }
            }
        }
    }

        return SFEM_SUCCESS;
    };

    return dispatch_history_update(history_storage, history_data, new_history_data, run);
}

// ============================================================================
// GRADIENT
// ============================================================================

int hex8_mooney_rivlin_visco_gradient_unique_hi(
    // 1. Mesh geometry
    const ptrdiff_t                   nelements,
    const ptrdiff_t                   stride,
    const ptrdiff_t                   nnodes,
    idx_t **const SFEM_RESTRICT       elements,
    geom_t **const SFEM_RESTRICT      points,
    // 2. Material parameters
    const real_t                      C10,
    const real_t                      C01,
    const real_t                      K,
    // 3. Viscoelastic parameters (precomputed)
    const int                         num_prony_terms,
    const real_t *const SFEM_RESTRICT alpha,
    const real_t *const SFEM_RESTRICT beta,
    const real_t                      gamma,
    // 4. History variables
    const ptrdiff_t                   history_stride,
    const int                         history_n_qp,
    const smesh::PrimitiveType        history_storage,
    const void *const SFEM_RESTRICT   history_data,
    // 5. Displacement input
    const ptrdiff_t                   u_stride,
    const real_t *const SFEM_RESTRICT prev_ux,
    const real_t *const SFEM_RESTRICT prev_uy,
    const real_t *const SFEM_RESTRICT prev_uz,
    const real_t *const SFEM_RESTRICT ux,
    const real_t *const SFEM_RESTRICT uy,
    const real_t *const SFEM_RESTRICT uz,
    // 6. Output
    const ptrdiff_t                   out_stride,
    real_t *const SFEM_RESTRICT       outx,
    real_t *const SFEM_RESTRICT       outy,
    real_t *const SFEM_RESTRICT       outz) {
    auto run = [&](const auto *const history) -> int {
    SFEM_UNUSED(nnodes);
    
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

    const ptrdiff_t history_per_qp = num_prony_terms * 6;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];
        scalar_t lx[8], ly[8], lz[8];
        scalar_t prev_edispx[8], prev_edispy[8], prev_edispz[8];
        scalar_t edispx[8], edispy[8], edispz[8];
        accumulator_t eoutx[8] = {0}, eouty[8] = {0}, eoutz[8] = {0};
        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 8; ++v) ev[v] = elements[v][i * stride];
        for (int d = 0; d < 8; d++) { lx[d] = x[ev[d]]; ly[d] = y[ev[d]]; lz[d] = z[ev[d]]; }
        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            prev_edispx[v] = prev_ux[idx]; prev_edispy[v] = prev_uy[idx]; prev_edispz[v] = prev_uz[idx];
            edispx[v] = ux[idx]; edispy[v] = uy[idx]; edispz[v] = uz[idx];
        }

        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], 
                                          jacobian_adjugate, &jacobian_determinant);
                    assert(jacobian_determinant > 0);

                    // Algorithmic gradient 
                    hex8_mooney_rivlin_grad_flexible(
                        jacobian_adjugate, jacobian_determinant,
                        qx[kx], qx[ky], qx[kz], qw[kx] * qw[ky] * qw[kz],
                        C10, C01, K, gamma,
                        edispx, edispy, edispz,
                        eoutx, eouty, eoutz);

                    // History contribution (OPTIMIZED: pre-accumulate S_hist, call once)
                    const ptrdiff_t qp_idx = (kz * n_qp * n_qp + ky * n_qp + kx);
                    const ptrdiff_t hist_offset = (i * history_stride) +
                                                 ((history_n_qp == 1) ? 0 : (qp_idx * history_per_qp));

                    scalar_t S_dev_prev[6];
                    hex8_mooney_rivlin_S_dev_from_disp(
                        jacobian_adjugate, jacobian_determinant,
                        qx[kx], qx[ky], qx[kz], qw[kx] * qw[ky] * qw[kz],
                        C10, C01, K,
                        prev_edispx, prev_edispy, prev_edispz,
                        S_dev_prev);

                    // Pre-accumulate: S_hist = sum(alpha_i * H_i - beta_i * S_dev_prev)
                    scalar_t S_hist[6] = {0};
                    for (int p = 0; p < num_prony_terms; ++p) {
                        const auto *H_i = history + hist_offset + p * 6;
                        for (int c = 0; c < 6; ++c) {
                            S_hist[c] += alpha[p] * H_i[c] - beta[p] * S_dev_prev[c];
                        }
                    }

                    // Single call with alpha=1, beta=0 trick
                    scalar_t dummy[6] = {0};
                    hex8_mooney_rivlin_grad_hist_single(
                        jacobian_adjugate, jacobian_determinant,
                        qx[kx], qx[ky], qx[kz], qw[kx] * qw[ky] * qw[kz],
                        1.0, 0.0,
                        S_hist, dummy,
                        edispx, edispy, edispz,
                        eoutx, eouty, eoutz);
                }
            }
        }

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;
#pragma omp atomic update
            outx[idx] += eoutx[edof_i];
#pragma omp atomic update
            outy[idx] += eouty[edof_i];
#pragma omp atomic update
            outz[idx] += eoutz[edof_i];
        }
    }

        return SFEM_SUCCESS;
    };

    return dispatch_history_read(history_storage, history_data, run);
}

// ============================================================================
// HESSIAN (BSR)
// ============================================================================

int hex8_mooney_rivlin_visco_bsr_unique_hi(
    // 1. Mesh geometry
    const ptrdiff_t                   nelements,
    const ptrdiff_t                   stride,
    const ptrdiff_t                   nnodes,
    idx_t **const SFEM_RESTRICT       elements,
    geom_t **const SFEM_RESTRICT      points,
    // 2. Material parameters
    const real_t                      C10,
    const real_t                      C01,
    const real_t                      K,
    // 3. Viscoelastic parameters (precomputed)
    const int                         num_prony_terms,
    const real_t *const SFEM_RESTRICT alpha,
    const real_t *const SFEM_RESTRICT beta,
    const real_t                      gamma,
    // 4. History variables
    const ptrdiff_t                   history_stride,
    const int                         history_n_qp,
    const smesh::PrimitiveType        history_storage,
    const void *const SFEM_RESTRICT   history_data,
    // 5. Displacement input
    const ptrdiff_t                   u_stride,
    const real_t *const SFEM_RESTRICT prev_ux,
    const real_t *const SFEM_RESTRICT prev_uy,
    const real_t *const SFEM_RESTRICT prev_uz,
    const real_t *const SFEM_RESTRICT ux,
    const real_t *const SFEM_RESTRICT uy,
    const real_t *const SFEM_RESTRICT uz,
    // 6. BSR output
    const ptrdiff_t                   out_stride,
    real_t *const SFEM_RESTRICT       values,
    const idx_t *const SFEM_RESTRICT  rowptr,
    const idx_t *const SFEM_RESTRICT  colidx) {
    auto run = [&](const auto *const history) -> int {
    SFEM_UNUSED(nnodes);
    SFEM_UNUSED(out_stride);
    
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

    const ptrdiff_t history_per_qp = num_prony_terms * 6;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];
        scalar_t lx[8], ly[8], lz[8];
        scalar_t prev_edispx[8], prev_edispy[8], prev_edispz[8];
        scalar_t edispx[8], edispy[8], edispz[8];
        scalar_t element_matrix[24 * 24] = {0};
        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 8; ++v) ev[v] = elements[v][i * stride];
        for (int d = 0; d < 8; d++) { lx[d] = x[ev[d]]; ly[d] = y[ev[d]]; lz[d] = z[ev[d]]; }
        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            prev_edispx[v] = prev_ux[idx]; prev_edispy[v] = prev_uy[idx]; prev_edispz[v] = prev_uz[idx];
            edispx[v] = ux[idx]; edispy[v] = uy[idx]; edispz[v] = uz[idx];
        }

        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], 
                                          jacobian_adjugate, &jacobian_determinant);
                    assert(jacobian_determinant > 0);

                    // Algorithmic Hessian
                    hex8_mooney_rivlin_hessian_algo_micro(
                        jacobian_adjugate, jacobian_determinant,
                        qx[kx], qx[ky], qx[kz], qw[kx] * qw[ky] * qw[kz],
                        C10, C01, K, gamma,
                        edispx, edispy, edispz,
                        element_matrix);

                    // History geometric stiffness (OPTIMIZED: pre-accumulate S_hist, call once)
                    const ptrdiff_t qp_idx = (kz * n_qp * n_qp + ky * n_qp + kx);
                    const ptrdiff_t hist_offset = (i * history_stride) +
                                                 ((history_n_qp == 1) ? 0 : (qp_idx * history_per_qp));

                    scalar_t S_dev_prev[6];
                    hex8_mooney_rivlin_S_dev_from_disp(
                        jacobian_adjugate, jacobian_determinant,
                        qx[kx], qx[ky], qx[kz], qw[kx] * qw[ky] * qw[kz],
                        C10, C01, K,
                        prev_edispx, prev_edispy, prev_edispz,
                        S_dev_prev);

                    // Pre-accumulate: S_hist = sum(alpha_i * H_i - beta_i * S_dev_prev)
                    scalar_t S_hist[6] = {0};
                    for (int p = 0; p < num_prony_terms; ++p) {
                        const auto *H_i = history + hist_offset + p * 6;
                        for (int c = 0; c < 6; ++c) {
                            S_hist[c] += alpha[p] * H_i[c] - beta[p] * S_dev_prev[c];
                        }
                    }

                    // Single call with alpha=1, beta=0 trick
                    scalar_t dummy[6] = {0};
                    hex8_mooney_rivlin_geom_stiff_single(
                        jacobian_adjugate, jacobian_determinant,
                        qx[kx], qx[ky], qx[kz], qw[kx] * qw[ky] * qw[kz],
                        1.0, 0.0,
                        S_hist, dummy,
                        element_matrix);
                }
            }
        }

        hex8_local_to_global_bsr3(ev, element_matrix, rowptr, colidx, values);
    }

        return SFEM_SUCCESS;
    };

    return dispatch_history_read(history_storage, history_data, run);
}

// ============================================================================
// HESSIAN DIAGONAL
// ============================================================================

int hex8_mooney_rivlin_visco_hessian_diag_unique_hi(
    // 1. Mesh geometry
    const ptrdiff_t                   nelements,
    const ptrdiff_t                   stride,
    const ptrdiff_t                   nnodes,
    idx_t **const SFEM_RESTRICT       elements,
    geom_t **const SFEM_RESTRICT      points,
    // 2. Material parameters
    const real_t                      C10,
    const real_t                      C01,
    const real_t                      K,
    // 3. Viscoelastic parameters (precomputed)
    const int                         num_prony_terms,
    const real_t *const SFEM_RESTRICT alpha,
    const real_t *const SFEM_RESTRICT beta,
    const real_t                      gamma,
    // 4. History variables
    const ptrdiff_t                   history_stride,
    const int                         history_n_qp,
    const smesh::PrimitiveType        history_storage,
    const void *const SFEM_RESTRICT   history_data,
    // 5. Displacement input
    const ptrdiff_t                   u_stride,
    const real_t *const SFEM_RESTRICT prev_ux,
    const real_t *const SFEM_RESTRICT prev_uy,
    const real_t *const SFEM_RESTRICT prev_uz,
    const real_t *const SFEM_RESTRICT ux,
    const real_t *const SFEM_RESTRICT uy,
    const real_t *const SFEM_RESTRICT uz,
    // 6. Output
    const ptrdiff_t                   out_stride,
    real_t *const SFEM_RESTRICT       outx,
    real_t *const SFEM_RESTRICT       outy,
    real_t *const SFEM_RESTRICT       outz) {
    auto run = [&](const auto *const history) -> int {
    SFEM_UNUSED(nnodes);
    
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    static const int       n_qp = line_q2_n;
    static const scalar_t *qx   = line_q2_x;
    static const scalar_t *qw   = line_q2_w;

    const ptrdiff_t history_per_qp = num_prony_terms * 6;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];
        scalar_t lx[8], ly[8], lz[8];
        scalar_t prev_edispx[8], prev_edispy[8], prev_edispz[8];
        scalar_t edispx[8], edispy[8], edispz[8];
        scalar_t element_matrix[24 * 24] = {0};
        scalar_t eoutx[8] = {0}, eouty[8] = {0}, eoutz[8] = {0};
        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 8; ++v) ev[v] = elements[v][i * stride];
        for (int d = 0; d < 8; d++) { lx[d] = x[ev[d]]; ly[d] = y[ev[d]]; lz[d] = z[ev[d]]; }
        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            prev_edispx[v] = prev_ux[idx]; prev_edispy[v] = prev_uy[idx]; prev_edispz[v] = prev_uz[idx];
            edispx[v] = ux[idx]; edispy[v] = uy[idx]; edispz[v] = uz[idx];
        }

        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    hex8_adjugate_and_det(lx, ly, lz, qx[kx], qx[ky], qx[kz], 
                                          jacobian_adjugate, &jacobian_determinant);
                    assert(jacobian_determinant > 0);

                    hex8_mooney_rivlin_hessian_algo_micro(
                        jacobian_adjugate, jacobian_determinant,
                        qx[kx], qx[ky], qx[kz], qw[kx] * qw[ky] * qw[kz],
                        C10, C01, K, gamma,
                        edispx, edispy, edispz,
                        element_matrix);

                    // History geometric stiffness (OPTIMIZED)
                    const ptrdiff_t qp_idx = (kz * n_qp * n_qp + ky * n_qp + kx);
                    const ptrdiff_t hist_offset = (i * history_stride) +
                                                 ((history_n_qp == 1) ? 0 : (qp_idx * history_per_qp));

                    scalar_t S_dev_prev[6];
                    hex8_mooney_rivlin_S_dev_from_disp(
                        jacobian_adjugate, jacobian_determinant,
                        qx[kx], qx[ky], qx[kz], qw[kx] * qw[ky] * qw[kz],
                        C10, C01, K,
                        prev_edispx, prev_edispy, prev_edispz,
                        S_dev_prev);

                    // Pre-accumulate: S_hist = sum(alpha_i * H_i - beta_i * S_dev_prev)
                    scalar_t S_hist[6] = {0};
                    for (int p = 0; p < num_prony_terms; ++p) {
                        const auto *H_i = history + hist_offset + p * 6;
                        for (int c = 0; c < 6; ++c) {
                            S_hist[c] += alpha[p] * H_i[c] - beta[p] * S_dev_prev[c];
                        }
                    }

                    // Single call with alpha=1, beta=0 trick
                    scalar_t dummy[6] = {0};
                    hex8_mooney_rivlin_geom_stiff_single(
                        jacobian_adjugate, jacobian_determinant,
                        qx[kx], qx[ky], qx[kz], qw[kx] * qw[ky] * qw[kz],
                        1.0, 0.0,
                        S_hist, dummy,
                        element_matrix);
                }
            }
        }

        // Extract diagonal
        for (int v = 0; v < 8; ++v) {
            eoutx[v] = element_matrix[(v * 3 + 0) * 24 + (v * 3 + 0)];
            eouty[v] = element_matrix[(v * 3 + 1) * 24 + (v * 3 + 1)];
            eoutz[v] = element_matrix[(v * 3 + 2) * 24 + (v * 3 + 2)];
        }

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;
#pragma omp atomic update
            outx[idx] += eoutx[edof_i];
#pragma omp atomic update
            outy[idx] += eouty[edof_i];
#pragma omp atomic update
            outz[idx] += eoutz[edof_i];
        }
    }

        return SFEM_SUCCESS;
    };

    return dispatch_history_read(history_storage, history_data, run);
}
