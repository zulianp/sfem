#include "hex8_kelvin_voigt_newmark.h"

#include "hex8_inline_cpu.h"
#include "hex8_kelvin_voigt_newmark_inline_cpu.h"
#include "hex8_laplacian_inline_cpu.h"
#include "line_quadrature.h"

#include <assert.h>
#include <stdio.h>

int affine_hex8_kelvin_voigt_newmark_lhs_apply(const ptrdiff_t             nelements,
                                               const ptrdiff_t             nnodes,
                                               idx_t **const SFEM_RESTRICT elements,

                                               const jacobian_t *const g_jacobian_adjugate,
                                               const jacobian_t *const g_jacobian_determinant,

                                               const real_t dt,
                                               const real_t gamma,
                                               const real_t beta,

                                               const real_t k,
                                               const real_t K,
                                               const real_t eta,
                                               const real_t rho,

                                               const ptrdiff_t     u_stride,
                                               const real_t *const ux,
                                               const real_t *const uy,
                                               const real_t *const uz,
                                               const ptrdiff_t     out_stride,
                                               real_t *const       outx,
                                               real_t *const       outy,
                                               real_t *const       outz) {
    SFEM_UNUSED(nnodes);

    int             n_qp = line_q2_n;
    const scalar_t *qx   = line_q2_x;
    const scalar_t *qw   = line_q2_w;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t    ev[8];
        scalar_t element_ux[8];
        scalar_t element_uy[8];
        scalar_t element_uz[8];

        accumulator_t element_outx[8];
        accumulator_t element_outy[8];
        accumulator_t element_outz[8];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = g_jacobian_determinant[i];

        for (int d = 0; d < 9; d++) {
            jacobian_adjugate[d] = g_jacobian_adjugate[i * 9 + d];
        }

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v]       = ux[idx];
            element_uy[v]       = uy[idx];
            element_uz[v]       = uz[idx];
        }

        for (int d = 0; d < 8; d++) {
            element_outx[d] = 0;
            element_outy[d] = 0;
            element_outz[d] = 0;
        }

        // hex8_adjugate_and_det(lx, ly, lz, 0.5, 0.5, 0.5, jacobian_adjugate, &jacobian_determinant);

        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    hex8_kelvin_voigt_newmark_lhs_apply_adj(k,
                                                            K,
                                                            eta,
                                                            dt,
                                                            gamma,
                                                            beta,
                                                            rho,
                                                            jacobian_adjugate,
                                                            jacobian_determinant,
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
            }
        }

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

#pragma omp atomic update
            outx[idx] += element_outx[edof_i];

#pragma omp atomic update
            outy[idx] += element_outy[edof_i];

#pragma omp atomic update
            outz[idx] += element_outz[edof_i];
        }
    }

    return SFEM_SUCCESS;
}

int affine_hex8_kelvin_voigt_newmark_gradient(const ptrdiff_t             nelements,
                                              const ptrdiff_t             nnodes,
                                              idx_t **const SFEM_RESTRICT elements,

                                              const jacobian_t *const g_jacobian_adjugate,
                                              const jacobian_t *const g_jacobian_determinant,

                                              const real_t k,
                                              const real_t K,
                                              const real_t eta,
                                              const real_t rho,

                                              const ptrdiff_t u_stride,

                                              const real_t *const ux,
                                              const real_t *const uy,
                                              const real_t *const uz,

                                              const real_t *const vx,
                                              const real_t *const vy,
                                              const real_t *const vz,

                                              const real_t *const ax,
                                              const real_t *const ay,
                                              const real_t *const az,

                                              const ptrdiff_t out_stride,
                                              real_t *const   outx,
                                              real_t *const   outy,
                                              real_t *const   outz) {
    SFEM_UNUSED(nnodes);

    int             n_qp = line_q2_n;
    const scalar_t *qx   = line_q2_x;
    const scalar_t *qw   = line_q2_w;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t    ev[8];
        scalar_t element_ux[8];
        scalar_t element_uy[8];
        scalar_t element_uz[8];

        scalar_t element_vx[8];
        scalar_t element_vy[8];
        scalar_t element_vz[8];

        scalar_t element_ax[8];
        scalar_t element_ay[8];
        scalar_t element_az[8];

        accumulator_t element_outx[8];
        accumulator_t element_outy[8];
        accumulator_t element_outz[8];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = g_jacobian_determinant[i];

        for (int d = 0; d < 9; d++) {
            jacobian_adjugate[d] = g_jacobian_adjugate[i * 9 + d];
        }

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v]       = ux[idx];
            element_uy[v]       = uy[idx];
            element_uz[v]       = uz[idx];
            element_vx[v]       = vx[idx];
            element_vy[v]       = vy[idx];
            element_vz[v]       = vz[idx];
            element_ax[v]       = ax[idx];
            element_ay[v]       = ay[idx];
            element_az[v]       = az[idx];
        }

        for (int d = 0; d < 8; d++) {
            element_outx[d] = 0;
            element_outy[d] = 0;
            element_outz[d] = 0;
        }

        // hex8_adjugate_and_det(lx, ly, lz, 0.5, 0.5, 0.5, jacobian_adjugate, &jacobian_determinant);

        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    hex8_kelvin_voigt_newmark_gradient_adj(k,
                                                           K,
                                                           eta,
                                                           rho,
                                                           jacobian_adjugate,
                                                           jacobian_determinant,
                                                           qx[kx],
                                                           qx[ky],
                                                           qx[kz],
                                                           qw[kx] * qw[ky] * qw[kz],
                                                           element_ux,
                                                           element_uy,
                                                           element_uz,
                                                           element_vx,
                                                           element_vy,
                                                           element_vz,
                                                           element_ax,
                                                           element_ay,
                                                           element_az,
                                                           element_outx,
                                                           element_outy,
                                                           element_outz);
                }
            }
        }
        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

#pragma omp atomic update
            outx[idx] += element_outx[edof_i];

#pragma omp atomic update
            outy[idx] += element_outy[edof_i];

#pragma omp atomic update
            outz[idx] += element_outz[edof_i];
        }
    }

    return SFEM_SUCCESS;
}

int affine_hex8_kelvin_voigt_newmark_diag(const ptrdiff_t              nelements,
                                          const ptrdiff_t              nnodes,
                                          idx_t **const SFEM_RESTRICT  elements,
                                          geom_t **const SFEM_RESTRICT points,
                                          const real_t                 beta,
                                          const real_t                 gamma,
                                          const real_t                 dt,
                                          const real_t                 k,
                                          const real_t                 K,
                                          const real_t                 eta,
                                          const real_t                 rho,
                                          const ptrdiff_t              out_stride,
                                          real_t *const                outx,
                                          real_t *const                outy,
                                          real_t *const                outz) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];

        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];

        accumulator_t element_diag[3 * 8];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        for (int d = 0; d < 8; d++) {
            lx[d] = x[ev[d]];
            ly[d] = y[ev[d]];
            lz[d] = z[ev[d]];
        }

        hex8_adjugate_and_det(lx, ly, lz, 0.5, 0.5, 0.5, jacobian_adjugate, &jacobian_determinant);

        sshex8_kelvin_voigt_newmark_diag(beta, gamma, dt, k, K, eta, rho, jacobian_adjugate, jacobian_determinant, element_diag);

        for (int edof_i = 0; edof_i < 8; edof_i++) {
            const ptrdiff_t idx = ev[edof_i] * out_stride;

#pragma omp atomic update
            outx[idx] += element_diag[0 * 8 + edof_i];

#pragma omp atomic update
            outy[idx] += element_diag[1 * 8 + edof_i];

#pragma omp atomic update
            outz[idx] += element_diag[2 * 8 + edof_i];
        }
    }

    return SFEM_SUCCESS;
}

int affine_hex8_kelvin_voigt_newmark_crs_sym(
        const ptrdiff_t                    nelements,
        const ptrdiff_t                    nnodes,
        idx_t **const SFEM_RESTRICT        elements,
        geom_t **const SFEM_RESTRICT       points,
        const real_t                       beta,
        const real_t                       gamma,
        const real_t                       dt,
        const real_t                       k,
        const real_t                       K,
        const real_t                       eta,
        const real_t                       rho,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT   colidx,
        const ptrdiff_t                    block_stride,  // stride of the block matrix to interchange SoA and AoS.
        real_t **const SFEM_RESTRICT       block_diag,
        real_t **const SFEM_RESTRICT       block_offdiag) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

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
                            scalar_t test_fun = hex8_ref_shape(edof_i, qx[xi], qx[yi], qx[zi]);

                            kelvin_voight_newmark_matrix_sym(beta,
                                                             gamma,
                                                             dt,
                                                             k,
                                                             K,
                                                             eta,
                                                             rho,
                                                             jacobian_adjugate,
                                                             jacobian_determinant,
                                                             test_fun,
                                                             test_grad,
                                                             test_fun,
                                                             test_grad,
                                                             qw[xi] * qw[yi] * qw[zi],
                                                             element_matrix);
                        }
                    }
                }

                // printf("(%d) -> (%d):\n", edof_i, ev[edof_i]);
                // print_matrix(1, 6, element_matrix);

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
                                    scalar_t trial_fun = hex8_ref_shape(edof_i, qx[xi], qx[yi], qx[zi]);
                                    scalar_t test_fun  = hex8_ref_shape(edof_j, qx[xi], qx[yi], qx[zi]);
                                    kelvin_voight_newmark_matrix_sym(beta,
                                                                     gamma,
                                                                     dt,
                                                                     k,
                                                                     K,
                                                                     eta,
                                                                     rho,
                                                                     jacobian_adjugate,
                                                                     jacobian_determinant,
                                                                     trial_fun,
                                                                     trial_grad,
                                                                     test_fun,
                                                                     test_grad,
                                                                     qw[xi] * qw[yi] * qw[zi],
                                                                     element_matrix);
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

int affine_hex8_kelvin_voigt_newmark_block_diag_sym(const ptrdiff_t              nelements,
                                                    const ptrdiff_t              nnodes,
                                                    idx_t **const SFEM_RESTRICT  elements,
                                                    geom_t **const SFEM_RESTRICT points,
                                                    const real_t                 beta,
                                                    const real_t                 gamma,
                                                    const real_t                 dt,
                                                    const real_t                 k,
                                                    const real_t                 K,
                                                    const real_t                 eta,
                                                    const real_t                 rho,
                                                    const ptrdiff_t              out_stride,
                                                    real_t *const                out0,
                                                    real_t *const                out1,
                                                    real_t *const                out2,
                                                    real_t *const                out3,
                                                    real_t *const                out4,
                                                    real_t *const                out5) {
    SFEM_UNUSED(nnodes);

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

#pragma omp parallel for
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
                        scalar_t test_fun = hex8_ref_shape(edof_i, qx[xi], qx[yi], qx[zi]);
                        kelvin_voight_newmark_matrix_sym(beta,
                                                         gamma,
                                                         dt,
                                                         k,
                                                         K,
                                                         eta,
                                                         rho,
                                                         jacobian_adjugate,
                                                         jacobian_determinant,
                                                         test_fun,
                                                         test_grad,
                                                         test_fun,
                                                         test_grad,
                                                         qw[xi] * qw[yi] * qw[zi],
                                                         element_matrix);
                    }
                }
            }

            const ptrdiff_t v = ev[edof_i];

            // local to global
#pragma omp atomic update
            out0[v * out_stride] += element_matrix[0];
#pragma omp atomic update
            out1[v * out_stride] += element_matrix[1];
#pragma omp atomic update
            out2[v * out_stride] += element_matrix[2];
#pragma omp atomic update
            out3[v * out_stride] += element_matrix[3];
#pragma omp atomic update
            out4[v * out_stride] += element_matrix[4];
#pragma omp atomic update
            out5[v * out_stride] += element_matrix[5];
        }
    }

    return SFEM_SUCCESS;
}
