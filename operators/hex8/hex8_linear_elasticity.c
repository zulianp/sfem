#include "hex8_linear_elasticity.h"

#include "hex8_inline_cpu.h"
#include "hex8_linear_elasticity_inline_cpu.h"
// #include "hex8_quadrature.h"
#include "hex8_laplacian_inline_cpu.h"
#include "line_quadrature.h"

#include <stdio.h>

static void print_matrix(int r, int c, const accumulator_t *const m) {
    printf("-------------------\n");
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%g\t", m[i * c + j]);
        }
        printf("\n");
    }
    printf("-------------------\n");
}

int hex8_linear_elasticity_apply(const ptrdiff_t nelements,
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
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    int SFEM_HEX8_QUADRATURE_ORDER = 2;
    SFEM_READ_ENV(SFEM_HEX8_QUADRATURE_ORDER, atoi);
    // printf("SFEM_HEX8_QUADRATURE_ORDER = %d\n", SFEM_HEX8_QUADRATURE_ORDER);

    int n_qp = line_q3_n;
    const scalar_t *qx = line_q3_x;
    const scalar_t *qw = line_q3_w;
    if (SFEM_HEX8_QUADRATURE_ORDER == 1) {
        n_qp = line_q2_n;
        qx = line_q2_x;
        qw = line_q2_w;
    } else if (SFEM_HEX8_QUADRATURE_ORDER == 5) {
        n_qp = line_q6_n;
        qx = line_q6_x;
        qw = line_q6_w;
    }

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];
        scalar_t element_ux[8];
        scalar_t element_uy[8];
        scalar_t element_uz[8];

        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];

        accumulator_t element_outx[8];
        accumulator_t element_outy[8];
        accumulator_t element_outz[8];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v] = ux[idx];
            element_uy[v] = uy[idx];
            element_uz[v] = uz[idx];
        }

        for (int d = 0; d < 8; d++) {
            lx[d] = x[ev[d]];
            ly[d] = y[ev[d]];
            lz[d] = z[ev[d]];
        }

        for (int d = 0; d < 8; d++) {
            element_outx[d] = 0;
            element_outy[d] = 0;
            element_outz[d] = 0;
        }

        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    hex8_adjugate_and_det(lx,
                                          ly,
                                          lz,
                                          qx[kx],
                                          qx[ky],
                                          qx[kz],
                                          jacobian_adjugate,
                                          &jacobian_determinant);

                    hex8_linear_elasticity_apply_adj(mu,
                                                     lambda,
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

int affine_hex8_linear_elasticity_apply(const ptrdiff_t nelements,
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
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    int SFEM_HEX8_QUADRATURE_ORDER = 2;
    SFEM_READ_ENV(SFEM_HEX8_QUADRATURE_ORDER, atoi);
    // printf("SFEM_HEX8_QUADRATURE_ORDER = %d\n", SFEM_HEX8_QUADRATURE_ORDER);

    int n_qp = line_q3_n;
    const scalar_t *qx = line_q3_x;
    const scalar_t *qw = line_q3_w;
    if (SFEM_HEX8_QUADRATURE_ORDER == 1) {
        n_qp = line_q2_n;
        qx = line_q2_x;
        qw = line_q2_w;
    } else if (SFEM_HEX8_QUADRATURE_ORDER == 5) {
        n_qp = line_q6_n;
        qx = line_q6_x;
        qw = line_q6_w;
    }

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[8];
        scalar_t element_ux[8];
        scalar_t element_uy[8];
        scalar_t element_uz[8];

        scalar_t lx[8];
        scalar_t ly[8];
        scalar_t lz[8];

        accumulator_t element_outx[8];
        accumulator_t element_outy[8];
        accumulator_t element_outz[8];

        scalar_t jacobian_adjugate[9];
        scalar_t jacobian_determinant = 0;

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 8; ++v) {
            const ptrdiff_t idx = ev[v] * u_stride;
            element_ux[v] = ux[idx];
            element_uy[v] = uy[idx];
            element_uz[v] = uz[idx];
        }

        for (int d = 0; d < 8; d++) {
            lx[d] = x[ev[d]];
            ly[d] = y[ev[d]];
            lz[d] = z[ev[d]];
        }

        for (int d = 0; d < 8; d++) {
            element_outx[d] = 0;
            element_outy[d] = 0;
            element_outz[d] = 0;
        }

        hex8_adjugate_and_det(lx, ly, lz, 0.5, 0.5, 0.5, jacobian_adjugate, &jacobian_determinant);

        for (int kz = 0; kz < n_qp; kz++) {
            for (int ky = 0; ky < n_qp; ky++) {
                for (int kx = 0; kx < n_qp; kx++) {
                    hex8_linear_elasticity_apply_adj(mu,
                                                     lambda,
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

int affine_hex8_linear_elasticity_bsr(const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const SFEM_RESTRICT elements,
                                      geom_t **const SFEM_RESTRICT points,
                                      const real_t mu,
                                      const real_t lambda,
                                      const count_t *const SFEM_RESTRICT rowptr,
                                      const idx_t *const SFEM_RESTRICT colidx,
                                      real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    // int SFEM_HEX8_QUADRATURE_ORDER = 2;
    // SFEM_READ_ENV(SFEM_HEX8_QUADRATURE_ORDER, atoi);
    // // printf("SFEM_HEX8_QUADRATURE_ORDER = %d\n", SFEM_HEX8_QUADRATURE_ORDER);

    // int n_qp = line_q3_n;
    // const scalar_t *qx = line_q3_x;
    // const scalar_t *qw = line_q3_w;
    // if (SFEM_HEX8_QUADRATURE_ORDER == 1) {
    //     n_qp = line_q2_n;
    //     qx = line_q2_x;
    //     qw = line_q2_w;
    // } else if (SFEM_HEX8_QUADRATURE_ORDER == 5) {
    //     n_qp = line_q6_n;
    //     qx = line_q6_x;
    //     qw = line_q6_w;
    // }

#pragma omp parallel
    {
        scalar_t element_matrix[(3 * 8) * (3 * 8)];
#pragma omp for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[8];

            scalar_t lx[8];
            scalar_t ly[8];
            scalar_t lz[8];

            for (int v = 0; v < 8; ++v) {
                ev[v] = elements[v][i];
            }

            for (int d = 0; d < 8; d++) {
                lx[d] = x[ev[d]];
                ly[d] = y[ev[d]];
                lz[d] = z[ev[d]];
            }

            scalar_t jacobian_adjugate[9];
            scalar_t jacobian_determinant;
            hex8_adjugate_and_det(
                    lx, ly, lz, 0.5, 0.5, 0.5, jacobian_adjugate, &jacobian_determinant);

            hex8_linear_elasticity_matrix(
                    mu, lambda, jacobian_adjugate, jacobian_determinant, element_matrix);

            hex8_local_to_global_bsr3(ev, element_matrix, rowptr, colidx, values);
        }
    }

    return SFEM_SUCCESS;
}

// The input CRS is created only for the upper-triangular part of the matrix
// And the diagonal is stored with for each node with a stride of 6
int affine_hex8_linear_elasticity_crs_sym(
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const SFEM_RESTRICT elements,
        geom_t **const SFEM_RESTRICT points,
        const real_t mu,
        const real_t lambda,
        const count_t *const SFEM_RESTRICT rowptr,
        const idx_t *const SFEM_RESTRICT colidx,
        const ptrdiff_t block_stride,  // stride of the block matrix to interchange SoA and AoS.
        real_t **const SFEM_RESTRICT block_diag,
        real_t **const SFEM_RESTRICT block_offdiag) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    int SFEM_HEX8_QUADRATURE_ORDER = 2;
    SFEM_READ_ENV(SFEM_HEX8_QUADRATURE_ORDER, atoi);
    // printf("SFEM_HEX8_QUADRATURE_ORDER = %d\n", SFEM_HEX8_QUADRATURE_ORDER);

    int n_qp = line_q3_n;
    const scalar_t *qx = line_q3_x;
    const scalar_t *qw = line_q3_w;
    if (SFEM_HEX8_QUADRATURE_ORDER == 1) {
        n_qp = line_q2_n;
        qx = line_q2_x;
        qw = line_q2_w;
    } else if (SFEM_HEX8_QUADRATURE_ORDER == 5) {
        n_qp = line_q6_n;
        qx = line_q6_x;
        qw = line_q6_w;
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
            hex8_adjugate_and_det(
                    lx, ly, lz, 0.5, 0.5, 0.5, jacobian_adjugate, &jacobian_determinant);

            // Assemble the diagonal part of the matrix
            for (int edof_i = 0; edof_i < 8; edof_i++) {
                accumulator_t element_matrix[6] = {0, 0, 0, 0, 0, 0};
                // Using Taylor expansion technique for symbolic integration for i,j pair
                // hex8_linear_elasticity_matrix_coord_taylor_sym(mu,
                //                                                lambda,
                //                                                jacobian_adjugate,
                //                                                jacobian_determinant,
                //                                                hex8_g_0[edof_i],
                //                                                hex8_g_0[edof_i],
                //                                                hex8_H_0[edof_i],
                //                                                hex8_H_0[edof_i],
                //                                                hex8_diff3_0,
                //                                                hex8_diff3_0,
                //                                                element_matrix);
                

                for (int zi = 0; zi < n_qp; zi++) {
                    for (int yi = 0; yi < n_qp; yi++) {
                        for (int xi = 0; xi < n_qp; xi++) {
                            scalar_t test_grad[3];
                            hex8_ref_shape_grad(edof_i, qx[xi], qx[yi], qx[zi], test_grad);
                            linear_elasticity_matrix_sym(mu,
                                                         lambda,
                                                         jacobian_adjugate,
                                                         jacobian_determinant,
                                                         test_grad,
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
                const int lenrow = rowptr[ev[edof_i] + 1] - rowptr[ev[edof_i]];
                const idx_t *cols = &colidx[rowptr[ev[edof_i]]];
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
                        // Using Taylor expansion technique for symbolic integration for i,j pair
                        // (2667 * 6)/3953 = 4 X Ops than assemblying the whole element matrix
                        // 6/576 = 1/96 of the buffer memory used to store the local results or 1/50
                        // for symmetric storage for whole element matrix.
                        // Overall smaller code size of the computational kernel.
                        // hex8_linear_elasticity_matrix_coord_taylor_sym(mu,
                        //                                                lambda,
                        //                                                jacobian_adjugate,
                        //                                                jacobian_determinant,
                        //                                                hex8_g_0[edof_i],
                        //                                                hex8_g_0[edof_j],
                        //                                                hex8_H_0[edof_i],
                        //                                                hex8_H_0[edof_j],
                        //                                                hex8_diff3_0,
                        //                                                hex8_diff3_0,
                        //                                                element_matrix);

                        for (int zi = 0; zi < n_qp; zi++) {
                            for (int yi = 0; yi < n_qp; yi++) {
                                for (int xi = 0; xi < n_qp; xi++) {
                                    scalar_t trial_grad[3];
                                    scalar_t test_grad[3];
                                    hex8_ref_shape_grad(edof_i, qx[xi], qx[yi], qx[zi], trial_grad);
                                    hex8_ref_shape_grad(edof_j, qx[xi], qx[yi], qx[zi], test_grad);
                                    linear_elasticity_matrix_sym(mu,
                                                                 lambda,
                                                                 jacobian_adjugate,
                                                                 jacobian_determinant,
                                                                 trial_grad,
                                                                 test_grad,
                                                                 qw[xi] * qw[yi] * qw[zi],
                                                                 element_matrix);
                                }
                            }
                        }

                        // printf("(%d, %d) -> (%d, %d):\n", edof_i, edof_j, ev[edof_i],
                        // ev[edof_j]); print_matrix(1, 6, element_matrix);

                        // local to global
                        int d_idx = 0;
                        for (int d1 = 0; d1 < 3; d1++) {
                            for (int d2 = d1; d2 < 3; d2++, d_idx++) {
                                real_t *values =
                                        &block_offdiag[d_idx][(rowptr[ev[edof_i]] + ks[edof_j]) *
                                                              block_stride];
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
