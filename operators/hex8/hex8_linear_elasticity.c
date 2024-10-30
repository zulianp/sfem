#include "hex8_linear_elasticity.h"

#include "hex8_inline_cpu.h"
#include "hex8_linear_elasticity_inline_cpu.h"
// #include "hex8_quadrature.h"
#include "line_quadrature.h"

#include <stdio.h>

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
