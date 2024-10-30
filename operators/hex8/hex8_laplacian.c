#include "hex8_laplacian.h"

#include "sfem_defs.h"

#include "tet4_inline_cpu.h"
#include "hex8_quadrature.h"
#include "hex8_laplacian_inline_cpu.h"

int hex8_laplacian_apply(const ptrdiff_t nelements,
                         const ptrdiff_t nnodes,
                         idx_t **const SFEM_RESTRICT elements,
                         geom_t **const SFEM_RESTRICT points,
                         const real_t *const SFEM_RESTRICT u,
                         real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    int SFEM_HEX8_ASSUME_AXIS_ALIGNED = 0;
    int SFEM_HEX8_ASSUME_AFFINE = 0;
    SFEM_READ_ENV(SFEM_HEX8_ASSUME_AXIS_ALIGNED, atoi);
    SFEM_READ_ENV(SFEM_HEX8_ASSUME_AFFINE, atoi);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    if (SFEM_HEX8_ASSUME_AXIS_ALIGNED) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[8];
            accumulator_t element_vector[8];
            scalar_t element_u[8];
            scalar_t jac_diag[3];

            for (int v = 0; v < 8; ++v) {
                ev[v] = elements[v][i];
            }

            for (int v = 0; v < 8; ++v) {
                element_u[v] = u[ev[v]];
            }

            // Assume affine here!
            aahex8_jac_diag(x[ev[0]], x[ev[6]], y[ev[0]], y[ev[6]], z[ev[0]], z[ev[6]], jac_diag);
            aahex8_laplacian_apply_integral(jac_diag, element_u, element_vector);

            for (int edof_i = 0; edof_i < 8; ++edof_i) {
                const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
                values[dof_i] += element_vector[edof_i];
            }
        }
    } else if (SFEM_HEX8_ASSUME_AFFINE) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[8];
            accumulator_t element_vector[8];
            scalar_t element_u[8];
            scalar_t fff[6];

            for (int v = 0; v < 8; ++v) {
                ev[v] = elements[v][i];
            }

            for (int v = 0; v < 8; ++v) {
                element_u[v] = u[ev[v]];
            }

            const scalar_t lx[8] = {
                    x[ev[0]], x[ev[1]], x[ev[2]], x[ev[3]], x[ev[4]], x[ev[5]], x[ev[6]], x[ev[7]]};

            const scalar_t ly[8] = {
                    y[ev[0]], y[ev[1]], y[ev[2]], y[ev[3]], y[ev[4]], y[ev[5]], y[ev[6]], y[ev[7]]};

            const scalar_t lz[8] = {
                    z[ev[0]], z[ev[1]], z[ev[2]], z[ev[3]], z[ev[4]], z[ev[5]], z[ev[6]], z[ev[7]]};

            // Assume affine here!
            hex8_fff(lx, ly, lz, 0.5, 0.5, 0.5, fff);
            hex8_laplacian_apply_fff_integral(fff, element_u, element_vector);
            // hex8_laplacian_apply_fff_taylor(fff, element_u, element_vector);

            for (int edof_i = 0; edof_i < 8; ++edof_i) {
                const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
                values[dof_i] += element_vector[edof_i];
            }
        }

    } else {
        int SFEM_HEX8_QUADRATURE_ORDER = 27;
        SFEM_READ_ENV(SFEM_HEX8_QUADRATURE_ORDER, atoi);

        int n_qp = q27_n;
        const scalar_t *qx = q27_x;
        const scalar_t *qy = q27_y;
        const scalar_t *qz = q27_z;
        const scalar_t *qw = q27_w;

        if (SFEM_HEX8_QUADRATURE_ORDER == 58) {
            n_qp = q58_n;
            qx = q58_x;
            qy = q58_y;
            qz = q58_z;
            qw = q58_w;
        } else if (SFEM_HEX8_QUADRATURE_ORDER == 6) {
            n_qp = q6_n;
            qx = q6_x;
            qy = q6_y;
            qz = q6_z;
            qw = q6_w;
        }

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[8];
            accumulator_t element_vector[8] = {0};
            scalar_t element_u[8];

            for (int v = 0; v < 8; ++v) {
                ev[v] = elements[v][i];
            }

            for (int v = 0; v < 8; ++v) {
                element_u[v] = u[ev[v]];
            }

            const scalar_t lx[8] = {
                    x[ev[0]], x[ev[1]], x[ev[2]], x[ev[3]], x[ev[4]], x[ev[5]], x[ev[6]], x[ev[7]]};

            const scalar_t ly[8] = {
                    y[ev[0]], y[ev[1]], y[ev[2]], y[ev[3]], y[ev[4]], y[ev[5]], y[ev[6]], y[ev[7]]};

            const scalar_t lz[8] = {
                    z[ev[0]], z[ev[1]], z[ev[2]], z[ev[3]], z[ev[4]], z[ev[5]], z[ev[6]], z[ev[7]]};

            for (int k = 0; k < n_qp; k++) {
                hex8_laplacian_apply_points(
                        lx, ly, lz, qx[k], qy[k], qz[k], qw[k], element_u, element_vector);
            }

            for (int edof_i = 0; edof_i < 8; ++edof_i) {
                const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
                values[dof_i] += element_vector[edof_i];
            }
        }
    }

    return SFEM_SUCCESS;
}
