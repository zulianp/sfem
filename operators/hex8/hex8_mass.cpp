#include "hex8_mass.h"
#include "hex8_quadrature.h"
#include "hex8_mass_inline_cpu.h"

#include <stdio.h>

int hex8_apply_mass(const ptrdiff_t                   nelements,
                    const ptrdiff_t                   nnodes,
                    idx_t **const SFEM_RESTRICT       elements,
                    geom_t **const SFEM_RESTRICT      points,
                    const ptrdiff_t                   stride_u,
                    const real_t *const SFEM_RESTRICT u,
                    const ptrdiff_t                   stride_values,
                    real_t *const SFEM_RESTRICT       values) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    int             n_qp = q27_n;
    const scalar_t *qx   = q27_x;
    const scalar_t *qy   = q27_y;
    const scalar_t *qz   = q27_z;
    const scalar_t *qw   = q27_w;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t         ev[8];
        accumulator_t element_vector[8] = {0};
        scalar_t      element_u[8];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 8; ++v) {
            element_u[v] = u[ev[v] * stride_u];
        }

        const scalar_t lx[8] = {x[ev[0]], x[ev[1]], x[ev[2]], x[ev[3]], x[ev[4]], x[ev[5]], x[ev[6]], x[ev[7]]};

        const scalar_t ly[8] = {y[ev[0]], y[ev[1]], y[ev[2]], y[ev[3]], y[ev[4]], y[ev[5]], y[ev[6]], y[ev[7]]};

        const scalar_t lz[8] = {z[ev[0]], z[ev[1]], z[ev[2]], z[ev[3]], z[ev[4]], z[ev[5]], z[ev[6]], z[ev[7]]};

        for (int k = 0; k < n_qp; k++) {
            hex8_mass_apply_points(lx, ly, lz, qx[k], qy[k], qz[k], qw[k], element_u, element_vector);
        }

        for (int edof_i = 0; edof_i < 8; ++edof_i) {
            const idx_t dof_i = ev[edof_i] * stride_values;

#pragma omp atomic update
            values[dof_i] += element_vector[edof_i];
        }
    }

    return SFEM_SUCCESS;
}

int hex8_assemble_lumped_mass(const ptrdiff_t              nelements,
                              const ptrdiff_t              nnodes,
                              idx_t **const SFEM_RESTRICT  elements,
                              geom_t **const SFEM_RESTRICT points,
                              const ptrdiff_t              stride_values,
                              real_t *const SFEM_RESTRICT  values) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    int             n_qp = q27_n;
    const scalar_t *qx   = q27_x;
    const scalar_t *qy   = q27_y;
    const scalar_t *qz   = q27_z;
    const scalar_t *qw   = q27_w;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t         ev[8];
        accumulator_t element_vector[8] = {0};

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        const scalar_t lx[8] = {x[ev[0]], x[ev[1]], x[ev[2]], x[ev[3]], x[ev[4]], x[ev[5]], x[ev[6]], x[ev[7]]};

        const scalar_t ly[8] = {y[ev[0]], y[ev[1]], y[ev[2]], y[ev[3]], y[ev[4]], y[ev[5]], y[ev[6]], y[ev[7]]};

        const scalar_t lz[8] = {z[ev[0]], z[ev[1]], z[ev[2]], z[ev[3]], z[ev[4]], z[ev[5]], z[ev[6]], z[ev[7]]};

        for (int k = 0; k < n_qp; k++) {
            hex8_lumped_mass_points(lx, ly, lz, qx[k], qy[k], qz[k], qw[k], element_vector);
        }

        for (int edof_i = 0; edof_i < 8; ++edof_i) {
            const ptrdiff_t dof_i = ev[edof_i] * stride_values;
#pragma omp atomic update
            values[dof_i] += element_vector[edof_i];
        }
    }

    return SFEM_SUCCESS;
}
