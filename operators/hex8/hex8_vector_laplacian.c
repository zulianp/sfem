#include "hex8_vector_laplacian.h"

#include "sfem_defs.h"

#include "hex8_laplacian_inline_cpu.h"
#include "hex8_quadrature.h"
#include "tet4_inline_cpu.h"

int affine_hex8_vector_laplacian_apply(const ptrdiff_t              nelements,
                                       const ptrdiff_t              nnodes,
                                       idx_t **const SFEM_RESTRICT  elements,
                                       geom_t **const SFEM_RESTRICT points,
                                       const int                    vector_size,
                                       const ptrdiff_t              stride,
                                       real_t **const SFEM_RESTRICT u,
                                       real_t **const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t         ev[8];
        accumulator_t element_vector[8];
        scalar_t      element_u[8];
        scalar_t      fff[6];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        const scalar_t lx[8] = {x[ev[0]], x[ev[1]], x[ev[2]], x[ev[3]], x[ev[4]], x[ev[5]], x[ev[6]], x[ev[7]]};
        const scalar_t ly[8] = {y[ev[0]], y[ev[1]], y[ev[2]], y[ev[3]], y[ev[4]], y[ev[5]], y[ev[6]], y[ev[7]]};
        const scalar_t lz[8] = {z[ev[0]], z[ev[1]], z[ev[2]], z[ev[3]], z[ev[4]], z[ev[5]], z[ev[6]], z[ev[7]]};

        // Assume affine here!
        hex8_fff(lx, ly, lz, (scalar_t)0.5, (scalar_t)0.5, (scalar_t)0.5, fff);

        accumulator_t laplacian_matrix[8 * 8];
        hex8_laplacian_matrix_fff_integral(fff, laplacian_matrix);

        for (int d = 0; d < vector_size; d++) {
            const real_t *const ud = u[d];

            for (int v = 0; v < 8; ++v) {
                element_u[v] = ud[ev[v] * stride];
            }

            for (int i = 0; i < 8; i++) {
                element_vector[i] = 0;
            }

            for (int i = 0; i < 8; i++) {
                const scalar_t *const row = &laplacian_matrix[i * 8];
                const scalar_t        ui  = element_u[i];
                assert(ui == ui);

                for (int j = 0; j < 8; j++) {
                    assert(row[j] == row[j]);
                    element_vector[j] += ui * row[j];
                }
            }

            real_t *const vd = values[d];
            for (int edof_i = 0; edof_i < 8; ++edof_i) {
                const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
                vd[dof_i * stride] += element_vector[edof_i];
            }
        }
    }

    return SFEM_SUCCESS;
}

int affine_hex8_vector_laplacian_apply_fff(const ptrdiff_t                       nelements,
                                           idx_t **const SFEM_RESTRICT           elements,
                                           const jacobian_t *const SFEM_RESTRICT g_fff,
                                           const int                             vector_size,
                                           const ptrdiff_t                       stride,
                                           real_t **const SFEM_RESTRICT          u,
                                           real_t **const SFEM_RESTRICT          values) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t         ev[8];
        accumulator_t element_vector[8];
        scalar_t      element_u[8];

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        accumulator_t laplacian_matrix[8 * 8];
        {
            scalar_t fff[6];
            for (int d = 0; d < 6; d++) {
                fff[d] = g_fff[i * 6 + d];
            }

            hex8_laplacian_matrix_fff_integral(fff, laplacian_matrix);
        }

        for (int d = 0; d < vector_size; d++) {
            const real_t *const ud = u[d];

            for (int v = 0; v < 8; ++v) {
                element_u[v] = ud[ev[v] * stride];
            }

            for (int i = 0; i < 8; i++) {
                element_vector[i] = 0;
            }

            for (int i = 0; i < 8; i++) {
                const scalar_t *const row = &laplacian_matrix[i * 8];
                const scalar_t        ui  = element_u[i];
                assert(ui == ui);

                for (int j = 0; j < 8; j++) {
                    assert(row[j] == row[j]);
                    element_vector[j] += ui * row[j];
                }
            }

            real_t *const vd = values[d];
            for (int edof_i = 0; edof_i < 8; ++edof_i) {
                const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
                vd[dof_i * stride] += element_vector[edof_i];
            }
        }
    }

    return SFEM_SUCCESS;
}
