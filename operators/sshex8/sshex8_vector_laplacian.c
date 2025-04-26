#include "sshex8_vector_laplacian.h"

#include "hex8_inline_cpu.h"
#include "hex8_laplacian_inline_cpu.h"

#include "hex8_quadrature.h"
#include "sshex8_skeleton_stencil.h"
#include "stencil3.h"
#include "tet4_inline_cpu.h"

#include <math.h>
#include <stdio.h>

#include "sshex8.h"

int affine_sshex8_vector_laplacian_apply_fff(const int                             level,
                                             const ptrdiff_t                       nelements,
                                             idx_t **const SFEM_RESTRICT           elements,
                                             const jacobian_t *const SFEM_RESTRICT g_fff,
                                             const int                             vector_size,
                                             const ptrdiff_t                       stride,
                                             real_t **const SFEM_RESTRICT          u,
                                             real_t **const SFEM_RESTRICT          values) {
    const int nxe = sshex8_nxe(level);
    const int txe = sshex8_txe(level);

    const int Lm1  = level - 1;
    const int Lm13 = Lm1 * Lm1 * Lm1;

#pragma omp parallel
    {
        // Allocation per thread
        scalar_t      *eu = malloc(nxe * sizeof(scalar_t));
        idx_t         *ev = malloc(nxe * sizeof(idx_t));
        accumulator_t *v  = malloc(nxe * sizeof(accumulator_t));

        scalar_t      element_u[8];
        accumulator_t element_vector[8];
        scalar_t      fff[6];

        const scalar_t h = 1. / level;

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            {
                // Gather elemental data
                for (int d = 0; d < nxe; d++) {
                    ev[d] = elements[d][e];
                }

                for (int d = 0; d < 6; d++) {
                    // Sub fff
                    fff[d] = g_fff[e * 6 + d] * h;
                }
            }

            accumulator_t laplacian_matrix[8 * 8];
            hex8_laplacian_matrix_fff_integral(fff, laplacian_matrix);
            scalar_t laplacian_stencil[3 * 3 * 3];
            hex8_matrix_to_stencil(laplacian_matrix, laplacian_stencil);

            for (int b = 0; b < vector_size; b++) {
                const real_t *const ud = u[b];

                for (int d = 0; d < nxe; d++) {
                    eu[d] = ud[ev[d] * stride];
                    assert(eu[d] == eu[d]);
                }

                memset(v, 0, nxe * sizeof(accumulator_t));

                sshex8_stencil(level + 1, level + 1, level + 1, laplacian_stencil, eu, v);

                // This is slowing down things
                sshex8_surface_stencil(
                        level + 1, level + 1, level + 1, 1, level + 1, (level + 1) * (level + 1), laplacian_matrix, eu, v);

                real_t *const vd = values[b];
                {
                    // Scatter elemental data
                    for (int d = 0; d < nxe; d++) {
                        assert(v[d] == v[d]);
#pragma omp atomic update
                        vd[ev[d] * stride] += v[d];
                    }
                }
            }
        }

        // Clean-up
        free(ev);
        free(eu);
        free(v);
    }

    return SFEM_SUCCESS;
}
