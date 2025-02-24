#include "sshex8_stencil_element_matrix_apply.h"

#include "sshex8.h"
#include "sshex8_skeleton_stencil.h"
#include "stencil3.h"

#include <string.h>

int sshex8_stencil_element_matrix_apply(const int                         level,
                                        const ptrdiff_t                   nelements,
                                        ptrdiff_t                         interior_start,
                                        idx_t **const SFEM_RESTRICT       elements,
                                        scalar_t *const SFEM_RESTRICT     g_element_matrix,
                                        const real_t *const SFEM_RESTRICT u,
                                        real_t *const SFEM_RESTRICT       values) {
    const int nxe  = sshex8_nxe(level);
    const int txe  = sshex8_txe(level);
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

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            {
                // Gather elemental data
                for (int d = 0; d < nxe; d++) {
                    ev[d] = elements[d][e];
                }

                for (int d = 0; d < nxe; d++) {
                    eu[d] = u[ev[d]];
                    assert(eu[d] == eu[d]);
                }

                memset(v, 0, nxe * sizeof(accumulator_t));
            }

            scalar_t *element_matrix = &g_element_matrix[e * 64];

            scalar_t laplacian_stencil[3 * 3 * 3];
            hex8_matrix_to_stencil(element_matrix, laplacian_stencil);
            sshex8_stencil(level + 1, level + 1, level + 1, laplacian_stencil, eu, v);
            sshex8_surface_stencil(
                    level + 1, level + 1, level + 1, 1, level + 1, (level + 1) * (level + 1), element_matrix, eu, v);

            {
                // Scatter elemental data
                for (int d = 0; d < nxe; d++) {
                    assert(v[d] == v[d]);
#pragma omp atomic update
                    values[ev[d]] += v[d];
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