#include "sshex8_stencil_element_matrix_apply.hpp"

#include "packed_elements.hpp"
#include "sshex8.hpp"
#include "sshex8_skeleton_stencil.hpp"
#include "stencil3.hpp"

#include <string.h>

int sshex8_stencil_element_matrix_apply(const int                           level,
                                        const ptrdiff_t                     nelements,
                                        idx_t **const SFEM_RESTRICT         elements,
                                        const scalar_t *const SFEM_RESTRICT g_element_matrix,
                                        const real_t *const SFEM_RESTRICT   u,
                                        real_t *const SFEM_RESTRICT         values) {
    const int nxe  = sshex8_nxe(level);
    const int txe  = sshex8_txe(level);
    const int Lm1  = level - 1;
    const int Lm13 = Lm1 * Lm1 * Lm1;

#pragma omp parallel
    {
        // Allocation per thread
        scalar_t      *eu = (scalar_t *)malloc(nxe * sizeof(scalar_t));
        idx_t         *ev = (idx_t *)malloc(nxe * sizeof(idx_t));
        accumulator_t *v  = (accumulator_t *)malloc(nxe * sizeof(accumulator_t));

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

            const scalar_t *const element_matrix = &g_element_matrix[e * 64];

            scalar_t laplacian_stencil[3 * 3 * 3];
            hex8_matrix_to_stencil(element_matrix, laplacian_stencil);
            sshex8_stencil(
                // count
                level + 1, level + 1, level + 1, 
                // buffers
                laplacian_stencil, eu, v);
            
            sshex8_surface_stencil(
                    // count
                    level + 1, level + 1, level + 1, 
                    // stide
                    1, level + 1, (level + 1) * (level + 1), 
                    // buffers
                    element_matrix, eu, v);

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

int sshex8_stencil_element_matrix_apply_hyteg(const int                           level,
                                              const ptrdiff_t                     nelements,
                                              idx_t **const SFEM_RESTRICT         elements,
                                              const scalar_t *const SFEM_RESTRICT g_element_matrix,
                                              const scalar_t *const SFEM_RESTRICT g_stencil,
                                              const real_t *const SFEM_RESTRICT   u,
                                              real_t *const SFEM_RESTRICT         values) {
    const int nxe = sshex8_nxe(level);

#pragma omp parallel
    {
        scalar_t      *eu = (scalar_t *)malloc(nxe * sizeof(scalar_t));
        idx_t         *ev = (idx_t *)malloc(nxe * sizeof(idx_t));
        accumulator_t *v  = (accumulator_t *)malloc(nxe * sizeof(accumulator_t));

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            for (int d = 0; d < nxe; d++) {
                ev[d] = elements[d][e];
            }

            for (int d = 0; d < nxe; d++) {
                eu[d] = u[ev[d]];
                assert(eu[d] == eu[d]);
            }

            memset(v, 0, nxe * sizeof(accumulator_t));

            const scalar_t *const element_matrix = &g_element_matrix[e * 64];
            const scalar_t *const stencil        = &g_stencil[e * 27];
            scalar_t             laplacian_stencil[3 * 3 * 3];
            for (int d = 0; d < 3 * 3 * 3; ++d) {
                laplacian_stencil[d] = stencil[d];
            }

            sshex8_stencil(
                    level + 1,
                    level + 1,
                    level + 1,
                    laplacian_stencil,
                    eu,
                    v);

            sshex8_surface_stencil(
                    level + 1,
                    level + 1,
                    level + 1,
                    1,
                    level + 1,
                    (level + 1) * (level + 1),
                    element_matrix,
                    eu,
                    v);

            for (int d = 0; d < nxe; d++) {
                assert(v[d] == v[d]);
#pragma omp atomic update
                values[ev[d]] += v[d];
            }
        }

        free(ev);
        free(eu);
        free(v);
    }

    return SFEM_SUCCESS;
}

int sshex8_stencil_element_matrix_apply_hyteg_vectorized(const int                           level,
                                                         const ptrdiff_t                     nelements,
                                                         idx_t **const SFEM_RESTRICT         elements,
                                                         const scalar_t *const SFEM_RESTRICT g_element_matrix,
                                                         const scalar_t *const SFEM_RESTRICT g_stencil,
                                                         const real_t *const SFEM_RESTRICT   u,
                                                         real_t *const SFEM_RESTRICT         values) {
    const int nxe = sshex8_nxe(level);

#pragma omp parallel
    {
        scalar_t      *eu = (scalar_t *)malloc(nxe * sizeof(scalar_t));
        idx_t         *ev = (idx_t *)malloc(nxe * sizeof(idx_t));
        accumulator_t *v  = (accumulator_t *)malloc(nxe * sizeof(accumulator_t));

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            for (int d = 0; d < nxe; d++) {
                ev[d] = elements[d][e];
            }

            for (int d = 0; d < nxe; d++) {
                eu[d] = u[ev[d]];
                assert(eu[d] == eu[d]);
            }

            memset(v, 0, nxe * sizeof(accumulator_t));

            const scalar_t *const element_matrix    = &g_element_matrix[e * 64];
            const scalar_t *const laplacian_stencil = &g_stencil[e * 27];

            sshex8_stencil_fused_vectorized(
                    level + 1,
                    level + 1,
                    level + 1,
                    laplacian_stencil,
                    eu,
                    v);

            sshex8_surface_stencil(
                    level + 1,
                    level + 1,
                    level + 1,
                    1,
                    level + 1,
                    (level + 1) * (level + 1),
                    element_matrix,
                    eu,
                    v);

            for (int d = 0; d < nxe; d++) {
                assert(v[d] == v[d]);
#pragma omp atomic update
                values[ev[d]] += v[d];
            }
        }

        free(ev);
        free(eu);
        free(v);
    }

    return SFEM_SUCCESS;
}

int sshex8_stencil_element_matrix_apply3(const int                           level,
                                         const ptrdiff_t                     nelements,
                                         idx_t **const SFEM_RESTRICT         elements,
                                         const scalar_t *const SFEM_RESTRICT g_element_matrix,
                                         const ptrdiff_t                     u_stride,
                                         const real_t *const SFEM_RESTRICT   ux,
                                         const real_t *const SFEM_RESTRICT   uy,
                                         const real_t *const SFEM_RESTRICT   uz,
                                         const ptrdiff_t                     out_stride,
                                         real_t *const SFEM_RESTRICT         outx,
                                         real_t *const SFEM_RESTRICT         outy,
                                         real_t *const SFEM_RESTRICT         outz) {
    const int nxe = sshex8_nxe(level);
    const int txe = sshex8_txe(level);

#pragma omp parallel
    {
        scalar_t *eu[3];
        scalar_t *v[3];

        for (int d = 0; d < 3; d++) {
            eu[d] = (scalar_t *)malloc(nxe * sizeof(scalar_t));
            v[d]  = (scalar_t *)malloc(nxe * sizeof(scalar_t));
        }

        idx_t    *ev = (idx_t *)malloc(nxe * sizeof(idx_t));
        scalar_t *X  = (scalar_t *)malloc(txe * 24 * sizeof(scalar_t));
        scalar_t *Y  = (scalar_t *)malloc(txe * 24 * sizeof(scalar_t));

#pragma omp for
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            for (int d = 0; d < nxe; d++) {
                ev[d] = elements[d][e];
            }

            for (int d = 0; d < nxe; d++) {
                const ptrdiff_t idx = ev[d] * u_stride;
                eu[0][d]           = ux[idx];
                eu[1][d]           = uy[idx];
                eu[2][d]           = uz[idx];
                assert(eu[0][d] == eu[0][d]);
                assert(eu[1][d] == eu[1][d]);
                assert(eu[2][d] == eu[2][d]);
            }

            sshex8_SoA_pack_elements(level, eu, X);
            packed_elements_matmul(24, txe, 24, &g_element_matrix[e * 24 * 24], X, Y);

            for (int d = 0; d < 3; d++) {
                memset(v[d], 0, nxe * sizeof(scalar_t));
            }

            sshex8_SoA_unpack_add_elements(level, Y, v);

            for (int d = 0; d < nxe; d++) {
                const ptrdiff_t idx = ev[d] * out_stride;

#pragma omp atomic update
                outx[idx] += v[0][d];

#pragma omp atomic update
                outy[idx] += v[1][d];

#pragma omp atomic update
                outz[idx] += v[2][d];
            }
        }

        free(ev);
        free(X);
        free(Y);

        for (int d = 0; d < 3; d++) {
            free(eu[d]);
            free(v[d]);
        }
    }

    return SFEM_SUCCESS;
}
