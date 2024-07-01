#include "vtet4_laplacian.h"

#include "vtet4_inline_cpu.h"
#include "vtet4_laplacian_inline_cpu.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

int vtet4_laplacian_apply(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const SFEM_RESTRICT elements,
                          geom_t **const SFEM_RESTRICT points,
                          const real_t *const SFEM_RESTRICT u,
                          real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; i += SFEM_VEC_SIZE) {
        const int vec_size = MIN(SFEM_VEC_SIZE, nelements - i);

        vec_t fff[6];
        {
            vec_t px[4];
            vec_t py[4];
            vec_t pz[4];

            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t *const ii = elements[edof_i];
                for (int d = 0; d < vec_size; d++) {
                    px[edof_i][d] = (scalar_t)x[ii[i + d]];
                    py[edof_i][d] = (scalar_t)y[ii[i + d]];
                    pz[edof_i][d] = (scalar_t)z[ii[i + d]];
                }
            }

            vtet4_fff(px[0],
                      px[1],
                      px[2],
                      px[3],
                      py[0],
                      py[1],
                      py[2],
                      py[3],
                      pz[0],
                      pz[1],
                      pz[2],
                      pz[3],
                      fff);
        }

        vec_t element_u[4];
        {
            for (int edof_i = 0; edof_i < 4; ++edof_i) {
                const idx_t *const ii = elements[edof_i];
                for (int d = 0; d < vec_size; d++) {
                    element_u[edof_i][d] = u[ii[i + d]];
                }
            }
        }

        vec_t element_vector[4];
        {
            vtet4_laplacian_apply_fff(fff,
                                      element_u[0],
                                      element_u[1],
                                      element_u[2],
                                      element_u[3],
                                      &element_vector[0],
                                      &element_vector[1],
                                      &element_vector[2],
                                      &element_vector[3]);
        }

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t *const ii = elements[edof_i];

            for (int d = 0; d < vec_size; d++) {
                const idx_t dof_i = ii[i + d];
#pragma omp atomic update
                values[dof_i] += element_vector[edof_i][d];
            }
        }
    }

    return 0;
}

// Optimized for matrix-free
int vtet4_laplacian_apply_opt(const ptrdiff_t nelements,
                              idx_t **const SFEM_RESTRICT elements,
                              const jacobian_t *const SFEM_RESTRICT fff_all,
                              const real_t *const SFEM_RESTRICT u,
                              real_t *const SFEM_RESTRICT values) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; i += SFEM_VEC_SIZE) {
        const int vec_size = MIN(SFEM_VEC_SIZE, nelements - i);

        vec_t element_u[4];
        vec_t element_vector[4];
        vec_t fff[6];

        for (int v = 0; v < 6; ++v) {
            for (int d = 0; d < vec_size; d++) {
                const ptrdiff_t offset = (i + d) * 6 + v;
                fff[v][d] = fff_all[offset + v];
            }
        }

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t *const ii = elements[edof_i];
            for (int d = 0; d < vec_size; d++) {
                element_u[edof_i][d] = u[ii[i + d]];
            }
        }

        vtet4_laplacian_apply_fff(fff,
                                  element_u[0],
                                  element_u[1],
                                  element_u[2],
                                  element_u[3],
                                  &element_vector[0],
                                  &element_vector[1],
                                  &element_vector[2],
                                  &element_vector[3]);

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t *const ii = elements[edof_i];

            for (int d = 0; d < vec_size; d++) {
                const idx_t dof_i = ii[i + d];
#pragma omp atomic update
                values[dof_i] += element_vector[edof_i][d];
            }
        }
    }

    return 0;
}
