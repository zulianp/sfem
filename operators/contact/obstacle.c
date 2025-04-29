#include "obstacle.h"

int         obstacle_normal_project(const int                         dim,
                                    const ptrdiff_t                   n,
                                    const idx_t *const SFEM_RESTRICT  idx,
                                    real_t **const SFEM_RESTRICT      normals,
                                    const real_t *const SFEM_RESTRICT h,
                                    real_t *const SFEM_RESTRICT       out) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; ++i) {
        const ptrdiff_t     ii  = idx[i] * dim;
        const real_t *const hii = &h[ii];
        for (int d = 0; d < dim; d++) {
            out[i] += hii[d] * normals[d][i];
        }
    }

    return SFEM_SUCCESS;
}

int         obstacle_distribute_contact_forces(const int                         dim,
                                               const ptrdiff_t                   n,
                                               const idx_t *const SFEM_RESTRICT  idx,
                                               real_t **const SFEM_RESTRICT      normals,
                                               const real_t *const SFEM_RESTRICT m,
                                               const real_t *const               f,
                                               real_t *const                     out) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; ++i) {
        const ptrdiff_t ii  = idx[i] * dim;
        real_t *const   oii = &out[ii];
        const real_t    fi  = f[i] * m[i];
        for (int d = 0; d < dim; d++) {
            oii[d] += normals[d][i] * fi;
        }
    }

    return SFEM_SUCCESS;
}
