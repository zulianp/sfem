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
        const ptrdiff_t ii  = static_cast<ptrdiff_t>(idx[i]) * dim;
        real_t *const   oii = &out[ii];
        const real_t    fi  = f[i] * m[i];
        for (int d = 0; d < dim; d++) {
            oii[d] += normals[d][i] * fi;
        }
    }

    return SFEM_SUCCESS;
}

int         obstacle_hessian_block_diag_sym(const int                         dim,
                                            const ptrdiff_t                   n,
                                            const idx_t *const SFEM_RESTRICT  idx,
                                            real_t **const SFEM_RESTRICT      normals,
                                            const real_t *const SFEM_RESTRICT m,
                                            const real_t *const               x,
                                            real_t *const                     values) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; ++i) {
        real_t *const v = &values[i * 6];

        int d_idx = 0;
        for (int d1 = 0; d1 < dim; d1++) {
            for (int d2 = d1; d2 < dim; d2++) {
                v[d_idx++] += m[i] * normals[d1][i] * normals[d2][i];
            }
        }
    }

    return SFEM_SUCCESS;
}

int         obstacle_contact_stress(const int                         dim,
                                    const ptrdiff_t                   n,
                                    const idx_t *const SFEM_RESTRICT  idx,
                                    real_t **const SFEM_RESTRICT      normals,
                                    const real_t *const SFEM_RESTRICT m,
                                    const real_t *const               r,
                                    real_t *const                     s) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; ++i) {
        const ptrdiff_t b = static_cast<ptrdiff_t>(idx[i]);
        for (int d = 0; d < dim; d++) {
            const real_t ri = r[b * dim + d] / m[i];
            s[idx[i] * dim] += normals[d][i] * ri;
        }
    }

    return SFEM_SUCCESS;
}
