#include "quad4_laplacian.hpp"

#include "quad4_laplacian_inline_cpu.hpp"

int quad4_laplacian_apply(const ptrdiff_t                   nelements,
                          const ptrdiff_t                   nnodes,
                          idx_t **const SFEM_RESTRICT       elements,
                          geom_t **const SFEM_RESTRICT      points,
                          const real_t *const SFEM_RESTRICT u,
                          real_t *const SFEM_RESTRICT       values) {
    (void)nnodes;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        const idx_t ev[4] = {elements[0][e], elements[1][e], elements[2][e], elements[3][e]};
        real_t      element_vector[4];
        quad4_laplacian_apply_micro(ev, points, u, element_vector);
        for (int a = 0; a < 4; ++a) {
#pragma omp atomic update
            values[ev[a]] += element_vector[a];
        }
    }

    return SFEM_SUCCESS;
}

int quad4_laplacian_diag(const ptrdiff_t              nelements,
                         const ptrdiff_t              nnodes,
                         idx_t **const SFEM_RESTRICT  elements,
                         geom_t **const SFEM_RESTRICT points,
                         real_t *const SFEM_RESTRICT  values) {
    (void)nnodes;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        const idx_t ev[4] = {elements[0][e], elements[1][e], elements[2][e], elements[3][e]};
        real_t      element_diag[4];
        quad4_laplacian_diag_micro(ev, points, element_diag);
        for (int a = 0; a < 4; ++a) {
#pragma omp atomic update
            values[ev[a]] += element_diag[a];
        }
    }

    return SFEM_SUCCESS;
}
