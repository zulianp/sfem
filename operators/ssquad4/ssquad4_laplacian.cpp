#include "ssquad4_laplacian.hpp"

#include "quad4_laplacian_inline_cpu.hpp"
#include "smesh_ssquad4.hpp"

int ssquad4_laplacian_apply(const int                         level,
                            const ptrdiff_t                   nelements,
                            idx_t **const SFEM_RESTRICT       elements,
                            geom_t **const SFEM_RESTRICT      points,
                            const real_t *const SFEM_RESTRICT u,
                            real_t *const SFEM_RESTRICT       values) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        for (int yi = 0; yi < level; ++yi) {
            for (int xi = 0; xi < level; ++xi) {
                const idx_t ev[4] = {elements[smesh::ssquad4_lidx(level, xi, yi)][e],
                                     elements[smesh::ssquad4_lidx(level, xi + 1, yi)][e],
                                     elements[smesh::ssquad4_lidx(level, xi + 1, yi + 1)][e],
                                     elements[smesh::ssquad4_lidx(level, xi, yi + 1)][e]};
                real_t      element_vector[4];
                quad4_laplacian_apply_micro(ev, points, u, element_vector);
                for (int a = 0; a < 4; ++a) {
#pragma omp atomic update
                    values[ev[a]] += element_vector[a];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}

int ssquad4_laplacian_diag(const int                    level,
                           const ptrdiff_t              nelements,
                           idx_t **const SFEM_RESTRICT  elements,
                           geom_t **const SFEM_RESTRICT points,
                           real_t *const SFEM_RESTRICT  values) {
#pragma omp parallel for schedule(static)
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        for (int yi = 0; yi < level; ++yi) {
            for (int xi = 0; xi < level; ++xi) {
                const idx_t ev[4] = {elements[smesh::ssquad4_lidx(level, xi, yi)][e],
                                     elements[smesh::ssquad4_lidx(level, xi + 1, yi)][e],
                                     elements[smesh::ssquad4_lidx(level, xi + 1, yi + 1)][e],
                                     elements[smesh::ssquad4_lidx(level, xi, yi + 1)][e]};
                real_t      element_diag[4];
                quad4_laplacian_diag_micro(ev, points, element_diag);
                for (int a = 0; a < 4; ++a) {
#pragma omp atomic update
                    values[ev[a]] += element_diag[a];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}
