#include "hex8_jacobian.h"

#include "hex8_inline_cpu.h"

int hex8_adjugate_and_det_fill(const ptrdiff_t                 nelements,
                               idx_t **const SFEM_RESTRICT     elements,
                               geom_t **const SFEM_RESTRICT    points,
                               jacobian_t *const SFEM_RESTRICT adjugate,
                               jacobian_t *const SFEM_RESTRICT determinant) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t    ev[8];
        scalar_t l_adjugate[9];
        scalar_t l_determinant;

        for (int v = 0; v < 8; ++v) {
            ev[v] = elements[v][i];
        }

        const scalar_t lx[8] = {x[ev[0]], x[ev[1]], x[ev[2]], x[ev[3]], x[ev[4]], x[ev[5]], x[ev[6]], x[ev[7]]};
        const scalar_t ly[8] = {y[ev[0]], y[ev[1]], y[ev[2]], y[ev[3]], y[ev[4]], y[ev[5]], y[ev[6]], y[ev[7]]};
        const scalar_t lz[8] = {z[ev[0]], z[ev[1]], z[ev[2]], z[ev[3]], z[ev[4]], z[ev[5]], z[ev[6]], z[ev[7]]};

        // Assume affine here!
        hex8_adjugate_and_det(lx, ly, lz, 0.5, 0.5, 0.5, l_adjugate, &l_determinant);

        for (int d = 0; d < 9; d++) {
            adjugate[i * 9 + d] = l_adjugate[d];
        }

        determinant[i] = l_determinant;
    }

    return SFEM_SUCCESS;
}