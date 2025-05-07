#include "hex8_kelvin_voigt_newmark.h"

#include "line_quadrature.h"

#include <assert.h>
#include <stdio.h>

// HAOYU

int hex8_kelvin_voigt_newmark_apply(const ptrdiff_t              nelements,
                                    const ptrdiff_t              nnodes,
                                    idx_t **const SFEM_RESTRICT  elements,
                                    geom_t **const SFEM_RESTRICT points,
                                    const ptrdiff_t              in_stride,
                                    // unified interface for both SoA and AoS
                                    const real_t *const SFEM_RESTRICT ux,
                                    const real_t *const SFEM_RESTRICT uy,
                                    const real_t *const SFEM_RESTRICT uz,
                                    const ptrdiff_t                   out_stride,
                                    real_t *const SFEM_RESTRICT       outx,
                                    real_t *const SFEM_RESTRICT       outy,
                                    real_t *const SFEM_RESTRICT       outz) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    int             n_qp = line_q2_n;
    const scalar_t *qx   = line_q2_x;
    const scalar_t *qw   = line_q2_w;

#pragma omp parallel for
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        idx_t ev[8];

        for(int v = 0;v < 8; v++) {
        	ev[v] = elements[v][e];
        }

        // Example
        const ptrdiff_t idx = ev[0] * in_stride;
        scalar_t ux0 = ux[idx];
        scalar_t uy0 = uy[idx];
        scalar_t uz0 = uz[idx];

        SFEM_ERROR("IMPLEMENT ME!");
    }

    return SFEM_SUCCESS;
}

//  F(x, x', x'') = 0
int hex8_kelvin_voigt_newmark_gradient(const ptrdiff_t              nelements,
                                       const ptrdiff_t              nnodes,
                                       idx_t **const SFEM_RESTRICT  elements,
                                       geom_t **const SFEM_RESTRICT points,
                                       // unified interface for both SoA and AoS
                                       const ptrdiff_t in_stride,
                                       // Displacement
                                       const real_t *const SFEM_RESTRICT u_oldx,
                                       const real_t *const SFEM_RESTRICT u_oldy,
                                       const real_t *const SFEM_RESTRICT u_oldz,
                                       // Velocity
                                       const real_t *const SFEM_RESTRICT v_oldx,
                                       const real_t *const SFEM_RESTRICT v_oldy,
                                       const real_t *const SFEM_RESTRICT v_oldz,
                                       // Accleration
                                       const real_t *const SFEM_RESTRICT a_oldx,
                                       const real_t *const SFEM_RESTRICT a_oldy,
                                       const real_t *const SFEM_RESTRICT a_oldz,
                                       // Current input
                                       const real_t *const SFEM_RESTRICT ux,
                                       const real_t *const SFEM_RESTRICT uy,
                                       const real_t *const SFEM_RESTRICT uz,
                                       // Output
                                       const ptrdiff_t             out_stride,
                                       real_t *const SFEM_RESTRICT outx,
                                       real_t *const SFEM_RESTRICT outy,
                                       real_t *const SFEM_RESTRICT outz) {
    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

    int             n_qp = line_q2_n;
    const scalar_t *qx   = line_q2_x;
    const scalar_t *qw   = line_q2_w;

#pragma omp parallel for
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        // TODO
        SFEM_ERROR("IMPLEMENT ME!");
    }

    return SFEM_SUCCESS;
}
