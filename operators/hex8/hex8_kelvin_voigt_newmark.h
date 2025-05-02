#ifndef HEX8_KELVIN_VOIGT_NEWMARK_H
#define HEX8_KELVIN_VOIGT_NEWMARK_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

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
                                    real_t *const SFEM_RESTRICT       outz);

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
                                       real_t *const SFEM_RESTRICT outz);

#ifdef __cplusplus
}
#endif
#endif  // HEX8_KELVIN_VOIGT_NEWMARK_H
