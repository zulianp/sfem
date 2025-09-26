#ifndef CU_HEX8_KELVIN_VOIGHT_H
#define CU_HEX8_KELVIN_VOIGHT_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_affine_hex8_kelvin_voigt_apply(const ptrdiff_t                 nelements,
    idx_t **const SFEM_RESTRICT     elements,
    const ptrdiff_t                 jacobian_stride,
    const void *const SFEM_RESTRICT jacobian_adjugate,
    const void *const SFEM_RESTRICT jacobian_determinant,
    const real_t                    k,
    const real_t                    K,
    const real_t                    eta,
    const real_t                    rho,
    const enum RealType             real_type,
    const ptrdiff_t                 u_stride,
    const void *const SFEM_RESTRICT ux,
    const void *const SFEM_RESTRICT uy,
    const void *const SFEM_RESTRICT uz,
    const void *const SFEM_RESTRICT vx,
    const void *const SFEM_RESTRICT vy,
    const void *const SFEM_RESTRICT vz,
    const void *const SFEM_RESTRICT ax,
    const void *const SFEM_RESTRICT ay,
    const void *const SFEM_RESTRICT az,
    const ptrdiff_t                 out_stride,
    void *const SFEM_RESTRICT       outx,
    void *const SFEM_RESTRICT       outy,
    void *const SFEM_RESTRICT       outz,
    void                           *stream);;


#ifdef __cplusplus
}
#endif
#endif  // CU_HEX8_KELVIN_VOIGHT_H
