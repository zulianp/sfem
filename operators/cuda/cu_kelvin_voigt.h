#ifndef KELVIN_VOIGT_INCORE_CUDA_H
#define KELVIN_VOIGT_INCORE_CUDA_H

#include <stddef.h>

#include "boundary_condition.h"
#include "sfem_base.h"

#include "cu_hex8_kelvin_voigt.h"

#ifdef __cplusplus
extern "C" {
#endif


int cu_kelvin_voigt_apply(const enum ElemType             element_type,
    const ptrdiff_t                 nelements,
    idx_t **const SFEM_RESTRICT     elements,
    const ptrdiff_t                 jacobian_stride,
    const void *const SFEM_RESTRICT jacobian_adjugate,
    const void *const SFEM_RESTRICT jacobian_determinant,
    const real_t                    k,
    const real_t                    K,
    const real_t                    eta,
    const real_t                    rho,
    const enum RealType             real_type,
    const real_t *const             d_x,
    const real_t *const             d_v,
    const real_t *const             d_a,
    real_t *const                   d_y,
    void                           *stream);

    
#ifdef __cplusplus
}
#endif
#endif  // KELVIN_VOIGT_INCORE_CUDA_H
