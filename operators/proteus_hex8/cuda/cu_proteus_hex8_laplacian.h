#ifndef CU_PROTEUS_HEX8_LAPLACIAN_H
#define CU_PROTEUS_HEX8_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_proteus_affine_hex8_laplacian_apply(const int level,
                                           const ptrdiff_t nelements,
                                           const ptrdiff_t stride,  // Stride for elements and fff
                                           const ptrdiff_t interior_start,
                                           const idx_t *const SFEM_RESTRICT elements,
                                           const void *const SFEM_RESTRICT fff,
                                           const enum RealType real_type_xy,
                                           const void *const SFEM_RESTRICT x,
                                           void *const SFEM_RESTRICT y,
                                           void *stream);
#ifdef __cplusplus
}
#endif
#endif  // CU_PROTEUS_HEX8_LAPLACIAN_H
