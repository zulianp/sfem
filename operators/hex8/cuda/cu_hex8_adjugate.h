#ifndef CU_HEX8_ADJUGATE_H
#define CU_HEX8_ADJUGATE_H

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_hex8_adjugate_allocate(const ptrdiff_t nelements,
                              void **const SFEM_RESTRICT jacobian_adjugate,
                              void **const SFEM_RESTRICT jacobian_determinant);

int cu_hex8_adjugate_fill(const ptrdiff_t nelements,
                          idx_t **const SFEM_RESTRICT elements,
                          geom_t **const SFEM_RESTRICT points,
                          void *const SFEM_RESTRICT jacobian_adjugate,
                          void *const SFEM_RESTRICT jacobian_determinant);

#ifdef __cplusplus
}
#endif

#endif  // CU_HEX8_ADJUGATE_H