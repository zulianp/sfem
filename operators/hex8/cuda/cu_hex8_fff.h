#ifndef CU_HEX8_FFF_H
#define CU_HEX8_FFF_H

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_hex8_fff_allocate(const ptrdiff_t nelements,
                         void **const SFEM_RESTRICT fff);

int cu_hex8_fff_fill(const ptrdiff_t nelements,
                     idx_t **const SFEM_RESTRICT elements,
                     geom_t **const SFEM_RESTRICT points,
                     void *const SFEM_RESTRICT fff);


#ifdef __cplusplus
}
#endif

#endif  // CU_HEX8_FFF_H