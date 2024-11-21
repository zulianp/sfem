#ifndef HEX8_FFF_H
#define HEX8_FFF_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int hex8_fff_fill(const ptrdiff_t nelements,
                   idx_t **const SFEM_RESTRICT elements,
                   geom_t **const SFEM_RESTRICT points,
                   jacobian_t *const SFEM_RESTRICT fff);

#ifdef __cplusplus
}
#endif
#endif  // HEX8_FFF_H
