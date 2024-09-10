#ifndef PROTEUS_HEX8_LAPLACIAN_H
#define PROTEUS_HEX8_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int proteus_hex8_nxe(int level);
int proteus_hex8_txe(int level);

int proteus_hex8_laplacian_apply(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT values);

#ifdef __cplusplus
}
#endif
#endif  // PROTEUS_HEX8_LAPLACIAN_H
