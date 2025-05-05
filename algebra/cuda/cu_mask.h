#ifndef CU_MASK_H
#define CU_MASK_H

#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_mask_nodes(const ptrdiff_t                  nnodes,
                  const idx_t *const SFEM_RESTRICT nodes,
                  const int                        block_size,
                  const int                        component,
                  mask_t *const SFEM_RESTRICT      inout);

#ifdef __cplusplus
}
#endif

#endif  // CU_MASK_H
