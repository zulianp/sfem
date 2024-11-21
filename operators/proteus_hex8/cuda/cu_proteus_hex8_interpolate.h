#ifndef CU_PROTEUS_HEX8_INTERPOLATE_H
#define CU_PROTEUS_HEX8_INTERPOLATE_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_proteus_hex8_hierarchical_prolongation(const int level,
                                              const ptrdiff_t nelements,
                                              const ptrdiff_t stride,
                                              const idx_t *const SFEM_RESTRICT elements,
                                              const int vec_size,
                                              const enum RealType from_type,
                                              const ptrdiff_t from_stride,
                                              const void *const SFEM_RESTRICT from,
                                              const enum RealType to_type,
                                              const ptrdiff_t to_stride,
                                              void *const SFEM_RESTRICT to,
                                              void *stream);

int cu_proteus_hex8_hierarchical_restriction(const int level,
                                              const ptrdiff_t nelements,
                                              const ptrdiff_t stride,
                                              const idx_t *const SFEM_RESTRICT elements,
                                              const uint16_t *const SFEM_RESTRICT
                                                      element_to_node_incidence_count,
                                              const int vec_size,
                                              const enum RealType from_type,
                                              const ptrdiff_t from_stride,
                                              const void *const SFEM_RESTRICT from,
                                              const enum RealType to_type,
                                              const ptrdiff_t to_stride,
                                              void *const SFEM_RESTRICT to,
                                              void *stream);
#ifdef __cplusplus
}
#endif
#endif  // CU_PROTEUS_HEX8_INTERPOLATE_H
