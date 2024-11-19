#ifndef PROTEUS_HEX8_INTERPOLATE_H
#define PROTEUS_HEX8_INTERPOLATE_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int proteus_hex8_hierarchical_restriction(int level,
                                          const ptrdiff_t nelements,
                                          idx_t **const SFEM_RESTRICT elements,
                                          const uint16_t *const SFEM_RESTRICT
                                                  element_to_node_incidence_count,
                                          const int vec_size,
                                          const real_t *const SFEM_RESTRICT from,
                                          real_t *const SFEM_RESTRICT to);

int proteus_hex8_hierarchical_prolongation(int level,
                                           const ptrdiff_t nelements,
                                           idx_t **const SFEM_RESTRICT elements,
                                           const int vec_size,
                                           const real_t *const SFEM_RESTRICT from,
                                           real_t *const SFEM_RESTRICT to);

#ifdef __cplusplus
}
#endif
#endif  // PROTEUS_HEX8_INTERPOLATE_H
