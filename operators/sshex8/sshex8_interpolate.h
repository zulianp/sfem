#ifndef SSHEX8_INTERPOLATE_H
#define SSHEX8_INTERPOLATE_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int sshex8_hierarchical_restriction(int                                 level,
                                    const ptrdiff_t                     nelements,
                                    idx_t **const SFEM_RESTRICT         elements,
                                    const uint16_t *const SFEM_RESTRICT element_to_node_incidence_count,
                                    const int                           vec_size,
                                    const real_t *const SFEM_RESTRICT   from,
                                    real_t *const SFEM_RESTRICT         to);

int sshex8_hierarchical_prolongation(int                               level,
                                     const ptrdiff_t                   nelements,
                                     idx_t **const SFEM_RESTRICT       elements,
                                     const int                         vec_size,
                                     const real_t *const SFEM_RESTRICT from,
                                     real_t *const SFEM_RESTRICT       to);

int sshex8_element_node_incidence_count(const int                     level,
                                        const int                     stride,
                                        const ptrdiff_t               nelements,
                                        idx_t **const SFEM_RESTRICT   elements,
                                        uint16_t *const SFEM_RESTRICT count);

int sshex8_restrict(const ptrdiff_t                     nelements,
                    const int                           from_level,
                    const int                           from_level_stride,
                    idx_t **const SFEM_RESTRICT         from_elements,
                    const uint16_t *const SFEM_RESTRICT from_element_to_node_incidence_count,
                    const int                           to_level,
                    const int                           to_level_stride,
                    idx_t **const SFEM_RESTRICT         to_elements,
                    const int                           vec_size,
                    const real_t *const SFEM_RESTRICT   from,
                    real_t *const SFEM_RESTRICT         to);

int sshex8_prolongate(const ptrdiff_t                   nelements,
                      const int                         from_level,
                      const int                         from_level_stride,
                      idx_t **const SFEM_RESTRICT       from_elements,
                      const int                         to_level,
                      const int                         to_level_stride,
                      idx_t **const SFEM_RESTRICT       to_elements,
                      const int                         vec_size,
                      const real_t *const SFEM_RESTRICT from,
                      real_t *const SFEM_RESTRICT       to);

#ifdef __cplusplus
}
#endif
#endif  // SSHEX8_INTERPOLATE_H
