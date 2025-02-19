#ifndef SSQUAD4_INTERPOLATE_H
#define SSQUAD4_INTERPOLATE_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int ssquad4_element_node_incidence_count(const int                     level,
                                         const int                     stride,
                                         const ptrdiff_t               nelements,
                                         idx_t **const SFEM_RESTRICT   elements,
                                         uint16_t *const SFEM_RESTRICT count);

int ssquad4_restrict(const ptrdiff_t                     nelements,
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

int ssquad4_prolongate(const ptrdiff_t                   nelements,
                       const int                         from_level,
                       const int                         from_level_stride,
                       idx_t **const SFEM_RESTRICT       from_elements,
                       const int                         to_level,
                       const int                         to_level_stride,
                       idx_t **const SFEM_RESTRICT       to_elements,
                       const int                         vec_size,
                       const real_t *const SFEM_RESTRICT from,
                       real_t *const SFEM_RESTRICT       to);

int ssquad4_prolongation_crs_nnz(const int                    level,
                                 const ptrdiff_t              nelements,
                                 idx_t **const SFEM_RESTRICT  elements,
                                 const ptrdiff_t              to_nnodes,
                                 count_t *const SFEM_RESTRICT rowptr);

int ssquad4_prolongation_crs_fill(const int                    level,
                                  const ptrdiff_t              nelements,
                                  idx_t **const SFEM_RESTRICT  elements,
                                  const ptrdiff_t              to_nnodes,
                                  count_t *const SFEM_RESTRICT rowptr,
                                  idx_t *const SFEM_RESTRICT   colidx,
                                  real_t *const SFEM_RESTRICT  values);

#ifdef __cplusplus
}
#endif

#endif  // SSQUAD4_INTERPOLATE_H