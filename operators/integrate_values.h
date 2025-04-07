#ifndef INTEGRATE_VALUES_H
#define INTEGRATE_VALUES_H

#include "sfem_base.h"
#include "stddef.h"

#ifdef __cplusplus
extern "C" {
#endif

int integrate_value(const int                    element_type,
                    const ptrdiff_t              nelements,
                    const ptrdiff_t              nnodes,
                    idx_t **const SFEM_RESTRICT  elems,
                    geom_t **const SFEM_RESTRICT xyz,
                    const real_t                 value,
                    const int                    block_size,
                    const int                    component,
                    real_t *const SFEM_RESTRICT  out);

int integrate_values(const int                         element_type,
                     const ptrdiff_t                   nelements,
                     const ptrdiff_t                   nnodes,
                     idx_t **const SFEM_RESTRICT       elems,
                     geom_t **const SFEM_RESTRICT      xyz,
                     const real_t                      scale_factor,
                     const real_t *const SFEM_RESTRICT values,
                     const int                         block_size,
                     const int                         component,
                     real_t *const SFEM_RESTRICT       out);

#ifdef __cplusplus
}
#endif

#endif  // INTEGRATE_VALUES_H
