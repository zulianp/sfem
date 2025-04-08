#ifndef QUADSHELL4_INTEGRATE_VALUES_H
#define QUADSHELL4_INTEGRATE_VALUES_H

#include "sfem_base.h"
#include "stddef.h"

#ifdef __cplusplus
extern "C" {
#endif

int quadshell4_integrate_value(const ptrdiff_t              nelements,
                               const ptrdiff_t              nnodes,
                               idx_t **const SFEM_RESTRICT  elements,
                               geom_t **const SFEM_RESTRICT points,
                               const real_t                 value,
                               const int                    block_size,
                               const int                    component,
                               real_t *const SFEM_RESTRICT  out);

int quadshell4_integrate_values(const ptrdiff_t                   nelements,
                                const ptrdiff_t                   nnodes,
                                idx_t **const SFEM_RESTRICT       elements,
                                geom_t **const SFEM_RESTRICT      points,
                                const real_t                      scale_factor,
                                const real_t *const SFEM_RESTRICT values,
                                const int                         block_size,
                                const int                         component,
                                real_t *const SFEM_RESTRICT       out);

#ifdef __cplusplus
}
#endif

#endif  // QUADSHELL4_INTEGRATE_VALUES_H
