#ifndef HEX8_MASS_H
#define HEX8_MASS_H
#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int hex8_apply_mass(const ptrdiff_t nelements,
                     const ptrdiff_t nnodes,
                     idx_t **const SFEM_RESTRICT elements,
                     geom_t **const SFEM_RESTRICT points,
                     const ptrdiff_t stride_u,
                     const real_t *const SFEM_RESTRICT u,
                     const ptrdiff_t stride_values,
                     real_t *const SFEM_RESTRICT values);

int hex8_assemble_lumped_mass(const ptrdiff_t nelements,
                               const ptrdiff_t nnodes,
                               idx_t **const SFEM_RESTRICT elements,
                               geom_t **const SFEM_RESTRICT points,
                               const ptrdiff_t stride_values,
                               real_t *const SFEM_RESTRICT values);

#ifdef __cplusplus
}
#endif

#endif  // HEX8_MASS_H
