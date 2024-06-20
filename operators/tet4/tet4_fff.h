#ifndef TET4_FFF_H
#define TET4_FFF_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Per element mesh representation with fff = (J * J^T) * det(J)/6,
 * where J is the Jacobian of the element transformation
 */
typedef struct {
    ptrdiff_t nelements;
    jacobian_t *fff;
    idx_t **elements;
} tet4_fff_t;

void tet4_fff_fill(const ptrdiff_t nelements,
                   idx_t **const SFEM_RESTRICT elements,
                   geom_t **const SFEM_RESTRICT points,
                   geom_t *const SFEM_RESTRICT fff);

void tet4_fff_create(tet4_fff_t *ctx,
                     const ptrdiff_t nelements,
                     idx_t **const SFEM_RESTRICT elements,
                     geom_t **const SFEM_RESTRICT points);

#ifdef __cplusplus
}
#endif
#endif  // TET4_FFF_H
