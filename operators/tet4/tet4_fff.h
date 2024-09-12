#ifndef TET4_FFF_H
#define TET4_FFF_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Per element mesh representation with fff = (J * J^T) * det(J)/6,
 * where J is the Jacobian of the element transformation
 */
typedef struct {
    ptrdiff_t nelements;
    jacobian_t *data;
    idx_t **elements;
    enum ElemType element_type;
} fff_t;

int tet4_fff_fill(const ptrdiff_t nelements,
                   idx_t **const SFEM_RESTRICT elements,
                   geom_t **const SFEM_RESTRICT points,
                   jacobian_t *const SFEM_RESTRICT fff);

int tet4_fff_create(fff_t *ctx,
                     const ptrdiff_t nelements,
                     idx_t **const SFEM_RESTRICT elements,
                     geom_t **const SFEM_RESTRICT points);

int tet4_fff_destroy(fff_t *ctx);

#ifdef __cplusplus
}
#endif
#endif  // TET4_FFF_H
