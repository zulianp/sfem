#ifndef TET4_ADJUGATE_H
#define TET4_ADJUGATE_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"


#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Per element mesh representation with adjugate = (J^-1) * det(J)
 * where J is the Jacobian of the element transformation
 */
typedef struct {
    ptrdiff_t nelements;
    jacobian_t *jacobian_adjugate;
    jacobian_t *jacobian_determinant;
    idx_t **elements;
    enum ElemType element_type;
} tet4_adjugate_t;

void tet4_adjugate_fill(const ptrdiff_t nelements,
                        idx_t **const SFEM_RESTRICT elements,
                        geom_t **const SFEM_RESTRICT points,
                        jacobian_t *const SFEM_RESTRICT jacobian_adjugate,
                        jacobian_t *const SFEM_RESTRICT jacobian_determinant);

void tet4_adjugate_create(tet4_adjugate_t *ctx,
                          const ptrdiff_t nelements,
                          idx_t **const SFEM_RESTRICT elements,
                          geom_t **const SFEM_RESTRICT points);

void tet4_adjugate_destroy(tet4_adjugate_t *ctx);

#ifdef __cplusplus
}
#endif
#endif  // TET4_ADJUGATE_H
