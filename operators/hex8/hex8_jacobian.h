#ifndef HEX8_JACOBIAN_H
#define HEX8_JACOBIAN_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int hex8_adjugate_and_det_fill(const ptrdiff_t                 nelements,
                               idx_t **const SFEM_RESTRICT     elements,
                               geom_t **const SFEM_RESTRICT    points,
                               jacobian_t *const SFEM_RESTRICT adjugate,
                               jacobian_t *const SFEM_RESTRICT determinant);

#ifdef __cplusplus
}
#endif
#endif  // HEX8_JACOBIAN_H
