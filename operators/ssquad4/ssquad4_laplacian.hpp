#ifndef SFEM_SSQUAD4_LAPLACIAN_H
#define SFEM_SSQUAD4_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.hpp"

#ifdef __cplusplus
extern "C" {
#endif

int ssquad4_laplacian_apply(const int                         level,
                            const ptrdiff_t                   nelements,
                            idx_t **const SFEM_RESTRICT       elements,
                            geom_t **const SFEM_RESTRICT      points,
                            const real_t *const SFEM_RESTRICT u,
                            real_t *const SFEM_RESTRICT       values);

int ssquad4_laplacian_diag(const int                    level,
                           const ptrdiff_t              nelements,
                           idx_t **const SFEM_RESTRICT  elements,
                           geom_t **const SFEM_RESTRICT points,
                           real_t *const SFEM_RESTRICT  values);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_SSQUAD4_LAPLACIAN_H
