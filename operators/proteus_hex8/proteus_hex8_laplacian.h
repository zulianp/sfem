#ifndef PROTEUS_HEX8_LAPLACIAN_H
#define PROTEUS_HEX8_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int proteus_hex8_laplacian_apply(const int level,
                                 const ptrdiff_t nelements,
                                 ptrdiff_t interior_start,
                                 idx_t **const SFEM_RESTRICT elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT values);

int proteus_affine_hex8_laplacian_apply(const int level,
                                        const ptrdiff_t nelements,
                                        ptrdiff_t interior_start,
                                        idx_t **const SFEM_RESTRICT elements,
                                        geom_t **const SFEM_RESTRICT points,
                                        const real_t *const SFEM_RESTRICT u,
                                        real_t *const SFEM_RESTRICT values);

int proteus_affine_hex8_laplacian_diag(const int level,
                                        const ptrdiff_t nelements,
                                        ptrdiff_t interior_start,
                                        idx_t **const SFEM_RESTRICT elements,
                                        geom_t **const SFEM_RESTRICT std_hex8_points,
                                        real_t *const SFEM_RESTRICT diag);

#ifdef __cplusplus
}
#endif
#endif  // PROTEUS_HEX8_LAPLACIAN_H
