#ifndef HEX8_LAPLACIAN_H
#define HEX8_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int hex8_laplacian_apply(const ptrdiff_t                   nelements,
                         const ptrdiff_t                   nnodes,
                         idx_t **const SFEM_RESTRICT       elements,
                         geom_t **const SFEM_RESTRICT      points,
                         const real_t *const SFEM_RESTRICT u,
                         real_t *const SFEM_RESTRICT       values);

int hex8_laplacian_crs_sym(const ptrdiff_t                    nelements,
                           const ptrdiff_t                    nnodes,
                           idx_t **const SFEM_RESTRICT        elements,
                           geom_t **const SFEM_RESTRICT       points,
                           const count_t *const SFEM_RESTRICT rowptr,
                           const idx_t *const SFEM_RESTRICT   colidx,
                           real_t *const SFEM_RESTRICT        diag,
                           real_t *const SFEM_RESTRICT        offdiag);

int hex8_laplacian_crs(const ptrdiff_t                    nelements,
                       const ptrdiff_t                    nnodes,
                       idx_t **const SFEM_RESTRICT        elements,
                       geom_t **const SFEM_RESTRICT       points,
                       const count_t *const SFEM_RESTRICT rowptr,
                       const idx_t *const SFEM_RESTRICT   colidx,
                       real_t *const SFEM_RESTRICT        values);


int hex8_laplacian_diag(const ptrdiff_t              nelements,
                        const ptrdiff_t              nnodes,
                        idx_t **const SFEM_RESTRICT  elements,
                        geom_t **const SFEM_RESTRICT points,
                        real_t *const SFEM_RESTRICT  diag);

#ifdef __cplusplus
}
#endif
#endif  // HEX8_LAPLACIAN_H
