#ifndef CU_HEX8_LAPLACIAN_H
#define CU_HEX8_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_affine_hex8_laplacian_apply(const ptrdiff_t nelements,
                                   const ptrdiff_t stride,  // Stride for elements and fff
                                   const idx_t *const SFEM_RESTRICT elements,
                                   const void *const SFEM_RESTRICT fff,
                                   const enum RealType real_type_xy,
                                   const void *const SFEM_RESTRICT x, void *const SFEM_RESTRICT y,
                                   void *stream);

int cu_affine_hex8_laplacian_taylor_apply(const ptrdiff_t nelements,
                                          const ptrdiff_t stride,  // Stride for elements and fff
                                          const idx_t *const SFEM_RESTRICT elements,
                                          const void *const SFEM_RESTRICT fff,
                                          const enum RealType real_type_xy, const void *const x,
                                          void *const y, void *stream);

int cu_affine_hex8_laplacian_crs_sym(const ptrdiff_t nelements,
                                          const ptrdiff_t stride,  // Stride for elements and fff
                                          const idx_t *const SFEM_RESTRICT elements,
                                          const void *const SFEM_RESTRICT fff,
                                          const count_t *const SFEM_RESTRICT rowptr,
                                          const idx_t *const SFEM_RESTRICT colidx,
                                          const enum RealType real_type,
                                          void *const SFEM_RESTRICT diag,
                                          void *const SFEM_RESTRICT offdiag, void *stream);

#ifdef __cplusplus
}
#endif
#endif  // CU_HEX8_LAPLACIAN_H
