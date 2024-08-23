#ifndef CU_MACRO_TET4_LAPLACIAN_H
#define CU_MACRO_TET4_LAPLACIAN_H

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_macro_tet4_laplacian_apply(const ptrdiff_t nelements,
                                  const ptrdiff_t stride,  // Stride for elements and fff
                                  const idx_t *const SFEM_RESTRICT elements,
                                  const void *const SFEM_RESTRICT fff,
                                  const enum RealType real_type_xy,
                                  const void *const SFEM_RESTRICT x,
                                  void *const SFEM_RESTRICT y,
                                  void *stream);

int cu_macro_tet4_laplacian_diag(const ptrdiff_t nelements,
                                 const ptrdiff_t stride,  // Stride for elements and fff
                                 const idx_t *const SFEM_RESTRICT elements,
                                 const void *const SFEM_RESTRICT fff,
                                 const enum RealType real_type_xy,
                                 void *const SFEM_RESTRICT diag,
                                 void *stream);

// FIXME Possibly buggy
int cu_macro_tet4_laplacian_crs(const ptrdiff_t nelements,
                                const ptrdiff_t stride,  // Stride for elements and fff
                                const idx_t *const SFEM_RESTRICT elements,
                                const void *const SFEM_RESTRICT fff,
                                const count_t *const SFEM_RESTRICT rowptr,
                                const idx_t *const SFEM_RESTRICT colidx,
                                const enum RealType real_type,
                                void *const SFEM_RESTRICT values,
                                void *stream);

#ifdef __cplusplus
}

#endif
#endif  // CU_MACRO_TET4_LAPLACIAN_H
