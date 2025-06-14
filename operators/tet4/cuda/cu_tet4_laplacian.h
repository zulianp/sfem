#ifndef TET4_CUDA_INCORE_LAPLACIAN_H
#define TET4_CUDA_INCORE_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_tet4_laplacian_apply(const ptrdiff_t                 nelements,
                            idx_t **const SFEM_RESTRICT     elements,
                            const ptrdiff_t                 fff_stride,
                            const void *const SFEM_RESTRICT fff,
                            const enum RealType             real_type_xy,
                            const void *const SFEM_RESTRICT x,
                            void *const SFEM_RESTRICT       y,
                            void                           *stream);

int cu_tet4_laplacian_diag(const ptrdiff_t                 nelements,
                           idx_t **const SFEM_RESTRICT     elements,
                           const ptrdiff_t                 fff_stride,
                           const void *const SFEM_RESTRICT fff,
                           const enum RealType             real_type_diag,
                           void *const SFEM_RESTRICT       diag,
                           void                           *stream);

int cu_tet4_laplacian_crs(const ptrdiff_t                    nelements,
                          idx_t **const SFEM_RESTRICT        elements,
                          const ptrdiff_t                    fff_stride,
                          const void *const SFEM_RESTRICT    fff,
                          const count_t *const SFEM_RESTRICT rowptr,
                          const idx_t *const SFEM_RESTRICT   colidx,
                          const enum RealType                real_type,
                          void *const SFEM_RESTRICT          values,
                          void                              *stream);

#ifdef __cplusplus
}
#endif
#endif  // TET4_CUDA_INCORE_LAPLACIAN_H
