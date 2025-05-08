#ifndef LAPLACIAN_INCORE_CUDA_H
#define LAPLACIAN_INCORE_CUDA_H

#include "sfem_base.h"
#include "sfem_defs.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int cu_laplacian_apply(const enum ElemType             element_type,
                       const ptrdiff_t                 nelements,
                       idx_t **const SFEM_RESTRICT     elements,
                       const ptrdiff_t                 fff_stride,
                       const void *const SFEM_RESTRICT fff,
                       const enum RealType             real_type_xy,
                       const void *const SFEM_RESTRICT x,
                       void *const SFEM_RESTRICT       y,
                       void                           *stream);

int cu_laplacian_diag(const enum ElemType             element_type,
                      const ptrdiff_t                 nelements,
                      idx_t **const SFEM_RESTRICT     elements,
                      const ptrdiff_t                 fff_stride,
                      const void *const SFEM_RESTRICT fff,
                      const enum RealType             real_type_xy,
                      void *const SFEM_RESTRICT       diag,
                      void                           *stream);

int cu_laplacian_crs(const enum ElemType                element_type,
                     const ptrdiff_t                    nelements,
                     idx_t **const SFEM_RESTRICT        elements,
                     const ptrdiff_t                    fff_stride,
                     const void *const SFEM_RESTRICT    fff,
                     const count_t *const SFEM_RESTRICT rowptr,
                     const idx_t *const SFEM_RESTRICT   colidx,
                     const enum RealType                real_type,
                     void *const SFEM_RESTRICT          values,
                     void                              *stream);

int cu_laplacian_crs_sym(const enum ElemType                element_type,
                         const ptrdiff_t                    nelements,
                         idx_t **const SFEM_RESTRICT        elements,
                         const ptrdiff_t                    fff_stride,
                         const void *const SFEM_RESTRICT    fff,
                         const count_t *const SFEM_RESTRICT rowptr,
                         const idx_t *const SFEM_RESTRICT   colidx,
                         const enum RealType                real_type,
                         void *const SFEM_RESTRICT          diag,
                         void *const SFEM_RESTRICT          offdiag,
                         void                              *stream);

#ifdef __cplusplus
}
#endif
#endif  // LAPLACIAN_INCORE_CUDA_H
