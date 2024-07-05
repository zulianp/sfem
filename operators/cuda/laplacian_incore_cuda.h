#ifndef LAPLACIAN_INCORE_CUDA_H
#define LAPLACIAN_INCORE_CUDA_H

#include "sfem_base.h"
#include "sfem_defs.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// int cuda_incore_laplacian_init(const enum ElemType element_type,
//                                cuda_incore_laplacian_t *ctx,
//                                const ptrdiff_t nelements,
//                                idx_t **const SFEM_RESTRICT elements,
//                                geom_t **const SFEM_RESTRICT points);

// int cuda_incore_laplacian_destroy(cuda_incore_laplacian_t *ctx);

int cu_laplacian_apply(const enum ElemType element_type,
                       const ptrdiff_t nelements,
                       const idx_t *const SFEM_RESTRICT elements,
                       const void *const SFEM_RESTRICT fff,
                       const enum RealType real_type_xy,
                       const void *const x,
                       void *const y,
                       void *stream);

int cu_laplacian_diag(const enum ElemType element_type,
                      const ptrdiff_t nelements,
                      const idx_t *const SFEM_RESTRICT elements,
                      const void *const SFEM_RESTRICT fff,
                      const enum RealType real_type_xy,
                      void *const diag,
                      void *stream);

#ifdef __cplusplus
}
#endif
#endif  // LAPLACIAN_INCORE_CUDA_H
