#ifndef TET10_CUDA_INCORE_LAPLACIAN_H
#define TET10_CUDA_INCORE_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"

#include "cu_tet4_laplacian.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_tet10_laplacian_apply(const ptrdiff_t nelements,
                             const idx_t *const SFEM_RESTRICT elements,
                             const void *const SFEM_RESTRICT fff,
                             const enum RealType real_type_xy,
                             const void *const SFEM_RESTRICT x,
                             void *const SFEM_RESTRICT y,
                             void *stream);

#ifdef __cplusplus
}
#endif
#endif  // TET10_CUDA_INCORE_LAPLACIAN_H
