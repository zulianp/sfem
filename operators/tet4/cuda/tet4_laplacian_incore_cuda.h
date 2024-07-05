#ifndef TET4_CUDA_INCORE_LAPLACIAN_H
#define TET4_CUDA_INCORE_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_tet4_laplacian_apply(const ptrdiff_t nelements,
                            const idx_t *const SFEM_RESTRICT elements,
                            const void *const SFEM_RESTRICT fff,
                            const enum RealType real_type_xy,
                            const void *const x,
                            void *const y,
                            void *stream);

int cu_tet4_laplacian_diag(const ptrdiff_t nelements,
                      const idx_t *const SFEM_RESTRICT elements,
                      const void *const SFEM_RESTRICT fff,
                      const enum RealType real_type_xy,
                      void *const diag,
                      void *stream);

#ifdef __cplusplus
}
#endif
#endif  // TET4_CUDA_INCORE_LAPLACIAN_H
