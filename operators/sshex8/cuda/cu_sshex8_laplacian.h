#ifndef CU_SSHEX8_LAPLACIAN_H
#define CU_SSHEX8_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_affine_sshex8_laplacian_apply(const int                       level,
                                     const ptrdiff_t                 nelements,
                                     idx_t **const SFEM_RESTRICT     elements,
                                     const ptrdiff_t                 fff_stride,
                                     const void *const SFEM_RESTRICT fff,
                                     const enum RealType             real_type_xy,
                                     const void *const SFEM_RESTRICT x,
                                     void *const SFEM_RESTRICT       y,
                                     void                           *stream);

int cu_affine_sshex8_laplacian_diag(const int                       level,
                                    const ptrdiff_t                 nelements,
                                    idx_t **const SFEM_RESTRICT     elements,
                                    const ptrdiff_t                 fff_stride,
                                    const void *const SFEM_RESTRICT fff,
                                    const enum RealType             real_type_out,
                                    void *const SFEM_RESTRICT       out,
                                    void                           *stream);

#ifdef __cplusplus
}
#endif
#endif  // CU_SSHEX8_LAPLACIAN_H
