#ifndef CU_SSHEX8_LAPLACIAN_H
#define CU_SSHEX8_LAPLACIAN_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_affine_sshex8_laplacian_tc_allocate_macro_ops(const ptrdiff_t nelements,
                                                           const enum RealType real_type,
                                                           void **mem);

int cu_affine_sshex8_laplacian_tc_fill_ops(const int level,
                                                 const ptrdiff_t nelements,
                                                 const ptrdiff_t stride,
                                                 const ptrdiff_t interior_start,
                                                 const idx_t *const SFEM_RESTRICT elements,
                                                 const void *const SFEM_RESTRICT fff,
                                                 const enum RealType real_type,
                                                 void *const SFEM_RESTRICT macro_element_ops, void *stream);

int cu_affine_sshex8_laplacian_tc_apply(
        const int level,
        const ptrdiff_t nelements,
        const ptrdiff_t stride,  // Stride for elements and fff
        const ptrdiff_t interior_start,
        const idx_t *const SFEM_RESTRICT elements,
        const enum RealType real_type,
        const void *const SFEM_RESTRICT macro_element_ops,
        const void *const SFEM_RESTRICT x,
        void *const SFEM_RESTRICT y,
        void *stream);

#ifdef __cplusplus
}
#endif
#endif  // CU_SSHEX8_LAPLACIAN_H
