#ifndef CU_PROTEUS_TET4_LAPLACIAN_H
#define CU_PROTEUS_TET4_LAPLACIAN_H

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_proteus_tet4_laplacian_apply(const int level,
                                    const ptrdiff_t nelements,
                                    const ptrdiff_t stride,
                                    const void *const SFEM_RESTRICT fff,
                                    const enum RealType real_type_xy,
                                    const void *const SFEM_RESTRICT x,
                                    void *const SFEM_RESTRICT y,
                                    void *stream);

#ifdef __cplusplus
}

#endif
#endif  // CU_PROTEUS_TET4_LAPLACIAN_H
