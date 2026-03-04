#ifndef TET4_PATCH_GRADIENT_H
#define TET4_PATCH_GRADIENT_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int tet4_patch_gradient(const ptrdiff_t                                        nelements,
                        const idx_t* const SFEM_RESTRICT* const SFEM_RESTRICT  elements,
                        const ptrdiff_t                                        nnodes,
                        const ptrdiff_t                                        max_indicence,
                        const count_t* const SFEM_RESTRICT                     n2e_ptr,
                        const element_idx_t* const SFEM_RESTRICT               n2e_idx,
                        const geom_t* const SFEM_RESTRICT* const SFEM_RESTRICT points,
                        const real_t* const SFEM_RESTRICT                      in,
                        const ptrdiff_t                                        out_stride,
                        real_t* const SFEM_RESTRICT                            outx,
                        real_t* const SFEM_RESTRICT                            outy,
                        real_t* const SFEM_RESTRICT                            outz);

#ifdef __cplusplus
}
#endif

#endif  // TET4_PATCH_GRADIENT_H
