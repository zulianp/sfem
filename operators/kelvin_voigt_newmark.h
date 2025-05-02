#ifndef KELVIN_VOIGT_NEWMARK
#define KELVIN_VOIGT_NEWMARK

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// HAOYU

int kelvin_voigt_newmark_apply_aos(const enum ElemType               element_type,
                                   const ptrdiff_t                   nelements,
                                   const ptrdiff_t                   nnodes,
                                   idx_t **const SFEM_RESTRICT       elements,
                                   geom_t **const SFEM_RESTRICT      points,
                                   const real_t *const SFEM_RESTRICT u,
                                   real_t *const SFEM_RESTRICT       values);

#ifdef __cplusplus
}
#endif

#endif  // KELVIN_VOIGT_NEWMARK
