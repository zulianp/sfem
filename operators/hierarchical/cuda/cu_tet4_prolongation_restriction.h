#ifndef CUT_TET4_PROLONGATION_RESTRICTION_H
#define CUT_TET4_PROLONGATION_RESTRICTION_H

#include <stddef.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_tet4_to_macro_tet4_prolongation(const ptrdiff_t coarse_nnodes,
                                       const count_t *const SFEM_RESTRICT coarse_rowptr,
                                       const idx_t *const SFEM_RESTRICT coarse_colidx,
                                       const idx_t *const SFEM_RESTRICT fine_node_map,
                                       const int vec_size,
                                       const enum RealType from_type,
                                       const real_t *const SFEM_RESTRICT from,
                                       const enum RealType to_type,
                                       real_t *const SFEM_RESTRICT to,
                                       void *stream);

int cu_macrotet4_to_tet4_restriction(
        const ptrdiff_t coarse_nnodes,
        const count_t *const SFEM_RESTRICT coarse_rowptr,
        const idx_t *const SFEM_RESTRICT coarse_colidx,
        const idx_t *const SFEM_RESTRICT fine_node_map,
        const int vec_size,
        const enum RealType from_type,
        const real_t *const SFEM_RESTRICT from,
        const enum RealType to_type,
        real_t *const SFEM_RESTRICT to,
        void *stream);

#ifdef __cplusplus
}
#endif

#endif  // CUT_TET4_PROLONGATION_RESTRICTION_H
