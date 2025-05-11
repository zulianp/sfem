#ifndef CUT_TET4_PROLONGATION_RESTRICTION_H
#define CUT_TET4_PROLONGATION_RESTRICTION_H

#include <stddef.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_tet4_to_macrotet4_prolongation(const ptrdiff_t                    coarse_nnodes,
                                      const count_t *const SFEM_RESTRICT coarse_rowptr,
                                      const idx_t *const SFEM_RESTRICT   coarse_colidx,
                                      const idx_t *const SFEM_RESTRICT   fine_node_map,
                                      const int                          vec_size,
                                      const enum RealType                from_type,
                                      const void *const SFEM_RESTRICT    from,
                                      const enum RealType                to_type,
                                      void *const SFEM_RESTRICT          to,
                                      void                              *stream);

int cu_macrotet4_to_tet4_restriction(const ptrdiff_t                    coarse_nnodes,
                                     const count_t *const SFEM_RESTRICT coarse_rowptr,
                                     const idx_t *const SFEM_RESTRICT   coarse_colidx,
                                     const idx_t *const SFEM_RESTRICT   fine_node_map,
                                     const int                          vec_size,
                                     const enum RealType                from_type,
                                     const void *const SFEM_RESTRICT    from,
                                     const enum RealType                to_type,
                                     void *const SFEM_RESTRICT          to,
                                     void                              *stream);

// Element-based (more generic)

int cu_macrotet4_to_tet4_prolongation_element_based(const ptrdiff_t                 nelements,
                                                    idx_t **const SFEM_RESTRICT     elements,
                                                    const int                       vec_size,
                                                    const enum RealType             from_type,
                                                    const ptrdiff_t                 from_stride,
                                                    const void *const SFEM_RESTRICT from,
                                                    const enum RealType             to_type,
                                                    const ptrdiff_t                 to_stride,
                                                    void *const SFEM_RESTRICT       to,
                                                    void                           *stream);

int cu_macrotet4_to_tet4_restriction_element_based(const ptrdiff_t                     nelements,
                                                   idx_t **const SFEM_RESTRICT         elements,
                                                   const uint16_t *const SFEM_RESTRICT element_to_node_incidence_count,
                                                   const int                           vec_size,
                                                   const enum RealType                 from_type,
                                                   const ptrdiff_t                     from_stride,
                                                   const void *const SFEM_RESTRICT     from,
                                                   const enum RealType                 to_type,
                                                   const ptrdiff_t                     to_stride,
                                                   void *const SFEM_RESTRICT           to,
                                                   void                               *stream);

#ifdef __cplusplus
}
#endif

#endif  // CUT_TET4_PROLONGATION_RESTRICTION_H
