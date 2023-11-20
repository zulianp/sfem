#ifndef SFEM_EXTRACT_SHARP_FEATURES_H
#define SFEM_EXTRACT_SHARP_FEATURES_H

#include <stddef.h>

#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int extract_sharp_edges(const enum ElemType element_type,
                        const ptrdiff_t nelements,
                        const ptrdiff_t nnodes,
                        idx_t **const SFEM_RESTRICT elements,
                        geom_t **const SFEM_RESTRICT points,
                        // CRS-graph (node to node)
                        const count_t *const SFEM_RESTRICT rowptr,
                        const idx_t *const SFEM_RESTRICT colidx,
                        const geom_t angle_threshold,
                        ptrdiff_t *out_n_sharp_edges,
                        count_t **out_e0,
                        count_t **out_e1);

int extract_sharp_corners(const ptrdiff_t nnodes,
                          const ptrdiff_t n_sharp_edges,
                          const count_t *const SFEM_RESTRICT e0,
                          const count_t *const SFEM_RESTRICT e1,
                          ptrdiff_t *out_ncorners,
                          idx_t **out_corners);

int extract_disconnected_faces(const enum ElemType element_type,
                               const ptrdiff_t nelements,
                               const ptrdiff_t nnodes,
                               idx_t **const SFEM_RESTRICT elements,
                               const ptrdiff_t n_sharp_edges,
                               const count_t *const SFEM_RESTRICT e0,
                               const count_t *const SFEM_RESTRICT e1,
                               ptrdiff_t *out_n_disconnected_elements,
                               element_idx_t **out_disconnected_elements);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_EXTRACT_SHARP_FEATURES_H
