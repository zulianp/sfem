#ifndef SFEM_PROLONGATION_RESTRICTION_H
#define SFEM_PROLONGATION_RESTRICTION_H

#include <stddef.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

ptrdiff_t max_node_id(const enum ElemType type,
                      const ptrdiff_t nelements,
                      idx_t **const SFEM_RESTRICT elements);

int hierarchical_create_coarse_indices(const idx_t max_coarse_idx,
                                       const ptrdiff_t n_indices,
                                       idx_t *const SFEM_RESTRICT fine_indices,
                                       ptrdiff_t *n_coarse_indices,
                                       idx_t **SFEM_RESTRICT coarse_indices);

int hierarchical_collect_coarse_values(const idx_t max_coarse_idx,
                                       const ptrdiff_t n_indices,
                                       idx_t *const SFEM_RESTRICT fine_indices,
                                       const real_t *const SFEM_RESTRICT fine_values,
                                       real_t *const SFEM_RESTRICT coarse_values);

int hierarchical_prolongation(const enum ElemType from_element,
                              const enum ElemType to_element,
                              const ptrdiff_t nelements,
                              idx_t **const SFEM_RESTRICT elements,
                              const int vec_size,
                              const real_t *const SFEM_RESTRICT from,
                              real_t *const SFEM_RESTRICT to);

int hierarchical_restriction_with_counting(
                              const enum ElemType from_element,
                              const enum ElemType to_element,
                              const ptrdiff_t nelements,
                              idx_t **const SFEM_RESTRICT elements,
                              const uint16_t *const SFEM_RESTRICT element_to_node_incidence_count,
                              const int vec_size,
                              const real_t *const SFEM_RESTRICT from,
                              real_t *const SFEM_RESTRICT to);

int hierarchical_restriction(
        // CRS-node-graph of the coarse mesh
        const ptrdiff_t nnodes,
        const count_t *const SFEM_RESTRICT coarse_rowptr,
        const idx_t *const SFEM_RESTRICT coarse_colidx,
        const int vec_size,
        const real_t *const SFEM_RESTRICT from,
        real_t *const SFEM_RESTRICT to);

// Edge-map versions
int build_p1_to_p2_edge_map(const ptrdiff_t nnodes,
                            const count_t *const SFEM_RESTRICT coarse_rowptr,
                            const idx_t *const SFEM_RESTRICT coarse_colidx,
                            idx_t *const SFEM_RESTRICT p2_vertices);

int hierarchical_prolongation_with_edge_map(const ptrdiff_t nnodes,
                                            const count_t *const SFEM_RESTRICT coarse_rowptr,
                                            const idx_t *const SFEM_RESTRICT coarse_colidx,
                                            const idx_t *const SFEM_RESTRICT p2_vertices,
                                            const int vec_size,
                                            const real_t *const SFEM_RESTRICT from,
                                            real_t *const SFEM_RESTRICT to);

int hierarchical_restriction_with_edge_map(
        // CRS-node-graph of the coarse mesh
        const ptrdiff_t nnodes,
        const count_t *const SFEM_RESTRICT coarse_rowptr,
        const idx_t *const SFEM_RESTRICT coarse_colidx,
        const idx_t *const SFEM_RESTRICT p2_vertices,
        const int vec_size,
        const real_t *const SFEM_RESTRICT from,
        real_t *const SFEM_RESTRICT to);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_PROLONGATION_RESTRICTION_H
