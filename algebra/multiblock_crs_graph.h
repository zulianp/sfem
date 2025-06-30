#ifndef MULTIBLOCK_CRS_GRAPH_H
#define MULTIBLOCK_CRS_GRAPH_H

#include "sfem_defs.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int build_multiblock_n2e(const uint16_t      n_blocks,
                         const enum ElemType element_types[],
                         const ptrdiff_t     n_elements[],
                         idx_t **const       elements[],
                         const ptrdiff_t     n_nodes,
                         uint16_t          **out_block_number,
                         count_t           **out_n2eptr,
                         element_idx_t     **out_elindex);

int build_multiblock_crs_graph_from_n2e(const uint16_t                           n_blocks,
                                         const enum ElemType                      element_types[],
                                         const ptrdiff_t                          n_elements[],
                                         const ptrdiff_t                          n_nodes,
                                         idx_t **const SFEM_RESTRICT              elems[],
                                         const count_t *const SFEM_RESTRICT       n2eptr,
                                         const element_idx_t *const SFEM_RESTRICT elindex,
                                         const uint16_t *const SFEM_RESTRICT      block_number,
                                         count_t                                **out_rowptr,
                                         idx_t                                  **out_colidx);

int build_multiblock_crs_graph(const uint16_t      n_blocks,
                               const enum ElemType element_types[],
                               const ptrdiff_t     n_elements[],
                               idx_t **const       elems[],
                               const ptrdiff_t     n_nodes,
                               count_t           **out_rowptr,
                               idx_t             **out_colidx);

int build_multiblock_crs_graph_upper_triangular(const uint16_t      n_blocks,
                                                const enum ElemType element_types[],
                                                const ptrdiff_t     n_elements[],
                                                idx_t **const       elems[],
                                                const ptrdiff_t     n_nodes,
                                                count_t           **out_rowptr,
                                                idx_t             **out_colidx);

#ifdef __cplusplus
}
#endif

#endif /* MULTIBLOCK_CRS_GRAPH_H */