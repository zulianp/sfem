#ifndef CRS_GRAPH_H
#define CRS_GRAPH_H

#include <stddef.h>

#include "sfem_base.h"

int build_crs_graph(const ptrdiff_t nelements,
                    const ptrdiff_t nnodes,
                    idx_t **const elems,
                    count_t **out_rowptr,
                    idx_t **out_colidx);

int build_crs_graph_3(const ptrdiff_t nelements,
                    const ptrdiff_t nnodes,
                    idx_t **const elems,
                    count_t **out_rowptr,
                    idx_t **out_colidx);

int block_crs_to_crs(const ptrdiff_t nnodes,
                     const int block_size,
                     const count_t *const block_rowptr,
                     const idx_t *const block_colidx,
                     const real_t *const block_values,
                     count_t *const rowptr,
                     idx_t *const colidx,
                     real_t *const values);

// for crs insertion
idx_t find_idx(const idx_t key, const idx_t *arr, idx_t size);
// idx_t find_idx_binary_search(const idx_t key, const idx_t *arr, idx_t size);

#endif
