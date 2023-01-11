#ifndef CRS_GRAPH_H
#define CRS_GRAPH_H

#include <stddef.h>

#include "sfem_base.h"

int build_crs_graph(
	const ptrdiff_t nelements,
	const ptrdiff_t nnodes,
	idx_t *const elems[4],
	idx_t **out_rowptr,
	idx_t **out_colidx
	);

// for crs insertion
idx_t find_idx(const idx_t key, const idx_t *restrict arr, idx_t size);
// idx_t find_idx_binary_search(const idx_t key, const idx_t *arr, idx_t size);

#endif
