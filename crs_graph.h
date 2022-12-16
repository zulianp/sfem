#ifndef CRS_GRAPH_H
#define CRS_GRAPH_H

#include <stddef.h>

typedef int idx_t;

int build_crs_graph(
	const ptrdiff_t nelements,
	const ptrdiff_t nnodes,
	idx_t *const elems[4],
	idx_t **out_rowptr,
	idx_t **out_colidx
	);

#endif
