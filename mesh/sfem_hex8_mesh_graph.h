#ifndef SFEM_HEX8_MESH_GRAPH_H
#define SFEM_HEX8_MESH_GRAPH_H

#include <stddef.h>

#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int hex8_build_edge_graph(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const elems,
                          count_t **out_rowptr,
                          idx_t **out_colidx);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_HEX8_MESH_GRAPH_H
