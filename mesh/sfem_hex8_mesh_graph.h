#ifndef SFEM_HEX8_MESH_GRAPH_H
#define SFEM_HEX8_MESH_GRAPH_H

#include <stddef.h>

#include "sfem_base.h"
#include "sfem_mesh.h"

#ifdef __cplusplus
extern "C" {
#endif

int hex8_build_edge_graph(const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const elems,
                          count_t **out_rowptr,
                          idx_t **out_colidx);

ptrdiff_t nxe_max_node_id(const ptrdiff_t nelements,
                          const int nxe,
                          idx_t **const SFEM_RESTRICT elements);

int proteus_hex8_create_full_idx(const int L,
                                 mesh_t *mesh,
                                 idx_t **elements,
                                 ptrdiff_t *n_unique_nodes_out,
                                 ptrdiff_t *interior_start_out);

int proteus_hex8_crs_graph(const int L,
                           const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const elements,
                           count_t **out_rowptr,
                           idx_t **out_colidx);

int sshex8_hierarchical_renumbering(const int       L,
                                    const int       nlevels,
                                    int *const      levels,
                                    const ptrdiff_t nelements,
                                    const ptrdiff_t nnodes,
                                    idx_t **const   elements);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_HEX8_MESH_GRAPH_H
