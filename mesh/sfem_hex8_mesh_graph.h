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
                          idx_t **const   elems,
                          count_t       **out_rowptr,
                          idx_t         **out_colidx);

ptrdiff_t nxe_max_node_id(const ptrdiff_t nelements, const int nxe, idx_t **const SFEM_RESTRICT elements);

int sshex8_generate_elements(const int       L,
                             const ptrdiff_t m_nelements,
                             const ptrdiff_t m_nnodes,
                             idx_t **const   m_elements,
                             idx_t         **elements,
                             ptrdiff_t      *n_unique_nodes_out,
                             ptrdiff_t      *interior_start_out);

int sshex8_crs_graph(const int       L,
                     const ptrdiff_t nelements,
                     const ptrdiff_t nnodes,
                     idx_t **const   elements,
                     count_t       **out_rowptr,
                     idx_t         **out_colidx);

int sshex8_hierarchical_renumbering(const int       L,
                                    const int       nlevels,
                                    int *const      levels,
                                    const ptrdiff_t nelements,
                                    const ptrdiff_t nnodes,
                                    idx_t **const   elements);

int  sshex8_hierarchical_n_levels(const int L);
void sshex8_hierarchical_mesh_levels(const int L, const int nlevels, int *const levels);

int sshex8_extract_surface_from_sideset(const int                                L,
                                        idx_t **const SFEM_RESTRICT              elems,
                                        const ptrdiff_t                          n_surf_elements,
                                        const element_idx_t *const SFEM_RESTRICT parent_element,
                                        const int16_t *const SFEM_RESTRICT       side_idx,
                                        idx_t **const SFEM_RESTRICT              sides);

int sshex8_extract_nodeset_from_sideset(const int                                L,
                                        idx_t **const SFEM_RESTRICT              elems,
                                        const ptrdiff_t                          n_surf_elements,
                                        const element_idx_t *const SFEM_RESTRICT parent_element,
                                        const int16_t *const SFEM_RESTRICT       side_idx,
                                        ptrdiff_t *n_nodes_out,
                                        idx_t **SFEM_RESTRICT                    nodes_out);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_HEX8_MESH_GRAPH_H
