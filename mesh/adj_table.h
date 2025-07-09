#ifndef ADJ_TABLE_H
#define ADJ_TABLE_H

#include "sfem_base.h"
#include "sfem_defs.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void fill_local_side_table(enum ElemType element_type, int *local_side_table);

void fill_element_adj_table(const ptrdiff_t                    n_elements,
                            const ptrdiff_t                    n_nodes,
                            enum ElemType                      element_type,
                            idx_t **const SFEM_RESTRICT        elems,
                            element_idx_t *const SFEM_RESTRICT table);

void create_element_adj_table_from_dual_graph(const ptrdiff_t                     n_elements,
                                              const ptrdiff_t                     n_nodes,
                                              enum ElemType                       element_type,
                                              idx_t **const SFEM_RESTRICT         elems,
                                              const count_t *const                adj_ptr,
                                              const element_idx_t *const          adj_idx,
                                              element_idx_t **const SFEM_RESTRICT table_out);

void create_element_adj_table_from_dual_graph_soa(const ptrdiff_t                     n_elements,
                                                  const ptrdiff_t                     n_nodes,
                                                  enum ElemType                       element_type,
                                                  idx_t **const SFEM_RESTRICT         elems,
                                                  const count_t *const                adj_ptr,
                                                  const element_idx_t *const          adj_idx,
                                                  element_idx_t **const SFEM_RESTRICT table);

void extract_surface_connectivity_with_adj_table(const ptrdiff_t             n_elements,
                                                 const ptrdiff_t             n_nodes,
                                                 const int                   element_type,
                                                 idx_t **const SFEM_RESTRICT elems,
                                                 ptrdiff_t                  *n_surf_elements,
                                                 idx_t                     **surf_elems,
                                                 element_idx_t             **parent_element);

void create_element_adj_table(const ptrdiff_t                     n_elements,
                              const ptrdiff_t                     n_nodes,
                              enum ElemType                       element_type,
                              idx_t **const SFEM_RESTRICT         elems,
                              element_idx_t **const SFEM_RESTRICT table_out);

int extract_skin_sideset(const ptrdiff_t               n_elements,
                         const ptrdiff_t               n_nodes,
                         const int                     element_type,
                         idx_t **const SFEM_RESTRICT   elems,
                         ptrdiff_t *SFEM_RESTRICT      n_surf_elements,
                         element_idx_t **SFEM_RESTRICT parent_element,
                         int16_t **SFEM_RESTRICT       side_idx);

int extract_surface_from_sideset(const int                                element_type,
                                 idx_t **const SFEM_RESTRICT              elems,
                                 const ptrdiff_t                          n_surf_elements,
                                 const element_idx_t *const SFEM_RESTRICT parent_element,
                                 const int16_t *const SFEM_RESTRICT       side_idx,
                                 idx_t **const SFEM_RESTRICT              sides);

int extract_nodeset_from_sideset(const int                                element_type,
                                 idx_t **const SFEM_RESTRICT              elems,
                                 const ptrdiff_t                          n_surf_elements,
                                 const element_idx_t *const SFEM_RESTRICT parent_element,
                                 const int16_t *const SFEM_RESTRICT       side_idx,
                                 ptrdiff_t                               *n_nodes_out,
                                 idx_t **SFEM_RESTRICT                    nodes_out);

int extract_nodeset_from_sidesets(uint16_t                                 n_sidesets,
                                  const enum ElemType                      element_type[],
                                  idx_t **const SFEM_RESTRICT              elems[],
                                  const ptrdiff_t                          n_surf_elements[],
                                  const element_idx_t *const SFEM_RESTRICT parent_element[],
                                  const int16_t *const SFEM_RESTRICT       side_idx[],
                                  ptrdiff_t                               *n_nodes_out,
                                  idx_t **SFEM_RESTRICT                    nodes_out);

int extract_sideset_from_adj_table(const enum ElemType                      element_type,
                                   const ptrdiff_t                          n_elements,
                                   const element_idx_t *const SFEM_RESTRICT table,
                                   ptrdiff_t *SFEM_RESTRICT                 n_surf_elements,
                                   element_idx_t **SFEM_RESTRICT            parent_element,
                                   int16_t **SFEM_RESTRICT                  side_idx);

#ifdef __cplusplus
}
#endif
#endif  // ADJ_TABLE_H
