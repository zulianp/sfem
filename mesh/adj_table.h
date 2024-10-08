#ifndef ADJ_TABLE_H
#define ADJ_TABLE_H

#include "sfem_base.h"
#include "sfem_defs.h"


#ifdef __cplusplus
extern "C" {
#endif


void fill_local_side_table(enum ElemType element_type, int *local_side_table);

void fill_element_adj_table(const ptrdiff_t n_elements,
                            const ptrdiff_t n_nodes,
                            enum ElemType element_type,
                            idx_t **const SFEM_RESTRICT elems,
                            element_idx_t *const SFEM_RESTRICT table);

void create_element_adj_table_from_dual_graph(const ptrdiff_t n_elements,
                                              const ptrdiff_t n_nodes,
                                              enum ElemType element_type,
                                              idx_t **const SFEM_RESTRICT elems,
                                              const count_t *const adj_ptr,
                                              const element_idx_t *const adj_idx,
                                              element_idx_t **const SFEM_RESTRICT table_out);

void create_element_adj_table_from_dual_graph_soa(const ptrdiff_t n_elements,
                            const ptrdiff_t n_nodes,
                            enum ElemType element_type,
                            idx_t **const SFEM_RESTRICT elems,
                            const count_t *const adj_ptr,
                            const element_idx_t *const adj_idx,
                            element_idx_t **const SFEM_RESTRICT table);

void extract_surface_connectivity_with_adj_table(const ptrdiff_t n_elements,
                                                 const ptrdiff_t n_nodes,
                                                 const int element_type,
                                                 idx_t **const SFEM_RESTRICT elems,
                                                 ptrdiff_t *n_surf_elements,
                                                 idx_t **surf_elems,
                                                 element_idx_t **parent_element);

void create_element_adj_table(const ptrdiff_t n_elements,
                              const ptrdiff_t n_nodes,
                              enum ElemType element_type,
                              idx_t **const SFEM_RESTRICT elems,
                              element_idx_t **const SFEM_RESTRICT table_out);


#ifdef __cplusplus
}
#endif
#endif  // ADJ_TABLE_H
