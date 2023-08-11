#ifndef ADJ_TABLE_H
#define ADJ_TABLE_H

#include "sfem_base.h"
#include "sfem_defs.h"


void fill_element_adj_table(const ptrdiff_t n_elements,
							const ptrdiff_t n_nodes,
							enum ElemType element_type,
                            idx_t** const SFEM_RESTRICT elems,
                            ptrdiff_t* const SFEM_RESTRICT table);

void extract_surface_connectivity_with_adj_table(const ptrdiff_t n_elements,
                                                 const ptrdiff_t n_nodes,
                                                 const int element_type,
                                                 idx_t **const SFEM_RESTRICT elems,
                                                 ptrdiff_t *n_surf_elements,
                                                 idx_t **surf_elems,
                                                 element_idx_t **parent_element);

#endif  // ADJ_TABLE_H
