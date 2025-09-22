#ifndef CRS_GRAPH_H
#define CRS_GRAPH_H

#include <stddef.h>

#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int build_n2e(const ptrdiff_t nelements,
              const ptrdiff_t nnodes,
              const int nnodesxelem,
              idx_t **const elems,
              count_t **out_n2eptr,
              element_idx_t **out_elindex);

int build_n2ln(const ptrdiff_t nelements,
              const ptrdiff_t nnodes,
              const int nnodesxelem,
              idx_t **const elems,
              count_t **out_n2ln_ptr,
              count_t **out_ln_index);

int build_crs_graph_for_elem_type(const int element_type,
                                  const ptrdiff_t nelements,
                                  const ptrdiff_t nnodes,
                                  idx_t **const elems,
                                  count_t **out_rowptr,
                                  idx_t **out_colidx);

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

int crs_to_coo(const ptrdiff_t n, const count_t *const rowptr, idx_t *const row_idx);
int sorted_coo_to_crs(const count_t nnz, const idx_t *const row_idx, const ptrdiff_t n, count_t *const rowptr);


// for crs insertion
idx_t find_idx(const idx_t key, const idx_t *arr, idx_t size);
// idx_t find_idx_binary_search(const idx_t key, const idx_t *arr, idx_t size);

int create_dual_graph(const ptrdiff_t n_elements,
                      const ptrdiff_t n_nodes,
                      const int element_type,
                      idx_t **const elems,
                      count_t **out_rowptr,
                      element_idx_t **out_colidx);

int crs_graph_block_to_scalar(const ptrdiff_t nnodes,
                              const int block_size,
                              const count_t *const block_rowptr,
                              const idx_t *const block_colidx,
                              count_t *const rowptr,
                              idx_t *const colidx);

int build_crs_graph_from_element(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 int nxe,
                                 idx_t **const elems,
                                 count_t **out_rowptr,
                                 idx_t **out_colidx);

int build_crs_graph_upper_triangular_from_element(const ptrdiff_t nelements,
                                                  const ptrdiff_t nnodes,
                                                  int nxe,
                                                  idx_t **const elems,
                                                  count_t **out_rowptr,
                                                  idx_t **out_colidx);

#ifdef __cplusplus
}
#endif
#endif
