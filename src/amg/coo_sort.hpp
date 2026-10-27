#ifndef COO_SORT_H
#define COO_SORT_H

#include "sfem_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Ignores elements with negative row or column indices as way to filter
void sum_duplicates(count_t *sort_indices, idx_t *row_indices, idx_t *col_indices, real_t *weights,
                    count_t *N_ptr);

/** Generic sorting API over cuda and host. With cuda, `sort_indices` isn't needed and is ignored.
 * As implemented, these expect pointers to host arrays and they are copied to device for sorting.
 */
void sort_weights(count_t *sort_indices, idx_t *row_indices, idx_t *col_indices, real_t *weights,
                  const count_t N);
void sort_rows_cols(count_t *sort_indices, idx_t *row_indices, idx_t *col_indices, real_t *weights,
                    const count_t N);

#ifdef __cplusplus
}
#endif

#endif  // COO_SORT_H
