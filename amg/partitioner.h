#ifndef PARTITIONER_H
#define PARTITIONER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include "sfem_config.h"

typedef struct {
    // Array with length of fine mat.nrows
    // This is where the resulting partition is stored
    idx_t *partition;

    // Array with length of fine mat.nrows
    // Used for storing the rowsums of the strength of connection graph's
    // adjacency matrix
    real_t *rowsums;

    // Workspace for greedy matching, each must be the length of nnz above the
    // diagonal, i.e. length = (nnz - nrows) / 2
    count_t *sort_indices;  // Sorting workspace and used to track who has been paired
    idx_t *ptr_i;           // Row indices
    idx_t *ptr_j;           // Column indices
    real_t *weights;        // Non-zero values
} PartitionerWorkspace;

int partition(const mask_t *bdy_dofs, const real_t coarsening_factor, real_t *near_null,
              idx_t *offdiag_row_indices, idx_t *offdiag_col_indices, real_t *offdiag_values,
              count_t *offdiag_nnz, ptrdiff_t *ndofs, PartitionerWorkspace *ws);
#ifdef __cplusplus
}
#endif
#endif  // PARTITIONER_H
