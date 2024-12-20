#ifndef PARTITIONER_H
#define PARTITIONER_H

#include "sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

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
    idx_t *sort_indices;  // Sorting workspace and used to track who has been paired
    idx_t *ptr_i;         // Row indices
    idx_t *ptr_j;         // Column indices
    real_t *weights;      // Non-zero values
} PartitionerWorkspace;

int partition(real_t *near_null,
              const mask_t *bdy_dofs,
              const real_t coarsening_factor,
              SymmCOOMatrix *symm_coo,
              PartitionerWorkspace *ws);

#ifdef __cplusplus
}
#endif
#endif  // PARTITIONER_H
