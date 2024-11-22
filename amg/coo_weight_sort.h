#ifndef COO_WEIGHT_SORT_H
#define COO_WEIGHT_SORT_H

#include "partitioner.h"
#include "sfem_config.h"

void sort_weights(PartitionerWorkspace* ws, count_t nweights);

#ifdef SFEM_ENABLE_CUDA
/**
 * Sorts the sparse matrix represented by col_indices, row_indices, and values
 * in-place based on values in descending order using CUDA and thrust::device_vector.
 *
 * @param col_indices Pointer to the column indices array.
 * @param row_indices Pointer to the row indices array.
 * @param values      Pointer to the values array.
 * @param N           The number of elements in the arrays.
 */
void sort_sparse_matrix_cuda(idx_t* col_indices, idx_t* row_indices, real_t* values, count_t N);
#endif

#endif  // COO_WEIGHT_SORT_H
