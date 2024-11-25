#include "sfem_base.h"

#ifndef SFEM_ENABLE_CUDA

#include <stdlib.h>
#include "coo_weight_sort.h"
#include "partitioner.h"
#include "sfem_config.h"

// Global variables needed for qsort...
real_t *global_weights;

int compare_indices(const void *a, const void *b);

void sort_weights(PartitionerWorkspace *ws, count_t nweights) {
    idx_t *indices = ws->sort_indices;
    idx_t *ptr_i = ws->ptr_i;
    idx_t *ptr_j = ws->ptr_j;
    real_t *weights = ws->weights;
    global_weights = weights;

    for (idx_t i = 0; i < nweights; i++) {
        indices[i] = i;
    }

    // TODO parallel sort is must here
    qsort(indices, nweights, sizeof(idx_t), compare_indices);
    cycle_leader_swap(ptr_i, ptr_j, weights, indices, nweights);
}

int compare_indices(const void *a, const void *b) {
    idx_t idx_a = *(const idx_t *)a;
    idx_t idx_b = *(const idx_t *)b;
    if (global_weights[idx_a] < global_weights[idx_b]) return 1;
    if (global_weights[idx_a] > global_weights[idx_b]) return -1;
    return 0;
}
#endif
