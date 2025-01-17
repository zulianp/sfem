#include "sfem_base.h"

#ifndef SFEM_ENABLE_CUDA

#include <stdlib.h>
#include "coo_sort.h"
#include "sfem_config.h"

// Global variables needed for qsort...
real_t *global_weights;
idx_t *global_row_indices;
idx_t *global_col_indices;

int compare_weights(const void *a, const void *b);
int compare_indices(const void *a, const void *b);
void cycle_leader_swap(idx_t *rows, idx_t *cols, real_t *values, idx_t *indices, count_t N);

void sort_weights(idx_t *sort_indices, idx_t *row_indices, idx_t *col_indices, real_t *weights,
                  const count_t N) {
    global_weights = weights;

    for (idx_t i = 0; i < N; i++) {
        sort_indices[i] = i;
    }

    // TODO parallel sort is better, but with if built with cuda uses thrust for sorting
    qsort(sort_indices, N, sizeof(idx_t), compare_weights);
    cycle_leader_swap(row_indices, col_indices, weights, sort_indices, N);
}

void sort_rows_cols(idx_t *sort_indices, idx_t *row_indices, idx_t *col_indices, real_t *weights,
                    const count_t N) {
    // Assign the global pointers
    global_row_indices = row_indices;
    global_col_indices = col_indices;

    for (idx_t i = 0; i < N; i++) {
        sort_indices[i] = i;
    }

    // Sort the indices array
    qsort(sort_indices, N, sizeof(idx_t), compare_indices);
    cycle_leader_swap(row_indices, col_indices, weights, sort_indices, N);
}

int compare_weights(const void *a, const void *b) {
    idx_t idx_a = *(const idx_t *)a;
    idx_t idx_b = *(const idx_t *)b;
    if (global_weights[idx_a] < global_weights[idx_b]) return 1;
    if (global_weights[idx_a] > global_weights[idx_b]) return -1;
    return 0;
}

int compare_indices(const void *a, const void *b) {
    idx_t idx_a = *(const idx_t *)a;
    idx_t idx_b = *(const idx_t *)b;

    if (global_row_indices[idx_a] != global_row_indices[idx_b])
        return global_row_indices[idx_a] - global_row_indices[idx_b];
    else
        return global_col_indices[idx_a] - global_col_indices[idx_b];
}

void cycle_leader_swap(idx_t *rows, idx_t *cols, real_t *values, idx_t *indices, count_t N) {
    for (idx_t i = 0; i < N; i++) {
        idx_t current = i;
        while (indices[current] != i) {
            idx_t next = indices[current];

            idx_t temp_row = rows[current];
            rows[current] = rows[next];
            rows[next] = temp_row;

            idx_t temp_col = cols[current];
            cols[current] = cols[next];
            cols[next] = temp_col;

            real_t temp_val = values[current];
            values[current] = values[next];
            values[next] = temp_val;

            indices[current] = current;
            current = next;
        }
        indices[current] = current;
    }
}
#endif
