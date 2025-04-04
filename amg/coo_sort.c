#include "coo_sort.h"
#include <assert.h>
#include <stdio.h>

void sum_duplicates(count_t *sort_indices, idx_t *row_indices, idx_t *col_indices, real_t *weights,
                    count_t *N_ptr) {
    count_t N = *N_ptr;

    sort_rows_cols(sort_indices, row_indices, col_indices, weights, N);

    for (idx_t k = 0; k < N - 1; k++) {
        // printf("%d %d %f\n", row_indices[k], col_indices[k], weights[k]);
        assert(row_indices[k] <= row_indices[k + 1]);
        if (row_indices[k] == row_indices[k + 1]) {
            assert(col_indices[k] <= col_indices[k + 1]);
        }
    }

    // Compact the arrays
    idx_t write_pos = 0;
    idx_t prev_row = SFEM_IDX_INVALID;
    idx_t prev_col = SFEM_IDX_INVALID;

    for (idx_t k = 0; k < N; k++) {
        idx_t i = row_indices[k];
        idx_t j = col_indices[k];
        real_t v = weights[k];

        if (i >= 0 && j >= 0) {
            if (i == prev_row && j == prev_col) {
                // Duplicate entry found; sum the values
                weights[write_pos - 1] += v;
            } else {
                // New unique entry; copy to the write position
                row_indices[write_pos] = i;
                col_indices[write_pos] = j;
                weights[write_pos] = v;
                write_pos++;
            }

            prev_row = i;
            prev_col = j;
        }
    }

    *N_ptr = write_pos;
}
