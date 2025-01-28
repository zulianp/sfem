#include <stddef.h>
#include <string.h>
#include "crs.h"
#include "sfem_base.h"

int crs_spmm(const count_t                      rows_a,
             const count_t                      cols_b,
             const count_t *const SFEM_RESTRICT rowptr_a,
             const idx_t *const SFEM_RESTRICT   colidx_a,
             const real_t *const SFEM_RESTRICT  values_a,
             const count_t *const SFEM_RESTRICT rowptr_b,
             const idx_t *const SFEM_RESTRICT   colidx_b,
             const real_t *const SFEM_RESTRICT  values_b,
             count_t                          **rowptr_c,  // [out]
             idx_t                            **colidx_c,  // [out]
             real_t                           **values_c   // [out]
) {
    *rowptr_c = (count_t *)malloc((rows_a + 1) * sizeof(count_t));
    if (!(*rowptr_c)) {
        return -1;
    }
    (*rowptr_c)[0] = 0;

    size_t capacity = 16 + (rowptr_a[rows_a] + rowptr_b[rowptr_b[0]]) / 2;
    *colidx_c       = (idx_t *)malloc(capacity * sizeof(idx_t));
    *values_c       = (real_t *)malloc(capacity * sizeof(real_t));
    if (!(*colidx_c) || !(*values_c)) {
        free(*rowptr_c);
        if (*colidx_c) free(*colidx_c);
        if (*values_c) free(*values_c);
        return -1;
    }

    size_t nnz_so_far = 0;

    // Temporary buffer to accumulate each row's partial sums.
    // Could improve this a lot by using hashmap instead
    real_t *partial_sum = (real_t *)calloc(cols_b, sizeof(real_t));
    if (!partial_sum) {
        free(*rowptr_c);
        free(*colidx_c);
        free(*values_c);
        return -1;
    }

    for (count_t ai = 0; ai < rows_a; ai++) {
        memset(partial_sum, 0, cols_b * sizeof(real_t));

        // could pragma omp one of these loops but everything will be fighting
        // for partial_sum syncronization... probably only want to make parallel
        // if using thread-local workspaces and then it gets complicated
        for (count_t idx_a = rowptr_a[ai]; idx_a < rowptr_a[ai + 1]; idx_a++) {
            idx_t  aj    = colidx_a[idx_a];
            real_t val_a = values_a[idx_a];

            for (count_t idx_b = rowptr_b[aj]; idx_b < rowptr_b[aj + 1]; idx_b++) {
                idx_t  bj    = colidx_b[idx_b];
                real_t val_b = values_b[idx_b];
                partial_sum[bj] += val_a * val_b;
            }
        }

        count_t row_nnz = 0;

        for (count_t col = 0; col < cols_b; col++) {
            real_t v = partial_sum[col];
            if (v != 0.0) {
                if (nnz_so_far + 1 > capacity) {
                    capacity  = (capacity * 3) / 2 + 16;
                    *colidx_c = (idx_t *)realloc(*colidx_c, capacity * sizeof(idx_t));
                    *values_c = (real_t *)realloc(*values_c, capacity * sizeof(real_t));
                    if (!(*colidx_c) || !(*values_c)) {
                        free(*rowptr_c);
                        free(partial_sum);
                        return -1;
                    }
                }

                (*colidx_c)[nnz_so_far] = col;
                (*values_c)[nnz_so_far] = v;
                ++nnz_so_far;
                ++row_nnz;
            }
        }

        (*rowptr_c)[ai + 1] = (*rowptr_c)[ai] + row_nnz;
    }

    free(partial_sum);

    if (nnz_so_far < capacity) {
        *colidx_c = (idx_t *)realloc(*colidx_c, nnz_so_far * sizeof(idx_t));
        *values_c = (real_t *)realloc(*values_c, nnz_so_far * sizeof(real_t));
    }

    return 0;
}
