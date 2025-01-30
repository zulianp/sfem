#include <stddef.h>
#include "crs.h"
#include "sfem_base.h"

int crs_transpose(const count_t        rows,
                  const count_t        cols,
                  const count_t *const rowptr,
                  const idx_t *const   colidx,
                  const real_t *const  values,
                  count_t *const       rowptr_t,  // out
                  idx_t *const         colidx_t,  // out
                  real_t *const        values_t   // out
) {
    count_t nnz = rowptr[rows];
    for (int i = 0; i <= cols; ++i) {
        rowptr_t[i] = 0;
    }

    for (int i = 0; i < nnz; ++i) {
        int col = colidx[i];
        rowptr_t[col + 1]++;
    }

    for (int i = 1; i <= cols; ++i) {
        rowptr_t[i] += rowptr_t[i - 1];
    }

    // We'll use an array 'offset' to track the next free slot in each row of A^T
    // as we populate colidx_t and values_t
    count_t *offset = (count_t *)malloc(cols * sizeof(count_t));

    // Initialize offset[] from rowptr_t
    for (count_t c = 0; c < cols; c++) {
        offset[c] = rowptr_t[c];
    }

    for (count_t i = 0; i < rows; i++) {
        for (count_t j = rowptr[i]; j < rowptr[i + 1]; j++) {
            idx_t  c = colidx[j];
            real_t v = values[j];

            count_t pos   = offset[c];
            colidx_t[pos] = i;
            values_t[pos] = v;
            offset[c]++;
        }
    }

    free(offset);
    return 0;
}
