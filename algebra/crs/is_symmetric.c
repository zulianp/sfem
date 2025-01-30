#include <math.h>
#include <stdio.h>
#include <sys/stat.h>
#include "crs.h"
#include "sfem_base.h"

int crs_is_symmetric(const count_t                      rows,
                     const count_t *const SFEM_RESTRICT rowptr,
                     const idx_t *const SFEM_RESTRICT   colidx,
                     const real_t *const SFEM_RESTRICT  values) {
    int invalid = crs_validate(rows, rows, rowptr, colidx, values);
    if (invalid) {
        printf("can't be symmetric if failed crs validation\n");
        return 0;
    }

    count_t        nnz      = rowptr[rows];
    count_t *const rowptr_t = (count_t *)malloc((rows + 1) * sizeof(count_t));
    idx_t *const   colidx_t = (idx_t *)malloc(nnz * sizeof(idx_t));
    real_t *const  values_t = (real_t *)malloc(nnz * sizeof(real_t));

    crs_transpose(rows, rows, rowptr, colidx, values, rowptr_t, colidx_t, values_t);
    // int violations = 0;
    // printf("nnz: %d\n", nnz);

    for (count_t i = 0; i <= rows; i++) {
        if (rowptr[i] != rowptr_t[i]) {
            // printf("rowptr[%d] = %d, rowptr_t[%d] = %d\n", i, rowptr[i], i, rowptr_t[i]);
            // violations += 1;
            free(rowptr_t);
            free(colidx_t);
            free(values_t);
            return 0;
        }
    }
    for (count_t i = 0; i < nnz; i++) {
        if (colidx[i] != colidx_t[i]) {
            // printf("colidx[%d] = %d, colidx_t[%d] = %d\n", i, colidx[i], i, colidx_t[i]);
            // violations += 1;
            free(rowptr_t);
            free(colidx_t);
            free(values_t);
            return 0;
        }

        real_t err = fabs(values[i] - values_t[i]);
        if (err > 1e-12) {
            // printf("|values[%d] - values_t[%d]| = %.2f\n", i, i, err);
            // violations += 1;
            free(rowptr_t);
            free(colidx_t);
            free(values_t);
            return 0;
        }
    }

    /*
    if (violations > 0) {
        printf("%d violations\n", violations);
        return 0;
    }
    */

    free(rowptr_t);
    free(colidx_t);
    free(values_t);
    return 1;
}
