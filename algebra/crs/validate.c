#include <math.h>
#include <stdio.h>
#include <sys/stat.h>
#include "crs.h"
#include "sfem_base.h"

int crs_validate(const count_t                      rows,
                 const count_t                      cols,
                 const count_t *const SFEM_RESTRICT rowptr,
                 const idx_t *const SFEM_RESTRICT   colidx,
                 const real_t *const SFEM_RESTRICT  values) {
    count_t nnz     = rowptr[rows];
    int     invalid = 0;
    // I'd like to verify sizes but I guess that's hard for heap allocated stuff...
    /*
    size_t  rowptr_size = sizeof(rowptr) / sizeof(count_t);
    size_t  colidx_size = sizeof(colidx) / sizeof(idx_t);
    size_t  values_size = sizeof(values) / sizeof(real_t);
    if (nnz > colidx_size) {
        printf("colidx array too small, size: %d nnz: %d\n", (count_t)colidx_size, nnz);
        invalid += 1;
    }
    if (nnz > values_size) {
        printf("values array too small\n");
        printf("values array too small, size: %d nnz: %d\n", (count_t)values_size, nnz);
        invalid += 1;
    }
    */
    if (rowptr[0]) {
        printf("rowptr first element must be 0, rowptr[0] = %d\n", rowptr[0]);
        invalid += 1;
    }
    for (count_t i = 0; i < rows; i++) {
        if (rowptr[i + 1] < rowptr[i]) {
            printf("rowptr[%d] = %d < rowptr[%d] = %d\n", i + 1, rowptr[i + 1], i, rowptr[i]);
            invalid += 1;
        }
    }

    for (count_t i = 0; i < nnz; i++) {
        if (colidx[i] < 0 || colidx[i] >= cols) {
            printf("colidx[%d] is %d for matrix with %d cols\n", i, colidx[i], cols);
            invalid += 1;
        }
        if (isnan(values[i])) {
            printf("values[%d] is NaN\n", i);
            invalid += 1;
        }
    }

    if (invalid > 0) {
        printf("%d violations...\n", invalid);
    }

    return invalid;
}
