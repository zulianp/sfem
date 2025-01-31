#include <stddef.h>
#include <stdio.h>
#include "crs.h"
#include "sfem_base.h"
#include "sfem_mask.h"

int crs_write_mm(const count_t       ndofs,
                 const mask_t *const bdy_dofs,
                 const char *const   matfile,
                 const char *const   bdyfile,
                 count_t *const      rowptr,
                 idx_t *const        colidx,
                 real_t *const       values) {
    FILE *fptr;
    if (bdy_dofs && bdyfile) {
        fptr = fopen(bdyfile, "w");

        count_t bdy_dofs_count = 0;
        for (count_t i = 0; i < ndofs; i++) {
            if (mask_get(i, bdy_dofs)) {
                bdy_dofs_count++;
            }
        }

        fprintf(fptr, "%d", bdy_dofs_count);
        for (count_t i = 0; i < ndofs; i++) {
            if (mask_get(i, bdy_dofs)) {
                fprintf(fptr, "\n%d", i);
            }
        }
        fclose(fptr);
    }

    fptr        = fopen(matfile, "w");
    count_t nnz = rowptr[ndofs];
    fprintf(fptr, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(fptr, "%%Generated by sfem\n");
    fprintf(fptr, "%d %d %d", ndofs, ndofs, nnz);

    for (count_t i = 0; i < ndofs; i++) {
        count_t start = rowptr[i];
        count_t end   = rowptr[i + 1];
        for (count_t idx = start; idx < end; idx++) {
            idx_t  j   = colidx[idx];
            real_t val = values[idx];
            fprintf(fptr, "\n%d %d %f", i + 1, j + 1, val);
        }
    }

    fclose(fptr);
    return 0;
}
