
#include <stddef.h>
#include "sfem_base.h"

int scal(const ptrdiff_t nnodes, const real_t scale_factor, real_t *x) {
#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            x[i] *= scale_factor;
        }
    }
    return 0;
}

int crs_spmv(const ptrdiff_t nnodes,
             const count_t *const SFEM_RESTRICT rowptr,
             const idx_t *const SFEM_RESTRICT colidx,
             const real_t *const SFEM_RESTRICT values,
             const real_t *const SFEM_RESTRICT in,
             real_t *const SFEM_RESTRICT out) {
#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            const count_t row_begin = rowptr[i];
            const count_t row_end = rowptr[i + 1];

            real_t val = 0;
            for (count_t k = row_begin; k < row_end; k++) {
                const idx_t j = colidx[k];
                const real_t aij = values[k];

                val += aij * in[j];
            }

            out[i] = val;
        }
    }

    return 0;
}

int crs_diag(const ptrdiff_t nnodes,
             const count_t *const SFEM_RESTRICT rowptr,
             const idx_t *const SFEM_RESTRICT colidx,
             const real_t *const SFEM_RESTRICT values,
             real_t *const SFEM_RESTRICT diag) {
#pragma omp parallel
    {
#pragma omp for  // nowait
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            const count_t row_begin = rowptr[i];
            const count_t row_end = rowptr[i + 1];

            diag[i] = 0;

            real_t val = 0;
            for (count_t k = row_begin; k < row_end; k++) {
                const idx_t j = colidx[k];

                if (i == j) {
                    diag[i] = values[k];
                    break;
                }
            }
        }
    }

    return 0;
}
