#include "smoothed_aggregation.h"
#include <math.h>
#include <stddef.h>
#include "crs.h"
#include "sfem_base.h"

int smoothed_aggregation(const ptrdiff_t                    ndofs,
                         const ptrdiff_t                    ndofs_coarse,
                         const real_t                       jacobi_weight,
                         const real_t *const SFEM_RESTRICT  near_null,
                         const idx_t *const SFEM_RESTRICT   partition,
                         const count_t *const SFEM_RESTRICT rowptr_a,
                         const idx_t *const SFEM_RESTRICT   colidx_a,
                         const real_t *const SFEM_RESTRICT  values_a,
                         count_t                          **rowptr_p,       // [out]
                         idx_t                            **colidx_p,       // [out]
                         real_t                           **values_p,       // [out]
                         count_t                          **rowptr_pt,      // [out]
                         idx_t                            **colidx_pt,      // [out]
                         real_t                           **values_pt,      // [out]
                         count_t                          **rowptr_coarse,  // [out]
                         idx_t                            **colidx_coarse,  // [out]
                         real_t                           **values_coarse   // [out]
) {
    count_t *rowptr_unsmoothed = (count_t *)malloc((ndofs + 1) * sizeof(count_t));
    idx_t   *colidx_unsmoothed = (idx_t *)malloc(ndofs * sizeof(idx_t));
    real_t  *values_unsmoothed = (real_t *)malloc(ndofs * sizeof(real_t));
    real_t  *agg_norms         = (real_t *)calloc(ndofs_coarse, sizeof(real_t));
    real_t  *diag_inv          = (real_t *)malloc(ndofs * sizeof(real_t));

#pragma omp parallel for
    for (idx_t i = 0; i < ndofs; i++) {
        real_t diag_val;

        for (count_t idx = rowptr_a[i]; idx < rowptr_a[i + 1]; idx++) {
            // could binary search...
            if (colidx_a[idx] == i) {
                diag_inv[i] = jacobi_weight / values_a[idx];
                break;
            }
        }
    }

    rowptr_unsmoothed[0] = 0;

    for (idx_t i = 0; i < ndofs; i++) {
        idx_t ic = partition[i];
        agg_norms[ic] += near_null[i] * near_null[i];
    }

#pragma omp parallel for
    for (idx_t ic = 0; ic < ndofs_coarse; ic++) {
        agg_norms[ic] = sqrt(agg_norms[ic]);
    }

#pragma omp parallel for
    for (idx_t i = 0; i < ndofs; i++) {
        rowptr_unsmoothed[i + 1] = i;
        idx_t ic                 = partition[i];
        colidx_unsmoothed[i]     = ic;
        values_unsmoothed[i]     = near_null[i] / agg_norms[ic];
    }

    // AP (unsmoothed)
    crs_spmm(ndofs,
             ndofs_coarse,
             rowptr_a,
             colidx_a,
             values_a,
             rowptr_unsmoothed,
             colidx_unsmoothed,
             values_unsmoothed,
             rowptr_p,
             colidx_p,
             values_p);

    // (I - w D^-1 A) P
    for (idx_t i = 0; i < ndofs; i++) {
        for (count_t idx = (*rowptr_p)[i]; idx < (*rowptr_p)[i + 1]; idx++) {
            (*values_p)[idx] *= -diag_inv[i];
            if (partition[i] == (*colidx_p)[idx]) {
                (*values_p)[idx] += values_unsmoothed[i];
            }
        }
    }

    count_t *rowptr_ap;
    idx_t   *colidx_ap;
    real_t  *values_ap;

    // AP (smoothed)
    crs_spmm(ndofs,
             ndofs_coarse,
             rowptr_a,
             colidx_a,
             values_a,
             *rowptr_p,
             *colidx_p,
             *values_p,
             &rowptr_ap,
             &colidx_ap,
             &values_ap);

    *rowptr_pt = (count_t *)malloc((ndofs + 1) * sizeof(count_t));
    *colidx_pt = (idx_t *)malloc((*rowptr_p)[ndofs] * sizeof(idx_t));
    *values_pt = (real_t *)malloc((*rowptr_p)[ndofs] * sizeof(real_t));
    crs_transpose(ndofs, ndofs_coarse, *rowptr_p, *colidx_p, *values_p, *rowptr_pt, *colidx_pt, *values_pt);

    // Pt A P
    crs_spmm(ndofs_coarse,
             ndofs_coarse,
             rowptr_a,
             colidx_a,
             values_a,
             *rowptr_p,
             *colidx_p,
             *values_p,
             &rowptr_ap,
             &colidx_ap,
             &values_ap);

    free(rowptr_ap);
    free(colidx_ap);
    free(values_ap);
    free(rowptr_unsmoothed);
    free(colidx_unsmoothed);
    free(values_unsmoothed);
    free(agg_norms);
    free(diag_inv);
    return 0;
}
