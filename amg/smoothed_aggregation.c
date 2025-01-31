#include "smoothed_aggregation.h"
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include "crs.h"
#include "sfem_base.h"

int smoothed_aggregation(const ptrdiff_t                    ndofs,
                         const ptrdiff_t                    ndofs_coarse,
                         const real_t                       jacobi_weight,
                         const idx_t *const SFEM_RESTRICT   partition,
                         const count_t *const SFEM_RESTRICT rowptr_a,
                         const idx_t *const SFEM_RESTRICT   colidx_a,
                         const real_t *const SFEM_RESTRICT  values_a,
                         const real_t *const SFEM_RESTRICT  diag_a,
                         real_t                            *near_null,
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

    assert(!crs_validate(ndofs, ndofs, rowptr_a, colidx_a, values_a));

    rowptr_unsmoothed[0] = 0;

    for (idx_t i = 0; i < ndofs; i++) {
        idx_t ic = partition[i];
        if (ic >= 0) {
            agg_norms[ic] += near_null[i] * near_null[i];
        }
    }

#pragma omp parallel for
    for (idx_t ic = 0; ic < ndofs_coarse; ic++) {
        agg_norms[ic] = sqrt(agg_norms[ic]);
    }

    idx_t counter = 0;
    for (idx_t i = 0; i < ndofs; i++) {
        idx_t ic = partition[i];
        if (ic >= 0) {
            colidx_unsmoothed[counter] = ic;
            values_unsmoothed[counter] = near_null[i] / agg_norms[ic];
            counter++;
        }
        rowptr_unsmoothed[i + 1] = counter;
    }

#pragma omp parallel for
    for (idx_t ic = 0; ic < ndofs_coarse; ic++) {
        near_null[ic] = agg_norms[ic];
    }

    assert(!crs_validate(ndofs, ndofs_coarse, rowptr_unsmoothed, colidx_unsmoothed, values_unsmoothed));
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

    assert(!crs_validate(ndofs, ndofs_coarse, *rowptr_p, *colidx_p, *values_p));

    // sanity validation still looking for bug
    for (idx_t i = 0; i < ndofs; i++) {
        if (partition[i] < 0) {
            assert((*rowptr_p)[i] == (*rowptr_p)[i + 1]);
        }
    }

    // (I - w D^-1 A) P
    for (idx_t i = 0; i < ndofs; i++) {
        for (count_t idx = (*rowptr_p)[i]; idx < (*rowptr_p)[i + 1]; idx++) {
            (*values_p)[idx] *= -1.0 * jacobi_weight / diag_a[i];
            if (partition[i] == (*colidx_p)[idx]) {
                count_t idx_unsmoothed = rowptr_unsmoothed[i];
                (*values_p)[idx] += values_unsmoothed[idx_unsmoothed];
            }
        }
    }

    free(rowptr_unsmoothed);
    free(colidx_unsmoothed);
    free(values_unsmoothed);

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

    assert(!crs_validate(ndofs, ndofs_coarse, rowptr_ap, colidx_ap, values_ap));

    *rowptr_pt = (count_t *)malloc((ndofs + 1) * sizeof(count_t));
    *colidx_pt = (idx_t *)malloc((*rowptr_p)[ndofs] * sizeof(idx_t));
    *values_pt = (real_t *)malloc((*rowptr_p)[ndofs] * sizeof(real_t));
    crs_transpose(ndofs, ndofs_coarse, *rowptr_p, *colidx_p, *values_p, *rowptr_pt, *colidx_pt, *values_pt);

    assert(!crs_validate(ndofs_coarse, ndofs, *rowptr_pt, *colidx_pt, *values_pt));
    // Pt A P
    crs_spmm(ndofs_coarse,
             ndofs_coarse,
             *rowptr_pt,
             *colidx_pt,
             *values_pt,
             rowptr_ap,
             colidx_ap,
             values_ap,
             rowptr_coarse,
             colidx_coarse,
             values_coarse);

    assert(!crs_validate(ndofs_coarse, ndofs_coarse, *rowptr_coarse, *colidx_coarse, *values_coarse));
    assert(crs_is_symmetric(ndofs_coarse, *rowptr_coarse, *colidx_coarse, *values_coarse));

    free(rowptr_ap);
    free(colidx_ap);
    free(values_ap);
    free(agg_norms);
    return 0;
}
