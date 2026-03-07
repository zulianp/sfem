#include "lumped_ptdp.h"

int lumped_ptdp_crs(const ptrdiff_t                    fine_nodes,
                    const count_t *const SFEM_RESTRICT rowptr,
                    const idx_t *const SFEM_RESTRICT   colidx,
                    const real_t *const SFEM_RESTRICT  values,
                    const real_t *const SFEM_RESTRICT  diagonal,
                    real_t *const SFEM_RESTRICT        coarse) {
    for (ptrdiff_t m = 0; m < fine_nodes; m++) {
        const count_t len = rowptr[m + 1] - rowptr[m];
        const real_t  d   = diagonal[m];

        const idx_t *const  cols = &colidx[rowptr[m]];
        const real_t *const vals = &values[rowptr[m]];

        for (count_t k = 0; k < len; k++) {
            for (count_t l = 0; l < len; l++) {
                coarse[cols[k]] += vals[k] * vals[l] * d;
            }
        }
    }

    return SFEM_SUCCESS;
}

int lumped_ptdp_crs_v(const ptrdiff_t                    fine_nodes,
                      const count_t *const SFEM_RESTRICT rowptr,
                      const idx_t *const SFEM_RESTRICT   colidx,
                      const real_t *const SFEM_RESTRICT  values,
                      const int                          block_size,
                      const real_t *const SFEM_RESTRICT  diagonal,
                      real_t *const SFEM_RESTRICT        coarse) {
    for (ptrdiff_t m = 0; m < fine_nodes; m++) {
        const count_t       len = rowptr[m + 1] - rowptr[m];
        const real_t *const d   = &diagonal[m * block_size];

        const idx_t *const  cols = &colidx[rowptr[m]];
        const real_t *const vals = &values[rowptr[m]];

        for (count_t k = 0; k < len; k++) {
            real_t *const c = &coarse[cols[k] * block_size];
            for (count_t l = 0; l < len; l++) {
                const real_t w = vals[k] * vals[l];
                for (int b = 0; b < block_size; b++) {
                    c[b] += w * d[b];
                }
            }
        }
    }

    return SFEM_SUCCESS;
}
