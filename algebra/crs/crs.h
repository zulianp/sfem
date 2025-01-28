
#ifndef SFEM_SPMM_H
#define SFEM_SPMM_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

// C = AB
// Not parallel / good impl, should only be used for AMG setup.
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
);

/**
 * Transpose a CSR matrix A of dimension (rows x cols) into At (cols x rows).
 *
 *  rowptr, colidx, values -> input
 *  rowptr_t, colidx_t, values_t -> output
 *
 * On entry, rowptr_t, colidx_t, values_t must be allocated by the caller to the
 * correct lengths:
 *   - rowptr_t must be length (cols + 1)
 *   - colidx_t, values_t must be length (rowptr[rows]) i.e. NNZ of A and At
 */
int crs_transpose(const count_t        rows,
                  const count_t        cols,
                  const count_t *const rowptr,
                  const idx_t *const   colidx,
                  const real_t *const  values,
                  count_t *const       rowptr_t,  // [out]
                  idx_t *const         colidx_t,  // [out]
                  real_t *const        values_t   // [out]
);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_SPMM_H
