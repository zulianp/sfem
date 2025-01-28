
#ifndef SFEM_SPMM_H
#define SFEM_SPMM_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

// C = AB
//
// Not parallel or good implementation, should probably only be used
// for AMG setup.
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

#ifdef __cplusplus
}
#endif

#endif  // SFEM_SPMM_H
