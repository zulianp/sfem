#ifndef SFEM_SPMV_H
#define SFEM_SPMV_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int scal(const ptrdiff_t nnodes, const real_t scale_factor, real_t *x);

int crs_spmv(const ptrdiff_t nnodes,
             const count_t *const SFEM_RESTRICT rowptr,
             const idx_t *const SFEM_RESTRICT colidx,
             const real_t *const SFEM_RESTRICT values,
             const real_t *const SFEM_RESTRICT in,
             real_t *const SFEM_RESTRICT out);

int crs_diag(
    const ptrdiff_t nnodes,
                 const count_t *const SFEM_RESTRICT rowptr,
                 const idx_t *const SFEM_RESTRICT colidx,
                 const real_t *const SFEM_RESTRICT values,
                 real_t *const SFEM_RESTRICT diag
    );

#ifdef __cplusplus
}
#endif


#endif  // SFEM_SPMV_H
