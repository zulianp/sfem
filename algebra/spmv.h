#ifndef SFEM_SPMV_H
#define SFEM_SPMV_H

#include <stddef.h>
#include "sfem_base.h"

int spmv_crs(const ptrdiff_t nnodes,
             const count_t *const SFEM_RESTRICT rowptr,
             const idx_t *const SFEM_RESTRICT colidx,
             const real_t *const SFEM_RESTRICT values,
             const real_t *const SFEM_RESTRICT in,
             real_t *const SFEM_RESTRICT out);

#endif  // SFEM_SPMV_H
