#ifndef MACRO_TET4_LAPLACIAN_H
#define MACRO_TET4_LAPLACIAN_H

// Adapted from Bole Ma implementation in https://github.com/zulianp/hpcfem

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int sstet4_nxe(int level);
int sstet4_txe(int level);

int sstet4_laplacian_apply(const int level,
                                 const ptrdiff_t nelements,
                                 const jacobian_t *const SFEM_RESTRICT fff,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT values);

#ifdef __cplusplus
}
#endif
#endif  // MACRO_TET4_LAPLACIAN_H
