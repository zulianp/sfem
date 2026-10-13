#pragma once

#include "sfem_base.hpp"

#include <stddef.h>

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif

#ifndef SFEM_FAILURE
#define SFEM_FAILURE 1
#endif

/*
 * In-place structured DSS on scalar fields stored in BXYZE layout.
 *
 * The element ordering is the structured HEX ordering used by create_hex8_cube:
 *   e = ex + nx * (ey + ny * ez)
 *
 * The cell-wise storage layout is:
 *   bxyze[b + B * (x + N * (y + N * (z + N * eb)))]
 *
 * with:
 *   N  = level + 1
 *   B  = element_block_size
 *   eb = e / B
 *   b  = e % B
 *
 * The algorithm performs three sequential face-exchange passes: z, y, x.
 * Each pass is parallel over disjoint interface pairs; edge and corner sums are
 * accumulated by the cascade of these passes and are not handled separately.
 */
int sfem_dss_bxyze_structured_inplace(int level,
                                      ptrdiff_t nx,
                                      ptrdiff_t ny,
                                      ptrdiff_t nz,
                                      int element_block_size,
                                      real_t *const bxyze);

int sfem_dss_bxyze_structured_inplace_real32(int level,
                                             ptrdiff_t nx,
                                             ptrdiff_t ny,
                                             ptrdiff_t nz,
                                             int element_block_size,
                                             float *const bxyze);

int sfem_dss_bxyze_structured_inplace_real64(int level,
                                             ptrdiff_t nx,
                                             ptrdiff_t ny,
                                             ptrdiff_t nz,
                                             int element_block_size,
                                             double *const bxyze);
