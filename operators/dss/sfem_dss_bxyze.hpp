#pragma once

#include <stddef.h>
#include <stdint.h>

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif

#ifndef SFEM_FAILURE
#define SFEM_FAILURE 1
#endif

/*
 * Scalar nodal field conversion from SFEM global unique storage to blocked
 * element-wise BXYZE storage.
 *
 * Input connectivity uses the SFEM semi-structured SoA layout:
 *   elements[lidx][e] = global node id of local node lidx in element e
 *
 * Output layout:
 *   bxyze[b + B * (x + N * (y + N * (z + N * eb)))]
 *
 * where:
 *   N  = level + 1
 *   B  = element_block_size
 *   eb = element / B
 *   b  = element % B
 *
 * The number of stored values includes padded lanes in the last element block.
 * Conversion routines set those padded lanes to zero.
 */
ptrdiff_t sfem_dss_bxyze_n_values(int level, ptrdiff_t nelements, int element_block_size);

ptrdiff_t sfem_dss_bxyze_index(int level,
                               int element_block_size,
                               ptrdiff_t element,
                               int ix,
                               int iy,
                               int iz);

int sfem_dss_global_to_bxyze_real32(int level,
                                    ptrdiff_t nelements,
                                    int element_block_size,
                                    const int32_t *const *const elements,
                                    const float *const global,
                                    float *const bxyze);

int sfem_dss_global_to_bxyze_real64(int level,
                                    ptrdiff_t nelements,
                                    int element_block_size,
                                    const int32_t *const *const elements,
                                    const double *const global,
                                    double *const bxyze);
