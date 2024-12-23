#ifndef NODE_INTERPOLATE_H
#define NODE_INTERPOLATE_H

#include <stddef.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int interpolate_gap(const ptrdiff_t              nnodes,
                    geom_t** const SFEM_RESTRICT xyz,
                    // SDF
                    const ptrdiff_t* const SFEM_RESTRICT n,
                    const ptrdiff_t* const SFEM_RESTRICT stride,
                    const geom_t* const SFEM_RESTRICT    origin,
                    const geom_t* const SFEM_RESTRICT    delta,
                    const geom_t* const SFEM_RESTRICT    data,
                    // Output
                    real_t* const SFEM_RESTRICT g,
                    real_t* const SFEM_RESTRICT xnormal,
                    real_t* const SFEM_RESTRICT ynormal,
                    real_t* const SFEM_RESTRICT znormal);

int interpolate_gap_value(const ptrdiff_t              nnodes,
                          geom_t** const SFEM_RESTRICT xyz,
                          // SDF
                          const ptrdiff_t* const SFEM_RESTRICT n,
                          const ptrdiff_t* const SFEM_RESTRICT stride,
                          const geom_t* const SFEM_RESTRICT    origin,
                          const geom_t* const SFEM_RESTRICT    delta,
                          const geom_t* const SFEM_RESTRICT    data,
                          // Output
                          real_t* const SFEM_RESTRICT g);

int interpolate_gap_normals(const ptrdiff_t              nnodes,
                            geom_t** const SFEM_RESTRICT xyz,
                            // SDF
                            const ptrdiff_t* const SFEM_RESTRICT n,
                            const ptrdiff_t* const SFEM_RESTRICT stride,
                            const geom_t* const SFEM_RESTRICT    origin,
                            const geom_t* const SFEM_RESTRICT    delta,
                            const geom_t* const SFEM_RESTRICT    data,
                            // Output
                            real_t* const SFEM_RESTRICT xnormal,
                            real_t* const SFEM_RESTRICT ynormal,
                            real_t* const SFEM_RESTRICT znormal);

#ifdef __cplusplus
}
#endif

#endif  // NODE_INTERPOLATE_H
