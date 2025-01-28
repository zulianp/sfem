#ifndef SMOOTHED_AGGREGATION_H
#define SMOOTHED_AGGREGATION_H

#include "sfem_base.h"
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include "sfem_config.h"

int smoothed_aggregation(const ptrdiff_t                    ndofs,
                         const ptrdiff_t                    ndofs_coarse,
                         const real_t                       jacobi_weight,
                         const real_t *const SFEM_RESTRICT  near_null,
                         const idx_t *const SFEM_RESTRICT   partition,
                         const count_t *const SFEM_RESTRICT rowptr_a,
                         const idx_t *const SFEM_RESTRICT   colidx_a,
                         const real_t *const SFEM_RESTRICT  values_a,
                         count_t                          **rowptr_p,       // [out]
                         idx_t                            **colidx_p,       // [out]
                         real_t                           **values_p,       // [out]
                         count_t                          **rowptr_pt,      // [out]
                         idx_t                            **colidx_pt,      // [out]
                         real_t                           **values_pt,      // [out]
                         count_t                          **rowptr_coarse,  // [out]
                         idx_t                            **colidx_coarse,  // [out]
                         real_t                           **values_coarse   // [out]
);

#ifdef __cplusplus
}
#endif
#endif  // SMOOTHED_AGGREGATION_H
