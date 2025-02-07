#ifndef LUMPED_PTPD_H
#define LUMPED_PTPD_H

#include "sfem_base.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int lumped_ptdp_crs(const ptrdiff_t                    fine_nodes,
                    const count_t *const SFEM_RESTRICT rowptr,
                    const idx_t *const SFEM_RESTRICT   colidx,
                    const real_t *const SFEM_RESTRICT  values,
                    const real_t *const SFEM_RESTRICT  diagonal,
                    real_t *const SFEM_RESTRICT        coarse);

int lumped_ptdp_crs_v(const ptrdiff_t                    fine_nodes,
                      const count_t *const SFEM_RESTRICT rowptr,
                      const idx_t *const SFEM_RESTRICT   colidx,
                      const real_t *const SFEM_RESTRICT  values,
                      const int                          block_size,
                      const real_t *const SFEM_RESTRICT  diagonal,
                      real_t *const SFEM_RESTRICT        coarse);

#ifdef __cplusplus
}
#endif

#endif  // LUMPED_PTPD_H
