#ifndef CU_TET4_FFF_H
#define CU_TET4_FFF_H

#include "sfem_base.hpp"
#include "sfem_defs.hpp"

#ifdef __cplusplus
extern "C" {
#endif

int cu_tet4_fff_allocate(const ptrdiff_t nelements,
                         void **const SFEM_RESTRICT fff);

int cu_tet4_fff_fill(const ptrdiff_t nelements,
                     idx_t **const SFEM_RESTRICT elements,
                     geom_t **const SFEM_RESTRICT points,
                     void *const SFEM_RESTRICT fff);


#ifdef __cplusplus
}
#endif

#endif  // CU_TET4_FFF_H