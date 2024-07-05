#ifndef CU_TET4_FFF_H
#define CU_TET4_FFF_H

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int cu_tet4_fff_allocate(const ptrdiff_t nelements,
                         const enum RealType real_type,
                         void **const SFEM_RESTRICT fff);

int cu_tet4_fff_fill(const ptrdiff_t nelements,
                     idx_t **const SFEM_RESTRICT elements,
                     geom_t **const SFEM_RESTRICT points,
                     const enum RealType real_type,
                     void *const SFEM_RESTRICT fff);

// Currently this is the only one supported
inline int cu_tet4_fff_allocate_default(const ptrdiff_t nelements, void **const SFEM_RESTRICT fff) {
    return cu_tet4_fff_allocate(nelements, SFEM_FLOAT_DEFAULT, fff);
}

// Currently this is the only one supported
inline int cu_tet4_fff_fill_default(const ptrdiff_t nelements,
                                    idx_t **const SFEM_RESTRICT elements,
                                    geom_t **const SFEM_RESTRICT points,
                                    void *const SFEM_RESTRICT fff) {
    return cu_tet4_fff_fill(nelements, elements, points, SFEM_FLOAT_DEFAULT, fff);
}

int elements_to_device(const ptrdiff_t nelements,
                       const int num_nodes_x_element,
                       idx_t **const SFEM_RESTRICT h_elements,
                       idx_t **const SFEM_RESTRICT d_elements);

#ifdef __cplusplus
}
#endif

#endif  // CU_TET4_FFF_H