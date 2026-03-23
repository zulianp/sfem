#ifndef CU_INTEGRATE_VALUES_H
#define CU_INTEGRATE_VALUES_H

#include "sfem_base.hpp"
#include "sfem_defs.hpp"

#include <stddef.h>

namespace sfem {

    int cu_integrate_value(const enum smesh::ElemType         element_type,
                           const ptrdiff_t                    nelements,
                           idx_t **const SFEM_RESTRICT        elements,
                           const geom_t **const SFEM_RESTRICT coords,  // coords are stored per element
                           const real_t                       value,
                           const int                          block_size,
                           const int                          component,
                           const enum smesh::PrimitiveType                real_type,
                           void *const SFEM_RESTRICT          out,
                           void                              *stream);

    int cu_integrate_values(const enum smesh::ElemType         element_type,
                            const ptrdiff_t                    nelements,
                            idx_t **const SFEM_RESTRICT        elements,
                            const geom_t **const SFEM_RESTRICT coords,  // coords are stored per element
                            const enum smesh::PrimitiveType                real_type,
                            void *const SFEM_RESTRICT          values,
                            const int                          block_size,
                            const int                          component,
                            void *const SFEM_RESTRICT          out,
                            void                              *stream);
}

#endif  // CU_INTEGRATE_VALUES_H
