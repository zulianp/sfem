#include "cu_integrate_values.h"

#include "cu_quadshell4_integrate_values.h"

#include "sfem_base.h"
#include "sfem_defs.h"

#include <assert.h>
#include <stdio.h>

int cu_integrate_value(const int                          element_type,
                       const ptrdiff_t                    nelements,
                       idx_t **const SFEM_RESTRICT        elements,
                       const geom_t **const SFEM_RESTRICT coords,  // coords are stored per element
                       const real_t                       value,
                       const int                          block_size,
                       const int                          component,
                       const enum RealType                real_type,
                       void *const SFEM_RESTRICT          out,
                       void                              *stream) {
    if (!nelements) return SFEM_SUCCESS;

    const ptrdiff_t coords_stride = nelements;

    // Debug
    // printf("[DBG wrap val] type=%d ne=%ld bs=%d comp=%d\n", element_type, (long)nelements, block_size, component);

    switch (element_type) {
        case QUADSHELL4: {
            return cu_quadshell4_integrate_value(
                    nelements, elements, coords_stride, coords, value, block_size, component, real_type, out, stream);
        }

        default: {
            SFEM_ERROR("cu_integrate_value: not implemented for type %s\n", type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

int cu_integrate_values(const int                          element_type,
                        const ptrdiff_t                    nelements,
                        idx_t **const SFEM_RESTRICT        elements,
                        const geom_t **const SFEM_RESTRICT coords,  // coords are stored per element
                        const enum RealType                real_type,
                        void *const SFEM_RESTRICT          values,
                        const int                          block_size,
                        const int                          component,
                        void *const SFEM_RESTRICT          out,
                        void                              *stream) {
    if (!nelements) return SFEM_SUCCESS;

    const ptrdiff_t coords_stride = nelements;

    // Debug
    // printf("[DBG wrap arr] type=%d ne=%ld bs=%d comp=%d\n", element_type, (long)nelements, block_size, component);

    switch (element_type) {
        case QUADSHELL4: {
            return cu_quadshell4_integrate_values(
                    nelements, elements, coords_stride, coords, real_type, values, block_size, component, out, stream);
        }

        default: {
            SFEM_ERROR("cu_integrate_values: not implemented for type %s\n", type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}