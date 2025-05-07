#include "kelvin_voigt_newmark.h"

#include "hex8_kelvin_voigt_newmark.h"

#include <assert.h>
#include <stdio.h>

int kelvin_voigt_newmark_apply_aos(const enum ElemType               element_type,
                                   const ptrdiff_t                   nelements,
                                   const ptrdiff_t                   nnodes,
                                   idx_t **const SFEM_RESTRICT       elements,
                                   geom_t **const SFEM_RESTRICT      points,
                                   const real_t *const SFEM_RESTRICT u,
                                   real_t *const SFEM_RESTRICT       values) {
    switch (element_type) {
        case HEX8: {
            return hex8_kelvin_voigt_newmark_apply(
                    nelements, nnodes, elements, points, 3, &u[0], &u[1], &u[2], 3, &values[0], &values[1], &values[2]);
        }
        default: {
            SFEM_ERROR("kelvin_voigt_newmark_apply_aos not implemented for type %s\n", type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}
