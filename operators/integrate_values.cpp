#include "integrate_values.hpp"

#include "sfem_base.hpp"
#include "sfem_defs.hpp"

#include "edgeshell2_integrate_values.hpp"
#include "quadshell4_integrate_values.hpp"
#include "trishell3_integrate_values.hpp"
#include "trishell6_integrate_values.hpp"

#include <assert.h>
#include <stdio.h>

int integrate_value(const int                    element_type,
                    const ptrdiff_t              nelements,
                    const ptrdiff_t              nnodes,
                    idx_t **const SFEM_RESTRICT  elems,
                    geom_t **const SFEM_RESTRICT xyz,
                    const real_t                 value,
                    const int                    block_size,
                    const int                    component,
                    real_t *const SFEM_RESTRICT  out) {
    if (!nelements) return SFEM_SUCCESS;

    switch (element_type) {
        case smesh::BEAM2: {
            return edgeshell2_integrate_value(nelements, nnodes, elems, xyz, value, block_size, component, out);
        }
        case smesh::EDGESHELL2: {
            return edgeshell2_integrate_value(nelements, nnodes, elems, xyz, value, block_size, component, out);
        }
        case smesh::TRISHELL3: {
            return trishell3_integrate_value(nelements, nnodes, elems, xyz, value, block_size, component, out);
        }
        case smesh::TRISHELL6: {
            return trishell6_integrate_value(nelements, nnodes, elems, xyz, value, block_size, component, out);
        }
        case smesh::QUADSHELL4: {
            return quadshell4_integrate_value(nelements, nnodes, elems, xyz, value, block_size, component, out);
        }

        default: {
            SFEM_ERROR("integrate_value not implemented for type %s\n", sfem::type_to_string((smesh::ElemType)element_type));
        }
    }

    return SFEM_FAILURE;
}

int integrate_values(const int                         element_type,
                     const ptrdiff_t                   nelements,
                     const ptrdiff_t                   nnodes,
                     idx_t **const SFEM_RESTRICT       elems,
                     geom_t **const SFEM_RESTRICT      xyz,
                     const real_t                      scale_factor,
                     const real_t *const SFEM_RESTRICT values,
                     const int                         block_size,
                     const int                         component,
                     real_t *const SFEM_RESTRICT       out) {
    if (!nelements) return SFEM_SUCCESS;

    switch (element_type) {
        case smesh::BEAM2: {
            return edgeshell2_integrate_values(nelements, nnodes, elems, xyz, scale_factor, values, block_size, component, out);
        }
        case smesh::EDGESHELL2: {
            return edgeshell2_integrate_values(nelements, nnodes, elems, xyz, scale_factor, values, block_size, component, out);
        }
        case smesh::TRISHELL3: {
            return trishell3_integrate_values(nelements, nnodes, elems, xyz, scale_factor, values, block_size, component, out);
        }
        case smesh::TRISHELL6: {
            return trishell6_integrate_values(nelements, nnodes, elems, xyz, scale_factor, values, block_size, component, out);
        }
        case smesh::QUADSHELL4: {
            return quadshell4_integrate_values(nelements, nnodes, elems, xyz, scale_factor, values, block_size, component, out);
        }

        default: {
            SFEM_ERROR("integrate_values not implemented for type %s\n", sfem::type_to_string((smesh::ElemType)element_type));
        }
    }

    return SFEM_FAILURE;
}
