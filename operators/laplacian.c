
#include "tet10_laplacian.h"
#include "tet4_laplacian.h"
#include "tri3_laplacian.h"
#include "tri6_laplacian.h"

#include "macro_tet4_laplacian.h"
#include "macro_tri3_laplacian.h"

#include "hex8_laplacian.h"
#include "spectral_hex_laplacian.h"

#include "sfem_defs.h"

#include <mpi.h>
#include <stdio.h>

int laplacian_is_opt(int element_type) {
    return element_type == TRI3 || element_type == TET10 || element_type == TET4 || element_type == MACRO_TET4 ||
           element_type == MACRO_TRI3;
}

int laplacian_assemble_value(int                               element_type,
                             const ptrdiff_t                   nelements,
                             const ptrdiff_t                   nnodes,
                             idx_t **const SFEM_RESTRICT       elements,
                             geom_t **const SFEM_RESTRICT      points,
                             const real_t *const SFEM_RESTRICT u,
                             real_t *const SFEM_RESTRICT       value) {
    switch (element_type) {
        case TRI3: {
            return tri3_laplacian_assemble_value(nelements, nnodes, elements, points, u, value);
        }
        case TRI6: {
            return tri6_laplacian_assemble_value(nelements, nnodes, elements, points, u, value);
        }
        case TET4: {
            return tet4_laplacian_assemble_value(nelements, nnodes, elements, points, u, value);
        }
        case TET10: {
            return tet10_laplacian_assemble_value(nelements, nnodes, elements, points, u, value);
        }
        // case MACRO_TRI3: {
        //     return macro_tri3_laplacian_assemble_value(nelements, nnodes, elements, points, u,
        //     value);
        //
        // }
        default: {
            SFEM_ERROR("laplacian_assemble_value not implemented for type %s\n", type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

int laplacian_apply(int                               element_type,
                    const ptrdiff_t                   nelements,
                    const ptrdiff_t                   nnodes,
                    idx_t **const SFEM_RESTRICT       elements,
                    geom_t **const SFEM_RESTRICT      points,
                    const real_t *const SFEM_RESTRICT u,
                    real_t *const SFEM_RESTRICT       values) {
    switch (element_type) {
        case TRI3: {
            return tri3_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        case TRI6: {
            return tri6_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        case TET4: {
            return tet4_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        case TET10: {
            return tet10_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        case MACRO_TET4: {
            return macro_tet4_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        case MACRO_TRI3: {
            return macro_tri3_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        case HEX8: {
            return hex8_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        default: {
            SFEM_ERROR("laplacian_apply not implemented for type %s\n", type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

int laplacian_assemble_gradient(int                               element_type,
                                const ptrdiff_t                   nelements,
                                const ptrdiff_t                   nnodes,
                                idx_t **const SFEM_RESTRICT       elements,
                                geom_t **const SFEM_RESTRICT      points,
                                const real_t *const SFEM_RESTRICT u,
                                real_t *const SFEM_RESTRICT       values) {
    return laplacian_apply(element_type, nelements, nnodes, elements, points, u, values);
}

int laplacian_crs(int                                element_type,
                  const ptrdiff_t                    nelements,
                  const ptrdiff_t                    nnodes,
                  idx_t **const SFEM_RESTRICT        elements,
                  geom_t **const SFEM_RESTRICT       points,
                  const count_t *const SFEM_RESTRICT rowptr,
                  const idx_t *const SFEM_RESTRICT   colidx,
                  real_t *const SFEM_RESTRICT        values) {
    switch (element_type) {
        case TRI3: {
            return tri3_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        case TRI6: {
            return tri6_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        case TET4: {
            return tet4_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        case HEX8: {
            return hex8_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        case TET10: {
            return tet10_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        case MACRO_TET4: {
            return macro_tet4_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        case MACRO_TRI3: {
            return macro_tri3_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        default: {
            SFEM_ERROR("laplacian_crs not implemented for type %s\n", type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

int laplacian_diag(int                          element_type,
                   const ptrdiff_t              nelements,
                   const ptrdiff_t              nnodes,
                   idx_t **const SFEM_RESTRICT  elements,
                   geom_t **const SFEM_RESTRICT points,
                   real_t *const SFEM_RESTRICT  values) {
    switch (element_type) {
        case TRI3: {
            return tri3_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        case TRI6: {
            return tri6_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        case TET4: {
            return tet4_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        case HEX8: {
            return hex8_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        case TET10: {
            return tet10_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        case MACRO_TET4: {
            return macro_tet4_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        case MACRO_TRI3: {
            return macro_tri3_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        default: {
            SFEM_ERROR("laplacian_diag not implemented for type %s\n", type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

int laplacian_apply_opt(int                                   element_type,
                        const ptrdiff_t                       nelements,
                        idx_t **const SFEM_RESTRICT           elements,
                        const jacobian_t *const SFEM_RESTRICT fff,
                        const real_t *const SFEM_RESTRICT     u,
                        real_t *const SFEM_RESTRICT           values) {
    switch (element_type) {
        case TRI3: {
            return tri3_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        case TRI6: {
            return tri6_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        case TET4: {
            return tet4_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        case TET10: {
            return tet10_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        case MACRO_TET4: {
            return macro_tet4_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        case MACRO_TRI3: {
            return macro_tri3_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        default: {
            SFEM_ERROR("laplacian_apply_opt not implemented for type %s\n", type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

int laplacian_crs_sym(int                                element_type,
                      const ptrdiff_t                    nelements,
                      const ptrdiff_t                    nnodes,
                      idx_t **const SFEM_RESTRICT        elements,
                      geom_t **const SFEM_RESTRICT       points,
                      const count_t *const SFEM_RESTRICT rowptr,
                      const idx_t *const SFEM_RESTRICT   colidx,
                      real_t *const SFEM_RESTRICT        diag,
                      real_t *const SFEM_RESTRICT        offdiag) {
    switch (element_type) {
        case HEX8: {
            return hex8_laplacian_crs_sym(nelements, nnodes, elements, points, rowptr, colidx, diag, offdiag);
        }
        default : {
            SFEM_ERROR("laplacian_crs_sym not implemented for type %s\n", type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}
