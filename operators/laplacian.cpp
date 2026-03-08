
#include "tet10_laplacian.hpp"
#include "tet4_laplacian.hpp"
#include "tri3_laplacian.hpp"
#include "tri6_laplacian.hpp"

#include "macro_tet4_laplacian.hpp"
#include "macro_tri3_laplacian.hpp"

#include "hex8_laplacian.hpp"
#include "spectral_hex_laplacian.hpp"

#include "sfem_defs.hpp"

#include <mpi.h>
#include <stdio.h>

int laplacian_is_opt(int element_type) {
    return element_type == smesh::TRI3 || element_type == smesh::TET10 || element_type == smesh::TET4 || element_type == smesh::MACRO_TET4 ||
           element_type == smesh::MACRO_TRI3;
}

int laplacian_assemble_value(int                               element_type,
                             const ptrdiff_t                   nelements,
                             const ptrdiff_t                   nnodes,
                             idx_t **const SFEM_RESTRICT       elements,
                             geom_t **const SFEM_RESTRICT      points,
                             const real_t *const SFEM_RESTRICT u,
                             real_t *const SFEM_RESTRICT       value) {
    switch (element_type) {
        case smesh::TRI3: {
            return tri3_laplacian_assemble_value(nelements, nnodes, elements, points, u, value);
        }
        case smesh::TRI6: {
            return tri6_laplacian_assemble_value(nelements, nnodes, elements, points, u, value);
        }
        case smesh::TET4: {
            return tet4_laplacian_assemble_value(nelements, nnodes, elements, points, u, value);
        }
        case smesh::TET10: {
            return tet10_laplacian_assemble_value(nelements, nnodes, elements, points, u, value);
        }
        // case smesh::MACRO_TRI3: {
        //     return macro_tri3_laplacian_assemble_value(nelements, nnodes, elements, points, u,
        //     value);
        //
        // }
        default: {
            SFEM_ERROR("laplacian_assemble_value not implemented for type %s\n", sfem::type_to_string((smesh::ElemType)element_type));
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
        case smesh::TRI3: {
            return tri3_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        case smesh::TRI6: {
            return tri6_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        case smesh::TET4: {
            return tet4_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        case smesh::TET10: {
            return tet10_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        case smesh::MACRO_TET4: {
            return macro_tet4_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        case smesh::MACRO_TRI3: {
            return macro_tri3_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        case smesh::HEX8: {
            return hex8_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        default: {
            SFEM_ERROR("laplacian_apply not implemented for type %s\n", sfem::type_to_string((smesh::ElemType)element_type));
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
        case smesh::TRI3: {
            return tri3_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        case smesh::TRI6: {
            return tri6_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        case smesh::TET4: {
            return tet4_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        case smesh::HEX8: {
            return hex8_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        case smesh::TET10: {
            return tet10_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        case smesh::MACRO_TET4: {
            return macro_tet4_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        case smesh::MACRO_TRI3: {
            return macro_tri3_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        default: {
            SFEM_ERROR("laplacian_crs not implemented for type %s\n", sfem::type_to_string((smesh::ElemType)element_type));
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
        case smesh::TRI3: {
            return tri3_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        case smesh::TRI6: {
            return tri6_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        case smesh::TET4: {
            return tet4_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        case smesh::HEX8: {
            return hex8_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        case smesh::TET10: {
            return tet10_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        case smesh::MACRO_TET4: {
            return macro_tet4_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        case smesh::MACRO_TRI3: {
            return macro_tri3_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        default: {
            SFEM_ERROR("laplacian_diag not implemented for type %s\n", sfem::type_to_string((smesh::ElemType)element_type));
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
        case smesh::TRI3: {
            return tri3_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        case smesh::TRI6: {
            return tri6_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        case smesh::TET4: {
            return tet4_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        case smesh::HEX8: {
            return hex8_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        case smesh::TET10: {
            return tet10_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        case smesh::MACRO_TET4: {
            return macro_tet4_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        case smesh::MACRO_TRI3: {
            return macro_tri3_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        default: {
            SFEM_ERROR("laplacian_apply_opt not implemented for type %s\n", sfem::type_to_string((smesh::ElemType)element_type));
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
        case smesh::HEX8: {
            return hex8_laplacian_crs_sym(nelements, nnodes, elements, points, rowptr, colidx, diag, offdiag);
        }
        default: {
            SFEM_ERROR("laplacian_crs_sym not implemented for type %s\n", sfem::type_to_string((smesh::ElemType)element_type));
        }
    }

    return SFEM_FAILURE;
}
