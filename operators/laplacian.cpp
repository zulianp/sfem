#include "laplacian.hpp"

#include "tet10_laplacian.hpp"
#include "tet4_laplacian.hpp"
#include "tri3_laplacian.hpp"
#include "tri6_laplacian.hpp"

#include "macro_tet4_laplacian.hpp"
#include "macro_tri3_laplacian.hpp"

#include "hex8_laplacian.hpp"
#include "quad4_laplacian.hpp"
#include "spectral_hex_laplacian.hpp"
#include "sshex8_laplacian.hpp"
#include "ssquad4_laplacian.hpp"
#include "sstet4_laplacian.hpp"

#include "sfem_defs.hpp"

#include <mpi.h>
#include <stdio.h>

int laplacian_is_opt(smesh::ElemType element_type) {
    return element_type == smesh::TRI3 || element_type == smesh::TET10 || element_type == smesh::TET4 ||
           element_type == smesh::MACRO_TET4 || element_type == smesh::MACRO_TRI3 ||
           (sfem::is_semistructured_type(element_type) && smesh::is_hex_ss_family(element_type));
}

int laplacian_has_kernel(smesh::ElemType element_type) {
    if (sfem::is_semistructured_type(element_type)) {
        return smesh::is_hex_ss_family(element_type) || smesh::is_quad_ss_family(element_type) ||
               smesh::is_tet_ss_family(element_type);
    }

    switch (element_type) {
        case smesh::TRI3:
        case smesh::TRI6:
        case smesh::TET4:
        case smesh::TET10:
        case smesh::MACRO_TET4:
        case smesh::MACRO_TRI3:
        case smesh::HEX8:
        case smesh::QUAD4:
            return 1;
        default:
            return 0;
    }
}

static int laplacian_error_unsupported(const char *const fn, const smesh::ElemType element_type) {
    if (smesh::is_wedge_ss_family(element_type)) {
        SFEM_ERROR("%s: no kernel for WEDGE family (%s); hex-dominant apply is not implemented\n",
                   fn,
                   sfem::type_to_string(element_type));
    } else if (smesh::is_pyramid_ss_family(element_type)) {
        SFEM_ERROR("%s: no kernel for PYRAMID family (%s); hex-dominant apply is not implemented\n",
                   fn,
                   sfem::type_to_string(element_type));
    } else {
        SFEM_ERROR("%s not implemented for type %s\n", fn, sfem::type_to_string(element_type));
    }
    return SFEM_FAILURE;
}

int laplacian_assemble_value(smesh::ElemType                   element_type,
                             const ptrdiff_t                   nelements,
                             const ptrdiff_t                   nnodes,
                             idx_t **const SFEM_RESTRICT       elements,
                             geom_t **const SFEM_RESTRICT      points,
                             const real_t *const SFEM_RESTRICT u,
                             real_t *const SFEM_RESTRICT       value) {
    if (sfem::is_semistructured_type(element_type)) {
        return laplacian_error_unsupported("laplacian_assemble_value", element_type);
    }

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
        default: {
            return laplacian_error_unsupported("laplacian_assemble_value", element_type);
        }
    }

    return SFEM_FAILURE;
}

int laplacian_apply(smesh::ElemType                   element_type,
                    const ptrdiff_t                   nelements,
                    const ptrdiff_t                   nnodes,
                    idx_t **const SFEM_RESTRICT       elements,
                    geom_t **const SFEM_RESTRICT      points,
                    const real_t *const SFEM_RESTRICT u,
                    real_t *const SFEM_RESTRICT       values) {
    if (sfem::is_semistructured_type(element_type)) {
        const int level = smesh::semistructured_level(element_type);
        if (smesh::is_hex_ss_family(element_type)) {
            return sshex8_laplacian_apply(level, nelements, elements, points, u, values);
        }
        if (smesh::is_quad_ss_family(element_type)) {
            return ssquad4_laplacian_apply(level, nelements, elements, points, u, values);
        }
        if (smesh::is_tet_ss_family(element_type)) {
            return sstet4_laplacian_apply_points(level, nelements, elements, points, u, values);
        }

        return laplacian_error_unsupported("laplacian_apply", element_type);
    }

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
        case smesh::QUAD4: {
            return quad4_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        default: {
            return laplacian_error_unsupported("laplacian_apply", element_type);
        }
    }

    return SFEM_FAILURE;
}

int laplacian_assemble_gradient(smesh::ElemType                   element_type,
                                const ptrdiff_t                   nelements,
                                const ptrdiff_t                   nnodes,
                                idx_t **const SFEM_RESTRICT       elements,
                                geom_t **const SFEM_RESTRICT      points,
                                const real_t *const SFEM_RESTRICT u,
                                real_t *const SFEM_RESTRICT       values) {
    return laplacian_apply(element_type, nelements, nnodes, elements, points, u, values);
}

int laplacian_crs(smesh::ElemType                    element_type,
                  const ptrdiff_t                    nelements,
                  const ptrdiff_t                    nnodes,
                  idx_t **const SFEM_RESTRICT        elements,
                  geom_t **const SFEM_RESTRICT       points,
                  const count_t *const SFEM_RESTRICT rowptr,
                  const idx_t *const SFEM_RESTRICT   colidx,
                  real_t *const SFEM_RESTRICT        values) {
    if (sfem::is_semistructured_type(element_type)) {
        return laplacian_error_unsupported("laplacian_crs", element_type);
    }

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
            return laplacian_error_unsupported("laplacian_crs", element_type);
        }
    }

    return SFEM_FAILURE;
}

int laplacian_diag(smesh::ElemType              element_type,
                   const ptrdiff_t              nelements,
                   const ptrdiff_t              nnodes,
                   idx_t **const SFEM_RESTRICT  elements,
                   geom_t **const SFEM_RESTRICT points,
                   real_t *const SFEM_RESTRICT  values) {
    if (sfem::is_semistructured_type(element_type)) {
        const int level = smesh::semistructured_level(element_type);
        if (smesh::is_hex_ss_family(element_type)) {
            return affine_sshex8_laplacian_diag(level, nelements, elements, points, values);
        }
        if (smesh::is_quad_ss_family(element_type)) {
            return ssquad4_laplacian_diag(level, nelements, elements, points, values);
        }
        if (smesh::is_tet_ss_family(element_type)) {
            return sstet4_laplacian_diag_points(level, nelements, elements, points, values);
        }

        return laplacian_error_unsupported("laplacian_diag", element_type);
    }

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
        case smesh::QUAD4: {
            return quad4_laplacian_diag(nelements, nnodes, elements, points, values);
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
            return laplacian_error_unsupported("laplacian_diag", element_type);
        }
    }

    return SFEM_FAILURE;
}

int laplacian_apply_opt(smesh::ElemType                       element_type,
                        const ptrdiff_t                       nelements,
                        idx_t **const SFEM_RESTRICT           elements,
                        const jacobian_t *const SFEM_RESTRICT fff,
                        const real_t *const SFEM_RESTRICT     u,
                        real_t *const SFEM_RESTRICT           values) {
    if (sfem::is_semistructured_type(element_type)) {
        const int level = smesh::semistructured_level(element_type);
        if (smesh::is_hex_ss_family(element_type)) {
            return affine_sshex8_laplacian_stencil_apply_fff(level, nelements, elements, fff, u, values);
        }
        if (smesh::is_tet_ss_family(element_type)) {
            sstet4_laplacian_stencil_t *stencil = nullptr;
            int err = sstet4_laplacian_stencil_create(level, nelements, fff, &stencil);
            if (err != SFEM_SUCCESS) {
                return err;
            }

            err = sstet4_laplacian_apply_stencil_global(stencil, nelements, elements, u, values);
            sstet4_laplacian_stencil_destroy(stencil);
            return err;
        }

        return laplacian_error_unsupported("laplacian_apply_opt", element_type);
    }

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
            return laplacian_error_unsupported("laplacian_apply_opt", element_type);
        }
    }

    return SFEM_FAILURE;
}

int laplacian_crs_sym(smesh::ElemType                    element_type,
                      const ptrdiff_t                    nelements,
                      const ptrdiff_t                    nnodes,
                      idx_t **const SFEM_RESTRICT        elements,
                      geom_t **const SFEM_RESTRICT       points,
                      const count_t *const SFEM_RESTRICT rowptr,
                      const idx_t *const SFEM_RESTRICT   colidx,
                      real_t *const SFEM_RESTRICT        diag,
                      real_t *const SFEM_RESTRICT        offdiag) {
    if (sfem::is_semistructured_type(element_type)) {
        return laplacian_error_unsupported("laplacian_crs_sym", element_type);
    }

    switch (element_type) {
        case smesh::HEX8: {
            return hex8_laplacian_crs_sym(nelements, nnodes, elements, points, rowptr, colidx, diag, offdiag);
        }
        default: {
            return laplacian_error_unsupported("laplacian_crs_sym", element_type);
        }
    }

    return SFEM_FAILURE;
}
