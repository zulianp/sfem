#include "cu_laplacian.hpp"

#include "cu_hex8_laplacian.hpp"
#include "cu_macro_tet4_laplacian.hpp"
#include "cu_sshex8_laplacian.hpp"
#include "cu_tet10_laplacian.hpp"
#include "cu_tet4_laplacian.hpp"

#include <mpi.h>
#include <stdio.h>

int cu_laplacian_apply(const smesh::ElemType             element_type,
                       const ptrdiff_t                 nelements,
                       idx_t **const SFEM_RESTRICT     elements,
                       const ptrdiff_t                 fff_stride,
                       const void *const SFEM_RESTRICT fff,
                       const enum smesh::PrimitiveType             real_type_xy,
                       const void *const SFEM_RESTRICT x,
                       void *const SFEM_RESTRICT       y,
                       void                           *stream) {
    if (sfem::is_semistructured_type(element_type)) {
        const int level = smesh::semistructured_level(element_type);
        return cu_affine_sshex8_laplacian_apply(level, nelements, elements, fff_stride, fff, real_type_xy, x, y, stream);
    }

    switch (element_type) {
        case smesh::TET4: {
            return cu_tet4_laplacian_apply(nelements, elements, fff_stride, fff, real_type_xy, x, y, stream);
        }
        case smesh::TET10: {
            return cu_tet10_laplacian_apply(nelements, elements, fff_stride, fff, real_type_xy, x, y, stream);
        }
        case smesh::MACRO_TET4: {
            return cu_macro_tet4_laplacian_apply(nelements, elements, fff_stride, fff, real_type_xy, x, y, stream);
        }
        case smesh::HEX8: {
            return cu_affine_hex8_laplacian_apply(nelements, elements, fff_stride, fff, real_type_xy, x, y, stream);
        }
        default: {
            SFEM_ERROR("cu_laplacian_apply: Invalid element type %s (code = %d)\n", type_to_string(element_type), element_type);
            return SFEM_FAILURE;
        }
    }
}

int cu_laplacian_diag(const smesh::ElemType             element_type,
                      const ptrdiff_t                 nelements,
                      idx_t **const SFEM_RESTRICT     elements,
                      const ptrdiff_t                 fff_stride,
                      const void *const SFEM_RESTRICT fff,
                      const enum smesh::PrimitiveType             real_type_xy,
                      void *const SFEM_RESTRICT       diag,
                      void                           *stream) {
    if (sfem::is_semistructured_type(element_type)) {
        const int level = smesh::semistructured_level(element_type);
        return cu_affine_sshex8_laplacian_diag(level, nelements, elements, fff_stride, fff, real_type_xy, diag, stream);
    }

    switch (element_type) {
        case smesh::TET4: {
            return cu_tet4_laplacian_diag(nelements, elements, fff_stride, fff, real_type_xy, diag, stream);
        }
        // case smesh::TET10: {
        //  return cu_tet10_laplacian_diag(nelements, elements, fff_stride, fff, real_type_xy, diag,
        // stream);
        // }
        case smesh::MACRO_TET4: {
            return cu_macro_tet4_laplacian_diag(nelements, elements, fff_stride, fff, real_type_xy, diag, stream);
        }
        default: {
            SFEM_ERROR("cu_laplacian_diag: Invalid element type %s (code = %d)\n", type_to_string(element_type), element_type);
            return SFEM_FAILURE;
        }
    }
}

int cu_laplacian_crs(const smesh::ElemType                element_type,
                     const ptrdiff_t                    nelements,
                     idx_t **const SFEM_RESTRICT        elements,
                     const ptrdiff_t                    fff_stride,
                     const void *const SFEM_RESTRICT    fff,
                     const count_t *const SFEM_RESTRICT rowptr,
                     const idx_t *const SFEM_RESTRICT   colidx,
                     const enum smesh::PrimitiveType                real_type,
                     void *const SFEM_RESTRICT          values,
                     void                              *stream) {
    switch (element_type) {
        case smesh::TET4: {
            return cu_tet4_laplacian_crs(nelements, elements, fff_stride, fff, rowptr, colidx, real_type, values, stream);
        }
        // case smesh::TET10: {
        //  return cu_tet10_laplacian_crs(nelements, elements, fff_stride, fff, rowptr, colidx, real_type,
        //  values,
        // stream);
        // }
        case smesh::MACRO_TET4: {
            return cu_macro_tet4_laplacian_crs(
                    nelements, elements, fff_stride, fff, rowptr, colidx, real_type, values, stream);
        }
        default: {
            SFEM_ERROR("cu_laplacian_diag: Invalid element type %s (code = %d)\n", type_to_string(element_type), element_type);
            return SFEM_FAILURE;
        }
    }
}

int cu_laplacian_crs_sym(const smesh::ElemType                element_type,
                         const ptrdiff_t                    nelements,
                         idx_t **const SFEM_RESTRICT        elements,
                         const ptrdiff_t                    fff_stride,
                         const void *const SFEM_RESTRICT    fff,
                         const count_t *const SFEM_RESTRICT rowptr,
                         const idx_t *const SFEM_RESTRICT   colidx,
                         const enum smesh::PrimitiveType                real_type,
                         void *const SFEM_RESTRICT          diag,
                         void *const SFEM_RESTRICT          offdiag,
                         void                              *stream) {
    switch (element_type) {
        case smesh::HEX8: {
            return cu_affine_hex8_laplacian_crs_sym(
                    nelements, elements, fff_stride, fff, rowptr, colidx, real_type, diag, offdiag, stream);
        }

        default: {
            SFEM_ERROR("cu_laplacian_crs_sym: Invalid element type %s (code = %d)\n", type_to_string(element_type), element_type);
            return SFEM_FAILURE;
        }
    }
}
