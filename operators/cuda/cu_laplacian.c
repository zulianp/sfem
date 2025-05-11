#include "cu_laplacian.h"

#include "cu_hex8_laplacian.h"
#include "cu_macro_tet4_laplacian.h"
#include "cu_tet10_laplacian.h"
#include "cu_tet4_laplacian.h"

#include <mpi.h>
#include <stdio.h>

int cu_laplacian_apply(const enum ElemType             element_type,
                       const ptrdiff_t                 nelements,
                       idx_t **const SFEM_RESTRICT     elements,
                       const ptrdiff_t                 fff_stride,
                       const void *const SFEM_RESTRICT fff,
                       const enum RealType             real_type_xy,
                       const void *const SFEM_RESTRICT x,
                       void *const SFEM_RESTRICT       y,
                       void                           *stream) {
    switch (element_type) {
        case TET4: {
            return cu_tet4_laplacian_apply(nelements, elements, fff_stride, fff, real_type_xy, x, y, stream);
        }
        case TET10: {
            return cu_tet10_laplacian_apply(nelements, elements, fff_stride, fff, real_type_xy, x, y, stream);
        }
        case MACRO_TET4: {
            return cu_macro_tet4_laplacian_apply(nelements, elements, fff_stride, fff, real_type_xy, x, y, stream);
        }
        case HEX8: {
            return cu_affine_hex8_laplacian_apply(nelements, elements, fff_stride, fff, real_type_xy, x, y, stream);
        }
        default: {
            SFEM_ERROR("cu_laplacian_apply: Invalid element type %s (code = %d)\n", type_to_string(element_type), element_type);
            return SFEM_FAILURE;
        }
    }
}

int cu_laplacian_diag(const enum ElemType             element_type,
                      const ptrdiff_t                 nelements,
                      idx_t **const SFEM_RESTRICT     elements,
                      const ptrdiff_t                 fff_stride,
                      const void *const SFEM_RESTRICT fff,
                      const enum RealType             real_type_xy,
                      void *const SFEM_RESTRICT       diag,
                      void                           *stream) {
    switch (element_type) {
        case TET4: {
            return cu_tet4_laplacian_diag(nelements, elements, fff_stride, fff, real_type_xy, diag, stream);
        }
        // case TET10: {
        //  return cu_tet10_laplacian_diag(nelements, elements, fff_stride, fff, real_type_xy, diag,
        // stream);
        // }
        case MACRO_TET4: {
            return cu_macro_tet4_laplacian_diag(nelements, elements, fff_stride, fff, real_type_xy, diag, stream);
        }
        default: {
            SFEM_ERROR("cu_laplacian_diag: Invalid element type %s (code = %d)\n", type_to_string(element_type), element_type);
            return SFEM_FAILURE;
        }
    }
}

int cu_laplacian_crs(const enum ElemType                element_type,
                     const ptrdiff_t                    nelements,
                     idx_t **const SFEM_RESTRICT        elements,
                     const ptrdiff_t                    fff_stride,
                     const void *const SFEM_RESTRICT    fff,
                     const count_t *const SFEM_RESTRICT rowptr,
                     const idx_t *const SFEM_RESTRICT   colidx,
                     const enum RealType                real_type,
                     void *const SFEM_RESTRICT          values,
                     void                              *stream) {
    switch (element_type) {
        case TET4: {
            return cu_tet4_laplacian_crs(nelements, elements, fff_stride, fff, rowptr, colidx, real_type, values, stream);
        }
        // case TET10: {
        //  return cu_tet10_laplacian_crs(nelements, elements, fff_stride, fff, rowptr, colidx, real_type,
        //  values,
        // stream);
        // }
        case MACRO_TET4: {
            return cu_macro_tet4_laplacian_crs(
                    nelements, elements, fff_stride, fff, rowptr, colidx, real_type, values, stream);
        }
        default: {
            SFEM_ERROR("cu_laplacian_diag: Invalid element type %s (code = %d)\n", type_to_string(element_type), element_type);
            return SFEM_FAILURE;
        }
    }
}

int cu_laplacian_crs_sym(const enum ElemType                element_type,
                         const ptrdiff_t                    nelements,
                         idx_t **const SFEM_RESTRICT        elements,
                         const ptrdiff_t                    fff_stride,
                         const void *const SFEM_RESTRICT    fff,
                         const count_t *const SFEM_RESTRICT rowptr,
                         const idx_t *const SFEM_RESTRICT   colidx,
                         const enum RealType                real_type,
                         void *const SFEM_RESTRICT          diag,
                         void *const SFEM_RESTRICT          offdiag,
                         void                              *stream) {
    switch (element_type) {
        case HEX8: {
            return cu_affine_hex8_laplacian_crs_sym(
                    nelements, elements, fff_stride, fff, rowptr, colidx, real_type, diag, offdiag, stream);
        }

        default: {
            SFEM_ERROR("cu_laplacian_crs_sym: Invalid element type %s (code = %d)\n", type_to_string(element_type), element_type);
            return SFEM_FAILURE;
        }
    }
}
