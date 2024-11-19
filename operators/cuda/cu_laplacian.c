#include "cu_laplacian.h"

#include "cu_hex8_laplacian.h"
#include "cu_macro_tet4_laplacian.h"
#include "cu_tet10_laplacian.h"
#include "cu_tet4_laplacian.h"

#include <mpi.h>
#include <stdio.h>

int cu_laplacian_apply(const enum ElemType element_type,
                       const ptrdiff_t nelements,
                       const ptrdiff_t stride,
                       const idx_t *const SFEM_RESTRICT elements,
                       const void *const SFEM_RESTRICT fff,
                       const enum RealType real_type_xy,
                       const void *const SFEM_RESTRICT x,
                       void *const SFEM_RESTRICT y,
                       void *stream) {
    switch (element_type) {
        case TET4: {
            return cu_tet4_laplacian_apply(
                    nelements, stride, elements, fff, real_type_xy, x, y, stream);
        }
        case TET10: {
            return cu_tet10_laplacian_apply(
                    nelements, stride, elements, fff, real_type_xy, x, y, stream);
        }
        case MACRO_TET4: {
            return cu_macro_tet4_laplacian_apply(
                    nelements, stride, elements, fff, real_type_xy, x, y, stream);
        }
        case HEX8: {
            return cu_affine_hex8_laplacian_apply(
                    nelements, stride, elements, fff, real_type_xy, x, y, stream);
        }
        default: {
            fprintf(stderr,
                    "cu_laplacian_apply: Invalid element type %s (code = %d)\n (%s:%d)",
                    type_to_string(element_type),
                    element_type,
                    __FILE__,
                    __LINE__);
            fflush(stderr);
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return SFEM_FAILURE;
        }
    }
}

int cu_laplacian_diag(const enum ElemType element_type,
                      const ptrdiff_t nelements,
                      const ptrdiff_t stride,
                      const idx_t *const SFEM_RESTRICT elements,
                      const void *const SFEM_RESTRICT fff,
                      const enum RealType real_type_xy,
                      void *const SFEM_RESTRICT diag,
                      void *stream) {
    switch (element_type) {
        case TET4: {
            return cu_tet4_laplacian_diag(
                    nelements, stride, elements, fff, real_type_xy, diag, stream);
        }
        // case TET10: {
        //  return cu_tet10_laplacian_diag(nelements, elements, fff, real_type_xy, diag,
        // stream);
        // }
        case MACRO_TET4: {
            return cu_macro_tet4_laplacian_diag(
                    nelements, stride, elements, fff, real_type_xy, diag, stream);
        }
        default: {
            fprintf(stderr,
                    "cu_laplacian_diag: Invalid element type %s (code = %d)\n (%s:%d)",
                    type_to_string(element_type),
                    element_type,
                    __FILE__,
                    __LINE__);
            fflush(stderr);
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return SFEM_FAILURE;
        }
    }
}

int cu_laplacian_crs(const enum ElemType element_type,
                     const ptrdiff_t nelements,
                     const ptrdiff_t stride,  // Stride for elements and fff
                     const idx_t *const SFEM_RESTRICT elements,
                     const void *const SFEM_RESTRICT fff,
                     const count_t *const SFEM_RESTRICT rowptr,
                     const idx_t *const SFEM_RESTRICT colidx,
                     const enum RealType real_type,
                     void *const SFEM_RESTRICT values,
                     void *stream) {
    switch (element_type) {
        case TET4: {
            return cu_tet4_laplacian_crs(
                    nelements, stride, elements, fff, rowptr, colidx, real_type, values, stream);
        }
        // case TET10: {
        //  return cu_tet10_laplacian_crs(nelements, elements, fff, rowptr, colidx, real_type,
        //  values,
        // stream);
        // }
        case MACRO_TET4: {
            return cu_macro_tet4_laplacian_crs(
                    nelements, stride, elements, fff, rowptr, colidx, real_type, values, stream);
        }
        default: {
            fprintf(stderr,
                    "cu_laplacian_diag: Invalid element type %s (code = %d)\n (%s:%d)",
                    type_to_string(element_type),
                    element_type,
                    __FILE__,
                    __LINE__);
            fflush(stderr);
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return SFEM_FAILURE;
        }
    }
}
