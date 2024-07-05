#include "laplacian_incore_cuda.h"

// #include "macro_tet4_laplacian_incore_cuda.h"
// #include "tet10_laplacian_incore_cuda.h"
#include "tet4_laplacian_incore_cuda.h"

#include <stdio.h>
#include <mpi.h>

int cu_laplacian_apply(const enum ElemType element_type,
                       const ptrdiff_t nelements,
                       const idx_t *const SFEM_RESTRICT elements,
                       const void *const SFEM_RESTRICT fff,
                       const enum RealType real_type_xy,
                       const void *const x,
                       void *const y,
                       void *stream) {
    switch (element_type) {
        case TET4: {
            return cu_tet4_laplacian_apply(nelements, elements, fff, real_type_xy, x, y, stream);
        }
        // case MACRO_TET4: {
        //     return cu_macro_tet4_laplacian_apply(nelements, elements, fff, real_type_xy,
        //     x, y, stream);
        // }
        // case TET10: {
        //     return cu_tet10_cu_laplacian_apply(nelements, elements, fff, real_type_xy, x,
        //     y, stream);
        // }
        default: {
            fprintf(stderr,
                    "cu_laplacian_apply: Invalid element type %d\n (%s:%d)",
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
                      const idx_t *const SFEM_RESTRICT elements,
                      const void *const SFEM_RESTRICT fff,
                      const enum RealType real_type_xy,
                      void *const diag,
                      void *stream) {
    switch (element_type) {
        case TET4: {
            return cu_tet4_laplacian_diag(nelements, elements, fff, real_type_xy, diag, stream);
        }
        // case MACRO_TET4: {
        //     return cu_macro_tet4_laplacian_diag(nelements, fff, real_type_xy, diag, stream);
        // }
        // case TET10: {
        // 	return cu_tet10_laplacian_diag(nelements, fff, real_type_xy, diag, stream);
        // }
        default: {
            fprintf(stderr,
                    "cu_laplacian_diag: Invalid element type %d\n (%s:%d)",
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
