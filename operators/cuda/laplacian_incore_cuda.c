#include "laplacian_incore_cuda.h"

#include "macro_tet4_laplacian_incore_cuda.h"
#include "tet10_laplacian_incore_cuda.h"
#include "tet4_laplacian_incore_cuda.h"

#include <mpi.h>
#include <stdio.h>

int cuda_incore_laplacian_init(const enum ElemType element_type,
                               cuda_incore_laplacian_t *ctx,
                               const ptrdiff_t nelements,
                               idx_t **const SFEM_RESTRICT elements,
                               geom_t **const SFEM_RESTRICT points) {
    switch (element_type) {
        case TET4: {
            return tet4_cuda_incore_laplacian_init(ctx, nelements, elements, points);
        }
        case MACRO_TET4: {
            return macro_tet4_cuda_incore_laplacian_init(ctx, nelements, elements, points);
        }
        case TET10: {
            return tet10_cuda_incore_laplacian_init(ctx, nelements, elements, points);
        }
        default: {
            fprintf(stderr, "Invalid element type %d\n (%s %s:%d)", ctx->element_type, __FUNCTION__, __FILE__, __LINE__);
            fflush(stderr);
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
}

int cuda_incore_laplacian_destroy(cuda_incore_laplacian_t *ctx) {
    switch (ctx->element_type) {
        case TET4: {
            return tet4_cuda_incore_laplacian_destroy(ctx);
        }
        case MACRO_TET4: {
            return macro_tet4_cuda_incore_laplacian_destroy(ctx);
        }
        case TET10: {
            return tet10_cuda_incore_laplacian_destroy(ctx);
        }
        default: {
            fprintf(stderr, "Invalid element type %d\n (%s %s:%d)", ctx->element_type, __FUNCTION__, __FILE__, __LINE__);
            fflush(stderr);
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
}
int cuda_incore_laplacian_apply(cuda_incore_laplacian_t *ctx,
                                const real_t *const d_x,
                                real_t *const d_y) {
    switch (ctx->element_type) {
        case TET4: {
            return tet4_cuda_incore_laplacian_apply(ctx, d_x, d_y);
        }
        case MACRO_TET4: {
            return macro_tet4_cuda_incore_laplacian_apply(ctx, d_x, d_y);
        }
        case TET10: {
            return tet10_cuda_incore_laplacian_apply(ctx, d_x, d_y);
        }
        default: {
            fprintf(stderr, "Invalid element type %d\n (%s %s:%d)", ctx->element_type, __FUNCTION__, __FILE__, __LINE__);
            fflush(stderr);
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
}

int cuda_incore_laplacian_diag(cuda_incore_laplacian_t *ctx, real_t *const d_t) {
    switch (ctx->element_type) {
        // case TET4: {
        // 	return tet4_cuda_incore_laplacian_diag(ctx, d_t);
        // }
        case MACRO_TET4: {
            return macro_tet4_cuda_incore_laplacian_diag(ctx, d_t);
        }
        // case TET10: {
        // 	return tet10_cuda_incore_laplacian_diag(ctx, d_t);
        // }
        default: {
            // for the moment we gracefully decline
            // assert(0);
            // MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
}