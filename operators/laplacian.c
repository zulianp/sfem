
#include "tet10_laplacian.h"
#include "tet4_laplacian.h"
#include "tri3_laplacian.h"
#include "tri6_laplacian.h"

#include "macro_tri3_laplacian.h"
#include "macro_tet4_laplacian.h"

#include "sfem_defs.h"

#include <mpi.h>

void laplacian_assemble_value(int element_type,
                              const ptrdiff_t nelements,
                              const ptrdiff_t nnodes,
                              idx_t **const SFEM_RESTRICT elems,
                              geom_t **const SFEM_RESTRICT xyz,
                              const real_t *const SFEM_RESTRICT u,
                              real_t *const SFEM_RESTRICT value) {
    switch (element_type) {
        case TET4: {
            tet4_laplacian_assemble_value(nelements, nnodes, elems, xyz, u, value);
            break;
        }
        case TET10: {
            tet10_laplacian_assemble_value(nelements, nnodes, elems, xyz, u, value);
            break;
        }
        default: {
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void laplacian_assemble_gradient(int element_type,
                                 const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case TET4: {
            tet4_laplacian_assemble_gradient(nelements, nnodes, elems, xyz, u, values);
            break;
        }
        case TET10: {
            tet10_laplacian_assemble_gradient(nelements, nnodes, elems, xyz, u, values);
            break;
        }
        case MACRO_TET4: {
            macro_tet4_laplacian_apply(nelements, nnodes, elems, xyz, u, values);
            break;
        }
        default: {
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void laplacian_assemble_hessian(int element_type,
                                const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                const count_t *const SFEM_RESTRICT rowptr,
                                const idx_t *const SFEM_RESTRICT colidx,
                                real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_laplacian_assemble_hessian(nelements, nnodes, elems, xyz, rowptr, colidx, values);
            break;
        }
        case TRI6: {
            tri6_laplacian_assemble_hessian(nelements, nnodes, elems, xyz, rowptr, colidx, values);
            break;
        }
        case TET4: {
            tet4_laplacian_assemble_hessian(nelements, nnodes, elems, xyz, rowptr, colidx, values);
            break;
        }
        case TET10: {
            tet10_laplacian_assemble_hessian(nelements, nnodes, elems, xyz, rowptr, colidx, values);
            break;
        }
        default: {
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void laplacian_apply(int element_type,
                     const ptrdiff_t nelements,
                     const ptrdiff_t nnodes,
                     idx_t **const SFEM_RESTRICT elems,
                     geom_t **const SFEM_RESTRICT xyz,
                     const real_t *const SFEM_RESTRICT u,
                     real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case TET4: {
            tet4_laplacian_apply(nelements, nnodes, elems, xyz, u, values);
            break;
        }
        case TET10: {
            tet10_laplacian_apply(nelements, nnodes, elems, xyz, u, values);
            break;
        }
        case MACRO_TET4: {
            macro_tet4_laplacian_apply(nelements, nnodes, elems, xyz, u, values);
            break;
        }
        case MACRO_TRI3: {
            macro_tri3_laplacian_apply(nelements, nnodes, elems, xyz, u, values);
            break;
        }
        default: {
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}
