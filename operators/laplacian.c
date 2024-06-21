
#include "tet10_laplacian.h"
#include "tet4_laplacian.h"
#include "tri3_laplacian.h"
#include "tri6_laplacian.h"

#include "macro_tet4_laplacian.h"
#include "macro_tri3_laplacian.h"

#include "sfem_defs.h"

#include <mpi.h>

#include <stdio.h>

int laplacian_is_opt(int element_type) {
    return element_type == TET10 || element_type == TET4 || element_type == MACRO_TET4;
}

void laplacian_assemble_value(int element_type,
                              const ptrdiff_t nelements,
                              const ptrdiff_t nnodes,
                              idx_t **const SFEM_RESTRICT elements,
                              geom_t **const SFEM_RESTRICT points,
                              const real_t *const SFEM_RESTRICT u,
                              real_t *const SFEM_RESTRICT value) {
    switch (element_type) {
        case TET4: {
            tet4_laplacian_assemble_value(nelements, nnodes, elements, points, u, value);
            break;
        }
        case TET10: {
            tet10_laplacian_assemble_value(nelements, nnodes, elements, points, u, value);
            break;
        }
        default: {
            fprintf(stderr,
                    "laplacian_assemble_value not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void laplacian_apply(int element_type,
                     const ptrdiff_t nelements,
                     const ptrdiff_t nnodes,
                     idx_t **const SFEM_RESTRICT elements,
                     geom_t **const SFEM_RESTRICT points,
                     const real_t *const SFEM_RESTRICT u,
                     real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_laplacian_apply(nelements, nnodes, elements, points, u, values);
            break;
        }
        case TET4: {
            tet4_laplacian_apply(nelements, nnodes, elements, points, u, values);
            break;
        }
        case TET10: {
            tet10_laplacian_apply(nelements, nnodes, elements, points, u, values);
            break;
        }
        case MACRO_TET4: {
            macro_tet4_laplacian_apply(nelements, nnodes, elements, points, u, values);
            break;
        }
        case MACRO_TRI3: {
            macro_tri3_laplacian_apply(nelements, nnodes, elements, points, u, values);
            break;
        }
        default: {
            fprintf(stderr,
                    "laplacian_apply not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void laplacian_assemble_gradient(int element_type,
                                 const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT values) {
    laplacian_apply(element_type, nelements, nnodes, elements, points, u, values);
}

void laplacian_assemble_hessian(int element_type,
                                const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elements,
                                geom_t **const SFEM_RESTRICT points,
                                const count_t *const SFEM_RESTRICT rowptr,
                                const idx_t *const SFEM_RESTRICT colidx,
                                real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_laplacian_assemble_hessian(
                    nelements, nnodes, elements, points, rowptr, colidx, values);
            break;
        }
        case TRI6: {
            tri6_laplacian_assemble_hessian(
                    nelements, nnodes, elements, points, rowptr, colidx, values);
            break;
        }
        case TET4: {
            tet4_laplacian_assemble_hessian(
                    nelements, nnodes, elements, points, rowptr, colidx, values);
            break;
        }
        case TET10: {
            tet10_laplacian_assemble_hessian(
                    nelements, nnodes, elements, points, rowptr, colidx, values);
            break;
        }
        case MACRO_TET4: {
            macro_tet4_laplacian_assemble_hessian(
                    nelements, nnodes, elements, points, rowptr, colidx, values);
            break;
        }
        default: {
            fprintf(stderr,
                    "laplacian_assemble_hessian not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void laplacian_diag(int element_type,
                    const ptrdiff_t nelements,
                    const ptrdiff_t nnodes,
                    idx_t **const SFEM_RESTRICT elements,
                    geom_t **const SFEM_RESTRICT points,
                    real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        // case TRI3: {
        //     tri3_laplacian_diag(nelements, nnodes, elements, points, values);
        //     break;
        // }
        case TET4: {
            tet4_laplacian_diag(nelements, nnodes, elements, points, values);
            break;
        }
        case TET10: {
            tet10_laplacian_diag(nelements, nnodes, elements, points, values);
            break;
        }
        case MACRO_TET4: {
            macro_tet4_laplacian_diag(nelements, nnodes, elements, points, values);
            return;
        }
        // case MACRO_TRI3: {
        //     macro_tri3_laplacian_diag(nelements, nnodes, elements, points, values);
        //     break;
        // }
        default: {
            fprintf(stderr,
                    "laplacian_diag not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

int laplacian_apply_opt(int element_type,
                        const ptrdiff_t nelements,
                        idx_t **const SFEM_RESTRICT elements,
                        const jacobian_t *const SFEM_RESTRICT fff,
                        const real_t *const SFEM_RESTRICT u,
                        real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        // case TRI3: {
        //     tri3_laplacian_apply_opt(nelements, elements, fff, u, values);
        //     break;
        // }
        case TET4: {
            return tet4_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        // case TET10: {
        //     tet10_laplacian_apply_opt(nelements, elements, fff, u, values);
        //     break;
        // }
        case MACRO_TET4: {
            return macro_tet4_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        // case MACRO_TRI3: {
        //     macro_tri3_laplacian_apply_opt(nelements, elements, fff, u, values);
        //     break;
        // }
        default: {
            fprintf(stderr,
                    "laplacian_apply_opt not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    return -1;
}
