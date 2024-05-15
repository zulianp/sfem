#include "linear_elasticity.h"

#include "macro_tet4_linear_elasticity.h"
#include "tet10_linear_elasticity.h"
#include "tet4_linear_elasticity.h"
#include "tri3_linear_elasticity.h"

#include <mpi.h>

void linear_elasticity_assemble_value_soa(const enum ElemType element_type,
                                          const ptrdiff_t nelements,
                                          const ptrdiff_t nnodes,
                                          idx_t **const SFEM_RESTRICT elems,
                                          geom_t **const SFEM_RESTRICT xyz,
                                          const real_t mu,
                                          const real_t lambda,
                                          const real_t **const SFEM_RESTRICT u,
                                          real_t *const SFEM_RESTRICT value) {
    switch (element_type) {
        case TRI3: {
            tri3_linear_elasticity_assemble_value_soa(
                nelements, nnodes, elems, xyz, mu, lambda, u, value);
            break;
        }
        // case TET4: {
        //     tet4_linear_elasticity_assemble_value_soa(
        //         nelements, nnodes, elems, xyz, mu, lambda, u, value);
        //     break;
        // }
        default: {
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void linear_elasticity_assemble_gradient_soa(const enum ElemType element_type,
                                             const ptrdiff_t nelements,
                                             const ptrdiff_t nnodes,
                                             idx_t **const SFEM_RESTRICT elems,
                                             geom_t **const SFEM_RESTRICT xyz,
                                             const real_t mu,
                                             const real_t lambda,
                                             const real_t **const SFEM_RESTRICT u,
                                             real_t **const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_linear_elasticity_assemble_gradient_soa(
                nelements, nnodes, elems, xyz, mu, lambda, u, values);
            break;
        }
        // case TET4: {
        //     tri4_linear_elasticity_assemble_gradient_soa(
        //         nelements, nnodes, elems, xyz, mu, lambda, u, values);
        //     break;
        // }
        default: {
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void linear_elasticity_assemble_hessian_soa(const enum ElemType element_type,
                                            const ptrdiff_t nelements,
                                            const ptrdiff_t nnodes,
                                            idx_t **const SFEM_RESTRICT elems,
                                            geom_t **const SFEM_RESTRICT xyz,
                                            const real_t mu,
                                            const real_t lambda,
                                            const count_t *const SFEM_RESTRICT rowptr,
                                            const idx_t *const SFEM_RESTRICT colidx,
                                            real_t **const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_linear_elasticity_assemble_hessian_soa(
                nelements, nnodes, elems, xyz, mu, lambda, rowptr, colidx, values);
            break;
        }
        // case TET4: {
        //     tet4_linear_elasticity_assemble_hessian_soa(
        //         nelements, nnodes, elems, xyz, mu, lambda, rowptr, colidx, values);
        //     break;
        // }
        default: {
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void linear_elasticity_apply_soa(const enum ElemType element_type,
                                 const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 const real_t mu,
                                 const real_t lambda,
                                 const real_t **const SFEM_RESTRICT u,
                                 real_t **const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_linear_elasticity_apply_soa(nelements, nnodes, elems, xyz, mu, lambda, u, values);
            break;
        }
        case TET4: {
            tet4_linear_elasticity_apply_soa(nelements, nnodes, elems, xyz, mu, lambda, u, values);
            break;
        }
        default: {
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void linear_elasticity_assemble_value_aos(const enum ElemType element_type,
                                          const ptrdiff_t nelements,
                                          const ptrdiff_t nnodes,
                                          idx_t **const SFEM_RESTRICT elems,
                                          geom_t **const SFEM_RESTRICT xyz,
                                          const real_t mu,
                                          const real_t lambda,
                                          const real_t *const SFEM_RESTRICT u,
                                          real_t *const SFEM_RESTRICT value) {
    switch (element_type) {
        case TRI3: {
            tri3_linear_elasticity_assemble_value_aos(
                nelements, nnodes, elems, xyz, mu, lambda, u, value);
            break;
        }
        case TET4: {
            tet4_linear_elasticity_assemble_value_aos(
                nelements, nnodes, elems, xyz, mu, lambda, u, value);
            break;
        }
        default: {
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void linear_elasticity_assemble_gradient_aos(const enum ElemType element_type,
                                             const ptrdiff_t nelements,
                                             const ptrdiff_t nnodes,
                                             idx_t **const SFEM_RESTRICT elems,
                                             geom_t **const SFEM_RESTRICT xyz,
                                             const real_t mu,
                                             const real_t lambda,
                                             const real_t *const SFEM_RESTRICT u,
                                             real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_linear_elasticity_assemble_gradient_aos(
                nelements, nnodes, elems, xyz, mu, lambda, u, values);
            break;
        }
        case TET4: {
            tet4_linear_elasticity_assemble_gradient_aos(
                nelements, nnodes, elems, xyz, mu, lambda, u, values);
            break;
        }
        case MACRO_TET4: {
            macro_tet4_linear_elasticity_apply_aos(
                nelements, nnodes, elems, xyz, mu, lambda, u, values);
            break;
        }
        case TET10: {
            tet10_linear_elasticity_apply_aos(nelements, nnodes, elems, xyz, mu, lambda, u, values);
            break;
        }
        default: {
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void linear_elasticity_assemble_hessian_aos(const enum ElemType element_type,
                                            const ptrdiff_t nelements,
                                            const ptrdiff_t nnodes,
                                            idx_t **const SFEM_RESTRICT elems,
                                            geom_t **const SFEM_RESTRICT xyz,
                                            const real_t mu,
                                            const real_t lambda,
                                            const count_t *const SFEM_RESTRICT rowptr,
                                            const idx_t *const SFEM_RESTRICT colidx,
                                            real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_linear_elasticity_assemble_hessian_aos(
                nelements, nnodes, elems, xyz, mu, lambda, rowptr, colidx, values);
            break;
        }
        case TET4: {
            tet4_linear_elasticity_assemble_hessian_aos(
                nelements, nnodes, elems, xyz, mu, lambda, rowptr, colidx, values);
            break;
        }
        default: {
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void linear_elasticity_assemble_diag_aos(const enum ElemType element_type,
                                             const ptrdiff_t nelements,
                                             const ptrdiff_t nnodes,
                                             idx_t **const SFEM_RESTRICT elems,
                                             geom_t **const SFEM_RESTRICT xyz,
                                             const real_t mu,
                                             const real_t lambda,
                                             real_t *const SFEM_RESTRICT values)
{
    switch (element_type) {
        // case TRI3: {
        //     tri3_linear_elasticity_assemble_diag(
        //         nelements, nnodes, elems, xyz, mu, lambda, values);
        //     break;
        // }
        case TET4: {
            tet4_linear_elasticity_assemble_diag_aos(
                nelements, nnodes, elems, xyz, mu, lambda, values);
            break;
        }
        case TET10: {
            tet10_linear_elasticity_assemble_diag_aos(
                nelements, nnodes, elems, xyz, mu, lambda, values);
            break;
        }
        default: {
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void linear_elasticity_apply_aos(const enum ElemType element_type,
                                 const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 const real_t mu,
                                 const real_t lambda,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_linear_elasticity_apply_aos(nelements, nnodes, elems, xyz, mu, lambda, u, values);
            break;
        }
        case TET4: {
            tet4_linear_elasticity_apply_aos(nelements, nnodes, elems, xyz, mu, lambda, u, values);
            break;
        }
        case MACRO_TET4: {
            macro_tet4_linear_elasticity_apply_aos(
                nelements, nnodes, elems, xyz, mu, lambda, u, values);
            break;
        }
        case TET10: {
            tet10_linear_elasticity_apply_aos(nelements, nnodes, elems, xyz, mu, lambda, u, values);
            break;
        }
        default: {
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}
