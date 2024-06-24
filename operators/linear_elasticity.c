#include "linear_elasticity.h"

#include "macro_tet4_linear_elasticity.h"
#include "tet10_linear_elasticity.h"
#include "tet4_linear_elasticity.h"
#include "tri3_linear_elasticity.h"

#include <mpi.h>
#include <stdio.h>

int linear_elasticity_assemble_value_soa(const enum ElemType element_type,
                                         const ptrdiff_t nelements,
                                         const ptrdiff_t nnodes,
                                         idx_t **const SFEM_RESTRICT elements,
                                         geom_t **const SFEM_RESTRICT points,
                                         const real_t mu,
                                         const real_t lambda,
                                         const real_t **const SFEM_RESTRICT u,
                                         real_t *const SFEM_RESTRICT value) {
    switch (element_type) {
        case TRI3: {
            tri3_linear_elasticity_value(
                    nelements, nnodes, elements, points, mu, lambda, 1, u[0], u[1], value);
            break;
        }
        // case TET4: {
        //     tet4_linear_elasticity_assemble_value_soa(
        //         nelements, nnodes, elements, points, mu, lambda, u, value);
        //     break;
        // }
        default: {
            fprintf(stderr,
                    "linear_elasticity_assemble_value_soa not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    return -1;
}

int linear_elasticity_apply_soa(const enum ElemType element_type,
                                const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elements,
                                geom_t **const SFEM_RESTRICT points,
                                const real_t mu,
                                const real_t lambda,
                                const real_t **const SFEM_RESTRICT u,
                                real_t **const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_linear_elasticity_apply(nelements,
                                         nnodes,
                                         elements,
                                         points,
                                         mu,
                                         lambda,
                                         1,
                                         u[0],
                                         u[1],
                                         1,
                                         values[0],
                                         values[1]);
            break;
        }
        case TET4: {
            tet4_linear_elasticity_apply(nelements,
                                         nnodes,
                                         elements,
                                         points,
                                         mu,
                                         lambda,
                                         1,
                                         u[0],
                                         u[1],
                                         u[2],
                                         1,
                                         values[0],
                                         values[1],
                                         values[2]);
            break;
        }
        default: {
            fprintf(stderr,
                    "linear_elasticity_apply_soa not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    return -1;
}

int linear_elasticity_assemble_value_aos(const enum ElemType element_type,
                                         const ptrdiff_t nelements,
                                         const ptrdiff_t nnodes,
                                         idx_t **const SFEM_RESTRICT elements,
                                         geom_t **const SFEM_RESTRICT points,
                                         const real_t mu,
                                         const real_t lambda,
                                         const real_t *const SFEM_RESTRICT u,
                                         real_t *const SFEM_RESTRICT value) {
    switch (element_type) {
        case TRI3: {
            tri3_linear_elasticity_value(
                    nelements, nnodes, elements, points, mu, lambda, 2, &u[0], &u[1], value);
            break;
        }
        case TET4: {
            return tet4_linear_elasticity_value(
                    nelements, nnodes, elements, points, mu, lambda, 3, &u[0], &u[1], &u[2], value);
        }
        default: {
            fprintf(stderr,
                    "linear_elasticity_assemble_value_aos not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    return -1;
}

int linear_elasticity_assemble_gradient_aos(const enum ElemType element_type,
                                            const ptrdiff_t nelements,
                                            const ptrdiff_t nnodes,
                                            idx_t **const SFEM_RESTRICT elements,
                                            geom_t **const SFEM_RESTRICT points,
                                            const real_t mu,
                                            const real_t lambda,
                                            const real_t *const SFEM_RESTRICT u,
                                            real_t *const SFEM_RESTRICT values) {
    return linear_elasticity_apply_aos(
            element_type, nelements, nnodes, elements, points, mu, lambda, u, values);
}

int linear_elasticity_assemble_hessian_aos(const enum ElemType element_type,
                                           const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const SFEM_RESTRICT elements,
                                           geom_t **const SFEM_RESTRICT points,
                                           const real_t mu,
                                           const real_t lambda,
                                           const count_t *const SFEM_RESTRICT rowptr,
                                           const idx_t *const SFEM_RESTRICT colidx,
                                           real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_linear_elasticity_hessian_aos(
                    nelements, nnodes, elements, points, mu, lambda, rowptr, colidx, values);
            break;
        }
        case TET4: {
            return tet4_linear_elasticity_hessian(
                    nelements, nnodes, elements, points, mu, lambda, rowptr, colidx, values);
        }
        default: {
            fprintf(stderr,
                    "linear_elasticity_assemble_hessian_aos not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    return -1;
}

int linear_elasticity_assemble_diag_aos(const enum ElemType element_type,
                                        const ptrdiff_t nelements,
                                        const ptrdiff_t nnodes,
                                        idx_t **const SFEM_RESTRICT elements,
                                        geom_t **const SFEM_RESTRICT points,
                                        const real_t mu,
                                        const real_t lambda,
                                        real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        // case TRI3: {
        //     tri3_linear_elasticity_assemble_diag(
        //         nelements, nnodes, elements, points, mu, lambda, values);
        //     break;
        // }
        case TET4: {
            return tet4_linear_elasticity_diag(nelements,
                                               nnodes,
                                               elements,
                                               points,
                                               mu,
                                               lambda,
                                               3,
                                               &values[0],
                                               &values[1],
                                               &values[2]);
        }
        case TET10: {
            return tet10_linear_elasticity_diag(nelements,
                                                nnodes,
                                                elements,
                                                points,
                                                mu,
                                                lambda,
                                                3,
                                                &values[0],
                                                &values[1],
                                                &values[2]);
        }
        default: {
            fprintf(stderr,
                    "linear_elasticity_assemble_diag_aos not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    return -1;
}

int linear_elasticity_apply_aos(const enum ElemType element_type,
                                const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elements,
                                geom_t **const SFEM_RESTRICT points,
                                const real_t mu,
                                const real_t lambda,
                                const real_t *const SFEM_RESTRICT u,
                                real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_linear_elasticity_apply(nelements,
                                         nnodes,
                                         elements,
                                         points,
                                         mu,
                                         lambda,
                                         2,
                                         &u[0],
                                         &u[1],
                                         2,
                                         &values[0],
                                         &values[1]);
            break;
        }
        case TET4: {
            return tet4_linear_elasticity_apply(nelements,
                                                nnodes,
                                                elements,
                                                points,
                                                mu,
                                                lambda,
                                                3,
                                                &u[0],
                                                &u[1],
                                                &u[2],
                                                3,
                                                &values[0],
                                                &values[1],
                                                &values[2]);
        }
        case MACRO_TET4: {
            return macro_tet4_linear_elasticity_apply(nelements,
                                                      nnodes,
                                                      elements,
                                                      points,
                                                      mu,
                                                      lambda,
                                                      3,
                                                      &u[0],
                                                      &u[1],
                                                      &u[2],
                                                      3,
                                                      &values[0],
                                                      &values[1],
                                                      &values[2]);
        }
        case TET10: {
            return tet10_linear_elasticity_apply(nelements,
                                                 nnodes,
                                                 elements,
                                                 points,
                                                 mu,
                                                 lambda,
                                                 3,
                                                 &u[0],
                                                 &u[1],
                                                 &u[2],
                                                 3,
                                                 &values[0],
                                                 &values[1],
                                                 &values[2]);
        }
        default: {
            fprintf(stderr,
                    "linear_elasticity_apply_aos not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    return -1;
}

int linear_elasticity_assemble_hessian_soa(const enum ElemType element_type,
                                           const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const SFEM_RESTRICT elements,
                                           geom_t **const SFEM_RESTRICT points,
                                           const real_t mu,
                                           const real_t lambda,
                                           const count_t *const SFEM_RESTRICT rowptr,
                                           const idx_t *const SFEM_RESTRICT colidx,
                                           real_t **const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_linear_elasticity_assemble_hessian_soa(
                    nelements, nnodes, elements, points, mu, lambda, rowptr, colidx, values);
            break;
        }
        default: {
            fprintf(stderr,
                    "linear_elasticity_assemble_hessian_soa not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    return -1;
}
