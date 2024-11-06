#include "linear_elasticity.h"

#include "hex8_linear_elasticity.h"
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
            return tri3_linear_elasticity_value(
                    nelements, nnodes, elements, points, mu, lambda, 1, u[0], u[1], value);
        }
        // case TET4: {
        //     return tet4_linear_elasticity_assemble_value_soa(
        //         nelements, nnodes, elements, points, mu, lambda, u, value);
        //
        // }
        default: {
            fprintf(stderr,
                    "linear_elasticity_assemble_value_soa not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    return SFEM_FAILURE;
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
            return tri3_linear_elasticity_apply(nelements,
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
        }
        case TET4: {
            return tet4_linear_elasticity_apply(nelements,
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
        }
        case HEX8: {
            int SFEM_HEX8_ASSUME_AFFINE = 0;
            SFEM_READ_ENV(SFEM_HEX8_ASSUME_AFFINE, atoi);
            if (SFEM_HEX8_ASSUME_AFFINE) {
                return affine_hex8_linear_elasticity_apply(nelements,
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
            } else {
                return hex8_linear_elasticity_apply(nelements,
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
            }
        }
        default: {
            fprintf(stderr,
                    "linear_elasticity_apply_soa not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    return SFEM_FAILURE;
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
            return tri3_linear_elasticity_value(
                    nelements, nnodes, elements, points, mu, lambda, 2, &u[0], &u[1], value);
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

    return SFEM_FAILURE;
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

int linear_elasticity_crs_aos(const enum ElemType element_type,
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
            return tri3_linear_elasticity_crs_aos(
                    nelements, nnodes, elements, points, mu, lambda, rowptr, colidx, values);
        }
        case TET4: {
            return tet4_linear_elasticity_crs(
                    nelements, nnodes, elements, points, mu, lambda, rowptr, colidx, values);
        }
        case TET10: {
            return tet10_linear_elasticity_crs(
                    nelements, nnodes, elements, points, mu, lambda, rowptr, colidx, values);
        }
        case MACRO_TET4: {
            return macro_tet4_linear_elasticity_crs(
                    nelements, nnodes, elements, points, mu, lambda, rowptr, colidx, values);
        }
        default: {
            fprintf(stderr,
                    "linear_elasticity_crs_aos not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    return SFEM_FAILURE;
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
        //     return tri3_linear_elasticity_assemble_diag(
        //         nelements, nnodes, elements, points, mu, lambda, values);
        //
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
        case MACRO_TET4: {
            return macro_tet4_linear_elasticity_diag(nelements,
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

    return SFEM_FAILURE;
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
            return tri3_linear_elasticity_apply(nelements,
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
        case HEX8: {
            int SFEM_HEX8_ASSUME_AFFINE = 0;
            SFEM_READ_ENV(SFEM_HEX8_ASSUME_AFFINE, atoi);

            if (SFEM_HEX8_ASSUME_AFFINE) {
                return affine_hex8_linear_elasticity_apply(nelements,
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

            } else {
                return hex8_linear_elasticity_apply(nelements,
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

    return SFEM_FAILURE;
}

int linear_elasticity_crs_soa(const enum ElemType element_type,
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
            return tri3_linear_elasticity_crs_soa(
                    nelements, nnodes, elements, points, mu, lambda, rowptr, colidx, values);
        }
        default: {
            fprintf(stderr,
                    "linear_elasticity_crs_soa not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    return SFEM_FAILURE;
}

int linear_elasticity_bsr(const enum ElemType element_type,
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
            return tet4_linear_elasticity_bsr(
                    nelements, nnodes, elements, points, mu, lambda, rowptr, colidx, values);
        }
        default: {
            fprintf(stderr,
                    "linear_elasticity_bsr is not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
            return SFEM_FAILURE;
        }
    }
}
