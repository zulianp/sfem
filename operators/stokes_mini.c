#include "stokes_mini.h"

#include "tet4_stokes_mini.h"
#include "tri3_stokes_mini.h"

#include <mpi.h>

// void stokes_mini_assemble_value_soa(const enum ElemType element_type,
//                                           const ptrdiff_t nelements,
//                                           const ptrdiff_t nnodes,
//                                           idx_t **const SFEM_RESTRICT elems,
//                                           geom_t **const SFEM_RESTRICT xyz,
//                                           const real_t mu,
//                                           const real_t lambda,
//                                           const real_t **const SFEM_RESTRICT u,
//                                           real_t *const SFEM_RESTRICT value) {
//     switch (element_type) {
//         case TRI3: {
//             tri3_stokes_mini_assemble_value_soa(
//                 nelements, nnodes, elems, xyz, mu, u, value);
//             break;
//         }
//         // case TET4: {
//         //     tet4_stokes_mini_assemble_value_soa(
//         //         nelements, nnodes, elems, xyz, mu, u, value);
//         //     break;
//         // }
//         default: {
//             MPI_Abort(MPI_COMM_WORLD, -1);
//         }
//     }
// }

// void stokes_mini_assemble_gradient_soa(const enum ElemType element_type,
//                                              const ptrdiff_t nelements,
//                                              const ptrdiff_t nnodes,
//                                              idx_t **const SFEM_RESTRICT elems,
//                                              geom_t **const SFEM_RESTRICT xyz,
//                                              const real_t mu,
//                                              const real_t lambda,
//                                              const real_t **const SFEM_RESTRICT u,
//                                              real_t **const SFEM_RESTRICT values) {
//     switch (element_type) {
//         case TRI3: {
//             tri3_stokes_mini_assemble_gradient_soa(
//                 nelements, nnodes, elems, xyz, mu, u, values);
//             break;
//         }
//         // case TET4: {
//         //     tri4_stokes_mini_assemble_gradient_soa(
//         //         nelements, nnodes, elems, xyz, mu, u, values);
//         //     break;
//         // }
//         default: {
//             MPI_Abort(MPI_COMM_WORLD, -1);
//         }
//     }
// }

void stokes_mini_assemble_hessian_soa(const enum ElemType element_type,
                                      const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const SFEM_RESTRICT elems,
                                      geom_t **const SFEM_RESTRICT xyz,
                                      const real_t mu,
                                      const count_t *const SFEM_RESTRICT rowptr,
                                      const idx_t *const SFEM_RESTRICT colidx,
                                      real_t **const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_stokes_mini_assemble_hessian_soa(
                nelements, nnodes, elems, xyz, mu, rowptr, colidx, values);
            break;
        }
        case TET4: {
            tet4_stokes_mini_assemble_hessian_soa(
                nelements, nnodes, elems, xyz, mu, rowptr, colidx, values);
            break;
        }
        default: {
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

// void stokes_mini_apply_soa(const enum ElemType element_type,
//                                  const ptrdiff_t nelements,
//                                  const ptrdiff_t nnodes,
//                                  idx_t **const SFEM_RESTRICT elems,
//                                  geom_t **const SFEM_RESTRICT xyz,
//                                  const real_t mu,
//                                  const real_t lambda,
//                                  const real_t **const SFEM_RESTRICT u,
//                                  real_t **const SFEM_RESTRICT values) {
//     switch (element_type) {
//         case TRI3: {
//             tri3_stokes_mini_apply_soa(nelements, nnodes, elems, xyz, mu, u, values);
//             break;
//         }
//         // case TET4: {
//         //     tet4_stokes_mini_apply_soa(nelements, nnodes, elems, xyz, mu, u,
//         //     values); break;
//         // }
//         default: {
//             MPI_Abort(MPI_COMM_WORLD, -1);
//         }
//     }
// }

// void stokes_mini_assemble_value_aos(const enum ElemType element_type,
//                                           const ptrdiff_t nelements,
//                                           const ptrdiff_t nnodes,
//                                           idx_t **const SFEM_RESTRICT elems,
//                                           geom_t **const SFEM_RESTRICT xyz,
//                                           const real_t mu,
//                                           const real_t lambda,
//                                           const real_t *const SFEM_RESTRICT u,
//                                           real_t *const SFEM_RESTRICT value) {
//     switch (element_type) {
//         case TRI3: {
//             tri3_stokes_mini_assemble_value_aos(
//                 nelements, nnodes, elems, xyz, mu, u, value);
//             break;
//         }
//         case TET4: {
//             tet4_stokes_mini_assemble_value_aos(
//                 nelements, nnodes, elems, xyz, mu, u, value);
//             break;
//         }
//         default: {
//             MPI_Abort(MPI_COMM_WORLD, -1);
//         }
//     }
// }

void stokes_mini_assemble_gradient_aos(const enum ElemType element_type,
                                             const ptrdiff_t nelements,
                                             const ptrdiff_t nnodes,
                                             idx_t **const SFEM_RESTRICT elems,
                                             geom_t **const SFEM_RESTRICT xyz,
                                             const real_t mu,
                                             const real_t *const SFEM_RESTRICT u,
                                             real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_stokes_mini_assemble_gradient_aos(
                nelements, nnodes, elems, xyz, mu, u, values);
            break;
        }
        // case TET4: {
        //     tet4_stokes_mini_assemble_gradient_aos(
        //         nelements, nnodes, elems, xyz, mu, u, values);
        //     break;
        // }
        default: {
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void stokes_mini_assemble_hessian_aos(const enum ElemType element_type,
                                      const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const SFEM_RESTRICT elems,
                                      geom_t **const SFEM_RESTRICT xyz,
                                      const real_t mu,
                                      const count_t *const SFEM_RESTRICT rowptr,
                                      const idx_t *const SFEM_RESTRICT colidx,
                                      real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_stokes_mini_assemble_hessian_aos(
                nelements, nnodes, elems, xyz, mu, rowptr, colidx, values);
            break;
        }
        case TET4: {
            tet4_stokes_mini_assemble_hessian_aos(
                nelements, nnodes, elems, xyz, mu, rowptr, colidx, values);
            break;
        }
        default: {
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void stokes_mini_apply_aos(const enum ElemType element_type,
                                 const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 const real_t mu,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            tri3_stokes_mini_apply_aos(nelements, nnodes, elems, xyz, mu, u, values);
            break;
        }
        // case TET4: {
        //     tet4_stokes_mini_apply_aos(nelements, nnodes, elems, xyz, mu, u, values);
        //     break;
        // }
        default: {
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void stokes_mini_assemble_rhs_soa(enum ElemType element_type,
                                       const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const elems,
                                       geom_t **const points,
                                       const real_t mu,
                                       const real_t rho,
                                       real_t **SFEM_RESTRICT forcing,
                                       real_t **const SFEM_RESTRICT rhs)
{
        switch (element_type) {
            case TRI3: {
                tri3_stokes_mini_assemble_rhs_soa(nelements, nnodes, elems, points, mu, rho, forcing, rhs);
                break;
            }
            case TET4: {
                tet4_stokes_mini_assemble_rhs_soa(nelements, nnodes, elems, points, mu, rho, forcing, rhs);
                break;
            }
            default: {
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }
}



void stokes_mini_assemble_rhs_aos(enum ElemType element_type,
                                       const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const elems,
                                       geom_t **const points,
                                       const real_t mu,
                                       const real_t rho,
                                       real_t **SFEM_RESTRICT forcing,
                                       real_t *const SFEM_RESTRICT rhs)
{
    switch (element_type) {
        case TRI3: {
            tri3_stokes_mini_assemble_rhs_aos(nelements, nnodes, elems, points, mu, rho, forcing, rhs);
            break;
        }
        case TET4: {
            tet4_stokes_mini_assemble_rhs_aos(nelements, nnodes, elems, points, mu, rho, forcing, rhs);
            break;
        }
        default: {
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}
