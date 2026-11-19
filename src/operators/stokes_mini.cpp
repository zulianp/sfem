#include "stokes_mini.hpp"

#include "tet4_stokes_mini.hpp"
#include "tri3_stokes_mini.hpp"

#include <mpi.h>
#include <stdio.h>

// void stokes_mini_assemble_value_soa(const smesh::ElemType element_type,
//                                           const ptrdiff_t nelements,
//                                           const ptrdiff_t nnodes,
//                                           idx_t **const SFEM_RESTRICT elems,
//                                           geom_t **const SFEM_RESTRICT xyz,
//                                           const real_t mu,
//                                           const real_t lambda,
//                                           const real_t **const SFEM_RESTRICT u,
//                                           real_t *const SFEM_RESTRICT value) {
//     switch (element_type) {
//         case smesh::TRI3: {
//             tri3_stokes_mini_assemble_value_soa(
//                 nelements, nnodes, elems, xyz, mu, u, value);
//             break;
//         }
//         // case smesh::TET4: {
//         //     tet4_stokes_mini_assemble_value_soa(
//         //         nelements, nnodes, elems, xyz, mu, u, value);
//         //     break;
//         // }
//         default: {
//             SFEM_ERROR("IMPLEMENT ME!\n");
//         }
//     }
// }

// void stokes_mini_assemble_gradient_soa(const smesh::ElemType element_type,
//                                              const ptrdiff_t nelements,
//                                              const ptrdiff_t nnodes,
//                                              idx_t **const SFEM_RESTRICT elems,
//                                              geom_t **const SFEM_RESTRICT xyz,
//                                              const real_t mu,
//                                              const real_t lambda,
//                                              const real_t **const SFEM_RESTRICT u,
//                                              real_t **const SFEM_RESTRICT values) {
//     switch (element_type) {
//         case smesh::TRI3: {
//             tri3_stokes_mini_assemble_gradient_soa(
//                 nelements, nnodes, elems, xyz, mu, u, values);
//             break;
//         }
//         // case smesh::TET4: {
//         //     tri4_stokes_mini_assemble_gradient_soa(
//         //         nelements, nnodes, elems, xyz, mu, u, values);
//         //     break;
//         // }
//         default: {
//             SFEM_ERROR("IMPLEMENT ME!\n");
//         }
//     }
// }

void stokes_mini_assemble_hessian_soa(const smesh::ElemType element_type,
                                      const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const SFEM_RESTRICT elems,
                                      geom_t **const SFEM_RESTRICT xyz,
                                      const real_t mu,
                                      const count_t *const SFEM_RESTRICT rowptr,
                                      const idx_t *const SFEM_RESTRICT colidx,
                                      real_t **const SFEM_RESTRICT values) {
    switch (element_type) {
        case smesh::TRI3: {
            tri3_stokes_mini_assemble_hessian_soa(
                nelements, nnodes, elems, xyz, mu, rowptr, colidx, values);
            break;
        }
        case smesh::TET4: {
            tet4_stokes_mini_assemble_hessian_soa(
                nelements, nnodes, elems, xyz, mu, rowptr, colidx, values);
            break;
        }
        default: {
            SFEM_ERROR("IMPLEMENT ME!\n");
        }
    }
}

// void stokes_mini_apply_soa(const smesh::ElemType element_type,
//                                  const ptrdiff_t nelements,
//                                  const ptrdiff_t nnodes,
//                                  idx_t **const SFEM_RESTRICT elems,
//                                  geom_t **const SFEM_RESTRICT xyz,
//                                  const real_t mu,
//                                  const real_t lambda,
//                                  const real_t **const SFEM_RESTRICT u,
//                                  real_t **const SFEM_RESTRICT values) {
//     switch (element_type) {
//         case smesh::TRI3: {
//             tri3_stokes_mini_apply_soa(nelements, nnodes, elems, xyz, mu, u, values);
//             break;
//         }
//         // case smesh::TET4: {
//         //     tet4_stokes_mini_apply_soa(nelements, nnodes, elems, xyz, mu, u,
//         //     values); break;
//         // }
//         default: {
//             SFEM_ERROR("IMPLEMENT ME!\n");
//         }
//     }
// }

// void stokes_mini_assemble_value_aos(const smesh::ElemType element_type,
//                                           const ptrdiff_t nelements,
//                                           const ptrdiff_t nnodes,
//                                           idx_t **const SFEM_RESTRICT elems,
//                                           geom_t **const SFEM_RESTRICT xyz,
//                                           const real_t mu,
//                                           const real_t lambda,
//                                           const real_t *const SFEM_RESTRICT u,
//                                           real_t *const SFEM_RESTRICT value) {
//     switch (element_type) {
//         case smesh::TRI3: {
//             tri3_stokes_mini_assemble_value_aos(
//                 nelements, nnodes, elems, xyz, mu, u, value);
//             break;
//         }
//         case smesh::TET4: {
//             tet4_stokes_mini_assemble_value_aos(
//                 nelements, nnodes, elems, xyz, mu, u, value);
//             break;
//         }
//         default: {
//             SFEM_ERROR("IMPLEMENT ME!\n");
//         }
//     }
// }

void stokes_mini_assemble_gradient_aos(const smesh::ElemType element_type,
                                             const ptrdiff_t nelements,
                                             const ptrdiff_t nnodes,
                                             idx_t **const SFEM_RESTRICT elems,
                                             geom_t **const SFEM_RESTRICT xyz,
                                             const real_t mu,
                                             const real_t *const SFEM_RESTRICT u,
                                             real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case smesh::TRI3: {
            tri3_stokes_mini_assemble_gradient_aos(
                nelements, nnodes, elems, xyz, mu, u, values);
            break;
        }
        case smesh::TET4: {
            tet4_stokes_mini_assemble_gradient_aos(
                nelements, nnodes, elems, xyz, mu, u, values);
            break;
        }
        default: {
            SFEM_ERROR("IMPLEMENT ME!\n");
        }
    }
}

void stokes_mini_assemble_hessian_aos(const smesh::ElemType element_type,
                                      const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const SFEM_RESTRICT elems,
                                      geom_t **const SFEM_RESTRICT xyz,
                                      const real_t mu,
                                      const count_t *const SFEM_RESTRICT rowptr,
                                      const idx_t *const SFEM_RESTRICT colidx,
                                      real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case smesh::TRI3: {
            tri3_stokes_mini_assemble_hessian_aos(
                nelements, nnodes, elems, xyz, mu, rowptr, colidx, values);
            break;
        }
        case smesh::TET4: {
            tet4_stokes_mini_assemble_hessian_aos(
                nelements, nnodes, elems, xyz, mu, rowptr, colidx, values);
            break;
        }
        default: {
            SFEM_ERROR("IMPLEMENT ME!\n");
        }
    }
}

void stokes_mini_apply_aos(const smesh::ElemType element_type,
                                 const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 const real_t mu,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case smesh::TRI3: {
            tri3_stokes_mini_apply_aos(nelements, nnodes, elems, xyz, mu, u, values);
            break;
        }
        case smesh::TET4: {
            tet4_stokes_mini_apply_aos(nelements, nnodes, elems, xyz, mu, u, values);
            break;
        }
        default: {
            SFEM_ERROR("IMPLEMENT ME!\n");
        }
    }
}

void stokes_mini_assemble_rhs_soa(smesh::ElemType element_type,
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
            case smesh::TRI3: {
                tri3_stokes_mini_assemble_rhs_soa(nelements, nnodes, elems, points, mu, rho, forcing, rhs);
                break;
            }
            case smesh::TET4: {
                tet4_stokes_mini_assemble_rhs_soa(nelements, nnodes, elems, points, mu, rho, forcing, rhs);
                break;
            }
            default: {
                SFEM_ERROR("IMPLEMENT ME!\n");
            }
        }
}



void stokes_mini_assemble_rhs_aos(smesh::ElemType element_type,
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
        case smesh::TRI3: {
            tri3_stokes_mini_assemble_rhs_aos(nelements, nnodes, elems, points, mu, rho, forcing, rhs);
            break;
        }
        case smesh::TET4: {
            tet4_stokes_mini_assemble_rhs_aos(nelements, nnodes, elems, points, mu, rho, forcing, rhs);
            break;
        }
        default: {
            SFEM_ERROR("IMPLEMENT ME!\n");
        }
    }
}
