#include "cvfem_operators.h"
#include "mpi.h"
#include "sfem_defs.h"

#include "cvfem_tri3_convection.h"
#include "tri3_laplacian.h"

#include "cvfem_quad4_convection.h"
#include "cvfem_quad4_laplacian.h"

#include "cvfem_tet4_convection.h"
#include "tet4_laplacian.h"

#include <stdio.h>

void cvfem_laplacian_crs(const enum ElemType element_type,
                                      const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const SFEM_RESTRICT elems,
                                      geom_t **const SFEM_RESTRICT xyz,
                                      const count_t *const SFEM_RESTRICT rowptr,
                                      const idx_t *const SFEM_RESTRICT colidx,
                                      real_t *const SFEM_RESTRICT values)

{
    switch (element_type) {
        case TRI3: {
            tri3_laplacian_crs(nelements, nnodes, elems, xyz, rowptr, colidx, values);
            return;
        }
        case QUAD4: {
            cvfem_quad4_laplacian_crs(
                nelements, nnodes, elems, xyz, rowptr, colidx, values);
            return;
        }
        case TET4: {
            tet4_laplacian_crs(nelements, nnodes, elems, xyz, rowptr, colidx, values);
            return;
        }
        default: {
            SFEM_ERROR("IMPLEMENT ME!\n");
        }
    }
}

void cvfem_laplacian_apply(const enum ElemType element_type,
                           const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const SFEM_RESTRICT elems,
                           geom_t **const SFEM_RESTRICT xyz,
                           const real_t *const SFEM_RESTRICT u,
                           real_t *const SFEM_RESTRICT values)

{
    switch (element_type) {
        case TRI3: {
            tri3_laplacian_apply(nelements, nnodes, elems, xyz, u, values);
            return;
        }
        case QUAD4: {
            cvfem_quad4_laplacian_apply(nelements, nnodes, elems, xyz, u, values);
            return;
        }
        case TET4: {
            tet4_laplacian_apply(nelements, nnodes, elems, xyz, u, values);
            return;
        }
        default: {
            SFEM_ERROR("IMPLEMENT ME!\n");
        }
    }
}

void cvfem_convection_assemble_hessian(const enum ElemType element_type,
                                       const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const SFEM_RESTRICT elems,
                                       geom_t **const SFEM_RESTRICT xyz,
                                       real_t **const SFEM_RESTRICT velocity,
                                       const count_t *const SFEM_RESTRICT rowptr,
                                       const idx_t *const SFEM_RESTRICT colidx,
                                       real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        // case TRI3: {
        //     return;
        // }
        // case QUAD4: {
        //     return;
        // }
        // case TET4: {
        //     return;
        // }
        default: {
            SFEM_ERROR("IMPLEMENT ME!\n");
        }
    }
}

void cvfem_convection_apply(const enum ElemType element_type,
                            const ptrdiff_t nelements,
                            const ptrdiff_t nnodes,
                            idx_t **const SFEM_RESTRICT elems,
                            geom_t **const SFEM_RESTRICT xyz,
                            real_t **const SFEM_RESTRICT velocity,
                            const real_t *const SFEM_RESTRICT u,
                            real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case TRI3: {
            cvfem_tri3_convection_apply(nelements, nnodes, elems, xyz, velocity, u, values);
            return;
        }
        case QUAD4: {
            cvfem_quad4_convection_apply(nelements, nnodes, elems, xyz, velocity, u, values);
            return;
        }
        case TET4: {
            cvfem_tet4_convection_apply(nelements, nnodes, elems, xyz, velocity, u, values);
            return;
        }
        default: {
            SFEM_ERROR("IMPLEMENT ME!\n");
        }
    }
}

void cvfem_cv_volumes(const enum ElemType element_type,
                      const ptrdiff_t nelements,
                      const ptrdiff_t nnodes,
                      idx_t **const SFEM_RESTRICT elems,
                      geom_t **const SFEM_RESTRICT xyz,
                      real_t *const SFEM_RESTRICT values)

{
    switch (element_type) {
        case TRI3: {
            cvfem_tri3_cv_volumes(nelements, nnodes, elems, xyz, values);
            return;
        }
        case QUAD4: {
            cvfem_quad4_cv_volumes(nelements, nnodes, elems, xyz, values);
            return;
        }
        case TET4: {
            cvfem_tet4_cv_volumes(nelements, nnodes, elems, xyz, values);
            return;
        }
        default: {
            SFEM_ERROR("IMPLEMENT ME!\n");
        }
    }
}
