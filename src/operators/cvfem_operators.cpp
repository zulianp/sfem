#include "cvfem_operators.hpp"
#include "mpi.h"
#include "sfem_defs.hpp"

#include "tri3_laplacian.hpp"

#include "cvfem_quad4_laplacian.hpp"

#include "tet4_laplacian.hpp"

#include <stdio.h>

void cvfem_laplacian_crs(const smesh::ElemType element_type,
                         const ptrdiff_t nelements,
                         const ptrdiff_t nnodes,
                         idx_t **const SFEM_RESTRICT elems,
                         geom_t **const SFEM_RESTRICT xyz,
                         const count_t *const SFEM_RESTRICT rowptr,
                         const idx_t *const SFEM_RESTRICT colidx,
                         real_t *const SFEM_RESTRICT values)

{
    switch (element_type) {
        case smesh::TRI3: {
            tri3_laplacian_crs(nelements, nnodes, elems, xyz, rowptr, colidx, values);
            return;
        }
        case smesh::QUAD4: {
            cvfem_quad4_laplacian_crs(
                nelements, nnodes, elems, xyz, rowptr, colidx, values);
            return;
        }
        case smesh::TET4: {
            tet4_laplacian_crs(nelements, nnodes, elems, xyz, rowptr, colidx, values);
            return;
        }
        default: {
            SFEM_ERROR("IMPLEMENT ME!\n");
        }
    }
}

void cvfem_laplacian_apply(const smesh::ElemType element_type,
                           const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const SFEM_RESTRICT elems,
                           geom_t **const SFEM_RESTRICT xyz,
                           const real_t *const SFEM_RESTRICT u,
                           real_t *const SFEM_RESTRICT values)

{
    switch (element_type) {
        case smesh::TRI3: {
            tri3_laplacian_apply(nelements, nnodes, elems, xyz, u, values);
            return;
        }
        case smesh::QUAD4: {
            cvfem_quad4_laplacian_apply(nelements, nnodes, elems, xyz, u, values);
            return;
        }
        case smesh::TET4: {
            tet4_laplacian_apply(nelements, nnodes, elems, xyz, u, values);
            return;
        }
        default: {
            SFEM_ERROR("IMPLEMENT ME!\n");
        }
    }
}
