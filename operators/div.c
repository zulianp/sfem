
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>

#include <mpi.h>

#include "tet10_div.h"
#include "tet4_div.h"

#include "sfem_defs.h"

void div_apply(const int element_type,
               const ptrdiff_t nelements,
               const ptrdiff_t nnodes,
               idx_t **const elems,
               geom_t **const xyz,
               const real_t *const ux,
               const real_t *const uy,
               const real_t *const uz,
               real_t *const values) {
    switch (element_type) {
        case TET4: {
            tet4_div_apply(nelements, nnodes, elems, xyz, ux, uy, uz, values);
            break;
        }

        case TET10: {
            tet10_div_apply(nelements, nnodes, elems, xyz, ux, uy, uz, values);
            break;
        }

        default: {
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void integrate_div(const int element_type,
                   const ptrdiff_t nelements,
                   const ptrdiff_t nnodes,
                   idx_t **const elems,
                   geom_t **const xyz,
                   const real_t *const ux,
                   const real_t *const uy,
                   const real_t *const uz,
                   real_t *const value) {
    switch (element_type) {
        case TET4: {
            tet4_integrate_div(nelements, nnodes, elems, xyz, ux, uy, uz, value);
            break;
        }

        case TET10: {
            tet10_integrate_div(nelements, nnodes, elems, xyz, ux, uy, uz, value);
            break;
        }

        default: {
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void cdiv(const int element_type,
          const ptrdiff_t nelements,
          const ptrdiff_t nnodes,
          idx_t **const SFEM_RESTRICT elems,
          geom_t **const SFEM_RESTRICT xyz,
          const real_t *const SFEM_RESTRICT ux,
          const real_t *const SFEM_RESTRICT uy,
          const real_t *const SFEM_RESTRICT uz,
          real_t *const SFEM_RESTRICT div) {
    switch (element_type) {
        case TET4: {
            tet4_cdiv(nelements, nnodes, elems, xyz, ux, uy, uz, div);
            break;
        }

        case TET10: {
            tet10_cdiv(nelements, nnodes, elems, xyz, ux, uy, uz, div);
            break;
        }

        default: {
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void p1_u_dot_grad_q_apply(const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const SFEM_RESTRICT elems,
                           geom_t **const SFEM_RESTRICT xyz,
                           const real_t *const SFEM_RESTRICT ux,
                           const real_t *const SFEM_RESTRICT uy,
                           const real_t *const SFEM_RESTRICT uz,
                           real_t *const SFEM_RESTRICT values) {
    tet4_p1_u_dot_grad_q_apply(nelements, nnodes, elems, xyz, ux, uy, uz, values);
}

void p0_u_dot_grad_q_apply(const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const SFEM_RESTRICT elems,
                           geom_t **const SFEM_RESTRICT xyz,
                           const real_t *const SFEM_RESTRICT ux,
                           const real_t *const SFEM_RESTRICT uy,
                           const real_t *const SFEM_RESTRICT uz,
                           real_t *const SFEM_RESTRICT values) {
    p0_u_dot_grad_q_apply(nelements, nnodes, elems, xyz, ux, uy, uz, values);
}
