#include "surface_l2_projection.h"

#include <mpi.h>

#include "sfem_defs.h"

#include "trishell3_l2_projection_p0_p1.h"
#include "trishell6_l2_projection_p1_p2.h"


void surface_e_projection_apply(const int element_type,
                                const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                const real_t *const SFEM_RESTRICT element_wise_u,
                                real_t *const SFEM_RESTRICT u) {
    switch (element_type) {
        case TRI3: {
        	trishell3_p0_p1_l2_projection_apply(nelements, nnodes, elems, xyz, element_wise_u, u);
            break;
        }

        case TRI6: {
            trishell6_ep1_p2_l2_projection_apply(nelements, nnodes, elems, xyz, element_wise_u, u);
            break;
        }

        default: {
            assert(false);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void surface_e_projection_coeffs(const int element_type,
                                 const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 const real_t *const SFEM_RESTRICT element_wise_u,
                                 real_t *const SFEM_RESTRICT u) {
    switch (element_type) {
        case TRI3: {
            trishell3_p0_p1_projection_coeffs(nelements, nnodes, elems, xyz, element_wise_u, u);
            break;
        }

        case TRI6: {
            trishell6_ep1_p2_projection_coeffs(nelements, nnodes, elems, xyz, element_wise_u, u);
            break;
        }

        default: {
            assert(false);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}
