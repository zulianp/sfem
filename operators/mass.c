#include "mass.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"

#include "sfem_defs.h"

#include "beam2_mass.h"
#include "quadshell4_mass.h"
#include "tet10_mass.h"
#include "tet4_mass.h"
#include "tri3_mass.h"
#include "tri6_mass.h"
#include "trishell3_mass.h"

void assemble_mass(const int element_type,
                   const ptrdiff_t nelements,
                   const ptrdiff_t nnodes,
                   idx_t **const SFEM_RESTRICT elems,
                   geom_t **const SFEM_RESTRICT xyz,
                   const count_t *const SFEM_RESTRICT rowptr,
                   const idx_t *const SFEM_RESTRICT colidx,
                   real_t *const SFEM_RESTRICT values) {
    if (!nelements) return;

    switch (element_type) {
        case TET4: {
            tet4_assemble_mass(nelements, nnodes, elems, xyz, rowptr, colidx, values);
            break;
        }
        case TET10: {
            tet10_assemble_mass(nelements, nnodes, elems, xyz, rowptr, colidx, values);
            break;
        }
        case TRI3: {
            tri3_assemble_mass(nelements, nnodes, elems, xyz, rowptr, colidx, values);
            break;
        }
        case TRISHELL3: {
            trishell3_assemble_mass(nelements, nnodes, elems, xyz, rowptr, colidx, values);
            break;
        }
        case TRI6: {
            tri6_assemble_mass(nelements, nnodes, elems, xyz, rowptr, colidx, values);
            break;
        }

        default: {
            fprintf(stderr,
                    "assemble_mass not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void assemble_lumped_mass(const int element_type,
                          const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const SFEM_RESTRICT elems,
                          geom_t **const SFEM_RESTRICT xyz,
                          real_t *const SFEM_RESTRICT values) {
    if (!nelements) return;

    switch (element_type) {
        case TRI3: {
            tri3_assemble_lumped_mass(nelements, nnodes, elems, xyz, values);
            break;
        }
        case TRISHELL3: {
            trishell3_assemble_lumped_mass(nelements, nnodes, elems, xyz, values);
            break;
        }
        case TRI6: {
            tri6_assemble_lumped_mass(nelements, nnodes, elems, xyz, values);
            break;
        }
        case TET4: {
            tet4_assemble_lumped_mass(nelements, nnodes, elems, xyz, values);
            break;
        }
        case TET10: {
            tet10_assemble_lumped_mass(nelements, nnodes, elems, xyz, values);
            break;
        }
        case BEAM2: {
            beam2_assemble_lumped_mass(nelements, nnodes, elems, xyz, values);
            break;
        }

        default: {
            fprintf(stderr,
                    "assemble_lumped_mass not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void apply_inv_lumped_mass(const int element_type,
                           const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const SFEM_RESTRICT elems,
                           geom_t **const SFEM_RESTRICT xyz,
                           const real_t *const x,
                           real_t *const values) {
    if (!nelements) return;

    switch (element_type) {
        case TRI3: {
            tri3_apply_inv_lumped_mass(nelements, nnodes, elems, xyz, x, values);
            break;
        }
        case TRISHELL3: {
            trishell3_apply_inv_lumped_mass(nelements, nnodes, elems, xyz, x, values);
            break;
        }
        case TRI6: {
            tri6_apply_inv_lumped_mass(nelements, nnodes, elems, xyz, x, values);
            break;
        }
        case TET4: {
            tet4_apply_inv_lumped_mass(nelements, nnodes, elems, xyz, x, values);
            break;
        }
        case TET10: {
            tet10_apply_inv_lumped_mass(nelements, nnodes, elems, xyz, x, values);
            break;
        }
        case BEAM2: {
            beam2_apply_inv_lumped_mass(nelements, nnodes, elems, xyz, x, values);
            break;
        }

        default: {
            fprintf(stderr,
                    "apply_inv_lumped_mass not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}

void apply_mass(const int element_type,
                const ptrdiff_t nelements,
                const ptrdiff_t nnodes,
                idx_t **const SFEM_RESTRICT elems,
                geom_t **const SFEM_RESTRICT xyz,
                const ptrdiff_t stride_x,
                const real_t *const x,
                const ptrdiff_t stride_values,
                real_t *const values) {
    if (!nelements) return;

    switch (element_type) {
        case TRI3: {
            tri3_apply_mass(nelements, nnodes, elems, xyz, stride_x, x, stride_values, values);
            break;
        }
        case TRISHELL3: {
            trishell3_apply_mass(nelements, nnodes, elems, xyz, stride_x, x, stride_values, values);
            break;
        }
        case QUADSHELL4: {
            quadshell4_apply_mass(
                    nelements, nnodes, elems, xyz, stride_x, x, stride_values, values);
            break;
        }
            // case TRI6: {
            //         tri6_apply_mass(nelements, nnodes, elems, xyz, x, values);
            //         break;
            //     }

            // case TET4: {
            //     tet4_apply_mass(nelements, nnodes, elems, xyz, x, values);
            //     break;
            // }

            // case TET10: {
            //     tet10_apply_mass(nelements, nnodes, elems, xyz, x, values);
            //     break;
            // }

        default: {
            fprintf(stderr,
                    "apply_mass not implemented for type %s\n",
                    type_to_string(element_type));
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}
