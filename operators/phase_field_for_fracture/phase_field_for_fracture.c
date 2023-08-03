#include "phase_field_for_fracture.h"

#include "sfem_defs.h"
#include "tet4_phase_field_for_fracture.h"

#include <assert.h>
#include <mpi.h>

void phase_field_for_fracture_assemble_hessian_aos(const enum ElemType element_type,
                                                   const ptrdiff_t nelements,
                                                   const ptrdiff_t nnodes,
                                                   idx_t **const SFEM_RESTRICT elems,
                                                   geom_t **const SFEM_RESTRICT xyz,
                                                   const real_t mu,
                                                   const real_t lambda,
                                                   const real_t Gc,
                                                   const real_t ls,
                                                   const real_t *const SFEM_RESTRICT solution,
                                                   const count_t *const SFEM_RESTRICT rowptr,
                                                   const idx_t *const SFEM_RESTRICT colidx,
                                                   real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case TET4: {
            tet4_phase_field_for_fracture_assemble_hessian_aos(nelements,
                                                               nnodes,
                                                               elems,
                                                               xyz,
                                                               mu,
                                                               lambda,
                                                               Gc,
                                                               ls,
                                                               solution,
                                                               rowptr,
                                                               colidx,
                                                               values);
            break;
        }
        default: {
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
            break;
        }
    }
}

void phase_field_for_fracture_assemble_gradient_aos(const enum ElemType element_type,
                                                    const ptrdiff_t nelements,
                                                    const ptrdiff_t nnodes,
                                                    idx_t **const SFEM_RESTRICT elems,
                                                    geom_t **const SFEM_RESTRICT xyz,
                                                    const real_t mu,
                                                    const real_t lambda,
                                                    const real_t Gc,
                                                    const real_t ls,
                                                    const real_t *const SFEM_RESTRICT solution,
                                                    real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case TET4: {
            tet4_phase_field_for_fracture_assemble_gradient_aos(
                nelements, nnodes, elems, xyz, mu, lambda, Gc, ls, solution, values);
            break;
        }
        default: {
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
            break;
        }
    }
}

void phase_field_for_fracture_assemble_value_aos(const enum ElemType element_type,
                                                 const ptrdiff_t nelements,
                                                 const ptrdiff_t nnodes,
                                                 idx_t **const SFEM_RESTRICT elems,
                                                 geom_t **const SFEM_RESTRICT xyz,
                                                 const real_t mu,
                                                 const real_t lambda,
                                                 const real_t Gc,
                                                 const real_t ls,
                                                 const real_t *const SFEM_RESTRICT solution,
                                                 real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case TET4: {
            tet4_phase_field_for_fracture_assemble_value_aos(
                nelements, nnodes, elems, xyz, mu, lambda, Gc, ls, solution, values);
            break;
        }
        default: {
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, -1);
            break;
        }
    }
}
