#include "navier_stokes.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"


void navier_stokes_assemble_value_aos(const enum ElemType element_type,
                                          const ptrdiff_t nelements,
                                          const ptrdiff_t nnodes,
                                          idx_t **const SFEM_RESTRICT elems,
                                          geom_t **const SFEM_RESTRICT xyz,
                                          const real_t nu,
                                          const real_t rho,
                                          const real_t *const SFEM_RESTRICT u,
                                          real_t *const SFEM_RESTRICT value)
{
	// TODO
}

void navier_stokes_assemble_gradient_aos(const enum ElemType element_type,
                                             const ptrdiff_t nelements,
                                             const ptrdiff_t nnodes,
                                             idx_t **const SFEM_RESTRICT elems,
                                             geom_t **const SFEM_RESTRICT xyz,
                                             const real_t nu,
                                             const real_t rho,
                                             const real_t *const SFEM_RESTRICT u,
                                             real_t *const SFEM_RESTRICT values)
{
	// TODO
}

void navier_stokes_assemble_hessian_aos(const enum ElemType element_type,
                                            const ptrdiff_t nelements,
                                            const ptrdiff_t nnodes,
                                            idx_t **const SFEM_RESTRICT elems,
                                            geom_t **const SFEM_RESTRICT xyz,
                                            const real_t nu,
                                            const real_t rho,
                                            const count_t *const SFEM_RESTRICT rowptr,
                                            const idx_t *const SFEM_RESTRICT colidx,
                                            real_t *const SFEM_RESTRICT values)
{
	// TODO
}

void navier_stokes_apply_aos(const enum ElemType element_type,
                                 const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 const real_t nu,
                                 const real_t rho,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT values)
{
	// TODO
}
