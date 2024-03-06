#ifndef PHASE_FIELD_FOR_FRACTURE_H
#define PHASE_FIELD_FOR_FRACTURE_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// rowptr and colidx represent the node-to-node graph of the mesh while values are in the standard
// scalar crs format separate rowptr and colidx are to be used for solvers
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
                                                   real_t *const SFEM_RESTRICT values);

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
                                                    real_t *const SFEM_RESTRICT values);

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
                                                 real_t *const SFEM_RESTRICT values);

#ifdef __cplusplus
}
#endif
#endif  // PHASE_FIELD_FOR_FRACTURE_H
