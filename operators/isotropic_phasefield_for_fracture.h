#ifndef ISOTROPIC_PHASEFIELD_FOR_FRACTURE_H
#define ISOTROPIC_PHASEFIELD_FOR_FRACTURE_H

#include <stddef.h>

#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void isotropic_phasefield_for_fracture_assemble_hessian(const ptrdiff_t nelements,
                                                         const ptrdiff_t nnodes,
                                                         idx_t *const elems[4],
                                                         geom_t *const xyz[3],
                                                         const real_t mu,
                                                         const real_t lambda,
                                                         const real_t Gc,
                                                         const real_t ls,
                                                         const real_t *const u,
                                                         count_t *const rowptr,
                                                         idx_t *const colidx,
                                                         real_t *const values);

void isotropic_phasefield_for_fracture_assemble_gradient(const ptrdiff_t nelements,
                                                          const ptrdiff_t nnodes,
                                                          idx_t *const elems[4],
                                                          geom_t *const xyz[3],
                                                          const real_t mu,
                                                          const real_t lambda,
                                                          const real_t Gc,
                                                          const real_t ls,
                                                          const real_t *const u,
                                                          real_t *const values);

void isotropic_phasefield_for_fracture_assemble_value(const ptrdiff_t nelements,
                                                       const ptrdiff_t nnodes,
                                                       idx_t *const elems[4],
                                                       geom_t *const xyz[3],
                                                       const real_t mu,
                                                       const real_t lambda,
                                                       const real_t Gc,
                                                       const real_t ls,
                                                       const real_t *const u,
                                                       real_t *const value);

#ifdef __cplusplus
}
#endif

#endif  // ISOTROPIC_PHASEFIELD_FOR_FRACTURE_H
