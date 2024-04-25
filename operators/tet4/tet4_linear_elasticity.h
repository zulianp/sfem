#ifndef TET4_LINEAR_ELASTICITY_H
#define TET4_LINEAR_ELASTICITY_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    real_t mu;
    real_t lambda;
    enum ElemType element_type;
    ptrdiff_t nelements;
    void *jacobian_determinant;
    void *jacobian_adjugate;
    idx_t **elements;
} linear_elasticity_t;

void tet4_linear_elasticity_assemble_value_aos(const ptrdiff_t nelements,
                                               const ptrdiff_t nnodes,
                                               idx_t **const SFEM_RESTRICT elems,
                                               geom_t **const SFEM_RESTRICT xyz,
                                               const real_t mu,
                                               const real_t lambda,
                                               const real_t *const SFEM_RESTRICT displacement,
                                               real_t *const SFEM_RESTRICT value);

void tet4_linear_elasticity_assemble_gradient_aos(const ptrdiff_t nelements,
                                                  const ptrdiff_t nnodes,
                                                  idx_t **const SFEM_RESTRICT elems,
                                                  geom_t **const SFEM_RESTRICT xyz,
                                                  const real_t mu,
                                                  const real_t lambda,
                                                  const real_t *const SFEM_RESTRICT displacement,
                                                  real_t *const SFEM_RESTRICT values);

void tet4_linear_elasticity_assemble_hessian_aos(const ptrdiff_t nelements,
                                                 const ptrdiff_t nnodes,
                                                 idx_t **const SFEM_RESTRICT elems,
                                                 geom_t **const SFEM_RESTRICT xyz,
                                                 const real_t mu,
                                                 const real_t lambda,
                                                 const count_t *const SFEM_RESTRICT rowptr,
                                                 const idx_t *const SFEM_RESTRICT colidx,
                                                 real_t *const SFEM_RESTRICT values);

void tet4_linear_elasticity_apply_aos(const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const SFEM_RESTRICT elems,
                                      geom_t **const SFEM_RESTRICT xyz,
                                      const real_t mu,
                                      const real_t lambda,
                                      const real_t *const SFEM_RESTRICT displacement,
                                      real_t *const SFEM_RESTRICT values);

void tet4_linear_elasticity_apply_soa(const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const SFEM_RESTRICT elems,
                                      geom_t **const SFEM_RESTRICT xyz,
                                      const real_t mu,
                                      const real_t lambda,
                                      const real_t **const SFEM_RESTRICT u,
                                      real_t **const SFEM_RESTRICT values);

void tet4_linear_elasticity_assemble_diag_aos(const ptrdiff_t nelements,
                                const ptrdiff_t nnodes,
                                idx_t **const SFEM_RESTRICT elems,
                                geom_t **const SFEM_RESTRICT xyz,
                                const real_t mu,
                                const real_t lambda,
                                real_t *const SFEM_RESTRICT values);

#ifdef __cplusplus
}
#endif
#endif  // TET4_LINEAR_ELASTICITY_H
