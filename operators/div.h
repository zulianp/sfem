#ifndef SFEM_DIV_H
#define SFEM_DIV_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void div_apply(const int element_type,
               const ptrdiff_t nelements,
               const ptrdiff_t nnodes,
               idx_t **const elems,
               geom_t **const xyz,
               const real_t *const ux,
               const real_t *const uy,
               const real_t *const uz,
               real_t *const values);

void integrate_div(const int element_type,
                   const ptrdiff_t nelements,
                   const ptrdiff_t nnodes,
                   idx_t **const elems,
                   geom_t **const xyz,
                   const real_t *const ux,
                   const real_t *const uy,
                   const real_t *const uz,
                   real_t *const value);

void cdiv(const int element_type,
          const ptrdiff_t nelements,
          const ptrdiff_t nnodes,
          idx_t **const SFEM_RESTRICT elems,
          geom_t **const SFEM_RESTRICT xyz,
          const real_t *const SFEM_RESTRICT ux,
          const real_t *const SFEM_RESTRICT uy,
          const real_t *const SFEM_RESTRICT uz,
          real_t *const SFEM_RESTRICT div);

void p0_u_dot_grad_q_apply(const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const SFEM_RESTRICT elems,
                           geom_t **const SFEM_RESTRICT xyz,
                           const real_t *const SFEM_RESTRICT ux,
                           const real_t *const SFEM_RESTRICT uy,
                           const real_t *const SFEM_RESTRICT uz,
                           real_t *const SFEM_RESTRICT values);

void p1_u_dot_grad_q_apply(const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const SFEM_RESTRICT elems,
                           geom_t **const SFEM_RESTRICT xyz,
                           const real_t *const SFEM_RESTRICT ux,
                           const real_t *const SFEM_RESTRICT uy,
                           const real_t *const SFEM_RESTRICT uz,
                           real_t *const SFEM_RESTRICT values);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_DIV_H
