#ifndef SFEM_TET4_DIV_H
#define SFEM_TET4_DIV_H

#include <stddef.h>
#include "sfem_base.h"

void tet4_div_apply(const ptrdiff_t nelements,
               const ptrdiff_t nnodes,
               idx_t **const elems,
               geom_t **const xyz,
               const real_t *const ux,
               const real_t *const uy,
               const real_t *const uz,
               real_t *const values);

void tet4_integrate_div(const ptrdiff_t nelements,
                   const ptrdiff_t nnodes,
                   idx_t **const elems,
                   geom_t **const xyz,
                   const real_t *const ux,
                   const real_t *const uy,
                   const real_t *const uz,
                   real_t *const value);

void tet4_cdiv(const ptrdiff_t nelements,
          const ptrdiff_t nnodes,
          idx_t **const SFEM_RESTRICT elems,
          geom_t **const SFEM_RESTRICT xyz,
          const real_t *const SFEM_RESTRICT ux,
          const real_t *const SFEM_RESTRICT uy,
          const real_t *const SFEM_RESTRICT uz,
          real_t *const SFEM_RESTRICT div);


void tet4_p0_u_dot_grad_q_apply(const ptrdiff_t nelements,
          const ptrdiff_t nnodes,
          idx_t **const SFEM_RESTRICT elems,
          geom_t **const SFEM_RESTRICT xyz,
          const real_t *const SFEM_RESTRICT ux,
          const real_t *const SFEM_RESTRICT uy,
          const real_t *const SFEM_RESTRICT uz,
          real_t *const SFEM_RESTRICT values);

void tet4_p1_u_dot_grad_q_apply(const ptrdiff_t nelements,
          const ptrdiff_t nnodes,
          idx_t **const SFEM_RESTRICT elems,
          geom_t **const SFEM_RESTRICT xyz,
          const real_t *const SFEM_RESTRICT ux,
          const real_t *const SFEM_RESTRICT uy,
          const real_t *const SFEM_RESTRICT uz,
          real_t *const SFEM_RESTRICT values);

#endif  // SFEM_TET4_DIV_H
