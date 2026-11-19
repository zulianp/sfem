#ifndef MASS_H
#define MASS_H

#include <stddef.h>
#include "sfem_base.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void assemble_mass(const int element_type,
                   const ptrdiff_t nelements,
                   const ptrdiff_t nnodes,
                   idx_t **const SFEM_RESTRICT elems,
                   geom_t **const SFEM_RESTRICT xyz,
                   const count_t *const SFEM_RESTRICT rowptr,
                   const idx_t *const SFEM_RESTRICT colidx,
                   real_t *const SFEM_RESTRICT values);

void assemble_lumped_mass(const int element_type,
                          const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const SFEM_RESTRICT elems,
                          geom_t **const SFEM_RESTRICT xyz,
                          real_t *const SFEM_RESTRICT values);

void apply_inv_lumped_mass(const int element_type,
                           const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const SFEM_RESTRICT elems,
                           geom_t **const SFEM_RESTRICT xyz,
                           const real_t *const x,
                           real_t *const values);

/// 1 if apply_mass has a kernel for this element type.
int apply_mass_supported(const int element_type);

int apply_mass(const int element_type,
               const ptrdiff_t nelements,
               const ptrdiff_t nnodes,
               idx_t **const SFEM_RESTRICT elems,
               geom_t **const SFEM_RESTRICT xyz,
               const ptrdiff_t stride_x,
               const real_t *const x,
               const ptrdiff_t stride_values,
               real_t *const values);

/// apply_mass by the threads of the caller's region (orphaned `omp for`, trailing barrier).
int apply_mass_team(const int element_type,
                    const ptrdiff_t nelements,
                    const ptrdiff_t nnodes,
                    idx_t **const SFEM_RESTRICT elems,
                    geom_t **const SFEM_RESTRICT xyz,
                    const ptrdiff_t stride_x,
                    const real_t *const x,
                    const ptrdiff_t stride_values,
                    real_t *const values);

#ifdef __cplusplus
}
#endif

#endif  // MASS_H
