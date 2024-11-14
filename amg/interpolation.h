#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include "sparse.h"

typedef struct {
  ptrdiff_t fine_dim;
  ptrdiff_t coarse_dim;
  // Length of `fine_dim` and values indicate coarse grid indices
  idx_t *partition;
  // Length of `fine_dim` and values weight the PWC gridfunction
  real_t *weights;
} PiecewiseConstantTransfer;

// Internally allocates a workspace with same memory requirement as `a`
// (this could be passed in as arg...)
void coarsen(const SymmCOOMatrix *a, const PiecewiseConstantTransfer *p,
             SymmCOOMatrix *a_coarse);

void pwc_interpolate(const PiecewiseConstantTransfer *p, const real_t *v_coarse,
                     real_t *v);
void pwc_restrict(const PiecewiseConstantTransfer *p, const real_t *v,
                  real_t *v_coarse);

#endif // INTERPOLATION_H
