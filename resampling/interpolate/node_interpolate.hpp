#ifndef NODE_INTERPOLATE_H
#define NODE_INTERPOLATE_H

#include <stddef.h>

#include "sfem_base.hpp"
#include "sfem_defs.hpp"

#ifdef __cplusplus
extern "C" {
#endif

int interpolate_gap(const ptrdiff_t              nnodes,
                    geom_t** const SFEM_RESTRICT xyz,
                    // SDF
                    const ptrdiff_t* const SFEM_RESTRICT n,
                    const ptrdiff_t* const SFEM_RESTRICT stride,
                    const geom_t* const SFEM_RESTRICT    origin,
                    const geom_t* const SFEM_RESTRICT    delta,
                    const geom_t* const SFEM_RESTRICT    data,
                    // Output
                    real_t* const SFEM_RESTRICT g,
                    real_t* const SFEM_RESTRICT xnormal,
                    real_t* const SFEM_RESTRICT ynormal,
                    real_t* const SFEM_RESTRICT znormal);

int interpolate_gap_value(const ptrdiff_t              nnodes,
                          geom_t** const SFEM_RESTRICT xyz,
                          // SDF
                          const ptrdiff_t* const SFEM_RESTRICT n,
                          const ptrdiff_t* const SFEM_RESTRICT stride,
                          const geom_t* const SFEM_RESTRICT    origin,
                          const geom_t* const SFEM_RESTRICT    delta,
                          const geom_t* const SFEM_RESTRICT    data,
                          // Output
                          real_t* const SFEM_RESTRICT g);

int interpolate_gap_normals(const ptrdiff_t              nnodes,
                            geom_t** const SFEM_RESTRICT xyz,
                            // SDF
                            const ptrdiff_t* const SFEM_RESTRICT n,
                            const ptrdiff_t* const SFEM_RESTRICT stride,
                            const geom_t* const SFEM_RESTRICT    origin,
                            const geom_t* const SFEM_RESTRICT    delta,
                            const geom_t* const SFEM_RESTRICT    data,
                            // Output
                            real_t* const SFEM_RESTRICT xnormal,
                            real_t* const SFEM_RESTRICT ynormal,
                            real_t* const SFEM_RESTRICT znormal);

// At the points X[idx[i]] + disp[3 idx[i] + d] (all nnodes nodes when idx is null; disp, AoS with
// three components, may be null), summed in real_t: the gap -sdf / |grad sdf| and the unit normal
// grad sdf / |grad sdf| (3 rows), entry i for point i. Outside the grid the gap is
// SFEM_SDF_UNREACHABLE and the normal zero.
int interpolate_displaced_gap(const ptrdiff_t                   nnodes,
                              const idx_t* const SFEM_RESTRICT  idx,
                              geom_t** const SFEM_RESTRICT      xyz,
                              const real_t* const SFEM_RESTRICT disp,
                              // SDF
                              const ptrdiff_t* const SFEM_RESTRICT n,
                              const ptrdiff_t* const SFEM_RESTRICT stride,
                              const geom_t* const SFEM_RESTRICT    origin,
                              const geom_t* const SFEM_RESTRICT    delta,
                              const geom_t* const SFEM_RESTRICT    data,
                              // Output
                              real_t* const SFEM_RESTRICT  g,
                              real_t** const SFEM_RESTRICT normals);

// The obstacle's curvature at the same points: (H - n m^T - m n^T) / |grad sdf|, H the Hessian of
// the B-spline, n its unit gradient and m = H n = grad |grad sdf|, packed xx, xy, xz, yy, yz, zz
// (6 rows). Where the point is on the surface (sdf = 0) it is minus the Hessian of the gap, so a
// reaction mu >= 0 there adds mu times it to the Newton tangent; for an exact SDF it is the SDF's
// Hessian, -(I - n n^T) / R on a sphere of radius R. Zero outside the grid.
//
// From the prefiltered coefficients, as everything else: the B-spline interpolates the samples.
// It is O(h^2) accurate, and exact for polynomials up to degree 3 away from the grid's ends; within
// about 8 cells of them the filter's end extension pulls it towards zero.
int interpolate_displaced_curvature(const ptrdiff_t                   nnodes,
                                    const idx_t* const SFEM_RESTRICT  idx,
                                    geom_t** const SFEM_RESTRICT      xyz,
                                    const real_t* const SFEM_RESTRICT disp,
                                    // SDF
                                    const ptrdiff_t* const SFEM_RESTRICT n,
                                    const ptrdiff_t* const SFEM_RESTRICT stride,
                                    const geom_t* const SFEM_RESTRICT    origin,
                                    const geom_t* const SFEM_RESTRICT    delta,
                                    const geom_t* const SFEM_RESTRICT    data,
                                    // Output
                                    real_t** const SFEM_RESTRICT curvature);

#ifdef __cplusplus
}
#endif

#endif  // NODE_INTERPOLATE_H
