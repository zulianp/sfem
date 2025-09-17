#ifndef SSHEX8_NEOHOOKEAN_OGDEN_H
#define SSHEX8_NEOHOOKEAN_OGDEN_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int sshex8_neohookean_ogden_objective(int                               level,
                                      const ptrdiff_t                   nelements,
                                      const ptrdiff_t                   stride,
                                      const ptrdiff_t                   nnodes,
                                      idx_t **const SFEM_RESTRICT       elements,
                                      geom_t **const SFEM_RESTRICT      points,
                                      const real_t                      mu,
                                      const real_t                      lambda,
                                      const ptrdiff_t                   u_stride,
                                      const real_t *const SFEM_RESTRICT ux,
                                      const real_t *const SFEM_RESTRICT uy,
                                      const real_t *const SFEM_RESTRICT uz,
                                      const int                         is_element_wise,
                                      real_t *const SFEM_RESTRICT       out);

int sshex8_neohookean_ogden_objective_steps(int                               level,
                                            const ptrdiff_t                   nelements,
                                            const ptrdiff_t                   stride,
                                            const ptrdiff_t                   nnodes,
                                            idx_t **const SFEM_RESTRICT       elements,
                                            geom_t **const SFEM_RESTRICT      points,
                                            const real_t                      mu,
                                            const real_t                      lambda,
                                            const ptrdiff_t                   u_stride,
                                            const real_t *const SFEM_RESTRICT ux,
                                            const real_t *const SFEM_RESTRICT uy,
                                            const real_t *const SFEM_RESTRICT uz,
                                            const ptrdiff_t                   inc_stride,
                                            const real_t *const SFEM_RESTRICT incx,
                                            const real_t *const SFEM_RESTRICT incy,
                                            const real_t *const SFEM_RESTRICT incz,
                                            const int                         nsteps,
                                            const real_t *const SFEM_RESTRICT steps,
                                            real_t *const SFEM_RESTRICT       out);

int sshex8_neohookean_ogden_gradient(int                               level,
                                     const ptrdiff_t                   nelements,
                                     const ptrdiff_t                   stride,
                                     const ptrdiff_t                   nnodes,
                                     idx_t **const SFEM_RESTRICT       elements,
                                     geom_t **const SFEM_RESTRICT      points,
                                     const real_t                      mu,
                                     const real_t                      lambda,
                                     const ptrdiff_t                   u_stride,
                                     const real_t *const SFEM_RESTRICT ux,
                                     const real_t *const SFEM_RESTRICT uy,
                                     const real_t *const SFEM_RESTRICT uz,
                                     const ptrdiff_t                   out_stride,
                                     real_t *const SFEM_RESTRICT       outx,
                                     real_t *const SFEM_RESTRICT       outy,
                                     real_t *const SFEM_RESTRICT       outz);

int SShex8_neohookean_ogden_hessian_partial_assembly(int                                  level,
                                                     const ptrdiff_t                      nelements,
                                                     const ptrdiff_t                      stride,
                                                     idx_t **const SFEM_RESTRICT          elements,
                                                     geom_t **const SFEM_RESTRICT         points,
                                                     const real_t                         mu,
                                                     const real_t                         lambda,
                                                     const ptrdiff_t                      u_stride,
                                                     const real_t *const SFEM_RESTRICT    ux,
                                                     const real_t *const SFEM_RESTRICT    uy,
                                                     const real_t *const SFEM_RESTRICT    uz,
                                                     metric_tensor_t *const SFEM_RESTRICT partial_assembly);

int sshex8_neohookean_ogden_partial_assembly_apply(int                                        level,
                                                   const ptrdiff_t                            nelements,
                                                   const ptrdiff_t                            stride,
                                                   idx_t **const SFEM_RESTRICT                elements,
                                                   const metric_tensor_t *const SFEM_RESTRICT partial_assembly,
                                                   const ptrdiff_t                            h_stride,
                                                   const real_t *const SFEM_RESTRICT          hx,
                                                   const real_t *const SFEM_RESTRICT          hy,
                                                   const real_t *const SFEM_RESTRICT          hz,
                                                   const ptrdiff_t                            out_stride,
                                                   real_t *const SFEM_RESTRICT                outx,
                                                   real_t *const SFEM_RESTRICT                outy,
                                                   real_t *const SFEM_RESTRICT                outz);

int sshex8_neohookean_ogden_compressed_partial_assembly_apply(int                                     level,
                                                              const ptrdiff_t                         nelements,
                                                              const ptrdiff_t                         stride,
                                                              idx_t **const SFEM_RESTRICT             elements,
                                                              const compressed_t *const SFEM_RESTRICT partial_assembly,
                                                              const scaling_t *const SFEM_RESTRICT    scaling,
                                                              const ptrdiff_t                         h_stride,
                                                              const real_t *const SFEM_RESTRICT       hx,
                                                              const real_t *const SFEM_RESTRICT       hy,
                                                              const real_t *const SFEM_RESTRICT       hz,
                                                              const ptrdiff_t                         out_stride,
                                                              real_t *const SFEM_RESTRICT             outx,
                                                              real_t *const SFEM_RESTRICT             outy,
                                                              real_t *const SFEM_RESTRICT             outz);

#ifdef __cplusplus
}
#endif

#endif  // SSHEX8_NEOHOOKEAN_OGDEN_H