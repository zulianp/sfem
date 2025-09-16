#ifndef HEX8_NEOHOOKEAN_OGDEN_H
#define HEX8_NEOHOOKEAN_OGDEN_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int hex8_neohookean_ogden_gradient(const ptrdiff_t              nelements,
                                   const ptrdiff_t              stride,
                                   const ptrdiff_t              nnodes,
                                   idx_t **const SFEM_RESTRICT  elements,
                                   geom_t **const SFEM_RESTRICT points,
                                   const real_t                 mu,
                                   const real_t                 lambda,
                                   const ptrdiff_t              u_stride,
                                   const real_t *const          ux,
                                   const real_t *const          uy,
                                   const real_t *const          uz,
                                   const ptrdiff_t              out_stride,
                                   real_t *const                outx,
                                   real_t *const                outy,
                                   real_t *const                outz);

int hex8_neohookean_ogden_hessian_partial_assembly(const ptrdiff_t                      nelements,
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

int hex8_neohookean_ogden_partial_assembly_apply(const ptrdiff_t                            nelements,
                                                 const ptrdiff_t                            stride,
                                                 idx_t **const SFEM_RESTRICT                elements,
                                                 const metric_tensor_t *const SFEM_RESTRICT partial_assembly,
                                                 const ptrdiff_t                            h_stride,
                                                 const real_t *const                        hx,
                                                 const real_t *const                        hy,
                                                 const real_t *const                        hz,
                                                 const ptrdiff_t                            out_stride,
                                                 real_t *const                              outx,
                                                 real_t *const                              outy,
                                                 real_t *const                              outz);

int hex8_neohookean_ogden_compressed_partial_assembly_apply(const ptrdiff_t                         nelements,
                                                            const ptrdiff_t                         stride,
                                                            idx_t **const SFEM_RESTRICT             elements,
                                                            const compressed_t *const SFEM_RESTRICT partial_assembly,
                                                            const scaling_t *const SFEM_RESTRICT    scaling,
                                                            const ptrdiff_t                         h_stride,
                                                            const real_t *const                     hx,
                                                            const real_t *const                     hy,
                                                            const real_t *const                     hz,
                                                            const ptrdiff_t                         out_stride,
                                                            real_t *const                           outx,
                                                            real_t *const                           outy,
                                                            real_t *const                           outz);

#ifdef __cplusplus
}
#endif

#endif  // HEX8_NEOHOOKEAN_OGDEN_H