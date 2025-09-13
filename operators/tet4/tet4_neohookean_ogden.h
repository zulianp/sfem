#ifndef TET4_NEOHOOKEAN_OGDEN_H
#define TET4_NEOHOOKEAN_OGDEN_H

#include <stddef.h>
#include "sfem_base.h"

#ifdef __cplusplus
extern "C" {
#endif

int tet4_neohookean_ogden_value(const ptrdiff_t              nelements,
                                const ptrdiff_t              nnodes,
                                idx_t **const SFEM_RESTRICT  elements,
                                geom_t **const SFEM_RESTRICT points,
                                const real_t                 mu,
                                const real_t                 lambda,
                                const ptrdiff_t              u_stride,
                                const real_t *const          ux,
                                const real_t *const          uy,
                                const real_t *const          uz,
                                real_t *const SFEM_RESTRICT  value);

int tet4_neohookean_ogden_gradient(const ptrdiff_t              nelements,
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

int tet4_neohookean_ogden_apply(const ptrdiff_t              nelements,
                                const ptrdiff_t              nnodes,
                                idx_t **const SFEM_RESTRICT  elements,
                                geom_t **const SFEM_RESTRICT points,
                                const real_t                 mu,
                                const real_t                 lambda,
                                const ptrdiff_t              u_stride,
                                const real_t *const          ux,
                                const real_t *const          uy,
                                const real_t *const          uz,
                                const ptrdiff_t              h_stride,
                                const real_t *const          hx,
                                const real_t *const          hy,
                                const real_t *const          hz,
                                const ptrdiff_t              out_stride,
                                real_t *const                outx,
                                real_t *const                outy,
                                real_t *const                outz);

int tet4_neohookean_ogden_diag(const ptrdiff_t              nelements,
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

int tet4_neohookean_ogden_hessian(const ptrdiff_t                   nelements,
                                  const ptrdiff_t                   nnodes,
                                  idx_t **const SFEM_RESTRICT       elements,
                                  geom_t **const SFEM_RESTRICT      points,
                                  const real_t                      mu,
                                  const real_t                      lambda,
                                  const ptrdiff_t                   u_stride,
                                  const real_t *const SFEM_RESTRICT ux,
                                  const real_t *const SFEM_RESTRICT uy,
                                  const real_t *const SFEM_RESTRICT uz,
                                  count_t *const SFEM_RESTRICT      rowptr,
                                  idx_t *const SFEM_RESTRICT        colidx,
                                  real_t *const SFEM_RESTRICT       values);

int tet4_neohookean_ogden_hessian_partial_assembly(const ptrdiff_t                       nelements,
                                                   idx_t **const SFEM_RESTRICT           elements,
                                                   geom_t **const SFEM_RESTRICT          points,
                                                   const real_t                          mu,
                                                   const real_t                          lambda,
                                                   const ptrdiff_t                       u_stride,
                                                   const real_t *const SFEM_RESTRICT     ux,
                                                   const real_t *const SFEM_RESTRICT     uy,
                                                   const real_t *const SFEM_RESTRICT     uz,
                                                   const ptrdiff_t                       S_ikmn_stride,
                                                   metric_tensor_t **const SFEM_RESTRICT partial_assembly);

int tet4_neohookean_ogden_partial_assembly_apply(const ptrdiff_t                       nelements,
                                                 idx_t **const SFEM_RESTRICT           elements,
                                                 const ptrdiff_t                       S_ikmn_stride,
                                                 metric_tensor_t **const SFEM_RESTRICT partial_assembly,
                                                 const ptrdiff_t                       h_stride,
                                                 const real_t *const                   hx,
                                                 const real_t *const                   hy,
                                                 const real_t *const                   hz,
                                                 const ptrdiff_t                       out_stride,
                                                 real_t *const                         outx,
                                                 real_t *const                         outy,
                                                 real_t *const                         outz);

int tet4_neohookean_ogden_compressed_partial_assembly_apply(const ptrdiff_t                      nelements,
                                                            idx_t **const SFEM_RESTRICT          elements,
                                                            const ptrdiff_t                      S_ikmn_stride,
                                                            compressed_t **const SFEM_RESTRICT   partial_assembly,
                                                            const scaling_t *const SFEM_RESTRICT scaling,
                                                            const ptrdiff_t                      h_stride,
                                                            const real_t *const                  hx,
                                                            const real_t *const                  hy,
                                                            const real_t *const                  hz,
                                                            const ptrdiff_t                      out_stride,
                                                            real_t *const                        outx,
                                                            real_t *const                        outy,
                                                            real_t *const                        outz);

// Opt version later

// int tet4_neohookean_ogden_gradient_opt(const ptrdiff_t nelements,
//                                  idx_t **const SFEM_RESTRICT elements,
//                                  const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
//                                  const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
//                                  const real_t mu,
//                                  const real_t lambda,
//                                  const ptrdiff_t u_stride,
//                                  const real_t *const ux,
//                                  const real_t *const uy,
//                                  const real_t *const uz,
//                                  const ptrdiff_t out_stride,
//                                  real_t *const outx,
//                                  real_t *const outy,
//                                  real_t *const outz);

// int tet4_neohookean_ogden_apply_opt(const ptrdiff_t nelements,
//                                  idx_t **const SFEM_RESTRICT elements,
//                                  const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
//                                  const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
//                                  const real_t mu,
//                                  const real_t lambda,
//                                  const ptrdiff_t u_stride,
//                                  const real_t *const ux,
//                                  const real_t *const uy,
//                                  const real_t *const uz,
//                                  const ptrdiff_t h_stride,
//                                  const real_t *const hx,
//                                  const real_t *const hy,
//                                  const real_t *const hz,
//                                  const ptrdiff_t out_stride,
//                                  real_t *const outx,
//                                  real_t *const outy,
//                                  real_t *const outz);

// int tet4_neohookean_ogden_diag_opt(const ptrdiff_t nelements,
//                                     idx_t **const SFEM_RESTRICT elements,
//                                     const jacobian_t *const SFEM_RESTRICT g_jacobian_adjugate,
//                                     const jacobian_t *const SFEM_RESTRICT g_jacobian_determinant,
//                                     const real_t mu,
//                                     const real_t lambda,
//                                     const ptrdiff_t u_stride,
//                                     const real_t *const ux,
//                                     const real_t *const uy,
//                                     const real_t *const uz,
//                                     const ptrdiff_t out_stride,
//                                     real_t *const outx,
//                                     real_t *const outy,
//                                     real_t *const outz);

#ifdef __cplusplus
}
#endif
#endif  // TET4_NEOHOOKEAN_OGDEN_H
