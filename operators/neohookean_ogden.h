#ifndef NEOHOOKEAN_OGDEN_H
#define NEOHOOKEAN_OGDEN_H

#include <stddef.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

//////////////////////////
// Structure of arrays
//////////////////////////

int neohookean_ogden_value_soa(const enum ElemType          element_type,
                               const ptrdiff_t              nelements,
                               const ptrdiff_t              nnodes,
                               idx_t **const SFEM_RESTRICT  elements,
                               geom_t **const SFEM_RESTRICT points,
                               const real_t                 mu,
                               const real_t                 lambda,
                               real_t **const SFEM_RESTRICT u,
                               real_t *const SFEM_RESTRICT  value);

int neohookean_ogden_gradient_soa(const enum ElemType          element_type,
                                  const ptrdiff_t              nelements,
                                  const ptrdiff_t              nnodes,
                                  idx_t **const SFEM_RESTRICT  elements,
                                  geom_t **const SFEM_RESTRICT points,
                                  const real_t                 mu,
                                  const real_t                 lambda,
                                  real_t **const SFEM_RESTRICT u,
                                  real_t **const SFEM_RESTRICT values);

int neohookean_ogden_apply_soa(const enum ElemType          element_type,
                               const ptrdiff_t              nelements,
                               const ptrdiff_t              nnodes,
                               idx_t **const SFEM_RESTRICT  elements,
                               geom_t **const SFEM_RESTRICT points,
                               const real_t                 mu,
                               const real_t                 lambda,
                               real_t **const SFEM_RESTRICT u,
                               real_t **const SFEM_RESTRICT h,
                               real_t **const SFEM_RESTRICT values);

int neohookean_ogden_hessian_soa(const enum ElemType                element_type,
                                 const ptrdiff_t                    nelements,
                                 const ptrdiff_t                    nnodes,
                                 idx_t **const SFEM_RESTRICT        elements,
                                 geom_t **const SFEM_RESTRICT       points,
                                 const real_t                       mu,
                                 const real_t                       lambda,
                                 const count_t *const SFEM_RESTRICT rowptr,
                                 const idx_t *const SFEM_RESTRICT   colidx,
                                 real_t **const SFEM_RESTRICT       values);

//////////////////////////
// Array of structures
//////////////////////////

int neohookean_ogden_value_aos(const enum ElemType               element_type,
                               const ptrdiff_t                   nelements,
                               const ptrdiff_t                   nnodes,
                               idx_t **const SFEM_RESTRICT       elements,
                               geom_t **const SFEM_RESTRICT      points,
                               const real_t                      mu,
                               const real_t                      lambda,
                               const real_t *const SFEM_RESTRICT u,
                               real_t *const SFEM_RESTRICT       value);

int neohookean_ogden_apply_aos(const enum ElemType               element_type,
                               const ptrdiff_t                   nelements,
                               const ptrdiff_t                   nnodes,
                               idx_t **const SFEM_RESTRICT       elements,
                               geom_t **const SFEM_RESTRICT      points,
                               const real_t                      mu,
                               const real_t                      lambda,
                               const real_t *const SFEM_RESTRICT u,
                               const real_t *const SFEM_RESTRICT h,
                               real_t *const SFEM_RESTRICT       values);

int neohookean_ogden_gradient_aos(const enum ElemType               element_type,
                                  const ptrdiff_t                   nelements,
                                  const ptrdiff_t                   nnodes,
                                  idx_t **const SFEM_RESTRICT       elements,
                                  geom_t **const SFEM_RESTRICT      points,
                                  const real_t                      mu,
                                  const real_t                      lambda,
                                  const real_t *const SFEM_RESTRICT u,
                                  real_t *const SFEM_RESTRICT       values);

int neohookean_ogden_hessian_aos(const enum ElemType                element_type,
                                 const ptrdiff_t                    nelements,
                                 const ptrdiff_t                    nnodes,
                                 idx_t **const SFEM_RESTRICT        elements,
                                 geom_t **const SFEM_RESTRICT       points,
                                 const real_t                       mu,
                                 const real_t                       lambda,
                                 const real_t *const SFEM_RESTRICT  u,
                                 const count_t *const SFEM_RESTRICT rowptr,
                                 const idx_t *const SFEM_RESTRICT   colidx,
                                 real_t *const SFEM_RESTRICT        values);

int neohookean_ogden_diag_aos(const enum ElemType               element_type,
                              const ptrdiff_t                   nelements,
                              const ptrdiff_t                   nnodes,
                              idx_t **const SFEM_RESTRICT       elements,
                              geom_t **const SFEM_RESTRICT      points,
                              const real_t                      mu,
                              const real_t                      lambda,
                              const real_t *const SFEM_RESTRICT u,
                              real_t *const SFEM_RESTRICT       values);

int neohookean_ogden_hessian_partial_assembly(const enum ElemType                   element_type,
                                              const ptrdiff_t                       nelements,
                                              const ptrdiff_t                       stride,
                                              idx_t **const SFEM_RESTRICT           elements,
                                              geom_t **const SFEM_RESTRICT          points,
                                              const real_t                          mu,
                                              const real_t                          lambda,
                                              const ptrdiff_t                       u_stride,
                                              const real_t *const SFEM_RESTRICT     ux,
                                              const real_t *const SFEM_RESTRICT     uy,
                                              const real_t *const SFEM_RESTRICT     uz,
                                              metric_tensor_t *const SFEM_RESTRICT partial_assembly);

int neohookean_ogden_partial_assembly_apply(const enum ElemType                   element_type,
                                            const ptrdiff_t                       nelements,
                                            const ptrdiff_t                       stride,
                                            idx_t **const SFEM_RESTRICT           elements,
                                            const metric_tensor_t *const SFEM_RESTRICT partial_assembly,
                                            const ptrdiff_t                       h_stride,
                                            const real_t *const                   hx,
                                            const real_t *const                   hy,
                                            const real_t *const                   hz,
                                            const ptrdiff_t                       out_stride,
                                            real_t *const                         outx,
                                            real_t *const                         outy,
                                            real_t *const                         outz);

int neohookean_ogden_compressed_partial_assembly_apply(const enum ElemType                  element_type,
                                                       const ptrdiff_t                      nelements,
                                                       const ptrdiff_t                      stride,
                                                       idx_t **const SFEM_RESTRICT          elements,
                                                       const compressed_t *const SFEM_RESTRICT   partial_assembly,
                                                       const scaling_t *const SFEM_RESTRICT scaling,
                                                       const ptrdiff_t                      h_stride,
                                                       const real_t *const                  hx,
                                                       const real_t *const                  hy,
                                                       const real_t *const                  hz,
                                                       const ptrdiff_t                      out_stride,
                                                       real_t *const                        outx,
                                                       real_t *const                        outy,
                                                       real_t *const                        outz);

#ifdef __cplusplus
}
#endif
#endif  // NEOHOOKEAN_OGDEN_H
