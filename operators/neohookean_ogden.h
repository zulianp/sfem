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

int neohookean_ogden_value_soa(const enum ElemType element_type,
                               const ptrdiff_t nelements,
                               const ptrdiff_t nnodes,
                               idx_t **const SFEM_RESTRICT elements,
                               geom_t **const SFEM_RESTRICT points,
                               const real_t mu,
                               const real_t lambda,
                               const real_t **const SFEM_RESTRICT u,
                               real_t *const SFEM_RESTRICT value);

int neohookean_ogden_gradient_soa(const enum ElemType element_type,
                                  const ptrdiff_t nelements,
                                  const ptrdiff_t nnodes,
                                  idx_t **const SFEM_RESTRICT elements,
                                  geom_t **const SFEM_RESTRICT points,
                                  const real_t mu,
                                  const real_t lambda,
                                  const real_t **const SFEM_RESTRICT u,
                                  real_t **const SFEM_RESTRICT values);

int neohookean_ogden_apply_soa(const enum ElemType element_type,
                               const ptrdiff_t nelements,
                               const ptrdiff_t nnodes,
                               idx_t **const SFEM_RESTRICT elements,
                               geom_t **const SFEM_RESTRICT points,
                               const real_t mu,
                               const real_t lambda,
                               const real_t **const SFEM_RESTRICT u,
                               const real_t **const SFEM_RESTRICT h,
                               real_t **const SFEM_RESTRICT values);

int neohookean_ogden_hessian_soa(const enum ElemType element_type,
                                 const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 const real_t mu,
                                 const real_t lambda,
                                 const count_t *const SFEM_RESTRICT rowptr,
                                 const idx_t *const SFEM_RESTRICT colidx,
                                 real_t **const SFEM_RESTRICT values);

//////////////////////////
// Array of structures
//////////////////////////

int neohookean_ogden_value_aos(const enum ElemType element_type,
                               const ptrdiff_t nelements,
                               const ptrdiff_t nnodes,
                               idx_t **const SFEM_RESTRICT elements,
                               geom_t **const SFEM_RESTRICT points,
                               const real_t mu,
                               const real_t lambda,
                               const real_t *const SFEM_RESTRICT u,
                               real_t *const SFEM_RESTRICT value);

int neohookean_ogden_apply_aos(const enum ElemType element_type,
                               const ptrdiff_t nelements,
                               const ptrdiff_t nnodes,
                               idx_t **const SFEM_RESTRICT elements,
                               geom_t **const SFEM_RESTRICT points,
                               const real_t mu,
                               const real_t lambda,
                               const real_t *const SFEM_RESTRICT u,
                               const real_t *const SFEM_RESTRICT h,
                               real_t *const SFEM_RESTRICT values);

int neohookean_ogden_gradient_aos(const enum ElemType element_type,
                                  const ptrdiff_t nelements,
                                  const ptrdiff_t nnodes,
                                  idx_t **const SFEM_RESTRICT elements,
                                  geom_t **const SFEM_RESTRICT points,
                                  const real_t mu,
                                  const real_t lambda,
                                  const real_t *const SFEM_RESTRICT u,
                                  real_t *const SFEM_RESTRICT values);

int neohookean_ogden_hessian_aos(const enum ElemType element_type,
                                 const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elements,
                                 geom_t **const SFEM_RESTRICT points,
                                 const real_t mu,
                                 const real_t lambda,
                                 const real_t *const SFEM_RESTRICT u,
                                 const count_t *const SFEM_RESTRICT rowptr,
                                 const idx_t *const SFEM_RESTRICT colidx,
                                 real_t *const SFEM_RESTRICT values);

int neohookean_ogden_diag_aos(const enum ElemType element_type,
                              const ptrdiff_t nelements,
                              const ptrdiff_t nnodes,
                              idx_t **const SFEM_RESTRICT elements,
                              geom_t **const SFEM_RESTRICT points,
                              const real_t mu,
                              const real_t lambda,
                              const real_t *const SFEM_RESTRICT u,
                              real_t *const SFEM_RESTRICT values);

#ifdef __cplusplus
}
#endif
#endif  // NEOHOOKEAN_OGDEN_H
