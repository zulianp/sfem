#ifndef MOONEY_RIVLIN_VISCO_H
#define MOONEY_RIVLIN_VISCO_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int mooney_rivlin_visco_gradient_aos(const enum ElemType element_type,
                                     const ptrdiff_t nelements,
                                     const ptrdiff_t nnodes,
                                     idx_t **const SFEM_RESTRICT elements,
                                     geom_t **const SFEM_RESTRICT points,
                                     const real_t C10,
                                     const real_t K,
                                     const real_t C01,
                                     const real_t dt,
                                     const int num_prony_terms,
                                     const real_t *const SFEM_RESTRICT g,
                                     const real_t *const SFEM_RESTRICT tau,
                                     const ptrdiff_t history_stride,
                                     const real_t *const SFEM_RESTRICT history,
                                     const real_t *const SFEM_RESTRICT u,
                                     real_t *const SFEM_RESTRICT out);

int mooney_rivlin_visco_gradient_soa(const enum ElemType element_type,
                                     const ptrdiff_t nelements,
                                     const ptrdiff_t nnodes,
                                     idx_t **const SFEM_RESTRICT elements,
                                     geom_t **const SFEM_RESTRICT points,
                                     const real_t C10,
                                     const real_t K,
                                     const real_t C01,
                                     const real_t dt,
                                     const int num_prony_terms,
                                     const real_t *const SFEM_RESTRICT g,
                                     const real_t *const SFEM_RESTRICT tau,
                                     const ptrdiff_t history_stride,
                                     const real_t *const SFEM_RESTRICT history,
                                     real_t **const SFEM_RESTRICT u,
                                     real_t **const SFEM_RESTRICT out);

int mooney_rivlin_visco_update_history_aos(const enum ElemType element_type,
                                           const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const SFEM_RESTRICT elements,
                                           geom_t **const SFEM_RESTRICT points,
                                           const real_t C10,
                                           const real_t K,
                                           const real_t C01,
                                           const real_t dt,
                                           const int num_prony_terms,
                                           const real_t *const SFEM_RESTRICT g,
                                           const real_t *const SFEM_RESTRICT tau,
                                           const ptrdiff_t history_stride,
                                           const real_t *const SFEM_RESTRICT history,
                                           real_t *const SFEM_RESTRICT new_history,
                                           const real_t *const SFEM_RESTRICT u);

int mooney_rivlin_visco_update_history_soa(const enum ElemType element_type,
                                           const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const SFEM_RESTRICT elements,
                                           geom_t **const SFEM_RESTRICT points,
                                           const real_t C10,
                                           const real_t K,
                                           const real_t C01,
                                           const real_t dt,
                                           const int num_prony_terms,
                                           const real_t *const SFEM_RESTRICT g,
                                           const real_t *const SFEM_RESTRICT tau,
                                           const ptrdiff_t history_stride,
                                           const real_t *const SFEM_RESTRICT history,
                                           real_t *const SFEM_RESTRICT new_history,
                                           real_t **const SFEM_RESTRICT u);

int mooney_rivlin_visco_bsr(const enum ElemType element_type,
                            const ptrdiff_t nelements,
                            const ptrdiff_t nnodes,
                            idx_t **const SFEM_RESTRICT elements,
                            geom_t **const SFEM_RESTRICT points,
                            const real_t C10,
                            const real_t K,
                            const real_t C01,
                            const real_t dt,
                            const int num_prony_terms,
                            const real_t *const SFEM_RESTRICT g,
                            const real_t *const SFEM_RESTRICT tau,
                            const ptrdiff_t history_stride,
                            const real_t *const SFEM_RESTRICT history,
                            const ptrdiff_t u_stride,
                            const real_t *const SFEM_RESTRICT ux,
                            const real_t *const SFEM_RESTRICT uy,
                            const real_t *const SFEM_RESTRICT uz,
                            const count_t *const SFEM_RESTRICT rowptr,
                            const idx_t *const SFEM_RESTRICT colidx,
                            real_t *const SFEM_RESTRICT values);

int mooney_rivlin_visco_hessian_diag_aos(const enum ElemType element_type,
                                         const ptrdiff_t nelements,
                                         const ptrdiff_t nnodes,
                                         idx_t **const SFEM_RESTRICT elements,
                                         geom_t **const SFEM_RESTRICT points,
                                         const real_t C10,
                                         const real_t K,
                                         const real_t C01,
                                         const real_t dt,
                                         const int num_prony_terms,
                                         const real_t *const SFEM_RESTRICT g,
                                         const real_t *const SFEM_RESTRICT tau,
                                         const ptrdiff_t history_stride,
                                         const real_t *const SFEM_RESTRICT history,
                                         const real_t *const SFEM_RESTRICT u,
                                         real_t *const SFEM_RESTRICT out);

int mooney_rivlin_visco_hessian_diag_soa(const enum ElemType element_type,
                                         const ptrdiff_t nelements,
                                         const ptrdiff_t nnodes,
                                         idx_t **const SFEM_RESTRICT elements,
                                         geom_t **const SFEM_RESTRICT points,
                                         const real_t C10,
                                         const real_t K,
                                         const real_t C01,
                                         const real_t dt,
                                         const int num_prony_terms,
                                         const real_t *const SFEM_RESTRICT g,
                                         const real_t *const SFEM_RESTRICT tau,
                                         const ptrdiff_t history_stride,
                                         const real_t *const SFEM_RESTRICT history,
                                         real_t **const SFEM_RESTRICT u,
                                         real_t **const SFEM_RESTRICT out);

#ifdef __cplusplus
}
#endif
#endif // MOONEY_RIVLIN_VISCO_H



#define MOONEY_RIVLIN_VISCO_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int mooney_rivlin_visco_gradient_aos(const enum ElemType element_type,
                                     const ptrdiff_t nelements,
                                     const ptrdiff_t nnodes,
                                     idx_t **const SFEM_RESTRICT elements,
                                     geom_t **const SFEM_RESTRICT points,
                                     const real_t C10,
                                     const real_t K,
                                     const real_t C01,
                                     const real_t dt,
                                     const int num_prony_terms,
                                     const real_t *const SFEM_RESTRICT g,
                                     const real_t *const SFEM_RESTRICT tau,
                                     const ptrdiff_t history_stride,
                                     const real_t *const SFEM_RESTRICT history,
                                     const real_t *const SFEM_RESTRICT u,
                                     real_t *const SFEM_RESTRICT out);

int mooney_rivlin_visco_gradient_soa(const enum ElemType element_type,
                                     const ptrdiff_t nelements,
                                     const ptrdiff_t nnodes,
                                     idx_t **const SFEM_RESTRICT elements,
                                     geom_t **const SFEM_RESTRICT points,
                                     const real_t C10,
                                     const real_t K,
                                     const real_t C01,
                                     const real_t dt,
                                     const int num_prony_terms,
                                     const real_t *const SFEM_RESTRICT g,
                                     const real_t *const SFEM_RESTRICT tau,
                                     const ptrdiff_t history_stride,
                                     const real_t *const SFEM_RESTRICT history,
                                     real_t **const SFEM_RESTRICT u,
                                     real_t **const SFEM_RESTRICT out);

int mooney_rivlin_visco_update_history_aos(const enum ElemType element_type,
                                           const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const SFEM_RESTRICT elements,
                                           geom_t **const SFEM_RESTRICT points,
                                           const real_t C10,
                                           const real_t K,
                                           const real_t C01,
                                           const real_t dt,
                                           const int num_prony_terms,
                                           const real_t *const SFEM_RESTRICT g,
                                           const real_t *const SFEM_RESTRICT tau,
                                           const ptrdiff_t history_stride,
                                           const real_t *const SFEM_RESTRICT history,
                                           real_t *const SFEM_RESTRICT new_history,
                                           const real_t *const SFEM_RESTRICT u);

int mooney_rivlin_visco_update_history_soa(const enum ElemType element_type,
                                           const ptrdiff_t nelements,
                                           const ptrdiff_t nnodes,
                                           idx_t **const SFEM_RESTRICT elements,
                                           geom_t **const SFEM_RESTRICT points,
                                           const real_t C10,
                                           const real_t K,
                                           const real_t C01,
                                           const real_t dt,
                                           const int num_prony_terms,
                                           const real_t *const SFEM_RESTRICT g,
                                           const real_t *const SFEM_RESTRICT tau,
                                           const ptrdiff_t history_stride,
                                           const real_t *const SFEM_RESTRICT history,
                                           real_t *const SFEM_RESTRICT new_history,
                                           real_t **const SFEM_RESTRICT u);

int mooney_rivlin_visco_bsr(const enum ElemType element_type,
                            const ptrdiff_t nelements,
                            const ptrdiff_t nnodes,
                            idx_t **const SFEM_RESTRICT elements,
                            geom_t **const SFEM_RESTRICT points,
                            const real_t C10,
                            const real_t K,
                            const real_t C01,
                            const real_t dt,
                            const int num_prony_terms,
                            const real_t *const SFEM_RESTRICT g,
                            const real_t *const SFEM_RESTRICT tau,
                            const ptrdiff_t history_stride,
                            const real_t *const SFEM_RESTRICT history,
                            const ptrdiff_t u_stride,
                            const real_t *const SFEM_RESTRICT ux,
                            const real_t *const SFEM_RESTRICT uy,
                            const real_t *const SFEM_RESTRICT uz,
                            const count_t *const SFEM_RESTRICT rowptr,
                            const idx_t *const SFEM_RESTRICT colidx,
                            real_t *const SFEM_RESTRICT values);

int mooney_rivlin_visco_hessian_diag_aos(const enum ElemType element_type,
                                         const ptrdiff_t nelements,
                                         const ptrdiff_t nnodes,
                                         idx_t **const SFEM_RESTRICT elements,
                                         geom_t **const SFEM_RESTRICT points,
                                         const real_t C10,
                                         const real_t K,
                                         const real_t C01,
                                         const real_t dt,
                                         const int num_prony_terms,
                                         const real_t *const SFEM_RESTRICT g,
                                         const real_t *const SFEM_RESTRICT tau,
                                         const ptrdiff_t history_stride,
                                         const real_t *const SFEM_RESTRICT history,
                                         const real_t *const SFEM_RESTRICT u,
                                         real_t *const SFEM_RESTRICT out);

int mooney_rivlin_visco_hessian_diag_soa(const enum ElemType element_type,
                                         const ptrdiff_t nelements,
                                         const ptrdiff_t nnodes,
                                         idx_t **const SFEM_RESTRICT elements,
                                         geom_t **const SFEM_RESTRICT points,
                                         const real_t C10,
                                         const real_t K,
                                         const real_t C01,
                                         const real_t dt,
                                         const int num_prony_terms,
                                         const real_t *const SFEM_RESTRICT g,
                                         const real_t *const SFEM_RESTRICT tau,
                                         const ptrdiff_t history_stride,
                                         const real_t *const SFEM_RESTRICT history,
                                         real_t **const SFEM_RESTRICT u,
                                         real_t **const SFEM_RESTRICT out);

#ifdef __cplusplus
}
#endif
#endif // MOONEY_RIVLIN_VISCO_H


