#ifndef MOONEY_RIVLIN_VISCO_H
#define MOONEY_RIVLIN_VISCO_H

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Mooney-Rivlin Viscoelastic Material
// Uses precomputed alpha/beta/gamma coefficients, supports arbitrary Prony terms
// History layout: num_prony_terms * 6 floats per quadrature point (only H_i)
// =============================================================================

int mooney_rivlin_visco_update_history_flexible(
    const enum ElemType element_type,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const SFEM_RESTRICT elements,
    geom_t **const SFEM_RESTRICT points,
    const real_t C10,
    const real_t C01,
    const real_t K,
    const int num_prony_terms,
    const real_t *const SFEM_RESTRICT alpha,
    const real_t *const SFEM_RESTRICT beta,
    const ptrdiff_t history_stride,
    const real_t *const SFEM_RESTRICT history,
    real_t *const SFEM_RESTRICT new_history,
    const ptrdiff_t u_stride,
    const real_t *const SFEM_RESTRICT prev_ux,
    const real_t *const SFEM_RESTRICT prev_uy,
    const real_t *const SFEM_RESTRICT prev_uz,
    const real_t *const SFEM_RESTRICT ux,
    const real_t *const SFEM_RESTRICT uy,
    const real_t *const SFEM_RESTRICT uz);

int mooney_rivlin_visco_gradient_flexible(
    const enum ElemType element_type,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const SFEM_RESTRICT elements,
    geom_t **const SFEM_RESTRICT points,
    const real_t C10,
    const real_t C01,
    const real_t K,
    const int num_prony_terms,
    const real_t *const SFEM_RESTRICT alpha,
    const real_t *const SFEM_RESTRICT beta,
    const real_t gamma,
    const ptrdiff_t history_stride,
    const real_t *const SFEM_RESTRICT history,
    const ptrdiff_t u_stride,
    const real_t *const SFEM_RESTRICT prev_ux,
    const real_t *const SFEM_RESTRICT prev_uy,
    const real_t *const SFEM_RESTRICT prev_uz,
    const real_t *const SFEM_RESTRICT ux,
    const real_t *const SFEM_RESTRICT uy,
    const real_t *const SFEM_RESTRICT uz,
    real_t *const SFEM_RESTRICT out);

int mooney_rivlin_visco_bsr_flexible(
    const enum ElemType element_type,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const SFEM_RESTRICT elements,
    geom_t **const SFEM_RESTRICT points,
    const real_t C10,
    const real_t C01,
    const real_t K,
    const int num_prony_terms,
    const real_t *const SFEM_RESTRICT alpha,
    const real_t *const SFEM_RESTRICT beta,
    const real_t gamma,
    const ptrdiff_t history_stride,
    const real_t *const SFEM_RESTRICT history,
    const ptrdiff_t u_stride,
    const real_t *const SFEM_RESTRICT prev_ux,
    const real_t *const SFEM_RESTRICT prev_uy,
    const real_t *const SFEM_RESTRICT prev_uz,
    const real_t *const SFEM_RESTRICT ux,
    const real_t *const SFEM_RESTRICT uy,
    const real_t *const SFEM_RESTRICT uz,
    const count_t *const SFEM_RESTRICT rowptr,
    const idx_t *const SFEM_RESTRICT colidx,
    real_t *const SFEM_RESTRICT values);

int mooney_rivlin_visco_hessian_diag_flexible(
    const enum ElemType element_type,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const SFEM_RESTRICT elements,
    geom_t **const SFEM_RESTRICT points,
    const real_t C10,
    const real_t C01,
    const real_t K,
    const int num_prony_terms,
    const real_t *const SFEM_RESTRICT alpha,
    const real_t *const SFEM_RESTRICT beta,
    const real_t gamma,
    const ptrdiff_t history_stride,
    const real_t *const SFEM_RESTRICT history,
    const ptrdiff_t u_stride,
    const real_t *const SFEM_RESTRICT prev_ux,
    const real_t *const SFEM_RESTRICT prev_uy,
    const real_t *const SFEM_RESTRICT prev_uz,
    const real_t *const SFEM_RESTRICT ux,
    const real_t *const SFEM_RESTRICT uy,
    const real_t *const SFEM_RESTRICT uz,
    real_t *const SFEM_RESTRICT out);

#ifdef __cplusplus
}
#endif

#endif  // MOONEY_RIVLIN_VISCO_H

