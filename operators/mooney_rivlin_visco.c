#include "mooney_rivlin_visco.h"
#include "hex8/hex8_mooney_rivlin_visco_flexible.h"
#include <stdio.h>

// =============================================================================
// Mooney-Rivlin Viscoelastic Material Dispatchers
// Uses precomputed alpha/beta/gamma interface, supports arbitrary Prony terms
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
    const real_t *const SFEM_RESTRICT uz) {
    switch (element_type) {
        case HEX8: {
            return hex8_mooney_rivlin_visco_update_history_unique_hi(
                nelements, 1, nnodes, elements, points,
                C10, C01, K,
                num_prony_terms, alpha, beta,
                history_stride, history, new_history,
                u_stride, prev_ux, prev_uy, prev_uz, ux, uy, uz);
        }
        default: {
            SFEM_ERROR("mooney_rivlin_visco_update_history_flexible not implemented for type %d\n", element_type);
        }
    }
    return SFEM_FAILURE;
}

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
    real_t *const SFEM_RESTRICT out) {
    switch (element_type) {
        case HEX8: {
            return hex8_mooney_rivlin_visco_gradient_unique_hi(
                nelements, 1, nnodes, elements, points,
                C10, C01, K,
                num_prony_terms, alpha, beta, gamma,
                history_stride, history,
                u_stride, prev_ux, prev_uy, prev_uz, ux, uy, uz,
                3, &out[0], &out[1], &out[2]);
        }
        default: {
            SFEM_ERROR("mooney_rivlin_visco_gradient_flexible not implemented for type %d\n", element_type);
        }
    }
    return SFEM_FAILURE;
}

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
    real_t *const SFEM_RESTRICT values) {
    switch (element_type) {
        case HEX8: {
            return hex8_mooney_rivlin_visco_bsr_unique_hi(
                nelements, 1, nnodes, elements, points,
                C10, C01, K,
                num_prony_terms, alpha, beta, gamma,
                history_stride, history,
                u_stride, prev_ux, prev_uy, prev_uz, ux, uy, uz,
                1, values, (const idx_t*)rowptr, colidx);
        }
        default: {
            SFEM_ERROR("mooney_rivlin_visco_bsr_flexible not implemented for type %d\n", element_type);
        }
    }
    return SFEM_FAILURE;
}

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
    real_t *const SFEM_RESTRICT out) {
    switch (element_type) {
        case HEX8: {
            return hex8_mooney_rivlin_visco_hessian_diag_unique_hi(
                nelements, 1, nnodes, elements, points,
                C10, C01, K,
                num_prony_terms, alpha, beta, gamma,
                history_stride, history,
                u_stride, prev_ux, prev_uy, prev_uz, ux, uy, uz,
                3, &out[0], &out[1], &out[2]);
        }
        default: {
            SFEM_ERROR("mooney_rivlin_visco_hessian_diag_flexible not implemented for type %d\n", element_type);
        }
    }
    return SFEM_FAILURE;
}

