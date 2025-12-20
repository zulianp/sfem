#ifndef HEX8_MOONEY_RIVLIN_VISCO_FLEXIBLE_H
#define HEX8_MOONEY_RIVLIN_VISCO_FLEXIBLE_H

/**
 * @file hex8_mooney_rivlin_visco_flexible.h
 * @brief Flexible Mooney-Rivlin viscoelastic operators for Hex8 elements.
 *
 * Key features:
 * 1. Only stores H_i in history (not S_dev_n) - reduced memory footprint
 * 2. S_dev is recomputed from displacement when needed
 * 3. Uses symbolic gamma parameter for arbitrary Prony term count
 * 4. All operators accept PRECOMPUTED alpha/beta/gamma coefficients
 *
 * Usage:
 *   1. Call precompute_prony_coeffs() once per timestep (when dt changes)
 *   2. Pass alpha[], beta[], gamma to all operators
 *
 * Parameter order convention:
 *   1. Mesh geometry (nelements, stride, nnodes, elements, points)
 *   2. Material parameters (C10, C01, K)
 *   3. Viscoelastic parameters (num_prony_terms, alpha, beta, gamma)
 *   4. History variables (history_stride, history, new_history)
 *   5. Displacement input (u_stride, prev_u*, u*)
 *   6. Output (out_stride, out* or BSR arrays)
 *
 * Memory layout per quadrature point: num_prony_terms * 6 floats (H_i only)
 */

#include <math.h>
#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Utility functions
// ============================================================================

/**
 * @brief Precompute Prony coefficients (alpha, beta, gamma).
 *
 * MUST be called once per timestep before using any operator.
 */
static inline void precompute_prony_coeffs(
    const real_t                      dt,
    const int                         num_prony_terms,
    const real_t *const SFEM_RESTRICT g,
    const real_t *const SFEM_RESTRICT tau,
    real_t *const SFEM_RESTRICT       alpha_out,
    real_t *const SFEM_RESTRICT       beta_out,
    real_t *                          gamma_out) {
    
    real_t g_inf = 1.0;
    real_t gamma_sum = 0.0;
    
    for (int i = 0; i < num_prony_terms; ++i) {
        g_inf -= g[i];
        const real_t x = dt / tau[i];
        const real_t alpha = exp(-x);
        const real_t beta = g[i] * (1.0 - alpha) / x;
        
        alpha_out[i] = alpha;
        beta_out[i] = beta;
        gamma_sum += beta;
    }
    
    *gamma_out = g_inf + gamma_sum;
}

/**
 * @brief Get history size per quadrature point.
 */
static inline ptrdiff_t history_size_per_qp_flexible(const int num_prony_terms) {
    return num_prony_terms * 6;
}

// ============================================================================
// Operators
// ============================================================================

/**
 * @brief Update history variables H_i.
 *
 * H_i^{n+1} = alpha_i * H_i^n + beta_i * (S_dev^{n+1} - S_dev^n)
 */
int hex8_mooney_rivlin_visco_update_history_unique_hi(
    // 1. Mesh geometry
    const ptrdiff_t                   nelements,
    const ptrdiff_t                   stride,
    const ptrdiff_t                   nnodes,
    idx_t **const SFEM_RESTRICT       elements,
    geom_t **const SFEM_RESTRICT      points,
    // 2. Material parameters
    const real_t                      C10,
    const real_t                      C01,
    const real_t                      K,
    // 3. Viscoelastic parameters (precomputed)
    const int                         num_prony_terms,
    const real_t *const SFEM_RESTRICT alpha,
    const real_t *const SFEM_RESTRICT beta,
    // 4. History variables
    const ptrdiff_t                   history_stride,
    const real_t *const SFEM_RESTRICT history,
    real_t *const SFEM_RESTRICT       new_history,
    // 5. Displacement input
    const ptrdiff_t                   u_stride,
    const real_t *const SFEM_RESTRICT prev_ux,
    const real_t *const SFEM_RESTRICT prev_uy,
    const real_t *const SFEM_RESTRICT prev_uz,
    const real_t *const SFEM_RESTRICT ux,
    const real_t *const SFEM_RESTRICT uy,
    const real_t *const SFEM_RESTRICT uz);

/**
 * @brief Compute gradient (residual).
 */
int hex8_mooney_rivlin_visco_gradient_unique_hi(
    // 1. Mesh geometry
    const ptrdiff_t                   nelements,
    const ptrdiff_t                   stride,
    const ptrdiff_t                   nnodes,
    idx_t **const SFEM_RESTRICT       elements,
    geom_t **const SFEM_RESTRICT      points,
    // 2. Material parameters
    const real_t                      C10,
    const real_t                      C01,
    const real_t                      K,
    // 3. Viscoelastic parameters (precomputed)
    const int                         num_prony_terms,
    const real_t *const SFEM_RESTRICT alpha,
    const real_t *const SFEM_RESTRICT beta,
    const real_t                      gamma,
    // 4. History variables
    const ptrdiff_t                   history_stride,
    const real_t *const SFEM_RESTRICT history,
    // 5. Displacement input
    const ptrdiff_t                   u_stride,
    const real_t *const SFEM_RESTRICT prev_ux,
    const real_t *const SFEM_RESTRICT prev_uy,
    const real_t *const SFEM_RESTRICT prev_uz,
    const real_t *const SFEM_RESTRICT ux,
    const real_t *const SFEM_RESTRICT uy,
    const real_t *const SFEM_RESTRICT uz,
    // 6. Output
    const ptrdiff_t                   out_stride,
    real_t *const SFEM_RESTRICT       outx,
    real_t *const SFEM_RESTRICT       outy,
    real_t *const SFEM_RESTRICT       outz);

/**
 * @brief Assemble Hessian in BSR format.
 */
int hex8_mooney_rivlin_visco_bsr_unique_hi(
    // 1. Mesh geometry
    const ptrdiff_t                   nelements,
    const ptrdiff_t                   stride,
    const ptrdiff_t                   nnodes,
    idx_t **const SFEM_RESTRICT       elements,
    geom_t **const SFEM_RESTRICT      points,
    // 2. Material parameters
    const real_t                      C10,
    const real_t                      C01,
    const real_t                      K,
    // 3. Viscoelastic parameters (precomputed)
    const int                         num_prony_terms,
    const real_t *const SFEM_RESTRICT alpha,
    const real_t *const SFEM_RESTRICT beta,
    const real_t                      gamma,
    // 4. History variables
    const ptrdiff_t                   history_stride,
    const real_t *const SFEM_RESTRICT history,
    // 5. Displacement input
    const ptrdiff_t                   u_stride,
    const real_t *const SFEM_RESTRICT prev_ux,
    const real_t *const SFEM_RESTRICT prev_uy,
    const real_t *const SFEM_RESTRICT prev_uz,
    const real_t *const SFEM_RESTRICT ux,
    const real_t *const SFEM_RESTRICT uy,
    const real_t *const SFEM_RESTRICT uz,
    // 6. BSR output
    const ptrdiff_t                   out_stride,
    real_t *const SFEM_RESTRICT       values,
    const idx_t *const SFEM_RESTRICT  rowptr,
    const idx_t *const SFEM_RESTRICT  colidx);

/**
 * @brief Compute Hessian diagonal.
 */
int hex8_mooney_rivlin_visco_hessian_diag_unique_hi(
    // 1. Mesh geometry
    const ptrdiff_t                   nelements,
    const ptrdiff_t                   stride,
    const ptrdiff_t                   nnodes,
    idx_t **const SFEM_RESTRICT       elements,
    geom_t **const SFEM_RESTRICT      points,
    // 2. Material parameters
    const real_t                      C10,
    const real_t                      C01,
    const real_t                      K,
    // 3. Viscoelastic parameters (precomputed)
    const int                         num_prony_terms,
    const real_t *const SFEM_RESTRICT alpha,
    const real_t *const SFEM_RESTRICT beta,
    const real_t                      gamma,
    // 4. History variables
    const ptrdiff_t                   history_stride,
    const real_t *const SFEM_RESTRICT history,
    // 5. Displacement input
    const ptrdiff_t                   u_stride,
    const real_t *const SFEM_RESTRICT prev_ux,
    const real_t *const SFEM_RESTRICT prev_uy,
    const real_t *const SFEM_RESTRICT prev_uz,
    const real_t *const SFEM_RESTRICT ux,
    const real_t *const SFEM_RESTRICT uy,
    const real_t *const SFEM_RESTRICT uz,
    // 6. Output
    const ptrdiff_t                   out_stride,
    real_t *const SFEM_RESTRICT       outx,
    real_t *const SFEM_RESTRICT       outy,
    real_t *const SFEM_RESTRICT       outz);

#ifdef __cplusplus
}
#endif

#endif  // HEX8_MOONEY_RIVLIN_VISCO_FLEXIBLE_H

