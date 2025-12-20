#ifndef HEX8_MOONEY_RIVLIN_VISCO_H
#define HEX8_MOONEY_RIVLIN_VISCO_H

/**
 * @file hex8_mooney_rivlin_visco.h
 * @brief Fixed Mooney-Rivlin viscoelastic operators (10 Prony terms hardcoded).
 *
 * Parameter order convention:
 *   1. Mesh geometry (nelements, stride, nnodes, elements, points)
 *   2. Material parameters (C10, C01, K)
 *   3. Viscoelastic parameters (dt, num_prony_terms, g, tau)
 *   4. History variables (history_stride, history, new_history)
 *   5. Displacement input (u_stride, ux, uy, uz)
 *   6. Output (out_stride, out* or BSR arrays)
 */

#include <stddef.h>
#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

int hex8_mooney_rivlin_visco_gradient(
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
    // 3. Viscoelastic parameters
    const real_t                      dt,
    const int                         num_prony_terms,
    const real_t *const SFEM_RESTRICT g,
    const real_t *const SFEM_RESTRICT tau,
    // 4. History variables
    const ptrdiff_t                   history_stride,
    const real_t *const SFEM_RESTRICT history,
    // 5. Displacement input
    const ptrdiff_t                   u_stride,
    const real_t *const SFEM_RESTRICT ux,
    const real_t *const SFEM_RESTRICT uy,
    const real_t *const SFEM_RESTRICT uz,
    // 6. Output
    const ptrdiff_t                   out_stride,
    real_t *const SFEM_RESTRICT       outx,
    real_t *const SFEM_RESTRICT       outy,
    real_t *const SFEM_RESTRICT       outz);

int hex8_mooney_rivlin_visco_update_history(
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
    // 3. Viscoelastic parameters
    const real_t                      dt,
    const int                         num_prony_terms,
    const real_t *const SFEM_RESTRICT g,
    const real_t *const SFEM_RESTRICT tau,
    // 4. History variables
    const ptrdiff_t                   history_stride,
    const real_t *const SFEM_RESTRICT history,
    real_t *const SFEM_RESTRICT       new_history,
    // 5. Displacement input
    const ptrdiff_t                   u_stride,
    const real_t *const SFEM_RESTRICT ux,
    const real_t *const SFEM_RESTRICT uy,
    const real_t *const SFEM_RESTRICT uz);

int hex8_mooney_rivlin_visco_bsr(
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
    // 3. Viscoelastic parameters
    const real_t                      dt,
    const int                         num_prony_terms,
    const real_t *const SFEM_RESTRICT g,
    const real_t *const SFEM_RESTRICT tau,
    // 4. History variables
    const ptrdiff_t                   history_stride,
    const real_t *const SFEM_RESTRICT history,
    // 5. Displacement input
    const ptrdiff_t                   u_stride,
    const real_t *const SFEM_RESTRICT ux,
    const real_t *const SFEM_RESTRICT uy,
    const real_t *const SFEM_RESTRICT uz,
    // 6. BSR output
    const ptrdiff_t                   out_stride,
    real_t *const SFEM_RESTRICT       values,
    const idx_t *const SFEM_RESTRICT  rowptr,
    const idx_t *const SFEM_RESTRICT  colidx);

int hex8_mooney_rivlin_visco_hessian_diag(
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
    // 3. Viscoelastic parameters
    const real_t                      dt,
    const int                         num_prony_terms,
    const real_t *const SFEM_RESTRICT g,
    const real_t *const SFEM_RESTRICT tau,
    // 4. History variables
    const ptrdiff_t                   history_stride,
    const real_t *const SFEM_RESTRICT history,
    // 5. Displacement input
    const ptrdiff_t                   u_stride,
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

#endif  // HEX8_MOONEY_RIVLIN_VISCO_H

