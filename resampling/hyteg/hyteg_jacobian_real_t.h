/**
 * Jacobian determinant calculation for tetrahedron categories
 * Auto-generated code - DO NOT EDIT MANUALLY
 * Data type: real_t
 */

#ifndef HYTEG_JACOBIAN_REAL_T_H
#define HYTEG_JACOBIAN_REAL_T_H

#include <stddef.h>
#include <stdlib.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Calculate determinant of Jacobian matrix for tetrahedron category 0
 */
real_t det_jacobian_cat0_real(
    const real_t x0, const real_t y0, const real_t z0, //
    const real_t x1, const real_t y1, const real_t z1, //
    const real_t x2, const real_t y2, const real_t z2, //
    const real_t x3, const real_t y3, const real_t z3, //
    const real_t L //
);

/**
 * Calculate determinant of Jacobian matrix for tetrahedron category 1
 */
real_t det_jacobian_cat1_real(
    const real_t x0, const real_t y0, const real_t z0, //
    const real_t x1, const real_t y1, const real_t z1, //
    const real_t x2, const real_t y2, const real_t z2, //
    const real_t x3, const real_t y3, const real_t z3, //
    const real_t L //
);

/**
 * Calculate determinant of Jacobian matrix for tetrahedron category 2
 */
real_t det_jacobian_cat2_real(
    const real_t x0, const real_t y0, const real_t z0, //
    const real_t x1, const real_t y1, const real_t z1, // 
    const real_t x2, const real_t y2, const real_t z2, //
    const real_t x3, const real_t y3, const real_t z3, //
    const real_t L //
);

/**
 * Calculate determinant of Jacobian matrix for tetrahedron category 3
 */
real_t det_jacobian_cat3_real(
    const real_t x0, const real_t y0, const real_t z0, //
    const real_t x1, const real_t y1, const real_t z1, // 
    const real_t x2, const real_t y2, const real_t z2, //
    const real_t x3, const real_t y3, const real_t z3, //
    const real_t L //
);

/**
 * Calculate determinant of Jacobian matrix for tetrahedron category 4
 */
real_t det_jacobian_cat4_real(
    const real_t x0, const real_t y0, const real_t z0, //
    const real_t x1, const real_t y1, const real_t z1, // 
    const real_t x2, const real_t y2, const real_t z2, //
    const real_t x3, const real_t y3, const real_t z3, //
    const real_t L //
);

/**
 * Calculate determinant of Jacobian matrix for tetrahedron category 5
 */
real_t det_jacobian_cat5_real(
    const real_t x0, const real_t y0, const real_t z0, //
    const real_t x1, const real_t y1, const real_t z1, // 
    const real_t x2, const real_t y2, const real_t z2, //
    const real_t x3, const real_t y3, const real_t z3, //
    const real_t L //
);

#ifdef __cplusplus
}
#endif

#endif /* HYTEG_JACOBIAN_REAL_T_H */
