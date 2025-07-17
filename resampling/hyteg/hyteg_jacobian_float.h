/**
 * Jacobian determinant calculation for tetrahedron categories
 * Auto-generated code - DO NOT EDIT MANUALLY
 * Data type: float
 */

#ifndef HYTEG_JACOBIAN_FLOAT_H
#define HYTEG_JACOBIAN_FLOAT_H

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
float det_jacobian_cat0_float(
    const float x0, const float y0, const float z0, //
    const float x1, const float y1, const float z1, //
    const float x2, const float y2, const float z2, //
    const float x3, const float y3, const float z3, //
    const float L //
);

/**
 * Calculate determinant of Jacobian matrix for tetrahedron category 1
 */
float det_jacobian_cat1_float(
    const float x0, const float y0, const float z0, //
    const float x1, const float y1, const float z1, //
    const float x2, const float y2, const float z2, //
    const float x3, const float y3, const float z3, //
    const float L //
);

/**
 * Calculate determinant of Jacobian matrix for tetrahedron category 2
 */
float det_jacobian_cat2_float(
    const float x0, const float y0, const float z0, //
    const float x1, const float y1, const float z1, // 
    const float x2, const float y2, const float z2, //
    const float x3, const float y3, const float z3, //
    const float L //
);

/**
 * Calculate determinant of Jacobian matrix for tetrahedron category 3
 */
float det_jacobian_cat3_float(
    const float x0, const float y0, const float z0, //
    const float x1, const float y1, const float z1, // 
    const float x2, const float y2, const float z2, //
    const float x3, const float y3, const float z3, //
    const float L //
);

/**
 * Calculate determinant of Jacobian matrix for tetrahedron category 4
 */
float det_jacobian_cat4_float(
    const float x0, const float y0, const float z0, //
    const float x1, const float y1, const float z1, // 
    const float x2, const float y2, const float z2, //
    const float x3, const float y3, const float z3, //
    const float L //
);

/**
 * Calculate determinant of Jacobian matrix for tetrahedron category 5
 */
float det_jacobian_cat5_float(
    const float x0, const float y0, const float z0, //
    const float x1, const float y1, const float z1, // 
    const float x2, const float y2, const float z2, //
    const float x3, const float y3, const float z3, //
    const float L //
);

#ifdef __cplusplus
}
#endif

#endif /* HYTEG_JACOBIAN_FLOAT_H */
