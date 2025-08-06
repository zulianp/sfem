/**
 * Jacobian matrix calculation for tetrahedron categories
 * Auto-generated code - DO NOT EDIT MANUALLY
 * Data type: float
 */

#ifndef HYTEG_JACOBIAN_MATRIX_FLOAT_H
#define HYTEG_JACOBIAN_MATRIX_FLOAT_H

#include <stddef.h>
#include <stdlib.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Calculate 3x3 Jacobian matrix for tetrahedron category 0
 * @param x0, y0, z0 Coordinates of vertex 0
 * @param x1, y1, z1 Coordinates of vertex 1
 * @param x2, y2, z2 Coordinates of vertex 2
 * @param x3, y3, z3 Coordinates of vertex 3
 * @param L Level parameter
 * @param J Output 3x3 Jacobian matrix stored in row-major order
 */
void jacobian_matrix_cat0_float(
    const float x0, const float y0, const float z0, //
    const float x1, const float y1, const float z1, //
    const float x2, const float y2, const float z2, //
    const float x3, const float y3, const float z3, //
    const float L, //
    float J[9] //
);

/**
 * Calculate 3x3 Jacobian matrix for tetrahedron category 1
 */
void jacobian_matrix_cat1_float(
    const float x0, const float y0, const float z0, //
    const float x1, const float y1, const float z1, //
    const float x2, const float y2, const float z2, //
    const float x3, const float y3, const float z3, //
    const float L, //
    float J[9] //
);

/**
 * Calculate 3x3 Jacobian matrix for tetrahedron category 2
 */
void jacobian_matrix_cat2_float(
    const float x0, const float y0, const float z0, //
    const float x1, const float y1, const float z1, //
    const float x2, const float y2, const float z2, //
    const float x3, const float y3, const float z3, //
    const float L, //
    float J[9] //
);

/**
 * Calculate 3x3 Jacobian matrix for tetrahedron category 3
 */
void jacobian_matrix_cat3_float(
    const float x0, const float y0, const float z0, //
    const float x1, const float y1, const float z1, //
    const float x2, const float y2, const float z2, //
    const float x3, const float y3, const float z3, //
    const float L, //
    float J[9] //
);

/**
 * Calculate 3x3 Jacobian matrix for tetrahedron category 4
 */
void jacobian_matrix_cat4_float(
    const float x0, const float y0, const float z0, //
    const float x1, const float y1, const float z1, //
    const float x2, const float y2, const float z2, //
    const float x3, const float y3, const float z3, //
    const float L, //
    float J[9] //
);

/**
 * Calculate 3x3 Jacobian matrix for tetrahedron category 5
 */
void jacobian_matrix_cat5_float(
    const float x0, const float y0, const float z0, //
    const float x1, const float y1, const float z1, //
    const float x2, const float y2, const float z2, //
    const float x3, const float y3, const float z3, //
    const float L, //
    float J[9] //
);

#ifdef __cplusplus
}
#endif

#endif /* HYTEG_JACOBIAN_MATRIX_FLOAT_H */
