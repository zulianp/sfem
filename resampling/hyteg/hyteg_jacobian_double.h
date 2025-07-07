/**
 * Jacobian determinant calculation for tetrahedron categories
 * Auto-generated code - DO NOT EDIT MANUALLY
 * Data type: double
 */

#ifndef HYTEG_JACOBIAN_DOUBLE_H
#define HYTEG_JACOBIAN_DOUBLE_H

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
double det_jacobian_cat0_dbl(
    const double x0, const double y0, const double z0, //
    const double x1, const double y1, const double z1, //
    const double x2, const double y2, const double z2, //
    const double x3, const double y3, const double z3, //
    const double L //
);

/**
 * Calculate determinant of Jacobian matrix for tetrahedron category 1
 */
double det_jacobian_cat1_dbl(
    const double x0, const double y0, const double z0, //
    const double x1, const double y1, const double z1, //
    const double x2, const double y2, const double z2, //
    const double x3, const double y3, const double z3, //
    const double L //
);

/**
 * Calculate determinant of Jacobian matrix for tetrahedron category 2
 */
double det_jacobian_cat2_dbl(
    const double x0, const double y0, const double z0, //
    const double x1, const double y1, const double z1, // 
    const double x2, const double y2, const double z2, //
    const double x3, const double y3, const double z3, //
    const double L //
);

/**
 * Calculate determinant of Jacobian matrix for tetrahedron category 3
 */
double det_jacobian_cat3_dbl(
    const double x0, const double y0, const double z0, //
    const double x1, const double y1, const double z1, // 
    const double x2, const double y2, const double z2, //
    const double x3, const double y3, const double z3, //
    const double L //
);

/**
 * Calculate determinant of Jacobian matrix for tetrahedron category 4
 */
double det_jacobian_cat4_dbl(
    const double x0, const double y0, const double z0, //
    const double x1, const double y1, const double z1, // 
    const double x2, const double y2, const double z2, //
    const double x3, const double y3, const double z3, //
    const double L //
);

/**
 * Calculate determinant of Jacobian matrix for tetrahedron category 5
 */
double det_jacobian_cat5_dbl(
    const double x0, const double y0, const double z0, //
    const double x1, const double y1, const double z1, // 
    const double x2, const double y2, const double z2, //
    const double x3, const double y3, const double z3, //
    const double L //
);

#ifdef __cplusplus
}
#endif

#endif /* HYTEG_JACOBIAN_DOUBLE_H */
