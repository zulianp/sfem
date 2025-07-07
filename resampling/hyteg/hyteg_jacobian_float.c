/**
 * Jacobian determinant calculation for tetrahedron categories
 * Auto-generated code - DO NOT EDIT MANUALLY
 * Data type: float
 */

#include "hyteg_jacobian_float.h"
#include <math.h>


float det_jacobian_cat0_float(
    const float x0, const float y0, const float z0, //
    const float x1, const float y1, const float z1, // 
    const float x2, const float y2, const float z2, //
    const float x3, const float y3, const float z3, //
    const float L //
) {
    // Calculate determinant of Jacobian for tetrahedron category 0
    float det = (x0*(y1*(-z2 + z3) + y2*(z1 - z3) + y3*(-z1 + z2)) + x1*(y0*(z2 - z3) + y2*(-z0 + z3) + y3*(z0 - z2)) + x2*(y0*(-z1 + z3) + y1*(z0 - z3) + y3*(-z0 + z1)) + x3*(y0*(z1 - z2) + y1*(-z0 + z2) + y2*(z0 - z1)))/pow(L, 3);
    return det;
}

float det_jacobian_cat1_float(
    const float x0, const float y0, const float z0, //
    const float x1, const float y1, const float z1, // 
    const float x2, const float y2, const float z2, //
    const float x3, const float y3, const float z3, //
    const float L //
) {
    // Calculate determinant of Jacobian for tetrahedron category 1
    float det = (x0*(y1*(-z2 + z3) + y2*(z1 - z3) + y3*(-z1 + z2)) + x1*(y0*(z2 - z3) + y2*(-z0 + z3) + y3*(z0 - z2)) + x2*(y0*(-z1 + z3) + y1*(z0 - z3) + y3*(-z0 + z1)) + x3*(y0*(z1 - z2) + y1*(-z0 + z2) + y2*(z0 - z1)))/pow(L, 3);
    return det;
}

float det_jacobian_cat2_float(
    const float x0, const float y0, const float z0, //
    const float x1, const float y1, const float z1, // 
    const float x2, const float y2, const float z2, //
    const float x3, const float y3, const float z3, //
    const float L //
) {
    // Calculate determinant of Jacobian for tetrahedron category 2
    float det = (x0*(y1*(-z2 + z3) + y2*(z1 - z3) + y3*(-z1 + z2)) + x1*(y0*(z2 - z3) + y2*(-z0 + z3) + y3*(z0 - z2)) + x2*(y0*(-z1 + z3) + y1*(z0 - z3) + y3*(-z0 + z1)) + x3*(y0*(z1 - z2) + y1*(-z0 + z2) + y2*(z0 - z1)))/pow(L, 3);
    return det;
}

float det_jacobian_cat3_float(
    const float x0, const float y0, const float z0, //
    const float x1, const float y1, const float z1, // 
    const float x2, const float y2, const float z2, //
    const float x3, const float y3, const float z3, //
    const float L //
) {
    // Calculate determinant of Jacobian for tetrahedron category 3
    float det = (x0*(y1*(-z2 + z3) + y2*(z1 - z3) + y3*(-z1 + z2)) + x1*(y0*(z2 - z3) + y2*(-z0 + z3) + y3*(z0 - z2)) + x2*(y0*(-z1 + z3) + y1*(z0 - z3) + y3*(-z0 + z1)) + x3*(y0*(z1 - z2) + y1*(-z0 + z2) + y2*(z0 - z1)))/pow(L, 3);
    return det;
}

float det_jacobian_cat4_float(
    const float x0, const float y0, const float z0, //
    const float x1, const float y1, const float z1, // 
    const float x2, const float y2, const float z2, //
    const float x3, const float y3, const float z3, //
    const float L //
) {
    // Calculate determinant of Jacobian for tetrahedron category 4
    float det = (x0*(y1*(-z2 + z3) + y2*(z1 - z3) + y3*(-z1 + z2)) + x1*(y0*(z2 - z3) + y2*(-z0 + z3) + y3*(z0 - z2)) + x2*(y0*(-z1 + z3) + y1*(z0 - z3) + y3*(-z0 + z1)) + x3*(y0*(z1 - z2) + y1*(-z0 + z2) + y2*(z0 - z1)))/pow(L, 3);
    return det;
}

float det_jacobian_cat5_float(
    const float x0, const float y0, const float z0, //
    const float x1, const float y1, const float z1, // 
    const float x2, const float y2, const float z2, //
    const float x3, const float y3, const float z3, //
    const float L //
) {
    // Calculate determinant of Jacobian for tetrahedron category 5
    float det = (x0*(y1*(-z2 + z3) + y2*(z1 - z3) + y3*(-z1 + z2)) + x1*(y0*(z2 - z3) + y2*(-z0 + z3) + y3*(z0 - z2)) + x2*(y0*(-z1 + z3) + y1*(z0 - z3) + y3*(-z0 + z1)) + x3*(y0*(z1 - z2) + y1*(-z0 + z2) + y2*(z0 - z1)))/pow(L, 3);
    return det;
}
