/**
 * Jacobian determinant calculation for tetrahedron categories
 * Auto-generated code - DO NOT EDIT MANUALLY
 * Data type: real_t
 */

#include "hyteg_jacobian_real_t.h"
#include <math.h>


real_t det_jacobian_cat0_real(
    const real_t x0, const real_t y0, const real_t z0, //
    const real_t x1, const real_t y1, const real_t z1, // 
    const real_t x2, const real_t y2, const real_t z2, //
    const real_t x3, const real_t y3, const real_t z3, //
    const real_t L //
) {
    // Calculate determinant of Jacobian for tetrahedron category 0
    real_t det = (x0*(y1*(-z2 + z3) + y2*(z1 - z3) + y3*(-z1 + z2)) + x1*(y0*(z2 - z3) + y2*(-z0 + z3) + y3*(z0 - z2)) + x2*(y0*(-z1 + z3) + y1*(z0 - z3) + y3*(-z0 + z1)) + x3*(y0*(z1 - z2) + y1*(-z0 + z2) + y2*(z0 - z1)))/pow(L, 3);
    return det;
}

real_t det_jacobian_cat1_real(
    const real_t x0, const real_t y0, const real_t z0, //
    const real_t x1, const real_t y1, const real_t z1, // 
    const real_t x2, const real_t y2, const real_t z2, //
    const real_t x3, const real_t y3, const real_t z3, //
    const real_t L //
) {
    // Calculate determinant of Jacobian for tetrahedron category 1
    real_t det = (x0*(y1*(z2 - z3) + y2*(-z1 + z3) + y3*(z1 - z2)) + x1*(y0*(-z2 + z3) + y2*(z0 - z3) + y3*(-z0 + z2)) + x2*(y0*(z1 - z3) + y1*(-z0 + z3) + y3*(z0 - z1)) + x3*(y0*(-z1 + z2) + y1*(z0 - z2) + y2*(-z0 + z1)))/pow(L, 3);
    return det;
}

real_t det_jacobian_cat2_real(
    const real_t x0, const real_t y0, const real_t z0, //
    const real_t x1, const real_t y1, const real_t z1, // 
    const real_t x2, const real_t y2, const real_t z2, //
    const real_t x3, const real_t y3, const real_t z3, //
    const real_t L //
) {
    // Calculate determinant of Jacobian for tetrahedron category 2
    real_t det = (x0*(y1*(z2 - z3) + y2*(-z1 + z3) + y3*(z1 - z2)) + x1*(y0*(-z2 + z3) + y2*(z0 - z3) + y3*(-z0 + z2)) + x2*(y0*(z1 - z3) + y1*(-z0 + z3) + y3*(z0 - z1)) + x3*(y0*(-z1 + z2) + y1*(z0 - z2) + y2*(-z0 + z1)))/pow(L, 3);
    return det;
}

real_t det_jacobian_cat3_real(
    const real_t x0, const real_t y0, const real_t z0, //
    const real_t x1, const real_t y1, const real_t z1, // 
    const real_t x2, const real_t y2, const real_t z2, //
    const real_t x3, const real_t y3, const real_t z3, //
    const real_t L //
) {
    // Calculate determinant of Jacobian for tetrahedron category 3
    real_t det = (x0*(y1*(z2 - z3) + y2*(-z1 + z3) + y3*(z1 - z2)) + x1*(y0*(-z2 + z3) + y2*(z0 - z3) + y3*(-z0 + z2)) + x2*(y0*(z1 - z3) + y1*(-z0 + z3) + y3*(z0 - z1)) + x3*(y0*(-z1 + z2) + y1*(z0 - z2) + y2*(-z0 + z1)))/pow(L, 3);
    return det;
}

real_t det_jacobian_cat4_real(
    const real_t x0, const real_t y0, const real_t z0, //
    const real_t x1, const real_t y1, const real_t z1, // 
    const real_t x2, const real_t y2, const real_t z2, //
    const real_t x3, const real_t y3, const real_t z3, //
    const real_t L //
) {
    // Calculate determinant of Jacobian for tetrahedron category 4
    real_t det = (x0*(y1*(z2 - z3) + y2*(-z1 + z3) + y3*(z1 - z2)) + x1*(y0*(-z2 + z3) + y2*(z0 - z3) + y3*(-z0 + z2)) + x2*(y0*(z1 - z3) + y1*(-z0 + z3) + y3*(z0 - z1)) + x3*(y0*(-z1 + z2) + y1*(z0 - z2) + y2*(-z0 + z1)))/pow(L, 3);
    return det;
}

real_t det_jacobian_cat5_real(
    const real_t x0, const real_t y0, const real_t z0, //
    const real_t x1, const real_t y1, const real_t z1, // 
    const real_t x2, const real_t y2, const real_t z2, //
    const real_t x3, const real_t y3, const real_t z3, //
    const real_t L //
) {
    // Calculate determinant of Jacobian for tetrahedron category 5
    real_t det = (x0*(y1*(-z2 + z3) + y2*(z1 - z3) + y3*(-z1 + z2)) + x1*(y0*(z2 - z3) + y2*(-z0 + z3) + y3*(z0 - z2)) + x2*(y0*(-z1 + z3) + y1*(z0 - z3) + y3*(-z0 + z1)) + x3*(y0*(z1 - z2) + y1*(-z0 + z2) + y2*(z0 - z1)))/pow(L, 3);
    return det;
}
