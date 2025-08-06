/**
 * Jacobian determinant calculation for tetrahedron categories
 * Auto-generated code - DO NOT EDIT MANUALLY
 * Data type: double
 */

#include "hyteg_jacobian_double.h"
#include <math.h>


double det_jacobian_cat0_dbl(
    const double x0, const double y0, const double z0, //
    const double x1, const double y1, const double z1, // 
    const double x2, const double y2, const double z2, //
    const double x3, const double y3, const double z3, //
    const double L //
) {
    // Calculate determinant of Jacobian for tetrahedron category 0
    double det = (x0*(y1*(-z2 + z3) + y2*(z1 - z3) + y3*(-z1 + z2)) + x1*(y0*(z2 - z3) + y2*(-z0 + z3) + y3*(z0 - z2)) + x2*(y0*(-z1 + z3) + y1*(z0 - z3) + y3*(-z0 + z1)) + x3*(y0*(z1 - z2) + y1*(-z0 + z2) + y2*(z0 - z1)))/pow(L, 3);
    return det;
}

double det_jacobian_cat1_dbl(
    const double x0, const double y0, const double z0, //
    const double x1, const double y1, const double z1, // 
    const double x2, const double y2, const double z2, //
    const double x3, const double y3, const double z3, //
    const double L //
) {
    // Calculate determinant of Jacobian for tetrahedron category 1
    double det = (x0*(y1*(z2 - z3) + y2*(-z1 + z3) + y3*(z1 - z2)) + x1*(y0*(-z2 + z3) + y2*(z0 - z3) + y3*(-z0 + z2)) + x2*(y0*(z1 - z3) + y1*(-z0 + z3) + y3*(z0 - z1)) + x3*(y0*(-z1 + z2) + y1*(z0 - z2) + y2*(-z0 + z1)))/pow(L, 3);
    return det;
}

double det_jacobian_cat2_dbl(
    const double x0, const double y0, const double z0, //
    const double x1, const double y1, const double z1, // 
    const double x2, const double y2, const double z2, //
    const double x3, const double y3, const double z3, //
    const double L //
) {
    // Calculate determinant of Jacobian for tetrahedron category 2
    double det = (x0*(y1*(z2 - z3) + y2*(-z1 + z3) + y3*(z1 - z2)) + x1*(y0*(-z2 + z3) + y2*(z0 - z3) + y3*(-z0 + z2)) + x2*(y0*(z1 - z3) + y1*(-z0 + z3) + y3*(z0 - z1)) + x3*(y0*(-z1 + z2) + y1*(z0 - z2) + y2*(-z0 + z1)))/pow(L, 3);
    return det;
}

double det_jacobian_cat3_dbl(
    const double x0, const double y0, const double z0, //
    const double x1, const double y1, const double z1, // 
    const double x2, const double y2, const double z2, //
    const double x3, const double y3, const double z3, //
    const double L //
) {
    // Calculate determinant of Jacobian for tetrahedron category 3
    double det = (x0*(y1*(z2 - z3) + y2*(-z1 + z3) + y3*(z1 - z2)) + x1*(y0*(-z2 + z3) + y2*(z0 - z3) + y3*(-z0 + z2)) + x2*(y0*(z1 - z3) + y1*(-z0 + z3) + y3*(z0 - z1)) + x3*(y0*(-z1 + z2) + y1*(z0 - z2) + y2*(-z0 + z1)))/pow(L, 3);
    return det;
}

double det_jacobian_cat4_dbl(
    const double x0, const double y0, const double z0, //
    const double x1, const double y1, const double z1, // 
    const double x2, const double y2, const double z2, //
    const double x3, const double y3, const double z3, //
    const double L //
) {
    // Calculate determinant of Jacobian for tetrahedron category 4
    double det = (x0*(y1*(z2 - z3) + y2*(-z1 + z3) + y3*(z1 - z2)) + x1*(y0*(-z2 + z3) + y2*(z0 - z3) + y3*(-z0 + z2)) + x2*(y0*(z1 - z3) + y1*(-z0 + z3) + y3*(z0 - z1)) + x3*(y0*(-z1 + z2) + y1*(z0 - z2) + y2*(-z0 + z1)))/pow(L, 3);
    return det;
}

double det_jacobian_cat5_dbl(
    const double x0, const double y0, const double z0, //
    const double x1, const double y1, const double z1, // 
    const double x2, const double y2, const double z2, //
    const double x3, const double y3, const double z3, //
    const double L //
) {
    // Calculate determinant of Jacobian for tetrahedron category 5
    double det = (x0*(y1*(-z2 + z3) + y2*(z1 - z3) + y3*(-z1 + z2)) + x1*(y0*(z2 - z3) + y2*(-z0 + z3) + y3*(z0 - z2)) + x2*(y0*(-z1 + z3) + y1*(z0 - z3) + y3*(-z0 + z1)) + x3*(y0*(z1 - z2) + y1*(-z0 + z2) + y2*(z0 - z1)))/pow(L, 3);
    return det;
}
