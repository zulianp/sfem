/**
 * Jacobian determinant calculation for tetrahedron categories
 * Auto-generated code - DO NOT EDIT MANUALLY
 * Data type: real_t
 */

#include "hyteg_jacobian_real_t.h"
#include <math.h>

real_t det_jacobian_cat0_real(const real_t x0, const real_t y0, const real_t z0, const real_t x1, const real_t y1,
                              const real_t z1, const real_t x2, const real_t y2, const real_t z2, const real_t x3,
                              const real_t y3, const real_t z3, const real_t L) {
    // Calculate edge vectors
    const real_t e1x = x1 - x0;
    const real_t e1y = y1 - y0;
    const real_t e1z = z1 - z0;

    const real_t e2x = x2 - x0;
    const real_t e2y = y2 - y0;
    const real_t e2z = z2 - z0;

    const real_t e3x = x3 - x0;
    const real_t e3y = y3 - y0;
    const real_t e3z = z3 - z0;

    // Calculate cross product (e1 × e2)
    const real_t cx = e1y * e2z - e1z * e2y;
    const real_t cy = e1z * e2x - e1x * e2z;
    const real_t cz = e1x * e2y - e1y * e2x;

    // Calculate determinant as dot product of cross with e3
    const real_t det_unnormalized = cx * e3x + cy * e3y + cz * e3z;

    // Scale by L³ (more stable than pow(L, 3))
    const real_t L3  = L * L * L;
    const real_t det = det_unnormalized / L3;

    return det;
}

real_t det_jacobian_cat1_real(const real_t x0, const real_t y0, const real_t z0, const real_t x1, const real_t y1,
                              const real_t z1, const real_t x2, const real_t y2, const real_t z2, const real_t x3,
                              const real_t y3, const real_t z3, const real_t L) {
    // Calculate edge vectors
    const real_t e1x = x1 - x0;
    const real_t e1y = y1 - y0;
    const real_t e1z = z1 - z0;

    const real_t e2x = x2 - x0;
    const real_t e2y = y2 - y0;
    const real_t e2z = z2 - z0;

    const real_t e3x = x3 - x0;
    const real_t e3y = y3 - y0;
    const real_t e3z = z3 - z0;

    // Calculate cross product (e1 × e2)
    const real_t cx = e1y * e2z - e1z * e2y;
    const real_t cy = e1z * e2x - e1x * e2z;
    const real_t cz = e1x * e2y - e1y * e2x;

    // Calculate determinant as dot product of cross with e3
    const real_t det_unnormalized = cx * e3x + cy * e3y + cz * e3z;

    // Scale by L³ (more stable than pow(L, 3))
    const real_t L3  = L * L * L;
    const real_t det = det_unnormalized / L3;

    return det;
}
real_t det_jacobian_cat2_real(const real_t x0, const real_t y0, const real_t z0, const real_t x1, const real_t y1,
                              const real_t z1, const real_t x2, const real_t y2, const real_t z2, const real_t x3,
                              const real_t y3, const real_t z3, const real_t L) {
    // Calculate edge vectors
    const real_t e1x = x1 - x0;
    const real_t e1y = y1 - y0;
    const real_t e1z = z1 - z0;

    const real_t e2x = x2 - x0;
    const real_t e2y = y2 - y0;
    const real_t e2z = z2 - z0;

    const real_t e3x = x3 - x0;
    const real_t e3y = y3 - y0;
    const real_t e3z = z3 - z0;

    // Calculate cross product (e1 × e2)
    const real_t cx = e1y * e2z - e1z * e2y;
    const real_t cy = e1z * e2x - e1x * e2z;
    const real_t cz = e1x * e2y - e1y * e2x;

    // Calculate determinant as dot product of cross with e3
    const real_t det_unnormalized = cx * e3x + cy * e3y + cz * e3z;

    // Scale by L³ (more stable than pow(L, 3))
    const real_t L3  = L * L * L;
    const real_t det = det_unnormalized / L3;

    return det;
}

real_t det_jacobian_cat3_real(const real_t x0, const real_t y0, const real_t z0, const real_t x1, const real_t y1,
                              const real_t z1, const real_t x2, const real_t y2, const real_t z2, const real_t x3,
                              const real_t y3, const real_t z3, const real_t L) {
    // Calculate edge vectors
    const real_t e1x = x1 - x0;
    const real_t e1y = y1 - y0;
    const real_t e1z = z1 - z0;

    const real_t e2x = x2 - x0;
    const real_t e2y = y2 - y0;
    const real_t e2z = z2 - z0;

    const real_t e3x = x3 - x0;
    const real_t e3y = y3 - y0;
    const real_t e3z = z3 - z0;

    // Calculate cross product (e1 × e2)
    const real_t cx = e1y * e2z - e1z * e2y;
    const real_t cy = e1z * e2x - e1x * e2z;
    const real_t cz = e1x * e2y - e1y * e2x;

    // Calculate determinant as dot product of cross with e3
    const real_t det_unnormalized = cx * e3x + cy * e3y + cz * e3z;

    // Scale by L³ (more stable than pow(L, 3))
    const real_t L3  = L * L * L;
    const real_t det = det_unnormalized / L3;

    return det;
}

real_t det_jacobian_cat4_real(const real_t x0, const real_t y0, const real_t z0, const real_t x1, const real_t y1,
                              const real_t z1, const real_t x2, const real_t y2, const real_t z2, const real_t x3,
                              const real_t y3, const real_t z3, const real_t L) {
    // Calculate edge vectors
    const real_t e1x = x1 - x0;
    const real_t e1y = y1 - y0;
    const real_t e1z = z1 - z0;

    const real_t e2x = x2 - x0;
    const real_t e2y = y2 - y0;
    const real_t e2z = z2 - z0;

    const real_t e3x = x3 - x0;
    const real_t e3y = y3 - y0;
    const real_t e3z = z3 - z0;

    // Calculate cross product (e1 × e2)
    const real_t cx = e1y * e2z - e1z * e2y;
    const real_t cy = e1z * e2x - e1x * e2z;
    const real_t cz = e1x * e2y - e1y * e2x;

    // Calculate determinant as dot product of cross with e3
    const real_t det_unnormalized = cx * e3x + cy * e3y + cz * e3z;

    // Scale by L³ (more stable than pow(L, 3))
    const real_t L3  = L * L * L;
    const real_t det = det_unnormalized / L3;

    return det;
}

real_t det_jacobian_cat5_real(const real_t x0, const real_t y0, const real_t z0, const real_t x1, const real_t y1,
                              const real_t z1, const real_t x2, const real_t y2, const real_t z2, const real_t x3,
                              const real_t y3, const real_t z3, const real_t L) {
    // Calculate edge vectors
    const real_t e1x = x1 - x0;
    const real_t e1y = y1 - y0;
    const real_t e1z = z1 - z0;

    const real_t e2x = x2 - x0;
    const real_t e2y = y2 - y0;
    const real_t e2z = z2 - z0;

    const real_t e3x = x3 - x0;
    const real_t e3y = y3 - y0;
    const real_t e3z = z3 - z0;

    // Calculate cross product (e1 × e2)
    const real_t cx = e1y * e2z - e1z * e2y;
    const real_t cy = e1z * e2x - e1x * e2z;
    const real_t cz = e1x * e2y - e1y * e2x;

    // Calculate determinant as dot product of cross with e3
    const real_t det_unnormalized = cx * e3x + cy * e3y + cz * e3z;

    // Scale by L³ (more stable than pow(L, 3))
    const real_t L3  = L * L * L;
    const real_t det = det_unnormalized / L3;

    return det;
}
