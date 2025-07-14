/**
 * Jacobian matrix calculation for tetrahedron categories
 * Auto-generated code - DO NOT EDIT MANUALLY
 * Data type: double
 */

#ifndef HYTEG_JACOBIAN_MATRIX_DOUBLE_H
#define HYTEG_JACOBIAN_MATRIX_DOUBLE_H

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
void jacobian_matrix_cat0_double(const double x0, const double y0, const double z0,  //
                                 const double x1, const double y1, const double z1,  //
                                 const double x2, const double y2, const double z2,  //
                                 const double x3, const double y3, const double z3,  //
                                 const double L,                                     //
                                 double       J[9]                                   //
);

/**
 * Calculate 3x3 Jacobian matrix for tetrahedron category 1
 */
void jacobian_matrix_cat1_double(const double x0, const double y0, const double z0,  //
                                 const double x1, const double y1, const double z1,  //
                                 const double x2, const double y2, const double z2,  //
                                 const double x3, const double y3, const double z3,  //
                                 const double L,                                     //
                                 double       J[9]                                   //
);

/**
 * Calculate 3x3 Jacobian matrix for tetrahedron category 2
 */
void jacobian_matrix_cat2_double(const double x0, const double y0, const double z0,  //
                                 const double x1, const double y1, const double z1,  //
                                 const double x2, const double y2, const double z2,  //
                                 const double x3, const double y3, const double z3,  //
                                 const double L,                                     //
                                 double       J[9]                                   //
);

/**
 * Calculate 3x3 Jacobian matrix for tetrahedron category 3
 */
void jacobian_matrix_cat3_double(const double x0, const double y0, const double z0,  //
                                 const double x1, const double y1, const double z1,  //
                                 const double x2, const double y2, const double z2,  //
                                 const double x3, const double y3, const double z3,  //
                                 const double L,                                     //
                                 double       J[9]                                   //
);

/**
 * Calculate 3x3 Jacobian matrix for tetrahedron category 4
 */
void jacobian_matrix_cat4_double(const double x0, const double y0, const double z0,  //
                                 const double x1, const double y1, const double z1,  //
                                 const double x2, const double y2, const double z2,  //
                                 const double x3, const double y3, const double z3,  //
                                 const double L,                                     //
                                 double       J[9]                                   //
);

/**
 * Calculate 3x3 Jacobian matrix for tetrahedron category 5
 */
void jacobian_matrix_cat5_double(const double x0, const double y0, const double z0,  //
                                 const double x1, const double y1, const double z1,  //
                                 const double x2, const double y2, const double z2,  //
                                 const double x3, const double y3, const double z3,  //
                                 const double L,                                     //
                                 double       J[9]                                   //
);

#ifdef __cplusplus
}
#endif

#endif /* HYTEG_JACOBIAN_MATRIX_DOUBLE_H */
