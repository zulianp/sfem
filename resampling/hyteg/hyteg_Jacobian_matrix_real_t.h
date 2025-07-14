/**
 * Jacobian matrix calculation for tetrahedron categories
 * Auto-generated code - DO NOT EDIT MANUALLY
 * Data type: real_t
 */

#ifndef HYTEG_JACOBIAN_MATRIX_REAL_T_H
#define HYTEG_JACOBIAN_MATRIX_REAL_T_H

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
void jacobian_matrix_cat0_real_t(const real_t x0, const real_t y0, const real_t z0,  //
                                 const real_t x1, const real_t y1, const real_t z1,  //
                                 const real_t x2, const real_t y2, const real_t z2,  //
                                 const real_t x3, const real_t y3, const real_t z3,  //
                                 const real_t L,                                     //
                                 real_t       J[9]                                   //
);

/**
 * Calculate 3x3 Jacobian matrix for tetrahedron category 1
 */
void jacobian_matrix_cat1_real_t(const real_t x0, const real_t y0, const real_t z0,  //
                                 const real_t x1, const real_t y1, const real_t z1,  //
                                 const real_t x2, const real_t y2, const real_t z2,  //
                                 const real_t x3, const real_t y3, const real_t z3,  //
                                 const real_t L,                                     //
                                 real_t       J[9]                                   //
);

/**
 * Calculate 3x3 Jacobian matrix for tetrahedron category 2
 */
void jacobian_matrix_cat2_real_t(const real_t x0, const real_t y0, const real_t z0,  //
                                 const real_t x1, const real_t y1, const real_t z1,  //
                                 const real_t x2, const real_t y2, const real_t z2,  //
                                 const real_t x3, const real_t y3, const real_t z3,  //
                                 const real_t L,                                     //
                                 real_t       J[9]                                   //
);

/**
 * Calculate 3x3 Jacobian matrix for tetrahedron category 3
 */
void jacobian_matrix_cat3_real_t(const real_t x0, const real_t y0, const real_t z0,  //
                                 const real_t x1, const real_t y1, const real_t z1,  //
                                 const real_t x2, const real_t y2, const real_t z2,  //
                                 const real_t x3, const real_t y3, const real_t z3,  //
                                 const real_t L,                                     //
                                 real_t       J[9]                                   //
);

/**
 * Calculate 3x3 Jacobian matrix for tetrahedron category 4
 */
void jacobian_matrix_cat4_real_t(const real_t x0, const real_t y0, const real_t z0,  //
                                 const real_t x1, const real_t y1, const real_t z1,  //
                                 const real_t x2, const real_t y2, const real_t z2,  //
                                 const real_t x3, const real_t y3, const real_t z3,  //
                                 const real_t L,                                     //
                                 real_t       J[9]                                   //
);

/**
 * Calculate 3x3 Jacobian matrix for tetrahedron category 5
 */
void jacobian_matrix_cat5_real_t(const real_t x0, const real_t y0, const real_t z0,  //
                                 const real_t x1, const real_t y1, const real_t z1,  //
                                 const real_t x2, const real_t y2, const real_t z2,  //
                                 const real_t x3, const real_t y3, const real_t z3,  //
                                 const real_t L,                                     //
                                 real_t       J[9]                                   //
);

/**
 * @brief Calculate 3x3 Jacobian matrix for tetrahedron based on category
 * 
 * @param category 
 * @param x0 
 * @param y0 
 * @param z0 
 * @param x1 
 * @param y1 
 * @param z1 
 * @param x2 
 * @param y2 
 * @param z2 
 * @param x3 
 * @param y3 
 * @param z3 
 * @param L 
 * @param J 
 */
void jacobian_matrix_real_t(const int    category,                                               //
                            const real_t x0, const real_t y0, const real_t z0, const real_t x1,  //
                            const real_t y1, const real_t z1, const real_t x2,                   //
                            const real_t y2, const real_t z2, const real_t x3,                   //
                            const real_t y3, const real_t z3, const real_t L,                    //
                            real_t J[9]);                                                        //

#ifdef __cplusplus
}
#endif

#endif /* HYTEG_JACOBIAN_MATRIX_REAL_T_H */
