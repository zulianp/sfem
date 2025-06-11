#ifndef HYTEG_COORDINATES_H
#define HYTEG_COORDINATES_H

/**
 * Auto-generated coordinate arrays for tetrahedral meshes
 */

#include <stddef.h>
#include <stdlib.h>

#include "sfem_base.h"
#include "sfem_defs.h"

// Extern declarations for coordinate arrays
// Coordinate arrays for L = 1
extern real_t hyteg_L1_x[];    // X coordinates
extern real_t hyteg_L1_y[];    // Y coordinates
extern real_t hyteg_L1_z[];    // Z coordinates
extern real_t hyteg_L1_xyz[];  // Combined XYZ coordinates

// Coordinate arrays for L = 2
extern real_t hyteg_L2_x[];    // X coordinates
extern real_t hyteg_L2_y[];    // Y coordinates
extern real_t hyteg_L2_z[];    // Z coordinates
extern real_t hyteg_L2_xyz[];  // Combined XYZ coordinates

// Coordinate arrays for L = 3
extern real_t hyteg_L3_x[];    // X coordinates
extern real_t hyteg_L3_y[];    // Y coordinates
extern real_t hyteg_L3_z[];    // Z coordinates
extern real_t hyteg_L3_xyz[];  // Combined XYZ coordinates

// Coordinate arrays for L = 4
extern real_t hyteg_L4_x[];    // X coordinates
extern real_t hyteg_L4_y[];    // Y coordinates
extern real_t hyteg_L4_z[];    // Z coordinates
extern real_t hyteg_L4_xyz[];  // Combined XYZ coordinates

// Coordinate arrays for L = 5
extern real_t hyteg_L5_x[];    // X coordinates
extern real_t hyteg_L5_y[];    // Y coordinates
extern real_t hyteg_L5_z[];    // Z coordinates
extern real_t hyteg_L5_xyz[];  // Combined XYZ coordinates

// Coordinate arrays for L = 6
extern real_t hyteg_L6_x[];    // X coordinates
extern real_t hyteg_L6_y[];    // Y coordinates
extern real_t hyteg_L6_z[];    // Z coordinates
extern real_t hyteg_L6_xyz[];  // Combined XYZ coordinates

// Coordinate arrays for L = 7
extern real_t hyteg_L7_x[];    // X coordinates
extern real_t hyteg_L7_y[];    // Y coordinates
extern real_t hyteg_L7_z[];    // Z coordinates
extern real_t hyteg_L7_xyz[];  // Combined XYZ coordinates

// Coordinate arrays for L = 8
extern real_t hyteg_L8_x[];    // X coordinates
extern real_t hyteg_L8_y[];    // Y coordinates
extern real_t hyteg_L8_z[];    // Z coordinates
extern real_t hyteg_L8_xyz[];  // Combined XYZ coordinates

// Coordinate arrays for L = 9
extern real_t hyteg_L9_x[];    // X coordinates
extern real_t hyteg_L9_y[];    // Y coordinates
extern real_t hyteg_L9_z[];    // Z coordinates
extern real_t hyteg_L9_xyz[];  // Combined XYZ coordinates

// Coordinate arrays for L = 10
extern real_t hyteg_L10_x[];    // X coordinates
extern real_t hyteg_L10_y[];    // Y coordinates
extern real_t hyteg_L10_z[];    // Z coordinates
extern real_t hyteg_L10_xyz[];  // Combined XYZ coordinates

// Coordinate arrays for L = 11
extern real_t hyteg_L11_x[];    // X coordinates
extern real_t hyteg_L11_y[];    // Y coordinates
extern real_t hyteg_L11_z[];    // Z coordinates
extern real_t hyteg_L11_xyz[];  // Combined XYZ coordinates

// Coordinate arrays for L = 12
extern real_t hyteg_L12_x[];    // X coordinates
extern real_t hyteg_L12_y[];    // Y coordinates
extern real_t hyteg_L12_z[];    // Z coordinates
extern real_t hyteg_L12_xyz[];  // Combined XYZ coordinates

// Function declarations
/**
 * Returns a pointer to the xyz coordinate array for the specified level.
 *
 * @param L Level parameter (valid values: 1-10)
 * @return Pointer to the corresponding xyz array, or NULL if L is invalid
 */
real_t* get_hyteg_xyz(int L);

/**
 * Returns a pointer to the x coordinates array for the specified level.
 *
 * @param L Level parameter (valid values: 1-10)
 * @return Pointer to the corresponding x array, or NULL if L is invalid
 */
real_t* get_hyteg_x(int L);

/**
 * Returns a pointer to the y coordinates array for the specified level.
 *
 * @param L Level parameter (valid values: 1-10)
 * @return Pointer to the corresponding y array, or NULL if L is invalid
 */
real_t* get_hyteg_y(int L);

/**
 * Returns a pointer to the z coordinates array for the specified level.
 *
 * @param L Level parameter (valid values: 1-10)
 * @return Pointer to the corresponding z array, or NULL if L is invalid
 */
real_t* get_hyteg_z(int L);

#endif  // HYTEG_COORDINATES_H
