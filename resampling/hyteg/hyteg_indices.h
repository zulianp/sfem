#ifndef HYTEG_INDICES_H
#define HYTEG_INDICES_H

/**
 * Auto-generated tetrahedral indices for meshes
 */
 
#include <stddef.h>
#include <stdlib.h>

#include "sfem_base.h"
#include "sfem_defs.h"

// Tetrahedral indices for L = 1
extern int hyteg_L1_indices[];
extern int hyteg_L1_categories[];
extern int hyteg_L1_v0[];
extern int hyteg_L1_v1[];
extern int hyteg_L1_v2[];
extern int hyteg_L1_v3[];

// Tetrahedral indices for L = 2
extern int hyteg_L2_indices[];
extern int hyteg_L2_categories[];
extern int hyteg_L2_v0[];
extern int hyteg_L2_v1[];
extern int hyteg_L2_v2[];
extern int hyteg_L2_v3[];

// Tetrahedral indices for L = 3
extern int hyteg_L3_indices[];
extern int hyteg_L3_categories[];
extern int hyteg_L3_v0[];
extern int hyteg_L3_v1[];
extern int hyteg_L3_v2[];
extern int hyteg_L3_v3[];

// Tetrahedral indices for L = 4
extern int hyteg_L4_indices[];
extern int hyteg_L4_categories[];
extern int hyteg_L4_v0[];
extern int hyteg_L4_v1[];
extern int hyteg_L4_v2[];
extern int hyteg_L4_v3[];

// Tetrahedral indices for L = 5
extern int hyteg_L5_indices[];
extern int hyteg_L5_categories[];
extern int hyteg_L5_v0[];
extern int hyteg_L5_v1[];
extern int hyteg_L5_v2[];
extern int hyteg_L5_v3[];

// Tetrahedral indices for L = 6
extern int hyteg_L6_indices[];
extern int hyteg_L6_categories[];
extern int hyteg_L6_v0[];
extern int hyteg_L6_v1[];
extern int hyteg_L6_v2[];
extern int hyteg_L6_v3[];

// Tetrahedral indices for L = 7
extern int hyteg_L7_indices[];
extern int hyteg_L7_categories[];
extern int hyteg_L7_v0[];
extern int hyteg_L7_v1[];
extern int hyteg_L7_v2[];
extern int hyteg_L7_v3[];

// Tetrahedral indices for L = 8
extern int hyteg_L8_indices[];
extern int hyteg_L8_categories[];
extern int hyteg_L8_v0[];
extern int hyteg_L8_v1[];
extern int hyteg_L8_v2[];
extern int hyteg_L8_v3[];

// Tetrahedral indices for L = 9
extern int hyteg_L9_indices[];
extern int hyteg_L9_categories[];
extern int hyteg_L9_v0[];
extern int hyteg_L9_v1[];
extern int hyteg_L9_v2[];
extern int hyteg_L9_v3[];

// Tetrahedral indices for L = 10
extern int hyteg_L10_indices[];
extern int hyteg_L10_categories[];
extern int hyteg_L10_v0[];
extern int hyteg_L10_v1[];
extern int hyteg_L10_v2[];
extern int hyteg_L10_v3[];

// Tetrahedral indices for L = 11
extern int hyteg_L11_indices[];
extern int hyteg_L11_categories[];
extern int hyteg_L11_v0[];
extern int hyteg_L11_v1[];
extern int hyteg_L11_v2[];
extern int hyteg_L11_v3[];

// Tetrahedral indices for L = 12
extern int hyteg_L12_indices[];
extern int hyteg_L12_categories[];
extern int hyteg_L12_v0[];
extern int hyteg_L12_v1[];
extern int hyteg_L12_v2[];
extern int hyteg_L12_v3[];

// Function declarations
/**
 * Returns a pointer to the tetrahedral indices array for the specified level.
 * Each tetrahedron has 4 consecutive indices in this array.
 *
 * @param L Level parameter
 * @return Pointer to the corresponding indices array, or NULL if L is invalid
 */
int* get_hyteg_indices(int L);

/**
 * Returns a pointer to the tetrahedral categories array for the specified level.
 * Each tetrahedron has one category value (0-5).
 *
 * @param L Level parameter
 * @return Pointer to the corresponding categories array, or NULL if L is invalid
 */
int* get_hyteg_categories(int L);

/**
 * Returns a pointer to the first vertex indices (v0) of all tetrahedra.
 *
 * @param L Level parameter
 * @return Pointer to the v0 indices array, or NULL if L is invalid
 */
int* get_hyteg_v0(int L);

/**
 * Returns a pointer to the second vertex indices (v1) of all tetrahedra.
 *
 * @param L Level parameter
 * @return Pointer to the v1 indices array, or NULL if L is invalid
 */
int* get_hyteg_v1(int L);

/**
 * Returns a pointer to the third vertex indices (v2) of all tetrahedra.
 *
 * @param L Level parameter
 * @return Pointer to the v2 indices array, or NULL if L is invalid
 */
int* get_hyteg_v2(int L);

/**
 * Returns a pointer to the fourth vertex indices (v3) of all tetrahedra.
 *
 * @param L Level parameter
 * @return Pointer to the v3 indices array, or NULL if L is invalid
 */
int* get_hyteg_v3(int L);

/**
 * Returns the number of tetrahedra for the specified level.
 *
 * @param L Level parameter
 * @return Number of tetrahedra, or 0 if L is invalid
 */
int get_hyteg_num_tetrahedra(int L);

#endif // HYTEG_INDICES_H
