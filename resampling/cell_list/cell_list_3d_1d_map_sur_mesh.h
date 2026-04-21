#ifndef __CELL_LIST_3D_1D_MAP_SUR_MESH_H__

#include <stdbool.h>
#include "sfem_base.h"

/**
 * @brief Check if the projection of a triangle onto the XY plane contains the point (x, y).
 * This function checks if the point (x, y) is inside the triangle defined by vertices v0, v1, v2 when projected onto the XY
 * plane.
 * @param v0 First vertex of the triangle (array of 3 real_t: x, y, z).
 * @param v1 Second vertex of the triangle (array of 3 real_t: x, y, z).
 * @param v2 Third vertex of the triangle (array of 3 real_t: x, y, z).
 * @param x X coordinate of the point to check.
 * @param y Y coordinate of the point to check.
 * @return true if the point (x, y) is inside the triangle's projection on the XY plane, false otherwise.
 */
bool                                       //
intersect_triangle_xy(const real_t v0[3],  //
                      const real_t v1[3],  //
                      const real_t v2[3],  //
                      const real_t x,      //
                      const real_t y);     //

#endif  // __CELL_LIST_3D_1D_MAP_SUR_MESH_H__