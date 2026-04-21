#ifndef __CELL_LIST_3D_1D_MAP_SUR_MESH_H__

#include <stdbool.h>
#include "cell_arg_sort.h"
#include "cell_list_3d_1d_map.h"
#include "cell_list_3d_map.h"
#include "sfem_base.h"
#include "sfem_mesh.h"

typedef struct {
    real_t split_x;
    real_t split_y;

    cell_list_3d_1d_map_t *map_lower;
    cell_list_3d_1d_map_t *map_upper;

} cell_list_split_3d_1d_map_t;

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

/**
 * @brief Compute the z coordinate of the intersection point between a vertical ray from (x, y) and the triangle defined by v0,
 * v1, v2. This function assumes that (x, y) is inside the projection of the triangle onto the XY plane. It computes the z
 * coordinate of the point on the triangle that corresponds to (x, y).
 * @param v0 First vertex of the triangle (array of 3 real_t: x, y, z).
 * @param v1 Second vertex of the triangle (array of 3 real_t: x, y, z).
 * @param v2 Third vertex of the triangle (array of 3 real_t: x, y, z).
 * @param x X coordinate of the point to check.
 * @param y Y coordinate of the point to check.
 * @param out_z Pointer to store the computed z coordinate of the intersection point.
 * @return void. The computed z coordinate is stored in *out_z.
 */
void                                                 //
intersection_point_triangle_xy(const real_t v0[3],   //
                               const real_t v1[3],   //
                               const real_t v2[3],   //
                               const real_t x,       //
                               const real_t y,       //
                               real_t      *out_z);  //

/**
 * @brief Query a non-split 3D->1D cell-list map at fixed (x, y) over candidate z values for triangle meshes.
 * The function searches the cell corresponding to (x, y), filters candidates in z, and performs box and triangle inclusion
 * checks. It returns as soon as one containing triangle is found.
 * @param map Pointer to the non-split cell-list map.
 * @param boxes Pointer to the bounding boxes associated with triangles.
 * @param mesh_geom Pointer to triangle geometric data used for point-in-triangle tests.
 * @param x Query x coordinate.
 * @param y Query y coordinate.
 * @param size_z Number of entries in z_array.
 * @return Triangle index (>= 0) if found, -1 if no containing triangle is found.
 */
int                                                                                             //
query_cell_list_3d_1d_map_mesh_given_xy_tri3_v(const cell_list_3d_1d_map_t *map,                //
                                               const boxes_t               *boxes,              //
                                               const mesh_tri3_geom_t      *mesh_geom,          //
                                               const real_t                 x,                  //
                                               const real_t                 y,                  //
                                               int                         *size_t3_array,      //
                                               real_t                     **tri3_intersect_z);  //

/**
 * @brief Compute the z coordinate of the intersection point between a vertical ray from (x, y) and the triangle defined by v0,
 * v1, v2. This function assumes that (x, y) is inside the projection of the triangle onto the XY plane. It computes the z
 * coordinate of the point on the triangle that corresponds to (x, y).
 * @param v0 First vertex of the triangle (array of 3 real_t: x, y, z).
 * @param v1 Second vertex of the triangle (array of 3 real_t: x, y, z).
 * @param v2 Third vertex of the triangle (array of 3 real_t: x, y, z).
 * @param x X coordinate of the point to check.
 * @param y Y coordinate of the point to check.
 * @param out_z Pointer to store the computed z coordinate of the intersection point.
 * @return void. The computed z coordinate is stored in *out_z.
 */
int                                                                                                          //
query_cell_list_3d_1d_split_map_mesh_given_xy_tri3_v(const cell_list_split_3d_1d_map_t *map,                 //
                                                     const boxes_t                     *boxes,               //
                                                     const mesh_tri3_geom_t            *mesh_geom,           //
                                                     const real_t                       x,                   //
                                                     const real_t                       y,                   //
                                                     int                               *size_t3_array,       //
                                                     real_t                           **tri3_intersect_z);   //

#endif  // __CELL_LIST_3D_1D_MAP_SUR_MESH_H__