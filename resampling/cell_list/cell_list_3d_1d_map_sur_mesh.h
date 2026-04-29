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
int                                                                                                  //
query_cell_list_3d_1d_map_mesh_given_xy_tri3_v(const cell_list_3d_1d_map_t *map,                     //
                                               const boxes_t               *boxes,                   //
                                               const mesh_tri3_geom_t      *mesh_geom,               //
                                               const real_t                 x,                       //
                                               const real_t                 y,                       //
                                               const int                    start_index_tri3_array,  //
                                               int                         *size_t3_array,           //
                                               real_t                     **tri3_intersect_z);       //

/**
 * @brief Build a split 3D->1D cell-list map for triangle meshes, partitioning triangles based on their centroids relative to a
 * split line. This function computes the centroids of the triangles, determines a split line (e.g., median x or y), partitions
 * the triangles into "lower" and "upper" groups, and builds separate cell-list maps for each group.
 * @param mesh Pointer to the input mesh containing the triangles.
 * @param boxes Pointer to the bounding boxes associated with triangles.
 * @param split_map Output pointer to the allocated split map. Caller owns the memory and must free it using
 * free_cell_list_split_3d_1d_map_mesh.
 * @return EXIT_SUCCESS on success, or EXIT_FAILURE on error (e.g., memory allocation failure).
 */
int                                                                            //
build_cell_list_split_3d_1d_map_mesh(cell_list_split_3d_1d_map_t **split_map,  //
                                     const mesh_t                 *mesh,       //
                                     const boxes_t                *boxes);     //

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
int                                                                                                         //
query_cell_list_3d_1d_split_map_mesh_given_xy_tri3_v(const cell_list_split_3d_1d_map_t *map,                //
                                                     const boxes_t                     *boxes,              //
                                                     const mesh_tri3_geom_t            *mesh_geom,          //
                                                     const real_t                       x,                  //
                                                     const real_t                       y,                  //
                                                     int                               *size_t3_array,      //
                                                     real_t                           **tri3_intersect_z);  //

/**
 * @brief Rasterize a triangle mesh using the 3D->1D cell-list map to efficiently find candidate triangles for each (x, y) and
 * compute the corresponding z values. This function queries the split map for each (x, y), checks candidate triangles for
 * intersection, and fills out_z with the z coordinates of the intersections.
 * @param map Pointer to the split cell-list map.
 * @param boxes Pointer to the bounding boxes associated with triangles.
 * @param mesh_geom Pointer to triangle geometric data used for point-in-triangle tests.
 * @param x Query x coordinate.
 * @param y Query y coordinate.
 * @param z_coords Array of z coordinates to check for intersection (used for filtering candidates).
 * @param num_z_coords Number of entries in z_coords.
 * @param out_z Output array to store the z coordinates of the intersections. Must be pre-allocated with size at least
 * num_z_coords.
 * @return EXIT_SUCCESS on success, or EXIT_FAILURE on error (e.g., memory allocation failure).
 */
int                                                                                                      //
raster_cell_list_3d_1d_split_map_mesh_given_xyz_tri3_v(const cell_list_split_3d_1d_map_t *map,           //
                                                       const boxes_t                     *boxes,         //
                                                       const mesh_tri3_geom_t            *mesh_geom,     //
                                                       const real_t                       x,             //
                                                       const real_t                       y,             //
                                                       const real_t                      *z_coords,      //
                                                       const int                          num_z_coords,  //
                                                       real_t                            *out_z);        //

#endif  // __CELL_LIST_3D_1D_MAP_SUR_MESH_H__