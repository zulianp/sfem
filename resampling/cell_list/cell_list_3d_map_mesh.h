#ifndef __CELL_LIST_3D_MAP_MESH_H__
#define __CELL_LIST_3D_MAP_MESH_H__

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "cell_arg_sort.h"
#include "cell_list_3d_map.h"
#include "precision_types.h"
#include "sfem_mesh.h"

/**
 * @brief Query a non-split 3D->2D cell-list map at fixed (x, y) over candidate z values.
 *
 * The function searches the cell corresponding to (x, y), filters candidates in z,
 * and performs box and tetrahedron inclusion checks. It returns as soon as one
 * containing tetrahedron is found.
 *
 * @param map Pointer to the non-split cell-list map.
 * @param boxes Pointer to the bounding boxes associated with tetrahedra.
 * @param mesh_geom Pointer to tetrahedron geometric data used for point-in-tet tests.
 * @param x Query x coordinate.
 * @param y Query y coordinate.
 * @param z_array Array of z candidates to test at the same (x, y).
 * @param size_z Number of entries in z_array.
 * @return Tetrahedron index (>= 0) if found, -1 if no containing tetrahedron is found.
 */
int                                                                              //
query_cell_list_3d_2d_map_mesh_given_xy(const cell_list_3d_2d_map_t *map,        //
                                        const boxes_t               *boxes,      //
                                        const mesh_tet_geom_t       *mesh_geom,  //
                                        const real_t                 x,          //
                                        const real_t                 y,          //
                                        const real_t                *z_array,    //
                                        const int                    size_z);    //

/**
 * @brief Query a non-split 3D->2D cell-list map at fixed (x, y) for each z in z_array.
 *
 * For each z_array[i], the function stores the containing tetrahedron index in
 * tets_array[i] when found.
 *
 * @param map Pointer to the non-split cell-list map.
 * @param boxes Pointer to the bounding boxes associated with tetrahedra.
 * @param mesh_geom Pointer to tetrahedron geometric data used for point-in-tet tests.
 * @param x Query x coordinate.
 * @param y Query y coordinate.
 * @param z_array Array of z candidates to test.
 * @param size_z Number of entries in z_array and tets_array.
 * @param tets_array Output array of size size_z receiving tetrahedron indices.
 * @return 0 on completion.
 */
int                                                                                       //
query_cell_list_3d_2d_map_mesh_given_xy_tets_v(const cell_list_3d_2d_map_t *map,          //
                                               const boxes_t               *boxes,        //
                                               const mesh_tet_geom_t       *mesh_geom,    //
                                               const real_t                 x,            //
                                               const real_t                 y,            //
                                               const real_t                *z_array,      //
                                               const int                    size_z,       //
                                               int                         *tets_array);  //

/**
 * @brief Query a split 3D->2D cell-list map at fixed (x, y) over candidate z values.
 *
 * The lower map is queried first; if no tetrahedron is found, the upper map is
 * queried.
 *
 * @param map Pointer to the split cell-list map.
 * @param boxes Pointer to the bounding boxes associated with tetrahedra.
 * @param mesh_geom Pointer to tetrahedron geometric data used for point-in-tet tests.
 * @param x Query x coordinate.
 * @param y Query y coordinate.
 * @param z_array Array of z candidates to test at the same (x, y).
 * @param size_z Number of entries in z_array.
 * @return Tetrahedron index (>= 0) if found, -1 if no containing tetrahedron is found
 *         or if an input pointer is invalid.
 */
int                                                                                          //
query_cell_list_3d_2d_split_map_mesh_given_xy(const cell_list_split_3d_2d_map_t *map,        //
                                              const boxes_t                     *boxes,      //
                                              const mesh_tet_geom_t             *mesh_geom,  //
                                              const real_t                       x,          //
                                              const real_t                       y,          //
                                              const real_t                      *z_array,    //
                                              const int                          size_z);    //

/**
 * @brief Query a split 3D->2D cell-list map at fixed (x, y) for each z in z_array.
 *
 * The output array is initialized to -1 and then populated by querying both the
 * lower and upper maps.
 *
 * @param map Pointer to the split cell-list map.
 * @param boxes Pointer to the bounding boxes associated with tetrahedra.
 * @param mesh_geom Pointer to tetrahedron geometric data used for point-in-tet tests.
 * @param x Query x coordinate.
 * @param y Query y coordinate.
 * @param z_array Array of z candidates to test.
 * @param size_z Number of entries in z_array and tets_array.
 * @param tets_array Output array of size size_z receiving tetrahedron indices.
 * @return -1 if an input pointer is invalid. Otherwise, completion status is
 *         implementation-defined in the current source.
 */
int                                                                                                   //
query_cell_list_3d_2d_split_map_mesh_given_xy_tets_v(const cell_list_split_3d_2d_map_t *map,          //
                                                     const boxes_t                     *boxes,        //
                                                     const mesh_tet_geom_t             *mesh_geom,    //
                                                     const real_t                       x,            //
                                                     const real_t                       y,            //
                                                     const real_t                      *z_array,      //
                                                     const int                          size_z,       //
                                                     int                               *tets_array);  //

#endif  // __CELL_LIST_3D_MAP_MESH_H__
