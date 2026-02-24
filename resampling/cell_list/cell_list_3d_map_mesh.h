#ifndef __CELL_LIST_3D_MAP_MESH_H__
#define __CELL_LIST_3D_MAP_MESH_H__

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "cell_arg_sort.h"
#include "cell_list_3d_map.h"
#include "precision_types.h"
#include "sfem_mesh.h"

int                                                                              //
query_cell_list_3d_2d_map_mesh_given_xy(const cell_list_3d_2d_map_t *map,        //
                                        const boxes_t               *boxes,      //
                                        const mesh_tet_geom_t       *mesh_geom,  //
                                        const real_t                 x,          //
                                        const real_t                 y,          //
                                        const real_t                *z_array,    //
                                        const int                    size_z);                       //

int                                                                                     //
query_cell_list_3d_2d_map_mesh_given_xy_tets_v(const cell_list_3d_2d_map_t *map,        //
                                               const boxes_t               *boxes,      //
                                               const mesh_tet_geom_t       *mesh_geom,  //
                                               const real_t                 x,          //
                                               const real_t                 y,          //
                                               const real_t                *z_array,    //
                                               const int                    size_z,     //
                                               int                         *tets_array);                        //

int                                                                                          //
query_cell_list_3d_2d_split_map_mesh_given_xy(const cell_list_split_3d_2d_map_t *map,        //
                                              const boxes_t                     *boxes,      //
                                              const mesh_tet_geom_t             *mesh_geom,  //
                                              const real_t                       x,          //
                                              const real_t                       y,          //
                                              const real_t                      *z_array,    //
                                              const int                          size_z);                             //

int                                                                                                 //
query_cell_list_3d_2d_split_map_mesh_given_xy_tets_v(const cell_list_split_3d_2d_map_t *map,        //
                                                     const boxes_t                     *boxes,      //
                                                     const mesh_tet_geom_t             *mesh_geom,  //
                                                     const real_t                       x,          //
                                                     const real_t                       y,          //
                                                     const real_t                      *z_array,    //
                                                     const int                          size_z,     //
                                                     int                               *tets_array);                              //

#endif  // __CELL_LIST_3D_MAP_MESH_H__
