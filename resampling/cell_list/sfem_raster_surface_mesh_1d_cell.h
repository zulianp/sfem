#ifndef __SFEM_RASTER_SURFACE_MESH_1D_CELL_H__
#define __SFEM_RASTER_SURFACE_MESH_1D_CELL_H__

#include "cell_list_3d_1d_map_sur_mesh.h"

int                                                                                       //
raster_to_hex_field_cell_split_par_tri3(const cell_list_split_3d_1d_map_t   *split_map,   // Cell list split map data structure
                                        boxes_t                             *boxes,       // Boxes data structure
                                        const mesh_tri3_geom_t              *mesh_geom,   // Mesh geometry data structure
                                        const mesh_t *const SFEM_RESTRICT    mesh,        // Mesh: mesh_t struct
                                        const ptrdiff_t *const SFEM_RESTRICT n,           // SDF: n[3]
                                        const ptrdiff_t *const SFEM_RESTRICT stride,      // SDF: stride[3]
                                        const geom_t *const SFEM_RESTRICT    origin,      // SDF: origin[3]
                                        const geom_t *const SFEM_RESTRICT    delta,       // SDF: delta[3]
                                        real_t *const SFEM_RESTRICT          hex_field);  //

#endif  // __SFEM_RASTER_SURFACE_MESH_1D_CELL_H__