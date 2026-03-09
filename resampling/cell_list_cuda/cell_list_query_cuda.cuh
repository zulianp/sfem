#ifndef __CELL_LIST_QUERY_CUDA_CUH__
#define __CELL_LIST_QUERY_CUDA_CUH__

#include "cell_list_cuda.cuh"

//////////////////////////////////////////////
// Convert 2D coordinates to grid indices
// coord_to_grid_indices
/////////////////////////////////////////////////
__device__ inline void                           //
coord_to_grid_indices_gpu(const real_t coord0,   //
                          const real_t coord1,   //
                          const real_t origin0,  //
                          const real_t origin1,  //
                          const real_t delta0,   //
                          const real_t delta1,   //
                          int          indices[2]) {      //
                                                 //
    const real_t inv_delta0 = 1.0 / delta0;
    const real_t inv_delta1 = 1.0 / delta1;

    indices[0] = (int)((coord0 - origin0) * inv_delta0);
    indices[1] = (int)((coord1 - origin1) * inv_delta1);
}

__device__ inline int                                      //
grid_to_cell_index_gpu(int ix, int iy, int num_cells_x) {  //
    return ix + iy * num_cells_x;
}

//////////////////////////////////////////////////////////
// Query the cell list for a given 3D point (x, y, z)
// and return the corresponding tetrahedra in tets_array
//////////////////////////////////////////////////////////
__device__ int                                                                          //
query_cell_list_3d_2d_map_mesh_given_xy_tets_v(const cell_list_3d_2d_map_t *map,        //
                                               const boxes_t               *boxes,      //
                                               const mesh_tet_geom_t       *mesh_geom,  //
                                               const real_t                 x,          //
                                               const real_t                 y,          //
                                               const real_t                 z,          //
                                               const int                    size_z) {                      //

    int ixiy[2];
    coord_to_grid_indices_gpu(x, y, map->min_x, map->min_y, map->delta_x, map->delta_y, ixiy);

    int ix = ixiy[0];
    int iy = ixiy[1];

    ix = (ix < 0) ? 0 : (ix >= map->num_cells_x) ? map->num_cells_x - 1 : ix;
    iy = (iy < 0) ? 0 : (iy >= map->num_cells_y) ? map->num_cells_y - 1 : iy;

    const int cell_index = grid_to_cell_index_gpu(ix, iy, map->num_cells_x);

    const int start_index = map->cell_ptr[cell_index];
    const int end_index   = map->cell_ptr[cell_index + 1];

    int num_boxes_local = end_index - start_index;

    int boxes_found = 0;
}

//////////////////////////////////////////////////////////
// Query the cell list for a given 3D point (x, y, z)
// and return the corresponding tetrahedra in tets_array
//////////////////////////////////////////////////////////
__device__ int                                         //
query_cell_list_3d_2d_split_map_mesh_given_xy_cuda(    //
        const cell_list_split_3d_2d_map_t *map,        //
        const boxes_t                     *boxes,      //
        const mesh_tet_geom_t             *mesh_geom,  //
        const real_t                       x,          //
        const real_t                       y,          //
        const real_t                       z,          //
        const int                          size_z) {                            //

    if (map == NULL || boxes == NULL || mesh_geom == NULL) {
        return -1;  // Invalid pointer
    }
    return 0;  // Success
}

#endif  // __CELL_LIST_QUERY_CUDA_CUH__