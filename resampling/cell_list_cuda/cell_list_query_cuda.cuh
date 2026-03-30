#ifndef __CELL_LIST_QUERY_CUDA_CUH__
#define __CELL_LIST_QUERY_CUDA_CUH__

#include "cell_list_cuda.cuh"
#include "sfem_gpu_math.cuh"

//////////////////////////////////////////////
// Convert 2D coordinates to grid indices
// coord_to_grid_indices
/////////////////////////////////////////////////
__device__ __forceinline__ void                       //
coord_to_grid_indices_gpu(const real_t coord0,        //
                          const real_t coord1,        //
                          const real_t origin0,       //
                          const real_t origin1,       //
                          const real_t delta0,        //
                          const real_t delta1,        //
                          int          indices[2]) {  //
                                                      //
    const real_t inv_delta0 = (real_t)(1) / delta0;
    const real_t inv_delta1 = (real_t)(1) / delta1;

    indices[0] = (int)((coord0 - origin0) * inv_delta0);
    indices[1] = (int)((coord1 - origin1) * inv_delta1);
}

//////////////////////////////////////////////
// Convert 2D coordinates to grid indices
// coord_to_grid_indices
/////////////////////////////////////////////////
__device__ __forceinline__ void                           //
coord_to_grid_indices_inv_gpu(const real_t coord0,        //
                              const real_t coord1,        //
                              const real_t origin0,       //
                              const real_t origin1,       //
                              const real_t inv_delta0,    //
                              const real_t inv_delta1,    //
                              int          indices[2]) {  //
                                                          //
    // const real_t inv_delta0 = (real_t)(1) * inv_delta0;
    // const real_t inv_delta1 = (real_t)(1) * inv_delta1;

    indices[0] = (int)((coord0 - origin0) * inv_delta0);
    indices[1] = (int)((coord1 - origin1) * inv_delta1);
}

/////////////////////////////////////////
// Convert grid indices to cell index
//////////////////////////////////////////
__device__ __forceinline__ int                             //
grid_to_cell_index_gpu(int ix, int iy, int num_cells_x) {  //
    return ix + iy * num_cells_x;
}

////////////////////////////////////////////////
// upper_bound_float32
////////////////////////////////////////////////
__device__ __forceinline__ int  //
upper_bound_float32_gpu(const float *__restrict__ elements_array, size_t nmemb, float to_search) {
    int low = 0, high = (int)nmemb;
    while (low < high) {
        const int  mid  = low + ((high - low) >> 1);
        const bool cond = __ldg(&elements_array[mid]) <= to_search;
        low             = cond ? mid + 1 : low;
        high            = cond ? high : mid;
    }
    return low;
}

////////////////////////////////////////////////
// lower_bound_float32
////////////////////////////////////////////////
__device__ __forceinline__ int  //
lower_bound_float32_gpu(const float *__restrict__ elements_array, size_t nmemb, float to_search) {
    int low = 0, high = (int)nmemb;
    while (low < high) {
        const int  mid  = low + ((high - low) >> 1);
        const bool cond = __ldg(&elements_array[mid]) < to_search;
        low             = cond ? mid + 1 : low;
        high            = cond ? high : mid;
    }
    return low;
}

////////////////////////////////////////////////
// upper_bound_float64
////////////////////////////////////////////////
__device__ __forceinline__ int  //
upper_bound_float64_gpu(const double *__restrict__ elements_array, size_t nmemb, double to_search) {
    int low = 0, high = (int)nmemb;
    while (low < high) {
        const int  mid  = low + ((high - low) >> 1);
        const bool cond = __ldg(&elements_array[mid]) <= to_search;
        low             = cond ? mid + 1 : low;
        high            = cond ? high : mid;
    }
    return low;
}
////////////////////////////////////////////////
// lower_bound_float64
////////////////////////////////////////////////
__device__ __forceinline__ int  //
lower_bound_float64_gpu(const double *__restrict__ elements_array, size_t nmemb, double to_search) {
    int low = 0, high = (int)nmemb;
    while (low < high) {
        const int  mid  = low + ((high - low) >> 1);
        const bool cond = __ldg(&elements_array[mid]) < to_search;
        low             = cond ? mid + 1 : low;
        high            = cond ? high : mid;
    }
    return low;
}
////////////////////////////////////////////////
// upper_bound_float (template)
////////////////////////////////////////////////
template <typename real_t>
__device__ __forceinline__ int  //
upper_bound_float_gpu(const real_t *elements_array, size_t nmemb, real_t to_search);

////////////////////////////////////////////////
// lower_bound_float (template)
////////////////////////////////////////////////
template <>
__device__ __forceinline__ int  //
upper_bound_float_gpu<float>(const float *elements_array, size_t nmemb, float to_search) {
    return upper_bound_float32_gpu(elements_array, nmemb, to_search);
}

/////////////////////////////////////////////
// upper_bound_float (template specialization for double)
/////////////////////////////////////////////
template <>
__device__ __forceinline__ int  //
upper_bound_float_gpu<double>(const double *elements_array, size_t nmemb, double to_search) {
    return upper_bound_float64_gpu(elements_array, nmemb, to_search);
}

////////////////////////////////////////////////
// lower_bound_float (template)
////////////////////////////////////////////////
template <typename real_t>
__device__ __forceinline__ int  //
lower_bound_float_gpu(const real_t *elements_array, size_t nmemb, real_t to_search);

template <>
__device__ __forceinline__ int  //
lower_bound_float_gpu<float>(const float *elements_array, size_t nmemb, float to_search) {
    return lower_bound_float32_gpu(elements_array, nmemb, to_search);
}

template <>
__device__ __forceinline__ int  //
lower_bound_float_gpu<double>(const double *elements_array, size_t nmemb, double to_search) {
    return lower_bound_float64_gpu(elements_array, nmemb, to_search);
}

/////////////////////////////////////////////////////
// get_inv_Jacobian_geom
/////////////////////////////////////////////////////
__device__ __forceinline__ real_t *                                                   //
get_inv_Jacobian_geom_gpu(const mesh_tet_geom_device_t *geom, ptrdiff_t element_i) {  //
    if (geom == NULL || geom->inv_Jacobian == NULL) {
        printf("Error: Invalid input to get_inv_Jacobian_geom\n");
        return NULL;
    }

    if (element_i < 0 || element_i >= geom->nelements) {
        printf("Error: element_i out of bounds in get_inv_Jacobian_geom\n");
        return NULL;
    }

    return &geom->inv_Jacobian[element_i * 9];
}

/////////////////////////////////////////////////////
// get_vertices_zero_geom
/////////////////////////////////////////////////////
__device__ __forceinline__ real_t *                                                    //
get_vertices_zero_geom_gpu(const mesh_tet_geom_device_t *geom, ptrdiff_t element_i) {  //
    if (geom == NULL || geom->vetices_zero == NULL) {
        printf("Error: Invalid input to get_vertices_zero_geom\n");
        return NULL;
    }

    if (element_i < 0 || element_i >= geom->nelements) {
        printf("Error: element_i out of bounds in get_vertices_zero_geom\n");
        return NULL;
    }

    return &geom->vetices_zero[element_i * 3];
}

/////////////////////////////////////////////////////
// get_inv_Jacobian_geom_fast_gpu
// Hot-path accessor: no NULL/bounds checks, no printf.
/////////////////////////////////////////////////////
__device__ __forceinline__ const real_t *                                                  //
get_inv_Jacobian_geom_fast_gpu(const mesh_tet_geom_device_t *geom, const int element_i) {  //
    return &geom->inv_Jacobian[element_i * 9];
}  // END Function: get_inv_Jacobian_geom_fast_gpu

/////////////////////////////////////////////////////
// get_vertices_zero_geom_fast_gpu
// Hot-path accessor: no NULL/bounds checks, no printf.
/////////////////////////////////////////////////////
__device__ __forceinline__ const real_t *                                                   //
get_vertices_zero_geom_fast_gpu(const mesh_tet_geom_device_t *geom, const int element_i) {  //
    return &geom->vetices_zero[element_i * 3];
}  // END Function: get_vertices_zero_geom_fast_gpu

////////////////////////////////////////////////
// check_box_containment
////////////////////////////////////////////////
__device__ __forceinline__ bool                      //
check_box_contains_pt_gpu(const boxes_t *boxes,      //
                          const int      box_index,  //
                          const real_t   x,          //
                          const real_t   y,          //
                          const real_t   z) {        //
                                                     //
    return (x >= boxes->min_x[box_index] &&          //
            x <= boxes->max_x[box_index] &&          //
            y >= boxes->min_y[box_index] &&          //
            y <= boxes->max_y[box_index] &&          //
            z >= boxes->min_z[box_index] &&          //
            z <= boxes->max_z[box_index]);
}

////////////////////////////////////////////////
// check_box_containment
////////////////////////////////////////////////
__device__ __forceinline__ bool                                     //
check_box_il_contains_pt_gpu(const boxes_interleaved_t *boxes,      //
                             const int                  box_index,  //
                             const real_t               x,          //
                             const real_t               y,          //
                             const real_t               z) {        //

    const real_t *bound = &boxes->min_max_xyz[box_index * 6];
    //
    return (x >= bound[0] &&  //
            x <= bound[3] &&  //
            y >= bound[1] &&  //
            y <= bound[4] &&  //
            z >= bound[2] &&  //
            z <= bound[5]);
}

////////////////////////////////////////////////
// check_box_containment for bound managed
// by the user (not strictly necessarily in SoA format)
////////////////////////////////////////////////
__device__ __forceinline__ bool                          //
check_box_il_contains_pt_bound_gpu(const real_t bound0,  //
                                   const real_t bound1,  //
                                   const real_t bound2,  //
                                   const real_t bound3,  //
                                   const real_t bound4,  //
                                   const real_t bound5,  //
                                   const real_t x,       //
                                   const real_t y,       //
                                   const real_t z) {     //

    //
    return (x >= bound0 &&  //
            x <= bound3 &&  //
            y >= bound1 &&  //
            y <= bound4 &&  //
            z >= bound2 &&  //
            z <= bound5);
}

/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
// is_point_out_of_tet
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
__device__ __forceinline__ bool                     //
is_point_out_of_tet_gpu(const real_t inv_J_tet[9],  //
                        const real_t tet_origin_x,  //
                        const real_t tet_origin_y,  //
                        const real_t tet_origin_z,  //
                        const real_t vertex_x,      //
                        const real_t vertex_y,      //
                        const real_t vertex_z) {    //

    // Precompute inverse Jacobian components for better cache utilization
    const real_t inv_J00 = inv_J_tet[0];
    const real_t inv_J01 = inv_J_tet[1];
    const real_t inv_J02 = inv_J_tet[2];
    const real_t inv_J10 = inv_J_tet[3];
    const real_t inv_J11 = inv_J_tet[4];
    const real_t inv_J12 = inv_J_tet[5];
    const real_t inv_J20 = inv_J_tet[6];
    const real_t inv_J21 = inv_J_tet[7];
    const real_t inv_J22 = inv_J_tet[8];

    // Transform point to tet reference space
    const real_t dx = vertex_x - tet_origin_x;
    const real_t dy = vertex_y - tet_origin_y;
    const real_t dz = vertex_z - tet_origin_z;

    const real_t ref_x = fast_fma(inv_J00, dx, fast_fma(inv_J01, dy, inv_J02 * dz));
    const real_t ref_y = fast_fma(inv_J10, dx, fast_fma(inv_J11, dy, inv_J12 * dz));
    const real_t ref_z = fast_fma(inv_J20, dx, fast_fma(inv_J21, dy, inv_J22 * dz));

    // Check if point is inside reference tetrahedron
    // A point is inside if: ref_x >= 0, ref_y >= 0, ref_z >= 0, and ref_x + ref_y + ref_z <= 1
    const bool inside = ref_x >= (real_t)(0) &&                    //
                        ref_y >= (real_t)(0) &&                    //
                        ref_z >= (real_t)(0) &&                    //
                        ((ref_x + ref_y + ref_z) <= (real_t)(1));  //

    // Return true if point is outside
    return !(inside);

}  // END Function: is_point_out_of_tet

/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
// is_point_out_of_tet_cached
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
__device__ __forceinline__ bool                            //
is_point_out_of_tet_cached_gpu(const real_t inv_J00,       //
                               const real_t inv_J01,       //
                               const real_t inv_J02,       //
                               const real_t inv_J10,       //
                               const real_t inv_J11,       //
                               const real_t inv_J12,       //
                               const real_t inv_J20,       //
                               const real_t inv_J21,       //
                               const real_t inv_J22,       //
                               const real_t tet_origin_x,  //
                               const real_t tet_origin_y,  //
                               const real_t tet_origin_z,  //
                               const real_t vertex_x,      //
                               const real_t vertex_y,      //
                               const real_t vertex_z) {    //

    const real_t dx = vertex_x - tet_origin_x;
    const real_t dy = vertex_y - tet_origin_y;
    const real_t dz = vertex_z - tet_origin_z;

    const real_t ref_x = fast_fma(inv_J00, dx, fast_fma(inv_J01, dy, inv_J02 * dz));
    const real_t ref_y = fast_fma(inv_J10, dx, fast_fma(inv_J11, dy, inv_J12 * dz));
    const real_t ref_z = fast_fma(inv_J20, dx, fast_fma(inv_J21, dy, inv_J22 * dz));

    const bool inside = ref_x >= (real_t)(0) &&                    //
                        ref_y >= (real_t)(0) &&                    //
                        ref_z >= (real_t)(0) &&                    //
                        ((ref_x + ref_y + ref_z) <= (real_t)(1));  //

    return !(inside);
}  // END Function: is_point_out_of_tet_cached_gpu

//////////////////////////////////////////////////////////
// Query the cell list for a given 3D point (x, y, z)
// and return the corresponding tetrahedra in tets_array
//////////////////////////////////////////////////////////
__device__ int                                                                            //
query_cell_list_3d_2d_map_mesh_given_xy_tet_gpu(const cell_list_3d_2d_map_t  *map,        //
                                                const boxes_t                *boxes,      //
                                                const mesh_tet_geom_device_t *mesh_geom,  //
                                                const real_t                  x,          //
                                                const real_t                  y,          //
                                                const real_t                  z) {        //

    int ixiy[2];
    coord_to_grid_indices_gpu(x, y, map->min_x, map->min_y, map->delta_x, map->delta_y, ixiy);

    int ix = ixiy[0];
    int iy = ixiy[1];

    ix = (ix < 0) ? 0 : (ix >= map->num_cells_x) ? map->num_cells_x - 1 : ix;
    iy = (iy < 0) ? 0 : (iy >= map->num_cells_y) ? map->num_cells_y - 1 : iy;

    const int cell_index = grid_to_cell_index_gpu(ix, iy, map->num_cells_x);

    const int start_index = map->cell_ptr[cell_index];
    const int end_index   = map->cell_ptr[cell_index + 1];

    const int num_boxes_local = end_index - start_index;

    if (num_boxes_local > 0) {
        int lower_bound_index = lower_bound_float_gpu<real_t>(&map->upper_bounds_z[start_index], num_boxes_local, z);

        const int start_index_up = (lower_bound_index > 1) ? start_index + lower_bound_index - 2 : start_index;
        const int size_up        = (lower_bound_index > 1) ? num_boxes_local - (lower_bound_index - 2) : num_boxes_local;
        const int offset_up      = start_index_up - start_index;

        int upper_bound_index = upper_bound_float_gpu<real_t>(&map->lower_bounds_z[start_index_up], size_up, z);

        // Adjust upper_bound_index back to be relative to start_index
        upper_bound_index += offset_up;

        lower_bound_index =
                lower_bound_index < 0 ? 0 : (lower_bound_index > num_boxes_local ? num_boxes_local : lower_bound_index);
        upper_bound_index =
                upper_bound_index < 0 ? 0 : (upper_bound_index > num_boxes_local ? num_boxes_local : upper_bound_index);

        if (lower_bound_index >= upper_bound_index) {
            return -1;  // No boxes found for this z value
        }

        for (int i = lower_bound_index; i < upper_bound_index; i++) {
            const int box_index = map->cell_dict[start_index + i];

            if (check_box_contains_pt_gpu(boxes, box_index, x, y, z)) {
                const real_t *inv_Jacobian  = get_inv_Jacobian_geom_gpu(mesh_geom, box_index);
                const real_t *vertices_zero = get_vertices_zero_geom_gpu(mesh_geom, box_index);

                const bool is_out = is_point_out_of_tet_gpu(inv_Jacobian,  //
                                                            vertices_zero[0],
                                                            vertices_zero[1],
                                                            vertices_zero[2],
                                                            x,
                                                            y,
                                                            z);

                if (!is_out) {
                    return box_index;  // Return the index of the first box found containing the point
                    // If the mesh is well-behaved and the boxes are tight around the tets,
                    // we can expect to find at most one box containing the point.
                }
            }
        }
    }  // END if (num_boxes_local > 0)
    return -1;  // No box found containing the point
}  // END Function: query_cell_list_3d_2d_map_mesh_given_xy_tet_gpu

//////////////////////////////////////////////////////////
// Query the cell list for a given 3D point (x, y, z)
// and return the corresponding tetrahedra in tets_array
//////////////////////////////////////////////////////////
__device__ __forceinline__ int                         //
query_cell_list_3d_2d_split_map_mesh_given_xy_gpu(     //
        const cell_list_split_3d_2d_map_t *map,        //
        const boxes_t                     *boxes,      //
        const mesh_tet_geom_device_t      *mesh_geom,  //
        const real_t                       x,          //
        const real_t                       y,          //
        const real_t                       z) {        //

    if (map == NULL || boxes == NULL || mesh_geom == NULL) {
        return -1;  // Invalid pointer
    }

    const int tet_lower_idx = query_cell_list_3d_2d_map_mesh_given_xy_tet_gpu(map->map_lower, boxes, mesh_geom, x, y, z);

    if (tet_lower_idx != -1) {
        return tet_lower_idx;  // Found in lower map
    }

    return query_cell_list_3d_2d_map_mesh_given_xy_tet_gpu(map->map_upper, boxes, mesh_geom, x, y, z);
}  // END Function: query_cell_list_3d_2d_split_map_mesh_given_xy_gpu

//////////////////////////////////////////////////////////
// Query the cell list for a given 3D point (x, y, z)
// and return the corresponding tetrahedra in tets_array
//////////////////////////////////////////////////////////
__device__ int                                                                               //
query_cell_list_3d_2d_map_mesh_given_xy_tet_il_gpu(const cell_list_3d_2d_map_t  *map,        //
                                                   const boxes_interleaved_t    *boxes,      //
                                                   const mesh_tet_geom_device_t *mesh_geom,  //
                                                   const real_t                  x,          //
                                                   const real_t                  y,          //
                                                   const real_t                  z) {        //

    int ixiy[2];
    coord_to_grid_indices_inv_gpu(x,  //
                                  y,  //
                                  map->min_x,
                                  map->min_y,
                                  map->inv_delta_x,
                                  map->inv_delta_y,
                                  ixiy);

    int ix = ixiy[0];
    int iy = ixiy[1];

    ix = (ix < 0) ? 0 : (ix >= map->num_cells_x) ? map->num_cells_x - 1 : ix;
    iy = (iy < 0) ? 0 : (iy >= map->num_cells_y) ? map->num_cells_y - 1 : iy;

    const int cell_index = grid_to_cell_index_gpu(ix, iy, map->num_cells_x);

    const int start_index = __ldg(&map->cell_ptr[cell_index]);
    const int end_index   = __ldg(&map->cell_ptr[cell_index + 1]);

    const int num_boxes_local = end_index - start_index;

    if (num_boxes_local > 0) {
        // Early reject: z is outside the full z-range of all boxes in this cell
        if (z < __ldg(&map->lower_bounds_z[start_index]) || z > __ldg(&map->upper_bounds_z[end_index - 1])) {
            return -1;  // No boxes found for this z value
        }

        int lower_bound_index = lower_bound_float_gpu<real_t>(&map->upper_bounds_z[start_index], num_boxes_local, z);

        const int start_index_up = (lower_bound_index > 1) ? start_index + lower_bound_index - 2 : start_index;
        const int size_up        = (lower_bound_index > 1) ? num_boxes_local - (lower_bound_index - 2) : num_boxes_local;
        const int offset_up      = start_index_up - start_index;

        int upper_bound_index = upper_bound_float_gpu<real_t>(&map->lower_bounds_z[start_index_up], size_up, z);

        // Adjust upper_bound_index back to be relative to start_index
        upper_bound_index += offset_up;

        lower_bound_index =
                lower_bound_index < 0 ? 0 : (lower_bound_index > num_boxes_local ? num_boxes_local : lower_bound_index);
        upper_bound_index =
                upper_bound_index < 0 ? 0 : (upper_bound_index > num_boxes_local ? num_boxes_local : upper_bound_index);

        if (lower_bound_index >= upper_bound_index) {
            return -1;  // No boxes found for this z value
        }

        for (int i = lower_bound_index; i < upper_bound_index; i++) {
            const int box_index = __ldg(&map->cell_dict[start_index + i]);

            const real_t *box_bound = &boxes->min_max_xyz[box_index * 6];

            if (check_box_il_contains_pt_bound_gpu(__ldg(&box_bound[0]),  //
                                                   __ldg(&box_bound[1]),
                                                   __ldg(&box_bound[2]),
                                                   __ldg(&box_bound[3]),
                                                   __ldg(&box_bound[4]),
                                                   __ldg(&box_bound[5]),
                                                   x,
                                                   y,
                                                   z)) {
                const real_t *inv_Jacobian  = get_inv_Jacobian_geom_fast_gpu(mesh_geom, box_index);
                const real_t *vertices_zero = get_vertices_zero_geom_fast_gpu(mesh_geom, box_index);

                const bool is_out = is_point_out_of_tet_cached_gpu(inv_Jacobian[0],  //
                                                                   inv_Jacobian[1],
                                                                   inv_Jacobian[2],
                                                                   inv_Jacobian[3],
                                                                   inv_Jacobian[4],
                                                                   inv_Jacobian[5],
                                                                   inv_Jacobian[6],
                                                                   inv_Jacobian[7],
                                                                   inv_Jacobian[8],
                                                                   vertices_zero[0],
                                                                   vertices_zero[1],
                                                                   vertices_zero[2],
                                                                   x,
                                                                   y,
                                                                   z);

                if (!is_out) {
                    return box_index;  // Return the index of the first box found containing the point
                    // If the mesh is well-behaved and the boxes are tight around the tets,
                    // we can expect to find at most one box containing the point.
                }
            }
        }
    }  // END if (num_boxes_local > 0)
    return -1;  // No box found containing the point
}  // END Function: query_cell_list_3d_2d_map_mesh_given_xy_tet_gpu

//////////////////////////////////////////////////////////
// get_cell_range_2d_gpu
// Compute the [start_index, end_index) range in cell_dict
// for the 2D cell containing point (x, y).
// This separates the xy-lookup from the z-search so it can
// be hoisted out of a z-iteration loop.
//////////////////////////////////////////////////////////
__device__ __forceinline__ void                                                   //
get_cell_range_2d_gpu(const cell_list_3d_2d_map_t *__restrict__ map,             //
                      const real_t                              x,               //
                      const real_t                              y,               //
                      int                                      *start_index,     //
                      int                                      *end_index) {     //

    int ixiy[2];
    coord_to_grid_indices_inv_gpu(x, y, map->min_x, map->min_y,
                                  map->inv_delta_x, map->inv_delta_y, ixiy);

    int ix = ixiy[0];
    int iy = ixiy[1];

    ix = (ix < 0) ? 0 : (ix >= map->num_cells_x) ? map->num_cells_x - 1 : ix;
    iy = (iy < 0) ? 0 : (iy >= map->num_cells_y) ? map->num_cells_y - 1 : iy;

    const int cell_index = grid_to_cell_index_gpu(ix, iy, map->num_cells_x);

    *start_index = __ldg(&map->cell_ptr[cell_index]);
    *end_index   = __ldg(&map->cell_ptr[cell_index + 1]);
}  // END Function: get_cell_range_2d_gpu

//////////////////////////////////////////////////////////
// query_cell_list_z_given_range_il_gpu
// Given a precomputed cell range [start_index, end_index)
// (from get_cell_range_2d_gpu), search only in z.
// Avoids repeating the 2D coord-to-cell lookup when
// (x, y) is fixed across many z values.
//////////////////////////////////////////////////////////
__device__ int                                                                                          //
query_cell_list_z_given_range_il_gpu(const cell_list_3d_2d_map_t  *__restrict__ map,                   //
                                     const boxes_interleaved_t    *__restrict__ boxes,                 //
                                     const mesh_tet_geom_device_t *__restrict__ mesh_geom,             //
                                     const int                                  start_index,           //
                                     const int                                  end_index,             //
                                     const real_t                               x,                     //
                                     const real_t                               y,                     //
                                     const real_t                               z) {                   //

    const int num_boxes_local = end_index - start_index;

    if (num_boxes_local <= 0) {
        return -1;
    }

    // Early reject: z outside the full z-range of all boxes in this cell
    if (z < __ldg(&map->lower_bounds_z[start_index]) || z > __ldg(&map->upper_bounds_z[end_index - 1])) {
        return -1;
    }

    int lower_bound_index = lower_bound_float_gpu<real_t>(&map->upper_bounds_z[start_index], num_boxes_local, z);

    const int start_index_up = (lower_bound_index > 1) ? start_index + lower_bound_index - 2 : start_index;
    const int size_up        = (lower_bound_index > 1) ? num_boxes_local - (lower_bound_index - 2) : num_boxes_local;
    const int offset_up      = start_index_up - start_index;

    int upper_bound_index = upper_bound_float_gpu<real_t>(&map->lower_bounds_z[start_index_up], size_up, z);
    upper_bound_index += offset_up;

    lower_bound_index =
            lower_bound_index < 0 ? 0 : (lower_bound_index > num_boxes_local ? num_boxes_local : lower_bound_index);
    upper_bound_index =
            upper_bound_index < 0 ? 0 : (upper_bound_index > num_boxes_local ? num_boxes_local : upper_bound_index);

    if (lower_bound_index >= upper_bound_index) {
        return -1;
    }

    for (int i = lower_bound_index; i < upper_bound_index; i++) {
        const int     box_index = __ldg(&map->cell_dict[start_index + i]);
        const real_t *box_bound = &boxes->min_max_xyz[box_index * 6];

        if (check_box_il_contains_pt_bound_gpu(__ldg(&box_bound[0]),  //
                                               __ldg(&box_bound[1]),
                                               __ldg(&box_bound[2]),
                                               __ldg(&box_bound[3]),
                                               __ldg(&box_bound[4]),
                                               __ldg(&box_bound[5]),
                                               x, y, z)) {
            const real_t *inv_Jacobian  = get_inv_Jacobian_geom_fast_gpu(mesh_geom, box_index);
            const real_t *vertices_zero = get_vertices_zero_geom_fast_gpu(mesh_geom, box_index);

            const bool is_out = is_point_out_of_tet_cached_gpu(__ldg(&inv_Jacobian[0]),  //
                                                               __ldg(&inv_Jacobian[1]),
                                                               __ldg(&inv_Jacobian[2]),
                                                               __ldg(&inv_Jacobian[3]),
                                                               __ldg(&inv_Jacobian[4]),
                                                               __ldg(&inv_Jacobian[5]),
                                                               __ldg(&inv_Jacobian[6]),
                                                               __ldg(&inv_Jacobian[7]),
                                                               __ldg(&inv_Jacobian[8]),
                                                               __ldg(&vertices_zero[0]),
                                                               __ldg(&vertices_zero[1]),
                                                               __ldg(&vertices_zero[2]),
                                                               x, y, z);

            if (!is_out) {
                return box_index;
            }
        }
    }
    return -1;
}  // END Function: query_cell_list_z_given_range_il_gpu

//////////////////////////////////////////////////////////
// Query the cell list for a given 3D point (x, y, z)
// and return the corresponding tetrahedra in tets_array
//////////////////////////////////////////////////////////
__device__ int                                         //
query_cell_list_3d_2d_split_map_mesh_given_xy_il_gpu(  //
        const cell_list_split_3d_2d_map_t *map,        //
        const boxes_interleaved_t         *boxes,      //
        const mesh_tet_geom_device_t      *mesh_geom,  //
        const real_t                       x,          //
        const real_t                       y,          //
        const real_t                       z) {        //

    if (map == NULL || boxes == NULL || mesh_geom == NULL) {
        return -1;  // Invalid pointer
    }

    const int tet_lower_idx = query_cell_list_3d_2d_map_mesh_given_xy_tet_il_gpu(map->map_lower, boxes, mesh_geom, x, y, z);

    if (tet_lower_idx != -1) {
        return tet_lower_idx;  // Found in lower map
    }

    return query_cell_list_3d_2d_map_mesh_given_xy_tet_il_gpu(map->map_upper, boxes, mesh_geom, x, y, z);
}  // END Function: query_cell_list_3d_2d_split_map_mesh_given_xy_il_gpu

#endif  // __CELL_LIST_QUERY_CUDA_CUH__