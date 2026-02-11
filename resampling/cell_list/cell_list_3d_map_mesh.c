#include "cell_list_3d_map_mesh.h"
#include "cell_arg_sort.h"
#include "cell_list_3d_map.h"
#include "sfem_mesh.h"

////////////////////////////////////////////////
// coord_to_grid_indices
////////////////////////////////////////////////
static inline void coord_to_grid_indices(const real_t coord[2], const real_t origin[2], const real_t delta[2], int indices[2]) {
    real_t inv_delta[2];

    for (int ii = 0; ii < 2; ii++) inv_delta[ii] = 1.0 / delta[ii];

    for (int ii = 0; ii < 2; ii++) {
        indices[ii] = (int)((coord[ii] - origin[ii]) * inv_delta[ii]);
    }
}

////////////////////////////////////////////////
// grid_to_cell_index
////////////////////////////////////////////////
static inline int grid_to_cell_index(int ix, int iy, int num_cells_x) { return ix + iy * num_cells_x; }

int                                              //
query_cell_list_3d_2d_map_mesh_given_xy(         //
        const cell_list_3d_2d_map_t *map,        //
        const boxes_t               *boxes,      //
        const mesh_tet_geom_t       *mesh_geom,  //
        const real_t                 x,          //
        const real_t                 y,          //
        const real_t                *z_array,    //
        const int                    size_z) {                      //

    // int ix = coord_to_grid_index(x, map->min_x, map->delta_x);
    // int iy = coord_to_grid_index(y, map->min_y, map->delta_y);

    int ixiy[2];
    coord_to_grid_indices((real_t[2]){x, y}, (real_t[2]){map->min_x, map->min_y}, (real_t[2]){map->delta_x, map->delta_y}, ixiy);
    int ix = ixiy[0];
    int iy = ixiy[1];

    ix = (ix < 0) ? 0 : (ix >= map->num_cells_x) ? map->num_cells_x - 1 : ix;
    iy = (iy < 0) ? 0 : (iy >= map->num_cells_y) ? map->num_cells_y - 1 : iy;

    const int cell_index = grid_to_cell_index(ix, iy, map->num_cells_x);

    const int start_index = map->cell_ptr[cell_index];
    const int end_index   = map->cell_ptr[cell_index + 1];

    int num_boxes_local = end_index - start_index;

    int boxes_found = 0;

    if (num_boxes_local > 0) {
        for (int iz = 0; iz < size_z; iz++) {
            const real_t z = z_array[iz];

            // int lower_bound_index = lower_bound_generic(
            //     &map->upper_bounds_z[start_index],
            //     (size_t)num_boxes_local,
            //     sizeof(real_t),
            //     &z,
            //     cmp_real_t);

            // int upper_bound_index = upper_bound_generic(
            //     &map->lower_bounds_z[start_index],
            //     (size_t)num_boxes_local,
            //     sizeof(real_t),
            //     &z,
            //     cmp_real_t);

            int lower_bound_index = lower_bound_float(&map->upper_bounds_z[start_index], num_boxes_local, z);

            const int start_index_up = (lower_bound_index > 1) ? start_index + lower_bound_index - 2 : start_index;
            const int size_up        = (lower_bound_index > 1) ? num_boxes_local - (lower_bound_index - 2) : num_boxes_local;
            const int offset_up      = start_index_up - start_index;

            int upper_bound_index = upper_bound_float(&map->lower_bounds_z[start_index_up], size_up, z);

            // Adjust upper_bound_index back to be relative to start_index
            upper_bound_index += offset_up;

            lower_bound_index =
                    lower_bound_index < 0 ? 0 : (lower_bound_index > num_boxes_local ? num_boxes_local : lower_bound_index);
            upper_bound_index =
                    upper_bound_index < 0 ? 0 : (upper_bound_index > num_boxes_local ? num_boxes_local : upper_bound_index);

            if (lower_bound_index >= upper_bound_index) {
                continue;
            }

            for (int i = lower_bound_index; i < upper_bound_index; i++) {
                const int box_index = map->cell_dict[start_index + i];

                if (check_box_contains_pt(boxes, box_index, x, y, z)) {
                    // We can further check if the point is actually inside the tet from which the box was generated, using
                    // mesh_geom and box_index to get the tet geometry and vertices.
                    real_t *inv_Jacobian  = get_inv_Jacobian_geom(mesh_geom, box_index);
                    real_t *vertices_zero = get_vertices_zero_geom(mesh_geom, box_index);

                    const bool is_out = is_point_out_of_tet(inv_Jacobian,  //
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

        }  // END: for (int iz = 0; iz < size_z; iz++)
    }  // END: if (num_boxes_local > 0)

    return -1;  // If not tet found containing the point, return -1 or a negative value to indicate not found.
}