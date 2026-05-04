#include "cell_list_3d_1d_map_sur_mesh.h"

#include "cell_tet2box.h"

////////////////////////////////////////////////
// coord_to_grid_index
////////////////////////////////////////////////
static inline int coord_to_grid_index(real_t coord, real_t origin, real_t delta) { return (int)((coord - origin) / delta); }

//////////////////////////////////////////////////
// count_box_into_map
//////////////////////////////////////////////////
static void count_box_into_map(cell_list_3d_1d_map_t *map, const real_t box_min, const real_t box_max) {
    int ix_min = coord_to_grid_index(box_min, map->min_x, map->delta_x);
    int ix_max = coord_to_grid_index(box_max, map->min_x, map->delta_x);

    if (ix_min < 0) ix_min = 0;
    if (ix_max >= map->num_cells_x) ix_max = map->num_cells_x - 1;

    for (int ix = ix_min; ix <= ix_max; ix++) {
        map->cell_ptr[ix + 1] += 1;
    }
}

//////////////////////////////////////////////////
// fill_box_into_map
//////////////////////////////////////////////////
static void fill_box_into_map(cell_list_3d_1d_map_t *map, const int box_index, const real_t *box_min_x, const real_t *box_max_x,
                              const real_t *box_min_y, const real_t *box_max_y, int *current_count) {
    int ix_min = coord_to_grid_index(box_min_x[box_index], map->min_x, map->delta_x);
    int ix_max = coord_to_grid_index(box_max_x[box_index], map->min_x, map->delta_x);

    if (ix_min < 0) ix_min = 0;
    if (ix_max >= map->num_cells_x) ix_max = map->num_cells_x - 1;

    for (int ix = ix_min; ix <= ix_max; ix++) {
        const int idx            = map->cell_ptr[ix] + current_count[ix];
        map->cell_dict[idx]      = box_index;
        map->lower_bounds_y[idx] = box_min_y[box_index];
        map->upper_bounds_y[idx] = box_max_y[box_index];
        current_count[ix]++;
    }
}

////////////////////////////////////////////////
// cmp_real_t
////////////////////////////////////////////////
static int cmp_real_t(const void *a, const void *b) {
    real_t diff = *(const real_t *)a - *(const real_t *)b;
    return (diff > 0) - (diff < 0);
}

////////////////////////////////////////////////
// sort_map_cells_by_lower_bounds_y
////////////////////////////////////////////////
static void propagate_upper_bounds_y(cell_list_3d_1d_map_t *map) {
    for (int cell_index = 0; cell_index < map->num_cells_x; cell_index++) {
        const int start_index = map->cell_ptr[cell_index];
        const int end_index   = map->cell_ptr[cell_index + 1];
        const int size        = end_index - start_index;
        if (size == 0) continue;

        for (int i = 1; i < size; i++) {
            if (map->upper_bounds_y[start_index + i - 1] > map->upper_bounds_y[start_index + i])
                map->upper_bounds_y[start_index + i] = map->upper_bounds_y[start_index + i - 1];
        }
    }
}

//////////////////////////////////////////////
// sort_map_cells_by_lower_bounds_y
//////////////////////////////////////////////
static void sort_map_cells_by_lower_bounds_y(cell_list_3d_1d_map_t *map, int *size_arg_indices, int **arg_indices,
                                             real_t **buffer, int **buffer_int) {
    for (int cell_index = 0; cell_index < map->num_cells_x; cell_index++) {
        const int start_index = map->cell_ptr[cell_index];
        const int end_index   = map->cell_ptr[cell_index + 1];
        const int size        = end_index - start_index;
        if (size == 0) continue;

        if (size > *size_arg_indices) {
            *size_arg_indices = size;
            *arg_indices      = (int *)realloc(*arg_indices, *size_arg_indices * sizeof(int));
            *buffer           = (real_t *)realloc(*buffer, *size_arg_indices * sizeof(real_t));
            *buffer_int       = (int *)realloc(*buffer_int, *size_arg_indices * sizeof(int));
        }

        for (int i = 0; i < size; i++) (*arg_indices)[i] = i;

        real_t *lower_y = &map->lower_bounds_y[start_index];
        argsort(*arg_indices, lower_y, (size_t)size, sizeof(real_t), cmp_real_t);

        for (int i = 0; i < size; i++) (*buffer_int)[i] = map->cell_dict[start_index + (*arg_indices)[i]];
        for (int i = 0; i < size; i++) map->cell_dict[start_index + i] = (*buffer_int)[i];

        for (int i = 0; i < size; i++) (*buffer)[i] = map->lower_bounds_y[start_index + (*arg_indices)[i]];
        for (int i = 0; i < size; i++) map->lower_bounds_y[start_index + i] = (*buffer)[i];

        for (int i = 0; i < size; i++) (*buffer)[i] = map->upper_bounds_y[start_index + (*arg_indices)[i]];
        for (int i = 0; i < size; i++) map->upper_bounds_y[start_index + i] = (*buffer)[i];
    }
}

///////////////////////////////////////////////
// fill_cell_lists_3d_1d_split_map
///////////////////////////////////////////////
int                                                                //
fill_cell_lists_3d_1d_split_map(cell_list_3d_1d_map_t *map_lower,  //
                                cell_list_3d_1d_map_t *map_upper,  //
                                const real_t           split_x,    //
                                const real_t          *box_min_x,  //
                                const real_t          *box_min_y,  //
                                const real_t          *box_min_z,  //
                                const real_t          *box_max_x,  //
                                const real_t          *box_max_y,  //
                                const real_t          *box_max_z,  //
                                const int              num_boxes,  //
                                const real_t           x_min,      //
                                const real_t           x_max,      //
                                const real_t           y_min,      //
                                const real_t           y_max,      //
                                const real_t           z_min,      //
                                const real_t           z_max) {    //

    real_t max_delta_x_lower = 0.0;
    real_t max_delta_y_lower = 0.0;
    real_t max_delta_z_lower = 0.0;

    map_upper->delta_x = 0.0;
    map_upper->delta_y = 0.0;
    map_upper->delta_z = 0.0;

    map_lower->min_x = x_min;
    map_lower->min_y = y_min;
    map_lower->min_z = z_min;
    map_lower->max_x = x_max;
    map_lower->max_y = y_max;
    map_lower->max_z = z_max;

    map_upper->min_x = x_min;
    map_upper->min_y = y_min;
    map_upper->min_z = z_min;
    map_upper->max_x = x_max;
    map_upper->max_y = y_max;
    map_upper->max_z = z_max;

    for (int i = 0; i < num_boxes; i++) {
        const real_t dx = fabs(box_max_x[i] - box_min_x[i]);
        const real_t dy = fabs(box_max_y[i] - box_min_y[i]);
        const real_t dz = fabs(box_max_z[i] - box_min_z[i]);

        if (dx < split_x) {
            if (dx > max_delta_x_lower) max_delta_x_lower = dx;
            if (dy > max_delta_y_lower) max_delta_y_lower = dy;
            if (dz > max_delta_z_lower) max_delta_z_lower = dz;
        } else {
            if (dx > map_upper->delta_x) map_upper->delta_x = dx;
            if (dy > map_upper->delta_y) map_upper->delta_y = dy;
            if (dz > map_upper->delta_z) map_upper->delta_z = dz;
        }
    }

    map_lower->delta_x = max_delta_x_lower;
    map_lower->delta_y = max_delta_y_lower;
    map_lower->delta_z = max_delta_z_lower;

    map_lower->num_cells_x = (int)ceil((x_max - x_min) / map_lower->delta_x);
    map_upper->num_cells_x = (int)ceil((x_max - x_min) / map_upper->delta_x);

    map_lower->cell_ptr = (int *)calloc(map_lower->num_cells_x + 1, sizeof(int));
    map_upper->cell_ptr = (int *)calloc(map_upper->num_cells_x + 1, sizeof(int));

    for (int i = 0; i < num_boxes; i++) {
        const real_t dx = fabs(box_max_x[i] - box_min_x[i]);

        if (dx < split_x) {
            count_box_into_map(map_lower, box_min_x[i], box_max_x[i]);
        } else {
            count_box_into_map(map_upper, box_min_x[i], box_max_x[i]);
        }
    }

    // Accumulate counts to make the lookup pointer array
    for (int i = 1; i <= map_lower->num_cells_x; i++) {
        map_lower->cell_ptr[i] += map_lower->cell_ptr[i - 1];
    }

    for (int i = 1; i <= map_upper->num_cells_x; i++) {
        map_upper->cell_ptr[i] += map_upper->cell_ptr[i - 1];
    }

    const int total_num_dict_entries_lower = map_lower->cell_ptr[map_lower->num_cells_x];
    map_lower->total_num_dict_entries      = total_num_dict_entries_lower;

    const int total_num_dict_entries_upper = map_upper->cell_ptr[map_upper->num_cells_x];
    map_upper->total_num_dict_entries      = total_num_dict_entries_upper;

    map_lower->cell_dict      = (int *)malloc(total_num_dict_entries_lower * sizeof(int));
    map_lower->lower_bounds_y = (real_t *)calloc(total_num_dict_entries_lower, sizeof(real_t));
    map_lower->upper_bounds_y = (real_t *)calloc(total_num_dict_entries_lower, sizeof(real_t));

    map_upper->cell_dict      = (int *)malloc(total_num_dict_entries_upper * sizeof(int));
    map_upper->lower_bounds_y = (real_t *)calloc(total_num_dict_entries_upper, sizeof(real_t));
    map_upper->upper_bounds_y = (real_t *)calloc(total_num_dict_entries_upper, sizeof(real_t));

    int *current_count_lower = (int *)calloc(map_lower->num_cells_x, sizeof(int));
    int *current_count_upper = (int *)calloc(map_upper->num_cells_x, sizeof(int));

    // Fill the cell dictionary array
    for (int i = 0; i < num_boxes; i++) {
        const real_t dx = fabs(box_max_x[i] - box_min_x[i]);

        if (dx < split_x) {
            fill_box_into_map(map_lower, i, box_min_x, box_max_x, box_min_y, box_max_y, current_count_lower);
        } else {
            fill_box_into_map(map_upper, i, box_min_x, box_max_x, box_min_y, box_max_y, current_count_upper);
        }
    }

    free(current_count_lower);
    free(current_count_upper);

    int     size_arg_indices = 2024;
    int    *arg_indices      = (int *)malloc(size_arg_indices * sizeof(int));
    real_t *buffer           = (real_t *)malloc(size_arg_indices * sizeof(real_t));
    int    *buffer_int       = (int *)malloc(size_arg_indices * sizeof(int));

    sort_map_cells_by_lower_bounds_y(map_lower, &size_arg_indices, &arg_indices, &buffer, &buffer_int);
    sort_map_cells_by_lower_bounds_y(map_upper, &size_arg_indices, &arg_indices, &buffer, &buffer_int);

    free(arg_indices);
    free(buffer);
    free(buffer_int);

    propagate_upper_bounds_y(map_lower);
    propagate_upper_bounds_y(map_upper);

#ifdef DEBUG_OUTPUT
    printf("\nCell 3D-1D list built successfully.\n");
#endif

    return EXIT_SUCCESS;

}  // END Function: fill_cell_lists_3d_1d_split_map

//////////////////////////////////////////////////
// build_cell_list_split_3d_1d_map_mesh
//////////////////////////////////////////////////
int                                                                            //
build_cell_list_split_3d_1d_map_mesh(cell_list_split_3d_1d_map_t **split_map,  //
                                     const mesh_t                 *mesh,       //
                                     const boxes_t                *boxes) {    //

    if (split_map == NULL) {
        fprintf(stderr, "Error: split_map pointer is NULL.\n");
        return -1;  // Invalid pointer
    }

    if (mesh == NULL) {
        fprintf(stderr, "Error: mesh pointer is NULL.\n");
        return -1;  // Invalid pointer
    }

    if (boxes == NULL) {
        fprintf(stderr, "Error: boxes pointer is NULL.\n");
        return -1;  // Invalid pointer
    }

    *split_map = (cell_list_split_3d_1d_map_t *)malloc(sizeof(cell_list_split_3d_1d_map_t));
    if (*split_map == NULL) {
        fprintf(stderr, "Error: Memory allocation for split_map failed.\n");
        return -1;  // Memory allocation failure
    }

    (*split_map)->split_x   = 0.0;
    (*split_map)->split_y   = 0.0;
    (*split_map)->map_lower = make_empty_cell_list_3d_1d_map();
    (*split_map)->map_upper = make_empty_cell_list_3d_1d_map();

    if ((*split_map)->map_lower == NULL || (*split_map)->map_upper == NULL) {
        fprintf(stderr, "Error: Memory allocation for split sub-maps failed.\n");
        free_cell_list_3d_1d_map((*split_map)->map_lower);
        free_cell_list_3d_1d_map((*split_map)->map_upper);
        free(*split_map);
        *split_map = NULL;
        return -1;
    }

    // Check if the mesh is empty
    if (mesh->nelements == 0 || mesh->nnodes == 0) {
        fprintf(stderr, "Error: Mesh is empty. Cannot build cell list.\n");
        free_cell_list_3d_1d_map((*split_map)->map_lower);
        free_cell_list_3d_1d_map((*split_map)->map_upper);
        free(*split_map);
        *split_map = NULL;
        return EXIT_FAILURE;
    }

    const int num_boxes = boxes->num_boxes;

    if (num_boxes <= 0) {
        fprintf(stderr, "Error: No boxes available. Cannot build split cell list.\n");
        free_cell_list_3d_1d_map((*split_map)->map_lower);
        free_cell_list_3d_1d_map((*split_map)->map_upper);
        free(*split_map);
        *split_map = NULL;
        return EXIT_FAILURE;
    }

    bounding_box_statistics_t    stats      = calculate_bounding_box_statistics(boxes);
    side_length_histograms_t     histograms = calculate_side_length_histograms(boxes, &stats, 50);
    side_length_cdf_thresholds_t thresholds = calculate_cdf_thresholds(&histograms, 0.96, 0.96, 0.96);
    free_side_length_histograms(&histograms);

    (*split_map)->split_x = thresholds.threshold_x;
    (*split_map)->split_y = thresholds.threshold_y;

    const int ret = fill_cell_lists_3d_1d_split_map((*split_map)->map_lower,  //
                                                    (*split_map)->map_upper,  //
                                                    (*split_map)->split_x,    //
                                                    boxes->min_x,             //
                                                    boxes->min_y,             //
                                                    boxes->min_z,             //
                                                    boxes->max_x,             //
                                                    boxes->max_y,             //
                                                    boxes->max_z,             //
                                                    num_boxes,                //
                                                    stats.min_x,              //
                                                    stats.max_x,              //
                                                    stats.min_y,              //
                                                    stats.max_y,              //
                                                    stats.min_z,              //
                                                    stats.max_z);             //

    if (ret != EXIT_SUCCESS) {
        free_cell_list_3d_1d_map((*split_map)->map_lower);
        free_cell_list_3d_1d_map((*split_map)->map_upper);
        free(*split_map);
        *split_map = NULL;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

//////////////////////////////////////////////////
// intersect_triangle_xy
//////////////////////////////////////////////////
bool                                       //
intersect_triangle_xy(const real_t v0[3],  //
                      const real_t v1[3],  //
                      const real_t v2[3],  //
                      const real_t x,      //
                      const real_t y) {    //
    // Signed area of each sub-triangle formed by an edge and the query point.
    // A point is inside iff all three have the same sign (all CW or all CCW).
    const real_t d0 = (v1[0] - v0[0]) * (y - v0[1]) - (v1[1] - v0[1]) * (x - v0[0]);
    const real_t d1 = (v2[0] - v1[0]) * (y - v1[1]) - (v2[1] - v1[1]) * (x - v1[0]);
    const real_t d2 = (v0[0] - v2[0]) * (y - v2[1]) - (v0[1] - v2[1]) * (x - v2[0]);

    const bool has_neg = (d0 < 0) | (d1 < 0) | (d2 < 0);
    const bool has_pos = (d0 > 0) | (d1 > 0) | (d2 > 0);
    return !(has_neg & has_pos);  // inside iff all same sign (or zero)
}

////////////////////////////////////////////////////
// intersection_point_triangle_xy
////////////////////////////////////////////////////
void                                                  //
intersection_point_triangle_xy(const real_t v0[3],    //
                               const real_t v1[3],    //
                               const real_t v2[3],    //
                               const real_t x,        //
                               const real_t y,        //
                               real_t      *out_z) {  //

    // Raw edge vectors (no need to normalise — plane equation is scale-invariant)
    const real_t e1x = v1[0] - v0[0], e1y = v1[1] - v0[1], e1z = v1[2] - v0[2];
    const real_t e2x = v2[0] - v0[0], e2y = v2[1] - v0[1], e2z = v2[2] - v0[2];

    // Cross product n = e1 × e2  (unnormalised plane normal)
    const real_t nx = e1y * e2z - e1z * e2y;
    const real_t ny = e1z * e2x - e1x * e2z;
    const real_t nz = e1x * e2y - e1y * e2x;

    // Degenerate or axis-aligned triangle: nz ≈ 0 means no unique z solution
    if (nz * nz < 1e-24) {
        *out_z = v0[2];
        return;
    }

    // Plane equation:  n · (P − v0) = 0  →  solve for z
    *out_z = v0[2] - (nx * (x - v0[0]) + ny * (y - v0[1])) / nz;
}

//////////////////////////////////////////////////
// query_cell_list_3d_1d_map_mesh_given_xy_tri3_v
//////////////////////////////////////////////////
int                                                                                                  //
query_cell_list_3d_1d_map_mesh_given_xy_tri3_v(const cell_list_3d_1d_map_t *map,                     //
                                               const boxes_t               *boxes,                   //
                                               const mesh_tri3_geom_t      *mesh_geom,               //
                                               const real_t                 x,                       //
                                               const real_t                 y,                       //
                                               const int                    start_index_tri3_array,  //
                                               int                         *size_t3_array,           //
                                               real_t                     **tri3_intersect_z) {      //

    const int ix_tmp = coord_to_grid_index(x, map->min_x, map->delta_x);
    const int ix     = (ix_tmp < 0) ? 0 : (ix_tmp >= map->num_cells_x) ? map->num_cells_x - 1 : ix_tmp;

    const int cell_index = ix;

    const int start_index = map->cell_ptr[cell_index];
    const int end_index   = map->cell_ptr[cell_index + 1];

    const int num_boxes_local = end_index - start_index;

    int triangles_found = 0;

    if (num_boxes_local > 0) {
        int lower_bound_index = lower_bound_float(&map->upper_bounds_y[start_index], (size_t)num_boxes_local, y);

        const int start_index_up = (lower_bound_index > 1) ? start_index + lower_bound_index - 2 : start_index;
        const int size_up        = (lower_bound_index > 1) ? num_boxes_local - (lower_bound_index - 2) : num_boxes_local;
        const int offset_up      = start_index_up - start_index;

        // int upper_bound_index = upper_bound_float(
        // &map->lower_bounds_y[start_index],
        // (size_t)num_boxes_local,
        // y);

        int upper_bound_index = upper_bound_float(     //
                &map->lower_bounds_y[start_index_up],  //
                size_up,                               //
                y);                                    //

        // Adjust upper_bound_index back to be relative to start_index
        upper_bound_index += offset_up;

        lower_bound_index =
                lower_bound_index < 0 ? 0 : (lower_bound_index > num_boxes_local ? num_boxes_local : lower_bound_index);
        upper_bound_index =
                upper_bound_index < 0 ? 0 : (upper_bound_index > num_boxes_local ? num_boxes_local : upper_bound_index);

        // const int *cell_dict_local = &map->cell_dict[start_index];

        for (int i = lower_bound_index; i < upper_bound_index; i++) {
            const int box_index = map->cell_dict[start_index + i];

            // Capture the triangle vertices for the current box
            // mesh_geom->ref_mesh->points

            idx_t ev[3];
            for (int v = 0; v < 3; ++v) {
                ev[v] = mesh_geom->ref_mesh->elements[v][box_index];
            }

            // Read the coordinates of the vertices of the tetrahedron
            const real_t x0 = mesh_geom->ref_mesh->points[0][ev[0]];
            const real_t x1 = mesh_geom->ref_mesh->points[0][ev[1]];
            const real_t x2 = mesh_geom->ref_mesh->points[0][ev[2]];

            const real_t y0 = mesh_geom->ref_mesh->points[1][ev[0]];
            const real_t y1 = mesh_geom->ref_mesh->points[1][ev[1]];
            const real_t y2 = mesh_geom->ref_mesh->points[1][ev[2]];

            const real_t z0 = mesh_geom->ref_mesh->points[2][ev[0]];
            const real_t z1 = mesh_geom->ref_mesh->points[2][ev[1]];
            const real_t z2 = mesh_geom->ref_mesh->points[2][ev[2]];

            // Check if the vertical ray from (x, y) intersects the triangle
            // defined by the vertices of the current box.
            // If it does, compute the intersection z coordinate and store it in tri3_intersect_z[triangles_found], and store
            // the triangle index If the array tri3_intersect_z is not large enough to hold all found triangles, reallocate it
            // with a larger size (e.g., double the current size).

            if (intersect_triangle_xy((real_t[3]){x0, y0, z0},  //
                                      (real_t[3]){x1, y1, z1},  //
                                      (real_t[3]){x2, y2, z2},  //
                                      x,
                                      y)) {
                if (triangles_found + start_index_tri3_array >= *size_t3_array) {
                    *size_t3_array *= 2;
                    real_t *tmp = realloc(*tri3_intersect_z, (*size_t3_array) * sizeof(real_t));
                    if (!tmp) return EXIT_FAILURE;
                    *tri3_intersect_z = tmp;
                }

                intersection_point_triangle_xy((real_t[3]){x0, y0, z0},  //
                                               (real_t[3]){x1, y1, z1},  //
                                               (real_t[3]){x2, y2, z2},  //
                                               x,
                                               y,
                                               &(*tri3_intersect_z)[triangles_found + start_index_tri3_array]);

                triangles_found++;
            }
        }
    }
    return triangles_found;
}

////////////////////////////////////////////////////////////////
// query_cell_list_3d_1d_split_map_mesh_given_xy_tri3_v
////////////////////////////////////////////////////////////////
int                                                                                                          //
query_cell_list_3d_1d_split_map_mesh_given_xy_tri3_v(const cell_list_split_3d_1d_map_t *map,                 //
                                                     const boxes_t                     *boxes,               //
                                                     const mesh_tri3_geom_t            *mesh_geom,           //
                                                     const real_t                       x,                   //
                                                     const real_t                       y,                   //
                                                     int                               *size_t3_array,       //
                                                     real_t                           **tri3_intersect_z) {  //

    if (map == NULL || boxes == NULL || mesh_geom == NULL) {
        return -1;  // Invalid pointer
    }

    const int num_found_lower = query_cell_list_3d_1d_map_mesh_given_xy_tri3_v(map->map_lower,     //
                                                                               boxes,              //
                                                                               mesh_geom,          //
                                                                               x,                  //
                                                                               y,                  //
                                                                               0,                  // start_index_tri3_array
                                                                               size_t3_array,      //
                                                                               tri3_intersect_z);  //

    const int num_found_upper = query_cell_list_3d_1d_map_mesh_given_xy_tri3_v(map->map_upper,     //
                                                                               boxes,              //
                                                                               mesh_geom,          //
                                                                               x,                  //
                                                                               y,                  //
                                                                               num_found_lower,    // start_index_tri3_array
                                                                               size_t3_array,      //
                                                                               tri3_intersect_z);  //
    return num_found_lower + num_found_upper;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// check_intervals
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
static void check_intervals(const real_t A[], const int n, const real_t I[], const int i_size, real_t out[]) {
    int i = 0;
    int k = 0;

    while (i < n && k < i_size) {
        const real_t lo = I[k];
        const real_t hi = I[k + 1];

        while (i < n && A[i] < lo) out[i++] = 0.0;
        while (i < n && A[i] <= hi) out[i++] = 1.0;

        k += 2;
    }

    while (i < n) out[i++] = 0.0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// raster_cell_list_3d_1d_split_map_mesh_given_xyz_tri3_v
/////////////////////////////////////////////////////////////////////////////////////////////////////////
int                                                                                                      //
raster_cell_list_3d_1d_split_map_mesh_given_xyz_tri3_v(const cell_list_split_3d_1d_map_t *map,           //
                                                       const boxes_t                     *boxes,         //
                                                       const mesh_tri3_geom_t            *mesh_geom,     //
                                                       const real_t                       x,             //
                                                       const real_t                       y,             //
                                                       const real_t                      *z_coords,      //
                                                       const int                          num_z_coords,  //
                                                       real_t                            *out_z) {       //

    if (map == NULL || boxes == NULL || mesh_geom == NULL || out_z == NULL) {
        return -1;  // Invalid pointer
    }

    int     size_t3_array    = 64;  // Initial size for intersecting triangles array
    real_t *tri3_intersect_z = malloc(size_t3_array * sizeof(real_t));
    if (!tri3_intersect_z) {
        return -1;  // Memory allocation failure
    }

    const int num_triangles = query_cell_list_3d_1d_split_map_mesh_given_xy_tri3_v(map,                 //
                                                                                   boxes,               //
                                                                                   mesh_geom,           //
                                                                                   x,                   //
                                                                                   y,                   //
                                                                                   &size_t3_array,      //
                                                                                   &tri3_intersect_z);  //

    if (num_triangles <= 0) {
        free(tri3_intersect_z);
        return -1;  // No intersecting triangles found
    }

    if (num_triangles % 2 != 0) {
        free(tri3_intersect_z);
        return -1;  // Odd number of intersections suggests a problem (e.g., non-manifold geometry)
    }

    // sort the intersecting triangles' z values
    qsort(tri3_intersect_z, num_triangles, sizeof(real_t), cmp_real_t);

    check_intervals(z_coords, num_z_coords, tri3_intersect_z, num_triangles, out_z);

    free(tri3_intersect_z);
    return 0;
}
