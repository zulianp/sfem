#include "cell_list_3d_1d_map_sur_mesh.h"

////////////////////////////////////////////////
// coord_to_grid_index
////////////////////////////////////////////////
static inline int coord_to_grid_index(real_t coord, real_t origin, real_t delta) { return (int)((coord - origin) / delta); }

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
            // If it does, compute the intersection z coordinate and store it in tri3_intersect_z[triangles_found], and store the
            // triangle index If the array tri3_intersect_z is not large enough to hold all found triangles, reallocate it with a
            // larger size (e.g., double the current size).

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

static int cmp_real_t(const void *a, const void *b) {
    real_t diff = *(const real_t *)a - *(const real_t *)b;
    return (diff > 0) - (diff < 0);  // returns 1 if a > b, -1 if a < b, 0 if equal
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
