#ifdef _OPENMP
#include <omp.h>
#endif

#include "cell_tet2box.h"
#include "sfem_raster_surface_mesh_1d_cell.h"

int                                                                                        //
raster_to_hex_field_cell_split_par_tri3(const cell_list_split_3d_1d_map_t   *split_map,    // Cell list split map data structure
                                        boxes_t                             *boxes,        // Boxes data structure
                                        const mesh_tri3_geom_t              *mesh_geom,    // Mesh geometry data structure
                                        const mesh_t *const SFEM_RESTRICT    mesh,         // Mesh: mesh_t struct
                                        const ptrdiff_t *const SFEM_RESTRICT n,            // SDF: n[3]
                                        const ptrdiff_t *const SFEM_RESTRICT stride,       // SDF: stride[3]
                                        const geom_t *const SFEM_RESTRICT    origin,       // SDF: origin[3]
                                        const geom_t *const SFEM_RESTRICT    delta,        // SDF: delta[3]
                                        real_t *const SFEM_RESTRICT          hex_field) {  //

    PRINT_CURRENT_FUNCTION;

#ifdef _OPENMP
    int num_procs = omp_get_num_procs();
    omp_set_dynamic(0);
    omp_set_num_threads(num_procs);
#endif

    const ptrdiff_t x_size = n[0];
    const ptrdiff_t y_size = n[1];

#pragma omp parallel
    {
        const ptrdiff_t z_size = n[2];

        real_t *z_coords = malloc(z_size * sizeof(real_t));
        real_t *out_z    = malloc(z_size * sizeof(real_t));

        for (ptrdiff_t i = 0; i < z_size; i++) {
            z_coords[i] = origin[2] + (real_t)(i)*delta[2];
            out_z[i]    = 0.0;  // Initialize out_z to zero (or any default value as needed)
        }

        if (z_coords && out_z) {
            for (ptrdiff_t start_i = 0; start_i < 3; start_i++) {
                for (ptrdiff_t start_j = 0; start_j < 3; start_j++) {
                    ptrdiff_t loop_count_i = (x_size > start_i) ? ((x_size - start_i + 2) / 3) : 0;
                    ptrdiff_t loop_count_j = (y_size > start_j) ? ((y_size - start_j + 2) / 3) : 0;

#pragma omp for collapse(2) schedule(guided) nowait
                    for (ptrdiff_t k = 0; k < loop_count_i; k++) {
                        for (ptrdiff_t m = 0; m < loop_count_j; m++) {
                            ptrdiff_t i_grid = start_i + (k * 3);
                            ptrdiff_t j_grid = start_j + (m * 3);

                            real_t grid_x = origin[0] + i_grid * delta[0];
                            real_t grid_y = origin[1] + j_grid * delta[1];

                            const int flag =                                                           //
                                    raster_cell_list_3d_1d_split_map_mesh_given_xyz_tri3_v(split_map,  //
                                                                                           boxes,      //
                                                                                           mesh_geom,  //
                                                                                           grid_x,     //
                                                                                           grid_y,     //
                                                                                           z_coords,   //
                                                                                           z_size,     //
                                                                                           out_z);     //
                        }
                    }
#pragma omp barrier
                }
            }
        }  // END if (z_coords && out_z)

        free(z_coords);
        z_coords = NULL;

        free(out_z);
        out_z = NULL;
    }

    RETURN_FROM_FUNCTION(0);
}

///////////////////////////////////////////////////////////////////////////
// tri3_raster_mesh_cell_quad
///////////////////////////////////////////////////////////////////////////
int                                                                              //
tri3_raster_mesh_cell_quad(const ptrdiff_t                      start_element,   // Mesh
                           const ptrdiff_t                      end_element,     //
                           const mesh_t                        *mesh,            //
                           const ptrdiff_t *const SFEM_RESTRICT n,               // SDF
                           const ptrdiff_t *const SFEM_RESTRICT stride,          //
                           const geom_t *const SFEM_RESTRICT    origin,          //
                           const geom_t *const SFEM_RESTRICT    delta,           //
                           const real_t *const SFEM_RESTRICT    weighted_field,  // Input weighted field
                           real_t *const SFEM_RESTRICT          data) {          // END Function: tri3_raster_mesh_cell_quad

    PRINT_CURRENT_FUNCTION;

    boxes_t *bounding_boxes_ptr = NULL;

    const int fb_error =                                          //
            make_mesh_tri3_boxes(start_element,                   //
                                 end_element,                     //
                                 mesh->nnodes,                    //
                                 (const idx_t **)mesh->elements,  //
                                 (const geom_t **)mesh->points,   //
                                 &bounding_boxes_ptr);            //

    bounding_box_statistics_t stats = calculate_bounding_box_statistics(bounding_boxes_ptr);
    print_bounding_box_statistics(&stats);

    side_length_histograms_t histograms =                         //
            calculate_side_length_histograms(bounding_boxes_ptr,  //
                                             &stats,              //
                                             50);                 //
    print_side_length_histograms(&histograms);

    side_length_cdf_thresholds_t thresholds =                         //
            calculate_cdf_thresholds(&histograms, 0.96, 0.96, 0.96);  //

    mesh_tri3_geom_t *geom = NULL;

finalize:

    // free_mesh_tri3_geometry(geom);

    free_boxes_t(bounding_boxes_ptr);
    bounding_boxes_ptr = NULL;

    RETURN_FROM_FUNCTION(0);
}
