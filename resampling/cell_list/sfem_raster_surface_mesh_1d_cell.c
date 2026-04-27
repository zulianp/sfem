#ifdef _OPENMP
#include <omp.h>
#endif

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
