#ifdef _OPENMP
#include <omp.h>
#endif

#include <time.h>

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
    {  // OpenMP parallel region
        const ptrdiff_t z_size = n[2];

        real_t *z_coords = malloc(z_size * sizeof(real_t));
        real_t *out_z    = malloc(z_size * sizeof(real_t));

        for (ptrdiff_t i = 0; i < z_size; i++) {
            z_coords[i] = origin[2] + (real_t)(i)*delta[2];
            out_z[i]    = 0.0;  // Initialize out_z to zero (or any default value as needed)
        }

        if (z_coords && out_z) {
            const ptrdiff_t step = 3;  // Step size for collapsing loops

            for (ptrdiff_t start_i = 0; start_i < step; start_i++) {
                for (ptrdiff_t start_j = 0; start_j < step; start_j++) {
                    ptrdiff_t loop_count_i = (x_size > start_i) ? ((x_size - start_i + step - 1) / step) : 0;
                    ptrdiff_t loop_count_j = (y_size > start_j) ? ((y_size - start_j + step - 1) / step) : 0;

#pragma omp for collapse(2) schedule(guided) nowait
                    for (ptrdiff_t i = 0; i < loop_count_i; i++) {
                        for (ptrdiff_t j = 0; j < loop_count_j; j++) {
                            ptrdiff_t i_grid = start_i + (i * step);
                            ptrdiff_t j_grid = start_j + (j * step);

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

                            if (flag == 0) {
                                for (ptrdiff_t iz = 0; iz < z_size; iz++) {
                                    hex_field[i_grid * stride[0] + j_grid * stride[1] + iz * stride[2]] = out_z[iz];
                                }  // END for (iz)
                            }  // END if (flag == 0)
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
    }  // END OpenMP parallel region

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

    struct timespec tick, tock;
    clock_gettime(CLOCK_MONOTONIC, &tick);

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

    mesh_tri3_geom_t *geom = mesh_tri3_geometry_alloc(mesh);

    cell_list_split_3d_1d_map_t *split_map = NULL;

    {
        struct timespec build_tick, build_tock;
        clock_gettime(CLOCK_MONOTONIC, &build_tick);
        build_cell_list_split_3d_1d_map_mesh(&split_map,           //
                                             mesh,                 //
                                             bounding_boxes_ptr);  //
        clock_gettime(CLOCK_MONOTONIC, &build_tock);
        const double build_elapsed_s =
                (double)(build_tock.tv_sec - build_tick.tv_sec) + (double)(build_tock.tv_nsec - build_tick.tv_nsec) / 1e9;
        printf("[build_cell_list_3d_1d_map] elapsed time: %.6f s\n", build_elapsed_s);
    }  // END block: clock build_cell_list_3d_1d_map

    free_side_length_histograms(&histograms);

    int64_t cell_list_mem_bytes = 0;
    if (split_map != NULL) {
        cell_list_mem_bytes += cell_list_3d_1d_map_bytes(split_map->map_lower);
        cell_list_mem_bytes += cell_list_3d_1d_map_bytes(split_map->map_upper);
    }  // END if (split_map != NULL)

    const double cell_list_MB = ((double)cell_list_mem_bytes) / (1024.0 * 1024.0);
    //
    printf("[build_cell_list_3d_1d_map] Cell list uses %ld bytes of memory (%.2f MB).\n", cell_list_mem_bytes, cell_list_MB);

    raster_to_hex_field_cell_split_par_tri3(split_map,           //
                                            bounding_boxes_ptr,  //
                                            geom,                //
                                            mesh,                //
                                            n,                   //
                                            stride,              //
                                            origin,              //
                                            delta,               //
                                            data);               //

finalize:

    clock_gettime(CLOCK_MONOTONIC, &tock);
    const double elapsed_s = (double)(tock.tv_sec - tick.tv_sec) + (double)(tock.tv_nsec - tick.tv_nsec) / 1e9;
    printf("[tri3_raster_mesh_cell_quad] elapsed time: %.6f s\n", elapsed_s);

    mesh_tri3_geometry_free(geom);
    geom = NULL;

    if (split_map != NULL) {
        free_cell_list_3d_1d_map(split_map->map_lower);
        free_cell_list_3d_1d_map(split_map->map_upper);
        free(split_map);
        split_map = NULL;
    }  // END if (split_map != NULL)

    free_boxes_t(bounding_boxes_ptr);
    bounding_boxes_ptr = NULL;

    RETURN_FROM_FUNCTION(0);
}
