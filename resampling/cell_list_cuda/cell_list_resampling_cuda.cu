#include "cell_list_3d_map_mesh.h"
#include "resample_field_adjoint_cell_cuda.cuh"
#include "sfem_resample_field_adjoint_cell.h"
#include "sfem_resample_field_adjoint_hex_quad.h"

/////////////////////////////////////////////////
// tet4_resample_field_adjoint_hex_quad_norm
/////////////////////////////////////////////////
int                                                                                                  //
tet4_resample_field_adjoint_cell_quad_gpu(const ptrdiff_t                      start_element,        // Mesh
                                          const ptrdiff_t                      end_element,          //
                                          const mesh_t                        *mesh,                 //
                                          const ptrdiff_t *const SFEM_RESTRICT n,                    // SDF
                                          const ptrdiff_t *const SFEM_RESTRICT stride,               //
                                          const geom_t *const SFEM_RESTRICT    origin,               //
                                          const geom_t *const SFEM_RESTRICT    delta,                //
                                          const real_t *const SFEM_RESTRICT    weighted_field,       // Input weighted field
                                          const mini_tet_parameters_t          mini_tet_parameters,  //
                                          real_t *const SFEM_RESTRICT          data) {                        // SDF: data (output)

    PRINT_CURRENT_FUNCTION;

    const double tick = MPI_Wtime();

    boxes_t  *bounding_boxes_ptr = NULL;                          //
    const int fb_error           =                                //
            make_mesh_tets_boxes(0,                               //
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

    mesh_tet_geom_t *geom = mesh_tet_geometry_alloc(mesh);

    mesh_tet_geometry_compute_inv_Jacobian(geom);

    const real_t min_grid_x = origin[0];
    const real_t min_grid_y = origin[1];
    const real_t min_grid_z = origin[2];

    const real_t max_grid_x = origin[0] + delta[0] * n[0];
    const real_t max_grid_y = origin[1] + delta[1] * n[1];
    const real_t max_grid_z = origin[2] + delta[2] * n[2];

    cell_list_split_3d_2d_map_t *split_map = NULL;

    build_cell_list_3d_2d_split_map(&split_map,                     //
                                    thresholds.threshold_x,         //
                                    thresholds.threshold_y,         //
                                    bounding_boxes_ptr->min_x,      //
                                    bounding_boxes_ptr->min_y,      //
                                    bounding_boxes_ptr->min_z,      //
                                    bounding_boxes_ptr->max_x,      //
                                    bounding_boxes_ptr->max_y,      //
                                    bounding_boxes_ptr->max_z,      //
                                    bounding_boxes_ptr->num_boxes,  //
                                    min_grid_x,                     //
                                    max_grid_x,                     //
                                    min_grid_y,                     //
                                    max_grid_y,                     //
                                    min_grid_z,                     //
                                    max_grid_z);                    //

    cudaStream_t stream = 0;  // Default stream
    cudaStreamCreate(&stream);

    boxes_t bounding_boxes_device_ptr =                        //
            copy_boxes_to_device(bounding_boxes_ptr, stream);  //

    cell_list_split_3d_2d_map_t split_map_device =                        //
            copy_cell_list_split_3d_2d_map_to_device(split_map, stream);  //

    mesh_tet_geom_device_t geom_device =                                  //
            copy_mesh_tet_geom_to_device(geom, mesh->nelements, stream);  //

    cudaStreamSynchronize(stream);  // Ensure all data is copied before launching the kernel

    const ptrdiff_t delta_i = 3;  // Process one grid point at a time in the i direction
    const ptrdiff_t delta_j = 3;  // Process one grid point at a time in the j direction

    const ptrdiff_t x_size = n[0];  // Number of grid points in the i direction
    const ptrdiff_t y_size = n[1];  // Number of grid points in the j direction

    for (ptrdiff_t start_i = 0; start_i < delta_i; start_i++) {
        for (ptrdiff_t start_j = 0; start_j < delta_j; start_j++) {
            // Launch the CUDA kernel in a way that there is no race condition on the output data.
            // This can be achieved by launching the kernel for each (start_i, start_j) pair separately,
            // ensuring that each thread block works on a distinct portion of the output data.

            ptrdiff_t loop_count_i = (x_size > start_i) ? ((x_size - start_i + 2) / 3) : 0;
            ptrdiff_t loop_count_j = (y_size > start_j) ? ((y_size - start_j + 2) / 3) : 0;

            // Replace the following function call with the actual GPU implementation
            // transfer_to_hex_field_cell_split_par_tet4(split_map,           //
            //                                           bounding_boxes_ptr,  //
            //                                           geom,                //
            //                                           mesh,                //
            //                                           n,                   //
            //                                           stride,              //
            //                                           origin,              //
            //                                           delta,               //
            //                                           weighted_field,      //
            //                                           data);               //
        }
    }

    ///////// FREE RESOURCES /////////
    // Free the device memory for the split map

    free_mesh_tet_geom_device(&geom_device, stream);

    free_cell_list_split_3d_2d_map_device(&split_map_device, stream);

    free_boxes_device(&bounding_boxes_device_ptr, stream);

    // Destroy the CUDA stream
    cudaStreamDestroy(stream);

    free_cell_list_split_3d_2d_map(split_map);
    split_map = NULL;

    mesh_tet_geometry_free(geom);
    geom = NULL;

    free_boxes_t(bounding_boxes_ptr);
    bounding_boxes_ptr = NULL;

    free_side_length_histograms(&histograms);

    MPI_Barrier(MPI_COMM_WORLD);
    const double tock = MPI_Wtime();
    printf("Time taken for tet4_resample_field_adjoint_cell_quad: %f seconds\n", tock - tick);

    RETURN_FROM_FUNCTION(0);
}  // END Function: tet4_resample_field_adjoint_cell_quad