#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cell_list_resampling_gpu.h"

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_adjoint_cell_quad_gpu_init_cpu_data
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
static void tet4_resample_field_adjoint_cell_quad_gpu_init_cpu_data(
        tet4_resample_field_adjoint_cell_quad_gpu_cpu_data_t *cpu_data) {
    if (cpu_data == NULL) {
        return;
    }  // END if (cpu_data == NULL)

    cpu_data->bounding_boxes = NULL;
    cpu_data->geom           = NULL;
    cpu_data->split_map      = NULL;
    memset(&cpu_data->histograms, 0, sizeof(cpu_data->histograms));
}  // END Function: tet4_resample_field_adjoint_cell_quad_gpu_init_cpu_data

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_adjoint_cell_quad_gpu_destroy_cpu_data
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
static void tet4_resample_field_adjoint_cell_quad_gpu_destroy_cpu_data(
        tet4_resample_field_adjoint_cell_quad_gpu_cpu_data_t *cpu_data) {
    if (cpu_data == NULL) {
        return;
    }  // END if (cpu_data == NULL)

    if (cpu_data->split_map != NULL) {
        free_cell_list_split_3d_2d_map(cpu_data->split_map);
        cpu_data->split_map = NULL;
    }  // END if (cpu_data->split_map != NULL)

    if (cpu_data->geom != NULL) {
        mesh_tet_geometry_free(cpu_data->geom);
        cpu_data->geom = NULL;
    }  // END if (cpu_data->geom != NULL)

    if (cpu_data->bounding_boxes != NULL) {
        free_boxes_t(cpu_data->bounding_boxes);
        cpu_data->bounding_boxes = NULL;
    }  // END if (cpu_data->bounding_boxes != NULL)

    free_side_length_histograms(&cpu_data->histograms);
    memset(&cpu_data->histograms, 0, sizeof(cpu_data->histograms));
}  // END Function: tet4_resample_field_adjoint_cell_quad_gpu_destroy_cpu_data

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_adjoint_cell_quad_gpu_build_cpu_data
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
static int                                                                                                                     //
tet4_resample_field_adjoint_cell_quad_gpu_build_cpu_data(const ptrdiff_t                                       start_element,  //
                                                         const ptrdiff_t                                       end_element,    //
                                                         const mesh_t                                         *mesh,           //
                                                         const ptrdiff_t *const SFEM_RESTRICT                  n,              //
                                                         const geom_t *const SFEM_RESTRICT                     origin,         //
                                                         const geom_t *const SFEM_RESTRICT                     delta,          //
                                                         tet4_resample_field_adjoint_cell_quad_gpu_cpu_data_t *cpu_data) {     //
    int ret = 0;

    (void)start_element;

    if (mesh == NULL || n == NULL || origin == NULL || delta == NULL || cpu_data == NULL) {
        fprintf(stderr, "Error: Invalid input to tet4_resample_field_adjoint_cell_quad_gpu_build_cpu_data\n");
        ret = EXIT_FAILURE;
        goto exit;
    }  // END if (mesh == NULL || n == NULL || origin == NULL || delta == NULL || cpu_data == NULL)

    const int fb_error = make_mesh_tets_boxes(0,
                                              end_element,
                                              mesh->nnodes,
                                              (const idx_t **)mesh->elements,
                                              (const geom_t **)mesh->points,
                                              &cpu_data->bounding_boxes);

    if (fb_error != 0 || cpu_data->bounding_boxes == NULL) {
        fprintf(stderr, "Error: make_mesh_tets_boxes failed %s:%d\n", __FILE__, __LINE__);
        ret = EXIT_FAILURE;
        goto exit;
    }  // END if (fb_error != 0 || cpu_data->bounding_boxes == NULL)

    bounding_box_statistics_t stats = calculate_bounding_box_statistics(cpu_data->bounding_boxes);
    print_bounding_box_statistics(&stats);

    cpu_data->histograms = calculate_side_length_histograms(cpu_data->bounding_boxes, &stats, 50);
    print_side_length_histograms(&cpu_data->histograms);

    side_length_cdf_thresholds_t thresholds = calculate_cdf_thresholds(&cpu_data->histograms, 0.96, 0.96, 0.96);

    cpu_data->geom = mesh_tet_geometry_alloc(mesh);
    if (cpu_data->geom == NULL) {
        fprintf(stderr, "Error: mesh_tet_geometry_alloc failed %s:%d\n", __FILE__, __LINE__);
        ret = EXIT_FAILURE;
        goto exit;
    }  // END if (cpu_data->geom == NULL)

    mesh_tet_geometry_compute_inv_Jacobian(cpu_data->geom);

    const real_t min_grid_x = origin[0];
    const real_t min_grid_y = origin[1];
    const real_t min_grid_z = origin[2];

    const real_t max_grid_x = origin[0] + delta[0] * n[0];
    const real_t max_grid_y = origin[1] + delta[1] * n[1];
    const real_t max_grid_z = origin[2] + delta[2] * n[2];

    ret = build_cell_list_3d_2d_split_map(&cpu_data->split_map,
                                          thresholds.threshold_x,
                                          thresholds.threshold_y,
                                          cpu_data->bounding_boxes->min_x,
                                          cpu_data->bounding_boxes->min_y,
                                          cpu_data->bounding_boxes->min_z,
                                          cpu_data->bounding_boxes->max_x,
                                          cpu_data->bounding_boxes->max_y,
                                          cpu_data->bounding_boxes->max_z,
                                          cpu_data->bounding_boxes->num_boxes,
                                          min_grid_x,
                                          max_grid_x,
                                          min_grid_y,
                                          max_grid_y,
                                          min_grid_z,
                                          max_grid_z);

    if (ret != 0 || cpu_data->split_map == NULL) {
        fprintf(stderr, "Error: build_cell_list_3d_2d_split_map failed %s:%d\n", __FILE__, __LINE__);
        ret = EXIT_FAILURE;
        goto exit;
    }  // END if (ret != 0 || cpu_data->split_map == NULL)

exit:
    RETURN_FROM_FUNCTION(ret);
}  // END Function: tet4_resample_field_adjoint_cell_quad_gpu_build_cpu_data

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_adjoint_cell_quad_gpu
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
int tet4_resample_field_adjoint_cell_quad_gpu(const ptrdiff_t                      start_element,        //
                                              const ptrdiff_t                      end_element,          //
                                              const mesh_t                        *mesh,                 //
                                              const ptrdiff_t *const SFEM_RESTRICT n,                    //
                                              const ptrdiff_t *const SFEM_RESTRICT stride,               //
                                              const geom_t *const SFEM_RESTRICT    origin,               //
                                              const geom_t *const SFEM_RESTRICT    delta,                //
                                              const real_t *const SFEM_RESTRICT    weighted_field,       //
                                              const mini_tet_parameters_t          mini_tet_parameters,  //
                                              real_t *const SFEM_RESTRICT          data) {                        //
    int ret = 0;

    (void)mini_tet_parameters;

    PRINT_CURRENT_FUNCTION;

    const double tick = MPI_Wtime();

    tet4_resample_field_adjoint_cell_quad_gpu_cpu_data_t cpu_data;
    tet4_resample_field_adjoint_cell_quad_gpu_init_cpu_data(&cpu_data);

    ret = tet4_resample_field_adjoint_cell_quad_gpu_build_cpu_data(start_element, end_element, mesh, n, origin, delta, &cpu_data);
    if (ret != 0) {
        goto cleanup;
    }  // END if (ret != 0)

    const double tick_transfer = MPI_Wtime();

    ret = tet4_resample_field_adjoint_cell_quad_gpu_launch(&cpu_data, mesh, n, stride, origin, delta, weighted_field, data);

    if (ret != 0) {
        goto cleanup;
    }  // END if (ret != 0)

    MPI_Barrier(MPI_COMM_WORLD);
    const double tock = MPI_Wtime();
    printf("Time taken for tet4_resample_field_adjoint_cell_quad: %f seconds\n", tock - tick);
    printf("Time taken for data transfer: %f seconds\n", tock - tick_transfer);
    printf("Time taken for cell list construction and setup: %f seconds\n", tick_transfer - tick);

cleanup:
    tet4_resample_field_adjoint_cell_quad_gpu_destroy_cpu_data(&cpu_data);

    RETURN_FROM_FUNCTION(ret);
}  // END Function: tet4_resample_field_adjoint_cell_quad_gpu