#include "resample_adjoint_main.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "cell_list_bench.h"
#include "cell_tet2box.h"
#include "resampling_utils.h"
#include "sfem_base.h"
#include "sfem_resample_field.h"

int main_test_ccell(int argc, char* argv[]) {  //

    // return main_ccel_test(argc, argv);

    PRINT_CURRENT_FUNCTION;

    printf("========================================\n");
    printf("Starting sfem_resample_field_ccell test\n");
    printf("========================================\n\n");

    printf("<sizeof_real_t> %zu\n", sizeof(real_t));

    sfem_resample_field_info info;

    info.element_type = TET4;

    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    const function_XYZ_t mesh_fun_XYZ = mesh_fun_ones;

    char out_base_directory[2048];

    get_output_base_directory(out_base_directory, 2048);

#if SFEM_LOG_LEVEL >= 5
    printf("Using SFEM_OUT_BASE_DIRECTORY: %s\n", out_base_directory);
    print_mesh_function_name(mesh_fun_XYZ, mpi_rank);
#endif  // SFEM_LOG_LEVEL

    print_command_line_arguments(argc, argv, mpi_rank);

    if (argc < 13 && argc > 14) {
        fprintf(stderr, "Error: Invalid number of arguments\n\n");

        fprintf(stderr,
                "usage: %s <nx> <ny> <nz> <ox> <oy> <oz> <dx> <dy> <dz> "
                "<data.float32.raw> <folder> <output_path>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    int SFEM_INTERPOLATE = 1;
    SFEM_READ_ENV(SFEM_INTERPOLATE, atoi);

    double tick = MPI_Wtime();

    ptrdiff_t nglobal[3] = {atol(argv[1]), atol(argv[2]), atol(argv[3])};
    geom_t    origin[3]  = {atof(argv[4]), atof(argv[5]), atof(argv[6])};
    geom_t    delta[3]   = {atof(argv[7]), atof(argv[8]), atof(argv[9])};

    const char* data_path   = argv[10];
    const char* folder      = argv[11];
    const char* output_path = argv[12];

    if (parse_element_type_from_args(argc, argv, &info, mpi_rank)) return EXIT_FAILURE;

    info.use_accelerator = SFEM_ACCELERATOR_TYPE_CPU;

#ifdef SFEM_ENABLE_CUDA

    if (check_string_in_args(argc, (const char**)argv, "CUDA", mpi_rank == 0)) {
        info.use_accelerator = SFEM_ACCELERATOR_TYPE_CUDA;
        if (mpi_rank == 0) printf("info.use_accelerator = 1\n");

    } else if (check_string_in_args(argc, (const char**)argv, "CPU", mpi_rank == 0)) {
        info.use_accelerator = SFEM_ACCELERATOR_TYPE_CPU;
        if (mpi_rank == 0) printf("info.use_accelerator = 0\n");

    } else {
        fprintf(stderr, "Error: Invalid accelerator type\n\n");
        fprintf(stderr,
                "usage: %s <nx> <ny> <nz> <ox> <oy> <oz> <dx> <dy> <dz> "
                "<data.float32.raw> <folder> <output_path> <element_type> <accelerator_type>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

#endif

    if (info.element_type == TET4) {
        if (mpi_rank == 0) printf("info.element_type = TET4,    %s:%d\n", __FILE__, __LINE__);
    } else if (info.element_type == TET10) {
        if (mpi_rank == 0) printf("info.element_type = TET10,   %s:%d\n", __FILE__, __LINE__);
    } else {
        if (mpi_rank == 0) printf("info.element_type = UNKNOWN, %s:%d\n", __FILE__, __LINE__);
    }

    info.quad_nodes_cnt = TET_QUAD_NQP;

    printf("Reading mesh from folder: %s\n", folder);

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        fprintf(stderr, "Error: mesh_read failed %s:%d\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    ptrdiff_t nlocal[3];
    real_t*   field = NULL;
    ptrdiff_t n_zyx = 0;
    {
        nlocal[0] = nglobal[0];
        nlocal[1] = nglobal[1];
        nlocal[2] = nglobal[2];

        n_zyx = nlocal[0] * nlocal[1] * nlocal[2];

        printf("nlocal: %ld %ld %ld, %s:%d\n", nlocal[0], nlocal[1], nlocal[2], __FILE__, __LINE__);

        field = calloc(n_zyx, sizeof(real_t));
    }

    boxes_t*  bounding_boxes_ptr = NULL;                        //
    const int fb_error           =                              //
            make_mesh_tets_boxes(0,                             //
                                 mesh.nelements,                //
                                 mesh.nnodes,                   //
                                 (const idx_t**)mesh.elements,  //
                                 (const geom_t**)mesh.points,   //
                                 &bounding_boxes_ptr);          //

    bounding_box_statistics_t stats = calculate_bounding_box_statistics(bounding_boxes_ptr);
    print_bounding_box_statistics(&stats);

    side_length_histograms_t histograms =                         //
            calculate_side_length_histograms(bounding_boxes_ptr,  //
                                             &stats,              //
                                             50);                 //
    print_side_length_histograms(&histograms);

    char histogram_output_dir[2048];
    snprintf(histogram_output_dir, 2048, "%s/side_length_histograms", out_base_directory);
    printf("Writing side length histograms to output directory: %s\n", histogram_output_dir);
    write_side_length_histograms(&histograms, histogram_output_dir);

    free_side_length_histograms(&histograms);

    mesh_tet_geom_t* geom = mesh_tet_geometry_alloc(&mesh);

    mesh_tet_geometry_compute_inv_Jacobian(geom);

    if (fb_error) {
        fprintf(stderr, "Error: make_mesh_tets_boxes failed %s:%d\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    const real_t min_grid_x = origin[0];
    const real_t min_grid_y = origin[1];
    const real_t min_grid_z = origin[2];

    const real_t max_grid_x = origin[0] + delta[0] * nglobal[0];
    const real_t max_grid_y = origin[1] + delta[1] * nglobal[1];
    const real_t max_grid_z = origin[2] + delta[2] * nglobal[2];

    cell_list_3d_2d_map_t* cell_list_map = make_empty_cell_list_3d_2d_map();

    double cell_list_tick = MPI_Wtime();

    printf("Building cell list\n");
    printf("Grid bounding box: [%g, %g] x [%g, %g] x [%g, %g]\n",
           min_grid_x,
           max_grid_x,
           min_grid_y,
           max_grid_y,
           min_grid_z,
           max_grid_z);

    int cell_flag =                                                   //
            build_cell_list_3d_2d_map(cell_list_map,                  //
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

    double cell_list_tock = MPI_Wtime();

    if (mpi_rank == 0) {
        printf("Cell list build time: %g seconds\n", cell_list_tock - cell_list_tick);
    }  // END if (mpi_rank == 0)

    if (cell_flag) {
        fprintf(stderr, "Error: build_cell_list_3d_2d_map failed %s:%d\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    int64_t cell_map_bytes = cell_list_3d_2d_map_bytes(cell_list_map);
    if (mpi_rank == 0) {
        printf("Cell list memory usage: %g Mbytes\n", (double)(cell_map_bytes) / (1024 * 1024));
    }  // END if (mpi_rank == 0)

    // query_cell_list_test(cell_list_map, bounding_boxes_ptr, 1000);
    // query_cell_list_given_xy_test(cell_list_map, bounding_boxes_ptr, 500, 5000);

    query_cell_list_given_xy_bench(cell_list_map, bounding_boxes_ptr, 500, 100000);

    query_tet_cell_list_2d_3d_given_xy_bench(cell_list_map,
                                             bounding_boxes_ptr,
                                             geom,  //
                                             500,
                                             100000);

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // Finalize mesh and other data structures or arrays.
    // Free of destroy mesh should be done after resampling since we use mesh points in resampling

    mesh_tet_geometry_free(geom);
    geom = NULL;

    if (bounding_boxes_ptr) {
        free_boxes_t(bounding_boxes_ptr);
    }

    if (cell_list_map) {
        free_cell_list_3d_2d_map(cell_list_map);
    }

    free(field);
    mesh_destroy(&mesh);

    RETURN_FROM_FUNCTION(1);
}