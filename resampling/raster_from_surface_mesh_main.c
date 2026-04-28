#include "resample_adjoint_main.h"

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "cell_list_bench.h"
#include "cell_tet2box.h"
#include "resampling_utils.h"
#include "sfem_base.h"
#include "sfem_resample_field.h"

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// print_mesh_info_and_coordinate_bounds
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
static int print_mesh_info_and_coordinate_bounds(MPI_Comm comm, const int mpi_rank, const mesh_t* const mesh,
                                                 const char* const folder) {
    const long local_nelements = (long)mesh->nelements;
    const long local_nnodes    = (long)mesh->nnodes;
    const long local_nowned    = (long)mesh->n_owned_nodes;

    long global_nelements = 0;
    long global_nnodes    = 0;
    long global_nowned    = 0;

    MPI_Reduce(&local_nelements, &global_nelements, 1, MPI_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_nnodes, &global_nnodes, 1, MPI_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_nowned, &global_nowned, 1, MPI_LONG, MPI_SUM, 0, comm);

    geom_t local_min[3]  = {INFINITY, INFINITY, INFINITY};
    geom_t local_max[3]  = {-INFINITY, -INFINITY, -INFINITY};
    geom_t global_min[3] = {INFINITY, INFINITY, INFINITY};
    geom_t global_max[3] = {-INFINITY, -INFINITY, -INFINITY};

    const int scan_dim = mesh->spatial_dim < 3 ? mesh->spatial_dim : 3;

    for (int d = 0; d < scan_dim; d++) {
        for (ptrdiff_t i = 0; i < mesh->nnodes; i++) {
            const geom_t x = mesh->points[d][i];
            local_min[d]   = x < local_min[d] ? x : local_min[d];
            local_max[d]   = x > local_max[d] ? x : local_max[d];
        }
    }

    MPI_Allreduce(local_min, global_min, scan_dim, SFEM_MPI_GEOM_T, MPI_MIN, comm);
    MPI_Allreduce(local_max, global_max, scan_dim, SFEM_MPI_GEOM_T, MPI_MAX, comm);

    if (mpi_rank == 0) {
        printf("Mesh info:\n");
        printf("--------------------------------------------\n");
        printf("  name                     : %s\n", folder);
        printf("  element_type             : %s (%d)\n", type_to_string(mesh->element_type), mesh->element_type);
        printf("  spatial_dim              : %d\n", mesh->spatial_dim);
        printf("  local nelements          : %ld\n", local_nelements);
        printf("  local nnodes             : %ld\n", local_nnodes);
        printf("  local n_owned_nodes      : %ld\n", local_nowned);
        printf("  global nelements (sum)   : %ld\n", global_nelements);
        printf("  global nnodes (sum)      : %ld\n", global_nnodes);
        printf("  global n_owned_nodes     : %ld\n", global_nowned);

        static const char* axis_name[3] = {"x", "y", "z"};
        for (int d = 0; d < scan_dim; d++) {
            printf("  local %s min/max         : %g / %g\n", axis_name[d], (double)local_min[d], (double)local_max[d]);
            printf("  global %s min/max        : %g / %g\n", axis_name[d], (double)global_min[d], (double)global_max[d]);
        }
    }  // END if (mpi_rank == 0)

    RETURN_FROM_FUNCTION(0);
}  // END Function: print_mesh_info_and_coordinate_bounds

int main_raster_from_surface_mesh(int argc, char* argv[]) {  //

    PRINT_CURRENT_FUNCTION;

    printf("========================================\n");
    printf("Starting sfem_raster_from_surface_mesh test\n");
    printf("========================================\n\n");

    printf("<sizeof_real_t> %zu\n", sizeof(real_t));

    sfem_resample_field_info info;

    info.element_type = TET4;

    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    const function_XYZ_t mesh_fun_XYZ = mesh_fun_trig_pos;

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

    info.quad_nodes_cnt = TET_QUAD_NQP;

    printf("Reading mesh from folder: %s\n", folder);

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        fprintf(stderr, "Error: mesh_read failed %s:%d\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    if (print_mesh_info_and_coordinate_bounds(comm, mpi_rank, &mesh, folder)) {
        fprintf(stderr, "Error: print_mesh_info_and_coordinate_bounds failed %s:%d\n", __FILE__, __LINE__);
        return EXIT_FAILURE;
    }

    int SFEM_READ_FP32 = 1;
    SFEM_READ_ENV(SFEM_READ_FP32, atoi);

    printf("SFEM_READ_FP32 = %d, %s:%d\n", SFEM_READ_FP32, __FILE__, __LINE__);

    real_t*   field = NULL;
    ptrdiff_t nlocal[3];

    ptrdiff_t n_zyx = 0;
    {
        nlocal[0] = nglobal[0];
        nlocal[1] = nglobal[1];
        nlocal[2] = nglobal[2];

        n_zyx = nlocal[0] * nlocal[1] * nlocal[2];

        printf("nlocal: %ld %ld %ld, %s:%d\n", nlocal[0], nlocal[1], nlocal[2], __FILE__, __LINE__);

        field = calloc(n_zyx, sizeof(real_t));
    }

    // X is contiguous
    ptrdiff_t stride[3] = {1, nlocal[0], nlocal[0] * nlocal[1]};

#if SFEM_LOG_LEVEL >= 5
    printf("stride: %ld %ld %ld, %s:%d\n", stride[0], stride[1], stride[2], __FILE__, __LINE__);
#endif

    if (mpi_size > 1) {
        // TODO: implement field_view for surface meshes
        // real_t* pfield;
        // field_view(comm,
        //            mesh.nnodes,
        //            mesh.points[2],
        //            nlocal,
        //            nglobal,
        //            stride,
        //            origin,
        //            delta,
        //            field,
        //            &pfield,
        //            &nlocal[2],
        //            &origin[2]);

        // n_zyx = nlocal[0] * nlocal[1] * nlocal[2];  // Update n_zyx after field_view
        // printf("nlocal: %ld %ld %ld, %s:%d\n", nlocal[0], nlocal[1], nlocal[2], __FILE__, __LINE__);

        // free(field);
        // field = pfield;
        goto finalize;
    }

    const double raster_tick_start = MPI_Wtime();

    const double raster_tick_end = MPI_Wtime();
    printf("Rasterization time: %.6f seconds\n", raster_tick_end - raster_tick_start);

finalize:

    if (field) free(field);

    mesh_destroy(&mesh);

    MPI_Finalize();
    RETURN_FROM_FUNCTION(0);
}