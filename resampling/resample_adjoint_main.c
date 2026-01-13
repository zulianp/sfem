

#include "resample_adjoint_main.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "resampling_utils.h"
#include "sfem_base.h"
#include "sfem_resample_field.h"

#define RED_TEXT "\x1b[31m"
#define GREEN_TEXT "\x1b[32m"
#define RESET_TEXT "\x1b[0m"

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// print_command_line_arguments
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
void print_command_line_arguments(int argc, char* argv[], int mpi_rank) {
    if (mpi_rank == 0) {
        printf("argc: %d\n", argc);
        printf("argv: \n");
        for (int i = 0; i < argc; i++) {
            printf(" %s", argv[i]);
        }
        printf("\n");
    }
}  // END Function: print_command_line_arguments

int main_adjoint(int argc, char* argv[]) {
    PRINT_CURRENT_FUNCTION;

    printf("========================================\n");
    printf("Starting sfem_resample_field_adjoint test\n");
    printf("========================================\n\n");

    printf("<sizeof_real_t> %zu\n", sizeof(real_t));

    sfem_resample_field_info info;

    info.element_type = TET10;

    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    const function_XYZ_t mesh_fun_XYZ = mesh_fun_ones;

    char out_base_directory[2048];

    if (getenv("SFEM_OUT_BASE_DIRECTORY") != NULL) {
        snprintf(out_base_directory, 2048, "%s", getenv("SFEM_OUT_BASE_DIRECTORY"));
    } else {
        snprintf(out_base_directory, 2048, "/tmp/");
    }

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

    int SFEM_ADJOINT = 0;
    SFEM_READ_ENV(SFEM_ADJOINT, atoi);

    double tick = MPI_Wtime();

    ptrdiff_t nglobal[3] = {atol(argv[1]), atol(argv[2]), atol(argv[3])};
    geom_t    origin[3]  = {atof(argv[4]), atof(argv[5]), atof(argv[6])};
    geom_t    delta[3]   = {atof(argv[7]), atof(argv[8]), atof(argv[9])};

    const char* data_path   = argv[10];
    const char* folder      = argv[11];
    const char* output_path = argv[12];

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

    // ptrdiff_t n = nglobal[0] * nglobal[1] * nglobal[2];
    real_t*       field         = NULL;
    unsigned int* field_cnt     = NULL;  // TESTING used to count the number of times a field is updated
    real_t*       field_alpha   = NULL;  // TESTING used to store the alpha field
    real_t*       field_volume  = NULL;  // TESTING used to store the volume field
    real_t*       field_fun_XYZ = NULL;  // TESTING used to store the analytical function

    ptrdiff_t nlocal[3];

    int SFEM_READ_FP32 = 1;
    SFEM_READ_ENV(SFEM_READ_FP32, atoi);
}
