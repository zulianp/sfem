

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
}