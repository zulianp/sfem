#include "resample_adjoint_main.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "resampling_utils.h"
#include "sfem_base.h"
#include "sfem_resample_field.h"

int main_test_ccell(int argc, char* argv[]) {  //

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

    RETURN_FROM_FUNCTION(1);
}