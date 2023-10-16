#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sortreduce.h"

#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    const char *help = "usage: %s <input.idx_t.raw> <output.idx_t.raw>";

    if (argc < 3) {
        fprintf(stderr, help, argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_path = argv[2];
    double tick = MPI_Wtime();

    MPI_Datatype mpi_idx_t = SFEM_MPI_IDX_T;

    idx_t *values;
    ptrdiff_t nlocal, nnodes;
    array_create_from_file(comm, argv[1], mpi_idx_t, (void **)&values, &nlocal, &nnodes);
    ptrdiff_t new_nnodes = sortreduce(values, nnodes);
    array_write(comm, output_path, mpi_idx_t, (void *)values, new_nnodes, new_nnodes);
    free(values);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
