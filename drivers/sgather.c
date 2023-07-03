#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/matrixio_crs.h"
#include "../matrix.io/utils.h"

typedef int idx_t;
#define MPI_IDX_T MPI_INT

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

    const char *help = "usage: %s <gather_idx.raw> <n_bytes_x_entry> <input_array.raw> [out=sgather_out.raw]\n";

    if (argc < 4) {
        fprintf(stderr, help, argv[0]);
        return EXIT_FAILURE;
    }

    const char *gather_idx_path = argv[1];
    int n_bytes_x_entry = atoi(argv[2]);
    const char *input_array_path = argv[3];
    const char *output_path = "./sgather_out.raw";

    if (argc > 4) {
        output_path = argv[4];
    }

    if (strcmp(output_path, input_array_path) == 0 || strcmp(output_path, gather_idx_path) == 0) {
        fprintf(stderr, "Input and output are the same! Quitting!\n");
        fprintf(stderr, help, argv[0]);
        return EXIT_FAILURE;
    }

    double tick = MPI_Wtime();

    MPI_Datatype values_mpi_t = MPI_CHAR;
    char *values;
    ptrdiff_t nlocal_, n_bytes;
    array_create_from_file(comm, input_array_path, values_mpi_t, (void **)&values, &nlocal_, &n_bytes);

    ptrdiff_t n_values = n_bytes / n_bytes_x_entry;
    if ((n_values * n_bytes_x_entry) != n_bytes) {
        fprintf(stderr, "Bad input! %ld != %d * %ld\n", (long)n_bytes, n_bytes_x_entry, n_values);
        return EXIT_FAILURE;
    }

    ptrdiff_t nlocal_gather, n_gather;
    idx_t *gather_idx;
    array_create_from_file(comm, gather_idx_path, MPI_IDX_T, (void **)&gather_idx, &nlocal_gather, &n_gather);

    // assert(n_values >= n_gather);

    char *selection = (char *)malloc(n_gather * n_bytes_x_entry);

    for (ptrdiff_t i = 0; i < n_gather; ++i) {
        assert(gather_idx[i] < n_values);
        ptrdiff_t offset = i * n_bytes_x_entry;
        ptrdiff_t gather_offset = gather_idx[i] * n_bytes_x_entry;
        memcpy(&selection[offset], &values[gather_offset], n_bytes_x_entry);
    }

    array_write(
        comm, output_path, values_mpi_t, (void *)selection, n_gather * n_bytes_x_entry, n_gather * n_bytes_x_entry);

    free(values);
    free(gather_idx);
    free(selection);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
