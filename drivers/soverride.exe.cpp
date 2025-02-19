#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sfem_base.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

static void check_sizes(ptrdiff_t n_bytes, int n_bytes_x_entry) {
    ptrdiff_t n_values = n_bytes / n_bytes_x_entry;
    if ((n_values * n_bytes_x_entry) != n_bytes) {
        fprintf(stderr, "Bad input! %ld != %d * %ld\n", (long)n_bytes, n_bytes_x_entry, (long)n_values);
        MPI_Abort(MPI_COMM_WORLD, SFEM_FAILURE);
    }
}

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

    const char *help = "usage: %s <idx.raw> <n_bytes_x_entry> <override_values.raw> <input.raw> <output.raw>\n";

    if (argc != 6) {
        fprintf(stderr, help, argv[0]);
        return EXIT_FAILURE;
    }

    const char *path_idx = argv[1];
    int n_bytes_x_entry = atoi(argv[2]);
    const char *path_override_values = argv[3];
    const char *path_input = argv[4];
    const char *path_output = argv[5];

    double tick = MPI_Wtime();

    MPI_Datatype values_mpi_t = MPI_CHAR;

    char *override_values;
    ptrdiff_t _ignore_, n_bytes_overrride;
    array_create_from_file(
        comm, path_override_values, values_mpi_t, (void **)&override_values, &_ignore_, &n_bytes_overrride);

    const ptrdiff_t n_override_values = n_bytes_overrride / n_bytes_x_entry;

    check_sizes(n_bytes_overrride, n_bytes_x_entry);

    ptrdiff_t  n_override;
    idx_t *override_idx;
    array_create_from_file(comm, path_idx, SFEM_MPI_IDX_T, (void **)&override_idx, &_ignore_, &n_override);

    if (n_override != n_override_values) {
        SFEM_ERROR(
                "Inconsistent lenght of override values and idx! %ld != %ld\n",
                (long)n_override_values,
                (long)n_override);
    }

    char *input;
    ptrdiff_t n_bytes_input;
    array_create_from_file(comm, path_input, values_mpi_t, (void **)&input, &_ignore_, &n_bytes_input);

    check_sizes(n_bytes_input, n_bytes_x_entry);

    for (ptrdiff_t i = 0; i < n_override; ++i) {
        ptrdiff_t offset = i * n_bytes_x_entry;
        ptrdiff_t override_offset = override_idx[i] * n_bytes_x_entry;
        memcpy(&input[override_offset], &override_values[offset], n_bytes_x_entry);
    }

    array_write(comm, path_output, values_mpi_t, (void *)input, n_bytes_input, n_bytes_input);

    free(input);
    free(override_idx);
    free(override_values);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
