#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sfem_base.h"

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

    const char *help = "usage: %s <aos.raw> <n_bytes_x_entry> <block_size> <output_prefix.raw>\n";

    if (argc != 5) {
        fprintf(stderr, help, argv[0]);
        return EXIT_FAILURE;
    }

    const char *aos_path = argv[1];
    int n_bytes_x_entry = atoi(argv[2]);
    int block_size = atoi(argv[3]);

    const char *output_prefix = argv[4];
    double tick = MPI_Wtime();

    MPI_Datatype values_mpi_t = MPI_CHAR;
    char *aos_data;
    ptrdiff_t _nope_, n_bytes = 0;
    array_create_from_file(comm, aos_path, values_mpi_t, (void **)&aos_data, &_nope_, &n_bytes);
    ptrdiff_t n_values = n_bytes / n_bytes_x_entry;

    if ((n_values * n_bytes_x_entry) != n_bytes) {
        fprintf(stderr, "Bad input! %ld != %d * %ld\n", (long)n_bytes, n_bytes_x_entry, n_values);
        return EXIT_FAILURE;
    }

    ptrdiff_t single_component_n_values = n_values / block_size;
    if ((single_component_n_values * block_size) != n_values) {
        fprintf(stderr,
                "Bad input! Structure size wrong. %ld != %d * %ld\n",
                (long)n_values,
                block_size,
                single_component_n_values);
        return EXIT_FAILURE;
    }

    ptrdiff_t n_bytes_soa = single_component_n_values * n_bytes_x_entry;
    char *soa_data = (char*)malloc(n_bytes_soa);

    for (int b = 0; b < block_size; b++) {
        for (ptrdiff_t i = 0; i < single_component_n_values; i++) {
            memcpy(&soa_data[i * n_bytes_x_entry],
                   &aos_data[(i * block_size + b) * n_bytes_x_entry],
                   n_bytes_x_entry);
        }

        char output_path[SFEM_MAX_PATH_LENGTH];
        sprintf(output_path, "%s.%d.raw", output_prefix, b);
        array_write(comm, output_path, values_mpi_t, (void *)soa_data, n_bytes_soa, n_bytes_soa);
    }

    free(aos_data);
    free(soa_data);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
