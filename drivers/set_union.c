#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"
#include "sortreduce.h"

#include "sfem_base.h"

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

    const char *help = "usage: %s <in1.raw> <in2.raw> <output.raw>";

    if (argc != 4) {
        fprintf(stderr, help, argv[0]);
        return EXIT_FAILURE;
    }

    const char *in1_path = argv[1];
    const char *in2_path = argv[2];
    const char *output_path = argv[3];

    if (strcmp(output_path, in1_path) == 0 || strcmp(output_path, in2_path) == 0) {
        fprintf(stderr, "Input and output are the same! Quitting!\n");
        fprintf(stderr, help, argv[0]);
        return EXIT_FAILURE;
    }

    double tick = MPI_Wtime();

    idx_t *in1;
    ptrdiff_t _ignore_, n_entries_1;
    array_create_from_file(comm, in1_path, SFEM_MPI_IDX_T, (void **)&in1, &_ignore_, &n_entries_1);

    idx_t *in2;
    ptrdiff_t n_entries_2;
    array_create_from_file(comm, in2_path, SFEM_MPI_IDX_T, (void **)&in2, &_ignore_, &n_entries_2);

    idx_t *output = (idx_t *)malloc((n_entries_1 + n_entries_2) * sizeof(idx_t));
    
    memcpy(output, in1, n_entries_1 * sizeof(idx_t));
    memcpy(&output[n_entries_1], in2, n_entries_2 * sizeof(idx_t));

    ptrdiff_t n_output = sortreduce(output, n_entries_1 + n_entries_2);
    array_write(comm, output_path, SFEM_MPI_IDX_T, (void *)output, n_output, n_output);

    free(in1);
    free(in2);
    free(output);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
