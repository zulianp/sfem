#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

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

    const char *help = "usage: %s <input.raw> <toberemoved.raw> <output.raw>";

    if (argc != 4) {
        fprintf(stderr, help, argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_path = argv[1];
    const char *toberemoved_path = argv[2];
    const char *output_path = argv[3];

    if (strcmp(output_path, input_path) == 0 || strcmp(output_path, toberemoved_path) == 0) {
        fprintf(stderr, "Input and output are the same! Quitting!\n");
        fprintf(stderr, help, argv[0]);
        return EXIT_FAILURE;
    }

    double tick = MPI_Wtime();

    idx_t *input;
    ptrdiff_t _ignore_, n_entries;
    array_read(comm, input_path, SFEM_MPI_IDX_T, (void **)&input, &_ignore_, &n_entries);

    idx_t *toberemoved;
    ptrdiff_t n_to_remove;
    array_read(comm, toberemoved_path, SFEM_MPI_IDX_T, (void **)&toberemoved, &_ignore_, &n_to_remove);

    idx_t *output = (idx_t *)malloc(n_entries * sizeof(idx_t));

    idx_t max_idx = input[0];

    for (ptrdiff_t i = 1; i < n_entries; ++i) {
        max_idx = MAX(max_idx, input[i]);
    }

    uint8_t *keep = malloc((max_idx + 1) * sizeof(uint8_t));
    memset(keep, 0, (max_idx + 1) * sizeof(uint8_t));

    for (ptrdiff_t i = 0; i < n_entries; ++i) {
        keep[input[i]] = 1;
    }

    for (ptrdiff_t i = 0; i < n_to_remove; ++i) {
        if (toberemoved[i] <= max_idx) {
            keep[toberemoved[i]] = 0;
        }
    }

    ptrdiff_t n_output = 0;
    for (ptrdiff_t i = 0; i < n_entries; ++i) {
        if (keep[input[i]]) {
            output[n_output++] = input[i];
        }
    }

    printf("n_entries = %ld n_output = %ld, max_idx = %d n_to_remove = %ld \n", n_entries, n_output, max_idx, n_to_remove);

    array_write(comm, output_path, SFEM_MPI_IDX_T, (void *)output, n_output, n_output);

    free(input);
    free(toberemoved);
    free(output);
    free(keep);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
