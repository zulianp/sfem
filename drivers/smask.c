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

    if (argc < 4) {
        fprintf(stderr, "usage: %s <idx.raw> <input.raw> <masked.raw> [mask_value=0]", argv[0]);
        return EXIT_FAILURE;
    }

    const char*idx_path=argv[1];
    const char*input_path=argv[2];
    const char*output_path = argv[3];

    real_t mask_value = 0;
    if(argc > 4) {
        mask_value = atof(argv[4]);
    }

    double tick = MPI_Wtime();

    ptrdiff_t _nope_, ndofs;
    real_t *input = 0;
    array_create_from_file(comm, input_path, SFEM_MPI_REAL_T, (void **)&input, &_nope_, &ndofs);

    {
        idx_t *indices = 0;
        ptrdiff_t nidx;
        array_create_from_file(comm, idx_path, SFEM_MPI_IDX_T, (void **)&indices, &_nope_, &nidx);

        for(ptrdiff_t i = 0; i < nidx; ++i) {
            idx_t idx = indices[i];
            assert(idx < ndofs);
            input[idx] = mask_value;
        }

        free(indices);
    }

    array_write(comm, output_path, SFEM_MPI_REAL_T, (void*)input, ndofs, ndofs);

    free(input);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
