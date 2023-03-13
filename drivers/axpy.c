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

    if (argc != 4) {
        if(!rank) {
            fprintf(stderr, "usage: %s <alpha> <x> <y>\nApplies y += alpha * x\n", argv[0]);
        }
        return EXIT_FAILURE;
    }

    const real_t alpha = atof(argv[1]);
    const char*x_path=argv[2];
    const char*y_path=argv[3];

    double tick = MPI_Wtime();

    ptrdiff_t x_local_ndofs, x_ndofs;
    real_t *x = 0;
    array_create_from_file(comm, x_path, SFEM_MPI_REAL_T, (void **)&x, &x_local_ndofs, &x_ndofs);

    ptrdiff_t y_local_ndofs, y_ndofs;
    real_t *y = 0;
    array_create_from_file(comm, y_path, SFEM_MPI_REAL_T, (void **)&y, &y_local_ndofs, &y_ndofs);

    if(y_ndofs != x_ndofs) {
        if(!rank) {
            fprintf(stderr, "Non matching vector sizes size(x) = %ld size(y) = %ld\n", x_ndofs, y_ndofs);
        }

        MPI_Abort(comm, -1);
    }

    for(ptrdiff_t i = 0; i < x_local_ndofs; ++i) {
        y[i] += alpha * x[i];
    }

    array_write(comm, y_path, SFEM_MPI_REAL_T, (void*)y, y_local_ndofs, y_ndofs);

    free(x);
    free(y);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
