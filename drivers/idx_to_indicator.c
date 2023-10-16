#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

typedef float geom_t;
typedef int idx_t;
typedef double real_t;

#ifdef NDEBUG
#define INLINE inline
#else
#define INLINE
#endif

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

    if (argc < 3) {
        fprintf(stderr, "usage: %s <idx.raw> <n> [out=indicator.raw]", argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_path = "./indicator.raw";

    if (argc > 3) {
        output_path = argv[3];
    }

    double tick = MPI_Wtime();

    typedef float indicator_t;
    MPI_Datatype indicator_mpi_t = MPI_FLOAT;

    indicator_t *indicator = 0;
    ptrdiff_t nnodes = atoll(argv[2]);

    {
        indicator = (indicator_t *)malloc(nnodes * sizeof(indicator_t));
        memset(indicator, 0, nnodes * sizeof(indicator_t));

        idx_t *indices = 0;

        ptrdiff_t _nope_, ndirichlet;
        array_create_from_file(comm, argv[1], MPI_INT, (void **)&indices, &_nope_, &ndirichlet);

        for (ptrdiff_t node = 0; node < ndirichlet; ++node) {
            idx_t i = indices[node];
            assert(i < nnodes);

            indicator[i] = (indicator_t)1;
        }

        free(indices);
    }

    array_write(comm, output_path, indicator_mpi_t, (void *)indicator, nnodes, nnodes);

    free(indicator);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
