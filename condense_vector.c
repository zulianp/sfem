#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/matrixio_crs.h"
#include "../matrix.io/utils.h"

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
        fprintf(stderr, "usage: %s <array.raw> <dirichlet_nodes.raw> [out=condensed.raw]", argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_path = "./condensed.raw";

    if (argc > 3) {
        output_path = argv[3];
    }

    if (strcmp(output_path, argv[1]) == 0) {
        fprintf(stderr, "Input and output are the same! Quitting!\n");
        fprintf(stderr, "usage: %s <array.raw> <dirichlet_nodes.raw> [output_path=./condensed]", argv[0]);
        return EXIT_FAILURE;
    }

    double tick = MPI_Wtime();

    real_t *values;
    ptrdiff_t nlocal_, nnodes;
    array_read(comm, argv[1], MPI_DOUBLE, (void **)&values, &nlocal_, &nnodes);

    idx_t *is_dirichlet = 0;
    ptrdiff_t new_nnodes = 0;
    {
        is_dirichlet = (idx_t *)malloc(nnodes * sizeof(idx_t));
        memset(is_dirichlet, 0, nnodes * sizeof(idx_t));

        idx_t *dirichlet_nodes = 0;

        ptrdiff_t nlocal_, nn;
        array_read(comm, argv[2], MPI_INT, (void **)&dirichlet_nodes, &nlocal_, &nn);

        new_nnodes = nnodes - nn;
        for (ptrdiff_t node = 0; node < nn; ++node) {
            idx_t i = dirichlet_nodes[node];
            is_dirichlet[i] = 1;
        }

        free(dirichlet_nodes);
    }

    real_t *new_values = (real_t*)malloc(new_nnodes * sizeof(real_t));
    for (ptrdiff_t node = 0, new_node_idx = 0; node < nnodes; ++node) {
        if (!is_dirichlet[node]) {
            new_values[new_node_idx++] = values[node];
        }
    }

    array_write(comm, output_path, MPI_DOUBLE, (void*)values, new_nnodes, new_nnodes);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("Condensed dofs: from %ld to %ld\n", (long)nnodes, (long)new_nnodes);
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
