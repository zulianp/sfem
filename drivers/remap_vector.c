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

    const char * help = "usage: %s <condensed.raw> <dirichlet_nodes.raw> [out=full.raw]";

    if (argc < 3) {
        fprintf(stderr, help, argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_path = "./full.raw";

    if (argc > 3) {
        output_path = argv[3];
    }

    if (strcmp(output_path, argv[1]) == 0) {
        fprintf(stderr, "Input and output are the same! Quitting!\n");
        fprintf(stderr, help, argv[0]);
        return EXIT_FAILURE;
    }

    double tick = MPI_Wtime();

    MPI_Datatype values_mpi_t = MPI_DOUBLE;

    real_t *values;
    ptrdiff_t nlocal_, nnodes;
    array_create_from_file(comm, argv[1], values_mpi_t, (void **)&values, &nlocal_, &nnodes);

    idx_t *is_dirichlet = 0;
    ptrdiff_t new_nnodes = 0;
    {
        idx_t *dirichlet_nodes = 0;
        ptrdiff_t nlocal_, ndirichlet;
        array_create_from_file(comm, argv[2], MPI_INT, (void **)&dirichlet_nodes, &nlocal_, &ndirichlet);

        new_nnodes = nnodes + ndirichlet;

        is_dirichlet = (idx_t *)malloc(new_nnodes * sizeof(idx_t));
        memset(is_dirichlet, 0, new_nnodes * sizeof(idx_t));
        
        for (ptrdiff_t node = 0; node < ndirichlet; ++node) {
            idx_t i = dirichlet_nodes[node];
            is_dirichlet[i] = 1;
        }

        free(dirichlet_nodes);
    }

    real_t *new_values = (real_t*)malloc(new_nnodes * sizeof(real_t));
    memset(new_values, 0, new_nnodes * sizeof(real_t));

    for (ptrdiff_t node = 0, old_node_idx = 0; node < new_nnodes; ++node) {
        if (!is_dirichlet[node]) {
            new_values[node] = values[old_node_idx++];
        }
    }

    array_write(comm, output_path, values_mpi_t, (void*)new_values, new_nnodes, new_nnodes);

    free(is_dirichlet);
    free(values);
    free(new_values);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("Remapped dofs: from %ld to %ld\n", (long)nnodes, (long)new_nnodes);
        printf("TTS: %g seconds\n", tock - tick);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
