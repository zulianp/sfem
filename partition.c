#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../matrix.io/array_dtof.h"
#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/matrixio_crs.h"
#include "../matrix.io/utils.h"

#include "crs_graph.h"
#include "sfem_base.h"
#include "read_mesh.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc < 2) {
        if(!rank) {
            fprintf(stderr, "usage: %s <folder> [output_folder=./]", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char *output_folder = "./";
    if (argc > 2) {
        output_folder = argv[2];
    }

    if(!rank) {
        printf("%s %s %s\n", argv[0], argv[1], output_folder);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];
    char path[1024 * 10];

    mesh_t mesh;
    if (read_mesh(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    MPI_Barrier(comm);

    double tock = MPI_Wtime();

    for(int r = 0;r < size; ++r){
        if(r == rank) {
            printf("[%d] #elements %ld #nodes %ld\n", rank, (long)mesh.nelements, (long)mesh.nnodes);
        }

        fflush(stdout);

        MPI_Barrier(comm);
    }

    MPI_Barrier(comm);

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();

}
