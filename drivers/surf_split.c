#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_mesh_write.h"

#include "argsort.h"



int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc < 2) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> [output_folder=./]", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char *output_folder = "./";
    if (argc > 2) {
        output_folder = argv[2];
    }

    struct stat st = {0};
    if (stat(output_folder, &st) == -1) {
        mkdir(output_folder, 0700);
    }

    if (!rank) {
        printf("%s %s %s\n", argv[0], argv[1], output_folder);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    mesh_t mesh;
    if (mesh_surf_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    // Create dual-graph
    {
        // TODO

    }

    // Create marker for different faces based on dihedral angles
    {
        // TODO
    }

    // Create separate meshes for the different parts
    {
        // TODO
    }

    {
        // handle mapping
        char path[2048];
        sprintf(path, "%s/node_mapping.raw", folder);

        idx_t *node_mapping = 0;
        ptrdiff_t nlocal, nglobal;
        array_read(comm, path, SFEM_MPI_IDX_T, (void **)&node_mapping, &nlocal, &nglobal);

        // create parts
        // TODO

        // clean-up
        free(node_mapping);
    }

    mesh_destroy(&mesh);
    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
