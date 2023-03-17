#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "crs_graph.h"
#include "sfem_base.h"
#include "boundary_mass.h"

#include "read_mesh.h"

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

    if (argc != 4) {
        fprintf(stderr, "usage: %s <folder> <in.raw> <out.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *folder = argv[1];
    const char *path_input = argv[2];
    const char *path_output = argv[3];

    printf("%s %s %s %s\n", argv[0], folder, path_input, path_output);

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    mesh_t mesh;
    if (mesh_surf_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    real_t *input;
    ptrdiff_t input_n_local, input_n_global;
    array_create_from_file(comm, path_input, SFEM_MPI_REAL_T, (void **)&input, &input_n_local, &input_n_global);

    ptrdiff_t nelements = mesh.nelements;
    ptrdiff_t nnodes = mesh.nnodes;

    assert(input_n_local == nnodes);

    real_t *output = (real_t *)malloc(nnodes * sizeof(real_t));
    memset(output, 0, nnodes * sizeof(real_t));

    ///////////////////////////////////////////////////////////////////////////////
    // Apply lumped mass-matrix inverse
    ///////////////////////////////////////////////////////////////////////////////

    // Store mass-vector into output buffer
    assemble_lumped_boundary_mass(nelements, nnodes, mesh.elements, mesh.points, output);

    for(ptrdiff_t i = 0; i < input_n_local; i++) {
        output[i] = input[i] / output[i];
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Write cell data
    ///////////////////////////////////////////////////////////////////////////////

    array_write(comm, path_output, SFEM_MPI_REAL_T, output, nnodes, nnodes);

    ///////////////////////////////////////////////////////////////////////////////
    // Free resources
    ///////////////////////////////////////////////////////////////////////////////

    free(input);
    free(output);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)mesh.nelements, (long)mesh.nnodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
