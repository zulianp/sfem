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
#include "mass.h"

#include "read_mesh.h"

#include "sfem_API.hpp"

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

    auto mesh = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), folder);
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes = mesh->n_nodes();

    real_t *input;
    ptrdiff_t input_n_local, input_n_global;
    array_create_from_file(comm, path_input, SFEM_MPI_REAL_T, (void **)&input, &input_n_local, &input_n_global);

    ptrdiff_t nelements = n_elements;
    ptrdiff_t nnodes = n_nodes;

    assert(input_n_local == nnodes);

    real_t *output = (real_t *)malloc(nnodes * sizeof(real_t));
    memset(output, 0, nnodes * sizeof(real_t));

    ///////////////////////////////////////////////////////////////////////////////
    // Apply lumped mass-matrix inverse
    ///////////////////////////////////////////////////////////////////////////////

    // Apply inverse lumped-mass matrix and store result into output buffer
    apply_inv_lumped_mass(mesh->element_type(), nelements, nnodes, mesh->elements()->data(), mesh->points()->data(), input, output);

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
        printf("#elements %ld #nodes %ld\n", (long)n_elements, (long)n_nodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
