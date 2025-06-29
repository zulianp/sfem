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

#include "read_mesh.h"

#include "surface_l2_projection.h"

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
        fprintf(stderr, "usage: %s <folder> <in_p0.raw> <out_p1.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *folder = argv[1];
    const char *path_p0 = argv[2];
    const char *path_p1 = argv[3];

    printf("%s %s %s %s\n", argv[0], folder, path_p0, path_p1);

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    auto mesh = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), folder);

    real_t *p0;
    ptrdiff_t p0_n_local, p0_n_global;
    array_create_from_file(comm, path_p0, SFEM_MPI_REAL_T, (void **)&p0, &p0_n_local, &p0_n_global);

    ptrdiff_t nelements = mesh->n_elements();
    ptrdiff_t nnodes = mesh->n_nodes();

    assert(p0_n_local == nelements);

    real_t *p1 = (real_t *)malloc(nnodes * sizeof(real_t));
    memset(p1, 0, nnodes * sizeof(real_t));

    ///////////////////////////////////////////////////////////////////////////////
    // Compute surface_projection
    ///////////////////////////////////////////////////////////////////////////////

    int SFEM_COMPUTE_COEFFICIENTS = 1;

    SFEM_READ_ENV(SFEM_COMPUTE_COEFFICIENTS, atoi);

    if (SFEM_COMPUTE_COEFFICIENTS) {
        surface_e_projection_coeffs(mesh->element_type(), mesh->n_elements(), mesh->n_nodes(), mesh->elements()->data(), mesh->points()->data(), p0, p1);
    } else {
        surface_e_projection_apply(mesh->element_type(), mesh->n_elements(), mesh->n_nodes(), mesh->elements()->data(), mesh->points()->data(), p0, p1);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Write cell data
    ///////////////////////////////////////////////////////////////////////////////

    array_write(comm, path_p1, SFEM_MPI_REAL_T, p1, nnodes, nnodes);

    ///////////////////////////////////////////////////////////////////////////////
    // Free resources
    ///////////////////////////////////////////////////////////////////////////////

    free(p0);
    free(p1);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)mesh->n_elements(), (long)mesh->n_nodes());
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
