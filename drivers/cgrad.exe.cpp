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
#include "tet4_grad.h"

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

    if (argc < 6) {
        fprintf(stderr, "usage: %s <folder> <f.raw> <dfdx.raw> <dfdy.raw> <dfdz.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *folder = argv[1];
    const char *path_f = argv[2];
    const char *path_outputs[3] = {argv[3], argv[4], argv[5]};

    printf("%s %s %s %s %s %s\n", argv[0], folder, path_f, path_outputs[0], path_outputs[1], path_outputs[2]);

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    auto mesh = sfem::Mesh::create_from_file(comm, folder);
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes = mesh->n_nodes();

    real_t *f;
    ptrdiff_t u_n_local, u_n_global;
    array_create_from_file(comm, path_f, SFEM_MPI_REAL_T, (void **)&f, &u_n_local, &u_n_global);

    real_t *df[3];
    for (int d = 0; d < 3; ++d) {
        df[d] = (real_t *)malloc(n_elements * sizeof(real_t));
        memset(df[d], 0, n_elements * sizeof(real_t));
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Compute gradient coefficients
    ///////////////////////////////////////////////////////////////////////////////

    tet4_grad(n_elements, n_nodes, mesh->elements()->data(), mesh->points()->data(), f, df[0], df[1], df[2]);

    real_t SFEM_SCALE=1;
    SFEM_READ_ENV(SFEM_SCALE, atof);

    if(SFEM_SCALE != 1.) {
        for (int d = 0; d < 3; ++d) {
            for(ptrdiff_t i = 0; i < n_elements; i++) {
                df[d][i] *= SFEM_SCALE;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Write cell data
    ///////////////////////////////////////////////////////////////////////////////

    for (int d = 0; d < 3; ++d) {
        array_write(comm, path_outputs[d], SFEM_MPI_REAL_T, df[d], n_elements, n_elements);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Free resources
    ///////////////////////////////////////////////////////////////////////////////

    free(f);
    for (int d = 0; d < 3; ++d) {
        free(df[d]);
    }

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)n_elements, (long)n_nodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
