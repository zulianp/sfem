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
#include "operators/grad_p1.h"

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

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    real_t *f;
    ptrdiff_t u_n_local, u_n_global;
    array_create_from_file(comm, path_f, SFEM_MPI_REAL_T, (void **)&f, &u_n_local, &u_n_global);

    ptrdiff_t nelements = mesh.nelements;

    real_t *df[3];
    for (int d = 0; d < 3; ++d) {
        df[d] = (real_t *)malloc(nelements * sizeof(real_t));
        memset(df[d], 0, nelements * sizeof(real_t));
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Compute gradient coefficients
    ///////////////////////////////////////////////////////////////////////////////

    p1_grad3(mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, f, df[0], df[1], df[2]);

    ///////////////////////////////////////////////////////////////////////////////
    // Write cell data
    ///////////////////////////////////////////////////////////////////////////////

    for (int d = 0; d < 3; ++d) {
        array_write(comm, path_outputs[d], SFEM_MPI_REAL_T, df[d], nelements, nelements);
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
        printf("#elements %ld #nodes %ld\n", (long)mesh.nelements, (long)mesh.nnodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
