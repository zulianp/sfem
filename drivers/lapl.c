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

#include "operators/laplacian.h"

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
        fprintf(stderr, "usage: %s <folder> <u.raw> <output.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *folder = argv[1];
    const char *path_u = argv[2];
    const char *path_output = argv[3];

    printf("%s %s %s %s\n", argv[0], folder, path_u, path_output);

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    real_t *u;

    ptrdiff_t u_n_local, u_n_global;
    array_create_from_file(comm, path_u, SFEM_MPI_REAL_T, (void **)&u, &u_n_local, &u_n_global);

    assert(u_n_global != 0);

    if (u_n_global != mesh.nnodes) {
        fprintf(stderr,
                "Input field does not have correct size. Expected %ld, actual = %ld",
                (long)mesh.nnodes,
                (long)u_n_global);
        return EXIT_FAILURE;
    }

    real_t *lapl_u = (real_t *)malloc(u_n_local * sizeof(real_t));
    memset(lapl_u, 0, u_n_local * sizeof(real_t));

    laplacian_apply(mesh.element_type, mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, u, lapl_u);

    real_t SFEM_SCALE = 1;
    SFEM_READ_ENV(SFEM_SCALE, atof);

    if (SFEM_SCALE != 1) {
        for (ptrdiff_t i = 0; i < u_n_local; ++i) {
            lapl_u[i] *= SFEM_SCALE;
        }
    }

    array_write(comm, path_output, SFEM_MPI_REAL_T, lapl_u, u_n_local, u_n_global);

    free(u);
    free(lapl_u);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)mesh.nelements, (long)mesh.nnodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
