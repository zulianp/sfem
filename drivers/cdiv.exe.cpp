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

#include "operators/div.h"

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

    if (argc != 6) {
        fprintf(stderr, "usage: %s <folder> <ux.raw> <uy.raw> <uz.raw> <output.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *folder = argv[1];
    const char *path_u[3] = {argv[2], argv[3], argv[4]};
    const char *path_output = argv[5];

    printf("%s %s %s %s %s %s\n", argv[0], folder, path_u[0], path_u[1], path_u[2], path_output);

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    real_t *u[3];

    ptrdiff_t u_n_local, u_n_global;

    for (int d = 0; d < 3; ++d) {
        array_create_from_file(comm, path_u[d], SFEM_MPI_REAL_T, (void **)&u[d], &u_n_local, &u_n_global);
    }

    real_t *div_u = (real_t *)malloc(mesh.nelements * sizeof(real_t));
    memset(div_u, 0, mesh.nelements * sizeof(real_t));

    cdiv(
        mesh.element_type,
        mesh.nelements, 
        mesh.nnodes, 
        mesh.elements, mesh.points, 
        u[0], u[1], u[2], div_u);

    real_t SFEM_SCALE = 1;
    SFEM_READ_ENV(SFEM_SCALE, atof);

    if (SFEM_SCALE != 1) {
        for (ptrdiff_t i = 0; i < mesh.nelements; ++i) {
            div_u[i] *= SFEM_SCALE;
        }
    }

    int SFEM_VERBOSE = 1;
    SFEM_READ_ENV(SFEM_VERBOSE, atoi);
    array_write(comm, path_output, SFEM_MPI_REAL_T, div_u, mesh.nelements, mesh.nelements);

    for (int d = 0; d < 3; ++d) {
        free(u[d]);
    }

    free(div_u);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)mesh.nelements, (long)mesh.nnodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
