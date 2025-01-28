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

    if (argc < 5) {
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

    ptrdiff_t _nope_, nu_p0_or_p1;

    for (int d = 0; d < 3; ++d) {
        array_create_from_file(comm, path_u[d], SFEM_MPI_REAL_T, (void **)&u[d], &_nope_, &nu_p0_or_p1);
    }

    assert(nu_p0_or_p1 == mesh.nelements || nu_p0_or_p1 == mesh.nnodes);

    real_t *div_u = (real_t *)malloc(mesh.nnodes * sizeof(real_t));
    memset(div_u, 0, mesh.nnodes * sizeof(real_t));

    if(nu_p0_or_p1 == mesh.nelements) {
        p0_u_dot_grad_q_apply(mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, u[0], u[1], u[2], div_u);
    } else {
        p1_u_dot_grad_q_apply(mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, u[0], u[1], u[2], div_u);
    }

    real_t SFEM_SCALE = 1;
    SFEM_READ_ENV(SFEM_SCALE, atof);

    if (SFEM_SCALE != 1) {
        for (ptrdiff_t i = 0; i < mesh.nnodes; ++i) {
            div_u[i] *= SFEM_SCALE;
        }
    }

    int SFEM_VERBOSE = 0;
    SFEM_READ_ENV(SFEM_VERBOSE, atoi);

    if (SFEM_VERBOSE) {
        real_t integral = 0.;
        for (ptrdiff_t i = 0; i < mesh.nnodes; ++i) {
            integral += div_u[i];
        }

        if (!rank) {
            printf("integral div(u) = %g\n", (double)integral);
        }
    }

    array_write(comm, path_output, SFEM_MPI_REAL_T, div_u, mesh.nnodes , mesh.nnodes );

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
