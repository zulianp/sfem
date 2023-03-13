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

#include "neohookean.h"

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

    if (argc != 7) {
        fprintf(stderr, "usage: %s <material> <mu> <lambda> <folder> <u.raw> <stress_prefix>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *material = argv[1];

    real_t mu = atof(argv[2]);
    real_t lambda = atof(argv[3]);

    const char *folder = argv[4];
    const char *path_u = argv[5];
    const char *output_prefix = argv[6];

    printf(
        "%s %s %g %g %s %s %s\n", 
        argv[0], material, (double)mu, (double)lambda, folder, path_u, output_prefix);

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

    real_t *stress[9];
    for (int d = 0; d < 9; ++d) {
        stress[d] = (real_t *)malloc(mesh.nelements * sizeof(real_t));
    }

    // TODO
    // if(strcmp(material, "neohookean") == 0) { }
    neohookean_cauchy_stress(mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, mu, lambda, u, stress);

    char path[2048];
    for (int d = 0; d < 9; ++d) {
        sprintf(path, "%s.%d.raw", output_prefix, d);
        array_write(comm, path, SFEM_MPI_REAL_T, stress[d], mesh.nelements, mesh.nelements);
        free(stress[d]);
    }

    free(u);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)mesh.nelements, (long)mesh.nnodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
