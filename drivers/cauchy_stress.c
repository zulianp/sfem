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

    if (argc != 7 && argc != 9) {
        fprintf(stderr, "usage: (input can be AoS or SoA. output is always SoA)\n");
        fprintf(stderr, " (AoS): %s <material> <mu> <lambda> <folder> <uxyz.raw> <stress_prefix>\n", argv[0]);
        fprintf(stderr,
                " (SoA): %s <material> <mu> <lambda> <folder> <ux.raw> <uy.raw> <uz.raw> <stress_prefix>\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char *material = argv[1];

    real_t mu = atof(argv[2]);
    real_t lambda = atof(argv[3]);

    const char *folder = argv[4];
    const char *path_u[3];
    const char *output_prefix;

    const char * SFEM_OUTPUT_POSTFIX = "";
    SFEM_READ_ENV(SFEM_OUTPUT_POSTFIX, );

    int is_AoS = argc == 7;

    if (is_AoS) {
        path_u[0] = argv[5];
        output_prefix = argv[6];

        printf("(AoS) %s %s %g %g %s %s %s\n", argv[0], material, (double)mu, (double)lambda, folder, path_u[0], output_prefix);

    } else {
        path_u[0] = argv[5];
        path_u[1] = argv[6];
        path_u[2] = argv[7];
        output_prefix = argv[8];

        printf("(SoA) %s %s %g %g %s %s %s %s %s\n",
               argv[0],
               material,
               (double)mu,
               (double)lambda,
               folder,
               path_u[0],
               path_u[1],
               path_u[2],
               output_prefix);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    real_t *stress[6];
    for (int d = 0; d < 6; ++d) {
        stress[d] = (real_t *)malloc(mesh.nelements * sizeof(real_t));
    }

    // TODO
    // if(strcmp(material, "neohookean") == 0) { }

    if (is_AoS) {
        real_t *u;
        ptrdiff_t u_n_local, u_n_global;
        array_create_from_file(comm, path_u[0], SFEM_MPI_REAL_T, (void **)&u, &u_n_local, &u_n_global);
        neohookean_cauchy_stress_aos(mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, mu, lambda, u, stress);
        free(u);
    } else {
        real_t *u[3];

        ptrdiff_t u_n_local, u_n_global;

        for (int d = 0; d < 3; d++) {
            array_create_from_file(comm, path_u[d], SFEM_MPI_REAL_T, (void **)&u[d], &u_n_local, &u_n_global);
        }
        neohookean_cauchy_stress_soa(mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, mu, lambda, u, stress);
        for (int d = 0; d < 3; d++) {
            free(u[d]);
        }
    }

    char path[2048];
    for (int d = 0; d < 6; ++d) {
        sprintf(path, "%s.%d%s.raw", output_prefix, d, SFEM_OUTPUT_POSTFIX);
        array_write(comm, path, SFEM_MPI_REAL_T, stress[d], mesh.nelements, mesh.nelements);
        free(stress[d]);
    }

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)mesh.nelements, (long)mesh.nnodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
