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

#include "tet4_strain.h"

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

    if (argc < 5) {
        fprintf(stderr, "usage: %s <folder> <ux.raw> <uy.raw> <uz.raw> <strain_prefix>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char * SFEM_OUTPUT_POSTFIX = "";
    SFEM_READ_ENV(SFEM_OUTPUT_POSTFIX, );

    const char *folder = argv[1];
    const char *path_u[3] = {argv[2], argv[3], argv[4]};
    const char *output_prefix = argv[5];

    printf("%s %s %s %s %s %s\n", argv[0], folder, path_u[0], path_u[1], path_u[2], output_prefix);

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    auto mesh = sfem::Mesh::create_from_file(comm, folder);
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes = mesh->n_nodes();

    real_t *u[3];

    ptrdiff_t u_n_local, u_n_global;

    for (int d = 0; d < 3; ++d) {
        array_create_from_file(comm, path_u[d], SFEM_MPI_REAL_T, (void **)&u[d], &u_n_local, &u_n_global);
    }
    
    real_t *principal_strains_3[3];
    for (int d = 0; d < 3; ++d) {
        principal_strains_3[d] = (real_t *)malloc(n_elements * sizeof(real_t));
    }

    principal_strains(n_elements,
          n_nodes,
          mesh->elements()->data(),
          mesh->points()->data(),
          u[0],
          u[1],
          u[2],
          principal_strains_3[0],
          principal_strains_3[1],
          principal_strains_3[2]);

    char path[2048];
    for (int d = 0; d < 3; ++d) {
        snprintf(path, sizeof(path), "%s.%d%s.raw", output_prefix, d, SFEM_OUTPUT_POSTFIX);
        array_write(comm, path, SFEM_MPI_REAL_T, principal_strains_3[d], n_elements, n_elements);
        free(principal_strains_3[d]);
    }

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)n_elements, (long)n_nodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
