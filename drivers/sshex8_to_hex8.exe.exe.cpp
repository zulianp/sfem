#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "crs_graph.h"
#include "sfem_base.h"
#include "sfem_defs.h"

#include "read_mesh.h"

#include "laplacian.h"
#include "sfem_hex8_mesh_graph.h"
#include "sshex8.h"
#include "sshex8_laplacian.h"
#include "sshex8_mesh.h"

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

    if (argc < 4) {
        fprintf(stderr, "usage: %s <level> <folder> <output>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int level = atoi(argv[1]);

    const char *folder      = argv[2];
    const char *path_output = argv[3];

    printf("%s %d %s %s\n", argv[0], level, folder, path_output);

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Set-up (read and init)
    ///////////////////////////////////////////////////////////////////////////////

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    const int nxe = sshex8_nxe(level);
    const int txe = sshex8_txe(level);

    idx_t **elements = 0;

    elements = (idx_t**)malloc(nxe * sizeof(idx_t *));
    for (int d = 0; d < nxe; d++) {
        elements[d] = (idx_t*)malloc(mesh.nelements * sizeof(idx_t));
    }

    for (int d = 0; d < nxe; d++) {
        for (ptrdiff_t i = 0; i < mesh.nelements; i++) {
            elements[d][i] = -1;
        }
    }

    ptrdiff_t n_unique_nodes, interior_start;
    sshex8_generate_elements(level, mesh.nelements, mesh.nnodes, mesh.elements, elements, &n_unique_nodes, &interior_start);

    // ///////////////////////////////////////////////////////////////////////////////
    // Generate explicit hex8 micro-mesh
    // ///////////////////////////////////////////////////////////////////////////////
    ptrdiff_t n_micro_elements = mesh.nelements * txe;

    idx_t **hex8_elements = (idx_t**)malloc(8 * sizeof(idx_t *));
    for (int d = 0; d < 8; d++) {
        hex8_elements[d] = (idx_t*)malloc(n_micro_elements * sizeof(idx_t));
    }

    // Elements
    sshex8_to_standard_hex8_mesh(level, mesh.nelements, elements, hex8_elements);

    geom_t **hex8_points = (geom_t**)malloc(3 * sizeof(geom_t *));
    for (int d = 0; d < 3; d++) {
        hex8_points[d] = (geom_t*)calloc(n_unique_nodes, sizeof(geom_t));
    }

    sshex8_fill_points(level, mesh.nelements, elements, mesh.points, hex8_points);

    // ///////////////////////////////////////////////////////////////////////////////
    // Write to disk
    // ///////////////////////////////////////////////////////////////////////////////

    struct stat st = {0};
    if (stat(path_output, &st) == -1) {
        mkdir(path_output, 0700);
    }

    char path[1024 * 10];
    for (int lnode = 0; lnode < 8; lnode++) {
        sprintf(path, "%s/i%d.raw", path_output, lnode);
        array_write(comm, path, SFEM_MPI_IDX_T, hex8_elements[lnode], mesh.nelements * txe, mesh.nelements * txe);
    }

    const char *tags[3] = {"x", "y", "z"};
    for (int d = 0; d < 3; d++) {
        sprintf(path, "%s/%s.raw", path_output, tags[d]);
        array_write(comm, path, SFEM_MPI_GEOM_T, hex8_points[d], n_unique_nodes, n_unique_nodes);
    }

    // ///////////////////////////////////////////////////////////////////////////////
    // Free resources
    // ///////////////////////////////////////////////////////////////////////////////

    mesh_destroy(&mesh);

    for (int d = 0; d < nxe; d++) {
        free(elements[d]);
    }

    free(elements);

    // --

    for (int d = 0; d < 8; d++) {
        free(hex8_elements[d]);
    }

    free(hex8_elements);

    // --

    for (int d = 0; d < 3; d++) {
        free(hex8_points[d]);
    }

    free(hex8_points);

    // ///////////////////////////////////////////////////////////////////////////////
    // Stats
    // ///////////////////////////////////////////////////////////////////////////////

    double tock = MPI_Wtime();
    float  TTS  = tock - tick;

    printf("Generated HEX8 mesh in %g [s]\n", TTS);
    printf("nelements %ld\n", n_micro_elements);
    printf("nnodes    %ld\n", n_unique_nodes);
    printf("nxe       %d\n", nxe);
    printf("txe       %d\n", txe);

    return MPI_Finalize();
}
