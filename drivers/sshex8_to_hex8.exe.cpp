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
#include "sfem_glob.hpp"

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

    auto mesh = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), folder);
    const ptrdiff_t n_elements = mesh->n_elements();

    const int nxe = sshex8_nxe(level);
    const int txe = sshex8_txe(level);

    auto sshex8_elements = sfem::create_host_buffer<idx_t>(nxe, mesh->n_elements());
    auto d_sshex8_elements = sshex8_elements->data();
 

    for (int d = 0; d < nxe; d++) {
        for (ptrdiff_t i = 0; i < n_elements; i++) {
            d_sshex8_elements[d][i] = SFEM_IDX_INVALID;
        }
    }

    ptrdiff_t n_unique_nodes, interior_start;
    sshex8_generate_elements(level, n_elements, mesh->n_nodes(), mesh->elements()->data(), d_sshex8_elements, &n_unique_nodes, &interior_start);

    // ///////////////////////////////////////////////////////////////////////////////
    // Generate explicit hex8 micro-mesh
    // ///////////////////////////////////////////////////////////////////////////////
    ptrdiff_t n_micro_elements = n_elements * txe;
    auto hex8_elements = sfem::create_host_buffer<idx_t>(8, n_micro_elements);
    auto d_hex8_elements = hex8_elements->data();

    // Elements
    sshex8_to_standard_hex8_mesh(level, n_elements, d_sshex8_elements, d_hex8_elements);

    auto hex8_points = sfem::create_host_buffer<geom_t>(3, n_unique_nodes);
    auto d_hex8_points = hex8_points->data();


    sshex8_fill_points(level, n_elements, d_sshex8_elements, mesh->points()->data(), d_hex8_points);

    // ///////////////////////////////////////////////////////////////////////////////
    // Write to disk
    // ///////////////////////////////////////////////////////////////////////////////

    sfem::create_directory(path_output);

    char path[1024 * 10];
    for (int lnode = 0; lnode < 8; lnode++) {
        snprintf(path, sizeof(path), "%s/i%d.raw", path_output, lnode);
        array_write(comm, path, SFEM_MPI_IDX_T, d_hex8_elements[lnode], n_elements * txe, n_elements * txe);
    }

    const char *tags[3] = {"x", "y", "z"};
    for (int d = 0; d < 3; d++) {
        snprintf(path, sizeof(path), "%s/%s.raw", path_output, tags[d]);
        array_write(comm, path, SFEM_MPI_GEOM_T, d_hex8_points[d], n_unique_nodes, n_unique_nodes);
    }

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
