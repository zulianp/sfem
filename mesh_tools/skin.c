#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "../matrix.io/array_dtof.h"
#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/matrixio_crs.h"
#include "../matrix.io/utils.h"

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_mesh_write.h"

#include "extract_surface_graph.h"

#include "argsort.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc < 2) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> [output_folder=./]", argv[0]);
        }

        return EXIT_FAILURE;
    }


    const char *output_folder = "./";
    if (argc > 2) {
        output_folder = argv[2];
    }

    struct stat st = {0};
    if (stat(output_folder, &st) == -1) {
        mkdir(output_folder, 0700);
    }

    if (!rank) {
        printf("%s %s %s\n",
               argv[0],
               argv[1],
               output_folder);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    ptrdiff_t n_surf_elements = 0;
    idx_t** surf_elems = malloc(3 * sizeof(idx_t *));
    extract_surface_connectivity(mesh.nelements, mesh.elements, &n_surf_elements, surf_elems);
    ptrdiff_t n_surf_nodes = mesh.nnodes;
    // FIXME
    geom_t ** points = mesh.points;

    mesh_t surf;
    surf.comm = mesh.comm;
    surf.mem_space = mesh.mem_space;

    surf.spatial_dim = mesh.spatial_dim;
    surf.element_type = mesh.element_type - 1;

    surf.nelements = n_surf_elements;
    surf.nnodes = n_surf_nodes;

    surf.elements = surf_elems;
    surf.points = points;

    // surf.node_mapping = mapping;
    surf.node_mapping = 0;
    surf.node_owner = 0;

    mesh_write(output_folder, &surf);

    mesh_destroy(&mesh);

    // FIXME
    // mesh_destroy(&surf);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
