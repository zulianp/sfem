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
            fprintf(stderr, "usage: %s <folder> [output_folder=./]\n", argv[0]);
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
        printf("%s %s %s\n", argv[0], argv[1], output_folder);
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
    idx_t **surf_elems = (idx_t **)malloc(3 * sizeof(idx_t *));
    idx_t *parent;
    extract_surface_connectivity(mesh.nelements, mesh.elements, &n_surf_elements, surf_elems, &parent);

    idx_t *vol2surf = (idx_t *)malloc(mesh.nnodes * sizeof(idx_t));
    for (ptrdiff_t i = 0; i < mesh.nnodes; ++i) {
        vol2surf[i] = -1;
    }

    ptrdiff_t next_id = 0;
    for (ptrdiff_t i = 0; i < n_surf_elements; ++i) {
        for (int d = 0; d < 3; ++d) {
            idx_t idx = surf_elems[d][i];
            if (vol2surf[idx] < 0) {
                vol2surf[idx] = next_id++;
            }
        }
    }

    ptrdiff_t n_surf_nodes = next_id;
    geom_t **points = (geom_t **)malloc(3 * sizeof(geom_t *));

    idx_t *mapping = (idx_t *)malloc(n_surf_nodes * sizeof(idx_t));

    for (int d = 0; d < 3; ++d) {
        points[d] = (geom_t *)malloc(n_surf_nodes * sizeof(geom_t));
    }

    for (ptrdiff_t i = 0; i < mesh.nnodes; ++i) {
        if (vol2surf[i] < 0) continue;

        mapping[vol2surf[i]] = i;

        for (int d = 0; d < 3; ++d) {
            points[d][vol2surf[i]] = mesh.points[d][i];
        }
    }

    // Reindex elements
    for(ptrdiff_t i = 0; i < n_surf_elements; ++i) {
        for (int d = 0; d < 3; ++d) {
            surf_elems[d][i] = vol2surf[surf_elems[d][i]]; 
        }
    }

    free(vol2surf);

    mesh_t surf;
    surf.comm = mesh.comm;
    surf.mem_space = mesh.mem_space;

    surf.spatial_dim = mesh.spatial_dim;
    surf.element_type = mesh.element_type - 1;

    surf.nelements = n_surf_elements;
    surf.nnodes = n_surf_nodes;

    surf.elements = surf_elems;
    surf.points = points;

    surf.node_mapping = mapping;
    surf.element_mapping = 0;
    surf.node_owner = 0;

    mesh_write(output_folder, &surf);

    char path[2048];
    sprintf(path, "%s/parent.raw", output_folder);
    array_write(comm, path, SFEM_MPI_IDX_T, parent, n_surf_elements, n_surf_elements);

    // Clean-up

    mesh_destroy(&mesh);
    mesh_destroy(&surf);
    free(parent);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
