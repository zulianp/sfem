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

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 3) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> <output_folder>", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char *output_folder = argv[2];

    if (!rank) {
        printf("%s %s %s\n", argv[0], argv[1], output_folder);
    }

    double tick = MPI_Wtime();

    struct stat st = {0};
    if (stat(output_folder, &st) == -1) {
        mkdir(output_folder, 0700);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];
    // char path[1024 * 10];

    mesh_t p1_mesh;
    if (mesh_read(comm, folder, &p1_mesh)) {
        return EXIT_FAILURE;
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Build CRS graph of P1 mesh
    ///////////////////////////////////////////////////////////////////////////////

    count_t *rowptr = 0;
    idx_t *colidx = 0;

    // This only works for TET4 or TRI3
    build_crs_graph(p1_mesh.nelements, p1_mesh.nnodes, p1_mesh.elements, &rowptr, &colidx);

    ///////////////////////////////////////////////////////////////////////////////

    mesh_t p2_mesh;

    const count_t nnz = rowptr[p1_mesh.nnodes];
    idx_t *p2idx = (idx_t *)malloc(nnz * sizeof(idx_t));
    memset(p2idx, 0, nnz * sizeof(idx_t));

    ptrdiff_t p2_nodes = 0;
    idx_t next_id = p1_mesh.nnodes;
    for (ptrdiff_t i = 0; i < p1_mesh.nnodes; i++) {
        const count_t begin = rowptr[i];
        const count_t end = rowptr[i + 1];

        for (count_t k = begin; k < end; k++) {
            const idx_t j = colidx[k];

            if (i < j) {
                p2_nodes += 1;
                p2idx[k] = next_id++;
            }
        }
    }

    if (!rank) {
        printf("----------------------------------------\n");
        printf("mesh_p1_to_p2.c: nodes #p1 %ld, #p2 %ld\n", p1_mesh.nnodes, p2_nodes);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Create P2 mesh
    ///////////////////////////////////////////////////////////////////////////////

    const int n_p2_node_x_element = (p1_mesh.element_type == 4 ? 6 : 3);

    p2_mesh.mem_space = p1_mesh.mem_space;
    p2_mesh.nelements = p1_mesh.nelements;
    p2_mesh.nnodes = p1_mesh.nnodes + p2_nodes;
    p2_mesh.spatial_dim = p1_mesh.spatial_dim;

    p2_mesh.n_owned_nodes = p2_mesh.nnodes;
    p2_mesh.n_owned_elements = p2_mesh.nelements;
    p2_mesh.n_shared_elements = 0;

    p2_mesh.node_mapping = 0;
    p2_mesh.node_owner = 0;
    p2_mesh.element_mapping = 0;

    p2_mesh.element_type = p1_mesh.element_type + n_p2_node_x_element;
    p2_mesh.elements = (idx_t **)malloc(p2_mesh.element_type * sizeof(idx_t *));
    p2_mesh.points = (geom_t **)malloc(p2_mesh.spatial_dim * sizeof(geom_t));

    for(int d = 0; d < p1_mesh.element_type; d++) {
        p2_mesh.elements[d] = p1_mesh.elements[d];
    }

    // Allocate space for p2 nodes
    for(int d = p1_mesh.element_type; d < p2_mesh.element_type; d++) {
        p2_mesh.elements[d] = (idx_t*)malloc(p2_mesh.nelements * sizeof(idx_t));
    }

    for(int d = 0; d < p2_mesh.spatial_dim; d++) {
        p2_mesh.points[d] = (geom_t*)malloc(p2_mesh.nnodes * sizeof(geom_t));

        // Copy p1 portion
        memcpy(p2_mesh.points[d], p1_mesh.points[d], p2_mesh.nnodes * sizeof(geom_t));
    }

    // TODO generate p2 Nodes


    ///////////////////////////////////////////////////////////////////////////////
    // Write P2 mesh to disk
    ///////////////////////////////////////////////////////////////////////////////

    // mesh_write(output_folder, &p2_mesh);

    // Make sure we do not delete the same array twice
    for(int d = 0; d < p1_mesh.element_type; d++) {
        p1_mesh.elements[d] = 0;
    }

    mesh_destroy(&p1_mesh);
    // mesh_destroy(&p2_mesh);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
