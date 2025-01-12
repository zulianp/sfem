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
#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_mesh_write.h"

#include "extract_surface_graph.h"

#include "sfem_defs.h"

#include "argsort.h"

#include "adj_table.h"

#include "sfem_glob.hpp"

SFEM_INLINE static void wedge6_to_3Xtet4(const ptrdiff_t wedge_idx,
                                         idx_t **const SFEM_RESTRICT wedges,
                                         const ptrdiff_t tet_idx_offset,
                                         idx_t **const SFEM_RESTRICT tets) {
    const idx_t i0 = wedges[0][wedge_idx];
    const idx_t i1 = wedges[1][wedge_idx];
    const idx_t i2 = wedges[2][wedge_idx];
    const idx_t i3 = wedges[3][wedge_idx];
    const idx_t i4 = wedges[4][wedge_idx];
    const idx_t i5 = wedges[5][wedge_idx];

    idx_t *node0 = &tets[0][tet_idx_offset];
    idx_t *node1 = &tets[1][tet_idx_offset];
    idx_t *node2 = &tets[2][tet_idx_offset];
    idx_t *node3 = &tets[3][tet_idx_offset];

    // tet 0
    node0[0] = i0;
    node1[0] = i1;
    node2[0] = i2;
    node3[0] = i3;

    // tet 1
    node0[1] = i1;
    node1[1] = i4;
    node2[1] = i5;
    node3[1] = i3;

    // tet 2
    node0[2] = i2;
    node1[2] = i1;
    node2[2] = i5;
    node3[2] = i3;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 3) {
        if (!rank) {
            fprintf(stderr, "usage: %s <wedge6_mesh> <output_tet4_mesh>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char *output_folder = argv[2];
    sfem::create_directory(output_folder);

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    if (elem_num_nodes(mesh.element_type) != elem_num_nodes(WEDGE6)) {
        fprintf(stderr, "This code only supports mesh with element type WEDGE6 (or compatible)\n");
        return EXIT_FAILURE;
    }

    mesh_t tet4_mesh;
    mesh_init(&tet4_mesh);

    tet4_mesh.comm = mesh.comm;
    tet4_mesh.mem_space = mesh.mem_space;

    tet4_mesh.spatial_dim = mesh.spatial_dim;
    tet4_mesh.element_type = TET4;

    tet4_mesh.nelements = 3 * mesh.nelements;
    tet4_mesh.nnodes = mesh.nnodes;
    tet4_mesh.n_owned_elements = tet4_mesh.nelements;

    tet4_mesh.node_mapping = 0;
    tet4_mesh.element_mapping = 0;
    tet4_mesh.node_owner = 0;
    tet4_mesh.points = mesh.points;

    int nnxe_tet4_mesh = 4;
    tet4_mesh.elements = (idx_t**)malloc(nnxe_tet4_mesh * sizeof(idx_t*));
    for (int d = 0; d < nnxe_tet4_mesh; d++) {
        tet4_mesh.elements[d] = (idx_t*)malloc(tet4_mesh.nelements * sizeof(idx_t));
    }

    for (ptrdiff_t i = 0; i < mesh.nelements; i++) {
        wedge6_to_3Xtet4(i, mesh.elements, i * 3, tet4_mesh.elements);
    }

    mesh_write(output_folder, &tet4_mesh);

    if (!rank) {
        printf("----------------------------------------\n");
        printf("Volume: #elements %ld #nodes %ld\n", (long)mesh.nelements, (long)mesh.nnodes);
        printf("Surface: #elements %ld #nodes %ld\n",
               (long)tet4_mesh.nelements,
               (long)tet4_mesh.nnodes);
    }

    // Clean-up
    mesh.points = 0;
    mesh_destroy(&mesh);
    mesh_destroy(&tet4_mesh);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
