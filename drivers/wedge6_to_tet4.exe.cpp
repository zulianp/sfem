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
#include "sfem_API.hpp"

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

    auto mesh = sfem::Mesh::create_from_file(comm, folder);

    if (elem_num_nodes(mesh->element_type()) != elem_num_nodes(WEDGE6)) {
        fprintf(stderr, "This code only supports mesh with element type WEDGE6 (or compatible)\n");
        return EXIT_FAILURE;
    }

    auto tet4_elements = sfem::create_host_buffer<idx_t>(4, 3 * mesh->n_elements());
    auto tet4_mesh = std::make_shared<sfem::Mesh>(
        mesh->comm(), mesh->spatial_dimension(), 
        TET4, 3 * mesh->n_elements(), tet4_elements, mesh->n_nodes(), mesh->points());


    auto elements = mesh->elements()->data();
    const ptrdiff_t n_elements = mesh->n_elements();
  
    for (ptrdiff_t i = 0; i < n_elements; i++) {
        wedge6_to_3Xtet4(i, elements, i * 3, tet4_elements->data());
    }

    tet4_mesh->write(output_folder);

    if (!rank) {
        printf("----------------------------------------\n");
        printf("Volume: #elements %ld #nodes %ld\n", (long)mesh->n_elements(), (long)mesh->n_nodes());
        printf("Surface: #elements %ld #nodes %ld\n",
               (long)tet4_mesh->n_elements(),
               (long)tet4_mesh->n_nodes());
    }

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
