#include "sfem_API.hpp"

#include "sfem_macros.h"
#include "sfem_mask.h"

#include "sfem_Packed.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

typedef uint8_t        pack_idx_t;
static const ptrdiff_t max_nodes_per_pack = std::numeric_limits<pack_idx_t>::max() + 1l;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    auto comm = sfem::Communicator::wrap(MPI_COMM_WORLD);

    if (argc != 3) {
        if (!comm->rank()) {
            fprintf(stderr, "usage: %s <mesh> <reordered_mesh>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    auto        mesh          = sfem::Mesh::create_from_file(comm, argv[1]);
    std::string output_folder = argv[2];

    auto packed = sfem::Packed<pack_idx_t>::create(mesh);

    for (ptrdiff_t b = 0; b < packed->n_blocks(); b++) {
        auto elements = packed->elements(b);
        printf("--------------------------------\n");
        for (ptrdiff_t e = 0; e < elements->extent(1); e++) {
            for (int v = 0; v < elements->extent(0); v++) {
                printf("%d ", elements->data()[v][e]);
            }
            printf("\n");
        }
        printf("\n");
        printf("--------------------------------\n");
    }

    // sfem::create_directory(output_folder.c_str());
    // mesh->write(output_folder.c_str());

    // element_perm_buffer->to_file((output_folder + "/element_permutation.raw").c_str());
    // node_perm_buffer->to_file((output_folder + "/node_permutation.raw").c_str());

    return MPI_Finalize();
}
