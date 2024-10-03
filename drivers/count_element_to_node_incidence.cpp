#include "sfem_API.hpp"

#include "sfem_defs.h"
#include "matrixio_array.h"

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

    if (argc != 3) {
        fprintf(stderr, "usage: %s <folder> <output>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *path_count = argv[2];

    const char *folder = argv[1];
    auto m = sfem::Mesh::create_from_file(comm, folder);

    auto &mesh = *((mesh_t *)m->impl_mesh());

    int *count = (int*)calloc(mesh.nnodes, sizeof(int));

    int nxe = elem_num_nodes((enum ElemType)mesh.element_type);

    for (int d = 0; d < nxe; d++) {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < mesh.nelements; ++i) {
#pragma omp atomic update
            count[mesh.elements[d][i]]++;
        }
    }

    if (array_write(comm, path_count, MPI_INT, count, mesh.nnodes, mesh.nnodes)) {
        return SFEM_FAILURE;
    }

    free(count);
    return MPI_Finalize();
}
