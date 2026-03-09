#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>


#include "smesh_mesh.hpp"
#include "sfem_prolongation_restriction.hpp"

#include "matrixio_array.h"

#include "sfem_API.hpp"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 6) {
        if (!rank) {
            fprintf(stderr,
                    "usage: %s <mesh> <from_element> <to_element> <input.float64> "
                    "<output.float64>\n",
                    argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char *folder = argv[1];
    smesh::ElemType from_element = sfem::type_from_string(argv[2]);
    smesh::ElemType to_element = sfem::type_from_string(argv[3]);
    const char *path_input = argv[4];
    const char *path_output = argv[5];

    auto mesh = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), smesh::Path(folder));
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes = mesh->n_nodes();

    ptrdiff_t coarse_nodes = max_node_id(from_element, n_elements, mesh->elements(0)->data()) + 1;

    real_t *from = (real_t *) malloc(coarse_nodes * sizeof(real_t));
    real_t *to = (real_t *)calloc(n_nodes, sizeof(real_t));

    if (array_read(comm, path_input, SFEM_MPI_REAL_T, from, coarse_nodes, coarse_nodes) ||
        hierarchical_prolongation(
                from_element, to_element, n_elements, mesh->elements(0)->data(), 1, from, to) ||
        array_write(comm, path_output, SFEM_MPI_REAL_T, to, n_nodes, n_nodes)) {
        return EXIT_FAILURE;
    }

    return MPI_Finalize();
}
