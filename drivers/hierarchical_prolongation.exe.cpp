#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "read_mesh.h"
#include "sfem_mesh.h"
#include "sfem_prolongation_restriction.h"

#include "matrixio_array.h"

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
    enum ElemType from_element = type_from_string(argv[2]);
    enum ElemType to_element = type_from_string(argv[3]);
    const char *path_input = argv[4];
    const char *path_output = argv[5];

    mesh_t mesh;
    mesh_read(comm, folder, &mesh);

    ptrdiff_t coarse_nodes = max_node_id(from_element, mesh.nelements, mesh.elements) + 1;

    real_t *from = (real_t *) malloc(coarse_nodes * sizeof(real_t));
    real_t *to = (real_t *)calloc(mesh.nnodes, sizeof(real_t));

    if (array_read(comm, path_input, SFEM_MPI_REAL_T, from, coarse_nodes, coarse_nodes) ||
        hierarchical_prolongation(
                from_element, to_element, mesh.nelements, mesh.elements, 1, from, to) ||
        array_write(comm, path_output, SFEM_MPI_REAL_T, to, mesh.nnodes, mesh.nnodes)) {
        return EXIT_FAILURE;
    }

    return MPI_Finalize();
}
