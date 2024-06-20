#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "crs_graph.h"
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
            fprintf(
                stderr,
                "usage: %s <mesh> <from_element> <to_element> <input.float64> <output.float64>\n",
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

    ptrdiff_t n_coarse_nodes = max_node_id(to_element, mesh.nelements, mesh.elements) + 1;

    real_t *from = malloc(mesh.nnodes * sizeof(real_t));
    real_t *to = calloc(n_coarse_nodes, sizeof(real_t));

    idx_t *colidx = 0;
    count_t *rowptr = 0;

    int err = build_crs_graph_for_elem_type(
                  to_element, mesh.nelements, n_coarse_nodes, mesh.elements, &rowptr, &colidx) ||
              array_read(comm, path_input, SFEM_MPI_REAL_T, from, mesh.nnodes, mesh.nnodes) ||
              hierarchical_restriction(n_coarse_nodes, rowptr, colidx, from, to) ||
              array_write(comm, path_output, SFEM_MPI_REAL_T, to, n_coarse_nodes, n_coarse_nodes);

    free(colidx);
    free(rowptr);
    free(from);
    free(to);
    return MPI_Finalize() || err;
}
