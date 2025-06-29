#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_mesh.h"
#include "sfem_prolongation_restriction.h"

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
    enum ElemType from_element = type_from_string(argv[2]);
    enum ElemType to_element = type_from_string(argv[3]);
    const char *path_input = argv[4];
    const char *path_output = argv[5];

    int SFEM_USE_CRS_GRAPH_RESTRICT = 0;
    SFEM_READ_ENV(SFEM_USE_CRS_GRAPH_RESTRICT, atoi);

    auto mesh = sfem::Mesh::create_from_file(comm, folder);
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes = mesh->n_nodes();

    ptrdiff_t n_coarse_nodes = max_node_id(to_element, n_elements, mesh->elements()->data()) + 1;

    real_t *from = (real_t*)malloc(n_nodes * sizeof(real_t));
    real_t *to = (real_t*)calloc(n_coarse_nodes, sizeof(real_t));

    int err = array_read(comm, path_input, SFEM_MPI_REAL_T, from, n_nodes, n_nodes);
    if (err) return EXIT_FAILURE;

    double tick = MPI_Wtime();
    double setup_elapsed = -1, compute_elapsed = -1;

    if (!SFEM_USE_CRS_GRAPH_RESTRICT) {
        printf("hierarchical_restriction_with_counting\n");

        double tack = MPI_Wtime();

        int nxe = elem_num_nodes(mesh->element_type());
        uint16_t *element_to_node_incidence_count = (uint16_t *)calloc(n_nodes, sizeof(int));
        for (int d = 0; d < nxe; d++) {
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < n_elements; ++i) {
#pragma omp atomic update
                element_to_node_incidence_count[mesh->elements()->data()[d][i]]++;
            }
        }

        setup_elapsed = MPI_Wtime() - tack;
        tack = MPI_Wtime();

        err = hierarchical_restriction_with_counting(from_element,
                                                     to_element,
                                                     n_elements,
                                                     mesh->elements()->data(),
                                                     element_to_node_incidence_count,
                                                     1,
                                                     from,
                                                     to);

        compute_elapsed = MPI_Wtime() - tack;

        free(element_to_node_incidence_count);

    } else {
        printf("hierarchical_restriction with crs graph\n");

        idx_t *colidx = 0;
        count_t *rowptr = 0;

        double tack = MPI_Wtime();

        err = build_crs_graph_for_elem_type(
                to_element, n_elements, n_coarse_nodes, mesh->elements()->data(), &rowptr, &colidx);

        setup_elapsed = MPI_Wtime() - tack;

        tack = MPI_Wtime();

        if (!err) {
            err = hierarchical_restriction(n_coarse_nodes, rowptr, colidx, 1, from, to);
        }

        compute_elapsed = MPI_Wtime() - tack;

        free(colidx);
        free(rowptr);
    }

    double elapsed = MPI_Wtime() - tick;

    if (!err) {
        err = array_write(comm, path_output, SFEM_MPI_REAL_T, to, n_coarse_nodes, n_coarse_nodes);
    }

    printf("---------------------------------------------------------------\n");
    printf("TTS: %g [s] (setup %g [s], compute %g [s])\n", elapsed, setup_elapsed, compute_elapsed);
    printf("---------------------------------------------------------------\n");

    free(from);
    free(to);
    return MPI_Finalize() || err;
}
