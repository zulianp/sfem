#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"



#include "sfem_base.hpp"


#include "extract_surface_graph.hpp"

#include "sfem_defs.hpp"

#include "argsort.hpp"
#include "sfem_API.hpp"
#include "sfem_glob.hpp"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc < 2) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> <output_folder>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    if (size != 1) {
        fprintf(stderr, "Only serial runs supported!\n");
        return EXIT_FAILURE;
    }

    const char *output_folder = argv[2];
    sfem::create_directory(output_folder);

    if (!rank) {
        printf("%s %s %s\n", argv[0], argv[1], output_folder);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];
    char        path[SFEM_MAX_PATH_LENGTH];
    auto        mesh = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), smesh::Path(folder));
    smesh::ElemType element_type          = mesh->element_type(0);
    smesh::ElemType element_type_for_algo = element_type;
    if (element_type == smesh::TET10) {
        element_type_for_algo = smesh::TET4;
    } else if (element_type == smesh::TRI6) {
        element_type_for_algo = smesh::TRI3;
    }

    // Read only the data we need
    const int nnxe = elem_num_nodes(element_type_for_algo);
    idx_t **const elems = mesh->elements(0)->data();
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes = mesh->n_nodes();

    count_t       *adj_ptr = 0;
    element_idx_t *adj_idx = 0;
    create_dual_graph(n_elements, n_nodes, element_type_for_algo, elems, &adj_ptr, &adj_idx);

    if (!rank) {
        printf("Dual graph %ld elements, %ld nnz\n", (long)n_elements, (long)adj_ptr[n_elements]);
    }

    snprintf(path, sizeof(path), "%s/adj_ptr.raw", output_folder);
    array_write(comm, path, SFEM_MPI_COUNT_T, adj_ptr, n_elements + 1, n_elements + 1);

    snprintf(path, sizeof(path), "%s/adj_idx.raw", output_folder);
    array_write(comm, path, SFEM_MPI_ELEMENT_IDX_T, adj_idx, adj_ptr[n_elements], adj_ptr[n_elements]);

    free(adj_ptr);
    free(adj_idx);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
