#include "sfem_API.hpp"

#include "sfem_macros.h"
#include "sortreduce.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 3) {
        if (!rank) {
            fprintf(stderr, "usage: %s <mesh> <reordered_mesh>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    auto        mesh          = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), argv[1]);
    std::string output_folder = argv[2];

    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes    = mesh->n_nodes();

    auto elements = mesh->elements()->data();
    auto points   = mesh->points()->data();

    const ptrdiff_t nelements = mesh->n_elements();
    const int       nxe       = mesh->n_nodes_per_element();

    // TODO: renumber nodes and elements to minimize (elements[v][e] - e) distance

    auto celements = sfem::create_host_buffer<int16_t>(nxe, nelements);
    auto d_celements = celements->data();
    for (int v = 0; v < nxe; v++) {
        for (ptrdiff_t e = 0; e < nelements; e++) {
            d_celements[v][e] = elements[v][e] - e;
            assert(d_celements[v][e] + e == elements[v][e]);
        }
    }

    // Output
    sfem::create_directory(output_folder.c_str());
    return MPI_Finalize();
}
