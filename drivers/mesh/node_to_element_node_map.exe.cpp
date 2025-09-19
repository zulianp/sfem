#include "sfem_API.hpp"

#include "crs_graph.h"
#include "sfem_clustering.h"
#include "sfem_macros.h"
#include "sortreduce.h"

#include "sfem_API.hpp"
#include "sfem_DualGraph.hpp"
#include "sfem_Env.hpp"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 3) {
        if (!rank) {
            fprintf(stderr, "usage: %s <mesh> <output_dir>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    int  cluster_size = sfem::Env::read("SFEM_CLUSTER_SIZE", 32);
    bool use_openmp   = sfem::Env::read("SFEM_USE_OPENMP", true);

    auto        mesh          = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), argv[1]);
    std::string output_folder = argv[2];

    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes    = mesh->n_nodes();

    count_t *ln_ptr, *ln_index;
    build_n2ln(
        n_elements, n_nodes, 
        elem_num_nodes(mesh->element_type()), 
        mesh->elements()->data(), &ln_ptr, &ln_index);

    auto mln_ptr   = sfem::manage_host_buffer<count_t>(n_nodes + 1, ln_ptr);
    auto mln_index = sfem::manage_host_buffer<count_t>(ln_ptr[n_nodes], ln_index);

    mln_ptr->to_file((output_folder + "/ln_ptr." + dtype_COUNT_T + ".raw").c_str());
    mln_index->to_file((output_folder + "/ln_idx." + dtype_COUNT_T + ".raw").c_str());

    // mln_index->print(std::cout);
    return MPI_Finalize();
}
