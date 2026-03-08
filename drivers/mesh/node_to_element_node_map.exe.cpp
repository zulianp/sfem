// #include "sfem_API.hpp"

// #include "crs_graph.hpp"
// #include "sfem_clustering.hpp"
// #include "sfem_macros.hpp"
// #include "sortreduce.hpp"

// #include "sfem_API.hpp"
// #include "sfem_DualGraph.hpp"
// #include "smesh_env.hpp"

// int main(int argc, char *argv[]) {
//     MPI_Init(&argc, &argv);

//     MPI_Comm comm = MPI_COMM_WORLD;

//     int rank, size;
//     MPI_Comm_rank(comm, &rank);
//     MPI_Comm_size(comm, &size);

//     if (argc != 3) {
//         if (!rank) {
//             fprintf(stderr, "usage: %s <mesh> <output_dir>\n", argv[0]);
//         }

//         return EXIT_FAILURE;
//     }

//     int  cluster_size = smesh::Env::read("SFEM_CLUSTER_SIZE", 32);
//     bool use_openmp   = smesh::Env::read("SFEM_USE_OPENMP", true);

//     auto        mesh          = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), smesh::Path(argv[1]));
//     std::string output_folder = argv[2];

//     const ptrdiff_t n_elements = mesh->n_elements();
//     const ptrdiff_t n_nodes    = mesh->n_nodes();

//     count_t *ln_ptr, *ln_index;
//     smesh::create_n2ln(
//         n_elements, n_nodes, 
//         smesh::elem_num_nodes(mesh->element_type(0)), 
//         mesh->elements(0)->data(), &ln_ptr, &ln_index);

//     auto mln_ptr   = sfem::manage_host_buffer<count_t>(n_nodes + 1, ln_ptr);
//     auto mln_index = sfem::manage_host_buffer<count_t>(ln_ptr[n_nodes], ln_index);

//     mln_ptr->to_file(smesh::Path((output_folder + "/ln_ptr." + dtype_COUNT_T + ".raw")));
//     mln_index->to_file(smesh::Path((output_folder + "/ln_idx." + dtype_COUNT_T + ".raw")));

//     // mln_index->print(std::cout);
//     return MPI_Finalize();
// }


int main(){return 0;}