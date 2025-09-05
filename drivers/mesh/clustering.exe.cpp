#include "sfem_API.hpp"

#include "sfem_macros.h"
#include "sortreduce.h"

#include "sfem_DualGraph.hpp"
#include "sfem_Env.hpp"
#include "sfem_clustering.h"

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

    int cluster_size = sfem::Env::read("SFEM_CLUSTER_SIZE", 32);

    auto        mesh          = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), argv[1]);
    std::string output_folder = argv[2];

    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes    = mesh->n_nodes();

    auto dual_graph = sfem::DualGraph::create(mesh);
    auto adj_ptr    = dual_graph->adj_ptr()->data();
    auto adj_idx    = dual_graph->adj_idx()->data();

    auto elem2cluster      = sfem::create_host_buffer<idx_t>(n_elements);
    auto elem2cluster_data = elem2cluster->data();

    // Calculate number of clusters needed
    const ptrdiff_t n_clusters = (n_elements + cluster_size - 1) / cluster_size;

    // Track cluster sizes
    std::vector<ptrdiff_t> cluster_sizes(n_clusters, 0);

    // Perform clustering using the refactored function
    ptrdiff_t actual_n_clusters = n_clusters;
    int result = sfem_element_clustering(
        n_elements,
        cluster_size,
        adj_ptr,
        adj_idx,
        elem2cluster_data,
        cluster_sizes.data(),
        &actual_n_clusters
    );

    if (result != 0) {
        if (!rank) {
            fprintf(stderr, "Error: Clustering algorithm failed\n");
        }
        return MPI_Finalize();
    }

    // Output
    sfem::create_directory(output_folder.c_str());

    // Write the clustering mapping as binary file
    // elem2cluster->to_file((output_folder + "/elem2cluster.raw").c_str());

    // Write reordered mesh
    mesh->reorder_elements_from_tags(elem2cluster);
    mesh->renumber_nodes();
    mesh->write(output_folder.c_str());

    if (!rank) {
        // Calculate cluster quality metrics
        idx_t total_connectivity = 0;
        idx_t max_connectivity = 0;
        idx_t min_connectivity = 0;

        auto econ = sfem::create_host_buffer<int>(n_elements);
        auto d_econ = econ->data();

        result = sfem_calculate_cluster_connectivity(
            n_elements,
            adj_ptr,
            adj_idx,
            elem2cluster_data,
            d_econ,
            &total_connectivity,
            &max_connectivity,
            &min_connectivity
        );

        if (result != 0) {
            fprintf(stderr, "Error: Connectivity calculation failed\n");
            return MPI_Finalize();
        }

        econ->to_file((output_folder + "/connectivity.raw").c_str());

        double avg_connectivity = (double)total_connectivity / n_elements;

        printf("Element clustering completed.\n");
        printf("Clustered %td elements into %td clusters.\n", n_elements, actual_n_clusters);
        printf("Cluster quality metrics:\n");
        printf("  Average connectivity: %.2f\n", avg_connectivity);
        printf("  Max connectivity: %d\n", max_connectivity);
        printf("  Min connectivity: %d\n", min_connectivity);
        printf("Output files written to: %s\n", output_folder.c_str());
    }

    


    return MPI_Finalize();
}
