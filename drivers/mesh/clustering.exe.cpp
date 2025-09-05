#include "sfem_API.hpp"

#include "sfem_macros.h"
#include "sortreduce.h"

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

    int sxe = elem_num_sides(mesh->element_type());

    // Initialize all elements as unassigned
    for (ptrdiff_t i = 0; i < n_elements; i++) {
        elem2cluster_data[i] = SFEM_IDX_INVALID;
    }

    // Calculate number of clusters needed
    const ptrdiff_t n_clusters = (n_elements + cluster_size - 1) / cluster_size;

    // Track cluster sizes
    std::vector<ptrdiff_t> cluster_sizes(n_clusters, 0);

    // Use a simple approach: maintain a list of unassigned elements
    std::vector<ptrdiff_t> unassigned_elements;
    unassigned_elements.reserve(n_elements);
    unassigned_elements.push_back(0);

    // Function to calculate connectivity to current cluster
    auto calculate_connectivity = [&](ptrdiff_t elem, ptrdiff_t cluster_id) -> idx_t {
        idx_t connectivity = 0;
        for (ptrdiff_t j = adj_ptr[elem]; j < adj_ptr[elem + 1]; j++) {
            const ptrdiff_t neighbor = adj_idx[j];
            if (neighbor != elem && elem2cluster_data[neighbor] == cluster_id) {
                connectivity++;
            }
        }
        return connectivity;
    };

    // Simple greedy clustering algorithm
    ptrdiff_t current_cluster   = 0;
    ptrdiff_t elements_assigned = 0;

    while (elements_assigned < n_elements) {
        // If current cluster is full, start new cluster
        if (cluster_sizes[current_cluster] >= cluster_size) {
            current_cluster++;
            if (current_cluster >= n_clusters) break;

            printf("Cluster %ld is full, starting new cluster, %ld / %ld elements assigned\n",
                   current_cluster - 1,
                   elements_assigned,
                   n_elements);
        }

        // Find the best unassigned element for current cluster
        ptrdiff_t best_elem         = -1;
        idx_t     best_connectivity = -1;
        ptrdiff_t best_elem_idx     = -1;

        ptrdiff_t start = unassigned_elements.size() - 1;
        ptrdiff_t end = -1;
        ptrdiff_t inc = -1;

        if(!cluster_sizes[current_cluster]) {
            // Do not take it from the last neighbors when creating a new cluster
            start = 0;
            end = unassigned_elements.size();
            inc = 1;
        }

        for (ptrdiff_t i = start; i != end; i += inc) {
            ptrdiff_t elem = unassigned_elements[i];
            if (elem2cluster_data[elem] == SFEM_IDX_INVALID) {
                idx_t connectivity = calculate_connectivity(elem, current_cluster);

                // Prefer elements with higher connectivity to current cluster
                if (connectivity > best_connectivity) {
                    best_elem         = elem;
                    best_connectivity = connectivity;
                    best_elem_idx     = i;

                    if (best_connectivity == sxe) {
                        break;
                    }
                }
            }
        }

        // If no element found, take the first unassigned one
        if (best_elem == -1) {
            fprintf(stderr, "Should never come here!\n");
            // for (ptrdiff_t i = unassigned_elements.size() - 1; i >= 0 ; i--) {
            for (ptrdiff_t i = 0; i < unassigned_elements.size(); i++) {
                ptrdiff_t elem = unassigned_elements[i];
                if (elem2cluster_data[elem] == SFEM_IDX_INVALID) {
                    best_elem     = elem;
                    best_elem_idx = i;
                    break;
                }
            }
        }

        if (best_elem == -1) break;  // No more unassigned elements

        // Assign to current cluster
        elem2cluster_data[best_elem] = current_cluster;
        cluster_sizes[current_cluster]++;
        elements_assigned++;

        // Remove from unassigned list (swap with last element and pop)
        assert(unassigned_elements[best_elem_idx] == best_elem);
        std::swap(unassigned_elements[best_elem_idx], unassigned_elements.back());
        unassigned_elements.pop_back();

        for (ptrdiff_t k = adj_ptr[best_elem]; k < adj_ptr[best_elem + 1]; k++) {
            ptrdiff_t neigh = adj_idx[k];
            if (elem2cluster_data[neigh] == SFEM_IDX_INVALID) {
                unassigned_elements.push_back(neigh);
            }
        }
    }

    assert(elements_assigned == n_elements);

    // Output
    sfem::create_directory(output_folder.c_str());

    // Write the clustering mapping as binary file
    // elem2cluster->to_file((output_folder + "/elem2cluster.raw").c_str());

    // Write reordered mesh
    mesh->reorder_elements_from_tags(elem2cluster);
    mesh->write(output_folder.c_str());

    if (!rank) {
        // Calculate cluster quality metrics
        idx_t total_connectivity = 0;
        idx_t max_connectivity   = 0;
        idx_t min_connectivity   = n_elements;

        auto econ = sfem::create_host_buffer<int>(n_elements);
        auto d_econ = econ->data();

        for (ptrdiff_t i = 0; i < n_elements; i++) {
            idx_t connectivity = calculate_connectivity(i, elem2cluster_data[i]);
            d_econ[i] = connectivity;
            total_connectivity += connectivity;
            if (connectivity > max_connectivity) max_connectivity = connectivity;
            if (connectivity < min_connectivity) min_connectivity = connectivity;
        }


        econ->to_file((output_folder + "/connectivity.raw").c_str());

        double avg_connectivity = (double)total_connectivity / n_elements;

        printf("Element clustering completed.\n");
        printf("Clustered %td elements into %td clusters.\n", n_elements, n_clusters);
        printf("Cluster quality metrics:\n");
        printf("  Average connectivity: %.2f\n", avg_connectivity);
        printf("  Max connectivity: %d\n", max_connectivity);
        printf("  Min connectivity: %d\n", min_connectivity);
        printf("Output files written to: %s\n", output_folder.c_str());
    }

    


    return MPI_Finalize();
}
