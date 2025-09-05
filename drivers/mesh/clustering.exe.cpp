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
    auto adj_ptr = dual_graph->adj_ptr()->data();
    auto adj_idx = dual_graph->adj_idx()->data();

    auto elem2cluster = sfem::create_host_buffer<idx_t>(n_elements);
    auto elem2cluster_data = elem2cluster->data();

    // Initialize all elements as unassigned
    for (ptrdiff_t i = 0; i < n_elements; i++) {
        elem2cluster_data[i] = SFEM_IDX_INVALID;
    }

    // Calculate number of clusters needed
    const ptrdiff_t n_clusters = (n_elements + cluster_size - 1) / cluster_size;
    
    // Track cluster sizes
    std::vector<ptrdiff_t> cluster_sizes(n_clusters, 0);
    
    // Use a priority queue for efficient element selection
    // Priority: (connectivity_score, element_id) - higher connectivity first
    std::priority_queue<std::pair<idx_t, ptrdiff_t>> candidate_queue;
    
    // Track which elements are in the queue to avoid duplicates
    std::vector<bool> in_queue(n_elements, false);
    
    // Start with a random element (or first element)
    ptrdiff_t start_elem = 0;
    elem2cluster_data[start_elem] = 0;
    cluster_sizes[0]++;
    in_queue[start_elem] = true;
    
    // Add neighbors of start element to queue
    for (ptrdiff_t j = adj_ptr[start_elem]; j < adj_ptr[start_elem + 1]; j++) {
        const ptrdiff_t neighbor = adj_idx[j];
        if (neighbor != start_elem && elem2cluster_data[neighbor] == SFEM_IDX_INVALID) {
            candidate_queue.push({1, neighbor}); // Start with connectivity score 1
            in_queue[neighbor] = true;
        }
    }
    
    ptrdiff_t current_cluster = 0;
    ptrdiff_t elements_assigned = 1;
    
    // Process all elements
    while (elements_assigned < n_elements) {
        // If current cluster is full, start new cluster
        if (cluster_sizes[current_cluster] >= cluster_size) {
            current_cluster++;
            if (current_cluster >= n_clusters) break;
            
            // Find next unassigned element for new cluster
            ptrdiff_t next_start = -1;
            for (ptrdiff_t elem = 0; elem < n_elements; elem++) {
                if (elem2cluster_data[elem] == SFEM_IDX_INVALID) {
                    next_start = elem;
                    break;
                }
            }
            
            if (next_start == -1) break;
            
            // Assign to new cluster
            elem2cluster_data[next_start] = current_cluster;
            cluster_sizes[current_cluster]++;
            elements_assigned++;
            
            // Add its neighbors to queue
            for (ptrdiff_t j = adj_ptr[next_start]; j < adj_ptr[next_start + 1]; j++) {
                const ptrdiff_t neighbor = adj_idx[j];
                if (neighbor != next_start && elem2cluster_data[neighbor] == SFEM_IDX_INVALID && !in_queue[neighbor]) {
                    candidate_queue.push({1, neighbor});
                    in_queue[neighbor] = true;
                }
            }
            continue;
        }
        
        // Process candidates for current cluster
        while (!candidate_queue.empty() && cluster_sizes[current_cluster] < cluster_size) {
            auto [score, elem] = candidate_queue.top();
            candidate_queue.pop();
            in_queue[elem] = false;
            
            // Skip if already assigned
            if (elem2cluster_data[elem] != SFEM_IDX_INVALID) continue;
            
            // Assign to current cluster
            elem2cluster_data[elem] = current_cluster;
            cluster_sizes[current_cluster]++;
            elements_assigned++;
            
            // Add its unassigned neighbors to queue with updated scores
            for (ptrdiff_t j = adj_ptr[elem]; j < adj_ptr[elem + 1]; j++) {
                const ptrdiff_t neighbor = adj_idx[j];
                if (neighbor != elem && elem2cluster_data[neighbor] == SFEM_IDX_INVALID) {
                    // Calculate connectivity to current cluster
                    idx_t new_score = 0;
                    for (ptrdiff_t k = adj_ptr[neighbor]; k < adj_ptr[neighbor + 1]; k++) {
                        if (elem2cluster_data[adj_idx[k]] == current_cluster) {
                            new_score++;
                        }
                    }
                    
                    if (!in_queue[neighbor]) {
                        candidate_queue.push({new_score, neighbor});
                        in_queue[neighbor] = true;
                    }
                }
            }
        }
        
        // If queue is empty but we still need elements, find any unassigned element
        if (candidate_queue.empty() && elements_assigned < n_elements) {
            for (ptrdiff_t elem = 0; elem < n_elements; elem++) {
                if (elem2cluster_data[elem] == SFEM_IDX_INVALID) {
                    elem2cluster_data[elem] = current_cluster;
                    cluster_sizes[current_cluster]++;
                    elements_assigned++;
                    
                    // Add its neighbors to queue
                    for (ptrdiff_t j = adj_ptr[elem]; j < adj_ptr[elem + 1]; j++) {
                        const ptrdiff_t neighbor = adj_idx[j];
                        if (neighbor != elem && elem2cluster_data[neighbor] == SFEM_IDX_INVALID && !in_queue[neighbor]) {
                            candidate_queue.push({1, neighbor});
                            in_queue[neighbor] = true;
                        }
                    }
                    break;
                }
            }
        }
    }

    // Output
    sfem::create_directory(output_folder.c_str());
    
    // Write the clustering mapping as binary file
    elem2cluster->to_file((output_folder + "/elem2cluster.raw").c_str());
    
    // Write reordered mesh
    mesh->write(output_folder.c_str());

    if (!rank) {
        printf("Element clustering completed.\n");
        printf("Clustered %td elements into %td clusters.\n", n_elements, n_clusters);
        printf("Output files written to: %s\n", output_folder.c_str());
    }

    return MPI_Finalize();
}
