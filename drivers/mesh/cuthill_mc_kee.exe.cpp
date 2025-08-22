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
    auto            edge_graph = mesh->node_to_node_graph();
    auto            rowptr     = edge_graph->rowptr();
    auto            colidx     = edge_graph->colidx();

    // Output
    sfem::create_directory(output_folder.c_str());

    auto eccentricity = sfem::create_host_buffer<idx_t>(n_nodes);
    auto reordering   = sfem::create_host_buffer<idx_t>(n_nodes);


    // Get raw pointers to buffer data
    auto eccentricity_data = eccentricity->data();
    auto reordering_data   = reordering->data();

    auto rowptr_data       = rowptr->data();
    auto colidx_data       = colidx->data();

    auto degree = [&rowptr_data, &colidx_data](ptrdiff_t i) {
        return rowptr_data[i + 1] - rowptr_data[i];
    };

    // Initialize arrays
    for (ptrdiff_t i = 0; i < n_nodes; i++) {
        reordering_data[i]  = SFEM_IDX_INVALID;
    }

    // Compute eccentricity (maximum distance from diagonal)
    for (ptrdiff_t i = 0; i < n_nodes; i++) {
        for (ptrdiff_t j = rowptr_data[i]; j < rowptr_data[i + 1]; j++) {
            const ptrdiff_t neighbor = colidx_data[j];
            if (neighbor != i) {
                const ptrdiff_t dist = abs(neighbor - i);
                if (dist > eccentricity_data[i]) {
                    eccentricity_data[i] = dist;
                }
            }
        }
    }

    // Find starting node with minimum degree
    ptrdiff_t start_node = 0;
    idx_t     min_degree = degree(0);
    idx_t     max_degree = min_degree;
    for (ptrdiff_t i = 1; i < n_nodes; i++) {
        idx_t d = degree(i);
        if (d < min_degree) {
            min_degree = d;
            start_node = i;
        } else if (d > max_degree) {
            max_degree = d;
        }
    }

    // Cuthill-McKee reordering
    ptrdiff_t              reorder_idx = 0;
    std::vector<ptrdiff_t> queue;
    queue.reserve(n_nodes);

    std::vector<ptrdiff_t> next_level;
    next_level.reserve(n_nodes);

    std::vector<ptrdiff_t> neighbors;
    neighbors.reserve(max_degree);

    // Start with the minimum degree node
    queue.push_back(start_node);
    // mask_set(start_node, visited_data);
    reordering_data[start_node] = reorder_idx++;

    while (!queue.empty()) {
        next_level.clear();

        // Process current level
        for (ptrdiff_t current : queue) {
            // Get neighbors of current node
            neighbors.clear();

            for (ptrdiff_t j = rowptr_data[current]; j < rowptr_data[current + 1]; j++) {
                const ptrdiff_t neighbor = colidx_data[j];
                if (neighbor != current && reordering_data[neighbor] == SFEM_IDX_INVALID) {
                    neighbors.push_back(neighbor);
                }
            }

            // Sort neighbors by degree (ascending) for better bandwidth
            std::sort(neighbors.begin(), neighbors.end(), [&](ptrdiff_t a, ptrdiff_t b) {
                return degree(a) < degree(b);
            });

            // Add unvisited neighbors to next level
            for (ptrdiff_t neighbor : neighbors) {
                if (reordering_data[neighbor] == SFEM_IDX_INVALID) {
                    reordering_data[neighbor] = reorder_idx++;
                    next_level.push_back(neighbor);
                }
            }
        }

        std::swap(queue, next_level);
    }

    // Handle any remaining unvisited nodes (disconnected components)
    for (ptrdiff_t i = 0; i < n_nodes; i++) {
        if (reordering_data[i] == SFEM_IDX_INVALID) {
            reordering_data[i] = reorder_idx++;
        }
    }

    // Create inverse mapping
    auto inverse_reordering      = sfem::create_host_buffer<idx_t>(n_nodes);
    auto inverse_reordering_data = inverse_reordering->data();
    for (ptrdiff_t i = 0; i < n_nodes; i++) {
        inverse_reordering_data[reordering_data[i]] = i;
    }


    auto copied_points = sfem::copy(mesh->points());
    auto copied_points_data = copied_points->data();
    auto points_data = mesh->points()->data();
    for(ptrdiff_t d = 0; d < mesh->spatial_dimension(); d++) {
        for(ptrdiff_t i = 0; i < n_nodes; i++) {
            points_data[d][reordering_data[i]] = copied_points_data[d][i];
        }
    }

    int nxe = mesh->n_nodes_per_element();
    auto elements_data = mesh->elements()->data();
    for(int d = 0; d < nxe; d++) {
        for(ptrdiff_t i = 0; i < n_elements; i++) {
                elements_data[d][i] = reordering_data[elements_data[d][i]];
        }
    }

    mesh->write(output_folder.c_str());
    inverse_reordering->to_file((output_folder + "/map.raw").c_str());

    return MPI_Finalize();
}
