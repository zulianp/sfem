#include "sfem_API.hpp"

#include "sfem_macros.h"

#include "sfem_DualGraph.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

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

    auto dual_graph = sfem::DualGraph::create(mesh);
    auto adj_ptr    = dual_graph->adj_ptr()->data();
    auto adj_idx    = dual_graph->adj_idx()->data();

    std::vector<char>      visited(nelements, 0);
    std::vector<ptrdiff_t> element_new_to_old;
    element_new_to_old.reserve(nelements);

    auto degree = [&](ptrdiff_t e) -> ptrdiff_t { return static_cast<ptrdiff_t>(adj_ptr[e + 1] - adj_ptr[e]); };

    std::vector<ptrdiff_t> queue;
    queue.reserve(nelements);
    std::vector<ptrdiff_t> neighbors;

    // Order elements via a BFS on the dual graph so consecutive indices share nodes
    while (element_new_to_old.size() < static_cast<size_t>(nelements)) {
        ptrdiff_t start       = -1;
        ptrdiff_t best_degree = std::numeric_limits<ptrdiff_t>::max();

        for (ptrdiff_t e = 0; e < nelements; ++e) {
            if (!visited[e]) {
                const ptrdiff_t deg = degree(e);
                if (deg < best_degree) {
                    best_degree = deg;
                    start       = e;
                }
            }
        }

        if (start == -1) {
            break;
        }

        queue.clear();
        queue.push_back(start);
        visited[start] = 1;

        size_t head = 0;
        while (head < queue.size()) {
            const ptrdiff_t current = queue[head++];
            element_new_to_old.push_back(current);

            neighbors.clear();
            for (ptrdiff_t idx = adj_ptr[current]; idx < adj_ptr[current + 1]; ++idx) {
                const ptrdiff_t neighbor = adj_idx[idx];
                if (!visited[neighbor]) {
                    visited[neighbor] = 1;
                    neighbors.push_back(neighbor);
                }
            }

            std::sort(neighbors.begin(), neighbors.end(), [&](ptrdiff_t a, ptrdiff_t b) {
                const ptrdiff_t da = degree(a);
                const ptrdiff_t db = degree(b);
                if (da == db) {
                    return a < b;
                }
                return da < db;
            });

            queue.insert(queue.end(), neighbors.begin(), neighbors.end());
        }
    }

    for (ptrdiff_t e = 0; e < nelements; ++e) {
        if (!visited[e]) {
            visited[e] = 1;
            element_new_to_old.push_back(e);
        }
    }

    assert(element_new_to_old.size() == static_cast<size_t>(nelements));

    std::vector<ptrdiff_t> element_old_to_new(nelements);
    for (ptrdiff_t new_idx = 0; new_idx < nelements; ++new_idx) {
        element_old_to_new[element_new_to_old[new_idx]] = new_idx;
    }

    auto reordered_elements   = sfem::create_host_buffer<idx_t>(nxe, nelements);
    auto d_reordered_elements = reordered_elements->data();
    for (int v = 0; v < nxe; ++v) {
        for (ptrdiff_t new_idx = 0; new_idx < nelements; ++new_idx) {
            const ptrdiff_t old_idx          = element_new_to_old[new_idx];
            d_reordered_elements[v][new_idx] = elements[v][old_idx];
        }
        std::memcpy(elements[v], d_reordered_elements[v], nelements * sizeof(idx_t));
    }

    std::vector<idx_t> node_old_to_new(n_nodes, SFEM_IDX_INVALID);
    std::vector<idx_t> node_new_to_old;
    node_new_to_old.reserve(n_nodes);

    // Assign node ids in the order they are first touched by the reordered elements
    idx_t next_node_id = 0;
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        for (int v = 0; v < nxe; ++v) {
            const idx_t node = elements[v][e];
            if (node_old_to_new[node] == SFEM_IDX_INVALID) {
                node_old_to_new[node] = next_node_id++;
                node_new_to_old.push_back(node);
            }
        }
    }

    for (idx_t node = 0; node < static_cast<idx_t>(n_nodes); ++node) {
        if (node_old_to_new[node] == SFEM_IDX_INVALID) {
            node_old_to_new[node] = next_node_id++;
            node_new_to_old.push_back(node);
        }
    }

    assert(node_new_to_old.size() == static_cast<size_t>(n_nodes));

    const int spatial_dim   = mesh->spatial_dimension();
    auto      points_copy   = sfem::create_host_buffer<geom_t>(spatial_dim, n_nodes);
    auto      d_points_copy = points_copy->data();
    for (int d = 0; d < spatial_dim; ++d) {
        std::memcpy(d_points_copy[d], points[d], n_nodes * sizeof(geom_t));
    }

    for (int d = 0; d < spatial_dim; ++d) {
        for (ptrdiff_t new_node = 0; new_node < n_nodes; ++new_node) {
            const idx_t old_node = node_new_to_old[new_node];
            points[d][new_node]  = d_points_copy[d][old_node];
        }
    }

    for (int v = 0; v < nxe; ++v) {
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            const idx_t old_node = elements[v][e];
            elements[v][e]       = node_old_to_new[old_node];
        }
    }

    int64_t min_offset     = std::numeric_limits<int64_t>::max();
    int64_t max_offset     = std::numeric_limits<int64_t>::min();
    int64_t max_abs_offset = 0;

    for (int v = 0; v < nxe; ++v) {
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            const int64_t diff = static_cast<int64_t>(elements[v][e]) - e;
            min_offset         = std::min(min_offset, diff);
            max_offset         = std::max(max_offset, diff);
            max_abs_offset     = std::max<int64_t>(max_abs_offset, static_cast<int64_t>(std::llabs(diff)));
        }
    }

    const bool fits_int8  = max_abs_offset <= std::numeric_limits<int8_t>::max();
    const bool fits_int16 = max_abs_offset <= std::numeric_limits<int16_t>::max();

    enum class OffsetStorage { Int8, Int16, Int32 };
    OffsetStorage storage = fits_int8 ? OffsetStorage::Int8 : (fits_int16 ? OffsetStorage::Int16 : OffsetStorage::Int32);

    switch (storage) {
        case OffsetStorage::Int8: {
            auto celements_i8 = sfem::create_host_buffer<int8_t>(nxe, nelements);
            auto d_celements  = celements_i8->data();
            for (int v = 0; v < nxe; ++v) {
                for (ptrdiff_t e = 0; e < nelements; ++e) {
                    d_celements[v][e] = static_cast<int8_t>(static_cast<int64_t>(elements[v][e]) - e);
                }
            }

            printf("Storing connectivity offsets as int8 (max abs %lld)\n", static_cast<long long>(max_abs_offset));

            break;
        }
        case OffsetStorage::Int16: {
            auto celements_i16 = sfem::create_host_buffer<int16_t>(nxe, nelements);
            auto d_celements   = celements_i16->data();
            for (int v = 0; v < nxe; ++v) {
                for (ptrdiff_t e = 0; e < nelements; ++e) {
                    d_celements[v][e] = static_cast<int16_t>(static_cast<int64_t>(elements[v][e]) - e);
                }
            }

            printf("Storing connectivity offsets as int16 (max abs %lld)\n", static_cast<long long>(max_abs_offset));

            break;
        }
        case OffsetStorage::Int32: {
            auto celements_i32 = sfem::create_host_buffer<int32_t>(nxe, nelements);
            auto d_celements   = celements_i32->data();
            for (int v = 0; v < nxe; ++v) {
                for (ptrdiff_t e = 0; e < nelements; ++e) {
                    d_celements[v][e] = static_cast<int32_t>(static_cast<int64_t>(elements[v][e]) - e);
                }
            }

            printf("Connectivity offsets require int32 (max abs %lld)\n", static_cast<long long>(max_abs_offset));

            break;
        }
    }

    printf("Offset range: [%lld, %lld]\n", static_cast<long long>(min_offset), static_cast<long long>(max_offset));

    auto element_perm_buffer = sfem::create_host_buffer<element_idx_t>(nelements);
    auto d_element_perm      = element_perm_buffer->data();
    for (ptrdiff_t new_idx = 0; new_idx < nelements; ++new_idx) {
        d_element_perm[new_idx] = static_cast<element_idx_t>(element_new_to_old[new_idx]);
    }

    auto node_perm_buffer = sfem::create_host_buffer<idx_t>(n_nodes);
    auto d_node_perm      = node_perm_buffer->data();
    for (ptrdiff_t new_idx = 0; new_idx < n_nodes; ++new_idx) {
        d_node_perm[new_idx] = node_new_to_old[new_idx];
    }

    // Output
    sfem::create_directory(output_folder.c_str());
    mesh->write(output_folder.c_str());

    element_perm_buffer->to_file((output_folder + "/element_permutation.raw").c_str());
    node_perm_buffer->to_file((output_folder + "/node_permutation.raw").c_str());

    return MPI_Finalize();
}
