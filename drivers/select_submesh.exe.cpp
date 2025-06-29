#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_mesh_write.h"

#include "argsort.h"

#include "sfem_API.hpp"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc < 6) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> <x> <y> <z> <max_nodes> [output_folder=./]", argv[0]);
        }

        return EXIT_FAILURE;
    }

    geom_t    roi[3]    = {(geom_t)atof(argv[2]), (geom_t)atof(argv[3]), (geom_t)atof(argv[4])};
    ptrdiff_t max_nodes = atol(argv[5]);

    const char *output_folder = "./";
    if (argc > 6) {
        output_folder = argv[6];
    }

    struct stat st = {0};
    if (stat(output_folder, &st) == -1) {
        mkdir(output_folder, 0700);
    }

    if (!rank) {
        printf("%s %s %g %g %g %ld %s\n",
               argv[0],
               argv[1],
               (double)roi[0],
               (double)roi[1],
               (double)roi[2],
               (long)max_nodes,
               output_folder);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];
    // char path[1024 * 10];

    auto mesh = sfem::Mesh::create_from_file(comm, folder);

    if (max_nodes > mesh->n_nodes()) {
        SFEM_ERROR("max_nodes > mesh.nnodes");
    }

    // double tack = MPI_Wtime();

    geom_t    closest_sq_dist = 1000000;
    ptrdiff_t closest_node    = -1;

    const int dim      = mesh->spatial_dimension();
    auto sq_dists = sfem::create_host_buffer<geom_t>(mesh->n_nodes());
    auto d_sq_dists = sq_dists->data();

    auto points = mesh->points()->data();
    for (ptrdiff_t node = 0; node < mesh->n_nodes(); ++node) {
        geom_t sq_dist = 0;
        for (int d = 0; d < dim; ++d) {
            const real_t m_x   = points[d][node];
            const real_t roi_x = roi[d];
            const real_t diff  = m_x - roi_x;
            sq_dist += diff * diff;
        }

        d_sq_dists[node] = sq_dist;

        if (sq_dist < closest_sq_dist) {
            closest_sq_dist = sq_dist;
            closest_node    = node;
        }
    }

    printf("found: %ld %g\n", closest_node, closest_sq_dist);

    if (closest_node < 0) {
        SFEM_ERROR("closest_node < 0");
    }

    auto selected_nodes   = sfem::create_host_buffer<idx_t>(mesh->n_nodes() + 1);
    auto d_selected_nodes = selected_nodes->data();
    auto additional_nodes = sfem::create_host_buffer<idx_t>(mesh->n_nodes() + 1);
    auto d_additional_nodes = additional_nodes->data();

    int SFEM_SELECT_EUCLIDEAN = 1;
    SFEM_READ_ENV(SFEM_SELECT_EUCLIDEAN, atoi);

    int SFEM_SELECT_GEODESIC = 0;
    SFEM_READ_ENV(SFEM_SELECT_GEODESIC, atoi);

    if (SFEM_SELECT_GEODESIC) {
        SFEM_SELECT_EUCLIDEAN = 0;
    }

    if (SFEM_SELECT_EUCLIDEAN) {
        auto args = sfem::create_host_buffer<idx_t>(mesh->n_nodes());
        argsort_f(mesh->n_nodes(), d_sq_dists, args->data());

        auto d_args = args->data();
        auto d_selected_nodes = selected_nodes->data();
        for (ptrdiff_t i = 0; i < max_nodes; ++i) {
            const idx_t idx         = d_args[i];
            d_selected_nodes[idx + 1] = 1;
        }

    } else {
        count_t *d_adj_ptr;
        idx_t   *d_adj_idx;
        build_crs_graph_3(mesh->n_elements(), mesh->n_nodes(), mesh->elements()->data(), &d_adj_ptr, &d_adj_idx);

        auto adj_ptr = sfem::manage_host_buffer<count_t>(mesh->n_nodes() + 1, d_adj_ptr);
        auto adj_idx = sfem::manage_host_buffer<idx_t>(d_adj_ptr[mesh->n_nodes()], d_adj_idx);

        ptrdiff_t  size_queue = (mesh->n_nodes() + 1);
        auto node_queue = sfem::create_host_buffer<ptrdiff_t>(size_queue);

        auto d_node_queue = node_queue->data();
        d_node_queue[0] = closest_node;
        for (ptrdiff_t e = 1; e < size_queue; ++e) {
            d_node_queue[e] = -1;
        }

        // Next slot
        ptrdiff_t next_slot        = 1;
        ptrdiff_t n_selected_nodes = 0;
        for (ptrdiff_t q = 0; d_node_queue[q] >= 0 && n_selected_nodes < max_nodes; q = (q + 1) % size_queue) {
            const ptrdiff_t node = d_node_queue[q];

            if (d_selected_nodes[node + 1]) continue;

            const count_t nodes_begin = d_adj_ptr[node];
            const count_t nodes_end   = d_adj_ptr[node + 1];

            for (count_t k = nodes_begin; k < nodes_end; ++k) {
                const idx_t node_adj = d_adj_idx[k];

                if (!d_selected_nodes[node_adj + 1]) {
                    d_node_queue[next_slot++ % size_queue] = node_adj;
                    continue;
                }
            }

            d_selected_nodes[node + 1] = 1;
            n_selected_nodes++;
        }
    }

    auto selected_elements = sfem::create_host_buffer<idx_t>(mesh->n_elements() + 1);
    auto d_selected_elements = selected_elements->data();

    auto      elements = mesh->elements()->data();
    const int nxe      = elem_num_nodes(mesh->element_type());
    for (ptrdiff_t i = 0; i < mesh->n_elements(); ++i) {
        for (int d = 0; d < nxe; ++d) {
            idx_t node = elements[d][i];
            idx_t sn   = d_selected_nodes[node + 1];

            if (sn != 0) {
                d_selected_elements[i + 1] = 1;
            }
        }

        if (d_selected_elements[i + 1]) {
            for (int d = 0; d < nxe; ++d) {
                idx_t node = elements[d][i];
                idx_t sn   = d_selected_nodes[node + 1];

                if (sn == 0) {
                    d_additional_nodes[node] = 1;
                }
            }
        }
    }

    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        d_selected_nodes[i + 1] += d_selected_nodes[i] + d_additional_nodes[i];
    }

    for (ptrdiff_t i = 0; i < mesh->n_elements(); ++i) {
        d_selected_elements[i + 1] += d_selected_elements[i];
    }

    ptrdiff_t n_selected_nodes    = d_selected_nodes[mesh->n_nodes()];
    ptrdiff_t n_selected_elements = d_selected_elements[mesh->n_elements()];

    auto mapping = sfem::create_host_buffer<idx_t>(n_selected_nodes);
    auto elems   = sfem::create_host_buffer<idx_t>(nxe, n_selected_elements);

    auto selected_points = sfem::create_host_buffer<geom_t>(dim, n_selected_nodes);

    auto d_mapping = mapping->data();
    auto d_elems = elems->data();
    auto d_selected_points = selected_points->data();

    for (ptrdiff_t e = 0; e < mesh->n_elements(); ++e) {
        const idx_t offset = d_selected_elements[e];
        if (offset == d_selected_elements[e + 1]) continue;

        for (int d = 0; d < nxe; ++d) {
            assert(d_selected_nodes[elements[d][e]] != d_selected_nodes[elements[d][e] + 1]);
            d_elems[d][offset] = d_selected_nodes[elements[d][e]];
        }
    }

    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        const idx_t offset = d_selected_nodes[i];
        if (offset == d_selected_nodes[i + 1]) continue;

        for (int d = 0; d < dim; ++d) {
            d_selected_points[d][offset] = points[d][i];
        }

        d_mapping[offset] = i;
    }

    if (!rank) {
        printf("select_submesh.c: nelements=%ld npoints=%ld\n", (long)n_selected_elements, n_selected_nodes);
    }

    auto selection = std::make_shared<sfem::Mesh>(
            mesh->comm(), dim, mesh->element_type(), n_selected_elements, elems, n_selected_nodes, selected_points);

    selection->set_node_mapping(mapping);
    selection->write(output_folder);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
