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
#include "sfem_defs.hpp"




#include "sfem_API.hpp"
#include "smesh_glob.hpp"
#include "smesh_distributed_aura.hpp"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc < 2) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> [output_folder=./]\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char *output_folder = "./";
    if (argc > 2) {
        output_folder = argv[2];
    }

    if (!rank) {
        printf("%s %s %s\n", argv[0], argv[1], output_folder);
    }

    double tick = MPI_Wtime();

    smesh::create_directory(output_folder);

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    auto            mesh              = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), smesh::Path(folder));
    const ptrdiff_t n_elements        = mesh->n_elements();
    const ptrdiff_t n_nodes           = mesh->n_nodes();
    auto            dist              = mesh->distributed();
    const ptrdiff_t n_owned_nodes     = dist ? dist->n_nodes_owned() : n_nodes;
    const ptrdiff_t n_local_nodes     = dist ? dist->n_nodes_local() : n_nodes;
    const ptrdiff_t n_owned_elements  = dist ? dist->n_elements_owned() : n_elements;
    const ptrdiff_t n_local_elements  = dist ? dist->n_elements_local() : n_elements;
    const ptrdiff_t n_shared_elements = dist ? dist->n_elements_shared() : 0;
    auto            node_owner        = dist ? dist->node_owner() : sfem::create_host_buffer<int>(n_nodes);
    auto            ghosts            = dist ? dist->ghosts() : sfem::create_host_buffer<idx_t>(0);
    auto            node_offsets      = sfem::create_host_buffer<idx_t>(size + 1);
    if (dist) {
        auto d_dist_node_offsets = dist->node_offsets()->data();
        auto d_node_offsets      = node_offsets->data();
        for (int i = 0; i < size + 1; ++i) {
            d_node_offsets[i] = static_cast<idx_t>(d_dist_node_offsets[i]);
        }
    } else {
        auto d_node_offsets = node_offsets->data();
        d_node_offsets[0]   = 0;
        d_node_offsets[1]   = static_cast<idx_t>(n_nodes);
    }

    MPI_Barrier(comm);

    char output_path[2048];
    for (int r = 0; r < size; ++r) {
        if (r == rank) {
            printf("[%d] #elements %ld #nodes %ld #owned_nodes %ld (with ghosts %ld) "
                   "#owned_elements %ld (with ghosts %ld) #shared_elements %ld\n",
                   rank,
                   (long)n_elements,
                   (long)n_nodes,
                   (long)n_owned_nodes,
                   (long)n_local_nodes,
                   (long)n_owned_elements,
                   (long)n_local_elements,
                   (long)n_shared_elements);
        }

        fflush(stdout);
        MPI_Barrier(comm);
    }

    // float for visualization
    float *neigh_count = (float *)calloc(n_nodes, sizeof(float));

    idx_t *nodeids = (idx_t *)malloc(n_nodes * sizeof(idx_t));
    if (dist) {
        auto d_node_mapping = dist->node_mapping()->data();
        for (ptrdiff_t i = 0; i < n_nodes; i++) {
            nodeids[i] = static_cast<idx_t>(d_node_mapping[i]);
        }
    } else {
        for (ptrdiff_t i = 0; i < n_nodes; i++) {
            nodeids[i] = static_cast<idx_t>(i);
        }
    }

    float *fnodeids = (float *)malloc(n_nodes * sizeof(float));

    for (ptrdiff_t i = 0; i < n_nodes; i++) {
        fnodeids[i] = nodeids[i];
    }

    float *frank = (float *)malloc(n_nodes * sizeof(float));
    for (ptrdiff_t i = 0; i < n_nodes; i++) {
        frank[i] = dist ? static_cast<float>(node_owner->data()[i]) : static_cast<float>(rank);
    }

    {  // Count how many neighboring processes a master node is connected to
        if (dist) {
            auto send_count = sfem::create_host_buffer<smesh::i64>(size);
            auto send_displs = sfem::create_host_buffer<smesh::i64>(size + 1);
            auto recv_count = sfem::create_host_buffer<smesh::i64>(size);
            auto recv_displs = sfem::create_host_buffer<smesh::i64>(size + 1);
            idx_t *sparse_idx = nullptr;

            smesh::exchange_create<idx_t>(comm,
                                          n_nodes,
                                          n_owned_nodes,
                                          node_owner->data(),
                                          dist->node_offsets()->data(),
                                          ghosts->data(),
                                          send_count->data(),
                                          send_displs->data(),
                                          recv_count->data(),
                                          recv_displs->data(),
                                          &sparse_idx);

            const ptrdiff_t buffer_size = recv_count->data()[size - 1] + recv_displs->data()[size - 1];
            auto sparse_idx_buffer = sfem::manage_host_buffer<idx_t>(recv_displs->data()[size], sparse_idx);
            auto real_buffer = sfem::create_host_buffer<float>(buffer_size);

            for (ptrdiff_t i = 0; i < n_owned_nodes; i++) {
                neigh_count[i] = 0;
            }

            for (ptrdiff_t i = n_owned_nodes; i < n_nodes; i++) {
                neigh_count[i] = 1;
                neigh_count[i] = -1;
            }

            for (ptrdiff_t i = n_owned_nodes; i < n_nodes; i++) {
                neigh_count[i] = 1;
            }

            smesh::exchange_scatter_add<idx_t, float>(comm,
                                                      n_owned_nodes,
                                                      send_count->data(),
                                                      send_displs->data(),
                                                      recv_count->data(),
                                                      recv_displs->data(),
                                                      sparse_idx_buffer->data(),
                                                      neigh_count,
                                                      real_buffer->data());

            for (ptrdiff_t i = 0; i < n_owned_nodes; i++) {
                neigh_count[i] = neigh_count[i];
            }

            for (ptrdiff_t i = n_owned_nodes; i < n_nodes; i++) {
                neigh_count[i] = -1;
            }
        } else {
            for (ptrdiff_t i = 0; i < n_nodes; i++) {
                neigh_count[i] = 0;
            }
        }
    }

    // mesh_t aura;
    // mesh_aura(&mesh, &aura);
    // mesh_aura_fix_indices(&mesh, &aura);

    // Everyone independent
    mesh->set_comm(sfem::Communicator::self());
    snprintf(output_path, sizeof(output_path), "%s/part_%0.5d", output_folder, rank);
    mesh->write(smesh::Path(output_path));

    snprintf(output_path, sizeof(output_path), "%s/part_%0.5d/frank.raw", output_folder, rank);
    array_write(MPI_COMM_SELF, output_path, MPI_FLOAT, frank, mesh->n_nodes(), mesh->n_nodes());

    snprintf(output_path, sizeof(output_path), "%s/part_%0.5d/neigh_count.raw", output_folder, rank);
    array_write(MPI_COMM_SELF, output_path, MPI_FLOAT, neigh_count, mesh->n_nodes(), mesh->n_nodes());

    snprintf(output_path, sizeof(output_path), "%s/part_%0.5d/fnodeids.raw", output_folder, rank);
    array_write(MPI_COMM_SELF, output_path, MPI_FLOAT, fnodeids, mesh->n_nodes(), mesh->n_nodes());

    MPI_Barrier(comm);

    free(neigh_count);
    free(nodeids);
    free(fnodeids);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
