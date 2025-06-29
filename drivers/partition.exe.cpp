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
#include "sfem_defs.h"

#include "sfem_mesh_write.h"

#include "mesh_aura.h"
#include "sfem_API.hpp"
#include "sfem_glob.hpp"

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

    sfem::create_directory(output_folder);

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    auto            mesh       = sfem::Mesh::create_from_file(comm, folder);
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes    = mesh->n_nodes();

    MPI_Barrier(comm);

    char output_path[2048];
    for (int r = 0; r < size; ++r) {
        if (r == rank) {
            printf("[%d] #elements %ld #nodes %ld #owned_nodes %ld (with ghosts %ld) "
                   "#owned_elements %ld (with ghosts %ld) #shared_elements %ld\n",
                   rank,
                   (long)n_elements,
                   (long)n_nodes,
                   (long)mesh->n_owned_nodes(),
                   (long)mesh->n_owned_nodes_with_ghosts(),
                   (long)mesh->n_owned_elements(),
                   (long)mesh->n_owned_elements_with_ghosts(),
                   (long)mesh->n_shared_elements());
        }

        fflush(stdout);
        MPI_Barrier(comm);
    }

    // float for visualization
    float *neigh_count = (float *)calloc(n_nodes, sizeof(float));

    idx_t *nodeids = (idx_t *)malloc(n_nodes * sizeof(idx_t));
    mesh_node_ids(comm, mesh->n_nodes(), mesh->n_owned_nodes(), mesh->node_offsets()->data(), mesh->ghosts()->data(), nodeids);

    float *fnodeids = (float *)malloc(n_nodes * sizeof(float));

    for (ptrdiff_t i = 0; i < n_nodes; i++) {
        fnodeids[i] = nodeids[i];
    }

    send_recv_t slave_to_master;
    mesh_create_nodal_send_recv(comm,
                                mesh->n_nodes(),
                                mesh->n_owned_nodes(),
                                mesh->node_owner()->data(),
                                mesh->node_offsets()->data(),
                                mesh->ghosts()->data(),
                                &slave_to_master);
                                
    float *frank = (float *)malloc(n_nodes * sizeof(float));
    for (ptrdiff_t i = 0; i < n_nodes; i++) {
        frank[i] = rank;
    }

    mesh_exchange_nodal_master_to_slave(mesh->n_owned_nodes(), &slave_to_master, MPI_FLOAT, frank);

    {  // Count how many neighboring processes a master node is connected to
        ptrdiff_t count      = mesh_exchange_master_buffer_count(&slave_to_master);
        int      *int_buffer = (int *)malloc(count * sizeof(int));

        ptrdiff_t n_ghosts = (mesh->n_nodes() - mesh->n_owned_nodes());
        int      *ones     = (int *)malloc(n_ghosts * sizeof(int));

        for (ptrdiff_t i = 0; i < n_ghosts; i++) {
            ones[i] = 1;
        }

        mesh_exchange_nodal_slave_to_master(comm, &slave_to_master, MPI_INT, ones, int_buffer);

        for (ptrdiff_t i = 0; i < n_ghosts; i++) {
            neigh_count[mesh->n_owned_nodes() + i] = -1;
        }

        for (ptrdiff_t i = 0; i < count; i++) {
            assert(neigh_count[slave_to_master.sparse_idx[i]] >= 0);
            neigh_count[slave_to_master.sparse_idx[i]] += int_buffer[i];
        }

        free(int_buffer);
        free(ones);
    }

    // mesh_t aura;
    // mesh_aura(&mesh, &aura);
    // mesh_aura_fix_indices(&mesh, &aura);

    count_t    *rowptr;
    idx_t      *colidx;
    send_recv_t exchange;
    mesh_remote_connectivity_graph(comm, mesh->element_type(), mesh->n_elements(), mesh->elements()->data(), mesh->n_nodes(), mesh->n_owned_nodes(), mesh->n_owned_elements_with_ghosts(), mesh->n_shared_elements(), mesh->node_owner()->data(), mesh->node_offsets()->data(), mesh->ghosts()->data(), &rowptr, &colidx, &exchange);

    // Everyone independent
    mesh->set_comm(MPI_COMM_SELF);
    snprintf(output_path, sizeof(output_path), "%s/part_%0.5d", output_folder, rank);
    mesh->write(output_path);

    snprintf(output_path, sizeof(output_path), "%s/part_%0.5d/frank.raw", output_folder, rank);
    array_write(MPI_COMM_SELF, output_path, MPI_FLOAT, frank, mesh->n_nodes(), mesh->n_nodes());

    snprintf(output_path, sizeof(output_path), "%s/part_%0.5d/neigh_count.raw", output_folder, rank);
    array_write(MPI_COMM_SELF, output_path, MPI_FLOAT, neigh_count, mesh->n_nodes(), mesh->n_nodes());

    snprintf(output_path, sizeof(output_path), "%s/part_%0.5d/fnodeids.raw", output_folder, rank);
    array_write(MPI_COMM_SELF, output_path, MPI_FLOAT, fnodeids, mesh->n_nodes(), mesh->n_nodes());

    MPI_Barrier(comm);

    send_recv_destroy(&slave_to_master);


    free(neigh_count);
    free(nodeids);
    free(fnodeids);

    free(rowptr);
    free(colidx);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
