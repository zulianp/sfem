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

    struct stat st = {0};
    if (stat(output_folder, &st) == -1) {
        mkdir(output_folder, 0700);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    MPI_Barrier(comm);

    char output_path[2048];
    for (int r = 0; r < size; ++r) {
        if (r == rank) {
            printf(
                "[%d] #elements %ld #nodes %ld #owned_nodes %ld (with ghosts %ld) "
                "#owned_elements %ld (with ghosts %ld) #shared_elements %ld\n",
                rank,
                (long)mesh.nelements,
                (long)mesh.nnodes,
                (long)mesh.n_owned_nodes,
                (long)mesh.n_owned_nodes_with_ghosts,
                (long)mesh.n_owned_elements,
                (long)mesh.n_owned_elements_with_ghosts,
                (long)mesh.n_shared_elements);
        }

        fflush(stdout);
        MPI_Barrier(comm);
    }

    // float for visualization
    float *neigh_count = (float *)calloc(mesh.nnodes, sizeof(float));

    idx_t *nodeids = (idx_t *)malloc(mesh.nnodes * sizeof(idx_t));
    mesh_node_ids(&mesh, nodeids);

    float *fnodeids = (float *)malloc(mesh.nnodes * sizeof(float));

    for (ptrdiff_t i = 0; i < mesh.nnodes; i++) {
        fnodeids[i] = nodeids[i];
    }

    send_recv_t slave_to_master;
    mesh_create_nodal_send_recv(&mesh, &slave_to_master);
    float *frank = (float *)malloc(mesh.nnodes * sizeof(float));
    for (ptrdiff_t i = 0; i < mesh.nnodes; i++) {
        frank[i] = rank;
    }

    mesh_exchange_nodal_master_to_slave(&mesh, &slave_to_master, MPI_FLOAT, frank);

    { // Count how many neighboring processes a master node is connected to
        ptrdiff_t count = mesh_exchange_master_buffer_count(&slave_to_master);
        int *int_buffer = (int*)malloc(count * sizeof(int));

        ptrdiff_t n_ghosts = (mesh.nnodes - mesh.n_owned_nodes);
        int *ones = (int*)malloc(n_ghosts * sizeof(int));

        for (ptrdiff_t i = 0; i < n_ghosts; i++) {
            ones[i] = 1;
        }

        mesh_exchange_nodal_slave_to_master(&mesh, &slave_to_master, MPI_INT, ones, int_buffer);

        for (ptrdiff_t i = 0; i < n_ghosts; i++) {
            neigh_count[mesh.n_owned_nodes + i] = -1;
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

    count_t *rowptr;
    idx_t *colidx;
    send_recv_t exchange;
    mesh_remote_connectivity_graph(&mesh, &rowptr, &colidx, &exchange);

    // Everyone independent
    mesh.comm = MPI_COMM_SELF;
    sprintf(output_path, "%s/part_%0.5d", output_folder, rank);
    mesh_write(output_path, &mesh);

    sprintf(output_path, "%s/part_%0.5d/frank.raw", output_folder, rank);
    array_write(MPI_COMM_SELF, output_path, MPI_FLOAT, frank, mesh.nnodes, mesh.nnodes);

    sprintf(output_path, "%s/part_%0.5d/neigh_count.raw", output_folder, rank);
    array_write(MPI_COMM_SELF, output_path, MPI_FLOAT, neigh_count, mesh.nnodes, mesh.nnodes);

    sprintf(output_path, "%s/part_%0.5d/fnodeids.raw", output_folder, rank);
    array_write(MPI_COMM_SELF, output_path, MPI_FLOAT, fnodeids, mesh.nnodes, mesh.nnodes);

    MPI_Barrier(comm);

    send_recv_destroy(&slave_to_master);
    mesh_destroy(&mesh);

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
