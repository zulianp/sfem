#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrix.io/array_dtof.h"
#include "matrix.io/matrixio_array.h"
#include "matrix.io/matrixio_crs.h"
#include "matrix.io/utils.h"

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
            printf("[%d] #elements %ld #nodes %ld #owned_nodes %ld #owned_elements %ld #shared_elements %ld\n",
                   rank,
                   (long)mesh.nelements,
                   (long)mesh.nnodes,
                   (long)mesh.n_owned_nodes,
                   (long)mesh.n_owned_elements,
                   (long)mesh.n_shared_elements);
        }

        fflush(stdout);
        MPI_Barrier(comm);
    }
    
    send_recv_t slave_to_master;
    mesh_create_nodal_send_recv(&mesh, &slave_to_master);
    float *frank = (float *)malloc(mesh.nnodes * sizeof(float));
    for (ptrdiff_t i = 0; i < mesh.nnodes; i++) {
        frank[i] = rank;
    }

    mesh_exchange_nodal_master_to_slave(&mesh, &slave_to_master, MPI_FLOAT, frank);

    mesh_t aura;
    mesh_aura(&mesh, &aura);

    // Everyone independent
    mesh.comm = MPI_COMM_SELF;
    sprintf(output_path, "%s/part_%0.5d", output_folder, rank);
    mesh_write(output_path, &mesh);

    sprintf(output_path, "%s/part_%0.5d/frank.raw", output_folder, rank);
    array_write(MPI_COMM_SELF, output_path, MPI_FLOAT, frank, mesh.nnodes, mesh.nnodes);

    MPI_Barrier(comm);

    send_recv_destroy(&slave_to_master);
    mesh_destroy(&mesh);
    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
