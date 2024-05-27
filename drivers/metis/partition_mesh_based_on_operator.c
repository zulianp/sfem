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

#include "sortreduce.h"

#include "sfem_metis.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 5) {
        if (!rank) {
            fprintf(stderr,
                    "usage: %s <folder> <crs_matrix_folder> <num_partitions> <partition_file>\n",
                    argv[0]);
        }

        return EXIT_FAILURE;
    }

    if (size > 1) {
        fprintf(stderr, "Parallel runs are not supported!\n");
        MPI_Abort(comm, -1);
    }

    const char *folder = argv[1];
    const char *matrix_folder = argv[2];
    const int num_partitions = atoi(argv[3]);
    const char *partition_file = argv[4];

    if (!rank) {
        printf("%s %s %s %d %s\n", argv[0], folder, matrix_folder, num_partitions, partition_file);
    }

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    crs_t crs;
    if (crs_read_folder(
            comm, matrix_folder, SFEM_MPI_COUNT_T, SFEM_MPI_IDX_T, SFEM_MPI_REAL_T, &crs)) {
        return EXIT_FAILURE;
    }

    idx_t *node_partitions = malloc(crs.lrows * sizeof(idx_t));
    memset(node_partitions, 0, crs.lrows * sizeof(idx_t));

    if (
        // decompose
        decompose_nnz_weighted(crs.lrows,
                               (count_t *)crs.rowptr,
                               (idx_t *)crs.colidx,
                               num_partitions,
                               node_partitions)) {
        return EXIT_FAILURE;
    }

    idx_t *element_partitions = malloc(mesh.nelements * sizeof(idx_t));

    const int nn = elem_num_nodes(mesh.element_type);

    idx_t ranks[nn];
    idx_t connects[nn];
    idx_t sranks[nn];
    for (ptrdiff_t e = 0; e < mesh.nelements; e++) {
        element_partitions[e] = mesh.nelements;

        int part = element_partitions[e];
        for (int i = 0; i < nn; i++) {
            idx_t ii = mesh.elements[i][e];
            ranks[i] = node_partitions[ii];
        }

        sort_idx(ranks, nn);

        memset(connects, 0, nn * sizeof(idx_t));

        int offset = 0;
        sranks[0] = ranks[0];
        for (int k = 1; k < nn; k++) {
            if (ranks[k] == ranks[k - 1]) {
                connects[offset]++;
            } else {
                sranks[offset++] = ranks[k];
            }
        }

        int argmax = 0;
        for (int k = 1; k < offset; k++) {
            if (connects[argmax] < connects[k]) {
                argmax = k;
            }
        }

        part = sranks[argmax];
        element_partitions[e] = part;
    }

    array_write(
        comm, partition_file, SFEM_MPI_IDX_T, element_partitions, mesh.nelements, mesh.nelements);

    mesh_destroy(&mesh);
    free(node_partitions);
    free(element_partitions);
    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
