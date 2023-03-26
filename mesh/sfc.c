#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../matrix.io/array_dtof.h"
#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/matrixio_crs.h"
#include "../matrix.io/utils.h"

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_mesh_write.h"

#include "argsort.h"

#ifdef DSFEM_ENABLE_MPI_SORT
#include "mpi-sort.h"
#endif

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 3) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> <output_folder>", argv[0]);
        }

        return EXIT_FAILURE;
    }

#ifndef DSFEM_ENABLE_MPI_SORT
    if (size > 1) {
        if (!rank) {
            fprintf(stderr, "Parallel runs not supported. Compile with mpi-sort\n");
        }

        MPI_Abort(comm, -1);
    }
#endif

    const char *folder = argv[1];
    const char *output_folder = argv[2];

    if (!rank) {
        printf("%s %s %s\n", argv[0], folder, output_folder);
    }

    double tick = MPI_Wtime();

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    geom_t *val = (geom_t *)malloc(mesh.n_owned_elements * sizeof(geom_t));
    memset(val, 0, mesh.n_owned_elements * sizeof(geom_t));

    idx_t *idx = (idx_t *)malloc(mesh.n_owned_elements * sizeof(idx_t));
    // memset(idx, 0, mesh.n_owned_elements * sizeof(idx_t));

    int coord = mesh.spatial_dim - 1;

    for (int d = 0; d < mesh.element_type; d++) {
        for (ptrdiff_t i = 0; i < mesh.n_owned_elements; i++) {
            val[i] += mesh.points[coord][mesh.elements[d][i]];
        }
    }

    for (ptrdiff_t i = 0; i < mesh.n_owned_elements; i++) {
        val[i] /= mesh.element_type;
    }

#ifdef DSFEM_ENABLE_MPI_SORT

    if (size > 1) {
        // TODO

        // MPI_Sort_bykey (
        //     void * sendkeys_destructive,
        //     void * sendvals_destructive,
        //     const int sendcount,
        //     MPI_Datatype keytype,
        //     MPI_Datatype valtype,
        //     void * recvkeys,
        //     void * recvvals,
        //     const int recvcount,
        //     comm);
    } else
#endif
    {
        for (ptrdiff_t i = 0; i < mesh.n_owned_elements; i++) {
            idx[i] = i;
        }

        argsort_f(mesh.n_owned_elements, val, idx);

        ptrdiff_t buff_size = MAX(mesh.n_owned_elements, mesh.n_owned_nodes) * MAX(sizeof(geom_t), sizeof(idx_t));
        void *buff = malloc(buff_size);

        // 1) rearrange elements

        idx_t *elem_buff = (idx_t *)buff;
        for (int d = 0; d < mesh.element_type; d++) {
            memcpy(elem_buff, mesh.elements[d], mesh.n_owned_elements * sizeof(idx_t));
            for (ptrdiff_t i = 0; i < mesh.n_owned_elements; i++) {
                mesh.elements[d][i] = elem_buff[idx[i]];
            }
        }

        // 2) rearrange element_mapping (if the case)

        // 3) rearrange nodes

        // 4) rearrange (or create node_mapping (for relating data))

        free(buff);
    }

    mesh_write(output_folder, &mesh);
    mesh_destroy(&mesh);
    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }
}
