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

    geom_t box[2][3];

    for (int coord = 0; coord < mesh.spatial_dim; coord++) {
        box[0][coord] = mesh.points[coord][0];
        box[1][coord] = mesh.points[coord][0];

        for (ptrdiff_t i = 0; i < mesh.n_owned_nodes; i++) {
            box[0][coord] = MIN(box[0][coord], mesh.points[coord][i]);
            box[1][coord] = MAX(box[1][coord], mesh.points[coord][i]);
        }
    }

    for (int d = 0; d < mesh.element_type; d++) {
        for (ptrdiff_t i = 0; i < mesh.n_owned_elements; i++) {
            geom_t scale = 1;
            for (int coord = 0; coord < mesh.spatial_dim; coord++) {
                geom_t x = mesh.points[coord][mesh.elements[coord][i]];
                x -= box[0][coord];
                x /= box[1][coord] - box[0][coord];
                x *= scale;

                val[i] += x;
                scale *= 10;
            }
        }
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

        ptrdiff_t buff_size = MAX(mesh.n_owned_elements, mesh.n_owned_nodes) * sizeof(idx_t);
        void *buff = malloc(buff_size);

        // 1) rearrange elements
        {
            idx_t *elem_buff = (idx_t *)buff;
            for (int d = 0; d < mesh.element_type; d++) {
                memcpy(elem_buff, mesh.elements[d], mesh.n_owned_elements * sizeof(idx_t));
                for (ptrdiff_t i = 0; i < mesh.n_owned_elements; i++) {
                    mesh.elements[d][i] = elem_buff[idx[i]];
                }
            }
        }

        // 2) rearrange element_mapping (if the case)
        // TODO

        // 3) rearrange nodes
        idx_t *node_buff = (idx_t *)buff;

        {
            memset(node_buff, 0, mesh.n_owned_nodes * sizeof(idx_t));

            idx_t next_node = 1;
            for (ptrdiff_t i = 0; i < mesh.n_owned_elements; i++) {
                for (int d = 0; d < mesh.element_type; d++) {
                    idx_t i0 = mesh.elements[d][i];

                    if (!node_buff[i0]) {
                        node_buff[i0] = next_node++;
                        assert(next_node - 1 <= mesh.n_owned_nodes);
                    }
                }
            }

            assert(next_node - 1 == mesh.n_owned_nodes);

            for (ptrdiff_t i = 0; i < mesh.n_owned_nodes; i++) {
                assert(node_buff[i] > 0);
                node_buff[i] -= 1;
            }
        }

        // Update e2n
        for (int d = 0; d < mesh.element_type; d++) {
            for (ptrdiff_t i = 0; i < mesh.n_owned_elements; i++) {
                idx_t i0 = mesh.elements[d][i];
                mesh.elements[d][i] = node_buff[i0];
            }
        }

        // update coordinates
        geom_t *x_buff = malloc(mesh.n_owned_nodes * sizeof(geom_t));
        for (int d = 0; d < mesh.spatial_dim; d++) {
            memcpy(x_buff, mesh.points[d], mesh.n_owned_nodes * sizeof(geom_t));

            for (ptrdiff_t i = 0; i < mesh.n_owned_nodes; i++) {
                mesh.points[d][node_buff[i]] = x_buff[i];
            }
        }

        // 4) rearrange (or create node_mapping (for relating data))
        // TODO

        free(buff);
        free(x_buff);
    }

    free(val);
    free(idx);

    mesh_write(output_folder, &mesh);
    mesh_destroy(&mesh);
    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }
}
