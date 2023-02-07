
#include "read_mesh.h"

#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/utils.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <mpi.h>

#include "sortreduce.h"

int read_mesh(MPI_Comm comm, const char *folder, mesh_t *mesh) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size > 1) {
        double tick = MPI_Wtime();
        ///////////////////////////////////////////////////////////////
        // FIXME check from folder
        int nnodesxelem = 4;
        int ndims = 3;

        MPI_Datatype mpi_geom_t = SFEM_MPI_GEOM_T;
        MPI_Datatype mpi_idx_t = SFEM_MPI_IDX_T;

        // ///////////////////////////////////////////////////////////////

        ptrdiff_t n_local_elements = 0, n_elements = 0;
        ptrdiff_t n_local_nodes = 0, n_nodes = 0;

        char path[1024 * 10];

        idx_t **elems = (idx_t **)malloc(sizeof(idx_t *) * nnodesxelem);

        {
            idx_t *idx = 0;

            for (int d = 0; d < nnodesxelem; ++d) {
                sprintf(path, "%s/i%d.raw", folder, d);
                array_read(comm, path, mpi_idx_t, (void **)&idx, &n_local_elements, &n_elements);
                elems[d] = idx;
            }
        }

        idx_t *unique_idx = (idx_t *)malloc(sizeof(idx_t) * n_local_elements * nnodesxelem);
        for (int d = 0; d < nnodesxelem; ++d) {
            memcpy(&unique_idx[d * n_local_elements], elems[d], sizeof(idx_t) * n_local_elements);
        }

        ptrdiff_t n_unique = sortreduce(unique_idx, n_local_elements * nnodesxelem);

        ////////////////////////////////////////////////////////////////////////////////
        // Read coordinates

        geom_t **xyz = (geom_t **)malloc(sizeof(geom_t *) * ndims);

        sprintf(path, "%s/x.raw", folder);
        array_read(comm, path, mpi_geom_t, (void **)&xyz[0], &n_local_nodes, &n_nodes);

        sprintf(path, "%s/y.raw", folder);
        array_read(comm, path, mpi_geom_t, (void **)&xyz[1], &n_local_nodes, &n_nodes);

        sprintf(path, "%s/z.raw", folder);
        array_read(comm, path, mpi_geom_t, (void **)&xyz[2], &n_local_nodes, &n_nodes);

        ////////////////////////////////////////////////////////////////////////////////

        idx_t *input_node_partitions = (idx_t *)malloc(sizeof(idx_t) * (size + 1));
        memset(input_node_partitions, 0, sizeof(idx_t) * (size + 1));
        input_node_partitions[rank + 1] = n_local_nodes;

        CATCH_MPI_ERROR(MPI_Allreduce(MPI_IN_PLACE, &input_node_partitions[1], size, SFEM_MPI_IDX_T, MPI_SUM, comm));

        for (int r = 0; r < size; ++r) {
            input_node_partitions[r + 1] += input_node_partitions[r];
        }

        idx_t *gather_node_count = malloc(size * sizeof(idx_t));
        memset(gather_node_count, 0, size * sizeof(idx_t));

        int *owner_rank = malloc(n_unique * sizeof(int));
        memset(owner_rank, 0, n_unique * sizeof(int));

        for (ptrdiff_t i = 0; i < n_unique; ++i) {
            idx_t idx = unique_idx[i];
            ptrdiff_t owner = MIN(size - 1, idx / n_local_nodes);

            if (input_node_partitions[owner] > idx) {
                assert(owner < size);
                while (input_node_partitions[--owner] > idx) {
                    assert(owner < size);
                }
            } else if (input_node_partitions[owner + 1] < idx) {
                assert(owner < size);

                while (input_node_partitions[(++owner + 1)] < idx) {
                    assert(owner < size);
                }
            }

            assert(owner < size);
            assert(owner >= 0);

            assert(input_node_partitions[owner] <= idx);
            assert(input_node_partitions[owner + 1] > idx);

            gather_node_count[owner]++;
            owner_rank[i] = owner;
        }

        idx_t *scatter_node_count = malloc(size * sizeof(idx_t));
        memset(scatter_node_count, 0, size * sizeof(idx_t));

        CATCH_MPI_ERROR(
            MPI_Alltoall(gather_node_count, 1, SFEM_MPI_IDX_T, scatter_node_count, 1, SFEM_MPI_IDX_T, comm));

        idx_t *gather_node_displs = malloc((size + 1) * sizeof(idx_t));
        idx_t *scatter_node_displs = malloc((size + 1) * sizeof(idx_t));

        gather_node_displs[0] = 0;
        scatter_node_displs[0] = 0;

        for (int i = 0; i < size; i++) {
            gather_node_displs[i + 1] = gather_node_displs[i] + gather_node_count[i];
        }

        for (int i = 0; i < size; i++) {
            scatter_node_displs[i + 1] = scatter_node_displs[i] + scatter_node_count[i];
        }

        ptrdiff_t size_send_list = scatter_node_displs[size];

        idx_t *send_list = malloc(sizeof(idx_t) * size_send_list);
        memset(send_list, 0, sizeof(idx_t) * size_send_list);

        CATCH_MPI_ERROR(MPI_Alltoallv(unique_idx,
                                      gather_node_count,
                                      gather_node_displs,
                                      SFEM_MPI_IDX_T,
                                      send_list,
                                      scatter_node_count,
                                      scatter_node_displs,
                                      SFEM_MPI_IDX_T,
                                      comm));

        // MPI_Barrier(comm);
        // for (int r = 0; r < size; ++r) {
        //     if (r == rank) {
        //         printf("[%d] n_unique=%ld/%ld size_send_list=%ld\n",
        //                rank,
        //                (long)n_unique,
        //                (long)(n_local_elements * nnodesxelem),
        //                (long)size_send_list);

        //         printf("[%d] input_node_partitions: ", rank);
        //         for (int i = 0; i < size + 1; ++i) {
        //             printf("%d ", (int)input_node_partitions[i]);
        //         }

        //         printf("\n");
        //         printf("[%d] send_list: ", rank);
        //         for (int i = 0; i < size_send_list; ++i) {
        //             printf("%d ", (int)send_list[i]);
        //         }

        //         printf("\n");
        //         printf("[%d] unique_idx: ", rank);
        //         for (int i = 0; i < n_unique; ++i) {
        //             printf("%d (%d), ", (int)unique_idx[i], owner_rank[i]);
        //         }

        //         printf("\n");
        //         printf("[%d] gather_node_count: ", rank);
        //         for (int i = 0; i < size; ++i) {
        //             printf("%d ", (int)gather_node_count[i]);
        //         }

        //         printf("\n");

        //         printf("[%d] gather_node_displs: ", rank);
        //         for (int i = 0; i < size + 1; ++i) {
        //             printf("%d ", (int)gather_node_displs[i]);
        //         }

        //         printf("\n");

        //         printf("[%d] scatter_node_count: ", rank);
        //         for (int i = 0; i < size; ++i) {
        //             printf("%d ", (int)scatter_node_count[i]);
        //         }

        //         printf("\n");

        //         printf("[%d] scatter_node_displs: ", rank);
        //         for (int i = 0; i < size + 1; ++i) {
        //             printf("%d ", (int)scatter_node_displs[i]);
        //         }

        //         printf("\n");

        //         printf("\n-----------------------\n");

        //         fflush(stdout);
        //     }

        //     sleep(0.1);

        //     MPI_Barrier(comm);
        // }

        // MPI_Barrier(comm);

        // printf("\n");
        // fflush(stdout);

        // for (ptrdiff_t i = 0; i < size_send_list; ++i) {
        //     if (send_list[i] < input_node_partitions[rank] || send_list[i] >= input_node_partitions[rank + 1]) {
        //         printf("[%d] %ld <= %ld < %ld\n",
        //                (int)rank,
        //                (long)input_node_partitions[rank],
        //                (long)send_list[i],
        //                (long)input_node_partitions[rank + 1]);
        //     }

        //     assert(send_list[i] >= input_node_partitions[rank]);
        //     assert(send_list[i] < input_node_partitions[rank + 1]);
        // }

        ///////////////////////////////////////////////////////////////////////

        // Remove offset
        for (ptrdiff_t i = 0; i < size_send_list; ++i) {
            send_list[i] -= input_node_partitions[rank];
        }

        ///////////////////////////////////////////////////////////////////////
        // echange points

        geom_t *sendx = (geom_t *)malloc(size_send_list * sizeof(geom_t));
        geom_t **part_xyz = (geom_t **)malloc(sizeof(geom_t *) * ndims);

        for (int d = 0; d < ndims; ++d) {
            // Fill buffer
            for (ptrdiff_t i = 0; i < size_send_list; ++i) {
                sendx[i] = xyz[d][send_list[i]];
            }

            geom_t *recvx = (geom_t *)malloc(n_unique * sizeof(geom_t));
            CATCH_MPI_ERROR(MPI_Alltoallv(sendx,
                                          scatter_node_count,
                                          scatter_node_displs,
                                          SFEM_MPI_GEOM_T,
                                          recvx,
                                          gather_node_count,
                                          gather_node_displs,
                                          SFEM_MPI_GEOM_T,
                                          comm));
            part_xyz[d] = recvx;

            // Free space
            free(xyz[d]);
        }

        ///////////////////////////////////////////////////////////////////////
        // Localize element index
        idx_t *local_offset = (idx_t *)malloc((size + 1) * sizeof(idx_t));
        memset(local_offset, 0, (size + 1) * sizeof(idx_t));

        for (ptrdiff_t i = 0; i < n_unique; ++i) {
            const int j = owner_rank[i];
            assert(j < size);
            assert(j >= 0);

            local_offset[j + 1] += 1;
        }

        for (ptrdiff_t i = 0; i < size; ++i) {
            local_offset[i + 1] += local_offset[i];
        }

        // for (ptrdiff_t d = 0; d < nnodesxelem; ++d) {
        //     idx_t *node_d = elems[d];

        //     for (ptrdiff_t e = 0; e < n_local_elements; ++e) {
        //         idx_t idx = node_d[e];
        //         ptrdiff_t owner = MIN(size - 1, idx / n_local_nodes);

        //         if (input_node_partitions[owner] > idx) {
        //             assert(owner < size);
        //             while (input_node_partitions[--owner] > idx) {
        //                 assert(owner < size);
        //             }
        //         } else if (input_node_partitions[owner + 1] < idx) {
        //             assert(owner < size);

        //             while (input_node_partitions[(++owner + 1)] < idx) {
        //                 assert(owner < size);
        //             }
        //         }

        //         // Remove global offset
        //         node_d[e] -= input_node_partitions[owner];
        //         node_d[e] += local_offset[owner];
        //     }
        // }

        ///////////////////////////////////////////////////////////////////////
        MPI_Barrier(comm);
        for (int r = 0; r < size; ++r) {
            if (r == rank) {
                printf("[%d] elements\n", rank);

                for (ptrdiff_t e = 0; e < n_local_elements; ++e) {
                    for (ptrdiff_t d = 0; d < nnodesxelem; ++d) {
                        printf("%d ", (int)elems[d][e]);
                    }

                    printf("\n");
                }
                printf("\noffset\n");

                for (ptrdiff_t i = 0; i < size + 1; ++i) {
                    printf("%d ", (int)local_offset[i]);
                }

                printf("\nparts\n");

                for (ptrdiff_t i = 0; i < size + 1; ++i) {
                    printf("(%d), ", (int)input_node_partitions[i]);
                }

                printf("\n");
                printf("------------------\n");

                fflush(stdout);
            }

            sleep(0.1);

            MPI_Barrier(comm);
        }

        ///////////////////////////////////////////////////////////////////////
        // Free space
        free(unique_idx);
        free(sendx);
        free(send_list);
        free(input_node_partitions);

        free(scatter_node_count);
        free(gather_node_count);
        free(gather_node_displs);
        free(scatter_node_displs);

        ///////////////////////////////////////////////////////////////////////

        mesh->elements = elems;
        mesh->points = part_xyz;
        mesh->nelements = n_local_elements;
        mesh->nnodes = n_local_nodes;

        double tock = MPI_Wtime();
        return 0;
    } else {
        // Serial fallback
        mesh->comm = comm;

        mesh->mem_space = SFEM_MEM_SPACE_HOST;

        mesh->spatial_dim = 3;
        mesh->element_type = 4;

        mesh->elements = (idx_t **)malloc(4 * sizeof(idx_t *));
        mesh->points = (geom_t **)malloc(3 * sizeof(geom_t *));

        return serial_read_tet_mesh(folder, &mesh->nelements, mesh->elements, &mesh->nnodes, mesh->points);
    }
}

static ptrdiff_t read_file(MPI_Comm comm, const char *path, void **data) {
    MPI_Status status;
    MPI_Offset nbytes;
    MPI_File file;
    CATCH_MPI_ERROR(MPI_File_open(comm, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &file));
    CATCH_MPI_ERROR(MPI_File_get_size(file, &nbytes));
    *data = malloc(nbytes);

    CATCH_MPI_ERROR(MPI_File_read_at_all(file, 0, *data, nbytes, MPI_CHAR, &status));
    return nbytes;
}

int serial_read_tet_mesh(const char *folder, ptrdiff_t *nelements, idx_t *elems[4], ptrdiff_t *nnodes, geom_t *xyz[3]) {
    char path[1024 * 10];

    {
        sprintf(path, "%s/x.raw", folder);
        ptrdiff_t x_nnodes = read_file(MPI_COMM_SELF, path, (void **)&xyz[0]);

        sprintf(path, "%s/y.raw", folder);
        ptrdiff_t y_nnodes = read_file(MPI_COMM_SELF, path, (void **)&xyz[1]);

        sprintf(path, "%s/z.raw", folder);
        ptrdiff_t z_nnodes = read_file(MPI_COMM_SELF, path, (void **)&xyz[2]);

        assert(x_nnodes == y_nnodes);
        assert(x_nnodes == z_nnodes);

        x_nnodes /= sizeof(geom_t);
        assert(x_nnodes * sizeof(geom_t) == y_nnodes);
        *nnodes = x_nnodes;
    }

    {
        sprintf(path, "%s/i0.raw", folder);
        ptrdiff_t nindex0 = read_file(MPI_COMM_SELF, path, (void **)&elems[0]);

        sprintf(path, "%s/i1.raw", folder);
        ptrdiff_t nindex1 = read_file(MPI_COMM_SELF, path, (void **)&elems[1]);

        sprintf(path, "%s/i2.raw", folder);
        ptrdiff_t nindex2 = read_file(MPI_COMM_SELF, path, (void **)&elems[2]);

        sprintf(path, "%s/i3.raw", folder);
        ptrdiff_t nindex3 = read_file(MPI_COMM_SELF, path, (void **)&elems[3]);

        assert(nindex0 == nindex1);
        assert(nindex3 == nindex2);

        nindex0 /= sizeof(idx_t);
        assert(nindex0 * sizeof(idx_t) == nindex1);
        *nelements = nindex0;
    }

    return 0;
}
