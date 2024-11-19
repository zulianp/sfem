#include "sfem_mesh_write.h"

#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <assert.h>
#include <stdio.h>
#include <string.h>

// matrix.io
#include "matrixio_array.h"
#include "utils.h"

#include "sfem_defs.h"

int mesh_write(const char *path, const mesh_t *mesh) {
    // TODO
    // MPI_Comm comm = mesh->comm;
    MPI_Comm comm = MPI_COMM_SELF;

    int rank, size;
    MPI_Comm_rank(mesh->comm, &rank);
    MPI_Comm_size(mesh->comm, &size);

    static const char *str_xyz = "xyzt";

    char folder[2048];
    char output_path[2048];

    {
        struct stat st = {0};
        if (stat(path, &st) == -1) {
            mkdir(path, 0700);
        }
    }

    ptrdiff_t nelements;
    if (size > 1) {
        sprintf(folder, "%s/%d", path, rank);
        nelements = mesh->n_owned_elements;
    } else {
        sprintf(folder, "%s", path);
        nelements = mesh->nelements;
    }

    {
        struct stat st = {0};
        if (stat(folder, &st) == -1) {
            mkdir(folder, 0700);
        }
    }

    if (!rank) {
        printf("Writing mesh in %s\n", folder);
    }

    // if (size == 1) {
    for (int d = 0; d < mesh->spatial_dim; ++d) {
        sprintf(output_path, "%s/%c.raw", folder, str_xyz[d]);
        array_write(
            comm, output_path, SFEM_MPI_GEOM_T, mesh->points[d], mesh->nnodes, mesh->nnodes);
    }

    const int nxe = elem_num_nodes(mesh->element_type);
    for (int d = 0; d < nxe; ++d) {
        sprintf(output_path, "%s/i%d.raw", folder, d);
        array_write(comm, output_path, SFEM_MPI_IDX_T, mesh->elements[d], nelements, nelements);
    }

    if (mesh->node_mapping) {
        sprintf(output_path, "%s/node_mapping.raw", folder);
        array_write(
            comm, output_path, SFEM_MPI_IDX_T, mesh->node_mapping, mesh->nnodes, mesh->nnodes);
    }

    if (mesh->element_mapping) {
        sprintf(output_path, "%s/element_mapping.raw", folder);
        array_write(comm, output_path, SFEM_MPI_IDX_T, mesh->element_mapping, nelements, nelements);
    }

    if (mesh->node_owner) {
        sprintf(output_path, "%s/node_owner.raw", folder);
        array_write(
            comm, output_path, SFEM_MPI_IDX_T, mesh->node_owner, mesh->nnodes, mesh->nnodes);
    }

    return 0;
    // } else {
    //     // TODO
    //     assert(0);
    //     return 1;
    // }
}

int mesh_write_nodal_field(const mesh_t *const mesh,
                           const char *path,
                           MPI_Datatype data_type,
                           const void *const data) {


                            // get MPI rank 
    int mpi_rank;
    MPI_Comm_rank(mesh->comm, &mpi_rank);

    count_t n_global_nodes = mesh->n_owned_nodes;
    MPI_CATCH_ERROR(
        MPI_Allreduce(MPI_IN_PLACE, &n_global_nodes, 1, SFEM_MPI_COUNT_T, MPI_SUM, mesh->comm));

    if (!mesh->node_mapping) {
#ifndef NDEBUG
        int size;
        MPI_Comm_size(mesh->comm, &size);
        assert(size == 1);
#endif

        if(mpi_rank == 0) printf("%s:%d: Writing using array_write\n", __FILE__, __LINE__);
        
        return array_write(mesh->comm, path, data_type, data, mesh->n_owned_nodes, n_global_nodes);

    } else {

        if (mpi_rank == 0) printf("%s:%d: Writing using write_mapped_field\n", __FILE__, __LINE__);
        
        return write_mapped_field(mesh->comm,
                                  path,
                                  mesh->n_owned_nodes,
                                  n_global_nodes,
                                  mesh->node_mapping,
                                  data_type,
                                  data);
    }
}

int write_mapped_field(MPI_Comm comm,
                       const char *output_path,
                       const ptrdiff_t n_local,
                       const ptrdiff_t n_global,
                       const idx_t *const mapping,
                       MPI_Datatype data_type,
                       const void *const data_in) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const uint8_t *const data = (const uint8_t *const)data_in;

    int type_size;
    MPI_CATCH_ERROR(MPI_Type_size(data_type, &type_size));

    const ptrdiff_t local_output_size_no_remainder = n_global / size;
    const ptrdiff_t begin = (n_global / size) * rank;

    ptrdiff_t local_output_size = local_output_size_no_remainder;
    if (rank == size - 1) {
        local_output_size = n_global - begin;
    }

    idx_t *send_count = (idx_t *)malloc((size) * sizeof(idx_t));
    memset(send_count, 0, (size) * sizeof(idx_t));

    for (ptrdiff_t i = 0; i < n_local; ++i) {
        const idx_t idx = mapping[i];
        int dest_rank = MIN(size - 1, idx / local_output_size_no_remainder);
        send_count[dest_rank]++;
    }

    idx_t *recv_count = (idx_t *)malloc((size) * sizeof(idx_t));
    MPI_CATCH_ERROR(
        MPI_Alltoall(send_count, 1, SFEM_MPI_IDX_T, recv_count, 1, SFEM_MPI_IDX_T, comm));

    int *send_displs = (int *)malloc(size * sizeof(int));
    int *recv_displs = (int *)malloc(size * sizeof(int));
    count_t *book_keeping = (count_t *)calloc(size, sizeof(count_t));

    send_displs[0] = 0;
    recv_displs[0] = 0;

    // Create data displacements for sending
    for (int i = 0; i < size - 1; ++i) {
        send_displs[i + 1] = send_displs[i] + send_count[i];
    }

    // Create data displacements for receiving
    for (int i = 0; i < size - 1; ++i) {
        recv_displs[i + 1] = recv_displs[i] + recv_count[i];
    }

    const ptrdiff_t total_recv = recv_displs[size - 1] + recv_count[size - 1];

    idx_t *send_list = (idx_t *)malloc(n_local * sizeof(idx_t));

    ptrdiff_t n_buff = MAX(n_local, local_output_size);
    uint8_t *send_data_and_final_storage = (uint8_t *)malloc(n_buff * type_size);

    // Pack data and indices
    for (ptrdiff_t i = 0; i < n_local; ++i) {
        const idx_t idx = mapping[i];
        int dest_rank = MIN(size - 1, idx / local_output_size_no_remainder);
        assert(dest_rank < size);

        // Put index and data into buffers
        const ptrdiff_t offset = send_displs[dest_rank] + book_keeping[dest_rank];
        send_list[offset] = idx;
        memcpy((void *)&send_data_and_final_storage[offset * type_size],
               (void *)&data[i * type_size],
               type_size);

        book_keeping[dest_rank]++;
    }

    idx_t *recv_list = (idx_t *)malloc(local_output_size * sizeof(idx_t));
    uint8_t *recv_data = (uint8_t *)malloc(local_output_size * type_size);

    ///////////////////////////////////
    // Send indices
    ///////////////////////////////////

    MPI_CATCH_ERROR(MPI_Alltoallv(send_list,
                                  send_count,
                                  send_displs,
                                  SFEM_MPI_IDX_T,
                                  recv_list,
                                  recv_count,
                                  recv_displs,
                                  SFEM_MPI_IDX_T,
                                  comm));

    ///////////////////////////////////
    // Send data
    ///////////////////////////////////

    MPI_CATCH_ERROR(MPI_Alltoallv(send_data_and_final_storage,
                                  send_count,
                                  send_displs,
                                  data_type,
                                  recv_data,
                                  recv_count,
                                  recv_displs,
                                  data_type,
                                  comm));

    if (0) {
        for (int r = 0; r < size; r++) {
            MPI_Barrier(comm);

            if (r == rank) {
                printf("[%d]\n", rank);
                printf("\nsend_count\n");
                for (int i = 0; i < size; i++) {
                    printf("%d ", send_count[i]);
                }

                printf("\nsend_displs\n");
                for (int i = 0; i < size; i++) {
                    printf("%d ", send_displs[i]);
                }

                printf("\nrecv_count\n");
                for (int i = 0; i < size; i++) {
                    printf("%d ", recv_count[i]);
                }

                printf("\nrecv_displs\n");
                for (int i = 0; i < size; i++) {
                    printf("%d ", recv_displs[i]);
                }

                printf("\n");

                idx_t min_idx = mapping[0];
                idx_t max_idx = mapping[0];
                for (ptrdiff_t i = 0; i < n_local; ++i) {
                    const idx_t idx = mapping[i];
                    min_idx = MIN(min_idx, idx);
                    max_idx = MAX(max_idx, idx);
                }

                printf("[%d, %d]\n", min_idx, max_idx);
                printf("%ld == %ld\n", total_recv, local_output_size);

                for (ptrdiff_t recv_rank = 0; recv_rank < size; ++recv_rank) {
                    if (recv_rank != rank) {
                        for (int i = 0; i < send_count[recv_rank]; i++) {
                            printf("%d ", (int)send_list[send_displs[recv_rank] + i]);
                        }
                    }
                }

                printf("\n");

                for (ptrdiff_t i = 0; i < local_output_size; ++i) {
                    ptrdiff_t dest = recv_list[i] - begin;

                    if (dest < 0 || dest >= local_output_size) {
                        printf("%d not in [%ld, %ld)\n",
                               recv_list[i],
                               begin,
                               begin + local_output_size);
                    }
                }

                fflush(stdout);
            }

            MPI_Barrier(comm);
        }
    }

    ///////////////////////////////////
    // Unpack indexed data
    ///////////////////////////////////

    for (ptrdiff_t i = 0; i < local_output_size; ++i) {
        ptrdiff_t dest = recv_list[i] - begin;
        assert(dest >= 0);
        assert(dest < local_output_size);
        memcpy((void *)&send_data_and_final_storage[dest * type_size],
               (void *)&recv_data[i * type_size],
               type_size);
    }

    array_write(comm,
                output_path,
                data_type,
                (void *)send_data_and_final_storage,
                local_output_size,
                n_global);

    ///////////////////////////////////
    // Clean-up
    ///////////////////////////////////
    free(send_count);
    free(send_displs);
    free(recv_count);
    free(recv_displs);
    free(book_keeping);
    free(send_list);
    free(recv_list);
    free(recv_data);
    free(send_data_and_final_storage);
    return 0;
}
