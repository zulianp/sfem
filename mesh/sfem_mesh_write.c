#include "sfem_mesh_write.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <stdint.h>
#include <unistd.h>

#include <assert.h>
#include <stdio.h>
#include <string.h>

// matrix.io
#include "matrixio_array.h"
#include "utils.h"

int mesh_write(const char *path, const mesh_t *mesh) {
    // TODO
    MPI_Comm comm = mesh->comm;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    static const char *str_xyz = "xyzt";

    char output_path[2048];

    struct stat st = {0};
    if (stat(path, &st) == -1) {
        mkdir(path, 0700);
    }

    if (size == 1) {
        for (int d = 0; d < mesh->spatial_dim; ++d) {
            sprintf(output_path, "%s/%c.raw", path, str_xyz[d]);
            array_write(comm, output_path, SFEM_MPI_GEOM_T, mesh->points[d], mesh->nnodes, mesh->nnodes);
        }

        for (int d = 0; d < mesh->element_type; ++d) {
            sprintf(output_path, "%s/i%d.raw", path, d);
            array_write(comm, output_path, SFEM_MPI_IDX_T, mesh->elements[d], mesh->nelements, mesh->nelements);
        }

        if (mesh->node_mapping) {
            sprintf(output_path, "%s/node_mapping.raw", path);
            array_write(comm, output_path, SFEM_MPI_IDX_T, mesh->node_mapping, mesh->nnodes, mesh->nnodes);
        }

        if (mesh->element_mapping) {
            sprintf(output_path, "%s/element_mapping.raw", path);
            array_write(comm, output_path, SFEM_MPI_IDX_T, mesh->element_mapping, mesh->nelements, mesh->nelements);
        }

        return 0;
    } else {
        // TODO
        assert(0);
        return 1;
    }
}

int mesh_write_nodal_field(const mesh_t *const mesh, const char *path, MPI_Datatype data_type, const void *const data) {
    count_t n_global_nodes = mesh->n_owned_nodes;
    CATCH_MPI_ERROR(MPI_Allreduce(MPI_IN_PLACE, &n_global_nodes, 1, SFEM_MPI_COUNT_T, MPI_SUM, mesh->comm));

    if (!mesh->node_mapping) {
#ifndef NDEBUG
        int size;
        MPI_Comm_size(mesh->comm, &size);
        assert(size == 1);
#endif
        return array_write(mesh->comm, path, data_type, data, mesh->n_owned_nodes, n_global_nodes);

    } else {
        return write_mapped_field(
            mesh->comm, path, mesh->n_owned_nodes, n_global_nodes, mesh->node_mapping, data_type, data);
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
    MPI_Comm_size(comm, &rank);
    MPI_Comm_size(comm, &size);

    const uint8_t *const data = (const uint8_t *const)data_in;

    int type_size;
    CATCH_MPI_ERROR(MPI_Type_size(data_type, &type_size));

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
        int dest_rank = idx / local_output_size_no_remainder;
        assert(dest_rank < size);
        send_count[dest_rank]++;
    }

    idx_t *recv_count = (idx_t *)malloc((size) * sizeof(idx_t));
    CATCH_MPI_ERROR(MPI_Alltoall(send_count, 1, SFEM_MPI_IDX_T, recv_count, 1, SFEM_MPI_IDX_T, comm));

    int *send_displs = (int *)malloc(size * sizeof(int));
    int *recv_displs = (int *)malloc(size * sizeof(int));
    count_t *book_keeping = (count_t *)malloc(size * sizeof(count_t));
    memset(book_keeping, 0, size * sizeof(count_t));

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

    idx_t *send_list = (idx_t *)malloc(n_local * sizeof(idx_t));

    ptrdiff_t n_buff = MAX(n_local, local_output_size);
    uint8_t *send_data_and_final_storage = (uint8_t *)malloc(n_buff * type_size);

    // Pack data and indices
    for (ptrdiff_t i = 0; i < n_local; ++i) {
        const idx_t idx = mapping[i];
        int dest_rank = idx / local_output_size_no_remainder;
        assert(dest_rank < size);

        // Put index and data into buffers
        const ptrdiff_t offset = book_keeping[dest_rank];
        send_list[offset] = idx;
        memcpy((void *)&send_data_and_final_storage[offset * type_size], (void *)&data[i * type_size], type_size);

        book_keeping[dest_rank]++;
    }

    idx_t *recv_list = (idx_t *)malloc(local_output_size * sizeof(idx_t));
    uint8_t *recv_data = (uint8_t *)malloc(local_output_size * type_size);

    ///////////////////////////////////
    // Send indices
    ///////////////////////////////////

    CATCH_MPI_ERROR(MPI_Alltoallv(
        send_list, send_count, send_displs, SFEM_MPI_IDX_T, recv_list, recv_count, recv_displs, SFEM_MPI_IDX_T, comm));

    ///////////////////////////////////
    // Send data
    ///////////////////////////////////

    CATCH_MPI_ERROR(MPI_Alltoallv(send_data_and_final_storage,
                                  send_count,
                                  send_displs,
                                  data_type,
                                  recv_data,
                                  recv_count,
                                  recv_displs,
                                  data_type,
                                  comm));

    ///////////////////////////////////
    // Unpack indexed data
    ///////////////////////////////////

    for (ptrdiff_t i = 0; i < local_output_size; ++i) {
        ptrdiff_t dest = recv_list[i] - begin;

        assert(dest >= 0);
        assert(dest < local_output_size);
        memcpy((void *)&send_data_and_final_storage[dest * type_size], (void *)&recv_data[i * type_size], type_size);
    }

    array_write(comm, output_path, data_type, (void *)send_data_and_final_storage, local_output_size, n_global);

    ///////////////////////////////////
    // Clean-up
    ///////////////////////////////////
    free(send_count);
    free(send_displs);
    free(recv_displs);
    free(book_keeping);
    free(send_list);
    free(recv_list);
    free(recv_data);
    free(send_data_and_final_storage);
    return 0;
}
