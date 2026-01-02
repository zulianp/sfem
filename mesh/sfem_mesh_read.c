// Read a globally stored field into a local buffer using a mapping of local entries -> global indices.
// This is the read-side analogue of write_mapped_field (implemented in sfem_mesh_write.c).
#include "sfem_mesh_read.h"

#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "matrixio_array.h"
#include "sfem_base.h"
#include "sfem_macros.h"
#include "utils.h"

// int array_read(MPI_Comm     comm,    //
//                const char  *path,    //
//                MPI_Datatype type,    //
//                void        *data,    //
//                ptrdiff_t    nlocal,  //
//                ptrdiff_t    nglobal) {  //

//     if (nglobal >= (ptrdiff_t)INT_MAX) {
//         // Comunication free fallback by exploiting global information
//         return array_read_segmented(comm, path, type, data, INT_MAX, nlocal, nglobal);
//     }

//     int mpi_rank, mpi_size;
//     MPI_Comm_rank(comm, &mpi_rank);
//     MPI_Comm_size(comm, &mpi_size);

//     assert(nlocal <= nglobal);

//     MPI_Status status;
//     MPI_Offset nbytes;
//     MPI_File   file;
//     int        type_size;
// }

int                                               //
read_mapped_field(MPI_Comm           comm,        //
                  const char        *input_path,  //
                  const ptrdiff_t    n_local,     //
                  const ptrdiff_t    n_global,    //
                  const idx_t *const mapping,     //
                  MPI_Datatype       data_type,   //
                  void *const        data_out) {         //
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    uint8_t *const out = (uint8_t *const)data_out;

    int type_size = 0;
    MPI_CATCH_ERROR(MPI_Type_size(data_type, &type_size));

    const ptrdiff_t local_size_no_remainder = n_global / size;
    const ptrdiff_t begin                   = (n_global / size) * rank;

    ptrdiff_t local_size = local_size_no_remainder;
    if (rank == size - 1) {
        local_size = n_global - begin;
    }

    // Read this rank's contiguous chunk of the global field
    uint8_t *local_chunk = NULL;
    if (local_size > 0) {
        local_chunk = (uint8_t *)malloc((size_t)local_size * (size_t)type_size);
        if (!local_chunk) return SFEM_FAILURE;
    }

    int err = array_read(comm, input_path, data_type, (void *)local_chunk, local_size, n_global);
    if (err) {
        free(local_chunk);
        return err;
    }

    // Build request counts: how many global indices we need from each rank
    int *req_count = (int *)calloc((size_t)size, sizeof(int));
    if (!req_count) {
        free(local_chunk);
        return SFEM_FAILURE;
    }

    for (ptrdiff_t i = 0; i < n_local; ++i) {
        const idx_t idx = mapping[i];

        int src_rank = 0;
        if (local_size_no_remainder > 0) {
            src_rank = MIN(size - 1, (int)(idx / local_size_no_remainder));
        } else {
            // Degenerate distribution (n_global < size): everything lives on last rank.
            src_rank = size - 1;
        }

        req_count[src_rank]++;
    }

    int *recv_req_count = (int *)malloc((size_t)size * sizeof(int));
    if (!recv_req_count) {
        free(req_count);
        free(local_chunk);
        return SFEM_FAILURE;
    }

    MPI_CATCH_ERROR(MPI_Alltoall(req_count, 1, MPI_INT32_T, recv_req_count, 1, MPI_INT32_T, comm));

    int *req_displs  = (int *)malloc((size_t)size * sizeof(int));
    int *recv_displs = (int *)malloc((size_t)size * sizeof(int));
    if (!req_displs || !recv_displs) {
        free(req_displs);
        free(recv_displs);
        free(recv_req_count);
        free(req_count);
        free(local_chunk);
        return SFEM_FAILURE;
    }

    req_displs[0]  = 0;
    recv_displs[0] = 0;
    for (int i = 0; i < size - 1; ++i) {
        req_displs[i + 1]  = req_displs[i] + req_count[i];
        recv_displs[i + 1] = recv_displs[i] + recv_req_count[i];
    }

    const ptrdiff_t total_recv_req = (ptrdiff_t)recv_displs[size - 1] + (ptrdiff_t)recv_req_count[size - 1];

    // Pack requests: global indices + local positions (for unpack)
    idx_t     *req_list  = NULL;
    ptrdiff_t *local_pos = NULL;
    if (n_local > 0) {
        req_list  = (idx_t *)malloc((size_t)n_local * sizeof(idx_t));
        local_pos = (ptrdiff_t *)malloc((size_t)n_local * sizeof(ptrdiff_t));
    }

    count_t *book_keeping = (count_t *)calloc((size_t)size, sizeof(count_t));
    if ((n_local > 0 && (!req_list || !local_pos)) || !book_keeping) {
        free(book_keeping);
        free(local_pos);
        free(req_list);
        free(recv_displs);
        free(req_displs);
        free(recv_req_count);
        free(req_count);
        free(local_chunk);
        return SFEM_FAILURE;
    }

    for (ptrdiff_t i = 0; i < n_local; ++i) {
        const idx_t idx = mapping[i];

        int src_rank = 0;
        if (local_size_no_remainder > 0) {
            src_rank = MIN(size - 1, (int)(idx / local_size_no_remainder));
        } else {
            src_rank = size - 1;
        }

        const ptrdiff_t off = (ptrdiff_t)req_displs[src_rank] + (ptrdiff_t)book_keeping[src_rank];
        req_list[off]       = idx;
        local_pos[off]      = i;
        book_keeping[src_rank]++;
    }

    idx_t *recv_req_list = NULL;
    if (total_recv_req > 0) {
        recv_req_list = (idx_t *)malloc((size_t)total_recv_req * sizeof(idx_t));
        if (!recv_req_list) {
            free(book_keeping);
            free(local_pos);
            free(req_list);
            free(recv_displs);
            free(req_displs);
            free(recv_req_count);
            free(req_count);
            free(local_chunk);
            return SFEM_FAILURE;
        }
    }

    // Exchange requested indices
    MPI_CATCH_ERROR(MPI_Alltoallv(
            req_list, req_count, req_displs, SFEM_MPI_IDX_T, recv_req_list, recv_req_count, recv_displs, SFEM_MPI_IDX_T, comm));

    // Build response buffer for received requests (same ordering as recv_req_list)
    uint8_t *send_resp = NULL;
    if (total_recv_req > 0) {
        send_resp = (uint8_t *)malloc((size_t)total_recv_req * (size_t)type_size);
        if (!send_resp) {
            free(recv_req_list);
            free(book_keeping);
            free(local_pos);
            free(req_list);
            free(recv_displs);
            free(req_displs);
            free(recv_req_count);
            free(req_count);
            free(local_chunk);
            return SFEM_FAILURE;
        }
    }

    for (ptrdiff_t i = 0; i < total_recv_req; ++i) {
        const idx_t     idx = recv_req_list[i];
        const ptrdiff_t loc = (ptrdiff_t)idx - begin;
        assert(loc >= 0);
        assert(loc < local_size);
        memcpy((void *)(send_resp + i * type_size), (const void *)(local_chunk + loc * type_size), (size_t)type_size);
    }

    // Exchange response data back to requesters
    uint8_t *recv_resp = NULL;
    if (n_local > 0) {
        recv_resp = (uint8_t *)malloc((size_t)n_local * (size_t)type_size);
        if (!recv_resp) {
            free(send_resp);
            free(recv_req_list);
            free(book_keeping);
            free(local_pos);
            free(req_list);
            free(recv_displs);
            free(req_displs);
            free(recv_req_count);
            free(req_count);
            free(local_chunk);
            return SFEM_FAILURE;
        }
    }

    MPI_CATCH_ERROR(
            MPI_Alltoallv(send_resp, recv_req_count, recv_displs, data_type, recv_resp, req_count, req_displs, data_type, comm));

    // Unpack into local ordering
    for (ptrdiff_t off = 0; off < n_local; ++off) {
        const ptrdiff_t i = local_pos[off];
        memcpy((void *)(out + i * type_size), (const void *)(recv_resp + off * type_size), (size_t)type_size);
    }

    free(recv_resp);
    free(send_resp);
    free(recv_req_list);
    free(book_keeping);
    free(local_pos);
    free(req_list);
    free(recv_displs);
    free(req_displs);
    free(recv_req_count);
    free(req_count);
    free(local_chunk);
    return SFEM_SUCCESS;
}

int mesh_read_nodal_field(const mesh_t *const mesh, const char *path, MPI_Datatype data_type, void *const data) {
    // get MPI rank
    int mpi_rank;
    MPI_Comm_rank(mesh->comm, &mpi_rank);

    count_t n_global_nodes = mesh->n_owned_nodes;
    MPI_CATCH_ERROR(MPI_Allreduce(MPI_IN_PLACE, &n_global_nodes, 1, SFEM_MPI_COUNT_T, MPI_SUM, mesh->comm));

    if (!mesh->node_mapping) {
#ifndef NDEBUG
        int size;
        MPI_Comm_size(mesh->comm, &size);
        assert(size == 1);
#endif

        if (mpi_rank == 0) printf("%s:%d: Reading using array_read\n", __FILE__, __LINE__);

        return array_read(mesh->comm, path, data_type, data, mesh->n_owned_nodes, n_global_nodes);

    } else {
        if (mpi_rank == 0) printf("%s:%d: Reading using read_mapped_field\n", __FILE__, __LINE__);

        return read_mapped_field(mesh->comm, path, mesh->n_owned_nodes, n_global_nodes, mesh->node_mapping, data_type, data);
    }
}
