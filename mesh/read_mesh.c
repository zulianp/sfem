
#include "read_mesh.h"

#include "matrixio_array.h"
#include "sfem_defs.h"
#include "utils.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
// #include <unistd.h>
#include <mpi.h>

#include "sortreduce.h"

// sfem_glob.hpp
size_t count_files(const char *pattern);

#define BARRIER_MSG(msg_)                \
    do {                                 \
        printf("[%d] %s\n", rank, msg_); \
        MPI_Barrier(MPI_COMM_WORLD);     \
    } while (0)

#define array_remap_scatter(n_, type_, mapping_, array_, temp_) \
    do {                                                        \
        type_ *temp_actual_ = (type_ *)temp_;                   \
        memcpy(temp_actual_, array_, n_ * sizeof(type_));       \
        for (ptrdiff_t i = 0; i < n_; ++i) {                    \
            array_[mapping_[i]] = temp_actual_[i];              \
        }                                                       \
    } while (0)

#define array_remap_gather(n_, type_, mapping_, array_, temp_) \
    do {                                                       \
        type_ *temp_actual_ = (type_ *)temp_;                  \
        memcpy(temp_actual_, array_, n_ * sizeof(type_));      \
        for (ptrdiff_t i = 0; i < n_; ++i) {                   \
            array_[i] = temp_actual_[mapping_[i]];             \
        }                                                      \
    } while (0)

int mesh_node_ids(MPI_Comm           comm,
                  const ptrdiff_t    n_nodes,
                  const ptrdiff_t    n_owned_nodes,
                  const idx_t *const node_offsets,
                  const idx_t *const ghosts,
                  idx_t *const       ids) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    ptrdiff_t n_owned_nodes_with_ghosts = node_offsets[rank + 1] - node_offsets[rank];

    for (ptrdiff_t i = 0; i < n_owned_nodes; i++) {
        ids[i] = node_offsets[rank] + i;
    }

    for (ptrdiff_t i = n_owned_nodes; i < n_nodes; i++) {
        ids[i] = ghosts[i - n_owned_nodes];
    }

    return 0;
}

static SFEM_INLINE int find_owner_rank(const idx_t                idx,
                                       const ptrdiff_t            n_local_nodes,
                                       const int                  size,
                                       const idx_t *SFEM_RESTRICT input_node_partitions) {
    int owner = MIN(size - 1, idx / n_local_nodes);
    int guess = owner;
    assert(owner >= 0);
    assert(owner < size);

    if (idx == input_node_partitions[owner]) {
        // Do nothing
    } else if (input_node_partitions[owner + 1] <= idx) {
        while (input_node_partitions[owner + 1] <= idx) {
            owner++;
            assert(owner < size);
        }
    } else if (input_node_partitions[owner] > idx) {
        while (input_node_partitions[owner] > idx) {
            --owner;
            assert(owner >= 0);
        }
    }

    return owner;
}

int mesh_build_global_ids(MPI_Comm        comm,
                          const ptrdiff_t n_nodes,
                          const ptrdiff_t n_owned_nodes,
                          const ptrdiff_t n_owned_nodes_with_ghosts,
                          idx_t          *node_mapping,
                          int            *node_owner,
                          idx_t         **node_offsets_out,
                          idx_t         **ghosts_out,
                          ptrdiff_t      *n_owned_nodes_with_ghosts_out) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    long n_gnodes           = n_owned_nodes;
    long global_node_offset = 0;
    MPI_Exscan(&n_gnodes, &global_node_offset, 1, MPI_LONG, MPI_SUM, comm);

    n_gnodes = global_node_offset + n_owned_nodes;
    MPI_Bcast(&n_gnodes, 1, MPI_LONG, size - 1, comm);

    idx_t *node_offsets = malloc((size + 1) * sizeof(idx_t));
    MPI_CATCH_ERROR(MPI_Allgather(&global_node_offset, 1, SFEM_MPI_IDX_T, node_offsets, 1, SFEM_MPI_IDX_T, comm));

    node_offsets[size] = n_gnodes;

    const ptrdiff_t n_lnodes_no_reminder = n_gnodes / size;
    const ptrdiff_t begin                = n_lnodes_no_reminder * rank;

    ptrdiff_t n_lnodes_temp = n_lnodes_no_reminder;
    if (rank == size - 1) {
        n_lnodes_temp = n_gnodes - begin;
    }

    const ptrdiff_t begin_owned_with_ghosts  = n_owned_nodes - n_owned_nodes_with_ghosts;
    const ptrdiff_t extent_owned_with_ghosts = n_owned_nodes_with_ghosts;
    const ptrdiff_t n_ghost_nodes            = n_nodes - n_owned_nodes;

    idx_t *ghost_keys = &node_mapping[begin_owned_with_ghosts];
    idx_t *ghost_ids  = malloc(MAX(extent_owned_with_ghosts, n_ghost_nodes) * sizeof(idx_t));

    int *send_displs = (int *)malloc((size + 1) * sizeof(int));
    int *recv_displs = (int *)malloc((size + 1) * sizeof(int));
    int *send_count  = (int *)calloc(size, sizeof(int));
    int *recv_count  = (int *)malloc(size * sizeof(int));

    for (ptrdiff_t i = 0; i < extent_owned_with_ghosts; i++) {
        ghost_ids[i] = global_node_offset + begin_owned_with_ghosts + i;
        assert(ghost_ids[i] < n_gnodes);
    }

    for (ptrdiff_t i = 0; i < extent_owned_with_ghosts; i++) {
        const idx_t idx       = ghost_keys[i];
        int         dest_rank = MIN(idx / n_lnodes_no_reminder, size - 1);
        assert(dest_rank < size);
        assert(dest_rank >= 0);

        send_count[dest_rank]++;
    }

    MPI_CATCH_ERROR(MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm));

    send_displs[0] = 0;
    for (int r = 0; r < size; r++) {
        send_displs[r + 1] = send_displs[r] + send_count[r];
    }

    recv_displs[0] = 0;
    for (int r = 0; r < size; r++) {
        recv_displs[r + 1] = recv_displs[r] + recv_count[r];
    }

    idx_t *recv_key_buff = malloc(recv_displs[size] * sizeof(idx_t));

    MPI_CATCH_ERROR(MPI_Alltoallv(
            ghost_keys, send_count, send_displs, SFEM_MPI_IDX_T, recv_key_buff, recv_count, recv_displs, SFEM_MPI_IDX_T, comm));

    idx_t *recv_ids_buff = (idx_t *)malloc(recv_displs[size] * sizeof(idx_t));

    MPI_CATCH_ERROR(MPI_Alltoallv(
            ghost_ids, send_count, send_displs, SFEM_MPI_IDX_T, recv_ids_buff, recv_count, recv_displs, SFEM_MPI_IDX_T, comm));

    idx_t *mapping = malloc(n_lnodes_temp * sizeof(idx_t));

#ifndef NDEBUG
    for (ptrdiff_t i = 0; i < n_lnodes_temp; i++) {
        mapping[i] = SFEM_IDX_INVALID;
    }
#endif

    // Fill mapping
    for (int r = 0; r < size; r++) {
        int proc_begin  = recv_displs[r];
        int proc_extent = recv_count[r];

        idx_t *keys = &recv_key_buff[proc_begin];
        idx_t *ids  = &recv_ids_buff[proc_begin];

        for (int k = 0; k < proc_extent; k++) {
            idx_t iii = keys[k] - begin;

            assert(iii >= 0);
            assert(iii < n_lnodes_temp);

            mapping[iii] = ids[k];
        }
    }

    /////////////////////////////////////////////////
    // Gather query ghost nodes
    memset(send_count, 0, size * sizeof(int));

    // Get the query nodes
    ghost_keys = &node_mapping[n_owned_nodes];

    idx_t *recv_idx      = (idx_t *)malloc(n_ghost_nodes * sizeof(idx_t));
    idx_t *exchange_buff = (idx_t *)malloc(n_ghost_nodes * sizeof(idx_t));
    {
        for (ptrdiff_t i = 0; i < n_ghost_nodes; i++) {
            const idx_t idx       = ghost_keys[i];
            int         dest_rank = MIN(idx / n_lnodes_no_reminder, size - 1);

            assert(dest_rank < size);
            assert(dest_rank >= 0);

            send_count[dest_rank]++;
        }

        send_displs[0] = 0;
        for (int r = 0; r < size; r++) {
            send_displs[r + 1] = send_displs[r] + send_count[r];
        }

        memset(send_count, 0, sizeof(int) * size);

        for (ptrdiff_t i = 0; i < n_ghost_nodes; i++) {
            const idx_t idx       = ghost_keys[i];
            int         dest_rank = MIN(idx / n_lnodes_no_reminder, size - 1);

            assert(dest_rank < size);
            assert(dest_rank >= 0);

            const idx_t offset    = send_displs[dest_rank] + send_count[dest_rank]++;
            exchange_buff[offset] = idx;
            recv_idx[offset]      = i;
        }
    }

    MPI_CATCH_ERROR(MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm));

    recv_displs[0] = 0;
    for (int r = 0; r < size; r++) {
        recv_displs[r + 1] = recv_displs[r] + recv_count[r];
    }

    recv_key_buff = realloc(recv_key_buff, recv_displs[size] * sizeof(idx_t));

    MPI_CATCH_ERROR(MPI_Alltoallv(exchange_buff,
                                  send_count,
                                  send_displs,
                                  SFEM_MPI_IDX_T,
                                  recv_key_buff,
                                  recv_count,
                                  recv_displs,
                                  SFEM_MPI_IDX_T,
                                  comm));

    // Query mapping
    for (int r = 0; r < size; r++) {
        int proc_begin  = recv_displs[r];
        int proc_extent = recv_count[r];

        idx_t *keys = &recv_key_buff[proc_begin];

        for (int k = 0; k < proc_extent; k++) {
            idx_t iii = keys[k] - begin;

            if (iii >= n_lnodes_temp) {
                printf("[%d] %ld < %d < %ld\n", rank, begin, keys[k], begin + n_lnodes_temp);
            }

            assert(iii < n_lnodes_temp);
            assert(iii >= 0);
            assert(mapping[iii] >= 0);
            keys[k] = mapping[iii];
        }
    }

    // Send back
    MPI_CATCH_ERROR(MPI_Alltoallv(recv_key_buff,
                                  recv_count,
                                  recv_displs,
                                  SFEM_MPI_IDX_T,
                                  exchange_buff,
                                  send_count,
                                  send_displs,
                                  SFEM_MPI_IDX_T,
                                  comm));

    idx_t *ghosts = malloc(n_ghost_nodes * sizeof(idx_t));
    for (ptrdiff_t i = 0; i < n_ghost_nodes; i++) {
        ghosts[recv_idx[i]] = exchange_buff[i];
    }

    free(recv_idx);
    free(exchange_buff);

    /////////////////////////////////////////////////

    *node_offsets_out = node_offsets;
    *ghosts_out       = ghosts;

    // Clean-up
    free(send_count);
    free(recv_count);
    free(send_displs);
    free(recv_displs);
    free(recv_key_buff);
    free(recv_ids_buff);
    free(mapping);
    free(ghost_ids);

    return SFEM_SUCCESS;
}

int mesh_read_elements(MPI_Comm         comm,
                       const int        nnodesxelem,
                       const char      *folder,
                       idx_t **const    elems,
                       ptrdiff_t *const n_local_elements,
                       ptrdiff_t *const n_elements) {
    char         path[1024 * 10];
    MPI_Datatype mpi_idx_t = SFEM_MPI_IDX_T;

    // DO this outside
    // idx_t **elems = (idx_t **)malloc(sizeof(idx_t *) * nnodesxelem);

    int ret = 0;
    {
        idx_t *idx = 0;
        for (int d = 0; d < nnodesxelem; ++d) {
            sprintf(path, "%s/i%d.raw", folder, d);
            ret |= array_create_from_file(comm, path, mpi_idx_t, (void **)&idx, n_local_elements, n_elements);
            elems[d] = idx;
        }
    }

#ifdef SFEM_ENABLE_MEM_DIAGNOSTICS
    printf("mesh_read_elements: allocated %g GB\n", nnodesxelem * *n_local_elements * sizeof(idx_t) * 1e-9);
#endif

    return ret;
}

int mesh_read_mpi(const MPI_Comm  comm,
                  const char     *folder,
                  int            *nnodesxelem_out,
                  ptrdiff_t      *nelements_out,
                  idx_t        ***elements_out,
                  int            *spatial_dim_out,
                  ptrdiff_t      *nnodes_out,
                  geom_t       ***points_out,
                  ptrdiff_t      *n_owned_nodes_out,
                  ptrdiff_t      *n_owned_elements_out,
                  element_idx_t **element_mapping_out,
                  idx_t         **node_mapping_out,
                  int           **node_owner_out,
                  idx_t         **node_offsets_out,
                  idx_t         **ghosts_out,
                  ptrdiff_t      *n_owned_nodes_with_ghosts_out,
                  ptrdiff_t      *n_shared_elements_out,
                  ptrdiff_t      *n_owned_elements_with_ghosts_out) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int counts[2] = {0, 0};

    if (!rank) {
        char pattern[1024 * 10];
        // sprintf(pattern, "%s/i[0-9].*", folder);
        sprintf(pattern, "%s/i*.raw", folder);

        counts[0] = count_files(pattern);

        sprintf(pattern, "%s/x.raw", folder);
        counts[1] += count_files(pattern);

        sprintf(pattern, "%s/y.raw", folder);
        counts[1] += count_files(pattern);

        sprintf(pattern, "%s/z.raw", folder);
        counts[1] += count_files(pattern);
    }

    MPI_Bcast(counts, 2, MPI_INT, 0, comm);

    if (!counts[0] || !counts[1]) {
        SFEM_ERROR(
                "Could not find any mesh files in directory %s (#i*.raw = %d, {x,y,z}.raw = %d)\n", folder, counts[0], counts[1]);
    }

    const int nnodesxelem = counts[0];
    const int ndims       = counts[1];

    static const int remap_elements = 1;

    if (size > 1) {
        double tick = MPI_Wtime();
        ///////////////////////////////////////////////////////////////

        MPI_Datatype mpi_geom_t = SFEM_MPI_GEOM_T;
        MPI_Datatype mpi_idx_t  = SFEM_MPI_IDX_T;

        // ///////////////////////////////////////////////////////////////

        ptrdiff_t n_local_elements = 0, n_elements = 0;
        ptrdiff_t n_local_nodes = 0, n_nodes = 0;

        char path[1024 * 10];

        idx_t **elems = (idx_t **)malloc(sizeof(idx_t *) * nnodesxelem);

        // {
        //     idx_t *idx = 0;

        //     for (int d = 0; d < nnodesxelem; ++d) {
        //         sprintf(path, "%s/i%d.raw", folder, d);
        //         array_create_from_file(
        //             comm, path, mpi_idx_t, (void **)&idx, &n_local_elements, &n_elements);
        //         elems[d] = idx;
        //     }
        // }

        mesh_read_elements(comm, nnodesxelem, folder, elems, &n_local_elements, &n_elements);

        idx_t *unique_idx = (idx_t *)malloc(sizeof(idx_t) * n_local_elements * nnodesxelem);
        for (int d = 0; d < nnodesxelem; ++d) {
            memcpy(&unique_idx[d * n_local_elements], elems[d], sizeof(idx_t) * n_local_elements);
        }

        ptrdiff_t n_unique = psortreduce(unique_idx, n_local_elements * nnodesxelem);

        ////////////////////////////////////////////////////////////////////////////////
        // Read coordinates
        ////////////////////////////////////////////////////////////////////////////////

        geom_t **xyz = (geom_t **)malloc(sizeof(geom_t *) * ndims);

        static const char *str_xyz = "xyzt";

        for (int d = 0; d < ndims; ++d) {
            sprintf(path, "%s/%c.raw", folder, str_xyz[d]);
            array_create_from_file(comm, path, mpi_geom_t, (void **)&xyz[d], &n_local_nodes, &n_nodes);
        }

        ////////////////////////////////////////////////////////////////////////////////

        idx_t *input_node_partitions = (idx_t *)malloc(sizeof(idx_t) * (size + 1));
        memset(input_node_partitions, 0, sizeof(idx_t) * (size + 1));
        input_node_partitions[rank + 1] = n_local_nodes;

        MPI_CATCH_ERROR(MPI_Allreduce(MPI_IN_PLACE, &input_node_partitions[1], size, SFEM_MPI_IDX_T, MPI_SUM, comm));

        for (int r = 0; r < size; ++r) {
            input_node_partitions[r + 1] += input_node_partitions[r];
        }

        int *gather_node_count = malloc(size * sizeof(idx_t));
        memset(gather_node_count, 0, size * sizeof(int));

        // int *owner_rank = malloc(n_unique * sizeof(int));
        // memset(owner_rank, 0, n_unique * sizeof(int));

        for (ptrdiff_t i = 0; i < n_unique; ++i) {
            idx_t     idx   = unique_idx[i];
            const int owner = find_owner_rank(idx, n_local_nodes, size, input_node_partitions);

            assert(owner < size);
            assert(owner >= 0);

            assert(idx >= input_node_partitions[owner]);
            assert(idx < input_node_partitions[owner + 1]);

            gather_node_count[owner]++;
            // owner_rank[i] = owner;
        }

        int *scatter_node_count = malloc(size * sizeof(int));
        memset(scatter_node_count, 0, size * sizeof(int));

        MPI_CATCH_ERROR(MPI_Alltoall(gather_node_count, 1, SFEM_MPI_IDX_T, scatter_node_count, 1, SFEM_MPI_IDX_T, comm));

        int *gather_node_displs  = malloc((size + 1) * sizeof(int));
        int *scatter_node_displs = malloc((size + 1) * sizeof(int));

        gather_node_displs[0]  = 0;
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

        if (0)  //
        {
            for (int r = 0; r < size; r++) {
                if (r == rank) {
                    printf("------------------------\n");
                    printf("[%d] gnc: ", rank);

                    for (int i = 0; i < size; i++) {
                        printf("%d ", gather_node_count[i]);
                    }
                    printf("\n");
                    printf("[%d] gnd: ", rank);
                    for (int i = 0; i < size; i++) {
                        printf("%d ", gather_node_displs[i]);
                    }
                    printf("\n");
                    printf("[%d] snc: ", rank);
                    for (int i = 0; i < size; i++) {
                        printf("%d ", scatter_node_count[i]);
                    }
                    printf("\n");
                    printf("[%d] snd: ", rank);
                    for (int i = 0; i < size; i++) {
                        printf("%d ", scatter_node_displs[i]);
                    }
                    printf("\n");
                    printf("------------------------\n");
                    fflush(stdout);
                }

                MPI_Barrier(comm);
            }
        }

        MPI_CATCH_ERROR(MPI_Alltoallv(unique_idx,
                                      gather_node_count,
                                      gather_node_displs,
                                      SFEM_MPI_IDX_T,
                                      send_list,
                                      scatter_node_count,
                                      scatter_node_displs,
                                      SFEM_MPI_IDX_T,
                                      comm));

        ///////////////////////////////////////////////////////////////////////

        // Remove offset
        for (ptrdiff_t i = 0; i < size_send_list; ++i) {
            send_list[i] -= input_node_partitions[rank];
        }

        ///////////////////////////////////////////////////////////////////////
        // Exchange points

        geom_t  *sendx    = (geom_t *)malloc(size_send_list * sizeof(geom_t));
        geom_t **part_xyz = (geom_t **)malloc(sizeof(geom_t *) * ndims);

        for (int d = 0; d < ndims; ++d) {
            // Fill buffer
            for (ptrdiff_t i = 0; i < size_send_list; ++i) {
                sendx[i] = xyz[d][send_list[i]];
            }

            geom_t *recvx = (geom_t *)malloc(n_unique * sizeof(geom_t));
            MPI_CATCH_ERROR(MPI_Alltoallv(sendx,
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

        free(xyz);

        ///////////////////////////////////////////////////////////////////////
        // Determine owners
        int *node_owner       = (int *)malloc(n_unique * sizeof(int));
        int *node_share_count = (int *)calloc(n_unique, sizeof(int));

        {
            int *decide_node_owner = (int *)malloc(n_local_nodes * sizeof(int));
            int *send_node_owner   = (int *)malloc(size_send_list * sizeof(int));

            int *decide_share_count = (int *)calloc(n_local_nodes, sizeof(int));
            int *send_share_count   = (int *)malloc(size_send_list * sizeof(int));

            for (ptrdiff_t i = 0; i < n_local_nodes; ++i) {
                decide_node_owner[i] = size;
            }

            for (int r = 0; r < size; ++r) {
                int begin = scatter_node_displs[r];
                int end   = scatter_node_displs[r + 1];

                for (int i = begin; i < end; ++i) {
                    decide_node_owner[send_list[i]] = MIN(decide_node_owner[send_list[i]], r);
                }

                for (int i = begin; i < end; ++i) {
                    decide_share_count[send_list[i]]++;
                }
            }

            for (int r = 0; r < size; ++r) {
                int begin = scatter_node_displs[r];
                int end   = scatter_node_displs[r + 1];

                for (int i = begin; i < end; ++i) {
                    send_node_owner[i] = decide_node_owner[send_list[i]];
                }

                for (int i = begin; i < end; ++i) {
                    send_share_count[i] = decide_share_count[send_list[i]];
                }
            }

            MPI_CATCH_ERROR(MPI_Alltoallv(send_node_owner,
                                          scatter_node_count,
                                          scatter_node_displs,
                                          MPI_INT,
                                          node_owner,
                                          gather_node_count,
                                          gather_node_displs,
                                          MPI_INT,
                                          comm));

            MPI_CATCH_ERROR(MPI_Alltoallv(send_share_count,
                                          scatter_node_count,
                                          scatter_node_displs,
                                          MPI_INT,
                                          node_share_count,
                                          gather_node_count,
                                          gather_node_displs,
                                          MPI_INT,
                                          comm));

            free(decide_node_owner);
            free(send_node_owner);

            free(decide_share_count);
            free(send_share_count);
        }

        ///////////////////////////////////////////////////////////////////////
        // Localize element index
        for (ptrdiff_t d = 0; d < nnodesxelem; ++d) {
            for (ptrdiff_t e = 0; e < n_local_elements; ++e) {
                elems[d][e] = find_idx_binary_search(elems[d][e], unique_idx, n_unique);
            }
        }

        ////////////////////////////////////////////////////////////
        // Remap node index
        // We reorder nodes with the following order
        // 1) locally owned
        // 2) locally owned and shared by a remote process
        // 3) Shared, hence owned by a remote process

        idx_t *proc_ptr = (idx_t *)calloc((size + 1), sizeof(idx_t));
        idx_t *offset   = (idx_t *)calloc((size), sizeof(idx_t));

        for (ptrdiff_t node = 0; node < n_unique; ++node) {
            proc_ptr[node_owner[node] + 1] += 1;
        }

        const ptrdiff_t n_owned_nodes = proc_ptr[rank + 1];
        // Remove offset
        proc_ptr[rank + 1] = 0;

        proc_ptr[0] = n_owned_nodes;
        for (int r = 0; r < size; ++r) {
            proc_ptr[r + 1] += proc_ptr[r];
        }

        // This rank comes first
        proc_ptr[rank] = 0;

        // Build local remap index
        idx_t *local_remap = (idx_t *)malloc((n_unique) * sizeof(idx_t));
        for (ptrdiff_t node = 0; node < n_unique; ++node) {
            int owner         = node_owner[node];
            local_remap[node] = proc_ptr[owner] + offset[owner]++;
        }

        if (1) {
            // Remap based on shared
            ptrdiff_t owned_shared_count[2] = {0, 0};

            for (ptrdiff_t node = 0; node < n_unique; ++node) {
                int owner = node_owner[node];
                if (owner != rank) continue;
                owned_shared_count[node_share_count[node] > 1]++;
            }

            ptrdiff_t owned_shared_offset[2] = {0, owned_shared_count[0]};
            for (ptrdiff_t node = 0; node < n_unique; ++node) {
                int owner = node_owner[node];
                if (owner != rank) continue;

                int owned_and_shared = node_share_count[node] > 1;
                local_remap[node]    = proc_ptr[owner] + owned_shared_offset[owned_and_shared]++;
            }

            *n_owned_nodes_with_ghosts_out = owned_shared_count[1];
        }

        const size_t max_sz    = MAX(sizeof(int), MAX(sizeof(idx_t), sizeof(real_t)));
        void        *temp_buff = malloc(n_unique * max_sz);

        for (int d = 0; d < ndims; ++d) {
            geom_t *x = part_xyz[d];
            array_remap_scatter(n_unique, geom_t, local_remap, x, temp_buff);
        }

        array_remap_scatter(n_unique, int, local_remap, node_owner, temp_buff);
        array_remap_scatter(n_unique, idx_t, local_remap, unique_idx, temp_buff);
        array_remap_scatter(n_unique, int, local_remap, node_share_count, temp_buff);

        for (int d = 0; d < nnodesxelem; ++d) {
            for (ptrdiff_t e = 0; e < n_local_elements; ++e) {
                elems[d][e] = local_remap[elems[d][e]];
            }
        }

        free(local_remap);
        free(proc_ptr);
        free(temp_buff);
        free(offset);

        ////////////////////////////////////////////////////////////
        // Remap element index

        // Reorder elements with the following order
        // 1) Locally owned element
        // 2) Elements that are locally owned but have node shared by a remote
        // process

        if (remap_elements) {
            idx_t   *temp_buff = (idx_t *)malloc(n_local_elements * sizeof(idx_t));
            uint8_t *is_local  = (uint8_t *)calloc(n_local_elements, sizeof(uint8_t));

            for (ptrdiff_t e = 0; e < n_local_elements; ++e) {
                int is_local_e          = 1;
                int is_owned_and_shared = 0;

                for (int d = 0; d < nnodesxelem; ++d) {
                    const idx_t idx = elems[d][e];

                    if (node_owner[idx] != rank) {
                        is_local_e = 0;
                    }

                    is_owned_and_shared += node_share_count[idx] > 1;
                }

                is_local[e] = is_local_e;
                if (is_local_e && is_owned_and_shared) {
                    is_local[e] += 1;
                }
            }

            // FIXME?
            idx_t *element_mapping = (idx_t *)malloc(n_local_elements * sizeof(idx_t));

            ptrdiff_t counter = 0;
            for (ptrdiff_t e = 0; e < n_local_elements; ++e) {
                if (is_local[e] == 1) {
                    element_mapping[counter++] = e;
                }
            }

            *n_owned_elements_with_ghosts_out = 0;
            for (ptrdiff_t e = 0; e < n_local_elements; ++e) {
                if (is_local[e] == 2) {
                    element_mapping[counter++] = e;
                    (*n_owned_elements_with_ghosts_out)++;
                }
            }

            *n_shared_elements_out = n_local_elements - counter;
            for (ptrdiff_t e = 0; e < n_local_elements; ++e) {
                if (!is_local[e]) {
                    element_mapping[counter++] = e;
                }
            }

            for (int d = 0; d < nnodesxelem; ++d) {
                array_remap_gather(n_local_elements, idx_t, element_mapping, elems[d], temp_buff);
            }

            *element_mapping_out = element_mapping;

            free(temp_buff);
            free(is_local);
        } else {
            *element_mapping_out              = 0;
            *n_shared_elements_out            = 0;
            *n_owned_elements_with_ghosts_out = 0;
        }

        ///////////////////////////////////////////////////////////////////////
        // Free space
        free(sendx);
        free(send_list);
        free(input_node_partitions);
        free(node_share_count);

        free(scatter_node_count);
        free(gather_node_count);
        free(gather_node_displs);
        free(scatter_node_displs);

        ///////////////////////////////////////////////////////////////////////
        *spatial_dim_out = ndims;
        *nnodesxelem_out = nnodesxelem;

        *nelements_out = n_local_elements;
        *nnodes_out    = n_unique;

        *elements_out = elems;
        *points_out   = part_xyz;

        *n_owned_nodes_out    = n_owned_nodes;
        *n_owned_elements_out = n_local_elements - *n_shared_elements_out;

        *node_mapping_out = unique_idx;
        *node_owner_out   = node_owner;

        *ghosts_out       = 0;
        *node_offsets_out = 0;

        mesh_build_global_ids(comm,
                              n_nodes,
                              n_local_nodes,
                              n_local_nodes,
                              *node_mapping_out,
                              *node_owner_out,
                              node_offsets_out,
                              ghosts_out,
                              n_owned_nodes_with_ghosts_out);

        // MPI_Barrier(comm);
        double tock = MPI_Wtime();
        if (!rank) {
            printf("read_mesh.c: read_mesh\t%g seconds\n", tock - tick);
        }

        return SFEM_SUCCESS;
    } else {
        SFEM_ERROR("Serial mesh reading not implemented\n");
        return SFEM_FAILURE;
    }
}

int read_raw_array(const char *path, size_t type_size, void **data, ptrdiff_t *n_elements) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open file %s\n", path);
        return SFEM_FAILURE;
    }
    fseek(fp, 0, SEEK_END);
    *n_elements = ftell(fp) / type_size;
    fseek(fp, 0, SEEK_SET);
    *data = malloc(*n_elements * type_size);
    if (fread(*data, type_size, *n_elements, fp) != *n_elements) {
        fprintf(stderr, "Failed to read file %s\n", path);
        return SFEM_FAILURE;
    }
    fclose(fp);
    return SFEM_SUCCESS;
}

int mesh_read_serial(const char *folder,
                     int        *nnodesxelem_out,
                     ptrdiff_t  *nelements_out,
                     idx_t    ***elems_out,
                     int        *spatial_dim_out,
                     ptrdiff_t  *nnodes_out,
                     geom_t   ***xyz_out) {
    MPI_Datatype mpi_geom_t = SFEM_MPI_GEOM_T;
    MPI_Datatype mpi_idx_t  = SFEM_MPI_IDX_T;

    ptrdiff_t n_local_elements = 0, n_elements = 0;
    ptrdiff_t n_local_nodes = 0, n_nodes = 0;

    char pattern[1024 * 10];
    snprintf(pattern, sizeof(pattern), "%s/i*.raw", folder);
    int nnodesxelem = count_files(pattern);

    snprintf(pattern, sizeof(pattern), "%s/x.raw", folder);
    int ndims = count_files(pattern);

    snprintf(pattern, sizeof(pattern), "%s/y.raw", folder);
    ndims += count_files(pattern);

    snprintf(pattern, sizeof(pattern), "%s/z.raw", folder);
    ndims += count_files(pattern);

    idx_t **elems = (idx_t **)malloc(sizeof(idx_t *) * nnodesxelem);
    for (int d = 0; d < nnodesxelem; d++) {
        elems[d] = 0;
    }

    char path[1024 * 10];
    {
        ptrdiff_t n_local_elements0 = 0, n_elements0 = 0;
        for (int d = 0; d < nnodesxelem; ++d) {
            snprintf(path, sizeof(path), "%s/i%d.raw", folder, d);

            idx_t *idx = 0;
            if (read_raw_array(path, sizeof(idx_t), (void **)&idx, &n_elements) != SFEM_SUCCESS) {
                return SFEM_FAILURE;
            }
            n_local_elements = n_elements;
            // End of Selection
            elems[d] = idx;

            if (d == 0) {
                n_local_elements0 = n_local_elements;
                n_elements0       = n_elements;
            } else {
                assert(n_local_elements0 == n_local_elements);
                assert(n_elements0 == n_elements);

                if (n_elements0 != n_elements) {
                    SFEM_ERROR("Inconsistent lenghts in input %ld != %ld\n", (long)n_local_elements0, (long)n_local_elements);
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Read coordinates
    ////////////////////////////////////////////////////////////////////////////////

    geom_t **xyz = (geom_t **)malloc(sizeof(geom_t *) * ndims);
    for (int d = 0; d < ndims; d++) {
        xyz[d] = 0;
    }

    static const char *str_xyz = "xyzt";
    for (int d = 0; d < ndims; ++d) {
        snprintf(path, sizeof(path), "%s/%c.raw", folder, str_xyz[d]);

        geom_t *xyz_d = 0;
        if (read_raw_array(path, sizeof(geom_t), (void **)&xyz_d, &n_local_nodes) != SFEM_SUCCESS) {
            return SFEM_FAILURE;
        }
        xyz[d] = xyz_d;
    }

    *nnodesxelem_out = nnodesxelem;
    *spatial_dim_out = ndims;
    *nelements_out   = n_local_elements;
    *elems_out       = elems;
    *nnodes_out      = n_local_nodes;
    *xyz_out         = xyz;

    return SFEM_SUCCESS;
}

#undef array_remap_scatter
#undef array_remap_gather
