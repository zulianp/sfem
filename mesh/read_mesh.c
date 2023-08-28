
#include "read_mesh.h"

#include "matrix.io/matrixio_array.h"
#include "matrix.io/utils.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <glob.h>
#include <mpi.h>

#include "sortreduce.h"

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

inline static int count_files(const char *pattern) {
    glob_t gl;
    glob(pattern, GLOB_MARK, NULL, &gl);

    int n_files = gl.gl_pathc;

    printf("n_files (%d):\n", n_files);
    for (int np = 0; np < n_files; np++) {
        printf("%s\n", gl.gl_pathv[np]);
    }

    globfree(&gl);
    return n_files;
}

int mesh_node_ids(mesh_t *mesh, idx_t *const ids) {
    int rank, size;
    MPI_Comm_rank(mesh->comm, &rank);
    MPI_Comm_size(mesh->comm, &size);

    for (ptrdiff_t i = 0; i < mesh->n_owned_nodes; i++) {
        ids[i] = mesh->node_offsets[rank] + i;
    }

    for (ptrdiff_t i = mesh->n_owned_nodes; i < mesh->nnodes; i++) {
        ids[i] = mesh->ghosts[i - mesh->n_owned_nodes];
    }

    return 0;
}

int mesh_build_global_ids(mesh_t *mesh) {
    MPI_Comm comm = mesh->comm;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    long n_gnodes = mesh->n_owned_nodes;
    long global_node_offset = 0;
    MPI_Exscan(&n_gnodes, &global_node_offset, 1, MPI_LONG, MPI_SUM, comm);

    n_gnodes = global_node_offset + mesh->n_owned_nodes;
    MPI_Bcast(&n_gnodes, 1, MPI_LONG, size - 1, comm);

    idx_t *node_offsets = malloc((size + 1) * sizeof(idx_t));
    CATCH_MPI_ERROR(MPI_Allgather(
        &global_node_offset, 1, SFEM_MPI_IDX_T, node_offsets, 1, SFEM_MPI_IDX_T, comm));

    const ptrdiff_t n_lnodes_no_reminder = n_gnodes / size;
    const ptrdiff_t begin = (n_gnodes / size) * rank;

    ptrdiff_t n_lnodes_temp = n_lnodes_no_reminder;
    if (rank == size - 1) {
        n_lnodes_temp = n_gnodes - begin;
    }

    const ptrdiff_t begin_owned_with_ghosts = mesh->n_owned_nodes - mesh->n_owned_nodes_with_ghosts;
    const ptrdiff_t extent_owned_with_ghosts = mesh->n_owned_nodes_with_ghosts;
    const ptrdiff_t n_ghost_nodes = mesh->nnodes - mesh->n_owned_nodes;

    idx_t *ghost_keys = &mesh->node_mapping[begin_owned_with_ghosts];
    idx_t *ghost_ids = malloc(MAX(extent_owned_with_ghosts, n_ghost_nodes) * sizeof(idx_t));

    int *send_displs = (int *)malloc((size + 1) * sizeof(int));
    int *recv_displs = (int *)malloc((size + 1) * sizeof(int));
    int *send_count = (int *)calloc(size, sizeof(int));
    int *recv_count = (int *)malloc(size * sizeof(int));

    for (ptrdiff_t i = 0; i < extent_owned_with_ghosts; i++) {
        ghost_ids[i] = global_node_offset + begin_owned_with_ghosts + i;
    }

    for (ptrdiff_t i = 0; i < extent_owned_with_ghosts; i++) {
        const idx_t idx = ghost_keys[i];
        int dest_rank = idx / n_lnodes_no_reminder;
        send_count[dest_rank]++;
    }

    CATCH_MPI_ERROR(MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm));

    send_displs[0] = 0;
    for (int r = 0; r < size; r++) {
        send_displs[r + 1] = send_displs[r] + send_count[r];
    }

    recv_displs[0] = 0;
    for (int r = 0; r < size; r++) {
        recv_displs[r + 1] = recv_displs[r] + recv_count[r];
    }

    idx_t *recv_key_buff = malloc(recv_displs[size] * sizeof(idx_t));

    CATCH_MPI_ERROR(MPI_Alltoallv(ghost_keys,
                                  send_count,
                                  send_displs,
                                  SFEM_MPI_IDX_T,
                                  recv_key_buff,
                                  recv_count,
                                  recv_displs,
                                  SFEM_MPI_IDX_T,
                                  mesh->comm));

    idx_t *recv_ids_buff = (idx_t *)malloc(recv_displs[size] * sizeof(idx_t));

    CATCH_MPI_ERROR(MPI_Alltoallv(ghost_ids,
                                  send_count,
                                  send_displs,
                                  SFEM_MPI_IDX_T,
                                  recv_ids_buff,
                                  recv_count,
                                  recv_displs,
                                  SFEM_MPI_IDX_T,
                                  mesh->comm));

    idx_t *mapping = malloc(n_lnodes_temp * sizeof(idx_t));

#ifndef NDEBUG
    for (ptrdiff_t i = 0; i < n_lnodes_temp; i++) {
        mapping[i] = -1;
    }
#endif

    // Fill mapping
    for (int r = 0; r < size; r++) {
        int proc_begin = recv_displs[r];
        int proc_extent = recv_count[r];

        idx_t *keys = &recv_key_buff[proc_begin];
        idx_t *ids = &recv_ids_buff[proc_begin];

        for (int k = 0; k < proc_extent; k++) {
            idx_t iii = keys[k] - begin;
            assert(iii < n_lnodes_temp);
            assert(iii >= 0);

            mapping[iii] = ids[k];
        }
    }

    // MPI_Barrier(comm);
    // for(int r_ = 0; r_ < size; r_++) {
    //     if(r_ == rank) {
    //         printf("[%d] ------------------------------------\n", rank);
    //         for (int r = 0; r < size; r++) {
    //             int proc_begin = recv_displs[r];
    //             int proc_extent = recv_count[r];

    //             idx_t *keys = &recv_key_buff[proc_begin];
    //             idx_t *ids = &recv_ids_buff[proc_begin];

    //             printf("%d)\n", r);
    //             for (int k = 0; k < proc_extent; k++) {
    //                 printf("(%d %d)",  keys[k], ids[k]);
    //             }
    //             printf("\n");
    //         }

    //     }

    //     MPI_Barrier(comm);
    // }

    /////////////////////////////////////////////////
    // Gather query ghost nodes
    memset(send_count, 0, size * sizeof(int));

    // Get the query nodes
    ghost_keys = &mesh->node_mapping[mesh->n_owned_nodes];

    for (ptrdiff_t i = 0; i < n_ghost_nodes; i++) {
        const idx_t idx = ghost_keys[i];
        int dest_rank = idx / n_lnodes_no_reminder;
        send_count[dest_rank]++;
    }

    CATCH_MPI_ERROR(MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm));

    send_displs[0] = 0;
    for (int r = 0; r < size; r++) {
        send_displs[r + 1] = send_displs[r] + send_count[r];
    }

    recv_displs[0] = 0;
    for (int r = 0; r < size; r++) {
        recv_displs[r + 1] = recv_displs[r] + recv_count[r];
    }

    recv_key_buff = realloc(recv_key_buff, recv_displs[size] * sizeof(idx_t));

    CATCH_MPI_ERROR(MPI_Alltoallv(ghost_keys,
                                  send_count,
                                  send_displs,
                                  SFEM_MPI_IDX_T,
                                  recv_key_buff,
                                  recv_count,
                                  recv_displs,
                                  SFEM_MPI_IDX_T,
                                  mesh->comm));

    // Query mapping
    for (int r = 0; r < size; r++) {
        int proc_begin = recv_displs[r];
        int proc_extent = recv_count[r];

        idx_t *keys = &recv_key_buff[proc_begin];

        for (int k = 0; k < proc_extent; k++) {
            idx_t iii = keys[k] - begin;
            assert(iii < n_lnodes_temp);
            assert(iii >= 0);
            assert(mapping[iii] >= 0);
            keys[k] = mapping[iii];
        }
    }

    mesh->ghosts = malloc(n_ghost_nodes * sizeof(idx_t));

    // Send back
    CATCH_MPI_ERROR(MPI_Alltoallv(recv_key_buff,
                                  recv_count,
                                  recv_displs,
                                  SFEM_MPI_IDX_T,
                                  mesh->ghosts,
                                  send_count,
                                  send_displs,
                                  SFEM_MPI_IDX_T,
                                  mesh->comm));

    /////////////////////////////////////////////////

    mesh->node_offsets = node_offsets;

    // for (int r_ = 0; r_ < size; r_++) {
    //     if (r_ == rank) {
    //         printf("[%d] ----------------------------\n", rank);
    //         printf("ghosts (%d):\n", (int)( mesh->nnodes - mesh->n_owned_nodes));
    //         for (int i = 0; i < mesh->nnodes - mesh->n_owned_nodes; i++) {
    //             printf("%d ", mesh->ghosts[i]);
    //         }
    //         printf("\n");

    //         printf("offsets:\n");
    //         for (ptrdiff_t i = 0; i < size; i++) {
    //             printf("%d ", mesh->node_offsets[i]);
    //         }

    //         printf("\n");

    //         fflush(stdout);
    //     }
    //     MPI_Barrier(comm);
    // }

    // Clean-up
    free(send_count);
    free(recv_count);
    free(send_displs);
    free(recv_displs);
    free(recv_key_buff);
    free(recv_ids_buff);

    return 0;
}

int mesh_read_elements(MPI_Comm comm,
                       const int nnodesxelem,
                       const char *folder,
                       idx_t **const elems,
                       ptrdiff_t *const n_local_elements,
                       ptrdiff_t *const n_elements) {
    char path[1024 * 10];
    MPI_Datatype mpi_idx_t = SFEM_MPI_IDX_T;

    // DO this outside
    // idx_t **elems = (idx_t **)malloc(sizeof(idx_t *) * nnodesxelem);

    int ret = 0;
    {
        idx_t *idx = 0;
        for (int d = 0; d < nnodesxelem; ++d) {
            sprintf(path, "%s/i%d.raw", folder, d);
            ret |= array_create_from_file(
                comm, path, mpi_idx_t, (void **)&idx, n_local_elements, n_elements);
            elems[d] = idx;
        }
    }

    return ret;
}

int mesh_read_generic(MPI_Comm comm,
                      const int nnodesxelem,
                      const int ndims,
                      const char *folder,
                      mesh_t *mesh) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // static const int remap_nodes = 1;
    static const int remap_elements = 1;

    if (size > 1) {
        double tick = MPI_Wtime();
        ///////////////////////////////////////////////////////////////

        MPI_Datatype mpi_geom_t = SFEM_MPI_GEOM_T;
        MPI_Datatype mpi_idx_t = SFEM_MPI_IDX_T;

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

        ptrdiff_t n_unique = sortreduce(unique_idx, n_local_elements * nnodesxelem);

        ////////////////////////////////////////////////////////////////////////////////
        // Read coordinates
        ////////////////////////////////////////////////////////////////////////////////

        geom_t **xyz = (geom_t **)malloc(sizeof(geom_t *) * ndims);

        static const char *str_xyz = "xyzt";

        for (int d = 0; d < ndims; ++d) {
            sprintf(path, "%s/%c.raw", folder, str_xyz[d]);
            array_create_from_file(
                comm, path, mpi_geom_t, (void **)&xyz[d], &n_local_nodes, &n_nodes);
        }

        ////////////////////////////////////////////////////////////////////////////////

        idx_t *input_node_partitions = (idx_t *)malloc(sizeof(idx_t) * (size + 1));
        memset(input_node_partitions, 0, sizeof(idx_t) * (size + 1));
        input_node_partitions[rank + 1] = n_local_nodes;

        CATCH_MPI_ERROR(MPI_Allreduce(
            MPI_IN_PLACE, &input_node_partitions[1], size, SFEM_MPI_IDX_T, MPI_SUM, comm));

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

        CATCH_MPI_ERROR(MPI_Alltoall(
            gather_node_count, 1, SFEM_MPI_IDX_T, scatter_node_count, 1, SFEM_MPI_IDX_T, comm));

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

        ///////////////////////////////////////////////////////////////////////

        // Remove offset
        for (ptrdiff_t i = 0; i < size_send_list; ++i) {
            send_list[i] -= input_node_partitions[rank];
        }

        ///////////////////////////////////////////////////////////////////////
        // Exchange points

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
        // Determine owners
        int *node_owner = (int *)malloc(n_unique * sizeof(int));
        int *node_share_count = (int *)calloc(n_unique, sizeof(int));

        {
            int *decide_node_owner = (int *)malloc(n_local_nodes * sizeof(int));
            int *send_node_owner = (int *)malloc(size_send_list * sizeof(int));

            int *decide_share_count = (int *)calloc(n_local_nodes, sizeof(int));
            int *send_share_count = (int *)malloc(size_send_list * sizeof(int));

            for (ptrdiff_t i = 0; i < n_local_nodes; ++i) {
                decide_node_owner[i] = size;
            }

            for (int r = 0; r < size; ++r) {
                idx_t begin = scatter_node_displs[r];
                idx_t end = scatter_node_displs[r + 1];

                for (idx_t i = begin; i < end; ++i) {
                    decide_node_owner[send_list[i]] = MIN(decide_node_owner[send_list[i]], r);
                }

                for (idx_t i = begin; i < end; ++i) {
                    decide_share_count[send_list[i]]++;
                }
            }

            for (int r = 0; r < size; ++r) {
                idx_t begin = scatter_node_displs[r];
                idx_t end = scatter_node_displs[r + 1];

                for (idx_t i = begin; i < end; ++i) {
                    send_node_owner[i] = decide_node_owner[send_list[i]];
                }

                for (idx_t i = begin; i < end; ++i) {
                    send_share_count[i] = decide_share_count[send_list[i]];
                }
            }

            CATCH_MPI_ERROR(MPI_Alltoallv(send_node_owner,
                                          scatter_node_count,
                                          scatter_node_displs,
                                          MPI_INT,
                                          node_owner,
                                          gather_node_count,
                                          gather_node_displs,
                                          MPI_INT,
                                          comm));

            CATCH_MPI_ERROR(MPI_Alltoallv(send_share_count,
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
        idx_t *offset = (idx_t *)calloc((size), sizeof(idx_t));

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
            int owner = node_owner[node];
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
                local_remap[node] = proc_ptr[owner] + owned_shared_offset[owned_and_shared]++;
            }

            mesh->n_owned_nodes_with_ghosts = owned_shared_count[1];
        }

        const size_t max_sz = MAX(sizeof(int), MAX(sizeof(idx_t), sizeof(real_t)));
        void *temp_buff = malloc(n_unique * max_sz);

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

        ////////////////////////////////////////////////////////////
        // Remap element index

        // Reorder elements with the following order
        // 1) Locally owned element
        // 2) Elements that are locally owned but have node shared by a remote
        // process

        if (remap_elements) {
            idx_t *temp_buff = (idx_t *)malloc(n_local_elements * sizeof(idx_t));
            uint8_t *is_local = (uint8_t *)calloc(n_local_elements, sizeof(uint8_t));

            for (ptrdiff_t e = 0; e < n_local_elements; ++e) {
                int is_local_e = 1;
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

            idx_t *element_mapping = (idx_t *)malloc(n_local_elements * sizeof(idx_t));

            ptrdiff_t counter = 0;
            for (ptrdiff_t e = 0; e < n_local_elements; ++e) {
                if (is_local[e] == 1) {
                    element_mapping[counter++] = e;
                }
            }

            mesh->n_owned_elements_with_ghosts = 0;
            for (ptrdiff_t e = 0; e < n_local_elements; ++e) {
                if (is_local[e] == 2) {
                    element_mapping[counter++] = e;
                    mesh->n_owned_elements_with_ghosts++;
                }
            }

            mesh->n_shared_elements = n_local_elements - counter;
            for (ptrdiff_t e = 0; e < n_local_elements; ++e) {
                if (!is_local[e]) {
                    element_mapping[counter++] = e;
                }
            }

            for (int d = 0; d < nnodesxelem; ++d) {
                array_remap_gather(n_local_elements, idx_t, element_mapping, elems[d], temp_buff);
            }

            mesh->element_mapping = element_mapping;

            free(temp_buff);
            free(is_local);
        } else {
            mesh->element_mapping = 0;
            mesh->n_shared_elements = 0;
            mesh->n_owned_elements_with_ghosts = 0;
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
        mesh->comm = comm;
        mesh->mem_space = SFEM_MEM_SPACE_HOST;

        mesh->spatial_dim = ndims;
        mesh->element_type = nnodesxelem;

        mesh->elements = elems;
        mesh->points = part_xyz;
        mesh->nelements = n_local_elements;
        mesh->nnodes = n_unique;

        mesh->n_owned_nodes = n_owned_nodes;
        mesh->n_owned_elements = n_local_elements - mesh->n_shared_elements;

        // Original indexing
        mesh->node_mapping = unique_idx;
        mesh->node_owner = node_owner;

        mesh->ghosts = 0;
        mesh->node_offsets = 0;

        mesh_build_global_ids(mesh);

        // MPI_Barrier(comm);
        double tock = MPI_Wtime();
        if (!rank) {
            printf("read_mesh.c: read_mesh\t%g seconds\n", tock - tick);
        }

        return 0;
    } else {
        // Serial fallback
        ///////////////////////////////////////////////////////////////

        MPI_Datatype mpi_geom_t = SFEM_MPI_GEOM_T;
        MPI_Datatype mpi_idx_t = SFEM_MPI_IDX_T;

        // ///////////////////////////////////////////////////////////////

        ptrdiff_t n_local_elements = 0, n_elements = 0;
        ptrdiff_t n_local_nodes = 0, n_nodes = 0;

        char path[1024 * 10];

        idx_t **elems = (idx_t **)malloc(sizeof(idx_t *) * nnodesxelem);
        for (int d = 0; d < nnodesxelem; d++) {
            elems[d] = 0;
        }

        {
            idx_t *idx = 0;

            ptrdiff_t n_local_elements0 = 0, n_elements0 = 0;
            for (int d = 0; d < nnodesxelem; ++d) {
                sprintf(path, "%s/i%d.raw", folder, d);
                array_create_from_file(
                    comm, path, mpi_idx_t, (void **)&idx, &n_local_elements, &n_elements);
                elems[d] = idx;

                if (d == 0) {
                    n_local_elements0 = n_local_elements;
                    n_elements0 = n_elements;
                } else {
                    assert(n_local_elements0 == n_local_elements);
                    assert(n_elements0 == n_elements);

                    if (n_elements0 != n_elements) {
                        fprintf(stderr,
                                "Inconsistent lenghts in input %ld != %ld\n",
                                (long)n_local_elements0,
                                (long)n_local_elements);
                        MPI_Abort(comm, -1);
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
            sprintf(path, "%s/%c.raw", folder, str_xyz[d]);
            array_create_from_file(
                comm, path, mpi_geom_t, (void **)&xyz[d], &n_local_nodes, &n_nodes);
        }

        mesh->comm = comm;

        mesh->mem_space = SFEM_MEM_SPACE_HOST;

        mesh->spatial_dim = ndims;
        mesh->element_type = nnodesxelem;

        mesh->nelements = n_local_elements;
        mesh->nnodes = n_local_nodes;

        mesh->elements = elems;
        mesh->points = xyz;

        mesh->n_owned_nodes = n_local_nodes;
        mesh->n_owned_elements = n_local_elements;
        mesh->n_shared_elements = 0;

        mesh->node_mapping = 0;
        mesh->node_owner = 0;

        mesh->element_mapping = 0;

        mesh->n_owned_nodes_with_ghosts = 0;
        mesh->n_owned_elements_with_ghosts = 0;

        mesh->ghosts = 0;
        mesh->node_offsets = 0;
        return 0;
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

int serial_read_tet_mesh(const char *folder,
                         ptrdiff_t *nelements,
                         idx_t *elems[4],
                         ptrdiff_t *nnodes,
                         geom_t *xyz[3]) {
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

        if (x_nnodes != y_nnodes || x_nnodes != z_nnodes) {
            fprintf(stderr, "Bad input lengths!\n");
            return -1;
        }

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

        if (nindex0 != nindex1 || nindex0 != nindex2 || nindex0 != nindex3) {
            fprintf(stderr, "Bad input lengths!\n");
            return -1;
        }

        nindex0 /= sizeof(idx_t);
        assert((ptrdiff_t)(nindex0 * sizeof(idx_t)) == nindex1);
        *nelements = nindex0;
    }

    return 0;
}

int mesh_surf_read(MPI_Comm comm, const char *folder, mesh_t *mesh) {
    int nnodesxelem = 3;
    int ndims = 3;
    return mesh_read_generic(comm, nnodesxelem, ndims, folder, mesh);
}

int mesh_read(MPI_Comm comm, const char *folder, mesh_t *mesh) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int counts[2] = {0, 0};

    if (!rank) {
        char pattern[1024 * 10];
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
        if (!rank) {
            fprintf(stderr, "Could not find any mesh files in directory %s\n", folder);
        }

        MPI_Abort(comm, -1);
    }

    return mesh_read_generic(comm, counts[0], counts[1], folder, mesh);
}

#undef array_remap_scatter
#undef array_remap_gather
