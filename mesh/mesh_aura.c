#include "mesh_aura.h"

#include "sfem_defs.h"
#include "sfem_mesh.h"
#include "sortreduce.h"

#include "matrix.io/utils.h"

#include <stdio.h>
#include <string.h>

void send_recv_destroy(send_recv_t *const sr) {
    sr->comm = MPI_COMM_NULL;
    free(sr->send_count);
    free(sr->recv_count);
    free(sr->send_displs);
    free(sr->recv_displs);
    free(sr->sparse_idx);
}

void mesh_exchange_nodal_master_to_slave(const mesh_t *mesh,
                                         send_recv_t *const slave_to_master,
                                         MPI_Datatype data_type,
                                         void *const inout) {
    int rank, size;
    MPI_Comm_rank(slave_to_master->comm, &rank);
    MPI_Comm_size(slave_to_master->comm, &size);

    int n = slave_to_master->recv_displs[size];

    int type_size;
    CATCH_MPI_ERROR(MPI_Type_size(data_type, &type_size));

    // Create and pack send data
    char *send_data = (char *)malloc(n * type_size);
    const char *in_ptr = (char *)inout;

    for (int i = 0; i < size; i++) {
        const int begin = slave_to_master->recv_displs[i];
        const int extent = slave_to_master->recv_count[i];
        char *buff = &send_data[begin * type_size];
        const idx_t *idx = &slave_to_master->sparse_idx[begin];

        for (int k = 0; k < extent; k++) {
            const int byte_offset = k * type_size;
            memcpy(&buff[byte_offset], &in_ptr[idx[k] * type_size], type_size);
        }
    }

    // Retrieve recv buffer
    char *recv_data = (char *)inout;

    // Offset for not owned data
    recv_data = &recv_data[mesh->n_owned_nodes * type_size];
    CATCH_MPI_ERROR(MPI_Alltoallv(send_data,
                                  slave_to_master->recv_count,
                                  slave_to_master->recv_displs,
                                  data_type,
                                  recv_data,
                                  slave_to_master->send_count,
                                  slave_to_master->send_displs,
                                  data_type,
                                  slave_to_master->comm));
    free(send_data);
}

// void mesh_exchange_nodal_slave_to_master(const mesh_t *mesh,
//     send_recv_t *const slave_to_master, MPI_Datatype data_type, void *const inout)
//     {
//         CATCH_MPI_ERROR(MPI_Alltoallv(send_data,
//                                       sr->send_count,
//                                       sr->send_displs,
//                                       data_type,
//                                       recv_count,
//                                       sr->recv_displs,
//                                       sr->recv_data,
//                                       data_type,
//                                       sr->comm));
//     }

void mesh_create_nodal_send_recv(const mesh_t *mesh, send_recv_t *const slave_to_master) {
    int rank, size;

    MPI_Comm_rank(mesh->comm, &rank);
    MPI_Comm_size(mesh->comm, &size);

    int *send_count = (int *)malloc(size * sizeof(int));
    memset(send_count, 0, size * sizeof(int));

    int *send_displs = (int *)malloc((size + 1) * sizeof(int));
    memset(send_displs, 0, size * sizeof(int));

    for (ptrdiff_t i = mesh->n_owned_nodes; i < mesh->nnodes; i++) {
        assert(mesh->node_owner[i] >= 0);
        assert(mesh->node_owner[i] < size);

        send_count[mesh->node_owner[i]]++;
    }

    send_displs[0] = 0;
    for (int r = 0; r < size; r++) {
        send_displs[r + 1] = send_displs[r] + send_count[r];
    }

    int *recv_count = (int *)malloc(size * sizeof(int));
    memset(recv_count, 0, size * sizeof(int));

    int *recv_displs = (int *)malloc((size + 1) * sizeof(int));
    memset(recv_displs, 0, size * sizeof(int));

    CATCH_MPI_ERROR(MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, mesh->comm));

    recv_displs[0] = 0;
    for (int r = 0; r < size; r++) {
        recv_displs[r + 1] = recv_displs[r] + recv_count[r];
    }

    idx_t *slave_nodes = (idx_t *)malloc(recv_displs[size] * sizeof(idx_t));

    // for (int r = 0; r < size; ++r) {
    //     if (r == rank) {
    //         printf("[%d] ---------------------\n", rank);
    //         printf("recv\n");
    //         for (int l = 0; l < size; l++) {
    //             const int begin = recv_displs[l];
    //             const int extent = recv_count[l];
    //             printf("%d -> %d %d\n", l, begin, extent);
    //         }

    //         printf("send\n");
    //         for (int l = 0; l < size; l++) {
    //             const int begin = send_displs[l];
    //             const int extent = send_count[l];
    //             printf("%d -> %d %d\n", l, begin, extent);
    //         }
    //     }

    //     fflush(stdout);

    //     MPI_Barrier(mesh->comm);
    // }

    // send slave nodes to process with master nodes
    CATCH_MPI_ERROR(MPI_Alltoallv(&mesh->node_mapping[mesh->n_owned_nodes],
                                  send_count,
                                  send_displs,
                                  SFEM_MPI_IDX_T,
                                  slave_nodes,
                                  recv_count,
                                  recv_displs,
                                  SFEM_MPI_IDX_T,
                                  mesh->comm));

    // Replace global indexing with local indexing for identifying master nodes
    for (int r = 0; r < size; r++) {
        const int begin = recv_displs[r];
        const int extent = recv_count[r];
        idx_t *nodes = &slave_nodes[begin];
        for (int k = 0; k < extent; k++) {
            idx_t global_idx = nodes[k];
            idx_t local_idx = find_idx_binary_search(global_idx, mesh->node_mapping, mesh->n_owned_nodes);
            nodes[k] = local_idx;
        }
    }

    // Construct recv and send list

    slave_to_master->comm = mesh->comm;
    slave_to_master->send_count = send_count;
    slave_to_master->send_displs = send_displs;

    slave_to_master->recv_count = recv_count;
    slave_to_master->recv_displs = recv_displs;

    slave_to_master->sparse_idx = slave_nodes;
}

void mesh_aura(const mesh_t *mesh, mesh_t *aura) {
    MPI_Comm comm = mesh->comm;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    send_recv_t slave_to_master;
    mesh_create_nodal_send_recv(mesh, &slave_to_master);

    int *elem_send_count = (int *)malloc(size * sizeof(int));
    uint8_t *marked_nodes = (uint8_t *)malloc(mesh->nnodes * sizeof(uint8_t));
    memset(marked_nodes, 0, mesh->nnodes * sizeof(uint8_t));

    for (int r = 0; r < size; r++) {
        const int begin = slave_to_master.recv_displs[r];
        const int extent = slave_to_master.recv_count[r];

        idx_t *idx = &slave_to_master.sparse_idx[begin];

        for (int k = 0; k < extent; k++) {
            idx_t node = idx[k];
            marked_nodes[node] = 1;
        }
    }

    const int nnxe = elem_num_nodes(mesh->element_type);

    uint8_t *marked_elements = (uint8_t *)malloc(mesh->n_owned_elements * sizeof(uint8_t));
    memset(marked_elements, 0, mesh->n_owned_elements * sizeof(uint8_t));

    for (int d = 0; d < nnxe; d++) {
        for (ptrdiff_t i = 0; i < mesh->n_owned_elements; i++) {
            const idx_t node = mesh->elements[d][i];
            if (marked_nodes[node]) {
                marked_elements[i] = 1;
            }
        }
    }

    ptrdiff_t n_marked_elements = 0;
    for (ptrdiff_t i = 0; i < mesh->n_owned_elements; i++) {
        n_marked_elements += marked_elements[i];
    }

    idx_t *marked_element_list = (idx_t *)malloc(n_marked_elements * sizeof(idx_t));
    for (ptrdiff_t i = 0, ii = 0; i < mesh->n_owned_elements; i++) {
        if (marked_elements[i]) {
            marked_element_list[ii++] = i;
        }
    }

    memset(marked_nodes, 0, mesh->nnodes * sizeof(uint8_t));

    int *send_count = (int *)malloc(size * sizeof(int));
    int *send_displs = (int *)malloc((size + 1) * sizeof(int));

    memset(send_count, 0, size * sizeof(int));
    memset(send_displs, 0, (size + 1) * sizeof(int));
    idx_t *owner = malloc(nnxe * sizeof(idx_t));

    {  // Counting
        // 1) Count master elements (for every slave)
        for (int r = 0; r < size; r++) {
            const int begin = slave_to_master.recv_displs[r];
            const int extent = slave_to_master.recv_count[r];

            const idx_t *const idx = &slave_to_master.sparse_idx[begin];

            for (int k = 0; k < extent; k++) {
                const idx_t node = idx[k];
                marked_nodes[node] = 1;
            }

            for (ptrdiff_t i = 0; i < n_marked_elements; i++) {
                const idx_t e = marked_elements[i];

                int is_marked = 0;
                for (int d = 0; d < nnxe; d++) {
                    is_marked += marked_nodes[mesh->elements[d][e]];
                }

                if (is_marked) {
                    send_displs[r + 1]++;
                }
            }

            // set-to zero marked nodes
            for (int k = 0; k < extent; k++) {
                const idx_t node = idx[k];
                marked_nodes[node] = 0;
            }
        }

        // 2) Count slave elements (once)
        const ptrdiff_t n_shared_elements = mesh->n_shared_elements;
        ptrdiff_t n_send_elements = 0;
        for (ptrdiff_t es = 0; es < n_shared_elements; es++) {
            for (int d = 0; d < nnxe; d++) {
                owner[d] = mesh->node_owner[mesh->elements[d][mesh->n_owned_elements + es]];
            }

            int n_owners = sortreduce(owner, nnxe);
            for (int i = 0; i < n_owners; i++) {
                send_displs[owner[i] + 1]++;
            }
        }
    }

    for (int r = 0; r < size; r++) {
        send_displs[r + 1] += send_displs[r];
    }

    const ptrdiff_t size_send_buffer = send_displs[size];
    idx_t *element_send_lists = (idx_t *)malloc(size_send_buffer * sizeof(idx_t));

    {  // Create send-element lists
        for (int r = 0; r < size; r++) {
            const int begin = slave_to_master.recv_displs[r];
            const int extent = slave_to_master.recv_count[r];

            const idx_t *const idx = &slave_to_master.sparse_idx[begin];

            for (int k = 0; k < extent; k++) {
                const idx_t node = idx[k];
                marked_nodes[node] = 1;
            }

            for (ptrdiff_t i = 0; i < n_marked_elements; i++) {
                const idx_t e = marked_elements[i];

                int is_marked = 0;
                for (int d = 0; d < nnxe; d++) {
                    is_marked += marked_nodes[mesh->elements[d][e]];
                }

                if (is_marked) {
                    // add element index to list and increment count
                    element_send_lists[send_displs[r] + send_count[r]++] = e;
                }
            }

            // set-to zero marked nodes
            for (int k = 0; k < extent; k++) {
                const idx_t node = idx[k];
                marked_nodes[node] = 0;
            }
        }

        const ptrdiff_t n_shared_elements = mesh->n_shared_elements;
        for (ptrdiff_t es = 0; es < n_shared_elements; es++) {
            for (int d = 0; d < nnxe; d++) {
                owner[d] = mesh->node_owner[mesh->elements[d][mesh->n_owned_elements + es]];
            }

            int n_owners = sortreduce(owner, nnxe);
            for (int i = 0; i < n_owners; i++) {
                element_send_lists[send_displs[owner[i]] + send_count[owner[i]]++] = mesh->n_owned_elements + es;
            }
        }
    }

#ifndef NDEBUG
    for (int r = 0; r < size; r++) {
        assert(send_count[r] == send_displs[r + 1] - send_displs[r]);
    }
#endif

    int *recv_count = (int *)malloc((size + 1) * sizeof(int));
    // memset(recv_count, 0, (size + 1) * sizeof(int));

    int *recv_displs = (int *)malloc((size + 1) * sizeof(int));
    // memset(recv_displs, 0, (size + 1) * sizeof(int));

    CATCH_MPI_ERROR(MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm));

    recv_displs[0] = 0;
    for (int r = 0; r < size; r++) {
        recv_displs[r + 1] = recv_displs[r] + recv_count[r];
    }

    // Use largest type size here
    idx_t *send_buffer = (idx_t *)malloc(size_send_buffer * sizeof(idx_t));

    aura->elements = (idx_t **)malloc(nnxe * sizeof(idx_t *));
    for (int d = 0; d < nnxe; d++) {
        aura->elements[d] = (idx_t *)malloc(recv_displs[size] * sizeof(idx_t));

        for(ptrdiff_t i = 0; i < size_send_buffer; i++) {
            send_buffer[i] = mesh->elements[d][element_send_lists[i]];
        }

        CATCH_MPI_ERROR(MPI_Alltoallv(send_buffer,
                                      send_count,
                                      send_displs,
                                      SFEM_MPI_IDX_T,
                                      aura->elements[d],
                                      recv_count,
                                      recv_displs,
                                      SFEM_MPI_IDX_T,
                                      comm));
    }

    aura->nelements = recv_displs[size];
    aura->n_owned_elements = 0;
    aura->n_shared_elements = aura->nelements;

    {  // Clean-up
        send_recv_destroy(&slave_to_master);
        free(marked_nodes);
        free(marked_elements);
        free(marked_element_list);
        free(send_buffer);
        free(owner);

        free(send_count);
        free(send_displs);

        free(recv_count);
        free(recv_displs);
    }
}

// void mesh_aura(const mesh_t *mesh, mesh_t *aura) {
//     MPI_Comm comm = mesh->comm;

//     int rank, size;
//     MPI_Comm_rank(comm, &rank);
//     MPI_Comm_size(comm, &size);

//     const ptrdiff_t n_owned_elements = mesh->n_owned_elements;
//     const ptrdiff_t n_shared_elements = mesh->n_shared_elements;

//     const int nnxe = elem_num_nodes(mesh->element_type);
//     idx_t **shared_elements = (idx_t **)malloc(nnxe * sizeof(idx_t *));

//     int *send_displs = (int *)malloc((size + 1) * sizeof(int));
//     memset(send_displs, 0, (size + 1) * sizeof(int));

//     idx_t *owner = malloc(nnxe * sizeof(idx_t));

//     ptrdiff_t n_send_elements = 0;
//     for (ptrdiff_t es = 0; es < n_shared_elements; es++) {
//         for (int d = 0; d < nnxe; d++) {
//             owner[d] = mesh->node_owner[mesh->elements[d][n_owned_elements + es]];
//         }

//         int n_owners = sortreduce(owner, nnxe);
//         for (int i = 0; i < n_owners; i++) {
//             send_displs[owner[i] + 1]++;
//             n_send_elements++;
//         }
//     }

//     for (int r = 0; r < size; r++) {
//         send_displs[r + 1] += send_displs[r];
//     }

//     for (int d = 0; d < nnxe; d++) {
//         shared_elements[d] = (idx_t *)malloc(n_send_elements * sizeof(idx_t));
//     }

//     int *send_count = (int *)malloc((size + 1) * sizeof(int));
//     memset(send_count, 0, (size + 1) * sizeof(int));

//     for (ptrdiff_t es = 0; es < n_shared_elements; es++) {
//         for (int d = 0; d < nnxe; d++) {
//             owner[d] = mesh->node_owner[mesh->elements[d][n_owned_elements + es]];
//         }

//         int n_owners = sortreduce(owner, nnxe);

//         for (int i = 0; i < n_owners; i++) {
//             int owner_rank = owner[i];
//             for (int d = 0; d < nnxe; d++) {
//                 // Retrieve original global identifier
//                 const idx_t nidx = mesh->node_mapping[mesh->elements[d][n_owned_elements + es]];
//                 shared_elements[d][send_displs[owner_rank] + send_count[owner_rank]] = nidx;
//             }

//             send_count[owner_rank]++;
//         }
//     }

//     int *recv_count = (int *)malloc((size + 1) * sizeof(int));
//     memset(recv_count, 0, (size + 1) * sizeof(int));

//     int *recv_displs = (int *)malloc((size + 1) * sizeof(int));
//     memset(recv_displs, 0, (size + 1) * sizeof(int));

//     CATCH_MPI_ERROR(MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm));

//     recv_displs[0] = 0;
//     for (int r = 0; r < size; r++) {
//         recv_displs[r + 1] = recv_displs[r] + recv_count[r];
//     }

//     idx_t **aura_elements = (idx_t **)malloc(nnxe * sizeof(idx_t *));
//     for (int d = 0; d < nnxe; d++) {
//         aura_elements[d] = (idx_t *)malloc(recv_displs[size] * sizeof(idx_t));

//         CATCH_MPI_ERROR(MPI_Alltoallv(shared_elements[d],
//                                       send_count,
//                                       send_displs,
//                                       SFEM_MPI_IDX_T,
//                                       recv_count,
//                                       recv_displs,
//                                       aura_elements[d],
//                                       SFEM_MPI_IDX_T,
//                                       comm));
//     }

//     const size_t n_unsorted = recv_displs[size] * nnxe;
//     idx_t *aura_nodes = (idx_t *)malloc(n_unsorted * sizeof(idx_t));

//     for (int d = 0; d < nnxe; d++) {
//         memcpy(&aura_nodes[d * recv_displs[size]], aura_elements[d], recv_displs[size] * sizeof(idx_t));
//     }

//     const ptrdiff_t n_aura_nodes = sortreduce(aura_nodes, n_unsorted);

//     idx_t *global_to_local_map = (idx_t *)malloc(n_aura_nodes * sizeof(idx_t));

//     for (ptrdiff_t i = 0; i < n_aura_nodes; i++) {
//         // FIXME check if not found!
//         global_to_local_map[i] = find_idx_binary_search(aura_nodes[i], mesh->node_mapping, mesh->nnodes);
//     }

//     uint8_t *marked = (uint8_t *)malloc(mesh->nnodes * sizeof(uint8_t));
//     memset(marked, 0, mesh->nnodes * sizeof(uint8_t));

//     for (ptrdiff_t i = 0; i < n_aura_nodes; i++) {
//         if (global_to_local_map[i] >= 0 && global_to_local_map[i] < mesh->nnodes) {
//             marked[global_to_local_map[i]] = 1;
//         }
//     }

//     ptrdiff_t n_send_elements_phase_2 = 0;

//     // Identify owned elements
//     for (ptrdiff_t i = 0; i < mesh->n_owned_elements; i++) {
//         int must_add = 0;
//         for (int d = 0; d < nnxe; d++) {
//             must_add += marked[mesh->elements[d][i]];
//         }

//         if (must_add) {
//             n_send_elements_phase_2++;
//         }
//     }

//     idx_t **shared_elements_2 = (idx_t **)malloc(nnxe * sizeof(idx_t *));
//     for (int d = 0; d < nnxe; d++) {
//         shared_elements_2[d] = (idx_t *)malloc(n_send_elements_phase_2 * sizeof(idx_t));
//     }

//     {  // Pack elements for sending to
//         ptrdiff_t offset = 0;
//         for (ptrdiff_t i = 0; i < mesh->n_owned_elements; i++) {
//             int must_add = 0;
//             for (int d = 0; d < nnxe; d++) {
//                 must_add += marked[mesh->elements[d][i]];
//             }

//             if (must_add) {
//                 for (int d = 0; d < nnxe; d++) {
//                     shared_elements_2[d][offset] = mesh->elements[d][i];
//                 }

//                 offset++;
//             }
//         }
//     }

//     // Store in output mesh
//     aura->comm = comm;
//     // aura->elements

//     // Clean-up
//     free(owner);
//     free(send_count);
//     free(send_displs);
//     free(recv_count);
//     free(recv_displs);
//     free(aura_nodes);
//     free(global_to_local_map);

//     for (int d = 0; d < nnxe; d++) {
//         free(aura_elements[d]);
//     }

//     free(aura_elements);

//     for (int d = 0; d < nnxe; d++) {
//         free(shared_elements_2[d]);
//     }

//     free(shared_elements_2);
// }
