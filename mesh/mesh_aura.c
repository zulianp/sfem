#include "mesh_aura.h"

#include "sfem_defs.h"
#include "sfem_mesh.h"
#include "sortreduce.h"

#include "crs_graph.h"

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

    double tick = MPI_Wtime();

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

    // send slave nodes to process with master nodes
    CATCH_MPI_ERROR(MPI_Alltoallv(mesh->ghosts,
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
            idx_t local_idx = global_idx - mesh->node_offsets[rank];
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

// void mesh_aura(const mesh_t *mesh, mesh_t *aura) {
//     double tick = MPI_Wtime();

//     MPI_Comm comm = mesh->comm;

//     int rank, size;
//     MPI_Comm_rank(comm, &rank);
//     MPI_Comm_size(comm, &size);

//     send_recv_t slave_to_master;
//     mesh_create_nodal_send_recv(mesh, &slave_to_master);

//     int *elem_send_count = (int *)malloc(size * sizeof(int));
//     uint8_t *marked_nodes = (uint8_t *)malloc(mesh->nnodes *
//     sizeof(uint8_t)); memset(marked_nodes, 0, mesh->nnodes *
//     sizeof(uint8_t));

//     for (int r = 0; r < size; r++) {
//         const int begin = slave_to_master.recv_displs[r];
//         const int extent = slave_to_master.recv_count[r];

//         idx_t *idx = &slave_to_master.sparse_idx[begin];

//         for (int k = 0; k < extent; k++) {
//             idx_t node = idx[k];
//             marked_nodes[node] = 1;
//         }
//     }

//     const int nnxe = elem_num_nodes(mesh->element_type);

//     uint8_t *marked_elements =
//         (uint8_t *)malloc(mesh->n_owned_elements * sizeof(uint8_t));
//     memset(marked_elements, 0, mesh->n_owned_elements * sizeof(uint8_t));

//     for (int d = 0; d < nnxe; d++) {
//         for (ptrdiff_t i = 0; i < mesh->n_owned_elements; i++) {
//             const idx_t node = mesh->elements[d][i];
//             if (marked_nodes[node]) {
//                 marked_elements[i] = 1;
//             }
//         }
//     }

//     ptrdiff_t n_marked_elements = 0;
//     for (ptrdiff_t i = 0; i < mesh->n_owned_elements; i++) {
//         n_marked_elements += marked_elements[i];
//     }

//     idx_t *marked_element_list =
//         (idx_t *)malloc(n_marked_elements * sizeof(idx_t));
//     for (ptrdiff_t i = 0, ii = 0; i < mesh->n_owned_elements; i++) {
//         if (marked_elements[i]) {
//             marked_element_list[ii++] = i;
//         }
//     }

//     memset(marked_nodes, 0, mesh->nnodes * sizeof(uint8_t));

//     int *send_count = (int *)malloc(size * sizeof(int));
//     int *send_displs = (int *)malloc((size + 1) * sizeof(int));

//     memset(send_count, 0, size * sizeof(int));
//     memset(send_displs, 0, (size + 1) * sizeof(int));
//     idx_t *owner = malloc(nnxe * sizeof(idx_t));

//     {  // Counting
//         // 1) Count master elements (for every slave)
//         for (int r = 0; r < size; r++) {
//             const int begin = slave_to_master.recv_displs[r];
//             const int extent = slave_to_master.recv_count[r];

//             const idx_t *const idx = &slave_to_master.sparse_idx[begin];

//             for (int k = 0; k < extent; k++) {
//                 const idx_t node = idx[k];
//                 marked_nodes[node] = 1;
//             }

//             for (ptrdiff_t i = 0; i < n_marked_elements; i++) {
//                 const idx_t e = marked_elements[i];

//                 int is_marked = 0;
//                 for (int d = 0; d < nnxe; d++) {
//                     is_marked += marked_nodes[mesh->elements[d][e]];
//                 }

//                 if (is_marked) {
//                     send_displs[r + 1]++;
//                 }
//             }

//             // set-to zero marked nodes
//             for (int k = 0; k < extent; k++) {
//                 const idx_t node = idx[k];
//                 marked_nodes[node] = 0;
//             }
//         }

//         // 2) Count slave elements (once)
//         const ptrdiff_t n_shared_elements = mesh->n_shared_elements;
//         ptrdiff_t n_send_elements = 0;
//         for (ptrdiff_t es = 0; es < n_shared_elements; es++) {
//             for (int d = 0; d < nnxe; d++) {
//                 owner[d] =
//                     mesh->node_owner[mesh->elements[d][mesh->n_owned_elements
//                     +
//                                                        es]];
//             }

//             int n_owners = sortreduce(owner, nnxe);
//             for (int i = 0; i < n_owners; i++) {
//                 send_displs[owner[i] + 1]++;
//             }
//         }
//     }

//     for (int r = 0; r < size; r++) {
//         send_displs[r + 1] += send_displs[r];
//     }

//     const ptrdiff_t size_send_buffer = send_displs[size];
//     idx_t *element_send_lists =
//         (idx_t *)malloc(size_send_buffer * sizeof(idx_t));

//     {  // Create send-element lists
//         for (int r = 0; r < size; r++) {
//             const int begin = slave_to_master.recv_displs[r];
//             const int extent = slave_to_master.recv_count[r];

//             const idx_t *const idx = &slave_to_master.sparse_idx[begin];

//             for (int k = 0; k < extent; k++) {
//                 const idx_t node = idx[k];
//                 marked_nodes[node] = 1;
//             }

//             for (ptrdiff_t i = 0; i < n_marked_elements; i++) {
//                 const idx_t e = marked_elements[i];

//                 int is_marked = 0;
//                 for (int d = 0; d < nnxe; d++) {
//                     is_marked += marked_nodes[mesh->elements[d][e]];
//                 }

//                 if (is_marked) {
//                     // add element index to list and increment count
//                     element_send_lists[send_displs[r] + send_count[r]++] = e;
//                 }
//             }

//             // set-to zero marked nodes
//             for (int k = 0; k < extent; k++) {
//                 const idx_t node = idx[k];
//                 marked_nodes[node] = 0;
//             }
//         }

//         const ptrdiff_t n_shared_elements = mesh->n_shared_elements;
//         for (ptrdiff_t es = 0; es < n_shared_elements; es++) {
//             for (int d = 0; d < nnxe; d++) {
//                 owner[d] =
//                     mesh->node_owner[mesh->elements[d][mesh->n_owned_elements
//                     +
//                                                        es]];
//             }

//             int n_owners = sortreduce(owner, nnxe);
//             for (int i = 0; i < n_owners; i++) {
//                 element_send_lists[send_displs[owner[i]] +
//                                    send_count[owner[i]]++] =
//                     mesh->n_owned_elements + es;
//             }
//         }
//     }

// #ifndef NDEBUG
//     for (int r = 0; r < size; r++) {
//         assert(send_count[r] == send_displs[r + 1] - send_displs[r]);
//     }
// #endif

//     int *recv_count = (int *)malloc((size + 1) * sizeof(int));
//     int *recv_displs = (int *)malloc((size + 1) * sizeof(int));

//     CATCH_MPI_ERROR(
//         MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm));

//     recv_displs[0] = 0;
//     for (int r = 0; r < size; r++) {
//         recv_displs[r + 1] = recv_displs[r] + recv_count[r];
//     }

//     // Use largest type size here
//     idx_t *send_buffer = (idx_t *)malloc(size_send_buffer * sizeof(idx_t));

//     aura->elements = (idx_t **)malloc(nnxe * sizeof(idx_t *));
//     for (int d = 0; d < nnxe; d++) {
//         aura->elements[d] = (idx_t *)malloc(recv_displs[size] *
//         sizeof(idx_t));

//         for (ptrdiff_t i = 0; i < size_send_buffer; i++) {
//             // Global ids
//             send_buffer[i] =
//                 mesh->node_mapping[mesh->elements[d][element_send_lists[i]]];
//         }

//         CATCH_MPI_ERROR(MPI_Alltoallv(send_buffer,
//                                       send_count,
//                                       send_displs,
//                                       SFEM_MPI_IDX_T,
//                                       aura->elements[d],
//                                       recv_count,
//                                       recv_displs,
//                                       SFEM_MPI_IDX_T,
//                                       comm));
//     }

//     aura->nelements = recv_displs[size];
//     aura->n_owned_elements = 0;
//     aura->n_shared_elements = aura->nelements;

//     {  // Exchange element mapping
//         aura->element_mapping =
//             (idx_t *)malloc(aura->nelements * sizeof(idx_t));
//         for (ptrdiff_t i = 0; i < size_send_buffer; i++) {
//             send_buffer[i] = mesh->element_mapping[element_send_lists[i]];
//         }

//         CATCH_MPI_ERROR(MPI_Alltoallv(send_buffer,
//                                       send_count,
//                                       send_displs,
//                                       SFEM_MPI_IDX_T,
//                                       aura->element_mapping,
//                                       recv_count,
//                                       recv_displs,
//                                       SFEM_MPI_IDX_T,
//                                       comm));
//     }

//     {  // Exchange node_mapping and xyz
//         int max_send_elems = 0;
//         for (int r = 0; r < size; r++) {
//             max_send_elems = MAX(max_send_elems, send_count[r]);
//         }

//         idx_t *node_send_lists =
//             (idx_t *)malloc(max_send_elems * nnxe * sizeof(idx_t));

//         int *node_send_count = (int *)malloc(size * sizeof(int));
//         int *node_send_displs = (int *)malloc((size + 1) * sizeof(int));

//         memset(node_send_count, 0, (size) * sizeof(int));
//         node_send_displs[0] = 0;

//         // Count number of nodes to send
//         for (int r = 0; r < size; r++) {
//             const int begin = send_displs[r];
//             const int extent = send_count[r];
//             const idx_t *const elems = &element_send_lists[begin];

//             int nnn = 0;
//             for (int d = 0; d < nnxe; d++) {
//                 for (int i = 0; i < extent; i++) {
//                     const idx_t node = mesh->elements[d][elems[i]];
//                     // Send only nodes the receiver does not already own
//                     if (mesh->node_owner[node] == r) {
//                         node_send_lists[nnn++] = node;
//                     }
//                 }
//             }

//             const idx_t n_send_nodes = sortreduce(node_send_lists, nnn);
//             node_send_displs[r + 1] = n_send_nodes + node_send_displs[r];
//         }

//         idx_t *node_send_buff =
//             (idx_t *)malloc(node_send_displs[size] * sizeof(idx_t));
//         geom_t *point_send_buff =
//             (geom_t *)malloc(node_send_displs[size] * sizeof(geom_t));

//         // Pack nodes to send
//         int offset = 0;
//         for (int r = 0; r < size; r++) {
//             const int begin = send_displs[r];
//             const int extent = send_count[r];
//             const idx_t *const elems = &element_send_lists[begin];

//             int nnn = 0;
//             for (int d = 0; d < nnxe; d++) {
//                 for (int i = 0; i < extent; i++) {
//                     const idx_t node = mesh->elements[d][elems[i]];
//                     // Send only nodes the receiver does not already own
//                     if (mesh->node_owner[node] == r) {
//                         assert(node >= 0);
//                         node_send_lists[nnn++] = node;
//                     }
//                 }
//             }

//             const int n_send_nodes = sortreduce(node_send_lists, nnn);

//             for (int i = 0; i < n_send_nodes; i++) {
//                 assert(node_send_lists[i] >= 0);
//                 assert(node_send_displs[r] + node_send_count[r] <
//                        node_send_displs[size]);
//                 node_send_buff[node_send_displs[r] + node_send_count[r]++] =
//                     node_send_lists[i];
//             }
//         }

//         // exchange
//         int *node_recv_count = (int *)malloc(size * sizeof(int));
//         int *node_recv_displs = (int *)malloc((size + 1) * sizeof(int));

//         CATCH_MPI_ERROR(MPI_Alltoall(
//             node_send_count, 1, MPI_INT, node_recv_count, 1, MPI_INT, comm));

//         node_recv_displs[0] = 0;

//         for (int r = 0; r < size; r++) {
//             node_recv_displs[r + 1] = node_recv_displs[r] +
//             node_recv_count[r];
//         }

//         idx_t *aura_nodes =
//             (idx_t *)malloc(node_recv_displs[size] * sizeof(idx_t));
//         geom_t **aura_points =
//             (geom_t **)malloc(mesh->spatial_dim * sizeof(geom_t *));

//         for (int d = 0; d < mesh->spatial_dim; d++) {
//             for (int i = 0; i < node_send_displs[size]; i++) {
//                 assert(node_send_buff[i] >= 0);
//                 assert(node_send_buff[i] < mesh->nnodes);
//                 point_send_buff[i] = mesh->points[d][node_send_buff[i]];
//             }

//             aura_points[d] =
//                 (geom_t *)malloc(node_recv_displs[size] * sizeof(geom_t));

//             CATCH_MPI_ERROR(MPI_Alltoallv(point_send_buff,
//                                           node_send_count,
//                                           node_send_displs,
//                                           SFEM_MPI_IDX_T,
//                                           aura_points[d],
//                                           node_recv_count,
//                                           node_recv_displs,
//                                           SFEM_MPI_IDX_T,
//                                           comm));
//         }

//         for (int i = 0; i < node_send_displs[size]; i++) {
//             node_send_buff[i] = mesh->node_mapping[node_send_buff[i]];
//         }

//         CATCH_MPI_ERROR(MPI_Alltoallv(node_send_buff,
//                                       node_send_count,
//                                       node_send_displs,
//                                       SFEM_MPI_IDX_T,
//                                       aura_nodes,
//                                       node_recv_count,
//                                       node_recv_displs,
//                                       SFEM_MPI_IDX_T,
//                                       comm));

//         aura->mem_space = mesh->mem_space;
//         aura->comm = mesh->comm;
//         aura->spatial_dim = mesh->spatial_dim;
//         aura->element_type = mesh->element_type;
//         aura->nnodes = node_recv_displs[size];
//         aura->n_owned_nodes = aura->nnodes;
//         aura->points = aura_points;
//         aura->node_mapping = aura_nodes;
//         aura->node_owner =
//             (idx_t *)malloc(node_recv_displs[size] * sizeof(idx_t));

//         sort_idx(aura_nodes, aura->nnodes);

//         for (int r = 0; r < size; r++) {
//             int begin = node_recv_displs[r];
//             int extent = node_recv_count[r];

//             idx_t *no = &aura->node_owner[begin];

//             for (int k = 0; k < extent; k++) {
//                 no[k] = r;
//             }
//         }

//         free(node_send_buff);
//         free(node_send_count);
//         free(node_recv_count);
//     }

//     {  // Clean-up
//         send_recv_destroy(&slave_to_master);
//         free(marked_nodes);
//         free(marked_elements);
//         free(marked_element_list);
//         free(send_buffer);
//         free(owner);

//         free(send_count);
//         free(send_displs);

//         free(recv_count);
//         free(recv_displs);
//     }

//     double tock = MPI_Wtime();
//     if (!rank) {
//         printf("mesh_aura.c: mesh_aura %g seconds\n", tock - tick);
//     }
// }

// void mesh_aura_to_complete_mesh(const mesh_t *const mesh,
//                                 const mesh_t *const aura,
//                                 mesh_t *const out) {
//     idx_t *marked = malloc(mesh->nnodes * sizeof(idx_t));
//     memset(marked, 0, mesh->nnodes * sizeof(idx_t));

//     ptrdiff_t n_missing_nodes = 0;
//     int nnxe = elem_num_nodes(mesh->element_type);
//     for (int d = 0; d < nnxe; d++) {
//         idx_t *es = aura->elements[d];
//         for (ptrdiff_t e = 0; e < aura->nelements; e++) {
//             if (es[e] < 0) {
//                 idx_t node = -es[e] - 1;

//                 if (!marked[node]) {
//                     // Plus 1 to ensure non-zero
//                     marked[node] = aura->nnodes + 1;
//                     n_missing_nodes++;
//                 }
//             }
//         }
//     }

//     out->nnodes = aura->nnodes + n_missing_nodes;
//     out->nelements = aura->nelements;
//     out->elements = (idx_t **)malloc(nnxe * sizeof(idx_t *));
//     out->points = (geom_t **)malloc(mesh->spatial_dim * sizeof(geom_t *));

//     for (int d = 0; d < nnxe; d++) {
//         out->elements[d] = malloc(aura->nelements * sizeof(idx_t));

//         idx_t *es = aura->elements[d];
//         for (ptrdiff_t e = 0; e < aura->nelements; e++) {
//             idx_t node = es[e];
//             if (es[e] < 0) {
//                 node = -node - 1;
//                 node = marked[node];
//             }

//             out->elements[d][e] = node;
//         }
//     }

//     // Compress marked
//     idx_t *marked_idx = malloc(n_missing_nodes * sizeof(idx_t));
//     for (ptrdiff_t i = 0, mi = 0; i < mesh->nnodes; i++) {
//         if (marked[i]) {
//             assert(mi < n_missing_nodes);
//             marked_idx[mi++] = i;
//         }
//     }

//     for (int d = 0; d < mesh->spatial_dim; d++) {
//         out->points[d] = malloc(out->nnodes * sizeof(geom_t));
//         memcpy(aura->points[d], out->points[d], aura->nnodes *
//         sizeof(geom_t));

//         // Copy coordinates from main mesh
//         for (ptrdiff_t i = 0; i < n_missing_nodes; i++) {
//             idx_t m_idx = marked_idx[i];
//             idx_t o_idx = marked[m_idx];
//             out->points[d][o_idx] = mesh->points[d][m_idx];
//         }
//     }

//     free(marked_idx);
//     free(marked);
// }

// void mesh_aura_fix_indices(const mesh_t *const mesh, mesh_t *const aura) {
//     int nnxe = elem_num_nodes(mesh->element_type);
//     for (int d = 0; d < nnxe; d++) {
//         idx_t *es = aura->elements[d];

//         for (ptrdiff_t i = 0; i < aura->nelements; i++) {
//             const idx_t gnode = es[i];
//             idx_t lnode = safe_find_idx_binary_search(
//                 gnode, aura->node_mapping, aura->nnodes);

//             if (lnode == -1) {
//                 // Change sign to identify that it is already in the mesh
//                 printf("%d\n", gnode);
//                 lnode = -(1 + find_idx_binary_search(
//                                   gnode, mesh->node_mapping, mesh->nnodes));
//             }

//             es[i] = lnode;
//         }
//     }
// }

// void mesh_shared_node_contiguous_idx(const mesh_t *mesh, idx_t *share_nodes_contiguou)

void mesh_remote_connectivity_graph(const mesh_t *mesh,
                                    count_t **out_rowptr,
                                    idx_t **out_colidx,
                                    send_recv_t *const exchange) {
    double tick = MPI_Wtime();

    MPI_Comm comm = mesh->comm;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int nnxe = elem_num_nodes(mesh->element_type);

    count_t *selection_rowptr = 0;
    idx_t *selection_colidx = 0;
    idx_t **elems = (idx_t **)malloc(nnxe * sizeof(idx_t *));

    ptrdiff_t n_elems = mesh->n_owned_elements_with_ghosts + mesh->n_shared_elements;
    ptrdiff_t e_begin = mesh->nelements - n_elems;

    for (int d = 0; d < nnxe; d++) {
        elems[d] = &mesh->elements[d][e_begin];
    }

    build_crs_graph_for_elem_type(
        mesh->element_type, n_elems, mesh->nnodes, elems, &selection_rowptr, &selection_colidx);

    count_t *shared_rowptr = (count_t *)calloc(mesh->nnodes + 1, sizeof(count_t));
    ptrdiff_t node_begin = mesh->n_owned_nodes;

    // Consider only nodes that are shared
    for (ptrdiff_t i = node_begin; i < mesh->nnodes; i++) {
        const count_t begin = selection_rowptr[i];
        const count_t extent = selection_rowptr[i + 1] - selection_rowptr[i];
        const idx_t *const cols = &selection_colidx[begin];

        for (count_t k = 0; k < extent; k++) {
            const idx_t c = cols[k];
            shared_rowptr[i + 1] += (mesh->node_owner[c] == rank);
        }
    }

    // Compute begin/end offsets
    for (ptrdiff_t i = 0; i < mesh->nnodes; i++) {
        shared_rowptr[i + 1] += shared_rowptr[i];
    }

    ptrdiff_t nnz = shared_rowptr[mesh->nnodes];
    idx_t *shared_colidx = (idx_t *)malloc(nnz * sizeof(idx_t));
    for (ptrdiff_t i = node_begin; i < mesh->nnodes; i++) {
        const count_t begin = selection_rowptr[i];
        const count_t extent = selection_rowptr[i + 1] - selection_rowptr[i];
        const idx_t *const cols = &selection_colidx[begin];

        idx_t *const colidx = &shared_colidx[shared_rowptr[i]];

        count_t kk = 0;
        for (count_t k = 0; k < extent; k++) {
            const idx_t c = cols[k];
            if (mesh->node_owner[c] == rank) {
                colidx[kk++] = mesh->node_offsets[rank] + c;
            }
        }

        assert(kk == shared_rowptr[i + 1] - shared_rowptr[i]);
    }

    int *send_displs = (int *)calloc((size + 1), sizeof(int));
    int *send_count = (int *)calloc(size, sizeof(int));

    for (ptrdiff_t i = mesh->n_owned_nodes; i < mesh->nnodes; i++) {
        assert(mesh->node_owner[i] != rank);
        const count_t extent = shared_rowptr[i + 1] - shared_rowptr[i];
        if (!extent) continue;
        send_displs[mesh->node_owner[i] + 1] += 2 + extent;
    }

    for(int r = 0; r < size; r++) {
        send_displs[r+1] += send_displs[r];
    }

    idx_t *send_data = (idx_t *)malloc(MAX(1, send_displs[size]) * sizeof(idx_t));
    memset(send_data, 0, send_displs[size] * sizeof(idx_t));

    for (ptrdiff_t i = mesh->n_owned_nodes; i < mesh->nnodes; i++) {
        const count_t begin = shared_rowptr[i];
        const count_t extent = shared_rowptr[i + 1] - begin;
        if (!extent) continue;
        const int owner = mesh->node_owner[i];

        const idx_t *cols = &shared_colidx[begin];
        idx_t *sd = &send_data[send_displs[owner]];

        assert(owner != rank);
        assert(owner < size);

        if (send_count[owner] >= send_displs[owner + 1]) {
            printf("[%d] Bug i=%d o=%d sc=%d sd=%d\n",
                   rank,
                   (int)i,
                   owner,
                   send_count[owner],
                   send_displs[owner + 1]);
        }

        assert(send_count[owner] < send_displs[owner + 1]);
        sd[send_count[owner]++] = mesh->ghosts[i - mesh->n_owned_nodes];

        assert(send_count[owner] < send_displs[owner + 1]);
        sd[send_count[owner]++] = extent;

        for (count_t k = 0; k < extent; k++) {
            assert(send_count[owner] < send_displs[owner + 1]);
            sd[send_count[owner]++] = cols[k];
        }
    }

    int *recv_count = (int *)malloc(size * sizeof(int));
    int *recv_displs = (int *)malloc((size + 1) * sizeof(int));

    CATCH_MPI_ERROR(MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, mesh->comm));

    recv_displs[0] = 0;
    for (int r = 0; r < size; r++) {
        recv_displs[r + 1] = recv_displs[r] + recv_count[r];
    }

    idx_t *recv_data = (idx_t *)malloc(recv_displs[size] * sizeof(idx_t));

    CATCH_MPI_ERROR(MPI_Alltoallv(send_data,
                                  send_count,
                                  send_displs,
                                  SFEM_MPI_IDX_T,
                                  recv_data,
                                  recv_count,
                                  recv_displs,
                                  SFEM_MPI_IDX_T,
                                  mesh->comm));

    count_t *rowptr = calloc(mesh->nnodes + 1, sizeof(count_t));
    count_t *count = calloc(mesh->nnodes, sizeof(count_t));

    for (int r = 0; r < size; r++) {
        int begin = recv_displs[r];
        int extent = recv_count[r];

        assert(r != rank || extent == 0);

        for (idx_t kk = 0; kk < extent;) {
            idx_t row = recv_data[begin + kk] - mesh->node_offsets[rank];
            assert(row >= 0);
            assert(row < mesh->n_owned_nodes);
            idx_t ncols = recv_data[begin + kk + 1];
            rowptr[row + 1] += ncols;
            kk += 2 + ncols;
        }
    }

    for (ptrdiff_t i = 0; i < mesh->nnodes; i++) {
        rowptr[i + 1] += rowptr[i];
    }

    idx_t *colidx = malloc(rowptr[mesh->nnodes] * sizeof(idx_t));

    for (int r = 0; r < size; r++) {
        int begin = recv_displs[r];
        int extent = recv_count[r];

        for (idx_t kk = 0; kk < extent;) {
            idx_t row = recv_data[begin + kk] - mesh->node_offsets[rank];
            idx_t ncols = recv_data[begin + kk + 1];
            idx_t *data = &recv_data[begin + kk + 2];

            for (idx_t k = 0; k < ncols; k++) {
                colidx[rowptr[row] + count[row]++] = data[k];
            }

            kk += 2 + ncols;
        }
    }

    // for (int r_ = 0; r_ < size; r_++) {
    //     if (r_ == rank) {
    //         printf("[%d] ----------------------------\n", rank);
    //         printf("ghosts:\n");
    //         for (int i = 0; i < mesh->nnodes - mesh->n_owned_nodes; i++) {
    //             printf("%d ", mesh->ghosts[i]);
    //         }
    //         printf("\n");

    //         printf("cols:\n");
    //         printf("start %d\n", mesh->node_offsets[rank] + mesh->n_owned_nodes - mesh->n_owned_nodes_with_ghosts);

    //         for(ptrdiff_t i = mesh->n_owned_nodes - mesh->n_owned_nodes_with_ghosts; i <  mesh->n_owned_nodes; i++) {
    //             // if(rowptr[i+1] == rowptr[i]) continue;

    //             printf("row %d :\n", (int)i);
    //             for (ptrdiff_t k = rowptr[i]; k < rowptr[i+1]; k++) {
    //                 printf("%d ", colidx[k]);
    //             }
    //             printf("\n");
    //         }

    //         printf("\n");

    //         fflush(stdout);
    //     }
    //     MPI_Barrier(comm);
    // }

    // MPI_Barrier(comm);

    // for (ptrdiff_t i = 0; i < rowptr[mesh->nnodes]; i++) {
    //     colidx[i] =
    //         // find_idx_binary_search
    //         safe_find_idx_binary_search
    //         (colidx[i], mesh->ghosts, mesh->nnodes - mesh->n_owned_nodes);
    //     assert(colidx[i] >= 0);
    // }

    *out_rowptr = rowptr;
    *out_colidx = colidx;

    {
        free(elems);

        free(selection_rowptr);
        free(selection_colidx);

        free(shared_rowptr);
        free(shared_colidx);

        free(send_displs);
        free(send_count);
        free(send_data);

        free(recv_count);
        free(recv_displs);
        free(recv_data);

        free(count);
    }
}
