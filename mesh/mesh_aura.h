#ifndef SFEM_AURA_H
#define SFEM_AURA_H

#include <mpi.h>

#include "sfem_mesh.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    MPI_Comm comm;
    int *send_count;
    int *send_displs;

    int *recv_count;
    int *recv_displs;

    idx_t *sparse_idx;
} send_recv_t;

void mesh_exchange_nodal_master_to_slave(const ptrdiff_t n_owned_nodes,
                                         send_recv_t *const slave_to_master,
                                         MPI_Datatype data_type,
                                         void *const inout);

void mesh_create_nodal_send_recv(MPI_Comm comm,
                                 const ptrdiff_t nnodes,
                                 const ptrdiff_t n_owned_nodes,
                                 int *const node_owner,
                                 const idx_t *const node_offsets,
                                 const idx_t *const ghosts,
                                 send_recv_t *const slave_to_master);

void send_recv_destroy(send_recv_t *const sr);

// void mesh_aura(const mesh_t *mesh, mesh_t *aura);
// void mesh_aura_to_complete_mesh(const mesh_t *const mesh, const mesh_t *const aura, mesh_t *const
// out); void mesh_aura_fix_indices(const mesh_t *const mesh, mesh_t *const aura);

void mesh_remote_connectivity_graph(MPI_Comm comm,
                                    const enum ElemType element_type,
                                    const ptrdiff_t nelements,
                                    idx_t **const elements,
                                    const ptrdiff_t nnodes,
                                    const ptrdiff_t n_owned_nodes,
                                    const ptrdiff_t n_owned_elements_with_ghosts,
                                    const ptrdiff_t n_shared_elements,
                                    const int *const node_owner,
                                    const idx_t *const node_offsets,
                                    const idx_t *const ghosts,
                                    count_t **rowptr,
                                    idx_t **colidx,
                                    send_recv_t *const exchange);

void mesh_exchange_nodal_slave_to_master(MPI_Comm comm,
                                         send_recv_t *const slave_to_master,
                                         MPI_Datatype data_type,
                                         void *const SFEM_RESTRICT ghost_data,
                                         void *const SFEM_RESTRICT buffer);

ptrdiff_t mesh_exchange_master_buffer_count(const send_recv_t *const slave_to_master);

void exchange_add(MPI_Comm comm,
                  const ptrdiff_t nnodes,
                  const ptrdiff_t n_owned_nodes,
                  send_recv_t *slave_to_master,
                  real_t *const SFEM_RESTRICT inout,
                  real_t *const SFEM_RESTRICT real_buffer);

void mesh_create_nodal_send_recv_deprecated(const mesh_t *const mesh,
                                            send_recv_t *const slave_to_master);

void exchange_add_deprecated(mesh_t *mesh,
                             send_recv_t *slave_to_master,
                             real_t *inout,
                             real_t *real_buffer);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_AURA_H
