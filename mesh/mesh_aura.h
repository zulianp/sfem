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

void mesh_exchange_nodal_master_to_slave(const mesh_t *mesh, 
    send_recv_t *const slave_to_master, MPI_Datatype data_type, void *const inout);

void mesh_create_nodal_send_recv(const mesh_t *mesh, 
    send_recv_t *const slave_to_master);

void send_recv_destroy(send_recv_t *const sr);

void mesh_aura(const mesh_t *mesh, mesh_t *aura);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_AURA_H
