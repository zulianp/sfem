#ifndef SFEM_MESH_UTILS_HPP
#define SFEM_MESH_UTILS_HPP

#include "sfem_base.hpp"
#include "sfem_defs.hpp"
#include "smesh_exchange.hpp"
#include "smesh_mesh.hpp"
#include "smesh_types.hpp"

#include <mpi.h>
#include <memory>
#include <stddef.h>

static const int SFEM_MEM_SPACE_NONE    = -1;
static const int SFEM_MEM_SPACE_HOST    = 0;
static const int SFEM_MEM_SPACE_CUDA    = 1;
static const int SFEM_MEM_SPACE_MANAGED = 2;
static const int SFEM_MEM_SPACE_UNIFIED = 3;

typedef struct {
    MPI_Comm comm;
    int      mem_space;

    int             spatial_dim;
    smesh::ElemType element_type;

    ptrdiff_t nelements;
    ptrdiff_t nnodes;

    idx_t**  elements;
    geom_t** points;

    ptrdiff_t n_owned_nodes;
    ptrdiff_t n_owned_nodes_with_ghosts;
    ptrdiff_t n_global_nodes;

    ptrdiff_t n_owned_elements;
    ptrdiff_t n_owned_elements_with_ghosts;
    ptrdiff_t n_global_elements;
    ptrdiff_t n_shared_elements;

    smesh::large_idx_t* node_mapping;
    int*   node_owner;

    smesh::large_idx_t* element_mapping;

    ptrdiff_t* node_offsets;
    idx_t*     ghosts;
} mesh_t;

struct send_recv_t {
    std::shared_ptr<smesh::Exchange> exchange;
};

idx_t** allocate_elements(const int nxe, const ptrdiff_t n_elements);
void    free_elements(const int nxe, idx_t** elements);
void    select_elements(const int                         nxe,
                        const ptrdiff_t                   nselected,
                        const element_idx_t* const        idx,
                        idx_t** const SFEM_RESTRICT       elements,
                        idx_t** const SFEM_RESTRICT       selection);

geom_t** allocate_points(const int dim, const ptrdiff_t n_points);
void     free_points(const int dim, geom_t** points);
void     select_points(const int                         dim,
                       const ptrdiff_t                   n_points,
                       const idx_t*                      idx,
                       geom_t** const SFEM_RESTRICT      points,
                       geom_t** const SFEM_RESTRICT      selection);

void minmax(const ptrdiff_t n, const geom_t* const SFEM_RESTRICT x, geom_t* xmin, geom_t* xmax);

int mesh_create_nodal_send_recv(MPI_Comm           comm,
                                ptrdiff_t          nnodes,
                                ptrdiff_t          n_owned_nodes,
                                const int*         node_owner,
                                const ptrdiff_t*   node_offsets,
                                const idx_t*       ghosts,
                                send_recv_t* const send_recv);

int mesh_create_nodal_send_recv_deprecated(const mesh_t* const mesh, send_recv_t* const send_recv);

ptrdiff_t mesh_exchange_master_buffer_count(const send_recv_t* const send_recv);

int exchange_add(MPI_Comm                 comm,
                 ptrdiff_t                nnodes,
                 ptrdiff_t                n_owned_nodes,
                 send_recv_t* const       send_recv,
                 real_t* const            inout,
                 real_t* const            real_buffer);

int exchange_add_deprecated(const mesh_t* const mesh,
                            send_recv_t* const  send_recv,
                            real_t* const       inout,
                            real_t* const       real_buffer);

void send_recv_destroy(send_recv_t* const send_recv);

int mesh_write_nodal_field(MPI_Comm           comm,
                           ptrdiff_t          n_owned_nodes,
                           const idx_t*       node_mapping,
                           const char*        output_path,
                           MPI_Datatype       data_type,
                           const void* const  data);

int mesh_write_nodal_field_deprecated(const mesh_t* const mesh,
                                      const char*         output_path,
                                      MPI_Datatype        data_type,
                                      const void* const   data);

#endif
