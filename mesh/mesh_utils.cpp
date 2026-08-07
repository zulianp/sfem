#include "mesh_utils.hpp"

#include "smesh_common.hpp"
#include "smesh_communicator.hpp"
#include "smesh_distributed_write.hpp"
#include "smesh_write.hpp"

#include <algorithm>
#include <assert.h>
#include <string.h>

idx_t** allocate_elements(const int nxe, const ptrdiff_t n_elements) {
    idx_t** elements = (idx_t**)malloc(nxe * sizeof(idx_t*));
    for (int d = 0; d < nxe; d++) {
        elements[d] = (idx_t*)malloc(n_elements * sizeof(idx_t));
    }

    return elements;
}

void free_elements(const int nxe, idx_t** elements) {
    for (int d = 0; d < nxe; d++) {
        free(elements[d]);
    }

    free(elements);
}

void select_elements(const int                         nxe,
                     const ptrdiff_t                   nselected,
                     const element_idx_t* const        idx,
                     idx_t** const SFEM_RESTRICT       elements,
                     idx_t** const SFEM_RESTRICT       selection) {
    for (int d = 0; d < nxe; d++) {
        for (ptrdiff_t i = 0; i < nselected; i++) {
            selection[d][i] = elements[d][idx[i]];
        }
    }
}

geom_t** allocate_points(const int dim, const ptrdiff_t n_points) {
    geom_t** points = (geom_t**)malloc(dim * sizeof(geom_t*));
    for (int d = 0; d < dim; d++) {
        points[d] = (geom_t*)malloc(n_points * sizeof(geom_t));
    }

    return points;
}

void free_points(const int dim, geom_t** points) {
    for (int d = 0; d < dim; d++) {
        free(points[d]);
    }

    free(points);
}

void select_points(const int                         dim,
                   const ptrdiff_t                   n_points,
                   const idx_t*                      idx,
                   geom_t** const SFEM_RESTRICT      points,
                   geom_t** const SFEM_RESTRICT      selection) {
    for (int d = 0; d < dim; d++) {
        for (ptrdiff_t i = 0; i < n_points; i++) {
            selection[d][i] = points[d][idx[i]];
        }
    }
}

void minmax(const ptrdiff_t n, const geom_t* const SFEM_RESTRICT x, geom_t* xmin, geom_t* xmax) {
    smesh::minmax(n, x, xmin, xmax);
}

static smesh::PrimitiveType primitive_type_from_mpi(MPI_Datatype data_type) {
    if (data_type == MPI_FLOAT) return smesh::SMESH_FLOAT32;
    if (data_type == MPI_DOUBLE) return smesh::SMESH_FLOAT64;
    if (data_type == MPI_CHAR) return smesh::SMESH_CHAR;
    if (data_type == MPI_INT8_T) return smesh::SMESH_INT8;
    if (data_type == MPI_INT16_T || data_type == MPI_SHORT) return smesh::SMESH_INT16;
    if (data_type == MPI_INT32_T || data_type == MPI_INT) return smesh::SMESH_INT32;
    if (data_type == MPI_INT64_T || data_type == MPI_LONG_LONG) return smesh::SMESH_INT64;
    if (data_type == MPI_UINT8_T || data_type == MPI_UNSIGNED_CHAR) return smesh::SMESH_UINT8;
    if (data_type == MPI_UINT16_T || data_type == MPI_UNSIGNED_SHORT) return smesh::SMESH_UINT16;
    if (data_type == MPI_UINT64_T || data_type == MPI_UNSIGNED_LONG_LONG) return smesh::SMESH_UINT64;
    SFEM_ERROR("Unsupported MPI datatype for mesh field output\n");
    return smesh::SMESH_TYPE_UNDEFINED;
}

int mesh_create_nodal_send_recv(MPI_Comm           comm,
                                ptrdiff_t          nnodes,
                                ptrdiff_t          n_owned_nodes,
                                const int*         node_owner,
                                const ptrdiff_t*   node_offsets,
                                const idx_t*       ghosts,
                                send_recv_t* const send_recv) {
    assert(send_recv);
    send_recv->exchange = smesh::Exchange::create(smesh::Communicator::wrap(comm),
                                                  smesh::Exchange::ExchangeScope::GhostsOnly,
                                                  nnodes,
                                                  n_owned_nodes,
                                                  node_owner,
                                                  node_offsets,
                                                  ghosts);
    return SFEM_SUCCESS;
}

int mesh_create_nodal_send_recv_deprecated(const mesh_t* const mesh, send_recv_t* const send_recv) {
    return mesh_create_nodal_send_recv(mesh->comm,
                                       mesh->nnodes,
                                       mesh->n_owned_nodes,
                                       mesh->node_owner,
                                       mesh->node_offsets,
                                       mesh->ghosts,
                                       send_recv);
}

ptrdiff_t mesh_exchange_master_buffer_count(const send_recv_t* const send_recv) {
    SFEM_UNUSED(send_recv);
    return 0;
}

int exchange_add(MPI_Comm                 comm,
                 ptrdiff_t                nnodes,
                 ptrdiff_t                n_owned_nodes,
                 send_recv_t* const       send_recv,
                 real_t* const            inout,
                 real_t* const            real_buffer) {
    SFEM_UNUSED(comm);
    SFEM_UNUSED(nnodes);
    SFEM_UNUSED(n_owned_nodes);
    SFEM_UNUSED(real_buffer);
    assert(send_recv);
    return send_recv->exchange->scatter_add(inout);
}

int exchange_add_deprecated(const mesh_t* const mesh,
                            send_recv_t* const  send_recv,
                            real_t* const       inout,
                            real_t* const       real_buffer) {
    return exchange_add(mesh->comm, mesh->nnodes, mesh->n_owned_nodes, send_recv, inout, real_buffer);
}

void send_recv_destroy(send_recv_t* const send_recv) {
    if (send_recv) {
        send_recv->exchange.reset();
    }
}

int mesh_write_nodal_field(MPI_Comm           comm,
                           ptrdiff_t          n_owned_nodes,
                           const idx_t*       node_mapping,
                           const char*        output_path,
                           MPI_Datatype       data_type,
                           const void* const  data) {
    int size = 1;
    MPI_Comm_size(comm, &size);

    const smesh::PrimitiveType primitive_type = primitive_type_from_mpi(data_type);
    if (size > 1) {
        ptrdiff_t n_global_nodes = 0;
        MPI_Allreduce(&n_owned_nodes, &n_global_nodes, 1, MPI_LONG, MPI_SUM, comm);
        return smesh::write_mapped_field(comm, smesh::Path(output_path), n_owned_nodes, n_global_nodes, node_mapping, data_type, data);
    }

    return smesh::array_write(smesh::Path(output_path), primitive_type, data, n_owned_nodes);
}

int mesh_write_nodal_field_deprecated(const mesh_t* const mesh,
                                      const char*         output_path,
                                      MPI_Datatype        data_type,
                                      const void* const   data) {
    int size = 1;
    MPI_Comm_size(mesh->comm, &size);

    const smesh::PrimitiveType primitive_type = primitive_type_from_mpi(data_type);
    if (size > 1) {
        return smesh::write_mapped_field(mesh->comm,
                                         smesh::Path(output_path),
                                         mesh->n_owned_nodes,
                                         mesh->n_global_nodes,
                                         mesh->node_mapping,
                                         data_type,
                                         data);
    }

    return smesh::array_write(smesh::Path(output_path), primitive_type, data, mesh->n_owned_nodes);
}
