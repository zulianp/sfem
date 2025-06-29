#ifndef SFEM_READ_MESH_H
#define SFEM_READ_MESH_H

#include <stddef.h>

#include "sfem_base.h"
#include "sfem_mesh.h"

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif



int mesh_read_serial(const char *folder,
                     int *nnodesxelem_out,
                     ptrdiff_t *nelements_out,
                     idx_t ***elems_out,
                     int *spatial_dim_out,
                     ptrdiff_t *nnodes_out,
                     geom_t ***xyz_out);

int mesh_node_ids(mesh_t *mesh, idx_t *const ids);
int mesh_read_elements(MPI_Comm comm,
                       const int nnodesxelem,
                       const char *folder,
                       idx_t **const elems,
                       ptrdiff_t *const n_local_elements,
                       ptrdiff_t *const n_elements);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_READ_MESH_H
