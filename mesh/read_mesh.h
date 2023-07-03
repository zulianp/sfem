#ifndef SFEM_READ_MESH_H
#define SFEM_READ_MESH_H

#include <stddef.h>

#include "sfem_base.h"
#include "sfem_mesh.h"

#include <mpi.h>

int serial_read_tet_mesh(const char *folder, ptrdiff_t *nelements, idx_t *elems[4], ptrdiff_t *nnodes, geom_t *xyz[3]);
int mesh_read(MPI_Comm comm, const char *folder, mesh_t *mesh);
int mesh_surf_read(MPI_Comm comm, const char *folder, mesh_t *mesh);
int mesh_node_ids(mesh_t *mesh, idx_t *const ids);

#endif  // SFEM_READ_MESH_H
