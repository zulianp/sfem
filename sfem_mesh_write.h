#ifndef SFEM_MESH_WRITE_H
#define SFEM_MESH_WRITE_H

#include "sfem_mesh.h"

int mesh_write(const char *path, const mesh_t *mesh);

int write_mapped_field(MPI_Comm comm,
                       const char *folder,
                       const ptrdiff_t n_local,
                       const ptrdiff_t n_global,
                       const idx_t * const mapping,
                       MPI_Datatype data_type,
                       void *const data);

#endif //SFEM_MESH_WRITE_H
