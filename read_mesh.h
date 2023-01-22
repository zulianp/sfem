#ifndef SFEM_READ_MESH_H
#define SFEM_READ_MESH_H

#include <stddef.h>
#include "sfem_base.h"

int serial_read_tet_mesh(const char *folder, ptrdiff_t *nelements, idx_t *elems[4], ptrdiff_t *nnodes, geom_t *xyz[4]);

#endif  // SFEM_READ_MESH_H
