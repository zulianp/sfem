#ifndef SFEM_READ_MESH_H
#define SFEM_READ_MESH_H

#include <stddef.h>
#include "sfem_base.h"

#include <mpi.h>

typedef struct {
	MPI_Comm comm;
	ptrdiff_t nelements;
	ptrdiff_t nnodes;
	idx_t **elements;
	geom_t **points;
} mesh_t;

int serial_read_tet_mesh(const char *folder, ptrdiff_t *nelements, idx_t *elems[4], ptrdiff_t *nnodes, geom_t *xyz[3]);

#endif  // SFEM_READ_MESH_H
