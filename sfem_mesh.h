#ifndef SFEM_MESH_H
#define SFEM_MESH_H

#include <stddef.h>
#include "sfem_base.h"

#include <mpi.h>

static const int SFEM_MEM_SPACE_HOST = 0;
static const int SFEM_MEM_SPACE_CUDA = 1;
static const int SFEM_MEM_SPACE_NONE = -1;

typedef struct {
	MPI_Comm comm;
	int mem_space;

	int spatial_dim;
	int element_type;

	ptrdiff_t nelements;
	ptrdiff_t nnodes;
	
	idx_t **elements;
	geom_t **points;

	idx_t *mapping;
	int *node_owner;
} mesh_t;

void mesh_destroy(mesh_t *mesh);


#endif //SFEM_MESH_H
