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


	ptrdiff_t n_owned_nodes;
	ptrdiff_t n_owned_elements;
	ptrdiff_t n_shared_elements;

	idx_t *node_mapping;
	int *node_owner;

	idx_t *element_mapping;
} mesh_t;

void mesh_destroy(mesh_t *mesh);


#endif //SFEM_MESH_H
