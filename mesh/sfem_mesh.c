#include "sfem_mesh.h"

#include "sfem_defs.h"

void mesh_destroy(mesh_t *mesh) {
    for (int d = 0; d < mesh->element_type; ++d) {
        free(mesh->elements[d]);
        mesh->elements[d] = 0;
    }

    for (int d = 0; d < mesh->spatial_dim; ++d) {
        free(mesh->points[d]);
        mesh->points[d] = 0;
    }

    free(mesh->node_mapping);
    free(mesh->node_owner);
    free(mesh->element_mapping);

    mesh->comm = MPI_COMM_NULL;
    mesh->mem_space = SFEM_MEM_SPACE_NONE;

    mesh->spatial_dim = 0;
    mesh->element_type = 0;

    mesh->nelements = 0;
    mesh->nnodes = 0;
}


void mesh_create_shared_elements_block(mesh_t *mesh, element_block_t *block)
{
    // 
    block->nelements = mesh->n_shared_elements;
    const int nn = elem_num_nodes(mesh->element_type);

    block->elements = (idx_t *)malloc(nn * sizeof(idx_t *));

    for(int i = 0; i < nn; i++) {
        block->elements[i] = &mesh->elements[i][mesh->n_owned_elements];
    }
}   

void mesh_destroy_shared_elements_block(mesh_t *mesh, element_block_t *block)
{
    free(block->elements);
    block->elements = 0;
    block->elements = 0;
}
