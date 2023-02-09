#include "sfem_mesh.h"

void mesh_destroy(mesh_t *mesh) {
    for (int d = 0; d < mesh->element_type; ++d) {
        free(mesh->elements[d]);
        mesh->elements[d] = 0;
    }

    for (int d = 0; d < mesh->spatial_dim; ++d) {
        free(mesh->points[d]);
        mesh->points[d] = 0;
    }

    free(mesh->mapping);
    free(mesh->node_owner);

    mesh->comm = MPI_COMM_NULL;
    mesh->mem_space = SFEM_MEM_SPACE_NONE;

    mesh->spatial_dim = 0;
    mesh->element_type = 0;

    mesh->nelements = 0;
    mesh->nnodes = 0;
}