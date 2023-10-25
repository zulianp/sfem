#include "sfem_mesh.h"
#include "sfem_defs.h"

#include <math.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

void mesh_init(mesh_t *mesh) {
    mesh->comm = MPI_COMM_NULL;
    mesh->mem_space = 0;

    mesh->spatial_dim = 0;
    mesh->element_type = 0;

    mesh->nelements = 0;
    mesh->nnodes = 0;

    mesh->elements = 0;
    mesh->points = 0;

    mesh->n_owned_nodes = 0;
    mesh->n_owned_nodes_with_ghosts = 0;

    mesh->n_owned_elements = 0;
    mesh->n_owned_elements_with_ghosts = 0;
    mesh->n_shared_elements = 0;

    mesh->node_mapping = 0;
    mesh->node_owner = 0;

    mesh->element_mapping = 0;

    mesh->node_offsets = 0;
    mesh->ghosts = 0;
}

void mesh_minmax_edge_length(const mesh_t *const mesh, real_t *emin, real_t *emax) {
    const int nnxe = elem_num_nodes(mesh->element_type);
    *emin = 1e10;
    *emax = 0;

    const int sdim = mesh->spatial_dim;

#pragma omp parallel
    {
#pragma omp for
        for (ptrdiff_t e = 0; e < mesh->nelements; e++) {
            real_t len_min = 1e10;
            real_t len_max = 0;
            for (int i = 0; i < nnxe; i++) {
                idx_t node_i = mesh->elements[i][e];

                for (int j = i + 1; j < nnxe; j++) {
                    idx_t node_j = mesh->elements[j][e];

                    real_t len = 0;
                    for (int d = 0; d < sdim; d++) {
                        real_t diff = mesh->points[d][node_i] - mesh->points[d][node_j];
                        len += diff * diff;
                    }

                    len = sqrt(len);

                    len_min = MIN(len_min, len);
                    len_max = MAX(len_max, len);
                }
            }

#pragma omp critical
            {
                *emin = MIN(*emin, len_min);
                *emax = MAX(*emax, len_max);
            }
        }
    }

    int size;
    MPI_Comm_size(mesh->comm, &size);

    if (size > 1) {
        // IMPLEMENT ME
        assert(0);
        MPI_Abort(mesh->comm, -1);
    }
}

void mesh_destroy(mesh_t *mesh) {
    const int nxe = elem_num_nodes(mesh->element_type);
    for (int d = 0; d < nxe; ++d) {
        free(mesh->elements[d]);
        mesh->elements[d] = 0;
    }

    if (mesh->points) {
        for (int d = 0; d < mesh->spatial_dim; ++d) {
            free(mesh->points[d]);
            mesh->points[d] = 0;
        }
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

    if (mesh->ghosts) {
        free(mesh->ghosts);
    }

    if (mesh->node_offsets) {
        free(mesh->node_offsets);
    }
}

void mesh_create_shared_elements_block(mesh_t *mesh, element_block_t *block) {
    //
    block->nelements = mesh->n_shared_elements;
    const int nn = elem_num_nodes(mesh->element_type);

    block->elements = (idx_t **)malloc(nn * sizeof(idx_t *));

    for (int i = 0; i < nn; i++) {
        block->elements[i] = &mesh->elements[i][mesh->n_owned_elements];
    }
}

void mesh_destroy_shared_elements_block(mesh_t *mesh, element_block_t *block) {
    free(block->elements);
    block->elements = 0;
    block->elements = 0;
}
