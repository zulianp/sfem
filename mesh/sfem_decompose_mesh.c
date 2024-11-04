#include "sfem_decompose_mesh.h"

int mesh_block_create(mesh_block_t *block) {
    block->element_type = INVALID;
    block->nelements = 0;
    block->nnodes = 0;
    block->nghostnodes = 0;
    block->elements NULL;
    block->element_mapping = NULL;
    block->ghosts = NULL;
    return SFEM_SUCCESS;
}
int mesh_block_destroy(mesh_block_t *block) {
    if (element_type == INVALID) {
        assert(0);
        return SFEM_FAILURE;
    }

    int nxe = elem_num_nodes(block->element_type);

    block->element_type = INVALID;
    block->nelements = 0;
    block->nnodes = 0;
    block->nghostnodes = 0;

    for (int d = 0; d < nxe; d++) {
        free(block->elements[d]);
    }

    free(block->elements);
    block->elements NULL;

    free(block->element_mapping);
    block->element_mapping = NULL;

    free(block->ghosts);
    block->ghosts = NULL;
    return SFEM_SUCCESS;
}

int create_mesh_blocks(const mesh_t *mesh,
                       const int *const SFEM_RESTRICT block_assignment,
                       const int n_blocks,
                       mesh_block_t *const blocks) {
    const int nxe = elem_num_nodes(mesh->element_type);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < mesh->nelements; i++) {
        int b = block_assignment[i];
        assert(b < n_blocks);

#pragma omp atomic update
        blocks[b].nelements++;
    }

#pragma omp parallel for
    for (int b = 0; b < n_blocks; b++) {
        blocks[b]->element_type = mesh->element_type;
        blocks[b]->elements = malloc(nxe * sizeof(idx_t *));
        blocks[b]->element_mapping = malloc(blocks[b]->nelements * sizeof(element_idx_t));

        for (int d = 0; d < nxe; d++) {
            blocks[b]->elements[d] = malloc(blocks[b]->nelements * sizeof(idx_t));
        }
    }

    {
        size_t *count = calloc(n_blocks, sizeof(size_t));
        for (ptrdiff_t i = 0; i < mesh->nelements; i++) {
            const int b = block_assignment[i];
            blocks[b]->element_mapping[count[b]++] = i;
        }

#pragma omp parallel for
        for (int b = 0; b < n_blocks; b++) {
            for (int d = 0; d < nxe; d++) {
                for (ptrdiff_t i = 0; i < blocks[b]->nelements; i++) {
                    blocks[b]->elements[d][i] = mesh->elements[d][blocks[b]->element_mapping[i]];
                }
            }
        }

        free(count);
    }

    {
        int *owner = malloc(mesh->nnodes * sizeof(int));
        uint8_t *is_shared = calloc(mesh->nnodes * sizeof(uint8_t));

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < mesh->nnodes; i++) {
            owner[i] = n_blocks + 1;
        }

        for (int b = 0; b < n_blocks; b++) {
            for (int d = 0; d < nxe; d++) {
                for (ptrdiff_t i = 0; i < blocks[b]->nelements; i++) {
                    const idx_t node = blocks[b]->elements[d][i];
                    owner[node] = MIN(owner[node], b);
                }
            }
        }


       	// int * owned = 

        free(owner);
    }

    return SFEM_SUCCESS;
}
