#include "sfem_decompose_mesh.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

int mesh_block_create(mesh_block_t *block) {
    block->element_type = INVALID;
    block->nelements = 0;
    block->nowned = 0;
    block->nowned_ghosted = 0;
    block->nshared = 0;
    block->elements = NULL;
    block->element_mapping = NULL;
    block->node_mapping = NULL;
    return SFEM_SUCCESS;
}
int mesh_block_destroy(mesh_block_t *block) {
    if (block->element_type == INVALID) {
        assert(0);
        return SFEM_FAILURE;
    }

    const int nxe = elem_num_nodes(block->element_type);

    block->element_type = INVALID;
    block->nelements = 0;
    block->nowned = 0;
    block->nowned_ghosted = 0;
    block->nshared = 0;

    for (int d = 0; d < nxe; d++) {
        free(block->elements[d]);
    }

    free(block->elements);
    block->elements = NULL;

    free(block->element_mapping);
    block->element_mapping = NULL;

    free(block->node_mapping);
    block->node_mapping = NULL;
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
        blocks[b].element_type = mesh->element_type;
        blocks[b].elements = malloc(nxe * sizeof(idx_t *));
        blocks[b].element_mapping = malloc(blocks[b].nelements * sizeof(element_idx_t));

        for (int d = 0; d < nxe; d++) {
            blocks[b].elements[d] = malloc(blocks[b].nelements * sizeof(idx_t));
        }
    }

    {
        size_t *count = calloc(n_blocks, sizeof(size_t));

        for (ptrdiff_t i = 0; i < mesh->nelements; i++) {
            const int b = block_assignment[i];
            blocks[b].element_mapping[count[b]++] = i;
        }

#pragma omp parallel for
        for (int b = 0; b < n_blocks; b++) {
            for (int d = 0; d < nxe; d++) {
                for (ptrdiff_t i = 0; i < blocks[b].nelements; i++) {
                    blocks[b].elements[d][i] = mesh->elements[d][blocks[b].element_mapping[i]];
                }
            }
        }

        free(count);
    }

    {
        int *owner = malloc(mesh->nnodes * sizeof(int));
        int *max_share = calloc(mesh->nnodes, sizeof(int));
        idx_t *global_to_local = malloc(mesh->nnodes * sizeof(idx_t));

#pragma omp parallel for
        for (ptrdiff_t i = 0; i < mesh->nnodes; i++) {
            owner[i] = n_blocks + 1;
            global_to_local[i] = SFEM_IDX_INVALID;
        }

        for (int b = 0; b < n_blocks; b++) {
            for (int d = 0; d < nxe; d++) {
                for (ptrdiff_t i = 0; i < blocks[b].nelements; i++) {
                    const idx_t node = blocks[b].elements[d][i];
                    owner[node] = MIN(owner[node], b);
                    max_share[node] = MAX(max_share[node], b);
                }
            }
        }

        for (int b = 0; b < n_blocks; b++) {
            ptrdiff_t nowned = 0;
            ptrdiff_t nowned_ghosted = 0;
            ptrdiff_t nshared = 0;

            for (int d = 0; d < nxe; d++) {
                for (ptrdiff_t i = 0; i < blocks[b].nelements; i++) {
                    const idx_t node = blocks[b].elements[d][i];

                    if (owner[node] == b) {
                        nowned++;
                        if (max_share[node] != b) {
                            nowned_ghosted++;
                        }
                    } else {
                        nshared++;
                    }
                }
            }

            const ptrdiff_t ntotal = nowned + nowned_ghosted + nshared;
            blocks[b].nowned = nowned;
            blocks[b].nowned_ghosted = nowned_ghosted;
            blocks[b].nshared = nshared;
            blocks[b].nnodes = ntotal;

            ptrdiff_t offset_owned = 0;
            ptrdiff_t offset_owned_ghosted = nowned;
            ptrdiff_t offset_shared = nowned + nowned_ghosted;

            blocks[b].node_mapping = malloc(ntotal * sizeof(idx_t));
            blocks[b].owner = malloc(nshared * sizeof(int));

            for (int d = 0; d < nxe; d++) {
                for (ptrdiff_t i = 0; i < blocks[b].nelements; i++) {
                    const idx_t node = blocks[b].elements[d][i];

                    if (global_to_local[node] == SFEM_IDX_INVALID) {
                        if (owner[node] == b) {
                            if (max_share[node] != b) {
                                global_to_local[node] = offset_owned_ghosted++;
                            } else {
                                global_to_local[node] = offset_owned++;
                            }
                        } else {
                            global_to_local[node] = offset_shared++;
                        }

                        blocks[b].node_mapping[global_to_local[node]] = node;
                    }

                    blocks[b].elements[d][i] = global_to_local[node];
                }
            }

            for (ptrdiff_t i = 0; i < nshared; i++) {
                const ptrdiff_t node = blocks[b].node_mapping[nowned + nowned_ghosted + i];
                blocks[b].owner[i] = owner[node];
            }

// Clean-up mapping
#pragma omp parallel for
            for (ptrdiff_t i = 0; i < ntotal; i++) {
                global_to_local[blocks[b].node_mapping[i]] = SFEM_IDX_INVALID;
            }
        }

        {  // Construct ghost gather/scatter index

            // Adjacency-matrix
            count_t *adjaciency_matrix = calloc(n_blocks * n_blocks, sizeof(count_t));

#pragma omp parallel for
            for (int b = 0; b < n_blocks; b++) {
                for (ptrdiff_t i = 0; i < blocks[b].nnodes; i++) {
                    if (blocks[b].owner[i] != b) {
#pragma omp atomic update
                        adjaciency_matrix[owner[i] * n_blocks + b]++;
                    }
                }
            }

#pragma omp parallel for
            for (int b = 0; b < n_blocks; b++) {
                blocks[b].rowptr_incoming = calloc(n_blocks + 1, sizeof(count_t));
                blocks[b].rowptr_outgoing = calloc(n_blocks + 1, sizeof(count_t));

                for (int neigh = 0; neigh < n_blocks; neigh++) {
                    const count_t incoming = adjaciency_matrix[b * n_blocks + neigh];
                    const count_t outgoing = adjaciency_matrix[neigh * n_blocks + b];

                    blocks[b].rowptr_incoming[neigh + 1] =
                            incoming + blocks[b].rowptr_incoming[neigh];

                    blocks[b].rowptr_outgoing[neigh + 1] =
                            outgoing + blocks[b].rowptr_outgoing[neigh];
                }

                blocks[b].colidx_incoming =
                        malloc(blocks[b].rowptr_incoming[n_blocks] * sizeof(idx_t));

                blocks[b].colidx_outgoing =
                        malloc(blocks[b].rowptr_outgoing[n_blocks] * sizeof(idx_t));
            }

            memset(adjaciency_matrix, 0, n_blocks * n_blocks * sizeof(count_t));
            for (int b = 0; b < n_blocks; b++) {
                for (ptrdiff_t i = 0; i < blocks[b].nnodes; i++) {
                    const int neigh = blocks[b].owner[i];
                    if (neigh != b) {
                        const count_t offset = adjaciency_matrix[neigh * n_blocks + b];
                        const count_t outgoing_offset = offset + blocks[b].rowptr_outgoing[neigh];
                        const count_t neigh_incoming_offset =
                                offset + blocks[neigh].rowptr_incoming[b];

                        blocks[b].colidx_outgoing[outgoing_offset] = i;

                        // Assigned with global indexing (global to local is done later)
                        blocks[neigh].colidx_incoming[neigh_incoming_offset] =
                                blocks[b].node_mapping[i];

                        adjaciency_matrix[neigh * n_blocks + b]++;
                    }
                }
            }

            for (int b = 0; b < n_blocks; b++) {
                const ptrdiff_t offset = blocks[b].nowned + blocks[b].nowned_ghosted;
                for (ptrdiff_t i = 0; i < blocks[b].nshared; i++) {
                    idx_t node = blocks[b].node_mapping[offset + i];
                    global_to_local[node] = offset + i;
                }

                for (int neigh = 0; neigh < n_blocks; neigh++) {
                	// global to local every index in incoming

                }

                // Clean-up
                for (ptrdiff_t i = 0; i < blocks[b].nshared; i++) {
                    idx_t node = blocks[b].node_mapping[offset + i];
                    global_to_local[node] = SFEM_IDX_INVALID;
                }
            }

            free(adjaciency_matrix);
        }

        free(owner);
        free(max_share);
        free(global_to_local);
    }

    return SFEM_SUCCESS;
}
