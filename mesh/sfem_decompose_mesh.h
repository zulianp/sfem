#ifndef SFEM_DECOMPOSE_MESH_H
#define SFEM_DECOMPOSE_MESH_H

#include "sfem_base.h"
#include "sfem_defs.h"

#include "sfem_mesh.h"

#include <mpi.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    enum ElemType element_type;
    ptrdiff_t nelements;    

    ptrdiff_t nnodes;
    ptrdiff_t nowned;
    ptrdiff_t nowned_ghosted;
    ptrdiff_t nshared;

    idx_t **elements;
    element_idx_t *element_mapping;
    idx_t *node_mapping;

    int *owner;

    count_t *rowptr_incoming;
    idx_t *colidx_incoming;

    count_t *rowptr_outgoing;
    idx_t *colidx_outgoing;
} mesh_block_t;

int mesh_block_create(mesh_block_t *block);
int mesh_block_destroy(mesh_block_t *block);
int create_mesh_blocks(const mesh_t *mesh,
                       const int *const SFEM_RESTRICT block_assignment,
                       const int n_blocks,
                       mesh_block_t *const blocks);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_DECOMPOSE_MESH_H
