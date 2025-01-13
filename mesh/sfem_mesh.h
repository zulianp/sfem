#ifndef SFEM_MESH_H
#define SFEM_MESH_H

#include "sfem_base.h"
#include "sfem_defs.h"

#include <mpi.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

static const int SFEM_MEM_SPACE_HOST = 0;
static const int SFEM_MEM_SPACE_CUDA = 1;
static const int SFEM_MEM_SPACE_NONE = -1;

typedef struct {
    MPI_Comm comm;
    int mem_space;

    int spatial_dim;
    enum ElemType element_type;

    ptrdiff_t nelements;
    ptrdiff_t nnodes;

    idx_t **elements;
    geom_t **points;

    ptrdiff_t n_owned_nodes;
    ptrdiff_t n_owned_nodes_with_ghosts;

    ptrdiff_t n_owned_elements;
    ptrdiff_t n_owned_elements_with_ghosts;
    ptrdiff_t n_shared_elements;

    idx_t *node_mapping;
    int *node_owner;

    idx_t *element_mapping;

    idx_t *node_offsets;
    idx_t *ghosts;
} mesh_t;


/**
 * @brief Initialize mesh data structure to empty
 *
 * This function initializes the mesh data structure to an empty state.
 *
 * @param mesh Pointer to the mesh data structure
 */
void mesh_init(mesh_t *mesh);

/**
 * @brief Destroy mesh data structure
 *
 * This function cleans up and destroys the mesh data structure.
 *
 * @param mesh Pointer to the mesh data structure
 */
void mesh_destroy(mesh_t *mesh);


void mesh_create_reference_hex8_cube(mesh_t *mesh);
void mesh_create_hex8_cube(mesh_t *mesh, const int nx, const int ny, const int nz);

/**
 * @brief Create mesh in serial mode
 *
 * This function creates the mesh data structure in serial mode.
 *
 * @param mesh Pointer to the mesh data structure
 * @param spatial_dim Spatial dimension
 */
void mesh_create_serial(
    mesh_t *mesh,
    int spatial_dim,
    enum ElemType element_type,
    ptrdiff_t nelements,
    idx_t **elements,
    ptrdiff_t nnodes,
    geom_t **points
    );

void mesh_minmax_edge_length(const mesh_t *const mesh, real_t *emin, real_t *emax);

typedef struct {
    ptrdiff_t nelements;
    idx_t **elements;
} element_block_t;

void mesh_create_shared_elements_block(mesh_t *mesh, element_block_t *block);
void mesh_destroy_shared_elements_block(mesh_t *mesh, element_block_t *block);

void remap_elements_to_contiguous_index(
    const ptrdiff_t n_elements, 
    const int nxe, idx_t **elements,
    ptrdiff_t *const out_n_contiguous,
    idx_t **const out_node_mapping);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_MESH_H
