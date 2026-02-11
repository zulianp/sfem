#ifndef SFEM_MESH_H
#define SFEM_MESH_H

#include "sfem_base.h"
#include "sfem_defs.h"

#include <mpi.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

static const int SFEM_MEM_SPACE_NONE    = -1;
static const int SFEM_MEM_SPACE_HOST    = 0;
static const int SFEM_MEM_SPACE_CUDA    = 1;
static const int SFEM_MEM_SPACE_MANAGED = 2;
static const int SFEM_MEM_SPACE_UNIFIED = 3;

typedef struct {
    MPI_Comm comm;
    int      mem_space;

    int spatial_dim;
    int element_type;

    ptrdiff_t nelements;
    ptrdiff_t nnodes;

    idx_t  **elements;
    geom_t **points;

    ptrdiff_t n_owned_nodes;
    ptrdiff_t n_owned_nodes_with_ghosts;

    ptrdiff_t n_owned_elements;
    ptrdiff_t n_owned_elements_with_ghosts;
    ptrdiff_t n_shared_elements;

    idx_t *node_mapping;
    int   *node_owner;

    idx_t *element_mapping;

    idx_t *node_offsets;
    idx_t *ghosts;
} mesh_t;

typedef struct {
    // This a reference to the main mesh
    // This "calss/struct" contains only the supplementary informations.
    mesh_t *ref_mesh;

    // An array of size (ref_mesh->nelements x 9) in row-major ordering storing the inverse of the Jacobians.
    real_t *inv_Jacobian;
    real_t *vetices_zero;
} mesh_tet_geom_t;

/**
 * @brief Initialize mesh_tet_geom_t structure for a given mesh
 */
mesh_tet_geom_t mesh_tet_geometry_init(const mesh_t *mesh);

mesh_tet_geom_t *mesh_tet_geometry_alloc(const mesh_t *mesh);

void mesh_tet_geometry_free(mesh_tet_geom_t *geom);

void mesh_tet_geometry_compute_inv_Jacobian(mesh_tet_geom_t *geom);

real_t *get_inv_Jacobian_geom(const mesh_tet_geom_t *geom, ptrdiff_t element_i);

real_t *get_vertices_zero_geom(const mesh_tet_geom_t *geom, ptrdiff_t element_i);

bool                                            //
is_point_out_of_tet(const real_t inv_J_tet[9],  //
                    const real_t tet_origin_x,  //
                    const real_t tet_origin_y,  //
                    const real_t tet_origin_z,  //
                    const real_t vertex_x,      //
                    const real_t vertex_y,      //
                    const real_t vertex_z);     //

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

/**
 * @brief Create mesh in serial mode
 *
 * This function creates the mesh data structure in serial mode.
 *
 * @param mesh Pointer to the mesh data structure
 * @param spatial_dim Spatial dimension
 */
void mesh_create_serial(mesh_t *mesh, int spatial_dim, enum ElemType element_type, ptrdiff_t nelements, idx_t **elements,
                        ptrdiff_t nnodes, geom_t **points);

void mesh_minmax_edge_length(const mesh_t *const mesh, real_t *emin, real_t *emax);

typedef struct {
    ptrdiff_t nelements;
    idx_t   **elements;
} element_block_t;

void mesh_create_shared_elements_block(mesh_t *mesh, element_block_t *block);
void mesh_destroy_shared_elements_block(mesh_t *mesh, element_block_t *block);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_MESH_H
