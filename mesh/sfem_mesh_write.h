#ifndef SFEM_MESH_WRITE_H
#define SFEM_MESH_WRITE_H
#include "sfem_defs.h"
#include "sfem_mesh.h"


#ifdef __cplusplus
extern "C" {
#endif

int mesh_write(const char *path, const mesh_t *mesh);

int mesh_write_serial(
    const char *path, 
    enum ElemType element_type, 
    const ptrdiff_t n_elements, 
    idx_t **const elements, 
    const int spatial_dim,
    const ptrdiff_t n_nodes, 
    geom_t **const points);

int write_mapped_field(MPI_Comm comm,
                       const char *output_path,
                       const ptrdiff_t n_local,
                       const ptrdiff_t n_global,
                       const idx_t *const mapping,
                       MPI_Datatype data_type,
                       const void *const data);

int mesh_write_nodal_field(MPI_Comm comm,
                           const ptrdiff_t n_owned_nodes,
                           const idx_t *const node_mapping,
                           const char *path,
                           MPI_Datatype data_type,
                           const void *const data);

#ifdef __cplusplus
}
#endif

#endif  // SFEM_MESH_WRITE_H
