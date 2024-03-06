#ifndef SFEM_CUDA_MESH_H
#define SFEM_CUDA_MESH_H

#include "sfem_mesh.h"

void cuda_mesh_create_from_host(const mesh_t *const host_mesh, mesh_t *device_mesh);
void cuda_mesh_free(mesh_t *device_mesh);

#endif //SFEM_CUDA_MESH_H
