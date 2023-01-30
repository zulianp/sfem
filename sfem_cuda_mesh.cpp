
extern "C" {
#include "sfem_cuda_mesh.h"
#include "sfem_base.h"
}

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

extern "C" void cuda_mesh_create_from_host(const mesh_t *const host_mesh, mesh_t *device_mesh) {
    const ptrdiff_t nelements = host_mesh->nelements;
    const ptrdiff_t nnodes = host_mesh->nnodes;
    const int spatial_dim = host_mesh->spatial_dim;
    const int element_type = host_mesh->element_type;

    const geom_t **xyz = host_mesh->points;
    const idx_t *elems = host_mesh->elems;

    idx_t *hd_elems[element_type];
    idx_t **d_elems = nullptr;

    {  // Copy element indices

        void *ptr;
        SFEM_CUDA_CHECK(cudaMalloc(&ptr, element_type * sizeof(idx_t *)));
        d_elems = (idx_t **)ptr;

        for (int d = 0; d < element_type; ++d) {
            SFEM_CUDA_CHECK(cudaMalloc(&hd_elems[d], nbatch * sizeof(idx_t)));
            SFEM_CUDA_CHECK(cudaMemcpy(hd_elems[d], elems[d], nbatch * sizeof(idx_t), cudaMemcpyHostToDevice));
        }

        SFEM_CUDA_CHECK(cudaMemcpy(d_elems, hd_elems, element_type * sizeof(idx_t *), cudaMemcpyHostToDevice));
    }

    geom_t *hd_xyz[element_type];
    geom_t **d_xyz = nullptr;

    {  // Copy coordinates
        SFEM_CUDA_CHECK(cudaMalloc(&d_xyz, spatial_dim * sizeof(geom_t *)));

        for (int d = 0; d < spatial_dim; ++d) {
            SFEM_CUDA_CHECK(cudaMalloc(&hd_xyz[d], nnodes * sizeof(geom_t)));
            SFEM_CUDA_CHECK(cudaMemcpy(hd_xyz[d], xyz[d], nnodes * sizeof(geom_t), cudaMemcpyHostToDevice));
        }

        SFEM_CUDA_CHECK(cudaMemcpy(d_xyz, hd_xyz, spatial_dim * sizeof(geom_t *), cudaMemcpyHostToDevice));
    }

    device_mesh->comm = host_mesh->comm;
    device_mesh->mem_space = SFEM_MEM_SPACE_CUDA;

    device_mesh->spatial_dim = spatial_dim;
    device_mesh->element_type = element_type;

    device_mesh->nelements = nelements;
    device_mesh->nnodes = nnodes;

    device_mesh->elems = d_elems;
    device_mesh->points = d_xyz;
}

extern "C" void cuda_mesh_free(mesh_t *device_mesh) {
    const int spatial_dim = device_mesh->spatial_dim;
    const int element_type = device_mesh->element_type;

    idx_t **d_elems = device_mesh->elems;
    geom_t **d_xyz = device_mesh->points;

    idx_t *hd_elems[element_type];
    SFEM_CUDA_CHECK(cudaMemcpy(hd_elems, d_elems, element_type * sizeof(idx_t *), cudaMemcpyDeviceToHost));

    {  // Free element indices
        for (int d = 0; d < element_type; ++d) {
            SFEM_CUDA_CHECK(cudaFree(hd_elems[d]));
        }

        SFEM_CUDA_CHECK(cudaFree(d_elems));
    }

    geom_t *hd_xyz[element_type];
    SFEM_CUDA_CHECK(cudaMemcpy(hd_xyz, d_xyz, spatial_dim * sizeof(geom_t *), cudaMemcpyDeviceToHost));

    {  // Free element coordinates
        for (int d = 0; d < spatial_dim; ++d) {
            SFEM_CUDA_CHECK(cudaFree(hd_xyz[d]));
        }

        SFEM_CUDA_CHECK(cudaFree(d_xyz));
    }

    device_mesh->mem_space = SFEM_MEM_SPACE_NONE;

    device_mesh->spatial_dim = 0;
    device_mesh->element_type = 0;

    device_mesh->nelements = 0;
    device_mesh->nnodes = 0;

    device_mesh->elems = nullptr;
    device_mesh->points = nullptr;
}
