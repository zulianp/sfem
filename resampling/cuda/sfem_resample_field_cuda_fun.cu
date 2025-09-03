#include "sfem_resample_field_cuda_fun.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include "sfem_base.h"

/**
 * @brief Make the elements struct
 *
 * @return elems_tet4_device
 */
elems_tet4_device           //
make_elems_tet4_device() {  //

    elems_tet4_device elems_device;

    elems_device.elems_v0 = NULL;
    elems_device.elems_v1 = NULL;
    elems_device.elems_v2 = NULL;
    elems_device.elems_v3 = NULL;

    return elems_device;
}

/**
 * @brief Functions for the elements struct
 *
 * @param elems_device
 * @param nelements
 */
void                                                              //
cuda_allocate_elems_tet4_device(elems_tet4_device* elems_device,  //
                                const ptrdiff_t    nelements) {      //

    cudaMalloc((void**)&elems_device->elems_v0, nelements * sizeof(idx_t));
    cudaMalloc((void**)&elems_device->elems_v1, nelements * sizeof(idx_t));
    cudaMalloc((void**)&elems_device->elems_v2, nelements * sizeof(idx_t));
    cudaMalloc((void**)&elems_device->elems_v3, nelements * sizeof(idx_t));
}

void                                                                    //
cuda_allocate_elems_tet4_device_async(elems_tet4_device* elems_device,  //
                                      const ptrdiff_t    nelements,     //
                                      cudaStream_t       stream) {            //

    cudaMallocAsync((void**)&elems_device->elems_v0, nelements * sizeof(idx_t), stream);
    cudaMallocAsync((void**)&elems_device->elems_v1, nelements * sizeof(idx_t), stream);
    cudaMallocAsync((void**)&elems_device->elems_v2, nelements * sizeof(idx_t), stream);
    cudaMallocAsync((void**)&elems_device->elems_v3, nelements * sizeof(idx_t), stream);
}

/**
 * @brief Functions for the elements struct
 *
 * @param elems_device
 * @param nelements
 */
void                                                                      //
cuda_allocate_elems_tet4_device_managed(elems_tet4_device* elems_device,  //
                                        const ptrdiff_t    nelements) {      //

    cudaMallocManaged((void**)&elems_device->elems_v0, nelements * sizeof(idx_t));
    cudaMallocManaged((void**)&elems_device->elems_v1, nelements * sizeof(idx_t));
    cudaMallocManaged((void**)&elems_device->elems_v2, nelements * sizeof(idx_t));
    cudaMallocManaged((void**)&elems_device->elems_v3, nelements * sizeof(idx_t));
}

/**
 * @brief
 *
 * @param elems
 * @param nelements
 * @param elems_device
 */
void                                                       //
copy_elems_tet4_device(const idx_t**      elems,           // elements from host
                       const ptrdiff_t    nelements,       // number of elements
                       elems_tet4_device* elems_device) {  // to device

    cudaMemcpy(elems_device->elems_v0, elems[0], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaMemcpy(elems_device->elems_v1, elems[1], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaMemcpy(elems_device->elems_v2, elems[2], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
    cudaMemcpy(elems_device->elems_v3, elems[3], nelements * sizeof(idx_t), cudaMemcpyHostToDevice);
}

void                                                           //
copy_elems_tet4_device_async(const idx_t**      elems,         // elements from host
                             const ptrdiff_t    nelements,     // number of elements
                             elems_tet4_device* elems_device,  // to device
                             cudaStream_t       stream) {            // stream

    cudaMemcpyAsync(elems_device->elems_v0, elems[0], nelements * sizeof(idx_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(elems_device->elems_v1, elems[1], nelements * sizeof(idx_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(elems_device->elems_v2, elems[2], nelements * sizeof(idx_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(elems_device->elems_v3, elems[3], nelements * sizeof(idx_t), cudaMemcpyHostToDevice, stream);
}

/**
 * @brief
 *
 * @param elems
 * @param nelements
 * @param elems_device
 */
void                                                               //
copy_elems_tet4_device_unified(const idx_t**      elems,           // elements from host
                               const ptrdiff_t    nelements,       // number of elements
                               elems_tet4_device* elems_device) {  // to device

    elems_device->elems_v0 = (idx_t*)elems[0];
    elems_device->elems_v1 = (idx_t*)elems[1];
    elems_device->elems_v2 = (idx_t*)elems[2];
    elems_device->elems_v3 = (idx_t*)elems[3];
}

/**
 * @brief Free memory for the elements struct
 *
 * @param elems_device
 */
void                                                       //
free_elems_tet4_device(elems_tet4_device* elems_device) {  //

    cudaFree(elems_device->elems_v0);
    cudaFree(elems_device->elems_v1);
    cudaFree(elems_device->elems_v2);
    cudaFree(elems_device->elems_v3);

    elems_device->elems_v0 = NULL;
    elems_device->elems_v1 = NULL;
    elems_device->elems_v2 = NULL;
    elems_device->elems_v3 = NULL;
}

void                                                           //
free_elems_tet4_device_async(elems_tet4_device* elems_device,  //
                             cudaStream_t       stream) {            //
    cudaFreeAsync(elems_device->elems_v0, stream);
    cudaFreeAsync(elems_device->elems_v1, stream);
    cudaFreeAsync(elems_device->elems_v2, stream);
    cudaFreeAsync(elems_device->elems_v3, stream);
    elems_device->elems_v0 = NULL;
    elems_device->elems_v1 = NULL;
    elems_device->elems_v2 = NULL;
    elems_device->elems_v3 = NULL;
}

/**
 * @brief Free memory for the elements struct
 * @brief This function is used when the memory is allocated in the unified memory
 * It simply sets the pointers to NULL (since the memory is not managed by CUDA)
 *
 * @param elems_device
 */
void                                                               //
free_elems_tet4_device_unified(elems_tet4_device* elems_device) {  //
    elems_device->elems_v0 = NULL;
    elems_device->elems_v1 = NULL;
    elems_device->elems_v2 = NULL;
    elems_device->elems_v3 = NULL;
}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
/// Struct for xyz
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

/**
 * @brief Allocate memory for the xyz struct
 *
 * @param xyz_device
 * @param nnodes
 */
void                                                        //
cuda_allocate_xyz_tet4_device(xyz_tet4_device* xyz_device,  //
                              const ptrdiff_t  nnodes) {     //

    cudaMalloc((void**)&xyz_device->x, nnodes * sizeof(geom_t));
    cudaMalloc((void**)&xyz_device->y, nnodes * sizeof(geom_t));
    cudaMalloc((void**)&xyz_device->z, nnodes * sizeof(geom_t));
}

void                                                              //
cuda_allocate_xyz_tet4_device_async(xyz_tet4_device* xyz_device,  //
                                    const ptrdiff_t  nnodes,      //
                                    cudaStream_t     stream) {        //

    cudaMallocAsync((void**)&xyz_device->x, nnodes * sizeof(geom_t), stream);
    cudaMallocAsync((void**)&xyz_device->y, nnodes * sizeof(geom_t), stream);
    cudaMallocAsync((void**)&xyz_device->z, nnodes * sizeof(geom_t), stream);
}

/**
 * @brief Allocate managed memory for the xyz struct
 *
 * @param xyz_device
 * @param nnodes
 */
void                                                                //
cuda_allocate_xyz_tet4_device_managed(xyz_tet4_device* xyz_device,  //
                                      const ptrdiff_t  nnodes) {     //

    cudaError_t err0 = cudaMallocManaged((void**)&xyz_device->x, nnodes * sizeof(geom_t));
    cudaError_t err1 = cudaMallocManaged((void**)&xyz_device->y, nnodes * sizeof(geom_t));
    cudaError_t err2 = cudaMallocManaged((void**)&xyz_device->z, nnodes * sizeof(geom_t));

    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess) {
        fprintf(stderr,
                "Failed to allocate managed memory (error codes: %s, %s, %s)!\n",
                cudaGetErrorString(err0),
                cudaGetErrorString(err1),
                cudaGetErrorString(err2));
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief   Make the xyz struct
 *
 * @return xyz_tet4_device
 */
xyz_tet4_device           //
make_xyz_tet4_device() {  //

    xyz_tet4_device xyz_device;

    xyz_device.x = NULL;
    xyz_device.y = NULL;
    xyz_device.z = NULL;

    return xyz_device;
}

/**
 * @brief Free memory for the xyz struct
 *
 * @param xyz_device
 */
void                                                 //
free_xyz_tet4_device(xyz_tet4_device* xyz_device) {  //
    cudaFree(xyz_device->x);
    cudaFree(xyz_device->y);
    cudaFree(xyz_device->z);

    xyz_device->x = NULL;
    xyz_device->y = NULL;
    xyz_device->z = NULL;
}

void                                                     //
free_xyz_tet4_device_async(xyz_tet4_device* xyz_device,  //
                           cudaStream_t     stream) {        //

    cudaFreeAsync(xyz_device->x, stream);
    cudaFreeAsync(xyz_device->y, stream);
    cudaFreeAsync(xyz_device->z, stream);

    xyz_device->x = NULL;
    xyz_device->y = NULL;
    xyz_device->z = NULL;
}

/**
 * @brief Free memory for the xyz struct
 * @brief This function is used when the memory is allocated in the unified memory
 * It simply sets the pointers to NULL (since the memory is not managed by CUDA)
 *
 * @param xyz_device
 */
void                                                         //
free_xyz_tet4_device_unified(xyz_tet4_device* xyz_device) {  //
    xyz_device->x = NULL;
    xyz_device->y = NULL;
    xyz_device->z = NULL;
}

/**
 * @brief Copy the xyz struct from host to device
 *
 * @param xyz
 * @param nnodes
 * @param xyz_device
 */
void                                                 //
copy_xyz_tet4_device(const geom_t**   xyz,           //
                     const ptrdiff_t  nnodes,        //
                     xyz_tet4_device* xyz_device) {  //
                                                     //
    cudaMemcpy(xyz_device->x, xyz[0], nnodes * sizeof(geom_t), cudaMemcpyHostToDevice);
    cudaMemcpy(xyz_device->y, xyz[1], nnodes * sizeof(geom_t), cudaMemcpyHostToDevice);
    cudaMemcpy(xyz_device->z, xyz[2], nnodes * sizeof(geom_t), cudaMemcpyHostToDevice);
}

void                                                     //
copy_xyz_tet4_device_async(const geom_t**   xyz,         //
                           const ptrdiff_t  nnodes,      //
                           xyz_tet4_device* xyz_device,  //
                           cudaStream_t     stream) {        //

    cudaMemcpyAsync(xyz_device->x, xyz[0], nnodes * sizeof(geom_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(xyz_device->y, xyz[1], nnodes * sizeof(geom_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(xyz_device->z, xyz[2], nnodes * sizeof(geom_t), cudaMemcpyHostToDevice, stream);
}

/**
 * @brief Copy the xyz struct from host to device
 *
 * @param xyz
 * @param nnodes
 * @param xyz_device
 */
void                                                         //
copy_xyz_tet4_device_unified(const geom_t**   xyz,           //
                             const ptrdiff_t  nnodes,        //
                             xyz_tet4_device* xyz_device) {  //
                                                             //
    xyz_device->x = (geom_t*)xyz[0];
    xyz_device->y = (geom_t*)xyz[1];
    xyz_device->z = (geom_t*)xyz[2];
}