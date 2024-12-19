#ifndef SFEM_RESAMPLE_FIELD_CUDA_FUN_CUH
#define SFEM_RESAMPLE_FIELD_CUDA_FUN_CUH

#include <stdio.h>
#include "sfem_base.h"

// Struct for the elements
typedef struct {
    int* elems_v0;
    int* elems_v1;
    int* elems_v2;
    int* elems_v3;
} elems_tet4_device;

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

    cudaMalloc((void**)&elems_device->elems_v0, nelements * sizeof(int));
    cudaMalloc((void**)&elems_device->elems_v1, nelements * sizeof(int));
    cudaMalloc((void**)&elems_device->elems_v2, nelements * sizeof(int));
    cudaMalloc((void**)&elems_device->elems_v3, nelements * sizeof(int));
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

    cudaMallocManaged((void**)&elems_device->elems_v0, nelements * sizeof(int));
    cudaMallocManaged((void**)&elems_device->elems_v1, nelements * sizeof(int));
    cudaMallocManaged((void**)&elems_device->elems_v2, nelements * sizeof(int));
    cudaMallocManaged((void**)&elems_device->elems_v3, nelements * sizeof(int));
}

/**
 * @brief
 *
 * @param elems
 * @param nelements
 * @param elems_device
 */
void                                                       //
copy_elems_tet4_device(const int**        elems,           // elements from host
                       const ptrdiff_t    nelements,       // number of elements
                       elems_tet4_device* elems_device) {  // to device

    cudaMemcpy(elems_device->elems_v0, elems[0], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(elems_device->elems_v1, elems[1], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(elems_device->elems_v2, elems[2], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(elems_device->elems_v3, elems[3], nelements * sizeof(int), cudaMemcpyHostToDevice);
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

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
/// Struct for xyz
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

// Struct for xyz
typedef struct {
    float* x;
    float* y;
    float* z;
} xyz_tet4_device;

/**
 * @brief Allocate memory for the xyz struct
 *
 * @param xyz_device
 * @param nnodes
 */
void                                                        //
cuda_allocate_xyz_tet4_device(xyz_tet4_device* xyz_device,  //
                              const ptrdiff_t  nnodes) {     //

    cudaMalloc((void**)&xyz_device->x, nnodes * sizeof(float));
    cudaMalloc((void**)&xyz_device->y, nnodes * sizeof(float));
    cudaMalloc((void**)&xyz_device->z, nnodes * sizeof(float));
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

    cudaMallocManaged((void**)&xyz_device->x, nnodes * sizeof(float));
    cudaMallocManaged((void**)&xyz_device->y, nnodes * sizeof(float));
    cudaMallocManaged((void**)&xyz_device->z, nnodes * sizeof(float));
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

/**
 * @brief Copy the xyz struct from host to device
 *
 * @param xyz
 * @param nnodes
 * @param xyz_device
 */
void                                                 //
copy_xyz_tet4_device(const float**    xyz,           //
                     const ptrdiff_t  nnodes,        //
                     xyz_tet4_device* xyz_device) {  //
                                                     //
    cudaMemcpy(xyz_device->x, xyz[0], nnodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xyz_device->y, xyz[1], nnodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xyz_device->z, xyz[2], nnodes * sizeof(float), cudaMemcpyHostToDevice);
}

#endif  // SFEM_RESAMPLE_FIELD_CUDA_FUN_CUH