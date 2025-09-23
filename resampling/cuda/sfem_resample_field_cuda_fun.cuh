#ifndef SFEM_RESAMPLE_FIELD_CUDA_FUN_CUH
#define SFEM_RESAMPLE_FIELD_CUDA_FUN_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include "sfem_base.h"

// Struct for the elements
typedef struct {
    idx_t* elems_v0;
    idx_t* elems_v1;
    idx_t* elems_v2;
    idx_t* elems_v3;

    ptrdiff_t nelements;

} elems_tet4_device;

/**
 * @brief Make the elements struct
 *
 * @return elems_tet4_device
 */
elems_tet4_device          //
make_elems_tet4_device();  //

/**
 * @brief Functions for the elements struct
 *
 * @param elems_device
 * @param nelements
 */
void                                                              //
cuda_allocate_elems_tet4_device(elems_tet4_device* elems_device,  //
                                const ptrdiff_t    nelements);       //

void                                                                    //
cuda_allocate_elems_tet4_device_async(elems_tet4_device* elems_device,  //
                                      const ptrdiff_t    nelements,     //
                                      cudaStream_t       stream);             //

/**
 * @brief Functions for the elements struct
 *
 * @param elems_device
 * @param nelements
 */
void                                                                      //
cuda_allocate_elems_tet4_device_managed(elems_tet4_device* elems_device,  //
                                        const ptrdiff_t    nelements);       //

/**
 * @brief
 *
 * @param elems
 * @param nelements
 * @param elems_device
 */
void                                                      //
copy_elems_tet4_device(const idx_t**      elems,          //
                       const ptrdiff_t    nelements,      //
                       elems_tet4_device* elems_device);  //

void                                                           //
copy_elems_tet4_device_async(const idx_t**      elems,         // elements from host
                             const ptrdiff_t    nelements,     // number of elements
                             elems_tet4_device* elems_device,  // to device
                             cudaStream_t       stream);             // stream

/**
 * @brief
 *
 * @param elems
 * @param nelements
 * @param elems_device
 */
void                                                              //
copy_elems_tet4_device_unified(const idx_t**      elems,          //
                               const ptrdiff_t    nelements,      //
                               elems_tet4_device* elems_device);  //

/**
 * @brief Free memory for the elements struct
 *
 * @param elems_device
 */
void                                                      //
free_elems_tet4_device(elems_tet4_device* elems_device);  //

void                                                           //
free_elems_tet4_device_async(elems_tet4_device* elems_device,  //
                             cudaStream_t       stream);             //

/**
 * @brief Free memory for the elements struct
 * @brief This function is used when the memory is allocated in the unified memory
 * It simply sets the pointers to NULL (since the memory is not managed by CUDA)
 *
 * @param elems_device
 */
void                                                              //
free_elems_tet4_device_unified(elems_tet4_device* elems_device);  //

// Struct for xyz
typedef struct {
    geom_t* x;
    geom_t* y;
    geom_t* z;
} xyz_tet4_device;

/**
 * @brief Allocate memory for the xyz struct
 *
 * @param xyz_device
 * @param nnodes
 */
void                                                        //
cuda_allocate_xyz_tet4_device(xyz_tet4_device* xyz_device,  //
                              const ptrdiff_t  nnodes);      //

void                                                              //
cuda_allocate_xyz_tet4_device_async(xyz_tet4_device* xyz_device,  //
                                    const ptrdiff_t  nnodes,      //
                                    cudaStream_t     stream);         //

/**
 * @brief Allocate managed memory for the xyz struct
 *
 * @param xyz_device
 * @param nnodes
 */
void                                                                //
cuda_allocate_xyz_tet4_device_managed(xyz_tet4_device* xyz_device,  //
                                      const ptrdiff_t  nnodes);      //

/**
 * @brief   Make the xyz struct
 *
 * @return xyz_tet4_device
 */
xyz_tet4_device          //
make_xyz_tet4_device();  //

/**
 * @brief Free memory for the xyz struct
 *
 * @param xyz_device
 */
void                                                //
free_xyz_tet4_device(xyz_tet4_device* xyz_device);  //

void                                                     //
free_xyz_tet4_device_async(xyz_tet4_device* xyz_device,  //
                           cudaStream_t     stream);         //

/**
 * @brief Free memory for the xyz struct
 * @brief This function is used when the memory is allocated in the unified memory
 * It simply sets the pointers to NULL (since the memory is not managed by CUDA)
 *
 * @param xyz_device
 */
void                                                        //
free_xyz_tet4_device_unified(xyz_tet4_device* xyz_device);  //

/**
 * @brief Copy the xyz struct from host to device
 *
 * @param xyz
 * @param nnodes
 * @param xyz_device
 */
void                                                //
copy_xyz_tet4_device(const geom_t**   xyz,          //
                     const ptrdiff_t  nnodes,       //
                     xyz_tet4_device* xyz_device);  //

void                                                     //
copy_xyz_tet4_device_async(const geom_t**   xyz,         //
                           const ptrdiff_t  nnodes,      //
                           xyz_tet4_device* xyz_device,  //
                           cudaStream_t     stream);         //

/**
 * @brief Copy the xyz struct from host to device
 *
 * @param xyz
 * @param nnodes
 * @param xyz_device
 */
void                                                        //
copy_xyz_tet4_device_unified(const geom_t**   xyz,          //
                             const ptrdiff_t  nnodes,       //
                             xyz_tet4_device* xyz_device);  //

#endif  // SFEM_RESAMPLE_FIELD_CUDA_FUN_CUH