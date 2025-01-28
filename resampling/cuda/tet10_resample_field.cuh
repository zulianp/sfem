#ifndef __TET10_RESAMPLE_FIELD_CUH__
#define __TET10_RESAMPLE_FIELD_CUH__

#include <cuda_runtime.h>
#include <sfem_base.h>

/////////////////////////////////////////////////////////////////
// Struct for xyz
/////////////////////////////////////////////////////////////////
typedef struct {
    float* x = NULL;
    float* y = NULL;
    float* z = NULL;
} xyz_tet10_device;
// end struct xyz_tet10_device //////////////////////////////////
/////////////////////////////////////////////////////////////////

// Function declarations

// xyz_tet10_device functions
/**
 * @brief
 *
 * @param nnodes
 * @return xyz_tet10_device
 */
xyz_tet10_device                                //
make_xyz_tet10_device(const ptrdiff_t nnodes);  //

/**
 * @brief
 *
 * @param nnodes
 * @param xyz
 * @param xyz_host
 */
void                                                                                           //
copy_xyz_tet10_device(const ptrdiff_t nnodes, xyz_tet10_device* xyz, const float** xyz_host);  //

/**
 * @brief
 *
 * @param xyz
 */
void                                           //
free_xyz_tet10_device(xyz_tet10_device* xyz);  //

// elems_tet10_device functions for unified memory

/**
 * @brief
 *
 * @param nnodes
 * @return xyz_tet10_device
 */
xyz_tet10_device                                        //
make_xyz_tet10_device_unified(const ptrdiff_t nnodes);  //

/**
 * @brief
 *
 * @param nnodes
 * @param xyz
 * @param xyz_host
 */
void                                                     //
copy_xyz_tet10_device_unified(const ptrdiff_t   nnodes,  //
                              xyz_tet10_device* xyz,     //
                              const float**     xyz_host);   //

/**
 * @brief
 *
 * @param xyz
 */
void                                                   //
free_xyz_tet10_device_unified(xyz_tet10_device* xyz);  //

/**
 * @brief
 *
 * @param xyz
 */
void                                                            //
memory_hint_xyz_tet10_device_unified(const ptrdiff_t   nnodes,  //
                                     xyz_tet10_device* xyz);    //

/////////////////////////////////////////////////////////////////
// elems_tet10_device functions
/////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////
// Struct for elems
/////////////////////////////////////////////////////////////////
typedef struct {
    int* elems_v0 = NULL;
    int* elems_v1 = NULL;
    int* elems_v2 = NULL;
    int* elems_v3 = NULL;
    int* elems_v4 = NULL;
    int* elems_v5 = NULL;
    int* elems_v6 = NULL;
    int* elems_v7 = NULL;
    int* elems_v8 = NULL;
    int* elems_v9 = NULL;
} elems_tet10_device;
// end struct elems_tet10_device
/////////////////////////////////////////////////////////////////

/**
 * @brief
 *
 * @param nelements
 * @return elems_tet10_device
 */
elems_tet10_device                                   //
make_elems_tet10_device(const ptrdiff_t nelements);  //

/**
 * @brief
 *
 * @param nelements
 * @param elems
 * @param elems_host
 * @return cudaError_t
 */
cudaError_t                                             //
copy_elems_tet10_device(const ptrdiff_t     nelements,  //
                        elems_tet10_device* elems,      //
                        const idx_t**       elems_host);      //

/**
 * @brief
 *
 * @param elems
 */
void                                                 //
free_elems_tet10_device(elems_tet10_device* elems);  //

/**
 * @brief
 *
 * @param nelements
 * @return elems_tet10_device
 */
elems_tet10_device                                           //
make_elems_tet10_device_unified(const ptrdiff_t nelements);  //

/**
 * @brief
 *
 * @param nelements
 * @param elems
 * @param elems_host
 * @return cudaError_t
 */
cudaError_t                                                     //
copy_elems_tet10_device_unified(const ptrdiff_t     nelements,  //
                                elems_tet10_device* elems,      //
                                const idx_t**       elems_host);      //

/**
 * @brief
 *
 * @param elems
 */
void                                                         //
free_elems_tet10_device_unified(elems_tet10_device* elems);  //

/**
 * @brief
 *
 * @param nelements
 * @param elems
 */
void                                                                                     //
memory_hint_elems_tet10_device_unified(ptrdiff_t nelements, elems_tet10_device* elems);  //

/**
 * @brief
 *
 * @param array_size
 * @param element_size
 * @param ptr
 */
void                                                                                           //
memory_hint_read_mostly(const ptrdiff_t array_size, const ptrdiff_t element_size, void* ptr);  //

/**
 * @brief
 *
 * @param array_size
 * @param element_size
 * @param ptr
 */
void                                                                                            //
memory_hint_write_mostly(const ptrdiff_t array_size, const ptrdiff_t element_size, void* ptr);  //

#endif  // __TET10_RESAMPLE_FIELD_CUH__