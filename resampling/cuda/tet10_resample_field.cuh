#ifndef __TET10_RESAMPLE_FIELD_CUH__
#define __TET10_RESAMPLE_FIELD_CUH__

#include <cuda_runtime.h>
#include <sfem_base.h>

/////////////////////////////////////////////////////////////////
// Struct for xyz
/////////////////////////////////////////////////////////////////
typedef struct {
    geom_t* x = NULL;
    geom_t* y = NULL;
    geom_t* z = NULL;
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
 * @param stream
 * @return xyz_tet10_device
 */
xyz_tet10_device                                                           //
make_xyz_tet10_device_async(const ptrdiff_t nnodes, cudaStream_t stream);  //

/**
 * @brief
 *
 * @param nnodes
 * @param xyz
 * @param xyz_host
 */
void                                             //
copy_xyz_tet10_device(const ptrdiff_t   nnodes,  //
                      xyz_tet10_device* xyz,     //
                      const float**     xyz_host);   //

/**
 * @brief
 *
 * @param nnodes
 * @param xyz
 * @param xyz_host
 * @param stream
 */
void                                                     //
copy_xyz_tet10_device_async(const ptrdiff_t   nnodes,    //
                            xyz_tet10_device* xyz,       //
                            const float**     xyz_host,  //
                            cudaStream_t      stream);        //

/**
 * @brief
 *
 */
void                                                     //
copy_xyz_tet10_device_async(const ptrdiff_t   nnodes,    //
                            xyz_tet10_device* xyz,       //
                            const geom_t**    xyz_host,  //
                            cudaStream_t      stream);        //

/**
 * @brief
 *
 * @param xyz
 */
void                                           //
free_xyz_tet10_device(xyz_tet10_device* xyz);  //

/**
 * @brief
 *
 * @param xyz
 * @param stream
 */
void free_xyz_tet10_device_async(xyz_tet10_device* xyz, cudaStream_t stream);

/////////////////////////////////////////////////////////////////
// xyz_tet10_device functions for managed memory
/////////////////////////////////////////////////////////////////

/**
 * @brief
 *
 * @param nnodes
 * @return xyz_tet10_device
 */
xyz_tet10_device                                 //
make_xyz_tet10_managed(const ptrdiff_t nnodes);  //

/**
 * @brief
 *
 * @param nnodes
 * @param xyz
 * @param xyz_host
 */
void                                              //
copy_xyz_tet10_managed(const ptrdiff_t   nnodes,  //
                       xyz_tet10_device* xyz,     //
                       const float**     xyz_host);   //

/**
 * @brief
 *
 * @param xyz
 */
void                                            //
free_xyz_tet10_managed(xyz_tet10_device* xyz);  //

/**
 * @brief
 *
 * @param nnodes
 * @param xyz
 */
void                                                     //
memory_hint_xyz_tet10_managed(const ptrdiff_t   nnodes,  //
                              xyz_tet10_device* xyz);    //

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
    idx_t* elems_v0 = NULL;
    idx_t* elems_v1 = NULL;
    idx_t* elems_v2 = NULL;
    idx_t* elems_v3 = NULL;
    idx_t* elems_v4 = NULL;
    idx_t* elems_v5 = NULL;
    idx_t* elems_v6 = NULL;
    idx_t* elems_v7 = NULL;
    idx_t* elems_v8 = NULL;
    idx_t* elems_v9 = NULL;
} elems_tet10_device;
// end struct elems_tet10_device
/////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////
// Device memory
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
 * @param stream
 * @return elems_tet10_device
 */
elems_tet10_device                                                              //
make_elems_tet10_device_async(const ptrdiff_t nelements, cudaStream_t stream);  //

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
 * @param nelements
 * @param elems
 * @param elems_host
 * @param stream
 * @return cudaError_t
 */
cudaError_t                                                    //
copy_elems_tet10_device_async(const ptrdiff_t     nelements,   //
                              elems_tet10_device* elems,       //
                              const idx_t**       elems_host,  //
                              cudaStream_t        stream);            //

/**
 * @brief
 *
 * @param elems
 */
void                                                 //
free_elems_tet10_device(elems_tet10_device* elems);  //

/////////////////////////////////////////////////////////////////
// Manged memory
/////////////////////////////////////////////////////////////////

/**
 * @brief
 *
 * @param nelements
 * @return elems_tet10_device
 */
elems_tet10_device                                    //
make_elems_tet10_managed(const ptrdiff_t nelements);  //

/**
 * @brief
 *
 * @param nelements
 * @param elems
 * @param elems_host
 * @return cudaError_t
 */
cudaError_t                                              //
copy_elems_tet10_managed(const ptrdiff_t     nelements,  //
                         elems_tet10_device* elems,      //
                         const idx_t**       elems_host);      //

/**
 * @brief
 *
 * @param elems
 */
void                                                  //
free_elems_tet10_managed(elems_tet10_device* elems);  //

/**
 * @brief
 *
 * @param elems
 * @param stream
 */
void                                                                     //
free_elems_tet10_async(elems_tet10_device* elems, cudaStream_t stream);  //

/////////////////////////////////////////////////////////////////
// Unified memory
/////////////////////////////////////////////////////////////////

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

/////////////////////////////////////////////////////////////////
// Memory hints
/////////////////////////////////////////////////////////////////

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

/**
 * @brief Apply the inverse of the lumped mass matrix to a field.
 * Referencing the function in the original code: mass.c
 */
extern "C" void                                                   //
apply_inv_lumped_mass(const int                    element_type,  //
                      const ptrdiff_t              nelements,     //
                      const ptrdiff_t              nnodes,        //
                      idx_t** const SFEM_RESTRICT  elems,         //
                      geom_t** const SFEM_RESTRICT xyz,           //
                      const real_t* const          x,             //
                      real_t* const                values);                      //

#endif  // __TET10_RESAMPLE_FIELD_CUH__