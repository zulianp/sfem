#include "sfem_adjoint_mini_tet_gpu_wrapper.h"
#include "sfem_resample_field_adjoint_hyteg.h"

int                                                                                                      //
tet4_resample_field_local_refine_adjoint_hyteg_gpu(const ptrdiff_t                      start_element,   // Mesh
                                                   const ptrdiff_t                      end_element,     //
                                                   const ptrdiff_t                      nnodes,          //
                                                   const idx_t** const SFEM_RESTRICT    elems,           //
                                                   const geom_t** const SFEM_RESTRICT   xyz,             //
                                                   const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
                                                   const ptrdiff_t* const SFEM_RESTRICT stride,          //
                                                   const geom_t* const SFEM_RESTRICT    origin,          //
                                                   const geom_t* const SFEM_RESTRICT    delta,           //
                                                   const real_t* const SFEM_RESTRICT    weighted_field,  // Input weighted field
                                                   const mini_tet_parameters_t          mini_tet_parameters,
                                                   real_t* const SFEM_RESTRICT          data) {  //

    PRINT_CURRENT_FUNCTION;

    // printf("Strides = %ld %ld %ld ************************************************ \n", stride[0], stride[1], stride[2]);

#define TEST_KERNEL_MODEL 3

#if TEST_KERNEL_MODEL == 0
    call_sfem_adjoint_mini_tet_kernel_gpu(
#elif TEST_KERNEL_MODEL == 1
    call_sfem_adjoint_mini_tet_shared_info_kernel_gpu(
#elif TEST_KERNEL_MODEL == 2
    call_sfem_adjoint_mini_tet_cluster_kernel_gpu(
#elif TEST_KERNEL_MODEL == 3
    call_sfem_adjoint_mini_tet_buffer_cluster_info_kernel_gpu(
#endif
            start_element,                  // Mesh
            end_element,                    //
            (end_element - start_element),  // nelements
            nnodes,                         //
            elems,                          //
            xyz,                            //
            n[0],                           // SDF
            n[1],                           //
            n[2],                           //
            stride[0],                      // Stride
            stride[1],                      //
            stride[2],                      //
            origin[0],                      // Origin
            origin[1],                      //
            origin[2],                      //
            delta[0],                       // Delta
            delta[1],                       //
            delta[2],                       //
            weighted_field,                 // Input weighted field
            mini_tet_parameters,            // Threshold for alpha
            data);                          //

    RETURN_FROM_FUNCTION(EXIT_SUCCESS);
}

int                                                                                                                   //
hex8_to_isoparametric_tet10_resample_field_hyteg_mt_adjoint_gpu(const ptrdiff_t                      start_element,   // Mesh
                                                                const ptrdiff_t                      end_element,     //
                                                                const ptrdiff_t                      nnodes,          //
                                                                const idx_t** const SFEM_RESTRICT    elems,           //
                                                                const geom_t** const SFEM_RESTRICT   xyz,             //
                                                                const ptrdiff_t* const SFEM_RESTRICT n,               // SDF
                                                                const ptrdiff_t* const SFEM_RESTRICT stride,          //
                                                                const geom_t* const SFEM_RESTRICT    origin,          //
                                                                const geom_t* const SFEM_RESTRICT    delta,           //
                                                                const real_t* const SFEM_RESTRICT    weighted_field,  // Input WF
                                                                real_t* const SFEM_RESTRICT          data,            //
                                                                const mini_tet_parameters_t          mini_tet_parameters) {    //

    PRINT_CURRENT_FUNCTION;

    call_hex8_to_isoparametric_tet10_resample_field_hyteg_mt_adjoint_kernel(start_element,                  // Mesh
                                                                            end_element,                    //
                                                                            (end_element - start_element),  // nelements
                                                                            nnodes,                         //
                                                                            elems,                          //
                                                                            xyz,                            //
                                                                            n[0],                           // SDF
                                                                            n[1],                           //
                                                                            n[2],                           //
                                                                            stride[0],                      // Stride
                                                                            stride[1],                      //
                                                                            stride[2],                      //
                                                                            origin[0],                      // Origin
                                                                            origin[1],                      //
                                                                            origin[2],                      //
                                                                            delta[0],                       // Delta
                                                                            delta[1],                       //
                                                                            delta[2],                       //
                                                                            weighted_field,        // Input weighted field
                                                                            data,                  //
                                                                            mini_tet_parameters);  //

    RETURN_FROM_FUNCTION(EXIT_SUCCESS);
}