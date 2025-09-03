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
                                                   const real_t                         alpha_th,        // Threshold for alpha
                                                   real_t* const SFEM_RESTRICT          data) {                   //

    PRINT_CURRENT_FUNCTION;

    mini_tet_parameters_t mini_tet_parameters;

    mini_tet_parameters.alpha_min_threshold = alpha_th;
    mini_tet_parameters.alpha_max_threshold = 8.0;
    mini_tet_parameters.min_refinement_L    = 1;
    mini_tet_parameters.max_refinement_L    = 15;

    call_sfem_adjoint_mini_tet_kernel_gpu(  //
            start_element,                  // Mesh
            end_element,                    //
            end_element - start_element,    // nelements
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

    return EXIT_SUCCESS;
}