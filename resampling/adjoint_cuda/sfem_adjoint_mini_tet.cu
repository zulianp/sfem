#include <stdio.h>

#include "sfem_adjoint_mini_tet.cuh"

extern "C" void                                                                         //
call_sfem_adjoint_mini_tet_kernel_gpu(const ptrdiff_t             start_element,        // Mesh
                                      const ptrdiff_t             end_element,          //
                                      const ptrdiff_t             nnodes,               //
                                      const idx_t** const         elems,                //
                                      const geom_t** const        xyz,                  //
                                      const ptrdiff_t             n0,                   // SDF
                                      const ptrdiff_t             n1,                   //
                                      const ptrdiff_t             n2,                   //
                                      const ptrdiff_t             stride0,              // Stride
                                      const ptrdiff_t             stride1,              //
                                      const ptrdiff_t             stride2,              //
                                      const geom_t                origin0,              // Origin
                                      const geom_t                origin1,              //
                                      const geom_t                origin2,              //
                                      const geom_t                dx,                   // Delta
                                      const geom_t                dy,                   //
                                      const geom_t                dz,                   //
                                      const real_t* const         weighted_field,       // Input weighted field
                                      const mini_tet_parameters_t mini_tet_parameters,  // Threshold for alpha
                                      real_t* const               data) {
    //

    const unsigned int threads_per_block      = 256;
    const unsigned int total_threads_per_grid = (end_element - start_element) * LANES_PER_TILE;
    const unsigned int blocks_per_grid        = (total_threads_per_grid + threads_per_block - 1) / threads_per_block;

    cudaStream_t cuda_stream = 0;  // default stream
    cudaStreamCreate(&cuda_stream);

    sfem_adjoint_mini_tet_kernel_gpu<real_t><<<blocks_per_grid,                     //
                                               threads_per_block,                   //
                                               0,                                   //
                                               cuda_stream>>>(start_element,        // Mesh
                                                              end_element,          //
                                                              nnodes,               //
                                                              elems,                //
                                                              xyz,                  //
                                                              n0,                   // SDF
                                                              n1,                   //
                                                              n2,                   //
                                                              stride0,              // Stride
                                                              stride1,              //
                                                              stride2,              //
                                                              origin0,              // Origin
                                                              origin1,              //
                                                              origin2,              //
                                                              dx,                   // Delta
                                                              dy,                   //
                                                              dz,                   //
                                                              weighted_field,       // Input weighted field
                                                              mini_tet_parameters,  // Threshold for alpha
                                                              data);                //

    cudaStreamSynchronize(cuda_stream);

    // Optional: check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s, at file:%s:%d \n", cudaGetErrorString(error), __FILE__, __LINE__);
    }

    cudaStreamDestroy(cuda_stream);

}  // END: call_sfem_adjoint_mini_tet_kernel_gpu
   // ////////////////////////////////////////////////////////////////////////////////////////////////

// // #define __TESTING__
// #ifdef __TESTING__

// int main() {
//     mini_tet_parameters_t mini_tet_parameters;

//     mini_tet_parameters.alpha_min_threshold = 2.0;
//     mini_tet_parameters.alpha_max_threshold = 8.0;
//     mini_tet_parameters.min_refinement_L    = 1;
//     mini_tet_parameters.max_refinement_L    = 20;

//     // const int L = 4;  // Example refinement level
//     sfem_adjoint_mini_tet_kernel_gpu<double><<<1, 1>>>(0,        //
//                                                        1,        //
//                                                        1,        //
//                                                        nullptr,  //
//                                                        nullptr,
//                                                        1,        //
//                                                        1,        //
//                                                        1,        //
//                                                        1,        //
//                                                        1,        //
//                                                        1,        //
//                                                        0.0,      //
//                                                        0.0,      //
//                                                        0.0,      //
//                                                        1.0,      //
//                                                        1.0,      //
//                                                        1.0,      //
//                                                        nullptr,  //
//                                                        mini_tet_parameters,
//                                                        nullptr);  //
//     cudaDeviceSynchronize();
//     return 0;
// }
