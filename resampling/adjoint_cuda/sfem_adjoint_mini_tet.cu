#include <stdio.h>

#include "sfem_adjoint_mini_tet.cuh"
#include "sfem_resample_field_cuda_fun.cuh"

extern "C" void                                                                         //
call_sfem_adjoint_mini_tet_kernel_gpu(const ptrdiff_t             start_element,        // Mesh
                                      const ptrdiff_t             end_element,          //
                                      const ptrdiff_t             nelements,            //
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

    cudaStream_t cuda_stream_alloc = 0;  // default stream
    cudaStreamCreate(&cuda_stream_alloc);

    real_t* data_device = NULL;
    cudaMallocAsync((void**)&data_device, (n0 * n1 * n2) * sizeof(real_t), cuda_stream_alloc);

    elems_tet4_device elements_device = make_elems_tet4_device();
    cuda_allocate_elems_tet4_device_async(&elements_device, nelements, cuda_stream_alloc);

    xyz_tet4_device xyz_device = make_xyz_tet4_device();
    cuda_allocate_xyz_tet4_device_async(&xyz_device, nnodes, cuda_stream_alloc);

    cudaStreamSynchronize(cuda_stream_alloc);

    cudaMemsetAsync((void*)data_device, 0, (n0 * n1 * n2) * sizeof(real_t), cuda_stream_alloc);

    copy_elems_tet4_device_async(elems, nelements, &elements_device, cuda_stream_alloc);
    copy_xyz_tet4_device_async(xyz, nnodes, &xyz_device, cuda_stream_alloc);

    cudaStreamSynchronize(cuda_stream_alloc);

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
                                                              elements_device,      //
                                                              xyz_device,           //
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

    cudaMemcpyAsync((void*)data, (void*)data_device, (n0 * n1 * n2) * sizeof(real_t), cudaMemcpyDeviceToHost, cuda_stream_alloc);

    free_xyz_tet4_device_async(&xyz_device, cuda_stream_alloc);
    free_elems_tet4_device_async(&elements_device, cuda_stream_alloc);

    cudaStreamSynchronize(cuda_stream_alloc);

    cudaFreeAsync(data_device, cuda_stream_alloc);

    cudaStreamDestroy(cuda_stream_alloc);

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
