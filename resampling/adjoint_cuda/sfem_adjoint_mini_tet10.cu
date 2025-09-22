#include <stdio.h>
#include <stdlib.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include "sfem_adjoint_mini_loc_tet.cuh"
#include "sfem_adjoint_mini_tet.cuh"
#include "sfem_adjoint_mini_tet10.cuh"
#include "sfem_adjoint_mini_tet_fun.cuh"
#include "sfem_resample_field_cuda_fun.cuh"

void  //                                                                                               //
call_hex8_to_isoparametric_tet10_resample_field_hyteg_mt_adjoint_kernel(  //
        const ptrdiff_t      start_element,                               // Mesh
        const ptrdiff_t      end_element,                                 //
        const ptrdiff_t      nelements,                                   //
        const ptrdiff_t      nnodes,                                      //
        const idx_t** const  elems,                                       //
        const geom_t** const xyz,                                         //
        const ptrdiff_t      n0,                                          // SDF
        const ptrdiff_t      n1,                                          //
        const ptrdiff_t      n2,                                          //
        const ptrdiff_t      stride0,                                     //
        const ptrdiff_t      stride1,                                     //
        const ptrdiff_t      stride2,                                     //
        const geom_t         ox,                                          //
        const geom_t         oy,                                          //
        const geom_t         oz,                                          //
        const geom_t         dx,                                          //
        const geom_t         dy,                                          //
        const geom_t         dz,                                          //
        const real_t* const __restrict__ weighted_field,                  // Input WF
        real_t* const __restrict__ data,                                  // Output
        const mini_tet_parameters_t mini_tet_parameters) {                //

    cudaStream_t cuda_stream_alloc = NULL;  // default stream
    cudaStreamCreate(&cuda_stream_alloc);

    real_t* data_device           = NULL;
    real_t* weighted_field_device = NULL;

    cudaMallocAsync((void**)&data_device, (n0 * n1 * n2) * sizeof(real_t), cuda_stream_alloc);
    cudaMallocAsync((void**)&weighted_field_device, nnodes * sizeof(real_t), cuda_stream_alloc);

    elems_tet10_device elems_d = make_elems_tet10_device_async(nelements, cuda_stream_alloc);
    xyz_tet10_device   xyz_d   = make_xyz_tet10_device_async(nnodes, cuda_stream_alloc);

    cudaStreamSynchronize(cuda_stream_alloc);

    cudaMemcpyAsync(weighted_field_device,    //
                    weighted_field,           //
                    nnodes * sizeof(real_t),  //
                    cudaMemcpyHostToDevice,   //
                    cuda_stream_alloc);       //

    cudaMemset((void*)data_device, 0, (n0 * n1 * n2) * sizeof(real_t));

    copy_elems_tet10_device_async(nelements,           //
                                  &elems_d,            //
                                  elems,               //
                                  cuda_stream_alloc);  //

    copy_xyz_tet10_device_async(nnodes,              //
                                &xyz_d,              //
                                xyz,                 //
                                cuda_stream_alloc);  //

    cudaStreamSynchronize(cuda_stream_alloc);

    // Optional: check for errors
    {  // Begin: Error check block
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s, at file:%s:%d \n", cudaGetErrorString(error), __FILE__, __LINE__);
        }
    }  // End: Error check block

    const unsigned int threads_per_block      = 256;
    const unsigned int total_threads_per_grid = (end_element - start_element + 1) * LANES_PER_TILE;
    const unsigned int blocks_per_grid        = (total_threads_per_grid + threads_per_block - 1) / threads_per_block;

#if SFEM_LOG_LEVEL >= 5
    printf("Kernel args: start_element: %ld, end_element: %ld, nelements: %ld, nnodes: %ld\n",
           start_element,
           end_element,
           nelements,
           nnodes);
    printf("Kernel launch: blocks_per_grid: %u, threads_per_block: %u, total_threads_per_grid: %u\n",
           blocks_per_grid,
           threads_per_block,
           total_threads_per_grid);
#endif

    cudaStream_t cuda_stream = 0;  // default stream
    cudaStreamCreate(&cuda_stream);

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event, cuda_stream);

    ///////////////////////////////////
    // Launch kernel
    ///////////////////////////////////

    hex8_to_isoparametric_tet10_resample_field_hyteg_mt_adjoint_kernel  //
            <<<blocks_per_grid,                                         //
               threads_per_block,                                       //
               0,                                                       //
               cuda_stream>>>(                                          //
                    start_element,                                      //
                    end_element,                                        //
                    nelements,                                          //
                    elems_d,                                            //
                    xyz_d,                                              //
                    n0,                                                 //
                    n1,                                                 //
                    n2,                                                 //
                    stride0,                                            //
                    stride1,                                            //
                    stride2,                                            //
                    ox,                                                 //
                    oy,                                                 //
                    oz,                                                 //
                    dx,                                                 //
                    dy,                                                 //
                    dz,                                                 //
                    weighted_field_device,                              //
                    data_device,                                        //
                    mini_tet_parameters);                               //

    cudaStreamSynchronize(cuda_stream);

    // Optional: check for errors
    {  // Begin: Error check block
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s, at file:%s:%d \n", cudaGetErrorString(error), __FILE__, __LINE__);
        }
    }  // End: Error check block

    cudaEventRecord(stop_event, cuda_stream);
    cudaEventSynchronize(stop_event);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);

    if (SFEM_LOG_LEVEL >= 5) {
        printf("================= SFEM Adjoint Mini-Tet TET10 Kernel GPU ================\n");
        printf("Kernel execution time: %f ms\n", milliseconds);
        printf("  Tet per second: %e \n", (float)(end_element - start_element) / (milliseconds * 1.0e-3));
        printf("===================================================================\n");
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    cudaStreamDestroy(cuda_stream);

    cudaMemcpy((void*)data, (void*)data_device, (n0 * n1 * n2) * sizeof(real_t), cudaMemcpyDeviceToHost);

    cudaFreeAsync((void*)weighted_field_device, cuda_stream_alloc);

    free_xyz_tet10_device_async(&xyz_d, cuda_stream_alloc);

    free_elems_tet10_async(&elems_d, cuda_stream_alloc);

    cudaFreeAsync((void*)data_device, cuda_stream_alloc);
    cudaFreeAsync((void*)weighted_field_device, cuda_stream_alloc);

    cudaStreamSynchronize(cuda_stream_alloc);
    cudaStreamDestroy(cuda_stream_alloc);
}
