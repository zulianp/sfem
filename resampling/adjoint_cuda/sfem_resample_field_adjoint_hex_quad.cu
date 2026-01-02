#include <cuda_runtime.h>
#include "device_utils.cuh"
#include "sfem_resample_field_adjoint_hex_wquad.cuh"

#ifdef __cplusplus
extern "C" {
#endif


/////////////////////////////////////////////////////////////////////////
// Function: call_tet4_resample_field_adjoint_hex_quad_kernel_gpu
/////////////////////////////////////////////////////////////////////////
void                                                                                       //
call_tet4_resample_field_adjoint_hex_quad_kernel_gpu(const ptrdiff_t      start_element,   // Mesh
                                                     const ptrdiff_t      end_element,     //
                                                     const ptrdiff_t      nelements,       //
                                                     const ptrdiff_t      nnodes,          //
                                                     const idx_t** const  elems,           //
                                                     const geom_t** const xyz,             //
                                                     const ptrdiff_t      n0,              // SDF
                                                     const ptrdiff_t      n1,              //
                                                     const ptrdiff_t      n2,              //
                                                     const ptrdiff_t      stride0,         // Stride
                                                     const ptrdiff_t      stride1,         //
                                                     const ptrdiff_t      stride2,         //
                                                     const geom_t         origin0,         // Origin
                                                     const geom_t         origin1,         //
                                                     const geom_t         origin2,         //
                                                     const geom_t         dx,              // Delta
                                                     const geom_t         dy,              //
                                                     const geom_t         dz,              //
                                                     const real_t* const  weighted_field,  // Input weighted field
                                                     real_t* const        data) {
    //
    PRINT_CURRENT_FUNCTION;

    cudaStream_t cuda_stream_alloc = NULL;  // default stream
    cudaStreamCreate(&cuda_stream_alloc);

    real_t*  data_device           = NULL;
    real_t*  weighted_field_device = NULL;
    real_t*  tet_volumes_device    = NULL;
    int32_t* in_out_mesh           = NULL;

    cudaMallocAsync((void**)&data_device, (n0 * n1 * n2) * sizeof(real_t), cuda_stream_alloc);
    cudaMallocAsync((void**)&weighted_field_device, nnodes * sizeof(real_t), cuda_stream_alloc);
    cudaMallocAsync((void**)&tet_volumes_device, nelements * sizeof(real_t), cuda_stream_alloc);
    cudaMallocAsync((void**)&in_out_mesh, (n0 * n1 * n2) * sizeof(int32_t), cuda_stream_alloc);

    elems_tet4_device elements_device = make_elems_tet4_device();
    cuda_allocate_elems_tet4_device_async(&elements_device, nelements, cuda_stream_alloc);

    xyz_tet4_device xyz_device = make_xyz_tet4_device();
    cuda_allocate_xyz_tet4_device_async(&xyz_device, nnodes, cuda_stream_alloc);

    cudaStreamSynchronize(cuda_stream_alloc);  /// Ensure allocations are done before proceeding further with copies

    cudaMemcpy((void*)weighted_field_device, (void*)weighted_field, nnodes * sizeof(real_t), cudaMemcpyHostToDevice);

    cudaMemset((void*)data_device, 0, (n0 * n1 * n2) * sizeof(real_t));
    cudaMemset((void*)in_out_mesh, 0, (n0 * n1 * n2) * sizeof(int32_t));

    copy_elems_tet4_device_async(elems, nelements, &elements_device, cuda_stream_alloc);

    copy_xyz_tet4_device_async(xyz, nnodes, &xyz_device, cuda_stream_alloc);

    cudaStreamSynchronize(cuda_stream_alloc);

#if SFEM_LOG_LEVEL >= 5
    char* devive_desc = acc_get_device_properties(0);  // Just to ensure the device is set correctly
    printf("%s\n", devive_desc);
#endif

    const real_t volume_tet_grid =                                        //
            compute_total_tet_volume_gpu<real_t,                          //
                                         ptrdiff_t>(nelements,            //
                                                    elements_device,      //
                                                    xyz_device,           //
                                                    tet_volumes_device);  //

    const real_t volume_hex_grid      = dx * dy * dz * ((real_t)(n0 * n1 * n2));
    const int    num_hex              = n0 * n1 * n2;
    const real_t tet_hex_volume_ratio = volume_tet_grid / volume_hex_grid;

#if SFEM_LOG_LEVEL >= 5
    printf("Total volume (tet_grid_volumes): %e \n", (double)volume_tet_grid);
    printf("Total volume (hex_grid):         %e \n", (double)volume_hex_grid);
#endif  // END if (SFEM_LOG_LEVEL >= 5)

    // Optional: check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s, at file:%s:%d \n", cudaGetErrorString(error), __FILE__, __LINE__);
    }  // END if (error != cudaSuccess)

    // Launch kernel
    const unsigned int blocks_per_grid   = min((unsigned int)(end_element - start_element + 1), getSMCount() * 15000);
    const unsigned int threads_per_block = 256;

#if SFEM_LOG_LEVEL >= 5
    printf("Launching tet_grid_hex_boudary_kernel_gpu with: \n");
    printf("* blocks_per_grid:         %u \n", blocks_per_grid);
    printf("* threads_per_block:       %u \n", threads_per_block);
#endif

    cudaStream_t cuda_stream = NULL;  // default stream
    cudaStreamCreate(&cuda_stream);

    // Create timing events for boundary kernel
    cudaEvent_t boundary_start_event, boundary_stop_event;
    cudaEventCreate(&boundary_start_event);
    cudaEventCreate(&boundary_stop_event);
    cudaEventRecord(boundary_start_event, cuda_stream);  // Record on the SAME stream

    tet_grid_hex_indicator_IO_kernel_gpu<float_t, int><<<blocks_per_grid,    //
                                                         threads_per_block,  //
                                                         0,                  //
                                                         cuda_stream>>>(0,
                                                                        nelements,  //
                                                                        elements_device,
                                                                        xyz_device,    //
                                                                        n0,            //
                                                                        n1,            //
                                                                        n2,            //
                                                                        stride0,       //
                                                                        stride1,       //
                                                                        stride2,       //
                                                                        origin0,       //
                                                                        origin1,       //
                                                                        origin2,       //
                                                                        dx,            //
                                                                        dy,            //
                                                                        dz,            //
                                                                        in_out_mesh);  //

    cudaEventRecord(boundary_stop_event, cuda_stream);  // Record on the SAME stream
    cudaEventSynchronize(boundary_stop_event);          // Wait for completion

    // Optional: check for errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s, at file:%s:%d \n", cudaGetErrorString(error), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }  // END if (error != cudaSuccess)

    float boundary_milliseconds = 0.0f;
    cudaEventElapsedTime(&boundary_milliseconds, boundary_start_event, boundary_stop_event);
    printf("Boundary kernel elapsed time: %e ms\n", boundary_milliseconds);
    cudaEventDestroy(boundary_start_event);
    cudaEventDestroy(boundary_stop_event);

    // if (true) {
    //     // Debug: copy back the in_out_mesh
    //     int32_t* in_out_mesh_host = (int32_t*)malloc((n0 * n1 * n2) * sizeof(int32_t));
    //     cudaMemcpy((void*)in_out_mesh_host, (const void*)in_out_mesh, (n0 * n1 * n2) * sizeof(int32_t),
    //     cudaMemcpyDeviceToHost); size_t count_in  = 0; size_t count_out = 0; for (ptrdiff_t i = 0; i < (n0 * n1 * n2); i++) {
    //         if (in_out_mesh_host[i] == 1) count_in++;
    //         if (in_out_mesh_host[i] == 0) count_out++;
    //     }
    //     printf("In out mesh counts: in=%lu , out=%lu, sum=%lu \n", count_in, count_out, count_in + count_out);
    //     free(in_out_mesh_host);
    // }

#if SFEM_LOG_LEVEL >= 5
    printf("Launching tet4_resample_field_adjoint_hex_quad_kernel_gpu with: \n");
    printf("* blocks_per_grid:         %u \n", blocks_per_grid);
    printf("* threads_per_block:       %u \n", threads_per_block);
    printf("* LANES_PER_TILE_HEX_QUAD: %u \n", (unsigned int)LANES_PER_TILE_HEX_QUAD);
#endif

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event, cuda_stream);

    ////////////////////////////////////////////////////////////////////
    /// Launch the kernel //////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////
    tet4_resample_field_adjoint_hex_quad_v2_kernel_gpu<real_t, int><<<blocks_per_grid,                       //
                                                                      threads_per_block,                     //
                                                                      0,                                     //
                                                                      cuda_stream>>>(start_element,          //
                                                                                     end_element,            //
                                                                                     nelements,              //
                                                                                     elements_device,        //
                                                                                     xyz_device,             //
                                                                                     n0,                     //
                                                                                     n1,                     //
                                                                                     n2,                     //
                                                                                     stride0,                //
                                                                                     stride1,                //
                                                                                     stride2,                //
                                                                                     origin0,                //
                                                                                     origin1,                //
                                                                                     origin2,                //
                                                                                     dx,                     //
                                                                                     dy,                     //
                                                                                     dz,                     //
                                                                                     weighted_field_device,  //
                                                                                     data_device);           //

    cudaStreamSynchronize(cuda_stream);

    // Optional: check for errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s, at file:%s:%d \n", cudaGetErrorString(error), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }  // END if (error != cudaSuccess)

    cudaEventRecord(stop_event, cuda_stream);
    cudaEventSynchronize(stop_event);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);

    if (SFEM_LOG_LEVEL >= 5) {
        printf("================= SFEM Adjoint Hex Quad Resampling GPU =================\n");
        printf("* Kernel execution time:    %f ms\n", milliseconds);
        printf("*   Tet per second:         %e \n", (float)(end_element - start_element) / (milliseconds * 1.0e-3));
        printf("*   Hex nodes per second:   %e (approx)\n",
               (float)(n0 * n1 * n2) * (tet_hex_volume_ratio) / (milliseconds * 1.0e-3));
        printf("*   Tet Nodes per second:   %e (approx)\n", (float)(nnodes) / (milliseconds * 1.0e-3));
        printf("*   Number of elements:      %d \n", (int)(end_element - start_element));
        printf("*   Number of nodes:         %d \n", (int)(nnodes));
        printf(" -----------------------------------------------------------------------\n");
        printf("<quad_bench_head> nelements, time(s), tet/s, hex_nodes/s, tet_nodes/s, nnodes, n0, n1, n2, dx, dy, dz, origin0, "
               "origin1, "
               "origin2, volume_tet_grid \n");
        printf("<quad_bench> %d , %e, %e, %e , %e, %d, %d , %d , %d , %e , %e , %e, %e , %e , %e, %e \n",
               (end_element - start_element),
               (milliseconds * 1.0e-3),
               (double)(end_element - start_element) / (milliseconds * 1.0e-3),
               (double)(n0 * n1 * n2) * (tet_hex_volume_ratio) / (milliseconds * 1.0e-3),
               (double)(nnodes) / (milliseconds * 1.0e-3),
               nnodes,
               n0,
               n1,
               n2,
               (double)dx,
               (double)dy,
               (double)dz,
               (double)origin0,
               (double)origin1,
               (double)origin2,
               (double)volume_tet_grid);
        printf("*   function: %s, in file: %s:%d \n", __FUNCTION__, __FILE__, __LINE__);
        printf("=========================================================================\n");
    }  // END if (SFEM_LOG_LEVEL >= 5)

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    cudaStreamDestroy(cuda_stream);

    cudaMemcpy((void*)data, (const void*)data_device, (n0 * n1 * n2) * sizeof(real_t), cudaMemcpyDeviceToHost);

    // Cleanup (was unreachable due to early return)
    cudaFreeAsync((void*)weighted_field_device, cuda_stream_alloc);
    cudaFreeAsync((void*)in_out_mesh, cuda_stream_alloc);
    free_xyz_tet4_device_async(&xyz_device, cuda_stream_alloc);
    free_elems_tet4_device_async(&elements_device, cuda_stream_alloc);
    cudaStreamSynchronize(cuda_stream_alloc);
    cudaFreeAsync(data_device, cuda_stream_alloc);
    cudaFreeAsync(tet_volumes_device, cuda_stream_alloc);
    cudaStreamSynchronize(cuda_stream_alloc);
    cudaStreamDestroy(cuda_stream_alloc);

    RETURN_FROM_FUNCTION();
}  // END Function: call_tet4_resample_field_adjoint_hex_quad_kernel_gpu

#ifdef __cplusplus
}  // extern "C"
#endif