#include <cuda_runtime.h>
#include "sfem_resample_field_adjoint_hex_quad.cuh"

char* acc_get_device_properties(const int device_id) {
    cudaDeviceProp prop;
    cudaError_t    cuda_error_status = cudaGetDeviceProperties(&prop, device_id);
    if (cuda_error_status != cudaSuccess) {
        fprintf(stderr, "CUDA error in cudaGetDeviceProperties: %s\n", cudaGetErrorString(cuda_error_status));
    }  // END if (cuda_error_status != cudaSuccess)

    // Get allocated memory info
    size_t free_mem = 0, total_mem = 0;
    cuda_error_status = cudaSetDevice(device_id);
    if (cuda_error_status != cudaSuccess) {
        fprintf(stderr, "CUDA error in cudaSetDevice: %s\n", cudaGetErrorString(cuda_error_status));
    }  // END if (cuda_error_status != cudaSuccess)

    cuda_error_status = cudaMemGetInfo(&free_mem, &total_mem);
    if (cuda_error_status != cudaSuccess) {
        fprintf(stderr, "CUDA error in cudaMemGetInfo: %s\n", cudaGetErrorString(cuda_error_status));
    }  // END if (cuda_error_status != cudaSuccess)

    size_t allocated_mem = total_mem - free_mem;

    // Allocate buffer for properties string (adjust size as needed)
    size_t buffer_size = 4096;
    char*  properties  = (char*)malloc(buffer_size);
    if (properties == NULL) {
        fprintf(stderr, "Failed to allocate memory for properties string\n");
        return NULL;
    }  // END if (properties == NULL)

    char temp_buffer[512];
    properties[0] = '\0';  // Initialize empty string

    snprintf(temp_buffer, sizeof(temp_buffer), "Device %d Properties:\n", device_id);
    strcat(properties, temp_buffer);

    snprintf(temp_buffer, sizeof(temp_buffer), "  Name: %s\n", prop.name);
    strcat(properties, temp_buffer);

    // Note: acc_get_device_uuid needs to be implemented separately in C
    // Commenting out for now or implement if needed
    // char* uuid = acc_get_device_uuid(device_id);
    // snprintf(temp_buffer, sizeof(temp_buffer), "  UUID: %s\n", uuid);
    // strcat(properties, temp_buffer);
    // free(uuid);

    snprintf(temp_buffer, sizeof(temp_buffer), "  Total Global Memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    strcat(properties, temp_buffer);

    snprintf(temp_buffer,
             sizeof(temp_buffer),
             "  Allocated Memory: %zu MB, %zu bytes\n",
             allocated_mem / (1024 * 1024),
             allocated_mem);
    strcat(properties, temp_buffer);

    snprintf(temp_buffer, sizeof(temp_buffer), "  Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    strcat(properties, temp_buffer);

    snprintf(temp_buffer, sizeof(temp_buffer), "  Registers per Block: %d\n", prop.regsPerBlock);
    strcat(properties, temp_buffer);

    snprintf(temp_buffer, sizeof(temp_buffer), "  Warp Size: %d\n", prop.warpSize);
    strcat(properties, temp_buffer);

#if CUDART_VERSION < 13000
    snprintf(temp_buffer, sizeof(temp_buffer), "  Memory Clock Rate: %d MHz\n", prop.memoryClockRate / 1000);
    strcat(properties, temp_buffer);

    snprintf(temp_buffer,
             sizeof(temp_buffer),
             "  Peak Memory Bandwidth: %.2f GB/s\n",
             2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    strcat(properties, temp_buffer);
#else
    // For CUDA 13.0+
    if (prop.memoryBusWidth > 0) {
        double estimated_bandwidth_gbps = (prop.memoryBusWidth / 8.0) * 1000.0 / 1000.0;

        snprintf(temp_buffer, sizeof(temp_buffer), "  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        strcat(properties, temp_buffer);

        snprintf(temp_buffer,
                 sizeof(temp_buffer),
                 "  Estimated Memory Bandwidth: ~%.2f GB/s (approximate)\n",
                 estimated_bandwidth_gbps);
        strcat(properties, temp_buffer);

        strcat(properties, "  Note: Exact memory clock rate not available in CUDA 13+\n");
    } else {
        strcat(properties, "  Memory information: Not available in CUDA 13+\n");
    }  // END if (prop.memoryBusWidth > 0)

    // Get memory-related attributes available in CUDA 13+
    int l2CacheSize = 0;
    int memPitch    = 0;

    cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, device_id);
    cudaDeviceGetAttribute(&memPitch, cudaDevAttrMaxPitch, device_id);

    snprintf(temp_buffer, sizeof(temp_buffer), "  L2 Cache Size: %d KB\n", l2CacheSize / 1024);
    strcat(properties, temp_buffer);

    snprintf(temp_buffer, sizeof(temp_buffer), "  Max Memory Pitch: %d MB\n", memPitch / (1024 * 1024));
    strcat(properties, temp_buffer);

    if (prop.major >= 10) {
        strcat(properties, "  Memory Type: Likely GDDR7/HBM3E or newer\n");
    } else if (prop.major == 9) {
        strcat(properties, "  Memory Type: Likely HBM3\n");
    } else if (prop.major == 8) {
        strcat(properties, "  Memory Type: Likely GDDR6/GDDR6X or HBM2e\n");
    } else if (prop.major == 7 && prop.minor >= 5) {
        strcat(properties, "  Memory Type: Likely GDDR6\n");
    } else if (prop.major == 7 && prop.minor == 0) {
        strcat(properties, "  Memory Type: Likely HBM2\n");
    } else if (prop.major == 6) {
        strcat(properties, "  Memory Type: Likely GDDR5/GDDR5X\n");
    }  // END if (prop.major >= 10)
#endif

    snprintf(temp_buffer, sizeof(temp_buffer), "  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    strcat(properties, temp_buffer);

    snprintf(temp_buffer, sizeof(temp_buffer), "  Multiprocessor Count: %d\n", prop.multiProcessorCount);
    strcat(properties, temp_buffer);

    snprintf(temp_buffer, sizeof(temp_buffer), "  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    strcat(properties, temp_buffer);

    snprintf(temp_buffer,
             sizeof(temp_buffer),
             "  Max Threads Dim: (%d, %d, %d)\n",
             prop.maxThreadsDim[0],
             prop.maxThreadsDim[1],
             prop.maxThreadsDim[2]);
    strcat(properties, temp_buffer);

    snprintf(temp_buffer,
             sizeof(temp_buffer),
             "  Max Grid Size: (%d, %d, %d)\n",
             prop.maxGridSize[0],
             prop.maxGridSize[1],
             prop.maxGridSize[2]);
    strcat(properties, temp_buffer);

    snprintf(temp_buffer, sizeof(temp_buffer), "  Compute Capability: %d.%d\n", prop.major, prop.minor);
    strcat(properties, temp_buffer);

    return properties;
}  // END Function: acc_get_device_properties

int getSMCount() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    return props.multiProcessorCount;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// compute_total_tet_volume_gpu //////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename FloatType, typename IntType>
FloatType                                                              //
compute_total_tet_volume_gpu(const IntType           nelements,        //
                             const elems_tet4_device elements_device,  //
                             const xyz_tet4_device   xyz_device,       //
                             FloatType*              tet_volumes_device) {          //
    //
    cudaStream_t cuda_stream_vol = NULL;  // default stream
    cudaStreamCreate(&cuda_stream_vol);

    tet_grid_volumes<FloatType, IntType><<<(nelements + 255) / 256,  //
                                           256,                      //
                                           0,                        //
                                           cuda_stream_vol>>>(0,     //
                                                              nelements,
                                                              elements_device,
                                                              xyz_device,
                                                              tet_volumes_device);

    cudaStreamSynchronize(cuda_stream_vol);

    const FloatType volume_tet_tot = thrust::reduce(thrust::cuda::par.on(cuda_stream_vol),  //
                                                    tet_volumes_device,
                                                    tet_volumes_device + nelements,
                                                    (FloatType)0,
                                                    thrust::plus<FloatType>());
    cudaStreamSynchronize(cuda_stream_vol);
    cudaStreamDestroy(cuda_stream_vol);

    return volume_tet_tot;
}  // END Function: compute_total_tet_volume_gpu

#ifdef __cplusplus
extern "C" {
#endif
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

    real_t* data_device           = NULL;
    real_t* weighted_field_device = NULL;
    real_t* tet_volumes_device    = NULL;

    cudaMallocAsync((void**)&data_device, (n0 * n1 * n2) * sizeof(real_t), cuda_stream_alloc);
    cudaMallocAsync((void**)&weighted_field_device, nnodes * sizeof(real_t), cuda_stream_alloc);
    cudaMallocAsync((void**)&tet_volumes_device, nelements * sizeof(real_t), cuda_stream_alloc);

    elems_tet4_device elements_device = make_elems_tet4_device();
    cuda_allocate_elems_tet4_device_async(&elements_device, nelements, cuda_stream_alloc);

    xyz_tet4_device xyz_device = make_xyz_tet4_device();
    cuda_allocate_xyz_tet4_device_async(&xyz_device, nnodes, cuda_stream_alloc);

    cudaStreamSynchronize(cuda_stream_alloc);  /// Ensure allocations are done before proceeding further with copies

    cudaMemcpy((void*)weighted_field_device, (void*)weighted_field, nnodes * sizeof(real_t), cudaMemcpyHostToDevice);

    cudaMemset((void*)data_device, 0, (n0 * n1 * n2) * sizeof(real_t));

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
    printf("Launching tet4_resample_field_adjoint_hex_quad_kernel_gpu with: \n");
    printf("* blocks_per_grid:         %u \n", blocks_per_grid);
    printf("* threads_per_block:       %u \n", threads_per_block);
    printf("* LANES_PER_TILE_HEX_QUAD: %u \n", (unsigned int)LANES_PER_TILE_HEX_QUAD);
#endif

    cudaStream_t cuda_stream = NULL;  // default stream
    cudaStreamCreate(&cuda_stream);

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