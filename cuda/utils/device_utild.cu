#include "device_utils.cuh"

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