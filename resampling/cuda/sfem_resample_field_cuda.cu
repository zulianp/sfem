

#include <cooperative_groups.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include "sfem_base.h"

// #define real_t double
#define real_type real_t

#define MY_RESTRICT __restrict__

#include "sfem_mesh.h"
#include "sfem_resample_field_cuda.cuh"

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_v2 //////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
extern "C" int                                                                 //
tet4_resample_field_local_CUDA(const ptrdiff_t                    nelements,   // Mesh
                               const ptrdiff_t                    nnodes,      // Mesh
                               int** const MY_RESTRICT            elems,       // Mesh
                               float** const MY_RESTRICT          xyz,         // Mesh
                               const ptrdiff_t* const MY_RESTRICT n,           // SDF
                               const ptrdiff_t* const MY_RESTRICT stride,      // SDF
                               const float* const MY_RESTRICT     origin,      // SDF
                               const float* const MY_RESTRICT     delta,       // SDF
                               const real_type* const MY_RESTRICT data,        // SDF
                               real_type* const MY_RESTRICT       weighted_field) {  // Output
    //

    printf("=============================================\n");
    printf("nelements = %ld\n", nelements);
    printf("=============================================\n");

    // Allocate memory on the device

    // Allocate weighted_field on the device
    real_type* weighted_field_device;
    cudaMalloc((void**)&weighted_field_device, nnodes * sizeof(real_type));
    cudaMemset(weighted_field_device, 0, sizeof(real_type) * nnodes);

    // copy the elements to the device
    elems_tet4_device elems_device;
    cuda_allocate_elems_tet4_device(&elems_device, nelements);

    cudaMemcpy(elems_device.elems_v0, elems[0], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(elems_device.elems_v1, elems[1], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(elems_device.elems_v2, elems[2], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(elems_device.elems_v3, elems[3], nelements * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate xyz on the device
    xyz_tet4_device xyz_device;
    cudaMalloc((void**)&xyz_device, 3 * sizeof(float*));
    cuda_allocate_xyz_tet4_device(&xyz_device, nnodes);
    cudaMemcpy(xyz_device.x, xyz[0], nnodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xyz_device.y, xyz[1], nnodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xyz_device.z, xyz[2], nnodes * sizeof(float), cudaMemcpyHostToDevice);

    real_type*      data_device;
    const ptrdiff_t size_data = n[0] * n[1] * n[2];
    cudaMalloc((void**)&data_device, size_data * sizeof(real_type));
    cudaMemcpy(data_device, data, size_data * sizeof(real_type), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    ///////////////////////////////////////////////////////////////////////////////
    // Call the kernel
    cudaEvent_t start, stop;

    // Number of threads
    const ptrdiff_t threadsPerBlock = 128;

    // Number of blocks
    const ptrdiff_t numBlocks = (nelements + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("============================================================================\n");
    printf("GPU:    Launching the kernel\n");
    printf("GPU:    Number of blocks:            %ld\n", numBlocks);
    printf("GPU:    Number of threads per block: %ld\n", threadsPerBlock);
    printf("GPU:    Total number of threads:     %ld\n", (numBlocks * threadsPerBlock));
    printf("GPU:    Number of elements:           %ld\n", nelements);
    printf("============================================================================\n");

    cudaEventRecord(start);

    tet4_resample_field_local_kernel<<<numBlocks, threadsPerBlock>>>(0,             //
                                                                     nelements,     //
                                                                     nnodes,        //
                                                                     elems_device,  //
                                                                     xyz_device,    //
                                                                     //  NULL, //

                                                                     stride[0],
                                                                     stride[1],
                                                                     stride[2],

                                                                     origin[0],
                                                                     origin[1],
                                                                     origin[2],

                                                                     delta[0],
                                                                     delta[1],
                                                                     delta[2],

                                                                     data_device,
                                                                     weighted_field_device);

    // Stop the timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // get cuda error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(error));
    }

    // end kernel
    ///////////////////////////////////////////////////////////////////////////////

    double time = milliseconds / 1000.0;

    const double flops = calculate_flops(nelements, TET4_NQP, time);

    const double elements_second = (double)nelements / time;

    printf("============================================================================\n");
    printf("GPU:    Elapsed time:  %e s\n", time);
    printf("GPU:    Throughput:    %e elements/second\n", elements_second);
    printf("GPU:    FLOPS:         %e FLOP/S \n", flops);
    printf("============================================================================\n");

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Free memory on the device
    free_elems_tet4_device(&elems_device);
    free_xyz_tet4_device(&xyz_device);

    // Copy the result back to the host
    cudaMemcpy(weighted_field,              //
               weighted_field_device,       //
               nnodes * sizeof(real_type),  //
               cudaMemcpyDeviceToHost);     //

    cudaFree(weighted_field_device);

    cudaFree(data_device);

    return 0;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_v2 //////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
extern "C" int                                                                        //
tet4_resample_field_local_reduce_CUDA(const ptrdiff_t                    nelements,   // Mesh: Number of elements
                                      const ptrdiff_t                    nnodes,      // Mesh: Number of nodes
                                      int** const MY_RESTRICT            elems,       // Mesh: Elements
                                      float** const MY_RESTRICT          xyz,         // Mesh: Coordinates
                                      const ptrdiff_t* const MY_RESTRICT n,           // SDF: Number of points
                                      const ptrdiff_t* const MY_RESTRICT stride,      // SDF: Stride stride[3]
                                      const float* const MY_RESTRICT     origin,      // SDF: Origin origin[3]
                                      const float* const MY_RESTRICT     delta,       // SDF: Delta delta[3]
                                      const real_type* const MY_RESTRICT data,        // SDF: Data data[n[0]*n[1]*n[2]]
                                      real_type* const MY_RESTRICT       weighted_field) {  // Output
    //
    PRINT_CURRENT_FUNCTION;

    //
    //
    printf("=============================================\n");
    printf("== tet4_resample_field_local_reduce_CUDA ====\n");
    printf("=============================================\n");
    printf("nelements = %ld\n", nelements);
    printf("=============================================\n");

    //////////////////////////////////////////////////////////////////////////
    // Allocate memory on the device

    // Allocate weighted_field on the device
    real_type* weighted_field_device;
    cudaMalloc((void**)&weighted_field_device, nnodes * sizeof(real_type));
    cudaMemset(weighted_field_device, 0, sizeof(real_type) * nnodes);

    // copy the elements to the device
    elems_tet4_device elems_device;
    cuda_allocate_elems_tet4_device(&elems_device, nelements);

    cudaMemcpy(elems_device.elems_v0, elems[0], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(elems_device.elems_v1, elems[1], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(elems_device.elems_v2, elems[2], nelements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(elems_device.elems_v3, elems[3], nelements * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate xyz on the device
    xyz_tet4_device xyz_device;
    cudaMalloc((void**)&xyz_device, 3 * sizeof(float*));
    cuda_allocate_xyz_tet4_device(&xyz_device, nnodes);
    cudaMemcpy(xyz_device.x, xyz[0], nnodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xyz_device.y, xyz[1], nnodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xyz_device.z, xyz[2], nnodes * sizeof(float), cudaMemcpyHostToDevice);

    real_type*      data_device;
    const ptrdiff_t size_data = n[0] * n[1] * n[2];
    cudaMalloc((void**)&data_device, size_data * sizeof(real_type));
    cudaMemcpy(data_device, data, size_data * sizeof(real_type), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    ///////////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////////
    // Call the kernel
    cudaEvent_t start, stop;

    // Number of threads
    const ptrdiff_t warp_per_block  = 8;
    const ptrdiff_t threadsPerBlock = warp_per_block * __WARP_SIZE__;

    // Number of blocks
    const ptrdiff_t numBlocks = (nelements / warp_per_block) + (nelements % warp_per_block) + 1;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("============================================================================\n");
    printf("GPU:    Launching the kernel Reduce \n");
    printf("GPU:    Number of blocks:            %ld\n", numBlocks);
    printf("GPU:    Number of threads per block: %ld\n", threadsPerBlock);
    printf("GPU:    Total number of threads:     %ld\n", (numBlocks * threadsPerBlock));
    printf("GPU:    Number of elements:           %ld\n", nelements);
    printf("============================================================================\n");

    cudaEventRecord(start);

    {
        tet4_resample_field_reduce_local_kernel<<<numBlocks, threadsPerBlock>>>(0,             //
                                                                                nelements,     //
                                                                                nnodes,        //
                                                                                elems_device,  //
                                                                                xyz_device,    //
                                                                                //  NULL, //

                                                                                stride[0],
                                                                                stride[1],
                                                                                stride[2],

                                                                                origin[0],
                                                                                origin[1],
                                                                                origin[2],

                                                                                delta[0],
                                                                                delta[1],
                                                                                delta[2],

                                                                                data_device,
                                                                                weighted_field_device);
    }
    //////////////////////////////////////
    //////////////////////////////////////
    //////////////////////////////////////

    // Stop the timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // get cuda error
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("!!!!!!!! ERROR: %s  !!!!!!!!!!!!!!!!!!!!!!!!!\n", cudaGetErrorString(error));
    }

    // end kernel
    ///////////////////////////////////////////////////////////////////////////////

    double time = milliseconds / 1000.0;

    const double flops = calculate_flops(nelements, TET4_NQP, time);

    const double elements_second = (double)nelements / time;

    printf("============================================================================\n");
    printf("GPU:    End kernel Reduce \n");
    printf("GPU:    Elapsed time:  %e s\n", time);
    printf("GPU:    Throughput:    %e elements/second\n", elements_second);
    printf("GPU:    FLOPS:         %e FLOP/S \n", flops);
    printf("============================================================================\n");

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Free memory on the device
    free_elems_tet4_device(&elems_device);
    free_xyz_tet4_device(&xyz_device);

    // Copy the result back to the host
    cudaMemcpy(weighted_field,              //
               weighted_field_device,       //
               nnodes * sizeof(real_type),  //
               cudaMemcpyDeviceToHost);     //

    cudaFree(weighted_field_device);

    cudaFree(data_device);

    RETURN_FROM_FUNCTION(0);
    // return 0;
}

extern "C" int                                                                              //
tet4_resample_field_local_reduce_CUDA_wrapper(const int mpi_size,                           // MPI size
                                              const int mpi_rank,                           // MPI rank
                                              mesh_t*   mesh,                               // Mesh
                                              int*      bool_assemble_dual_mass_vector,     // assemble dual mass vector
                                              const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
                                              const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data
                                              const geom_t* const SFEM_RESTRICT    origin,  // origin of the domain
                                              const geom_t* const SFEM_RESTRICT    delta,   // delta of the domain
                                              const real_t* const SFEM_RESTRICT    data,    // SDF
                                              real_t* const SFEM_RESTRICT          g_host) {         // Output

#if SFEM_CUDA_MEMORY_MODEL == CUDA_UNIFIED_MEMORY

#pragma message "CUDA_UNIFIED_MEMORY is enabled"

    *bool_assemble_dual_mass_vector = 1;

    printf("TODO: Implement the CUDA_UNIFIED_MEMORY: %s:%d\n", __FILE__, __LINE__);

    exit(EXIT_FAILURE);

#elif SFEM_CUDA_MEMORY_MODEL == CUDA_MANAGED_MEMORY

#pragma message "CUDA_MEMORY_MANAGED is enabled:"

    *bool_assemble_dual_mass_vector = 1;

    printf("TODO: Implement the CUDA_MANAGED_MEMORY: %s:%d\n", __FILE__, __LINE__);

    exit(EXIT_FAILURE);

#elif SFEM_CUDA_MEMORY_MODEL == CUDA_HOST_MEMORY

    // Default memory model is CUDA_HOST_MEMORY.
#pragma message "CUDA_HOST_MEMORY is enabled"

    const int mesh_nnodes           = mpi_size > 1 ? mesh->nnodes : mesh->n_owned_nodes;
    *bool_assemble_dual_mass_vector = 0;

    tet4_resample_field_local_reduce_CUDA(mesh->n_owned_elements,  //
                                          mesh_nnodes,             //
                                          mesh->elements,          //
                                          mesh->points,            //
                                          n,                       //
                                          stride,                  //
                                          origin,                  //
                                          delta,                   //
                                          data,                    //
                                          g_host);

#endif
}