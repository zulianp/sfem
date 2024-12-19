

#include <cooperative_groups.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include "sfem_base.h"

// #define real_t double
#define real_type real_t

#define MY_RESTRICT __restrict__

#include "mass.h"
#include "sfem_mesh.h"
#include "sfem_resample_field_cuda_kernel.cuh"

extern "C" void                                   //
perform_exchange_operations(mesh_t* mesh,         //
                            real_t* mass_vector,  //
                            real_t* g);

int                                                                                                //
launch_kernels_tet4_resample_field_CUDA_unified(const int         mpi_size,                        //
                                                const int         mpi_rank,                        //
                                                const int         numBlocks,                       //
                                                const int         threadsPerBlock,                 //
                                                mesh_t*           mesh,                            // Mesh
                                                const int         bool_assemble_dual_mass_vector,  // assemble dual mass vector
                                                int               nelements,                       //
                                                ptrdiff_t         nnodes,                          //
                                                elems_tet4_device elems_device,                    //
                                                xyz_tet4_device   xyz_device,                      //
                                                const ptrdiff_t* const SFEM_RESTRICT n,            //
                                                const ptrdiff_t* const SFEM_RESTRICT stride,       //
                                                const geom_t* const SFEM_RESTRICT    origin,       //
                                                const geom_t* const SFEM_RESTRICT    delta,        //
                                                const real_t*                        data_device,  //
                                                real_t*                              mass_vector,  //
                                                real_t*                              weighted_field_device_g) {                 //
    //
    PRINT_CURRENT_FUNCTION;

    int ret = 0;

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
                                                                                weighted_field_device_g);

        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("ERROR: %s, %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
    }

    if (bool_assemble_dual_mass_vector == 1) {
        real_t* mass_vector = (real_t*)calloc(mesh->nnodes, sizeof(real_t));

        {
            enum ElemType st = (enum ElemType)shell_type((enum ElemType)mesh->element_type);
            st               = (enum ElemType)((st == INVALID) ? mesh->element_type : st);

            assemble_lumped_mass(st,               //
                                 mesh->nelements,  //
                                 mesh->nnodes,     //
                                 mesh->elements,   //
                                 mesh->points,     //
                                 mass_vector);     //
        }

        {
            // exchange ghost nodes and add contribution

            if (mpi_size > 1) {
                perform_exchange_operations(mesh, mass_vector, weighted_field_device_g);
            }  // end if mpi_size > 1

            // divide by the mass vector
            for (ptrdiff_t i = 0; i < mesh->n_owned_nodes; i++) {
                if (mass_vector[i] == 0)
                    fprintf(stderr, "Found 0 mass at %ld, info (%ld, %ld)\n", i, mesh->n_owned_nodes, mesh->nnodes);

                assert(mass_vector[i] != 0);
                weighted_field_device_g[i] /= mass_vector[i];
            }  // end for i < mesh.n_owned_nodes
        }

        free(mass_vector);
        mass_vector = NULL;
    }

    RETURN_FROM_FUNCTION(ret);
}

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
    elems_tet4_device elems_device = make_elems_tet4_device();

    cuda_allocate_elems_tet4_device(&elems_device,  //
                                    nelements);     //

    copy_elems_tet4_device((const int**)elems,  //
                           nelements,           //
                           &elems_device);      //

    // make and allocate xyz on the device
    xyz_tet4_device xyz_device = make_xyz_tet4_device();

    cuda_allocate_xyz_tet4_device(&xyz_device,  //
                                  nnodes);      //

    copy_xyz_tet4_device((const float**)xyz,  //
                         nnodes,              //
                         &xyz_device);        //

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

    cudaDeviceSynchronize();

    printf("============================================================================\n");
    printf("GPU:    Elapsed time:  %e s\n", time);
    printf("GPU:    Throughput:    %e elements/second\n", elements_second);
    printf("GPU:    FLOPS:         %e FLOP/S \n", flops);
    printf("============================================================================\n");

    // Wait for GPU to finish before accessing on host

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
    elems_tet4_device elems_device = make_elems_tet4_device();

    cuda_allocate_elems_tet4_device(&elems_device,  //
                                    nelements);     //

    copy_elems_tet4_device((const int**)elems,  //
                           nelements,           //
                           &elems_device);      //

    // make and allocate xyz on the device
    xyz_tet4_device xyz_device = make_xyz_tet4_device();

    cuda_allocate_xyz_tet4_device(&xyz_device,  //
                                  nnodes);      //

    copy_xyz_tet4_device((const float**)xyz,  //
                         nnodes,              //
                         &xyz_device);        //

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

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_v2 //////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
extern "C" int                                                                               //
tet4_resample_field_local_reduce_CUDA_wrapper(const int     mpi_size,                        // MPI size
                                              const int     mpi_rank,                        // MPI rank
                                              const mesh_t* mesh,                            // Mesh
                                              int*          bool_assemble_dual_mass_vector,  // assemble dual mass vector (Output)
                                              const ptrdiff_t* const SFEM_RESTRICT n,        // number of nodes in each direction
                                              const ptrdiff_t* const SFEM_RESTRICT stride,   // stride of the data
                                              const geom_t* const SFEM_RESTRICT    origin,   // origin of the domain
                                              const geom_t* const SFEM_RESTRICT    delta,    // delta of the domain
                                              const real_t* const SFEM_RESTRICT    data,     // SDF
                                              real_t* const SFEM_RESTRICT          g_host) {          // Output

    const int mesh_nnodes = mpi_size > 1 ? mesh->nnodes : mesh->n_owned_nodes;

    int ret = 0;

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

    *bool_assemble_dual_mass_vector = 0;

    ret = tet4_resample_field_local_reduce_CUDA(mesh->nelements,  //
                                                mesh_nnodes,      //
                                                mesh->elements,   //
                                                mesh->points,     //
                                                n,                //
                                                stride,           //
                                                origin,           //
                                                delta,            //
                                                data,             //
                                                g_host);          //

#endif

    RETURN_FROM_FUNCTION(ret);
}