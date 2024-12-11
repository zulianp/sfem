#include <cooperative_groups.h>
#include <cuda_profiler_api.h>
#include <sfem_base.h>
#include <stdio.h>

// #define real_type real_t

#include "mesh_aura.h"
#include "sfem_mesh.h"

#include "tet10_weno_cuda.cuh"

#include "quadratures_rule_cuda.h"
#include "tet10_resample_field.cuh"

#include "tet10_resample_field_kernels.cuh"

#define MY_RESTRICT __restrict__

#define __WARP_SIZE__ 32
#define WENO_CUDA 1

#if SFEM_TET10_WENO == ON
#define CUBE1 1
#else
#define CUBE1 0
#endif

/**
 * @brief lanches the kernels sequentially to
 * resample the field from hex8 to tet10 by applying all the necessary steps
 *
 * @param numBlocks
 * @param threadsPerBlock
 * @param nelements
 * @param nnodes
 * @param elems_device
 * @param xyz_device
 * @param n
 * @param stride
 * @param origin
 * @param delta
 * @param data_device
 * @param weighted_field_device
 * @param mass_vector
 * @param g_device
 * @return int
 */
int                                                                                               //
launch_kernels_hex8_to_tet10_resample_field_local_CUDA(const int numBlocks,                       //
                                                       const int threadsPerBlock,                 //
                                                       const int bool_assemble_dual_mass_vector,  // assemble dual mass vector
                                                       int       nelements,                       //
                                                       ptrdiff_t nnodes,                          //
                                                       elems_tet10_device                   elems_device,  //
                                                       xyz_tet10_device                     xyz_device,    //
                                                       const ptrdiff_t* const SFEM_RESTRICT n,             //
                                                       const ptrdiff_t* const SFEM_RESTRICT stride,        //
                                                       const geom_t* const SFEM_RESTRICT    origin,        //
                                                       const geom_t* const SFEM_RESTRICT    delta,         //
                                                       const real_t*                        data_device,   //
                                                       real_t*                              mass_vector,   //
                                                       real_t*                              g_device) {                                 //
    //
    PRINT_CURRENT_FUNCTION;

    // Set to zero the mass vector
    cudaMemset(mass_vector, 0, nnodes * sizeof(real_t));

    // Launch the appropriate resample field kernel based on CUBE1
#if CUBE1 == 1  // WENO
    hex8_to_isoparametric_tet10_resample_field_local_cube1_kernel
#else
    hex8_to_isoparametric_tet10_resample_field_local_reduce_kernel
#endif
            <<<numBlocks, threadsPerBlock>>>(0,
                                             nelements,
                                             nnodes,
                                             elems_device,
                                             xyz_device,
                                             n[0],
                                             n[1],
                                             n[2],
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
                                             g_device);

    // Synchronize device
    cudaDeviceSynchronize();

    if (bool_assemble_dual_mass_vector == 1) {
        // Launch isoparametric_tet10_assemble_dual_mass_vector_kernel
        isoparametric_tet10_assemble_dual_mass_vector_kernel<<<numBlocks, threadsPerBlock>>>(
                0, nelements, nnodes, elems_device, xyz_device, mass_vector);

        // Synchronize device
        cudaDeviceSynchronize();

        compute_g_kernel_v2<<<(nnodes / threadsPerBlock) + 1, threadsPerBlock>>>(nnodes, mass_vector, g_device);

        // Synchronize device
        cudaDeviceSynchronize();
    }

    RETURN_FROM_FUNCTION(0);
}

/**
 * @brief lanches the kernels sequentially to
 * resample the field from hex8 to tet10 by applying all the necessary steps for the cases where unified and Managed memory is
 * used so that the MPI communication is handled directely from / to the device.
 *
 * @param numBlocks
 * @param threadsPerBlock
 * @param nelements
 * @param nnodes
 * @param elems_device
 * @param xyz_device
 * @param n
 * @param stride
 * @param origin
 * @param delta
 * @param data_device
 * @param weighted_field_device
 * @param mass_vector
 * @param g_device
 * @return int
 */
int                                                                           //
launch_kernels_hex8_to_tet10_resample_field_local_CUDA_unified(               //
        const int                            mpi_size,                        //
        const int                            mpi_rank,                        //
        const int                            numBlocks,                       //
        const int                            threadsPerBlock,                 //
        mesh_t*                              mesh,                            // Mesh
        const int                            bool_assemble_dual_mass_vector,  // assemble dual mass vector
        int                                  nelements,                       //
        ptrdiff_t                            nnodes,                          //
        elems_tet10_device                   elems_device,                    //
        xyz_tet10_device                     xyz_device,                      //
        const ptrdiff_t* const SFEM_RESTRICT n,                               //
        const ptrdiff_t* const SFEM_RESTRICT stride,                          //
        const geom_t* const SFEM_RESTRICT    origin,                          //
        const geom_t* const SFEM_RESTRICT    delta,                           //
        const real_t*                        data_device,                     //
        real_t*                              mass_vector,                     //
        real_t*                              g_device) {
    //
    PRINT_CURRENT_FUNCTION;

    // Set to zero the mass vector
    cudaMemset(mass_vector, 0, nnodes * sizeof(real_t));

    // Launch the appropriate resample field kernel based on CUBE1
#if CUBE1 == 1  // WENO
    hex8_to_isoparametric_tet10_resample_field_local_cube1_kernel
#else
    hex8_to_isoparametric_tet10_resample_field_local_reduce_kernel
#endif
            <<<numBlocks, threadsPerBlock>>>(0,
                                             nelements,
                                             nnodes,
                                             elems_device,
                                             xyz_device,
                                             n[0],
                                             n[1],
                                             n[2],
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
                                             g_device);

    // Synchronize device
    cudaDeviceSynchronize();

    if (bool_assemble_dual_mass_vector == 1) {
        // Launch isoparametric_tet10_assemble_dual_mass_vector_kernel
        isoparametric_tet10_assemble_dual_mass_vector_kernel<<<numBlocks, threadsPerBlock>>>(
                0, nelements, nnodes, elems_device, xyz_device, mass_vector);

        // Synchronize device
        cudaDeviceSynchronize();

        if (mpi_size > 1) {
            printf("MPI:    Launching the exchange, %s:%d\n", __FILE__, __LINE__);
            send_recv_t slave_to_master;
            mesh_create_nodal_send_recv(mesh, &slave_to_master);

            ptrdiff_t count       = mesh_exchange_master_buffer_count(&slave_to_master);
            real_t*   real_buffer = (real_t*)malloc(count * sizeof(real_t));

            exchange_add(mesh, &slave_to_master, mass_vector, real_buffer);
            exchange_add(mesh, &slave_to_master, g_device, real_buffer);

            free(real_buffer);
            send_recv_destroy(&slave_to_master);
        }

        // // Launch compute_g_kernel
        // compute_g_kernel<<<(nnodes / threadsPerBlock) + 1, threadsPerBlock>>>(
        //         nnodes, weighted_field_device, mass_vector, g_device);

        compute_g_kernel_v2<<<(nnodes / threadsPerBlock) + 1, threadsPerBlock>>>(nnodes, mass_vector, g_device);

        // Synchronize device
        cudaDeviceSynchronize();
    }

    RETURN_FROM_FUNCTION(0);
}

////////////////////////////////////////////////////////////////////////
// calculate_threads_and_blocks
// Function to calculate the number of threads and blocks
// based on the number of elements and the number of warps per block
////////////////////////////////////////////////////////////////////////
void                                                      //
calculate_threads_and_blocks(ptrdiff_t  nelements,        //
                             ptrdiff_t  warp_per_block,   //
                             ptrdiff_t* threadsPerBlock,  //
                             ptrdiff_t* numBlocks) {      //

    *threadsPerBlock = warp_per_block * __WARP_SIZE__;
    *numBlocks       = (nelements / warp_per_block) + (nelements % warp_per_block) + 1;
}

////////////////////////////////////////////////////////////////////////
// hex8_to_tet10_resample_field_local_CUDA_unified
////////////////////////////////////////////////////////////////////////
extern "C" int                                                        //
hex8_to_tet10_resample_field_local_CUDA_unified(                      //
                                                                      // Mesh
        const ptrdiff_t              nelements,                       // number of elements
        const ptrdiff_t              nnodes,                          // number of nodes
        const int                    bool_assemble_dual_mass_vector,  // assemble dual mass vector
        idx_t** const SFEM_RESTRICT  elems,                           // connectivity
        geom_t** const SFEM_RESTRICT xyz,                             // coordinates
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data

        const geom_t* const SFEM_RESTRICT origin,  // origin of the domain
        const geom_t* const SFEM_RESTRICT delta,   // delta of the domain
        const real_t* const SFEM_RESTRICT data,    // SDF
        // Output //
        real_t* const SFEM_RESTRICT g_host) {  //
                                               //
    PRINT_CURRENT_FUNCTION;

    int size_data = n[0] * n[1] * n[2];

    // Device memory
    const real_t* data_device = data;
    real_t*       mass_vector = NULL;
    real_t*       g_device    = g_host;

    memory_hint_write_mostly(nelements, sizeof(real_t), (void*)g_device);
    memory_hint_read_mostly(size_data, sizeof(real_t), (void*)data_device);

    mass_vector = (real_t*)malloc(nnodes * sizeof(real_t));

    memory_hint_write_mostly(nnodes, sizeof(real_t), (void*)mass_vector);

    //// Initialize the data on the device
    elems_tet10_device elems_device =                    //
            make_elems_tet10_device_unified(nelements);  //

    copy_elems_tet10_device_unified(nelements, &elems_device, (const idx_t**)elems);
    memory_hint_elems_tet10_device_unified(nelements, &elems_device);

    xyz_tet10_device xyz_device =                   //
            make_xyz_tet10_device_unified(nnodes);  //

    copy_xyz_tet10_device_unified(nnodes, &xyz_device, (const float**)xyz);
    memory_hint_xyz_tet10_device_unified(nnodes, &xyz_device);

    const ptrdiff_t warp_per_block  = 8;  /// 8 warps per block /////
    ptrdiff_t       threadsPerBlock = 0;
    ptrdiff_t       numBlocks       = 0;

    calculate_threads_and_blocks(nelements, warp_per_block, &threadsPerBlock, &numBlocks);

#if CUBE1 == 0  // WENO ..
    char kernel_name[] = "hex8_to_isoparametric_tet10_resample_field_local_reduce_kernel";
#else
    char kernel_name[] = "hex8_to_isoparametric_tet10_resample_field_local_cube1_kernel";
#endif

    printf("============================================================================\n");
    printf("GPU:    Unified Memory Model\n");
    printf("GPU:    Launching the kernel %s \n", kernel_name);
    printf("GPU:    Number of blocks:            %ld\n", numBlocks);
    printf("GPU:    Number of threads per block: %ld\n", threadsPerBlock);
    printf("GPU:    Total number of threads:     %ld\n", (numBlocks * threadsPerBlock));
    printf("GPU:    Number of elements:          %ld\n", nelements);
    printf("GPU:    Use WENO:                    %s\n", (WENO_CUDA == 1 & CUBE1 == 1) ? "Yes" : "No");
    printf("============================================================================\n");

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch the kernels
    launch_kernels_hex8_to_tet10_resample_field_local_CUDA(numBlocks,
                                                           threadsPerBlock,
                                                           bool_assemble_dual_mass_vector,
                                                           nelements,
                                                           nnodes,
                                                           elems_device,
                                                           xyz_device,
                                                           n,
                                                           stride,
                                                           origin,
                                                           delta,
                                                           data_device,
                                                           mass_vector,
                                                           g_device);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    const double seconds = milliseconds / 1000.0;

    printf("============================================================================\n");
    printf("GPU:    Time for the kernel (%s):\n"  //
           "GPU:    %f seconds\n",                //
           kernel_name,
           seconds);
    const double elements_per_second = (double)(nelements) / seconds;
    printf("GPU:    Number of elements: %d.\n", nelements);
    printf("GPU:    Throughput for the kernel: %e elements/second\n", elements_per_second);
    printf("GPU:    %d, %f   (CSV friendly) \n", nelements, elements_per_second);
    printf("============================================================================\n");

    ////////////////////////////////////////
    /// Finalize the memory allocation
    free(mass_vector);
    mass_vector = NULL;  // free the memory allocated for mass_vector

    // The g device is already allocated in the unified memory
    // and managed by the main program
    g_device = NULL;

    free_elems_tet10_device_unified(&elems_device);
    free_xyz_tet10_device_unified(&xyz_device);

    RETURN_FROM_FUNCTION(0);
}

////////////////////////////////////////////////////////////////////////
// hex8_to_tet10_resample_field_local_CUDA_unified
////////////////////////////////////////////////////////////////////////
extern "C" int                                       //
hex8_to_tet10_resample_field_local_CUDA_unified_v2(  //
        const int mpi_size,                          //
        const int mpi_rank,                          //
        mesh_t*   mesh,                              // Mesh
        const int bool_assemble_dual_mass_vector,    // assemble dual mass vector
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data
        // Geometry
        const geom_t* const SFEM_RESTRICT origin,  // origin of the domain
        const geom_t* const SFEM_RESTRICT delta,   // delta of the domain
        // Data
        const real_t* const SFEM_RESTRICT data,  // SDF
        // Output //
        real_t* const SFEM_RESTRICT g_host) {  //
                                               //
    PRINT_CURRENT_FUNCTION;

    int size_data = n[0] * n[1] * n[2];

    // Device memory
    const real_t* data_device = data;
    real_t*       mass_vector = NULL;
    real_t*       g_device    = g_host;

    memory_hint_write_mostly(mesh->nelements, sizeof(real_t), (void*)g_device);
    memory_hint_read_mostly(size_data, sizeof(real_t), (void*)data_device);

    mass_vector = (real_t*)malloc(mesh->nnodes * sizeof(real_t));
    memory_hint_write_mostly(mesh->nnodes, sizeof(real_t), (void*)mass_vector);

    //// Initialize the data on the device
    elems_tet10_device elems_device =                          //
            make_elems_tet10_device_unified(mesh->nelements);  //

    copy_elems_tet10_device_unified(mesh->nelements, &elems_device, (const idx_t**)mesh->elements);
    memory_hint_elems_tet10_device_unified(mesh->nelements, &elems_device);

    xyz_tet10_device xyz_device =                         //
            make_xyz_tet10_device_unified(mesh->nnodes);  //

    copy_xyz_tet10_device_unified(mesh->nnodes, &xyz_device, (const float**)mesh->points);
    memory_hint_xyz_tet10_device_unified(mesh->nnodes, &xyz_device);

    const ptrdiff_t warp_per_block  = 8;  /// 8 warps per block /////
    ptrdiff_t       threadsPerBlock = 0;
    ptrdiff_t       numBlocks       = 0;

    calculate_threads_and_blocks(mesh->nelements, warp_per_block, &threadsPerBlock, &numBlocks);

#if CUBE1 == 0  // WENO ..
    char kernel_name[] = "hex8_to_isoparametric_tet10_resample_field_local_reduce_kernel";
#else
    char kernel_name[] = "hex8_to_isoparametric_tet10_resample_field_local_cube1_kernel";
#endif

    printf("============================================================================\n");
    printf("GPU:    Unified Memory Model V2 %s:%d \n", __FILE__, __LINE__);
    printf("GPU:    Mpi size:                    %d\n", mpi_size);
    printf("GPU:    Mpi rank:                    %d\n", mpi_rank);
    printf("GPU:    Launching the kernel %s \n", kernel_name);
    printf("GPU:    Number of blocks:            %ld\n", numBlocks);
    printf("GPU:    Number of threads per block: %ld\n", threadsPerBlock);
    printf("GPU:    Total number of threads:     %ld\n", (numBlocks * threadsPerBlock));
    printf("GPU:    Number of elements:          %ld\n", mesh->nelements);
    printf("GPU:    Use WENO:                    %s\n", (WENO_CUDA == 1 & CUBE1 == 1) ? "Yes" : "No");
    printf("============================================================================\n");

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    ////  Launch the kernel
    launch_kernels_hex8_to_tet10_resample_field_local_CUDA_unified(mpi_size,
                                                                   mpi_rank,
                                                                   numBlocks,
                                                                   threadsPerBlock,
                                                                   mesh,
                                                                   bool_assemble_dual_mass_vector,
                                                                   mesh->nelements,
                                                                   mesh->nnodes,
                                                                   elems_device,
                                                                   xyz_device,
                                                                   n,
                                                                   stride,
                                                                   origin,
                                                                   delta,
                                                                   data_device,
                                                                   mass_vector,
                                                                   g_device);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    const double seconds = milliseconds / 1000.0;

    printf("============================================================================\n");
    printf("GPU:    Time for the kernel (%s):\n"  //
           "GPU:    %f seconds\n",                //
           kernel_name,
           seconds);
    const double elements_per_second = (double)(mesh->nelements) / seconds;
    printf("GPU:    Number of elements: %d.\n", mesh->nelements);
    printf("GPU:    Throughput for the kernel: %e elements/second\n", elements_per_second);
    printf("GPU:    %d, %f   (CSV friendly) \n", mesh->nelements, elements_per_second);
    printf("============================================================================\n");

    ////////////////////////////////////////
    /// Finalize the memory allocation
    free(mass_vector);
    mass_vector = NULL;  // free the memory allocated for mass_vector

    // The g device is already allocated in the unified memory
    // and managed by the main program
    g_device = NULL;

    free_elems_tet10_device_unified(&elems_device);
    free_xyz_tet10_device_unified(&xyz_device);

    RETURN_FROM_FUNCTION(0);
}

////////////////////////////////////////////////////////////////////////
// hex8_to_tet10_resample_field_local_CUDA
////////////////////////////////////////////////////////////////////////
extern "C" int                                                        //
hex8_to_tet10_resample_field_local_CUDA(                              //
                                                                      // Mesh
        const ptrdiff_t              nelements,                       // number of elements
        const ptrdiff_t              nnodes,                          // number of nodes
        const int                    bool_assemble_dual_mass_vector,  // assemble dual mass vector
        idx_t** const SFEM_RESTRICT  elems,                           // connectivity
        geom_t** const SFEM_RESTRICT xyz,                             // coordinates
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data

        const geom_t* const SFEM_RESTRICT origin,  // origin of the domain
        const geom_t* const SFEM_RESTRICT delta,   // delta of the domain
        const real_t* const SFEM_RESTRICT data,    // SDF
        // Output //
        real_t* const SFEM_RESTRICT g_host) {  //
                                               // geom_t** const SFEM_RESTRICT xyz

#if SFEM_CUDA_MEMORY_MODEL == CUDA_UNIFIED_MEMORY
#pragma message "CUDA_UNIFIED_MEMORY is enabled"
    return hex8_to_tet10_resample_field_local_CUDA_unified(nelements,  //
                                                           nnodes,
                                                           bool_assemble_dual_mass_vector,
                                                           elems,
                                                           xyz,
                                                           n,
                                                           stride,
                                                           origin,
                                                           delta,
                                                           data,
                                                           g_host);
#elif SFEM_CUDA_MEMORY_MODEL == CUDA_MEMORY_MANAGED
#pragma message "CUDA_MEMORY_MANAGED is enabled: Not implemented yet"
    return -1;
#endif

    PRINT_CURRENT_FUNCTION;

    // Device memory
    real_t* data_device = NULL;
    real_t* mass_vector = NULL;
    real_t* g_device    = NULL;

    int size_data = n[0] * n[1] * n[2];
    cudaMalloc(&data_device, size_data * sizeof(real_t));
    cudaMemcpy(data_device, data, size_data * sizeof(real_t), cudaMemcpyHostToDevice);

    elems_tet10_device elems_device = make_elems_tet10_device(nelements);
    copy_elems_tet10_device(nelements, &elems_device, (const idx_t**)elems);

    xyz_tet10_device xyz_device = make_xyz_tet10_device(nnodes);
    copy_xyz_tet10_device(nnodes, &xyz_device, (const float**)xyz);

    // Number of threads
    const ptrdiff_t warp_per_block  = 8;
    const ptrdiff_t threadsPerBlock = warp_per_block * __WARP_SIZE__;

    // Number of blocks
    const ptrdiff_t numBlocks = (nelements / warp_per_block) + (nelements % warp_per_block) + 1;

    cudaMalloc(&mass_vector, nnodes * sizeof(real_t));
    cudaMalloc(&g_device, nnodes * sizeof(real_t));

#if CUBE1 == 0  // WENO ..
    char kernel_name[] = "hex8_to_isoparametric_tet10_resample_field_local_reduce_kernel";
#else
    char kernel_name[] = "hex8_to_isoparametric_tet10_resample_field_local_cube1_kernel";
#endif

    printf("============================================================================\n");
    printf("GPU:    Host Memory Model [Default] \n");
    printf("GPU:    Launching the kernel %s \n", kernel_name);
    printf("GPU:    Number of blocks:            %ld\n", numBlocks);
    printf("GPU:    Number of threads per block: %ld\n", threadsPerBlock);
    printf("GPU:    Total number of threads:     %ld\n", (numBlocks * threadsPerBlock));
    printf("GPU:    Number of elements:          %ld\n", nelements);
    printf("GPU:    Use WENO:                    %s\n", (WENO_CUDA == 1 & CUBE1 == 1) ? "Yes" : "No");
    printf("============================================================================\n");

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch the kernels
    launch_kernels_hex8_to_tet10_resample_field_local_CUDA(numBlocks,
                                                           threadsPerBlock,
                                                           bool_assemble_dual_mass_vector,
                                                           nelements,
                                                           nnodes,
                                                           elems_device,
                                                           xyz_device,
                                                           n,
                                                           stride,
                                                           origin,
                                                           delta,
                                                           data_device,
                                                           //    weighted_field_device,
                                                           mass_vector,
                                                           g_device);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    const double seconds = milliseconds / 1000.0;

    printf("============================================================================\n");
    printf("GPU:    Time for the kernel (%s):\n"  //
           "GPU:    %f seconds\n",                //
           kernel_name,
           seconds);
    const double elements_per_second = (double)(nelements) / seconds;
    printf("GPU:    Number of elements: %d.\n", nelements);
    printf("GPU:    Throughput for the kernel: %e elements/second\n", elements_per_second);
    printf("GPU:    %d, %f   (CSV friendly) \n", nelements, elements_per_second);
    printf("============================================================================\n");

    {
        cudaError_t errdd = cudaFree(data_device);
        if (errdd != cudaSuccess) printf("Error freeing device memory for data_device: %s\n", cudaGetErrorString(errdd));
    }

    free_elems_tet10_device(&elems_device);
    free_xyz_tet10_device(&xyz_device);

    cudaMemcpy(g_host,                   //
               g_device,                 //
               nnodes * sizeof(real_t),  //
               cudaMemcpyDeviceToHost);  //

    cudaError_t errg = cudaFree(g_device);
    if (errg != cudaSuccess) {
        printf("Error freeing device memory for g_device: %s\n", cudaGetErrorString(errg));
    }
    g_device = NULL;

    cudaError_t errmv = cudaFree(mass_vector);
    if (errmv != cudaSuccess) {
        printf("Error freeing device memory for mass_vector: %s\n", cudaGetErrorString(errmv));
    }
    mass_vector = NULL;

    RETURN_FROM_FUNCTION(0);
    // return 0;
}

////////////////////////////////////////////////////////////////////////
// hex8_to_tet10_resample_field_local_CUDA_wrapper
////////////////////////////////////////////////////////////////////////
extern "C" int                                     //
hex8_to_tet10_resample_field_local_CUDA_wrapper(   //
        const int mpi_size,                        // MPI size
        const int mpi_rank,                        // MPI rank
                                                   // Mesh
        mesh_t*   mesh,                            // Mesh
        const int bool_assemble_dual_mass_vector,  // assemble dual mass vector
        // SDF
        const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
        const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data

        const geom_t* const SFEM_RESTRICT origin,  // origin of the domain
        const geom_t* const SFEM_RESTRICT delta,   // delta of the domain
        const real_t* const SFEM_RESTRICT data,    // SDF
        // Output //
        real_t* const SFEM_RESTRICT g_host) {  //

#if SFEM_CUDA_MEMORY_MODEL == CUDA_UNIFIED_MEMORY

#pragma message "CUDA_UNIFIED_MEMORY is enabled"
    return hex8_to_tet10_resample_field_local_CUDA_unified_v2(mpi_size,                        //
                                                              mpi_rank,                        //
                                                              mesh,                            //
                                                              bool_assemble_dual_mass_vector,  //
                                                              n,
                                                              stride,
                                                              origin,
                                                              delta,
                                                              data,
                                                              g_host);
#elif SFEM_CUDA_MEMORY_MODEL == CUDA_MANAGED_MEMORY

#pragma message "CUDA_MEMORY_MANAGED is enabled: Not implemented yet"
    return -1;

#elif SFEM_CUDA_MEMORY_MODEL == CUDA_HOST_MEMORY

    // Default memory model is CUDA_HOST_MEMORY.

    return hex8_to_tet10_resample_field_local_CUDA(mesh->nelements,                 //
                                                   mesh->nnodes,                    //
                                                   bool_assemble_dual_mass_vector,  //
                                                   mesh->elements,                  //
                                                   mesh->points,                    //
                                                   n,                               //
                                                   stride,                          //
                                                   origin,                          //
                                                   delta,                           //
                                                   data,                            //
                                                   g_host);                         //

#endif
}
