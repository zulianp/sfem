

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

#define HANDLE_CUDA_ERROR(err)                                 \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "Error: %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                    \
    }

/**
 * @brief Exchange ghost nodes and add contribution
 */
extern "C" void                                   //
perform_exchange_operations(mesh_t* mesh,         //
                            real_t* mass_vector,  //
                            real_t* g);           //

/**
 * @brief
 *
 * @param px0
 * @param px1
 * @param px2
 * @param px3
 * @param py0
 * @param py1
 * @param py2
 * @param py3
 * @param pz0
 * @param pz1
 * @param pz2
 * @param pz3
 * @param element_vector
 * @return __device__
 */
__device__ void                                                                         //
lumped_mass_cu(const real_t px0, const real_t px1, const real_t px2, const real_t px3,  //
               const real_t py0, const real_t py1, const real_t py2, const real_t py3,  //
               const real_t pz0, const real_t pz1, const real_t pz2, const real_t pz3,  //
               real_t* element_vector) {                                                //
                                                                                        //
    // FLOATING POINT OPS!
    //       - Result: 4*ASSIGNMENT
    //       - Subexpressions: 11*ADD + 16*DIV + 48*MUL + 12*SUB
    const real_t x0 = (1.0 / 24.0) * px0;
    const real_t x1 = (1.0 / 24.0) * px1;
    const real_t x2 = (1.0 / 24.0) * px2;
    const real_t x3 = (1.0 / 24.0) * px3;
    const real_t x4 = (1.0 / 24.0) * px0 * py1 * pz3 + (1.0 / 24.0) * px0 * py2 * pz1 + (1.0 / 24.0) * px0 * py3 * pz2 +
                      (1.0 / 24.0) * px1 * py0 * pz2 + (1.0 / 24.0) * px1 * py2 * pz3 + (1.0 / 24.0) * px1 * py3 * pz0 +
                      (1.0 / 24.0) * px2 * py0 * pz3 + (1.0 / 24.0) * px2 * py1 * pz0 + (1.0 / 24.0) * px2 * py3 * pz1 +
                      (1.0 / 24.0) * px3 * py0 * pz1 + (1.0 / 24.0) * px3 * py1 * pz2 + (1.0 / 24.0) * px3 * py2 * pz0 -
                      py0 * pz1 * x2 - py0 * pz2 * x3 - py0 * pz3 * x1 - py1 * pz0 * x3 - py1 * pz2 * x0 - py1 * pz3 * x2 -
                      py2 * pz0 * x1 - py2 * pz1 * x3 - py2 * pz3 * x0 - py3 * pz0 * x2 - py3 * pz1 * x0 - py3 * pz2 * x1;
    element_vector[0] = x4;
    element_vector[1] = x4;
    element_vector[2] = x4;
    element_vector[3] = x4;
}

/**
 * @brief
 *
 * @param nelements
 * @param nnodes
 * @param elems_device
 * @param xyz
 * @param values
 * @return __global__
 */
__global__ void                                                             //
tet4_assemble_lumped_mass_kernel(const ptrdiff_t             nelements,     //
                                 const ptrdiff_t             nnodes,        //
                                 elems_tet4_device           elems_device,  //
                                 xyz_tet4_device             xyz,           //
                                 real_t* const SFEM_RESTRICT values) {      //

    // SFEM_UNUSED(nnodes);

    // double tick = MPI_Wtime();

    idx_t ev[4];
    idx_t ks[4];

    real_t element_vector[4];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelements) {
        return;
    }

    ev[0] = elems_device.elems_v0[i];
    ev[1] = elems_device.elems_v1[i];
    ev[2] = elems_device.elems_v2[i];
    ev[3] = elems_device.elems_v3[i];

    // Element indices
    const idx_t i0 = ev[0];
    const idx_t i1 = ev[1];
    const idx_t i2 = ev[2];
    const idx_t i3 = ev[3];

    lumped_mass_cu(
            // X-coordinates
            xyz.x[i0],
            xyz.x[i1],
            xyz.x[i2],
            xyz.x[i3],
            // Y-coordinates
            xyz.y[i0],
            xyz.y[i1],
            xyz.y[i2],
            xyz.y[i3],
            // Z-coordinates
            xyz.z[i0],
            xyz.z[i1],
            xyz.z[i2],
            xyz.z[i3],
            element_vector);

    for (int edof_i = 0; edof_i < 4; ++edof_i) {
        values[ev[edof_i]] += element_vector[edof_i];
    }

    // double tock = MPI_Wtime();
    // printf("tet4_mass.c: tet4_assemble_lumped_mass\t%g seconds\n", tock - tick);
}

// CUDA kernel to divide by the mass vector
__global__ void                                                      //
divide_by_mass_vector_kernel(ptrdiff_t     n_owned_nodes,            //
                             real_t*       weighted_field_device_g,  //
                             const real_t* mass_vector) {            //

    ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_owned_nodes) {
        return;
    }

    if (mass_vector[i] == 0) {
        assert(mass_vector[i] != 0 && "Found 0 mass");
    }

    assert(mass_vector[i] != 0);
    weighted_field_device_g[i] /= mass_vector[i];
}

/**
 * @brief kernel for unified and managed memory
 *
 * @param mpi_size
 * @param mpi_rank
 * @param numBlocks
 * @param threadsPerBlock
 * @param mesh
 * @param bool_assemble_dual_mass_vector
 * @param nelements
 * @param nnodes
 * @param elems_device
 * @param xyz_device
 * @param n
 * @param stride
 * @param origin
 * @param delta
 * @param data_device
 * @param mass_vector
 * @param weighted_field_device_g
 * @return int
 */
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
        HANDLE_CUDA_ERROR(error);
    }

    if (bool_assemble_dual_mass_vector == 1) {
        // real_t* mass_vector = (real_t*)malloc(mesh->nnodes * sizeof(real_t));
        real_t* mass_vector = NULL;

        cudaError_t err = cudaMalloc((void**)&mass_vector,            //
                                     mesh->nnodes * sizeof(real_t));  //
        HANDLE_CUDA_ERROR(err);

        const int threadPerBlock = 256;

        {
            const int numBlocks = (nelements + threadPerBlock - 1) / threadPerBlock;  //
            tet4_assemble_lumped_mass_kernel<<<numBlocks,                             //
                                               threadPerBlock>>>(mesh->nelements,     //
                                                                 mesh->nnodes,        //
                                                                 elems_device,        //
                                                                 xyz_device,          //
                                                                 mass_vector);        //
        }

        {  // exchange ghost nodes and add contribution

            if (mpi_size > 1) {
                perform_exchange_operations(mesh, mass_vector, weighted_field_device_g);
            }  // end if mpi_size > 1

            const int numBlocks = (mesh->nnodes + threadPerBlock - 1) / threadPerBlock;  //
            divide_by_mass_vector_kernel<<<numBlocks,                                    //
                                           threadPerBlock>>>(mesh->nnodes,               //
                                                             weighted_field_device_g,    //
                                                             mass_vector);               //
        }

        cudaError_t error = cudaFree(mass_vector);
        mass_vector       = NULL;
        HANDLE_CUDA_ERROR(error);
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

/**
 * @brief Construct a new tet4 resample field local reduce CUDA managed object
 *
 * @param mpi_size
 * @param mpi_rank
 * @param mesh
 * @param bool_assemble_dual_mass_vector
 * @param n
 * @param stride
 * @param origin
 * @param delta
 * @param data
 * @param g_host
 */
int                                                                                          //
tet4_resample_field_local_reduce_CUDA_managed(const int     mpi_size,                        // MPI size
                                              const int     mpi_rank,                        // MPI rank
                                              const mesh_t* mesh,                            // Mesh
                                              int*          bool_assemble_dual_mass_vector,  // assemble dual mass vector (Output)
                                              const ptrdiff_t* const SFEM_RESTRICT n,        // number of nodes in each direction
                                              const ptrdiff_t* const SFEM_RESTRICT stride,   // stride of the data
                                              const geom_t* const SFEM_RESTRICT    origin,   // origin of the domain
                                              const geom_t* const SFEM_RESTRICT    delta,    // delta of the domain
                                              const real_t* const SFEM_RESTRICT    data,     // SDF
                                              real_t* const SFEM_RESTRICT          g_host) {          // Output

    PRINT_CURRENT_FUNCTION;

    const int mesh_nnodes = mpi_size > 1 ? mesh->nnodes : mesh->n_owned_nodes;

    int ret = 0;

    // Allocate weighted_field on the device
    real_type* weighted_field_device;
    cudaMallocManaged((void**)&weighted_field_device, mesh->nnodes * sizeof(real_type));
    cudaMemset(weighted_field_device, 0, sizeof(real_type) * mesh->nnodes);

    // copy the elements to the device
    elems_tet4_device elems_device = make_elems_tet4_device();

    cuda_allocate_elems_tet4_device_managed(&elems_device,     //
                                            mesh->nelements);  //

    copy_elems_tet4_device((const int**)mesh->elements,  //
                           mesh->nelements,              //
                           &elems_device);               //

    // make and allocate xyz on the device
    xyz_tet4_device xyz_device = make_xyz_tet4_device();

    cuda_allocate_xyz_tet4_device(&xyz_device,    //
                                  mesh->nnodes);  //

    copy_xyz_tet4_device((const float**)mesh->points,  //
                         mesh->nnodes,                 //
                         &xyz_device);                 //

    real_type*      data_device;
    const ptrdiff_t size_data = n[0] * n[1] * n[2];
    cudaMallocManaged((void**)&data_device, size_data * sizeof(real_type));
    cudaMemcpy(data_device, data, size_data * sizeof(real_type), cudaMemcpyHostToDevice);

    ///////////////////////////////////////////////////////////////////////////////
    // Call the kernel
    cudaEvent_t start, stop;

    // // TODO Launch the kernel
    // launch_kernels_tet4_resample_field_CUDA_unified(mpi_size,                //
    //                                                 mpi_rank,                //
    //                                                 mesh->nelements,         //
    //                                                 mesh_nnodes,             //
    //                                                 elems_device,            //
    //                                                 xyz_device,              //
    //                                                 n,                       //
    //                                                 stride,                  //
    //                                                 origin,                  //
    //                                                 delta,                   //
    //                                                 data_device,             //
    //                                                 weighted_field_device);  //

    cudaDeviceSynchronize();

    // Free memory on the device
    free_elems_tet4_device(&elems_device);
    free_xyz_tet4_device(&xyz_device);

    // Copy the result back to the host
    cudaMemcpy(g_host,                            //
               weighted_field_device,             //
               mesh->nnodes * sizeof(real_type),  //
               cudaMemcpyDeviceToHost);           //

    cudaFree(weighted_field_device);

    cudaFree(data_device);

    RETURN_FROM_FUNCTION(ret);
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

    PRINT_CURRENT_FUNCTION;

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