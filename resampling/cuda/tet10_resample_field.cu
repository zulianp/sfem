#include <cooperative_groups.h>
#include <cuda_profiler_api.h>
#include <sfem_base.h>
#include <stdio.h>
#include <time.h>

// #define real_type real_t

#include "mesh_aura.h"
#include "sfem_defs.h"
#include "sfem_mesh.h"

#include "tet10_weno_cuda.cuh"

#include "quadratures_rule_cuda.cuh"
#include "tet10_resample_field.cuh"

#include "tet10_resample_field_kernels.cuh"

#define MY_RESTRICT __restrict__

// #define __TET10_TILE_SIZE__ 32
// #define WENO_CUDA 1

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
int                                                                                                                          //
launch_kernels_hex8_to_tet10_resample_field_local_CUDA(const int                            numBlocks,                       //
                                                       const int                            threadsPerBlock,                 //
                                                       const int                            bool_assemble_dual_mass_vector,  //
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
                                                       real_t*                              g_device) {                                                   //
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
        enum ElemType st = shell_type((ElemType)mesh->element_type);
        if (st == INVALID) {
            // Launch isoparametric_tet10_assemble_dual_mass_vector_kernel

            isoparametric_tet10_assemble_dual_mass_vector_kernel<<<numBlocks,                        //
                                                                   threadsPerBlock>>>(0,             //
                                                                                      nelements,     //
                                                                                      nnodes,        //
                                                                                      elems_device,  //
                                                                                      xyz_device,    //
                                                                                      mass_vector);  //

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

            compute_g_kernel_v2<<<(nnodes / threadsPerBlock) + 1,  //
                                  threadsPerBlock>>>(nnodes,       //
                                                     mass_vector,  //
                                                     g_device);    //

            // Synchronize device
            cudaDeviceSynchronize();
        } else {
            apply_inv_lumped_mass(st,               //
                                  mesh->nelements,  //
                                  mesh->nnodes,     //
                                  mesh->elements,   //
                                  mesh->points,     //
                                  g_device,         //
                                  g_device);        //
        }
    }

    RETURN_FROM_FUNCTION(0);
}

/**
 * Calculates the elapsed time between two timespec structures.
 *
 * @param start The starting timespec structure.
 * @param end The ending timespec structure.
 * @return The elapsed time in milliseconds.
 */
double get_time_tet10(struct timespec start,  //
                      struct timespec end) {
    double elapsed = (double)(end.tv_sec - start.tv_sec) * (double)1000LL;  // Convert seconds to milliseconds
    elapsed += (double)(end.tv_nsec - start.tv_nsec) / (double)1000000LL;   // Convert nanoseconds to milliseconds

    return elapsed;
}

/**
 * @brief Print the performance metrics
 *
 * @param output_file
 * @param kernel_name
 * @param mpi_rank
 * @param mpi_size
 * @param seconds
 * @param file
 * @param line
 * @param function
 * @param n_points_struct
 * @param quad_nodes_cnt
 * @param mesh
 */
void                                                            //
print_performance_metrics_tet10(FILE*         output_file,      //
                                const char*   kernel_name,      //
                                const int     mpi_rank,         //
                                const int     mpi_size,         //
                                const double  seconds,          //
                                const char*   file,             //
                                const int     line,             //
                                const char*   function,         //
                                const int     n_points_struct,  //
                                const int     quad_nodes_cnt,   //
                                const mesh_t* mesh) {           //

    MPI_Comm comm = MPI_COMM_WORLD;

    int tot_npoints_struct = 0;
    MPI_Reduce(&n_points_struct, &tot_npoints_struct, 1, MPI_INT, MPI_SUM, 0, comm);

    int tot_nelements = 0;
    MPI_Reduce(&mesh->nelements, &tot_nelements, 1, MPI_INT, MPI_SUM, 0, comm);

    int tot_nnodes = 0;
    MPI_Reduce(&mesh->n_owned_nodes, &tot_nnodes, 1, MPI_INT, MPI_SUM, 0, comm);

    if (mpi_rank != 0) return;

    const double elements_per_second          = (double)(tot_nelements) / seconds;
    const double nodes_per_second             = (double)(tot_nnodes) / seconds;
    const double quadrature_points_per_second = (double)(tot_nelements * quad_nodes_cnt) / seconds;
    const double nodes_struc_second           = (double)(tot_npoints_struct) / seconds;

    const int real_t_bits = sizeof(real_t) * 8;

    char memory_model[1000];

    if (SFEM_CUDA_MEMORY_MODEL == CUDA_HOST_MEMORY) {
        snprintf(memory_model, 1000, "Host Memory Model");
    } else if (SFEM_CUDA_MEMORY_MODEL == CUDA_MANAGED_MEMORY) {
        snprintf(memory_model, 1000, "Managed Memory Model");
    } else {
        snprintf(memory_model, 1000, "Unified Memory Model");
    }

    fprintf(output_file, "============================================================================\n");
    fprintf(output_file, "GPU TET10:    Time for the kernel (%s):\n", kernel_name);
    fprintf(output_file, "GPU TET10:    file: %s:%d \n", file, line);
    fprintf(output_file, "GPU TET10:    MPI rank: %d\n", mpi_rank);
    fprintf(output_file, "GPU TET10:    MPI size: %d\n", mpi_size);
    fprintf(output_file, "GPU TET10:    %d-bit real_t\n", real_t_bits);
    fprintf(output_file, "GPU TET10:    Memory model: %s\n", memory_model);
    fprintf(output_file, "GPU TET10:    Tile size: %d\n", __TET10_TILE_SIZE__);
    fprintf(output_file, "GPU TET10:    %f seconds\n", seconds);
    fprintf(output_file, "GPU TET10:    function:                  %s\n", function);
    fprintf(output_file, "GPU TET10:    Number of elements:        %d.\n", tot_nelements);
    fprintf(output_file, "GPU TET10:    Number of nodes:           %d.\n", tot_nnodes);
    fprintf(output_file, "GPU TET10:    Number of points struct:   %d.\n", tot_npoints_struct);
    fprintf(output_file, "GPU TET10:    Throughput for the kernel: %e elements/second\n", elements_per_second);
    fprintf(output_file, "GPU TET10:    Throughput for the kernel: %e points_struct/second\n", nodes_struc_second);
    fprintf(output_file, "GPU TET10:    Throughput for the kernel: %e nodes/second\n", nodes_per_second);
    fprintf(output_file, "GPU TET10:    Throughput for the kernel: %e quadrature_points/second\n", quadrature_points_per_second);
    fprintf(output_file, "============================================================================\n\n");
    fprintf(output_file,
            "<BenchH> mpi_rank, mpi_size, real_t_bits, tot_nelements, tot_nnodes, npoint_struc, clock, elements_second, "
            "nodes_second, "
            "nodes_struc_second, quadrature_points_second\n");
    fprintf(output_file,
            "<BenchR> %d,   %d,  %d,   %d,   %d,   %d,   %g,   %g,   %g,   %g,  %g\n",  //
            mpi_rank,                                                                   //
            mpi_size,                                                                   //
            real_t_bits,                                                                //
            tot_nelements,                                                              //
            tot_nnodes,                                                                 //
            tot_npoints_struct,                                                         //
            seconds,                                                                    //
            elements_per_second,                                                        //
            nodes_per_second,                                                           //
            nodes_struc_second,                                                         //
            quadrature_points_per_second);                                              //
    fprintf(output_file, "============================================================================\n");
}

// Function to handle printing performance metrics
void                                                           //
handle_print_performance_metrics(const char* kernel_name,      //
                                 int         mpi_rank,         //
                                 int         mpi_size,         //
                                 double      seconds,          //
                                 const char* file,             //
                                 int         line,             //
                                 const char* function,         //
                                 int         n_points_struct,  //
                                 int         npq,              //
                                 mesh_t*     mesh,             //
                                 int         print_to_file) {          //

    FILE* output_file_print = NULL;

    if (print_to_file == 1 && mpi_rank == 0) {
        char      filename[1000];
        const int real_t_bits = sizeof(real_t) * 8;
        snprintf(filename, 1000, "resampling_tet10_CUDA_mpi_size_%d_%dbit.log", mpi_size, real_t_bits);
        output_file_print = fopen(filename, "w");
    }

    // This function must be called by all ranks
    // Internally it will check if the rank is 0
    // All ranks are used to calculate the performance metrics
    print_performance_metrics_tet10(
            stdout, kernel_name, mpi_rank, mpi_size, seconds, file, line, function, n_points_struct, npq, mesh);

    if (print_to_file == 1) {
        print_performance_metrics_tet10(
                output_file_print, kernel_name, mpi_rank, mpi_size, seconds, file, line, function, n_points_struct, npq, mesh);

        if (output_file_print != NULL) fclose(output_file_print);
    }
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

    *threadsPerBlock = warp_per_block * __TET10_TILE_SIZE__;
    *numBlocks       = (nelements / warp_per_block) + (nelements % warp_per_block) + 1;
}

////////////////////////////////////////////////////////////////////////
// hex8_to_tet10_resample_field_local_CUDA_unified
////////////////////////////////////////////////////////////////////////
extern "C" int                                                                                //
hex8_to_tet10_resample_field_local_CUDA_unified_v2(const int mpi_size,                        // MPI size
                                                   const int mpi_rank,                        // MPI rank
                                                   mesh_t*   mesh,                            // Mesh data
                                                   const int bool_assemble_dual_mass_vector,  // assemble dual mass vector: 0 or 1
                                                   const ptrdiff_t* const SFEM_RESTRICT n,    // SDF: number of nodes in x y z
                                                   const ptrdiff_t* const SFEM_RESTRICT stride,  // SDF: stride of the data
                                                   const geom_t* const SFEM_RESTRICT    origin,  // Geometry: origin of the domain
                                                   const geom_t* const SFEM_RESTRICT    delta,   // Geometry: delta of the domain
                                                   const real_t* const SFEM_RESTRICT    data,    // Data: SDF
                                                   real_t* const SFEM_RESTRICT          g_host) {         // Output: g_host

    PRINT_CURRENT_FUNCTION;

    const int size_data = n[0] * n[1] * n[2];

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

    copy_elems_tet10_device_unified(mesh->nelements,                 //
                                    &elems_device,                   //
                                    (const idx_t**)mesh->elements);  //

    memory_hint_elems_tet10_device_unified(mesh->nelements,  //
                                           &elems_device);   //

    xyz_tet10_device xyz_device =                         //
            make_xyz_tet10_device_unified(mesh->nnodes);  //

    copy_xyz_tet10_device_unified(mesh->nnodes,                  //
                                  &xyz_device,                   //
                                  (const float**)mesh->points);  //

    memory_hint_xyz_tet10_device_unified(mesh->nnodes,  //
                                         &xyz_device);  //

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

    struct timespec start, end;

    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC, &start);

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

    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double clock_ms = get_time_tet10(start, end);

    const double seconds = clock_ms / 1000.0;

    const int n_points_struct = n[0] * n[1] * n[2];

    if (SFEM_LOG_LEVEL >= 5) {
        const int print_to_file = 1;
        handle_print_performance_metrics(kernel_name,      //
                                         mpi_rank,         //
                                         mpi_size,         //
                                         seconds,          //
                                         __FILE__,         //
                                         __LINE__,         //
                                         __FUNCTION__,     //
                                         n_points_struct,  //
                                         TET_QUAD_NQP,         //
                                         mesh,             //
                                         print_to_file);   //
    }

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
extern "C" int                                     //
hex8_to_tet10_resample_field_local_CUDA_Managed(   //
        const int mpi_size,                        //
        const int mpi_rank,                        //
        mesh_t*   mesh,                            // Mesh
        const int bool_assemble_dual_mass_vector,  // assemble dual mass vector
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

    const int size_data = n[0] * n[1] * n[2];

    // Managed memory
    real_t* data_device = NULL;
    real_t* mass_vector = NULL;
    real_t* g_device    = NULL;

    cudaMallocManaged(&data_device, size_data * sizeof(real_t));
    cudaMemcpy(data_device, data, size_data * sizeof(real_t), cudaMemcpyHostToDevice);

    cudaMallocManaged(&mass_vector, mesh->nnodes * sizeof(real_t));
    cudaMallocManaged(&g_device, mesh->nnodes * sizeof(real_t));

    memory_hint_write_mostly(mesh->nelements, sizeof(real_t), (void*)g_device);
    memory_hint_read_mostly(size_data, sizeof(real_t), (void*)data_device);

    elems_tet10_device elems_managed = make_elems_tet10_managed(mesh->nelements);
    copy_elems_tet10_managed(mesh->nelements, &elems_managed, (const idx_t**)mesh->elements);

    xyz_tet10_device xyz_managed = make_xyz_tet10_managed(mesh->nnodes);
    copy_xyz_tet10_managed(mesh->nnodes, &xyz_managed, (const float**)mesh->points);

    const ptrdiff_t warp_per_block  = 8;  /// 8 warps per block /////
    ptrdiff_t       numBlocks       = 0;
    ptrdiff_t       threadsPerBlock = 0;

    calculate_threads_and_blocks(mesh->nelements, warp_per_block, &threadsPerBlock, &numBlocks);

#if CUBE1 == 0  // WENO ..
    char kernel_name[] = "hex8_to_isoparametric_tet10_resample_field_local_reduce_kernel";
#else
    char kernel_name[] = "hex8_to_isoparametric_tet10_resample_field_local_cube1_kernel";
#endif

    printf("============================================================================\n");
    printf("GPU:    Managed Memory Model %s:%d \n", __FILE__, __LINE__);
    printf("GPU:    MPI size:                    %d\n", mpi_size);
    printf("GPU:    MPI rank:                    %d\n", mpi_rank);
    printf("GPU:    Launching the kernel %s \n", kernel_name);
    printf("GPU:    Number of blocks:            %ld\n", numBlocks);
    printf("GPU:    Number of threads per block: %ld\n", threadsPerBlock);
    printf("GPU:    Total number of threads:     %ld\n", (numBlocks * threadsPerBlock));
    printf("GPU:    Number of elements:          %ld\n", mesh->nelements);
    printf("GPU:    Use WENO:                    %s\n", (WENO_CUDA == 1 & CUBE1 == 1) ? "Yes" : "No");
    printf("============================================================================\n");

    cudaDeviceSynchronize();

    struct timespec start, end;

    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC, &start);

    ////  Launch the kernel
    launch_kernels_hex8_to_tet10_resample_field_local_CUDA_unified(mpi_size,                        //
                                                                   mpi_rank,                        //
                                                                   numBlocks,                       //
                                                                   threadsPerBlock,                 //
                                                                   mesh,                            //
                                                                   bool_assemble_dual_mass_vector,  //
                                                                   mesh->nelements,                 //
                                                                   mesh->nnodes,                    //
                                                                   elems_managed,                   //
                                                                   xyz_managed,                     //
                                                                   n,                               //
                                                                   stride,                          //
                                                                   origin,                          //
                                                                   delta,                           //
                                                                   data_device,                     //
                                                                   mass_vector,                     //
                                                                   g_device);                       //

    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double clock_ms = get_time_tet10(start, end);

    const double seconds = clock_ms / 1000.0;

    const int n_points_struct = n[0] * n[1] * n[2];

    if (SFEM_LOG_LEVEL >= 5) {
        const int print_to_file = 1;
        handle_print_performance_metrics(kernel_name,      //
                                         mpi_rank,         //
                                         mpi_size,         //
                                         seconds,          //
                                         __FILE__,         //
                                         __LINE__,         //
                                         __FUNCTION__,     //
                                         n_points_struct,  //
                                         TET_QUAD_NQP,         //
                                         mesh,             //
                                         print_to_file);   //
    }

    cudaMemcpy(g_host,                         //
               g_device,                       //
               mesh->nnodes * sizeof(real_t),  //
               cudaMemcpyDeviceToHost);        //

    cudaFree(data_device);
    data_device = NULL;

    cudaFree(mass_vector);
    mass_vector = NULL;

    cudaFree(g_device);
    g_device = NULL;

    free_elems_tet10_managed(&elems_managed);
    free_xyz_tet10_managed(&xyz_managed);

    RETURN_FROM_FUNCTION(0);
}

////////////////////////////////////////////////////////////////////////
// hex8_to_tet10_resample_field_local_CUDA
////////////////////////////////////////////////////////////////////////
extern "C" int                                                                                        //
hex8_to_tet10_resample_field_local_CUDA(const int                    mpi_size,                        // MPI size
                                        const int                    mpi_rank,                        // MPI rank
                                        mesh_t* const SFEM_RESTRICT  mesh,                            // Mesh data
                                        const ptrdiff_t              nelements,                       // number of elements Mesh
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

    PRINT_CURRENT_FUNCTION;

    int size_data = n[0] * n[1] * n[2];

    // Device memory
    real_t* data_device = NULL;
    real_t* mass_vector = NULL;
    real_t* g_device    = NULL;

    cudaMalloc(&data_device, size_data * sizeof(real_t));
    cudaMemcpy(data_device, data, size_data * sizeof(real_t), cudaMemcpyHostToDevice);

    elems_tet10_device elems_device = make_elems_tet10_device(nelements);
    copy_elems_tet10_device(nelements, &elems_device, (const idx_t**)elems);

    xyz_tet10_device xyz_device = make_xyz_tet10_device(nnodes);
    copy_xyz_tet10_device(nnodes, &xyz_device, (const float**)xyz);

    // Number of threads
    const ptrdiff_t warp_per_block  = 8;
    const ptrdiff_t threadsPerBlock = warp_per_block * __TET10_TILE_SIZE__;

    // Number of blocks
    const ptrdiff_t numBlocks = (nelements / warp_per_block) + (nelements % warp_per_block) + 1;

    cudaMalloc(&mass_vector, nnodes * sizeof(real_t));
    cudaMalloc(&g_device, nnodes * sizeof(real_t));

#if CUBE1 == 0  // WENO ..
    char kernel_name[] = "hex8_to_isoparametric_tet10_resample_field_local_reduce_kernel";
#else
    char kernel_name[] = "hex8_to_isoparametric_tet10_resample_field_local_cube1_kernel";
#endif

    if (SFEM_LOG_LEVEL >= 5) {
        printf("============================================================================\n");
        printf("GPU:    file: %s:%d \n", __FILE__, __LINE__);
        printf("GPU:    Host Memory Model [Default] \n");
        printf("GPU:    Launching the kernel %s \n", kernel_name);
        printf("GPU:    Number of blocks:            %ld\n", numBlocks);
        printf("GPU:    Number of threads per block: %ld\n", threadsPerBlock);
        printf("GPU:    Total number of threads:     %ld\n", (numBlocks * threadsPerBlock));
        printf("GPU:    Number of elements:          %ld\n", nelements);
        printf("GPU:    Use WENO:                    %s\n", (WENO_CUDA == 1 & CUBE1 == 1) ? "Yes" : "No");
        printf("============================================================================\n");
    }

    cudaDeviceSynchronize();

    struct timespec start, end;

    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Launch the kernels
    launch_kernels_hex8_to_tet10_resample_field_local_CUDA(numBlocks,                       //
                                                           threadsPerBlock,                 //
                                                           bool_assemble_dual_mass_vector,  //
                                                           nelements,                       //
                                                           nnodes,                          //
                                                           elems_device,                    //
                                                           xyz_device,                      //
                                                           n,                               //
                                                           stride,                          //
                                                           origin,                          //
                                                           delta,                           //
                                                           data_device,                     //
                                                           mass_vector,                     //
                                                           g_device);                       //

    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double clock_ms = get_time_tet10(start, end);

    const double seconds = clock_ms / 1000.0;

    const int n_points_struct = n[0] * n[1] * n[2];

    if (SFEM_LOG_LEVEL >= 5) {
        const int print_to_file = 1;
        handle_print_performance_metrics(kernel_name,      //
                                         mpi_rank,         //
                                         mpi_size,         //
                                         seconds,          //
                                         __FILE__,         //
                                         __LINE__,         //
                                         __FUNCTION__,     //
                                         n_points_struct,  //
                                         TET_QUAD_NQP,         //
                                         mesh,             //
                                         print_to_file);   //
    }

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
extern "C" int                                                                                //
hex8_to_tet10_resample_field_local_CUDA_wrapper(const int mpi_size,                           // MPI size
                                                const int mpi_rank,                           // MPI rank
                                                mesh_t*   mesh,                               // Mesh
                                                int*      bool_assemble_dual_mass_vector,     // assemble dual mass vector
                                                const ptrdiff_t* const SFEM_RESTRICT n,       // number of nodes in each direction
                                                const ptrdiff_t* const SFEM_RESTRICT stride,  // stride of the data
                                                const geom_t* const SFEM_RESTRICT    origin,  // origin of the domain
                                                const geom_t* const SFEM_RESTRICT    delta,   // delta of the domain
                                                const real_t* const SFEM_RESTRICT    data,    // SDF
                                                real_t* const SFEM_RESTRICT          g_host) {         // // Output //

#if SFEM_CUDA_MEMORY_MODEL == CUDA_UNIFIED_MEMORY

#pragma message "CUDA_UNIFIED_MEMORY is enabled"

    *bool_assemble_dual_mass_vector = 1;

    return hex8_to_tet10_resample_field_local_CUDA_unified_v2(mpi_size,                         //
                                                              mpi_rank,                         //
                                                              mesh,                             //
                                                              *bool_assemble_dual_mass_vector,  //
                                                              n,
                                                              stride,
                                                              origin,
                                                              delta,
                                                              data,
                                                              g_host);
#elif SFEM_CUDA_MEMORY_MODEL == CUDA_MANAGED_MEMORY

#pragma message "CUDA_MEMORY_MANAGED is enabled:"

    *bool_assemble_dual_mass_vector = 1;

    return hex8_to_tet10_resample_field_local_CUDA_Managed(mpi_size,                         //
                                                           mpi_rank,                         //
                                                           mesh,                             //
                                                           *bool_assemble_dual_mass_vector,  //
                                                           n,
                                                           stride,
                                                           origin,
                                                           delta,
                                                           data,
                                                           g_host);

#elif SFEM_CUDA_MEMORY_MODEL == CUDA_HOST_MEMORY

    // Default memory model is CUDA_HOST_MEMORY.
#pragma message "CUDA_HOST_MEMORY is enabled"

    const int mesh_nnodes           = mpi_size >= 1 ? mesh->nnodes : mesh->n_owned_nodes;
    *bool_assemble_dual_mass_vector = 0;

    return hex8_to_tet10_resample_field_local_CUDA(mpi_size,                         //
                                                   mpi_rank,                         //
                                                   mesh,                             //
                                                   mesh->nelements,                  //
                                                   mesh_nnodes,                      //
                                                   *bool_assemble_dual_mass_vector,  //
                                                   mesh->elements,                   //
                                                   mesh->points,                     //
                                                   n,                                //
                                                   stride,                           //
                                                   origin,                           //
                                                   delta,                            //
                                                   data,                             //
                                                   g_host);                          //

#endif
}
