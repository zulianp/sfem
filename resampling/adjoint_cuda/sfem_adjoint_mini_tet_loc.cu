#include <stdio.h>
#include <stdlib.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include "sfem_adjoint_mini_loc_tet.cuh"
#include "sfem_adjoint_mini_tet.cuh"
#include "sfem_adjoint_mini_tet10.cuh"
#include "sfem_resample_field_cuda_fun.cuh"

// ////////////////////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void                                                                                     //
call_sfem_adjoint_mini_tet_shared_info_kernel_gpu(const ptrdiff_t             start_element,        // Mesh
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

    cudaStream_t cuda_stream_alloc = NULL;  // default stream
    cudaStreamCreate(&cuda_stream_alloc);

    tet_properties_info_t<real_t> tet_properties_info;
    tet_properties_info.alloc_async(nelements, cuda_stream_alloc);

    real_t* data_device           = NULL;
    real_t* weighted_field_device = NULL;

    cudaMallocAsync((void**)&data_device, (n0 * n1 * n2) * sizeof(real_t), cuda_stream_alloc);
    cudaMallocAsync((void**)&weighted_field_device, nnodes * sizeof(real_t), cuda_stream_alloc);

    elems_tet4_device elements_device = make_elems_tet4_device();
    cuda_allocate_elems_tet4_device_async(&elements_device, nelements, cuda_stream_alloc);

    xyz_tet4_device xyz_device = make_xyz_tet4_device();
    cuda_allocate_xyz_tet4_device_async(&xyz_device, nnodes, cuda_stream_alloc);

    cudaStreamSynchronize(cuda_stream_alloc);  /// Ensure allocations are done before proceeding further with copies

    cudaMemcpyAsync((void*)weighted_field_device,
                    (void*)weighted_field,
                    nnodes * sizeof(real_t),
                    cudaMemcpyHostToDevice,
                    cuda_stream_alloc);

    cudaMemset((void*)data_device, 0, (n0 * n1 * n2) * sizeof(real_t));

    copy_elems_tet4_device_async(elems, nelements, &elements_device, cuda_stream_alloc);

    copy_xyz_tet4_device_async(xyz, nnodes, &xyz_device, cuda_stream_alloc);

    cudaStreamSynchronize(cuda_stream_alloc);

    // Optional: check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s, at file:%s:%d \n", cudaGetErrorString(error), __FILE__, __LINE__);
    }

    const unsigned int tets_per_block = 8;
    cudaStream_t       cuda_stream    = 0;  // default stream
    cudaStreamCreate(&cuda_stream);

    cudaStream_t cuda_stream_clock = NULL;  // default stream
    cudaStreamCreate(&cuda_stream_clock);

    // cudaMemset((void*)tet_properties_info.total_size_local, 7777700087766, nelements * sizeof(ptrdiff_t));///////////

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event, cuda_stream_clock);

    {  // BEGIN: Compute local grid sizes for each element
        const unsigned int threads_per_block           = LANES_PER_TILE * tets_per_block;
        const unsigned int total_threads_per_grid_prop = (end_element - start_element + 1);
        const unsigned int blocks_per_grid = (total_threads_per_grid_prop + threads_per_block - 1) / threads_per_block;

        sfem_make_local_data_tets_kernel_gpu<real_t><<<blocks_per_grid,                      //
                                                       threads_per_block,                    //
                                                       0,                                    //
                                                       cuda_stream>>>(start_element,         // Mesh
                                                                      end_element,           //
                                                                      nnodes,                //
                                                                      elements_device,       //
                                                                      xyz_device,            //
                                                                      n0,                    // SDF
                                                                      n1,                    //
                                                                      n2,                    //
                                                                      stride0,               // Stride
                                                                      stride1,               //
                                                                      stride2,               //
                                                                      origin0,               // Origin
                                                                      origin1,               //
                                                                      origin2,               //
                                                                      dx,                    // Delta
                                                                      dy,                    //
                                                                      dz,                    //
                                                                      tet_properties_info);  //
    }  // END: Compute local grid sizes for each element

    cudaStreamSynchronize(cuda_stream);

    // Optional: check for errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s, at file:%s:%d \n", cudaGetErrorString(error), __FILE__, __LINE__);
    }

    ptrdiff_t max_total_size_local = -1;
    ptrdiff_t max_idx_global       = -1;

    ptrdiff_t min_total_size_local = -1;
    ptrdiff_t min_idx_global       = -1;

    {  // Find max and min total_size_local across all elements
        const ptrdiff_t count = (end_element - start_element);

        auto d_begin = thrust::device_pointer_cast(tet_properties_info.total_size_local) + start_element;
        auto d_end   = d_begin + count;

        auto max_it          = thrust::max_element(d_begin, d_end);
        max_total_size_local = *max_it;
        max_idx_global       = (max_it - d_begin) + start_element;

        // auto min_it          = thrust::min_element(d_begin, d_end);
        // min_total_size_local = *min_it;

        // min_idx_global = (min_it - d_begin) + start_element;
    }

    {
        const ptrdiff_t shared_memory_size = max_total_size_local * tets_per_block + tets_per_block;

        const unsigned int threads_per_block      = LANES_PER_TILE * tets_per_block;
        const unsigned int total_threads_per_grid = (end_element - start_element + 1) * LANES_PER_TILE;
        const unsigned int blocks_per_grid        = (total_threads_per_grid + threads_per_block - 1) / threads_per_block;

        sfem_adjoint_mini_tet_shared_loc_kernel_gpu<real_t>  //
                <<<blocks_per_grid,                          //
                   threads_per_block,                        //
                   sizeof(real_t) * shared_memory_size,      //
                   cuda_stream>>>(shared_memory_size,        //
                                  tets_per_block,            //
                                  start_element,             // Mesh
                                  end_element,               //
                                  nnodes,                    //
                                  elements_device,           //
                                  xyz_device,                //
                                  n0,                        // SDF
                                  n1,                        //
                                  n2,                        //
                                  stride0,                   // Stride
                                  stride1,                   //
                                  stride2,                   //
                                  origin0,                   // Origin
                                  origin1,                   //
                                  origin2,                   //
                                  dx,                        // Delta
                                  dy,                        //
                                  dz,                        //
                                  weighted_field_device,     // Input weighted field
                                  mini_tet_parameters,       // Threshold for alpha
                                  tet_properties_info,       //
                                  data_device);              //
    }

    cudaEventRecord(stop_event, cuda_stream_clock);
    cudaEventSynchronize(stop_event);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s, at file:%s:%d \n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);

    if (SFEM_LOG_LEVEL >= 1) {
        printf("================= SFEM Adjoint Mini-Tet Kernel GPU ================\n");
        printf("Kernel execution time: %f ms\n", milliseconds);
        printf("Throughput: %e   tet/s\n", (float)(end_element - start_element) / (milliseconds / 1e3));
        printf("===================================================================\n");

        printf("  Max total_size_local = %lld\n", (long long)max_total_size_local);
        printf("  Max idx global       = %lld\n", (long long)max_idx_global);

        printf("  Min total_size_local = %lld\n", (long long)min_total_size_local);
        printf("  Min idx global       = %lld\n", (long long)min_idx_global);
        printf("===================================================================\n");
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    cudaStreamDestroy(cuda_stream);
    cudaStreamDestroy(cuda_stream_clock);

    cudaMemcpy((void*)data, (void*)data_device, (n0 * n1 * n2) * sizeof(real_t), cudaMemcpyDeviceToHost);

    cudaFreeAsync((void*)weighted_field_device, cuda_stream_alloc);

    free_xyz_tet4_device_async(&xyz_device, cuda_stream_alloc);

    free_elems_tet4_device_async(&elements_device, cuda_stream_alloc);

    cudaFreeAsync(data_device, cuda_stream_alloc);
    cudaStreamDestroy(cuda_stream_alloc);
}  // END: call_sfem_adjoint_mini_tet_shared_info_kernel_gpu

// ////////////////////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
// call_sfem_adjoint_mini_tet_buffer_cluster_info_kernel_gpu
//////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
// call_sfem_adjoint_mini_tet_buffer_cluster_info_kernel_gpu
/////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" void                                                                                        //
call_sfem_adjoint_mini_tet_buffer_cluster_info_kernel_gpu(const ptrdiff_t             start_element,   // Mesh
                                                          const ptrdiff_t             end_element,     //
                                                          const ptrdiff_t             nelements,       //
                                                          const ptrdiff_t             nnodes,          //
                                                          const idx_t** const         elems,           //
                                                          const geom_t** const        xyz,             //
                                                          const ptrdiff_t             n0,              // SDF
                                                          const ptrdiff_t             n1,              //
                                                          const ptrdiff_t             n2,              //
                                                          const ptrdiff_t             stride0,         // Stride
                                                          const ptrdiff_t             stride1,         //
                                                          const ptrdiff_t             stride2,         //
                                                          const geom_t                origin0,         // Origin
                                                          const geom_t                origin1,         //
                                                          const geom_t                origin2,         //
                                                          const geom_t                dx,              // Delta
                                                          const geom_t                dy,              //
                                                          const geom_t                dz,              //
                                                          const real_t* const         weighted_field,  // Input weighted field
                                                          const mini_tet_parameters_t mini_tet_parameters,  // Threshold for alpha
                                                          real_t* const               data) {                             //
    //

    //// launch clustered kernel ////
    unsigned int cluster_size_tmp = 16;
    const char*  env_cluster_size = getenv("SFEM_CLUSTER_SIZE_ADJOINT");
    if (env_cluster_size) {
        cluster_size_tmp = atoi(env_cluster_size);
    }

    unsigned int tets_per_block_tmp = 8;
    const char*  env_tets_per_block = getenv("SFEM_TET_PER_BLOCK_ADJOINT");
    if (env_tets_per_block) {
        tets_per_block_tmp = atoi(env_tets_per_block);
    }

    cudaStream_t cuda_stream_alloc = NULL;  // default stream
    cudaStreamCreate(&cuda_stream_alloc);
    cudaStream_t cuda_stream_memset = NULL;  // default stream
    cudaStreamCreate(&cuda_stream_memset);

    tet_properties_info_t<real_t> tet_properties_info;
    tet_properties_info.alloc_async(nelements, cuda_stream_alloc);

    real_t* data_device           = NULL;
    real_t* weighted_field_device = NULL;

    cudaMallocAsync((void**)&data_device, (n0 * n1 * n2) * sizeof(real_t), cuda_stream_alloc);
    cudaMallocAsync((void**)&weighted_field_device, nnodes * sizeof(real_t), cuda_stream_alloc);

    cudaMemcpyAsync((void*)weighted_field_device,
                    (void*)weighted_field,
                    nnodes * sizeof(real_t),
                    cudaMemcpyHostToDevice,
                    cuda_stream_alloc);

    cudaMemsetAsync((void*)data_device, 0, (n0 * n1 * n2) * sizeof(real_t), cuda_stream_memset);

    elems_tet4_device elements_device = make_elems_tet4_device();
    cuda_allocate_elems_tet4_device_async(&elements_device, nelements, cuda_stream_alloc);

    xyz_tet4_device xyz_device = make_xyz_tet4_device();
    cuda_allocate_xyz_tet4_device_async(&xyz_device, nnodes, cuda_stream_alloc);

    cudaStreamSynchronize(cuda_stream_alloc);  /// Ensure allocations are done before proceeding further with copies

    copy_elems_tet4_device_async(elems, nelements, &elements_device, cuda_stream_alloc);

    copy_xyz_tet4_device_async(xyz, nnodes, &xyz_device, cuda_stream_alloc);

    cudaStreamSynchronize(cuda_stream_alloc);
    cudaStreamSynchronize(cuda_stream_memset);
    cudaStreamDestroy(cuda_stream_memset);

    // Optional: check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s, at file:%s:%d \n", cudaGetErrorString(error), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    const unsigned int tets_per_block    = tets_per_block_tmp;
    cudaStream_t       cuda_stream       = NULL;  // default stream
    cudaStream_t       cuda_stream_clock = NULL;
    cudaStreamCreate(&cuda_stream);
    cudaStreamCreate(&cuda_stream_clock);

    // cudaMemset((void*)tet_properties_info.total_size_local, 7777700087766, nelements * sizeof(ptrdiff_t));///////////

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event, cuda_stream_clock);

    {  // BEGIN: Compute local grid sizes for each element
        const unsigned int threads_per_block           = LANES_PER_TILE * tets_per_block;
        const unsigned int total_threads_per_grid_prop = (end_element - start_element + 1);
        const unsigned int blocks_per_grid = (total_threads_per_grid_prop + threads_per_block - 1) / threads_per_block;

        sfem_make_local_data_tets_kernel_gpu<real_t><<<blocks_per_grid,                      //
                                                       threads_per_block,                    //
                                                       0,                                    //
                                                       cuda_stream>>>(start_element,         // Mesh
                                                                      end_element,           //
                                                                      nnodes,                //
                                                                      elements_device,       //
                                                                      xyz_device,            //
                                                                      n0,                    // SDF
                                                                      n1,                    //
                                                                      n2,                    //
                                                                      stride0,               // Stride
                                                                      stride1,               //
                                                                      stride2,               //
                                                                      origin0,               // Origin
                                                                      origin1,               //
                                                                      origin2,               //
                                                                      dx,                    // Delta
                                                                      dy,                    //
                                                                      dz,                    //
                                                                      tet_properties_info);  //

        cudaStreamSynchronize(cuda_stream);

        // Optional: check for errors
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s, at file:%s:%d \n", cudaGetErrorString(error), __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }
    }  // END: Compute local grid sizes for each element

    ptrdiff_t max_total_size_local = -1;
    ptrdiff_t max_idx_global       = -1;

    ptrdiff_t min_total_size_local = -1;
    ptrdiff_t min_idx_global       = -1;

    {  // Find max and min total_size_local across all elements
        const ptrdiff_t count = (end_element - start_element);

        auto d_begin = thrust::device_pointer_cast(tet_properties_info.total_size_local) + start_element;
        auto d_end   = d_begin + count;

        auto max_it          = thrust::max_element(d_begin, d_end);
        max_total_size_local = *max_it;
        max_idx_global       = (max_it - d_begin) + start_element;
    }  // END: Find max and min total_size_local across all elements

    //// launch clustered kernel ////
    const unsigned int elements_per_block_approx = 1500000;
    const unsigned int cluster_size              = cluster_size_tmp;
    const unsigned int elements_per_block        = elements_per_block_approx;
    const ptrdiff_t    buffer_memory_size        = max_total_size_local * tets_per_block;

    buffer_cluster_t<real_t> buffer_cluster;                                                            //
    allocate_buffer_cluster(buffer_cluster,                                                             //
                            (elements_per_block * max_total_size_local + cluster_size) / cluster_size,  //
                            cuda_stream_alloc);                                                         //
    // cudaMemset((void*)buffer_cluster.buffer, (-1 * 0x1A7DAF1C), buffer_cluster.size * sizeof(real_t));

#if SFEM_LOG_LEVEL >= 5
    printf("max_total_size_local = %lld \n", (long long)max_total_size_local);
    printf("nelements            = %lld \n", (long long)nelements);
    printf("buffer_memory_size   = %lld \n", (long long)buffer_memory_size);
    printf("buffer_cluster.size  = %lld \n", (long long)buffer_cluster.size);
    printf("elements_per_block   = %d \n", elements_per_block);
    printf("stride0              = %lld \n", (long long)stride0);
    printf("stride1              = %lld \n", (long long)stride1);
    printf("stride2              = %lld \n", (long long)stride2);
#endif

    cudaStreamSynchronize(cuda_stream_alloc);

    {  // BEGIN: Compute local grid sizes for each element
        // const unsigned int threads_per_block    = LANES_PER_TILE * tets_per_block;
        const dim3 threads_per_block_2d = dim3(LANES_PER_TILE, tets_per_block, 1);
        const int  threads_per_block    = threads_per_block_2d.x * threads_per_block_2d.y;

        for (ptrdiff_t start_element_local = start_element;  //
             start_element_local < end_element;              //
             start_element_local += elements_per_block) {    //
            //
            ptrdiff_t end_element_local = start_element_local + elements_per_block;
            if (end_element_local > end_element) {
                end_element_local = end_element;
            }

            const unsigned int total_threads_per_grid_prop = ((end_element_local - start_element_local + 1) / cluster_size) *  //
                                                             LANES_PER_TILE;

            const unsigned int blocks_per_grid = (total_threads_per_grid_prop + threads_per_block - 1) / threads_per_block + 1;

#if SFEM_LOG_LEVEL >= 6
            printf("Launching sfem_adjoint_mini_tet_buffer_cluster_loc_kernel_gpu with: \n");
            printf("<<<blocks_per_grid, threads_per_block>>>(start_element_local, end_element_local, nelem_local);\n");
            printf("  blocks_per_grid = %u, threads_per_block = %u, nelem_local = %lld, start = %lld, end = %lld\n\n",
                   blocks_per_grid,
                   threads_per_block,
                   (long long)(end_element_local - start_element_local),
                   (long long)start_element_local,
                   (long long)end_element_local);
#endif

            sfem_adjoint_mini_tet_buffer_cluster_loc_kernel_gpu<real_t>  //
                    <<<blocks_per_grid,                                  //
                       threads_per_block_2d,
                       0,
                       cuda_stream>>>(buffer_memory_size,     // Mesh
                                      buffer_cluster,         //
                                      tets_per_block,         //
                                      cluster_size,           //
                                      start_element_local,    //
                                      end_element_local,      //
                                      nnodes,                 //
                                      elements_device,        //
                                      xyz_device,             //
                                      n0,                     // SDF
                                      n1,                     //
                                      n2,                     //
                                      stride0,                // Stride
                                      stride1,                //
                                      stride2,                //
                                      origin0,                // Origin
                                      origin1,                //
                                      origin2,                //
                                      dx,                     // Delta
                                      dy,                     //
                                      dz,                     //
                                      weighted_field_device,  // Input weighted field
                                      mini_tet_parameters,    // Threshold for alpha
                                      tet_properties_info,    //
                                      data_device);           //

            cudaStreamSynchronize(cuda_stream);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA error: %s, at file:%s:%d \n", cudaGetErrorString(err), __FILE__, __LINE__);
                exit(EXIT_FAILURE);
            }

        }  // END: Compute local grid sizes for each element

        cudaEventRecord(stop_event, cuda_stream_clock);
        cudaEventSynchronize(stop_event);

        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);

#if SFEM_LOG_LEVEL >= 5
        printf("=== Cluster Buffer Kernel: SFEM Adjoint Mini-Tet Kernel GPU ================\n");
        printf(" File: %s:%d \n", __FILE__, __LINE__);
        printf(" Kernel execution time: %f ms\n", milliseconds);
        printf(" Throughput: %e elements/s\n", (float)(end_element - start_element) / (milliseconds / 1000.0f));
        printf("<cluster_bench>  %d , %d , %d , %e, %e \n",
               cluster_size,
               tets_per_block,
               (end_element - start_element),
               (milliseconds * 1.0e-3),
               (float)(end_element - start_element) / (milliseconds * 1.0e-3));
        printf("============================================================================\n");

        printf("  Max total_size_local = %lld\n", (long long)max_total_size_local);
        printf("  Max idx global       = %lld\n", (long long)max_idx_global);

        printf("  Min total_size_local = %lld\n", (long long)min_total_size_local);
        printf("  Min idx global       = %lld\n", (long long)min_idx_global);
        printf("===================================================================\n");
#endif

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cudaStreamDestroy(cuda_stream);
        cudaStreamDestroy(cuda_stream_clock);

        clear_buffer_cluster_async(buffer_cluster, cuda_stream_alloc);
        tet_properties_info.free_async(cuda_stream_alloc);

        cudaMemcpy((void*)data, (void*)data_device, (n0 * n1 * n2) * sizeof(real_t), cudaMemcpyDeviceToHost);

        cudaFreeAsync((void*)weighted_field_device, cuda_stream_alloc);

        free_xyz_tet4_device_async(&xyz_device, cuda_stream_alloc);

        free_elems_tet4_device_async(&elements_device, cuda_stream_alloc);

        cudaFreeAsync(data_device, cuda_stream_alloc);
        cudaStreamDestroy(cuda_stream_alloc);
    }
}  // END: call_sfem_adjoint_mini_tet_shared_info_kernel_gpu