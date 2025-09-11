#include <stdio.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include "sfem_adjoint_mini_loc_tet.cuh"
#include "sfem_adjoint_mini_tet.cuh"
#include "sfem_resample_field_cuda_fun.cuh"

extern "C" void                                                                         //
call_sfem_adjoint_mini_tet_kernel_gpu(const ptrdiff_t             start_element,        // Mesh
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

    const unsigned int threads_per_block      = 256;
    const unsigned int total_threads_per_grid = (end_element - start_element + 1) * LANES_PER_TILE;
    const unsigned int blocks_per_grid        = (total_threads_per_grid + threads_per_block - 1) / threads_per_block;

    cudaStream_t cuda_stream = 0;  // default stream
    cudaStreamCreate(&cuda_stream);

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

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event, cuda_stream);

    sfem_adjoint_mini_tet_kernel_gpu<real_t><<<blocks_per_grid,                       //
                                               threads_per_block,                     //
                                               0,                                     //
                                               cuda_stream>>>(start_element,          // Mesh
                                                              end_element,            //
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
                                                              data_device);           //

    cudaStreamSynchronize(cuda_stream);

    // Optional: check for errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s, at file:%s:%d \n", cudaGetErrorString(error), __FILE__, __LINE__);
    }

    cudaEventRecord(stop_event, cuda_stream);
    cudaEventSynchronize(stop_event);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);

    if (SFEM_LOG_LEVEL >= 5) {
        printf("================= SFEM Adjoint Mini-Tet Kernel GPU ================\n");
        printf("Kernel execution time: %f ms\n", milliseconds);
        printf("===================================================================\n");
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    cudaStreamDestroy(cuda_stream);

    cudaMemcpy((void*)data, (void*)data_device, (n0 * n1 * n2) * sizeof(real_t), cudaMemcpyDeviceToHost);

    cudaFreeAsync((void*)weighted_field_device, cuda_stream_alloc);

    free_xyz_tet4_device_async(&xyz_device, cuda_stream_alloc);

    free_elems_tet4_device_async(&elements_device, cuda_stream_alloc);

    cudaFreeAsync(data_device, cuda_stream_alloc);
    cudaStreamDestroy(cuda_stream_alloc);

}  // END: call_sfem_adjoint_mini_tet_kernel_gpu
// ////////////////////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////////////////////

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

    // cudaMemset((void*)tet_properties_info.total_size_local, 7777700087766, nelements * sizeof(ptrdiff_t));///////////

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event, cuda_stream);

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
        const ptrdiff_t shared_memory_size = max_total_size_local * tets_per_block;

        const unsigned int threads_per_block      = LANES_PER_TILE * tets_per_block;
        const unsigned int total_threads_per_grid = (end_element - start_element + 1) * LANES_PER_TILE;
        const unsigned int blocks_per_grid        = (total_threads_per_grid + threads_per_block - 1) / threads_per_block;

        sfem_adjoint_mini_tet_shared_loc_kernel_gpu<real_t><<<blocks_per_grid,                       //
                                                              threads_per_block,                     //
                                                              shared_memory_size,                    //
                                                              cuda_stream>>>(shared_memory_size,     //
                                                                             start_element,          // Mesh
                                                                             end_element,            //
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
                                                                             data_device);           //
    }

    cudaEventRecord(stop_event, cuda_stream);
    cudaEventSynchronize(stop_event);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);

    if (SFEM_LOG_LEVEL >= 1) {
        printf("================= SFEM Adjoint Mini-Tet Kernel GPU ================\n");
        printf("Kernel execution time: %f ms\n", milliseconds);
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

    cudaMemcpy((void*)data, (void*)data_device, (n0 * n1 * n2) * sizeof(real_t), cudaMemcpyDeviceToHost);

    cudaFreeAsync((void*)weighted_field_device, cuda_stream_alloc);

    free_xyz_tet4_device_async(&xyz_device, cuda_stream_alloc);

    free_elems_tet4_device_async(&elements_device, cuda_stream_alloc);

    cudaFreeAsync(data_device, cuda_stream_alloc);
    cudaStreamDestroy(cuda_stream_alloc);
}
