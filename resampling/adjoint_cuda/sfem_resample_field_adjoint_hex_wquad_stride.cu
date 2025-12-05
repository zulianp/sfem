#include <cuda_runtime.h>
#include "device_utils.cuh"
#include "sfem_resample_field_adjoint_hex_wquad.cuh"
#include "sfem_resample_field_adjoint_hex_wquad_stride.cuh"
#include "sfem_resample_field_adjoint_hex_wquad_stride_gen.cuh"

#define _N_VF_ (1)  // Number of weighted fields

#ifdef __cplusplus
extern "C" {
#endif
void                                                                                                       //
call_tet4_resample_field_adjoint_hex_wquad_stride_kernel_gpu(const ptrdiff_t      start_element,           // Mesh
                                                             const ptrdiff_t      end_element,             //
                                                             const ptrdiff_t      nelements,               //
                                                             const ptrdiff_t      nnodes,                  //
                                                             const idx_t** const  elems,                   //
                                                             const geom_t** const xyz,                     //
                                                             const ptrdiff_t      n0,                      // SDF
                                                             const ptrdiff_t      n1,                      //
                                                             const ptrdiff_t      n2,                      //
                                                             const ptrdiff_t      stride0,                 // Stride
                                                             const ptrdiff_t      stride1,                 //
                                                             const ptrdiff_t      stride2,                 //
                                                             const geom_t         origin0,                 // Origin
                                                             const geom_t         origin1,                 //
                                                             const geom_t         origin2,                 //
                                                             const geom_t         dx,                      // Delta
                                                             const geom_t         dy,                      //
                                                             const geom_t         dz,                      //
                                                             const real_t* const  weighted_field[_N_VF_],  // Input weighted field
                                                             real_t* const        data[_N_VF_],            //
                                                             real_t* const        I_data) {                       //

    PRINT_CURRENT_FUNCTION;

    cudaStream_t cuda_stream_alloc = NULL;  // default stream
    cudaStreamCreate(&cuda_stream_alloc);

    // Host arrays to hold device pointers
    real_t* h_data_device[_N_VF_]           = {NULL};
    real_t* h_weighted_field_device[_N_VF_] = {NULL};

    // Device arrays of pointers
    real_t** data_device           = NULL;
    real_t** weighted_field_device = NULL;

    cudaMallocAsync((void**)&data_device, _N_VF_ * sizeof(real_t*), cuda_stream_alloc);
    cudaMallocAsync((void**)&weighted_field_device, _N_VF_ * sizeof(real_t*), cuda_stream_alloc);

    real_t* I_data_device      = NULL;
    real_t* tet_volumes_device = NULL;
    // int32_t* in_out_mesh        = NULL;

    for (int i = 0; i < _N_VF_; i++) {
        cudaMallocAsync((void**)&h_data_device[i], nnodes * sizeof(real_t), cuda_stream_alloc);
        cudaMallocAsync((void**)&h_weighted_field_device[i], nnodes * sizeof(real_t), cuda_stream_alloc);
    }  // END for (int i = 0; i < _N_VF_; i++)

    int h_stride_dim_in[_N_VF_]  = {1};  //
    int h_stride_dim_out[_N_VF_] = {1};  //

    int* stride_dim_in  = NULL;
    int* stride_dim_out = NULL;

    cudaMallocAsync((void**)&stride_dim_in, _N_VF_ * sizeof(int), cuda_stream_alloc);
    cudaMallocAsync((void**)&stride_dim_out, _N_VF_ * sizeof(int), cuda_stream_alloc);

    cudaMallocAsync((void**)&tet_volumes_device, nelements * sizeof(real_t), cuda_stream_alloc);
    // cudaMallocAsync((void**)&in_out_mesh, (n0 * n1 * n2) * sizeof(int32_t), cuda_stream_alloc);
    cudaMallocAsync((void**)&I_data_device, (n0 * n1 * n2) * sizeof(real_t), cuda_stream_alloc);

    elems_tet4_device elements_device = make_elems_tet4_device();
    cuda_allocate_elems_tet4_device_async(&elements_device, nelements, cuda_stream_alloc);

    xyz_tet4_device xyz_device = make_xyz_tet4_device();
    cuda_allocate_xyz_tet4_device_async(&xyz_device, nnodes, cuda_stream_alloc);

    cudaStreamSynchronize(cuda_stream_alloc);  /// Ensure allocations are done before proceeding further with copies

    // Now perform all memory copies after allocations are complete
    for (int i = 0; i < _N_VF_; i++) {
        cudaMemcpyAsync(  //
                h_weighted_field_device[i],
                weighted_field[i],
                nnodes * sizeof(real_t),
                cudaMemcpyHostToDevice,
                cuda_stream_alloc);
    }  // END for (int i = 0; i < _N_VF_; i++)

    // Copy the arrays of pointers to device
    cudaMemcpyAsync(data_device, h_data_device, _N_VF_ * sizeof(real_t*), cudaMemcpyHostToDevice, cuda_stream_alloc);
    cudaMemcpyAsync(weighted_field_device,     //
                    h_weighted_field_device,   //
                    _N_VF_ * sizeof(real_t*),  //
                    cudaMemcpyHostToDevice,    //
                    cuda_stream_alloc);        //

    cudaMemcpyAsync(stride_dim_in, h_stride_dim_in, _N_VF_ * sizeof(int), cudaMemcpyHostToDevice, cuda_stream_alloc);
    cudaMemcpyAsync(stride_dim_out, h_stride_dim_out, _N_VF_ * sizeof(int), cudaMemcpyHostToDevice, cuda_stream_alloc);

    cudaMemset((void*)data_device, 0, (n0 * n1 * n2) * sizeof(real_t));
    // cudaMemset((void*)in_out_mesh, 0, (n0 * n1 * n2) * sizeof(int32_t));

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

    ///////////////////////

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

    tet4_resample_field_adjoint_hex_quad_element_nw_strides_gpu_kernel<real_t,                                         //
                                                                       int,                                            //
                                                                       _N_VF_,                                         //
                                                                       matrix_ordering_t::COL_MAJOR,                   //
                                                                       matrix_ordering_t::COL_MAJOR,                   //
                                                                       3,                                              //
                                                                       false><<<blocks_per_grid,                       //
                                                                                threads_per_block,                     //
                                                                                0,                                     //
                                                                                cuda_stream>>>(start_element,          //
                                                                                               end_element,            //
                                                                                               nnodes,                 //
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
                                                                                               stride_dim_in,          //
                                                                                               data_device,            //
                                                                                               stride_dim_out,         //
                                                                                               I_data_device);         //

    cudaEventRecord(stop_event, cuda_stream);
    cudaEventSynchronize(stop_event);

    ///////////////////////

    cudaStream_t streams_copy[_N_VF_];

    for (int i = 0; i < _N_VF_; i++) {
        cudaStreamCreate(&streams_copy[i]);
        cudaMemcpyAsync((void*)data[i],
                        (const void*)h_data_device[i],
                        (n0 * n1 * n2) * sizeof(real_t),
                        cudaMemcpyDeviceToHost,
                        streams_copy[i]);
        cudaFreeAsync((void*)h_weighted_field_device[i], cuda_stream_alloc);
    }  // END for (int i = 0; i < _N_VF_; i++)
    cudaMemcpyAsync((void*)I_data,  //
                    (const void*)I_data_device,
                    (n0 * n1 * n2) * sizeof(real_t),
                    cudaMemcpyDeviceToHost,
                    cuda_stream_alloc);

    for (int i = 0; i < _N_VF_; i++) {
        cudaStreamSynchronize(streams_copy[i]);
        cudaStreamDestroy(streams_copy[i]);
    }  // END for (int i = 0; i < _N_VF_; i++)
    cudaStreamSynchronize(cuda_stream_alloc);

    // Cleanup
    for (int i = 0; i < _N_VF_; i++) {
        cudaFreeAsync(h_data_device[i], cuda_stream_alloc);
    }  // END for (int i = 0; i < _N_VF_; i++)
    cudaFreeAsync(data_device, cuda_stream_alloc);
    cudaFreeAsync(weighted_field_device, cuda_stream_alloc);
    cudaFreeAsync(stride_dim_in, cuda_stream_alloc);
    cudaFreeAsync(stride_dim_out, cuda_stream_alloc);
    // cudaFreeAsync((void*)in_out_mesh, cuda_stream_alloc);
    free_xyz_tet4_device_async(&xyz_device, cuda_stream_alloc);
    free_elems_tet4_device_async(&elements_device, cuda_stream_alloc);
    cudaStreamSynchronize(cuda_stream_alloc);
    cudaFreeAsync(I_data_device, cuda_stream_alloc);
    cudaFreeAsync(tet_volumes_device, cuda_stream_alloc);
    cudaStreamSynchronize(cuda_stream_alloc);
    cudaStreamDestroy(cuda_stream_alloc);

    RETURN_FROM_FUNCTION();
}  // END Function: call_tet4_resample_field_adjoint_hex_wquad_stride_kernel_gpu

#ifdef __cplusplus
}
#endif
//////////////////////////////////////////////////////////