#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "cell_list_cuda.cuh"
#include "cell_list_raster_gpu.h"
#include "cell_list_resampling_gpu.h"
#include "raster_cell_list_bl_gpu.cuh"
#include "raster_cell_list_gpu.cuh"

extern "C" int                                                                           //
tri3_raster_cell_quad_gpu_launch(const tri3_raster_cell_gpu_cpu_data_t *cpu_data,        //
                                 const mesh_t                          *mesh,            //
                                 const ptrdiff_t *const SFEM_RESTRICT   n,               //
                                 const ptrdiff_t *const SFEM_RESTRICT   stride,          //
                                 const geom_t *const SFEM_RESTRICT      origin,          //
                                 const geom_t *const SFEM_RESTRICT      delta,           //
                                 const real_t *const SFEM_RESTRICT      weighted_field,  //
                                 real_t *const SFEM_RESTRICT            data) {          //
    int ret = 0;

    PRINT_CURRENT_FUNCTION;

    if (cpu_data == NULL || cpu_data->split_map == NULL || cpu_data->bounding_boxes_interleaved == NULL ||
        cpu_data->geom == NULL || mesh == NULL || n == NULL || stride == NULL || origin == NULL || delta == NULL ||
        data == NULL) {
        fprintf(stderr, "Error: Invalid input to tri3_raster_cell_quad_gpu_launch\n");
        ret = EXIT_FAILURE;
        RETURN_FROM_FUNCTION(ret);
    }

    /* ── Copy split map to device ── */
    cudaStream_t stream_copy;
    cudaStreamCreate(&stream_copy);
    cell_list_split_3d_1d_map_t split_map_device = copy_cell_list_split_3d_1d_map_to_device(cpu_data->split_map, stream_copy);
    cudaStreamSynchronize(stream_copy);
    cudaStreamDestroy(stream_copy);

    /* ── Copy mesh geometry to device ── */
    cudaStream_t stream_geom;
    cudaStreamCreate(&stream_geom);
    mesh_tri3_geom_device_t geom_device = copy_mesh_tri3_geom_to_device(cpu_data->geom, mesh->nelements, stream_geom);

    /* ── Allocate output data buffer on device ── */
    cudaStream_t stream_data;
    cudaStreamCreate(&stream_data);
    real_t *data_device_ptr = NULL;
    cudaMallocAsync((void **)&data_device_ptr, sizeof(real_t) * n[0] * n[1] * n[2], stream_data);
    cudaStreamSynchronize(stream_data);
    cudaMemsetAsync(data_device_ptr, 0, sizeof(real_t) * n[0] * n[1] * n[2], stream_data);

    const ptrdiff_t delta_i = 2;
    const ptrdiff_t delta_j = 2;
    const ptrdiff_t i_size  = n[0];
    const ptrdiff_t j_size  = n[1];

    /* ── Scratch for z-intersections now lives in per-block shared memory ── */
    const int    size_tri3_intersect = 64;
    const size_t shm_bytes           = size_tri3_intersect * sizeof(real_t);

    cudaStreamSynchronize(stream_geom);
    cudaStreamSynchronize(stream_data);

#define index_type int

    /* 128 threads/block: thread 0 runs the query+sort, all threads walk k in parallel. */
    const dim3 block_size(128, 1, 1);

    printf("Launching raster kernel with GPU grid size (%d, %d, 1) and TB block size (%d, 1, 1)\n",
           (int)(i_size / delta_i + delta_i),
           (int)(j_size / delta_j + delta_j),
           block_size.x);

    cudaStream_t stream_kernel;
    cudaStreamCreate(&stream_kernel);

    cudaEvent_t kernel_start_event, kernel_stop_event;
    cudaEventCreate(&kernel_start_event);
    cudaEventCreate(&kernel_stop_event);
    cudaEventRecord(kernel_start_event, stream_kernel);

    /* The delta_i x delta_j launches write to disjoint (i,j) cells, so no sync between them. */
    for (ptrdiff_t start_i = 0; start_i < delta_i; start_i++) {
        for (ptrdiff_t start_j = 0; start_j < delta_j; start_j++) {
            const dim3 grid_size(i_size / delta_i + delta_i, j_size / delta_j + delta_j, 1);

            raster_to_hex_field_tri3_kernel<index_type>                    //
                    <<<grid_size,                                          //
                       block_size,                                         //
                       shm_bytes,                                          //
                       stream_kernel>>>(split_map_device,                  //
                                        geom_device,                       //
                                        size_tri3_intersect,               //
                                        static_cast<index_type>(start_i),  //
                                        static_cast<index_type>(start_j),  //
                                        static_cast<index_type>(delta_i),  //
                                        static_cast<index_type>(delta_j),  //
                                        static_cast<index_type>(i_size),   //
                                        static_cast<index_type>(j_size),   //
                                        static_cast<index_type>(n[2]),     //
                                        origin[0],                         //
                                        origin[1],                         //
                                        origin[2],                         //
                                        delta[0],                          //
                                        delta[1],                          //
                                        delta[2],                          //
                                        data_device_ptr);                  //
        }
    }  // END for (ptrdiff_t start_i = 0; start_i < delta_i; start_i++)

    cudaEventRecord(kernel_stop_event, stream_kernel);
    cudaEventSynchronize(kernel_stop_event);

    float kernel_milliseconds = 0.0f;
    cudaEventElapsedTime(&kernel_milliseconds, kernel_start_event, kernel_stop_event);
    printf("raster_to_hex_field_tri3_kernel (%d launches) elapsed time: %e ms\n", (int)(delta_i * delta_j), kernel_milliseconds);
    cudaEventDestroy(kernel_start_event);
    cudaEventDestroy(kernel_stop_event);

    cudaStreamSynchronize(stream_kernel);

    cudaMemcpyAsync(data, data_device_ptr, sizeof(real_t) * n[0] * n[1] * n[2], cudaMemcpyDeviceToHost, stream_data);

    /* ── Free device resources ── */
    free_mesh_tri3_geom_device(&geom_device, stream_geom);

    cudaStream_t stream_free;
    cudaStreamCreate(&stream_free);
    free_cell_list_split_3d_1d_map_device(&split_map_device, stream_free);
    cudaStreamSynchronize(stream_free);
    cudaStreamDestroy(stream_free);

    cudaStreamSynchronize(stream_geom);
    cudaStreamDestroy(stream_geom);

    cudaStreamSynchronize(stream_data);
    cudaFreeAsync(data_device_ptr, stream_data);
    cudaStreamDestroy(stream_data);

    cudaStreamSynchronize(stream_kernel);
    cudaStreamDestroy(stream_kernel);

    RETURN_FROM_FUNCTION(ret);
}  // END Function: tri3_raster_cell_quad_gpu_launch

extern "C" int                                                                              //
tri3_raster_cell_quad_bl_gpu_launch(const tri3_raster_cell_gpu_cpu_data_t *cpu_data,        //
                                    const mesh_t                          *mesh,            //
                                    const ptrdiff_t *const SFEM_RESTRICT   n,               //
                                    const ptrdiff_t *const SFEM_RESTRICT   stride,          //
                                    const geom_t *const SFEM_RESTRICT      origin,          //
                                    const geom_t *const SFEM_RESTRICT      delta,           //
                                    const real_t *const SFEM_RESTRICT      weighted_field,  //
                                    real_t *const SFEM_RESTRICT            data) {          //
    int ret = 0;

    PRINT_CURRENT_FUNCTION;

    if (cpu_data == NULL || cpu_data->split_map == NULL || cpu_data->bounding_boxes_interleaved == NULL ||
        cpu_data->geom == NULL || mesh == NULL || n == NULL || stride == NULL || origin == NULL || delta == NULL ||
        data == NULL) {
        fprintf(stderr, "Error: Invalid input to tri3_raster_cell_quad_gpu_launch\n");
        ret = EXIT_FAILURE;
        RETURN_FROM_FUNCTION(ret);
    }

    /* ── Copy split map to device ── */
    cudaStream_t stream_copy;
    cudaStreamCreate(&stream_copy);
    cell_list_split_3d_1d_map_t split_map_device = copy_cell_list_split_3d_1d_map_to_device(cpu_data->split_map, stream_copy);
    cudaStreamSynchronize(stream_copy);
    cudaStreamDestroy(stream_copy);

    /* ── Copy mesh geometry to device ── */
    cudaStream_t stream_geom;
    cudaStreamCreate(&stream_geom);
    mesh_tri3_geom_device_t geom_device = copy_mesh_tri3_geom_to_device(cpu_data->geom, mesh->nelements, stream_geom);

    /* ── Allocate output data buffer on device ── */
    cudaStream_t stream_data;
    cudaStreamCreate(&stream_data);
    real_t *data_device_ptr = NULL;
    cudaMallocAsync((void **)&data_device_ptr, sizeof(real_t) * n[0] * n[1] * n[2], stream_data);
    cudaStreamSynchronize(stream_data);
    cudaMemsetAsync(data_device_ptr, 0, sizeof(real_t) * n[0] * n[1] * n[2], stream_data);

    const ptrdiff_t i_size = n[0];
    const ptrdiff_t j_size = n[1];
    const ptrdiff_t k_size = n[2];

    /* ── Scratch for z-intersections now lives in per-block shared memory ── */

    cudaStreamSynchronize(stream_geom);
    cudaStreamSynchronize(stream_data);

#define index_type int

    /* 256 threads/block, one (i,j) column per thread. The z-intersection
       scratch is a per-thread local array inside the kernel (capacity is the
       MAX_INTERSECT template argument), so no dynamic shared memory is needed
       and occupancy is not shm-bound. */
    const dim3 block_size(256, 1, 1);

    constexpr int max_tri3_intersect = 64; /* per-thread capacity, same as the block-per-column kernel */

    const dim3 grid_size((i_size * j_size + block_size.x - 1) / block_size.x, 1, 1);

    printf("Launching raster kernel with GPU grid size (%d, 1, 1) and TB block size (%d, 1, 1)\n", grid_size.x, block_size.x);

    cudaStream_t stream_kernel;
    cudaStreamCreate(&stream_kernel);

    cudaEvent_t kernel_start_event, kernel_stop_event;
    cudaEventCreate(&kernel_start_event);
    cudaEventCreate(&kernel_stop_event);
    cudaEventRecord(kernel_start_event, stream_kernel);

    {
        raster_to_hex_field_bl_tri3_kernel<index_type, max_tri3_intersect>  //
                <<<grid_size,                                         //
                   block_size,                                        //
                   0,                                                 //
                   stream_kernel>>>(split_map_device,                 //
                                    geom_device,                      //
                                    static_cast<index_type>(i_size),  //
                                    static_cast<index_type>(j_size),  //
                                    static_cast<index_type>(k_size),  //
                                    origin[0],                        //
                                    origin[1],                        //
                                    origin[2],                        //
                                    delta[0],                         //
                                    delta[1],                         //
                                    delta[2],                         //
                                    data_device_ptr);                 //
    }

    cudaEventRecord(kernel_stop_event, stream_kernel);
    cudaEventSynchronize(kernel_stop_event);

    float kernel_milliseconds = 0.0f;
    cudaEventElapsedTime(&kernel_milliseconds, kernel_start_event, kernel_stop_event);
    printf("raster_to_hex_field_bl_tri3_kernel elapsed time: %e ms\n", kernel_milliseconds);
    cudaEventDestroy(kernel_start_event);
    cudaEventDestroy(kernel_stop_event);

    cudaStreamSynchronize(stream_kernel);

    const cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        fprintf(stderr, "Error: raster_to_hex_field_bl_tri3_kernel failed: %s\n", cudaGetErrorString(kernel_err));
        ret = EXIT_FAILURE;
    }

    cudaMemcpyAsync(data, data_device_ptr, sizeof(real_t) * i_size * j_size * k_size, cudaMemcpyDeviceToHost, stream_data);

    /* ── Free device resources ── */
    free_mesh_tri3_geom_device(&geom_device, stream_geom);

    cudaStream_t stream_free;
    cudaStreamCreate(&stream_free);
    free_cell_list_split_3d_1d_map_device(&split_map_device, stream_free);
    cudaStreamSynchronize(stream_free);
    cudaStreamDestroy(stream_free);

    cudaStreamSynchronize(stream_geom);
    cudaStreamDestroy(stream_geom);

    cudaStreamSynchronize(stream_data);
    cudaFreeAsync(data_device_ptr, stream_data);
    cudaStreamDestroy(stream_data);

    cudaStreamSynchronize(stream_kernel);
    cudaStreamDestroy(stream_kernel);

    RETURN_FROM_FUNCTION(ret);
}  // END Function: tri3_raster_cell_quad_gpu_launch
