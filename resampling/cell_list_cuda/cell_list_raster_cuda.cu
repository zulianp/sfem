#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "cell_build_tet_geom.cuh"
#include "cell_list_cuda.cuh"
#include "cell_list_raster_gpu.h"
#include "cell_list_resampling_gpu.h"
#include "raster_cell_list_gpu.cuh"
#include "resample_field_adjoint_cell_cuda.cuh"
#include "resample_field_adjoint_cell_cuda_shm.cuh"

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
        cpu_data->geom == NULL || mesh == NULL || n == NULL || stride == NULL ||
        origin == NULL || delta == NULL || data == NULL) {
        fprintf(stderr, "Error: Invalid input to tri3_raster_cell_quad_gpu_launch\n");
        ret = EXIT_FAILURE;
        RETURN_FROM_FUNCTION(ret);
    }

    /* ── Copy split map to device ── */
    cudaStream_t                stream_copy;
    cudaStreamCreate(&stream_copy);
    cell_list_split_3d_1d_map_t split_map_device =
            copy_cell_list_split_3d_1d_map_to_device(cpu_data->split_map, stream_copy);
    cudaStreamSynchronize(stream_copy);
    cudaStreamDestroy(stream_copy);

    /* ── Copy mesh geometry to device ── */
    cudaStream_t            stream_geom;
    cudaStreamCreate(&stream_geom);
    mesh_tri3_geom_device_t geom_device =
            copy_mesh_tri3_geom_to_device(cpu_data->geom, mesh->nelements, stream_geom);

    /* ── Allocate output data buffer on device ── */
    cudaStream_t stream_data;
    cudaStreamCreate(&stream_data);
    real_t *data_device_ptr = NULL;
    cudaMallocAsync((void **)&data_device_ptr, sizeof(real_t) * n[0] * n[1], stream_data);
    cudaStreamSynchronize(stream_data);
    cudaMemsetAsync(data_device_ptr, 0, sizeof(real_t) * n[0] * n[1], stream_data);

    const ptrdiff_t delta_i = 2;
    const ptrdiff_t delta_j = 2;
    const ptrdiff_t i_size  = n[0];
    const ptrdiff_t j_size  = n[1];

    /* ── Allocate per-block scratch buffer for tri3 z-intersections ──
     * Each block processes one (i,j) column independently; allocate enough
     * capacity for all concurrent blocks to avoid inter-block aliasing. */
    const int       size_tri3_intersect    = 64;
    const ptrdiff_t max_blocks_x           = i_size / delta_i + delta_i + 1;
    const ptrdiff_t max_blocks_y           = j_size / delta_j + delta_j + 1;
    const ptrdiff_t total_blocks           = max_blocks_x * max_blocks_y;
    cudaStream_t    stream_intersect;
    cudaStreamCreate(&stream_intersect);
    real_t *tri3_intersect_z_device = NULL;
    cudaMallocAsync((void **)&tri3_intersect_z_device,
                    sizeof(real_t) * total_blocks * size_tri3_intersect,
                    stream_intersect);

    cudaStreamSynchronize(stream_geom);
    cudaStreamSynchronize(stream_data);
    cudaStreamSynchronize(stream_intersect);

#define index_type int

    printf("Launching raster kernel with GPU grid size (%d, %d, 1) and TB block size (1, 1, 1)\n",
           (int)(i_size / delta_i + delta_i),
           (int)(j_size / delta_j + delta_j));

    cudaStream_t stream_kernel;
    cudaStreamCreate(&stream_kernel);

    for (ptrdiff_t start_i = 0; start_i < delta_i; start_i++) {
        for (ptrdiff_t start_j = 0; start_j < delta_j; start_j++) {
            dim3 grid_size(i_size / delta_i + delta_i, j_size / delta_j + delta_j, 1);
            dim3 block_size(1, 1, 1);

            raster_to_hex_field_tri3_kernel<index_type>  //
                    <<<grid_size,                         //
                       block_size,                        //
                       0,                                 //
                       stream_kernel>>>(split_map_device,                              //
                                        geom_device,                                  //
                                        tri3_intersect_z_device,                      //
                                        size_tri3_intersect,                          //
                                        static_cast<index_type>(start_i),             //
                                        static_cast<index_type>(start_j),             //
                                        static_cast<index_type>(delta_i),             //
                                        static_cast<index_type>(delta_j),             //
                                        static_cast<index_type>(i_size),              //
                                        static_cast<index_type>(j_size),              //
                                        origin[0],                                    //
                                        origin[1],                                    //
                                        delta[0],                                     //
                                        delta[1]);                                    //

            cudaStreamSynchronize(stream_kernel);
        }
    }  // END for (ptrdiff_t start_i = 0; start_i < delta_i; start_i++)

    cudaMemcpyAsync(data, data_device_ptr, sizeof(real_t) * n[0] * n[1], cudaMemcpyDeviceToHost, stream_data);

    /* ── Free device resources ── */
    cudaFreeAsync(tri3_intersect_z_device, stream_intersect);
    free_mesh_tri3_geom_device(&geom_device, stream_geom);

    cudaStream_t stream_free;
    cudaStreamCreate(&stream_free);
    free_cell_list_split_3d_1d_map_device(&split_map_device, stream_free);
    cudaStreamSynchronize(stream_free);
    cudaStreamDestroy(stream_free);

    cudaStreamSynchronize(stream_intersect);
    cudaStreamDestroy(stream_intersect);

    cudaStreamSynchronize(stream_geom);
    cudaStreamDestroy(stream_geom);

    cudaStreamSynchronize(stream_data);
    cudaFreeAsync(data_device_ptr, stream_data);
    cudaStreamDestroy(stream_data);

    cudaStreamSynchronize(stream_kernel);
    cudaStreamDestroy(stream_kernel);

    RETURN_FROM_FUNCTION(ret);
}  // END Function: tri3_raster_cell_quad_gpu_launch