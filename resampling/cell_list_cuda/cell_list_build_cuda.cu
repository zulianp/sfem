#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cell_list_build_cuda.cuh"

/* ─────────────────────────────────────────────────────────────────────────────
 * Internal helper
 * ───────────────────────────────────────────────────────────────────────────── */
static inline int ceildiv(int a, int b) { return (a + b - 1) / b; }

/* ─────────────────────────────────────────────────────────────────────────────
 * build_cell_list_split_map_on_device
 *
 * Builds a cell_list_split_3d_2d_map_t entirely on the GPU in six phases:
 *
 *   1. Reduce max bounding-box side lengths per split group   (1 kernel)
 *   2. Compute grid dimensions on host
 *   3. Count overlaps for both maps in one pass               (1 kernel)
 *   4. Prefix-sum each cell_ptr array                         (2 kernels)
 *   5. Fill cell dictionaries for both maps in one pass       (1 kernel)
 *   6. Sort + fix upper-bounds for each map                   (4 kernels)
 *
 * The returned struct has the same device-memory layout as the one produced by
 * copy_cell_list_split_3d_2d_map_to_device() and can be freed with
 * free_cell_list_split_3d_2d_map_device().
 * ───────────────────────────────────────────────────────────────────────────── */
cell_list_split_3d_2d_map_t                                                 //
build_cell_list_split_map_on_device(const real_t *d_box_min_x,              //
                                    const real_t *d_box_min_y,              //
                                    const real_t *d_box_min_z,              //
                                    const real_t *d_box_max_x,              //
                                    const real_t *d_box_max_y,              //
                                    const real_t *d_box_max_z,              //
                                    const int     num_boxes,                //
                                    const real_t  split_x,                  //
                                    const real_t  split_y,                  //
                                    const real_t  x_min,                    //
                                    const real_t  x_max,                    //
                                    const real_t  y_min,                    //
                                    const real_t  y_max,                    //
                                    const real_t  z_min,                    //
                                    const real_t  z_max,                    //
                                    cudaStream_t  stream)                    //
{
    cell_list_split_3d_2d_map_t d_split;
    d_split.split_x   = split_x;
    d_split.split_y   = split_y;
    d_split.map_lower = NULL;
    d_split.map_upper = NULL;

    const int block_size = 256;
    const int grid_boxes = ceildiv(num_boxes, block_size);

    /* ══════════════════════════════════════════════════════════════════════════
     * Phase 1 – Reduce max deltas per split group
     * ══════════════════════════════════════════════════════════════════════════ */

    real_t *d_max_lower = NULL;  /* [3]: max dx, dy, dz  — lower group */
    real_t *d_max_upper = NULL;  /* [3]: max dx, dy, dz  — upper group */

    cudaMallocAsync((void **)&d_max_lower, 3 * sizeof(real_t), stream);
    cudaMallocAsync((void **)&d_max_upper, 3 * sizeof(real_t), stream);
    cudaMemsetAsync(d_max_lower, 0, 3 * sizeof(real_t), stream);
    cudaMemsetAsync(d_max_upper, 0, 3 * sizeof(real_t), stream);
    cudaStreamSynchronize(stream);

    if (grid_boxes > 0) {
        reduce_max_delta_split_kernel<<<grid_boxes, block_size, 0, stream>>>(
                d_box_min_x, d_box_min_y, d_box_min_z,
                d_box_max_x, d_box_max_y, d_box_max_z,
                num_boxes, split_x, split_y,
                d_max_lower, d_max_upper);
    }

    real_t h_max_lower[3] = {(real_t)0, (real_t)0, (real_t)0};
    real_t h_max_upper[3] = {(real_t)0, (real_t)0, (real_t)0};

    cudaMemcpyAsync(h_max_lower, d_max_lower, 3 * sizeof(real_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_max_upper, d_max_upper, 3 * sizeof(real_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFreeAsync(d_max_lower, stream);
    cudaFreeAsync(d_max_upper, stream);

    /* ══════════════════════════════════════════════════════════════════════════
     * Phase 2 – Compute grid dimensions on host
     * ══════════════════════════════════════════════════════════════════════════ */

    const real_t inv_lx = (h_max_lower[0] > (real_t)0) ? (real_t)1.0 / h_max_lower[0] : (real_t)0;
    const real_t inv_ly = (h_max_lower[1] > (real_t)0) ? (real_t)1.0 / h_max_lower[1] : (real_t)0;
    const real_t inv_ux = (h_max_upper[0] > (real_t)0) ? (real_t)1.0 / h_max_upper[0] : (real_t)0;
    const real_t inv_uy = (h_max_upper[1] > (real_t)0) ? (real_t)1.0 / h_max_upper[1] : (real_t)0;

    const int ncx_lower = (h_max_lower[0] > (real_t)0) ? (int)ceil((double)(x_max - x_min) * inv_lx) : 0;
    const int ncy_lower = (h_max_lower[1] > (real_t)0) ? (int)ceil((double)(y_max - y_min) * inv_ly) : 0;
    const int ncx_upper = (h_max_upper[0] > (real_t)0) ? (int)ceil((double)(x_max - x_min) * inv_ux) : 0;
    const int ncy_upper = (h_max_upper[1] > (real_t)0) ? (int)ceil((double)(y_max - y_min) * inv_uy) : 0;

    const int total_2d_lower = ncx_lower * ncy_lower;
    const int total_2d_upper = ncx_upper * ncy_upper;

    const cell_list_grid_params_gpu_t lower_params = {x_min, y_min, inv_lx, inv_ly, ncx_lower, ncy_lower};
    const cell_list_grid_params_gpu_t upper_params = {x_min, y_min, inv_ux, inv_uy, ncx_upper, ncy_upper};

    printf("[build_cell_list_split_map_on_device] lower grid: %d x %d  upper grid: %d x %d\n",
           ncx_lower, ncy_lower, ncx_upper, ncy_upper);

    /* ══════════════════════════════════════════════════════════════════════════
     * Phase 3 – Count overlaps (both maps in one kernel pass)
     * ══════════════════════════════════════════════════════════════════════════ */

    int *d_cell_ptr_lower = NULL;
    int *d_cell_ptr_upper = NULL;

    cudaMallocAsync((void **)&d_cell_ptr_lower, (size_t)(total_2d_lower + 1) * sizeof(int), stream);
    cudaMallocAsync((void **)&d_cell_ptr_upper, (size_t)(total_2d_upper + 1) * sizeof(int), stream);
    cudaMemsetAsync(d_cell_ptr_lower, 0, (size_t)(total_2d_lower + 1) * sizeof(int), stream);
    cudaMemsetAsync(d_cell_ptr_upper, 0, (size_t)(total_2d_upper + 1) * sizeof(int), stream);
    cudaStreamSynchronize(stream);

    if (grid_boxes > 0) {
        count_cell_overlaps_split_kernel<<<grid_boxes, block_size, 0, stream>>>(
                d_box_min_x, d_box_min_y, d_box_max_x, d_box_max_y,
                num_boxes, split_x, split_y,
                lower_params, upper_params,
                d_cell_ptr_lower, d_cell_ptr_upper);
    }

    /* ══════════════════════════════════════════════════════════════════════════
     * Phase 4 – Prefix sum: transform per-cell counts into CSR offsets
     * ══════════════════════════════════════════════════════════════════════════ */

    /* n = total_2d + 1; loop runs from i=1 so n=1 (empty map) is fine */
    prefix_sum_inplace_kernel<<<1, 1, 0, stream>>>(d_cell_ptr_lower, total_2d_lower + 1);
    prefix_sum_inplace_kernel<<<1, 1, 0, stream>>>(d_cell_ptr_upper, total_2d_upper + 1);

    /* Read back total dict entries (= cell_ptr[total_2d_cells]) */
    int total_dict_lower = 0;
    int total_dict_upper = 0;
    cudaMemcpyAsync(&total_dict_lower, d_cell_ptr_lower + total_2d_lower, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&total_dict_upper, d_cell_ptr_upper + total_2d_upper, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    printf("[build_cell_list_split_map_on_device] dict entries: lower=%d  upper=%d\n",
           total_dict_lower, total_dict_upper);

    /* ══════════════════════════════════════════════════════════════════════════
     * Phase 5 – Fill cell dictionaries (both maps in one kernel pass)
     * ══════════════════════════════════════════════════════════════════════════ */

    int   *d_cell_dict_lower = NULL;
    real_t *d_lower_z_lower  = NULL;
    real_t *d_upper_z_lower  = NULL;

    int   *d_cell_dict_upper = NULL;
    real_t *d_lower_z_upper  = NULL;
    real_t *d_upper_z_upper  = NULL;

    int *d_temp_count_lower = NULL;
    int *d_temp_count_upper = NULL;

    /* Guard size-0 allocations (CUDA spec allows them but we avoid for clarity) */
    const size_t dict_lower_bytes = (total_dict_lower > 0) ? (size_t)total_dict_lower : 1;
    const size_t dict_upper_bytes = (total_dict_upper > 0) ? (size_t)total_dict_upper : 1;
    const size_t cnt_lower_bytes  = (total_2d_lower   > 0) ? (size_t)total_2d_lower   : 1;
    const size_t cnt_upper_bytes  = (total_2d_upper   > 0) ? (size_t)total_2d_upper   : 1;

    cudaMallocAsync((void **)&d_cell_dict_lower, dict_lower_bytes * sizeof(int), stream);
    cudaMallocAsync((void **)&d_lower_z_lower,   dict_lower_bytes * sizeof(real_t), stream);
    cudaMallocAsync((void **)&d_upper_z_lower,   dict_lower_bytes * sizeof(real_t), stream);

    cudaMallocAsync((void **)&d_cell_dict_upper, dict_upper_bytes * sizeof(int), stream);
    cudaMallocAsync((void **)&d_lower_z_upper,   dict_upper_bytes * sizeof(real_t), stream);
    cudaMallocAsync((void **)&d_upper_z_upper,   dict_upper_bytes * sizeof(real_t), stream);

    cudaMallocAsync((void **)&d_temp_count_lower, cnt_lower_bytes * sizeof(int), stream);
    cudaMallocAsync((void **)&d_temp_count_upper, cnt_upper_bytes * sizeof(int), stream);
    cudaMemsetAsync(d_temp_count_lower, 0, cnt_lower_bytes * sizeof(int), stream);
    cudaMemsetAsync(d_temp_count_upper, 0, cnt_upper_bytes * sizeof(int), stream);
    cudaStreamSynchronize(stream);

    if (grid_boxes > 0) {
        fill_cell_dict_split_kernel<<<grid_boxes, block_size, 0, stream>>>(
                d_box_min_x, d_box_min_y, d_box_min_z,
                d_box_max_x, d_box_max_y, d_box_max_z,
                num_boxes, split_x, split_y,
                lower_params, upper_params,
                /* lower outputs */
                d_cell_ptr_lower, d_cell_dict_lower, d_lower_z_lower, d_upper_z_lower, d_temp_count_lower,
                /* upper outputs */
                d_cell_ptr_upper, d_cell_dict_upper, d_lower_z_upper, d_upper_z_upper, d_temp_count_upper);
    }  /* END if (grid_boxes > 0) */

    cudaStreamSynchronize(stream);

    cudaFreeAsync(d_temp_count_lower, stream);
    cudaFreeAsync(d_temp_count_upper, stream);

    /* ══════════════════════════════════════════════════════════════════════════
     * Phase 6 – Sort entries by lower_z; enforce non-decreasing upper_bounds_z
     * ══════════════════════════════════════════════════════════════════════════ */

    const int grid_lower = (total_2d_lower > 0) ? ceildiv(total_2d_lower, block_size) : 0;
    const int grid_upper = (total_2d_upper > 0) ? ceildiv(total_2d_upper, block_size) : 0;

    if (grid_lower > 0) {
        sort_cells_by_lower_z_kernel<<<grid_lower, block_size, 0, stream>>>(
                d_cell_ptr_lower, d_cell_dict_lower, d_lower_z_lower, d_upper_z_lower, total_2d_lower);

        fix_upper_bounds_z_kernel<<<grid_lower, block_size, 0, stream>>>(
                d_cell_ptr_lower, d_upper_z_lower, total_2d_lower);
    }

    if (grid_upper > 0) {
        sort_cells_by_lower_z_kernel<<<grid_upper, block_size, 0, stream>>>(
                d_cell_ptr_upper, d_cell_dict_upper, d_lower_z_upper, d_upper_z_upper, total_2d_upper);

        fix_upper_bounds_z_kernel<<<grid_upper, block_size, 0, stream>>>(
                d_cell_ptr_upper, d_upper_z_upper, total_2d_upper);
    }

    cudaStreamSynchronize(stream);

    /* ══════════════════════════════════════════════════════════════════════════
     * Assemble result: copy map structs (with device pointers) to device memory
     *
     * This matches the layout produced by copy_cell_list_split_3d_2d_map_to_device()
     * so the result is directly usable with free_cell_list_split_3d_2d_map_device().
     * ══════════════════════════════════════════════════════════════════════════ */

    cell_list_3d_2d_map_t h_map_lower;
    cell_list_3d_2d_map_t h_map_upper;
    memset(&h_map_lower, 0, sizeof(h_map_lower));
    memset(&h_map_upper, 0, sizeof(h_map_upper));

    /* Scalars */
    h_map_lower.total_num_2d_cells     = total_2d_lower;
    h_map_lower.total_num_dict_entries = total_dict_lower;
    h_map_lower.min_x                  = x_min;
    h_map_lower.min_y                  = y_min;
    h_map_lower.min_z                  = z_min;
    h_map_lower.max_x                  = x_max;
    h_map_lower.max_y                  = y_max;
    h_map_lower.max_z                  = z_max;
    h_map_lower.delta_x                = h_max_lower[0];
    h_map_lower.delta_y                = h_max_lower[1];
    h_map_lower.delta_z                = h_max_lower[2];
    h_map_lower.inv_delta_x            = inv_lx;
    h_map_lower.inv_delta_y            = inv_ly;
    h_map_lower.inv_delta_z            = (h_max_lower[2] > (real_t)0) ? (real_t)1.0 / h_max_lower[2] : (real_t)0;
    h_map_lower.num_cells_x            = ncx_lower;
    h_map_lower.num_cells_y            = ncy_lower;
    h_map_lower.num_cells_z            = 0;
    /* Device pointers */
    h_map_lower.cell_ptr       = d_cell_ptr_lower;
    h_map_lower.cell_dict      = d_cell_dict_lower;
    h_map_lower.lower_bounds_z = d_lower_z_lower;
    h_map_lower.upper_bounds_z = d_upper_z_lower;

    h_map_upper.total_num_2d_cells     = total_2d_upper;
    h_map_upper.total_num_dict_entries = total_dict_upper;
    h_map_upper.min_x                  = x_min;
    h_map_upper.min_y                  = y_min;
    h_map_upper.min_z                  = z_min;
    h_map_upper.max_x                  = x_max;
    h_map_upper.max_y                  = y_max;
    h_map_upper.max_z                  = z_max;
    h_map_upper.delta_x                = h_max_upper[0];
    h_map_upper.delta_y                = h_max_upper[1];
    h_map_upper.delta_z                = h_max_upper[2];
    h_map_upper.inv_delta_x            = inv_ux;
    h_map_upper.inv_delta_y            = inv_uy;
    h_map_upper.inv_delta_z            = (h_max_upper[2] > (real_t)0) ? (real_t)1.0 / h_max_upper[2] : (real_t)0;
    h_map_upper.num_cells_x            = ncx_upper;
    h_map_upper.num_cells_y            = ncy_upper;
    h_map_upper.num_cells_z            = 0;
    /* Device pointers */
    h_map_upper.cell_ptr       = d_cell_ptr_upper;
    h_map_upper.cell_dict      = d_cell_dict_upper;
    h_map_upper.lower_bounds_z = d_lower_z_upper;
    h_map_upper.upper_bounds_z = d_upper_z_upper;

    /* Allocate device storage for the map structs themselves */
    cudaMallocAsync((void **)&d_split.map_lower, sizeof(cell_list_3d_2d_map_t), stream);
    cudaMallocAsync((void **)&d_split.map_upper, sizeof(cell_list_3d_2d_map_t), stream);
    cudaStreamSynchronize(stream);

    cudaMemcpyAsync(d_split.map_lower, &h_map_lower, sizeof(cell_list_3d_2d_map_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_split.map_upper, &h_map_upper, sizeof(cell_list_3d_2d_map_t), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    return d_split;
}  // END Function: build_cell_list_split_map_on_device
