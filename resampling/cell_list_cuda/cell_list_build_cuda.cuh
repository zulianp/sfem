#ifndef __CELL_LIST_BUILD_CUDA_CUH__
#define __CELL_LIST_BUILD_CUDA_CUH__

#include <cuda_runtime.h>

#include "cell_list_3d_map.h"
#include "sfem_gpu_math.cuh"

/* ─────────────────────────────────────────────────────────────────────────────
 * Helper: compact grid-parameter struct (no device pointers → safe to pass by
 * value to __global__ kernels).
 * ───────────────────────────────────────────────────────────────────────────── */
typedef struct {
    real_t min_x;
    real_t min_y;
    real_t inv_delta_x;
    real_t inv_delta_y;
    int    num_cells_x;
    int    num_cells_y;
} cell_list_grid_params_gpu_t;

/* ─────────────────────────────────────────────────────────────────────────────
 * atomicMax helpers for real_t (float and double overloads).
 * CUDA provides atomicMax only for integer types; the CAS-loop below handles
 * floating-point values correctly for non-negative inputs (which box sizes are).
 * ───────────────────────────────────────────────────────────────────────────── */
__device__ __forceinline__ void  //
atomicMax_real_gpu(float *addr, float val) {
    int *addr_as_int = (int *)addr;
    int  old         = *addr_as_int;
    int  assumed;
    do {
        assumed = old;
        old     = atomicCAS(addr_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__device__ __forceinline__ void  //
atomicMax_real_gpu(double *addr, double val) {
    unsigned long long *addr_as_ull = (unsigned long long *)addr;
    unsigned long long  old         = *addr_as_ull;
    unsigned long long  assumed;
    do {
        assumed = old;
        old     = atomicCAS(addr_as_ull, assumed, __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Device helper: for a box [bmin_x,bmax_x] × [bmin_y,bmax_y], compute the
 * range of 2-D cell indices it overlaps in the given grid.
 * ───────────────────────────────────────────────────────────────────────────── */
__device__ __forceinline__ void                                          //
compute_cell_range_for_box_gpu(const real_t bmin_x,                      //
                               const real_t bmax_x,                      //
                               const real_t bmin_y,                      //
                               const real_t bmax_y,                      //
                               const cell_list_grid_params_gpu_t params,  //
                               int *ix_min,                               //
                               int *ix_max,                               //
                               int *iy_min,                               //
                               int *iy_max) {                             //

    /* Empty grid: return a range where min > max so nested loops never execute */
    if (params.num_cells_x <= 0 || params.num_cells_y <= 0) {
        *ix_min = 0;  *ix_max = -1;
        *iy_min = 0;  *iy_max = -1;
        return;
    }

    *ix_min = (int)((bmin_x - params.min_x) * params.inv_delta_x);
    *iy_min = (int)((bmin_y - params.min_y) * params.inv_delta_y);
    *ix_max = (int)((bmax_x - params.min_x) * params.inv_delta_x);
    *iy_max = (int)((bmax_y - params.min_y) * params.inv_delta_y);

    *ix_min = (*ix_min < 0) ? 0 : ((*ix_min >= params.num_cells_x) ? params.num_cells_x - 1 : *ix_min);
    *iy_min = (*iy_min < 0) ? 0 : ((*iy_min >= params.num_cells_y) ? params.num_cells_y - 1 : *iy_min);
    *ix_max = (*ix_max < 0) ? 0 : ((*ix_max >= params.num_cells_x) ? params.num_cells_x - 1 : *ix_max);
    *iy_max = (*iy_max < 0) ? 0 : ((*iy_max >= params.num_cells_y) ? params.num_cells_y - 1 : *iy_max);
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Phase 1 – Reduce max bounding-box side lengths per split group.
 *
 * Each thread processes one box.  Boxes with (dx < split_x && dy < split_y)
 * belong to the "lower" group; the rest to the "upper" group.
 *
 * d_max_lower[3] and d_max_upper[3] must be pre-zeroed (cudaMemset) before
 * this kernel is launched.
 *   [0] = max delta_x
 *   [1] = max delta_y
 *   [2] = max delta_z
 * ───────────────────────────────────────────────────────────────────────────── */
__global__ void                                                       //
reduce_max_delta_split_kernel(const real_t *__restrict__ box_min_x,   //
                              const real_t *__restrict__ box_min_y,   //
                              const real_t *__restrict__ box_min_z,   //
                              const real_t *__restrict__ box_max_x,   //
                              const real_t *__restrict__ box_max_y,   //
                              const real_t *__restrict__ box_max_z,   //
                              const int                  num_boxes,   //
                              const real_t               split_x,     //
                              const real_t               split_y,     //
                              real_t *__restrict__        d_max_lower,  // [3]
                              real_t *__restrict__        d_max_upper)  // [3]
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    const real_t dx = fabs(box_max_x[idx] - box_min_x[idx]);
    const real_t dy = fabs(box_max_y[idx] - box_min_y[idx]);
    const real_t dz = fabs(box_max_z[idx] - box_min_z[idx]);

    if (dx < split_x && dy < split_y) {
        atomicMax_real_gpu(&d_max_lower[0], dx);
        atomicMax_real_gpu(&d_max_lower[1], dy);
        atomicMax_real_gpu(&d_max_lower[2], dz);
    } else {
        atomicMax_real_gpu(&d_max_upper[0], dx);
        atomicMax_real_gpu(&d_max_upper[1], dy);
        atomicMax_real_gpu(&d_max_upper[2], dz);
    }
}  // END Kernel: reduce_max_delta_split_kernel

/* ─────────────────────────────────────────────────────────────────────────────
 * Phase 2 – Count how many dict entries each 2-D cell accumulates.
 *
 * Each thread processes one box and atomically increments cell_ptr[cell_idx+1]
 * for every (ix, iy) cell the box overlaps.  Both the lower and upper maps are
 * handled in a single pass.
 *
 * cell_ptr_lower and cell_ptr_upper must be pre-zeroed.
 * ───────────────────────────────────────────────────────────────────────────── */
__global__ void                                                          //
count_cell_overlaps_split_kernel(const real_t *__restrict__ box_min_x,   //
                                 const real_t *__restrict__ box_min_y,   //
                                 const real_t *__restrict__ box_max_x,   //
                                 const real_t *__restrict__ box_max_y,   //
                                 const int                  num_boxes,   //
                                 const real_t               split_x,     //
                                 const real_t               split_y,     //
                                 cell_list_grid_params_gpu_t lower,       //
                                 cell_list_grid_params_gpu_t upper,       //
                                 int *__restrict__           cell_ptr_lower,  //
                                 int *__restrict__           cell_ptr_upper)  //
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    const real_t bmin_x = box_min_x[idx];
    const real_t bmin_y = box_min_y[idx];
    const real_t bmax_x = box_max_x[idx];
    const real_t bmax_y = box_max_y[idx];

    const real_t dx = fabs(bmax_x - bmin_x);
    const real_t dy = fabs(bmax_y - bmin_y);

    int                         ix_min, ix_max, iy_min, iy_max;
    int                        *cell_ptr;
    cell_list_grid_params_gpu_t params;

    if (dx < split_x && dy < split_y) {
        params    = lower;
        cell_ptr  = cell_ptr_lower;
    } else {
        params    = upper;
        cell_ptr  = cell_ptr_upper;
    }

    compute_cell_range_for_box_gpu(bmin_x, bmax_x, bmin_y, bmax_y, params, &ix_min, &ix_max, &iy_min, &iy_max);

    for (int iy = iy_min; iy <= iy_max; iy++) {
        for (int ix = ix_min; ix <= ix_max; ix++) {
            const int cell_idx = ix + iy * params.num_cells_x;
            atomicAdd(&cell_ptr[cell_idx + 1], 1);
        }
    }
}  // END Kernel: count_cell_overlaps_split_kernel

/* ─────────────────────────────────────────────────────────────────────────────
 * Phase 3 – In-place inclusive prefix sum on an integer array.
 *
 * Designed for small arrays (total_num_2d_cells + 1, typically << 1e6).
 * Single-thread serial scan: O(n) time, trivially correct.
 *
 * After the count kernel arr[1..n-1] holds per-cell counts and arr[0] == 0.
 * After this kernel arr[i] = sum of counts for cells 0..i-1  (CSR pointer).
 * ───────────────────────────────────────────────────────────────────────────── */
__global__ void                                          //
prefix_sum_inplace_kernel(int *__restrict__ arr,          //
                          const int          n) {         //
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    for (int i = 1; i < n; i++) {
        arr[i] += arr[i - 1];
    }
}  // END Kernel: prefix_sum_inplace_kernel

/* ─────────────────────────────────────────────────────────────────────────────
 * Phase 4 – Fill cell dictionaries.
 *
 * Each thread processes one box and, for every overlapping (ix, iy) cell,
 * atomically claims a unique slot (via atomicAdd on temp_count) and writes:
 *   cell_dict[slot]      = box index
 *   lower_bounds_z[slot] = box_min_z
 *   upper_bounds_z[slot] = box_max_z
 *
 * temp_count_lower / temp_count_upper must be pre-zeroed (same size as
 * total_num_2d_cells for each map).
 * ───────────────────────────────────────────────────────────────────────────── */
__global__ void                                                         //
fill_cell_dict_split_kernel(const real_t *__restrict__ box_min_x,       //
                            const real_t *__restrict__ box_min_y,       //
                            const real_t *__restrict__ box_min_z,       //
                            const real_t *__restrict__ box_max_x,       //
                            const real_t *__restrict__ box_max_y,       //
                            const real_t *__restrict__ box_max_z,       //
                            const int                  num_boxes,       //
                            const real_t               split_x,         //
                            const real_t               split_y,         //
                            cell_list_grid_params_gpu_t lower,           //
                            cell_list_grid_params_gpu_t upper,           //
                            /* lower map output arrays */
                            const int *__restrict__ cell_ptr_lower,      //
                            int *__restrict__        cell_dict_lower,     //
                            real_t *__restrict__     lower_z_lower,       //
                            real_t *__restrict__     upper_z_lower,       //
                            int *__restrict__        temp_count_lower,    //
                            /* upper map output arrays */
                            const int *__restrict__ cell_ptr_upper,      //
                            int *__restrict__        cell_dict_upper,     //
                            real_t *__restrict__     lower_z_upper,       //
                            real_t *__restrict__     upper_z_upper,       //
                            int *__restrict__        temp_count_upper)    //
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    const real_t bmin_x = box_min_x[idx];
    const real_t bmin_y = box_min_y[idx];
    const real_t bmin_z = box_min_z[idx];
    const real_t bmax_x = box_max_x[idx];
    const real_t bmax_y = box_max_y[idx];
    const real_t bmax_z = box_max_z[idx];

    const real_t dx = fabs(bmax_x - bmin_x);
    const real_t dy = fabs(bmax_y - bmin_y);

    int                          ix_min, ix_max, iy_min, iy_max;
    cell_list_grid_params_gpu_t  params;
    const int *cell_ptr;
    int       *cell_dict;
    real_t    *lower_z;
    real_t    *upper_z;
    int       *temp_count;

    if (dx < split_x && dy < split_y) {
        params     = lower;
        cell_ptr   = cell_ptr_lower;
        cell_dict  = cell_dict_lower;
        lower_z    = lower_z_lower;
        upper_z    = upper_z_lower;
        temp_count = temp_count_lower;
    } else {
        params     = upper;
        cell_ptr   = cell_ptr_upper;
        cell_dict  = cell_dict_upper;
        lower_z    = lower_z_upper;
        upper_z    = upper_z_upper;
        temp_count = temp_count_upper;
    }

    compute_cell_range_for_box_gpu(bmin_x, bmax_x, bmin_y, bmax_y, params, &ix_min, &ix_max, &iy_min, &iy_max);

    for (int iy = iy_min; iy <= iy_max; iy++) {
        for (int ix = ix_min; ix <= ix_max; ix++) {
            const int cell_idx = ix + iy * params.num_cells_x;
            const int slot     = cell_ptr[cell_idx] + atomicAdd(&temp_count[cell_idx], 1);

            cell_dict[slot] = idx;
            lower_z[slot]   = bmin_z;
            upper_z[slot]   = bmax_z;
        }
    }
}  // END Kernel: fill_cell_dict_split_kernel

/* ─────────────────────────────────────────────────────────────────────────────
 * Phase 5 – Sort each 2-D cell's dict entries by lower_bounds_z (ascending).
 *
 * One GPU thread per 2-D cell.  Performs in-place insertion sort, which is
 * efficient for the small per-cell sizes that the split strategy produces.
 * All three parallel arrays (cell_dict, lower_bounds_z, upper_bounds_z) are
 * permuted together to maintain consistency.
 * ───────────────────────────────────────────────────────────────────────────── */
__global__ void                                                //
sort_cells_by_lower_z_kernel(const int *__restrict__ cell_ptr,  //
                             int *__restrict__        cell_dict,  //
                             real_t *__restrict__     lower_bounds_z,  //
                             real_t *__restrict__     upper_bounds_z,  //
                             const int                total_num_2d_cells) {  //

    const int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= total_num_2d_cells) return;

    const int start = cell_ptr[cell_idx];
    const int end   = cell_ptr[cell_idx + 1];
    const int size  = end - start;

    if (size <= 1) return;

    /* Insertion sort – O(size^2) but size is small for well-tuned split params */
    for (int i = start + 1; i < end; i++) {
        const int    key_dict  = cell_dict[i];
        const real_t key_lower = lower_bounds_z[i];
        const real_t key_upper = upper_bounds_z[i];

        int j = i - 1;
        while (j >= start && lower_bounds_z[j] > key_lower) {
            cell_dict[j + 1]      = cell_dict[j];
            lower_bounds_z[j + 1] = lower_bounds_z[j];
            upper_bounds_z[j + 1] = upper_bounds_z[j];
            j--;
        }

        cell_dict[j + 1]      = key_dict;
        lower_bounds_z[j + 1] = key_lower;
        upper_bounds_z[j + 1] = key_upper;
    }
}  // END Kernel: sort_cells_by_lower_z_kernel

/* ─────────────────────────────────────────────────────────────────────────────
 * Phase 6 – Ensure upper_bounds_z is non-decreasing within each 2-D cell.
 *
 * Required by the binary-search query logic.  Called after the sort.
 * One GPU thread per 2-D cell; serial scan within the cell.
 * ───────────────────────────────────────────────────────────────────────────── */
__global__ void                                                    //
fix_upper_bounds_z_kernel(const int *__restrict__ cell_ptr,         //
                          real_t *__restrict__     upper_bounds_z,  //
                          const int                total_num_2d_cells) {  //

    const int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= total_num_2d_cells) return;

    const int start = cell_ptr[cell_idx];
    const int end   = cell_ptr[cell_idx + 1];

    if (end - start <= 1) return;

    for (int i = start + 1; i < end; i++) {
        if (upper_bounds_z[i - 1] > upper_bounds_z[i]) {
            upper_bounds_z[i] = upper_bounds_z[i - 1];
        }
    }
}  // END Kernel: fix_upper_bounds_z_kernel

/* ─────────────────────────────────────────────────────────────────────────────
 * Host API
 *
 * build_cell_list_split_map_on_device
 *   Builds a cell_list_split_3d_2d_map_t entirely on the GPU.
 *   All input arrays (d_box_min_x, …) must reside in device memory.
 *   Returns a host struct whose map_lower / map_upper members are device
 *   pointers, exactly matching the layout produced by
 *   copy_cell_list_split_3d_2d_map_to_device().
 *   The caller is responsible for freeing the result with
 *   free_cell_list_split_3d_2d_map_device().
 *
 * NOTE: This function performs several cudaStreamSynchronize() calls
 *       internally (needed to read back scalar values from device).
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
                                    cudaStream_t  stream);                   //

#endif  // __CELL_LIST_BUILD_CUDA_CUH__
