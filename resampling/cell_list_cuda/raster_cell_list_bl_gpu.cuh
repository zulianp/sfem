#ifndef __RASTER_CELL_LIST_BL_GPU_CUH__
#define __RASTER_CELL_LIST_BL_GPU_CUH__

#include <cuda_runtime.h>
#include <cstdlib>

#include "cell_list_3d_1d_map_sur_mesh.h"
#include "cell_list_3d_map.h"
#include "cell_list_cuda.cuh"
#include "cell_list_query_cuda.cuh"
#include "cubature_cuda.cuh"
#include "raster_cell_list_gpu.cuh"
#include "sfem_gpu_math.cuh"
#include "sfem_resample_field_cuda_fun.cuh"

// sfem_base.h (via sfem_resample_field_cuda_fun.cuh) must precede cell_list_3d_1d_map.h
// so that real_t is defined when the struct body is parsed.
#include "cell_list_3d_1d_map.h"

////////////////////////////////////////////////////
// raster_to_hex_field_bl_tri3_kernel  (one thread per (i,j) column)
//
// Each thread owns a single (i,j) column: it runs the cell-list query,
// sorts the z-intersections, and fills every k-cell whose center lies
// inside a [z_lower, z_upper] span (solid volume fill, same semantics as
// the block-per-column kernel in raster_cell_list_gpu.cuh).
//
// The z-intersection scratch is a per-thread local array (MAX_INTERSECT
// compile-time capacity). Local memory is interleaved per-thread, so
// warp-uniform accesses are coalesced and bank-conflict free, and no
// dynamic shared memory is needed — occupancy is no longer shm-bound.
//
// Launch requirements:
//   - 1D grid of at least (size_i * size_j) threads, zero dynamic shm.
////////////////////////////////////////////////////
template <typename index_t = int, int MAX_INTERSECT = 64>
__global__ void                                       //
raster_to_hex_field_bl_tri3_kernel(                   //
        const cell_list_split_3d_1d_map_t split_map,  //
        const mesh_tri3_geom_device_t     mesh_geom,  //
        const index_t                     size_i,     //
        const index_t                     size_j,     //
        const index_t                     size_k,     //
        const geom_t                      origin0,    //
        const geom_t                      origin1,    //
        const geom_t                      origin2,    //
        const geom_t                      delta0,     //
        const geom_t                      delta1,     //
        const geom_t                      delta2,     //
        real_t *const __restrict__ data) {            //

    // Per-thread scratch in local memory (interleaved per-thread by the
    // hardware: conflict-free, L1-cached, costs no shared memory).
    real_t z_scratch[MAX_INTERSECT];

    const index_t id = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= size_i * size_j) return;  // total-column bounds check (no wraparound)

    const index_t i_grid = id % size_i;
    const index_t j_grid = id / size_i;

    const real_t x = origin0 + i_grid * delta0;
    const real_t y = origin1 + j_grid * delta1;

    const int n =                                                                  //
            query_cell_list_3d_1d_split_map_mesh_given_xy_tri3_gpu(&split_map,     //
                                                                   &mesh_geom,     //
                                                                   x,              //
                                                                   y,              //
                                                                   MAX_INTERSECT,  //
                                                                   z_scratch);     //

    if (n <= 0) return;
    if (n % 2 != 0) {
        printf("Warning: Odd number of triangle intersections at (%d, %d): %d found.\n",
               static_cast<int>(i_grid),
               static_cast<int>(j_grid),
               n);
        return;
    }

    sort_real_array_ascending_gpu(z_scratch, n);

    // Fill every k-cell whose center lies within a [z0, z1] span.
    for (int p = 0; p + 1 < n; p += 2) {
        const real_t z0 = z_scratch[p];
        const real_t z1 = z_scratch[p + 1];

        // k whose center origin2 + k*delta2 satisfies z0 <= center <= z1.
        // The ceil/floor range is widened by one on each side and each candidate is
        // re-tested with the same geom_t expression as the block-per-column kernel,
        // so boundary cells match it exactly.
        index_t k_lo = static_cast<index_t>(ceil((z0 - origin2) / delta2)) - 1;
        index_t k_hi = static_cast<index_t>(floor((z1 - origin2) / delta2)) + 1;

        if (k_lo < 0) k_lo = 0;
        if (k_hi >= size_k) k_hi = size_k - 1;

        for (index_t k = k_lo; k <= k_hi; ++k) {
            const geom_t z = origin2 + k * delta2;
            if (z >= z0 && z <= z1) {
                const index_t id_hex = i_grid + j_grid * size_i + k * size_i * size_j;
                data[id_hex]         = static_cast<real_t>(1);  // inside the solid
            }
        }
    }

}  // END Kernel: raster_to_hex_field_bl_tri3_kernel

#endif  // __RASTER_CELL_LIST_BL_GPU_CUH__