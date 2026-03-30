#ifndef RESAMPLE_FIELD_ADJOINT_CELL_CUDA_SH_CUH
#define RESAMPLE_FIELD_ADJOINT_CELL_CUDA_SH_CUH

#include <cuda_runtime.h>

#include "cell_list_cuda.cuh"
#include "cell_list_query_cuda.cuh"
#include "cubature_cuda.cuh"
#include "sfem_gpu_math.cuh"
#include "sfem_resample_field_cuda_fun.cuh"

__device__ __forceinline__ void                          //
compress_tet_indices_gpu(int *__restrict__ tet_indices,  // Input array of tet indices (size n2)
                         real_t   *z_coords,             // Input array of z-coordinates (size n2)
                         const int size) {               // Size of the input array (n2)

    int write_index = 0;

    for (int read_index = 0; read_index < size; read_index++) {
        if (tet_indices[read_index] != -1) {
            tet_indices[write_index] = tet_indices[read_index];
            tet_indices[read_index]  = -1;
            z_coords[write_index]    = z_coords[read_index];
            write_index++;
        }
    }
}

__device__ __forceinline__ void                              //
compress_tet_indices_par_gpu(int *__restrict__ tet_indices,  //
                             real_t *__restrict__ z_coords,  //
                             int *__restrict__ compact_tet_indices, real_t *__restrict__ compact_z_coords,
                             int      *shared_write_pos,  //
                             const int thread_id,         //
                             const int size) {            //
                                                          //
    // Each thread handles its own segment or uses shared memory
    // This is a simplified version - typically you'd use a parallel prefix sum approach

    // extern __shared__ int shared_write_pos[];
    // int tid = threadIdx.x;

    if (thread_id == 0) shared_write_pos[0] = 0;

    __syncthreads();

    if (thread_id < size) {
        compact_tet_indices[thread_id] = -1;
    }  // END if (thread_id < size)

    __syncthreads();

    if (thread_id < size && tet_indices[thread_id] != -1) {
        const int write_pos            = atomicAdd(&shared_write_pos[0], 1);
        compact_tet_indices[write_pos] = tet_indices[thread_id];
        compact_z_coords[write_pos]    = z_coords[thread_id];
    }  // END if (thread_id < size && tet_indices[thread_id] != -1)

    __syncthreads();

    if (thread_id < size) {
        tet_indices[thread_id] = compact_tet_indices[thread_id];
        if (compact_tet_indices[thread_id] != -1) {
            z_coords[thread_id] = compact_z_coords[thread_id];
        }
    }  // END if (thread_id < size)
}

///////////////////////////////////////////////
// update_hex_field_shm_il
// Accelerated version:
//  - No compression step (no __syncthreads in inner loop)
//  - 2D cell lookup hoisted outside the z-iteration loop
//  - register variables instead of shared memory for per-thread data
///////////////////////////////////////////////
template <typename index_t>
__device__ __forceinline__ int                                             //
update_hex_field_shm_il(const cell_list_split_3d_2d_map_t *split_map,      // Cell list split map data structure
                        const boxes_interleaved_t         *boxes,          // Interleaved boxes data structure
                        const mesh_tet_geom_device_t      *mesh_geom,      // Mesh geometry data structure
                        const index_t                      i_grid,         // The i index of the grid point in the hex mesh
                        const index_t                      j_grid,         // The j index of the grid point in the hex mesh
                        const elems_tet4_device *const __restrict__ mesh,  // Mesh: mesh_t struct
                        const index_t n0,                                  // SDF: n[3]
                        const index_t n1,                                  //
                        const index_t n2,                                  //
                        const index_t stride0,                             // SDF: stride[3]
                        const index_t stride1,                             //
                        const index_t stride2,                             //
                        const geom_t  origin0,                             // SDF: origin[3]
                        const geom_t  origin1,                             //
                        const geom_t  origin2,                             //
                        const geom_t  delta0,                              // SDF: delta[3]
                        const geom_t  delta1,                              //
                        const geom_t  delta2,                              //
                        const real_t *const __restrict__ weighted_field,   // Weighted field
                        real_t *const __restrict__ hex_field) {            //

    (void)n0;
    (void)n1;

    const int thread_id = threadIdx.x +                                           //
                          blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);  //

    const int threads_per_block = blockDim.x * blockDim.y * blockDim.z;

    const real_t grid_x     = real_t(origin0) + real_t(i_grid) * real_t(delta0);
    const real_t grid_y     = real_t(origin1) + real_t(j_grid) * real_t(delta1);
    const real_t delta_z    = delta2;
    const real_t inv_delta0 = real_t(1) / delta0;
    const real_t inv_delta1 = real_t(1) / delta1;
    const real_t inv_delta2 = real_t(1) / delta2;

    const real_t *quad_x = QuadPoints<real_t>::x();
    const real_t *quad_y = QuadPoints<real_t>::y();
    const real_t *quad_z = QuadPoints<real_t>::z();
    const real_t *quad_w = QuadPoints<real_t>::w();

    const cell_list_3d_2d_map_t *map_lower = split_map->map_lower;
    const cell_list_3d_2d_map_t *map_upper = split_map->map_upper;

    for (int q_ijk = 0; q_ijk < QUAD_TOTAL; q_ijk++) {
        const real_t q_x = quad_x[q_ijk];
        const real_t q_y = quad_y[q_ijk];
        const real_t q_z = quad_z[q_ijk];
        const real_t q_w = quad_w[q_ijk];

        const real_t x_q         = grid_x + q_x * delta0;
        const real_t y_q         = grid_y + q_y * delta1;
        const real_t phys_z_base = origin2 + q_z * delta2;

        int lower_cell_start, lower_cell_end;
        get_cell_range_2d_gpu(map_lower, x_q, y_q, &lower_cell_start, &lower_cell_end);

        int upper_cell_start, upper_cell_end;
        get_cell_range_2d_gpu(map_upper, x_q, y_q, &upper_cell_start, &upper_cell_end);

        for (int block_k = 0; block_k < n2; block_k += threads_per_block) {
            const int global_k = block_k + thread_id;

            if (global_k < n2) {
                const real_t z_local = phys_z_base + (real_t)global_k * delta_z;

                // Query lower map first using precomputed 2D cell range
                int tet_idx = query_cell_list_z_given_range_il_gpu(map_lower,         //
                                                                   boxes,             //
                                                                   mesh_geom,         //
                                                                   lower_cell_start,  //
                                                                   lower_cell_end,    //
                                                                   x_q,
                                                                   y_q,
                                                                   z_local);

                // Fall through to upper map if not found
                if (tet_idx == -1) {
                    tet_idx = query_cell_list_z_given_range_il_gpu(map_upper,         //
                                                                   boxes,             //
                                                                   mesh_geom,         //
                                                                   upper_cell_start,  //
                                                                   upper_cell_end,    //
                                                                   x_q,
                                                                   y_q,
                                                                   z_local);
                }

                if (tet_idx > -1) {
                    update_hex_quad_node_cuda<index_t>(x_q,                            //
                                                       y_q,                            //
                                                       z_local,                        //
                                                       q_w,                            //
                                                       static_cast<index_t>(tet_idx),  //
                                                       mesh,                           //
                                                       mesh_geom,                      //
                                                       stride0,
                                                       stride1,
                                                       stride2,  //
                                                       origin0,
                                                       origin1,
                                                       origin2,  //
                                                       inv_delta0,
                                                       inv_delta1,
                                                       inv_delta2,      //
                                                       weighted_field,  //
                                                       hex_field);      //
                }  // END if (tet_idx > -1)
            }  // END if (global_k < n2)
        }  // END for (int block_k = 0; block_k < n2; block_k += threads_per_block)
    }  // END for (int q_ijk = 0; q_ijk < QUAD_TOTAL; q_ijk++)

    return 0;
}

////////////////////////////////////////////////////
// transfer_to_hex_field_cell_split_tet4_shm_il_kernel
////////////////////////////////////////////////////
template <typename index_t = int>
__global__ void                                       //
transfer_to_hex_field_cell_split_tet4_shm_il_kernel(  //
        const cell_list_split_3d_2d_map_t split_map,  // Cell list split map data structure
        const boxes_interleaved_t         boxes,      // Interleaved boxes data structure
        const mesh_tet_geom_device_t      mesh_geom,  // Mesh geometry data structure
        const elems_tet4_device           mesh,       // Mesh: mesh_t struct
        const index_t                     start_i,    // Starting i index for the grid points in the hex mesh
        const index_t                     start_j,    // Starting j index for the grid points in the hex mesh
        const index_t                     delta_i,    // Cell list jump in x direction.
        const index_t                     delta_j,    // Cell list jump in y direction.
        const index_t                     size_i,     // Number of grid points in x direction
        const index_t                     size_j,     // Number of grid points in y direction
        const index_t                     n0,         // SDF: n[3]
        const index_t                     n1,         //
        const index_t                     n2,         //
        const index_t                     stride0,    // SDF: stride[3]
        const index_t                     stride1,    //
        const index_t                     stride2,    //
        const geom_t                      origin0,    // SDF: origin[3]
        const geom_t                      origin1,    //
        const geom_t                      origin2,    //
        const geom_t                      delta0,     // SDF: delta[3]
        const geom_t                      delta1,     //
        const geom_t                      delta2,     //
        const real_t *const __restrict__ weighted_field, real_t *const __restrict__ hex_field) {
    const index_t i_grid = start_i + static_cast<index_t>(blockIdx.x) * delta_i;
    const index_t j_grid = start_j + static_cast<index_t>(blockIdx.y) * delta_j;

    if (i_grid >= size_i - 1 || j_grid >= size_j - 1) {
        return;  // Out of bounds, exit the kernel
    }

    update_hex_field_shm_il<index_t>(&split_map,
                                     &boxes,
                                     &mesh_geom,
                                     i_grid,
                                     j_grid,
                                     &mesh,
                                     n0,
                                     n1,
                                     n2,
                                     stride0,
                                     stride1,
                                     stride2,
                                     origin0,
                                     origin1,
                                     origin2,
                                     delta0,
                                     delta1,
                                     delta2,
                                     weighted_field,
                                     hex_field);
}

#endif  // RESAMPLE_FIELD_ADJOINT_CELL_CUDA_SH_CUH