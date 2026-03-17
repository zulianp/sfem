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

    // In the kernel call .... set the size of the shared memory to be at least the number of threads in the block, so that
    // each thread can initialize one element of the shared memory array. kernelName<<<gridDim, blockDim, blockDim.x *
    // blockDim.y * sizeof(int)>>>(...)

    const int threads_per_block_x = blockDim.x * blockDim.y * blockDim.z;

    extern __shared__ int8_t shared_memory[];
    int *tet_indices = reinterpret_cast<int *>(shared_memory);  // Shared memory array to store tet indices for the current block

    // The z_coords array is used to store the z-coordinates of the quadrature points for the current block. It is placed in
    real_t *z_coords = reinterpret_cast<real_t *>(shared_memory + threads_per_block_x * sizeof(int));

    int *compact_tet_indices = reinterpret_cast<int *>(shared_memory +                         //
                                                       threads_per_block_x * sizeof(int) +     //
                                                       threads_per_block_x * sizeof(real_t));  //

    real_t *compact_z_coords = reinterpret_cast<real_t *>(shared_memory +                          //
                                                          2 * threads_per_block_x * sizeof(int) +  //
                                                          threads_per_block_x * sizeof(real_t));   //

    int *shared_write_pos = reinterpret_cast<int *>(shared_memory +                             //
                                                    2 * threads_per_block_x * sizeof(int) +     //
                                                    2 * threads_per_block_x * sizeof(real_t));  //
    // for compressing tet indices

    const int thread_id = threadIdx.x +                                           //
                          blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);  //

    const int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    // const int block_thread_id   = threadIdx.y * blockDim.x + threadIdx.x;

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

    for (int q_ijk = 0; q_ijk < QUAD_TOTAL; q_ijk++) {
        const real_t q_x = quad_x[q_ijk];
        const real_t q_y = quad_y[q_ijk];
        const real_t q_z = quad_z[q_ijk];
        const real_t q_w = quad_w[q_ijk];

        const real_t x_q         = grid_x + q_x * delta0;
        const real_t y_q         = grid_y + q_y * delta1;
        const real_t phys_z_base = origin2 + q_z * delta2;

        for (int block_k = 0; block_k < n2; block_k += threads_per_block) {
            const int local_k  = thread_id;
            const int global_k = block_k + local_k;

            tet_indices[local_k] = -1;

            __syncthreads();

            if (global_k < n2) {
                z_coords[local_k] = phys_z_base + (real_t)global_k * delta_z;

                // Query of the tet. for GPU CUDA with interleaved boxes ...
                tet_indices[local_k] =                                                   //
                        query_cell_list_3d_2d_split_map_mesh_given_xy_il_gpu(split_map,  //
                                                                             boxes,      //
                                                                             mesh_geom,  //
                                                                             x_q,        //
                                                                             y_q,        //
                                                                             z_coords[local_k]);
            }  // END if (global_k < n2)

            __syncthreads();

            compress_tet_indices_par_gpu(tet_indices,
                                         z_coords,
                                         compact_tet_indices,
                                         compact_z_coords,
                                         shared_write_pos,
                                         thread_id,
                                         threads_per_block_x);

            __syncthreads();

            if (global_k < n2 && tet_indices[local_k] > -1) {
                // Update field given the tet.
                update_hex_quad_node_cuda<index_t>(x_q,                                         //
                                                   y_q,                                         //
                                                   z_coords[local_k],                           //
                                                   q_w,                                         //
                                                   static_cast<index_t>(tet_indices[local_k]),  //
                                                   mesh,                                        //
                                                   mesh_geom,
                                                   stride0,  //
                                                   stride1,  //
                                                   stride2,  //
                                                   origin0,  //
                                                   origin1,  //
                                                   origin2,  //
                                                   inv_delta0,
                                                   inv_delta1,
                                                   inv_delta2,
                                                   weighted_field,
                                                   hex_field);  //
            }  // END if (global_k < n2 && tet_indices[local_k] > -1)
        }  // END for (int block_k = 0; block_k < n2; block_k += threads_per_block)
    }  // END for (int q_ijk = 0; q_ijk < QUAD_TOTAL; q_ijk++)

    return 0;
}

////////////////////////////////////////////////////
// transfer_to_hex_field_cell_split_tet4_shm_il_kernel
////////////////////////////////////////////////////
template <typename index_t = int>
__global__ void                                             //
transfer_to_hex_field_cell_split_tet4_shm_il_kernel(        //
        const int                         shared_mem_size,  // Shared memory size
        const cell_list_split_3d_2d_map_t split_map,        // Cell list split map data structure
        const boxes_interleaved_t         boxes,            // Interleaved boxes data structure
        const mesh_tet_geom_device_t      mesh_geom,        // Mesh geometry data structure
        const elems_tet4_device           mesh,             // Mesh: mesh_t struct
        const index_t                     start_i,          // Starting i index for the grid points in the hex mesh
        const index_t                     start_j,          // Starting j index for the grid points in the hex mesh
        const index_t                     delta_i,          // Cell list jump in x direction.
        const index_t                     delta_j,          // Cell list jump in y direction.
        const index_t                     size_i,           // Number of grid points in x direction
        const index_t                     size_j,           // Number of grid points in y direction
        const index_t                     n0,               // SDF: n[3]
        const index_t                     n1,               //
        const index_t                     n2,               //
        const index_t                     stride0,          // SDF: stride[3]
        const index_t                     stride1,          //
        const index_t                     stride2,          //
        const geom_t                      origin0,          // SDF: origin[3]
        const geom_t                      origin1,          //
        const geom_t                      origin2,          //
        const geom_t                      delta0,           // SDF: delta[3]
        const geom_t                      delta1,           //
        const geom_t                      delta2,           //
        const real_t *const __restrict__ weighted_field,    // Weighted field
        real_t *const __restrict__ hex_field) {             // Output field values for the hex nodes

    const index_t i_grid = start_i + static_cast<index_t>(blockIdx.x) * delta_i;
    const index_t j_grid = start_j + static_cast<index_t>(blockIdx.y) * delta_j;

    if (i_grid >= size_i - 1 || j_grid >= size_j - 1) {
        return;  // Out of bounds, exit the kernel
    }

    update_hex_field_shm_il<index_t>(&split_map,  //
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