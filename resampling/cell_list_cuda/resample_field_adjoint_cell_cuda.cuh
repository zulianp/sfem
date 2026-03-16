#ifndef __RESAMPLE_FIELD_ADJOINT_CELL_CUDA_CUH__
#define __RESAMPLE_FIELD_ADJOINT_CELL_CUDA_CUH__

#include <cuda_runtime.h>

#include "cell_list_cuda.cuh"
#include "cell_list_query_cuda.cuh"
#include "cubature_cuda.cuh"
#include "sfem_gpu_math.cuh"
#include "sfem_resample_field_cuda_fun.cuh"

//////////////////////////////////////////////
// update_hex_field
//////////////////////////////////////////////
template <typename index_t>
__device__ __forceinline__ int                                       //
update_hex_quad_node_cuda(                                           //
        const real_t  x,                                             // Physical x coordinate of the quadrature point
        const real_t  y,                                             // Physical y coordinate of the quadrature point
        const real_t  z,                                             // Physical z coordinate of the quadrature point
        const real_t  phys_w,                                        // Quadrature weight for the quadrature point
        const index_t index_tet,                                     // The index of the tet containing the quadrature point
        const elems_tet4_device *const __restrict__ mesh,            // Mesh: mesh_t struct
        const mesh_tet_geom_device_t *const __restrict__ mesh_geom,  // Mesh geometry data structure
        const index_t stride0,                                       // SDF: stride[3]
        const index_t stride1,                                       //
        const index_t stride2,                                       //
        const geom_t  origin0,                                       // SDF: origin[3]
        const geom_t  origin1,                                       //
        const geom_t  origin2,                                       //
        const real_t  inv_delta0,                                    // Precomputed 1.0 / delta0
        const real_t  inv_delta1,                                    // Precomputed 1.0 / delta1
        const real_t  inv_delta2,                                    // Precomputed 1.0 / delta2
        const real_t *const __restrict__ weighted_field,             // Weighted field
        real_t *const SFEM_RESTRICT hex_element_field) {             // Output field values for the 8 hex nodes

    const index_t off0 = 0;
    const index_t off1 = stride0;
    const index_t off2 = stride0 + stride1;
    const index_t off3 = stride1;
    const index_t off4 = stride2;
    const index_t off5 = stride0 + stride2;
    const index_t off6 = stride0 + stride1 + stride2;
    const index_t off7 = stride1 + stride2;

    const real_t ox = origin0;
    const real_t oy = origin1;
    const real_t oz = origin2;

    const real_t grid_x = (x - ox) * inv_delta0;
    const real_t grid_y = (y - oy) * inv_delta1;
    const real_t grid_z = (z - oz) * inv_delta2;

    const index_t i = fast_floor(grid_x);
    const index_t j = fast_floor(grid_y);
    const index_t k = fast_floor(grid_z);

    const real_t l_x = (grid_x - (real_t)i);
    const real_t l_y = (grid_y - (real_t)j);
    const real_t l_z = (grid_z - (real_t)k);

    const index_t base_index = i * stride0 +  //
                               j * stride1 +  //
                               k * stride2;   //

    const idx_t ev0 = mesh->elems_v0[index_tet];
    const idx_t ev1 = mesh->elems_v1[index_tet];
    const idx_t ev2 = mesh->elems_v2[index_tet];
    const idx_t ev3 = mesh->elems_v3[index_tet];

    const real_t wf0 = weighted_field[ev0];  // Weighted field at vertex 0
    const real_t wf1 = weighted_field[ev1];  // Weighted field at vertex 1
    const real_t wf2 = weighted_field[ev2];  // Weighted field at vertex 2
    const real_t wf3 = weighted_field[ev3];  // Weighted field at vertex 3

    const real_t *inv_J_tet = &(mesh_geom->inv_Jacobian[index_tet * 9]);  // Inverse Jacobian for the current tet

    const index_t base_vertex_idx = index_tet * 3;
    const real_t *base_vertex     = &mesh_geom->vetices_zero[base_vertex_idx];  // Pointer to vertex-0 xyz in packed array

    // real_t xyz_n[3];
    // memcpy(xyz_n, &mesh_geom->vetices_zero[base_vertex_idx], 3 * sizeof(real_t));

    const real_t x0_n = base_vertex[0];  // x coordinate of vertex 0
    const real_t y0_n = base_vertex[1];  // y coordinate of vertex 0
    const real_t z0_n = base_vertex[2];  // z coordinate of vertex 0

    // Compute the coordinates of the quadrature point in the reference tetrahedron using the inverse Jacobian transformation.
    const real_t x_o = x - x0_n;
    const real_t y_o = y - y0_n;
    const real_t z_o = z - z0_n;

    // real_t inv_J[9];
    // memcpy(inv_J, inv_J_tet, 9 * sizeof(real_t));

    const real_t inv_J_00 = inv_J_tet[0];
    const real_t inv_J_01 = inv_J_tet[1];
    const real_t inv_J_02 = inv_J_tet[2];
    const real_t inv_J_10 = inv_J_tet[3];
    const real_t inv_J_11 = inv_J_tet[4];
    const real_t inv_J_12 = inv_J_tet[5];
    const real_t inv_J_20 = inv_J_tet[6];
    const real_t inv_J_21 = inv_J_tet[7];
    const real_t inv_J_22 = inv_J_tet[8];

    const real_t x_ref = fast_fma(inv_J_02, z_o, fast_fma(inv_J_01, y_o, inv_J_00 * x_o));
    const real_t y_ref = fast_fma(inv_J_12, z_o, fast_fma(inv_J_11, y_o, inv_J_10 * x_o));
    const real_t z_ref = fast_fma(inv_J_22, z_o, fast_fma(inv_J_21, y_o, inv_J_20 * x_o));

    const real_t f0 = real_t(1) - x_ref - y_ref - z_ref;
    const real_t f1 = x_ref;
    const real_t f2 = y_ref;
    const real_t f3 = z_ref;

    const real_t wf_quad = fast_fma(f3, wf3, fast_fma(f2, wf2, fast_fma(f1, wf1, f0 * wf0)));

    const real_t one_minus_lx = (real_t(1) - l_x);
    const real_t one_minus_ly = (real_t(1) - l_y);
    const real_t one_minus_lz = (real_t(1) - l_z);

    const real_t c0 = one_minus_lx * one_minus_ly;
    const real_t c1 = l_x * one_minus_ly;
    const real_t c2 = l_x * l_y;
    const real_t c3 = one_minus_lx * l_y;

    const real_t wf_quad_QW = wf_quad * phys_w;

    const real_t w_c0 = wf_quad_QW * c0;
    const real_t w_c1 = wf_quad_QW * c1;
    const real_t w_c2 = wf_quad_QW * c2;
    const real_t w_c3 = wf_quad_QW * c3;

    real_t *const SFEM_RESTRICT out = &hex_element_field[base_index];

    // atomicAdd(&out[off0], w_c0 * one_minus_lz);
    // atomicAdd(&out[off1], w_c1 * one_minus_lz);
    // atomicAdd(&out[off2], w_c2 * one_minus_lz);
    // atomicAdd(&out[off3], w_c3 * one_minus_lz);
    // atomicAdd(&out[off4], w_c0 * l_z);
    // atomicAdd(&out[off5], w_c1 * l_z);
    // atomicAdd(&out[off6], w_c2 * l_z);
    // atomicAdd(&out[off7], w_c3 * l_z);

    // There is no conflict in between threads.
    // Since threads update the grid points shifted by 1 along z-axis.
    // beside they update adiacent grid points.
    out[off0] += w_c0 * one_minus_lz;
    out[off1] += w_c1 * one_minus_lz;
    out[off2] += w_c2 * one_minus_lz;
    out[off3] += w_c3 * one_minus_lz;
    out[off4] += w_c0 * l_z;
    out[off5] += w_c1 * l_z;
    out[off6] += w_c2 * l_z;
    out[off7] += w_c3 * l_z;

    return 0;
}  // END Function: update_hex_quad_node

///////////////////////////////////////////////
// update_hex_field
///////////////////////////////////////////////
template <typename index_t>
__device__ __forceinline__ int                                      //
update_hex_field(const cell_list_split_3d_2d_map_t *split_map,      // Cell list split map data structure
                 const boxes_t                     *boxes,          // Boxes data structure
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

    const int threads_per_block = blockDim.x * blockDim.y;
    const int block_thread_id   = threadIdx.y * blockDim.x + threadIdx.x;

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
            const int k = block_k + block_thread_id;

            if (k < n2) {
                const real_t z = phys_z_base + (real_t)k * delta_z;

                // Query of the tet. for GPU CUDA ...
                const int tet_idx =                                                   //
                        query_cell_list_3d_2d_split_map_mesh_given_xy_gpu(split_map,  //
                                                                          boxes,      //
                                                                          mesh_geom,  //
                                                                          x_q,        //
                                                                          y_q,        //
                                                                          z);         //

                if (tet_idx > -1) {
                    // Update field given the tet.
                    update_hex_quad_node_cuda<index_t>(x_q,  //
                                                       y_q,  //
                                                       z,    //
                                                       q_w,  //
                                                       static_cast<index_t>(tet_idx),
                                                       mesh,  //
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
                }  // END if (tet_idx > -1)
            }  // END if (k < n2)
        }  // END for (int block_k = 0; block_k < n2; block_k += threads_per_block)
    }  // END for (int q_ijk = 0; q_ijk < QUAD_TOTAL; q_ijk++)

    return 0;
}

///////////////////////////////////////////////
// update_hex_field_il
///////////////////////////////////////////////
template <typename index_t>
__device__ __forceinline__ int                                         //
update_hex_field_il(const cell_list_split_3d_2d_map_t *split_map,      // Cell list split map data structure
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

    const int threads_per_block = blockDim.x * blockDim.y;
    const int block_thread_id   = threadIdx.y * blockDim.x + threadIdx.x;

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
            const int k = block_k + block_thread_id;

            if (k < n2) {
                const real_t z = phys_z_base + (real_t)k * delta_z;

                // Query of the tet. for GPU CUDA with interleaved boxes ...
                const int tet_idx =                                                      //
                        query_cell_list_3d_2d_split_map_mesh_given_xy_il_gpu(split_map,  //
                                                                             boxes,      //
                                                                             mesh_geom,  //
                                                                             x_q,        //
                                                                             y_q,        //
                                                                             z);         //

                if (tet_idx > -1) {
                    // Update field given the tet.
                    update_hex_quad_node_cuda<index_t>(x_q,  //
                                                       y_q,  //
                                                       z,    //
                                                       q_w,  //
                                                       static_cast<index_t>(tet_idx),
                                                       mesh,  //
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
                }  // END if (tet_idx > -1)
            }  // END if (k < n2)
        }  // END for (int block_k = 0; block_k < n2; block_k += threads_per_block)
    }  // END for (int q_ijk = 0; q_ijk < QUAD_TOTAL; q_ijk++)

    return 0;
}

/////////////////////////////////////////////////
// transfer_to_hex_field_cell_split_tet4_kernel
/////////////////////////////////////////////////
template <typename index_t = int>
__global__ void                                           //
transfer_to_hex_field_cell_split_tet4_kernel(             //
        const cell_list_split_3d_2d_map_t split_map,      // Cell list split map data structure
        const boxes_t                     boxes,          // Boxes data structure
        const mesh_tet_geom_device_t      mesh_geom,      // Mesh geometry data structure
        const elems_tet4_device           mesh,           // Mesh: mesh_t struct
        const index_t                     start_i,        // Starting i index for the grid points in the hex mesh
        const index_t                     start_j,        // Starting j index for the grid points in the hex mesh
        const index_t                     delta_i,        // Cell list jump in x direction.
        const index_t                     delta_j,        // Cell list jump in y direction.
        const index_t                     size_i,         // Number of grid points in x direction
        const index_t                     size_j,         // Number of grid points in y direction
        const index_t                     n0,             // SDF: n[3]
        const index_t                     n1,             //
        const index_t                     n2,             //
        const index_t                     stride0,        // SDF: stride[3]
        const index_t                     stride1,        //
        const index_t                     stride2,        //
        const geom_t                      origin0,        // SDF: origin[3]
        const geom_t                      origin1,        //
        const geom_t                      origin2,        //
        const geom_t                      delta0,         // SDF: delta[3]
        const geom_t                      delta1,         //
        const geom_t                      delta2,         //
        const real_t *const __restrict__ weighted_field,  // Weighted field
        real_t *const __restrict__ hex_field) {           // Output field values for the hex nodes

    const index_t i_grid = start_i + static_cast<index_t>(blockIdx.x) * delta_i;
    const index_t j_grid = start_j + static_cast<index_t>(blockIdx.y) * delta_j;

    if (i_grid >= size_i - 1 || j_grid >= size_j - 1) {
        return;  // Out of bounds, exit the kernel
    }

    update_hex_field<index_t>(&split_map,  //
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

////////////////////////////////////////////////////
// transfer_to_hex_field_cell_split_tet4_il_kernel
////////////////////////////////////////////////////
template <typename index_t = int>
__global__ void                                           //
transfer_to_hex_field_cell_split_tet4_il_kernel(          //
        const cell_list_split_3d_2d_map_t split_map,      // Cell list split map data structure
        const boxes_interleaved_t         boxes,          // Interleaved boxes data structure
        const mesh_tet_geom_device_t      mesh_geom,      // Mesh geometry data structure
        const elems_tet4_device           mesh,           // Mesh: mesh_t struct
        const index_t                     start_i,        // Starting i index for the grid points in the hex mesh
        const index_t                     start_j,        // Starting j index for the grid points in the hex mesh
        const index_t                     delta_i,        // Cell list jump in x direction.
        const index_t                     delta_j,        // Cell list jump in y direction.
        const index_t                     size_i,         // Number of grid points in x direction
        const index_t                     size_j,         // Number of grid points in y direction
        const index_t                     n0,             // SDF: n[3]
        const index_t                     n1,             //
        const index_t                     n2,             //
        const index_t                     stride0,        // SDF: stride[3]
        const index_t                     stride1,        //
        const index_t                     stride2,        //
        const geom_t                      origin0,        // SDF: origin[3]
        const geom_t                      origin1,        //
        const geom_t                      origin2,        //
        const geom_t                      delta0,         // SDF: delta[3]
        const geom_t                      delta1,         //
        const geom_t                      delta2,         //
        const real_t *const __restrict__ weighted_field,  // Weighted field
        real_t *const __restrict__ hex_field) {           // Output field values for the hex nodes

    const index_t i_grid = start_i + static_cast<index_t>(blockIdx.x) * delta_i;
    const index_t j_grid = start_j + static_cast<index_t>(blockIdx.y) * delta_j;

    if (i_grid >= size_i - 1 || j_grid >= size_j - 1) {
        return;  // Out of bounds, exit the kernel
    }

    update_hex_field_il<index_t>(&split_map,  //
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

#endif /* __RESAMPLE_FIELD_ADJOINT_CELL_CUDA_CUH__ */