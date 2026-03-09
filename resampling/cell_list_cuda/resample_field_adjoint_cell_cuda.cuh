#ifndef __RESAMPLE_FIELD_ADJOINT_CELL_CUDA_CUH__
#define __RESAMPLE_FIELD_ADJOINT_CELL_CUDA_CUH__

#include <cuda_runtime.h>

#include "cell_list_cuda.cuh"
#include "cell_list_query_cuda.cuh"
#include "cubature_cuda.cuh"
#include "sfem_resample_field_cuda_fun.cuh"

//////////////////////////////////////////////
// update_hex_field
//////////////////////////////////////////////
__device__ int                                                       //
update_hex_quad_node_cuda(                                           //
        const real_t    x,                                           // Physical x coordinate of the quadrature point
        const real_t    y,                                           // Physical y coordinate of the quadrature point
        const real_t    z,                                           // Physical z coordinate of the quadrature point
        const real_t    phys_w,                                      // Quadrature weight for the quadrature point
        const ptrdiff_t index_tet,                                   // The index of the tet containing the quadrature point
        const elems_tet4_device *const __restrict__ mesh,            // Mesh: mesh_t struct
        const mesh_tet_geom_device_t *const __restrict__ mesh_geom,  // Mesh geometry data structure
        const ptrdiff_t stride0,                                     // SDF: stride[3]
        const ptrdiff_t stride1,                                     //
        const ptrdiff_t stride2,                                     //
        const geom_t    origin0,                                     // SDF: origin[3]
        const geom_t    origin1,                                     //
        const geom_t    origin2,                                     //
        const geom_t    delta0,                                      // SDF: delta[3]
        const geom_t    delta1,                                      //
        const geom_t    delta2,                                      //
        const real_t *const __restrict__ weighted_field,             // Weighted field
        real_t *const SFEM_RESTRICT hex_element_field) {             // Output field values for the 8 hex nodes

    const ptrdiff_t off0 = 0;
    const ptrdiff_t off1 = stride0;
    const ptrdiff_t off2 = stride0 + stride1;
    const ptrdiff_t off3 = stride1;
    const ptrdiff_t off4 = stride2;
    const ptrdiff_t off5 = stride0 + stride2;
    const ptrdiff_t off6 = stride0 + stride1 + stride2;
    const ptrdiff_t off7 = stride1 + stride2;

    const real_t ox = origin0;
    const real_t oy = origin1;
    const real_t oz = origin2;

    const real_t inv_dx = 1.0 / delta0;
    const real_t inv_dy = 1.0 / delta1;
    const real_t inv_dz = 1.0 / delta2;

    const real_t grid_x = (x - ox) * inv_dx;
    const real_t grid_y = (y - oy) * inv_dy;
    const real_t grid_z = (z - oz) * inv_dz;

    const ptrdiff_t i = floor(grid_x);
    const ptrdiff_t j = floor(grid_y);
    const ptrdiff_t k = floor(grid_z);

    const real_t l_x = (grid_x - (real_t)i);
    const real_t l_y = (grid_y - (real_t)j);
    const real_t l_z = (grid_z - (real_t)k);

    const ptrdiff_t base_index = i * stride0 +  //
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

    const real_t x0_n = mesh_geom->vetices_zero[index_tet * 3 + 0];  // x coordinate of vertex 0
    const real_t y0_n = mesh_geom->vetices_zero[index_tet * 3 + 1];  // y coordinate of vertex 0
    const real_t z0_n = mesh_geom->vetices_zero[index_tet * 3 + 2];  // z coordinate of vertex 0

    // Compute the coordinates of the quadrature point in the reference tetrahedron using the inverse Jacobian transformation.
    const real_t x_o = x - x0_n;
    const real_t y_o = y - y0_n;
    const real_t z_o = z - z0_n;

    const real_t inv_J_00 = inv_J_tet[0];
    const real_t inv_J_01 = inv_J_tet[1];
    const real_t inv_J_02 = inv_J_tet[2];
    const real_t inv_J_10 = inv_J_tet[3];
    const real_t inv_J_11 = inv_J_tet[4];
    const real_t inv_J_12 = inv_J_tet[5];
    const real_t inv_J_20 = inv_J_tet[6];
    const real_t inv_J_21 = inv_J_tet[7];
    const real_t inv_J_22 = inv_J_tet[8];

    const real_t x_ref = inv_J_00 * x_o + inv_J_01 * y_o + inv_J_02 * z_o;
    const real_t y_ref = inv_J_10 * x_o + inv_J_11 * y_o + inv_J_12 * z_o;
    const real_t z_ref = inv_J_20 * x_o + inv_J_21 * y_o + inv_J_22 * z_o;

    const real_t f0 = 1.0 - x_ref - y_ref - z_ref;
    const real_t f1 = x_ref;
    const real_t f2 = y_ref;
    const real_t f3 = z_ref;

    const real_t wf_quad = f0 * wf0 + f1 * wf1 + f2 * wf2 + f3 * wf3;

    const real_t one_minus_lx = (1.0 - l_x);
    const real_t one_minus_ly = (1.0 - l_y);
    const real_t one_minus_lz = (1.0 - l_z);

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
__device__ int                                                      //
update_hex_field(cell_list_split_3d_2d_map_t *split_map,            // Cell list split map data structure
                 boxes_t                     *boxes,                // Boxes data structure
                 const mesh_tet_geom_t       *mesh_geom,            // Mesh geometry data structure
                 const ptrdiff_t              i_grid,               // The i index of the grid point in the hex mesh
                 const ptrdiff_t              j_grid,               // The j index of the grid point in the hex mesh
                 const elems_tet4_device *const __restrict__ mesh,  // Mesh: mesh_t struct
                 const ptrdiff_t n0,                                // SDF: n[3]
                 const ptrdiff_t n1,                                //
                 const ptrdiff_t n2,                                //
                 const ptrdiff_t const __restrict__ stride0,        // SDF: stride[3]
                 const ptrdiff_t const __restrict__ stride1,        //
                 const ptrdiff_t const __restrict__ stride2,        //
                 const geom_t const __restrict__ origin0,           // SDF: origin[3]
                 const geom_t const __restrict__ origin1,           //
                 const geom_t const __restrict__ origin2,           //
                 const geom_t const __restrict__ delta0,            // SDF: delta[3]
                 const geom_t const __restrict__ delta1,            //
                 const geom_t const __restrict__ delta2,            //
                 const real_t *const __restrict__ weighted_field,   // Weighted field
                 real_t *const __restrict__ hex_field) {            //

    const int threads_per_block = blockDim.x * blockDim.y;
    const int block_thread_id   = threadIdx.y * blockDim.x + threadIdx.x;

    const real_t grid_x  = real_t(origin0) + real_t(i_grid) * real_t(delta0);
    const real_t grid_y  = real_t(origin1) + real_t(j_grid) * real_t(delta1);
    const real_t delta_z = delta2;

    const real_t *quad_x = QuadPoints<real_t>::x();
    const real_t *quad_y = QuadPoints<real_t>::y();
    const real_t *quad_z = QuadPoints<real_t>::z();
    const real_t *quad_w = QuadPoints<real_t>::w();

    for (int q_ijk = 0; q_ijk < QUAD_TOTAL; q_ijk++) {
        const real_t q_x = quad_x[q_ijk];
        const real_t q_y = quad_y[q_ijk];
        const real_t q_z = quad_z[q_ijk];
        const real_t q_w = quad_w[q_ijk];

        const real_t phys_z_base = origin2 + q_z * delta2;

        for (int block_k = 0; block_k < n2; block_k += threads_per_block) {
            const int k = block_k + block_thread_id;

            if (k >= n2) break;

            const real_t z = phys_z_base + (real_t)k * delta_z;

            // Query of the tet. for GPU CUDA ...
            // That's tricky

            // Update filed given the tet.
        }
    }
}

/////////////////////////////////////////////////
// transfer_to_hex_field_cell_split_tet4_kernel
/////////////////////////////////////////////////
__global__ void                                            //
transfer_to_hex_field_cell_split_tet4_kernel(              //
        cell_list_split_3d_2d_map_t *split_map,            // Cell list split map data structure
        boxes_t                     *boxes,                // Boxes data structure
        const mesh_tet_geom_t       *mesh_geom,            // Mesh geometry data structure
        const elems_tet4_device *const __restrict__ mesh,  // Mesh: mesh_t struct
        const int       delta_x,                           // Cell list box size in x direction
        const int       delta_y,                           // Cell list box size in y direction
        const int       size_x,                            // Number of grid points in x direction
        const int       size_y,                            // Number of grid points in y direction
        const ptrdiff_t n0,                                // SDF: n[3]
        const ptrdiff_t n1,                                //
        const ptrdiff_t n2,                                //
        const ptrdiff_t stride0,                           // SDF: stride[3]
        const ptrdiff_t stride1,                           //
        const ptrdiff_t stride2,                           //
        const geom_t    origin0,                           // SDF: origin[3]
        const geom_t    origin1,                           //
        const geom_t    origin2,                           //
        const geom_t    delta0,                            // SDF: delta[3]
        const geom_t    delta1,                            //
        const geom_t    delta2,                            //
        const real_t *const __restrict__ weighted_field,   // Weighted field
        real_t *const __restrict__ hex_field) {            // Output field values for the hex nodes

    const int i_grid = (blockIdx.x * blockDim.x + threadIdx.x) * delta_x;
    const int j_grid = (blockIdx.y * blockDim.y + threadIdx.y) * delta_y;

    if (i_grid >= size_x || j_grid >= size_y) {
        return;  // Out of bounds, exit the kernel
    }

    update_hex_field(split_map,  //
                     boxes,
                     mesh_geom,
                     i_grid,
                     j_grid,
                     mesh,
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