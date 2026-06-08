#ifndef __RASTER_CELL_LIST_GPU_CUH__
#define __RASTER_CELL_LIST_GPU_CUH__

#include <cuda_runtime.h>
#include <cstdlib>

#include "cell_list_cuda.cuh"
#include "cell_list_query_cuda.cuh"
#include "cubature_cuda.cuh"
#include "sfem_gpu_math.cuh"
#include "sfem_resample_field_cuda_fun.cuh"

//////////////////////////////////////////////////
// intersect_triangle_xy_gpu
//////////////////////////////////////////////////
__host__ __device__ __forceinline__ bool       //
intersect_triangle_xy_gpu(const real_t v0[3],  //
                          const real_t v1[3],  //
                          const real_t v2[3],  //
                          const real_t x,      //
                          const real_t y) {    //
    // Signed area of each sub-triangle formed by an edge and the query point.
    // A point is inside iff all three have the same sign (all CW or all CCW).
    const real_t d0 = (v1[0] - v0[0]) * (y - v0[1]) - (v1[1] - v0[1]) * (x - v0[0]);
    const real_t d1 = (v2[0] - v1[0]) * (y - v1[1]) - (v2[1] - v1[1]) * (x - v1[0]);
    const real_t d2 = (v0[0] - v2[0]) * (y - v2[1]) - (v0[1] - v2[1]) * (x - v2[0]);

    const bool has_neg = (d0 < 0) | (d1 < 0) | (d2 < 0);
    const bool has_pos = (d0 > 0) | (d1 > 0) | (d2 > 0);
    return !(has_neg & has_pos);  // inside iff all same sign (or zero)
}  // END Function: intersect_triangle_xy_gpu

////////////////////////////////////////////////////
// intersection_point_triangle_xy_gpu
////////////////////////////////////////////////////
__host__ __device__ __forceinline__ int                   //
intersection_point_triangle_xy_gpu(const real_t v0[3],    //
                                   const real_t v1[3],    //
                                   const real_t v2[3],    //
                                   const real_t x,        //
                                   const real_t y,        //
                                   real_t      *out_z) {  //

    // Raw edge vectors (no need to normalize; plane equation is scale-invariant)
    const real_t e1x = v1[0] - v0[0], e1y = v1[1] - v0[1], e1z = v1[2] - v0[2];
    const real_t e2x = v2[0] - v0[0], e2y = v2[1] - v0[1], e2z = v2[2] - v0[2];

    // Cross product n = e1 x e2 (unnormalized plane normal)
    const real_t nx = e1y * e2z - e1z * e2y;
    const real_t ny = e1z * e2x - e1x * e2z;
    const real_t nz = e1x * e2y - e1y * e2x;

    // Degenerate or axis-aligned triangle: nz ~= 0 means no unique z solution.
    if (nz * nz < 1e-24) {
        *out_z = v0[2];
        return EXIT_FAILURE;
    }  // END if (nz * nz < 1e-24)

    // Plane equation: n . (P - v0) = 0  ->  solve for z
    *out_z = v0[2] - (nx * (x - v0[0]) + ny * (y - v0[1])) / nz;
    return EXIT_SUCCESS;
}  // END Function: intersection_point_triangle_xy_gpu

////////////////////////////////////////////////////////////
// raster_cell_list_3d_1d_split_map_mesh_given_xyz_tri3_v
////////////////////////////////////////////////////////////
__device__ void                                          //
raster_cell_list_3d_1d_split_map_mesh_given_xyz_tri3_v(  //
        const cell_list_split_3d_2d_map_t split_map,     // Cell list split map data structure
        const boxes_interleaved_t         boxes) {       // Interleaved boxes data structure)
}

////////////////////////////////////////////////////
// raster_to_hex_field_tri3_kernel
////////////////////////////////////////////////////
template <typename index_t = int>
__global__ void                                                 //
raster_to_hex_field_tri3_kernel(                                //
        const cell_list_split_3d_2d_map_t split_map,            // Cell list split map data structure
        const boxes_interleaved_t         boxes,                // Interleaved boxes data structure
        const mesh_tet_geom_device_t      mesh_geom,            // Mesh geometry data structure
        const elems_tet4_device           mesh,                 // Mesh: mesh_t struct
        const real_t                      tri3_intersect_z,     // Intersection z-coordinate for tri3
        const int                         size_tri3_intersect,  // Number of intersecting tri3 elements
        const index_t                     start_i,              // Starting i index for the grid points in the hex mesh
        const index_t                     start_j,              // Starting j index for the grid points in the hex mesh
        const index_t                     delta_i,              // Cell list jump in x direction.
        const index_t                     delta_j,              // Cell list jump in y direction.
        const index_t                     size_i,               // Number of grid points in x direction
        const index_t                     size_j,               // Number of grid points in y direction
        const index_t                     n0,                   // SDF: n[3]
        const index_t                     n1,                   //
        const index_t                     n2,                   //
        const index_t                     stride0,              // SDF: stride[3]
        const index_t                     stride1,              //
        const index_t                     stride2,              //
        const geom_t                      origin0,              // SDF: origin[3]
        const geom_t                      origin1,              //
        const geom_t                      origin2,              //
        const geom_t                      delta0,               // SDF: delta[3]
        const geom_t                      delta1,               //
        const geom_t                      delta2,               //
        const real_t *const __restrict__ weighted_field,        //
        real_t *const __restrict__ hex_field) {                 //

    const index_t i_grid = start_i + static_cast<index_t>(blockIdx.x) * delta_i;
    const index_t j_grid = start_j + static_cast<index_t>(blockIdx.y) * delta_j;

    if (i_grid >= size_i - 1 || j_grid >= size_j - 1) {
        return;  // Out of bounds, exit the kernel
    }

    // TODO: We can further optimize this by precomputing the grid point coordinates and passing them as arguments to the kernel,
    // instead of computing them on the fly for each thread.

}  // END Kernel: raster_to_hex_field_tri3_kernel

#endif  // __RASTER_CELL_LIST_GPU_CUH__