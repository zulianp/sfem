#ifndef __RASTER_CELL_LIST_GPU_CUH__
#define __RASTER_CELL_LIST_GPU_CUH__

#include <cuda_runtime.h>
#include <cstdlib>

#include "cell_list_3d_1d_map_sur_mesh.h"
#include "cell_list_3d_map.h"
#include "cell_list_cuda.cuh"
#include "cell_list_query_cuda.cuh"
#include "cubature_cuda.cuh"
#include "sfem_gpu_math.cuh"
#include "sfem_resample_field_cuda_fun.cuh"

// sfem_base.h (via sfem_resample_field_cuda_fun.cuh) must precede cell_list_3d_1d_map.h
// so that real_t is defined when the struct body is parsed.
#include "cell_list_3d_1d_map.h"

//////////////////////////////////////////////////
// coord_to_grid_index
//////////////////////////////////////////////////
__device__ __forceinline__ int                 //
coord_to_grid_index_gpu(const real_t coord,    //
                        const real_t origin,   //
                        const real_t delta) {  //
    return (int)((coord - origin) / delta);
}  // END Function: coord_to_grid_index_gpu

//////////////////////////////////////////////////
// intersect_triangle_xy_gpu
//////////////////////////////////////////////////
__device__ __forceinline__ bool                //
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
__device__ __forceinline__ int                            //
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

////////////////////////////////////////////////////
// sort_real_array_ascending_gpu
////////////////////////////////////////////////////
__device__ __forceinline__ void                  //
sort_real_array_ascending_gpu(real_t   *values,  //
                              const int size) {  //
    // Insertion sort is efficient for the small intersection counts expected here.
    for (int i = 1; i < size; i++) {
        const real_t key = values[i];
        int          j   = i - 1;

        while (j >= 0 && values[j] > key) {
            values[j + 1] = values[j];
            j--;
        }  // END while (j >= 0 && values[j] > key)

        values[j + 1] = key;
    }  // END for (int i = 1; i < size; i++)
}  // END Function: sort_real_array_ascending_gpu

////////////////////////////////////////////////////
// check_intervals_gpu
////////////////////////////////////////////////////
__device__ __forceinline__ void            //
check_intervals_gpu(const real_t A[],      //
                    const int    n,        //
                    const real_t I[],      //
                    const int    i_size,   //
                    real_t       out[]) {  //
    int i = 0;
    int k = 0;

    while (i < n && k < i_size) {
        const real_t lo = I[k];
        const real_t hi = I[k + 1];

        while (i < n && A[i] < lo) out[i++] = 0.0;
        while (i < n && A[i] <= hi) out[i++] = 1.0;

        k += 2;
    }  // END while (i < n && k < i_size)

    while (i < n) out[i++] = 0.0;
}  // END Function: check_intervals_gpu

////////////////////////////////////////////////////////////
// raster_cell_list_3d_1d_split_map_mesh_given_xyz_tri3_gpu
////////////////////////////////////////////////////////////
__device__ void                                            //
raster_cell_list_3d_1d_split_map_mesh_given_xyz_tri3_gpu(  //
        const cell_list_split_3d_2d_map_t split_map,       // Cell list split map data structure
        const boxes_interleaved_t         boxes) {         // Interleaved boxes data structure)
}

//////////////////////////////////////////////////
// query_cell_list_3d_1d_map_mesh_given_xy_tri3_v
//////////////////////////////////////////////////
__device__ int                                                                                           //
query_cell_list_3d_1d_map_mesh_given_xy_tri3_gpu(const cell_list_3d_1d_map_t   *map,                     //
                                                 const mesh_tri3_geom_device_t *mesh_geom,               //
                                                 const real_t                   x,                       //
                                                 const real_t                   y,                       //
                                                 const int                      start_index_tri3_array,  //
                                                 const int                      size_tri3_intersect,     //
                                                 real_t                        *tri3_intersect_z) {      //

    const int ix_tmp = coord_to_grid_index_gpu(x, map->min_x, map->delta_x);
    const int ix     = (ix_tmp < 0) ? 0 : (ix_tmp >= map->num_cells_x) ? map->num_cells_x - 1 : ix_tmp;

    const int cell_index = ix;

    const int start_index = map->cell_ptr[cell_index];
    const int end_index   = map->cell_ptr[cell_index + 1];

    const int num_boxes_local = end_index - start_index;

    int triangles_found = 0;

    if (num_boxes_local > 0) {
        int lower_bound_index = lower_bound_float_gpu<real_t>(&map->upper_bounds_y[start_index],  //
                                                              (size_t)num_boxes_local,            //
                                                              y);                                 //

        const int start_index_up = (lower_bound_index > 1) ? start_index + lower_bound_index - 2 : start_index;
        const int size_up        = (lower_bound_index > 1) ? num_boxes_local - (lower_bound_index - 2) : num_boxes_local;
        const int offset_up      = start_index_up - start_index;

        int upper_bound_index =                                //
                upper_bound_float_gpu<real_t>(                 //
                        &map->lower_bounds_y[start_index_up],  //
                        size_up,                               //
                        y);                                    //

        // Adjust upper_bound_index back to be relative to start_index
        upper_bound_index += offset_up;

        lower_bound_index = lower_bound_index < 0 ? 0 :                                                           //
                                    (lower_bound_index > num_boxes_local ? num_boxes_local : lower_bound_index);  //
        upper_bound_index = upper_bound_index < 0 ? 0 :                                                           //
                                    (upper_bound_index > num_boxes_local ? num_boxes_local : upper_bound_index);  //

        const int *const    cell_dict = &map->cell_dict[start_index];
        const geom_t *const ec        = mesh_geom->element_coords;

        for (int i = lower_bound_index; i < upper_bound_index; i++) {
            const int           box_index = cell_dict[i];
            const geom_t *const row       = ec + box_index * 9;

            const real_t x0 = row[0], y0 = row[1], z0 = row[2];
            const real_t x1 = row[3], y1 = row[4], z1 = row[5];
            const real_t x2 = row[6], y2 = row[7], z2 = row[8];

            if (intersect_triangle_xy_gpu((real_t[3]){x0, y0, z0},  //
                                          (real_t[3]){x1, y1, z1},  //
                                          (real_t[3]){x2, y2, z2},  //
                                          x,
                                          y)) {
                real_t intersection_z;

                int f = intersection_point_triangle_xy_gpu((real_t[3]){x0, y0, z0},  //
                                                           (real_t[3]){x1, y1, z1},  //
                                                           (real_t[3]){x2, y2, z2},  //
                                                           x,
                                                           y,
                                                           &intersection_z);
                if (f == EXIT_SUCCESS) {
                    if (triangles_found + start_index_tri3_array >= size_tri3_intersect) {
                        return -1;  // buffer overflow: caller must pre-allocate a larger buffer
                    }

                    tri3_intersect_z[triangles_found + start_index_tri3_array] = intersection_z;
                    triangles_found++;
                }  // END if (f == EXIT_SUCCESS)
            }  // END if (intersect_triangle_xy_gpu(...))
        }  // END for (int i = lower_bound_index; i < upper_bound_index; i++)

    }  // END if (num_boxes_local > 0)

    return triangles_found;
}  // END Function: query_cell_list_3d_1d_map_mesh_given_xy_tri3_gpu

////////////////////////////////////////////////////////////////
// query_cell_list_3d_1d_split_map_mesh_given_xy_tri3_gpu
////////////////////////////////////////////////////////////////
__device__ int                                                                                                  //
query_cell_list_3d_1d_split_map_mesh_given_xy_tri3_gpu(const cell_list_split_3d_1d_map_t *map,                  //
                                                       const mesh_tri3_geom_device_t     *mesh_geom,            //
                                                       const real_t                       x,                    //
                                                       const real_t                       y,                    //
                                                       const int                          size_tri3_intersect,  //
                                                       real_t                            *tri3_intersect_z) {   //

    if (map == NULL || mesh_geom == NULL) {
        return -1;
    }

    const int num_found_lower = query_cell_list_3d_1d_map_mesh_given_xy_tri3_gpu(map->map_lower,       //
                                                                                 mesh_geom,            //
                                                                                 x,                    //
                                                                                 y,                    //
                                                                                 0,                    // start_index_tri3_array
                                                                                 size_tri3_intersect,  //
                                                                                 tri3_intersect_z);    //

    const int num_found_upper = query_cell_list_3d_1d_map_mesh_given_xy_tri3_gpu(map->map_upper,       //
                                                                                 mesh_geom,            //
                                                                                 x,                    //
                                                                                 y,                    //
                                                                                 num_found_lower,      // start_index_tri3_array
                                                                                 size_tri3_intersect,  //
                                                                                 tri3_intersect_z);    //

    if ((num_found_lower + num_found_upper) % 2 != 0) {
        printf("Warning: Odd number of triangle intersections found. "
               "\n*  %d found in lower map, %d found in upper map.\n",
               num_found_lower,
               num_found_upper);
    }

    return num_found_lower + num_found_upper;
}  // END Function: query_cell_list_3d_1d_split_map_mesh_given_xy_tri3_gpu

////////////////////////////////////////////////////
// raster_to_hex_field_tri3_kernel
////////////////////////////////////////////////////
template <typename index_t = int>
__global__ void                                       //
raster_to_hex_field_tri3_kernel(                      //
        const cell_list_split_3d_1d_map_t split_map,  // Cell list split map data structure
        const mesh_tri3_geom_device_t     mesh_geom,  // Mesh geometry data structure
        real_t *const __restrict__ tri3_intersect_z,  // Per-block scratch: size = total_blocks * size_tri3_intersect
        const int     size_tri3_intersect,            // Capacity per block
        const index_t start_i,                        // Starting i index
        const index_t start_j,                        // Starting j index
        const index_t delta_i,                        // Cell list jump in x direction
        const index_t delta_j,                        // Cell list jump in y direction
        const index_t size_i,                         // Number of grid points in x direction
        const index_t size_j,                         // Number of grid points in y direction
        const index_t size_k,                         // Number of grid points in z direction
        const geom_t  origin0,                        // Grid origin x
        const geom_t  origin1,                        // Grid origin y
        const geom_t  origin2,                        // Grid origin z
        const geom_t  delta0,                         // Grid spacing x
        const geom_t  delta1,                         // Grid spacing y
        const geom_t  delta2,                         // Grid spacing z
        real_t *const __restrict__ data) {            // Output: size_i * size_j * size_k

    const index_t i_grid = start_i + static_cast<index_t>(blockIdx.x) * delta_i;
    const index_t j_grid = start_j + static_cast<index_t>(blockIdx.y) * delta_j;

    if (i_grid >= size_i || j_grid >= size_j) {
        return;  // Out of bounds, exit the kernel
    }

    // Each block gets its own slice of the scratch buffer to avoid inter-block aliasing.
    const int block_linear_id              = blockIdx.y * gridDim.x + blockIdx.x;
    real_t   *my_intersect_z               = tri3_intersect_z + block_linear_id * size_tri3_intersect;

    const int num_tri3_intersect =
            query_cell_list_3d_1d_split_map_mesh_given_xy_tri3_gpu(&split_map,                 //
                                                                   &mesh_geom,                 //
                                                                   origin0 + i_grid * delta0,  //
                                                                   origin1 + j_grid * delta1,  //
                                                                   size_tri3_intersect,        //
                                                                   my_intersect_z);            //

    if (num_tri3_intersect == 0) {
        return;
    }

    if (num_tri3_intersect % 2 != 0) {
        printf("Warning: Odd number of triangle intersections found at grid point (%d, %d). "
               "\n*  %d intersecting triangles found.\n",
               i_grid,
               j_grid,
               num_tri3_intersect);
        return;
    }

    sort_real_array_ascending_gpu(my_intersect_z, num_tri3_intersect);

    // Walk the k column and mark voxels whose z-center falls inside a [lo, hi] pair.
    for (index_t k = 0; k < size_k; k++) {
        const geom_t z = origin2 + k * delta2;
        for (int p = 0; p < num_tri3_intersect; p += 2) {
            if (z >= my_intersect_z[p] && z <= my_intersect_z[p + 1]) {
                data[i_grid + j_grid * size_i + k * size_i * size_j] = (real_t)1;
                break;
            }
        }
    }

}  // END Kernel: raster_to_hex_field_tri3_kernel

#endif  // __RASTER_CELL_LIST_GPU_CUH__