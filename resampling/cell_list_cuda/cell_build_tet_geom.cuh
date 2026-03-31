#ifndef __CELL_BUILD_TET_GEOM_CUH__
#define __CELL_BUILD_TET_GEOM_CUH__

#include "cell_list_cuda.cuh"
#include "cell_list_resampling_gpu.h"
#include "resample_field_adjoint_cell_cuda.cuh"
#include "resample_field_adjoint_cell_cuda_shm.cuh"

/////////////////////////////////////////////////////////
// tet4_inv_Jacobian ////////////////////////////
/////////////////////////////////////////////////////////
__device__ static void                                //
tet_inv_Jacobian_mesh_geom_gpu(const real_t px0,      //
                               const real_t px1,      //
                               const real_t px2,      //
                               const real_t px3,      //
                               const real_t py0,      //
                               const real_t py1,      //
                               const real_t py2,      //
                               const real_t py3,      //
                               const real_t pz0,      //
                               const real_t pz1,      //
                               const real_t pz2,      //
                               const real_t pz3,      //
                               real_t      *J_inv) {  //
    //
    //

    /**
     ****************************************************************************************
    J^{-1} =
    \begin{bmatrix}
    inv_J11 & inv_J12 & inv_J13 \\
    inv_J21 & inv_J22 & inv_J23 \\
    inv_J31 & inv_J32 & inv_J33
    \end{bmatrix}
    *************************************************************************************************
     */

    // Compute the Jacobian matrix components
    const real_t J11 = -px0 + px1;
    const real_t J12 = -px0 + px2;
    const real_t J13 = -px0 + px3;

    const real_t J21 = -py0 + py1;
    const real_t J22 = -py0 + py2;
    const real_t J23 = -py0 + py3;

    const real_t J31 = -pz0 + pz1;
    const real_t J32 = -pz0 + pz2;
    const real_t J33 = -pz0 + pz3;

    // Compute common subexpressions for cofactor matrix
    const real_t J22_J33 = J22 * J33;
    const real_t J23_J32 = J23 * J32;
    const real_t J21_J33 = J21 * J33;
    const real_t J23_J31 = J23 * J31;
    const real_t J21_J32 = J21 * J32;
    const real_t J22_J31 = J22 * J31;
    const real_t J11_J33 = J11 * J33;
    const real_t J13_J31 = J13 * J31;
    const real_t J11_J23 = J11 * J23;
    const real_t J13_J21 = J13 * J21;
    const real_t J11_J32 = J11 * J32;
    const real_t J12_J31 = J12 * J31;
    const real_t J11_J22 = J11 * J22;
    const real_t J12_J21 = J12 * J21;
    const real_t J12_J33 = J12 * J33;
    const real_t J13_J32 = J13 * J32;
    const real_t J12_J23 = J12 * J23;
    const real_t J13_J22 = J13 * J22;

    // Compute cofactor differences (reused in determinant and inverse)
    const real_t cof00 = J22_J33 - J23_J32;
    const real_t cof01 = J23_J31 - J21_J33;
    const real_t cof02 = J21_J32 - J22_J31;
    const real_t cof10 = J13_J32 - J12_J33;
    const real_t cof11 = J11_J33 - J13_J31;
    const real_t cof12 = J12_J31 - J11_J32;
    const real_t cof20 = J12_J23 - J13_J22;
    const real_t cof21 = J13_J21 - J11_J23;
    const real_t cof22 = J11_J22 - J12_J21;

    // Compute the determinant of the Jacobian using cofactors
    const real_t det_J     = J11 * cof00 + J12 * cof01 + J13 * cof02;
    const real_t inv_det_J = 1.0 / det_J;

    // Compute the inverse of the Jacobian matrix using precomputed cofactors
    J_inv[0] = cof00 * inv_det_J;
    J_inv[1] = cof10 * inv_det_J;
    J_inv[2] = cof20 * inv_det_J;
    J_inv[3] = cof01 * inv_det_J;
    J_inv[4] = cof11 * inv_det_J;
    J_inv[5] = cof21 * inv_det_J;
    J_inv[6] = cof02 * inv_det_J;
    J_inv[7] = cof12 * inv_det_J;
    J_inv[8] = cof22 * inv_det_J;
}  // END: tet_inv_Jacobian_mesh_geom


/**
 * Compute the inverse Jacobian for each tetrahedral element in the mesh and store it in the device memory.
 * Additionally, store the coordinates of the first vertex of each tetrahedron for later use in res
 */
__global__ void                                                                   //
mesh_tet_geometry_compute_inv_Jacobian_gpu(mesh_tet_geom_device_t geom_device,    //
                                           xyz_tet4_device        xyz_device,     //
                                           elems_tet4_device      elems_device) { //

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int nelements = geom_device.nelements;

    if (idx >= nelements) return;

    const idx_t i0 = elems_device.elems_v0[idx];
    const idx_t i1 = elems_device.elems_v1[idx];
    const idx_t i2 = elems_device.elems_v2[idx];
    const idx_t i3 = elems_device.elems_v3[idx];

    const real_t x0_n = xyz_device.x[i0];
    const real_t y0_n = xyz_device.y[i0];
    const real_t z0_n = xyz_device.z[i0];

    const real_t x1_n = xyz_device.x[i1];
    const real_t y1_n = xyz_device.y[i1];
    const real_t z1_n = xyz_device.z[i1];

    const real_t x2_n = xyz_device.x[i2];
    const real_t y2_n = xyz_device.y[i2];
    const real_t z2_n = xyz_device.z[i2];

    const real_t x3_n = xyz_device.x[i3];
    const real_t y3_n = xyz_device.y[i3];
    const real_t z3_n = xyz_device.z[i3];

    real_t *J_inv = &geom_device.inv_Jacobian[idx * 9];

    tet_inv_Jacobian_mesh_geom_gpu(x0_n,  //
                                   x1_n,  //
                                   x2_n,
                                   x3_n,
                                   y0_n,
                                   y1_n,
                                   y2_n,
                                   y3_n,
                                   z0_n,
                                   z1_n,
                                   z2_n,
                                   z3_n,
                                   J_inv);

    geom_device.vetices_zero[idx * 3 + 0] = x0_n;
    geom_device.vetices_zero[idx * 3 + 1] = y0_n;
    geom_device.vetices_zero[idx * 3 + 2] = z0_n;
}

/**
 * @brief Uploads node coordinates to the GPU and computes inverse Jacobians for all tet4 elements.
 *        Allocates a temporary xyz device buffer, launches the kernel, then frees the buffer.
 * @param mesh       Host mesh (provides nnodes and points).
 * @param geom_device Pre-allocated device geometry struct (filled in-place by this function).
 * @param mesh_device Pre-allocated device element connectivity.
 * @param stream     CUDA stream used for all async operations.
 */
static inline void                                                          //
mesh_tet_geometry_build_on_device(const mesh_t          *mesh,             //
                                  mesh_tet_geom_device_t geom_device,      //
                                  elems_tet4_device      mesh_device,      //
                                  cudaStream_t           stream) {         //
    xyz_tet4_device xyz_device = make_xyz_tet4_device();
    cuda_allocate_xyz_tet4_device_async(&xyz_device, mesh->nnodes, stream);
    cudaStreamSynchronize(stream);

    copy_xyz_tet4_device_async((const geom_t **)mesh->points, mesh->nnodes, &xyz_device, stream);
    cudaStreamSynchronize(stream);

    const int block_size = 256;
    const int grid_size  = (mesh->nelements + block_size - 1) / block_size;
    mesh_tet_geometry_compute_inv_Jacobian_gpu<<<grid_size, block_size, 0, stream>>>(geom_device,  //
                                                                                     xyz_device,   //
                                                                                     mesh_device); //

    free_xyz_tet4_device_async(&xyz_device, stream);
}

#endif  // __CELL_BUILD_TET_GEOM_CUH__