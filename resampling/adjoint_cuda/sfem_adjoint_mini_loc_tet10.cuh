#ifndef __SFEM_ADJOINT_MINI_LOC_TET10_CUH__
#define __SFEM_ADJOINT_MINI_LOC_TET10_CUH__

#include "sfem_adjoint_mini_tet.cuh"
#include "sfem_resample_field_cuda_fun.cuh"

////////////////////////////////////////////////////////////////////////////////
// Compute matrix J_phys * J_ref = J_tot
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ __inline__ void                                                         //
compute_matrix_mult_3x3_gpu(const typename Float3<FloatType>::type* const J_phys,  // Jacobian matrix of the physical tetrahedron
                            const typename Float3<FloatType>::type* const J_ref,   // Jacobian matrix of the reference tetrahedron
                            typename Float3<FloatType>::type*             J_tot) {             // Output Jacobian matrix
    // Row 1: x-components (indices 0,1,2)
    J_tot[0].x = J_phys[0].x * J_ref[0].x + J_phys[0].y * J_ref[1].x + J_phys[0].z * J_ref[2].x;
    J_tot[0].y = J_phys[0].x * J_ref[0].y + J_phys[0].y * J_ref[1].y + J_phys[0].z * J_ref[2].y;
    J_tot[0].z = J_phys[0].x * J_ref[0].z + J_phys[0].y * J_ref[1].z + J_phys[0].z * J_ref[2].z;

    // Row 2: y-components (indices 3,4,5)
    J_tot[1].x = J_phys[1].x * J_ref[0].x + J_phys[1].y * J_ref[1].x + J_phys[1].z * J_ref[2].x;
    J_tot[1].y = J_phys[1].x * J_ref[0].y + J_phys[1].y * J_ref[1].y + J_phys[1].z * J_ref[2].y;
    J_tot[1].z = J_phys[1].x * J_ref[0].z + J_phys[1].y * J_ref[1].z + J_phys[1].z * J_ref[2].z;

    // Row 3: z-components (indices 6,7,8)
    J_tot[2].x = J_phys[2].x * J_ref[0].x + J_phys[2].y * J_ref[1].x + J_phys[2].z * J_ref[2].x;
    J_tot[2].y = J_phys[2].x * J_ref[0].y + J_phys[2].y * J_ref[1].y + J_phys[2].z * J_ref[2].y;
    J_tot[2].z = J_phys[2].x * J_ref[0].z + J_phys[2].y * J_ref[1].z + J_phys[2].z * J_ref[2].z;

    return;
}

////////////////////////////////////////////////////////////////////////////////
// Compute the physical coordinates of the mini-tetrahedra
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
void                                                                             //
compute_tet10_phys_mini_gpu(const typename Float3<FloatType>::type* const J_fc,  //
                            const typename Float3<FloatType>::type* const J_phys,
                            const typename Float3<FloatType>::type        b0,   //
                            const typename Float3<FloatType>::type        v0f,  //
                            FloatType*                                    x_m,  //
                            FloatType*                                    y_m,  //
                            FloatType*                                    z_m) {
    const FloatType x_unit[10] = {0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0};
    const FloatType y_unit[10] = {0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5};
    const FloatType z_unit[10] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5};

    for (int i = 0; i < 10; i++) {
        x_m[i] = J_fc[0].x * x_unit[i] + J_fc[0].y * y_unit[i] + J_fc[0].z * z_unit[i] +  //
                 J_phys[0].x * b0.x + J_phys[0].y * b0.y + J_phys[0].z * b0.z + v0f.x;    //

        y_m[i] = J_fc[1].x * x_unit[i] + J_fc[1].y * y_unit[i] + J_fc[1].z * z_unit[i] +  //
                 J_phys[1].x * b0.x + J_phys[1].y * b0.y + J_phys[1].z * b0.z + v0f.y;    //

        z_m[i] = J_fc[2].x * x_unit[i] + J_fc[2].y * y_unit[i] + J_fc[2].z * z_unit[i] +  //
                 J_phys[2].x * b0.x + J_phys[2].y * b0.y + J_phys[2].z * b0.z + v0f.z;    //
    }

    return;
}

////////////////////////////////////////////////////////////////////////////////
// Compute the values of the tet10 basis functions at (qx,qy,qz)
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ __forceinline__ void  //
tet10_Lagrange_basis_gpu(const FloatType qx, const FloatType qy, const FloatType qz, FloatType* const f) {
    const FloatType x0 = qx + qy + qz - 1;
    const FloatType x1 = 2 * qy;
    const FloatType x2 = 2 * qz;
    const FloatType x3 = 2 * qx - 1;
    const FloatType x4 = 4 * qx;
    const FloatType x5 = 4 * x0;

    f[0] = x0 * (x1 + x2 + x3);
    f[1] = qx * x3;
    f[2] = qy * (x1 - 1);
    f[3] = qz * (x2 - 1);
    f[4] = -x0 * x4;
    f[5] = qy * x4;
    f[6] = -qy * x5;
    f[7] = -qz * x5;
    f[8] = qz * x4;
    f[9] = 4 * qy * qz;

}  // END: tet10_Lagrange_basis

////////////////////////////////////////////////////////////////////////////////
// Compute the values of the tet10 basis functions at the mini-tet points
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ __inline__ void                                                         //
compute_wf_tet10_mini_gpu(const FloatType*                              wf_tet10,  //
                          const typename Float3<FloatType>::type* const J_ref_c,   //
                          const typename Float3<FloatType>::type        b0,        //
                          FloatType*                                    wf_tet10_mini) {                              //

    const FloatType x_unit[10] = {0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0};
    const FloatType y_unit[10] = {0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5};
    const FloatType z_unit[10] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5};

    FloatType tet10_f[10];

    for (int i = 0; i < 10; i++) {
        const FloatType x_mini = J_ref_c[0].x * x_unit[i] + J_ref_c[0].y * y_unit[i] + J_ref_c[0].z * z_unit[i] + b0.x;
        const FloatType y_mini = J_ref_c[1].x * x_unit[i] + J_ref_c[1].y * y_unit[i] + J_ref_c[1].z * z_unit[i] + b0.y;
        const FloatType z_mini = J_ref_c[2].x * x_unit[i] + J_ref_c[2].y * y_unit[i] + J_ref_c[2].z * z_unit[i] + b0.z;

        tet10_Lagrange_basis_gpu(x_mini, y_mini, z_mini, tet10_f);

        wf_tet10_mini[i] = tet10_f[0] * wf_tet10[0] +  //
                           tet10_f[1] * wf_tet10[1] +  //
                           tet10_f[2] * wf_tet10[2] +  //
                           tet10_f[3] * wf_tet10[3] +  //
                           tet10_f[4] * wf_tet10[4] +  //
                           tet10_f[5] * wf_tet10[5] +  //
                           tet10_f[6] * wf_tet10[6] +  //
                           tet10_f[7] * wf_tet10[7] +  //
                           tet10_f[8] * wf_tet10[8] +  //
                           tet10_f[9] * wf_tet10[9];   //
    }
}

///////////////////////////////////////////////////////////////////////
// tet10_measure
///////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ __inline__ FloatType                                 //
tet10_measure_Float_gpu(const FloatType* const __restrict__ x,  //
                        const FloatType* const __restrict__ y,  //
                        const FloatType* const __restrict__ z,  //
                        const FloatType qx,                     // Quadrature point
                        const FloatType qy,                     //
                        const FloatType qz) {                   //

    const FloatType x0  = 4 * qz;
    const FloatType x1  = x0 - 1;
    const FloatType x2  = 4 * qy;
    const FloatType x3  = 4 * qx;
    const FloatType x4  = x3 - 4;
    const FloatType x5  = -8 * qz - x2 - x4;
    const FloatType x6  = -x3 * y[4];
    const FloatType x7  = x0 + x2;
    const FloatType x8  = x3 + x7 - 3;
    const FloatType x9  = x8 * y[0];
    const FloatType x10 = -x2 * y[6] + x9;
    const FloatType x11 = x1 * y[3] + x10 + x2 * y[9] + x3 * y[8] + x5 * y[7] + x6;
    const FloatType x12 = -x2 * z[6];
    const FloatType x13 = -x0 * z[7];
    const FloatType x14 = x3 - 1;
    const FloatType x15 = x8 * z[0];
    const FloatType x16 = -8 * qx - x7 + 4;
    const FloatType x17 = x0 * z[8] + x12 + x13 + x14 * z[1] + x15 + x16 * z[4] + x2 * z[5];
    const FloatType x18 = x2 - 1;
    const FloatType x19 = -8 * qy - x0 - x4;
    const FloatType x20 = -x3 * x[4];
    const FloatType x21 = x8 * x[0];
    const FloatType x22 = -x0 * x[7] + x21;
    const FloatType x23 = (1.0 / 6.0) * x0 * x[9] + (1.0 / 6.0) * x18 * x[2] + (1.0 / 6.0) * x19 * x[6] + (1.0 / 6.0) * x20 +
                          (1.0 / 6.0) * x22 + (1.0 / 6.0) * x3 * x[5];
    const FloatType x24 = -x0 * y[7];
    const FloatType x25 = x0 * y[8] + x10 + x14 * y[1] + x16 * y[4] + x2 * y[5] + x24;
    const FloatType x26 = x15 - x3 * z[4];
    const FloatType x27 = x1 * z[3] + x12 + x2 * z[9] + x26 + x3 * z[8] + x5 * z[7];
    const FloatType x28 = x0 * y[9] + x18 * y[2] + x19 * y[6] + x24 + x3 * y[5] + x6 + x9;
    const FloatType x29 = -x2 * x[6];
    const FloatType x30 = (1.0 / 6.0) * x1 * x[3] + (1.0 / 6.0) * x2 * x[9] + (1.0 / 6.0) * x20 + (1.0 / 6.0) * x21 +
                          (1.0 / 6.0) * x29 + (1.0 / 6.0) * x3 * x[8] + (1.0 / 6.0) * x5 * x[7];
    const FloatType x31 = x0 * z[9] + x13 + x18 * z[2] + x19 * z[6] + x26 + x3 * z[5];
    const FloatType x32 = (1.0 / 6.0) * x0 * x[8] + (1.0 / 6.0) * x14 * x[1] + (1.0 / 6.0) * x16 * x[4] +
                          (1.0 / 6.0) * x2 * x[5] + (1.0 / 6.0) * x22 + (1.0 / 6.0) * x29;

    return x11 * x17 * x23 - x11 * x31 * x32 - x17 * x28 * x30 - x23 * x25 * x27 + x25 * x30 * x31 + x27 * x28 * x32;
}

////////////////////////////////////////////////////////////////////////
// hex_aa_8_collect_indices
////////////////////////////////////////////////////////////////////////
__device__ __inline__ void                                //
hex_aa_8_collect_indices_gpu(const ptrdiff_t  stride0,    //
                             const ptrdiff_t  stride1,    //
                             const ptrdiff_t  stride2,    //
                             const ptrdiff_t  i,          //
                             const ptrdiff_t  j,          //
                             const ptrdiff_t  k,          //
                             ptrdiff_t* const indices) {  //

    indices[0] = i * stride0 + j * stride1 + k * stride2;
    indices[1] = (i + 1) * stride0 + j * stride1 + k * stride2;
    indices[2] = (i + 1) * stride0 + (j + 1) * stride1 + k * stride2;
    indices[3] = i * stride0 + (j + 1) * stride1 + k * stride2;
    indices[4] = i * stride0 + j * stride1 + (k + 1) * stride2;
    indices[5] = (i + 1) * stride0 + j * stride1 + (k + 1) * stride2;
    indices[6] = (i + 1) * stride0 + (j + 1) * stride1 + (k + 1) * stride2;
    indices[7] = i * stride0 + (j + 1) * stride1 + (k + 1) * stride2;
}

////////////////////////////////////////////////////////////////////////////////
// Hex8 to isoparametric tet10 local adjoint category
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ __inline__ FloatType                                                        //
hex8_to_isoparametric_tet10_local_adjoint_category_gpu(const int        L,             //
                                                       const FloatType* bc,            // transposition vector for category
                                                       const FloatType  J_phys[9],     // Jacobian matrix
                                                       const FloatType  J_fc[9],       // Jacobian matrix
                                                       const FloatType  J_ref[9],      // Jacobian matrix
                                                       const FloatType  det_J_phys,    //
                                                       const geom_t     x[10],         // Tetrahedron vertices X-coordinates
                                                       const geom_t     y[10],         // Tetrahedron vertices Y-coordinates
                                                       const geom_t     z[10],         // Tetrahedron vertices Z-coordinates
                                                       const FloatType  ox,            // Origin of the grid
                                                       const FloatType  oy,            //
                                                       const FloatType  oz,            //
                                                       const FloatType  dx,            // Spacing of the grid
                                                       const FloatType  dy,            //
                                                       const FloatType  dz,            //
                                                       const FloatType  wf_tet10[10],  // Weighted field at the vertices
                                                       const ptrdiff_t  stride0,       // Stride
                                                       const ptrdiff_t  stride1,       //
                                                       const ptrdiff_t  stride2,       //
                                                       FloatType*       data) {
    // mini-tet parameters

    const FloatType N_micro_tet     = static_cast<FloatType>(L) * static_cast<FloatType>(L) * static_cast<FloatType>(L);
    const FloatType inv_N_micro_tet = static_cast<FloatType>(1.0) / N_micro_tet;
    const FloatType theta_volume    = det_J_phys / static_cast<FloatType>(6.0);

    FloatType hex8_f[8];
    FloatType tet10_f[10];

    const FloatType fx0 = static_cast<FloatType>(x[0]);  // Tetrahedron Origin X-coordinate
    const FloatType fy0 = static_cast<FloatType>(y[0]);  // Tetrahedron Origin Y-coordinate
    const FloatType fz0 = static_cast<FloatType>(z[0]);  // Tetrahedron Origin Z-coordinate

    FloatType x_m[10];
    FloatType y_m[10];
    FloatType z_m[10];

    typename Float3<FloatType>::type b0v = {bc[0], bc[1], bc[2]};
    typename Float3<FloatType>::type v0f = {
            static_cast<FloatType>(x[0]), static_cast<FloatType>(y[0]), static_cast<FloatType>(z[0])};

    compute_tet10_phys_mini_gpu(J_fc, J_phys, b0v, v0f, x_m, y_m, z_m);

    FloatType wf_tet10_mini[10];
    compute_wf_tet10_mini_gpu(
            wf_tet10, reinterpret_cast<const typename Float3<FloatType>::type* const>(J_ref), b0v, wf_tet10_mini);
    // If J_ref is already Float3 array, drop the reinterpret_cast.

    for (int quad_i = 0; quad_i < TET_QUAD_NQP; quad_i++) {  // loop over the quadrature points

        const FloatType xq = static_cast<FloatType>(tet_qx[quad_i]);
        const FloatType yq = static_cast<FloatType>(tet_qy[quad_i]);
        const FloatType zq = static_cast<FloatType>(tet_qz[quad_i]);
        const FloatType wq = static_cast<FloatType>(tet_qw[quad_i]);

        FloatType g_qx;
        FloatType g_qy;
        FloatType g_qz;

        tet10_transform_real_t(x_m, y_m, z_m, xq, yq, zq, &g_qx, &g_qy, &g_qz);

        // Basis on (xq,yq,zq)
        tet10_Lagrange_basis_gpu(xq, yq, zq, tet10_f);

        const FloatType grid_x = (g_qx - ox) / dx;
        const FloatType grid_y = (g_qy - oy) / dy;
        const FloatType grid_z = (g_qz - oz) / dz;

        const ptrdiff_t i = floor(grid_x);
        const ptrdiff_t j = floor(grid_y);
        const ptrdiff_t k = floor(grid_z);

        // Get the remainder [0, 1]
        FloatType l_x = (grid_x - static_cast<FloatType>(i));
        FloatType l_y = (grid_y - static_cast<FloatType>(j));
        FloatType l_z = (grid_z - static_cast<FloatType>(k));

        // assert(l_x >= -1e-8);
        // assert(l_y >= -1e-8);
        // assert(l_z >= -1e-8);

        // assert(l_x <= 1 + 1e-8);
        // assert(l_y <= 1 + 1e-8);
        // assert(l_z <= 1 + 1e-8);

        ptrdiff_t indices[8];
        hex_aa_8_collect_indices_gpu(stride0, stride1, stride2, i, j, k, indices);

        hex_aa_8_eval_fun(l_x, l_y, l_z, hex8_f);

        const FloatType measure = tet10_measure_Float_gpu(x_m, y_m, z_m, xq, yq, zq);
        const FloatType dV      = measure * static_cast<FloatType>(tet_qw[quad_i]);  // or use wq

        const FloatType It = (tet10_f[0] * wf_tet10_mini[0] +  //
                              tet10_f[1] * wf_tet10_mini[1] +  //
                              tet10_f[2] * wf_tet10_mini[2] +  //
                              tet10_f[3] * wf_tet10_mini[3] +  //
                              tet10_f[4] * wf_tet10_mini[4] +  //
                              tet10_f[5] * wf_tet10_mini[5] +  //
                              tet10_f[6] * wf_tet10_mini[6] +  //
                              tet10_f[7] * wf_tet10_mini[7] +  //
                              tet10_f[8] * wf_tet10_mini[8] +  //
                              tet10_f[9] * wf_tet10_mini[9]);  //

        const FloatType d0 = It * hex8_f[0] * dV;
        const FloatType d1 = It * hex8_f[1] * dV;
        const FloatType d2 = It * hex8_f[2] * dV;
        const FloatType d3 = It * hex8_f[3] * dV;
        const FloatType d4 = It * hex8_f[4] * dV;
        const FloatType d5 = It * hex8_f[5] * dV;
        const FloatType d6 = It * hex8_f[6] * dV;
        const FloatType d7 = It * hex8_f[7] * dV;

        data[indices[0]] += d0;
        data[indices[1]] += d1;
        data[indices[2]] += d2;
        data[indices[3]] += d3;
        data[indices[4]] += d4;
        data[indices[5]] += d5;
        data[indices[6]] += d6;
        data[indices[7]] += d7;
    }
}

#endif  // __SFEM_ADJOINT_MINI_LOC_TET10_CUH__