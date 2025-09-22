#ifndef __SFEM_ADJOINT_MINI_TET10_CUH__
#define __SFEM_ADJOINT_MINI_TET10_CUH__

#include "sfem_adjoint_mini_tet.cuh"
#include "sfem_resample_field_cuda_fun.cuh"
#include "tet10_resample_field.cuh"

////////////////////////////////////////////////////////////////////////////////
// Compute matrix J_phys * J_ref = J_tot
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ __inline__ void                                                         //
compute_matrix_mult_3x3_gpu(const typename Float3<FloatType>::type* const J_phys,  // Jacobian matrix of the physical tetrahedron
                            const typename Float3<FloatType>::type* const J_ref,   // Jacobian matrix of the reference tetrahedron
                            typename Float3<FloatType>::type*             J_tot) {             // Output Jacobian matrix
    // Row 1
    J_tot[0].x = fast_fma(J_phys[0].y, J_ref[1].x, J_phys[0].x * J_ref[0].x);
    J_tot[0].x = fast_fma(J_phys[0].z, J_ref[2].x, J_tot[0].x);
    J_tot[0].y = fast_fma(J_phys[0].y, J_ref[1].y, J_phys[0].x * J_ref[0].y);
    J_tot[0].y = fast_fma(J_phys[0].z, J_ref[2].y, J_tot[0].y);
    J_tot[0].z = fast_fma(J_phys[0].y, J_ref[1].z, J_phys[0].x * J_ref[0].z);
    J_tot[0].z = fast_fma(J_phys[0].z, J_ref[2].z, J_tot[0].z);

    // Row 2
    J_tot[1].x = fast_fma(J_phys[1].y, J_ref[1].x, J_phys[1].x * J_ref[0].x);
    J_tot[1].x = fast_fma(J_phys[1].z, J_ref[2].x, J_tot[1].x);
    J_tot[1].y = fast_fma(J_phys[1].y, J_ref[1].y, J_phys[1].x * J_ref[0].y);
    J_tot[1].y = fast_fma(J_phys[1].z, J_ref[2].y, J_tot[1].y);
    J_tot[1].z = fast_fma(J_phys[1].y, J_ref[1].z, J_phys[1].x * J_ref[0].z);
    J_tot[1].z = fast_fma(J_phys[1].z, J_ref[2].z, J_tot[1].z);

    // Row 3
    J_tot[2].x = fast_fma(J_phys[2].y, J_ref[1].x, J_phys[2].x * J_ref[0].x);
    J_tot[2].x = fast_fma(J_phys[2].z, J_ref[2].x, J_tot[2].x);
    J_tot[2].y = fast_fma(J_phys[2].y, J_ref[1].y, J_phys[2].x * J_ref[0].y);
    J_tot[2].y = fast_fma(J_phys[2].z, J_ref[2].y, J_tot[2].y);
    J_tot[2].z = fast_fma(J_phys[2].y, J_ref[1].z, J_phys[2].x * J_ref[0].z);
    J_tot[2].z = fast_fma(J_phys[2].z, J_ref[2].z, J_tot[2].z);
    return;
}

////////////////////////////////////////////////////////////////////////////////
// Compute the physical coordinates of the mini-tetrahedra
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ __inline__ void                                                         //
compute_tet10_phys_mini_gpu(const typename Float3<FloatType>::type* const J_fc,    //
                            const typename Float3<FloatType>::type* const J_phys,  //
                            const typename Float3<FloatType>::type        b0,      //
                            const typename Float3<FloatType>::type        v0f,     //
                            FloatType*                                    x_m,     //
                            FloatType*                                    y_m,     //
                            FloatType*                                    z_m) {
    const FloatType x_unit[10] = {0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0};
    const FloatType y_unit[10] = {0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5};
    const FloatType z_unit[10] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5};

    for (int i = 0; i < 10; i++) {
        // x
        FloatType tx = J_fc[0].x * x_unit[i];
        tx           = fast_fma(J_fc[0].y, y_unit[i], tx);
        tx           = fast_fma(J_fc[0].z, z_unit[i], tx);
        tx           = fast_fma(J_phys[0].x, b0.x, tx);
        tx           = fast_fma(J_phys[0].y, b0.y, tx);
        tx           = fast_fma(J_phys[0].z, b0.z, tx);
        x_m[i]       = tx + v0f.x;

        // y
        FloatType ty = J_fc[1].x * x_unit[i];
        ty           = fast_fma(J_fc[1].y, y_unit[i], ty);
        ty           = fast_fma(J_fc[1].z, z_unit[i], ty);
        ty           = fast_fma(J_phys[1].x, b0.x, ty);
        ty           = fast_fma(J_phys[1].y, b0.y, ty);
        ty           = fast_fma(J_phys[1].z, b0.z, ty);
        y_m[i]       = ty + v0f.y;

        // z
        FloatType tz = J_fc[2].x * x_unit[i];
        tz           = fast_fma(J_fc[2].y, y_unit[i], tz);
        tz           = fast_fma(J_fc[2].z, z_unit[i], tz);
        tz           = fast_fma(J_phys[2].x, b0.x, tz);
        tz           = fast_fma(J_phys[2].y, b0.y, tz);
        tz           = fast_fma(J_phys[2].z, b0.z, tz);
        z_m[i]       = tz + v0f.z;
    }
    return;
}

///////////////////////////////////////////////////////////////////////
// tet10_edge_lengths
///////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ __inline__ FloatType                    //
tet10_edge_lengths_gpu(const FloatType* x,         //
                       const FloatType* y,         //
                       const FloatType* z,         //
                       int&             vertex_a,  //
                       int&             vertex_b,  //
                       FloatType*       edge_lengths) {  //

    vertex_a = -1;
    vertex_b = -1;

    FloatType max_length          = 0.0;
    const int edges_pattern[6][2] = {{0, 1},   //
                                     {1, 2},   //
                                     {2, 0},   //
                                     {0, 3},   //
                                     {1, 3},   //
                                     {2, 3}};  //

    for (int i = 0; i < 6; i++) {
        const int i0 = edges_pattern[i][0];
        const int i1 = edges_pattern[i][1];

        const real_t len = sqrt((x[i0] - x[i1]) * (x[i0] - x[i1]) +  //
                                (y[i0] - y[i1]) * (y[i0] - y[i1]) +  //
                                (z[i0] - z[i1]) * (z[i0] - z[i1]));  //

        edge_lengths[i] = len;

        if (len > max_length) {
            vertex_a   = i0;
            vertex_b   = i1;
            max_length = len;
        }
    }

    if (vertex_a > vertex_b) {
        const int tmp = vertex_a;
        vertex_a      = vertex_b;
        vertex_b      = tmp;
    }

    return max_length;
}  // END tet10_edge_lengths

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
__device__ __inline__ void compute_wf_tet10_mini_gpu(const FloatType*                              wf_tet10,  //
                                                     const typename Float3<FloatType>::type* const J_ref_c,   //
                                                     const typename Float3<FloatType>::type        b0,        //
                                                     FloatType*                                    wf_tet10_mini) {
    const FloatType x_unit[10] = {0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0};
    const FloatType y_unit[10] = {0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5};
    const FloatType z_unit[10] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5};

    FloatType tet10_f[10];

    for (int i = 0; i < 10; i++) {
        const FloatType x_mini  = fast_fma(J_ref_c[0].y, y_unit[i], J_ref_c[0].x * x_unit[i]);
        const FloatType x_mini2 = fast_fma(J_ref_c[0].z, z_unit[i], x_mini);
        const FloatType y_mini  = fast_fma(J_ref_c[1].y, y_unit[i], J_ref_c[1].x * x_unit[i]);
        const FloatType y_mini2 = fast_fma(J_ref_c[1].z, z_unit[i], y_mini);
        const FloatType z_mini  = fast_fma(J_ref_c[2].y, y_unit[i], J_ref_c[2].x * x_unit[i]);
        const FloatType z_mini2 = fast_fma(J_ref_c[2].z, z_unit[i], z_mini);

        tet10_Lagrange_basis_gpu(x_mini2 + b0.x, y_mini2 + b0.y, z_mini2 + b0.z, tet10_f);

        FloatType acc = 0;
#pragma unroll
        for (int j = 0; j < 10; ++j) {
            acc = fast_fma(tet10_f[j], wf_tet10[j], acc);
        }
        wf_tet10_mini[i] = acc;
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
    const FloatType x10 = fast_fma(-x2, y[6], x9);
    // x11 = x1*y3 + x10 + x2*y9 + x3*y8 + x5*y7 + x6
    FloatType x11 = x10 + x6;
    x11           = fast_fma(x1, y[3], x11);
    x11           = fast_fma(x2, y[9], x11);
    x11           = fast_fma(x3, y[8], x11);
    x11           = fast_fma(x5, y[7], x11);

    const FloatType x12 = fast_fma(-x2, z[6], static_cast<FloatType>(0));
    const FloatType x13 = fast_fma(-x0, z[7], static_cast<FloatType>(0));
    const FloatType x14 = x3 - 1;
    const FloatType x15 = fast_fma(x8, z[0], static_cast<FloatType>(0));
    const FloatType x16 = -8 * qx - x7 + 4;

    // x17 = x0*z8 + x12 + x13 + x14*z1 + x15 + x16*z4 + x2*z5
    FloatType x17 = 0;
    x17           = fast_fma(x0, z[8], x17);
    x17 += x12;
    x17 += x13;
    x17 = fast_fma(x14, z[1], x17);
    x17 += x15;
    x17 = fast_fma(x16, z[4], x17);
    x17 = fast_fma(x2, z[5], x17);

    const FloatType x18 = x2 - 1;
    const FloatType x19 = -8 * qy - x0 - x4;
    const FloatType x20 = fast_fma(-x3, x[4], static_cast<FloatType>(0));
    const FloatType x21 = x8 * x[0];
    const FloatType x22 = fast_fma(-x0, x[7], x21);

    // x23 = (1/6)*(x0*x9 + x18*x2 + x19*x6 + x20 + x22 + x3*x5)
    const FloatType one_sixth = static_cast<FloatType>(1.0 / 6.0);
    FloatType       sum23     = 0;
    sum23                     = fast_fma(x0, x[9], sum23);
    sum23                     = fast_fma(x18, x[2], sum23);
    sum23                     = fast_fma(x19, x[6], sum23);
    sum23 += x20;
    sum23 += x22;
    sum23               = fast_fma(x3, x[5], sum23);
    const FloatType x23 = one_sixth * sum23;

    const FloatType x24 = fast_fma(-x0, y[7], static_cast<FloatType>(0));

    // x25 = x0*y8 + x10 + x14*y1 + x16*y4 + x2*y5 + x24
    FloatType x25 = x10 + x24;
    x25           = fast_fma(x0, y[8], x25);
    x25           = fast_fma(x14, y[1], x25);
    x25           = fast_fma(x16, y[4], x25);
    x25           = fast_fma(x2, y[5], x25);

    const FloatType x26 = fast_fma(-x3, z[4], x15);

    // x27 = x1*z3 + x12 + x2*z9 + x26 + x3*z8 + x5*z7
    FloatType x27 = 0;
    x27           = fast_fma(x1, z[3], x27);
    x27 += x12;
    x27 = fast_fma(x2, z[9], x27);
    x27 += x26;
    x27 = fast_fma(x3, z[8], x27);
    x27 = fast_fma(x5, z[7], x27);

    // x28 = x0*y9 + x18*y2 + x19*y6 + x24 + x3*y5 + x6 + x9
    FloatType x28 = 0;
    x28           = fast_fma(x0, y[9], x28);
    x28           = fast_fma(x18, y[2], x28);
    x28           = fast_fma(x19, y[6], x28);
    x28 += x24;
    x28 = fast_fma(x3, y[5], x28);
    x28 += x6;
    x28 += x9;

    const FloatType x29 = fast_fma(-x2, x[6], static_cast<FloatType>(0));

    // x30 = (1/6)*(x1*x3 + x2*x9 + x20 + x21 + x29 + x3*x8 + x5*x7)
    FloatType sum30 = 0;
    sum30           = fast_fma(x1, x[3], sum30);
    sum30           = fast_fma(x2, x[9], sum30);
    sum30 += x20;
    sum30 += x21;
    sum30 += x29;
    sum30               = fast_fma(x3, x[8], sum30);
    sum30               = fast_fma(x5, x[7], sum30);
    const FloatType x30 = one_sixth * sum30;

    // x31 = x0*z9 + x13 + x18*z2 + x19*z6 + x26 + x3*z5
    FloatType x31 = 0;
    x31           = fast_fma(x0, z[9], x31);
    x31 += x13;
    x31 = fast_fma(x18, z[2], x31);
    x31 = fast_fma(x19, z[6], x31);
    x31 += x26;
    x31 = fast_fma(x3, z[5], x31);

    // x32 = (1/6)*(x0*x8 + x14*x1 + x16*x4 + x2*x5 + x22 + x29)
    FloatType sum32 = 0;
    sum32           = fast_fma(x0, x[8], sum32);
    sum32           = fast_fma(x14, x[1], sum32);
    sum32           = fast_fma(x16, x[4], sum32);
    sum32           = fast_fma(x2, x[5], sum32);
    sum32 += x22;
    sum32 += x29;
    const FloatType x32 = one_sixth * sum32;

    // Final determinant-like expression using FMAs for accumulation
    FloatType acc = 0;
    acc           = fast_fma((x11 * x17), x23, acc);
    acc           = fast_fma(-(x11 * x31), x32, acc);
    acc           = fast_fma(-(x17 * x28), x30, acc);
    acc           = fast_fma(-(x23 * x25), x27, acc);
    acc           = fast_fma((x25 * x30), x31, acc);
    acc           = fast_fma((x27 * x28), x32, acc);

    return acc;
}

////////////////////////////////////////////////////////////////////////////////
// tet10_transform_gpu
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>
__device__ __inline__ void                                  //
tet10_transform_gpu(const FloatType* const __restrict__ x,  //
                    const FloatType* const __restrict__ y,  //
                    const FloatType* const __restrict__ z,  //
                    const FloatType qx,                     // Quadrature point
                    const FloatType qy,                     //
                    const FloatType qz,                     //
                    FloatType&      out_x,                  // Output physical coordinates
                    FloatType&      out_y,                  //
                    FloatType&      out_z) {                     //
    //
    const FloatType x0  = 4 * qx;
    const FloatType x1  = qy * x0;
    const FloatType x2  = qz * x0;
    const FloatType x3  = 4 * qy;
    const FloatType x4  = qz * x3;
    const FloatType x5  = 2 * qx - 1;
    const FloatType x6  = qx * x5;
    const FloatType x7  = 2 * qy;
    const FloatType x8  = qy * (x7 - 1);
    const FloatType x9  = 2 * qz;
    const FloatType x10 = qz * (x9 - 1);
    const FloatType x11 = -4 * qz - x0 - x3 + 4;
    const FloatType x12 = qx * x11;
    const FloatType x13 = qy * x11;
    const FloatType x14 = qz * x11;
    const FloatType x15 = (-x5 - x7 - x9) * (-qx - qy - qz + 1);

    // FMA-accumulated outputs
    FloatType accx = 0, accy = 0, accz = 0;
    accx = fast_fma(x[0], x15, accx);
    accx = fast_fma(x[1], x6, accx);
    accx = fast_fma(x[2], x8, accx);
    accx = fast_fma(x[3], x10, accx);
    accx = fast_fma(x[4], x12, accx);
    accx = fast_fma(x[5], x1, accx);
    accx = fast_fma(x[6], x13, accx);
    accx = fast_fma(x[7], x14, accx);
    accx = fast_fma(x[8], x2, accx);
    accx = fast_fma(x[9], x4, accx);

    accy = fast_fma(y[0], x15, accy);
    accy = fast_fma(y[1], x6, accy);
    accy = fast_fma(y[2], x8, accy);
    accy = fast_fma(y[3], x10, accy);
    accy = fast_fma(y[4], x12, accy);
    accy = fast_fma(y[5], x1, accy);
    accy = fast_fma(y[6], x13, accy);
    accy = fast_fma(y[7], x14, accy);
    accy = fast_fma(y[8], x2, accy);
    accy = fast_fma(y[9], x4, accy);

    accz = fast_fma(z[0], x15, accz);
    accz = fast_fma(z[1], x6, accz);
    accz = fast_fma(z[2], x8, accz);
    accz = fast_fma(z[3], x10, accz);
    accz = fast_fma(z[4], x12, accz);
    accz = fast_fma(z[5], x1, accz);
    accz = fast_fma(z[6], x13, accz);
    accz = fast_fma(z[7], x14, accz);
    accz = fast_fma(z[8], x2, accz);
    accz = fast_fma(z[9], x4, accz);

    out_x = accx;
    out_y = accy;
    out_z = accz;
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
__device__ __inline__ FloatType                                              //
hex8_to_isoparametric_tet10_local_adjoint_category_gpu(const int        L,   //
                                                       const FloatType* bc,  // transposition vector for category
                                                       const typename Float3<FloatType>::type* J_phys,      // Jacobian matrix
                                                       const typename Float3<FloatType>::type* J_fc,        // Jacobian matrix
                                                       const typename Float3<FloatType>::type* J_ref,       // Jacobian matrix
                                                       const FloatType                         det_J_phys,  //
                                                       const geom_t    x[10],         // Tetrahedron vertices X-coordinates
                                                       const geom_t    y[10],         // Tetrahedron vertices Y-coordinates
                                                       const geom_t    z[10],         // Tetrahedron vertices Z-coordinates
                                                       const FloatType ox,            // Origin of the grid
                                                       const FloatType oy,            //
                                                       const FloatType oz,            //
                                                       const FloatType dx,            // Spacing of the grid
                                                       const FloatType dy,            //
                                                       const FloatType dz,            //
                                                       const FloatType wf_tet10[10],  // Weighted field at the vertices
                                                       const ptrdiff_t stride0,       // Stride
                                                       const ptrdiff_t stride1,       //
                                                       const ptrdiff_t stride2,       //
                                                       FloatType*      data) {
    // mini-tet parameters

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_id   = thread_id % LANES_PER_TILE;

    const ptrdiff_t quad_iterations = TET_QUAD_NQP / LANES_PER_TILE + ((TET_QUAD_NQP % LANES_PER_TILE) ? 1 : 0);
    const ptrdiff_t quad_start      = lane_id * quad_iterations;

    const FloatType N_micro_tet     = static_cast<FloatType>(L) * static_cast<FloatType>(L) * static_cast<FloatType>(L);
    const FloatType inv_N_micro_tet = static_cast<FloatType>(1.0) / N_micro_tet;
    const FloatType theta_volume    = det_J_phys / static_cast<FloatType>(6.0);

    // Per-thread cell-aware accumulation (one-entry cache)
    ptrdiff_t cache_base = -1;  // invalid
    // Relative offsets for the 8 hex nodes (constant for given strides)
    const ptrdiff_t off0 = 0;
    const ptrdiff_t off1 = stride0;
    const ptrdiff_t off2 = stride0 + stride1;
    const ptrdiff_t off3 = stride1;
    const ptrdiff_t off4 = stride2;
    const ptrdiff_t off5 = stride0 + stride2;
    const ptrdiff_t off6 = stride0 + stride1 + stride2;
    const ptrdiff_t off7 = stride1 + stride2;

    FloatType acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    FloatType acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;

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

    FloatType wf_tet10_mini[10];               //
    compute_wf_tet10_mini_gpu(wf_tet10,        //
                              J_ref,           //
                              b0v,             //
                              wf_tet10_mini);  //
    // If J_ref is already Float3 array, drop the reinterpret_cast.

    // for (int quad_i = 0; quad_i < TET_QUAD_NQP; quad_i++) {  // loop over the quadrature points
    for (ptrdiff_t quad_i = 0; quad_i < quad_iterations; quad_i++) {  // loop over the quadrature points

        const ptrdiff_t quad_i_tile = quad_start + quad_i;

        if (quad_i_tile >= TET_QUAD_NQP) continue;

        const FloatType xq = static_cast<FloatType>(tet_qx[quad_i_tile]);
        const FloatType yq = static_cast<FloatType>(tet_qy[quad_i_tile]);
        const FloatType zq = static_cast<FloatType>(tet_qz[quad_i_tile]);
        const FloatType wq = static_cast<FloatType>(tet_qw[quad_i_tile]);

        FloatType g_qx;
        FloatType g_qy;
        FloatType g_qz;

        tet10_transform_gpu(x_m, y_m, z_m, xq, yq, zq, g_qx, g_qy, g_qz);

        // Basis on (xq,yq,zq)
        tet10_Lagrange_basis_gpu(xq, yq, zq, tet10_f);

        const FloatType grid_x = (g_qx - ox) / dx;
        const FloatType grid_y = (g_qy - oy) / dy;
        const FloatType grid_z = (g_qz - oz) / dz;

        const ptrdiff_t i = fast_floor(grid_x);
        const ptrdiff_t j = fast_floor(grid_y);
        const ptrdiff_t k = fast_floor(grid_z);

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

        hex_aa_8_eval_fun_T_gpu(l_x,
                                l_y,
                                l_z,  //
                                hex8_f[0],
                                hex8_f[1],
                                hex8_f[2],
                                hex8_f[3],
                                hex8_f[4],
                                hex8_f[5],
                                hex8_f[6],
                                hex8_f[7]);

        const FloatType measure = tet10_measure_Float_gpu(x_m, y_m, z_m, xq, yq, zq);
        const FloatType dV      = measure * static_cast<FloatType>(wq);  // or use wq

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

        const ptrdiff_t base = i * stride0 + j * stride1 + k * stride2;
        if (base == cache_base) {
            // Same cell as previous iteration: accumulate locally
            acc0 += d0;
            acc1 += d1;
            acc2 += d2;
            acc3 += d3;
            acc4 += d4;
            acc5 += d5;
            acc6 += d6;
            acc7 += d7;
        } else {
            // Flush previous cell if any
            if (cache_base != -1) {
                store_add(&data[cache_base + off0], acc0);
                store_add(&data[cache_base + off1], acc1);
                store_add(&data[cache_base + off2], acc2);
                store_add(&data[cache_base + off3], acc3);
                store_add(&data[cache_base + off4], acc4);
                store_add(&data[cache_base + off5], acc5);
                store_add(&data[cache_base + off6], acc6);
                store_add(&data[cache_base + off7], acc7);

                // printf("data[%ld] = %e\n", (long)(cache_base + off0), (double)data[cache_base + off0]);
            }
            // Start accumulating for the new cell
            cache_base = base;
            acc0       = d0;
            acc1       = d1;
            acc2       = d2;
            acc3       = d3;
            acc4       = d4;
            acc5       = d5;
            acc6       = d6;
            acc7       = d7;
        }
    }

    if (cache_base != -1) {
        store_add(&data[cache_base + off0], acc0);
        store_add(&data[cache_base + off1], acc1);
        store_add(&data[cache_base + off2], acc2);
        store_add(&data[cache_base + off3], acc3);
        store_add(&data[cache_base + off4], acc4);
        store_add(&data[cache_base + off5], acc5);
        store_add(&data[cache_base + off6], acc6);
        store_add(&data[cache_base + off7], acc7);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Hex8 to isoparametric tet10 resample field kernel - multi-threaded version
////////////////////////////////////////////////////////////////////////////////
template <typename FloatType>  //
__global__ void                //                                                                                               //
hex8_to_isoparametric_tet10_resample_field_hyteg_mt_adjoint_kernel(  //
        const ptrdiff_t          start_element,                      // Mesh
        const ptrdiff_t          end_element,                        //
        const ptrdiff_t          nnodes,                             //
        const elems_tet10_device elems,                              //
        const xyz_tet10_device   xyz,                                //
        const ptrdiff_t          n0,                                 // SDF
        const ptrdiff_t          n1,                                 //
        const ptrdiff_t          n2,                                 //
        const ptrdiff_t          stride0,                            //
        const ptrdiff_t          stride1,                            //
        const ptrdiff_t          stride2,                            //
        const geom_t             ox,                                 //
        const geom_t             oy,                                 //
        const geom_t             oz,                                 //
        const geom_t             dx,                                 //
        const geom_t             dy,                                 //
        const geom_t             dz,                                 //
        const FloatType* const __restrict__ weighted_field,          // Input WF
        FloatType* const __restrict__ data,                          // Output
        const mini_tet_parameters_t mini_tet_parameters) {           //
                                                                     //
    const FloatType d_min             = dx < dy ? (dx < dz ? dx : dz) : (dy < dz ? dy : dz);
    const FloatType hexahedron_volume = dx * dy * dz;
    const FloatType inv_dmin          = static_cast<FloatType>(1.0) / d_min;

    int degenerated_tetrahedra_cnt = 0;
    int uniform_refine_cnt         = 0;

    // Unit tetrahedron vertices
    const FloatType x0_unit = 0.0;
    const FloatType x1_unit = 1.0;
    const FloatType x2_unit = 0.0;
    const FloatType x3_unit = 0.0;

    const FloatType y0_unit = 0.0;
    const FloatType y1_unit = 0.0;
    const FloatType y2_unit = 1.0;
    const FloatType y3_unit = 0.0;

    const FloatType z0_unit = 0.0;
    const FloatType z1_unit = 0.0;
    const FloatType z2_unit = 0.0;
    const FloatType z3_unit = 1.0;

    typename Float3<FloatType>::type J_vec_mini[6][3];  // Jacobian matrices for the 6 categories
    typename Float3<FloatType>::type J_fc[6][3];  // Jacobian matrices for the 6 categories of tetrahedra for the physical current
    typename Float3<FloatType>::type J_phy[3];    // Jacobian matrix for the physical current

    FloatType hex8_f[8];
    FloatType tet10_f[10];

    const int tet_id    = (blockIdx.x * blockDim.x + threadIdx.x) / LANES_PER_TILE;
    const int element_i = start_element + tet_id;  // Global element index

    if (element_i < end_element) {
        //
        // ISOPARAMETRIC
        FloatType x[10], y[10], z[10];
        ptrdiff_t ev[10];

        FloatType hex8_f[8];
        FloatType coeffs[8];

        FloatType tet10_f[10];
        // FloatType element_field[10];

        ev[0] = elems.elems_v0[element_i];
        ev[1] = elems.elems_v1[element_i];
        ev[2] = elems.elems_v2[element_i];
        ev[3] = elems.elems_v3[element_i];
        ev[4] = elems.elems_v4[element_i];
        ev[5] = elems.elems_v5[element_i];
        ev[6] = elems.elems_v6[element_i];
        ev[7] = elems.elems_v7[element_i];
        ev[8] = elems.elems_v8[element_i];
        ev[9] = elems.elems_v9[element_i];

// ISOPARAMETRIC
#pragma unroll
        for (int v = 0; v < 10; ++v) {
            x[v] = (FloatType)(xyz.x[ev[v]]);  // x-coordinates
            y[v] = (FloatType)(xyz.y[ev[v]]);  // y-coordinates
            z[v] = (FloatType)(xyz.z[ev[v]]);  // z-coordinates
        }

        const FloatType det_J_phys =                    //
                fast_abs(make_Jacobian_matrix_tet_gpu(  //
                        x[0],                           // Tetrahedron vertices X-coordinates
                        x[1],                           //
                        x[2],                           //
                        x[3],                           //
                        y[0],                           // Tetrahedron vertices Y-coordinates
                        y[1],                           //
                        y[2],                           //
                        y[3],                           //
                        z[0],                           // Tetrahedron vertices Z-coordinates
                        z[1],                           //
                        z[2],                           //
                        z[3],                           // Vertex 3 coordinates
                        J_phy));                        // Output Jacobian matrix

        const FloatType wf_tet10[10] = {weighted_field[ev[0]],
                                        weighted_field[ev[1]],
                                        weighted_field[ev[2]],
                                        weighted_field[ev[3]],
                                        weighted_field[ev[4]],
                                        weighted_field[ev[5]],
                                        weighted_field[ev[6]],
                                        weighted_field[ev[7]],
                                        weighted_field[ev[8]],
                                        weighted_field[ev[9]]};

        FloatType edges_length[6];
        int       vertex_a, vertex_b;

        const FloatType max_edge_len = tet10_edge_lengths_gpu(x,              //
                                                              y,              //
                                                              z,              //
                                                              vertex_a,       //
                                                              vertex_b,       //
                                                              edges_length);  //

        const FloatType alpha = max_edge_len * inv_dmin;  // Aspect ratio of the tetrahedron

        const int L = alpha_to_hyteg_level_gpu(alpha,                                    //
                                               mini_tet_parameters.alpha_min_threshold,  //
                                               mini_tet_parameters.alpha_max_threshold,  //
                                               mini_tet_parameters.min_refinement_L,     //
                                               mini_tet_parameters.max_refinement_L);    //

        const FloatType h = 1.0 / (FloatType)(L);  // Size of the mini-tetrahedra in the reference space

        FloatType theta_volume_main = 0.0;  // Volume of the HyTeg tetrahedron

        // Calculate the Jacobian matrices for the 6 categories of tetrahedra for the reference element
        for (int cat_i = 0; cat_i < 6; cat_i++) {
            // Calculate the Jacobian matrix for the current category
            get_category_Jacobian<FloatType>(cat_i, L, J_vec_mini[cat_i]);

            compute_matrix_mult_3x3_gpu<FloatType>(J_phy,              // Jacobian matrix of the physical tetrahedron
                                                   J_vec_mini[cat_i],  // Jacobian matrix of the reference tetrahedron
                                                   J_fc[cat_i]);       // Output Jacobian matrix

        }  // END of for (int cat_i = 0; cat_i < 6; cat_i++)

        for (int k = 0; k < L + 1; k++) {  // Loop over the refinement levels

            const unsigned int nodes_pes_side = (L - k) + 1;
            // const unsigned int nodes_per_layer = nodes_pes_side * (nodes_pes_side + 1) / 2;

            for (int i = 0; i < nodes_pes_side - 1; i++) {          // Loop over the nodes on the first edge
                for (int j = 0; j < nodes_pes_side - i - 1; j++) {  // Loop over the nodes on the second edge

                    const FloatType b0[3] = {(FloatType)(j)*h,   //
                                             (FloatType)(i)*h,   //
                                             (FloatType)(k)*h};  //

                    {  // BEGIN: Cat 0
                        const unsigned int cat0 = 0;

                        hex8_to_isoparametric_tet10_local_adjoint_category_gpu(  //
                                L,                                               //
                                b0,                                              // Translation vector for category 0
                                J_phy,                                           // Jacobian matrix for the physical current
                                J_fc[cat0],                                      // Jacobian matrix for the physical current
                                J_vec_mini[cat0],                                // Reference Jacobian matrix
                                det_J_phys,                                      // Determinant of the Jacobian matrix
                                x,                                               // Tetrahedron vertices X-coordinates
                                y,                                               //
                                z,                                               // Tetrahedron vertices Z-coordinates
                                ox,                                              // Origin of the grid
                                oy,                                              //
                                oz,                                              //
                                dx,                                              // Spacing of the grid
                                dy,                                              //
                                dz,                                              //
                                wf_tet10,                                        // Weighted field at the vertices
                                stride0,                                         // Stride
                                stride1,                                         //
                                stride2,                                         //
                                data);                                           // Size of the grid
                    }

                    if (j >= 1) {
                        for (int cat_i = 1; cat_i < 5; cat_i++) {
                            hex8_to_isoparametric_tet10_local_adjoint_category_gpu(  //
                                    L,                                               //
                                    b0,                                              // Translation vector for category 0
                                    J_phy,                                           // Jacobian matrix
                                    J_fc[cat_i],                                     // Jacobian matrix for the physical current
                                    J_vec_mini[cat_i],                               // Reference Jacobian matrix
                                    det_J_phys,                                      // Determinant of the Jacobian matrix
                                    x,                                               // Tetrahedron vertices X-coordinates
                                    y,                                               //
                                    z,                                               // Tetrahedron vertices Z-coordinates
                                    ox,                                              // Origin of the grid
                                    oy,                                              //
                                    oz,                                              //
                                    dx,                                              // Spacing of the grid
                                    dy,                                              //
                                    dz,                                              //
                                    wf_tet10,                                        // Weighted field at the vertices
                                    stride0,                                         // Stride
                                    stride1,                                         //
                                    stride2,                                         //
                                    data);                                           // Size of the grid
                        }
                    }

                    {
                        const unsigned int cat5 = 5;
                        if (j >= 1 && i >= 1) {
                            hex8_to_isoparametric_tet10_local_adjoint_category_gpu(  //
                                    L,                                               //
                                    b0,                                              // Translation vector for category 0
                                    J_phy,                                           // Jacobian matrix
                                    J_fc[cat5],                                      // Jacobian matrix for the physical current
                                    J_vec_mini[cat5],                                // Reference Jacobian matrix
                                    det_J_phys,                                      // Determinant of the Jacobian matrix
                                    x,                                               // Tetrahedron vertices X-coordinates
                                    y,                                               //
                                    z,                                               // Tetrahedron vertices Z-coordinates
                                    ox,                                              // Origin of the grid
                                    oy,                                              //
                                    oz,                                              //
                                    dx,                                              // Spacing of the grid
                                    dy,                                              //
                                    dz,                                              //
                                    wf_tet10,                                        // Weighted field at the vertices
                                    stride0,                                         // Stride
                                    stride1,                                         //
                                    stride2,                                         //
                                    data);                                           // Size of the grid
                        }
                    }
                }
            }
        }
    }

    return;
}

extern "C" void  //                                                                                               //
call_hex8_to_isoparametric_tet10_resample_field_hyteg_mt_adjoint_kernel(  //
        const ptrdiff_t      start_element,                               // Mesh
        const ptrdiff_t      end_element,                                 //
        const ptrdiff_t      nelements,                                   //
        const ptrdiff_t      nnodes,                                      //
        const idx_t** const  elems,                                       //
        const geom_t** const xyz,                                         //
        const ptrdiff_t      n0,                                          // SDF
        const ptrdiff_t      n1,                                          //
        const ptrdiff_t      n2,                                          //
        const ptrdiff_t      stride0,                                     //
        const ptrdiff_t      stride1,                                     //
        const ptrdiff_t      stride2,                                     //
        const geom_t         ox,                                          //
        const geom_t         oy,                                          //
        const geom_t         oz,                                          //
        const geom_t         dx,                                          //
        const geom_t         dy,                                          //
        const geom_t         dz,                                          //
        const real_t* const __restrict__ weighted_field,                  // Input WF
        real_t* const __restrict__ data,                                  // Output
        const mini_tet_parameters_t mini_tet_parameters);

#endif  // __SFEM_ADJOINT_MINI_TET10_CUH__