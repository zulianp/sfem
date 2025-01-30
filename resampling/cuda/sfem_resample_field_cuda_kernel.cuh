#ifndef SFEM_RESAMPLE_FIELD_CUDA_CUH
#define SFEM_RESAMPLE_FIELD_CUDA_CUH

#include <cooperative_groups.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include "sfem_base.h"

// #define real_t double
#define real_type real_t

#define MY_RESTRICT __restrict__

#include "quadratures_rule_cuda.cuh"
#include "sfem_resample_field_cuda_fun.cuh"

////////////////////////////////////////////////////////
// tet4_transform_v2
////////////////////////////////////////////////////////
__device__ void  //
tet4_transform_cu(
        // X-coordinates
        const real_type px0, const real_type px1, const real_type px2, const real_type px3,
        // Y-coordinates
        const real_type py0, const real_type py1, const real_type py2, const real_type py3,
        // Z-coordinates
        const real_type pz0, const real_type pz1, const real_type pz2, const real_type pz3,
        // Quadrature point
        const real_type qx, const real_type qy, const real_type qz,
        // Output
        real_type* const out_x, real_type* const out_y, real_type* const out_z) {
    //

    /****************************************************************************************
     ****************************************************************************************
    \begin{bmatrix}
    out_x \\
    out_y \\
    out_z
    \end{bmatrix}
    =
    \begin{bmatrix}
    px_0 \\
    py_0 \\
    pz_0
    \end{bmatrix}
    +
    \begin{bmatrix}
    px_1 - px_0 & px_2 - px_0 & px_3 - px_0 \\
    py_1 - py_0 & py_2 - py_0 & py_3 - py_0 \\
    pz_1 - pz_0 & pz_2 - pz_0 & pz_3 - pz_0
    \end{bmatrix}
    \cdot
    \begin{bmatrix}
    qx \\
    qy \\
    qz
    \end{bmatrix}
    *************************************************************************************************
  */

    *out_x = px0 + qx * (-px0 + px1) + qy * (-px0 + px2) + qz * (-px0 + px3);
    *out_y = py0 + qx * (-py0 + py1) + qy * (-py0 + py2) + qz * (-py0 + py3);
    *out_z = pz0 + qx * (-pz0 + pz1) + qy * (-pz0 + pz2) + qz * (-pz0 + pz3);
}

////////////////////////////////////////////////////////
// tet4_measure_v2
////////////////////////////////////////////////////////
__device__ inline real_type             //
tet4_measure_cu(const real_type px0,    // X-coordinates
                const real_type px1,    // X-coordinates
                const real_type px2,    // X-coordinates
                const real_type px3,    // X-coordinates
                const real_type py0,    // Y-coordinates
                const real_type py1,    // Y-coordinates
                const real_type py2,    // Y-coordinates
                const real_type py3,    // Y-coordinates
                const real_type pz0,    // Z-coordinates
                const real_type pz1,    // Z-coordinates
                const real_type pz2,    // Z-coordinates
                const real_type pz3) {  // Z-coordinates
    //
    // determinant of the Jacobian
    // M = [px0, py0, pz0, 1]
    //     [px1, py1, pz1, 1]
    //     [px2, py2, pz2, 1]
    //     [px3, py3, pz3, 1]
    //
    // V = (1/6) * det(M)

    const real_type r1_6 = 1.0 / 6.0;

    const real_type x0 = -pz0 + pz3;
    const real_type x1 = -py0 + py2;
    const real_type x2 = -r1_6 * px0 + r1_6 * px1;
    const real_type x3 = -py0 + py3;
    const real_type x4 = -pz0 + pz2;
    const real_type x5 = -py0 + py1;
    const real_type x6 = -r1_6 * px0 + r1_6 * px2;
    const real_type x7 = -pz0 + pz1;
    const real_type x8 = -r1_6 * px0 + r1_6 * px3;

    return x0 * x1 * x2 - x0 * x5 * x6 - x1 * x7 * x8 - x2 * x3 * x4 + x3 * x6 * x7 + x4 * x5 * x8;
}

////////////////////////////////////////////////////////
// hex_aa_8_eval_fun_V
////////////////////////////////////////////////////////
// Quadrature point (local coordinates)
// With respect to the hat functions of a cube element
// In a local coordinate system
//
__device__ void                                       //
hex_aa_8_eval_fun_cu(const real_t              x,     //
                     const real_t              y,     //
                     const real_t              z,     //
                     real_t* const MY_RESTRICT f0,    // Output
                     real_t* const MY_RESTRICT f1,    // Output
                     real_t* const MY_RESTRICT f2,    // Output
                     real_t* const MY_RESTRICT f3,    // Output
                     real_t* const MY_RESTRICT f4,    // Output
                     real_t* const MY_RESTRICT f5,    // Output
                     real_t* const MY_RESTRICT f6,    // Output
                     real_t* const MY_RESTRICT f7) {  // Output
    //
    const real_t r1 = 1.0;

    *f0 = (r1 - x) * (r1 - y) * (r1 - z);
    *f1 = x * (r1 - y) * (r1 - z);
    *f2 = x * y * (r1 - z);
    *f3 = (r1 - x) * y * (r1 - z);
    *f4 = (r1 - x) * (r1 - y) * z;
    *f5 = x * (r1 - y) * z;
    *f6 = x * y * z;
    *f7 = (r1 - x) * y * z;
}

////////////////////////////////////////////////////////
// hex_aa_8_collect_coeffs_cu
////////////////////////////////////////////////////////
__device__ void                                                      //
hex_aa_8_collect_coeffs_cu(const ptrdiff_t MY_RESTRICT     stride0,  // stride0
                           const ptrdiff_t MY_RESTRICT     stride1,  // stride1
                           const ptrdiff_t MY_RESTRICT     stride2,  // stride2
                           const ptrdiff_t                 i,        //
                           const ptrdiff_t                 j,        //
                           const ptrdiff_t                 k,        //
                           const real_t* const MY_RESTRICT data,  // Attention this is geometric data transformed to solver data!
                           real_t* MY_RESTRICT             out0,  // output
                           real_t* MY_RESTRICT             out1,  //
                           real_t* MY_RESTRICT             out2,  //
                           real_t* MY_RESTRICT             out3,  //
                           real_t* MY_RESTRICT             out4,  //
                           real_t* MY_RESTRICT             out5,  //
                           real_t* MY_RESTRICT             out6,  //
                           real_t* MY_RESTRICT             out7) {            //
    //
    const ptrdiff_t i0 = i * stride0 + j * stride1 + k * stride2;
    const ptrdiff_t i1 = (i + 1) * stride0 + j * stride1 + k * stride2;
    const ptrdiff_t i2 = (i + 1) * stride0 + (j + 1) * stride1 + k * stride2;
    const ptrdiff_t i3 = i * stride0 + (j + 1) * stride1 + k * stride2;
    const ptrdiff_t i4 = i * stride0 + j * stride1 + (k + 1) * stride2;
    const ptrdiff_t i5 = (i + 1) * stride0 + j * stride1 + (k + 1) * stride2;
    const ptrdiff_t i6 = (i + 1) * stride0 + (j + 1) * stride1 + (k + 1) * stride2;
    const ptrdiff_t i7 = i * stride0 + (j + 1) * stride1 + (k + 1) * stride2;

    *out0 = data[i0];
    *out1 = data[i1];
    *out2 = data[i2];
    *out3 = data[i3];
    *out4 = data[i4];
    *out5 = data[i5];
    *out6 = data[i6];
    *out7 = data[i7];
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_local_kernel //////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__ void                                                                      //
tet4_resample_field_local_kernel(const ptrdiff_t                     start_element,  // Mesh
                                 const ptrdiff_t                     end_element,    // Mesh
                                 const ptrdiff_t                     nnodes,         // Mesh
                                 const elems_tet4_device MY_RESTRICT elems,          // Mesh
                                 const xyz_tet4_device MY_RESTRICT   xyz,            // Mesh
                                 const ptrdiff_t MY_RESTRICT         stride0,        // SDF stride0
                                 const ptrdiff_t MY_RESTRICT         stride1,        // SDF stride1
                                 const ptrdiff_t MY_RESTRICT         stride2,        // SDF stride2
                                 const float                         origin_x,       // Origin x
                                 const float                         origin_y,       // Origin y
                                 const float                         origin_z,       // Origin z
                                 const float                         delta_x,        // Delta x
                                 const float                         delta_y,        // Delta y
                                 const float                         delta_z,        // Delta z
                                 const real_type* const MY_RESTRICT  data,           // Data
                                 real_type* const MY_RESTRICT        weighted_field) {      // Output field
    //
    // Thread index
    const ptrdiff_t element_i = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("element_i = %ld\n", element_i);

    if (element_i < start_element || element_i >= end_element) {
        return;
    }

    // weighted_field[element_i] = 11.0;

    ////////////////////////////////////////////////////

    const real_type ox = (real_type)origin_x;
    const real_type oy = (real_type)origin_y;
    const real_type oz = (real_type)origin_z;

    const real_type dx = (real_type)delta_x;
    const real_type dy = (real_type)delta_y;
    const real_type dz = (real_type)delta_z;

    ////////////////////////////////////////////////////

    real_type x0 = 0.0, x1 = 0.0, x2 = 0.0, x3 = 0.0;
    real_type y0 = 0.0, y1 = 0.0, y2 = 0.0, y3 = 0.0;
    real_type z0 = 0.0, z1 = 0.0, z2 = 0.0, z3 = 0.0;

    // real_type hex8_f[8];
    real_type hex8_f0 = 0.0, hex8_f1 = 0.0, hex8_f2 = 0.0, hex8_f3 = 0.0, hex8_f4 = 0.0, hex8_f5 = 0.0, hex8_f6 = 0.0,
              hex8_f7 = 0.0;

    // real_type coeffs[8];
    real_type coeffs0 = 0.0, coeffs1 = 0.0, coeffs2 = 0.0, coeffs3 = 0.0, coeffs4 = 0.0, coeffs5 = 0.0, coeffs6 = 0.0,
              coeffs7 = 0.0;

    // real_type tet4_f[4];
    real_type tet4_f0 = 0.0, tet4_f1 = 0.0, tet4_f2 = 0.0, tet4_f3 = 0.0;

    // real_type element_field[4];
    real_type element_field0 = 0.0, element_field1 = 0.0, element_field2 = 0.0, element_field3 = 0.0;

    // loop over the 4 vertices of the tetrahedron
    int ev[4];
    ev[0] = elems.elems_v0[element_i];
    ev[1] = elems.elems_v1[element_i];
    ev[2] = elems.elems_v2[element_i];
    ev[3] = elems.elems_v3[element_i];

    {
        x0 = xyz.x[ev[0]];
        x1 = xyz.x[ev[1]];
        x2 = xyz.x[ev[2]];
        x3 = xyz.x[ev[3]];

        y0 = xyz.y[ev[0]];
        y1 = xyz.y[ev[1]];
        y2 = xyz.y[ev[2]];
        y3 = xyz.y[ev[3]];

        z0 = xyz.z[ev[0]];
        z1 = xyz.z[ev[1]];
        z2 = xyz.z[ev[2]];
        z3 = xyz.z[ev[3]];
    }

    // Volume of the tetrahedron
    const real_type theta_volume = tet4_measure_cu(x0,
                                                   x1,
                                                   x2,
                                                   x3,
                                                   //
                                                   y0,
                                                   y1,
                                                   y2,
                                                   y3,
                                                   //
                                                   z0,
                                                   z1,
                                                   z2,
                                                   z3);

    ////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
    // loop over the quadrature points
    for (int quad_i = 0; quad_i < TET4_NQP; quad_i++) {  // loop over the quadrature points

        real_type g_qx, g_qy, g_qz;

        tet4_transform_cu(x0,
                          x1,
                          x2,
                          x3,

                          y0,
                          y1,
                          y2,
                          y3,

                          z0,
                          z1,
                          z2,
                          z3,

                          tet4_qx[quad_i],
                          tet4_qy[quad_i],
                          tet4_qz[quad_i],

                          &g_qx,
                          &g_qy,
                          &g_qz);

#ifdef SFEM_RESAMPLE_GAP_DUAL
        // Standard basis function
        {
            tet4_f[0] = 1 - tet4_qx[q] - tet4_qy[q] - tet4_qz[q];
            tet4_f[1] = tet4_qx[q];
            tet4_f[2] = tet4_qy[q];
            tet4_f[2] = tet4_qz[q];
        }
#else
        // DUAL basis function
        {
            const real_type f0 = 1.0 - tet4_qx[quad_i] - tet4_qy[quad_i] - tet4_qz[quad_i];
            const real_type f1 = tet4_qx[quad_i];
            const real_type f2 = tet4_qy[quad_i];
            const real_type f3 = tet4_qz[quad_i];

            const real_type r4 = 4.0;

            tet4_f0 = r4 * f0 - f1 - f2 - f3;
            tet4_f1 = -f0 + r4 * f1 - f2 - f3;
            tet4_f2 = -f0 - f1 + r4 * f2 - f3;
            tet4_f3 = -f0 - f1 - f2 + r4 * f3;
        }
#endif

        const real_type grid_x = (g_qx - ox) / dx;
        const real_type grid_y = (g_qy - oy) / dy;
        const real_type grid_z = (g_qz - oz) / dz;

        const ptrdiff_t i = floor_real_t(grid_x);
        const ptrdiff_t j = floor_real_t(grid_y);
        const ptrdiff_t k = floor_real_t(grid_z);

        // Get the reminder [0, 1]
        real_type l_x = (grid_x - (real_t)i);
        real_type l_y = (grid_y - (real_t)j);
        real_type l_z = (grid_z - (real_t)k);

        // Critical point
        hex_aa_8_eval_fun_cu(l_x, l_y, l_z, &hex8_f0, &hex8_f1, &hex8_f2, &hex8_f3, &hex8_f4, &hex8_f5, &hex8_f6, &hex8_f7);

        hex_aa_8_collect_coeffs_cu(stride0,
                                   stride1,
                                   stride2,
                                   i,
                                   j,
                                   k,
                                   data,
                                   &coeffs0,
                                   &coeffs1,
                                   &coeffs2,
                                   &coeffs3,
                                   &coeffs4,
                                   &coeffs5,
                                   &coeffs6,
                                   &coeffs7);

        // Integrate gap function
        {
            real_type eval_field = 0.0;

            // UNROLL_ZERO
            // for (int edof_j = 0; edof_j < 8; edof_j++) {
            //     eval_field += hex8_f[edof_j] * coeffs[edof_j];
            // }
            eval_field += hex8_f0 * coeffs0;
            eval_field += hex8_f1 * coeffs1;
            eval_field += hex8_f2 * coeffs2;
            eval_field += hex8_f3 * coeffs3;
            eval_field += hex8_f4 * coeffs4;
            eval_field += hex8_f5 * coeffs5;
            eval_field += hex8_f6 * coeffs6;
            eval_field += hex8_f7 * coeffs7;

            // UNROLL_ZERO
            // for (int edof_i = 0; edof_i < 4; edof_i++) {
            //     element_field[edof_i] += eval_field * tet4_f[edof_i] * dV;
            // }  // end edof_i loop

            const real_type dV = theta_volume * tet4_qw[quad_i];
            // dV = 1.0;

            element_field0 += eval_field * tet4_f0 * dV;
            element_field1 += eval_field * tet4_f1 * dV;
            element_field2 += eval_field * tet4_f2 * dV;
            element_field3 += eval_field * tet4_f3 * dV;

        }  // end integrate gap function

    }  // end loop over the quadrature points

    atomicAdd(&weighted_field[ev[0]], element_field0);
    atomicAdd(&weighted_field[ev[1]], element_field1);
    atomicAdd(&weighted_field[ev[2]], element_field2);
    atomicAdd(&weighted_field[ev[3]], element_field3);

}  // end kernel tet4_resample_field_local_CU

__global__ void mykernel() { printf("hello fron kernel\n"); }

double calculate_flops(const ptrdiff_t nelements, const ptrdiff_t quad_nodes, double time_sec) {
    const double flops = (nelements * (35 + 166 * quad_nodes)) / time_sec;
    return flops;
}

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// quadrature_node ///////////////////////////////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
template <typename Real_Type>
__device__ inline void                                              //
quadrature_node(const Real_Type                    tet4_qx_v,       //
                const Real_Type                    tet4_qy_v,       //
                const Real_Type                    tet4_qz_v,       //
                const Real_Type                    tet4_qw_v,       //
                const Real_Type                    theta_volume,    //
                const Real_Type                    x0,              //
                const Real_Type                    x1,              //
                const Real_Type                    x2,              //
                const Real_Type                    x3,              //
                const Real_Type                    y0,              //
                const Real_Type                    y1,              //
                const Real_Type                    y2,              //
                const Real_Type                    y3,              //
                const Real_Type                    z0,              //
                const Real_Type                    z1,              //
                const Real_Type                    z2,              //
                const Real_Type                    z3,              //
                const Real_Type                    dx,              //
                const Real_Type                    dy,              //
                const Real_Type                    dz,              //
                const Real_Type                    ox,              //
                const Real_Type                    oy,              //
                const Real_Type                    oz,              //
                const ptrdiff_t                    stride0,         //
                const ptrdiff_t                    stride1,         //
                const ptrdiff_t                    stride2,         //
                const real_type* const MY_RESTRICT data,            //
                Real_Type&                         element_field0,  //
                Real_Type&                         element_field1,  //
                Real_Type&                         element_field2,  //
                Real_Type&                         element_field3) {                        //
    //
    Real_Type g_qx = 0.0, g_qy = 0.0, g_qz = 0.0;

    // real_type tet4_f[4];
    Real_Type tet4_f0 = 0.0, tet4_f1 = 0.0, tet4_f2 = 0.0, tet4_f3 = 0.0;

    // real_type hex8_f[8];
    Real_Type hex8_f0 = 0.0, hex8_f1 = 0.0, hex8_f2 = 0.0, hex8_f3 = 0.0, hex8_f4 = 0.0, hex8_f5 = 0.0, hex8_f6 = 0.0,
              hex8_f7 = 0.0;

    // real_type coeffs[8];
    Real_Type coeffs0 = 0.0, coeffs1 = 0.0, coeffs2 = 0.0, coeffs3 = 0.0, coeffs4 = 0.0, coeffs5 = 0.0, coeffs6 = 0.0,
              coeffs7 = 0.0;

    // element_field0 = 0.0;
    // element_field1 = 0.0;
    // element_field2 = 0.0;
    // element_field3 = 0.0;

    tet4_transform_cu(x0,
                      x1,
                      x2,
                      x3,

                      y0,
                      y1,
                      y2,
                      y3,

                      z0,
                      z1,
                      z2,
                      z3,

                      tet4_qx_v,
                      tet4_qy_v,
                      tet4_qz_v,

                      &g_qx,
                      &g_qy,
                      &g_qz);

    // DUAL basis function
    {
        const Real_Type r4 = 4.0;
        const Real_Type r1 = 1.0;

        const Real_Type f0 = r1 - tet4_qx_v - tet4_qy_v - tet4_qz_v;
        const Real_Type f1 = tet4_qx_v;
        const Real_Type f2 = tet4_qy_v;
        const Real_Type f3 = tet4_qz_v;

        tet4_f0 = r4 * f0 - f1 - f2 - f3;
        tet4_f1 = -f0 + r4 * f1 - f2 - f3;
        tet4_f2 = -f0 - f1 + r4 * f2 - f3;
        tet4_f3 = -f0 - f1 - f2 + r4 * f3;
    }

    const Real_Type grid_x = (g_qx - ox) / dx;
    const Real_Type grid_y = (g_qy - oy) / dy;
    const Real_Type grid_z = (g_qz - oz) / dz;

    const ptrdiff_t i = floor_real_t(grid_x);
    const ptrdiff_t j = floor_real_t(grid_y);
    const ptrdiff_t k = floor_real_t(grid_z);

    // Get the reminder [0, 1]
    Real_Type l_x = (grid_x - (Real_Type)i);
    Real_Type l_y = (grid_y - (Real_Type)j);
    Real_Type l_z = (grid_z - (Real_Type)k);

    // Critical point
    hex_aa_8_eval_fun_cu(l_x, l_y, l_z, &hex8_f0, &hex8_f1, &hex8_f2, &hex8_f3, &hex8_f4, &hex8_f5, &hex8_f6, &hex8_f7);

    hex_aa_8_collect_coeffs_cu(stride0,
                               stride1,
                               stride2,
                               i,
                               j,
                               k,
                               data,
                               &coeffs0,
                               &coeffs1,
                               &coeffs2,
                               &coeffs3,
                               &coeffs4,
                               &coeffs5,
                               &coeffs6,
                               &coeffs7);

    // Integrate gap function
    {
        real_type eval_field = 0.0;

        // UNROLL_ZERO
        // for (int edof_j = 0; edof_j < 8; edof_j++) {
        //     eval_field += hex8_f[edof_j] * coeffs[edof_j];
        // }
        eval_field += hex8_f0 * coeffs0;
        eval_field += hex8_f1 * coeffs1;
        eval_field += hex8_f2 * coeffs2;
        eval_field += hex8_f3 * coeffs3;
        eval_field += hex8_f4 * coeffs4;
        eval_field += hex8_f5 * coeffs5;
        eval_field += hex8_f6 * coeffs6;
        eval_field += hex8_f7 * coeffs7;

        // UNROLL_ZERO
        // for (int edof_i = 0; edof_i < 4; edof_i++) {
        //     element_field[edof_i] += eval_field * tet4_f[edof_i] * dV;
        // }  // end edof_i loop

        const real_type dV = theta_volume * tet4_qw_v;
        // dV = 1.0;

        element_field0 += eval_field * tet4_f0 * dV;
        element_field1 += eval_field * tet4_f1 * dV;
        element_field2 += eval_field * tet4_f2 * dV;
        element_field3 += eval_field * tet4_f3 * dV;

    }  // end integrate gap function
}  // end quadrature_node function
//////////////////////////////////////////////////////////

#define __WARP_SIZE__ 32

//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
// tet4_resample_field_reduce_local_kernel ///////////////
//////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////
__global__ void                                                                             //
tet4_resample_field_reduce_local_kernel(const ptrdiff_t                     start_element,  //
                                        const ptrdiff_t                     end_element,    //
                                        const ptrdiff_t                     nnodes,         //
                                        const elems_tet4_device MY_RESTRICT elems,          //
                                        const xyz_tet4_device MY_RESTRICT   xyz,            //
                                        const ptrdiff_t                     stride0,        //
                                        const ptrdiff_t                     stride1,        //
                                        const ptrdiff_t                     stride2,        //
                                        const float                         origin_x,       //
                                        const float                         origin_y,       //
                                        const float                         origin_z,       //
                                        const float                         delta_x,        //
                                        const float                         delta_y,        //
                                        const float                         delta_z,        //
                                        const real_type* const MY_RESTRICT  data,           //
                                        real_type* const MY_RESTRICT        weighted_field) {      //// Output

    real_type x0 = 0.0, x1 = 0.0, x2 = 0.0, x3 = 0.0;
    real_type y0 = 0.0, y1 = 0.0, y2 = 0.0, y3 = 0.0;
    real_type z0 = 0.0, z1 = 0.0, z2 = 0.0, z3 = 0.0;

    const real_type ox = (real_type)origin_x;
    const real_type oy = (real_type)origin_y;
    const real_type oz = (real_type)origin_z;

    const real_type dx = (real_type)delta_x;
    const real_type dy = (real_type)delta_y;
    const real_type dz = (real_type)delta_z;

    namespace cg = cooperative_groups;

    cg::thread_block g = cg::this_thread_block();

    const unsigned int element_i = (blockIdx.x * blockDim.x + threadIdx.x) / __WARP_SIZE__;

    if (element_i < start_element || element_i >= end_element) {
        return;
    }

    auto           tile      = cg::tiled_partition<__WARP_SIZE__>(g);
    const unsigned tile_rank = tile.thread_rank();

    // loop over the 4 vertices of the tetrahedron
    int ev[4];
    ev[0] = (elems.elems_v0[element_i]);
    ev[1] = (elems.elems_v1[element_i]);
    ev[2] = (elems.elems_v2[element_i]);
    ev[3] = (elems.elems_v3[element_i]);

    {
        x0 = xyz.x[ev[0]];
        x1 = xyz.x[ev[1]];
        x2 = xyz.x[ev[2]];
        x3 = xyz.x[ev[3]];

        y0 = xyz.y[ev[0]];
        y1 = xyz.y[ev[1]];
        y2 = xyz.y[ev[2]];
        y3 = xyz.y[ev[3]];

        z0 = xyz.z[ev[0]];
        z1 = xyz.z[ev[1]];
        z2 = xyz.z[ev[2]];
        z3 = xyz.z[ev[3]];
    }

    // Volume of the tetrahedron
    const real_type theta_volume = tet4_measure_cu(x0,
                                                   x1,
                                                   x2,
                                                   x3,
                                                   //
                                                   y0,
                                                   y1,
                                                   y2,
                                                   y3,
                                                   //
                                                   z0,
                                                   z1,
                                                   z2,
                                                   z3);

    const size_t nr_warp_loop = (TET4_NQP / __WARP_SIZE__) +                //
                                ((TET4_NQP % __WARP_SIZE__) == 0 ? 0 : 1);  //

    real_type element_field0_reduce = real_t(0.0);
    real_type element_field1_reduce = real_t(0.0);
    real_type element_field2_reduce = real_t(0.0);
    real_type element_field3_reduce = real_t(0.0);

    for (size_t i = 0; i < nr_warp_loop; i++) {
        const size_t q_i = i * size_t(__WARP_SIZE__) + tile_rank;

        // real_type element_field0 = 0.0;
        // real_type element_field1 = 0.0;
        // real_type element_field2 = 0.0;
        // real_type element_field3 = 0.0;

        const real_type tet4_qx_v = (q_i < TET4_NQP) ? tet4_qx[q_i] : tet4_qx[0];
        const real_type tet4_qy_v = (q_i < TET4_NQP) ? tet4_qy[q_i] : tet4_qy[0];
        const real_type tet4_qz_v = (q_i < TET4_NQP) ? tet4_qz[q_i] : tet4_qz[0];
        const real_type tet4_qw_v = (q_i < TET4_NQP) ? tet4_qw[q_i] : 0.0;

        quadrature_node<real_type>(tet4_qx_v,
                                   tet4_qy_v,
                                   tet4_qz_v,
                                   tet4_qw_v,
                                   theta_volume,
                                   x0,
                                   x1,
                                   x2,
                                   x3,
                                   y0,
                                   y1,
                                   y2,
                                   y3,
                                   z0,
                                   z1,
                                   z2,
                                   z3,
                                   dx,
                                   dy,
                                   dz,
                                   ox,
                                   oy,
                                   oz,
                                   stride0,
                                   stride1,
                                   stride2,
                                   data,
                                   // Output: Accumulate the field
                                   element_field0_reduce,
                                   element_field1_reduce,
                                   element_field2_reduce,
                                   element_field3_reduce);
    }

    for (int i = tile.size() / 2; i > 0; i /= 2) {
        element_field0_reduce += tile.shfl_down(element_field0_reduce, i);
        element_field1_reduce += tile.shfl_down(element_field1_reduce, i);
        element_field2_reduce += tile.shfl_down(element_field2_reduce, i);
        element_field3_reduce += tile.shfl_down(element_field3_reduce, i);
    }

    if (tile_rank == 0) {
        atomicAdd(&weighted_field[ev[0]], element_field0_reduce);
        atomicAdd(&weighted_field[ev[1]], element_field1_reduce);
        atomicAdd(&weighted_field[ev[2]], element_field2_reduce);
        atomicAdd(&weighted_field[ev[3]], element_field3_reduce);
    }
}

#endif