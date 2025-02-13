#ifndef __TET10_RESAMPLE_FIELD_KERNELS_CUH__
#define __TET10_RESAMPLE_FIELD_KERNELS_CUH__

#include "mesh_aura.h"
#include "sfem_cuda_math.cuh"
#include "sfem_mesh.h"

#include "tet10_weno_cuda.cuh"

#include "quadratures_rule_cuda.cuh"
#include "tet10_resample_field.cuh"

#define MY_RESTRICT __restrict__

#define __TET10_TILE_SIZE__ 8
#define WENO_CUDA 1

//-------------------------------------------
/// iso-parametric version
//-------------------------------------------
/**
 * @brief Compute the measure of a tet10 element
 *
 * @param x
 * @param y
 * @param z
 * @param qx
 * @param qy
 * @param qz
 * @return __device__
 */
__device__ real_t                                     //
tet10_measure_cu(const real_t* const MY_RESTRICT x,   //
                 const real_t* const MY_RESTRICT y,   //
                 const real_t* const MY_RESTRICT z,   //
                 const real_t                    qx,  // // Quadrature point x
                 const real_t                    qy,  // // Quadrature point y
                 const real_t                    qz) {                   // // Quadrature point z
    //
    const real_t r1   = 1.0;
    const real_t r3   = 3.0;
    const real_t r4   = 4.0;
    const real_t r8   = 8.0;
    const real_t r1_6 = 1.0 / 6.0;

    const real_t x0  = r4 * qz;
    const real_t x1  = x0 - r1;
    const real_t x2  = r4 * qy;
    const real_t x3  = r4 * qx;
    const real_t x4  = x3 - r4;
    const real_t x5  = -r8 * qz - x2 - x4;
    const real_t x6  = -x3 * y[4];
    const real_t x7  = x0 + x2;
    const real_t x8  = x3 + x7 - r3;
    const real_t x9  = x8 * y[0];
    const real_t x10 = -x2 * y[6] + x9;
    const real_t x11 = x1 * y[3] + x10 + x2 * y[9] + x3 * y[8] + x5 * y[7] + x6;
    const real_t x12 = -x2 * z[6];
    const real_t x13 = -x0 * z[7];
    const real_t x14 = x3 - r1;
    const real_t x15 = x8 * z[0];
    const real_t x16 = -r8 * qx - x7 + r4;
    const real_t x17 = x0 * z[8] + x12 + x13 + x14 * z[1] + x15 + x16 * z[4] + x2 * z[5];
    const real_t x18 = x2 - r1;
    const real_t x19 = -r8 * qy - x0 - x4;
    const real_t x20 = -x3 * x[4];
    const real_t x21 = x8 * x[0];
    const real_t x22 = -x0 * x[7] + x21;
    const real_t x23 = r1_6 * x0 * x[9] + r1_6 * x18 * x[2] + r1_6 * x19 * x[6] + r1_6 * x20 + r1_6 * x22 + r1_6 * x3 * x[5];
    const real_t x24 = -x0 * y[7];
    const real_t x25 = x0 * y[8] + x10 + x14 * y[1] + x16 * y[4] + x2 * y[5] + x24;
    const real_t x26 = x15 - x3 * z[4];
    const real_t x27 = x1 * z[3] + x12 + x2 * z[9] + x26 + x3 * z[8] + x5 * z[7];
    const real_t x28 = x0 * y[9] + x18 * y[2] + x19 * y[6] + x24 + x3 * y[5] + x6 + x9;
    const real_t x29 = -x2 * x[6];
    const real_t x30 =
            r1_6 * x1 * x[3] + r1_6 * x2 * x[9] + r1_6 * x20 + r1_6 * x21 + r1_6 * x29 + r1_6 * x3 * x[8] + r1_6 * x5 * x[7];
    const real_t x31 = x0 * z[9] + x13 + x18 * z[2] + x19 * z[6] + x26 + x3 * z[5];
    const real_t x32 = r1_6 * x0 * x[8] + r1_6 * x14 * x[1] + r1_6 * x16 * x[4] + r1_6 * x2 * x[5] + r1_6 * x22 + r1_6 * x29;
    //
    return x11 * x17 * x23 - x11 * x31 * x32 - x17 * x28 * x30 - x23 * x25 * x27 + x25 * x30 * x31 + x27 * x28 * x32;
}  // end tet10_measure_cu

#define POW2_D(x) ((x) * (x))
#define POW3_D(x) ((x) * (x) * (x))

/**
 * @brief Compute the measure of a tet10 element
 *
 * @param x
 * @param y
 * @param z
 * @param qx
 * @param qy
 * @param qz
 * @return __device__
 */
__device__ void                                                       //
isoparametric_lumped_mass_kernel_hrt_cu(const real_t dV,              // Volume of the element
                                        const real_t qx,              // Quadrature point x
                                        const real_t qy,              // Quadrature point y
                                        const real_t qz,              // Quadrature point z
                                        real_t*      element_diag_0,  //
                                        real_t*      element_diag_1,  //
                                        real_t*      element_diag_2,  //
                                        real_t*      element_diag_3,  //
                                        real_t*      element_diag_4,  //
                                        real_t*      element_diag_5,  //
                                        real_t*      element_diag_6,  //
                                        real_t*      element_diag_7,  //
                                        real_t*      element_diag_8,  //
                                        real_t*      element_diag_9) {     //
    //

    const real_t r5_18  = 5.0 / 18.0;
    const real_t r40_27 = 40.0 / 27.0;

    const real_t x0  = 4 * qx;
    const real_t x1  = qy * qz;
    const real_t x2  = 2 * qx - 1;
    const real_t x3  = qx * x2;
    const real_t x4  = 2 * qy;
    const real_t x5  = qy * (x4 - 1);
    const real_t x6  = 2 * qz;
    const real_t x7  = qz * (x6 - 1);
    const real_t x8  = qx + qy + qz - 1;
    const real_t x9  = qx * x8;
    const real_t x10 = qy * x8;
    const real_t x11 = qz * x8;
    const real_t x12 = x8 * (x2 + x4 + x6);
    const real_t x13 = r5_18 * dV;
    const real_t x14 = POW2_D(qx);
    const real_t x15 = POW2_D(qy);
    const real_t x16 = POW2_D(qz);
    const real_t x17 = 11 * qx;
    const real_t x18 = -qy * x17 + 11 * x11 + x12 + 4 * x7;
    const real_t x19 = -qz * x17 + 11 * x10 + 4 * x5;
    const real_t x20 = r40_27 * dV;
    const real_t x21 = qx * qy;
    const real_t x22 = -24 * qx + 21 * x14 + 4;
    const real_t x23 = -24 * qy + 21 * x15;
    const real_t x24 = qx * qz;
    const real_t x25 = -11 * x1 + 4 * x3 + 11 * x9;
    const real_t x26 = -24 * qz + 21 * x16;

    *element_diag_0 += x12 * x13 * (qy * x0 + qz * x0 + 4 * x1 - x10 - x11 + 10 * x12 + x3 + x5 + x7 - x9);
    *element_diag_1 += qx * x13 * (42 * (POW3_D(qx)) + 14 * qx - 45 * x14 - 1);
    *element_diag_2 += qy * x13 * (42 * (POW3_D(qy)) + 14 * qy - 45 * x15 - 1);
    *element_diag_3 += qz * x13 * (42 * (POW3_D(qz)) + 14 * qz - 45 * x16 - 1);
    *element_diag_4 += -x20 * x9 * (16 * x1 + x18 + x19 + x3 - 46 * x9);
    *element_diag_5 += x20 * x21 * (84 * x21 + x22 + x23);
    *element_diag_6 += -x10 * x20 * (-46 * x10 + x18 + 16 * x24 + x25 + x5);
    *element_diag_7 += -x11 * x20 * (-46 * x11 + x12 + x19 + 16 * x21 + x25 + x7);
    *element_diag_8 += x20 * x24 * (x22 + 84 * x24 + x26);
    *element_diag_9 += x1 * x20 * (84 * x1 + x23 + x26 + 4);

}  // end isoparametric_lumped_mass_kernel_hrt_cu

/**
 * @brief Transform a quadrature point from the reference tet10 element to the physical space
 *
 * @param x
 * @param y
 * @param z
 * @param qx
 * @param qy
 * @param qz
 * @param out_x
 * @param out_y
 * @param out_z
 * @return __device__
 */
__device__ void                                        //
tet10_transform_cu(const real_t* const MY_RESTRICT x,  //
                   const real_t* const MY_RESTRICT y,  //
                   const real_t* const MY_RESTRICT z,  //
                   // Quadrature point
                   const real_t qx,  //
                   const real_t qy,  //
                   const real_t qz,  //
                   // Output
                   real_t* const MY_RESTRICT out_x,    //
                   real_t* const MY_RESTRICT out_y,    //
                   real_t* const MY_RESTRICT out_z) {  //

    const real_t r4 = 4.0;
    const real_t r2 = 2.0;
    const real_t r1 = 1.0;

    const real_t x0  = r4 * qx;
    const real_t x1  = qy * x0;
    const real_t x2  = qz * x0;
    const real_t x3  = r4 * qy;
    const real_t x4  = qz * x3;
    const real_t x5  = r2 * qx - r1;
    const real_t x6  = qx * x5;
    const real_t x7  = r2 * qy;
    const real_t x8  = qy * (x7 - r1);
    const real_t x9  = r2 * qz;
    const real_t x10 = qz * (x9 - r1);
    const real_t x11 = -r4 * qz - x0 - x3 + r4;
    const real_t x12 = qx * x11;
    const real_t x13 = qy * x11;
    const real_t x14 = qz * x11;
    const real_t x15 = (-x5 - x7 - x9) * (-qx - qy - qz + r1);

    *out_x = x[0] * x15 + x[1] * x6 + x[2] * x8 + x[3] * x10 + x[4] * x12 + x[5] * x1 + x[6] * x13 + x[7] * x14 + x[8] * x2 +
             x[9] * x4;
    *out_y = y[0] * x15 + y[1] * x6 + y[2] * x8 + y[3] * x10 + y[4] * x12 + y[5] * x1 + y[6] * x13 + y[7] * x14 + y[8] * x2 +
             y[9] * x4;
    *out_z = z[0] * x15 + z[1] * x6 + z[2] * x8 + z[3] * x10 + z[4] * x12 + z[5] * x1 + z[6] * x13 + z[7] * x14 + z[8] * x2 +
             z[9] * x4;

}  // end tet10_transform_cu

/**
 * @brief Compute the dual basis of the tet10 element
 *
 * @param qx
 * @param qy
 * @param qz
 * @param f
 * @return __device__
 */
__device__ void                             //
tet10_dual_basis_hrt_cu(const real_t  qx,   //
                        const real_t  qy,   //
                        const real_t  qz,   //
                        real_t* const f) {  //

    const real_t r2      = 2.0;
    const real_t r1      = 1.0;
    const real_t r4      = 4.0;
    const real_t r5_18   = 5.0 / 18.0;
    const real_t r10_9   = 10.0 / 9.0;
    const real_t r5_72   = 5.0 / 72.0;
    const real_t r40_27  = 40.0 / 27.0;
    const real_t r115_27 = 115.0 / 27.0;
    const real_t r110_27 = 110.0 / 27.0;
    const real_t r55_54  = 55.0 / 54.0;
    const real_t r10_27  = 10.0 / 27.0;
    const real_t r25_9   = 25.0 / 9.0;
    const real_t r160_27 = 160.0 / 27.0;
    const real_t r460_27 = 460.0 / 27.0;

    const real_t x0  = r2 * qy;
    const real_t x1  = r2 * qz;
    const real_t x2  = r2 * qx - r1;
    const real_t x3  = (-x0 - x1 - x2) * (-qx - qy - qz + r1);
    const real_t x4  = x0 - r1;
    const real_t x5  = r5_18 * qy;
    const real_t x6  = x4 * x5;
    const real_t x7  = x1 - r1;
    const real_t x8  = r5_18 * qz;
    const real_t x9  = x7 * x8;
    const real_t x10 = -r4 * qx - r4 * qy - r4 * qz + r4;
    const real_t x11 = r5_72 * x10;
    const real_t x12 = qy * qz;
    const real_t x13 = qx * x11 + r10_9 * x12 + x6 + x9;
    const real_t x14 = r5_18 * qx;
    const real_t x15 = x14 * x2;
    const real_t x16 = r10_9 * qx;
    const real_t x17 = qy * x11 + qz * x16 + x15;
    const real_t x18 = qy * x16 + qz * x11;
    const real_t x19 = qx * x2;
    const real_t x20 = r5_18 * x3;
    const real_t x21 = qy * x14 + x10 * x8 + x20;
    const real_t x22 = qz * x14 + x10 * x5;
    const real_t x23 = qy * x4;
    const real_t x24 = qz * x5 + x10 * x14;
    const real_t x25 = qz * x7;
    const real_t x26 = r40_27 * x23;
    const real_t x27 = r115_27 * x10;
    const real_t x28 = r110_27 * qx;
    const real_t x29 = -qz * x28;
    const real_t x30 = r55_54 * x10;
    const real_t x31 = -qy * x30;
    const real_t x32 = r10_27 * x19;
    const real_t x33 = r40_27 * x25;
    const real_t x34 = x29 + x31 + x32 + x33;
    const real_t x35 = -qy * x28;
    const real_t x36 = -qz * x30;
    const real_t x37 = r10_27 * x3;
    const real_t x38 = x35 + x36 + x37;
    const real_t x39 = r40_27 * x10;
    const real_t x40 = qx * qy;
    const real_t x41 = -qx * x30 - r110_27 * x12;
    const real_t x42 = r10_27 * x23;
    const real_t x43 = r40_27 * x3;
    const real_t x44 = x42 + x43;
    const real_t x45 = qx * qz;
    const real_t x46 = r40_27 * x19;
    const real_t x47 = x41 + x46;
    const real_t x48 = r10_27 * x25;
    const real_t x49 = x26 + x48;
    const real_t x50 = x29 + x31;
    const real_t x51 = x35 + x36;

    f[0] = x13 + x17 + x18 + r25_9 * x3;
    f[1] = x13 + r25_9 * x19 + x21 + x22;
    f[2] = x17 + x21 + r25_9 * x23 + x24 + x9;
    f[3] = x15 + x18 + x20 + x22 + x24 + r25_9 * x25 + x6;
    f[4] = qx * x27 + r160_27 * x12 + x26 + x34 + x38;
    f[5] = qz * x39 + x34 + r460_27 * x40 + x41 + x44;
    f[6] = qy * x27 + x33 + x38 + x42 + r160_27 * x45 + x47;
    f[7] = qz * x27 + x37 + r160_27 * x40 + x47 + x49 + x50;
    f[8] = qy * x39 + x32 + x41 + x43 + r460_27 * x45 + x49 + x51;
    f[9] = qx * x39 + r460_27 * x12 + x44 + x46 + x48 + x50 + x51;
}  //    end tet10_dual_basis_hrt_cu

/////////////////////////////////////////////////////////////////
// hex_aa_8_eval_fun_cu
/////////////////////////////////////////////////////////////////
__device__ void hex_aa_8_eval_fun_cu(const real_t x,  //
                                     const real_t y,  //
                                     const real_t z,  //
                                     // Output
                                     real_t* const MY_RESTRICT f) {
    //
    // Quadrature point (local coordinates)
    // With respect to the hat functions of a cube element
    // In a local coordinate system

    const real_t r1 = 1.0;

    f[0] = (r1 - x) * (r1 - y) * (r1 - z);
    f[1] = x * (r1 - y) * (r1 - z);
    f[2] = x * y * (r1 - z);
    f[3] = (r1 - x) * y * (r1 - z);
    f[4] = (r1 - x) * (r1 - y) * z;
    f[5] = x * (r1 - y) * z;
    f[6] = x * y * z;
    f[7] = (r1 - x) * y * z;
}  // end hex_aa_8_eval_fun_cu

/////////////////////////////////////////////////////////////////
// hex_aa_8_eval_grad_cu
/////////////////////////////////////////////////////////////////
// Attention this is geometric data transformed to solver data!
__device__ void                                                //
hex_aa_8_collect_coeffs_cu(const ptrdiff_t           stride0,  //
                           const ptrdiff_t           stride1,  //
                           const ptrdiff_t           stride2,  //
                           const ptrdiff_t           i,        //
                           const ptrdiff_t           j,        //
                           const ptrdiff_t           k,        //
                           const real_t* MY_RESTRICT data,     //
                           real_t* MY_RESTRICT       out) {          //
    //
    const ptrdiff_t i0 = i * stride0 + j * stride1 + k * stride2;
    const ptrdiff_t i1 = (i + 1) * stride0 + j * stride1 + k * stride2;
    const ptrdiff_t i2 = (i + 1) * stride0 + (j + 1) * stride1 + k * stride2;
    const ptrdiff_t i3 = i * stride0 + (j + 1) * stride1 + k * stride2;
    const ptrdiff_t i4 = i * stride0 + j * stride1 + (k + 1) * stride2;
    const ptrdiff_t i5 = (i + 1) * stride0 + j * stride1 + (k + 1) * stride2;
    const ptrdiff_t i6 = (i + 1) * stride0 + (j + 1) * stride1 + (k + 1) * stride2;
    const ptrdiff_t i7 = i * stride0 + (j + 1) * stride1 + (k + 1) * stride2;

    out[0] = data[i0];
    out[1] = data[i1];
    out[2] = data[i2];
    out[3] = data[i3];
    out[4] = data[i4];
    out[5] = data[i5];
    out[6] = data[i6];
    out[7] = data[i7];
}  // end hex_aa_8_collect_coeffs_cu

/**
 * @brief Compute the indices of the field for third order interpolation
 *
 * @param stride
 * @param i
 * @param j
 * @param k
 * @param i0 .. i15
 * @return SFEM_INLINE
 */
__device__ void                                                  //
hex_aa_8_indices_O3_cuda(const ptrdiff_t SFEM_RESTRICT stride0,  //
                         const ptrdiff_t SFEM_RESTRICT stride1,  //
                         const ptrdiff_t SFEM_RESTRICT stride2,  //
                         const ptrdiff_t               i,        //
                         const ptrdiff_t               j,        //
                         const ptrdiff_t               k,        //
                         const ptrdiff_t               k_diff,   //
                         ptrdiff_t*                    i0,       // Output
                         ptrdiff_t*                    i1,       // Output
                         ptrdiff_t*                    i2,       // Output
                         ptrdiff_t*                    i3,       // Output
                         ptrdiff_t*                    i4,       // Output
                         ptrdiff_t*                    i5,       // Output
                         ptrdiff_t*                    i6,       // Output
                         ptrdiff_t*                    i7,       // Output
                         ptrdiff_t*                    i8,       // Output
                         ptrdiff_t*                    i9,       // Output
                         ptrdiff_t*                    i10,      // Output
                         ptrdiff_t*                    i11,      // Output
                         ptrdiff_t*                    i12,      // Output
                         ptrdiff_t*                    i13,      // Output
                         ptrdiff_t*                    i14,      // Output
                         ptrdiff_t*                    i15) {                       // Output
    //
    const ptrdiff_t stride_x = stride0;
    const ptrdiff_t stride_y = stride1;
    const ptrdiff_t stride_z = stride2;

    *i0 = (i - 1) * stride_x + (j - 1) * stride_y + (k + k_diff) * stride_z;
    *i1 = (i + 0) * stride_x + (j - 1) * stride_y + (k + k_diff) * stride_z;
    *i2 = (i + 1) * stride_x + (j - 1) * stride_y + (k + k_diff) * stride_z;
    *i3 = (i + 2) * stride_x + (j - 1) * stride_y + (k + k_diff) * stride_z;

    *i4 = (i - 1) * stride_x + (j + 0) * stride_y + (k + k_diff) * stride_z;
    *i5 = (i + 0) * stride_x + (j + 0) * stride_y + (k + k_diff) * stride_z;
    *i6 = (i + 1) * stride_x + (j + 0) * stride_y + (k + k_diff) * stride_z;
    *i7 = (i + 2) * stride_x + (j + 0) * stride_y + (k + k_diff) * stride_z;

    *i8  = (i - 1) * stride_x + (j + 1) * stride_y + (k + k_diff) * stride_z;
    *i9  = (i + 0) * stride_x + (j + 1) * stride_y + (k + k_diff) * stride_z;
    *i10 = (i + 1) * stride_x + (j + 1) * stride_y + (k + k_diff) * stride_z;
    *i11 = (i + 2) * stride_x + (j + 1) * stride_y + (k + k_diff) * stride_z;

    *i12 = (i - 1) * stride_x + (j + 2) * stride_y + (k + k_diff) * stride_z;
    *i13 = (i + 0) * stride_x + (j + 2) * stride_y + (k + k_diff) * stride_z;
    *i14 = (i + 1) * stride_x + (j + 2) * stride_y + (k + k_diff) * stride_z;
    *i15 = (i + 2) * stride_x + (j + 2) * stride_y + (k + k_diff) * stride_z;
}

/**
 * @brief Compute the coefficients of the field for third order interpolation
 *
 * @param stride
 * @param i
 * @param j
 * @param k
 * @param data
 * @param out
 * @return SFEM_INLINE
 */
__device__ void                                                             //
hex_aa_8_collect_coeffs_O3_cuda(const ptrdiff_t SFEM_RESTRICT     stride0,  //
                                const ptrdiff_t SFEM_RESTRICT     stride1,  //
                                const ptrdiff_t SFEM_RESTRICT     stride2,  //
                                const ptrdiff_t                   i,        //
                                const ptrdiff_t                   j,        //
                                const ptrdiff_t                   k,        //
                                const real_t* const SFEM_RESTRICT data,     //
                                real_t* const SFEM_RESTRICT       out) {          //
    //
    ptrdiff_t i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15;

    ptrdiff_t i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31;

    ptrdiff_t i32, i33, i34, i35, i36, i37, i38, i39, i40, i41, i42, i43, i44, i45, i46, i47;

    ptrdiff_t i48, i49, i50, i51, i52, i53, i54, i55, i56, i57, i58, i59, i60, i61, i62, i63;

    hex_aa_8_indices_O3_cuda(stride0,
                             stride1,
                             stride2,
                             i,
                             j,
                             k,
                             -1,
                             &i0,
                             &i1,
                             &i2,
                             &i3,
                             &i4,
                             &i5,
                             &i6,
                             &i7,
                             &i8,
                             &i9,
                             &i10,
                             &i11,
                             &i12,
                             &i13,
                             &i14,
                             &i15);

    hex_aa_8_indices_O3_cuda(stride0,
                             stride1,
                             stride2,
                             i,
                             j,
                             k,
                             0,
                             &i16,
                             &i17,
                             &i18,
                             &i19,
                             &i20,
                             &i21,
                             &i22,
                             &i23,
                             &i24,
                             &i25,
                             &i26,
                             &i27,
                             &i28,
                             &i29,
                             &i30,
                             &i31);

    hex_aa_8_indices_O3_cuda(stride0,
                             stride1,
                             stride2,
                             i,
                             j,
                             k,
                             1,
                             &i32,
                             &i33,
                             &i34,
                             &i35,
                             &i36,
                             &i37,
                             &i38,
                             &i39,
                             &i40,
                             &i41,
                             &i42,
                             &i43,
                             &i44,
                             &i45,
                             &i46,
                             &i47);

    hex_aa_8_indices_O3_cuda(stride0,
                             stride1,
                             stride2,
                             i,
                             j,
                             k,
                             2,
                             &i48,
                             &i49,
                             &i50,
                             &i51,
                             &i52,
                             &i53,
                             &i54,
                             &i55,
                             &i56,
                             &i57,
                             &i58,
                             &i59,
                             &i60,
                             &i61,
                             &i62,
                             &i63);

    out[0]  = data[i0];
    out[1]  = data[i1];
    out[2]  = data[i2];
    out[3]  = data[i3];
    out[4]  = data[i4];
    out[5]  = data[i5];
    out[6]  = data[i6];
    out[7]  = data[i7];
    out[8]  = data[i8];
    out[9]  = data[i9];
    out[10] = data[i10];
    out[11] = data[i11];
    out[12] = data[i12];
    out[13] = data[i13];
    out[14] = data[i14];
    out[15] = data[i15];
    out[16] = data[i16];
    out[17] = data[i17];
    out[18] = data[i18];
    out[19] = data[i19];
    out[20] = data[i20];
    out[21] = data[i21];
    out[22] = data[i22];
    out[23] = data[i23];
    out[24] = data[i24];
    out[25] = data[i25];
    out[26] = data[i26];
    out[27] = data[i27];
    out[28] = data[i28];
    out[29] = data[i29];
    out[30] = data[i30];
    out[31] = data[i31];
    out[32] = data[i32];
    out[33] = data[i33];
    out[34] = data[i34];
    out[35] = data[i35];
    out[36] = data[i36];
    out[37] = data[i37];
    out[38] = data[i38];
    out[39] = data[i39];
    out[40] = data[i40];
    out[41] = data[i41];
    out[42] = data[i42];
    out[43] = data[i43];
    out[44] = data[i44];
    out[45] = data[i45];
    out[46] = data[i46];
    out[47] = data[i47];
    out[48] = data[i48];
    out[49] = data[i49];
    out[50] = data[i50];
    out[51] = data[i51];
    out[52] = data[i52];
    out[53] = data[i53];
    out[54] = data[i54];
    out[55] = data[i55];
    out[56] = data[i56];
    out[57] = data[i57];
    out[58] = data[i58];
    out[59] = data[i59];
    out[60] = data[i60];
    out[61] = data[i61];
    out[62] = data[i62];
    out[63] = data[i63];
}

////////////////////////////////////////////////////////////////////////
// hex_aa_8_eval_weno4_3D
////////////////////////////////////////////////////////////////////////
__device__ real_t                                                       //
hex_aa_8_eval_weno4_3D_cuda(const real_t                      x_,       //
                            const real_t                      y_,       //
                            const real_t                      z_,       //
                            const real_t                      ox,       //
                            const real_t                      oy,       //
                            const real_t                      oz,       //
                            const real_t                      h,        //
                            const ptrdiff_t                   i,        //
                            const ptrdiff_t                   j,        //
                            const ptrdiff_t                   k,        //
                            const ptrdiff_t                   stride0,  //
                            const ptrdiff_t                   stride1,  //
                            const ptrdiff_t                   stride2,  //
                            const real_t* const SFEM_RESTRICT data) {   //

    real_t out[64];
    hex_aa_8_collect_coeffs_O3_cuda(stride0, stride1, stride2, i, j, k, data, out);

    real_t x = (x_ - ox) - (real_t)i * h + h;
    real_t y = (y_ - oy) - (real_t)j * h + h;
    real_t z = (z_ - oz) - (real_t)k * h + h;

    const real_t w4 = weno4_3D_ConstH_cuda(x,  //
                                           y,
                                           z,
                                           h,
                                           out,
                                           1,
                                           4,
                                           16);

    return w4;
}

/**
 * @brief Resample a field from a hex8 mesh to a tet10 mesh
 *
 */
__global__ void                                                                                    //
hex8_to_isoparametric_tet10_resample_field_local_reduce_kernel(const ptrdiff_t     start_element,  // start element Mesh
                                                               const ptrdiff_t     end_element,    // end element Mesh
                                                               const ptrdiff_t     nnodes,         // number of nodes in the mesh
                                                               elems_tet10_device  elems,          // connectivity of the mesh
                                                               xyz_tet10_device    xyz,            // coordinates of the mesh
                                                               const ptrdiff_t     nx,  // number of nodes in each direction x SDF
                                                               const ptrdiff_t     ny,  // number of nodes in each direction y SDF
                                                               const ptrdiff_t     nz,  // number of nodes in each direction z SDF
                                                               const ptrdiff_t     stride0,     // stride of the data SDF
                                                               const ptrdiff_t     stride1,     // stride of the data SDF
                                                               const ptrdiff_t     stride2,     // stride of the data SDF
                                                               const geom_t        originx,     // origin of the domain SDF
                                                               const geom_t        originy,     // origin of the domain SDF
                                                               const geom_t        originz,     // origin of the domain SDF
                                                               const geom_t        deltax,      // delta of the domain SDF
                                                               const geom_t        deltay,      // delta of the domain SDF
                                                               const geom_t        deltaz,      // delta of the domain SDF
                                                               const real_t* const data,        // SDF data
                                                               real_t* const       weighted_field) {  // Output field
    //
    // printf("============================================================\n");
    // printf("Start: hex8_to_isoparametric_tet10_resample_field_local_reduce_kernel\n");
    // printf("============================================================\n");

    const real_t ox = (real_t)originx;
    const real_t oy = (real_t)originy;
    const real_t oz = (real_t)originz;

    const real_t dx = (real_t)deltax;
    const real_t dy = (real_t)deltay;
    const real_t dz = (real_t)deltaz;

    ////////////////////////////////////////
    // Kernel specific variables

    namespace cg               = cooperative_groups;
    cg::thread_block g         = cg::this_thread_block();
    auto             tile      = cg::tiled_partition<__TET10_TILE_SIZE__>(g);
    const unsigned   tile_rank = tile.thread_rank();

    const ptrdiff_t element_i = (blockIdx.x * blockDim.x + threadIdx.x) / __TET10_TILE_SIZE__;

    if (element_i < start_element or element_i >= end_element) return;

    ////////////////////////////////////////
    // Quadrature points
    ptrdiff_t ev[10];

    // ISOPARAMETRIC
    real_t x[10], y[10], z[10];

    real_t hex8_f[8];
    real_t coeffs[8];

    real_t tet10_f[10];

    // loop over the ndes of the element
    ev[0] = __ldg(&elems.elems_v0[element_i]);
    ev[1] = __ldg(&elems.elems_v1[element_i]);
    ev[2] = __ldg(&elems.elems_v2[element_i]);
    ev[3] = __ldg(&elems.elems_v3[element_i]);
    ev[4] = __ldg(&elems.elems_v4[element_i]);
    ev[5] = __ldg(&elems.elems_v5[element_i]);
    ev[6] = __ldg(&elems.elems_v6[element_i]);
    ev[7] = __ldg(&elems.elems_v7[element_i]);
    ev[8] = __ldg(&elems.elems_v8[element_i]);
    ev[9] = __ldg(&elems.elems_v9[element_i]);

    if (tile_rank == 0) {
#pragma unroll
        for (int v = 0; v < 10; ++v) {
            // ISOPARAMETRIC
            x[v] = __ldg(&xyz.x[ev[v]]);  // x-coordinates
            y[v] = __ldg(&xyz.y[ev[v]]);  // y-coordinates
            z[v] = __ldg(&xyz.z[ev[v]]);  // z-coordinates
        }
    }

    tile.sync();
    for (int v = 0; v < 10; ++v) {
        // ISOPARAMETRIC
        x[v] = tile.shfl(x[v], 0);
        y[v] = tile.shfl(y[v], 0);
        z[v] = tile.shfl(z[v], 0);
    }

    // SUBPARAMETRIC (for iso-parametric tassellation of tet10 might be necessary)

    real_t element_field_v0_reduce = 0.0;
    real_t element_field_v1_reduce = 0.0;
    real_t element_field_v2_reduce = 0.0;
    real_t element_field_v3_reduce = 0.0;
    real_t element_field_v4_reduce = 0.0;
    real_t element_field_v5_reduce = 0.0;
    real_t element_field_v6_reduce = 0.0;
    real_t element_field_v7_reduce = 0.0;
    real_t element_field_v8_reduce = 0.0;
    real_t element_field_v9_reduce = 0.0;

    const size_t nr_warp_loop = (TET_QUAD_NQP / __TET10_TILE_SIZE__) +                //
                                ((TET_QUAD_NQP % __TET10_TILE_SIZE__) == 0 ? 0 : 1);  //

    for (size_t warp_i = 0; warp_i < nr_warp_loop; warp_i++) {
        //
        const size_t q_i = warp_i * size_t(__TET10_TILE_SIZE__) + tile_rank;

        const real_t tet4_qx_v = (q_i < TET_QUAD_NQP) ? tet4_qx[q_i] : tet4_qx[0];
        const real_t tet4_qy_v = (q_i < TET_QUAD_NQP) ? tet4_qy[q_i] : tet4_qy[0];
        const real_t tet4_qz_v = (q_i < TET_QUAD_NQP) ? tet4_qz[q_i] : tet4_qz[0];
        const real_t tet4_qw_v = (q_i < TET_QUAD_NQP) ? tet4_qw[q_i] : real_t(0.0);

        const real_t measure = tet10_measure_cu(x, y, z, tet4_qx_v, tet4_qy_v, tet4_qz_v);

        // assert(measure > 0);
        const real_t dV = measure * tet4_qw_v;
        // printf("dV[%d]: %e\n", q, dV);

        real_t g_qx, g_qy, g_qz;
        // Transform quadrature point to physical space
        // g_qx, g_qy, g_qz are the coordinates of the quadrature point in the physical
        // space
        tet10_transform_cu(x,          //
                           y,          //
                           z,          //
                           tet4_qx_v,  //
                           tet4_qy_v,
                           tet4_qz_v,
                           &g_qx,
                           &g_qy,
                           &g_qz);

        tet10_dual_basis_hrt_cu(tet4_qx_v, tet4_qy_v, tet4_qz_v, tet10_f);

        ///// ======================================================

        const real_t grid_x = (g_qx - ox) / dx;
        const real_t grid_y = (g_qy - oy) / dy;
        const real_t grid_z = (g_qz - oz) / dz;

        const ptrdiff_t i = floor_real_t(grid_x);
        const ptrdiff_t j = floor_real_t(grid_y);
        const ptrdiff_t k = floor_real_t(grid_z);

        // If outside
        // if (i < 0 || j < 0 || k < 0 || (i + 1 >= n[0]) || (j + 1 >= n[1]) || (k + 1 >= n[2])) {
        //     fprintf(stderr,
        //             "warning (%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, "
        //             "%ld)!\n",
        //             g_qx,
        //             g_qy,
        //             g_qz,
        //             i,
        //             j,
        //             k,
        //             n[0],
        //             n[1],
        //             n[2]);
        //     continue;
        // }

        // Get the reminder [0, 1]
        real_t l_x = (grid_x - (real_t)(i));
        real_t l_y = (grid_y - (real_t)(j));
        real_t l_z = (grid_z - (real_t)(k));

        // assert(l_x >= -1e-8);
        // assert(l_y >= -1e-8);
        // assert(l_z >= -1e-8);

        // assert(l_x <= 1 + 1e-8);
        // assert(l_y <= 1 + 1e-8);
        // assert(l_z <= 1 + 1e-8);

        hex_aa_8_eval_fun_cu(l_x, l_y, l_z, hex8_f);
        hex_aa_8_collect_coeffs_cu(stride0, stride1, stride2, i, j, k, data, coeffs);

        // #define WENO_CUDA 1
        // Integrate field
        {
            // #if WENO_CUDA == 7777
            // printf("WENO_CUDA == 777\n");
            //             real_t eval_field = hex_aa_8_eval_weno4_3D_cuda(g_qx,
            //                                                             g_qy,
            //                                                             g_qz,  //
            //                                                             ox,
            //                                                             oy,
            //                                                             oz, // (dx + dy + dz) *
            //                                                             (1.0 / 3.0),  // i, j, k,
            //                                                             stride0,
            //                                                             stride1,
            //                                                             stride2,
            //                                                             data);
            // #else

            real_t eval_field = real_t(0.0);

            for (int edof_j = 0; edof_j < 8; edof_j++) {
                eval_field += hex8_f[edof_j] * coeffs[edof_j];
            }

            // eval_field = sin (grid_x) * cos (grid_y) * sin (grid_z); ///////////////////// DEBUG

            // #endif
            // // UNROLL_ZERO?
            // for (int edof_i = 0; edof_i < 10; edof_i++) {
            //     element_field[edof_i] += eval_field * tet10_f[edof_i] * dV;
            // }  // end edof_i loop

            element_field_v0_reduce += eval_field * tet10_f[0] * dV;
            element_field_v1_reduce += eval_field * tet10_f[1] * dV;
            element_field_v2_reduce += eval_field * tet10_f[2] * dV;
            element_field_v3_reduce += eval_field * tet10_f[3] * dV;
            element_field_v4_reduce += eval_field * tet10_f[4] * dV;
            element_field_v5_reduce += eval_field * tet10_f[5] * dV;
            element_field_v6_reduce += eval_field * tet10_f[6] * dV;
            element_field_v7_reduce += eval_field * tet10_f[7] * dV;
            element_field_v8_reduce += eval_field * tet10_f[8] * dV;
            element_field_v9_reduce += eval_field * tet10_f[9] * dV;
        }
    }  // end quadrature loop

    for (int i = tile.size() / 2; i > 0; i /= 2) {
        element_field_v0_reduce += tile.shfl_down(element_field_v0_reduce, i);
        element_field_v1_reduce += tile.shfl_down(element_field_v1_reduce, i);
        element_field_v2_reduce += tile.shfl_down(element_field_v2_reduce, i);
        element_field_v3_reduce += tile.shfl_down(element_field_v3_reduce, i);
        element_field_v4_reduce += tile.shfl_down(element_field_v4_reduce, i);
        element_field_v5_reduce += tile.shfl_down(element_field_v5_reduce, i);
        element_field_v6_reduce += tile.shfl_down(element_field_v6_reduce, i);
        element_field_v7_reduce += tile.shfl_down(element_field_v7_reduce, i);
        element_field_v8_reduce += tile.shfl_down(element_field_v8_reduce, i);
        element_field_v9_reduce += tile.shfl_down(element_field_v9_reduce, i);
    }

    if (tile_rank == 0) {
        atomicAdd(&weighted_field[ev[0]], element_field_v0_reduce);
        atomicAdd(&weighted_field[ev[1]], element_field_v1_reduce);
        atomicAdd(&weighted_field[ev[2]], element_field_v2_reduce);
        atomicAdd(&weighted_field[ev[3]], element_field_v3_reduce);
        atomicAdd(&weighted_field[ev[4]], element_field_v4_reduce);
        atomicAdd(&weighted_field[ev[5]], element_field_v5_reduce);
        atomicAdd(&weighted_field[ev[6]], element_field_v6_reduce);
        atomicAdd(&weighted_field[ev[7]], element_field_v7_reduce);
        atomicAdd(&weighted_field[ev[8]], element_field_v8_reduce);
        atomicAdd(&weighted_field[ev[9]], element_field_v9_reduce);
    }

}  // end kernel hex8_to_isoparametric_tet10_resample_field_local_reduce_kernel

///////////////////////////////////////////////////////////////////////
// subparametric_tet10_assemble_dual_mass_vector
///////////////////////////////////////////////////////////////////////
__global__ void                                                                         //
isoparametric_tet10_assemble_dual_mass_vector_kernel(const ptrdiff_t    start_element,  // start element
                                                     const ptrdiff_t    end_element,    // end element
                                                     const ptrdiff_t    nnodes,         // number of nodes
                                                     elems_tet10_device elems,          // connectivity
                                                     xyz_tet10_device   xyz,            // coordinates
                                                     real_t*            diag) {                    // Output diagonal
    //

    // for (ptrdiff_t i = 0; i < nelements; ++i)

    ////////////////////////////////////////
    // Kernel specific variables
    namespace cg = cooperative_groups;

    cg::thread_block g = cg::this_thread_block();

    const ptrdiff_t element_i = (blockIdx.x * blockDim.x + threadIdx.x) / __TET10_TILE_SIZE__;

    if (element_i < start_element or element_i >= end_element) return;

    auto           tile      = cg::tiled_partition<__TET10_TILE_SIZE__>(g);
    const unsigned tile_rank = tile.thread_rank();

    {
        idx_t ev[10];  // Element indices

        real_t element_diag_0;
        real_t element_diag_1;
        real_t element_diag_2;
        real_t element_diag_3;
        real_t element_diag_4;
        real_t element_diag_5;
        real_t element_diag_6;
        real_t element_diag_7;
        real_t element_diag_8;
        real_t element_diag_9;

        real_t x[10], y[10], z[10];

        // for (int v = 0; v < 10; ++v) {
        //     ev[v] = elems[v][element_i];
        // }

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

        for (int v = 0; v < 10; ++v) {
            x[v] = xyz.x[ev[v]];  // x-coordinates
            y[v] = xyz.y[ev[v]];  // y-coordinates
            z[v] = xyz.z[ev[v]];  // z-coordinates
        }

        /// intialize the element_diag to zero
        element_diag_0 = 0.0;
        element_diag_1 = 0.0;
        element_diag_2 = 0.0;
        element_diag_3 = 0.0;
        element_diag_4 = 0.0;
        element_diag_5 = 0.0;
        element_diag_6 = 0.0;
        element_diag_7 = 0.0;
        element_diag_8 = 0.0;
        element_diag_9 = 0.0;

        // We do this numerical integration due to the det J
        const size_t nr_warp_loop = (TET_QUAD_NQP / __TET10_TILE_SIZE__) +                //
                                    ((TET_QUAD_NQP % __TET10_TILE_SIZE__) == 0 ? 0 : 1);  //

        for (size_t warp_i = 0; warp_i < nr_warp_loop; warp_i++) {
            // loop over the quadrature points

            const size_t q_i = warp_i * size_t(__TET10_TILE_SIZE__) + tile_rank;

            const real_t tet4_qx_v = (q_i < TET_QUAD_NQP) ? tet4_qx[q_i] : tet4_qx[0];
            const real_t tet4_qy_v = (q_i < TET_QUAD_NQP) ? tet4_qy[q_i] : tet4_qy[0];
            const real_t tet4_qz_v = (q_i < TET_QUAD_NQP) ? tet4_qz[q_i] : tet4_qz[0];
            const real_t tet4_qw_v = (q_i < TET_QUAD_NQP) ? tet4_qw[q_i] : real_t(0.0);

            const real_t dV = tet10_measure_cu(x, y, z, tet4_qx_v, tet4_qy_v, tet4_qz_v) * tet4_qw_v;

            isoparametric_lumped_mass_kernel_hrt_cu(dV,
                                                    // Quadrature
                                                    tet4_qx_v,
                                                    tet4_qy_v,
                                                    tet4_qz_v,
                                                    &element_diag_0,
                                                    &element_diag_1,
                                                    &element_diag_2,
                                                    &element_diag_3,
                                                    &element_diag_4,
                                                    &element_diag_5,
                                                    &element_diag_6,
                                                    &element_diag_7,
                                                    &element_diag_8,
                                                    &element_diag_9);
        }  // end quadrature loop

        for (int i = tile.size() / 2; i > 0; i /= 2) {
            element_diag_0 += tile.shfl_down(element_diag_0, i);
            element_diag_1 += tile.shfl_down(element_diag_1, i);
            element_diag_2 += tile.shfl_down(element_diag_2, i);
            element_diag_3 += tile.shfl_down(element_diag_3, i);
            element_diag_4 += tile.shfl_down(element_diag_4, i);
            element_diag_5 += tile.shfl_down(element_diag_5, i);
            element_diag_6 += tile.shfl_down(element_diag_6, i);
            element_diag_7 += tile.shfl_down(element_diag_7, i);
            element_diag_8 += tile.shfl_down(element_diag_8, i);
            element_diag_9 += tile.shfl_down(element_diag_9, i);
        }

        if (tile_rank == 0) {
            atomicAdd(&diag[ev[0]], element_diag_0);
            atomicAdd(&diag[ev[1]], element_diag_1);
            atomicAdd(&diag[ev[2]], element_diag_2);
            atomicAdd(&diag[ev[3]], element_diag_3);
            atomicAdd(&diag[ev[4]], element_diag_4);
            atomicAdd(&diag[ev[5]], element_diag_5);
            atomicAdd(&diag[ev[6]], element_diag_6);
            atomicAdd(&diag[ev[7]], element_diag_7);
            atomicAdd(&diag[ev[8]], element_diag_8);
            atomicAdd(&diag[ev[9]], element_diag_9);
        }
    }  // end element block

}  // end isoparametric_tet10_assemble_dual_mass_vector

/**
 * @brief Cuda kernel to compute the mass vector
 *
 * @param nnodes number of nodes
 * @param weighted_field the weighted field
 * @param mass_vector the mass vector
 * @param g the resulting vector
 * @return void
 */
__global__ void                                 //
compute_g_kernel(ptrdiff_t     nnodes,          //
                 const real_t* weighted_field,  //
                 const real_t* mass_vector,     //
                 real_t*       g) {                   //
                                                //
    ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnodes) {
        assert(mass_vector[i] != 0);
        g[i] = weighted_field[i] / mass_vector[i];
        // g[i] = mass_vector[i]
    }
}

/**
 * @brief Cuda kernel to compute the mass vector
 *
 * @param nnodes
 * @param mass_vector
 * @param g_wf
 * @return __global__
 */
__global__ void                                 //
compute_g_kernel_v2(ptrdiff_t     nnodes,       //
                    const real_t* mass_vector,  //
                    real_t*       g_wf) {             //
                                                //
    ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnodes) {
        assert(mass_vector[i] != 0);
        g_wf[i] /= mass_vector[i];
        // g[i] = mass_vector[i]
    }
}

/**
 * @brief Set the array to zero
 *
 * @param n
 * @param a
 */
__global__ void zeros_kernel(const ptrdiff_t n,  //
                             real_t*         a) {        //

    const ptrdiff_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 0.0;
}

/**
 * @brief Compute the indices of the field for third order interpolation
 *
 * @param stride
 * @param i
 * @param j
 * @param k
 * @return __device__
 */
__device__ ptrdiff_t                                           //
hex_aa_8_indices_O3_first_index_cuda(const ptrdiff_t stride0,  //
                                     const ptrdiff_t stride1,  //
                                     const ptrdiff_t stride2,  //
                                     const ptrdiff_t i,        //
                                     const ptrdiff_t j,        //
                                     const ptrdiff_t k) {      //
    //
    return (i - 1) * stride0 + (j - 1) * stride1 + (k - 1) * stride2;
}

/**
 * @brief Compute the indices of the field for third order interpolation
 *
 * @param stride
 * @param i
 * @param j
 * @param k
 * @param data
 * @return __device__*
 */
__device__ real_t*                                                //
hex_aa_8_collect_coeffs_O3_ptr_cuda(const ptrdiff_t     stride0,  //
                                    const ptrdiff_t     stride1,  //
                                    const ptrdiff_t     stride2,  //
                                    const ptrdiff_t     i,        //
                                    const ptrdiff_t     j,        //
                                    const ptrdiff_t     k,        //
                                    const real_t* const data) {   //

    const ptrdiff_t first_index = hex_aa_8_indices_O3_first_index_cuda(stride0, stride1, stride2, i, j, k);

    return (real_t*)&data[first_index];
}

////////////////////////////////////////////////////////////////////////
// hex_aa_8_eval_weno4_3D
////////////////////////////////////////////////////////////////////////
__device__ real_t                                                            //
hex_aa_8_eval_weno4_3D_Unit_cuda(const real_t                      x_unit,   //
                                 const real_t                      y_unit,   //
                                 const real_t                      z_unit,   //
                                 const real_t                      ox_unit,  //
                                 const real_t                      oy_unit,  //
                                 const real_t                      oz_unit,  //
                                 const ptrdiff_t                   i,        // it must be the absulte index
                                 const ptrdiff_t                   j,        // Used to retrive the data
                                 const ptrdiff_t                   k,        // From the data array
                                 const ptrdiff_t                   stride0,  //
                                 const ptrdiff_t                   stride1,  //
                                 const ptrdiff_t                   stride2,  //
                                 const real_t* const SFEM_RESTRICT data) {   //

#define WENO_DIRECT_CUDA 1

#if WENO_DIRECT_CUDA == 1
#pragma message "WENO_DIRECT_CUDA is enabled"

    const int stride_x = stride0;
    const int stride_y = stride1;
    const int stride_z = stride2;

    real_t* out = NULL;
    out         = hex_aa_8_collect_coeffs_O3_ptr_cuda(stride0, stride1, stride2, i, j, k, data);

#else
    // collect the data for the WENO interpolation

    const int stride_x = 1;
    const int stride_y = 4;
    const int stride_z = 16;

    real_t out[64];
    hex_aa_8_collect_coeffs_O3_cuda(stride0, stride1, stride2, i, j, k, data, out);

#endif
    ////// Compute the local indices
    // ptrdiff_t i_local, j_local, k_local;

    const real_t r1 = 1.0;

    const ptrdiff_t i_local = floor_real_t(x_unit - ox_unit);
    const ptrdiff_t j_local = floor_real_t(y_unit - oy_unit);
    const ptrdiff_t k_local = floor_real_t(z_unit - oz_unit);

    const real_t x = (x_unit - ox_unit) - (real_t)i_local + r1;
    const real_t y = (y_unit - oy_unit) - (real_t)j_local + r1;
    const real_t z = (z_unit - oz_unit) - (real_t)k_local + r1;

    // printf("x = %f, x_ = %f, i = %d\n", x, x_, i);
    // printf("y = %f, y_ = %f, j = %d\n", y, y_, j);
    // printf("z = %f, z_ = %f, k = %d\n", z, z_, k);

    // printf("delta = %f\n", h);

    const real_t w4 = weno4_3D_HOne_cuda(x,  //
                                         y,
                                         z,
                                         out,
                                         stride_x,
                                         stride_y,
                                         stride_z);

    return w4;
}

///////////////////////////////////////////////////////////////////////
// hex8_to_isoparametric_tet10_resample_field_local_cube1_cuda
///////////////////////////////////////////////////////////////////////
__global__ void                                                                                  //
hex8_to_isoparametric_tet10_resample_field_local_cube1_kernel(const ptrdiff_t    start_element,  // Mesh: start element
                                                              const ptrdiff_t    end_element,    // Mesh: end element
                                                              const ptrdiff_t    nnodes,         // Mesh: number of nodes
                                                              elems_tet10_device elems,          // Mesh: connectivity
                                                              xyz_tet10_device   xyz,            // Mesh: coordinates
                                                              const ptrdiff_t    nx,  // SDF: number of nodes in each direction x
                                                              const ptrdiff_t    ny,  // SDF: number of nodes in each direction y
                                                              const ptrdiff_t    nz,  // SDF: number of nodes in each direction z
                                                              const ptrdiff_t    stride0,  // SDF: stride of the data
                                                              const ptrdiff_t    stride1,  // SDF: stride of the data
                                                              const ptrdiff_t    stride2,  // SDF: stride of the data
                                                              const geom_t       originx,  // SDF: origin x of the domain
                                                              const geom_t       originy,  // SDF: origin y of the domain
                                                              const geom_t       originz,  // SDF: origin z of the domain
                                                              const geom_t       deltax,   // SDF: delta x of the domain
                                                              const geom_t       deltay,   // SDF: delta y of the domain
                                                              const geom_t       deltaz,   // SDF: delta z of the domain
                                                              const real_t* const MY_RESTRICT data,        // SDF
                                                              real_t* const MY_RESTRICT       weighted_field) {  // Output field

#define WENO_CUBE 1

    //
    // printf("============================================================\n");
    // printf("Start: hex8_to_isoparametric_tet10_resample_field_local_reduce_kernel\n");
    // printf("============================================================\n");

    const real_t ox = (real_t)originx;
    const real_t oy = (real_t)originy;
    const real_t oz = (real_t)originz;

    const real_t dx = (real_t)deltax;
    const real_t dy = (real_t)deltay;
    const real_t dz = (real_t)deltaz;

    ////////////////////////////////////////
    // Kernel specific variables

    namespace cg               = cooperative_groups;
    cg::thread_block g         = cg::this_thread_block();
    auto             tile      = cg::tiled_partition<__TET10_TILE_SIZE__>(g);
    const unsigned   tile_rank = tile.thread_rank();

    const ptrdiff_t element_i = (blockIdx.x * blockDim.x + threadIdx.x) / __TET10_TILE_SIZE__;
    if (element_i < start_element or element_i >= end_element) return;

    ////////////////////////////////////////
    // Quadrature points
    ptrdiff_t ev[10];

    // ISOPARAMETRIC
    real_t x[10], y[10], z[10];
    real_t x_unit[10], y_unit[10], z_unit[10];

#if WENO_CUBE == 0
    real_t hex8_f[8];
    real_t coeffs[8];
#endif

    real_t tet10_f[10];

    real_t element_field_v0_reduce = 0.0;
    real_t element_field_v1_reduce = 0.0;
    real_t element_field_v2_reduce = 0.0;
    real_t element_field_v3_reduce = 0.0;
    real_t element_field_v4_reduce = 0.0;
    real_t element_field_v5_reduce = 0.0;
    real_t element_field_v6_reduce = 0.0;
    real_t element_field_v7_reduce = 0.0;
    real_t element_field_v8_reduce = 0.0;
    real_t element_field_v9_reduce = 0.0;

    const real_t cVolume = dx * dy * dz;

    // loop over the ndes of the element

    ev[0] = __ldg(&elems.elems_v0[element_i]);
    ev[1] = __ldg(&elems.elems_v1[element_i]);
    ev[2] = __ldg(&elems.elems_v2[element_i]);
    ev[3] = __ldg(&elems.elems_v3[element_i]);
    ev[4] = __ldg(&elems.elems_v4[element_i]);
    ev[5] = __ldg(&elems.elems_v5[element_i]);
    ev[6] = __ldg(&elems.elems_v6[element_i]);
    ev[7] = __ldg(&elems.elems_v7[element_i]);
    ev[8] = __ldg(&elems.elems_v8[element_i]);
    ev[9] = __ldg(&elems.elems_v9[element_i]);

    // // ISOPARAMETRIC
    // for (int v = 0; v < 10; ++v) {
    //     x[v] = xyz.x[ev[v]];  // x-coordinates
    //     y[v] = xyz.y[ev[v]];  // y-coordinates
    //     z[v] = xyz.z[ev[v]];  // z-coordinates
    // }

    // ISOPARAMETRIC
    // and search the node closest to the origin
    int    v_orig   = 0;
    real_t dist_min = 1e14;

#pragma unroll
    for (int v = 0; v < 10; ++v) {
        x[v] = __ldg(&xyz.x[ev[v]]);  // x-coordinates
        y[v] = __ldg(&xyz.y[ev[v]]);  // y-coordinates
        z[v] = __ldg(&xyz.z[ev[v]]);  // z-coordinates

        const real_t dist = sqrt_real_t((x[v] - ox) * (x[v] - ox) +  //
                                        (y[v] - oy) * (y[v] - oy) +  //
                                        (z[v] - oz) * (z[v] - oz));  //

        if (dist < dist_min) {
            dist_min = dist;
            v_orig   = v;
        }
    }

    const real_t grid_x_orig = (x[v_orig] - ox) / dx;
    const real_t grid_y_orig = (y[v_orig] - oy) / dy;
    const real_t grid_z_orig = (z[v_orig] - oz) / dz;

    const ptrdiff_t i_orig = floor_real_t(grid_x_orig);
    const ptrdiff_t j_orig = floor_real_t(grid_y_orig);
    const ptrdiff_t k_orig = floor_real_t(grid_z_orig);

    const real_t x_orig = ox + ((real_t)i_orig) * dx;
    const real_t y_orig = oy + ((real_t)j_orig) * dy;
    const real_t z_orig = oz + ((real_t)k_orig) * dz;

    // Map element to the grid based on unitary spacing
    for (int v = 0; v < 10; ++v) {
        x_unit[v] = (x[v] - x_orig) / dx;
        y_unit[v] = (y[v] - y_orig) / dy;
        z_unit[v] = (z[v] - z_orig) / dz;
    }

    // SUBPARAMETRIC (for iso-parametric tassellation of tet10 might be necessary)

    const size_t nr_warp_loop = (TET_QUAD_NQP / __TET10_TILE_SIZE__) +                //
                                ((TET_QUAD_NQP % __TET10_TILE_SIZE__) == 0 ? 0 : 1);  //

    for (size_t warp_i = 0; warp_i < nr_warp_loop; warp_i++) {
        const size_t q_i = warp_i * size_t(__TET10_TILE_SIZE__) + tile_rank;

        const real_t tet4_qx_v = (q_i < TET_QUAD_NQP) ? tet4_qx[q_i] : tet4_qx[0];
        const real_t tet4_qy_v = (q_i < TET_QUAD_NQP) ? tet4_qy[q_i] : tet4_qy[0];
        const real_t tet4_qz_v = (q_i < TET_QUAD_NQP) ? tet4_qz[q_i] : tet4_qz[0];
        const real_t tet4_qw_v = (q_i < TET_QUAD_NQP) ? tet4_qw[q_i] : real_t(0.0);

        const real_t measure = tet10_measure_cu(x_unit,
                                                y_unit,
                                                z_unit,  //
                                                tet4_qx_v,
                                                tet4_qy_v,
                                                tet4_qz_v);

        const real_t dV = measure * tet4_qw_v * cVolume;

        // printf("dV[%d]: %e\n", q, dV);

        // Transform quadrature point to physical space
        // g_qx_glob, g_qy_glob, g_qz_glob are the coordinates of the quadrature point in
        // the global space
        real_t g_qx_glob, g_qy_glob, g_qz_glob;
        tet10_transform_cu(x,  //
                           y,
                           z,
                           tet4_qx_v,
                           tet4_qy_v,
                           tet4_qz_v,
                           &g_qx_glob,
                           &g_qy_glob,
                           &g_qz_glob);

        tet10_dual_basis_hrt_cu(tet4_qx_v, tet4_qy_v, tet4_qz_v, tet10_f);

        // Transform quadrature point to unitary space
        // g_qx_unit, g_qy_unit, g_qz_unit are the coordinates of the quadrature point in
        // the unitary space
        real_t g_qx_unit, g_qy_unit, g_qz_unit;
        tet10_transform_cu(x_unit,  //
                           y_unit,
                           z_unit,
                           tet4_qx_v,
                           tet4_qy_v,
                           tet4_qz_v,
                           &g_qx_unit,
                           &g_qy_unit,
                           &g_qz_unit);

        ///// ======================================================

        // Get the global grid coordinates
        const real_t grid_x = (g_qx_glob - ox) / dx;
        const real_t grid_y = (g_qy_glob - oy) / dy;
        const real_t grid_z = (g_qz_glob - oz) / dz;

        const ptrdiff_t i_glob = floor_real_t(grid_x);
        const ptrdiff_t j_glob = floor_real_t(grid_y);
        const ptrdiff_t k_glob = floor_real_t(grid_z);

        // /* If outside */
        // if (i_glob < 0 || j_glob < 0 || k_glob < 0 || (i_glob + 1 >= n[0]) ||
        //     (j_glob + 1 >= n[1]) || (k_glob + 1 >= n[2])) {
        //     fprintf(stderr,
        //             "ERROR: (%g, %g, %g) (%ld, %ld, %ld) outside domain  (%ld, %ld, "
        //             "%ld)!\n",
        //             g_qx_glob,
        //             g_qy_glob,
        //             g_qz_glob,
        //             i_glob,
        //             j_glob,
        //             k_glob,
        //             n[0],
        //             n[1],
        //             n[2]);
        //     exit(1);
        // }

#if WENO_CUBE == 1

        // if (nSizes_global != nNodesData) {
        //     fprintf(stderr, "nSizes_global != nNodes .. %ld != %ld\n", nSizes_global,
        //     nNodesData);
        // }

#if WENO_DIRECT_CUDA == 1
        // Calculate the origin of the 4x4x4 cube in the global space
        // And transform the coordinates to the the unitary space
        const real_t x_cube_origin = (ox + ((real_t)i_glob - (real_t)(1.0)) * dx) / dx;
        const real_t y_cube_origin = (oy + ((real_t)j_glob - (real_t)(1.0)) * dy) / dy;
        const real_t z_cube_origin = (oz + ((real_t)k_glob - (real_t)(1.0)) * dz) / dz;
#else
        const real_t x_cube_origin = 0.0;
        const real_t y_cube_origin = 0.0;
        const real_t z_cube_origin = 0.0;
#endif

        //
        // printf("nSizes_global = %ld\n", nSizes_global);
        // printf("origin = (%f, %f, %f)\n", ox, oy, oz);
        real_t eval_field = hex_aa_8_eval_weno4_3D_Unit_cuda(g_qx_unit,      //
                                                             g_qy_unit,      //
                                                             g_qz_unit,      //
                                                             x_cube_origin,  //
                                                             y_cube_origin,  //
                                                             z_cube_origin,  //
                                                             i_glob,         //
                                                             j_glob,         //
                                                             k_glob,         //
                                                             stride0,        //
                                                             stride1,        //
                                                             stride2,        //
                                                             data);          //

        // eval_field = 1.0;  ///////////////////// DEBUG
#else  // WENO_CUBE == 0

        // Get the reminder [0, 1]
        real_t l_x = (grid_x - i_glob);
        real_t l_y = (grid_y - j_glob);
        real_t l_z = (grid_z - k_glob);

        assert(l_x >= -1e-8);
        assert(l_y >= -1e-8);
        assert(l_z >= -1e-8);

        assert(l_x <= 1 + 1e-8);
        assert(l_y <= 1 + 1e-8);
        assert(l_z <= 1 + 1e-8);

        hex_aa_8_eval_fun_cu(l_x, l_y, l_z, hex8_f);
        hex_aa_8_collect_coeffs_cu(stride0, stride1, stride2, i_glob, j_glob, k_glob, data, coeffs);

        real_t eval_field = 0;
        // UNROLL_ZERO?
        for (int edof_j = 0; edof_j < 8; edof_j++) {
            eval_field += hex8_f[edof_j] * coeffs[edof_j];
        }
#endif

        element_field_v0_reduce += eval_field * tet10_f[0] * dV;
        element_field_v1_reduce += eval_field * tet10_f[1] * dV;
        element_field_v2_reduce += eval_field * tet10_f[2] * dV;
        element_field_v3_reduce += eval_field * tet10_f[3] * dV;
        element_field_v4_reduce += eval_field * tet10_f[4] * dV;
        element_field_v5_reduce += eval_field * tet10_f[5] * dV;
        element_field_v6_reduce += eval_field * tet10_f[6] * dV;
        element_field_v7_reduce += eval_field * tet10_f[7] * dV;
        element_field_v8_reduce += eval_field * tet10_f[8] * dV;
        element_field_v9_reduce += eval_field * tet10_f[9] * dV;
    }  // end quadrature loop

    for (int i = tile.size() / 2; i > 0; i /= 2) {
        element_field_v0_reduce += tile.shfl_down(element_field_v0_reduce, i);
        element_field_v1_reduce += tile.shfl_down(element_field_v1_reduce, i);
        element_field_v2_reduce += tile.shfl_down(element_field_v2_reduce, i);
        element_field_v3_reduce += tile.shfl_down(element_field_v3_reduce, i);
        element_field_v4_reduce += tile.shfl_down(element_field_v4_reduce, i);
        element_field_v5_reduce += tile.shfl_down(element_field_v5_reduce, i);
        element_field_v6_reduce += tile.shfl_down(element_field_v6_reduce, i);
        element_field_v7_reduce += tile.shfl_down(element_field_v7_reduce, i);
        element_field_v8_reduce += tile.shfl_down(element_field_v8_reduce, i);
        element_field_v9_reduce += tile.shfl_down(element_field_v9_reduce, i);
    }

    if (tile_rank == 0) {
        atomicAdd(&weighted_field[ev[0]], element_field_v0_reduce);
        atomicAdd(&weighted_field[ev[1]], element_field_v1_reduce);
        atomicAdd(&weighted_field[ev[2]], element_field_v2_reduce);
        atomicAdd(&weighted_field[ev[3]], element_field_v3_reduce);
        atomicAdd(&weighted_field[ev[4]], element_field_v4_reduce);
        atomicAdd(&weighted_field[ev[5]], element_field_v5_reduce);
        atomicAdd(&weighted_field[ev[6]], element_field_v6_reduce);
        atomicAdd(&weighted_field[ev[7]], element_field_v7_reduce);
        atomicAdd(&weighted_field[ev[8]], element_field_v8_reduce);
        atomicAdd(&weighted_field[ev[9]], element_field_v9_reduce);
    }
}

#endif  // __TET10_RESAMPLE_FIELD_KERNELS_CUH__